"""REMWorkflow — Relative Entropy Minimization.

Single-sampler iteration loop: each epoch runs one free CG simulation,
computes CG energy-gradient statistics via MPI engine, and updates the
forcefield via ``REMTrainerAnalytic``.
"""

from __future__ import annotations

import pickle
import shutil
import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..configs.models import ACGConfig
from ..io.logger import get_screen_logger
from ..schedulers.task_scheduler import TaskSpec
from ..trainers.analytic.rem import REMTrainerAnalytic
from .base import _run_workflow_cli
from .sampling import AAStats, SamplingWorkflow

logger = get_screen_logger("rem")


class REMWorkflow(SamplingWorkflow):
    """Relative Entropy Minimization workflow.

    Adds on top of ``SamplingWorkflow``:
        trainer – ``REMTrainerAnalytic``
    """

    def __init__(self, config: ACGConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.optimizer = self._build_optimizer(self.forcefield)
        self.trainer = self._build_trainer()
        self.aa_data_strategy = self._build_aa_data_strategy()

    # ── builders ────────────────────────────────────────────────

    def _build_trainer(self) -> REMTrainerAnalytic:
        return REMTrainerAnalytic(
            forcefield=self.forcefield,
            optimizer=self.optimizer,
            beta=self.beta,
        )

    # ── run ─────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute the REM training loop."""
        cfg = self.config
        n_epochs = cfg.training.n_epochs
        start_epoch = cfg.training.start_epoch
        results = []
        need_hessian = bool(
            self.trainer.optimizer_accepts_hessian() or cfg.training.need_hessian
        )
        aa_data_strategy = self.aa_data_strategy
        if aa_data_strategy is None:
            raise RuntimeError("AA data strategy was not constructed.")
        constant_aa_stats = aa_data_strategy if isinstance(aa_data_strategy, AAStats) else None

        if start_epoch > 0:
            prev_ff_dir = (
                self.output_dir / f"iter_{start_epoch - 1:04d}" / "ff"
            )
            prev_snapshot = prev_ff_dir / "workflow_checkpoint.pkl"
            if not prev_snapshot.exists():
                raise FileNotFoundError(
                    f"Cannot resume from epoch {start_epoch}: "
                    f"checkpoint {prev_snapshot} not found."
                )
            self._load_workflow_checkpoint(prev_ff_dir)
            self.trainer = self._build_trainer()
            logger.info(
                "Resuming from epoch %d (loaded %s)", start_epoch, prev_snapshot,
            )

        for epoch in range(start_epoch, n_epochs):
            iter_dir = self.output_dir / f"iter_{epoch:04d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            aa_stats = (
                constant_aa_stats
                if constant_aa_stats is not None
                else aa_data_strategy(self.forcefield)
            )

            # ── Phase 1: write FF ────────────────────────────────
            ff_dir = iter_dir / "ff"
            ff_dir.mkdir(exist_ok=True)
            self._write_forcefield(ff_dir)
            ff_snapshot_path = self._snapshot_forcefield(ff_dir)

            # ── Phase 2: sampler staging ─────────────────────────
            epoch_dir = iter_dir / "epoch"
            state = self.sampler.init_epoch(
                iteration_index=epoch,
                epoch_dir=epoch_dir,
                n_runs=1,
            )
            plan = state.replica_plans[0]
            for src in ff_dir.rglob("*"):
                if not src.is_file() or src.suffix == ".pkl":
                    continue
                dst = plan.run_dir / src.relative_to(ff_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

            # ── Phase 3: build task spec ─────────────────────────
            xz_cpu, xz_min, xz_pref, xz_max = self._elastic_core_bounds(
                self.config.sampling.ncores,
            )
            trajectory_relpath = str(plan.trajectory_path.relative_to(plan.run_dir))
            post_spec: Dict[str, Any] = {
                "work_dir": str(plan.run_dir),
                "forcefield_path": str(ff_snapshot_path),
                "topology": str(self._resolve_config_path(cfg.system.topology_file)),
                "trajectory": str(plan.trajectory_path),
                "exclude_bonded": cfg.system.exclude_bonded,
                "exclude_option": cfg.system.exclude_option,
                "cutoff": cfg.system.cutoff,
                "steps": [
                    {
                        "step_mode": "rem",
                        "need_hessian": need_hessian,
                        "output_file": "result.pkl",
                    }
                ],
            }
            if cfg.system.type_names is not None:
                post_spec["atom_type_name_aliases"] = cfg.system.type_names
            if cfg.vp is not None:
                post_spec["vp_names"] = list(cfg.vp.vp_names)
            xz_task = TaskSpec(
                task_class="xz",
                frame_id=None,
                run_dir=str(plan.run_dir.resolve()),
                cpu_cores=xz_cpu,
                min_cores=xz_min,
                preferred_cores=xz_pref,
                max_cores=xz_max,
                sim_input=plan.input_script_path.name,
                sim_backend=cfg.sampling.sim_backend,
                sim_var=dict(cfg.sampling.sim_var),
                post_spec=post_spec,
                post_exec={"mode": "mpi"},
                archive_trajectory=False,
                trajectory_files=[trajectory_relpath],
                single_host_only=False,
            )

            # ── Phase 4: scheduler execution ─────────────────────
            iter_result = self.scheduler.run_iteration(
                xz_tasks=[xz_task],
                zbx_tasks=[],
                iter_dir=iter_dir,
            )
            if not iter_result.xz_ok:
                logger.error("REM xz sampling failed at epoch %d", epoch)
                break

            # ── Phase 5: read post result + train ────────────────
            output_path = plan.run_dir / "result.pkl"
            if not output_path.exists():
                raise RuntimeError(f"CG post-processing produced no output at {output_path}")
            with open(output_path, "rb") as fh:
                cg_stats = pickle.load(fh)

            batch = REMTrainerAnalytic.make_batch(
                energy_grad_CG=np.asarray(cg_stats["energy_grad_avg"], dtype=np.float64),
                energy_grad_AA=aa_stats.energy_grad,
                d2U_AA=aa_stats.d2U,
                d2U_CG=cg_stats.get("d2U_avg"),
                grad_outer_CG=cg_stats.get("grad_outer_avg"),
                step_index=epoch,
            )
            step_out = self.trainer.step(batch, apply_update=True)
            self.forcefield.update_params(self.trainer.get_params())
            results.append(step_out)
            logger.info("Epoch %d: grad_norm=%.6g", epoch, float(np.linalg.norm(step_out["grad"])))

            # ── Phase 6: sampler cleanup ─────────────────────────
            self.sampler.clean_epoch(state)
            self._snapshot_optimizer(ff_dir)
            self._write_workflow_checkpoint(ff_dir)

        return {"epochs": len(results), "results": results}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``acg-rem`` entry point."""
    return _run_workflow_cli(
        REMWorkflow,
        prog="acg-rem",
        description="Run the AceCG REM workflow.",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
