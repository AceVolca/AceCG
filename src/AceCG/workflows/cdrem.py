"""CDREMWorkflow — Conditioned Relative Entropy Minimization.

Dual-sampling iteration loop: each epoch runs one free CG simulation (xz),
K conditioned simulations (zbx), computes energy-gradient statistics via
MPI engine, and updates the forcefield via ``CDREMTrainerAnalytic``.
"""

from __future__ import annotations

import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..configs.models import ACGConfig
from ..configs.utils import extract_frame_id_from_data_file
from ..io.logger import get_screen_logger
from ..samplers.base import EpochState, InitConfigRecord, ReplicaPlan
from ..samplers.conditioned import ConditionedSampler
from ..schedulers.task_scheduler import TaskSpec
from ..trainers.analytic.cdrem import CDREMTrainerAnalytic
from .base import _run_workflow_cli
from .sampling import SamplingWorkflow

logger = get_screen_logger("cdrem")


class CDREMWorkflow(SamplingWorkflow):
    """CD-REM: dual-sampling (xz free + zbx conditioned) iteration loop.

    Per-epoch flow:
        1. Write FF to ff_dir
        2. Stage xz replica via ``self.sampler`` (BaseSampler)
        3. Stage zbx replicas via ``self.zbx_sampler`` (ConditionedSampler)
        4. Build TaskSpec lists for xz (1 task) and zbx (K tasks)
        5. ``scheduler.run_iteration(xz_tasks, zbx_tasks)``
        6. Read ``result.pkl`` from xz and each zbx
        7. ``CDREMTrainerAnalytic.step(batch)``
        8. ``sampler.clean_epoch()``
    """

    def __init__(self, config: ACGConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.optimizer = self._build_optimizer(self.forcefield)
        self.trainer = self._build_trainer()
        self.zbx_sampler = self._build_zbx_sampler()

    # ── builders ────────────────────────────────────────────────

    def _build_trainer(self) -> CDREMTrainerAnalytic:
        return CDREMTrainerAnalytic(
            forcefield=self.forcefield,
            optimizer=self.optimizer,
            beta=self.beta,
        )

    def _build_zbx_sampler(self) -> ConditionedSampler:
        cond = self.config.conditioning
        pool = [
            InitConfigRecord(path=p, frame_id=extract_frame_id_from_data_file(p))
            for p in self._glob_config_paths(cond.init_config_pool)
        ]
        if not pool:
            raise ValueError(
                "conditioning.init_config_pool glob matched no files: "
                f"{cond.init_config_pool!r}"
            )
        sim_input = self._resolve_config_path(cond.input)
        if sim_input is None:
            raise ValueError("conditioning.input is required for CDREM.")
        return ConditionedSampler(
            sim_input=sim_input,
            sim_backend=self.config.sampling.sim_backend,
            init_config_pool=pool,
            rng=self.workflow_rng,
        )

    # ── FF copy helper ──────────────────────────────────────────

    def _copy_ff_to_run(self, ff_dir: Path, run_dir: Path) -> None:
        """Copy forcefield bundle (table files + settings) to a replica dir.

        Skips ``.pkl`` snapshots — those are consumed by the MPI engine via
        the ``forcefield_path`` key in post_spec, not by LAMMPS.
        """
        for src in ff_dir.rglob("*"):
            if src.is_file() and src.suffix != ".pkl":
                dst = run_dir / src.relative_to(ff_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    # ── task-spec builders ──────────────────────────────────────

    def _build_xz_task(
        self,
        plan: ReplicaPlan,
        ff_snapshot_path: Path,
    ) -> TaskSpec:
        """Build a ``TaskSpec`` for the single free-CG xz replica."""
        cfg = self.config
        xz_cpu, xz_min, xz_pref, xz_max = self._elastic_core_bounds(
            cfg.sampling.ncores,
        )
        need_hessian = bool(
            self.trainer.optimizer_accepts_hessian() or cfg.training.need_hessian
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
                    # canonicalizes to "rem" in reducer
                    "step_mode": "cdrem",
                    "need_hessian": need_hessian,
                    "output_file": "result.pkl",
                }
            ],
        }
        self._apply_rem_statistics_options(post_spec["steps"][0])
        if cfg.system.type_names is not None:
            post_spec["atom_type_name_aliases"] = cfg.system.type_names
        if cfg.vp is not None:
            post_spec["vp_names"] = list(cfg.vp.vp_names)

        return TaskSpec(
            task_class="xz",
            # xz: no conditioning frame
            frame_id=None,
            run_dir=str(plan.run_dir.resolve()),
            cpu_cores=xz_cpu,
            min_cores=xz_min,
            preferred_cores=xz_pref,
            max_cores=xz_max,
            sim_input=plan.input_script_path.name,
            sim_backend=cfg.sampling.sim_backend,
            # default LAMMPS log name
            sim_log="sim.log",
            post_spec=post_spec,
            post_exec={"mode": "mpi"},
            sim_var=dict(cfg.sampling.sim_var),
            # trajectories cleaned by sampler
            archive_trajectory=False,
            trajectory_files=[trajectory_relpath],
            single_host_only=False,
        )

    def _build_zbx_task(
        self,
        plan: ReplicaPlan,
        ff_snapshot_path: Path,
    ) -> TaskSpec:
        """Build a ``TaskSpec`` for one conditioned zbx replica."""
        cfg = self.config
        zbx_cpu, zbx_min, zbx_pref, zbx_max = self._elastic_core_bounds(
            cfg.conditioning.ncores_per_task,
        )
        need_hessian = bool(
            self.trainer.optimizer_accepts_hessian() or cfg.training.need_hessian
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
                    # canonicalizes to "rem" in reducer
                    "step_mode": "cdrem",
                    "need_hessian": need_hessian,
                    "output_file": "result.pkl",
                }
            ],
        }
        self._apply_rem_statistics_options(post_spec["steps"][0])
        if cfg.system.type_names is not None:
            post_spec["atom_type_name_aliases"] = cfg.system.type_names
        if cfg.vp is not None:
            post_spec["vp_names"] = list(cfg.vp.vp_names)

        return TaskSpec(
            task_class="zbx",
            # conditioning frame index
            frame_id=plan.frame_id,
            run_dir=str(plan.run_dir.resolve()),
            cpu_cores=zbx_cpu,
            min_cores=zbx_min,
            preferred_cores=zbx_pref,
            max_cores=zbx_max,
            sim_input=plan.input_script_path.name,
            sim_backend=cfg.sampling.sim_backend,
            # default LAMMPS log name
            sim_log="sim.log",
            post_spec=post_spec,
            post_exec={"mode": "mpi"},
            sim_var=dict(cfg.sampling.sim_var),
            # trajectories cleaned after read
            archive_trajectory=False,
            trajectory_files=[trajectory_relpath],
            single_host_only=False,
        )

    # ── batch collection ────────────────────────────────────────

    def _collect_cdrem_batch(
        self,
        xz_plan: ReplicaPlan,
        zbx_state: EpochState,
        epoch: int,
    ) -> Dict[str, Any]:
        """Read ``result.pkl`` from xz and each zbx, assemble ``CDREMBatch``."""
        need_hessian = bool(
            self.trainer.optimizer_accepts_hessian()
            or self.config.training.need_hessian
        )

        # ── xz result: energy_grad_avg shape (n_params,) ────────
        xz_result_path = xz_plan.run_dir / "result.pkl"
        if not xz_result_path.exists():
            raise RuntimeError(
                f"xz post-processing produced no output at {xz_result_path}"
            )
        with open(xz_result_path, "rb") as fh:
            xz_stats = pickle.load(fh)
        energy_grad_xz = np.asarray(xz_stats["energy_grad_avg"], dtype=np.float64)
        unmasked_energy_grad_xz = xz_stats.get("unmasked_energy_grad_avg")

        # ── zbx results: one energy_grad_avg per frame ──────────
        energy_grad_z_by_x = []
        unmasked_energy_grad_z_by_x = []
        d2U_z_by_x_list = [] if need_hessian else None
        cov_z_by_x_list = [] if need_hessian else None

        for plan in zbx_state.replica_plans:
            zbx_path = plan.run_dir / "result.pkl"
            if not zbx_path.exists():
                logger.warning(
                    "zbx result missing for frame %s: %s",
                    plan.frame_id, zbx_path,
                )
                continue
            with open(zbx_path, "rb") as fh:
                zbx_stats = pickle.load(fh)

            energy_grad_z_by_x.append(
                np.asarray(zbx_stats["energy_grad_avg"], dtype=np.float64)
            )
            if "unmasked_energy_grad_avg" in zbx_stats:
                unmasked_energy_grad_z_by_x.append(
                    np.asarray(zbx_stats["unmasked_energy_grad_avg"], dtype=np.float64)
                )
            if need_hessian:
                d2U_z_by_x_list.append(
                    np.asarray(zbx_stats["d2U_avg"], dtype=np.float64)
                )
                # conditional covariance: Cov_{z|x}[dU/dλ] =
                #   E[outer] - E[grad] ⊗ E[grad]
                grad_outer = np.asarray(
                    zbx_stats["grad_outer_avg"], dtype=np.float64
                )
                eg = np.asarray(
                    zbx_stats["energy_grad_avg"], dtype=np.float64
                )
                cov_z_by_x_list.append(grad_outer - np.outer(eg, eg))

        if not energy_grad_z_by_x:
            raise RuntimeError(
                "No zbx results collected — cannot build CDREM batch."
            )

        # ── assemble batch ──────────────────────────────────────
        batch_kwargs: Dict[str, Any] = {
            "energy_grad_z_by_x": np.array(energy_grad_z_by_x),
            "energy_grad_xz": energy_grad_xz,
            # x_weight — omitted; defaults to uniform averaging over x
            "optimizer_gradient_mode": self._optimizer_gradient_mode(),
            "outside_aux_weight": self._outside_aux_weight(),
            "allow_unmasked_optimizer_gradient": self._allow_unmasked_optimizer_gradient(),
            "step_index": epoch,
        }
        if unmasked_energy_grad_xz is not None:
            if len(unmasked_energy_grad_z_by_x) != len(energy_grad_z_by_x):
                raise RuntimeError(
                    "CDREM collected only a partial set of unmasked zbx gradients."
                )
            batch_kwargs["unmasked_energy_grad_xz"] = np.asarray(
                unmasked_energy_grad_xz,
                dtype=np.float64,
            )
            batch_kwargs["unmasked_energy_grad_z_by_x"] = np.array(
                unmasked_energy_grad_z_by_x,
            )

        if need_hessian:
            batch_kwargs["d2U_z_by_x"] = np.array(d2U_z_by_x_list)
            batch_kwargs["d2U_xz"] = np.asarray(
                xz_stats["d2U_avg"], dtype=np.float64
            )
            batch_kwargs["energy_grad_outer_xz"] = np.asarray(
                xz_stats["grad_outer_avg"], dtype=np.float64
            )
            batch_kwargs["cov_z_by_x"] = np.array(cov_z_by_x_list)

        return CDREMTrainerAnalytic.make_batch(**batch_kwargs)

    # ── run ─────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute the CDREM training loop."""
        cfg = self.config
        n_epochs = cfg.training.n_epochs
        start_epoch = cfg.training.start_epoch
        K = cfg.conditioning.n_samples
        results = []

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

            # ── Phase 1: write FF ────────────────────────────────
            ff_dir = iter_dir / "ff"
            ff_dir.mkdir(exist_ok=True)
            self._write_forcefield(ff_dir)
            ff_snapshot_path = self._snapshot_forcefield(ff_dir)

            # ── Phase 2: stage xz replica (1 free CG+VP run) ────
            xz_dir = iter_dir / "xz"
            xz_state = self.sampler.init_epoch(
                iteration_index=epoch,
                epoch_dir=xz_dir,
                n_runs=1,
            )
            xz_plan = xz_state.replica_plans[0]
            self._copy_ff_to_run(ff_dir, xz_plan.run_dir)

            # ── Phase 3: stage zbx replicas (K conditioned) ─────
            zbx_dir = iter_dir / "zbx"
            zbx_state = self.zbx_sampler.init_epoch(
                iteration_index=epoch,
                epoch_dir=zbx_dir,
                n_runs=K,
            )
            for plan in zbx_state.replica_plans:
                self._copy_ff_to_run(ff_dir, plan.run_dir)

            # ── Phase 4: build task specs ────────────────────────
            xz_task = self._build_xz_task(xz_plan, ff_snapshot_path)
            zbx_tasks = [
                self._build_zbx_task(plan, ff_snapshot_path)
                for plan in zbx_state.replica_plans
            ]

            # ── Phase 5: scheduler execution ─────────────────────
            iter_result = self.scheduler.run_iteration(
                xz_tasks=[xz_task],
                zbx_tasks=zbx_tasks,
                iter_dir=iter_dir,
            )
            if not iter_result.xz_ok:
                logger.error("CDREM xz sampling failed at epoch %d", epoch)
                break

            # ── Phase 6: read results + train ────────────────────
            batch = self._collect_cdrem_batch(xz_plan, zbx_state, epoch)
            step_out = self.trainer.step(batch, apply_update=True)
            self.forcefield.update_params(self.trainer.get_params())
            results.append(step_out)
            logger.info(
                "Epoch %d: grad_norm=%.6g  n_zbx=%d",
                epoch,
                float(np.linalg.norm(step_out["grad"])),
                iter_result.succeeded_zbx,
            )

            # ── Phase 7: sampler cleanup ─────────────────────────
            self.sampler.clean_epoch(xz_state)
            self._snapshot_optimizer(ff_dir)
            self._write_workflow_checkpoint(ff_dir)

        return {"epochs": len(results), "results": results}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``acg-cdrem`` entry point."""
    return _run_workflow_cli(
        CDREMWorkflow,
        prog="acg-cdrem",
        description="Run the AceCG CDREM workflow.",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
