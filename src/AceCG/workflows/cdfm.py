"""CDFMWorkflow — Conditioned Force Matching.

zbx-only conditioned sampling plus force matching iteration loop. Each zbx
replica is seeded with one ``(init config, reference force)`` pair drawn
from the two conditioning pools and paired by frame id. rank 0 of the
replica's MPI post computes the single-frame ``y_eff = y_ref -
f_theta_cg_only(R_init)`` once and broadcasts it to every rank; the
one-pass frame loop then accumulates CDFM gradient statistics, which are
reduced across replicas by ``CDFMTrainerAnalytic``.
"""

from __future__ import annotations

import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..configs.models import ACGConfig
from ..configs.utils import (
    extract_frame_id_from_data_file,
    extract_frame_id_from_force_file,
)
from ..io.logger import get_screen_logger
from ..samplers.base import EpochState, InitConfigRecord, ReplicaPlan
from ..samplers.conditioned import ConditionedSampler
from ..schedulers.task_scheduler import TaskSpec
from ..trainers.analytic.cdfm import CDFMTrainerAnalytic
from .base import _run_workflow_cli
from .sampling import SamplingWorkflow

logger = get_screen_logger("cdfm")


class CDFMWorkflow(SamplingWorkflow):
    """CD-FM: zbx-only conditioned sampling + force matching.

    Per-epoch flow:
        1. Write FF to ff_dir
        2. Stage zbx replicas via ``self.zbx_sampler`` (ConditionedSampler)
        3. Build one TaskSpec per replica (K tasks) — no xz
        4. ``scheduler.run_iteration(xz_tasks=[], zbx_tasks=zbx_tasks)``
        5. Read ``result.pkl`` from each zbx
        6. ``CDFMTrainerAnalytic.step(batch)``
    """

    def __init__(self, config: ACGConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._install_cdfm_mask()
        self.optimizer = self._build_optimizer(self.forcefield)
        self.trainer = self._build_trainer()
        self.zbx_sampler = self._build_zbx_sampler()

    # ── overrides ───────────────────────────────────────────────

    def _build_sampler(self):  # type: ignore[override]
        # CDFM has no xz sampling; all replicas are zbx-conditioned
        # via self.zbx_sampler.  Return None to skip BaseSampler
        # construction (no free-CG input script needed).
        return None

    # ── mask install ────────────────────────────────────────────

    def _install_cdfm_mask(self) -> None:
        """Compose the CDFM training mask on top of the user mask.

        With ``conditioning.mask_cg_only=True`` (default) every real-only
        parameter is disabled so the CDFM gradient update acts only on
        virtual-only and mixed (real-virtual) parameters. With
        ``mask_cg_only=False`` the training mask is left unchanged. Either
        way the composition is AND-ed with the current
        ``forcefield.param_mask`` so any ``[system] forcefield_mask`` entries
        set upstream are preserved. CDFM uses the derived key mask for the main
        zbx interaction set, matching the historical CG-only masking behavior
        even though ``param_mask`` and ``key_mask`` are now independent.

        The baseline force computation in ``compute/mpi_engine.py`` explicitly
        swaps both ``param_mask`` and the derived CG-only ``key_mask``
        temporarily, then restores the original masks before the main frame
        loop.
        """
        current = self.forcefield.param_mask
        if self.config.conditioning.mask_cg_only:
            real_mask = self.forcefield.real_mask
            init_mask = np.logical_and(current, ~real_mask)
        else:
            init_mask = np.asarray(current, dtype=bool).copy()
        self.forcefield.build_mask(init_mask=init_mask)
        self.forcefield.key_mask = self.forcefield.derive_l1_mask(init_mask)

    # ── builders ────────────────────────────────────────────────

    def _build_trainer(self) -> CDFMTrainerAnalytic:
        return CDFMTrainerAnalytic(
            forcefield=self.forcefield,
            optimizer=self.optimizer,
            beta=self.beta,
        )

    def _build_zbx_sampler(self) -> ConditionedSampler:
        cond = self.config.conditioning
        cfg_paths = self._glob_config_paths(cond.init_config_pool)
        if not cfg_paths:
            raise ValueError(
                "conditioning.init_config_pool glob matched no files: "
                f"{cond.init_config_pool!r}"
            )
        frc_paths = self._glob_config_paths(cond.init_force_pool)
        if not frc_paths:
            raise ValueError(
                "conditioning.init_force_pool glob matched no files: "
                f"{cond.init_force_pool!r}"
            )

        cfg_by_fid: Dict[int, Path] = {}
        for path in cfg_paths:
            fid = extract_frame_id_from_data_file(path)
            if fid in cfg_by_fid:
                raise ValueError(
                    f"Duplicate frame_id={fid} in init_config_pool: "
                    f"{cfg_by_fid[fid]!s} vs {path!s}"
                )
            cfg_by_fid[fid] = path

        frc_by_fid: Dict[int, Path] = {}
        for path in frc_paths:
            fid = extract_frame_id_from_force_file(path)
            if fid in frc_by_fid:
                raise ValueError(
                    f"Duplicate frame_id={fid} in init_force_pool: "
                    f"{frc_by_fid[fid]!s} vs {path!s}"
                )
            frc_by_fid[fid] = path

        cfg_ids = set(cfg_by_fid)
        frc_ids = set(frc_by_fid)
        missing_in_forces = sorted(cfg_ids - frc_ids)
        missing_in_configs = sorted(frc_ids - cfg_ids)
        if missing_in_forces or missing_in_configs:
            preview_forces = missing_in_forces[:10]
            preview_configs = missing_in_configs[:10]
            raise ValueError(
                "init_config_pool and init_force_pool must pair 1:1 by "
                f"frame_id; missing in init_force_pool "
                f"(len={len(missing_in_forces)}, first {len(preview_forces)}): "
                f"{preview_forces}; missing in init_config_pool "
                f"(len={len(missing_in_configs)}, first {len(preview_configs)}): "
                f"{preview_configs}"
            )

        pool = [
            InitConfigRecord(
                path=cfg_by_fid[fid],
                frame_id=fid,
                force_path=frc_by_fid[fid],
            )
            for fid in sorted(cfg_ids)
        ]
        sim_input = self._resolve_config_path(cond.input)
        if sim_input is None:
            raise ValueError("conditioning.input is required for CDFM.")
        return ConditionedSampler(
            sim_input=sim_input,
            sim_backend=self.config.sampling.sim_backend,
            init_config_pool=pool,
            require_force_path=True,
            rng=self.workflow_rng,
        )

    # ── FF copy helper ──────────────────────────────────────────

    def _copy_ff_to_run(self, ff_dir: Path, run_dir: Path) -> None:
        """Copy forcefield bundle (table files + settings) to a replica dir.

        Skips ``.pkl`` snapshots — those are consumed by the MPI engine
        via the ``forcefield_path`` key in post_spec, not by LAMMPS.
        """
        for src in ff_dir.rglob("*"):
            if src.is_file() and src.suffix != ".pkl":
                dst = run_dir / src.relative_to(ff_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    # ── task-spec builder ───────────────────────────────────────

    def _build_zbx_task(
        self,
        plan: ReplicaPlan,
        ff_snapshot_path: Path,
        *,
        mode: str = "direct",
    ) -> TaskSpec:
        """Build a ``TaskSpec`` for one conditioned zbx replica."""
        cfg = self.config
        zbx_cpu = cfg.conditioning.ncores_per_task or max(
            host.n_cpus for host in self.resource_pool.hosts
        )
        post_n_ranks = cfg.conditioning.post_n_ranks or zbx_cpu
        if post_n_ranks < 1 or post_n_ranks > zbx_cpu:
            raise ValueError(
                "conditioning.post_n_ranks must be between 1 and "
                f"conditioning.ncores_per_task ({zbx_cpu}); got {post_n_ranks}."
            )
        trajectory_relpath = str(plan.trajectory_path.relative_to(plan.run_dir))
        if plan.init_force_path is None:
            raise ValueError(
                f"zbx replica {plan.run_id} (frame_id={plan.frame_id}) has no "
                "init_force_path; the CDFM workflow requires every replica to "
                "carry its paired reference-force .npy."
            )

        post_spec: Dict[str, Any] = {
            "work_dir": str(plan.run_dir),
            "forcefield_path": str(ff_snapshot_path),
            # The replica's "topology" for the MPI post is its own init
            # config; rank 0 uses this single .data file both to compute
            # the baseline y_eff and to provide the topology for the
            # main frame loop over the replica trajectory.
            "topology": str(plan.init_config_path),
            "trajectory": str(plan.trajectory_path),
            "exclude_bonded": cfg.system.exclude_bonded,
            "exclude_option": cfg.system.exclude_option,
            "cutoff": cfg.system.cutoff,
            "steps": [
                {
                    "step_mode": "cdfm_zbx",
                    "mode": mode,
                    "beta": self.beta,
                    "init_force_path": str(plan.init_force_path),
                    "init_frame_id": int(plan.frame_id)
                    if plan.frame_id is not None
                    else 0,
                    "output_file": "result.pkl",
                }
            ],
        }
        if cfg.system.type_names is not None:
            post_spec["atom_type_name_aliases"] = cfg.system.type_names
        if cfg.vp is not None:
            post_spec["vp_names"] = list(cfg.vp.vp_names)

        return TaskSpec(
            task_class="zbx",
            frame_id=plan.frame_id,                # conditioning frame index
            run_dir=str(plan.run_dir.resolve()),
            cpu_cores=zbx_cpu,
            sim_input=plan.input_script_path.name,
            sim_backend=cfg.sampling.sim_backend,
            sim_log="sim.log",                     # default LAMMPS log name
            post_spec=post_spec,
            post_exec={"mode": "mpi", "n_ranks": post_n_ranks},
            sim_var=dict(cfg.sampling.sim_var),
            archive_trajectory=False,              # trajectories cleaned after read
            trajectory_files=[trajectory_relpath],
            single_host_only=False,
        )

    # ── batch collection ────────────────────────────────────────

    def _load_resume_state(self, start_epoch: int) -> None:
        """Load CDFM resume state from the completed workflow checkpoint."""
        if start_epoch <= 0:
            return

        prev_ff_dir = self.output_dir / f"iter_{start_epoch - 1:04d}" / "ff"
        prev_snapshot = prev_ff_dir / "workflow_checkpoint.pkl"
        if prev_snapshot.exists():
            self._load_workflow_checkpoint(prev_ff_dir)
            self.trainer = self._build_trainer()
            logger.info(
                "Resuming from epoch %d (loaded %s)", start_epoch, prev_snapshot,
            )
            return

        raise FileNotFoundError(
            f"Cannot resume CDFM from epoch {start_epoch}: completed workflow "
            f"checkpoint {prev_snapshot} does not exist."
        )

    def _collect_cdfm_batch(
        self,
        zbx_state: EpochState,
        epoch: int,
    ) -> Dict[str, Any]:
        """Read ``result.pkl`` from each zbx, assemble ``CDFMBatch``.

        ``obs_rows`` is not an input: every replica reports its own
        ``obs_rows`` from the one-pass reducer and the values are
        cross-validated across replicas so inconsistent shapes surface
        as an explicit error rather than a silent reshape.
        """
        grad_direct_by_x = []
        grad_reinforce_by_x = []
        sse_by_x = []
        n_samples_by_x = []
        obs_rows_by_x: list[int] = []

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

            grad_direct_by_x.append(
                np.asarray(zbx_stats["grad_direct"], dtype=np.float64)
            )
            grad_reinforce_by_x.append(
                np.asarray(zbx_stats["grad_reinforce"], dtype=np.float64)
            )
            sse_by_x.append(float(zbx_stats["sse"]))
            n_samples_by_x.append(int(zbx_stats["n_samples"]))
            obs_rows_by_x.append(int(zbx_stats["obs_rows"]))

        if not grad_direct_by_x:
            raise RuntimeError(
                "No zbx results collected — cannot build CDFM batch."
            )

        unique_obs = set(obs_rows_by_x)
        if len(unique_obs) != 1:
            raise ValueError(
                f"Inconsistent obs_rows across zbx replicas: {obs_rows_by_x}. "
                "Every replica must report the same number of observed real-site "
                "force rows; mismatch indicates a topology drift between init "
                "configs."
            )
        obs_rows = obs_rows_by_x[0]

        mode = str(
            self.config.training.extras.get("cdfm_mode", "direct")
        ).strip().lower()

        return CDFMTrainerAnalytic.make_batch(
            grad_direct_by_x=np.array(grad_direct_by_x),
            grad_reinforce_by_x=np.array(grad_reinforce_by_x),
            sse_by_x=np.array(sse_by_x),
            n_samples_by_x=np.array(n_samples_by_x, dtype=np.int64),
            obs_rows=obs_rows,
            # x_weight — omitted; defaults to uniform averaging over x
            mode=mode,
            step_index=epoch,
        )

    # ── run ─────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute the CDFM training loop."""
        cfg = self.config
        n_epochs = cfg.training.n_epochs
        start_epoch = cfg.training.start_epoch
        K = cfg.conditioning.n_samples
        mode = str(cfg.training.extras.get("cdfm_mode", "direct")).strip().lower()
        results = []
        pending_validation_epoch: Optional[int] = None

        self._load_resume_state(start_epoch)
        if (
            start_epoch > 0
            and self._validation_enabled()
            and start_epoch < n_epochs
            and self._validation_due_after_epoch(start_epoch - 1)
        ):
            pending_validation_epoch = start_epoch - 1

        for epoch in range(start_epoch, n_epochs):
            iter_dir = self.output_dir / f"iter_{epoch:04d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # ── Phase 1: write FF ────────────────────────────────
            ff_dir = iter_dir / "ff"
            ff_dir.mkdir(exist_ok=True)
            self._write_forcefield(ff_dir)
            ff_snapshot_path = self._snapshot_forcefield(ff_dir)

            # ── Phase 2: stage zbx replicas (K conditioned) ─────
            zbx_dir = iter_dir / "zbx"
            zbx_state = self.zbx_sampler.init_epoch(
                iteration_index=epoch,
                epoch_dir=zbx_dir,
                n_runs=K,
            )
            for plan in zbx_state.replica_plans:
                self._copy_ff_to_run(ff_dir, plan.run_dir)

            # ── Phase 3: build task specs ────────────────────────
            zbx_tasks = [
                self._build_zbx_task(plan, ff_snapshot_path, mode=mode)
                for plan in zbx_state.replica_plans
            ]
            xz_tasks = []
            if pending_validation_epoch is not None:
                xz_tasks.append(
                    self._build_validation_task(
                        label=f"epoch_{pending_validation_epoch:04d}",
                        epoch=pending_validation_epoch,
                        resource_pool=self.scheduler.pool,
                    )
                )
                pending_validation_epoch = None

            # ── Phase 4: scheduler execution ─────────────────────
            iter_result = self.scheduler.run_iteration(
                xz_tasks=xz_tasks,
                zbx_tasks=zbx_tasks,
                iter_dir=iter_dir,
            )
            self._record_validation_results(iter_result)

            # ── Phase 5: read results + train ────────────────────
            batch = self._collect_cdfm_batch(zbx_state, epoch)
            step_out = self.trainer.step(batch, apply_update=True)
            self.forcefield.update_params(self.trainer.get_params())
            results.append(step_out)
            param_mask = self.forcefield.param_mask
            logger.info(
                "Epoch %d: loss=%.6g  grad_norm=%.6g  n_zbx=%d  "
                "mask_active=%d/%d",
                epoch,
                float(step_out["loss"]),
                float(np.linalg.norm(step_out["grad"])),
                iter_result.succeeded_zbx,
                int(np.asarray(param_mask).sum()),
                int(np.asarray(param_mask).size),
            )
            self.zbx_sampler.clean_epoch(zbx_state)
            self._snapshot_optimizer(ff_dir)
            self._write_workflow_checkpoint(ff_dir)
            if (
                self._validation_enabled()
                and self._validation_due_after_epoch(epoch)
                and epoch < n_epochs - 1
            ):
                pending_validation_epoch = epoch

        if self._validation_enabled():
            self._run_validation_blocking(
                label="final",
                epoch=int(n_epochs) - 1,
                scheduler=self.scheduler,
            )

        return {
            "epochs": len(results),
            "results": results,
            "validation": self._validation_result_payload(),
        }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``acg-cdfm`` entry point."""
    return _run_workflow_cli(
        CDFMWorkflow,
        prog="acg-cdfm",
        description="Run the AceCG CDFM workflow.",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
