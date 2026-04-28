"""Conditioned sampler for VP-based zbx branches."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Sequence, Union

from .base import BaseSampler, EpochState, InitConfigRecord

PathLike = Union[str, Path]


class ConditionedSampler(BaseSampler):
    """Sampler for conditioned zbx replicas in CDREM / CDFM workflows.

    Contract differences from ``BaseSampler``:

    - ``init_config_pool`` is mandatory
    - every init-config entry must carry ``frame_id``
    - replay is always disabled
    - ``init_epoch`` samples K unique init configs
    """

    def __init__(
        self,
        *,
        sim_input: PathLike,
        sim_backend: str = "lammps",
        init_config_pool: Sequence[InitConfigRecord],
        require_force_path: bool = False,
        rng: Any = None,
    ) -> None:
        super().__init__(
            sim_input=sim_input,
            sim_backend=sim_backend,
            init_config_pool=init_config_pool,
            replay_mode="off",
            rng=rng,
        )
        if self._init_pool is None or not self._init_pool:
            raise ValueError("ConditionedSampler requires a non-empty init_config_pool")
        for rec in self._init_pool:
            if rec.frame_id is None:
                raise ValueError(
                    f"ConditionedSampler requires frame_id for every entry, "
                    f"but {rec.path} has frame_id=None"
                )
            if require_force_path:
                if rec.force_path is None:
                    raise ValueError(
                        f"ConditionedSampler(require_force_path=True) needs a "
                        f"force_path for every entry, but {rec.path} "
                        f"(frame_id={rec.frame_id}) has force_path=None"
                    )
                if not rec.force_path.exists():
                    raise FileNotFoundError(
                        f"Init force file not found: {rec.force_path}"
                    )

    def init_epoch(
        self,
        *,
        iteration_index: int,
        epoch_dir: PathLike,
        n_runs: int,
    ) -> EpochState:
        """Prepare one conditioned-sampling epoch.

        Parameters
        ----------
        iteration_index : int
            Workflow iteration index.
        epoch_dir : PathLike
            Directory where run subdirectories are created.
        n_runs : int
            Number of conditioned replicas to stage.

        Returns
        -------
        EpochState
            Staged run plans and metadata for the epoch.
        """
        if n_runs > len(self._init_pool):
            raise ValueError(f"n_runs={n_runs} exceeds pool size {len(self._init_pool)}")

        epoch_path = Path(epoch_dir)
        epoch_path.mkdir(parents=True, exist_ok=True)

        chosen = self._rng.sample(self._init_pool, k=n_runs)
        plans = []
        for run_id, rec in enumerate(chosen):
            run_dir = epoch_path / f"zbx_{run_id:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            script_copy = self._stage_run_dir(run_dir)
            plan = self._build_plan(
                run_id,
                run_dir,
                script_copy,
                rec.path,
                rec.frame_id,
                force_path=rec.force_path,
            )
            plan.read_data_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plan.init_config_path, plan.read_data_target)
            plans.append(plan)

        return EpochState(
            iteration_index=iteration_index,
            epoch_dir=epoch_path,
            script_info=self.script_info,
            replica_plans=plans,
        )
