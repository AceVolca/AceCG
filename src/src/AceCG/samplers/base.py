"""Sampler classes for replica planning and light run management.

The sampler layer owns only:

- script inspection
- init-config selection
- replay-pool maintenance
- self-contained replica directory staging

It does not rewrite user scripts or invent new simulation logic.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Union

from ._lammps_script import stage_lammps_input_tree
from ._script_inspector import ScriptInfo, parse_script

PathLike = Union[str, Path]


@dataclass(frozen=True)
class InitConfigRecord:
    """One entry in an init-config pool."""

    path: Path
    frame_id: Optional[int] = None
    force_path: Optional[Path] = None


@dataclass(frozen=True)
class ReplicaPlan:
    """Plan for one replica run within an epoch."""

    run_id: int
    run_dir: Path
    input_script_path: Path
    init_config_path: Path
    frame_id: Optional[int]
    read_data_target: Path
    trajectory_path: Path
    write_data_path: Optional[Path]
    init_force_path: Optional[Path] = None


@dataclass
class EpochState:
    """State for one sampling epoch."""

    iteration_index: int
    epoch_dir: Path
    script_info: ScriptInfo
    replica_plans: list[ReplicaPlan] = field(default_factory=list)


@dataclass(frozen=True)
class RunResult:
    """Result of a single local sampler run."""

    run_id: int
    run_dir: Path
    frame_id: Optional[int]
    trajectory_path: Path
    returncode: int = 0


class BaseSampler:
    """Simulation sampling utility.

    The current public contract is intentionally small:

    - ``__init__`` validates script structure and config pools
    - ``init_epoch`` prepares self-contained replica directories
    - ``run`` executes one local replica as a debug / fallback path
    - ``clean_epoch`` updates replay state from completed runs
    """

    def __init__(
        self,
        *,
        sim_input: PathLike,
        sim_backend: str = "lammps",
        init_config_pool: Sequence[PathLike | InitConfigRecord] | None = None,
        replay_mode: Literal["off", "latest", "random"] = "off",
        rng: random.Random | None = None,
    ) -> None:
        self._sim_input = Path(sim_input)
        self._sim_backend = sim_backend
        self._replay_mode = replay_mode
        self._replay_pool: list[Path] = []
        self._rng = rng if rng is not None else random.Random()

        self._script_info: ScriptInfo = parse_script(self._sim_input, backend=sim_backend)
        self._validate_runtime_paths(self._script_info)

        if replay_mode != "off" and self._script_info.checkpoint_path is None:
            raise ValueError(
                f"Replay mode '{replay_mode}' requires the script to produce "
                f"a checkpoint output, but none was found in {self._sim_input}."
            )

        self._init_pool = self._normalize_init_pool(init_config_pool)
        if self._init_pool is None:
            rd = self._resolve_default_init_path()
            if not rd.exists():
                raise FileNotFoundError(
                    f"Script's init data file not found: {rd}. "
                    "Provide init_config_pool or fix the script."
                )

    @property
    def script_info(self) -> ScriptInfo:
        """Return parsed metadata for the sampler input script."""
        return self._script_info

    @property
    def replay_pool(self) -> list[Path]:
        """Return checkpoint/data files available for replay initialization."""
        return list(self._replay_pool)

    def state_dict(self) -> Dict[str, Any]:
        """Return the minimal sampler state required for resume."""
        return {
            "replay_pool": [str(path) for path in self._replay_pool],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore sampler state written by :meth:`state_dict`."""
        replay_pool = state.get("replay_pool", [])
        self._replay_pool = [Path(item).resolve(strict=False) for item in replay_pool]

    def init_epoch(
        self,
        *,
        iteration_index: int,
        epoch_dir: PathLike,
        n_runs: int = 1,
    ) -> EpochState:
        """Prepare *n_runs* replica directories and return an EpochState."""
        epoch_path = Path(epoch_dir)
        epoch_path.mkdir(parents=True, exist_ok=True)

        plans: list[ReplicaPlan] = []
        for run_id in range(n_runs):
            run_dir = epoch_path / f"run_{run_id:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            script_copy = self._stage_run_dir(run_dir)
            init_cfg, fid = self._choose_init_config()
            plan = self._build_plan(run_id, run_dir, script_copy, init_cfg, fid)
            plan.read_data_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(plan.init_config_path, plan.read_data_target)
            plans.append(plan)

        return EpochState(
            iteration_index=iteration_index,
            epoch_dir=epoch_path,
            script_info=self._script_info,
            replica_plans=plans,
        )

    def run(
        self,
        plan: ReplicaPlan,
        *,
        sim_cmd: Sequence[str] | None = None,
    ) -> RunResult:
        """Execute one replica locally (debug / non-scheduler fallback)."""
        if sim_cmd is None:
            sim_cmd = ["lmp"] if self._sim_backend == "lammps" else []
        cmd = list(sim_cmd) + ["-in", plan.input_script_path.name]
        result = subprocess.run(cmd, cwd=str(plan.run_dir))

        return RunResult(
            run_id=plan.run_id,
            run_dir=plan.run_dir,
            frame_id=plan.frame_id,
            trajectory_path=plan.trajectory_path,
            returncode=result.returncode,
        )

    def clean_epoch(self, state: EpochState) -> None:
        """Update replay state from completed replicas in *state*.

        Replay ownership stays inside the sampler lifecycle. Any produced
        ``write_data`` file is copied into an epoch-local replay archive before
        the archived path is appended to the pool.
        """
        if self._replay_mode == "off":
            return

        replay_dir = state.epoch_dir / "replay_pool"
        replay_dir.mkdir(parents=True, exist_ok=True)
        for plan in state.replica_plans:
            if plan.write_data_path is None or not plan.write_data_path.exists():
                continue
            archived = replay_dir / f"run_{plan.run_id:04d}_{plan.write_data_path.name}"
            shutil.copy2(plan.write_data_path, archived)
            self._replay_pool.append(archived)

    def _normalize_init_pool(
        self,
        init_config_pool: Sequence[PathLike | InitConfigRecord] | None,
    ) -> list[InitConfigRecord] | None:
        if init_config_pool is None:
            return None

        normalized: list[InitConfigRecord] = []
        for entry in init_config_pool:
            if isinstance(entry, InitConfigRecord):
                rec = InitConfigRecord(
                    path=Path(entry.path),
                    frame_id=entry.frame_id,
                    force_path=Path(entry.force_path) if entry.force_path is not None else None,
                )
            else:
                rec = InitConfigRecord(path=Path(entry))
            if not rec.path.exists():
                raise FileNotFoundError(f"Init config file not found: {rec.path}")
            normalized.append(rec)
        return normalized

    def _resolve_default_init_path(self) -> Path:
        rd = self._script_info.init_data_path
        if not rd.is_absolute():
            rd = self._sim_input.parent / rd
        return rd

    def _stage_run_dir(self, run_dir: Path) -> Path:
        if self._sim_backend == "lammps":
            return stage_lammps_input_tree(self._sim_input, run_dir)
        raise NotImplementedError(
            f"Replica staging for backend '{self._sim_backend}' is not implemented."
        )

    def _build_plan(
        self,
        run_id: int,
        run_dir: Path,
        script_copy: Path,
        init_cfg: Path,
        frame_id: Optional[int],
        force_path: Optional[Path] = None,
    ) -> ReplicaPlan:
        si = self._script_info
        return ReplicaPlan(
            run_id=run_id,
            run_dir=run_dir,
            input_script_path=script_copy,
            init_config_path=init_cfg,
            frame_id=frame_id,
            read_data_target=run_dir / Path(si.init_data_path),
            trajectory_path=run_dir / Path(si.trajectory_path),
            write_data_path=(run_dir / Path(si.checkpoint_path)) if si.checkpoint_path else None,
            init_force_path=force_path,
        )

    def _choose_init_config(self) -> tuple[Path, int | None]:
        if self._replay_mode != "off" and self._replay_pool:
            if self._replay_mode == "latest":
                return self._replay_pool[-1], None
            return self._rng.choice(self._replay_pool), None

        if self._init_pool is not None:
            rec = self._rng.choice(self._init_pool)
            return rec.path, rec.frame_id

        return self._resolve_default_init_path(), None

    @staticmethod
    def _validate_runtime_paths(script_info: ScriptInfo) -> None:
        """Reject script-managed runtime paths that would escape replica dirs.

        The sampler stages each replica into its own directory and copies data
        files onto the paths used by the script. Absolute paths or ``..``
        segments would break isolation and can make replicas overwrite each
        other or the user's canonical files.
        """
        BaseSampler._validate_local_path(script_info.init_data_path, "read_data path")
        BaseSampler._validate_local_path(script_info.trajectory_path, "dump path")
        if script_info.checkpoint_path is not None:
            BaseSampler._validate_local_path(script_info.checkpoint_path, "write_data path")

    @staticmethod
    def _validate_local_path(path: Path, label: str) -> None:
        if path.is_absolute():
            raise ValueError(
                f"{label} must be replica-local, got absolute path {path!s}"
            )
        if ".." in path.parts:
            raise ValueError(
                f"{label} must not escape the replica directory, got {path!s}"
            )
