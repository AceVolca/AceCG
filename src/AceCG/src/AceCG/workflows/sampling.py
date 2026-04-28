"""SamplingWorkflow — base for all sampling-based workflows (REM, CDREM, CDFM).

Adds forcefield loading (``ReadLmpFF``), sampler construction, scheduler
construction, and AA-data strategy on top of ``BaseWorkflow``.
"""

from __future__ import annotations

import copy
import pickle
import shlex
import shutil
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..configs.models import ACGConfig
from ..configs.utils import parse_pair_style_options
from ..io.forcefield import ReadLmpFF, WriteLmpFF
from ..samplers.base import BaseSampler, InitConfigRecord
from ..schedulers.task_runner import run_post
from ..schedulers.task_scheduler import TaskScheduler
from ..topology.forcefield import Forcefield
from .base import BaseWorkflow


# kcal/(mol·K)
_BOLTZMANN_KCAL = 0.001987204


@dataclass(frozen=True)
class AAStats:
    """All-atom reference derivative statistics for sampling workflows."""

    energy_grad: np.ndarray
    d2U: Optional[np.ndarray] = None

    @classmethod
    def from_payload(
        cls,
        payload: Dict[str, Any],
        *,
        grad_key: str,
        hess_key: str,
        need_hessian: bool,
        label: str,
    ) -> "AAStats":
        """Build AA statistics from a serialized payload.

        Parameters
        ----------
        payload : dict
            Source dictionary containing gradient and optional Hessian arrays.
        grad_key : str
            Key used to read the energy-gradient vector.
        hess_key : str
            Key used to read the Hessian matrix.
        need_hessian : bool
            Whether missing Hessian data should raise an error.
        label : str
            Human-readable label included in validation errors.
        """
        if not isinstance(payload, dict):
            raise ValueError(f"{label} must be a dict, got {type(payload).__name__}.")
        if grad_key not in payload:
            raise KeyError(f"{label} must contain {grad_key!r}.")
        d2u_raw = payload.get(hess_key)
        if need_hessian and d2u_raw is None:
            raise KeyError(f"{label} must contain {hess_key!r} when Hessian data is required.")
        return cls(
            energy_grad=np.asarray(payload[grad_key], dtype=np.float64),
            d2U=None if d2u_raw is None else np.asarray(d2u_raw, dtype=np.float64),
        )


class SamplingWorkflow(BaseWorkflow):
    """Base class for sampling-based workflows (REM, CDREM, CDFM).

    Adds on top of ``BaseWorkflow``:
        forcefield      – ``Forcefield`` loaded from LAMMPS settings
        resource_pool   – discovered compute resources
        scheduler       – ``TaskScheduler`` for running sim + post tasks
        sampler         – ``BaseSampler`` for xz sampling
        beta            – inverse temperature (kcal/mol)^{-1}
        aa_data_strategy – ``AAStats`` or ``Callable[[Forcefield], AAStats]``
    """

    def __init__(self, config: ACGConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.forcefield = self._build_forcefield()
        if self.config.vp is not None and self.config.vp.vp_names:
            self.forcefield.set_vp_masks(self.config.vp.vp_names)
        if self.config.system.forcefield_mask is not None:
            self.forcefield.build_mask(init_mask=self._build_forcefield_mask(self.forcefield))
        self.resource_pool = self._build_resource_pool()
        self.scheduler = self._build_scheduler()
        self.sampler = self._build_sampler()
        self.beta = self._derive_beta()
        self.aa_data_strategy: Optional[AAStats | Callable[[Forcefield], AAStats]] = None

    # ── builders ────────────────────────────────────────────────

    def _build_forcefield(self) -> Forcefield:
        """Load forcefield via ``ReadLmpFF`` from config paths."""
        cfg = self.config.system
        pair_style, sel_styles = parse_pair_style_options(cfg.pair_style)
        ff_path = self._resolve_config_path(cfg.forcefield_path)
        if ff_path is None:
            raise ValueError("system.forcefield_path is required for sampling workflows.")
        return ReadLmpFF(
            str(ff_path),
            pair_style,
            pair_typ_sel=sel_styles,
            cutoff=float(cfg.cutoff) if cfg.cutoff is not None else None,
            table_fit=cfg.table_fit or "multigaussian",
            table_fit_overrides=cfg.table_fit_overrides,
            topology_arrays=self.topology,
        )

    def _build_scheduler(self) -> TaskScheduler:
        cfg = self.config.scheduler
        return TaskScheduler(
            self.resource_pool,
            task_timeout=cfg.task_timeout,
            min_success_zbx=cfg.min_success_zbx,
            python_exe=cfg.python_exe,
            rng_seed=self.workflow_rng.randint(0, 2**31 - 1),
        )

    def _build_sampler(self) -> BaseSampler:
        cfg = self.config.sampling
        init_pool = [InitConfigRecord(path=p) for p in self._glob_config_paths(cfg.init_config_pool)]
        sim_input = self._resolve_config_path(cfg.input)
        if sim_input is None:
            raise ValueError("sampling.input is required for sampling workflows.")
        return BaseSampler(
            sim_input=sim_input,
            sim_backend=cfg.sim_backend,
            init_config_pool=init_pool or None,
            replay_mode=cfg.replay_mode,
            rng=self.workflow_rng,
        )

    def _derive_beta(self) -> float:
        """Derive β = 1/(kB·T) from config.training.temperature."""
        T = self.config.training.temperature
        if T is None or T <= 0:
            raise ValueError(
                "training.temperature must be a positive float for "
                f"{self.config.training.method} workflows."
            )
        return 1.0 / (_BOLTZMANN_KCAL * T)

    def _build_aa_data_strategy(self) -> AAStats | Callable[[Forcefield], AAStats]:
        """Build AA-reference data as a constant object or per-epoch callable."""
        if not hasattr(self, "trainer"):
            raise AttributeError("AA data strategy requires a constructed trainer.")

        need_hessian = bool(
            self.trainer.optimizer_accepts_hessian() or self.config.training.need_hessian
        )
        cache_path = self._resolve_config_path(self.config.aa_ref.all_atom_data_path)
        is_linear = self.trainer.is_optimization_linear()

        if is_linear:
            if cache_path is not None:
                if not cache_path.exists():
                    raise FileNotFoundError(f"all_atom_data_path {cache_path!s} does not exist.")
                with open(cache_path, "rb") as fh:
                    payload = pickle.load(fh)
                return AAStats.from_payload(
                    payload,
                    grad_key="energy_grad_AA",
                    hess_key="d2U_AA",
                    need_hessian=need_hessian,
                    label=f"AA data at {cache_path!s}",
                )
            return self._run_aa_post(
                work_dir=self.output_dir / "aa_precompute",
                forcefield=self.forcefield,
                need_hessian=need_hessian,
            )

        if not self.config.aa_ref.trajectory_files:
            raise ValueError(
                "Nonlinear REM requires aa_ref.trajectory_files so AA statistics "
                "can be recomputed for the current forcefield each epoch."
            )

        aa_counter = count()

        def compute_aa_stats(forcefield: Forcefield) -> AAStats:
            step_index = next(aa_counter)
            return self._run_aa_post(
                work_dir=self.output_dir / f"aa_recompute_{step_index:04d}",
                forcefield=forcefield,
                need_hessian=need_hessian,
            )

        return compute_aa_stats

    # ── FF write / snapshot helpers ─────────────────────────────

    def _write_forcefield(self, ff_dir: Path) -> Path:
        """Write the current runtime forcefield bundle under *ff_dir*."""
        cfg = self.config.system
        pair_style, sel_styles = parse_pair_style_options(cfg.pair_style)
        source = Path(cfg.forcefield_path)
        runtime_relpath = Path(source.name)
        ff_file = ff_dir / runtime_relpath
        ff_file.parent.mkdir(parents=True, exist_ok=True)
        # WriteLmpFF needs an existing source file to copy structure from
        src = self._resolve_config_path(cfg.forcefield_path)
        if src is None:
            raise ValueError("system.forcefield_path is required for runtime forcefield export.")
        WriteLmpFF(
            str(src),
            str(ff_file),
            self.forcefield,
            pair_style,
            pair_typ_sel=sel_styles,
            topology_arrays=self.topology,
        )
        return ff_file

    def _snapshot_forcefield(self, ff_dir: Path, forcefield: Optional[Forcefield] = None) -> Path:
        """Pickle a deep-copy of the forcefield for MPI engine consumption."""
        snapshot_path = ff_dir / "forcefield_snapshot.pkl"
        ff_dir.mkdir(parents=True, exist_ok=True)
        with open(snapshot_path, "wb") as fh:
            pickle.dump(
                copy.deepcopy(self.forcefield if forcefield is None else forcefield), fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return snapshot_path

    def _snapshot_optimizer(self, ff_dir: Path) -> Path:
        """Persist optimizer state so Adam/AdamW resume keeps its moments.

        Why: Newton resumes cleanly from ``L`` alone, but stateful optimizers
        (Adam, AdamW, RMSprop) carry moment buffers that define the effective
        step size.  Without this snapshot, resume silently restarts from
        zero-moment and disrupts convergence.
        How to apply: call per-epoch next to ``_snapshot_forcefield``; pair
        with ``_load_optimizer_snapshot`` in the workflow's resume branch.
        """
        snapshot_path = ff_dir / "optimizer_snapshot.pkl"
        ff_dir.mkdir(parents=True, exist_ok=True)
        with open(snapshot_path, "wb") as fh:
            pickle.dump(self.optimizer.state_dict(), fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return snapshot_path

    def _write_workflow_checkpoint(self, ff_dir: Path) -> Path:
        """Persist the completed-epoch state used for workflow resume."""
        snapshot_path = ff_dir / "workflow_checkpoint.pkl"
        ff_dir.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "forcefield": copy.deepcopy(self.forcefield),
            "optimizer_state": self.optimizer.state_dict(),
            "workflow_rng_state": self.workflow_rng.getstate(),
        }
        sampler = getattr(self, "sampler", None)
        if sampler is not None and hasattr(sampler, "state_dict"):
            payload["sampler_state"] = sampler.state_dict()
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            payload["scheduler_state"] = scheduler.state_dict()
        with open(snapshot_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return snapshot_path

    def _load_workflow_checkpoint(self, ff_dir: Path) -> Path:
        """Restore a completed-epoch state written by :meth:`_write_workflow_checkpoint`."""
        snapshot_path = ff_dir / "workflow_checkpoint.pkl"
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Workflow checkpoint {snapshot_path} not found.")
        with open(snapshot_path, "rb") as fh:
            payload = pickle.load(fh)
        self.forcefield = payload["forcefield"]
        self.optimizer = self._build_optimizer(self.forcefield)
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.workflow_rng.setstate(payload["workflow_rng_state"])
        sampler = getattr(self, "sampler", None)
        sampler_state = payload.get("sampler_state")
        if sampler is not None and sampler_state is not None and hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(sampler_state)
        scheduler = getattr(self, "scheduler", None)
        scheduler_state = payload.get("scheduler_state")
        if scheduler is not None and scheduler_state is not None and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(scheduler_state)
        return snapshot_path

    def _load_optimizer_snapshot(self, snapshot_path: Path) -> None:
        """Restore optimizer state written by :meth:`_snapshot_optimizer`.

        Silently tolerates a missing file (pre-snapshot runs) — resume then
        degrades to zero-moment init, same as before this helper existed.
        """
        if not snapshot_path.exists():
            return
        with open(snapshot_path, "rb") as fh:
            state = pickle.load(fh)
        self.optimizer.load_state_dict(state)

    def _elastic_core_bounds(
        self, explicit_ncores: Optional[int],
    ) -> tuple[int, int, int, int]:
        """Derive (cpu_cores, min_cores, preferred_cores, max_cores).

        When the user pins ``ncores`` in the config, all four collapse to the
        same value — fixed-width placement.  Otherwise the bounds span the
        pool: ``min`` = smallest host, ``preferred`` = largest host,
        ``max`` = total pool.  This lets the Placer scale a task across a
        fragmented allocation instead of wedging at ``max(host.n_cpus)``,
        which only fits the widest single node.

        ``cpu_cores`` is the nominal default written into ``TaskSpec`` — the
        Placer overrides it at launch time with the actual allocation.
        """
        hosts = self.resource_pool.hosts
        if not hosts:
            raise RuntimeError("resource_pool has no hosts")
        if explicit_ncores is not None:
            n = int(explicit_ncores)
            return n, n, n, n
        max_host = max(h.n_cpus for h in hosts)
        min_host = min(h.n_cpus for h in hosts)
        total = sum(h.n_cpus for h in hosts)
        return max_host, min_host, max_host, total

    # ── AA engine post-processing helpers ───────────────────────

    def _run_aa_post(
        self,
        *,
        work_dir: Path,
        forcefield: Forcefield,
        need_hessian: bool,
    ) -> AAStats:
        """Run MPI engine in ``rem`` mode on AA reference trajectory.

        Returns ``AAStats`` for the provided forcefield.
        """
        cfg = self.config
        work_dir.mkdir(parents=True, exist_ok=True)
        ff_snapshot_path = self._snapshot_forcefield(work_dir / "ff", forcefield)
        output_file = work_dir / "aa_stats.pkl"
        trajectories = [str(self._resolve_config_path(t)) for t in cfg.aa_ref.trajectory_files]
        if not trajectories:
            raise ValueError("aa_ref.trajectory_files is required to compute AA statistics.")
        topology_path = self._resolve_config_path(cfg.system.topology_file)
        if topology_path is None:
            raise ValueError("system.topology_file is required to compute AA statistics.")

        spec: Dict[str, Any] = {
            "work_dir": str(work_dir),
            "forcefield_path": str(ff_snapshot_path),
            "topology": str(topology_path),
            "trajectory": trajectories,
            "trajectory_format": cfg.aa_ref.trajectory_format,
            "exclude_bonded": cfg.system.exclude_bonded,
            "exclude_option": cfg.system.exclude_option,
            "cutoff": cfg.system.cutoff,
            "steps": [
                {
                    "step_mode": "rem",
                    "need_hessian": need_hessian,
                    "output_file": str(output_file),
                }
            ],
        }
        # AA-ref topology / alias overrides (from parser resolution)
        if cfg.aa_ref.ref_topo is not None:
            spec["topology"] = str(self._resolve_config_path(cfg.aa_ref.ref_topo))
        if cfg.aa_ref.ref_resolved_aliases is not None:
            spec["atom_type_name_aliases"] = cfg.aa_ref.ref_resolved_aliases
        elif cfg.system.type_names is not None:
            spec["atom_type_name_aliases"] = cfg.system.type_names
        if cfg.aa_ref.ref_has_vp and cfg.vp is not None:
            spec["vp_names"] = list(cfg.vp.vp_names)
        if cfg.aa_ref.every != 1:
            spec["every"] = cfg.aa_ref.every
        if cfg.aa_ref.skip_frames > 0:
            spec["frame_start"] = cfg.aa_ref.skip_frames
        if cfg.aa_ref.n_frames > 0:
            spec["frame_end"] = cfg.aa_ref.skip_frames + cfg.aa_ref.n_frames

        run_post(
            spec,
            self.resource_pool,
            run_dir=work_dir,
            python_exe=cfg.scheduler.python_exe or None,
        )
        if not output_file.exists():
            raise RuntimeError(
                f"AA post-processing produced no output at {output_file}"
            )
        with open(output_file, "rb") as fh:
            payload = pickle.load(fh)
        return AAStats.from_payload(
            payload,
            grad_key="energy_grad_avg",
            hess_key="d2U_avg",
            need_hessian=need_hessian,
            label=f"AA engine result at {output_file!s}",
        )
