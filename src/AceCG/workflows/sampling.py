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

from ..configs.energy_mask import normalize_energy_mask_spec
from ..configs.models import ACGConfig
from ..configs.utils import parse_pair_style_options
from ..io.forcefield import ReadLmpFF
from ..samplers.base import BaseSampler, InitConfigRecord
from ..schedulers.task_runner import run_post
from ..schedulers.task_scheduler import TaskScheduler
from ..topology.forcefield import Forcefield
from .base import BaseWorkflow


_BOLTZMANN_KCAL = 0.001987204  # kcal/(mol·K)


def resolve_aa_noise_sigma(noise: Any, *, n_epochs: int, stage: int) -> float:
    sigma0 = float(noise.sigma)
    sigma1 = sigma0 if noise.sigma_final is None else float(noise.sigma_final)
    if sigma0 == sigma1:
        return sigma0
    interval = max(int(noise.update_interval), 1)
    n_stages = max((int(n_epochs) + interval - 1) // interval, 1)
    progress = 0.0 if n_stages <= 1 else min(max(int(stage), 0), n_stages - 1) / float(n_stages - 1)
    schedule = str(noise.schedule).strip().lower()
    if schedule == "constant":
        return sigma0
    if schedule == "cosine":
        return sigma1 + 0.5 * (sigma0 - sigma1) * (1.0 + np.cos(np.pi * progress))
    if schedule == "exponential":
        if sigma0 <= 0.0 or sigma1 <= 0.0:
            return sigma1 if progress >= 1.0 else sigma0
        return sigma0 * ((sigma1 / sigma0) ** progress)
    raise ValueError(f"Unsupported aa_ref noise schedule {schedule!r}.")


@dataclass(frozen=True)
class AAStats:
    """All-atom reference derivative statistics for sampling workflows."""

    energy_grad: np.ndarray
    d2U: Optional[np.ndarray] = None
    gradient_convention: str = "physical"
    unmasked_energy_grad: Optional[np.ndarray] = None

    @staticmethod
    def _normalize_gradient_convention(value: Any) -> Optional[str]:
        """Normalize serialized gradient-convention labels."""
        if value is None:
            return None
        convention = str(value).strip()
        if not convention:
            return None
        return convention.lower()

    @classmethod
    def from_payload(
        cls,
        payload: Dict[str, Any],
        *,
        grad_key: str,
        hess_key: str,
        need_hessian: bool,
        label: str,
        expected_gradient_convention: Optional[str] = None,
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
        expected_gradient_convention : str, optional
            Required gradient convention for the serialized payload.
        """
        if not isinstance(payload, dict):
            raise ValueError(f"{label} must be a dict, got {type(payload).__name__}.")
        if grad_key not in payload:
            raise KeyError(f"{label} must contain {grad_key!r}.")
        d2u_raw = payload.get(hess_key)
        if need_hessian and d2u_raw is None:
            raise KeyError(f"{label} must contain {hess_key!r} when Hessian data is required.")
        gradient_convention = cls._normalize_gradient_convention(
            payload.get("gradient_convention")
        )
        expected_gradient_convention = cls._normalize_gradient_convention(
            expected_gradient_convention
        )
        if expected_gradient_convention == "gauge_free" and not gradient_convention:
            raise KeyError(
                f"{label} must contain 'gradient_convention' for cached REM statistics. "
                "Accepted value is 'gauge_free'."
            )
        if (
            expected_gradient_convention is not None
            and gradient_convention is not None
            and gradient_convention != expected_gradient_convention
        ):
            raise ValueError(
                f"{label} uses gradient_convention={gradient_convention!r}, "
                f"expected {expected_gradient_convention!r}."
            )
        return cls(
            energy_grad=np.asarray(payload[grad_key], dtype=np.float64),
            d2U=None if d2u_raw is None else np.asarray(d2u_raw, dtype=np.float64),
            gradient_convention=gradient_convention or expected_gradient_convention or "physical",
            unmasked_energy_grad=None
            if payload.get("unmasked_energy_grad_avg") is None
            else np.asarray(payload["unmasked_energy_grad_avg"], dtype=np.float64),
        )


class SamplingWorkflow(BaseWorkflow):
    """Base class for sampling-based workflows (REM, CDREM, CDFM).

    Adds on top of ``BaseWorkflow``:
        forcefield      – ``Forcefield`` loaded from LAMMPS settings
        resource_pool   – discovered compute resources
        scheduler       – ``TaskScheduler`` for running sim + post tasks
        sampler         – ``BaseSampler`` for xz sampling
        beta            – inverse temperature (kcal/mol)^{-1}
        aa_data_strategy – ``AAStats`` or an epoch-aware AAStats callable
    """

    def __init__(self, config: ACGConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.forcefield = self._build_forcefield()
        if self.config.vp is not None and self.config.vp.vp_names:
            self.forcefield.set_vp_masks(self.config.vp.vp_names)
        if self.config.system.forcefield_mask is not None:
            self.forcefield.build_mask(init_mask=self._build_forcefield_mask(self.forcefield))
        if self.config.system.forcefield_bounds is not None:
            self._apply_forcefield_bounds(self.forcefield)
        self.resource_pool = self._build_resource_pool()
        self.scheduler = self._build_scheduler()
        self.sampler = self._build_sampler()
        self.beta = self._derive_beta()
        self.aa_data_strategy: Optional[AAStats | Callable[..., AAStats]] = None

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

    def _build_aa_data_strategy(self) -> AAStats | Callable[..., AAStats]:
        """Build AA-reference data as a constant object or per-epoch callable."""
        if not hasattr(self, "trainer"):
            raise AttributeError("AA data strategy requires a constructed trainer.")

        need_hessian = bool(
            self.trainer.optimizer_accepts_hessian() or self.config.training.need_hessian
        )
        cache_path = self._resolve_config_path(self.config.aa_ref.all_atom_data_path)
        training_method = str(self.config.training.method).strip().lower()
        expected_gradient_convention = "gauge_free" if training_method == "rem" else "physical"
        is_cacheable = (
            self.trainer.is_gauge_free_energy_grad_cacheable()
            if expected_gradient_convention == "gauge_free"
            else self.trainer.is_optimization_linear()
        )
        noise_enabled = bool(getattr(self.config.aa_ref.noise, "enabled", False))

        if noise_enabled:
            if not self.config.aa_ref.trajectory_files:
                raise ValueError(
                    "aa_ref.trajectory_files is required when aa_ref noise is enabled."
                )
            aa_cache: Dict[int, AAStats] = {}
            aa_counter = count()

            def compute_noisy_aa_stats(
                forcefield: Forcefield,
                epoch: Optional[int] = None,
            ) -> AAStats:
                stage = self._aa_noise_stage(epoch)
                use_stage_cache = is_cacheable and self.config.aa_ref.noise.cache_policy == "stage"
                if use_stage_cache and stage in aa_cache:
                    return aa_cache[stage]
                if epoch is None:
                    suffix = next(aa_counter)
                    work_dir = self.output_dir / f"aa_noise_stage_{stage:04d}_{suffix:04d}"
                elif is_cacheable:
                    work_dir = self.output_dir / f"aa_noise_stage_{stage:04d}"
                else:
                    work_dir = self.output_dir / f"aa_recompute_{int(epoch):04d}"
                stats = self._run_aa_post(
                    work_dir=work_dir,
                    forcefield=forcefield,
                    need_hessian=need_hessian,
                    noise_spec=self._aa_noise_runtime_spec(stage, epoch=epoch),
                    gradient_convention=expected_gradient_convention,
                )
                if use_stage_cache:
                    aa_cache[stage] = stats
                return stats

            return compute_noisy_aa_stats

        if is_cacheable:
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
                    expected_gradient_convention=expected_gradient_convention,
                )
            return self._run_aa_post(
                work_dir=self.output_dir / "aa_precompute",
                forcefield=self.forcefield,
                need_hessian=need_hessian,
                gradient_convention=expected_gradient_convention,
            )

        if not self.config.aa_ref.trajectory_files:
            raise ValueError(
                "Non-cacheable REM requires aa_ref.trajectory_files so AA statistics "
                "can be recomputed for the current forcefield each epoch."
            )

        aa_counter = count()

        def compute_aa_stats(
            forcefield: Forcefield,
            epoch: Optional[int] = None,
        ) -> AAStats:
            step_index = next(aa_counter) if epoch is None else int(epoch)
            return self._run_aa_post(
                work_dir=self.output_dir / f"aa_recompute_{step_index:04d}",
                forcefield=forcefield,
                need_hessian=need_hessian,
                gradient_convention=expected_gradient_convention,
            )

        return compute_aa_stats

    # ── FF write / snapshot helpers ─────────────────────────────

    def _aa_noise_stage(self, epoch: Optional[int]) -> int:
        noise = self.config.aa_ref.noise
        if epoch is None:
            return 0
        return int(epoch) // int(noise.update_interval)

    def _aa_noise_runtime_spec(
        self,
        stage: int,
        *,
        epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        noise = self.config.aa_ref.noise
        runtime = noise.to_runtime_dict()
        runtime["sigma"] = self._aa_noise_sigma(stage)
        runtime["seed"] = int(noise.seed) + int(stage)
        if int(noise.subsample_per_epoch) > 0:
            subsample_stage = int(stage) if epoch is None else int(epoch)
            runtime["subsample_seed"] = int(noise.seed) + subsample_stage
        return runtime

    def _aa_noise_sigma(self, stage: int) -> float:
        return resolve_aa_noise_sigma(
            self.config.aa_ref.noise,
            n_epochs=int(self.config.training.n_epochs),
            stage=stage,
        )

    def _write_forcefield(self, ff_dir: Path) -> Path:
        """Write the current runtime forcefield bundle under *ff_dir*."""
        return self._write_lammps_forcefield_bundle(ff_dir)

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
        trainer = getattr(self, "trainer", None)
        optimizer = getattr(trainer, "optimizer", self.optimizer)
        with open(snapshot_path, "wb") as fh:
            pickle.dump(optimizer.state_dict(), fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        return snapshot_path

    def _write_workflow_checkpoint(self, ff_dir: Path) -> Path:
        """Persist the completed-epoch state used for workflow resume."""
        snapshot_path = ff_dir / "workflow_checkpoint.pkl"
        ff_dir.mkdir(parents=True, exist_ok=True)
        trainer = getattr(self, "trainer", None)
        optimizer = getattr(trainer, "optimizer", self.optimizer)
        payload: Dict[str, Any] = {
            "forcefield": copy.deepcopy(self.forcefield),
            "optimizer_state": optimizer.state_dict(),
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
        return self._core_bounds_from_pool(self.resource_pool, explicit_ncores)

    # ── AA engine post-processing helpers ───────────────────────

    def _run_aa_post(
        self,
        *,
        work_dir: Path,
        forcefield: Forcefield,
        need_hessian: bool,
        noise_spec: Optional[Dict[str, Any]] = None,
        gradient_convention: str = "physical",
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
                    "step_mode": cfg.training.method,
                    "need_hessian": need_hessian,
                    "output_file": str(output_file),
                }
            ],
        }
        self._apply_rem_statistics_options(spec["steps"][0])
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
        if noise_spec is not None:
            spec["noise"] = dict(noise_spec)
        self._apply_post_runtime_options(spec)

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
            expected_gradient_convention=gradient_convention,
        )

    def _apply_post_runtime_options(self, spec: Dict[str, Any]) -> None:
        if self.config.sampling.perf_trace:
            spec["perf_trace"] = True
            interval = self.config.sampling.extras.get("heartbeat_interval", 25)
            spec["heartbeat_interval"] = int(interval)
        if bool(self.config.sampling.extras.get("perf_trace_all_ranks", False)):
            spec["perf_trace_all_ranks"] = True

    def _energy_mask_runtime_spec(self) -> Optional[dict[str, dict[str, float | None]]]:
        raw = self.config.training.extras.get("energy_mask")
        if raw in (None, False):
            return None
        return normalize_energy_mask_spec(raw)

    def _optimizer_gradient_mode(self) -> str:
        return str(
            self.config.training.extras.get("optimizer_gradient_mode", "masked")
        ).strip().lower()

    def _outside_aux_weight(self) -> float:
        return float(self.config.training.extras.get("outside_aux_weight", 1.0))

    def _allow_unmasked_optimizer_gradient(self) -> bool:
        return bool(
            self.config.training.extras.get("allow_unmasked_optimizer_gradient", False)
        )

    def _apply_rem_statistics_options(self, step: Dict[str, Any]) -> None:
        energy_mask = self._energy_mask_runtime_spec()
        if energy_mask is not None:
            step["energy_mask"] = energy_mask
        # Only request auxiliary unmasked statistics when the trainer may use
        # them for the optimizer step. The default masked mode keeps the
        # post-processing payload small.
        mode = self._optimizer_gradient_mode()
        if mode in {"hybrid_aux", "unmasked"}:
            step["need_aux_unmasked_energy_grad"] = True
