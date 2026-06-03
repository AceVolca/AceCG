"""Frozen dataclass models for AceCG configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .vp_config import VPConfig

if TYPE_CHECKING:
    from ..topology.types import InteractionKey


# ─── FM interaction spec ──────────────────────────────────────────────

@dataclass(frozen=True)
class FMInteractionSpec:
    """One force-matching interaction model specification.

    Attributes describe the interaction key, model family/size, coordinate
    domain, optional force cap, initialization mode, and model-specific
    overrides parsed from an ``.acg`` config.
    """
    style: str
    types: Tuple[str, ...]
    model: str
    model_size: int
    domain: Tuple[float, float]
    max_force: Optional[float] = None
    model_overrides: Dict[str, Any] = field(default_factory=dict)
    init_mode: str = "source_table_fit"
    resolution: Optional[float] = None

    @property
    def ikey(self) -> "InteractionKey":
        """Canonical ``InteractionKey`` for this spec.

        Types are already normalized at parse time, so direct construction
        is correct here.
        """
        from ..topology.types import InteractionKey

        return InteractionKey(style=self.style, types=self.types)

    def to_runtime_dict(self) -> Dict[str, Any]:
        """Convert this spec to the runtime dictionary consumed by workflows."""
        payload: Dict[str, Any] = {
            "style": self.style,
            "types": list(self.types),
            "model": self.model,
            "model_size": self.model_size,
            "min": self.domain[0],
            "max": self.domain[1],
            "init_mode": self.init_mode,
        }
        if self.max_force is not None:
            payload["max_force"] = self.max_force
        if self.model_overrides:
            payload["model_overrides"] = dict(self.model_overrides)
        if self.resolution is not None:
            payload["resolution"] = self.resolution
        return payload


@dataclass(frozen=True)
class FMTrainingSpecs:
    """Grouped force-matching specs for pair, bond, and angle interactions."""
    pair_specs: Tuple[FMInteractionSpec, ...] = ()
    bond_specs: Tuple[FMInteractionSpec, ...] = ()
    angle_specs: Tuple[FMInteractionSpec, ...] = ()

    def flattened(self) -> Tuple[FMInteractionSpec, ...]:
        """Return all FM specs in pair, bond, then angle order."""
        return self.pair_specs + self.bond_specs + self.angle_specs

    def to_runtime_dict(self) -> Dict[str, Any]:
        """Convert grouped specs to workflow runtime dictionaries."""
        return {
            "pair_specs": [s.to_runtime_dict() for s in self.pair_specs],
            "bond_specs": [s.to_runtime_dict() for s in self.bond_specs],
            "angle_specs": [s.to_runtime_dict() for s in self.angle_specs],
        }


@dataclass(frozen=True)
class ForcefieldMaskSpec:
    """Parsed forcefield mask entries keyed by interaction and optional style."""
    entries: Tuple[Tuple["InteractionKey", Optional[str], Tuple[str, ...]], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.entries)


@dataclass(frozen=True)
class ForcefieldBoundsSpec:
    """Parsed forcefield bounds entries keyed by interaction and optional style."""
    entries: Tuple[
        Tuple["InteractionKey", Optional[str], Tuple[str, ...], Tuple[str, ...]],
        ...,
    ] = ()

    def __bool__(self) -> bool:
        return bool(self.entries)


# ─── Section configs ──────────────────────────────────────────────────

@dataclass(frozen=True)
class SystemConfig:
    """System-level configuration (topology, force field, tables)."""

    topology_file: Optional[str] = None
    forcefield_path: Optional[str] = None
    fixed_forcefield_path: Optional[str] = None
    forcefield_mask_path: Optional[str] = None
    forcefield_mask: Optional[ForcefieldMaskSpec] = None
    forcefield_bounds_path: Optional[str] = None
    forcefield_bounds: Optional[ForcefieldBoundsSpec] = None
    forcefield_format: Optional[str] = None
    pair_style: Optional[str] = None
    cutoff: Optional[float] = None
    exclude: Optional[str] = None
    exclude_bonded: str = "111"
    exclude_option: str = "resid"
    type_names: Optional[Dict[int, str]] = None
    table_fit: Optional[str] = None
    table_fit_overrides: Optional[Dict[str, Any]] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    """Training-method, optimizer, solver, and output configuration."""
    method: str = ""
    para_path: Optional[str] = None
    fm_specs: FMTrainingSpecs = field(default_factory=FMTrainingSpecs)
    fm_method: str = "auto"
    solver_mode: str = "ols"
    solver_ridge_alpha: float = 0.0
    optimizer: Optional[str] = None
    trainer: Optional[str] = None
    lr: Optional[float] = None
    n_epochs: int = 1
    start_epoch: int = 0
    convergence_tol: float = 0.0
    output_dir: Optional[str] = None
    seed: int = 0
    temperature: Optional[float] = None
    need_hessian: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplingConfig:
    """MD sampling configuration."""

    sim_backend: str = "lammps"
    input: Optional[str] = None
    engine_command: Optional[str] = None
    init_config_pool: Optional[str] = None
    replay_mode: str = "off"
    ncores: Optional[int] = None
    perf_trace: bool = False
    sim_var: Dict[str, str] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationConfig:
    """Optional simulation-only forcefield validation configuration."""

    input: Optional[str] = None
    engine_command: Optional[str] = None
    sim_backend: str = "lammps"
    init_config_pool: Optional[str] = None
    ncores: Optional[int] = None
    num_epochs_per_validation: Optional[int] = None
    sim_var: Dict[str, str] = field(default_factory=dict)
    forcefield_template_path: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        """Return whether validation is configured to run."""
        return bool(self.input)


@dataclass(frozen=True)
class SchedulerConfig:
    """Task-scheduler and MPI launcher configuration."""
    mpirun_path: Optional[str] = None
    mpi_family: Optional[str] = None
    python_exe: str = "python"
    task_timeout: Optional[float] = None
    min_success_zbx: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AARefNoiseConfig:
    """Noisy AA-reference coordinate augmentation for FM, REM, and DSM."""

    enabled: bool = False
    samples_per_frame: int = 0
    sigma: float = 0.0
    sigma_final: Optional[float] = None
    schedule: str = "constant"
    update_interval: int = 1
    seed: int = 0
    distribution: str = "gaussian"
    selection: Any = "all"
    include_original: bool = False
    wrap: bool = False
    batch_size: Optional[int] = None
    subsample_per_epoch: int = 0
    cache_policy: str = "stage"
    force_mix_ratio: float = 0.0
    neighbor_mode: str = "shared"
    neighbor_skin: float = 0.0

    def to_runtime_dict(self) -> Dict[str, Any]:
        """Return the epoch-local ``run_post`` noise spec.

        Scheduling/cache-policy fields stay in the workflow layer; a post spec
        is valid only for one epoch and carries the already resolved sigma.
        """
        payload: Dict[str, Any] = {
            "samples_per_frame": self.samples_per_frame,
            "sigma": self.sigma,
            "seed": self.seed,
            "distribution": self.distribution,
            "selection": self.selection,
            "include_original": self.include_original,
            "wrap": self.wrap,
        }
        if self.batch_size is not None:
            payload["batch_size"] = self.batch_size
        if self.subsample_per_epoch > 0:
            payload["subsample_per_epoch"] = int(self.subsample_per_epoch)
        if self.force_mix_ratio > 0.0:
            payload["force_mix_ratio"] = float(self.force_mix_ratio)
        if self.neighbor_mode != "shared":
            payload["neighbor_mode"] = self.neighbor_mode
        if self.neighbor_skin > 0.0:
            payload["neighbor_skin"] = float(self.neighbor_skin)
        return payload


@dataclass(frozen=True)
class AARefConfig:
    """All-atom reference trajectory configuration."""
    trajectory_files: Tuple[str, ...] = ()
    trajectory_format: str = "LAMMPSDUMP"
    skip_frames: int = 0
    every: int = 1
    n_frames: int = 0
    all_atom_data_path: Optional[str] = None
    ref_topo: Optional[str] = None
    ref_has_vp: bool = True
    ref_type_names: Optional[Dict[str, str]] = None
    ref_type_map: Optional[Dict[str, str]] = None
    ref_resolved_aliases: Optional[Dict[int, str]] = None
    noise: AARefNoiseConfig = field(default_factory=AARefNoiseConfig)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditioningConfig:
    """Conditioned latent-sampling configuration for CDREM/CDFM workflows."""
    input: Optional[str] = None
    init_config_pool: Optional[str] = None
    init_force_pool: Optional[str] = None
    mask_cg_only: bool = True
    n_samples: int = 15
    ncores_per_task: Optional[int] = None
    post_n_ranks: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)


# ─── Top-level config ─────────────────────────────────────────────────

@dataclass(frozen=True)
class ACGConfig:
    """Top-level parsed AceCG configuration model."""
    path: Optional[Path] = None
    system: SystemConfig = field(default_factory=SystemConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    aa_ref: AARefConfig = field(default_factory=AARefConfig)
    vp: Optional[VPConfig] = None
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
