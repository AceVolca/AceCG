"""Parser and validator for ``.acg`` configuration files."""

from __future__ import annotations

import ast
import glob
import math
import types
import warnings
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

from .models import (
    ACGConfig,
    AARefConfig,
    AARefNoiseConfig,
    ConditioningConfig,
    FMInteractionSpec,
    FMTrainingSpecs,
    SamplingConfig,
    SchedulerConfig,
    SystemConfig,
    TrainingConfig,
    ValidationConfig,
)
from .utils import (
    extract_frame_id_from_data_file,
    parse_exclude_setting,
    parse_pair_style_options,
)
from ..io.forcefield import ReadLmpFFBounds, ReadLmpFFMask
from ..topology.topology_array import TopologyArrays, collect_topology_arrays
from ..topology.types import InteractionKey


class ACGConfigError(ValueError):
    """Raised when a ``.acg`` file cannot be parsed or validated."""


# ─── Parser constants ─────────────────────────────────────────────────

_SUPPORTED_METHODS = frozenset({"fm", "rem", "cdrem", "cdfm", "dsm"})
_SUPPORTED_FM_METHODS = frozenset({"auto", "solver", "iterator"})
_SUPPORTED_OPTIMIZER_TOKENS = frozenset(
    {
        "newton",
        "newtonraphson",
        "newton_raphson",
        "adam",
        "adammaskedoptimizer",
        "adamw",
        "adamwmaskedoptimizer",
        "rmsprop",
        "rmspropmaskedoptimizer",
    }
)
_RESERVED_SECTIONS = frozenset({"vp", "conditioning"})
_KNOWN_SECTIONS = (
    frozenset({"system", "training", "sampling", "validation", "scheduler", "aa_ref"})
    | _RESERVED_SECTIONS
)
_KNOWN_NESTED_SECTIONS = frozenset({"training.fm_specs"})

_SECTION_ALLOWED_KEYS: Dict[str, frozenset] = {
    "system": frozenset(
        {
            "topology_file",
            "forcefield_path",
            "fixed_forcefield_path",
            "forcefield_mask_path",
            "forcefield_bounds_path",
            "forcefield_format",
            "pair_style",
            "cutoff",
            "exclude",
            "type_names",
            "table_fit",
            "table_fit_overrides",
        }
    ),
    "training": frozenset(
        {
            "method",
            "para_path",
            "fm_method",
            "solver_mode",
            "solver_ridge_alpha",
            "optimizer",
            "trainer",
            "lr",
            "n_epochs",
            "start_epoch",
            "convergence_tol",
            "output_dir",
            "seed",
            "temperature",
            "beta",
            "cdfm_mode",
            "need_hessian",
            "energy_mask",
            "optimizer_gradient_mode",
            "outside_aux_weight",
            "allow_unmasked_optimizer_gradient",
            "boundary_prior",
            "boundary_prior_path",
            "boundary_prior_pair_decay",
            "boundary_prior_pair_strength",
            "boundary_prior_wall_k_min",
        }
    ),
    "sampling": frozenset(
        {
            "sim_backend",
            "input",
            "engine_command",
            "trajectory_format",
            "init_config_pool",
            "replay_mode",
            "ncores",
            "archive_trajectory",
            "perf_trace",
            "perf_trace_all_ranks",
            "heartbeat_interval",
            "sim_var",
        }
    ),
    "validation": frozenset(
        {
            "sim_backend",
            "input",
            "engine_command",
            "init_config_pool",
            "ncores",
            "num_epochs_per_validation",
            "sim_var",
            "forcefield_template_path",
            "replay_mode",
            "archive_trajectory",
        }
    ),
    "scheduler": frozenset(
        {
            "mpirun_path",
            "mpi_family",
            "python_exe",
            "task_timeout",
            "min_success_zbx",
            "extra_env",
        }
    ),
    "aa_ref": frozenset(
        {
            "trajectory_files",
            "trajectory_format",
            "skip_frames",
            "every",
            "n_frames",
            "all_atom_data_path",
            "ref_topo",
            "ref_has_vp",
            "ref_type_names",
            "ref_type_map",
            "noise_enabled",
            "noise_samples_per_frame",
            "noise_sigma",
            "noise_sigma_final",
            "noise_schedule",
            "noise_update_interval",
            "noise_seed",
            "noise_distribution",
            "noise_selection",
            "noise_include_original",
            "noise_wrap",
            "noise_batch_size",
            "noise_subsample_per_epoch",
            "noise_cache_policy",
            "noise_force_mix_ratio",
            "noise_neighbor_mode",
            "noise_neighbor_skin",
        }
    ),
    "conditioning": frozenset(
        {
            "input",
            "init_config_pool",
            "init_force_pool",
            "mask_cg_only",
            "n_samples",
            "ncores_per_task",
            "post_n_ranks",
        }
    ),
}
_FM_SPEC_ALLOWED_KEYS = frozenset({"pair_specs", "bond_specs", "angle_specs"})
_FM_SPEC_MODELS = frozenset({"bspline", "multigaussian"})
_FM_SPEC_STYLE_TO_KEY = {
    "pair": "pair_specs",
    "bond": "bond_specs",
    "angle": "angle_specs",
}
_AUTHORED_FM_EXPORT_RESOLUTION = {
    "pair": 0.1,
    "bond": 0.05,
    "angle": 0.5,
}

_BOLTZMANN_KCAL = 0.001987204  # kcal/(mol·K)


# ─── Public API ───────────────────────────────────────────────────────

def parse_acg_file(path: str | Path) -> ACGConfig:
    """Load one ``.acg`` file and build the validated config model."""
    config_path = Path(path).expanduser().resolve()
    raw = parse_acg_text(
        config_path.read_text(encoding="utf-8"), source=str(config_path)
    )
    return build_acg_config(raw, path=config_path)


def parse_acg_text(
    text: str,
    *,
    source: str = "<memory>",
    extra_sections: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Parse raw ``.acg`` text into section→key dictionaries.

    Parameters
    ----------
    text
        Full file content.
    source
        Label used in error messages; typically the file path.
    extra_sections
        Extra top-level section names to accept on top of the built-in
        ``_KNOWN_SECTIONS`` allowlist. Provided so standalone-workflow
        parsers (e.g. VP Growth) can reuse the tokenizer without forking.
    """
    current_section: Optional[str] = None
    sections: Dict[str, Dict[str, Any]] = {}
    lines = text.splitlines()
    index = 0
    known_sections = _KNOWN_SECTIONS
    if extra_sections:
        known_sections = known_sections | frozenset(extra_sections)

    while index < len(lines):
        raw_line = lines[index]
        line = _strip_inline_comment(raw_line).strip()
        index += 1
        if not line:
            continue

        section_name = _parse_section_header(line)
        if section_name is not None:
            if "." in section_name:
                if section_name in _KNOWN_NESTED_SECTIONS:
                    current_section = section_name
                    sections.setdefault(current_section, {})
                    continue
                raise ACGConfigError(
                    f"Nested section '{section_name}' is not supported in {source}. "
                    "Phase W uses a single-file flat section layout."
                )
            if section_name not in known_sections:
                raise ACGConfigError(
                    f"Unknown section '{section_name}' in {source}. "
                    f"Supported sections: {sorted(known_sections)}"
                )
            current_section = section_name
            sections.setdefault(current_section, {})
            continue

        if current_section is None:
            raise ACGConfigError(
                f"Key-value line appeared before any section header in {source}: "
                f"{raw_line!r}"
            )
        if "=" not in line:
            raise ACGConfigError(
                f"Expected 'key = value' inside section [{current_section}] in "
                f"{source}: {raw_line!r}"
            )
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ACGConfigError(
                f"Empty key inside section [{current_section}] in {source}"
            )

        value_lines = [value.strip()]
        while _value_needs_more_input("\n".join(value_lines)):
            if index >= len(lines):
                raise ACGConfigError(
                    f"Unterminated multi-line value for '{key}' in section "
                    f"[{current_section}] in {source}"
                )
            continuation = _strip_inline_comment(lines[index]).rstrip()
            value_lines.append(continuation)
            index += 1

        sections[current_section][key] = _parse_scalar_or_literal(
            "\n".join(value_lines).strip()
        )

    return sections


def build_acg_config(
    raw: Mapping[str, Mapping[str, Any]], *, path: Path
) -> ACGConfig:
    """Build the normalized ``ACGConfig`` model from parsed section data."""
    system_raw = dict(raw.get("system", {}))
    training_raw = dict(raw.get("training", {}))
    sampling_raw = dict(raw.get("sampling", {}))
    validation_raw = dict(raw.get("validation", {}))
    scheduler_raw = dict(raw.get("scheduler", {}))
    aa_ref_raw = dict(raw.get("aa_ref", {}))
    fm_specs_raw = dict(raw.get("training.fm_specs", {}))
    vp_raw = dict(raw.get("vp", {}))
    conditioning_raw = dict(raw.get("conditioning", {}))

    if "n_iterations" in training_raw:
        raise ACGConfigError(
            "Use training.n_epochs instead of training.n_iterations."
        )

    _warn_unknown_keys("system", system_raw)
    _warn_unknown_keys("training", training_raw)
    _warn_unknown_keys("sampling", sampling_raw)
    _warn_unknown_keys("validation", validation_raw)
    _warn_unknown_keys("scheduler", scheduler_raw)
    _warn_unknown_keys("aa_ref", aa_ref_raw)
    if conditioning_raw:
        _warn_unknown_keys("conditioning", conditioning_raw)

    method = str(training_raw.get("method", "")).strip().lower()
    if method not in _SUPPORTED_METHODS:
        raise ACGConfigError(
            f"training.method must be one of {sorted(_SUPPORTED_METHODS)}, "
            f"got {method!r}"
        )

    fm_method = str(training_raw.pop("fm_method", "auto")).strip().lower()
    if fm_method not in _SUPPORTED_FM_METHODS:
        raise ACGConfigError(
            f"training.fm_method must be one of {sorted(_SUPPORTED_FM_METHODS)}, "
            f"got {fm_method!r}"
        )

    optimizer = _pop_optional_str(training_raw, "optimizer")
    if optimizer is not None:
        optimizer_token = optimizer.split()[0].lower()
        if optimizer_token not in _SUPPORTED_OPTIMIZER_TOKENS:
            raise ACGConfigError(
                "training.optimizer must start with one of "
                f"{sorted(_SUPPORTED_OPTIMIZER_TOKENS)}, got {optimizer!r}"
            )

    # temperature / beta handling
    temperature = _pop_optional_float(training_raw, "temperature")
    beta_raw = _pop_optional_float(training_raw, "beta")
    if temperature is not None and beta_raw is not None:
        raise ACGConfigError(
            "Set either training.temperature or training.beta, not both."
        )
    if beta_raw is not None:
        if beta_raw <= 0:
            raise ACGConfigError("training.beta must be positive.")
        temperature = 1.0 / (_BOLTZMANN_KCAL * beta_raw)

    fm_specs = _build_fm_training_specs(fm_specs_raw, method=method)

    exclude_raw = _pop_optional_str(system_raw, "exclude")
    exclude_bonded, exclude_option = parse_exclude_setting(exclude_raw)

    type_names_raw = system_raw.pop("type_names", None)
    type_names = _parse_type_names(type_names_raw, base_dir=path.parent)

    topology_file = _pop_optional_str(system_raw, "topology_file")
    forcefield_path = _pop_optional_str(system_raw, "forcefield_path")
    fixed_forcefield_path = _pop_optional_str(system_raw, "fixed_forcefield_path")
    forcefield_mask_path = _pop_optional_str(system_raw, "forcefield_mask_path")
    forcefield_bounds_path = _pop_optional_str(system_raw, "forcefield_bounds_path")
    forcefield_format = _pop_optional_str(system_raw, "forcefield_format")
    pair_style = _pop_optional_str(system_raw, "pair_style")
    forcefield_spec_topology = None
    if forcefield_path or forcefield_mask_path or forcefield_bounds_path:
        forcefield_spec_topology = _collect_system_topology_arrays(
            config_path=path,
            topology_file=topology_file,
            exclude_bonded=exclude_bonded,
            exclude_option=exclude_option,
            type_names=type_names,
            vp_raw=vp_raw,
        )

    forcefield_mask = None
    mask_source = forcefield_mask_path or forcefield_path
    if mask_source is not None:
        if pair_style is None:
            raise ACGConfigError(
                "system.pair_style is required to parse the forcefield mask."
            )
        pair_kind, sel_styles = parse_pair_style_options(pair_style)
        mask_path = Path(mask_source)
        if not mask_path.is_absolute():
            mask_path = (path.parent / mask_path).resolve()
        try:
            forcefield_mask = ReadLmpFFMask(
                str(mask_path),
                pair_kind,
                pair_typ_sel=sel_styles,
                topology_arrays=forcefield_spec_topology,
            )
        except Exception as exc:
            raise ACGConfigError(
                f"Failed to parse forcefield mask from {mask_path}: {exc}"
            ) from exc

    forcefield_bounds = None
    if forcefield_bounds_path is not None:
        if pair_style is None:
            raise ACGConfigError(
                "system.pair_style is required to parse the forcefield bounds."
            )
        pair_kind, sel_styles = parse_pair_style_options(pair_style)
        bounds_path = Path(forcefield_bounds_path)
        if not bounds_path.is_absolute():
            bounds_path = (path.parent / bounds_path).resolve()
        try:
            forcefield_bounds = ReadLmpFFBounds(
                str(bounds_path),
                pair_kind,
                pair_typ_sel=sel_styles,
                topology_arrays=forcefield_spec_topology,
            )
        except Exception as exc:
            raise ACGConfigError(
                f"Failed to parse forcefield bounds from {bounds_path}: {exc}"
            ) from exc

    system = SystemConfig(
        topology_file=topology_file,
        forcefield_path=forcefield_path,
        fixed_forcefield_path=fixed_forcefield_path,
        forcefield_mask_path=forcefield_mask_path,
        forcefield_mask=forcefield_mask,
        forcefield_bounds_path=forcefield_bounds_path,
        forcefield_bounds=forcefield_bounds,
        forcefield_format=forcefield_format,
        pair_style=pair_style,
        cutoff=_pop_optional_float(system_raw, "cutoff"),
        exclude=exclude_raw,
        exclude_bonded=exclude_bonded,
        exclude_option=exclude_option,
        type_names=type_names,
        table_fit=_pop_optional_str(system_raw, "table_fit"),
        table_fit_overrides=_pop_optional_mapping(system_raw, "table_fit_overrides"),
        extras=system_raw,
    )

    solver_mode = str(training_raw.pop("solver_mode", "ols")).strip().lower()
    if solver_mode not in {"ols", "ridge", "bayesian"}:
        raise ACGConfigError(
            f"training.solver_mode must be 'ols', 'ridge', or 'bayesian', "
            f"got {solver_mode!r}"
        )
    solver_ridge_alpha = _pop_optional_float(
        training_raw, "solver_ridge_alpha", default=0.0
    )

    training = TrainingConfig(
        method=method,
        para_path=_pop_optional_str(training_raw, "para_path"),
        fm_specs=fm_specs,
        fm_method=fm_method,
        solver_mode=solver_mode,
        solver_ridge_alpha=solver_ridge_alpha,
        optimizer=optimizer,
        trainer=_pop_optional_str(training_raw, "trainer"),
        lr=_pop_optional_float(training_raw, "lr"),
        n_epochs=_pop_optional_int(training_raw, "n_epochs", default=1),
        start_epoch=_pop_optional_int(training_raw, "start_epoch", default=0),
        convergence_tol=_pop_optional_float(
            training_raw, "convergence_tol", default=0.0
        ),
        output_dir=_pop_optional_str(training_raw, "output_dir"),
        seed=_pop_optional_int(training_raw, "seed", default=0),
        temperature=temperature,
        need_hessian=_pop_optional_bool(training_raw, "need_hessian", default=False),
        extras=training_raw,
    )

    sampling = SamplingConfig(
        sim_backend=str(
            sampling_raw.pop("sim_backend", "lammps")
        ).strip().lower(),
        input=_pop_optional_str(sampling_raw, "input"),
        engine_command=_pop_optional_str(sampling_raw, "engine_command"),
        trajectory_format=_normalize_trajectory_format(
            sampling_raw.pop("trajectory_format", None)
        ),
        init_config_pool=_pop_optional_str(sampling_raw, "init_config_pool"),
        replay_mode=str(
            sampling_raw.pop("replay_mode", "off")
        ).strip().lower(),
        ncores=_pop_optional_int(sampling_raw, "ncores"),
        archive_trajectory=_pop_optional_bool(
            sampling_raw, "archive_trajectory", default=False
        ),
        perf_trace=_pop_optional_bool(sampling_raw, "perf_trace", default=False),
        sim_var=_pop_sim_var(sampling_raw),
        extras=sampling_raw,
    )

    validation = ValidationConfig(
        sim_backend=str(
            validation_raw.pop("sim_backend", "lammps")
        ).strip().lower(),
        input=_pop_optional_str(validation_raw, "input"),
        engine_command=_pop_optional_str(validation_raw, "engine_command"),
        init_config_pool=_pop_optional_str(validation_raw, "init_config_pool"),
        ncores=_pop_optional_int(validation_raw, "ncores"),
        num_epochs_per_validation=_pop_optional_int(
            validation_raw, "num_epochs_per_validation"
        ),
        sim_var=_pop_sim_var(validation_raw),
        forcefield_template_path=_pop_optional_str(
            validation_raw, "forcefield_template_path"
        ),
        extras=validation_raw,
    )

    _mpirun_path_raw = scheduler_raw.pop("mpirun_path", None)
    _mpi_family_raw = scheduler_raw.pop("mpi_family", None)
    scheduler = SchedulerConfig(
        mpirun_path=str(_mpirun_path_raw).strip() if _mpirun_path_raw is not None else None,
        mpi_family=(str(_mpi_family_raw).strip() or None) if _mpi_family_raw is not None else None,
        python_exe=str(
            scheduler_raw.pop("python_exe", "python")
        ).strip(),
        task_timeout=_pop_optional_float(scheduler_raw, "task_timeout"),
        min_success_zbx=_pop_optional_int(scheduler_raw, "min_success_zbx"),
        extras=scheduler_raw,
    )

    trajectory_files = aa_ref_raw.pop("trajectory_files", ())
    if isinstance(trajectory_files, str):
        trajectory_files = (trajectory_files,)
    elif isinstance(trajectory_files, Iterable):
        trajectory_files = tuple(str(item) for item in trajectory_files)
    else:
        raise ACGConfigError(
            "aa_ref.trajectory_files must be a string or a list of strings"
        )

    ref_type_names = _pop_dict_or_file(
        aa_ref_raw, "ref_type_names", base_dir=path.parent,
    )
    ref_type_map = _pop_dict_or_file(
        aa_ref_raw, "ref_type_map", base_dir=path.parent,
    )
    ref_has_vp = _pop_optional_bool(aa_ref_raw, "ref_has_vp", default=True)
    ref_topo = _pop_optional_str(aa_ref_raw, "ref_topo")
    noise = _pop_aa_ref_noise_config(aa_ref_raw)

    # Resolve atom_type_name_aliases for the AA-ref engine spec.
    #   Priority: (1) explicit ref_type_map  (2) dual type_names  (3) fallback
    ref_resolved_aliases = None if method == "cdfm" else _resolve_aa_ref_aliases(
        sys_type_names=type_names,
        ref_type_names=ref_type_names,
        ref_type_map=ref_type_map,
        ref_topo=ref_topo,
        system_topo=topology_file,
    )

    aa_ref = AARefConfig(
        trajectory_files=trajectory_files,
        trajectory_format=_normalize_trajectory_format(
            aa_ref_raw.pop("trajectory_format", "LAMMPSDUMP")
        ) or "LAMMPSDUMP",
        skip_frames=_pop_optional_int(aa_ref_raw, "skip_frames", default=0),
        every=_pop_optional_int(aa_ref_raw, "every", default=1),
        n_frames=_pop_optional_int(aa_ref_raw, "n_frames", default=0),
        all_atom_data_path=_pop_optional_str(aa_ref_raw, "all_atom_data_path"),
        ref_topo=ref_topo,
        ref_has_vp=ref_has_vp,
        ref_type_names=ref_type_names,
        ref_type_map=ref_type_map,
        ref_resolved_aliases=ref_resolved_aliases,
        noise=noise,
        extras=aa_ref_raw,
    )

    conditioning = ConditioningConfig(
        input=_pop_optional_str(conditioning_raw, "input"),
        init_config_pool=_pop_optional_str(
            conditioning_raw, "init_config_pool"
        ),
        init_force_pool=_pop_optional_str(
            conditioning_raw, "init_force_pool"
        ),
        mask_cg_only=_pop_optional_bool(
            conditioning_raw, "mask_cg_only", default=True
        ),
        n_samples=_pop_optional_int(
            conditioning_raw, "n_samples", default=15
        ),
        ncores_per_task=_pop_optional_int(
            conditioning_raw, "ncores_per_task"
        ),
        post_n_ranks=_pop_optional_int(
            conditioning_raw, "post_n_ranks"
        ),
        extras=conditioning_raw,
    )

    config = ACGConfig(
        path=path,
        system=system,
        training=training,
        sampling=sampling,
        validation=validation,
        scheduler=scheduler,
        aa_ref=aa_ref,
        vp=types.SimpleNamespace(**vp_raw) if vp_raw else None,
        conditioning=conditioning,
    )
    _validate_config(config)
    return config


def _collect_system_topology_arrays(
    *,
    config_path: Path,
    topology_file: Optional[str],
    exclude_bonded: Tuple[str, str, str],
    exclude_option: str,
    type_names: Optional[Mapping[str, str]],
    vp_raw: Mapping[str, Any],
) -> Optional[TopologyArrays]:
    if not topology_file:
        return None
    topology_path = Path(topology_file).expanduser()
    if not topology_path.is_absolute():
        topology_path = config_path.parent / topology_path
    if not topology_path.exists():
        return None

    raw_vp_names = vp_raw.get("vp_names") if vp_raw else None
    if raw_vp_names is None:
        vp_names = None
    elif isinstance(raw_vp_names, str):
        vp_names = (raw_vp_names,)
    elif isinstance(raw_vp_names, Iterable):
        vp_names = tuple(str(item) for item in raw_vp_names)
    else:
        vp_names = None

    import MDAnalysis as mda

    universe = mda.Universe(str(topology_path))
    return collect_topology_arrays(
        universe,
        exclude_bonded=exclude_bonded,
        exclude_option=exclude_option,
        atom_type_name_aliases=type_names,
        vp_names=vp_names,
    )


# ─── Validation ───────────────────────────────────────────────────────

def _validate_config(config: ACGConfig) -> None:  # noqa: C901
    method = config.training.method
    interactions = config.training.fm_specs.flattened()
    source_free_fm = (
        method in {"fm", "dsm"}
        and len(interactions) > 0
        and all(
            spec.init_mode in {"authored_zero", "authored_direct"}
            for spec in interactions
        )
    )

    # System
    if not config.system.topology_file:
        raise ACGConfigError("system.topology_file is required.")
    if config.system.cutoff is not None and config.system.cutoff <= 0:
        raise ACGConfigError("system.cutoff must be positive.")
    if (not source_free_fm) and not (config.system.forcefield_path or config.training.para_path):
        raise ACGConfigError(
            "Define either system.forcefield_path or training.para_path so "
            "AceCG can load real.settings, or use authored FM specs only."
        )
    if config.system.fixed_forcefield_path and not config.system.pair_style:
        raise ACGConfigError(
            "system.pair_style is required when system.fixed_forcefield_path is set."
        )

    # Training
    if not config.training.output_dir:
        raise ACGConfigError("training.output_dir is required.")
    if config.training.lr is not None and config.training.lr <= 0:
        raise ACGConfigError("training.lr must be positive.")
    if config.training.convergence_tol < 0:
        raise ACGConfigError("training.convergence_tol must be non-negative.")
    if config.training.n_epochs <= 0:
        raise ACGConfigError("training.n_epochs must be positive.")

    # temperature / beta for statistical objectives
    if method in ("rem", "cdrem", "cdfm", "dsm"):
        if config.training.temperature is None:
            raise ACGConfigError(
                "training.temperature (or training.beta) is required for "
                f"method={method!r}."
            )
        if config.training.temperature <= 0:
            raise ACGConfigError("training.temperature must be positive.")

    # FM specs
    if method in {"fm", "dsm"} and not interactions:
        raise ACGConfigError(
            "training.fm_specs must define at least one trainable interaction "
            f"for {method.upper()}."
        )
    if method == "dsm" and config.training.fm_method == "solver":
        raise ACGConfigError("DSM is iterative and does not support training.fm_method='solver'.")

    # AA ref
    if config.aa_ref.skip_frames < 0:
        raise ACGConfigError("aa_ref.skip_frames must be non-negative.")
    if config.aa_ref.every <= 0:
        raise ACGConfigError("aa_ref.every must be positive.")
    if config.aa_ref.n_frames < 0:
        raise ACGConfigError("aa_ref.n_frames must be non-negative.")
    noise = config.aa_ref.noise
    if noise.schedule not in {"constant", "cosine", "exponential"}:
        raise ACGConfigError(
            "aa_ref.noise_schedule must be one of 'constant', 'cosine', or 'exponential'."
        )
    if noise.distribution not in {"gaussian", "normal"}:
        raise ACGConfigError(
            "aa_ref.noise_distribution must be 'gaussian' or 'normal'."
        )
    if noise.cache_policy not in {"stage", "none"}:
        raise ACGConfigError("aa_ref.noise_cache_policy must be 'stage' or 'none'.")
    if noise.neighbor_mode not in {"shared", "skin", "chunk"}:
        raise ACGConfigError(
            "aa_ref.noise_neighbor_mode must be one of 'shared', 'skin', or 'chunk'."
        )
    if noise.neighbor_skin < 0.0:
        raise ACGConfigError("aa_ref.noise_neighbor_skin must be non-negative.")
    if noise.update_interval <= 0:
        raise ACGConfigError("aa_ref.noise_update_interval must be positive.")
    if noise.samples_per_frame < 0:
        raise ACGConfigError("aa_ref.noise_samples_per_frame must be non-negative.")
    if noise.sigma < 0.0:
        raise ACGConfigError("aa_ref.noise_sigma must be non-negative.")
    if noise.sigma_final is not None and noise.sigma_final < 0.0:
        raise ACGConfigError("aa_ref.noise_sigma_final must be non-negative.")
    if noise.batch_size is not None and noise.batch_size <= 0:
        raise ACGConfigError("aa_ref.noise_batch_size must be positive when set.")
    if noise.subsample_per_epoch < 0:
        raise ACGConfigError("aa_ref.noise_subsample_per_epoch must be non-negative.")
    if not 0.0 <= noise.force_mix_ratio <= 1.0:
        raise ACGConfigError("aa_ref.noise_force_mix_ratio must be between 0 and 1.")
    if noise.force_mix_ratio > 0.0:
        if method != "fm":
            raise ACGConfigError(
                "aa_ref.noise_force_mix_ratio is supported only for method='fm'."
            )
        if not noise.enabled:
            raise ACGConfigError(
                "aa_ref.noise_force_mix_ratio requires aa_ref.noise_enabled=true."
            )
        if noise.samples_per_frame <= 0:
            raise ACGConfigError(
                "aa_ref.noise_force_mix_ratio requires noise_samples_per_frame > 0."
            )
        if noise.sigma <= 0.0:
            raise ACGConfigError(
                "aa_ref.noise_force_mix_ratio requires aa_ref.noise_sigma > 0."
            )
        if config.training.temperature is None or config.training.temperature <= 0.0:
            raise ACGConfigError(
                "aa_ref.noise_force_mix_ratio requires training.temperature "
                "(or training.beta)."
            )
    if noise.enabled:
        if method not in {"fm", "rem", "dsm"}:
            raise ACGConfigError(
                "aa_ref.noise_enabled is supported only for method='fm' or "
                "method='rem' or method='dsm'. Use restrained sampling inputs "
                "for CDREM/CDFM coordinate perturbations."
            )
        if noise.samples_per_frame == 0 and not noise.include_original:
            raise ACGConfigError(
                "aa_ref.noise_enabled requires noise_samples_per_frame > 0 "
                "unless noise_include_original=true."
            )
        if method == "rem" and not config.aa_ref.trajectory_files:
            raise ACGConfigError(
                "aa_ref.noise_enabled requires aa_ref.trajectory_files so noisy "
                "AA statistics can be computed from coordinates."
            )
    if method == "dsm":
        if not noise.enabled:
            raise ACGConfigError("DSM requires aa_ref.noise_enabled=true.")
        if noise.sigma <= 0.0:
            raise ACGConfigError("DSM requires aa_ref.noise_sigma > 0.")
        if noise.sigma_final is not None and noise.sigma_final <= 0.0:
            raise ACGConfigError("DSM requires aa_ref.noise_sigma_final > 0 when set.")
        if not config.aa_ref.trajectory_files:
            raise ACGConfigError(
                "aa_ref.trajectory_files must contain at least one trajectory "
                "path for DSM."
            )
    if method == "fm" and not config.aa_ref.trajectory_files:
        raise ACGConfigError(
            "aa_ref.trajectory_files must contain at least one trajectory "
            "path for FM."
        )
    force_free_format = _known_force_free_trajectory_format(
        config.aa_ref.trajectory_format
    )
    if method == "fm" and force_free_format is not None:
        raise ACGConfigError(
            f"FM does not support aa_ref.trajectory_format={force_free_format!r} "
            "because that format does not carry reference forces. Use a "
            "force-bearing format such as a LAMMPS dump with fx/fy/fz columns."
        )

    if method in ("rem", "cdrem"):
        has_traj = bool(config.aa_ref.trajectory_files)
        has_cache = bool(config.aa_ref.all_atom_data_path)
        if not has_traj and not has_cache:
            raise ACGConfigError(
                "For method={!r}, at least one of aa_ref.trajectory_files "
                "or aa_ref.all_atom_data_path must be provided.".format(method)
            )

    if method == "cdfm" and config.aa_ref.trajectory_files:
        warnings.warn(
            "aa_ref.trajectory_files is set but method=cdfm no longer consumes "
            "the AA reference trajectory for y_eff computation. The field is "
            "ignored by the CDFM training loop; it may still be used by "
            "offline analysis tooling.",
            stacklevel=2,
        )

    # Sampling — xz simulation required for rem/cdrem only (not cdfm)
    if method in ("rem", "cdrem"):
        if not config.sampling.input:
            raise ACGConfigError(
                f"sampling.input is required for method={method!r}."
            )
        if not config.sampling.engine_command:
            raise ACGConfigError(
                f"sampling.engine_command is required for method={method!r}."
            )
    if config.sampling.trajectory_format is not None and not config.sampling.trajectory_format:
        raise ACGConfigError("sampling.trajectory_format must be non-empty when set.")
    if config.sampling.replay_mode not in {"off", "latest", "random"}:
        raise ACGConfigError(
            "sampling.replay_mode must be one of 'off', 'latest', or "
            f"'random', got {config.sampling.replay_mode!r}."
        )
    if (
        config.sampling.init_config_pool is not None
        and config.sampling.replay_mode != "off"
    ):
        raise ACGConfigError(
            "sampling.init_config_pool and sampling.replay_mode are mutually "
            "exclusive. Remove sampling.init_config_pool when replaying from "
            "prior xz checkpoints, or set sampling.replay_mode = off to draw "
            "from the configured init pool."
        )

    # Optional validation is simulation-only: no replay and no trajectory cleanup.
    validation = config.validation
    if "replay_mode" in validation.extras:
        raise ACGConfigError(
            "validation simulations are not replayable; remove validation.replay_mode."
        )
    if "archive_trajectory" in validation.extras:
        raise ACGConfigError(
            "validation trajectories are always archived; remove validation.archive_trajectory."
        )
    if validation.enabled:
        if validation.sim_backend != "lammps":
            raise ACGConfigError(
                "validation.sim_backend must be 'lammps'."
            )
        if not validation.engine_command:
            raise ACGConfigError(
                "validation.engine_command is required when validation.input is set."
            )
        if validation.ncores is not None and validation.ncores <= 0:
            raise ACGConfigError("validation.ncores must be positive when set.")
        if (
            validation.num_epochs_per_validation is not None
            and validation.num_epochs_per_validation <= 0
        ):
            raise ACGConfigError(
                "validation.num_epochs_per_validation must be positive when set."
            )
        if not (config.system.forcefield_path or validation.forcefield_template_path):
            raise ACGConfigError(
                "validation.forcefield_template_path is required when validation "
                "is enabled and system.forcefield_path is absent. Validation "
                "needs a LAMMPS-compatible forcefield template for WriteLmpFF."
            )
        if not config.system.pair_style:
            raise ACGConfigError(
                "system.pair_style is required when validation is enabled."
            )
        if config.scheduler.task_timeout is None or config.scheduler.task_timeout <= 0:
            raise ACGConfigError(
                "scheduler.task_timeout is required when validation is enabled."
            )

    # Conditioning (cdrem/cdfm only)
    if method in ("cdrem", "cdfm"):
        if not config.conditioning.input:
            raise ACGConfigError(
                f"conditioning.input is required for method={method!r}."
            )
        if not config.conditioning.init_config_pool:
            raise ACGConfigError(
                f"conditioning.init_config_pool is required for method={method!r}."
            )
        if method == "cdfm" and not config.conditioning.init_force_pool:
            raise ACGConfigError(
                "conditioning.init_force_pool is required for method='cdfm': "
                "CDFM reads one reference-force .npy per init-config frame and "
                "no longer consumes aa_ref.trajectory_files."
            )
        if config.conditioning.ncores_per_task is None and config.sampling.ncores is None:
            raise ACGConfigError(
                f"Either conditioning.ncores_per_task or sampling.ncores is "
                f"required for method={method!r}."
            )
        if not config.vp:
            raise ACGConfigError(
                f"[vp] section is required for method={method!r}."
            )
        _validate_conditioning_init_config_pool(config)

    # Conditioning forbidden for pure REM/FM
    if method in ("fm", "rem", "dsm") and config.conditioning.input is not None:
        raise ACGConfigError(
            f"[conditioning] section is not used by method={method!r}."
        )


# ─── Text-level parser helpers ────────────────────────────────────────

def _parse_section_header(line: str) -> Optional[str]:
    if line.startswith("[") and line.endswith("]"):
        return line[1:-1].strip().lower()
    if line.startswith("<") and line.endswith(">"):
        return line[1:-1].strip().lower()
    return None


def _normalize_trajectory_format(value: Any) -> Optional[str]:
    """Normalize common trajectory format aliases for MDAnalysis."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "auto":
        return None
    aliases = {
        "lammpstrj": "LAMMPSDUMP",
        "lammpsdump": "LAMMPSDUMP",
        "lammps_dump": "LAMMPSDUMP",
        "dump": "LAMMPSDUMP",
        "xtc": "XTC",
        "dcd": "DCD",
        "h5md": "H5MD",
        "trr": "TRR",
        "xyz": "XYZ",
    }
    lowered = text.lower()
    if lowered in aliases:
        return aliases[lowered]
    if all(char.isalnum() or char in {"_", "-"} for char in text):
        return text.upper().replace("-", "_")
    return text


def _known_force_free_trajectory_format(value: Any) -> Optional[str]:
    normalized = _normalize_trajectory_format(value)
    if normalized is None:
        return None
    token = normalized.upper()
    if token in {"XTC", "DCD"}:
        return token
    return None


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    out = []
    for char in line:
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\":
            out.append(char)
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            out.append(char)
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            out.append(char)
            continue
        if char == "#" and not in_single and not in_double:
            break
        out.append(char)
    return "".join(out)


def _value_needs_more_input(value: str) -> bool:
    stack = []
    in_single = False
    in_double = False
    escaped = False
    pairs = {"[": "]", "{": "}", "(": ")"}
    closing = {"]", "}", ")"}
    for char in value:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if char in pairs:
            stack.append(pairs[char])
        elif char in closing:
            if not stack or stack.pop() != char:
                return False
    return in_single or in_double or bool(stack)


def _parse_scalar_or_literal(value: str) -> Any:
    text = value.strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        if any(text.startswith(prefix) for prefix in ('"', "'", "[", "{", "(")):
            return ast.literal_eval(text)
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except Exception:
        return text


# ─── Section-level helpers ────────────────────────────────────────────

def _warn_unknown_section_keys(
    section_label: str,
    allowed_keys: frozenset,
    mapping: Mapping[str, Any],
    *,
    preserved: str,
) -> None:
    """Warn, with a close-match hint, for each key not in ``allowed_keys``."""
    for key in mapping:
        if key in allowed_keys:
            continue
        hint = get_close_matches(key, sorted(allowed_keys), n=1, cutoff=0.6)
        hint_text = f" Did you mean '{hint[0]}'?" if hint else ""
        warnings.warn(
            f"Unknown key '{key}' in section [{section_label}]; {preserved}.{hint_text}",
            UserWarning,
            stacklevel=4,
        )


def _warn_unknown_keys(section_name: str, mapping: Mapping[str, Any]) -> None:
    _warn_unknown_section_keys(
        section_name,
        _SECTION_ALLOWED_KEYS[section_name],
        mapping,
        preserved="preserving it in `.extras`",
    )


def _validate_conditioning_init_config_pool(config: ACGConfig) -> None:
    base_dir = config.path.parent if config.path is not None else Path.cwd().resolve()
    raw_pattern = Path(config.conditioning.init_config_pool).expanduser()
    query = str(raw_pattern if raw_pattern.is_absolute() else base_dir / raw_pattern)
    matches = [Path(item).resolve(strict=False) for item in sorted(glob.glob(query))]
    if not matches:
        raise ACGConfigError(
            "conditioning.init_config_pool matched no files: "
            f"{config.conditioning.init_config_pool!r}"
        )

    parent_dirs = {path.parent for path in matches}
    if len(parent_dirs) != 1:
        rendered = ", ".join(str(path) for path in sorted(parent_dirs))
        raise ACGConfigError(
            "conditioning.init_config_pool must resolve to a flat directory of "
            f"frame_<integer>.data files, but matched multiple parent directories: {rendered}"
        )

    seen_frame_ids: dict[int, Path] = {}
    for path in matches:
        try:
            frame_id = extract_frame_id_from_data_file(path)
        except ValueError as exc:
            raise ACGConfigError(
                "conditioning.init_config_pool entries must be named "
                f"frame_<integer>.data; got {path.name!r}"
            ) from exc
        prev = seen_frame_ids.get(frame_id)
        if prev is not None:
            raise ACGConfigError(
                "conditioning.init_config_pool contains duplicate frame ids: "
                f"{prev.name!r} and {path.name!r} both map to frame_id={frame_id}"
            )
        seen_frame_ids[frame_id] = path


def _parse_type_names(
    raw: Any, *, base_dir: Path, field_label: str = "system.type_names"
) -> Optional[Dict[int, str]]:
    """Parse a type-names field into ``{int_id: str_name}`` or ``None``.

    Accepted formats:
      - ``None`` → ``None``
      - dict ``{1: "HG", 2: "MG"}`` → pass through with int keys
      - comma string ``"HG, MG, TL"`` → ``{1: "HG", 2: "MG", 3: "TL"}``
      - file path (relative to *base_dir*) → loaded via ``_load_dict_file``

    *field_label* names the field in error messages so callers (e.g. the VP
    Growth ``[aa_ref]`` section) get section-specific diagnostics.
    """
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return {int(k): str(v) for k, v in raw.items()}
    if not isinstance(raw, str):
        raise ACGConfigError(
            f"{field_label} must be a dict, comma-separated string, "
            f"or file path; got {type(raw).__name__}."
        )
    text = raw.strip()
    if not text:
        return None
    # Try as file path (relative to the config directory)
    result = _load_dict_file(text, base_dir=base_dir)
    if result is not None:
        return {int(k): str(v) for k, v in result.items()}
    # Comma-separated list: "HG, MG, TL" → {1: "HG", 2: "MG", 3: "TL"}
    names = [n.strip() for n in text.split(",") if n.strip()]
    if names:
        return {i + 1: name for i, name in enumerate(names)}
    return None


def _load_dict_file(
    text: str, *, base_dir: Path
) -> Optional[Dict[str, Any]]:
    """Try to load *text* as a JSON file path relative to *base_dir*.

    Returns the parsed dict if the file exists and parses as JSON/literal,
    ``None`` if the file does not exist (caller may try other formats).
    """
    import json as _json

    candidate = (base_dir / text).resolve()
    if not candidate.is_file():
        return None
    raw_text = candidate.read_text(encoding="utf-8").strip()
    if not raw_text:
        return None
    try:
        result = _json.loads(raw_text)
    except _json.JSONDecodeError:
        try:
            result = ast.literal_eval(raw_text)
        except Exception as exc:
            raise ACGConfigError(
                f"Cannot parse dict file {candidate}: {exc}"
            ) from exc
    if not isinstance(result, dict):
        raise ACGConfigError(
            f"Expected a dict in {candidate}, got {type(result).__name__}."
        )
    return result


def _pop_dict_or_file(
    mapping: MutableMapping[str, Any],
    key: str,
    *,
    base_dir: Path,
) -> Optional[Dict[str, str]]:
    """Pop *key* from *mapping* as a ``{str: str}`` dict.

    Accepts:
      - ``None`` (absent) → ``None``
      - an inline dict ``{"1": "VP", ...}`` → coerce keys & values to str
      - a string file path (relative to *base_dir*) → load JSON/literal
    """
    raw = mapping.pop(key, None)
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        loaded = _load_dict_file(text, base_dir=base_dir)
        if loaded is not None:
            return {str(k): str(v) for k, v in loaded.items()}
        raise ACGConfigError(
            f"'{key}' value '{text}' is not a valid dict literal or an "
            f"existing file path (resolved against {base_dir})."
        )
    raise ACGConfigError(
        f"'{key}' must be a dict or a file path; got {type(raw).__name__}."
    )


def _resolve_aa_ref_aliases(
    *,
    sys_type_names: Optional[Dict[int, str]],
    ref_type_names: Optional[Dict[str, str]],
    ref_type_map: Optional[Dict[str, str]],
    ref_topo: Optional[str],
    system_topo: Optional[str],
) -> Optional[Dict[int, str]]:
    """Compute the ``atom_type_name_aliases`` dict for AA-ref engine specs.

    Priority chain:
      1. ``ref_type_map`` provided → compose ref_code → sys_code → sys_name
      2. Both ``sys_type_names`` and ``ref_type_names`` defined → use ref_type_names
      3. Neither aliases, no map → ``None``; warn if ref_topo differs from system_topo
      4. Only one side has aliases, no map → warn; return ref_type_names or ``None``

    Returns ``{int_code: str_name}`` ready for ``collect_topology_arrays``,
    or ``None`` to keep the default (raw LAMMPS codes as names).
    """
    has_sys = sys_type_names is not None
    has_ref = ref_type_names is not None
    has_map = ref_type_map is not None

    if has_map:
        # Case 1: explicit map always wins.  Compose ref_code → sys_code → sys_name.
        result: Dict[int, str] = {}
        for ref_code_str, sys_code_str in ref_type_map.items():
            ref_code = int(ref_code_str)
            sys_code = int(sys_code_str)
            if has_sys:
                name = sys_type_names.get(sys_code, sys_code_str)
            else:
                name = sys_code_str
            result[ref_code] = str(name)
        return result

    if has_ref and has_sys:
        # Case 2: both sides have type_names.  IKeys will match via alias names.
        ref_alias_values = set(ref_type_names.values())
        sys_alias_values = set(sys_type_names.values())
        extra = ref_alias_values - sys_alias_values
        if extra:
            warnings.warn(
                f"aa_ref.ref_type_names aliases {sorted(extra)} are not "
                f"present in system.type_names. IKey mismatches may occur.",
                UserWarning,
                stacklevel=4,
            )
        return {int(k): str(v) for k, v in ref_type_names.items()}

    if has_ref and not has_sys:
        # Case 4a: ref has aliases, system does not → warn.
        warnings.warn(
            "aa_ref.ref_type_names is set but system.type_names is not. "
            "IKey alignment depends on ref alias names matching system raw codes.",
            UserWarning,
            stacklevel=4,
        )
        return {int(k): str(v) for k, v in ref_type_names.items()}

    if has_sys and not has_ref:
        # Case 4b: system has aliases, ref does not → warn.
        warnings.warn(
            "system.type_names is set but aa_ref.ref_type_names is not. "
            "Consider adding ref_type_names or ref_type_map for correct "
            "IKey alignment.",
            UserWarning,
            stacklevel=4,
        )
        return None

    # Case 3: neither aliases, no map.
    if ref_topo is not None and ref_topo != system_topo:
        warnings.warn(
            "aa_ref.ref_topo differs from system.topology_file but neither "
            "type_names nor ref_type_map is set. Atom type codes may not "
            "align between topologies.",
            UserWarning,
            stacklevel=4,
        )
    return None


def validate_fm_spec_domain(
    spec_domain: Tuple[float, float],
    source_table_path: str,
) -> Tuple[float, float, float]:
    """Validate that an FM spec domain matches the source table and return resolution.

    Returns ``(spec_min, spec_max, resolution)``.
    Raises ``ACGConfigError`` on mismatch.
    """
    import numpy as np
    from ..io.tables import parse_lammps_table

    r_values, _, _ = parse_lammps_table(source_table_path)
    if r_values.size < 2:
        raise ACGConfigError(
            f"Source table {source_table_path} must contain at least "
            "two grid points."
        )
    source_min = float(r_values[0])
    source_max = float(r_values[-1])
    spec_min, spec_max = spec_domain
    tol = 1.0e-8
    if abs(spec_min - source_min) > tol or abs(spec_max - source_max) > tol:
        raise ACGConfigError(
            f"FM spec domain [{spec_min}, {spec_max}] does not match source "
            f"table domain [{source_min}, {source_max}] for "
            f"{source_table_path}."
        )
    resolution = float(np.median(np.diff(np.asarray(r_values, dtype=float))))
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ACGConfigError(
            f"Source table {source_table_path} must use a strictly "
            "increasing grid."
        )
    return spec_min, spec_max, resolution


def _pop_optional_str(
    mapping: MutableMapping[str, Any], key: str
) -> Optional[str]:
    value = mapping.pop(key, None)
    if value is None:
        return None
    return str(value)


def _pop_optional_float(
    mapping: MutableMapping[str, Any],
    key: str,
    *,
    default: Optional[float] = None,
) -> Optional[float]:
    value = mapping.pop(key, default)
    if value is None:
        return None
    return float(value)


def _pop_optional_int(
    mapping: MutableMapping[str, Any],
    key: str,
    *,
    default: Optional[int] = None,
) -> Optional[int]:
    value = mapping.pop(key, default)
    if value is None:
        return None
    return int(value)


def _pop_optional_mapping(
    mapping: MutableMapping[str, Any], key: str
) -> Optional[Dict[str, Any]]:
    value = mapping.pop(key, None)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ACGConfigError(
            f"{key} must be a mapping, got {type(value).__name__}."
        )
    return {str(subkey): subvalue for subkey, subvalue in value.items()}


def _pop_sim_var(mapping: MutableMapping[str, Any]) -> Dict[str, str]:
    import json as _json

    raw = mapping.pop("sim_var", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str):
        try:
            parsed = _json.loads(raw)
        except _json.JSONDecodeError as exc:
            raise ACGConfigError(
                f"sampling.sim_var must be valid JSON, got: {raw!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ACGConfigError(
                f"sampling.sim_var must be a JSON object, "
                f"got {type(parsed).__name__}."
            )
        return {str(k): str(v) for k, v in parsed.items()}
    raise ACGConfigError(
        f"sampling.sim_var must be a JSON string or mapping, "
        f"got {type(raw).__name__}."
    )


def _pop_aa_ref_noise_config(mapping: MutableMapping[str, Any]) -> AARefNoiseConfig:
    """Parse flat ``[aa_ref] noise_*`` keys into a stored noise config."""
    enabled = _pop_optional_bool(mapping, "noise_enabled", default=False)
    return AARefNoiseConfig(
        enabled=enabled,
        samples_per_frame=_pop_optional_int(
            mapping, "noise_samples_per_frame", default=0,
        ),
        sigma=_pop_optional_float(mapping, "noise_sigma", default=0.0),
        sigma_final=_pop_optional_float(mapping, "noise_sigma_final"),
        schedule=str(mapping.pop("noise_schedule", "constant")).strip().lower(),
        update_interval=_pop_optional_int(
            mapping, "noise_update_interval", default=1,
        ),
        seed=_pop_optional_int(mapping, "noise_seed", default=0),
        distribution=str(mapping.pop("noise_distribution", "gaussian")).strip().lower(),
        selection=mapping.pop("noise_selection", "all"),
        include_original=_pop_optional_bool(
            mapping, "noise_include_original", default=False,
        ),
        wrap=_pop_optional_bool(mapping, "noise_wrap", default=False),
        batch_size=_pop_optional_int(mapping, "noise_batch_size"),
        subsample_per_epoch=_pop_optional_int(
            mapping, "noise_subsample_per_epoch", default=0,
        ),
        cache_policy=str(mapping.pop("noise_cache_policy", "stage")).strip().lower(),
        force_mix_ratio=_pop_optional_float(
            mapping, "noise_force_mix_ratio", default=0.0,
        ),
        neighbor_mode=str(mapping.pop("noise_neighbor_mode", "shared")).strip().lower(),
        neighbor_skin=_pop_optional_float(
            mapping, "noise_neighbor_skin", default=0.0,
        ),
    )


def _pop_optional_bool(
    mapping: MutableMapping[str, Any],
    key: str,
    *,
    default: bool = False,
) -> bool:
    value = mapping.pop(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


# ─── FM spec parsing ─────────────────────────────────────────────────

def _build_fm_training_specs(
    fm_specs_raw: MutableMapping[str, Any],
    *,
    method: str,
) -> FMTrainingSpecs:
    _warn_unknown_nested_fm_spec_keys(fm_specs_raw)
    pair_specs = _parse_fm_spec_group(
        fm_specs_raw.pop("pair_specs", ()), expected_style="pair"
    )
    bond_specs = _parse_fm_spec_group(
        fm_specs_raw.pop("bond_specs", ()), expected_style="bond"
    )
    angle_specs = _parse_fm_spec_group(
        fm_specs_raw.pop("angle_specs", ()), expected_style="angle"
    )
    if method not in {"fm", "dsm"} and (pair_specs or bond_specs or angle_specs):
        raise ACGConfigError(
            "training.fm_specs is only supported when training.method is 'fm' or 'dsm'."
        )
    return FMTrainingSpecs(
        pair_specs=pair_specs,
        bond_specs=bond_specs,
        angle_specs=angle_specs,
    )


def _warn_unknown_nested_fm_spec_keys(mapping: Mapping[str, Any]) -> None:
    _warn_unknown_section_keys(
        "training.fm_specs",
        _FM_SPEC_ALLOWED_KEYS,
        mapping,
        preserved="preserving it nowhere",
    )


def _parse_fm_spec_group(
    raw_specs: Any, *, expected_style: str
) -> tuple[FMInteractionSpec, ...]:
    if raw_specs in (None, ""):
        return ()
    if not isinstance(raw_specs, Iterable) or isinstance(
        raw_specs, (str, bytes, Mapping)
    ):
        raise ACGConfigError(
            f"training.fm_specs.{_FM_SPEC_STYLE_TO_KEY[expected_style]} "
            "must be a list of tuple specs."
        )
    parsed = []
    for index, item in enumerate(raw_specs):
        parsed.append(
            _parse_one_fm_spec(
                item,
                expected_style=expected_style,
                entry_index=index,
            )
        )
    return tuple(parsed)


def _parse_one_fm_spec(
    raw_spec: Any,
    *,
    expected_style: str,
    entry_index: int,
) -> FMInteractionSpec:
    if isinstance(raw_spec, Mapping):
        return _parse_named_fm_spec(
            raw_spec,
            expected_style=expected_style,
            entry_index=entry_index,
        )
    if not isinstance(raw_spec, (list, tuple)):
        raise ACGConfigError(
            f"training.fm_specs.{_FM_SPEC_STYLE_TO_KEY[expected_style]}"
            f"[{entry_index}] must be a list/tuple."
        )
    if expected_style == "pair":
        if len(raw_spec) not in {6, 7}:
            raise ACGConfigError(
                f"training.fm_specs.{_FM_SPEC_STYLE_TO_KEY[expected_style]}"
                f"[{entry_index}] must have 6 or 7 elements: "
                "[style, types, model, model_size, domain, max_force, "
                "?model_overrides]."
            )
    elif len(raw_spec) not in {5, 6}:
        raise ACGConfigError(
            f"training.fm_specs.{_FM_SPEC_STYLE_TO_KEY[expected_style]}"
            f"[{entry_index}] must have 5 or 6 elements: "
            "[style, types, model, model_size, domain, ?model_overrides]."
        )

    style = str(raw_spec[0]).strip().lower()
    if style != expected_style:
        raise ACGConfigError(
            f"training.fm_specs.{_FM_SPEC_STYLE_TO_KEY[expected_style]}"
            f"[{entry_index}] declared style {style!r}; "
            f"expected {expected_style!r}."
        )

    raw_types = raw_spec[1]
    if not isinstance(raw_types, (list, tuple)):
        raise ACGConfigError(
            f"training.fm_specs.{_FM_SPEC_STYLE_TO_KEY[expected_style]}"
            f"[{entry_index}] types must be a list/tuple."
        )
    normalized_types = _normalize_fm_spec_types(
        style, raw_types, entry_index=entry_index
    )

    model = str(raw_spec[2]).strip().lower()
    if model not in _FM_SPEC_MODELS:
        raise ACGConfigError(
            f"training.fm_specs.{_FM_SPEC_STYLE_TO_KEY[expected_style]}"
            f"[{entry_index}] model must be one of "
            f"{sorted(_FM_SPEC_MODELS)}, got {model!r}."
        )

    model_size = int(raw_spec[3])
    if model_size <= 0:
        raise ACGConfigError(
            f"training.fm_specs.{_FM_SPEC_STYLE_TO_KEY[expected_style]}"
            f"[{entry_index}] model_size must be positive."
        )

    raw_domain = raw_spec[4]
    if not isinstance(raw_domain, (list, tuple)) or len(raw_domain) != 2:
        raise ACGConfigError(
            f"training.fm_specs.{_FM_SPEC_STYLE_TO_KEY[expected_style]}"
            f"[{entry_index}] domain must be a two-value list/tuple "
            "[min, max]."
        )
    minimum = float(raw_domain[0])
    maximum = float(raw_domain[1])
    if not minimum < maximum:
        raise ACGConfigError(
            f"training.fm_specs.{_FM_SPEC_STYLE_TO_KEY[expected_style]}"
            f"[{entry_index}] requires domain min < max, "
            f"got {minimum} >= {maximum}."
        )

    max_force: Optional[float] = None
    model_overrides: Dict[str, Any] = {}
    if style == "pair":
        raw_max_force = raw_spec[5]
        if isinstance(raw_max_force, Mapping):
            raise ACGConfigError(
                f"training.fm_specs.pair_specs[{entry_index}] must include "
                "max_force before optional model_overrides."
            )
        max_force = float(raw_max_force)
        if not math.isfinite(max_force) or max_force <= 0.0:
            raise ACGConfigError(
                f"training.fm_specs.pair_specs[{entry_index}] max_force must "
                f"be a finite positive number, got {raw_max_force!r}."
            )
        if len(raw_spec) == 7:
            if not isinstance(raw_spec[6], Mapping):
                raise ACGConfigError(
                    f"training.fm_specs.pair_specs[{entry_index}] "
                    "model_overrides must be a dict when provided."
                )
            model_overrides = dict(raw_spec[6])
    elif len(raw_spec) == 6:
        if not isinstance(raw_spec[5], Mapping):
            raise ACGConfigError(
                f"training.fm_specs.{_FM_SPEC_STYLE_TO_KEY[expected_style]}"
                f"[{entry_index}] does not accept the pair-only max_force "
                "slot; provide only optional model_overrides in position 6."
            )
        model_overrides = dict(raw_spec[5])

    return FMInteractionSpec(
        style=style,
        types=normalized_types,
        model=model,
        model_size=model_size,
        domain=(minimum, maximum),
        max_force=max_force,
        model_overrides=model_overrides,
        init_mode="source_table_fit",
    )


def _parse_named_fm_spec(
    raw_spec: Mapping[str, Any],
    *,
    expected_style: str,
    entry_index: int,
) -> FMInteractionSpec:
    key_prefix = (
        f"training.fm_specs.{_FM_SPEC_STYLE_TO_KEY[expected_style]}"
        f"[{entry_index}]"
    )
    style = str(raw_spec.get("style", expected_style)).strip().lower()
    if style != expected_style:
        raise ACGConfigError(
            f"{key_prefix} declared style {style!r}; "
            f"expected {expected_style!r}."
        )
    if "model_size" in raw_spec:
        raise ACGConfigError(
            f"{key_prefix} uses the legacy model_size field. "
            "Use n_coeffs in authored B-spline specs."
        )

    model = str(raw_spec.get("model", "bspline")).strip().lower()
    if model in {"gaussian", "gauss/cut"}:
        return _parse_named_gaussian_fm_spec(
            raw_spec,
            expected_style=expected_style,
            entry_index=entry_index,
            key_prefix=key_prefix,
        )
    if model != "bspline":
        raise ACGConfigError(
            f"{key_prefix} authored specs support model='bspline' or "
            f"model='gauss/cut', got {raw_spec.get('model')!r}."
        )

    raw_types = raw_spec.get("types")
    if not isinstance(raw_types, (list, tuple)):
        raise ACGConfigError(f"{key_prefix} types must be a list/tuple.")
    normalized_types = _normalize_fm_spec_types(
        style, raw_types, entry_index=entry_index
    )

    if "n_coeffs" not in raw_spec:
        raise ACGConfigError(f"{key_prefix} must define n_coeffs.")
    n_coeffs = int(raw_spec["n_coeffs"])
    if n_coeffs <= 0:
        raise ACGConfigError(f"{key_prefix} n_coeffs must be positive.")

    raw_domain = raw_spec.get("domain")
    if not isinstance(raw_domain, (list, tuple)) or len(raw_domain) != 2:
        raise ACGConfigError(
            f"{key_prefix} domain must be a two-value list/tuple [min, max]."
        )
    minimum = float(raw_domain[0])
    maximum = float(raw_domain[1])
    if not minimum < maximum:
        raise ACGConfigError(
            f"{key_prefix} requires domain min < max, "
            f"got {minimum} >= {maximum}."
        )

    if "degree" not in raw_spec:
        raise ACGConfigError(f"{key_prefix} must define degree.")
    degree = int(raw_spec["degree"])
    if degree <= 0:
        raise ACGConfigError(f"{key_prefix} degree must be positive.")

    max_force: Optional[float] = None
    if style == "pair":
        if "max_force" not in raw_spec:
            raise ACGConfigError(
                f"{key_prefix} pair specs must define max_force."
            )
        max_force = float(raw_spec["max_force"])
        if not math.isfinite(max_force) or max_force <= 0.0:
            raise ACGConfigError(
                f"{key_prefix} max_force must be a finite positive number, "
                f"got {raw_spec['max_force']!r}."
            )
    elif "max_force" in raw_spec:
        raise ACGConfigError(
            f"{key_prefix} does not accept the pair-only max_force field."
        )

    return FMInteractionSpec(
        style=style,
        types=normalized_types,
        model="bspline",
        model_size=n_coeffs,
        domain=(minimum, maximum),
        max_force=max_force,
        model_overrides={"degree": degree},
        init_mode="authored_zero",
        resolution=_AUTHORED_FM_EXPORT_RESOLUTION[style],
    )


def _parse_named_gaussian_fm_spec(
    raw_spec: Mapping[str, Any],
    *,
    expected_style: str,
    entry_index: int,
    key_prefix: str,
) -> FMInteractionSpec:
    if expected_style != "pair":
        raise ACGConfigError(f"{key_prefix} gauss/cut specs are pair-only.")

    style = str(raw_spec.get("style", expected_style)).strip().lower()
    if style != "pair":
        raise ACGConfigError(
            f"{key_prefix} declared style {style!r}; expected 'pair'."
        )

    raw_types = raw_spec.get("types")
    if not isinstance(raw_types, (list, tuple)):
        raise ACGConfigError(f"{key_prefix} types must be a list/tuple.")
    normalized_types = _normalize_fm_spec_types(
        "pair", raw_types, entry_index=entry_index
    )

    raw_domain = raw_spec.get("domain")
    if not isinstance(raw_domain, (list, tuple)) or len(raw_domain) != 2:
        raise ACGConfigError(
            f"{key_prefix} gauss/cut specs must define domain = [min, max]."
        )
    minimum = float(raw_domain[0])
    maximum = float(raw_domain[1])
    if not minimum < maximum:
        raise ACGConfigError(
            f"{key_prefix} requires domain min < max, got {minimum} >= {maximum}."
        )

    if "params" in raw_spec:
        params = raw_spec["params"]
        if not isinstance(params, (list, tuple)) or len(params) != 3:
            raise ACGConfigError(f"{key_prefix} params must be [A, r0, sigma].")
        A, r0, sigma = (float(params[0]), float(params[1]), float(params[2]))
    else:
        missing = [name for name in ("A", "r0", "sigma") if name not in raw_spec]
        if missing:
            raise ACGConfigError(
                f"{key_prefix} gauss/cut specs must define A, r0, and sigma."
            )
        A = float(raw_spec["A"])
        r0 = float(raw_spec["r0"])
        sigma = float(raw_spec["sigma"])
    if not math.isfinite(A):
        raise ACGConfigError(f"{key_prefix} A must be finite.")
    if not math.isfinite(r0):
        raise ACGConfigError(f"{key_prefix} r0 must be finite.")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ACGConfigError(f"{key_prefix} sigma must be finite and positive.")

    cutoff = float(raw_spec.get("cutoff", maximum))
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ACGConfigError(f"{key_prefix} cutoff must be finite and positive.")
    resolution = float(raw_spec.get("resolution", _AUTHORED_FM_EXPORT_RESOLUTION["pair"]))
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ACGConfigError(f"{key_prefix} resolution must be finite and positive.")

    return FMInteractionSpec(
        style="pair",
        types=normalized_types,
        model="gauss/cut",
        model_size=3,
        domain=(minimum, maximum),
        max_force=None,
        model_overrides={
            "A": A,
            "r0": r0,
            "sigma": sigma,
            "cutoff": cutoff,
        },
        init_mode="authored_direct",
        resolution=resolution,
    )


def _normalize_fm_spec_types(
    style: str,
    raw_types: Sequence[Any],
    *,
    entry_index: int,
) -> tuple[str, ...]:
    token_types = tuple(str(item) for item in raw_types)
    if style in {"pair", "bond"} and len(token_types) != 2:
        raise ACGConfigError(
            f"training.fm_specs.{_FM_SPEC_STYLE_TO_KEY[style]}"
            f"[{entry_index}] requires exactly 2 types for "
            f"style {style!r}."
        )
    if style == "angle" and len(token_types) != 3:
        raise ACGConfigError(
            f"training.fm_specs.angle_specs[{entry_index}] requires exactly "
            "3 types for style 'angle'."
        )
    if style == "pair":
        return InteractionKey.pair(*token_types).types
    if style == "bond":
        return InteractionKey.bond(*token_types).types
    if style == "angle":
        return InteractionKey.angle(*token_types).types
    raise ACGConfigError(f"Unsupported FM spec style {style!r}.")
