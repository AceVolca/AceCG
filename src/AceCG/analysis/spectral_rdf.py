"""Spectral Monte Carlo estimators for three-dimensional partial RDFs.

The mathematical state machine in this module is shared by the serial API and
the MPI reducer.  It deliberately contains no MPI imports or communication
logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional
import warnings

import numpy as np
from MDAnalysis.lib.mdamath import triclinic_vectors

from .rdf import _interaction_keys_from_forcefield
from .spectral_basis import make_basis
from ..topology.neighbor import (
    count_eligible_pairs_by_type,
    parse_exclude_option,
)
from ..topology.topology_array import TopologyArrays
from ..topology.types import InteractionKey


@dataclass(frozen=True)
class SpectralRDFResult:
    """Final spectral estimate and diagnostic data for one pair key."""

    key: InteractionKey
    r: np.ndarray
    values: np.ndarray
    derivative: np.ndarray
    coefficients: np.ndarray
    coefficient_stderr: np.ndarray
    block_coefficient_stderr: Optional[np.ndarray]
    mode_cutoff: int
    basis: str
    domain: tuple[float, float]
    n_frames: int
    weight_sum: float
    meta: dict[str, Any]


def _resolve_pair_keys(
    forcefield_snapshot: Any,
    pair_keys: Optional[Iterable[InteractionKey]],
) -> list[InteractionKey]:
    keys = [
        key
        for key in _interaction_keys_from_forcefield(forcefield_snapshot, pair_keys)
        if key.style == "pair"
    ]
    keys = list(dict.fromkeys(keys))
    for key in keys:
        if len(key.types) != 2:
            raise ValueError(f"Pair key must have two type labels, got {key!r}")
    return keys


def _normalize_mode_value(value: Any, max_modes: int, *, label: str) -> str | int:
    if isinstance(value, str):
        token = value.strip().lower()
        if token == "auto":
            return token
        raise ValueError(f"{label} must be an integer or 'auto', got {value!r}")
    cutoff = int(value)
    if cutoff < 0 or cutoff > int(max_modes):
        raise ValueError(f"{label} must be between 0 and {max_modes}, got {cutoff}")
    return cutoff


def _normalize_mode_overrides(
    values: Optional[Mapping[InteractionKey | str, str | int]],
    max_modes: int,
) -> dict[str, str | int]:
    if values is None:
        return {}
    out: dict[str, str | int] = {}
    for raw_key, raw_value in values.items():
        key = raw_key if isinstance(raw_key, InteractionKey) else InteractionKey.from_label(str(raw_key))
        if key.style != "pair":
            raise ValueError(f"mode_cutoff_by_key only accepts pair keys, got {key}")
        out[key.label()] = _normalize_mode_value(
            raw_value,
            max_modes,
            label=f"mode_cutoff_by_key[{key.label()!r}]",
        )
    return out


def _cell_geometry(box: np.ndarray) -> tuple[float, np.ndarray, float]:
    dimensions = np.asarray(box, dtype=np.float64)
    if dimensions.shape != (6,) or not np.all(np.isfinite(dimensions)):
        raise ValueError(f"box must be a finite MDAnalysis dimensions vector, got {dimensions}")
    vectors = np.asarray(triclinic_vectors(dimensions, dtype=np.float64), dtype=np.float64)
    if vectors.shape != (3, 3):
        raise ValueError(f"Could not construct triclinic cell vectors from box {dimensions}")
    volume = float(abs(np.linalg.det(vectors)))
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError(f"Invalid periodic cell volume {volume} from box {dimensions}")
    face_areas = np.asarray(
        [
            np.linalg.norm(np.cross(vectors[1], vectors[2])),
            np.linalg.norm(np.cross(vectors[2], vectors[0])),
            np.linalg.norm(np.cross(vectors[0], vectors[1])),
        ],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(face_areas)) or np.any(face_areas <= 0.0):
        raise ValueError(f"Invalid triclinic face areas {face_areas}")
    heights = volume / face_areas
    return volume, heights, 0.5 * float(np.min(heights))


def _auto_mode_cutoff(coefficients: np.ndarray) -> tuple[int, dict[str, Any]]:
    coeff = np.asarray(coefficients, dtype=np.float64)
    max_modes = int(coeff.size - 1)
    diagnostics: dict[str, Any] = {
        "method": "piecewise_log_decay_plateau",
        "valid": False,
        "fallback": max_modes,
    }
    if max_modes < 8:
        diagnostics["reason"] = "at least eight nonconstant modes are required"
        return max_modes, diagnostics

    scale = max(1.0, float(np.max(np.abs(coeff[1:]))))
    epsilon = np.finfo(np.float64).eps * scale
    y = np.log(np.maximum(np.abs(coeff[1:]), epsilon))
    modes = np.arange(1, max_modes + 1, dtype=np.float64)
    first_candidate = 4
    last_candidate = max_modes - 4
    diagnostics.update(
        {
            "epsilon": epsilon,
            "search_range": [first_candidate, last_candidate],
        }
    )
    candidates: list[tuple[float, int, float, float]] = []
    for cutoff in range(first_candidate, last_candidate + 1):
        front_mask = modes <= float(cutoff)
        x_front = modes[front_mask]
        y_front = y[front_mask]
        design = np.column_stack((x_front, np.ones_like(x_front)))
        slope, intercept = np.linalg.lstsq(design, y_front, rcond=None)[0]
        plateau = float(np.mean(y[~front_mask]))
        front_residual = y_front - (slope * x_front + intercept)
        rear_residual = y[~front_mask] - plateau
        rss = float(np.dot(front_residual, front_residual) + np.dot(rear_residual, rear_residual))
        candidates.append((rss, cutoff, float(slope), plateau))

    rss, cutoff, slope, plateau = min(candidates, key=lambda item: item[0])
    diagnostics.update(
        {
            "candidate": int(cutoff),
            "slope": float(slope),
            "plateau": float(plateau),
            "residual_sum_squares": float(rss),
        }
    )
    if cutoff in {first_candidate, last_candidate}:
        diagnostics["reason"] = "best changepoint lies on the search boundary"
        return max_modes, diagnostics
    if not np.isfinite(slope) or slope >= 0.0:
        diagnostics["reason"] = "fitted decay segment does not have a negative slope"
        return max_modes, diagnostics
    diagnostics["valid"] = True
    diagnostics["fallback"] = None
    return int(cutoff), diagnostics


def init_spectral_rdf_state(
    topology_arrays: TopologyArrays,
    forcefield_snapshot: Any,
    *,
    pair_keys: Optional[Iterable[InteractionKey]] = None,
    cutoff: float = 30.0,
    r_min: float = 0.0,
    r_max: Optional[float] = None,
    basis: str = "cosine",
    max_modes: int = 200,
    mode_cutoff: str | int = "auto",
    mode_cutoff_by_key: Optional[Mapping[InteractionKey | str, str | int]] = None,
    grid_size: int = 2000,
    pair_chunk_size: int = 100_000,
    exclude_option: str = "resid",
    sel_indices: Optional[np.ndarray] = None,
    block_size: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Initialize the shared serial/MPI spectral RDF sufficient statistics."""
    maximum_mode = int(max_modes)
    if maximum_mode < 1:
        raise ValueError("max_modes must be at least 1")
    grid_count = int(grid_size)
    if grid_count < 2:
        raise ValueError("grid_size must be at least 2")
    chunk_size = int(pair_chunk_size)
    if chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")
    left = float(r_min)
    right = float(cutoff) if r_max is None else float(r_max)
    if not np.isfinite(left) or not np.isfinite(right) or left < 0.0 or right <= left:
        raise ValueError(f"Expected 0 <= r_min < r_max, got {(left, right)}")
    if block_size is not None and int(block_size) <= 0:
        raise ValueError("block_size must be positive when provided")

    basis_object = make_basis(basis)
    keys = _resolve_pair_keys(forcefield_snapshot, pair_keys)
    if not keys:
        raise ValueError("spectral RDF requires at least one pair InteractionKey")
    option = parse_exclude_option(exclude_option)
    selected = (
        np.arange(topology_arrays.n_atoms, dtype=np.int32)
        if sel_indices is None
        else np.asarray(sel_indices, dtype=np.int32)
    )
    eligible_map = count_eligible_pairs_by_type(
        topology_arrays,
        keys,
        sel_indices=selected,
        exclude_option=option,
    )
    eligible = np.asarray([eligible_map[key] for key in keys], dtype=np.int64)
    empty_keys = [key.label() for key, count in zip(keys, eligible) if int(count) <= 0]
    if empty_keys:
        raise ValueError(f"No eligible pairs remain for spectral RDF keys: {empty_keys}")

    type_codes = np.asarray(topology_arrays.atom_type_codes, dtype=np.int32)
    type_counts = []
    for key in keys:
        code_a = topology_arrays.atom_type_name_to_code.get(str(key.types[0]))
        code_b = topology_arrays.atom_type_name_to_code.get(str(key.types[1]))
        type_counts.append(
            (
                int(np.count_nonzero(type_codes[selected] == int(code_a))),
                int(np.count_nonzero(type_codes[selected] == int(code_b))),
            )
        )

    global_mode = _normalize_mode_value(mode_cutoff, maximum_mode, label="mode_cutoff")
    overrides = _normalize_mode_overrides(mode_cutoff_by_key, maximum_mode)
    n_coefficients = maximum_mode + 1
    n_pairs = len(keys)
    return {
        "pair_keys": keys,
        "basis_name": basis_object.name,
        "domain": (left, right),
        "max_modes": maximum_mode,
        "mode_cutoff": global_mode,
        "mode_cutoff_by_key": overrides,
        "grid_size": grid_count,
        "pair_chunk_size": chunk_size,
        "exclude_option": option,
        "selection_size": int(selected.size),
        "eligible_pair_counts": eligible,
        "type_counts": np.asarray(type_counts, dtype=np.int64),
        "block_size": None if block_size is None else int(block_size),
        "metadata": dict(metadata or {}),
        "coefficient_sum": np.zeros((n_pairs, n_coefficients), dtype=np.float64),
        "coefficient_square_sum": np.zeros((n_pairs, n_coefficients), dtype=np.float64),
        "weight_sum": 0.0,
        "weight_square_sum": 0.0,
        "n_frames": 0,
        "block_coefficient_sum": {},
        "block_weight_sum": {},
        "negative_min_half_height": -np.inf,
        "max_half_height_ratio": 0.0,
        "volume_sum": 0.0,
        "max_volume": -np.inf,
        "negative_min_volume": -np.inf,
        "max_frame_idx": -1,
        "negative_min_frame_idx": -np.inf,
    }


def accumulate_spectral_rdf_frame(
    state: dict[str, Any],
    frame_geometry: Any,
    *,
    frame_weight: float = 1.0,
    frame_idx: Optional[int] = None,
) -> None:
    """Accumulate one frame's direct spectral coefficient estimators."""
    weight = float(frame_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(f"frame_weight must be finite and nonnegative, got {weight}")
    if frame_idx is None:
        frame_idx = getattr(frame_geometry, "frame_idx", None)
    if frame_idx is None and state.get("block_size") is not None:
        raise ValueError("frame_idx is required when block_size is enabled")
    frame_number = -1 if frame_idx is None else int(frame_idx)

    box = np.asarray(getattr(frame_geometry, "box", None), dtype=np.float64)
    volume, _heights, half_height = _cell_geometry(box)
    r_min, r_max = state["domain"]
    tolerance = max(1.0e-6, 1.0e-7 * half_height)
    if r_max > half_height + tolerance:
        raise ValueError(
            f"spectral RDF r_max={r_max:g} exceeds the unique minimum-image "
            f"radius {half_height:g} at frame {frame_number}"
        )

    basis_object = make_basis(state["basis_name"])
    n_modes = int(state["max_modes"]) + 1
    frame_coefficients = np.zeros_like(state["coefficient_sum"], dtype=np.float64)
    distance_map = getattr(frame_geometry, "pair_distances", None)
    if distance_map is None:
        raise ValueError("spectral RDF requires pair distances in the frame payload")
    zero_epsilon = np.sqrt(np.finfo(np.float64).tiny)

    for pair_index, key in enumerate(state["pair_keys"]):
        if key not in distance_map:
            raise ValueError(
                f"spectral RDF frame payload is missing pair distances for {key} "
                f"at frame {frame_number}"
            )
        distances = np.asarray(distance_map[key], dtype=np.float64).reshape(-1)
        if np.any(~np.isfinite(distances)):
            bad = int(np.count_nonzero(~np.isfinite(distances)))
            raise ValueError(f"{key} frame {frame_number} contains {bad} non-finite distances")
        zero_count = int(np.count_nonzero(distances <= zero_epsilon))
        if zero_count:
            raise ValueError(
                f"{key} frame {frame_number} contains {zero_count} distinct-particle "
                "distances at or below numerical zero"
            )
        selected = distances[(distances >= r_min) & (distances <= r_max)]
        for start in range(0, selected.size, int(state["pair_chunk_size"])):
            chunk = selected[start : start + int(state["pair_chunk_size"])]
            phi = basis_object.evaluate(chunk, n_modes, state["domain"])
            radial_weight = 1.0 / (4.0 * np.pi * chunk * chunk)
            frame_coefficients[pair_index] += phi.T @ radial_weight
        frame_coefficients[pair_index] *= (
            volume / float(state["eligible_pair_counts"][pair_index])
        )

    state["coefficient_sum"] += weight * frame_coefficients
    state["coefficient_square_sum"] += weight * frame_coefficients * frame_coefficients
    state["weight_sum"] += weight
    state["weight_square_sum"] += weight * weight
    state["n_frames"] += 1
    state["negative_min_half_height"] = max(
        float(state["negative_min_half_height"]), -half_height
    )
    state["max_half_height_ratio"] = max(
        float(state["max_half_height_ratio"]), r_max / half_height
    )
    state["volume_sum"] += volume
    state["max_volume"] = max(float(state["max_volume"]), volume)
    state["negative_min_volume"] = max(float(state["negative_min_volume"]), -volume)
    if frame_idx is not None:
        state["max_frame_idx"] = max(int(state["max_frame_idx"]), int(frame_idx))
        state["negative_min_frame_idx"] = max(
            float(state["negative_min_frame_idx"]), -float(frame_idx)
        )

    block_size = state.get("block_size")
    if block_size is not None:
        block_id = int(frame_idx) // int(block_size)
        if block_id not in state["block_coefficient_sum"]:
            state["block_coefficient_sum"][block_id] = np.zeros_like(frame_coefficients)
            state["block_weight_sum"][block_id] = 0.0
        state["block_coefficient_sum"][block_id] += weight * frame_coefficients
        state["block_weight_sum"][block_id] += weight


def _spectral_config(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: state[key]
        for key in (
            "pair_keys",
            "basis_name",
            "domain",
            "max_modes",
            "mode_cutoff",
            "mode_cutoff_by_key",
            "grid_size",
            "pair_chunk_size",
            "exclude_option",
            "selection_size",
            "eligible_pair_counts",
            "type_counts",
            "block_size",
            "metadata",
        )
    }


def spectral_rdf_local_partials(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return MPI-reducible sufficient statistics with no frame geometry."""
    return {
        "spectral_config": _spectral_config(state),
        "coefficient_sum": np.asarray(state["coefficient_sum"], dtype=np.float64),
        "coefficient_square_sum": np.asarray(
            state["coefficient_square_sum"], dtype=np.float64
        ),
        "weight_sum": float(state["weight_sum"]),
        "weight_square_sum": float(state["weight_square_sum"]),
        "n_frames": int(state["n_frames"]),
        "block_coefficient_sum": dict(state["block_coefficient_sum"]),
        "block_weight_sum": dict(state["block_weight_sum"]),
        "negative_min_half_height": float(state["negative_min_half_height"]),
        "max_half_height_ratio": float(state["max_half_height_ratio"]),
        "volume_sum": float(state["volume_sum"]),
        "max_volume": float(state["max_volume"]),
        "negative_min_volume": float(state["negative_min_volume"]),
        "max_frame_idx": int(state["max_frame_idx"]),
        "negative_min_frame_idx": float(state["negative_min_frame_idx"]),
    }


def _weighted_sem(
    coefficient_sum: np.ndarray,
    coefficient_square_sum: np.ndarray,
    weight_sum: float,
    weight_square_sum: float,
) -> tuple[np.ndarray, float]:
    if weight_sum <= 0.0 or weight_square_sum <= 0.0:
        raise ValueError("spectral RDF requires a positive total frame weight")
    n_effective = weight_sum * weight_sum / weight_square_sum
    mean = coefficient_sum / weight_sum
    variance_numerator = coefficient_square_sum - weight_sum * mean * mean
    variance_numerator = np.maximum(variance_numerator, 0.0)
    variance_denominator = weight_sum - weight_square_sum / weight_sum
    if n_effective <= 1.0 or variance_denominator <= 0.0:
        return np.full_like(mean, np.nan, dtype=np.float64), float(n_effective)
    sample_variance = variance_numerator / variance_denominator
    return np.sqrt(sample_variance / n_effective), float(n_effective)


def _block_sem(state: Mapping[str, Any], shape: tuple[int, int]) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    if state.get("block_size") is None:
        return None, {"enabled": False}
    means = []
    block_weights = []
    for block_id in sorted(state.get("block_coefficient_sum", {})):
        weight = float(np.asarray(state["block_weight_sum"].get(block_id, 0.0)))
        if weight <= 0.0:
            continue
        means.append(np.asarray(state["block_coefficient_sum"][block_id], dtype=np.float64) / weight)
        block_weights.append({"block_id": int(block_id), "weight_sum": weight})
    if len(means) < 2:
        stderr = np.full(shape, np.nan, dtype=np.float64)
    else:
        stacked = np.stack(means, axis=0)
        stderr = np.std(stacked, axis=0, ddof=1) / np.sqrt(float(len(means)))
    return stderr, {
        "enabled": True,
        "block_size": int(state["block_size"]),
        "n_blocks": len(means),
        "blocks": block_weights,
        "method": "equal-weight SEM of nonempty block means",
    }


def finalize_spectral_rdf_state(
    state: Mapping[str, Any],
) -> dict[InteractionKey, SpectralRDFResult]:
    """Finalize local or MPI-reduced spectral sufficient statistics."""
    reduced_config = state.get("spectral_config")
    config = dict(
        _spectral_config(state) if reduced_config is None else reduced_config
    )
    weight_sum = float(state["weight_sum"])
    coefficient_stderr, n_effective = _weighted_sem(
        np.asarray(state["coefficient_sum"], dtype=np.float64),
        np.asarray(state["coefficient_square_sum"], dtype=np.float64),
        weight_sum,
        float(state["weight_square_sum"]),
    )
    coefficients = np.asarray(state["coefficient_sum"], dtype=np.float64) / weight_sum
    block_stderr, block_meta = _block_sem(
        {**config, **state}, tuple(coefficients.shape)
    )
    basis_object = make_basis(config["basis_name"])
    grid = np.linspace(
        float(config["domain"][0]),
        float(config["domain"][1]),
        int(config["grid_size"]),
        dtype=np.float64,
    )
    full_basis = basis_object.evaluate(grid, int(config["max_modes"]) + 1, config["domain"])
    full_derivative = basis_object.derivative(
        grid, int(config["max_modes"]) + 1, config["domain"]
    )
    n_frames = int(state["n_frames"])
    min_half_height = -float(state["negative_min_half_height"])
    min_volume = -float(state["negative_min_volume"])
    min_frame_idx = (
        None
        if not np.isfinite(float(state["negative_min_frame_idx"]))
        else int(round(-float(state["negative_min_frame_idx"])))
    )
    results: dict[InteractionKey, SpectralRDFResult] = {}
    overrides = dict(config["mode_cutoff_by_key"])
    for pair_index, key in enumerate(config["pair_keys"]):
        requested = overrides.get(key.label(), config["mode_cutoff"])
        if requested == "auto":
            selected_cutoff, cutoff_meta = _auto_mode_cutoff(coefficients[pair_index])
            if not cutoff_meta["valid"]:
                warnings.warn(
                    f"{key}: automatic spectral mode cutoff failed "
                    f"({cutoff_meta.get('reason', 'unknown reason')}); using max_modes={config['max_modes']}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            selected_cutoff = int(requested)
            cutoff_meta = {"method": "manual", "valid": True, "requested": selected_cutoff}
        active = slice(0, selected_cutoff + 1)
        values = full_basis[:, active] @ coefficients[pair_index, active]
        derivative = full_derivative[:, active] @ coefficients[pair_index, active]
        metadata = dict(config.get("metadata", {}))
        metadata.update(
            {
                "n_type_i": int(config["type_counts"][pair_index, 0]),
                "n_type_j": int(config["type_counts"][pair_index, 1]),
                "n_eligible_pairs": int(config["eligible_pair_counts"][pair_index]),
                "selection_size": int(config["selection_size"]),
                "exclude_option": str(config["exclude_option"]),
                "max_modes": int(config["max_modes"]),
                "pair_chunk_size": int(config["pair_chunk_size"]),
                "mode_cutoff_diagnostics": cutoff_meta,
                "iid_uncertainty": {
                    "method": "weighted IID SEM",
                    "n_effective": n_effective,
                    "corrects_time_autocorrelation": False,
                },
                "block_uncertainty": block_meta,
                "box_validation": {
                    "minimum_half_cell_height": min_half_height,
                    "maximum_rmax_to_half_height_ratio": float(state["max_half_height_ratio"]),
                    "minimum_volume": min_volume,
                    "maximum_volume": float(state["max_volume"]),
                    "mean_volume": float(state["volume_sum"]) / max(n_frames, 1),
                },
                "frame_range": {
                    "minimum_frame_idx": min_frame_idx,
                    "maximum_frame_idx": int(state["max_frame_idx"]),
                },
            }
        )
        results[key] = SpectralRDFResult(
            key=key,
            r=grid.copy(),
            values=np.asarray(values, dtype=np.float64),
            derivative=np.asarray(derivative, dtype=np.float64),
            coefficients=coefficients[pair_index].copy(),
            coefficient_stderr=coefficient_stderr[pair_index].copy(),
            block_coefficient_stderr=None
            if block_stderr is None
            else block_stderr[pair_index].copy(),
            mode_cutoff=int(selected_cutoff),
            basis=basis_object.name,
            domain=(float(config["domain"][0]), float(config["domain"][1])),
            n_frames=n_frames,
            weight_sum=weight_sum,
            meta=metadata,
        )
    return results


__all__ = [
    "SpectralRDFResult",
    "init_spectral_rdf_state",
    "accumulate_spectral_rdf_frame",
    "spectral_rdf_local_partials",
    "finalize_spectral_rdf_state",
]
