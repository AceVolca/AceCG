"""Central MPI compute engine for local frame computation and one-pass post-processing.

This module supports an optional light-weight per-frame geometry cache:

- ``FrameCache`` stores only the per-frame observables that were actually
  requested / computed for the current force field selection.
- ``TrajectoryCache`` stores a trajectory digest as
  ``{frame_idx: FrameCache}``.
- ``compute(..., request={"need_frame_cache": True, ...})`` returns the sliced
  per-frame cache under ``results["frame_cache"]``.
- ``compute(..., return_observables=True)`` keeps the legacy
  ``results["frame_observables"]`` spelling as a compatibility alias.
- ``run_post(spec)`` reads observable-cache options from ``spec`` and can
  optionally collect these frame observables locally and, under MPI, gather
  them to rank 0, merge them into a single trajectory cache, and write that
  cache to a pickle file just like other post-processing outputs.
"""

from __future__ import annotations

import json
import os
import pickle
import socket
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .energy import energy
from .force import force
from .frame_geometry import FrameGeometry, compute_frame_geometry
from .reducers import (
    _REQUEST_FLAGS,
    accumulate_cdfm_zbx_stats,
    canonical_step_mode,
    consume_step_payload,
    finalize_step_root,
    init_step_state,
    local_step_partials,
    slice_observed_rows,
    step_reduce_plan,
    step_request,
)
from ..io.logger import get_screen_logger
from ..topology.forcefield import Forcefield
from ..topology.neighbor import compute_pairs_by_type
from ..topology.topology_array import TopologyArrays
from ..topology.types import InteractionKey

SCREEN_LOGGER = get_screen_logger("mpi_engine")


def _backup_existing_output(output_path: Path) -> None:
    """Move an existing output file aside before writing a fresh result."""
    if not output_path.exists():
        return
    backup_path = output_path.with_name(output_path.name + ".bak")
    if backup_path.exists():
        backup_path.unlink()
    output_path.replace(backup_path)


@dataclass
class _RegisteredFn:
    fn: Callable
    reduce: str
    desc: str = ""


@dataclass(frozen=True)
class FrameCache:
    """Light-weight per-frame geometry cache.

    Notes
    -----
    Field semantics intentionally distinguish two cases:

    - ``None``: this observable family was not requested / not computed.
    - ``{}``: this observable family was requested, but the current frame has
      no matching interaction terms for that family.

    This allows downstream code to distinguish "not part of this job" from
    "part of this job but empty in this frame".
    """

    frame_idx: int
    pair_distances: Optional[Dict[InteractionKey, np.ndarray]] = None
    bond_distances: Optional[Dict[InteractionKey, np.ndarray]] = None
    angle_values: Optional[Dict[InteractionKey, np.ndarray]] = None
    dihedral_values: Optional[Dict[InteractionKey, np.ndarray]] = None
    box: Optional[np.ndarray] = None


@dataclass
class TrajectoryCache:
    """Trajectory-level digest of per-frame geometry caches.

    Stored as a dictionary keyed by the unique global frame index so that the
    cache can be constructed locally and later merged across ranks.
    """

    frames: Dict[int, FrameCache] = field(default_factory=dict)

    def add(self, frame: FrameCache) -> None:
        """Insert or overwrite one frame's cache."""
        self.frames[int(frame.frame_idx)] = frame

    def merge(self, other: "TrajectoryCache") -> None:
        """Merge another trajectory cache into this one by frame index."""
        self.frames.update(other.frames)


# Backward-compatible names used by older RDF/observables-cache code paths.
FrameObservables = FrameCache
TrajectoryObservablesCache = TrajectoryCache



def _labels_to_interaction_keys(
    labels: Optional[Sequence[str]],
) -> Optional[List[InteractionKey]]:
    """Parse JSON-friendly interaction-key labels into ``InteractionKey`` objects.

    Parameters
    ----------
    labels
        Optional sequence of labels such as ``["pair:A:B", "bond:C:D"]``.
        ``None`` is returned unchanged so callers can distinguish between
        "no selection provided" and an explicit empty list.
    """
    if labels is None:
        return None
    return [InteractionKey.from_label(str(label)) for label in labels]


def _labels_to_mode_by_key(
    mapping: Optional[Mapping[str, str]],
) -> Optional[Dict[InteractionKey, str]]:
    """Parse a JSON-friendly ``{label: mode}`` mapping for RDF/PDF selection."""
    if mapping is None:
        return None
    return {InteractionKey.from_label(str(key)): str(value) for key, value in mapping.items()}


def _selected_frame_ids_from_spec(
    shared_spec: Dict[str, Any],
    total_frames: int,
) -> List[int]:
    """Return the global frame ids selected by the current post-processing spec."""
    discrete_ids = shared_spec.get("frame_ids")
    if discrete_ids is not None:
        selected = [int(fid) for fid in discrete_ids]
        return _subsample_frame_ids_from_noise_spec(selected, shared_spec)

    frame_start = 0 if shared_spec.get("frame_start") is None else int(shared_spec["frame_start"])
    frame_end = int(total_frames) if shared_spec.get("frame_end") is None else int(shared_spec["frame_end"])
    every = int(shared_spec.get("every", 1))
    n_subsample = _noise_subsample_per_epoch_from_spec(shared_spec)
    selected_range = range(frame_start, frame_end, every)
    n_selected = len(selected_range)
    if n_subsample <= 0 or n_subsample >= n_selected:
        return list(selected_range)

    rng = np.random.default_rng(_noise_subsample_seed_from_spec(shared_spec))
    picked_offsets = rng.choice(n_selected, size=n_subsample, replace=False)
    picked_offsets.sort()
    return [selected_range[int(offset)] for offset in picked_offsets]


def _noise_subsample_per_epoch_from_spec(shared_spec: Mapping[str, Any]) -> int:
    """Return the requested AA-noise frame subsample size for one epoch."""
    noise_spec = shared_spec.get("noise")
    if noise_spec is None:
        return 0
    if not isinstance(noise_spec, Mapping):
        return 0
    if not bool(noise_spec.get("enabled", True)):
        return 0
    raw = noise_spec.get("subsample_per_epoch", 0)
    if raw is None:
        return 0
    n_subsample = int(raw)
    if n_subsample < 0:
        raise ValueError("spec['noise']['subsample_per_epoch'] must be non-negative.")
    return n_subsample


def _subsample_frame_ids_from_noise_spec(
    frame_ids: Sequence[int],
    shared_spec: Mapping[str, Any],
) -> List[int]:
    """Apply sorted, no-replacement AA-noise frame subsampling."""
    selected = [int(fid) for fid in frame_ids]
    n_subsample = _noise_subsample_per_epoch_from_spec(shared_spec)
    if n_subsample <= 0 or n_subsample >= len(selected):
        return selected

    rng = np.random.default_rng(_noise_subsample_seed_from_spec(shared_spec))
    picked_offsets = rng.choice(len(selected), size=n_subsample, replace=False)
    picked = [selected[int(offset)] for offset in picked_offsets]
    picked.sort()
    return picked


def _noise_subsample_seed_from_spec(shared_spec: Mapping[str, Any]) -> int:
    """Return the seed used only for AA-noise frame subsampling."""
    noise_spec = shared_spec.get("noise")
    if not isinstance(noise_spec, Mapping):
        return 0
    return int(noise_spec.get("subsample_seed", noise_spec.get("seed", 0)))


def _load_frame_weight_file(
    frame_weight_file: Optional[str],
    *,
    work_dir: Path,
) -> Optional[np.ndarray]:
    """Load full-trajectory frame weights from a ``.npy`` or ``.npz`` file."""
    if frame_weight_file is None:
        return None

    weight_path = Path(str(frame_weight_file))
    if not weight_path.is_absolute():
        weight_path = work_dir / weight_path
    if not weight_path.exists():
        raise FileNotFoundError(f"frame_weight_file does not exist: {weight_path}")

    suffix = weight_path.suffix.lower()
    if suffix == ".npy":
        values = np.load(weight_path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(weight_path, allow_pickle=False) as payload:
            files = list(payload.files)
            if "frame_weight" in payload:
                values = payload["frame_weight"]
            elif len(files) == 1:
                values = payload[files[0]]
            else:
                raise ValueError(
                    "frame_weight_file .npz must contain exactly one array or a "
                    "'frame_weight' array."
                )
    else:
        raise ValueError(f"frame_weight_file must end with .npy or .npz, got: {weight_path}")

    return np.asarray(values, dtype=np.float32)


def _frame_weight_array_from_spec(
    shared_spec: Dict[str, Any],
    total_frames: int,
    *,
    loaded_frame_weight: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Return a full-trajectory frame-weight array for the current spec.

    Inline ``spec['frame_weight']`` keeps the current selected-frame convention:
    it may have length equal to the selected frame set. File-backed weights are
    full-trajectory arrays, matching the historical ``frame_weight_file`` path.
    """
    raw = loaded_frame_weight if loaded_frame_weight is not None else shared_spec.get("frame_weight")
    if raw is None:
        return None

    selected_ids = _selected_frame_ids_from_spec(shared_spec, total_frames)
    selected = np.asarray(selected_ids, dtype=np.int64)
    if selected.size and (np.any(selected < 0) or np.any(selected >= int(total_frames))):
        raise ValueError("selected frame ids must be within the input trajectory length")

    weights = np.asarray(raw, dtype=np.float32)
    if weights.ndim != 1:
        raise ValueError(f"frame_weight must be a 1D array, got shape {weights.shape}")
    if np.any(weights < 0.0):
        raise ValueError("frame_weight must be nonnegative with positive sum")

    if loaded_frame_weight is not None or weights.shape == (int(total_frames),):
        if weights.shape != (int(total_frames),):
            raise ValueError(
                "frame_weight_file must have length equal to the input trajectory length: "
                f"expected {int(total_frames)}, got {weights.shape[0]}"
            )
        selected_weights = weights[selected] if selected.size else np.empty((0,), dtype=np.float32)
        if float(np.sum(selected_weights)) <= 0.0:
            raise ValueError("frame_weight must be nonnegative with positive sum")
        return weights

    if weights.shape != (len(selected_ids),):
        raise ValueError(
            "spec['frame_weight'] must have the same length as the selected frame set: "
            f"expected {len(selected_ids)}, got {weights.shape}"
        )
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("frame_weight must be nonnegative with positive sum")

    full_weights = np.ones(int(total_frames), dtype=np.float32)
    if selected.size:
        full_weights[selected] = weights
    return full_weights


def _run_rdf_step(
    step: Dict[str, Any],
    *,
    source: Any,
    topology_arrays: TopologyArrays,
    forcefield_snapshot: Forcefield,
    frame_weights: Optional[Sequence[float]],
    default_cutoff: Optional[float],
    default_sel_indices: Optional[np.ndarray],
    default_exclude_option: str,
) -> Dict[InteractionKey, Any]:
    """Execute one ``step_mode='rdf'`` step via ``analysis.rdf``.

    The step is JSON-friendly. Any interaction-key selections or per-key modes
    must therefore be supplied using string labels such as ``"pair:A:B"``.
    """
    from ..analysis.rdf import interaction_distributions

    interaction_keys = _labels_to_interaction_keys(step.get("interaction_keys"))
    mode_by_key = _labels_to_mode_by_key(step.get("mode_by_key"))
    cutoff = default_cutoff if step.get("cutoff") is None else float(step["cutoff"])
    if cutoff is None:
        cutoff = 30.0

    sel_indices = default_sel_indices
    if step.get("sel_indices") is not None:
        sel_indices = np.asarray(step["sel_indices"], dtype=np.int32)

    return interaction_distributions(
        source,
        topology_arrays,
        forcefield_snapshot,
        interaction_keys=interaction_keys,
        frame_weights=frame_weights,
        mode_by_key=mode_by_key,
        start=int(step.get("frame_start", 0)),
        end=None if step.get("frame_end") is None else int(step["frame_end"]),
        every=int(step.get("every", 1)),
        cutoff=float(cutoff),
        r_max=None if step.get("r_max") is None else float(step["r_max"]),
        nbins_pair=int(step.get("nbins_pair", 200)),
        nbins_bond=int(step.get("nbins_bond", 200)),
        nbins_angle=int(step.get("nbins_angle", 180)),
        nbins_dihedral=int(step.get("nbins_dihedral", 180)),
        exclude_option=str(step.get("exclude_option", default_exclude_option)),
        sel_indices=sel_indices,
        angle_degrees=bool(step.get("angle_degrees", True)),
        dihedral_degrees=bool(step.get("dihedral_degrees", True)),
        dihedral_periodic=bool(step.get("dihedral_periodic", True)),
        default_pair_mode=str(step.get("default_pair_mode", "rdf")),
        default_bonded_mode=str(step.get("default_bonded_mode", "pdf")),
    )

def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _trace(enabled: bool, rank: int, message: str, *, all_ranks: bool = False) -> None:
    if not enabled:
        return
    if rank != 0 and not all_ranks:
        return
    SCREEN_LOGGER.info(message, rank=rank)


def _add_timing(bucket: Dict[str, Any], key: str, dt: float) -> None:
    bucket[key] = float(bucket.get(key, 0.0)) + float(dt)


def _int_spec_value(spec: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(spec.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _write_rank_heartbeat(
    work_dir: Path,
    *,
    rank: int,
    size: int,
    processed: int,
    local_total: int,
    global_total: int,
    frame_id: int | None,
    start_time: float,
) -> None:
    elapsed = max(time.monotonic() - start_time, 0.0)
    payload = {
        "rank": int(rank),
        "size": int(size),
        "host": socket.gethostname().split(".")[0],
        "pid": int(os.getpid()),
        "processed": int(processed),
        "local_total": int(local_total),
        "global_total": int(global_total),
        "last_frame_id": None if frame_id is None else int(frame_id),
        "elapsed_sec": float(elapsed),
        "frames_per_sec": 0.0 if elapsed == 0.0 else float(processed) / elapsed,
        "updated_at": time.time(),
    }
    path = work_dir / f"progress_rank_{rank:04d}.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _normalize_box_batch(box: np.ndarray, n_samples: int) -> np.ndarray:
    bx = np.asarray(box, dtype=np.float64)
    if bx.ndim == 1:
        return np.broadcast_to(bx, (int(n_samples), bx.shape[0])).copy()
    if bx.ndim == 2 and bx.shape[0] == int(n_samples):
        return bx.copy()
    raise ValueError(
        "box must have shape (6,) or (n_samples, 6), "
        f"got {bx.shape} for n_samples={n_samples}"
    )


def _normalize_compute_box(box: np.ndarray, sample_shape: Tuple[int, ...]) -> np.ndarray:
    bx = np.asarray(box, dtype=np.float64)
    if bx.shape[-1:] != (6,):
        raise ValueError(f"box last dimension must be 6, got {bx.shape}")
    if not sample_shape:
        if bx.ndim != 1:
            raise ValueError(f"single-frame box must have shape (6,), got {bx.shape}")
        return bx
    if bx.ndim == 1:
        return np.broadcast_to(bx, sample_shape + (bx.shape[0],)).copy()
    if bx.shape == sample_shape + (bx.shape[-1],):
        return bx.copy()
    n_samples = int(np.prod(sample_shape, dtype=np.int64))
    if bx.ndim == 2 and bx.shape[0] == n_samples:
        return bx.reshape(sample_shape + (bx.shape[1],)).copy()
    raise ValueError(
        "batched box must have shape (6,), sample_shape + (6,), or "
        f"(n_samples, 6); got {bx.shape} for sample_shape={sample_shape}."
    )


def _normalize_sample_weights(
    frame_weights: Optional[np.ndarray],
    n_samples: int,
) -> np.ndarray:
    if frame_weights is None:
        return np.full(int(n_samples), 1.0 / float(n_samples), dtype=np.float64)
    weights = np.asarray(frame_weights, dtype=np.float64).reshape(-1)
    if weights.shape != (int(n_samples),):
        raise ValueError(
            "frame_weights must flatten to the compute batch sample count "
            f"{int(n_samples)}; got {weights.shape}."
        )
    weight_sum = float(np.sum(weights))
    if np.any(weights < 0.0) or weight_sum <= 0.0:
        raise ValueError("frame_weights must be nonnegative with positive sum.")
    return weights / weight_sum


def _flatten_reference_for_compute(
    reference_forces: Optional[np.ndarray],
    *,
    sample_shape: Tuple[int, ...],
    n_samples: int,
    n_atoms: int,
) -> Tuple[Optional[np.ndarray], bool]:
    if reference_forces is None:
        return None, False
    ref = np.asarray(reference_forces)
    if not sample_shape:
        return ref, False
    if ref.ndim == 1 or ref.shape == (int(n_atoms), 3):
        return ref, False
    leading_ndim = len(sample_shape)
    if ref.ndim > leading_ndim and ref.shape[:leading_ndim] == sample_shape:
        return ref.reshape((int(n_samples),) + ref.shape[leading_ndim:]), True
    if ref.ndim >= 3 and ref.shape[-2:] == (int(n_atoms), 3):
        leading = int(np.prod(ref.shape[:-2], dtype=np.int64))
        if leading == int(n_samples):
            return ref.reshape((int(n_samples), int(n_atoms), 3)), True
    if ref.ndim >= 2 and ref.shape[0] == int(n_samples):
        return ref.reshape((int(n_samples),) + ref.shape[1:]), True
    return ref, False


def _batch_atom_values_like_reference(
    values: np.ndarray,
    reference_batch: np.ndarray,
    topology_arrays: TopologyArrays,
    *,
    n_atoms: int,
) -> np.ndarray:
    """Return full-atom batch values reshaped/sliced like a reference batch."""
    arr = np.asarray(values)
    ref = np.asarray(reference_batch)
    if arr.ndim != 3 or arr.shape[1:] != (int(n_atoms), 3):
        raise ValueError(
            "values must have shape (n_samples, n_atoms, 3), "
            f"got {arr.shape}."
        )
    n_samples = int(arr.shape[0])
    if ref.shape == arr.shape:
        return arr.astype(ref.dtype, copy=False)

    flat = arr.reshape(n_samples, int(n_atoms) * 3)
    if ref.shape == flat.shape:
        return flat.astype(ref.dtype, copy=False)

    real_site_indices = getattr(topology_arrays, "real_site_indices", None)
    if real_site_indices is not None:
        atoms = np.asarray(real_site_indices, dtype=np.int32).reshape(-1)
        observed = arr[:, atoms, :]
        if ref.shape == observed.shape:
            return observed.astype(ref.dtype, copy=False)
        observed_flat = slice_observed_rows(flat, atoms)
        if ref.shape == observed_flat.shape:
            return observed_flat.astype(ref.dtype, copy=False)

    raise ValueError(
        "reference force shape is not compatible with full or observed noisy "
        f"force rows: reference={ref.shape}, full={arr.shape}."
    )


def _weighted_accumulate(
    out: Dict[str, Any],
    key: str,
    values: np.ndarray,
    weights: np.ndarray,
) -> None:
    arr = np.asarray(values)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if arr.ndim == 0:
        contribution = float(np.sum(w)) * arr
    elif arr.shape[0] == w.size:
        contribution = np.tensordot(w, arr, axes=(0, 0))
    elif w.size == 1:
        contribution = float(w[0]) * arr
    else:
        raise ValueError(
            f"{key} output shape {arr.shape} is not compatible with weights shape {w.shape}."
        )
    if key in out:
        out[key] = np.asarray(out[key]) + contribution
    else:
        out[key] = contribution


def _merge_fm_stats(out: Dict[str, Any], partial: Dict[str, Any]) -> None:
    if "fm_stats" not in out:
        out["fm_stats"] = {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            for key, value in partial.items()
        }
        out["fm_stats"]["n_frames"] = 1
        return
    total = out["fm_stats"]
    for key in ("JtJ", "Jtf", "Jty"):
        total[key] = np.asarray(total[key]) + np.asarray(partial[key])
    for key in ("ftf", "fTy", "yty", "weight_sum"):
        total[key] = float(total[key]) + float(partial[key])
    total["n_force_rows"] = max(int(total["n_force_rows"]), int(partial["n_force_rows"]))
    total["n_atoms_obs"] = max(int(total["n_atoms_obs"]), int(partial["n_atoms_obs"]))
    total["n_frames"] = 1


def _noise_selection_indices(
    selection: Any,
    topology_arrays: TopologyArrays,
    n_atoms: int,
) -> np.ndarray:
    if selection is None:
        selection = "all"
    if isinstance(selection, str):
        key = selection.strip().lower()
        if key == "all":
            return np.arange(n_atoms, dtype=np.int32)
        if key == "real":
            real_site_indices = getattr(topology_arrays, "real_site_indices", None)
            if real_site_indices is None:
                raise ValueError("noise selection 'real' requires topology real_site_indices.")
            return np.asarray(real_site_indices, dtype=np.int32).reshape(-1)
        if key == "none":
            return np.empty(0, dtype=np.int32)
        parts = [part.strip() for part in selection.split(",") if part.strip()]
        if parts:
            return np.asarray([int(part) for part in parts], dtype=np.int32)
        raise ValueError(f"Unsupported noise selection {selection!r}.")
    indices = np.asarray(selection, dtype=np.int32).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= int(n_atoms)):
        raise ValueError("noise selection indices are outside the atom range.")
    return indices


def _wrap_positions_in_box(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    lengths = np.asarray(box, dtype=np.float64)[..., :3]
    if np.any(lengths <= 0.0):
        return positions
    return np.mod(positions, lengths[:, None, :])


def _write_timing_report(
    work_dir: Path,
    gathered: List[Dict[str, Any]],
    *,
    metadata: Dict[str, Any],
) -> Path:
    numeric_keys = sorted(
        {
            key
            for payload in gathered
            for key, value in payload.items()
            if isinstance(value, (int, float))
        }
    )
    summary: Dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(payload.get(key, 0.0)) for payload in gathered]
        summary[key] = {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(sum(values) / len(values)),
        }
    payload = {
        "metadata": metadata,
        "summary": summary,
        "per_rank": gathered,
    }
    path = work_dir / "mpi_post_timing.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _has_enabled_style(
    keys: Sequence[Any],
    interaction_mask: Optional[Dict[InteractionKey, bool]],
    style: str,
) -> bool:
    """Return ``True`` if at least one enabled key of ``style`` exists."""
    for key in keys:
        if getattr(key, "style", None) != style:
            continue
        if interaction_mask is not None and not interaction_mask.get(key, False):
            continue
        return True
    return False


def geometry_to_observables(
    frame_idx: int,
    geom: FrameGeometry,
    *,
    include_pair: bool = True,
    include_bond: bool = True,
    include_angle: bool = True,
    include_dihedral: bool = True,
    include_box: bool = True,
) -> FrameCache:
    """Slice a heavy ``FrameGeometry`` object into a light observable cache.

    Parameters
    ----------
    frame_idx
        Global frame index associated with ``geom``.
    geom
        Full per-frame geometry object constructed inside ``compute()``.
    include_pair / include_bond / include_angle / include_dihedral
        Whether the corresponding observable family should be included in the
        returned ``FrameCache`` object. If ``False``, that field is set to
        ``None`` to mark "not requested / not computed".
    include_box
        Whether to store the periodic box for the frame.

    Returns
    -------
    FrameCache
        Lightweight cache containing only the requested observable families.
    """
    return FrameCache(
        frame_idx=int(frame_idx),
        pair_distances=dict(geom.pair_distances) if include_pair else None,
        bond_distances=dict(geom.bond_distances) if include_bond else None,
        angle_values=dict(geom.angle_values) if include_angle else None,
        dihedral_values=dict(geom.dihedral_values) if include_dihedral else None,
        box=np.asarray(geom.box, dtype=np.float32).copy() if include_box else None,
    )


class MPIComputeEngine:
    """Task-scoped MPI runtime.

    The core fast path still performs one-pass post-processing. Optional
    observable caching can be enabled when downstream workflows need a per-frame
    digest of computed distances / angles / dihedrals.
    """

    def __init__(self, serial_threshold: int = 10, *, comm=None) -> None:
        self._registry: Dict[str, _RegisteredFn] = {}
        self._serial_threshold = serial_threshold
        self.comm = comm

    def register(
        self,
        name: str,
        fn: Callable,
        reduce: str,
        description: str = "",
    ) -> None:
        """Register one frame-level compute function.

        Parameters
        ----------
        name : str
            Observable name used in compute requests.
        fn : Callable
            Function called as ``fn(geometry, forcefield, **kwargs)`` for each
            frame.
        reduce : {"sum", "gather", "stack", "dict_sum"}
            Reduction mode used across frames/ranks.
        description : str, default=""
            Human-readable description for diagnostics.
        """
        if reduce not in ("sum", "gather", "stack", "dict_sum"):
            raise ValueError(f"Invalid reduce mode: {reduce!r}")
        self._registry[name] = _RegisteredFn(fn=fn, reduce=reduce, desc=description)

    @property
    def registered_names(self) -> List[str]:
        """Return names of observables currently registered on the engine."""
        return list(self._registry.keys())

    def compute(
        self,
        request: Dict[str, bool],
        frame: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
        topology_arrays: TopologyArrays,
        forcefield_snapshot: Forcefield,
        frame_weight: float = 1.0,
        frame_weights: Optional[np.ndarray] = None,
        interaction_mask: Optional[np.ndarray] = None,
        pair_type_list: Optional[List[Any]] = None,
        pair_cutoff: Optional[float] = None,
        sel_indices: Optional[np.ndarray] = None,
        exclude_option: str = "resid",
        timing: Optional[Dict[str, Any]] = None,
        return_observables: bool = False,
        frame_idx: Optional[int] = None,
        pair_cache_override: Optional[Dict[InteractionKey, Tuple[np.ndarray, np.ndarray]]] = None,
        batch_size: Optional[int] = None,
        neighbor_mode: str = "shared",
        neighbor_skin: float = 0.0,
        neighbor_reference_positions: Optional[np.ndarray] = None,
        neighbor_reference_box: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute registered observables for one frame or same-frame batch.

        Parameters
        ----------
        request
            Direct observable requests aggregated from all active trainers /
            post-processing steps.
        frame
            Frame tuple in the format ``(frame_id, positions, box,
            reference_forces)``. ``positions`` may have shape ``(n_atoms, 3)``
            or ``(..., n_atoms, 3)``.
        frame_weights
            Optional sample weights for batched coordinates. They are normalized
            within this compute call. The scalar ``frame_weight`` remains the
            outer per-base-frame weight used by reducers.
        return_observables
            If ``True``, append a light-weight ``FrameCache`` object to the
            returned payload under the legacy key
            ``results["frame_observables"]``.
        frame_idx
            Optional explicit frame index to use when constructing
            ``FrameCache``. If omitted, the first entry of ``frame`` is
            used, which matches the normal iter_frames contract.
        neighbor_mode
            Pair-cache policy for batched same-frame coordinates. ``"shared"``
            builds one cache from the first sample, ``"skin"`` builds one
            skin-expanded cache from the supplied reference frame when present,
            and ``"chunk"`` rebuilds once per ``batch_size`` chunk.
        neighbor_skin
            Nonnegative extra distance added to ``pair_cutoff`` during pair
            search. Force/energy kernels still apply each potential's true
            cutoff, so skin-region pairs are candidates only.

        Notes
        -----
        ``interaction_mask`` defaults to ``forcefield_snapshot.key_mask`` when
        not provided.
        """
        results: Dict[str, Any] = {}

        frame_id, positions, box, reference_forces = frame
        # Normalize both single frames and same-frame batches to one flat sample
        # axis. Single-frame chunks are later passed back as the original 2D
        # shape to preserve the public compute() contract.
        positions_arr = np.asarray(positions, dtype=np.float64)
        if positions_arr.ndim == 2 and positions_arr.shape[-1] == 3:
            sample_shape: Tuple[int, ...] = ()
            n_samples = 1
            n_atoms = int(positions_arr.shape[0])
            flat_positions = positions_arr.reshape(1, n_atoms, 3)
        elif positions_arr.ndim >= 3 and positions_arr.shape[-1] == 3:
            sample_shape = tuple(int(v) for v in positions_arr.shape[:-2])
            n_samples = int(np.prod(sample_shape, dtype=np.int64))
            if n_samples <= 0:
                raise ValueError("batched compute positions must contain at least one sample.")
            n_atoms = int(positions_arr.shape[-2])
            flat_positions = positions_arr.reshape(n_samples, n_atoms, 3)
        else:
            raise ValueError(
                "frame positions must have shape (n_atoms, 3) or "
                f"(..., n_atoms, 3), got {positions_arr.shape}."
            )
        is_batch = bool(sample_shape)
        box_arr = _normalize_compute_box(box, sample_shape)
        flat_boxes = box_arr.reshape(n_samples, -1) if is_batch else box_arr.reshape(1, -1)
        reference_flat, reference_is_batched = _flatten_reference_for_compute(
            reference_forces,
            sample_shape=sample_shape,
            n_samples=n_samples,
            n_atoms=n_atoms,
        )
        if batch_size is not None:
            batch_size = int(batch_size)
            if batch_size <= 0:
                raise ValueError("batch_size must be positive when set.")
        neighbor_mode = str(neighbor_mode or "shared").strip().lower()
        neighbor_mode_aliases = {
            "per_chunk": "chunk",
            "per-batch": "chunk",
            "per_batch": "chunk",
            "per_sample": "chunk",
            "per-sample": "chunk",
        }
        neighbor_mode = neighbor_mode_aliases.get(neighbor_mode, neighbor_mode)
        if neighbor_mode not in {"shared", "skin", "chunk"}:
            raise ValueError(
                "neighbor_mode must be one of 'shared', 'skin', or 'chunk'."
            )
        neighbor_skin = float(neighbor_skin or 0.0)
        if neighbor_skin < 0.0:
            raise ValueError("neighbor_skin must be non-negative.")

        if frame_idx is None:
            frame_ids_arr = np.asarray(frame_id, dtype=np.int64).reshape(-1)
            frame_idx = int(frame_ids_arr[0]) if frame_ids_arr.size else int(frame_id)
        results["frame_idx"] = int(frame_idx)

        active_interaction_mask = (
            interaction_mask
            if interaction_mask is not None
            else forcefield_snapshot.key_mask
        )
        ff_keys = list(forcefield_snapshot.keys())

        build_pairs = bool(pair_type_list) and pair_cutoff is not None
        build_bonds = _has_enabled_style(ff_keys, active_interaction_mask, "bond")
        build_angles = _has_enabled_style(ff_keys, active_interaction_mask, "angle")
        build_dihedrals = _has_enabled_style(ff_keys, active_interaction_mask, "dihedral")

        pair_cache = pair_cache_override if build_pairs else None
        pair_search_skin = neighbor_skin if neighbor_mode in {"skin", "chunk"} else 0.0
        pair_search_cutoff = (
            float(pair_cutoff) + float(pair_search_skin)
            if pair_cutoff is not None
            else None
        )

        need_frame_cache = bool(request.get("need_frame_cache", False))
        # Current cache semantics are per real trajectory frame, not per noisy
        # same-frame sample. Noisy FM/REM therefore keep cache requests off.
        if is_batch and (return_observables or need_frame_cache):
            raise ValueError("batched compute does not support frame-cache requests.")

        weights = _normalize_sample_weights(frame_weights, n_samples)
        chunk_size = n_samples if batch_size is None else min(int(batch_size), n_samples)
        cdfm_stats: Optional[Dict[str, Any]] = None
        cache_geom: Optional[FrameGeometry] = None

        # Single frames and batches share this loop. batch_size limits per-chunk
        # memory; neighbor_mode decides whether this chunk rebuilds pair_cache.
        for start in range(0, n_samples, chunk_size):

            stop = min(start + chunk_size, n_samples)
            chunk_weights = weights[start:stop]
            if float(np.sum(chunk_weights)) <= 0.0:
                continue
            chunk_abs_weights = chunk_weights * float(frame_weight)
            chunk_reference = (
                reference_flat[start:stop]
                if reference_is_batched and reference_flat is not None
                else reference_flat
            )
            chunk_positions = flat_positions[start:stop] if is_batch else flat_positions[0]
            chunk_boxes = flat_boxes[start:stop] if is_batch else flat_boxes[0]
            chunk_pair_cache = pair_cache

            # Step 1: Build pair list if needed.
            rebuild_neighbor = (
                build_pairs
                and pair_cache_override is None
                and (neighbor_mode == "chunk" or pair_cache is None)
            )
            if rebuild_neighbor:
                if neighbor_mode == "skin" and neighbor_reference_positions is not None:
                    ref_positions_arr = np.asarray(
                        neighbor_reference_positions,
                        dtype=np.float64,
                    )
                    if ref_positions_arr.ndim == 2 and ref_positions_arr.shape == (n_atoms, 3):
                        ref_positions = ref_positions_arr
                    elif (
                        ref_positions_arr.ndim >= 3
                        and ref_positions_arr.shape[-2:] == (n_atoms, 3)
                    ):
                        ref_positions = ref_positions_arr.reshape(-1, n_atoms, 3)[0]
                    else:
                        raise ValueError(
                            "neighbor_reference_positions must have shape "
                            f"(n_atoms, 3) or (..., n_atoms, 3), got "
                            f"{ref_positions_arr.shape}."
                        )
                elif neighbor_mode == "chunk":
                    ref_positions = flat_positions[start]
                else:
                    ref_positions = flat_positions[0]

                if neighbor_mode == "skin" and neighbor_reference_box is not None:
                    ref_box_arr = np.asarray(neighbor_reference_box, dtype=np.float64)
                    if ref_box_arr.shape[-1:] != (6,):
                        raise ValueError(
                            "neighbor_reference_box last dimension must be 6, "
                            f"got {ref_box_arr.shape}."
                        )
                    ref_box = (
                        ref_box_arr
                        if ref_box_arr.ndim == 1
                        else ref_box_arr.reshape(-1, ref_box_arr.shape[-1])[0]
                    )
                elif neighbor_mode == "chunk":
                    ref_box = flat_boxes[start]
                else:
                    ref_box = flat_boxes[0]

                t_pair = time.monotonic()
                # Actually calls the neighbor list builder.
                chunk_pair_cache = compute_pairs_by_type(
                    positions=ref_positions,
                    box=ref_box,
                    pair_type_list=pair_type_list,
                    cutoff=float(pair_search_cutoff),
                    topology_arrays=topology_arrays,
                    sel_indices=sel_indices,
                    exclude_option=exclude_option,
                )
                if timing is not None:
                    _add_timing(timing, "pair_search", time.monotonic() - t_pair)
                if neighbor_mode != "chunk":
                    pair_cache = chunk_pair_cache
            # End step 1: build neighbor list.

            # Step 2: compute frame geometry. This has been batch-optimized.
            t0 = time.monotonic()
            geom = compute_frame_geometry(
                chunk_positions,
                chunk_boxes,
                topology_arrays,
                interaction_mask=active_interaction_mask,
                pair_cache=chunk_pair_cache,
            )
            if timing is not None:
                _add_timing(timing, "geometry", time.monotonic() - t0)
            if start == 0 and not is_batch and (return_observables or need_frame_cache):
                cache_geom = geom

            # Step 3: Compute energy-related observables if requested.
            energy_result: Dict[str, Any] = {}
            if (
                bool(request.get("need_energy_value", False))
                or bool(request.get("need_energy_grad", False))
                or bool(request.get("need_energy_hessian", False))
                or bool(request.get("need_energy_grad_outer", False))
                or bool(request.get("need_gauge_free_energy_grad", False))
                or bool(request.get("need_gauge_free_energy_grad_outer", False))
            ):
                t0 = time.monotonic()
                energy_result = energy(
                    geom,
                    forcefield_snapshot,
                    return_value=bool(request.get("need_energy_value", False)),
                    return_grad=bool(request.get("need_energy_grad", False)),
                    return_hessian=bool(request.get("need_energy_hessian", False)),
                    return_grad_outer=bool(request.get("need_energy_grad_outer", False)),
                    return_gauge_free_energy_grad=bool(
                        request.get("need_gauge_free_energy_grad", False)
                    ),
                    return_gauge_free_energy_grad_outer=bool(
                        request.get("need_gauge_free_energy_grad_outer", False)
                    ),
                )
                for key, value in energy_result.items():
                    # Reducers consume single-frame-shaped payloads, so batch
                    # observables are folded to weighted averages here.
                    _weighted_accumulate(results, key, value, chunk_weights)
                if timing is not None:
                    _add_timing(timing, "energy_kernel", time.monotonic() - t0)

            # Step 4: Compute force-related observables if requested.
            force_result: Dict[str, Any] = {}
            if (
                bool(request.get("need_force_value", False))
                or bool(request.get("need_force_grad", False))
                or bool(request.get("need_fm_stats", False))
            ):
                t0 = time.monotonic()
                force_result = force(
                    geom,
                    forcefield_snapshot,
                    return_value=bool(request.get("need_force_value", False)),
                    return_grad=bool(request.get("need_force_grad", False)),
                    reference_force=chunk_reference,
                    frame_weights=chunk_abs_weights
                    if bool(request.get("need_fm_stats", False))
                    else None,
                    return_fm_stats=bool(request.get("need_fm_stats", False)),
                    timing=timing,
                    )
                for key in ("force", "force_grad", "force_hessian"):
                    if key in force_result:
                        # Force values/Jacobians follow the same reducer-facing
                        # average convention as energy observables.
                        _weighted_accumulate(results, key, force_result[key], chunk_weights)
                if "fm_stats_sum" in force_result:
                    _merge_fm_stats(results, force_result["fm_stats_sum"])
                elif "fm_stats" in force_result:
                    _merge_fm_stats(results, force_result["fm_stats"])
                if timing is not None:
                    _add_timing(timing, "force_kernel", time.monotonic() - t0)
            # Special treatment for CDFM batch stats. Restrained CDFM normally
            # avoids same-frame noisy batches, but this keeps batched sufficient
            # statistics correct for callers that request them.
            if (
                is_batch
                and "energy_grad" in energy_result
                and "force" in force_result
                and "force_grad" in force_result
            ):
                # CDFM reinforce needs sum_w outer(energy_grad, force), which
                # cannot be reconstructed from averaged gradients and forces.
                cdfm_stats = accumulate_cdfm_zbx_stats(
                    cdfm_stats,
                    force_value=force_result["force"],
                    force_grad=force_result["force_grad"],
                    energy_grad=energy_result["energy_grad"],
                    weights=chunk_abs_weights,
                )

        if cdfm_stats is not None:
            results["cdfm_stats"] = cdfm_stats
        if "energy" in results and np.asarray(results["energy"]).ndim == 0:
            results["energy"] = float(results["energy"])
        if cache_geom is not None:
            frame_cache = geometry_to_observables(
                frame_idx=frame_idx,
                geom=cache_geom,
                include_pair=build_pairs,
                include_bond=build_bonds,
                include_angle=build_angles,
                include_dihedral=build_dihedrals,
                include_box=True,
            )
            if need_frame_cache:
                results["frame_cache"] = frame_cache
            if return_observables:
                results["frame_observables"] = frame_cache

        # Add other observable requests from registered functions.
        # Placeholder for future registered observable requests.

        return results

    def add_noise(
        self,
        frame: Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]],
        noise: Mapping[str, Any],
        topology_arrays: TopologyArrays,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray]:
        """Return a noisy same-frame coordinate batch and normalized sample weights.

        ``noise`` is an epoch-local runtime spec, passed from workflow to engine via post_spec dict.
        """
        frame_id, positions, box, reference_forces = frame
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[-1] != 3:
            raise ValueError(f"frame positions must have shape (n_atoms, 3), got {pos.shape}")

        n_atoms = int(pos.shape[0])
        n_noisy = int(noise.get("samples_per_frame", 1))
        if n_noisy < 0:
            raise ValueError("noise samples_per_frame must be non-negative.")
        include_original = bool(noise.get("include_original", False))
        total_samples = n_noisy + (1 if include_original else 0)
        if total_samples <= 0:
            raise ValueError(
                "noise requires samples_per_frame > 0 unless include_original is true."
            )

        frame_ids = np.full(total_samples, int(frame_id), dtype=np.int64)
        positions_batch = np.empty((total_samples, n_atoms, 3), dtype=np.float64)
        displacement_batch = np.zeros((total_samples, n_atoms, 3), dtype=np.float64)
        next_slot = 0
        if include_original:
            positions_batch[0] = pos
            next_slot = 1

        sigma = float(noise.get("sigma", 0.0))
        if sigma < 0.0:
            raise ValueError("noise sigma must be non-negative.")
        selection = _noise_selection_indices(
            noise.get("selection", "all"), topology_arrays, n_atoms,
        )
        distribution = str(noise.get("distribution", "gaussian")).strip().lower()
        if distribution not in {"gaussian", "normal"}:
            raise ValueError("noise distribution must be 'gaussian' or 'normal'.")

        if n_noisy:
            noisy = np.broadcast_to(pos, (n_noisy, n_atoms, 3)).copy()
            if sigma > 0.0 and selection.size:
                seed = int(noise.get("seed", 0))
                entropy = [seed & 0xFFFFFFFF, int(frame_id) & 0xFFFFFFFF]
                rng = np.random.default_rng(np.random.SeedSequence(entropy))
                shape = (n_noisy, int(selection.size), 3)
                delta = rng.normal(loc=0.0, scale=sigma, size=shape)
                noisy[:, selection, :] += delta
                displacement_batch[next_slot: next_slot + n_noisy, selection, :] = delta
            positions_batch[next_slot: next_slot + n_noisy] = noisy

        box_batch = _normalize_box_batch(box, total_samples)
        if bool(noise.get("wrap", False)):
            positions_batch = _wrap_positions_in_box(positions_batch, box_batch)

        force_mix_ratio = float(noise.get("force_mix_ratio", 0.0))
        if not 0.0 <= force_mix_ratio <= 1.0:
            raise ValueError("noise force_mix_ratio must be between 0 and 1.")

        noise_target = str(noise.get("target", "")).strip().lower()
        if noise_target == "dsm":
            if force_mix_ratio > 0.0:
                raise ValueError("noise force_mix_ratio is not compatible with DSM target.")
            if sigma <= 0.0:
                raise ValueError("DSM noise target requires sigma > 0.")
            beta = float(noise.get("beta", 0.0))
            if beta <= 0.0:
                raise ValueError("DSM noise target requires beta > 0.")
            reference_force_batch = (
                -displacement_batch / (beta * sigma * sigma)
            ).astype(np.float32, copy=False)
        elif reference_forces is None:
            if force_mix_ratio > 0.0:
                raise ValueError("noise force_mix_ratio requires reference forces.")
            reference_force_batch = None
        else:
            reference_force_batch = np.broadcast_to(
                np.asarray(reference_forces),
                (total_samples,) + np.asarray(reference_forces).shape,
            ).copy()
            if force_mix_ratio > 0.0 and n_noisy > 0:
                if sigma <= 0.0:
                    raise ValueError("noise force_mix_ratio requires sigma > 0.")
                beta = float(noise.get("beta", 0.0))
                if beta <= 0.0:
                    raise ValueError("noise force_mix_ratio requires beta > 0.")
                score_batch = -displacement_batch / (beta * sigma * sigma)
                score_like_ref = _batch_atom_values_like_reference(
                    score_batch,
                    reference_force_batch,
                    topology_arrays,
                    n_atoms=n_atoms,
                )
                mix_mask = np.zeros_like(displacement_batch, dtype=bool)
                noisy_start = 1 if include_original else 0
                if selection.size:
                    mix_mask[noisy_start : noisy_start + n_noisy, selection, :] = True
                mask_like_ref = _batch_atom_values_like_reference(
                    mix_mask,
                    reference_force_batch,
                    topology_arrays,
                    n_atoms=n_atoms,
                ).astype(bool, copy=False)
                mixed = reference_force_batch.astype(np.float32, copy=True)
                score_like_ref = score_like_ref.astype(np.float32, copy=False)
                mixed[mask_like_ref] = (
                    (1.0 - force_mix_ratio) * mixed[mask_like_ref]
                    + force_mix_ratio * score_like_ref[mask_like_ref]
                )
                reference_force_batch = mixed
        weights = np.full(total_samples, 1.0 / float(total_samples), dtype=np.float64)
        return (frame_ids, positions_batch, box_batch, reference_force_batch), weights

    def _shared_step_requests(self, steps: Sequence[dict]) -> Dict[str, bool]:
        request = {flag: False for flag in _REQUEST_FLAGS}
        for step in steps:
            request.update(
                {
                    key: request[key] or value
                    for key, value in step_request(step).items()
                }
            )
        return request

    def _reduce_step_partials(
        self,
        step: dict,
        local_result: Dict[str, Any],
        *,
        discrete_frame_ids: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Reduce per-step local partials across ranks.

        Parameters
        ----------
        discrete_frame_ids
            When ``True``, the caller is in discrete ``frame_ids`` mode (the
            input ids may arrive in arbitrary order). After concatenation the
            reducer sorts every stacked array by ``frame_ids`` so the returned
            stacks are deterministic with respect to frame ordering. When
            ``False`` the contiguous-range slicing guarantees rank-ordered
            gather already yields sorted output, and no resort is performed
            (keeps cost at O(size log size) instead of O(N log N)).
        """
        comm = self.comm
        rank = 0 if comm is None else comm.Get_rank()
        size = 1 if comm is None else comm.Get_size()
        plan = step_reduce_plan(step)
        sum_keys = list(plan["sum"])
        max_keys = set(plan["max"])
        stack_keys = list(plan["stack"])
        dict_sum_keys = list(plan.get("dict_sum", ()))
        dict_update_keys = list(plan.get("dict_update", ()))

        if comm is not None and size > 1:
            from mpi4py import MPI

            reduced: Optional[Dict[str, Any]] = {} if rank == 0 else None
            reduced_keys = set(sum_keys)
            for key in sum_keys:
                op = MPI.MAX if key in max_keys else MPI.SUM
                value = comm.reduce(local_result[key], op=op, root=0)
                if rank == 0:
                    reduced[key] = value
            for key in max_keys:
                if key in reduced_keys:
                    continue
                value = comm.reduce(local_result[key], op=MPI.MAX, root=0)
                if rank == 0:
                    reduced[key] = value
            for key in stack_keys:
                gathered = comm.gather(local_result[key], root=0)
                if rank == 0:
                    arrays = [
                        np.asarray(item)
                        for item in gathered
                        if np.asarray(item).size
                    ]
                    if arrays:
                        reduced[key] = np.concatenate(arrays, axis=0)
                    else:
                        reduced[key] = np.asarray(local_result[key])
            for key in dict_sum_keys:
                gathered = comm.gather(local_result.get(key, {}), root=0)
                if rank == 0:
                    merged: Dict[Any, Any] = {}
                    for item in gathered:
                        for subkey, value in dict(item).items():
                            arr = np.asarray(value)
                            if subkey in merged:
                                merged[subkey] = np.asarray(merged[subkey]) + arr
                            else:
                                merged[subkey] = arr.copy()
                    reduced[key] = merged
            for key in dict_update_keys:
                gathered = comm.gather(local_result.get(key, {}), root=0)
                if rank == 0:
                    merged: Dict[Any, Any] = {}
                    for item in gathered:
                        merged.update(dict(item))
                    reduced[key] = merged
        else:
            reduced = dict(local_result)

        if (
            reduced is not None
            and discrete_frame_ids
            and stack_keys
            and "frame_ids" in reduced
        ):
            ids = np.asarray(reduced["frame_ids"])
            if ids.size > 1:
                order = np.argsort(ids, kind="stable")
                if not np.array_equal(order, np.arange(ids.size)):
                    for key in stack_keys:
                        arr = reduced.get(key)
                        if arr is None:
                            continue
                        arr_np = np.asarray(arr)
                        if arr_np.ndim >= 1 and arr_np.shape[0] == ids.size:
                            reduced[key] = arr_np[order]
        return reduced

    def _preprocess_cdfm_zbx_steps(
        self,
        *,
        one_pass_steps: Sequence[Dict[str, Any]],
        work_dir: Path,
        init_topology: str,
        rank: int,
        size: int,
        forcefield_snapshot: Forcefield,
        topology_arrays: TopologyArrays,
        pair_type_list: Sequence[Any],
        pair_cutoff: Optional[float],
        sel_indices: Optional[np.ndarray],
        exclude_option: str,
    ) -> None:
        """Populate per-step ``y_eff`` for ``cdfm_zbx`` before state init."""
        import MDAnalysis as mda

        # cdfm_zbx preprocessing: rank 0 computes the per-replica single-frame
        # y_eff = y_ref - f_theta_cg_only(R_init) from the replica's own init
        # data file and the paired reference-force .npy, then broadcasts it to
        # every rank. The forcefield mask is temporarily flipped to CG-only
        # during the baseline force call and fully restored before the main
        # frame loop, so downstream reducers see the original training mask.
        for step in one_pass_steps:
            mode = str(step["step_mode"]).strip().lower()
            if mode != "cdfm_zbx":
                continue
            if "init_force_path" not in step:
                raise ValueError(
                    "cdfm_zbx step requires 'init_force_path' pointing to the "
                    "CG-mapped reference-force .npy paired with this replica's "
                    "init config."
                )
            if rank == 0:
                init_force_path = Path(str(step["init_force_path"]))
                if not init_force_path.is_absolute():
                    init_force_path = work_dir / init_force_path
                y_ref = np.load(init_force_path).astype(np.float64, copy=False)
                n_real = int(len(topology_arrays.real_site_indices))
                if y_ref.shape == (n_real, 3):
                    y_ref_flat = y_ref.reshape(-1)
                elif y_ref.shape == (3 * n_real,):
                    y_ref_flat = y_ref
                else:
                    raise ValueError(
                        f"cdfm_zbx init_force_path={init_force_path!s} has shape "
                        f"{tuple(y_ref.shape)}; expected ({n_real}, 3) or "
                        f"({3 * n_real},) where n_real=len(real_site_indices)."
                    )

                init_universe = mda.Universe(
                    init_topology,
                    format="DATA",
                    topology_format="DATA",
                )
                init_positions = np.asarray(
                    init_universe.atoms.positions, dtype=np.float64
                )
                init_box = np.asarray(
                    init_universe.dimensions, dtype=np.float64
                )
                init_frame_tuple = (
                    int(step.get("init_frame_id", 0)),
                    init_positions,
                    init_box,
                    None,
                )

                original_param_mask = forcefield_snapshot.param_mask.copy()
                cg_mask = forcefield_snapshot.real_mask.copy()
                forcefield_snapshot.param_mask = cg_mask
                try:
                    baseline_request = {flag: False for flag in _REQUEST_FLAGS}
                    baseline_request["need_force_value"] = True
                    baseline_result = self.compute(
                        request=baseline_request,
                        frame=init_frame_tuple,
                        topology_arrays=topology_arrays,
                        forcefield_snapshot=forcefield_snapshot,
                        frame_weight=1.0,
                        interaction_mask=forcefield_snapshot.key_mask,
                        pair_type_list=pair_type_list,
                        pair_cutoff=pair_cutoff,
                        sel_indices=sel_indices,
                        exclude_option=exclude_option,
                    )
                    f_model_full = np.asarray(
                        baseline_result["force"], dtype=np.float64
                    ).ravel()
                    if f_model_full.size == y_ref_flat.size:
                        f_model_real = f_model_full
                    else:
                        f_model_real = slice_observed_rows(
                            f_model_full,
                            topology_arrays.real_site_indices,
                        )
                finally:
                    forcefield_snapshot.param_mask = original_param_mask

                y_eff_single = y_ref_flat - f_model_real
            else:
                y_eff_single = None
            if self.comm is not None and size > 1:
                y_eff_single = self.comm.bcast(y_eff_single, root=0)
            step["y_eff"] = y_eff_single
            step.pop("init_force_path", None)

    def run_post(
        self,
        spec: dict,
    ) -> None:
        """Run one-pass post-processing for one trajectory.

        Parameters
        ----------
        spec
            Serialized post-processing specification. Observable-cache controls
            are also read from ``spec``:

            - ``collect_observables``: if ``True``, store one
              ``FrameCache`` object per processed frame on each rank.
            - ``gather_observables``: if ``True`` and MPI is active, gather the
              per-rank observable caches to rank 0 and merge them into a single
              ``TrajectoryCache``. This option requires
              ``collect_observables=True``.
            - ``observables_output_file``: optional pickle output path for the
              trajectory cache. Relative paths are resolved under
              ``spec['work_dir']``. In MPI mode, writing a single cache file
              requires ``gather_observables=True``.

        Notes
        -----
        Observable caches follow the same file-oriented convention as other
        post-processing outputs: when an output path is provided, rank 0 writes
        the pickle payload and ``run_post()`` itself returns ``None``.
        """
        import MDAnalysis as mda

        from ..io.trajectory import iter_frames
        from ..topology.topology_array import collect_topology_arrays

        collect_observables = bool(spec.get("collect_observables", False))
        gather_observables = bool(spec.get("gather_observables", False))
        observables_output_file = spec.get("observables_output_file")

        if gather_observables and not collect_observables:
            raise ValueError(
                "spec['gather_observables']=True requires spec['collect_observables']=True."
            )
        if observables_output_file is not None and not collect_observables:
            raise ValueError(
                "spec['observables_output_file'] requires spec['collect_observables']=True."
            )

        comm = self.comm
        rank = 0 if comm is None else comm.Get_rank()
        size = 1 if comm is None else comm.Get_size()
        shared_spec = {key: value for key, value in spec.items() if key != "steps"}
        expected_mpi_size = shared_spec.get("expected_mpi_size")
        if expected_mpi_size is not None and int(expected_mpi_size) != int(size):
            raise RuntimeError(
                f"MPI size mismatch: launcher expected {int(expected_mpi_size)} rank(s), "
                f"but mpi4py COMM_WORLD has size {int(size)}."
            )
        noise_spec = shared_spec.get("noise")
        if noise_spec is not None:
            if not isinstance(noise_spec, Mapping):
                raise ValueError("spec['noise'] must be a mapping when provided.")
            if not bool(noise_spec.get("enabled", True)):
                noise_spec = None
        all_steps = [dict(step) for step in spec.get("steps", [])]
        one_pass_steps = [step for step in all_steps if canonical_step_mode(step) != "rdf"]
        rdf_steps = [step for step in all_steps if canonical_step_mode(step) == "rdf"]
        perf_trace = bool(shared_spec.get("perf_trace", _env_flag("ACECG_POST_PERF_TRACE")))
        trace_all_ranks = bool(
            shared_spec.get(
                "perf_trace_all_ranks",
                _env_flag("ACECG_POST_PERF_TRACE_ALL_RANKS"),
            )
        )
        local_timing: Dict[str, Any] = {"rank": rank, "size": size}
        local_observables = (
            TrajectoryCache() if collect_observables else None
        )

        work_dir = Path(shared_spec["work_dir"])
        heartbeat_interval = _int_spec_value(shared_spec, "heartbeat_interval", 0)
        heartbeat_start = time.monotonic()
        _trace(perf_trace, rank, f"run_post start, MPI size={size}", all_ranks=trace_all_ranks)
        t0 = time.monotonic()
        with open(shared_spec["forcefield_path"], "rb") as handle:
            forcefield_snapshot = pickle.load(handle)
        _add_timing(local_timing, "load_forcefield", time.monotonic() - t0)

        if rank == 0:
            _trace(perf_trace, rank, "opening root universe", all_ranks=trace_all_ranks)
            t0 = time.monotonic()
            topology = str(shared_spec["topology"])
            traj = shared_spec["trajectory"]
            if isinstance(traj, str):
                traj = [traj]
            topology_format = shared_spec.get("topology_format")
            if topology_format is None and Path(topology).suffix.lower() == ".data":
                topology_format = "DATA"
            universe = mda.Universe(
                topology,
                *[str(path) for path in traj],
                format=shared_spec.get("trajectory_format", "LAMMPSDUMP"),
                topology_format=topology_format,
            )
            _add_timing(local_timing, "root_open_universe", time.monotonic() - t0)
            t0 = time.monotonic()
            topology_arrays = collect_topology_arrays(
                universe,
                exclude_bonded=shared_spec.get("exclude_bonded", "111"),
                exclude_option=shared_spec.get("exclude_option", "resid"),
                atom_type_name_aliases=shared_spec.get("atom_type_name_aliases"),
                vp_names=shared_spec.get("vp_names", shared_spec.get("vp_types")),
            )
            _add_timing(local_timing, "collect_topology_arrays", time.monotonic() - t0)
            t0 = time.monotonic()
            sel_indices = np.asarray(
                universe.select_atoms(str(shared_spec.get("sel", "all"))).indices,
                dtype=np.int32,
            )
            total_frames = len(universe.trajectory)
            _add_timing(local_timing, "select_atoms", time.monotonic() - t0)
        else:
            universe = None
            topology_arrays = None
            sel_indices = None
            total_frames = None

        if comm is not None and size > 1:
            t0 = time.monotonic()
            universe, topology_arrays, sel_indices, total_frames = comm.bcast(
                (universe, topology_arrays, sel_indices, total_frames)
                if rank == 0
                else None,
                root=0,
            )
            _add_timing(local_timing, "broadcast_shared_context", time.monotonic() - t0)

        if universe is None:
            raise RuntimeError("MPIComputeEngine.run_post() requires a local Universe.")

        # Frame distribution.
        # Two modes: (a) discrete frame_ids list, (b) contiguous range.
        # Mode (a) is opt-in via spec["frame_ids"] — it lets a caller process
        # an arbitrary non-contiguous subset of frames, e.g. a K-frame
        # subsample out of a longer trajectory.
        discrete_ids = shared_spec.get("frame_ids")
        noise_subsample_per_epoch = _noise_subsample_per_epoch_from_spec(shared_spec)
        use_discrete_frame_ids = discrete_ids is not None or noise_subsample_per_epoch > 0
        if use_discrete_frame_ids:
            all_selected_frame_ids = _selected_frame_ids_from_spec(
                shared_spec,
                int(total_frames),
            )
            all_ids = all_selected_frame_ids
            n_selected = len(all_ids)
            base_count, remainder = divmod(n_selected, size)
            local_count = base_count + (1 if rank < remainder else 0)
            local_offset = rank * base_count + min(rank, remainder)
            local_ids = all_ids[local_offset : local_offset + local_count]
            # Contiguous-range variables are unused on this path.
            local_start = local_end = every = None
            selected_frame_ids = list(local_ids)
        else:
            local_ids = None
            frame_start = (
                0 if shared_spec.get("frame_start") is None else int(shared_spec["frame_start"])
            )
            frame_end = (
                int(total_frames)
                if shared_spec.get("frame_end") is None
                else int(shared_spec["frame_end"])
            )
            every = int(shared_spec.get("every", 1))

            n_selected = len(range(frame_start, frame_end, every))
            base_count, remainder = divmod(n_selected, size)
            local_count = base_count + (1 if rank < remainder else 0)
            local_offset = rank * base_count + min(rank, remainder)
            local_start = frame_start + local_offset * every
            local_end = frame_start + (local_offset + local_count) * every
            selected_frame_ids = list(range(local_start, local_end, every))

        if heartbeat_interval > 0:
            _write_rank_heartbeat(
                work_dir,
                rank=rank,
                size=size,
                processed=0,
                local_total=local_count,
                global_total=n_selected,
                frame_id=None,
                start_time=heartbeat_start,
            )

        if shared_spec.get("frame_weight") is not None and shared_spec.get("frame_weight_file") is not None:
            raise ValueError("Specify only one of spec['frame_weight'] or spec['frame_weight_file'].")

        frame_weight_load_error = None
        loaded_frame_weight = None
        if rank == 0:
            try:
                loaded_frame_weight = _load_frame_weight_file(
                    shared_spec.get("frame_weight_file"),
                    work_dir=work_dir,
                )
            except Exception as exc:
                frame_weight_load_error = exc
        if comm is not None and size > 1:
            frame_weight_load_error = comm.bcast(frame_weight_load_error, root=0)
            if frame_weight_load_error is not None:
                raise frame_weight_load_error
            loaded_frame_weight = comm.bcast(loaded_frame_weight, root=0)
        elif frame_weight_load_error is not None:
            raise frame_weight_load_error

        frame_weight_all = _frame_weight_array_from_spec(
            shared_spec,
            int(total_frames),
            loaded_frame_weight=loaded_frame_weight,
        )
        if frame_weight_all is None:
            frame_weight_local = None
        else:
            frame_weight_local = frame_weight_all[np.asarray(selected_frame_ids, dtype=np.int64)]

        pair_cutoff = (
            None if shared_spec.get("cutoff") is None else float(shared_spec["cutoff"])
        )
        exclude_option = shared_spec.get("exclude_option", "none")

        # These are needed by the cdfm_zbx baseline preprocessing block below
        # (and reused in the main frame loop). They depend only on
        # ``forcefield_snapshot`` and ``shared_spec``, so they are safe to
        # compute here before per-step preprocessing.
        pair_type_list = [
            key
            for key in forcefield_snapshot.keys()
            if getattr(key, "style", None) == "pair"
        ]
        interaction_mask = getattr(forcefield_snapshot, "key_mask", None)

        self._preprocess_cdfm_zbx_steps(
            one_pass_steps=one_pass_steps,
            work_dir=work_dir,
            init_topology=str(shared_spec["topology"]),
            rank=rank,
            size=size,
            forcefield_snapshot=forcefield_snapshot,
            topology_arrays=topology_arrays,
            pair_type_list=pair_type_list,
            pair_cutoff=pair_cutoff,
            sel_indices=sel_indices,
            exclude_option=exclude_option,
        )
        # Refresh the cached interaction_mask in case the mask setter
        # produced a new key_mask object during preprocessing.
        interaction_mask = getattr(forcefield_snapshot, "key_mask", None)
        
        step_states = [
            init_step_state(step, forcefield_snapshot, topology_arrays) for step in one_pass_steps
        ]
        request = self._shared_step_requests(one_pass_steps)
        if noise_spec is not None and (collect_observables or request.get("need_frame_cache", False)):
            raise ValueError(
                "spec['noise'] batch processing does not support frame-cache or observables requests."
            )
        need_reference_forces = bool(request["need_reference_force"])

        if local_ids is not None:
            _trace(
                perf_trace,
                rank,
                f"frame loop start (discrete) local_count={len(local_ids)}",
                all_ranks=trace_all_ranks,
            )
            frame_iter = iter_frames(
                universe,
                frame_ids=local_ids,
                include_forces=need_reference_forces,
            )
        else:
            _trace(
                perf_trace,
                rank,
                f"frame loop start local_count={local_count} local_start={local_start} local_end={local_end}",
                all_ranks=trace_all_ranks,
            )
            frame_iter = iter_frames(
                universe,
                start=local_start,
                end=local_end,
                every=every,
                include_forces=need_reference_forces,
            )

        frame_iter_obj = iter(frame_iter)
        i = 0
        while True:
            t0 = time.monotonic()
            try:
                frame = next(frame_iter_obj)
            except StopIteration:
                break
            _add_timing(local_timing, "frame_fetch", time.monotonic() - t0)
            frame_total_start = time.monotonic()
            frame_fetch_sec = frame_total_start - t0
            frame_id, positions, box, reference_forces = frame
            local_timing["local_frame_count"] = int(local_timing.get("local_frame_count", 0)) + 1
            wi = 1.0 if frame_weight_local is None else float(frame_weight_local[i])

            if noise_spec is None:
                t_compute = time.monotonic()
                frame_result = self.compute(
                    request=request,
                    frame=frame,
                    topology_arrays=topology_arrays,
                    forcefield_snapshot=forcefield_snapshot,
                    frame_weight=wi,
                    interaction_mask=interaction_mask,
                    pair_type_list=pair_type_list,
                    pair_cutoff=pair_cutoff,
                    sel_indices=sel_indices,
                    exclude_option=exclude_option,
                    timing=local_timing,
                    return_observables=collect_observables,
                    frame_idx=frame_id,
                )
                compute_sec = time.monotonic() - t_compute
                noise_sec = 0.0
            else:
                t_noise = time.monotonic()
                frame_batch, sample_weights = self.add_noise(
                    frame, noise_spec, topology_arrays,
                )
                noise_sec = time.monotonic() - t_noise
                _add_timing(local_timing, "noise_generation", noise_sec)
                t_compute = time.monotonic()
                neighbor_mode = str(noise_spec.get("neighbor_mode", "shared")).strip().lower()
                use_reference_neighbor_frame = neighbor_mode == "skin"
                # Noisy replicas are same-frame samples: compute() handles
                # chunking, neighbor-cache policy, and reducer folding.
                frame_result = self.compute(
                    request=request,
                    frame=frame_batch,
                    topology_arrays=topology_arrays,
                    forcefield_snapshot=forcefield_snapshot,
                    frame_weight=wi,
                    frame_weights=sample_weights,
                    interaction_mask=interaction_mask,
                    pair_type_list=pair_type_list,
                    pair_cutoff=pair_cutoff,
                    sel_indices=sel_indices,
                    exclude_option=exclude_option,
                    timing=local_timing,
                    frame_idx=frame_id,
                    batch_size=noise_spec.get("batch_size"),
                    neighbor_mode=neighbor_mode,
                    neighbor_skin=float(noise_spec.get("neighbor_skin", 0.0) or 0.0),
                    neighbor_reference_positions=positions
                    if use_reference_neighbor_frame
                    else None,
                    neighbor_reference_box=box
                    if use_reference_neighbor_frame
                    else None,
                )
                compute_sec = time.monotonic() - t_compute
                _add_timing(local_timing, "compute_noisy_total", compute_sec)

            if local_observables is not None:
                frame_cache = frame_result.get("frame_observables", frame_result.get("frame_cache"))
                if frame_cache is None:
                    raise RuntimeError(
                        "collect_observables requested, but compute() did not return a frame cache."
                    )
                local_observables.add(frame_cache)

            t0 = time.monotonic()
            for step, state in zip(one_pass_steps, step_states):
                # Reducers see the same payload shape for ordinary frames and
                # noisy batches; compute() has already folded sample axes away.
                consume_step_payload(
                    step,
                    state,
                    payload=frame_result,
                    frame_weight=wi,
                    reference_force=reference_forces,
                )
            _add_timing(local_timing, "step_consume", time.monotonic() - t0)
            consume_sec = time.monotonic() - t0
            processed = i + 1
            frame_total_sec = time.monotonic() - frame_total_start
            _add_timing(local_timing, "frame_total", frame_total_sec)
            if perf_trace:
                local_timing.setdefault("frame_timings", []).append(
                    {
                        "frame_id": int(frame_id),
                        "fetch_sec": float(frame_fetch_sec),
                        "noise_sec": float(noise_sec),
                        "compute_sec": float(compute_sec),
                        "consume_sec": float(consume_sec),
                        "total_sec": float(frame_total_sec),
                    }
                )
            if heartbeat_interval > 0 and (
                processed == local_count or processed % heartbeat_interval == 0
            ):
                _write_rank_heartbeat(
                    work_dir,
                    rank=rank,
                    size=size,
                    processed=processed,
                    local_total=local_count,
                    global_total=n_selected,
                    frame_id=frame_id,
                    start_time=heartbeat_start,
                )
            i += 1

        _trace(perf_trace, rank, "frame loop finished", all_ranks=trace_all_ranks)

        for step, state in zip(one_pass_steps, step_states):
            t0 = time.monotonic()
            local_result = local_step_partials(step, state)
            reduced = self._reduce_step_partials(
                step,
                local_result,
                discrete_frame_ids=use_discrete_frame_ids,
            )
            _add_timing(local_timing, "reduce", time.monotonic() - t0)
            if rank != 0 or reduced is None:
                continue
            t0 = time.monotonic()
            result = finalize_step_root(step, reduced)
            if (
                isinstance(result, dict)
                and "step_index" in shared_spec
                and "step_index" not in result
            ):
                result["step_index"] = int(shared_spec["step_index"])
            output_path = Path(step["output_file"])
            if not output_path.is_absolute():
                output_path = work_dir / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _backup_existing_output(output_path)
            with open(output_path, "wb") as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            _add_timing(local_timing, "write_output", time.monotonic() - t0)

        merged_observables: Optional[TrajectoryCache]
        if local_observables is None:
            merged_observables = None
        elif comm is not None and size > 1 and gather_observables:
            gathered_caches = comm.gather(local_observables, root=0)
            if rank == 0:
                merged_observables = TrajectoryCache()
                for cache in gathered_caches:
                    merged_observables.merge(cache)
            else:
                merged_observables = None
        else:
            merged_observables = local_observables

        if observables_output_file is not None:
            if comm is not None and size > 1 and not gather_observables:
                raise ValueError(
                    "spec['observables_output_file'] requires spec['gather_observables']=True "
                    "when MPI size > 1 so that rank 0 can write a complete cache."
                )
            if rank == 0:
                if merged_observables is None:
                    raise RuntimeError(
                        "Observable cache output was requested, but no merged cache is available on root."
                    )
                output_path = Path(str(observables_output_file))
                if not output_path.is_absolute():
                    output_path = work_dir / output_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                _backup_existing_output(output_path)
                with open(output_path, "wb") as handle:
                    pickle.dump(merged_observables, handle, protocol=pickle.HIGHEST_PROTOCOL)
                _trace(
                    perf_trace,
                    rank,
                    f"wrote observables cache {output_path}",
                    all_ranks=trace_all_ranks,
                )

        if rank == 0 and rdf_steps:
            for step in rdf_steps:
                rdf_source_mode = str(step.get("rdf_source", "auto")).strip().lower()
                if rdf_source_mode not in {"auto", "cache", "universe"}:
                    raise ValueError(
                        f"rdf step rdf_source must be 'auto', 'cache', or 'universe', got {rdf_source_mode!r}"
                    )
                if rdf_source_mode == "cache":
                    if merged_observables is None:
                        raise ValueError(
                            "rdf step requested rdf_source='cache', but no merged observables cache is available."
                        )
                    rdf_source = merged_observables
                elif rdf_source_mode == "universe":
                    rdf_source = universe
                else:
                    rdf_source = merged_observables if merged_observables is not None else universe

                rdf_result = _run_rdf_step(
                    step,
                    source=rdf_source,
                    topology_arrays=topology_arrays,
                    forcefield_snapshot=forcefield_snapshot,
                    frame_weights=frame_weight_all,
                    default_cutoff=pair_cutoff,
                    default_sel_indices=sel_indices,
                    default_exclude_option=exclude_option,
                )
                output_path = Path(str(step["output_file"]))
                if not output_path.is_absolute():
                    output_path = work_dir / output_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                _backup_existing_output(output_path)
                with open(output_path, "wb") as handle:
                    pickle.dump(rdf_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                _trace(
                    perf_trace,
                    rank,
                    f"wrote rdf output {output_path}",
                    all_ranks=trace_all_ranks,
                )

        if perf_trace:
            if comm is not None and size > 1:
                gathered = comm.gather(local_timing, root=0)
            else:
                gathered = [local_timing]
            if rank == 0:
                report = _write_timing_report(
                    work_dir,
                    gathered,
                    metadata={
                        "size": size,
                        "n_steps": len(all_steps),
                        "need_reference_forces": need_reference_forces,
                    },
                )
                _trace(perf_trace, rank, f"wrote timing report {report}", all_ranks=trace_all_ranks)

        return None


if __name__ == "__main__":
    # Main function is the canonical entry for MPI-enabled post-processing.
    # It initializes mpi4py COMM_WORLD.

    # Outside callers must use MPI4Py COMM_WORLD to build this engine to enable MPI.
    # Otherwise it will run in serial mode.
    import sys

    if len(sys.argv) != 2:
        SCREEN_LOGGER.error("Usage: python -m %s.mpi_engine <spec.json>", __package__)
        sys.exit(1)

    spec_path = sys.argv[1]
    with open(spec_path, "r", encoding="utf-8") as handle:
        spec = json.load(handle)

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    except ImportError:
        warnings.warn(
            "mpi4py is not installed — running mpi_engine.__main__ in serial mode. "
            "Install mpi4py to enable MPI parallelism.",
            RuntimeWarning,
            stacklevel=1,
        )
        comm = None

    from .registry import build_default_engine

    engine = build_default_engine(comm=comm)
    engine.run_post(spec)
