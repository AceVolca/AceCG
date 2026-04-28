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
    canonical_step_mode,
    consume_step_frame,
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
        return [int(fid) for fid in discrete_ids]

    frame_start = 0 if shared_spec.get("frame_start") is None else int(shared_spec["frame_start"])
    frame_end = int(total_frames) if shared_spec.get("frame_end") is None else int(shared_spec["frame_end"])
    every = int(shared_spec.get("every", 1))
    return list(range(frame_start, frame_end, every))


def _frame_weight_mapping_from_spec(
    shared_spec: Dict[str, Any],
    total_frames: int,
) -> Optional[Dict[int, float]]:
    """Convert ``spec['frame_weight']`` into a ``{frame_idx: weight}`` mapping.

    The mapping form is convenient because ``analysis.rdf`` can consume it both
    for ``Universe`` input and for ``TrajectoryCache`` input, and
    unspecified frame indices naturally default to weight 1 when a mapping is
    used there.
    """
    raw = shared_spec.get("frame_weight")
    if raw is None:
        return None
    selected_ids = _selected_frame_ids_from_spec(shared_spec, total_frames)
    weights = np.asarray(raw, dtype=np.float64)
    if weights.shape != (len(selected_ids),):
        raise ValueError(
            "spec['frame_weight'] must have the same length as the selected frame set: "
            f"expected {len(selected_ids)}, got {weights.shape}"
        )
    return {int(fid): float(w) for fid, w in zip(selected_ids, weights)}


def _run_rdf_step(
    step: Dict[str, Any],
    *,
    source: Any,
    topology_arrays: TopologyArrays,
    forcefield_snapshot: Forcefield,
    frame_weights: Optional[Mapping[int, float]],
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
    """Return ``True`` if at least one enabled key of ``style`` exists.

    Parameters
    ----------
    keys
        All force-field keys available in the current snapshot.
    interaction_mask
        Optional per-key enable mask. When present, only keys with truthy mask
        values are considered active.
    style
        Interaction style name such as ``"pair"``, ``"bond"``, ``"angle"``,
        or ``"dihedral"``.
    """
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
        interaction_mask: Optional[np.ndarray] = None,
        pair_type_list: Optional[List[Any]] = None,
        pair_cutoff: Optional[float] = None,
        sel_indices: Optional[np.ndarray] = None,
        exclude_option: str = "resid",
        timing: Optional[Dict[str, Any]] = None,
        return_observables: bool = False,
        frame_idx: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Compute registered observables for one frame.

        Parameters
        ----------
        request
            Direct observable requests aggregated from all active trainers /
            post-processing steps.
        frame
            Frame tuple in the format ``(frame_id, positions, box,
            reference_forces)``.
        return_observables
            If ``True``, append a light-weight ``FrameCache`` object to the
            returned payload under the legacy key
            ``results["frame_observables"]``.
        frame_idx
            Optional explicit frame index to use when constructing
            ``FrameCache``. If omitted, the first entry of ``frame`` is
            used, which matches the normal iter_frames contract.

        Notes
        -----
        ``interaction_mask`` defaults to ``forcefield_snapshot.key_mask`` when
        not provided.
        """
        results: Dict[str, Any] = {}

        frame_id, positions, box, reference_forces = frame
        if frame_idx is None:
            frame_idx = int(frame_id)
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

        pair_cache = None
        if build_pairs:
            t0 = time.monotonic()
            pair_cache = compute_pairs_by_type(
                positions=positions,
                box=box,
                pair_type_list=pair_type_list,
                cutoff=float(pair_cutoff),
                topology_arrays=topology_arrays,
                sel_indices=sel_indices,
                exclude_option=exclude_option,
            )
            if timing is not None:
                _add_timing(timing, "pair_search", time.monotonic() - t0)

        t0 = time.monotonic()
        geom = compute_frame_geometry(
            positions,
            box,
            topology_arrays,
            interaction_mask=active_interaction_mask,
            pair_cache=pair_cache,
        )
        if timing is not None:
            _add_timing(timing, "geometry", time.monotonic() - t0)

        need_frame_cache = bool(request.get("need_frame_cache", False))
        if return_observables or need_frame_cache:
            frame_cache = geometry_to_observables(
                frame_idx=frame_idx,
                geom=geom,
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

        # Add energy-based observable requests.
        if (
            request["need_energy_value"]
            or request["need_energy_grad"]
            or request["need_energy_hessian"]
            or request["need_energy_grad_outer"]
        ):
            t0 = time.monotonic()
            results.update(
                energy(
                    geom,
                    forcefield_snapshot,
                    return_value=request["need_energy_value"],
                    return_grad=request["need_energy_grad"],
                    return_hessian=request["need_energy_hessian"],
                    return_grad_outer=request["need_energy_grad_outer"],
                )
            )
            if timing is not None:
                _add_timing(timing, "energy_kernel", time.monotonic() - t0)

        # Add force-based observable requests.
        if (
            request["need_force_value"]
            or request["need_force_grad"]
            or request["need_fm_stats"]
        ):
            t0 = time.monotonic()
            results.update(
                force(
                    geom,
                    forcefield_snapshot,
                    return_value=request["need_force_value"],
                    return_grad=request["need_force_grad"],
                    reference_force=reference_forces,
                    frame_weight=frame_weight,
                    return_fm_stats=request["need_fm_stats"],
                )
            )
            if timing is not None:
                _add_timing(timing, "force_kernel", time.monotonic() - t0)

        # Add other observable requests from registered functions.
        # A placeholder for future extensions. DO NOT DELETE THIS COMMENT.

        return results

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
        # TODO: HUMAN COMMENT: Consider moving this preprocessing to a separate function for clarity
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
                    f_model_real = f_model_full if f_model_full.size == y_ref_flat.size else slice_observed_rows(
                        f_model_full, topology_arrays.real_site_indices
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
        # END TODO - HUMAN COMMENT.

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
        if discrete_ids is not None:
            all_ids = [int(fid) for fid in discrete_ids]
            n_selected = len(all_ids)
            base_count, remainder = divmod(n_selected, size)
            local_count = base_count + (1 if rank < remainder else 0)
            local_offset = rank * base_count + min(rank, remainder)
            local_ids = all_ids[local_offset : local_offset + local_count]
            # Contiguous-range variables are unused on this path.
            local_start = local_end = every = None
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

        if shared_spec.get("frame_weight") is None:
            frame_weight_local = None
        else:
            frame_weight_all = np.asarray(shared_spec["frame_weight"], dtype=np.float32)
            if np.any(frame_weight_all < 0.0) or np.sum(frame_weight_all) <= 0.0:
                raise ValueError("frame_weight must be nonnegative with positive sum")
            frame_weight_local = frame_weight_all[local_offset : local_offset + local_count]

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

        for i, frame in enumerate(frame_iter):
            frame_id, positions, box, reference_forces = frame
            t0 = time.monotonic()
            _add_timing(local_timing, "frame_fetch", time.monotonic() - t0)
            local_timing["local_frame_count"] = int(local_timing.get("local_frame_count", 0)) + 1
            wi = 1.0 if frame_weight_local is None else float(frame_weight_local[i])

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

            if local_observables is not None:
                frame_cache = frame_result.get("frame_observables", frame_result.get("frame_cache"))
                if frame_cache is None:
                    raise RuntimeError(
                        "collect_observables requested, but compute() did not return a frame cache."
                    )
                local_observables.add(frame_cache)

            t0 = time.monotonic()
            for step, state in zip(one_pass_steps, step_states):
                consume_step_frame(
                    step,
                    state,
                    payload=frame_result,
                    frame_weight=wi,
                    reference_force=reference_forces,
                )
            _add_timing(local_timing, "step_consume", time.monotonic() - t0)

        _trace(perf_trace, rank, "frame loop finished", all_ranks=trace_all_ranks)

        for step, state in zip(one_pass_steps, step_states):
            t0 = time.monotonic()
            local_result = local_step_partials(step, state)
            reduced = self._reduce_step_partials(
                step,
                local_result,
                discrete_frame_ids=discrete_ids is not None,
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
            rdf_frame_weights = _frame_weight_mapping_from_spec(shared_spec, int(total_frames))
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
                    frame_weights=rdf_frame_weights,
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
