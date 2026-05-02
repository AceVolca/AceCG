"""Distribution analysis for pair / bond / angle / dihedral interactions.

Computes RDF / PDF histograms over a trajectory or an MPI-produced
``TrajectoryCache``. The module provides both one-shot convenience wrappers
(``pair_distributions``, ``bond_distributions``, ``angle_distributions``,
``dihedral_distributions``, ``interaction_distributions``) and the underlying
state-machine interface (``init_distribution_state`` /
``accumulate_distribution_frame`` / ``finalize_distribution_state``) that can
share a single accumulator across multiple sources.

From Ace (merged 2026-04-23): this entire module was grafted from the Ace tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from MDAnalysis import Universe

from ..compute.frame_geometry import compute_frame_geometry
from ..io.trajectory import iter_frames
from ..topology.neighbor import compute_pairs_by_type
from ..topology.topology_array import TopologyArrays
from ..topology.types import InteractionKey


@dataclass(frozen=True)
class DistributionResult:
    """Per-interaction distribution result.

    Attributes
    ----------
    key : InteractionKey
        The interaction key for a pair, bond, angle, or dihedral.
    x : np.ndarray
        Bin centers, using distances for RDFs and angular coordinates for PDFs.
    values : np.ndarray
        Normalized distribution values.
    counts : np.ndarray
        Raw weighted histogram counts.
    edges : np.ndarray
        Bin edges matching ``x``.
    mode : str
        Either ``"rdf"`` or ``"pdf"``.
    variable : str
        One of ``"distance"``, ``"angle"``, or ``"dihedral"``.
    n_frames : int
        Number of frames accumulated.
    weight_sum : float
        Sum of frame weights used for normalization.
    meta : dict
        Per-key metadata such as atom type counts or instance counts.
    """

    key: InteractionKey
    x: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    edges: np.ndarray
    # "rdf" or "pdf"
    mode: str
    # "distance" | "angle" | "dihedral"
    variable: str
    n_frames: int
    weight_sum: float
    meta: dict


def _normalize_weights(
    total_frames: int,
    selected_frame_ids: Sequence[int],
    frame_weights: Optional[Sequence[float]],
) -> np.ndarray:
    """Return selected-frame weights from a full-trajectory 1D array."""
    n_selected = len(selected_frame_ids)
    if frame_weights is None:
        return np.ones(n_selected, dtype=np.float64)
    w_full = np.asarray(frame_weights, dtype=np.float64)
    if w_full.ndim != 1:
        raise ValueError(f"frame_weights must be a 1D array, got shape {w_full.shape}")
    if w_full.shape != (total_frames,):
        raise ValueError(
            f"frame_weights must have shape ({total_frames},), got {w_full.shape}"
        )
    return w_full[np.asarray(selected_frame_ids, dtype=np.int64)]


def _normalize_cache_weights(
    selected_frame_ids: Sequence[int],
    frame_weights: Optional[Sequence[float]],
) -> np.ndarray:
    """Return selected-frame weights for a cache source.

    For cache-backed analysis we only know the explicit frame ids present in the
    cache. When frame weights are provided they are still interpreted as a
    full-trajectory 1D array, so every selected frame id must be in range.
    """
    n_selected = len(selected_frame_ids)
    if frame_weights is None:
        return np.ones(n_selected, dtype=np.float64)
    w_full = np.asarray(frame_weights, dtype=np.float64)
    if w_full.ndim != 1:
        raise ValueError(f"frame_weights must be a 1D array, got shape {w_full.shape}")
    if n_selected == 0:
        return np.empty((0,), dtype=np.float64)
    sel = np.asarray(selected_frame_ids, dtype=np.int64)
    if np.any(sel < 0) or np.any(sel >= w_full.shape[0]):
        raise ValueError(
            "frame_weights is shorter than at least one frame_idx stored in the cache"
        )
    return w_full[sel]


def _interaction_keys_from_forcefield(
    forcefield_snapshot,
    interaction_keys: Optional[Iterable[InteractionKey]] = None,
) -> list[InteractionKey]:
    """Return the interaction keys to use (explicit arg wins over forcefield snapshot)."""
    if interaction_keys is not None:
        return list(interaction_keys)
    return list(forcefield_snapshot.keys())


def _binning(
    *,
    variable: str,
    nbins: int,
    x_max: Optional[float],
    periodic_dihedral: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(edges, centers)`` for a distance / angle / dihedral histogram."""
    if variable == "distance":
        if x_max is None:
            raise ValueError("x_max/r_max must be provided for distance distributions.")
        edges = np.linspace(0.0, float(x_max), int(nbins) + 1, dtype=np.float64)
    elif variable == "angle":
        edges = np.linspace(0.0, np.pi, int(nbins) + 1, dtype=np.float64)
    elif variable == "dihedral":
        if periodic_dihedral:
            edges = np.linspace(-np.pi, np.pi, int(nbins) + 1, dtype=np.float64)
        else:
            edges = np.linspace(0.0, np.pi, int(nbins) + 1, dtype=np.float64)
    else:
        raise ValueError(f"Unknown variable: {variable!r}")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _count_atoms_by_type(
    topology_arrays: TopologyArrays,
    sel_indices: np.ndarray,
) -> dict[int, int]:
    """Return ``{atom_type_code: count}`` restricted to ``sel_indices``."""
    codes = np.asarray(topology_arrays.atom_type_codes, dtype=np.int32)
    sel_codes = codes[np.asarray(sel_indices, dtype=np.int32)]
    uniq, cnt = np.unique(sel_codes, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, cnt)}


def _expected_shell_count_per_frame(
    n_i: int,
    n_j: int,
    volume: float,
    shell_vol: np.ndarray,
    same_type: bool,
) -> np.ndarray:
    """Ideal-gas expected pair count per shell for RDF normalisation."""
    if volume <= 0.0:
        raise ValueError(f"Invalid box volume: {volume}")
    if same_type:
        prefactor = n_i * max(n_i - 1, 0) / (2.0 * volume)
    else:
        prefactor = (n_i * n_j) / volume
    return prefactor * shell_vol


def _safe_pdf_from_counts(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Normalise a histogram to a PDF; returns zeros when total mass is zero."""
    total = float(np.sum(counts))
    widths = np.diff(edges)
    if total <= 0.0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts / (total * widths)


def _distance_pbc(ri: np.ndarray, rj: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Minimum-image distance between rows of ``ri`` and ``rj`` under orthorhombic PBC."""
    dr = np.asarray(rj - ri, dtype=np.float64)
    boxv = np.asarray(box[:3], dtype=np.float64)
    for dim in range(3):
        L = boxv[dim]
        if L > 0:
            dr[:, dim] -= np.round(dr[:, dim] / L) * L
    return np.sqrt(np.einsum("ij,ij->i", dr, dr))


def _angle_values(ra: np.ndarray, rb: np.ndarray, rc: np.ndarray) -> np.ndarray:
    """Row-wise bend angle (radians) for triplets ``(a, b, c)`` around the middle atom."""
    v1 = ra - rb
    v2 = rc - rb
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    cosang = np.divide(
        np.einsum("ij,ij->i", v1, v2),
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=denom > 0.0,
    )
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.arccos(cosang)


def _dihedral_values(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    r4: np.ndarray,
) -> np.ndarray:
    """Row-wise signed dihedral angle (radians) for quadruplets ``(1, 2, 3, 4)``."""
    b1 = r2 - r1
    b2 = r3 - r2
    b3 = r4 - r3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1, axis=1)
    n2_norm = np.linalg.norm(n2, axis=1)
    b2_norm = np.linalg.norm(b2, axis=1)

    n1u = np.divide(n1, n1_norm[:, None], out=np.zeros_like(n1), where=n1_norm[:, None] > 0.0)
    n2u = np.divide(n2, n2_norm[:, None], out=np.zeros_like(n2), where=n2_norm[:, None] > 0.0)
    b2u = np.divide(b2, b2_norm[:, None], out=np.zeros_like(b2), where=b2_norm[:, None] > 0.0)

    m1 = np.cross(n1u, b2u)

    x = np.einsum("ij,ij->i", n1u, n2u)
    y = np.einsum("ij,ij->i", m1, n2u)
    return np.arctan2(y, x)


def _as_cache_mapping(source: Any) -> Optional[Mapping[int, Any]]:
    """Return a frame-cache mapping if *source* looks like a TrajectoryCache."""
    if isinstance(source, Mapping):
        return source
    frames = getattr(source, "frames", None)
    if isinstance(frames, Mapping):
        return frames
    return None


def _selected_cache_frame_ids(
    frames: Mapping[int, Any],
    *,
    start: int,
    end: Optional[int],
    every: int,
) -> list[int]:
    """Return cached frame ids satisfying ``start <= fid < end`` with stride ``every``."""
    ids = sorted(int(fid) for fid in frames.keys())
    if end is None:
        end = (ids[-1] + 1) if ids else int(start)
    start_i = int(start)
    end_i = int(end)
    step = int(every)
    if step <= 0:
        raise ValueError("every must be a positive integer")
    return [
        fid for fid in ids
        if fid >= start_i and fid < end_i and ((fid - start_i) % step == 0)
    ]


def _accumulate_from_source(
    source: Any,
    state: dict[str, Any],
    topology_arrays: TopologyArrays,
    forcefield_snapshot,
    *,
    frame_weights: Optional[Sequence[float]],
    start: int,
    end: Optional[int],
    every: int,
    cutoff: float,
    exclude_option: str,
    sel_indices: Optional[np.ndarray],
) -> None:
    """Accumulate distributions from either a ``Universe`` or a ``TrajectoryCache``.

    Dispatches on ``source`` type: for a live ``Universe`` we iterate frames and
    build per-frame geometry on the fly; for a cache we replay the pre-computed
    ``FrameCache`` entries directly.
    """
    pair_keys = list(state.get("pair_keys", []))

    # Branch 1: raw MDAnalysis Universe -> compute pair neighbor lists and
    # frame geometry on the fly before feeding the distribution accumulator.
    if isinstance(source, Universe):
        if end is None:
            end = len(source.trajectory)
        selected_frame_ids = list(range(int(start), int(end), int(every)))
        weights = _normalize_weights(len(source.trajectory), selected_frame_ids, frame_weights)
        frame_iter = iter_frames(
            source,
            start=int(start),
            end=int(end),
            every=int(every),
            include_forces=False,
        )
        for iframe, frame in enumerate(frame_iter):
            _, positions, box, _ = frame
            pair_cache = None
            if pair_keys:
                pair_cache = compute_pairs_by_type(
                    positions=positions,
                    box=box,
                    topology_arrays=topology_arrays,
                    pair_type_list=pair_keys,
                    cutoff=float(cutoff),
                    sel_indices=sel_indices,
                    exclude_option=exclude_option,
                )
            geom = compute_frame_geometry(
                positions=positions,
                box=box,
                topology_arrays=topology_arrays,
                pair_cache=pair_cache,
            )
            accumulate_distribution_frame(state, geom, frame_weight=float(weights[iframe]))
        return

    # Branch 2: TrajectoryCache -> frames already carry cached geometry, so we
    # only need to select the requested frame ids and replay them.
    frames = _as_cache_mapping(source)
    if frames is None:
        raise TypeError(
            "source must be either an MDAnalysis.Universe or a TrajectoryCache-like "
            "object exposing a '.frames' mapping"
        )

    selected_frame_ids = _selected_cache_frame_ids(
        frames,
        start=int(start),
        end=end,
        every=int(every),
    )
    weights = _normalize_cache_weights(selected_frame_ids, frame_weights)
    for iframe, frame_idx in enumerate(selected_frame_ids):
        accumulate_distribution_frame(
            state,
            frames[int(frame_idx)],
            frame_weight=float(weights[iframe]),
        )



def init_distribution_state(
    topology_arrays: TopologyArrays,
    forcefield_snapshot,
    *,
    interaction_keys: Optional[Iterable[InteractionKey]] = None,
    mode_by_key: Optional[Mapping[InteractionKey, str]] = None,
    cutoff: float = 30.0,
    r_max: Optional[float] = None,
    nbins_pair: int = 200,
    nbins_bond: int = 200,
    nbins_angle: int = 180,
    nbins_dihedral: int = 180,
    sel_indices: Optional[np.ndarray] = None,
    angle_degrees: bool = True,
    dihedral_degrees: bool = True,
    dihedral_periodic: bool = True,
    default_pair_mode: str = "rdf",
    default_bonded_mode: str = "pdf",
) -> dict[str, Any]:
    """Initialize a one-pass distribution accumulator state."""
    keys = _interaction_keys_from_forcefield(forcefield_snapshot, interaction_keys)
    pair_keys = [k for k in keys if k.style == "pair"]
    bond_keys = [k for k in keys if k.style == "bond"]
    angle_keys = [k for k in keys if k.style == "angle"]
    dihedral_keys = [k for k in keys if k.style == "dihedral"]

    if sel_indices is None:
        sel_indices = np.arange(len(topology_arrays.atom_type_codes), dtype=np.int32)
    else:
        sel_indices = np.asarray(sel_indices, dtype=np.int32)

    state: dict[str, Any] = {
        "mode_by_key": {} if mode_by_key is None else dict(mode_by_key),
        "default_pair_mode": str(default_pair_mode),
        "default_bonded_mode": str(default_bonded_mode),
        "angle_degrees": bool(angle_degrees),
        "dihedral_degrees": bool(dihedral_degrees),
        "dihedral_periodic": bool(dihedral_periodic),
        "pair_keys": pair_keys,
        "bond_keys": bond_keys,
        "angle_keys": angle_keys,
        "dihedral_keys": dihedral_keys,
        "weight_sum": 0.0,
        "n_frames": 0,
    }

    if pair_keys:
        pair_r_max = float(cutoff) if r_max is None else float(r_max)
        pair_edges, pair_centers = _binning(
            variable="distance", nbins=nbins_pair, x_max=pair_r_max, periodic_dihedral=True
        )
        shell_vol = (4.0 / 3.0) * np.pi * (pair_edges[1:] ** 3 - pair_edges[:-1] ** 3)
        type_counts = _count_atoms_by_type(topology_arrays, sel_indices)
        name_to_code = topology_arrays.atom_type_name_to_code
        pair_meta = {}
        for key in pair_keys:
            code_i = int(name_to_code[str(key.types[0])])
            code_j = int(name_to_code[str(key.types[1])])
            pair_meta[key] = {
                "n_type_i": int(type_counts.get(code_i, 0)),
                "n_type_j": int(type_counts.get(code_j, 0)),
                "same_type": bool(code_i == code_j),
            }
        state.update({
            "pair_edges": pair_edges,
            "pair_centers": pair_centers,
            "pair_shell_vol": shell_vol,
            "pair_hist_by_key": {k: np.zeros(nbins_pair, dtype=np.float64) for k in pair_keys},
            "pair_expected_by_key": {k: np.zeros(nbins_pair, dtype=np.float64) for k in pair_keys},
            "pair_meta_by_key": pair_meta,
        })

    if bond_keys:
        bond_r_max = 30.0 if r_max is None else float(r_max)
        bond_edges, bond_centers = _binning(
            variable="distance", nbins=nbins_bond, x_max=bond_r_max, periodic_dihedral=True
        )
        bonds = np.asarray(topology_arrays.bonds, dtype=np.int32)
        key_index = np.asarray(topology_arrays.bond_key_index, dtype=np.int32)
        all_keys = list(topology_arrays.keys_bondtypes)
        bond_meta = {}
        for key in bond_keys:
            topo_idx = all_keys.index(key)
            bond_meta[key] = {"n_instances": int(np.count_nonzero(key_index == topo_idx))}
        state.update({
            "bond_edges": bond_edges,
            "bond_centers": bond_centers,
            "bond_hist_by_key": {k: np.zeros(nbins_bond, dtype=np.float64) for k in bond_keys},
            "bond_meta_by_key": bond_meta,
        })

    if angle_keys:
        angle_edges, angle_centers = _binning(
            variable="angle", nbins=nbins_angle, x_max=None, periodic_dihedral=True
        )
        angles = np.asarray(topology_arrays.angles, dtype=np.int32)
        key_index = np.asarray(topology_arrays.angle_key_index, dtype=np.int32)
        all_keys = list(topology_arrays.keys_angletypes)
        angle_meta = {}
        for key in angle_keys:
            topo_idx = all_keys.index(key)
            angle_meta[key] = {"n_instances": int(np.count_nonzero(key_index == topo_idx)), "degrees": bool(angle_degrees)}
        state.update({
            "angle_edges": angle_edges,
            "angle_centers": angle_centers,
            "angle_hist_by_key": {k: np.zeros(nbins_angle, dtype=np.float64) for k in angle_keys},
            "angle_meta_by_key": angle_meta,
        })

    if dihedral_keys:
        dihedral_edges, dihedral_centers = _binning(
            variable="dihedral", nbins=nbins_dihedral, x_max=None, periodic_dihedral=bool(dihedral_periodic)
        )
        dihedrals = np.asarray(topology_arrays.dihedrals, dtype=np.int32)
        key_index = np.asarray(topology_arrays.dihedral_key_index, dtype=np.int32)
        all_keys = list(topology_arrays.keys_dihedraltypes)
        dihedral_meta = {}
        for key in dihedral_keys:
            topo_idx = all_keys.index(key)
            dihedral_meta[key] = {
                "n_instances": int(np.count_nonzero(key_index == topo_idx)),
                "degrees": bool(dihedral_degrees),
                "periodic": bool(dihedral_periodic),
            }
        state.update({
            "dihedral_edges": dihedral_edges,
            "dihedral_centers": dihedral_centers,
            "dihedral_hist_by_key": {k: np.zeros(nbins_dihedral, dtype=np.float64) for k in dihedral_keys},
            "dihedral_meta_by_key": dihedral_meta,
        })

    return state


def accumulate_distribution_frame(
    state: dict[str, Any],
    frame_geometry,
    *,
    frame_weight: float = 1.0,
) -> None:
    """Accumulate one frame of geometry values into a distribution state."""
    w = float(frame_weight)

    for key in state.get("pair_keys", []):
        dist = np.asarray(frame_geometry.pair_distances.get(key, np.empty(0, dtype=np.float64)), dtype=np.float64)
        if dist.size:
            hist, _ = np.histogram(dist, bins=state["pair_edges"])
            state["pair_hist_by_key"][key] += w * hist
        mode = state.get("mode_by_key", {}).get(key, state.get("default_pair_mode", "rdf")).lower()
        if mode == "rdf":
            meta = state["pair_meta_by_key"][key]
            volume = float(np.prod(np.asarray(frame_geometry.box[:3], dtype=np.float64)))
            state["pair_expected_by_key"][key] += w * _expected_shell_count_per_frame(
                n_i=int(meta["n_type_i"]),
                n_j=int(meta["n_type_j"]),
                volume=volume,
                shell_vol=np.asarray(state["pair_shell_vol"], dtype=np.float64),
                same_type=bool(meta["same_type"]),
            )
        elif mode != "pdf":
            raise ValueError(f"Pair key {key}: mode must be 'rdf' or 'pdf', got {mode!r}")

    for key in state.get("bond_keys", []):
        mode = state.get("mode_by_key", {}).get(key, state.get("default_bonded_mode", "pdf")).lower()
        if mode != "pdf":
            raise ValueError(f"Bond key {key}: only 'pdf' is supported, got {mode!r}")
        vals = np.asarray(frame_geometry.bond_distances.get(key, np.empty(0, dtype=np.float64)), dtype=np.float64)
        if vals.size:
            hist, _ = np.histogram(vals, bins=state["bond_edges"])
            state["bond_hist_by_key"][key] += w * hist

    for key in state.get("angle_keys", []):
        mode = state.get("mode_by_key", {}).get(key, state.get("default_bonded_mode", "pdf")).lower()
        if mode != "pdf":
            raise ValueError(f"Angle key {key}: only 'pdf' is supported, got {mode!r}")
        vals = np.asarray(frame_geometry.angle_values.get(key, np.empty(0, dtype=np.float64)), dtype=np.float64)
        if vals.size:
            hist, _ = np.histogram(vals, bins=state["angle_edges"])
            state["angle_hist_by_key"][key] += w * hist

    for key in state.get("dihedral_keys", []):
        mode = state.get("mode_by_key", {}).get(key, state.get("default_bonded_mode", "pdf")).lower()
        if mode != "pdf":
            raise ValueError(f"Dihedral key {key}: only 'pdf' is supported, got {mode!r}")
        vals = np.asarray(frame_geometry.dihedral_values.get(key, np.empty(0, dtype=np.float64)), dtype=np.float64)
        if vals.size:
            if not bool(state.get("dihedral_periodic", True)):
                vals = np.abs(vals)
            hist, _ = np.histogram(vals, bins=state["dihedral_edges"])
            state["dihedral_hist_by_key"][key] += w * hist

    state["weight_sum"] = float(state.get("weight_sum", 0.0)) + w
    state["n_frames"] = int(state.get("n_frames", 0)) + 1


def finalize_distribution_state(state: dict[str, Any]) -> Dict[InteractionKey, DistributionResult]:
    """Finalize a one-pass distribution accumulator state."""
    results: Dict[InteractionKey, DistributionResult] = {}
    n_frames = int(state.get("n_frames", 0))
    weight_sum = float(state.get("weight_sum", 0.0))
    mode_by_key = state.get("mode_by_key", {})

    for key in state.get("pair_keys", []):
        counts = np.asarray(state["pair_hist_by_key"][key], dtype=np.float64)
        mode = mode_by_key.get(key, state.get("default_pair_mode", "rdf")).lower()
        if mode == "rdf":
            denom = np.asarray(state["pair_expected_by_key"][key], dtype=np.float64)
            values = np.divide(counts, denom, out=np.zeros_like(counts, dtype=np.float64), where=denom > 0.0)
        elif mode == "pdf":
            values = _safe_pdf_from_counts(counts, np.asarray(state["pair_edges"], dtype=np.float64))
        else:
            raise ValueError(f"Pair key {key}: mode must be 'rdf' or 'pdf', got {mode!r}")
        meta = state["pair_meta_by_key"][key]
        results[key] = DistributionResult(
            key=key,
            x=np.asarray(state["pair_centers"], dtype=np.float64).copy(),
            values=values,
            counts=counts.copy(),
            edges=np.asarray(state["pair_edges"], dtype=np.float64).copy(),
            mode=mode,
            variable="distance",
            n_frames=n_frames,
            weight_sum=weight_sum,
            meta={"n_type_i": int(meta["n_type_i"]), "n_type_j": int(meta["n_type_j"])},
        )

    for key in state.get("bond_keys", []):
        counts = np.asarray(state["bond_hist_by_key"][key], dtype=np.float64)
        values = _safe_pdf_from_counts(counts, np.asarray(state["bond_edges"], dtype=np.float64))
        results[key] = DistributionResult(
            key=key,
            x=np.asarray(state["bond_centers"], dtype=np.float64).copy(),
            values=values,
            counts=counts.copy(),
            edges=np.asarray(state["bond_edges"], dtype=np.float64).copy(),
            mode="pdf",
            variable="distance",
            n_frames=n_frames,
            weight_sum=weight_sum,
            meta=dict(state["bond_meta_by_key"][key]),
        )

    angle_degrees = bool(state.get("angle_degrees", True))
    angle_edges = np.asarray(state.get("angle_edges", np.empty(0)), dtype=np.float64)
    angle_centers = np.asarray(state.get("angle_centers", np.empty(0)), dtype=np.float64)
    plot_angle_edges = np.degrees(angle_edges) if angle_degrees else angle_edges
    plot_angle_centers = np.degrees(angle_centers) if angle_degrees else angle_centers
    for key in state.get("angle_keys", []):
        counts = np.asarray(state["angle_hist_by_key"][key], dtype=np.float64)
        values = _safe_pdf_from_counts(counts, plot_angle_edges)
        results[key] = DistributionResult(
            key=key,
            x=plot_angle_centers.copy(),
            values=values,
            counts=counts.copy(),
            edges=plot_angle_edges.copy(),
            mode="pdf",
            variable="angle",
            n_frames=n_frames,
            weight_sum=weight_sum,
            meta=dict(state["angle_meta_by_key"][key]),
        )

    dihedral_degrees = bool(state.get("dihedral_degrees", True))
    dihedral_edges = np.asarray(state.get("dihedral_edges", np.empty(0)), dtype=np.float64)
    dihedral_centers = np.asarray(state.get("dihedral_centers", np.empty(0)), dtype=np.float64)
    plot_dihedral_edges = np.degrees(dihedral_edges) if dihedral_degrees else dihedral_edges
    plot_dihedral_centers = np.degrees(dihedral_centers) if dihedral_degrees else dihedral_centers
    for key in state.get("dihedral_keys", []):
        counts = np.asarray(state["dihedral_hist_by_key"][key], dtype=np.float64)
        values = _safe_pdf_from_counts(counts, plot_dihedral_edges)
        results[key] = DistributionResult(
            key=key,
            x=plot_dihedral_centers.copy(),
            values=values,
            counts=counts.copy(),
            edges=plot_dihedral_edges.copy(),
            mode="pdf",
            variable="dihedral",
            n_frames=n_frames,
            weight_sum=weight_sum,
            meta=dict(state["dihedral_meta_by_key"][key]),
        )

    return results


def pair_distributions(
    source: Any,
    topology_arrays: TopologyArrays,
    forcefield_snapshot,
    *,
    pair_keys: Optional[Iterable[InteractionKey]] = None,
    frame_weights: Optional[Sequence[float]] = None,
    start: int = 0,
    end: Optional[int] = None,
    every: int = 1,
    cutoff: float = 30.0,
    r_max: Optional[float] = None,
    nbins: int = 200,
    exclude_option: str = "resid",
    sel_indices: Optional[np.ndarray] = None,
    mode_by_key: Optional[Mapping[InteractionKey, str]] = None,
    default_mode: str = "rdf",
) -> Dict[InteractionKey, DistributionResult]:
    """Compute pair-style distributions (RDF or PDF) for selected pair types.

    ``source`` may be an ``MDAnalysis.Universe`` or a ``TrajectoryCache``.
    ``default_mode`` selects ``"rdf"`` (ideal-gas normalisation) or ``"pdf"``
    (probability density) for keys missing from ``mode_by_key``.
    """
    keys = [
        k for k in _interaction_keys_from_forcefield(forcefield_snapshot, pair_keys)
        if k.style == "pair"
    ]
    if not keys:
        return {}
    return interaction_distributions(
        source,
        topology_arrays,
        forcefield_snapshot,
        interaction_keys=keys,
        frame_weights=frame_weights,
        mode_by_key=mode_by_key,
        start=start,
        end=end,
        every=every,
        cutoff=cutoff,
        r_max=r_max,
        nbins_pair=nbins,
        exclude_option=exclude_option,
        sel_indices=sel_indices,
        default_pair_mode=default_mode,
        default_bonded_mode="pdf",
    )


def bond_distributions(
    source: Any,
    topology_arrays: TopologyArrays,
    forcefield_snapshot,
    *,
    bond_keys: Optional[Iterable[InteractionKey]] = None,
    frame_weights: Optional[Sequence[float]] = None,
    start: int = 0,
    end: Optional[int] = None,
    every: int = 1,
    nbins: int = 200,
    r_max: Optional[float] = None,
    mode_by_key: Optional[Mapping[InteractionKey, str]] = None,
    default_mode: str = "pdf",
) -> Dict[InteractionKey, DistributionResult]:
    """Compute bond-length distributions for the selected bond types (PDF)."""
    keys = [
        k for k in _interaction_keys_from_forcefield(forcefield_snapshot, bond_keys)
        if k.style == "bond"
    ]
    if not keys:
        return {}
    return interaction_distributions(
        source,
        topology_arrays,
        forcefield_snapshot,
        interaction_keys=keys,
        frame_weights=frame_weights,
        mode_by_key=mode_by_key,
        start=start,
        end=end,
        every=every,
        r_max=r_max,
        nbins_bond=nbins,
        default_pair_mode="rdf",
        default_bonded_mode=default_mode,
    )


def angle_distributions(
    source: Any,
    topology_arrays: TopologyArrays,
    forcefield_snapshot,
    *,
    angle_keys: Optional[Iterable[InteractionKey]] = None,
    frame_weights: Optional[Sequence[float]] = None,
    start: int = 0,
    end: Optional[int] = None,
    every: int = 1,
    nbins: int = 180,
    degrees: bool = True,
    mode_by_key: Optional[Mapping[InteractionKey, str]] = None,
    default_mode: str = "pdf",
) -> Dict[InteractionKey, DistributionResult]:
    """Compute bend-angle distributions for the selected angle types (PDF)."""
    keys = [
        k for k in _interaction_keys_from_forcefield(forcefield_snapshot, angle_keys)
        if k.style == "angle"
    ]
    if not keys:
        return {}
    return interaction_distributions(
        source,
        topology_arrays,
        forcefield_snapshot,
        interaction_keys=keys,
        frame_weights=frame_weights,
        mode_by_key=mode_by_key,
        start=start,
        end=end,
        every=every,
        nbins_angle=nbins,
        angle_degrees=degrees,
        default_pair_mode="rdf",
        default_bonded_mode=default_mode,
    )


def dihedral_distributions(
    source: Any,
    topology_arrays: TopologyArrays,
    forcefield_snapshot,
    *,
    dihedral_keys: Optional[Iterable[InteractionKey]] = None,
    frame_weights: Optional[Sequence[float]] = None,
    start: int = 0,
    end: Optional[int] = None,
    every: int = 1,
    nbins: int = 180,
    degrees: bool = True,
    periodic: bool = True,
    mode_by_key: Optional[Mapping[InteractionKey, str]] = None,
    default_mode: str = "pdf",
) -> Dict[InteractionKey, DistributionResult]:
    """Compute dihedral-angle distributions for the selected dihedral types (PDF).

    When ``periodic`` is False, dihedrals are folded to ``[0, pi]`` (abs-value);
    otherwise the full ``[-pi, pi]`` range is used.
    """
    keys = [
        k for k in _interaction_keys_from_forcefield(forcefield_snapshot, dihedral_keys)
        if k.style == "dihedral"
    ]
    if not keys:
        return {}
    return interaction_distributions(
        source,
        topology_arrays,
        forcefield_snapshot,
        interaction_keys=keys,
        frame_weights=frame_weights,
        mode_by_key=mode_by_key,
        start=start,
        end=end,
        every=every,
        nbins_dihedral=nbins,
        dihedral_degrees=degrees,
        dihedral_periodic=periodic,
        default_pair_mode="rdf",
        default_bonded_mode=default_mode,
    )


def interaction_distributions(
    source: Any,
    topology_arrays: TopologyArrays,
    forcefield_snapshot,
    *,
    interaction_keys: Optional[Iterable[InteractionKey]] = None,
    frame_weights: Optional[Sequence[float]] = None,
    mode_by_key: Optional[Mapping[InteractionKey, str]] = None,
    start: int = 0,
    end: Optional[int] = None,
    every: int = 1,
    cutoff: float = 30.0,
    r_max: Optional[float] = None,
    nbins_pair: int = 200,
    nbins_bond: int = 200,
    nbins_angle: int = 180,
    nbins_dihedral: int = 180,
    exclude_option: str = "resid",
    sel_indices: Optional[np.ndarray] = None,
    angle_degrees: bool = True,
    dihedral_degrees: bool = True,
    dihedral_periodic: bool = True,
    default_pair_mode: str = "rdf",
    default_bonded_mode: str = "pdf",
) -> Dict[InteractionKey, DistributionResult]:
    """Compute mixed-style distributions (pair/bond/angle/dihedral) in one pass.

    This is the most general entry point: any interaction key types present in
    the forcefield snapshot (or explicitly listed via ``interaction_keys``) are
    accumulated from a single source with a shared histogram state.
    """
    keys = _interaction_keys_from_forcefield(forcefield_snapshot, interaction_keys)

    if sel_indices is None:
        sel_indices = np.arange(len(topology_arrays.atom_type_codes), dtype=np.int32)
    else:
        sel_indices = np.asarray(sel_indices, dtype=np.int32)

    state = init_distribution_state(
        topology_arrays,
        forcefield_snapshot,
        interaction_keys=keys,
        mode_by_key=mode_by_key,
        cutoff=float(cutoff),
        r_max=r_max,
        nbins_pair=nbins_pair,
        nbins_bond=nbins_bond,
        nbins_angle=nbins_angle,
        nbins_dihedral=nbins_dihedral,
        sel_indices=sel_indices,
        angle_degrees=angle_degrees,
        dihedral_degrees=dihedral_degrees,
        dihedral_periodic=dihedral_periodic,
        default_pair_mode=default_pair_mode,
        default_bonded_mode=default_bonded_mode,
    )

    _accumulate_from_source(
        source,
        state,
        topology_arrays,
        forcefield_snapshot,
        frame_weights=frame_weights,
        start=int(start),
        end=end,
        every=int(every),
        cutoff=float(cutoff),
        exclude_option=exclude_option,
        sel_indices=sel_indices,
    )
    return finalize_distribution_state(state)


def multi_source_interaction_distributions(
    sources: Sequence[Any],
    topology_arrays: TopologyArrays,
    forcefield_snapshot,
    *,
    interaction_keys: Optional[Iterable[InteractionKey]] = None,
    frame_weights_list: Optional[Sequence[Optional[Sequence[float]]]] = None,
    mode_by_key: Optional[Mapping[InteractionKey, str]] = None,
    starts: int | Sequence[int] = 0,
    ends: Optional[int] | Sequence[Optional[int]] = None,
    everys: int | Sequence[int] = 1,
    cutoff: float = 30.0,
    r_max: Optional[float] = None,
    nbins_pair: int = 200,
    nbins_bond: int = 200,
    nbins_angle: int = 180,
    nbins_dihedral: int = 180,
    exclude_option: str = "resid",
    sel_indices: Optional[np.ndarray] = None,
    angle_degrees: bool = True,
    dihedral_degrees: bool = True,
    dihedral_periodic: bool = True,
    default_pair_mode: str = "rdf",
    default_bonded_mode: str = "pdf",
) -> Dict[InteractionKey, DistributionResult]:
    """Accumulate distributions over multiple trajectory/cache sources.

    This is intended for multi-window ensembles where each window contributes a
    separate trajectory/cache and optional per-frame weights. All sources are
    accumulated into the same histogram/normalization state, which avoids
    window-by-window renormalization artifacts.
    """
    source_list = list(sources)
    n_sources = len(source_list)
    if n_sources == 0:
        return {}

    keys = _interaction_keys_from_forcefield(forcefield_snapshot, interaction_keys)

    if sel_indices is None:
        sel_indices = np.arange(len(topology_arrays.atom_type_codes), dtype=np.int32)
    else:
        sel_indices = np.asarray(sel_indices, dtype=np.int32)

    def _broadcast_param(
        value: Any,
        *,
        name: str,
        default: Any,
    ) -> list[Any]:
        if value is None:
            return [default] * n_sources
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            seq = list(value)
            if len(seq) != n_sources:
                raise ValueError(f"{name} must have length {n_sources}, got {len(seq)}")
            return seq
        return [value] * n_sources

    weights_by_source = _broadcast_param(
        frame_weights_list,
        name="frame_weights_list",
        default=None,
    )
    starts_by_source = [int(v) for v in _broadcast_param(starts, name="starts", default=0)]
    ends_by_source = _broadcast_param(ends, name="ends", default=None)
    everys_by_source = [int(v) for v in _broadcast_param(everys, name="everys", default=1)]

    state = init_distribution_state(
        topology_arrays,
        forcefield_snapshot,
        interaction_keys=keys,
        mode_by_key=mode_by_key,
        cutoff=float(cutoff),
        r_max=r_max,
        nbins_pair=nbins_pair,
        nbins_bond=nbins_bond,
        nbins_angle=nbins_angle,
        nbins_dihedral=nbins_dihedral,
        sel_indices=sel_indices,
        angle_degrees=angle_degrees,
        dihedral_degrees=dihedral_degrees,
        dihedral_periodic=dihedral_periodic,
        default_pair_mode=default_pair_mode,
        default_bonded_mode=default_bonded_mode,
    )

    for source, weights, start_i, end_i, every_i in zip(
        source_list,
        weights_by_source,
        starts_by_source,
        ends_by_source,
        everys_by_source,
    ):
        _accumulate_from_source(
            source,
            state,
            topology_arrays,
            forcefield_snapshot,
            frame_weights=weights,
            start=start_i,
            end=end_i,
            every=every_i,
            cutoff=float(cutoff),
            exclude_option=exclude_option,
            sel_indices=sel_indices,
        )
    return finalize_distribution_state(state)
