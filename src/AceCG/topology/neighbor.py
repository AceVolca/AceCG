"""Topology-aware pair-search helpers.

This module lives in the topology layer. It consumes raw coordinates and a
``TopologyArrays`` snapshot, applies topology-defined exclusions, and returns
atom-index pairs. It does not compute distances, energies, forces, or
``FrameGeometry``.

Current boundary:

- bonded exclusions (1-2, 1-3, 1-4) come from ``TopologyArrays`` and are
  always applied here;
- additional nonbonded exclusion is selected by ``exclude_option``;
- returned indices are always global atom indices, even when ``sel_indices``
  narrows the neighbor search;
- pair distances and per-key cutoff filtering happen later in
  ``compute_frame_geometry()`` and the registered compute kernels.

Canonical public entry points:

- ``compute_pairs_by_type()``: current engine path for pair interactions;
- ``compute_neighbor_list()``: generic adjacency helper, not used by the
  current compute core.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from MDAnalysis.lib.nsgrid import FastNS
import numpy as np

from .topology_array import TopologyArrays, _encode_pairs
from .types import InteractionKey

VALID_EXCLUDE_OPTIONS = frozenset({"resid", "molid", "none"})

# ---------------------------------------------------------------------------
# Exclusion helpers
# ---------------------------------------------------------------------------

def parse_exclude_option(exclude_option: str) -> str:
    """Normalize nonbonded exclusion option to one of the canonical strings."""
    token = str(exclude_option).strip().lower()
    aliases = {
        "residue": "resid",
        "same_resid": "resid",
        "same-resid": "resid",
        "molecule": "molid",
        "mol": "molid",
        "same_molid": "molid",
        "same-molid": "molid",
        "none": "none",
    }
    token = aliases.get(token, token)
    if token not in VALID_EXCLUDE_OPTIONS:
        raise ValueError(f"Invalid exclude_option: {exclude_option}")
    return token


def _normalize_selection_indices(
    topology_arrays: TopologyArrays,
    sel_indices: Optional[np.ndarray],
) -> np.ndarray:
    """Return a validated, duplicate-free global atom selection."""
    if sel_indices is None:
        return np.arange(topology_arrays.n_atoms, dtype=np.int32)
    indices = np.asarray(sel_indices, dtype=np.int32)
    if indices.ndim != 1:
        raise ValueError(f"sel_indices must be one-dimensional, got shape {indices.shape}")
    if np.any(indices < 0) or np.any(indices >= int(topology_arrays.n_atoms)):
        raise ValueError("sel_indices contains atom indices outside the topology")
    if np.unique(indices).size != indices.size:
        raise ValueError("sel_indices must not contain duplicate atom indices")
    return indices


def _bonded_exclusion_pairs(topology_arrays: TopologyArrays) -> np.ndarray:
    """Return unique canonical bonded-exclusion pairs as ``(i, j)`` rows."""
    parts = [
        np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        for pairs in (
            topology_arrays.exclude_12,
            topology_arrays.exclude_13,
            topology_arrays.exclude_14,
        )
        if np.asarray(pairs).size
    ]
    if not parts:
        return np.empty((0, 2), dtype=np.int32)
    encoded = np.unique(
        np.concatenate(
            [_encode_pairs(pairs, topology_arrays.n_atoms) for pairs in parts]
        )
    )
    return np.column_stack(
        (
            encoded // np.int32(topology_arrays.n_atoms),
            encoded % np.int32(topology_arrays.n_atoms),
        )
    ).astype(np.int32, copy=False)


def _group_labels_for_exclusion(
    topology_arrays: TopologyArrays,
    exclude_option: str,
) -> np.ndarray | None:
    """Return per-atom group labels for the canonical exclusion mode."""
    mode = parse_exclude_option(exclude_option)
    if mode == "resid":
        return np.asarray(topology_arrays.resids)[
            np.asarray(topology_arrays.atom_resindex, dtype=np.int32)
        ]
    if mode == "molid":
        return np.asarray(topology_arrays.molnums)
    return None


def count_eligible_pairs_by_type(
    topology_arrays: TopologyArrays,
    pair_type_list: Iterable[InteractionKey],
    *,
    sel_indices: Optional[np.ndarray] = None,
    exclude_option: str = "resid",
) -> dict[InteractionKey, int]:
    """Count topology-eligible unordered pairs without building an ``N^2`` table.

    The result follows the same selection, bonded-exclusion, residue, and
    molecule contract as :func:`compute_pairs_by_type`.  Group exclusions are
    counted combinatorially from per-group type populations.  Only the sparse
    bonded-exclusion arrays are materialized, and overlap with a group
    exclusion is removed before subtraction.
    """
    mode = parse_exclude_option(exclude_option)
    indices = _normalize_selection_indices(topology_arrays, sel_indices)
    keys = list(dict.fromkeys(pair_type_list))
    for key in keys:
        if key.style != "pair" or len(key.types) != 2:
            raise ValueError(f"Expected pair InteractionKey, got {key!r}")

    selected = np.zeros(topology_arrays.n_atoms, dtype=bool)
    selected[indices] = True
    type_codes = np.asarray(topology_arrays.atom_type_codes, dtype=np.int32)
    name_to_code = topology_arrays.atom_type_name_to_code
    selected_codes = type_codes[indices]

    counts_by_code = {
        int(code): int(count)
        for code, count in zip(*np.unique(selected_codes, return_counts=True))
    }
    eligible: dict[InteractionKey, int] = {}
    code_pairs: dict[InteractionKey, tuple[int | None, int | None]] = {}
    for key in keys:
        code_a = name_to_code.get(str(key.types[0]))
        code_b = name_to_code.get(str(key.types[1]))
        code_pairs[key] = (code_a, code_b)
        if code_a is None or code_b is None:
            eligible[key] = 0
            continue
        n_a = counts_by_code.get(int(code_a), 0)
        n_b = counts_by_code.get(int(code_b), 0)
        if int(code_a) == int(code_b):
            eligible[key] = n_a * max(n_a - 1, 0) // 2
        else:
            eligible[key] = n_a * n_b

    group_labels = _group_labels_for_exclusion(topology_arrays, mode)

    if group_labels is not None and indices.size:
        selected_groups = group_labels[indices]
        for group in np.unique(selected_groups):
            group_codes = selected_codes[selected_groups == group]
            group_counts = {
                int(code): int(count)
                for code, count in zip(*np.unique(group_codes, return_counts=True))
            }
            for key, (code_a, code_b) in code_pairs.items():
                if code_a is None or code_b is None:
                    continue
                n_a = group_counts.get(int(code_a), 0)
                n_b = group_counts.get(int(code_b), 0)
                if int(code_a) == int(code_b):
                    eligible[key] -= n_a * max(n_a - 1, 0) // 2
                else:
                    eligible[key] -= n_a * n_b

    bonded_pairs = _bonded_exclusion_pairs(topology_arrays)
    if bonded_pairs.size:
        in_selection = selected[bonded_pairs[:, 0]] & selected[bonded_pairs[:, 1]]
        bonded_pairs = bonded_pairs[in_selection]
        if group_labels is not None and bonded_pairs.size:
            already_group_excluded = (
                group_labels[bonded_pairs[:, 0]] == group_labels[bonded_pairs[:, 1]]
            )
            bonded_pairs = bonded_pairs[~already_group_excluded]

        if bonded_pairs.size:
            first_codes = type_codes[bonded_pairs[:, 0]]
            second_codes = type_codes[bonded_pairs[:, 1]]
            for key, (code_a, code_b) in code_pairs.items():
                if code_a is None or code_b is None:
                    continue
                if int(code_a) == int(code_b):
                    excluded = np.count_nonzero(
                        (first_codes == int(code_a)) & (second_codes == int(code_a))
                    )
                else:
                    excluded = np.count_nonzero(
                        ((first_codes == int(code_a)) & (second_codes == int(code_b)))
                        | ((first_codes == int(code_b)) & (second_codes == int(code_a)))
                    )
                eligible[key] -= int(excluded)

    for key, count in eligible.items():
        if count < 0:
            raise RuntimeError(
                f"Eligible pair count became negative for {key}: {count}; "
                "selection/exclusion metadata is inconsistent"
            )
    return eligible




def _build_exclusion_mask(
    topo: TopologyArrays,
    pair_indices: np.ndarray,
    exclude_option: str,
) -> np.ndarray:
    """Return boolean mask of pairs to exclude.

    Bonded exclusions (from TopologyArrays) are always applied.
    Additional nonbonded exclusion via *exclude_option*:
      - ``"resid"`` : exclude same-residue pairs
      - ``"molid"`` : exclude same-molecule pairs
      - ``"none"``  : no additional nonbonded exclusion
    """
    n = len(pair_indices)
    if n == 0:
        return np.zeros(0, dtype=bool)

    exclude_option = parse_exclude_option(exclude_option)
    cached_mode = getattr(topo, "excluded_nb_mode", None)
    cached_all = bool(getattr(topo, "excluded_nb_all", False))
    cached_ids = getattr(topo, "excluded_nb", None)
    if cached_mode == exclude_option and cached_ids is not None:
        if cached_all:
            return np.ones(n, dtype=bool)
        if cached_ids.size == 0:
            return np.zeros(n, dtype=bool)
        return np.isin(_encode_pairs(pair_indices, topo.n_atoms), cached_ids)

    # bonded exclusions — always applied
    parts = [
        _encode_pairs(arr, topo.n_atoms)
        for arr in (topo.exclude_12, topo.exclude_13, topo.exclude_14)
        if arr.size > 0
    ]
    excl_ids = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int32)
    if excl_ids.size > 0:
        mask = np.isin(_encode_pairs(pair_indices, topo.n_atoms), excl_ids)
    else:
        mask = np.zeros(n, dtype=bool)

    # nonbonded exclusion
    group_labels = _group_labels_for_exclusion(topo, exclude_option)
    if group_labels is not None:
        mask |= group_labels[pair_indices[:, 0]] == group_labels[pair_indices[:, 1]]

    return mask

def compute_neighbor_list(
    positions: np.ndarray,
    box: np.ndarray,
    cutoff: float,
    topology_arrays: TopologyArrays,
    sel_indices: Optional[np.ndarray] = None,
    exclude_option: str = "resid",
) -> List[List[int]]:
    """Return per-atom adjacency lists after topology-aware exclusion filtering.

    This is a generic helper around one global ``FastNS`` self-search. It is
    not part of the current ``MPIComputeEngine`` path, but it follows the same
    exclusion contract as ``compute_pairs_by_type()``.

    Parameters
    ----------
    positions : (n_atoms, 3) array
    box : MDAnalysis dimensions-like array
    cutoff : float
        Distance cutoff (same units as your coordinates) for neighbors.
    topology_arrays : TopologyArrays
        Immutable topology data (exclusions, resids, etc.).
    sel_indices : int64 array, optional
        Global atom indices to include in the neighbor construction.
    exclude_option : str
        Nonbonded exclusion mode: ``"resid"``, ``"molid"``, or ``"none"``.

    Returns
    -------
    list[list[int]]
        Symmetric neighbor lists in global atom indices.
    """
    exclude_option = parse_exclude_option(exclude_option)
    pos = np.asarray(positions, dtype=np.float32)
    bx = np.asarray(box, dtype=np.float32)
    n_atoms = pos.shape[0]
    indices = _normalize_selection_indices(topology_arrays, sel_indices)
    if indices.size == 0:
        return [[] for _ in range(n_atoms)]

    ns = FastNS(cutoff, pos[indices], box=bx)
    pairs = ns.self_search().get_pairs()
    if len(pairs) == 0:
        return [[] for _ in range(n_atoms)]

    pairs = np.asarray(pairs, dtype=np.int32)
    global_pairs = indices[pairs]
    exclude_mask = _build_exclusion_mask(
        topology_arrays,
        global_pairs,
        exclude_option,
    )
    keep_pairs = global_pairs[~exclude_mask]

    neighbor_list = [[] for _ in range(n_atoms)]
    for i, j in keep_pairs:
        neighbor_list[i].append(j)
        neighbor_list[j].append(i)

    return neighbor_list


def compute_pairs_by_type(
    positions: np.ndarray,
    box: np.ndarray,
    pair_type_list: list[InteractionKey],
    cutoff: float,
    topology_arrays: TopologyArrays,
    sel_indices: Optional[np.ndarray] = None,
    exclude_option: str = "resid",
) -> dict:
    """Return pair candidates grouped by canonical ``InteractionKey``.

    This is the current pair-search entry point used by
    ``MPIComputeEngine._compute_local_frames()``. It performs one conservative
    global neighbor search, applies topology-aware exclusions, then bins the
    surviving pairs by pair-type key using integer atom-type codes.

    No distances are computed here. The output is only a routing structure for
    downstream ``compute_frame_geometry()``.

    Parameters
    ----------
    positions : (n_atoms, 3) array
    box : MDAnalysis dimensions-like array
    cutoff : float
        Maximum cutoff for neighbor search (should be max over all pair pots).
    topology_arrays : TopologyArrays
        Immutable topology data (exclusions, resids, type codes, etc.).
    sel_indices : int64 array, optional
        Global atom indices to include in the neighbor construction.
    exclude_option : str
        Nonbonded exclusion mode: ``"resid"``, ``"molid"``, or ``"none"``.

    Returns
    -------
    dict
        ``{InteractionKey: (a_idx, b_idx)}`` with global atom-index arrays.

    Notes
    -----
    Caller contract:

    - pass one global cutoff, usually the maximum over all pair potentials;
    - pass ``sel_indices`` only to narrow the search domain, not to renumber
      atoms;
    - keep per-key masking and downstream observable logic out of this module.
    """
    exclude_option = parse_exclude_option(exclude_option)
    empty = (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
    out: dict = {k: empty for k in pair_type_list}

    pos = np.asarray(positions, dtype=np.float32)
    bx = np.asarray(box, dtype=np.float32)
    indices = _normalize_selection_indices(topology_arrays, sel_indices)
    if indices.size == 0:
        return out

    ns = FastNS(cutoff, pos[indices], box=bx)
    # NOTICE TO AGENTS: Don't use double/float64 here.

    pairs = ns.self_search().get_pairs()
    if len(pairs) == 0:
        return out

    pairs = np.asarray(pairs, dtype=np.int32)
    global_pairs = indices[pairs]
    exclude_mask = _build_exclusion_mask(topology_arrays, global_pairs, exclude_option)
    kept = pairs[~exclude_mask]
    if len(kept) == 0:
        return out

    # Integer type-code routing (SIMD-friendly int32 comparisons)
    type_codes = topology_arrays.atom_type_codes
    name_to_code = topology_arrays.atom_type_name_to_code
    ci = type_codes[indices[kept[:, 0]]]
    cj = type_codes[indices[kept[:, 1]]]
    gi = indices[kept[:, 0]]
    gj = indices[kept[:, 1]]

    for key in pair_type_list:
        if key.style != "pair": continue
        c0 = name_to_code.get(str(key.types[0]))
        c1 = name_to_code.get(str(key.types[1]))
        if c0 is None or c1 is None: continue
        if c0 == c1:
            mask = (ci == c0) & (cj == c1)
            if np.any(mask):
                out[key] = (gi[mask], gj[mask])
        else:
            mf = (ci == c0) & (cj == c1)
            mr = (ci == c1) & (cj == c0)
            if np.any(mf) or np.any(mr):
                out[key] = (
                    np.concatenate([gi[mf], gj[mr]]),
                    np.concatenate([gj[mf], gi[mr]]),
                )

    return out
