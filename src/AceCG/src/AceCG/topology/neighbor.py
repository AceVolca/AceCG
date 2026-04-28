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

from typing import List, Optional

from MDAnalysis.lib.nsgrid import FastNS
import numpy as np

from .topology_array import TopologyArrays
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


_parse_exclude_option = parse_exclude_option


def _encode_pairs(pairs: np.ndarray, n: int) -> np.ndarray:
    """Canonical int32 id for each (i, j) row: min*n + max."""
    if pairs.size == 0:
        return np.empty(0, dtype=np.int32)
    sp = np.sort(np.asarray(pairs, dtype=np.int32), axis=1)
    return sp[:, 0] * np.int32(n) + sp[:, 1]


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

    exclude_option = _parse_exclude_option(exclude_option)
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
    if exclude_option == "resid":
        atom_resids = topo.resids[topo.atom_resindex] 
        mask |= atom_resids[pair_indices[:, 0]] == atom_resids[pair_indices[:, 1]]
    elif exclude_option == "molid":
        mask |= topo.molnums[pair_indices[:, 0]] == topo.molnums[pair_indices[:, 1]]

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
    exclude_option = _parse_exclude_option(exclude_option)
    pos = np.asarray(positions, dtype=np.float32)
    bx = np.asarray(box, dtype=np.float32)
    n_atoms = pos.shape[0]
    indices = (
        np.arange(n_atoms, dtype=np.int32)
        if sel_indices is None
        else np.asarray(sel_indices, dtype=np.int32)
    )
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
    exclude_option = _parse_exclude_option(exclude_option)
    empty = (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
    out: dict = {k: empty for k in pair_type_list}

    pos = np.asarray(positions, dtype=np.float32)
    bx = np.asarray(box, dtype=np.float32)
    indices = (
        np.arange(pos.shape[0], dtype=np.int32)
        if sel_indices is None
        else np.asarray(sel_indices, dtype=np.int32)
    )
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
