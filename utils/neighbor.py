# AceCG/utils/neighbor.py
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict
from MDAnalysis import Universe
from MDAnalysis.core.groups import Atom
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.lib.distances import calc_bonds


ExcludeMode = Union[bool, int, str]


def _as_pairs(indices: np.ndarray) -> np.ndarray:
    """Return canonical sorted unique (i, j) pairs with i < j."""
    if indices.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.asarray(indices, dtype=np.int64).reshape(-1, 2)
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0)


def _pair_ids_from_pairs(pairs: np.ndarray, n_atoms: int) -> np.ndarray:
    """Encode sorted pairs (i, j) as unique int64 ids: i * n_atoms + j."""
    if pairs.size == 0:
        return np.empty(0, dtype=np.int64)
    return pairs[:, 0].astype(np.int64) * np.int64(n_atoms) + pairs[:, 1].astype(np.int64)


def _pair_ids_from_raw_pairs(pairs: np.ndarray, n_atoms: int) -> np.ndarray:
    """Encode unsorted 2-col pairs into canonical int64 ids."""
    if pairs.size == 0:
        return np.empty(0, dtype=np.int64)
    sp = np.sort(np.asarray(pairs, dtype=np.int64), axis=1)
    return _pair_ids_from_pairs(sp, n_atoms)


def _parse_exclude_mode(exclude: ExcludeMode) -> Tuple[str, Tuple[bool, bool, bool]]:
    """
    Parse exclusion mode.

    Returns
    -------
    mode : {"none", "resid", "bonded"}
    bits : tuple(bool, bool, bool)
        For bonded mode, bits correspond to excluding (1-2, 1-3, 1-4).
    """
    if exclude is None:
        return "none", (False, False, False)

    if isinstance(exclude, bool):
        return ("resid", (False, False, False)) if exclude else ("none", (False, False, False))

    if isinstance(exclude, (int, np.integer)):
        token = f"{int(exclude):03d}"
    else:
        token = str(exclude).strip().lower()

    if token in {"", "none", "false", "off", "0", "000"}:
        return "none", (False, False, False)
    if token in {"true", "resid", "same_resid", "same-resid", "molecule", "intra", "intra_molecular", "intra-molecular"}:
        return "resid", (False, False, False)
    if len(token) == 3 and set(token) <= {"0", "1"}:
        return "bonded", (token[0] == "1", token[1] == "1", token[2] == "1")

    raise ValueError(
        f"Unsupported exclude mode: {exclude!r}. "
        "Use bool/None, 'resid'/'molecule', or a 3-bit string/int like 100/110/111."
    )


def _get_exclusion_ids(
    u: Universe,
    bits: Tuple[bool, bool, bool],
) -> np.ndarray:
    """Get encoded pair ids to exclude for bonded mode bits (1-2, 1-3, 1-4)."""
    if not any(bits):
        return np.empty(0, dtype=np.int64)
    info = GetBondedInfo(u)
    arrays = []
    if bits[0]:
        arrays.append(info["bond_12_ids"])
    if bits[1]:
        arrays.append(info["angle_13_ids"])
    if bits[2]:
        arrays.append(info["dihedral_14_ids"])
    if not arrays:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(arrays))


def _exclude_mask_from_mode(
    *,
    u: Universe,
    pair_indices: np.ndarray,
    mode: str,
    bits: Tuple[bool, bool, bool],
    resids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Return boolean mask for pairs that should be excluded.

    Parameters
    ----------
    pair_indices : np.ndarray
        Shape (N,2), global atom indices.
    """
    n_pairs = len(pair_indices)
    if n_pairs == 0 or mode == "none":
        return np.zeros(n_pairs, dtype=bool)

    if mode == "resid":
        if resids is None:
            resids = u.atoms.resids
        return resids[pair_indices[:, 0]] == resids[pair_indices[:, 1]]

    exclusion_ids = _get_exclusion_ids(u, bits)
    if exclusion_ids.size == 0:
        return np.zeros(n_pairs, dtype=bool)
    pair_ids = _pair_ids_from_raw_pairs(pair_indices, len(u.atoms))
    return np.isin(pair_ids, exclusion_ids)


def AtomDistance(u: Universe, a: Atom, b: Atom) -> float:
    """Efficient single-pair distance with PBC support."""
    return calc_bonds(a.position, b.position, box=u.dimensions)[0]


def _pair2atom_to_index_arrays(
    pair2atom: Dict[Tuple[str, str], List[Tuple[Atom, Atom]]]
) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """Convert pair->[(Atom,Atom), ...] to pair->(idx_i, idx_j) arrays for fast per-frame distance eval."""
    pair2idx: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for pair, tuples in pair2atom.items():
        n = len(tuples)
        if n == 0:
            empty = np.empty(0, dtype=np.int64)
            pair2idx[pair] = (empty, empty)
            continue
        ai = np.fromiter((a.index for a, _ in tuples), dtype=np.int64, count=n)
        bi = np.fromiter((b.index for _, b in tuples), dtype=np.int64, count=n)
        pair2idx[pair] = (ai, bi)
    return pair2idx


def ComputeNeighborList(
    u: Universe,
    cutoff: float,
    frame: Optional[int] = None,
    exclude: ExcludeMode = True,
) -> List[List[int]]:
    """
    Compute neighbor lists for each CG site using FastNS (grid-based neighbor search).

    Parameters
    ----------
    u : MDAnalysis.Universe
        An already-loaded Universe (with topology & trajectory).
    cutoff : float
        Distance cutoff (same units as your coordinates) for neighbors.
    frame : int, optional
        0-based index of the frame to jump to. If None (default),
        assumes `u` is already positioned at the desired frame.
    exclude : bool | int | str
        Exclusion behavior for candidate pairs:
        - `True` (default): exclude same-resid (legacy behavior).
        - `False` or `"none"`: no exclusion.
        - `"resid"` / `"molecule"` / `"intra"`: exclude same-resid pairs.
        - `100/110/111` (int or 3-char bit string): exclude bonded 1-2 / 1-3 / 1-4
          pairs according to each bit.

    Returns
    -------
    neighbor_list : list of lists
        neighbor_list[i] is a Python list of all atom-indices within `cutoff`
        of atom i in the current frame.
    """
    if frame is not None:
        u.trajectory[frame]

    positions = u.atoms.positions
    box = u.dimensions  # Must be [lx, ly, lz, alpha, beta, gamma]

    resids = u.atoms.resids
    n_atoms = len(positions)
    mode, bits = _parse_exclude_mode(exclude)

    # Use FastNS for neighbor search
    ns = FastNS(cutoff, positions, box=box)
    pairs = ns.self_search().get_pairs()  # shape: (N_pairs, 2)
    if len(pairs) == 0:
        return [[] for _ in range(n_atoms)]

    exclude_mask = _exclude_mask_from_mode(
        u=u,
        pair_indices=np.asarray(pairs, dtype=np.int64),
        mode=mode,
        bits=bits,
        resids=resids,
    )
    keep_pairs = np.asarray(pairs, dtype=np.int64)[~exclude_mask]

    # Build neighbor list from pairs
    neighbor_list = [[] for _ in range(n_atoms)]
    for i, j in keep_pairs:
        neighbor_list[i].append(j)
        neighbor_list[j].append(i)  # symmetric

    return neighbor_list


def NeighborList2Pair(
    u: Universe,
    pair2potential: Dict[Tuple[str, str], object],
    sel: str,
    cutoff: float,
    frame: Optional[int] = None,
    exclude: ExcludeMode = True,
) -> Dict[Tuple[str, str], List[Tuple[Atom, Atom]]]:
    """
    Build a mapping of atom-type pairs to atom pairs within a given distance cutoff.

    This function first computes a neighbor list for the Universe `u` and then filters
    atom pairs based on both atom types (specified by `pair2potential` keys) and
    potential-specific cutoffs.

    Parameters
    ----------
    u : MDAnalysis.Universe
        An already-loaded Universe with topology and trajectory.
    pair2potential : dict
        A dictionary mapping a tuple of atom type strings (e.g., ("1", "2"))
        to potential objects. Each potential must have a `.cutoff` attribute
        defining the maximum interaction distance for that type pair.
    sel : str
        MDAnalysis selection string to limit the set of atoms considered for
        pair-finding (e.g., "name CA" or "type 1 2").
    cutoff : float
        Global cutoff distance (same units as coordinates) for building the neighbor list.
        This acts as a pre-filter before checking individual potential cutoffs.
    frame : int, optional
        0-based trajectory frame index. If provided, `u.trajectory[frame]` is loaded.
    exclude : bool | int | str, optional
        Exclusion behavior for candidate pairs:
        - `True` (default): exclude same-resid (legacy behavior).
        - `False` or `"none"`: no exclusion.
        - `"resid"` / `"molecule"` / `"intra"`: exclude same-resid pairs.
        - `100/110/111` (int or 3-char bit string): exclude bonded 1-2 / 1-3 / 1-4
          pairs according to each bit.

    Returns
    -------
    pair2atom : dict
        Dictionary where each key is a type pair (from `pair2potential`),
        and the value is a list of (atom1, atom2) tuples such that:
        - atom1 is of type pair[0],
        - atom2 is of type pair[1],
        - atom2 is in the neighbor list of atom1,
        - the distance between atom1 and atom2 is less than the potential's `.cutoff`.

    Notes
    -----
    - The MDAnalysis selection `sel` is combined with atom type filters to reduce
      the number of candidate atoms.
    - Distance filtering is refined per pair type by using each potential's `.cutoff`.
    """
    if frame is not None:
        u.trajectory[frame]

    sel_atoms = u.select_atoms(sel)
    if len(sel_atoms) == 0:
        return defaultdict(list)

    positions = sel_atoms.positions
    box = u.dimensions
    indices = sel_atoms.indices  # map to u.atoms
    mode, bits = _parse_exclude_mode(exclude)

    # Use FastNS on selected atoms
    ns = FastNS(cutoff, positions, box=box)
    pairs = ns.self_search().get_pairs()

    pair2atom = defaultdict(list)

    if len(pairs) == 0:
        return pair2atom

    global_pairs = np.asarray(indices[np.asarray(pairs, dtype=np.int64)], dtype=np.int64)
    exclude_mask = _exclude_mask_from_mode(
        u=u,
        pair_indices=global_pairs,
        mode=mode,
        bits=bits,
        resids=u.atoms.resids,
    )

    kept_local_pairs = np.asarray(pairs, dtype=np.int64)[~exclude_mask]
    if len(kept_local_pairs) == 0:
        return pair2atom

    dists = calc_bonds(
        positions[kept_local_pairs[:, 0]],
        positions[kept_local_pairs[:, 1]],
        box=box,
    )

    # Benchmarked on the provided test case: key-mask vectorization was slower than this per-pair loop.
    atom_types = np.asarray(sel_atoms.types)
    for (ii, jj), d in zip(kept_local_pairs, dists):
        a = sel_atoms[ii]
        b = sel_atoms[jj]
        ti, tj = atom_types[ii], atom_types[jj]

        key = (ti, tj)
        pot = pair2potential.get(key)
        if pot is not None and d < pot.cutoff:
            pair2atom[key].append((a, b))

        rkey = (tj, ti)
        rpot = pair2potential.get(rkey)
        if rpot is not None and d < rpot.cutoff:
            pair2atom[rkey].append((a, b) if rkey == key else (b, a))

    return pair2atom


def NeighborList2PairIndices(
    u: Universe,
    pair2potential: Dict[Tuple[str, str], object],
    sel: str,
    cutoff: float,
    frame: Optional[int] = None,
    exclude: ExcludeMode = True,
) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """
    Vectorized variant of NeighborList2Pair that returns atom-index arrays.

    Returns
    -------
    pair2idx : dict
        key -> (idx_i, idx_j) where each array is int64 and aligned.
    """
    if frame is not None:
        u.trajectory[frame]

    sel_atoms = u.select_atoms(sel)
    out: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for k in pair2potential.keys():
        out[k] = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

    if len(sel_atoms) == 0:
        return out

    positions = sel_atoms.positions
    box = u.dimensions
    indices = np.asarray(sel_atoms.indices, dtype=np.int64)
    mode, bits = _parse_exclude_mode(exclude)

    ns = FastNS(cutoff, positions, box=box)
    pairs = ns.self_search().get_pairs()
    if len(pairs) == 0:
        return out

    pairs = np.asarray(pairs, dtype=np.int64)
    global_pairs = indices[pairs]
    exclude_mask = _exclude_mask_from_mode(
        u=u,
        pair_indices=global_pairs,
        mode=mode,
        bits=bits,
        resids=u.atoms.resids,
    )

    kept_local_pairs = pairs[~exclude_mask]
    if len(kept_local_pairs) == 0:
        return out

    dists = calc_bonds(
        positions[kept_local_pairs[:, 0]],
        positions[kept_local_pairs[:, 1]],
        box=box,
    )

    atom_types = np.asarray(sel_atoms.types).astype(str)
    ti = atom_types[kept_local_pairs[:, 0]]
    tj = atom_types[kept_local_pairs[:, 1]]
    gi = indices[kept_local_pairs[:, 0]]
    gj = indices[kept_local_pairs[:, 1]]

    for key, pot in pair2potential.items():
        k0, k1 = str(key[0]), str(key[1])
        cut_mask = dists < float(pot.cutoff)
        if k0 == k1:
            mask = (ti == k0) & (tj == k1) & cut_mask
            if np.any(mask):
                out[key] = (gi[mask].astype(np.int64), gj[mask].astype(np.int64))
            else:
                out[key] = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        else:
            mf = (ti == k0) & (tj == k1) & cut_mask
            mr = (ti == k1) & (tj == k0) & cut_mask
            if np.any(mf) or np.any(mr):
                ai = np.concatenate([gi[mf], gj[mr]]).astype(np.int64, copy=False)
                bj = np.concatenate([gj[mf], gi[mr]]).astype(np.int64, copy=False)
                out[key] = (ai, bj)
            else:
                out[key] = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

    return out


def GetBondedInfo(u: Universe):
    """
    Build and cache bonded-topology exclusion data.

    Returns
    -------
    info : dict
        {
          "bond_12_pairs": ndarray (n_bond_pairs, 2),
          "angle_13_pairs": ndarray (n_angle_pairs, 2),
          "dihedral_14_pairs": ndarray (n_dihedral_pairs, 2),
          "bond_12_ids": ndarray (n_bond_pairs,),
          "angle_13_ids": ndarray (n_angle_pairs,),
          "dihedral_14_ids": ndarray (n_dihedral_pairs,),
          "bond_terms": ndarray (n_bonds, 2),
          "angle_terms": ndarray (n_angles, 3),
          "dihedral_terms": ndarray (n_dihedrals, 4),
          "atom_lookup": dict[int, dict[str, ndarray]],
        }

    Notes
    -----
    - Pair arrays are canonicalized with i < j and deduplicated.
    - Encoded ids use: `pair_id = i * n_atoms + j`, allowing fast `np.isin` checks.
    - Supports exclusion modes like 100/110/111:
      `pair AND NOT (bond_12 OR angle_13 OR dihedral_14)`.
    - Cached on the Universe as `_acecg_bonded_info_cache` (assumes static topology).
    """
    cache = getattr(u, "_acecg_bonded_info_cache", None)
    if cache is not None:
        return cache

    n_atoms = len(u.atoms)
    try:
        atom_types = np.asarray(u.atoms.types).astype(str)
    except Exception:
        atom_types = np.asarray([str(getattr(a, "type", getattr(a, "name", a.index))) for a in u.atoms])

    # 1-2 from bonds
    if hasattr(u, "bonds") and len(u.bonds) > 0:
        bond_terms = np.asarray(u.bonds.indices, dtype=np.int64)
    else:
        bond_terms = np.empty((0, 2), dtype=np.int64)
    bond_12_pairs = _as_pairs(bond_terms)

    # 1-3 from angles
    if hasattr(u, "angles") and len(u.angles) > 0:
        angle_terms = np.asarray(u.angles.indices, dtype=np.int64)
        angle_13_pairs = _as_pairs(angle_terms[:, [0, 2]])
    else:
        angle_terms = np.empty((0, 3), dtype=np.int64)
        angle_13_pairs = np.empty((0, 2), dtype=np.int64)

    # 1-4 from dihedrals
    if hasattr(u, "dihedrals") and len(u.dihedrals) > 0:
        dihedral_terms = np.asarray(u.dihedrals.indices, dtype=np.int64)
        dihedral_14_pairs = _as_pairs(dihedral_terms[:, [0, 3]])
    else:
        dihedral_terms = np.empty((0, 4), dtype=np.int64)
        dihedral_14_pairs = np.empty((0, 2), dtype=np.int64)

    # Encoded ids for fast np.isin-based exclusion checks.
    bond_12_ids = _pair_ids_from_pairs(bond_12_pairs, n_atoms)
    angle_13_ids = _pair_ids_from_pairs(angle_13_pairs, n_atoms)
    dihedral_14_ids = _pair_ids_from_pairs(dihedral_14_pairs, n_atoms)

    # Per-atom quick lookup for force-matching style access.
    atom_lookup = {
        i: {
            "bond_12_neighbors": set(),
            "angle_13_neighbors": set(),
            "dihedral_14_neighbors": set(),
            "all_111_neighbors": set(),
            "bond_terms": set(),
            "angle_terms": set(),
            "dihedral_terms": set(),
            "angle_partner_types": set(),
            "dihedral_partner_types": set(),
        }
        for i in range(n_atoms)
    }

    # This section runs once and is cached, so keeping simple loops is faster to maintain and not a per-frame bottleneck.
    for i, j in bond_12_pairs:
        atom_lookup[int(i)]["bond_12_neighbors"].add(int(j))
        atom_lookup[int(j)]["bond_12_neighbors"].add(int(i))

    for i, j in angle_13_pairs:
        atom_lookup[int(i)]["angle_13_neighbors"].add(int(j))
        atom_lookup[int(j)]["angle_13_neighbors"].add(int(i))
        atom_lookup[int(i)]["angle_partner_types"].add(str(atom_types[j]))
        atom_lookup[int(j)]["angle_partner_types"].add(str(atom_types[i]))

    for i, j in dihedral_14_pairs:
        atom_lookup[int(i)]["dihedral_14_neighbors"].add(int(j))
        atom_lookup[int(j)]["dihedral_14_neighbors"].add(int(i))
        atom_lookup[int(i)]["dihedral_partner_types"].add(str(atom_types[j]))
        atom_lookup[int(j)]["dihedral_partner_types"].add(str(atom_types[i]))

    for term_idx, term in enumerate(bond_terms):
        for atom_idx in term:
            atom_lookup[int(atom_idx)]["bond_terms"].add(int(term_idx))

    for term_idx, term in enumerate(angle_terms):
        for atom_idx in term:
            atom_lookup[int(atom_idx)]["angle_terms"].add(int(term_idx))

    for term_idx, term in enumerate(dihedral_terms):
        for atom_idx in term:
            atom_lookup[int(atom_idx)]["dihedral_terms"].add(int(term_idx))

    for v in atom_lookup.values():
        all_neighbors = v["bond_12_neighbors"] | v["angle_13_neighbors"] | v["dihedral_14_neighbors"]
        v["all_111_neighbors"] = all_neighbors
        v["bond_12_neighbors"] = np.array(sorted(v["bond_12_neighbors"]), dtype=np.int64)
        v["angle_13_neighbors"] = np.array(sorted(v["angle_13_neighbors"]), dtype=np.int64)
        v["dihedral_14_neighbors"] = np.array(sorted(v["dihedral_14_neighbors"]), dtype=np.int64)
        v["all_111_neighbors"] = np.array(sorted(v["all_111_neighbors"]), dtype=np.int64)
        v["bond_terms"] = np.array(sorted(v["bond_terms"]), dtype=np.int64)
        v["angle_terms"] = np.array(sorted(v["angle_terms"]), dtype=np.int64)
        v["dihedral_terms"] = np.array(sorted(v["dihedral_terms"]), dtype=np.int64)
        v["angle_partner_types"] = np.array(sorted(v["angle_partner_types"]), dtype=object)
        v["dihedral_partner_types"] = np.array(sorted(v["dihedral_partner_types"]), dtype=object)

    info = {
        "bond_12_pairs": bond_12_pairs,
        "angle_13_pairs": angle_13_pairs,
        "dihedral_14_pairs": dihedral_14_pairs,
        "bond_12_ids": bond_12_ids,
        "angle_13_ids": angle_13_ids,
        "dihedral_14_ids": dihedral_14_ids,
        "bond_terms": bond_terms,
        "angle_terms": angle_terms,
        "dihedral_terms": dihedral_terms,
        "atom_lookup": atom_lookup,
    }

    setattr(u, "_acecg_bonded_info_cache", info)
    return info


def Pair2DistanceByFrame(
    u: Universe,
    start: int,
    end: int,
    cutoff: float,
    pair2potential: Dict[Tuple[str, str], object],
    sel: str = "all",
    nstlist: int = 10,
    exclude: ExcludeMode = True,
) -> Dict[int, Dict[Tuple[str, str], np.ndarray]]:
    """
    Compute per-frame distances for atom pairs specified by type and filtered through neighbor lists.

    This function iterates over a trajectory segment (from `start` to `end`) and collects
    distances between atom pairs defined in `pair2potential`. The neighbor list is updated
    every `nstlist` frames (or fixed if `nstlist == 0`), and filtered by the specified
    atom selection and molecular exclusions.

    Parameters
    ----------
    u : MDAnalysis.Universe
        An already-loaded Universe with trajectory and topology information.
    start : int
        Starting frame index (inclusive).
    end : int
        Ending frame index (exclusive).
    cutoff : float
        Global cutoff used to build the neighbor list. Each potential may have its own
        `.cutoff` used as a stricter distance filter.
    pair2potential : dict
        Dictionary mapping a tuple of atom type strings (e.g., ("1", "2")) to potential objects.
        Each potential must define a `.cutoff` attribute to limit valid pairwise interactions.
    sel : str, optional
        Atom selection string used to restrict atoms considered in pair search (default is "all").
    nstlist : int, optional
        Number of frames between neighbor list updates. If 0, neighbor list is computed once
        at the first frame and reused. Default is 10.
    exclude : bool | int | str, optional
        Exclusion behavior for candidate pairs:
        - `True` (default): exclude same-resid (legacy behavior).
        - `False` or `"none"`: no exclusion.
        - `"resid"` / `"molecule"` / `"intra"`: exclude same-resid pairs.
        - `100/110/111` (int or 3-char bit string): exclude bonded 1-2 / 1-3 / 1-4
          pairs according to each bit.

    Returns
    -------
    pair2distance_frame : dict
        Nested dictionary structure:
            pair2distance_frame[frame][pair] = 1D NumPy array of distances
        - `frame` is the trajectory frame index.
        - `pair` is a (type1, type2) string tuple.
        - Value is a NumPy array of distances between atom pairs of the given type at that frame.

    Notes
    -----
    - Neighbor lists are constructed using `NeighborList2Pair(...)`, and filtered by the
      per-pair `.cutoff` distances from each potential.
    - Pair selection is unidirectional: only (a, b) is stored when b is in a's neighbor list.
    - A user-defined `AtomDistance(u, a, b)` function is expected to be available for
      computing interatomic distances with proper PBC handling.
    """
    if nstlist == 0:
        update = False
        pair2atom = NeighborList2Pair(u, pair2potential, sel, cutoff, start, exclude) # use start frame to calculate pair2atom
        pair2idx = _pair2atom_to_index_arrays(pair2atom)
    else:
        update = True

    pair2distance_frame = {}
    for frame in range(start, end):
        pair2distance_frame[frame] = {} # initialize empty dict, in case there're no pairs within the cutoff
        u.trajectory[frame]

        if update and (frame - start) % nstlist == 0: # update neighbor list
            pair2atom = NeighborList2Pair(u, pair2potential, sel, cutoff, frame, exclude)
            pair2idx = _pair2atom_to_index_arrays(pair2atom)

        positions = u.atoms.positions
        for pair, (a_idx, b_idx) in pair2idx.items():
            if a_idx.size == 0:
                pair2distance_frame[frame][pair] = np.array([])
                continue

            pair2distance_frame[frame][pair] = calc_bonds(positions[a_idx], positions[b_idx], box=u.dimensions)

    return pair2distance_frame


def combine_Pair2DistanceByFrame(
    dicts: List[Dict],
    start_frame=0,
):
    """
    Combine multiple Pair2DistanceByFrame dictionaries into one,
    with continuous global frame indices.

    Parameters
    ----------
    dicts : list of dict
        Each element is a Pair2DistanceByFrame result:
        {frame_idx: {pair: distances}}
        frame_idx is assumed to be local (relative) index.
        Order in list defines concatenation order.
    start_frame : int
        Global starting frame index.

    Returns
    -------
    combined : dict
        {global_frame_idx: {pair: distances}}
    """
    combined = {}
    cur = start_frame

    for d in dicts:
        # sorted by frame_idx
        for _, frame_data in sorted(d.items(), key=lambda x: x[0]):
            combined[cur] = frame_data
            cur += 1

    return combined
