# AceCG/utils/neighbor.py
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from MDAnalysis.lib.distances import distance_array
from MDAnalysis import Universe
from MDAnalysis.core.groups import Atom, AtomGroup
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.lib.distances import calc_bonds

def AtomDistance(u: Universe, a: Atom, b: Atom) -> float:
    """Efficient single-pair distance with PBC support."""
    return calc_bonds(a.position, b.position, box=u.dimensions)[0]


def ComputeNeighborList(
    u: Universe,
    cutoff: float,
    frame: Optional[int] = None,
    exclude: bool = True,
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
    exclude : bool
        If True, excludes atoms from the same molecule (resid).

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

    # Use FastNS for neighbor search
    ns = FastNS(cutoff, positions, box=box)
    pairs = ns.self_search().get_pairs()  # shape: (N_pairs, 2)

    # Build neighbor list from pairs
    neighbor_list = [[] for _ in range(n_atoms)]
    for i, j in pairs:
        if exclude and resids[i] == resids[j]:
            continue
        neighbor_list[i].append(j)
        neighbor_list[j].append(i)  # symmetric

    return neighbor_list


def NeighborList2Pair(
    u: Universe,
    pair2potential: Dict[Tuple[str, str], object],
    sel: str,
    cutoff: float,
    frame: Optional[int] = None,
    exclude: bool = True,
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
    exclude : bool, optional
        If True, excludes atoms belonging to the same molecule (same `resid`)
        when constructing the neighbor list. Default is True.

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
    positions = sel_atoms.positions
    box = u.dimensions
    indices = sel_atoms.indices  # map to u.atoms

    # Build a type lookup table for atoms in selection
    index2atom = {i: a for i, a in zip(indices, sel_atoms)}

    # Use FastNS on selected atoms
    ns = FastNS(cutoff, positions, box=box)
    pairs = ns.self_search().get_pairs()

    pair2atom = defaultdict(list)

    for ii, jj in pairs:
        i = indices[ii]  # real atom index in u.atoms
        j = indices[jj]
        a = index2atom[i]
        b = index2atom[j]

        if exclude and a.resid == b.resid:
            continue

        ti, tj = a.type, b.type

        # check both (ti, tj) and (tj, ti)
        for key in [(ti, tj), (tj, ti)]:
            if key in pair2potential:
                d = distance_array(a.position, b.position, box=box)[0, 0]
                if d < pair2potential[key].cutoff:
                    pair2atom[key].append((a, b) if key == (ti, tj) else (b, a))

    return pair2atom


def Pair2DistanceByFrame(
    u: Universe,
    start: int,
    end: int,
    cutoff: float,
    pair2potential: Dict[Tuple[str, str], object],
    sel: str = "all",
    nstlist: int = 10,
    exclude: bool = True,
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
    exclude : bool, optional
        If True, atoms in the same molecule (same `resid`) are excluded when building
        the neighbor list. Default is True.

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
    else:
        update = True

    pair2distance_frame = {}
    for frame in range(start, end):
        pair2distance_frame[frame] = {} # initialize empty dict, in case there're no pairs within the cutoff
        u.trajectory[frame]

        if update and (frame - start) % nstlist == 0: # update neighbor list
            pair2atom = NeighborList2Pair(u, pair2potential, sel, cutoff, frame, exclude)

        for pair, tuples in pair2atom.items():
            if not tuples:
                pair2distance_frame[frame][pair] = np.array([])
                continue

            a_positions = np.array([a1.position for a1, _ in tuples])
            b_positions = np.array([a2.position for _, a2 in tuples])

            pair2distance_frame[frame][pair] = calc_bonds(a_positions, b_positions, box=u.dimensions)

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
