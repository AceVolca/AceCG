import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from MDAnalysis.lib.distances import distance_array
from MDAnalysis import Universe
from MDAnalysis.core.groups import Atom, AtomGroup

def AtomDistance(u, a, b): # a,b: mda.atom
	return distance_array(a.position, b.position, box=u.dimensions)[0][0]

def ComputeNeighborList(
    u: Universe,
    cutoff: float,
    frame: Optional[int] = None,
    exclude: bool = True,
) -> List[List[int]]:
    """
    Compute neighbor lists for each CG site in a given Universe.

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

    coords = u.atoms.positions
    resids = u.atoms.resids
    dist_mat = distance_array(coords, coords, box=u.dimensions)

    n_atoms = len(u.atoms)
    indices = np.arange(n_atoms)
    neighbor_list: List[List[int]] = []

    for i in indices:
        mask = (dist_mat[i] < cutoff) & (dist_mat[i] > 1e-6)
        if exclude:
            mask &= (resids != resids[i])
        neighbors = indices[mask]
        neighbor_list.append(neighbors.tolist())

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
    - Neighbor list construction uses `ComputeNeighborList(u, cutoff, frame, exclude)`.
    - The MDAnalysis selection `sel` is combined with atom type filters to reduce
      the number of candidate atoms.
    - Distance filtering is refined per pair type by using each potential's `.cutoff`.
    - Neighbor detection is unidirectional: if (i,j) is stored, (j,i) is not
      automatically added.
    """
    if frame is not None:
        u.trajectory[frame]
    neigh_list = ComputeNeighborList(u, cutoff, frame, exclude)
    
    pair2atom = defaultdict(list)

    # Pre-cache atom groups by type for faster lookup
    type_groups: Dict[str, AtomGroup] = {}
    for pair in pair2potential:
        t1, t2 = pair
        if t1 not in type_groups:
            type_groups[t1] = u.select_atoms(f"type {t1} and {sel}")
        if t2 not in type_groups:
            type_groups[t2] = u.select_atoms(f"type {t2} and {sel}")

    for pair in pair2potential:
        typ1_atoms = type_groups[pair[0]]
        typ2_atoms = type_groups[pair[1]]
        typ2_indices = set(a.index for a in typ2_atoms)  # for faster `in` check

        for a in typ1_atoms:
            neighbors = neigh_list[a.index]
            for j in neighbors:
                if j in typ2_indices:
                    b = u.atoms[j]
                    if AtomDistance(u, a, b) < pair2potential[pair].cutoff: # record atom pair within potential cutoff
                        pair2atom[pair].append((a, b))

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
        
    pair2distance_frame = defaultdict(dict)
    for frame in range(start, end):
        u.trajectory[frame]
        
        if update and (frame-start) % nstlist == 0: # update neighbor list
            pair2atom = NeighborList2Pair(u, pair2potential, sel, cutoff, frame, exclude)
        
        for pair, tuple_lis in pair2atom.items(): # read atom_pairs corresponding to the pair potential
            pair2distance_frame[frame][pair] = np.array([AtomDistance(u, atom_pair[0], atom_pair[1]) for atom_pair in tuple_lis])
    
    return pair2distance_frame