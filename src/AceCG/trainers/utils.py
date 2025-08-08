# AceCG/trainers/utils.py
from ..utils.neighbor import Pair2DistanceByFrame

def optimizer_accepts_hessian(optimizer) -> bool:
    """
    Check whether the optimizer's `step` method accepts a 'hessian' argument.

    Parameters
    ----------
    optimizer : object
        An optimizer instance with a `.step()` method.

    Returns
    -------
    bool
        True if 'hessian' is a parameter of the .step() method, else False.
    """
    return hasattr(optimizer, 'step') and 'hessian' in optimizer.step.__code__.co_varnames

def prepare_Trainer_data(u, pair2potential, start, end, cutoff, sel="all", nstlist=10, exclude=True, weight=None):
    """
    Extracts pairwise distances from trajectory into Trainer-compatible format.

    Parameters
    ----------
    u : MDAnalysis.Universe
        Input simulation trajectory.
    pair2potential : dict
        Mapping of (type1, type2) to BasePotential objects.
    start : int
        Starting frame index (inclusive).
    end : int
        Ending frame index (exclusive).
    cutoff : float
        Global neighbor cutoff.
    sel : str, optional
        Atom selection string. Default is 'all'.
    nstlist : int, optional
        Neighbor list update interval.
    exclude : bool, optional
        Whether to exclude bonded neighbors. Default is True.

    Returns
    -------
    data : dict
        Dictionary with keys:
        - 'dist': frame-indexed pairwise distance dictionary
        - 'weight': None (uniform weighting)
    """
    dist = Pair2DistanceByFrame(
        u, start=start, end=end, cutoff=cutoff,
        pair2potential=pair2potential, sel=sel,
        nstlist=nstlist, exclude=exclude
    )
    return {"dist": dist, "weight": weight}