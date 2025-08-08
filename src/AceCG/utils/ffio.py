# AceCG/utils/ffio.py
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..potentials.base import BasePotential

from ..potentials.gaussian import GaussianPotential


POTENTIAL_REGISTRY = {
    "gauss/cut": GaussianPotential,
    "gauss/wall": GaussianPotential,
    # "lj/cut": LJPotential,  # Future extension
}


def FFParamArray(pair2potential: Dict[Tuple[str, str], BasePotential]) -> np.ndarray:
    """
    Concatenate all potential parameters into a single 1D NumPy array.

    Parameters
    ----------
    pair2potential : dict
        Mapping (type1, type2) to potential objects, each with `.get_params` method.

    Returns
    -------
    np.ndarray
        1D array of all force field parameters.
    """
    return np.concatenate([pot.get_params() for pot in pair2potential.values()])


def FFParamIndexMap(pair2potential: Dict[Tuple[str, str], BasePotential]) -> List[Tuple[Tuple[str, str], str]]:
    """
    Create a map from parameter index to (pair_type, param_name).

    Returns
    -------
    List of tuples: [ ((type1, type2), param_name), ... ]
    Matching order of FFParamArray.
    """
    index_map = []
    for pair, pot in pair2potential.items():
        for name in pot.param_names():
            index_map.append((pair, name))
    return index_map


def ReadLmpFF(file: str, typ_sel: Optional[List[str]] = None) -> Dict[Tuple[str, str], BasePotential]:
    """
    Generalized reader for LAMMPS force field files using registered potential types.

    Parameters
    ----------
    file : str
        Path to LAMMPS pair_coeff file.
    typ_sel : list of str, optional
        Only load specified pair styles (e.g., ["gauss/cut"])

    Returns
    -------
    pair2potential : dict
        Mapping (type1, type2) → BasePotential
    """
    pair2potential = {}

    with open(file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if "pair_coeff" in line:
            tmp = line.split()
            style = tmp[3]
            if typ_sel is None or style in typ_sel:
                if style in POTENTIAL_REGISTRY:
                    constructor = POTENTIAL_REGISTRY[style]
                    params = list(map(float, tmp[4:]))
                    pair2potential[(tmp[1], tmp[2])] = constructor(tmp[1], tmp[2], *params)

    return pair2potential


def WriteLmpFF(
    old_file: str,
    new_file: str,
    pair2potential: Dict[Tuple[str, str], BasePotential],
    typ_sel: Optional[List[str]] = None
):
    """
    Write updated parameters to a new LAMMPS-style force field file.

    Parameters
    ----------
    old_file : str
        Source file path.
    new_file : str
        Destination file path.
    pair2potential : dict
        Mapping (type1, type2) → BasePotential
    typ_sel : list of str, optional
        Only update selected pair styles.
    """
    L_new = FFParamArray(pair2potential)
    idx = 0
    with open(old_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "pair_coeff" in line:
            tmp = line.split()
            style = tmp[3]
            pair = (tmp[1], tmp[2])

            if typ_sel is None or style in typ_sel:
                if pair in pair2potential:
                    n_param = pair2potential[pair].n_params()
                    tmp[4:4 + n_param] = map(str, L_new[idx:idx + n_param])
                    idx += n_param
                    lines[i] = "   ".join(tmp) + "\n"

    with open(new_file, "w") as f:
        f.writelines(lines)