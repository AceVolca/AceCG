# AceCG/utils/ffio.py
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..potentials.gaussian import GaussianPotential


def FFParamArray(pair2potential: Dict[Tuple[str, str], GaussianPotential]) -> np.ndarray:
    """
    Concatenate all potential parameters into a single 1D NumPy array.

    Parameters
    ----------
    pair2potential : dict
        Mapping (type1, type2) to potential objects, each with `.params` attribute.

    Returns
    -------
    np.ndarray
        1D array of all force field parameters.
    """
    return np.concatenate([pot.params() for pot in pair2potential.values()])


def FFParamIndexMap(pair2potential: Dict[Tuple[str, str], GaussianPotential]) -> List[Tuple[Tuple[str, str], str]]:
    """
    Create a map from parameter index to (pair_type, param_name).

    Returns
    -------
    List of tuples: [ ((type1, type2), param_name), ... ]
    Matching order of FFParamArray.
    """
    index_map = []
    for pair, pot in pair2potential.items():
        for name in pot.param_names:
            index_map.append((pair, name))
    return index_map


def ReadLmpFF(file: str, typ_sel: Optional[List[str]] = None) -> Dict[Tuple[str, str], GaussianPotential]:
    """
    Read pairwise force field parameters from a LAMMPS-style input file.

    Parameters
    ----------
    file : str
        Path to LAMMPS pair_coeff file.
    typ_sel : list of str, optional
        Only load specified pair styles (e.g., ["gauss/cut"])

    Returns
    -------
    pair2potential : dict
        Mapping (type1, type2) → GaussianPotential
    """
    pair2potential = {}
    with open(file, "r") as f:
        for line in f:
            if "pair_coeff" not in line:
                continue
            tmp = line.split()
            style = tmp[3]
            if typ_sel is None or style in typ_sel:
                if style in ("gauss/wall", "gauss/cut"):
                    pair2potential[(tmp[1], tmp[2])] = GaussianPotential(
                        tmp[1], tmp[2], float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])
                    )
    return pair2potential


def WriteLmpFF(
    old_file: str,
    new_file: str,
    L_new: np.ndarray,
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
    L_new : np.ndarray
        New parameter vector.
    typ_sel : list of str, optional
        Only update selected pair styles.
    """
    idx = 0
    with open(old_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "pair_coeff" in line:
            tmp = line.split()
            style = tmp[3]
            if typ_sel is None or style in typ_sel:
                if style in ("gauss/wall", "gauss/cut"): # gaussian potential
                    for j in range(3):
                        tmp[4 + j] = str(L_new[idx + j])
                    idx += 3
                    lines[i] = "   ".join(tmp) + "\n"

    with open(new_file, "w") as f:
        f.writelines(lines)
