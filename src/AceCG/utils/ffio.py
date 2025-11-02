# AceCG/utils/ffio.py
import os
import numpy as np
from typing import Dict, Tuple, Optional, List
from ..potentials.base import BasePotential

from ..potentials.gaussian import GaussianPotential
from ..potentials.lennardjones import LennardJonesPotential
from ..potentials.lennardjones96 import LennardJones96Potential
from ..potentials.multi_gaussian import MultiGaussianPotential
from ..fitters.base import TABLE_FITTERS


POTENTIAL_REGISTRY = {
    "gauss/cut": GaussianPotential,
    "gauss/wall": GaussianPotential,
    "lj/cut": LennardJonesPotential,
    "lj96/cut": LennardJones96Potential,
    "table": MultiGaussianPotential,
    "double/gauss": MultiGaussianPotential, 
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
        

def ParseLmpTable(table_path: str):
    """
    Read LAMMPS pair_style table file.
    Lines format: idx r V [F]   or   r V [F]
    Returns
    -------
    r : np.ndarray
        Distances
    V : np.ndarray
        Potential values
    F : np.ndarray or None
        Forces if present in file, else None
    """
    r_list, v_list, f_list = [], [], []
    with open(table_path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                if len(parts) >= 4:      # idx r V F
                    float(parts[0]); float(parts[1]); float(parts[2]); float(parts[3])
                    # idx present
                    r_val = float(parts[1]); v_val = float(parts[2]); f_val = float(parts[3])
                    r_list.append(r_val); v_list.append(v_val); f_list.append(f_val)
                elif len(parts) == 3:    # idx r V   or   r V F
                    # try as (idx,r,V)
                    try:
                        r_val = float(parts[1]); v_val = float(parts[2])
                        r_list.append(r_val); v_list.append(v_val)
                    except Exception:
                        # try as (r,V,F)
                        r_val = float(parts[0]); v_val = float(parts[1]); f_val = float(parts[2])
                        r_list.append(r_val); v_list.append(v_val); f_list.append(f_val)
                elif len(parts) == 2:    # r V
                    r_val = float(parts[0]); v_val = float(parts[1])
                    r_list.append(r_val); v_list.append(v_val)
            except ValueError:
                continue

    if not r_list:
        raise ValueError(f"No numeric (r,V) rows found in {table_path}")

    r = np.asarray(r_list, dtype=float)
    V = np.asarray(v_list, dtype=float)
    F = np.asarray(f_list, dtype=float) if f_list else None

    # filter nan/inf, sort by r
    m = np.isfinite(r) & np.isfinite(V)
    if F is not None:
        m = m & np.isfinite(F)
    r, V = r[m], V[m]
    F = F[m] if F is not None else None
    idx = np.argsort(r)
    r, V = r[idx], V[idx]
    F = F[idx] if F is not None else None

    return r, V, F


def ReadLmpFF(
        file: str,
        pair_style: str,
        pair_typ_sel: Optional[List[str]] = None, 
        cutoff: Optional[int] = None,
        table_fit: str = "multigaussian",
        table_fit_overrides: Optional[dict] = None,
) -> Dict[Tuple[str, str], BasePotential]:
    """
    Generalized reader for LAMMPS force field files using registered potential types.

    For analytic pair styles (e.g., "gauss/cut", "lj/cut"), parameters on the
    `pair_coeff` line are parsed and used to construct the corresponding potential.

    For table-based styles (".table"), the table file is *fitted* to a potential
    via a pluggable "table fitter" selected by name.

    Parameters
    ----------
    file : str
        Path to a LAMMPS force field file containing `pair_coeff` lines.
    pair_style : str
        Pair style defined in LAMMPS; For hybrid or hybrid/overlay pair style, parse `hybrid`; 
        otherwise parse the defined pair style
    pair_typ_sel : list[str], optional
        Effective only if pair_style is hybrid. If provided, only load the specified pair styles. 
        Otherwise, all supported styles found in the file are considered.
    cutoff : int, optional
        Cutoff for all the selected pairs. If pair-specific cutoff is written in pair_coeff, 
        the "cutoff" here should be None. Otherwise, the "cutoff" here should be provided.
    table_fit : str, default "multigauss"
        Name of the table fitter to use for ".table" entries. The default
        fitter "multigauss" fits a MultiGaussianPotential with robust defaults
        (cutoff anchoring, a repulsive Gaussian component, mild sigma lower bound).
        Additional fitters can be registered in the TABLE_FITTERS registry.
    table_fit_overrides : dict, optional
        Keyword overrides passed to the selected fitter's constructor. Use this to
        tweak a small number of fitting options without changing global defaults.
        Example (multigauss):
            {"n_gauss": 12, "use_repulsive": False}

    Returns
    -------
    pair2potential : dict[(type1, type2) -> BasePotential]
        Mapping from pair identifiers to constructed/fitted potential objects.
        For ".table" lines this will be the fitted potential instance.

    Notes
    -----
    • Users who prefer to declaratively set a full configuration for the multigauss
      fitter can import and pass a config object via `table_fit_overrides`, e.g.:

        from AceCG.fitters.multigauss import MultiGaussConfig
        cfg = MultiGaussConfig(n_gauss=12, use_repulsive=True, repulsive_A_min=1e-3)
        pair2pot = ReadLmpFF("in.ff", table_fit="multigauss",
                             table_fit_overrides={"config": cfg})

      Any fields in `MultiGaussConfig` may also be overridden directly with kwargs:
        ReadLmpFF("in.ff", table_fit="multigauss",
                  table_fit_overrides={"n_gauss": 12, "anchor_span": 1.0})

    • The fitter registry (TABLE_FITTERS) allows adding alternative table-fitting
      strategies without changing this reader's logic.
    """
    assert pair_style is not None
    if pair_style != "hybrid":
        pair_typ_sel = None # turn off pair_typ_sel if not hybrid pair style
        param_offset = 3
    else:
        param_offset = 4

    base_dir = os.path.dirname(os.path.abspath(file)) # convert to absolute path

    pair2potential: Dict[Tuple[str, str], BasePotential] = {}

    with open(file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if "pair_coeff" in line:
            tmp = line.split()
            if pair_style == "hybrid":
                style = tmp[3]
            else:
                style = pair_style
            pair = (tmp[1], tmp[2])

            if pair_typ_sel is None or style in pair_typ_sel:
                if style == "table":
                    table_file = tmp[param_offset]
                    if not os.path.isabs(table_file): # convert to absolute path to the table
                        table_file = os.path.join(base_dir, table_file)
                    fitter = TABLE_FITTERS.create(table_fit, **(table_fit_overrides or {}))
                    pot = fitter.fit(table_file, typ1=pair[0], typ2=pair[1])
                    pair2potential[pair] = pot
                else:
                    # analytical lammps potentials
                    if style in POTENTIAL_REGISTRY:
                        constructor = POTENTIAL_REGISTRY[style]
                        params = list(map(float, tmp[param_offset:]))
                        if style == "double/gauss": # using multigauss for double/gauss
                            if cutoff is None: gauss_params, cutoff = params[:-1], params[-1]
                            else: gauss_params = params[:]
                            pair2potential[pair] = constructor(pair[0], pair[1], 2, cutoff, gauss_params)
                        else:
                            if cutoff is not None:
                                params.append(cutoff)
                            pair2potential[pair] = constructor(pair[0], pair[1], *params)
    return pair2potential


def WriteLmpTable(
    filename: str,
    r: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    comment: str = "LAMMPS Table written by AceCG",
    table_name: str = "Table1"
):
    """
    Write a LAMMPS-style pair_style table file from arrays of r, V(r), F(r).

    Parameters
    ----------
    filename : str
        Output file path.
    r : np.ndarray
        1D array of distances.
    V : np.ndarray
        1D array of potential energy values at r.
    F : np.ndarray
        1D array of forces (-dV/dr) at r.
    comment : str, optional
        Comment string to put at the top of the file.
    table_name : str, optional
        Label for the LAMMPS table (default "Table1"). pair_coeff * * morse.table LABEL
    """
    r = np.asarray(r, dtype=float)
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=float)
    assert r.shape == V.shape == F.shape, "r, V, F must have the same shape"

    with open(filename, "w") as f:
        if comment is not None:
            for line in comment.splitlines():
                f.write(f"# {line}\n")

        npoints = len(r)
        f.write(f"\n{table_name}\n")
        f.write(f"N {npoints} R {r[0]:.6f} {r[-1]:.6f}\n\n")

        for i, (ri, vi, fi) in enumerate(zip(r, V, F), start=1):
            f.write(f"{i:6d}  {ri:16.8f}  {vi:16.8e}  {fi:16.8e}\n")


def WriteLmpFF(
    old_file: str,
    new_file: str,
    pair2potential: Dict[Tuple[str, str], BasePotential],
    pair_style: str,
    pair_typ_sel: Optional[List[str]] = None
):
    """
    Write updated parameters to a new LAMMPS-style force field file.

    This function updates the `pair_coeff` lines in a LAMMPS force field file
    based on the current parameters stored in `pair2potential`.

    Behavior differs depending on the pair style:

    • Analytic styles (e.g., "gauss/cut", "lj/cut"):
        The numeric parameters in the corresponding `pair_coeff` line are replaced
        with the updated values from the potential object.

    • Table style (".table"):
        The `pair_coeff` line in the file is left unchanged, but the associated
        table file itself is regenerated. The new table file contains:
            - column 1: index
            - column 2: r
            - column 3: V(r)   (recomputed from the fitted potential)
            - column 4: F(r)   (recomputed as -dV/dr)
        A header comment is also written to indicate the table source.

    Parameters
    ----------
    old_file : str
        Path to the original LAMMPS force field file (to be read and used as a template).
    new_file : str
        Path to the new file to write (will contain updated coefficients or table references).
    pair2potential : dict
        Mapping (type1, type2) → BasePotential object.
        For ".table" entries, this must be a MultiGaussianPotential (or similar) with
        `.value(r)` and `.force(r)` methods implemented.
    pair_style : str
        Pair style defined in LAMMPS; For hybrid or hybrid/overlay pair style, parse `hybrid`; otherwise parse the defined pair style
    pair_typ_sel : list[str], optional
        Effective only if pair_style is hybrid. If provided, only load the specified pair styles. Otherwise, all supported
        styles found in the file are considered.

    Notes
    -----
    • For ".table" styles, the table file specified in the original pair_coeff line
      is overwritten with new (r, V, F) values computed from the potential object.
    • The main LAMMPS force field file (`new_file`) is always written,
      even if only comments or table updates were applied.
    """
    assert pair_style is not None
    if pair_style != "hybrid":
        pair_typ_sel = None # turn off pair_typ_sel if not hybrid pair style
        param_offset = 3
    else:
        param_offset = 4

    base_dir = os.path.dirname(os.path.abspath(old_file)) # convert to absolute path

    L_new = FFParamArray(pair2potential)
    idx = 0
    with open(old_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "pair_coeff" in line:
            tmp = line.split()
            if pair_style == "hybrid":
                style = tmp[3]
            else:
                style = pair_style
            pair = (tmp[1], tmp[2])

            if pair_typ_sel is None or style in pair_typ_sel:
                if pair in pair2potential:
                    if style == "table": # do not change the output line, update table file
                        table_file = tmp[param_offset]
                        if not os.path.isabs(table_file): # not absolute table
                            table_file = os.path.join(base_dir, table_file)

                        r, v, f = ParseLmpTable(table_file)
                        WriteLmpTable(
                            table_file, 
                            r, pair2potential[pair].value(r), pair2potential[pair].force(r), 
                            f"Table {tmp[param_offset]}: id, r, potential, force", tmp[param_offset+1]
                        )
                    else:
                        n_param = pair2potential[pair].n_params()
                        tmp[param_offset:param_offset + n_param] = map(str, L_new[idx:idx + n_param])
                        idx += n_param
                        lines[i] = "   ".join(tmp) + "\n"

    with open(new_file, "w") as f:

        f.writelines(lines)
