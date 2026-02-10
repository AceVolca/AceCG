# AceCG/utils/cgcoords.py
"""
AceCG CG coordinate builder.

This module builds coarse-grained (CG) bead coordinates from an all-atom (AA)
structure/trajectory and a YAML mapping file.

Key features
------------
- Mapping semantics:
  mapping["system"] describes residue groups with anchor/offset/repeat/sites.
  mapping["site-types"][bead_type] defines per-bead AA indices and optional x-weight.
- Coordinate:
  bead coordinate is computed by x-weighted average (if available), otherwise
  mass-weighted center-of-mass (COM).
- PBC robustness:
  if the AA Universe contains a valid periodic box, atoms within each bead can be
  MIC-mapped ("make whole") relative to the bead's first atom to prevent internal
  bead splitting across boundaries.
- Outputs:
  write .gro / .pdb / LAMMPS data (atomic style) with residue information.
- Returns:
  a pure-Python dict schema suitable for pickle / multiprocessing.

Return schema
-------------
cg = {
  "coords_A": np.ndarray (N,3),          # Å
  "box_A": np.ndarray (6,) or None,      # [lx,ly,lz,alpha,beta,gamma] in Å/deg
  "beads": [
     {"bead_id": int, "bead_type": str, "resid": int, "resname": str,
      "aa_indices": List[int], "mass": float},
     ...
  ],
  "type_masses": {bead_type: float},
  "type2id": {bead_type: int},
}

Notes
-----
- AA atom indices are interpreted as 0-based MDAnalysis "index".
- PDB coordinate unit is Å. GRO unit is nm, thus Å -> nm conversion is applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml
import MDAnalysis as mda
from MDAnalysis.lib.distances import minimize_vectors


# -----------------------------
# YAML / mapping helpers
# -----------------------------

def load_mapping_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load mapping YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def bead_aa_indices(mapping: Dict[str, Any], bead_type: str, base: int) -> List[int]:
    """
    Absolute AA indices (0-based, MDAnalysis 'index') for one bead instance.

    base = residue_anchor + residue_offset*repeat_i + site_anchor
    """
    rel = mapping["site-types"][bead_type]["index"]
    return [base + int(x) for x in rel]


# -----------------------------
# PBC / geometry helpers
# -----------------------------

def has_valid_box(u: mda.Universe) -> bool:
    """
    Return True if Universe has a usable periodic box.
    MDAnalysis dimensions are [lx, ly, lz, alpha, beta, gamma] in Å/deg.
    """
    if u.dimensions is None:
        return False
    dims = np.asarray(u.dimensions, dtype=float)
    if dims.shape[0] < 3:
        return False
    if not np.all(np.isfinite(dims[:3])):
        return False
    return np.all(dims[:3] > 0.0)


def make_bead_positions_whole(positions_A: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    """
    MIC-map all atoms in a bead relative to the first atom.

    This is a local "make whole" that ensures atoms within the bead are contiguous
    under periodic boundary conditions, without needing bond topology.
    """
    if positions_A.shape[0] <= 1:
        return positions_A
    ref = positions_A[0].copy()
    dr = positions_A - ref
    dr_mic = minimize_vectors(dr, box=dimensions)
    return ref + dr_mic


def bead_position_and_mass(
    u: mda.Universe,
    aa_idx: Sequence[int],
    x_weight: Optional[Sequence[float]],
    do_make_whole: bool,
) -> Tuple[np.ndarray, float]:
    """
    Compute bead coordinate (Å) and bead mass (sum of AA masses).

    - If x_weight is provided: weighted average of AA positions.
    - Else: mass-weighted COM.
    - If do_make_whole: apply MIC mapping within bead before averaging.
    """
    ag = u.atoms[list(aa_idx)]
    pos = ag.positions.astype(float)  # Å
    masses = ag.masses.astype(float)
    mass_sum = float(masses.sum())

    if do_make_whole:
        pos = make_bead_positions_whole(pos, np.asarray(u.dimensions, dtype=float))

    if x_weight is None:
        if mass_sum <= 0:
            raise ValueError("Bead mass sum <= 0; cannot compute COM.")
        xyz = (pos * masses[:, None]).sum(axis=0) / mass_sum
        return xyz, mass_sum

    w = np.asarray(x_weight, dtype=float)
    if w.shape[0] != pos.shape[0]:
        raise ValueError(f"x-weight length {w.shape[0]} != n_atoms {pos.shape[0]} for bead indices {list(aa_idx)}")
    wsum = float(w.sum())
    if wsum == 0.0:
        raise ValueError("Sum of x-weight is zero; cannot compute weighted position.")
    xyz = (pos * w[:, None]).sum(axis=0) / wsum
    return xyz, mass_sum


# -----------------------------
# Sanity checks
# -----------------------------

def sanity_check_mapping(mapping: Dict[str, Any]) -> None:
    """
    Basic structural checks on mapping dict.
    Raises ValueError with actionable messages.
    """
    if "system" not in mapping or not isinstance(mapping["system"], list):
        raise ValueError("mapping must contain a list key 'system'.")
    if "site-types" not in mapping or not isinstance(mapping["site-types"], dict):
        raise ValueError("mapping must contain a dict key 'site-types'.")

    for bead_type, st in mapping["site-types"].items():
        if "index" not in st:
            raise ValueError(f"site-types['{bead_type}'] missing 'index'.")
        if not isinstance(st["index"], list) or len(st["index"]) == 0:
            raise ValueError(f"site-types['{bead_type}']['index'] must be a non-empty list.")
        if "x-weight" in st:
            if not isinstance(st["x-weight"], list):
                raise ValueError(f"site-types['{bead_type}']['x-weight'] must be a list if provided.")

    for gi, grp in enumerate(mapping["system"]):
        for k in ("anchor", "offset", "repeat", "sites"):
            if k not in grp:
                raise ValueError(f"system[{gi}] missing key '{k}'.")
        if int(grp["repeat"]) < 1:
            raise ValueError(f"system[{gi}]['repeat'] must be >= 1.")
        if int(grp["offset"]) < 0:
            raise ValueError(f"system[{gi}]['offset'] must be >= 0.")
        if not isinstance(grp["sites"], list) or len(grp["sites"]) == 0:
            raise ValueError(f"system[{gi}]['sites'] must be a non-empty list.")


def sanity_check_against_universe(
    u: mda.Universe,
    mapping: Dict[str, Any],
    *,
    strict_weights: bool = True,
) -> None:
    """
    Checks that:
    - bead AA indices are within range for the given Universe
    - x-weight length matches bead atom count (if strict)
    """
    n_atoms = u.atoms.n_atoms
    for gi, grp in enumerate(mapping["system"]):
        anchor = int(grp["anchor"])
        offset = int(grp["offset"])
        repeat = int(grp["repeat"])
        sites = grp["sites"]

        for rep_i in range(repeat):
            for bead_type, site_anchor in sites:
                bead_type = str(bead_type)
                site_anchor = int(site_anchor)
                base = anchor + offset * rep_i + site_anchor
                aa_idx = bead_aa_indices(mapping, bead_type, base)
                if len(aa_idx) == 0:
                    raise ValueError(f"Empty aa_idx for bead_type={bead_type} in system[{gi}] rep={rep_i}.")
                if min(aa_idx) < 0 or max(aa_idx) >= n_atoms:
                    raise ValueError(
                        f"AA index out of range for bead_type={bead_type} in system[{gi}] rep={rep_i}: "
                        f"min={min(aa_idx)}, max={max(aa_idx)}, AA_n_atoms={n_atoms}. "
                        "Check anchor/offset/site_anchor/index base."
                    )

                if strict_weights:
                    xw = mapping["site-types"][bead_type].get("x-weight", None)
                    if xw is not None and len(xw) != len(aa_idx):
                        raise ValueError(
                            f"x-weight length mismatch for bead_type={bead_type}: "
                            f"len(x-weight)={len(xw)} vs len(index)={len(aa_idx)}."
                        )


# -----------------------------
# Writers
# -----------------------------

def write_gro(
    path: Union[str, Path],
    title: str,
    coords_A: np.ndarray,
    beads: Sequence[Dict[str, Any]],
    box_A: Optional[np.ndarray],
) -> None:
    """Write .gro with residue info. Coordinates converted Å -> nm."""
    path = Path(path)
    n = coords_A.shape[0]
    coords_nm = coords_A / 10.0

    with path.open("w") as f:
        f.write(f"{title}\n")
        f.write(f"{n:5d}\n")
        for i, (xyz, br) in enumerate(zip(coords_nm, beads), start=1):
            f.write(
                f"{int(br['resid']):5d}{str(br['resname']):<5s}{str(br['bead_type']):>5s}{i:5d}"
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}\n"
            )

        if box_A is not None and np.all(np.isfinite(box_A[:3])) and np.all(box_A[:3] > 0):
            box_nm = np.asarray(box_A[:3], dtype=float) / 10.0
            f.write(f"{box_nm[0]:10.5f}{box_nm[1]:10.5f}{box_nm[2]:10.5f}\n")
        else:
            f.write(f"{0.0:10.5f}{0.0:10.5f}{0.0:10.5f}\n")


def write_pdb(
    path: Union[str, Path],
    title: str,
    coords_A: np.ndarray,
    beads: Sequence[Dict[str, Any]],
) -> None:
    """Write minimal PDB with residue info. Coordinates remain in Å."""
    path = Path(path)
    with path.open("w") as f:
        f.write(f"TITLE     {title}\n")
        for i, (xyz, br) in enumerate(zip(coords_A, beads), start=1):
            name = str(br["bead_type"])[:4]
            resname = str(br["resname"])
            resname = resname[:3] if len(resname) > 3 else resname
            resid = int(br["resid"])
            f.write(
                "ATOM  {serial:5d} {name:<4s} {resname:>3s} A{resid:4d}    "
                "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bf:6.2f}          {elem:>2s}\n".format(
                    serial=i,
                    name=name,
                    resname=resname,
                    resid=resid,
                    x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                    occ=1.00, bf=0.00,
                    elem="C",
                )
            )
        f.write("END\n")


def write_lammps_data(
    path: Union[str, Path],
    title: str,
    coords_A: np.ndarray,
    beads: Sequence[Dict[str, Any]],
    type2id: Dict[str, int],
    type_masses: Dict[str, float],
    box_A: Optional[np.ndarray] = None,
) -> None:
    """
    Write minimal LAMMPS data (atomic style):
      Masses
      Atoms: atom-ID mol-ID type x y z

    - Coordinates are in Å.
    - mol-ID is set to resid.
    - If box is available, bounds are [0, L] (orthorhombic assumption).
    """
    path = Path(path)
    n_atoms = coords_A.shape[0]
    n_types = len(type2id)

    if box_A is not None and np.all(np.isfinite(box_A[:3])) and np.all(box_A[:3] > 0):
        lx, ly, lz = map(float, box_A[:3])
        xlo, xhi = 0.0, lx
        ylo, yhi = 0.0, ly
        zlo, zhi = 0.0, lz
    else:
        mins = coords_A.min(axis=0)
        maxs = coords_A.max(axis=0)
        pad = 10.0
        xlo, xhi = float(mins[0] - pad), float(maxs[0] + pad)
        ylo, yhi = float(mins[1] - pad), float(maxs[1] + pad)
        zlo, zhi = float(mins[2] - pad), float(maxs[2] + pad)

    with path.open("w") as f:
        f.write(f"{title}\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_types} atom types\n\n")
        f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n\n")

        f.write("Masses\n\n")
        for bead_type, tid in sorted(type2id.items(), key=lambda kv: kv[1]):
            f.write(f"{tid:d} {type_masses[bead_type]:.6f} # {bead_type}\n")

        f.write("\nAtoms # atomic\n\n")
        for i, (xyz, br) in enumerate(zip(coords_A, beads), start=1):
            mol_id = int(br["resid"])
            bead_type = str(br["bead_type"])
            tid = int(type2id[bead_type])
            f.write(f"{i:d} {mol_id:d} {tid:d} {float(xyz[0]):.6f} {float(xyz[1]):.6f} {float(xyz[2]):.6f}\n")


# -----------------------------
# Public API (AceCG-style entry)
# -----------------------------

def build_CG_coords(
    aa_coord: Union[str, Path],
    mapping: Union[Dict[str, Any], str, Path],
    *,
    aa_topology: Optional[Union[str, Path]] = None,
    resname: Union[str, Sequence[str]] = "CG",
    resname_repeat_suffix: bool = False,
    make_whole_per_bead: bool = True,
    strict_sanity: bool = True,
    strict_weights: bool = True,
    outputs: Optional[Dict[str, Union[str, Path]]] = None,
    title: str = "CG generated by AceCG",
) -> Dict[str, Any]:
    """
    Build CG coordinates from AA coordinates and mapping.

    Parameters
    ----------
    aa_coord
        AA coordinate file path (any MDAnalysis-supported format).
    mapping
        Either a mapping dict OR a YAML file path to load.
    aa_topology
        Optional topology file if required (e.g., XTC needs GRO/TPR).
    resname
        Residue name specification:
        - str: same resname for all residues
        - Sequence[str]: one resname per mapping["system"] group (applied to its repeats)
    resname_repeat_suffix
        If True, append repeat index to resname, e.g. "FERM1", "FERM2", ...
        If False, all repeats share the same resname.
    make_whole_per_bead
        If True and AA Universe has a valid box, apply MIC per-bead make-whole.
    strict_sanity
        If True, run mapping sanity checks and AA-index range checks.
    strict_weights
        If True, require len(x-weight) == bead atom count for all beads.
    outputs
        Optional dict controlling file outputs:
          outputs = {
            "gro": "out.gro",
            "pdb": "out.pdb",
            "data": "out.data",
          }
        Any subset is allowed.
    title
        Title used in file headers.

    Returns
    -------
    dict
        See module docstring "Return schema".
    """
    # Load mapping
    if isinstance(mapping, (str, Path)):
        mapping_dict = load_mapping_yaml(mapping)
    else:
        mapping_dict = mapping

    if strict_sanity:
        sanity_check_mapping(mapping_dict)

    # Build Universe
    if aa_topology is None:
        u = mda.Universe(str(aa_coord))
    else:
        u = mda.Universe(str(aa_topology), str(aa_coord))

    if strict_sanity:
        sanity_check_against_universe(u, mapping_dict, strict_weights=strict_weights)

    box_ok = has_valid_box(u)
    do_make_whole = bool(make_whole_per_bead and box_ok)
    box_A = np.asarray(u.dimensions, dtype=float) if box_ok else None

    system_groups = mapping_dict["system"]
    if isinstance(resname, (list, tuple)):
        if len(resname) != len(system_groups):
            raise ValueError("If resname is a list/tuple, it must match mapping['system'] length.")
        group_resnames = [str(x) for x in resname]
    else:
        group_resnames = [str(resname) for _ in system_groups]

    beads: List[Dict[str, Any]] = []
    coords_A_list: List[np.ndarray] = []

    type_masses: Dict[str, float] = {}
    type2id: Dict[str, int] = {}
    next_type_id = 1

    bead_id = 1
    resid = 1

    for gi, grp in enumerate(system_groups):
        anchor = int(grp["anchor"])
        offset = int(grp["offset"])
        repeat = int(grp["repeat"])
        sites = grp["sites"]
        base_resname = group_resnames[gi]

        for rep_i in range(repeat):
            resname_i = f"{base_resname}{rep_i + 1}" if resname_repeat_suffix else base_resname

            for bead_type, site_anchor in sites:
                bead_type = str(bead_type)
                site_anchor = int(site_anchor)

                base = anchor + offset * rep_i + site_anchor
                aa_idx = bead_aa_indices(mapping_dict, bead_type, base)

                xw = mapping_dict["site-types"][bead_type].get("x-weight", None)
                xyz_A, mass = bead_position_and_mass(u, aa_idx, xw, do_make_whole)

                if bead_type not in type2id:
                    type2id[bead_type] = next_type_id
                    next_type_id += 1

                if bead_type not in type_masses:
                    type_masses[bead_type] = float(mass)
                else:
                    if abs(type_masses[bead_type] - float(mass)) > 1e-6:
                        raise ValueError(
                            f"Inconsistent mass for bead_type={bead_type}: "
                            f"cached {type_masses[bead_type]} vs new {mass}. "
                            "This usually indicates varying AA composition for the same bead_type."
                        )

                beads.append(
                    {
                        "bead_id": bead_id,
                        "bead_type": bead_type,
                        "resid": resid,
                        "resname": resname_i,
                        "aa_indices": list(map(int, aa_idx)),
                        "mass": float(mass),
                    }
                )
                coords_A_list.append(xyz_A)
                bead_id += 1

            resid += 1

    coords_A = np.asarray(coords_A_list, dtype=float)

    cg = {
        "coords_A": coords_A,
        "box_A": box_A,
        "beads": beads,
        "type_masses": type_masses,
        "type2id": type2id,
    }

    # Optional outputs
    if outputs:
        if "gro" in outputs and outputs["gro"] is not None:
            write_gro(outputs["gro"], title, coords_A, beads, box_A)
        if "pdb" in outputs and outputs["pdb"] is not None:
            write_pdb(outputs["pdb"], title, coords_A, beads)
        if "data" in outputs and outputs["data"] is not None:
            write_lammps_data(outputs["data"], title, coords_A, beads, type2id, type_masses, box_A)

    return cg
