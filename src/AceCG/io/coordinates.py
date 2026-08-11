# AceCG/io/coordinates.py
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
- Optional wrapping:
  when a valid box exists, CG coordinates can be wrapped into the primary unit
  cell before writing files (recommended for GRO/LAMMPS).
- Outputs:
  write .gro / .pdb / LAMMPS data with residue information.
- Returns:
  a pure-Python dict schema suitable for pickle / multiprocessing.

Return schema
-------------
cg = {
  "coords_A": np.ndarray (N,3),          # Å
  "box_A": np.ndarray (6,) or None,      # [lx,ly,lz,alpha,beta,gamma] in Å/deg
  "beads": [
     {"bead_id": int, "bead_type": str, "resid": int, "resname": str,
      "aa_indices": List[int], "mass": float, "q": float},
     ...
  ],
  "type_masses": {bead_type: float},
  "type2id": {bead_type: int},
}

Notes
-----
- AA atom indices are interpreted as 0-based MDAnalysis "index" by default.
  If your mapping YAML was built with 1-based indices, use `index_base=1`.
- PDB coordinate unit is Å. GRO unit is nm, thus Å -> nm conversion is applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml
import MDAnalysis as mda
from MDAnalysis.lib.distances import minimize_vectors, apply_PBC
from MDAnalysis.lib.mdamath import triclinic_vectors

from .coordinates_writers import write_gro, write_lammps_data, write_pdb


# -----------------------------
# YAML / mapping helpers
# -----------------------------

def load_mapping_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load an AceCG coordinate mapping YAML file.

    Parameters
    ----------
    path : str or Path
        YAML file path.

    Returns
    -------
    dict
        Parsed mapping dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _as_int(x: Any) -> int:
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"Expected int-like value, got {x!r}") from e


def bead_aa_indices(
    mapping: Dict[str, Any],
    bead_type: str,
    base: int,
    *,
    index_base: int = 0,
) -> List[int]:
    """
    Absolute AA indices (0-based, MDAnalysis 'index') for one bead instance.

    base = residue_anchor + residue_offset*repeat_i + site_anchor

    Parameters
    ----------
    index_base
        Index base used in mapping YAML. Use 0 for 0-based, 1 for 1-based.
        This is applied to BOTH site-type indices and base anchors consistently.

    Returns
    -------
    list[int]
        0-based MDAnalysis atom indices.
    """
    rel = mapping["site-types"][bead_type]["index"]
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1.")
    shift = 0 if index_base == 0 else -1
    # base is in the same base convention as YAML; shift converts to 0-based.
    base0 = base + shift
    return [base0 + (_as_int(x) + shift) for x in rel]


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


def classify_box(
    box: Optional[np.ndarray],
) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
    """Classify an MDAnalysis-style unit cell.

    One place to answer "is this box usable, and is it triclinic?", a test that
    used to be spelled out inline in several modules.

    Parameters
    ----------
    box : np.ndarray or None
        ``[lx, ly, lz]`` or ``[lx, ly, lz, alpha, beta, gamma]`` in Å/deg.

    Returns
    -------
    kind : {"none", "ortho", "triclinic"}
        ``"none"`` when the box is absent, non-finite, or has a non-positive
        edge — callers should then skip all PBC work.
    lengths : np.ndarray or None
        ``(3,)`` float64 edge lengths; ``None`` when ``kind == "none"``.
    matrix : np.ndarray or None
        ``(3, 3)`` float64 lattice vectors as rows (``triclinic_vectors``
        convention), only for ``kind == "triclinic"``.
    """
    if box is None:
        return "none", None, None
    arr = np.asarray(box, dtype=np.float64).ravel()
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])) or not np.all(arr[:3] > 0.0):
        return "none", None, None
    lengths = arr[:3].copy()
    if arr.size < 6 or bool(np.allclose(arr[3:6], 90.0)):
        return "ortho", lengths, None
    return "triclinic", lengths, np.asarray(triclinic_vectors(arr[:6]), dtype=np.float64)


def minimum_image_displacements(
    displacements: np.ndarray,
    box: np.ndarray,
    *,
    triclinic: str = "exact",
    scratch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reduce displacement vectors to their minimum image, **in place**.

    Works in the input dtype and returns the same array, so a float32 frame
    buffer is imaged without a float64 round trip. This is the primitive behind
    bead-level and molecule-level "make whole".

    Parameters
    ----------
    displacements : np.ndarray
        ``(..., 3)`` displacements, modified in place.
    box : np.ndarray
        Unit cell, see :func:`classify_box`. A ``"none"`` box is a no-op.
    triclinic : {"exact", "fast"}, default "exact"
        Strategy for non-orthorhombic cells. ``"exact"`` delegates to
        :func:`MDAnalysis.lib.distances.minimize_vectors`, which is a true
        minimum image even for displacements spanning several cells. ``"fast"``
        rounds in fractional coordinates: ~9x cheaper, but it only guarantees
        *a* lattice image whose fractional components lie in ``[-0.5, 0.5]``,
        which is not the shortest vector for skewed cells (measured agreement
        with the exact result: 100% orthorhombic, ~83% at gamma=60 deg, ~68%
        for a 45/45/45 cell). Opt in only after verifying it is adequate.
    scratch : np.ndarray, optional
        Buffer shaped and typed like ``displacements``, reused for the
        orthorhombic path's intermediate to avoid per-call allocation.

    Returns
    -------
    np.ndarray
        ``displacements``, imaged in place.

    Notes
    -----
    ``d -= L * rint(d / L)`` is a true minimum image. OpenMSCG's ``cgmap``
    instead applies at most a single +-1 box shift
    (``d += (d/L > 0.5) * -L; d += (d/L < -0.5) * L``), equivalent to
    ``d -= L * clip(rint(d / L), -1, 1)`` and therefore correct only while
    ``|d| < L``. See ``unwrap="deprecated"`` in
    :class:`AceCG.compute.cgmap.CGMapper` for the bit-compatible reproduction.
    """
    kind, lengths, matrix = classify_box(box)
    if kind == "none":
        return displacements
    if kind == "ortho":
        edges = lengths.astype(displacements.dtype, copy=False)
        if (
            scratch is None
            or scratch.shape != displacements.shape
            or scratch.dtype != displacements.dtype
        ):
            scratch = np.empty_like(displacements)
        np.divide(displacements, edges, out=scratch)
        np.rint(scratch, out=scratch)
        np.multiply(scratch, edges, out=scratch)
        np.subtract(displacements, scratch, out=displacements)
        return displacements
    flat = displacements.reshape(-1, 3)
    if triclinic == "exact":
        flat[:] = minimize_vectors(
            np.ascontiguousarray(flat), box=np.asarray(box, dtype=np.float64).ravel()[:6]
        )
        return displacements
    if triclinic != "fast":
        raise ValueError(f"triclinic must be 'exact' or 'fast', got {triclinic!r}")
    inverse = np.linalg.inv(matrix).astype(flat.dtype, copy=False)
    lattice = matrix.astype(flat.dtype, copy=False)
    fractional = flat @ inverse
    np.subtract(fractional, np.rint(fractional), out=fractional)
    flat[:] = fractional @ lattice
    return displacements


def make_bead_positions_whole(positions_A: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    """
    MIC-map all atoms in a bead relative to the first atom.

    This is a local "make whole" that ensures atoms within the bead are contiguous
    under periodic boundary conditions, without needing bond topology.
    """
    if positions_A.shape[0] <= 1:
        return positions_A
    ref = positions_A[0].copy()
    dr = np.array(positions_A - ref, dtype=np.float64)
    minimum_image_displacements(dr, dimensions)
    return ref + dr


def wrap_positions_in_box(
    positions: np.ndarray,
    box: np.ndarray,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Wrap positions into the primary unit cell.

    Accepts a single frame ``(n_atoms, 3)`` with box ``(6,)`` or a batch
    ``(..., n_atoms, 3)`` with per-sample boxes ``(..., 6)`` in MDAnalysis
    ``[lx, ly, lz, alpha, beta, gamma]`` (Å/deg) format. ``box`` may also be a
    single ``(6,)`` broadcast across a batch.

    Orthorhombic boxes (all angles 90°, or length-only boxes) take a fast
    per-axis modulo that vectorizes over the whole batch; triclinic boxes fall
    back to MDAnalysis ``apply_PBC`` (looped per frame). A box with any
    non-positive length is treated as absent and the positions are returned
    unchanged.

    Parameters
    ----------
    out : np.ndarray, optional
        Destination array. When supplied the result is written into it (casting
        to its dtype), so passing the input array wraps in place — useful for a
        float32 frame buffer. When omitted a new float64 array is returned,
        preserving the historical contract.

    Notes
    -----
    ``np.mod`` can return the box length itself instead of a value strictly
    below it: in float32, ``np.mod(-1e-8, 338.07)`` is exactly ``338.07``. A
    coordinate equal to ``L`` lies outside the half-open cell ``[0, L)`` and
    trips downstream readers that bin by ``floor(x / L)``, so such values are
    clamped to ``0``.

    The clamp is applied again in *out*'s own dtype after the cast. Comparing
    only in float64 is not enough: a float64 result one float32 ulp below ``L``
    rounds up to exactly ``L`` on the way into a float32 buffer, which is the
    buffer every TrajMap run wraps into (``CGMapper`` is built with
    ``out_dtype=np.float32``). See FINDINGS.md F-034.
    """
    pos = np.asarray(positions, dtype=np.float64)
    box_arr = np.asarray(box, dtype=np.float64)
    lengths = box_arr[..., :3]
    if np.any(lengths <= 0.0):
        if out is not None and out is not positions:
            out[...] = positions
            return out
        return positions
    orthorhombic = box_arr.shape[-1] < 6 or bool(np.allclose(box_arr[..., 3:6], 90.0))
    if orthorhombic:
        # Per-axis modulo; ``lengths[..., None, :]`` broadcasts the box over the
        # atom axis for both single ``(n,3)`` and batched ``(...,n,3)`` inputs.
        edges = lengths[..., None, :]
        wrapped = np.mod(pos, edges)
        np.copyto(wrapped, 0.0, where=wrapped >= edges)
    elif pos.ndim <= 2:
        wrapped = apply_PBC(pos, box=box_arr)
    else:
        flat_pos = pos.reshape((-1,) + pos.shape[-2:])
        flat_box = np.broadcast_to(
            box_arr, pos.shape[:-2] + (box_arr.shape[-1],)
        ).reshape((-1, box_arr.shape[-1]))
        wrapped = np.empty_like(flat_pos)
        for i in range(flat_pos.shape[0]):
            wrapped[i] = apply_PBC(flat_pos[i], box=flat_box[i])
        wrapped = wrapped.reshape(pos.shape)
    if out is None:
        return wrapped
    out[...] = wrapped
    if orthorhombic:
        # Re-clamp in out's dtype: the cast above can round a value that was
        # strictly below L up onto L exactly.
        np.copyto(out, 0.0, where=out >= lengths[..., None, :].astype(out.dtype))
    return out


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
        if "x-weight" in st and not isinstance(st["x-weight"], list):
            raise ValueError(f"site-types['{bead_type}']['x-weight'] must be a list if provided.")
        if "q" in st and not isinstance(st["q"], (int, float)):
            raise ValueError(f"site-types['{bead_type}']['q'] must be a number if provided.")

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
    index_base: int = 0,
    strict_weights: bool = True,
) -> None:
    """
    Checks that:
    - bead AA indices are within range for the given Universe
    - x-weight length matches bead atom count (if strict)
    """
    n_atoms = u.atoms.n_atoms
    for gi, grp in enumerate(mapping["system"]):
        anchor = _as_int(grp["anchor"])
        offset = _as_int(grp["offset"])
        repeat = _as_int(grp["repeat"])
        sites = grp["sites"]

        for rep_i in range(repeat):
            for bead_type, site_anchor in sites:
                bead_type = str(bead_type)
                site_anchor = _as_int(site_anchor)
                base = anchor + offset * rep_i + site_anchor
                aa_idx = bead_aa_indices(mapping, bead_type, base, index_base=index_base)

                if len(aa_idx) == 0:
                    raise ValueError(f"Empty aa_idx for bead_type={bead_type} in system[{gi}] rep={rep_i}.")

                if min(aa_idx) < 0 or max(aa_idx) >= n_atoms:
                    raise ValueError(
                        f"AA index out of range for bead_type={bead_type} in system[{gi}] rep={rep_i}: "
                        f"min={min(aa_idx)}, max={max(aa_idx)}, AA_n_atoms={n_atoms}. "
                        f"Check anchor/offset/site_anchor/index base; consider index_base={index_base}."
                    )

                if strict_weights:
                    xw = mapping["site-types"][bead_type].get("x-weight", None)
                    if xw is not None and len(xw) != len(aa_idx):
                        raise ValueError(
                            f"x-weight length mismatch for bead_type={bead_type}: "
                            f"len(x-weight)={len(xw)} vs len(index)={len(aa_idx)}."
                        )


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
    wrap: bool = True,
    index_base: int = 0,
    strict_sanity: bool = True,
    strict_weights: bool = True,
    outputs: Optional[Dict[str, Union[str, Path]]] = None,
    title: str = "CG generated by AceCG",
    lammps_atom_style: str = "full",
) -> Dict[str, Any]:
    """
    Build CG coordinates from AA coordinates and a mapping.

    Parameters
    ----------
    aa_coord
        AA coordinate file path in any MDAnalysis-supported format.
    mapping
        Either a mapping dict OR a YAML file path to load.
    aa_topology
        Optional topology file if required (e.g., XTC needs GRO/TPR).
    resname
        Residue name specification:
        - str: same resname for all residues
        - Sequence[str]: one resname per mapping["system"] group (applied to its repeats)
    resname_repeat_suffix
        If True, append repeat index to resname, e.g. "POPC1", "POPC2", ...
        If False, all repeats share the same resname for that group.
    make_whole_per_bead
        If True and AA Universe has a valid box, apply MIC per-bead make-whole
        before averaging (prevents bead internal splitting).
    wrap
        If True and a valid box exists, wrap final CG coordinates into the primary
        unit cell before writing outputs. This usually fixes "split slab" visuals.
    index_base
        Index base used in mapping YAML (0 or 1). If your mapping uses 1-based
        anchors/indices, set index_base=1.
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
    lammps_atom_style
        LAMMPS atom style for data output: "full" (default) or "atomic".

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
        sanity_check_against_universe(u, mapping_dict, index_base=index_base, strict_weights=strict_weights)

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
        anchor = _as_int(grp["anchor"])
        offset = _as_int(grp["offset"])
        repeat = _as_int(grp["repeat"])
        sites = grp["sites"]
        base_resname = group_resnames[gi]

        for rep_i in range(repeat):
            resname_i = f"{base_resname}{rep_i + 1}" if resname_repeat_suffix else base_resname

            for bead_type, site_anchor in sites:
                bead_type = str(bead_type)
                site_anchor = _as_int(site_anchor)

                base = anchor + offset * rep_i + site_anchor
                aa_idx = bead_aa_indices(mapping_dict, bead_type, base, index_base=index_base)

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

                q = float(mapping_dict["site-types"][bead_type].get("q", 0.0))

                beads.append(
                    {
                        "bead_id": bead_id,
                        "bead_type": bead_type,
                        "resid": resid,
                        "resname": resname_i,
                        "aa_indices": list(map(int, aa_idx)),
                        "mass": float(mass),
                        "q": q,
                    }
                )
                coords_A_list.append(xyz_A)
                bead_id += 1

            resid += 1

    coords_A = np.asarray(coords_A_list, dtype=float)

    # Wrap final CG coordinates if requested and box exists
    if wrap and box_A is not None:
        coords_A = wrap_positions_in_box(coords_A, box_A)

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
            write_lammps_data(
                outputs["data"], title, coords_A, beads, type2id, type_masses, box_A,
                atom_style=lammps_atom_style,
            )

    return cg
