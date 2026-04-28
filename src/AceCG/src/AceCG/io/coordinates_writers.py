"""Output writers for coarse-grained coordinate exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from MDAnalysis.lib.distances import minimize_vectors
from MDAnalysis.lib.mdamath import triclinic_vectors


def write_gro(
    path: Union[str, Path],
    title: str,
    coords_A: np.ndarray,
    beads: Sequence[Dict[str, Any]],
    box_A: Optional[np.ndarray],
) -> None:
    """Write a GROMACS ``.gro`` coordinate file.

    Parameters
    ----------
    path : str or Path
        Destination file.
    title : str
        Header title.
    coords_A : np.ndarray
        Coordinates in Angstrom, shape ``(n_beads, 3)``. They are converted to
        nanometers for GRO output.
    beads : Sequence[dict]
        Bead records containing at least ``resid``, ``resname``, and
        ``bead_type``.
    box_A : np.ndarray or None
        Optional MDAnalysis-style box in Angstrom.
    """
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
    """Write a minimal PDB coordinate file.

    Parameters
    ----------
    path : str or Path
        Destination file.
    title : str
        PDB title line text.
    coords_A : np.ndarray
        Coordinates in Angstrom, shape ``(n_beads, 3)``.
    beads : Sequence[dict]
        Bead records containing residue and bead-name metadata.
    """
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
                    x=float(xyz[0]),
                    y=float(xyz[1]),
                    z=float(xyz[2]),
                    occ=1.00,
                    bf=0.00,
                    elem="C",
                )
            )
        f.write("END\n")


def _wrap_positions_with_images(
    positions: np.ndarray,
    dimensions: np.ndarray,
    *,
    bonds: Optional[np.ndarray] = None,
    bead_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    wrapped = np.asarray(positions, dtype=float).copy()
    images = np.zeros((wrapped.shape[0], 3), dtype=np.int64)
    dims = np.asarray(dimensions, dtype=float)
    if dims.size < 3 or not np.all(np.isfinite(dims[:3])) or not np.all(dims[:3] > 0.0):
        return wrapped, images
    if dims.size == 3:
        dims = np.array([dims[0], dims[1], dims[2], 90.0, 90.0, 90.0], dtype=float)

    lengths = np.asarray(dims[:3], dtype=float)
    bonds_array = (
        np.asarray(bonds, dtype=np.int64)
        if bonds is not None
        else np.empty((0, 2), dtype=np.int64)
    )
    if bonds_array.size == 0:
        images = np.floor(wrapped / lengths).astype(np.int64)
        wrapped = wrapped - images * lengths
        return wrapped, images

    n_atoms = int(wrapped.shape[0])
    adjacency: list[list[int]] = [[] for _ in range(n_atoms)]
    for index_i, index_j in bonds_array.reshape(-1, 2):
        atom_i = int(index_i)
        atom_j = int(index_j)
        if atom_i < 0 or atom_j < 0 or atom_i >= n_atoms or atom_j >= n_atoms:
            continue
        adjacency[atom_i].append(atom_j)
        adjacency[atom_j].append(atom_i)

    continuous = np.asarray(wrapped, dtype=float).copy()
    assigned = np.zeros(n_atoms, dtype=bool)
    visit_order: list[int] = []
    if bead_records is not None and len(bead_records) == n_atoms:
        seen_resids = set()
        for atom_index, record in enumerate(bead_records):
            resid = int(record["resid"])
            if resid in seen_resids:
                continue
            seen_resids.add(resid)
            visit_order.append(int(atom_index))
    visit_order.extend(atom_index for atom_index in range(n_atoms) if atom_index not in visit_order)

    for seed in visit_order:
        if assigned[int(seed)]:
            continue
        queue = [int(seed)]
        assigned[int(seed)] = True
        while queue:
            current = int(queue.pop(0))
            current_position = np.asarray(continuous[current], dtype=float)
            for neighbor in adjacency[current]:
                if assigned[int(neighbor)]:
                    continue
                delta = minimize_vectors(
                    (wrapped[int(neighbor)] - wrapped[current])[np.newaxis],
                    box=dims,
                )[0]
                continuous[int(neighbor)] = current_position + delta
                assigned[int(neighbor)] = True
                queue.append(int(neighbor))

    images = np.floor(continuous / lengths).astype(np.int64)
    wrapped = continuous - images * lengths
    return wrapped, images


def _box_bounds(
    positions: np.ndarray,
    box_A: Optional[np.ndarray],
    *,
    pad: float = 10.0,
) -> Tuple[float, float, float, float, float, float]:
    dims = np.asarray(box_A, dtype=float) if box_A is not None else np.zeros(0, dtype=float)
    if dims.size >= 3 and np.all(np.isfinite(dims[:3])) and np.all(dims[:3] > 0.0):
        if dims.size >= 6 and not np.allclose(dims[3:6], [90.0, 90.0, 90.0]):
            triv = np.asarray(triclinic_vectors(dims), dtype=float)
            return (0.0, float(triv[0, 0]), 0.0, float(triv[1, 1]), 0.0, float(triv[2, 2]))
        return (0.0, float(dims[0]), 0.0, float(dims[1]), 0.0, float(dims[2]))
    mins = np.min(np.asarray(positions, dtype=float), axis=0)
    maxs = np.max(np.asarray(positions, dtype=float), axis=0)
    return (
        float(mins[0] - pad),
        float(maxs[0] + pad),
        float(mins[1] - pad),
        float(maxs[1] + pad),
        float(mins[2] - pad),
        float(maxs[2] + pad),
    )


def write_lammps_data(
    path: Union[str, Path],
    title: str,
    coords_A: np.ndarray,
    beads: Sequence[Mapping[str, Any]],
    type2id: Dict[str, int],
    type_masses: Dict[str, float],
    box_A: Optional[np.ndarray] = None,
    *,
    atom_style: str = "full",
    bonds: Optional[np.ndarray] = None,
    bond_type_ids: Optional[np.ndarray] = None,
    angles: Optional[np.ndarray] = None,
    angle_type_ids: Optional[np.ndarray] = None,
    dihedrals: Optional[np.ndarray] = None,
    dihedral_type_ids: Optional[np.ndarray] = None,
) -> None:
    """Write a minimal LAMMPS data file for CG coordinates.

    Parameters
    ----------
    path : str or Path
        Destination data file.
    title : str
        File title line.
    coords_A : np.ndarray
        Coordinates in Angstrom, shape ``(n_atoms, 3)``.
    beads : Sequence[Mapping[str, Any]]
        Bead records with ids, types, residue ids, masses, and charges.
    type2id : dict[str, int]
        Mapping from bead type name to LAMMPS atom type id.
    type_masses : dict[str, float]
        Mapping from bead type name to mass.
    box_A : np.ndarray, optional
        MDAnalysis-style box in Angstrom. If omitted, bounds are inferred from
        coordinates with padding.
    atom_style : {"full", "atomic"}, default="full"
        LAMMPS atom style for the ``Atoms`` section.
    bonds, angles, dihedrals : np.ndarray, optional
        Optional bonded topology arrays using zero-based atom indices.
    bond_type_ids, angle_type_ids, dihedral_type_ids : np.ndarray, optional
        One-based type ids for each provided bonded term.
    """
    atom_style = atom_style.lower().strip()
    if atom_style not in ("atomic", "full"):
        raise ValueError("atom_style must be 'atomic' or 'full'.")

    def _term_array(values: Optional[np.ndarray], width: int) -> np.ndarray:
        if values is None:
            return np.empty((0, width), dtype=np.int64)
        arr = np.asarray(values, dtype=np.int64)
        if arr.size == 0:
            return np.empty((0, width), dtype=np.int64)
        return arr.reshape(-1, width)

    def _term_type_array(values: Optional[np.ndarray], count: int, label: str) -> np.ndarray:
        if count == 0:
            return np.empty(0, dtype=np.int64)
        if values is None:
            raise ValueError(f"{label}_type_ids are required when {label}s are provided")
        arr = np.asarray(values, dtype=np.int64).reshape(-1)
        if arr.shape[0] != count:
            raise ValueError(f"{label}_type_ids length mismatch")
        if np.any(arr < 1):
            raise ValueError(f"{label}_type_ids must be positive 1-based integers")
        return arr

    path = Path(path)
    n_atoms = coords_A.shape[0]
    n_types = len(type2id)
    bonds_array = _term_array(bonds, 2)
    angles_array = _term_array(angles, 3)
    dihedrals_array = _term_array(dihedrals, 4)
    bond_type_id_array = _term_type_array(bond_type_ids, bonds_array.shape[0], "bond")
    angle_type_id_array = _term_type_array(angle_type_ids, angles_array.shape[0], "angle")
    dihedral_type_id_array = _term_type_array(
        dihedral_type_ids,
        dihedrals_array.shape[0],
        "dihedral",
    )
    n_bond_types = int(np.max(bond_type_id_array)) if bond_type_id_array.size else 0
    n_angle_types = int(np.max(angle_type_id_array)) if angle_type_id_array.size else 0
    n_dihedral_types = int(np.max(dihedral_type_id_array)) if dihedral_type_id_array.size else 0

    wrapped_coords, image_flags = _wrap_positions_with_images(
        coords_A,
        np.asarray(box_A, dtype=float) if box_A is not None else np.zeros(6, dtype=float),
        bonds=bonds_array,
        bead_records=beads,
    )
    xlo, xhi, ylo, yhi, zlo, zhi = _box_bounds(wrapped_coords, box_A)

    with path.open("w") as f:
        f.write(f"{title}\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{bonds_array.shape[0]} bonds\n")
        f.write(f"{angles_array.shape[0]} angles\n")
        f.write(f"{dihedrals_array.shape[0]} dihedrals\n")
        f.write("0 impropers\n")
        f.write("\n")
        f.write(f"{n_types} atom types\n")
        f.write(f"{n_bond_types} bond types\n")
        f.write(f"{n_angle_types} angle types\n")
        f.write(f"{n_dihedral_types} dihedral types\n")
        f.write("\n")
        f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n")
        dims_arr = np.asarray(box_A, dtype=float) if box_A is not None else np.zeros(0, dtype=float)
        if dims_arr.size >= 6 and not np.allclose(dims_arr[3:6], [90.0, 90.0, 90.0]):
            triv = np.asarray(triclinic_vectors(dims_arr), dtype=float)
            f.write(f"{triv[1, 0]:.6f} {triv[2, 0]:.6f} {triv[2, 1]:.6f} xy xz yz\n")
        f.write("\n")

        f.write("Masses\n\n")
        for bead_type, tid in sorted(type2id.items(), key=lambda kv: kv[1]):
            f.write(f"{tid:d} {type_masses[bead_type]:.6f} # {bead_type}\n")

        f.write(f"\nAtoms # {atom_style}\n\n")
        for i, (xyz, images, br) in enumerate(zip(wrapped_coords, image_flags, beads), start=1):
            mol_id = int(br["resid"])
            bead_type = str(br.get("bead_type", br.get("type")))
            tid = int(type2id[bead_type])

            if atom_style == "atomic":
                f.write(
                    f"{i:d} {mol_id:d} {tid:d} "
                    f"{float(xyz[0]):.6f} {float(xyz[1]):.6f} {float(xyz[2]):.6f} "
                    f"{int(images[0])} {int(images[1])} {int(images[2])}\n"
                )
            else:
                q = float(br.get("q", br.get("charge", 0.0)))
                f.write(
                    f"{i:d} {mol_id:d} {tid:d} {q:.6f} "
                    f"{float(xyz[0]):.6f} {float(xyz[1]):.6f} {float(xyz[2]):.6f} "
                    f"{int(images[0])} {int(images[1])} {int(images[2])}\n"
                )

        if bonds_array.shape[0] > 0:
            f.write("\nBonds\n\n")
            for bond_id, (type_id, (index_i, index_j)) in enumerate(
                zip(bond_type_id_array, bonds_array),
                start=1,
            ):
                f.write(f"{bond_id} {int(type_id)} {int(index_i) + 1} {int(index_j) + 1}\n")

        if angles_array.shape[0] > 0:
            f.write("\nAngles\n\n")
            for angle_id, (type_id, (index_i, index_j, index_k)) in enumerate(
                zip(angle_type_id_array, angles_array),
                start=1,
            ):
                f.write(
                    f"{angle_id} {int(type_id)} {int(index_i) + 1} {int(index_j) + 1} {int(index_k) + 1}\n"
                )

        if dihedrals_array.shape[0] > 0:
            f.write("\nDihedrals\n\n")
            for dihedral_id, (type_id, (index_i, index_j, index_k, index_l)) in enumerate(
                zip(dihedral_type_id_array, dihedrals_array),
                start=1,
            ):
                f.write(
                    f"{dihedral_id} {int(type_id)} {int(index_i) + 1} {int(index_j) + 1} {int(index_k) + 1} {int(index_l) + 1}\n"
                )
