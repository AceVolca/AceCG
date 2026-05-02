"""Helpers to parse OpenMSCG-style top.in and attach bonded topology to MDAnalysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
from MDAnalysis import Universe

from .types import InteractionKey


def _strip_comments(lines: Iterable[str]) -> List[str]:
    out: List[str] = []
    for raw in lines:
        line = raw.split("//", 1)[0].strip()
        if line:
            out.append(line)
    return out


@dataclass
class MoleculeTemplate:
    """Per-molecule topology description parsed from top.in."""

    n_sites: int
    bond_mode: int
    site_types: List[str]
    bonds_1based: List[Tuple[int, int]]


@dataclass
class MSCGTopology:
    """Minimal topology representation from OpenMSCG `top.in`."""

    atom_type_names: List[str]
    molecule_templates: Dict[int, MoleculeTemplate]
    system_counts: List[Tuple[int, int]]
    n_cgsites_declared: int


def parse_mscg_top(top_path: Union[str, Path]) -> MSCGTopology:
    """Parse OpenMSCG `top.in` format used by cgfm."""
    lines = _strip_comments(Path(top_path).read_text().splitlines())
    idx = 0

    def _take() -> str:
        nonlocal idx
        if idx >= len(lines):
            raise ValueError("Unexpected end of top.in")
        val = lines[idx]
        idx += 1
        return val

    head = _take().split()
    if len(head) != 2 or head[0].lower() != "cgsites":
        raise ValueError("Expected `cgsites <N>` at beginning of top.in")
    n_cgsites_declared = int(head[1])

    head = _take().split()
    if len(head) != 2 or head[0].lower() != "cgtypes":
        raise ValueError("Expected `cgtypes <N>`")
    n_cgtypes = int(head[1])
    atom_type_names = [_take().split()[0] for _ in range(n_cgtypes)]

    head = _take().split()
    if len(head) != 2 or head[0].lower() != "moltypes":
        raise ValueError("Expected `moltypes <N>`")
    n_moltypes = int(head[1])

    mol_templates: Dict[int, MoleculeTemplate] = {}
    for _ in range(n_moltypes):
        toks = _take().split()
        if len(toks) != 3 or toks[0].lower() != "mol":
            raise ValueError("Expected `mol <n_sites> <bond_mode>` line")
        n_sites = int(toks[1])
        bond_mode = int(toks[2])

        if _take().lower() != "sitetypes":
            raise ValueError("Expected `sitetypes` section")
        site_types = [_take().split()[0] for _ in range(n_sites)]

        btoks = _take().split()
        if len(btoks) != 2 or btoks[0].lower() != "bonds":
            raise ValueError("Expected `bonds <N>` section")
        n_bonds = int(btoks[1])
        bonds_1based: List[Tuple[int, int]] = []
        for _ in range(n_bonds):
            a, b = _take().split()[:2]
            bonds_1based.append((int(a), int(b)))

        mol_id = len(mol_templates) + 1
        mol_templates[mol_id] = MoleculeTemplate(
            n_sites=n_sites,
            bond_mode=bond_mode,
            site_types=site_types,
            bonds_1based=bonds_1based,
        )

    stoks = _take().split()
    if len(stoks) != 2 or stoks[0].lower() != "system":
        raise ValueError("Expected `system <N>` section")
    n_system_rows = int(stoks[1])
    system_counts: List[Tuple[int, int]] = []
    for _ in range(n_system_rows):
        mtid, count = _take().split()[:2]
        system_counts.append((int(mtid), int(count)))

    return MSCGTopology(
        atom_type_names=atom_type_names,
        molecule_templates=mol_templates,
        system_counts=system_counts,
        n_cgsites_declared=n_cgsites_declared,
    )


def _generate_angles_from_bonds(
    n_sites: int,
    bonds_0based: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int, int]]:
    nbr: List[List[int]] = [[] for _ in range(n_sites)]
    for i, j in bonds_0based:
        nbr[i].append(j)
        nbr[j].append(i)

    angles: List[Tuple[int, int, int]] = []
    for center in range(n_sites):
        neigh = sorted(set(nbr[center]))
        for i, k in combinations(neigh, 2):
            angles.append((i, center, k))
    return angles


def _generate_dihedrals_from_bonds(
    n_sites: int,
    bonds_0based: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int, int, int]]:
    nbr: List[List[int]] = [[] for _ in range(n_sites)]
    for i, j in bonds_0based:
        nbr[i].append(j)
        nbr[j].append(i)

    seen = set()
    dihs: List[Tuple[int, int, int, int]] = []
    for j in range(n_sites):
        for k in nbr[j]:
            if j == k:
                continue
            for i in nbr[j]:
                if i == k:
                    continue
                for l in nbr[k]:
                    if l == j:
                        continue
                    quad = (i, j, k, l)
                    rev = (l, k, j, i)
                    key = quad if quad <= rev else rev
                    if key in seen:
                        continue
                    seen.add(key)
                    dihs.append(quad)
    return dihs


def build_replicated_topology_arrays(top: MSCGTopology) -> Dict[str, np.ndarray]:
    """Build replicated atom types and bonded term arrays for the full system."""
    atom_types: List[str] = []
    atom_names: List[str] = []
    bonds: List[Tuple[int, int]] = []
    angles: List[Tuple[int, int, int]] = []
    dihedrals: List[Tuple[int, int, int, int]] = []
    atom_resindex: List[int] = []
    resids: List[int] = []
    resnames: List[str] = []
    molnums: List[int] = []
    moltypes: List[str] = []
    resid = 1
    molnum = 0

    for mol_type_id, count in top.system_counts:
        templ = top.molecule_templates[mol_type_id]
        bonds_local = [(a - 1, b - 1) for (a, b) in templ.bonds_1based]
        angles_local = _generate_angles_from_bonds(templ.n_sites, bonds_local)
        dihs_local = _generate_dihedrals_from_bonds(templ.n_sites, bonds_local)
        moltype = f"MOL{int(mol_type_id)}"

        for _ in range(count):
            shift = len(atom_types)
            atom_types.extend(templ.site_types)
            atom_names.extend(templ.site_types)
            bonds.extend([(i + shift, j + shift) for (i, j) in bonds_local])
            angles.extend([(i + shift, j + shift, k + shift) for (i, j, k) in angles_local])
            dihedrals.extend(
                [(i + shift, j + shift, k + shift, l + shift) for (i, j, k, l) in dihs_local]
            )
            atom_resindex.extend([molnum] * templ.n_sites)
            resids.append(resid)
            resnames.append(moltype)
            molnums.append(molnum)
            moltypes.append(moltype)
            resid += 1
            molnum += 1

    n_atom = len(atom_types)
    if top.n_cgsites_declared and n_atom != top.n_cgsites_declared:
        raise ValueError(
            f"top.in cgsites={top.n_cgsites_declared} but replicated system has {n_atom} atoms."
        )

    bond_types = np.array(
        [":".join(InteractionKey.bond(atom_types[i], atom_types[j]).types) for (i, j) in bonds],
        dtype=object,
    )
    angle_types = np.array(
        [
            ":".join(InteractionKey.angle(atom_types[i], atom_types[j], atom_types[k]).types)
            for (i, j, k) in angles
        ],
        dtype=object,
    )
    dihedral_types = np.array(
        [
            ":".join(
                InteractionKey.dihedral(
                    atom_types[i], atom_types[j], atom_types[k], atom_types[l]
                ).types
            )
            for (i, j, k, l) in dihedrals
        ],
        dtype=object,
    )

    return {
        "atom_types": np.asarray(atom_types, dtype=object),
        "names": np.asarray(atom_names, dtype=object),
        "atom_resindex": np.asarray(atom_resindex, dtype=np.int64),
        "n_residues": np.asarray(len(resids), dtype=np.int64),
        "resids": np.asarray(resids, dtype=np.int64),
        "resnames": np.asarray(resnames, dtype=object),
        "molnums": np.asarray(molnums, dtype=np.int64),
        "moltypes": np.asarray(moltypes, dtype=object),
        "bonds": np.asarray(bonds, dtype=np.int64) if bonds else np.empty((0, 2), dtype=np.int64),
        "angles": np.asarray(angles, dtype=np.int64) if angles else np.empty((0, 3), dtype=np.int64),
        "dihedrals": np.asarray(dihedrals, dtype=np.int64)
        if dihedrals
        else np.empty((0, 4), dtype=np.int64),
        "bond_types": bond_types,
        "angle_types": angle_types,
        "dihedral_types": dihedral_types,
    }


def attach_topology_from_mscg_top(
    u: Universe,
    top_path: Union[str, Path],
    *,
    overwrite_existing: bool = False,
) -> Dict[str, np.ndarray]:
    """Attach types/resids/bonds/angles/dihedrals to a Universe when missing."""
    parsed = parse_mscg_top(top_path)
    arrays = build_replicated_topology_arrays(parsed)

    if len(u.atoms) != len(arrays["atom_types"]):
        raise ValueError(
            f"Universe has {len(u.atoms)} atoms but top.in implies {len(arrays['atom_types'])}."
        )

    existing_types = np.asarray(u.atoms.types).astype(str) if hasattr(u.atoms, "types") else None

    if overwrite_existing or not hasattr(u.atoms, "types"):
        u.add_TopologyAttr("types", arrays["atom_types"].astype(str))
    elif len(u.atoms.types) == len(arrays["atom_types"]):
        u.atoms.types = arrays["atom_types"].astype(str)
    if overwrite_existing or not hasattr(u.atoms, "names"):
        u.add_TopologyAttr("names", arrays["names"].astype(str))
    elif len(u.atoms.names) == len(arrays["names"]):
        u.atoms.names = arrays["names"].astype(str)
    if len(u.residues) == int(arrays["n_residues"]):
        if overwrite_existing or not hasattr(u.residues, "resids"):
            u.add_TopologyAttr("resids", arrays["resids"])
        else:
            u.residues.resids = arrays["resids"]
        if overwrite_existing or not hasattr(u.residues, "resnames"):
            u.add_TopologyAttr("resnames", arrays["resnames"].astype(str))
        else:
            u.residues.resnames = arrays["resnames"].astype(str)
        if overwrite_existing or not hasattr(u.residues, "molnums"):
            u.add_TopologyAttr("molnums", arrays["molnums"])
        else:
            u.residues.molnums = arrays["molnums"]
        if overwrite_existing or not hasattr(u.residues, "moltypes"):
            u.add_TopologyAttr("moltypes", arrays["moltypes"].astype(str))
        else:
            u.residues.moltypes = arrays["moltypes"].astype(str)

    if overwrite_existing or not hasattr(u, "bonds") or len(u.bonds) == 0:
        if arrays["bonds"].size > 0:
            u.add_TopologyAttr("bonds", arrays["bonds"])
    if overwrite_existing or not hasattr(u, "angles") or len(u.angles) == 0:
        if arrays["angles"].size > 0:
            u.add_TopologyAttr("angles", arrays["angles"])
    if overwrite_existing or not hasattr(u, "dihedrals") or len(u.dihedrals) == 0:
        if arrays["dihedrals"].size > 0:
            u.add_TopologyAttr("dihedrals", arrays["dihedrals"])

    atom_type_map: Dict[str, int] = {}
    if existing_types is not None:
        uniq_rt = sorted(set(existing_types.tolist()))
        if all(v.isdigit() for v in uniq_rt) and len(uniq_rt) == len(parsed.atom_type_names):
            for i, name in enumerate(parsed.atom_type_names, start=1):
                atom_type_map[str(name)] = i
    if not atom_type_map:
        for i, name in enumerate(parsed.atom_type_names, start=1):
            atom_type_map[str(name)] = i

    def _unique_ordered(arr: np.ndarray) -> Dict[str, int]:
        seen: Dict[str, int] = {}
        for val in arr:
            s = str(val)
            if s not in seen:
                seen[s] = len(seen) + 1
        return seen

    bond_type_map = _unique_ordered(arrays["bond_types"]) if len(arrays["bond_types"]) else {}
    angle_type_map = _unique_ordered(arrays["angle_types"]) if len(arrays["angle_types"]) else {}
    dihedral_type_map = _unique_ordered(arrays["dihedral_types"]) if len(arrays["dihedral_types"]) else {}

    type_translation = {
        "atom_types": {str(key): str(value) for key, value in atom_type_map.items()},
        "bond_types": {f"bond:{str(key)}": str(value) for key, value in bond_type_map.items()},
        "angle_types": {f"angle:{str(key)}": str(value) for key, value in angle_type_map.items()},
        "dihedral_types": {
            f"dihedral:{str(key)}": str(value) for key, value in dihedral_type_map.items()
        },
    }
    setattr(u, "_type_translation", type_translation)

    return arrays
