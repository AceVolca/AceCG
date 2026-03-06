"""Helpers to parse OpenMSCG-style top.in and attach bonded topology to MDAnalysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
from MDAnalysis import Universe


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


def _canonical_bond_type(ti: str, tj: str) -> str:
    return f"{ti}:{tj}" if ti <= tj else f"{tj}:{ti}"


def _canonical_angle_type(ti: str, tj: str, tk: str) -> str:
    left = f"{ti}:{tj}:{tk}"
    right = f"{tk}:{tj}:{ti}"
    return left if left <= right else right


def _canonical_dihedral_type(ti: str, tj: str, tk: str, tl: str) -> str:
    left = f"{ti}:{tj}:{tk}:{tl}"
    right = f"{tl}:{tk}:{tj}:{ti}"
    return left if left <= right else right


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
    bonds: List[Tuple[int, int]] = []
    angles: List[Tuple[int, int, int]] = []
    dihedrals: List[Tuple[int, int, int, int]] = []

    for mol_type_id, count in top.system_counts:
        templ = top.molecule_templates[mol_type_id]
        bonds_local = [(a - 1, b - 1) for (a, b) in templ.bonds_1based]
        angles_local = _generate_angles_from_bonds(templ.n_sites, bonds_local)
        dihs_local = _generate_dihedrals_from_bonds(templ.n_sites, bonds_local)

        for _ in range(count):
            shift = len(atom_types)
            atom_types.extend(templ.site_types)
            bonds.extend([(i + shift, j + shift) for (i, j) in bonds_local])
            angles.extend([(i + shift, j + shift, k + shift) for (i, j, k) in angles_local])
            dihedrals.extend(
                [(i + shift, j + shift, k + shift, l + shift) for (i, j, k, l) in dihs_local]
            )

    n_atom = len(atom_types)
    if top.n_cgsites_declared and n_atom != top.n_cgsites_declared:
        raise ValueError(
            f"top.in cgsites={top.n_cgsites_declared} but replicated system has {n_atom} atoms."
        )

    resids: List[int] = []
    rid = 1
    for mol_type_id, count in top.system_counts:
        templ = top.molecule_templates[mol_type_id]
        for _ in range(count):
            resids.extend([rid] * templ.n_sites)
            rid += 1

    bond_types = np.array(
        [_canonical_bond_type(atom_types[i], atom_types[j]) for (i, j) in bonds],
        dtype=object,
    )
    angle_types = np.array(
        [_canonical_angle_type(atom_types[i], atom_types[j], atom_types[k]) for (i, j, k) in angles],
        dtype=object,
    )
    dihedral_types = np.array(
        [
            _canonical_dihedral_type(atom_types[i], atom_types[j], atom_types[k], atom_types[l])
            for (i, j, k, l) in dihedrals
        ],
        dtype=object,
    )

    return {
        "atom_types": np.asarray(atom_types, dtype=object),
        "resids": np.asarray(resids, dtype=np.int64),
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
    """Attach types/resids/bonds/angles/dihedrals to a Universe when missing.

    Existing bonded topology is authoritative by default and is preserved unless
    `overwrite_existing=True`.
    """
    parsed = parse_mscg_top(top_path)
    arrays = build_replicated_topology_arrays(parsed)

    if len(u.atoms) != len(arrays["atom_types"]):
        raise ValueError(
            f"Universe has {len(u.atoms)} atoms but top.in implies {len(arrays['atom_types'])}."
        )

    existing_types = np.asarray(u.atoms.types).astype(str) if hasattr(u.atoms, "types") else None

    if overwrite_existing or not hasattr(u.atoms, "types"):
        u.add_TopologyAttr("types", arrays["atom_types"].astype(str))
    if overwrite_existing or not hasattr(u.atoms, "resids"):
        u.add_TopologyAttr("resids", arrays["resids"])

    if overwrite_existing or not hasattr(u, "bonds") or len(u.bonds) == 0:
        if arrays["bonds"].size > 0:
            u.add_TopologyAttr("bonds", arrays["bonds"])
    if overwrite_existing or not hasattr(u, "angles") or len(u.angles) == 0:
        if arrays["angles"].size > 0:
            u.add_TopologyAttr("angles", arrays["angles"])
    if overwrite_existing or not hasattr(u, "dihedrals") or len(u.dihedrals) == 0:
        if arrays["dihedrals"].size > 0:
            u.add_TopologyAttr("dihedrals", arrays["dihedrals"])

    # Build label->runtime-type alias map so FM interaction labels from top.in
    # can be matched against trajectories that store numeric type ids.
    alias: Dict[str, str] = {}
    for name in parsed.atom_type_names:
        alias[str(name)] = str(name)

    if existing_types is not None and existing_types.shape[0] == arrays["atom_types"].shape[0]:
        grouped: Dict[str, set] = {}
        for lbl, rt in zip(arrays["atom_types"].astype(str), existing_types):
            grouped.setdefault(str(lbl), set()).add(str(rt))
        for lbl, vals in grouped.items():
            if len(vals) == 1:
                alias[lbl] = next(iter(vals))

    # Fallback for typical LAMMPS type-id encoding 1..N if atom-wise mapping above
    # cannot be inferred (e.g., missing per-atom type labels).
    if existing_types is not None:
        uniq_rt = sorted(set(existing_types.tolist()))
        if all(v.isdigit() for v in uniq_rt) and len(uniq_rt) == len(parsed.atom_type_names):
            for i, name in enumerate(parsed.atom_type_names, start=1):
                alias.setdefault(str(name), str(i))
        for rt in uniq_rt:
            alias.setdefault(str(rt), str(rt))

    setattr(u, "_acecg_type_alias", alias)

    return arrays
