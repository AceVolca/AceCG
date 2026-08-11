# AceCG/topology/cgmap_builder.py
"""Author a mapping YAML — the write side of :mod:`AceCG.topology.cgmap`.

This is the port of OpenMSCG's ``cgyaml`` CLI. The input is one *bead table* per
residue name plus a real AA topology; the output is the ``site-types`` +
``system`` document that :class:`~AceCG.topology.cgmap.CGMapSpec` compiles, with
an optional ``cg-topology`` block carrying CG bonded terms.

Bead tables
-----------
A table assigns every mapped atom of one residue to a named bead. Two spellings:

* **by local index**, the OpenMSCG ``*_mapping.txt`` format — a bead-name line
  followed by a Python list of 0-based atom indices within the residue::

      PHG
      [0, 1, 2, 3]
      PMG
      [4, 5]

  A name may repeat (a lipid with two identical tails lists ``PT1`` twice).
* **by atom name**, which is how published CG models are actually specified::

      {"NC3": ["N", "C13", "C14"], "PO4": ["P", "O13"]}

  :func:`bead_table_from_names` resolves those against the residue's real atom
  order, optionally attaching each hydrogen to the heavy atom it follows.

Group discovery, and the bug it fixes
-------------------------------------
``cgyaml`` starts a new ``system`` group whenever the residue name changes from
the previous *mapped* residue. Unmapped residues (water, ions) do not reset that
state, so a system laid out as ``DPPC … water … DPPC …`` collapses into one group
whose ``repeat`` spans both lipid blocks — and the second block's sites then read
atom indices belonging to the water in between. :func:`build_mapping` instead
starts a new group whenever a residue is not exactly ``offset`` atoms past the
previous one, so a tiled or interleaved system produces one group per contiguous
run. For a genuinely contiguous system the two rules agree.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml


__all__ = [
    "BeadTable",
    "SiteSpec",
    "bead_table_from_names",
    "build_mapping",
    "build_mapping_from_sites",
    "derive_angles_from_bonds",
    "parse_bead_table",
    "parse_gromacs_itp",
    "write_mapping_yaml",
]


# One residue's mapping: ordered ``(bead name, local atom indices)`` pairs.
BeadTable = List[Tuple[str, Tuple[int, ...]]]

# One CG site of a heterogeneous molecule: its site-type name and the *absolute*
# 0-based AA atom indices that form it.
SiteSpec = Tuple[str, Sequence[int]]

_HYDROGEN_MASS_CEILING = 2.0


# ─── bead tables ──────────────────────────────────────────────────────


def parse_bead_table(source: Union[str, Path]) -> BeadTable:
    """Parse an OpenMSCG ``*_mapping.txt`` bead table.

    Parameters
    ----------
    source : str or Path
        Path to the table, or its text when it contains a newline.

    Returns
    -------
    BeadTable
        ``(bead name, sorted local indices)`` in file order.
    """
    text = (
        Path(source).read_text(encoding="utf-8")
        if "\n" not in str(source) and Path(source).exists()
        else str(source)
    )
    table: BeadTable = []
    pending: Optional[str] = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            if pending is None:
                raise ValueError(
                    f"line {lineno}: index list {line!r} has no bead name above it."
                )
            try:
                indices = ast.literal_eval(line)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"line {lineno}: cannot parse {line!r}.") from exc
            if not isinstance(indices, (list, tuple)) or not indices:
                raise ValueError(f"line {lineno}: {line!r} is not a non-empty list.")
            table.append((pending, tuple(sorted(int(index) for index in indices))))
            pending = None
        else:
            if pending is not None:
                raise ValueError(
                    f"line {lineno}: bead {pending!r} was not followed by an index list."
                )
            pending = line
    if pending is not None:
        raise ValueError(f"bead {pending!r} at end of table has no index list.")
    if not table:
        raise ValueError("bead table is empty.")
    return table


def bead_table_from_names(
    atom_names: Sequence[str],
    assignment: Mapping[str, Sequence[str]],
    *,
    masses: Optional[Sequence[float]] = None,
    attach_hydrogens: bool = False,
    require_complete: bool = True,
) -> BeadTable:
    """Resolve a name-keyed bead assignment against one residue's atom order.

    Parameters
    ----------
    atom_names : sequence of str
        Atom names of one residue, in topology order.
    assignment : mapping
        ``{bead name: [atom name, ...]}``. Insertion order becomes bead order, so
        it defines the CG site order within the repeat unit.
    masses : sequence of float, optional
        Per-atom masses, used to identify hydrogens. Falls back to a leading
        ``H`` in the atom name.
    attach_hydrogens : bool, default False
        Assign every hydrogen not named in *assignment* to the same bead as the
        nearest preceding non-hydrogen atom. This is the CHARMM/AMBER lipid and
        protein convention — hydrogens directly follow their parent heavy atom —
        and it keeps a 12-bead lipid table to its 46 heavy-atom names instead of
        all 130. Verify the resulting masses; the function reports them via
        :func:`build_mapping`'s ``x-weight``.
    require_complete : bool, default True
        Require every atom of the residue to end up in exactly one bead. Turn off
        for a deliberately partial mapping.

    Returns
    -------
    BeadTable
    """
    names = [str(name) for name in atom_names]
    index_of: Dict[str, int] = {}
    for index, name in enumerate(names):
        if name in index_of:
            raise ValueError(
                f"atom name {name!r} occurs twice in the residue (positions "
                f"{index_of[name]} and {index}); a name-keyed assignment needs "
                "unique names."
            )
        index_of[name] = index

    bead_of_atom: Dict[int, str] = {}
    order: List[str] = []
    for bead, wanted in assignment.items():
        bead = str(bead)
        order.append(bead)
        for atom_name in wanted:
            try:
                index = index_of[str(atom_name)]
            except KeyError:
                raise ValueError(
                    f"bead {bead!r} names atom {atom_name!r}, which the residue "
                    f"does not have. Known names: {sorted(index_of)[:8]}..."
                ) from None
            if index in bead_of_atom:
                raise ValueError(
                    f"atom {atom_name!r} is claimed by both {bead_of_atom[index]!r} "
                    f"and {bead!r}."
                )
            bead_of_atom[index] = bead

    if attach_hydrogens:
        hydrogen = _hydrogen_flags(names, masses)
        parent: Optional[int] = None
        for index in range(len(names)):
            if not hydrogen[index]:
                parent = index
                continue
            if index in bead_of_atom:
                continue
            if parent is None:
                raise ValueError(
                    f"atom {names[index]!r} looks like a hydrogen but no heavy atom "
                    "precedes it, so it cannot be attached."
                )
            if parent not in bead_of_atom:
                continue  # its heavy atom is unmapped, so leave it unmapped too
            bead_of_atom[index] = bead_of_atom[parent]

    if require_complete:
        missing = [names[i] for i in range(len(names)) if i not in bead_of_atom]
        if missing:
            raise ValueError(
                f"{len(missing)} atom(s) are in no bead: {missing[:10]}"
                f"{' ...' if len(missing) > 10 else ''}. Name them, enable "
                "attach_hydrogens, or pass require_complete=False."
            )

    grouped: Dict[str, List[int]] = {bead: [] for bead in order}
    for index, bead in bead_of_atom.items():
        grouped[bead].append(index)
    return [(bead, tuple(sorted(grouped[bead]))) for bead in order if grouped[bead]]


def _hydrogen_flags(
    names: Sequence[str], masses: Optional[Sequence[float]]
) -> List[bool]:
    """Per-atom "is a hydrogen" flags, by mass when available, else by name."""
    if masses is not None:
        mass_array = np.asarray(masses, dtype=np.float64)
        if mass_array.size == len(names) and np.all(mass_array > 0.0):
            return [bool(mass < _HYDROGEN_MASS_CEILING) for mass in mass_array]
    return [str(name).lstrip("0123456789").upper().startswith("H") for name in names]


# ─── the mapping document ─────────────────────────────────────────────


def build_mapping(
    universe: Any,
    tables: Mapping[str, BeadTable],
    *,
    cg_topology: Optional[Mapping[str, Any]] = None,
    x_weight: str = "mass",
    f_weight: float = 1.0,
) -> Dict[str, Any]:
    """Build a cgyaml-format mapping document from an AA topology.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        AA topology. Only residue names, atom order, and masses are read.
    tables : mapping
        ``{residue name: BeadTable}``. Residues with no table are skipped, so a
        lipid-only mapping of a solvated system is the normal case.
    cg_topology : mapping, optional
        A ``cg-topology`` block to embed, e.g. from :func:`parse_gromacs_itp`.
        Extra top-level keys are ignored by OpenMSCG, so the file stays readable
        by ``cgmap``.
    x_weight : {"mass", "ones"}, default "mass"
        ``mass`` gives centre-of-mass sites (the usual choice, and what ``cgyaml``
        writes); ``ones`` gives geometric centres.
    f_weight : float, default 1.0
        Constant force weight per contributing atom, so ``F_I = Σ f_i``.

    Returns
    -------
    dict
        ``{"site-types": ..., "system": [...]}`` plus ``cg-topology`` when given.
    """
    if x_weight not in ("mass", "ones"):
        raise ValueError(f"x_weight must be 'mass' or 'ones', got {x_weight!r}.")

    residues = list(universe.residues)
    unmatched = set(tables) - {str(residue.resname) for residue in residues}
    if unmatched:
        raise ValueError(
            f"no residue named {sorted(unmatched)} in the topology; known names: "
            f"{sorted({str(r.resname) for r in residues})[:12]}"
        )

    site_types: Dict[str, Dict[str, Any]] = {}
    groups: List[Dict[str, Any]] = []
    previous: Optional[Tuple[str, int, int]] = None  # resname, first index, offset

    for residue in residues:
        resname = str(residue.resname)
        table = tables.get(resname)
        if table is None:
            continue
        indices = np.asarray(residue.atoms.indices, dtype=np.int64)
        first = int(indices[0])
        offset = int(indices.size)
        _check_residue_shape(residue, table, first, offset)
        _register_site_types(site_types, residue, table, x_weight, f_weight)

        # A new group starts unless this residue continues the previous run
        # exactly: same residue kind, and its atoms begin one stride on.
        continues = (
            previous is not None
            and previous[0] == resname
            and previous[2] == offset
            and first == previous[1] + offset
        )
        if continues:
            groups[-1]["repeat"] += 1
        else:
            groups.append(
                {
                    "anchor": first,
                    "repeat": 1,
                    "offset": offset,
                    "sites": [
                        [bead, int(local[0])] for bead, local in table
                    ],
                }
            )
        previous = (resname, first, offset)

    if not groups:
        raise ValueError("no residue matched a bead table; nothing to map.")

    mapping: Dict[str, Any] = {"site-types": site_types, "system": groups}
    if cg_topology:
        mapping["cg-topology"] = dict(cg_topology)
    return mapping


def _check_residue_shape(residue: Any, table: BeadTable, first: int, offset: int) -> None:
    """Reject a residue whose atoms are not a contiguous block of the right size."""
    indices = np.asarray(residue.atoms.indices, dtype=np.int64)
    if not np.array_equal(indices, np.arange(first, first + offset)):
        raise ValueError(
            f"residue {residue.resname}{getattr(residue, 'resid', '?')} does not own "
            "a contiguous block of atom indices, which the anchor/offset/repeat "
            "schema cannot express."
        )
    highest = max(local[-1] for _, local in table)
    if highest >= offset:
        raise ValueError(
            f"bead table for {residue.resname} references local atom {highest} but "
            f"the residue has only {offset} atoms."
        )


def _register_site_types(
    site_types: Dict[str, Dict[str, Any]],
    residue: Any,
    table: BeadTable,
    x_weight: str,
    f_weight: float,
) -> None:
    """Add each bead's index/weight template, checking repeats agree.

    Site-type templates are *relative to the bead's own first atom*, which is what
    lets one ``PT1`` entry serve both tails of a lipid.
    """
    masses = np.asarray(residue.atoms.masses, dtype=np.float64)
    for bead, local in table:
        _register_site_type(
            site_types,
            bead,
            local,
            masses,
            x_weight=x_weight,
            f_weight=f_weight,
            where=f"{residue.resname}{getattr(residue, 'resid', '?')}",
        )


def _register_site_type(
    site_types: Dict[str, Dict[str, Any]],
    name: str,
    indices: Sequence[int],
    masses: np.ndarray,
    *,
    x_weight: str,
    f_weight: float,
    where: str,
) -> None:
    """Register one site type, or check it against an existing definition.

    ``indices`` index into ``masses``, and the emitted template is *relative to
    the site's own first atom* — the convention that lets one type serve every
    copy of the same chemical group wherever it sits in the system. Two sites that
    share a name but not a shape are an error, not a silent overwrite.
    """
    ordered = [int(index) for index in indices]
    if not ordered:
        raise ValueError(f"site {name!r} in {where} has no atoms.")
    first = ordered[0]
    relative = [index - first for index in ordered]
    if x_weight == "mass":
        weights = [float(masses[index]) for index in ordered]
        if not all(weight > 0.0 for weight in weights):
            raise ValueError(
                f"site {name!r} in {where} has a non-positive AA mass; the topology "
                "has no usable masses, so pass x_weight='ones' or supply masses."
            )
    else:
        weights = [1.0] * len(ordered)
    entry = {
        "index": relative,
        "x-weight": weights,
        "f-weight": [float(f_weight)] * len(ordered),
    }
    existing = site_types.get(name)
    if existing is None:
        site_types[name] = entry
        return
    if existing["index"] != entry["index"] or not np.allclose(
        existing["x-weight"], entry["x-weight"]
    ):
        raise ValueError(
            f"site type {name!r} is defined inconsistently (seen again in {where}): "
            f"{existing['index']} with weights {existing['x-weight']} versus "
            f"{entry['index']} with {entry['x-weight']}. Give the two shapes "
            "different site-type names."
        )


def build_mapping_from_sites(
    universe: Any,
    groups: Sequence[Sequence[SiteSpec]],
    *,
    cg_topology: Optional[Mapping[str, Any]] = None,
    x_weight: str = "mass",
    f_weight: float = 1.0,
) -> Dict[str, Any]:
    """Build a mapping document from explicit per-site atom lists.

    :func:`build_mapping` covers the replicated case — many copies of one residue,
    each the same size — which the ``anchor``/``repeat``/``offset`` schema
    expresses in one group. A protein chain is the opposite: every residue has a
    different atom count, and terminal residues differ again, so there is no
    stride to repeat. This builder therefore takes the sites already resolved and
    writes each molecule as one ``repeat: 1`` group, which the schema does express.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        AA topology; only masses are read.
    groups : sequence of sequence of (str, sequence of int)
        One entry per output ``system`` group — normally one per molecule. Each
        entry lists that molecule's sites in the order they should appear, as
        ``(site-type name, absolute 0-based AA atom indices)``. Index lists need
        not be contiguous.
    cg_topology : mapping, optional
        A ``cg-topology`` block to embed, in either supported form.
    x_weight : {"mass", "ones"}, default "mass"
        ``mass`` gives centre-of-mass sites, ``ones`` geometric centres.
    f_weight : float, default 1.0
        Constant force weight per contributing atom.

    Returns
    -------
    dict
        ``{"site-types": ..., "system": [...]}`` plus ``cg-topology`` when given.
        Every group declares ``anchor``/``repeat``/``offset``, so the file stays
        readable by OpenMSCG's ``cgmap``.
    """
    if x_weight not in ("mass", "ones"):
        raise ValueError(f"x_weight must be 'mass' or 'ones', got {x_weight!r}.")
    if not groups:
        raise ValueError("build_mapping_from_sites needs at least one group.")

    n_atoms = int(universe.atoms.n_atoms)
    masses = np.asarray(universe.atoms.masses, dtype=np.float64)
    site_types: Dict[str, Dict[str, Any]] = {}
    system: List[Dict[str, Any]] = []

    for group_index, sites in enumerate(groups):
        where = f"groups[{group_index}]"
        if not sites:
            raise ValueError(f"{where} declares no sites.")
        resolved: List[Tuple[str, List[int]]] = []
        for site_index, (name, indices) in enumerate(sites):
            ordered = sorted(int(index) for index in indices)
            if not ordered:
                raise ValueError(f"{where}.sites[{site_index}] has no atoms.")
            if ordered[0] < 0 or ordered[-1] >= n_atoms:
                raise ValueError(
                    f"{where}.sites[{site_index}] ({name}) references AA atom "
                    f"{ordered[0] if ordered[0] < 0 else ordered[-1]}, outside the "
                    f"topology's {n_atoms} atoms."
                )
            resolved.append((str(name), ordered))

        anchor = min(ordered[0] for _, ordered in resolved)
        for name, ordered in resolved:
            _register_site_type(
                site_types,
                name,
                ordered,
                masses,
                x_weight=x_weight,
                f_weight=f_weight,
                where=where,
            )
        system.append(
            {
                "anchor": int(anchor),
                "repeat": 1,
                "offset": 0,
                "sites": [[name, int(ordered[0] - anchor)] for name, ordered in resolved],
            }
        )

    mapping: Dict[str, Any] = {"site-types": site_types, "system": system}
    if cg_topology:
        mapping["cg-topology"] = dict(cg_topology)
    return mapping


def derive_angles_from_bonds(
    bonds: Sequence[Sequence[int]],
) -> List[List[int]]:
    """Every angle implied by a bond list: all ``i-j-k`` with ``j`` bonded to both.

    Handy when a CG model's angles are simply "all of them", which is the usual
    choice for a protein backbone: writing them out beats hand-listing 250 terms,
    and the result is explicit in the YAML rather than inferred at load time.

    Returns
    -------
    list of [i, j, k]
        Sorted by central site then by endpoints, with ``i < k`` so each angle
        appears once.
    """
    neighbours: Dict[int, set] = {}
    for bond in bonds:
        if len(bond) != 2:
            raise ValueError(f"bond {list(bond)} does not have two sites.")
        left, right = int(bond[0]), int(bond[1])
        if left == right:
            raise ValueError(f"bond {list(bond)} joins a site to itself.")
        neighbours.setdefault(left, set()).add(right)
        neighbours.setdefault(right, set()).add(left)
    angles: List[List[int]] = []
    for centre in sorted(neighbours):
        partners = sorted(neighbours[centre])
        for position, left in enumerate(partners):
            for right in partners[position + 1 :]:
                angles.append([left, centre, right])
    return angles


def write_mapping_yaml(mapping: Mapping[str, Any], path: Union[str, Path]) -> Path:
    """Write a mapping document, flow-style lists like ``cgyaml`` does."""
    path = Path(path)
    path.write_text(
        yaml.dump(dict(mapping), sort_keys=False, default_flow_style=None),
        encoding="utf-8",
    )
    return path


# ─── GROMACS itp → cg-topology ────────────────────────────────────────


_ITP_SECTION = re.compile(r"^\[\s*(?P<name>[a-zA-Z_]+)\s*\]")


def parse_gromacs_itp(
    path: Union[str, Path], *, molecule: Optional[str] = None
) -> Dict[str, Any]:
    """Read a GROMACS ``.itp`` into a ``cg-topology`` molecule block.

    The itp describes the CG model itself — bead names, charges, masses, and
    bonded terms — and says nothing about which AA atoms form each bead. So this
    supplies the ``cg-topology`` half of a mapping file while the bead tables
    supply the AA half.

    Parameters
    ----------
    path : str or Path
        The ``.itp``. ``#include`` / ``#ifdef`` lines and ``;`` comments are
        ignored; only ``[ atoms ]``, ``[ bonds ]``, ``[ angles ]``, and
        ``[ dihedrals ]`` are read.
    molecule : str, optional
        Which ``[ moleculetype ]`` to read when the file holds several. Defaults
        to the first.

    Returns
    -------
    dict
        ``{"molecule": {"names": [...], "charges": [...], "masses": [...],
        "bonds": [[i, j], ...], "angles": [...], "dihedrals": [...]}}`` with
        0-based site indices local to the repeat unit, ready to pass as
        ``cg_topology`` to :func:`build_mapping` or
        :meth:`~AceCG.topology.cgmap.CGMapSpec.with_cg_topology`.
    """
    path = Path(path)
    blocks = _split_itp_molecules(path.read_text(encoding="utf-8"))
    if not blocks:
        raise ValueError(f"{path} contains no [ moleculetype ] block.")
    if molecule is None:
        name, sections = blocks[0]
    else:
        matches = [item for item in blocks if item[0] == molecule]
        if not matches:
            raise ValueError(
                f"{path} has no moleculetype {molecule!r}; found "
                f"{[item[0] for item in blocks]}."
            )
        name, sections = matches[0]

    atoms = sections.get("atoms", [])
    if not atoms:
        raise ValueError(f"moleculetype {name!r} in {path} has no [ atoms ] section.")
    names: List[str] = []
    charges: List[float] = []
    masses: List[float] = []
    for fields in atoms:
        # nr type resnr residue atom cgnr [charge [mass]]
        if len(fields) < 5:
            raise ValueError(f"malformed [ atoms ] row in {path}: {' '.join(fields)}")
        names.append(fields[4])
        charges.append(float(fields[6]) if len(fields) > 6 else 0.0)
        masses.append(float(fields[7]) if len(fields) > 7 else 0.0)

    n_sites = len(names)
    block: Dict[str, Any] = {"names": names, "charges": charges}
    if all(mass > 0.0 for mass in masses):
        # The model's own bead masses. Mapped sites usually want their AA mass
        # sum instead, so this stays available but is not forced on the caller.
        block["itp_masses"] = masses
    for key, width in (("bonds", 2), ("angles", 3), ("dihedrals", 4)):
        rows = sections.get(key, [])
        terms: List[List[int]] = []
        for fields in rows:
            if len(fields) < width:
                raise ValueError(f"malformed [ {key} ] row in {path}: {' '.join(fields)}")
            term = [int(field) - 1 for field in fields[:width]]
            bad = [index for index in term if index < 0 or index >= n_sites]
            if bad:
                raise ValueError(
                    f"[ {key} ] row {' '.join(fields[:width])} in {path} references a "
                    f"site outside 1..{n_sites}."
                )
            terms.append(term)
        if terms:
            block[key] = terms
    return {"molecule": block}


def _split_itp_molecules(text: str) -> List[Tuple[str, Dict[str, List[List[str]]]]]:
    """Split itp text into ``(moleculetype name, {section: rows})`` blocks."""
    blocks: List[Tuple[str, Dict[str, List[List[str]]]]] = []
    section: Optional[str] = None
    current: Optional[Dict[str, List[List[str]]]] = None
    pending_name = False

    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        match = _ITP_SECTION.match(line)
        if match:
            section = match.group("name").lower()
            if section == "moleculetype":
                current = {}
                pending_name = True
            continue
        if section is None or current is None:
            continue
        fields = line.split()
        if pending_name:
            blocks.append((fields[0], current))
            pending_name = False
            continue
        current.setdefault(section, []).append(fields)
    return blocks
