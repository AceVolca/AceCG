# AceCG/topology/cgmap.py
"""Compiled AA→CG mapping plan for the OpenMSCG ``cgyaml`` schema.

The mapping YAML that OpenMSCG's ``cgyaml`` writes and its ``cgmap`` reads is a
*template + replication* description::

    site-types:
      <TYPE>:
        index:    [int, ...]     # atom offsets relative to the site anchor
        x-weight: [float, ...]   # normalized by its sum  → COM if masses, COG if ones
        f-weight: [float, ...]   # NOT normalized         → F_I = Σ w_f,i f_i
        q:        float          # optional site charge
    system:
      - anchor: int              # absolute 0-based AA atom index of the group start
        repeat: int              # number of consecutive repeat units
        offset: int              # atoms per repeat unit (stride between repeats)
        sites:  [[<TYPE>, local_offset], ...]
        groups: [ <nested group>, ... ]     # optional, may be combined with `sites`

The absolute AA atom index of contributing atom ``k`` of the site declared at
``sites[j]`` in repeat ``r`` of group ``g`` is::

    i = g.anchor + r * g.offset + sites[j][1] + site-types[TYPE].index[k]

:class:`CGMapSpec` compiles that description **once** into flat NumPy arrays so
that mapping a frame is pure array arithmetic. OpenMSCG instead re-derives a
Python list of ``[type, anchor]`` pairs and then loops over it per frame, which
is what makes ``Mapper.process`` interpreter-bound.

Deviations from OpenMSCG's ``mapper.py`` — all deliberate, all bug fixes
-----------------------------------------------------------------------
* ``anchor`` is added **exactly once**. OpenMSCG's ``unpack_group`` passes
  ``root_anchor + group['anchor']`` down into the recursive ``groups`` branch and
  then adds it a *second* time in its own repeat loop, so every site inside a
  nested group whose parent has a non-zero anchor is offset by twice that anchor.
* Nested children may omit ``anchor`` / ``repeat`` / ``offset``; they default to
  ``0 / 1 / 0``. OpenMSCG raises ``KeyError: 'repeat'`` on such a file, which
  real ``map.yaml`` files in our archives do use.
* ``site-types`` keys may be ints (YAML parses a bare ``1:`` as an int), so
  lookups accept both the raw key and its string form.
* Integer ``x-weight`` lists are accepted. OpenMSCG does ``v /= v.sum()``
  in-place on the parsed list and raises ``TypeError`` for integer input.
* ``f-weight`` is optional and defaults to all ones.
* Duplicate indices inside one site **sum**, matching ``Mapper.process``'s
  matmul. OpenMSCG's ``Mapper.get_matrix`` *overwrites* instead, so its two code
  paths disagree; we adopt the summing one.
* The caller's mapping dict is never mutated (``from_topology`` mutates it).

Site ordering is preserved bit-for-bit: group → repeat → site-within-unit.

Molecule identity
-----------------
A *molecule* is one repeat unit of one top-level ``system`` group. This matches
the CG ``resid`` assignment that :func:`AceCG.io.coordinates.build_CG_coords`
already uses, and it is the unit that ``unwrap="molecule"`` keeps whole. Note
this is an **all-atom-side** notion derived from ``anchor``/``repeat``/``offset``;
it does not depend on CG bonded topology being known.

Optional CG bonded topology
---------------------------
The ``cgyaml`` schema carries no bond/angle/dihedral information, and no residue
or molecule naming either. Two optional channels supply both (see
:meth:`CGMapSpec.with_cg_topology`):

1. a ``cg-topology:`` block at the top level of the same YAML file — extra
   top-level keys are ignored by OpenMSCG, so such a file stays readable by it;
2. an OpenMSCG ``top.in`` / ``cgtop`` file, handled by the existing
   :mod:`AceCG.topology.mscg` parser (wired up by the workflow layer).

Both are optional: a spec with no bonded topology still maps trajectories, it
just emits an atoms-only CG topology file with a caller-supplied residue name.

The block itself comes in two forms.

**Residue form** (preferred). Bonded topology is declared once per *residue*, and
a group states which residues its repeat unit is made of. This is what lets one
file describe a lipid, a one-site-per-residue protein and a four-site-per-residue
protein without repeating anything::

    cg-topology:
      residues:                      # single-residue templates, keyed by resname
        - resname: DPPC
          names:   [NC3, PO4, ...]   # per-site labels inside the residue
          linkable: false            # a lone, unlinkable molecule
          bonds:   [[0, 1], ...]     # residue-local site indices
          angles:  [[0, 1, 2], ...]
        - resname: ALA
          names:   [N, CA, C, SA1]
          linkable: true
          left_linker_atom_type:  N  # bonds to the previous residue
          right_linker_atom_type: C  # bonds to the next residue
          bonds:   [[0, 1], [1, 2], [1, 3]]
          angles:  [[0, 1, 2], [0, 1, 3], [2, 1, 3]]
      groups:                        # one entry per `system` group, in order
        - molname:  PROT_A
          resnames: [ALA, GLY, ...]  # per residue of the repeat unit
          resids:   [0, 0, 0, 0, 1, ...]   # per *site* of the repeat unit
          linker_angles: true         # all angles involving a sequential linker bond
          additional_bonds:  [[[12, SG], [47, SG]]]   # e.g. a disulfide
          additional_angles: [...]   # only nonstandard, explicitly added angles

Only while constructing a molecule from one ``groups`` entry, consecutive
residues are linked 1-D sequentially: when both are ``linkable``, a bond is
added between the left residue's
``right_linker_atom_type`` site and the right residue's
``left_linker_atom_type`` site. Set that group's boolean ``linker_angles`` to
``true`` to add every angle in the residue bond graph that contains at least one
such sequential linker bond; ``false`` (the default) adds none. Anything else
that is not intra-residue or a sequential-link consequence goes in
``additional_bonds`` / ``additional_angles`` / ``additional_dihedrals``, whose
items are either unit-local site indices or ``[resid, site]`` pairs naming a
site inside a residue of the unit.

Linking is confined to one group repeat unit: it never joins separate
``system.repeat`` molecules, and a standalone residue remains a standalone
molecule even when its reusable template has ``linkable: true``.

A many-residues-per-site model has no per-residue structure left to describe, so
it is expressed the same way a lipid is: one unlinkable super-residue whose
``names`` are the custom site names.

**Molecule form** (legacy, still supported). One flat template per repeat unit,
indices local to the unit, and residue naming left to the caller::

    cg-topology:
      molecule: {names: [...], charges: [...], bonds: [[0, 1], ...]}
      # or: groups: [{...}, ...]   one template per `system` group
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import yaml

from .types import InteractionKey


__all__ = [
    "CGMapSpec",
    "SiteArrays",
    "expand_mapping_sites",
    "load_cgmap_spec",
    "load_mapping_yaml",
]


# Index dtypes. The largest absolute AA index we ever see is bounded by the
# atom count of the reference topology, so int32 is plenty for the compact-space
# arrays (and halves the bytes broadcast over MPI); absolute AA indices stay
# int64 so a >2^31-atom system still works.
_COMPACT_DTYPE = np.int32
_ABS_DTYPE = np.int64


# ─── low-level helpers ────────────────────────────────────────────────


def load_mapping_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a mapping YAML file.

    Parameters
    ----------
    path : str or Path
        YAML file written by OpenMSCG ``cgyaml`` (or hand-authored in the same
        schema).

    Returns
    -------
    dict
        Parsed mapping. Nothing is normalized here; :meth:`CGMapSpec.from_mapping`
        owns all interpretation.
    """
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _as_int(value: Any, what: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what}: expected an int-like value, got {value!r}") from exc


def _site_type_entry(site_types: Mapping[Any, Any], key: Any) -> Mapping[str, Any]:
    """Look up a ``site-types`` entry tolerating int/str key mismatch.

    YAML parses a bare ``1:`` key as the integer ``1`` while ``sites`` entries
    may spell the same type as ``"1"`` (or vice versa), so both spellings are
    tried before failing.
    """
    if key in site_types:
        return site_types[key]
    text = str(key)
    if text in site_types:
        return site_types[text]
    try:
        numeric = int(text)
    except ValueError:
        numeric = None
    if numeric is not None and numeric in site_types:
        return site_types[numeric]
    raise ValueError(
        f"site-types has no entry for site type {key!r}. "
        f"Known types: {sorted(str(k) for k in site_types)}"
    )


def expand_mapping_sites(
    mapping: Mapping[str, Any],
) -> Tuple[List[Tuple[Any, int]], List[int], List[int]]:
    """Expand ``mapping["system"]`` into a flat per-site list.

    This is the reference (pure-Python) expansion. :meth:`CGMapSpec.from_mapping`
    uses a vectorized equivalent; both are pinned against each other in the
    tests, and this one is the readable statement of the semantics.

    Parameters
    ----------
    mapping : Mapping
        Parsed mapping YAML.

    Returns
    -------
    sites : list of (type_key, site_anchor)
        One entry per CG site in OpenMSCG output order (group → repeat → site).
        ``site_anchor`` is the absolute AA index that ``site-types[...].index``
        offsets are added to, i.e. ``anchor + r*offset + local_offset``.
    group_ids : list of int
        Top-level ``system`` group index owning each site.
    mol_ids : list of int
        Global repeat-unit (molecule) id owning each site.
    """
    system = mapping["system"]
    sites: List[Tuple[Any, int]] = []
    group_ids: List[int] = []
    mol_ids: List[int] = []
    mol_base = 0
    for group_index, group in enumerate(system):
        unit = _expand_unit(group, f"system[{group_index}]")
        repeat = _as_int(group.get("repeat", 1), f"system[{group_index}].repeat")
        offset = _as_int(group.get("offset", 0), f"system[{group_index}].offset")
        anchor = _as_int(group.get("anchor", 0), f"system[{group_index}].anchor")
        for rep in range(repeat):
            shift = anchor + rep * offset
            for type_key, local_offset in unit:
                sites.append((type_key, local_offset + shift))
                group_ids.append(group_index)
                mol_ids.append(mol_base + rep)
        mol_base += repeat
    return sites, group_ids, mol_ids


def _expand_unit(group: Mapping[str, Any], where: str) -> List[Tuple[Any, int]]:
    """Return one repeat unit of ``group`` as ``[(type_key, local_offset)]``.

    The offsets returned are relative to the *unit origin*, i.e. they exclude
    ``group["anchor"]`` — the caller adds it exactly once, per repeat. Nested
    ``groups`` are flattened first (matching OpenMSCG's ordering), then the
    group's own ``sites``.
    """
    unit: List[Tuple[Any, int]] = []
    children = group.get("groups")
    if children is not None:
        if not isinstance(children, (list, tuple)):
            raise ValueError(f"{where}.groups must be a list, got {type(children).__name__}")
        for child_index, child in enumerate(children):
            child_where = f"{where}.groups[{child_index}]"
            child_unit = _expand_unit(child, child_where)
            child_repeat = _as_int(child.get("repeat", 1), f"{child_where}.repeat")
            child_offset = _as_int(child.get("offset", 0), f"{child_where}.offset")
            child_anchor = _as_int(child.get("anchor", 0), f"{child_where}.anchor")
            for rep in range(child_repeat):
                shift = child_anchor + rep * child_offset
                unit.extend((type_key, local + shift) for type_key, local in child_unit)
    own = group.get("sites")
    if own is not None:
        if not isinstance(own, (list, tuple)):
            raise ValueError(f"{where}.sites must be a list, got {type(own).__name__}")
        for site_index, site in enumerate(own):
            if not isinstance(site, (list, tuple)) or len(site) != 2:
                raise ValueError(
                    f"{where}.sites[{site_index}] must be a [type, offset] pair, got {site!r}"
                )
            unit.append((site[0], _as_int(site[1], f"{where}.sites[{site_index}][1]")))
    if not unit:
        raise ValueError(f"{where} declares neither 'sites' nor a non-empty 'groups'.")
    return unit


# ─── the compiled spec ────────────────────────────────────────────────


@dataclass(frozen=True, eq=False)
class CGMapSpec:
    """A mapping YAML compiled into flat arrays.

    Built once (usually on MPI rank 0), broadcast to every rank, then consumed
    per frame by :class:`AceCG.compute.cgmap.CGMapper`. ``eq=False`` because the
    fields are arrays and dataclass equality on them raises.

    Attributes
    ----------
    site_type_ids : np.ndarray, shape (n_sites,), int32
        1-based canonical CG type id. By default ids follow ``site-types``
        insertion order, matching OpenMSCG. An optional per-site
        ``cg-topology.types`` list may instead merge distinct mapping templates
        onto shared output types while ``cg-topology.names`` retains positional
        bead labels.
    site_mol_ids : np.ndarray, shape (n_sites,), int32
        0-based molecule (repeat-unit) id.
    site_group_ids : np.ndarray, shape (n_sites,), int32
        0-based index into ``mapping["system"]``.
    site_ref_pos : np.ndarray, shape (n_sites,), int32
        Compact-buffer position of each site's ``index[0]`` atom. This is the
        PBC reference atom that OpenMSCG uses, needed by the ``bead`` and
        ``deprecated`` unwrap modes.
    atom_indices : np.ndarray, shape (n_required,), int64
        Sorted unique absolute AA atom indices that the mapping touches. The
        "compact buffer" is an array indexed in this order, so only mapped atoms
        are gathered and imaged.
    csr_indptr, csr_cols, csr_wx, csr_wf
        CSR of the ``(n_sites, n_required)`` mapping operator over the compact
        buffer. ``csr_wx`` is row-normalized (each site's weights sum to 1) so
        positions are a weighted average; ``csr_wf`` is raw so forces are a
        weighted sum. Stored as components rather than a ``scipy`` matrix to keep
        this module free of ``scipy`` and to keep the broadcast payload small.
    mol_indptr : np.ndarray, shape (n_mol + 1,), int64
        Offsets into the molecule-grouped view of the compact buffer.
    mol_atom_pos : np.ndarray or None
        Compact positions grouped by molecule, or ``None`` when molecules already
        own contiguous ascending slices of the compact buffer (the common case
        for ``anchor``/``offset`` replicated systems) — then ``mol_indptr`` slices
        the buffer directly and no gather is needed.
    mol_ref_pos : np.ndarray, shape (n_mol,), int32
        Compact position of each molecule's PBC reference atom, used by the
        ``molecule`` unwrap mode.
    type_names : tuple of str
        Site-type names in id order (``type_names[type_id - 1]``).
    type_xweight_sum : np.ndarray, shape (n_types,), float64
        Sum of each type's **raw** ``x-weight``. Equals the summed AA mass of the
        site when the weights are masses, which is the usual convention; the
        caller decides whether to use it as the CG bead mass.
    type_charges : np.ndarray, shape (n_types,), float64
        Per-type charge from the optional ``q`` key (0.0 when absent). A
        ``cg-topology`` block, if present, overrides this per site position.
    group_repeats, group_unit_sites, group_site_offsets, group_mol_offsets
        Per-group replication bookkeeping, used to broadcast a ``cg-topology``
        block's repeat-unit-local indices across every repeat of every group.
    site_labels : tuple of str or None
        Per-site bead label from ``cg-topology`` (cosmetic: gro/pdb atom names).
        ``None`` when no block was supplied.
    site_charges : np.ndarray or None
        Per-site charge from ``cg-topology``, overriding ``type_charges``.
    site_masses : np.ndarray or None
        Per-site mass from ``cg-topology``, overriding ``type_xweight_sum``.
    site_res_ids : np.ndarray or None
        Global 0-based CG *residue* index per site, from a residue-form
        ``cg-topology`` block. ``None`` for the legacy molecule form, where a
        residue is a whole repeat unit and ``site_mol_ids`` already says it.
    res_name_ids : np.ndarray or None
        Per global CG residue, an index into ``res_names``.
    res_names : tuple of str or None
        Residue-name table; ``res_names[res_name_ids[r]]`` names residue ``r``.
    group_molnames : tuple of str or None
        Per ``system`` group molecule name, from the residue form's ``molname``.
    bonds, angles, dihedrals : np.ndarray or None
        Global 0-based site-index tuples, shapes ``(n,2)``/``(n,3)``/``(n,4)``.
    bond_type_ids, angle_type_ids, dihedral_type_ids : np.ndarray or None
        1-based interaction type ids.
    bond_type_keys, angle_type_keys, dihedral_type_keys : tuple or None
        :class:`~AceCG.topology.types.InteractionKey` per type id, in id order.
    """

    # site level
    site_type_ids: np.ndarray
    site_mol_ids: np.ndarray
    site_group_ids: np.ndarray
    site_ref_pos: np.ndarray
    # CSR over the compact atom buffer
    atom_indices: np.ndarray
    csr_indptr: np.ndarray
    csr_cols: np.ndarray
    csr_wx: np.ndarray
    csr_wf: np.ndarray
    # molecule level
    mol_indptr: np.ndarray
    mol_atom_pos: Optional[np.ndarray]
    mol_ref_pos: np.ndarray
    # type level
    type_names: Tuple[str, ...]
    type_xweight_sum: np.ndarray
    type_charges: np.ndarray
    # group replication bookkeeping
    group_repeats: np.ndarray
    group_unit_sites: np.ndarray
    group_site_offsets: np.ndarray
    group_mol_offsets: np.ndarray
    # optional CG bonded topology
    site_labels: Optional[Tuple[str, ...]] = None
    site_charges: Optional[np.ndarray] = None
    site_masses: Optional[np.ndarray] = None
    site_res_ids: Optional[np.ndarray] = None
    res_name_ids: Optional[np.ndarray] = None
    res_names: Optional[Tuple[str, ...]] = None
    group_molnames: Optional[Tuple[str, ...]] = None
    bonds: Optional[np.ndarray] = None
    bond_type_ids: Optional[np.ndarray] = None
    bond_type_keys: Optional[Tuple[InteractionKey, ...]] = None
    angles: Optional[np.ndarray] = None
    angle_type_ids: Optional[np.ndarray] = None
    angle_type_keys: Optional[Tuple[InteractionKey, ...]] = None
    dihedrals: Optional[np.ndarray] = None
    dihedral_type_ids: Optional[np.ndarray] = None
    dihedral_type_keys: Optional[Tuple[InteractionKey, ...]] = None
    # One private compressed fitted-force payload: matrices, atom placements,
    # site placements, coordinate maps, authored maps, pairs, metadata.
    _force_operator: Optional[Tuple[Any, ...]] = None

    # ── derived sizes ─────────────────────────────────────────────

    @property
    def n_sites(self) -> int:
        """Number of CG sites (beads) per frame."""
        return int(self.site_type_ids.shape[0])

    @property
    def n_required_atoms(self) -> int:
        """Number of distinct AA atoms the mapping reads."""
        return int(self.atom_indices.shape[0])

    @property
    def n_mol(self) -> int:
        """Number of molecules (repeat units)."""
        return int(self.mol_indptr.shape[0] - 1)

    @property
    def n_types(self) -> int:
        """Number of distinct site types."""
        return len(self.type_names)

    @property
    def nnz(self) -> int:
        """Number of (site, atom) weight entries."""
        return int(self.csr_cols.shape[0])

    @property
    def molecules_contiguous(self) -> bool:
        """``True`` when ``mol_indptr`` slices the compact buffer directly."""
        return self.mol_atom_pos is None

    @property
    def has_bonded_topology(self) -> bool:
        """``True`` when a CG bonded topology was supplied."""
        return self.bonds is not None or self.angles is not None or self.dihedrals is not None

    @property
    def has_force_operator(self) -> bool:
        """Whether a non-authored force operator is attached."""
        return self._force_operator is not None

    def with_force_operator(
        self,
        matrices: Sequence[np.ndarray],
        atom_positions: Sequence[np.ndarray],
        site_positions: Sequence[np.ndarray],
        coordinate_maps: Sequence[np.ndarray],
        authored_maps: Sequence[np.ndarray],
        constraint_pairs: Sequence[np.ndarray],
        metadata: Mapping[str, Any],
    ) -> "CGMapSpec":
        """Attach checked, template-compressed fitted-force blocks."""
        blocks = tuple(np.asarray(value, dtype=np.float64) for value in matrices)
        atoms = tuple(np.asarray(value, dtype=np.int64) for value in atom_positions)
        sites = tuple(np.asarray(value, dtype=np.int64) for value in site_positions)
        coordinates = tuple(np.asarray(value, dtype=np.float64) for value in coordinate_maps)
        authored = tuple(np.asarray(value, dtype=np.float64) for value in authored_maps)
        pairs = tuple(np.asarray(value, dtype=np.int64).reshape(-1, 2) for value in constraint_pairs)
        if not blocks or not (len(blocks) == len(atoms) == len(sites) == len(coordinates) == len(authored) == len(pairs)):
            raise ValueError("force operator blocks must be non-empty and have matching lengths.")
        covered = []
        for matrix, atom_block, site_block, coordinate, authored_block, pair_block in zip(
            blocks, atoms, sites, coordinates, authored, pairs
        ):
            if matrix.ndim != 2 or coordinate.ndim != 2 or matrix.shape != coordinate.shape or authored_block.shape != coordinate.shape:
                raise ValueError("force operator and coordinate/authored maps must have one matching two-dimensional shape.")
            if atom_block.ndim != 2 or atom_block.shape[1] != matrix.shape[1]:
                raise ValueError("force operator atom placements have the wrong local width.")
            if site_block.shape != (atom_block.shape[0], matrix.shape[0]):
                raise ValueError("force operator site placements have the wrong shape.")
            if atom_block.size and (int(atom_block.min()) < 0 or int(atom_block.max()) >= self.n_required_atoms):
                raise ValueError("force operator atom placements exceed the compact atom buffer.")
            if site_block.size and (int(site_block.min()) < 0 or int(site_block.max()) >= self.n_sites):
                raise ValueError("force operator site placements exceed CG sites.")
            if pair_block.size and (int(pair_block.min()) < 0 or int(pair_block.max()) >= matrix.shape[1]):
                raise ValueError("force operator constraint pairs exceed local atoms.")
            covered.append(site_block.ravel())
        all_sites = np.concatenate(covered)
        if all_sites.size != self.n_sites or np.unique(all_sites).size != self.n_sites:
            raise ValueError("force operator site placements must cover every CG site exactly once.")
        for values in (*blocks, *atoms, *sites, *coordinates, *authored, *pairs):
            values.setflags(write=False)
        return replace(
            self,
            _force_operator=(blocks, atoms, sites, coordinates, authored, pairs, dict(metadata)),
        )

    def nbytes(self) -> int:
        """Total bytes of the array payload (what an MPI broadcast moves)."""
        total = 0
        for value in self.__dict__.values():
            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
        return total

    # ── construction ──────────────────────────────────────────────

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        index_base: int = 0,
        masses: Optional[np.ndarray] = None,
        n_atoms: Optional[int] = None,
        mol_reference: Union[str, int] = "first",
        cg_topology: Optional[Mapping[str, Any]] = None,
        strict_weights: bool = True,
    ) -> "CGMapSpec":
        """Compile a parsed mapping dict.

        Parameters
        ----------
        mapping : Mapping
            Parsed mapping YAML (see the module docstring). Not mutated.
        index_base : {0, 1}, default 0
            Index base of the YAML. OpenMSCG files are 0-based. ``1`` shifts both
            anchors and site-type indices by -1, matching the long-standing
            convention of :func:`AceCG.io.coordinates.bead_aa_indices`.
        masses : np.ndarray, optional
            Per-atom AA masses of the whole system, used only as the ``x-weight``
            fallback for site types that omit it (a mass-weighted centre of
            mass). Required if any type omits ``x-weight``.
        n_atoms : int, optional
            AA atom count to range-check the compiled indices against.
        mol_reference : {"first", "anchor"} or int, default "first"
            PBC reference atom for ``unwrap="molecule"``. ``"first"`` uses each
            molecule's lowest mapped atom index; ``"anchor"`` uses the repeat
            unit's origin atom; an int uses ``origin + mol_reference``. When the
            requested atom is not part of the mapping, ``"first"`` is used for
            that molecule and a warning is emitted.
        cg_topology : Mapping, optional
            A ``cg-topology`` block. Defaults to ``mapping.get("cg-topology")``,
            so a single-file YAML needs nothing extra here. Pass ``{}`` to
            explicitly ignore an in-file block.
        strict_weights : bool, default True
            Require ``len(x-weight) == len(index)`` for every type.

        Returns
        -------
        CGMapSpec
        """
        if index_base not in (0, 1):
            raise ValueError(f"index_base must be 0 or 1, got {index_base!r}")
        shift = 0 if index_base == 0 else -1

        site_types = mapping.get("site-types")
        if not isinstance(site_types, Mapping):
            raise ValueError("mapping must contain a dict key 'site-types'.")
        system = mapping.get("system")
        if not isinstance(system, (list, tuple)) or len(system) == 0:
            raise ValueError("mapping must contain a non-empty list key 'system'.")

        # Type table in YAML insertion order → 1-based type ids, matching
        # OpenMSCG's `list(self.types.keys()).index(name) + 1`.
        type_names = tuple(str(key) for key in site_types)
        type_id_of: Dict[str, int] = {name: i + 1 for i, name in enumerate(type_names)}

        type_index: Dict[str, np.ndarray] = {}
        type_wx_raw: Dict[str, Optional[np.ndarray]] = {}
        type_wf: Dict[str, Optional[np.ndarray]] = {}
        type_charges = np.zeros(len(type_names), dtype=np.float64)
        for position, raw_key in enumerate(site_types):
            name = str(raw_key)
            entry = site_types[raw_key]
            if not isinstance(entry, Mapping) or "index" not in entry:
                raise ValueError(f"site-types[{raw_key!r}] must be a mapping with an 'index' list.")
            idx = np.asarray(entry["index"], dtype=_ABS_DTYPE).ravel()
            if idx.size == 0:
                raise ValueError(f"site-types[{raw_key!r}]['index'] must be non-empty.")
            type_index[name] = idx
            raw_weight = entry.get("x-weight")
            if raw_weight is None:
                type_wx_raw[name] = None
            else:
                weight = np.asarray(raw_weight, dtype=np.float64).ravel()
                if strict_weights and weight.size != idx.size:
                    raise ValueError(
                        f"site-types[{raw_key!r}]: len(x-weight)={weight.size} != "
                        f"len(index)={idx.size}."
                    )
                if weight.size != idx.size:
                    raise ValueError(
                        f"site-types[{raw_key!r}]: x-weight/index length mismatch "
                        f"({weight.size} vs {idx.size}) cannot be reconciled."
                    )
                total = float(weight.sum())
                if not np.isfinite(total) or total == 0.0:
                    raise ValueError(
                        f"site-types[{raw_key!r}]: x-weight sums to {total!r}; "
                        "cannot normalize into a weighted average."
                    )
                type_wx_raw[name] = weight
            raw_force = entry.get("f-weight")
            if raw_force is None:
                type_wf[name] = None  # → all ones, unlike OpenMSCG which KeyErrors
            else:
                force = np.asarray(raw_force, dtype=np.float64).ravel()
                if force.size != idx.size:
                    raise ValueError(
                        f"site-types[{raw_key!r}]: len(f-weight)={force.size} != "
                        f"len(index)={idx.size}."
                    )
                type_wf[name] = force
            charge = entry.get("q")
            if charge is not None:
                type_charges[position] = float(charge)

        # ── expand the system tree, per group, vectorized over repeats ──
        n_groups = len(system)
        group_repeats = np.zeros(n_groups, dtype=np.int64)
        group_unit_sites = np.zeros(n_groups, dtype=np.int64)
        unit_type_names: List[List[str]] = []
        unit_local_offsets: List[np.ndarray] = []
        unit_origins: List[np.ndarray] = []
        for group_index, group in enumerate(system):
            where = f"system[{group_index}]"
            unit = _expand_unit(group, where)
            repeat = _as_int(group.get("repeat", 1), f"{where}.repeat")
            offset = _as_int(group.get("offset", 0), f"{where}.offset")
            anchor = _as_int(group.get("anchor", 0), f"{where}.anchor")
            if repeat < 1:
                raise ValueError(f"{where}.repeat must be >= 1, got {repeat}.")
            if offset < 0:
                raise ValueError(f"{where}.offset must be >= 0, got {offset}.")
            # Validate every referenced type exists, then key it by the string
            # form so int and str spellings of the same type collapse.
            for key, _ in unit:
                _site_type_entry(site_types, key)
            names = [str(key) for key, _ in unit]
            group_repeats[group_index] = repeat
            group_unit_sites[group_index] = len(unit)
            unit_type_names.append(names)
            unit_local_offsets.append(
                np.asarray([local for _, local in unit], dtype=_ABS_DTYPE)
            )
            unit_origins.append(anchor + np.arange(repeat, dtype=_ABS_DTYPE) * offset)

        group_site_counts = group_repeats * group_unit_sites
        group_site_offsets = np.concatenate(
            ([0], np.cumsum(group_site_counts))
        ).astype(np.int64)
        group_mol_offsets = np.concatenate(([0], np.cumsum(group_repeats))).astype(np.int64)
        n_sites = int(group_site_offsets[-1])
        n_mol = int(group_mol_offsets[-1])

        site_type_ids = np.empty(n_sites, dtype=_COMPACT_DTYPE)
        site_mol_ids = np.empty(n_sites, dtype=_COMPACT_DTYPE)
        site_group_ids = np.empty(n_sites, dtype=_COMPACT_DTYPE)
        site_nnz = np.empty(n_sites, dtype=np.int64)

        for group_index in range(n_groups):
            start = int(group_site_offsets[group_index])
            stop = int(group_site_offsets[group_index + 1])
            names = unit_type_names[group_index]
            repeat = int(group_repeats[group_index])
            per_unit = int(group_unit_sites[group_index])
            ids_unit = np.asarray([type_id_of[name] for name in names], dtype=_COMPACT_DTYPE)
            nnz_unit = np.asarray([type_index[name].size for name in names], dtype=np.int64)
            site_type_ids[start:stop] = np.tile(ids_unit, repeat)
            site_nnz[start:stop] = np.tile(nnz_unit, repeat)
            site_group_ids[start:stop] = group_index
            site_mol_ids[start:stop] = int(group_mol_offsets[group_index]) + np.repeat(
                np.arange(repeat, dtype=_COMPACT_DTYPE), per_unit
            )

        csr_indptr = np.concatenate(([0], np.cumsum(site_nnz))).astype(np.int64)
        total_nnz = int(csr_indptr[-1])
        cols_abs = np.empty(total_nnz, dtype=_ABS_DTYPE)
        csr_wx = np.empty(total_nnz, dtype=np.float64)
        csr_wf = np.empty(total_nnz, dtype=np.float64)
        needs_mass_fallback: List[str] = []

        for group_index in range(n_groups):
            base_site = int(group_site_offsets[group_index])
            repeat = int(group_repeats[group_index])
            per_unit = int(group_unit_sites[group_index])
            names = unit_type_names[group_index]
            locals_ = unit_local_offsets[group_index]
            origins = unit_origins[group_index]
            rows_of_repeat = base_site + np.arange(repeat, dtype=np.int64) * per_unit
            for position in range(per_unit):
                name = names[position]
                idx = type_index[name]
                rows = rows_of_repeat + position
                starts = csr_indptr[rows]
                dest = starts[:, None] + np.arange(idx.size, dtype=np.int64)[None, :]
                anchors = origins + int(locals_[position])
                cols_abs[dest] = (anchors[:, None] + shift) + (idx[None, :] + shift)
                weight = type_wx_raw[name]
                if weight is None:
                    needs_mass_fallback.append(name)
                    csr_wx[dest] = np.nan  # filled after the compact buffer exists
                else:
                    csr_wx[dest] = weight / weight.sum()
                force = type_wf[name]
                csr_wf[dest] = 1.0 if force is None else force

        # ── compact atom buffer ──
        atom_indices = np.unique(cols_abs)
        if atom_indices[0] < 0:
            raise ValueError(
                f"Mapping produced negative AA atom index {int(atom_indices[0])}. "
                f"Check anchor/offset/site offsets and index_base (currently {index_base})."
            )
        if n_atoms is not None and int(atom_indices[-1]) >= int(n_atoms):
            raise ValueError(
                f"Mapping references AA atom index {int(atom_indices[-1])} but the "
                f"topology has only {int(n_atoms)} atoms. Check anchor/offset/repeat "
                f"and index_base (currently {index_base})."
            )
        csr_cols = np.searchsorted(atom_indices, cols_abs).astype(_COMPACT_DTYPE)

        if needs_mass_fallback:
            if masses is None:
                raise ValueError(
                    "site-types "
                    f"{sorted(set(needs_mass_fallback))} omit 'x-weight' and no "
                    "`masses` array was supplied for the centre-of-mass fallback."
                )
            mass_arr = np.asarray(masses, dtype=np.float64).ravel()
            per_nnz = mass_arr[atom_indices[csr_cols]]
            missing = np.isnan(csr_wx)
            if np.any(missing):
                row_of_nnz = np.repeat(np.arange(n_sites, dtype=np.int64), site_nnz)
                sums = np.zeros(n_sites, dtype=np.float64)
                np.add.at(sums, row_of_nnz[missing], per_nnz[missing])
                bad = np.nonzero(missing)[0]
                row_sums = sums[row_of_nnz[bad]]
                if np.any(row_sums <= 0.0):
                    raise ValueError(
                        "Centre-of-mass fallback needs positive summed masses, but "
                        f"{int(np.count_nonzero(row_sums <= 0.0))} site(s) sum to <= 0."
                    )
                csr_wx[bad] = per_nnz[bad] / row_sums

        site_ref_pos = csr_cols[csr_indptr[:-1]].astype(_COMPACT_DTYPE)

        # ── molecule grouping over the compact buffer ──
        nnz_mol = np.repeat(site_mol_ids.astype(np.int64), site_nnz)
        mol_of_atom = np.full(atom_indices.size, -1, dtype=np.int64)
        # Reverse-order scatter makes the *earliest* claiming site win, so the
        # grouping is deterministic and matches site order.
        mol_of_atom[csr_cols[::-1]] = nnz_mol[::-1]
        shared = mol_of_atom[csr_cols] != nnz_mol
        if np.any(shared):
            n_shared = int(np.unique(csr_cols[shared]).size)
            warnings.warn(
                f"{n_shared} AA atom(s) are mapped into more than one molecule; "
                "unwrap='molecule' will image them with the first claiming "
                "molecule. Use unwrap='bead' if that is not what you want.",
                RuntimeWarning,
                stacklevel=2,
            )
        if np.any(mol_of_atom < 0):  # pragma: no cover - every atom comes from a site
            raise AssertionError("internal error: unclaimed atom in molecule grouping")
        mol_counts = np.bincount(mol_of_atom, minlength=n_mol).astype(np.int64)
        if np.any(mol_counts == 0):
            empty = int(np.count_nonzero(mol_counts == 0))
            raise ValueError(
                f"{empty} repeat unit(s) own no mapped atoms; the mapping is "
                "inconsistent with its own anchor/repeat/offset declaration."
            )
        mol_indptr = np.concatenate(([0], np.cumsum(mol_counts))).astype(np.int64)
        # `mol_of_atom` is non-decreasing exactly when each molecule owns a
        # contiguous ascending slice of the compact buffer, which is the case for
        # ordinary anchor/offset replication. Then no gather is needed.
        if bool(np.all(np.diff(mol_of_atom) >= 0)):
            mol_atom_pos = None
        else:
            mol_atom_pos = np.argsort(mol_of_atom, kind="stable").astype(_COMPACT_DTYPE)

        mol_ref_pos = _resolve_mol_reference(
            mol_reference=mol_reference,
            atom_indices=atom_indices,
            mol_indptr=mol_indptr,
            mol_atom_pos=mol_atom_pos,
            group_repeats=group_repeats,
            group_mol_offsets=group_mol_offsets,
            unit_origins=unit_origins,
            shift=shift,
        )

        type_xweight_sum = np.zeros(len(type_names), dtype=np.float64)
        for position, name in enumerate(type_names):
            weight = type_wx_raw[name]
            if weight is not None:
                type_xweight_sum[position] = float(weight.sum())

        spec = cls(
            site_type_ids=site_type_ids,
            site_mol_ids=site_mol_ids,
            site_group_ids=site_group_ids,
            site_ref_pos=site_ref_pos,
            atom_indices=atom_indices.astype(_ABS_DTYPE),
            csr_indptr=csr_indptr,
            csr_cols=csr_cols,
            csr_wx=csr_wx,
            csr_wf=csr_wf,
            mol_indptr=mol_indptr,
            mol_atom_pos=mol_atom_pos,
            mol_ref_pos=mol_ref_pos,
            type_names=type_names,
            type_xweight_sum=type_xweight_sum,
            type_charges=type_charges,
            group_repeats=group_repeats,
            group_unit_sites=group_unit_sites,
            group_site_offsets=group_site_offsets,
            group_mol_offsets=group_mol_offsets,
        )

        block = mapping.get("cg-topology") if cg_topology is None else cg_topology
        if block:
            spec = spec.with_cg_topology(block)
        return spec

    # ── optional CG bonded topology ───────────────────────────────

    def with_cg_topology(self, block: Mapping[str, Any]) -> "CGMapSpec":
        """Return a copy carrying the CG bonded topology from ``block``.

        Parameters
        ----------
        block : Mapping
            A ``cg-topology`` block in either form (see the module docstring).

            *Residue form* — ``{"residues": [...], "groups": [...]}``. Bonded
            terms are declared per residue and each group lists the residues its
            repeat unit is made of; sequential linker bonds are added for it.
            A per-group ``linker_angles`` boolean controls whether all angles
            involving those bonds are also added. Links never cross repeat-unit
            molecule boundaries. This form also supplies per-site residue ids
            and names.

            *Molecule form* — ``{"molecule": {...}}``, one repeat-unit template
            for every ``system`` group, or ``{"groups": [{...}, ...]}`` with one
            template per group. Templates index sites by their **position within
            one repeat unit** (0-based, in the order they appear in ``sites``), so
            a 12-bead lipid declares 11 bonds, not ``11 × n_lipids``. Recognized
            keys: ``names``, ``types``, ``charges``, ``masses`` (``"auto"`` ⇒
            summed ``x-weight``), ``bonds``, ``angles``, ``dihedrals``.

        Returns
        -------
        CGMapSpec
            A new spec; ``self`` is unchanged.
        """
        layouts: Optional[List[Optional["_ResidueLayout"]]] = None
        if _is_residue_form(block):
            templates, layouts = _residue_form_templates(
                block,
                n_groups=len(self.group_repeats),
                group_unit_sites=self.group_unit_sites,
            )
        else:
            templates = _cg_topology_templates(block, n_groups=len(self.group_repeats))

        labels: List[Optional[str]] = [None] * self.n_sites
        canonical_types = [
            self.type_names[int(type_id) - 1] for type_id in self.site_type_ids
        ]
        charges = np.array(self.type_charges[self.site_type_ids - 1], dtype=np.float64)
        masses = np.array(self.type_xweight_sum[self.site_type_ids - 1], dtype=np.float64)
        have_labels = False
        have_types = False
        have_charges = False
        have_masses = False
        bonds_all: List[np.ndarray] = []
        angles_all: List[np.ndarray] = []
        dihedrals_all: List[np.ndarray] = []

        for group_index, template in enumerate(templates):
            if template is None:
                continue
            per_unit = int(self.group_unit_sites[group_index])
            repeat = int(self.group_repeats[group_index])
            base = int(self.group_site_offsets[group_index])
            where = f"cg-topology group {group_index}"

            names = template.get("names")
            if names is not None:
                _check_len(names, per_unit, f"{where}.names")
                have_labels = True
                for rep in range(repeat):
                    offset = base + rep * per_unit
                    for position, name in enumerate(names):
                        labels[offset + position] = str(name)

            template_types = template.get("types")
            if template_types is not None:
                _check_len(template_types, per_unit, f"{where}.types")
                have_types = True
                type_unit = [str(name).strip() for name in template_types]
                if any(not name for name in type_unit):
                    raise ValueError(f"{where}.types entries must be non-empty strings.")
                for rep in range(repeat):
                    offset = base + rep * per_unit
                    canonical_types[offset : offset + per_unit] = type_unit

            template_charges = template.get("charges")
            if template_charges is not None:
                _check_len(template_charges, per_unit, f"{where}.charges")
                have_charges = True
                tiled = np.tile(np.asarray(template_charges, dtype=np.float64), repeat)
                charges[base : base + repeat * per_unit] = tiled

            template_masses = template.get("masses")
            if template_masses is not None and not (
                isinstance(template_masses, str) and template_masses.lower() == "auto"
            ):
                _check_len(template_masses, per_unit, f"{where}.masses")
                have_masses = True
                tiled = np.tile(np.asarray(template_masses, dtype=np.float64), repeat)
                masses[base : base + repeat * per_unit] = tiled

            for key, arity, sink in (
                ("bonds", 2, bonds_all),
                ("angles", 3, angles_all),
                ("dihedrals", 4, dihedrals_all),
            ):
                local = template.get(key)
                if not local:
                    continue
                local_arr = np.asarray(local, dtype=np.int64)
                if local_arr.ndim != 2 or local_arr.shape[1] != arity:
                    raise ValueError(
                        f"{where}.{key} must have shape (n, {arity}), got "
                        f"{local_arr.shape}."
                    )
                if local_arr.size and (
                    local_arr.min() < 0 or local_arr.max() >= per_unit
                ):
                    raise ValueError(
                        f"{where}.{key} indices must lie in [0, {per_unit}) — they "
                        "are site positions within one repeat unit, not global "
                        f"site ids. Got min={local_arr.min()}, max={local_arr.max()}."
                    )
                offsets = base + np.arange(repeat, dtype=np.int64) * per_unit
                sink.append((local_arr[None, :, :] + offsets[:, None, None]).reshape(-1, arity))

        typed_spec = self
        if have_types:
            type_table: List[str] = []
            type_index: Dict[str, int] = {}
            canonical_type_ids = np.asarray(
                [
                    _intern(type_index, type_table, name) + 1
                    for name in canonical_types
                ],
                dtype=_COMPACT_DTYPE,
            )
            canonical_mass = np.empty(len(type_table), dtype=np.float64)
            canonical_charge = np.empty(len(type_table), dtype=np.float64)
            for type_position, type_name in enumerate(type_table, start=1):
                selected = canonical_type_ids == type_position
                selected_masses = masses[selected]
                selected_charges = charges[selected]
                if not np.allclose(
                    selected_masses,
                    selected_masses[0],
                    rtol=1.0e-10,
                    atol=1.0e-12,
                ):
                    raise ValueError(
                        "cg-topology.types merges sites with inconsistent masses "
                        f"onto canonical type {type_name!r}; provide one common "
                        "per-site mass for that type."
                    )
                if not np.allclose(
                    selected_charges,
                    selected_charges[0],
                    rtol=1.0e-10,
                    atol=1.0e-12,
                ):
                    raise ValueError(
                        "cg-topology.types merges sites with inconsistent charges "
                        f"onto canonical type {type_name!r}; provide one common "
                        "per-site charge for that type."
                    )
                canonical_mass[type_position - 1] = float(selected_masses[0])
                canonical_charge[type_position - 1] = float(selected_charges[0])
            typed_spec = replace(
                self,
                site_type_ids=canonical_type_ids,
                type_names=tuple(type_table),
                type_xweight_sum=canonical_mass,
                type_charges=canonical_charge,
            )

        bonds, bond_ids, bond_keys = typed_spec._interaction_type_ids(bonds_all, 2)
        angles, angle_ids, angle_keys = typed_spec._interaction_type_ids(angles_all, 3)
        dihedrals, dihedral_ids, dihedral_keys = typed_spec._interaction_type_ids(
            dihedrals_all, 4
        )

        site_res_ids, res_name_ids, res_names, molnames = self._replicate_residues(layouts)

        return replace(
            typed_spec,
            site_labels=(
                tuple(
                    label if label is not None else typed_spec.type_names[type_id - 1]
                    for label, type_id in zip(labels, typed_spec.site_type_ids)
                )
                if have_labels
                else None
            ),
            site_charges=charges if have_charges else None,
            site_masses=masses if have_masses else None,
            site_res_ids=site_res_ids,
            res_name_ids=res_name_ids,
            res_names=res_names,
            group_molnames=molnames,
            bonds=bonds,
            bond_type_ids=bond_ids,
            bond_type_keys=bond_keys,
            angles=angles,
            angle_type_ids=angle_ids,
            angle_type_keys=angle_keys,
            dihedrals=dihedrals,
            dihedral_type_ids=dihedral_ids,
            dihedral_type_keys=dihedral_keys,
        )

    def _replicate_residues(
        self, layouts: Optional[List[Optional["_ResidueLayout"]]]
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[Tuple[str, ...]],
        Optional[Tuple[str, ...]],
    ]:
        """Broadcast per-repeat-unit residue layouts over every repeat.

        Returns ``(site_res_ids, res_name_ids, res_names, group_molnames)``, or
        four ``None`` when no group declared residues — i.e. the legacy molecule
        form, where the repeat unit *is* the residue and ``site_mol_ids`` says so
        already.
        """
        if not layouts or all(layout is None for layout in layouts):
            return None, None, None, None

        site_res_ids = np.full(self.n_sites, -1, dtype=np.int64)
        name_ids: List[int] = []
        name_table: List[str] = []
        name_index: Dict[str, int] = {}
        molnames: List[str] = []
        res_base = 0
        for group_index, layout in enumerate(layouts):
            per_unit = int(self.group_unit_sites[group_index])
            repeat = int(self.group_repeats[group_index])
            base = int(self.group_site_offsets[group_index])
            if layout is None:
                # A group with no residue declaration keeps one residue per
                # repeat unit, so the two forms can coexist in one file.
                molnames.append("")
                for rep in range(repeat):
                    offset = base + rep * per_unit
                    site_res_ids[offset : offset + per_unit] = res_base + rep
                    name_ids.append(_intern(name_index, name_table, "CG"))
                res_base += repeat
                continue
            molnames.append(layout.molname)
            n_res = len(layout.resnames)
            resids = layout.resids
            for rep in range(repeat):
                offset = base + rep * per_unit
                site_res_ids[offset : offset + per_unit] = (
                    res_base + rep * n_res + resids
                )
                for resname in layout.resnames:
                    name_ids.append(_intern(name_index, name_table, resname))
            res_base += repeat * n_res

        if np.any(site_res_ids < 0):  # pragma: no cover - every group is covered
            raise AssertionError("internal error: site without a CG residue id")
        return (
            site_res_ids,
            np.asarray(name_ids, dtype=_COMPACT_DTYPE),
            tuple(name_table),
            tuple(molnames),
        )

    def _interaction_type_ids(
        self, chunks: List[np.ndarray], arity: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[InteractionKey, ...]]]:
        """Assign 1-based interaction type ids to replicated bonded items.

        Type identity comes from the **site-type name tuple**, canonicalized by
        :class:`~AceCG.topology.types.InteractionKey` so that endpoint-reversed
        items share an id. Using type names (not per-site labels) is what makes
        e.g. both ``MG-PT1`` bonds of a lipid one bond type, and it matches the
        ``types: ["1", "2"]`` convention the trainers' ``[training.fm_specs]``
        already uses — so emitted ``.data`` type ids line up with FM/REM specs.

        Ids are assigned in order of first appearance, so type 1 is the first
        item in the file.
        """
        if not chunks:
            return None, None, None
        items = np.concatenate(chunks, axis=0).astype(_COMPACT_DTYPE)
        builder = {
            2: InteractionKey.bond,
            3: InteractionKey.angle,
            4: InteractionKey.dihedral,
        }[arity]
        # Distinct *type* patterns are few (bounded by the repeat unit), so
        # canonicalize only those and map back through the inverse index.
        type_rows = self.site_type_ids[items]
        unique_rows, inverse = np.unique(type_rows, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        n_items = items.shape[0]
        first_seen = np.full(unique_rows.shape[0], n_items, dtype=np.int64)
        np.minimum.at(first_seen, inverse, np.arange(n_items, dtype=np.int64))
        keys: List[InteractionKey] = []
        key_to_id: Dict[InteractionKey, int] = {}
        row_to_id = np.zeros(unique_rows.shape[0], dtype=_COMPACT_DTYPE)
        for row_index in np.argsort(first_seen, kind="stable"):
            key = builder(
                *(self.type_names[int(type_id) - 1] for type_id in unique_rows[row_index])
            )
            found = key_to_id.get(key)
            if found is None:
                found = len(keys) + 1
                key_to_id[key] = found
                keys.append(key)
            row_to_id[row_index] = found
        return items, row_to_id[inverse], tuple(keys)

    # ── consumer-facing views ─────────────────────────────────────

    def site_masses_array(self) -> np.ndarray:
        """Per-site CG mass, shape ``(n_sites,)``.

        Uses the ``cg-topology`` masses when supplied, else the summed raw
        ``x-weight`` of each site's type — the AA mass sum under the usual
        convention that ``x-weight`` holds atomic masses.
        """
        if self.site_masses is not None:
            return self.site_masses
        return self.type_xweight_sum[self.site_type_ids - 1]

    def site_charges_array(self) -> np.ndarray:
        """Per-site CG charge, shape ``(n_sites,)``."""
        if self.site_charges is not None:
            return self.site_charges
        return self.type_charges[self.site_type_ids - 1]

    def site_labels_array(self) -> Tuple[str, ...]:
        """Per-site bead label; falls back to the site-type name."""
        if self.site_labels is not None:
            return self.site_labels
        return tuple(self.type_names[type_id - 1] for type_id in self.site_type_ids)

    def site_arrays(
        self, *, resname: Union[str, Sequence[str]] = "CG"
    ) -> "SiteArrays":
        """Every per-site and per-residue quantity, as arrays, computed once.

        The mapping plan is fixed for the life of a spec, so these are too: the
        result is cached on the spec and returned on every later call. That matters
        because the consumers are per-rank — :func:`AceCG.workflows.trajmap` builds
        a CG universe on every rank, and each build used to re-derive a
        ``n_sites``-long label tuple, a resname tuple and two fancy-indexed arrays.

        The cache is deliberately *not* pickled, so an MPI broadcast of the spec
        still moves only the compiled plan.

        Parameters
        ----------
        resname : str or sequence of str
            Fallback residue name(s), used only when the mapping YAML declared no
            residues. Passing a different value than a cached call raises rather
            than silently returning the cached names.

        Returns
        -------
        SiteArrays
        """
        # Validate before consulting the cache, so a malformed argument is still
        # rejected on a second call. When the YAML declared residues the argument
        # is ignored entirely, and the key is empty so it can never mismatch.
        key: Tuple[str, ...] = ()
        if self.site_res_ids is None:
            self._group_resnames(resname)
            key = _resname_key(resname)

        cached = getattr(self, "_site_arrays_cache", None)
        if cached is not None:
            if cached.resname_key != key:
                raise ValueError(
                    f"site_arrays() was first built with resname={cached.resname_key!r} "
                    f"and is now asked for {key!r}. Residue naming is fixed for a "
                    "spec; compile a second spec if two namings are needed."
                )
            return cached

        res_ids, residue_names = self._residue_layout(resname)
        arrays = SiteArrays(
            labels=self.site_labels_array(),
            type_ids=self.site_type_ids,
            type_names=self.type_names,
            masses=self.site_masses_array(),
            charges=self.site_charges_array(),
            mol_ids=np.asarray(self.site_mol_ids, dtype=np.int64),
            res_ids=res_ids,
            residue_names=residue_names,
            resname_key=key,
        )
        # `frozen=True` blocks normal assignment; this is the standard lazy-cache
        # escape hatch and keeps the field out of the dataclass contract.
        object.__setattr__(self, "_site_arrays_cache", arrays)
        return arrays

    def _residue_layout(
        self, resname: Union[str, Sequence[str]]
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        """``(per-site residue id, per-residue name)`` from the YAML or *resname*."""
        if self.site_res_ids is not None:
            return self.site_res_ids, self.residue_names_array()
        # No residue form: a residue is one repeat unit, the historical meaning of
        # `site_mol_ids`, and its name comes from the owning group.
        n_groups = len(self.group_repeats)
        group_resnames = self._group_resnames(resname)
        per_residue: List[str] = []
        for group_index in range(n_groups):
            per_residue.extend(
                [group_resnames[group_index]] * int(self.group_repeats[group_index])
            )
        return np.asarray(self.site_mol_ids, dtype=np.int64), tuple(per_residue)

    def _group_resnames(self, resname: Union[str, Sequence[str]]) -> List[str]:
        """One fallback residue name per ``system`` group, length-checked."""
        n_groups = len(self.group_repeats)
        if isinstance(resname, (list, tuple)):
            if len(resname) != n_groups:
                raise ValueError(
                    f"resname sequence has {len(resname)} entries but the mapping has "
                    f"{n_groups} system group(s)."
                )
            return [str(name) for name in resname]
        return [str(resname)] * n_groups

    def __getstate__(self) -> Dict[str, Any]:
        """Pickle the compiled plan only, never the derived-array cache."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "_site_arrays_cache"
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        for key, value in state.items():
            object.__setattr__(self, key, value)

    def site_residues(
        self, *, resname: Union[str, Sequence[str]] = "CG"
    ) -> Tuple[np.ndarray, Tuple[str, ...]]:
        """Per-site ``(residue id, residue name)``, YAML first.

        With a residue-form ``cg-topology`` the ids and names come straight from
        the mapping file. Otherwise a residue is one repeat unit — the historical
        meaning of ``site_mol_ids`` — and the names come from *resname*.

        Parameters
        ----------
        resname : str or sequence of str
            Fallback residue name, one for everything or one per ``system``
            group. Unused when the YAML declared residues.

        Returns
        -------
        res_ids : np.ndarray, shape (n_sites,)
            0-based global CG residue index per site.
        res_names : tuple of str
            Residue name per site. Expanding this costs one string reference per
            site; prefer :meth:`site_arrays` and its per-*residue* ``residue_names``
            when the caller does not need one name per site.
        """
        arrays = self.site_arrays(resname=resname)
        return arrays.res_ids, arrays.site_resnames()

    def residue_names_array(self) -> Tuple[str, ...]:
        """Residue name per global CG residue, or ``()`` without a residue form.

        See :meth:`site_arrays` for the form that also covers the no-residue-form
        case, where a residue is one repeat unit named after its group.
        """
        if self.res_name_ids is None or self.res_names is None:
            return ()
        return tuple(self.res_names[int(name_id)] for name_id in self.res_name_ids)

    @property
    def n_residues(self) -> int:
        """CG residue count: from the YAML when declared, else one per repeat unit."""
        if self.res_name_ids is not None:
            return int(self.res_name_ids.shape[0])
        return self.n_mol

    def bead_records(
        self,
        *,
        resname: Union[str, Sequence[str]] = "CG",
        resid_base: int = 1,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, float]]:
        """Build the ``beads`` / ``type2id`` / ``type_masses`` trio.

        These are exactly the arguments
        :func:`AceCG.io.coordinates_writers.write_lammps_data` and
        :func:`~AceCG.io.coordinates_writers.write_gro` take, so CG topology
        emission reuses the existing writers unchanged.

        Residue identity comes from the mapping YAML whenever the
        ``cg-topology`` block declared it (the residue form), because one system
        can hold several species — two protein chains, a lipid mixture, a
        protein-lipid system — and a single config-level name cannot describe
        that. The *resname* argument is then unused. It remains the source for
        mappings whose YAML says nothing about residues.

        Parameters
        ----------
        resname : str or sequence of str
            Fallback only: one residue name for every molecule, or one per
            ``system`` group. Ignored when the YAML supplied residue names.
        resid_base : int, default 1
            First CG residue id.

        Returns
        -------
        beads : list of dict
            Per-site records with ``bead_id``, ``bead_type``, ``resid``,
            ``resname``, ``mass``, ``q``, ``label``.
        type2id : dict
            ``{site-type name: 1-based type id}``.
        type_masses : dict
            ``{site-type name: mass}``. Per-site masses that disagree within a
            type are reduced to their mean, which only happens when a
            ``cg-topology`` block assigns different masses to the same type.
        """
        arrays = self.site_arrays(resname=resname)
        site_res_ids = arrays.res_ids
        residue_names = arrays.residue_names
        masses = arrays.masses
        charges = arrays.charges
        labels = arrays.labels
        beads: List[Dict[str, Any]] = []
        for site_index in range(self.n_sites):
            type_id = int(arrays.type_ids[site_index])
            beads.append(
                {
                    "bead_id": site_index + 1,
                    "bead_type": self.type_names[type_id - 1],
                    "resid": resid_base + int(site_res_ids[site_index]),
                    "resname": residue_names[int(site_res_ids[site_index])],
                    # The LAMMPS molecule id stays the *molecule* even when the
                    # residue form makes `resid` finer. Writing `resid` there
                    # would silently redefine what `exclude_option="resid"`
                    # groups: for a 99-residue protein chain it would exclude
                    # only intra-amino-acid nonbonded pairs instead of the whole
                    # chain. One residue per repeat unit (a lipid) is unaffected,
                    # which is why this stayed invisible on DPPC.
                    "molid": resid_base + int(arrays.mol_ids[site_index]),
                    "mass": float(masses[site_index]),
                    "q": float(charges[site_index]),
                    "label": labels[site_index],
                }
            )
        type2id = {name: index + 1 for index, name in enumerate(self.type_names)}
        type_masses: Dict[str, float] = {}
        for index, name in enumerate(self.type_names):
            selected = masses[self.site_type_ids == index + 1]
            type_masses[name] = float(selected.mean()) if selected.size else 0.0
        return beads, type2id, type_masses


# ─── module-level helpers ─────────────────────────────────────────────


def _check_len(values: Any, expected: int, where: str) -> None:
    try:
        length = len(values)
    except TypeError as exc:
        raise ValueError(f"{where} must be a sequence of length {expected}.") from exc
    if length != expected:
        raise ValueError(
            f"{where} has {length} entries but the repeat unit has {expected} site(s)."
        )


def _resname_key(resname: Union[str, Sequence[str]]) -> Tuple[str, ...]:
    """Canonical form of a fallback-resname argument, for cache identity."""
    if isinstance(resname, (list, tuple)):
        return tuple(str(name) for name in resname)
    return (str(resname),)


@dataclass(frozen=True, eq=False)
class SiteArrays:
    """Every per-site quantity of a :class:`CGMapSpec`, as arrays.

    Built once by :meth:`CGMapSpec.site_arrays` and cached there. Nothing here is
    derived per frame — it is all fixed by the compiled mapping plan — so a
    consumer that needs names, types, masses, charges and residues (a CG universe,
    a topology writer) reads them instead of rebuilding a dict per site.

    Attributes
    ----------
    labels : tuple of str
        Per-site bead label, ``n_sites`` long.
    type_ids : np.ndarray
        1-based site-type id per site.
    type_names : tuple of str
        Site-type names in id order, so ``type_names[type_ids[i] - 1]`` is site
        ``i``'s type.
    masses, charges : np.ndarray
        Per-site mass and charge.
    mol_ids : np.ndarray
        0-based molecule (repeat-unit) id per site.
    res_ids : np.ndarray
        0-based CG residue id per site.
    residue_names : tuple of str
        Residue name per *residue* — ``n_residues`` long, not ``n_sites``. This is
        the form MDAnalysis' per-residue topology attributes want.
    resname_key : tuple of str
        The fallback-resname argument these were built with.
    """

    labels: Tuple[str, ...]
    type_ids: np.ndarray
    type_names: Tuple[str, ...]
    masses: np.ndarray
    charges: np.ndarray
    mol_ids: np.ndarray
    res_ids: np.ndarray
    residue_names: Tuple[str, ...]
    resname_key: Tuple[str, ...]

    @property
    def n_sites(self) -> int:
        """Number of CG sites."""
        return int(self.type_ids.shape[0])

    @property
    def n_residues(self) -> int:
        """Number of CG residues."""
        return len(self.residue_names)

    def site_types(self) -> Tuple[str, ...]:
        """Per-site type *name*, for writers that name types rather than number them."""
        return tuple(self.type_names[int(type_id) - 1] for type_id in self.type_ids)

    def site_resnames(self) -> Tuple[str, ...]:
        """Per-*site* residue name — one entry per site, expanded on request."""
        return tuple(self.residue_names[int(res_id)] for res_id in self.res_ids)


def _intern(index: Dict[str, int], table: List[str], value: str) -> int:
    """Append ``value`` to ``table`` if new; return its position."""
    found = index.get(value)
    if found is None:
        found = len(table)
        index[value] = found
        table.append(value)
    return found


# ─── residue-form cg-topology ─────────────────────────────────────────


_RESIDUE_KEYS = frozenset(
    {
        "resname",
        "resid",
        "names",
        "types",
        "charges",
        "masses",
        "linkable",
        "left_linker_atom_type",
        "right_linker_atom_type",
        "bonds",
        "angles",
        "dihedrals",
    }
)

_RESIDUE_GROUP_KEYS = frozenset(
    {
        "molname",
        "resnames",
        "resids",
        "link",
        "linker_angles",
        "additional_bonds",
        "additional_angles",
        "additional_dihedrals",
    }
)

_BONDED_ARITY = {"bonds": 2, "angles": 3, "dihedrals": 4}


@dataclass(frozen=True)
class _ResidueLayout:
    """Which residue each site of one repeat unit belongs to.

    Attributes
    ----------
    molname : str
        The group's molecule name.
    resids : np.ndarray, shape (n_unit_sites,), int64
        Residue index within the repeat unit, per site.
    resnames : tuple of str
        Residue name per residue of the repeat unit.
    """

    molname: str
    resids: np.ndarray
    resnames: Tuple[str, ...]


def _is_residue_form(block: Mapping[str, Any]) -> bool:
    """``True`` for the residue form, ``False`` for the legacy molecule form."""
    return isinstance(block, Mapping) and bool(block.get("residues"))


def _residue_form_templates(
    block: Mapping[str, Any],
    *,
    n_groups: int,
    group_unit_sites: np.ndarray,
) -> Tuple[List[Optional[Mapping[str, Any]]], List[Optional[_ResidueLayout]]]:
    """Compile the residue form into legacy per-group templates plus layouts.

    The output feeds the same replication machinery the molecule form uses, so
    residue templates are a *front end* rather than a second code path: one
    repeat-unit-local template per group, with linker bonds and residue-qualified
    additional terms already resolved to unit-local site indices.
    """
    residues = block.get("residues")
    if not isinstance(residues, (list, tuple)) or not residues:
        raise ValueError("cg-topology.residues must be a non-empty list.")
    if "linker_angles" in block:
        raise ValueError(
            "cg-topology.linker_angles is group-local; put the boolean inside "
            "each cg-topology.groups entry that constructs a linked molecule."
        )

    templates_by_resname: Dict[str, Dict[str, Any]] = {}
    for position, raw in enumerate(residues):
        where = f"cg-topology.residues[{position}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where} must be a mapping.")
        unknown = set(raw) - _RESIDUE_KEYS
        if unknown:
            raise ValueError(
                f"{where} has unknown key(s) {sorted(unknown)}. "
                f"Allowed: {sorted(_RESIDUE_KEYS)}"
            )
        resname = str(raw.get("resname", "")).strip()
        if not resname:
            raise ValueError(f"{where} needs a non-empty 'resname'.")
        if resname in templates_by_resname:
            raise ValueError(
                f"{where}: residue template {resname!r} is declared twice; "
                "resname is the template key and must be unique."
            )
        templates_by_resname[resname] = _normalize_residue_template(raw, where)

    groups = block.get("groups")
    if groups is None:
        # A single-group mapping needs no `groups` block: one residue template is
        # unambiguous, so accept the shorthand.
        if n_groups == 1 and len(templates_by_resname) == 1:
            only = next(iter(templates_by_resname))
            groups = [{"molname": only, "resnames": [only]}]
        else:
            raise ValueError(
                "cg-topology.groups is required: the mapping has "
                f"{n_groups} system group(s) and {len(templates_by_resname)} "
                "residue template(s), so which residues make up each group cannot "
                "be inferred."
            )
    if not isinstance(groups, (list, tuple)):
        raise ValueError("cg-topology.groups must be a list.")
    if len(groups) != n_groups:
        raise ValueError(
            f"cg-topology.groups has {len(groups)} entries but the mapping has "
            f"{n_groups} system group(s); they pair up in order."
        )

    compiled: List[Optional[Mapping[str, Any]]] = []
    layouts: List[Optional[_ResidueLayout]] = []
    for group_index, raw_group in enumerate(groups):
        where = f"cg-topology.groups[{group_index}]"
        if not raw_group:
            compiled.append(None)
            layouts.append(None)
            continue
        template, layout = _compile_residue_group(
            raw_group,
            templates_by_resname,
            per_unit=int(group_unit_sites[group_index]),
            where=where,
        )
        compiled.append(template)
        layouts.append(layout)
    return compiled, layouts


def _normalize_residue_template(raw: Mapping[str, Any], where: str) -> Dict[str, Any]:
    """Validate one residue template and resolve its linker site positions."""
    names = raw.get("names")
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError(f"{where}.names must be a non-empty list of site labels.")
    labels = [str(name) for name in names]
    n_sites = len(labels)

    types = raw.get("types")
    if types is not None:
        if not isinstance(types, (list, tuple)) or len(types) != n_sites:
            raise ValueError(
                f"{where}.types must be a list of {n_sites} site-type name(s) to "
                f"match 'names'."
            )
        types = [str(item) for item in types]

    for key in ("charges", "masses"):
        values = raw.get(key)
        if values is None:
            continue
        if isinstance(values, str):
            if values.strip().lower() != "auto":
                raise ValueError(f"{where}.{key} must be a list or the word 'auto'.")
            continue
        _check_len(values, n_sites, f"{where}.{key}")

    bonded: Dict[str, np.ndarray] = {}
    for key, arity in _BONDED_ARITY.items():
        rows = raw.get(key)
        if not rows:
            continue
        array = np.asarray(rows, dtype=np.int64)
        if array.ndim != 2 or array.shape[1] != arity:
            raise ValueError(
                f"{where}.{key} must have shape (n, {arity}), got {array.shape}."
            )
        if array.size and (array.min() < 0 or array.max() >= n_sites):
            raise ValueError(
                f"{where}.{key} indices must lie in [0, {n_sites}) — they are site "
                f"positions inside this residue. Got min={array.min()}, "
                f"max={array.max()}."
            )
        bonded[key] = array

    linkable = bool(raw.get("linkable", False))
    linkers: Dict[str, Optional[int]] = {"left": None, "right": None}
    for side in ("left", "right"):
        key = f"{side}_linker_atom_type"
        declared = raw.get(key)
        if declared is None:
            continue
        linkers[side] = _resolve_linker_site(str(declared), labels, types, f"{where}.{key}")
    if linkable and (linkers["left"] is None or linkers["right"] is None):
        missing = [
            f"{side}_linker_atom_type" for side in ("left", "right") if linkers[side] is None
        ]
        raise ValueError(
            f"{where} is linkable but declares no {missing}; 1-D sequential "
            "linking needs both ends named."
        )

    template: Dict[str, Any] = {
        "resname": str(raw["resname"]).strip(),
        "names": labels,
        "n_sites": n_sites,
        "linkable": linkable,
        "left_linker": linkers["left"],
        "right_linker": linkers["right"],
    }
    if raw.get("resid") is not None:
        template["resid"] = _as_int(raw["resid"], f"{where}.resid")
    if types is not None:
        template["types"] = types
    for key in ("charges", "masses"):
        if raw.get(key) is not None:
            template[key] = raw[key]
    template.update(bonded)
    return template


def _resolve_linker_site(
    declared: str,
    labels: Sequence[str],
    types: Optional[Sequence[str]],
    where: str,
) -> int:
    """Position of the named linker site inside one residue.

    Resolved against the residue's site labels first, then its declared site-type
    names, so both spellings of ``left_linker_atom_type: N`` work.
    """
    for candidates, what in ((labels, "names"), (types, "types")):
        if not candidates:
            continue
        matches = [index for index, name in enumerate(candidates) if name == declared]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"{where}={declared!r} matches {len(matches)} entries of {what}; "
                "a linker must name exactly one site."
            )
    raise ValueError(
        f"{where}={declared!r} is not one of this residue's site names "
        f"{list(labels)}"
        + ("" if types is None else f" or site types {list(types)}")
        + "."
    )


def _compile_residue_group(
    raw_group: Mapping[str, Any],
    templates_by_resname: Mapping[str, Mapping[str, Any]],
    *,
    per_unit: int,
    where: str,
) -> Tuple[Dict[str, Any], _ResidueLayout]:
    """Assemble one group's repeat-unit template from its residue sequence."""
    if not isinstance(raw_group, Mapping):
        raise ValueError(f"{where} must be a mapping.")
    unknown = set(raw_group) - _RESIDUE_GROUP_KEYS
    if unknown:
        raise ValueError(
            f"{where} has unknown key(s) {sorted(unknown)}. "
            f"Allowed: {sorted(_RESIDUE_GROUP_KEYS)}"
        )
    linker_angles = raw_group.get("linker_angles", False)
    if not isinstance(linker_angles, bool):
        raise ValueError(
            f"{where}.linker_angles must be a boolean (true or false)."
        )

    raw_resnames = raw_group.get("resnames")
    if not isinstance(raw_resnames, (list, tuple)) or not raw_resnames:
        raise ValueError(f"{where}.resnames must be a non-empty list of residue names.")
    resnames = tuple(str(name) for name in raw_resnames)
    missing = sorted({name for name in resnames if name not in templates_by_resname})
    if missing:
        raise ValueError(
            f"{where}.resnames names residue template(s) {missing} that "
            f"cg-topology.residues does not declare. Known: "
            f"{sorted(templates_by_resname)}"
        )
    residue_templates = [templates_by_resname[name] for name in resnames]

    raw_resids = raw_group.get("resids")
    if raw_resids is None:
        # One residue in the unit: every site belongs to it, no list needed.
        if len(resnames) != 1:
            raise ValueError(
                f"{where}.resids is required: the repeat unit has "
                f"{len(resnames)} residues, so which sites belong to which cannot "
                "be inferred."
            )
        resids = np.zeros(per_unit, dtype=np.int64)
    else:
        resids = np.asarray(raw_resids, dtype=np.int64).ravel()
        if resids.size != per_unit:
            raise ValueError(
                f"{where}.resids has {resids.size} entries but the repeat unit has "
                f"{per_unit} site(s); it is one residue index per site."
            )
        if resids.size and (resids.min() < 0 or resids.max() >= len(resnames)):
            raise ValueError(
                f"{where}.resids must lie in [0, {len(resnames)}) to index "
                f"'resnames'. Got min={resids.min()}, max={resids.max()}."
            )
        seen = np.unique(resids)
        if seen.size != len(resnames):
            raise ValueError(
                f"{where}: 'resids' uses {seen.size} of the {len(resnames)} "
                "residue(s) in 'resnames'; every declared residue must own at "
                "least one site."
            )

    # Site positions of each residue, in unit order: this is what turns a
    # residue-local index into a unit-local one.
    sites_of_residue: List[np.ndarray] = [
        np.nonzero(resids == index)[0] for index in range(len(resnames))
    ]
    for index, (template, sites) in enumerate(zip(residue_templates, sites_of_residue)):
        if sites.size != int(template["n_sites"]):
            raise ValueError(
                f"{where}: residue {index} ({resnames[index]}) owns {sites.size} "
                f"site(s) by 'resids' but its template declares "
                f"{int(template['n_sites'])} name(s)."
            )

    names: List[str] = [""] * per_unit
    types: Optional[List[str]] = []
    charges: Optional[List[float]] = []
    masses: Optional[List[Any]] = []
    for template, sites in zip(residue_templates, sites_of_residue):
        for local, unit_site in enumerate(sites):
            names[int(unit_site)] = str(template["names"][local])
        if types is not None:
            values = template.get("types")
            types = None if values is None else types + [str(v) for v in values]
        # A residue that says nothing (or ``"auto"``) about charges/masses makes
        # the whole group fall back to the type-level defaults, rather than
        # leaving some sites unset.
        if charges is not None:
            values = template.get("charges")
            values = None if isinstance(values, str) else values
            charges = None if values is None else charges + [float(v) for v in values]
        if masses is not None:
            values = template.get("masses")
            values = None if isinstance(values, str) else values
            masses = None if values is None else masses + [float(v) for v in values]

    compiled: Dict[str, Any] = {"names": names}
    if types is not None and len(types) == per_unit:
        compiled["types"] = _reorder_to_unit(types, sites_of_residue, per_unit)
    if charges is not None and len(charges) == per_unit:
        compiled["charges"] = _reorder_to_unit(charges, sites_of_residue, per_unit)
    if masses is not None and len(masses) == per_unit:
        compiled["masses"] = _reorder_to_unit(masses, sites_of_residue, per_unit)

    local_terms: Dict[str, List[List[int]]] = {}
    for key in _BONDED_ARITY:
        terms: List[List[int]] = []
        for template, sites in zip(residue_templates, sites_of_residue):
            rows = template.get(key)
            if rows is None:
                continue
            terms.extend(sites[np.asarray(rows, dtype=np.int64)].tolist())
        local_terms[key] = terms

    sequential_bonds: List[List[int]] = []
    if bool(raw_group.get("link", True)):
        sequential_bonds = _sequential_link_bonds(
            residue_templates, resnames, sites_of_residue, where
        )

    for key, arity in _BONDED_ARITY.items():
        terms = list(local_terms[key])
        if key == "bonds":
            terms.extend(sequential_bonds)
        elif key == "angles" and linker_angles:
            terms.extend(
                _angles_involving_linker_bonds(
                    local_terms["bonds"], sequential_bonds
                )
            )
        extra = raw_group.get(f"additional_{key}")
        if extra:
            terms.extend(
                _resolve_additional_terms(
                    extra,
                    arity=arity,
                    residue_templates=residue_templates,
                    sites_of_residue=sites_of_residue,
                    per_unit=per_unit,
                    where=f"{where}.additional_{key}",
                )
            )
        if terms:
            compiled[key] = terms

    return compiled, _ResidueLayout(
        molname=str(raw_group.get("molname", resnames[0])),
        resids=resids,
        resnames=resnames,
    )


def _angles_involving_linker_bonds(
    local_bonds: Sequence[Sequence[int]],
    linker_bonds: Sequence[Sequence[int]],
) -> List[List[int]]:
    """All unique bond-graph angles containing an automatic linker bond.

    Purely intra-residue angles remain controlled by each residue template.
    ``additional_bonds`` are deliberately excluded: angles around exceptional
    bonds such as disulfides remain explicit ``additional_angles``.
    """
    if not linker_bonds:
        return []

    linker_edges = {
        frozenset((int(bond[0]), int(bond[1]))) for bond in linker_bonds
    }
    neighbours: Dict[int, set] = {}
    for bond in [*local_bonds, *linker_bonds]:
        left, right = int(bond[0]), int(bond[1])
        neighbours.setdefault(left, set()).add(right)
        neighbours.setdefault(right, set()).add(left)

    angles: List[List[int]] = []
    for centre in sorted(neighbours):
        partners = sorted(neighbours[centre])
        for position, left in enumerate(partners):
            for right in partners[position + 1 :]:
                if (
                    frozenset((left, centre)) in linker_edges
                    or frozenset((centre, right)) in linker_edges
                ):
                    angles.append([left, centre, right])
    return angles


def _reorder_to_unit(
    per_residue_values: Sequence[Any],
    sites_of_residue: Sequence[np.ndarray],
    per_unit: int,
) -> List[Any]:
    """Scatter values gathered residue-by-residue back into unit-site order."""
    out: List[Any] = [None] * per_unit
    cursor = 0
    for sites in sites_of_residue:
        for unit_site in sites:
            out[int(unit_site)] = per_residue_values[cursor]
            cursor += 1
    return out


def _sequential_link_bonds(
    residue_templates: Sequence[Mapping[str, Any]],
    resnames: Sequence[str],
    sites_of_residue: Sequence[np.ndarray],
    where: str,
) -> List[List[int]]:
    """Bonds joining consecutive residues through their declared linkers.

    Only 1-D sequential linking is supported: residue ``r``'s
    ``right_linker_atom_type`` bonds to residue ``r + 1``'s
    ``left_linker_atom_type``, N-to-C for a protein chain.
    """
    bonds: List[List[int]] = []
    for index in range(len(residue_templates) - 1):
        left, right = residue_templates[index], residue_templates[index + 1]
        if not left["linkable"] and not right["linkable"]:
            continue
        if bool(left["linkable"]) != bool(right["linkable"]):
            unlinkable = index if not left["linkable"] else index + 1
            raise ValueError(
                f"{where}: residue {index} ({resnames[index]}) and residue "
                f"{index + 1} ({resnames[index + 1]}) are consecutive but only one "
                f"is linkable, so the molecule would be disconnected at residue "
                f"{unlinkable}. Set 'linkable' consistently, or set "
                "'link: false' and declare the bonds explicitly."
            )
        bonds.append(
            [
                int(sites_of_residue[index][int(left["right_linker"])]),
                int(sites_of_residue[index + 1][int(right["left_linker"])]),
            ]
        )
    return bonds


def _resolve_additional_terms(
    entries: Any,
    *,
    arity: int,
    residue_templates: Sequence[Mapping[str, Any]],
    sites_of_residue: Sequence[np.ndarray],
    per_unit: int,
    where: str,
) -> List[List[int]]:
    """Resolve ``additional_*`` entries into unit-local site indices.

    Each item of an entry is either a plain unit-local site index, or a
    ``[resid, site]`` pair naming a site inside one residue of the unit — by its
    label or by its position in that residue.
    """
    if not isinstance(entries, (list, tuple)):
        raise ValueError(f"{where} must be a list of {arity}-item terms.")
    resolved: List[List[int]] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, (list, tuple)) or len(entry) != arity:
            raise ValueError(
                f"{where}[{position}] must list exactly {arity} sites, got {entry!r}."
            )
        term: List[int] = []
        for item in entry:
            term.append(
                _resolve_additional_site(
                    item,
                    residue_templates=residue_templates,
                    sites_of_residue=sites_of_residue,
                    per_unit=per_unit,
                    where=f"{where}[{position}]",
                )
            )
        if len(set(term)) != arity:
            raise ValueError(f"{where}[{position}] repeats a site: {term}.")
        resolved.append(term)
    return resolved


def _resolve_additional_site(
    item: Any,
    *,
    residue_templates: Sequence[Mapping[str, Any]],
    sites_of_residue: Sequence[np.ndarray],
    per_unit: int,
    where: str,
) -> int:
    """One ``additional_*`` site reference as a unit-local site index."""
    if isinstance(item, (list, tuple)):
        if len(item) != 2:
            raise ValueError(
                f"{where}: a residue-qualified site must be [resid, site], got {item!r}."
            )
        resid = _as_int(item[0], f"{where} resid")
        if not 0 <= resid < len(residue_templates):
            raise ValueError(
                f"{where}: resid {resid} is outside [0, {len(residue_templates)})."
            )
        template = residue_templates[resid]
        labels = [str(name) for name in template["names"]]
        raw_site = item[1]
        if isinstance(raw_site, str) and not raw_site.lstrip("-").isdigit():
            local = _resolve_linker_site(
                raw_site, labels, template.get("types"), f"{where} site"
            )
        else:
            local = _as_int(raw_site, f"{where} site")
            if not 0 <= local < len(labels):
                raise ValueError(
                    f"{where}: site {local} is outside residue {resid}'s "
                    f"[0, {len(labels)})."
                )
        return int(sites_of_residue[resid][local])
    index = _as_int(item, f"{where} site")
    if not 0 <= index < per_unit:
        raise ValueError(
            f"{where}: unit-local site {index} is outside [0, {per_unit})."
        )
    return index


def _cg_topology_templates(
    block: Mapping[str, Any], *, n_groups: int
) -> List[Optional[Mapping[str, Any]]]:
    """Normalize a legacy molecule-form ``cg-topology`` block into per-group templates."""
    if not isinstance(block, Mapping):
        raise ValueError(
            f"cg-topology must be a mapping with a 'molecule' or 'groups' key, got "
            f"{type(block).__name__}."
        )
    if "groups" in block:
        groups = block["groups"]
        if not isinstance(groups, (list, tuple)):
            raise ValueError("cg-topology.groups must be a list.")
        if len(groups) != n_groups:
            raise ValueError(
                f"cg-topology.groups has {len(groups)} entries but the mapping has "
                f"{n_groups} system group(s)."
            )
        return [g if g else None for g in groups]
    if "molecule" in block:
        molecule = block["molecule"]
        if not isinstance(molecule, Mapping):
            raise ValueError("cg-topology.molecule must be a mapping.")
        return [molecule] * n_groups
    # Tolerate a bare template (the keys inlined at the top of the block).
    if any(
        key in block
        for key in ("names", "types", "bonds", "angles", "dihedrals", "charges", "masses")
    ):
        return [block] * n_groups
    raise ValueError(
        "cg-topology must contain 'molecule', 'groups', or an inline template "
        "(names/types/charges/masses/bonds/angles/dihedrals)."
    )


def _resolve_mol_reference(
    *,
    mol_reference: Union[str, int],
    atom_indices: np.ndarray,
    mol_indptr: np.ndarray,
    mol_atom_pos: Optional[np.ndarray],
    group_repeats: np.ndarray,
    group_mol_offsets: np.ndarray,
    unit_origins: List[np.ndarray],
    shift: int,
) -> np.ndarray:
    """Compact-buffer position of each molecule's PBC reference atom."""
    n_mol = int(mol_indptr.shape[0] - 1)
    # Default: the molecule's lowest mapped atom, which is where `mol_indptr`
    # already points.
    starts = mol_indptr[:-1]
    first = starts if mol_atom_pos is None else mol_atom_pos[starts]
    first = first.astype(_COMPACT_DTYPE)
    if isinstance(mol_reference, str):
        mode = mol_reference.lower()
        if mode == "first":
            return first
        if mode != "anchor":
            raise ValueError(
                f"mol_reference must be 'first', 'anchor', or an int offset, got "
                f"{mol_reference!r}"
            )
        local_offset = 0
    else:
        local_offset = _as_int(mol_reference, "mol_reference")

    wanted = np.empty(n_mol, dtype=_ABS_DTYPE)
    for group_index, origins in enumerate(unit_origins):
        start = int(group_mol_offsets[group_index])
        stop = start + int(group_repeats[group_index])
        # `shift` is applied twice on purpose: the reference atom plays the role
        # of `base + idx` in the site arithmetic (base = the unit origin,
        # idx = local_offset), and index_base=1 shifts both terms. This keeps
        # 1-based mappings self-consistent with the compiled site indices.
        wanted[start:stop] = origins + 2 * shift + local_offset
    found = np.searchsorted(atom_indices, wanted)
    valid = (found < atom_indices.size) & (atom_indices[np.minimum(found, atom_indices.size - 1)] == wanted)
    out = np.where(valid, found, first).astype(_COMPACT_DTYPE)
    if not np.all(valid):
        warnings.warn(
            f"{int(np.count_nonzero(~valid))} of {n_mol} molecule reference atoms "
            f"(mol_reference={mol_reference!r}) are not part of the mapping; those "
            "molecules fall back to their first mapped atom.",
            RuntimeWarning,
            stacklevel=3,
        )
    return out


def load_cgmap_spec(
    path: Union[str, Path],
    *,
    index_base: int = 0,
    masses: Optional[np.ndarray] = None,
    n_atoms: Optional[int] = None,
    mol_reference: Union[str, int] = "first",
    cg_topology: Optional[Mapping[str, Any]] = None,
    strict_weights: bool = True,
    comm: Optional[Any] = None,
    root_error: Optional[Exception] = None,
) -> CGMapSpec:
    """Load a mapping YAML and compile it.

    Thin convenience wrapper over :func:`load_mapping_yaml` plus
    :meth:`CGMapSpec.from_mapping`; see the latter for the parameters.
    """
    rank = 0 if comm is None else int(comm.Get_rank())
    compiled = error = None
    if rank == 0:
        try:
            if root_error is not None:
                raise root_error
            mapping = load_mapping_yaml(path)
            if not isinstance(mapping, Mapping):
                raise ValueError(f"{path} did not parse into a mapping (got {type(mapping).__name__}).")
            compiled = CGMapSpec.from_mapping(
                mapping, index_base=index_base, masses=masses, n_atoms=n_atoms,
                mol_reference=mol_reference, cg_topology=cg_topology, strict_weights=strict_weights,
            )
        except Exception as exc:
            error = exc
    if comm is None:
        if error is not None:
            raise error
        assert compiled is not None
        return compiled
    from ..io.trajectory import broadcast_root_outcome
    return broadcast_root_outcome((compiled, error) if rank == 0 else None, comm=comm)
