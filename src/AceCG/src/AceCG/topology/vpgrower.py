"""Virtual-particle (VP) grower.

This module owns the *static* VP topology template and the *per-frame*
geometric VP placement. It is entirely independent of trainers,
optimizers, samplers, and the legacy ``_workflows_old/vp.py``.

Public surface
--------------
``VPTopologyTemplate``
    Frozen snapshot of what the LAMMPS data file for a grown frame
    will look like, sans positions: atom type map, masses, bond /
    angle tables, which indices are "real" (present in the AA-mapped
    trajectory) and which are newly inserted VP beads.

``VPGrownFrame``
    Frozen per-frame artifact: ``positions`` aligned with
    ``template`` atom indices, plus box ``dimensions``.

``VPGrower``
    Stateless (after construction) grower. ``from_universe(u, vp_config)``
    builds a :class:`VPTopologyTemplate`; ``grow_frame(positions_real,
    dimensions, orientation_seed)`` returns a :class:`VPGrownFrame` by
    seeding VP positions at the residue center and iterating an
    analytic placement + anti-clash loop.

``write_vp_data(template, frame, path, title=...)``
    Thin wrapper around :func:`AceCG.io.coordinates_writers.write_lammps_data`.

Design constraints (see ``human_comments/design/claude8_vpgrower_v3.md``)
- CG-only template: the reference topology / trajectory contains zero
  VP atoms. VPs are appended per carrier residue.
- Only ``[vp_bonds]`` and ``[vp_angles]`` harmonic terms shape the
  growth geometry. ``[vp_dihedrals]`` is accepted and passed through
  to the emitted LAMMPS data file (real + VP-inserted 4-tuples,
  type ids, per-type InteractionKey map) but it does not influence
  the geometric VP placement.
- Alias resolution: ``[vp_*]`` labels may be bead aliases
  (``"HG"``, ``"MG"``, …) or bare LAMMPS type ids; the grower accepts
  the ``{int_id: name}`` map and resolves consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from MDAnalysis.lib.distances import capped_distance, self_capped_distance, minimize_vectors

import MDAnalysis as mda

from ..configs.vp_config import VPConfig, VPInteractionDef
from ..io.coordinates_writers import write_lammps_data
from .types import InteractionKey


# ─── Public dataclasses ───────────────────────────────────────────────


@dataclass(frozen=True)
class VPTopologyTemplate:
    """Static VP-grown topology.

    All index arrays use the *combined* (real + VP) atom ordering that
    the grown frame will expose. ``real_indices`` selects positions in
    AA-mapped-trajectory order; filling those rows with trajectory
    positions is the only real-side requirement for :meth:`VPGrower.grow_frame`.

    Attributes
    ----------
    atom_names
        Length-``n_total`` tuple of atom-type labels (resolved names,
        not LAMMPS ids). Real atoms keep their resolved name; VP atoms
        carry their ``VPAtomDef.type_label``.
    atom_masses
        Length-``n_total`` float array, Da.
    resids
        Length-``n_total`` int array, 1-based LAMMPS mol id.
    resnames
        Length-``n_total`` tuple of residue name strings.
    real_indices
        ``np.ndarray`` of int, the positions in the combined ordering
        that came from the AA-mapped trajectory. ``real_indices.size``
        equals the number of atoms in the AA universe.
    vp_indices_by_name
        ``{vp_name: np.ndarray[int]}``: one entry per VP atom type,
        listing its combined-ordering indices in carrier-residue order.
    carrier_resids
        Tuple of LAMMPS resids carrying any VP atom.
    type2id
        ``{atom_name: lammps_type_id}`` respecting
        ``vp_config.atomtype_order``.
    type_masses
        ``{atom_name: mass}`` derived from universe and ``[vp_atoms]``.
    bonds, angles, dihedrals
        ``(n, 2)`` / ``(n, 3)`` / ``(n, 4)`` int arrays of combined-ordering
        indices. Real-only entries appear first, then inserted VP entries.
    bond_type_ids, angle_type_ids, dihedral_type_ids
        1-based LAMMPS type ids aligned with ``bonds`` / ``angles`` /
        ``dihedrals`` rows.
    bond_type_key_by_id, angle_type_key_by_id, dihedral_type_key_by_id
        ``{lammps_type_id: InteractionKey}``. Useful for
        :class:`WriteLmpFF` shims.
    """

    atom_names: Tuple[str, ...]
    atom_masses: np.ndarray
    resids: np.ndarray
    resnames: Tuple[str, ...]
    real_indices: np.ndarray
    vp_indices_by_name: Dict[str, np.ndarray]
    carrier_resids: Tuple[int, ...]
    type2id: Dict[str, int]
    type_masses: Dict[str, float]
    bonds: np.ndarray
    angles: np.ndarray
    dihedrals: np.ndarray
    bond_type_ids: np.ndarray
    angle_type_ids: np.ndarray
    dihedral_type_ids: np.ndarray
    bond_type_key_by_id: Dict[int, InteractionKey]
    angle_type_key_by_id: Dict[int, InteractionKey]
    dihedral_type_key_by_id: Dict[int, InteractionKey]
    # Growth metadata (not written to .data): bond / angle definitions
    # resolved into name-level spec, needed by ``VPGrower.grow_frame``.
    # Dihedral specs are kept for FF-builder consumption; they do not
    # influence growth geometry.
    vp_bond_specs: Tuple["VPBondSpec", ...] = ()
    vp_angle_specs: Tuple["VPAngleSpec", ...] = ()
    vp_dihedral_specs: Tuple["VPDihedralSpec", ...] = ()
    clash_max_passes: int = 12
    clash_min_distance: float = 2.0

    @property
    def n_atoms(self) -> int:
        """Return total atoms in the grown topology."""
        return len(self.atom_names)

    @property
    def n_real(self) -> int:
        """Return the number of real, non-VP atoms."""
        return int(self.real_indices.size)

    @property
    def n_vp(self) -> int:
        """Return the number of inserted virtual-particle atoms."""
        return sum(int(arr.size) for arr in self.vp_indices_by_name.values())


@dataclass(frozen=True)
class VPBondSpec:
    """Resolved VP bond: carrier label ↔ VP label with harmonic params."""
    vp_name: str
    carrier_name: str
    k: float
    r0: float
    astable: bool = False


@dataclass(frozen=True)
class VPAngleSpec:
    """Resolved VP angle: three name-level labels + harmonic params.

    The angle is ``labels[0]-labels[1]-labels[2]`` with ``labels`` in
    order given. Exactly one of the three labels equals a VP name.
    """
    vp_name: str
    labels: Tuple[str, str, str]
    k: float
    theta0_deg: float
    astable: bool = False


@dataclass(frozen=True)
class VPDihedralSpec:
    """Resolved VP dihedral: four name-level labels + raw LAMMPS style/params.

    Exactly one of ``labels`` equals a VP name. ``pot_style`` carries
    the raw LAMMPS ``dihedral_style`` token (typically ``harmonic``);
    ``pot_kwargs`` carries its coefficients as authored in
    ``[vp_dihedrals]``. The grower itself never consumes these values
    geometrically; they flow through the template into the FF builder.
    """
    vp_name: str
    labels: Tuple[str, str, str, str]
    pot_style: str
    pot_kwargs: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VPGrownFrame:
    """Per-frame grown positions + box dimensions.

    ``positions`` is ``(template.n_atoms, 3)``. ``dimensions`` is an
    MDAnalysis-style ``(6,)`` ``[lx, ly, lz, α, β, γ]``.
    """
    positions: np.ndarray
    dimensions: np.ndarray


# ─── Grower ───────────────────────────────────────────────────────────


class VPGrower:
    """Stateless VP grower.

    Construct via :meth:`from_universe`; call :meth:`grow_frame` per
    trajectory frame. The grower itself holds only the immutable
    :class:`VPTopologyTemplate`.
    """

    def __init__(self, template: VPTopologyTemplate) -> None:
        self.template = template

    # ── factory ──────────────────────────────────────────────────
    @classmethod
    def from_universe(
        cls,
        universe: mda.Universe,
        vp_config: VPConfig,
        *,
        type_aliases: Optional[Mapping[int, str]] = None,
    ) -> "VPGrower":
        """Build a :class:`VPTopologyTemplate` from a CG-only universe.

        Parameters
        ----------
        universe
            MDAnalysis universe representing the AA-mapped CG topology.
            Must contain zero VP atoms.
        vp_config
            :class:`VPConfig` describing VP atoms / bonds / angles /
            pairs and static topology knobs (selection, atomtype_order,
            clash settings).
        type_aliases
            Optional ``{lammps_type_id: name}`` mapping used to resolve
            bead-name labels in ``[vp_bonds]`` / ``[vp_angles]`` type
            keys. When absent, ``[vp_*]`` keys must be bare stringified
            LAMMPS ids (``"1"``, ``"2"``, …).
        """
        name_resolver = _NameResolver.from_universe(universe, type_aliases)
        carrier_mask = _carrier_mask(universe, vp_config.selection)

        template = _build_template(
            universe=universe,
            vp_config=vp_config,
            carrier_mask=carrier_mask,
            name_resolver=name_resolver,
        )
        return cls(template)

    # ── per-frame growth ─────────────────────────────────────────
    def grow_frame(
        self,
        positions_real: np.ndarray,
        dimensions: np.ndarray,
        *,
        orientation_seed: int,
    ) -> VPGrownFrame:
        """Return a :class:`VPGrownFrame` from real-atom positions.

        Parameters
        ----------
        positions_real
            ``(template.n_real, 3)`` float array in AA-mapped-trajectory
            order (matches ``template.real_indices``).
        dimensions
            MDAnalysis-style ``(6,)`` box.
        orientation_seed
            Seed for the per-frame RNG used to pick VP orientations when
            the bond + (optional) angle system under-determines the
            placement.
        """
        template = self.template
        positions_real = np.asarray(positions_real, dtype=np.float64)
        if positions_real.shape != (template.n_real, 3):
            raise ValueError(
                f"positions_real shape {positions_real.shape} does not match "
                f"template real-bead count {(template.n_real, 3)}."
            )

        dimensions = np.asarray(dimensions, dtype=np.float64).reshape(-1)
        if dimensions.size == 3:
            dimensions = np.concatenate([dimensions, np.array([90.0, 90.0, 90.0])])
        elif dimensions.size < 6:
            raise ValueError(f"dimensions must have at least 3 entries, got {dimensions.size}")

        positions = np.zeros((template.n_atoms, 3), dtype=np.float64)
        positions[template.real_indices] = positions_real

        rng = np.random.default_rng(int(orientation_seed))
        bonds_by_vp = _group_bonds_by_vp(template)
        angles_by_vp = _group_angles_by_vp(template)

        # Build a resid → combined-index map for real atoms by atom name,
        # used for label-based anchor lookup.
        resid_real_indices = _resid_real_indices(template)

        for vp_name, vp_indices in template.vp_indices_by_name.items():
            bond_specs = bonds_by_vp.get(vp_name, ())
            angle_specs = angles_by_vp.get(vp_name, ())
            for local, vp_idx in enumerate(vp_indices):
                resid = int(template.resids[int(vp_idx)])
                positions[int(vp_idx)] = _seed_vp_position(
                    positions=positions,
                    template=template,
                    vp_name=vp_name,
                    resid=resid,
                    resid_real_indices=resid_real_indices,
                    bond_specs=bond_specs,
                    angle_specs=angle_specs,
                    rng=rng,
                )

        _resolve_clashes(
            positions=positions,
            template=template,
            dimensions=dimensions,
        )

        return VPGrownFrame(positions=positions, dimensions=dimensions)


# ─── LAMMPS data writer ──────────────────────────────────────────────


def write_vp_data(
    template: VPTopologyTemplate,
    frame: VPGrownFrame,
    path: str | Path,
    *,
    title: str = "VP grown topology",
    atom_style: str = "full",
) -> Path:
    """Write a grown frame as a LAMMPS data file.

    Delegates to :func:`AceCG.io.coordinates_writers.write_lammps_data`;
    this function just assembles the per-bead records it expects.
    """
    beads: List[Dict[str, object]] = []
    for i in range(template.n_atoms):
        name = template.atom_names[i]
        beads.append(
            {
                "resid": int(template.resids[i]),
                "resname": str(template.resnames[i]),
                "bead_type": str(name),
                "type": str(name),
                "q": 0.0,
            }
        )

    out_path = Path(path)
    write_lammps_data(
        path=out_path,
        title=title,
        coords_A=np.asarray(frame.positions, dtype=float),
        beads=beads,
        type2id=template.type2id,
        type_masses=template.type_masses,
        box_A=np.asarray(frame.dimensions, dtype=float),
        atom_style=atom_style,
        bonds=template.bonds,
        bond_type_ids=template.bond_type_ids,
        angles=template.angles,
        angle_type_ids=template.angle_type_ids,
        dihedrals=template.dihedrals,
        dihedral_type_ids=template.dihedral_type_ids,
    )
    return out_path


# ─── Internal: name resolution ───────────────────────────────────────


class _NameResolver:
    """Resolve ``[vp_*]`` type labels to per-atom names used in the template.

    ``resolve_label(label)`` maps a user-authored label (alias or bare
    id) to the canonical name used throughout the template. Raises if
    the label cannot be resolved.
    """

    def __init__(
        self,
        *,
        atom_names: np.ndarray,
        alias_to_name: Dict[str, str],
        bare_mode: bool,
    ) -> None:
        self.atom_names = atom_names
        self.alias_to_name = alias_to_name
        self.bare_mode = bare_mode

    @classmethod
    def from_universe(
        cls,
        universe: mda.Universe,
        type_aliases: Optional[Mapping[int, str]],
    ) -> "_NameResolver":
        # Use MDAnalysis atom.types for per-atom type codes; fall back
        # to name. For LAMMPS DATA universes without aliases, types are
        # stringified ids like "1", "2", ...
        types_attr = np.asarray(universe.atoms.types).astype(str)

        if type_aliases is None or not type_aliases:
            # Bare mode: atom name = LAMMPS type id (string), and
            # ``[vp_*]`` labels must also be bare ids.
            return cls(
                atom_names=types_attr.copy(),
                alias_to_name={},
                bare_mode=True,
            )

        alias_to_name: Dict[str, str] = {}
        atom_names = np.empty(len(types_attr), dtype=object)
        for i, tid_str in enumerate(types_attr):
            # Resolve each atom's name via the alias table.
            try:
                tid_int = int(tid_str)
            except ValueError:
                # universe already carries names; pass through.
                atom_names[i] = str(tid_str)
                alias_to_name[str(tid_str)] = str(tid_str)
                continue
            name = type_aliases.get(tid_int)
            if name is None:
                raise ValueError(
                    f"Atom {i} has LAMMPS type id {tid_int} but no alias "
                    f"was provided (known: {sorted(type_aliases)})."
                )
            atom_names[i] = str(name)
            alias_to_name[str(name)] = str(name)
            # Also allow looking up via the bare id.
            alias_to_name.setdefault(str(tid_int), str(name))

        return cls(
            atom_names=atom_names.astype(str),
            alias_to_name=alias_to_name,
            bare_mode=False,
        )

    def resolve_label(self, label: str) -> str:
        """Map a user-authored label to the canonical atom name.

        ``VP`` names (declared in ``[vp_atoms]``) are returned as-is;
        callers are expected to register VP names separately.
        """
        if self.bare_mode:
            return str(label)
        mapped = self.alias_to_name.get(str(label))
        if mapped is None:
            raise ValueError(f"cannot resolve VP label {label!r} via atom-type aliases")
        return mapped


def _carrier_mask(universe: mda.Universe, selection: Optional[str]) -> np.ndarray:
    """Boolean mask of length ``n_residues``: ``True`` ⇒ that residue carries a VP."""
    n_res = len(universe.residues)
    if n_res == 0:
        return np.zeros(0, dtype=bool)
    if not selection:
        return np.ones(n_res, dtype=bool)

    selected = universe.select_atoms(selection)
    if len(selected) == 0:
        raise ValueError(
            f"VP selection {selection!r} matched zero atoms in universe."
        )
    # Translate selected atom indices → residue positions in
    # ``universe.residues`` (which is what ``resindex`` already gives us).
    mask = np.zeros(n_res, dtype=bool)
    residue_indices = np.asarray(selected.atoms.resindices, dtype=np.int64)
    mask[np.unique(residue_indices)] = True
    return mask


# ─── Internal: template assembly ────────────────────────────────────


def _build_template(
    *,
    universe: mda.Universe,
    vp_config: VPConfig,
    carrier_mask: np.ndarray,
    name_resolver: _NameResolver,
) -> VPTopologyTemplate:
    real_names = list(name_resolver.atom_names.tolist())
    real_masses = np.asarray(universe.atoms.masses, dtype=float).copy()
    real_resids = np.asarray(universe.atoms.resids, dtype=np.int64).copy()
    residue_name_map = {
        int(res.resid): str(getattr(res, "resname", f"RES{int(res.resid)}"))
        for res in universe.residues
    }
    resindex_by_atom = np.asarray(universe.atoms.resindices, dtype=np.int64)

    # Build atom list by walking each residue and appending VP atoms
    # after the last real atom of every carrier residue. This mirrors
    # the AA-mapped trajectory order in ``real_indices``.
    raw_vp_names = [str(a.type_label) for a in vp_config.atoms]
    if len(raw_vp_names) != len(set(raw_vp_names)):
        raise ValueError("[vp_atoms] defines duplicate VP type_label entries.")
    overlap = sorted(set(real_names) & set(raw_vp_names))
    if overlap:
        raise ValueError(
            "[vp_atoms] type_label values must not collide with real atom names: "
            + ", ".join(overlap)
        )
    vp_masses = {a.type_label: float(a.mass) for a in vp_config.atoms}
    vp_names = list(vp_masses.keys())

    atom_names: List[str] = []
    atom_masses: List[float] = []
    resids: List[int] = []
    resnames: List[str] = []
    real_indices: List[int] = []
    vp_indices_by_name: Dict[str, List[int]] = {n: [] for n in vp_names}
    carrier_resids: List[int] = []

    atoms_by_residue = _group_atom_indices_by_residue(resindex_by_atom)
    n_res = len(universe.residues)

    for res_pos in range(n_res):
        res_resid = int(universe.residues[res_pos].resid)
        is_carrier = bool(carrier_mask[res_pos])
        for atom_idx in atoms_by_residue[res_pos]:
            real_indices.append(len(atom_names))
            atom_names.append(real_names[atom_idx])
            atom_masses.append(float(real_masses[atom_idx]))
            resids.append(int(real_resids[atom_idx]))
            resnames.append(residue_name_map[int(real_resids[atom_idx])])
        if is_carrier and vp_names:
            carrier_resids.append(res_resid)
            for vp_name in vp_names:
                vp_indices_by_name[vp_name].append(len(atom_names))
                atom_names.append(vp_name)
                atom_masses.append(vp_masses[vp_name])
                resids.append(res_resid)
                resnames.append(residue_name_map[res_resid])

    real_indices_arr = np.asarray(real_indices, dtype=np.int64)
    vp_indices_arr = {
        name: np.asarray(lst, dtype=np.int64) for name, lst in vp_indices_by_name.items()
    }

    # Type id map respecting atomtype_order.
    type2id = _build_type2id(
        real_names_ordered=_ordered_unique(real_names),
        vp_names_ordered=_ordered_unique(vp_names),
        atomtype_order=vp_config.atomtype_order,
    )
    type_masses: Dict[str, float] = {}
    for name, mass in zip(atom_names, atom_masses):
        type_masses.setdefault(name, float(mass))

    # Bonds: existing real-only bonds first.
    real_bonds = _extract_indices(universe, "bonds", 2)
    real_angles = _extract_indices(universe, "angles", 3)
    real_dihedrals = _extract_indices(universe, "dihedrals", 4)

    # Remap real-atom indices into combined ordering.
    real_to_combined = np.asarray(real_indices, dtype=np.int64)
    real_bonds_mapped = real_to_combined[real_bonds] if real_bonds.size else real_bonds
    real_angles_mapped = real_to_combined[real_angles] if real_angles.size else real_angles
    real_dihedrals_mapped = (
        real_to_combined[real_dihedrals] if real_dihedrals.size else real_dihedrals
    )

    # Resolve VP bond/angle/dihedral specs (name-level) from vp_config.
    vp_bond_specs, vp_angle_specs, vp_dihedral_specs = _resolve_vp_specs(
        vp_config, name_resolver, vp_names,
    )

    # Build per-residue inserted VP bonds / angles / dihedrals.
    inserted_bonds, inserted_angles, inserted_dihedrals = _build_inserted_bonded(
        vp_bond_specs=vp_bond_specs,
        vp_angle_specs=vp_angle_specs,
        vp_dihedral_specs=vp_dihedral_specs,
        carrier_resids=carrier_resids,
        atom_names=atom_names,
        resids=np.asarray(resids, dtype=np.int64),
        vp_indices_by_name=vp_indices_arr,
    )

    bonds = np.vstack([real_bonds_mapped, inserted_bonds]) if inserted_bonds.size else real_bonds_mapped
    angles = np.vstack([real_angles_mapped, inserted_angles]) if inserted_angles.size else real_angles_mapped
    dihedrals = (
        np.vstack([real_dihedrals_mapped, inserted_dihedrals])
        if inserted_dihedrals.size else real_dihedrals_mapped
    )
    if bonds.size == 0:
        bonds = np.empty((0, 2), dtype=np.int64)
    if angles.size == 0:
        angles = np.empty((0, 3), dtype=np.int64)
    if dihedrals.size == 0:
        dihedrals = np.empty((0, 4), dtype=np.int64)

    bond_type_ids, bond_type_key_by_id = _assign_bonded_type_ids(
        bonded=bonds,
        atom_names=atom_names,
        width=2,
    )
    angle_type_ids, angle_type_key_by_id = _assign_bonded_type_ids(
        bonded=angles,
        atom_names=atom_names,
        width=3,
    )
    dihedral_type_ids, dihedral_type_key_by_id = _assign_bonded_type_ids(
        bonded=dihedrals,
        atom_names=atom_names,
        width=4,
    )

    return VPTopologyTemplate(
        atom_names=tuple(atom_names),
        atom_masses=np.asarray(atom_masses, dtype=float),
        resids=np.asarray(resids, dtype=np.int64),
        resnames=tuple(resnames),
        real_indices=real_indices_arr,
        vp_indices_by_name=vp_indices_arr,
        carrier_resids=tuple(carrier_resids),
        type2id=type2id,
        type_masses=type_masses,
        bonds=bonds,
        angles=angles,
        dihedrals=dihedrals,
        bond_type_ids=bond_type_ids,
        angle_type_ids=angle_type_ids,
        dihedral_type_ids=dihedral_type_ids,
        bond_type_key_by_id=bond_type_key_by_id,
        angle_type_key_by_id=angle_type_key_by_id,
        dihedral_type_key_by_id=dihedral_type_key_by_id,
        vp_bond_specs=tuple(vp_bond_specs),
        vp_angle_specs=tuple(vp_angle_specs),
        vp_dihedral_specs=tuple(vp_dihedral_specs),
        clash_max_passes=int(vp_config.clash_max_passes),
        clash_min_distance=float(vp_config.clash_min_distance),
    )


def _group_atom_indices_by_residue(resindex_by_atom: np.ndarray) -> List[List[int]]:
    n_res = int(resindex_by_atom.max()) + 1 if resindex_by_atom.size else 0
    buckets: List[List[int]] = [[] for _ in range(n_res)]
    for atom_idx, res_pos in enumerate(resindex_by_atom.tolist()):
        buckets[int(res_pos)].append(int(atom_idx))
    return buckets


def _ordered_unique(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(str(item))
    return out


def _build_type2id(
    *,
    real_names_ordered: Sequence[str],
    vp_names_ordered: Sequence[str],
    atomtype_order: str,
) -> Dict[str, int]:
    if atomtype_order == "front":
        ordered = list(vp_names_ordered) + list(real_names_ordered)
    elif atomtype_order == "back":
        ordered = list(real_names_ordered) + list(vp_names_ordered)
    else:
        raise ValueError(f"atomtype_order must be 'front' or 'back', got {atomtype_order!r}.")
    return {name: i + 1 for i, name in enumerate(ordered)}


def _extract_indices(universe: mda.Universe, attr: str, width: int) -> np.ndarray:
    obj = getattr(universe, attr, None)
    if obj is None or len(obj) == 0:
        return np.empty((0, width), dtype=np.int64)
    arr = np.asarray(obj.indices, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0, width), dtype=np.int64)
    return arr.reshape(-1, width)


def _resolve_vp_specs(
    vp_config: VPConfig,
    name_resolver: _NameResolver,
    vp_names: Sequence[str],
) -> Tuple[List[VPBondSpec], List[VPAngleSpec], List[VPDihedralSpec]]:
    vp_name_set = set(vp_names)
    bond_specs: List[VPBondSpec] = []
    for bdef in vp_config.bonds:
        if bdef.pot_style != "harmonic":
            raise NotImplementedError(
                f"VP bond pot_style={bdef.pot_style!r} not supported (harmonic only)."
            )
        resolved = tuple(
            label if label in vp_name_set else name_resolver.resolve_label(label)
            for label in bdef.type_keys
        )
        vp_label = next((l for l in resolved if l in vp_name_set), None)
        if vp_label is None or len(resolved) != 2:
            raise ValueError(
                f"[vp_bonds] entry {bdef.type_keys!r} must contain exactly one VP name."
            )
        carrier_label = resolved[0] if resolved[1] == vp_label else resolved[1]
        k = float(bdef.pot_kwargs.get("k"))
        r0 = float(bdef.pot_kwargs.get("r0"))
        bond_specs.append(
            VPBondSpec(
                vp_name=vp_label,
                carrier_name=carrier_label,
                k=k,
                r0=r0,
                astable=_interaction_astable(bdef),
            )
        )

    angle_specs: List[VPAngleSpec] = []
    for adef in vp_config.angles:
        if adef.pot_style != "harmonic":
            raise NotImplementedError(
                f"VP angle pot_style={adef.pot_style!r} not supported (harmonic only)."
            )
        if len(adef.type_keys) != 3:
            raise ValueError(f"[vp_angles] entry {adef.type_keys!r} must have 3 labels.")
        resolved = tuple(
            label if label in vp_name_set else name_resolver.resolve_label(label)
            for label in adef.type_keys
        )
        if sum(1 for l in resolved if l in vp_name_set) != 1:
            raise ValueError(
                f"[vp_angles] entry {adef.type_keys!r} must contain exactly one VP name."
            )
        k = float(adef.pot_kwargs.get("k"))
        theta0 = float(adef.pot_kwargs.get("theta0"))
        vp_label = next(l for l in resolved if l in vp_name_set)
        angle_specs.append(VPAngleSpec(
            vp_name=vp_label,
            labels=(resolved[0], resolved[1], resolved[2]),
            k=k,
            theta0_deg=theta0,
            astable=_interaction_astable(adef),
        ))

    dihedral_specs: List[VPDihedralSpec] = []
    for ddef in vp_config.dihedrals:
        if len(ddef.type_keys) != 4:
            raise ValueError(
                f"[vp_dihedrals] entry {ddef.type_keys!r} must have 4 labels."
            )
        resolved = tuple(
            label if label in vp_name_set else name_resolver.resolve_label(label)
            for label in ddef.type_keys
        )
        if sum(1 for l in resolved if l in vp_name_set) != 1:
            raise ValueError(
                f"[vp_dihedrals] entry {ddef.type_keys!r} must contain exactly one VP name."
            )
        vp_label = next(l for l in resolved if l in vp_name_set)
        dihedral_specs.append(VPDihedralSpec(
            vp_name=vp_label,
            labels=(resolved[0], resolved[1], resolved[2], resolved[3]),
            pot_style=str(ddef.pot_style),
            pot_kwargs=dict(ddef.pot_kwargs),
        ))

    return bond_specs, angle_specs, dihedral_specs


def _interaction_astable(interaction: VPInteractionDef) -> bool:
    astable = interaction.pot_kwargs.get("astable")
    if astable is None:
        return False
    if isinstance(astable, bool):
        return astable
    if isinstance(astable, str):
        return astable.strip().lower() in {"yes", "true", "1", "on"}
    if isinstance(astable, (int, float)):
        return bool(astable)
    return False


def _build_inserted_bonded(
    *,
    vp_bond_specs: Sequence[VPBondSpec],
    vp_angle_specs: Sequence[VPAngleSpec],
    vp_dihedral_specs: Sequence[VPDihedralSpec],
    carrier_resids: Sequence[int],
    atom_names: Sequence[str],
    resids: np.ndarray,
    vp_indices_by_name: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct inserted bond / angle / dihedral index rows for every
    carrier resid."""
    # Per-resid name→atom-index lookup limited to real atoms.
    resid_to_atoms: Dict[int, Dict[str, int]] = {}
    for idx, (name, resid_val) in enumerate(zip(atom_names, resids.tolist())):
        if name in vp_indices_by_name:
            # skip VPs
            continue
        resid_to_atoms.setdefault(int(resid_val), {})[str(name)] = idx

    # Build a resid → {vp_name: idx} lookup from the pre-built vp_indices_by_name.
    vp_by_resid: Dict[int, Dict[str, int]] = {}
    for vp_name, idx_arr in vp_indices_by_name.items():
        for combined_idx in idx_arr.tolist():
            vp_by_resid.setdefault(int(resids[combined_idx]), {})[vp_name] = int(combined_idx)

    bond_rows: List[Tuple[int, int]] = []
    for spec in vp_bond_specs:
        for resid_val in carrier_resids:
            vp_idx = vp_by_resid.get(int(resid_val), {}).get(spec.vp_name)
            carrier_idx = resid_to_atoms.get(int(resid_val), {}).get(spec.carrier_name)
            if vp_idx is None or carrier_idx is None:
                raise ValueError(
                    f"Residue {resid_val}: cannot resolve bond {spec.vp_name}-{spec.carrier_name} "
                    f"(missing carrier atom or VP atom)."
                )
            lo, hi = sorted((int(vp_idx), int(carrier_idx)))
            bond_rows.append((lo, hi))

    angle_rows: List[Tuple[int, int, int]] = []
    for spec in vp_angle_specs:
        for resid_val in carrier_resids:
            row = _resolve_tuple_row(
                spec.labels, spec.vp_name, resid_val,
                resid_to_atoms, vp_by_resid, kind="angle",
            )
            if row is not None:
                angle_rows.append(tuple(row))  # type: ignore[arg-type]

    dihedral_rows: List[Tuple[int, int, int, int]] = []
    for spec in vp_dihedral_specs:
        for resid_val in carrier_resids:
            row = _resolve_tuple_row(
                spec.labels, spec.vp_name, resid_val,
                resid_to_atoms, vp_by_resid, kind="dihedral",
            )
            if row is not None:
                dihedral_rows.append(tuple(row))  # type: ignore[arg-type]

    bonds = np.asarray(bond_rows, dtype=np.int64) if bond_rows else np.empty((0, 2), dtype=np.int64)
    angles = np.asarray(angle_rows, dtype=np.int64) if angle_rows else np.empty((0, 3), dtype=np.int64)
    dihedrals = (
        np.asarray(dihedral_rows, dtype=np.int64) if dihedral_rows
        else np.empty((0, 4), dtype=np.int64)
    )
    return bonds, angles, dihedrals


def _resolve_tuple_row(
    labels: Sequence[str],
    vp_name: str,
    resid_val: int,
    resid_to_atoms: Mapping[int, Mapping[str, int]],
    vp_by_resid: Mapping[int, Mapping[str, int]],
    *,
    kind: str,
) -> Optional[Tuple[int, ...]]:
    """Resolve a name-tuple into a combined-index tuple for one residue.

    Used by both angle (3 labels) and dihedral (4 labels) insertion.
    Returns ``None`` if the VP atom is absent (non-carrier residue);
    raises :class:`ValueError` when a real carrier atom is missing.
    """
    real_atoms = resid_to_atoms.get(int(resid_val), {})
    vps = vp_by_resid.get(int(resid_val), {})
    indices: List[int] = []
    for label in labels:
        if label == vp_name:
            vp_idx = vps.get(label)
            if vp_idx is None:
                return None
            indices.append(int(vp_idx))
        else:
            carrier_idx = real_atoms.get(label)
            if carrier_idx is None:
                raise ValueError(
                    f"Residue {resid_val}: {kind} {tuple(labels)!r} "
                    f"references missing atom {label!r}."
                )
            indices.append(int(carrier_idx))
    return tuple(indices)


def _assign_bonded_type_ids(
    *,
    bonded: np.ndarray,
    atom_names: Sequence[str],
    width: int,
) -> Tuple[np.ndarray, Dict[int, InteractionKey]]:
    """Assign 1-based type ids to bonded rows based on their canonical key."""
    if bonded.size == 0:
        return np.empty(0, dtype=np.int64), {}

    key_to_id: Dict[InteractionKey, int] = {}
    id_to_key: Dict[int, InteractionKey] = {}
    type_ids = np.empty(bonded.shape[0], dtype=np.int64)
    for row_i, row in enumerate(bonded.tolist()):
        names = [str(atom_names[int(j)]) for j in row]
        if width == 2:
            key = InteractionKey.bond(*names)
        elif width == 3:
            key = InteractionKey.angle(*names)
        elif width == 4:
            key = InteractionKey.dihedral(*names)
        else:
            raise ValueError(f"Unsupported bonded width {width}")
        if key not in key_to_id:
            new_id = len(key_to_id) + 1
            key_to_id[key] = new_id
            id_to_key[new_id] = key
        type_ids[row_i] = key_to_id[key]
    return type_ids, id_to_key


# ─── Internal: per-frame seeding and clash resolution ───────────────


def _group_bonds_by_vp(template: VPTopologyTemplate) -> Dict[str, Tuple[VPBondSpec, ...]]:
    groups: Dict[str, List[VPBondSpec]] = {}
    for spec in template.vp_bond_specs:
        groups.setdefault(spec.vp_name, []).append(spec)
    return {name: tuple(lst) for name, lst in groups.items()}


def _group_angles_by_vp(template: VPTopologyTemplate) -> Dict[str, Tuple[VPAngleSpec, ...]]:
    groups: Dict[str, List[VPAngleSpec]] = {}
    for spec in template.vp_angle_specs:
        groups.setdefault(spec.vp_name, []).append(spec)
    return {name: tuple(lst) for name, lst in groups.items()}


def _resid_real_indices(template: VPTopologyTemplate) -> Dict[int, Dict[str, int]]:
    out: Dict[int, Dict[str, int]] = {}
    real_set = set(template.real_indices.tolist())
    for i, (name, resid) in enumerate(zip(template.atom_names, template.resids.tolist())):
        if i not in real_set:
            continue
        out.setdefault(int(resid), {})[str(name)] = int(i)
    return out


def _seed_vp_position(
    *,
    positions: np.ndarray,
    template: VPTopologyTemplate,
    vp_name: str,
    resid: int,
    resid_real_indices: Mapping[int, Mapping[str, int]],
    bond_specs: Sequence[VPBondSpec],
    angle_specs: Sequence[VPAngleSpec],
    rng: np.random.Generator,
) -> np.ndarray:
    """Produce an initial VP position honoring the first bond + angle."""
    real_atoms = resid_real_indices.get(int(resid), {})
    if not real_atoms:
        # Degenerate case: carrier residue with zero real atoms.
        return np.zeros(3, dtype=np.float64)

    residue_positions = np.asarray(
        [positions[i] for i in real_atoms.values()], dtype=np.float64
    )
    center = residue_positions.mean(axis=0)

    if not bond_specs:
        # No bond anchor → place at residue center.
        return center

    primary = bond_specs[0]
    anchor_idx = real_atoms.get(primary.carrier_name)
    if anchor_idx is None:
        raise ValueError(
            f"Residue {resid}: missing carrier atom {primary.carrier_name!r} for VP {vp_name}."
        )
    anchor = np.asarray(positions[anchor_idx], dtype=np.float64)
    bond_length = float(primary.r0)
    direction = _unit(anchor - center)
    if np.linalg.norm(direction) < 1e-12:
        direction = _unit(rng.normal(size=3))

    # Try to honor the first angle spec that references this VP + carrier.
    for angle in angle_specs:
        labels = list(angle.labels)
        if labels.count(vp_name) != 1:
            continue
        vp_pos_in_angle = labels.index(vp_name)
        if vp_pos_in_angle not in (0, 2):
            # VP must be an endpoint of the angle for the analytic scheme.
            continue
        center_label = labels[1]
        if center_label != primary.carrier_name:
            continue
        reference_label = labels[2] if vp_pos_in_angle == 0 else labels[0]
        reference_idx = real_atoms.get(reference_label)
        if reference_idx is None:
            continue
        reference_axis = _unit(np.asarray(positions[reference_idx], dtype=np.float64) - anchor)
        if np.linalg.norm(reference_axis) < 1e-12:
            continue
        basis_1, basis_2 = _orthonormal_basis(reference_axis)
        angle_rad = float(np.deg2rad(angle.theta0_deg))
        azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
        perpendicular = np.cos(azimuth) * basis_1 + np.sin(azimuth) * basis_2
        if vp_pos_in_angle == 0:
            direction = np.cos(angle_rad) * reference_axis + np.sin(angle_rad) * perpendicular
        else:
            direction = -np.cos(angle_rad) * reference_axis + np.sin(angle_rad) * perpendicular
        direction = _unit(direction)
        break

    if np.linalg.norm(direction) < 1e-12:
        direction = _unit(rng.normal(size=3))
    return anchor + bond_length * direction


def _resolve_clashes(
    *,
    positions: np.ndarray,
    template: VPTopologyTemplate,
    dimensions: np.ndarray,
) -> None:
    """Push VP atoms away from real atoms and from each other until
    inter-atom distance exceeds ``template.clash_min_distance``.

    Implements the same approach as the legacy grower: iterate up to
    ``clash_max_passes``, on each pass shift VP positions by a weighted
    combination of inverse-distance vectors against overlapping
    neighbours. Terminates early when no overlap remains.
    """
    max_passes = int(template.clash_max_passes)
    if max_passes < 1:
        return
    threshold = float(template.clash_min_distance)
    pad = 1e-3

    vp_idx_list: List[int] = []
    for arr in template.vp_indices_by_name.values():
        vp_idx_list.extend(int(i) for i in arr.tolist())
    vp_indices = np.asarray(vp_idx_list, dtype=np.int64)
    if vp_indices.size == 0:
        return

    real_indices = np.asarray(template.real_indices, dtype=np.int64)

    box = np.asarray(dimensions, dtype=float)
    distance_box = (
        box
        if box.size >= 3 and np.all(np.isfinite(box[:3])) and np.all(box[:3] > 0.0)
        else None
    )

    # Remember anchor direction for each VP, used as a fallback push
    # direction when the vector from a clashing neighbour is ill-defined.
    anchor_direction = _per_vp_anchor_direction(positions, template, vp_indices)

    for _ in range(max_passes):
        changed = False
        # (a) VP ↔ real clashes.
        if real_indices.size > 0:
            real_positions = positions[real_indices]
            pairs, distances = capped_distance(
                positions[vp_indices],
                real_positions,
                max_cutoff=threshold,
                box=distance_box,
                return_distances=True,
            )
            if pairs.size > 0:
                deltas = minimize_vectors(
                    positions[vp_indices[pairs[:, 0]]] - real_positions[pairs[:, 1]],
                    box=box,
                )
                for local_vp in np.unique(pairs[:, 0]):
                    mask = pairs[:, 0] == local_vp
                    local_deltas = deltas[mask]
                    local_distances = np.asarray(distances[mask], dtype=float)
                    if not np.any(local_distances < threshold):
                        continue
                    push = anchor_direction[int(local_vp)]
                    directions = np.divide(
                        local_deltas,
                        local_distances[:, None],
                        out=np.tile(push, (local_deltas.shape[0], 1)),
                        where=local_distances[:, None] > 1e-12,
                    )
                    shifts = (threshold - local_distances + pad)[:, None] * directions
                    vp_idx = int(vp_indices[int(local_vp)])
                    positions[vp_idx] = positions[vp_idx] + shifts.sum(axis=0)
                    changed = True

        # (b) VP ↔ VP clashes.
        if vp_indices.size > 1:
            vp_positions = positions[vp_indices]
            pairs, distances = self_capped_distance(
                vp_positions,
                max_cutoff=threshold,
                box=distance_box,
                return_distances=True,
            )
            if distances.size > 0:
                a_idx = np.asarray(pairs[:, 0], dtype=np.int64)
                b_idx = np.asarray(pairs[:, 1], dtype=np.int64)
                deltas = minimize_vectors(vp_positions[a_idx] - vp_positions[b_idx], box=box)
                directions = np.divide(
                    deltas,
                    distances[:, None],
                    out=np.tile(np.array([1.0, 0.0, 0.0]), (deltas.shape[0], 1)),
                    where=distances[:, None] > 1e-12,
                )
                shifts = 0.5 * (threshold - distances + pad)[:, None] * directions
                accumulated = np.zeros_like(vp_positions)
                np.add.at(accumulated, a_idx, shifts)
                np.add.at(accumulated, b_idx, -shifts)
                positions[vp_indices] = vp_positions + accumulated
                changed = True

        if not changed:
            break


def _per_vp_anchor_direction(
    positions: np.ndarray,
    template: VPTopologyTemplate,
    vp_indices: np.ndarray,
) -> np.ndarray:
    """For each VP atom, compute a unit vector from its bonded real
    neighbour(s) toward the VP — used as a fallback push direction."""
    # Gather VP → real-neighbour list from ``template.bonds``.
    bonds = template.bonds
    vp_set = set(int(i) for i in vp_indices.tolist())
    neighbours: Dict[int, List[int]] = {int(v): [] for v in vp_indices.tolist()}
    for row in bonds.tolist():
        i, j = int(row[0]), int(row[1])
        if i in vp_set and j not in vp_set:
            neighbours[i].append(j)
        elif j in vp_set and i not in vp_set:
            neighbours[j].append(i)

    out = np.zeros((vp_indices.size, 3), dtype=np.float64)
    for local, vp_idx in enumerate(vp_indices.tolist()):
        partners = neighbours.get(int(vp_idx), [])
        if not partners:
            out[local] = np.array([1.0, 0.0, 0.0])
            continue
        anchor_positions = positions[np.asarray(partners, dtype=np.int64)]
        direction = positions[int(vp_idx)] - anchor_positions.mean(axis=0)
        norm = float(np.linalg.norm(direction))
        out[local] = direction / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    return out


# ─── Tiny vector helpers ────────────────────────────────────────────


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(vec, dtype=np.float64) / norm


def _orthonormal_basis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axis = _unit(axis)
    if np.linalg.norm(axis) < 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    probe = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(axis, probe))) > 0.9:
        probe = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    b1 = _unit(np.cross(axis, probe))
    b2 = _unit(np.cross(axis, b1))
    return b1, b2
