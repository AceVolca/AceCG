"""Topology serialization helpers for MDAnalysis universes."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, TypedDict, List

from dataclasses import dataclass
import numpy as np
from MDAnalysis import Universe

from .types import InteractionKey


def _encode_pairs(pairs: np.ndarray, n_atoms: int) -> np.ndarray:
    """Encode unordered atom-index pairs into stable integer ids."""
    if pairs.size == 0:
        return np.empty(0, dtype=np.int32)
    ordered = np.sort(np.asarray(pairs, dtype=np.int32), axis=1)
    return ordered[:, 0] * np.int32(n_atoms) + ordered[:, 1]


def _group_exclusion_ids(labels: np.ndarray, n_atoms: int) -> tuple[np.ndarray, bool]:
    """Return encoded same-label exclusion ids and an all-excluded sentinel."""
    if labels.size == 0:
        return np.empty(0, dtype=np.int32), False
    order = np.argsort(labels, kind="stable")
    labels_sorted = labels[order]
    split_points = np.flatnonzero(np.diff(labels_sorted)) + 1
    groups = np.split(order.astype(np.int32, copy=False), split_points)
    if len(groups) == 1 and groups[0].size == n_atoms:
        return np.empty(0, dtype=np.int32), True
    parts: list[np.ndarray] = []
    for group in groups:
        group = np.asarray(group, dtype=np.int32)
        if group.size < 2:
            continue
        row, col = np.triu_indices(group.size, k=1)
        parts.append(_encode_pairs(np.column_stack((group[row], group[col])), n_atoms))
    if not parts:
        return np.empty(0, dtype=np.int32), False
    return np.unique(np.concatenate(parts)), False


@dataclass(frozen=True)
class TopologyArrays:
    """Immutable topology out. Built once from MDAnalysis, broadcast to MPI workers.

    All fields are required. Fields that are absent in the source Universe are set to None.
    Compute code accesses fields by attribute, never by string key.
    """

    # ── atom-level ──────────────────────────────────────────────
    # total number of atoms (real + virtual)
    n_atoms: int
    # (n_atoms,) object/str — per-atom name
    names: np.ndarray
    # (n_atoms,) object/str — per-atom type name
    types: np.ndarray
    # (n_unique_types,) object/str — ordered unique type names
    atom_type_names: np.ndarray
    # (n_atoms,) int32 — per-atom type code
    atom_type_codes: np.ndarray
    
    # ── residue / molecule metadata ─────────────────────────────
    n_residues: int
    atom_resindex: np.ndarray
    masses: np.ndarray
    charges: np.ndarray
    resids: np.ndarray
    molnums: np.ndarray

    # ── bonded term instances ───────────────────────────────────
    bonds: np.ndarray
    angles: np.ndarray
    dihedrals: np.ndarray
    
    # ── exclusions (for neighbor list construction) ─────────────
    exclude_12: np.ndarray
    exclude_13: np.ndarray
    exclude_14: np.ndarray
    excluded_nb: np.ndarray
    excluded_nb_mode: str
    excluded_nb_all: bool
    
    # ── site classification ─────────────────────────────────────
    real_site_indices: np.ndarray
    virtual_site_mask: np.ndarray
    virtual_site_indices: np.ndarray

    # ── instance → canonical key index ──────────────────────────
    # (n_bond,) int32
    bond_key_index: np.ndarray
    # (n_angle,) int32
    angle_key_index: np.ndarray
    # (n_dihedral,) int32
    dihedral_key_index: np.ndarray

    # ── canonical key tables ────────────────────────────────────
    keys_bondtypes: List[InteractionKey]
    keys_angletypes: List[InteractionKey]
    keys_dihedraltypes: List[InteractionKey]
    
    # ── type translators ───────────────────────────────────────
    atom_type_name_to_code: dict[str, int]
    atom_type_code_to_name: dict[int, str]
    bond_type_id_to_key: dict[int, InteractionKey]
    angle_type_id_to_key: dict[int, InteractionKey]
    dihedral_type_id_to_key: dict[int, InteractionKey]
    key_to_bonded_type_id: dict[InteractionKey, int]
    

def collect_topology_arrays(
    u: Universe,
    exclude_bonded: str = "111",
    exclude_option: str = "resid",
    atom_type_name_aliases: Optional[Mapping[int, str]] = None,
    vp_names: Optional[Iterable[str]] = None,
) -> TopologyArrays:
    """Serialize topology data from an MDAnalysis universe.

    Parameters
    ----------
    u : MDAnalysis.Universe
        Universe containing atom metadata and optional bonded topology.
    exclude_bonded : str, default="111"
        Three-character string of ``"1"``/``"0"`` flags controlling whether
        1-2, 1-3, and 1-4 bonded exclusions are included.
    exclude_option : {"resid", "molid", "none"}, default="resid"
        Nonbonded exclusion mode cached for neighbor-list construction.
    atom_type_name_aliases : Mapping[int, str], optional
        Mapping from LAMMPS atom type id to canonical type name. This is needed
        when the source universe lacks explicit atom names.
    vp_names : Iterable[str], optional
        Type names treated as virtual-particle sites.

    Returns
    -------
    TopologyArrays
        Immutable collection of atom arrays, bonded instance arrays, exclusion
        ids, type translators, and canonical bonded interaction keys.
    """
    out = {}
    
    # Sanity checks
    exclude_bonded = str(exclude_bonded)
    assert len(exclude_bonded) == 3 and all(c in "01" for c in exclude_bonded), \
        f"exclude_bonded must be a 3-character string of '0' / '1', got '{exclude_bonded}'"
    exclude_option = str(exclude_option)
    if exclude_option not in {"resid", "molid", "none"}:
        raise ValueError(f"Invalid exclude_option: {exclude_option}")

    if atom_type_name_aliases is not None:
        normalized_aliases = {}
        alias_name_to_type: dict[str, int] = {}
        for raw_type, raw_name in atom_type_name_aliases.items():
            try:
                type_id = int(raw_type)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "atom_type_name_aliases keys must be integer-like"
                ) from exc
            name = str(raw_name)
            previous = normalized_aliases.get(type_id)
            if previous is not None and previous != name:
                raise ValueError(
                    f"Conflicting aliases for atom type {type_id}: {previous!r} vs {name!r}"
                )
            existing_type_id = alias_name_to_type.get(name)
            if existing_type_id is not None and existing_type_id != type_id:
                raise ValueError(
                    "atom_type_name_aliases must map each atom type id to a unique canonical name"
                )
            normalized_aliases[type_id] = name
            alias_name_to_type[name] = type_id
        atom_type_name_aliases = normalized_aliases
    
    # Mandatory atom-level attributes
    for attr in ("types", "masses", "charges", "resids"):
        assert hasattr(u.atoms, attr), f"Universe.atoms missing expected attribute '{attr}'"
    
    # For LAMMPS data file, atom names are not usually defined in MDAnalysis universe.
    # In this case, we use atom_type_name_alises if provided.
    out["types"] = np.asarray(u.atoms.types)


    _from_LAMMPS = True
    _has_names = True
    # case 1
    if hasattr(u.atoms, "names"):
        # PDB, PSF, GRO, etc.
        out["names"] = np.asarray(u.atoms.names)
        _from_LAMMPS = False
    # case 2
    elif atom_type_name_aliases is not None:
        # LAMMPS data: use alias to map type code to name.
        # Notice that bonds types are defined by atom names in LAMMPS.
        # Will construct InteractionKeys based on type aliases.
        int_types = u.atoms.types.astype(np.int32)
        out["names"] = np.asarray(
            [str(atom_type_name_aliases.get(t, t)) for t in int_types],
            dtype=object,
        )
        u.add_TopologyAttr("names", out["names"])
        # In this path, the universe will still have names
    # case 3
    else:
        # LAMMPS data: no names, no alias.
        # Use types as names.
        # Later, will construct InteractionKeys based on LAMMPS type codes.
        out["names"] = np.asarray([str(t) for t in u.atoms.types], dtype=object)
        u.add_TopologyAttr("names", out["names"])
        # In this case, the universe "names" are actually numerical types
        # And they shall be sorted as integers rather than strings down below.
        _has_names = False

    # For LAMMPS 
    out["masses"] = np.asarray(u.atoms.masses, dtype=np.float32)
    out["charges"] = np.asarray(u.atoms.charges, dtype=np.float32)
    
    # Pair-type translation must be self-consistent across all fields.
    #
    # Case 1: canonical topology files (PDB/PSF/GRO/...) with real names
    #   Build a compact internal code space from unique names.
    # Case 2: LAMMPS + alias
    #   Keep raw LAMMPS atom-type ids as codes, and map alias names <-> raw ids.
    # Case 3: LAMMPS + no alias
    #   Keep raw LAMMPS atom-type ids as codes, and use their string form as names.
    # case 1
    if _has_names and not _from_LAMMPS:
        type_names = np.asarray(out["types"], dtype=object).astype(str)
        atom_type_names, atom_type_codes = np.unique(type_names, return_inverse=True)

        out["atom_type_names"] = atom_type_names
        out["atom_type_codes"] = atom_type_codes.astype(np.int32) + 1
        out["atom_type_name_to_code"] = {
            str(name): int(code) + 1 for code, name in enumerate(atom_type_names)
        }
        out["atom_type_code_to_name"] = {
            int(code) + 1: str(name) for code, name in enumerate(atom_type_names)
        }
    # case 2
    elif _has_names:
        raw_type_codes = np.asarray(u.atoms.types, dtype=np.int32)
        unique_raw_types = np.unique(raw_type_codes)
        atom_type_names = np.asarray(
            [str(atom_type_name_aliases.get(int(type_id), int(type_id))) for type_id in unique_raw_types],
            dtype=object,
        )

        out["atom_type_names"] = atom_type_names
        out["atom_type_codes"] = raw_type_codes
        out["atom_type_name_to_code"] = {
            str(name): int(type_id) for type_id, name in zip(unique_raw_types, atom_type_names)
        }
        out["atom_type_code_to_name"] = {
            int(type_id): str(name) for type_id, name in zip(unique_raw_types, atom_type_names)
        }
    # case 3
    else:
        raw_type_codes = np.asarray(u.atoms.types, dtype=np.int32)
        unique_raw_types = np.unique(raw_type_codes)
        atom_type_names = np.asarray([str(int(type_id)) for type_id in unique_raw_types], dtype=object)

        out["atom_type_names"] = atom_type_names
        out["atom_type_codes"] = raw_type_codes
        out["atom_type_name_to_code"] = {
            str(int(type_id)): int(type_id) for type_id in unique_raw_types
        }
        out["atom_type_code_to_name"] = {
            int(type_id): str(int(type_id)) for type_id in unique_raw_types
        }
    
    
    # Residue / molecule metadata
    out["n_atoms"] = len(u.atoms)
    out["n_residues"] = len(u.residues)
    out["atom_resindex"] = np.asarray(u.atoms.resindices, dtype=np.int32)
    out["resids"] = np.asarray(u.residues.resids, dtype=np.int32)

    if hasattr(u.atoms, "molnums"):
        out["molnums"] = np.asarray(u.atoms.molnums, dtype=np.int32)
    else:
        out["molnums"] = np.zeros(out["n_atoms"], dtype=np.int32)

    # Basic Bonded list & excluded list construction
    bonded_types = ("bonds", "angles", "dihedrals")
    bonded_type_singles = ("bond", "angle", "dihedral")
    exclude_attrs = ("exclude_12", "exclude_13", "exclude_14")
    
    names_arr = out["names"] if _from_LAMMPS else out["types"]
    # If the topology is from LAMMPS, we construct InteractionKeys based on atom names (which are mapped from type codes via alias if provided).
    # Otherwise, we construct InteractionKeys based on atom types.
    # For more "canonical" topology files, such as PSF, TPR, etc.
    # They have BOTH atom names AND types defined.
    # For PDB and GRO files, atom types are usually still NOT GUARANTEED to be the same as names.

    # Loop over bonds, angles, dihedrals
    for i in range(3):
        attr = bonded_types[i]
        exclude_attr = exclude_attrs[i]
        attr1 = bonded_type_singles[i]
        width = i + 2
        
        if not hasattr(u, attr):
            # bonded list
            out[attr] = np.empty((0, width), dtype=np.int32)
            # exclusion
            out[exclude_attr] = np.empty((0, 2), dtype=np.int32)
            # bonded key index
            out[f"{attr1}_key_index"] = np.empty(0, dtype=np.int32)
            # interaction keys
            out[f"keys_{attr1}types"] = []
            continue
            
        else: 
            obj = getattr(u, attr)
            attr_indices = np.asarray(obj.indices, dtype=np.int32)
            ikey_builder = getattr(InteractionKey, attr1)
            unique_term_types = sorted(
                obj.types(),
                key=lambda tokens: tuple(str(token) for token in tokens),
            )
            key_tokens = np.asarray(names_arr[attr_indices], dtype=object)
            if width == 2:
                swap_mask = key_tokens[:, 0] > key_tokens[:, 1]
            elif width == 3:
                swap_mask = key_tokens[:, 0] > key_tokens[:, 2]
            else:
                swap_mask = (key_tokens[:, 0] > key_tokens[:, 3]) | (
                    (key_tokens[:, 0] == key_tokens[:, 3]) & (key_tokens[:, 1] > key_tokens[:, 2])
                )
            canonical_tokens = key_tokens.copy()
            if np.any(swap_mask):
                canonical_tokens[swap_mask] = canonical_tokens[swap_mask][:, ::-1]
            key_labels = canonical_tokens[:, 0].astype(str)
            for col in range(1, width):
                key_labels = np.char.add(np.char.add(key_labels, ":"), canonical_tokens[:, col].astype(str))
            key_labels = np.char.add(f"{attr1}:", key_labels)
            
            if _from_LAMMPS:
                # attr type IDs in types are numeric.
                # The attr_types are 1-indexed integers.
                # Directly convert these to integers (0-indexed) when building the maps.
                # use u.bonds.atoms.types to get atom type indices corresponding to this bond for building InteractionKeys.
                raw_type_ids = np.asarray(getattr(obj, "_bondtypes", ()), dtype=object)
                type_ids = None
                if raw_type_ids.shape == (attr_indices.shape[0],):
                    try:
                        type_ids = raw_type_ids.astype(np.int32)
                    except (TypeError, ValueError):
                        pass

                if type_ids is None:
                    unique_labels = np.unique(key_labels)
                    key_index = np.searchsorted(unique_labels, key_labels)
                else:
                    unique_type_ids = sorted(int(type_id) for type_id in obj.types())
                    expected_type_ids = list(range(1, len(unique_type_ids) + 1))
                    if unique_type_ids != expected_type_ids:
                        raise ValueError(
                            f"{attr1} type ids must be contiguous 1-based integers, got {unique_type_ids!r}"
                        )
                    labels_by_type = [np.unique(key_labels[type_ids == type_id]) for type_id in unique_type_ids]
                    if any(labels.size != 1 for labels in labels_by_type):
                        raise ValueError(
                            f"{attr1} type ids must map one-to-one to canonical keys."
                        )
                    unique_labels = np.asarray([str(labels[0]) for labels in labels_by_type], dtype=str)
                    key_index = type_ids - 1
            
            else: 
                # attr type IDs in types are already tuples of atom types
                # Directly use that for building interaction keys.
                # maps shall be lazily built by sorting all unique attr types and using the sorted list to assign type IDs (although this is nolonger necessary.)
                unique_labels = np.asarray(
                    [
                        ikey_builder(*(str(token) for token in tokens)).label()
                        for tokens in unique_term_types
                    ],
                    dtype=str,
                )
                key_index = np.searchsorted(unique_labels, key_labels)
            keys_list = [InteractionKey.from_label(label) for label in unique_labels.tolist()]
                
            excl_list = attr_indices[:, [0, -1]]
            out[attr] = attr_indices
            if exclude_bonded[i] == "1":
                out[exclude_attr] = excl_list.astype(np.int32)
            else:
                out[exclude_attr] = np.empty((0, 2), dtype=np.int32)
            out[f"{attr1}_key_index"] = np.asarray(key_index, dtype=np.int32)
            out[f"keys_{attr1}types"] = list(keys_list)

    # Bonded type ids are 0-indexed internally.
            
    # Type id <-> key mappings
    out["bond_type_id_to_key"] = {}
    out["angle_type_id_to_key"] = {}
    out["dihedral_type_id_to_key"] = {}
    out["key_to_bonded_type_id"] = {}
    for i, k in enumerate(out["keys_bondtypes"]):
        out["bond_type_id_to_key"][i] = k
        out["key_to_bonded_type_id"][k] = i
    for i, k in enumerate(out["keys_angletypes"]):
        out["angle_type_id_to_key"][i] = k
        out["key_to_bonded_type_id"][k] = i
    for i, k in enumerate(out["keys_dihedraltypes"]):
        out["dihedral_type_id_to_key"][i] = k
        out["key_to_bonded_type_id"][k] = i

    # AGENT: VP site classification
    n = u.atoms.n_atoms
    if vp_names:
        vp_set = set(vp_names)
        vp_labels = np.asarray(out["names"], dtype=object).reshape(-1)
        vp_mask = np.array([str(label) in vp_set for label in vp_labels], dtype=bool)
    else:
        vp_mask = np.zeros(n, dtype=bool)
    all_idx = np.arange(n, dtype=np.int32)
    out["virtual_site_mask"] = vp_mask
    out["real_site_indices"] = all_idx[~vp_mask]
    out["virtual_site_indices"] = all_idx[vp_mask]

    bonded_parts = [
        _encode_pairs(out[name], out["n_atoms"])
        for name in exclude_attrs
        if out[name].size > 0
    ]
    bonded_ids = (
        np.unique(np.concatenate(bonded_parts))
        if bonded_parts
        else np.empty(0, dtype=np.int32)
    )
    excluded_nb_all = False
    nonbonded_ids = np.empty(0, dtype=np.int32)
    if exclude_option == "resid":
        atom_resids = out["resids"][out["atom_resindex"]]
        nonbonded_ids, excluded_nb_all = _group_exclusion_ids(atom_resids, out["n_atoms"])
    elif exclude_option == "molid":
        nonbonded_ids, excluded_nb_all = _group_exclusion_ids(out["molnums"], out["n_atoms"])

    if excluded_nb_all:
        out["excluded_nb"] = np.empty(0, dtype=np.int32)
    elif bonded_ids.size and nonbonded_ids.size:
        out["excluded_nb"] = np.unique(np.concatenate((bonded_ids, nonbonded_ids)))
    elif bonded_ids.size:
        out["excluded_nb"] = bonded_ids
    else:
        out["excluded_nb"] = nonbonded_ids
    out["excluded_nb_mode"] = exclude_option
    out["excluded_nb_all"] = bool(excluded_nb_all)

    return TopologyArrays(**out)
