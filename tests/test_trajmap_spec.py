"""Pin the mapping-YAML → :class:`CGMapSpec` compilation.

The expansion arithmetic is where OpenMSCG's ``cgmap`` has its worst latent bug
(a double-counted anchor in the recursive ``groups`` branch) and where real
archive files hit an outright ``KeyError``. These tests nail down our semantics
against both the reference Python expansion and a verbatim reimplementation of
OpenMSCG's own algorithm.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from AceCG.topology.cgmap import (
    CGMapSpec,
    expand_mapping_sites,
    load_mapping_yaml,
)
from AceCG.topology.types import InteractionKey
from real_frames import DPPC_MARTINI12

# Shipped fixture (tests/test_data/dppc_martini12/, see its README.md).
ARCHIVE_MAP = DPPC_MARTINI12 / "map.yaml"


# ─── reference implementations ────────────────────────────────────────


def openmscg_unpack_group(group, root_anchor=0):
    """Verbatim transcription of ``mscg/mapper.py:unpack_group`` (v0.9.0).

    Kept here so the tests can demonstrate exactly what we changed and why,
    without importing OpenMSCG (whose ``mscg.core`` C extension is not built in
    this environment).
    """
    unit_sites = []
    if "groups" in group:
        for item in group["groups"]:
            for site in openmscg_unpack_group(item, root_anchor + group["anchor"]):
                unit_sites.append(site[:])
    if "sites" in group:
        for site in group["sites"]:
            unit_sites.append(site[:])
    sites = []
    for i in range(group["repeat"]):
        offset = root_anchor + group["anchor"] + i * group["offset"]
        for site in unit_sites:
            sites.append([site[0], site[1] + offset])
    return sites


def slow_map_positions(mapping, positions, *, index_base=0):
    """Site positions by an explicit per-site Python loop, no PBC.

    The deliberately naive reference the vectorized CSR path is checked against.
    """
    sites, _, _ = expand_mapping_sites(mapping)
    shift = 0 if index_base == 0 else -1
    out = np.empty((len(sites), 3), dtype=np.float64)
    for row, (type_key, anchor) in enumerate(sites):
        entry = mapping["site-types"][type_key]
        idx = np.asarray(entry["index"], dtype=np.int64)
        abs_idx = (anchor + shift) + (idx + shift)
        weight = np.asarray(entry.get("x-weight", np.ones(idx.size)), dtype=np.float64)
        weight = weight / weight.sum()
        out[row] = (positions[abs_idx] * weight[:, None]).sum(axis=0)
    return out


def absolute_site_indices(spec, site):
    """Absolute AA indices contributing to ``site``, in mapping order."""
    lo = int(spec.csr_indptr[site])
    hi = int(spec.csr_indptr[site + 1])
    return spec.atom_indices[spec.csr_cols[lo:hi]]


# ─── fixtures ─────────────────────────────────────────────────────────


def simple_mapping(**overrides):
    mapping = {
        "site-types": {
            "A": {"index": [0, 1, 2], "x-weight": [12.0, 1.0, 1.0], "f-weight": [1.0, 1.0, 1.0]},
            "B": {"index": [0, 1], "x-weight": [16.0, 1.0], "f-weight": [1.0, 1.0]},
        },
        "system": [
            {"anchor": 0, "repeat": 3, "offset": 5, "sites": [["A", 0], ["B", 3]]},
        ],
    }
    mapping.update(overrides)
    return mapping


# ─── expansion semantics ──────────────────────────────────────────────


def test_expansion_order_is_group_repeat_site():
    mapping = simple_mapping()
    sites, group_ids, mol_ids = expand_mapping_sites(mapping)
    assert sites == [
        ("A", 0), ("B", 3),
        ("A", 5), ("B", 8),
        ("A", 10), ("B", 13),
    ]
    assert group_ids == [0] * 6
    assert mol_ids == [0, 0, 1, 1, 2, 2]


def test_multiple_groups_keep_group_major_order_and_distinct_molecules():
    mapping = simple_mapping(
        system=[
            {"anchor": 0, "repeat": 2, "offset": 5, "sites": [["A", 0]]},
            {"anchor": 100, "repeat": 2, "offset": 5, "sites": [["B", 3]]},
        ]
    )
    sites, group_ids, mol_ids = expand_mapping_sites(mapping)
    assert sites == [("A", 0), ("A", 5), ("B", 103), ("B", 108)]
    assert group_ids == [0, 0, 1, 1]
    assert mol_ids == [0, 1, 2, 3]

    spec = CGMapSpec.from_mapping(mapping)
    assert spec.n_mol == 4
    assert np.array_equal(spec.site_group_ids, [0, 0, 1, 1])
    # Non-contiguous groups: molecule 2 starts at absolute index 103, so the
    # compact buffer is still ascending and no gather is needed.
    assert spec.molecules_contiguous


def test_nested_groups_without_repeat_offset_anchor_are_accepted():
    """The real archive ``map.yaml`` shape. OpenMSCG raises ``KeyError: 'repeat'``."""
    mapping = simple_mapping(
        system=[
            {
                "anchor": 0,
                "repeat": 3,
                "offset": 5,
                "groups": [{"sites": [["A", 0], ["B", 3]]}],
            }
        ]
    )
    sites, _, mol_ids = expand_mapping_sites(mapping)
    assert sites == [("A", 0), ("B", 3), ("A", 5), ("B", 8), ("A", 10), ("B", 13)]
    assert mol_ids == [0, 0, 1, 1, 2, 2]

    with pytest.raises(KeyError):
        openmscg_unpack_group({"anchor": 0, "offset": 0, "repeat": 1, "groups": mapping["system"]})


def test_nested_anchor_is_added_exactly_once():
    """OpenMSCG double-counts ``root_anchor + anchor`` in the ``groups`` branch."""
    mapping = simple_mapping(
        system=[
            {
                "anchor": 400,
                "repeat": 2,
                "offset": 10,
                "groups": [{"anchor": 0, "repeat": 1, "offset": 0, "sites": [["A", 0]]}],
            }
        ]
    )
    sites, _, _ = expand_mapping_sites(mapping)
    assert sites == [("A", 400), ("A", 410)]

    # What OpenMSCG produces instead: every site shifted by an extra 400.
    legacy = openmscg_unpack_group(
        {"anchor": 0, "offset": 0, "repeat": 1, "groups": mapping["system"]}
    )
    assert legacy == [["A", 800], ["A", 810]]


def test_nested_child_repeat_is_flattened_into_the_unit():
    mapping = simple_mapping(
        system=[
            {
                "anchor": 1000,
                "repeat": 2,
                "offset": 50,
                "groups": [{"anchor": 0, "repeat": 2, "offset": 5, "sites": [["A", 0]]}],
            }
        ]
    )
    sites, _, mol_ids = expand_mapping_sites(mapping)
    # One outer repeat unit contains both inner repeats, so it is one molecule.
    assert sites == [("A", 1000), ("A", 1005), ("A", 1050), ("A", 1055)]
    assert mol_ids == [0, 0, 1, 1]


def test_vectorized_compilation_matches_the_reference_expansion():
    mapping = simple_mapping(
        system=[
            {"anchor": 7, "repeat": 4, "offset": 5, "sites": [["A", 0], ["B", 3], ["A", 0]]},
            {"anchor": 200, "repeat": 2, "offset": 5, "groups": [{"sites": [["B", 1]]}]},
        ]
    )
    sites, group_ids, mol_ids = expand_mapping_sites(mapping)
    spec = CGMapSpec.from_mapping(mapping)

    assert spec.n_sites == len(sites)
    assert np.array_equal(spec.site_group_ids, group_ids)
    assert np.array_equal(spec.site_mol_ids, mol_ids)
    expected_ids = [spec.type_names.index(str(key)) + 1 for key, _ in sites]
    assert np.array_equal(spec.site_type_ids, expected_ids)
    for site, (type_key, anchor) in enumerate(sites):
        want = anchor + np.asarray(mapping["site-types"][type_key]["index"], dtype=np.int64)
        assert np.array_equal(absolute_site_indices(spec, site), want)


# ─── site-type table quirks ───────────────────────────────────────────


def test_integer_site_type_keys_resolve():
    """YAML parses a bare ``1:`` key as an int; ``sites`` may spell it either way."""
    mapping = {
        "site-types": {1: {"index": [0, 1], "x-weight": [1.0, 1.0]}},
        "system": [{"anchor": 0, "repeat": 2, "offset": 2, "sites": [[1, 0]]}],
    }
    spec = CGMapSpec.from_mapping(mapping)
    assert spec.type_names == ("1",)
    assert np.array_equal(spec.site_type_ids, [1, 1])

    mixed = {
        "site-types": {1: {"index": [0, 1], "x-weight": [1.0, 1.0]}},
        "system": [{"anchor": 0, "repeat": 1, "offset": 0, "sites": [["1", 0]]}],
    }
    assert CGMapSpec.from_mapping(mixed).n_sites == 1


def test_integer_x_weight_is_accepted():
    """OpenMSCG's in-place ``v /= v.sum()`` raises ``TypeError`` on int lists."""
    mapping = {
        "site-types": {"A": {"index": [0, 1], "x-weight": [3, 1]}},
        "system": [{"anchor": 0, "repeat": 1, "offset": 0, "sites": [["A", 0]]}],
    }
    spec = CGMapSpec.from_mapping(mapping)
    assert np.allclose(spec.csr_wx, [0.75, 0.25])


def test_missing_f_weight_defaults_to_ones():
    mapping = {
        "site-types": {"A": {"index": [0, 1], "x-weight": [3.0, 1.0]}},
        "system": [{"anchor": 0, "repeat": 1, "offset": 0, "sites": [["A", 0]]}],
    }
    spec = CGMapSpec.from_mapping(mapping)
    assert np.allclose(spec.csr_wf, 1.0)


def test_x_weight_is_row_normalized_and_f_weight_is_raw():
    mapping = {
        "site-types": {"A": {"index": [0, 1], "x-weight": [3.0, 1.0], "f-weight": [2.0, 5.0]}},
        "system": [{"anchor": 0, "repeat": 2, "offset": 2, "sites": [["A", 0]]}],
    }
    spec = CGMapSpec.from_mapping(mapping)
    for site in range(spec.n_sites):
        lo, hi = int(spec.csr_indptr[site]), int(spec.csr_indptr[site + 1])
        assert spec.csr_wx[lo:hi].sum() == pytest.approx(1.0)
    assert np.allclose(spec.csr_wf, [2.0, 5.0, 2.0, 5.0])


def test_negative_index_offsets_are_supported():
    """The documented ``index: [-1, 0, 1]`` form with a non-zero site offset."""
    mapping = {
        "site-types": {"A": {"index": [-1, 0, 1], "x-weight": [1.0, 1.0, 1.0]}},
        "system": [{"anchor": 0, "repeat": 2, "offset": 10, "sites": [["A", 1]]}],
    }
    spec = CGMapSpec.from_mapping(mapping)
    assert np.array_equal(absolute_site_indices(spec, 0), [0, 1, 2])
    assert np.array_equal(absolute_site_indices(spec, 1), [10, 11, 12])


def test_negative_absolute_index_is_rejected():
    mapping = {
        "site-types": {"A": {"index": [-2, 0], "x-weight": [1.0, 1.0]}},
        "system": [{"anchor": 0, "repeat": 1, "offset": 0, "sites": [["A", 0]]}],
    }
    with pytest.raises(ValueError, match="negative AA atom index"):
        CGMapSpec.from_mapping(mapping)


def test_duplicate_indices_inside_one_site_sum():
    """``Mapper.process`` sums duplicates; ``Mapper.get_matrix`` overwrites. We sum."""
    mapping = {
        "site-types": {"A": {"index": [0, 0, 1], "x-weight": [1.0, 1.0, 2.0]}},
        "system": [{"anchor": 0, "repeat": 1, "offset": 0, "sites": [["A", 0]]}],
    }
    spec = CGMapSpec.from_mapping(mapping)
    assert spec.nnz == 3
    assert spec.n_required_atoms == 2
    # Atom 0 keeps two separate weight entries pointing at the same column.
    assert list(spec.csr_cols) == [0, 0, 1]
    assert np.allclose(spec.csr_wx, [0.25, 0.25, 0.5])

    positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    expected = slow_map_positions(mapping, positions)
    assert expected[0] == pytest.approx([2.0, 0.0, 0.0])


def test_out_of_range_index_is_rejected_against_n_atoms():
    mapping = simple_mapping()
    with pytest.raises(ValueError, match="only 10 atoms"):
        CGMapSpec.from_mapping(mapping, n_atoms=10)


def test_unknown_site_type_is_reported_with_known_types():
    mapping = simple_mapping(
        system=[{"anchor": 0, "repeat": 1, "offset": 0, "sites": [["MISSING", 0]]}]
    )
    with pytest.raises(ValueError, match="no entry for site type 'MISSING'"):
        CGMapSpec.from_mapping(mapping)


def test_x_weight_length_mismatch_is_rejected():
    mapping = {
        "site-types": {"A": {"index": [0, 1, 2], "x-weight": [1.0, 1.0]}},
        "system": [{"anchor": 0, "repeat": 1, "offset": 0, "sites": [["A", 0]]}],
    }
    with pytest.raises(ValueError, match="len\\(x-weight\\)=2"):
        CGMapSpec.from_mapping(mapping)


def test_zero_x_weight_sum_is_rejected():
    mapping = {
        "site-types": {"A": {"index": [0, 1], "x-weight": [1.0, -1.0]}},
        "system": [{"anchor": 0, "repeat": 1, "offset": 0, "sites": [["A", 0]]}],
    }
    with pytest.raises(ValueError, match="sums to 0"):
        CGMapSpec.from_mapping(mapping)


# ─── centre-of-mass fallback ──────────────────────────────────────────


def test_missing_x_weight_falls_back_to_masses():
    mapping = {
        "site-types": {"A": {"index": [0, 1]}},
        "system": [{"anchor": 0, "repeat": 2, "offset": 2, "sites": [["A", 0]]}],
    }
    masses = np.array([12.0, 4.0, 16.0, 4.0])
    spec = CGMapSpec.from_mapping(mapping, masses=masses)
    assert np.allclose(spec.csr_wx, [0.75, 0.25, 0.8, 0.2])


def test_missing_x_weight_without_masses_is_rejected():
    mapping = {
        "site-types": {"A": {"index": [0, 1]}},
        "system": [{"anchor": 0, "repeat": 1, "offset": 0, "sites": [["A", 0]]}],
    }
    with pytest.raises(ValueError, match="no `masses` array was supplied"):
        CGMapSpec.from_mapping(mapping)


# ─── molecule grouping and PBC reference atoms ────────────────────────


def test_molecule_grouping_is_contiguous_for_replicated_systems():
    spec = CGMapSpec.from_mapping(simple_mapping())
    assert spec.molecules_contiguous
    assert list(spec.mol_indptr) == [0, 5, 10, 15]
    assert [int(spec.atom_indices[p]) for p in spec.mol_ref_pos] == [0, 5, 10]


def test_molecule_grouping_gathers_when_sites_interleave_molecules():
    """A mapping whose repeat units interleave in the AA file needs a gather."""
    mapping = {
        "site-types": {"A": {"index": [0, 2], "x-weight": [1.0, 1.0]}},
        # offset 1 with index stride 2 makes molecule 0 own {0,2} and molecule 1
        # own {1,3} — ascending order no longer groups by molecule.
        "system": [{"anchor": 0, "repeat": 2, "offset": 1, "sites": [["A", 0]]}],
    }
    spec = CGMapSpec.from_mapping(mapping)
    assert not spec.molecules_contiguous
    grouped = [
        sorted(int(spec.atom_indices[p]) for p in spec.mol_atom_pos[
            int(spec.mol_indptr[m]) : int(spec.mol_indptr[m + 1])
        ])
        for m in range(spec.n_mol)
    ]
    assert grouped == [[0, 2], [1, 3]]


def test_mol_reference_anchor_selects_the_repeat_unit_origin():
    mapping = {
        "site-types": {"A": {"index": [3, 4], "x-weight": [1.0, 1.0]}},
        "system": [{"anchor": 0, "repeat": 2, "offset": 10, "sites": [["A", 0]]}],
    }
    # Atom 0 (the unit origin) is not mapped, so 'anchor' must fall back.
    with pytest.warns(RuntimeWarning, match="not part of the mapping"):
        spec = CGMapSpec.from_mapping(mapping, mol_reference="anchor")
    assert [int(spec.atom_indices[p]) for p in spec.mol_ref_pos] == [3, 13]

    spec_int = CGMapSpec.from_mapping(mapping, mol_reference=4)
    assert [int(spec_int.atom_indices[p]) for p in spec_int.mol_ref_pos] == [4, 14]


def test_atom_shared_between_molecules_warns():
    mapping = {
        "site-types": {"A": {"index": [0, 1], "x-weight": [1.0, 1.0]}},
        # offset 1 < the 2-atom span, so consecutive molecules share an atom.
        "system": [{"anchor": 0, "repeat": 2, "offset": 1, "sites": [["A", 0]]}],
    }
    with pytest.warns(RuntimeWarning, match="more than one molecule"):
        CGMapSpec.from_mapping(mapping)


def test_site_ref_pos_points_at_index_zero():
    mapping = {
        "site-types": {"A": {"index": [5, 0, 2], "x-weight": [1.0, 1.0, 1.0]}},
        "system": [{"anchor": 0, "repeat": 1, "offset": 0, "sites": [["A", 0]]}],
    }
    spec = CGMapSpec.from_mapping(mapping)
    # index[0] is 5, not the lowest index — OpenMSCG pivots PBC on exactly this.
    assert int(spec.atom_indices[spec.site_ref_pos[0]]) == 5


# ─── optional CG bonded topology ──────────────────────────────────────


def test_cg_topology_block_replicates_across_repeats_and_groups():
    mapping = simple_mapping(
        system=[
            {"anchor": 0, "repeat": 2, "offset": 5, "sites": [["A", 0], ["B", 3]]},
            {"anchor": 100, "repeat": 2, "offset": 5, "sites": [["A", 0], ["B", 3]]},
        ],
    )
    mapping["cg-topology"] = {
        "molecule": {
            "names": ["HEAD", "TAIL"],
            "charges": [1.0, -1.0],
            "masses": "auto",
            "bonds": [[0, 1]],
            "angles": [],
        }
    }
    spec = CGMapSpec.from_mapping(mapping)
    assert spec.has_bonded_topology
    # One bond per repeat unit of each group: 2 + 2.
    assert spec.bonds.shape == (4, 2)
    assert np.array_equal(spec.bonds, [[0, 1], [2, 3], [4, 5], [6, 7]])
    assert np.array_equal(spec.bond_type_ids, [1, 1, 1, 1])
    assert spec.bond_type_keys == (InteractionKey.bond("A", "B"),)
    assert spec.site_labels == ("HEAD", "TAIL") * 4
    assert np.allclose(spec.site_charges, [1.0, -1.0] * 4)
    # masses: "auto" leaves the summed x-weight in place.
    assert np.allclose(spec.site_masses_array(), [14.0, 17.0] * 4)


def test_cg_topology_interaction_ids_group_by_site_type_not_position():
    """Two occurrences of the same type must share one bond type id."""
    mapping = {
        "site-types": {
            "HG": {"index": [0], "x-weight": [1.0]},
            "MG": {"index": [0], "x-weight": [1.0]},
            "T": {"index": [0], "x-weight": [1.0]},
        },
        # HG-MG-T ... T, with T appearing at two positions.
        "system": [
            {"anchor": 0, "repeat": 2, "offset": 4,
             "sites": [["HG", 0], ["MG", 1], ["T", 2], ["T", 3]]}
        ],
        "cg-topology": {
            "molecule": {"bonds": [[0, 1], [1, 2], [1, 3]]},
        },
    }
    spec = CGMapSpec.from_mapping(mapping)
    # bond 0 = HG-MG (type 1); bonds 1 and 2 are both MG-T → both type 2.
    assert list(spec.bond_type_ids[:3]) == [1, 2, 2]
    assert spec.bond_type_keys == (
        InteractionKey.bond("HG", "MG"),
        InteractionKey.bond("MG", "T"),
    )


def test_cg_topology_types_merge_distinct_mapping_templates():
    mapping = {
        "site-types": {
            name: {"index": [0], "x-weight": [mass]}
            for name, mass in zip("ABCD", (10.0, 20.0, 30.0, 40.0))
        },
        "system": [
            {
                "anchor": 0,
                "repeat": 2,
                "offset": 4,
                "sites": [[name, index] for index, name in enumerate("ABCD")],
            }
        ],
        "cg-topology": {
            "molecule": {
                "names": ["H", "T1", "T2", "T3"],
                "types": ["HEAD", "TAIL", "TAIL", "TAIL"],
                "masses": [10.0, 30.0, 30.0, 30.0],
                "charges": [1.0, 0.0, 0.0, 0.0],
                "bonds": [[0, 1], [1, 2], [2, 3]],
                "angles": [[0, 1, 2], [1, 2, 3]],
                "dihedrals": [[0, 1, 2, 3]],
            }
        },
    }
    spec = CGMapSpec.from_mapping(mapping)
    assert spec.type_names == ("HEAD", "TAIL")
    assert np.array_equal(spec.site_type_ids, [1, 2, 2, 2, 1, 2, 2, 2])
    assert spec.site_labels == ("H", "T1", "T2", "T3") * 2
    assert np.allclose(spec.site_masses_array(), [10.0, 30.0, 30.0, 30.0] * 2)
    assert spec.bond_type_keys == (
        InteractionKey.bond("HEAD", "TAIL"),
        InteractionKey.bond("TAIL", "TAIL"),
    )
    assert spec.angle_type_keys == (
        InteractionKey.angle("HEAD", "TAIL", "TAIL"),
        InteractionKey.angle("TAIL", "TAIL", "TAIL"),
    )
    assert spec.dihedral_type_keys == (
        InteractionKey.dihedral("HEAD", "TAIL", "TAIL", "TAIL"),
    )


def test_cg_topology_types_reject_inconsistent_mass_for_one_type():
    mapping = simple_mapping()
    mapping["cg-topology"] = {
        "molecule": {
            "types": ["T", "T"],
            "masses": [14.0, 17.0],
        }
    }
    with pytest.raises(ValueError, match="inconsistent masses.*canonical type 'T'"):
        CGMapSpec.from_mapping(mapping)


def test_cg_topology_angles_and_dihedrals_get_their_own_type_tables():
    mapping = {
        "site-types": {name: {"index": [0], "x-weight": [1.0]} for name in "ABCD"},
        "system": [
            {"anchor": 0, "repeat": 2, "offset": 4,
             "sites": [["A", 0], ["B", 1], ["C", 2], ["D", 3]]}
        ],
        "cg-topology": {
            "molecule": {
                "bonds": [[0, 1], [1, 2], [2, 3]],
                "angles": [[0, 1, 2], [1, 2, 3]],
                "dihedrals": [[0, 1, 2, 3]],
            }
        },
    }
    spec = CGMapSpec.from_mapping(mapping)
    assert spec.bonds.shape == (6, 2)
    assert spec.angles.shape == (4, 3)
    assert spec.dihedrals.shape == (2, 4)
    assert spec.angle_type_keys == (
        InteractionKey.angle("A", "B", "C"),
        InteractionKey.angle("B", "C", "D"),
    )
    assert spec.dihedral_type_keys == (InteractionKey.dihedral("A", "B", "C", "D"),)
    assert np.array_equal(spec.dihedrals, [[0, 1, 2, 3], [4, 5, 6, 7]])


def test_cg_topology_per_group_templates():
    mapping = simple_mapping(
        system=[
            {"anchor": 0, "repeat": 1, "offset": 5, "sites": [["A", 0], ["B", 3]]},
            {"anchor": 100, "repeat": 1, "offset": 5, "sites": [["A", 0]]},
        ],
    )
    mapping["cg-topology"] = {
        "groups": [
            {"names": ["X", "Y"], "bonds": [[0, 1]]},
            {"names": ["Z"]},
        ]
    }
    spec = CGMapSpec.from_mapping(mapping)
    assert spec.site_labels == ("X", "Y", "Z")
    assert np.array_equal(spec.bonds, [[0, 1]])


def test_cg_topology_local_indices_out_of_range_are_rejected():
    mapping = simple_mapping()
    mapping["cg-topology"] = {"molecule": {"bonds": [[0, 5]]}}
    with pytest.raises(ValueError, match="site positions within one repeat unit"):
        CGMapSpec.from_mapping(mapping)


def test_cg_topology_length_mismatch_is_rejected():
    mapping = simple_mapping()
    mapping["cg-topology"] = {"molecule": {"names": ["only-one"]}}
    with pytest.raises(ValueError, match="has 1 entries but the repeat unit has 2"):
        CGMapSpec.from_mapping(mapping)


def test_cg_topology_can_be_ignored_explicitly():
    mapping = simple_mapping()
    mapping["cg-topology"] = {"molecule": {"bonds": [[0, 1]]}}
    spec = CGMapSpec.from_mapping(mapping, cg_topology={})
    assert not spec.has_bonded_topology


def test_spec_without_cg_topology_still_emits_bead_records():
    """Use case (ii): CG bonded topology not determined yet."""
    spec = CGMapSpec.from_mapping(simple_mapping())
    assert not spec.has_bonded_topology
    beads, type2id, type_masses = spec.bead_records(resname="LIP")
    assert len(beads) == spec.n_sites
    assert beads[0]["resid"] == 1 and beads[2]["resid"] == 2
    assert beads[0]["resname"] == "LIP"
    assert type2id == {"A": 1, "B": 2}
    assert type_masses == pytest.approx({"A": 14.0, "B": 17.0})


def test_bead_records_accept_one_resname_per_group():
    mapping = simple_mapping(
        system=[
            {"anchor": 0, "repeat": 1, "offset": 5, "sites": [["A", 0]]},
            {"anchor": 50, "repeat": 1, "offset": 5, "sites": [["B", 0]]},
        ]
    )
    spec = CGMapSpec.from_mapping(mapping)
    beads, _, _ = spec.bead_records(resname=["DPPC", "CHOL"])
    assert [bead["resname"] for bead in beads] == ["DPPC", "CHOL"]
    with pytest.raises(ValueError, match="1 entries but the mapping has 2"):
        spec.bead_records(resname=["ONLY"])


# ─── serialization ────────────────────────────────────────────────────


def test_spec_round_trips_through_pickle():
    mapping = simple_mapping()
    mapping["cg-topology"] = {"molecule": {"names": ["X", "Y"], "bonds": [[0, 1]]}}
    spec = CGMapSpec.from_mapping(mapping)
    restored = pickle.loads(pickle.dumps(spec, protocol=pickle.HIGHEST_PROTOCOL))
    assert restored.n_sites == spec.n_sites
    assert np.array_equal(restored.csr_wx, spec.csr_wx)
    assert np.array_equal(restored.bonds, spec.bonds)
    assert restored.bond_type_keys == spec.bond_type_keys
    assert restored.site_labels == spec.site_labels


def test_spec_is_frozen():
    spec = CGMapSpec.from_mapping(simple_mapping())
    with pytest.raises(Exception):
        spec.site_type_ids = None  # type: ignore[misc]


# ─── the real archive mapping ─────────────────────────────────────────


def test_archive_map_yaml_compiles_to_the_expected_plan():
    """Pins the 6-bead DPPC map that produced our known-good CG reference."""
    mapping = load_mapping_yaml(ARCHIVE_MAP)
    # The file uses integer site-type keys and a nested `groups` child with no
    # anchor/repeat/offset — both of which OpenMSCG mishandles.
    assert list(mapping["site-types"]) == [1, 2, 3, 4]
    assert "groups" in mapping["system"][0]

    sites, _, mol_ids = expand_mapping_sites(mapping)
    assert len(sites) == 27648
    assert sites[:6] == [(1, 0), (2, 24), (3, 44), (4, 65), (3, 87), (4, 108)]
    assert sites[-1] == (4, 599018)
    assert max(mol_ids) + 1 == 4608

    spec = CGMapSpec.from_mapping(mapping, n_atoms=599040)
    assert spec.n_sites == 27648
    assert spec.n_mol == 4608
    assert spec.n_types == 4
    assert spec.type_names == ("1", "2", "3", "4")
    # Every one of the 599040 DPPC atoms is used exactly once.
    assert spec.n_required_atoms == 599040
    assert spec.nnz == 599040
    assert int(spec.atom_indices[0]) == 0
    assert int(spec.atom_indices[-1]) == 599039
    assert spec.molecules_contiguous
    # x-weight holds CHARMM36 masses → summed site masses.
    assert np.allclose(spec.type_xweight_sum, [181.9, 157.0, 98.0, 99.0])
    assert np.allclose(spec.csr_wf, 1.0)
    assert spec.nbytes() < 32 * 1024 * 1024


def test_archive_map_matches_the_slow_reference_on_random_positions():
    mapping = load_mapping_yaml(ARCHIVE_MAP)
    spec = CGMapSpec.from_mapping(mapping, n_atoms=599040)
    rng = np.random.default_rng(20260729)
    positions = rng.random((599040, 3)) * 100.0

    # Compare a random sample of sites against the explicit per-site loop.
    sites, _, _ = expand_mapping_sites(mapping)
    compact = positions[spec.atom_indices]
    for site in rng.choice(spec.n_sites, size=40, replace=False):
        lo, hi = int(spec.csr_indptr[site]), int(spec.csr_indptr[site + 1])
        got = (compact[spec.csr_cols[lo:hi]] * spec.csr_wx[lo:hi, None]).sum(axis=0)
        type_key, anchor = sites[int(site)]
        entry = mapping["site-types"][type_key]
        idx = anchor + np.asarray(entry["index"], dtype=np.int64)
        weight = np.asarray(entry["x-weight"], dtype=np.float64)
        want = (positions[idx] * weight[:, None]).sum(axis=0) / weight.sum()
        assert got == pytest.approx(want, abs=1e-10)


def test_site_arrays_is_cached_and_never_pickled():
    """The derived arrays are fixed by the plan, so they are built once — and the
    cache must not ride along in an MPI broadcast of the spec."""
    mapping = simple_mapping(
        system=[{"anchor": 0, "repeat": 3, "offset": 5, "sites": [["A", 0], ["B", 1]]}]
    )
    spec = CGMapSpec.from_mapping(mapping)

    first = spec.site_arrays()
    assert spec.site_arrays() is first  # same object, not an equal rebuild
    assert first.n_sites == spec.n_sites
    assert first.n_residues == spec.n_residues == 3
    assert len(first.residue_names) == 3          # per residue, not per site
    assert first.site_resnames() == ("CG",) * 6   # per site, on request
    np.testing.assert_array_equal(first.res_ids, spec.site_mol_ids)
    np.testing.assert_allclose(first.masses, spec.site_masses_array())
    assert first.site_types() == tuple(
        spec.type_names[int(tid) - 1] for tid in spec.site_type_ids
    )

    restored = pickle.loads(pickle.dumps(spec))
    assert getattr(restored, "_site_arrays_cache", None) is None
    assert restored.site_arrays().labels == first.labels

    # A malformed resname is still rejected after the cache is warm, and a valid
    # but different one is refused rather than silently ignored.
    with pytest.raises(ValueError, match="2 entries but the mapping has 1"):
        spec.site_arrays(resname=["ONE", "TOO_MANY"])
    with pytest.raises(ValueError, match="fixed for a spec"):
        spec.site_arrays(resname="OTHER")


def test_site_arrays_ignores_resname_when_the_yaml_declares_residues():
    mapping = simple_mapping(
        system=[{"anchor": 0, "repeat": 2, "offset": 5, "sites": [["A", 0], ["B", 1]]}]
    )
    mapping["cg-topology"] = {
        "residues": [{"resname": "LIP", "names": ["H", "T"], "linkable": False}],
        "groups": [{"molname": "LIP", "resnames": ["LIP"]}],
    }
    spec = CGMapSpec.from_mapping(mapping)
    arrays = spec.site_arrays(resname="IGNORED")
    assert arrays.residue_names == ("LIP", "LIP")
    # The argument plays no part, so asking again with anything else is fine.
    assert spec.site_arrays(resname="ALSO_IGNORED") is arrays
