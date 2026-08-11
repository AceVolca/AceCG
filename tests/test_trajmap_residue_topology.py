"""Residue/group topology semantics and the real protein acceptance mappings."""

from __future__ import annotations

from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
import yaml

from AceCG.topology.cgmap import CGMapSpec
from AceCG.topology.cgmap_builder import (
    build_mapping_from_sites,
    derive_angles_from_bonds,
)


REPO = Path(__file__).resolve().parents[1]

# Shipped fixture, not a private path: `experiments/` is repository-local and
# never published, so a test that read it could not run for anyone who cloned
# the repository. See tests/test_data/protein_helix_ac/README.md.
PROTEIN_INPUTS = Path(__file__).resolve().parent / "test_data" / "protein_helix_ac"
PROTEIN_TOPOLOGY = PROTEIN_INPUTS / "md_start.pdb"


def _synthetic_universe():
    universe = mda.Universe.empty(
        8,
        n_residues=4,
        atom_resindex=np.repeat(np.arange(4), 2),
        trajectory=True,
    )
    universe.add_TopologyAttr("name", ["A", "B"] * 4)
    universe.add_TopologyAttr("type", ["A", "B"] * 4)
    universe.add_TopologyAttr("resid", [1, 2, 1, 2])
    universe.add_TopologyAttr("resname", ["ALA", "GLY", "ALA", "GLY"])
    universe.add_TopologyAttr("mass", [12.0, 1.0] * 4)
    return universe


def test_residue_templates_link_sequentially_and_keep_group_identity():
    universe = _synthetic_universe()
    groups = [
        [("A", [0]), ("B", [1]), ("G", [2, 3])],
        [("A", [4]), ("B", [5]), ("G", [6, 7])],
    ]
    cg_topology = {
        "residues": [
            {
                "resname": "ALA",
                "names": ["A", "B"],
                "linkable": True,
                "left_linker_atom_type": "A",
                "right_linker_atom_type": "B",
                "bonds": [[0, 1]],
            },
            {
                "resname": "GLY",
                "names": ["G"],
                "linkable": True,
                "left_linker_atom_type": "G",
                "right_linker_atom_type": "G",
            },
        ],
        "groups": [
            {
                "molname": "PROT_A",
                "resids": [0, 0, 1],
                "resnames": ["ALA", "GLY"],
                "linker_angles": True,
            },
            {
                "molname": "PROT_B",
                "resids": [0, 0, 1],
                "resnames": ["ALA", "GLY"],
                "linker_angles": True,
            },
        ],
    }
    mapping = build_mapping_from_sites(
        universe,
        groups,
        cg_topology=cg_topology,
    )
    spec = CGMapSpec.from_mapping(mapping, n_atoms=8)

    assert spec.group_molnames == ("PROT_A", "PROT_B")
    assert spec.site_res_ids.tolist() == [0, 0, 1, 2, 2, 3]
    assert spec.residue_names_array() == ("ALA", "GLY", "ALA", "GLY")
    assert spec.bonds.tolist() == [[0, 1], [1, 2], [3, 4], [4, 5]]
    assert spec.angles.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_linker_angles_can_be_disabled_with_one_boolean():
    universe = _synthetic_universe()
    mapping = build_mapping_from_sites(
        universe,
        [[("A", [0]), ("B", [1]), ("G", [2, 3])]],
        cg_topology={
            "residues": [
                {
                    "resname": "ALA",
                    "names": ["A", "B"],
                    "linkable": True,
                    "left_linker_atom_type": "A",
                    "right_linker_atom_type": "B",
                    "bonds": [[0, 1]],
                },
                {
                    "resname": "GLY",
                    "names": ["G"],
                    "linkable": True,
                    "left_linker_atom_type": "G",
                    "right_linker_atom_type": "G",
                },
            ],
            "groups": [
                {
                    "molname": "PROT_A",
                    "resids": [0, 0, 1],
                    "resnames": ["ALA", "GLY"],
                    "linker_angles": False,
                }
            ],
        },
    )
    spec = CGMapSpec.from_mapping(mapping, n_atoms=4)

    assert spec.bonds.tolist() == [[0, 1], [1, 2]]
    assert spec.angles is None


def test_standalone_linkable_residues_are_not_linked_across_molecule_repeats():
    mapping = {
        "site-types": {
            "A": {"index": [0], "x-weight": [1.0]},
            "B": {"index": [0], "x-weight": [1.0]},
        },
        "system": [
            {
                "anchor": 0,
                "repeat": 2,
                "offset": 2,
                "sites": [["A", 0], ["B", 1]],
            }
        ],
        "cg-topology": {
            "residues": [
                {
                    "resname": "ALA",
                    "names": ["A", "B"],
                    "linkable": True,
                    "left_linker_atom_type": "A",
                    "right_linker_atom_type": "B",
                    "bonds": [[0, 1]],
                }
            ]
        },
    }
    spec = CGMapSpec.from_mapping(mapping, n_atoms=4)

    assert spec.bonds.tolist() == [[0, 1], [2, 3]]
    assert [1, 2] not in spec.bonds.tolist()
    assert spec.angles is None


def test_linker_angles_requires_a_boolean():
    mapping = {
        "site-types": {"A": {"index": [0], "x-weight": [1.0]}},
        "system": [
            {
                "anchor": 0,
                "repeat": 1,
                "offset": 0,
                "sites": [["A", 0]],
            }
        ],
        "cg-topology": {
            "residues": [{"resname": "ALA", "names": ["A"], "linkable": False}],
            "groups": [{"resnames": ["ALA"], "linker_angles": "all"}],
        },
    }
    with pytest.raises(ValueError, match="linker_angles must be a boolean"):
        CGMapSpec.from_mapping(mapping)


def test_linker_angles_is_rejected_outside_a_group():
    mapping = {
        "site-types": {"A": {"index": [0], "x-weight": [1.0]}},
        "system": [
            {
                "anchor": 0,
                "repeat": 1,
                "offset": 0,
                "sites": [["A", 0]],
            }
        ],
        "cg-topology": {
            "linker_angles": True,
            "residues": [{"resname": "ALA", "names": ["A"], "linkable": False}],
        },
    }
    with pytest.raises(ValueError, match="linker_angles is group-local"):
        CGMapSpec.from_mapping(mapping)


def test_residue_schema_rejects_a_one_sided_link():
    mapping = {
        "site-types": {
            "A": {"index": [0], "x-weight": [1.0]},
            "B": {"index": [0], "x-weight": [1.0]},
        },
        "system": [
            {
                "anchor": 0,
                "repeat": 1,
                "offset": 0,
                "sites": [["A", 0], ["B", 1]],
            }
        ],
        "cg-topology": {
            "residues": [
                {
                    "resname": "ALA",
                    "names": ["A"],
                    "linkable": True,
                    "left_linker_atom_type": "A",
                    "right_linker_atom_type": "A",
                },
                {"resname": "GLY", "names": ["B"], "linkable": False},
            ],
            "groups": [
                {
                    "resids": [0, 1],
                    "resnames": ["ALA", "GLY"],
                }
            ],
        },
    }
    with pytest.raises(ValueError, match="only one is linkable"):
        CGMapSpec.from_mapping(mapping)


def test_angles_are_the_unique_neighbour_pairs_of_a_bond_graph():
    assert derive_angles_from_bonds([[0, 1], [1, 2], [1, 3], [3, 4]]) == [
        [0, 1, 2],
        [0, 1, 3],
        [2, 1, 3],
        [1, 3, 4],
    ]


@pytest.mark.parametrize(
    "scheme,n_sites,n_types,n_residues,n_bonds,n_angles",
    [
        ("1res", 99, 25, 99, 97, 95),
        ("2res", 50, 46, 2, 48, 46),
        ("4site", 392, 27, 99, 390, 483),
    ],
)
def test_real_protein_mapping_is_complete_and_bonded(
    scheme,
    n_sites,
    n_types,
    n_residues,
    n_bonds,
    n_angles,
):
    path = PROTEIN_INPUTS / f"prot_{scheme}_map.yaml"
    mapping = yaml.safe_load(path.read_text())
    spec = CGMapSpec.from_mapping(mapping, n_atoms=1687)
    universe = mda.Universe(str(PROTEIN_TOPOLOGY))

    for group in mapping["cg-topology"]["groups"]:
        assert "additional_bonds" not in group
        assert "additional_angles" not in group
    assert spec.n_sites == n_sites
    assert spec.n_types == n_types
    assert spec.n_residues == n_residues
    assert spec.group_molnames == ("PROT_A", "PROT_C")
    assert spec.bonds.shape == (n_bonds, 2)
    assert spec.angles.shape == (n_angles, 3)
    assert np.unique(spec.atom_indices).size == universe.atoms.n_atoms
    assert np.array_equal(np.sort(spec.atom_indices), np.arange(1687))
    assert spec.site_masses_array().sum() == pytest.approx(
        universe.atoms.masses.sum(), abs=1e-6
    )

    if scheme == "1res":
        assert all(
            group["linker_angles"] is True
            for group in mapping["cg-topology"]["groups"]
        )
        assert {"HSD", "HSE"} <= set(spec.type_names)
        assert spec.residue_names_array().count("HSD") == 1
        assert spec.residue_names_array().count("HSE") == 1
    elif scheme == "2res":
        assert all("-" in name or name.endswith("c") for name in spec.type_names)
    else:
        assert all(
            group["linker_angles"] is True
            for group in mapping["cg-topology"]["groups"]
        )
        assert {"SHSD1", "SHSE1"} <= set(spec.type_names)
        assert spec.residue_names_array().count("HSD") == 1
        assert spec.residue_names_array().count("HSE") == 1


def test_molecule_id_stays_the_molecule_when_residues_are_finer():
    """`resid` may be per-residue, but the LAMMPS molecule column must not be.

    ``write_lammps_data`` writes the molecule id from the bead record, and
    ``collect_topology_arrays(exclude_option="resid")`` groups nonbonded
    exclusions by it. If a 99-residue chain reported 99 molecules, only
    intra-residue pairs would be group-excluded.
    """
    mapping = {
        "site-types": {"A": {"index": [0], "x-weight": [12.0]}},
        "system": [{"anchor": 0, "repeat": 2, "offset": 3, "sites": [["A", 0], ["A", 1], ["A", 2]]}],
        "cg-topology": {
            "residues": [
                {
                    "resname": "R",
                    "names": ["S"],
                    "linkable": True,
                    "left_linker_atom_type": "S",
                    "right_linker_atom_type": "S",
                }
            ],
            "groups": [{"molname": "M", "resnames": ["R", "R", "R"], "resids": [0, 1, 2]}],
        },
    }
    spec = CGMapSpec.from_mapping(mapping, n_atoms=6)
    beads, _, _ = spec.bead_records()

    assert spec.n_mol == 2 and spec.n_residues == 6
    assert [bead["molid"] for bead in beads] == [1, 1, 1, 2, 2, 2]
    assert [bead["resid"] for bead in beads] == [1, 2, 3, 4, 5, 6]
