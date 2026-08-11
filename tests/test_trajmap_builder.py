"""Pin the mapping-authoring side: bead tables, group discovery, itp parsing.

The high-value case is group discovery. OpenMSCG's ``cgyaml`` decides "new group"
by comparing residue names, so a tiled system whose lipid blocks are separated by
water collapses into a single group whose ``repeat`` walks straight through the
solvent. Ours splits on atom-index contiguity. Both the collapse and the fix are
asserted here, against the real DPPC archive layout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

import MDAnalysis as mda

from AceCG.topology.cgmap import CGMapSpec, expand_mapping_sites
from AceCG.topology.cgmap_builder import (
    bead_table_from_names,
    build_mapping,
    parse_bead_table,
    parse_gromacs_itp,
    write_mapping_yaml,
)
from AceCG.topology.types import InteractionKey


from real_frames import DPPC_AA, DPPC_MARTINI12, dppc_aa_universe

# Shipped fixtures (tests/test_data/dppc_aa/, tests/test_data/dppc_martini12/;
# see their README.md files). These are ours to redistribute and are what the
# DPPC-residue tests below run on, so that coverage exists for anyone who
# clones the repository.
MARTINI_BEADS = DPPC_AA / "martini_dppc_charmm36_beads.yaml"
SHIPPED_MAP = DPPC_AA / "martini12_2mol_map.yaml"
ARCHIVE_MAP = DPPC_MARTINI12 / "map.yaml"
MARTINI_ITP = DPPC_MARTINI12 / "martini_v2.0_DPPC_01-alt-opt.itp"

# `DPPC_mapping.txt` belongs to the OpenMSCG example set, not this
# repository, so it is not this repository's to redistribute; the one test
# that cross-references it stays skippable on a machine without that
# third-party install.
SIX_BEAD_TABLE = Path(
    "/beagle3/gavoth/weizhixue/programs/OpenMSCG/examples/CGYaml/"
    "Example1_3-component_membrane/DPPC/DPPC_mapping.txt"
)

# The four DPPC blocks of NVT.gro, measured from the file.
DPPC_BLOCK_STARTS = (0, 298590, 597180, 895770)
LIPIDS_PER_BLOCK = 1152
ATOMS_PER_LIPID = 130


def _dppc_monomer():
    """One real DPPC molecule from the shipped all-atom fixture.

    Byte-equivalent in atom order, names and masses to the archive's
    ``DPPC_monomer.gro``, which is what the mapping index lists are defined
    against — so the assertions below are the same assertions, on data the
    repository can ship.
    """
    return dppc_aa_universe().residues[0].atoms


def _shipped_cg_topology():
    """The Martini-12 CG topology, read from the shipped mapping.

    ``parse_gromacs_itp`` has its own dedicated tests; here the CG topology is
    an *input*, so taking it from the shipped mapping (which was generated from
    the itp by this project's own builder) covers the same behaviour without
    redistributing the itp.
    """
    return yaml.safe_load(SHIPPED_MAP.read_text())["cg-topology"]


# ─── bead tables ──────────────────────────────────────────────────────


def test_index_table_round_trips_a_repeated_bead_name():
    table = parse_bead_table("A\n[2, 0, 1]\nB\n[3, 4]\nA\n[5, 6, 7]\n")
    assert table == [("A", (0, 1, 2)), ("B", (3, 4)), ("A", (5, 6, 7))]


def test_index_table_rejects_a_dangling_name_or_list():
    with pytest.raises(ValueError, match="no bead name above it"):
        parse_bead_table("[0, 1]\n")
    with pytest.raises(ValueError, match="not followed by an index list"):
        parse_bead_table("A\nB\n[0]\n")
    with pytest.raises(ValueError, match="has no index list"):
        parse_bead_table("A\n[0]\nB\n")
    with pytest.raises(ValueError, match="empty"):
        parse_bead_table("# only a comment\n")


@pytest.mark.skipif(not SIX_BEAD_TABLE.is_file(), reason="OpenMSCG examples not readable")
def test_the_real_six_bead_dppc_table_parses_with_two_repeated_beads():
    table = parse_bead_table(SIX_BEAD_TABLE)
    assert [bead for bead, _ in table] == ["PHG", "PMG", "PT1", "PT2", "PT1", "PT2"]
    assert [len(local) for _, local in table] == [24, 20, 21, 22, 21, 22]
    assert sum(len(local) for _, local in table) == ATOMS_PER_LIPID
    # Contiguous, non-overlapping cover of the residue.
    covered = sorted(index for _, local in table for index in local)
    assert covered == list(range(ATOMS_PER_LIPID))


def test_name_table_attaches_hydrogens_to_the_preceding_heavy_atom():
    names = ["C1", "H1A", "H1B", "O1", "C2", "H2A"]
    masses = [12.0, 1.0, 1.0, 16.0, 12.0, 1.0]
    table = bead_table_from_names(
        names, {"HEAD": ["C1", "O1"], "TAIL": ["C2"]}, masses=masses, attach_hydrogens=True
    )
    assert table == [("HEAD", (0, 1, 2, 3)), ("TAIL", (4, 5))]


def test_name_table_identifies_hydrogens_by_name_when_masses_are_absent():
    names = ["C1", "HA", "N1", "HN"]
    table = bead_table_from_names(
        names, {"B": ["C1", "N1"]}, attach_hydrogens=True
    )
    assert table == [("B", (0, 1, 2, 3))]


def test_name_table_refuses_to_guess():
    names = ["C1", "H1", "C2"]
    with pytest.raises(ValueError, match="atom 'C2' .*does not have|in no bead"):
        bead_table_from_names(names, {"A": ["C1"]}, attach_hydrogens=True)
    with pytest.raises(ValueError, match="does not have"):
        bead_table_from_names(names, {"A": ["C9"]})
    with pytest.raises(ValueError, match="claimed by both"):
        bead_table_from_names(names, {"A": ["C1"], "B": ["C1", "C2"]}, require_complete=False)
    with pytest.raises(ValueError, match="occurs twice"):
        bead_table_from_names(["C", "C"], {"A": ["C"]}, require_complete=False)


def test_partial_mappings_are_allowed_explicitly():
    table = bead_table_from_names(
        ["C1", "H1", "C2", "H2"], {"A": ["C1"]}, attach_hydrogens=True, require_complete=False
    )
    assert table == [("A", (0, 1))]


# ─── group discovery ──────────────────────────────────────────────────


def tiled_universe(*, blocks=2, lipids=3, atoms_per_lipid=4, waters=5):
    """A ``lipid…water…lipid…water`` layout, like the real DPPC system."""
    resnames: list[str] = []
    atom_counts: list[int] = []
    for _ in range(blocks):
        resnames += ["LIP"] * lipids + ["SOL"] * waters
        atom_counts += [atoms_per_lipid] * lipids + [3] * waters
    n_atoms = sum(atom_counts)
    universe = mda.Universe.empty(
        n_atoms,
        n_residues=len(resnames),
        atom_resindex=np.repeat(np.arange(len(resnames)), atom_counts),
        trajectory=True,
    )
    universe.add_TopologyAttr("resname", resnames)
    universe.add_TopologyAttr("resid", list(range(1, len(resnames) + 1)))
    universe.add_TopologyAttr(
        "name", [f"A{i % atoms_per_lipid}" for i in range(n_atoms)]
    )
    universe.add_TopologyAttr("mass", np.full(n_atoms, 12.0))
    return universe


def test_a_new_group_starts_at_every_break_in_atom_index_contiguity():
    universe = tiled_universe()
    table = [("H", (0, 1)), ("T", (2, 3))]
    mapping = build_mapping(universe, {"LIP": table})

    assert len(mapping["system"]) == 2
    first, second = mapping["system"]
    assert (first["anchor"], first["repeat"], first["offset"]) == (0, 3, 4)
    # 3 lipids * 4 + 5 waters * 3 = 27 atoms before the second block.
    assert (second["anchor"], second["repeat"], second["offset"]) == (27, 3, 4)

    sites, _, mol_ids = expand_mapping_sites(mapping)
    assert len(sites) == 12
    assert max(mol_ids) + 1 == 6
    # No site may reach into the solvent between the blocks.
    spec = CGMapSpec.from_mapping(mapping, n_atoms=len(universe.atoms))
    lipid_atoms = set(universe.select_atoms("resname LIP").indices.tolist())
    assert set(spec.atom_indices.tolist()) <= lipid_atoms


def test_the_resname_only_rule_would_have_walked_into_the_solvent():
    """What ``cgyaml`` produces for the same universe, and why it is wrong."""
    universe = tiled_universe()
    legacy_repeat = len(universe.select_atoms("resname LIP").residues)  # 6, one group
    legacy = {
        "site-types": {
            "H": {"index": [0, 1], "x-weight": [1.0, 1.0]},
            "T": {"index": [0, 1], "x-weight": [1.0, 1.0]},
        },
        "system": [
            {"anchor": 0, "repeat": legacy_repeat, "offset": 4,
             "sites": [["H", 0], ["T", 2]]}
        ],
    }
    legacy_spec = CGMapSpec.from_mapping(legacy, n_atoms=len(universe.atoms))
    solvent = set(universe.select_atoms("resname SOL").indices.tolist())
    assert set(legacy_spec.atom_indices.tolist()) & solvent

    ours = CGMapSpec.from_mapping(
        build_mapping(universe, {"LIP": [("H", (0, 1)), ("T", (2, 3))]}),
        n_atoms=len(universe.atoms),
    )
    assert not set(ours.atom_indices.tolist()) & solvent
    assert ours.n_sites == legacy_spec.n_sites  # same site count, right atoms


def test_a_contiguous_system_gives_the_single_group_cgyaml_would_give():
    universe = tiled_universe(blocks=1, waters=0)
    mapping = build_mapping(universe, {"LIP": [("H", (0, 1)), ("T", (2, 3))]})
    assert len(mapping["system"]) == 1
    assert mapping["system"][0] == {
        "anchor": 0, "repeat": 3, "offset": 4, "sites": [["H", 0], ["T", 2]]
    }


def test_site_types_are_relative_to_each_beads_own_first_atom():
    universe = tiled_universe(blocks=1, lipids=1, atoms_per_lipid=6, waters=0)
    mapping = build_mapping(universe, {"LIP": [("X", (1, 3)), ("Y", (4, 5))]})
    # Anchors carry the absolute offset; index lists start at 0.
    assert mapping["system"][0]["sites"] == [["X", 1], ["Y", 4]]
    assert mapping["site-types"]["X"]["index"] == [0, 2]
    assert mapping["site-types"]["Y"]["index"] == [0, 1]


def test_x_weight_defaults_to_mass_and_ones_is_opt_in():
    universe = tiled_universe(blocks=1, lipids=1, waters=0)
    universe.atoms.masses = [12.0, 1.0, 16.0, 1.0]
    table = [("H", (0, 1)), ("T", (2, 3))]
    by_mass = build_mapping(universe, {"LIP": table})
    assert by_mass["site-types"]["H"]["x-weight"] == [12.0, 1.0]
    assert by_mass["site-types"]["T"]["x-weight"] == [16.0, 1.0]
    geometric = build_mapping(universe, {"LIP": table}, x_weight="ones")
    assert geometric["site-types"]["T"]["x-weight"] == [1.0, 1.0]
    assert geometric["site-types"]["T"]["f-weight"] == [1.0, 1.0]


def test_inconsistent_reuse_of_one_bead_name_is_rejected():
    universe = tiled_universe(blocks=1, lipids=1, atoms_per_lipid=6, waters=0)
    universe.atoms.masses = [12.0, 1.0, 1.0, 12.0, 1.0, 16.0]
    # Same name, different shape: (0,1,2) vs (3,4,5) → different weights.
    with pytest.raises(ValueError, match="defined inconsistently"):
        build_mapping(universe, {"LIP": [("T", (0, 1, 2)), ("T", (3, 4, 5))]})


def test_a_table_that_overruns_its_residue_is_rejected():
    universe = tiled_universe(blocks=1, lipids=1, waters=0)
    with pytest.raises(ValueError, match="only 4 atoms"):
        build_mapping(universe, {"LIP": [("H", (0, 9))]})


def test_an_unknown_residue_name_is_rejected():
    universe = tiled_universe(blocks=1, waters=0)
    with pytest.raises(ValueError, match="no residue named"):
        build_mapping(universe, {"CHOL": [("X", (0,))]})


def test_write_mapping_yaml_round_trips_through_the_compiler(tmp_path):
    universe = tiled_universe()
    mapping = build_mapping(
        universe,
        {"LIP": [("H", (0, 1)), ("T", (2, 3))]},
        cg_topology={"molecule": {"names": ["H", "T"], "bonds": [[0, 1]]}},
    )
    path = write_mapping_yaml(mapping, tmp_path / "map.yaml")
    reloaded = yaml.safe_load(path.read_text())
    spec = CGMapSpec.from_mapping(reloaded, n_atoms=len(universe.atoms))
    assert spec.n_sites == 12
    assert spec.bonds.shape == (6, 2)
    assert spec.site_labels == ("H", "T") * 6


# ─── GROMACS itp ──────────────────────────────────────────────────────


def test_martini_itp_becomes_a_twelve_site_cg_topology_block():
    block = parse_gromacs_itp(MARTINI_ITP)["molecule"]
    assert block["names"] == [
        "NC3", "PO4", "GL1", "GL2",
        "C1A", "C2A", "C3A", "C4A",
        "C1B", "C2B", "C3B", "C4B",
    ]
    assert block["charges"] == [1.0, -1.0] + [0.0] * 10
    assert block["itp_masses"] == [72.0] * 12
    # 1-based itp indices become 0-based site positions.
    assert block["bonds"][0] == [0, 1]
    assert block["bonds"][3] == [2, 4]  # GL1-C1A
    assert len(block["bonds"]) == 11
    assert block["angles"][0] == [0, 1, 2]
    assert len(block["angles"]) == 11
    assert "dihedrals" not in block


def test_itp_bonded_terms_survive_compilation_into_interaction_keys():
    """The CG topology's bonded terms must reach the compiled spec.

    Reads the CG topology from the shipped mapping rather than re-parsing the
    Martini itp, so this runs everywhere; `parse_gromacs_itp` itself is checked
    against the real itp above, where that file is available.
    """
    universe = tiled_universe(blocks=1, lipids=2, atoms_per_lipid=12, waters=0)
    table = [(f"S{i}", (i,)) for i in range(12)]
    mapping = build_mapping(
        universe, {"LIP": table}, cg_topology=_shipped_cg_topology()
    )
    spec = CGMapSpec.from_mapping(mapping, n_atoms=len(universe.atoms))
    assert spec.bonds.shape == (22, 2)
    assert spec.angles.shape == (22, 3)
    assert spec.site_labels[:4] == ("NC3", "PO4", "GL1", "GL2")
    # The itp's own charges reach the sites.
    assert spec.site_charges_array()[:2] == pytest.approx([1.0, -1.0])
    # Distinct site types → one bond type per distinct type pair.
    assert InteractionKey.bond("S0", "S1") in spec.bond_type_keys


def test_itp_parser_rejects_out_of_range_and_empty_files(tmp_path):
    bad = tmp_path / "bad.itp"
    bad.write_text("[ moleculetype ]\nX 1\n[ atoms ]\n1 Q0 1 X A 1\n[ bonds ]\n1 5 1\n")
    with pytest.raises(ValueError, match="outside 1..1"):
        parse_gromacs_itp(bad)

    empty = tmp_path / "empty.itp"
    empty.write_text("; nothing here\n")
    with pytest.raises(ValueError, match="no \\[ moleculetype \\]"):
        parse_gromacs_itp(empty)

    no_atoms = tmp_path / "no_atoms.itp"
    no_atoms.write_text("[ moleculetype ]\nX 1\n[ bonds ]\n1 2 1\n")
    with pytest.raises(ValueError, match="no \\[ atoms \\] section"):
        parse_gromacs_itp(no_atoms)


def test_itp_parser_selects_a_named_moleculetype(tmp_path):
    path = tmp_path / "two.itp"
    path.write_text(
        "[ moleculetype ]\nAAA 1\n[ atoms ]\n1 Q0 1 AAA X 1 0.0 72.0\n"
        "[ moleculetype ]\nBBB 1\n[ atoms ]\n1 Q0 1 BBB Y 1 0.0 72.0\n"
        "2 Q0 1 BBB Z 2 0.0 72.0\n[ bonds ]\n1 2 1\n"
    )
    assert parse_gromacs_itp(path)["molecule"]["names"] == ["X"]
    picked = parse_gromacs_itp(path, molecule="BBB")["molecule"]
    assert picked["names"] == ["Y", "Z"] and picked["bonds"] == [[0, 1]]
    with pytest.raises(ValueError, match="no moleculetype 'CCC'"):
        parse_gromacs_itp(path, molecule="CCC")


# ─── the real DPPC residue ────────────────────────────────────────────


@pytest.mark.skipif(
    not SIX_BEAD_TABLE.is_file(),
    reason=(
        "cross-reference against the OpenMSCG example bead table, a "
        "third-party file this repository may not redistribute, so this "
        "check runs only where a private copy is readable. map.yaml is a "
        "shipped fixture (tests/test_data/dppc_martini12/) and no longer "
        "gates this test. The builder itself is covered unconditionally by "
        "the Martini-12 tests on the shipped tests/test_data/dppc_aa/ "
        "fixture."
    ),
)
def test_six_bead_site_types_reproduce_the_archive_map_yaml():
    """Our builder, the archive's bead table, the archive's own map.yaml."""
    monomer = _dppc_monomer()
    table = parse_bead_table(SIX_BEAD_TABLE)
    mapping = build_mapping(monomer, {"DPPC": table})
    archive = yaml.safe_load(ARCHIVE_MAP.read_text())

    # The archive numbers its site types 1..4 in first-appearance order.
    ours = list(mapping["site-types"])
    assert ours == ["PHG", "PMG", "PT1", "PT2"]
    for our_name, archive_key in zip(ours, [1, 2, 3, 4]):
        assert mapping["site-types"][our_name]["index"] == archive["site-types"][archive_key]["index"]
        # The archive's weights are rounded (H 1.0, C 12.0, N 14.0, O 16.0,
        # P 30.9) while MDAnalysis supplies element masses (H 1.008, P 30.974),
        # so this is agreement in the mapping, not byte equality of the weights.
        # Reproducing that CG reference bit-for-bit would mean feeding those
        # rounded weights back in, not deriving them from the topology.
        assert mapping["site-types"][our_name]["x-weight"] == pytest.approx(
            archive["site-types"][archive_key]["x-weight"], rel=1e-2
        )
    assert mapping["system"][0]["sites"] == [
        ["PHG", 0], ["PMG", 24], ["PT1", 44], ["PT2", 65], ["PT1", 87], ["PT2", 108]
    ]
    assert mapping["system"][0]["offset"] == ATOMS_PER_LIPID


def test_martini_twelve_bead_table_covers_the_charmm_residue_exactly():
    monomer = _dppc_monomer()
    assignment = yaml.safe_load(MARTINI_BEADS.read_text())
    table = bead_table_from_names(
        monomer.atoms.names,
        assignment,
        masses=monomer.atoms.masses,
        attach_hydrogens=True,
    )

    assert [bead for bead, _ in table] == list(assignment)
    counts = [len(local) for _, local in table]
    assert counts == [19, 5, 8, 6, 12, 12, 12, 10, 12, 12, 12, 10]
    assert sum(counts) == ATOMS_PER_LIPID
    covered = sorted(index for _, local in table for index in local)
    assert covered == list(range(ATOMS_PER_LIPID))
    # 50 heavy atoms named, 80 hydrogens attached automatically.
    assert sum(len(atoms) for atoms in assignment.values()) == 50

    mapping = build_mapping(monomer, {"DPPC": table}, cg_topology=_shipped_cg_topology())
    spec = CGMapSpec.from_mapping(mapping, n_atoms=ATOMS_PER_LIPID)
    assert spec.n_sites == 12
    assert spec.site_labels == tuple(assignment)
    # Mapped site masses: choline ester + phosphate + glycerol + four tails.
    masses = spec.site_masses_array()
    assert masses.sum() == pytest.approx(float(monomer.atoms.masses.sum()), rel=1e-9)
    assert masses[0] == pytest.approx(87.166, abs=0.01)   # NC3
    assert masses[1] == pytest.approx(94.971, abs=0.01)   # PO4
    assert masses[4] == pytest.approx(56.108, abs=0.01)   # C1A, four CH2
    assert masses[7] == pytest.approx(43.089, abs=0.01)   # C4A, CH2 CH2 CH3
    # C1A and C1B are deliberately non-contiguous index lists.
    c1a = dict(table)["C1A"]
    assert c1a[:3] == (32, 33, 34) and c1a[3] == 44


def test_martini_mapping_for_the_full_tiled_system_has_four_groups(tmp_path):
    """Group anchors must land on the measured DPPC block starts of NVT.gro."""
    monomer = _dppc_monomer()
    assignment = yaml.safe_load(MARTINI_BEADS.read_text())
    table = bead_table_from_names(
        monomer.atoms.names, assignment, masses=monomer.atoms.masses, attach_hydrogens=True
    )
    # A stand-in for NVT.gro with the same block structure and sizes, so the test
    # stays cheap: the real 82 MB topology is only read by the production run.
    resnames: list[str] = []
    counts: list[int] = []
    for _ in range(4):
        resnames += ["DPPC"] * LIPIDS_PER_BLOCK + ["SOD"] * 111 + ["CLA"] * 111
        counts += [ATOMS_PER_LIPID] * LIPIDS_PER_BLOCK + [1] * 222
        resnames += ["TIP3"] * 49536
        counts += [3] * 49536
    universe = mda.Universe.empty(
        sum(counts),
        n_residues=len(resnames),
        atom_resindex=np.repeat(np.arange(len(resnames)), counts),
        trajectory=True,
    )
    universe.add_TopologyAttr("resname", resnames)
    universe.add_TopologyAttr("resid", list(range(1, len(resnames) + 1)))
    tiled_masses = np.tile(np.asarray(monomer.atoms.masses, dtype=np.float64), LIPIDS_PER_BLOCK)
    masses = np.concatenate(
        [np.concatenate([tiled_masses, np.full(222 + 3 * 49536, 16.0)])] * 4
    )
    universe.add_TopologyAttr("mass", masses)
    universe.add_TopologyAttr("name", list(monomer.atoms.names) * (LIPIDS_PER_BLOCK * 4)
                              + ["X"] * (sum(counts) - ATOMS_PER_LIPID * LIPIDS_PER_BLOCK * 4))

    mapping = build_mapping(universe, {"DPPC": table}, cg_topology=_shipped_cg_topology())
    assert [group["anchor"] for group in mapping["system"]] == list(DPPC_BLOCK_STARTS)
    assert [group["repeat"] for group in mapping["system"]] == [LIPIDS_PER_BLOCK] * 4
    assert all(group["offset"] == ATOMS_PER_LIPID for group in mapping["system"])

    spec = CGMapSpec.from_mapping(mapping, n_atoms=sum(counts))
    assert spec.n_sites == 12 * LIPIDS_PER_BLOCK * 4 == 55296
    assert spec.n_mol == LIPIDS_PER_BLOCK * 4
    assert spec.n_required_atoms == 599040
    assert spec.molecules_contiguous
    assert spec.bonds.shape == (11 * 4608, 2)
    assert spec.angles.shape == (11 * 4608, 3)
    lipid_atoms = set(universe.select_atoms("resname DPPC").indices.tolist())
    assert set(spec.atom_indices.tolist()) == lipid_atoms
