"""Unit tests for :mod:`AceCG.topology.vpgrower`.

The fixture is a minimal 2-residue × 4-bead CG universe with a linear
chain topology (``HG-MG-T1-T2`` per residue) used to exercise template
construction, bond/angle index insertion, name resolution, and
per-frame growth.
"""

from __future__ import annotations

import math
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from AceCG.configs.vp_config import (
    VPAtomDef,
    VPConfig,
    VPInteractionDef,
)
from AceCG.topology.vpgrower import (
    VPGrownFrame,
    VPGrower,
    VPTopologyTemplate,
    write_vp_data,
)


# ─── Fixtures ───────────────────────────────────────────────────────


def _build_cg_universe(
    *,
    name_mode: str = "aliases",
) -> mda.Universe:
    """Build a 2-residue × 4-bead CG universe (no VP atoms).

    ``name_mode``:
      - ``"aliases"``  → atom ``types`` are bare LAMMPS ids ``"1".."4"``
        so the caller must supply ``type_aliases`` to the grower.
      - ``"names"``    → atom ``types`` are the bead names directly.
    """
    n_atoms = 8
    atom_resindex = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    u = mda.Universe.empty(
        n_atoms, n_residues=2, atom_resindex=atom_resindex, trajectory=True
    )
    if name_mode == "aliases":
        types = np.array(["1", "2", "3", "4"] * 2, dtype=object)
    elif name_mode == "names":
        types = np.array(["HG", "MG", "T1", "T2"] * 2, dtype=object)
    else:
        raise ValueError(name_mode)
    u.add_TopologyAttr("types", types)
    u.add_TopologyAttr("names", types.copy())
    u.add_TopologyAttr("masses", np.array([72.0] * n_atoms, dtype=float))
    u.add_TopologyAttr("charges", np.zeros(n_atoms, dtype=float))
    u.add_TopologyAttr("resids", np.array([1, 2], dtype=np.int64))
    u.add_TopologyAttr("resnames", np.array(["DOPC", "DOPC"], dtype=object))

    # Linear per-residue bonds: HG-MG, MG-T1, T1-T2.
    bonds = np.array(
        [
            (0, 1), (1, 2), (2, 3),
            (4, 5), (5, 6), (6, 7),
        ],
        dtype=np.int64,
    )
    u.add_TopologyAttr("bonds", bonds)
    # One angle per residue (HG-MG-T1) just to exercise angle carry-over.
    angles = np.array(
        [
            (0, 1, 2),
            (4, 5, 6),
        ],
        dtype=np.int64,
    )
    u.add_TopologyAttr("angles", angles)

    # Place each residue linearly along x; shift residue 2 along y to
    # keep them well separated.
    positions = np.array(
        [
            [0.0, 0.0, 0.0],   # res1 HG
            [2.0, 0.0, 0.0],   # res1 MG
            [4.0, 0.0, 0.0],   # res1 T1
            [6.0, 0.0, 0.0],   # res1 T2
            [0.0, 6.0, 0.0],   # res2 HG
            [2.0, 6.0, 0.0],   # res2 MG
            [4.0, 6.0, 0.0],   # res2 T1
            [6.0, 6.0, 0.0],   # res2 T2
        ],
        dtype=float,
    )
    u.atoms.positions = positions
    u.dimensions = np.array([30.0, 30.0, 30.0, 90.0, 90.0, 90.0], dtype=float)
    return u


def _vp_config_harmonic() -> VPConfig:
    return VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=(
            VPInteractionDef(
                type_keys=("VP", "MG"),
                pot_style="harmonic",
                pot_kwargs={"k": 2.5, "r0": 1.5},
            ),
        ),
        angles=(
            VPInteractionDef(
                type_keys=("VP", "MG", "HG"),
                pot_style="harmonic",
                pot_kwargs={"k": 2.45, "theta0": 135.0},
            ),
        ),
        pairs=(),
        selection="resname DOPC",
        atomtype_order="back",
        clash_max_passes=8,
        clash_min_distance=1.5,
    )


# ─── Template assembly tests ───────────────────────────────────────


def test_template_shape_with_aliases() -> None:
    u = _build_cg_universe(name_mode="aliases")
    aliases = {1: "HG", 2: "MG", 3: "T1", 4: "T2"}
    grower = VPGrower.from_universe(u, _vp_config_harmonic(), type_aliases=aliases)
    t = grower.template

    # 2 residues × (4 real + 1 VP) = 10 atoms.
    assert t.n_atoms == 10
    assert t.n_real == 8
    assert t.n_vp == 2
    # VPs are inserted after each residue's last real atom.
    assert list(t.vp_indices_by_name["VP"]) == [4, 9]
    # real_indices skip those VP slots.
    assert list(t.real_indices) == [0, 1, 2, 3, 5, 6, 7, 8]
    assert t.atom_names[4] == "VP"
    assert t.atom_names[9] == "VP"
    assert t.atom_names[0] == "HG"
    assert t.atom_names[1] == "MG"


def test_template_type2id_back_order() -> None:
    u = _build_cg_universe(name_mode="aliases")
    grower = VPGrower.from_universe(
        u, _vp_config_harmonic(),
        type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )
    # ``atomtype_order = "back"`` ⇒ VP gets the last id.
    ids = grower.template.type2id
    assert ids == {"HG": 1, "MG": 2, "T1": 3, "T2": 4, "VP": 5}


def test_template_type2id_front_order() -> None:
    u = _build_cg_universe(name_mode="aliases")
    cfg = _vp_config_harmonic()
    cfg = VPConfig(
        atoms=cfg.atoms, bonds=cfg.bonds, angles=cfg.angles, pairs=cfg.pairs,
        default_pair=cfg.default_pair, selection=cfg.selection,
        atomtype_order="front", clash_max_passes=cfg.clash_max_passes,
        clash_min_distance=cfg.clash_min_distance,
    )
    grower = VPGrower.from_universe(
        u, cfg, type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )
    assert grower.template.type2id == {"VP": 1, "HG": 2, "MG": 3, "T1": 4, "T2": 5}


def test_template_inserts_vp_bonds_and_angles() -> None:
    u = _build_cg_universe(name_mode="aliases")
    grower = VPGrower.from_universe(
        u, _vp_config_harmonic(),
        type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )
    t = grower.template

    # 6 real bonds + 2 inserted VP-MG bonds = 8 rows.
    assert t.bonds.shape == (8, 2)
    # 2 real angles + 2 inserted VP-MG-HG angles = 4 rows.
    assert t.angles.shape == (4, 3)
    # VP-MG bond endpoints touch VP indices 4 / 9.
    vp_atoms_touched = {int(row[0]) if t.atom_names[int(row[0])] == "VP" else int(row[1])
                        for row in t.bonds.tolist()
                        if "VP" in (t.atom_names[int(row[0])], t.atom_names[int(row[1])])}
    assert vp_atoms_touched == {4, 9}

    # Angle rows that contain a VP must have that VP as the first label
    # ("VP","MG","HG" in spec). ``InteractionKey`` canonicalization flips
    # the order, so check on raw bond/angle tables instead.
    angle_rows = t.angles.tolist()
    vp_angle_rows = [row for row in angle_rows if any(t.atom_names[int(j)] == "VP" for j in row)]
    assert len(vp_angle_rows) == 2
    for row in vp_angle_rows:
        names = [t.atom_names[int(j)] for j in row]
        assert names == ["VP", "MG", "HG"]


def test_template_bond_angle_type_ids_canonicalize() -> None:
    u = _build_cg_universe(name_mode="aliases")
    grower = VPGrower.from_universe(
        u, _vp_config_harmonic(),
        type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )
    t = grower.template
    # Two distinct bond types from real bonds (HG-MG, MG-T1, T1-T2) plus
    # one VP-MG type.
    assert set(t.bond_type_ids.tolist()) == {1, 2, 3, 4}
    # Angles: real HG-MG-T1 + VP-MG-HG.
    assert set(t.angle_type_ids.tolist()) == {1, 2}


def test_template_without_aliases_uses_bare_ids() -> None:
    u = _build_cg_universe(name_mode="aliases")
    cfg = VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=(
            VPInteractionDef(
                type_keys=("VP", "2"),  # bare "MG" = LAMMPS id 2
                pot_style="harmonic",
                pot_kwargs={"k": 2.5, "r0": 1.5},
            ),
        ),
        angles=(),
        pairs=(),
        selection=None,
        atomtype_order="back",
    )
    grower = VPGrower.from_universe(u, cfg)
    # ``"2"`` → matches atom type name "2".
    t = grower.template
    assert "2" in t.type2id
    assert t.bonds.shape[0] == 6 + 2  # 6 real + 2 inserted


def test_unknown_alias_label_raises() -> None:
    u = _build_cg_universe(name_mode="aliases")
    cfg = VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=(
            VPInteractionDef(
                type_keys=("VP", "UNKNOWN"),
                pot_style="harmonic",
                pot_kwargs={"k": 2.5, "r0": 1.5},
            ),
        ),
        angles=(),
        pairs=(),
    )
    with pytest.raises(ValueError, match=r"cannot resolve VP label"):
        VPGrower.from_universe(
            u, cfg, type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
        )


def test_vp_type_label_must_not_collide_with_real_atom_names() -> None:
    u = _build_cg_universe(name_mode="names")
    cfg = VPConfig(
        atoms=(VPAtomDef(type_label="HG", mass=72.0),),
        bonds=(),
        angles=(),
        pairs=(),
    )
    with pytest.raises(ValueError, match=r"must not collide with real atom names"):
        VPGrower.from_universe(u, cfg)


def test_duplicate_vp_type_labels_raise() -> None:
    u = _build_cg_universe(name_mode="names")
    cfg = VPConfig(
        atoms=(
            VPAtomDef(type_label="VP", mass=72.0),
            VPAtomDef(type_label="VP", mass=73.0),
        ),
        bonds=(),
        angles=(),
        pairs=(),
    )
    with pytest.raises(ValueError, match=r"duplicate VP type_label"):
        VPGrower.from_universe(u, cfg)


# ─── grow_frame tests ───────────────────────────────────────────────


def test_grow_frame_positions_real_aligned() -> None:
    u = _build_cg_universe(name_mode="aliases")
    grower = VPGrower.from_universe(
        u, _vp_config_harmonic(),
        type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )
    real_positions = np.asarray(u.atoms.positions, dtype=float)
    frame = grower.grow_frame(
        real_positions,
        np.asarray(u.dimensions, dtype=float),
        orientation_seed=42,
    )
    assert isinstance(frame, VPGrownFrame)
    assert frame.positions.shape == (10, 3)
    # Real slots equal input.
    np.testing.assert_allclose(
        frame.positions[grower.template.real_indices], real_positions
    )
    # VP positions are finite.
    vp_idx = grower.template.vp_indices_by_name["VP"]
    assert np.all(np.isfinite(frame.positions[vp_idx]))


def test_grow_frame_bond_length_within_tolerance() -> None:
    u = _build_cg_universe(name_mode="aliases")
    grower = VPGrower.from_universe(
        u, _vp_config_harmonic(),
        type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )
    real_positions = np.asarray(u.atoms.positions, dtype=float)
    frame = grower.grow_frame(
        real_positions,
        np.asarray(u.dimensions, dtype=float),
        orientation_seed=7,
    )

    # VP atoms and their MG partners come from the inserted VP bonds;
    # their distance must match r0 = 1.5 before the anti-clash pass
    # (which here should be a no-op since VP is far from all reals).
    t = grower.template
    for row in t.bonds.tolist():
        names = [t.atom_names[int(j)] for j in row]
        if "VP" in names:
            i, j = int(row[0]), int(row[1])
            d = float(np.linalg.norm(frame.positions[i] - frame.positions[j]))
            # Clash pass can move VP if MG sits inside clash threshold;
            # on this fixture MG-residue neighbours are > 1.5 Å away
            # so bond length should match r0 tightly.
            assert abs(d - 1.5) < 1e-2


def test_grow_frame_determinism_with_seed() -> None:
    u = _build_cg_universe(name_mode="aliases")
    grower = VPGrower.from_universe(
        u, _vp_config_harmonic(),
        type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )
    real_positions = np.asarray(u.atoms.positions, dtype=float)
    f1 = grower.grow_frame(real_positions, u.dimensions, orientation_seed=1)
    f2 = grower.grow_frame(real_positions, u.dimensions, orientation_seed=1)
    np.testing.assert_allclose(f1.positions, f2.positions)
    f3 = grower.grow_frame(real_positions, u.dimensions, orientation_seed=2)
    assert not np.allclose(f1.positions, f3.positions)


def test_grow_frame_selection_restricts_vp(tmp_path: Path) -> None:
    u = _build_cg_universe(name_mode="aliases")
    cfg = _vp_config_harmonic()
    # Restrict to just residue 1.
    cfg = VPConfig(
        atoms=cfg.atoms, bonds=cfg.bonds, angles=cfg.angles, pairs=cfg.pairs,
        default_pair=cfg.default_pair, selection="resid 1",
        atomtype_order=cfg.atomtype_order, clash_max_passes=cfg.clash_max_passes,
        clash_min_distance=cfg.clash_min_distance,
    )
    grower = VPGrower.from_universe(
        u, cfg, type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )
    assert grower.template.n_vp == 1
    assert grower.template.carrier_resids == (1,)


def test_write_vp_data_emits_parseable_file(tmp_path: Path) -> None:
    u = _build_cg_universe(name_mode="aliases")
    grower = VPGrower.from_universe(
        u, _vp_config_harmonic(),
        type_aliases={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
    )
    frame = grower.grow_frame(
        np.asarray(u.atoms.positions, dtype=float),
        np.asarray(u.dimensions, dtype=float),
        orientation_seed=11,
    )
    out = tmp_path / "grown.data"
    write_vp_data(grower.template, frame, out, title="test")
    text = out.read_text()
    assert "Atoms" in text
    assert "Bonds" in text
    assert "Angles" in text
    # Re-read through MDAnalysis for structural consistency.
    u2 = mda.Universe(
        str(out), format="DATA",
        atom_style="id resid type charge x y z",
    )
    assert len(u2.atoms) == 10
    assert len(u2.bonds) == 8
    assert len(u2.angles) == 4
