"""Tests for ``io.vp_ffbuilder``.

Covers
------
1. :func:`build_vp_forcefield` returns the expected
   :class:`InteractionKey`-keyed potentials with correct angle scaling.
2. :func:`render_vp_latent_template` emits parseable LAMMPS lines.
3. :func:`write_latent_settings` round-trip: emits a ``latent.settings``
   file plus initial ``Pair_*.table`` tables that :class:`WriteLmpFF`
   finalizes.
4. Angle scale diligence: ``HarmonicPotential.value(theta0_deg)`` at the
   VP-side configured ``theta0_deg`` is (near-)zero when ``scale=(π/180)²``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from AceCG.configs.vp_config import VPAtomDef, VPConfig, VPInteractionDef
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.topology.types import InteractionKey
from AceCG.topology.vpgrower import VPTopologyTemplate, VPBondSpec, VPAngleSpec
from AceCG.io.vp_ffbuilder import (
    _is_table,
    build_vp_forcefield,
    render_vp_latent_template,
    write_latent_settings,
)
from AceCG.io.tables import parse_lammps_table
from AceCG.potentials.soft import SoftPotential


# ─── Helpers ────────────────────────────────────────────────────────


def _make_template() -> VPTopologyTemplate:
    """Minimal DOPC-like template: HG, MG, T1, T2 plus VP (appended)."""
    atom_names = ("HG", "MG", "T1", "T2", "VP")
    atom_masses = np.array([100.0, 70.0, 60.0, 60.0, 72.0])
    resids = np.array([1, 1, 1, 1, 1])
    resnames = np.array(["DOPC", "DOPC", "DOPC", "DOPC", "DOPC"])
    real_indices = np.array([0, 1, 2, 3])
    type2id = {"HG": 1, "MG": 2, "T1": 3, "T2": 4, "VP": 5}
    type_masses = {1: 100.0, 2: 70.0, 3: 60.0, 4: 60.0, 5: 72.0}
    bonds = np.array([[0, 1], [1, 2], [2, 3], [4, 1]])
    angles = np.array([[0, 1, 2], [1, 2, 3], [0, 1, 4]])
    bond_type_ids = np.array([1, 2, 3, 4])
    angle_type_ids = np.array([1, 2, 3])
    bond_type_key_by_id = {
        1: InteractionKey.bond("HG", "MG"),
        2: InteractionKey.bond("MG", "T1"),
        3: InteractionKey.bond("T1", "T2"),
        4: InteractionKey.bond("MG", "VP"),
    }
    angle_type_key_by_id = {
        1: InteractionKey.angle("HG", "MG", "T1"),
        2: InteractionKey.angle("MG", "T1", "T2"),
        3: InteractionKey.angle("HG", "MG", "VP"),
    }
    vp_bond_specs = (VPBondSpec(vp_name="VP", carrier_name="MG", k=2.50, r0=1.50),)
    vp_angle_specs = (
        VPAngleSpec(vp_name="VP", labels=("HG", "MG", "VP"), k=2.45, theta0_deg=135.0),
    )
    return VPTopologyTemplate(
        atom_names=atom_names,
        atom_masses=atom_masses,
        resids=resids,
        resnames=resnames,
        real_indices=real_indices,
        vp_indices_by_name={"VP": np.array([4])},
        carrier_resids=np.array([1]),
        type2id=type2id,
        type_masses=type_masses,
        bonds=bonds,
        angles=angles,
        dihedrals=np.empty((0, 4), dtype=np.int64),
        bond_type_ids=bond_type_ids,
        angle_type_ids=angle_type_ids,
        dihedral_type_ids=np.empty((0,), dtype=np.int64),
        bond_type_key_by_id=bond_type_key_by_id,
        angle_type_key_by_id=angle_type_key_by_id,
        dihedral_type_key_by_id={},
        vp_bond_specs=vp_bond_specs,
        vp_angle_specs=vp_angle_specs,
        clash_max_passes=8,
        clash_min_distance=1.5,
    )


def _make_vp_config() -> VPConfig:
    return VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=(
            VPInteractionDef(
                type_keys=("VP", "MG"), pot_style="harmonic",
                pot_kwargs={"k": 2.50, "r0": 1.50},
            ),
        ),
        angles=(
            VPInteractionDef(
                type_keys=("VP", "MG", "HG"), pot_style="harmonic",
                pot_kwargs={"k": 2.45, "theta0": 135.0},
            ),
        ),
        pairs=(
            VPInteractionDef(
                type_keys=("VP", "HG"), pot_style="table",
                pot_kwargs={"file": "Pair_VP-HG.table", "cutoff": 25.0},
            ),
            VPInteractionDef(
                type_keys=("VP", "MG"), pot_style="table",
                pot_kwargs={"file": "Pair_VP-MG.table", "cutoff": 25.0},
            ),
            VPInteractionDef(
                type_keys=("VP", "T1"), pot_style="table",
                pot_kwargs={"file": "Pair_VP-T1.table", "cutoff": 25.0},
            ),
            VPInteractionDef(
                type_keys=("VP", "T2"), pot_style="table",
                pot_kwargs={"file": "Pair_VP-T2.table", "cutoff": 25.0},
            ),
            VPInteractionDef(
                type_keys=("VP", "VP"), pot_style="table",
                pot_kwargs={"file": "Pair_VP-VP.table", "cutoff": 25.0},
            ),
        ),
        selection="resname DOPC",
        atomtype_order="back",
        clash_max_passes=8,
        clash_min_distance=1.5,
    )


def _make_astable_template() -> VPTopologyTemplate:
    base = _make_template()
    return replace(
        base,
        vp_bond_specs=(
            VPBondSpec(vp_name="VP", carrier_name="MG", k=2.50, r0=1.50, astable=True),
        ),
        vp_angle_specs=(
            VPAngleSpec(vp_name="VP", labels=("VP", "HG", "MG"), k=2.45, theta0_deg=135.0, astable=True),
        ),
        angle_type_key_by_id={
            1: InteractionKey.angle("HG", "MG", "T1"),
            2: InteractionKey.angle("MG", "T1", "T2"),
            3: InteractionKey.angle("MG", "HG", "VP"),
        },
    )


def _make_astable_vp_config() -> VPConfig:
    return VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=(
            VPInteractionDef(
                type_keys=("VP", "MG"), pot_style="harmonic",
                pot_kwargs={"k": 2.50, "r0": 1.50, "astable": True},
            ),
        ),
        angles=(
            VPInteractionDef(
                type_keys=("VP", "HG", "MG"), pot_style="harmonic",
                pot_kwargs={"k": 2.45, "theta0": 135.0, "astable": True},
            ),
        ),
        pairs=(),
        selection="resname DOPC",
        atomtype_order="back",
    )


# ─── Tests ──────────────────────────────────────────────────────────


def test_build_vp_forcefield_has_bond_angle_pair_keys():
    template = _make_template()
    cfg = _make_vp_config()
    ff = build_vp_forcefield(cfg, template)

    # Bond key present, correct HarmonicPotential values.
    bkey = InteractionKey.bond("VP", "MG")
    assert bkey in ff
    bp = ff[bkey][0]
    assert isinstance(bp, HarmonicPotential)
    params = list(bp.get_params())
    assert params[0] == pytest.approx(2.50)
    assert params[1] == pytest.approx(1.50)

    # Angle key present, scaled correctly.
    akey = InteractionKey.angle("HG", "MG", "VP")
    assert akey in ff
    ap = ff[akey][0]
    assert isinstance(ap, HarmonicPotential)
    # At the equilibrium angle (degrees) the harmonic energy is zero.
    assert ap.value(np.array([135.0]))[0] == pytest.approx(0.0, abs=1e-10)
    # Slight displacement gives positive energy.
    assert ap.value(np.array([140.0]))[0] > 0.0

    # All five VP pair keys present.
    for other in ("HG", "MG", "T1", "T2", "VP"):
        assert InteractionKey.pair("VP", other) in ff


def test_harmonic_angle_scale_is_degrees_squared():
    """Diligence: the convention is that harmonic angle ``k`` is stored in
    ``energy/rad²`` and ``HarmonicPotential.value(theta_deg)`` works
    when ``scale=(π/180)²``.
    """
    k = 2.45
    theta0 = 135.0
    scale = (np.pi / 180.0) ** 2
    pot = HarmonicPotential("HG", "VP", k, theta0, typ3="MG", scale=scale)
    # Zero at equilibrium.
    assert pot.value(np.array([theta0]))[0] == pytest.approx(0.0, abs=1e-12)
    # Expected energy at theta0 + 10 deg: k * (10*π/180)²
    # (HarmonicPotential convention: U = k*scale*(θ-θ0)² with no 1/2 factor.)
    expected = k * (10.0 * np.pi / 180.0) ** 2
    got = pot.value(np.array([theta0 + 10.0]))[0]
    assert got == pytest.approx(expected, rel=1e-6)


def test_render_vp_latent_template_shape():
    template = _make_template()
    cfg = _make_vp_config()
    text = render_vp_latent_template(cfg, template, default_cutoff=25.0)
    lines = [l for l in text.splitlines() if l.strip()]
    # Five pair_coeff lines (VP-HG, VP-MG, VP-T1, VP-T2, VP-VP).
    pair_lines = [l for l in lines if "pair_coeff" in l]
    assert len(pair_lines) == 5
    # All pair lines reference type id 5 (VP).
    for l in pair_lines:
        toks = l.split()
        assert "5" in toks[1:3]
        assert toks[3] == "table"
    # One bond_coeff line for bond type id 4.
    bond_lines = [l for l in lines if "bond_coeff" in l]
    assert len(bond_lines) == 1
    assert bond_lines[0].split()[1] == "4"
    # One angle_coeff line for angle type id 3.
    angle_lines = [l for l in lines if "angle_coeff" in l]
    assert len(angle_lines) == 1
    assert angle_lines[0].split()[1] == "3"


def test_render_vp_latent_template_bond_angle_astable():
    text = render_vp_latent_template(
        _make_astable_vp_config(), _make_astable_template(), default_cutoff=180.0,
    )
    assert "bond_coeff   4  table  VP_MG_bon.table VP_MG_bon" in text
    assert "angle_coeff  3  table  VP_HG_MG_ang.table VP_HG_MG_ang" in text


def test_write_latent_settings_round_trip(tmp_path):
    template = _make_template()
    cfg = _make_vp_config()
    out_path = write_latent_settings(
        template=template, vp_config=cfg, output_dir=tmp_path,
        table_points=101, table_rmin=0.0, table_rmax=25.0,
    )
    assert out_path.exists()
    content = out_path.read_text()

    # Five pair_coeff table lines emitted.
    assert content.count("pair_coeff") == 5
    for nm in ("Pair_VP-HG.table", "Pair_VP-MG.table", "Pair_VP-VP.table"):
        assert nm in content
        assert (tmp_path / nm).exists()

    # bond/angle coeffs rewritten (placeholder 0.0 0.0 replaced).
    bond_line = [l for l in content.splitlines() if l.strip().startswith("bond_coeff")][0]
    bond_toks = bond_line.split()
    # Harmonic → id, 'harmonic', k, r0
    assert bond_toks[2] == "harmonic"
    assert float(bond_toks[3]) == pytest.approx(2.50)
    assert float(bond_toks[4]) == pytest.approx(1.50)

    angle_line = [l for l in content.splitlines() if l.strip().startswith("angle_coeff")][0]
    angle_toks = angle_line.split()
    assert angle_toks[2] == "harmonic"
    assert float(angle_toks[3]) == pytest.approx(2.45)
    assert float(angle_toks[4]) == pytest.approx(135.0)


def test_write_latent_settings_bond_angle_astable_tables(tmp_path):
    out_path = write_latent_settings(
        template=_make_astable_template(),
        vp_config=_make_astable_vp_config(),
        output_dir=tmp_path,
        table_points=101,
        table_rmin=0.0,
        table_rmax=180.0,
    )
    content = out_path.read_text()
    assert "VP_MG_bon.table" in content
    assert "VP_HG_MG_ang.table" in content
    assert (tmp_path / "VP_MG_bon.table").exists()
    assert (tmp_path / "VP_HG_MG_ang.table").exists()


def test_write_latent_settings_angle_table_uses_degree_grid(tmp_path):
    out_path = write_latent_settings(
        template=_make_astable_template(),
        vp_config=_make_astable_vp_config(),
        output_dir=tmp_path,
        table_points=101,
        table_rmin=0.01,
        table_rmax=25.0,
    )
    assert out_path.exists()
    angle_grid, _, _ = parse_lammps_table(tmp_path / "VP_HG_MG_ang.table")
    assert angle_grid[0] == pytest.approx(0.0)
    assert angle_grid[-1] == pytest.approx(180.0)


def test_is_table_detects_astable_kwarg():
    d1 = VPInteractionDef(type_keys=("VP", "HG"), pot_style="table", pot_kwargs={})
    d2 = VPInteractionDef(
        type_keys=("VP", "HG"), pot_style="gauss/cut",
        pot_kwargs={"astable": "yes", "H": 1.0, "rmh": 2.0, "sigma": 1.0, "cutoff": 10.0},
    )
    d3 = VPInteractionDef(
        type_keys=("VP", "HG"), pot_style="lj/cut",
        pot_kwargs={"epsilon": 0.5, "sigma": 3.0, "cutoff": 12.0},
    )
    assert _is_table(d1) is True
    assert _is_table(d2) is True
    assert _is_table(d3) is False


def test_build_vp_forcefield_supports_native_pair_style():
    """A default_pair with a native style (no astable) still produces a
    valid potential for every missing (VP, *) pair."""
    template = _make_template()
    cfg = VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=_make_vp_config().bonds,
        angles=_make_vp_config().angles,
        pairs=(),
        default_pair=VPInteractionDef(
            type_keys=("*", "*"),
            pot_style="lj/cut",
            pot_kwargs={"epsilon": 0.1, "sigma": 3.0, "cutoff": 10.0},
        ),
        selection="resname DOPC",
        atomtype_order="back",
    )
    ff = build_vp_forcefield(cfg, template)
    for other in ("HG", "MG", "T1", "T2", "VP"):
        assert InteractionKey.pair("VP", other) in ff


def test_build_vp_forcefield_supports_soft_default_pair():
    template = _make_template()
    cfg = VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=_make_vp_config().bonds,
        angles=_make_vp_config().angles,
        pairs=(),
        default_pair=VPInteractionDef(
            type_keys=("*", "*"),
            pot_style="soft",
            pot_kwargs={"A": 25.0, "r_c": 5.0, "astable": True},
        ),
        selection="resname DOPC",
        atomtype_order="back",
    )
    ff = build_vp_forcefield(cfg, template)
    pair = ff[InteractionKey.pair("VP", "HG")][0]
    assert isinstance(pair, SoftPotential)
    assert pair.get_params()[0] == pytest.approx(25.0)
    assert pair.get_params()[1] == pytest.approx(5.0)
    assert pair.cutoff == pytest.approx(5.0)


def test_render_vp_latent_template_soft_uses_single_rc_token():
    template = _make_template()
    cfg = VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=_make_vp_config().bonds,
        angles=_make_vp_config().angles,
        pairs=(
            VPInteractionDef(
                type_keys=("VP", "HG"),
                pot_style="soft",
                pot_kwargs={"A": 25.0, "r_c": 5.0},
            ),
        ),
        selection="resname DOPC",
        atomtype_order="back",
    )
    text = render_vp_latent_template(cfg, template, default_cutoff=25.0)
    line = next(ln for ln in text.splitlines() if "pair_coeff" in ln)
    assert line.split()[-3:] == ["soft", "25", "5"]


def test_render_vp_latent_template_keeps_native_pair_cutoff_token():
    template = _make_template()
    cfg = VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=_make_vp_config().bonds,
        angles=_make_vp_config().angles,
        pairs=(
            VPInteractionDef(
                type_keys=("VP", "HG"),
                pot_style="lj/cut",
                pot_kwargs={"epsilon": 0.5, "sigma": 4.0, "rcut": 12.0},
            ),
        ),
        selection="resname DOPC",
        atomtype_order="back",
    )
    text = render_vp_latent_template(cfg, template, default_cutoff=25.0)
    line = next(ln for ln in text.splitlines() if "pair_coeff" in ln)
    assert line.split()[-1] == "12"


def test_native_pair_style_requires_explicit_cutoff():
    template = _make_template()
    cfg = VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=_make_vp_config().bonds,
        angles=_make_vp_config().angles,
        pairs=(
            VPInteractionDef(
                type_keys=("VP", "HG"),
                pot_style="lj/cut",
                pot_kwargs={"epsilon": 0.5, "sigma": 4.0},
            ),
        ),
        selection="resname DOPC",
        atomtype_order="back",
    )
    with pytest.raises(ValueError, match="requires an explicit cutoff"):
        render_vp_latent_template(cfg, template, default_cutoff=25.0)


def test_default_table_pair_uses_per_pair_generated_filenames(tmp_path):
    template = _make_template()
    cfg = VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=_make_vp_config().bonds,
        angles=_make_vp_config().angles,
        pairs=(),
        default_pair=VPInteractionDef(
            type_keys=("*", "*"),
            pot_style="table",
            pot_kwargs={"file": "shared.table", "cutoff": 12.0},
        ),
        selection="resname DOPC",
        atomtype_order="back",
    )
    text = render_vp_latent_template(cfg, template, default_cutoff=25.0)
    assert "Pair_VP-HG.table" in text
    assert "Pair_VP-MG.table" in text
    assert "shared.table" not in text

    out_path = write_latent_settings(
        template=template,
        vp_config=cfg,
        output_dir=tmp_path,
        table_points=101,
        table_rmin=0.0,
        table_rmax=25.0,
    )
    assert out_path.exists()
    assert (tmp_path / "Pair_VP-HG.table").exists()
    assert (tmp_path / "Pair_VP-MG.table").exists()
    assert not (tmp_path / "shared.table").exists()


def test_write_latent_settings_soft_default_emits_tables(tmp_path):
    template = _make_template()
    cfg = VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=_make_vp_config().bonds,
        angles=_make_vp_config().angles,
        pairs=(),
        default_pair=VPInteractionDef(
            type_keys=("*", "*"),
            pot_style="soft",
            pot_kwargs={"A": 25.0, "r_c": 5.0, "astable": True},
        ),
        selection="resname DOPC",
        atomtype_order="back",
    )
    out_path = write_latent_settings(
        template=template, vp_config=cfg, output_dir=tmp_path,
        table_points=101, table_rmin=0.0, table_rmax=25.0,
    )
    content = out_path.read_text()
    assert content.count("pair_coeff") == 5
    assert "Pair_VP-HG.table" in content


# ─── Dihedral support (forward-compatible) ──────────────────────────


def _make_template_with_dihedral() -> VPTopologyTemplate:
    """Like :func:`_make_template` but adds a single VP dihedral type."""
    from AceCG.topology.vpgrower import VPDihedralSpec

    base = _make_template()
    # One synthetic VP dihedral (HG-MG-T1-VP); tid=1 in the empty slot.
    dihedrals = np.array([[0, 1, 2, 4]], dtype=np.int64)
    dihedral_type_ids = np.array([1], dtype=np.int64)
    dihedral_type_key_by_id = {1: InteractionKey.dihedral("HG", "MG", "T1", "VP")}
    vp_dihedral_specs = (
        VPDihedralSpec(
            vp_name="VP",
            labels=("HG", "MG", "T1", "VP"),
            pot_style="harmonic",
            pot_kwargs={"K": 3.5, "d": 1, "n": 2},
        ),
    )
    return VPTopologyTemplate(
        atom_names=base.atom_names,
        atom_masses=base.atom_masses,
        resids=base.resids,
        resnames=base.resnames,
        real_indices=base.real_indices,
        vp_indices_by_name=base.vp_indices_by_name,
        carrier_resids=base.carrier_resids,
        type2id=base.type2id,
        type_masses=base.type_masses,
        bonds=base.bonds,
        angles=base.angles,
        dihedrals=dihedrals,
        bond_type_ids=base.bond_type_ids,
        angle_type_ids=base.angle_type_ids,
        dihedral_type_ids=dihedral_type_ids,
        bond_type_key_by_id=base.bond_type_key_by_id,
        angle_type_key_by_id=base.angle_type_key_by_id,
        dihedral_type_key_by_id=dihedral_type_key_by_id,
        vp_bond_specs=base.vp_bond_specs,
        vp_angle_specs=base.vp_angle_specs,
        vp_dihedral_specs=vp_dihedral_specs,
        clash_max_passes=base.clash_max_passes,
        clash_min_distance=base.clash_min_distance,
    )


def test_render_vp_latent_template_emits_dihedral_line():
    """Dihedral specs flow straight into the template with final numerics."""
    template = _make_template_with_dihedral()
    cfg = _make_vp_config()
    text = render_vp_latent_template(cfg, template, default_cutoff=25.0)
    assert "dihedral_coeff" in text
    # Harmonic dihedral line: "dihedral_coeff 1 harmonic 3.5 1 2"
    lines = [ln for ln in text.splitlines() if "dihedral_coeff" in ln]
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "dihedral_coeff"
    assert parts[1] == "1"
    assert parts[2] == "harmonic"
    assert float(parts[3]) == pytest.approx(3.5)
    assert parts[4] == "1"
    assert parts[5] == "2"


def test_parse_vp_dihedrals_positional_coeffs():
    """``vp_dihedrals`` parses positional ``K d n`` tokens into ``coeffs``."""
    from AceCG.configs.vp_config import parse_vp_config

    sections = {
        "vp_atoms": {"VP": "72.0"},
        "vp_dihedrals": {"HG-MG-T1-VP": "harmonic 3.5 1 2"},
    }
    cfg = parse_vp_config(sections)
    assert len(cfg.dihedrals) == 1
    d = cfg.dihedrals[0]
    assert d.type_keys == ("HG", "MG", "T1", "VP")
    assert d.pot_style == "harmonic"
    assert d.pot_kwargs["coeffs"] == (3.5, 1, 2)
