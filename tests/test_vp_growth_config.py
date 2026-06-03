"""Unit tests for the standalone VP Growth config parser."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from AceCG.configs import (
    ACGConfigError,
    VPGrowthConfig,
    parse_vp_growth_file,
    parse_vp_growth_text,
)


# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def alias_file(tmp_path: Path) -> Path:
    p = tmp_path / "cg6_aliases.json"
    p.write_text(json.dumps({"1": "HG", "2": "MG", "3": "T1", "4": "T2"}))
    return p


@pytest.fixture()
def full_config_text(alias_file: Path) -> str:
    """A realistic VP Growth config using alias names throughout."""
    return textwrap.dedent(f"""\
        [aa_ref]
        trajectory_files    = ["cg6_1.lammpstrj", "cg6_2.lammpstrj"]
        trajectory_format   = LAMMPSDUMP
        ref_topo            = cg6.data
        ref_topo_type_names = {alias_file.name}
        skip_frames         = 0
        every               = 1
        n_frames            = 500
        include_forces      = true

        [vp]
        output_dir            = pool
        frame_ids             = all
        orientation_seed_base = 20260420
        selection             = resname DOPC
        atomtype_order        = front
        clash_max_passes      = 12
        clash_min_distance    = 2.0
        table_points          = 2500
        table_rmax            = 25.0
        overwrite             = false

        [vp_atoms]
        VP = 72.0

        [vp_bonds]
        VP-MG = harmonic k=2.50 r0=1.50

        [vp_angles]
        VP-MG-HG = harmonic k=2.45 theta0=135.0

        [vp_pairs]
        VP-HG   = table Pair_VP-HG.table astable=yes
        VP-MG   = lj/cut eps=0.5 sigma=4.0 rcut=12.0
        VP-T1   = lj/cut eps=0.2 sigma=4.5 rcut=12.0 astable=yes
        VP-T2   = lj/cut eps=0.2 sigma=4.5 rcut=12.0
        VP-VP   = lj/cut/soft eps=0.1 sigma=5.0 lambda_=1.0 rcut=10.0
        default = gauss/cut H=0.0 sigma=3.0 rmh=0.0 rcut=12.0
    """)


# ─── Happy path ───────────────────────────────────────────────────────


def test_parse_vp_growth_file_full(
    tmp_path: Path, alias_file: Path, full_config_text: str
) -> None:
    cfg_path = tmp_path / "vp_growth.acg"
    cfg_path.write_text(full_config_text)

    cfg = parse_vp_growth_file(cfg_path)

    assert isinstance(cfg, VPGrowthConfig)
    assert cfg.path == cfg_path.resolve()

    aa = cfg.aa_ref
    assert aa.trajectory_files == ("cg6_1.lammpstrj", "cg6_2.lammpstrj")
    assert aa.trajectory_format == "LAMMPSDUMP"
    assert aa.ref_topo == "cg6.data"
    assert aa.ref_topo_type_names == {1: "HG", 2: "MG", 3: "T1", 4: "T2"}
    assert aa.skip_frames == 0
    assert aa.every == 1
    assert aa.n_frames == 500
    assert aa.include_forces is True

    run = cfg.run
    assert run.output_dir == "pool"
    assert run.frame_ids is None
    assert run.orientation_seed_base == 20260420
    assert run.latent_settings_name == "latent.settings"
    assert run.table_points == 2500
    assert run.table_rmax == 25.0
    assert run.overwrite is False

    vp = cfg.vp
    assert vp.vp_names == ("VP",)
    assert vp.selection == "resname DOPC"
    assert vp.atomtype_order == "front"
    assert vp.clash_max_passes == 12
    assert vp.clash_min_distance == 2.0
    assert len(vp.bonds) == 1
    assert vp.bonds[0].type_keys == ("VP", "MG")
    assert vp.bonds[0].pot_style == "harmonic"
    assert vp.bonds[0].pot_kwargs == {"k": 2.50, "r0": 1.50}
    assert len(vp.angles) == 1
    assert vp.angles[0].type_keys == ("VP", "MG", "HG")
    assert vp.default_pair is not None
    assert vp.default_pair.pot_style == "gauss/cut"


def test_frame_ids_range_form(tmp_path: Path, alias_file: Path) -> None:
    text = textwrap.dedent(f"""\
        [aa_ref]
        ref_topo            = cg6.data
        ref_topo_type_names = {alias_file.name}

        [vp]
        output_dir = pool
        frame_ids  = 0-9
    """)
    (tmp_path / "c.acg").write_text(text)
    cfg = parse_vp_growth_file(tmp_path / "c.acg")
    assert cfg.run.frame_ids == tuple(range(0, 10))


def test_frame_ids_list_form(tmp_path: Path, alias_file: Path) -> None:
    text = textwrap.dedent(f"""\
        [aa_ref]
        ref_topo            = cg6.data
        ref_topo_type_names = {alias_file.name}

        [vp]
        output_dir = pool
        frame_ids  = [1, 3, 5]
    """)
    (tmp_path / "c.acg").write_text(text)
    cfg = parse_vp_growth_file(tmp_path / "c.acg")
    assert cfg.run.frame_ids == (1, 3, 5)


def test_ref_topo_type_names_csv_form(tmp_path: Path) -> None:
    text = textwrap.dedent("""\
        [aa_ref]
        ref_topo            = cg6.data
        ref_topo_type_names = HG, MG, T1, T2

        [vp]
        output_dir = pool

        [vp_atoms]
        VP = 72.0

        [vp_bonds]
        VP-MG = harmonic k=2.5 r0=1.5
    """)
    (tmp_path / "c.acg").write_text(text)
    cfg = parse_vp_growth_file(tmp_path / "c.acg")
    assert cfg.aa_ref.ref_topo_type_names == {1: "HG", 2: "MG", 3: "T1", 4: "T2"}


def test_ref_topo_type_names_inline_dict(tmp_path: Path) -> None:
    text = textwrap.dedent("""\
        [aa_ref]
        ref_topo            = cg6.data
        ref_topo_type_names = {"1": "HG", "2": "MG"}

        [vp]
        output_dir = pool

        [vp_atoms]
        VP = 72.0

        [vp_bonds]
        VP-MG = harmonic k=2.5 r0=1.5
    """)
    (tmp_path / "c.acg").write_text(text)
    cfg = parse_vp_growth_file(tmp_path / "c.acg")
    assert cfg.aa_ref.ref_topo_type_names == {1: "HG", 2: "MG"}


def test_bare_integer_labels_without_aliases(tmp_path: Path) -> None:
    text = textwrap.dedent("""\
        [aa_ref]
        ref_topo = cg6.data

        [vp]
        output_dir = pool

        [vp_atoms]
        VP = 72.0

        [vp_bonds]
        VP-2 = harmonic k=2.5 r0=1.5
    """)
    (tmp_path / "c.acg").write_text(text)
    cfg = parse_vp_growth_file(tmp_path / "c.acg")
    assert cfg.aa_ref.ref_topo_type_names is None
    assert cfg.vp.bonds[0].type_keys == ("VP", "2")


# ─── Error paths ──────────────────────────────────────────────────────


def test_unknown_section_rejected(tmp_path: Path) -> None:
    text = textwrap.dedent("""\
        [aa_ref]
        ref_topo = cg6.data

        [training]
        method = fm

        [vp]
        output_dir = pool
    """)
    (tmp_path / "c.acg").write_text(text)
    with pytest.raises(ACGConfigError, match="unsupported sections"):
        parse_vp_growth_file(tmp_path / "c.acg")


def test_missing_output_dir_rejected(tmp_path: Path) -> None:
    text = textwrap.dedent("""\
        [aa_ref]
        ref_topo = cg6.data

        [vp]
        overwrite = true
    """)
    (tmp_path / "c.acg").write_text(text)
    with pytest.raises(ACGConfigError, match="output_dir is required"):
        parse_vp_growth_file(tmp_path / "c.acg")


def test_missing_ref_topo_rejected(tmp_path: Path) -> None:
    text = textwrap.dedent("""\
        [aa_ref]
        trajectory_files = ["a.lammpstrj"]

        [vp]
        output_dir = pool
    """)
    (tmp_path / "c.acg").write_text(text)
    with pytest.raises(ACGConfigError, match="ref_topo is required"):
        parse_vp_growth_file(tmp_path / "c.acg")


def test_mixed_alias_and_bare_int_rejected(
    tmp_path: Path, alias_file: Path
) -> None:
    text = textwrap.dedent(f"""\
        [aa_ref]
        ref_topo            = cg6.data
        ref_topo_type_names = {alias_file.name}

        [vp]
        output_dir = pool

        [vp_atoms]
        VP = 72.0

        [vp_bonds]
        VP-MG = harmonic k=2.5 r0=1.5
        VP-2  = harmonic k=2.5 r0=1.5
    """)
    (tmp_path / "c.acg").write_text(text)
    with pytest.raises(ACGConfigError, match="mixes alias names with bare integer"):
        parse_vp_growth_file(tmp_path / "c.acg")


def test_unknown_label_with_aliases_rejected(
    tmp_path: Path, alias_file: Path
) -> None:
    text = textwrap.dedent(f"""\
        [aa_ref]
        ref_topo            = cg6.data
        ref_topo_type_names = {alias_file.name}

        [vp]
        output_dir = pool

        [vp_atoms]
        VP = 72.0

        [vp_bonds]
        VP-XYZ = harmonic k=2.5 r0=1.5
    """)
    (tmp_path / "c.acg").write_text(text)
    with pytest.raises(ACGConfigError, match="unknown type labels"):
        parse_vp_growth_file(tmp_path / "c.acg")


def test_bad_atomtype_order_rejected(tmp_path: Path) -> None:
    text = textwrap.dedent("""\
        [aa_ref]
        ref_topo = cg6.data

        [vp]
        output_dir     = pool
        atomtype_order = sideways
    """)
    (tmp_path / "c.acg").write_text(text)
    with pytest.raises(ACGConfigError, match="atomtype_order must be"):
        parse_vp_growth_file(tmp_path / "c.acg")


def test_unknown_vp_key_rejected(tmp_path: Path) -> None:
    text = textwrap.dedent("""\
        [aa_ref]
        ref_topo = cg6.data

        [vp]
        output_dir = pool
        wiggle     = 3
    """)
    (tmp_path / "c.acg").write_text(text)
    with pytest.raises(ACGConfigError, match="Unknown \\[vp\\] keys"):
        parse_vp_growth_file(tmp_path / "c.acg")


def test_parse_vp_growth_text_roundtrip() -> None:
    text = textwrap.dedent("""\
        [aa_ref]
        ref_topo            = cg6.data
        ref_topo_type_names = {"1": "HG", "2": "MG", "3": "T1"}

        [vp]
        output_dir = pool

        [vp_atoms]
        VP = 72.0

        [vp_bonds]
        VP-MG = harmonic k=2.5 r0=1.5
    """)
    cfg = parse_vp_growth_text(text)
    assert cfg.path is None
    assert cfg.run.output_dir == "pool"
    assert cfg.vp.vp_names == ("VP",)
    assert cfg.aa_ref.ref_topo_type_names == {1: "HG", 2: "MG", 3: "T1"}
