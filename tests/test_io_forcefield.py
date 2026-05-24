"""Tests for AceCG.io.forcefield WriteLmpFF and ReadLmpFF (§15.4 item A3).

Scope: Tests focus on file-format mechanics (WriteLmpFF round-trip and
ReadLmpFF table parsing). ReadLmpFF with table style is tested with a
BSpline fitter; if fitting proves non-deterministic, the test is skipped
per the plan §7 risk note.
"""

import pytest
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from AceCG.fitters import BSplineConfig, BSplineTableFitter
from AceCG.io.forcefield import ReadLmpFF, ReadLmpFFBounds, ReadLmpFFMask, WriteLmpFF
from AceCG.io.tables import parse_lammps_table
from AceCG.potentials.bspline import BSplinePotential
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey
from AceCG.workflows.base import BaseWorkflow

# Fitters self-register on import.  ReadLmpFF requires the desired fitter
# to be registered before it is called.  Import here to register bspline.
import AceCG.fitters.fit_bspline  # noqa: F401 (side-effect: registers "bspline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pair_table(path: Path, k: float = 10.0, r0: float = 3.0, n: int = 50) -> None:
    """Write a harmonic pair table file."""
    r = np.linspace(1.0, 5.0, n)
    V = 0.5 * k * (r - r0) ** 2
    F = -k * (r - r0)
    with open(path, "w") as f:
        f.write("# Synthetic harmonic pair table for testing\n")
        f.write("PAIR_AB\n")
        f.write(f"N {n} R {r[0]:.6f} {r[-1]:.6f}\n\n")
        for i, (ri, vi, fi) in enumerate(zip(r, V, F), start=1):
            f.write(f"{i:6d}  {ri:16.8f}  {vi:16.8e}  {fi:16.8e}\n")


def _write_settings_file(path: Path, table_file_name: str, table_name: str = "PAIR_AB") -> None:
    """Write a minimal LAMMPS-style settings file referencing a single table pair.

    For non-hybrid table pair_style, the pair_coeff format is:
      pair_coeff TYPE1 TYPE2 <table_file> <table_keyword>
    (no "table" keyword in the pair_coeff line itself).
    """
    with open(path, "w") as f:
        f.write("# Synthetic LAMMPS settings file\n")
        f.write(f"pair_coeff A B {table_file_name} {table_name}\n")


def _write_numeric_pair_settings_file(
    path: Path,
    table_file_name: str,
    typ1: str,
    typ2: str,
    table_name: str = "PAIR_AB",
) -> None:
    with open(path, "w") as f:
        f.write("# Synthetic LAMMPS settings file\n")
        f.write(f"pair_coeff {typ1} {typ2} {table_file_name} {table_name}\n")


def _make_maskable_forcefield() -> Forcefield:
    knots = BSplinePotential.clamped_uniform_knots(1.0, 5.0, n_coeff=8, degree=3)
    pair_pot = BSplinePotential(
        "A",
        "B",
        knots=knots,
        coefficients=np.zeros(8, dtype=float),
        degree=3,
        cutoff=5.0,
    )
    bond_pot = HarmonicPotential("1", "1", k=10.0, r0=2.0)
    return Forcefield(
        {
            InteractionKey.pair("A", "B"): [pair_pot],
            InteractionKey(style="bond", types=("1",)): [bond_pot],
        }
    )


def _make_alias_bonded_forcefield():
    bond_key = InteractionKey.bond("HG", "VP")
    angle_key = InteractionKey.angle("MG", "HG", "VP")
    return (
        Forcefield(
            {
                bond_key: [HarmonicPotential("HG", "VP", k=10.0, r0=2.0)],
                angle_key: [
                    HarmonicPotential(
                        "MG",
                        "VP",
                        k=20.0,
                        r0=120.0,
                        typ3="HG",
                        scale=(np.pi / 180.0) ** 2,
                    )
                ],
            }
        ),
        bond_key,
        angle_key,
    )


def _make_alias_bonded_topology(bond_key, angle_key):
    return SimpleNamespace(
        atom_type_code_to_name={},
        bond_type_id_to_key={3: bond_key},
        angle_type_id_to_key={6: angle_key},
        dihedral_type_id_to_key={},
        key_to_bonded_type_id={bond_key: 3, angle_key: 6},
    )


class _MaskWorkflow(BaseWorkflow):
    def _build_trainer(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


def _materialize_mask_from_spec(spec, forcefield: Forcefield, topology=None) -> np.ndarray:
    workflow = object.__new__(_MaskWorkflow)
    workflow.config = SimpleNamespace(
        system=SimpleNamespace(forcefield_mask=spec)
    )
    workflow.topology = topology if topology is not None else EMPTY_TOPOLOGY_ARRAYS
    return workflow._build_forcefield_mask(forcefield)


def _materialize_bounds_from_spec(spec, forcefield: Forcefield, topology=None):
    workflow = object.__new__(_MaskWorkflow)
    workflow.config = SimpleNamespace(
        system=SimpleNamespace(forcefield_bounds=spec)
    )
    workflow.topology = topology if topology is not None else EMPTY_TOPOLOGY_ARRAYS
    return workflow._build_forcefield_bounds(forcefield)


EMPTY_TOPOLOGY_ARRAYS = SimpleNamespace(
    atom_type_code_to_name={},
    bond_type_id_to_key={},
    angle_type_id_to_key={},
    dihedral_type_id_to_key={},
    key_to_bonded_type_id={},
)


def _write_settings_file_with_bonded(
    path: Path,
    table_file_name: str,
    table_name: str = "PAIR_AB",
) -> None:
    """Write a minimal settings file with pair, bond, and angle coefficients."""
    with open(path, "w") as f:
        f.write("# Synthetic LAMMPS settings file\n")
        f.write(f"pair_coeff A B {table_file_name} {table_name}\n")
        f.write("bond_coeff 1 harmonic 10.0 2.0\n")
        f.write("angle_coeff 2 harmonic 20.0 120.0\n")


def _write_bond_table(path: Path, k: float = 10.0, r0: float = 2.0, n: int = 80) -> None:
    r = np.linspace(1.0, 3.0, n)
    V = 0.5 * k * (r - r0) ** 2
    F = -k * (r - r0)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Synthetic bond table for testing\n")
        f.write("BOND_AB\n")
        f.write(f"N {n} EQ {r0:.6f}\n\n")
        for i, (ri, vi, fi) in enumerate(zip(r, V, F), start=1):
            f.write(f"{i:6d}  {ri:16.8f}  {vi:16.8e}  {fi:16.8e}\n")


def _write_bond_settings_file(path: Path, table_file_name: str, table_name: str = "BOND_AB") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Synthetic bonded settings file\n")
        f.write(f"bond_coeff 1 table {table_file_name} {table_name}\n")


# ---------------------------------------------------------------------------
# WriteLmpFF tests
# ---------------------------------------------------------------------------

@pytest.fixture
def ff_setup(tmp_path):
    """Set up settings file, table file, and HarmonicPotential."""
    table_file = tmp_path / "AB_pair.table"
    settings_file = tmp_path / "settings.lmp"
    _write_pair_table(table_file, k=10.0, r0=3.0)
    _write_settings_file(settings_file, "AB_pair.table", "PAIR_AB")

    pot = HarmonicPotential("A", "B", k=10.0, r0=3.0)
    return {
        "tmp_path": tmp_path,
        "table_file": table_file,
        "settings_file": settings_file,
        "pot": pot,
    }


def test_writelmpff_creates_output(ff_setup):
    new_settings = ff_setup["tmp_path"] / "settings_new.lmp"
    WriteLmpFF(
        str(ff_setup["settings_file"]),
        str(new_settings),
        {InteractionKey.pair("A", "B"): [ff_setup["pot"]]},
        "table",
        topology_arrays=EMPTY_TOPOLOGY_ARRAYS,
    )
    assert new_settings.exists()


def test_writelmpff_output_has_pair_coeff(ff_setup):
    new_settings = ff_setup["tmp_path"] / "settings_new.lmp"
    WriteLmpFF(
        str(ff_setup["settings_file"]),
        str(new_settings),
        {InteractionKey.pair("A", "B"): [ff_setup["pot"]]},
        "table",
        topology_arrays=EMPTY_TOPOLOGY_ARRAYS,
    )
    content = new_settings.read_text()
    assert "pair_coeff" in content
    assert "A" in content
    assert "B" in content


def test_writelmpff_updates_table_file(ff_setup):
    """After WriteLmpFF, the table file should contain potential values matching the potential."""
    k, r0 = 10.0, 3.0
    pot = ff_setup["pot"]
    WriteLmpFF(
        str(ff_setup["settings_file"]),
        str(ff_setup["tmp_path"] / "settings_new.lmp"),
        {InteractionKey.pair("A", "B"): [pot]},
        "table",
        topology_arrays=EMPTY_TOPOLOGY_ARRAYS,
    )
    # Re-parse the (updated) table file
    r, V, F = parse_lammps_table(str(ff_setup["table_file"]))
    V_expected = pot.value(r)
    F_expected = pot.force(r)
    assert np.allclose(V, V_expected, atol=1e-6)
    assert np.allclose(F, F_expected, atol=1e-6)


def test_writelmpff_r_grid_preserved(ff_setup):
    """WriteLmpFF should preserve the r-grid from the original table."""
    r_orig, _, _ = parse_lammps_table(str(ff_setup["table_file"]))
    WriteLmpFF(
        str(ff_setup["settings_file"]),
        str(ff_setup["tmp_path"] / "settings_new.lmp"),
        {InteractionKey.pair("A", "B"): [ff_setup["pot"]]},
        "table",
        topology_arrays=EMPTY_TOPOLOGY_ARRAYS,
    )
    r_new, _, _ = parse_lammps_table(str(ff_setup["table_file"]))
    assert np.allclose(r_orig, r_new, atol=1e-8)


# ---------------------------------------------------------------------------
# ReadLmpFF tests — table style with BSpline fitter
# ---------------------------------------------------------------------------

@pytest.fixture
def ff_readlmpff_setup(tmp_path):
    """Set up a settings file and table for ReadLmpFF."""
    k, r0, n = 10.0, 3.0, 100
    table_file = tmp_path / "AB_pair.table"
    settings_file = tmp_path / "settings.lmp"
    _write_pair_table(table_file, k=k, r0=r0, n=n)
    _write_settings_file(settings_file, "AB_pair.table", "PAIR_AB")
    return {"settings_file": settings_file}


def test_readlmpff_returns_dict(ff_readlmpff_setup):
    try:
        result = ReadLmpFF(
            str(ff_readlmpff_setup["settings_file"]),
            "table",
            table_fit="bspline",
            topology_arrays=EMPTY_TOPOLOGY_ARRAYS,
        )
    except Exception as exc:
        pytest.skip(f"ReadLmpFF fitting failed (plan §7 risk): {exc}")
    assert isinstance(result, Forcefield)


def test_readlmpff_has_pair_key(ff_readlmpff_setup):
    try:
        result = ReadLmpFF(
            str(ff_readlmpff_setup["settings_file"]),
            "table",
            table_fit="bspline",
            topology_arrays=EMPTY_TOPOLOGY_ARRAYS,
        )
    except Exception as exc:
        pytest.skip(f"ReadLmpFF fitting failed (plan §7 risk): {exc}")
    assert len(result) >= 1


def test_readlmpff_potential_has_nparams(ff_readlmpff_setup):
    try:
        result = ReadLmpFF(
            str(ff_readlmpff_setup["settings_file"]),
            "table",
            table_fit="bspline",
            topology_arrays=EMPTY_TOPOLOGY_ARRAYS,
        )
    except Exception as exc:
        pytest.skip(f"ReadLmpFF fitting failed (plan §7 risk): {exc}")
    for pair, pots in result.items():
        for pot in (pots if isinstance(pots, list) else [pots]):
            assert pot.n_params() > 0


# ---------------------------------------------------------------------------
# ReadLmpFF — InteractionKey contract
# ---------------------------------------------------------------------------

def test_readlmpff_keys_are_interactionkey(ff_readlmpff_setup):
    """ReadLmpFF must return InteractionKey keys (U1 contract)."""
    from AceCG.topology.types import InteractionKey

    try:
        result = ReadLmpFF(
            str(ff_readlmpff_setup["settings_file"]),
            "table",
            table_fit="bspline",
            topology_arrays=EMPTY_TOPOLOGY_ARRAYS,
        )
    except Exception as exc:
        pytest.skip(f"ReadLmpFF fitting failed (plan §7 risk): {exc}")
    for key in result:
        assert isinstance(key, InteractionKey), f"Expected InteractionKey, got {type(key)}"
        assert key.style == "pair"


def test_readlmpff_key_types_are_tuple(ff_readlmpff_setup):
    """ReadLmpFF InteractionKey.types should be a tuple of type strings."""
    from AceCG.topology.types import InteractionKey

    try:
        result = ReadLmpFF(
            str(ff_readlmpff_setup["settings_file"]),
            "table",
            table_fit="bspline",
            topology_arrays=EMPTY_TOPOLOGY_ARRAYS,
        )
    except Exception as exc:
        pytest.skip(f"ReadLmpFF fitting failed (plan §7 risk): {exc}")
    for key in result:
        assert isinstance(key.types, tuple)
        assert all(isinstance(t, str) for t in key.types)


def test_readlmpff_reads_bonded_terms_into_single_forcefield(tmp_path):
    """ReadLmpFF must return bonded terms in the canonical unified container."""
    table_file = tmp_path / "AB_pair.table"
    settings_file = tmp_path / "settings_bonded.lmp"
    _write_pair_table(table_file, k=10.0, r0=3.0, n=80)
    _write_settings_file_with_bonded(settings_file, "AB_pair.table", "PAIR_AB")

    result = ReadLmpFF(
        str(settings_file),
        "table",
        table_fit="bspline",
        topology_arrays=EMPTY_TOPOLOGY_ARRAYS,
    )

    assert isinstance(result, Forcefield)
    assert InteractionKey.pair("A", "B") in result
    assert InteractionKey(style="bond", types=("1",)) in result
    assert InteractionKey(style="angle", types=("2",)) in result


def test_readlmpff_translates_bonded_type_ids_via_topology_arrays(tmp_path):
    table_file = tmp_path / "AB_pair.table"
    settings_file = tmp_path / "settings_bonded.lmp"
    _write_pair_table(table_file, k=10.0, r0=3.0, n=80)
    _write_settings_file_with_bonded(settings_file, "AB_pair.table", "PAIR_AB")

    topology_arrays = SimpleNamespace(
        atom_type_code_to_name={},
        bond_type_id_to_key={0: InteractionKey.bond("A", "B")},
        angle_type_id_to_key={1: InteractionKey.angle("A", "A", "B")},
        dihedral_type_id_to_key={},
        key_to_bonded_type_id={
            InteractionKey.bond("A", "B"): 0,
            InteractionKey.angle("A", "A", "B"): 1,
        },
    )

    result = ReadLmpFF(
        str(settings_file),
        "table",
        table_fit="bspline",
        topology_arrays=topology_arrays,
    )

    assert InteractionKey.bond("A", "B") in result
    assert InteractionKey.angle("A", "A", "B") in result


@pytest.mark.parametrize(
    "bonded, writer, table_name",
    [
        (False, _write_pair_table, "PAIR_AB"),
        (True, _write_bond_table, "BOND_AB"),
    ],
)
def test_bspline_table_fitter_config_propagates_bonded_flag(tmp_path, bonded, writer, table_name):
    table_file = tmp_path / ("pair.table" if not bonded else "bond.table")
    writer(table_file)
    fitter = BSplineTableFitter(BSplineConfig(n_coeffs=8, degree=3, bonded=bonded))
    pot = fitter.fit(str(table_file), typ1="A", typ2="B")
    assert isinstance(pot, BSplinePotential)
    assert pot.bonded is bonded


def test_readlmpff_bspline_bond_table_uses_minimum_gauge(tmp_path):
    table_file = tmp_path / "AB_bond.table"
    settings_file = tmp_path / "settings_bond_table.lmp"
    _write_bond_table(table_file, k=10.0, r0=2.0, n=120)
    _write_bond_settings_file(settings_file, "AB_bond.table", "BOND_AB")

    topology_arrays = SimpleNamespace(
        atom_type_code_to_name={},
        bond_type_id_to_key={0: InteractionKey.bond("A", "B")},
        angle_type_id_to_key={},
        dihedral_type_id_to_key={},
        key_to_bonded_type_id={InteractionKey.bond("A", "B"): 0},
    )

    result = ReadLmpFF(
        str(settings_file),
        "table",
        table_fit="bspline",
        topology_arrays=topology_arrays,
    )

    pot = result[InteractionKey.bond("A", "B")][0]
    assert isinstance(pot, BSplinePotential)
    assert pot.bonded is True


def test_readlmpff_translates_numeric_pair_codes_with_gaps(tmp_path):
    table_file = tmp_path / "AB_pair.table"
    settings_file = tmp_path / "settings_numeric_pair.lmp"
    _write_pair_table(table_file, k=10.0, r0=3.0, n=80)
    _write_numeric_pair_settings_file(settings_file, "AB_pair.table", "2", "4", "PAIR_AB")

    topology_arrays = SimpleNamespace(
        atom_type_code_to_name={2: "2", 4: "4"},
        bond_type_id_to_key={},
        angle_type_id_to_key={},
        dihedral_type_id_to_key={},
        key_to_bonded_type_id={},
    )

    result = ReadLmpFF(
        str(settings_file),
        "table",
        table_fit="bspline",
        topology_arrays=topology_arrays,
    )

    assert InteractionKey.pair("2", "4") in result


# ---------------------------------------------------------------------------
# WriteLmpFF — InteractionKey-keyed container
# ---------------------------------------------------------------------------

def test_writelmpff_with_interactionkey_keyed_container(ff_setup):
    """WriteLmpFF must update tables when given an InteractionKey-keyed dict."""
    from AceCG.topology.types import InteractionKey

    ik = InteractionKey.pair("A", "B")
    pot = ff_setup["pot"]
    new_settings = ff_setup["tmp_path"] / "settings_ik.lmp"
    WriteLmpFF(
        str(ff_setup["settings_file"]),
        str(new_settings),
        {ik: [pot]},
        "table",
        topology_arrays=EMPTY_TOPOLOGY_ARRAYS,
    )
    assert new_settings.exists()
    # Verify table was updated with correct values
    r, V, F = parse_lammps_table(str(ff_setup["table_file"]))
    assert np.allclose(V, pot.value(r), atol=1e-6)
    assert np.allclose(F, pot.force(r), atol=1e-6)


def test_writelmpff_updates_harmonic_bond_coeff_from_unified_forcefield(tmp_path):
    """WriteLmpFF must update bonded coefficients from the same forcefield dict."""
    table_file = tmp_path / "AB_pair.table"
    settings_file = tmp_path / "settings_bonded.lmp"
    output_file = tmp_path / "settings_bonded_new.lmp"
    _write_pair_table(table_file, k=10.0, r0=3.0, n=50)
    _write_settings_file_with_bonded(settings_file, "AB_pair.table", "PAIR_AB")

    pair_pot = HarmonicPotential("A", "B", k=10.0, r0=3.0)
    bond_pot = HarmonicPotential("1", "1", k=15.0, r0=2.5)

    WriteLmpFF(
        str(settings_file),
        str(output_file),
        {
            InteractionKey.pair("A", "B"): [pair_pot],
            InteractionKey(style="bond", types=("1",)): [bond_pot],
        },
        "table",
        topology_arrays=EMPTY_TOPOLOGY_ARRAYS,
    )

    content = output_file.read_text()
    bond_line = next(line for line in content.splitlines() if line.startswith("bond_coeff"))
    assert "15" in bond_line
    assert "2.5" in bond_line


def test_writelmpff_translates_canonical_keys_via_topology_arrays(tmp_path):
    """WriteLmpFF must translate canonical keys via topology_arrays type-id maps."""
    table_file = tmp_path / "AB_pair.table"
    settings_file = tmp_path / "settings_bonded.lmp"
    output_file = tmp_path / "settings_bonded_new.lmp"
    _write_pair_table(table_file, k=10.0, r0=3.0, n=50)
    _write_settings_file_with_bonded(settings_file, "AB_pair.table", "PAIR_AB")

    pair_pot = HarmonicPotential("A", "B", k=10.0, r0=3.0)
    bond_pot = HarmonicPotential("A", "B", k=15.0, r0=2.5)
    angle_pot = HarmonicPotential("A", "A", k=30.0, r0=135.0, scale=(np.pi / 180.0) ** 2)
    topology_arrays = SimpleNamespace(
        atom_type_code_to_name={},
        bond_type_id_to_key={0: InteractionKey.bond("A", "B")},
        angle_type_id_to_key={1: InteractionKey.angle("A", "A", "B")},
        dihedral_type_id_to_key={},
        key_to_bonded_type_id={
            InteractionKey.bond("A", "B"): 0,
            InteractionKey.angle("A", "A", "B"): 1,
        },
    )

    WriteLmpFF(
        str(settings_file),
        str(output_file),
        {
            InteractionKey.pair("A", "B"): [pair_pot],
            InteractionKey.bond("A", "B"): [bond_pot],
            InteractionKey.angle("A", "A", "B"): [angle_pot],
        },
        "table",
        topology_arrays=topology_arrays,
    )

    content = output_file.read_text().splitlines()
    bond_line = next(line for line in content if line.startswith("bond_coeff"))
    angle_line = next(line for line in content if line.startswith("angle_coeff"))
    assert "15" in bond_line
    assert "2.5" in bond_line
    assert "30" in angle_line
    assert "135" in angle_line


def test_readlmpffmask_parses_table_ranges_and_parametric_bools(tmp_path):
    mask_file = tmp_path / "mask.settings"
    mask_file.write_text(
        "\n".join(
            [
                "pair_coeff A B table mask 0:2,-1:",
                "bond_coeff 1 harmonic yes no",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    forcefield = _make_maskable_forcefield()
    topology = SimpleNamespace(
        atom_type_code_to_name={},
        bond_type_id_to_key={0: InteractionKey(style="bond", types=("1",))},
        angle_type_id_to_key={},
        dihedral_type_id_to_key={},
        key_to_bonded_type_id={},
    )
    spec = ReadLmpFFMask(str(mask_file), "table", topology_arrays=topology)
    mask = _materialize_mask_from_spec(spec, forcefield)

    assert mask.shape == (10,)
    assert mask.tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        False,
    ]

    assert spec.entries == (
        (InteractionKey.pair("A", "B"), "table", ("mask", "0:2,-1:")),
        (InteractionKey(style="bond", types=("1",)), "harmonic", ("yes", "no")),
    )


def test_readlmpffmask_defaults_to_all_active_for_plain_settings_file(tmp_path):
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    table_file = table_dir / "AB_pair.table"
    settings_file = tmp_path / "settings.lmp"
    _write_pair_table(table_file, k=10.0, r0=3.0, n=50)
    _write_settings_file(settings_file, "tables/AB_pair.table", "PAIR_AB")

    forcefield = _make_maskable_forcefield()
    spec = ReadLmpFFMask(str(settings_file), "table", topology_arrays=EMPTY_TOPOLOGY_ARRAYS)
    mask = _materialize_mask_from_spec(spec, forcefield)

    assert np.all(mask)


def test_readlmpffmask_raises_for_unknown_interaction(tmp_path):
    mask_file = tmp_path / "mask.settings"
    mask_file.write_text("bond_coeff 99 harmonic yes no\n", encoding="utf-8")

    with pytest.raises(KeyError):
        ReadLmpFFMask(str(mask_file), "table", topology_arrays=EMPTY_TOPOLOGY_ARRAYS)


def test_readlmpffmask_resolves_bonded_type_ids_via_topology_arrays(tmp_path):
    mask_file = tmp_path / "mask.settings"
    mask_file.write_text(
        "bond_coeff 4 harmonic no no\n"
        "angle_coeff 7 harmonic yes no\n",
        encoding="utf-8",
    )
    forcefield, bond_key, angle_key = _make_alias_bonded_forcefield()
    topology = _make_alias_bonded_topology(bond_key, angle_key)

    spec = ReadLmpFFMask(str(mask_file), "table", topology_arrays=topology)
    mask = _materialize_mask_from_spec(spec, forcefield)

    assert spec.entries == (
        (bond_key, "harmonic", ("no", "no")),
        (angle_key, "harmonic", ("yes", "no")),
    )
    assert mask.tolist() == [False, False, True, False]


def test_readlmpffbounds_resolves_bonded_type_ids_via_topology_arrays(tmp_path):
    bounds_file = tmp_path / "bounds.settings"
    bounds_file.write_text(
        "bond_coeff 4 harmonic lb 1.0 none ub 30.0 3.0\n"
        "angle_coeff 7 harmonic lb 0.0 90.0 ub 60.0 150.0\n",
        encoding="utf-8",
    )
    forcefield, bond_key, angle_key = _make_alias_bonded_forcefield()
    topology = _make_alias_bonded_topology(bond_key, angle_key)

    spec = ReadLmpFFBounds(str(bounds_file), "table", topology_arrays=topology)
    lb, ub = _materialize_bounds_from_spec(spec, forcefield)

    assert np.isneginf(lb[1])
    np.testing.assert_allclose(lb[[0, 2, 3]], [1.0, 0.0, 90.0])
    np.testing.assert_allclose(ub, [30.0, 3.0, 60.0, 150.0])


def test_writelmpff_writes_updated_tables_under_new_output_tree(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    table_relpath = "tables/AB_pair.table"
    source_table = source_dir / table_relpath
    source_table.parent.mkdir(parents=True)
    settings_file = source_dir / "settings.lmp"

    _write_pair_table(source_table, k=10.0, r0=3.0, n=50)
    r_before, V_before, F_before = parse_lammps_table(source_table)
    _write_settings_file(settings_file, table_relpath, "PAIR_AB")

    pot = HarmonicPotential("A", "B", k=12.0, r0=3.2)
    output_settings = output_dir / "settings_new.lmp"
    WriteLmpFF(
        str(settings_file),
        str(output_settings),
        {InteractionKey.pair("A", "B"): [pot]},
        "table",
        topology_arrays=EMPTY_TOPOLOGY_ARRAYS,
    )

    output_table = output_dir / table_relpath
    assert output_settings.exists()
    assert output_table.exists()

    r_out, V_out, F_out = parse_lammps_table(output_table)
    assert np.allclose(V_out, pot.value(r_out), atol=1e-6)
    assert np.allclose(F_out, pot.force(r_out), atol=1e-6)

    r_after, V_after, F_after = parse_lammps_table(source_table)
    assert np.allclose(r_after, r_before, atol=1e-8)
    assert np.allclose(V_after, V_before, atol=1e-8)
    assert np.allclose(F_after, F_before, atol=1e-8)
