"""Focused live-package tests for AceCG tabulated dihedral support."""

from __future__ import annotations

from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pytest

from AceCG.compute.energy import energy
from AceCG.compute.force import force
from AceCG.compute.frame_geometry import compute_frame_geometry
from AceCG.configs.parser import (
    ACGConfigError,
    _parse_named_fm_spec,
    validate_fm_spec_domain,
)
from AceCG.fitters.fit_bspline import DihedralBSplineTableFitter
from AceCG.io.forcefield import ReadLmpFF, WriteLmpFF
from AceCG.io.tables import (
    export_grid,
    parse_lammps_table,
    read_lammps_table_section,
    write_lammps_table,
)
from AceCG.potentials.dihedral_bspline import DihedralBSplinePotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.topology_array import collect_topology_arrays
from AceCG.topology.types import InteractionKey


TYPES = ("HG", "MG", "T1", "T2")


def _periodic_potential(n_coeffs: int = 8) -> DihedralBSplinePotential:
    coefficients = np.linspace(-0.08, 0.11, n_coeffs, dtype=float)
    return DihedralBSplinePotential(
        TYPES,
        minimum=-180.0,
        maximum=180.0,
        coefficients=coefficients,
        degree=3,
        boundary_mode="periodic",
    )


def _cutoff_potential(n_coeffs: int = 8) -> DihedralBSplinePotential:
    return DihedralBSplinePotential(
        TYPES,
        minimum=-100.0,
        maximum=125.0,
        coefficients=np.linspace(-0.05, 0.07, n_coeffs),
        degree=3,
        boundary_mode="cutoff",
    )


def test_periodic_force_and_energy_are_continuous_at_seam():
    potential = _periodic_potential()
    epsilon = 1.0e-6

    np.testing.assert_allclose(
        potential.force(np.array([-180.0 + epsilon])),
        potential.force(np.array([180.0 - epsilon])),
        rtol=0.0,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        potential.value(np.array([-180.0])),
        potential.value(np.array([180.0])),
        rtol=0.0,
        atol=1.0e-10,
    )


@pytest.mark.parametrize("boundary_mode", ["periodic", "cutoff"])
def test_dihedral_force_is_negative_degree_derivative(boundary_mode):
    if boundary_mode == "periodic":
        potential = _periodic_potential()
        points = np.array([-137.25, -12.5, 96.75])
    else:
        potential = DihedralBSplinePotential(
            TYPES,
            minimum=-120.0,
            maximum=135.0,
            coefficients=np.linspace(-0.05, 0.07, 8),
            degree=3,
            boundary_mode="cutoff",
        )
        points = np.array([-95.25, -12.5, 102.75])

    step = 1.0e-5
    numerical_force = -(
        potential.value(points + step) - potential.value(points - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        potential.force(points),
        numerical_force,
        rtol=2.0e-6,
        atol=2.0e-8,
    )


def test_cutoff_mode_is_zero_outside_model_window():
    potential = _cutoff_potential()
    outside = np.array([-179.0, -120.0, 150.0, 179.0])
    np.testing.assert_allclose(potential.force(outside), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(potential.value(outside), 0.0, atol=1.0e-12)


def test_cartesian_dihedral_force_matches_energy_finite_difference():
    key = InteractionKey.dihedral("A", "B", "C", "D")
    topology = SimpleNamespace(
        n_atoms=4,
        bonds=np.empty((0, 2), dtype=np.int64),
        bond_key_index=np.empty(0, dtype=np.int32),
        keys_bondtypes=[],
        angles=np.empty((0, 3), dtype=np.int64),
        angle_key_index=np.empty(0, dtype=np.int32),
        keys_angletypes=[],
        dihedrals=np.array([[0, 1, 2, 3]], dtype=np.int64),
        dihedral_key_index=np.array([0], dtype=np.int32),
        keys_dihedraltypes=[key],
        real_site_indices=None,
    )
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.25, 1.35, 0.75],
        ]
    )
    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
    potential = _periodic_potential()
    forcefield = Forcefield({key: [potential]})
    geometry = compute_frame_geometry(positions, box, topology)
    analytic = force(geometry, forcefield, return_value=True)["force"].reshape(4, 3)

    step = 2.0e-3
    numerical = np.zeros_like(positions)
    for atom_index in range(4):
        for axis in range(3):
            plus = positions.copy()
            minus = positions.copy()
            plus[atom_index, axis] += step
            minus[atom_index, axis] -= step
            energy_plus = energy(
                compute_frame_geometry(plus, box, topology),
                forcefield,
                return_value=True,
            )["energy"]
            energy_minus = energy(
                compute_frame_geometry(minus, box, topology),
                forcefield,
                return_value=True,
            )["energy"]
            numerical[atom_index, axis] = -(
                energy_plus - energy_minus
            ) / (2.0 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=2.0e-4, atol=4.0e-4)


def test_authored_grid_follows_requested_resolution_and_is_half_open():
    grid = export_grid(
        {
            "style": "dihedral",
            "min": -180.0,
            "max": 180.0,
            "resolution": 2.0,
        }
    )
    assert grid.size == 180
    assert grid[0] == pytest.approx(-180.0)
    assert grid[-1] == pytest.approx(178.0)


def test_source_grid_is_preserved_exactly():
    source_grid = np.array([-180.0, -121.5, -47.0, 3.0, 89.5, 179.0])
    grid = export_grid(
        {
            "style": "dihedral",
            "min": -180.0,
            "max": 180.0,
            "resolution": 0.5,
            "table_grid": source_grid,
        }
    )
    np.testing.assert_array_equal(grid, source_grid)


def test_shifted_periodic_source_grid_is_valid_and_preserved(tmp_path):
    source_grid = np.array([-168.0, -36.0, 84.0, 180.0])
    path = tmp_path / "shifted_periodic.table"
    write_lammps_table(
        path,
        source_grid,
        np.zeros(source_grid.size),
        np.zeros(source_grid.size),
        table_name="DIH",
        table_style="dihedral",
    )
    _, _, resolution = validate_fm_spec_domain(
        (-180.0, 180.0),
        str(path),
        table_name="DIH",
        style="dihedral",
        boundary_mode="periodic",
    )
    assert resolution == pytest.approx(120.0)
    np.testing.assert_array_equal(
        export_grid(
            {
                "style": "dihedral",
                "min": -180.0,
                "max": 180.0,
                "resolution": 0.5,
                "table_grid": source_grid,
            }
        ),
        source_grid,
    )


def test_writer_and_parser_preserve_degree_dihedral_table(tmp_path):
    potential = _periodic_potential()
    grid = np.arange(-180.0, 180.0, 2.0)
    path = tmp_path / "periodic.table"
    write_lammps_table(
        path,
        grid,
        potential.value(grid),
        potential.force(grid),
        table_name="DPPC_DIH",
        table_style="dihedral",
        metadata=potential.table_metadata(),
    )

    text = path.read_text(encoding="utf-8")
    assert "N 180 DEGREES" in text
    parsed_grid, parsed_value, parsed_force = parse_lammps_table(
        path,
        table_name="DPPC_DIH",
        table_style="dihedral",
    )
    np.testing.assert_allclose(parsed_grid, grid)
    np.testing.assert_allclose(parsed_value, potential.value(grid), atol=1.0e-8)
    np.testing.assert_allclose(parsed_force, potential.force(grid), atol=1.0e-8)


def test_radian_table_is_converted_to_degree_force_convention(tmp_path):
    path = tmp_path / "radian.table"
    path.write_text(
        "\n".join(
            [
                "DIH",
                "N 3 RADIANS",
                "",
                "1 -3.0 1.0 2.0",
                "2 0.0 0.0 3.0",
                "3 3.0 1.0 4.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    grid, _, force = parse_lammps_table(
        path,
        table_name="DIH",
        table_style="dihedral",
    )
    np.testing.assert_allclose(grid, np.degrees([-3.0, 0.0, 3.0]))
    np.testing.assert_allclose(force, np.array([2.0, 3.0, 4.0]) * np.pi / 180.0)


def test_dihedral_conversion_is_limited_to_the_declared_section(tmp_path):
    path = tmp_path / "mixed_styles.table"
    path.write_text(
        "DIH\nN 2 RADIANS\n1 -1.0 0.0 2.0\n2 1.0 0.0 4.0\n"
        "PAIR\nN 2 R 1.0 2.0\n1 1.0 0.0 6.0\n2 2.0 0.0 8.0\n",
        encoding="utf-8",
    )

    dihedral = read_lammps_table_section(
        path,
        table_name="DIH",
        table_style="dihedral",
    )
    pair = read_lammps_table_section(path, table_name="PAIR")

    np.testing.assert_allclose(dihedral.x, np.degrees([-1.0, 1.0]))
    np.testing.assert_allclose(dihedral.force, np.array([2.0, 4.0]) * np.pi / 180.0)
    np.testing.assert_array_equal(pair.x, [1.0, 2.0])
    np.testing.assert_array_equal(pair.force, [6.0, 8.0])


def test_multi_section_selection_and_metadata_are_section_local(tmp_path):
    path = tmp_path / "multi.table"
    path.write_text(
        "\n".join(
            [
                "# ACECG-TABLE boundary_mode=periodic n_coeffs=8",
                "FIRST",
                "N 2 DEGREES",
                "",
                "1 -180 0 0",
                "2 0 0 0",
                "",
                "# ACECG-TABLE boundary_mode=cutoff n_coeffs=6",
                "SECOND",
                "N 2 DEGREES",
                "",
                "1 -90 0 0",
                "2 90 0 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple table sections"):
        parse_lammps_table(path, table_style="dihedral")
    first = read_lammps_table_section(
        path,
        table_name="FIRST",
        table_style="dihedral",
    )
    second = read_lammps_table_section(
        path,
        table_name="SECOND",
        table_style="dihedral",
    )
    assert first.metadata["boundary_mode"] == "periodic"
    assert first.metadata["n_coeffs"] == "8"
    assert second.metadata["boundary_mode"] == "cutoff"
    assert second.metadata["n_coeffs"] == "6"


def test_numeric_underscore_table_keyword_is_not_misread_as_a_number(tmp_path):
    path = tmp_path / "numeric_keyword.table"
    path.write_text(
        "\n".join(
            [
                "1_2",
                "N 2 R 1.0 2.0",
                "",
                "1 1.0 0.0 1.0",
                "2 2.0 0.0 -1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    section = read_lammps_table_section(path, table_name="1_2")
    assert section.keyword == "1_2"
    np.testing.assert_array_equal(section.x, [1.0, 2.0])


def test_table_header_accepts_n_parameter_in_any_order(tmp_path):
    path = tmp_path / "reordered_header.table"
    path.write_text(
        "\n".join(
            [
                "DIH",
                "DEGREES N 2",
                "",
                "1 -90.0 0.0 1.0",
                "2 90.0 0.0 -1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    section = read_lammps_table_section(
        path,
        table_name="DIH",
        table_style="dihedral",
    )
    np.testing.assert_array_equal(section.x, [-90.0, 90.0])
    np.testing.assert_array_equal(section.force, [1.0, -1.0])


def test_dihedral_fitter_round_trip_and_explicit_size_override(tmp_path):
    source = _periodic_potential(n_coeffs=8)
    grid = np.arange(-180.0, 180.0, 1.0)
    path = tmp_path / "fit.table"
    write_lammps_table(
        path,
        grid,
        source.value(grid),
        source.force(grid),
        table_name="DIH",
        table_style="dihedral",
        metadata=source.table_metadata(),
    )

    fitted = DihedralBSplineTableFitter(
        n_coeffs=7,
        degree=3,
        boundary_mode="periodic",
        minimum=-180.0,
        maximum=180.0,
    ).fit(
        str(path),
        typ1=TYPES[0],
        typ2=TYPES[-1],
        types=TYPES,
        table_name="DIH",
    )
    assert fitted.n_params() == 7
    assert np.sqrt(np.mean((fitted.force(grid) - source.force(grid)) ** 2)) < 0.01


def test_named_dihedral_spec_requires_explicit_boundary_mode():
    base = {
        "types": list(TYPES),
        "model": "bspline",
        "n_coeffs": 12,
        "degree": 3,
        "domain": [-180.0, 180.0],
        "resolution": 2.0,
    }
    with pytest.raises(ACGConfigError, match="boundary_mode"):
        _parse_named_fm_spec(base, expected_style="dihedral", entry_index=0)

    spec = _parse_named_fm_spec(
        {**base, "boundary_mode": "periodic"},
        expected_style="dihedral",
        entry_index=0,
    )
    assert spec.resolution == pytest.approx(2.0)
    assert spec.model_overrides["boundary_mode"] == "periodic"


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("boundary_mode", ["periodic", "cutoff"])
def test_forcefield_dihedral_table_round_trip(tmp_path, hybrid, boundary_mode):
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    table_path = old_dir / "dppc.table"
    settings_path = old_dir / "settings.in"
    source = (
        _periodic_potential()
        if boundary_mode == "periodic"
        else _cutoff_potential()
    )
    grid = np.arange(-180.0, 180.0, 2.0)
    write_lammps_table(
        table_path,
        grid,
        source.value(grid),
        source.force(grid),
        table_name="DPPC_DIH",
        table_style="dihedral",
        metadata=source.table_metadata(),
    )
    style_token = "table " if hybrid else ""
    settings_path.write_text(
        f"dihedral_coeff 1 {style_token}dppc.table DPPC_DIH\n",
        encoding="utf-8",
    )
    key = InteractionKey.dihedral(*TYPES)
    topology = SimpleNamespace(
        atom_type_code_to_name={},
        bond_type_id_to_key={},
        angle_type_id_to_key={},
        dihedral_type_id_to_key={0: key},
        key_to_bonded_type_id={key: 0},
    )

    forcefield = ReadLmpFF(
        str(settings_path),
        pair_style="table",
        table_fit="bspline",
        topology_arrays=topology,
    )
    fitted = forcefield[key][0]
    assert isinstance(fitted, DihedralBSplinePotential)
    assert fitted.boundary_mode == boundary_mode

    new_settings = new_dir / "settings.in"
    WriteLmpFF(
        str(settings_path),
        str(new_settings),
        forcefield,
        pair_style="table",
        topology_arrays=topology,
    )
    new_grid, _, new_force = parse_lammps_table(
        new_dir / "dppc.table",
        table_name="DPPC_DIH",
        table_style="dihedral",
    )
    np.testing.assert_array_equal(new_grid, grid)
    np.testing.assert_allclose(new_force, fitted.force(grid), atol=1.0e-8)


@pytest.mark.parametrize(
    "coefficient_prefix",
    [
        "aat 1 177 180",
        "table/cut aat 1 177 180",
    ],
)
def test_forcefield_table_cut_dihedral_round_trip(tmp_path, coefficient_prefix):
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    table_path = old_dir / "dppc.table"
    settings_path = old_dir / "settings.in"
    source = _periodic_potential()
    grid = np.arange(-180.0, 180.0, 2.0)
    write_lammps_table(
        table_path,
        grid,
        source.value(grid),
        source.force(grid),
        table_name="DPPC_DIH",
        table_style="dihedral",
        metadata=source.table_metadata(),
    )
    settings_path.write_text(
        f"dihedral_coeff 1 {coefficient_prefix} dppc.table DPPC_DIH\n",
        encoding="utf-8",
    )
    key = InteractionKey.dihedral(*TYPES)
    topology = SimpleNamespace(
        atom_type_code_to_name={},
        bond_type_id_to_key={},
        angle_type_id_to_key={},
        dihedral_type_id_to_key={0: key},
        key_to_bonded_type_id={key: 0},
    )

    forcefield = ReadLmpFF(
        str(settings_path),
        pair_style="table",
        table_fit="bspline",
        topology_arrays=topology,
    )
    assert isinstance(forcefield[key][0], DihedralBSplinePotential)

    new_settings = new_dir / "settings.in"
    WriteLmpFF(
        str(settings_path),
        str(new_settings),
        forcefield,
        pair_style="table",
        topology_arrays=topology,
    )
    assert new_settings.read_text(encoding="utf-8") == settings_path.read_text(
        encoding="utf-8"
    )
    new_grid, _, new_force = parse_lammps_table(
        new_dir / "dppc.table",
        table_name="DPPC_DIH",
        table_style="dihedral",
    )
    np.testing.assert_array_equal(new_grid, grid)
    np.testing.assert_allclose(
        new_force,
        forcefield[key][0].force(grid),
        atol=1.0e-8,
    )


def test_non_table_dihedral_coeff_is_not_misread_as_a_filename(tmp_path):
    settings_path = tmp_path / "settings.in"
    settings_path.write_text(
        "dihedral_coeff 1 harmonic 3.5 1 2\n",
        encoding="utf-8",
    )
    key = InteractionKey.dihedral(*TYPES)
    topology = SimpleNamespace(
        atom_type_code_to_name={},
        bond_type_id_to_key={},
        angle_type_id_to_key={},
        dihedral_type_id_to_key={0: key},
        key_to_bonded_type_id={key: 0},
    )

    forcefield = ReadLmpFF(
        str(settings_path),
        pair_style="table",
        table_fit="bspline",
        topology_arrays=topology,
    )
    assert key not in forcefield


@pytest.mark.parametrize(
    ("kind", "section", "width"),
    [
        ("bond", "Bonds", 2),
        ("angle", "Angles", 3),
        ("dihedral", "Dihedrals", 4),
    ],
)
def test_duplicate_lammps_bonded_types_are_rejected(
    tmp_path,
    kind,
    section,
    width,
):
    data_path = tmp_path / f"duplicate_{kind}_types.data"
    n_atoms = 2 * width
    atom_rows = [
        f"{atom_id} {1 if atom_id <= width else 2} 1 0.0 {atom_id}.0 0.0 0.0"
        for atom_id in range(1, n_atoms + 1)
    ]
    first_atoms = " ".join(str(atom_id) for atom_id in range(1, width + 1))
    second_atoms = " ".join(
        str(atom_id) for atom_id in range(width + 1, n_atoms + 1)
    )
    data_path.write_text(
        "\n".join(
            [
                "LAMMPS data file",
                "",
                f"{n_atoms} atoms",
                f"2 {kind}s",
                "",
                "1 atom types",
                f"2 {kind} types",
                "",
                "0.0 20.0 xlo xhi",
                "0.0 20.0 ylo yhi",
                "0.0 20.0 zlo zhi",
                "",
                "Masses",
                "",
                "1 1.0",
                "",
                "Atoms # full",
                "",
                *atom_rows,
                "",
                section,
                "",
                f"1 1 {first_atoms}",
                f"2 2 {second_atoms}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    universe = mda.Universe(str(data_path), format="DATA")
    with pytest.raises(ValueError, match="one-to-one"):
        collect_topology_arrays(
            universe,
            atom_type_name_aliases={1: "A"},
        )
