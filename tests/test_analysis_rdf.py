from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from AceCG.analysis.rdf import (
    accumulate_distribution_frame,
    finalize_distribution_state,
    init_distribution_state,
)
from AceCG.topology.topology_array import TopologyArrays
from AceCG.topology.types import InteractionKey


def _minimal_topology_arrays() -> TopologyArrays:
    return TopologyArrays(
        n_atoms=2,
        names=np.array(["A", "A"], dtype=object),
        types=np.array(["A", "A"], dtype=object),
        atom_type_names=np.array(["A"], dtype=object),
        atom_type_codes=np.array([1, 1], dtype=np.int32),
        n_residues=1,
        atom_resindex=np.array([0, 0], dtype=np.int32),
        masses=np.array([1.0, 1.0], dtype=np.float32),
        charges=np.array([0.0, 0.0], dtype=np.float32),
        resids=np.array([1], dtype=np.int32),
        molnums=np.array([0, 0], dtype=np.int32),
        bonds=np.empty((0, 2), dtype=np.int32),
        angles=np.empty((0, 3), dtype=np.int32),
        dihedrals=np.empty((0, 4), dtype=np.int32),
        exclude_12=np.empty((0,), dtype=np.int32),
        exclude_13=np.empty((0,), dtype=np.int32),
        exclude_14=np.empty((0,), dtype=np.int32),
        excluded_nb=np.empty((0,), dtype=np.int32),
        excluded_nb_mode="none",
        excluded_nb_all=False,
        real_site_indices=np.array([0, 1], dtype=np.int32),
        virtual_site_mask=np.array([False, False], dtype=bool),
        virtual_site_indices=np.empty((0,), dtype=np.int32),
        bond_key_index=np.empty((0,), dtype=np.int32),
        angle_key_index=np.empty((0,), dtype=np.int32),
        dihedral_key_index=np.empty((0,), dtype=np.int32),
        keys_bondtypes=[],
        keys_angletypes=[],
        keys_dihedraltypes=[],
        atom_type_name_to_code={"A": 1},
        atom_type_code_to_name={1: "A"},
        bond_type_id_to_key={},
        angle_type_id_to_key={},
        dihedral_type_id_to_key={},
        key_to_bonded_type_id={},
    )


def _frame_geometry(pair_key: InteractionKey, distance: float):
    return SimpleNamespace(
        pair_distances={pair_key: np.array([distance], dtype=np.float64)},
        bond_distances={},
        angle_values={},
        dihedral_values={},
        box=np.array([3.0, 3.0, 3.0, 90.0, 90.0, 90.0], dtype=np.float64),
    )


def _distribution_topology_arrays() -> tuple[
    TopologyArrays, InteractionKey, InteractionKey, InteractionKey, InteractionKey
]:
    pair_key = InteractionKey.pair("A", "B")
    bond_key = InteractionKey.bond("A", "B")
    angle_key = InteractionKey.angle("A", "B", "C")
    dihedral_key = InteractionKey.dihedral("A", "B", "C", "D")
    topology = TopologyArrays(
        n_atoms=4,
        names=np.array(["A", "B", "C", "D"], dtype=object),
        types=np.array(["A", "B", "C", "D"], dtype=object),
        atom_type_names=np.array(["A", "B", "C", "D"], dtype=object),
        atom_type_codes=np.array([1, 2, 3, 4], dtype=np.int32),
        n_residues=1,
        atom_resindex=np.zeros(4, dtype=np.int32),
        masses=np.ones(4, dtype=np.float32),
        charges=np.zeros(4, dtype=np.float32),
        resids=np.array([1], dtype=np.int32),
        molnums=np.zeros(4, dtype=np.int32),
        bonds=np.array([[0, 1]], dtype=np.int32),
        angles=np.array([[0, 1, 2]], dtype=np.int32),
        dihedrals=np.array([[0, 1, 2, 3]], dtype=np.int32),
        exclude_12=np.empty((0,), dtype=np.int32),
        exclude_13=np.empty((0,), dtype=np.int32),
        exclude_14=np.empty((0,), dtype=np.int32),
        excluded_nb=np.empty((0,), dtype=np.int32),
        excluded_nb_mode="none",
        excluded_nb_all=False,
        real_site_indices=np.arange(4, dtype=np.int32),
        virtual_site_mask=np.zeros(4, dtype=bool),
        virtual_site_indices=np.empty((0,), dtype=np.int32),
        bond_key_index=np.array([0], dtype=np.int32),
        angle_key_index=np.array([0], dtype=np.int32),
        dihedral_key_index=np.array([0], dtype=np.int32),
        keys_bondtypes=[bond_key],
        keys_angletypes=[angle_key],
        keys_dihedraltypes=[dihedral_key],
        atom_type_name_to_code={"A": 1, "B": 2, "C": 3, "D": 4},
        atom_type_code_to_name={1: "A", 2: "B", 3: "C", 4: "D"},
        bond_type_id_to_key={1: bond_key},
        angle_type_id_to_key={1: angle_key},
        dihedral_type_id_to_key={1: dihedral_key},
        key_to_bonded_type_id={bond_key: 1, angle_key: 1, dihedral_key: 1},
    )
    return topology, pair_key, bond_key, angle_key, dihedral_key


def _geometry_with_distributions(
    pair_key: InteractionKey,
    bond_key: InteractionKey,
    angle_key: InteractionKey,
    dihedral_key: InteractionKey,
    *,
    pair_distance: float,
    bond_distance: float,
    angle: float,
    dihedral: float,
):
    return SimpleNamespace(
        pair_distances={pair_key: np.array([pair_distance], dtype=np.float64)},
        bond_distances={bond_key: np.array([bond_distance], dtype=np.float64)},
        angle_values={angle_key: np.array([angle], dtype=np.float64)},
        dihedral_values={dihedral_key: np.array([dihedral], dtype=np.float64)},
        box=np.array([4.0, 4.0, 4.0, 90.0, 90.0, 90.0], dtype=np.float64),
    )


def test_pair_rdf_accumulator_normalizes_against_shell_counts() -> None:
    pair_key = InteractionKey.pair("A", "A")
    state = init_distribution_state(
        _minimal_topology_arrays(),
        {pair_key: object()},
        cutoff=3.0,
        nbins_pair=3,
    )

    accumulate_distribution_frame(state, _frame_geometry(pair_key, distance=1.25), frame_weight=2.0)
    result = finalize_distribution_state(state)[pair_key]

    shell_vol = (4.0 / 3.0) * np.pi * (2.0**3 - 1.0**3)
    assert result.mode == "rdf"
    assert result.n_frames == 1
    assert result.weight_sum == 2.0
    np.testing.assert_allclose(result.values, np.array([0.0, 27.0 / shell_vol, 0.0]))


def test_pair_pdf_mode_normalizes_histogram_mass() -> None:
    pair_key = InteractionKey.pair("A", "A")
    state = init_distribution_state(
        _minimal_topology_arrays(),
        {pair_key: object()},
        cutoff=3.0,
        nbins_pair=3,
        default_pair_mode="pdf",
    )

    accumulate_distribution_frame(state, _frame_geometry(pair_key, distance=2.25), frame_weight=1.0)
    result = finalize_distribution_state(state)[pair_key]

    assert result.mode == "pdf"
    np.testing.assert_allclose(result.values, np.array([0.0, 0.0, 1.0]))


def test_weighted_multiple_frame_cross_type_rdf_uses_cross_shell_count() -> None:
    topology, pair_key, bond_key, angle_key, dihedral_key = _distribution_topology_arrays()
    state = init_distribution_state(
        topology,
        {pair_key: object()},
        cutoff=3.0,
        nbins_pair=3,
    )
    for distance, weight in ((1.25, 2.0), (2.25, 3.0)):
        accumulate_distribution_frame(
            state,
            _geometry_with_distributions(
                pair_key,
                bond_key,
                angle_key,
                dihedral_key,
                pair_distance=distance,
                bond_distance=1.0,
                angle=np.pi / 2.0,
                dihedral=0.0,
            ),
            frame_weight=weight,
        )
    result = finalize_distribution_state(state)[pair_key]

    shell_volumes = (4.0 / 3.0) * np.pi * np.array([1.0, 7.0, 19.0])
    expected = np.array([0.0, 2.0, 3.0]) / (5.0 * shell_volumes / 64.0)
    np.testing.assert_allclose(result.counts, [0.0, 2.0, 3.0])
    np.testing.assert_allclose(result.values, expected)
    assert result.n_frames == 2
    assert result.weight_sum == 5.0


def test_bond_angle_and_dihedral_pdf_units_and_folding() -> None:
    topology, pair_key, bond_key, angle_key, dihedral_key = _distribution_topology_arrays()
    # angle_values/dihedral_values are degrees on the real path
    # (compute_frame_geometry uses np.degrees(...); see F-033), so the
    # synthetic geometry must supply degrees too, not radians.
    geometry = _geometry_with_distributions(
        pair_key,
        bond_key,
        angle_key,
        dihedral_key,
        pair_distance=1.0,
        bond_distance=1.25,
        angle=90.0,
        dihedral=-135.0,
    )

    degree_state = init_distribution_state(
        topology,
        {bond_key: object(), angle_key: object(), dihedral_key: object()},
        r_max=2.0,
        nbins_bond=4,
        nbins_angle=4,
        nbins_dihedral=4,
        angle_degrees=True,
        dihedral_degrees=True,
        dihedral_periodic=True,
    )
    accumulate_distribution_frame(degree_state, geometry)
    degree_results = finalize_distribution_state(degree_state)
    for key in (bond_key, angle_key, dihedral_key):
        result = degree_results[key]
        assert result.mode == "pdf"
        np.testing.assert_allclose(np.sum(result.values * np.diff(result.edges)), 1.0)
    np.testing.assert_allclose(degree_results[angle_key].edges, [0.0, 45.0, 90.0, 135.0, 180.0])
    np.testing.assert_allclose(degree_results[dihedral_key].edges, [-180.0, -90.0, 0.0, 90.0, 180.0])
    np.testing.assert_allclose(degree_results[dihedral_key].counts, [1.0, 0.0, 0.0, 0.0])

    radian_state = init_distribution_state(
        topology,
        {angle_key: object(), dihedral_key: object()},
        nbins_angle=4,
        nbins_dihedral=4,
        angle_degrees=False,
        dihedral_degrees=False,
        dihedral_periodic=False,
    )
    accumulate_distribution_frame(radian_state, geometry)
    radian_results = finalize_distribution_state(radian_state)
    np.testing.assert_allclose(radian_results[angle_key].edges, np.linspace(0.0, np.pi, 5))
    np.testing.assert_allclose(radian_results[dihedral_key].edges, np.linspace(0.0, np.pi, 5))
    np.testing.assert_allclose(radian_results[dihedral_key].counts, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        np.sum(radian_results[dihedral_key].values * np.diff(radian_results[dihedral_key].edges)),
        1.0,
    )
