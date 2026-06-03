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
