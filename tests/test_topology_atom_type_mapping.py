from __future__ import annotations

import MDAnalysis as mda
import numpy as np
import pytest

from AceCG.topology.neighbor import compute_pairs_by_type
from AceCG.topology.topology_array import collect_topology_arrays
from AceCG.topology.types import InteractionKey


def _make_universe(
    types: list[str],
    *,
    names: list[str] | None = None,
) -> mda.Universe:
    n_atoms = len(types)
    u = mda.Universe.empty(
        n_atoms,
        n_residues=1,
        atom_resindex=np.zeros(n_atoms, dtype=np.int64),
        trajectory=True,
    )
    u.add_TopologyAttr("types", np.asarray(types, dtype=object))
    if names is not None:
        u.add_TopologyAttr("names", np.asarray(names, dtype=object))
    u.add_TopologyAttr("masses", np.ones(n_atoms, dtype=np.float64))
    u.add_TopologyAttr("charges", np.zeros(n_atoms, dtype=np.float64))
    u.add_TopologyAttr("resids", np.array([1], dtype=np.int64))
    u.atoms.positions = np.column_stack(
        (
            np.arange(n_atoms, dtype=np.float32),
            np.zeros(n_atoms, dtype=np.float32),
            np.zeros(n_atoms, dtype=np.float32),
        )
    )
    u.dimensions = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0], dtype=np.float32)
    return u


def test_case1_pair_translation_uses_atom_types_not_names():
    u = _make_universe(["A", "B"], names=["foo", "bar"])
    topo = collect_topology_arrays(u, exclude_option="none")

    np.testing.assert_array_equal(topo.atom_type_names, np.array(["A", "B"]))
    np.testing.assert_array_equal(topo.atom_type_codes, np.array([1, 2], dtype=np.int32))

    pair_cache = compute_pairs_by_type(
        positions=u.atoms.positions,
        box=u.dimensions,
        pair_type_list=[InteractionKey.pair("A", "B")],
        cutoff=5.0,
        topology_arrays=topo,
        exclude_option="none",
    )

    a_idx, b_idx = pair_cache[InteractionKey.pair("A", "B")]
    np.testing.assert_array_equal(a_idx, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(b_idx, np.array([1], dtype=np.int32))


def test_case2_aliases_follow_numeric_order_with_partial_fallback():
    u = _make_universe(["10", "2", "1"])
    topo = collect_topology_arrays(
        u,
        exclude_option="none",
        atom_type_name_aliases={10: "T10", 1: "T1"},
    )

    np.testing.assert_array_equal(
        topo.atom_type_names,
        np.array(["T1", "2", "T10"], dtype=object),
    )
    np.testing.assert_array_equal(topo.atom_type_codes, np.array([10, 2, 1], dtype=np.int32))
    assert topo.atom_type_name_to_code == {"T1": 1, "2": 2, "T10": 10}
    assert topo.atom_type_code_to_name == {1: "T1", 2: "2", 10: "T10"}
    np.testing.assert_array_equal(
        topo.names,
        np.array(["T10", "2", "T1"], dtype=object),
    )


def test_case2_rejects_duplicate_alias_names():
    u = _make_universe(["1", "2"])

    with pytest.raises(ValueError, match="unique canonical name"):
        collect_topology_arrays(
            u,
            exclude_option="none",
            atom_type_name_aliases={1: "C", 2: "C"},
        )


def test_case3_keeps_raw_lammps_type_ids_with_gaps():
    u = _make_universe(["4", "2", "4"])
    topo = collect_topology_arrays(u, exclude_option="none")

    np.testing.assert_array_equal(
        topo.atom_type_names,
        np.array(["2", "4"], dtype=object),
    )
    np.testing.assert_array_equal(topo.atom_type_codes, np.array([4, 2, 4], dtype=np.int32))
    assert topo.atom_type_name_to_code == {"2": 2, "4": 4}
    assert topo.atom_type_code_to_name == {2: "2", 4: "4"}
