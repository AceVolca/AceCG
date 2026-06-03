"""Tests for the no-frame-cache compute stack.

Covers:
- FrameGeometry construction
- energy() and force() kernels
- MPIComputeEngine local compute semantics
- one-pass run_post() multi-step behavior
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import AceCG.compute.mpi_engine as mpi_engine_module
from AceCG.compute.energy import energy
from AceCG.compute.force import force
from AceCG.compute.frame_geometry import (
    FrameGeometry,
    compute_frame_geometry,
)
from AceCG.compute.mpi_engine import (
    FrameCache,
    MPIComputeEngine,
    TrajectoryCache,
    _write_rank_heartbeat,
    build_default_engine,
)
from AceCG.compute.reducers import (
    consume_step_payload,
    finalize_step_root,
    init_step_state,
    local_step_partials,
    step_request,
)
from AceCG.potentials.bspline import BSplinePotential
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey


def _request(*names):
    return frozenset(names)


class _DummyForcefield:
    key_mask = None

    def keys(self):
        return []


def test_compute_skin_neighbor_mode_uses_reference_frame(monkeypatch):
    pair_key = InteractionKey.pair("A", "A")
    calls = []
    geometry_caches = []

    def fake_pairs_by_type(**kwargs):
        calls.append(
            {
                "positions": np.asarray(kwargs["positions"]).copy(),
                "box": np.asarray(kwargs["box"]).copy(),
                "cutoff": float(kwargs["cutoff"]),
            }
        )
        return {pair_key: (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))}

    def fake_compute_frame_geometry(*args, **kwargs):
        geometry_caches.append(kwargs.get("pair_cache"))
        return SimpleNamespace()

    monkeypatch.setattr(mpi_engine_module, "compute_pairs_by_type", fake_pairs_by_type)
    monkeypatch.setattr(mpi_engine_module, "compute_frame_geometry", fake_compute_frame_geometry)

    positions = np.zeros((3, 2, 3), dtype=np.float64)
    positions[0, :, 0] = 1.0
    reference_positions = np.full((2, 3), 7.0, dtype=np.float64)
    box = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0], dtype=np.float64)

    MPIComputeEngine().compute(
        request=_request(),
        frame=(0, positions, box, None),
        topology_arrays=SimpleNamespace(),
        forcefield_snapshot=_DummyForcefield(),
        pair_type_list=[pair_key],
        pair_cutoff=10.0,
        batch_size=2,
        neighbor_mode="skin",
        neighbor_skin=0.5,
        neighbor_reference_positions=reference_positions,
        neighbor_reference_box=box,
    )

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0]["positions"], reference_positions)
    assert calls[0]["cutoff"] == pytest.approx(10.5)
    assert len(geometry_caches) == 2
    assert geometry_caches[0] is geometry_caches[1]


def test_compute_chunk_neighbor_mode_rebuilds_per_batch_chunk(monkeypatch):
    pair_key = InteractionKey.pair("A", "A")
    starts = []

    def fake_pairs_by_type(**kwargs):
        positions = np.asarray(kwargs["positions"])
        starts.append(float(positions[0, 0]))
        assert float(kwargs["cutoff"]) == pytest.approx(10.25)
        return {pair_key: (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))}

    monkeypatch.setattr(mpi_engine_module, "compute_pairs_by_type", fake_pairs_by_type)
    monkeypatch.setattr(
        mpi_engine_module,
        "compute_frame_geometry",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    positions = np.zeros((5, 2, 3), dtype=np.float64)
    positions[:, 0, 0] = np.arange(5, dtype=np.float64)
    box = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0], dtype=np.float64)

    MPIComputeEngine().compute(
        request=_request(),
        frame=(0, positions, box, None),
        topology_arrays=SimpleNamespace(),
        forcefield_snapshot=_DummyForcefield(),
        pair_type_list=[pair_key],
        pair_cutoff=10.0,
        batch_size=2,
        neighbor_mode="chunk",
        neighbor_skin=0.25,
    )

    assert starts == [0.0, 2.0, 4.0]


def test_dsm_reducer_requests_fm_stats_without_reference_force():
    request = step_request({"step_mode": "dsm"})

    assert "fm_stats" in request
    assert "reference_force" not in request


def test_rem_reducer_requests_gauge_free_gradient_but_cdrem_uses_physical():
    rem_request = step_request({"step_mode": "rem", "need_hessian": True})
    cdrem_request = step_request({"step_mode": "cdrem", "need_hessian": True})

    assert "gauge_free_energy_grad" in rem_request
    assert "gauge_free_energy_grad_outer" in rem_request
    assert "energy_grad" not in rem_request
    assert "energy_grad_outer" not in rem_request

    assert "energy_grad" in cdrem_request
    assert "energy_grad_outer" in cdrem_request
    assert "gauge_free_energy_grad" not in cdrem_request


def test_rem_reducer_consumes_gauge_free_payload_and_marks_convention():
    bond_key = InteractionKey.bond("A", "B")
    ff = Forcefield({bond_key: [_bonded_bspline()]})
    topo = SimpleNamespace()
    step = {"step_mode": "rem", "need_hessian": True}
    grad = np.array([1.0, 2.0], dtype=np.float64)
    state = init_step_state(step, ff, topo)

    consume_step_payload(
        step,
        state,
        payload={
            "frame_idx": 7,
            "gauge_free_energy_grad": grad,
            "gauge_free_energy_grad_outer": np.outer(grad, grad),
            "energy_hessian": np.eye(2, dtype=np.float64),
        },
        frame_weight=2.0,
        reference_force=None,
    )
    result = finalize_step_root(step, local_step_partials(step, state))

    np.testing.assert_allclose(result["energy_grad_avg"], grad)
    assert result["gradient_convention"] == "gauge_free"
    np.testing.assert_allclose(result["grad_outer_avg"], np.outer(grad, grad))


def test_cdrem_reducer_keeps_physical_gradient_convention():
    bond_key = InteractionKey.bond("A", "B")
    ff = Forcefield({bond_key: [_bonded_bspline()]})
    step = {"step_mode": "cdrem", "need_hessian": False}
    grad = np.array([3.0, 4.0], dtype=np.float64)
    state = init_step_state(step, ff, SimpleNamespace())

    consume_step_payload(
        step,
        state,
        payload={"frame_idx": 3, "energy_grad": grad},
        frame_weight=1.0,
        reference_force=None,
    )
    result = finalize_step_root(step, local_step_partials(step, state))

    np.testing.assert_allclose(result["energy_grad_avg"], grad)
    assert result["gradient_convention"] == "physical"


def test_cdfm_zbx_finalize_reinjects_reinforce_controls_after_reduction():
    step = {
        "step_mode": "cdfm_zbx",
        "y_eff": np.zeros(2, dtype=np.float64),
        "mode": "reinforce",
        "beta": 0.5,
    }
    reduced_state = {
        "J_sum": np.zeros((2, 2), dtype=np.float64),
        "f_sum": np.array([1.0, 2.0], dtype=np.float64),
        "gu_sum": np.array([3.0, 5.0], dtype=np.float64),
        "gu_f_sum": np.array([[7.0, 11.0], [13.0, 17.0]], dtype=np.float64),
        "weight_sum": 1.0,
        "n_samples": 4,
        "obs_rows": 2,
    }

    result = finalize_step_root(step, reduced_state)

    np.testing.assert_allclose(result["grad_direct"], np.zeros(2, dtype=np.float64))
    np.testing.assert_allclose(
        result["grad_reinforce"],
        np.array([-7.0, -11.0], dtype=np.float64),
    )


def _bonded_bspline() -> BSplinePotential:
    return BSplinePotential(
        "A",
        "B",
        knots=np.array([0.0, 0.0, 2.0, 2.0], dtype=float),
        coefficients=np.array([1.0, -1.0], dtype=float),
        degree=1,
        cutoff=2.0,
        bonded=True,
    )


def _empty_topo():
    return SimpleNamespace(
        n_atoms=0,
        bonds=None,
        bond_key_index=None,
        keys_bondtypes=[],
        angles=None,
        angle_key_index=None,
        keys_angletypes=[],
        dihedrals=None,
        dihedral_key_index=None,
        keys_dihedraltypes=[],
        real_site_indices=None,
    )


def test_write_rank_heartbeat_writes_progress_json(tmp_path):
    start_time = mpi_engine_module.time.monotonic() - 2.0

    _write_rank_heartbeat(
        tmp_path,
        rank=3,
        size=8,
        processed=5,
        local_total=10,
        global_total=80,
        frame_id=42,
        start_time=start_time,
    )

    payload = mpi_engine_module.json.loads(
        (tmp_path / "progress_rank_0003.json").read_text(encoding="utf-8")
    )
    assert payload["rank"] == 3
    assert payload["size"] == 8
    assert payload["processed"] == 5
    assert payload["local_total"] == 10
    assert payload["global_total"] == 80
    assert payload["last_frame_id"] == 42
    assert payload["frames_per_sec"] > 0.0


def _bond_topo(bond_key, bonds, bond_key_index):
    return SimpleNamespace(
        n_atoms=int(np.max(bonds)) + 1 if bonds.size else 0,
        bonds=bonds,
        bond_key_index=bond_key_index,
        keys_bondtypes=[bond_key],
        angles=np.empty((0, 3), dtype=np.int64),
        angle_key_index=np.empty(0, dtype=np.int32),
        keys_angletypes=[],
        dihedrals=np.empty((0, 4), dtype=np.int64),
        dihedral_key_index=np.empty(0, dtype=np.int32),
        keys_dihedraltypes=[],
        real_site_indices=None,
    )


@pytest.fixture()
def bond_key():
    return InteractionKey.bond("A", "B")


@pytest.fixture()
def angle_key():
    return InteractionKey.angle("A", "B", "A")


@pytest.fixture()
def harmonic_bond():
    return HarmonicPotential("A", "B", k=10.0, r0=5.0)


@pytest.fixture()
def harmonic_angle():
    return HarmonicPotential("A", "B", k=2.0, r0=120.0, typ3="A")


@pytest.fixture()
def simple_forcefield(bond_key, harmonic_bond):
    return Forcefield({bond_key: [harmonic_bond]})


@pytest.fixture()
def simple_frame_geometry(bond_key):
    positions = np.array(
        [[0, 0, 0], [4, 0, 0], [8, 0, 0], [12, 0, 0]],
        dtype=np.float64,
    )
    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
    bond_indices = np.array([[0, 1], [2, 3]], dtype=np.int64)
    bond_vectors = positions[bond_indices[:, 1]] - positions[bond_indices[:, 0]]
    bond_distances = np.linalg.norm(bond_vectors, axis=1)
    return FrameGeometry(
        pair_distances={},
        pair_indices={},
        pair_vectors={},
        bond_distances={bond_key: bond_distances},
        bond_vectors={bond_key: bond_vectors},
        bond_indices={bond_key: bond_indices},
        angle_values={},
        angle_indices={},
        dihedral_values={},
        dihedral_indices={},
        positions=positions,
        box=box,
        n_atoms=4,
        real_site_indices=None,
    )


@pytest.fixture()
def angle_frame_geometry(angle_key):
    positions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
    angle_indices_arr = np.array([[0, 1, 2]], dtype=np.int64)
    angle_vals = np.array([90.0], dtype=np.float64)
    return FrameGeometry(
        pair_distances={},
        pair_indices={},
        pair_vectors={},
        bond_distances={},
        bond_vectors={},
        bond_indices={},
        angle_values={angle_key: angle_vals},
        angle_indices={angle_key: angle_indices_arr},
        dihedral_values={},
        dihedral_indices={},
        positions=positions,
        box=box,
        n_atoms=3,
        real_site_indices=None,
    )


class TestFrameGeometry:
    def test_frozen(self, simple_frame_geometry):
        with pytest.raises(AttributeError):
            simple_frame_geometry.n_atoms = 99

    def test_compute_bond_geometry(self, bond_key):
        positions = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0]], dtype=np.float64)
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1], [1, 2]], dtype=np.int64),
            bond_key_index=np.array([0, 0], dtype=np.int32),
        )
        geom = compute_frame_geometry(positions, box, topo)
        np.testing.assert_allclose(geom.bond_distances[bond_key], [3.0, 3.0])
        assert geom.n_atoms == 3

    def test_empty_topology(self):
        positions = np.zeros((5, 3), dtype=np.float64)
        box = np.array([50.0, 50.0, 50.0, 90.0, 90.0, 90.0], dtype=np.float64)
        geom = compute_frame_geometry(positions, box, _empty_topo())
        assert len(geom.bond_distances) == 0
        assert len(geom.pair_distances) == 0
        assert geom.n_atoms == 5

    def test_compute_batched_bond_geometry(self, bond_key):
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        geom = compute_frame_geometry(positions, box, topo)
        assert geom.positions.shape[:-2] == (2,)
        assert geom.n_atoms == 2
        np.testing.assert_allclose(geom.bond_distances[bond_key], [[3.0], [4.0]])

    def test_compute_geometry_preserves_arbitrary_batch_dims_with_shared_indices(
        self,
        bond_key,
        harmonic_bond,
    ):
        distances = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        positions = np.zeros(distances.shape + (2, 3), dtype=np.float64)
        positions[..., 1, 0] = distances
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})
        pair_key = InteractionKey.pair("A", "B")

        geom = compute_frame_geometry(
            positions,
            box,
            topo,
            pair_cache={
                pair_key: (
                    np.array([0], dtype=np.int64),
                    np.array([1], dtype=np.int64),
                )
            },
        )

        assert geom.positions.shape[:-2] == distances.shape
        assert geom.box.shape == distances.shape + (6,)
        assert geom.pair_indices[pair_key][0].shape == (1,)
        assert geom.pair_indices[pair_key][1].shape == (1,)
        assert geom.bond_indices[bond_key].shape == (1, 2)
        np.testing.assert_allclose(geom.pair_distances[pair_key][..., 0], distances)
        np.testing.assert_allclose(geom.bond_distances[bond_key][..., 0], distances)

        e_result = energy(geom, ff, return_value=True, return_grad=True)
        f_result = force(geom, ff, return_value=True, return_grad=True)

        assert e_result["energy"].shape == distances.shape
        assert e_result["energy_grad"].shape == distances.shape + (2,)
        assert f_result["force"].shape == distances.shape + (6,)
        assert f_result["force_grad"].shape == distances.shape + (6, 2)
        stats_result = force(
            geom,
            ff,
            return_fm_stats=True,
            reference_force=np.zeros(distances.shape + (2, 3), dtype=np.float64),
            frame_weights=np.full(distances.shape, 0.25, dtype=np.float64),
        )
        assert stats_result["fm_stats_sum"]["weight_sum"] == pytest.approx(1.0)

        single_geom = compute_frame_geometry(positions[1, 0], box, topo)
        single_energy = energy(single_geom, ff, return_value=True, return_grad=True)
        single_force = force(single_geom, ff, return_value=True, return_grad=True)
        np.testing.assert_allclose(e_result["energy"][1, 0], single_energy["energy"])
        np.testing.assert_allclose(e_result["energy_grad"][1, 0], single_energy["energy_grad"])
        np.testing.assert_allclose(f_result["force"][1, 0], single_force["force"])
        np.testing.assert_allclose(f_result["force_grad"][1, 0], single_force["force_grad"])

    def test_batched_pair_geometry_uses_shared_orthorhombic_minimum_image(self):
        pair_ab = InteractionKey.pair("A", "B")
        pair_ac = InteractionKey.pair("A", "C")
        positions = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [6.0, 0.0, 0.0],
                    [0.0, -6.0, 0.0],
                    [0.0, 0.0, 6.0],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [-6.0, 0.0, 0.0],
                    [0.0, 6.0, 0.0],
                    [0.0, 0.0, -6.0],
                ],
            ],
            dtype=np.float64,
        )
        box = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0], dtype=np.float64)

        geom = compute_frame_geometry(
            positions,
            box,
            _empty_topo(),
            pair_cache={
                pair_ab: (
                    np.array([0, 0], dtype=np.int64),
                    np.array([1, 3], dtype=np.int64),
                ),
                pair_ac: (
                    np.array([0], dtype=np.int64),
                    np.array([2], dtype=np.int64),
                ),
            },
        )

        np.testing.assert_allclose(geom.pair_distances[pair_ab], 4.0)
        np.testing.assert_allclose(geom.pair_distances[pair_ac], 4.0)
        np.testing.assert_allclose(
            geom.pair_vectors[pair_ab],
            [[[-4.0, 0.0, 0.0], [0.0, 0.0, -4.0]], [[4.0, 0.0, 0.0], [0.0, 0.0, 4.0]]],
        )
        np.testing.assert_allclose(
            geom.pair_vectors[pair_ac],
            [[[0.0, 4.0, 0.0]], [[0.0, -4.0, 0.0]]],
        )


class TestForcefieldCaches:
    def test_insert_new_key_rebuilds_cached_param_structure(self):
        key_ab = InteractionKey.bond("A", "B")
        key_bc = InteractionKey.bond("B", "C")
        ff = Forcefield({key_ab: [HarmonicPotential("A", "B", k=10.0, r0=5.0)]})

        ff[key_bc] = [HarmonicPotential("B", "C", k=12.0, r0=6.0)]

        blocks = ff.param_blocks()
        assert [(key, sl.start, sl.stop) for key, _, sl in blocks] == [
            (key_ab, 0, 2),
            (key_bc, 2, 4),
        ]
        assert [(key, pi, sl.start, sl.stop) for key, pi, sl in ff.param_slices()] == [
            (key_ab, 0, 0, 2),
            (key_bc, 0, 2, 4),
        ]
        assert [(sl.start, sl.stop) for sl in ff.interaction_offsets()] == [(0, 2), (2, 4)]


class TestEnergy:
    def test_empty_request(self, simple_frame_geometry, simple_forcefield):
        assert energy(simple_frame_geometry, simple_forcefield) == {}

    def test_energy_value(self, simple_frame_geometry, simple_forcefield):
        result = energy(simple_frame_geometry, simple_forcefield, return_value=True)
        expected = 10.0 * (4.0 - 5.0) ** 2 * 2
        assert abs(result["energy"] - expected) < 1e-12

    def test_energy_grad(self, simple_frame_geometry, simple_forcefield):
        result = energy(simple_frame_geometry, simple_forcefield, return_grad=True)
        grad = result["energy_grad"]
        assert grad.shape == (2,)
        assert abs(grad[0] - 2.0) < 1e-12
        assert abs(grad[1] - 40.0) < 1e-12

    def test_bonded_bspline_gauge_free_energy_grad_preserves_batch_shape(self, bond_key):
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.50, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        pot = _bonded_bspline()
        ff = Forcefield({bond_key: [pot]})
        geom = compute_frame_geometry(positions, box, topo)

        result = energy(
            geom,
            ff,
            return_grad=True,
            return_gauge_free_energy_grad=True,
            return_gauge_free_energy_grad_outer=True,
        )

        distances = np.array([[0.25], [1.50]], dtype=np.float64)
        expected_gauge_free = pot.gauge_free_energy_grad_sum_by_sample(distances)
        np.testing.assert_allclose(
            result["gauge_free_energy_grad"],
            expected_gauge_free,
        )
        assert result["gauge_free_energy_grad"].shape == (2, pot.n_params())
        assert result["gauge_free_energy_grad_outer"].shape == (
            2,
            pot.n_params(),
            pot.n_params(),
        )
        assert not np.allclose(
            result["energy_grad"],
            result["gauge_free_energy_grad"],
        )

    def test_interaction_mask(self, simple_frame_geometry, simple_forcefield, bond_key):
        ff = Forcefield(simple_forcefield)
        ff.key_mask = {bond_key: False}
        result = energy(simple_frame_geometry, ff, return_value=True)
        assert result["energy"] == 0.0

    def test_batched_energy_geometry_matches_single_frame_energy_path(self, bond_key, harmonic_bond):
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})
        geom_batch = compute_frame_geometry(positions, box, topo)
        batch = energy(
            geom_batch,
            ff,
            return_value=True,
            return_grad=True,
            return_hessian=True,
            return_grad_outer=True,
        )

        single_results = []
        for sample_index in range(positions.shape[0]):
            geom = compute_frame_geometry(positions[sample_index], box, topo)
            single_results.append(
                energy(
                    geom,
                    ff,
                    return_value=True,
                    return_grad=True,
                    return_hessian=True,
                    return_grad_outer=True,
                )
        )

        np.testing.assert_allclose(
            batch["energy"],
            [item["energy"] for item in single_results],
        )
        np.testing.assert_allclose(
            batch["energy_grad"],
            np.vstack([item["energy_grad"] for item in single_results]),
        )
        np.testing.assert_allclose(
            batch["energy_hessian"],
            np.stack([item["energy_hessian"] for item in single_results], axis=0),
        )
        np.testing.assert_allclose(
            batch["energy_grad_outer"],
            np.stack([item["energy_grad_outer"] for item in single_results], axis=0),
        )

    def test_energy_accepts_batch_geometry_directly(self, bond_key, harmonic_bond):
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})
        geom_batch = compute_frame_geometry(positions, box, topo)

        direct = energy(
            geom_batch,
            ff,
            return_value=True,
            return_grad=True,
            return_hessian=True,
            return_grad_outer=True,
        )
        assert direct["energy"].shape == (positions.shape[0],)
        assert direct["energy_grad"].shape == (positions.shape[0], 2)
        assert direct["energy_hessian"].shape == (positions.shape[0], 2, 2)
        assert direct["energy_grad_outer"].shape == (positions.shape[0], 2, 2)


class TestForce:
    def test_force_grad_shape(self, simple_frame_geometry, simple_forcefield):
        result = force(simple_frame_geometry, simple_forcefield, return_grad=True)
        assert result["force_grad"].shape == (12, 2)

    def test_force_value(self, simple_frame_geometry, simple_forcefield):
        result = force(simple_frame_geometry, simple_forcefield, return_value=True)
        assert result["force"].shape == (12,)

    def test_force_hessian(self, simple_frame_geometry, simple_forcefield):
        result = force(simple_frame_geometry, simple_forcefield, return_hessian=True)
        eigvals = np.linalg.eigvalsh(result["force_hessian"])
        assert np.all(eigvals >= -1e-10)

    def test_force_fm_stats(self, simple_frame_geometry, simple_forcefield):
        y_ref = np.zeros(12, dtype=np.float64)
        result = force(
            simple_frame_geometry,
            simple_forcefield,
            return_fm_stats=True,
            reference_force=y_ref,
            frame_weight=1.5,
        )
        stats = result["fm_stats"]
        assert stats["JtJ"].shape == (2, 2)
        assert stats["Jty"].shape == (2,)
        assert stats["n_frames"] == 1
        assert stats["weight_sum"] == pytest.approx(1.5)

    def test_angle_force_grad(self, angle_frame_geometry, angle_key, harmonic_angle):
        ff = Forcefield({angle_key: [harmonic_angle]})
        result = force(angle_frame_geometry, ff, return_grad=True)
        J = result["force_grad"]
        assert J.shape == (9, 2)
        assert np.any(np.abs(J) > 1e-10)

    def test_batched_force_geometry_matches_single_frame_angle_dihedral_path(self):
        angle_key = InteractionKey.angle("A", "B", "C")
        dihedral_key = InteractionKey.dihedral("A", "B", "C", "D")
        topo = SimpleNamespace(
            n_atoms=4,
            bonds=np.empty((0, 2), dtype=np.int64),
            bond_key_index=np.empty(0, dtype=np.int32),
            keys_bondtypes=[],
            angles=np.array([[0, 1, 2]], dtype=np.int64),
            angle_key_index=np.array([0], dtype=np.int32),
            keys_angletypes=[angle_key],
            dihedrals=np.array([[0, 1, 2, 3]], dtype=np.int64),
            dihedral_key_index=np.array([0], dtype=np.int32),
            keys_dihedraltypes=[dihedral_key],
            real_site_indices=None,
        )
        positions = np.array(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [1.25, 1.35, 0.75],
                ],
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.1, 0.0],
                    [1.35, 1.25, 0.85],
                ],
            ],
            dtype=np.float64,
        )
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        ff = Forcefield(
            {
                angle_key: [HarmonicPotential("A", "B", k=2.0, r0=105.0, typ3="C")],
                dihedral_key: [HarmonicPotential("A", "B", k=1.5, r0=45.0)],
            }
        )
        reference_forces = np.array(
            [
                np.linspace(-0.3, 0.3, 12),
                np.linspace(0.2, -0.4, 12),
            ],
            dtype=np.float64,
        )
        weights = np.array([0.35, 0.65], dtype=np.float64)

        geom_batch = compute_frame_geometry(positions, box, topo)
        batch = force(
            geom_batch,
            ff,
            return_value=True,
            return_grad=True,
            return_hessian=True,
            reference_force=reference_forces,
            frame_weights=weights,
            return_fm_stats=True,
        )

        singles = []
        stats = []
        for sample_index in range(positions.shape[0]):
            geom = compute_frame_geometry(positions[sample_index], box, topo)
            single = force(
                geom,
                ff,
                return_value=True,
                return_grad=True,
                return_hessian=True,
                reference_force=reference_forces[sample_index],
                frame_weight=float(weights[sample_index]),
                return_fm_stats=True,
            )
            singles.append(single)
            stats.append(single["fm_stats"])

        np.testing.assert_allclose(
            batch["force"],
            np.stack([item["force"] for item in singles]),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            batch["force_grad"],
            np.stack([item["force_grad"] for item in singles]),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            batch["force_hessian"],
            np.stack([item["force_hessian"] for item in singles]),
            rtol=1e-5,
            atol=1e-5,
        )
        partial = batch["fm_stats_sum"]
        for key in ("JtJ", "Jtf", "Jty"):
            np.testing.assert_allclose(partial[key], sum(item[key] for item in stats), rtol=1e-5, atol=1e-5)
        assert partial["ftf"] == pytest.approx(sum(item["ftf"] for item in stats), rel=1e-5, abs=1e-5)
        assert partial["fTy"] == pytest.approx(sum(item["fTy"] for item in stats), rel=1e-5, abs=1e-5)
        assert partial["yty"] == pytest.approx(sum(item["yty"] for item in stats), rel=1e-5, abs=1e-5)
        assert partial["n_frames"] == 1
        assert partial["weight_sum"] == pytest.approx(float(weights.sum()))

    def test_force_accepts_batch_geometry_directly(self, bond_key, harmonic_bond):
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})
        geom_batch = compute_frame_geometry(positions, box, topo)
        reference_forces = np.zeros((positions.shape[0], 6), dtype=np.float64)
        weights = np.array([0.4, 0.6], dtype=np.float64)

        direct = force(
            geom_batch,
            ff,
            return_value=True,
            return_grad=True,
            return_hessian=True,
            reference_force=reference_forces,
            frame_weights=weights,
            return_fm_stats=True,
        )
        assert direct["force"].shape == (positions.shape[0], 6)
        assert direct["force_grad"].shape == (positions.shape[0], 6, 2)
        assert direct["force_hessian"].shape == (positions.shape[0], 2, 2)
        assert direct["fm_stats_sum"]["weight_sum"] == pytest.approx(float(weights.sum()))


class TestMPIComputeEngine:
    def test_compute_force_value_single_frame(self, bond_key, harmonic_bond):
        engine = build_default_engine()
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})
        result = engine.compute(
            request=_request("force"),
            frame=(
                0,
                np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64),
                np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64),
                None,
            ),
            topology_arrays=topo,
            forcefield_snapshot=ff,
        )
        assert result["frame_idx"] == 0
        assert result["force"].shape == (6,)

    def test_compute_fm_stats_single_frame(self, bond_key, harmonic_bond):
        engine = build_default_engine()
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})
        result = engine.compute(
            request=_request("fm_stats"),
            frame=(
                0,
                np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64),
                np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64),
                np.zeros(6, dtype=np.float64),
            ),
            topology_arrays=topo,
            forcefield_snapshot=ff,
            frame_weight=1.5,
        )
        stats = result["fm_stats"]
        assert stats["JtJ"].shape == (2, 2)
        assert stats["Jty"].shape == (2,)
        assert stats["n_frames"] == 1
        assert stats["weight_sum"] == pytest.approx(1.5)

    def test_compute_frame_cache_uses_canonical_and_observable_names(self, bond_key, harmonic_bond):
        engine = build_default_engine()
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})

        result = engine.compute(
            request=_request("frame_cache"),
            frame=(
                7,
                np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64),
                np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64),
                None,
            ),
            topology_arrays=topo,
            forcefield_snapshot=ff,
            return_observables=True,
        )

        assert result["frame_cache"] is result["frame_observables"]
        assert isinstance(result["frame_cache"], FrameCache)
        assert result["frame_cache"].frame_idx == 7
        np.testing.assert_allclose(result["frame_cache"].bond_distances[bond_key], [4.0])

    def test_engine_recomputes_geometry_each_call(self, bond_key, harmonic_bond, monkeypatch):
        engine = build_default_engine()
        positions = np.array([[0, 0, 0], [3.0, 0, 0]], dtype=np.float64)
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})
        frames = [(0, positions, box)]
        call_count = {"n": 0}
        orig = mpi_engine_module.compute_frame_geometry

        def counted(*args, **kwargs):
            call_count["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(mpi_engine_module, "compute_frame_geometry", counted)
        engine.compute(
            request=_request("energy_grad"),
            frame=(frames[0][0], frames[0][1], frames[0][2], None),
            topology_arrays=topo,
            forcefield_snapshot=ff,
        )
        engine.compute(
            request=_request("energy_grad"),
            frame=(frames[0][0], frames[0][1], frames[0][2], None),
            topology_arrays=topo,
            forcefield_snapshot=ff,
        )
        assert call_count["n"] == 2

    def test_compute_accepts_weighted_batch_energy_path(self, bond_key, harmonic_bond):
        engine = build_default_engine()
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})
        request = _request("energy_grad", "energy_hessian", "energy_grad_outer")
        weights = np.array([0.25, 0.75], dtype=np.float64)

        batch = engine.compute(
            request=request,
            frame=(np.array([10, 10], dtype=np.int64), positions, box, None),
            topology_arrays=topo,
            forcefield_snapshot=ff,
            frame_weights=weights,
        )

        singles = []
        for sample_index in range(positions.shape[0]):
            singles.append(
                engine.compute(
                    request=request,
                    frame=(10, positions[sample_index], box, None),
                    topology_arrays=topo,
                    forcefield_snapshot=ff,
                )
            )

        grad_stack = np.vstack([item["energy_grad"] for item in singles])
        hessian_stack = np.stack([item["energy_hessian"] for item in singles], axis=0)
        outer_stack = np.stack([item["energy_grad_outer"] for item in singles], axis=0)
        np.testing.assert_allclose(batch["energy_grad"], weights @ grad_stack)
        np.testing.assert_allclose(
            batch["energy_hessian"],
            np.tensordot(weights, hessian_stack, axes=(0, 0)),
        )
        np.testing.assert_allclose(
            batch["energy_grad_outer"],
            np.tensordot(weights, outer_stack, axes=(0, 0)),
        )

    def test_compute_accepts_weighted_batch_gauge_free_energy_path(self, bond_key):
        engine = build_default_engine()
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.50, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [_bonded_bspline()]})
        request = _request("gauge_free_energy_grad", "gauge_free_energy_grad_outer")
        weights = np.array([0.25, 0.75], dtype=np.float64)

        batch = engine.compute(
            request=request,
            frame=(np.array([10, 10], dtype=np.int64), positions, box, None),
            topology_arrays=topo,
            forcefield_snapshot=ff,
            frame_weights=weights,
        )
        singles = [
            engine.compute(
                request=request,
                frame=(10, positions[sample_index], box, None),
                topology_arrays=topo,
                forcefield_snapshot=ff,
            )
            for sample_index in range(positions.shape[0])
        ]

        grad_stack = np.vstack([item["gauge_free_energy_grad"] for item in singles])
        outer_stack = np.stack(
            [item["gauge_free_energy_grad_outer"] for item in singles],
            axis=0,
        )
        np.testing.assert_allclose(
            batch["gauge_free_energy_grad"],
            weights @ grad_stack,
        )
        np.testing.assert_allclose(
            batch["gauge_free_energy_grad_outer"],
            np.tensordot(weights, outer_stack, axes=(0, 0)),
        )

    def test_compute_accepts_weighted_batch_fm_stats(self, bond_key, harmonic_bond):
        engine = build_default_engine()
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [5.5, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})
        request = _request("fm_stats", "reference_force")
        reference_forces = np.array(
            [
                [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.5, 0.2, 0.0, -0.5, -0.2, 0.0],
                [-0.1, 0.3, 0.0, 0.1, -0.3, 0.0],
            ],
            dtype=np.float64,
        )
        weights = np.array([0.2, 0.3, 0.5], dtype=np.float64)

        batch = engine.compute(
            request=request,
            frame=(np.array([10, 10, 10], dtype=np.int64), positions, box, reference_forces),
            topology_arrays=topo,
            forcefield_snapshot=ff,
            frame_weights=weights,
            batch_size=2,
        )

        singles = []
        for sample_index in range(positions.shape[0]):
            singles.append(
                engine.compute(
                    request=request,
                    frame=(10, positions[sample_index], box, reference_forces[sample_index]),
                    topology_arrays=topo,
                    forcefield_snapshot=ff,
                    frame_weight=float(weights[sample_index]),
                )["fm_stats"]
            )

        partial = batch["fm_stats"]
        for key in ("JtJ", "Jty", "Jtf"):
            np.testing.assert_allclose(partial[key], sum(item[key] for item in singles))
        assert partial["yty"] == pytest.approx(sum(item["yty"] for item in singles))
        assert partial["ftf"] == pytest.approx(sum(item["ftf"] for item in singles))
        assert partial["fTy"] == pytest.approx(sum(item["fTy"] for item in singles))
        assert partial["weight_sum"] == pytest.approx(float(weights.sum()))
        assert partial["n_frames"] == 1

    def test_fm_reducer_consumes_reducer_ready_fm_stats(self, bond_key, harmonic_bond):
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})
        reference_forces = np.array(
            [
                [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                [0.5, 0.2, 0.0, -0.5, -0.2, 0.0],
            ],
            dtype=np.float64,
        )
        weights = np.array([0.4, 0.6], dtype=np.float64)
        stats_result = force(
            compute_frame_geometry(positions, box, topo),
            ff,
            return_fm_stats=True,
            reference_force=reference_forces,
            frame_weights=weights,
        )
        payload = {"frame_idx": 10, "fm_stats": stats_result["fm_stats_sum"]}

        step = {"step_mode": "fm"}
        state = init_step_state(step, ff, topo)
        consume_step_payload(
            step,
            state,
            payload,
            frame_weight=1.0,
            reference_force=None,
        )
        reduced = finalize_step_root(step, state)
        partial = payload["fm_stats"]
        scale = 1.0 / partial["weight_sum"]
        np.testing.assert_allclose(reduced["JtJ"], partial["JtJ"] * scale)
        np.testing.assert_allclose(reduced["Jty"], partial["Jty"] * scale)
        np.testing.assert_allclose(reduced["Jtf"], partial["Jtf"] * scale)
        assert reduced["y_sumsq"] == pytest.approx(partial["yty"] * scale)
        assert reduced["f_sumsq"] == pytest.approx(partial["ftf"] * scale)
        assert reduced["fty"] == pytest.approx(partial["fTy"] * scale)
        assert reduced["nframe"] == 1
        assert reduced["weight_sum"] == pytest.approx(float(weights.sum()))

    def test_compute_accepts_weighted_batch_force_path(self, bond_key, harmonic_bond):
        engine = build_default_engine()
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [5.5, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})
        request = _request("force", "force_grad")
        weights = np.array([0.2, 0.3, 0.5], dtype=np.float64)

        batch = engine.compute(
            request=request,
            frame=(np.array([10, 10, 10], dtype=np.int64), positions, box, None),
            topology_arrays=topo,
            forcefield_snapshot=ff,
            frame_weights=weights,
            batch_size=2,
        )

        singles = [
            engine.compute(
                request=request,
                frame=(10, coords, box, None),
                topology_arrays=topo,
                forcefield_snapshot=ff,
            )
            for coords in positions
        ]
        force_stack = np.stack([item["force"] for item in singles], axis=0)
        grad_stack = np.stack([item["force_grad"] for item in singles], axis=0)
        np.testing.assert_allclose(batch["force"], weights @ force_stack)
        np.testing.assert_allclose(
            batch["force_grad"],
            np.tensordot(weights, grad_stack, axes=(0, 0)),
        )

    def test_batched_compute_pair_cache_uses_first_sample_cutoff(
        self,
        bond_key,
        harmonic_bond,
        monkeypatch,
    ):
        engine = build_default_engine()
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [4.2, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})
        pair_key = InteractionKey.pair("A", "B")
        seen = {}

        def fake_compute_pairs_by_type(**kwargs):
            seen["cutoff"] = kwargs["cutoff"]
            seen["positions"] = np.asarray(kwargs["positions"]).copy()
            empty = np.empty(0, dtype=np.int32)
            return {pair_key: (empty, empty)}

        monkeypatch.setattr(
            mpi_engine_module,
            "compute_pairs_by_type",
            fake_compute_pairs_by_type,
        )

        engine.compute(
            request=_request("energy_grad"),
            frame=(np.array([10, 10], dtype=np.int64), positions, box, None),
            topology_arrays=topo,
            forcefield_snapshot=ff,
            pair_type_list=[pair_key],
            pair_cutoff=6.0,
        )

        assert seen["cutoff"] == pytest.approx(6.0)
        np.testing.assert_allclose(seen["positions"], positions[0])

    def test_cdfm_zbx_reducer_consumes_batched_compute_stats(self, bond_key, harmonic_bond):
        engine = build_default_engine()
        positions = np.array(
            [
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [5.5, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})
        request = _request("force", "force_grad", "energy_grad")
        weights = np.array([0.2, 0.3, 0.5], dtype=np.float64)
        step = {
            "step_mode": "cdfm_zbx",
            "y_eff": np.zeros(6, dtype=np.float64),
            "mode": "reinforce",
            "beta": 0.7,
        }

        batched_payload = engine.compute(
            request=request,
            frame=(np.array([10, 10, 10], dtype=np.int64), positions, box, None),
            topology_arrays=topo,
            forcefield_snapshot=ff,
            frame_weights=weights,
            batch_size=2,
        )
        batch_state = init_step_state(step, ff, topo)
        consume_step_payload(
            step,
            batch_state,
            batched_payload,
            frame_weight=1.0,
            reference_force=None,
        )
        batch_result = finalize_step_root(step, batch_state)

        single_state = init_step_state(step, ff, topo)
        for coords, sample_weight in zip(positions, weights):
            single_payload = engine.compute(
                request=request,
                frame=(10, coords, box, None),
                topology_arrays=topo,
                forcefield_snapshot=ff,
            )
            consume_step_payload(
                step,
                single_state,
                single_payload,
                frame_weight=float(sample_weight),
                reference_force=None,
            )
        single_result = finalize_step_root(step, single_state)

        np.testing.assert_allclose(batch_result["grad_direct"], single_result["grad_direct"])
        np.testing.assert_allclose(batch_result["grad_reinforce"], single_result["grad_reinforce"])
        assert batch_result["sse"] == pytest.approx(single_result["sse"])
        assert batch_result["n_samples"] == 1
        assert single_result["n_samples"] == 3

    def test_batched_compute_rejects_frame_cache_requests(self, bond_key, harmonic_bond):
        engine = build_default_engine()
        positions = np.array([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]], dtype=np.float64)
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        ff = Forcefield({bond_key: [harmonic_bond]})

        with pytest.raises(ValueError, match="frame-cache"):
            engine.compute(
                request=_request("frame_cache"),
                frame=(np.array([0], dtype=np.int64), positions, box, None),
                topology_arrays=topo,
                forcefield_snapshot=ff,
            )

    def test_add_noise_builds_normalized_coordinate_batch(self, bond_key):
        engine = build_default_engine()
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)

        frame_batch, weights = engine.add_noise(
            (7, positions, box, None),
            {
                "samples_per_frame": 2,
                "sigma": 0.0,
                "seed": 11,
                "include_original": True,
            },
            topo,
        )

        frame_ids, positions_batch, box_batch, reference_batch = frame_batch
        assert reference_batch is None
        np.testing.assert_array_equal(frame_ids, [7, 7, 7])
        np.testing.assert_allclose(positions_batch, np.broadcast_to(positions, (3, 2, 3)))
        np.testing.assert_allclose(box_batch, np.broadcast_to(box, (3, 6)))
        np.testing.assert_allclose(weights, np.full(3, 1.0 / 3.0))

    def test_add_noise_broadcasts_reference_forces_unchanged(self, bond_key):
        engine = build_default_engine()
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        reference_forces = np.arange(6, dtype=np.float64)
        sigma = 0.2

        frame_batch, _ = engine.add_noise(
            (7, positions, box, reference_forces),
            {
                "samples_per_frame": 2,
                "sigma": sigma,
                "seed": 11,
                "include_original": True,
            },
            topo,
        )

        _, positions_batch, _, reference_batch = frame_batch
        assert not np.allclose(
            positions_batch,
            np.broadcast_to(positions, positions_batch.shape),
        )
        np.testing.assert_allclose(
            reference_batch,
            np.broadcast_to(reference_forces, (3, 6)),
        )

    def test_add_noise_does_not_require_beta_for_reference_forces(self, bond_key):
        engine = build_default_engine()
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)

        frame_batch, _ = engine.add_noise(
            (7, positions, box, np.zeros(6, dtype=np.float64)),
            {"samples_per_frame": 1, "sigma": 0.2},
            topo,
        )

        assert frame_batch[3].shape == (1, 6)

    def test_reduce_step_partials_non_root_handles_max_keys(self, monkeypatch):
        class _FakeComm:
            def __init__(self):
                self.reductions = []

            def Get_rank(self):
                return 1

            def Get_size(self):
                return 2

            def reduce(self, value, op=None, root=0):
                self.reductions.append((op, root, np.asarray(value).shape))
                return None

            def gather(self, value, root=0):
                raise AssertionError("gather should not be used for fm reduction")

        fake_mpi4py = SimpleNamespace(
            MPI=SimpleNamespace(MAX="MAX", SUM="SUM"),
        )
        monkeypatch.setitem(sys.modules, "mpi4py", fake_mpi4py)
        comm = _FakeComm()
        engine = MPIComputeEngine(comm=comm)

        reduced = engine._reduce_step_partials(
            {"step_mode": "fm"},
            {
                "JtJ_sum": np.zeros((2, 2), dtype=np.float32),
                "Jty_sum": np.zeros(2, dtype=np.float32),
                "y_sumsq_sum": 0.0,
                "Jtf_sum": np.zeros(2, dtype=np.float32),
                "f_sumsq_sum": 0.0,
                "fty_sum": 0.0,
                "nframe": 0,
                "weight_sum": 0.0,
                "n_atoms_obs": 0,
            },
        )

        assert reduced is None
        assert comm.reductions[-1][0] == "MAX"
        assert len(comm.reductions) == 9


class TestIntegration:
    def test_bond_energy_force_roundtrip(self, bond_key):
        positions = np.array([[0, 0, 0], [3.0, 0, 0]], dtype=np.float64)
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        pot = HarmonicPotential("A", "B", k=5.0, r0=4.0)
        ff = Forcefield({bond_key: [pot]})
        geom = compute_frame_geometry(positions, box, topo)
        e_result = energy(geom, ff, return_value=True, return_grad=True)
        f_result = force(geom, ff, return_grad=True, return_hessian=True)
        assert abs(e_result["energy"] - 5.0) < 1e-12
        np.testing.assert_allclose(f_result["force_grad"][0, :], [-2.0, -10.0], atol=1e-10)
        np.testing.assert_allclose(f_result["force_grad"][3, :], [2.0, 10.0], atol=1e-10)
        assert np.all(np.linalg.eigvalsh(f_result["force_hessian"]) >= -1e-10)

    def test_run_post_multi_builds_geometry_once_per_frame(self, tmp_path, bond_key, harmonic_bond, monkeypatch):
        ff = Forcefield({bond_key: [harmonic_bond]})
        ff_path = tmp_path / "ff.pkl"
        with ff_path.open("wb") as fh:
            pickle.dump(ff, fh, protocol=pickle.HIGHEST_PROTOCOL)

        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        positions = np.array([[0, 0, 0], [3.0, 0, 0]], dtype=np.float64)
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        sel_indices = np.array([0, 1], dtype=np.int64)

        class _DummyTrajectory:
            def __len__(self):
                return 1

        class _DummyUniverse:
            def __init__(self):
                self.trajectory = _DummyTrajectory()

            def select_atoms(self, sel):
                assert sel == "all"
                return SimpleNamespace(indices=sel_indices)

        import MDAnalysis as mda
        import AceCG.io.trajectory as trajectory_module

        monkeypatch.setattr(mda, "Universe", lambda *args, **kwargs: _DummyUniverse())
        monkeypatch.setattr(
            trajectory_module,
            "iter_frames",
            lambda universe, *, start=0, end=None, every=1, include_forces=False: iter(
                [(0, positions, box, None)]
            ),
        )
        monkeypatch.setattr(
            "AceCG.topology.topology_array.collect_topology_arrays",
            lambda *args, **kwargs: topo,
        )

        call_count = {"n": 0}
        orig = mpi_engine_module.compute_frame_geometry

        def counted(*args, **kwargs):
            call_count["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(mpi_engine_module, "compute_frame_geometry", counted)

        topo_path = tmp_path / "topology.data"
        traj_path = tmp_path / "trajectory.lammpstrj"
        topo_path.write_text("", encoding="utf-8")
        traj_path.write_text("", encoding="utf-8")

        spec = {
            "post_mode": "one_pass",
            "forcefield_path": str(ff_path),
            "topology": str(topo_path),
            "trajectory": str(traj_path),
            "trajectory_format": "LAMMPSDUMP",
            "work_dir": str(tmp_path),
            "cutoff": 10.0,
            "steps": [
                {"step_mode": "rem", "output_file": "rem_1.pkl"},
                {"step_mode": "rem", "output_file": "rem_2.pkl"},
            ],
        }

        build_default_engine().run_post(spec)
        assert (tmp_path / "rem_1.pkl").exists()
        assert (tmp_path / "rem_2.pkl").exists()
        assert call_count["n"] == 1

    def test_run_post_noisy_rem_zero_sigma_matches_base_frame(self, tmp_path, bond_key, harmonic_bond, monkeypatch):
        ff = Forcefield({bond_key: [harmonic_bond]})
        ff_path = tmp_path / "ff.pkl"
        with ff_path.open("wb") as fh:
            pickle.dump(ff, fh, protocol=pickle.HIGHEST_PROTOCOL)

        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        positions = np.array([[0, 0, 0], [3.0, 0, 0]], dtype=np.float64)
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        sel_indices = np.array([0, 1], dtype=np.int64)

        class _DummyTrajectory:
            def __len__(self):
                return 1

        class _DummyUniverse:
            def __init__(self):
                self.trajectory = _DummyTrajectory()

            def select_atoms(self, sel):
                assert sel == "all"
                return SimpleNamespace(indices=sel_indices)

        import MDAnalysis as mda
        import AceCG.io.trajectory as trajectory_module

        monkeypatch.setattr(mda, "Universe", lambda *args, **kwargs: _DummyUniverse())
        monkeypatch.setattr(
            trajectory_module,
            "iter_frames",
            lambda universe, *, start=0, end=None, every=1, include_forces=False: iter(
                [(0, positions, box, None)]
            ),
        )
        monkeypatch.setattr(
            "AceCG.topology.topology_array.collect_topology_arrays",
            lambda *args, **kwargs: topo,
        )

        topo_path = tmp_path / "topology.data"
        traj_path = tmp_path / "trajectory.lammpstrj"
        topo_path.write_text("", encoding="utf-8")
        traj_path.write_text("", encoding="utf-8")

        spec = {
            "post_mode": "one_pass",
            "forcefield_path": str(ff_path),
            "topology": str(topo_path),
            "trajectory": str(traj_path),
            "trajectory_format": "LAMMPSDUMP",
            "work_dir": str(tmp_path),
            "cutoff": 10.0,
            "noise": {
                "enabled": True,
                "samples_per_frame": 3,
                "sigma": 0.0,
                "seed": 123,
                "include_original": True,
                "batch_size": 2,
            },
            "steps": [
                {"step_mode": "rem", "need_hessian": True, "output_file": "rem_noisy.pkl"},
            ],
        }

        build_default_engine().run_post(spec)
        with (tmp_path / "rem_noisy.pkl").open("rb") as handle:
            payload = pickle.load(handle)

        expected = build_default_engine().compute(
            request=_request("energy_grad", "energy_hessian", "energy_grad_outer"),
            frame=(0, positions, box, None),
            topology_arrays=topo,
            forcefield_snapshot=ff,
            pair_type_list=[],
            pair_cutoff=10.0,
        )
        np.testing.assert_allclose(payload["energy_grad_avg"], expected["energy_grad"])
        np.testing.assert_allclose(payload["d2U_avg"], expected["energy_hessian"])
        np.testing.assert_allclose(payload["grad_outer_avg"], expected["energy_grad_outer"])
        assert payload["n_frames"] == 1
        assert payload["weight_sum"] == pytest.approx(1.0)

    def test_run_post_noisy_fm_zero_sigma_matches_base_frame(self, tmp_path, bond_key, harmonic_bond, monkeypatch):
        ff = Forcefield({bond_key: [harmonic_bond]})
        ff_path = tmp_path / "ff.pkl"
        with ff_path.open("wb") as fh:
            pickle.dump(ff, fh, protocol=pickle.HIGHEST_PROTOCOL)

        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        positions = np.array([[0, 0, 0], [3.0, 0, 0]], dtype=np.float64)
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        reference_forces = np.array([0.25, 0.0, 0.0, -0.25, 0.0, 0.0], dtype=np.float64)
        sel_indices = np.array([0, 1], dtype=np.int64)

        class _DummyTrajectory:
            def __len__(self):
                return 1

        class _DummyUniverse:
            def __init__(self):
                self.trajectory = _DummyTrajectory()

            def select_atoms(self, sel):
                assert sel == "all"
                return SimpleNamespace(indices=sel_indices)

        import MDAnalysis as mda
        import AceCG.io.trajectory as trajectory_module

        monkeypatch.setattr(mda, "Universe", lambda *args, **kwargs: _DummyUniverse())
        monkeypatch.setattr(
            trajectory_module,
            "iter_frames",
            lambda universe, *, start=0, end=None, every=1, include_forces=False: iter(
                [(0, positions, box, reference_forces)]
            ),
        )
        monkeypatch.setattr(
            "AceCG.topology.topology_array.collect_topology_arrays",
            lambda *args, **kwargs: topo,
        )

        topo_path = tmp_path / "topology.data"
        traj_path = tmp_path / "trajectory.lammpstrj"
        topo_path.write_text("", encoding="utf-8")
        traj_path.write_text("", encoding="utf-8")

        spec = {
            "post_mode": "one_pass",
            "forcefield_path": str(ff_path),
            "topology": str(topo_path),
            "trajectory": str(traj_path),
            "trajectory_format": "LAMMPSDUMP",
            "work_dir": str(tmp_path),
            "cutoff": 10.0,
            "noise": {
                "enabled": True,
                "samples_per_frame": 3,
                "sigma": 0.0,
                "seed": 123,
                "include_original": True,
                "batch_size": 2,
            },
            "steps": [
                {"step_mode": "fm", "output_file": "fm_noisy.pkl"},
            ],
        }

        build_default_engine().run_post(spec)
        with (tmp_path / "fm_noisy.pkl").open("rb") as handle:
            payload = pickle.load(handle)

        expected = build_default_engine().compute(
            request=_request("fm_stats", "reference_force"),
            frame=(0, positions, box, reference_forces),
            topology_arrays=topo,
            forcefield_snapshot=ff,
            pair_type_list=[],
            pair_cutoff=10.0,
        )["fm_stats"]
        np.testing.assert_allclose(payload["JtJ"], expected["JtJ"])
        np.testing.assert_allclose(payload["Jty"], expected["Jty"])
        np.testing.assert_allclose(payload["Jtf"], expected["Jtf"])
        assert payload["y_sumsq"] == pytest.approx(expected["yty"])
        assert payload["f_sumsq"] == pytest.approx(expected["ftf"])
        assert payload["fty"] == pytest.approx(expected["fTy"])
        assert payload["nframe"] == 1
        assert payload["weight_sum"] == pytest.approx(1.0)

    def test_run_post_cache_step_writes_trajectory_cache(self, tmp_path, bond_key, harmonic_bond, monkeypatch):
        ff = Forcefield({bond_key: [harmonic_bond]})
        ff_path = tmp_path / "ff.pkl"
        with ff_path.open("wb") as fh:
            pickle.dump(ff, fh, protocol=pickle.HIGHEST_PROTOCOL)

        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        positions = np.array([[0, 0, 0], [3.0, 0, 0]], dtype=np.float64)
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        sel_indices = np.array([0, 1], dtype=np.int64)

        class _DummyTrajectory:
            def __len__(self):
                return 1

        class _DummyUniverse:
            def __init__(self):
                self.trajectory = _DummyTrajectory()

            def select_atoms(self, sel):
                assert sel == "all"
                return SimpleNamespace(indices=sel_indices)

        import MDAnalysis as mda
        import AceCG.io.trajectory as trajectory_module

        monkeypatch.setattr(mda, "Universe", lambda *args, **kwargs: _DummyUniverse())
        monkeypatch.setattr(
            trajectory_module,
            "iter_frames",
            lambda universe, *, start=0, end=None, every=1, include_forces=False: iter(
                [(0, positions, box, None)]
            ),
        )
        monkeypatch.setattr(
            "AceCG.topology.topology_array.collect_topology_arrays",
            lambda *args, **kwargs: topo,
        )

        topo_path = tmp_path / "topology.data"
        traj_path = tmp_path / "trajectory.lammpstrj"
        topo_path.write_text("", encoding="utf-8")
        traj_path.write_text("", encoding="utf-8")

        spec = {
            "post_mode": "one_pass",
            "forcefield_path": str(ff_path),
            "topology": str(topo_path),
            "trajectory": str(traj_path),
            "trajectory_format": "LAMMPSDUMP",
            "work_dir": str(tmp_path),
            "cutoff": 10.0,
            "steps": [
                {"step_mode": "cache", "output_file": "cache.pkl"},
            ],
        }

        build_default_engine().run_post(spec)

        with (tmp_path / "cache.pkl").open("rb") as handle:
            cache = pickle.load(handle)
        assert isinstance(cache, TrajectoryCache)
        assert sorted(cache.frames) == [0]
        np.testing.assert_allclose(cache.frames[0].bond_distances[bond_key], [3.0])

    @pytest.mark.skip(reason="run_post no longer auto-backs up existing output files (Ace merge; see MERGE_REPORT.md).")
    def test_run_post_backs_up_existing_output_file(self, tmp_path, bond_key, harmonic_bond, monkeypatch):
        ff = Forcefield({bond_key: [harmonic_bond]})
        ff_path = tmp_path / "ff.pkl"
        with ff_path.open("wb") as fh:
            pickle.dump(ff, fh, protocol=pickle.HIGHEST_PROTOCOL)

        topo = _bond_topo(
            bond_key,
            bonds=np.array([[0, 1]], dtype=np.int64),
            bond_key_index=np.array([0], dtype=np.int32),
        )
        positions = np.array([[0, 0, 0], [3.0, 0, 0]], dtype=np.float64)
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
        sel_indices = np.array([0, 1], dtype=np.int64)

        class _DummyTrajectory:
            def __len__(self):
                return 1

        class _DummyUniverse:
            def __init__(self):
                self.trajectory = _DummyTrajectory()

            def select_atoms(self, sel):
                assert sel == "all"
                return SimpleNamespace(indices=sel_indices)

        import MDAnalysis as mda
        import AceCG.io.trajectory as trajectory_module

        monkeypatch.setattr(mda, "Universe", lambda *args, **kwargs: _DummyUniverse())
        monkeypatch.setattr(
            trajectory_module,
            "iter_frames",
            lambda universe, *, start=0, end=None, every=1, include_forces=False: iter(
                [(0, positions, box, None)]
            ),
        )
        monkeypatch.setattr(
            "AceCG.topology.topology_array.collect_topology_arrays",
            lambda *args, **kwargs: topo,
        )

        topo_path = tmp_path / "topology.data"
        traj_path = tmp_path / "trajectory.lammpstrj"
        topo_path.write_text("", encoding="utf-8")
        traj_path.write_text("", encoding="utf-8")

        output_path = tmp_path / "rem.pkl"
        with output_path.open("wb") as fh:
            pickle.dump({"old": True}, fh, protocol=pickle.HIGHEST_PROTOCOL)

        spec = {
            "post_mode": "one_pass",
            "forcefield_path": str(ff_path),
            "topology": str(topo_path),
            "trajectory": str(traj_path),
            "trajectory_format": "LAMMPSDUMP",
            "work_dir": str(tmp_path),
            "cutoff": 10.0,
            "steps": [
                {"step_mode": "rem", "output_file": output_path.name},
            ],
        }

        build_default_engine().run_post(spec)

        backup_path = tmp_path / "rem.pkl.bak"
        assert output_path.exists()
        assert backup_path.exists()
        with backup_path.open("rb") as fh:
            assert pickle.load(fh) == {"old": True}
        with output_path.open("rb") as fh:
            payload = pickle.load(fh)
        assert "energy_grad_avg" in payload
