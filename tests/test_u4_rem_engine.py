"""Tests for U4 — REM/CDREM engine wiring.

Covers:
- NB-1: energy_hessian (true Hessian) vs energy_grad_outer (Fisher/GGN)
- NB-2: collect_topology_arrays key index tables
- one-pass-compatible REM statistics accumulation
- REM request wiring for Hessian/outer-gradient observables
- REMTrainerAnalytic engine_stats batch mode
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import AceCG.compute.mpi_engine as mpi_engine_module
from AceCG.compute.energy import energy
from AceCG.compute.frame_geometry import FrameGeometry
from AceCG.compute.mpi_engine import MPIComputeEngine, build_default_engine
from AceCG.compute.reducers import step_request
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey


def _request(*names):
    return frozenset(names)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def bond_key():
    return InteractionKey.bond("A", "B")


@pytest.fixture()
def harmonic_bond(bond_key):
    return HarmonicPotential("A", "B", k=10.0, r0=5.0)


@pytest.fixture()
def simple_forcefield(bond_key, harmonic_bond):
    return Forcefield({bond_key: [harmonic_bond]})


@pytest.fixture()
def simple_geom(bond_key):
    """FrameGeometry with 4 atoms, 2 bonds (r=4 each)."""
    positions = np.array([
        [0, 0, 0], [4, 0, 0], [8, 0, 0], [12, 0, 0],
    ], dtype=np.float64)
    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
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


# ---------------------------------------------------------------------------
# NB-1: energy_hessian and energy_grad_outer semantics
# ---------------------------------------------------------------------------

class TestEnergyHessianNB1:
    """Verify true Hessian vs gradient outer product separation."""

    def test_energy_hessian_is_true_d2u(self, simple_geom, simple_forcefield):
        """energy_hessian uses pot.d2param_names() for true d²U/dθ²."""
        result = energy(simple_geom, simple_forcefield, return_hessian=True)
        H = result["energy_hessian"]
        assert H.shape == (2, 2)
        # For HarmonicPotential U = k*(r-r0)^2:
        #   d²U/dk² = 0  (linear in k)
        #   d²U/dkdr0 = -2*(r-r0) summed over bonds
        #   d²U/dr0² = 2*k summed over bonds
        # With r=4, r0=5, k=10, 2 bonds:
        # d2U_dk_dk = 0
        # d2U_dk_dr0 = -2*(4-5)*2 = 4
        # d2U_dr0_dr0 = 2*10*2 = 40
        assert abs(H[0, 0] - 0.0) < 1e-12, f"d²U/dk² = {H[0,0]}, expected 0"
        assert abs(H[0, 1] - 4.0) < 1e-10, f"d²U/dkdr0 = {H[0,1]}, expected 4"
        assert abs(H[1, 0] - 4.0) < 1e-10, "Hessian should be symmetric"
        assert abs(H[1, 1] - 40.0) < 1e-10, f"d²U/dr0² = {H[1,1]}, expected 40"

    def test_energy_grad_outer_is_fisher(self, simple_geom, simple_forcefield):
        """energy_grad_outer computes (dU/dθ)(dU/dθ)^T — full gradient outer product."""
        result = energy(simple_geom, simple_forcefield, return_grad_outer=True)
        G = result["energy_grad_outer"]
        assert G.shape == (2, 2)
        # For HarmonicPotential:
        #   dU/dk per sample = (r-r0)^2
        #   dU/dr0 per sample = -2*k*(r-r0)
        # With r=4, r0=5, k=10:
        #   dU/dk per bond = (4-5)^2 = 1
        #   dU/dr0 per bond = -2*10*(4-5) = 20
        # Total frame gradient: [1+1, 20+20] = [2, 40]
        # Full outer product: outer([2, 40], [2, 40]) = [[4,80],[80,1600]]
        assert abs(G[0, 0] - 4.0) < 1e-10
        assert abs(G[0, 1] - 80.0) < 1e-10
        assert abs(G[1, 0] - 80.0) < 1e-10
        assert abs(G[1, 1] - 1600.0) < 1e-10

    def test_hessian_and_grad_outer_differ(self, simple_geom, simple_forcefield):
        """True Hessian and gradient outer product are distinct quantities."""
        result = energy(
            simple_geom, simple_forcefield,
            return_hessian=True, return_grad_outer=True,
        )
        H = result["energy_hessian"]
        G = result["energy_grad_outer"]
        # They should NOT be equal in general
        assert not np.allclose(H, G), "Hessian and grad_outer should differ"

    def test_shared_grad_computation(self, simple_geom, simple_forcefield):
        """Requesting both return_grad and return_grad_outer should work."""
        result = energy(
            simple_geom, simple_forcefield,
            return_grad=True, return_grad_outer=True,
        )
        assert "energy_grad" in result
        assert "energy_grad_outer" in result
        grad = result["energy_grad"]
        G = result["energy_grad_outer"]
        assert grad.shape == (2,)
        assert G.shape == (2, 2)


# ---------------------------------------------------------------------------
# REM request wiring: energy_grad_outer
# ---------------------------------------------------------------------------

class TestRequestU4:
    def test_energy_grad_outer_requested(self):
        assert "energy_grad_outer" in step_request({"step_mode": "cdrem", "need_hessian": True})

    def test_energy_hessian_requested(self):
        assert "energy_hessian" in step_request({"step_mode": "cdrem", "need_hessian": True})


# ---------------------------------------------------------------------------
# REM statistics accumulation (serial engine path)
# ---------------------------------------------------------------------------

class TestComputeREMEnergyStats:
    @pytest.fixture()
    def mini_engine(self, bond_key, harmonic_bond):
        """Engine with frames ready for serial compute."""
        positions = np.array([
            [0, 0, 0], [4, 0, 0], [8, 0, 0], [12, 0, 0],
        ], dtype=np.float64)
        box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
        # Topology arrays for compute_frame_geometry
        topology_arrays = SimpleNamespace(
            bonds=np.array([[0, 1], [2, 3]], dtype=np.int64),
            bond_key_index=np.array([0, 0], dtype=np.int32),
            keys_bondtypes=[bond_key],
            angles=np.empty((0, 3), dtype=np.int64),
            angle_key_index=np.empty(0, dtype=np.int32),
            keys_angletypes=[],
            dihedrals=np.empty((0, 4), dtype=np.int64),
            dihedral_key_index=np.empty(0, dtype=np.int32),
            keys_dihedraltypes=[],
        )
        forcefield = Forcefield({bond_key: [harmonic_bond]})
        frames = [
            (0, positions.copy(), box.copy()),
            (1, positions.copy(), box.copy()),
        ]
        return {
            "frames": frames,
            "topology_arrays": topology_arrays,
            "forcefield": forcefield,
        }

    def test_basic_avg(self, mini_engine):
        stats = _expected_rem_stats(
            frames=mini_engine["frames"],
            topology_arrays=mini_engine["topology_arrays"],
            forcefield_snapshot=mini_engine["forcefield"],
        )
        assert stats is not None
        assert "energy_grad_avg" in stats
        assert stats["n_frames"] == 2
        # Both frames identical → avg = single-frame value
        # dU/dk = sum over bonds of (r-r0)^2 = 2*1 = 2
        # dU/dr0 = sum over bonds of -2*k*(r-r0) = 2*20 = 40
        assert abs(stats["energy_grad_avg"][0] - 2.0) < 1e-10
        assert abs(stats["energy_grad_avg"][1] - 40.0) < 1e-10

    def test_with_hessian(self, mini_engine):
        stats = _expected_rem_stats(
            frames=mini_engine["frames"],
            topology_arrays=mini_engine["topology_arrays"],
            forcefield_snapshot=mini_engine["forcefield"],
            need_hessian=True,
        )
        assert "d2U_avg" in stats
        assert "grad_outer_avg" in stats
        assert "energy_grad_frame" in stats
        # d2U_avg should be true Hessian / n_frames
        d2U = stats["d2U_avg"]
        assert d2U.shape == (2, 2)
        # Per frame: d2U/dr0^2 = 2*k*n_bonds = 2*10*2 = 40
        # Average: 40
        assert abs(d2U[1, 1] - 40.0) < 1e-10

    def test_empty_frames(self, mini_engine):
        stats = _expected_rem_stats(
            frames=[],
            topology_arrays=mini_engine["topology_arrays"],
            forcefield_snapshot=mini_engine["forcefield"],
        )
        assert stats["n_frames"] == 0
        assert np.all(stats["energy_grad_avg"] == 0)

    def test_engine_recomputes_geometry_without_frame_cache(self, mini_engine, monkeypatch):
        engine = build_default_engine()
        call_count = {"n": 0}
        orig = mpi_engine_module.compute_frame_geometry

        def counted(*args, **kwargs):
            call_count["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(mpi_engine_module, "compute_frame_geometry", counted)

        stats1 = _expected_rem_stats(
            engine=engine,
            frames=mini_engine["frames"],
            topology_arrays=mini_engine["topology_arrays"],
            forcefield_snapshot=mini_engine["forcefield"],
        )
        stats2 = _expected_rem_stats(
            engine=engine,
            frames=mini_engine["frames"],
            topology_arrays=mini_engine["topology_arrays"],
            forcefield_snapshot=mini_engine["forcefield"],
        )

        assert call_count["n"] == 2 * len(mini_engine["frames"])
        np.testing.assert_allclose(stats1["energy_grad_avg"], stats2["energy_grad_avg"])


def _expected_rem_stats(
    *,
    frames,
    topology_arrays,
    forcefield_snapshot,
    engine=None,
    need_hessian: bool = False,
    frame_weight=None,
):
    if engine is None:
        engine = build_default_engine()
    n_params = forcefield_snapshot.n_params()
    n_frames = len(frames)
    if frame_weight is None:
        weights = np.ones(n_frames, dtype=np.float64)
    else:
        weights = np.asarray(frame_weight, dtype=np.float64).reshape(-1)
    weight_sum = float(weights.sum())
    grad_rows = []
    hessian_rows = []
    outer_rows = []
    request_names = ["energy_grad"]
    if need_hessian:
        request_names.extend(["energy_hessian", "energy_grad_outer"])
    request = _request(*request_names)
    for frame_id, positions, box in frames:
        payload = engine.compute(
            request=request,
            frame=(frame_id, positions, box, None),
            topology_arrays=topology_arrays,
            forcefield_snapshot=forcefield_snapshot,
        )
        grad_rows.append(np.asarray(payload["energy_grad"], dtype=np.float64))
        if need_hessian:
            hessian_rows.append(np.asarray(payload["energy_hessian"], dtype=np.float64))
            outer_rows.append(np.asarray(payload["energy_grad_outer"], dtype=np.float64))
    if n_frames:
        grad_stack = np.asarray(grad_rows, dtype=np.float64).reshape(n_frames, n_params)
        grad_sum = np.tensordot(weights, grad_stack, axes=1)
    else:
        grad_stack = np.empty((0, n_params), dtype=np.float64)
        grad_sum = np.zeros(n_params, dtype=np.float64)
    out = {
        "energy_grad_avg": grad_sum / weight_sum if weight_sum > 0.0 else np.zeros(n_params, dtype=np.float64),
        "n_frames": n_frames,
        "weight_sum": weight_sum,
    }
    if need_hessian:
        if n_frames:
            hessian_stack = np.asarray(hessian_rows, dtype=np.float64)
            outer_stack = np.asarray(outer_rows, dtype=np.float64)
            d2_sum = np.tensordot(weights, hessian_stack, axes=1)
            outer_sum = np.tensordot(weights, outer_stack, axes=1)
        else:
            d2_sum = np.zeros((n_params, n_params), dtype=np.float64)
            outer_sum = np.zeros((n_params, n_params), dtype=np.float64)
        scale = 1.0 / weight_sum if weight_sum > 0.0 else 0.0
        out["d2U_avg"] = d2_sum * scale
        out["grad_outer_avg"] = outer_sum * scale
        out["energy_grad_frame"] = grad_stack
    return out


# ---------------------------------------------------------------------------
# REMTrainerAnalytic engine_stats batch mode
# ---------------------------------------------------------------------------

class TestTrainerEngineStats:
    @pytest.fixture()
    def trainer_kit(self):
        """Minimal REMTrainer with a trivial forcefield."""
        from AceCG.trainers.analytic.rem import REMTrainerAnalytic
        from AceCG.optimizers.adam import AdamMaskedOptimizer
        from AceCG.topology.forcefield import Forcefield

        pot = HarmonicPotential("A", "B", k=10.0, r0=5.0)
        ff = {("A", "B"): pot}
        L = Forcefield(ff).param_array()
        mask = np.ones(len(L), dtype=bool)
        optimizer = AdamMaskedOptimizer(L=L, mask=mask, lr=0.01)
        trainer = REMTrainerAnalytic(
            forcefield=ff,
            optimizer=optimizer,
            beta=1.0,
        )
        return trainer

    def test_make_batch_from_stats(self, trainer_kit):
        """make_batch produces the current statistics-only REM batch."""
        batch = trainer_kit.make_batch(
            step_index=5,
            energy_grad_CG=np.array([1.0, 2.0]),
            energy_grad_AA=np.array([3.0, 4.0]),
        )
        assert batch["step_index"] == 5
        assert np.array_equal(batch["energy_grad_AA"], [3.0, 4.0])
        assert np.array_equal(batch["energy_grad_CG"], [1.0, 2.0])

    def test_step_with_precomputed_stats(self, trainer_kit):
        """Trainer.step() accepts the current precomputed-statistics batch."""
        energy_grad_AA = np.array([2.0, 40.0])
        energy_grad_CG = np.array([1.8, 38.0])
        batch = trainer_kit.make_batch(
            energy_grad_CG=energy_grad_CG,
            energy_grad_AA=energy_grad_AA,
        )
        result = trainer_kit.step(batch, apply_update=False)
        # grad = beta * (energy_grad_AA - energy_grad_CG)
        expected_grad = 1.0 * (energy_grad_AA - energy_grad_CG)
        assert np.allclose(result["grad"], expected_grad)
        assert result["name"] == "REM"
