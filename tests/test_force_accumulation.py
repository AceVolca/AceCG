"""Tests for force accumulation with real-sized duplicate-index topologies."""
from __future__ import annotations

import numpy as np
import sys
from types import SimpleNamespace

from AceCG.compute.force import _valid_flat_terms, force
from AceCG.compute.frame_geometry import compute_frame_geometry
from AceCG.compute.mpi_engine import MPIComputeEngine
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey
from AceCG.workflows.cdfm import CDFMWorkflow


# ---------------------------------------------------------------------------
# Full force-value accumulation with duplicate-index topology
# ---------------------------------------------------------------------------


def test_valid_flat_terms_masks_skin_region_pairs():
    sample_ids, term_ids = _valid_flat_terms(
        np.array([[0.5, 1.5, 2.5], [0.0, 1.0, 3.0]], dtype=np.float32),
        cutoff=2.0,
    )

    np.testing.assert_array_equal(sample_ids, np.array([0, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(term_ids, np.array([0, 1, 1], dtype=np.int64))


def test_frozen_prior_force_uses_key_mask():
    bond_key = InteractionKey.bond("C", "C")
    topo = SimpleNamespace(
        bonds=np.array([[0, 1]], dtype=np.int64),
        bond_key_index=np.array([0], dtype=np.int32),
        keys_bondtypes=[bond_key],
        angles=None,
        angle_key_index=None,
        keys_angletypes=None,
        dihedrals=None,
        dihedral_key_index=None,
        keys_dihedraltypes=None,
    )
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    box = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0], dtype=float)
    pot = HarmonicPotential("C", "C", k=5.0, r0=1.0)
    ff = Forcefield({bond_key: [pot]})
    ff.param_mask = np.array([False, False], dtype=bool)
    assert ff.key_mask[bond_key] is True

    geom = compute_frame_geometry(
        positions,
        box,
        topo,
        interaction_mask=ff.key_mask,
    )
    result = force(geom, ff, return_value=True, return_grad=True)

    assert not np.allclose(result["force"], 0.0)
    np.testing.assert_allclose(result["force_grad"], 0.0)


def test_fm_stats_skip_frozen_prior_jacobian_but_keep_force_value():
    bond_key = InteractionKey.bond("C", "C")
    topo = SimpleNamespace(
        bonds=np.array([[0, 1]], dtype=np.int64),
        bond_key_index=np.array([0], dtype=np.int32),
        keys_bondtypes=[bond_key],
        angles=None,
        angle_key_index=None,
        keys_angletypes=None,
        dihedrals=None,
        dihedral_key_index=None,
        keys_dihedraltypes=None,
    )
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    box = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0], dtype=float)

    class FrozenPrior(HarmonicPotential):
        def force_grad(self, r):
            raise AssertionError("frozen prior force_grad should not be evaluated")

    trainable = HarmonicPotential("C", "C", k=5.0, r0=1.0)
    frozen = FrozenPrior("C", "C", k=7.0, r0=1.5)
    frozen.param_mask = np.zeros(frozen.n_params(), dtype=bool)
    ff = Forcefield({bond_key: [trainable, frozen]})
    assert ff.key_mask[bond_key] is True

    geom = compute_frame_geometry(
        positions,
        box,
        topo,
        interaction_mask=ff.key_mask,
    )
    result = force(
        geom,
        ff,
        return_fm_stats=True,
        reference_force=np.zeros((2, 3), dtype=float),
    )
    stats = result["fm_stats"]

    assert stats["JtJ"].shape == (4, 4)
    assert stats["Jtf"].shape == (4,)
    np.testing.assert_allclose(stats["JtJ"][2:, :], 0.0)
    np.testing.assert_allclose(stats["JtJ"][:, 2:], 0.0)
    np.testing.assert_allclose(stats["Jtf"][2:], 0.0)
    assert stats["ftf"] > 0.0


def test_cdfm_mask_install_sets_training_key_mask():
    real_key = InteractionKey.bond("C", "C")
    vp_key = InteractionKey.bond("C", "VP")
    ff = Forcefield(
        {
            real_key: [HarmonicPotential("C", "C", k=5.0, r0=1.0)],
            vp_key: [HarmonicPotential("C", "VP", k=5.0, r0=1.0)],
        }
    )
    ff.set_vp_masks(["VP"])

    workflow = SimpleNamespace(
        forcefield=ff,
        config=SimpleNamespace(conditioning=SimpleNamespace(mask_cg_only=True)),
    )

    CDFMWorkflow._install_cdfm_mask(workflow)

    assert ff.key_mask[real_key] is False
    assert ff.key_mask[vp_key] is True
    assert not np.any(ff.param_mask[ff.real_mask])


def test_cdfm_baseline_preprocess_temporarily_uses_real_key_mask(tmp_path, monkeypatch):
    real_key = InteractionKey.bond("C", "C")
    vp_key = InteractionKey.bond("C", "VP")
    ff = Forcefield(
        {
            real_key: [HarmonicPotential("C", "C", k=5.0, r0=1.0)],
            vp_key: [HarmonicPotential("C", "VP", k=5.0, r0=1.0)],
        }
    )
    ff.set_vp_masks(["VP"])
    train_mask = ~ff.real_mask
    ff.build_mask(init_mask=train_mask)
    ff.key_mask = ff.derive_l1_mask(train_mask)
    original_param_mask = ff.param_mask.copy()
    original_key_mask = dict(ff.key_mask)

    force_path = tmp_path / "frame_000001.forces.npy"
    np.save(force_path, np.zeros((2, 3), dtype=np.float64))

    class FakeUniverse:
        def __init__(self, *args, **kwargs):
            self.atoms = SimpleNamespace(
                positions=np.array(
                    [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64
                )
            )
            self.dimensions = np.array(
                [20.0, 20.0, 20.0, 90.0, 90.0, 90.0], dtype=np.float64
            )

    monkeypatch.setitem(sys.modules, "MDAnalysis", SimpleNamespace(Universe=FakeUniverse))

    engine = MPIComputeEngine()
    seen_masks = []

    def fake_compute(**kwargs):
        mask = dict(kwargs["interaction_mask"])
        seen_masks.append(mask)
        assert mask[real_key] is True
        assert mask[vp_key] is False
        return {"force": np.zeros(6, dtype=np.float64)}

    monkeypatch.setattr(engine, "compute", fake_compute)
    steps = [
        {
            "step_mode": "cdfm_zbx",
            "init_force_path": str(force_path),
            "init_frame_id": 1,
        }
    ]

    engine._preprocess_cdfm_zbx_steps(
        one_pass_steps=steps,
        work_dir=tmp_path,
        init_topology=str(tmp_path / "init.data"),
        rank=0,
        size=1,
        forcefield_snapshot=ff,
        topology_arrays=SimpleNamespace(real_site_indices=np.array([0, 1])),
        pair_type_list=[],
        pair_cutoff=None,
        sel_indices=None,
        exclude_option="none",
    )

    assert len(seen_masks) == 1
    np.testing.assert_array_equal(ff.param_mask, original_param_mask)
    assert ff.key_mask == original_key_mask
    np.testing.assert_allclose(steps[0]["y_eff"], np.zeros(6))


class TestForceValueDuplicateIndices:
    """Verify force-value accumulation correctness when multiple bonds share atoms."""

    @staticmethod
    def _make_chain_system(n_atoms: int):
        """Build a linear chain with n_atoms-1 bonds, all same type.

        Returns (topo, ff, positions, box).
        """
        bk = InteractionKey.bond("C", "C")
        bonds = np.array([[i, i + 1] for i in range(n_atoms - 1)], dtype=np.int64)
        topo = SimpleNamespace(
            bonds=bonds,
            bond_key_index=np.zeros(len(bonds), dtype=np.int32),
            keys_bondtypes=[bk],
            angles=None, angle_key_index=None, keys_angletypes=None,
            dihedrals=None, dihedral_key_index=None, keys_dihedraltypes=None,
        )
        pot = HarmonicPotential("C", "C", k=5.0, r0=3.5)
        ff = Forcefield({bk: [pot]})
        rng = np.random.default_rng(7)
        positions = rng.standard_normal((n_atoms, 3)) * 2.0 + np.arange(n_atoms)[:, None] * 4.0
        box = np.array([200.0, 200.0, 200.0, 90.0, 90.0, 90.0])
        return topo, ff, positions, box

    def test_small_chain(self):
        """3-atom chain: atom 1 appears in bonds (0,1) and (1,2)."""
        topo, ff, pos, box = self._make_chain_system(3)
        geom = compute_frame_geometry(pos, box, topo)
        result = force(geom, ff, return_value=True, return_grad=True)
        fvec = result["force"]
        J = result["force_grad"]

        # Verify Newton's third law: total force should be zero
        total_force = fvec.reshape(-1, 3).sum(axis=0)
        np.testing.assert_allclose(total_force, 0.0, atol=1e-10)

        # Verify J is well-formed (finite, correct shape)
        assert J.shape == (9, ff[InteractionKey.bond("C", "C")][0].n_params())
        assert np.all(np.isfinite(J))

    def test_real_size_chain(self):
        """100-atom chain — many duplicate atom indices in force accumulation."""
        topo, ff, pos, box = self._make_chain_system(100)
        geom = compute_frame_geometry(pos, box, topo)
        result = force(geom, ff, return_value=True, return_grad=True)
        fvec = result["force"]
        J = result["force_grad"]

        # Newton's third law
        total_force = fvec.reshape(-1, 3).sum(axis=0)
        np.testing.assert_allclose(total_force, 0.0, atol=2e-5)

        # Jacobian consistency: f \approx J @ theta for linear potentials
        n_params = J.shape[1]
        assert fvec.shape == (300,)
        assert J.shape == (300, n_params)
        assert np.all(np.isfinite(J))

    def test_large_chain(self):
        """1000-atom chain stress-tests duplicate-row force accumulation."""
        topo, ff, pos, box = self._make_chain_system(1000)
        geom = compute_frame_geometry(pos, box, topo)
        result = force(geom, ff, return_value=True, return_grad=True)
        fvec = result["force"]

        total_force = fvec.reshape(-1, 3).sum(axis=0)
        np.testing.assert_allclose(total_force, 0.0, atol=1e-4)
