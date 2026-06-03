from types import SimpleNamespace

import numpy as np

from AceCG.compute.energy import energy
from AceCG.compute.mpi_engine import MPIComputeEngine
from AceCG.optimizers.adam import AdamMaskedOptimizer
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey
from AceCG.trainers.analytic.rem import REMTrainerAnalytic


def _request(*names):
    return frozenset(names)


def _bond_topo(bond_key: InteractionKey) -> SimpleNamespace:
    return SimpleNamespace(
        n_atoms=2,
        bonds=np.array([[0, 1]], dtype=np.int64),
        bond_key_index=np.array([0], dtype=np.int32),
        keys_bondtypes=[bond_key],
        angles=np.empty((0, 3), dtype=np.int64),
        angle_key_index=np.empty(0, dtype=np.int32),
        keys_angletypes=[],
        dihedrals=np.empty((0, 4), dtype=np.int64),
        dihedral_key_index=np.empty(0, dtype=np.int32),
        keys_dihedraltypes=[],
        real_site_indices=None,
    )


def _forcefield() -> tuple[InteractionKey, Forcefield]:
    key = InteractionKey.bond("A", "B")
    return key, Forcefield({key: [HarmonicPotential("A", "B", k=10.0, r0=5.0)]})


def test_energy_coordinate_mask_keeps_auxiliary_unmasked_gradient():
    key, ff = _forcefield()
    geom = SimpleNamespace(
        positions=np.zeros((4, 3), dtype=np.float64),
        pair_distances={},
        bond_distances={key: np.array([4.0, 4.0], dtype=np.float64)},
        angle_values={},
        dihedral_values={},
    )

    result = energy(
        geom,
        ff,
        return_gauge_free_energy_grad=True,
        return_unmasked_gauge_free_energy_grad=True,
        coordinate_mask={key.label(): {"min": 4.5}},
    )

    np.testing.assert_allclose(result["gauge_free_energy_grad"], [0.0, 0.0])
    np.testing.assert_allclose(
        result["unmasked_gauge_free_energy_grad"],
        [2.0, 40.0],
    )
    assert result["energy_mask_diagnostics"]["active"] == 0
    assert result["energy_mask_diagnostics"]["total"] == 2


def test_mpi_compute_weighted_batch_energy_mask_diagnostics():
    key, ff = _forcefield()
    engine = MPIComputeEngine()
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    box = np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64)
    weights = np.array([0.25, 0.75], dtype=np.float64)

    result = engine.compute(
        request=_request("gauge_free_energy_grad", "unmasked_gauge_free_energy_grad"),
        frame=(np.array([0, 0], dtype=np.int64), positions, box, None),
        topology_arrays=_bond_topo(key),
        forcefield_snapshot=ff,
        frame_weights=weights,
        coordinate_mask={key.label(): {"min": 3.5}},
    )

    np.testing.assert_allclose(result["gauge_free_energy_grad"], [0.75, 15.0])
    np.testing.assert_allclose(
        result["unmasked_gauge_free_energy_grad"],
        [1.75, 25.0],
    )
    diagnostics = result["energy_mask_diagnostics"]
    assert diagnostics["active"] == 1
    assert diagnostics["total"] == 2
    assert diagnostics["by_key"][key.label()]["active"] == 1


def test_rem_trainer_hybrid_auxiliary_gradient_selection():
    _, ff = _forcefield()
    optimizer = AdamMaskedOptimizer(
        L=ff.param_array(),
        mask=np.ones(ff.n_params(), dtype=bool),
        lr=0.01,
    )
    trainer = REMTrainerAnalytic(forcefield=ff, optimizer=optimizer, beta=1.0)
    batch = REMTrainerAnalytic.make_batch(
        energy_grad_AA=np.array([2.0, 3.0], dtype=np.float64),
        energy_grad_CG=np.array([1.0, 1.0], dtype=np.float64),
        unmasked_energy_grad_AA=np.array([5.0, 6.0], dtype=np.float64),
        unmasked_energy_grad_CG=np.array([2.0, 2.0], dtype=np.float64),
        optimizer_gradient_mode="hybrid_aux",
        outside_aux_weight=0.5,
    )

    out = trainer.step(batch, apply_update=False)

    np.testing.assert_allclose(out["grad"], [1.0, 2.0])
    np.testing.assert_allclose(out["optimizer_grad"], [2.0, 3.0])
    assert out["meta"]["optimizer_gradient_mode"] == "hybrid_aux"
