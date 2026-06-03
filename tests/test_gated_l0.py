from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from AceCG.io.forcefield import WriteLmpFF
from AceCG.optimizers import AdamMaskedOptimizer
from AceCG.potentials.gaussian import GaussianPotential
from AceCG.potentials.gated import (
    GatedPotential,
    iter_gated_potentials,
    sample_L0_gates,
    set_L0_gates_deterministic,
    wrap_forcefield_with_L0_gates,
)
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey
from AceCG.trainers.analytic.l_zero import L0InteractionTrainerAnalytic


EMPTY_TOPOLOGY_ARRAYS = SimpleNamespace(
    atom_type_code_to_name={},
    bond_type_id_to_key={},
    angle_type_id_to_key={},
    dihedral_type_id_to_key={},
    key_to_bonded_type_id={},
)


def _gaussian() -> GaussianPotential:
    return GaussianPotential("A", "B", A=2.0, r0=1.0, sigma=0.5, cutoff=4.0)


def _gated_forcefield(*, log_alpha: float = 0.0) -> tuple[Forcefield, InteractionKey]:
    key = InteractionKey.pair("A", "B")
    return Forcefield({key: [GatedPotential(_gaussian(), log_alpha=log_alpha, beta=0.5)]}), key


def test_gated_potential_scales_value_force_and_gradients() -> None:
    inner = _gaussian()
    gated = GatedPotential(inner, log_alpha=0.0, beta=0.5)
    r = np.array([0.8, 1.0, 1.2], dtype=float)

    z = gated.gate_value()
    dz = gated.gate_grad_log_alpha()

    assert np.isclose(z, 0.5)
    assert np.isclose(dz, 0.3)
    np.testing.assert_allclose(gated.value(r), z * inner.value(r))
    np.testing.assert_allclose(gated.force(r), z * inner.force(r))

    grad = gated.energy_grad(r)
    np.testing.assert_allclose(grad[..., :-1], z * inner.energy_grad(r))
    np.testing.assert_allclose(grad[..., -1], dz * inner.value(r))


def test_gated_potential_carries_local_mask_and_bounds() -> None:
    inner = _gaussian()
    inner.param_mask = np.array([True, False, True], dtype=bool)
    inner.param_bounds = (
        np.array([0.0, -np.inf, 0.01], dtype=float),
        np.array([10.0, np.inf, 2.0], dtype=float),
    )

    gated = GatedPotential(inner, log_alpha=0.0, log_alpha_bounds=(-8.0, 7.0))

    np.testing.assert_array_equal(gated.param_mask, [True, False, True, True])
    lb, ub = gated.param_bounds
    np.testing.assert_allclose(lb, [0.0, -np.inf, 0.01, -8.0])
    np.testing.assert_allclose(ub, [10.0, np.inf, 2.0, 7.0])


def test_wrap_sample_and_reset_l0_gates() -> None:
    key = InteractionKey.pair("A", "B")
    forcefield = Forcefield({key: [_gaussian()]})
    wrapped = wrap_forcefield_with_L0_gates(forcefield, interaction_keys=[key])

    original_pot = forcefield[key][0]
    wrapped_pot = wrapped[key][0]
    assert not isinstance(original_pot, GatedPotential)
    assert isinstance(wrapped_pot, GatedPotential)
    assert list(iter_gated_potentials(wrapped))[0][0] == key

    sampled = sample_L0_gates(wrapped, seed=0)
    assert key in sampled
    assert wrapped[key][0].gate_mode == "sample"

    deterministic = set_L0_gates_deterministic(wrapped)
    assert key in deterministic
    assert wrapped[key][0].gate_mode == "deterministic"


def test_l0_trainer_gradients_touch_only_gate_parameter() -> None:
    forcefield, key = _gated_forcefield(log_alpha=0.0)
    optimizer = AdamMaskedOptimizer(
        L=forcefield.param_array(),
        mask=forcefield.param_mask,
        lr=0.01,
        seed=0,
    )
    trainer = L0InteractionTrainerAnalytic(
        forcefield,
        optimizer,
        L0_lambda=2.0,
        cost_weights={key: 3.0},
    )

    out = trainer.step({"step_index": 5}, apply_update=False)
    gate = trainer.forcefield[key][0]
    expected_grad = 2.0 * 3.0 * gate.active_probability_grad()
    gate_idx = forcefield.n_params() - 1

    assert out["name"] == "L0"
    assert out["hessian"] is None
    assert out["meta"]["n_gates"] == 1
    assert np.isclose(out["loss"], 2.0 * 3.0 * gate.active_probability())
    np.testing.assert_allclose(out["grad"][:gate_idx], 0.0)
    assert out["grad"][gate_idx] == expected_grad


def test_l0_trainer_apply_update_changes_gate_log_alpha() -> None:
    forcefield, _ = _gated_forcefield(log_alpha=0.0)
    optimizer = AdamMaskedOptimizer(
        L=forcefield.param_array(),
        mask=forcefield.param_mask,
        lr=0.01,
        seed=0,
    )
    trainer = L0InteractionTrainerAnalytic(forcefield, optimizer, L0_lambda=1.0)
    before = trainer.forcefield.param_array().copy()

    out = trainer.step({"step_index": 0}, apply_update=True)
    after = trainer.forcefield.param_array()

    gate_idx = forcefield.n_params() - 1
    assert out["update"][gate_idx] < 0.0
    assert after[gate_idx] < before[gate_idx]


def test_writelmpff_uses_lammps_params_for_gated_potential(tmp_path) -> None:
    settings = tmp_path / "settings.lmp"
    output = tmp_path / "settings_out.lmp"
    settings.write_text("pair_coeff A B 9.0 9.0 9.0\n", encoding="utf-8")

    forcefield, key = _gated_forcefield(log_alpha=0.0)
    WriteLmpFF(
        str(settings),
        str(output),
        {key: [forcefield[key][0]]},
        "gauss/cut",
        topology_arrays=EMPTY_TOPOLOGY_ARRAYS,
    )

    tokens = output.read_text(encoding="utf-8").split()
    np.testing.assert_allclose([float(x) for x in tokens[3:6]], [1.0, 1.0, 0.5])
