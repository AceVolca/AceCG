"""Direct tests for the redesigned FMMatrixSolver."""

from __future__ import annotations

import numpy as np

from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.solvers.fm_matrix import FMMatrixSolver
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey


def _make_forcefield(*, k=1.0, r0=0.5) -> Forcefield:
    key = InteractionKey.bond("A", "B")
    pot = HarmonicPotential("A", "B", k=k, r0=r0)
    return Forcefield({key: [pot]})


def _make_batch(
    JtJ,
    Jty,
    y_sumsq,
    *,
    nframe=1,
    weight_sum=1.0,
    n_atoms_obs=2,
    step_index=0,
):
    return {
        "JtJ": np.asarray(JtJ, dtype=np.float64),
        "Jty": np.asarray(Jty, dtype=np.float64),
        "y_sumsq": float(y_sumsq),
        "nframe": int(nframe),
        "weight_sum": float(weight_sum),
        "n_atoms_obs": int(n_atoms_obs),
        "step_index": int(step_index),
    }


def test_schema_exposes_batch_and_return_contract():
    schema = FMMatrixSolver.schema()
    assert "batch" in schema
    assert "return" in schema
    assert "JtJ" in schema["batch"]
    assert "params" in schema["return"]


def test_ols_solves_full_system_and_keeps_original_forcefield_untouched():
    forcefield = _make_forcefield(k=1.0, r0=0.5)
    solver = FMMatrixSolver(forcefield, mode="ols")

    JtJ = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    theta_true = np.array([2.0, 0.5], dtype=np.float64)
    Jty = JtJ @ theta_true
    batch = _make_batch(JtJ, Jty, theta_true @ JtJ @ theta_true)

    result = solver.solve(batch)

    np.testing.assert_allclose(result["params"], theta_true, atol=1.0e-12)
    assert abs(result["loss"]) < 1.0e-12
    np.testing.assert_allclose(forcefield.param_array(), [1.0, 0.5], atol=1.0e-12)
    np.testing.assert_allclose(solver.get_params(), theta_true, atol=1.0e-12)


def test_masked_ols_uses_shifted_rhs_for_frozen_parameters():
    forcefield = _make_forcefield(k=1.0, r0=0.5)
    forcefield.param_mask = np.array([True, False], dtype=bool)
    solver = FMMatrixSolver(forcefield, mode="ols")

    JtJ = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    theta_expected = np.array([2.0, 0.5], dtype=np.float64)
    Jty = JtJ @ theta_expected
    batch = _make_batch(JtJ, Jty, theta_expected @ JtJ @ theta_expected)

    result = solver.solve(batch)

    np.testing.assert_allclose(result["params"], theta_expected, atol=1.0e-12)
    assert result["meta"]["active_n_params"] == 1


def test_ridge_solves_regularized_system():
    forcefield = _make_forcefield(k=1.0, r0=0.5)
    solver = FMMatrixSolver(forcefield, mode="ridge", ridge_alpha=1.0)

    JtJ = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    Jty = np.array([4.0, 0.5], dtype=np.float64)
    expected = np.linalg.solve(JtJ + np.eye(2, dtype=np.float64), Jty)
    batch = _make_batch(JtJ, Jty, 0.0)

    result = solver.solve(batch)

    np.testing.assert_allclose(result["params"], expected, atol=1.0e-12)
    assert result["meta"]["ridge_alpha"] == 1.0


def test_bayesian_returns_finite_coefficients_and_meta():
    forcefield = _make_forcefield(k=1.0, r0=0.5)
    solver = FMMatrixSolver(
        forcefield,
        mode="bayesian",
        bayesian_min_iter=1,
        bayesian_max_iter=25,
        bayesian_tol=1.0e-8,
    )

    JtJ = np.array([[4.0, 0.5], [0.5, 2.0]], dtype=np.float64)
    theta_ref = np.array([1.5, 0.25], dtype=np.float64)
    Jty = JtJ @ theta_ref
    batch = _make_batch(
        JtJ,
        Jty,
        theta_ref @ JtJ @ theta_ref,
        nframe=10,
        weight_sum=10.0,
        n_atoms_obs=2,
    )

    result = solver.solve(batch)

    assert result["mode"] == "bayesian"
    assert result["params"].shape == (2,)
    assert np.all(np.isfinite(result["params"]))
    assert result["meta"]["bayesian_iterations"] >= 1
    assert result["meta"]["bayesian_beta"] > 0.0
    assert result["meta"]["bayesian_alpha"].shape == (2,)


def test_empty_batch_returns_current_parameters():
    forcefield = _make_forcefield(k=1.25, r0=0.75)
    solver = FMMatrixSolver(forcefield, mode="ols")
    batch = _make_batch(
        np.zeros((2, 2), dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        0.0,
        nframe=0,
        weight_sum=0.0,
        n_atoms_obs=0,
    )

    result = solver.solve(batch)

    np.testing.assert_allclose(result["params"], [1.25, 0.75], atol=1.0e-12)
    assert result["loss"] == 0.0
