"""Real-frame oracle for the gradient estimators named in the review brief §3.

Everything here runs on five real frames of the 6-site CG DOPC system shipped
in ``tests/test_data/dopc_cg6/`` and driven through the production kernels. No
stub, no ``SimpleNamespace``, no hand-built ``FrameGeometry``, no invented
distances. That is the point: these tests can fail for a sign error, a missing
or wrong β, a dropped minimum image, a wrong cutoff convention, or a broken
ensemble average — none of which perturbs a two-particle toy.

Covers findings F-003 (analytical ``dU/dλ`` with no finite-difference check),
F-004 (CD-REM gradient estimator with no coverage at all) and F-005
(``MSETrainerAnalytic`` / ``MultiTrainerAnalytic`` with no coverage at all).

Layout of the ensembles built from the fixture:

* frames 0, 1        -> the reference ("AA") ensemble
* frames 2, 3, 4     -> the model ("CG") ensemble
* CD-REM x-subsamples: x0 conditioned on frames {0, 1}, x1 on frames {2, 3};
  the joint model ensemble is all five frames.

These splits are arbitrary but fixed, which is all an estimator test needs —
what must be real is the geometry underneath them.
"""

from __future__ import annotations

import numpy as np
import pytest

from real_frames import beta_at, dopc_energy_statistics, dopc_frame_geometries

from AceCG.compute.energy import energy
from AceCG.optimizers.newton_raphson import NewtonRaphsonOptimizer
from AceCG.optimizers.sgd import SGDMaskedOptimizer
from AceCG.trainers.analytic.cdrem import CDREMTrainerAnalytic
from AceCG.trainers.analytic.mse import MSETrainerAnalytic
from AceCG.trainers.analytic.multi import MultiTrainerAnalytic
from AceCG.trainers.analytic.rem import REMTrainerAnalytic


TEMPERATURE = 300.0
BETA = beta_at(TEMPERATURE)

AA_FRAMES = (0, 1)
CG_FRAMES = (2, 3, 4)
CDREM_CONDITIONALS = ((0, 1), (2, 3))


@pytest.fixture(scope="module")
def statistics():
    """``(forcefield, energies, energy_grad, energy_hessian)`` on real frames."""
    return dopc_energy_statistics(12)


def _sgd_trainer(cls, forcefield, **kwargs):
    optimizer = SGDMaskedOptimizer(
        forcefield.param_array().copy(),
        np.ones(forcefield.n_params(), dtype=bool),
        lr=0.0,
    )
    return cls(forcefield=forcefield, optimizer=optimizer, **kwargs)


def _newton_trainer(cls, forcefield, **kwargs):
    optimizer = NewtonRaphsonOptimizer(
        forcefield.param_array().copy(),
        np.ones(forcefield.n_params(), dtype=bool),
        lr=0.0,
    )
    return cls(forcefield=forcefield, optimizer=optimizer, **kwargs)


def _frame_energies(forcefield, geometries, params):
    """Total energy of every frame at a given parameter vector."""
    original = forcefield.param_array().copy()
    try:
        forcefield.update_params(params)
        return np.asarray(
            [
                float(energy(geometry, forcefield, return_value=True)["energy"])
                for geometry in geometries
            ],
            dtype=float,
        )
    finally:
        forcefield.update_params(original)


def _displacement(forcefield, direction, rel_step=1.0e-6):
    """Scale a unit direction per parameter, not globally.

    The pair blocks carry coefficients four orders of magnitude larger than the
    angle blocks, so one global step size is either negligible for the pairs or
    enormous for the angles. An enormous angle step is not merely inaccurate:
    the bonded ``min(U) = 0`` gauge picks a *global* argmin, and a large
    excursion moves it discontinuously between competing wells, at which point
    U is not differentiable along the path and no finite difference means
    anything. Per-parameter scaling keeps every block in its linear regime.
    """
    params = np.asarray(forcefield.param_array(), dtype=float)
    return rel_step * np.maximum(1.0, np.abs(params)) * np.asarray(direction, dtype=float)


def _directional_derivative(forcefield, geometries, direction, rel_step=1.0e-6):
    """Central difference of every frame energy along a scaled ``direction``.

    Returns ``dU · displacement``, i.e. the directional derivative already
    multiplied by the displacement, so callers compare it against
    ``energy_grad @ displacement``.
    """
    params = forcefield.param_array().copy()
    displacement = _displacement(forcefield, direction, rel_step)
    plus = _frame_energies(forcefield, geometries, params + displacement)
    minus = _frame_energies(forcefield, geometries, params - displacement)
    return (plus - minus) / 2.0


@pytest.fixture(scope="module")
def probe_directions(statistics):
    """A few reproducible unit directions in parameter space."""
    forcefield = statistics[0]
    rng = np.random.default_rng(20260810)
    n_params = forcefield.n_params()
    directions = [rng.standard_normal(n_params) for _ in range(3)]
    return [direction / np.linalg.norm(direction) for direction in directions]


# ─────────────────────────────────────────────────────────────────────────────
# The foundation: analytical dU/dλ against a finite difference of real energies
# ─────────────────────────────────────────────────────────────────────────────


def test_real_frame_energy_gradient_matches_finite_difference(
    statistics, probe_directions
):
    """``energy()['energy_grad']`` is the parameter derivative of ``U``.

    Every estimator below consumes this array, so if it is wrong nothing
    downstream can be right. The check runs on the real pair / bond / angle
    B-spline blocks at once, which means it also covers the bonded minimum-gauge
    shift: that gauge coordinate moves with the parameters, and the analytic
    gradient claims the movement contributes nothing.
    """
    forcefield, _, energy_grad, _ = statistics
    geometries = dopc_frame_geometries(forcefield)

    for direction in probe_directions:
        numeric = _directional_derivative(forcefield, geometries, direction)
        analytic = energy_grad @ _displacement(forcefield, direction)
        np.testing.assert_allclose(analytic, numeric, rtol=1.0e-6, atol=1.0e-9)


def test_real_frame_energy_hessian_is_zero_for_a_linear_force_basis(statistics):
    """B-spline coefficients enter U linearly, so d²U/dλ² must vanish exactly.

    Not a triviality: a nonzero entry here would mean either the linearity mask
    is lying or the gauge shift is being differentiated twice, and the REM /
    CD-REM Hessians would inherit the error.
    """
    forcefield, _, _, energy_hessian = statistics
    linear = np.concatenate(
        [
            np.asarray(potential.is_param_linear(), dtype=bool).reshape(-1)
            for _, potential in forcefield.iter_potentials()
        ]
    )
    assert linear.all()
    assert np.max(np.abs(energy_hessian)) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# REM
# ─────────────────────────────────────────────────────────────────────────────


def _rem_batch(energy_grad, *, aa=AA_FRAMES, cg=CG_FRAMES):
    return {
        "energy_grad_AA": energy_grad[list(aa)].mean(axis=0),
        "energy_grad_CG": energy_grad[list(cg)].mean(axis=0),
    }


def test_rem_gradient_is_the_derivative_of_the_fixed_sample_objective(
    statistics, probe_directions
):
    """grad = ∂/∂λ of β(⟨U⟩_AA − ⟨U⟩_CG) at fixed samples.

    This is the definition of the REM gradient with the sampling held fixed, so
    a flipped sign, a missing β, or an AA/CG swap all show up here as a factor
    the finite difference does not reproduce.
    """
    forcefield, _, energy_grad, _ = statistics
    geometries = dopc_frame_geometries(forcefield)
    trainer = _sgd_trainer(REMTrainerAnalytic, forcefield, beta=BETA)
    grad = trainer.step(_rem_batch(energy_grad), apply_update=False)["grad"]

    for direction in probe_directions:
        per_frame = _directional_derivative(forcefield, geometries, direction)
        numeric = BETA * (
            per_frame[list(AA_FRAMES)].mean() - per_frame[list(CG_FRAMES)].mean()
        )
        analytic = grad @ _displacement(forcefield, direction)
        np.testing.assert_allclose(analytic, numeric, rtol=1.0e-6, atol=1.0e-9)


def test_rem_gradient_vanishes_when_the_two_ensembles_coincide(statistics):
    """Identical reference and model ensembles must give an exactly zero drive."""
    forcefield, _, energy_grad, _ = statistics
    trainer = _sgd_trainer(REMTrainerAnalytic, forcefield, beta=BETA)
    batch = _rem_batch(energy_grad, aa=CG_FRAMES, cg=CG_FRAMES)
    grad = trainer.step(batch, apply_update=False)["grad"]
    assert np.max(np.abs(grad)) == 0.0


def test_rem_gradient_is_antisymmetric_and_scales_as_one_over_temperature(statistics):
    """Swapping the ensembles flips the sign; doubling T halves the gradient."""
    forcefield, _, energy_grad, _ = statistics
    forward = _sgd_trainer(REMTrainerAnalytic, forcefield, beta=BETA).step(
        _rem_batch(energy_grad), apply_update=False
    )["grad"]
    reversed_grad = _sgd_trainer(REMTrainerAnalytic, forcefield, beta=BETA).step(
        _rem_batch(energy_grad, aa=CG_FRAMES, cg=AA_FRAMES), apply_update=False
    )["grad"]
    np.testing.assert_allclose(reversed_grad, -forward, rtol=1.0e-12, atol=0.0)

    hot = _sgd_trainer(
        REMTrainerAnalytic, forcefield, beta=beta_at(2.0 * TEMPERATURE)
    ).step(_rem_batch(energy_grad), apply_update=False)["grad"]
    np.testing.assert_allclose(hot, 0.5 * forward, rtol=1.0e-12, atol=0.0)


def test_rem_hessian_matches_the_fluctuation_formula(statistics):
    """H = β[⟨d²U⟩_AA − ⟨d²U⟩_CG + β Cov_CG(dU/dλ)] on real frames."""
    forcefield, _, energy_grad, energy_hessian = statistics
    cg = list(CG_FRAMES)
    grad_outer_CG = np.einsum("fi,fj->ij", energy_grad[cg], energy_grad[cg]) / len(cg)
    batch = dict(_rem_batch(energy_grad))
    batch.update(
        {
            "d2U_AA": energy_hessian[list(AA_FRAMES)].mean(axis=0),
            "d2U_CG": energy_hessian[cg].mean(axis=0),
            "grad_outer_CG": grad_outer_CG,
        }
    )
    trainer = _newton_trainer(REMTrainerAnalytic, forcefield, beta=BETA)
    hessian = trainer.step(batch, apply_update=False)["hessian"]

    mean_cg = energy_grad[cg].mean(axis=0)
    expected = BETA * (
        batch["d2U_AA"]
        - batch["d2U_CG"]
        + BETA * (grad_outer_CG - np.outer(mean_cg, mean_cg))
    )
    np.testing.assert_allclose(hessian, expected, rtol=1.0e-12, atol=0.0)
    # The covariance is a real, positive-semidefinite fluctuation matrix, not a
    # numerical artefact of the outer-product bookkeeping.
    covariance = grad_outer_CG - np.outer(mean_cg, mean_cg)
    np.testing.assert_allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-6)
    assert np.min(np.linalg.eigvalsh(covariance)) > -1.0e-6 * np.max(
        np.abs(covariance)
    )


def test_rem_gradient_pins_real_frame_values(statistics):
    """Golden values for the REM drive on the shipped frames."""
    forcefield, _, energy_grad, _ = statistics
    out = _sgd_trainer(REMTrainerAnalytic, forcefield, beta=BETA).step(
        _rem_batch(energy_grad), apply_update=False
    )
    grad = out["grad"]
    assert grad.shape == (192,)
    assert float(np.linalg.norm(grad)) == pytest.approx(784.8324125458, rel=1.0e-9)
    assert float(grad.sum()) == pytest.approx(-4084.9553995016, rel=1.0e-9)
    np.testing.assert_allclose(
        grad[[3, 40, 100, 150, 191]],
        [
            -8.4017354e00,
            -5.8227342e00,
            -4.0041175e01,
            5.7610416e-03,
            -3.7014969e00,
        ],
        rtol=1.0e-7,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CD-REM  (F-004: this estimator previously had no test of any kind)
# ─────────────────────────────────────────────────────────────────────────────


def _cdrem_batch(energy_grad, energy_hessian=None, *, x_weight=None):
    z_by_x = np.stack(
        [energy_grad[list(group)].mean(axis=0) for group in CDREM_CONDITIONALS]
    )
    batch = {
        "energy_grad_z_by_x": z_by_x,
        "energy_grad_xz": energy_grad.mean(axis=0),
    }
    if x_weight is not None:
        batch["x_weight"] = np.asarray(x_weight, dtype=float)
    if energy_hessian is not None:
        n_frames = energy_grad.shape[0]
        outer = (
            np.einsum("fi,fj->ij", energy_grad, energy_grad) / n_frames
        )
        cov_z_by_x = []
        d2U_z_by_x = []
        for group in CDREM_CONDITIONALS:
            rows = energy_grad[list(group)]
            mean = rows.mean(axis=0)
            cov_z_by_x.append(
                np.einsum("fi,fj->ij", rows, rows) / len(group) - np.outer(mean, mean)
            )
            d2U_z_by_x.append(energy_hessian[list(group)].mean(axis=0))
        batch.update(
            {
                "d2U_z_by_x": np.stack(d2U_z_by_x),
                "d2U_xz": energy_hessian.mean(axis=0),
                "energy_grad_outer_xz": outer,
                "cov_z_by_x": np.stack(cov_z_by_x),
            }
        )
    return batch


def test_cdrem_gradient_is_the_derivative_of_the_fixed_sample_objective(
    statistics, probe_directions
):
    """grad = ∂/∂λ of β(E_x E_{z|x}[U] − E_{x,z}[U]) at fixed samples.

    The positive phase is a *weighted average over x of conditional averages*,
    not a flat average over all conditioned frames; the two differ whenever the
    conditional groups have unequal size, so this also pins the bookkeeping.
    """
    forcefield, _, energy_grad, _ = statistics
    geometries = dopc_frame_geometries(forcefield)
    weights = np.array([0.25, 0.75])
    trainer = _sgd_trainer(CDREMTrainerAnalytic, forcefield, beta=BETA)
    out = trainer.step(_cdrem_batch(energy_grad, x_weight=weights), apply_update=False)

    for direction in probe_directions:
        per_frame = _directional_derivative(forcefield, geometries, direction)
        positive = sum(
            weight * per_frame[list(group)].mean()
            for weight, group in zip(weights, CDREM_CONDITIONALS)
        )
        numeric = BETA * (positive - per_frame.mean())
        analytic = out["grad"] @ _displacement(forcefield, direction)
        np.testing.assert_allclose(analytic, numeric, rtol=1.0e-6, atol=1.0e-9)


def test_cdrem_positive_phase_honours_x_weights(statistics):
    """A degenerate x-weight must reproduce exactly that conditional average."""
    forcefield, _, energy_grad, _ = statistics
    for index, group in enumerate(CDREM_CONDITIONALS):
        weights = np.zeros(len(CDREM_CONDITIONALS))
        weights[index] = 3.0  # unnormalized on purpose: the trainer normalizes
        out = _sgd_trainer(CDREMTrainerAnalytic, forcefield, beta=BETA).step(
            _cdrem_batch(energy_grad, x_weight=weights), apply_update=False
        )
        np.testing.assert_allclose(
            out["energy_grad_pos"],
            energy_grad[list(group)].mean(axis=0),
            rtol=1.0e-12,
        )
        np.testing.assert_allclose(
            out["grad"],
            BETA * (out["energy_grad_pos"] - energy_grad.mean(axis=0)),
            rtol=1.0e-12,
        )

    uniform = _sgd_trainer(CDREMTrainerAnalytic, forcefield, beta=BETA).step(
        _cdrem_batch(energy_grad), apply_update=False
    )
    explicit = _sgd_trainer(CDREMTrainerAnalytic, forcefield, beta=BETA).step(
        _cdrem_batch(energy_grad, x_weight=[0.5, 0.5]), apply_update=False
    )
    np.testing.assert_allclose(uniform["grad"], explicit["grad"], rtol=1.0e-12)


def test_cdrem_hessian_matches_the_latent_variable_formula(statistics):
    """H = β(⟨d²U⟩_pos − ⟨d²U⟩_neg + β(Cov_{x,z} − E_x Cov_{z|x}))."""
    forcefield, _, energy_grad, energy_hessian = statistics
    batch = _cdrem_batch(energy_grad, energy_hessian, x_weight=[0.25, 0.75])
    trainer = _newton_trainer(CDREMTrainerAnalytic, forcefield, beta=BETA)
    out = trainer.step(batch, apply_update=False)

    weights = np.array([0.25, 0.75])
    weights = weights / weights.sum()
    joint_mean = energy_grad.mean(axis=0)
    cov_neg = batch["energy_grad_outer_xz"] - np.outer(joint_mean, joint_mean)
    cov_pos = np.tensordot(weights, batch["cov_z_by_x"], axes=(0, 0))
    expected = BETA * (
        np.tensordot(weights, batch["d2U_z_by_x"], axes=(0, 0))
        - batch["d2U_xz"]
        + BETA * (cov_neg - cov_pos)
    )
    np.testing.assert_allclose(out["hessian"], expected, rtol=1.0e-10, atol=1.0e-12)
    np.testing.assert_allclose(out["cov_neg"], cov_neg, rtol=1.0e-10, atol=1.0e-12)
    np.testing.assert_allclose(out["cov_pos_cond"], cov_pos, rtol=1.0e-10, atol=1.0e-12)


def test_cdrem_rejects_a_hessian_optimizer_without_second_order_statistics(statistics):
    """The failure mode must be a raise, never a silently first-order step."""
    forcefield, _, energy_grad, _ = statistics
    trainer = _newton_trainer(CDREMTrainerAnalytic, forcefield, beta=BETA)
    with pytest.raises(ValueError, match="second-order batch statistics"):
        trainer.step(_cdrem_batch(energy_grad), apply_update=False)


def test_cdrem_gradient_pins_real_frame_values(statistics):
    """Golden values for the CD-REM drive on the shipped frames."""
    forcefield, _, energy_grad, _ = statistics
    out = _sgd_trainer(CDREMTrainerAnalytic, forcefield, beta=BETA).step(
        _cdrem_batch(energy_grad, x_weight=[0.25, 0.75]), apply_update=False
    )
    grad = out["grad"]
    assert out["name"] == "CDREM"
    assert out["meta"]["n_x"] == 2
    assert grad.shape == (192,)
    assert float(np.linalg.norm(grad)) == pytest.approx(213.0033004111, rel=1.0e-9)
    assert float(grad.sum()) == pytest.approx(-137.7427452069, rel=1.0e-9)
    np.testing.assert_allclose(
        grad[[3, 40, 100, 150, 191]],
        [
            1.2117908e00,
            -1.1526910e00,
            -1.8659703e00,
            1.5150456e-03,
            1.0608093e00,
        ],
        rtol=1.0e-7,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MSE / PMF matching  (F-005)
# ─────────────────────────────────────────────────────────────────────────────


def _mse_batch(energy_grad, *, offset_CG=0.0, offset_AA=0.0):
    n_frames = energy_grad.shape[0]
    bin_idx = np.array([0, 0, 1, 1, 2], dtype=np.int64)[:n_frames]
    pmf_AA = np.array([0.0, 0.6, 1.4], dtype=float) + offset_AA
    pmf_CG = np.array([0.2, 0.5, 2.0], dtype=float) + offset_CG
    return {
        "pmf_AA": pmf_AA,
        "pmf_CG": pmf_CG,
        "CG_bin_idx_frame": bin_idx,
        "energy_grad_frame": energy_grad,
    }


def test_mse_loss_and_gradient_are_invariant_under_a_constant_pmf_offset(statistics):
    """A PMF is defined up to an additive constant; the objective must agree.

    The gauge shift ``c = mean(pmf_CG - pmf_AA)`` exists precisely for this. If
    it were dropped or computed over the wrong axis, shifting either PMF would
    move the loss and the optimizer would chase an unphysical offset.
    """
    forcefield, _, energy_grad, _ = statistics
    reference = _sgd_trainer(MSETrainerAnalytic, forcefield).step(
        _mse_batch(energy_grad), apply_update=False
    )
    for offset_CG, offset_AA in ((7.5, 0.0), (0.0, -3.25), (2.0, 2.0)):
        shifted = _sgd_trainer(MSETrainerAnalytic, forcefield).step(
            _mse_batch(energy_grad, offset_CG=offset_CG, offset_AA=offset_AA),
            apply_update=False,
        )
        # The offsets are cancelled by subtraction, so agreement is limited by
        # float64 cancellation on the offset, not by the estimator.
        assert shifted["loss"] == pytest.approx(reference["loss"], rel=1.0e-9)
        np.testing.assert_allclose(
            shifted["grad"], reference["grad"], rtol=1.0e-8, atol=1.0e-9
        )


def test_mse_gradient_matches_the_gauge_fixed_derivation(statistics):
    """grad = Σ_s ΔF(s)·⟨dU/dλ⟩_{CG|s}, with ΔF measured after gauge fixing."""
    forcefield, _, energy_grad, _ = statistics
    batch = _mse_batch(energy_grad)
    out = _sgd_trainer(MSETrainerAnalytic, forcefield).step(batch, apply_update=False)

    gauge = float(np.mean(batch["pmf_CG"] - batch["pmf_AA"]))
    delta = batch["pmf_CG"] - gauge - batch["pmf_AA"]
    bin_idx = batch["CG_bin_idx_frame"]
    expected = np.zeros(energy_grad.shape[1], dtype=float)
    for bin_index in np.unique(bin_idx):
        conditional = energy_grad[bin_idx == bin_index].mean(axis=0)
        expected += delta[bin_index] * conditional

    assert out["meta"]["gauge_shift"] == pytest.approx(gauge, rel=1.0e-12)
    assert out["loss"] == pytest.approx(0.5 * float(np.sum(delta ** 2)), rel=1.0e-12)
    np.testing.assert_allclose(out["grad"], expected, rtol=1.0e-10, atol=1.0e-12)


def test_mse_frame_weights_reweight_the_conditional_bin_average(statistics):
    """A zero weight must remove that frame from its bin's conditional average."""
    forcefield, _, energy_grad, _ = statistics
    batch = _mse_batch(energy_grad)
    weights = np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    weighted = dict(batch, frame_weight=weights)
    out = _sgd_trainer(MSETrainerAnalytic, forcefield).step(weighted, apply_update=False)

    gauge = float(np.mean(batch["pmf_CG"] - batch["pmf_AA"]))
    delta = batch["pmf_CG"] - gauge - batch["pmf_AA"]
    bin_idx = batch["CG_bin_idx_frame"]
    expected = np.zeros(energy_grad.shape[1], dtype=float)
    for bin_index in np.unique(bin_idx):
        mask = (bin_idx == bin_index) & (weights > 0.0)
        expected += delta[bin_index] * energy_grad[mask].mean(axis=0)
    np.testing.assert_allclose(out["grad"], expected, rtol=1.0e-10, atol=1.0e-12)
    assert out["meta"]["frame_weight_source"] == "batch"


def test_mse_reports_bins_no_frame_visited(statistics):
    """An unvisited bin contributes nothing and is reported, not silently dropped."""
    forcefield, _, energy_grad, _ = statistics
    batch = _mse_batch(energy_grad)
    batch["pmf_AA"] = np.append(batch["pmf_AA"], 2.2)
    batch["pmf_CG"] = np.append(batch["pmf_CG"], 3.0)
    out = _sgd_trainer(MSETrainerAnalytic, forcefield).step(batch, apply_update=False)
    assert out["meta"]["missing_bins"] == [3]
    assert out["meta"]["n_observed_bins"] == 3


def test_mse_pins_real_frame_values(statistics):
    """Golden values for the PMF-matching objective on the shipped frames."""
    forcefield, _, energy_grad, _ = statistics
    out = _sgd_trainer(MSETrainerAnalytic, forcefield).step(
        _mse_batch(energy_grad), apply_update=False
    )
    assert out["name"] == "MSE"
    assert out["hessian"] is None
    assert out["loss"] == pytest.approx(0.1233333333333, rel=1.0e-10)
    assert float(np.linalg.norm(out["grad"])) == pytest.approx(
        183.2118934110, rel=1.0e-9
    )
    assert float(out["grad"].sum()) == pytest.approx(717.4937904636, rel=1.0e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Multi  (F-005)
# ─────────────────────────────────────────────────────────────────────────────


def test_multi_grad_mode_combines_sub_trainer_gradients(statistics):
    """The meta gradient is the weighted sum of the sub-trainer gradients.

    Built out of the two estimators that actually get combined in production —
    a REM drive and a PMF-matching drive — on the same real frames, so a
    mis-ordered batch list or a dropped weight is visible as a number.
    """
    forcefield, _, energy_grad, _ = statistics
    weights = [2.0, 0.5]

    rem_grad = _sgd_trainer(REMTrainerAnalytic, forcefield, beta=BETA).step(
        _rem_batch(energy_grad), apply_update=False
    )["grad"]
    mse_grad = _sgd_trainer(MSETrainerAnalytic, forcefield).step(
        _mse_batch(energy_grad), apply_update=False
    )["grad"]

    multi = MultiTrainerAnalytic(
        forcefield=forcefield,
        optimizer=SGDMaskedOptimizer(
            forcefield.param_array().copy(),
            np.ones(forcefield.n_params(), dtype=bool),
            lr=0.0,
        ),
        trainer_list=[
            _sgd_trainer(REMTrainerAnalytic, forcefield, beta=BETA),
            _sgd_trainer(MSETrainerAnalytic, forcefield),
        ],
        weight_array=np.asarray(weights, dtype=float),
        combine_mode="grad",
        beta=BETA,
    )
    out = multi.step([_rem_batch(energy_grad), _mse_batch(energy_grad)])

    assert out["mode"] == "grad"
    np.testing.assert_allclose(
        out["combined_grad"],
        weights[0] * rem_grad + weights[1] * mse_grad,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert [entry["name"] for entry in out["sub"]] == ["REM", "MSE"]
    np.testing.assert_allclose(out["sub"][0]["grad"], rem_grad, rtol=1.0e-12)
    np.testing.assert_allclose(out["sub"][1]["grad"], mse_grad, rtol=1.0e-12)
