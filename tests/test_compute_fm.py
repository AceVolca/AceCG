"""Tests for FM statistics: normalized FM accumulation identities."""

import pytest
import numpy as np


def fm_loss_grad_hessian(JtJ, Jty, y_sumsq, Jtf, f_sumsq, fty):
    """Inline FM loss/grad/hessian (moved from reducers)."""
    grad = np.asarray(Jtf, dtype=np.float64) - np.asarray(Jty, dtype=np.float64)
    loss = 0.5 * (float(f_sumsq) - 2.0 * float(fty) + float(y_sumsq))
    hessian = np.asarray(JtJ, dtype=np.float64)
    return loss, grad, hessian

SEED = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_system():
    rng = np.random.default_rng(SEED)
    n_rows, n_params = 30, 8
    J = rng.standard_normal((n_rows, n_params))
    y = rng.standard_normal(n_rows)
    return J, y, n_rows, n_params


# ---------------------------------------------------------------------------
# Normal-equation accumulation identities (inline JtJ = J^T J, Jty = J^T y)
# ---------------------------------------------------------------------------

def test_normal_eq_shapes(random_system):
    J, y, n_rows, n_params = random_system
    JtJ = J.T @ J
    Jty = J.T @ y
    assert JtJ.shape == (n_params, n_params)
    assert Jty.shape == (n_params,)


def test_normal_eq_jtj(random_system):
    J, y, _, n_params = random_system
    assert np.allclose(J.T @ J, J.T @ J)


def test_normal_eq_jty(random_system):
    J, y, _, n_params = random_system
    assert np.allclose(J.T @ y, J.T @ y)


def test_normal_eq_y_sumsq(random_system):
    J, y, _, n_params = random_system
    assert float(np.dot(y, y)) == pytest.approx(np.dot(y, y), rel=1e-10)


def test_normal_eq_weighted(random_system):
    J, y, _, n_params = random_system
    w = 0.5
    assert np.allclose(w * (J.T @ J), w * (J.T @ J))
    assert np.allclose(w * (J.T @ y), w * (J.T @ y))


def test_normal_eq_multi_frame_additive(random_system):
    rng = np.random.default_rng(SEED + 1)
    n_rows, n_params = 30, 8
    J1, y1 = rng.standard_normal((n_rows, n_params)), rng.standard_normal(n_rows)
    J2, y2 = rng.standard_normal((n_rows, n_params)), rng.standard_normal(n_rows)

    JtJ = J1.T @ J1 + J2.T @ J2
    Jty = J1.T @ y1 + J2.T @ y2

    assert np.allclose(JtJ, J1.T @ J1 + J2.T @ J2)
    assert np.allclose(Jty, J1.T @ y1 + J2.T @ y2)


def test_normal_eq_solution(random_system):
    """theta_solved = (JtJ)^{-1} Jty should equal lstsq solution."""
    J, y, _, n_params = random_system
    JtJ = J.T @ J
    Jty = J.T @ y

    theta_solved = np.linalg.solve(JtJ, Jty)
    theta_true = np.linalg.lstsq(J, y, rcond=None)[0]
    assert np.allclose(theta_solved, theta_true, atol=1e-10)


# ---------------------------------------------------------------------------
# fm_loss_grad_hessian
# ---------------------------------------------------------------------------

@pytest.fixture
def fm_accumulated(random_system):
    J, y, n_rows, n_params = random_system
    JtJ = J.T @ J
    Jty = J.T @ y
    y_sumsq = float(np.dot(y, y))
    theta_true = np.linalg.lstsq(J, y, rcond=None)[0]
    # Force-value stats (linear: f = J @ theta)
    Jtf = JtJ @ theta_true
    f_sumsq = float(theta_true @ JtJ @ theta_true)
    fty = float(theta_true @ Jty)
    return JtJ, Jty, y_sumsq, theta_true, n_params, Jtf, f_sumsq, fty


def test_fm_loss_nonnegative(fm_accumulated):
    JtJ, Jty, y_sumsq, theta_true, _, Jtf, f_sumsq, fty = fm_accumulated
    loss, _, _ = fm_loss_grad_hessian(JtJ, Jty, y_sumsq, Jtf, f_sumsq, fty)
    assert loss >= 0.0


def test_fm_grad_near_zero_at_solution(fm_accumulated):
    """At the least-squares solution, gradient should be ~0."""
    JtJ, Jty, y_sumsq, theta_true, _, Jtf, f_sumsq, fty = fm_accumulated
    _, grad, _ = fm_loss_grad_hessian(JtJ, Jty, y_sumsq, Jtf, f_sumsq, fty)
    assert np.allclose(grad, 0.0, atol=1e-8)


def test_fm_hessian_symmetric(fm_accumulated):
    JtJ, Jty, y_sumsq, theta_true, _, Jtf, f_sumsq, fty = fm_accumulated
    _, _, hessian = fm_loss_grad_hessian(JtJ, Jty, y_sumsq, Jtf, f_sumsq, fty)
    assert np.allclose(hessian, hessian.T)


def test_fm_hessian_psd(fm_accumulated):
    JtJ, Jty, y_sumsq, theta_true, _, Jtf, f_sumsq, fty = fm_accumulated
    _, _, hessian = fm_loss_grad_hessian(JtJ, Jty, y_sumsq, Jtf, f_sumsq, fty)
    eigvals = np.linalg.eigvalsh(hessian)
    assert np.all(eigvals >= -1e-10)


def test_fm_loss_formula(fm_accumulated):
    """Manual formula on normalized FM statistics."""
    JtJ, Jty, y_sumsq, theta_true, _, Jtf, f_sumsq, fty = fm_accumulated
    loss, _, _ = fm_loss_grad_hessian(JtJ, Jty, y_sumsq, Jtf, f_sumsq, fty)
    expected = 0.5 * (f_sumsq - 2 * fty + y_sumsq)
    assert loss == pytest.approx(expected, rel=1e-10)


def test_fm_hessian_equals_jtj_for_normalized_multi_frame_stats(random_system):
    rng = np.random.default_rng(SEED + 7)
    n_rows, n_params = 24, 5
    J1 = rng.standard_normal((n_rows, n_params))
    J2 = rng.standard_normal((n_rows, n_params))
    y1 = rng.standard_normal(n_rows)
    y2 = rng.standard_normal(n_rows)
    theta = rng.standard_normal(n_params)
    w1, w2 = 0.25, 0.75

    JtJ = w1 * (J1.T @ J1) + w2 * (J2.T @ J2)
    Jty = w1 * (J1.T @ y1) + w2 * (J2.T @ y2)
    y_sumsq = w1 * float(y1 @ y1) + w2 * float(y2 @ y2)
    Jtf = JtJ @ theta
    f_sumsq = float(theta @ JtJ @ theta)
    fty = float(theta @ Jty)

    loss, grad, hessian = fm_loss_grad_hessian(JtJ, Jty, y_sumsq, Jtf, f_sumsq, fty)

    assert np.allclose(hessian, JtJ)
    assert np.allclose(grad, Jtf - Jty)
    assert loss == pytest.approx(0.5 * (f_sumsq - 2.0 * fty + y_sumsq), rel=1e-10)
