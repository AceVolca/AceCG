"""Numerical helpers for constrained multi-Gaussian table fitting."""

import numpy as np
from scipy.optimize import lsq_linear
from typing import Dict, Optional


# ── Gaussian basis helpers ───────────────────────────────────────────────


def _gaussian_basis(r: np.ndarray, r0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return the normalized Gaussian basis matrix with shape ``(N, K)``.

    Notes
    -----
    Column ``k`` is
    ``B[:, k] = exp(-(r - r0[k])**2 / (2*sigma[k]**2))
    / (sigma[k]*sqrt(2*pi))``.

    Multi-Gaussian potentials then use ``V(r) = sum_k A[k] * B[:, k]``.
    """
    r = r.reshape(-1, 1)
    r0 = r0.reshape(1, -1)
    sigma = sigma.reshape(1, -1)
    phi = np.exp(- (r - r0)**2 / (2.0 * sigma**2))
    return phi / (sigma * np.sqrt(2.0 * np.pi))


def _gaussian_basis_dr(r: np.ndarray, r0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return derivatives of the normalized Gaussian basis with respect to ``r``.

    Notes
    -----
    The derivative of each basis column is
    ``dB[:, k]/dr = -(r - r0[k]) / sigma[k]**2 * B[:, k]``.
    """
    B = _gaussian_basis(r, r0, sigma)
    r = r.reshape(-1, 1)
    r0 = r0.reshape(1, -1)
    sigma = sigma.reshape(1, -1)
    return (-(r - r0) / (sigma**2)) * B


def _model_from_params(r, params):
    A, r0, sig = _unpack_params(params)
    return _gaussian_basis(r, r0, sig) @ A


def _model_dr_from_params(r, params):
    A, r0, sig = _unpack_params(params)
    return _gaussian_basis_dr(r, r0, sig) @ A


def _init_grid(
    r: np.ndarray,
    V: np.ndarray,
    n_gauss: int,
    bounds: Optional[Dict] = None,
):
    """Return bounded initial ``A``, ``r0``, and ``sigma`` arrays.

    The initializer spaces centers across the feasible data span, starts
    widths broad enough to overlap neighboring Gaussians, and solves amplitudes
    with a ridge-stabilized linear least-squares estimate.

    Notes
    -----
    For fixed centers and widths, amplitudes solve
    ``argmin_A ||B A - V||_2**2 + lambda ||A||_2**2`` before optional
    clipping to amplitude bounds.
    """
    r = np.asarray(r, dtype=float)
    V = np.asarray(V, dtype=float)
    rmin, rmax = float(r.min()), float(r.max())
    width = rmax - rmin if rmax > rmin else 1.0

    # ── Bounds normalization ────────────────────────────────────────────
    # Fit configs can omit individual bounds or use open-ended intervals;
    # converting them once keeps the initialization logic shape-stable.

    def _pair(x, default):
        if x is None: return default
        lo, hi = x
        lo = -np.inf if lo is None else float(lo)
        hi =  np.inf if hi is None else float(hi)
        return lo, hi

    A_lb, A_ub     = _pair(bounds.get("A"),     (-np.inf, np.inf)) if bounds else (-np.inf, np.inf)
    r0_lb, r0_ub   = _pair(bounds.get("r0"),    (rmin, rmax))       if bounds else (rmin, rmax)
    sigma_lb, sigma_ub = _pair(bounds.get("sigma"), (1e-6, np.inf)) if bounds else (1e-6, np.inf)

    # Intersect center bounds with the observed table span for numerical sanity.
    r0_lb_eff = max(r0_lb, rmin)
    r0_ub_eff = min(r0_ub, rmax)
    if r0_lb_eff > r0_ub_eff:
        # Bounds outside the table cannot seed useful centers.
        r0_lb_eff, r0_ub_eff = rmin, rmax

    # ── Center initialization ───────────────────────────────────────────
    # Centers are spread across the feasible data span with a small interior
    # margin so the first nonlinear step is not pinned to active bounds.

    if n_gauss == 1:
        r0_init = np.array([(r0_lb_eff + r0_ub_eff) * 0.5], dtype=float)
    else:
        # The margin is skipped only when the feasible interval is too narrow.
        margin = 0.15 * max(r0_ub_eff - r0_lb_eff, 1e-12)
        a = max(r0_lb_eff, r0_lb_eff + margin)
        b = min(r0_ub_eff, r0_ub_eff - margin)
        # Tight intervals still need deterministic, bounded initial centers.
        if a >= b:
            a, b = r0_lb_eff, r0_ub_eff
        r0_init = np.linspace(a, b, n_gauss)

    # ── Width initialization ────────────────────────────────────────────
    # Each Gaussian starts broad enough to overlap neighbors, then clips to
    # user bounds to avoid invalid or singular widths.

    sigma_guess = max(width / (2.0 * n_gauss), 1e-3)
    sigma_init = np.full(n_gauss, sigma_guess, dtype=float)
    sigma_init = np.clip(sigma_init, sigma_lb, sigma_ub if np.isfinite(sigma_ub) else sigma_init.max())

    # ── Amplitude initialization ────────────────────────────────────────
    # For fixed centers and widths, amplitudes are linear coefficients. A
    # tiny ridge stabilizes near-collinear basis functions without changing
    # the target table scale.

    B = _gaussian_basis(r, r0_init, sigma_init)
    lam = 1e-8
    A_init = np.linalg.lstsq(B.T @ B + lam * np.eye(n_gauss), B.T @ V, rcond=None)[0]

    # ── Amplitude bounds ────────────────────────────────────────────────
    # Clipping happens after the least-squares estimate so the initializer
    # honors user constraints even when the unconstrained table fit would not.

    if np.isfinite(A_lb) or np.isfinite(A_ub):
        A_init = np.clip(A_init, A_lb, A_ub)

    return A_init, r0_init, sigma_init


def _pack_params(A: np.ndarray, r0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Pack parameter arrays as ``A0, r0_0, sigma_0, A1, ...``."""
    n = len(A)
    out = np.empty(3*n, dtype=float)
    out[0::3] = A
    out[1::3] = r0
    out[2::3] = sigma
    return out


def _unpack_params(p: np.ndarray):
    """Unpack interleaved multi-Gaussian parameters into ``A``, ``r0``, and ``sigma``."""
    A = p[0::3]
    r0 = p[1::3]
    sigma = p[2::3]
    return A, r0, sigma


def _solve_A_with_anchors(r, V, r0, sigma, anchors_r, w_data, w_c0, w_c1,
                          A_lower=None, A_upper=None):
    """Solve the bounded linear amplitude subproblem with cutoff anchors.

    Notes
    -----
    The solved objective is
    ``min_A ||w_data * (B A - V)||_2**2
    + ||w_c0 * (Ba A)||_2**2
    + ||w_c1 * (Da A)||_2**2``
    subject to optional element-wise amplitude bounds.

    ``Ba A`` represents ``V(anchor)`` and ``Da A`` represents
    ``dV/dr(anchor)``.
    """
    B  = _gaussian_basis(r, r0, sigma)
    Ba = _gaussian_basis(anchors_r, r0, sigma)
    Da = _gaussian_basis_dr(anchors_r, r0, sigma)

    M = np.vstack([w_data * B, w_c0 * Ba, w_c1 * Da])
    y = np.concatenate([w_data * V, np.zeros_like(anchors_r), np.zeros_like(anchors_r)])

    K = B.shape[1]
    lb = np.full(K, -np.inf) if A_lower is None else np.array(A_lower, dtype=float)
    ub = np.full(K,  np.inf) if A_upper is None else np.array(A_upper, dtype=float)

    # ``lsq_linear`` handles the bound constraints for amplitudes directly.
    res = lsq_linear(M, y, bounds=(lb, ub), method="trf", max_iter=5000, lsq_solver="exact")
    return res.x


def make_cutoff_anchors(r: np.ndarray, cutoff: float, n_anchor: int, span: float) -> np.ndarray:
    """Generate inclusive anchor points near the effective cutoff.

    If the requested cutoff exceeds the table range, the table maximum becomes
    the effective cutoff so anchors remain inside available data.

    Notes
    -----
    Anchors are sampled from ``[rc - span, rc]`` after clipping ``rc`` to the
    maximum table radius.
    """
    rc = float(cutoff) if np.isfinite(cutoff) else float(r.max())
    rmax = float(r.max())
    rc = min(rc, rmax)
    r_start = max(rc - span, r.min())
    if n_anchor <= 1 or r_start >= rc:
        return np.array([rc], dtype=float)
    return np.linspace(r_start, rc, n_anchor, dtype=float)
