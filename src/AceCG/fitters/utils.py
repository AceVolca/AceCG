# AceCG/utils/ffio.py
import numpy as np
from scipy.optimize import lsq_linear
from typing import Dict, Optional


# MultiGaussian table fitting stuffs
def _gaussian_basis(r: np.ndarray, r0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Construct Gaussian basis matrix B, shape (N, K):
        B[:,k] = exp(-(r - r0_k)^2 / (2 sigma_k^2)) / (sigma_k * sqrt(2π))
    """
    r = r.reshape(-1, 1)
    r0 = r0.reshape(1, -1)
    sigma = sigma.reshape(1, -1)
    phi = np.exp(- (r - r0)**2 / (2.0 * sigma**2))
    return phi / (sigma * np.sqrt(2.0 * np.pi))


def _gaussian_basis_dr(r: np.ndarray, r0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Derivative of Gaussian basis with respect to r.
        d/dr[phi_k(r)] = [-(r - r0_k) / sigma_k^2] * phi_k(r)
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
    """
    Initial guess respecting optional bounds:
      - r0_k: equal spacing within feasible [r0_lb, r0_ub]
      - sigma_k: start from width/(2*n_gauss), then clipped to [sigma_lb, sigma_ub]
      - A_k: ridge linear fit for fixed (r0, sigma), then clipped to [A_lb, A_ub]
    """
    r = np.asarray(r, dtype=float)
    V = np.asarray(V, dtype=float)
    rmin, rmax = float(r.min()), float(r.max())
    width = rmax - rmin if rmax > rmin else 1.0

    # ---- parse bounds (allow None / +/-inf) ----
    def _pair(x, default):
        if x is None: return default
        lo, hi = x
        lo = -np.inf if lo is None else float(lo)
        hi =  np.inf if hi is None else float(hi)
        return lo, hi

    A_lb, A_ub     = _pair(bounds.get("A"),     (-np.inf, np.inf)) if bounds else (-np.inf, np.inf)
    r0_lb, r0_ub   = _pair(bounds.get("r0"),    (rmin, rmax))       if bounds else (rmin, rmax)
    sigma_lb, sigma_ub = _pair(bounds.get("sigma"), (1e-6, np.inf)) if bounds else (1e-6, np.inf)

    # make r0 feasible interval (intersect with data span for sanity)
    r0_lb_eff = max(r0_lb, rmin)
    r0_ub_eff = min(r0_ub, rmax)
    if r0_lb_eff > r0_ub_eff:
        # if user bounds exclude data range, fall back to data span
        r0_lb_eff, r0_ub_eff = rmin, rmax

    # ---- r0 initial: equally spaced within feasible interval ----
    if n_gauss == 1:
        r0_init = np.array([(r0_lb_eff + r0_ub_eff) * 0.5], dtype=float)
    else:
        # keep a small margin to avoid sitting exactly on bounds unless forced
        margin = 0.15 * max(r0_ub_eff - r0_lb_eff, 1e-12)
        a = max(r0_lb_eff, r0_lb_eff + margin)
        b = min(r0_ub_eff, r0_ub_eff - margin)
        if a >= b:  # interval too tight → just fill uniformly on [r0_lb_eff, r0_ub_eff]
            a, b = r0_lb_eff, r0_ub_eff
        r0_init = np.linspace(a, b, n_gauss)

    # ---- sigma initial: start from width/(2*n_gauss) and clip to bounds ----
    sigma_guess = max(width / (2.0 * n_gauss), 1e-3)
    sigma_init = np.full(n_gauss, sigma_guess, dtype=float)
    sigma_init = np.clip(sigma_init, sigma_lb, sigma_ub if np.isfinite(sigma_ub) else sigma_init.max())

    # ---- amplitudes via ridge LS for fixed (r0, sigma) ----
    B = _gaussian_basis(r, r0_init, sigma_init)  # (N,K)
    lam = 1e-8  # small ridge for stability
    # Solve (B^T B + lam I) A = B^T V
    A_init = np.linalg.lstsq(B.T @ B + lam * np.eye(n_gauss), B.T @ V, rcond=None)[0]

    # ---- clip A into bounds if provided ----
    if np.isfinite(A_lb) or np.isfinite(A_ub):
        A_init = np.clip(A_init, A_lb, A_ub)

    return A_init, r0_init, sigma_init


def _pack_params(A: np.ndarray, r0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    flatten (A, r0, sigma) as [A0,r0_0,sigma_0, A1,r0_1,sigma_1, ...]
    """
    n = len(A)
    out = np.empty(3*n, dtype=float)
    out[0::3] = A
    out[1::3] = r0
    out[2::3] = sigma
    return out


def _unpack_params(p: np.ndarray):
    """
    reversed _pack_params
    """
    A = p[0::3]
    r0 = p[1::3]
    sigma = p[2::3]
    return A, r0, sigma


def _solve_A_with_anchors(r, V, r0, sigma, anchors_r, w_data, w_c0, w_c1,
                          A_lower=None, A_upper=None):
    """
    Solve min || w_data*(B A - V) ||^2 + || w_c0*(Ba A) ||^2 + || w_c1*(Da A) ||^2
    with element-wise bounds on A (optional).
    """
    B  = _gaussian_basis(r, r0, sigma)
    Ba = _gaussian_basis(anchors_r, r0, sigma)
    Da = _gaussian_basis_dr(anchors_r, r0, sigma)

    M = np.vstack([w_data * B, w_c0 * Ba, w_c1 * Da])
    y = np.concatenate([w_data * V, np.zeros_like(anchors_r), np.zeros_like(anchors_r)])

    K = B.shape[1]
    lb = np.full(K, -np.inf) if A_lower is None else np.array(A_lower, dtype=float)
    ub = np.full(K,  np.inf) if A_upper is None else np.array(A_upper, dtype=float)

    # small ridge via Tikhonov can be emulated by augmenting rows
    res = lsq_linear(M, y, bounds=(lb, ub), method="trf", max_iter=5000, lsq_solver="exact")
    return res.x


def _make_cutoff_anchors(r: np.ndarray, cutoff: float, n_anchor: int, span: float) -> np.ndarray:
    """
    Generate anchor points in [cutoff - span, cutoff] (inclusive).
    If cutoff exceeds data max, use data max instead.
    """
    rc = float(cutoff) if np.isfinite(cutoff) else float(r.max())
    rmax = float(r.max())
    rc = min(rc, rmax)
    r_start = max(rc - span, r.min())
    if n_anchor <= 1 or r_start >= rc:
        return np.array([rc], dtype=float)
    return np.linspace(r_start, rc, n_anchor, dtype=float)