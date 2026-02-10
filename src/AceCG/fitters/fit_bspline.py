# AceCG/fitters/fit_bspline.py
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear

from .base import BaseTableFitter, TABLE_FITTERS
from ..potentials.bspline import BSplinePotential
from ..utils.ffio import ParseLmpTable
from .utils import _make_cutoff_anchors
from ..utils.bounds import BuildGlobalBounds


# ---------- helpers ----------

def _clamped_uniform_knots(xmin: float, xmax: float, degree: int, n_coeffs: int) -> np.ndarray:
    """
    Build an open (clamped) uniform knot vector on [xmin, xmax].

    Parameters
    ----------
    xmin, xmax : float
        Domain range.
    degree : int
        Spline degree k.
    n_coeffs : int
        Number of basis functions / coefficients (m).

    Notes
    -----
    Relationship:
        m = len(t) - k - 1
    """
    k = int(degree)
    m = int(n_coeffs)
    if m < k + 1:
        raise ValueError(f"n_coeffs must be >= degree+1 (got {m} vs {k+1})")
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        raise ValueError(f"Invalid knot domain: xmin={xmin}, xmax={xmax}")

    # interior knot count
    n_int = m - k - 1
    if n_int <= 0:
        # No interior breaks: just clamped ends
        return np.r_[np.full(k + 1, xmin), np.full(k + 1, xmax)]

    # open uniform interior knots
    interior = np.linspace(xmin, xmax, n_int + 2)[1:-1]  # drop endpoints
    return np.r_[np.full(k + 1, xmin), interior, np.full(k + 1, xmax)]


def _bspline_basis_matrix(t: np.ndarray, k: int, x: np.ndarray) -> np.ndarray:
    """
    Design matrix B with B[i, j] = B_j(x_i) for the global knot vector t and degree k.

    IMPORTANT:
    We intentionally avoid BSpline.basis_element(local_knots) because with clamped knots
    local slices can contain "internal repeated knots" for that local spline, and SciPy's
    derivative routines can fail. Building the global BSpline with unit coefficients is robust.
    """
    x = np.asarray(x, dtype=float).ravel()
    m = len(t) - k - 1
    if m <= 0:
        raise ValueError(f"Invalid (t, k): len(t)={len(t)}, k={k}")

    if x.size == 0:
        return np.zeros((0, m), dtype=float)

    B = np.empty((x.size, m), dtype=float)
    eye = np.eye(m, dtype=float)
    for j in range(m):
        spl = BSpline(t, eye[j], k, extrapolate=True)
        B[:, j] = spl(x)
    return B


def _bspline_basis_deriv_matrix(t: np.ndarray, k: int, x: np.ndarray) -> np.ndarray:
    """
    First-derivative design matrix D with D[i, j] = d/dx B_j(x_i).

    Uses global BSpline(t, e_j, k).derivative(1) for stability with clamped knots.
    """
    x = np.asarray(x, dtype=float).ravel()
    m = len(t) - k - 1
    if m <= 0:
        raise ValueError(f"Invalid (t, k): len(t)={len(t)}, k={k}")

    if x.size == 0:
        return np.zeros((0, m), dtype=float)

    D = np.empty((x.size, m), dtype=float)
    eye = np.eye(m, dtype=float)
    for j in range(m):
        spl_p = BSpline(t, eye[j], k, extrapolate=True).derivative(1)
        D[:, j] = spl_p(x)
    return D


def _stack_weighted(
    A_blocks: Tuple[np.ndarray, ...],
    b_blocks: Tuple[np.ndarray, ...],
    weights: Tuple[float, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack blocks with scalar weights: [sqrt(w_i)*A_i] and [sqrt(w_i)*b_i].
    """
    A_list, b_list = [], []
    for A, b, w in zip(A_blocks, b_blocks, weights):
        if A is None or b is None:
            continue
        if A.size == 0:
            continue
        w = float(w)
        if w <= 0.0:
            continue
        w_sqrt = w**0.5
        A_list.append(w_sqrt * A)
        b_list.append(w_sqrt * np.asarray(b, dtype=float).ravel())

    if not A_list:
        # Should not happen in normal usage (data block always present)
        return np.zeros((1, 1), dtype=float), np.zeros(1, dtype=float)

    return np.vstack(A_list), np.concatenate(b_list)


# ---------- config ----------

@dataclass
class BSplineConfig:
    """
    Configuration for fitting a BSplinePotential from LAMMPS table data.

    Parameters
    ----------
    degree : int
        Spline degree k (e.g., 3 for cubic).
    n_coeffs : int
        Number of spline coefficients / basis functions.
    anchor_to_cutoff : bool
        If True, include soft anchors near cutoff to enforce V(rc)≈0 and V'(rc)≈0.
    n_anchor : int
        Number of anchor points in [cutoff - anchor_span, cutoff].
    anchor_span : float
        Span width before the cutoff (same units as r).
    weight_data : float
        Weight for table (r, V) residuals.
    weight_c0 : float
        Weight for V(anchor) residuals (target 0).
    weight_c1 : float
        Weight for dV/dr(anchor) residuals (target 0).
    bounds : dict
        Global parameter bounds pattern dict, e.g. {"c_*": (-10.0, 10.0)}.
    pair_bounds : dict
        Pair-specific bounds pattern dict, e.g. {("1","1"): {"c_*": (-5, 12)}}.
    clamp_init_to_bounds : bool
        If True, clip the initial coefficient guess into [lb, ub] before solving.
    """
    degree: int = 3
    n_coeffs: int = 24

    anchor_to_cutoff: bool = True
    n_anchor: int = 4
    anchor_span: float = 0.6

    weight_data: float = 1.0
    weight_c0: float = 20.0
    weight_c1: float = 10.0

    bounds: Dict = field(default_factory=dict)
    pair_bounds: Dict = field(default_factory=dict)
    clamp_init_to_bounds: bool = True


# ---------- fitter ----------

class BSplineTableFitter(BaseTableFitter):
    """
    Fit a BSplinePotential to a LAMMPS table (r, V[, F]) with optional cutoff anchoring.

    Notes
    -----
    - This fitter performs a *bounded linear least-squares* solve in the spline coefficients.
    - Anchors (if enabled) add extra linear rows to softly enforce cutoff smoothness.
    """
    def __init__(self, config: Optional[BSplineConfig] = None, **overrides):
        self.cfg = (config or BSplineConfig())
        for k, v in overrides.items():
            if not hasattr(self.cfg, k):
                raise AttributeError(f"Unknown BSplineConfig field '{k}'")
            setattr(self.cfg, k, v)

    def profile_name(self) -> str:
        return "bspline"

    def fit(self, table_path: str, typ1: str, typ2: str) -> BSplinePotential:
        r, V, _ = ParseLmpTable(table_path)
        r = np.asarray(r, dtype=float).ravel()
        V = np.asarray(V, dtype=float).ravel()

        if r.size < 4:
            raise ValueError(f"Too few table points for bspline fit: N={r.size}")
        if np.any(~np.isfinite(r)) or np.any(~np.isfinite(V)):
            raise ValueError("Non-finite values found in table r/V.")
        if np.any(np.diff(r) <= 0):
            # spline fitting expects strictly increasing x
            raise ValueError("Table r grid must be strictly increasing (found duplicates or non-monotonic r).")

        cfg = self.cfg
        k = int(cfg.degree)
        m = int(cfg.n_coeffs)

        rmin = float(r[0])
        rc = float(r[-1])

        # knots and initial coefficients
        knots = _clamped_uniform_knots(rmin, rc, k, m)
        c0 = np.zeros(m, dtype=float)

        # anchors near cutoff (optional)
        anchors_r = (
            _make_cutoff_anchors(r, rc, cfg.n_anchor, cfg.anchor_span)
            if cfg.anchor_to_cutoff else np.array([], dtype=float)
        )

        # design matrices
        B_data = _bspline_basis_matrix(knots, k, r)                 # (N, m)
        A_c0 = _bspline_basis_matrix(knots, k, anchors_r)           # (Na, m) for V≈0
        A_c1 = _bspline_basis_deriv_matrix(knots, k, anchors_r)     # (Na, m) for V'≈0

        # stacked weighted system
        A, b = _stack_weighted(
            (B_data, A_c0, A_c1),
            (V,      np.zeros(A_c0.shape[0]), np.zeros(A_c1.shape[0])),
            (cfg.weight_data, cfg.weight_c0, cfg.weight_c1),
        )

        # Build bounds for coefficients via generic pattern expander
        tmp_pair2pot = {
            (typ1, typ2): BSplinePotential(
                typ1, typ2,
                knots=knots,
                coefficients=c0,
                degree=k,
                cutoff=rc,
            )
        }
        lb, ub = BuildGlobalBounds(
            tmp_pair2pot,
            global_bounds=cfg.bounds,
            pair_bounds=cfg.pair_bounds
        )

        lb = np.asarray(lb, dtype=float).ravel()
        ub = np.asarray(ub, dtype=float).ravel()
        if lb.size != m or ub.size != m:
            raise ValueError(f"Bounds size mismatch: expected {m}, got lb={lb.size}, ub={ub.size}")

        if cfg.clamp_init_to_bounds:
            c0 = np.clip(c0, lb, ub)

        # bounded linear least squares
        res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol="auto", verbose=0, max_iter=None)
        c_opt = res.x

        return BSplinePotential(
            typ1, typ2,
            knots=knots,
            coefficients=c_opt,
            degree=k,
            cutoff=rc,
        )


# register
TABLE_FITTERS.register("bspline", lambda **kw: BSplineTableFitter(**kw))
