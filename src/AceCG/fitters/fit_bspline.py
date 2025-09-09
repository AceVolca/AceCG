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
    Build an open (clamped) uniform knot vector on [xmin, xmax] for a spline of:
      - degree = k
      - number of basis functions = n_coeffs
    Relationship: n_coeffs = len(t) - k - 1
    """
    k = int(degree)
    m = int(n_coeffs)
    if m < k + 1:
        raise ValueError(f"n_coeffs must be >= degree+1 (got {m} vs {k+1})")
    # interior knot count:
    n_int = m - k - 1
    if n_int <= 0:
        # piecewise polynomial without interior breaks -> just clamped at ends
        t = np.r_[np.full(k+1, xmin), np.full(k+1, xmax)]
        return t

    # uniform interior knots (open uniform spline)
    interior = np.linspace(xmin, xmax, n_int + 2)[1:-1]  # drop endpoints
    t = np.r_[np.full(k+1, xmin), interior, np.full(k+1, xmax)]
    return t


def _bspline_basis_matrix(t: np.ndarray, k: int, x: np.ndarray) -> np.ndarray:
    """
    Design matrix B with B[i,j] = B_j(x_i) where B_j is the j-th B-spline basis
    for the clamped knot vector t of degree k.
    """
    x = np.asarray(x, dtype=float).ravel()
    m = len(t) - k - 1
    B = np.empty((x.size, m), dtype=float)
    for j in range(m):
        # Each basis element uses a local knot subsequence of length k+2
        bj = BSpline.basis_element(t[j:j + k + 2], extrapolate=True)
        B[:, j] = bj(x)
    return B


def _bspline_basis_deriv_matrix(t: np.ndarray, k: int, x: np.ndarray) -> np.ndarray:
    """
    First-derivative design matrix D with D[i,j] = d/dx B_j(x_i).
    """
    x = np.asarray(x, dtype=float).ravel()
    m = len(t) - k - 1
    D = np.empty((x.size, m), dtype=float)
    for j in range(m):
        bjp = BSpline.basis_element(t[j:j + k + 2], extrapolate=True).derivative(1)
        D[:, j] = bjp(x)
    return D


def _stack_weighted(A_blocks: Tuple[np.ndarray, ...], b_blocks: Tuple[np.ndarray, ...],
                    weights: Tuple[float, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack blocks with scalar weights: [sqrt(w_i)*A_i] and [sqrt(w_i)*b_i].
    """
    A_list, b_list = [], []
    for A, b, w in zip(A_blocks, b_blocks, weights):
        if A.size == 0:
            continue
        w_sqrt = float(w) ** 0.5
        A_list.append(w_sqrt * A)
        b_list.append(w_sqrt * b)
    if not A_list:
        # fall back to zeros to avoid lsq_linear complaining; but practically shouldn't happen
        return np.zeros((1, 1)), np.zeros(1)
    return np.vstack(A_list), np.concatenate(b_list)


# ---------- config ----------

@dataclass
class BSplineConfig:
    """
    Configuration for fitting a BSplinePotential from table data.

    Parameters
    ----------
    degree : int
        Spline degree k (e.g., 3 for cubic).
    n_coeffs : int
        Number of spline coefficients (basis functions).
    anchor_to_cutoff : bool
        If True, include anchor rows to softly enforce V(rc)=0 and dV/dr|rc=0.
    n_anchor : int
        Number of evenly spaced anchors in [cutoff - anchor_span, cutoff].
    anchor_span : float
        Span width (absolute distance units) before the cutoff.
    weight_data : float
        Weight for table (r, V) residuals.
    weight_c0 : float
        Weight for V(anchor) residuals (≈0).
    weight_c1 : float
        Weight for dV/dr(anchor) residuals (≈0).
    bounds : dict
        Global parameter bounds pattern dict, e.g. {"c_*": (-10.0, 10.0)}.
    pair_bounds : dict
        Pair-specific bounds pattern dict, e.g. {("1","1"): {"c_*": (-5, 12)}}.
    clamp_init_to_bounds : bool
        If True, clip initial coefficients to [lb, ub] before solving.
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

        cfg = self.cfg
        k = int(cfg.degree)
        m = int(cfg.n_coeffs)

        rmin = float(np.min(r))
        rc = float(np.max(r))

        # knots and initial coefficients (zero init is fine; we'll solve linearly)
        knots = _clamped_uniform_knots(rmin, rc, k, m)
        c0 = np.zeros(m, dtype=float)

        # build anchors
        anchors_r = _make_cutoff_anchors(r, rc, cfg.n_anchor, cfg.anchor_span) if cfg.anchor_to_cutoff else np.array([], float)

        # design matrices
        B_data = _bspline_basis_matrix(knots, k, r)               # (N, m)
        A_c0 = _bspline_basis_matrix(knots, k, anchors_r)         # (Na, m) for V≈0
        A_c1 = _bspline_basis_deriv_matrix(knots, k, anchors_r)   # (Na, m) for V'≈0

        # stack weighted system:  [sqrt(wd)*B; sqrt(w0)*A_c0; sqrt(w1)*A_c1] c ≈ [sqrt(wd)*V; 0; 0]
        A, b = _stack_weighted(
            (B_data, A_c0, A_c1),
            (V,      np.zeros(A_c0.shape[0]), np.zeros(A_c1.shape[0])),
            (cfg.weight_data, cfg.weight_c0, cfg.weight_c1)
        )

        # Build bounds for coefficients via the generic pattern expander
        tmp_pair2pot = {
            (typ1, typ2): BSplinePotential(typ1, typ2, knots=knots, coefficients=c0, degree=k, cutoff=rc)
        }
        lb, ub = BuildGlobalBounds(
            tmp_pair2pot,
            global_bounds=cfg.bounds,
            pair_bounds=cfg.pair_bounds
        )  # arrays of length m (parameter order: c0..c{m-1})

        # bounded linear least squares in coefficient space
        # (linear model, so no nonlinear refinement needed)
        res = lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto', verbose=0, max_iter=None)
        c_opt = res.x

        pot = BSplinePotential(typ1, typ2, knots=knots, coefficients=c_opt, degree=k, cutoff=rc)
        return pot


# register
TABLE_FITTERS.register("bspline", lambda **kw: BSplineTableFitter(**kw))