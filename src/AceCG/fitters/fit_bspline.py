"""AceCG fitters fit bspline implementation."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import lsq_linear

from .base import BaseTableFitter, TABLE_FITTERS
from ..potentials.bspline import BSplinePotential
from ..io.tables import parse_lammps_table
from .utils import make_cutoff_anchors
from ..topology.forcefield import Forcefield
from ..topology.types import InteractionKey


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


# ── config ───────────────────────────────────────────────

@dataclass
class BSplineConfig:
    """
    Configuration for fitting a BSplinePotential (force-basis) from LAMMPS table data.

    Parameters
    ----------
    degree : int
        Spline degree k (e.g., 3 for cubic).
    n_coeffs : int
        Number of spline coefficients / basis functions.
    anchor_to_cutoff : bool
        If True, include soft anchors near cutoff to enforce F(rc)≈0 and F'(rc)≈0.
    n_anchor : int
        Number of anchor points in [cutoff - anchor_span, cutoff].
    anchor_span : float
        Span width before the cutoff (same units as r).
    weight_data : float
        Weight for table (r, F) residuals.
    weight_c0 : float
        Weight for F(anchor) residuals (target 0).
    weight_c1 : float
        Weight for dF/dr(anchor) residuals (target 0).
    bounds : dict
        Global parameter bounds pattern dict, e.g. {"c_*": (-10.0, 10.0)}.
    pair_bounds : dict
        InteractionKey-specific bounds pattern dict, e.g.
        {InteractionKey.pair("1", "1"): {"c_*": (-5, 12)}}.
    clamp_init_to_bounds : bool
        If True, clip the initial coefficient guess into [lb, ub] before solving.
    bonded : bool
        If True, construct a bonded BSplinePotential using ``u_min = 0``.
        If False, construct a nonbonded BSplinePotential using ``U(r_max) = 0``.
    """
    degree: int = 3
    n_coeffs: int = 64

    anchor_to_cutoff: bool = True
    n_anchor: int = 4
    anchor_span: float = 0.6

    weight_data: float = 1.0
    weight_c0: float = 20.0
    weight_c1: float = 10.0

    bounds: Dict = field(default_factory=dict)
    pair_bounds: Dict = field(default_factory=dict)
    clamp_init_to_bounds: bool = True
    bonded: bool = False


# ── fitter ───────────────────────────────────────────────

class BSplineTableFitter(BaseTableFitter):
    """
    Fit a BSplinePotential (force-basis) to a LAMMPS table with optional cutoff anchoring.

    The coefficients parameterize force: F(r) = B(r)^T c. The fitter solves
    B @ c ≈ F from the table's force column (or -dV/dr if force is absent).

    Notes
    -----
    - This fitter performs a *bounded linear least-squares* solve in the spline coefficients.
    - Anchors (if enabled) add extra linear rows to softly enforce F(rc)≈0 and F'(rc)≈0.
    """
    def __init__(self, config: Optional[BSplineConfig] = None, **overrides):
        self.cfg = (config or BSplineConfig())
        for k, v in overrides.items():
            if not hasattr(self.cfg, k):
                raise AttributeError(f"Unknown BSplineConfig field '{k}'")
            setattr(self.cfg, k, v)

    def profile_name(self) -> str:
        """Return this fitter's registry profile name."""
        return "bspline"

    def fit(self, table_path: str, typ1: str, typ2: str) -> BSplinePotential:
        """Fit a force-basis B-spline potential from a table file."""
        r, V, F = parse_lammps_table(table_path)
        r = np.asarray(r, dtype=float).ravel()

        # Force-basis: fit B @ c ≈ F (force), not B @ c ≈ V (energy)
        if F is not None:
            F = np.asarray(F, dtype=float).ravel()
        else:
            # Fall back to numerical -dV/dr when force column is absent
            V = np.asarray(V, dtype=float).ravel()
            F = -np.gradient(V, r)

        if r.size < 4:
            raise ValueError(f"Too few table points for bspline fit: N={r.size}")
        if np.any(~np.isfinite(r)) or np.any(~np.isfinite(F)):
            raise ValueError("Non-finite values found in table r/F.")
        if np.any(np.diff(r) <= 0):
            # spline fitting expects strictly increasing x
            raise ValueError("Table r grid must be strictly increasing (found duplicates or non-monotonic r).")

        cfg = self.cfg
        k = int(cfg.degree)
        m = int(cfg.n_coeffs)
        bonded = bool(cfg.bonded)

        rmin = float(r[0])
        rc = float(r[-1])

        # Build a temporary BSplinePotential for knots, basis matrices, and bounds
        knots = BSplinePotential.clamped_uniform_knots(rmin, rc, m, k)
        c0 = np.zeros(m, dtype=float)
        tmp_pot = BSplinePotential(
            typ1,
            typ2,
            knots=knots,
            coefficients=c0,
            degree=k,
            cutoff=rc,
            bonded=bonded,
        )

        # anchors near cutoff (optional)
        anchors_r = (
            make_cutoff_anchors(r, rc, cfg.n_anchor, cfg.anchor_span)
            if cfg.anchor_to_cutoff else np.array([], dtype=float)
        )

        # design matrices via BSplinePotential (force-basis: B @ c = F)
        # (N, m)
        B_data = tmp_pot.basis_values(r)
        # (Na, m) for F≈0
        A_c0 = tmp_pot.basis_values(anchors_r)
        # (Na, m) for F'≈0
        A_c1 = tmp_pot.basis_derivatives(anchors_r)

        # stacked weighted system
        A, b = _stack_weighted(
            (B_data, A_c0, A_c1),
            (F,      np.zeros(A_c0.shape[0]), np.zeros(A_c1.shape[0])),
            (cfg.weight_data, cfg.weight_c0, cfg.weight_c1),
        )

        # Build bounds for coefficients via generic pattern expander
        tmp_ff = Forcefield({InteractionKey.pair(typ1, typ2): tmp_pot})
        lb, ub = tmp_ff.build_bounds(
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
            bonded=bonded,
        )


# register
TABLE_FITTERS.register("bspline", lambda **kw: BSplineTableFitter(**kw))
