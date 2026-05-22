"""Fit LAMMPS pair tables with constrained multi-Gaussian potentials."""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from scipy.optimize import least_squares

from .base import BaseTableFitter, TABLE_FITTERS
from ..potentials.multi_gaussian import MultiGaussianPotential
from ..io.tables import parse_lammps_table
from .utils import (
    _init_grid, _pack_params, _unpack_params,
    make_cutoff_anchors,
    _gaussian_basis, _gaussian_basis_dr, _solve_A_with_anchors
)
from ..topology.forcefield import Forcefield
from ..topology.types import InteractionKey


@dataclass
class MultiGaussianConfig:
    """Configuration for constrained multi-Gaussian table fitting.

    Attributes
    ----------
    n_gauss : int
        Number of Gaussian components in the expansion.
    anchor_to_cutoff : bool
        Whether to add cutoff anchors that enforce ``V(rc) ~= 0`` and
        ``dV/dr|rc ~= 0``.
    n_anchor : int
        Number of synthetic anchor points near the cutoff.
    anchor_span : float
        Width of the anchoring region before the cutoff.
    weight_data : float
        Residual weight for tabulated potential values.
    weight_c0 : float
        Residual weight for cutoff value constraints ``V(anchor)``.
    weight_c1 : float
        Residual weight for cutoff slope constraints ``dV/dr(anchor)``.
    use_repulsive : bool
        Whether to reserve one component for short-range repulsion.
    repulsive_index : int or None
        Component index used for the repulsive constraint.
    repulsive_A_min : float
        Minimum amplitude in ``A >= repulsive_A_min``.
    repulsive_r0_max : float
        Maximum center location in ``r0 <= repulsive_r0_max``.
    bounds : dict
        Global pattern bounds expanded through ``Forcefield.build_bounds``.
    pair_bounds : dict
        Pair-specific pattern bounds expanded through ``Forcefield.build_bounds``.
    clamp_init_to_bounds : bool
        Whether out-of-bounds initial parameters are clipped before refinement.
    use_scipy : bool
        Whether to run nonlinear ``least_squares`` after the linear solve.
    max_nfev : int
        Maximum number of nonlinear least-squares evaluations.
    """

    # Model size.
    n_gauss: int = 16

    # Cutoff anchoring keeps generated tables smooth at rc.
    anchor_to_cutoff: bool = True
    n_anchor: int = 4
    anchor_span: float = 0.6
    weight_data: float = 1.0
    weight_c0: float = 20.0
    weight_c1: float = 10.0

    # A designated short-range component prevents attractive spikes at contact.
    use_repulsive: bool = True
    repulsive_index: Optional[int] = 0
    repulsive_A_min: float = 1e-4
    repulsive_r0_max: float = 0.0

    # Bound dictionaries use the same pattern syntax as forcefield bounds.
    bounds: Dict = field(default_factory=dict)
    pair_bounds: Dict = field(default_factory=dict)

    # Clamping allows robust fitting when constraints tighten after init.
    clamp_init_to_bounds: bool = True

    # Nonlinear refinement improves the joint fit after linear amplitude init.
    use_scipy: bool = True
    max_nfev: int = 3000


class MultiGaussianTableFitter(BaseTableFitter):
    """Fit a ``MultiGaussianPotential`` from one LAMMPS table.

    The defaults favor table stability: smooth cutoff anchoring, one
    short-range repulsive component, and bounded Gaussian widths. Advanced
    callers can override individual ``MultiGaussianConfig`` fields.

    Notes
    -----
    The fitted model is ``V(r) = sum_k A_k phi_k(r; r0_k, sigma_k)`` with
    optional anchor residuals for ``V(rc)`` and ``dV/dr|rc``.
    """
    def __init__(self, config: Optional[MultiGaussianConfig] = None, **overrides):
        """Initialize the fitter and apply config-field overrides."""
        self.cfg = (config or MultiGaussianConfig())
        # Keyword overrides mirror dataclass field names for config files.
        for k, v in overrides.items():
            if not hasattr(self.cfg, k):
                raise AttributeError(f"Unknown MultiGaussianConfig field '{k}'")
            setattr(self.cfg, k, v)

    def profile_name(self) -> str:
        """Return this fitter's registry profile name."""
        return "multigaussian"

    def fit(self, table_path: str, typ1: str, typ2: str) -> MultiGaussianPotential:
        """Fit a multi-Gaussian potential to one LAMMPS ``(r, V[, F])`` table.

        Parameters
        ----------
        table_path : str
            LAMMPS table file containing tabulated radii and potential values.
        typ1, typ2 : str
            Pair-type labels for the constructed potential.

        Returns
        -------
        MultiGaussianPotential
            The fitted potential object.

        Notes
        -----
        Cutoff anchoring adds residuals for ``V(rc) ~= 0`` and
        ``dV/dr|rc ~= 0`` at synthetic anchor points near the cutoff radius.
        Repulsive-component constraints enforce
        ``A >= repulsive_A_min`` and ``r0 <= repulsive_r0_max`` for the
        configured component.
        """
        r, V, _ = parse_lammps_table(table_path)
        cfg = self.cfg
        n_gauss = int(cfg.n_gauss)

        # ── Initial parameter guess ─────────────────────────────────────
        # The nonlinear solve starts from a bounded grid so later constraints
        # refine a physically plausible potential instead of repairing a poor
        # random initialization.

        A0, r00, s0 = _init_grid(r, V, n_gauss, cfg.bounds)
        p0 = _pack_params(A0, r00, s0)

        # Cutoff anchors add synthetic equations near rc to keep the table
        # compatible with forcefield readers that expect smooth truncation.
        anchors_r = make_cutoff_anchors(r, np.max(r), cfg.n_anchor, cfg.anchor_span) if cfg.anchor_to_cutoff else np.array([], float)

        # ── Constrained linear amplitude fit ────────────────────────────
        # With centers and widths fixed, amplitudes are linear. Solving that
        # subproblem first gives the nonlinear refinement a better basin and
        # enforces any short-range repulsive component from the start.

        A_lin, r0_lin, sigma_lin = _unpack_params(p0)
        K = n_gauss
        A_lower = np.full(K, -np.inf)
        if cfg.use_repulsive and cfg.repulsive_index is not None and 0 <= cfg.repulsive_index < K:
            A_lower[cfg.repulsive_index] = max(0.0, cfg.repulsive_A_min)

        A_lin = _solve_A_with_anchors(
            r, V, r0_lin, sigma_lin, anchors_r,
            w_data=cfg.weight_data, w_c0=cfg.weight_c0, w_c1=cfg.weight_c1,
            A_lower=A_lower, A_upper=None,
        )
        p0 = _pack_params(A_lin, r0_lin, sigma_lin)

        # The linear solve constrains amplitudes only; center bounds for the
        # designated repulsive Gaussian are projected before nonlinear fitting.
        if cfg.use_repulsive and cfg.repulsive_index is not None and 0 <= cfg.repulsive_index < K:
            A1, r01, s1 = _unpack_params(p0)
            A1[cfg.repulsive_index]  = max(A1[cfg.repulsive_index],  cfg.repulsive_A_min)
            r01[cfg.repulsive_index] = min(r01[cfg.repulsive_index], cfg.repulsive_r0_max)
            p0 = _pack_params(A1, r01, s1)

        # ── Nonlinear refinement ────────────────────────────────────────
        # This step adjusts all Gaussian parameters together while preserving
        # anchor residuals and repulsive feasibility through the configured
        # residual and bounds.

        if cfg.use_scipy:
            def resid(params):
                A, r0v, sig = _unpack_params(params)
                res_data = (_gaussian_basis(r, r0v, sig) @ A) - V
                if cfg.anchor_to_cutoff and anchors_r.size > 0:
                    Ba = _gaussian_basis(anchors_r, r0v, sig)
                    Da = _gaussian_basis_dr(anchors_r, r0v, sig)
                    # Anchor residuals represent V(anchor) and dV/dr(anchor).
                    res_c0 = Ba @ A
                    res_c1 = Da @ A
                    return np.concatenate([cfg.weight_data*res_data, cfg.weight_c0*res_c0, cfg.weight_c1*res_c1])
                return cfg.weight_data * res_data

            # Fit bounds are resolved through a temporary forcefield entry.
            tmp_pair2pot = {
                InteractionKey.pair(typ1, typ2): MultiGaussianPotential(
                    typ1, typ2, n_gauss=n_gauss,
                    cutoff=float(np.max(r)), init_params=p0
                )
            }

            # Pattern bounds must expand through Forcefield to match parameter order.
            tmp_ff = Forcefield(tmp_pair2pot)
            lb, ub = tmp_ff.build_bounds(
                global_bounds=cfg.bounds,
                pair_bounds=cfg.pair_bounds
            )

            # The generic bound expander does not know about the semantic
            # repulsive component, so tighten those two coordinates here.
            if cfg.use_repulsive and cfg.repulsive_index is not None and 0 <= cfg.repulsive_index < n_gauss:
                k = int(cfg.repulsive_index)
                lb[3*k+0] = max(lb[3*k+0], cfg.repulsive_A_min)
                ub[3*k+1] = min(ub[3*k+1], cfg.repulsive_r0_max)
            
            # Fail early or clamp before SciPy sees an infeasible initial point.
            viol = (p0 < lb) | (p0 > ub)
            if np.any(viol):
                if cfg.clamp_init_to_bounds:
                    p0 = np.clip(p0, lb, ub)
                else:
                    bad = np.where((p0 < lb) | (p0 > ub))[0]
                    raise ValueError(f"Initial guess p0 violates bounds at indices {bad}.")

            res = least_squares(resid, p0, bounds=(lb, ub), method="trf", max_nfev=cfg.max_nfev, verbose=0)
            p_opt = res.x
        else:
            p_opt = p0

        pot = MultiGaussianPotential(typ1, typ2, n_gauss=n_gauss, cutoff=float(np.max(r)), init_params=p_opt)
        return pot


# ── Table-fitter registry ────────────────────────────────────────────────
TABLE_FITTERS.register("multigaussian", lambda **kw: MultiGaussianTableFitter(**kw))
