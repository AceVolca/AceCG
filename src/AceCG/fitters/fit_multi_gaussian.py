# AceCG/fitters/multigauss.py
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from scipy.optimize import least_squares

from .base import BaseTableFitter, TABLE_FITTERS
from ..potentials.multi_gaussian import MultiGaussianPotential
from ..utils.ffio import ParseLmpTable
from .utils import ( # fitting utils
    _init_grid, _pack_params, _unpack_params,
    _make_cutoff_anchors,
    _gaussian_basis, _gaussian_basis_dr, _solve_A_with_anchors
)
from ..utils.bounds import BuildGlobalBounds


@dataclass
class MultiGaussianConfig:
    """
    Parameters
		----------
		n_gauss : int
			Number of Gaussian components in the expansion.
		anchor_to_cutoff : bool
			If True, include cutoff anchors to enforce V(rc)=0, dV/dr|rc=0.
		n_anchor : int
			Number of anchor points in [cutoff - anchor_span, cutoff].
		anchor_span : float
			Width of the anchoring region before cutoff.
		weight_data : float
			Weight factor for data residuals.
		weight_c0 : float
			Weight factor for cutoff value (V=0) constraints.
		weight_c1 : float
			Weight factor for cutoff slope (dV/dr=0) constraints.
        use_repulsive : bool
        	If True, fix a Gaussian component to be repulsive.
        repulsive_index : int or None
			Index of the Gaussian component to force as repulsive.
			If None, no repulsive constraint is applied.
		repulsive_A_min : float
			Minimum amplitude for the repulsive component (A >= this).
		repulsive_r0_max : float
			Maximum center for the repulsive component (r0 <= this).
			Typically set ≤ 0 to place repulsion at/inside contact.
		bounds : dict
			Parameter bounds, e.g. {"A": (amin, amax),
											"r0": (rmin, rmax),
											"sigma": (smin, smax)}.
		pair_bounds : dict
			Pair specific parameter bounds, e.g. {
                                            ("1", "1"): {"A_*": (amin, amax), "r0": (rmin, rmax), "sigma": (smin, smax)},
                                            }.
        clamp_init_to_bounds : bool = True
            Clamp to [lb, ub] if the initial guess is out of bounds
        use_scipy : bool
			Whether to run nonlinear least_squares refinement after
			the initial linear solve.
		max_nfev: int : int
			maximum number of iterations for scipy optimization
    """
    # model size
    n_gauss: int = 16
    # cutoff anchoring
    anchor_to_cutoff: bool = True
    n_anchor: int = 4
    anchor_span: float = 0.6
    weight_data: float = 1.0
    weight_c0: float = 20.0
    weight_c1: float = 10.0
    # repulsive component (index 0)
    use_repulsive: bool = True
    repulsive_index: Optional[int] = 0
    repulsive_A_min: float = 1e-4
    repulsive_r0_max: float = 0.0
    # bounds
    bounds: Dict = field(default_factory=dict)       # global bounds (pattern dict)
    pair_bounds: Dict = field(default_factory=dict)  # pair-specific bounds
    clamp_init_to_bounds: bool = True   # clamp p0 to [lb,ub]
    # nonlinear refine
    use_scipy: bool = True
    max_nfev: int = 3000


class MultiGaussianTableFitter(BaseTableFitter):
    """
    Fit a MultiGaussianPotential from a LAMMPS table file with robust defaults:
      - smooth anchoring to 0 at cutoff (value & slope)
      - one repulsive Gaussian component (A>0, r0<=0) by default
      - mild lower bound on sigma to avoid ultra-sharp spikes
    Most users don't need to tune anything; advanced users can pass overrides.
    """
    def __init__(self, config: Optional[MultiGaussianConfig] = None, **overrides):
        self.cfg = (config or MultiGaussianConfig())
        # allow dict-like overrides: MultiGaussianTableFitter(n_gauss=12, repulsive_index=None, ...)
        for k, v in overrides.items():
            if not hasattr(self.cfg, k):
                raise AttributeError(f"Unknown MultiGaussConfig field '{k}'")
            setattr(self.cfg, k, v)

    def profile_name(self) -> str:
        return "multigauss"

    def fit(self, table_path: str, typ1: str, typ2: str) -> MultiGaussianPotential:
        """
		Fit a MultiGaussianPotential to LAMMPS table data (r, V[, F]) with
		optional cutoff anchoring and repulsive component constraints.

		Features
		--------
		• Cutoff anchoring: Enforces V(rc) ≈ 0 and dV/dr|rc ≈ 0 using
			additional "anchor points" near the cutoff radius.
		• Repulsive Gaussian constraint: Designate one Gaussian
			component as a short-range repulsive term with:
				A >= repulsive_A_min (positive amplitude)
				r0 <= repulsive_r0_max (usually ≤ 0, i.e. at/inside contact)

		Returns
		-------
		pot : MultiGaussianPotential
			The fitted potential object.
		"""
        r, V, _ = ParseLmpTable(table_path)
        cfg = self.cfg
        n_gauss = int(cfg.n_gauss)

        # ----- init -----
        A0, r00, s0 = _init_grid(r, V, n_gauss, cfg.bounds)
        p0 = _pack_params(A0, r00, s0)

        # cutoff anchors
        anchors_r = _make_cutoff_anchors(r, np.max(r), cfg.n_anchor, cfg.anchor_span) if cfg.anchor_to_cutoff else np.array([], float)

        # ----- linear A with anchors + repulsive A>=0 (if used) -----
        A_lin, r0_lin, sigma_lin = _unpack_params(p0)
        # bounds on amplitudes for lsq_linear (only A)
        K = n_gauss
        A_lower = np.full(K, -np.inf)
        if cfg.use_repulsive and cfg.repulsive_index is not None and 0 <= cfg.repulsive_index < K:
            A_lower[cfg.repulsive_index] = max(0.0, cfg.repulsive_A_min)

        # build augmented linear system M A ≈ y
        A_lin = _solve_A_with_anchors(
            r, V, r0_lin, sigma_lin, anchors_r,
            w_data=cfg.weight_data, w_c0=cfg.weight_c0, w_c1=cfg.weight_c1,
            A_lower=A_lower, A_upper=None,
        )
        p0 = _pack_params(A_lin, r0_lin, sigma_lin)

        # project repulsive init onto feasible region
        if cfg.use_repulsive and cfg.repulsive_index is not None and 0 <= cfg.repulsive_index < K:
            A1, r01, s1 = _unpack_params(p0)
            A1[cfg.repulsive_index]  = max(A1[cfg.repulsive_index],  cfg.repulsive_A_min)
            r01[cfg.repulsive_index] = min(r01[cfg.repulsive_index], cfg.repulsive_r0_max)
            p0 = _pack_params(A1, r01, s1)

        # ----- nonlinear refine -----
        if cfg.use_scipy:
            def resid(params):
                A, r0v, sig = _unpack_params(params)
                res_data = (_gaussian_basis(r, r0v, sig) @ A) - V
                if cfg.anchor_to_cutoff and anchors_r.size > 0:
                    Ba = _gaussian_basis(anchors_r, r0v, sig)
                    Da = _gaussian_basis_dr(anchors_r, r0v, sig)
                    res_c0 = Ba @ A   # V(anchors)
                    res_c1 = Da @ A   # dV/dr(anchors)
                    return np.concatenate([cfg.weight_data*res_data, cfg.weight_c0*res_c0, cfg.weight_c1*res_c1])
                return cfg.weight_data * res_data
            
            # bounds
            # build temporary pair2potential with just this pair
            tmp_pair2pot = {
                (typ1, typ2): MultiGaussianPotential(
                    typ1, typ2, n_gauss=n_gauss,
                    cutoff=float(np.max(r)), init_params=p0
                )
            }

            # use BuildGlobalBounds to expand pattern bounds to arrays
            lb, ub = BuildGlobalBounds(
                tmp_pair2pot,
                global_bounds=cfg.bounds,
                pair_bounds=cfg.pair_bounds
            )

            # tighten for the repulsive component
            if cfg.use_repulsive and cfg.repulsive_index is not None and 0 <= cfg.repulsive_index < n_gauss:
                k = int(cfg.repulsive_index)
                lb[3*k+0] = max(lb[3*k+0], cfg.repulsive_A_min)
                ub[3*k+1] = min(ub[3*k+1], cfg.repulsive_r0_max)
            
            # feasibility check
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

# register
TABLE_FITTERS.register("multigaussian", lambda **kw: MultiGaussianTableFitter(**kw))
