# AceCG/fitters/multigauss.py
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from scipy.optimize import least_squares, lsq_linear

from .base import BaseTableFitter, TABLE_FITTERS
from ..potentials.multi_gaussian import MultiGaussianPotential
from ..utils.ffio import ParseLmpTable
from .utils import ( # fitting utils
    _init_grid, _pack_params, _unpack_params,
    _make_cutoff_anchors,
    _gaussian_basis, _gaussian_basis_dr,
)


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
			Defaults: A free, r0 free, sigma ∈ [0.1, ∞).
        use_scipy : bool
			Whether to run nonlinear least_squares refinement after
			the initial linear solve.
		max_nfev: int : int
			maximum number of iterations for scipy optimization
		random_state : int
			Random seed for reproducibility.
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
    bounds: Dict = field(default_factory=lambda: {"sigma": (0.1, np.inf)})
    # nonlinear refine
    use_scipy: bool = True
    max_nfev: int = 3000
    # seeding
    random_state: int = 0


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
        rng = np.random.default_rng(cfg.random_state)
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
        B  = _gaussian_basis(r, r0_lin, sigma_lin)
        Ba = _gaussian_basis(anchors_r, r0_lin, sigma_lin)
        Da = _gaussian_basis_dr(anchors_r, r0_lin, sigma_lin)
        M = np.vstack([cfg.weight_data * B, cfg.weight_c0 * Ba, cfg.weight_c1 * Da])
        y = np.concatenate([cfg.weight_data * V, np.zeros_like(anchors_r), np.zeros_like(anchors_r)])
        res_lin = lsq_linear(M, y, bounds=(A_lower, np.full(K, np.inf)), method="trf", lsq_solver="exact", max_iter=5000)
        A_lin = res_lin.x
        p0 = _pack_params(A_lin, r0_lin, sigma_lin)

        # project repulsive init onto feasible region
        if cfg.use_repulsive and cfg.repulsive_index is not None and 0 <= cfg.repulsive_index < K:
            A1, r01, s1 = _unpack_params(p0)
            A1[cfg.repulsive_index]  = max(A1[cfg.repulsive_index],  cfg.repulsive_A_min)
            r01[cfg.repulsive_index] = min(r01[cfg.repulsive_index], cfg.repulsive_r0_max)
            p0 = _pack_params(A1, r01, s1)

        # ----- nonlinear refine -----
        if cfg.use_scipy:
            def model(params):
                A, r0v, sig = _unpack_params(params)
                return _gaussian_basis(r, r0v, sig) @ A
            def model_dr(params):
                A, r0v, sig = _unpack_params(params)
                return _gaussian_basis_dr(r, r0v, sig) @ A
            def resid(params):
                res_data = model(params) - V
                if cfg.anchor_to_cutoff and anchors_r.size > 0:
                    res_c0 = (_gaussian_basis(anchors_r, _unpack_params(params)[1], _unpack_params(params)[2]) @ _unpack_params(params)[0])
                    res_c1 = (_gaussian_basis_dr(anchors_r, _unpack_params(params)[1], _unpack_params(params)[2]) @ _unpack_params(params)[0])
                    return np.concatenate([cfg.weight_data*res_data, cfg.weight_c0*res_c0, cfg.weight_c1*res_c1])
                return cfg.weight_data * res_data

            def _pair(v): return (-np.inf, np.inf) if v is None else v
            A_lb, A_ub = _pair(cfg.bounds.get("A")) if "A" in cfg.bounds else (-np.inf, np.inf)
            sig_lb, sig_ub = _pair(cfg.bounds.get("sigma")) if "sigma" in cfg.bounds else (1e-6, np.inf)
            # default r0 bounds: allow ≤ 0 to keep repulsive feasible
            if "r0" in cfg.bounds: r0_lb, r0_ub = _pair(cfg.bounds["r0"])
            else:                  r0_lb, r0_ub = (-np.inf, r.max())

            lb = np.empty_like(p0);  ub = np.empty_like(p0)
            lb[0::3] = A_lb;   ub[0::3] = A_ub
            lb[1::3] = r0_lb;  ub[1::3] = r0_ub
            lb[2::3] = sig_lb; ub[2::3] = sig_ub

            # tighten for the repulsive component
            if cfg.use_repulsive and cfg.repulsive_index is not None and 0 <= cfg.repulsive_index < n_gauss:
                k = int(cfg.repulsive_index)
                lb[3*k+0] = max(lb[3*k+0], cfg.repulsive_A_min)
                ub[3*k+1] = min(ub[3*k+1], cfg.repulsive_r0_max)

            res = least_squares(resid, p0, bounds=(lb, ub), method="trf", max_nfev=cfg.max_nfev, verbose=0)
            p_opt = res.x
        else:
            p_opt = p0

        pot = MultiGaussianPotential(typ1, typ2, n_gauss=n_gauss, cutoff=float(np.max(r)), init_params=p_opt)
        return pot

# register
TABLE_FITTERS.register("multigaussian", lambda **kw: MultiGaussianTableFitter(**kw))
