"""Closed-form force-matching solver.

The solver consumes normalized FM sufficient statistics from the one-pass
compute engine, typically read back from a post artifact such as
``fm_batch.pkl``. The runtime contract is that compute preserves the full
parameter system; masking remains a forcefield concern inside the solver.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg

from .base import BaseSolver
from ..topology.forcefield import Forcefield


_EPS = 1.0e-30


def _vec(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _loss(JtJ: np.ndarray, Jty: np.ndarray, y_sumsq: float, theta: np.ndarray) -> float:
    return 0.5 * float(theta @ JtJ @ theta - 2.0 * theta @ Jty + y_sumsq)


def _scaled_lstsq(JtJ: np.ndarray, Jty: np.ndarray) -> np.ndarray:
    if JtJ.size == 0:
        return np.zeros(JtJ.shape[0], dtype=np.float64)
    scale = np.sqrt(np.where(np.abs(np.diag(JtJ)) > _EPS, np.abs(np.diag(JtJ)), 1.0))
    JtJ_scaled = JtJ / scale[:, None] / scale[None, :]
    Jty_scaled = Jty / scale
    theta_scaled, *_ = linalg.lstsq(
        JtJ_scaled,
        Jty_scaled,
        cond=None,
        check_finite=False,
        lapack_driver="gelsy",
    )
    return _vec(theta_scaled) / scale


def _cho_solve(JtJ: np.ndarray, Jty: np.ndarray) -> np.ndarray:
    factor, lower = linalg.cho_factor(JtJ, lower=True, check_finite=False)
    return _vec(linalg.cho_solve((factor, lower), Jty, check_finite=False))


def _chol_inverse_diag(factor: np.ndarray, eye: np.ndarray) -> np.ndarray:
    inv_factor = linalg.solve_triangular(factor, eye, lower=True, check_finite=False)
    return np.sum(inv_factor * inv_factor, axis=0)


class FMMatrixSolver(BaseSolver):
    """One-shot OLS, ridge, or diagonal-Bayesian solver for FM statistics.

    Parameters
    ----------
    forcefield : Forcefield
        Forcefield copied into the solver. Its parameter mask selects active
        degrees of freedom.
    mode : {"ols", "ridge", "bayesian"}, default="ols"
        Linear solve strategy. ``"ols"`` uses a scaled least-squares solve,
        ``"ridge"`` adds scalar Tikhonov regularization, and ``"bayesian"``
        iteratively estimates diagonal prior precisions.
    ridge_alpha : float, default=0.0
        Ridge penalty added to active diagonal entries when ``mode="ridge"``.
    bayesian_tol : float, default=1.0e-6
        Relative convergence tolerance for Bayesian alpha/beta updates.
    bayesian_min_iter : int, default=10
        Minimum Bayesian iterations before convergence may be accepted.
    bayesian_max_iter : int, default=100
        Maximum Bayesian iterations.
    bayesian_alpha_init : float, optional
        Initial prior precision. If omitted, an automatic scale is used.
    bayesian_beta_init : float, optional
        Initial noise precision. If omitted, it is estimated from residuals.
    logger : object, optional
        Optional scalar logger exposing ``add_scalar``.
    """

    BATCH_SCHEMA = {
        "JtJ": "(p, p) normalized force-matching normal matrix",
        "Jty": "(p,) normalized force-matching right-hand side",
        "y_sumsq": "normalized target-force norm square",
        "nframe": "number of contributing frames",
        "weight_sum": "pre-normalization total frame weight",
        "n_atoms_obs": "observed atoms per frame",
    }

    RETURN_SCHEMA = {
        "params": "(p,) solved full parameter vector",
        "loss": "normalized force-matching loss at the solution",
        "mode": "solver mode",
        "meta": "solver diagnostics",
    }

    def __init__(
        self,
        forcefield: Forcefield,
        *,
        mode: str = "ols",
        ridge_alpha: float = 0.0,
        bayesian_tol: float = 1.0e-6,
        bayesian_min_iter: int = 10,
        bayesian_max_iter: int = 100,
        bayesian_alpha_init: float | None = None,
        bayesian_beta_init: float | None = None,
        logger=None,
    ):
        """Initialize the FM matrix solver.

        Parameters are the same as documented on :class:`FMMatrixSolver`.
        """
        super().__init__(forcefield, logger=logger)
        self.mode = str(mode).strip().lower()
        self.ridge_alpha = float(ridge_alpha)
        self.bayesian_tol = float(bayesian_tol)
        self.bayesian_min_iter = int(bayesian_min_iter)
        self.bayesian_max_iter = int(bayesian_max_iter)
        self.bayesian_alpha_init = bayesian_alpha_init
        self.bayesian_beta_init = bayesian_beta_init

        if self.mode not in {"ols", "ridge", "bayesian"}:
            raise ValueError("mode must be 'ols', 'ridge', or 'bayesian'")
        if self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative")
        if self.bayesian_min_iter < 1 or self.bayesian_max_iter < self.bayesian_min_iter:
            raise ValueError("invalid Bayesian iteration limits")
        if self.bayesian_tol <= 0.0:
            raise ValueError("bayesian_tol must be positive")

    def _active_system(
        self,
        JtJ: np.ndarray,
        Jty: np.ndarray,
        theta0: np.ndarray,
        active_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        active = np.flatnonzero(active_mask)
        frozen = np.flatnonzero(~active_mask)
        JtJ_aa = JtJ[np.ix_(active, active)]
        Jty_a = Jty[active].copy()
        if frozen.size:
            Jty_a -= JtJ[np.ix_(active, frozen)] @ theta0[frozen]
        return active, frozen, JtJ_aa, Jty_a

    def _solve_bayesian(
        self,
        JtJ: np.ndarray,
        Jty: np.ndarray,
        y_sumsq: float,
        theta0: np.ndarray,
        active_mask: np.ndarray,
        weight_sum: float,
        n_atoms_obs: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        active, frozen, JtJ_aa, Jty_a = self._active_system(JtJ, Jty, theta0, active_mask)
        theta_f = theta0[frozen]

        JtJ_raw = weight_sum * JtJ_aa
        Jty_raw = weight_sum * Jty_a
        y_sumsq_raw = weight_sum * float(y_sumsq)
        if frozen.size:
            Jty_f = weight_sum * Jty[frozen]
            JtJ_ff = weight_sum * JtJ[np.ix_(frozen, frozen)]
            y_sumsq_raw += float(theta_f @ JtJ_ff @ theta_f - 2.0 * theta_f @ Jty_f)

        theta = _scaled_lstsq(JtJ_aa, Jty_a)
        if self.bayesian_alpha_init is None:
            alpha = np.full(
                active.size,
                max(float(active.size), 1.0) / max(float(theta @ theta), _EPS),
                dtype=np.float64,
            )
        else:
            alpha = np.full(active.size, float(self.bayesian_alpha_init), dtype=np.float64)

        rss = max(_loss(JtJ_raw, Jty_raw, y_sumsq_raw, theta) * 2.0, _EPS)
        if self.bayesian_beta_init is None:
            n_obs = 3.0 * float(n_atoms_obs) * weight_sum
            beta = max(n_obs, 1.0) / rss
        else:
            beta = float(self.bayesian_beta_init)

        eye = np.eye(active.size, dtype=np.float64)
        converged = False
        for iteration in range(1, self.bayesian_max_iter + 1):
            system = beta * JtJ_raw.copy()
            system[np.diag_indices_from(system)] += alpha
            factor, lower = linalg.cho_factor(system, lower=True, check_finite=False)
            if not lower:
                raise RuntimeError("expected lower-triangular Cholesky factor")
            theta_new = _vec(linalg.cho_solve((factor, lower), beta * Jty_raw, check_finite=False))
            gamma = np.clip(1.0 - alpha * _chol_inverse_diag(np.tril(factor), eye), 0.0, None)
            rss = max(_loss(JtJ_raw, Jty_raw, y_sumsq_raw, theta_new) * 2.0, _EPS)
            alpha_new = np.maximum(gamma / np.maximum(theta_new * theta_new, _EPS), _EPS)
            beta_new = max(3.0 * float(n_atoms_obs) * weight_sum - float(gamma.sum()), _EPS) / rss

            rel_alpha = np.max(np.abs(alpha_new - alpha) / np.maximum(np.abs(alpha), _EPS))
            rel_beta = abs(beta_new - beta) / max(abs(beta), _EPS)
            theta = theta_new
            alpha = alpha_new
            beta = beta_new

            if iteration >= self.bayesian_min_iter and max(rel_alpha, rel_beta) < self.bayesian_tol:
                converged = True
                break

        return theta, {
            "bayesian_iterations": iteration,
            "bayesian_converged": converged,
            "bayesian_alpha": alpha.copy(),
            "bayesian_beta": float(beta),
        }

    def solve(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Solve one normalized FM normal-equation batch.

        Parameters
        ----------
        batch : dict
            Batch with ``JtJ`` shape ``(n_params, n_params)``, ``Jty`` shape
            ``(n_params,)``, scalar ``y_sumsq``, integer ``nframe``, scalar
            ``weight_sum``, and integer ``n_atoms_obs``. ``step_index`` is
            optional and only used for logging.

        Returns
        -------
        dict
            ``params`` contains the solved full parameter vector, ``loss`` is
            the normalized FM loss at that vector, ``mode`` echoes the solver
            mode, and ``meta`` stores diagnostics.
        """
        JtJ = np.asarray(batch["JtJ"], dtype=np.float64)
        Jty = _vec(batch["Jty"])
        y_sumsq = float(batch["y_sumsq"])
        nframe = int(batch["nframe"])
        weight_sum = float(batch["weight_sum"])
        n_atoms_obs = int(batch["n_atoms_obs"])

        theta0 = self.get_params()
        n_params = theta0.size
        if JtJ.shape != (n_params, n_params):
            raise ValueError(f"JtJ shape must be {(n_params, n_params)}, got {JtJ.shape}")
        if Jty.shape != (n_params,):
            raise ValueError(f"Jty shape must be {(n_params,)}, got {Jty.shape}")
        if nframe < 0 or weight_sum < 0.0 or n_atoms_obs < 0:
            raise ValueError("nframe, weight_sum, and n_atoms_obs must be non-negative")

        active_mask = np.asarray(self.forcefield.param_mask, dtype=bool).reshape(-1)
        if active_mask.shape != (n_params,):
            raise ValueError(f"param_mask shape must be {(n_params,)}, got {active_mask.shape}")

        theta = theta0.copy()
        meta = {
            "nframe": nframe,
            "weight_sum": weight_sum,
            "n_atoms_obs": n_atoms_obs,
            "active_n_params": int(active_mask.sum()),
        }
        if "step_index" in batch:
            meta["step_index"] = int(batch["step_index"])

        if nframe > 0 and weight_sum > 0.0 and np.any(active_mask):
            active, _, JtJ_aa, Jty_a = self._active_system(JtJ, Jty, theta0, active_mask)
            if self.mode == "ols":
                theta[active] = _scaled_lstsq(JtJ_aa, Jty_a)
            elif self.mode == "ridge":
                if self.ridge_alpha == 0.0:
                    theta[active] = _scaled_lstsq(JtJ_aa, Jty_a)
                else:
                    system = JtJ_aa.copy()
                    system[np.diag_indices_from(system)] += self.ridge_alpha
                    theta[active] = _cho_solve(system, Jty_a)
                meta["ridge_alpha"] = self.ridge_alpha
            else:
                if n_atoms_obs <= 0:
                    raise ValueError("Bayesian mode requires positive n_atoms_obs")
                theta[active], bayes_meta = self._solve_bayesian(
                    JtJ,
                    Jty,
                    y_sumsq,
                    theta0,
                    active_mask,
                    weight_sum,
                    n_atoms_obs,
                )
                meta.update(bayes_meta)

        loss = _loss(JtJ, Jty, y_sumsq, theta)
        self.update_forcefield(theta)
        if self.logger is not None:
            self.logger.add_scalar("FM/solver_loss", loss, int(meta.get("step_index", 0)))

        return {
            "params": theta.copy(),
            "loss": loss,
            "mode": self.mode,
            "meta": meta,
        }
