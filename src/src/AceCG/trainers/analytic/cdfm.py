"""CDFM trainer with by-x batch aggregation inside ``step()``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..base import BaseTrainer

try:
    from typing import NotRequired, TypedDict
except ImportError:
    from typing_extensions import NotRequired, TypedDict


class CDFMBatch(TypedDict, total=False):
    """Batch dictionary consumed by :class:`CDFMTrainerAnalytic`.

    Keys
    ----
    grad_direct_by_x : array-like, shape (n_x, n_params)
        Direct force-matching gradient contribution for each conditioned
        ``x`` replica.
    grad_reinforce_by_x : array-like, shape (n_x, n_params)
        REINFORCE/covariance gradient contribution for each conditioned
        ``x`` replica. It may be all zeros in direct-only workflows.
    sse_by_x : array-like, shape (n_x,)
        Sum of squared observed-force errors for each conditioned ``x``.
    n_samples_by_x : array-like, shape (n_x,)
        Number of sampled ``z`` configurations contributing to each ``x``.
    obs_rows : int
        Number of observed force rows used when reporting RMSE.
    x_weight : array-like, optional, shape (n_x,)
        Nonnegative weights for averaging across ``x`` replicas. Weights are
        normalized internally; uniform weights are used when omitted.
    mode : {"direct", "reinforce"}
        Gradient mode requested by the workflow.
    step_index : int, optional
        Iteration index used for scalar logging.
    """
    grad_direct_by_x: Any
    grad_reinforce_by_x: Any
    sse_by_x: Any
    n_samples_by_x: Any
    obs_rows: int
    x_weight: NotRequired[Any]
    mode: str
    step_index: NotRequired[int]


class CDFMOut(TypedDict, total=False):
    """Return dictionary produced by :meth:`CDFMTrainerAnalytic.step`.

    Keys include the scalar loss, masked total gradient, separated direct and
    reinforce gradients, optional Hessian placeholder, optimizer update, and a
    ``meta`` dictionary with norms, clipping status, and guardrail diagnostics.
    """
    name: str
    loss: float
    grad: Any
    grad_direct: Any
    grad_reinforce: Any
    hessian: NotRequired[Any]
    update: Any
    meta: Dict[str, Any]


class CDFMTrainerAnalytic(BaseTrainer):
    """Analytic conditional force-matching trainer.

    CDFM receives pre-reduced statistics grouped by observed ``x`` replica,
    averages those groups with optional ``x_weight`` values, applies guardrails
    for latent-only parameters, and performs one optimizer update.

    Parameters
    ----------
    forcefield : Forcefield
        Forcefield copied and updated by the trainer.
    optimizer : BaseOptimizer
        Optimizer whose mask defines trainable global parameter coordinates.
    beta : float, optional
        Inverse temperature. Required when ``mode="reinforce"`` is used.
    logger : object, optional
        Optional scalar logger exposing ``add_scalar``.
    """

    BATCH_SCHEMA: Dict[str, Any] = {
        "grad_direct_by_x": "required np.ndarray; shape (n_x, n_params)",
        "grad_reinforce_by_x": "required np.ndarray; shape (n_x, n_params)",
        "sse_by_x": "required np.ndarray; shape (n_x,)",
        "n_samples_by_x": "required np.ndarray; shape (n_x,)",
        "obs_rows": "required int; common observed-row count for every x in the batch",
        "x_weight": "optional np.ndarray; shape (n_x,); defaults to uniform averaging",
        "mode": "required str; one of {'direct', 'reinforce'}",
        "step_index": "optional int; logging step counter; default 0",
    }

    RETURN_SCHEMA: Dict[str, Any] = {
        "name": 'str; always "CDFM"',
        "loss": "float; 0.5 * weighted mean squared error over x",
        "grad": "np.ndarray; shape (n_params,); total trainer gradient after mask/guardrail",
        "grad_direct": "np.ndarray; shape (n_params,); aggregated direct gradient",
        "grad_reinforce": "np.ndarray; shape (n_params,); aggregated reinforce gradient",
        "hessian": "None; CDFM currently runs as a first-order trainer",
        "update": "np.ndarray; optimizer update if apply_update=True else zeros_like(grad)",
        "meta": "dict; diagnostics (mode, step_index, grad norms, rmse, clipping)",
    }

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Return human-readable batch and return schemas for CDFM."""
        return {"batch": cls.BATCH_SCHEMA, "return": cls.RETURN_SCHEMA}

    def get_offsets(self) -> List[slice]:
        """Return global parameter slices for each forcefield interaction.

        Returns
        -------
        list[slice]
            One slice per interaction, ordered like the forcefield. Each slice
            selects that interaction's parameters inside the global vector.
        """
        return self.forcefield.interaction_offsets()

    @staticmethod
    def make_batch(
        *,
        grad_direct_by_x: Any,
        grad_reinforce_by_x: Any,
        sse_by_x: Any,
        n_samples_by_x: Any,
        obs_rows: int,
        x_weight: Optional[Any] = None,
        mode: str = "direct",
        step_index: int = 0,
    ) -> CDFMBatch:
        """Create a validated CDFM batch dictionary from array-like inputs.

        Parameters
        ----------
        grad_direct_by_x : array-like, shape (n_x, n_params)
            Direct gradient contribution for each conditioned ``x`` replica.
        grad_reinforce_by_x : array-like, shape (n_x, n_params)
            REINFORCE/covariance gradient contribution for each ``x``.
        sse_by_x : array-like, shape (n_x,)
            Sum of squared observed-force errors per ``x``.
        n_samples_by_x : array-like, shape (n_x,)
            Number of sampled latent configurations per ``x``.
        obs_rows : int
            Number of observed force rows used to compute RMSE diagnostics.
        x_weight : array-like, optional, shape (n_x,)
            Nonnegative replica weights. They are normalized in :meth:`step`.
        mode : {"direct", "reinforce"}, default="direct"
            CDFM gradient mode.
        step_index : int, default=0
            Iteration index used for logging.

        Returns
        -------
        CDFMBatch
            Dictionary with NumPy arrays in the expected dtypes.
        """
        batch: CDFMBatch = {
            "grad_direct_by_x": np.asarray(grad_direct_by_x, dtype=np.float64),
            "grad_reinforce_by_x": np.asarray(grad_reinforce_by_x, dtype=np.float64),
            "sse_by_x": np.asarray(sse_by_x, dtype=np.float64),
            "n_samples_by_x": np.asarray(n_samples_by_x, dtype=np.int64),
            "obs_rows": int(obs_rows),
            "mode": str(mode),
            "step_index": int(step_index),
        }
        if x_weight is not None:
            batch["x_weight"] = np.asarray(x_weight, dtype=np.float64)
        return batch

    def _optimizer_step(
        self,
        grad: np.ndarray,
        *,
        apply_update: bool,
        mask_override: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not apply_update:
            return np.zeros_like(grad)
        original_mask = None
        if mask_override is not None:
            original_mask = np.asarray(self.optimizer.mask, dtype=bool).copy()
            self.optimizer.mask = np.asarray(mask_override, dtype=bool).copy()
        try:
            if self.optimizer_accepts_hessian():
                update = self.optimizer.step(grad, hessian=None)
            else:
                update = self.optimizer.step(grad)
            self.clamp_and_update()
            return np.asarray(update, dtype=np.float64)
        finally:
            if original_mask is not None:
                self.optimizer.mask = original_mask

    def step(self, batch: Dict[str, Any], apply_update: bool = True) -> CDFMOut:
        """Aggregate a CDFM batch and optionally apply one optimizer update.

        Parameters
        ----------
        batch : CDFMBatch
            Batch containing by-``x`` direct/reinforce gradient arrays, error
            sums, sample counts, and optional weights.
        apply_update : bool, default=True
            If ``True``, call the optimizer and synchronize the forcefield.
            If ``False``, compute loss/gradients without changing state.

        Returns
        -------
        CDFMOut
            Result dictionary with ``loss``, ``grad``, separated gradient
            components, ``update``, and diagnostics under ``meta``.
        """
        grad_direct_by_x = np.asarray(batch["grad_direct_by_x"], dtype=np.float64)
        grad_reinforce_by_x = np.asarray(batch["grad_reinforce_by_x"], dtype=np.float64)
        sse_by_x = np.asarray(batch["sse_by_x"], dtype=np.float64)
        n_samples_by_x = np.asarray(batch["n_samples_by_x"], dtype=np.int64)
        obs_rows = int(batch["obs_rows"])
        mode = str(batch.get("mode", "direct")).strip().lower()
        step_index = int(batch.get("step_index", 0))

        if grad_direct_by_x.ndim != 2:
            raise ValueError("batch['grad_direct_by_x'] must be 2D")
        if grad_reinforce_by_x.shape != grad_direct_by_x.shape:
            raise ValueError("batch['grad_reinforce_by_x'] must match grad_direct_by_x")
        if sse_by_x.ndim != 1 or sse_by_x.shape[0] != grad_direct_by_x.shape[0]:
            raise ValueError("batch['sse_by_x'] must have length n_x")
        if n_samples_by_x.ndim != 1 or n_samples_by_x.shape[0] != grad_direct_by_x.shape[0]:
            raise ValueError("batch['n_samples_by_x'] must have length n_x")
        if mode not in {"direct", "reinforce"}:
            raise ValueError("mode must be 'direct' or 'reinforce'")
        if mode == "reinforce" and self.beta is None:
            raise ValueError("beta must be provided to use REINFORCE mode")

        n_x = grad_direct_by_x.shape[0]
        if n_x == 0:
            raise ValueError("CDFM batch must contain at least one x replica")

        x_weight = batch.get("x_weight")
        if x_weight is None:
            w_x = np.full(n_x, 1.0 / float(n_x), dtype=np.float64)
        else:
            w_x = np.asarray(x_weight, dtype=np.float64)
            if w_x.ndim != 1 or w_x.shape[0] != n_x:
                raise ValueError("batch['x_weight'] must have shape (n_x,)")
            if np.any(w_x < 0.0):
                raise ValueError("batch['x_weight'] must be nonnegative.")
            w_sum = float(np.sum(w_x))
            if w_sum <= 0.0:
                raise ValueError("batch['x_weight'] must have positive sum.")
            w_x = w_x / w_sum

        # Collapse the by-x task dimension into one trainer-level gradient.
        grad_direct = w_x @ grad_direct_by_x
        grad_reinforce = w_x @ grad_reinforce_by_x
        weighted_sse = float(w_x @ sse_by_x)
        loss_total = 0.5 * weighted_sse
        observed_force_rmse = float(np.sqrt(weighted_sse / float(max(obs_rows, 1))))

        direct_norm = float(np.linalg.norm(grad_direct))
        reinforce_norm_before = float(np.linalg.norm(grad_reinforce))
        cov_clip_ratio = getattr(self, "cov_clip_ratio", None)
        reinforce_clipped = False
        if (
            cov_clip_ratio is not None
            and direct_norm > 0.0
            and reinforce_norm_before > float(cov_clip_ratio) * direct_norm
        ):
            scale = float(cov_clip_ratio) * direct_norm / reinforce_norm_before
            grad_reinforce *= scale
            reinforce_clipped = True
        reinforce_norm_after = float(np.linalg.norm(grad_reinforce))

        em_guardrail = str(getattr(self, "em_guardrail", "freeze")).strip().lower()
        if em_guardrail not in {"freeze", "raise"}:
            raise ValueError("em_guardrail must be 'freeze' or 'raise'")

        # Direct mode cannot identify latent-only parameters; freeze or fail
        # before those coordinates reach the optimizer.
        latent_only_mask = (
            np.asarray(self.forcefield.virtual_mask, dtype=bool)
            & np.asarray(self.optimizer.mask, dtype=bool)
        )
        guardrail_action = "none"
        if mode == "direct" and np.any(latent_only_mask):
            if em_guardrail == "raise":
                indices = np.flatnonzero(latent_only_mask).tolist()
                raise ValueError(
                    "EM-only direct mode cannot update latent-only parameter indices "
                    f"{indices}"
                )
            guardrail_action = "freeze"

        grad = grad_direct + grad_reinforce
        if guardrail_action == "freeze":
            grad[latent_only_mask] = 0.0
            grad_direct[latent_only_mask] = 0.0

        # Apply the optimizer mask last so all returned gradient components
        # match the trainable-coordinate convention used elsewhere in AceCG.
        mask = np.asarray(self.optimizer.mask, dtype=bool)
        grad[~mask] = 0.0
        grad_direct[~mask] = 0.0
        grad_reinforce[~mask] = 0.0

        step_mask = mask.copy()
        if guardrail_action == "freeze":
            step_mask[latent_only_mask] = False
        update = self._optimizer_step(grad, apply_update=apply_update, mask_override=step_mask)

        if self.logger is not None:
            self.logger.add_scalar("CDFM/loss", float(loss_total), step_index)
            self.logger.add_scalar("CDFM/observed_force_rmse", observed_force_rmse, step_index)
            self.logger.add_scalar("CDFM/grad_norm", float(np.linalg.norm(grad)), step_index)
            self.logger.add_scalar("CDFM/direct_grad_norm", direct_norm, step_index)
            self.logger.add_scalar("CDFM/reinforce_grad_norm", reinforce_norm_after, step_index)
            self.logger.add_scalar("CDFM/update_norm", float(np.linalg.norm(update)), step_index)

        return {
            "name": "CDFM",
            "loss": float(loss_total),
            "grad": grad,
            "grad_direct": grad_direct,
            "grad_reinforce": grad_reinforce,
            "hessian": None,
            "update": update,
            "meta": {
                "mode": mode,
                "step_index": step_index,
                "n_x": n_x,
                "n_samples_by_x": n_samples_by_x.tolist(),
                "obs_rows": obs_rows,
                "observed_force_rmse": observed_force_rmse,
                "grad_norm": float(np.linalg.norm(grad)),
                "direct_grad_norm": direct_norm,
                "reinforce_grad_norm_before_clip": reinforce_norm_before,
                "reinforce_grad_norm": reinforce_norm_after,
                "reinforce_clipped": reinforce_clipped,
                "update_norm": float(np.linalg.norm(update)),
                "guardrail_action": guardrail_action,
            },
        }
