# AceCG/trainers/analytic/fm.py
"""Force-matching trainer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base import BaseTrainer
from ...potentials.base import BasePotential
from ...topology.types import InteractionKey

try:
    from typing import TypedDict, NotRequired
except ImportError:
    from typing_extensions import TypedDict, NotRequired


# -----------------------------------------------------------------------------
# TypedDict schema
# -----------------------------------------------------------------------------

class FMBatch(TypedDict, total=False):
    """Batch schema for FMTrainerAnalytic.step().

    Pre-accumulated normalized normal-equation statistics plus force-value
    statistics for nonlinear-safe gradient computation.
    """
    JtJ: Any
    Jty: Any
    y_sumsq: float
    Jtf: Any
    f_sumsq: float
    fty: float
    nframe: int
    step_index: NotRequired[int]


class FMOut(TypedDict, total=False):
    """Return schema for FMTrainerAnalytic.step()."""
    name: str
    loss: float
    grad: Any
    hessian: NotRequired[Any]
    update: Any
    meta: Dict[str, Any]


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class FMTrainerAnalytic(BaseTrainer):
    """Pure gradient-provider FM trainer using pre-accumulated normal equations."""

    BATCH_SCHEMA: Dict[str, Any] = {
        "JtJ": (
            "required np.ndarray; shape (n_params, n_params); normalized "
            "weighted average of J_i^T J_i"
        ),
        "Jty": (
            "required np.ndarray; shape (n_params,); normalized weighted "
            "average of J_i^T y_i"
        ),
        "y_sumsq": (
            "required float; normalized weighted average of ||y_i||^2"
        ),
        "Jtf": (
            "required np.ndarray; shape (n_params,); normalized weighted "
            "average of J_i^T f_i where f_i are actual model forces"
        ),
        "f_sumsq": (
            "required float; normalized weighted average of ||f_i||^2"
        ),
        "fty": (
            "required float; normalized weighted average of f_i^T y_i"
        ),
        "nframe": (
            "required int; number of frames contributing to the accumulated "
            "statistics"
        ),
        "step_index": "optional int; logging step counter; default 0",
    }

    RETURN_SCHEMA: Dict[str, Any] = {
        "name": 'str; always "FM"',
        "loss": (
            "float; 0.5 * (f_sumsq - 2*fty + y_sumsq) evaluated from "
            "normalized FM statistics"
        ),
        "grad": (
            "np.ndarray; shape (n_params,); FM gradient derived from the "
            "normalized normal equations"
        ),
        "hessian": (
            "np.ndarray|None; shape (n_params,n_params); FM Hessian when the "
            "optimizer accepts a Hessian, else None"
        ),
        "update": (
            "np.ndarray; shape (n_params,); optimizer update if apply_update=True "
            "else zeros_like(grad)"
        ),
        "meta": (
            "dict; diagnostics (nframe, step_index, grad_norm, update_norm)"
        ),
    }

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Return a dict with `batch` and `return` schema for introspection."""
        return {"batch": cls.BATCH_SCHEMA, "return": cls.RETURN_SCHEMA}

    def get_offsets(self) -> List[slice]:
        """Return parameter slices for each interaction in the forcefield.

        Returns
        -------
        list[slice]
            Slices into the global parameter vector, ordered like the owned
            forcefield interactions.
        """
        return self.forcefield.interaction_offsets()

    @staticmethod
    def make_batch(
        *,
        JtJ: Any,
        Jty: Any,
        y_sumsq: float,
        Jtf: Any,
        f_sumsq: float,
        fty: float,
        nframe: int,
        step_index: int = 0,
    ) -> FMBatch:
        """Build an FM batch from pre-accumulated statistics.

        Parameters
        ----------
        JtJ : array-like, shape (n_params, n_params)
            Normalized force-matching normal matrix.
        Jty : array-like, shape (n_params,)
            Normalized right-hand side from target forces.
        y_sumsq : float
            Normalized squared norm of target forces.
        Jtf : array-like, shape (n_params,)
            Normalized ``J.T @ f_model`` term for the current model forces.
        f_sumsq : float
            Normalized squared norm of current model forces.
        fty : float
            Normalized dot product between model and target forces.
        nframe : int
            Number of frames contributing to the statistics.
        step_index : int, default=0
            Iteration index used for logging.

        Returns
        -------
        FMBatch
            Batch dictionary with NumPy arrays and scalar diagnostics.
        """
        return FMBatch(
            JtJ=np.asarray(JtJ, dtype=np.float64),
            Jty=np.asarray(Jty, dtype=np.float64),
            y_sumsq=float(y_sumsq),
            Jtf=np.asarray(Jtf, dtype=np.float64),
            f_sumsq=float(f_sumsq),
            fty=float(fty),
            nframe=int(nframe),
            step_index=int(step_index),
        )

    def _apply_mask_to_grad_hessian(
        self,
        grad: np.ndarray,
        hessian: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        param_mask = np.asarray(self.optimizer.mask, dtype=bool)
        if np.all(param_mask):
            return grad, hessian

        g = np.asarray(grad, dtype=np.float64).copy()
        g[~param_mask] = 0.0
        if hessian is None:
            return g, None
        h = np.asarray(hessian, dtype=np.float64).copy()
        h[~param_mask, :] = 0.0
        h[:, ~param_mask] = 0.0
        return g, h

    def step(self, batch: Dict[str, Any], apply_update: bool = True) -> FMOut:
        """Run one FM optimizer step from normalized FM statistics.

        Parameters
        ----------
        batch : FMBatch
            Batch returned by :meth:`make_batch` or an equivalent dictionary.
            It must contain normalized ``JtJ``, ``Jty``, ``y_sumsq``, ``Jtf``,
            ``f_sumsq``, ``fty``, and ``nframe``.
        apply_update : bool, default=True
            If ``True``, update parameters through the configured optimizer.
            If ``False``, return zero ``update`` while still computing loss and
            gradients.

        Returns
        -------
        FMOut
            Dictionary with scalar force-matching loss, masked gradient,
            optional Hessian, optimizer update, and diagnostics.
        """
        JtJ = np.asarray(batch["JtJ"], dtype=np.float64)
        Jty = np.asarray(batch["Jty"], dtype=np.float64)
        y_sumsq = float(batch["y_sumsq"])
        Jtf = np.asarray(batch["Jtf"], dtype=np.float64)
        f_sumsq = float(batch["f_sumsq"])
        fty = float(batch["fty"])
        nframe = int(batch["nframe"])
        step_index = int(batch.get("step_index", 0))

        if nframe <= 0:
            raise ValueError("nframe must be positive.")
        
        # Nonlinear potentials require the current-model force term Jtf; for a
        # linear model this reduces to the usual normal-equation gradient.
        grad = np.asarray(Jtf, dtype=np.float64) - np.asarray(Jty, dtype=np.float64)
        loss = 0.5 * (float(f_sumsq) - 2.0 * float(fty) + float(y_sumsq))
        hessian_full = np.asarray(JtJ, dtype=np.float64)
        hessian: Optional[np.ndarray] = hessian_full if self.optimizer_accepts_hessian() else None

        grad, hessian = self._apply_mask_to_grad_hessian(grad, hessian)

        update = self._optimizer_step(grad, hessian, apply_update=apply_update)

        if self.logger is not None:
            self.logger.add_scalar("FM/loss", float(loss), step_index)
            self.logger.add_scalar("FM/grad_norm", float(np.linalg.norm(grad)), step_index)
            self.logger.add_scalar("FM/update_norm", float(np.linalg.norm(update)), step_index)

        return {
            "name": "FM",
            "loss": float(loss),
            "grad": grad,
            "hessian": hessian,
            "update": update,
            "meta": {
                "nframe": nframe,
                "step_index": step_index,
                "grad_norm": float(np.linalg.norm(grad)),
                "update_norm": float(np.linalg.norm(update)),
            },
        }
