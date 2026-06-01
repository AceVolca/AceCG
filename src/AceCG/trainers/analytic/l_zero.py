"""First-order expected-L0 trainer for hard-concrete interaction gates."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    from typing import NotRequired, TypedDict
except ImportError:  # Python < 3.11
    from typing_extensions import NotRequired, TypedDict

from ...potentials.gated import GatedPotential
from ..base import BaseTrainer


class L0Batch(TypedDict, total=False):
    """Batch schema for :class:`L0InteractionTrainerAnalytic`."""

    step_index: NotRequired[int]
    L0_lambda: NotRequired[float]


class L0Out(TypedDict, total=False):
    """Return schema for :class:`L0InteractionTrainerAnalytic.step`."""

    name: str
    loss: float
    grad: np.ndarray
    hessian: Optional[np.ndarray]
    update: np.ndarray
    active_probability: np.ndarray
    gate_value: np.ndarray
    labels: list[str]
    meta: Dict[str, Any]


class L0InteractionTrainerAnalytic(BaseTrainer):
    """Gradient provider for expected hard-concrete L0 interaction cost.

    This trainer only contributes gradients for the ``log_alpha`` parameter
    appended by :class:`~AceCG.potentials.gated.GatedPotential`. It is intended
    to be combined with REM/MSE/FM data trainers through
    ``MultiTrainerAnalytic(..., combine_mode="grad")`` and a first-order
    optimizer.
    """

    BATCH_SCHEMA: Dict[str, Any] = {
        "step_index": "optional int; logging step counter; default 0",
        "L0_lambda": "optional float; overrides the trainer-level penalty weight",
    }

    RETURN_SCHEMA: Dict[str, Any] = {
        "name": 'str; always "L0"',
        "loss": "float; lambda * sum(cost_weight * P(z>0))",
        "grad": "np.ndarray; shape (n_params,); nonzero only at gate log_alpha entries",
        "hessian": "None; first-order-only trainer",
        "update": "np.ndarray; optimizer update if apply_update=True else zeros_like(grad)",
        "active_probability": "np.ndarray; P(z>0) for each gated potential",
        "gate_value": "np.ndarray; current gate value for each gated potential",
        "labels": "list[str]; gated interaction labels",
        "meta": "dict; diagnostics",
    }

    def __init__(
        self,
        forcefield,
        optimizer,
        *,
        L0_lambda: float = 1.0,
        cost_weights: Optional[Dict[object, float]] = None,
        beta: Optional[float] = None,
        logger=None,
    ) -> None:
        super().__init__(forcefield, optimizer, beta, logger)
        self.L0_lambda = float(L0_lambda)
        self.cost_weights = dict(cost_weights or {})

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        return {"batch": cls.BATCH_SCHEMA, "return": cls.RETURN_SCHEMA}

    @staticmethod
    def make_batch(*, step_index: int = 0, L0_lambda: Optional[float] = None) -> L0Batch:
        batch: L0Batch = {"step_index": int(step_index)}
        if L0_lambda is not None:
            batch["L0_lambda"] = float(L0_lambda)
        return batch

    def _cost_weight(self, key: object, pot: GatedPotential) -> float:
        if key in self.cost_weights:
            return float(self.cost_weights[key])
        label = key.label() if hasattr(key, "label") else str(key)
        if label in self.cost_weights:
            return float(self.cost_weights[label])
        if pot.name in self.cost_weights:
            return float(self.cost_weights[pot.name])
        return 1.0

    def step(self, batch: Optional[L0Batch] = None, apply_update: bool = True) -> L0Out:
        if batch is None:
            batch = {}
        if not isinstance(batch, dict):
            raise TypeError("L0InteractionTrainerAnalytic.step expects batch as a dict.")

        if apply_update and self.optimizer_accepts_hessian():
            raise ValueError(
                "L0InteractionTrainerAnalytic is first-order only; use Adam, "
                "AdamW, or RMSProp instead of a Hessian optimizer."
            )

        step_index = int(batch.get("step_index", 0))
        L0_lambda = float(batch.get("L0_lambda", self.L0_lambda))

        n_params = self.forcefield.n_params()
        grad = np.zeros(n_params, dtype=np.float64)
        probs = []
        gates = []
        labels = []
        weights = []

        for key, pot, sl in self.forcefield.param_blocks():
            if not isinstance(pot, GatedPotential):
                continue
            gate_idx = sl.stop - 1
            weight = self._cost_weight(key, pot)
            p_active = pot.active_probability()
            grad[gate_idx] += L0_lambda * weight * pot.active_probability_grad()
            probs.append(p_active)
            gates.append(pot.gate_value())
            weights.append(weight)
            labels.append(pot.name or (key.label() if hasattr(key, "label") else str(key)))

        probs_arr = np.asarray(probs, dtype=np.float64)
        gates_arr = np.asarray(gates, dtype=np.float64)
        weights_arr = np.asarray(weights, dtype=np.float64)
        loss = float(L0_lambda * np.sum(weights_arr * probs_arr))

        if apply_update:
            update = self.optimizer.step(grad)
            self.clamp_and_update()
        else:
            update = np.zeros_like(grad)

        if self.logger is not None:
            self.logger.add_scalar("L0/loss", loss, step_index)
            self.logger.add_scalar("L0/grad_norm", float(np.linalg.norm(grad)), step_index)
            self.logger.add_scalar("L0/expected_active", float(np.sum(probs_arr)), step_index)
            self.logger.add_scalar("L0/current_active", float(np.sum(gates_arr > 0.0)), step_index)

        return {
            "name": "L0",
            "loss": loss,
            "grad": grad,
            "hessian": None,
            "update": update,
            "active_probability": probs_arr,
            "gate_value": gates_arr,
            "labels": labels,
            "meta": {
                "step_index": step_index,
                "L0_lambda": L0_lambda,
                "n_gates": int(probs_arr.size),
                "expected_active": float(np.sum(probs_arr)),
                "current_active": int(np.sum(gates_arr > 0.0)),
                "grad_norm": float(np.linalg.norm(grad)),
                "update_norm": float(np.linalg.norm(update)),
            },
        }


__all__ = ["L0Batch", "L0Out", "L0InteractionTrainerAnalytic"]
