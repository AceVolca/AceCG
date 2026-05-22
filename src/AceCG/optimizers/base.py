"""Shared optimizer contracts for AceCG trainer-owned parameter state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseOptimizer(ABC):
    """Base class for masked optimizers used by AceCG trainers.

    Parameters
    ----------
    L : np.ndarray
        Initial full parameter vector.
    mask : np.ndarray
        Boolean array with the same shape as ``L``. ``True`` entries are
        trainable; ``False`` entries remain fixed.
    lr : float
        Learning-rate multiplier used by concrete optimizers.

    Attributes
    ----------
    L : np.ndarray
        Mutable optimizer-side parameter vector.
    mask : np.ndarray
        Boolean active-coordinate mask.
    lr : float
        Current learning rate.
    """

    _CHECKPOINT_BASE_KEYS = ("optimizer_class", "L", "mask", "lr")

    def __init__(self, L: np.ndarray, mask: np.ndarray, lr: float) -> None:
        """Initialize the optimizer parameter vector, mask, and learning rate."""
        self.L = L.copy()
        self.mask = mask.copy()
        self.lr = lr

    def set_params(self, L_new: np.ndarray) -> None:
        """
        Update the internal parameter vector L.

        Parameters
        ----------
        new_L : np.ndarray
            New parameter values (must match shape of self.L).
        """
        assert L_new.shape == self.L.shape, "Parameter shape mismatch"
        self.L = L_new.copy()

    def state_dict(self) -> dict[str, Any]:
        """Return serializable optimizer state (parameters + base scalars).

        Returns
        -------
        dict
            JSON-compatible dictionary containing ``L``, ``mask``, and ``lr``.

        Subclasses override to include moment buffers, step counters, etc.
        """
        return {
            "L": self.L.tolist(),
            "mask": self.mask.tolist(),
            "lr": float(self.lr),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore optimizer state from a dictionary.

        Parameters
        ----------
        state : dict
            Dictionary produced by :meth:`state_dict` on a compatible optimizer.
        """
        self.L = np.asarray(state["L"], dtype=self.L.dtype)
        self.mask = np.asarray(state["mask"], dtype=bool)
        self.lr = float(state["lr"])

    def checkpoint_state_dict(self) -> dict[str, Any]:
        """Return optimizer state with the class identity required for resume."""
        state = dict(self.state_dict())
        state["optimizer_class"] = type(self).__name__
        return state

    def validate_checkpoint_state(
        self,
        state: Any,
        forcefield: Any,
        *,
        require_completed_step: bool = False,
    ) -> None:
        """Validate base optimizer checkpoint fields against a forcefield.

        Parameters
        ----------
        state : Any
            Candidate checkpoint payload.
        forcefield : Any
            Object exposing ``param_array()`` for the checkpoint forcefield.
        require_completed_step : bool, default=False
            Accepted for subclasses that own step counters.
        """
        del require_completed_step
        if not isinstance(state, dict):
            raise ValueError(
                "optimizer checkpoint state must be a dictionary, got "
                f"{type(state).__name__}."
            )
        self._require_checkpoint_keys(state, self._CHECKPOINT_BASE_KEYS)
        optimizer_class = str(state["optimizer_class"])
        current_class = type(self).__name__
        if optimizer_class != current_class:
            raise ValueError(
                "optimizer checkpoint class "
                f"{optimizer_class!r} is not compatible with current "
                f"optimizer class {current_class!r}."
            )

        forcefield_params = self._checkpoint_forcefield_params(forcefield)
        expected_shape = forcefield_params.shape
        checkpoint_l = self._checkpoint_array(state, "L", expected_shape)
        if not np.allclose(
            np.asarray(checkpoint_l, dtype=np.float64),
            forcefield_params,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "optimizer checkpoint field 'L' does not match checkpoint "
                "forcefield parameters."
            )

        checkpoint_mask = self._checkpoint_array(state, "mask", expected_shape)
        current_mask = np.asarray(self.mask, dtype=bool)
        if current_mask.shape != expected_shape:
            raise ValueError(
                "current optimizer mask shape does not match checkpoint "
                "forcefield parameter shape."
            )
        if not np.array_equal(np.asarray(checkpoint_mask, dtype=bool), current_mask):
            raise ValueError(
                "optimizer checkpoint field 'mask' is not compatible with "
                "the current optimizer mask."
            )
        self._validate_checkpoint_scalar_match(state, "lr")

    def load_checkpoint_state(
        self,
        state: Any,
        forcefield: Any,
        *,
        require_completed_step: bool = False,
    ) -> None:
        """Validate and restore optimizer state from a strict checkpoint."""
        self.validate_checkpoint_state(
            state,
            forcefield,
            require_completed_step=require_completed_step,
        )
        self.load_state_dict(state)

    def _checkpoint_forcefield_params(self, forcefield: Any) -> np.ndarray:
        """Return the forcefield parameter vector used to validate checkpoints."""
        if not hasattr(forcefield, "param_array"):
            raise ValueError("checkpoint forcefield must expose param_array().")
        return np.asarray(forcefield.param_array(), dtype=np.float64)

    def _require_checkpoint_keys(
        self,
        state: dict[str, Any],
        keys: tuple[str, ...],
    ) -> None:
        """Raise when a checkpoint payload omits required optimizer fields."""
        missing = [key for key in keys if key not in state]
        if missing:
            formatted = ", ".join(repr(key) for key in missing)
            raise ValueError(
                "optimizer checkpoint is missing required field(s): "
                f"{formatted}."
            )

    def _checkpoint_array(
        self,
        state: dict[str, Any],
        key: str,
        expected_shape: tuple[int, ...],
        *,
        allow_none: bool = False,
    ) -> np.ndarray | None:
        """Return a checkpoint array after validating its parameter shape."""
        value = state.get(key)
        if value is None:
            if allow_none:
                return None
            raise ValueError(f"optimizer checkpoint field {key!r} must not be None.")
        array = np.asarray(value)
        if array.shape != expected_shape:
            raise ValueError(
                f"optimizer checkpoint field {key!r} has shape {array.shape}, "
                f"expected {expected_shape}."
            )
        return array

    def _validate_checkpoint_scalar_match(
        self,
        state: dict[str, Any],
        key: str,
    ) -> None:
        """Require a scalar optimizer checkpoint field to match this optimizer."""
        expected = getattr(self, key)
        actual = state[key]
        if not np.isclose(
            float(actual),
            float(expected),
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError(
                f"optimizer checkpoint field {key!r}={actual!r} is not "
                f"compatible with current value {expected!r}."
            )

    def _validate_checkpoint_bool_match(
        self,
        state: dict[str, Any],
        key: str,
    ) -> None:
        """Require a boolean optimizer checkpoint field to match this optimizer."""
        expected = bool(getattr(self, key))
        actual = bool(state[key])
        if actual != expected:
            raise ValueError(
                f"optimizer checkpoint field {key!r}={actual!r} is not "
                f"compatible with current value {expected!r}."
            )

    @abstractmethod
    def step(
        self,
        grad: np.ndarray,
        hessian: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply one optimization step.

        Parameters
        ----------
        grad : np.ndarray
            Gradient vector (same shape as L).
        hessian : np.ndarray, optional
            Hessian matrix (only used by second-order methods).

        Returns
        -------
        update : np.ndarray
            The update vector applied to self.L.
        """
        pass
