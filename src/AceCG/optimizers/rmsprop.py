"""RMSprop optimizer implementation with AceCG checkpoint validation."""

from typing import Any

import numpy as np

from .base import BaseOptimizer


class RMSpropMaskedOptimizer(BaseOptimizer):
    """RMSprop optimizer with masked parameter updates.

    Parameters
    ----------
    L : np.ndarray
        Initial parameter vector.
    mask : np.ndarray
        Boolean mask selecting trainable entries of ``L``.
    lr : float, default=1e-2
        Learning rate.
    alpha : float, default=0.99
        Smoothing constant for the squared-gradient running average.
    eps : float, default=1e-8
        Numerical stability term added to denominator.
    weight_decay : float, default=0.0
        L2 penalty coefficient (added into the gradient).
    momentum : float, default=0.0
        Momentum coefficient (0 disables momentum).
    centered : bool, default=False
        Whether to use the centered variance estimate.
    noise_sigma : float, default=0.0
        Standard deviation of optional preconditioned Gaussian noise.
    seed : int or None, optional
        Seed for reproducible optimizer noise.
    """

    _CHECKPOINT_REQUIRED_KEYS = (
        "alpha",
        "eps",
        "weight_decay",
        "momentum",
        "centered",
        "noise_sigma",
        "square_avg",
        "momentum_buffer",
        "grad_avg",
    )
    _CHECKPOINT_SCALAR_KEYS = (
        "alpha",
        "eps",
        "weight_decay",
        "momentum",
        "noise_sigma",
    )
    _CHECKPOINT_BOOL_KEYS = ("centered",)

    def __init__(
        self,
        L,
        mask,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0.0,
        momentum=0.0,
        centered=False,
        noise_sigma=0.0,
        seed=None,
    ):
        """Initialize RMSprop running averages and optional buffers."""
        super().__init__(L, mask, lr)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.centered = bool(centered)

        # Full-size buffers keep checkpoint shapes stable across masks.
        self.square_avg = np.zeros_like(L)
        self.momentum_buffer = np.zeros_like(L) if self.momentum > 0.0 else None
        self.grad_avg = np.zeros_like(L) if self.centered else None

        self.noise_sigma = float(noise_sigma)
        self.rng = np.random.default_rng(seed)

        self.last_update = None

    def step(self, grad: np.ndarray) -> np.ndarray:
        """Perform one RMSprop update step using a masked gradient.

        Parameters
        ----------
        grad : np.ndarray
            Full gradient vector (same shape as self.L).

        Returns
        -------
        update : np.ndarray
            Full update vector (zeros at masked-out indices) that was subtracted.
        """
        g = grad.copy()
        idx = self.mask

        # ── Phase 1: in-gradient L2 decay ──────────────────────────────
        # RMSprop's weight decay is coupled: it changes the gradient before
        # the running averages are updated, unlike AdamW's decoupled form.
        if self.weight_decay != 0.0:
            g[idx] = g[idx] + self.weight_decay * self.L[idx]

        # ── Phase 2: squared-gradient average ─────────────────────────
        # The square average is full-size checkpoint state even though only
        # masked entries are later applied to the parameter vector.
        self.square_avg = self.alpha * self.square_avg + (1.0 - self.alpha) * (g * g)

        # ── Phase 3: RMS denominator ──────────────────────────────────
        # Centered RMSprop subtracts the squared mean gradient to estimate a
        # variance; the non-centered path uses the raw second moment.
        if self.centered:
            self.grad_avg = self.alpha * self.grad_avg + (1.0 - self.alpha) * g
            var = self.square_avg - (self.grad_avg * self.grad_avg)
            # Centered variance can go slightly negative from roundoff.
            var = np.maximum(var, 0.0)
            denom = np.sqrt(var) + self.eps
        else:
            denom = np.sqrt(self.square_avg) + self.eps

        precond = 1.0 / denom

        # ── Phase 4: masked RMSprop displacement ──────────────────────
        # Momentum, when enabled, stores preconditioned gradients rather than
        # raw gradients so resumed runs reproduce the same next displacement.
        update = np.zeros_like(g)

        if self.momentum > 0.0:
            if self.momentum_buffer is None:
                self.momentum_buffer = np.zeros_like(g)
            self.momentum_buffer[idx] = (
                self.momentum * self.momentum_buffer[idx] + (g[idx] / denom[idx])
            )
            update[idx] = self.lr * self.momentum_buffer[idx]
        else:
            update[idx] = self.lr * (g[idx] / denom[idx])

        # ── Phase 5: optional preconditioned noise ────────────────────
        # Stochastic perturbations use the same RMS denominator as the
        # deterministic displacement and never touch inactive parameters.
        if self.noise_sigma > 0.0:
            z = np.zeros_like(g)
            z[idx] = self.rng.standard_normal(np.count_nonzero(idx)).astype(self.L.dtype, copy=False)
            update[idx] += (self.noise_sigma * self.lr) * (z[idx] * precond[idx])

        # ── Phase 6: apply AceCG sign convention ──────────────────────
        # The return value follows AceCG trainer expectations: signed parameter
        # displacement, not the positive quantity subtracted from ``L``.
        self.L -= update

        self.last_update = -update
        return self.last_update

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer state including RMSprop accumulators.

        Returns
        -------
        dict
            Serializable state with base fields, running averages, optional
            momentum/centered buffers, and hyperparameters.
        """
        d = super().state_dict()
        d.update({
            "alpha": float(self.alpha),
            "eps": float(self.eps),
            "weight_decay": float(self.weight_decay),
            "momentum": float(self.momentum),
            "centered": bool(self.centered),
            "noise_sigma": float(self.noise_sigma),
            "square_avg": self.square_avg.tolist(),
            "momentum_buffer": (
                self.momentum_buffer.tolist() if self.momentum_buffer is not None else None
            ),
            "grad_avg": self.grad_avg.tolist() if self.grad_avg is not None else None,
        })
        return d

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore RMSprop state from :meth:`state_dict`.

        Parameters
        ----------
        state : dict
            State dictionary produced by a compatible
            :class:`RMSpropMaskedOptimizer`.
        """
        super().load_state_dict(state)
        self.alpha = float(state["alpha"])
        self.eps = float(state["eps"])
        self.weight_decay = float(state["weight_decay"])
        self.momentum = float(state["momentum"])
        self.centered = bool(state["centered"])
        self.noise_sigma = float(state["noise_sigma"])
        self.square_avg = np.asarray(state["square_avg"], dtype=self.L.dtype)
        self.momentum_buffer = (
            np.asarray(state["momentum_buffer"], dtype=self.L.dtype)
            if state.get("momentum_buffer") is not None else None
        )
        self.grad_avg = (
            np.asarray(state["grad_avg"], dtype=self.L.dtype)
            if state.get("grad_avg") is not None else None
        )

    def validate_checkpoint_state(
        self,
        state: Any,
        forcefield: Any,
        *,
        require_completed_step: bool = False,
    ) -> None:
        """Validate RMSprop accumulators and hyperparameters in a checkpoint."""
        super().validate_checkpoint_state(
            state,
            forcefield,
            require_completed_step=require_completed_step,
        )
        self._require_checkpoint_keys(state, self._CHECKPOINT_REQUIRED_KEYS)
        expected_shape = self._checkpoint_forcefield_params(forcefield).shape
        for key in self._CHECKPOINT_SCALAR_KEYS:
            self._validate_checkpoint_scalar_match(state, key)
        for key in self._CHECKPOINT_BOOL_KEYS:
            self._validate_checkpoint_bool_match(state, key)
        self._checkpoint_array(state, "square_avg", expected_shape)

        has_momentum = float(state["momentum"]) > 0.0
        momentum_buffer = self._checkpoint_array(
            state,
            "momentum_buffer",
            expected_shape,
            allow_none=not has_momentum,
        )
        if not has_momentum and momentum_buffer is not None:
            raise ValueError(
                "optimizer checkpoint field 'momentum_buffer' must be None "
                "when 'momentum' is zero."
            )

        centered = bool(state["centered"])
        grad_avg = self._checkpoint_array(
            state,
            "grad_avg",
            expected_shape,
            allow_none=not centered,
        )
        if not centered and grad_avg is not None:
            raise ValueError(
                "optimizer checkpoint field 'grad_avg' must be None when "
                "'centered' is false."
            )
