"""AdamW optimizer implementation with AceCG checkpoint validation."""

from typing import Any

import numpy as np

from .base import BaseOptimizer


class AdamWMaskedOptimizer(BaseOptimizer):
    """AdamW optimizer with masked parameter updates.

    Parameters
    ----------
    L : np.ndarray
        Initial full parameter vector.
    mask : np.ndarray
        Boolean mask selecting trainable entries of ``L``.
    lr : float, default=1e-2
        AdamW learning rate.
    beta1 : float, default=0.9
        Exponential decay rate for the first moment estimate.
    beta2 : float, default=0.999
        Exponential decay rate for the second moment estimate.
    eps : float, default=1e-8
        Numerical stability term added to the variance denominator.
    weight_decay : float, default=0.0
        Decoupled weight-decay coefficient applied only to active entries.
    amsgrad : bool, default=False
        If ``True``, use the AMSGrad maximum-variance denominator.
    noise_sigma : float, default=0.0
        Standard deviation of optional preconditioned Gaussian noise.
    seed : int or None, optional
        Seed for reproducible optimizer noise.

    Notes
    -----
    The returned update follows AceCG's convention: it is the signed parameter
    displacement applied to ``L`` (negative for a standard descent step).
    """

    _CHECKPOINT_REQUIRED_KEYS = (
        "t",
        "m",
        "v",
        "vmax",
        "beta1",
        "beta2",
        "eps",
        "weight_decay",
        "amsgrad",
        "noise_sigma",
    )
    _CHECKPOINT_ARRAY_KEYS = ("m", "v")
    _CHECKPOINT_SCALAR_KEYS = (
        "beta1",
        "beta2",
        "eps",
        "weight_decay",
        "noise_sigma",
    )
    _CHECKPOINT_BOOL_KEYS = ("amsgrad",)

    def __init__(
        self,
        L,
        mask,
        lr=1e-2,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        noise_sigma=0.0,
        seed=None,
    ):
        """Initialize AdamW moments, AMSGrad state, and noise state."""
        super().__init__(L, mask, lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.amsgrad = bool(amsgrad)

        self.t = 0
        self.m = np.zeros_like(L)
        self.v = np.zeros_like(L)
        self.vmax = np.zeros_like(L) if self.amsgrad else None

        self.noise_sigma = float(noise_sigma)
        self.rng = np.random.default_rng(seed)

        self.last_update = None

    def step(self, grad: np.ndarray) -> np.ndarray:
        """Perform one AdamW update step using a masked gradient.

        Parameters
        ----------
        grad : np.ndarray
            Full gradient vector (same shape as self.L)

        Returns
        -------
        update : np.ndarray
            Full update vector that was SUBTRACTED from parameters
            (zeros at masked-out indices).
        """
        self.t += 1

        # ── Phase 1: Adam moment estimates ──────────────────────────────
        # Moment buffers are full-size checkpoint state; masking is applied
        # only when constructing the displacement.
        g = grad.copy()
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g ** 2)

        # ── Phase 2: bias-corrected denominator ────────────────────────
        # AMSGrad replaces the current variance with a monotone maximum, which
        # must be checkpointed separately as ``vmax``.
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        if self.amsgrad:
            self.vmax = np.maximum(self.vmax, v_hat)
            v_denom = np.sqrt(self.vmax) + self.eps
        else:
            v_denom = np.sqrt(v_hat) + self.eps

        precond = 1.0 / v_denom

        # ── Phase 3: masked Adam displacement ───────────────────────────
        # ``idx`` is the only write mask for parameters; inactive entries keep
        # zero displacement even when their moments are nonzero.
        update = np.zeros_like(g)
        idx = self.mask
        update[idx] = self.lr * (m_hat[idx] / v_denom[idx])

        # ── Phase 4: optional preconditioned noise ──────────────────────
        # Noise is sampled only for trainable coordinates and uses Adam's
        # denominator so its scale follows the same geometry as the step.
        if self.noise_sigma > 0.0:
            z = np.zeros_like(g)
            z[idx] = self.rng.standard_normal(np.count_nonzero(idx)).astype(self.L.dtype, copy=False)
            update[idx] += (self.noise_sigma * self.lr) * (z[idx] * precond[idx])

        # ── Phase 5: decoupled weight decay ─────────────────────────────
        if self.weight_decay != 0.0:
            # AdamW adds decay directly to the displacement, not the gradient.
            update[idx] += self.lr * self.weight_decay * self.L[idx]

        # ── Phase 6: apply AceCG sign convention ────────────────────────
        # The stored update is what gets subtracted from ``L``; callers receive
        # the signed parameter displacement after the subtraction.
        self.L -= update
        self.last_update = -update
        return self.last_update

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer state including AdamW moments and hyperparameters."""
        d = super().state_dict()
        d.update({
            "t": int(self.t),
            "m": self.m.tolist(),
            "v": self.v.tolist(),
            "vmax": self.vmax.tolist() if self.vmax is not None else None,
            "beta1": float(self.beta1),
            "beta2": float(self.beta2),
            "eps": float(self.eps),
            "weight_decay": float(self.weight_decay),
            "amsgrad": bool(self.amsgrad),
            "noise_sigma": float(self.noise_sigma),
        })
        return d

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore AdamW state from a compatible state dictionary.

        Parameters
        ----------
        state : dict
            Dictionary produced by :meth:`state_dict`.
        """
        super().load_state_dict(state)
        self.t = int(state["t"])
        self.m = np.asarray(state["m"], dtype=self.L.dtype)
        self.v = np.asarray(state["v"], dtype=self.L.dtype)
        self.vmax = (
            np.asarray(state["vmax"], dtype=self.L.dtype)
            if state.get("vmax") is not None else None
        )
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.eps = float(state["eps"])
        self.weight_decay = float(state["weight_decay"])
        self.amsgrad = bool(state["amsgrad"])
        self.noise_sigma = float(state["noise_sigma"])

    def validate_checkpoint_state(
        self,
        state: Any,
        forcefield: Any,
        *,
        require_completed_step: bool = False,
    ) -> None:
        """Validate AdamW moment buffers and hyperparameters in a checkpoint."""
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
        for key in self._CHECKPOINT_ARRAY_KEYS:
            self._checkpoint_array(state, key, expected_shape)

        amsgrad = bool(state["amsgrad"])
        vmax = self._checkpoint_array(
            state,
            "vmax",
            expected_shape,
            allow_none=not amsgrad,
        )
        if not amsgrad and vmax is not None:
            raise ValueError(
                "optimizer checkpoint field 'vmax' must be None when "
                "'amsgrad' is false."
            )

        if require_completed_step:
            step_count = int(state["t"])
            if step_count <= 0:
                raise ValueError(
                    f"optimizer checkpoint field 't' is {step_count}; "
                    "strict resume requires a completed optimizer step."
                )
