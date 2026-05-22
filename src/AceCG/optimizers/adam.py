"""Adam optimizer implementation with AceCG checkpoint validation."""

from typing import Any

import numpy as np

from .base import BaseOptimizer


class AdamMaskedOptimizer(BaseOptimizer):
    """Adam optimizer with masked parameter updates.

    Parameters
    ----------
    L : np.ndarray
        Initial full parameter vector.
    mask : np.ndarray
        Boolean mask selecting trainable entries of ``L``.
    lr : float, default=1e-2
        Adam learning rate.
    beta1 : float, default=0.9
        Exponential decay rate for the first moment estimate.
    beta2 : float, default=0.999
        Exponential decay rate for the second moment estimate.
    eps : float, default=1e-8
        Numerical stability term added to the square-root variance.
    noise_sigma : float, default=0.0
        Standard deviation of optional preconditioned Gaussian noise. ``0``
        disables stochastic perturbations.
    seed : int or None, optional
        Seed passed to NumPy's random generator for reproducible noise.
    """

    _CHECKPOINT_REQUIRED_KEYS = (
        "t",
        "m",
        "v",
        "beta1",
        "beta2",
        "eps",
        "noise_sigma",
    )
    _CHECKPOINT_ARRAY_KEYS = ("m", "v")
    _CHECKPOINT_SCALAR_KEYS = ("beta1", "beta2", "eps", "noise_sigma")

    def __init__(
        self,
        L,
        mask,
        lr=1e-2,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        noise_sigma=0.0,
        seed=None,
    ):
        """Initialize Adam moments and optional stochastic noise state."""
        super().__init__(L, mask, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = np.zeros_like(L)
        self.v = np.zeros_like(L)

        self.last_update = None
        self.noise_sigma = float(noise_sigma)
        self.rng = np.random.default_rng(seed)

    def step(self, grad: np.ndarray) -> np.ndarray:
        """Perform one Adam update step using masked gradient.

        Parameters
        ----------
        grad : np.ndarray
            Full gradient vector (same shape as self.L)

        Returns
        -------
        update : np.ndarray
            Full update vector (zeros at masked-out indices)
        """
        self.t += 1

        # ── Phase 1: Adam moment estimates ──────────────────────────────
        # Moments are kept at full parameter-vector size so checkpoints stay
        # compatible even when only a masked subset is currently trainable.
        g = grad.copy()
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g ** 2)

        # ── Phase 2: bias-corrected preconditioner ──────────────────────
        # The same denominator is reused for deterministic descent and
        # optional preconditioned noise, keeping stochastic steps scale-aware.
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        denom = np.sqrt(v_hat) + self.eps
        precond = 1.0 / denom

        # ── Phase 3: masked Adam displacement ───────────────────────────
        # Inactive entries must remain exactly zero in the applied update, but
        # their moment buffers still track the full incoming gradient history.
        update = np.zeros_like(g)
        update[self.mask] = self.lr * m_hat[self.mask] / (np.sqrt(v_hat[self.mask]) + self.eps)
        if self.noise_sigma > 0.0:
            z = np.zeros_like(g)
            z[self.mask] = self.rng.standard_normal(np.count_nonzero(self.mask)).astype(self.L.dtype, copy=False)
            update[self.mask] += (self.noise_sigma * self.lr) * (z[self.mask] * precond[self.mask])

        # ── Phase 4: apply AceCG sign convention ────────────────────────
        # Optimizers mutate ``L`` by subtracting the displacement, then return
        # the signed parameter change observed by callers and trainers.
        self.L -= update
        self.last_update = -update
        return self.last_update

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer state including Adam moments.

        Returns
        -------
        dict
            Serializable state containing base fields, step counter, moment
            buffers, hyperparameters, and noise scale.
        """
        d = super().state_dict()
        d.update({
            "t": int(self.t),
            "m": self.m.tolist(),
            "v": self.v.tolist(),
            "beta1": float(self.beta1),
            "beta2": float(self.beta2),
            "eps": float(self.eps),
            "noise_sigma": float(self.noise_sigma),
        })
        return d

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore Adam state from :meth:`state_dict`.

        Parameters
        ----------
        state : dict
            State dictionary produced by a compatible
            :class:`AdamMaskedOptimizer`.
        """
        super().load_state_dict(state)
        self.t = int(state["t"])
        self.m = np.asarray(state["m"], dtype=self.L.dtype)
        self.v = np.asarray(state["v"], dtype=self.L.dtype)
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.eps = float(state["eps"])
        self.noise_sigma = float(state["noise_sigma"])

    def validate_checkpoint_state(
        self,
        state: Any,
        forcefield: Any,
        *,
        require_completed_step: bool = False,
    ) -> None:
        """Validate Adam moment buffers and hyperparameters in a checkpoint."""
        super().validate_checkpoint_state(
            state,
            forcefield,
            require_completed_step=require_completed_step,
        )
        self._require_checkpoint_keys(state, self._CHECKPOINT_REQUIRED_KEYS)
        expected_shape = self._checkpoint_forcefield_params(forcefield).shape
        for key in self._CHECKPOINT_SCALAR_KEYS:
            self._validate_checkpoint_scalar_match(state, key)
        for key in self._CHECKPOINT_ARRAY_KEYS:
            self._checkpoint_array(state, key, expected_shape)
        if require_completed_step:
            step_count = int(state["t"])
            if step_count <= 0:
                raise ValueError(
                    f"optimizer checkpoint field 't' is {step_count}; "
                    "strict resume requires a completed optimizer step."
                )
