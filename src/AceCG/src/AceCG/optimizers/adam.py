"""Implement the staged AceCG Adam optimizer."""

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

    def __init__(self, L, mask, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, noise_sigma=0.0, seed=None):
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
        """
        Perform one Adam update step using masked gradient.

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
        g = grad.copy()
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        denom = np.sqrt(v_hat) + self.eps
        precond = 1.0 / denom

        update = np.zeros_like(g)
        update[self.mask] = self.lr * m_hat[self.mask] / (np.sqrt(v_hat[self.mask]) + self.eps)
        if self.noise_sigma > 0.0:
            z = np.zeros_like(g)
            z[self.mask] = self.rng.standard_normal(np.count_nonzero(self.mask)).astype(self.L.dtype, copy=False)
            update[self.mask] += (self.noise_sigma * self.lr) * (z[self.mask] * precond[self.mask])

        self.L -= update
        self.last_update = -update
        return self.last_update

    def state_dict(self) -> dict:
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

    def load_state_dict(self, state: dict) -> None:
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
