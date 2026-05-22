"""AceCG optimizers multithreaded adam implementation."""

from ..base import BaseOptimizer

import math
import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def _adam_masked_step_kernel(L, m, v, grad, mask, lr, beta1, beta2, eps,
                             t, noise_sigma, z, out_update):
    """Update masked Adam parameters in a Numba kernel.

    Parameters
    ----------
    L : np.ndarray
        Parameter array to be updated.
    m : np.ndarray
        First moment vector.
    v : np.ndarray
        Second moment vector.
    grad : np.ndarray
        Gradient array.
    mask : np.ndarray
        Boolean mask array indicating which parameters to update.
    lr : float
        Learning rate.
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    eps : float
        Small constant for numerical stability.
    t : int
        Time step (iteration count).
    noise_sigma : float
        Standard deviation of the noise to be added.
    z : np.ndarray
        Random noise array.
    out_update : np.ndarray
        Array to store the computed updates.
    """
    n = L.size
    b1corr = 1.0 - beta1**t
    b2corr = 1.0 - beta2**t
    for i in prange(n):
        g = grad[i]
        m_i = beta1*m[i] + (1.0 - beta1)*g
        v_i = beta2*v[i] + (1.0 - beta2)*(g*g)
        m[i] = m_i
        v[i] = v_i

        u = 0.0
        if mask[i]:
            m_hat = m_i / b1corr
            v_hat = v_i / b2corr
            denom = math.sqrt(v_hat) + eps
            u = lr * m_hat / denom
            if noise_sigma > 0.0:
                u += noise_sigma * lr * z[i] / denom
            L[i] -= u
        # Store the subtracted displacement before AceCG's return sign flip.
        out_update[i] = u

class MTAdamOptimizer(BaseOptimizer):
    """Numba-parallel Adam optimizer with masked parameter updates.

    Parameters
    ----------
    L : np.ndarray
        Initial full parameter vector.
    mask : np.ndarray
        Boolean mask selecting trainable entries of ``L``.
    lr : float, default=1e-2
        Adam learning rate.
    beta1 : float, default=0.9
        Exponential decay rate for first moments.
    beta2 : float, default=0.999
        Exponential decay rate for second moments.
    eps : float, default=1e-8
        Numerical stability term in the Adam denominator.
    noise_sigma : float, default=0.0
        Standard deviation of optional preconditioned Gaussian noise.
    seed : int or None, optional
        Seed for reproducible optimizer noise.

    Notes
    -----
    This class has the same public behavior as :class:`AdamMaskedOptimizer`
    but performs the elementwise update in a compiled Numba kernel.
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

        self._warmup_kernel()

    def _warmup_kernel(self) -> None:
        """Compile the numba kernel without mutating optimizer state."""
        warm_L = np.asarray(self.L).copy()
        warm_m = np.zeros_like(warm_L)
        warm_v = np.zeros_like(warm_L)
        warm_grad = np.zeros_like(warm_L)
        warm_z = np.zeros_like(warm_L)
        warm_update = np.zeros_like(warm_L)
        warm_mask = np.asarray(self.mask, dtype=np.bool_)

        _adam_masked_step_kernel(
            warm_L,
            warm_m,
            warm_v,
            warm_grad,
            warm_mask,
            float(self.lr),
            float(self.beta1),
            float(self.beta2),
            float(self.eps),
            1,
            0.0,
            warm_z,
            warm_update,
        )

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
        grad = np.asarray(grad, dtype=self.L.dtype)
        if grad.shape != self.L.shape:
            raise ValueError(f"grad shape mismatch: expected {self.L.shape}, got {grad.shape}")

        self.t += 1
        z = np.zeros_like(self.L)
        if self.noise_sigma > 0.0:
            # Draw exactly the same masked noise shape as serial Adam so
            # checkpointed stochastic runs do not depend on the backend.
            z[self.mask] = self.rng.standard_normal(np.count_nonzero(self.mask)).astype(
                self.L.dtype,
                copy=False,
            )
        out_update = np.zeros_like(self.L)
        _adam_masked_step_kernel(self.L, self.m, self.v, grad, self.mask,
                                 self.lr, self.beta1, self.beta2, self.eps,
                                 self.t, self.noise_sigma, z, out_update)
        # The kernel stores the positive displacement subtracted from ``L``;
        # trainers expect the signed parameter change returned by serial Adam.
        self.last_update = -out_update
        return self.last_update

    def state_dict(self) -> dict:
        """Return optimizer state including compiled-kernel Adam moments."""
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
        """Restore multithreaded Adam state from :meth:`state_dict`.

        Parameters
        ----------
        state : dict
            State dictionary produced by a compatible
            :class:`MTAdamOptimizer`.
        """
        super().load_state_dict(state)
        self.t = int(state["t"])
        self.m = np.asarray(state["m"], dtype=self.L.dtype)
        self.v = np.asarray(state["v"], dtype=self.L.dtype)
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.eps = float(state["eps"])
        self.noise_sigma = float(state["noise_sigma"])
