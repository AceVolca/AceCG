from ..base import BaseOptimizer

import math
import numpy as np
from numba import njit, prange, float64, boolean, int64

sig = (float64[:], float64[:], float64[:], float64[:], boolean[:],
       float64, float64, float64, float64, int64, float64, float64[:], float64[:])

@njit(parallel=True, fastmath=True)
def _adam_masked_step_kernel(L, m, v, grad, mask, lr, beta1, beta2, eps,
                             t, noise_sigma, z, out_update):
    '''
    Numba kernel for Adam optimizer with masked updates.
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
    '''
    n = L.size
    b1corr = 1.0 - beta1**t
    b2corr = 1.0 - beta2**t
    for i in prange(n):
        g = grad[i]
        # moments
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
        out_update[i] = u  # store the applied update (pre-sign)

class MTAdamOptimizer(BaseOptimizer):
    """
    Adam optimizer, multi-threaded with Numba with masked parameter updates.

    Supports standard Adam logic but only updates parameters where mask=True.
    
    Support random noise pertubation during the optimization
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
        z = np.zeros_like(self.L)
        if self.noise_sigma > 0.0:
            # only generate noise where needed
            z[self.mask] = self.rng.standard_normal(self.mask.sum())
        out_update = np.zeros_like(self.L)
        _adam_masked_step_kernel(self.L, self.m, self.v, grad, self.mask,
                                 self.lr, self.beta1, self.beta2, self.eps,
                                 self.t, self.noise_sigma, z, out_update)
        # match your original sign convention: return -update
        self.last_update = -out_update
        return self.last_update