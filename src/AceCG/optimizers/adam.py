import numpy as np
from .base import BaseOptimizer

class AdamMaskedOptimizer(BaseOptimizer):
    """
    Adam optimizer with masked parameter updates.

    Supports standard Adam logic but only updates parameters where mask=True.
    """

    def __init__(self, L, mask, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(L, mask, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = np.zeros_like(L)
        self.v = np.zeros_like(L)

        self.last_update = None

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

        update = np.zeros_like(g)
        update[self.mask] = self.lr * m_hat[self.mask] / (np.sqrt(v_hat[self.mask]) + self.eps)

        self.L -= update
        self.last_update = -update
        return self.last_update