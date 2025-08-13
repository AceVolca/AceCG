import numpy as np
from .base import BaseOptimizer

class AdamWMaskedOptimizer(BaseOptimizer):
    """
    AdamW optimizer with masked parameter updates (decoupled weight decay).

    - Matches the interface/usage of AdamMaskedOptimizer.
    - Only indices where mask==True are updated.
    - Decoupled weight decay: L <- L - lr * weight_decay * L (on masked entries).
    - Optional preconditioned Gaussian noise (same as your Adam version).
    - Optional AMSGrad variant.

    References:
      - torch.optim.AdamW: decoupled weight decay (does not accumulate into moments). 
      - Loshchilov & Hutter (AdamW): "Decoupled Weight Decay Regularization".
    """

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
        """
        Perform one AdamW update step using a masked gradient.

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

        # Moments
        g = grad.copy()
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g ** 2)

        # Bias correction
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        # AMSGrad (optional)
        if self.amsgrad:
            self.vmax = np.maximum(self.vmax, v_hat)
            v_denom = np.sqrt(self.vmax) + self.eps
        else:
            v_denom = np.sqrt(v_hat) + self.eps

        precond = 1.0 / v_denom

        # Allocate full-size update (zeros where mask=False)
        update = np.zeros_like(g)

        # ----- Gradient step (Adam piece) -----
        # Only update masked indices
        idx = self.mask
        update[idx] = self.lr * (m_hat[idx] / v_denom[idx])

        # ----- Optional noise (preconditioned, masked) -----
        if self.noise_sigma > 0.0:
            z = np.zeros_like(g)
            z[idx] = self.rng.standard_normal(np.count_nonzero(idx)).astype(self.L.dtype, copy=False)
            update[idx] += (self.noise_sigma * self.lr) * (z[idx] * precond[idx])

        # ----- Decoupled weight decay (AdamW) -----
        if self.weight_decay != 0.0:
            # Decoupled: add lr * wd * param directly to the update (masked entries only)
            update[idx] += self.lr * self.weight_decay * self.L[idx]

        # Apply update (descent)
        self.L -= update

        self.last_update = -update  # what the caller saw as "Adam update" previously
        return self.last_update