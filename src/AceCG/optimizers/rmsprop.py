import numpy as np
from .base import BaseOptimizer

class RMSpropMaskedOptimizer(BaseOptimizer):
    """
    RMSprop optimizer with masked parameter updates.

    - Matches AdamMaskedOptimizer-style API and return value.
    - Only indices where mask==True are updated.
    - Weight decay is L2-style (added to the gradient), like torch.optim.RMSprop.
    - Optional momentum buffer.
    - Optional 'centered' variant (uses variance estimate).
    - Optional preconditioned Gaussian noise on masked entries.

    Args
    ----
    L : np.ndarray
        Initial parameter vector.
    mask : np.ndarray[bool]
        Update mask (True = train this coordinate).
    lr : float
        Learning rate.
    alpha : float
        Smoothing constant for squared-grad EMA (PyTorch default 0.99).
    eps : float
        Numerical stability term added to denominator.
    weight_decay : float
        L2 penalty coefficient (added into the gradient).
    momentum : float
        Momentum coefficient (0 disables momentum).
    centered : bool
        If True, uses sqrt(E[g^2] - (E[g])^2 + eps) in the denominator.
    noise_sigma : float
        Std of optional Gaussian noise (preconditioned, masked). 0 disables.
    seed : int or None
        RNG seed for noise.

    Returns from step()
    -------------------
    update : np.ndarray
        The full update vector that was SUBTRACTED from parameters
        (zeros for unmasked entries), same as your Adam code.
    """

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
        super().__init__(L, mask, lr)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.momentum = float(momentum)
        self.centered = bool(centered)

        self.square_avg = np.zeros_like(L)           # E[g^2]
        self.momentum_buffer = np.zeros_like(L) if self.momentum > 0.0 else None
        self.grad_avg = np.zeros_like(L) if self.centered else None  # E[g]

        self.noise_sigma = float(noise_sigma)
        self.rng = np.random.default_rng(seed)

        self.last_update = None

    def step(self, grad: np.ndarray) -> np.ndarray:
        """
        Perform one RMSprop update step using a masked gradient.

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

        # ----- L2 weight decay (in-gradient), masked -----
        if self.weight_decay != 0.0:
            g[idx] = g[idx] + self.weight_decay * self.L[idx]

        # ----- EMA of squared gradients -----
        # square_avg = alpha * square_avg + (1 - alpha) * g^2
        self.square_avg = self.alpha * self.square_avg + (1.0 - self.alpha) * (g * g)

        # ----- Centered variant (estimate variance) -----
        if self.centered:
            # grad_avg = alpha * grad_avg + (1 - alpha) * g
            self.grad_avg = self.alpha * self.grad_avg + (1.0 - self.alpha) * g
            # denom = sqrt(square_avg - grad_avg^2) + eps
            var = self.square_avg - (self.grad_avg * self.grad_avg)
            # numeric guard: variance can't be negative due to rounding
            var = np.maximum(var, 0.0)
            denom = np.sqrt(var) + self.eps
        else:
            # denom = sqrt(square_avg) + eps
            denom = np.sqrt(self.square_avg) + self.eps

        precond = 1.0 / denom

        # ----- Compute update (masked only) -----
        update = np.zeros_like(g)

        if self.momentum > 0.0:
            # momentum_buffer = momentum * buffer + g / denom
            if self.momentum_buffer is None:
                self.momentum_buffer = np.zeros_like(g)
            self.momentum_buffer[idx] = (
                self.momentum * self.momentum_buffer[idx] + (g[idx] / denom[idx])
            )
            update[idx] = self.lr * self.momentum_buffer[idx]
        else:
            # plain RMSprop step: lr * g / denom
            update[idx] = self.lr * (g[idx] / denom[idx])

        # ----- Optional preconditioned noise (masked) -----
        if self.noise_sigma > 0.0:
            z = np.zeros_like(g)
            z[idx] = self.rng.standard_normal(np.count_nonzero(idx)).astype(self.L.dtype, copy=False)
            update[idx] += (self.noise_sigma * self.lr) * (z[idx] * precond[idx])

        # ----- Apply (descent) -----
        self.L -= update

        self.last_update = -update
        return self.last_update