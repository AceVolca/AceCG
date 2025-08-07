# AceCG/optimizers/newton_raphson.py
import numpy as np
from .base import BaseOptimizer

class NewtonRaphsonOptimizer(BaseOptimizer):
    """
    Newton-Raphson optimizer with masked updates for selective parameter optimization.

    This optimizer computes the Newton-Raphson update Δλ using the provided gradient and Hessian:

        Δλ = -lr · H⁻¹ · ∇S_rel     (restricted to masked indices)

    Only parameters where mask=True are updated; others remain fixed.
    The optimizer also stores the latest gradient, Hessian, and update vector for logging purposes.

    Attributes
    ----------
    last_grad : np.ndarray or None
        Most recent gradient vector provided to .step().
    last_hessian : np.ndarray or None
        Most recent Hessian matrix provided to .step().
    last_update : np.ndarray or None
        Most recent full update vector returned by .step().
    """
    def __init__(self, L, mask, lr=1e-2):
        """
        Initialize the optimizer.

        Parameters
        ----------
        L : np.ndarray
            Initial parameter vector.
        mask : np.ndarray
            Boolean array specifying which parameters are trainable.
        lr : float, optional
            Learning rate multiplier (default is 1e-2).
        """
        super().__init__(L, mask, lr)
        self.last_grad = None
        self.last_hessian = None
        self.last_update = None

    def step(self, grad: np.ndarray, hessian: np.ndarray) -> np.ndarray:
        """
        Perform one Newton-Raphson update step using the masked gradient and Hessian.

        Parameters
        ----------
        grad : np.ndarray
            Gradient of the loss function with respect to all parameters.
        hessian : np.ndarray
            Hessian matrix of the loss function with respect to all parameters.

        Returns
        -------
        update : np.ndarray
            The full-length update vector with masked values applied.

        Notes
        -----
        - Updates only the parameters where mask is True.
        - The update is applied in-place to self.L.
        - The full update vector (including zeros at masked-out indices) is returned.
        - Internal state stores the most recent gradient, Hessian, and update for logging.
        """
        self.last_grad = grad.copy()
        self.last_hessian = hessian.copy()

        grad_masked = grad[self.mask]
        H_masked = hessian[np.ix_(self.mask, self.mask)]
        update_masked = np.linalg.solve(H_masked, grad_masked)

        update = np.zeros_like(grad)
        update[self.mask] = update_masked
        self.L -= self.lr * update  # in-place update

        self.last_update = -self.lr * update
        return self.last_update
