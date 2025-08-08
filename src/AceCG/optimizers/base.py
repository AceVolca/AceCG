# AceCG/optimizers/base.py
from abc import ABC, abstractmethod
import numpy as np

class BaseOptimizer(ABC):
    def __init__(self, L: np.ndarray, mask: np.ndarray, lr: float):
        """
        Parameters
        ----------
        L : np.ndarray
            Initial parameter vector.
        mask : np.ndarray
            Boolean mask array indicating which parameters are optimizable.
        lr : float
            Learning rate.
        """
        self.L = L.copy()
        self.mask = mask.copy()
        self.lr = lr
    
    def set_params(self, L_new: np.ndarray):
        """
        Update the internal parameter vector L.

        Parameters
        ----------
        new_L : np.ndarray
            New parameter values (must match shape of self.L).
        """
        assert L_new.shape == self.L.shape, "Parameter shape mismatch"
        self.L = L_new.copy()

    @abstractmethod
    def step(self, grad: np.ndarray, hessian: np.ndarray = None) -> np.ndarray:
        """
        Apply one optimization step.

        Parameters
        ----------
        grad : np.ndarray
            Gradient vector (same shape as L).
        hessian : np.ndarray, optional
            Hessian matrix (only used by second-order methods).

        Returns
        -------
        update : np.ndarray
            The update vector applied to self.L.
        """
        pass
