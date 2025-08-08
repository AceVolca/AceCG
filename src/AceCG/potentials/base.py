# AceCG/potentials/base.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BasePotential(ABC):
    @abstractmethod
    def value(self, r: np.ndarray) -> np.ndarray:
        """Compute potential energy at given distances r."""
        pass

    @abstractmethod
    def param_names(self) -> List[str]:
        """Number of parameters this potential depends on."""
        pass

    @abstractmethod
    def dparam_names(self) -> List[str]:
        """Return a List of first derivative method names (used in dUdL)."""
        pass

    @abstractmethod
    def d2param_names(self) -> List[List[str]]:
        """Return a 2D List of second derivative method names (for Hessian)."""
        pass

    @abstractmethod
    def n_params(self) -> int:
        """Number of parameters this potential depends on."""
        pass

    @abstractmethod
    def params(self) -> np.ndarray:
        """Return current parameter values as 1D array."""
        pass

    @abstractmethod
    def set_params(self, new_params: np.ndarray):
        """Update parameters with new values."""
        pass	