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
<<<<<<< HEAD
<<<<<<< HEAD
    def param_names(self) -> list[str]:
        """Return a list of parameter names used in the potential."""
        pass

    @abstractmethod
=======
>>>>>>> main
    def dparam_names(self) -> list[str]:
        """Return a list of first derivative method names (used in dUdL)."""
=======
    def param_names(self) -> List[str]:
        """Number of parameters this potential depends on."""
        pass

    @abstractmethod
    def dparam_names(self) -> List[str]:
        """Return a List of first derivative method names (used in dUdL)."""
>>>>>>> 0564f69b2261b942fe8b89bee2498cbbeafdc488
        pass

    @abstractmethod
    def d2param_names(self) -> List[List[str]]:
        """Return a 2D List of second derivative method names (for Hessian)."""
        pass

<<<<<<< HEAD
    def n_params(self) -> int:
        """Number of parameters this potential depends on."""
        n_pa
=======
    @abstractmethod
    def n_params(self) -> int:
        """Number of parameters this potential depends on."""
        pass
>>>>>>> main

    @abstractmethod
    def params(self) -> np.ndarray:
        """Return current parameter values as 1D array."""
        pass

    @abstractmethod
    def set_params(self, new_params: np.ndarray):
        """Update parameters with new values."""
        pass	