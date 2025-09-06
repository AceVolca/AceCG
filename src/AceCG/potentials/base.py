# AceCG/potentials/base.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BasePotential(ABC):
    def __init__(self):
        self._params = None
        self._param_names = None
        self._dparam_names = None
        self._d2param_names = None

    @abstractmethod
    def value(self, r: np.ndarray) -> np.ndarray:
        """Compute potential energy at given distances r."""
        pass

    # Common method
    def param_names(self) -> List[str]:
        """Parameter names of this potential depends on."""
        assert self._param_names is not None
        return self._param_names
    
    def dparam_names(self) -> List[str]:
        """Return a List of first derivative method names (used in dUdL)."""
        assert self._dparam_names is not None
        return self._dparam_names
    
    def d2param_names(self) -> List[List[str]]:
        """Return a 2D List of second derivative method names (for Hessian)."""
        assert self._d2param_names is not None
        return self._d2param_names
    
    def n_params(self) -> int:
        """Number of parameters this potential depends on."""
        assert self._params is not None
        return len(self._params)

    def get_params(self) -> np.ndarray:
        """Return current parameter values as 1D array."""
        assert self._params is not None
        return self._params.copy()

    def set_params(self, new_params: np.ndarray):
        """Update parameters with new values."""
        # if self._params is not None:
        #     assert len(new_params) == len(self._params)
        self._params = new_params.copy()