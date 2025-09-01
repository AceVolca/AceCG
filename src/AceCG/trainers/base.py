# AceCG/trainers/base.py
from abc import ABC, abstractmethod
import numpy as np

from ..utils.ffio import FFParamArray

class BaseTrainer(ABC):
    def __init__(self, potential, optimizer, beta=None, logger=None):
        self.potential = potential
        self.optimizer = optimizer
        self.beta = beta
        self.logger = logger
        # per-index bounds (aligned with FFParamArray order)
        self._lb = None
        self._ub = None
    
    def get_params(self) -> np.ndarray:
        """
        Concatenate and return the current parameter vector from all potentials.

        Returns
        -------
        np.ndarray
            1D parameter vector matching optimizer.L.
        """
        return FFParamArray(self.potential)

    def update_potential(self, L_new):
        """
        Update self.potential based on new parameters.
        Update parameters stored in the optimizer.

        Parameters
        ----------
        L_new : np.ndarray
            New parameter vector.
        """
        idx = 0
        for pair in self.potential.keys():
            pot = self.potential[pair]
            n = pot.n_params()
            pot.set_params(L_new[idx:idx + n])
            idx += n
        self.optimizer.set_params(L_new)

    # set bounds as arrays (shape = n_params)
    def set_param_bounds(self, lb: np.ndarray, ub: np.ndarray):
        """
        Register per-parameter bounds (aligned with FFParamArray order).
        """
        self._lb = None if lb is None else np.asarray(lb, dtype=float)
        self._ub = None if ub is None else np.asarray(ub, dtype=float)
        if self._lb is not None and self._ub is not None:
            assert self._lb.shape == self._ub.shape

    # clamp helper
    def _apply_bounds(self, L: np.ndarray) -> np.ndarray:
        """
        Project L into [lb, ub] if bounds are present. No-op if not set.
        """
        if self._lb is None and self._ub is None:
            return L
        Lc = L.copy()
        if self._lb is not None:
            Lc = np.maximum(Lc, self._lb)
        if self._ub is not None:
            Lc = np.minimum(Lc, self._ub)
        return Lc

    # clamp optimizer.L in-place and sync potentials
    def clamp_and_update(self):
        """
        Clamp self.optimizer.L to [lb,ub] and push to potentials.
        """
        if getattr(self.optimizer, "L", None) is None:
            return
        Lc = self._apply_bounds(self.optimizer.L)
        if not np.shares_memory(Lc, self.optimizer.L) or not np.allclose(Lc, self.optimizer.L):
            self.optimizer.set_params(Lc)
        # update_potential(L_new) is defined
        self.update_potential(self.optimizer.L)

    @abstractmethod
    def step(self, AA_data, CG_data, step_index: int = 0):
        """
        Perform one REM optimization step given AA and CG data.

        Parameters
        ----------
        AA_data : dict
            Dictionary with keys like 'dist', 'weight' for all-atom reference data.
        CG_data : dict
            Dictionary with keys like 'dist', 'weight' for coarse-grained samples.
        step_index : int
            Current optimization step index (used for logging).

        Returns
        -------
        dSdL : np.ndarray
            Gradient of relative entropy with respect to parameters.
        update : np.ndarray
            Update applied to parameter vector.
        """
        pass