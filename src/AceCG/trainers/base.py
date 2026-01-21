# AceCG/trainers/base.py
from abc import ABC, abstractmethod
import numpy as np
import copy
from typing import Optional

from ..utils.ffio import FFParamArray

class BaseTrainer(ABC):
    """
    Abstract base class for analytic trainers in AceCG.

    This base class:
    - Deep-copies the provided `potential` and `optimizer` to isolate trainer state.
    - Provides utilities to flatten/update parameters and to clamp to bounds.
    - Defines a unified `.step(...)` interface for subclasses.

    Parameters
    ----------
    potential : dict
        Mapping from (type1, type2) → BasePotential. Will be deep-copied.
    optimizer : BaseOptimizer
        Optimizer instance with attributes like `.L` and methods `step(...)`,
        `set_params(...)`. Will be deep-copied.
    beta : float, optional
        Inverse temperature β, stored for trainers that need it.
    logger : SummaryWriter or None, optional
        Optional logger (TensorBoard-like).

    Attributes
    ----------
    potential : dict
        Deep copy of the input potential dictionary.
    optimizer : BaseOptimizer
        Deep copy of the input optimizer, keeping trainer-local state (L/m/v/mask).
    beta : float or None
        The inverse temperature if provided.
    logger : any or None
        Optional logger.
    _lb, _ub : np.ndarray or None
        Per-parameter lower/upper bounds aligned to `FFParamArray` ordering.
    """
    def __init__(self, potential, optimizer, beta=None, logger=None):
        self.potential = copy.deepcopy(potential)
        self.optimizer = copy.deepcopy(optimizer)
        self.beta = beta
        self.logger = logger
        # per-index bounds (aligned with FFParamArray order)
        self._lb = None
        self._ub = None
        self.scale_factors = None

    def get_params(self) -> np.ndarray:
        """
        Concatenate and return the current parameter vector from all potentials.

        Returns
        -------
        np.ndarray
            1D parameter vector matching optimizer.L. Shape: (n_params,).
        """
        return FFParamArray(self.potential)

    def update_potential(self, L_new: np.ndarray):
        """
        Update `self.potential` and `self.optimizer` with a new parameter vector.

        Parameters
        ----------
        L_new : np.ndarray
            New parameter vector. Shape: (n_params,).

        Notes
        -----
        - Parameter ordering must be consistent with `FFParamArray(self.potential)`.
        """
        idx = 0
        for pair in self.potential.keys():
            pot = self.potential[pair]
            n = pot.n_params()
            pot.set_params(L_new[idx:idx + n])
            idx += n
        self.optimizer.set_params(L_new)

    def set_param_bounds(self, lb: Optional[np.ndarray], ub: Optional[np.ndarray]):
        """
        Register per-parameter bounds (aligned with FFParamArray order).

        Parameters
        ----------
        lb : np.ndarray or None
            Lower bounds or None (no lower bounds).
        ub : np.ndarray or None
            Upper bounds or None (no upper bounds).
        """
        self._lb = None if lb is None else np.asarray(lb, dtype=float)
        self._ub = None if ub is None else np.asarray(ub, dtype=float)
        if self._lb is not None and self._ub is not None:
            assert self._lb.shape == self._ub.shape, "lb/ub must have identical shapes"

    def _apply_bounds(self, L: np.ndarray) -> np.ndarray:
        """
        Project `L` into [lb, ub] if bounds are present; return possibly-clamped copy.

        Parameters
        ----------
        L : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Clamped parameter vector (copy).
        """
        if self._lb is None and self._ub is None:
            return L
        Lc = L.copy()
        if self._lb is not None:
            Lc = np.maximum(Lc, self._lb)
        if self._ub is not None:
            Lc = np.minimum(Lc, self._ub)
        return Lc

    def clamp_and_update(self):
        """
        Clamp `self.optimizer.L` to [lb, ub] (if set) and propagate back to potentials.

        Notes
        -----
        - Calls `self.update_potential(self.optimizer.L)` after clamping
          to keep potential objects in sync with the optimizer state.
        """
        if getattr(self.optimizer, "L", None) is None:
            return
        Lc = self._apply_bounds(self.optimizer.L)
        if not np.shares_memory(Lc, self.optimizer.L) or not np.allclose(Lc, self.optimizer.L):
            self.optimizer.set_params(Lc)
        self.update_potential(self.optimizer.L)

    @abstractmethod
    def step(self, AA_data, CG_data, step_index: int = 0):
        """
        Perform one optimization step (interface to be specialized by subclasses).

        Parameters
        ----------
        AA_data : Any
            Subclass-specific data (e.g., dict with 'dist', 'weight').
        CG_data : Any
            Subclass-specific data (e.g., dict with 'dist', 'weight').
        step_index : int
            Step index for logging.

        Returns
        -------
        Any
            Subclass-specific tuple (e.g., gradients, Hessian, update).
        """
        pass

    @abstractmethod
    def d_dz(self):
        pass

    def get_scaled_potential(self):

        the_potential = cp.deepcopy(self.potential)
        i = 0
        items = the_potential.items()
        assert len(scale_factor)  == len(items)
        for pair, pot in items:
            pot = the_potential.get_modified_potential(scale_factors[i])
            i += 1
        return the_potential

    def set_scale_factors(self, scale_factors):
        items = self.potential.items()
        assert len(scale_factors) == len(items)
        self.scale_factors = scale_factors

    def unset_scale_factors(self):

        self.scale_factors = None
