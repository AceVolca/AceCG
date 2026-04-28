"""Base interfaces for solver-layer objects."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from ..topology.forcefield import Forcefield


class BaseSolver(ABC):
    """Abstract base class for one-shot solver-layer objects.

    Parameters
    ----------
    forcefield : Forcefield
        Forcefield copied into the solver. Solvers write solved parameters back
        into this private copy.
    logger : object, optional
        Optional scalar logger exposing ``add_scalar``.
    """

    BATCH_SCHEMA: Dict[str, Any] = {}
    RETURN_SCHEMA: Dict[str, Any] = {}

    def __init__(self, forcefield: Forcefield, logger=None):
        """Create a solver with its own forcefield copy.

        Parameters
        ----------
        forcefield : Forcefield
            Source forcefield whose parameter order defines the solver system.
        logger : object, optional
            Optional scalar logger used by concrete solvers.
        """
        self.forcefield = copy.deepcopy(forcefield)
        self.logger = logger

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Return a dict with solver input/output schema descriptions."""
        return {
            "batch": dict(cls.BATCH_SCHEMA),
            "return": dict(cls.RETURN_SCHEMA),
        }

    def get_params(self) -> np.ndarray:
        """Return the current full parameter vector.

        Returns
        -------
        np.ndarray
            One-dimensional parameter vector in forcefield order.
        """
        return self.forcefield.param_array()

    def update_forcefield(self, params: np.ndarray) -> None:
        """Write a full parameter vector back into the owned Forcefield."""
        params = np.asarray(params, dtype=np.float64).reshape(-1)
        expected_shape = self.get_params().shape
        if params.shape != expected_shape:
            raise ValueError(f"parameter shape mismatch: expected {expected_shape}, got {params.shape}")
        self.forcefield.update_params(params)

    @abstractmethod
    def solve(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Solve one canonical statistics batch.

        Parameters
        ----------
        batch : dict
            Solver-specific statistics dictionary.

        Returns
        -------
        dict
            Solver-specific result dictionary, typically containing solved
            parameters and diagnostics.
        """
