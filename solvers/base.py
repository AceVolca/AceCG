"""Base interfaces for force-matching solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseSolver(ABC):
    """Abstract solver interface for FM-style matrix and iterative solvers."""

    @abstractmethod
    def step(self, batch: Dict[str, Any]) -> None:
        """Consume one batch/frame and update internal accumulators."""

    @abstractmethod
    def finalize(self) -> Dict[str, Any]:
        """Finalize and return solver outputs."""

    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Return current full parameter vector."""

    @abstractmethod
    def update_potential(self, params: np.ndarray) -> None:
        """Propagate a full parameter vector back to interaction potentials."""
