"""Define staged AceCG optimizer interfaces."""

from abc import ABC, abstractmethod
import numpy as np

class BaseOptimizer(ABC):
    """Base class for masked optimizers used by AceCG trainers.

    Parameters
    ----------
    L : np.ndarray
        Initial full parameter vector.
    mask : np.ndarray
        Boolean array with the same shape as ``L``. ``True`` entries are
        trainable; ``False`` entries remain fixed.
    lr : float
        Learning-rate multiplier used by concrete optimizers.

    Attributes
    ----------
    L : np.ndarray
        Mutable optimizer-side parameter vector.
    mask : np.ndarray
        Boolean active-coordinate mask.
    lr : float
        Current learning rate.
    """
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

    def state_dict(self) -> dict:
        """Return serializable optimizer state (parameters + base scalars).

        Returns
        -------
        dict
            JSON-compatible dictionary containing ``L``, ``mask``, and ``lr``.

        Subclasses override to include moment buffers, step counters, etc.
        """
        return {
            "L": self.L.tolist(),
            "mask": self.mask.tolist(),
            "lr": float(self.lr),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore optimizer state from a dictionary.

        Parameters
        ----------
        state : dict
            Dictionary produced by :meth:`state_dict` on a compatible optimizer.
        """
        self.L = np.asarray(state["L"], dtype=self.L.dtype)
        self.mask = np.asarray(state["mask"], dtype=bool)
        self.lr = float(state["lr"])

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
