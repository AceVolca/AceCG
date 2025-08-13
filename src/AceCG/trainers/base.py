# AceCG/trainers/base.py
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, potential, optimizer, beta=None, logger=None):
        self.potential = potential  # dict for analytic backend, or NN wrapper for torch
        self.optimizer = optimizer
        self.beta = beta
        self.logger = logger # tensorboard logger

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