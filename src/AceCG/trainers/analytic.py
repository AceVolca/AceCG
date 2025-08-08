# AceCG/trainers/analytic.py
import numpy as np
from .base import BaseREMTrainer
from ..utils.compute import dUdLByFrame, dUdL, d2UdLjdLk_Matrix, dUdLj_dUdLk_Matrix, Hessian
from ..utils.ffio import FFParamArray


def optimizer_accepts_hessian(optimizer) -> bool:
    """
    Check whether the optimizer's `step` method accepts a 'hessian' argument.

    Parameters
    ----------
    optimizer : object
        An optimizer instance with a `.step()` method.

    Returns
    -------
    bool
        True if 'hessian' is a parameter of the .step() method, else False.
    """
    return hasattr(optimizer, 'step') and 'hessian' in optimizer.step.__code__.co_varnames


class REMTrainerAnalytic(BaseREMTrainer):
    """
    REMTrainerAnalytic performs one step of relative entropy minimization (REM)
    using analytic (non-NN) coarse-grained potentials.

    It computes the gradient and optionally the Hessian of the REM objective
    using NumPy-based formulas, then applies the optimizer step. Supports both
    first- and second-order optimizers.

    Attributes
    ----------
    potential : dict
        Mapping from (type1, type2) → BasePotential.
    optimizer : BaseOptimizer
        Optimizer implementing a `.step(grad, hessian)` interface.
    beta : float
        Inverse temperature β = 1/(k_B T).
    logger : SummaryWriter or None
        TensorBoard writer for logging optimization metrics.
    """
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

    def step(self, AA_data, CG_data, step_index: int = 0):
        """
        Perform one REM update step using all-atom and CG sampled data.

        Parameters
        ----------
        AA_data : dict
            Contains 'dist' (pairwise distances) and optionally 'weight'.
        CG_data : dict
            Same format as AA_data, from CG simulation.
        step_index : int
            Current training step index for logging.

        Returns
        -------
        dSdL : np.ndarray
            Gradient of S_rel with respect to parameters.
        update : np.ndarray
            Update vector applied to parameters.
        """
        # === Compute dS/dL ===
        dUdL_AA_frame = dUdLByFrame(self.potential, AA_data['dist'])
        dUdL_CG_frame = dUdLByFrame(self.potential, CG_data['dist'])
        dUdL_AA = dUdL(dUdL_AA_frame, AA_data.get('weight'))
        dUdL_CG = dUdL(dUdL_CG_frame, CG_data.get('weight'))
        dSdL = self.beta * (dUdL_AA - dUdL_CG)

        # === Compute Hessian (if optimizer supports it) ===
        H = None
        if optimizer_accepts_hessian(self.optimizer):
            d2U = d2UdLjdLk_Matrix(self.potential, CG_data['dist'], CG_data.get('weight'))
            dUU = dUdLj_dUdLk_Matrix(dUdL_CG_frame, CG_data.get('weight'))
            H = Hessian(self.beta, d2U, dUU, dUdL_CG)

        # === Optimization Step ===
        update = self.optimizer.step(dSdL, hessian=H)
        self.update_potential(self.optimizer.L)

        # === Logging ===
        if self.logger:
            mask_ratio = np.mean(self.optimizer.mask.astype(float))
            self.logger.add_scalar("REM/mask_ratio", mask_ratio, step_index)
            self.logger.add_scalar("REM/lr", getattr(self.optimizer, "lr", np.nan), step_index)
            self.logger.add_scalar("REM/grad_norm", np.linalg.norm(dSdL), step_index)
            self.logger.add_scalar("REM/update_norm", np.linalg.norm(update), step_index)
            if H is not None:
                self.logger.add_scalar("REM/hessian_cond", np.linalg.cond(H), step_index)

        return dUdL_AA, dUdL_CG, dSdL, H, update
