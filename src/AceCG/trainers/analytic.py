# AceCG/trainers/analytic.py
import numpy as np
from .base import BaseTrainer
from ..utils.compute import dUdLByFrame, dUdL, d2UdLjdLk_Matrix, dUdLj_dUdLk_Matrix, Hessian, dUdLByBin
from ..utils.ffio import FFParamArray
from .utils import optimizer_accepts_hessian


class REMTrainerAnalytic(BaseTrainer):
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
        dUdL_AA : np.ndarray
            Gradient of CG potential with respect to parameters measured in AA refernce ensemble
        dUdL_CG : np.ndarray
            Gradient of CG potential with respect to parameters measured in CG ensemble
        dSdL : np.ndarray
            Gradient of S_rel with respect to parameters.
        H : np.ndarray
            The Hessian matrix of shape (n_params, n_params) representing ∂²S_rel / ∂λⱼ∂λₖ.
            None if self.optimizer does not require Hessian
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
        if H is not None:
            update = self.optimizer.step(dSdL, hessian=H) # hessian based optimizer
        else:
            update = self.optimizer.step(dSdL)
        self.update_potential(self.optimizer.L)

        # === Logging ===
        if self.logger is not None:
            mask_ratio = np.mean(self.optimizer.mask.astype(float))
            self.logger.add_scalar("REM/mask_ratio", mask_ratio, step_index)
            self.logger.add_scalar("REM/lr", getattr(self.optimizer, "lr", np.nan), step_index)
            self.logger.add_scalar("REM/grad_norm", np.linalg.norm(dSdL), step_index)
            self.logger.add_scalar("REM/update_norm", np.linalg.norm(update), step_index)
            if H is not None:
                self.logger.add_scalar("REM/hessian_cond", np.linalg.cond(H), step_index)

        return dUdL_AA, dUdL_CG, dSdL, H, update


class MSETrainerAnalytic(BaseTrainer):
    """
    Trainer for analytic potentials using Mean Squared Error (MSE) loss
    between atomistic (AA) and coarse-grained (CG) PMFs.

    This trainer:
    1. Computes MSE between PMF_AA and PMF_CG.
    2. Computes dU/dλ over all frames and bins for CG data.
    3. Computes gradient of the MSE with respect to potential parameters.
    4. Performs an optimization step using the provided optimizer.
    5. Updates the potential parameters and logs training metrics.

    Notes
    -----
    - Intended for analytic potentials where gradients can be computed
      explicitly via `dUdLByFrame` / `dUdLByBin`.
    - Optimizer should be 1D-friendly (parameters flattened into a vector).
    - `self.potential` is a mapping from (type1, type2) -> potential object.
    - The `logger` (if provided) is expected to be a TensorBoard-like object.

    See Also
    --------
    dUdLByFrame : Computes per-frame derivatives of U wrt parameters.
    dUdLByBin   : Computes per-bin conditional averages of derivatives.
    """

    def get_params(self) -> np.ndarray:
        """
        Concatenate and return the current parameter vector from all potentials.

        Returns
        -------
        np.ndarray
            1D array containing all potential parameters in the order expected
            by the optimizer. Shape: (n_params,).

        Notes
        -----
        - Uses `FFParamArray(self.potential)` to flatten parameters.
        - Parameter ordering must match `update_potential`.
        """
        return FFParamArray(self.potential)

    def update_potential(self, L_new: np.ndarray):
        """
        Update the potential objects and optimizer with new parameters.

        Parameters
        ----------
        L_new : np.ndarray
            New parameter vector, shape (n_params,).

        Notes
        -----
        - Iterates through `self.potential` in its key order and slices `L_new`
          accordingly using each potential's `n_params()` size.
        - Calls each potential's `set_params` method.
        - Also updates the optimizer's stored parameters to keep them in sync.
        """
        idx = 0
        for pair in self.potential.keys():
            pot = self.potential[pair]
            n = pot.n_params()
            pot.set_params(L_new[idx:idx + n])
            idx += n
        self.optimizer.set_params(L_new)

    def step(
        self,
        pmf_AA: np.ndarray,
        pmf_CG: np.ndarray,
        bin_width: float,
        CG_data: dict,
        CG_bin_idx_frame: np.ndarray,
        step_index: int = 0
    ):
        """
        Perform one MSE-based optimization step.

        Parameters
        ----------
        pmf_AA : np.ndarray
            Reference PMF from all-atom simulation. Shape: (n_bins,).
        pmf_CG : np.ndarray
            Current CG PMF from simulation. Shape: (n_bins,).
        bin_width : float
            Width of each histogram bin in reaction coordinate space.
        CG_data : dict
            CG simulation data, must include:
            - 'dist' : distances per frame per pair (for dUdLByFrame).
            - 'weight' (optional) : frame weights for reweighting.
        CG_bin_idx_frame : np.ndarray
            Array mapping each frame to its bin index. Shape: (n_frames,).
        step_index : int, optional
            Step number for logging (default is 0).

        Returns
        -------
        mse : float
            Mean squared error between pmf_AA and pmf_CG (Euclidean norm).
        dErrdL : np.ndarray
            Gradient of the error with respect to parameters.
            Shape: (n_params,).
        update : np.ndarray
            Parameter update vector returned by the optimizer. Shape: (n_params,).

        Notes
        -----
        - Gradient calculation:
            For each bin:
              dErr/dλ_j(bin) = bin_width * (PMF_CG - PMF_AA)
                               * (⟨dU/dλ_j⟩_CG|s - ⟨dU/dλ_j⟩_CG)
            Sum over all bins to get total dErr/dλ_j.
        - Updates potential and optimizer state.
        - Logs:
            mask ratio, learning rate, MSE, gradient norm, and update norm.
        """
        # === 1. Compute loss ===
        mse = np.linalg.norm(pmf_AA - pmf_CG)

        # === 2. Compute ⟨dU/dλ_j⟩ for CG ===
        dUdL_CG_frame = dUdLByFrame(self.potential, CG_data['dist'])
        dUdL_CG = dUdL(dUdL_CG_frame, CG_data.get('weight'))

        # Per-bin averages
        dUdL_CG_bin, p_CG_bin, dUdL_CG_given_bin = dUdLByBin(
            dUdL_CG_frame,
            CG_bin_idx_frame,
            CG_data.get('weight')
        )

        # === 3. Compute gradient of MSE wrt parameters ===
        idx_set = set(CG_bin_idx_frame)
        dErrdL_bin = {}
        for idx in idx_set:
            dErrdL_bin[idx] = bin_width * (pmf_CG[idx] - pmf_AA[idx]) * \
                              (dUdL_CG_given_bin[idx] - dUdL_CG)

        # Sum over bins (missing bins contribute zero)
        dErrdL = 0
        for idx in range(len(pmf_AA)):
            dErrdL += dErrdL_bin.get(idx, 0)

        # === 4. Optimization step ===
        update = self.optimizer.step(dErrdL)
        self.update_potential(self.optimizer.L)

        # === 5. Logging ===
        if self.logger is not None:
            mask_ratio = np.mean(self.optimizer.mask.astype(float))
            self.logger.add_scalar("MSE/mask_ratio", mask_ratio, step_index)
            self.logger.add_scalar("MSE/lr", getattr(self.optimizer, "lr", np.nan), step_index)
            self.logger.add_scalar("MSE/mse", mse, step_index)
            self.logger.add_scalar("MSE/grad_norm", np.linalg.norm(dErrdL), step_index)
            self.logger.add_scalar("MSE/update_norm", np.linalg.norm(update), step_index)

        return mse, dErrdL, update