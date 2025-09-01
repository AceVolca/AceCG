# AceCG/trainers/analytic.py
import numpy as np
from .base import BaseTrainer
from ..utils.compute import dUdLByFrame, dUdL, d2UdLjdLk_Matrix, dUdLj_dUdLk_Matrix, Hessian, dUdLByBin
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

        # === clamp & sync ===
        self.clamp_and_update()

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

    def step(
        self,
        pmf_AA: np.ndarray,
        pmf_CG: np.ndarray,
        CG_data: dict,
        CG_bin_idx_frame: np.ndarray,
        weighted_gauge = False,
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
        CG_data : dict
            CG simulation data, must include:
            - 'dist' : distances per frame per pair (for dUdLByFrame).
            - 'weight' (optional) : frame weights for reweighting.
        CG_bin_idx_frame : np.ndarray
            Array mapping each frame to its bin index. Shape: (n_frames,).
        weighted_gauge : bool, optional
            Whether to apply weighted gauge min{Σ_s q(s)*[PMF_CG(s) - PMF_AA(s)]^2}
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
              dErr/dλ_j(bin) = (PMF_CG(s) - PMF_AA(s))
                               * (⟨dU/dλ_j⟩_CG|s - ⟨dU/dλ_j⟩_CG)
            Sum over all bins to get total dErr/dλ_j.
        - PMF_CG is adjusted to min{Σ_s [PMF_CG(s) - PMF_AA(s)]^2}
                                            or min{Σ_s q(s)*[PMF_CG(s) - PMF_AA(s)]^2}
        - Updates potential and optimizer state.
        - Logs:
            mask ratio, learning rate, MSE, gradient norm, and update norm.
        """
        # === Compute ⟨dU/dλ_j⟩ for CG ===
        dUdL_CG_frame = dUdLByFrame(self.potential, CG_data['dist'])
        dUdL_CG = dUdL(dUdL_CG_frame, CG_data.get('weight'))

        # Per-bin averages
        dUdL_CG_bin, p_CG_bin, dUdL_CG_given_bin = dUdLByBin(
            dUdL_CG_frame,
            CG_bin_idx_frame,
            CG_data.get('weight')
        )

        # === Adjust pmf_CG ===
        if weighted_gauge:
            # min{Σ_s q(s)*[PMF_CG(s) - PMF_AA(s)]^2}
            c = (pmf_CG - pmf_AA) @ np.array([p_CG_bin.get(i, 0) for i in range(len(pmf_CG))])
        else:
            # min{Σ_s [PMF_CG(s) - PMF_AA(s)]^2}
            c = np.mean(pmf_CG - pmf_AA)
        pmf_CG_shifted = pmf_CG - c

        # === Compute loss ===
        mse = np.linalg.norm(pmf_AA - pmf_CG_shifted)

        # === Compute gradient of MSE wrt parameters ===
        idx_set = set(CG_bin_idx_frame)
        dErrdL_bin = {}
        for idx in idx_set:
            dErrdL_bin[idx] = (pmf_CG_shifted[idx] - pmf_AA[idx]) * (dUdL_CG_given_bin[idx] - dUdL_CG)

        # Sum over bins (missing bins contribute zero)
        dErrdL = 0
        for idx in range(len(pmf_AA)):
            dErrdL += dErrdL_bin.get(idx, 0)

        # === Optimization step ===
        update = self.optimizer.step(dErrdL)

        # === clamp & sync ===
        self.clamp_and_update()

        # === Logging ===
        if self.logger is not None:
            mask_ratio = np.mean(self.optimizer.mask.astype(float))
            self.logger.add_scalar("MSE/mask_ratio", mask_ratio, step_index)
            self.logger.add_scalar("MSE/lr", getattr(self.optimizer, "lr", np.nan), step_index)
            self.logger.add_scalar("MSE/mse", mse, step_index)
            self.logger.add_scalar("MSE/grad_norm", np.linalg.norm(dErrdL), step_index)
            self.logger.add_scalar("MSE/update_norm", np.linalg.norm(update), step_index)

        return mse, dErrdL, update
