# AceCG/trainers/analytic.py
import numpy as np
import copy
from typing import List, Sequence, Tuple, Any

from .base import BaseTrainer
from ..utils.compute import dUdLByFrame, dUdL, d2UdLjdLk_Matrix, dUdLj_dUdLk_Matrix, Hessian, dUdLByBin
from ..utils.compute import UByFrame, U_total, UByBin
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
            Gradient of CG potential w.r.t. parameters measured in the AA reference ensemble.
        dUdL_CG : np.ndarray
            Gradient of CG potential w.r.t. parameters measured in the CG ensemble.
        dSdL : np.ndarray
            Gradient of S_rel w.r.t. parameters.
        H : np.ndarray or None
            Hessian matrix of shape (n_params, n_params) representing ∂²S_rel / ∂λⱼ∂λₖ,
            or None if the optimizer does not require Hessian information.
        update : np.ndarray
            Update vector applied to parameters.
        """
        # === Compute dS/dL ===
        if self.scale_factors is not None:
            the_potential = self.get_scaled_potential(self.scale_factors)
        else:
            the_potential = self.potential

        dUdL_AA_frame = dUdLByFrame(the_potential, AA_data['dist'])
        dUdL_CG_frame = dUdLByFrame(the_potential, CG_data['dist'])
        dUdL_AA = dUdL(dUdL_AA_frame, AA_data.get('weight'))
        dUdL_CG = dUdL(dUdL_CG_frame, CG_data.get('weight'))
        dSdL = self.beta * (dUdL_AA - dUdL_CG)

        # === Compute Hessian (if optimizer supports it) ===
        H = None
        if optimizer_accepts_hessian(self.optimizer):
            d2U_AA = d2UdLjdLk_Matrix(the_potential, AA_data['dist'], AA_data.get('weight'))
            d2U_CG = d2UdLjdLk_Matrix(the_potential, CG_data['dist'], CG_data.get('weight'))
            dUU_CG = dUdLj_dUdLk_Matrix(dUdL_CG_frame, CG_data.get('weight'))
            H = Hessian(self.beta, d2U_AA, d2U_CG, dUU_CG, dUdL_CG)

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

    def d_dz(self, AA_data, CG_data, step_index: int = 0):

        U_AA = U_total(self.potential, AA_data['dist'])
        U_CG = U_total(self.potential, CG_data['dist'])

        return self.beta * np.subtract(U_AA, U_CG)

    def get_gradients(self, AA_data, CG_data):

        # === Compute dS/dL ===
        if self.scale_factors is not None:
            the_potential = self.get_scaled_potential(self.scale_factors)
        else:
            the_potential = self.potential

        dUdL_AA_frame = dUdLByFrame(the_potential, AA_data['dist'])
        dUdL_CG_frame = dUdLByFrame(the_potential, CG_data['dist'])
        dUdL_AA = dUdL(dUdL_AA_frame, AA_data.get('weight'))
        dUdL_CG = dUdL(dUdL_CG_frame, CG_data.get('weight'))
        dSdL = self.beta * (dUdL_AA - dUdL_CG)
        return dSdL


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
        if self.scale_factors is not None:
            the_potential = self.get_scaled_potential()
        else:
            the_potential = self.potential

        dUdL_CG_frame = dUdLByFrame(the_potential, CG_data['dist'])
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

    def get_gradients(self, pmf_AA: np.ndarray,
        pmf_CG: np.ndarray,
        CG_data: dict,
        CG_bin_idx_frame: np.ndarray,
        weighted_gauge = False):
        # === Compute ⟨dU/dλ_j⟩ for CG ===
        if self.scale_factors is not None:
            the_potential = self.get_scaled_potential()
        else:
            the_potential = self.potential

        dUdL_CG_frame = dUdLByFrame(the_potential, CG_data['dist'])
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

        return dErrdL


    def d_dz(self,pmf_AA: np.ndarray,
        pmf_CG: np.ndarray,
        CG_data: dict,
        CG_bin_idx_frame: np.ndarray,
        weighted_gauge = False, step_index: int = 0):

        """
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

        """

        # === Compute ⟨U⟩ for CG ===
        U_CG_frame = UByFrame(self.potential, CG_data['dist'])

        U_CG = U_total(U_CG_frame, CG_data.get('weight'))

        # Per-bin averages
        U_CG_bin, p_CG_bin, U_CG_given_bin = UByBin(
            U_CG_frame,
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
        Err_bin = {}
        for idx in idx_set:
            Err_bin[idx] = (pmf_CG_shifted[idx] - pmf_AA[idx]) * (U_CG_given_bin[idx] - U_CG)

        # Sum over bins (missing bins contribute zero)
        Err = 0
        for idx in range(len(pmf_AA)):
            Err += Err_bin.get(idx, 0)

        return Err



class MultiTrainerAnalytic(BaseTrainer):
    """
    Combine multiple analytic trainers and optimize with a weighted, combined update.

    This meta-trainer coordinates several sub-trainers (e.g., REMTrainerAnalytic,
    MSETrainerAnalytic). Each sub-trainer runs its own `.step(...)` using its
    own optimizer to produce an update vector. `MultiTrainerAnalytic` then forms
    a **weighted linear combination** of these update vectors and applies it to
    the meta-trainer's parameter vector `self.optimizer.L`, followed by
    `clamp_and_update()` to propagate parameters to `self.potential`.

    Typical usage
    -------------
    - Each sub-trainer focuses on a component objective (e.g., REM, MSE-to-PMF).
    - You pass the per-trainer positional-argument tuples via `param_list`.
    - You specify which returns to keep from each sub-trainer via `return_idx_list`.
    - The final combined update is appended to the returned list.

    Parameters
    ----------
    potential : dict
        Mapping from (type1, type2) → BasePotential (shared across all trainers).
    optimizer : BaseOptimizer
        Optimizer attached to this meta-trainer for maintaining the canonical
        parameter vector `L`. Sub-trainers use their own optimizers internally;
        this optimizer is used to apply the **final combined update**.
    trainer_list : list[BaseTrainer]
        Initialized trainer instances to be called in sequence. They will be
        deep-copied on construction to avoid side-effects.
    weight_array : np.ndarray
        1D array of shape (n_trainers,) giving the linear weights applied to
        each sub-trainer's **update** vector. The final update is:

            final_update = weights @ np.stack(updates, axis=0)

    beta : float, optional
        Inverse temperature β used by some trainers; stored for interface consistency.
    logger : SummaryWriter or None, optional
        TensorBoard-like logger. Not used directly here, but kept for API symmetry.

    Attributes
    ----------
    trainers : list[BaseTrainer]
        Deep copies of the trainers provided in `trainer_list`.
    weights : np.ndarray
        Copy of `weight_array` used in the weighted combination of sub-updates.
    _lb, _ub : np.ndarray or None
        Optional lower/upper bounds used by `clamp_and_update()` (inherited).

    Raises
    ------
    AssertionError
        If `trainer_list` and `weight_array` lengths mismatch; if `weight_array`
        is not 1D; or if the optimizers/updates are shape-incompatible.

    Notes
    -----
    - This class does **not** manage each sub-trainer's learning rate or mask.
    - All trainers must be compatible with the shared `potential` structure and
      consistent parameter ordering.
    - The combined signal is an **update-level** composition (not a direct sum
      of losses/gradients), enabling mixtures of first-/second-order trainers.
    """

    def __init__(
        self,
        potential,
        optimizer,
        trainer_list: Sequence[BaseTrainer],
        weight_array: np.ndarray,
        beta: float = None,
        logger=None
    ):
        # the optimizer is just for code consistency and the access to L
        # the optimizer in each pass-in trainer does the training together
        super().__init__(potential, optimizer, beta, logger)

        # ---- Assertions on inputs ----
        assert isinstance(trainer_list, (list, tuple)) and len(trainer_list) > 0, \
            "trainer_list must be a non-empty list/tuple of trainers"
        assert isinstance(weight_array, np.ndarray), "weight_array must be a NumPy array"
        assert weight_array.ndim == 1, "weight_array must be 1D"
        assert len(trainer_list) == weight_array.shape[0], \
            "each trainer should have exactly one corresponding weight"
        for i, tr in enumerate(trainer_list):
            assert isinstance(tr, BaseTrainer), f"trainer_list[{i}] is not a BaseTrainer"

        self.trainers: List[BaseTrainer] = copy.deepcopy(list(trainer_list))
        self.weights = np.array(weight_array, dtype=float)
        self._lb = None
        self._ub = None

        # Optional: sanity check presence of meta-optimizer L
        assert hasattr(self.optimizer, "L"), "Meta-optimizer must expose attribute `.L`"

    def step(self, param_list: Sequence[Tuple[Any, ...]], return_idx_list: Sequence[Sequence[int]]):
        """
        Run one combined step over all sub-trainers and apply a weighted update.

        Each sub-trainer is called as:
            result_i = trainers[i].step(*param_list[i])

        We collect:
        - the **last** element of each result (assumed to be that trainer's update),
        - any additional elements specified by `return_idx_list[i]`.

        Then compute the final update as a weighted combination of sub-updates:
            final_update = weights @ stack(updates)      # (n_params,)

        Apply this to `self.optimizer.L`, append `final_update` to the returned
        list, and finally call `clamp_and_update()` to synchronize parameters.

        Parameters
        ----------
        param_list : list[tuple]
            A list with length equal to `len(self.trainers)`. Entry i is the
            positional-argument tuple passed to `self.trainers[i].step(...)`.
        return_idx_list : list[list[int]]
            A list of index lists. For trainer i, we keep `result[i][j]` for
            each `j` in `return_idx_list[i]`. These are appended to the final
            return list in order. The **last** element of each trainer’s result
            is assumed to be its `update` vector and is *always* consumed to
            build the combined update (you do not need to list it here).

        Returns
        -------
        selected_and_update : list
            A flat list of selected items from each trainer’s returns (in the
            order specified by `return_idx_list`), with the **final combined
            update** appended at the end. The combined update has shape
            `(n_params,)` and has already been applied to `self.optimizer.L`.

        Raises
        ------
        AssertionError
            If lengths/shapes/types are inconsistent, or if updates returned by
            sub-trainers do not match the meta-optimizer parameter shape.

        Notes
        -----
        - All sub-trainers must return an update vector as the **last** item
          from their `.step(...)`.
        - The stacking/weighting assumes all updates share the same shape as
          `self.optimizer.L`.
        """
        # ---- Assertions on call-time inputs ----
        assert isinstance(param_list, (list, tuple)) and len(param_list) == len(self.trainers), \
            "param_list length must match the number of trainers"
        assert isinstance(return_idx_list, (list, tuple)) and len(return_idx_list) == len(self.trainers), \
            "return_idx_list length must match the number of trainers"

        updates, final_return = [], []

        for i, trainer in enumerate(self.trainers):
            # Each param_list[i] must be a tuple of positional args
            assert isinstance(param_list[i], (list, tuple)), \
                f"param_list[{i}] must be a tuple/list of positional arguments"
            result = trainer.step(*param_list[i])

            # `result` must be indexable and contain at least one element (update at -1)
            assert hasattr(result, "__getitem__") and len(result) >= 1, \
                f"trainer {i} returned an invalid result"

            upd = np.asarray(result[-1])
            # Check update shape against meta-optimizer L
            assert upd.shape == np.asarray(self.optimizer.L).shape, \
                f"update shape mismatch for trainer {i}: {upd.shape} vs {np.asarray(self.optimizer.L).shape}"
            updates.append(np.copy(upd))

            # Select extra returns by indices
            assert isinstance(return_idx_list[i], (list, tuple)), \
                f"return_idx_list[{i}] must be a list/tuple of indices"
            for j in return_idx_list[i]:
                assert isinstance(j, int) and (-len(result) <= j < len(result)), \
                    f"invalid return index {j} for trainer {i} with {len(result)} returns"
                final_return.append(result[j])

        # final update = linear combination of per-trainer updates
        U = np.stack(updates, axis=0)           # (n_trainers, n_params)
        assert U.shape[0] == self.weights.shape[0], \
            "updates count must match weights length"
        final_update = self.weights @ U         # (n_params,)
        assert final_update.shape == np.asarray(self.optimizer.L).shape, \
            "final_update shape mismatch with meta-optimizer .L"

        # Apply to meta-optimizer parameters
        self.optimizer.L += final_update
        final_return.append(final_update)

        # === clamp & sync meta parameters ===
        self.clamp_and_update()

        # === sync sub-trainers' potentials with the new global L ===
        # (Ensure consistent L across sub-trainers after meta update)
        for tr in self.trainers:
            tr.update_potential(self.optimizer.L)

        return final_return

    def set_lrs(self, lrs: Sequence[float]):
        """
        Set per-trainer learning rates.

        Parameters
        ----------
        lrs : sequence of float
            Learning rates for each sub-trainer, length must equal len(self.trainers).

        Raises
        ------
        AssertionError
            If length of `lrs` mismatches the number of trainers.
        """
        assert isinstance(lrs, (list, tuple, np.ndarray)) and len(lrs) == len(self.trainers), \
            "lrs length must match the number of trainers"
        for i, trainer in enumerate(self.trainers):
            trainer.optimizer.lr = float(lrs[i])
