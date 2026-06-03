# AceCG/trainers/base.py
from abc import ABC, abstractmethod
import numpy as np
import copy
from typing import Any, List, Optional, Tuple, Dict, NamedTuple

from ..potentials.base import IteratePotentials
from ..topology.forcefield import Forcefield
from ..topology.types import InteractionKey
from ..potentials.base import BasePotential
from ..optimizers.base import BaseOptimizer

class BaseTrainer(ABC):
    """Abstract base class shared by AceCG analytic trainers.

    ``BaseTrainer`` owns a private copy of a :class:`~AceCG.topology.Forcefield`
    and optimizer so that trainer experiments do not mutate the caller's
    original objects. Subclasses implement :meth:`step` and consume a
    trainer-specific statistics batch.

    Parameters
    ----------
    forcefield : Forcefield
        Forcefield container whose ordered parameters define the optimization
        vector. The object is deep-copied during initialization.
    optimizer : BaseOptimizer
        Optimizer with a parameter vector and active-parameter mask matching
        ``forcefield``. The optimizer is deep-copied during initialization.
    beta : float, optional
        Inverse temperature used by statistical trainers such as REM and CDREM.
        Trainers that do not need a thermodynamic beta may leave it as ``None``.
    logger : object, optional
        Optional logger exposing ``add_scalar(name, value, step)``. TensorBoard
        writers and lightweight logger adapters both work.

    Attributes
    ----------
    forcefield : Forcefield
        Private forcefield copy updated after every accepted optimizer step.
    optimizer : BaseOptimizer
        Private optimizer copy that stores the current parameter vector,
        learning-rate state, and active mask.
    beta : float or None
        Stored inverse temperature.
    logger : object or None
        Optional scalar logger.
    """
    def __init__(self,
                 forcefield: Forcefield,
                 optimizer: BaseOptimizer,
                 beta: Optional[float] = None,
                 logger=None):
        self.forcefield = copy.deepcopy(forcefield)
        self.optimizer = copy.deepcopy(optimizer)
        self.beta = beta
        self.logger = logger

    def get_params(self) -> np.ndarray:
        """
        Concatenate and return the current parameter vector from all potentials.

        Returns
        -------
        np.ndarray
            1D parameter vector matching optimizer.L. Shape: (n_params,).
        """
        return self.forcefield.param_array()

    def update_forcefield(self, L_new: np.ndarray):
        """Synchronize the trainer forcefield and optimizer to a parameter vector.

        Parameters
        ----------
        L_new : np.ndarray
            Full ordered parameter vector. Its shape must match the current
            forcefield parameter array and optimizer state.
        """
        self.forcefield.update_params(L_new)
        self.optimizer.set_params(L_new)

    def clamp_and_update(self):
        """
        Clamp `self.optimizer.L` to [lb, ub] (if set) and propagate back to potentials.

        Notes
        -----
        - Calls `self.update_forcefield(self.optimizer.L)` after clamping
          to keep potential objects in sync with the optimizer state.
        """
        if getattr(self.optimizer, "L", None) is None:
            return
        Lc = self.forcefield.apply_bounds(self.optimizer.L)
        if not np.shares_memory(Lc, self.optimizer.L) or not np.allclose(Lc, self.optimizer.L):
            self.optimizer.set_params(Lc)
        self.update_forcefield(self.optimizer.L)

    def _optimizer_step(
        self,
        grad: np.ndarray,
        hessian: Optional[np.ndarray] = None,
        *,
        apply_update: bool,
        mask_override: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply one optimizer update and re-sync the forcefield to its bounds.

        Returns a zero vector when ``apply_update`` is False. ``hessian`` is
        forwarded to optimizers that accept one; on a Hessian ``LinAlgError``
        the step falls back to a masked least-squares Newton solve.
        ``mask_override`` temporarily swaps the optimizer's active mask for the
        duration of the step (restored afterwards).
        """
        if not apply_update:
            return np.zeros_like(grad)
        original_mask = None
        if mask_override is not None:
            original_mask = np.asarray(self.optimizer.mask, dtype=bool).copy()
            self.optimizer.mask = np.asarray(mask_override, dtype=bool).copy()
        try:
            try:
                if self.optimizer_accepts_hessian():
                    update = self.optimizer.step(grad, hessian=hessian)
                else:
                    update = self.optimizer.step(grad)
            except np.linalg.LinAlgError:
                if (
                    hessian is None
                    or not hasattr(self.optimizer, "mask")
                    or not hasattr(self.optimizer, "lr")
                ):
                    raise
                mask = np.asarray(self.optimizer.mask, dtype=bool)
                grad_masked = np.asarray(grad, dtype=np.float64)[mask]
                h_masked = np.asarray(hessian, dtype=np.float64)[np.ix_(mask, mask)]
                step_masked, *_ = np.linalg.lstsq(h_masked, grad_masked, rcond=None)
                step = np.zeros_like(grad, dtype=np.float64)
                step[mask] = step_masked
                lr = float(self.optimizer.lr)
                self.optimizer.L = np.asarray(self.optimizer.L, dtype=np.float64) - lr * step
                if hasattr(self.optimizer, "last_grad"):
                    self.optimizer.last_grad = np.asarray(grad, dtype=np.float64).copy()
                if hasattr(self.optimizer, "last_hessian"):
                    self.optimizer.last_hessian = np.asarray(hessian, dtype=np.float64).copy()
                if hasattr(self.optimizer, "last_update"):
                    self.optimizer.last_update = -lr * step
                update = -lr * step
            self.clamp_and_update()
            return np.asarray(update, dtype=np.float64)
        finally:
            if original_mask is not None:
                self.optimizer.mask = original_mask

    def select_optimizer_gradient(
        self,
        masked_grad: np.ndarray,
        *,
        unmasked_grad: Optional[np.ndarray],
        mode: str = "masked",
        outside_aux_weight: float = 1.0,
        allow_unmasked: bool = False,
    ) -> np.ndarray:
        """Choose the gradient passed to the optimizer when a coordinate mask is active.

        ``masked_grad`` is the normal REM/CDREM statistic. ``unmasked_grad``
        is an optional auxiliary channel from the same frames before applying
        the coordinate mask; it should not be confused with parameter masks.
        """
        mode_norm = str(mode).strip().lower()
        masked = np.asarray(masked_grad, dtype=float)
        if mode_norm == "masked":
            return masked
        if unmasked_grad is None:
            raise KeyError(
                f"optimizer_gradient_mode={mode_norm!r} requires unmasked gradient statistics."
            )
        unmasked = np.asarray(unmasked_grad, dtype=float)
        if unmasked.shape != masked.shape:
            raise ValueError(
                "unmasked optimizer gradient shape mismatch: "
                f"masked={masked.shape}, unmasked={unmasked.shape}."
            )
        if mode_norm == "hybrid_aux":
            weight = float(outside_aux_weight)
            if weight < 0.0:
                raise ValueError("outside_aux_weight must be nonnegative.")
            return masked + weight * (unmasked - masked)
        if mode_norm == "unmasked":
            if not allow_unmasked:
                raise ValueError(
                    "optimizer_gradient_mode='unmasked' requires "
                    "training.allow_unmasked_optimizer_gradient = true."
                )
            return unmasked
        raise ValueError(
            "optimizer_gradient_mode must be 'masked', 'hybrid_aux', or 'unmasked', "
            f"got {mode!r}."
        )

    def get_param_names(self) -> List[str]:
        """Return ordered human-readable names for all forcefield parameters.

        Returns
        -------
        list[str]
            Names in the same order as :meth:`get_params`, formatted as
            ``"<interaction-label>[<local-index>]"``.
        """
        names = []
        for key, pot in IteratePotentials(self.forcefield):
            prefix = key.label()
            for i in range(pot.n_params()):
                names.append(f"{prefix}[{i}]")
        return names

    def get_param_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower and upper bounds for the global parameter vector.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(lower_bounds, upper_bounds)`` arrays ordered like
            :meth:`get_params`.
        """
        return self.forcefield.param_bounds

    def get_interaction_labels(self) -> List[str]:
        """Return ordered labels for forcefield interactions.

        Returns
        -------
        list[str]
            Labels generated from each :class:`InteractionKey` in forcefield
            order.
        """
        return [key.label() for key in self.forcefield.keys()]

    def n_total_params(self) -> int:
        """Return the total number of parameters in the owned forcefield.

        Returns
        -------
        int
            Full parameter count before applying the optimizer mask.
        """
        return sum(p.n_params() for _, p in IteratePotentials(self.forcefield))

    def active_interaction_mask(self) -> dict:
        """Return the interaction-level mask induced by the optimizer mask.

        Returns
        -------
        dict
            Mapping ``InteractionKey`` objects to booleans. A key is ``True``
            when at least one of its parameters is active in the optimizer's
            parameter-level mask.
        """
        n = self.n_total_params()
        mask = getattr(self.optimizer, "mask", None)
        l2 = np.asarray(mask, dtype=bool) if mask is not None else np.ones(n, dtype=bool)
        if l2.shape != (n,):
            raise ValueError(
                f"optimizer.mask shape mismatch: expected {(n,)}, got {l2.shape}"
            )
        return self.forcefield.derive_l1_mask(l2)

    def is_optimization_linear(self) -> bool:
        """Report whether all active parameters enter their potentials linearly.

        Returns
        -------
        bool
            ``True`` if every active parameter is marked linear by its
            potential; ``False`` if any active parameter is nonlinear.
        """
        n_params = self.forcefield.n_params()

        mask = getattr(self.optimizer, "mask", None)
        active_mask = np.asarray(mask, dtype=bool) if mask is not None else np.ones(n_params, dtype=bool)
        if active_mask.shape != (n_params,):
            raise ValueError(
                f"Active parameter mask shape mismatch: expected {(n_params,)}, got {active_mask.shape}"
            )

        offset = 0
        for _, pot in IteratePotentials(self.forcefield):
            linear_mask = np.asarray(pot.is_param_linear(), dtype=bool)
            n_local = pot.n_params()
            if linear_mask.shape != (n_local,):
                raise ValueError(
                    f"Linear mask shape mismatch for {type(pot).__name__}: "
                    f"expected {(n_local,)}, got {linear_mask.shape}"
                )
            submask = active_mask[offset:offset + n_local]
            if np.any(submask & ~linear_mask):
                return False
            offset += n_local

        if offset != n_params:
            raise ValueError(
                f"Linear scan mismatch: consumed {offset} params from a {n_params}-parameter container"
            )
        return True

    def optimizer_accepts_hessian(self) -> bool:
        """Return whether the configured optimizer can consume a Hessian.

        Returns
        -------
        bool
            ``True`` when the optimizer ``step`` signature includes a
            ``hessian`` argument. Second-order trainers use this to decide
            whether to require Hessian statistics in their batches.
        """
        return hasattr(self.optimizer, 'step') and 'hessian' in self.optimizer.step.__code__.co_varnames

    @abstractmethod
    def step(self, batch: Dict[str, Any], apply_update: bool = True):
        """
        Perform one optimization step from a pre-built trainer batch.

        Parameters
        ----------
        batch : dict
            Subclass-specific batch payload produced by the workflow.
        apply_update : bool, optional
            If True, apply the optimizer update. If False, return dry-run outputs.

        Returns
        -------
        dict
            Subclass-specific output payload.
        """
        pass

