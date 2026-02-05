# AceCG/trainers/analytic.py
"""Analytic trainers (non-NN) with dict-based I/O.

This module provides analytic (NumPy-based) training loops for coarse-grained potentials.

Key design choices
------------------
1) **Dictionary-based input**: `trainer.step(batch=...)` where `batch` is a dict.
   This makes call sites explicit and resilient to signature drift.

2) **Dictionary-based output**: every `step()` returns a dict with standardized keys:
   - "name": str
   - "grad": np.ndarray, shape (n_params,)
   - "hessian": np.ndarray | None, shape (n_params, n_params)
   - "update": np.ndarray, shape (n_params,)  (zeros if apply_update=False)
   - plus method-specific keys (e.g., "loss", "dUdL_AA", "dUdL_CG", "meta"...)

3) **Dry-run support**: `apply_update=False` computes gradients (and Hessians) but
   does not modify optimizer state nor parameters. This is required to implement
   true meta-optimization in `MultiTrainerAnalytic` (combine gradients then step once).

Expected batch schemas
----------------------
REMTrainerAnalytic.step(batch):
  batch = {
    "AA": {"dist": ..., "weight": optional},
    "CG": {"dist": ..., "weight": optional},
    "step_index": optional int,
  }

MSETrainerAnalytic.step(batch):
  batch = {
    "pmf_AA": np.ndarray, shape (n_bins,),
    "pmf_CG": np.ndarray, shape (n_bins,),
    "CG": {"dist": ..., "weight": optional},
    "CG_bin_idx_frame": np.ndarray, shape (n_frames,),
    "weighted_gauge": optional bool,
    "step_index": optional int,
  }

MultiTrainerAnalytic.step(batches):
  batches = [batch_for_trainer0, batch_for_trainer1, ...]  (same length as trainers)

"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# -----------------------------------------------------------------------------
# TypedDict schemas (IDE-friendly)
# -----------------------------------------------------------------------------
# These types are for developer ergonomics: IDE auto-completion and static checking.
# They do not change runtime behavior and remain compatible with plain dict inputs.

try:
    from typing import TypedDict, NotRequired
except ImportError:  # Python < 3.11
    from typing_extensions import TypedDict, NotRequired


class EnsembleBatch(TypedDict, total=False):
    """
    Per-ensemble data passed to analytic derivative routines.

    Keys
    ----
    dist : array-like (required by trainers using this batch)
        Per-frame geometric features (often pair distances). Consumed by
        `dUdLByFrame(potential, dist)`.
    weight : array-like, NotRequired
        Per-frame weights for reweighting ensemble averages; shape (n_frames,).
    """
    dist: Any
    weight: NotRequired[Any]


class REMBatch(TypedDict, total=False):
    """
    Batch schema for REMTrainerAnalytic.step.

    Required keys
    -------------
    AA : EnsembleBatch
        AA/reference ensemble. Must include AA["dist"].
    CG : EnsembleBatch
        CG/model ensemble. Must include CG["dist"].

    Optional keys
    -------------
    step_index : int
        Logging step counter.
    """
    AA: EnsembleBatch
    CG: EnsembleBatch
    step_index: NotRequired[int]


class REMOut(TypedDict, total=False):
    """
    Return schema for REMTrainerAnalytic.step.

    Common keys
    ----------
    name : str
    grad : np.ndarray
    hessian : np.ndarray | None
    update : np.ndarray
    meta : dict

    REM-specific keys
    -----------------
    dUdL_AA : np.ndarray
    dUdL_CG : np.ndarray
    """
    name: str
    grad: Any
    hessian: NotRequired[Any]
    update: Any
    dUdL_AA: Any
    dUdL_CG: Any
    meta: Dict[str, Any]


class MSEBatch(TypedDict, total=False):
    """
    Batch schema for MSETrainerAnalytic.step.

    Required keys
    -------------
    pmf_AA : np.ndarray, shape (n_bins,)
    pmf_CG : np.ndarray, shape (n_bins,)
    CG : EnsembleBatch
        Must include CG["dist"].
    CG_bin_idx_frame : np.ndarray, shape (n_frames,)
        Bin index per frame (0..n_bins-1).

    Optional keys
    -------------
    weighted_gauge : bool
    step_index : int
    """
    pmf_AA: Any
    pmf_CG: Any
    CG: EnsembleBatch
    CG_bin_idx_frame: Any
    weighted_gauge: NotRequired[bool]
    step_index: NotRequired[int]


class MSEOut(TypedDict, total=False):
    """
    Return schema for MSETrainerAnalytic.step.

    Common keys
    ----------
    name : str
    grad : np.ndarray
    hessian : None
    update : np.ndarray
    meta : dict

    MSE-specific keys
    -----------------
    loss : float
    """
    name: str
    loss: float
    grad: Any
    hessian: NotRequired[Any]
    update: Any
    meta: Dict[str, Any]


class MultiOut(TypedDict, total=False):
    """Return schema for MultiTrainerAnalytic.step."""
    mode: str
    update: Any
    combined_grad: NotRequired[Any]
    combined_hessian: NotRequired[Any]
    sub: List[Dict[str, Any]]
    meta: Dict[str, Any]

from .base import BaseTrainer
from .utils import optimizer_accepts_hessian
from ..utils.compute import (
    dUdLByFrame,
    dUdL,
    d2UdLjdLk_Matrix,
    dUdLj_dUdLk_Matrix,
    Hessian,
    dUdLByBin,
)


def _get_step_index(batch: Dict[str, Any]) -> int:
    """Helper: robustly extract step_index from a batch dict."""
    si = batch.get("step_index", 0)
    try:
        return int(si)
    except Exception:
        return 0


class REMTrainerAnalytic(BaseTrainer):
    """Analytic Relative Entropy Minimization (REM) trainer.

    Computes REM gradient:
        grad = β ( <dU/dλ>_AA - <dU/dλ>_CG )

    Optionally computes Hessian if the optimizer supports it.

    Returns a dict with standardized keys (see module docstring).
    """

    # ---- Public schema objects (for documentation / validation) ----
    # These describe the expected `batch` dict and returned `out` dict keys.
    BATCH_SCHEMA: Dict[str, Any] = {
        "AA": {
            "dist": "required; per-frame AA features passed to dUdLByFrame(potential, dist)",
            "weight": "optional; per-frame AA weights for reweighting; shape (n_frames,)",
        },
        "CG": {
            "dist": "required; per-frame CG features passed to dUdLByFrame(potential, dist)",
            "weight": "optional; per-frame CG weights for reweighting; shape (n_frames,)",
        },
        "step_index": "optional int; logging step counter (TensorBoard); default 0",
    }

    RETURN_SCHEMA: Dict[str, Any] = {
        "name": 'str; always "REM"',
        "grad": "np.ndarray; shape (n_params,); beta*(<dU/dλ>_AA - <dU/dλ>_CG)",
        "hessian": "np.ndarray|None; shape (n_params,n_params); only if optimizer_accepts_hessian is True",
        "update": "np.ndarray; shape (n_params,); optimizer update if apply_update=True else zeros_like(grad)",
        "meta": "dict; diagnostics (step_index, grad_norm, update_norm, ...)",
        "dUdL_AA": "np.ndarray; shape (n_params,); AA ensemble average <dU/dλ>_AA",
        "dUdL_CG": "np.ndarray; shape (n_params,); CG ensemble average <dU/dλ>_CG",
    }

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Return a dict with `batch` and `return` schema for introspection."""
        return {"batch": cls.BATCH_SCHEMA, "return": cls.RETURN_SCHEMA}
    @staticmethod
    def make_batch(
        AA_dist,
        CG_dist,
        AA_weight=None,
        CG_weight=None,
        step_index: int = 0,
    ) -> REMBatch:
        """
        Build a REMBatch dict for REMTrainerAnalytic.step().

        Parameters
        ----------
        AA_dist : array-like
            Per-frame AA features (e.g., distances). Passed to dUdLByFrame(potential, AA_dist).
        CG_dist : array-like
            Per-frame CG features. Passed to dUdLByFrame(potential, CG_dist).
        AA_weight : array-like, optional
            Per-frame weights for AA reweighting; shape (n_frames,).
        CG_weight : array-like, optional
            Per-frame weights for CG reweighting; shape (n_frames,).
        step_index : int, default 0
            Logging step counter.

        Returns
        -------
        batch : REMBatch
            {
              "AA": {"dist": AA_dist, "weight": AA_weight?},
              "CG": {"dist": CG_dist, "weight": CG_weight?},
              "step_index": step_index,
            }
        """
        batch: REMBatch = {
            "AA": {"dist": AA_dist},
            "CG": {"dist": CG_dist},
            "step_index": int(step_index),
        }
        if AA_weight is not None:
            batch["AA"]["weight"] = AA_weight
        if CG_weight is not None:
            batch["CG"]["weight"] = CG_weight
        return batch



    def step(self, batch: REMBatch, apply_update: bool = True) -> REMOut:
        # --- Parse batch ---
        assert isinstance(batch, dict), "REMTrainerAnalytic.step expects batch as a dict."
        AA = batch["AA"]
        CG = batch["CG"]
        step_index = _get_step_index(batch)

        # --- Compute per-frame derivatives in AA/CG ensembles ---
        dUdL_AA_frame = dUdLByFrame(self.potential, AA["dist"])
        dUdL_CG_frame = dUdLByFrame(self.potential, CG["dist"])

        # --- Ensemble averages (support reweighting) ---
        w_AA = AA.get("weight", None)
        w_CG = CG.get("weight", None)
        dUdL_AA = dUdL(dUdL_AA_frame, w_AA)
        dUdL_CG = dUdL(dUdL_CG_frame, w_CG)

        # --- REM gradient ---
        grad = self.beta * (dUdL_AA - dUdL_CG)

        # --- Optional Hessian ---
        hessian = None
        if optimizer_accepts_hessian(self.optimizer):
            d2U_AA = d2UdLjdLk_Matrix(self.potential, AA["dist"], w_AA)
            d2U_CG = d2UdLjdLk_Matrix(self.potential, CG["dist"], w_CG)
            dUU_CG = dUdLj_dUdLk_Matrix(dUdL_CG_frame, w_CG)
            hessian = Hessian(self.beta, d2U_AA, d2U_CG, dUU_CG, dUdL_CG)

        # --- Optimization step (optional) ---
        if apply_update:
            if hessian is not None:
                update = self.optimizer.step(grad, hessian=hessian)
            else:
                update = self.optimizer.step(grad)
            self.clamp_and_update()
        else:
            # Dry-run: do NOT touch optimizer state or parameters.
            update = np.zeros_like(grad)

        # --- Logging ---
        if self.logger is not None:
            mask_ratio = float(np.mean(self.optimizer.mask.astype(float)))
            self.logger.add_scalar("REM/mask_ratio", mask_ratio, step_index)
            self.logger.add_scalar("REM/lr", float(getattr(self.optimizer, "lr", np.nan)), step_index)
            self.logger.add_scalar("REM/grad_norm", float(np.linalg.norm(grad)), step_index)
            self.logger.add_scalar("REM/update_norm", float(np.linalg.norm(update)), step_index)
            if hessian is not None:
                self.logger.add_scalar("REM/hessian_cond", float(np.linalg.cond(hessian)), step_index)

        return {
            "name": "REM",
            "grad": grad,
            "hessian": hessian,
            "update": update,
            "dUdL_AA": dUdL_AA,
            "dUdL_CG": dUdL_CG,
            "meta": {
                "step_index": step_index,
                "grad_norm": float(np.linalg.norm(grad)),
                "update_norm": float(np.linalg.norm(update)),
            },
        }


class MSETrainerAnalytic(BaseTrainer):
    """Analytic PMF-matching trainer with an MSE-like objective.

    This follows your current implementation:
      - shift pmf_CG by a gauge constant c that minimizes mismatch (optionally weighted)
      - loss is defined as || pmf_AA - (pmf_CG - c) ||_2  (Euclidean norm)

    Returns a dict with standardized keys (see module docstring).
    """

    # ---- Public schema objects (for documentation / validation) ----
    BATCH_SCHEMA: Dict[str, Any] = {
        "pmf_AA": "required np.ndarray; shape (n_bins,); target/reference PMF",
        "pmf_CG": "required np.ndarray; shape (n_bins,); current CG PMF",
        "CG": {
            "dist": "required; per-frame CG features passed to dUdLByFrame(potential, dist)",
            "weight": "optional; per-frame CG weights for reweighting; shape (n_frames,)",
        },
        "CG_bin_idx_frame": "required np.ndarray; shape (n_frames,); integer bin index per frame (0..n_bins-1)",
        "weighted_gauge": "optional bool; default False; whether gauge shift uses CG bin probabilities",
        "step_index": "optional int; logging step counter; default 0",
    }

    RETURN_SCHEMA: Dict[str, Any] = {
        "name": 'str; always "MSE"',
        "loss": "float; L2 norm mismatch ||pmf_AA - (pmf_CG - c)||_2 (note: not mean-squared unless changed)",
        "grad": "np.ndarray; shape (n_params,); gradient of PMF mismatch wrt λ",
        "hessian": "None; reserved for uniform interface",
        "update": "np.ndarray; shape (n_params,); optimizer update if apply_update=True else zeros_like(grad)",
        "meta": "dict; diagnostics (step_index, gauge_shift, weighted_gauge, grad_norm, update_norm, ...)",
    }

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Return a dict with `batch` and `return` schema for introspection."""
        return {"batch": cls.BATCH_SCHEMA, "return": cls.RETURN_SCHEMA}
    @staticmethod
    def make_batch(
        pmf_AA,
        pmf_CG,
        CG_dist,
        CG_bin_idx_frame,
        CG_weight=None,
        weighted_gauge: bool = False,
        step_index: int = 0,
    ) -> MSEBatch:
        """
        Build a MSEBatch dict for MSETrainerAnalytic.step().

        Parameters
        ----------
        pmf_AA : np.ndarray, shape (n_bins,)
            Target/reference PMF.
        pmf_CG : np.ndarray, shape (n_bins,)
            Current CG PMF on the same bins.
        CG_dist : array-like
            Per-frame CG features (e.g., distances). Passed to dUdLByFrame(potential, CG_dist).
        CG_bin_idx_frame : np.ndarray, shape (n_frames,)
            Bin index (0..n_bins-1) for each CG frame.
        CG_weight : array-like, optional
            Per-frame CG weights for reweighting; shape (n_frames,).
        weighted_gauge : bool, default False
            Whether to use probability-weighted gauge shift.
        step_index : int, default 0
            Logging step counter.

        Returns
        -------
        batch : MSEBatch
            {
              "pmf_AA": pmf_AA,
              "pmf_CG": pmf_CG,
              "CG": {"dist": CG_dist, "weight": CG_weight?},
              "CG_bin_idx_frame": CG_bin_idx_frame,
              "weighted_gauge": weighted_gauge,
              "step_index": step_index,
            }
        """
        batch: MSEBatch = {
            "pmf_AA": pmf_AA,
            "pmf_CG": pmf_CG,
            "CG": {"dist": CG_dist},
            "CG_bin_idx_frame": CG_bin_idx_frame,
            "weighted_gauge": bool(weighted_gauge),
            "step_index": int(step_index),
        }
        if CG_weight is not None:
            batch["CG"]["weight"] = CG_weight
        return batch



    def step(self, batch: MSEBatch, apply_update: bool = True) -> MSEOut:
        assert isinstance(batch, dict), "MSETrainerAnalytic.step expects batch as a dict."
        pmf_AA = batch["pmf_AA"]
        pmf_CG = batch["pmf_CG"]
        CG = batch["CG"]
        CG_bin_idx_frame = batch["CG_bin_idx_frame"]
        weighted_gauge = bool(batch.get("weighted_gauge", False))
        step_index = _get_step_index(batch)

        # --- Compute <dU/dλ> for CG ---
        dUdL_CG_frame = dUdLByFrame(self.potential, CG["dist"])
        w_CG = CG.get("weight", None)
        dUdL_CG = dUdL(dUdL_CG_frame, w_CG)

        # --- Per-bin conditional averages ---
        dUdL_CG_bin, p_CG_bin, dUdL_CG_given_bin = dUdLByBin(
            dUdL_CG_frame,
            CG_bin_idx_frame,
            w_CG,
        )

        # --- Gauge shift for PMF_CG ---
        if weighted_gauge:
            q = np.array([p_CG_bin.get(i, 0.0) for i in range(len(pmf_CG))], dtype=float)
            c = float((pmf_CG - pmf_AA) @ q)
        else:
            c = float(np.mean(pmf_CG - pmf_AA))
        pmf_CG_shifted = pmf_CG - c

        # --- Loss (L2 norm) ---
        loss = float(np.linalg.norm(pmf_AA - pmf_CG_shifted))

        # --- Gradient of loss w.r.t parameters ---
        idx_set = set(CG_bin_idx_frame.tolist()) if hasattr(CG_bin_idx_frame, "tolist") else set(CG_bin_idx_frame)
        dErrdL_bin: Dict[int, np.ndarray] = {}
        for idx in idx_set:
            dErrdL_bin[idx] = (pmf_CG_shifted[idx] - pmf_AA[idx]) * (dUdL_CG_given_bin[idx] - dUdL_CG)

        grad = 0
        for idx in range(len(pmf_AA)):
            grad += dErrdL_bin.get(idx, 0)

        # --- Optimization step (optional) ---
        if apply_update:
            update = self.optimizer.step(grad)
            self.clamp_and_update()
        else:
            update = np.zeros_like(grad)

        # --- Logging ---
        if self.logger is not None:
            mask_ratio = float(np.mean(self.optimizer.mask.astype(float)))
            self.logger.add_scalar("MSE/mask_ratio", mask_ratio, step_index)
            self.logger.add_scalar("MSE/lr", float(getattr(self.optimizer, "lr", np.nan)), step_index)
            self.logger.add_scalar("MSE/loss", loss, step_index)
            self.logger.add_scalar("MSE/grad_norm", float(np.linalg.norm(grad)), step_index)
            self.logger.add_scalar("MSE/update_norm", float(np.linalg.norm(update)), step_index)

        return {
            "name": "MSE",
            "loss": loss,
            "grad": grad,
            "hessian": None,  # keep key for uniformity
            "update": update,
            "meta": {
                "step_index": step_index,
                "gauge_shift": c,
                "weighted_gauge": weighted_gauge,
                "grad_norm": float(np.linalg.norm(grad)),
                "update_norm": float(np.linalg.norm(update)),
            },
        }


class MultiTrainerAnalytic(BaseTrainer):
    """Meta-trainer that combines multiple analytic trainers.

    This version is fully dict-based:
      - Inputs are per-trainer batch dicts.
      - Outputs are dicts with semantic keys (no positional index bookkeeping).

    Parameters
    ----------
    combine_mode : {"update", "grad"}
      - "update": run each sub-trainer normally and combine their returned "update".
      - "grad":   run each sub-trainer in dry-run mode (apply_update=False), combine
                  their returned "grad" (and "hessian" if available), then perform a
                  single meta optimizer step.

    Notes
    -----
    - In "grad" mode, only the meta optimizer state evolves. Sub-trainers are evaluated
      as pure gradient/Hessian providers.
    - After the meta update, all sub-trainers are synchronized to the meta parameter
      vector via `tr.update_potential(self.optimizer.L)`.
    """

    # ---- Public schema objects (for documentation / validation) ----
    STEP_SCHEMA: Dict[str, Any] = {
        "batches": "required Sequence[dict]; length == len(trainers); batches[i] must satisfy trainers[i].BATCH_SCHEMA",
        "return_keys_list": "optional Sequence[Sequence[str]]; if provided, filters keys in out['sub'][i]",
    }

    RETURN_SCHEMA: Dict[str, Any] = {
        "mode": 'str; "update" or "grad"',
        "update": "np.ndarray; shape (n_params,); meta update applied to optimizer.L",
        "sub": "list[dict]; sub-trainer outputs (full or filtered by return_keys_list)",
        "meta": "dict; diagnostics (update_norm, and grad_norm in grad mode, ...)",
        "combined_grad": "np.ndarray; shape (n_params,); only in mode=='grad'",
        "combined_hessian": "np.ndarray|None; shape (n_params,n_params); only in mode=='grad'",
    }

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Return a dict with `step` and `return` schema for introspection."""
        return {"step": cls.STEP_SCHEMA, "return": cls.RETURN_SCHEMA}


    def __init__(
        self,
        potential,
        optimizer,
        trainer_list: Sequence[BaseTrainer],
        weight_array: np.ndarray,
        beta: Optional[float] = None,
        logger=None,
        combine_mode: str = "update",
    ):
        super().__init__(potential, optimizer, beta, logger)

        assert isinstance(trainer_list, (list, tuple)) and len(trainer_list) > 0, (
            "trainer_list must be a non-empty list/tuple of trainers"
        )
        for i, tr in enumerate(trainer_list):
            assert isinstance(tr, BaseTrainer), f"trainer_list[{i}] is not a BaseTrainer"

        assert isinstance(weight_array, np.ndarray), "weight_array must be a NumPy array"
        assert weight_array.ndim == 1, "weight_array must be 1D"
        assert len(trainer_list) == weight_array.shape[0], "each trainer must have exactly one weight"

        assert combine_mode in ("update", "grad"), "combine_mode must be 'update' or 'grad'"
        self.combine_mode = combine_mode

        # Keep deep copy to avoid side-effects (preserve your original intent).
        self.trainers: List[BaseTrainer] = copy.deepcopy(list(trainer_list))
        self.weights = np.asarray(weight_array, dtype=float)

        # Optional: sanity check presence of meta-optimizer L
        assert hasattr(self.optimizer, "L"), "Meta-optimizer must expose attribute `.L`"
    @staticmethod
    def make_batches(*batches: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convenience helper to build the `batches` list for MultiTrainerAnalytic.step().

        This exists purely for ergonomics, so call sites read naturally and keep
        multi-objective wiring explicit.

        Examples
        --------
        >>> rem_batch = REMTrainerAnalytic.make_batch(...)
        >>> mse_batch = MSETrainerAnalytic.make_batch(...)
        >>> batches = MultiTrainerAnalytic.make_batches(rem_batch, mse_batch)

        Returns
        -------
        list_of_batches : list[dict]
            The same objects passed in, collected into a list.
        """
        return list(batches)


    def step(
        self,
        batches: Sequence[Dict[str, Any]],
        return_keys_list: Optional[Sequence[Sequence[str]]] = None,
        parallel_grad: bool = False,
        n_workers: Optional[int] = None,
    ) -> MultiOut:
        """
        Perform one optimization step for a MultiTrainerAnalytic instance.

        This method coordinates multiple sub-trainers (e.g. REMTrainerAnalytic,
        MSETrainerAnalytic) and combines their contributions according to the
        configured ``combine_mode``.

        Two combine modes are supported:

        - ``combine_mode == "update"``:
            Each sub-trainer performs a full ``step`` with ``apply_update=True``.
            The resulting parameter updates are linearly combined and applied by
            the meta-optimizer. This mode updates sub-trainer optimizers internally
            and is therefore executed serially.

        - ``combine_mode == "grad"``:
            Each sub-trainer performs a dry-run step with ``apply_update=False``,
            returning gradients (and optionally Hessians) without modifying any
            optimizer or potential state. The gradients are combined and a single
            meta-optimizer step is applied. In this mode, sub-trainer evaluations
            can be executed in parallel.

        Parameters
        ----------
        batches : Sequence[Dict[str, Any]]
            A sequence of batch dictionaries, one per sub-trainer. Each batch must
            contain all keys required by the corresponding trainer's ``step`` method
            (e.g. distance data, weights, beta, masks, etc.). The ordering of
            ``batches`` must match the ordering of ``self.trainers``.

        return_keys_list : Optional[Sequence[Sequence[str]]], optional
            If provided, specifies which keys from each sub-trainer's output
            dictionary should be included in the returned ``sub_view`` field.
            The outer sequence must have the same length as ``self.trainers``.
            If ``None``, the full output dictionary of each sub-trainer is returned.

        parallel_grad : bool, default False
            If True and ``combine_mode == "grad"``, evaluate sub-trainer gradients
            in parallel using a thread pool. This only affects the dry-run gradient
            evaluation stage; the combination of gradients and the meta-optimizer
            update are always performed serially. This option has no effect when
            ``combine_mode == "update"``.

        n_workers : Optional[int], optional
            Number of worker threads used for parallel gradient evaluation when
            ``parallel_grad`` is True. If ``None``, a reasonable default based on
            the number of sub-trainers is used. This parameter is ignored when
            ``parallel_grad`` is False or when ``combine_mode == "update"``.

        Returns
        -------
        MultiOut
            A dictionary with the following keys:

            - ``"grad"`` :
                The combined gradient with respect to the global parameter vector
                ``L`` after applying trainer weights.

            - ``"hessian"`` :
                The combined Hessian matrix, if enabled and available; otherwise
                ``None``.

            - ``"update"`` :
                The parameter update applied by the meta-optimizer.

            - ``"sub_full"`` :
                A list of full output dictionaries returned by each sub-trainer
                ``step`` call.

            - ``"sub_view"`` :
                A list of filtered sub-trainer outputs containing only the keys
                specified by ``return_keys_list`` (or the full outputs if
                ``return_keys_list`` is ``None``).

        Notes
        -----
        - Parallel execution is implemented using threads rather than processes in
        order to avoid copying large batch data (e.g. distance arrays) between
        processes and to preserve trainer and potential state in the main thread.
        - Only the gradient evaluation stage is parallelized; all state-modifying
        operations are executed serially to ensure deterministic behavior.
    """
        assert isinstance(batches, (list, tuple)) and len(batches) == len(self.trainers), (
            "batches length must match the number of trainers"
        )
        if return_keys_list is not None:
            assert isinstance(return_keys_list, (list, tuple)) and len(return_keys_list) == len(self.trainers), (
                "return_keys_list length must match the number of trainers"
            )

        use_hessian = optimizer_accepts_hessian(self.optimizer)

        sub_full: List[Dict[str, Any]] = [] # full list of subtrainer output
        sub_view: List[Dict[str, Any]] = [] # return list of subtrainer output

        # ----------------------------
        # Mode A: combine sub-updates
        # ----------------------------
        if self.combine_mode == "update":
            updates = []

            for i, tr in enumerate(self.trainers):
                out_i = tr.step(batches[i], apply_update=True)
                assert isinstance(out_i, dict), "Sub-trainer step() must return a dict."
                assert "update" in out_i, "Sub-trainer output must include key 'update'."

                upd = np.asarray(out_i["update"])
                assert upd.shape == np.asarray(self.optimizer.L).shape, (
                    f"update shape mismatch for trainer {i}: {upd.shape} vs {np.asarray(self.optimizer.L).shape}"
                )
                updates.append(np.copy(upd))
                sub_full.append(out_i)

                if return_keys_list is None:
                    sub_view.append(out_i)
                else:
                    keys = return_keys_list[i]
                    sub_view.append({k: out_i.get(k, None) for k in keys})

            U = np.stack(updates, axis=0)         # (n_trainers, n_params)
            final_update = self.weights @ U       # (n_params,), linear combination of updates from subtrainers

            self.optimizer.L += final_update
            self.clamp_and_update()

            # Sync sub-trainers to the new global L
            for tr in self.trainers:
                tr.update_potential(self.optimizer.L)

            if self.logger is not None:
                self.logger.add_scalar("Multi/update_norm", float(np.linalg.norm(final_update)), _get_step_index(batches[0]))

            return {
                "mode": "update",
                "update": final_update,
                "sub": sub_view,
                "meta": {
                    "update_norm": float(np.linalg.norm(final_update)),
                },
            }

        # -----------------------------------------
        # Mode B: combine grads (+ Hessians) then step once (support multithreading)
        # -----------------------------------------
        grads = []
        Hs = []
        
        def _eval_one(args): # dry-run a trainer.step
            i, tr, b = args
            out_i = tr.step(b, apply_update=False)
            return i, out_i

        # 1) parallel and get out_i
        if parallel_grad and (n_workers is None or n_workers != 1):
            max_workers = n_workers or min(32, len(self.trainers))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(_eval_one, [(i, tr, batches[i]) for i, tr in enumerate(self.trainers)]))
            # sorted by i, deterministic
            results.sort(key=lambda x: x[0])
        else:
            results = [(i, self.trainers[i].step(batches[i], apply_update=False)) for i in range(len(self.trainers))]

        # 2) fill out_i into grads/Hs/sub_view
        grads = []
        Hs = []
        sub_full = []
        sub_view = []

        for i, out_i in results:
            assert isinstance(out_i, dict), "Sub-trainer step() must return a dict."
            assert "grad" in out_i, "Sub-trainer output must include key 'grad'."

            gi = np.asarray(out_i["grad"])
            assert gi.shape == np.asarray(self.optimizer.L).shape, (
                f"grad shape mismatch for trainer {i}: {gi.shape} vs {np.asarray(self.optimizer.L).shape}"
            )
            grads.append(np.copy(gi))

            if use_hessian:
                Hi = out_i.get("hessian", None)
                if Hi is None:
                    Hs.append(None)
                else:
                    Hi = np.asarray(Hi)
                    assert Hi.shape == (gi.size, gi.size), (
                        f"hessian shape mismatch for trainer {i}: {Hi.shape} vs ({gi.size}, {gi.size})"
                    )
                    Hs.append(np.copy(Hi))
            else:
                Hs.append(None)

            sub_full.append(out_i)
            if return_keys_list is None:
                sub_view.append(out_i)
            else:
                keys = return_keys_list[i]
                sub_view.append({k: out_i.get(k, None) for k in keys})


        G = np.stack(grads, axis=0)               # (n_trainers, n_params)
        g_total = self.weights @ G                # (n_params,)

        H_total = None
        if use_hessian and all(h is not None for h in Hs):
            H_stack = np.stack(Hs, axis=0)        # (n_trainers, n_params, n_params)
            H_total = np.tensordot(self.weights, H_stack, axes=(0, 0))

        if H_total is not None:
            update = self.optimizer.step(g_total, hessian=H_total)
        else:
            update = self.optimizer.step(g_total)

        self.optimizer.L += update
        self.clamp_and_update()

        for tr in self.trainers:
            tr.update_potential(self.optimizer.L)

        # Logging
        step_index = _get_step_index(batches[0])
        if self.logger is not None:
            self.logger.add_scalar("Multi/grad_norm", float(np.linalg.norm(g_total)), step_index)
            self.logger.add_scalar("Multi/update_norm", float(np.linalg.norm(update)), step_index)
            if H_total is not None:
                try:
                    self.logger.add_scalar("Multi/hessian_cond", float(np.linalg.cond(H_total)), step_index)
                except Exception:
                    pass

        return {
            "mode": "grad",
            "combined_grad": g_total,
            "combined_hessian": H_total,
            "update": update,
            "sub": sub_view,
            "meta": {
                "grad_norm": float(np.linalg.norm(g_total)),
                "update_norm": float(np.linalg.norm(update)),
            },
        }

    def set_lrs(self, lrs: Sequence[float]) -> None:
        """Set per-trainer learning rates."""
        assert isinstance(lrs, (list, tuple, np.ndarray)) and len(lrs) == len(self.trainers), (
            "lrs length must match the number of trainers"
        )
        for i, tr in enumerate(self.trainers):
            tr.optimizer.lr = float(lrs[i])