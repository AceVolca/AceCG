# AceCG/trainers/analytic/rem.py
"""Analytic REM (Relative Entropy Minimization) trainer.

This trainer is now a pure statistics-based optimizer frontend.

Expected inputs are precomputed ensemble statistics:
    - energy_grad_AA : <dU/dλ>_AA
    - energy_grad_CG : <dU/dλ>_CG

Optional Hessian-related statistics:
    - d2U_AA         : <d²U/dλj dλk>_AA
    - d2U_CG         : <d²U/dλj dλk>_CG
    - grad_outer_CG  : <(dU/dλj)(dU/dλk)>_CG

The trainer computes:
    grad = β ( <dU/dλ>_AA - <dU/dλ>_CG )

If the optimizer accepts a Hessian, it also computes:
    H_jk = β [ <d²U/dλj dλk>_AA - <d²U/dλj dλk>_CG
               + β ( <(dU/dλj)(dU/dλk)>_CG
                     - <dU/dλj>_CG <dU/dλk>_CG ) ]

Frame-weight modes
------------------
AceCG's MPI compute engine supports two frame-weight pipelines; both end up
in this trainer's ``step()`` via the same ``REMBatch`` schema:

1. **A-priori frame weights** — weights known at sampling time. The engine
   consumes ``spec['frame_weight']`` inside the one-pass frame loop and
   produces weighted ensemble averages directly. Use :py:meth:`make_batch`
   to wrap those averages.

2. **A-posteriori frame weights** — weights computed *after* sampling
   (e.g. WHAM against a set of biased simulations). The engine is run with
   ``step['reduce_stack'] = True`` and emits per-frame stacks
   (``energy_grad_frame``, optionally ``d2U_frame`` and ``grad_outer_frame``)
   plus ``frame_ids``. An external tool then produces per-frame CG weights.
   Use :py:meth:`make_batch_reweighted` to fold those weights into the
   ensemble averages; internally it just calls :py:meth:`make_batch`.

The trainer itself only ever sees ensemble-averaged statistics; both modes
converge on the same ``REMBatch`` contract.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    from typing import TypedDict, NotRequired
except ImportError:  # Python < 3.11
    from typing_extensions import TypedDict, NotRequired

from ..base import BaseTrainer


def load_reweighted_rem_stacks(  # From Ace
    energy_grad_frame_paths: Sequence[str | Path],
    frame_weight_paths: Sequence[str | Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate stacked REM gradients plus external frame weights.

    Parameters
    ----------
    energy_grad_frame_paths
        Ordered sequence of pickle files produced by ``run_post`` with
        ``reduce_stack=True``. Each file must contain an ``energy_grad_frame``
        array of shape ``(n_frames, n_params)``.
    frame_weight_paths
        Ordered sequence of ``.npy`` / ``.npz`` files containing one frame
        weight vector per REM stack file. The order must match
        ``energy_grad_frame_paths``.

    Returns
    -------
    energy_grad_frame : np.ndarray
        Concatenated per-frame gradient stack with shape
        ``(sum_i n_frames_i, n_params)``.
    frame_weight : np.ndarray
        Concatenated frame weights with shape ``(sum_i n_frames_i,)``.
    """
    grad_paths = [Path(path) for path in energy_grad_frame_paths]
    weight_paths = [Path(path) for path in frame_weight_paths]
    if len(grad_paths) != len(weight_paths):
        raise ValueError(
            "energy_grad_frame_paths and frame_weight_paths must have the same length, "
            f"got {len(grad_paths)} and {len(weight_paths)}."
        )
    if not grad_paths:
        raise ValueError("At least one energy_grad_frame path is required.")

    grad_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    n_params_ref: Optional[int] = None

    for grad_path, weight_path in zip(grad_paths, weight_paths):
        with open(grad_path, "rb") as handle:
            payload = pickle.load(handle)
        if "energy_grad_frame" not in payload:
            raise KeyError(
                f"{grad_path} does not contain 'energy_grad_frame'. "
                "Expected a reduce_stack=True REM output."
            )
        grad_frame = np.asarray(payload["energy_grad_frame"], dtype=np.float64)
        if grad_frame.ndim != 2:
            raise ValueError(
                f"{grad_path} energy_grad_frame must be 2-D (n_frames, n_params); "
                f"got shape {grad_frame.shape}."
            )
        if n_params_ref is None:
            n_params_ref = int(grad_frame.shape[1])
        elif grad_frame.shape[1] != n_params_ref:
            raise ValueError(
                "All energy_grad_frame arrays must have the same n_params. "
                f"Expected {n_params_ref}, got {grad_frame.shape[1]} in {grad_path}."
            )

        suffix = weight_path.suffix.lower()
        if suffix == ".npy":
            weight = np.load(weight_path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(weight_path, allow_pickle=False) as payload:
                files = list(payload.files)
                if "frame_weight" in payload:
                    weight = payload["frame_weight"]
                elif len(files) == 1:
                    weight = payload[files[0]]
                else:
                    raise ValueError(
                        f"{weight_path} must contain exactly one array or a 'frame_weight' array."
                    )
        else:
            raise ValueError(
                f"{weight_path} must be a .npy or .npz file, got suffix {weight_path.suffix!r}."
            )

        weight_arr = np.asarray(weight, dtype=np.float64).reshape(-1)
        if weight_arr.size != grad_frame.shape[0]:
            raise ValueError(
                f"{weight_path} length must match n_frames in {grad_path}: "
                f"got {weight_arr.size} vs {grad_frame.shape[0]}."
            )

        grad_blocks.append(grad_frame)
        weight_blocks.append(weight_arr)

    return (
        np.concatenate(grad_blocks, axis=0),
        np.concatenate(weight_blocks, axis=0),
    )


# -----------------------------------------------------------------------------
# TypedDict schemas
# -----------------------------------------------------------------------------
class EnsembleBatch(TypedDict, total=False):
    """
    Per-ensemble data passed to analytic derivative routines.

    Keys
    ----
    dist : array-like (required by trainers using this batch)
        Per-frame geometric features (often pair distances).
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
    energy_grad_AA : np.ndarray
        AA/reference ensemble average <dU/dλ>_AA. Shape (n_params,).

    energy_grad_CG : np.ndarray
        CG/model ensemble average <dU/dλ>_CG. Shape (n_params,).

    Optional keys
    -------------
    d2U_AA : np.ndarray
        AA/reference true second-derivative matrix
        <d²U/dλj dλk>_AA. Shape (n_params, n_params).
        Required only when the optimizer accepts a Hessian.

    d2U_CG : np.ndarray
        CG/model true second-derivative matrix
        <d²U/dλj dλk>_CG. Shape (n_params, n_params).
        Required only when the optimizer accepts a Hessian.

    grad_outer_CG : np.ndarray
        CG/model gradient outer product
        <(dU/dλj)(dU/dλk)>_CG. Shape (n_params, n_params).
        Required only when the optimizer accepts a Hessian.

    step_index : int
        Optional logging step counter.
    """
    energy_grad_AA: Any
    energy_grad_CG: Any
    unmasked_energy_grad_AA: NotRequired[Any]
    unmasked_energy_grad_CG: NotRequired[Any]
    d2U_AA: NotRequired[Any]
    d2U_CG: NotRequired[Any]
    grad_outer_CG: NotRequired[Any]
    optimizer_gradient_mode: NotRequired[str]
    outside_aux_weight: NotRequired[float]
    allow_unmasked_optimizer_gradient: NotRequired[bool]
    step_index: NotRequired[int]


class REMOut(TypedDict, total=False):
    """
    Return schema for REMTrainerAnalytic.step.

    Common keys
    -----------
    name : str
    grad : np.ndarray
    hessian : np.ndarray | None
    update : np.ndarray
    meta : dict

    REM-specific keys
    -----------------
    energy_grad_AA : np.ndarray
    energy_grad_CG : np.ndarray
    """
    name: str
    grad: Any
    optimizer_grad: NotRequired[Any]
    hessian: NotRequired[Any]
    update: Any
    energy_grad_AA: Any
    energy_grad_CG: Any
    meta: Dict[str, Any]


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class REMTrainerAnalytic(BaseTrainer):
    """Analytic Relative Entropy Minimization (REM) trainer.

    Computes REM gradient:
        grad = β ( <dU/dλ>_AA - <dU/dλ>_CG )

    Optionally computes Hessian if the optimizer supports it.

    This trainer no longer recomputes derivatives from frame-level features.
    It only consumes precomputed AA / CG ensemble statistics.
    """

    BATCH_SCHEMA: Dict[str, Any] = {
        "energy_grad_AA": "required; np.ndarray shape (n_params,); AA ensemble average <dU/dλ>_AA",
        "energy_grad_CG": "required; np.ndarray shape (n_params,); CG ensemble average <dU/dλ>_CG",
        "d2U_AA": "optional; np.ndarray shape (n_params,n_params); required only if optimizer_accepts_hessian=True",
        "d2U_CG": "optional; np.ndarray shape (n_params,n_params); required only if optimizer_accepts_hessian=True",
        "grad_outer_CG": "optional; np.ndarray shape (n_params,n_params); required only if optimizer_accepts_hessian=True",
        "step_index": "optional int; logging step counter; default 0",
    }

    RETURN_SCHEMA: Dict[str, Any] = {
        "name": 'str; always "REM"',
        "grad": "np.ndarray; shape (n_params,); beta*(<dU/dλ>_AA - <dU/dλ>_CG)",
        "hessian": "np.ndarray|None; shape (n_params,n_params); only if optimizer_accepts_hessian is True",
        "update": "np.ndarray; shape (n_params,); optimizer update if apply_update=True else zeros_like(grad)",
        "energy_grad_AA": "np.ndarray; shape (n_params,); AA ensemble average <dU/dλ>_AA",
        "energy_grad_CG": "np.ndarray; shape (n_params,); CG ensemble average <dU/dλ>_CG",
        "meta": "dict; diagnostics (step_index, grad_norm, update_norm, need_hessian, ...)",
    }

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Return a dict with `batch` and `return` schema for introspection."""
        return {"batch": cls.BATCH_SCHEMA, "return": cls.RETURN_SCHEMA}

    @staticmethod
    def make_batch(
        *,
        energy_grad_AA: np.ndarray,
        energy_grad_CG: np.ndarray,
        d2U_AA: Optional[np.ndarray] = None,
        d2U_CG: Optional[np.ndarray] = None,
        grad_outer_CG: Optional[np.ndarray] = None,
        unmasked_energy_grad_AA: Optional[np.ndarray] = None,
        unmasked_energy_grad_CG: Optional[np.ndarray] = None,
        optimizer_gradient_mode: str = "masked",
        outside_aux_weight: float = 1.0,
        allow_unmasked_optimizer_gradient: bool = False,
        step_index: int = 0,
    ) -> REMBatch:
        """
        Build a REMBatch from precomputed ensemble statistics.

        Parameters
        ----------
        energy_grad_AA : (n_params,)
            AA-side ensemble average <dU/dλ>_AA.
        energy_grad_CG : (n_params,)
            CG-side ensemble average <dU/dλ>_CG.
        d2U_AA : (n_params, n_params), optional
            AA-side true second derivative <d²U/dλj dλk>_AA.
        d2U_CG : (n_params, n_params), optional
            CG-side true second derivative <d²U/dλj dλk>_CG.
        grad_outer_CG : (n_params, n_params), optional
            CG-side gradient outer product <(dU/dλj)(dU/dλk)>_CG.
        step_index : int
            Logging step counter.

        Returns
        -------
        REMBatch
            Statistics-only REM batch.
        """
        batch: REMBatch = {
            "energy_grad_AA": energy_grad_AA,
            "energy_grad_CG": energy_grad_CG,
            "step_index": int(step_index),
        }
        if d2U_AA is not None:
            batch["d2U_AA"] = d2U_AA
        if d2U_CG is not None:
            batch["d2U_CG"] = d2U_CG
        if grad_outer_CG is not None:
            batch["grad_outer_CG"] = grad_outer_CG
        if unmasked_energy_grad_AA is not None:
            batch["unmasked_energy_grad_AA"] = unmasked_energy_grad_AA
        if unmasked_energy_grad_CG is not None:
            batch["unmasked_energy_grad_CG"] = unmasked_energy_grad_CG
        batch["optimizer_gradient_mode"] = optimizer_gradient_mode
        batch["outside_aux_weight"] = float(outside_aux_weight)
        batch["allow_unmasked_optimizer_gradient"] = bool(
            allow_unmasked_optimizer_gradient
        )
        return batch

    @staticmethod
    def make_batch_reweighted(
        *,
        energy_grad_AA: np.ndarray,
        energy_grad_CG_frame: np.ndarray,
        frame_weights_CG: np.ndarray,
        d2U_AA: Optional[np.ndarray] = None,
        d2U_CG_frame: Optional[np.ndarray] = None,
        grad_outer_CG_frame: Optional[np.ndarray] = None,
        unmasked_energy_grad_AA: Optional[np.ndarray] = None,
        unmasked_energy_grad_CG_frame: Optional[np.ndarray] = None,
        optimizer_gradient_mode: str = "masked",
        outside_aux_weight: float = 1.0,
        allow_unmasked_optimizer_gradient: bool = False,
        step_index: int = 0,
    ) -> REMBatch:
        """Build a REMBatch from AA averages + per-frame CG stacks + a-posteriori weights.

        This is the a-posteriori entry point: the MPI engine produced the CG
        stacks under ``reduce_stack=True`` and an external tool (e.g. a WHAM
        frontend) produced the per-frame CG reweighting vector
        ``frame_weights_CG``. Weights are normalized to sum to 1.0 before
        averaging; they do not need to be pre-normalized.

        Shapes
        ------
        energy_grad_AA        : (n_params,)
        energy_grad_CG_frame  : (n_frames, n_params)
        frame_weights_CG      : (n_frames,)
        d2U_AA                : (n_params, n_params), optional
        d2U_CG_frame          : (n_frames, n_params, n_params), optional
        grad_outer_CG_frame   : (n_frames, n_params, n_params), optional

        Notes
        -----
        The AA side is assumed to already be an ensemble average — AA/reference
        reweighting (if needed) should have been folded in upstream.
        """
        grad_frame = np.asarray(energy_grad_CG_frame, dtype=np.float64)
        weights = np.asarray(frame_weights_CG, dtype=np.float64).reshape(-1)
        if grad_frame.ndim != 2:
            raise ValueError(
                f"energy_grad_CG_frame must be 2-D (n_frames, n_params); got shape {grad_frame.shape}"
            )
        if weights.size != grad_frame.shape[0]:
            raise ValueError(
                "frame_weights_CG length must match n_frames in energy_grad_CG_frame: "
                f"got {weights.size} vs {grad_frame.shape[0]}"
            )
        if np.any(weights < 0.0):
            raise ValueError("frame_weights_CG must be nonnegative.")
        w_sum = float(weights.sum())
        if w_sum <= 0.0:
            raise ValueError("frame_weights_CG must have a positive sum.")
        w_norm = weights / w_sum

        energy_grad_CG = w_norm @ grad_frame

        unmasked_energy_grad_CG: Optional[np.ndarray] = None
        if unmasked_energy_grad_CG_frame is not None:
            unmasked_grad_frame = np.asarray(
                unmasked_energy_grad_CG_frame,
                dtype=np.float64,
            )
            if unmasked_grad_frame.ndim != 2 or unmasked_grad_frame.shape != grad_frame.shape:
                raise ValueError(
                    "unmasked_energy_grad_CG_frame must match energy_grad_CG_frame "
                    f"shape {grad_frame.shape}, got {unmasked_grad_frame.shape}."
                )
            unmasked_energy_grad_CG = w_norm @ unmasked_grad_frame

        d2U_CG: Optional[np.ndarray] = None
        if d2U_CG_frame is not None:
            d2U_stack = np.asarray(d2U_CG_frame, dtype=np.float64)
            if d2U_stack.ndim != 3 or d2U_stack.shape[0] != weights.size:
                raise ValueError(
                    "d2U_CG_frame must be (n_frames, n_params, n_params); "
                    f"got shape {d2U_stack.shape}"
                )
            d2U_CG = np.tensordot(w_norm, d2U_stack, axes=(0, 0))

        grad_outer_CG: Optional[np.ndarray] = None
        if grad_outer_CG_frame is not None:
            go_stack = np.asarray(grad_outer_CG_frame, dtype=np.float64)
            if go_stack.ndim != 3 or go_stack.shape[0] != weights.size:
                raise ValueError(
                    "grad_outer_CG_frame must be (n_frames, n_params, n_params); "
                    f"got shape {go_stack.shape}"
                )
            grad_outer_CG = np.tensordot(w_norm, go_stack, axes=(0, 0))

        return REMTrainerAnalytic.make_batch(
            energy_grad_AA=energy_grad_AA,
            energy_grad_CG=energy_grad_CG,
            d2U_AA=d2U_AA,
            d2U_CG=d2U_CG,
            grad_outer_CG=grad_outer_CG,
            unmasked_energy_grad_AA=unmasked_energy_grad_AA,
            unmasked_energy_grad_CG=unmasked_energy_grad_CG,
            optimizer_gradient_mode=optimizer_gradient_mode,
            outside_aux_weight=outside_aux_weight,
            allow_unmasked_optimizer_gradient=allow_unmasked_optimizer_gradient,
            step_index=step_index,
        )

    def step(self, batch: REMBatch, apply_update: bool = True) -> REMOut:
        """
        Execute one REM optimization step.

        Parameters
        ----------
        batch : REMBatch
            Statistics-only REM batch.
        apply_update : bool, default=True
            If True, apply the optimizer step and update the potential.
            If False, run in dry-run mode and return update=zeros_like(grad).

        Returns
        -------
        REMOut
            Dictionary with standardized REM outputs.
        """
        assert isinstance(batch, dict), "REMTrainerAnalytic.step expects batch as a dict."

        step_index = int(batch.get("step_index", 0))
        need_hessian = self.optimizer_accepts_hessian()

        # --- Required first-order statistics ---
        if "energy_grad_AA" not in batch:
            raise KeyError("batch['energy_grad_AA'] is required.")
        if "energy_grad_CG" not in batch:
            raise KeyError("batch['energy_grad_CG'] is required.")

        energy_grad_AA = np.asarray(batch["energy_grad_AA"], dtype=np.float64)
        energy_grad_CG = np.asarray(batch["energy_grad_CG"], dtype=np.float64)

        # --- REM gradient ---
        grad = self.beta * (energy_grad_AA - energy_grad_CG)
        unmasked_grad = None
        if "unmasked_energy_grad_AA" in batch or "unmasked_energy_grad_CG" in batch:
            missing_unmasked = [
                key for key in ("unmasked_energy_grad_AA", "unmasked_energy_grad_CG")
                if key not in batch
            ]
            if missing_unmasked:
                raise KeyError(
                    "Missing unmasked REM gradient statistics: "
                    + ", ".join(missing_unmasked)
                )
            unmasked_grad = self.beta * (
                np.asarray(batch["unmasked_energy_grad_AA"], dtype=np.float64)
                - np.asarray(batch["unmasked_energy_grad_CG"], dtype=np.float64)
            )
        optimizer_grad = self.select_optimizer_gradient(
            grad,
            unmasked_grad=unmasked_grad,
            mode=str(batch.get("optimizer_gradient_mode", "masked")),
            outside_aux_weight=float(batch.get("outside_aux_weight", 1.0)),
            allow_unmasked=bool(batch.get("allow_unmasked_optimizer_gradient", False)),
        )

        # --- Optional Hessian ---
        hessian = None
        if need_hessian:
            missing = [
                key for key in ("d2U_AA", "d2U_CG", "grad_outer_CG")
                if key not in batch
            ]
            if missing:
                raise KeyError(
                    "Missing Hessian statistics in REM batch: "
                    + ", ".join(missing)
                )

            d2U_AA = np.asarray(batch["d2U_AA"], dtype=np.float64)
            d2U_CG = np.asarray(batch["d2U_CG"], dtype=np.float64)
            grad_outer_CG = np.asarray(batch["grad_outer_CG"], dtype=np.float64)

            # H_jk = β [⟨∂²U/∂λⱼ∂λₖ⟩_AA − ⟨∂²U/∂λⱼ∂λₖ⟩_CG
            #           + β(⟨∂U/∂λⱼ·∂U/∂λₖ⟩_CG − ⟨∂U/∂λⱼ⟩_CG·⟨∂U/∂λₖ⟩_CG)]
            hessian = self.beta * (
                d2U_AA - d2U_CG
                + self.beta * (
                    grad_outer_CG - np.outer(energy_grad_CG, energy_grad_CG)
                )
            )

        # --- Optimization step ---
        if apply_update:
            if hessian is not None:
                update = self.optimizer.step(optimizer_grad, hessian=hessian)
            else:
                update = self.optimizer.step(optimizer_grad)
            self.clamp_and_update()
        else:
            update = np.zeros_like(optimizer_grad)

        # --- Logging ---
        if self.logger is not None:
            mask_ratio = float(np.mean(self.optimizer.mask.astype(float)))
            self.logger.add_scalar("REM/mask_ratio", mask_ratio, step_index)
            self.logger.add_scalar(
                "REM/lr",
                float(getattr(self.optimizer, "lr", np.nan)),
                step_index,
            )
            self.logger.add_scalar("REM/grad_norm", float(np.linalg.norm(grad)), step_index)
            self.logger.add_scalar(
                "REM/optimizer_grad_norm",
                float(np.linalg.norm(optimizer_grad)),
                step_index,
            )
            self.logger.add_scalar("REM/update_norm", float(np.linalg.norm(update)), step_index)
            if hessian is not None:
                self.logger.add_scalar(
                    "REM/hessian_cond",
                    float(np.linalg.cond(hessian)),
                    step_index,
                )

        return {
            "name": "REM",
            "grad": grad,
            "optimizer_grad": optimizer_grad,
            "hessian": hessian,
            "update": update,
            "energy_grad_AA": energy_grad_AA,
            "energy_grad_CG": energy_grad_CG,
            "meta": {
                "step_index": step_index,
                "grad_norm": float(np.linalg.norm(grad)),
                "optimizer_grad_norm": float(np.linalg.norm(optimizer_grad)),
                "update_norm": float(np.linalg.norm(update)),
                "need_hessian": need_hessian,
                "optimizer_gradient_mode": str(
                    batch.get("optimizer_gradient_mode", "masked")
                ),
            },
        }
