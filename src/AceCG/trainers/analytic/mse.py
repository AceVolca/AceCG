"""Analytic MSE (PMF-matching) trainer."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    from typing import TypedDict, NotRequired
# Python < 3.11
except ImportError:
    from typing_extensions import TypedDict, NotRequired

from ..base import BaseTrainer
from .rem import EnsembleBatch


def _load_array_file(path: Path, *, preferred_keys: Sequence[str]) -> np.ndarray:
    """Load a single array from .npy/.npz, using preferred keys for .npz files."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            files = list(payload.files)
            for key in preferred_keys:
                if key in files:
                    return payload[key]
            if len(files) == 1:
                return payload[files[0]]
            raise ValueError(
                f"{path} must contain exactly one array or one of: "
                + ", ".join(repr(key) for key in preferred_keys)
            )
    raise ValueError(f"{path} must be a .npy or .npz file, got suffix {path.suffix!r}.")


def load_reweighted_mse_stacks(
    bin_idx_frame_paths: Sequence[str | Path],
    energy_grad_frame_paths: Sequence[str | Path],
    frame_weight_paths: Sequence[str | Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate frame-level MSE data from multiple windows.

    Parameters
    ----------
    bin_idx_frame_paths
        Ordered sequence of ``bin_idx_CG_frame.npy`` / ``.npz`` files. Each file
        must contain one bin-index vector of shape ``(n_frames,)``.
    energy_grad_frame_paths
        Ordered sequence of pickle files produced by ``run_post`` with
        ``reduce_stack=True``. Each file must contain an ``energy_grad_frame``
        array of shape ``(n_frames, n_params)``.
    frame_weight_paths
        Ordered sequence of ``CG_FrameWeight.npy`` / ``.npz`` files containing
        one frame-weight vector per window.

    Returns
    -------
    CG_bin_idx_frame : np.ndarray
        Concatenated integer bin index per frame with shape
        ``(sum_i n_frames_i,)``.
    energy_grad_frame : np.ndarray
        Concatenated per-frame energy-gradient stack with shape
        ``(sum_i n_frames_i, n_params)``.
    frame_weight : np.ndarray
        Concatenated frame weights with shape ``(sum_i n_frames_i,)``.
    """
    bin_paths = [Path(path) for path in bin_idx_frame_paths]
    grad_paths = [Path(path) for path in energy_grad_frame_paths]
    weight_paths = [Path(path) for path in frame_weight_paths]
    n_windows = len(grad_paths)
    if len(bin_paths) != n_windows or len(weight_paths) != n_windows:
        raise ValueError(
            "bin_idx_frame_paths, energy_grad_frame_paths, and frame_weight_paths "
            "must have the same length; got "
            f"{len(bin_paths)}, {len(grad_paths)}, and {len(weight_paths)}."
        )
    if not grad_paths:
        raise ValueError("At least one window path set is required.")

    bin_blocks: list[np.ndarray] = []
    grad_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    n_params_ref: Optional[int] = None

    for bin_path, grad_path, weight_path in zip(bin_paths, grad_paths, weight_paths):
        with open(grad_path, "rb") as handle:
            payload = pickle.load(handle)
        if "energy_grad_frame" not in payload:
            raise KeyError(
                f"{grad_path} does not contain 'energy_grad_frame'. "
                "Expected a reduce_stack=True post-processing output."
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

        bin_idx_raw = _load_array_file(
            bin_path,
            preferred_keys=("CG_bin_idx_frame", "bin_idx_CG_frame", "bin_idx"),
        )
        bin_idx = np.asarray(bin_idx_raw).reshape(-1)
        if not np.issubdtype(bin_idx.dtype, np.integer):
            if not np.all(np.isfinite(bin_idx)) or not np.all(bin_idx == np.floor(bin_idx)):
                raise ValueError(f"{bin_path} must contain integer bin indices.")
        bin_idx_arr = bin_idx.astype(np.int64, copy=False)

        weight_raw = _load_array_file(
            weight_path,
            preferred_keys=("frame_weight", "CG_FrameWeight", "FrameWeight"),
        )
        weight_arr = np.asarray(weight_raw, dtype=np.float64).reshape(-1)

        n_frames = int(grad_frame.shape[0])
        if bin_idx_arr.size != n_frames:
            raise ValueError(
                f"{bin_path} length must match n_frames in {grad_path}: "
                f"got {bin_idx_arr.size} vs {n_frames}."
            )
        if weight_arr.size != n_frames:
            raise ValueError(
                f"{weight_path} length must match n_frames in {grad_path}: "
                f"got {weight_arr.size} vs {n_frames}."
            )
        if not np.all(np.isfinite(weight_arr)):
            raise ValueError(f"{weight_path} contains non-finite frame weights.")
        if np.any(weight_arr < 0.0):
            raise ValueError(f"{weight_path} contains negative frame weights.")

        bin_blocks.append(bin_idx_arr)
        grad_blocks.append(grad_frame)
        weight_blocks.append(weight_arr)

    frame_weight = np.concatenate(weight_blocks, axis=0)
    if float(np.sum(frame_weight)) <= 0.0:
        raise ValueError("Concatenated frame weights must have a positive sum.")

    return (
        np.concatenate(bin_blocks, axis=0),
        np.concatenate(grad_blocks, axis=0),
        frame_weight,
    )


# ── TypedDict schemas ───────────────────────────────────────────────

class MSEBatch(TypedDict, total=False):
    """
    Batch schema for MSETrainerAnalytic.step.

    Required keys
    -------------
    pmf_AA : np.ndarray, shape (n_bins,)
    pmf_CG : np.ndarray, shape (n_bins,)
    CG_bin_idx_frame : np.ndarray, shape (n_frames,)
        Bin index per frame (0..n_bins-1).

    CG-side energy gradients
    ------------------------
    energy_grad_frame : np.ndarray, shape (n_frames, n_params)
        Per-frame CG energy gradients.

    Optional keys
    -------------
    frame_weight : np.ndarray, shape (n_frames,)
        Per-frame weights used to compute CG conditional bin averages.
        If omitted, uniform frame weights are used.
    step_index : int
    """
    pmf_AA: Any
    pmf_CG: Any
    CG: NotRequired[EnsembleBatch]
    CG_bin_idx_frame: Any
    energy_grad_frame: Any
    frame_weight: NotRequired[Any]
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


# ── Trainer ───────────────────────────────────────────────

class MSETrainerAnalytic(BaseTrainer):
    """Analytic PMF-matching trainer with a gauge-fixed mean-squared objective.

    Let

        ΔF(s) = F_CG(s) - c - F_AA(s)

    where ``c`` is a gauge constant used to remove the arbitrary additive offset
    of the PMF.

    Objective
        ``loss = 1/2 Σ_s [F_CG(s) - c - F_AA(s)]^2``

        with gauge shift

        ``c = mean_s [F_CG(s) - F_AA(s)]``

    Using

        ``∂F_CG(s)/∂λ = ⟨dU/dλ⟩_{CG|s} - ⟨dU/dλ⟩_CG``

    and ``Σ_s ΔF(s) = 0`` after gauge fixing, the ensemble-average term cancels.
    The gradient is therefore

        ``∂loss/∂λ = Σ_s ΔF(s) * ⟨dU/dλ⟩_{CG|s}``

    where the conditional average uses per-frame weights when provided.
    Empty bins are skipped in the gradient because their conditional averages
    cannot be estimated from the available CG frames.

    Returns a dict with standardized keys (see module docstring).
    """

    # ── Public schema objects (for documentation / validation) ───────────────────────────────────────────────

    BATCH_SCHEMA: Dict[str, Any] = {
        "pmf_AA": "required np.ndarray; shape (n_bins,); target/reference PMF",
        "pmf_CG": "required np.ndarray; shape (n_bins,); current CG PMF",
        "energy_grad_frame": "required np.ndarray; shape (n_frames, n_params); per-frame CG energy gradients",
        "CG_bin_idx_frame": "required np.ndarray or dict; integer bin index per frame (0..n_bins-1)",
        "frame_weight": (
            "optional np.ndarray; shape (n_frames,); per-frame weights for "
            "conditional CG bin averages. If omitted, uniform weights are used."
        ),
        "step_index": "optional int; logging step counter; default 0",
    }

    RETURN_SCHEMA: Dict[str, Any] = {
        "name": 'str; always "MSE"',
        "loss": "float; gauge-fixed PMF mismatch objective; 1/2 Σ_s [F_CG(s)-c-F_AA(s)]^2",
        "grad": "np.ndarray; shape (n_params,); gradient of the gauge-fixed PMF mismatch wrt λ",
        "hessian": "None; reserved for uniform interface",
        "update": "np.ndarray; shape (n_params,); optimizer update if apply_update=True else zeros_like(grad)",
        "meta": "dict; diagnostics (step_index, gauge_shift, frame_weight_source, grad_norm, update_norm, ...)",
    }

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Return a dict with `batch` and `return` schema for introspection."""
        return {"batch": cls.BATCH_SCHEMA, "return": cls.RETURN_SCHEMA}

    @staticmethod
    def make_batch(
        pmf_AA,
        pmf_CG,
        CG_bin_idx_frame,
        energy_grad_frame,
        *,
        frame_weight=None,
        step_index: int = 0,
    ) -> MSEBatch:
        """
        Build a MSEBatch dict for MSETrainerAnalytic.step().

        Parameters
        ----------
        pmf_AA, pmf_CG
            PMFs defined on the same bins; shape (n_bins,).
        CG_bin_idx_frame
            Bin index per frame; shape (n_frames,).
        energy_grad_frame
            Per-frame CG energy gradients; shape (n_frames, n_params).
        frame_weight
            Optional per-frame weights used to compute conditional CG bin averages.
            If omitted, uniform frame weights are used.
        step_index
            Logging step counter.

        Returns
        -------
        MSEBatch
        """
        batch: MSEBatch = {
            "pmf_AA": pmf_AA,
            "pmf_CG": pmf_CG,
            "CG_bin_idx_frame": CG_bin_idx_frame,
            "energy_grad_frame": energy_grad_frame,
            "step_index": int(step_index),
        }
        if frame_weight is not None:
            batch["frame_weight"] = frame_weight
        return batch

    def step(self, batch: MSEBatch, apply_update: bool = True) -> MSEOut:
        """
        Execute one PMF-matching MSE optimization step.

        Parameters
        ----------
        batch : MSEBatch
            MSE batch dictionary.

            ``loss = 1/2 Σ_s [F_CG(s)-c-F_AA(s)]^2``
            with ``c = mean_s[F_CG(s)-F_AA(s)]``.

        apply_update : bool, default=True
            If True, apply the optimizer step and update the potential.
            If False, run in dry-run mode and return ``update=zeros_like(grad)``.

        Returns
        -------
        MSEOut
            Dictionary with standardized MSE outputs, including the scalar loss,
            gradient, and diagnostics.
        """
        assert isinstance(batch, dict), "MSETrainerAnalytic.step expects batch as a dict."
        pmf_AA = np.asarray(batch["pmf_AA"], dtype=float)
        pmf_CG = np.asarray(batch["pmf_CG"], dtype=float)
        CG_bin_idx_frame = batch["CG_bin_idx_frame"]
        step_index = int(batch.get("step_index", 0))

        # ── CG-side energy gradients ───────────────────────────────────────────────

        energy_grad_CG_frame = np.asarray(batch["energy_grad_frame"], dtype=np.float64)
        if energy_grad_CG_frame.ndim != 2:
            raise ValueError(
                "batch['energy_grad_frame'] must have shape (n_frames, n_params); "
                f"got {energy_grad_CG_frame.shape}"
            )

        n_frames, n_params = energy_grad_CG_frame.shape

        raw_frame_weight = batch.get("frame_weight")
        frame_weight_source = "batch" if raw_frame_weight is not None else "uniform"

        if raw_frame_weight is None:
            frame_weight = np.ones(n_frames, dtype=np.float64)
        else:
            frame_weight = np.asarray(raw_frame_weight, dtype=np.float64).reshape(-1)
            if frame_weight.shape != (n_frames,):
                raise ValueError(
                    "frame_weight must have shape (n_frames,), matching "
                    f"energy_grad_frame; got {frame_weight.shape} vs {(n_frames,)}"
                )
            if np.any(frame_weight < 0.0):
                raise ValueError("frame_weight must be nonnegative.")
            weight_sum = float(np.sum(frame_weight))
            if weight_sum <= 0.0 or not np.isfinite(weight_sum):
                raise ValueError("frame_weight must have a positive finite sum.")

        # ── Per-bin conditional averages (inlined) ───────────────────────────────────────────────

        bin_idx = np.asarray(CG_bin_idx_frame).reshape(-1)
        if bin_idx.shape != (n_frames,):
            raise ValueError(
                "CG_bin_idx_frame must have shape (n_frames,), matching "
                f"energy_grad_frame; got {bin_idx.shape} vs {(n_frames,)}"
            )
        if pmf_AA.ndim != 1 or pmf_CG.ndim != 1:
            raise ValueError("pmf_AA and pmf_CG must be one-dimensional arrays.")
        if pmf_AA.shape != pmf_CG.shape:
            raise ValueError(f"pmf_AA and pmf_CG shape mismatch: {pmf_AA.shape} vs {pmf_CG.shape}")
        n_bins = int(pmf_AA.shape[0])
        if np.any(bin_idx < 0) or np.any(bin_idx >= n_bins):
            raise ValueError("CG_bin_idx_frame contains bin indices outside [0, n_bins).")

        bin_idx_int = bin_idx.astype(np.int64, copy=False)
        weight_by_bin = np.bincount(bin_idx_int, weights=frame_weight, minlength=n_bins)
        idx_set = np.flatnonzero(weight_by_bin[:n_bins] > 0.0).astype(np.int64, copy=False)
        missing_bins = np.flatnonzero(weight_by_bin[:n_bins] <= 0.0)
        energy_grad_CG_given_bin: Dict[int, np.ndarray] = {}
        for sidx in idx_set:
            mask = bin_idx_int == sidx
            w_bin = frame_weight[mask]
            w_sum = float(np.sum(w_bin))
            if w_sum <= 0.0 or not np.isfinite(w_sum):
                raise ValueError(f"Bin {int(sidx)} has no positive finite frame weight.")
            w_bin = w_bin / w_sum
            energy_grad_CG_given_bin[int(sidx)] = energy_grad_CG_frame[mask].T @ w_bin

        # ── Gauge shift for PMF_CG ───────────────────────────────────────────────

        c = float(np.mean(pmf_CG - pmf_AA))
        pmf_CG_shifted = pmf_CG - c

        # ── Loss ───────────────────────────────────────────────

        delta = pmf_CG_shifted - pmf_AA
        loss = float(0.5 * np.sum(delta ** 2))

        # ── Gradient of loss w.r.t parameters ───────────────────────────────────────────────

        grad = np.zeros(n_params, dtype=float)
        for sidx in idx_set:
            grad += delta[int(sidx)] * energy_grad_CG_given_bin[int(sidx)]

        # ── Optimization step (optional) ───────────────────────────────────────────────

        if apply_update:
            update = self.optimizer.step(grad)
            self.clamp_and_update()
        else:
            update = np.zeros_like(grad)

        # ── Logging ───────────────────────────────────────────────

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
            # keep key for uniformity
            "hessian": None,
            "update": update,
            "meta": {
                "step_index": step_index,
                "gauge_shift": c,
                "frame_weight_source": frame_weight_source,
                "n_observed_bins": int(idx_set.size),
                "n_bins": n_bins,
                "missing_bins": missing_bins.tolist(),
                "grad_norm": float(np.linalg.norm(grad)),
                "update_norm": float(np.linalg.norm(update)),
            },
        }
