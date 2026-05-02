"""Analytic CDREM (latent-variable) trainer."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    from typing import TypedDict, NotRequired
# Python < 3.11
except ImportError:
    from typing_extensions import TypedDict, NotRequired

from ..base import BaseTrainer


# -----------------------------------------------------------------------------
# TypedDict schemas
# -----------------------------------------------------------------------------

class CDREMBatch(TypedDict, total=False):
    """
    Batch schema for CDREMTrainerAnalytic.step.

    Required first-order keys
    -------------------------
    energy_grad_z_by_x : np.ndarray, shape (n_x, n_params)
        Row i is the conditional first-derivative average for one x-subsample:
            E_{z~q(z|x_i)} [ dU/dλ ]
    energy_grad_xz : np.ndarray, shape (n_params,)
        Joint-model first-derivative average:
            E_{(x,z)~q(x,z)} [ dU/dλ ]

    Optional weighting keys
    -----------------------
    x_weight : np.ndarray, shape (n_x,)
        Weight for each x-subsample. If omitted, uniform average over x is used.

    Optional second-order keys
    --------------------------
    d2U_z_by_x : np.ndarray, shape (n_x, n_params, n_params)
        Row i is the conditional second-derivative average for x_i:
            E_{z~q(z|x_i)} [ d²U/dλ_i dλ_j ]
    d2U_xz : np.ndarray, shape (n_params, n_params)
        Joint-model second-derivative average:
            E_{(x,z)~q(x,z)} [ d²U/dλ_i dλ_j ]
    energy_grad_outer_xz : np.ndarray, shape (n_params, n_params)
        Joint-model second moment of first derivatives:
            E_{(x,z)~q(x,z)} [ (dU/dλ)(dU/dλ)^T ]
    cov_z_by_x : np.ndarray, shape (n_x, n_params, n_params)
        Conditional covariance of first derivatives for each x_i:
            Cov_{z~q(z|x_i)} [ dU/dλ ]

    Misc
    ----
    step_index : int
        Logging step counter.
    """
    energy_grad_z_by_x: Any
    energy_grad_xz: Any
    x_weight: NotRequired[Any]
    d2U_z_by_x: NotRequired[Any]
    d2U_xz: NotRequired[Any]
    energy_grad_outer_xz: NotRequired[Any]
    cov_z_by_x: NotRequired[Any]
    step_index: NotRequired[int]


class CDREMOut(TypedDict, total=False):
    """
    Return schema for CDREMTrainerAnalytic.step.

    Common keys
    ----------
    name : str
    grad : np.ndarray
    hessian : np.ndarray | None
    update : np.ndarray
    meta : dict

    CDREM-specific keys
    -------------------
    energy_grad_pos : np.ndarray
        Positive-phase derivative E_xE_{z|x}[dU/dλ].
    energy_grad_neg : np.ndarray
        Negative-phase derivative E_{x,z}[dU/dλ].
    d2U_pos : np.ndarray, optional
        Positive-phase second-derivative average E_xE_{z|x}[d²U].
    d2U_neg : np.ndarray, optional
        Negative-phase second-derivative average E_{x,z}[d²U].
    cov_neg : np.ndarray, optional
        Joint covariance Cov_{x,z}[dU/dλ].
    cov_pos_cond : np.ndarray, optional
        Weighted conditional covariance E_x Cov_{z|x}[dU/dλ].
    """
    name: str
    grad: Any
    hessian: NotRequired[Any]
    update: Any
    energy_grad_pos: Any
    energy_grad_neg: Any
    d2U_pos: NotRequired[Any]
    d2U_neg: NotRequired[Any]
    cov_neg: NotRequired[Any]
    cov_pos_cond: NotRequired[Any]
    meta: Dict[str, Any]


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class CDREMTrainerAnalytic(BaseTrainer):
    """Analytic latent-variable / CDREM trainer.

    Implements the latent-variable CDREM gradient

        grad = β ( E_x E_{z|x}[dU/dλ] - E_{x,z}[dU/dλ] )

    and, when the optimizer requires a Hessian and the needed second-order
    statistics are provided in the batch, the latent-variable Hessian

        H = β ( E_xE_{z|x}[d²U] - E_{x,z}[d²U]
                + β ( Cov_{x,z}[dU/dλ] - E_x Cov_{z|x}[dU/dλ] ) )

    The second-order part is optional at the batch level. However, if the
    optimizer accepts a Hessian, the required second-order statistics must be
    provided; otherwise this trainer raises a ValueError.
    """

    BATCH_SCHEMA: Dict[str, Any] = {
        "energy_grad_z_by_x": (
            "required np.ndarray; shape (n_x, n_params); row i is "
            "E_{z~q(z|x_i)}[dU/dλ]"
        ),
        "energy_grad_xz": (
            "required np.ndarray; shape (n_params,); "
            "E_{(x,z)~q(x,z)}[dU/dλ]"
        ),
        "x_weight": (
            "optional np.ndarray; shape (n_x,); weight for each x-subsample; "
            "if omitted, uniform average over x is used"
        ),
        "d2U_z_by_x": (
            "optional np.ndarray; shape (n_x, n_params, n_params); row i is "
            "E_{z~q(z|x_i)}[d²U/dλ_i dλ_j]; required if optimizer needs Hessian"
        ),
        "d2U_xz": (
            "optional np.ndarray; shape (n_params, n_params); "
            "E_{(x,z)~q(x,z)}[d²U/dλ_i dλ_j]; required if optimizer needs Hessian"
        ),
        "energy_grad_outer_xz": (
            "optional np.ndarray; shape (n_params, n_params); "
            "E_{(x,z)~q(x,z)}[(dU/dλ)(dU/dλ)^T]; required if optimizer needs Hessian"
        ),
        "cov_z_by_x": (
            "optional np.ndarray; shape (n_x, n_params, n_params); row i is "
            "Cov_{z~q(z|x_i)}[dU/dλ]; required if optimizer needs Hessian"
        ),
        "step_index": "optional int; logging step counter; default 0",
    }

    RETURN_SCHEMA: Dict[str, Any] = {
        "name": 'str; always "CDREM"',
        "grad": (
            "np.ndarray; shape (n_params,); "
            "beta*(E_xE_{z|x}[dU/dλ] - E_{x,z}[dU/dλ])"
        ),
        "hessian": (
            "np.ndarray|None; shape (n_params,n_params); latent-variable Hessian "
            "if optimizer_accepts_hessian is True and second-order batch stats are provided"
        ),
        "update": (
            "np.ndarray; shape (n_params,); "
            "optimizer update if apply_update=True else zeros_like(grad)"
        ),
        "energy_grad_pos": (
            "np.ndarray; shape (n_params,); positive-phase derivative "
            "E_xE_{z|x}[dU/dλ]"
        ),
        "energy_grad_neg": (
            "np.ndarray; shape (n_params,); negative-phase derivative "
            "E_{x,z}[dU/dλ]"
        ),
        "d2U_pos": (
            "optional np.ndarray; shape (n_params,n_params); positive-phase "
            "second-derivative average E_xE_{z|x}[d²U]"
        ),
        "d2U_neg": (
            "optional np.ndarray; shape (n_params,n_params); negative-phase "
            "second-derivative average E_{x,z}[d²U]"
        ),
        "cov_neg": (
            "optional np.ndarray; shape (n_params,n_params); joint covariance "
            "Cov_{x,z}[dU/dλ]"
        ),
        "cov_pos_cond": (
            "optional np.ndarray; shape (n_params,n_params); weighted conditional "
            "covariance E_x Cov_{z|x}[dU/dλ]"
        ),
        "meta": "dict; diagnostics (step_index, grad_norm, update_norm, n_x, ...)",
    }

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        """Return a dict with `batch` and `return` schema for introspection."""
        return {"batch": cls.BATCH_SCHEMA, "return": cls.RETURN_SCHEMA}

    @staticmethod
    def make_batch(
        energy_grad_z_by_x,
        energy_grad_xz,
        x_weight=None,
        d2U_z_by_x=None,
        d2U_xz=None,
        energy_grad_outer_xz=None,
        cov_z_by_x=None,
        step_index: int = 0,
    ) -> CDREMBatch:
        """
        Build a CDREMBatch dict for CDREMTrainerAnalytic.step().

        Parameters
        ----------
        energy_grad_z_by_x : np.ndarray, shape (n_x, n_params)
            Row i is the conditional first-derivative average for one x-subsample:
                E_{z~q(z|x_i)}[dU/dλ]
        energy_grad_xz : np.ndarray, shape (n_params,)
            Joint-model first-derivative average:
                E_{(x,z)~q(x,z)}[dU/dλ]
        x_weight : np.ndarray, optional, shape (n_x,)
            Weight for each x-subsample. If None, uniform averaging over x is used.
        d2U_z_by_x : np.ndarray, optional, shape (n_x, n_params, n_params)
            Conditional second-derivative average for each x-subsample.
        d2U_xz : np.ndarray, optional, shape (n_params, n_params)
            Joint-model second-derivative average.
        energy_grad_outer_xz : np.ndarray, optional, shape (n_params, n_params)
            Joint-model second moment of first derivatives.
        cov_z_by_x : np.ndarray, optional, shape (n_x, n_params, n_params)
            Conditional covariance of first derivatives for each x-subsample.
        step_index : int
            Logging step counter.

        Returns
        -------
        CDREMBatch
            Batch dictionary for CDREMTrainerAnalytic.step().
        """
        batch: CDREMBatch = {
            "energy_grad_z_by_x": energy_grad_z_by_x,
            "energy_grad_xz": energy_grad_xz,
            "step_index": int(step_index),
        }
        if x_weight is not None:
            batch["x_weight"] = x_weight
        if d2U_z_by_x is not None:
            batch["d2U_z_by_x"] = d2U_z_by_x
        if d2U_xz is not None:
            batch["d2U_xz"] = d2U_xz
        if energy_grad_outer_xz is not None:
            batch["energy_grad_outer_xz"] = energy_grad_outer_xz
        if cov_z_by_x is not None:
            batch["cov_z_by_x"] = cov_z_by_x
        return batch

    def step(self, batch: CDREMBatch, apply_update: bool = True) -> CDREMOut:
        """
        Execute one CDREM optimization step.

        First-order statistics are always required. Second-order statistics are
        optional unless the optimizer accepts a Hessian, in which case they are
        required and this method raises a ValueError if any are missing.

        Parameters
        ----------
        batch : CDREMBatch
            CDREM batch dictionary.
        apply_update : bool, default=True
            If True, apply the optimizer step and update the potential.
            If False, run in dry-run mode and return update=zeros_like(grad).

        Returns
        -------
        CDREMOut
            Dictionary with standardized CDREM outputs.
        """
        assert isinstance(batch, dict), "CDREMTrainerAnalytic.step expects batch as a dict."

        energy_grad_z_by_x = np.asarray(batch["energy_grad_z_by_x"], dtype=float)
        energy_grad_xz = np.asarray(batch["energy_grad_xz"], dtype=float)
        step_index = int(batch.get("step_index", 0))
        need_hessian = self.optimizer_accepts_hessian()

        if energy_grad_z_by_x.ndim != 2:
            raise ValueError(
                f"batch['energy_grad_z_by_x'] must be 2D, got shape {energy_grad_z_by_x.shape}"
            )
        if energy_grad_xz.ndim != 1:
            raise ValueError(
                f"batch['energy_grad_xz'] must be 1D, got shape {energy_grad_xz.shape}"
            )
        if energy_grad_z_by_x.shape[1] != energy_grad_xz.shape[0]:
            raise ValueError(
                "Dimension mismatch: energy_grad_z_by_x has shape "
                f"{energy_grad_z_by_x.shape}, but energy_grad_xz has shape {energy_grad_xz.shape}"
            )

        n_x = energy_grad_z_by_x.shape[0]
        n_params = energy_grad_z_by_x.shape[1]
        if n_x == 0:
            raise ValueError("batch['energy_grad_z_by_x'] must contain at least one x-subsample.")

        x_weight = batch.get("x_weight", None)
        if x_weight is None:
            w_x = np.ones(n_x, dtype=float) / float(n_x)
        else:
            w_x = np.asarray(x_weight, dtype=float)
            if w_x.ndim != 1:
                raise ValueError(
                    f"batch['x_weight'] must be 1D, got shape {w_x.shape}"
                )
            if w_x.shape[0] != n_x:
                raise ValueError(
                    f"batch['x_weight'] length {w_x.shape[0]} != n_x {n_x}"
                )
            if np.any(w_x < 0.0):
                raise ValueError("batch['x_weight'] must be nonnegative.")
            w_sum = float(np.sum(w_x))
            if w_sum <= 0.0:
                raise ValueError("batch['x_weight'] must have positive sum.")
            w_x = w_x / w_sum

        energy_grad_pos = w_x @ energy_grad_z_by_x
        energy_grad_neg = energy_grad_xz
        grad = self.beta * (energy_grad_pos - energy_grad_neg)

        d2U_pos = None
        d2U_neg = None
        cov_neg = None
        cov_pos_cond = None
        hessian = None

        if need_hessian:
            required = ["d2U_z_by_x", "d2U_xz", "energy_grad_outer_xz", "cov_z_by_x"]
            missing = [k for k in required if k not in batch]
            if missing:
                raise ValueError(
                    "CDREMTrainerAnalytic.step requires second-order batch statistics "
                    "when optimizer_accepts_hessian(self.optimizer) is True. Missing keys: "
                    + ", ".join(missing)
                )

            d2U_z_by_x = np.asarray(batch["d2U_z_by_x"], dtype=float)
            d2U_xz = np.asarray(batch["d2U_xz"], dtype=float)
            energy_grad_outer_xz = np.asarray(batch["energy_grad_outer_xz"], dtype=float)
            cov_z_by_x = np.asarray(batch["cov_z_by_x"], dtype=float)

            if d2U_z_by_x.ndim != 3 or d2U_z_by_x.shape != (n_x, n_params, n_params):
                raise ValueError(
                    "batch['d2U_z_by_x'] must have shape "
                    f"(n_x, n_params, n_params)=({n_x}, {n_params}, {n_params}), "
                    f"got {d2U_z_by_x.shape}"
                )
            if d2U_xz.ndim != 2 or d2U_xz.shape != (n_params, n_params):
                raise ValueError(
                    f"batch['d2U_xz'] must have shape ({n_params}, {n_params}), got {d2U_xz.shape}"
                )
            if energy_grad_outer_xz.ndim != 2 or energy_grad_outer_xz.shape != (n_params, n_params):
                raise ValueError(
                    "batch['energy_grad_outer_xz'] must have shape "
                    f"({n_params}, {n_params}), got {energy_grad_outer_xz.shape}"
                )
            if cov_z_by_x.ndim != 3 or cov_z_by_x.shape != (n_x, n_params, n_params):
                raise ValueError(
                    "batch['cov_z_by_x'] must have shape "
                    f"(n_x, n_params, n_params)=({n_x}, {n_params}, {n_params}), "
                    f"got {cov_z_by_x.shape}"
                )

            d2U_pos = np.tensordot(w_x, d2U_z_by_x, axes=(0, 0))
            d2U_neg = d2U_xz
            cov_neg = energy_grad_outer_xz - np.outer(energy_grad_neg, energy_grad_neg)
            cov_pos_cond = np.tensordot(w_x, cov_z_by_x, axes=(0, 0))

            hessian = self.beta * (
                d2U_pos - d2U_neg + self.beta * (cov_neg - cov_pos_cond)
            )

        if apply_update:
            if need_hessian:
                update = self.optimizer.step(grad, hessian=hessian)
            else:
                update = self.optimizer.step(grad)
            self.clamp_and_update()
        else:
            update = np.zeros_like(grad)

        if self.logger is not None:
            mask_ratio = float(np.mean(self.optimizer.mask.astype(float)))
            self.logger.add_scalar("CDREM/mask_ratio", mask_ratio, step_index)
            self.logger.add_scalar("CDREM/lr", float(getattr(self.optimizer, "lr", np.nan)), step_index)
            self.logger.add_scalar("CDREM/grad_norm", float(np.linalg.norm(grad)), step_index)
            self.logger.add_scalar("CDREM/update_norm", float(np.linalg.norm(update)), step_index)
            self.logger.add_scalar("CDREM/n_x", float(n_x), step_index)

        out: CDREMOut = {
            "name": "CDREM",
            "grad": grad,
            "hessian": hessian,
            "update": update,
            "energy_grad_pos": energy_grad_pos,
            "energy_grad_neg": energy_grad_neg,
            "meta": {
                "step_index": step_index,
                "n_x": int(n_x),
                "grad_norm": float(np.linalg.norm(grad)),
                "update_norm": float(np.linalg.norm(update)),
                "used_hessian": bool(need_hessian),
            },
        }
        if d2U_pos is not None:
            out["d2U_pos"] = d2U_pos
        if d2U_neg is not None:
            out["d2U_neg"] = d2U_neg
        if cov_neg is not None:
            out["cov_neg"] = cov_neg
        if cov_pos_cond is not None:
            out["cov_pos_cond"] = cov_pos_cond
        return out
