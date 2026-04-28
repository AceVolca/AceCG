"""Analytic MultiTrainerAnalytic: meta-trainer combining multiple sub-trainers."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

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

class MultiOut(TypedDict, total=False):
    """Return schema for MultiTrainerAnalytic.step."""
    mode: str
    update: Any
    combined_grad: NotRequired[Any]
    combined_hessian: NotRequired[Any]
    sub: List[Dict[str, Any]]
    meta: Dict[str, Any]


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

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
      vector via `tr.update_forcefield(self.optimizer.L)`.
    """

    # ---- Public schema objects (for documentation / validation) ----
    STEP_SCHEMA: Dict[str, Any] = {
        "batches": "required Sequence[dict]; length == len(trainers); batches[i] must satisfy trainers[i].BATCH_SCHEMA. For REM sub-batches, AA-side inputs may be supplied either as AA['dist'] with recompute_energy_grad_AA=True, or as precomputed energy_grad_AA (and d2U_AA when Hessian is required) with recompute_energy_grad_AA=False. CDREM sub-batches accept energy_grad_z_by_x, energy_grad_xz, and optional x_weight.",
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
        forcefield,
        optimizer,
        trainer_list: Sequence[BaseTrainer],
        weight_array: np.ndarray,
        beta: Optional[float] = None,
        logger=None,
        combine_mode: str = "update",
    ):
        """Initialize a meta-trainer over several analytic sub-trainers.

        Parameters
        ----------
        forcefield : Forcefield
            Global forcefield copied into the meta-trainer.
        optimizer : BaseOptimizer
            Optimizer used for the combined parameter vector.
        trainer_list : Sequence[BaseTrainer]
            Non-empty sequence of already configured sub-trainers. Each
            sub-trainer must use the same global parameter ordering.
        weight_array : np.ndarray, shape (n_trainers,)
            Linear weights applied to sub-trainer updates or gradients.
        beta : float, optional
            Optional inverse temperature stored for interface consistency.
        logger : object, optional
            Optional scalar logger exposing ``add_scalar``.
        combine_mode : {"update", "grad"}, default="update"
            ``"update"`` combines sub-trainer update vectors. ``"grad"``
            combines dry-run gradients and applies one meta-optimizer step.
        """
        super().__init__(forcefield, optimizer, beta, logger)

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
    def make_batches(
        *batches: Dict[str, Any],
        recompute_energy_grad_AA: Optional[bool] = None,
        override: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Convenience helper to build the `batches` list for MultiTrainerAnalytic.step().

        Parameters
        ----------
        *batches
            Per-trainer batch dicts. Ordering must match `self.trainers`.
        recompute_energy_grad_AA
            If not None, set `batch["recompute_energy_grad_AA"] = recompute_energy_grad_AA` for
            each sub-batch (unless `override=False` and the key already exists).
            This is mainly useful for REM batches. Non-REM trainers will simply
            ignore the extra key.
        override
            If True, overwrite existing keys in sub-batches. If False (default),
            preserve any per-batch custom settings already present.

        Returns
        -------
        list_of_batches : list[dict]
            The same objects passed in, collected into a list (mutated in-place
            if injection settings are provided).
        """
        out: List[Dict[str, Any]] = list(batches)

        if recompute_energy_grad_AA is None:
            return out

        for b in out:
            if not isinstance(b, dict):
                continue
            if recompute_energy_grad_AA is not None and (override or ("recompute_energy_grad_AA" not in b)):
                b["recompute_energy_grad_AA"] = bool(recompute_energy_grad_AA)

        return out


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
            contain all keys required by the corresponding trainer's ``step`` method.
            The ordering of ``batches`` must match the ordering of ``self.trainers``.

            In particular, REM sub-batches now support two AA-side input modes:

            - recompute mode:
              ``{"AA": {"dist": ...}, "CG": {"dist": ...}, "recompute_energy_grad_AA": True}``
            - cached-AA mode:
              ``{"energy_grad_AA": ..., "CG": {"dist": ...}, "recompute_energy_grad_AA": False}``
            - cached-AA mode:
              ``{"energy_grad_AA": ..., "CG": {"dist": ...}, "recompute_energy_grad_AA": False}``

            If the active optimizer accepts a Hessian, cached-AA REM batches must
            also include ``d2U_AA``.

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
            Dictionary with ``mode``, combined ``update``, filtered sub-trainer
            outputs under ``sub``, and diagnostics under ``meta``. In
            ``combine_mode == "grad"``, the dictionary also includes
            ``combined_grad`` and ``combined_hessian``.

        Notes
        -----
        - Sub-trainers may internally use multiprocessing for frame-parallel energy_grad evaluation (keys: parallel_energy_grad / energy_grad_n_parts / energy_grad_n_workers). This is independent of `parallel_grad`, which parallelizes across trainers via threads.
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

        use_hessian = self.optimizer_accepts_hessian()

        # full sub-trainer outputs
        sub_full: List[Dict[str, Any]] = []
        sub_view: List[Dict[str, Any]] = []  # returned full/filtered outputs

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

            # (n_trainers, n_params)
            U = np.stack(updates, axis=0)
            # (n_params,), linear combination of updates from subtrainers
            final_update = self.weights @ U

            self.optimizer.L += final_update
            self.clamp_and_update()

            # Sync sub-trainers to the new global L
            for tr in self.trainers:
                tr.update_forcefield(self.optimizer.L)

            if self.logger is not None:
                self.logger.add_scalar("Multi/update_norm", float(np.linalg.norm(final_update)), int(batches[0].get("step_index", 0)))

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

        # dry-run a trainer.step
        def _eval_one(args):
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


        # (n_trainers, n_params)
        G = np.stack(grads, axis=0)
        # (n_params,)
        g_total = self.weights @ G

        H_total = None
        if use_hessian and all(h is not None for h in Hs):
            # (n_trainers, n_params, n_params)
            H_stack = np.stack(Hs, axis=0)
            H_total = np.tensordot(self.weights, H_stack, axes=(0, 0))

        if H_total is not None:
            update = self.optimizer.step(g_total, hessian=H_total)
        else:
            update = self.optimizer.step(g_total)

        self.clamp_and_update()

        for tr in self.trainers:
            tr.update_forcefield(self.optimizer.L)

        # Logging
        step_index = int(batches[0].get("step_index", 0))
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
        """Set learning rates on all sub-trainer optimizers.

        Parameters
        ----------
        lrs : Sequence[float]
            One learning rate per sub-trainer, in the same order as
            ``self.trainers``.
        """
        assert isinstance(lrs, (list, tuple, np.ndarray)) and len(lrs) == len(self.trainers), (
            "lrs length must match the number of trainers"
        )
        for i, tr in enumerate(self.trainers):
            tr.optimizer.lr = float(lrs[i])
