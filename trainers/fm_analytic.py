"""Iterative force-matching trainer based on analytical Jacobian assembly."""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from MDAnalysis import Universe
from tqdm import tqdm

from .base import BaseTrainer
from .utils import optimizer_accepts_hessian
from ..utils.bonded_projectors import FMInteraction, interaction_offsets
from ..utils.fm_common import tqdm_enabled, collect_topology_arrays, FMComponentMixin
from ..solvers.fm_matrix import fm_worker_chunk

try:
    from typing import TypedDict, NotRequired
except ImportError:
    from typing_extensions import TypedDict, NotRequired


class FMBatch(TypedDict, total=False):
    """Batch schema for FMTrainerAnalytic.step()."""
    start: int
    end: int
    every: NotRequired[int]
    frame_weight: NotRequired[Any]
    forces_by_frame: NotRequired[Dict[int, Any]]
    step_index: NotRequired[int]
    parallel: NotRequired[bool]
    trajectory: NotRequired[str]
    topology: NotRequired[Optional[str]]
    n_parts: NotRequired[int]
    n_workers: NotRequired[int]
    worker_blas_threads: NotRequired[int]


class FMTrainerAnalytic(FMComponentMixin, BaseTrainer):
    """Per-step FM trainer using analytical Jacobian and optional Gauss-Newton Hessian."""

    def __init__(
        self,
        universe: Universe,
        interactions: Sequence[FMInteraction],
        optimizer,
        *,
        cutoff: float,
        exclude: Any = "111",
        sel: str = "all",
        logger=None,
    ):
        self.u = universe
        self.cutoff = float(cutoff)
        self.exclude = exclude
        self.sel = str(sel)

        self._interaction_ids: List[str] = []
        potential_map = OrderedDict()
        for i, it in enumerate(interactions):
            key = f"{it.style}:{':'.join(it.types)}#{i}"
            self._interaction_ids.append(key)
            potential_map[key] = copy.deepcopy(it.potential)

        super().__init__(potential=potential_map, optimizer=optimizer, beta=None, logger=logger)

        self.interactions: List[FMInteraction] = []
        for i, it in enumerate(interactions):
            key = self._interaction_ids[i]
            pot = self.potential[key]
            self.interactions.append(
                FMInteraction(
                    style=it.style,
                    types=tuple(it.types),
                    potential=pot,
                    metadata=dict(it.metadata),
                )
            )

        self._init_fm_component(self.interactions)

    @staticmethod
    def make_batch(
        start: int,
        end: int,
        every: int = 1,
        *,
        step_index: int = 0,
        frame_weight=None,
        forces_by_frame=None,
        parallel: bool = False,
        trajectory: Optional[str] = None,
        topology: Optional[str] = None,
        n_parts: int = 8,
        n_workers: int = 8,
        worker_blas_threads: int = 1,
    ) -> FMBatch:
        """Build an FMBatch dict for FMTrainerAnalytic.step()."""
        batch: FMBatch = {
            "start": int(start),
            "end": int(end),
            "every": int(every),
            "step_index": int(step_index),
        }
        if frame_weight is not None:
            batch["frame_weight"] = frame_weight
        if forces_by_frame is not None:
            batch["forces_by_frame"] = forces_by_frame
        if parallel:
            batch["parallel"] = True
            if trajectory is not None:
                batch["trajectory"] = str(trajectory)
            if topology is not None:
                batch["topology"] = str(topology)
            batch["n_parts"] = int(n_parts)
            batch["n_workers"] = int(n_workers)
            batch["worker_blas_threads"] = int(worker_blas_threads)
        return batch

    def _frame_forces(self) -> np.ndarray:
        return np.asarray(self.u.trajectory.ts.forces, dtype=np.float64).reshape(-1)

    def _step_parallel_uniform(
        self,
        *,
        frame_ids: List[int],
        step_index: int,
        apply_update: bool,
        trajectory: str,
        topology: Optional[str],
        n_parts: int,
        n_workers: int,
        worker_blas_threads: int,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if len(frame_ids) == 0:
            raise ValueError("Empty frame window for FMTrainerAnalytic.step().")

        frames = np.asarray(frame_ids, dtype=np.int64)
        parts = [arr for arr in np.array_split(frames, int(n_parts)) if arr.size > 0]
        n_workers_eff = max(1, min(int(n_workers), len(parts)))
        topology_arrays = collect_topology_arrays(self.u)

        XtX = np.zeros((self._n_params, self._n_params), dtype=np.float64)
        XtY = np.zeros(self._n_params, dtype=np.float64)
        y_sumsq = 0.0
        nframe = 0

        with ProcessPoolExecutor(max_workers=n_workers_eff) as ex:
            futs = [
                ex.submit(
                    fm_worker_chunk,
                    trajectory=str(trajectory),
                    topology=topology,
                    topology_arrays=topology_arrays,
                    frame_ids=np.asarray(local_ids, dtype=np.int64),
                    interactions=self.interactions,
                    cutoff=float(self.cutoff),
                    exclude=self.exclude,
                    sel=self.sel,
                    param_mask=self._param_mask,
                    worker_blas_threads=int(worker_blas_threads),
                )
                for local_ids in parts
            ]
            pbar = tqdm(
                total=int(frames.size),
                desc="FM iter frames",
                leave=False,
                mininterval=2.0,
                unit="fr",
                disable=not tqdm_enabled(),
            )
            for fut in as_completed(futs):
                part = fut.result()
                n = int(part["nframe"])
                XtX += np.asarray(part["XtX"], dtype=np.float64)
                XtY += np.asarray(part["XtY"], dtype=np.float64)
                y_sumsq += float(part["y_sumsq"])
                nframe += n
                pbar.update(n)
            pbar.close()

        if nframe <= 0:
            raise ValueError("No frames accumulated in parallel iterative step.")

        c = np.asarray(self.optimizer.L, dtype=np.float64).reshape(-1)
        nframe_f = float(nframe)
        grad = (XtX @ c - XtY) / nframe_f
        loss = 0.5 * float(np.dot(c, XtX @ c) - 2.0 * np.dot(XtY, c) + y_sumsq) / nframe_f
        hessian: Optional[np.ndarray] = XtX / nframe_f
        if not optimizer_accepts_hessian(self.optimizer):
            hessian = None

        grad, hessian = self._apply_mask_to_grad_hessian(grad, hessian)
        to0 = time.perf_counter()
        update = self._optimizer_step_with_optional_hessian(grad, hessian, apply_update=apply_update)
        t_opt = time.perf_counter() - to0
        total_seconds = time.perf_counter() - t0

        n_obs_total = int(nframe) * int(self.u.atoms.n_atoms) * 3

        if self.logger is not None:
            self.logger.add_scalar("FM/loss", float(loss), step_index)
            self.logger.add_scalar("FM/grad_norm", float(np.linalg.norm(grad)), step_index)
            self.logger.add_scalar("FM/update_norm", float(np.linalg.norm(update)), step_index)

        return {
            "name": "FM",
            "loss": float(loss),
            "grad": grad,
            "hessian": hessian,
            "update": update,
            "meta": {
                "mode": "frames_parallel",
                "nframe": int(nframe),
                "n_obs_total": int(n_obs_total),
                "step_index": int(step_index),
                "timing": {
                    "build_ne_seconds": float(total_seconds - t_opt),
                    "optimizer_step_seconds": float(t_opt),
                    "total_seconds": float(total_seconds),
                },
            },
        }

    def _apply_mask_to_grad_hessian(
        self,
        grad: np.ndarray,
        hessian: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if np.all(self._param_mask):
            return grad, hessian

        g = np.asarray(grad, dtype=np.float64).copy()
        g[~self._param_mask] = 0.0
        if hessian is None:
            return g, None
        h = np.asarray(hessian, dtype=np.float64).copy()
        h[~self._param_mask, :] = 0.0
        h[:, ~self._param_mask] = 0.0
        return g, h

    def _optimizer_step_with_optional_hessian(
        self,
        grad: np.ndarray,
        hessian: Optional[np.ndarray],
        *,
        apply_update: bool,
    ) -> np.ndarray:
        if not apply_update:
            return np.zeros_like(grad)
        try:
            if optimizer_accepts_hessian(self.optimizer):
                if hessian is None:
                    raise ValueError("Hessian is required for the configured optimizer.")
                update = self.optimizer.step(grad, hessian=hessian)
            else:
                update = self.optimizer.step(grad)
        except np.linalg.LinAlgError:
            if hessian is None or (not hasattr(self.optimizer, "mask")) or (not hasattr(self.optimizer, "lr")):
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

    def step(self, batch: Dict[str, Any], apply_update: bool = True) -> Dict[str, Any]:
        """Run one FM optimization step over a frame range.

        Batch schema
        ------------
        {
          "start": int,            # inclusive
          "end": int,              # exclusive
          "every": int,            # optional, default 1
          "frame_weight": array-like | None,  # optional
          "forces_by_frame": dict[int, np.ndarray],  # optional overrides
          "step_index": int,       # optional
        }
        """
        start = int(batch["start"])
        end = int(batch["end"])
        every = int(batch.get("every", 1))
        step_index = int(batch.get("step_index", 0))

        frame_ids = list(range(start, end, every))
        if len(frame_ids) == 0:
            raise ValueError("Empty frame window for FMTrainerAnalytic.step().")

        frame_weight = batch.get("frame_weight")
        if frame_weight is None:
            w = np.ones(len(frame_ids), dtype=np.float64)
        else:
            w = np.asarray(frame_weight, dtype=np.float64).reshape(-1)
            if w.shape[0] != len(frame_ids):
                raise ValueError("frame_weight length mismatch with selected frame count.")
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            raise ValueError("Sum of frame weights must be positive.")
        w = w / w_sum

        t0 = time.perf_counter()
        t_build = 0.0
        t_accum = 0.0
        t_opt = 0.0

        grad = np.zeros(self._n_params, dtype=np.float64)
        hessian: Optional[np.ndarray]
        hessian = np.zeros((self._n_params, self._n_params), dtype=np.float64)
        loss = 0.0
        n_obs_total = 0

        force_override = batch.get("forces_by_frame", {})
        use_parallel = bool(batch.get("parallel", False))
        if use_parallel and frame_weight is not None:
            raise ValueError("parallel iterative step does not support explicit frame_weight.")
        if use_parallel and len(force_override) > 0:
            raise ValueError("parallel iterative step does not support forces_by_frame overrides.")
        if use_parallel:
            trajectory = batch.get("trajectory")
            if trajectory is None:
                try:
                    trajectory = str(self.u.trajectory.filename)
                except Exception as exc:
                    raise ValueError("Unable to resolve trajectory path; pass batch['trajectory'].") from exc
            topology = batch.get("topology")
            return self._step_parallel_uniform(
                frame_ids=frame_ids,
                step_index=step_index,
                apply_update=apply_update,
                trajectory=str(trajectory),
                topology=str(topology) if topology else None,
                n_parts=int(batch.get("n_parts", 8)),
                n_workers=int(batch.get("n_workers", 8)),
                worker_blas_threads=int(batch.get("worker_blas_threads", 1)),
            )
        self._trace(
            f"iterative step start: start={start}, end={end}, every={every}, n_frames={len(frame_ids)}, apply_update={apply_update}"
        )

        frame_bar = tqdm(
            frame_ids,
            desc="FM iter frames",
            leave=False,
            mininterval=2.0,
            disable=not tqdm_enabled(),
        )
        for iw, fr in enumerate(frame_bar):
            self._trace(f"frame={fr} iterative build_A start")
            tA = time.perf_counter()
            A = self._build_A(fr)
            t_build += time.perf_counter() - tA
            self._trace(f"frame={fr} iterative build_A done in {time.perf_counter() - tA:.2f} s")
            if fr in force_override:
                y = np.asarray(force_override[fr], dtype=np.float64).reshape(-1)
            else:
                y = self._frame_forces()
            tcalc = time.perf_counter()
            pred = A @ self.optimizer.L
            residual = y - pred
            wf = float(w[iw])

            loss += 0.5 * wf * float(np.dot(residual, residual))
            grad += -wf * (A.T @ residual)
            if optimizer_accepts_hessian(self.optimizer):
                hessian += wf * (A.T @ A)
            n_obs_total += residual.size
            t_accum += time.perf_counter() - tcalc
            self._trace(f"frame={fr} iterative accum done in {time.perf_counter() - tcalc:.2f} s")

        if not optimizer_accepts_hessian(self.optimizer):
            hessian = None
        grad, hessian = self._apply_mask_to_grad_hessian(grad, hessian)

        to0 = time.perf_counter()
        update = self._optimizer_step_with_optional_hessian(
            grad,
            hessian,
            apply_update=apply_update,
        )
        t_opt += time.perf_counter() - to0

        self._trace(
            f"iterative step finish: loss={float(loss):.6e}, grad_norm={float(np.linalg.norm(grad)):.6e}, update_norm={float(np.linalg.norm(update)):.6e}"
        )

        if self.logger is not None:
            self.logger.add_scalar("FM/loss", float(loss), step_index)
            self.logger.add_scalar("FM/grad_norm", float(np.linalg.norm(grad)), step_index)
            self.logger.add_scalar("FM/update_norm", float(np.linalg.norm(update)), step_index)

        return {
            "name": "FM",
            "loss": float(loss),
            "grad": grad,
            "hessian": hessian if optimizer_accepts_hessian(self.optimizer) else None,
            "update": update,
            "meta": {
                "mode": "frame_loop",
                "start": start,
                "end": end,
                "every": every,
                "n_frames": len(frame_ids),
                "n_obs_total": n_obs_total,
                "step_index": step_index,
                "timing": {
                    "build_design_seconds": float(t_build),
                    "grad_hessian_seconds": float(t_accum),
                    "optimizer_step_seconds": float(t_opt),
                    "total_seconds": float(time.perf_counter() - t0),
                },
            },
        }
