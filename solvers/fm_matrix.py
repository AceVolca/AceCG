"""Matrix-based force matching solver (OpenMSCG-style normal equations)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import MDAnalysis as mda
import numpy as np
from MDAnalysis import Universe
from tqdm import tqdm

from .base import BaseSolver
from ..utils.bonded_projectors import FMInteraction, interaction_offsets
from ..utils.fm_common import (
    tqdm_enabled,
    set_worker_blas_threads,
    collect_topology_arrays,
    attach_topology_arrays,
    normalize_diagonal,
    normalize_strict,
    FMComponentMixin,
)

try:
    from typing import TypedDict, NotRequired
except ImportError:
    from typing_extensions import TypedDict, NotRequired


class FMSolverBatch(TypedDict, total=False):
    """Batch schema for FMMatrixSolver.run() / run_parallel()."""
    start: int
    end: int
    every: NotRequired[int]
    parallel: NotRequired[bool]
    n_parts: NotRequired[int]
    n_workers: NotRequired[Optional[int]]
    trajectory: NotRequired[Optional[str]]
    topology: NotRequired[Optional[str]]
    worker_blas_threads: NotRequired[int]


def fm_worker_chunk(
    *,
    trajectory: str,
    topology: Optional[str],
    topology_arrays: Mapping[str, np.ndarray],
    frame_ids: np.ndarray,
    interactions: Sequence[FMInteraction],
    cutoff: float,
    exclude: Any,
    sel: str,
    param_mask: np.ndarray,
    worker_blas_threads: int,
) -> Dict[str, Any]:
    """Accumulate FM normal equations for a chunk of frames in a worker process."""
    set_worker_blas_threads(int(worker_blas_threads))
    if topology is None:
        u = mda.Universe(trajectory, format="LAMMPSDUMP")
    else:
        u = mda.Universe(topology, trajectory, format="LAMMPSDUMP")
    attach_topology_arrays(u, topology_arrays)

    solver = FMMatrixSolver(
        universe=u,
        interactions=interactions,
        cutoff=float(cutoff),
        exclude=exclude,
        sel=str(sel),
        alpha=0.0,
        bayesian=0,
    )
    solver.set_param_mask(np.asarray(param_mask, dtype=bool))

    frames = np.asarray(frame_ids, dtype=np.int64)
    for fr in frames:
        solver.accumulate_frame(int(fr))

    return {
        "XtX": solver._XtX,
        "XtY": solver._XtY,
        "y_sumsq": float(solver._y_sumsq),
        "nframe": int(solver._nframe),
        "timing": solver.get_timing_summary(),
    }


@dataclass
class SolverState:
    n_atoms: int
    n_params: int
    nframe: int
    y_sumsq: float


@dataclass
class SolverTiming:
    frame_build_seconds: float = 0.0
    frame_load_force_seconds: float = 0.0
    frame_accumulate_seconds: float = 0.0
    frame_total_seconds: float = 0.0
    n_frames_accumulated: int = 0
    run_serial_seconds: float = 0.0
    run_parallel_seconds: float = 0.0
    parallel_dispatch_seconds: float = 0.0
    parallel_reduce_seconds: float = 0.0
    solve_ols_seconds: float = 0.0
    solve_bayesian_seconds: float = 0.0
    finalize_seconds: float = 0.0


class FMMatrixSolver(FMComponentMixin, BaseSolver):
    """FM normal-equation builder + OLS/Ridge/Bayesian solver."""

    def __init__(
        self,
        universe: Universe,
        interactions: Sequence[FMInteraction],
        *,
        cutoff: float,
        exclude: Any = "111",
        sel: str = "all",
        alpha: float = 0.0,
        bayesian: int = 0,
        bayesian_tol: float = 1.0e-6,
        bayesian_mode: str = "diagonal",
        bayesian_min_iter: Optional[int] = None,
    ):
        self.u = universe
        self.interactions = list(interactions)
        self.cutoff = float(cutoff)
        self.exclude = exclude
        self.sel = str(sel)
        self.alpha = float(alpha)
        self.bayesian = int(bayesian)
        self.bayesian_tol = float(bayesian_tol)
        mode = str(bayesian_mode).strip().lower()
        if mode in {"openmscg", "diagonal"}:
            mode = "diagonal"
        elif mode != "strict":
            raise ValueError("bayesian_mode must be one of {'diagonal', 'strict'}")
        self.bayesian_mode = mode
        if bayesian_min_iter is None:
            bayesian_min_iter = 10 if mode == "diagonal" else 20
        self.bayesian_min_iter = int(bayesian_min_iter)

        self._init_fm_component(self.interactions)
        self._coeff = np.zeros(self._n_params, dtype=np.float64)

        n_atoms = len(self.u.atoms)
        self._XtX = np.zeros((self._n_params, self._n_params), dtype=np.float64)
        self._XtY = np.zeros(self._n_params, dtype=np.float64)
        self._y_sumsq = 0.0
        self._nframe = 0
        self.state = SolverState(n_atoms=n_atoms, n_params=self._n_params, nframe=0, y_sumsq=0.0)
        self._timing = SolverTiming()
        self._bayesian_iterations_run = 0

    @staticmethod
    def make_batch(
        start: int,
        end: int,
        every: int = 1,
        *,
        parallel: bool = False,
        n_parts: int = 8,
        n_workers: Optional[int] = None,
        trajectory: Optional[str] = None,
        topology: Optional[str] = None,
        worker_blas_threads: int = 1,
    ) -> FMSolverBatch:
        """Build an FMSolverBatch dict for run_batch()."""
        batch: FMSolverBatch = {
            "start": int(start),
            "end": int(end),
            "every": int(every),
        }
        if parallel:
            batch["parallel"] = True
            batch["n_parts"] = int(n_parts)
            if n_workers is not None:
                batch["n_workers"] = int(n_workers)
            if trajectory is not None:
                batch["trajectory"] = str(trajectory)
            if topology is not None:
                batch["topology"] = str(topology)
            batch["worker_blas_threads"] = int(worker_blas_threads)
        return batch

    def run_batch(self, batch: FMSolverBatch) -> None:
        """Run accumulation from an FMSolverBatch dict."""
        start = int(batch["start"])
        end = int(batch["end"])
        every = int(batch.get("every", 1))
        if batch.get("parallel", False):
            self.run_parallel(
                start, end, every,
                n_parts=int(batch.get("n_parts", 8)),
                n_workers=batch.get("n_workers"),
                trajectory=batch.get("trajectory"),
                topology=batch.get("topology"),
                worker_blas_threads=int(batch.get("worker_blas_threads", 1)),
            )
        else:
            self.run(start, end, every)

    def _normalize_matrix(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.bayesian_mode == "diagonal":
            return normalize_diagonal(X)
        return normalize_strict(X)

    def _build_A(self, frame: int) -> np.ndarray:
        t0 = time.perf_counter()
        A = super()._build_A(frame)
        self._trace(f"frame={frame} build_design_matrix done in {time.perf_counter() - t0:.2f} s")
        return A

    def accumulate_frame(self, frame: int, forces: Optional[np.ndarray] = None) -> None:
        self._trace(f"frame={frame} start accumulate")
        t0 = time.perf_counter()
        A = self._build_A(frame)
        t1 = time.perf_counter()
        t_force = time.perf_counter()
        if forces is None:
            y = np.asarray(self.u.trajectory.ts.forces, dtype=np.float64).reshape(-1)
        else:
            y = np.asarray(forces, dtype=np.float64).reshape(-1)
        self._trace(f"frame={frame} load forces in {time.perf_counter() - t_force:.2f} s")
        t2 = time.perf_counter()

        t_acc = time.perf_counter()
        self._XtX += A.T @ A
        self._XtY += A.T @ y
        self._y_sumsq += float(np.square(y).sum())
        self._nframe += 1
        self.state.nframe = self._nframe
        self.state.y_sumsq = self._y_sumsq
        t3 = time.perf_counter()
        self._trace(f"frame={frame} accumulate XtX/XtY in {t3 - t_acc:.2f} s (nframe={self._nframe})")

        self._timing.frame_build_seconds += t1 - t0
        self._timing.frame_load_force_seconds += t2 - t1
        self._timing.frame_accumulate_seconds += t3 - t2
        self._timing.frame_total_seconds += t3 - t0
        self._timing.n_frames_accumulated += 1

    def step(self, batch: Dict[str, Any]) -> None:
        frame = int(batch["frame"])
        forces = batch.get("forces")
        self.accumulate_frame(frame=frame, forces=forces)

    def run(self, start: int, end: int, every: int = 1) -> None:
        self._trace(f"run start: start={start}, end={end}, every={every}")
        t0 = time.perf_counter()
        frame_iter = range(int(start), int(end), int(every))
        for frame in tqdm(
            frame_iter,
            desc="FM frames",
            leave=False,
            mininterval=2.0,
            disable=not tqdm_enabled(),
        ):
            self.accumulate_frame(frame=frame)
        self._timing.run_serial_seconds += time.perf_counter() - t0
        self._trace(f"run finish: total_frames={self._nframe}")

    def run_parallel(
        self,
        start: int,
        end: int,
        every: int = 1,
        *,
        n_parts: int = 8,
        n_workers: Optional[int] = None,
        trajectory: Optional[str] = None,
        topology: Optional[str] = None,
        chunk_dir: str | Path = "traj_chunks",
        keep_chunks: bool = False,
        worker_blas_threads: int = 1,
    ) -> None:
        """Parallel FM accumulation by frame partitioning + matrix reduction."""
        self._trace(
            f"run_parallel start: start={start}, end={end}, every={every}, n_parts={n_parts}, n_workers={n_workers}"
        )
        t_par0 = time.perf_counter()
        if n_workers is None:
            n_workers = int(n_parts)
        if n_parts < 1:
            raise ValueError("n_parts must be >= 1")
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")

        if trajectory is None:
            try:
                trajectory = str(self.u.trajectory.filename)
            except Exception as exc:
                raise ValueError("Unable to resolve trajectory path; pass trajectory=...") from exc

        traj_path = Path(trajectory)
        if not traj_path.exists():
            raise FileNotFoundError(traj_path)

        frame_ids = np.arange(int(start), int(end), int(every), dtype=np.int64)
        if frame_ids.size == 0:
            return

        topology_arrays = collect_topology_arrays(self.u)
        parts = [arr for arr in np.array_split(frame_ids, int(n_parts)) if arr.size > 0]
        n_workers_eff = min(int(n_workers), len(parts))
        t_dispatch0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_workers_eff) as ex:
            futs = [
                ex.submit(
                    fm_worker_chunk,
                    trajectory=str(traj_path),
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
            self._timing.parallel_dispatch_seconds += time.perf_counter() - t_dispatch0

            t_reduce0 = time.perf_counter()
            pbar = tqdm(
                total=int(frame_ids.size),
                desc="FM frames",
                leave=False,
                mininterval=2.0,
                unit="fr",
                disable=not tqdm_enabled(),
            )
            for fut in as_completed(futs):
                part = fut.result()
                n = int(part["nframe"])
                self._XtX += np.asarray(part["XtX"], dtype=np.float64)
                self._XtY += np.asarray(part["XtY"], dtype=np.float64)
                self._y_sumsq += float(part["y_sumsq"])
                self._nframe += n
                pbar.update(n)
                self._trace(f"chunk reduced: +{n} frames (total={self._nframe})")
            pbar.close()
            self._timing.parallel_reduce_seconds += time.perf_counter() - t_reduce0

        self.state.nframe = int(self._nframe)
        self.state.y_sumsq = float(self._y_sumsq)
        self._timing.run_parallel_seconds += time.perf_counter() - t_par0
        self._trace(f"run_parallel finish: total_frames={self._nframe}")

    @staticmethod
    def _get_residual(XtX: np.ndarray, XtY: np.ndarray, y_sumsq: float, c: np.ndarray) -> float:
        return float(np.dot(c, XtX @ c) - 2.0 * np.dot(XtY, c) + y_sumsq)

    def _solve_ols_or_ridge(self, XtX: np.ndarray, XtY: np.ndarray, alpha: float) -> np.ndarray:
        if alpha > 1.0e-6:
            XtX_n, scale = self._normalize_matrix(XtX)
            XtX_n = XtX_n + np.eye(XtX_n.shape[0], dtype=np.float64) * (alpha * alpha)
            c, *_ = np.linalg.lstsq(XtX_n, XtY, rcond=None)
            return c / scale
        c, *_ = np.linalg.lstsq(XtX, XtY, rcond=None)
        return c

    def _solve_bayesian(self, XtX: np.ndarray, XtY: np.ndarray, c0: np.ndarray) -> Dict[str, Any]:
        n = XtX.shape[0]
        N = self.state.n_atoms * max(self._nframe, 1)
        c = c0.copy()

        _a = n / max(float(np.dot(c, c)), 1.0e-30)
        _b = N / max(self._get_residual(XtX, XtY, self._y_sumsq, c), 1.0e-30)
        a = np.ones(n, dtype=np.float64) * _a
        eye = np.eye(n, dtype=np.float64)
        beta_hist = [_b]
        alpha_hist = [a.copy()]
        beta_rel_hist = [0.0]
        iter_seconds: List[float] = []
        converged_iter: Optional[int] = None

        _b_prev = _b
        c_prev = c.copy()
        min_iter = max(self.bayesian_min_iter, 0)
        for it in tqdm(range(self.bayesian)):
            t_it0 = time.perf_counter()
            XtX_reg = XtX + eye * (a / _b)

            XtX_norm, scale = self._normalize_matrix(XtX_reg)
            c, *_ = np.linalg.lstsq(XtX_norm, XtY, rcond=None)
            c = c / scale

            invXtX_reg = np.linalg.inv(XtX_reg)
            a = 1.0 / (c * c + np.diagonal(invXtX_reg) / _b)

            tr = float(np.trace(np.matmul(invXtX_reg, XtX)))
            _b = (N - tr) / max(self._get_residual(XtX, XtY, self._y_sumsq, c), 1.0e-30)
            beta_hist.append(_b)
            alpha_hist.append(a.copy())
            beta_denom = max(abs(_b), 1.0e-30)
            beta_rel = abs(_b - _b_prev) / beta_denom
            beta_rel_hist.append(beta_rel)
            iter_seconds.append(time.perf_counter() - t_it0)

            if it >= min_iter:
                should_stop = False
                if self.bayesian_mode == "diagonal":
                    should_stop = beta_rel < self.bayesian_tol
                else:
                    c_denom = max(float(np.linalg.norm(c)), 1.0e-30)
                    c_rel = float(np.linalg.norm(c - c_prev)) / c_denom
                    should_stop = beta_rel < self.bayesian_tol and c_rel < self.bayesian_tol
                if should_stop:
                    converged_iter = it + 1
                    break
            _b_prev = _b
            c_prev = c.copy()
        self._bayesian_iterations_run = len(beta_hist) - 1

        return {
            "c": c,
            "alpha_hist": alpha_hist,
            "beta_hist": beta_hist,
            "beta_rel_hist": beta_rel_hist,
            "iter_seconds": iter_seconds,
            "converged_iter": converged_iter,
        }

    def _masked_solve(self) -> Dict[str, Any]:
        idx = np.where(self._param_mask)[0]
        if idx.size == 0:
            return {"c": np.zeros(self._n_params, dtype=np.float64), "bayesian": None}

        XtX = self._XtX[np.ix_(idx, idx)]
        XtY = self._XtY[idx]
        t_ols0 = time.perf_counter()
        c_sub = self._solve_ols_or_ridge(XtX, XtY, self.alpha)
        self._timing.solve_ols_seconds += time.perf_counter() - t_ols0

        bayes_out = None
        if self.bayesian > 0:
            t_bayes0 = time.perf_counter()
            bayes_out = self._solve_bayesian(XtX, XtY, c_sub)
            self._timing.solve_bayesian_seconds += time.perf_counter() - t_bayes0
            c_sub = bayes_out["c"]

        c = np.zeros(self._n_params, dtype=np.float64)
        c[idx] = c_sub
        return {"c": c, "bayesian": bayes_out}

    def get_params(self) -> np.ndarray:
        return self._coeff.copy()

    def update_potential(self, params: np.ndarray) -> None:
        params = np.asarray(params, dtype=np.float64).reshape(-1)
        if params.shape != (self._n_params,):
            raise ValueError(f"parameter shape mismatch: expected {(self._n_params,)}, got {params.shape}")
        for it, sl in zip(self.interactions, self._offsets):
            it.potential.set_params(params[sl])
        self._coeff = params.copy()

    def get_timing_summary(self) -> Dict[str, Any]:
        return {
            **asdict(self._timing),
            "bayesian_iterations_run": int(self._bayesian_iterations_run),
        }

    def finalize(self) -> Dict[str, Any]:
        t_fin0 = time.perf_counter()
        solved = self._masked_solve()
        c = np.asarray(solved["c"], dtype=np.float64)
        self.update_potential(c)
        self._timing.finalize_seconds += time.perf_counter() - t_fin0

        inter_info = []
        for it, sl in zip(self.interactions, self._offsets):
            inter_info.append(
                {
                    "style": it.style,
                    "types": tuple(it.types),
                    "label": it.label(),
                    "n_params": it.n_params(),
                    "offset": [sl.start, sl.stop],
                    "metadata": dict(it.metadata),
                }
            )

        out = {
            "format": "acecg.fm.matrix.v1",
            "coefficients": c,
            "interactions": inter_info,
            "matrix": {
                "XtX": self._XtX.copy(),
                "XtY": self._XtY.copy(),
                "y_sumsq": float(self._y_sumsq),
                "row_per_frame": int(self.state.n_atoms),
                "nframe": int(self._nframe),
            },
            "solver": {
                "alpha": float(self.alpha),
                "bayesian": int(self.bayesian),
                "bayesian_tol": float(self.bayesian_tol),
                "bayesian_mode": str(self.bayesian_mode),
                "bayesian_min_iter": int(self.bayesian_min_iter),
                "param_mask": self._param_mask.copy(),
                "bayesian_log": solved["bayesian"],
            },
            "state": asdict(self.state),
            "timing": self.get_timing_summary(),
        }
        self._trace(f"finalize complete in {self._timing.finalize_seconds:.2f} s")
        return out
