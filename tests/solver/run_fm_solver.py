#!/usr/bin/env python3
"""FM linear solver test: accumulate normal equations and solve via FMMatrixSolver."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from AceCG.solvers.fm_matrix import FMMatrixSolver
from AceCG.utils.fm_workflow import build_interactions, frame_slice, load_config, load_window_universe
from AceCG.utils.ffio import build_forcefield_tables


def main() -> None:
    ap = argparse.ArgumentParser(description="Run AceCG FM matrix solver for one trajectory window.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--window", required=True, type=int)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--n-parts", type=int, default=8)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--worker-blas-threads", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    u = load_window_universe(cfg, window=args.window)
    interactions = build_interactions(cfg)
    fm = cfg["fm"]

    solver = FMMatrixSolver(
        universe=u,
        interactions=interactions,
        cutoff=float(fm["cutoff"]),
        exclude=fm["exclude"],
        sel=str(fm.get("selection", "all")),
        alpha=float(fm.get("alpha", 0.0)),
        bayesian=0,
        bayesian_mode=str(fm.get("bayesian_mode", "diagonal")),
        bayesian_min_iter=int(fm.get("bayesian_min_iter", 10)),
    )

    start, end, every = frame_slice(cfg, n_total=len(u.trajectory))

    t_run0 = time.perf_counter()
    lammps_data = (cfg.get("topology", {}) or {}).get("lammps_data")
    if args.parallel:
        solver.run_parallel(
            start=start, end=end, every=every,
            n_parts=args.n_parts, n_workers=args.n_workers,
            worker_blas_threads=args.worker_blas_threads,
            trajectory=u.trajectory.filename,
            topology=str(lammps_data) if lammps_data else None,
        )
    else:
        solver.run(start=start, end=end, every=every)
    t_run1 = time.perf_counter()

    out = solver.finalize()
    forcefield_tables = build_forcefield_tables(cfg=cfg, interactions=solver.interactions)
    t1 = time.perf_counter()

    payload = {
        "window": int(args.window),
        "config": cfg,
        "result": out,
        "forcefield_tables": forcefield_tables,
        "timing": {
            "run_seconds": float(t_run1 - t_run0),
            "total_seconds": float(t1 - t0),
            "solver": out.get("timing", {}),
        },
    }

    out_file = outdir / "result.pkl"
    with out_file.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[SOLVER] window={args.window} frames={out['matrix']['nframe']} "
          f"run={t_run1-t_run0:.3f}s total={t1-t0:.3f}s")
    print(f"[SOLVER] wrote {out_file}")


if __name__ == "__main__":
    main()
