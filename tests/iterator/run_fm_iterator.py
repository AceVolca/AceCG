#!/usr/bin/env python3
"""FM iterator test: per-window force-matching via FMTrainerAnalytic.step() frame loop.

This script exercises the true iterator path end-to-end. It does NOT use
the linear solver, normal equations, or any bypass mechanism. The trainer
accumulates gradient and Hessian frame-by-frame and performs a Newton step.
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from AceCG.optimizers.newton_raphson import NewtonRaphsonOptimizer
from AceCG.trainers.fm_analytic import FMTrainerAnalytic
from AceCG.utils.bonded_projectors import interaction_offsets
from AceCG.utils.ffio import build_forcefield_tables
from AceCG.utils.fm_workflow import build_interactions, frame_slice, load_config, load_window_universe


def main() -> None:
    ap = argparse.ArgumentParser(description="Run AceCG FM iterator for one trajectory window.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--window", required=True, type=int)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--n-parts", type=int, default=8)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--worker-blas-threads", type=int, default=1)
    ap.add_argument("--iter-lr", type=float, default=1.0)
    ap.add_argument("--iter-steps", type=int, default=1, help="Number of Newton iterations per window")
    args = ap.parse_args()

    cfg = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    u = load_window_universe(cfg, window=args.window)
    interactions = build_interactions(cfg)
    fm = cfg["fm"]

    offsets = interaction_offsets(interactions)
    n_params = 0 if not offsets else offsets[-1].stop
    optimizer = NewtonRaphsonOptimizer(
        L=np.zeros(n_params, dtype=np.float64),
        mask=np.ones(n_params, dtype=bool),
        lr=float(args.iter_lr),
    )
    trainer = FMTrainerAnalytic(
        universe=u,
        interactions=interactions,
        optimizer=optimizer,
        cutoff=float(fm["cutoff"]),
        exclude=fm["exclude"],
        sel=str(fm.get("selection", "all")),
    )

    start, end, every = frame_slice(cfg, n_total=len(u.trajectory))
    nframe_used = len(range(start, end, every))
    n_steps = max(int(args.iter_steps), 1)

    t_run0 = time.perf_counter()
    lammps_data = (cfg.get("topology", {}) or {}).get("lammps_data")
    first_step = None
    last_step = None
    iter_step_total_seconds = 0.0

    for step_i in range(n_steps):
        batch = {"start": start, "end": end, "every": every, "step_index": step_i}
        if args.parallel:
            batch.update({
                "parallel": True,
                "trajectory": u.trajectory.filename,
                "topology": str(lammps_data) if lammps_data else None,
                "n_parts": int(args.n_parts),
                "n_workers": int(args.n_workers),
                "worker_blas_threads": int(args.worker_blas_threads),
            })
        last_step = trainer.step(batch, apply_update=True)
        iter_step_total_seconds += float(
            last_step.get("meta", {}).get("timing", {}).get("total_seconds", 0.0)
        )
        if first_step is None:
            first_step = last_step

    assert first_step is not None and last_step is not None
    t_run1 = time.perf_counter()

    coeff = np.asarray(trainer.get_params(), dtype=np.float64)

    # Reverse-engineer XtX/XtY from the first step's grad/hessian for parity comparison.
    # The first step starts from c=0, so:  grad = (XtX @ 0 - XtY)/N = -XtY/N
    # and hessian = XtX/N.
    h_avg = np.asarray(first_step["hessian"], dtype=np.float64)
    g_avg = np.asarray(first_step["grad"], dtype=np.float64)
    loss_avg = float(first_step["loss"])
    XtX = h_avg * float(nframe_used)
    XtY = -g_avg * float(nframe_used)  # c=0, so XtX@c=0 => grad = -XtY/N
    y_sumsq = 2.0 * loss_avg * float(nframe_used)  # loss = 0.5 * y_sumsq/N when c=0

    inter_info = []
    for it, sl in zip(trainer.interactions, trainer._offsets):
        inter_info.append({
            "style": it.style,
            "types": tuple(it.types),
            "label": it.label(),
            "n_params": it.n_params(),
            "offset": [sl.start, sl.stop],
            "metadata": dict(it.metadata),
        })

    out = {
        "format": "acecg.fm.matrix.v1",
        "coefficients": coeff,
        "interactions": inter_info,
        "matrix": {
            "XtX": XtX,
            "XtY": XtY,
            "y_sumsq": float(y_sumsq),
            "row_per_frame": int(len(u.atoms)),
            "nframe": int(nframe_used),
        },
        "solver": {
            "alpha": float(fm.get("alpha", 0.0)),
            "bayesian": 0,
            "engine": "iterative",
            "iterative_lr": float(args.iter_lr),
            "iterative_steps": int(args.iter_steps),
            "iterative_backend": "true",
            "iterative_step": last_step.get("meta", {}),
        },
        "state": {
            "n_atoms": int(len(u.atoms)),
            "n_params": int(n_params),
            "nframe": int(nframe_used),
            "y_sumsq": float(y_sumsq),
        },
        "timing": {
            "iterative_step_total_seconds": float(iter_step_total_seconds),
            "iterative_step_timing": dict(last_step.get("meta", {}).get("timing", {})),
        },
    }

    forcefield_tables = build_forcefield_tables(cfg=cfg, interactions=trainer.interactions)
    t1 = time.perf_counter()

    payload = {
        "window": int(args.window),
        "config": cfg,
        "result": out,
        "forcefield_tables": forcefield_tables,
        "timing": {
            "run_seconds": float(iter_step_total_seconds),
            "total_seconds": float(t1 - t0),
            "solver": out.get("timing", {}),
        },
    }

    out_file = outdir / "result.pkl"
    with out_file.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"[ITERATOR] window={args.window} frames={nframe_used} steps={n_steps} "
        f"run={iter_step_total_seconds:.3f}s total={t1-t0:.3f}s"
    )
    print(f"[ITERATOR] wrote {out_file}")


if __name__ == "__main__":
    main()
