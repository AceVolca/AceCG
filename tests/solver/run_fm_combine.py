#!/usr/bin/env python3
"""FM linear solver combine: sum per-window normal equations, solve, and export tables."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from AceCG.solvers.fm_matrix import FMMatrixSolver
from AceCG.utils.ffio import build_forcefield_tables, export_tables
from AceCG.utils.fm_workflow import build_interactions, load_config, load_window_universe


def _load_window_result(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(description="Combine per-window FM matrices, solve, and export tables.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--windows-root", required=True, help="Directory containing window_XXX/result.pkl")
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--blas-threads",
        type=int,
        default=int(os.environ.get("ACECG_BLAS_THREADS", "1")),
    )
    args = ap.parse_args()

    n_threads = max(int(args.blas_threads), 1)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

    cfg = load_config(args.config)
    windows_root = Path(args.windows_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    windows = list(cfg["trajectory"]["windows"])
    window_files = [windows_root / f"window_{int(w):03d}" / "result.pkl" for w in windows]
    missing = [str(p) for p in window_files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing per-window result files:\n" + "\n".join(missing))

    t0 = time.perf_counter()

    # Accumulate XtX/XtY across windows
    first_payload = _load_window_result(window_files[0])
    m0 = first_payload["result"]["matrix"]
    XtX = np.asarray(m0["XtX"], dtype=np.float64).copy()
    XtY = np.asarray(m0["XtY"], dtype=np.float64).copy()
    y_sumsq = float(m0["y_sumsq"])
    nframe = int(m0["nframe"])

    run_times = [float(first_payload["timing"]["run_seconds"])]
    total_times = [float(first_payload["timing"]["total_seconds"])]

    for file in window_files[1:]:
        payload = _load_window_result(file)
        mat = payload["result"]["matrix"]
        XtX += np.asarray(mat["XtX"], dtype=np.float64)
        XtY += np.asarray(mat["XtY"], dtype=np.float64)
        y_sumsq += float(mat["y_sumsq"])
        nframe += int(mat["nframe"])
        run_times.append(float(payload["timing"]["run_seconds"]))
        total_times.append(float(payload["timing"]["total_seconds"]))

    # Instantiate solver with combined matrices and finalize
    u = load_window_universe(cfg, window=int(windows[0]))
    interactions = build_interactions(cfg)
    fm = cfg["fm"]
    solver = FMMatrixSolver(
        universe=u,
        interactions=interactions,
        cutoff=float(fm["cutoff"]),
        exclude=fm["exclude"],
        sel=str(fm.get("selection", "all")),
        alpha=float(fm.get("alpha", 0.0)),
        bayesian=int(fm.get("bayesian", 0)),
        bayesian_tol=float(fm.get("bayesian_tol", 1.0e-6)),
        bayesian_mode=str(fm.get("bayesian_mode", "diagonal")),
        bayesian_min_iter=int(fm.get("bayesian_min_iter", 10)),
    )

    solver._XtX = XtX
    solver._XtY = XtY
    solver._y_sumsq = float(y_sumsq)
    solver._nframe = int(nframe)
    solver.state.nframe = int(nframe)
    solver.state.y_sumsq = float(y_sumsq)

    t_solve0 = time.perf_counter()
    combined = solver.finalize()
    t_solve1 = time.perf_counter()

    # Export tables
    forcefield_tables = build_forcefield_tables(cfg=cfg, interactions=solver.interactions)
    tables_dir = outdir / "tables"
    manifest = export_tables(
        cfg=cfg,
        interactions=solver.interactions,
        outdir=tables_dir,
        table_payload=forcefield_tables,
    )

    t1 = time.perf_counter()

    summary = {
        "engine": "matrix",
        "n_windows": len(windows),
        "nframe_total": int(nframe),
        "window_run_seconds": run_times,
        "window_total_seconds": total_times,
        "window_run_seconds_mean": float(np.mean(run_times)),
        "window_total_seconds_mean": float(np.mean(total_times)),
        "solve_seconds": float(t_solve1 - t_solve0),
        "combine_total_seconds": float(t1 - t0),
        "tables_dir": str(tables_dir),
        "blas_threads": int(n_threads),
        "solver_timing": combined.get("timing", {}),
    }

    combined_path = outdir / "combined.pkl"
    with combined_path.open("wb") as fh:
        pickle.dump(
            {
                "config": cfg,
                "result": combined,
                "summary": summary,
                "tables": manifest,
                "forcefield_tables": forcefield_tables,
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (outdir / "tables_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(
        f"[SOLVER_COMBINE] n_windows={len(windows)} nframe={nframe} "
        f"solve={t_solve1-t_solve0:.3f}s total={t1-t0:.3f}s"
    )
    print(f"[SOLVER_COMBINE] wrote {combined_path}")


if __name__ == "__main__":
    main()
