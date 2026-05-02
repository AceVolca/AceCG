"""Preflight benchmark helpers for scheduler task sizing.

This module intentionally no longer defines ``RuntimeProfiler``.
It only keeps the benchmark utilities that probe candidate MPI rank
counts and estimate production-time task throughput.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .mpi_backend import detect_mpi_family, LocalMpirunBackend, Placement
from .task_runner import parse_loop_time


_BENCH_STEPS = 1000


def _shorten_lammps_input(script: str, bench_steps: int = _BENCH_STEPS) -> str:
    """Rewrite a LAMMPS input script for benchmarking.

    Replaces ``run N`` with ``run <bench_steps>`` and strips dump/restart
    commands so the trial measures simulation throughput with minimal I/O.
    """
    lines: List[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if re.match(r"^(dump|dump_modify|write_dump|write_restart|undump)\b", stripped):
            continue
        if re.match(r"^run\s+\d+", stripped):
            lines.append(f"run {bench_steps}")
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def _parse_total_steps(log_path: Path) -> Optional[int]:
    """Extract total steps from the last ``Loop time`` line in a LAMMPS log."""
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    matches = re.findall(
        r"Loop time of\s+[\d.eE+\-]+\s+on\s+\d+\s+procs?\s+for\s+(\d+)\s+steps",
        text,
    )
    return int(matches[-1]) if matches else None


def _compute_rank_candidates(cpn: int, min_ranks: int = 4) -> List[int]:
    """Return descending MPI rank counts that evenly divide ``cpn``."""
    return sorted(
        [r for r in range(min_ranks, cpn + 1) if cpn % r == 0],
        reverse=True,
    )


def preflight_benchmark(
    *,
    sim_input: str,
    run_dir: str,
    sim_cmd: List[str],
    mpirun_path: str,
    cpus_per_node: int,
    bench_steps: int = _BENCH_STEPS,
    candidate_divisors: Optional[Tuple[int, ...]] = None,
    timeout: float = 120.0,
    production_steps: int = 50000,
    n_tasks: int = 0,
    n_nodes: int = 0,
) -> Dict[str, Any]:
    """Run short benchmark trials and choose a rank count.

    Selection policy:
    - If ``n_tasks > 0`` and ``n_nodes > 0``: minimize estimated iteration
      makespan using ``slots_per_node = cpus_per_node // ranks``.
    - Otherwise: maximize throughput-per-core.
    """
    input_path = Path(sim_input)
    run_dir_path = Path(run_dir)
    original_script = input_path.read_text(encoding="utf-8")
    bench_script = _shorten_lammps_input(original_script, bench_steps)

    if candidate_divisors is not None:
        candidates: List[int] = []
        seen: set[int] = set()
        for divisor in candidate_divisors:
            ranks = max(1, cpus_per_node // int(divisor))
            if ranks not in seen:
                candidates.append(ranks)
                seen.add(ranks)
    else:
        candidates = _compute_rank_candidates(cpus_per_node)
    if not candidates:
        candidates = [max(1, cpus_per_node)]

    mpi_family = detect_mpi_family(mpirun_path)
    backend = LocalMpirunBackend(mpirun_path, mpi_family=mpi_family)
    bench_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(("PMI_", "SLURM_"))
    }

    trials: List[Dict[str, Any]] = []

    for nranks in candidates:
        bench_dir = run_dir_path / f"_preflight_{nranks}"
        if bench_dir.exists():
            shutil.rmtree(bench_dir)
        shutil.copytree(
            str(run_dir_path),
            str(bench_dir),
            ignore=shutil.ignore_patterns("_preflight*"),
        )
        (bench_dir / "bench.in").write_text(bench_script, encoding="utf-8")

        placement = Placement.from_host_cores(
            "localhost", tuple(range(nranks)), n_ranks=nranks,
        )
        launch = backend.realize(
            placement,
            [*sim_cmd, "-in", "bench.in", "-log", "bench.log"],
            bench_dir,
        )
        cmd = list(launch.argv)

        trial: Dict[str, Any] = {"ranks": nranks}
        try:
            t0 = time.monotonic()
            result = subprocess.run(
                cmd,
                cwd=str(bench_dir),
                timeout=timeout,
                capture_output=True,
                env=bench_env,
            )
            trial["wall_time"] = time.monotonic() - t0
            trial["returncode"] = result.returncode

            if result.returncode != 0:
                trial["error"] = "benchmark command failed"
                trial["stderr"] = result.stderr.decode("utf-8", errors="replace")[-500:]
                trials.append(trial)
                continue

            loop_t = parse_loop_time(bench_dir / "bench.log")
            actual_steps = _parse_total_steps(bench_dir / "bench.log")
            if loop_t is None or loop_t <= 0:
                trial["error"] = "no Loop time in log"
                trials.append(trial)
                continue

            steps = actual_steps if actual_steps else bench_steps
            sps = steps / loop_t
            est_task_time = float("inf") if sps <= 0 else production_steps / sps
            slots_per_node = max(1, cpus_per_node // nranks)
            total_slots = slots_per_node * max(1, n_nodes) if n_nodes > 0 else slots_per_node

            trial.update(
                loop_time=loop_t,
                steps=steps,
                steps_per_sec=sps,
                throughput_per_core=sps / nranks,
                estimated_task_time=est_task_time,
                slots_per_node=slots_per_node,
                total_slots=total_slots,
            )
            if n_tasks > 0 and n_nodes > 0:
                rounds = math.ceil(n_tasks / total_slots)
                trial["iter_makespan"] = rounds * est_task_time

        except subprocess.TimeoutExpired:
            trial["error"] = f"timeout after {timeout:.1f}s"
        except Exception as exc:
            trial["error"] = f"{type(exc).__name__}: {exc}"

        trials.append(trial)

    successful = [t for t in trials if "estimated_task_time" in t]
    if not successful:
        fallback_ranks = candidates[0]
        fallback_slots = max(1, cpus_per_node // fallback_ranks)
        return {
            "best_ranks": fallback_ranks,
            "best_cpu_cores": fallback_ranks,
            "best_slots_per_node": fallback_slots,
            "estimated_task_time": float("inf"),
            "trials": trials,
        }

    if n_tasks > 0 and n_nodes > 0:
        best = min(
            successful,
            key=lambda t: (
                t["iter_makespan"],
                t["estimated_task_time"],
                -t["throughput_per_core"],
            ),
        )
    else:
        best = max(
            successful,
            key=lambda t: (
                t["throughput_per_core"],
                -t["estimated_task_time"],
                t["ranks"],
            ),
        )

    return {
        "best_ranks": int(best["ranks"]),
        "best_cpu_cores": int(best["ranks"]),
        "best_slots_per_node": int(best["slots_per_node"]),
        "estimated_task_time": float(best["estimated_task_time"]),
        "trials": trials,
    }


__all__ = [
    "_BENCH_STEPS",
    "_shorten_lammps_input",
    "_parse_total_steps",
    "_compute_rank_candidates",
    "preflight_benchmark",
]
