"""Compute-node-local task runner.

Each task runner:

1. Reads a JSON task_spec from disk
2. Runs the simulation via subprocess (backend-dispatched)
3. Calls the post-processing function (via ``run_post(post_spec)``)
4. Writes task_timing.json
5. Optionally deletes the trajectory

Usage (from launcher backend)::

    python -m AceCG.schedulers.task_runner /path/to/task_spec.json

task_spec.json keys
-------------------
run_dir : str
    Task working directory (all paths resolved relative to this).
cpu_cores : int
    Number of CPU cores allocated to this task.
sim_backend : str
    Simulation engine identifier ("lammps"; others raise NotImplementedError).
sim_cmd : list of str
    Full mpirun + engine binary (without input/log/var args).
sim_input : str
    Input script filename (relative to run_dir).
sim_log : str
    Log filename (relative to run_dir).
sim_var : dict
    Engine variables.  For LAMMPS: appended as ``-var key value``.
post_spec : dict or null
    Dict passed directly to ``MPIComputeEngine.run_post()``.
    This is the **sole** post-processing payload.
archive_trajectory : bool
    If false, delete trajectory_files after post succeeds.
trajectory_files : list of str
    Paths to delete when archive_trajectory=false.
extra_env : dict
    Additional environment variables.
post_exec : dict, optional
    If present and ``mode="mpi"``, post-processing runs via a
    pre-built ``post_launch`` LaunchSpec.  Contains only launch metadata;
    the compute payload comes from ``post_spec``.
    Keys: mode ("inproc"|"mpi"), n_ranks.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..io.logger import get_screen_logger


SCREEN_LOGGER = get_screen_logger("task_runner")

warnings.filterwarnings("ignore", message="Reader has no dt information")
warnings.filterwarnings("ignore", message="Guessed all Masses")


# ---------------------------------------------------------------------------
# Engine dispatch
# ---------------------------------------------------------------------------

def _build_engine_args(
    sim_backend: str,
    sim_input: str,
    sim_log: str,
    sim_var: dict[str, str],
) -> list[str]:
    """Convert generic sim_* fields into engine-specific CLI arguments.

    For LAMMPS: ``[-in, input, -log, log, -var, k, v, ...]``
    """
    if sim_backend == "lammps":
        args = ["-in", sim_input, "-log", sim_log]
        for key, value in sim_var.items():
            args.extend(["-var", str(key), str(value)])
        return args
    raise NotImplementedError(
        f"Simulation backend '{sim_backend}' is not supported. "
        f"Supported: lammps"
    )


def _merge_engine_args(argv: list[str], engine_args: list[str]) -> list[str]:
    """Append *engine_args* to each MPMD segment in *argv*.

    Intel Hydra MPMD argv uses ``":"`` tokens to separate segments.
    If no ``":"`` is present, fall back to simple concatenation.
    """
    if ":" not in argv:
        return argv + engine_args
    # Split argv into segments delimited by ":"
    segments: list[list[str]] = [[]]
    for tok in argv:
        if tok == ":":
            segments.append([])
        else:
            segments[-1].append(tok)
    # Append engine_args to each segment, rejoin with ":"
    out: list[str] = []
    for i, seg in enumerate(segments):
        if i > 0:
            out.append(":")
        out.extend(seg + engine_args)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(spec_path: str) -> None:
    """Execute one task defined by *spec_path*."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    run_dir = Path(spec["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    timing: Dict[str, Any] = {}
    t_total_start = time.monotonic()

    # ------------------------------------------------------------------ #
    # Stage 1 — Simulation                                                #
    # ------------------------------------------------------------------ #
    sim_backend: str = spec.get("sim_backend", "lammps")
    sim_input: str = spec["sim_input"]
    sim_log: str = spec.get("sim_log", "sim.log")
    sim_var: Dict[str, str] = spec.get("sim_var", {})

    engine_args = _build_engine_args(sim_backend, sim_input, sim_log, sim_var)

    # New path: full LaunchSpec in sim_launch.
    # Legacy path: flat sim_cmd list (backward compat).
    task_extra_env: Dict[str, str] = spec.get("extra_env", {}) or {}
    sim_launch = spec.get("sim_launch")
    if sim_launch is not None:
        full_cmd = _merge_engine_args(list(sim_launch["argv"]), engine_args)
        env = _make_launch_env(sim_launch, task_extra_env=task_extra_env)
    else:
        full_cmd = list(spec["sim_cmd"]) + engine_args
        env = _make_task_env(spec)

    t0 = time.monotonic()
    stdout_path = run_dir / "sim_stdout.log"
    with open(stdout_path, "w") as fh:
        result = subprocess.run(
            full_cmd, cwd=str(run_dir), env=env,
            stdout=fh, stderr=subprocess.STDOUT,
        )
    timing["sim_wall"] = time.monotonic() - t0

    if sim_backend == "lammps":
        timing["sim_loop"] = parse_loop_time(run_dir / sim_log)

    if result.returncode != 0:
        timing["status"] = "sim_failed"
        timing["sim_returncode"] = result.returncode
        timing["total_wall"] = time.monotonic() - t_total_start
        write_timing(run_dir, timing)
        sys.exit(result.returncode)

    # ------------------------------------------------------------------ #
    # Stage 2 — Post-processing                                           #
    # ------------------------------------------------------------------ #
    post_spec: Optional[Dict[str, Any]] = spec.get("post_spec")
    post_launch = spec.get("post_launch")
    post_exec: Optional[Dict[str, Any]] = spec.get("post_exec")
    t1 = time.monotonic()

    try:
        if post_spec is not None:
            if post_launch is not None:
                _run_post_via_launch(
                    run_dir, post_spec, post_launch, timing,
                    task_extra_env=task_extra_env,
                )
            elif post_exec and post_exec.get("mode") == "mpi":
                raise RuntimeError(
                    "post_exec requests MPI mode but no post_launch found in "
                    "task_spec.json.  Regenerate the spec via the scheduler."
                )
            else:
                from AceCG.compute.registry import build_default_engine

                build_default_engine().run_post(post_spec)
    except Exception as exc:
        timing["post_wall"] = time.monotonic() - t1
        timing["status"] = "post_failed"
        timing["post_error"] = f"{type(exc).__name__}: {exc}"
        timing["total_wall"] = time.monotonic() - t_total_start
        write_timing(run_dir, timing)
        raise

    timing["post_wall"] = time.monotonic() - t1

    # ------------------------------------------------------------------ #
    # Stage 3 — Cleanup                                                   #
    # ------------------------------------------------------------------ #
    if not spec.get("archive_trajectory", False):
        run_dir_resolved = run_dir.resolve()
        for traj in spec.get("trajectory_files", []):
            p = (run_dir / traj).resolve()
            if not str(p).startswith(str(run_dir_resolved) + os.sep) and p != run_dir_resolved:
                warnings.warn(
                    f"Trajectory path {traj!r} resolves outside run_dir, skipping",
                    RuntimeWarning,
                )
                continue
            if p.exists():
                p.unlink()

    timing["status"] = "ok"
    timing["total_wall"] = time.monotonic() - t_total_start
    write_timing(run_dir, timing)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_launch_env(
    launch: Dict[str, Any],
    *,
    task_extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build env from a serialized LaunchSpec dict.

    Precedence (last wins): os.environ (minus strip prefixes) → launch.env_add
    → task_extra_env.  Task-level ``extra_env`` must override backend defaults
    because it carries user intent (e.g. ``OMP_NUM_THREADS`` per task).
    """
    strip = tuple(launch.get("env_strip_prefixes", ("PMI_",)))
    env = {
        k: v for k, v in os.environ.items()
        if not any(k.startswith(p) for p in strip)
    }
    env.update(launch.get("env_add", {}))
    if task_extra_env:
        for key, value in task_extra_env.items():
            env[str(key)] = str(value)
    return env


def _run_post_via_launch(
    run_dir: Path,
    post_spec: Dict[str, Any],
    post_launch: Dict[str, Any],
    timing: Dict[str, Any],
    *,
    task_extra_env: Optional[Dict[str, str]] = None,
) -> None:
    """Run MPI post-processing using a serialized post LaunchSpec."""
    spec_path = run_dir / "mpi_post_spec.json"
    spec_path.write_text(
        json.dumps(post_spec, indent=2) + "\n",
        encoding="utf-8",
    )
    argv = _merge_engine_args(list(post_launch["argv"]), [str(spec_path)])
    env = _make_launch_env(post_launch, task_extra_env=task_extra_env)

    stdout_path = run_dir / "mpi_post_stdout.log"
    with open(stdout_path, "w") as fh:
        result = subprocess.run(
            argv, cwd=str(run_dir),
            env=env,
            stdout=fh, stderr=subprocess.STDOUT,
        )
    timing["mpi_post_returncode"] = result.returncode
    if result.returncode != 0:
        raise RuntimeError(
            f"MPI post-processing failed (rc={result.returncode}). "
            f"See {stdout_path}"
        )


def _make_task_env(spec: Dict[str, Any]) -> Dict[str, str]:
    """Build environment for simulation subprocess.

    Strips only ``PMI_*`` (any leftover from an outer MPI step); preserves
    ``SLURM_*`` so Hydra's slurm bootstrap can see the allocation.
    """
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("PMI_")
    }
    for key, value in spec.get("extra_env", {}).items():
        env[str(key)] = str(value)
    return env


def run_post(
    post_spec: Dict[str, Any],
    resource_pool: Any,
    *,
    run_dir: Optional[Path] = None,
    python_exe: Optional[str] = None,
    extra_launcher_args: Optional[List[str]] = None,
) -> None:
    """Run MPI post-processing using a discovered ``ResourcePool``.

    Builds the launcher command from the pool's backend and host inventory,
    then launches via a serialized ``LaunchSpec``.
    """
    from .mpi_backend import Placement, HostSlice

    if run_dir is None:
        run_dir = Path(post_spec["work_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    n_ranks = sum(h.n_cpus for h in resource_pool.hosts)
    slices = tuple(
        HostSlice(host=h.hostname, cpu_ids=h.cpu_ids,
                  host_n_cpus=h.n_cpus)
        for h in resource_pool.hosts
    )
    placement = Placement(slices=slices, n_ranks=n_ranks)

    backend = resource_pool.backend
    py = python_exe or sys.executable
    payload = [py, "-m", "AceCG.compute.mpi_engine"]
    if extra_launcher_args:
        payload = list(extra_launcher_args) + payload
    launch = backend.realize(placement, payload, run_dir)

    post_launch = {
        "argv": list(launch.argv),
        "env_add": dict(launch.env_add),
        "env_strip_prefixes": list(launch.env_strip_prefixes),
    }
    timing: Dict[str, Any] = {}
    _run_post_via_launch(run_dir, post_spec, post_launch, timing)


def parse_loop_time(log_path: Path) -> Optional[float]:
    """Extract the LAST ``Loop time`` value from a LAMMPS log file."""
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"Loop time of\s+([\d.eE+\-]+)", text)
        if matches:
            return float(matches[-1])
    except Exception:
        pass
    return None


def write_timing(run_dir: Path, timing: Dict[str, Any]) -> None:
    """Write task_timing.json to the task's run_dir."""
    path = run_dir / "task_timing.json"
    path.write_text(
        json.dumps(timing, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# __main__ entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        SCREEN_LOGGER.error("Usage: python -m AceCG.schedulers.task_runner <spec_path>")
        sys.exit(1)
    run(sys.argv[1])
