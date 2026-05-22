"""Streaming task scheduler for AceCG workflows.

Provides ``TaskScheduler`` which streams simulation + post-processing tasks
across a ``LeasePool`` of CPU cores.  The controller side only launches
subprocesses and polls; all heavy computation runs in ``task_runner.py``.

Failure semantics
-----------------
- xz failure  → abort iteration immediately
- zbx failure → discard that replica, continue
- Too few zbx successes → abort iteration

With the dynamic ``LeasePool`` (per-CPU allocation), xz tasks placed first
in the queue naturally get priority.  No separate split mode is needed —
xz acquires its large CPU block first, then zbx tasks fill remaining
capacity concurrently.
"""

from __future__ import annotations

import atexit
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..io.logger import get_screen_logger
from .mpi_backend import Placement
from .resource_pool import ResourcePool, CpuLease, LeasePool, Placer, PlacementResult


_MAX_CORES_UNUSED_WARNING_EMITTED = False


def _warn_unused_max_cores_once() -> None:
    """Warn once that ``TaskSpec.max_cores`` is not yet used by the Placer."""
    global _MAX_CORES_UNUSED_WARNING_EMITTED
    if _MAX_CORES_UNUSED_WARNING_EMITTED:
        return
    _MAX_CORES_UNUSED_WARNING_EMITTED = True
    warnings.warn(
        "TaskSpec.max_cores is currently unused by the scheduler placer; "
        "allocations will not grow above preferred_cores yet.",
        RuntimeWarning,
        stacklevel=3,
    )


# ── Dataclasses ───────────────────────────────────────────────

@dataclass
class TaskSpec:
    """Specification for one simulation + post-processing task.

    Workflows set ``cpu_cores`` for fixed-width tasks.  The Placer uses
    ``min_cores`` / ``preferred_cores`` / ``max_cores`` to decide the
    actual core count; when unset they all default to ``cpu_cores``.
    """

    # "xz" | "zbx"
    task_class: str
    # conditioning frame index
    frame_id: Optional[int]
    # task working directory (absolute)
    run_dir: str
    cpu_cores: int                      # CPUs (default / fixed-width)
    # input script filename (relative to run_dir)
    sim_input: str
    sim_backend: str = "lammps"
    sim_log: str = "sim.log"
    post_spec: Optional[Dict[str, Any]] = None
    post_exec: Optional[Dict[str, Any]] = None
    sim_var: Dict[str, str] = field(default_factory=dict)
    archive_trajectory: bool = False
    trajectory_files: List[str] = field(default_factory=list)
    extra_env: Dict[str, str] = field(default_factory=dict)
    min_cores: Optional[int] = None
    preferred_cores: Optional[int] = None
    max_cores: Optional[int] = None
    single_host_only: bool = False

    def __post_init__(self) -> None:
        if self.min_cores is None:
            self.min_cores = self.cpu_cores
        if self.preferred_cores is None:
            self.preferred_cores = self.cpu_cores
        if self.max_cores is None:
            self.max_cores = self.cpu_cores
        if self.preferred_cores != self.max_cores:
            _warn_unused_max_cores_once()

    def to_spec_dict(self, cpu_cores: Optional[int] = None) -> Dict[str, Any]:
        """Serialize to dict for task_spec.json.

        *cpu_cores* overrides the stored value (set by the Placer at
        launch time when the actual allocated count may differ from the
        default).
        """
        d: Dict[str, Any] = {
            "run_dir": self.run_dir,
            "cpu_cores": cpu_cores if cpu_cores is not None else self.cpu_cores,
            "sim_backend": self.sim_backend,
            "sim_input": self.sim_input,
            "sim_log": self.sim_log,
            "post_spec": self.post_spec,
            "sim_var": self.sim_var,
            "archive_trajectory": self.archive_trajectory,
            "trajectory_files": self.trajectory_files,
            "extra_env": self.extra_env,
        }
        if self.post_exec is not None:
            d["post_exec"] = self.post_exec
        return d


@dataclass
class TaskResult:
    """Result of a completed (or failed) task."""

    task: TaskSpec
    returncode: int
    elapsed: float
    timing: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        """Return ``True`` if the task exited with return code zero."""
        return self.returncode == 0


@dataclass
class IterationResult:
    """Aggregated result for one scheduler iteration."""

    results: List[TaskResult]
    elapsed: float
    succeeded_zbx: int
    failed_zbx: int
    xz_ok: bool

    @property
    def n_total(self) -> int:
        """Return the total number of task results in this iteration."""
        return len(self.results)


class AllTasksFailedError(RuntimeError):
    """Raised when xz fails or too few zbx tasks succeed."""


# ── sim_var resolution ───────────────────────────────────────────────

def resolve_sim_var(
    sim_var: Dict[str, str],
    rng: random.Random | None = None,
) -> Dict[str, str]:
    """Resolve special placeholders in sim_var.

    ``{RANDOM}`` → random int in [10, 10_000_000].
    Called at launch time so each task gets a unique seed.
    """
    generator = rng if rng is not None else random
    return {
        k: str(generator.randint(10, 10_000_000)) if v == "{RANDOM}" else str(v)
        for k, v in sim_var.items()
    }


# ── TaskScheduler ───────────────────────────────────────────────

class TaskScheduler:
    """Stream tasks across a LeasePool with timestamped logging.

    Parameters
    ----------
    pool : ResourcePool
        Node + launcher information.
    task_timeout : float
        Per-task wall-time limit in seconds.  REQUIRED — a missing timeout
        lets a hung MPI rank or stuck srun daemon deadlock the whole
        iteration indefinitely.  Set to a generous upper bound in the
        config (e.g. 10x expected wall time) rather than disabling.
    min_success_zbx : int, optional
        Minimum number of zbx tasks that must succeed.  None = all.
    python_exe : str
        Python executable used to launch task_runner on compute node.
    rng_seed : int, optional
        Seed for per-scheduler placeholder expansion such as ``{RANDOM}``
        inside ``sim_var``.
    """

    def __init__(
        self,
        pool: ResourcePool,
        *,
        task_timeout: float,
        min_success_zbx: Optional[int] = None,
        python_exe: str = "python",
        rng_seed: Optional[int] = None,
    ):
        if task_timeout is None or task_timeout <= 0:
            raise ValueError(
                "TaskScheduler.task_timeout must be a positive float "
                "(seconds).  Unbounded timeouts let hung sim/post tasks "
                "block the entire iteration — set scheduler.task_timeout "
                "in the .acg config."
            )
        self.pool = pool
        self.task_timeout = float(task_timeout)
        self.min_success_zbx = min_success_zbx
        self.python_exe = python_exe
        self.rng = random.Random(rng_seed)

        self._t_start = time.monotonic()
        self._screen_logger = get_screen_logger("scheduler", start_time=self._t_start)
        self._active_pgids: List[int] = []
        self._pid_to_pgid: Dict[int, int] = {}
        self._pgid_lock = threading.Lock()

        atexit.register(self._cleanup)
        self._prev_sigterm = signal.signal(
            signal.SIGTERM, self._sigterm_handler
        )

    def state_dict(self) -> Dict[str, Any]:
        """Serialize scheduler state needed for deterministic resume."""
        return {"rng_state": self.rng.getstate()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore scheduler state from :meth:`state_dict`."""
        if not isinstance(state, dict) or "rng_state" not in state:
            raise ValueError("TaskScheduler state must contain 'rng_state'.")
        self.rng.setstate(state["rng_state"])

    # ── Public API ───────────────────────────────────────────────

    def run_iteration(
        self,
        xz_tasks: List[TaskSpec],
        zbx_tasks: List[TaskSpec],
        *,
        iter_dir: Optional[Path] = None,
    ) -> IterationResult:
        """Run one scheduler iteration.

        xz tasks are placed first in the queue so they acquire CPU leases
        before zbx tasks.  Remaining capacity is filled by zbx concurrently.
        """
        self._t_start = time.monotonic()
        lease_pool = self.pool.build_lease_pool()
        placer = Placer(lease_pool, backend=self.pool.backend)
        n_xz = len(xz_tasks)
        n_zbx = len(zbx_tasks)

        self._log(
            f"ITER start | xz={n_xz} zbx={n_zbx} | "
            f"cpus={lease_pool.free_total()}"
        )

        results = self._stream(
            list(xz_tasks) + list(zbx_tasks), lease_pool, placer,
        )

        # Check xz failures
        xz_results = [r for r in results if r.task.task_class == "xz"]
        xz_ok = all(r.ok for r in xz_results) if xz_results else True
        if not xz_ok:
            raise AllTasksFailedError("xz task failed — aborting iteration")

        # Check zbx success count
        zbx_results = [r for r in results if r.task.task_class == "zbx"]
        succeeded_zbx = sum(1 for r in zbx_results if r.ok)
        failed_zbx = len(zbx_results) - succeeded_zbx
        min_ok = self.min_success_zbx if self.min_success_zbx is not None else n_zbx

        if n_zbx > 0 and succeeded_zbx < min_ok:
            raise AllTasksFailedError(
                f"Only {succeeded_zbx}/{n_zbx} zbx tasks succeeded; "
                f"min_success_zbx={min_ok}"
            )

        elapsed = time.monotonic() - self._t_start
        self._log(
            f"ITER done  | {elapsed:.0f}s | "
            f"zbx ok={succeeded_zbx}/{n_zbx}"
        )

        if iter_dir is not None:
            self._write_iteration_timing(iter_dir, results, elapsed, xz_ok)

        return IterationResult(
            results=results,
            elapsed=elapsed,
            succeeded_zbx=succeeded_zbx,
            failed_zbx=failed_zbx,
            xz_ok=xz_ok,
        )

    # ── Streaming engine ───────────────────────────────────────────────

    def _stream(
        self,
        tasks: List[TaskSpec],
        pool: LeasePool,
        placer: Placer,
    ) -> List[TaskResult]:
        """Poll-based streaming loop over a LeasePool."""
        pending = list(tasks)
        active: Dict[int, tuple[subprocess.Popen, TaskSpec, PlacementResult, float]] = {}
        results: List[TaskResult] = []

        # Fail-fast: tasks that can't fit even with multi-host
        for task in pending:
            if task.single_host_only and task.min_cores > pool.max_free_on_any_host():
                raise RuntimeError(
                    f"Task {task.task_class} (fid={task.frame_id}) needs "
                    f"at least {task.min_cores} cores but largest host has "
                    f"only {pool.max_free_on_any_host()}"
                )

        def _fill():
            nonlocal pending
            still_pending: List[TaskSpec] = []
            for task in pending:
                pr = placer.place(
                    task.min_cores, task.preferred_cores, task.max_cores,
                    single_host_only=task.single_host_only,
                )
                if pr is not None:
                    proc = self._launch(task, pr)
                    active[proc.pid] = (proc, task, pr, time.monotonic())
                    hosts = ",".join(l.host for l in pr.leases)
                    self._log(
                        f"{task.task_class.upper()} launch | "
                        f"fid={task.frame_id} host={hosts} "
                        f"cores={pr.placement.n_ranks}"
                    )
                else:
                    still_pending.append(task)
            pending = still_pending

        _fill()

        while active:
            progressed = False
            for pid, (proc, task, pr, t0) in list(active.items()):
                rc = proc.poll()

                # Timeout check
                if rc is None:
                    if time.monotonic() - t0 > self.task_timeout:
                        self._log(
                            f"{task.task_class.upper()} TIMEOUT | "
                            f"fid={task.frame_id}"
                        )
                        self._kill_pid(pid)
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(os.getpgid(pid), signal.SIGKILL)
                            except (ProcessLookupError, PermissionError):
                                pass
                            try:
                                proc.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                pass
                        rc = proc.returncode if proc.returncode is not None else -1

                if rc is None:
                    continue

                elapsed = time.monotonic() - t0
                del active[pid]
                self._untrack_pgid(pid)
                for lease in pr.leases:
                    pool.release(lease)

                timing = _read_timing(task.run_dir)
                status = "done" if rc == 0 else "FAILED"
                sim_t = timing.get("sim_wall", "?") if timing else "?"
                post_t = timing.get("post_wall", "?") if timing else "?"
                hosts = ",".join(l.host for l in pr.leases)
                self._log(
                    f"{task.task_class.upper()} {status} | "
                    f"fid={task.frame_id} host={hosts} "
                    f"wall={elapsed:.0f}s sim={sim_t}s post={post_t}s"
                )

                results.append(TaskResult(
                    task=task, returncode=rc,
                    elapsed=elapsed, timing=timing,
                ))

                # xz failure → kill all, return partial results
                if rc != 0 and task.task_class == "xz":
                    self._kill_all(active)
                    for apid, (p, _, apr, _) in active.items():
                        try:
                            p.wait(timeout=5)
                        except Exception:
                            pass
                        self._untrack_pgid(apid)
                        for lease in apr.leases:
                            pool.release(lease)
                    return results

                progressed = True
                _fill()

            if not progressed:
                time.sleep(0.5)
                if pending and pool.max_free_on_any_host() > 0:
                    _fill()

        if pending:
            raise RuntimeError(
                f"{len(pending)} tasks were never scheduled: "
                + ", ".join(
                    f"{t.task_class}(fid={t.frame_id}, "
                    f"cores={t.min_cores}-{t.max_cores})"
                    for t in pending
                )
            )

        return results

    # ── Launching ───────────────────────────────────────────────

    def _launch(self, task: TaskSpec, pr: PlacementResult) -> subprocess.Popen:
        """Write spec JSON, resolve sim_var, launch task_runner."""
        run_dir = Path(task.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        placement = pr.placement
        cpu_cores = placement.n_ranks
        backend = self.pool.backend

        spec_dict = task.to_spec_dict(cpu_cores)

        # Build sim LaunchSpec and serialize it fully.
        sim_launch = backend.realize(placement, self.pool.sim_cmd, run_dir)
        spec_dict["sim_launch"] = {
            "argv": list(sim_launch.argv),
            "env_add": dict(sim_launch.env_add),
            "env_strip_prefixes": list(sim_launch.env_strip_prefixes),
        }

        # Build post LaunchSpec if post_exec requests MPI mode.
        pe = spec_dict.get("post_exec")
        if pe and pe.get("mode") == "mpi":
            n_ranks = int(pe.get("n_ranks", cpu_cores))
            # Build a post placement with the requested rank count
            if placement.single_host:
                s = placement.slices[0]
                post_placement = Placement.from_host_cores(
                    s.host, s.cpu_ids[:n_ranks], n_ranks=n_ranks,
                )
            else:
                post_placement = placement
            py_exe = self.python_exe
            post_payload = [py_exe, "-m", "AceCG.compute.mpi_engine"]
            post_launch = backend.realize(post_placement, post_payload, run_dir)
            if spec_dict.get("post_spec") is not None:
                spec_dict["post_spec"].setdefault(
                    "expected_mpi_size", int(post_placement.n_ranks),
                )
            spec_dict["post_launch"] = {
                "argv": list(post_launch.argv),
                "env_add": dict(post_launch.env_add),
                "env_strip_prefixes": list(post_launch.env_strip_prefixes),
            }

        if spec_dict.get("sim_var"):
            spec_dict["sim_var"] = resolve_sim_var(spec_dict["sim_var"], self.rng)

        spec_path = run_dir / "task_spec.json"
        spec_path.write_text(
            json.dumps(spec_dict, indent=2) + "\n",
            encoding="utf-8",
        )

        cmd = [
            self.python_exe, "-W", "ignore::RuntimeWarning:runpy",
            "-m", "AceCG.schedulers.task_runner", str(spec_path),
        ]

        # Build env: strip prefixes, add backend env, add pool extra_env.
        env = {
            k: v for k, v in os.environ.items()
            if not any(k.startswith(p) for p in sim_launch.env_strip_prefixes)
        }
        env.update(sim_launch.env_add)
        env.update(self.pool.extra_env)

        proc = subprocess.Popen(
            cmd, env=env,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        with self._pgid_lock:
            try:
                pgid = os.getpgid(proc.pid)
                self._active_pgids.append(pgid)
                self._pid_to_pgid[proc.pid] = pgid
            except (ProcessLookupError, PermissionError):
                pass
        return proc

    # ── Process management ───────────────────────────────────────────────

    def _kill_pid(self, pid: int) -> None:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    def _untrack_pgid(self, pid: int) -> None:
        with self._pgid_lock:
            pgid = self._pid_to_pgid.pop(pid, None)
            if pgid is not None and pgid in self._active_pgids:
                self._active_pgids.remove(pgid)

    def _kill_all(self, active: Dict[int, Any]) -> None:
        for pid in list(active.keys()):
            self._kill_pid(pid)
        time.sleep(3)
        for pid in list(active.keys()):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def _cleanup(self) -> None:
        with self._pgid_lock:
            pgids = list(self._active_pgids)
        for pgid in pgids:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    def _sigterm_handler(self, signum: int, frame: Any) -> None:
        self._cleanup()
        prev = self._prev_sigterm
        if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
            prev(signum, frame)
        sys.exit(1)

    # ── Logging + timing persistence ───────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        elapsed = time.monotonic() - self._t_start
        self._screen_logger.info(msg, elapsed=elapsed)

    def _write_iteration_timing(
        self,
        iter_dir: Path,
        results: List[TaskResult],
        elapsed: float,
        xz_ok: bool,
    ) -> None:
        iter_dir.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "elapsed": elapsed,
            "xz_ok": xz_ok,
            "tasks": [
                {
                    "task_class": r.task.task_class,
                    "frame_id": r.task.frame_id,
                    "returncode": r.returncode,
                    "elapsed": r.elapsed,
                    "sim_loop": (r.timing or {}).get("sim_loop"),
                    "sim_wall": (r.timing or {}).get("sim_wall"),
                    "post_wall": (r.timing or {}).get("post_wall"),
                }
                for r in results
            ],
        }
        (iter_dir / "timing.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


# ── Helpers ───────────────────────────────────────────────

def _read_timing(run_dir: str) -> Optional[Dict[str, Any]]:
    path = Path(run_dir) / "task_timing.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None
