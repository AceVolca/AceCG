"""Unit tests for the current TaskScheduler streaming path."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pytest

from AceCG.schedulers.mpi_backend import LaunchSpec, Placement
from AceCG.schedulers.resource_pool import HostInventory, PlacementResult, ResourcePool
from AceCG.schedulers.task_scheduler import AllTasksFailedError, TaskScheduler, TaskSpec


class _DirectBackend:
    """Test-only backend that runs the payload directly (no mpirun)."""

    name = "direct-test"
    supports_multi_host = False

    def realize(self, placement: Placement, payload_cmd: list[str], run_dir: Path) -> LaunchSpec:
        return LaunchSpec(
            argv=tuple(payload_cmd),
            env_add={},
            env_strip_prefixes=(),
        )


def _make_mock_sim_script(tmp_path: Path, *, exit_code: int = 0, sleep: float = 0.0) -> str:
    script = tmp_path / f"mock_sim_{exit_code}_{int(sleep * 1000)}.sh"
    loop_time = max(sleep * 0.9, 0.01)
    script.write_text(
        "#!/bin/bash\n"
        "LOG=$4\n"
        f"sleep {sleep:.3f}\n"
        f"echo 'Loop time of {loop_time:.3f} on 1 procs for 10 steps' >> \"$LOG\"\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return str(script)


def _make_pool(script_path: str, *, cpus: int = 8) -> ResourcePool:
    return ResourcePool(
        hosts=[HostInventory("localhost", tuple(range(cpus)))],
        sim_cmd=["bash", script_path],
        backend=_DirectBackend(),
    )


def _make_task(tmp_path: Path, *, task_class: str, frame_id: int | None, cpu_cores: int = 2) -> TaskSpec:
    run_dir = tmp_path / f"{task_class}_{frame_id if frame_id is not None else 'none'}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "test.in").write_text("# dummy\n", encoding="utf-8")
    return TaskSpec(
        task_class=task_class,
        frame_id=frame_id,
        run_dir=str(run_dir),
        cpu_cores=cpu_cores,
        sim_input="test.in",
        archive_trajectory=True,
    )


def test_streaming_iteration_runs_all_tasks(tmp_path):
    script = _make_mock_sim_script(tmp_path, exit_code=0, sleep=0.02)
    sched = TaskScheduler(
        _make_pool(script, cpus=8),
        task_timeout=30.0,
        python_exe=sys.executable,
    )

    xz_tasks = [_make_task(tmp_path, task_class="xz", frame_id=None, cpu_cores=4)]
    zbx_tasks = [
        _make_task(tmp_path, task_class="zbx", frame_id=0, cpu_cores=2),
        _make_task(tmp_path, task_class="zbx", frame_id=1, cpu_cores=2),
    ]
    result = sched.run_iteration(xz_tasks, zbx_tasks, iter_dir=tmp_path / "iter")

    assert result.xz_ok is True
    assert result.succeeded_zbx == 2
    assert result.failed_zbx == 0
    timing = json.loads((tmp_path / "iter" / "timing.json").read_text(encoding="utf-8"))
    assert timing["xz_ok"] is True
    assert len(timing["tasks"]) == 3


def test_streaming_xz_failure_aborts_iteration(tmp_path):
    fail_script = _make_mock_sim_script(tmp_path, exit_code=1)
    sched = TaskScheduler(
        _make_pool(fail_script, cpus=8),
        task_timeout=30.0,
        python_exe=sys.executable,
    )

    xz_tasks = [_make_task(tmp_path, task_class="xz", frame_id=None, cpu_cores=4)]
    zbx_tasks = [_make_task(tmp_path, task_class="zbx", frame_id=0, cpu_cores=2)]

    with pytest.raises(AllTasksFailedError):
        sched.run_iteration(xz_tasks, zbx_tasks)


def test_optional_validation_xz_failure_is_recorded_without_abort(tmp_path):
    ok_script = _make_mock_sim_script(tmp_path, exit_code=0)
    fail_script = _make_mock_sim_script(tmp_path, exit_code=1)
    sched = TaskScheduler(
        _make_pool(ok_script, cpus=8),
        task_timeout=30.0,
        python_exe=sys.executable,
    )

    validation_task = _make_task(
        tmp_path,
        task_class="xz",
        frame_id=None,
        cpu_cores=2,
    )
    validation_task.role = "validation"
    validation_task.required = False
    validation_task.sim_cmd = ["bash", fail_script]
    validation_task.metadata = {"validation_label": "epoch_0000"}
    zbx_tasks = [_make_task(tmp_path, task_class="zbx", frame_id=0, cpu_cores=2)]

    result = sched.run_iteration(
        [validation_task],
        zbx_tasks,
        iter_dir=tmp_path / "iter_validation",
    )

    assert result.xz_ok is True
    assert result.succeeded_zbx == 1
    assert result.failed_validation == 1
    timing = json.loads(
        (tmp_path / "iter_validation" / "timing.json").read_text(encoding="utf-8")
    )
    validation_rows = [
        row for row in timing["tasks"] if row["role"] == "validation"
    ]
    assert validation_rows[0]["required"] is False
    assert validation_rows[0]["returncode"] == 1


def test_streaming_min_success_zbx_is_enforced(tmp_path):
    ok_script = _make_mock_sim_script(tmp_path, exit_code=0)
    fail_script = _make_mock_sim_script(tmp_path, exit_code=1)

    pool = _make_pool(ok_script, cpus=8)
    sched = TaskScheduler(
        pool,
        task_timeout=30.0,
        min_success_zbx=2,
        python_exe=sys.executable,
    )

    xz_tasks = [_make_task(tmp_path, task_class="xz", frame_id=None, cpu_cores=4)]
    zbx_ok = _make_task(tmp_path, task_class="zbx", frame_id=0, cpu_cores=2)
    zbx_fail = _make_task(tmp_path, task_class="zbx", frame_id=1, cpu_cores=2)
    zbx_tasks = [zbx_ok, zbx_fail]

    # Patch _launch to swap sim_cmd per task (frame_id=1 → fail script).
    import AceCG.schedulers.task_scheduler as scheduler_module

    original_launch = scheduler_module.TaskScheduler._launch

    def _launch(self, task, lease):
        old_cmd = self.pool.sim_cmd
        try:
            self.pool.sim_cmd = ["bash", fail_script] if task.frame_id == 1 else ["bash", ok_script]
            return original_launch(self, task, lease)
        finally:
            self.pool.sim_cmd = old_cmd

    scheduler_module.TaskScheduler._launch = _launch
    try:
        with pytest.raises(AllTasksFailedError):
            sched.run_iteration(xz_tasks, zbx_tasks)
    finally:
        scheduler_module.TaskScheduler._launch = original_launch


def test_streaming_iteration_timing_records_task_fields(tmp_path):
    script = _make_mock_sim_script(tmp_path, exit_code=0, sleep=0.01)
    sched = TaskScheduler(
        _make_pool(script, cpus=4),
        task_timeout=30.0,
        python_exe=sys.executable,
    )

    result = sched.run_iteration(
        [_make_task(tmp_path, task_class="xz", frame_id=None, cpu_cores=2)],
        [],
        iter_dir=tmp_path / "iter0",
    )
    assert result.n_total == 1
    timing = json.loads((tmp_path / "iter0" / "timing.json").read_text(encoding="utf-8"))
    task = timing["tasks"][0]
    assert task["task_class"] == "xz"
    assert task["frame_id"] is None
    assert task["sim_wall"] is not None


def test_launch_injects_expected_mpi_size_for_mpi_post(monkeypatch, tmp_path):
    script = _make_mock_sim_script(tmp_path, exit_code=0)
    sched = TaskScheduler(
        _make_pool(script, cpus=4),
        task_timeout=30.0,
        python_exe=sys.executable,
    )
    task = _make_task(tmp_path, task_class="xz", frame_id=None, cpu_cores=4)
    task.post_exec = {"mode": "mpi"}
    task.post_spec = {"work_dir": str(Path(task.run_dir)), "steps": []}
    placement = Placement.from_host_cores("localhost", (0, 1, 2, 3), n_ranks=4)

    class _FakeProc:
        pid = 123456

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _FakeProc())
    monkeypatch.setattr("os.getpgid", lambda pid: pid)

    sched._launch(task, PlacementResult(placement, []))

    spec = json.loads((Path(task.run_dir) / "task_spec.json").read_text(encoding="utf-8"))
    assert spec["post_spec"]["expected_mpi_size"] == 4


def test_taskspec_warns_once_when_max_cores_differs(monkeypatch, tmp_path):
    import AceCG.schedulers.task_scheduler as scheduler_module

    monkeypatch.setattr(
        scheduler_module,
        "_MAX_CORES_UNUSED_WARNING_EMITTED",
        False,
    )

    run_dir = tmp_path / "warn_once"
    run_dir.mkdir()
    (run_dir / "test.in").write_text("# dummy\n", encoding="utf-8")

    with pytest.warns(
        RuntimeWarning,
        match=r"TaskSpec\.max_cores is currently unused",
    ):
        scheduler_module.TaskSpec(
            task_class="xz",
            frame_id=None,
            run_dir=str(run_dir),
            cpu_cores=4,
            min_cores=2,
            preferred_cores=4,
            max_cores=8,
            sim_input="test.in",
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scheduler_module.TaskSpec(
            task_class="zbx",
            frame_id=0,
            run_dir=str(run_dir),
            cpu_cores=4,
            min_cores=2,
            preferred_cores=4,
            max_cores=8,
            sim_input="test.in",
        )
    assert not caught


def test_scheduler_sim_var_rng_state_roundtrips(tmp_path):
    script = _make_mock_sim_script(tmp_path, exit_code=0, sleep=0.01)

    def _make_seed_task(tag: str) -> TaskSpec:
        run_dir = tmp_path / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "test.in").write_text("# dummy\n", encoding="utf-8")
        return TaskSpec(
            task_class="xz",
            frame_id=None,
            run_dir=str(run_dir),
            cpu_cores=2,
            sim_input="test.in",
            sim_var={"SEED": "{RANDOM}"},
            archive_trajectory=True,
        )

    sched_1 = TaskScheduler(
        _make_pool(script, cpus=4),
        task_timeout=30.0,
        python_exe=sys.executable,
        rng_seed=123,
    )
    sched_1.run_iteration([_make_seed_task("run_a")], [], iter_dir=tmp_path / "iter_a")
    first_spec = json.loads((tmp_path / "run_a" / "task_spec.json").read_text(encoding="utf-8"))
    saved_state = sched_1.state_dict()

    sched_1.run_iteration([_make_seed_task("run_b")], [], iter_dir=tmp_path / "iter_b")
    second_spec = json.loads((tmp_path / "run_b" / "task_spec.json").read_text(encoding="utf-8"))

    sched_2 = TaskScheduler(
        _make_pool(script, cpus=4),
        task_timeout=30.0,
        python_exe=sys.executable,
        rng_seed=999,
    )
    sched_2.load_state_dict(saved_state)
    sched_2.run_iteration([_make_seed_task("run_c")], [], iter_dir=tmp_path / "iter_c")
    resumed_spec = json.loads((tmp_path / "run_c" / "task_spec.json").read_text(encoding="utf-8"))

    assert first_spec["sim_var"]["SEED"] != "{RANDOM}"
    assert second_spec["sim_var"]["SEED"] == resumed_spec["sim_var"]["SEED"]
