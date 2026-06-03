"""Unit tests for task_runner.py with the current TaskSpec/run contract."""

import json
from pathlib import Path

import pytest
from AceCG.schedulers.task_runner import parse_loop_time, write_timing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_fake_lammps_log(path: Path, loop_time: float = 123.4) -> None:
    path.write_text(
        f"LAMMPS (22 Jul 2025)\n"
        f"Loop time of {loop_time} on 8 procs for 1000 steps\n"
        f"Performance: ... ns/day\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# parse_loop_time
# ---------------------------------------------------------------------------

def test_parse_loop_time(tmp_path):
    log = tmp_path / "lammps.log"
    _write_fake_lammps_log(log, loop_time=456.789)
    t = parse_loop_time(log)
    assert t == pytest.approx(456.789)


def test_parse_loop_time_missing(tmp_path):
    assert parse_loop_time(tmp_path / "nonexistent.log") is None


def test_parse_loop_time_no_match(tmp_path):
    log = tmp_path / "lammps.log"
    log.write_text("no loop time here", encoding="utf-8")
    assert parse_loop_time(log) is None


def test_parse_loop_time_multiple_runs(tmp_path):
    """LAMMPS logs with minimize + equilibration + sampling runs
    should return the LAST loop time (production sampling)."""
    log = tmp_path / "lammps.log"
    log.write_text(
        "LAMMPS (22 Jul 2025)\n"
        "Loop time of 0.437 on 8 procs for 23 steps with 8064 atoms\n"
        "Minimization stats:\n"
        "  Stopping criterion = ...\n"
        "Loop time of 16.18 on 8 procs for 5000 steps with 8064 atoms\n"
        "Loop time of 16.16 on 8 procs for 5000 steps with 8064 atoms\n"
        "Loop time of 16.36 on 8 procs for 5000 steps with 8064 atoms\n",
        encoding="utf-8",
    )
    t = parse_loop_time(log)
    assert t == pytest.approx(16.36)


# ---------------------------------------------------------------------------
# write_timing
# ---------------------------------------------------------------------------

def test_write_timing(tmp_path):
    timing = {"lammps_wall": 10.0, "post_wall": 2.0, "status": "ok"}
    write_timing(tmp_path, timing)
    data = json.loads((tmp_path / "task_timing.json").read_text())
    assert data["status"] == "ok"
    assert data["lammps_wall"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Full task_runner.run with mock LAMMPS
# ---------------------------------------------------------------------------

def _make_mock_lammps_script(tmp_path: Path, loop_time: float = 50.0) -> Path:
    """Write a shell script that mimics the current LAMMPS CLI."""
    script = tmp_path / "mock_lmp.sh"
    script.write_text(
        "#!/bin/bash\n"
        "LOG=$4\n"
        f"echo 'Loop time of {loop_time} on 1 procs for 10 steps' >> \"$LOG\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _make_task_spec(run_dir: Path, sim_cmd: list) -> dict:
    sim_input = "test.in"
    (run_dir / sim_input).write_text("# dummy\n", encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "cpu_cores": 1,
        "sim_backend": "lammps",
        "sim_launch": {
            "argv": sim_cmd,
            "env_add": {},
            "env_strip_prefixes": [],
        },
        "sim_input": sim_input,
        "sim_log": "sim.log",
        "post_spec": None,
        "archive_trajectory": True,
        "trajectory_files": [],
        "extra_env": {},
    }


def test_task_runner_happy_path(tmp_path):
    """Mock LAMMPS that succeeds; verify timing.json written with status=ok."""
    mock_script = _make_mock_lammps_script(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = _make_task_spec(run_dir, ["bash", str(mock_script)])
    spec_path = tmp_path / "task_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    from AceCG.schedulers.task_runner import run as tr_run
    tr_run(str(spec_path))

    timing = json.loads((run_dir / "task_timing.json").read_text())
    assert timing["status"] == "ok"
    assert timing["sim_wall"] > 0
    assert timing["sim_loop"] == pytest.approx(50.0)


def test_task_runner_lammps_fails(tmp_path):
    """Mock LAMMPS that exits with code 1; verify status=sim_failed."""
    fail_script = tmp_path / "fail_lmp.sh"
    fail_script.write_text("#!/bin/bash\nexit 1\n")
    fail_script.chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = _make_task_spec(run_dir, ["bash", str(fail_script)])
    spec_path = tmp_path / "task_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    from AceCG.schedulers.task_runner import run as tr_run
    with pytest.raises(SystemExit) as exc:
        tr_run(str(spec_path))
    assert exc.value.code == 1

    timing = json.loads((run_dir / "task_timing.json").read_text())
    assert timing["status"] == "sim_failed"


def test_task_runner_trajectory_deleted(tmp_path):
    """archive_trajectory=False removes trajectory_files after success."""
    mock_script = _make_mock_lammps_script(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    traj = run_dir / "out.lammpstrj"
    traj.write_text("# fake traj", encoding="utf-8")

    spec = _make_task_spec(run_dir, ["bash", str(mock_script)])
    spec["archive_trajectory"] = False
    spec["trajectory_files"] = [str(traj)]
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    from AceCG.schedulers.task_runner import run as tr_run
    tr_run(str(spec_path))
    assert not traj.exists(), "Trajectory should have been deleted"


def test_task_runner_trajectory_kept(tmp_path):
    """archive_trajectory=True keeps trajectory_files."""
    mock_script = _make_mock_lammps_script(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    traj = run_dir / "out.lammpstrj"
    traj.write_text("# fake traj", encoding="utf-8")

    spec = _make_task_spec(run_dir, ["bash", str(mock_script)])
    spec["archive_trajectory"] = True
    spec["trajectory_files"] = [str(traj)]
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    from AceCG.schedulers.task_runner import run as tr_run
    tr_run(str(spec_path))
    assert traj.exists(), "Trajectory should have been kept"


def test_task_runner_post_failure_writes_timing(tmp_path, monkeypatch):
    """Post-processing failures still write task_timing.json."""
    mock_script = _make_mock_lammps_script(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = _make_task_spec(run_dir, ["bash", str(mock_script)])
    spec["post_spec"] = {"work_dir": str(run_dir), "steps": []}
    spec["post_exec"] = {"mode": "inproc"}
    spec_path = tmp_path / "task_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    class _BrokenEngine:
        def run_post(self, post_spec):
            raise RuntimeError("post boom")

    import AceCG.compute.mpi_engine as mpi_engine

    monkeypatch.setattr(mpi_engine, "build_default_engine", lambda: _BrokenEngine())

    from AceCG.schedulers.task_runner import run as tr_run

    with pytest.raises(RuntimeError, match="post boom"):
        tr_run(str(spec_path))

    timing = json.loads((run_dir / "task_timing.json").read_text(encoding="utf-8"))
    assert timing["status"] == "post_failed"
    assert timing["post_wall"] >= 0
    assert "post boom" in timing["post_error"]
    assert timing["total_wall"] >= timing["sim_wall"]


def test_task_runner_requires_post_exec_for_post_spec(tmp_path):
    """post_spec no longer falls back to in-process post implicitly."""
    mock_script = _make_mock_lammps_script(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = _make_task_spec(run_dir, ["bash", str(mock_script)])
    spec["post_spec"] = {"work_dir": str(run_dir), "steps": []}
    spec_path = tmp_path / "task_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    from AceCG.schedulers.task_runner import run as tr_run

    with pytest.raises(RuntimeError, match="post_spec requires post_exec.mode"):
        tr_run(str(spec_path))

    timing = json.loads((run_dir / "task_timing.json").read_text(encoding="utf-8"))
    assert timing["status"] == "post_failed"
    assert "post_exec.mode" in timing["post_error"]


def test_task_runner_requires_post_launch_for_mpi_post_exec(tmp_path):
    """MPI post_exec specs now require a serialized post_launch."""
    mock_script = _make_mock_lammps_script(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = _make_task_spec(run_dir, ["bash", str(mock_script)])
    spec["post_spec"] = {"work_dir": str(run_dir), "steps": []}
    spec["post_exec"] = {"mode": "mpi"}
    spec_path = tmp_path / "task_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    from AceCG.schedulers.task_runner import run as tr_run

    with pytest.raises(RuntimeError, match="no post_launch found"):
        tr_run(str(spec_path))

    timing = json.loads((run_dir / "task_timing.json").read_text(encoding="utf-8"))
    assert timing["status"] == "post_failed"
    assert timing["post_wall"] >= 0
    assert "no post_launch found" in timing["post_error"]
