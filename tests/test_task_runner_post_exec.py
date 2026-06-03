import json
from pathlib import Path

from AceCG.schedulers import task_runner
from AceCG.schedulers.mpi_backend import LaunchSpec
from AceCG.schedulers.resource_pool import HostInventory


def test_run_post_via_launch_writes_spec_and_uses_launch_env(monkeypatch, tmp_path):
    captured = {}
    timing = {}

    class _Result:
        returncode = 0

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, env=None):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        captured["env"] = dict(env)
        return _Result()

    monkeypatch.setattr(task_runner.subprocess, "run", _fake_run)
    monkeypatch.setenv("PMI_RANK", "7")

    task_runner._run_post_via_launch(
        run_dir=tmp_path,
        post_spec={"post_mode": "one_pass", "work_dir": str(tmp_path), "steps": []},
        post_launch={
            "argv": ["mpirun", "-np", "6", "python", "-m", "AceCG.compute.mpi_engine"],
            "env_add": {"FROM_LAUNCH": "1"},
            "env_strip_prefixes": ["PMI_"],
        },
        timing=timing,
        task_extra_env={"OMP_NUM_THREADS": "2", "FROM_LAUNCH": "override"},
    )

    spec_path = tmp_path / "mpi_post_spec.json"
    assert captured["cwd"] == str(tmp_path)
    assert captured["cmd"][:-1] == [
        "mpirun", "-np", "6", "python", "-m", "AceCG.compute.mpi_engine"
    ]
    assert captured["cmd"][-1] == str(spec_path)
    assert captured["env"]["OMP_NUM_THREADS"] == "2"
    assert captured["env"]["FROM_LAUNCH"] == "override"
    assert "PMI_RANK" not in captured["env"]
    assert timing["mpi_post_returncode"] == 0
    assert json.loads(spec_path.read_text(encoding="utf-8"))["post_mode"] == "one_pass"


def test_run_post_builds_launch_from_resource_pool(monkeypatch, tmp_path):
    captured = {}
    realize_args = {}

    class _Result:
        returncode = 0

    class _Backend:
        def realize(self, placement, payload_cmd, run_dir):
            realize_args["placement"] = placement
            realize_args["payload_cmd"] = list(payload_cmd)
            realize_args["run_dir"] = run_dir
            return LaunchSpec(
                argv=("launcher", "--ranks", str(placement.n_ranks), *payload_cmd),
                env_add={"FAKE_MPI": "1"},
                env_strip_prefixes=("PMI_",),
            )

    class _Pool:
        hosts = [
            HostInventory("node-a", (0, 1, 2)),
            HostInventory("node-b", (4, 5, 6, 7, 8)),
        ]
        backend = _Backend()
        extra_env = {"OMP_NUM_THREADS": "1", "FROM_POOL": "yes"}

    def _fake_run(cmd, cwd=None, stdout=None, stderr=None, env=None):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        captured["env"] = dict(env)
        return _Result()

    monkeypatch.setattr(task_runner.subprocess, "run", _fake_run)

    task_runner.run_post(
        {"post_mode": "one_pass", "work_dir": str(tmp_path), "steps": []},
        _Pool(),
        run_dir=tmp_path,
        python_exe="/opt/python/bin/python",
    )

    placement = realize_args["placement"]
    assert realize_args["run_dir"] == tmp_path
    assert placement.n_ranks == 8
    assert [slice_.host for slice_ in placement.slices] == ["node-a", "node-b"]
    assert [slice_.cpu_ids for slice_ in placement.slices] == [(0, 1, 2), (4, 5, 6, 7, 8)]
    assert realize_args["payload_cmd"] == [
        "/opt/python/bin/python",
        "-m",
        "AceCG.compute.mpi_engine",
    ]
    assert captured["cwd"] == str(tmp_path)
    assert captured["cmd"][:3] == ["launcher", "--ranks", "8"]
    assert captured["cmd"][-1] == str(tmp_path / "mpi_post_spec.json")
    assert captured["env"]["FAKE_MPI"] == "1"
    assert captured["env"]["OMP_NUM_THREADS"] == "1"
    assert captured["env"]["FROM_POOL"] == "yes"
    assert Path(tmp_path / "mpi_post_spec.json").exists()
