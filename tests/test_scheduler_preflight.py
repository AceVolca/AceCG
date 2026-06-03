"""Unit tests for preflight benchmark helpers."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import AceCG
import AceCG.schedulers
from AceCG.schedulers.profiler import (
    _compute_rank_candidates,
    _parse_total_steps,
    _shorten_lammps_input,
    preflight_benchmark,
)


def test_preflight_export_present():
    assert AceCG.schedulers.preflight_benchmark is preflight_benchmark
    assert AceCG.preflight_benchmark is preflight_benchmark


def test_shorten_lammps_input_strips_io_and_rewrites_runs():
    script = """include system.init
run 10000
dump traj all custom 1 traj.lammpstrj id type x y z
dump_modify traj sort id
run 50000
write_dump all custom min.xyz id type x y z
write_restart final.restart
undump traj
"""
    shortened = _shorten_lammps_input(script, bench_steps=1000)
    assert shortened.count("run 1000") == 2
    assert "dump traj" not in shortened
    assert "dump_modify" not in shortened
    assert "write_dump" not in shortened
    assert "write_restart" not in shortened
    assert "undump" not in shortened


def test_parse_total_steps_uses_last_loop_time(tmp_path):
    log = tmp_path / "bench.log"
    log.write_text(
        "Loop time of 0.5 on 8 procs for 100 steps with 196 atoms\n"
        "Loop time of 3.2 on 8 procs for 1000 steps with 196 atoms\n",
        encoding="utf-8",
    )
    assert _parse_total_steps(log) == 1000


def test_compute_rank_candidates_descending_divisors():
    assert _compute_rank_candidates(16) == [16, 8, 4]


def test_preflight_benchmark_selects_best_makespan(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sim_input = run_dir / "in.lmp"
    sim_input.write_text("run 10000\n", encoding="utf-8")

    def fake_run(cmd, cwd, timeout, capture_output, env):
        nranks = int(cmd[2])
        loop_time = {8: 10.0, 4: 6.0}[nranks]
        bench_log = Path(cwd) / "bench.log"
        bench_log.write_text(
            f"Loop time of {loop_time} on {nranks} procs for 1000 steps with 196 atoms\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = preflight_benchmark(
        sim_input=str(sim_input),
        run_dir=str(run_dir),
        sim_cmd=["lmp"],
        mpirun_path="mpirun",
        cpus_per_node=8,
        candidate_divisors=(1, 2),
        production_steps=50000,
        n_tasks=6,
        n_nodes=1,
    )

    assert result["best_ranks"] == 4
    assert result["best_cpu_cores"] == 4
    assert result["best_slots_per_node"] == 2
    assert len(result["trials"]) == 2
