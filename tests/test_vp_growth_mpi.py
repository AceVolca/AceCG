"""Real-MPI VP workflow equivalence and collective failure closure."""

from __future__ import annotations

from functools import lru_cache
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_DRIVER = r'''
import argparse
import sys
from pathlib import Path

from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from AceCG.configs.vp_config import VPAtomDef, VPConfig, VPInteractionDef
from AceCG.configs.vp_growth_config import VPGrowthAARef, VPGrowthConfig, VPGrowthRun
from AceCG.topology.vpgrower import VPGrower
from AceCG.workflows.vp_growth import VPGrowthWorkflow


def write_inputs(root):
    topology = """CG reference

8 atoms
6 bonds
2 angles

4 atom types
3 bond types
1 angle types

0.0 30.0 xlo xhi
0.0 30.0 ylo yhi
0.0 30.0 zlo zhi

Masses

1 72.0
2 72.0
3 72.0
4 72.0

Atoms # full

1 1 1 0.0 0.0 0.0 0.0
2 1 2 0.0 2.0 0.0 0.0
3 1 3 0.0 4.0 0.0 0.0
4 1 4 0.0 6.0 0.0 0.0
5 2 1 0.0 0.0 6.0 0.0
6 2 2 0.0 2.0 6.0 0.0
7 2 3 0.0 4.0 6.0 0.0
8 2 4 0.0 6.0 6.0 0.0

Bonds

1 1 1 2
2 2 2 3
3 3 3 4
4 1 5 6
5 2 6 7
6 3 7 8

Angles

1 1 1 2 3
2 1 5 6 7
"""
    (root / "source.data").write_text(topology)
    base = [
        (0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
        (4.0, 0.0, 0.0), (6.0, 0.0, 0.0),
        (0.0, 6.0, 0.0), (2.0, 6.0, 0.0),
        (4.0, 6.0, 0.0), (6.0, 6.0, 0.0),
    ]
    lines = []
    for frame in range(6):
        lines.extend([
            "ITEM: TIMESTEP", str(frame), "ITEM: NUMBER OF ATOMS", "8",
            "ITEM: BOX BOUNDS pp pp pp", "0 30", "0 30", "0 30",
            "ITEM: ATOMS id type x y z fx fy fz",
        ])
        for atom_id, (x, y, z) in enumerate(base, 1):
            atom_type = (atom_id - 1) % 4 + 1
            shift = frame * 0.1
            lines.append(
                f"{atom_id} {atom_type} {x + shift} {y + shift} {z + shift} "
                f"{frame + 0.25} {frame + 0.5} {frame + 0.75}"
            )
    (root / "source.lammpstrj").write_text("\n".join(lines) + "\n")


def config(root, output, one_frame=False):
    vp = VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=(VPInteractionDef(type_keys=("VP", "MG"), pot_style="harmonic",
                                pot_kwargs={"k": 2.5, "r0": 1.5}),),
        angles=(VPInteractionDef(type_keys=("VP", "MG", "HG"), pot_style="harmonic",
                                 pot_kwargs={"k": 2.45, "theta0": 135.0}),),
        selection=None, atomtype_order="back", clash_max_passes=8,
        clash_min_distance=1.5,
    )
    return VPGrowthConfig(
        path=root / "run.acg",
        aa_ref=VPGrowthAARef(
            trajectory_files=("source.lammpstrj",),
            trajectory_format="LAMMPSDUMP", ref_topo="source.data",
            ref_topo_type_names={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
            include_forces=True,
        ),
        vp=vp,
        run=VPGrowthRun(
            output_dir=str(output),
            frame_ids=((5,) if one_frame else (5, 2, 2, 0)),
            orientation_seed_base=100, table_points=31,
            table_rmin=0.01, table_rmax=10.0,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("out")
    parser.add_argument("--fail-rank", type=int)
    parser.add_argument("--one-frame", action="store_true")
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    active_comm = comm if comm.Get_size() > 1 else None
    root = Path(args.root)
    if comm.Get_rank() == 0:
        root.mkdir(parents=True, exist_ok=True)
        write_inputs(root)
    comm.Barrier()

    if args.fail_rank == comm.Get_rank():
        def fail_growth(self, *positional, **keyword):
            raise LookupError("injected rank-local VP growth failure")
        VPGrower.grow_frame = fail_growth

    try:
        VPGrowthWorkflow(
            config(root, Path(args.out), one_frame=args.one_frame), comm=active_comm
        ).run()
    except Exception as exc:
        if args.fail_rank is None:
            raise
        print(
            f"FAILURE rank={comm.Get_rank()} type={type(exc).__name__} message={exc}",
            flush=True,
        )


if __name__ == "__main__":
    main()
'''


def _have_mpirun() -> bool:
    return shutil.which("mpirun") is not None


@lru_cache(maxsize=1)
def _mpi_runtime_unavailable_reason(python: str) -> str | None:
    if not _have_mpirun():
        return "mpirun not on PATH"
    probe = subprocess.run(
        [
            "mpirun",
            "-n",
            "1",
            python,
            "-c",
            "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ),
    )
    if probe.returncode != 0:
        return (probe.stderr or probe.stdout or "").strip() or (
            f"mpirun probe exited with code {probe.returncode}"
        )
    return None


@pytest.mark.skipif(not _have_mpirun(), reason="mpirun not on PATH")
def test_workflow_mpi_matches_serial_and_closes_rank_failure(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER)
    python = sys.executable
    unavailable_reason = _mpi_runtime_unavailable_reason(python)
    if unavailable_reason is not None:
        pytest.skip(f"MPI runtime unavailable: {unavailable_reason}")
    env = dict(os.environ)
    serial_root = tmp_path / "serial-input"
    mpi_root = tmp_path / "mpi-input"
    serial_out = tmp_path / "serial"
    mpi_out = tmp_path / "parallel"

    serial = subprocess.run(
        [python, str(driver), str(serial_root), str(serial_out)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert serial.returncode == 0, serial.stderr
    parallel = subprocess.run(
        ["mpirun", "-n", "2", python, str(driver), str(mpi_root), str(mpi_out)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert parallel.returncode == 0, parallel.stderr

    for frame_id in (5, 2, 0):
        for suffix in ("data", "forces.npy"):
            name = f"frame_{frame_id:06d}.{suffix}"
            assert (serial_out / name).read_bytes() == (mpi_out / name).read_bytes()
    serial_manifest = json.loads((serial_out / "manifest.json").read_text())
    mpi_manifest = json.loads((mpi_out / "manifest.json").read_text())
    assert [item["source_frame_id"] for item in serial_manifest["occurrences"]] == [
        5,
        2,
        2,
        0,
    ]
    assert [item["orientation_seed"] for item in mpi_manifest["occurrences"]] == [
        105,
        102,
        102,
        100,
    ]
    assert len(list(mpi_out.glob("frame_*.data"))) == 3

    empty_rank_out = tmp_path / "empty-rank"
    empty_rank = subprocess.run(
        [
            "mpirun",
            "-n",
            "2",
            python,
            str(driver),
            str(tmp_path / "empty-rank-input"),
            str(empty_rank_out),
            "--one-frame",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert empty_rank.returncode == 0, empty_rank.stderr
    assert len(list(empty_rank_out.glob("frame_*.data"))) == 1

    failed_out = tmp_path / "failed"
    failure = subprocess.run(
        [
            "mpirun",
            "-n",
            "2",
            python,
            str(driver),
            str(tmp_path / "failed-input"),
            str(failed_out),
            "--fail-rank",
            "1",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert failure.returncode == 0, failure.stderr
    confirmations = [
        line for line in failure.stdout.splitlines() if line.startswith("FAILURE rank=")
    ]
    assert len(confirmations) == 2
    assert all("type=LookupError" in line for line in confirmations)
    assert all("injected rank-local VP growth failure" in line for line in confirmations)
    assert not (failed_out / "manifest.json").exists()
    assert not (failed_out / ".vp-growth-stage").exists()
