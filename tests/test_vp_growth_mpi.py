"""MPI integration test for :func:`grow_vp_frames`.

Spawns a 2-rank ``mpirun`` subprocess that grows the same fixture used
by the serial test and compares each per-frame output against a
serial-run baseline. If ``mpirun`` is not on ``PATH`` the test is
skipped rather than failed.
"""

from __future__ import annotations

from functools import lru_cache
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_DRIVER = r"""
import os, sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
from AceCG.compute.vp_prepare import grow_vp_frames
from AceCG.configs.vp_config import VPAtomDef, VPConfig, VPInteractionDef
from AceCG.topology.vpgrower import VPGrower


def build_universe(n_frames):
    n_atoms = 8
    atom_resindex = np.asarray([0,0,0,0,1,1,1,1], dtype=np.int64)
    u = mda.Universe.empty(n_atoms, n_residues=2,
                           atom_resindex=atom_resindex, trajectory=False)
    types = np.array(["1","2","3","4"] * 2, dtype=object)
    u.add_TopologyAttr("types", types)
    u.add_TopologyAttr("names", types.copy())
    u.add_TopologyAttr("masses", np.full(n_atoms, 72.0))
    u.add_TopologyAttr("charges", np.zeros(n_atoms))
    u.add_TopologyAttr("resids", np.array([1,2], dtype=np.int64))
    u.add_TopologyAttr("resnames", np.array(["DOPC","DOPC"], dtype=object))
    u.add_TopologyAttr("bonds", np.array(
        [(0,1),(1,2),(2,3),(4,5),(5,6),(6,7)], dtype=np.int64))
    u.add_TopologyAttr("angles", np.array([(0,1,2),(4,5,6)], dtype=np.int64))
    base = np.array([
        [0.,0.,0.],[2.,0.,0.],[4.,0.,0.],[6.,0.,0.],
        [0.,6.,0.],[2.,6.,0.],[4.,6.,0.],[6.,6.,0.],
    ])
    coords = np.stack([base + i*0.1 for i in range(n_frames)], axis=0)
    dims = np.tile(np.array([30.,30.,30.,90.,90.,90.]), (n_frames,1))
    u.trajectory = MemoryReader(coords, order="fac", dimensions=dims, dt=1.0)
    return u


def vp_cfg():
    return VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=(VPInteractionDef(type_keys=("VP","MG"), pot_style="harmonic",
                                pot_kwargs={"k":2.5,"r0":1.5}),),
        angles=(VPInteractionDef(type_keys=("VP","MG","HG"), pot_style="harmonic",
                                 pot_kwargs={"k":2.45,"theta0":135.0}),),
        pairs=(), selection="resname DOPC", atomtype_order="back",
        clash_max_passes=8, clash_min_distance=1.5,
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()
    u = build_universe(6)
    grower = VPGrower.from_universe(
        u, vp_cfg(), type_aliases={1:"HG",2:"MG",3:"T1",4:"T2"},
    )
    comm = MPI.COMM_WORLD
    grow_vp_frames(grower=grower, universe=u,
                   frame_ids=[0,1,2,3,4,5],
                   output_dir=args.out,
                   orientation_seed_base=100,
                   comm=comm)

if __name__ == "__main__":
    main()
"""


def _have_mpirun() -> bool:
    return shutil.which("mpirun") is not None


@lru_cache(maxsize=1)
def _mpi_runtime_unavailable_reason(python: str) -> str | None:
    if not _have_mpirun():
        return "mpirun not on PATH"
    env = dict(os.environ)
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
        env=env,
    )
    if probe.returncode != 0:
        output = (probe.stderr or probe.stdout or "").strip()
        return output or f"mpirun probe exited with code {probe.returncode}"
    if "1" not in probe.stdout.split():
        output = (probe.stdout or probe.stderr or "").strip()
        return output or "mpirun probe returned unexpected output"
    return None


@pytest.mark.skipif(not _have_mpirun(), reason="mpirun not on PATH")
def test_mpi_matches_serial_byte_for_byte(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER)

    serial_out = tmp_path / "serial"
    parallel_out = tmp_path / "parallel"
    serial_out.mkdir()
    parallel_out.mkdir()

    python = sys.executable
    env = dict(os.environ)
    unavailable_reason = _mpi_runtime_unavailable_reason(python)
    if unavailable_reason is not None:
        pytest.skip(f"MPI runtime unavailable: {unavailable_reason}")

    # Run once with size=1 (no comm).
    r1 = subprocess.run(
        [python, str(driver), str(serial_out)],
        cwd=str(repo_root), env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert r1.returncode == 0, f"serial failed: {r1.stderr}"

    # Run with 2 MPI ranks.
    r2 = subprocess.run(
        ["mpirun", "-n", "2", python, str(driver), str(parallel_out)],
        cwd=str(repo_root), env=env,
        capture_output=True, text=True, timeout=180,
    )
    assert r2.returncode == 0, f"mpi failed: {r2.stderr}"

    for fid in range(6):
        name = f"frame_{fid:06d}.data"
        a = (serial_out / name).read_bytes()
        b = (parallel_out / name).read_bytes()
        assert a == b, f"frame {fid} differs between serial and 2-rank runs"
