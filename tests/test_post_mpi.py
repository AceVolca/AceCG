from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from AceCG.compute.mpi_engine import build_default_engine
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey


try:
    MPI = pytest.importorskip("mpi4py.MPI")
except RuntimeError as exc:
    # mpi4py's ABI loader raises RuntimeError, not ImportError, when no MPI
    # library is available; importorskip only converts ImportError, so without
    # this the whole suite aborts on a bare login node.
    pytest.skip(f"mpi4py cannot load an MPI library: {exc}", allow_module_level=True)


def _write_real_inputs(work_dir: Path) -> tuple[Path, Path, Path]:
    topology_path = work_dir / "two_atom.data"
    topology_path.write_text(
        """Two-atom FM integration topology

2 atoms
1 bonds
0 angles
0 dihedrals
0 impropers

2 atom types
1 bond types

0.0 20.0 xlo xhi
0.0 20.0 ylo yhi
0.0 20.0 zlo zhi

Masses

1 1.0
2 1.0

Atoms # full

1 1 1 0.0 1.0 1.0 1.0
2 1 2 0.0 4.0 1.0 1.0

Bonds

1 1 1 2
""",
        encoding="utf-8",
    )
    trajectory_path = work_dir / "two_atom.lammpstrj"
    rows = [(0, 4.0, 1.0, -1.0), (1, 4.5, 0.5, -0.5), (2, 5.0, -0.25, 0.25), (3, 5.5, -0.75, 0.75)]
    trajectory_path.write_text(
        "".join(
            f"""ITEM: TIMESTEP
{timestep}
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 20.0
0.0 20.0
0.0 20.0
ITEM: ATOMS id type x y z fx fy fz
1 1 1.0 1.0 1.0 {fx1} 0.0 0.0
2 2 {x2} 1.0 1.0 {fx2} 0.0 0.0
"""
            for timestep, x2, fx1, fx2 in rows
        ),
        encoding="utf-8",
    )
    bond_key = InteractionKey.bond("A", "B")
    forcefield = Forcefield({bond_key: [HarmonicPotential("A", "B", k=5.0, r0=4.0)]})
    forcefield_path = work_dir / "forcefield.pkl"
    with forcefield_path.open("wb") as handle:
        pickle.dump(forcefield, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return topology_path, trajectory_path, forcefield_path


def _base_spec(work_dir: Path, topology_path: Path, trajectory_path: Path, forcefield_path: Path) -> dict:
    return {
        "work_dir": str(work_dir), "topology": str(topology_path),
        "trajectory": str(trajectory_path), "trajectory_format": "LAMMPSDUMP",
        "forcefield_path": str(forcefield_path),
        "atom_type_name_aliases": {"1": "A", "2": "B"},
        "frame_weight": [1.0, 2.0, 3.0, 4.0], "step_index": 9,
    }


def _assert_fm_equal(actual: dict, expected: dict) -> None:
    for key in ("JtJ", "Jty", "Jtf"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1.0e-12, atol=1.0e-12)
    for key in ("y_sumsq", "f_sumsq", "fty", "weight_sum"):
        assert actual[key] == pytest.approx(expected[key], rel=1.0e-12, abs=1.0e-12)
    for key in ("nframe", "n_atoms_obs", "step_index"):
        assert actual[key] == expected[key]


def _assert_distribution_equal(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected)
    assert all(isinstance(key, InteractionKey) for key in actual)
    for key in actual:
        observed, reference = actual[key], expected[key]
        assert observed.key == reference.key
        for field in ("x", "values", "counts", "edges"):
            np.testing.assert_allclose(getattr(observed, field), getattr(reference, field))
        assert observed.mode == reference.mode
        assert observed.variable == reference.variable
        assert observed.n_frames == reference.n_frames
        assert observed.weight_sum == pytest.approx(reference.weight_sum)
        assert observed.meta == reference.meta


def test_post_mpi_fm_serial_two_rank_and_empty_slice(tmp_path: Path) -> None:
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        pytest.skip("P04B MPI gate requires exactly two ranks")
    rank = comm.Get_rank()
    work_dir = Path(comm.bcast(str(tmp_path) if rank == 0 else None, root=0))
    if rank == 0:
        paths = tuple(map(str, _write_real_inputs(work_dir)))
    else:
        paths = None
    topology_path, trajectory_path, forcefield_path = map(Path, comm.bcast(paths, root=0))

    serial_output, mpi_output = work_dir / "serial_fm.pkl", work_dir / "mpi_fm.pkl"
    if rank == 0:
        serial_spec = _base_spec(work_dir, topology_path, trajectory_path, forcefield_path)
        serial_spec["steps"] = [{"step_mode": "fm", "output_file": str(serial_output)}]
        build_default_engine().run_post(serial_spec)
    comm.Barrier()
    mpi_spec = _base_spec(work_dir, topology_path, trajectory_path, forcefield_path)
    mpi_spec.update(expected_mpi_size=2, steps=[{"step_mode": "fm", "output_file": str(mpi_output)}])
    build_default_engine(comm=comm).run_post(mpi_spec)
    comm.Barrier()
    if rank == 0:
        with serial_output.open("rb") as handle:
            serial_payload = pickle.load(handle)
        with mpi_output.open("rb") as handle:
            mpi_payload = pickle.load(handle)
        _assert_fm_equal(mpi_payload, serial_payload)
        assert not mpi_output.with_name(mpi_output.name + ".bak").exists()

    serial_stack, mpi_stack = work_dir / "serial_stack.pkl", work_dir / "mpi_stack.pkl"
    if rank == 0:
        serial_spec = _base_spec(work_dir, topology_path, trajectory_path, forcefield_path)
        serial_spec.update(frame_ids=[2], frame_weight=[3.0], steps=[{"step_mode": "fm", "reduce_stack": True, "output_file": str(serial_stack)}])
        build_default_engine().run_post(serial_spec)
    comm.Barrier()
    mpi_spec = _base_spec(work_dir, topology_path, trajectory_path, forcefield_path)
    mpi_spec.update(expected_mpi_size=2, frame_ids=[2], frame_weight=[3.0], steps=[{"step_mode": "fm", "reduce_stack": True, "output_file": str(mpi_stack)}])
    build_default_engine(comm=comm).run_post(mpi_spec)
    comm.Barrier()
    if rank == 0:
        with serial_stack.open("rb") as handle:
            serial_payload = pickle.load(handle)
        with mpi_stack.open("rb") as handle:
            mpi_payload = pickle.load(handle)
        for key in ("JtJ_frame", "Jty_frame", "y_sumsq_frame", "Jtf_frame", "f_sumsq_frame", "fty_frame", "weight_frame", "frame_ids"):
            np.testing.assert_allclose(mpi_payload[key], serial_payload[key])
        np.testing.assert_array_equal(mpi_payload["frame_ids"], [2])
        np.testing.assert_allclose(mpi_payload["weight_frame"], [3.0])
        assert mpi_payload["n_frames"] == 1
        assert mpi_payload["n_atoms_obs"] == serial_payload["n_atoms_obs"]
        assert not mpi_stack.with_name(mpi_stack.name + ".bak").exists()


def test_post_mpi_rdf_serial_two_rank_and_empty_slice(tmp_path: Path) -> None:
    comm = MPI.COMM_WORLD
    if comm.Get_size() != 2:
        pytest.skip("P04B MPI gate requires exactly two ranks")
    rank = comm.Get_rank()
    work_dir = Path(comm.bcast(str(tmp_path) if rank == 0 else None, root=0))
    if rank == 0:
        paths = tuple(map(str, _write_real_inputs(work_dir)))
    else:
        paths = None
    topology_path, trajectory_path, forcefield_path = map(Path, comm.bcast(paths, root=0))
    pair_key = InteractionKey.pair("A", "B")
    step = {"step_mode": "rdf", "interaction_keys": [pair_key.label()], "cutoff": 8.0, "nbins_pair": 8}
    serial_output, mpi_output = work_dir / "serial_rdf.pkl", work_dir / "mpi_rdf.pkl"
    if rank == 0:
        serial_spec = _base_spec(work_dir, topology_path, trajectory_path, forcefield_path)
        serial_spec["steps"] = [{**step, "output_file": str(serial_output)}]
        build_default_engine().run_post(serial_spec)
    comm.Barrier()
    mpi_spec = _base_spec(work_dir, topology_path, trajectory_path, forcefield_path)
    mpi_spec.update(expected_mpi_size=2, frame_ids=[2], frame_weight=[3.0], steps=[{**step, "output_file": str(mpi_output)}])
    build_default_engine(comm=comm).run_post(mpi_spec)
    comm.Barrier()
    if rank == 0:
        with serial_output.open("rb") as handle:
            serial_payload = pickle.load(handle)
        with mpi_output.open("rb") as handle:
            mpi_payload = pickle.load(handle)
        assert mpi_payload[pair_key].n_frames == 1
        assert mpi_payload[pair_key].weight_sum == pytest.approx(3.0)
        assert not mpi_output.with_name(mpi_output.name + ".bak").exists()
    comm.Barrier()
    if rank == 0:
        selected_serial = _base_spec(work_dir, topology_path, trajectory_path, forcefield_path)
        selected_serial.update(frame_ids=[2], frame_weight=[3.0], steps=[{**step, "output_file": str(serial_output)}])
        build_default_engine().run_post(selected_serial)
        with serial_output.open("rb") as handle:
            selected_payload = pickle.load(handle)
        _assert_distribution_equal(mpi_payload, selected_payload)
