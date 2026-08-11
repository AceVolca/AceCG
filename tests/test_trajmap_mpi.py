"""MPI characterizations for the public TrajMap workflow."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
import yaml

from AceCG.configs.trajmap_config import parse_trajmap_file
from AceCG.workflows.trajmap import TrajMapWorkflow


def _write_inputs(root: Path) -> dict[str, Path]:
    """Create a small self-describing source that exercises ordered selection."""
    universe = mda.Universe.empty(
        6,
        n_residues=1,
        atom_resindex=np.zeros(6, dtype=int),
        trajectory=True,
        forces=True,
    )
    universe.add_TopologyAttr("name", ["C", "H", "H", "C", "H", "H"])
    universe.add_TopologyAttr("type", ["C", "H", "H", "C", "H", "H"])
    universe.add_TopologyAttr("resname", ["LIP"])
    universe.add_TopologyAttr("resid", [1])
    universe.add_TopologyAttr("mass", [12.0, 1.0, 1.0, 12.0, 1.0, 1.0])
    box = np.array([25.0, 26.0, 27.0, 90.0, 90.0, 90.0])
    gro = root / "aa.gro"
    universe.atoms.positions = np.arange(18, dtype=np.float32).reshape(6, 3)
    universe.dimensions = box
    universe.atoms.write(gro)
    trr = root / "aa.trr"
    with mda.Writer(str(trr), n_atoms=6) as writer:
        for frame in range(6):
            universe.atoms.positions = np.arange(18, dtype=np.float32).reshape(6, 3) + frame
            universe.atoms.forces = np.arange(18, dtype=np.float32).reshape(6, 3) + frame
            universe.dimensions = box
            universe.trajectory.ts.time = float(frame)
            writer.write(universe.atoms)
    mapping = root / "map.yaml"
    mapping.write_text(
        yaml.safe_dump(
            {
                "site-types": {
                    "H": {"index": [0, 1, 2], "x-weight": [1.0, 1.0, 1.0]},
                    "T": {"index": [0, 1, 2], "x-weight": [1.0, 1.0, 1.0]},
                },
                "system": [{"anchor": 0, "repeat": 1, "offset": 6, "sites": [["H", 0], ["T", 3]]}],
                "cg-topology": {"molecule": {"names": ["H", "T"], "masses": [14.0, 14.0]}},
            },
            sort_keys=False,
        )
    )
    return {"gro": gro, "trr": trr, "map": mapping}


def _write_config(root: Path, inputs: dict[str, Path], output_dir: str, frame_ids: str, *, force_map: bool = False, force_fit_ids: str | None = None) -> Path:
    config = root / f"{output_dir}.acg"
    fit_line = "" if force_fit_ids is None else f"fit_frame_ids = {force_fit_ids}"
    force_mapping = "" if not force_map else f"""
[force_mapping]
method = optimal_linear
scope = per_template
backend = native
constraints = none
l2_regularization = 1e-8
{fit_line}
"""
    config.write_text(
        f"""
[aa]
topology = {inputs['gro']}
trajectory_files = {inputs['trr']}
trajectory_format = TRR
frame_ids = {frame_ids}
include_forces = {str(force_map).lower()}

[mapping]
map_file = {inputs['map']}
resname = LIP

[trajmap]
output_dir = {output_dir}
trajectory_name = cg.{"trr" if force_map else "xtc"}
{force_mapping}
"""
    )
    return config


def _run_mpi(
    config: Path,
    *,
    succeeds: bool = True,
    worker_setup: str = "",
) -> subprocess.CompletedProcess[str]:
    pytest.importorskip("mpi4py")
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    worker = f"""
from mpi4py import MPI
from AceCG.configs.trajmap_config import parse_trajmap_file
from AceCG.workflows.trajmap import TrajMapWorkflow
import sys
{worker_setup}
TrajMapWorkflow(parse_trajmap_file(sys.argv[1]), comm=MPI.COMM_WORLD).run()
MPI.COMM_WORLD.Barrier()
"""
    completed = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, "-c", worker, str(config)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    if succeeds:
        assert completed.returncode == 0, completed.stderr
    else:
        assert completed.returncode != 0
    return completed


def test_force_map_config_interpolates_optional_fit_ids(tmp_path):
    inputs = {name: tmp_path / f"{name}.input" for name in ("gro", "trr", "map")}
    config = _write_config(
        tmp_path, inputs, "force_config", "[0, 1]",
        force_map=True, force_fit_ids="[1]",
    )
    parsed = parse_trajmap_file(config)
    assert parsed.force_mapping.fit_frame_ids == (1,)


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="MPI launcher is unavailable")
def test_public_workflow_matches_serial_and_closes_an_empty_rank(tmp_path):
    inputs = _write_inputs(tmp_path)
    serial_config = _write_config(tmp_path, inputs, "serial", "[5, 0, 2, 2]")
    mpi_config = _write_config(tmp_path, inputs, "mpi", "[5, 0, 2, 2]")
    serial = TrajMapWorkflow(parse_trajmap_file(serial_config)).run()

    _run_mpi(mpi_config)

    mpi_dir = tmp_path / "mpi"
    assert (mpi_dir / "cg.xtc").read_bytes() == serial.trajectory_path.read_bytes()
    serial_universe = mda.Universe(serial.topology_path, serial.trajectory_path, format="XTC")
    mpi_universe = mda.Universe(mpi_dir / "cg.data", mpi_dir / "cg.xtc", format="XTC")
    for serial_ts, mpi_ts in zip(serial_universe.trajectory, mpi_universe.trajectory):
        assert mpi_ts.positions == pytest.approx(serial_ts.positions, abs=1e-3)
        assert mpi_ts.dimensions == pytest.approx(serial_ts.dimensions, abs=1e-3)
    mpi_report = yaml.safe_load((mpi_dir / "trajmap_report.json").read_text())
    assert mpi_report["source"]["frame_ids"] == [5, 0, 2, 2]
    assert [(item["selected_offset"], item["selected_count"]) for item in mpi_report["mpi"]["rank_slices"]] == [(0, 2), (2, 2)]

    empty_config = _write_config(tmp_path, inputs, "mpi_empty", "[3]")
    _run_mpi(empty_config)

    empty_report = yaml.safe_load((tmp_path / "mpi_empty" / "trajmap_report.json").read_text())
    assert [(item["selected_offset"], item["selected_count"], item["written_count"]) for item in empty_report["mpi"]["rank_slices"]] == [(0, 1, 1), (1, 0, 0)]


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="MPI launcher is unavailable")
def test_force_map_trr_matches_serial_operator_diagnostics_and_empty_rank(tmp_path):
    inputs = _write_inputs(tmp_path)
    serial_config = _write_config(tmp_path, inputs, "force_serial", "[5, 0, 2]", force_map=True)
    mpi_config = _write_config(tmp_path, inputs, "force_mpi", "[5, 0, 2]", force_map=True)
    serial = TrajMapWorkflow(parse_trajmap_file(serial_config)).run()
    _run_mpi(mpi_config)

    mpi_dir = tmp_path / "force_mpi"
    assert (mpi_dir / "cg.trr").read_bytes() == serial.trajectory_path.read_bytes()
    with np.load(serial.force_map_path, allow_pickle=False) as serial_operator, np.load(mpi_dir / "force_map.npz", allow_pickle=False) as mpi_operator:
        assert set(mpi_operator.files) == set(serial_operator.files)
        for name in serial_operator.files:
            if serial_operator[name].dtype.kind in "OUS":
                assert np.array_equal(mpi_operator[name], serial_operator[name])
            else:
                assert mpi_operator[name] == pytest.approx(serial_operator[name])
    serial_report = yaml.safe_load(serial.report_path.read_text())
    mpi_report = yaml.safe_load((mpi_dir / "trajmap_report.json").read_text())
    assert mpi_report["force_mapping"]["diagnostics"] == pytest.approx(serial_report["force_mapping"]["diagnostics"])

    empty_config = _write_config(tmp_path, inputs, "force_empty", "[3]", force_map=True)
    _run_mpi(empty_config)
    empty_report = yaml.safe_load((tmp_path / "force_empty" / "trajmap_report.json").read_text())
    assert [(item["selected_offset"], item["selected_count"], item["written_count"]) for item in empty_report["mpi"]["rank_slices"]] == [(0, 1, 1), (1, 0, 0)]

    failed = _write_config(tmp_path, inputs, "force_failure", "[3]", force_map=True, force_fit_ids="[99]")
    completed = _run_mpi(failed, succeeds=False)
    assert "fit_frame_ids must be selected mapping frames" in completed.stderr

    local_failure = _write_config(
        tmp_path, inputs, "force_local_failure", "[5, 0, 2]", force_map=True,
    )
    completed = _run_mpi(
        local_failure,
        succeeds=False,
        worker_setup="""
from AceCG.io.trajectory import MPITrajReader
_original_iter_local = MPITrajReader.iter_local
def _fail_one_fitting_rank(self, **kwargs):
    records = _original_iter_local(self, **kwargs)
    if MPI.COMM_WORLD.Get_rank() != 1 or not kwargs.get("include_forces"):
        return records
    def _fail_during_iteration():
        for _ in records:
            raise RuntimeError("injected rank-local force-fit failure")
    return _fail_during_iteration()
MPITrajReader.iter_local = _fail_one_fitting_rank
""",
    )
    assert "injected rank-local force-fit failure" in completed.stderr
