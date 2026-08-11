"""End-to-end ``acg-trajmap`` runs on a synthetic AA trajectory.

Everything here goes through the real config parser, the real workflow, and real
MDAnalysis readers/writers, so it covers the parts unit tests cannot: frame
selection, the writer contract, the CG topology file, segment merging, and the
JSON report. The system is deliberately tiny (18 AA atoms, 6 CG sites) — this
suite is about plumbing, not throughput.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import MDAnalysis as mda

import AceCG.workflows.trajmap as trajmap_workflow
import AceCG.io.trajmap as trajmap_io
from AceCG.compute.cgmap import CGMapper
from AceCG.configs.parser import ACGConfigError
from AceCG.configs.trajmap_config import parse_trajmap_file
from AceCG.topology.cgmap import CGMapSpec
from AceCG.workflows.trajmap import TrajMapWorkflow, main


N_MOL = 3
ATOMS_PER_MOL = 6
N_ATOMS = N_MOL * ATOMS_PER_MOL
N_FRAMES = 7
BOX = np.array([25.0, 26.0, 27.0, 90.0, 90.0, 90.0])


# ─── synthetic inputs ─────────────────────────────────────────────────


def aa_positions(frame: int) -> np.ndarray:
    """Deterministic per-frame AA coordinates, wrapped into the cell.

    Molecule 1 straddles the x boundary on every frame, so a run with
    ``unwrap="none"`` would give visibly different CG positions.
    """
    rng = np.random.default_rng(1000 + frame)
    positions = rng.random((N_ATOMS, 3)) * 4.0
    for mol in range(N_MOL):
        lo = mol * ATOMS_PER_MOL
        positions[lo : lo + ATOMS_PER_MOL] += np.array([8.0 * mol, 3.0, 4.0])
    positions[ATOMS_PER_MOL : 2 * ATOMS_PER_MOL, 0] += BOX[0] - 9.0
    return np.mod(positions, BOX[:3]).astype(np.float32)


def mapping_dict() -> dict:
    """Two sites per molecule, three atoms each, with a CG bond."""
    return {
        "site-types": {
            "HEAD": {"index": [0, 1, 2], "x-weight": [12.0, 1.0, 1.0], "f-weight": [1.0, 1.0, 1.0]},
            "TAIL": {"index": [0, 1, 2], "x-weight": [12.0, 1.0, 1.0], "f-weight": [1.0, 1.0, 1.0]},
        },
        "system": [
            {
                "anchor": 0,
                "repeat": N_MOL,
                "offset": ATOMS_PER_MOL,
                "sites": [["HEAD", 0], ["TAIL", 3]],
            }
        ],
        "cg-topology": {
            "molecule": {
                "names": ["H", "T"],
                "charges": [0.0, 0.0],
                "bonds": [[0, 1]],
            }
        },
    }


@pytest.fixture(scope="module")
def aa_inputs(tmp_path_factory) -> dict:
    """Write a GRO topology plus a TRR with positions, forces, and velocities."""
    root = tmp_path_factory.mktemp("aa")
    universe = mda.Universe.empty(
        N_ATOMS,
        n_residues=N_MOL,
        atom_resindex=np.repeat(np.arange(N_MOL), ATOMS_PER_MOL),
        trajectory=True,
        velocities=True,
        forces=True,
    )
    universe.add_TopologyAttr("name", ["C", "H", "H", "C", "H", "H"] * N_MOL)
    universe.add_TopologyAttr("type", ["C", "H", "H", "C", "H", "H"] * N_MOL)
    universe.add_TopologyAttr("resname", ["LIP"] * N_MOL)
    universe.add_TopologyAttr("resid", list(range(1, N_MOL + 1)))
    universe.add_TopologyAttr("mass", [12.0, 1.0, 1.0, 12.0, 1.0, 1.0] * N_MOL)
    universe.atoms.positions = aa_positions(0)
    universe.dimensions = BOX
    gro_path = root / "aa.gro"
    universe.atoms.write(str(gro_path))

    trr_path = root / "aa.trr"
    ts = universe.trajectory.ts
    with mda.Writer(str(trr_path), n_atoms=N_ATOMS) as writer:
        for frame in range(N_FRAMES):
            ts.frame = frame
            universe.atoms.positions = aa_positions(frame)
            universe.atoms.velocities = np.full((N_ATOMS, 3), 0.1 * (frame + 1))
            universe.atoms.forces = np.full((N_ATOMS, 3), float(frame + 1))
            universe.dimensions = BOX
            ts.time = 10.0 * frame
            ts.data["step"] = 100 * frame
            writer.write(universe.atoms)

    map_path = root / "map.yaml"
    map_path.write_text(yaml.safe_dump(mapping_dict(), sort_keys=False))
    return {"root": root, "gro": gro_path, "trr": trr_path, "map": map_path}


def write_config(
    tmp_path: Path,
    aa_inputs: dict,
    *,
    aa: str = "",
    force_mapping: str = "",
    trajmap: str = "",
) -> Path:
    """Compose a trajmap ``.acg`` with per-test extra keys."""
    text = f"""
[aa]
topology = {aa_inputs['gro']}
trajectory_files = {aa_inputs['trr']}
trajectory_format = TRR
{aa}

[mapping]
map_file = {aa_inputs['map']}
resname = LIP

{force_mapping}

[trajmap]
output_dir = out
{trajmap}
"""
    path = tmp_path / "trajmap.acg"
    path.write_text(text)
    return path


def reference_positions(*, unwrap="molecule", wrap=True) -> np.ndarray:
    """CG positions straight from the kernel, bypassing the workflow."""
    spec = CGMapSpec.from_mapping(mapping_dict())
    mapper = CGMapper(spec, unwrap=unwrap, wrap=wrap, out_dtype=np.float32)
    return np.stack(
        [
            mapper.map_frame(aa_positions(frame), box=BOX).positions
            for frame in range(N_FRAMES)
        ]
    )


# ─── the happy path ───────────────────────────────────────────────────


def test_serial_run_writes_a_merged_trajectory_matching_the_kernel(tmp_path, aa_inputs):
    config = write_config(tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc")
    result = TrajMapWorkflow(parse_trajmap_file(config)).run()

    assert result.n_frames == N_FRAMES
    assert result.n_sites == 2 * N_MOL
    assert result.trajectory_path is not None and result.trajectory_path.is_file()
    assert result.segment_paths == ()

    want = reference_positions()
    got = mda.Universe(str(result.topology_path), str(result.trajectory_path), format="XTC")
    assert len(got.trajectory) == N_FRAMES
    for frame, ts in enumerate(got.trajectory):
        # XTC quantizes to 0.001 nm = 0.01 Å.
        assert ts.positions == pytest.approx(want[frame], abs=1e-2)
        assert ts.dimensions == pytest.approx(BOX, abs=1e-3)


def test_the_default_unwrap_actually_changes_the_result(tmp_path, aa_inputs):
    """Guards against the workflow silently ignoring the unwrap setting."""
    naive = reference_positions(unwrap="none")
    whole = reference_positions(unwrap="molecule")
    assert not np.allclose(naive, whole)

    config = write_config(
        tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc\nunwrap = none"
    )
    result = TrajMapWorkflow(parse_trajmap_file(config)).run()
    universe = mda.Universe(str(result.topology_path), str(result.trajectory_path), format="XTC")
    universe.trajectory[0]
    assert universe.atoms.positions == pytest.approx(naive[0], abs=1e-2)


def test_forces_and_velocities_land_in_the_trr(tmp_path, aa_inputs):
    config = write_config(
        tmp_path,
        aa_inputs,
        aa="include_forces = true\ninclude_velocities = true",
        trajmap="trajectory_name = cg.trr",
    )
    result = TrajMapWorkflow(parse_trajmap_file(config)).run()

    universe = mda.Universe(str(result.topology_path), str(result.trajectory_path), format="TRR")
    for frame, ts in enumerate(universe.trajectory):
        assert ts.has_forces and ts.has_velocities
        # f-weight is all ones over three atoms of equal force (frame + 1).
        assert ts.forces == pytest.approx(np.full((2 * N_MOL, 3), 3.0 * (frame + 1)), rel=1e-5)
        # Velocities use the x-weights, which average to the input value.
        assert ts.velocities == pytest.approx(
            np.full((2 * N_MOL, 3), 0.1 * (frame + 1)), rel=1e-4
        )
        assert ts.time == pytest.approx(10.0 * frame)


def test_optimal_force_mapping_is_fitted_saved_and_applied(tmp_path, aa_inputs):
    config = write_config(
        tmp_path,
        aa_inputs,
        aa="include_forces = true",
        force_mapping="""
[force_mapping]
method = optimal_linear
scope = per_template
backend = native
constraints = none
l2_regularization = 1e-8
artifact_name = fitted_force_map.npz
""",
        trajmap="trajectory_name = cg.trr",
    )
    result = TrajMapWorkflow(parse_trajmap_file(config)).run()
    assert result.force_map_path is not None and result.force_map_path.is_file()

    source = mda.Universe(str(aa_inputs["gro"]), str(aa_inputs["trr"]), format="TRR")
    mapped = mda.Universe(
        str(result.topology_path), str(result.trajectory_path), format="TRR"
    )
    assert len(source.trajectory) == N_FRAMES
    assert len(mapped.trajectory) == N_FRAMES
    coordinate = np.zeros((2, ATOMS_PER_MOL), dtype=float)
    coordinate[0, :3] = np.array([12., 1., 1.]) / 14.
    coordinate[1, 3:] = np.array([12., 1., 1.]) / 14.
    amplitude = np.arange(1., N_FRAMES + 1.)
    quadratic = 3. * N_MOL * np.sum(amplitude * amplitude) * np.ones((ATOMS_PER_MOL, ATOMS_PER_MOL))
    pmat = quadratic + 1.e-8 * np.eye(ATOMS_PER_MOL)
    operator = np.linalg.solve(coordinate @ np.linalg.inv(pmat) @ coordinate.T, coordinate @ np.linalg.inv(pmat))
    for source_ts, mapped_ts in zip(source.trajectory, mapped.trajectory):
        forces = np.asarray(source_ts.forces, dtype=np.float64).reshape(N_MOL, ATOMS_PER_MOL, 3)
        expected = np.concatenate([operator @ molecule for molecule in forces], axis=0)
        assert mapped_ts.forces == pytest.approx(expected, rel=2e-5, abs=2e-5)

    report = json.loads(result.report_path.read_text())
    assert report["force_mapping"]["method"] == "optimal_linear"
    assert report["force_mapping"]["fit_frames"]["count"] == N_FRAMES
    assert report["force_mapping"]["diagnostics"][0]["n_force_instances"] == N_MOL * N_FRAMES


def test_force_mapping_reuses_the_workflow_scan_plan(tmp_path, aa_inputs, monkeypatch):
    scan_calls = []
    original_scan = trajmap_workflow.MPITrajReader.scan
    original_accumulate = trajmap_workflow.accumulate_force_map_statistics

    def scan_once(reader, **kwargs):
        plan = original_scan(reader, **kwargs)
        scan_calls.append(plan)
        return plan

    def capture_scanned_reader(**kwargs):
        assert len(scan_calls) == 1
        assert kwargs["plan"] is scan_calls[0]
        assert kwargs["reader"].plan is kwargs["plan"]
        return original_accumulate(**kwargs)

    monkeypatch.setattr(trajmap_workflow.MPITrajReader, "scan", scan_once)
    monkeypatch.setattr(trajmap_workflow, "accumulate_force_map_statistics", capture_scanned_reader)
    config = write_config(
        tmp_path,
        aa_inputs,
        aa="include_forces = true",
        force_mapping="""
[force_mapping]
method = optimal_linear
scope = per_template
backend = native
constraints = none
""",
        trajmap="trajectory_name = cg.trr",
    )
    TrajMapWorkflow(parse_trajmap_file(config)).run()
    assert len(scan_calls) == 1


def test_cg_topology_carries_types_masses_and_bonds(tmp_path, aa_inputs):
    config = write_config(tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc")
    result = TrajMapWorkflow(parse_trajmap_file(config)).run()

    universe = mda.Universe(str(result.topology_path), topology_format="DATA")
    assert universe.atoms.n_atoms == 2 * N_MOL
    assert len(universe.bonds) == N_MOL  # one per repeat unit
    # x-weight sums to 14 for both site types.
    assert universe.atoms.masses == pytest.approx(np.full(2 * N_MOL, 14.0))
    assert sorted(set(universe.atoms.types)) == ["1", "2"]

    assert result.gro_path is not None and result.gro_path.is_file()
    gro = mda.Universe(str(result.gro_path))
    assert set(gro.atoms.resnames) == {"LIP"}
    assert list(gro.atoms.names) == ["H", "T"] * N_MOL


def test_cg_topology_canonical_types_reach_data_and_alias_outputs(tmp_path, aa_inputs):
    mapping = yaml.safe_load(Path(aa_inputs["map"]).read_text())
    molecule = mapping["cg-topology"]["molecule"]
    molecule["types"] = ["BEAD", "BEAD"]
    molecule["masses"] = [14.0, 14.0]
    molecule["charges"] = [0.0, 0.0]
    map_path = tmp_path / "canonical_types.yaml"
    map_path.write_text(yaml.safe_dump(mapping, sort_keys=False))
    inputs = dict(aa_inputs)
    inputs["map"] = map_path

    config = write_config(tmp_path, inputs, trajmap="trajectory_name = cg.xtc")
    result = TrajMapWorkflow(parse_trajmap_file(config)).run()

    universe = mda.Universe(str(result.topology_path), topology_format="DATA")
    assert sorted(set(universe.atoms.types)) == ["1"]
    assert universe.atoms.masses == pytest.approx(np.full(2 * N_MOL, 14.0))
    aliases = json.loads(result.aliases_path.read_text())
    assert aliases == {"1": "BEAD"}
    gro = mda.Universe(str(result.gro_path))
    assert list(gro.atoms.names) == ["H", "T"] * N_MOL


def test_report_records_the_selection_and_the_mapping(tmp_path, aa_inputs):
    config = write_config(
        tmp_path,
        aa_inputs,
        aa="skip_frames = 1\nevery = 2",
        trajmap="trajectory_name = cg.xtc",
    )
    result = TrajMapWorkflow(parse_trajmap_file(config)).run()
    report = json.loads(result.report_path.read_text())

    assert report["frames"]["total_in_trajectory"] == N_FRAMES
    assert report["frames"]["selected"] == 3  # frames 1, 3, 5
    assert report["frames"]["written"] == 3
    assert report["source"]["frame_ids"] == [1, 3, 5]
    assert report["source"]["segment_frame_counts"] == [N_FRAMES]
    assert report["mapping"]["n_sites"] == 2 * N_MOL
    assert report["mapping"]["n_molecules"] == N_MOL
    assert report["mapping"]["type_names"] == ["HEAD", "TAIL"]
    assert report["mapping"]["has_bonded_topology"] is True
    assert report["kernel"]["unwrap"] == "molecule"
    assert report["mpi"]["size"] == 1
    assert report["phase_seconds"]["total"] > 0.0

    universe = mda.Universe(str(result.topology_path), str(result.trajectory_path), format="XTC")
    assert len(universe.trajectory) == 3


def test_explicit_frame_ids_override_the_window(tmp_path, aa_inputs):
    config = write_config(
        tmp_path,
        aa_inputs,
        aa="frame_ids = [5, 0, 2, 2]\nevery = 3",
        trajmap="trajectory_name = cg.xtc",
    )
    result = TrajMapWorkflow(parse_trajmap_file(config)).run()
    want = reference_positions()

    universe = mda.Universe(str(result.topology_path), str(result.trajectory_path), format="XTC")
    assert len(universe.trajectory) == 4
    # Written in the order requested, not sorted.
    for out_frame, source_frame in enumerate([5, 0, 2, 2]):
        universe.trajectory[out_frame]
        assert universe.atoms.positions == pytest.approx(want[source_frame], abs=1e-2)
    report = json.loads(result.report_path.read_text())
    assert report["source"]["frame_ids"] == [5, 0, 2, 2]
    assert report["mpi"]["rank_slices"][0]["selected_offset"] == 0
    assert report["mpi"]["rank_slices"][0]["selected_count"] == 4


def test_unmerged_mode_keeps_segments_and_a_manifest(tmp_path, aa_inputs):
    config = write_config(
        tmp_path,
        aa_inputs,
        trajmap="trajectory_name = cg.xtc\nmerge_segments = false",
    )
    result = TrajMapWorkflow(parse_trajmap_file(config)).run()
    assert result.trajectory_path is None
    assert len(result.segment_paths) == 1
    assert result.segment_paths[0].is_file()
    report = json.loads(result.report_path.read_text())
    assert report["outputs"]["trajectory"] is None
    assert len(report["outputs"]["segments"]) == 1


def test_keep_segments_leaves_both_the_merge_and_the_parts(tmp_path, aa_inputs):
    config = write_config(
        tmp_path,
        aa_inputs,
        trajmap="trajectory_name = cg.xtc\nkeep_segments = true",
    )
    result = TrajMapWorkflow(parse_trajmap_file(config)).run()
    assert result.trajectory_path.is_file()
    assert len(result.segment_paths) == 1 and result.segment_paths[0].is_file()


# ─── collective failure closure ───────────────────────────────────────


def test_root_setup_failure_stops_before_trajectory_scan(tmp_path, aa_inputs, monkeypatch):
    """The shared root outcome must close before peers can enter ``scan``."""
    workflow = TrajMapWorkflow(
        parse_trajmap_file(write_config(tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc"))
    )

    monkeypatch.setattr(
        workflow,
        "_compile_spec",
        lambda *args, **kwargs: (_ for _ in ()).throw(LookupError("bad mapping setup")),
    )
    monkeypatch.setattr(
        trajmap_workflow.MPITrajReader,
        "scan",
        lambda *args, **kwargs: pytest.fail("scan must follow successful root setup"),
    )

    with pytest.raises(LookupError, match="bad mapping setup"):
        workflow.run()


def test_root_staging_failure_stops_before_local_mapping(tmp_path, aa_inputs, monkeypatch):
    """A terminal root stage closes before ranks enter its local frame stream."""
    workflow = TrajMapWorkflow(
        parse_trajmap_file(write_config(tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc"))
    )
    monkeypatch.setattr(
        trajmap_io,
        "_stream_local_segment",
        lambda **kwargs: pytest.fail("mapping must follow successful root writer stages"),
    )
    monkeypatch.setattr(
        trajmap_io,
        "_write_cg_topology",
        lambda **kwargs: (_ for _ in ()).throw(OSError("topology writer failed")),
    )
    with pytest.raises(OSError, match="topology writer failed"):
        workflow.run()


class _TrajMapConsensusComm:
    """Root-side fake communicator that exposes post-map collective ordering."""

    def __init__(self):
        self.rank_errors = []

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 2

    def bcast(self, value, root=0):
        return value

    def allgather(self, value):
        self.rank_errors.append(value)
        return [value, None]

    def gather(self, value, root=0):
        pytest.fail("stats gather must follow rank-error consensus")


def test_local_mapping_failure_reaches_consensus_before_stats_gather(
    tmp_path, aa_inputs, monkeypatch
):
    """A mapper/reader/writer failure is closed before the explicit stats gather."""
    comm = _TrajMapConsensusComm()
    workflow = TrajMapWorkflow(
        parse_trajmap_file(write_config(tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc")),
        comm=comm,
    )
    monkeypatch.setattr(
        trajmap_io,
        "_stream_local_segment",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("local map writer failed")),
    )

    with pytest.raises(RuntimeError, match="local map writer failed"):
        workflow.run()
    assert any(isinstance(error, RuntimeError) for error in comm.rank_errors)


def test_report_publication_failure_is_reported_before_successful_return(tmp_path, aa_inputs, monkeypatch):
    """The terminal closes report serialization failure before a successful return."""
    workflow = TrajMapWorkflow(
        parse_trajmap_file(write_config(tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc"))
    )

    real_dumps = trajmap_io.json.dumps

    def fail_report(payload, *args, **kwargs):
        if "mpi" in payload:
            raise OSError("report write failed")
        return real_dumps(payload, *args, **kwargs)

    monkeypatch.setattr(trajmap_io.json, "dumps", fail_report)
    with pytest.raises(OSError, match="report write failed"):
        workflow.run()


def test_staging_cleanup_failure_is_reported_before_successful_return(tmp_path, aa_inputs, monkeypatch):
    """The final root cleanup closes before the terminal reports success."""
    workflow = TrajMapWorkflow(
        parse_trajmap_file(write_config(tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc"))
    )
    real_rmtree = trajmap_io.shutil.rmtree

    def fail_staging_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".trajmap-stage-"):
            raise OSError("staging cleanup failed")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(trajmap_io.shutil, "rmtree", fail_staging_cleanup)
    with pytest.raises(OSError, match="staging cleanup failed"):
        workflow.run()


# ─── refusals ─────────────────────────────────────────────────────────


def test_rerunning_into_the_same_output_needs_overwrite(tmp_path, aa_inputs):
    config = write_config(tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc")
    TrajMapWorkflow(parse_trajmap_file(config)).run()
    with pytest.raises(FileExistsError, match="overwrite = true"):
        TrajMapWorkflow(parse_trajmap_file(config)).run()

    permitted = write_config(
        tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc\noverwrite = true"
    )
    assert TrajMapWorkflow(parse_trajmap_file(permitted)).run().n_frames == N_FRAMES


@pytest.mark.parametrize(
    ("target_name", "aa", "force_mapping", "trajmap"),
    [
        ("cg.xtc", "", "", "trajectory_name = cg.xtc"),
        ("cg.data", "", "", "trajectory_name = cg.xtc"),
        ("cg_aliases.json", "", "", "trajectory_name = cg.xtc"),
        ("cg.gro", "", "", "trajectory_name = cg.xtc"),
        ("trajmap_report.json", "", "", "trajectory_name = cg.xtc"),
        ("segments", "", "", "trajectory_name = cg.xtc\nmerge_segments = false"),
        (
            "fitted_force_map.npz",
            "include_forces = true",
            """
[force_mapping]
method = optimal_linear
scope = per_template
backend = native
constraints = none
artifact_name = fitted_force_map.npz
""",
            "trajectory_name = cg.trr",
        ),
    ],
)
def test_preflight_refuses_each_existing_final_target_before_writing(
    tmp_path, aa_inputs, target_name, aa, force_mapping, trajmap
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    target = output_dir / target_name
    if target_name == "segments":
        target.mkdir()
        (target / "sentinel").write_text("keep")
    else:
        target.write_text("keep")

    config = write_config(
        tmp_path, aa_inputs, aa=aa, force_mapping=force_mapping, trajmap=trajmap
    )
    with pytest.raises(FileExistsError, match="overwrite = true"):
        TrajMapWorkflow(parse_trajmap_file(config)).run()

    if target.is_dir():
        assert (target / "sentinel").read_text() == "keep"
    else:
        assert target.read_text() == "keep"
    assert not list(output_dir.glob(".trajmap-stage-*"))


@pytest.mark.parametrize(
    ("force_mapping", "trajmap"),
    [
        ("", "trajectory_name = cg.xtc\ntopology_name = cg.xtc\noverwrite = true"),
        ("", "trajectory_name = cg.xtc\ntopology_name = trajmap_report.json\noverwrite = true"),
        ("", "trajectory_name = cg.xtc\ntopology_name = cg.gro\noverwrite = true"),
    ],
)
def test_resolved_target_overlap_is_rejected_even_with_overwrite(
    tmp_path, aa_inputs, force_mapping, trajmap
):
    aa = "include_forces = true" if force_mapping else ""
    config = write_config(
        tmp_path, aa_inputs, aa=aa, force_mapping=force_mapping, trajmap=trajmap
    )
    with pytest.raises(ValueError, match="resolve to the same path"):
        TrajMapWorkflow(parse_trajmap_file(config)).run()
    assert not list((tmp_path / "out").glob(".trajmap-stage-*"))


@pytest.mark.parametrize(
    "trajmap",
    [
        "trajectory_name = ../escaped.xtc\noverwrite = true",
        "trajectory_name = /tmp/escaped.xtc\noverwrite = true",
    ],
)
def test_preflight_rejects_targets_outside_output_dir(tmp_path, aa_inputs, trajmap):
    config = write_config(tmp_path, aa_inputs, trajmap=trajmap)

    with pytest.raises(ValueError, match="must be inside"):
        TrajMapWorkflow(parse_trajmap_file(config)).run()

    assert not list((tmp_path / "out").glob(".trajmap-stage-*"))


@pytest.mark.parametrize(
    "trajmap",
    [
        "trajectory_name = segments/cg.xtc\nkeep_segments = true\noverwrite = true",
        "report_name = segments/trajmap_report.json\nmerge_segments = false\noverwrite = true",
    ],
)
def test_preflight_rejects_parent_child_final_targets(tmp_path, aa_inputs, trajmap):
    config = write_config(tmp_path, aa_inputs, trajmap=trajmap)

    with pytest.raises(ValueError, match="must not contain one another"):
        TrajMapWorkflow(parse_trajmap_file(config)).run()

    assert not list((tmp_path / "out").glob(".trajmap-stage-*"))


@pytest.mark.parametrize(
    ("trajmap", "result_path"),
    [
        ("trajectory_name = segments/cg.xtc", "trajectory_path"),
        ("trajectory_name = cg.xtc\ntopology_name = segments/cg.data", "topology_path"),
    ],
)
def test_merged_outputs_can_use_the_public_segments_subdirectory(
    tmp_path, aa_inputs, trajmap, result_path
):
    """Final nested outputs do not collide with private rank intermediates."""
    config = write_config(tmp_path, aa_inputs, trajmap=trajmap)

    result = TrajMapWorkflow(parse_trajmap_file(config)).run()

    published = getattr(result, result_path)
    assert published.is_file()
    assert published.parent == result.output_dir / "segments"
    assert not (result.output_dir / ".rank-segments").exists()
    assert not list(result.output_dir.glob(".trajmap-stage-*"))


def test_overwrite_replaces_only_trajmap_targets(tmp_path, aa_inputs):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    unrelated = output_dir / "unrelated.txt"
    unrelated.write_text("preserve me")
    config = write_config(
        tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc\noverwrite = true"
    )

    result = TrajMapWorkflow(parse_trajmap_file(config)).run()

    assert result.trajectory_path.is_file()
    assert unrelated.read_text() == "preserve me"
    assert not list(output_dir.glob(".trajmap-stage-*"))


def test_merge_failure_leaves_existing_trajectory_and_report_untouched(
    tmp_path, aa_inputs, monkeypatch
):
    config = write_config(
        tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc\noverwrite = true"
    )
    initial = TrajMapWorkflow(parse_trajmap_file(config)).run()
    trajectory_before = initial.trajectory_path.read_bytes()
    report_before = initial.report_path.read_bytes()
    monkeypatch.setattr(
        trajmap_io,
        "build_cg_universe",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("merge failed")),
    )

    with pytest.raises(OSError, match="merge failed"):
        TrajMapWorkflow(parse_trajmap_file(config)).run()

    assert initial.trajectory_path.read_bytes() == trajectory_before
    assert initial.report_path.read_bytes() == report_before
    assert not list(initial.output_dir.glob(".trajmap-stage-*"))


def test_report_publication_failure_leaves_no_completion_marker(
    tmp_path, aa_inputs, monkeypatch
):
    config = write_config(
        tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc\noverwrite = true"
    )
    initial = TrajMapWorkflow(parse_trajmap_file(config)).run()
    report_path = initial.report_path
    real_replace = Path.replace

    def fail_report_replace(path, target):
        if Path(target) == report_path:
            raise OSError("report publish failed")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_report_replace)
    with pytest.raises(OSError, match="report publish failed"):
        TrajMapWorkflow(parse_trajmap_file(config)).run()

    assert not report_path.exists()
    assert not list(initial.output_dir.glob(".trajmap-stage-*"))


def test_requesting_forces_from_a_force_free_trajectory_is_rejected(tmp_path, aa_inputs):
    xtc_path = tmp_path / "positions_only.xtc"
    bare = mda.Universe.empty(N_ATOMS, trajectory=True)
    bare.atoms.positions = aa_positions(0)
    bare.dimensions = BOX
    with mda.Writer(str(xtc_path), n_atoms=N_ATOMS) as writer:
        writer.write(bare.atoms)

    config = tmp_path / "noforce.acg"
    config.write_text(
        f"""
[aa]
topology = {aa_inputs['gro']}
trajectory_files = {xtc_path}
trajectory_format = XTC
include_forces = true

[mapping]
map_file = {aa_inputs['map']}

[trajmap]
output_dir = out_noforce
trajectory_name = cg.trr
"""
    )
    with pytest.raises(ValueError, match="carries no forces"):
        TrajMapWorkflow(parse_trajmap_file(config)).run()


def test_a_mapping_that_outruns_the_topology_is_rejected(tmp_path, aa_inputs):
    oversized = dict(mapping_dict())
    oversized["system"] = [
        {"anchor": 0, "repeat": N_MOL + 4, "offset": ATOMS_PER_MOL,
         "sites": [["HEAD", 0], ["TAIL", 3]]}
    ]
    map_path = tmp_path / "too_big.yaml"
    map_path.write_text(yaml.safe_dump(oversized, sort_keys=False))

    config = tmp_path / "toobig.acg"
    config.write_text(
        f"""
[aa]
topology = {aa_inputs['gro']}
trajectory_files = {aa_inputs['trr']}
trajectory_format = TRR

[mapping]
map_file = {map_path}

[trajmap]
output_dir = out_toobig
trajectory_name = cg.xtc
"""
    )
    with pytest.raises(ValueError, match="only 18 atoms"):
        TrajMapWorkflow(parse_trajmap_file(config)).run()


def test_an_empty_frame_selection_is_rejected(tmp_path, aa_inputs):
    config = write_config(
        tmp_path, aa_inputs, aa="skip_frames = 99", trajmap="trajectory_name = cg.xtc"
    )
    with pytest.raises(ValueError, match="frame selection is empty"):
        TrajMapWorkflow(parse_trajmap_file(config)).run()


def test_frame_ids_beyond_the_trajectory_are_rejected(tmp_path, aa_inputs):
    config = write_config(
        tmp_path, aa_inputs, aa="frame_ids = [0, 99]", trajmap="trajectory_name = cg.xtc"
    )
    with pytest.raises(ValueError, match="outside the trajectory"):
        TrajMapWorkflow(parse_trajmap_file(config)).run()


def test_unsupported_output_container_is_rejected_at_parse_time(tmp_path, aa_inputs):
    config = write_config(tmp_path, aa_inputs, trajmap="trajectory_name = cg.lammpstrj")
    with pytest.raises(ACGConfigError, match="trajectory_name must end in"):
        parse_trajmap_file(config)


@pytest.mark.parametrize(
    ("trajectory_path", "explicit_format", "uses_external_topology"),
    [
        ("aa.trr", None, False),
        ("aa.xtc", "trr", False),
        ("aa.pdb", None, True),
        ("aa.trr", "pdb", True),
    ],
)
def test_trajectory_reader_topology_uses_canonical_format_policy(
    trajectory_path, explicit_format, uses_external_topology
):
    workflow = object.__new__(TrajMapWorkflow)
    workflow.config = SimpleNamespace(
        aa=SimpleNamespace(trajectory_format=explicit_format)
    )
    topology = Path("aa.gro")

    selected = workflow._trajectory_reader_topology(topology, [trajectory_path])

    assert (selected is topology) is uses_external_topology


# ─── CLI ──────────────────────────────────────────────────────────────


def test_cli_runs_the_workflow_and_persists_its_result(tmp_path, aa_inputs):
    config = write_config(tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc")
    assert main([str(config), "--no-mpi"]) == 0

    output_dir = tmp_path / "out"
    assert (output_dir / "cg.xtc").is_file()
    assert (output_dir / "trajmap_report.json").is_file()
    assert (output_dir / "acgreturn.pkl").is_file()


def test_cli_overrides_reach_the_config(tmp_path, aa_inputs):
    config = write_config(tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc")
    assert main([str(config), "--no-mpi", "--run.unwrap", "bead", "--aa.n_frames", "2"]) == 0
    report = json.loads((tmp_path / "out" / "trajmap_report.json").read_text())
    assert report["kernel"]["unwrap"] == "bead"
    assert report["frames"]["selected"] == 2


# ─── topology-free operation ──────────────────────────────────────────


def test_the_aa_topology_can_be_omitted_for_an_xdr_trajectory(tmp_path, aa_inputs):
    """A TRR carries its own atom count, and the mapping carries its weights.

    This is what keeps a 1.2-million-atom .gro from being parsed once per rank.
    """
    config = tmp_path / "no_topology.acg"
    config.write_text(
        f"""
[aa]
trajectory_files = {aa_inputs['trr']}
trajectory_format = TRR

[mapping]
map_file = {aa_inputs['map']}

[trajmap]
output_dir = out_notopo
trajectory_name = cg.xtc
"""
    )
    result = TrajMapWorkflow(parse_trajmap_file(config)).run()
    assert result.n_frames == N_FRAMES

    want = reference_positions()
    universe = mda.Universe(str(result.topology_path), str(result.trajectory_path), format="XTC")
    for frame, ts in enumerate(universe.trajectory):
        assert ts.positions == pytest.approx(want[frame], abs=1e-2)
    report = json.loads(result.report_path.read_text())
    assert report["mapping"]["masses_from_topology"] is False


def test_omitting_the_topology_still_needs_x_weight_in_the_mapping(tmp_path, aa_inputs):
    massless = mapping_dict()
    massless["site-types"] = {
        "HEAD": {"index": [0, 1, 2]},
        "TAIL": {"index": [0, 1, 2]},
    }
    map_path = tmp_path / "no_weights.yaml"
    map_path.write_text(yaml.safe_dump(massless, sort_keys=False))

    config = tmp_path / "needs_masses.acg"
    config.write_text(
        f"""
[aa]
trajectory_files = {aa_inputs['trr']}
trajectory_format = TRR

[mapping]
map_file = {map_path}

[trajmap]
output_dir = out_needs_masses
trajectory_name = cg.xtc
"""
    )
    with pytest.raises(ValueError, match="no `masses` array was supplied"):
        TrajMapWorkflow(parse_trajmap_file(config)).run()


def test_a_merge_leaves_no_segment_directory_behind(tmp_path, aa_inputs):
    """Reading a segment makes MDAnalysis drop a hidden offsets sidecar next to it."""
    config = write_config(
        tmp_path, aa_inputs, trajmap="trajectory_name = cg.trr\noverwrite = true"
    )
    result = TrajMapWorkflow(parse_trajmap_file(config)).run()
    assert result.trajectory_path.is_file()
    assert not (result.output_dir / "segments").exists()


def test_a_rerun_with_fewer_ranks_does_not_inherit_stale_segments(tmp_path, aa_inputs):
    """Stale segments would otherwise be merged in as extra frames."""
    config = write_config(
        tmp_path,
        aa_inputs,
        trajmap="trajectory_name = cg.xtc\nmerge_segments = false\noverwrite = true",
    )
    result = TrajMapWorkflow(parse_trajmap_file(config)).run()
    segment_dir = result.segment_paths[0].parent
    impostor = segment_dir / "segment_0007.xtc"
    impostor.write_bytes(result.segment_paths[0].read_bytes())
    assert len(list(segment_dir.glob("segment_*.xtc"))) == 2

    again = TrajMapWorkflow(parse_trajmap_file(config)).run()
    assert not impostor.exists()
    assert len(again.segment_paths) == 1
    assert json.loads(again.report_path.read_text())["frames"]["written"] == N_FRAMES


def test_a_type_alias_file_is_written_next_to_the_lammps_data(tmp_path, aa_inputs):
    """LAMMPS data can only number its types; AceCG configs want the names."""
    config = write_config(tmp_path, aa_inputs, trajmap="trajectory_name = cg.xtc")
    result = TrajMapWorkflow(parse_trajmap_file(config)).run()

    assert result.aliases_path is not None and result.aliases_path.is_file()
    assert result.aliases_path.name == "cg_aliases.json"
    assert json.loads(result.aliases_path.read_text()) == {"1": "HEAD", "2": "TAIL"}
    report = json.loads(result.report_path.read_text())
    assert report["outputs"]["type_aliases"] == str(result.aliases_path)
