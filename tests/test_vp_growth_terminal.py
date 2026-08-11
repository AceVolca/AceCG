"""Characterization of the concrete VP trajectory-growth terminal."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from AceCG.configs.vp_config import VPAtomDef, VPConfig, VPInteractionDef
from AceCG.configs.vp_growth_config import VPGrowthAARef, VPGrowthConfig, VPGrowthRun
from AceCG.io import vp_growth as vp_growth_io
from AceCG.io.trajectory import FrameRecord, TrajPlan, partition_frame_ids
from AceCG.io.vp_growth import grow_vp_trajectory
from AceCG.topology.vpgrower import VPGrower, write_vp_data


def _topology_universe() -> mda.Universe:
    atom_resindex = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    universe = mda.Universe.empty(
        8, n_residues=2, atom_resindex=atom_resindex, trajectory=True
    )
    types = np.array(["1", "2", "3", "4"] * 2, dtype=object)
    universe.add_TopologyAttr("types", types)
    universe.add_TopologyAttr("names", types.copy())
    universe.add_TopologyAttr("masses", np.full(8, 72.0))
    universe.add_TopologyAttr("charges", np.zeros(8))
    universe.add_TopologyAttr("resids", np.array([1, 2], dtype=np.int64))
    universe.add_TopologyAttr(
        "resnames", np.array(["DOPC", "DOPC"], dtype=object)
    )
    universe.add_TopologyAttr(
        "bonds",
        np.array(
            [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)],
            dtype=np.int64,
        ),
    )
    universe.add_TopologyAttr(
        "angles", np.array([(0, 1, 2), (4, 5, 6)], dtype=np.int64)
    )
    universe.dimensions = [30.0, 30.0, 30.0, 90.0, 90.0, 90.0]
    return universe


def _vp_config() -> VPConfig:
    return VPConfig(
        atoms=(VPAtomDef(type_label="VP", mass=72.0),),
        bonds=(
            VPInteractionDef(
                type_keys=("VP", "MG"),
                pot_style="harmonic",
                pot_kwargs={"k": 2.5, "r0": 1.5},
            ),
        ),
        angles=(
            VPInteractionDef(
                type_keys=("VP", "MG", "HG"),
                pot_style="harmonic",
                pot_kwargs={"k": 2.45, "theta0": 135.0},
            ),
        ),
        pairs=(
            VPInteractionDef(
                type_keys=("VP", "HG"),
                pot_style="table",
                pot_kwargs={"file": "Pair_VP-HG.table", "cutoff": 10.0},
            ),
        ),
        selection="resname DOPC",
        atomtype_order="back",
        clash_max_passes=8,
        clash_min_distance=1.5,
    )


def _config(tmp_path: Path, *, include_forces: bool, overwrite: bool = False):
    return VPGrowthConfig(
        path=tmp_path / "run.acg",
        aa_ref=VPGrowthAARef(
            trajectory_files=("source.lammpstrj",),
            trajectory_format="LAMMPSDUMP",
            ref_topo="source.data",
            ref_topo_type_names={1: "HG", 2: "MG", 3: "T1", 4: "T2"},
            include_forces=include_forces,
        ),
        vp=_vp_config(),
        run=VPGrowthRun(
            output_dir="unused",
            frame_ids=(5, 0, 2, 2),
            orientation_seed_base=100,
            table_points=31,
            table_rmin=0.01,
            table_rmax=10.0,
            overwrite=overwrite,
        ),
    )


class _Reader:
    def __init__(self, frame_ids, *, has_forces, missing_force_id=None):
        self.trajectory_files = ("source.lammpstrj",)
        self.trajectory_format = "LAMMPSDUMP"
        self.strategy = "reopen"
        self.frame_ids = tuple(frame_ids)
        self.has_forces = has_forces
        self.missing_force_id = missing_force_id
        self.plan = None
        self.opened_segment_numbers = (1,)

    def scan(self, **kwargs):
        assert kwargs["frame_ids"] == self.frame_ids
        self.plan = TrajPlan(
            total_frames=6,
            frame_ids=self.frame_ids,
            segment_frame_counts=(6,),
            has_forces=self.has_forces,
        )
        return self.plan

    def local_slice(self, rank=None):
        return partition_frame_ids(
            self.frame_ids, size=1, rank=0 if rank is None else int(rank)
        )

    def iter_local(self, **kwargs):
        assert kwargs["include_forces"] is bool(kwargs["include_forces"])
        base = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [0.0, 6.0, 0.0],
                [2.0, 6.0, 0.0],
                [4.0, 6.0, 0.0],
                [6.0, 6.0, 0.0],
            ],
            dtype=np.float32,
        )
        for frame_id in self.frame_ids:
            forces = np.full((8, 3), frame_id + 0.25, dtype=np.float32)
            yield FrameRecord(
                frame_id=int(frame_id),
                positions=base + frame_id * 0.1,
                box=np.array([30.0, 30.0, 30.0, 90.0, 90.0, 90.0]),
                forces=(
                    None
                    if frame_id == self.missing_force_id
                    else forces.reshape(-1)
                ),
            )


@pytest.fixture(autouse=True)
def _static_topology(monkeypatch):
    monkeypatch.setattr(vp_growth_io, "open_universe", lambda *args, **kwargs: _topology_universe())


def test_ordered_duplicates_share_one_physical_pair_and_seed(tmp_path):
    config = _config(tmp_path, include_forces=True)
    output_dir = tmp_path / "out"
    result = grow_vp_trajectory(
        config=config,
        reader=_Reader(config.run.frame_ids, has_forces=True),
        output_dir=output_dir,
        reference_topology=tmp_path / "source.data",
        comm=None,
    )

    manifest = json.loads(result["manifest_path"].read_text())
    records = manifest["occurrences"]
    assert [record["source_frame_id"] for record in records] == [5, 0, 2, 2]
    assert [record["selection_index"] for record in records] == [0, 1, 2, 3]
    assert [record["orientation_seed"] for record in records] == [105, 100, 102, 102]
    assert records[2]["data"] == records[3]["data"]
    assert records[2]["forces"] == records[3]["forces"]
    assert result["n_selected"] == 4
    assert result["n_unique"] == 3
    assert len(list(output_dir.glob("frame_*.data"))) == 3
    assert len(list(output_dir.glob("frame_*.forces.npy"))) == 3
    forces = np.load(output_dir / "frame_000002.forces.npy")
    assert forces.dtype == np.float32
    assert forces.shape == (8, 3)


def test_unique_outputs_are_deterministic_and_unrelated_files_survive_overwrite(
    tmp_path
):
    config = _config(tmp_path, include_forces=True)
    first = tmp_path / "first"
    second = tmp_path / "second"
    grow_vp_trajectory(
        config=config,
        reader=_Reader(config.run.frame_ids, has_forces=True),
        output_dir=first,
        reference_topology=tmp_path / "source.data",
        comm=None,
    )
    unrelated = first / "keep.me"
    unrelated.write_text("unrelated")
    overwrite_config = replace(config, run=replace(config.run, overwrite=True))
    grow_vp_trajectory(
        config=overwrite_config,
        reader=_Reader(config.run.frame_ids, has_forces=True),
        output_dir=first,
        reference_topology=tmp_path / "source.data",
        comm=None,
    )
    grow_vp_trajectory(
        config=config,
        reader=_Reader(config.run.frame_ids, has_forces=True),
        output_dir=second,
        reference_topology=tmp_path / "source.data",
        comm=None,
    )

    assert unrelated.read_text() == "unrelated"
    for frame_id in (5, 0, 2):
        assert (first / f"frame_{frame_id:06d}.data").read_bytes() == (
            second / f"frame_{frame_id:06d}.data"
        ).read_bytes()
        assert (first / f"frame_{frame_id:06d}.forces.npy").read_bytes() == (
            second / f"frame_{frame_id:06d}.forces.npy"
        ).read_bytes()


def test_unique_data_and_force_bytes_match_direct_scientific_baseline(tmp_path):
    config = _config(tmp_path, include_forces=True)
    config = replace(config, run=replace(config.run, frame_ids=(2,)))
    output_dir = tmp_path / "terminal"
    grow_vp_trajectory(
        config=config,
        reader=_Reader(config.run.frame_ids, has_forces=True),
        output_dir=output_dir,
        reference_topology=tmp_path / "source.data",
        comm=None,
    )

    grower = VPGrower.from_universe(
        _topology_universe(),
        config.vp,
        type_aliases=config.aa_ref.ref_topo_type_names,
    )
    frame = next(_Reader((2,), has_forces=True).iter_local(include_forces=True))
    baseline_data = tmp_path / "baseline.data"
    write_vp_data(
        grower.template,
        grower.grow_frame(
            frame.positions,
            frame.box,
            orientation_seed=config.run.orientation_seed_base + 2,
        ),
        baseline_data,
    )
    baseline_force = tmp_path / "baseline.forces.npy"
    np.save(
        baseline_force,
        np.asarray(frame.forces, dtype=np.float32).reshape(grower.template.n_real, 3),
    )

    assert (output_dir / "frame_000002.data").read_bytes() == baseline_data.read_bytes()
    assert (output_dir / "frame_000002.forces.npy").read_bytes() == baseline_force.read_bytes()


def test_known_missing_requested_forces_fail_before_staging(tmp_path):
    config = _config(tmp_path, include_forces=True)
    output_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="carries no forces"):
        grow_vp_trajectory(
            config=config,
            reader=_Reader(config.run.frame_ids, has_forces=False),
            output_dir=output_dir,
            reference_topology=tmp_path / "source.data",
            comm=None,
        )
    assert not output_dir.exists()


def test_unknown_missing_requested_forces_clean_staging_and_manifest(tmp_path):
    config = _config(tmp_path, include_forces=True)
    output_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="frame 2 is missing requested forces"):
        grow_vp_trajectory(
            config=config,
            reader=_Reader(
                config.run.frame_ids, has_forces=None, missing_force_id=2
            ),
            output_dir=output_dir,
            reference_topology=tmp_path / "source.data",
            comm=None,
        )
    assert not (output_dir / "manifest.json").exists()
    assert not (output_dir / ".vp-growth-stage").exists()


def test_local_writer_failure_cleans_staging_before_return(tmp_path, monkeypatch):
    config = _config(tmp_path, include_forces=False)
    output_dir = tmp_path / "out"
    original = vp_growth_io.write_vp_data

    def fail_frame_write(*args, **kwargs):
        if kwargs.get("title") == "VP topology (schema)":
            return original(*args, **kwargs)
        raise OSError("injected VP frame writer failure")

    monkeypatch.setattr(vp_growth_io, "write_vp_data", fail_frame_write)
    with pytest.raises(OSError, match="injected VP frame writer failure"):
        grow_vp_trajectory(
            config=config,
            reader=_Reader(config.run.frame_ids, has_forces=None),
            output_dir=output_dir,
            reference_topology=tmp_path / "source.data",
            comm=None,
        )
    assert not (output_dir / "manifest.json").exists()
    assert not (output_dir / ".vp-growth-stage").exists()


def test_root_rejects_corrupt_gathered_occurrence_before_publication(tmp_path):
    class CorruptGather:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def gather(self, value, root=0):
            value["records"][0]["selection_index"] = 99
            return [value]

    config = _config(tmp_path, include_forces=False)
    output_dir = tmp_path / "corrupt"
    with pytest.raises(RuntimeError, match="do not match the selected plan"):
        grow_vp_trajectory(
            config=config,
            reader=_Reader(config.run.frame_ids, has_forces=None),
            output_dir=output_dir,
            reference_topology=tmp_path / "source.data",
            comm=CorruptGather(),
        )
    assert not (output_dir / "manifest.json").exists()
    assert not (output_dir / ".vp-growth-stage").exists()


@pytest.mark.parametrize(
    "collision",
    (
        "vp_topology.data",
        "latent.settings",
        "Pair_VP-HG.table",
        "frame_000005.data",
        "frame_000005.forces.npy",
        "timing.json",
        "manifest.json",
    ),
)
def test_every_output_category_collides_before_writers(
    tmp_path, monkeypatch, collision
):
    config = _config(tmp_path, include_forces=True)
    output_dir = tmp_path / collision.replace(".", "-")
    output_dir.mkdir()
    (output_dir / collision).write_text("old")
    monkeypatch.setattr(
        vp_growth_io,
        "write_latent_settings",
        lambda **kwargs: pytest.fail("preflight must precede writers"),
    )
    with pytest.raises(FileExistsError, match=collision.replace(".", r"\.")):
        grow_vp_trajectory(
            config=config,
            reader=_Reader(config.run.frame_ids, has_forces=True),
            output_dir=output_dir,
            reference_topology=tmp_path / "source.data",
            comm=None,
        )


def test_parent_child_overlap_fails_before_writers(tmp_path, monkeypatch):
    config = _config(tmp_path, include_forces=False)
    monkeypatch.setattr(
        vp_growth_io,
        "write_latent_settings",
        lambda **kwargs: pytest.fail("preflight must precede writers"),
    )

    overlap_config = replace(
        config,
        run=replace(config.run, latent_settings_name="vp_topology.data/latent"),
    )
    with pytest.raises(ValueError, match="must not contain one another"):
        grow_vp_trajectory(
            config=overlap_config,
            reader=_Reader(config.run.frame_ids, has_forces=None),
            output_dir=tmp_path / "overlap",
            reference_topology=tmp_path / "source.data",
            comm=None,
        )


def test_target_containment_and_staging_conflict_are_preflighted(tmp_path):
    config = _config(tmp_path, include_forces=False)
    outside = replace(
        config,
        run=replace(config.run, latent_settings_name="../outside.settings"),
    )
    with pytest.raises(ValueError, match="must be inside"):
        grow_vp_trajectory(
            config=outside,
            reader=_Reader(config.run.frame_ids, has_forces=None),
            output_dir=tmp_path / "contained",
            reference_topology=tmp_path / "source.data",
            comm=None,
        )

    output_dir = tmp_path / "staging-conflict"
    (output_dir / ".vp-growth-stage").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="staging path already exists"):
        grow_vp_trajectory(
            config=config,
            reader=_Reader(config.run.frame_ids, has_forces=None),
            output_dir=output_dir,
            reference_topology=tmp_path / "source.data",
            comm=None,
        )
    assert (output_dir / ".vp-growth-stage").is_dir()


def test_manifest_is_published_last_and_failure_leaves_no_marker(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, include_forces=False)
    output_dir = tmp_path / "out"
    original_replace = Path.replace
    publications = []

    def recording_replace(source, target):
        if source.parent.name == ".vp-growth-stage":
            publications.append(Path(target).name)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", recording_replace)
    grow_vp_trajectory(
        config=config,
        reader=_Reader(config.run.frame_ids, has_forces=None),
        output_dir=output_dir,
        reference_topology=tmp_path / "source.data",
        comm=None,
    )
    assert publications[-1] == "manifest.json"

    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    (failed_dir / "manifest.json").write_text("old completion marker")
    overwrite_config = replace(config, run=replace(config.run, overwrite=True))

    def failing_replace(source, target):
        if Path(target).name == "timing.json":
            raise OSError("injected publication failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(OSError, match="injected publication failure"):
        grow_vp_trajectory(
            config=overwrite_config,
            reader=_Reader(config.run.frame_ids, has_forces=None),
            output_dir=failed_dir,
            reference_topology=tmp_path / "source.data",
            comm=None,
        )
    assert not (failed_dir / "manifest.json").exists()
    assert not (failed_dir / ".vp-growth-stage").exists()


def test_final_staging_cleanup_failure_is_not_reported_as_success(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, include_forces=False)
    output_dir = tmp_path / "cleanup-failure"
    original_rmtree = vp_growth_io.shutil.rmtree

    def fail_staging_cleanup(path, *args, **kwargs):
        if Path(path).name == ".vp-growth-stage":
            raise OSError("injected final staging cleanup failure")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(vp_growth_io.shutil, "rmtree", fail_staging_cleanup)
    with pytest.raises(OSError, match="injected final staging cleanup failure"):
        grow_vp_trajectory(
            config=config,
            reader=_Reader(config.run.frame_ids, has_forces=None),
            output_dir=output_dir,
            reference_topology=tmp_path / "source.data",
            comm=None,
        )
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / ".vp-growth-stage").is_dir()

    original_replace = Path.replace

    def fail_publication(source, target):
        if Path(target).name == "timing.json":
            raise OSError("earlier publication failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_publication)
    earlier_dir = tmp_path / "earlier-error"
    with pytest.raises(OSError, match="earlier publication failure"):
        grow_vp_trajectory(
            config=config,
            reader=_Reader(config.run.frame_ids, has_forces=None),
            output_dir=earlier_dir,
            reference_topology=tmp_path / "source.data",
            comm=None,
        )
    assert not (earlier_dir / "manifest.json").exists()
    assert (earlier_dir / ".vp-growth-stage").is_dir()
