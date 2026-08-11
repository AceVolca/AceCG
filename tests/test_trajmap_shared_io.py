"""Contract tests for the shared frame record and MPI trajectory reader."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

import AceCG.io.trajectory as trajectory_io
from AceCG.io.trajectory import (
    FRAME_FIELDS,
    FrameRecord,
    MPITrajReader,
    TrajPlan,
    broadcast_root_outcome,
    format_for_path,
    iter_frames,
    offsets_are_installable,
    open_universe,
    raise_if_rank_failed,
    reference_force_scale_to_lammps_real,
    reader_offsets,
)


def test_frame_record_has_fixed_keys_and_none_for_unrequested_fields():
    universe = mda.Universe.empty(
        2,
        trajectory=True,
        velocities=True,
        forces=True,
    )
    universe.atoms.positions = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    universe.atoms.velocities = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    universe.atoms.forces = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    universe.dimensions = [20.0, 21.0, 22.0, 90.0, 90.0, 90.0]
    universe.trajectory.ts.data["step"] = 123

    plain = next(iter_frames(universe))
    assert isinstance(plain, FrameRecord)
    assert tuple(plain) == FRAME_FIELDS
    assert tuple(dict(plain)) == FRAME_FIELDS
    assert plain["positions"].shape == (2, 3)
    assert plain["box"].shape == (6,)
    for name in ("forces", "velocities", "time", "dt", "step"):
        assert plain[name] is None
    with pytest.raises(KeyError, match="carries no"):
        plain.require("forces")
    with pytest.raises(FrozenInstanceError):
        plain.step = 456

    complete = next(
        iter_frames(
            universe,
            include_forces=True,
            include_velocities=True,
            include_time=True,
        )
    )
    forces, velocities, step = complete.require("forces", "velocities", "step")
    np.testing.assert_allclose(forces.reshape(2, 3), universe.atoms.forces)
    np.testing.assert_allclose(velocities.reshape(2, 3), universe.atoms.velocities)
    assert step == 123
    assert complete.time is not None
    assert complete.dt is not None


class _RankOneOfTwo:
    def Get_rank(self):
        return 1

    def Get_size(self):
        return 2

    def allgather(self, value):
        return [None, None]


def _reader(tmp_path, n_segments, suffix, *, comm=_RankOneOfTwo(), **kwargs):
    """A reader over ``n_segments`` unopened paths — strategy needs no real files."""
    return MPITrajReader(
        trajectory_files=[tmp_path / f"seg{i}{suffix}" for i in range(n_segments)],
        comm=comm,
        **kwargs,
    )


def test_a_topology_less_open_keeps_every_segment(tmp_path):
    """`Universe(*paths)` would eat paths[0] as the topology and lose its frames."""
    first = tmp_path / "first.xtc"
    second = tmp_path / "second.xtc"
    _write_one_atom_xtc(first, [10.0, 11.0])
    _write_one_atom_xtc(second, [20.0, 21.0])

    universe = open_universe(None, [first, second])
    assert len(universe.trajectory) == 4
    np.testing.assert_allclose(
        [float(ts.positions[0, 0]) for ts in universe.trajectory],
        [10.0, 11.0, 20.0, 21.0],
        atol=1e-3,
    )


def test_format_for_path_canonicalizes_explicit_and_known_suffixes():
    assert format_for_path("md.trr") == "TRR"
    assert format_for_path("md.xtc") == "XTC"
    assert format_for_path("cg.lammpstrj") == "LAMMPSDUMP"
    assert format_for_path("cg.dump") == "LAMMPSDUMP"
    assert format_for_path("system.data") == "DATA"
    assert format_for_path("md.dcd") == "DCD"
    assert format_for_path("md.h5md") == "H5MD"
    assert format_for_path("points.xyz") == "XYZ"
    assert format_for_path("system.pdb") is None
    assert format_for_path("unknown.bin") is None
    assert format_for_path("conflicting.xtc", explicit_format=" trr ") == "TRR"
    for alias in ("lammpstrj", "lammpsdump", "lammps_dump", "dump"):
        assert format_for_path("conflicting.trr", explicit_format=alias) == "LAMMPSDUMP"
    for token in ("xtc", "dcd", "h5md", "trr", "xyz"):
        assert format_for_path("unknown.bin", explicit_format=token) == token.upper()
    assert format_for_path("md.trr", explicit_format="") == "TRR"
    assert format_for_path("md.trr", explicit_format=" auto ") == "TRR"
    assert format_for_path(None, explicit_format="auto") is None
    assert format_for_path("md.trr", explicit_format="custom-format") == "CUSTOM_FORMAT"


def test_mpi_traj_reader_stores_one_canonical_format():
    inferred = MPITrajReader(trajectory_files=["md.trr"])
    explicit = MPITrajReader(
        trajectory_files=["md.xtc"], trajectory_format="trr"
    )
    assert inferred.trajectory_format == "TRR"
    assert explicit.trajectory_format == inferred.trajectory_format


def test_offsets_are_installable_only_for_the_xdr_family():
    assert offsets_are_installable("XTC")
    assert offsets_are_installable("TRR")
    assert offsets_are_installable(None, "md.xtc")
    assert not offsets_are_installable("LAMMPSDUMP")
    assert not offsets_are_installable(None, "cg6.lammpstrj")
    assert not offsets_are_installable(None, None)


def test_only_trr_reference_forces_need_kj_to_kcal_conversion():
    assert reference_force_scale_to_lammps_real("TRR") == 1.0 / 4.184
    assert reference_force_scale_to_lammps_real("trr") == 1.0 / 4.184
    assert reference_force_scale_to_lammps_real("LAMMPSDUMP") == 1.0
    assert reference_force_scale_to_lammps_real(None) == 1.0


def test_auto_strategy_is_format_and_segment_count_aware(tmp_path):
    # XDR: broadcasting a Universe loses the offsets, so reopen and install them;
    # past the segment limit, open only the local segments instead.
    assert _reader(tmp_path, 1, ".xtc").strategy == "reopen"
    assert _reader(tmp_path, 2, ".trr").strategy == "reopen"
    assert _reader(tmp_path, 3, ".xtc").strategy == "local_segments"
    assert _reader(tmp_path, 50, ".xtc").strategy == "local_segments"

    # LAMMPSDUMP: its offsets are plain Python state and do travel with a pickle,
    # so one text scan on rank 0 can serve every rank.
    assert _reader(tmp_path, 1, ".lammpstrj").strategy == "broadcast"
    assert _reader(tmp_path, 2, ".lammpstrj").strategy == "broadcast"
    assert _reader(tmp_path, 3, ".lammpstrj").strategy == "local_segments"

    # Serial has no peer to broadcast to and owns every segment.
    assert _reader(tmp_path, 50, ".lammpstrj", comm=None).strategy == "reopen"

    # An explicit strategy always wins.
    assert (
        _reader(tmp_path, 50, ".xtc", strategy="broadcast").strategy == "broadcast"
    )


def test_many_segment_lammpstrj_scan_uses_reader_owned_cheap_counts(
    tmp_path, monkeypatch
):
    class RootOfFour:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 4

        def bcast(self, value, root=0):
            return value

    paths = [tmp_path / f"segment_{index}.lammpstrj" for index in range(3)]
    reader = MPITrajReader(
        trajectory_files=paths,
        comm=RootOfFour(),
        strategy="auto",
        broadcast_segment_limit=2,
    )
    reader.open_full = lambda: pytest.fail(
        "many-segment LAMMPSDUMP scan must not open the full chain"
    )
    calls = []

    def cheap_count(path):
        calls.append(Path(path))
        return (2, 8)

    monkeypatch.setattr(
        trajectory_io, "count_lammpstrj_frames_and_atoms", cheap_count
    )
    plan = reader.scan(frame_ids=[5, 0, 2, 2])

    assert calls == paths
    assert plan.segment_frame_counts == (2, 2, 2)
    assert plan.frame_ids == (5, 0, 2, 2)
    assert plan.offsets is None
    assert plan.has_forces is None


def _write_one_atom_xtc(path, values):
    universe = mda.Universe.empty(1, trajectory=True)
    universe.add_TopologyAttr("name", ["CA"])
    universe.add_TopologyAttr("type", ["CA"])
    universe.add_TopologyAttr("resid", [1])
    universe.add_TopologyAttr("resname", ["ALA"])
    with mda.Writer(str(path), n_atoms=1) as writer:
        for value in values:
            universe.atoms.positions = [[float(value), 0.0, 0.0]]
            universe.dimensions = [30.0, 30.0, 30.0, 90.0, 90.0, 90.0]
            writer.write(universe.atoms)


def _write_one_atom_trr(path, values):
    universe = mda.Universe.empty(
        1,
        trajectory=True,
        velocities=True,
        forces=True,
    )
    universe.add_TopologyAttr("name", ["CA"])
    universe.add_TopologyAttr("type", ["CA"])
    universe.add_TopologyAttr("resid", [1])
    universe.add_TopologyAttr("resname", ["ALA"])
    timestep = universe.trajectory.ts
    with mda.Writer(str(path), n_atoms=1) as writer:
        for index, value in enumerate(values):
            timestep.frame = index
            timestep.time = float(value) * 2.0
            universe.atoms.positions = [[float(value), 0.0, 0.0]]
            universe.atoms.velocities = [[float(value) + 0.5, 0.0, 0.0]]
            universe.atoms.forces = [[float(value) * 10.0, 0.0, 0.0]]
            universe.dimensions = [30.0, 30.0, 30.0, 90.0, 90.0, 90.0]
            writer.write(universe.atoms)


def test_local_segment_reader_reindexes_global_ids_when_iterating(tmp_path):
    topology = tmp_path / "topology.pdb"
    first = tmp_path / "first.xtc"
    second = tmp_path / "second.xtc"

    template = mda.Universe.empty(1, trajectory=True)
    template.add_TopologyAttr("name", ["CA"])
    template.add_TopologyAttr("type", ["CA"])
    template.add_TopologyAttr("resid", [1])
    template.add_TopologyAttr("resname", ["ALA"])
    template.atoms.positions = [[0.0, 0.0, 0.0]]
    with mda.Writer(str(topology), n_atoms=1) as writer:
        writer.write(template.atoms)
    _write_one_atom_xtc(first, [10.0, 11.0])
    _write_one_atom_xtc(second, [20.0, 21.0])

    reader = MPITrajReader(
        topology=topology,
        trajectory_files=[first, second],
        comm=_RankOneOfTwo(),
        strategy="local_segments",
    )
    reader.plan = TrajPlan(
        total_frames=4,
        frame_ids=(0, 1, 2, 3),
        segment_frame_counts=(2, 2),
    )

    frames = list(reader.iter_local())
    assert [frame.frame_id for frame in frames] == [2, 3]
    np.testing.assert_allclose(
        [frame.positions[0, 0] for frame in frames],
        [20.0, 21.0],
        atol=1e-3,
    )


def test_local_segments_installs_only_its_own_segments_offsets(tmp_path, monkeypatch):
    """The scan's offsets must reach the local segments, not be thrown away."""
    first = tmp_path / "first.xtc"
    second = tmp_path / "second.xtc"
    _write_one_atom_xtc(first, [10.0, 11.0])
    _write_one_atom_xtc(second, [20.0, 21.0])

    reader = MPITrajReader(
        trajectory_files=[first, second],
        comm=_RankOneOfTwo(),
        strategy="local_segments",
    )
    full = reader.open_full()
    scanned = reader_offsets(full)
    assert scanned is not None and len(scanned) == 2
    reader.plan = TrajPlan(
        total_frames=4,
        frame_ids=(0, 1, 2, 3),
        segment_frame_counts=(2, 2),
        offsets=tuple(scanned),
    )

    # Rank 1 owns frames 2-3, which live in the second segment alone.
    assert reader._offsets_for_segments([2]) == [scanned[1]]
    captured = {}
    real_open = trajectory_io.open_universe

    def capture_open(*args, **kwargs):
        universe = real_open(*args, **kwargs)
        captured["universe"] = universe
        return universe

    monkeypatch.setattr(trajectory_io, "open_universe", capture_open)
    assert [frame.frame_id for frame in reader.iter_local()] == [2, 3]
    installed = reader_offsets(captured["universe"])
    assert installed is not None and len(installed) == 1
    np.testing.assert_array_equal(installed[0], scanned[1])
    assert reader.opened_segment_numbers == (2,)


def test_serial_reader_yields_global_ids_in_explicit_order(tmp_path):
    path = tmp_path / "serial.xtc"
    _write_one_atom_xtc(path, [10.0, 11.0, 12.0])

    reader = MPITrajReader(trajectory_files=[path])
    reader.scan(frame_ids=[2, 0, 2])
    frames = list(reader.iter_local())

    assert [frame.frame_id for frame in frames] == [2, 0, 2]
    np.testing.assert_allclose(
        [frame.positions[0, 0] for frame in frames],
        [12.0, 10.0, 12.0],
        atol=1e-3,
    )


def test_single_segment_fake_mpi_yields_global_ids(tmp_path):
    path = tmp_path / "single.xtc"
    _write_one_atom_xtc(path, [10.0, 11.0, 12.0, 13.0])
    reader = MPITrajReader(
        trajectory_files=[path],
        comm=_RankOneOfTwo(),
        strategy="reopen",
    )
    reader.plan = TrajPlan(
        total_frames=4,
        frame_ids=(0, 1, 2, 3),
        segment_frame_counts=(4,),
    )

    frames = list(reader.iter_local())
    assert [frame.frame_id for frame in frames] == [2, 3]
    np.testing.assert_allclose(
        [frame.positions[0, 0] for frame in frames],
        [12.0, 13.0],
        atol=1e-3,
    )


def test_cross_segment_explicit_order_pairs_global_ids_and_fields(tmp_path):
    topology = tmp_path / "topology.pdb"
    first = tmp_path / "first.trr"
    second = tmp_path / "second.trr"
    template = mda.Universe.empty(1, trajectory=True)
    template.add_TopologyAttr("name", ["CA"])
    template.add_TopologyAttr("type", ["CA"])
    template.add_TopologyAttr("resid", [1])
    template.add_TopologyAttr("resname", ["ALA"])
    template.atoms.positions = [[0.0, 0.0, 0.0]]
    with mda.Writer(str(topology), n_atoms=1) as writer:
        writer.write(template.atoms)
    _write_one_atom_trr(first, [10.0, 11.0])
    _write_one_atom_trr(second, [20.0, 21.0])

    selected = [3, 0, 2, 2]
    full = open_universe(topology, [first, second], trajectory_format="TRR")
    expected = list(
        iter_frames(
            full,
            frame_ids=selected,
            include_forces=True,
            include_time=True,
        )
    )
    reader = MPITrajReader(
        topology=topology,
        trajectory_files=[first, second],
        trajectory_format="TRR",
        strategy="local_segments",
    )
    reader.plan = TrajPlan(
        total_frames=4,
        frame_ids=tuple(selected),
        segment_frame_counts=(2, 2),
    )
    actual = list(reader.iter_local(include_forces=True, include_time=True))

    assert [frame.frame_id for frame in actual] == selected
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got.positions, want.positions)
        np.testing.assert_array_equal(got.forces, want.forces)
        assert got.time == pytest.approx(want.time)


class _EmptyRank:
    def Get_rank(self):
        return 3

    def Get_size(self):
        return 4

    def allgather(self, value):
        return [None, None, None, None]


def test_empty_rank_enters_reader_and_yields_no_frames(tmp_path):
    reader = MPITrajReader(
        trajectory_files=[tmp_path / "unused.xtc"],
        comm=_EmptyRank(),
        strategy="reopen",
    )
    reader.plan = TrajPlan(
        total_frames=1,
        frame_ids=(0,),
        segment_frame_counts=(1,),
    )
    reader.open_full = lambda: pytest.fail("an empty rank must not open a trajectory")
    assert list(reader.iter_local()) == []


class _RootCollective:
    def __init__(self):
        self.broadcasts = []

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 2

    def bcast(self, value, root=0):
        self.broadcasts.append(value)
        return value

    def allgather(self, value):
        return [None, None]


def test_scan_live_universe_is_reused_by_broadcast_reader(tmp_path):
    universe = mda.Universe.empty(1, trajectory=True)
    universe.atoms.positions = [[4.0, 0.0, 0.0]]
    comm = _RootCollective()
    reader = MPITrajReader(
        trajectory_files=[tmp_path / "unused.lammpstrj"],
        comm=comm,
        strategy="broadcast",
    )
    reader.scan(inspection_universe=universe)
    reader.open_full = lambda: pytest.fail("the inspection Universe must be reused")

    frames = list(reader.iter_local())
    assert [frame.frame_id for frame in frames] == [0]
    assert len(comm.broadcasts) == 2


class _SequentialBroadcastState:
    value = None


class _SequentialBroadcastComm:
    def __init__(self, rank, state):
        self.rank = rank
        self.state = state

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return 2

    def bcast(self, value, root=0):
        if self.rank == root:
            self.state.value = value
        return self.state.value


def test_collective_failure_primitives_preserve_root_error_and_lowest_rank():
    assert broadcast_root_outcome(("ready", None), comm=None) == "ready"

    state = _SequentialBroadcastState()
    for rank in (0, 1):
        with pytest.raises(LookupError, match="root failure"):
            broadcast_root_outcome(
                (None, LookupError("root failure")) if rank == 0 else None,
                comm=_SequentialBroadcastComm(rank, state),
            )

    class ConsensusComm:
        def Get_size(self):
            return 3

        def allgather(self, value):
            return [LookupError("lowest rank"), value, ValueError("later rank")]

    with pytest.raises(LookupError, match="lowest rank"):
        raise_if_rank_failed(ValueError("local rank"), comm=ConsensusComm())

    class EmptyRankComm:
        def Get_size(self):
            return 2

        def allgather(self, value):
            assert value is None
            return [None, OSError("other rank")]

    with pytest.raises(OSError, match="other rank"):
        raise_if_rank_failed(None, comm=EmptyRankComm())


def test_scan_failure_is_broadcast_with_original_type_and_message(tmp_path):
    state = _SequentialBroadcastState()
    root = MPITrajReader(
        trajectory_files=[tmp_path / "unused.xtc"],
        comm=_SequentialBroadcastComm(0, state),
    )
    peer = MPITrajReader(
        trajectory_files=[tmp_path / "unused.xtc"],
        comm=_SequentialBroadcastComm(1, state),
    )

    def fail_inspection():
        raise LookupError("rank-zero inspection failed")

    root.open_full = fail_inspection
    for reader in (root, peer):
        with pytest.raises(LookupError, match="rank-zero inspection failed"):
            reader.scan()


class _PresetOpenCollective:
    def __init__(self, rank, errors):
        self.rank = rank
        self.errors = errors

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return 2

    def allgather(self, value):
        return list(self.errors)


def test_rank_local_open_failure_is_raised_on_every_rank(tmp_path):
    failure = OSError("rank-zero local open failed")
    for rank in (0, 1):
        reader = MPITrajReader(
            trajectory_files=[tmp_path / "unused.xtc"],
            comm=_PresetOpenCollective(rank, [failure, None]),
            strategy="reopen",
        )
        reader.plan = TrajPlan(
            total_frames=2,
            frame_ids=(0, 1),
            segment_frame_counts=(2,),
        )
        if rank == 0:
            reader.open_full = lambda: (_ for _ in ()).throw(failure)
        else:
            reader.open_full = lambda: mda.Universe.empty(1, trajectory=True)
        with pytest.raises(OSError, match="rank-zero local open failed"):
            reader.iter_local()


def test_iter_local_rejects_wrong_reader_local_order(monkeypatch, tmp_path):
    reader = MPITrajReader(trajectory_files=[tmp_path / "unused.xtc"])
    reader.plan = TrajPlan(
        total_frames=22,
        frame_ids=(20, 21),
        segment_frame_counts=(22,),
    )
    reader._open_local = lambda: (object(), [0, 1])
    records = [FrameRecord(frame_id=1, positions=np.zeros((1, 3)))]
    monkeypatch.setattr(trajectory_io, "iter_frames", lambda *args, **kwargs: iter(records))

    with pytest.raises(RuntimeError, match="reader-local frame order"):
        list(reader.iter_local())


def test_iter_local_rejects_too_few_records(monkeypatch, tmp_path):
    reader = MPITrajReader(trajectory_files=[tmp_path / "unused.xtc"])
    reader.plan = TrajPlan(
        total_frames=22,
        frame_ids=(20, 21),
        segment_frame_counts=(22,),
    )
    reader._open_local = lambda: (object(), [0, 1])
    records = [FrameRecord(frame_id=0, positions=np.zeros((1, 3)))]
    monkeypatch.setattr(trajectory_io, "iter_frames", lambda *args, **kwargs: iter(records))

    with pytest.raises(RuntimeError, match="yielded 1 frame.*expected 2"):
        list(reader.iter_local())


def test_iter_local_rejects_too_many_records(monkeypatch, tmp_path):
    reader = MPITrajReader(trajectory_files=[tmp_path / "unused.xtc"])
    reader.plan = TrajPlan(
        total_frames=21,
        frame_ids=(20,),
        segment_frame_counts=(21,),
    )
    reader._open_local = lambda: (object(), [0])
    records = [
        FrameRecord(frame_id=0, positions=np.zeros((1, 3))),
        FrameRecord(frame_id=1, positions=np.zeros((1, 3))),
    ]
    monkeypatch.setattr(trajectory_io, "iter_frames", lambda *args, **kwargs: iter(records))

    with pytest.raises(RuntimeError, match="yielded more than 1 frame"):
        list(reader.iter_local())


def test_atom_subset_gather_matches_the_full_frame(tmp_path):
    """`atom_indices` must be a pure copy-saving change, not a numerical one."""
    universe = mda.Universe.empty(6, trajectory=True, forces=True)
    universe.atoms.positions = np.arange(18, dtype=np.float64).reshape(6, 3)
    universe.atoms.forces = -np.arange(18, dtype=np.float64).reshape(6, 3)
    universe.dimensions = [40.0, 40.0, 40.0, 90.0, 90.0, 90.0]

    wanted = [4, 0, 3]
    full = next(iter_frames(universe, include_forces=True))
    subset = next(iter_frames(universe, include_forces=True, atom_indices=wanted))

    assert subset["positions"].shape == (3, 3)
    np.testing.assert_array_equal(subset["positions"], full["positions"][wanted])
    np.testing.assert_array_equal(
        subset["forces"].reshape(3, 3), full["forces"].reshape(6, 3)[wanted]
    )
    with pytest.raises(ValueError, match="atom_indices spans"):
        next(iter_frames(universe, atom_indices=[0, 99]))


def test_scan_records_what_the_first_selected_frame_can_supply(tmp_path):
    """The capability flags ride on the plan so no rank reopens to read them."""
    path = tmp_path / "only.xtc"
    _write_one_atom_xtc(path, [1.0, 2.0, 3.0])

    reader = MPITrajReader(trajectory_files=[path])
    plan = reader.scan(start=1)
    assert plan.frame_ids == (1, 2)
    assert plan.has_forces is False  # XTC never carries forces
    assert plan.has_velocities is False

    # A caller-supplied shape means nothing was opened, so nothing is claimed.
    quiet = MPITrajReader(trajectory_files=[path]).scan(segment_frame_counts=[3])
    assert quiet.has_forces is None and quiet.has_velocities is None
