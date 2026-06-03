"""Workflow helper tests for VP MPI frame partitioning."""

from __future__ import annotations

from pathlib import Path

from AceCG.workflows.vp_growth import (
    _choose_universe_loading_strategy,
    _partition_frame_ids,
    _select_local_trajectory_inputs,
    _topology_format_for_path,
)


def test_partition_frame_ids_balances_contiguous_slices() -> None:
    frame_ids = list(range(10))
    local_count, local_offset, local_ids = _partition_frame_ids(frame_ids, size=3, rank=1)
    assert local_count == 3
    assert local_offset == 4
    assert local_ids == [4, 5, 6]


def test_topology_format_for_path_detects_lammps_data() -> None:
    assert _topology_format_for_path(Path("cg6.data")) == "DATA"
    assert _topology_format_for_path(Path("cg6.pdb")) is None


def test_choose_universe_loading_strategy_uses_measured_segment_threshold() -> None:
    assert _choose_universe_loading_strategy(segment_count=1, size=48) == "broadcast"
    assert _choose_universe_loading_strategy(segment_count=2, size=48) == "broadcast"
    assert _choose_universe_loading_strategy(segment_count=3, size=48) == "local_segments"
    assert _choose_universe_loading_strategy(segment_count=15, size=1) == "broadcast"


def test_select_local_trajectory_inputs_maps_global_frames_onto_local_subset() -> None:
    traj_paths = ["seg1", "seg2", "seg3"]
    local_paths, local_frame_ids, segment_ids = _select_local_trajectory_inputs(
        [95, 205, 215], traj_paths, [100, 100, 100],
    )
    assert local_paths == ["seg1", "seg3"]
    assert local_frame_ids == [95, 105, 115]
    assert segment_ids == [1, 3]