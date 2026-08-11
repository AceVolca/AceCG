"""Concrete VP trajectory-growth and publication terminal."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .trajectory import (
    MPITrajReader,
    broadcast_root_outcome,
    open_universe,
    raise_if_rank_failed,
)
from .vp_ffbuilder import vp_forcefield_inventory, write_latent_settings
from ..topology.vpgrower import VPGrower, VPGrownFrame, write_vp_data


def grow_vp_trajectory(
    *,
    config: Any,
    reader: MPITrajReader,
    output_dir: Path,
    reference_topology: Path,
    comm: Optional[Any],
) -> Dict[str, Any]:
    """Grow selected source frames and publish one complete VP output set."""
    rank = 0 if comm is None else int(comm.Get_rank())
    size = 1 if comm is None else int(comm.Get_size())
    total_start = time.monotonic()
    output_dir = output_dir.resolve()

    template_payload = None
    template_error: Optional[Exception] = None
    if rank == 0:
        try:
            started = time.monotonic()
            topology_universe = open_universe(
                reference_topology,
                atom_style="id resid type charge x y z",
            )
            template = VPGrower.from_universe(
                topology_universe,
                config.vp,
                type_aliases=config.aa_ref.ref_topo_type_names,
            ).template
            template_payload = (template, time.monotonic() - started)
        except Exception as exc:
            template_error = exc
    template, template_elapsed = broadcast_root_outcome(
        (template_payload, template_error) if rank == 0 else None,
        comm=comm,
    )

    scan_started = time.monotonic()
    plan = reader.scan(
        start=config.aa_ref.skip_frames,
        every=config.aa_ref.every,
        n_frames=config.aa_ref.n_frames,
        frame_ids=config.run.frame_ids,
    )
    scan_elapsed = time.monotonic() - scan_started
    if config.aa_ref.include_forces and plan.has_forces is False:
        raise ValueError(
            "[aa_ref] include_forces = true but the selected trajectory carries no forces."
        )

    selected_ids = [int(frame_id) for frame_id in plan.frame_ids]
    first_occurrence: Dict[int, int] = {}
    for selection_index, source_frame_id in enumerate(selected_ids):
        first_occurrence.setdefault(source_frame_id, selection_index)
    unique_ids = list(first_occurrence)

    forcefield_names = vp_forcefield_inventory(
        config.vp,
        template,
        include_name=config.run.latent_settings_name,
    )
    relative_targets = [Path("vp_topology.data")]
    relative_targets.extend(Path(name) for name in forcefield_names)
    relative_targets.extend(
        Path(f"frame_{source_frame_id:06d}.data") for source_frame_id in unique_ids
    )
    if config.aa_ref.include_forces:
        relative_targets.extend(
            Path(f"frame_{source_frame_id:06d}.forces.npy")
            for source_frame_id in unique_ids
        )
    relative_targets.extend((Path("timing.json"), Path("manifest.json")))
    final_targets = [(output_dir / path).resolve() for path in relative_targets]
    staging_dir = (output_dir / ".vp-growth-stage").resolve()

    preflight_error: Optional[Exception] = None
    staging_created = False
    if rank == 0:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            outside = next(
                (target for target in final_targets if output_dir not in target.parents),
                None,
            )
            if outside is not None:
                raise ValueError(
                    f"VP final output target {outside} must be inside {output_dir}."
                )
            if len(set(final_targets)) != len(final_targets):
                raise ValueError("VP final output targets resolve to the same path.")
            # Frame targets are direct files under output_dir with names keyed by
            # unique_ids (a dict, so already collision-free) and cannot be a
            # parent of anything, so only the small non-frame target set needs a
            # pairwise walk; each non-frame target is still checked against the
            # (potentially large, e.g. 18k-frame) frame set, but that is linear
            # rather than quadratic in the frame count. See F-012.
            frame_targets = {
                (output_dir / path).resolve()
                for path in relative_targets
                if path.name.startswith("frame_")
            }
            other_targets = [target for target in final_targets if target not in frame_targets]
            if any(
                first in second.parents or second in first.parents
                for index, first in enumerate(other_targets)
                for second in other_targets[index + 1 :]
            ) or any(
                other in frame.parents or frame in other.parents
                for other in other_targets
                for frame in frame_targets
            ):
                raise ValueError("VP final output targets must not contain one another.")
            if any(
                target == staging_dir
                or target in staging_dir.parents
                or staging_dir in target.parents
                for target in final_targets
            ):
                raise ValueError("VP final targets must not overlap internal staging.")
            if staging_dir.exists():
                raise FileExistsError(f"VP staging path already exists: {staging_dir}")
            if not config.run.overwrite:
                existing = next((target for target in final_targets if target.exists()), None)
                if existing is not None:
                    raise FileExistsError(
                        f"{existing} already exists. Set [vp] overwrite = true to replace it."
                    )
            staging_dir.mkdir()
            staging_created = True
            for relative in relative_targets:
                (staging_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            if staging_created and staging_dir.exists():
                try:
                    shutil.rmtree(staging_dir)
                except Exception as cleanup_exc:
                    exc = cleanup_exc
            preflight_error = exc
    staging_dir = broadcast_root_outcome(
        (staging_dir, preflight_error) if rank == 0 else None,
        comm=comm,
    )

    result: Optional[Dict[str, Any]] = None
    terminal_error: Optional[Exception] = None
    try:
        schema_elapsed = 0.0
        schema_error: Optional[Exception] = None
        if rank == 0:
            try:
                started = time.monotonic()
                zero = VPGrownFrame(
                    positions=np.zeros((template.n_atoms, 3), dtype=np.float64),
                    dimensions=np.array(
                        [1.0, 1.0, 1.0, 90.0, 90.0, 90.0], dtype=np.float64
                    ),
                )
                write_vp_data(
                    template,
                    zero,
                    staging_dir / "vp_topology.data",
                    title="VP topology (schema)",
                )
                write_latent_settings(
                    template=template,
                    vp_config=config.vp,
                    output_dir=staging_dir,
                    table_points=config.run.table_points,
                    table_rmin=config.run.table_rmin,
                    table_rmax=config.run.table_rmax,
                    include_name=config.run.latent_settings_name,
                )
                schema_elapsed = time.monotonic() - started
            except Exception as exc:
                schema_error = exc
        schema_elapsed = broadcast_root_outcome(
            (schema_elapsed, schema_error) if rank == 0 else None,
            comm=comm,
        )

        local_count, local_offset, local_ids = reader.local_slice()
        local_records = []
        local_written = []
        local_error: Optional[Exception] = None
        grow_started = time.monotonic()
        try:
            frames = reader.iter_local(include_forces=config.aa_ref.include_forces)
            grower = VPGrower(template)
            for local_index, frame in enumerate(frames):
                selection_index = int(local_offset + local_index)
                if local_index >= len(local_ids):
                    raise RuntimeError(
                        f"rank {rank}: VP reader yielded more than {local_count} "
                        "assigned occurrence(s)."
                    )
                source_frame_id = int(frame.frame_id)
                if source_frame_id != int(local_ids[local_index]):
                    raise RuntimeError(
                        f"rank {rank}: VP source id {source_frame_id} does not match "
                        f"selection index {selection_index} id {int(local_ids[local_index])}."
                    )
                positions = np.asarray(frame.positions)
                if positions.shape != (template.n_real, 3):
                    raise ValueError(
                        f"Trajectory frame {source_frame_id} has position shape {positions.shape}; "
                        f"VP template expects ({template.n_real}, 3)."
                    )
                force_array = None
                if config.aa_ref.include_forces:
                    if frame.forces is None:
                        raise ValueError(
                            f"Trajectory frame {source_frame_id} is missing requested forces."
                        )
                    force_array = np.asarray(frame.forces, dtype=np.float32).reshape(
                        template.n_real, 3
                    )
                data_relative = Path(f"frame_{source_frame_id:06d}.data")
                force_relative = (
                    Path(f"frame_{source_frame_id:06d}.forces.npy")
                    if config.aa_ref.include_forces
                    else None
                )
                if first_occurrence[source_frame_id] == selection_index:
                    grown = grower.grow_frame(
                        positions,
                        frame.box,
                        orientation_seed=(
                            config.run.orientation_seed_base + source_frame_id
                        ),
                    )
                    write_vp_data(template, grown, staging_dir / data_relative)
                    if force_relative is not None:
                        np.save(staging_dir / force_relative, force_array)
                    local_written.append(source_frame_id)
                local_records.append(
                    {
                        "selection_index": selection_index,
                        "source_frame_id": source_frame_id,
                        "orientation_seed": (
                            config.run.orientation_seed_base + source_frame_id
                        ),
                        "data": str(output_dir / data_relative),
                        "forces": (
                            None
                            if force_relative is None
                            else str(output_dir / force_relative)
                        ),
                    }
                )
            if len(local_records) != local_count:
                raise RuntimeError(
                    f"rank {rank}: VP reader yielded {len(local_records)} occurrence(s); "
                    f"expected {local_count}."
                )
        except Exception as exc:
            local_error = exc
        grow_elapsed = time.monotonic() - grow_started
        raise_if_rank_failed(local_error, comm=comm)

        local_payload = {
            "rank": rank,
            "selected_offset": int(local_offset),
            "selected_count": int(local_count),
            "source_frame_ids": [int(value) for value in local_ids],
            "records": local_records,
            "written_source_frame_ids": local_written,
            "segments_loaded": [int(value) for value in reader.opened_segment_numbers],
            "scan_elapsed_sec": scan_elapsed,
            "grow_elapsed_sec": grow_elapsed,
        }
        gathered = [local_payload] if comm is None else comm.gather(local_payload, root=0)

        final_error: Optional[Exception] = None
        if rank == 0:
            try:
                ordered_records = []
                for expected_rank in range(size):
                    expected_count, expected_offset, expected_ids = reader.local_slice(
                        expected_rank
                    )
                    item = gathered[expected_rank]
                    if (
                        int(item["rank"]) != expected_rank
                        or int(item["selected_offset"]) != expected_offset
                        or int(item["selected_count"]) != expected_count
                        or [int(value) for value in item["source_frame_ids"]]
                        != [int(value) for value in expected_ids]
                    ):
                        raise RuntimeError(
                            "VP rank slices did not cover the selected plan exactly."
                        )
                    records = list(item["records"])
                    expected_written = [
                        int(source_frame_id)
                        for selection_index, source_frame_id in zip(
                            range(expected_offset, expected_offset + expected_count),
                            expected_ids,
                        )
                        if first_occurrence[int(source_frame_id)] == selection_index
                    ]
                    if (
                        len(records) != expected_count
                        or [int(value) for value in item["written_source_frame_ids"]]
                        != expected_written
                    ):
                        raise RuntimeError(
                            f"rank {expected_rank} returned wrong VP occurrence/writer counts."
                        )
                    ordered_records.extend(records)
                expected_records = [
                    {
                        "selection_index": index,
                        "source_frame_id": source_frame_id,
                        "orientation_seed": config.run.orientation_seed_base
                        + source_frame_id,
                        "data": str(output_dir / f"frame_{source_frame_id:06d}.data"),
                        "forces": (
                            str(
                                output_dir
                                / f"frame_{source_frame_id:06d}.forces.npy"
                            )
                            if config.aa_ref.include_forces
                            else None
                        ),
                    }
                    for index, source_frame_id in enumerate(selected_ids)
                ]
                if ordered_records != expected_records:
                    raise RuntimeError(
                        "VP occurrence records do not match the selected plan."
                    )

                rank_slices = [
                    {key: value for key, value in item.items() if key != "records"}
                    for item in gathered
                ]
                timing_payload = {
                    "mpi": {
                        "enabled": comm is not None,
                        "size": size,
                        "universe_strategy": reader.strategy,
                        "rank_slices": rank_slices,
                    },
                    "phase_seconds": {
                        "template_build": float(template_elapsed),
                        "trajectory_scan": max(
                            float(item["scan_elapsed_sec"]) for item in gathered
                        ),
                        "schema_write": float(schema_elapsed),
                        "frame_growth_wall": max(
                            float(item["grow_elapsed_sec"]) for item in gathered
                        ),
                        "total": time.monotonic() - total_start,
                    },
                    "frames": {
                        "selected_occurrences": len(selected_ids),
                        "unique_source_frames": len(unique_ids),
                    },
                }
                timing_staged = staging_dir / "timing.json"
                timing_staged.write_text(json.dumps(timing_payload, indent=2))
                manifest_payload = {
                    "source": {
                        "trajectory_files": list(reader.trajectory_files),
                        "trajectory_format": reader.trajectory_format,
                        "segment_frame_counts": [
                            int(value) for value in plan.segment_frame_counts
                        ],
                        "selected_global_ids": selected_ids,
                    },
                    "mpi": {
                        "enabled": comm is not None,
                        "size": size,
                        "universe_strategy": reader.strategy,
                        "rank_slices": rank_slices,
                    },
                    "topology": {
                        "n_atoms": int(template.n_atoms),
                        "n_real": int(template.n_real),
                        "n_vp": int(template.n_vp),
                        "atomtype_order": config.vp.atomtype_order,
                    },
                    "occurrences": ordered_records,
                    "inventory": [str(path) for path in final_targets],
                    "timing": str(output_dir / "timing.json"),
                }
                manifest_staged = staging_dir / "manifest.json"
                manifest_staged.write_text(json.dumps(manifest_payload, indent=2))

                missing = [
                    str(relative)
                    for relative in relative_targets
                    if not (staging_dir / relative).is_file()
                ]
                if missing:
                    raise RuntimeError(
                        f"VP staging is missing enumerated output(s): {missing}"
                    )
                for source_frame_id in unique_ids:
                    data_path = staging_dir / f"frame_{source_frame_id:06d}.data"
                    if data_path.stat().st_size == 0:
                        raise RuntimeError(f"VP DATA output is empty: {data_path}")
                    if config.aa_ref.include_forces:
                        forces = np.load(
                            staging_dir / f"frame_{source_frame_id:06d}.forces.npy"
                        )
                        if forces.dtype != np.float32 or forces.shape != (
                            template.n_real,
                            3,
                        ):
                            raise RuntimeError(
                                f"VP force output for frame {source_frame_id} has "
                                f"dtype/shape {forces.dtype}/{forces.shape}."
                            )
                json.loads(timing_staged.read_text())
                json.loads(manifest_staged.read_text())

                manifest_final = output_dir / "manifest.json"
                if manifest_final.exists():
                    if manifest_final.is_dir():
                        shutil.rmtree(manifest_final)
                    else:
                        manifest_final.unlink()
                for relative, final_target in zip(relative_targets, final_targets):
                    if relative == Path("manifest.json"):
                        continue
                    staged_target = staging_dir / relative
                    if final_target.exists():
                        if final_target.is_dir():
                            shutil.rmtree(final_target)
                        else:
                            final_target.unlink()
                    final_target.parent.mkdir(parents=True, exist_ok=True)
                    staged_target.replace(final_target)
                manifest_staged.replace(manifest_final)
                result = {
                    "output_dir": output_dir,
                    "topology_path": output_dir / "vp_topology.data",
                    "latent_settings_path": (
                        output_dir / config.run.latent_settings_name
                    ).resolve(),
                    "timing_path": output_dir / "timing.json",
                    "manifest_path": manifest_final,
                    "n_selected": len(selected_ids),
                    "n_unique": len(unique_ids),
                }
            except Exception as exc:
                final_error = exc
        if rank == 0 and final_error is not None:
            terminal_error = final_error
    except Exception as exc:
        terminal_error = exc

    if rank == 0 and staging_dir.exists():
        try:
            shutil.rmtree(staging_dir)
        except Exception as exc:
            if terminal_error is None:
                terminal_error = exc
    result = broadcast_root_outcome(
        (result, terminal_error) if rank == 0 else None,
        comm=comm,
    )
    return result
