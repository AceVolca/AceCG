"""Concrete TrajMap transformation and publication terminal."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..compute.cgmap import CGMapper
from ..io.coordinates_writers import write_gro, write_lammps_data
from ..io.force_operator import write_force_operator
from ..io.trajectory import MPITrajReader, broadcast_root_outcome, iter_frames, raise_if_rank_failed
from ..topology.cgmap import CGMapSpec


def map_cg_trajectory(*, config: Any, spec: CGMapSpec, reader: MPITrajReader, plan: Any,
                      output_dir: Path, topology_path: Optional[Path], trajectory_paths: Sequence[str],
                      map_path: Path, aa_masses_used: bool,
                      force_mapping_report: Optional[Dict[str, Any]], phase_seconds: Dict[str, float],
                      total_start: float, comm: Optional[Any]) -> Dict[str, Any]:
    """Map one scanned plan and publish its complete TrajMap output set."""
    import MDAnalysis as mda

    rank = 0 if comm is None else int(comm.Get_rank())
    size = 1 if comm is None else int(comm.Get_size())
    mapper = CGMapper(spec, unwrap=config.run.unwrap, wrap=config.run.wrap,
                      triclinic=config.run.triclinic, dtype=np.dtype(config.run.precision),
                      out_dtype=np.float32)
    output_dir = output_dir.resolve()
    suffix = Path(config.run.trajectory_name).suffix
    trajectory_final = (output_dir / config.run.trajectory_name).resolve()
    segment_final = (output_dir / "segments").resolve()
    topology_final = None if not config.run.topology_name else (output_dir / config.run.topology_name).resolve()
    aliases_final = None if topology_final is None else topology_final.with_name(topology_final.stem + "_aliases.json")
    gro_final = None if not config.run.write_gro else (output_dir / (Path(config.run.topology_name or "cg").stem + ".gro")).resolve()
    report_final = (output_dir / config.run.report_name).resolve()
    force_map_final = None
    if spec.has_force_operator:
        force_map_final = (output_dir / config.force_mapping.artifact_name).resolve()
        if force_map_final.suffix.lower() != ".npz":
            force_map_final = force_map_final.with_suffix(".npz")
    targets = {
        "trajectory": trajectory_final if config.run.merge_segments else None,
        "segments": segment_final if not config.run.merge_segments or config.run.keep_segments else None,
        "topology": topology_final, "aliases": aliases_final, "gro": gro_final,
        "report": report_final, "force_map": force_map_final,
    }
    staging_dir: Optional[Path] = None
    rank_segment_dir: Optional[Path] = None
    try:
        root_error: Optional[Exception] = None
        topology_elapsed = 0.0
        if rank == 0:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                concrete = [path for path in targets.values() if path is not None]
                for path in concrete:
                    try:
                        path.relative_to(output_dir)
                    except ValueError as exc:
                        raise ValueError(f"TrajMap final output target {path} must be inside {output_dir}.") from exc
                if len({path.resolve() for path in concrete}) != len(concrete):
                    raise ValueError("TrajMap final output targets resolve to the same path.")
                for index, first in enumerate(concrete):
                    for second in concrete[index + 1:]:
                        try:
                            second.relative_to(first)
                        except ValueError:
                            try:
                                first.relative_to(second)
                            except ValueError:
                                continue
                        raise ValueError("TrajMap final output targets must not contain one another.")
                if not config.run.overwrite:
                    existing = next((path for path in concrete if path.exists()), None)
                    if existing is not None:
                        raise FileExistsError(f"{existing} already exists. Set [trajmap] overwrite = true to replace it.")
                if config.aa.include_forces and plan.has_forces is False:
                    raise ValueError("[aa] include_forces = true but the trajectory carries no forces "
                                     f"(frame {plan.frame_ids[0]}). XTC never does; a TRR must have been written with forces.")
                if config.aa.include_velocities and plan.has_velocities is False:
                    raise ValueError("[aa] include_velocities = true but the trajectory carries no "
                                     f"velocities (frame {plan.frame_ids[0]}).")
                staging_dir = Path(tempfile.mkdtemp(prefix=".trajmap-stage-", dir=output_dir))
                rank_segment_dir = staging_dir / ".rank-segments"
                for path in concrete:
                    staged_target = staging_dir / path.relative_to(output_dir)
                    try:
                        staged_target.relative_to(rank_segment_dir)
                    except ValueError:
                        try:
                            rank_segment_dir.relative_to(staged_target)
                        except ValueError:
                            continue
                    raise ValueError("TrajMap final output targets must not overlap internal rank-segment staging.")
                for path in (topology_final, gro_final, force_map_final):
                    if path is not None:
                        (staging_dir / path.relative_to(output_dir)).parent.mkdir(parents=True, exist_ok=True)
                if spec.has_force_operator:
                    assert force_map_final is not None
                    write_force_operator(staging_dir / force_map_final.relative_to(output_dir), spec, force_mapping_report or {})
                topology_started = time.monotonic()
                _write_cg_topology(
                    mapper=mapper, reader=reader, source_name=(Path(trajectory_paths[0]).name if topology_path is None else topology_path.name),
                    frame_id=int(plan.frame_ids[0]), topology_path=(None if topology_final is None else staging_dir / topology_final.relative_to(output_dir)),
                    aliases_path=(None if aliases_final is None else staging_dir / aliases_final.relative_to(output_dir)),
                    gro_path=(None if gro_final is None else staging_dir / gro_final.relative_to(output_dir)),
                    resname=config.mapping.resname,
                )
                topology_elapsed = time.monotonic() - topology_started
                rank_segment_dir.mkdir()
            except Exception as exc:
                root_error = exc
        staging_dir, rank_segment_dir = broadcast_root_outcome(
            ((staging_dir, rank_segment_dir), root_error) if rank == 0 else None, comm=comm
        )

        map_error: Optional[Exception] = None
        frames_written = 0
        t0 = time.monotonic()
        try:
            _, frames_written = _stream_local_segment(
                mapper=mapper, reader=reader, segment_dir=rank_segment_dir, rank=rank,
                include_forces=bool(config.aa.include_forces), include_velocities=bool(config.aa.include_velocities),
                resname=config.mapping.resname, suffix=suffix,
            )
        except Exception as exc:
            map_error = exc
        map_elapsed = time.monotonic() - t0
        raise_if_rank_failed(map_error, comm=comm)
        local_count, local_offset, _ = reader.local_slice()
        local_stats = {"rank": rank, "selected_offset": int(local_offset), "selected_count": int(local_count),
                       "written_count": int(frames_written), "map_elapsed_sec": map_elapsed,
                       "frames_per_sec": frames_written / map_elapsed if map_elapsed > 0 else None}
        gathered = [local_stats] if comm is None else comm.gather(local_stats, root=0)

        result: Optional[Dict[str, Any]] = None
        final_error: Optional[Exception] = None
        if rank == 0:
            try:
                ordered_segments = []
                for expected_rank in range(size):
                    expected_count, expected_offset, _ = reader.local_slice(expected_rank)
                    item = gathered[expected_rank]
                    if (int(item["rank"]) != expected_rank or int(item["selected_offset"]) != expected_offset
                            or int(item["selected_count"]) != expected_count or int(item["written_count"]) != expected_count):
                        raise RuntimeError("TrajMap rank slices did not cover the selected plan.")
                    if expected_count:
                        ordered_segments.append(rank_segment_dir / f"segment_{expected_rank:04d}{suffix}")
                n_written = sum(int(item["written_count"]) for item in gathered)
                if n_written != len(plan.frame_ids):
                    raise RuntimeError(f"Mapped {n_written} frame(s) but selected {len(plan.frame_ids)}; a rank lost frames.")
                t_merge_start = time.monotonic()
                if config.run.merge_segments:
                    staged_trajectory = staging_dir / trajectory_final.relative_to(output_dir)
                    staged_trajectory.parent.mkdir(parents=True, exist_ok=True)
                    template = build_cg_universe(spec, resname=config.mapping.resname,
                                                 velocities=bool(config.aa.include_velocities), forces=bool(config.aa.include_forces))
                    merged = 0
                    with mda.Writer(str(staged_trajectory), n_atoms=len(template.atoms)) as writer:
                        for segment in ordered_segments:
                            template.load_new(str(segment))
                            for _ in template.trajectory:
                                template.trajectory.ts.frame = merged
                                writer.write(template.atoms)
                                merged += 1
                    if merged != n_written:
                        raise RuntimeError(f"Merged {merged} frame(s) into {staged_trajectory} but expected {n_written}.")
                if config.run.merge_segments and not config.run.keep_segments:
                    for segment in ordered_segments:
                        _remove_segment(segment)
                    rank_segment_dir.rmdir()
                    staged_segments: Tuple[Path, ...] = ()
                else:
                    staged_segment_dir = staging_dir / segment_final.relative_to(output_dir)
                    rank_segment_dir.replace(staged_segment_dir)
                    staged_segments = tuple(staged_segment_dir / path.name for path in ordered_segments)
                final_segments = tuple(segment_final / path.name for path in staged_segments)
                t_merge = time.monotonic() - t_merge_start
                report = {
                    "mpi": {"enabled": comm is not None, "size": size, "universe_strategy": reader.strategy, "rank_slices": gathered},
                    "phase_seconds": {**phase_seconds, "cg_topology_write": topology_elapsed,
                                      "map_wall": max(float(item["map_elapsed_sec"]) for item in gathered), "merge": t_merge,
                                      "total": time.monotonic() - total_start},
                    "source": {"trajectory_files": list(trajectory_paths), "trajectory_format": reader.trajectory_format,
                               "segment_frame_counts": list(plan.segment_frame_counts), "frame_ids": list(plan.frame_ids)},
                    "frames": {"total_in_trajectory": int(plan.total_frames), "selected": len(plan.frame_ids),
                               "written": int(n_written), "every": int(config.aa.every)},
                    "mapping": {"map_file": str(map_path), "n_sites": int(spec.n_sites), "n_molecules": int(spec.n_mol),
                                "n_types": int(spec.n_types), "type_names": list(spec.type_names),
                                "n_required_atoms": int(spec.n_required_atoms), "nnz": int(spec.nnz),
                                "has_bonded_topology": bool(spec.has_bonded_topology),
                                "molecules_contiguous": bool(spec.molecules_contiguous), "masses_from_topology": bool(aa_masses_used),
                                "spec_bytes": int(spec.nbytes())},
                    "kernel": {"unwrap": config.run.unwrap, "wrap": bool(config.run.wrap), "triclinic": config.run.triclinic,
                               "precision": config.run.precision, "include_forces": bool(config.aa.include_forces),
                               "include_velocities": bool(config.aa.include_velocities)},
                    "force_mapping": force_mapping_report,
                    "outputs": {"trajectory": None if not config.run.merge_segments else str(trajectory_final),
                                "segments": [str(path) for path in final_segments], "topology": None if topology_final is None else str(topology_final),
                                "gro": None if gro_final is None else str(gro_final), "type_aliases": None if aliases_final is None else str(aliases_final),
                                "force_map": None if force_map_final is None else str(force_map_final)},
                }
                report["outputs"]["report"] = str(report_final)
                staged_report = staging_dir / report_final.relative_to(output_dir)
                staged_report.parent.mkdir(parents=True, exist_ok=True)
                staged_report.write_text(json.dumps(report, indent=2))
                if report_final.exists():
                    shutil.rmtree(report_final) if report_final.is_dir() else report_final.unlink()
                for name in ("trajectory", "segments", "topology", "aliases", "gro", "force_map"):
                    final_path = targets[name]
                    if final_path is None:
                        continue
                    staged_path = staging_dir / final_path.relative_to(output_dir)
                    if final_path.exists():
                        shutil.rmtree(final_path) if final_path.is_dir() else final_path.unlink()
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    staged_path.replace(final_path)
                staged_report.replace(report_final)
                result = {"trajectory_path": trajectory_final if config.run.merge_segments else None, "segment_paths": final_segments,
                          "topology_path": topology_final, "gro_path": gro_final, "aliases_path": aliases_final,
                          "report_path": report_final, "force_map_path": force_map_final, "n_frames": int(n_written)}
            except Exception as exc:
                final_error = exc
        if rank == 0:
            try:
                if staging_dir is not None and staging_dir.exists():
                    shutil.rmtree(staging_dir)
            except Exception as exc:
                if final_error is None:
                    final_error = exc
        result = broadcast_root_outcome((result, final_error) if rank == 0 else None, comm=comm)
        if rank != 0:
            return {"trajectory_path": None, "segment_paths": (), "topology_path": None, "gro_path": None,
                    "aliases_path": None, "report_path": None, "force_map_path": force_map_final, "n_frames": int(frames_written)}
        return result
    finally:
        if rank == 0 and staging_dir is not None and staging_dir.exists():
            try:
                shutil.rmtree(staging_dir)
            except OSError:
                pass


def build_cg_universe(spec: CGMapSpec, *, resname: Any = "CG", velocities: bool = False, forces: bool = False):
    """Return the in-memory MDAnalysis writer universe for ``spec`` CG sites."""
    import MDAnalysis as mda

    arrays = spec.site_arrays(resname=resname)
    universe = mda.Universe.empty(arrays.n_sites, n_residues=arrays.n_residues, atom_resindex=arrays.res_ids,
                                  trajectory=True, velocities=velocities, forces=forces)
    universe.add_TopologyAttr("name", list(arrays.labels))
    universe.add_TopologyAttr("type", list(arrays.site_types()))
    universe.add_TopologyAttr("mass", arrays.masses)
    universe.add_TopologyAttr("charge", arrays.charges)
    universe.add_TopologyAttr("resid", list(range(1, arrays.n_residues + 1)))
    universe.add_TopologyAttr("resname", list(arrays.residue_names))
    return universe


def _write_cg_topology(*, mapper: CGMapper, reader: MPITrajReader, source_name: str, frame_id: int,
                       topology_path: Optional[Path], aliases_path: Optional[Path], gro_path: Optional[Path], resname: Any) -> None:
    """Write staged DATA/GRO/aliases from the first selected mapped frame."""
    universe = reader.open_full()
    frame = next(iter(iter_frames(universe, frame_ids=[frame_id], atom_indices=mapper.spec.atom_indices)))
    cg = mapper.map_frame(frame["positions"], box=frame["box"], frame_id=frame["frame_id"], compact=True)
    del universe
    beads, type2id, type_masses = mapper.spec.bead_records(resname=resname)
    if topology_path is not None:
        write_lammps_data(topology_path, f"AceCG trajmap CG topology from {source_name} frame {frame_id}",
                          np.asarray(cg.positions, dtype=np.float64), beads, type2id, type_masses, cg.box,
                          bonds=mapper.spec.bonds, bond_type_ids=mapper.spec.bond_type_ids, angles=mapper.spec.angles,
                          angle_type_ids=mapper.spec.angle_type_ids, dihedrals=mapper.spec.dihedrals,
                          dihedral_type_ids=mapper.spec.dihedral_type_ids)
        assert aliases_path is not None
        aliases_path.write_text(json.dumps({str(type_id): name for name, type_id in sorted(type2id.items(), key=lambda item: item[1])}, indent=2))
    if gro_path is not None:
        write_gro(gro_path, f"AceCG trajmap CG frame {frame_id}", np.asarray(cg.positions, dtype=np.float64), beads, cg.box)


def _stream_local_segment(*, mapper: CGMapper, reader: MPITrajReader, segment_dir: Path, rank: int,
                          include_forces: bool, include_velocities: bool, resname: Any,
                          suffix: str) -> Tuple[Optional[Path], int]:
    """Stream one rank's contiguous selected slice to its fixed segment path."""
    import MDAnalysis as mda

    frames = reader.iter_local(
        include_forces=include_forces,
        include_velocities=include_velocities,
        include_time=True,
        atom_indices=mapper.spec.atom_indices,
    )
    if not reader.local_frame_ids:
        return None, 0
    segment_path = segment_dir / f"segment_{rank:04d}{suffix}"
    universe = build_cg_universe(mapper.spec, resname=resname, velocities=include_velocities, forces=include_forces)
    timestep = universe.trajectory.ts
    written = 0
    with mda.Writer(str(segment_path), n_atoms=mapper.n_sites) as writer:
        for local_index, frame in enumerate(frames):
            cg = mapper.map_frame(frame["positions"], box=frame["box"],
                                  forces=frame["forces"], velocities=frame["velocities"],
                                  frame_id=frame["frame_id"], compact=True)
            timestep.frame = local_index
            universe.atoms.positions = cg.positions
            if cg.box is not None:
                universe.dimensions = cg.box
            if include_forces:
                universe.atoms.forces = cg.forces
            if include_velocities:
                universe.atoms.velocities = cg.velocities
            timestep.time = float(frame["time"] or 0.0)
            timestep.dt = float(frame["dt"] or 0.0)
            if frame["step"] is not None:
                timestep.data["step"] = int(frame["step"])
            writer.write(universe.atoms)
            written += 1
    return segment_path, written


def _remove_segment(path: Path) -> None:
    """Remove one staged segment and its exact XDR offsets sidecar."""
    from MDAnalysis.coordinates.XDR import offsets_filename

    path.unlink(missing_ok=True)
    Path(offsets_filename(str(path))).unlink(missing_ok=True)
