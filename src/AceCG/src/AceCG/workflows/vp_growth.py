"""VP Growth workflow — build VP topology + latent settings + grown frames.

Top-level orchestrator for the Phase-3 VP pre-flight pipeline. Not a
:class:`BaseWorkflow` subclass: the VP grower has no training loop, no
frame cache, and no FM-style batch accumulation. It is a one-shot data
producer.

Pipeline
--------
1. Parse / load :class:`VPGrowthConfig`.
2. Pick the trajectory-loading strategy from the measured runtime
   heuristic: broadcast the full universe for 1-2 loaded segments,
   otherwise let each MPI rank open only the local segment subset it
   needs.
3. Rank 0 constructs :class:`VPGrower` via
   :meth:`VPGrower.from_universe`, determines the selected frame ids,
   and broadcasts either the shared universe or the shared template +
   segment metadata to all ranks.
4. Rank 0 writes ``vp_topology.data`` (empty frame so CDREM / CDFM
   have a schema file) and ``latent.settings`` + initial pair tables.
5. All ranks call :func:`grow_vp_frames`, which partitions the global
    frame-id list exactly like :mod:`AceCG.compute.mpi_engine` and uses
    :func:`iter_frames` on each rank's local slice.
6. Rank 0 writes ``manifest.json`` describing every frame path.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..compute.vp_prepare import VPGrowManifest, grow_vp_frames
from ..configs.vp_growth_config import VPGrowthConfig, parse_vp_growth_file
from ..io.logger import get_screen_logger
from ..io.trajectory import count_lammpstrj_frames_and_atoms
from ..topology.vpgrower import VPGrower, VPGrownFrame, write_vp_data
from ..io.vp_ffbuilder import write_latent_settings
from .base import (
    _apply_config_overrides,
    _build_workflow_cli_parser,
    _parse_cli_overrides,
)


_BROADCAST_SEGMENT_LIMIT = 2


@dataclass(frozen=True)
class VPGrowthResult:
    """Outcome of a :class:`VPGrowthWorkflow` run."""

    output_dir: Path
    universe_strategy: str
    latent_settings_path: Optional[Path]
    topology_path: Optional[Path]
    manifest_path: Optional[Path]
    manifest: Optional[VPGrowManifest]


class VPGrowthWorkflow:
    """One-shot VP growth driver."""

    def __init__(self, config: VPGrowthConfig, *, comm: Optional[Any] = None) -> None:
        self.config = config
        self.comm = comm

    def run(self) -> VPGrowthResult:
        """Run VP growth and return the written output summary."""
        import MDAnalysis as mda

        cfg = self.config
        comm = self.comm
        rank = 0 if comm is None else int(comm.Get_rank())
        size = 1 if comm is None else int(comm.Get_size())
        total_start = time.monotonic()

        base_dir = cfg.path.parent if cfg.path is not None else Path.cwd()
        output_dir = (base_dir / cfg.run.output_dir).resolve()
        ref_topo_path = (base_dir / cfg.aa_ref.ref_topo).resolve()
        traj_paths = [str((base_dir / t).resolve()) for t in cfg.aa_ref.trajectory_files]
        topology_format = _topology_format_for_path(ref_topo_path)
        universe_strategy = _choose_universe_loading_strategy(
            segment_count=len(traj_paths), size=size,
        )

        grower: Optional[VPGrower]
        universe = None
        template = None
        frame_ids: Optional[List[int]] = None
        segment_frame_counts: Optional[List[int]] = None
        t_topology_universe = 0.0
        t_trajectory_scan = 0.0
        t_universe = 0.0
        t_template = 0.0
        t_frame_select = 0.0

        if rank == 0:
            if universe_strategy == "broadcast":
                t_universe_start = time.monotonic()
                universe = mda.Universe(
                    str(ref_topo_path),
                    *traj_paths,
                    format=cfg.aa_ref.trajectory_format,
                    topology_format=topology_format,
                    atom_style="id resid type charge x y z",
                )
                t_universe = time.monotonic() - t_universe_start

                t_template_start = time.monotonic()
                grower = VPGrower.from_universe(
                    universe, cfg.vp, type_aliases=cfg.aa_ref.ref_topo_type_names,
                )
                template = grower.template
                t_template = time.monotonic() - t_template_start

                t_frame_select_start = time.monotonic()
                frame_ids = _resolve_frame_ids(universe, cfg)
                t_frame_select = time.monotonic() - t_frame_select_start
            else:
                t_topology_start = time.monotonic()
                topology_universe = mda.Universe(
                    str(ref_topo_path),
                    topology_format=topology_format,
                    atom_style="id resid type charge x y z",
                )
                t_topology_universe = time.monotonic() - t_topology_start

                t_template_start = time.monotonic()
                grower = VPGrower.from_universe(
                    topology_universe,
                    cfg.vp,
                    type_aliases=cfg.aa_ref.ref_topo_type_names,
                )
                template = grower.template
                t_template = time.monotonic() - t_template_start

                t_scan_start = time.monotonic()
                segment_frame_counts = []
                for traj_path in traj_paths:
                    n_frames, _ = count_lammpstrj_frames_and_atoms(Path(traj_path))
                    segment_frame_counts.append(int(n_frames))
                t_trajectory_scan = time.monotonic() - t_scan_start

                t_frame_select_start = time.monotonic()
                frame_ids = _resolve_frame_ids(sum(segment_frame_counts), cfg)
                t_frame_select = time.monotonic() - t_frame_select_start
        else:
            grower = None

        t_broadcast = 0.0
        if comm is not None and size > 1:
            t_broadcast_start = time.monotonic()
            if universe_strategy == "broadcast":
                universe, template, frame_ids = comm.bcast(
                    (universe, template, frame_ids) if rank == 0 else None,
                    root=0,
                )
            else:
                template, frame_ids, segment_frame_counts = comm.bcast(
                    (template, frame_ids, segment_frame_counts) if rank == 0 else None,
                    root=0,
                )
            t_broadcast = time.monotonic() - t_broadcast_start

        if template is None:
            raise RuntimeError("VPGrowthWorkflow.run() requires a shared VP template.")
        if frame_ids is None:
            raise RuntimeError("VPGrowthWorkflow.run() requires resolved frame ids.")
        if grower is None:
            grower = VPGrower(template)

        local_count, local_offset, local_ids = _partition_frame_ids(
            frame_ids, size=size, rank=rank,
        )
        local_frame_ids: Optional[List[int]] = None
        local_segment_ids: List[int] = []
        local_universe_load = 0.0

        if universe_strategy == "local_segments":
            if segment_frame_counts is None:
                raise RuntimeError(
                    "VPGrowthWorkflow.run() requires segment frame counts for local segment loading."
                )
            local_traj_paths, local_frame_ids, local_segment_ids = _select_local_trajectory_inputs(
                local_ids,
                traj_paths,
                segment_frame_counts,
            )
            if local_count > 0:
                t_local_universe_start = time.monotonic()
                universe = mda.Universe(
                    str(ref_topo_path),
                    *local_traj_paths,
                    format=cfg.aa_ref.trajectory_format,
                    topology_format=topology_format,
                    atom_style="id resid type charge x y z",
                )
                local_universe_load = time.monotonic() - t_local_universe_start
        else:
            local_universe_load = t_universe if rank == 0 else 0.0

        if local_count > 0 and universe is None:
            raise RuntimeError("VPGrowthWorkflow.run() requires a local Universe for assigned frames.")

        latent_path: Optional[Path] = None
        topology_path: Optional[Path] = None
        manifest_path: Optional[Path] = None
        t_schema_write = 0.0

        if rank == 0:
            t_schema_start = time.monotonic()
            output_dir.mkdir(parents=True, exist_ok=True)
            latent_path = write_latent_settings(
                template=template,
                vp_config=cfg.vp,
                output_dir=output_dir,
                table_points=cfg.run.table_points,
                table_rmin=cfg.run.table_rmin,
                table_rmax=cfg.run.table_rmax,
                include_name=cfg.run.latent_settings_name,
            )
            # Write a schema-only VP topology at zero positions.
            topology_path = output_dir / "vp_topology.data"
            zero = VPGrownFrame(
                positions=np.zeros((template.n_atoms, 3), dtype=np.float64),
                dimensions=np.array([1.0, 1.0, 1.0, 90.0, 90.0, 90.0]),
            )
            write_vp_data(template, zero, topology_path, title="VP topology (schema)")
            t_schema_write = time.monotonic() - t_schema_start
        if comm is not None:
            comm.Barrier()

        t_grow_start = time.monotonic()
        manifest = grow_vp_frames(
            grower=grower,
            universe=universe,
            frame_ids=frame_ids,
            local_frame_ids=local_frame_ids,
            output_dir=output_dir,
            orientation_seed_base=cfg.run.orientation_seed_base,
            include_forces=cfg.aa_ref.include_forces,
            overwrite=cfg.run.overwrite,
            comm=comm,
        )
        t_grow_local = time.monotonic() - t_grow_start

        local_growth_stats: Dict[str, Any] = {
            "rank": rank,
            "universe_strategy": universe_strategy,
            "frame_count": local_count,
            "frame_ids": [int(fid) for fid in local_ids],
            "segments_loaded": local_segment_ids,
            "universe_load_sec": local_universe_load,
            "broadcast_shared_context_sec": t_broadcast,
            "grow_elapsed_sec": t_grow_local,
        }
        if comm is None:
            gathered_growth_stats = [local_growth_stats]
        else:
            gathered_growth_stats = comm.gather(local_growth_stats, root=0)

        t_manifest_write = 0.0
        if rank == 0 and manifest is not None:
            t_manifest_start = time.monotonic()
            manifest_path = output_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "frames": {str(k): v for k, v in sorted(manifest.frames.items())},
                        "forces": {
                            str(k): v
                            for k, v in sorted(manifest.forces.items())
                        },
                    },
                    indent=2,
                )
            )
            t_manifest_write = time.monotonic() - t_manifest_start

            timing_payload = {
                "mpi": {
                    "enabled": comm is not None,
                    "size": size,
                    "universe_strategy": universe_strategy,
                    "rank_slices": gathered_growth_stats,
                },
                "phase_seconds": {
                    "topology_universe_load": t_topology_universe,
                    "trajectory_scan": t_trajectory_scan,
                    "universe_load": max(
                        float(item["universe_load_sec"]) for item in gathered_growth_stats
                    ),
                    "template_build": t_template,
                    "frame_selection": t_frame_select,
                    "broadcast_shared_context": max(
                        float(item["broadcast_shared_context_sec"])
                        for item in gathered_growth_stats
                    ),
                    "schema_write": t_schema_write,
                    "frame_growth_wall": max(
                        float(item["grow_elapsed_sec"]) for item in gathered_growth_stats
                    ),
                    "manifest_write": t_manifest_write,
                    "total": time.monotonic() - total_start,
                },
                "frames": {
                    "total_selected": len(frame_ids),
                    "first_frame_id": None if not frame_ids else int(frame_ids[0]),
                    "last_frame_id": None if not frame_ids else int(frame_ids[-1]),
                },
                "topology": {
                    "n_atoms": int(template.n_atoms),
                    "n_real": int(template.n_real),
                    "n_vp": int(template.n_vp),
                    "atomtype_order": cfg.vp.atomtype_order,
                },
            }
            (output_dir / "timing.json").write_text(json.dumps(timing_payload, indent=2))

        return VPGrowthResult(
            output_dir=output_dir,
            universe_strategy=universe_strategy,
            latent_settings_path=latent_path,
            topology_path=topology_path,
            manifest_path=manifest_path,
            manifest=manifest if rank == 0 else None,
        )


def _resolve_frame_ids(universe: Any, cfg: VPGrowthConfig) -> List[int]:
    """Produce the final list of frame ids.

    Precedence: explicit ``[vp] frame_ids`` > ``[aa_ref]`` window.
    """
    if cfg.run.frame_ids is not None:
        return [int(x) for x in cfg.run.frame_ids]
    if isinstance(universe, int):
        n_traj = int(universe)
    else:
        n_traj = len(universe.trajectory)
    start = int(cfg.aa_ref.skip_frames)
    step = max(1, int(cfg.aa_ref.every))
    requested = int(cfg.aa_ref.n_frames)
    end = n_traj if requested <= 0 else min(n_traj, start + requested * step)
    return list(range(start, end, step))


def _partition_frame_ids(
    frame_ids: List[int], *, size: int, rank: int,
) -> tuple[int, int, List[int]]:
    """Return this rank's contiguous balanced slice of ``frame_ids``."""
    base, rem = divmod(len(frame_ids), size)
    local_count = base + (1 if rank < rem else 0)
    local_offset = rank * base + min(rank, rem)
    return local_count, local_offset, frame_ids[local_offset : local_offset + local_count]


def _choose_universe_loading_strategy(*, segment_count: int, size: int) -> str:
    """Pick the measured trajectory-loading strategy for this MPI size."""
    if size <= 1 or segment_count <= _BROADCAST_SEGMENT_LIMIT:
        return "broadcast"
    return "local_segments"


def _select_local_trajectory_inputs(
    frame_ids: Sequence[int],
    traj_paths: Sequence[str],
    segment_frame_counts: Sequence[int],
) -> tuple[List[str], List[int], List[int]]:
    """Map a rank's global frame ids onto the minimal local trajectory subset."""
    if not frame_ids:
        return [], [], []

    boundaries = np.cumsum(np.asarray(segment_frame_counts, dtype=np.int64))
    included_segments: List[int] = []
    local_segment_offsets: Dict[int, int] = {}
    local_frame_ids: List[int] = []
    running_local_offset = 0

    for fid in frame_ids:
        frame_id = int(fid)
        seg_idx = int(np.searchsorted(boundaries, frame_id, side="right"))
        seg_start = 0 if seg_idx == 0 else int(boundaries[seg_idx - 1])
        if seg_idx not in local_segment_offsets:
            local_segment_offsets[seg_idx] = running_local_offset
            included_segments.append(seg_idx)
            running_local_offset += int(segment_frame_counts[seg_idx])
        local_frame_ids.append(local_segment_offsets[seg_idx] + frame_id - seg_start)

    local_traj_paths = [str(traj_paths[idx]) for idx in included_segments]
    one_based_segments = [int(idx + 1) for idx in included_segments]
    return local_traj_paths, local_frame_ids, one_based_segments


def _topology_format_for_path(path: Path) -> Optional[str]:
    """Infer the MDAnalysis topology format from the reference-topology path."""
    if path.suffix.lower() == ".data":
        return "DATA"
    return None


# ─── CLI ────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``acg-vpgrower`` entry point."""
    parser = _build_workflow_cli_parser(
        prog="acg-vpgrower",
        description="Grow VP atoms on a CG trajectory and emit LAMMPS data plus latent settings.",
    )
    parser.add_argument(
        "--no-mpi",
        action="store_true",
        help="Force serial mode even if mpi4py is importable.",
    )
    args, unknown = parser.parse_known_args(argv)
    overrides = _parse_cli_overrides(unknown)
    screen_logger = get_screen_logger("vp_growth", start_time=time.monotonic())

    comm = None
    mpi_reason: Optional[str] = None
    if not args.no_mpi:
        try:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            if comm.Get_size() == 1:
                comm = None
                mpi_reason = "single-rank MPI world"
        except ImportError:
            comm = None
            mpi_reason = "mpi4py not importable"

    cfg = parse_vp_growth_file(args.config) if args.config else VPGrowthConfig()
    cfg = _apply_config_overrides(cfg, overrides)
    workflow = VPGrowthWorkflow(cfg, comm=comm)
    result = workflow.run()

    rank = 0 if comm is None else int(comm.Get_rank())
    if rank == 0:
        output_path = result.output_dir / "acgreturn.pkl"
        with open(output_path, "wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if not args.no_mpi and comm is None and mpi_reason is not None:
            screen_logger.warning("running in serial (%s)", mpi_reason)
        screen_logger.info("universe_strategy = %s", result.universe_strategy)
        screen_logger.info("output_dir = %s", result.output_dir)
        if result.latent_settings_path is not None:
            screen_logger.info("latent = %s", result.latent_settings_path)
        if result.manifest_path is not None:
            n = 0 if result.manifest is None else len(result.manifest.frames)
            screen_logger.info("manifest = %s (%d frames)", result.manifest_path, n)
    return 0


def main_vp_prepare(argv: Optional[List[str]] = None) -> int:
    """Backward-compatible wrapper around :func:`main`."""
    return main(argv)


if __name__ == "__main__":
    sys.exit(main())
