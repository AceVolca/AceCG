"""MPI-parallel VP frame growth.

Distributes conditioning-frame iteration across ranks, grows VP atoms
per frame via :class:`AceCG.topology.vpgrower.VPGrower`, and writes a
LAMMPS ``*.data`` file (plus optional forces) per grown frame.

Partition strategy
------------------
Identical to :meth:`AceCG.compute.mpi_engine.MPIComputeEngine.run_post`
for discrete frame-id lists:

    base, rem = divmod(n_frames, size)
    local_count = base + (1 if rank < rem else 0)
    local_offset = rank * base + min(rank, rem)

The first ``rem`` ranks each own one extra frame so the partition is
balanced within one frame across any ``size``.

Output convention
-----------------
For every frame id ``fid`` the grower writes

* ``<output_dir>/frame_{fid:06d}.data``
* ``<output_dir>/frame_{fid:06d}.forces.npy`` (float32, shape
  ``(n_real, 3)``) if ``include_forces`` is set.

Rank 0 returns the merged ``{frame_id: path}`` manifest; other ranks
return ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import warnings

import numpy as np

from ..io.trajectory import iter_frames
from ..topology.vpgrower import VPGrower, write_vp_data


@dataclass(frozen=True)
class VPGrowManifest:
    """Mapping ``{frame_id: frame_data_path}`` plus optional forces path."""

    frames: Dict[int, str]
    forces: Dict[int, Optional[str]]


def grow_vp_frames(
    *,
    grower: VPGrower,
    universe: Any,
    frame_ids: Sequence[int],
    local_frame_ids: Optional[Sequence[int]] = None,
    output_dir: str | Path,
    orientation_seed_base: int,
    include_forces: bool = False,
    overwrite: bool = False,
    comm: Optional[Any] = None,
) -> Optional[VPGrowManifest]:
    """Grow VP atoms on every frame in ``frame_ids`` in parallel.

    Parameters
    ----------
    grower
        Pre-built :class:`VPGrower` (shared across ranks; the template
        carries only int/float/ndarray state so it is safe to hold on
        every rank).
    universe
        Local :class:`MDAnalysis.Universe` on every rank. Each rank
        must be able to seek into the trajectory independently.
    frame_ids
        Conditioning frame ids to grow. Global list; every rank sees
        the same list and partitions it internally.
    local_frame_ids
        Optional local-universe frame indices for this rank's assigned
        slice of ``frame_ids``. When supplied, filenames and RNG seeds
        still use the global frame ids from ``frame_ids``, but the
        trajectory seeks use these local frame indices instead.
    output_dir
        Directory to write ``frame_{fid:06d}.data`` (and optionally
        ``.forces.npy``).
    orientation_seed_base
        RNG seed base. Per-frame seed is ``orientation_seed_base + fid``.
    include_forces
        When ``True`` and the underlying trajectory provides forces,
        write a ``.forces.npy`` alongside each data file.
    overwrite
        When ``True``, existing per-frame files are silently replaced;
        otherwise a :class:`FileExistsError` is raised on first clash.
    comm
        Optional :class:`mpi4py.MPI.Intracomm`. ``None`` ⇒ single-rank
        mode (rank 0 with size 1).
    """
    frame_ids = [int(fid) for fid in frame_ids]
    n_frames = len(frame_ids)

    if comm is None:
        rank, size = 0, 1
    else:
        rank = int(comm.Get_rank())
        size = int(comm.Get_size())

    output_dir = Path(output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if comm is not None:
        comm.Barrier()

    # Contiguous partition of ``frame_ids`` matching run_post's split.
    base, rem = divmod(n_frames, size)
    local_count = base + (1 if rank < rem else 0)
    local_offset = rank * base + min(rank, rem)
    local_output_ids = frame_ids[local_offset : local_offset + local_count]
    if local_frame_ids is None:
        local_iter_ids = [int(fid) for fid in local_output_ids]
    else:
        local_iter_ids = [int(fid) for fid in local_frame_ids]
        if len(local_iter_ids) != local_count:
            raise ValueError(
                f"local_frame_ids has {len(local_iter_ids)} entries but rank {rank} "
                f"owns {local_count} output frames."
            )

    local_frames: Dict[int, str] = {}
    local_forces: Dict[int, Optional[str]] = {}

    if local_count == 0:
        local_records = _gather_records(local_frames, local_forces, comm, rank)
        return _merge_records(local_records) if rank == 0 else None

    if include_forces and not bool(getattr(universe.trajectory.ts, "has_forces", False)):
        warnings.warn(
            "Trajectory is missing force columns fx/fy/fz; continuing without force output.",
            stacklevel=2,
        )
        include_forces = False

    real_indices = grower.template.real_indices
    n_real = grower.template.n_real
    n_processed = 0
    for output_fid, (_, pos, box, forces) in zip(
        local_output_ids,
        iter_frames(universe, frame_ids=local_iter_ids, include_forces=include_forces),
    ):
        n_processed += 1
        data_path = output_dir / f"frame_{int(output_fid):06d}.data"
        if data_path.exists() and not overwrite:
            raise FileExistsError(str(data_path))

        # ``pos`` from a CG-only universe is already in real-bead order,
        # matching ``template.real_indices``. No VP-slot indexing needed.
        if pos.shape[0] != n_real:
            raise ValueError(
                f"Trajectory has {pos.shape[0]} atoms; template expects "
                f"{n_real} real beads."
            )
        grown = grower.grow_frame(
            pos, box, orientation_seed=orientation_seed_base + int(output_fid)
        )
        write_vp_data(grower.template, grown, data_path)
        local_frames[int(output_fid)] = str(data_path)

        if include_forces and forces is not None:
            forces_arr = np.asarray(forces, dtype=np.float32).reshape(n_real, 3)
            forces_path = output_dir / f"frame_{int(output_fid):06d}.forces.npy"
            if forces_path.exists() and not overwrite:
                raise FileExistsError(str(forces_path))
            np.save(forces_path, forces_arr)
            local_forces[int(output_fid)] = str(forces_path)
        else:
            local_forces[int(output_fid)] = None

    if n_processed != local_count:
        raise RuntimeError(
            f"Expected to grow {local_count} frames on rank {rank}, got {n_processed}."
        )

    local_records = _gather_records(local_frames, local_forces, comm, rank)
    return _merge_records(local_records) if rank == 0 else None


def _gather_records(
    frames: Dict[int, str],
    forces: Dict[int, Optional[str]],
    comm: Optional[Any],
    rank: int,
):
    if comm is None:
        return [(frames, forces)]
    return comm.gather((frames, forces), root=0)


def _merge_records(records) -> VPGrowManifest:
    all_frames: Dict[int, str] = {}
    all_forces: Dict[int, Optional[str]] = {}
    for frames, forces in records:
        all_frames.update(frames)
        all_forces.update(forces)
    return VPGrowManifest(frames=all_frames, forces=all_forces)
