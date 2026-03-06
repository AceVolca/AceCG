# AceCG/trainers/utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List
import os

import numpy as np
import MDAnalysis as mda
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..utils.neighbor import Pair2DistanceByFrame, combine_Pair2DistanceByFrame
from ..utils.trjio import split_lammpstrj


def optimizer_accepts_hessian(optimizer) -> bool:
    """
    Check whether the optimizer's `step` method accepts a 'hessian' argument.

    Parameters
    ----------
    optimizer : object
        An optimizer instance with a `.step()` method.

    Returns
    -------
    bool
        True if 'hessian' is a parameter of the .step() method, else False.
    """
    return hasattr(optimizer, 'step') and 'hessian' in optimizer.step.__code__.co_varnames


def _worker_pair2dist(
    topology: Optional[str],
    trj_path: str,
    cutoff: float,
    pair2potential: Dict[Tuple[str, str], Any],
    sel: str,
    nstlist: int,
    exclude: bool,
) -> Dict[int, Dict[Tuple[str, str], np.ndarray]]:
    """
    Worker: load a per-chunk Universe and compute Pair2DistanceByFrame for all frames in that chunk.
    The returned dict uses *local* frame indices (0..n_frames-1) within the chunk.
    """
    if topology is None:
        u = mda.Universe(trj_path, format="LAMMPSDUMP")
    else:
        u = mda.Universe(topology, trj_path, format="LAMMPSDUMP")

    return Pair2DistanceByFrame(
        u,
        start=0,
        end=len(u.trajectory),
        cutoff=cutoff,
        pair2potential=pair2potential,
        sel=sel,
        nstlist=nstlist,
        exclude=exclude,
    )


def prepare_Trainer_data(u, pair2potential, start, end, cutoff, sel="all", nstlist=10, exclude=True, weight=None):
    """
    Extracts pairwise distances from trajectory into Trainer-compatible format.

    Parameters
    ----------
    u : MDAnalysis.Universe
        Input simulation trajectory.
    pair2potential : dict
        Mapping of (type1, type2) to BasePotential objects.
    start : int
        Starting frame index (inclusive).
    end : int
        Ending frame index (exclusive).
    cutoff : float
        Global neighbor cutoff.
    sel : str, optional
        Atom selection string. Default is 'all'.
    nstlist : int, optional
        Neighbor list update interval.
    exclude : bool, optional
        Whether to exclude bonded neighbors. Default is True.
    weight : np.ndarray, optional
        Assign weight for each frame to obtain a reweighted ensemble
        Length-n_frames weight array. If None, uniform average is used.

    Returns
    -------
    data : dict
        Dictionary with keys:
        - 'dist': frame-indexed pairwise distance dictionary
        - 'weight': None (uniform weighting)
    """
    dist = Pair2DistanceByFrame(
        u, start=start, end=end, cutoff=cutoff,
        pair2potential=pair2potential, sel=sel,
        nstlist=nstlist, exclude=exclude
    )

    return {"dist": dist, "weight": weight}


def prepare_Trainer_data_parallel(
    *,
    # Either provide an already-loaded Universe OR provide topology+trajectory
    u: Optional[mda.Universe] = None,
    topology: Optional[str | Path] = None,
    trajectory: Optional[str | Path] = None,
    # Same parameters as prepare_Trainer_data
    pair2potential: Dict[Tuple[str, str], Any],
    start: int,
    end: int,
    cutoff: float,
    sel: str = "all",
    nstlist: int = 10,
    exclude: bool = True,
    weight: Optional[np.ndarray] = None,
    # Parallel controls
    n_parts: int = 8,
    n_workers: Optional[int] = None,
    chunk_dir: str | Path = "traj_chunks",
    chunk_prefix: str = "chunk",
    keep_chunks: bool = True,
) -> Dict[str, Any]:
    """
    Parallel version of prepare_Trainer_data: split trajectory into chunks, compute Pair2DistanceByFrame
    per chunk in parallel, then combine back into a single continuous-frame dictionary.

    Parameters
    ----------
    u : MDAnalysis.Universe, optional
        If provided, must contain the trajectory. This is used only to locate the trajectory path is NOT
        always reliable for MDAnalysis in-memory readers; recommended path-based use below.
    topology : str | Path, optional
        Topology file (e.g., LAMMPS data / PDB). If None, we try to load LAMMPSDUMP without topology.
    trajectory : str | Path, optional
        Input lammpstrj path. Required unless you pass a Universe *and* can resolve its filename.
    pair2potential, start, end, cutoff, sel, nstlist, exclude, weight
        Same meaning as in prepare_Trainer_data.
        Note: (start, end) apply to the *original* trajectory frame indices.
    n_parts : int
        How many chunks to split into (approximately equal frames).
    n_workers : int, optional
        Process count. Default: os.cpu_count().
    chunk_dir : str | Path
        Directory to write temporary chunk trajectories.
    chunk_prefix : str
        Prefix for chunk files.
    keep_chunks : bool
        If False, delete chunk files after combining.

    Returns
    -------
    data : dict
        {
          "dist": {global_frame: {pair: distances}},
          "weight": weight_slice_or_none
        }

    Important Notes
    ---------------
    - Uses multiprocessing (ProcessPoolExecutor) for MDAnalysis safety.
    - Each chunk is processed with local frame indices starting at 0; we then reindex continuously
      using combine_Pair2DistanceByFrame.
    - Neighbor list updates:
        * If nstlist>0, neighbor list is updated within each chunk independently.
        * If nstlist==0, neighbor list is computed once per chunk at chunk frame 0.
      This can create small boundary effects only if you intended a *single* neighborlist state
      carried across chunks (rarely needed; usually fine).
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    # Resolve trajectory path
    if trajectory is None:
        if u is None:
            raise ValueError("Provide `trajectory` path, or provide `u` (Universe).")
        # Best-effort: MDAnalysis may or may not expose the filename depending on reader
        try:
            trajectory = Path(u.trajectory.filename)  # type: ignore[attr-defined]
        except Exception as e:
            raise ValueError(
                "Could not resolve trajectory path from Universe. Please pass `trajectory=...` explicitly."
            ) from e

    trajectory = Path(trajectory)
    if not trajectory.exists():
        raise FileNotFoundError(trajectory)

    topo_str: Optional[str] = str(topology) if topology is not None else None

    # Build weights slice for [start:end)
    if weight is not None:
        if len(weight) < end:
            raise ValueError(f"weight length {len(weight)} < end {end}")
        weight_out = np.asarray(weight[start:end])
    else:
        weight_out = None

    # Split whole trajectory into chunks first, then we will only use frames that overlap [start, end)
    chunk_dir = Path(chunk_dir)
    chunk_paths = split_lammpstrj(
        trj_path=trajectory,
        n_parts=n_parts,
        out_dir=chunk_dir,
        prefix=chunk_prefix,
        digits=3,
        dry_run=False,
    )

    # We need to know which chunk contains which global frame range.
    # split_lammpstrj splits by frame count evenly; we can reconstruct ranges by counting frames per chunk.
    # To avoid rescanning the big file, scan each chunk quickly with MDAnalysis length (cheap per chunk).
    chunk_meta: List[Tuple[int, int, Path]] = []  # (global_start, global_end, path)
    gcur = 0
    for p in chunk_paths:
        uu = mda.Universe(str(p), format="LAMMPSDUMP") if topo_str is None else mda.Universe(topo_str, str(p), format="LAMMPSDUMP")
        nfr = len(uu.trajectory)
        chunk_meta.append((gcur, gcur + nfr, p))
        gcur += nfr

    # Select chunks that overlap [start, end)
    use_chunks: List[Tuple[int, int, Path]] = []
    for g0, g1, p in chunk_meta:
        if g1 <= start or g0 >= end:
            continue
        use_chunks.append((g0, g1, p))

    # Run parallel workers on selected chunks (but note: each worker computes ALL frames in that chunk)
    results_by_chunk_start: Dict[int, Dict[int, Dict[Tuple[str, str], np.ndarray]]] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {}
        for g0, g1, p in use_chunks:
            fut = ex.submit(
                _worker_pair2dist,
                topo_str,
                str(p),
                cutoff,
                pair2potential,
                sel,
                nstlist,
                exclude,
            )
            futs[fut] = (g0, g1, p)

        for fut in as_completed(futs):
            g0, g1, p = futs[fut]
            d_local = fut.result()
            results_by_chunk_start[g0] = d_local

    # Now crop each chunk result to only the frames that overlap [start, end),
    # and then combine in the correct order.
    dicts_to_combine = []
    global_start_for_combine = start

    # Sort chunks by global start
    for g0, g1, p in sorted(use_chunks, key=lambda x: x[0]):
        d_local = results_by_chunk_start[g0]
        # local frames correspond to global frames [g0, g1) => local i maps to global (g0+i)
        lo = max(start, g0) - g0
        hi = min(end, g1) - g0

        # Extract local frames in [lo, hi)
        d_slice = {i: d_local[i] for i in range(lo, hi)}
        dicts_to_combine.append(d_slice)

    dist = combine_Pair2DistanceByFrame(dicts_to_combine, start_frame=global_start_for_combine)

    # Optionally cleanup chunk files
    if not keep_chunks:
        for _, _, p in chunk_meta:
            try:
                p.unlink()
            except Exception:
                pass
        try:
            chunk_dir.rmdir()
        except Exception:
            pass

    return {"dist": dist, "weight": weight_out}