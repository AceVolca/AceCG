"""Trajectory mapping workflow — turn an AA trajectory into a CG trajectory.

Top-level orchestrator for ``acg-trajmap``. Like :mod:`AceCG.workflows.vp_growth`
this is a one-shot data producer, not a :class:`BaseWorkflow` subclass: there is
no training loop, no frame cache, and no forcefield.

Pipeline
--------
1. Parse :class:`~AceCG.configs.trajmap_config.TrajMapConfig`.
2. Rank 0 opens the AA *topology* alone (cheap), compiles the mapping YAML into a
   :class:`~AceCG.topology.cgmap.CGMapSpec`, and validates it against the real
   atom count and masses.
3. :class:`~AceCG.io.trajectory.MPITrajReader` scans the trajectory once on
   rank 0 and broadcasts the plan — frame count, selected frame ids, per-segment
   frame counts, XDR frame offsets — alongside the spec. No rank repeats the
   offset scan; that matters for a large read-only TRR, where MDAnalysis cannot
   cache offsets next to the file and silently falls back to rescanning per open.
4. The reader hands each rank a contiguous balanced slice, so a rank's output
   segment is a contiguous piece of the trajectory and the segments concatenate
   in rank order. Its ``"auto"`` strategy then decides what each rank opens: the
   whole chain with the broadcast offsets installed for one or two segments, or
   only the segments that rank's slice touches — with only those segments'
   offsets — once the input is a long list.
5. Each rank maps its slice with one :class:`~AceCG.compute.cgmap.CGMapper` and
   streams it straight into its own segment file. Nothing accumulates in memory:
   peak use is one AA frame plus the mapper's scratch.
6. Rank 0 writes the CG topology (LAMMPS ``data``, optional ``gro``) from the
   first selected frame, optionally concatenates the segments, and writes a JSON
   report.

Why the segments are per rank
-----------------------------
Gathering mapped frames to rank 0 would serialize the write and put the whole
trajectory through one process. Contiguous per-rank segments let every rank write
in parallel, and because AceCG's own configs accept a *list* of trajectory
segments, the unmerged form is directly consumable — ``merge_segments`` is a
convenience, not a requirement.
"""

from __future__ import annotations

import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..compute.force_mapping import accumulate_force_map_statistics, fit_force_map
from ..configs.trajmap_config import TrajMapConfig, parse_trajmap_file
from ..io.logger import get_screen_logger
from ..io.trajmap import map_cg_trajectory
from ..io.trajectory import (
    MPITrajReader,
    format_for_path,
    open_universe,
)
from ..topology.cgmap import CGMapSpec, load_cgmap_spec, load_mapping_yaml
from .base import (
    _apply_config_overrides,
    _build_workflow_cli_parser,
    _parse_cli_overrides,
)


__all__ = ["TrajMapResult", "TrajMapWorkflow", "main"]


@dataclass(frozen=True)
class TrajMapResult:
    """Outcome of a :class:`TrajMapWorkflow` run.

    Attributes
    ----------
    output_dir
        Run directory holding every product.
    trajectory_path
        Merged CG trajectory, or ``None`` when the segments were left unmerged.
    segment_paths
        Per-rank segments in trajectory order. Empty after a merge that removed
        them.
    topology_path, gro_path
        CG topology files written from the first mapped frame.
    aliases_path
        ``{lammps type id: bead name}`` JSON beside the data file, in the form
        AceCG configs consume as ``ref_topo_type_names``.
    report_path
        JSON run report.
    n_frames, n_sites
        Frames written and CG sites per frame.
    """

    output_dir: Path
    trajectory_path: Optional[Path]
    segment_paths: Tuple[Path, ...]
    topology_path: Optional[Path]
    gro_path: Optional[Path]
    aliases_path: Optional[Path]
    report_path: Optional[Path]
    force_map_path: Optional[Path]
    n_frames: int
    n_sites: int


class TrajMapWorkflow:
    """One-shot AA→CG trajectory mapping driver."""

    def __init__(self, config: TrajMapConfig, *, comm: Optional[Any] = None) -> None:
        self.config = config
        self.comm = comm

    # ── driver ────────────────────────────────────────────────────

    def run(self) -> TrajMapResult:
        """Compile and scan input, then invoke the concrete TrajMap terminal."""
        cfg = self.config
        comm = self.comm
        total_start = time.monotonic()
        base_dir = cfg.path.parent if cfg.path is not None else Path.cwd()
        output_dir = (base_dir / cfg.run.output_dir).resolve()
        topology_path = None if cfg.aa.topology is None else (base_dir / cfg.aa.topology).resolve()
        traj_paths = [str((base_dir / path).resolve()) for path in cfg.aa.trajectory_files]
        map_path = (base_dir / cfg.mapping.map_file).resolve()
        reader = MPITrajReader(
            trajectory_files=traj_paths, topology=self._trajectory_reader_topology(topology_path, traj_paths),
            trajectory_format=cfg.aa.trajectory_format, topology_format=cfg.aa.topology_format,
            comm=comm, strategy="auto",
        )
        t0 = time.monotonic()
        spec, aa_masses_used = self._compile_spec(topology_path, traj_paths, map_path)
        t_spec = time.monotonic() - t0
        t0 = time.monotonic()
        plan = reader.scan(start=cfg.aa.skip_frames, every=cfg.aa.every, n_frames=cfg.aa.n_frames,
                           frame_ids=cfg.aa.frame_ids)
        t_scan = time.monotonic() - t0
        force_mapping_report: Optional[Dict[str, Any]] = None
        t_force_fit = 0.0
        if cfg.force_mapping.enabled:
            t0 = time.monotonic()
            statistics = accumulate_force_map_statistics(
                config=cfg, spec=spec, reader=reader, plan=plan,
                topology_path=topology_path, comm=comm,
            )
            spec, force_mapping_report = fit_force_map(
                config=cfg, spec=spec, statistics=statistics, comm=comm,
            )
            t_force_fit = time.monotonic() - t0

        mapped = map_cg_trajectory(
            config=cfg, spec=spec, reader=reader, plan=plan, output_dir=output_dir,
            topology_path=topology_path, trajectory_paths=traj_paths, map_path=map_path,
            aa_masses_used=aa_masses_used,
            force_mapping_report=force_mapping_report,
            phase_seconds={"spec_compile": t_spec, "trajectory_scan": t_scan, "force_map_fit": t_force_fit},
            total_start=total_start, comm=comm,
        )
        return TrajMapResult(
            output_dir=output_dir, trajectory_path=mapped["trajectory_path"], segment_paths=tuple(mapped["segment_paths"]),
            topology_path=mapped["topology_path"], gro_path=mapped["gro_path"], aliases_path=mapped["aliases_path"],
            report_path=mapped["report_path"], force_map_path=mapped["force_map_path"],
            n_frames=int(mapped["n_frames"]), n_sites=int(spec.n_sites),
        )

    # ── stages ────────────────────────────────────────────────────

    def _compile_spec(
        self,
        topology_path: Optional[Path],
        traj_paths: Sequence[str],
        map_path: Path,
    ) -> Tuple[CGMapSpec, bool]:
        """Compile the mapping YAML against the real AA atom count.

        Returns the spec and whether topology masses were needed for the
        centre-of-mass fallback.
        """
        cfg = self.config
        rank = 0 if self.comm is None else int(self.comm.Get_rank())
        masses = None
        n_atoms = None
        cg_topology = None
        root_error = None
        if rank == 0:
            try:
                reader_topology = self._trajectory_reader_topology(topology_path, traj_paths)
                if topology_path is None or reader_topology is None:
                    n_atoms = int(_trajectory_n_atoms(traj_paths[0]))
                else:
                    topology_universe = open_universe(
                        topology_path, topology_format=cfg.aa.topology_format
                    )
                    n_atoms = int(topology_universe.atoms.n_atoms)
                if topology_path is not None and cfg.mapping.use_topology_masses:
                    if reader_topology is None:
                        topology_universe = open_universe(
                            topology_path, topology_format=cfg.aa.topology_format
                        )
                        if int(topology_universe.atoms.n_atoms) != n_atoms:
                            raise ValueError(
                                "AA topology and trajectory have different atom counts: "
                                f"{topology_universe.atoms.n_atoms} and {n_atoms}."
                            )
                    # A topology that carries no masses at all is a legitimate
                    # input — MDAnalysis signals it with NoDataError, which
                    # subclasses AttributeError, so `getattr` covers exactly
                    # that case. Any other failure is a real one and raises:
                    # these masses become the centre-of-mass x-weights, so
                    # silently dropping them changes where every CG site sits.
                    raw_masses = getattr(topology_universe.atoms, "masses", None)
                    masses = (
                        None
                        if raw_masses is None
                        else np.asarray(raw_masses, dtype=np.float64)
                    )
                if cfg.mapping.cg_topology is not None:
                    if cfg.mapping.cg_topology == "":
                        cg_topology = {}
                    else:
                        block_path = (map_path.parent / cfg.mapping.cg_topology).resolve()
                        block = load_mapping_yaml(block_path)
                        cg_topology = dict(block.get("cg-topology", block))
            except Exception as exc:
                root_error = exc

        spec = load_cgmap_spec(
            map_path,
            index_base=cfg.mapping.index_base,
            masses=masses,
            n_atoms=n_atoms,
            mol_reference=cfg.mapping.mol_reference,
            cg_topology=cg_topology,
            strict_weights=cfg.mapping.strict_weights,
            comm=self.comm,
            root_error=root_error,
        )
        return spec, masses is not None

    def _trajectory_reader_topology(
        self, topology_path: Optional[Path], traj_paths: Sequence[str]
    ) -> Optional[Path]:
        """Avoid reparsing a huge topology on every rank for self-describing XDR."""
        trajectory_format = format_for_path(
            traj_paths[0],
            explicit_format=self.config.aa.trajectory_format,
        )
        if trajectory_format in {"XTC", "TRR"}:
            return None
        return topology_path


# ─── helpers ────────────────────────────────────────────────────────


def _trajectory_n_atoms(path: str) -> int:
    """Atom count straight from a trajectory that stores it (XTC, TRR, …)."""
    import MDAnalysis as mda

    try:
        return int(mda.coordinates.core.get_reader_for(path).parse_n_atoms(path))
    except (AttributeError, NotImplementedError, ValueError) as exc:
        raise ValueError(
            f"cannot read an atom count from {path}, so [aa] topology is required "
            f"for this trajectory format ({exc})."
        ) from None


# ─── CLI ────────────────────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``acg-trajmap`` entry point."""
    parser = _build_workflow_cli_parser(
        prog="acg-trajmap",
        description=(
            "Map an all-atom trajectory onto CG sites with an OpenMSCG-format "
            "mapping YAML, in parallel over MPI."
        ),
    )
    parser.add_argument(
        "--no-mpi",
        action="store_true",
        help="Force serial mode even if mpi4py is importable.",
    )
    args, unknown = parser.parse_known_args(argv)
    overrides = _parse_cli_overrides(unknown)
    screen_logger = get_screen_logger("trajmap", start_time=time.monotonic())

    if not args.config:
        parser.error("acg-trajmap requires a trajmap .acg config file.")

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

    cfg = parse_trajmap_file(args.config)
    cfg = _apply_config_overrides(cfg, overrides)
    result = TrajMapWorkflow(cfg, comm=comm).run()

    rank = 0 if comm is None else int(comm.Get_rank())
    if rank == 0:
        with open(result.output_dir / "acgreturn.pkl", "wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if not args.no_mpi and comm is None and mpi_reason is not None:
            screen_logger.warning("running in serial (%s)", mpi_reason)
        screen_logger.info("output_dir = %s", result.output_dir)
        screen_logger.info("frames = %d, sites = %d", result.n_frames, result.n_sites)
        if result.trajectory_path is not None:
            screen_logger.info("trajectory = %s", result.trajectory_path)
        else:
            screen_logger.info("segments = %d files", len(result.segment_paths))
        if result.topology_path is not None:
            screen_logger.info("topology = %s", result.topology_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
