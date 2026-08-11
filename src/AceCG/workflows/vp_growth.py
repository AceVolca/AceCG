"""VP Growth workflow: configure one concrete trajectory transformation."""

from __future__ import annotations

import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from ..configs.vp_growth_config import VPGrowthConfig, parse_vp_growth_file
from ..io.logger import get_screen_logger
from ..io.trajectory import MPITrajReader
from ..io.vp_growth import grow_vp_trajectory
from .base import (
    _apply_config_overrides,
    _build_workflow_cli_parser,
    _parse_cli_overrides,
)


_BROADCAST_SEGMENT_LIMIT = 2


@dataclass(frozen=True)
class VPGrowthResult:
    """Published outcome of a :class:`VPGrowthWorkflow` run."""

    output_dir: Path
    universe_strategy: str
    latent_settings_path: Path
    topology_path: Path
    timing_path: Path
    manifest_path: Path
    n_selected: int
    n_unique: int


class VPGrowthWorkflow:
    """Resolve VP paths and invoke the VP trajectory terminal once."""

    def __init__(self, config: VPGrowthConfig, *, comm: Optional[Any] = None) -> None:
        self.config = config
        self.comm = comm

    def run(self) -> VPGrowthResult:
        """Run VP growth and return its published summary."""
        base_dir = (
            self.config.path.parent
            if self.config.path is not None
            else Path.cwd()
        )
        output_dir = (base_dir / self.config.run.output_dir).resolve()
        reference_topology = (base_dir / self.config.aa_ref.ref_topo).resolve()
        trajectory_paths = [
            str((base_dir / path).resolve())
            for path in self.config.aa_ref.trajectory_files
        ]
        reader = MPITrajReader(
            trajectory_files=trajectory_paths,
            topology=reference_topology,
            trajectory_format=self.config.aa_ref.trajectory_format,
            universe_kwargs={"atom_style": "id resid type charge x y z"},
            comm=self.comm,
            strategy="auto",
            broadcast_segment_limit=_BROADCAST_SEGMENT_LIMIT,
        )
        payload = grow_vp_trajectory(
            config=self.config,
            reader=reader,
            output_dir=output_dir,
            reference_topology=reference_topology,
            comm=self.comm,
        )
        return VPGrowthResult(
            output_dir=payload["output_dir"],
            universe_strategy=reader.strategy,
            latent_settings_path=payload["latent_settings_path"],
            topology_path=payload["topology_path"],
            timing_path=payload["timing_path"],
            manifest_path=payload["manifest_path"],
            n_selected=int(payload["n_selected"]),
            n_unique=int(payload["n_unique"]),
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``acg-vpgrower`` entry point."""
    parser = _build_workflow_cli_parser(
        prog="acg-vpgrower",
        description=(
            "Grow VP atoms on a CG trajectory and emit LAMMPS data plus "
            "latent settings."
        ),
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
            mpi_reason = "mpi4py not importable"

    config = parse_vp_growth_file(args.config) if args.config else VPGrowthConfig()
    config = _apply_config_overrides(config, overrides)
    result = VPGrowthWorkflow(config, comm=comm).run()

    rank = 0 if comm is None else int(comm.Get_rank())
    if rank == 0:
        with (result.output_dir / "acgreturn.pkl").open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if not args.no_mpi and comm is None and mpi_reason is not None:
            screen_logger.warning("running in serial (%s)", mpi_reason)
        screen_logger.info("universe_strategy = %s", result.universe_strategy)
        screen_logger.info("output_dir = %s", result.output_dir)
        screen_logger.info("latent = %s", result.latent_settings_path)
        screen_logger.info(
            "manifest = %s (%d occurrences, %d physical frames)",
            result.manifest_path,
            result.n_selected,
            result.n_unique,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
