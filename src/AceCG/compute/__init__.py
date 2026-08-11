"""Compute subpackage — numerical kernels + MPI engine.

Keep MPI-facing exports lazy so ``python -m AceCG.compute.mpi_engine`` does not
see ``AceCG.compute.mpi_engine`` preloaded via package import.
"""

from __future__ import annotations

from .cgmap import CGMapper, MappedFrame
from .frame_geometry import FrameGeometry, compute_frame_geometry
from .energy import energy
from .force import force


__all__ = [
    # cgmap
    "CGMapper",
    "MappedFrame",
    # frame_geometry
    "FrameGeometry",
    "compute_frame_geometry",
    # energy / force
    "energy",
    "force",
    # force mapping (lazy: qpsolvers is an optional backend)
    "accumulate_force_map_statistics",
    "fit_force_map",
    # mpi_engine
    "MPIComputeEngine",
    "build_default_engine",
]


def __getattr__(name: str):
    if name in {
        "accumulate_force_map_statistics",
        "fit_force_map",
    }:
        module = __import__(f"{__name__}.force_mapping", fromlist=[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "MPIComputeEngine":
        from .mpi_engine import MPIComputeEngine

        globals()[name] = MPIComputeEngine
        return MPIComputeEngine
    if name == "build_default_engine":
        from .mpi_engine import build_default_engine

        globals()[name] = build_default_engine
        return build_default_engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
