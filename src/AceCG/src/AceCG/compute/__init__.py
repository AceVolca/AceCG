"""Compute subpackage — numerical kernels + MPI engine.

Keep MPI-facing exports lazy so ``python -m AceCG.compute.mpi_engine`` does not
see ``AceCG.compute.mpi_engine`` preloaded via package import.
"""

from __future__ import annotations

from .frame_geometry import FrameGeometry, compute_frame_geometry
from .energy import energy
from .force import force


__all__ = [
    # frame_geometry
    "FrameGeometry",
    "compute_frame_geometry",
    # energy / force
    "energy",
    "force",
    # mpi_engine
    "MPIComputeEngine",
    "build_default_engine",
]


def __getattr__(name: str):
    if name == "MPIComputeEngine":
        from .mpi_engine import MPIComputeEngine

        globals()[name] = MPIComputeEngine
        return MPIComputeEngine
    if name == "build_default_engine":
        from .registry import build_default_engine

        globals()[name] = build_default_engine
        return build_default_engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
