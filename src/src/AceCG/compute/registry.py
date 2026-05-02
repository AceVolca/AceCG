"""Compute registry — builders for pre-registered MPIComputeEngine instances."""

from __future__ import annotations

from .energy import energy
from .force import force
from .mpi_engine import MPIComputeEngine


def build_default_engine(*, comm=None) -> MPIComputeEngine:
    """Create the default MPI compute engine.

    Parameters
    ----------
    comm : object, optional
        MPI communicator to pass to :class:`MPIComputeEngine`. ``None`` lets the
        engine use its default communicator behavior.

    Returns
    -------
    MPIComputeEngine
        Engine pre-registered with energy, energy-gradient, energy-Hessian,
        force, force-gradient, and force-matching statistics functions.
    """
    engine = MPIComputeEngine(serial_threshold=10, comm=comm)

    # --- Energy-side registrations ---

    engine.register(
        name="energy_grad",
        fn=lambda geom, ff, **kw: energy(geom, ff, return_grad=True, **kw)["energy_grad"],
        reduce="gather",
        description="Per-frame energy gradient dU/dtheta",
    )

    engine.register(
        name="energy_value",
        fn=lambda geom, ff, **kw: energy(geom, ff, return_value=True, **kw)["energy"],
        reduce="gather",
        description="Per-frame scalar energy",
    )

    engine.register(
        name="energy_hessian",
        fn=lambda geom, ff, **kw: energy(geom, ff, return_hessian=True, **kw)["energy_hessian"],
        reduce="gather",
        description="Per-frame true energy parameter Hessian d²U/dθ²",
    )

    engine.register(
        name="energy_grad_outer",
        fn=lambda geom, ff, **kw: energy(geom, ff, return_grad_outer=True, **kw)["energy_grad_outer"],
        reduce="gather",
        description="Per-frame gradient outer product (dU/dθ_j)(dU/dθ_k)",
    )

    # --- Force-side registrations ---

    engine.register(
        name="force_grad",
        fn=lambda geom, ff, **kw: force(geom, ff, return_grad=True, **kw)["force_grad"],
        reduce="gather",
        description="Per-frame force Jacobian df/dtheta",
    )

    engine.register(
        name="force_value",
        fn=lambda geom, ff, **kw: force(geom, ff, return_value=True, **kw)["force"],
        reduce="gather",
        description="Per-frame model forces",
    )

    engine.register(
        name="fm_stats",
        fn=lambda geom, ff, **kw: force(geom, ff, return_fm_stats=True, **kw)["fm_stats"],
        reduce="dict_sum",
        description="Per-frame force-matching sufficient statistics",
    )

    return engine
