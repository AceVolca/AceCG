"""
AceCG: A Python package for bottom-up coarse-graining.

The top-level package keeps historical convenience exports lazy. Importing
``AceCG`` should not initialize trainers, schedulers, MPI engines, or optional
accelerators.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Trainers
    "REMTrainerAnalytic": (".trainers.analytic", "REMTrainerAnalytic"),
    "load_reweighted_rem_stacks": (".trainers.analytic", "load_reweighted_rem_stacks"),
    "MSETrainerAnalytic": (".trainers.analytic", "MSETrainerAnalytic"),
    "load_reweighted_mse_stacks": (".trainers.analytic", "load_reweighted_mse_stacks"),
    "CDREMTrainerAnalytic": (".trainers.analytic", "CDREMTrainerAnalytic"),
    "MultiTrainerAnalytic": (".trainers.analytic", "MultiTrainerAnalytic"),
    "FMTrainerAnalytic": (".trainers.analytic", "FMTrainerAnalytic"),
    "CDFMTrainerAnalytic": (".trainers.analytic", "CDFMTrainerAnalytic"),
    "L0InteractionTrainerAnalytic": (".trainers.analytic", "L0InteractionTrainerAnalytic"),
    # Optimizers
    "BaseOptimizer": (".optimizers.base", "BaseOptimizer"),
    "NewtonRaphsonOptimizer": (".optimizers.newton_raphson", "NewtonRaphsonOptimizer"),
    "AdamMaskedOptimizer": (".optimizers.adam", "AdamMaskedOptimizer"),
    "AdamWMaskedOptimizer": (".optimizers.adamW", "AdamWMaskedOptimizer"),
    "RMSpropMaskedOptimizer": (".optimizers.rmsprop", "RMSpropMaskedOptimizer"),
    # Solvers
    "BaseSolver": (".solvers.base", "BaseSolver"),
    "FMMatrixSolver": (".solvers.fm_matrix", "FMMatrixSolver"),
    # I/O
    "ReadLmpFF": (".io.forcefield", "ReadLmpFF"),
    "ReadLmpFFBounds": (".io.forcefield", "ReadLmpFFBounds"),
    "ReadLmpFFMask": (".io.forcefield", "ReadLmpFFMask"),
    "write_lammps_table": (".io.forcefield", "write_lammps_table"),
    "WriteLmpFF": (".io.forcefield", "WriteLmpFF"),
    # Compute
    "build_default_engine": (".compute.mpi_engine", "build_default_engine"),
    # Potentials
    "MultiGaussianPotential": (".potentials.multi_gaussian", "MultiGaussianPotential"),
    "GaussianPotential": (".potentials.gaussian", "GaussianPotential"),
    "BSplinePotential": (".potentials.bspline", "BSplinePotential"),
    "LennardJonesPotential": (".potentials.lennardjones", "LennardJonesPotential"),
    "LennardJones96Potential": (".potentials.lennardjones96", "LennardJones96Potential"),
    "LennardJonesSoftPotential": (".potentials.lennardjones_soft", "LennardJonesSoftPotential"),
    "SRLRGaussianPotential": (".potentials.srlrgaussian", "SRLRGaussianPotential"),
    "UnnormalizedMultiGaussianPotential": (
        ".potentials.unnormalized_multi_gaussian",
        "UnnormalizedMultiGaussianPotential",
    ),
    "BasePotential": (".potentials.base", "BasePotential"),
    "IteratePotentials": (".potentials.base", "IteratePotentials"),
    "HarmonicPotential": (".potentials.harmonic", "HarmonicPotential"),
    "GatedPotential": (".potentials.gated", "GatedPotential"),
    "iter_gated_potentials": (".potentials.gated", "iter_gated_potentials"),
    "sample_L0_gates": (".potentials.gated", "sample_L0_gates"),
    "set_L0_gates_deterministic": (".potentials.gated", "set_L0_gates_deterministic"),
    "wrap_forcefield_with_L0_gates": (".potentials.gated", "wrap_forcefield_with_L0_gates"),
    # Topology
    "InteractionKey": (".topology", "InteractionKey"),
    "Forcefield": (".topology", "Forcefield"),
    "collect_topology_arrays": (".topology.topology_array", "collect_topology_arrays"),
    # Scheduler
    "HostInventory": (".schedulers", "HostInventory"),
    "CpuLease": (".schedulers", "CpuLease"),
    "LeasePool": (".schedulers", "LeasePool"),
    "ResourcePool": (".schedulers", "ResourcePool"),
    "TaskScheduler": (".schedulers", "TaskScheduler"),
    "TaskSpec": (".schedulers", "TaskSpec"),
    "TaskResult": (".schedulers", "TaskResult"),
    "IterationResult": (".schedulers", "IterationResult"),
    "AllTasksFailedError": (".schedulers", "AllTasksFailedError"),
    "resolve_sim_var": (".schedulers", "resolve_sim_var"),
    "preflight_benchmark": (".schedulers", "preflight_benchmark"),
}


__all__ = [
    # Trainers
    "REMTrainerAnalytic",
    "load_reweighted_rem_stacks",
    "MSETrainerAnalytic",
    "load_reweighted_mse_stacks",
    "CDREMTrainerAnalytic",
    "MultiTrainerAnalytic",
    "FMTrainerAnalytic",
    "CDFMTrainerAnalytic",
    "L0InteractionTrainerAnalytic",
    # Optimizers
    "BaseOptimizer",
    "NewtonRaphsonOptimizer",
    "AdamMaskedOptimizer",
    "AdamWMaskedOptimizer",
    "RMSpropMaskedOptimizer",
    "MTAdamOptimizer",
    # Solvers
    "BaseSolver",
    "FMMatrixSolver",
    # I/O
    "ReadLmpFF",
    "ReadLmpFFBounds",
    "ReadLmpFFMask",
    "write_lammps_table",
    "WriteLmpFF",
    # Compute
    "build_default_engine",
    # Potentials
    "MultiGaussianPotential",
    "GaussianPotential",
    "BSplinePotential",
    "LennardJonesPotential",
    "LennardJones96Potential",
    "LennardJonesSoftPotential",
    "SRLRGaussianPotential",
    "UnnormalizedMultiGaussianPotential",
    "BasePotential",
    "IteratePotentials",
    "HarmonicPotential",
    "GatedPotential",
    "iter_gated_potentials",
    "sample_L0_gates",
    "set_L0_gates_deterministic",
    "wrap_forcefield_with_L0_gates",
    # Topology
    "InteractionKey",
    "Forcefield",
    "collect_topology_arrays",
    # Scheduler
    "HostInventory",
    "CpuLease",
    "LeasePool",
    "ResourcePool",
    "TaskScheduler",
    "TaskSpec",
    "TaskResult",
    "IterationResult",
    "AllTasksFailedError",
    "resolve_sim_var",
    "preflight_benchmark",
]


def _load_mt_adam() -> Any:
    try:
        from .optimizers.multithreaded.adam import MTAdamOptimizer
    except ImportError as exc:
        missing = (getattr(exc, "name", "") or "").split(".")[0]
        if missing not in {"numba", "llvmlite"}:
            raise
        MTAdamOptimizer = None

    globals()["MTAdamOptimizer"] = MTAdamOptimizer
    return MTAdamOptimizer


def __getattr__(name: str) -> Any:
    if name == "MTAdamOptimizer":
        return _load_mt_adam()

    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
