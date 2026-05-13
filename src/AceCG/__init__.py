"""
AceCG: A Python package for bottom-up coarse-graining.
"""

# Core CG FF trainers
from .trainers.analytic import REMTrainerAnalytic
from .trainers.analytic import load_reweighted_rem_stacks
from .trainers.analytic import MSETrainerAnalytic
from .trainers.analytic import load_reweighted_mse_stacks
from .trainers.analytic import CDREMTrainerAnalytic
from .trainers.analytic import MultiTrainerAnalytic
from .trainers.analytic import FMTrainerAnalytic
from .trainers.analytic import CDFMTrainerAnalytic
from .trainers.analytic import L0InteractionTrainerAnalytic

# Optimizers
from .optimizers.base import BaseOptimizer
from .optimizers.newton_raphson import NewtonRaphsonOptimizer
from .optimizers.adam import AdamMaskedOptimizer
from .optimizers.adamW import AdamWMaskedOptimizer
from .optimizers.rmsprop import RMSpropMaskedOptimizer
try:
    from .optimizers.multithreaded.adam import MTAdamOptimizer
except Exception:  # pragma: no cover - optional dependency (numba) path
    MTAdamOptimizer = None

# Solvers
from .solvers.base import BaseSolver
from .solvers.fm_matrix import FMMatrixSolver

# I/O
from .io.forcefield import (
    ReadLmpFF,
    ReadLmpFFBounds,
    ReadLmpFFMask,
    write_lammps_table,
    WriteLmpFF,
)

# Compute
from .compute.registry import build_default_engine

# Potentials
from .potentials.multi_gaussian import MultiGaussianPotential
from .potentials.gaussian import GaussianPotential
from .potentials.bspline import BSplinePotential
from .potentials.lennardjones import LennardJonesPotential
from .potentials.lennardjones96 import LennardJones96Potential
from .potentials.lennardjones_soft import LennardJonesSoftPotential
from .potentials.srlrgaussian import SRLRGaussianPotential
from .potentials.unnormalized_multi_gaussian import UnnormalizedMultiGaussianPotential
from .potentials.base import BasePotential, IteratePotentials
from .potentials.harmonic import HarmonicPotential
from .potentials.gated import (
    GatedPotential,
    iter_gated_potentials,
    sample_L0_gates,
    set_L0_gates_deterministic,
    wrap_forcefield_with_L0_gates,
)

# Topology (new — PR0)
from .topology import InteractionKey, Forcefield
from .topology.topology_array import collect_topology_arrays

# Scheduler (v3)
from .schedulers import (
    HostInventory, CpuLease, LeasePool, ResourcePool,
    TaskScheduler, TaskSpec, TaskResult, IterationResult,
    AllTasksFailedError, resolve_sim_var,
    preflight_benchmark,
)

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
