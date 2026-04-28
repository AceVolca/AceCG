"""AceCG Scheduler — task scheduling for HPC.

Public API
----------
ResourcePool        — discover nodes, select MPI backend
HostInventory       — one compute host
CpuLease            — leased CPU block
LeasePool           — dynamic CPU allocation pool
Placer              — place TaskRequest onto LeasePool
MpiBackend          — vendor-specific MPI launcher (protocol)
Placement           — host + core allocation for a task
LaunchSpec          — ready-to-execute command spec
pick_backend        — auto-select MPI backend from mpirun path
TaskScheduler       — stream tasks across CPU leases
TaskSpec            — one simulation + post-processing task
TaskResult          — completed task result
IterationResult     — aggregated iteration result
preflight_benchmark — standalone MPI rank benchmark helper
"""

from .mpi_backend import (
    HostSlice,
    LaunchSpec,
    MpiBackend,
    Placement,
    pick_backend,
)
from .resource_pool import (
    HostInventory,
    CpuLease,
    LeasePool,
    Placer,
    PlacementResult,
    ResourcePool,
)
from .task_scheduler import (
    TaskScheduler,
    TaskSpec,
    TaskResult,
    IterationResult,
    AllTasksFailedError,
    resolve_sim_var,
)
from .profiler import preflight_benchmark

__all__ = [
    "HostSlice",
    "HostInventory",
    "CpuLease",
    "LaunchSpec",
    "LeasePool",
    "MpiBackend",
    "Placer",
    "PlacementResult",
    "Placement",
    "ResourcePool",
    "TaskScheduler",
    "TaskSpec",
    "TaskResult",
    "IterationResult",
    "AllTasksFailedError",
    "pick_backend",
    "resolve_sim_var",
    "preflight_benchmark",
]
