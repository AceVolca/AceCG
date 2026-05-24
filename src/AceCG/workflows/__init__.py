"""AceCG workflow classes."""

from .base import BaseWorkflow
from .cdfm import CDFMWorkflow
from .cdrem import CDREMWorkflow
from .dsm import DSMWorkflow
from .fm import FMWorkflow
from .rem import REMWorkflow
from .sampling import SamplingWorkflow
from .vp_growth import VPGrowthResult, VPGrowthWorkflow

__all__ = [
    "BaseWorkflow",
    "CDFMWorkflow",
    "CDREMWorkflow",
    "DSMWorkflow",
    "FMWorkflow",
    "REMWorkflow",
    "SamplingWorkflow",
    "VPGrowthResult",
    "VPGrowthWorkflow",
    "run_boundary_prior",
]


def __getattr__(name: str):
    if name == "run_boundary_prior":
        from .boundary_prior import run_boundary_prior

        return run_boundary_prior
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
