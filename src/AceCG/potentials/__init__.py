"""Public exports for AceCG potentials."""

from .base import BasePotential, IteratePotentials
from .bspline import BSplinePotential
from .gaussian import GaussianPotential
from .harmonic import HarmonicPotential
from .lennardjones import LennardJonesPotential
from .lennardjones96 import LennardJones96Potential
from .lennardjones_soft import LennardJonesSoftPotential
from .multi_gaussian import MultiGaussianPotential
from .soft import SoftPotential
from .srlrgaussian import SRLRGaussianPotential
from .unnormalized_multi_gaussian import UnnormalizedMultiGaussianPotential

# Potential classes coresponding to LAMMPS implementation
POTENTIAL_REGISTRY = {
    "gauss/cut": GaussianPotential,
    "gauss/wall": GaussianPotential,
    "harmonic": HarmonicPotential,
    "lj/cut": LennardJonesPotential,
    "lj96/cut": LennardJones96Potential,
    "lj/cut/soft": LennardJonesSoftPotential,
    "soft": SoftPotential,
    "table": MultiGaussianPotential,
    "double/gauss": UnnormalizedMultiGaussianPotential,
    "srlr_gauss": SRLRGaussianPotential,
}


def __getattr__(name: str):
    """Lazily expose optional potential wrappers without import cycles."""
    if name in {"BoundaryPriorPotential", "apply_boundary_prior"}:
        from .boundary_prior import BoundaryPriorPotential, apply_boundary_prior

        exports = {
            "BoundaryPriorPotential": BoundaryPriorPotential,
            "apply_boundary_prior": apply_boundary_prior,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
