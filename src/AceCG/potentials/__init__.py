# AceCG/potentials/__init__.py
from .base import BasePotential, IteratePotentials
from .bspline import BSplinePotential
from .gated import (
    GatedPotential,
    iter_gated_potentials,
    sample_L0_gates,
    set_L0_gates_deterministic,
    wrap_forcefield_with_L0_gates,
)
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
