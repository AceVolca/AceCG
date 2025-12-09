# AceCG/potentials/__init__.py
from .base import BasePotential
from .bspline import BSplinePotential
from .gaussian import GaussianPotential
from .lennardjones import LennardJonesPotential
from .lennardjones96 import LennardJones96Potential
from .lennardjones_soft import LennardJonesSoftPotential
from .multi_gaussian import MultiGaussianPotential
from .srlrgaussian import SRLRGaussianPotential

POTENTIAL_REGISTRY = {
    "gauss/cut": GaussianPotential,
    "gauss/wall": GaussianPotential,
    "lj/cut": LennardJonesPotential,
    "lj96/cut": LennardJones96Potential,
    "lj/cut/soft": LennardJonesSoftPotential,
    "table": MultiGaussianPotential,
    "double/gauss": MultiGaussianPotential,
    "srlr_gauss": SRLRGaussianPotential,
}