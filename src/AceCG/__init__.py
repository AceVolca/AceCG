"""
AceCG: A Python package for bottom-up coarse-graining.
"""

# Core REM trainers
from .trainers.analytic import REMTrainerAnalytic
from .trainers.analytic import MSETrainerAnalytic
from .trainers.analytic import MultiTrainerAnalytic
from .trainers.utils import prepare_Trainer_data
from .trainers.regularized import L0MultiTrainerAnalytic

# Optimizers
from .optimizers.base import BaseOptimizer
from .optimizers.newton_raphson import NewtonRaphsonOptimizer
from .optimizers.adam import AdamMaskedOptimizer
from .optimizers.adamW import AdamWMaskedOptimizer
from .optimizers.rmsprop import RMSpropMaskedOptimizer
from .optimizers.multithreaded.adam import MTAdamOptimizer

# Potentials
from .potentials.multi_gaussian import MultiGaussianPotential
from .potentials.gaussian import GaussianPotential
from .potentials.bspline import BSplinePotential
from .potentials.lennardjones import LennardJonesPotential
from .potentials.lennardjones96 import LennardJones96Potential
from .potentials.lennardjones_soft import LennardJonesSoftPotential
from .potentials.srlrgaussian import SRLRGaussianPotential
from .potentials.base import BasePotential

# Utilities
from .utils.compute import dUdLByFrame, dUdL, d2UdLjdLk_Matrix, dUdLj_dUdLk_Matrix, Hessian, KL_divergence, dUdLByBin
from .utils.neighbor import Pair2DistanceByFrame
from .utils.ffio import FFParamArray, FFParamIndexMap, ReadLmpFF, WriteLmpTable, WriteLmpFF, ParseLmpTable
from .utils.mask import BuildGlobalMask, DescribeMask
from .fitters.fit_multi_gaussian import MultiGaussianConfig, MultiGaussianTableFitter
from .utils.bounds import BuildGlobalBounds, DescribeBounds

__all__ = [
    "REMTrainerAnalytic",
	"MSETrainerAnalytic",
    "MultiTrainerAnalytic",
    "prepare_Trainer_data",
    "L0MultiTrainerAnalytic",
    "BaseOptimizer",
    "NewtonRaphsonOptimizer",
	"AdamMaskedOptimizer",
	"AdamWMaskedOptimizer",
	"RMSpropMaskedOptimizer",
    "MTAdamOptimizer",
	"MultiGaussianPotential",
    "GaussianPotential",
    "BSplinePotential",
	"LennardJonesPotential",
	"LennardJones96Potential",
    "LennardJonesSoftPotential",
    "SRLRGaussianPotential",
    "BasePotential",
    "dUdLByFrame",
    "dUdL",
    "d2UdLjdLk_Matrix",
    "dUdLj_dUdLk_Matrix",
    "Hessian",
	"KL_divergence",
	"dUdLByBin",
    "Pair2DistanceByFrame",
    "FFParamArray",
    "FFParamIndexMap",
    "ReadLmpFF",
    "WriteLmpTable",
    "WriteLmpFF",
	"ParseLmpTable",
	"BuildGlobalMask",
	"DescribeMask",
	"MultiGaussianConfig",
	"MultiGaussianTableFitter",
	"BuildGlobalBounds",
	"DescribeBounds",
]

