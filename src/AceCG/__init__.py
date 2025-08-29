"""
AceCG: A modular framework for relative entropy minimization (REM) with analytic and neural potentials.
"""

# Core REM trainers
from .trainers.analytic import REMTrainerAnalytic
from .trainers.analytic import MSETrainerAnalytic
from .trainers.utils import prepare_Trainer_data

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
from .potentials.lennardjones import LennardJonesPotential
from .potentials.base import BasePotential

# Utilities
from .utils.compute import dUdLByFrame, dUdL, d2UdLjdLk_Matrix, dUdLj_dUdLk_Matrix, Hessian, KL_divergence, dUdLByBin
from .utils.neighbor import Pair2DistanceByFrame
from .utils.ffio import FFParamArray, FFParamIndexMap, ReadLmpFF, WriteLmpFF, ParseLmpTable
from .utils.mask import BuildGlobalMask, DescribeMask
from .fitters.fit_multi_gaussian import MultiGaussianConfig, MultiGaussianTableFitter

__all__ = [
    "REMTrainerAnalytic",
	"MSETrainerAnalytic",
    "prepare_Trainer_data",
    "BaseOptimizer",
    "NewtonRaphsonOptimizer",
	"AdamMaskedOptimizer",
	"AdamWMaskedOptimizer",
	"RMSpropMaskedOptimizer",
    "GaussianPotential",
	"LennardJonesPotential",
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
    "WriteLmpFF",
	"ParseLmpTable",
	"BuildGlobalMask",
	"DescribeMask",
	"MultiGaussianConfig",
	"MultiGaussianTableFitter",
]
