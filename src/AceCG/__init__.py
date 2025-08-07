"""
AceCG: A modular framework for relative entropy minimization (REM) with analytic and neural potentials.
"""

# Core REM trainers
from .trainers.analytic import REMTrainerAnalytic
from .trainers.utils import prepare_REM_data

# Optimizers
from .optimizers.base import BaseOptimizer
from .optimizers.newton_raphson import NewtonRaphsonOptimizer

# Potentials
from .potentials.gaussian import GaussianPotential
from .potentials.base import BasePotential

# Utilities
from .utils.compute import dUdLByFrame, dUdL, d2UdLjdLk_Matrix, dUdLj_dUdLk_Matrix, Hessian
from .utils.neighbor import Pair2DistanceByFrame
from .utils.ffio import FFParamArray, FFParamIndexMap, ReadLmpFF, WriteLmpFF

__all__ = [
    "REMTrainerAnalytic",
    "prepare_REM_data",
    "BaseOptimizer",
    "NewtonRaphsonOptimizer",
    "GaussianPotential",
    "BasePotential",
    "dUdLByFrame",
    "dUdL",
    "d2UdLjdLk_Matrix",
    "dUdLj_dUdLk_Matrix",
    "Hessian",
    "Pair2DistanceByFrame",
    "FFParamArray",
    "FFParamIndexMap",
    "ReadLmpFF",
    "WriteLmpFF",
]