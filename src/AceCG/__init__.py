<<<<<<< HEAD
# Load packages
import AceCG.potentials as cgp
import AceCG.Compute as com
import AceCG.NeighborTools as nt
=======
"""
AceCG: A modular framework for relative entropy minimization (REM) with analytic and neural potentials.
"""

# Core REM trainers
from .trainers.analytic import REMTrainerAnalytic
from .trainers.utils import prepare_Trainer_data

# Optimizers
from .optimizers.base import BaseOptimizer
from .optimizers.newton_raphson import NewtonRaphsonOptimizer
from .optimizers.adam import AdamMaskedOptimizer

# Potentials
from .potentials.gaussian import GaussianPotential
from .potentials.base import BasePotential

# Utilities
from .utils.compute import dUdLByFrame, dUdL, d2UdLjdLk_Matrix, dUdLj_dUdLk_Matrix, Hessian
from .utils.neighbor import Pair2DistanceByFrame
from .utils.ffio import FFParamArray, FFParamIndexMap, ReadLmpFF, WriteLmpFF

__all__ = [
    "REMTrainerAnalytic",
    "prepare_Trainer_data",
    "BaseOptimizer",
    "NewtonRaphsonOptimizer",
	"AdamMaskedOptimizer",
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
>>>>>>> 95756921103cf82eb311687791547e42f6f76f88
