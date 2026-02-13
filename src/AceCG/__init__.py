"""
AceCG: A Python package for bottom-up coarse-graining.
"""

# Core CG FF trainers
from .trainers.analytic import REMTrainerAnalytic
from .trainers.analytic import MSETrainerAnalytic
from .trainers.analytic import MultiTrainerAnalytic
from .trainers.utils import prepare_Trainer_data, prepare_Trainer_data_parallel

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
from .potentials.unnormalized_multi_gaussian import UnnormalizedMultiGaussianPotential
from .potentials.base import BasePotential

# Utilities
from .utils.compute import dUdLByFrame, dUdL, dUdL_parallel, d2UdLjdLk_Matrix, dUdLj_dUdLk_Matrix, Hessian, KL_divergence, dUdLByBin, compute_weighted_rdf, compute_weighted_pair_distance_pdfs
from .utils.neighbor import Pair2DistanceByFrame, combine_Pair2DistanceByFrame
from .utils.ffio import FFParamArray, FFParamIndexMap, ReadLmpFF, WriteLmpTable, WriteLmpFF, ParseLmpTable
from .utils.mask import BuildGlobalMask, DescribeMask
from .utils.bounds import BuildGlobalBounds, DescribeBounds
from .utils.trjio import split_lammpstrj, split_lammpstrj_mdanalysis
from .utils.cgcoords import load_mapping_yaml, build_CG_coords

# Fitters
from .fitters.fit_bspline import BSplineConfig, BSplineTableFitter
from .fitters.fit_multi_gaussian import MultiGaussianConfig, MultiGaussianTableFitter

__all__ = [
    "REMTrainerAnalytic",
	"MSETrainerAnalytic",
    "MultiTrainerAnalytic",
    "prepare_Trainer_data",
    "prepare_Trainer_data_parallel",
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
    "UnnormalizedMultiGaussianPotential",
    "BasePotential",
    "dUdLByFrame",
    "dUdL",
	"dUdL_parallel",
	"d2UdLjdLk_Matrix",
    "d2UdLjdLk_Matrix",
    "dUdLj_dUdLk_Matrix",
    "Hessian",
	"KL_divergence",
	"dUdLByBin",
    "compute_weighted_rdf",
	"compute_weighted_pair_distance_pdfs",
    "Pair2DistanceByFrame",
    "combine_Pair2DistanceByFrame",
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
    "BSplineConfig",
    "BSplineTableFitter",
	"BuildGlobalBounds",
	"DescribeBounds",
    "load_mapping_yaml",
    "build_CG_coords",
    "split_lammpstrj",
    "split_lammpstrj_mdanalysis",
]

