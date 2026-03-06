"""
AceCG: A Python package for bottom-up coarse-graining.
"""

# Core CG FF trainers
from .trainers.analytic import REMTrainerAnalytic
from .trainers.analytic import MSETrainerAnalytic
from .trainers.analytic import MultiTrainerAnalytic
from .trainers.fm_analytic import FMTrainerAnalytic
from .trainers.utils import prepare_Trainer_data, prepare_Trainer_data_parallel

# Optimizers
from .optimizers.base import BaseOptimizer
from .optimizers.newton_raphson import NewtonRaphsonOptimizer
from .optimizers.adam import AdamMaskedOptimizer
from .optimizers.adamW import AdamWMaskedOptimizer
from .optimizers.rmsprop import RMSpropMaskedOptimizer
try:
    from .optimizers.multithreaded.adam import MTAdamOptimizer
except Exception:  # pragma: no cover - optional dependency (numba) path
    MTAdamOptimizer = None

# Solvers
from .solvers.base import BaseSolver
from .solvers.fm_matrix import FMMatrixSolver

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
from .utils.ffio import FFParamArray, FFParamIndexMap, ReadLmpFF, WriteLmpFF
from .utils.mask import BuildGlobalMask, DescribeMask
from .utils.bounds import BuildGlobalBounds, DescribeBounds
from .utils.trjio import split_lammpstrj, split_lammpstrj_mdanalysis
from .utils.cgcoords import load_mapping_yaml, build_CG_coords
from .utils.bonded_projectors import FMInteraction, build_design_matrix
from .utils.topology_mscg import MSCGTopology, parse_mscg_top, build_replicated_topology_arrays, attach_topology_from_mscg_top
from .utils.fm_workflow import load_config, build_interactions, load_window_universe, frame_slice, interaction_table_stem, find_equilibrium
from .utils.ffio import build_forcefield_tables, export_tables, compare_table_files

# Fitters
from .fitters.fit_bspline import BSplineConfig, BSplineTableFitter
from .fitters.fit_multi_gaussian import MultiGaussianConfig, MultiGaussianTableFitter

__all__ = [
    "REMTrainerAnalytic",
	"MSETrainerAnalytic",
    "MultiTrainerAnalytic",
    "FMTrainerAnalytic",
    "prepare_Trainer_data",
    "prepare_Trainer_data_parallel",
    "BaseSolver",
    "FMMatrixSolver",
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
    "WriteLmpFF",
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
    "FMInteraction",
    "build_design_matrix",
    "MSCGTopology",
    "parse_mscg_top",
    "build_replicated_topology_arrays",
    "attach_topology_from_mscg_top",
    "split_lammpstrj",
    "split_lammpstrj_mdanalysis",
]

