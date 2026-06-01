# AceCG/trainers/analytic/__init__.py
"""Analytic (NumPy-based) trainers package.

Re-exports all public trainers and TypedDicts from submodules:
  - rem.py    : EnsembleBatch, REMBatch, REMOut, REMTrainerAnalytic
  - mse.py    : MSEBatch, MSEOut, MSETrainerAnalytic, load_reweighted_mse_stacks
  - cdrem.py  : CDREMBatch, CDREMOut, CDREMTrainerAnalytic
  - l_zero.py : L0Batch, L0Out, L0InteractionTrainerAnalytic
  - multi.py  : MultiOut, MultiTrainerAnalytic
"""

from .rem import (
    EnsembleBatch,
    REMBatch,
    REMOut,
    REMTrainerAnalytic,
    load_reweighted_rem_stacks,
)
from .mse import MSEBatch, MSEOut, MSETrainerAnalytic, load_reweighted_mse_stacks
from .cdrem import CDREMBatch, CDREMOut, CDREMTrainerAnalytic
from .multi import MultiOut, MultiTrainerAnalytic
from .fm import FMBatch, FMTrainerAnalytic
from .cdfm import CDFMBatch, CDFMTrainerAnalytic
from .l_zero import L0Batch, L0Out, L0InteractionTrainerAnalytic

__all__ = [
    "EnsembleBatch",
    "REMBatch",
    "REMOut",
    "REMTrainerAnalytic",
    "load_reweighted_rem_stacks",
    "load_reweighted_mse_stacks",
    "MSEBatch",
    "MSEOut",
    "MSETrainerAnalytic",
    "CDREMBatch",
    "CDREMOut",
    "CDREMTrainerAnalytic",
    "MultiOut",
    "MultiTrainerAnalytic",
    "FMBatch",
    "FMTrainerAnalytic",
    "CDFMBatch",
    "CDFMTrainerAnalytic",
    "L0Batch",
    "L0Out",
    "L0InteractionTrainerAnalytic",
]
