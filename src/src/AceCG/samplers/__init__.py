"""AceCG sampler module.

Public API
----------
BaseSampler, ConditionedSampler   – sampler classes
InitConfigRecord, ReplicaPlan     – data classes
EpochState, RunResult             – lifecycle data
ScriptInfo, parse_script          – script inspection protocol
LammpsScriptInfo                  – LAMMPS-specific parser result
"""

from .base import (
    BaseSampler,
    EpochState,
    InitConfigRecord,
    ReplicaPlan,
    RunResult,
)
from .conditioned import ConditionedSampler
from ._script_inspector import ScriptInfo, parse_script
from ._lammps_script import LammpsScriptInfo, parse_lammps_script

__all__ = [
    "BaseSampler",
    "ConditionedSampler",
    "EpochState",
    "InitConfigRecord",
    "ReplicaPlan",
    "RunResult",
    "ScriptInfo",
    "parse_script",
    "LammpsScriptInfo",
    "parse_lammps_script",
]
