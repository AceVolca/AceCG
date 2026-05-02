"""AceCG configuration loading and parsing.

Public API
----------
parse_acg_file          – ``.acg`` → ``ACGConfig`` model
parse_acg_text          – raw text  → section dicts
build_acg_config        – section dicts → ``ACGConfig`` model
parse_vp_growth_file    – VP Growth ``.acg`` → ``VPGrowthConfig`` model
"""

from __future__ import annotations

from .models import ACGConfig
from .parser import (
    ACGConfigError,
    build_acg_config,
    parse_acg_file,
    parse_acg_text,
)
from .vp_config import VPConfig, parse_vp_config
from .vp_growth_config import (
    VPGrowthAARef,
    VPGrowthConfig,
    VPGrowthRun,
    parse_vp_growth_file,
    parse_vp_growth_text,
)

__all__ = [
    "ACGConfig",
    "ACGConfigError",
    "VPConfig",
    "VPGrowthAARef",
    "VPGrowthConfig",
    "VPGrowthRun",
    "build_acg_config",
    "parse_acg_file",
    "parse_acg_text",
    "parse_vp_config",
    "parse_vp_growth_file",
    "parse_vp_growth_text",
]

