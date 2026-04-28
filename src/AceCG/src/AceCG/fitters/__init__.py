"""Table-fitter package exports and registration side effects.

Import this package, not ``fitters.base``, when runtime code needs the shared
``TABLE_FITTERS`` registry. The concrete fitter modules register themselves at
import time, so package import is the stable boundary that guarantees the
registry is populated before a workflow asks for a named fitter profile.
"""

from .base import BaseTableFitter, FitterRegistry, TABLE_FITTERS
from .fit_bspline import BSplineConfig, BSplineTableFitter
from .fit_harmonic import HarmonicTableFitter
from .fit_multi_gaussian import MultiGaussianTableFitter

__all__ = [
	"BaseTableFitter",
	"FitterRegistry",
	"TABLE_FITTERS",
	"BSplineConfig",
	"BSplineTableFitter",
	"HarmonicTableFitter",
	"MultiGaussianTableFitter",
]
