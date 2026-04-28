"""Fit a LAMMPS table to a HarmonicPotential.

Provides ``HarmonicTableFitter`` — a :class:`BaseTableFitter` that
produces a 2-parameter harmonic ``(k, r0)`` potential from a LAMMPS
table, analogous to ``BSplineTableFitter`` and
``MultiGaussianTableFitter``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseTableFitter, TABLE_FITTERS
from ..potentials.harmonic import HarmonicPotential
from ..io.tables import find_equilibrium, parse_lammps_table


class HarmonicTableFitter(BaseTableFitter):
    """Fit a :class:`HarmonicPotential` from a LAMMPS bond/angle table.

    Reads the table, estimates the equilibrium position *r0* from the
    force zero-crossing and the spring constant *k* from the local
    force slope at that position.
    """

    def __init__(self, *, typ3: Optional[str] = None, **_overrides):
        self._typ3 = typ3

    def profile_name(self) -> str:
        """Return this fitter's registry profile name."""
        return "harmonic"

    def fit(self, table_path: str, typ1: str, typ2: str) -> HarmonicPotential:
        """Fit a harmonic potential from a bond/angle table file."""
        r, V, F = parse_lammps_table(table_path)
        r = np.asarray(r, dtype=float).ravel()
        if F is not None:
            F = np.asarray(F, dtype=float).ravel()
        else:
            V = np.asarray(V, dtype=float).ravel()
            F = -np.gradient(V, r)

        eq = find_equilibrium(r, F)
        idx = int(np.argmin(np.abs(r - eq)))
        if 0 < idx < len(r) - 1:
            dFdr = (F[idx + 1] - F[idx - 1]) / (r[idx + 1] - r[idx - 1])
            k = max(-dFdr / 2.0, 0.0)
        else:
            k = 1.0

        return HarmonicPotential(
            typ1, typ2, k, eq, cutoff=float(r[-1]), typ3=self._typ3,
        )


# register
TABLE_FITTERS.register("harmonic", lambda **kw: HarmonicTableFitter(**kw))
