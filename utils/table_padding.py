from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _uniform_grid(xmin: float, xmax: float, dx: float) -> np.ndarray:
    xmin_f = float(xmin)
    xmax_f = float(xmax)
    dx_f = float(dx)
    n = int(round((xmax_f - xmin_f) / dx_f)) + 1
    if n < 2:
        n = 2
    return np.linspace(xmin_f, xmax_f, n, dtype=float)


def integrate_force_to_potential(x: np.ndarray, force: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    f = np.asarray(force, dtype=float).ravel()
    if x.size == 0:
        return np.empty(0, dtype=float)
    if x.size == 1:
        return np.zeros(1, dtype=float)

    dx = np.diff(x)
    trap = 0.5 * (f[:-1] + f[1:]) * dx
    u = np.empty_like(f)
    u[-1] = 0.0
    u[:-1] = np.cumsum(trap[::-1])[::-1]
    return u


def constant_force_extrapolate(
    x_model: np.ndarray,
    potential_model: np.ndarray,
    force_model: np.ndarray,
    x_out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate in-range and linearly extrapolate potential with boundary force."""
    xm = np.asarray(x_model, dtype=float).ravel()
    um = np.asarray(potential_model, dtype=float).ravel()
    fm = np.asarray(force_model, dtype=float).ravel()
    xo = np.asarray(x_out, dtype=float).ravel()
    if xm.size < 2:
        raise ValueError("x_model must contain at least two points")
    if xm.size != um.size or xm.size != fm.size:
        raise ValueError("x_model/potential_model/force_model must have identical lengths")

    u = np.empty_like(xo)
    f = np.empty_like(xo)
    lo = float(xm[0])
    hi = float(xm[-1])

    for i, xv in enumerate(xo):
        if xv <= lo:
            f[i] = fm[0]
            u[i] = um[0] + fm[0] * (lo - xv)
        elif xv >= hi:
            f[i] = fm[-1]
            u[i] = um[-1] + fm[-1] * (hi - xv)
        else:
            j = int(np.searchsorted(xm, xv))
            j = max(1, min(j, xm.size - 1))
            t = (xv - xm[j - 1]) / max(xm[j] - xm[j - 1], 1.0e-30)
            u[i] = um[j - 1] + t * (um[j] - um[j - 1])
            f[i] = fm[j - 1] + t * (fm[j] - fm[j - 1])

    return u, f


def export_grid(spec: Dict[str, Any]) -> np.ndarray:
    """Build a uniform output grid from an interaction spec dict."""
    dx = float(spec.get("table_resolution", spec["resolution"]))
    xmin = float(spec.get("table_min", spec["min"]))
    xmax = float(spec.get("table_max", spec["max"]))
    return _uniform_grid(xmin, xmax, dx)
