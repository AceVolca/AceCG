"""Orthonormal one-dimensional bases for spectral RDF estimation."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SpectralBasis(Protocol):
    """Protocol implemented by spectral RDF basis families."""

    name: str

    def evaluate(
        self,
        r: np.ndarray,
        n_modes: int,
        domain: tuple[float, float],
    ) -> np.ndarray: ...

    def derivative(
        self,
        r: np.ndarray,
        n_modes: int,
        domain: tuple[float, float],
    ) -> np.ndarray: ...


def _validated_inputs(
    r: np.ndarray,
    n_modes: int,
    domain: tuple[float, float],
) -> tuple[np.ndarray, int, float, float, float]:
    values = np.asarray(r, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"r must be one-dimensional, got shape {values.shape}")
    count = int(n_modes)
    if count < 1:
        raise ValueError("n_modes must be at least 1")
    left, right = (float(domain[0]), float(domain[1]))
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        raise ValueError(f"Invalid spectral domain {domain!r}")
    return values, count, left, right, right - left


class CosineBasis:
    """Orthonormal cosine basis on a finite interval."""

    name = "cosine"

    def evaluate(
        self,
        r: np.ndarray,
        n_modes: int,
        domain: tuple[float, float],
    ) -> np.ndarray:
        values, count, left, _right, length = _validated_inputs(r, n_modes, domain)
        modes = np.arange(count, dtype=np.float64)
        phase = np.pi * (values[:, None] - left) * modes[None, :] / length
        out = np.cos(phase)
        out[:, 0] *= 1.0 / np.sqrt(length)
        if count > 1:
            out[:, 1:] *= np.sqrt(2.0 / length)
        return out

    def derivative(
        self,
        r: np.ndarray,
        n_modes: int,
        domain: tuple[float, float],
    ) -> np.ndarray:
        values, count, left, _right, length = _validated_inputs(r, n_modes, domain)
        modes = np.arange(count, dtype=np.float64)
        phase = np.pi * (values[:, None] - left) * modes[None, :] / length
        out = -np.sin(phase) * (np.pi * modes[None, :] / length)
        out[:, 0] = 0.0
        if count > 1:
            out[:, 1:] *= np.sqrt(2.0 / length)
        return out


class ShiftedLegendreBasis:
    """Orthonormal shifted Legendre basis on a finite interval."""

    name = "shifted_legendre"

    @staticmethod
    def _values_and_derivatives(
        x: np.ndarray,
        n_modes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.empty((x.size, n_modes), dtype=np.float64)
        deriv_x = np.empty_like(values)
        values[:, 0] = 1.0
        deriv_x[:, 0] = 0.0
        if n_modes == 1:
            return values, deriv_x
        values[:, 1] = x
        deriv_x[:, 1] = 1.0
        for degree in range(1, n_modes - 1):
            denom = float(degree + 1)
            factor = float(2 * degree + 1)
            values[:, degree + 1] = (
                factor * x * values[:, degree] - degree * values[:, degree - 1]
            ) / denom
            deriv_x[:, degree + 1] = (
                factor * (values[:, degree] + x * deriv_x[:, degree])
                - degree * deriv_x[:, degree - 1]
            ) / denom
        return values, deriv_x

    def evaluate(
        self,
        r: np.ndarray,
        n_modes: int,
        domain: tuple[float, float],
    ) -> np.ndarray:
        values, count, left, _right, length = _validated_inputs(r, n_modes, domain)
        x = 2.0 * (values - left) / length - 1.0
        polynomials, _ = self._values_and_derivatives(x, count)
        scales = np.sqrt((2.0 * np.arange(count, dtype=np.float64) + 1.0) / length)
        return polynomials * scales[None, :]

    def derivative(
        self,
        r: np.ndarray,
        n_modes: int,
        domain: tuple[float, float],
    ) -> np.ndarray:
        values, count, left, _right, length = _validated_inputs(r, n_modes, domain)
        x = 2.0 * (values - left) / length - 1.0
        _, deriv_x = self._values_and_derivatives(x, count)
        scales = np.sqrt((2.0 * np.arange(count, dtype=np.float64) + 1.0) / length)
        return deriv_x * (2.0 / length) * scales[None, :]


def make_basis(name: str) -> SpectralBasis:
    """Construct a supported spectral basis by canonical name or alias."""
    token = str(name).strip().lower().replace("-", "_")
    if token in {"cos", "cosine"}:
        return CosineBasis()
    if token in {"legendre", "shifted_legendre", "shiftedlegendre"}:
        return ShiftedLegendreBasis()
    raise ValueError(
        f"Unknown spectral basis {name!r}; expected 'cosine' or 'shifted_legendre'"
    )


__all__ = [
    "SpectralBasis",
    "CosineBasis",
    "ShiftedLegendreBasis",
    "make_basis",
]
