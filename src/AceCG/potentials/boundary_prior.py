"""Fixed endpoint priors for one-shot forcefield postprocessing."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from ..configs.energy_mask import normalize_energy_mask_spec
from ..topology.forcefield import Forcefield
from ..topology.types import InteractionKey
from .base import BasePotential


class BoundaryPriorPotential(BasePotential):
    """Potential wrapper that adds fixed endpoint behavior outside support."""

    def __init__(
        self,
        base: BasePotential,
        *,
        style: str,
        lower: float | None = None,
        upper: float | None = None,
        pair_decay: float = 0.25,
        pair_strength: float = 50.0,
        wall_k_min: float = 1.0,
    ) -> None:
        super().__init__(bonded=getattr(base, "bonded", False))
        self.base = base
        self.style = str(style).strip().lower()
        self.lower = None if lower is None else float(lower)
        self.upper = None if upper is None else float(upper)
        self.pair_decay = float(pair_decay)
        self.pair_strength = float(pair_strength)
        self.wall_k_min = float(wall_k_min)
        if self.pair_decay <= 0.0:
            raise ValueError("pair_decay must be positive.")
        if self.pair_strength <= 0.0:
            raise ValueError("pair_strength must be positive.")
        if self.wall_k_min <= 0.0:
            raise ValueError("wall_k_min must be positive.")
        self._lower_k = self._estimate_wall_k(self.lower, side="lower")
        self._upper_k = self._estimate_wall_k(self.upper, side="upper")

    def n_params(self) -> int:
        return self.base.n_params()

    def get_params(self) -> np.ndarray:
        return self.base.get_params()

    def set_params(self, new_params: np.ndarray):
        self.base.set_params(new_params)

    def param_names(self) -> list[str]:
        return self.base.param_names()

    def dparam_names(self) -> list[str]:
        return self.base.dparam_names()

    def d2param_names(self) -> list[list[str]]:
        return self.base.d2param_names()

    def df_dparam_names(self) -> list[str]:
        return self.base.df_dparam_names()

    def d2param_dr_names(self) -> list[list[str]]:
        return self.base.d2param_dr_names()

    def is_param_linear(self) -> np.ndarray:
        return self.base.is_param_linear()

    def is_gauge_free_energy_grad_cacheable(self) -> np.ndarray:
        return self.base.is_gauge_free_energy_grad_cacheable()

    def param_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(self.base, "param_bounds"):
            lb, ub = self.base.param_bounds()
            return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)
        n_params = self.n_params()
        return np.full(n_params, -np.inf), np.full(n_params, np.inf)

    def value(self, r: np.ndarray) -> np.ndarray:
        raw = np.asarray(r, dtype=float)
        scalar = raw.ndim == 0
        coords = np.atleast_1d(raw)
        out = np.asarray(self.base.value(coords), dtype=float).copy()
        if self.style == "pair" and self.lower is not None:
            mask = coords < self.lower
            out[mask] = self._pair_core_value(coords[mask])
        elif self.style == "bond":
            out = self._apply_bond_value(coords, out)
        elif self.style == "angle":
            out = self._apply_angle_value(coords, out)
        return out[0] if scalar else out.reshape(raw.shape)

    def force(self, r: np.ndarray) -> np.ndarray:
        raw = np.asarray(r, dtype=float)
        scalar = raw.ndim == 0
        coords = np.atleast_1d(raw)
        out = np.asarray(self.base.force(coords), dtype=float).copy()
        if self.style == "pair" and self.lower is not None:
            mask = coords < self.lower
            out[mask] = self._pair_core_force(coords[mask])
        elif self.style == "bond":
            out = self._apply_bond_force(coords, out)
        elif self.style == "angle":
            out = self._apply_angle_force(coords, out)
        return out[0] if scalar else out.reshape(raw.shape)

    def energy_grad(self, r: np.ndarray) -> np.ndarray:
        grad = np.asarray(self.base.energy_grad(r), dtype=float).copy()
        active = self._trainable_coordinate_mask(np.asarray(r, dtype=float))
        return np.where(active[..., None], grad, 0.0)

    def force_grad(self, r: np.ndarray) -> np.ndarray:
        grad = np.asarray(self.base.force_grad(r), dtype=float).copy()
        active = self._trainable_coordinate_mask(np.asarray(r, dtype=float))
        return np.where(active[..., None], grad, 0.0)

    def energy_grad_sum(self, r: np.ndarray) -> np.ndarray:
        coords = np.asarray(r, dtype=float)
        active = self._trainable_coordinate_mask(coords)
        return np.asarray(self.base.energy_grad_sum(coords[active]), dtype=float)

    def energy_grad_sum_by_sample(
        self,
        r: np.ndarray,
        *,
        active: np.ndarray | None = None,
    ) -> np.ndarray:
        coords = np.asarray(r, dtype=float)
        support = self._trainable_coordinate_mask(coords)
        if active is not None:
            support &= np.asarray(active, dtype=bool)
        return self.base.energy_grad_sum_by_sample(coords, active=support)

    def gauge_free_energy_grad_sum(self, r: np.ndarray) -> np.ndarray:
        coords = np.asarray(r, dtype=float)
        active = self._trainable_coordinate_mask(coords)
        return np.asarray(
            self.base.gauge_free_energy_grad_sum(coords[active]),
            dtype=float,
        )

    def gauge_free_energy_grad_sum_by_sample(
        self,
        r: np.ndarray,
        *,
        active: np.ndarray | None = None,
    ) -> np.ndarray:
        coords = np.asarray(r, dtype=float)
        support = self._trainable_coordinate_mask(coords)
        if active is not None:
            support &= np.asarray(active, dtype=bool)
        return self.base.gauge_free_energy_grad_sum_by_sample(coords, active=support)

    def _pair_amplitude(self) -> float:
        boundary_force = 0.0 if self.lower is None else abs(self._base_force_at(self.lower))
        return max(boundary_force, self.pair_strength)

    def _pair_core_value(self, values: np.ndarray) -> np.ndarray:
        assert self.lower is not None
        amplitude = self._pair_amplitude()
        exponent = np.exp((self.lower - values) / self.pair_decay)
        return self._base_value_at(self.lower) + amplitude * self.pair_decay * (exponent - 1.0)

    def _pair_core_force(self, values: np.ndarray) -> np.ndarray:
        assert self.lower is not None
        amplitude = self._pair_amplitude()
        return amplitude * np.exp((self.lower - values) / self.pair_decay)

    def _apply_bond_value(self, coords: np.ndarray, out: np.ndarray) -> np.ndarray:
        if self.lower is not None:
            mask = coords < self.lower
            delta = coords[mask] - self.lower
            out[mask] = self._base_value_at(self.lower) + 0.5 * self._lower_k * delta * delta
        if self.upper is not None:
            mask = coords > self.upper
            delta = coords[mask] - self.upper
            out[mask] = self._base_value_at(self.upper) + 0.5 * self._upper_k * delta * delta
        return out

    def _apply_bond_force(self, coords: np.ndarray, out: np.ndarray) -> np.ndarray:
        if self.lower is not None:
            mask = coords < self.lower
            out[mask] = self._lower_k * (self.lower - coords[mask])
        if self.upper is not None:
            mask = coords > self.upper
            out[mask] = -self._upper_k * (coords[mask] - self.upper)
        return out

    def _apply_angle_value(self, coords: np.ndarray, out: np.ndarray) -> np.ndarray:
        if self.lower is not None:
            mask = coords < self.lower
            delta = coords[mask] - self.lower
            out[mask] = self._base_value_at(self.lower) + 0.5 * self._lower_k * delta * delta
        if self.upper is not None:
            mask = coords > self.upper
            out[mask] = self._base_value_at(self.upper)
        return out

    def _apply_angle_force(self, coords: np.ndarray, out: np.ndarray) -> np.ndarray:
        if self.lower is not None:
            mask = coords < self.lower
            out[mask] = self._lower_k * (self.lower - coords[mask])
        if self.upper is not None:
            out[coords > self.upper] = 0.0
        return out

    def _trainable_coordinate_mask(self, coords: np.ndarray) -> np.ndarray:
        mask = np.ones(coords.shape, dtype=bool)
        if self.style in {"pair", "bond", "angle"} and self.lower is not None:
            mask &= coords >= self.lower
        if self.style in {"bond", "angle"} and self.upper is not None:
            mask &= coords <= self.upper
        return mask

    def _estimate_wall_k(self, boundary: float | None, *, side: str) -> float:
        if boundary is None:
            return self.wall_k_min
        eps = max(abs(float(boundary)) * 1.0e-4, 1.0e-4)
        if side == "lower":
            f0 = self._base_force_at(boundary)
            f1 = self._base_force_at(boundary + eps)
        else:
            f0 = self._base_force_at(boundary - eps)
            f1 = self._base_force_at(boundary)
        slope = (f1 - f0) / eps
        return max(abs(float(slope)), self.wall_k_min)

    def _base_value_at(self, coord: float) -> float:
        return float(np.asarray(self.base.value(np.asarray([coord], dtype=float))).reshape(-1)[0])

    def _base_force_at(self, coord: float) -> float:
        return float(np.asarray(self.base.force(np.asarray([coord], dtype=float))).reshape(-1)[0])


def apply_boundary_prior(
    forcefield: Forcefield,
    boundary_spec: Mapping[str, Any],
    *,
    pair_decay: float = 0.25,
    pair_strength: float = 50.0,
    wall_k_min: float = 1.0,
) -> Forcefield:
    """Return a forcefield copy with endpoint priors applied to matching terms."""
    normalized = normalize_energy_mask_spec(boundary_spec)
    updated = Forcefield(forcefield)
    for key, potentials in list(updated.items()):
        bounds = normalized.get(_canonical_label(key))
        if bounds is None:
            continue
        wrapped = [
            BoundaryPriorPotential(
                copy.deepcopy(potential),
                style=key.style,
                lower=bounds.get("min"),
                upper=bounds.get("max"),
                pair_decay=pair_decay,
                pair_strength=pair_strength,
                wall_k_min=wall_k_min,
            )
            for potential in potentials
        ]
        updated[key] = wrapped
    return updated


def _canonical_label(key: InteractionKey) -> str:
    if key.style == "pair" and len(key.types) == 2:
        return InteractionKey.pair(key.types[0], key.types[1]).label()
    if key.style == "bond" and len(key.types) == 2:
        return InteractionKey.bond(key.types[0], key.types[1]).label()
    if key.style == "angle" and len(key.types) == 3:
        return InteractionKey.angle(key.types[0], key.types[1], key.types[2]).label()
    return key.label()
