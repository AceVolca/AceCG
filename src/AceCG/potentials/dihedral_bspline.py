"""B-spline force bases for signed, cyclic dihedral coordinates.

The trainable coefficients parameterize ``F(phi) = -dU/dphi`` in
energy-per-degree units.  Two boundary modes are supported:

``periodic``
    Uniform cyclic B-splines over ``[-180, 180)``.

``cutoff``
    Compact B-splines on a strict sub-domain.  The force and all parameter
    derivatives are zero outside the model window.

Both modes eliminate one raw coefficient so every trainable force basis has
zero integral over a complete turn.  Consequently the integrated energy is
single-valued at the LAMMPS cyclic seam.
"""

from __future__ import annotations

import re
from functools import cached_property
from typing import Sequence

import numpy as np
from scipy.interpolate import BSpline

from .base import BasePotential


_PERIOD_MIN = -180.0
_PERIOD_MAX = 180.0
_PERIOD = 360.0


class DihedralBSplinePotential(BasePotential):
    """Linear force-basis potential for one four-site dihedral interaction."""

    lammps_style = "table"

    def __init__(
        self,
        types: Sequence[str],
        *,
        minimum: float,
        maximum: float,
        coefficients: np.ndarray,
        degree: int = 3,
        boundary_mode: str = "periodic",
    ) -> None:
        super().__init__(bonded=True)
        labels = tuple(str(item) for item in types)
        if len(labels) != 4:
            raise ValueError(
                "DihedralBSplinePotential requires exactly four type labels."
            )
        mode = str(boundary_mode).strip().lower()
        if mode not in {"periodic", "cutoff"}:
            raise ValueError(
                "dihedral boundary_mode must be 'periodic' or 'cutoff', "
                f"got {boundary_mode!r}."
            )
        minimum_f = float(minimum)
        maximum_f = float(maximum)
        if not _PERIOD_MIN <= minimum_f < maximum_f <= _PERIOD_MAX:
            raise ValueError(
                "dihedral model domain must lie within [-180, 180], "
                f"got [{minimum_f}, {maximum_f}]."
            )
        if mode == "periodic" and not (
            np.isclose(minimum_f, _PERIOD_MIN)
            and np.isclose(maximum_f, _PERIOD_MAX)
        ):
            raise ValueError(
                "periodic dihedral models require domain [-180, 180]."
            )
        if mode == "cutoff" and maximum_f - minimum_f >= _PERIOD - 1.0e-10:
            raise ValueError(
                "cutoff dihedral models require a domain strictly narrower "
                "than 360 degrees."
            )

        degree_i = int(degree)
        if degree_i < 1:
            raise ValueError(f"dihedral B-spline degree must be positive, got {degree}.")
        if mode == "cutoff" and degree_i < 2:
            raise ValueError(
                "cutoff dihedral B-splines require degree >= 2 for smooth endpoints."
            )
        coeff = np.asarray(coefficients, dtype=float).reshape(-1)
        if coeff.size < 1:
            raise ValueError("dihedral B-spline requires at least one coefficient.")

        self.types = labels
        self.typ1 = labels[0]
        self.typ2 = labels[-1]
        self.minimum = minimum_f
        self.maximum = maximum_f
        self.degree = degree_i
        self.boundary_mode = mode
        self._params = coeff.copy()
        self._params_to_scale = list(range(coeff.size))
        self._param_names = [f"c{i}" for i in range(coeff.size)]
        self._dparam_names = [f"dc{i}" for i in range(coeff.size)]
        self._d2param_names = [
            [f"dc{i}_2" if i == j else f"dc{i}dc{j}" for j in range(coeff.size)]
            for i in range(coeff.size)
        ]
        self._df_dparam_names = [f"dc{i}dr" for i in range(coeff.size)]
        self._d2param_dr_names = [
            [
                f"dc{i}_2dr" if i == j else f"dc{i}dc{j}dr"
                for j in range(coeff.size)
            ]
            for i in range(coeff.size)
        ]

        self._n_raw = coeff.size + 1
        if mode == "periodic":
            self._initialize_periodic_bank()
        else:
            self._initialize_cutoff_bank()

    @classmethod
    def zeros(
        cls,
        types: Sequence[str],
        *,
        minimum: float,
        maximum: float,
        n_coeffs: int,
        degree: int = 3,
        boundary_mode: str = "periodic",
    ) -> "DihedralBSplinePotential":
        """Construct a zero-initialized dihedral force basis."""
        return cls(
            types,
            minimum=minimum,
            maximum=maximum,
            coefficients=np.zeros(int(n_coeffs), dtype=float),
            degree=degree,
            boundary_mode=boundary_mode,
        )

    def _initialize_periodic_bank(self) -> None:
        spacing = _PERIOD / float(self._n_raw)
        knot_ids = np.arange(
            -self.degree,
            self._n_raw + self.degree + 1,
            dtype=float,
        )
        self._raw_knots = _PERIOD_MIN + spacing * knot_ids
        n_extended = self._raw_knots.size - self.degree - 1
        eye = np.eye(n_extended, dtype=float)
        self._raw_splines = tuple(
            BSpline(
                self._raw_knots,
                eye[index],
                self.degree,
                extrapolate=False,
            )
            for index in range(n_extended)
        )
        raw_integrals = self._periodic_raw_integrals(
            np.asarray([_PERIOD_MAX - 1.0e-10], dtype=float)
        )[0]
        self._set_integral_constraint(raw_integrals)

    def _initialize_cutoff_bank(self) -> None:
        knots = np.linspace(
            self.minimum,
            self.maximum,
            self._n_raw + self.degree + 1,
            dtype=float,
        )
        bank = []
        supports = []
        for index in range(self._n_raw):
            local_knots = knots[index : index + self.degree + 2]
            bank.append(BSpline.basis_element(local_knots, extrapolate=False))
            supports.append((float(local_knots[0]), float(local_knots[-1])))
        self._raw_splines = tuple(bank)
        self._raw_supports = tuple(supports)
        raw_integrals = self._cutoff_raw_integrals(
            np.asarray([self.maximum], dtype=float)
        )[0]
        self._set_integral_constraint(raw_integrals)

    def _set_integral_constraint(self, raw_integrals: np.ndarray) -> None:
        raw_integrals = np.asarray(raw_integrals, dtype=float).reshape(-1)
        if raw_integrals.shape != (self._n_raw,):
            raise ValueError("invalid raw dihedral basis integral shape")
        dependent = float(raw_integrals[-1])
        if not np.isfinite(dependent) or abs(dependent) < 1.0e-14:
            raise ValueError("dependent dihedral basis has a zero integral")
        self._constraint_weights = raw_integrals[:-1] / dependent

    @staticmethod
    def normalize_angles(phi: np.ndarray) -> np.ndarray:
        """Normalize degrees to the half-open LAMMPS convention [-180, 180)."""
        arr = np.asarray(phi, dtype=float)
        return (arr - _PERIOD_MIN) % _PERIOD + _PERIOD_MIN

    def _fold_periodic_columns(self, extended: np.ndarray) -> np.ndarray:
        folded = np.zeros((extended.shape[0], self._n_raw), dtype=float)
        for index in range(extended.shape[1]):
            folded[:, index % self._n_raw] += extended[:, index]
        return folded

    def _periodic_raw_values(
        self,
        phi: np.ndarray,
        *,
        derivative: int = 0,
    ) -> np.ndarray:
        x = self.normalize_angles(phi).reshape(-1)
        if x.size == 0:
            return np.empty((0, self._n_raw), dtype=float)
        if derivative == 0 and hasattr(BSpline, "design_matrix"):
            extended = BSpline.design_matrix(
                x,
                self._raw_knots,
                self.degree,
                extrapolate=False,
            ).toarray()
        else:
            columns = []
            for spline in self._raw_splines:
                evaluator = spline if derivative == 0 else spline.derivative(derivative)
                columns.append(np.asarray(evaluator(x), dtype=float))
            extended = np.column_stack(columns)
        return self._fold_periodic_columns(np.nan_to_num(extended, nan=0.0))

    @cached_property
    def _periodic_antiderivatives(self):
        return tuple(spline.antiderivative() for spline in self._raw_splines)

    def _periodic_raw_integrals(self, phi: np.ndarray) -> np.ndarray:
        x = self.normalize_angles(phi).reshape(-1)
        if x.size == 0:
            return np.empty((0, self._n_raw), dtype=float)
        columns = []
        for anti in self._periodic_antiderivatives:
            columns.append(
                np.asarray(anti(x), dtype=float) - float(anti(_PERIOD_MIN))
            )
        return self._fold_periodic_columns(
            np.nan_to_num(np.column_stack(columns), nan=0.0)
        )

    def _cutoff_raw_values(
        self,
        phi: np.ndarray,
        *,
        derivative: int = 0,
    ) -> np.ndarray:
        x = self.normalize_angles(phi).reshape(-1)
        out = np.zeros((x.size, self._n_raw), dtype=float)
        for index, (spline, support) in enumerate(
            zip(self._raw_splines, self._raw_supports)
        ):
            active = (x >= support[0]) & (x <= support[1])
            if not np.any(active):
                continue
            evaluator = spline if derivative == 0 else spline.derivative(derivative)
            out[active, index] = np.nan_to_num(
                np.asarray(evaluator(x[active]), dtype=float),
                nan=0.0,
            )
        return out

    @cached_property
    def _cutoff_antiderivatives(self):
        return tuple(spline.antiderivative() for spline in self._raw_splines)

    def _cutoff_raw_integrals(self, phi: np.ndarray) -> np.ndarray:
        x = self.normalize_angles(phi).reshape(-1)
        out = np.zeros((x.size, self._n_raw), dtype=float)
        for index, (anti, support) in enumerate(
            zip(self._cutoff_antiderivatives, self._raw_supports)
        ):
            lo, hi = support
            total = float(anti(hi) - anti(lo))
            above = x >= hi
            out[above, index] = total
            active = (x > lo) & (x < hi)
            if np.any(active):
                out[active, index] = np.nan_to_num(
                    np.asarray(anti(x[active]) - anti(lo), dtype=float),
                    nan=0.0,
                )
        return out

    def _raw_values(self, phi: np.ndarray, *, derivative: int = 0) -> np.ndarray:
        if self.boundary_mode == "periodic":
            return self._periodic_raw_values(phi, derivative=derivative)
        return self._cutoff_raw_values(phi, derivative=derivative)

    def _raw_integrals(self, phi: np.ndarray) -> np.ndarray:
        if self.boundary_mode == "periodic":
            return self._periodic_raw_integrals(phi)
        return self._cutoff_raw_integrals(phi)

    def _constrain(self, raw: np.ndarray) -> np.ndarray:
        return (
            np.asarray(raw[:, :-1], dtype=float)
            - raw[:, -1, None] * self._constraint_weights[None, :]
        )

    def basis_values(self, phi: np.ndarray) -> np.ndarray:
        """Return ``dF/dtheta`` with shape ``phi.shape + (n_params,)``."""
        arr = np.asarray(phi, dtype=float)
        constrained = self._constrain(self._raw_values(arr))
        return constrained.reshape(arr.shape + (self.n_params(),))

    def basis_values_sparse(self, phi: np.ndarray):
        """Dihedral bases currently use the dense projector path."""
        return None

    def basis_derivatives(self, phi: np.ndarray) -> np.ndarray:
        """Return ``d(dF/dtheta)/dphi``."""
        arr = np.asarray(phi, dtype=float)
        constrained = self._constrain(self._raw_values(arr, derivative=1))
        return constrained.reshape(arr.shape + (self.n_params(),))

    def basis_integrals(self, phi: np.ndarray) -> np.ndarray:
        """Return integrals of the force bases from the -180 degree seam."""
        arr = np.asarray(phi, dtype=float)
        constrained = self._constrain(self._raw_integrals(arr))
        return constrained.reshape(arr.shape + (self.n_params(),))

    def force_grad(self, phi: np.ndarray) -> np.ndarray:
        return self.basis_values(phi)

    def energy_grad(self, phi: np.ndarray) -> np.ndarray:
        return -self.basis_integrals(phi)

    def energy_grad_sum(self, phi: np.ndarray) -> np.ndarray:
        arr = np.asarray(phi, dtype=float)
        if arr.ndim > 1:
            return self.energy_grad_sum_by_sample(arr)
        if arr.size == 0:
            return np.zeros(self.n_params(), dtype=float)
        return -np.asarray(self.basis_integrals(arr), dtype=float).sum(axis=0)

    def gauge_free_energy_grad_sum(self, phi: np.ndarray) -> np.ndarray:
        return self.energy_grad_sum(phi)

    def force(self, phi: np.ndarray) -> np.ndarray:
        basis = self.basis_values(phi)
        return np.asarray(basis @ self._params, dtype=float)

    def value(self, phi: np.ndarray) -> np.ndarray:
        grad = self.energy_grad(phi)
        return np.asarray(grad @ self._params, dtype=float)

    def is_param_linear(self) -> np.ndarray:
        return np.ones(self.n_params(), dtype=bool)

    def table_metadata(self) -> dict[str, object]:
        """Return self-describing metadata written into AceCG tables."""
        return {
            "boundary_mode": self.boundary_mode,
            "model_min": self.minimum,
            "model_max": self.maximum,
            "degree": self.degree,
            "n_coeffs": self.n_params(),
        }

    def __getattr__(self, name: str):
        match = re.fullmatch(r"(?:d?c)(\d+)", name)
        if match:
            index = int(match.group(1))
            if not 0 <= index < self.n_params():
                raise AttributeError(name)

            def energy_derivative(phi, _index=index):
                return self.energy_grad(phi)[..., _index]

            return energy_derivative

        match = re.fullmatch(r"dc(\d+)dr", name)
        if match:
            index = int(match.group(1))
            if not 0 <= index < self.n_params():
                raise AttributeError(name)

            def force_derivative(phi, _index=index):
                return self.force_grad(phi)[..., _index]

            return force_derivative

        if name == "dr":
            return self.force
        if re.fullmatch(r"dc\d+_2", name) or re.fullmatch(
            r"dc\d+dc\d+", name
        ) or re.fullmatch(r"dc\d+_2dr", name) or re.fullmatch(
            r"dc\d+dc\d+dr", name
        ):
            return lambda phi: np.zeros_like(np.asarray(phi, dtype=float))
        raise AttributeError(
            f"{self.__class__.__name__} has no attribute {name!r}"
        )


__all__ = ["DihedralBSplinePotential"]
