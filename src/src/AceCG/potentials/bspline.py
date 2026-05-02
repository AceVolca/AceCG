"""Force-basis B-spline potential.

Stored coefficients θ parameterize the scalar force directly:
    F_θ(r) = Σ_i θ_i B_i(r)

Energy is derived by integration:
    U_θ(r) = -Σ_i θ_i I_i(r)

Gauge is controlled explicitly by ``self.bonded``:
    bonded=False  → U(r_max) = 0
    bonded=True   → u_min = 0

Dynamic derivative accessors:
    dc{i}(r)    → ∂U/∂c_i
    dc{i}dr(r)  → ∂F/∂c_i = B_i(r)
    dr(r)       → F(r) = force(r)
"""
import re
from functools import lru_cache

import numpy as np
from scipy.interpolate import BSpline, PPoly

from .base import BasePotential


class BSplinePotential(BasePotential):
    """Force-basis B-spline potential.

    Coefficients represent force: F(r) = B(r)^T θ.
    Energy is derived by analytic integration of the force basis.
    """

    def __init__(
        self,
        typ1,
        typ2,
        knots: np.ndarray,
        coefficients: np.ndarray,
        degree: int,
        cutoff: float,
        bonded: bool = False,
    ) -> None:
        """Initialize a force-basis B-spline potential.

        Parameters
        ----------
        typ1, typ2 : int or str
            Interaction type labels.
        knots : np.ndarray
            Clamped knot vector accepted by ``scipy.interpolate.BSpline``.
        coefficients : np.ndarray
            Force-basis coefficients. Their length is the number of trainable
            parameters.
        degree : int
            Polynomial degree of the B-spline basis.
        cutoff : float
            Cutoff/reference coordinate for nonbonded gauge handling.
        bonded : bool, default=False
            If ``True``, choose an energy gauge appropriate for bonded terms.
            If ``False``, anchor the energy at the cutoff side.
        """
        super().__init__(bonded=bonded)
        self.typ1  = typ1
        self.typ2  = typ2
        self.cutoff = float(cutoff)
        self.spline = BSpline(knots, coefficients, degree)

        n_params = len(coefficients)
        self._params_to_scale = list(range(n_params))
        self._param_names = [f"c{i}" for i in range(n_params)]
        self._dparam_names = [f"dc{i}" for i in range(n_params)]
        self._d2param_names = [
            [f"dc{i}_2" if i == j else f"dc{i}dc{j}" for j in range(n_params)]
            for i in range(n_params)
        ]
        self._df_dparam_names = [f"dc{i}dr" for i in range(n_params)]
        self._d2param_dr_names = [
            [f"dc{i}_2dr" if i == j else f"dc{i}dc{j}dr" for j in range(n_params)]
            for i in range(n_params)
        ]

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def n_coeff_from_range(minimum: float, maximum: float, resolution: float, degree: int) -> int:
        """Compute the coefficient count implied by a uniform spline range.

        Parameters
        ----------
        minimum, maximum : float
            Inclusive coordinate range.
        resolution : float
            Desired grid spacing used by AceCG FM configs.
        degree : int
            B-spline polynomial degree.

        Returns
        -------
        int
            Number of force-basis coefficients.
        """
        nbin = int(round((float(maximum) - float(minimum)) / float(resolution))) + 1
        n_coeff = nbin + int(degree) - 2
        if n_coeff <= 0:
            raise ValueError(
                f"Invalid BSpline parameter count from min/max/res/degree: {minimum}, {maximum}, {resolution}, {degree}"
            )
        return int(n_coeff)

    @staticmethod
    def clamped_uniform_knots(minimum: float, maximum: float, n_coeff: int, degree: int) -> np.ndarray:
        """Build a clamped uniform knot vector.

        Parameters
        ----------
        minimum, maximum : float
            Coordinate support of the spline.
        n_coeff : int
            Number of B-spline coefficients.
        degree : int
            Polynomial degree.

        Returns
        -------
        np.ndarray
            Knot vector suitable for ``scipy.interpolate.BSpline``.
        """
        n_internal = int(n_coeff) - int(degree) - 1
        if n_internal <= 0:
            return np.r_[np.full(int(degree) + 1, minimum), np.full(int(degree) + 1, maximum)].astype(float)
        interior = np.linspace(minimum, maximum, n_internal + 2)[1:-1]
        return np.r_[np.full(int(degree) + 1, minimum), interior, np.full(int(degree) + 1, maximum)].astype(float)

    @classmethod
    def from_range(
        cls,
        typ1,
        typ2,
        *,
        minimum: float,
        maximum: float,
        resolution: float,
        degree: int,
        bonded: bool = False,
    ) -> "BSplinePotential":
        """Create a zero-initialized spline from range/resolution settings.

        Parameters
        ----------
        typ1, typ2 : int or str
            Interaction type labels.
        minimum, maximum : float
            Coordinate support of the spline.
        resolution : float
            Desired grid spacing.
        degree : int
            B-spline polynomial degree.
        bonded : bool, default=False
            Whether to use bonded energy gauge handling.

        Returns
        -------
        BSplinePotential
            New force-basis spline with zero coefficients.
        """
        n_coeff = cls.n_coeff_from_range(minimum=minimum, maximum=maximum, resolution=resolution, degree=degree)
        knots = cls.clamped_uniform_knots(minimum=minimum, maximum=maximum, n_coeff=n_coeff, degree=degree)
        coeff = np.zeros(n_coeff, dtype=float)
        return cls(
            typ1=typ1,
            typ2=typ2,
            knots=knots,
            coefficients=coeff,
            degree=degree,
            cutoff=float(maximum),
            bonded=bonded,
        )

    @classmethod
    def from_order_spec(
        cls,
        typ1,
        typ2,
        *,
        minimum: float,
        maximum: float,
        resolution: float,
        order: int,
        bonded: bool = False,
    ) -> "BSplinePotential":
        """Build from the ``order = degree + 1`` convention used in FM configs.

        Parameters
        ----------
        typ1, typ2 : int or str
            Interaction type labels.
        minimum, maximum : float
            Coordinate support of the spline.
        resolution : float
            Desired grid spacing.
        order : int
            B-spline order, equal to ``degree + 1``.
        bonded : bool, default=False
            Whether to use bonded energy gauge handling.

        Returns
        -------
        BSplinePotential
            New force-basis spline with zero coefficients.
        """
        ord_i = int(order)
        if ord_i < 1:
            raise ValueError(f"BSpline order must be >= 1, got {order}")
        degree = ord_i - 1
        n_coeff = cls.n_coeff_from_range(
            minimum=minimum,
            maximum=maximum,
            resolution=resolution,
            degree=ord_i,
        )
        knots = cls.clamped_uniform_knots(minimum=minimum, maximum=maximum, n_coeff=n_coeff, degree=degree)
        coeff = np.zeros(n_coeff, dtype=float)
        return cls(
            typ1=typ1,
            typ2=typ2,
            knots=knots,
            coefficients=coeff,
            degree=degree,
            cutoff=float(maximum),
            bonded=bonded,
        )

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    @property
    def _params(self) -> np.ndarray:
        return self.spline.c

    @_params.setter
    def _params(self, new_params: np.ndarray):
        self.spline.c = np.asarray(new_params, dtype=float)

    @property
    def degree(self) -> int:
        """Return the B-spline polynomial degree."""
        return self.spline.k

    @property
    def knots(self) -> np.ndarray:
        """Return the underlying B-spline knot vector."""
        return self.spline.t

    @property
    def _r_ref(self) -> float:
        """Reference coordinate for the cutoff-anchored raw antiderivative."""
        if np.isfinite(self.cutoff):
            return float(min(self._support_max, self.cutoff))
        return self._support_max

    @property
    def _support_min(self) -> float:
        return float(self.knots[self.degree])

    @property
    def _support_max(self) -> float:
        return float(self.knots[-self.degree - 1])

    def _minimum_gauge_coordinate(self) -> float:
        lo = self._support_min
        hi = self._r_ref
        if hi <= lo:
            return lo

        anti = self.spline.antiderivative()
        piecewise = PPoly.from_spline(anti, extrapolate=False)
        roots = np.asarray(piecewise.derivative(1).roots(extrapolate=False), dtype=float)
        roots = roots[np.isfinite(roots)]
        roots = roots[(roots >= lo - 1.0e-12) & (roots <= hi + 1.0e-12)]
        roots = np.clip(roots, lo, hi)
        knots = np.asarray(piecewise.x, dtype=float)
        knots = knots[(knots >= lo - 1.0e-12) & (knots <= hi + 1.0e-12)]
        candidates = np.r_[lo, roots, hi, knots]
        candidates = np.unique(np.round(candidates, 14))
        values = -(piecewise(candidates) - piecewise(self._r_ref))
        return float(candidates[int(np.nanargmin(values))])

    # ------------------------------------------------------------------
    # Primary evaluation: force is direct, energy is integrated
    # ------------------------------------------------------------------

    def force(self, r: np.ndarray) -> np.ndarray:
        """Evaluate the modeled scalar force F_θ(r) = B(r)^T θ."""
        return self.spline(r)

    def is_param_linear(self) -> np.ndarray:
        """Return an all-true mask because force-basis coefficients are linear."""
        return np.ones(self.n_params(), dtype=bool)

    def value(self, r: np.ndarray) -> np.ndarray:
        """Compute energy from the antiderivative of the full spline."""
        r_flat = np.asarray(r, dtype=float).reshape(-1)
        if r_flat.size == 0:
            return np.empty(0, dtype=float)
        anti = self.spline.antiderivative()
        if self.bonded:
            reference = self._minimum_gauge_coordinate()
        else:
            reference = self._r_ref
        values = -(anti(r_flat) - anti(reference))
        if np.isfinite(self.cutoff):
            values[r_flat > self.cutoff] = 0.0
        return values

    def energy_grad(self, r: np.ndarray) -> np.ndarray:
        """Return ``dU/dtheta`` via the integrated B-spline basis matrix."""
        r_flat = np.asarray(r, dtype=float).reshape(-1)
        grad = -self.basis_integrals(r_flat)
        if self.bonded:
            grad += self.basis_integrals(np.asarray([self._minimum_gauge_coordinate()], dtype=float))
        if np.isfinite(self.cutoff):
            grad[r_flat > self.cutoff, :] = 0.0
        return grad

    def energy_grad_sum(self, r: np.ndarray) -> np.ndarray:
        """Return ``Σ_samples dU/dtheta`` without materializing the full matrix."""
        r_flat = np.asarray(r, dtype=float).reshape(-1)
        if np.isfinite(self.cutoff):
            r_eval = r_flat[r_flat <= self.cutoff]
        else:
            r_eval = r_flat
        if r_eval.size == 0:
            return np.zeros(self.n_params(), dtype=float)
        gauge_shift = np.zeros(self.n_params(), dtype=float)
        if self.bonded:
            gauge_shift = self.basis_integrals(np.asarray([self._minimum_gauge_coordinate()], dtype=float))[0]
        if hasattr(BSpline, "design_matrix"):
            try:
                payload = self._integral_basis_data()
                if payload is not None:
                    anti_knots, anti_degree, extrapolate, coeff_matrix, ref = payload
                    dm = BSpline.design_matrix(
                        r_eval,
                        anti_knots,
                        anti_degree,
                        extrapolate=extrapolate,
                    )
                    row_sum = np.asarray(dm.sum(axis=0), dtype=float).reshape(-1)
                    return -(row_sum @ coeff_matrix - r_eval.size * ref) + r_eval.size * gauge_shift
            except (TypeError, ValueError):
                # The sparse SciPy fast path is optional; fall back to the dense
                # path when design_matrix/reduction semantics are incompatible.
                pass
        return -np.asarray(self.basis_integrals(r_eval), dtype=float).sum(axis=0) + r_eval.size * gauge_shift

    def _raw_energy_grad_sum(self, r: np.ndarray) -> np.ndarray:
        """Return the summed integrated-basis score without bonded gauge terms."""
        r_flat = np.asarray(r, dtype=float).reshape(-1)
        if np.isfinite(self.cutoff):
            r_eval = r_flat[r_flat <= self.cutoff]
        else:
            r_eval = r_flat
        if r_eval.size == 0:
            return np.zeros(self.n_params(), dtype=float)
        if hasattr(BSpline, "design_matrix"):
            try:
                payload = self._integral_basis_data()
                if payload is not None:
                    anti_knots, anti_degree, extrapolate, coeff_matrix, ref = payload
                    dm = BSpline.design_matrix(
                        r_eval,
                        anti_knots,
                        anti_degree,
                        extrapolate=extrapolate,
                    )
                    row_sum = np.asarray(dm.sum(axis=0), dtype=float).reshape(-1)
                    return -(row_sum @ coeff_matrix - r_eval.size * ref)
            except (TypeError, ValueError):
                pass
        return -np.asarray(self.basis_integrals(r_eval), dtype=float).sum(axis=0)

    def gauge_free_energy_grad_sum(self, r: np.ndarray) -> np.ndarray:
        """Return the gauge-free energy gradient for bonded B-splines."""
        return self._raw_energy_grad_sum(r)

    def is_gauge_free_energy_grad_cacheable(self) -> np.ndarray:
        """Return an all-true mask for the force-basis gauge-free channels."""
        return np.ones(self.n_params(), dtype=bool)

    def force_grad(self, r: np.ndarray):
        """Return ``dF/dtheta`` via the force-basis design matrix."""
        r_flat = np.asarray(r, dtype=float).reshape(-1)
        if r_flat.size == 0:
            return np.empty((0, self.n_params()), dtype=float)
        if np.isfinite(self.cutoff) and np.any(r_flat > self.cutoff):
            dense = self.basis_values(r_flat)
            dense[r_flat > self.cutoff, :] = 0.0
            return dense
        sparse = self.basis_values_sparse(r_flat)
        if sparse is not None:
            return sparse
        return self.basis_values(r_flat)

    # ------------------------------------------------------------------
    # Basis function accessors
    # ------------------------------------------------------------------

    def basis_function(self, i: int, r: np.ndarray) -> np.ndarray:
        """Evaluate the i-th B-spline basis function B_i(r)."""
        bank = self._basis_bank(0)
        return bank[i](np.asarray(r, dtype=float))

    @lru_cache(maxsize=16)
    def _basis_bank(self, deriv_order: int):
        """Cache of per-basis-function splines.

        deriv_order > 0: d^n B_i / dr^n
        deriv_order = 0: B_i(r)
        deriv_order < 0: |deriv_order|-fold antiderivative of B_i
        """
        n_params = len(self._params)
        eye = np.eye(n_params, dtype=float)
        bank = []
        for i in range(n_params):
            sp = BSpline(self.knots, eye[i], self.degree, extrapolate=self.spline.extrapolate)
            if deriv_order > 0:
                sp = sp.derivative(deriv_order)
            elif deriv_order < 0:
                sp = sp.antiderivative(-deriv_order)
            bank.append(sp)
        return bank

    def basis_values(self, r: np.ndarray) -> np.ndarray:
        """Force basis matrix B_i(r), shape (n_samples, n_params).

        Under force-basis semantics: B_i(r) = ∂F/∂c_i.
        """
        r = np.asarray(r, dtype=float).ravel()
        n_params = len(self._params)
        if r.size == 0:
            return np.empty((0, n_params), dtype=float)
        if hasattr(BSpline, "design_matrix"):
            try:
                dm = BSpline.design_matrix(r, self.knots, self.degree, extrapolate=self.spline.extrapolate)
                return np.asarray(dm.toarray(), dtype=float)
            except Exception:
                pass
        bank = self._basis_bank(0)
        out = np.empty((r.size, len(bank)), dtype=float)
        for i, sp in enumerate(bank):
            out[:, i] = sp(r)
        return out

    def basis_values_sparse(self, r: np.ndarray):
        """Sparse force basis matrix (for pair projector optimization)."""
        r = np.asarray(r, dtype=float).ravel()
        n_params = len(self._params)
        if r.size == 0:
            return None
        if hasattr(BSpline, "design_matrix"):
            try:
                return BSpline.design_matrix(r, self.knots, self.degree, extrapolate=self.spline.extrapolate)
            except Exception:
                return None
        return None

    def basis_derivatives(self, r: np.ndarray) -> np.ndarray:
        """Derivative of force basis dB_i/dr, shape (n_samples, n_params)."""
        r = np.asarray(r, dtype=float).ravel()
        bank = self._basis_bank(1)
        if r.size == 0:
            return np.empty((0, len(bank)), dtype=float)
        out = np.empty((r.size, len(bank)), dtype=float)
        for i, sp in enumerate(bank):
            out[:, i] = sp(r)
        return out

    @lru_cache(maxsize=1)
    def _integral_basis_data(self):
        """Return cached data for vectorized integrated-basis evaluation."""
        bank = self._basis_bank(-1)
        if not bank:
            return None
        anti0 = bank[0]
        n_active_basis = anti0.t.size - anti0.k - 1
        coeff_matrix = np.column_stack(
            [np.asarray(sp.c[:n_active_basis], dtype=float) for sp in bank]
        )
        ref = np.asarray([sp(self._r_ref) for sp in bank], dtype=float)
        return anti0.t, anti0.k, anti0.extrapolate, coeff_matrix, ref

    def basis_integrals(self, r: np.ndarray) -> np.ndarray:
        """Raw integrated force basis I_i(r) = ∫_{r_ref}^{r} B_i(ξ) dξ.

        Shape (n_samples, n_params).
        ``r_ref`` is the nonbonded cutoff-side raw reference.
        """
        r = np.asarray(r, dtype=float).ravel()
        n_params = len(self._params)
        if r.size == 0:
            return np.empty((0, n_params), dtype=float)
        if hasattr(BSpline, "design_matrix"):
            try:
                payload = self._integral_basis_data()
                if payload is not None:
                    anti_knots, anti_degree, extrapolate, coeff_matrix, ref = payload
                    dm = BSpline.design_matrix(
                        r,
                        anti_knots,
                        anti_degree,
                        extrapolate=extrapolate,
                    )
                    out = np.asarray(dm @ coeff_matrix, dtype=float)
                    out -= ref[None, :]
                    return out
            except Exception:
                pass
        bank = self._basis_bank(-1)
        r_ref = self._r_ref
        out = np.empty((r.size, len(bank)), dtype=float)
        for i, anti in enumerate(bank):
            out[:, i] = anti(r) - anti(r_ref)
        return out

    # ------------------------------------------------------------------
    # Dynamic derivative accessors (__getattr__)
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        """Dynamic accessors for force-basis parameter sensitivities.

        Under force-basis semantics (coefficients parameterize force):
          - c{i}(r) / dc{i}(r):  ∂U/∂c_i = -I_i(r)
          - dc{i}_2(r):           ∂²U/∂c_i² = 0  (linear model)
          - dc{i}dc{j}(r):       ∂²U/∂c_i∂c_j = 0
          - dr(r):                F(r) = force(r)
          - dc{i}dr(r):           ∂F/∂c_i = B_i(r)
          - dc{i}_2dr(r):         ∂²F/∂c_i² = 0
          - dc{i}dc{j}dr(r):     ∂²F/∂c_i∂c_j = 0
        """
        # Energy derivative wrt coefficient: dc{i}(r) → ∂U/∂c_i.
        m = re.fullmatch(r'(?:d?c)(\d+)', name)
        if m:
            i = int(m.group(1))
            n_params = len(self._params)
            if not (0 <= i < n_params):
                raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}' (index {i} out of range 0..{n_params-1})")

            def energy_deriv_ci(r, _i=i):
                anti = self._basis_bank(-1)[_i]
                arr = np.asarray(r, dtype=float)
                flat = arr.ravel()
                if self.bonded:
                    reference = self._minimum_gauge_coordinate()
                else:
                    reference = self._r_ref
                return -(anti(flat) - anti(reference)).reshape(arr.shape)
            return energy_deriv_ci

        # Second derivative wrt same coefficient twice: dc{i}_2 → 0
        m = re.fullmatch(r'dc(\d+)_2', name)
        if m:
            i = int(m.group(1))
            n_params = len(self._params)
            if not (0 <= i < n_params):
                raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}' (index {i} out of range 0..{n_params-1})")

            def zero_fn(r):
                return np.zeros_like(np.asarray(r, dtype=float), dtype=float)
            return zero_fn

        # Mixed second derivative: dc{i}dc{j} → 0
        m = re.fullmatch(r'dc(\d+)dc(\d+)', name)
        if m:
            i = int(m.group(1))
            j = int(m.group(2))
            n_params = len(self._params)
            if not (0 <= i < n_params and 0 <= j < n_params):
                raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}' (indices out of range)")

            def zero_fn(r):
                return np.zeros_like(np.asarray(r, dtype=float), dtype=float)
            return zero_fn

        # Force channel: dr → F(r)
        if name == "dr":
            def dr_fn(r):
                return self.force(np.asarray(r, dtype=float))
            return dr_fn

        # Force derivative wrt coefficient: dc{i}dr → ∂F/∂c_i = B_i(r)
        m = re.fullmatch(r'dc(\d+)dr', name)
        if m:
            i = int(m.group(1))
            n_params = len(self._params)
            if not (0 <= i < n_params):
                raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}' (index {i} out of range)")

            def force_deriv_ci(r, _i=i):
                sp = self._basis_bank(0)[_i]
                arr = np.asarray(r, dtype=float)
                flat = arr.ravel()
                return sp(flat).reshape(arr.shape)
            return force_deriv_ci

        # Linear-in-parameter model ⇒ higher-order channels are zero.
        m = re.fullmatch(r'dc(\d+)_2dr', name)
        if m:
            def zero_fn(r):
                return np.zeros_like(np.asarray(r, dtype=float), dtype=float)
            return zero_fn

        m = re.fullmatch(r'dc(\d+)dc(\d+)dr', name)
        if m:
            def zero_fn(r):
                return np.zeros_like(np.asarray(r, dtype=float), dtype=float)
            return zero_fn

        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
