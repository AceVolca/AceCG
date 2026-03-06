#AceCG/potentials/bspline.py
import re
from functools import lru_cache
from .base import BasePotential
import numpy as np
from scipy.interpolate import BSpline

class BSplinePotential(BasePotential):
    """
    Class representing a B-spline potential.
    This potential uses B-spline interpolation to model interactions between particles.
    """

    def __init__(self, typ1, typ2, knots: np.ndarray, coefficients: np.ndarray, degree: int, cutoff: float) -> None:
        """
        Initialize the B-spline potential with given knots, coefficients, and degree.

        :param knots: Knot vector for the B-spline.
        :param coefficients: Coefficients for the B-spline basis functions.
        :param degree: Degree of the B-spline.
        :param cutoff: Cutoff distance for the potential.
        """
        self.typ1  = typ1
        self.typ2  = typ2
        self.cutoff = cutoff
        self.spline = BSpline(knots, coefficients, degree)

        n_params = len(coefficients)
        self._params_to_scale = list(range(n_params))
        self._param_names = [f"c{i}" for i in range(n_params)]
        self._dparam_names = [f"dc{i}" for i in range(n_params)]
        self._d2param_names = [
            [f"dc{i}_2" if i == j else f"dc{i}dc{j}" for j in range(n_params)]
            for i in range(n_params)
        ]
        self._dr_name = "dr"
        self._dparam_dr_names = [f"dc{i}dr" for i in range(n_params)]
        self._d2param_dr_names = [
            [f"dc{i}_2dr" if i == j else f"dc{i}dc{j}dr" for j in range(n_params)]
            for i in range(n_params)
        ]

    @staticmethod
    def n_coeff_from_range(minimum: float, maximum: float, resolution: float, degree: int) -> int:
        nbin = int(round((float(maximum) - float(minimum)) / float(resolution))) + 1
        n_coeff = nbin + int(degree) - 2
        if n_coeff <= 0:
            raise ValueError(
                f"Invalid BSpline parameter count from min/max/res/degree: {minimum}, {maximum}, {resolution}, {degree}"
            )
        return int(n_coeff)

    @staticmethod
    def clamped_uniform_knots(minimum: float, maximum: float, n_coeff: int, degree: int) -> np.ndarray:
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
    ) -> "BSplinePotential":
        n_coeff = cls.n_coeff_from_range(minimum=minimum, maximum=maximum, resolution=resolution, degree=degree)
        knots = cls.clamped_uniform_knots(minimum=minimum, maximum=maximum, n_coeff=n_coeff, degree=degree)
        coeff = np.zeros(n_coeff, dtype=float)
        return cls(typ1=typ1, typ2=typ2, knots=knots, coefficients=coeff, degree=degree, cutoff=float(maximum))

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
    ) -> "BSplinePotential":
        """Build from order (= degree + 1) convention used in FM configs."""
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
        return cls(typ1=typ1, typ2=typ2, knots=knots, coefficients=coeff, degree=degree, cutoff=float(maximum))

    @property
    def _params(self) -> np.ndarray:
        return self.spline.c
    @_params.setter
    def _params(self, new_params: np.ndarray):
        self.spline.c = new_params

    @property
    def degree(self) -> int:
        return self.spline.k
    
    @property
    def knots(self) -> np.ndarray:
        return self.spline.t

    def value(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the B-spline potential at a distance r.

        :param r: Distance between two particles.
        :return: The value of the B-spline potential at distance r.
        """
        return self.spline(r)
    
    def force(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the force (negative derivative) of the B-spline potential at a distance r.

        :param r: Distance between two particles.
        :return: The force of the B-spline potential at distance r.
        """
        return -self.spline(r, 1)

    def basis_function(self, i: int, r: np.ndarray) -> np.ndarray:
        """
        Compute the i-th B-spline basis function at a distance r.

        :param i: Index of the basis function.
        :param r: Distance at which to evaluate the basis function.
        :return: The value of the i-th B-spline basis function at distance r.
        """
        basis_spline = BSpline.basis_element(self.knots[i:i+self.degree+2], extrapolate=self.spline.extrapolate)
        return basis_spline(r)

    @lru_cache(maxsize=16)
    def _basis_bank(self, deriv_order: int):
        n_params = len(self._params)
        eye = np.eye(n_params, dtype=float)
        bank = []
        for i in range(n_params):
            sp = BSpline(self.knots, eye[i], self.degree, extrapolate=self.spline.extrapolate)
            if deriv_order > 0:
                sp = sp.derivative(deriv_order)
            bank.append(sp)
        return bank

    def basis_values(self, r: np.ndarray) -> np.ndarray:
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
        r = np.asarray(r, dtype=float).ravel()
        bank = self._basis_bank(1)
        if r.size == 0:
            return np.empty((0, len(bank)), dtype=float)
        out = np.empty((r.size, len(bank)), dtype=float)
        for i, sp in enumerate(bank):
            out[:, i] = sp(r)
        return out
    
    def __getattr__(self, name: str):
        """
        Dynamic accessors for parameter sensitivities:
          - c{i}(r):     returns B_i(r)         == dV/dc_i
          - dc{i}(r):    alias of c{i}(r)       == dV/dc_i
          - dc{i}_2(r):  second derivative wrt c_i twice -> 0
          - dc{i}dc{j}(r): mixed second derivative wrt c_i,c_j -> 0
        
        # TODO: attributes for force-matching objectives. These attributes are not (I think?)
        used in classical force-matching (because it is linear solver) but will be used in 
        iterative CD-FM force-matching. These attributes are:
          - dr:          == -dV/dr == force(r) # Note the sign here!
          - dc{i}dr:     == d^2V/dc_i dr == -dB_i(r)/dr
          - dc{i}2_dr:   == d^3V/dc_i^2 dr == -d^2B_i(r)/dr^2
          - dc{i}dc{j}dr: == d^3V/dc_idc_j dr == -d^2B_i(r)/dr^2 if i==j else 0 
          
        """
        # First-derivative: c{i} or dc{i}
        m = re.fullmatch(r'(?:d?c)(\d+)', name)
        if m:
            i = int(m.group(1))
            n_params = len(self._params)
            if not (0 <= i < n_params):
                raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}' (index {i} out of range 0..{n_params-1})")

            def basis_fn(r, i=i):
                # ∂V/∂c_i = B_i(r)
                return self.basis_function(i, np.asarray(r, dtype=float))
            return basis_fn

        # Second-derivative: dc{i}_2
        m = re.fullmatch(r'dc(\d+)_2', name)
        if m:
            i = int(m.group(1))
            n_params = len(self._params)
            if not (0 <= i < n_params):
                raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}' (index {i} out of range 0..{n_params-1})")

            def zero_fn(r):
                r = np.asarray(r, dtype=float)
                return np.zeros_like(r, dtype=float)
            return zero_fn

        # Mixed second-derivative: dc{i}dc{j}
        m = re.fullmatch(r'dc(\d+)dc(\d+)', name)
        if m:
            i = int(m.group(1))
            j = int(m.group(2))
            n_params = len(self._params)
            if not (0 <= i < n_params and 0 <= j < n_params):
                raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}' (indices out of range)")

            def zero_fn(r):
                r = np.asarray(r, dtype=float)
                return np.zeros_like(r, dtype=float)
            return zero_fn

        # Force channel: dr == -dV/dr
        if name == "dr":
            def dr_fn(r):
                r = np.asarray(r, dtype=float)
                return self.force(r)
            return dr_fn

        # FM channel: d(force)/d(c_i) = -dB_i/dr
        m = re.fullmatch(r'dc(\d+)dr', name)
        if m:
            i = int(m.group(1))
            n_params = len(self._params)
            if not (0 <= i < n_params):
                raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}' (index {i} out of range)")

            def dforce_dci(r, i=i):
                arr = np.asarray(r, dtype=float)
                flat = arr.ravel()
                out = -self.basis_derivatives(flat)[:, i]
                return out.reshape(arr.shape)
            return dforce_dci

        # Linear-in-parameter model => these are zero channels for iterative FM.
        m = re.fullmatch(r'dc(\d+)_2dr', name)
        if m:
            def zero_fn(r):
                r = np.asarray(r, dtype=float)
                return np.zeros_like(r, dtype=float)
            return zero_fn

        m = re.fullmatch(r'dc(\d+)dc(\d+)dr', name)
        if m:
            def zero_fn(r):
                r = np.asarray(r, dtype=float)
                return np.zeros_like(r, dtype=float)
            return zero_fn

        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
