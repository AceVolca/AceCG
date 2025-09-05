#AceCG/potentials/bsplinepotential.py
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
        self._param_names = [f"c{i}" for i in range(n_params)]
        self._dparam_names = [f"dc{i}" for i in range(n_params)]
        self._d2param_names = [
            [f"dc{i}_2" if i == j else f"dc{i}dc{j}" for j in range(n_params)]
            for i in range(n_params)
        ]

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