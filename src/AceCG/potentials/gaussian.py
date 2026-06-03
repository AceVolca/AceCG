# AceCG/potentials/gaussian.py
"""Single-Gaussian pair potential centered at ``r0`` with amplitude ``A`` and width ``sigma``."""
import numpy as np
from .base import BasePotential

class GaussianPotential(BasePotential):
    """Single-Gaussian pair potential ``U(r) = A * exp(-(r - r0)^2 / (2 sigma^2))``.

    Parameters are stored as ``[A, r0, sigma]``. Only ``A`` is linear.
    """
    def __init__(self, typ1, typ2, A, r0, sigma, cutoff):
        """Initialize a normalized Gaussian pair potential.

        Parameters
        ----------
        typ1, typ2 : int or str
            Interaction type labels for the two particles.
        A : float
            Gaussian amplitude.
        r0 : float
            Gaussian center.
        sigma : float
            Gaussian width. Must be positive for physically meaningful values.
        cutoff : float
            Distance cutoff stored for compatibility with forcefield writers.
        """
        super().__init__()
        self.typ1  = typ1
        self.typ2  = typ2
        self.cutoff = cutoff
        self._params = np.array([A, r0, sigma])
        self._params_to_scale = [0]
        self._param_names = ["A", "r0", "sigma"]
        self._dparam_names = ["dA", "dr0", "dsigma"]
        self._d2param_names = [
            ["dA_2", "dAdr0", "dAdsigma"],
            ["dAdr0", "dr0_2", "dr0dsigma"],
            ["dAdsigma", "dr0dsigma", "dsigma_2"]
        ]

    def is_param_linear(self) -> np.ndarray:
        """Return which Gaussian parameters enter the energy linearly."""
        return np.array([True, False, False], dtype=bool)

    def value(self, r):
        """Evaluate the Gaussian potential energy.

        Parameters
        ----------
        r : array-like
            Pair distance values.

        Returns
        -------
        np.ndarray
            Energy values at ``r``.
        """
        r = np.asarray(r, dtype=float)
        A, r0, sigma = self._params
        x = r - r0
        return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))

    def force(self, r):
        """Evaluate the scalar force ``-dU/dr`` for the Gaussian."""
        r = np.asarray(r, dtype=float)
        A, r0, sigma = self._params
        x = r - r0
        return A / (sigma**3 * np.sqrt(2*np.pi)) * x * np.exp(-x**2 / (2 * sigma**2))

    def force_grad(self, r):
        """Return the explicit force Jacobian ``dF/d[A, r0, sigma]``."""
        r_arr = np.asarray(r, dtype=float)
        A, r0, sigma = self._params
        x = r_arr - r0
        pref = np.exp(-x**2 / (2.0 * sigma**2)) / np.sqrt(2.0 * np.pi)
        grad = np.empty(r_arr.shape + (3,), dtype=float)
        grad[..., 0] = x * pref / sigma**3
        grad[..., 1] = A * pref * (x**2 / sigma**2 - 1.0) / sigma**3
        grad[..., 2] = A * x * pref * (x**2 - 3.0 * sigma**2) / sigma**6
        return grad

    def dA(self, r):
        """Return ``dU/dA`` evaluated at ``r``."""
        r = np.asarray(r, dtype=float)
        _, r0, sigma = self._params
        x = r - r0
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))

    def dr0(self, r):
        """Return ``dU/dr0`` evaluated at ``r``."""
        r = np.asarray(r, dtype=float)
        A, r0, sigma = self._params
        x = r - r0
        return A * x / (sigma**3 * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))

    def dsigma(self, r):
        """Return ``dU/dsigma`` evaluated at ``r``."""
        r = np.asarray(r, dtype=float)
        A, r0, sigma = self._params
        x = r - r0
        phi = np.exp(-x**2 / (2 * sigma**2))
        return A * phi / np.sqrt(2 * np.pi) * (x**2 - sigma**2) / sigma**4

    def dA_2(self, r):
        """Return ``d2U/dA2``, which is zero for this potential."""
        return np.zeros_like(np.asarray(r, dtype=float), dtype=float)

    def dAdr0(self, r):
        """Return the mixed derivative ``d2U/dA dr0``."""
        r = np.asarray(r, dtype=float)
        _, r0, sigma = self._params
        x = r - r0
        phi = np.exp(-x**2 / (2 * sigma**2))
        return x / (sigma**3 * np.sqrt(2 * np.pi)) * phi

    def dAdsigma(self, r):
        """Return the mixed derivative ``d2U/dA dsigma``."""
        r = np.asarray(r, dtype=float)
        _, r0, sigma = self._params
        x = r - r0
        phi = np.exp(-x**2 / (2 * sigma**2))
        return (x**2 - sigma**2) / (sigma**4 * np.sqrt(2 * np.pi)) * phi

    def dr0_2(self, r):
        """Return ``d2U/dr0^2`` evaluated at ``r``."""
        r = np.asarray(r, dtype=float)
        A, r0, sigma = self._params
        x = r - r0
        phi = np.exp(-x**2 / (2 * sigma**2))
        return A / (sigma**3 * np.sqrt(2 * np.pi)) * (x**2 / sigma**2 - 1) * phi

    def dr0dsigma(self, r):
        """Return the mixed derivative ``d2U/dr0 dsigma``."""
        r = np.asarray(r, dtype=float)
        A, r0, sigma = self._params
        x = r - r0
        phi = np.exp(-x**2 / (2 * sigma**2))
        return A * x / np.sqrt(2 * np.pi) * (x**2 - 3 * sigma**2) / sigma**6 * phi

    def dsigma_2(self, r):
        """Return ``d2U/dsigma^2`` evaluated at ``r``."""
        r = np.asarray(r, dtype=float)
        A, r0, sigma = self._params
        x = r - r0
        phi = np.exp(-x**2 / (2 * sigma**2))
        return A / np.sqrt(2 * np.pi) * (x**4 - 5 * x**2 * sigma**2 + 2 * sigma**4) / sigma**7 * phi
