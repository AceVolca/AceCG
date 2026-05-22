"""AceCG potentials srlrgaussian implementation."""

import numpy as np
from .base import BasePotential

class SRLRGaussianPotential(BasePotential):
    """
    Short-range + long-range Gaussian binding potential:

        U(r) = -[ A exp(-B r^2) + C exp(-D r^2) ],   r < cutoff

    Parameters
    ----------
    typ1, typ2 : int
        Particle types.
    A, B, C, D : float
        Strength and decay parameters.
    cutoff : float
        Cutoff distance.
    """

    def __init__(self, typ1, typ2, A, B, C, D, cutoff):
        """Initialize a short-range/long-range Gaussian potential.

        Parameters
        ----------
        typ1, typ2 : int or str
            Pair type labels.
        A, B : float
            Strength and decay parameter for the first Gaussian term.
        C, D : float
            Strength and decay parameter for the second Gaussian term.
        cutoff : float
            Pair cutoff distance.
        """
        super().__init__()
        self.typ1 = typ1
        self.typ2 = typ2
        self.cutoff = cutoff

        # Optimization parameters
        self._params = np.array([A, B, C, D])
        self._param_names = ["A", "B", "C", "D"]

        self._params_to_scale = [0, 2]

        # First derivatives
        self._dparam_names = ["dA", "dB", "dC", "dD"]

        # Second derivatives (Hessian elements)
        self._d2param_names = [
            ["dA_2", "dAdB", "dAdC", "dAdD"],
            ["dAdB", "dB_2", "dBdC", "dBdD"],
            ["dAdC", "dBdC", "dC_2", "dCdD"],
            ["dAdD", "dBdD", "dCdD", "dD_2"],
        ]

    def is_param_linear(self) -> np.ndarray:
        """Return which SR/LR Gaussian parameters enter the energy linearly."""
        return np.array([True, False, True, False], dtype=bool)

    # ── Core potential functions ───────────────────────────────────────────────

    def value(self, r):
        """
        Potential energy U(r) = -[A e^{-Br^2} + C e^{-Dr^2}].
        """
        r = np.asarray(r, dtype=float)
        A, B, C, D = self._params
        r2 = r * r
        return -(A * np.exp(-B * r2) + C * np.exp(-D * r2))

    def force(self, r):
        """
        Force magnitude F(r) = -dU/dr.
        """
        r = np.asarray(r, dtype=float)
        A, B, C, D = self._params
        r2 = r * r

        # d/dr exp(-B r^2) = -2 B r exp(-B r^2)
        term1 = A * (-2 * B * r) * np.exp(-B * r2)
        term2 = C * (-2 * D * r) * np.exp(-D * r2)

        # derivative of full potential
        dUdr = -(term1 + term2)
        # force = -dU/dr
        return -dUdr

    # ── First derivatives w.r.t parameters ───────────────────────────────────────────────

    def dA(self, r):
        """Return ``dU/dA`` evaluated at ``r``."""
        r = np.asarray(r, dtype=float)
        _, B, _, _ = self._params
        return -np.exp(-B * r * r)

    def dB(self, r):
        """Return ``dU/dB`` evaluated at ``r``."""
        r = np.asarray(r, dtype=float)
        A, B, _, _ = self._params
        r2 = r * r
        return A * r2 * np.exp(-B * r2)

    def dC(self, r):
        """Return ``dU/dC`` evaluated at ``r``."""
        r = np.asarray(r, dtype=float)
        _, _, _, D = self._params
        return -np.exp(-D * r * r)

    def dD(self, r):
        """Return ``dU/dD`` evaluated at ``r``."""
        r = np.asarray(r, dtype=float)
        _, _, C, D = self._params
        r2 = r * r
        return C * r2 * np.exp(-D * r2)

    # ── Second derivatives (mixed and diagonal) ───────────────────────────────────────────────

    def dA_2(self, r):
        """Return ``d2U/dA2``, which is zero."""
        return np.zeros_like(np.asarray(r, dtype=float), dtype=float)

    def dAdB(self, r):
        """Return the mixed derivative ``d2U/dA dB``."""
        r = np.asarray(r, dtype=float)
        _, B, _, _ = self._params
        r2 = r * r
        return r2 * np.exp(-B * r2)

    def dAdC(self, r):
        """Return the mixed derivative ``d2U/dA dC``, which is zero."""
        return np.zeros_like(np.asarray(r, dtype=float), dtype=float)

    def dAdD(self, r):
        """Return the mixed derivative ``d2U/dA dD``, which is zero."""
        return np.zeros_like(np.asarray(r, dtype=float), dtype=float)

    def dB_2(self, r):
        """Return ``d2U/dB2`` evaluated at ``r``."""
        r = np.asarray(r, dtype=float)
        A, B, _, _ = self._params
        r2 = r * r
        return A * r2**2 * np.exp(-B * r2)

    def dBdC(self, r):
        """Return the mixed derivative ``d2U/dB dC``, which is zero."""
        return np.zeros_like(np.asarray(r, dtype=float), dtype=float)

    def dBdD(self, r):
        """Return the mixed derivative ``d2U/dB dD``, which is zero."""
        return np.zeros_like(np.asarray(r, dtype=float), dtype=float)

    def dC_2(self, r):
        """Return ``d2U/dC2``, which is zero."""
        return np.zeros_like(np.asarray(r, dtype=float), dtype=float)

    def dCdD(self, r):
        """Return the mixed derivative ``d2U/dC dD``."""
        r = np.asarray(r, dtype=float)
        _, _, C, D = self._params
        r2 = r * r
        return r2 * np.exp(-D * r2)

    def dD_2(self, r):
        """Return ``d2U/dD2`` evaluated at ``r``."""
        r = np.asarray(r, dtype=float)
        _, _, C, D = self._params
        r2 = r * r
        return C * r2**2 * np.exp(-D * r2)
