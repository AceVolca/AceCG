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
        self.typ1 = typ1
        self.typ2 = typ2
        self.cutoff = cutoff

        # Optimization parameters
        self._params = np.array([A, B, C, D])
        self._param_names = ["A", "B", "C", "D"]

        # First derivatives
        self._dparam_names = ["dA", "dB", "dC", "dD"]

        # Second derivatives (Hessian elements)
        self._d2param_names = [
            ["dA_2", "dAdB", "dAdC", "dAdD"],
            ["dAdB", "dB_2", "dBdC", "dBdD"],
            ["dAdC", "dBdC", "dC_2", "dCdD"],
            ["dAdD", "dBdD", "dCdD", "dD_2"],
        ]

    # ----------------------------------------------------------------------
    # Core potential functions
    # ----------------------------------------------------------------------

    def value(self, r):
        """
        Potential energy U(r) = -[A e^{-Br^2} + C e^{-Dr^2}].
        """
        A, B, C, D = self._params
        r2 = r * r
        return -(A * np.exp(-B * r2) + C * np.exp(-D * r2))

    def force(self, r):
        """
        Force magnitude F(r) = -dU/dr.
        """
        A, B, C, D = self._params
        r2 = r * r

        # d/dr exp(-B r^2) = -2 B r exp(-B r^2)
        term1 = A * (-2 * B * r) * np.exp(-B * r2)
        term2 = C * (-2 * D * r) * np.exp(-D * r2)

        dUdr = -(term1 + term2)  # derivative of full potential
        return -dUdr              # force = -dU/dr

    # ----------------------------------------------------------------------
    # First derivatives w.r.t parameters
    # ----------------------------------------------------------------------

    def dA(self, r):
        _, B, _, _ = self._params
        return -np.exp(-B * r * r)

    def dB(self, r):
        A, B, _, _ = self._params
        r2 = r * r
        return A * r2 * np.exp(-B * r2)

    def dC(self, r):
        _, _, _, D = self._params
        return -np.exp(-D * r * r)

    def dD(self, r):
        _, _, C, D = self._params
        r2 = r * r
        return C * r2 * np.exp(-D * r2)

    # ----------------------------------------------------------------------
    # Second derivatives (mixed and diagonal)
    # ----------------------------------------------------------------------

    def dA_2(self, r):
        return np.zeros_like(r)

    def dAdB(self, r):
        _, B, _, _ = self._params
        r2 = r * r
        return r2 * np.exp(-B * r2)

    def dAdC(self, r):
        return np.zeros_like(r)

    def dAdD(self, r):
        return np.zeros_like(r)

    def dB_2(self, r):
        A, B, _, _ = self._params
        r2 = r * r
        return A * r2**2 * np.exp(-B * r2)

    def dBdC(self, r):
        return np.zeros_like(r)

    def dBdD(self, r):
        return np.zeros_like(r)

    def dC_2(self, r):
        return np.zeros_like(r)

    def dCdD(self, r):
        _, _, C, D = self._params
        r2 = r * r
        return r2 * np.exp(-D * r2)

    def dD_2(self, r):
        _, _, C, D = self._params
        r2 = r * r
        return C * r2**2 * np.exp(-D * r2)
