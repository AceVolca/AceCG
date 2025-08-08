# AceCG/potentials/gaussian.py
import numpy as np
from .base import BasePotential

class GaussianPotential(BasePotential):
    def __init__(self, typ1, typ2, A, r0, sigma, cutoff):
        self.typ1  = typ1
        self.typ2  = typ2
        self.cutoff = cutoff
        self._params = np.array([A, r0, sigma])
        self.param_names = ["A", "r0", "sigma"]
        self._dparams = ["dA", "dr0", "dsigma"]
        self._d2params = [
            ["dA_2", "dAdr0", "dAdsigma"],
            ["dAdr0", "dr0_2", "dr0dsigma"],
            ["dAdsigma", "dr0dsigma", "dsigma_2"]
        ]

    def value(self, r):
        A, r0, sigma = self._params
        x = r - r0
        return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))

    def dA(self, r):
        _, r0, sigma = self._params
        x = r - r0
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))

    def dr0(self, r):
        A, r0, sigma = self._params
        x = r - r0
        return A * x / (sigma**3 * np.sqrt(2 * np.pi)) * np.exp(-x**2 / (2 * sigma**2))

    def dsigma(self, r):
        A, r0, sigma = self._params
        x = r - r0
        phi = np.exp(-x**2 / (2 * sigma**2))
        return A * phi / np.sqrt(2 * np.pi) * (x**2 - sigma**2) / sigma**4

    def dA_2(self, r):
        return np.zeros_like(r)

    def dAdr0(self, r):
        _, r0, sigma = self._params
        x = r - r0
        phi = np.exp(-x**2 / (2 * sigma**2))
        return x / (sigma**3 * np.sqrt(2 * np.pi)) * phi

    def dAdsigma(self, r):
        _, r0, sigma = self._params
        x = r - r0
        phi = np.exp(-x**2 / (2 * sigma**2))
        return (x**2 - sigma**2) / (sigma**4 * np.sqrt(2 * np.pi)) * phi

    def dr0_2(self, r):
        A, r0, sigma = self._params
        x = r - r0
        phi = np.exp(-x**2 / (2 * sigma**2))
        return A / (sigma**3 * np.sqrt(2 * np.pi)) * (x**2 / sigma**2 - 1) * phi

    def dr0dsigma(self, r):
        A, r0, sigma = self._params
        x = r - r0
        phi = np.exp(-x**2 / (2 * sigma**2))
        return A * x / np.sqrt(2 * np.pi) * (x**2 - 3 * sigma**2) / sigma**6 * phi

    def dsigma_2(self, r):
        A, r0, sigma = self._params
        x = r - r0
        phi = np.exp(-x**2 / (2 * sigma**2))
        return A / np.sqrt(2 * np.pi) * (x**4 - 5 * x**2 * sigma**2 + 2 * sigma**4) / sigma**7 * phi

    def dparam_names(self):
        return self._dparams

    def d2param_names(self):
        return self._d2params

    def n_params(self):
        return len(self._params)

    def params(self):
        return self._params

    def set_params(self, new_params):
        assert len(new_params) == len(self._params)
        self._params = new_params.copy()
