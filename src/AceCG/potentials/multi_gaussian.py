# AceCG/potentials/multi_gaussian.py
import re
import numpy as np
from .base import BasePotential
from typing import Optional

SQRT2PI = np.sqrt(2.0 * np.pi)

class MultiGaussianPotential(BasePotential):
    """
    Sum of n_gauss Gaussian components:
        V(r) = sum_k A_k/(sigma_k*sqrt(2*pi)) * exp( -(r - r0_k)^2 / (2*sigma_k^2) )

    Parameters per component k: (A_k, r0_k, sigma_k).
    First derivatives available as: dA_k, dr0_k, dsigma_k
    Second derivatives (intra-component) as: dA_k_2, dA_kdr0_k, dA_kdsigma_k,
                                             dr0_k_2, dr0_kdsigma_k, dsigma_k_2
    Cross-component second derivatives are exactly zero and exposed via a "zero" method.
    """

    def __init__(self, typ1: str, typ2: str, n_gauss: int, cutoff: float = np.inf,
                 init_params: Optional[np.ndarray] = None):
        super().__init__()
        assert n_gauss >= 1
        self.typ1 = typ1
        self.typ2 = typ2
        self.n_gauss = int(n_gauss)
        self.cutoff = float(cutoff)

        # parameter layout: [A_0, r0_0, sigma_0, A_1, r0_1, sigma_1, ...]
        if init_params is None:
            # default: A=0, r0=0, sigma=1
            params = []
            for _ in range(self.n_gauss):
                params += [0.0, 0.0, 1.0]
            self._params = np.array(params, dtype=float)
        else:
            init_params = np.asarray(init_params, dtype=float)
            assert init_params.size == 3 * self.n_gauss, \
                f"init_params must have length {3*self.n_gauss}"
            self._params = init_params.copy()

        # Names for parameters and derivatives
        self._param_names = []
        self._dparam_names = []
        d2 = []
        for k in range(self.n_gauss):
            self._param_names.extend([f"A_{k}", f"r0_{k}", f"sigma_{k}"])
            self._dparam_names.extend([f"dA_{k}", f"dr0_{k}", f"dsigma_{k}"])

        # Build d2 names matrix (3n x 3n)
        # Use "zero" for cross-component second derivatives (i != j)
        for i in range(self.n_gauss):
            row_i = []
            for a in range(3):  # (A, r0, sigma) index in component i
                names_row = []
                for j in range(self.n_gauss):
                    for b in range(3):
                        if i == j:
                            # Intra-component names (mirror GaussianPotential)
                            if   (a, b) == (0, 0): names_row.append(f"dA_{i}_2")
                            elif (a, b) == (0, 1) or (a, b) == (1, 0): names_row.append(f"dA_{i}dr0_{i}")
                            elif (a, b) == (0, 2) or (a, b) == (2, 0): names_row.append(f"dA_{i}dsigma_{i}")
                            elif (a, b) == (1, 1): names_row.append(f"dr0_{i}_2")
                            elif (a, b) == (1, 2) or (a, b) == (2, 1): names_row.append(f"dr0_{i}dsigma_{i}")
                            elif (a, b) == (2, 2): names_row.append(f"dsigma_{i}_2")
                            else:
                                names_row.append("zero")
                        else:
                            names_row.append("zero")
                row_i.append(names_row)
            # flatten three subrows for component i
            d2.extend(row_i)
        # Collapse nested structure to a 2D list of strings
        self._d2param_names = []
        for row in d2:
            self._d2param_names.append(row)

    # ---------- helpers ----------
    def _idx(self, k: int):
        """Return slice indices for component k (A,r0,sigma)."""
        base = 3 * k
        return base, base + 1, base + 2

    def _params_of(self, k: int):
        a, b, c = self._idx(k)
        A = self._params[a]
        r0 = self._params[b]
        sigma = self._params[c]
        return A, r0, sigma

    # ---------- core API ----------
    def value(self, r: np.ndarray) -> np.ndarray:
        """Sum of all Gaussian components at distances r."""
        r = np.asarray(r, dtype=float)
        out = np.zeros_like(r, dtype=float)
        for k in range(self.n_gauss):
            A, r0, sigma = self._params_of(k)
            x = r - r0
            out += A / (sigma * SQRT2PI) * np.exp(-x * x / (2.0 * sigma * sigma))
        return out
    
    def force(self, r: np.ndarray) -> np.ndarray:
        """
        Compute radial force F(r) = -dV/dr for a batch of distances r.
        Shape matches the input r. If cutoff is finite, force is set to 0 for r > cutoff.
        """
        r = np.asarray(r, dtype=float)
        F = np.zeros_like(r, dtype=float)
        for k in range(self.n_gauss):
            A, r0, sigma = self._params_of(k)
            x = r - r0
            phi = np.exp(-x * x / (2.0 * sigma * sigma))
            F += A * x * phi / (sigma**3 * SQRT2PI)

        # zero out beyond cutoff (optional but handy for tables)
        if np.isfinite(self.cutoff):
            F = np.where(r <= self.cutoff, F, 0.0)
        return F


    # ---------- zero for cross-terms ----------
    def zero(self, r: np.ndarray) -> np.ndarray:
        return np.zeros_like(r, dtype=float)

    # ---------- dynamic derivative dispatch ----------
    def __getattr__(self, name: str):
        """
        Dynamically provide derivative callables such as:
          dA_k, dr0_k, dsigma_k,
          dA_k_2, dA_kdr0_k, dA_kdsigma_k, dr0_k_2, dr0_kdsigma_k, dsigma_k_2
        """
        # zero
        if name == "zero":
            return self.zero

        # First derivatives: dA_k, dr0_k, dsigma_k
        m = re.match(r'^(dA|dr0|dsigma)_(\d+)$', name)
        if m:
            kind, k = m.group(1), int(m.group(2))

            def dA_fn(r, k=k):
                _, r0, sigma = self._params_of(k)
                x = np.asarray(r, dtype=float) - r0
                return np.exp(-x * x / (2.0 * sigma * sigma)) / (sigma * SQRT2PI)

            def dr0_fn(r, k=k):
                A, r0, sigma = self._params_of(k)
                x = np.asarray(r, dtype=float) - r0
                return A * x * np.exp(-x * x / (2.0 * sigma * sigma)) / (sigma**3 * SQRT2PI)

            def dsigma_fn(r, k=k):
                A, r0, sigma = self._params_of(k)
                x = np.asarray(r, dtype=float) - r0
                phi = np.exp(-x * x / (2.0 * sigma * sigma))
                return A * phi * (x * x - sigma * sigma) / (sigma**4 * SQRT2PI)

            return {"dA": dA_fn, "dr0": dr0_fn, "dsigma": dsigma_fn}[kind]

        # Second derivatives (intra-component)
        # dA_k_2
        m = re.match(r'^dA_(\d+)_2$', name)
        if m:
            def fn(r):
                return np.zeros_like(r, dtype=float)
            return fn

        # dA_kdr0_k, dA_kdsigma_k, dr0_k_2, dr0_kdsigma_k, dsigma_k_2
        m = re.match(
            r'^(?:'
            r'dA_(?P<k1>\d+)dr0_(?P=k1)|'
            r'dA_(?P<k2>\d+)dsigma_(?P=k2)|'
            r'dr0_(?P<k3>\d+)_2|'
            r'dr0_(?P<k4>\d+)dsigma_(?P=k4)|'
            r'dsigma_(?P<k5>\d+)_2'
            r')$',
            name
        )
        if m:
            # Identify which pattern matched and extract k accordingly
            def make(name):
                # extract k
                k_str = next(g for g in (m.group('k1'), m.group('k2'), m.group('k3'),
                                     m.group('k4'), m.group('k5')) if g)
                k = int(k_str)

                def dA_dr0(r):
                    _, r0, sigma = self._params_of(k)
                    x = np.asarray(r, dtype=float) - r0
                    phi = np.exp(-x * x / (2.0 * sigma * sigma))
                    return x * phi / (sigma**3 * SQRT2PI)

                def dA_dsigma(r):
                    _, r0, sigma = self._params_of(k)
                    x = np.asarray(r, dtype=float) - r0
                    phi = np.exp(-x * x / (2.0 * sigma * sigma))
                    return (x * x - sigma * sigma) * phi / (sigma**4 * SQRT2PI)

                def dr0_2(r):
                    A, r0, sigma = self._params_of(k)
                    x = np.asarray(r, dtype=float) - r0
                    phi = np.exp(-x * x / (2.0 * sigma * sigma))
                    return A * (x * x / (sigma * sigma) - 1.0) * phi / (sigma**3 * SQRT2PI)

                def dr0_dsigma(r):
                    A, r0, sigma = self._params_of(k)
                    x = np.asarray(r, dtype=float) - r0
                    phi = np.exp(-x * x / (2.0 * sigma * sigma))
                    return A * x * (x * x - 3.0 * sigma * sigma) * phi / (sigma**6 * SQRT2PI)

                def dsigma_2(r):
                    A, r0, sigma = self._params_of(k)
                    x = np.asarray(r, dtype=float) - r0
                    phi = np.exp(-x * x / (2.0 * sigma * sigma))
                    return A * (x**4 - 5.0 * x * x * sigma * sigma + 2.0 * sigma**4) * phi / (sigma**7 * SQRT2PI)

                if name.startswith("dA_") and "dr0" in name:   return dA_dr0
                if name.startswith("dA_") and "dsigma" in name:return dA_dsigma
                if name.startswith("dr0_") and name.endswith("_2"): return dr0_2
                if name.startswith("dr0_") and "dsigma" in name:    return dr0_dsigma
                if name.startswith("dsigma_") and name.endswith("_2"): return dsigma_2
                return self.zero

            return make(name)

        # Cross-component second derivatives map to zero via names table,
        # but if someone tried to call an unexpected name, fall back to AttributeError
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
