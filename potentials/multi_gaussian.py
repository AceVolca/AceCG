# AceCG/potentials/multi_gaussian.py
import re
import numpy as np
from typing import Optional, Tuple
from .base import BasePotential

SQRT2PI = np.sqrt(2.0 * np.pi)
_RE_FIRST = re.compile(r'^(dA|dr0|dsigma)_(\d+)$')
_RE_DA2   = re.compile(r'^dA_(\d+)_2$')
_RE_SECOND = re.compile(
    r'^(?:'
    r'dA_(?P<k1>\d+)dr0_(?P=k1)|'
    r'dA_(?P<k2>\d+)dsigma_(?P=k2)|'
    r'dr0_(?P<k3>\d+)_2|'
    r'dr0_(?P<k4>\d+)dsigma_(?P=k4)|'
    r'dsigma_(?P<k5>\d+)_2'
    r')$'
)

class MultiGaussianPotential(BasePotential):
    """
    Sum of n_gauss normalized Gaussian components:
      V(r) = Σ_k A_k/(σ_k*sqrt(2π)) * exp( - (r - r0_k)^2 / (2 σ_k^2) )

    Params per k: (A_k, r0_k, sigma_k).
    First derivatives: dA_k, dr0_k, dsigma_k.
    Second derivatives (intra-component only): dA_k_2, dA_kdr0_k, dA_kdsigma_k,
                                              dr0_k_2, dr0_kdsigma_k, dsigma_k_2.
    Cross-component second derivatives are identically zero (exposed via names table).
    """

    def __init__(
        self,
        typ1: str,
        typ2: str,
        n_gauss: int,
        cutoff: float = np.inf,
        init_params: Optional[np.ndarray] = None,
        *,
        sigma_floor: float = 1e-8
    ):
        super().__init__()
        assert n_gauss >= 1
        self.typ1 = typ1
        self.typ2 = typ2
        self.n_gauss = int(n_gauss)
        self.cutoff = float(cutoff)
        self._sigma_floor = float(sigma_floor)

        if init_params is None:
            params = np.empty(3 * self.n_gauss, dtype=float)
            params[0::3] = 0.0  # A
            params[1::3] = 0.0  # r0
            params[2::3] = 1.0  # sigma
            self._params = params
        else:
            init_params = np.asarray(init_params, dtype=float)
            if init_params.size != 3 * self.n_gauss:
                raise ValueError(f"init_params must have length {3*self.n_gauss}")
            self._params = init_params.copy()
        self._params_to_scale = [i for i in range(len(self._params)) if i%3 == 0]

        # Names
        self._param_names = []
        self._dparam_names = []
        for k in range(self.n_gauss):
            self._param_names.extend([f"A_{k}", f"r0_{k}", f"sigma_{k}"])
            self._dparam_names.extend([f"dA_{k}", f"dr0_{k}", f"dsigma_{k}"])

        # 2nd-deriv name matrix (3n x 3n); cross terms -> "zero"
        d2 = []
        for i in range(self.n_gauss):
            for a in range(3):
                row = []
                for j in range(self.n_gauss):
                    for b in range(3):
                        if i == j:
                            if   (a, b) == (0, 0): row.append(f"dA_{i}_2")
                            elif (a, b) in [(0, 1), (1, 0)]: row.append(f"dA_{i}dr0_{i}")
                            elif (a, b) in [(0, 2), (2, 0)]: row.append(f"dA_{i}dsigma_{i}")
                            elif (a, b) == (1, 1): row.append(f"dr0_{i}_2")
                            elif (a, b) in [(1, 2), (2, 1)]: row.append(f"dr0_{i}dsigma_{i}")
                            elif (a, b) == (2, 2): row.append(f"dsigma_{i}_2")
                        else:
                            row.append("zero")
                d2.append(row)
        self._d2param_names = d2

        self._validate_sigmas()

    # -------- params as views --------
    @property
    def A(self) -> np.ndarray:
        return self._params[0::3]
    @property
    def r0(self) -> np.ndarray:
        return self._params[1::3]
    @property
    def sigma(self) -> np.ndarray:
        return self._params[2::3]


    def _validate_sigmas(self):
        # enforce a small positive floor for stability
        np.maximum(self._params[2::3], self._sigma_floor, out=self._params[2::3])

    # -------- core API (vectorized) --------
    def _xr_phi(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given r (...,), return:
          x  with shape (..., n_gauss) = r[...,None] - r0[None,...]
          phi with shape (..., n_gauss) = exp(-(x^2)/(2 σ^2))
        """
        r = np.asarray(r, dtype=float)
        x = r[..., None] - self.r0[None, ...]
        inv_sigma = 1.0 / self.sigma
        phi = np.exp(-0.5 * (x * inv_sigma)**2)
        return x, phi

    def value(self, r: np.ndarray) -> np.ndarray:
        x, phi = self._xr_phi(r)
        # Σ A/(σ√2π) * φ
        out = (self.A * (1.0 / (self.sigma * SQRT2PI)) * phi).sum(axis=-1)
        if np.isfinite(self.cutoff):
            out = np.where(r <= self.cutoff, out, 0.0)
        return out

    def force(self, r: np.ndarray) -> np.ndarray:
        # F = -dV/dr = Σ A * x * φ / (σ^3 √2π)
        x, phi = self._xr_phi(r)
        s = self.sigma
        F = (self.A * x * phi / (s**3 * SQRT2PI)).sum(axis=-1)
        if np.isfinite(self.cutoff):
            F = np.where(r <= self.cutoff, F, 0.0)
        return F

    # -------- zeros for cross-terms --------
    def zero(self, r: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(r, dtype=float))

    # -------- dynamic derivative dispatch (simplified) --------
    def __getattr__(self, name: str):
        if name == "zero":
            return self.zero

        # First derivatives
        m = _RE_FIRST.match(name)
        if m:
            kind, k = m.group(1), int(m.group(2))
            if not (0 <= k < self.n_gauss):
                raise AttributeError(f"{name}: component index out of range")

            if kind == "dA":
                def dA_fn(r, k=k):
                    r = np.asarray(r, dtype=float)
                    x = r[..., None] - self.r0[None, :]
                    s = self.sigma
                    phi = np.exp(-0.5 * (x[..., k] / s[k])**2)
                    return phi / (s[k] * SQRT2PI)
                return dA_fn

            if kind == "dr0":
                def dr0_fn(r, k=k):
                    r = np.asarray(r, dtype=float)
                    x = r - self.r0[k]
                    s = self.sigma[k]
                    return self.A[k] * x * np.exp(-0.5 * (x / s)**2) / (s**3 * SQRT2PI)
                return dr0_fn

            if kind == "dsigma":
                def dsigma_fn(r, k=k):
                    r = np.asarray(r, dtype=float)
                    x = r - self.r0[k]
                    s = self.sigma[k]
                    phi = np.exp(-0.5 * (x / s)**2)
                    return self.A[k] * phi * (x*x - s*s) / (s**4 * SQRT2PI)
                return dsigma_fn

        # dA_k_2 -> 0
        if _RE_DA2.match(name):
            def zero_fn(r):
                r = np.asarray(r, dtype=float)
                return np.zeros_like(r, dtype=float)
            return zero_fn

        # Mixed/intra second derivatives
        m = _RE_SECOND.match(name)
        if m:
            k = int(next(g for g in (m.group('k1'), m.group('k2'), m.group('k3'),
                                     m.group('k4'), m.group('k5')) if g))
            if not (0 <= k < self.n_gauss):
                raise AttributeError(f"{name}: component index out of range")

            def dA_dr0(r, k=k):
                r = np.asarray(r, dtype=float)
                x = r - self.r0[k]
                s = self.sigma[k]
                phi = np.exp(-0.5 * (x / s)**2)
                return x * phi / (s**3 * SQRT2PI)

            def dA_dsigma(r, k=k):
                r = np.asarray(r, dtype=float)
                x = r - self.r0[k]
                s = self.sigma[k]
                phi = np.exp(-0.5 * (x / s)**2)
                return (x*x - s*s) * phi / (s**4 * SQRT2PI)

            def dr0_2(r, k=k):
                r = np.asarray(r, dtype=float)
                x = r - self.r0[k]
                s = self.sigma[k]
                phi = np.exp(-0.5 * (x / s)**2)
                return self.A[k] * (x*x / (s*s) - 1.0) * phi / (s**3 * SQRT2PI)

            def dr0_dsigma(r, k=k):
                r = np.asarray(r, dtype=float)
                x = r - self.r0[k]
                s = self.sigma[k]
                phi = np.exp(-0.5 * (x / s)**2)
                return self.A[k] * x * (x*x - 3.0*s*s) * phi / (s**6 * SQRT2PI)

            def dsigma_2(r, k=k):
                r = np.asarray(r, dtype=float)
                x = r - self.r0[k]
                s = self.sigma[k]
                phi = np.exp(-0.5 * (x / s)**2)
                return self.A[k] * (x**4 - 5.0*x*x*s*s + 2.0*s**4) * phi / (s**7 * SQRT2PI)

            if name.startswith("dA_") and "dr0" in name:   return dA_dr0
            if name.startswith("dA_") and "dsigma" in name:return dA_dsigma
            if name.startswith("dr0_") and name.endswith("_2"): return dr0_2
            if name.startswith("dr0_") and "dsigma" in name:    return dr0_dsigma
            if name.startswith("dsigma_") and name.endswith("_2"): return dsigma_2

        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")