# Adding a New Potential to AceCG

AceCG provides a modular analytic framework for coarse-grained potentials.
All potentials support:

- Parameterized energy functions  
- Forces (−dU/dr)  
- First derivatives ∂U/∂θ  
- Second derivatives ∂²U/∂θ² (for Hessian-based optimizers)  
- Trainer/optimizer integration through a unified interface  

This document explains how to add a new potential class in a way that is fully compatible with the existing Gaussian, Lennard-Jones, and soft-core potentials.

---

## 1. File structure

All analytic potentials live inside:

AceCG/potentials/

To add a new potential, simply create a file such as:

AceCG/potentials/srlrgaussian.py

---

## 2. Required interface for potentials

A valid AceCG potential must implement:

### ✔ Parameter vector
```python
self._params = np.array([...])
```

### ✔ Parameter names
```python
self._param_names = ["A", "B", "C", ...]
```

### ✔ First-derivative method names
```python
self._dparam_names = ["dA", "dB", "dC", ...]
```

### ✔ Second-derivative (Hessian) method names

A symmetric matrix:
```python
self._d2param_names = [
	["dA_2", "dAdB", "dAdC"],
	["dAdB", "dB_2", "dBdC"],
	["dAdC", "dBdC", "dC_2"],
]
```

### ✔ Required methods

- `value(r)` → returns U(r)  
- `force(r)` → returns −dU/dr  
- One first-derivative function per parameter:  
  `dA(r), dB(r), dC(r), ...`  
- One second-derivative function per Hessian entry:  
  `dAdB(r), dB_2(r), dCdD(r), ...`

AceCG trainers automatically locate and call these methods using the names listed above.

---

## 3. Example: SRLRGaussianPotential

Potential form:

\[
U(r) = -[A e^{-Br^2} + C e^{-Dr^2}],\quad r < r_{\text{cut}}
\]

Example implementation:

```python
import numpy as np
from .base import BasePotential

class SRLRGaussianPotential(BasePotential):
    def __init__(self, typ1, typ2, A, B, C, D, cutoff):
        self.typ1 = typ1
        self.typ2 = typ2
        self.cutoff = cutoff

        self._params = np.array([A, B, C, D])
        self._param_names = ["A", "B", "C", "D"]
        self._dparam_names = ["dA", "dB", "dC", "dD"]

        self._d2param_names = [
            ["dA_2", "dAdB", "dAdC", "dAdD"],
            ["dAdB", "dB_2", "dBdC", "dBdD"],
            ["dAdC", "dBdC", "dC_2", "dCdD"],
            ["dAdD", "dBdD", "dCdD", "dD_2"],
        ]

    def value(self, r):
        A, B, C, D = self._params
        r2 = r * r
        return -(A*np.exp(-B*r2) + C*np.exp(-D*r2))

    def force(self, r):
        A, B, C, D = self._params
        r2 = r * r
        dUdr = -(A*(-2*B*r)*np.exp(-B*r2) +
                 C*(-2*D*r)*np.exp(-D*r2))
        return -dUdr

    def dA(self, r):
        return -np.exp(-self._params[1] * r*r)

    def dB(self, r):
        A, B, _, _ = self._params
        r2 = r*r
        return A * r2 * np.exp(-B*r2)

    def dC(self, r):
        return -np.exp(-self._params[3] * r*r)

    def dD(self, r):
        _, _, C, D = self._params
        r2 = r*r
        return C * r2 * np.exp(-D*r2)

    def dA_2(self, r): return np.zeros_like(r)
    def dAdB(self, r): return r*r * np.exp(-(self._params[1])*r*r)
    def dAdC(self, r): return np.zeros_like(r)
    def dAdD(self, r): return np.zeros_like(r)

    def dB_2(self, r):
        A, B, _, _ = self._params
        r2 = r*r
        return A * r2*r2 * np.exp(-B*r2)

    def dBdC(self, r): return np.zeros_like(r)
    def dBdD(self, r): return np.zeros_like(r)

    def dC_2(self, r): return np.zeros_like(r)

    def dCdD(self, r):
        _, _, C, D = self._params
        r2 = r*r
        return r2 * np.exp(-D*r2)

    def dD_2(self, r):
        _, _, C, D = self._params
        r2 = r*r
        return C * r2*r2 * np.exp(-D*r2)
```

---

## 4. Registering the new potential

Add the import to:

AceCG/potentials/__init__.py

```python
from .srlrgaussian import SRLRGaussianPotential

POTENTIAL_REGISTRY = {
    "gaussian": GaussianPotential,
    "srlr_gaussian": SRLRGaussianPotential,
}
```

AceCG will read the potentials from POTENTIAL_REGISTRY

Also add the import to for direct usage of the potential:

AceCG/__init__.py

```python
from .potentials.srlrgaussian import SRLRGaussianPotential

__all__ = [
    "SRLRGaussianPotential",
]
```