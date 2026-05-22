"""AceCG potentials harmonic implementation."""

import numpy as np
from .base import BasePotential


class HarmonicPotential(BasePotential):
    """Harmonic potential U(r) = k * (r - r0)^2 * scale.

    The constant ``scale`` absorbs unit-system differences so that *k*
    is always stored in the LAMMPS-native unit for the interaction:

    * **bonds**  — scale = 1.0, k in energy/Å², r in Å.
    * **angles** — scale = (π/180)², k in energy/rad², r and r0 in
      degrees.  The factor converts the degree-based displacement
      into radian-based energy.

    Because *k* stays in LAMMPS-native magnitude, the AdamW optimizer
    sees numerically reasonable values (∼2 for angles, ∼2.5 for bonds)
    rather than the ∼7e-4 values the old energy/deg² convention
    produced.
    """

    def __init__(self, typ1, typ2, k, r0, cutoff=None, *, typ3=None, scale=1.0):
        """Initialize a harmonic bonded potential.

        Parameters
        ----------
        typ1, typ2 : int or str
            Type labels for the first two atoms in the interaction.
        k : float
            Harmonic force constant stored in the forcefield's native units.
        r0 : float
            Equilibrium coordinate.
        cutoff : float, optional
            Optional cutoff stored for compatibility with pair-style tables.
        typ3 : int or str, optional
            Third type label for angle interactions.
        scale : float, default=1.0
            Multiplicative factor applied to the squared displacement. Angle
            terms use this to convert degree displacements into radians.
        """
        super().__init__(bonded=True)
        self.typ1 = typ1
        self.typ2 = typ2
        self.typ3 = typ3
        self.cutoff = cutoff
        self.scale = float(scale)
        self._params = np.array([float(k), float(r0)])
        self._params_to_scale = [0]
        self._param_names = ["k", "r0"]
        self._dparam_names = ["dk", "dr0"]
        self._d2param_names = [
            ["dk_2", "dkdr0"],
            ["dkdr0", "dr0_2"],
        ]
        self._df_dparam_names = ["dkdr", "dr0dr"]

    # ── Potential and force ───────────────────────────────────────────────

    # U(r) = k * scale * (r - r0)^2
    # F(r) = -dU/dr = -2 * k * scale * (r - r0)

    def value(self, r):
        """Evaluate ``U(r) = k * scale * (r - r0)^2``."""
        k, r0 = self._params
        return k * self.scale * (np.asarray(r) - r0) ** 2

    def is_param_linear(self) -> np.ndarray:
        """Return which harmonic parameters enter the energy linearly."""
        return np.array([True, False], dtype=bool)

    def force(self, r):
        """Evaluate the scalar force ``-dU/dr``."""
        k, r0 = self._params
        return -2.0 * k * self.scale * (np.asarray(r) - r0)

    # ── Energy derivatives (for energy_grad in REM/CDREM) ───────────────────────────────────────────────

    def dk(self, r):
        """dU/dk = scale * (r - r0)^2"""
        _, r0 = self._params
        return self.scale * (np.asarray(r) - r0) ** 2

    def dr0(self, r):
        """dU/dr0 = -2 * k * scale * (r - r0)"""
        k, r0 = self._params
        return -2.0 * k * self.scale * (np.asarray(r) - r0)

    def dk_2(self, r):
        """d2U/dk2 = 0"""
        return np.zeros_like(np.asarray(r))

    def dkdr0(self, r):
        """d2U/dkdr0 = -2 * scale * (r - r0)"""
        _, r0 = self._params
        return -2.0 * self.scale * (np.asarray(r) - r0)

    def dr0_2(self, r):
        """d2U/dr02 = 2 * k * scale"""
        k, _ = self._params
        return np.full_like(np.asarray(r, dtype=float), 2.0 * k * self.scale)

    # ── Force derivatives (for iterative FM) ───────────────────────────────────────────────

    def dkdr(self, r):
        """dF/dk = -2 * scale * (r - r0)"""
        _, r0 = self._params
        return -2.0 * self.scale * (np.asarray(r) - r0)

    def dr0dr(self, r):
        """dF/dr0 = 2 * k * scale"""
        k, _ = self._params
        return np.full_like(np.asarray(r, dtype=float), 2.0 * k * self.scale)

    # ── Parameter bounds ───────────────────────────────────────────────

    def param_bounds(self):
        """Return (lower_bounds, upper_bounds) arrays for [k, r0].

        k must be non-negative (physical constraint).
        """
        lb = np.array([0.0, -np.inf])
        ub = np.array([np.inf, np.inf])
        return lb, ub

    # ── Force basis (for FM/CDFM design matrix) ───────────────────────────────────────────────

    def basis_values(self, r):
        """Force Jacobian [dF/dk, dF/dr0] where F = -dU/dr.

        F(r) = -2*k*scale*(r - r0), so
        dF/dk  = -2*scale*(r - r0),
        dF/dr0 =  2*k*scale.
        """
        r = np.asarray(r, dtype=float).ravel()
        k, r0 = self._params
        s = self.scale
        out = np.empty((r.size, 2), dtype=float)
        out[:, 0] = -2.0 * s * (r - r0)
        out[:, 1] = 2.0 * k * s
        return out

    def basis_integrals(self, r):
        """Integrated force basis I_i(r) = integral from r0 to r of B_i(xi) dxi.

        I_k(r)  = -scale*(r - r0)^2,
        I_r0(r) =  2*k*scale*(r - r0).
        """
        r = np.asarray(r, dtype=float).ravel()
        k, r0 = self._params
        s = self.scale
        delta = r - r0
        out = np.empty((r.size, 2), dtype=float)
        out[:, 0] = -s * (delta ** 2)
        out[:, 1] = 2.0 * k * s * delta
        return out
