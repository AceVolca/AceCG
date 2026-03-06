# AceCG/potentials/lennardjones_soft.py
from .base import BasePotential
import numpy as np

class LennardJonesSoftPotential(BasePotential):
    """
    Class representing the soft-core Lennard-Jones potential.

    This implements the functional form used in the LAMMPS lj/cut/soft
    style for the 12-6 Lennard-Jones potential with a soft-core
    regularization:

        E(r) = λ^n * 4 ε * ( 1 / [ α_LJ (1 − λ)^2 + (r/σ)^6 ]^2
                            − 1 / [ α_LJ (1 − λ)^2 + (r/σ)^6 ] )

    The soft-core parameter λ removes the singularity at short distances
    and is typically used for alchemical transformations.
    """

    def __init__(
        self,
        typ1,
        typ2,
        epsilon: float,
        sigma: float,
        lam: float,
        cutoff: float,
        n: int,
        alpha_LJ: float,
    ) -> None:
        """
        Initialize the soft-core Lennard-Jones potential.

        :param epsilon: Depth of the potential well.
        :param sigma: Lennard-Jones sigma parameter.
        :param lam: Soft-core lambda parameter (0 <= lambda <= 1).
        :param cutoff: Cutoff distance (stored for consistency).
        :param n: Exponent on lambda in the soft-core prefactor, global variable.
        :param alpha_LJ: Soft-core parameter alpha_LJ, global variable.
        """
        self.typ1 = typ1
        self.typ2 = typ2
        self.cutoff = cutoff

        # Parameters that are varied during optimization
        self._params = np.array([epsilon, sigma, lam])

        # Names of parameters and their derivatives
        self._param_names = ["epsilon", "sigma", "lambda"]
        self._params_to_scale = [0]
        self._dparam_names = ["depsilon", "dsigma", "dlambda"]
        self._d2param_names = [
            ["depsilon_2", "depsilondsigma", "depsilondlambda"],
            ["depsilondsigma", "dsigma_2", "dsigmadlambda"],
            ["depsilondlambda", "dsigmadlambda", "dlambda_2"],
        ]

        # Fixed soft-core settings
        self.alpha_LJ = alpha_LJ
        self.n = n

    # ------------------------------------------------------------------
    # Core value and force
    # ------------------------------------------------------------------

    def _core_quantities(self, r: np.ndarray):
        """
        Helper to compute shared soft-core quantities for a given distance.

        Returns
        -------
        A, invA, invA2, invA3, invA4, g, gA, gAA, r6_over_sig6
        """
        epsilon, sigma, lam = self._params
        alpha_LJ = self.alpha_LJ

        # (r/sigma)^6
        r6_over_sig6 = (r / sigma) ** 6

        # Soft-core denominator A = α(1-λ)^2 + (r/σ)^6
        A = alpha_LJ * (1.0 - lam) ** 2 + r6_over_sig6

        invA = 1.0 / A
        invA2 = invA ** 2
        invA3 = invA ** 3
        invA4 = invA ** 4

        # g(A) = A^{-2} - A^{-1}
        g = invA2 - invA
        # g'(A)
        gA = -2.0 * invA3 + invA2
        # g''(A)
        gAA = 6.0 * invA4 - 2.0 * invA3

        return A, invA, invA2, invA3, invA4, g, gA, gAA, r6_over_sig6

    def value(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the soft-core Lennard-Jones potential at a distance r.

        :param r: Distance between two particles.
        :return: The potential value at distance r.
        """
        epsilon, sigma, lam = self._params
        _, _, _, _, _, g, _, _, _ = self._core_quantities(r)

        # E(r) = 4 ε λ^n g(A)
        return 4.0 * epsilon * lam**self.n * g

    def force(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the force (negative gradient of the potential) at a distance r.

        :param r: Distance between two particles.
        :return: The force at distance r.
        """
        epsilon, sigma, lam = self._params
        alpha_LJ = self.alpha_LJ

        A, _, _, _, _, _, gA, _, r6_over_sig6 = self._core_quantities(r)

        # A_r = dA/dr = 6 (r/σ)^5 / σ = 6 r6 / r
        A_r = 6.0 * r6_over_sig6 / r

        # dE/dr = 4 ε λ^n g'(A) A_r
        dEdr = 4.0 * epsilon * lam**self.n * gA * A_r

        # Force is negative gradient
        return -dEdr

    # ------------------------------------------------------------------
    # First derivatives with respect to parameters
    # ------------------------------------------------------------------

    def depsilon(self, r: float) -> float:
        """
        Derivative of the potential with respect to epsilon.

        :param r: Distance between two particles.
        :return: dV/depsilon at distance r.
        """
        _, _, lam = self._params
        _, _, _, _, _, g, _, _, _ = self._core_quantities(r)

        return 4.0 * lam**self.n * g

    def dsigma(self, r: float) -> float:
        """
        Derivative of the potential with respect to sigma.

        :param r: Distance between two particles.
        :return: dV/dsigma at distance r.
        """
        epsilon, sigma, lam = self._params
        alpha_LJ = self.alpha_LJ

        A, _, _, _, _, _, gA, _, r6_over_sig6 = self._core_quantities(r)

        # A_sigma = dA/dsigma = -6 (r/σ)^6 / σ
        A_sigma = -6.0 * r6_over_sig6 / sigma

        return 4.0 * epsilon * lam**self.n * gA * A_sigma

    def dlambda(self, r: float) -> float:
        """
        Derivative of the potential with respect to lambda.

        :param r: Distance between two particles.
        :return: dV/dlambda at distance r.
        """
        epsilon, sigma, lam = self._params
        alpha_LJ = self.alpha_LJ

        A, _, _, _, _, g, gA, _, _ = self._core_quantities(r)

        # h(λ) = λ^n
        h = lam**self.n
        h1 = self.n * lam ** (self.n - 1)

        # A_lambda = dA/dlambda = -2 α (1 - λ)
        A_lambda = -2.0 * alpha_LJ * (1.0 - lam)

        # dV/dλ = 4 ε [ h'(λ) g(A) + h(λ) g'(A) A_lambda ]
        return 4.0 * epsilon * (h1 * g + h * gA * A_lambda)

    # ------------------------------------------------------------------
    # Second derivatives with respect to parameters
    # ------------------------------------------------------------------

    def depsilon_2(self, r: float) -> float:
        """
        Second derivative with respect to epsilon (d²V/depsilon²).

        Since the potential is linear in epsilon, this is identically zero.
        """
        return 0.0

    def depsilondsigma(self, r: float) -> float:
        """
        Mixed derivative d²V/(depsilon dsigma).

        :param r: Distance between two particles.
        :return: Mixed derivative at distance r.
        """
        _, sigma, lam = self._params
        alpha_LJ = self.alpha_LJ

        A, _, _, _, _, _, gA, _, r6_over_sig6 = self._core_quantities(r)

        # A_sigma = dA/dsigma
        A_sigma = -6.0 * r6_over_sig6 / sigma

        # d²V / (depsilon dsigma) = d/depsilon (dV/dsigma)
        #                        = 4 λ^n g'(A) A_sigma
        return 4.0 * lam**self.n * gA * A_sigma

    def depsilondlambda(self, r: float) -> float:
        """
        Mixed derivative d²V/(depsilon dlambda).

        :param r: Distance between two particles.
        :return: Mixed derivative at distance r.
        """
        _, sigma, lam = self._params
        alpha_LJ = self.alpha_LJ

        A, _, _, _, _, g, gA, _, _ = self._core_quantities(r)

        h = lam**self.n
        h1 = self.n * lam ** (self.n - 1)

        # A_lambda = dA/dlambda
        A_lambda = -2.0 * alpha_LJ * (1.0 - lam)

        # dV/depsilon = 4 h g(A)
        # d²V/(depsilon dlambda) = 4 [ h'(λ) g + h g'(A) A_lambda ]
        return 4.0 * (h1 * g + h * gA * A_lambda)

    def dsigma_2(self, r: float) -> float:
        """
        Second derivative with respect to sigma (d²V/dsigma²).

        :param r: Distance between two particles.
        :return: Second derivative at distance r.
        """
        epsilon, sigma, lam = self._params
        alpha_LJ = self.alpha_LJ

        A, _, _, _, _, _, gA, gAA, r6_over_sig6 = self._core_quantities(r)

        # A_sigma and A_sigma2
        A_sigma = -6.0 * r6_over_sig6 / sigma
        A_sigma2 = 42.0 * r6_over_sig6 / sigma**2

        # d²V/dsigma² = 4 ε λ^n [ g''(A) (A_sigma)^2 + g'(A) A_sigma2 ]
        return 4.0 * epsilon * lam**self.n * (gAA * A_sigma**2 + gA * A_sigma2)

    def dsigmadlambda(self, r: float) -> float:
        """
        Mixed derivative d²V/(dsigma dlambda).

        :param r: Distance between two particles.
        :return: Mixed derivative at distance r.
        """
        epsilon, sigma, lam = self._params
        alpha_LJ = self.alpha_LJ

        A, _, _, _, _, _, gA, gAA, r6_over_sig6 = self._core_quantities(r)

        h = lam**self.n
        h1 = self.n * lam ** (self.n - 1)

        # A_sigma and A_lambda
        A_sigma = -6.0 * r6_over_sig6 / sigma
        A_lambda = -2.0 * alpha_LJ * (1.0 - lam)

        # dV/dsigma = 4 ε h g'(A) A_sigma
        # d²V/(dsigma dlambda) = 4 ε [ h'(λ) g'(A) A_sigma
        #                             + h g''(A) A_lambda A_sigma ]
        return 4.0 * epsilon * (
            h1 * gA * A_sigma + h * gAA * A_lambda * A_sigma
        )

    def dlambda_2(self, r: float) -> float:
        """
        Second derivative with respect to lambda (d²V/dlambda²).

        :param r: Distance between two particles.
        :return: Second derivative at distance r.
        """
        epsilon, sigma, lam = self._params
        alpha_LJ = self.alpha_LJ

        A, _, _, _, _, g, gA, gAA, _ = self._core_quantities(r)

        h = lam**self.n
        h1 = self.n * lam ** (self.n - 1)

        # Avoid numerical issues when n = 1 by treating h2 separately
        if self.n == 1:
            h2 = 0.0
        else:
            h2 = self.n * (self.n - 1) * lam ** (self.n - 2)

        # A_lambda and A_lambda2
        A_lambda = -2.0 * alpha_LJ * (1.0 - lam)
        A_lambda2 = 2.0 * alpha_LJ

        # For V = 4 ε h(λ) g(A(λ)):
        # dV/dλ   = 4 ε [ h' g + h g' A' ]
        # d²V/dλ² = 4 ε [ h'' g + 2 h' g' A' + h ( g'' (A')² + g' A'' ) ]
        return 4.0 * epsilon * (
            h2 * g
            + 2.0 * h1 * gA * A_lambda
            + h * (gAA * A_lambda**2 + gA * A_lambda2)
        )
