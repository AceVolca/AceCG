# AceCG/potentials/lennardjones.py
from .base import BasePotential
import numpy as np

class LennardJonesPotential(BasePotential):
    """
    Class representing the Lennard-Jones potential.
    This potential is commonly used to model interactions between particles.
    """

    def __init__(self, typ1, typ2, epsilon: float, sigma: float, cutoff: float) -> None:
        """Initialize the Lennard-Jones 12-6 potential.

        Parameters
        ----------
        typ1, typ2 : int or str
            Pair type labels.
        epsilon : float
            Depth of the potential well.
        sigma : float
            Distance scale in ``V(r) = 4 epsilon [(sigma/r)^12 - (sigma/r)^6]``.
        cutoff : float
            Pair cutoff distance stored for LAMMPS export.
        """
        super().__init__()
        self.typ1  = typ1
        self.typ2  = typ2
        self.cutoff = cutoff
        self._params = np.array([epsilon, sigma])
        self._params_to_scale = [0]
        self._param_names = ["epsilon", "sigma"]
        self._dparam_names = ["depsilon", "dsigma"]
        self._d2param_names = [
            ["depsilon_2", "depsilondsigma"],
            ["depsilondsigma", "dsigma_2"]
        ]

    def is_param_linear(self) -> np.ndarray:
        """Return which LJ parameters enter the energy linearly."""
        return np.array([True, False], dtype=bool)

    def value(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the Lennard-Jones potential at a distance r.

        :param r: Distance between two particles.
        :return: The value of the Lennard-Jones potential at distance r.
        """

        r = np.asarray(r, dtype=float)
        epsilon, sigma = self._params
        
        sigma_over_r = sigma / r
        return 4 * epsilon * (sigma_over_r**12 - sigma_over_r**6)
    
    def force(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the force (negative gradient of the potential) at a distance r.

        :param r: Distance between two particles.
        :return: The force at distance r.
        """

        r = np.asarray(r, dtype=float)
        epsilon, sigma = self._params

        sigma_over_r = sigma / r
        return 24 * epsilon / r * (2 * sigma_over_r**12 - sigma_over_r**6)

    def depsilon(self, r: float) -> float:
        """
        Compute the derivative of the Lennard-Jones potential with respect to epsilon.

        :param r: Distance between two particles.
        :return: The derivative of the potential with respect to epsilon at distance r.
        """
        
        r = np.asarray(r, dtype=float)
        _, sigma = self._params

        sigma_over_r = sigma / r
        return 4 * (sigma_over_r**12 - sigma_over_r**6)

    def dsigma(self, r: float) -> float:
        """
        Compute the derivative of the Lennard-Jones potential with respect to sigma.

        :param r: Distance between two particles.
        :return: The derivative of the potential with respect to sigma at distance r.
        """

        r = np.asarray(r, dtype=float)
        epsilon, sigma = self._params

        sigma_over_r = sigma / r
        return 24 * epsilon / r * (2 * sigma_over_r**11 - sigma_over_r**5)

    def depsilon_2(self, r: float) -> float:
        """
        Compute the second derivative of the Lennard-Jones potential with respect to epsilon.

        :param r: Distance between two particles.
        :return: The second derivative of the potential with respect to epsilon at distance r (which is 0).
        """
        return np.zeros_like(np.asarray(r, dtype=float), dtype=float)

    def depsilondsigma(self, r: float) -> float:
        """
        Compute the mixed derivative of the Lennard-Jones potential with respect to epsilon and sigma.

        :param r: Distance between two particles.
        :return: The mixed derivative of the potential at distance r.
        """

        r = np.asarray(r, dtype=float)
        epsilon, sigma = self._params

        sigma_over_r = sigma / r
        return 24 / r * (2 * sigma_over_r**11 - sigma_over_r**5)

    def dsigma_2(self, r: float) -> float:
        """
        Compute the second derivative of the Lennard-Jones potential with respect to sigma.

        :param r: Distance between two particles.
        :return: The second derivative of the potential with respect to sigma at distance r.
        """
        
        r = np.asarray(r, dtype=float)
        epsilon, sigma = self._params

        sigma_over_r = sigma / r
        return 24 * epsilon / r**2 * (22 * sigma_over_r**10 - 5 * sigma_over_r**4)
