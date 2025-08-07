class LennardJonesPotential ():
    """
    Class representing the Lennard-Jones potential.
    This potential is commonly used to model interactions between particles.
    """

    def __init__(self, epsilon: float, sigma: float) -> None:
        """
        Initialize the Lennard-Jones potential with parameters epsilon and sigma.
        V_LJ (r) = 4ε[(σ/r)¹² - (σ/r)⁶]

        :param epsilon: Depth of the potential well.
        :param sigma: Finite distance at which the potential is zero.
        """
        self.epsilon = epsilon
        self.sigma = sigma

    def value(self, r: float) -> float:
        """
        Compute the Lennard-Jones potential at a distance r.

        :param r: Distance between two particles.
        :return: The value of the Lennard-Jones potential at distance r.
        """
        if r <= 0:
            raise ValueError("Distance r must be positive.")
        
        sigma_over_r = self.sigma / r
        return 4 * self.epsilon * (sigma_over_r**12 - sigma_over_r**6)

    def dEpsilon(self, r: float) -> float:
        """
        Compute the derivative of the Lennard-Jones potential with respect to epsilon.

        :param r: Distance between two particles.
        :return: The derivative of the potential with respect to epsilon at distance r.
        """
        if r <= 0:
            raise ValueError("Distance r must be positive.")
        
        sigma_over_r = self.sigma / r
        return 4 * (sigma_over_r**12 - sigma_over_r**6)

    def dSigma(self, r: float) -> float:
        """
        Compute the derivative of the Lennard-Jones potential with respect to sigma.

        :param r: Distance between two particles.
        :return: The derivative of the potential with respect to sigma at distance r.
        """
        if r <= 0:
            raise ValueError("Distance r must be positive.")
        
        sigma_over_r = self.sigma / r
        return 4 * self.epsilon * (12 * sigma_over_r**11 / r - 6 * sigma_over_r**5 / r)

