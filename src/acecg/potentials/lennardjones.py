class LennardJonesPotential ():
    """
    Class representing the Lennard-Jones potential.
    This potential is commonly used to model interactions between particles.
    """

    def __init__(self, epsilon: float, sigma: float, cutoff: float) -> None:
        """
        Initialize the Lennard-Jones potential with parameters epsilon and sigma.
        V_LJ (r) = 4ε[(σ/r)¹² - (σ/r)⁶]

        :param epsilon: Depth of the potential well.
        :param sigma: Finite distance at which the potential is zero.
        """
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self._param_names = ["epsilon", "sigma"]
        self._dparam_names = ["dEpsilon", "dSigma"]
        self._d2param_names = [
            ["dEpsilon_2", "dEpsilondSigma"],
            ["dEpsilondSigma", "dSigma_2"]
        ]

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
        return 24 * self.epsilon / r * (2 * sigma_over_r**12 - sigma_over_r**6)

    def dEpsilon_2(self, r: float) -> float:
        """
        Compute the second derivative of the Lennard-Jones potential with respect to epsilon.

        :param r: Distance between two particles.
        :return: The second derivative of the potential with respect to epsilon at distance r (which is 0).
        """
        return 0.0

    def dEpsilondSigma(self, r: float) -> float:
        """
        Compute the mixed derivative of the Lennard-Jones potential with respect to epsilon and sigma.

        :param r: Distance between two particles.
        :return: The mixed derivative of the potential at distance r.
        """
        if r <= 0:
            raise ValueError("Distance r must be positive.")

        sigma_over_r = self.sigma / r
        return 24 * self.epsilon / r * ( 2 * sigma_over_r**12 - sigma_over_r**6)

    def dSigma_2(self, r: float) -> float:
        """
        Compute the second derivative of the Lennard-Jones potential with respect to sigma.

        :param r: Distance between two particles.
        :return: The second derivative of the potential with respect to sigma at distance r.
        """
        if r <= 0:
            raise ValueError("Distance r must be positive.")
        
        sigma_over_r = self.sigma / r
        return 24 * self.epsilon / r**2 * (22 * sigma_over_r**10 - 5 * sigma_over_r**4)

    def param_names(self):
        return self._param_names

    def dparam_names(self):
            return self._dparam_names

    def d2param_names(self):
        return self._d2param_names

    def n_params(self):
        return len(self._params)

    def params(self):
        return [self.epsilon, self.sigma]

    def set_params(self, new_params):
        assert len(new_params) == len(self._params)
        self._params = new_params.copy()
