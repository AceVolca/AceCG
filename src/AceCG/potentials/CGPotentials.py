import numpy as np
from typing import Dict, Tuple, List

class GaussianPotential:
    def __init__(self, typ1, typ2, A, r0, sigma, cutoff):
        self.typ1  = typ1
        self.typ2  = typ2
        self.A     = A
        self.r0    = r0
        self.sigma = sigma
        self.cutoff = cutoff
        self.params = np.array([A, r0, sigma])
        self.param_names = ["A", "r0", "sigma"] # must match self.params
        self.dparams = ["dA", "dr0", "dsigma"] # must match self.params
        self.d2params = [["dA_2", "dAdr0", "dAdsigma"], ["dAdr0", "dr0_2", "dr0dsigma"], ["dAdsigma", "dr0dsigma", "dsigma_2"]] # must match self.params
        self.pot_typ = "gauss"

    def value(self, r):
        x = r - self.r0
        return self.A / (self.sigma*np.sqrt(2*np.pi)) * np.exp(- x**2/(2*self.sigma**2))

    def dA(self, r):
        x = r - self.r0
        return 1   / (self.sigma*np.sqrt(2*np.pi)) * np.exp(- x**2/(2*self.sigma**2))

    def dr0(self, r):
        x = r - self.r0
        return self.A*x / (self.sigma**3*np.sqrt(2*np.pi)) * np.exp(- x**2/(2*self.sigma**2))

    def dsigma(self, r):
        x = r - self.r0
        phi = np.exp(- x**2/(2*self.sigma**2))
        return self.A * phi / np.sqrt(2*np.pi) * (x**2 - self.sigma**2) / self.sigma**4

    def dA_2(self, r):
        return 0

    def dAdr0(self, r):
        x = r - self.r0
        phi = np.exp(- x**2/(2*self.sigma**2))
        # ∂²U/∂A∂r0 =  x/(σ³√2π) · φ
        return x / (self.sigma**3*np.sqrt(2*np.pi)) * phi

    def dAdsigma(self, r):
        x = r - self.r0
        phi = np.exp(- x**2/(2*self.sigma**2))
        # ∂²U/∂A∂σ = (x² - σ²)/(σ⁴√2π) · φ
        return (x**2 - self.sigma**2) / (self.sigma**4*np.sqrt(2*np.pi)) * phi

    def dr0_2(self, r):
        x = r - self.r0
        phi = np.exp(- x**2/(2*self.sigma**2))
        # ∂²U/∂r0² = A/(σ³√2π) · (x²/σ² - 1) · φ
        return self.A / (self.sigma**3*np.sqrt(2*np.pi)) * (x**2/self.sigma**2 - 1) * phi

    def dr0dsigma(self, r):
        x = r - self.r0
        phi = np.exp(- x**2/(2*self.sigma**2))
        # ∂²U/∂r0∂σ = A·x/(√2π) · (x² - 3σ²)/σ⁶ · φ
        return self.A*x / np.sqrt(2*np.pi) * (x**2 - 3*self.sigma**2) / self.sigma**6 * phi

    def dsigma_2(self, r):
        x = r - self.r0
        phi = np.exp(- x**2/(2*self.sigma**2))
        # ∂²U/∂σ² = A/(√2π) · (x⁴ - 5x²σ² + 2σ⁴)/σ⁷ · φ
        return self.A / np.sqrt(2*np.pi) * (x**4 - 5*x**2*self.sigma**2 + 2*self.sigma**4) / self.sigma**7 * phi


def FFParamArray(
    pair2potential: Dict[Tuple[str, str], object]
) -> np.ndarray:
    """
    Concatenate all potential parameters into a single 1D NumPy array.

    Parameters
    ----------
    pair2potential : dict
        Dictionary mapping atom type pairs to potential objects.
        Each potential must have a `.params` property that returns a 1D NumPy array.

    Returns
    -------
    np.ndarray
        A 1D array of all force field parameters concatenated in order.
    """
    return np.concatenate([pot.params for pot in pair2potential.values()])


def FFParamIndexMap(
    pair2potential: Dict[Tuple[str, str], object]
) -> List[Tuple[Tuple[str, str], str]]:
    """
    Create an index map from parameter index to (pair_type, param_name).

    Parameters
    ----------
    pair2potential : dict
        Dictionary mapping (type1, type2) to potential objects.
        Each potential must define `.param_names`, a list of parameter names
        in the same order as `.params`.

    Returns
    -------
    index_map : list of tuples
        A list where index_map[i] = ((type1, type2), param_name) indicates
        that parameter i corresponds to that type-pair and parameter name.
        The order matches the output of `FFParamArray(pair2potential)`.
    """
    index_map = []

    for pair, potential in pair2potential.items():
        for param_name in potential.param_names:
            index_map.append((pair, param_name))

    return index_map


def ReadLmpFF(file, typ_sel=None):
    pair2potential = {}

    lines = open(file, "r").readlines()
    for line in lines:
        if line.find("pair_coeff") != -1:
            tmp = line.split()
            if typ_sel is None or tmp[3] in typ_sel: # this pair type is selected and defined
                if tmp[3] == "gauss/wall" or "gauss/cut":
                    pair2potential[(tmp[1], tmp[2])] = GaussianPotential(
                        tmp[1], tmp[2], float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])
                    )
    
    return pair2potential


def WriteLmpFF(old_file, new_file, L_new, typ_sel=None):
    idx = 0 # index of L_new

    lines = open(old_file, "r").readlines()
    for i, line in enumerate(lines):
        if line.find("pair_coeff") != -1:
            tmp = line.split()
            if typ_sel is None or tmp[3] in typ_sel: # this pair type is selected and defined
                if tmp[3] == "gauss/wall" or "gauss/cut":
                    for j in range(3): tmp[4+j] = str(L_new[idx+j]) # update A r0 sigma
                    idx += 3
                lines[i] = "   ".join(tmp)+"\n"
    
    # write new FF
    f = open(new_file, "w")
    for line in lines: f.write(line)
