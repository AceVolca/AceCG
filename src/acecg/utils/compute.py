# AceCG/utils/compute.py
import numpy as np
from typing import Dict, Optional, Tuple
from ..potentials.base import BasePotential

# ===========================
# Gradient and Hessian Terms
# ===========================

def dUdLByFrame(
    pair2potential: Dict[Tuple[str, str], BasePotential],
    pair2distance_frame: Dict[int, Dict[Tuple[str, str], np.ndarray]],
) -> np.ndarray:
    """
    Compute per-frame dU/dλ values from stored pairwise distances and potential derivatives.

    Parameters
    ----------
    pair2potential : dict
        Dictionary mapping (type1, type2) pairs to potential objects.
        Each potential must define:
        - .n_params() : number of parameters of this potential
        - .dparam_names() : list of derivative method names (e.g., ["dA"])
        - Each method should accept an array of distances and return an array of dU/dλ values.
    pair2distance_frame : dict
        pair2distance_frame[frame][pair] = array of distances at that frame.

    Returns
    -------
    dUdL_frame : np.ndarray
        Array of shape (n_frames, n_dparams), where each row is dU/dλ at that frame.
    """
    frames = sorted(pair2distance_frame.keys())
    n_frames = len(frames)
    n_params = sum(pot.n_params() for pot in pair2potential.values())
    dUdL_frame = np.zeros((n_frames, n_params))

    for i, frame in enumerate(frames):
        j = 0
        for pair, pot in pair2potential.items():
            distances = pair2distance_frame[frame].get(pair, np.array([]))
            for method in pot.dparam_names():
                dUdL_frame[i, j] = np.sum(getattr(pot, method)(distances)) if distances.size > 0 else 0.0
                j += 1

    return dUdL_frame


def dUdL(
    dUdL_frame: np.ndarray,
    frame_weight: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the time-averaged dU/dλ vector over all frames.

    Parameters
    ----------
    dUdL_frame : np.ndarray
        Array of shape (n_frames, n_params), typically from `dUdLByFrame(...)`,
        where each row is the gradient of U with respect to parameters at one frame.
    frame_weight : np.ndarray, optional
        Length-n_frames weight array. If None, uniform average is used.

    Returns
    -------
    dUdL_avg : np.ndarray
        1D array, averaged gradient vector.
    
    Notes
    -----
    - This is the ⟨dU/dλ⟩ vector used in gradient-based optimization or in analytical REM loss.
    - If `frame_weight` is provided, it will be normalized first then applied.
    """
    if frame_weight is None:
        return np.mean(dUdL_frame, axis=0)
    frame_weight = frame_weight / np.sum(frame_weight)
    return dUdL_frame.T @ frame_weight


def d2UdLjdLk_Matrix(
    pair2potential: Dict[Tuple[str, str], BasePotential],
    pair2distance_frame: Dict[int, Dict[Tuple[str, str], np.ndarray]],
    frame_weight: Optional[np.ndarray] = None
) -> np.ndarray: # <d2U/dLj/dLk>
    """
    Compute the Hessian matrix ⟨∂²U/∂λⱼ∂λₖ⟩ as a time-averaged sum over frames.

    Parameters
    ----------
    pair2potential : dict
        Each potential must define:
        - .n_params() : number of parameters of this potential
        - .d2param_names() : symmetric 2D list of method names (strings)
    pair2distance_frame : dict
        pair2distance_frame[frame][pair] = distances for each frame.
    frame_weight : np.ndarray, optional
        Weight per frame. Defaults to uniform average.

    Returns
    -------
    d2UdLjdLk : np.ndarray
        A square matrix of shape (n_params, n_params) representing the second derivatives
		of the potential energy with respect to each pair of parameters, time-averaged over frames.
        
    Notes
    -----
    - If `frame_weight` is provided, it will be normalized first then applied.
    """
    frames = sorted(pair2distance_frame.keys())
    n_frames = len(frames)
    n_params = sum(p.n_params() for p in pair2potential.values())

    if frame_weight is None:
        frame_weight = np.ones(n_frames)
    frame_weight /= frame_weight.sum()

    d2UdLjdLk = np.zeros((n_params, n_params))

    for i, frame in enumerate(frames):
        anchor = 0 # anchor for each pair potential in <d2U/dLj/dLk> matrix
        for pair, pot in pair2potential.items():
            distances = pair2distance_frame[frame].get(pair, np.array([]))
            # update [.n_params, .n_params] segment corresponding to the second derivative of this pair potential
            for j in range(pot.n_params()):
                for k in range(j, pot.n_params()):
                    name = pot.d2param_names()[j][k]
                    val = np.sum(getattr(pot, name)(distances)) if distances.size > 0 else 0.0
                    d2UdLjdLk[anchor+j, anchor+k] += val * frame_weight[i]
                    d2UdLjdLk[anchor+k, anchor+j] = d2UdLjdLk[anchor+j, anchor+k] # symmetric
            anchor += pot.n_params()

    return d2UdLjdLk


def dUdLj_dUdLk_Matrix(
    dUdL_frame: np.ndarray,
    frame_weight: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the outer product matrix ⟨dU/dλⱼ · dU/dλₖ⟩ averaged over trajectory frames.

    Parameters
    ----------
    dUdL_frame : np.ndarray
        Array of shape (n_frames, n_params) containing dU/dλ values for each parameter at each frame.
        Typically the output of `dUdLByFrame(...)`.
    frame_weight : np.ndarray, optional
        Length-n_frames array of weights for each frame. If None, a uniform average is used.

    Returns
    -------
    dUdLj_dUdLk : np.ndarray
        A square matrix of shape (n_params, n_params) representing the time-averaged
        ⟨dU/dλⱼ · dU/dλₖ⟩ correlation matrix.

    Notes
    -----
    - The result is symmetric and positive semi-definite.
    - This matrix is useful for computing the Fisher information or the covariance
      of energy gradients with respect to force field parameters.
    - The average is weighted per frame by `frame_weight[i]`, defaulting to uniform if not provided.
    - If `frame_weight` is provided, it will be normalized first then applied.
    """
    n_frames, n_params = dUdL_frame.shape
    if frame_weight is None:
        frame_weight = np.ones(n_frames)
    frame_weight /= frame_weight.sum()

    mat = np.zeros((n_params, n_params))
    for i in range(n_frames):
        mat += np.outer(dUdL_frame[i], dUdL_frame[i]) * frame_weight[i]

    return mat


def Hessian(
    beta: float,
    d2UdLjdLk: np.ndarray,
    dUdLj_dUdLk: np.ndarray,
    dUdL: np.ndarray
) -> np.ndarray:
    """
    Compute the Hessian matrix of the relative entropy S_rel with respect to λ parameters.

    This corresponds to the second derivative of the REM loss:
        H_jk = β · [ -⟨∂²U/∂λⱼ∂λₖ⟩ + β·⟨∂U/∂λⱼ · ∂U/∂λₖ⟩ - β·⟨∂U/∂λⱼ⟩ · ⟨∂U/∂λₖ⟩ ]

    Parameters
    ----------
    beta : float
        Inverse temperature 1/(k_B·T), where T is temperature and k_B is Boltzmann constant.
    d2UdLjdLk : np.ndarray
        Array of shape (n_params, n_params), the mean second derivatives ⟨∂²U/∂λⱼ∂λₖ⟩.
        Typically from `d2UdLjdLkMatrix(...)`.
    dUdLj_dUdLk : np.ndarray
        Array of shape (n_params, n_params), the mean outer products ⟨∂U/∂λⱼ · ∂U/∂λₖ⟩.
        From `dUdLj_dUdLk_Matrix(...)`.
    dUdL : np.ndarray
        Array of shape (n_params,), the mean ⟨∂U/∂λⱼ⟩ vector.
        From `dUdL(...)`.

    Returns
    -------
    hessian : np.ndarray
        The Hessian matrix of shape (n_params, n_params) representing ∂²S_rel / ∂λⱼ∂λₖ.

    Notes
    -----
    This is used in second-order optimization or uncertainty quantification of relative entropy minimization (REM).
    """
    return beta * (-d2UdLjdLk + beta * dUdLj_dUdLk - beta * np.outer(dUdL, dUdL))