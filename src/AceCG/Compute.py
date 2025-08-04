import numpy as np
from typing import List, Dict, Optional, Tuple

from . import CGPotentials as cgp

def dUdLByFrame(
    pair2potential: Dict[Tuple[str, str], object],
    pair2distance_frame: Dict[int, Dict[Tuple[str, str], np.ndarray]],
) -> np.ndarray:
    """
    Compute per-frame dU/dλ values from stored pairwise distances and potential derivatives.

    Parameters
    ----------
    pair2potential : dict
        Dictionary mapping (type1, type2) pairs to potential objects.
        Each potential must define:
        - .dparams : list of derivative method names (e.g., ["dU_dA", "dU_dB"])
        - Each method should accept an array of distances and return an array of dU/dλ values.
    pair2distance_frame : dict
        Nested dictionary where pair2distance_frame[frame][pair] gives an array of pairwise distances
        for that frame and that type pair.

    Returns
    -------
    dUdL_frame : np.ndarray
        Array of shape (n_frames, n_dparams), where each row corresponds to the summed dU/dλ
        contributions for each parameter at a specific frame.
    """
    frames = sorted(pair2distance_frame.keys())
    n_frames = len(frames)
    n_params = len(cgp.FFParamArray(pair2potential))  # dparams order matches FFParamArray

    dUdL_frame = np.zeros((n_frames, n_params))

    for i, frame in enumerate(frames):
        j = 0
        for pair, potential in pair2potential.items():
            distances = pair2distance_frame[frame].get(pair, np.array([]))
            for name in potential.dparams:
                if distances.size > 0:
                    dUdL_frame[i, j] = np.sum(getattr(potential, name)(distances))
                else:
                    dUdL_frame[i, j] = 0.0
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
        Length-n_frames array of weights for each frame. If None, a uniform average is used.

    Returns
    -------
    dUdL_avg : np.ndarray
        1D array of shape (n_params,) representing the time-averaged dU/dλ vector.

    Notes
    -----
    - This is the ⟨dU/dλ⟩ vector used in gradient-based optimization or in REM loss.
    - If `frame_weight` is provided, it is treated as a normalized weight vector.
    """
    if frame_weight is None:
        return np.mean(dUdL_frame, axis=0)
    else:
        return dUdL_frame.T @ frame_weight  # shape (n_params,)


def d2UdLjdLk_Matrix(
    pair2potential: Dict[Tuple[str, str], object],
    pair2distance_frame: Dict[int, Dict[Tuple[str, str], np.ndarray]],
    frame_weight: Optional[np.ndarray] = None
) -> np.ndarray: # <d2U/dLj/dLk>
	"""
	Compute the Hessian matrix of d²U/dλj dλk as a time-averaged sum over frames.

	Parameters
	----------
	pair2potential : dict
		Dictionary mapping atom type pairs to potential objects.
		Each potential must define:
		- .params     : 1D array of parameters
		- .d2params   : symmetric 2D list of method names (strings), with shape (n_params, n_params)
	pair2distance_frame : dict
		Nested dictionary where pair2distance_frame[frame][pair] gives an array of distances.
	frame_weight : np.ndarray, optional
		Length-n_frames array of weights per frame. If None, defaults to uniform average.

	Returns
	-------
	d2UdLjdLk : np.ndarray
		A square matrix of shape (n_params, n_params) representing the second derivatives
		of the potential energy with respect to each pair of parameters, time-averaged over frames.
	"""
	frames = sorted(pair2distance_frame.keys())
	n_frames = len(frames)
	n_params = len(cgp.FFParamArray(pair2potential))  # dparams order matches FFParamArray

	if frame_weight is None:
		frame_weight = np.ones(n_frames)/n_frames # default frame_weight, averaged over trajectory

	d2UdLjdLk = np.zeros((n_params, n_params))

	for i, frame in enumerate(frames): # calculate <d2UdLjdLk> by weighted sum of d2UdLjdLk of each frame
		anchor = 0
		for pair, potential in pair2potential.items():
			distances = pair2distance_frame[frame].get(pair, np.array([]))
			for j in range(len(potential.d2params)):
				for k in range(j, len(potential.d2params[j])):
					d2UdLjdLk[anchor+j, anchor+k] += np.sum(getattr(potential, potential.d2params[j][k])(distances)) * frame_weight[i]
					d2UdLjdLk[anchor+k, anchor+j] = d2UdLjdLk[anchor+j, anchor+k]
			anchor += len(potential.params)

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
    - This matrix is useful for computing the Fisher information or the covariance
      of energy gradients with respect to force field parameters.
    - The average is weighted per frame by `frame_weight[i]`, defaulting to uniform if not provided.
    - The result is symmetric and positive semi-definite.
    """
    n_frames, n_params = dUdL_frame.shape

    if frame_weight is None:
        frame_weight = np.ones(n_frames) / n_frames

    dUdLj_dUdLk = np.zeros((n_params, n_params))

    for i in range(n_frames):
        dUdLj_dUdLk += np.outer(dUdL_frame[i], dUdL_frame[i]) * frame_weight[i]

    return dUdLj_dUdLk


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


def NRUpdate(
    dSdL: np.ndarray,
    hessian: np.ndarray,
    L_mask: np.ndarray,
    eta: float,
) -> np.ndarray:
    """
    Compute the Newton-Raphson update Δλ using the masked gradient and Hessian.

    This function applies a binary mask to select a subset of parameters for optimization,
    solves the Newton-Raphson update equation for those parameters, and returns a full-length
    update vector with zeros in the masked-out entries.

        Δλ = H⁻¹ · ∇S_rel     (restricted to masked indices)

    Parameters
    ----------
    dSdL : np.ndarray
        1D array of shape (n_params,), the gradient of S_rel w.r.t. parameters.
    hessian : np.ndarray
        2D array of shape (n_params, n_params), the Hessian matrix of S_rel.
    L_mask : np.ndarray
        Boolean array of shape (n_params,), where True indicates parameters to be updated.
    eta : float
        Learning rate of NR update, Δλ' = η * Δλ

    Returns
    -------
    update : np.ndarray
        1D array of shape (n_params,) representing the update direction Δλ.
        Entries corresponding to masked (False) indices are zero.

    Notes
    -----
    - This masked update is useful for selectively optimizing a subset of parameters
      while holding others fixed.
    - Internally uses `np.linalg.solve(H_masked, grad_masked)` for stability and speed.
    """
    dSdL_masked = dSdL[L_mask]                     # Select active gradient components
    H_masked = hessian[np.ix_(L_mask, L_mask)]     # Select sub-Hessian
    update_masked = np.linalg.solve(H_masked, dSdL_masked)  # Solve restricted linear system

    update = np.zeros_like(dSdL)                   # Full-size update vector
    update[L_mask] = update_masked                 # Fill in only active indices

    return eta * update


def REM(beta, AA_pair2distance_frame, AA_frame_weight, CG_pair2distance_frame, CG_frame_weight, pair2potential, L_mask, optimizer, eta):
    
    dUdL_AA_frame = dUdLByFrame(pair2potential, AA_pair2distance_frame)
    dUdL_AA = dUdL(dUdL_AA_frame, AA_frame_weight)
    
    dUdL_CG_frame = dUdLByFrame(pair2potential, CG_pair2distance_frame)
    dUdL_CG = dUdL(dUdL_CG_frame, CG_frame_weight)
    
    dSdL = beta * (dUdL_AA - dUdL_CG) # partial derivative of relative entropy
    
    if optimizer == "NR":
        # calculate Hessian    
        d2UdLjdLk_CG = d2UdLjdLk_Matrix(pair2potential, CG_pair2distance_frame, CG_frame_weight)
        dUdLj_dUdLk_CG = dUdLj_dUdLk_Matrix(dUdL_CG_frame, CG_frame_weight)
        H = Hessian(beta, d2UdLjdLk_CG, dUdLj_dUdLk_CG, dUdL_CG)
        
        return dUdL_AA, dUdL_CG, dSdL, H, NRUpdate(dSdL, H, L_mask, eta)