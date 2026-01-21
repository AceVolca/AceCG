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

def UByFrame(
    pair2potential: Dict[Tuple[str, str], BasePotential],
    pair2distance_frame: Dict[int, Dict[Tuple[str, str], np.ndarray]],
) -> np.ndarray:
    """
    Compute per-frame U values from stored pairwise distances and potentials.

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
    U_frame : np.ndarray
        Array of shape (n_frames, k), where each row is U at that frame.
    """
    frames = sorted(pair2distance_frame.keys())
    n_frames = len(frames)
    U_frame = np.zeros((n_frames, k_interactions))

    for i, frame in enumerate(frames):
        k=0
        for pair, pot in pair2potential.items():
            distances = pair2distance_frame[frame].get(pair, np.array([]))
            U_frame[i, k] = pot.value(distances) if distances.size > 0 else 0.0
            k+=1
    return U_frame


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


def U_total(
        U_frame: np.ndarray,
        frame_weight: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the time-averaged U vector over all frames.

    Parameters
    ----------
    U_frame : np.ndarray
        Array of shape (n_frames, n_params), typically from `UByFrame(...)`,
        where each row is U for the interaction at one frame.
    frame_weight : np.ndarray, optional
        Length-n_frames weight array. If None, uniform average is used.

    Returns
    -------
    U_avg : np.ndarray
        1D array, averaged energy vector.

    Notes
    -----
    - This is the U vector used in gradient-based optimization or in analytical L0 REM loss.
    - If `frame_weight` is provided, it will be normalized first then applied.
    """
    if frame_weight is None:
        return np.mean(U_frame, axis=0)
    frame_weight = frame_weight / np.sum(frame_weight)
    return U_frame.T @ frame_weight


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
    d2UdLjdLk_AA: np.ndarray,
    d2UdLjdLk_CG: np.ndarray,
    dUdLj_dUdLk_CG: np.ndarray,
    dUdL_CG: np.ndarray
) -> np.ndarray:
    """
    Compute the Hessian matrix of the relative entropy S_rel with respect to λ parameters.

    This corresponds to the second derivative of the REM loss:
        H_jk = β · [⟨∂²U/∂λⱼ∂λₖ⟩_T - ⟨∂²U/∂λⱼ∂λₖ⟩_M + β·⟨∂U/∂λⱼ · ∂U/∂λₖ⟩_M - β·⟨∂U/∂λⱼ⟩_M · ⟨∂U/∂λₖ⟩_M ]

    Parameters
    ----------
    beta : float
        Inverse temperature 1/(k_B·T), where T is temperature and k_B is Boltzmann constant.
    d2UdLjdLk_AA : np.ndarray
        Array of shape (n_params, n_params), the mean second derivatives ⟨∂²U/∂λⱼ∂λₖ⟩ evaluated in the reference ensemble.
        Typically from `d2UdLjdLkMatrix(...)`.
    d2UdLjdLk_CG : np.ndarray
        Array of shape (n_params, n_params), the mean second derivatives ⟨∂²U/∂λⱼ∂λₖ⟩ evaluated in the model ensemble.
        Typically from `d2UdLjdLkMatrix(...)`.
    dUdLj_dUdLk_CG : np.ndarray
        Array of shape (n_params, n_params), the mean outer products ⟨∂U/∂λⱼ · ∂U/∂λₖ⟩ in the model ensemble.
        From `dUdLj_dUdLk_Matrix(...)`.
    dUdL_CG : np.ndarray
        Array of shape (n_params,), the mean ⟨∂U/∂λⱼ⟩ vector evaluated in the model ensemble.
        From `dUdL(...)`.

    Returns
    -------
    hessian : np.ndarray
        The Hessian matrix of shape (n_params, n_params) representing ∂²S_rel / ∂λⱼ∂λₖ.

    Notes
    -----
    This is used in second-order optimization or uncertainty quantification of relative entropy minimization (REM).
    """

    return beta * (d2UdLjdLk_AA - d2UdLjdLk_CG + beta * dUdLj_dUdLk_CG - beta * np.outer(dUdL_CG, dUdL_CG))


def KL_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute the discrete Kullback-Leibler (KL) divergence D_KL(p || q).

    This function measures how one discrete probability distribution `p`
    diverges from a second distribution `q`:

        D_KL(p || q) = Σ_i p_i * log(p_i / q_i)

    A small constant (1e-9) is added to both `p` and `q` to avoid numerical
    issues when any entry is zero.

    Parameters
    ----------
    p : np.ndarray
        1D array of non-negative values representing the first probability
        distribution. Should sum to ~1. Shape: (N,).
    q : np.ndarray
        1D array of non-negative values representing the second probability
        distribution. Should sum to ~1. Shape must match `p`.

    Returns
    -------
    divergence : float
        The KL divergence D_KL(p || q). Always >= 0; equals 0 only if
        `p` and `q` are identical (within numerical precision).

    Notes
    -----
    - This is the **discrete** version of KL divergence, assuming `p` and `q`
      are defined over the same set of N categories.
    - The addition of 1e-9 prevents log(0) and division-by-zero errors, but
      slightly biases the result for extremely sparse distributions.
    - KL divergence is not symmetric: D_KL(p || q) ≠ D_KL(q || p).
    """
    p += 1E-9
    q += 1E-9
    return np.sum(p * (np.log(p) - np.log(q)))


def dUdLByBin(
    dUdL_frame: np.ndarray,
    bin_idx_frame: np.ndarray,
    frame_weight: Optional[np.ndarray] = None
): # compute ⟨dU/dλⱼ⟩_CG|s = ⟨dU/dλⱼδ[s(r)-s]⟩ / ⟨δ[s(r)-s]⟩
    """
    Compute conditional averages of dU/dλ over histogram bins of a collective variable.

    Groups per-frame dU/dλ values by discrete bin indices (e.g., histogram bins
    of a reaction coordinate s(r)) and computes:

        dUdL_bin[idx]     = ⟨dU/dλ_j δ[s(r) - s_bin]⟩_CG
        p_bin[idx]        = ⟨δ[s(r) - s_bin]⟩_CG
        dUdL_given_bin[idx] = ⟨dU/dλ_j⟩_CG | s_bin

    where δ[...] is the Kronecker delta selecting frames in the given bin.
    Weighted averages are computed using `frame_weight` if provided.

    Parameters
    ----------
    dUdL_frame : np.ndarray
        Array of per-frame derivatives, shape (n_frames, n_dparams).
        Typically the output from `dUdLByFrame`.
    bin_idx_frame : np.ndarray
        1D integer array of length n_frames, assigning each frame to a bin
        index (e.g., histogram bin of reaction coordinate).
    frame_weight : np.ndarray, optional
        1D array of length n_frames giving weights for each frame
        (e.g., reweighting factors). If None, all frames are weighted equally.
        Will be normalized to sum to 1 internally.

    Returns
    -------
    dUdL_bin : dict[int, np.ndarray]
        Weighted numerator term for each bin:
        ⟨dU/dλ_j δ[s(r) - s_bin]⟩_CG.
        Value shape: (n_dparams,).
    p_bin : dict[int, float]
        Probability mass in each bin:
        ⟨δ[s(r) - s_bin]⟩_CG.
        This is the sum of normalized frame weights in that bin.
    dUdL_given_bin : dict[int, np.ndarray]
        Conditional average in each bin:
        ⟨dU/dλ_j⟩_CG | s_bin.
        Value shape: (n_dparams,).
        Should satisfy:
            dUdL_bin[idx] / p_bin[idx] ≈ dUdL_given_bin[idx]

    Notes
    -----
    - The equality above holds within numerical precision
    - The helper function `dUdL(...)` must accept the subset of dUdL values
      and weights and return the weighted average over frames in the bin.
    - Bin indices in `bin_idx_frame` need not be contiguous; any integer
      values are accepted.
    """
    if frame_weight is None:
        frame_weight = np.ones(len(dUdL_frame))
    frame_weight /= frame_weight.sum()
    
    dUdL_bin = {} # {bin_idx: weighted average of dUdL at this bin, ⟨dU/dλⱼδ[s(r)-s]⟩_CG}
    p_bin = {} # {bin_idx: probability at this bin, <δ[s(r)-s]>_CG}
    dUdL_given_bin = {} # {bin_idx: ⟨dU/dλⱼ⟩_CG|s}
    idx_set = set(bin_idx_frame)
    
    for idx in idx_set:
        frame_mask = bin_idx_frame == idx
        dUdL_bin[idx] = dUdL_frame[frame_mask].T @ frame_weight[frame_mask]
        p_bin[idx] = np.sum(frame_weight[frame_mask])
        dUdL_given_bin[idx] = dUdL(dUdL_frame[frame_mask], frame_weight[frame_mask])

    return dUdL_bin, p_bin, dUdL_given_bin


def UByBin(
        U_frame: np.ndarray,
        bin_idx_frame: np.ndarray,
        frame_weight: Optional[np.ndarray] = None
):  # compute ⟨U⟩_CG|s = ⟨Uδ[s(r)-s]⟩ / ⟨δ[s(r)-s]⟩
    """
    Compute conditional averages of U over histogram bins of a collective variable.

    Groups per-frame U values by discrete bin indices (e.g., histogram bins
    of a reaction coordinate s(r)) and computes:

        U_bin[idx]     = ⟨U δ[s(r) - s_bin]⟩_CG
        p_bin[idx]        = ⟨δ[s(r) - s_bin]⟩_CG
        U_given_bin[idx] = ⟨U⟩_CG | s_bin

    where δ[...] is the Kronecker delta selecting frames in the given bin.
    Weighted averages are computed using `frame_weight` if provided.

    Parameters
    ----------
    U_frame : np.ndarray
        Array of per-frame derivatives, shape (n_frames, k_interactions).
        Typically the output from `UByFrame`.
    bin_idx_frame : np.ndarray
        1D integer array of length n_frames, assigning each frame to a bin
        index (e.g., histogram bin of reaction coordinate).
    frame_weight : np.ndarray, optional
        1D array of length n_frames giving weights for each frame
        (e.g., reweighting factors). If None, all frames are weighted equally.
        Will be normalized to sum to 1 internally.

    Returns
    -------
    U_bin : dict[int, np.ndarray]
        Weighted numerator term for each bin:
        ⟨U δ[s(r) - s_bin]⟩_CG.
        Value shape: (k_interactions,).
    p_bin : dict[int, float]
        Probability mass in each bin:
        ⟨δ[s(r) - s_bin]⟩_CG.
        This is the sum of normalized frame weights in that bin.
    U_given_bin : dict[int, np.ndarray]
        Conditional average in each bin:
        <U>_CG | s_bin.
        Value shape: (K_interactions,).
        Should satisfy:
            U_bin[idx] / p_bin[idx] ≈ U_given_bin[idx]

    Notes
    -----
    - The equality above holds within numerical precision
    - The helper function `U_total(...)` must accept the subset of U values
      and weights and return the weighted average over frames in the bin.
    - Bin indices in `bin_idx_frame` need not be contiguous; any integer
      values are accepted.
    """
    if frame_weight is None:
        frame_weight = np.ones(len(dUdL_frame))
    frame_weight /= frame_weight.sum()

    U_bin = {}  # {bin_idx: weighted average of dUdL at this bin, ⟨Uδ[s(r)-s]⟩_CG}
    p_bin = {}  # {bin_idx: probability at this bin, <δ[s(r)-s]>_CG}
    U_given_bin = {}  # {bin_idx: ⟨U⟩_CG|s}
    idx_set = set(bin_idx_frame)

    for idx in idx_set:
        frame_mask = bin_idx_frame == idx
        U_bin[idx] = U_frame[frame_mask].T @ frame_weight[frame_mask]
        p_bin[idx] = np.sum(frame_weight[frame_mask])
        U_given_bin[idx] = U_total(U_frame[frame_mask], frame_weight[frame_mask])

    return dUdL_bin, p_bin, dUdL_given_bin