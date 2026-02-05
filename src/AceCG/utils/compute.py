# AceCG/utils/compute.py
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from ..potentials.base import BasePotential
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance

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


def compute_weighted_rdf(
    u: mda.Universe,
    sel1: str,
    sel2: str,
    weights: Optional[np.ndarray] = None,
    r_range: Tuple[float, float] = (0.0, 15.0),
    nbins: int = 150,
    pbc: bool = True,
    exclude_self: bool = False,
    mode: str = "liquid_gr",  # "liquid_gr" | "bound_pr" | "liquid_gr_fixed_rho"
    rho2_override: Optional[float] = None,
    dtype: np.dtype = np.float64,
) -> Dict[str, Any]:
    """
    Compute weighted distance distributions between two MDAnalysis selections.

    This function provides a unified interface for computing either:

    (1) **Liquid-style radial distribution functions g(r)**, normalized by the
        bulk number density of `sel2`, or
    (2) **Bound-state pair-distance probability densities p(r)**, suitable for
        protein–protein complexes that remain associated and do not sample bulk.

    The statistical estimator supports per-frame importance weights (e.g. from
    REM / umbrella reweighting). The weights do NOT need to be normalized; only
    their relative values matter.

    Parameters
    ----------
    u : MDAnalysis.Universe
        Universe containing topology and trajectory.
    sel1, sel2 : str
        MDAnalysis selection strings defining the two groups whose pairwise
        distances are analyzed.
    weights : np.ndarray, optional
        Per-frame importance weights of shape (n_frames,).
        - If None, all frames are weighted equally (standard time average).
        - If provided, weights are applied consistently in both numerator and
          normalization terms.
        Notes:
            Weights do NOT need to sum to 1. Global scaling cancels in all outputs.
    r_range : (float, float)
        Minimum and maximum distances (r_min, r_max) for histogramming.
    nbins : int
        Number of histogram bins.
    pbc : bool
        Whether to apply periodic boundary conditions. Required for liquid-style
        g(r) normalization.
    exclude_self : bool
        If True and `sel1` and `sel2` overlap, exclude i == j self-pairs.
    mode : str
        Defines the physical meaning and normalization of the output.
        Supported values:
            - "liquid_gr":
                Standard RDF:
                    g(r) = <h(r)> / [N1 * rho2 * shell_volume]
                where rho2 = N2 / V is computed per frame.
                Appropriate for bulk liquids or well-defined concentrations.
            - "liquid_gr_fixed_rho":
                Same as "liquid_gr", but using a fixed reference density
                `rho2_override` instead of per-frame N2/V.
                Useful when comparing trajectories with different box sizes
                or concentrations.
            - "bound_pr":
                Bound-state pair-distance probability density p(r), normalized
                such that:
                    sum_i p(r_i) * dr = 1
                This mode does NOT use box volume or bulk density and is the
                recommended choice for protein–protein complexes that remain bound.
    rho2_override : float, optional
        Reference number density used when mode == "liquid_gr_fixed_rho".
        Must be provided in that case.
    dtype : numpy dtype
        Floating-point type used for accumulation and normalization.

    Returns
    -------
    out : dict
        Dictionary containing histogrammed distance distributions and metadata.
        Common keys:
            - "r"        : (nbins,) bin centers
            - "edges"    : (nbins+1,) bin edges
            - "dr"       : float bin width
            - "hist"     : (nbins,) weighted distance counts
            - "shell_vol": (nbins,) spherical shell volumes

        Mode-dependent keys:
            - If mode in {"liquid_gr", "liquid_gr_fixed_rho"}:
                - "g"        : (nbins,) radial distribution function g(r)
                - "norm_gr"  : (nbins,) normalization term
            - If mode == "bound_pr":
                - "p"        : (nbins,) pair-distance probability density p(r)
                - "norm_pr"  : float normalization constant

        Metadata:
            - "meta" : dict with diagnostic information, including:
                * mode, selections, number of frames used
                * box volume range (for liquid modes)
                * weight sums and effective pair counts

    Notes
    -----
    - For **protein–protein bound states**, liquid-style g(r) can become extremely
      large and box-size dependent. In such cases, `mode="bound_pr"` should be used
      for meaningful AA–CG comparisons.
    - This function computes distributions over *all pairwise distances* between
      `sel1` and `sel2`. If a more restrictive definition (e.g. closest-contact
      distances) is desired, preprocessing of the distance list is recommended.
    - The implementation mirrors the weighting conventions used elsewhere in
      AceCG (e.g. REM estimators), ensuring internal consistency.
    """

    dtype = np.dtype(dtype).type  # callable scalar type
    ag1 = u.select_atoms(sel1)
    ag2 = u.select_atoms(sel2)
    n_frames = len(u.trajectory)

    # weights
    if weights is None:
        w = np.ones(n_frames, dtype=dtype)
    else:
        w = np.asarray(weights, dtype=dtype)
        if w.shape[0] != n_frames:
            raise ValueError(f"weights length {w.shape[0]} != n_frames {n_frames}")

    rmin, rmax = float(r_range[0]), float(r_range[1])
    edges = np.linspace(rmin, rmax, nbins + 1, dtype=dtype)
    dr = float(edges[1] - edges[0])
    r = 0.5 * (edges[:-1] + edges[1:])
    shell_vol = (4.0 * np.pi / 3.0) * (edges[1:]**3 - edges[:-1]**3)

    hist = np.zeros(nbins, dtype=dtype)

    # For liquid g(r)
    norm_gr = np.zeros(nbins, dtype=dtype)

    # For bound p(r): total weighted number of considered pairs (for normalization)
    # (Sum_t w_t * N_pairs_t)
    wNpairs_sum = dtype(0.0)

    V_list = []
    n_used = 0

    for iframe, ts in enumerate(u.trajectory):
        wt = w[iframe]
        if wt == 0.0:
            continue

        N1 = len(ag1)
        N2 = len(ag2)
        if N1 == 0 or N2 == 0:
            continue

        if pbc:
            dims = ts.dimensions
            if dims is None or dims[0] == 0:
                raise ValueError("pbc=True but no valid ts.dimensions.")
            V = dtype(dims[0] * dims[1] * dims[2])
            box = dims
        else:
            V = None
            box = None

        pairs, dists = capped_distance(
            ag1.positions, ag2.positions,
            max_cutoff=rmax, min_cutoff=rmin,
            box=box if pbc else None,
            return_distances=True,
        )
        if exclude_self:
            dists = dists[pairs[:, 0] != pairs[:, 1]]

        h, _ = np.histogram(dists, bins=edges)
        h = h.astype(dtype, copy=False)
        hist += wt * h

        # pair count used in this frame (for p(r) normalization)
        # Using N1*N2 is consistent with "all pairs" definition.
        # If you later change to unique pairs actually considered, replace here.
        Npairs = dtype(N1 * N2)
        wNpairs_sum += wt * Npairs

        # liquid-style normalization
        if mode in ("liquid_gr", "liquid_gr_fixed_rho"):
            if not pbc:
                raise ValueError("liquid_gr requires pbc and a meaningful volume.")
            if mode == "liquid_gr_fixed_rho":
                if rho2_override is None:
                    raise ValueError("rho2_override must be provided for liquid_gr_fixed_rho.")
                rho2 = dtype(rho2_override)
            else:
                rho2 = dtype(N2) / V
            norm_gr += wt * dtype(N1) * rho2 * shell_vol
            V_list.append(float(V))

        n_used += 1

    out: Dict[str, Any] = {
        "r": r,
        "edges": edges,
        "dr": dr,
        "hist": hist,
        "shell_vol": shell_vol,
    }

    # Compute outputs per mode
    if mode in ("liquid_gr", "liquid_gr_fixed_rho"):
        with np.errstate(divide="ignore", invalid="ignore"):
            g = hist / norm_gr
            g[~np.isfinite(g)] = 0.0
        out["g"] = g
        out["norm_gr"] = norm_gr

    if mode == "bound_pr":
        # probability density p(r) s.t. sum p*dr = 1
        # p(r_i) = (weighted counts in bin i) / (weighted total pair counts * dr)
        denom = wNpairs_sum * dtype(dr)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = hist / denom
            p[~np.isfinite(p)] = 0.0
        out["p"] = p
        out["norm_pr"] = denom  # scalar

    meta = {
        "mode": mode,
        "sel1": sel1,
        "sel2": sel2,
        "n_frames": int(n_frames),
        "n_used_frames": int(n_used),
        "pbc": bool(pbc),
        "exclude_self": bool(exclude_self),
        "r_range": (rmin, rmax),
        "nbins": int(nbins),
        "N1": int(len(ag1)),
        "N2": int(len(ag2)),
        "rho2_override": float(rho2_override) if rho2_override is not None else None,
        "V_min": float(np.min(V_list)) if V_list else None,
        "V_max": float(np.max(V_list)) if V_list else None,
        "w_sum": float(np.sum(w)),
        "wNpairs_sum": float(wNpairs_sum),
    }
    out["meta"] = meta
    return out


# defining types
Pair = Tuple[str, str]
Pair2DistanceByFrame = Dict[int, Dict[Pair, np.ndarray]]

def compute_weighted_pair_distance_pdfs(
    pair2distance_frame: Pair2DistanceByFrame,
    frame_weight: Optional[Union[np.ndarray, Dict[int, float]]] = None,
    r_range: Tuple[float, float] = (0.0, 25.0),
    nbins: int = 250,
    dtype: np.dtype = np.float64,
    drop_empty_pairs: bool = True,
) -> Dict[Pair, Dict[str, Any]]:
    """
    Compute per-pair weighted distance PDFs p(r) from stored per-frame distances.

    This is the "bound-state friendly" alternative to liquid-style g(r):
    it returns the probability density of pair distances, normalized such that
        sum_i p_i * dr = 1.

    Parameters
    ----------
    pair2distance_frame : dict
        Nested dictionary:
            pair2distance_frame[frame_idx][pair] = np.ndarray of distances in that frame
        where `pair` is typically (type1, type2) or (site_i, site_j).
    frame_weight : np.ndarray or dict, optional
        - If None: uniform weights over frames.
        - If np.ndarray: length must equal number of frames (sorted by frame index).
        - If dict: mapping {frame_idx: weight}. Missing frames default to 0.
        Notes:
            Weights do NOT need to sum to 1; only relative weights matter.
    r_range : (rmin, rmax)
        Histogram range in same distance units as your stored distances (usually Å).
    nbins : int
        Number of bins.
    dtype : numpy dtype
        Accumulation dtype.
    drop_empty_pairs : bool
        If True, pairs with zero total samples are omitted from output.

    Returns
    -------
    out_by_pair : dict
        out_by_pair[pair] = {
            "r": (nbins,) bin centers,
            "p": (nbins,) probability density p(r),
            "edges": (nbins+1,) bin edges,
            "dr": float bin width,
            "hist": (nbins,) weighted counts per bin (sum_t w_t * h_t),
            "norm": float, normalization scalar = (sum_t w_t * n_samples_t) * dr,
            "meta": {
                "pair": pair,
                "n_frames": int,
                "n_used_frames": int,
                "w_sum": float,
                "w_abs_sum": float,
                "n_samples_weighted": float,   # sum_t w_t * n_samples_t
                "n_samples_raw": int,          # sum_t n_samples_t (unweighted)
                "r_range": (rmin, rmax),
                "nbins": int,
            }
        }

    Notes
    -----
    - This computes a *distance PDF* p(r), not liquid RDF g(r). It is robust to
      differing box volumes / concentrations between trajectories and is usually
      the right object for protein-protein bound-state comparisons.
    """
    dtype = np.dtype(dtype).type

    frames = sorted(pair2distance_frame.keys())
    n_frames = len(frames)
    if n_frames == 0:
        return {}

    # Build frame weights aligned with sorted frames
    if frame_weight is None:
        w = np.ones(n_frames, dtype=dtype)
    elif isinstance(frame_weight, dict):
        w = np.array([frame_weight.get(fr, 0.0) for fr in frames], dtype=dtype)
    else:
        w = np.asarray(frame_weight, dtype=dtype)
        if w.shape[0] != n_frames:
            raise ValueError(f"frame_weight length {w.shape[0]} != n_frames {n_frames}")

    rmin, rmax = float(r_range[0]), float(r_range[1])
    if not (rmax > rmin >= 0.0):
        raise ValueError(f"Invalid r_range={r_range}")

    edges = np.linspace(rmin, rmax, nbins + 1, dtype=dtype)
    dr = float(edges[1] - edges[0])
    r = 0.5 * (edges[:-1] + edges[1:])

    # Collect union of pairs across frames
    pair_set = set()
    for fr in frames:
        pair_set.update(pair2distance_frame[fr].keys())

    out_by_pair: Dict[Pair, Dict[str, Any]] = {}

    w_sum = float(np.sum(w))
    w_abs_sum = float(np.sum(np.abs(w)))

    for pair in sorted(pair_set):
        hist = np.zeros(nbins, dtype=dtype)

        n_used = 0
        n_samples_raw = 0
        n_samples_weighted = dtype(0.0)

        for i, fr in enumerate(frames):
            wt = w[i]
            if wt == 0.0:
                continue

            d = pair2distance_frame[fr].get(pair, None)
            if d is None:
                continue
            d = np.asarray(d)
            if d.size == 0:
                continue

            # histogram within [rmin, rmax]
            h, _ = np.histogram(d, bins=edges)
            hist += wt * h.astype(dtype, copy=False)

            n = int(d.size)
            n_used += 1
            n_samples_raw += n
            n_samples_weighted += wt * dtype(n)

        if n_samples_raw == 0:
            if not drop_empty_pairs:
                out_by_pair[pair] = {
                    "r": r, "p": np.zeros_like(r), "edges": edges, "dr": dr,
                    "hist": hist, "norm": 0.0,
                    "meta": {
                        "pair": pair,
                        "n_frames": int(n_frames),
                        "n_used_frames": int(n_used),
                        "w_sum": w_sum,
                        "w_abs_sum": w_abs_sum,
                        "n_samples_weighted": float(n_samples_weighted),
                        "n_samples_raw": int(n_samples_raw),
                        "r_range": (rmin, rmax),
                        "nbins": int(nbins),
                    }
                }
            continue

        # PDF normalization: sum_i p_i * dr = 1
        norm = float(n_samples_weighted) * dr
        with np.errstate(divide="ignore", invalid="ignore"):
            p = hist / norm
            p[~np.isfinite(p)] = 0.0

        out_by_pair[pair] = {
            "r": r,
            "p": p,
            "edges": edges,
            "dr": dr,
            "hist": hist,
            "norm": norm,
            "meta": {
                "pair": pair,
                "n_frames": int(n_frames),
                "n_used_frames": int(n_used),
                "w_sum": w_sum,
                "w_abs_sum": w_abs_sum,
                "n_samples_weighted": float(n_samples_weighted),
                "n_samples_raw": int(n_samples_raw),
                "r_range": (rmin, rmax),
                "nbins": int(nbins),
            },
        }

    return out_by_pair