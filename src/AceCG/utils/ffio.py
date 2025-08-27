# AceCG/utils/ffio.py
import numpy as np
import fnmatch, re
from scipy.optimize import lsq_linear
from typing import Dict, Tuple, Iterable, Optional, List
from ..potentials.base import BasePotential

from ..potentials.gaussian import GaussianPotential
from ..potentials.lennardjones import LennardJonesPotential
from ..potentials.multi_gaussian import MultiGaussianPotential


POTENTIAL_REGISTRY = {
    "gauss/cut": GaussianPotential,
    "gauss/wall": GaussianPotential,
    "lj/cut": LennardJonesPotential,
    ".table": MultiGaussianPotential, 
}


def FFParamArray(pair2potential: Dict[Tuple[str, str], BasePotential]) -> np.ndarray:
    """
    Concatenate all potential parameters into a single 1D NumPy array.

    Parameters
    ----------
    pair2potential : dict
        Mapping (type1, type2) to potential objects, each with `.get_params` method.

    Returns
    -------
    np.ndarray
        1D array of all force field parameters.
    """
    return np.concatenate([pot.get_params() for pot in pair2potential.values()])


def FFParamIndexMap(pair2potential: Dict[Tuple[str, str], BasePotential]) -> List[Tuple[Tuple[str, str], str]]:
    """
    Create a map from parameter index to (pair_type, param_name).

    Returns
    -------
    List of tuples: [ ((type1, type2), param_name), ... ]
    Matching order of FFParamArray.
    """
    index_map = []
    for pair, pot in pair2potential.items():
        for name in pot.param_names():
            index_map.append((pair, name))
    return index_map


def BuildGlobalMask(
    pair2potential: Dict[Tuple[str, str], BasePotential],
    patterns: Optional[Dict[Tuple[str, str], Iterable[str]]] = None,
    mode: str = "freeze",
    strict: bool = False,
    case_sensitive: bool = True,
    global_patterns: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """
    Build a boolean mask aligned with FFParamArray(pair2potential),
    where True = trainable, False = frozen.

    Matching is per-pair and based on param_names() strings.

    Pattern syntax
    --------------
    • Exact match: "A_0"
    • Glob (default): 
        - "A*"     → matches "A", "A0", "A_0", "ABC"  (any name starting with "A")
        - "A_*"    → matches "A_0", "A_sigma" (requires the underscore)
        - "*r0*"   → matches any name containing "r0"
    • Regex (prefix with "re:"):
        - "re:^A$"     → matches only the exact name "A"
        - "re:^A_\\d+$" → matches "A_0", "A_1", etc.

    Parameters
    ----------
    pair2potential : dict[(type1,type2) -> BasePotential]
        Potentials concatenated by FFParamArray; iteration order must match FFParamArray.
    patterns : dict[(type1,type2) -> Iterable[str]], optional
        Pair-specific patterns.
    mode : {"freeze","train"}, default "freeze"
        - "freeze": black-list mode (all trainable by default, matching ones are frozen).
        - "train" : white-list mode (all frozen by default, matching ones are trainable).
    strict : bool, default False
        If True, raise if a pattern matches no parameter names.
    case_sensitive : bool, default True
        Match case-sensitively.
    global_patterns : list of str, optional
        Patterns applied to all pairs in addition to pair-specific patterns.
        Example: ["A_*"] freezes/trains every parameter named like A_* in all pairs.

    Returns
    -------
    mask : np.ndarray of bool
        Global mask aligned with FFParamArray order.
        True = trainable, False = frozen.
    
    Examples
    -------

    """
    # compute global offsets
    pair_offsets = {}
    total = 0
    for pair, pot in pair2potential.items():
        pair_offsets[pair] = total
        total += pot.n_params()

    if mode not in ("freeze","train"):
        raise ValueError(f"mode must be 'freeze' or 'train', got {mode}")

    default_train = (mode == "freeze")  # freeze模式：默认True, train模式：默认False
    mask = np.full(total, default_train, dtype=bool)

    def maybe_norm(s): return s if case_sensitive else s.lower()

    # iterate pairs
    for pair, pot in pair2potential.items():
        base = pair_offsets[pair]
        local_names = list(pot.param_names())
        norm_names  = [maybe_norm(n) for n in local_names]

        # collect patterns for this pair
        pats = []
        if global_patterns: pats.extend(global_patterns)
        if patterns and pair in patterns: pats.extend(patterns[pair])

        for pat in pats:
            use_regex = pat.startswith("re:")
            pat_body = pat[3:] if use_regex else pat
            hits = []
            if use_regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                regex = re.compile(pat_body, flags=flags)
                hits = [i for i,name in enumerate(local_names) if regex.search(name)]
            else:
                pat_cmp = maybe_norm(pat_body)
                hits = [i for i,n in enumerate(norm_names) if fnmatch.fnmatch(n, pat_cmp)]

            if not hits and strict:
                raise KeyError(f"No params matched pattern '{pat}' for pair {pair}. "
                               f"Available: {local_names}")

            for i_local in hits:
                gi = base + i_local
                if mode == "freeze":
                    mask[gi] = False
                else:  # "train"
                    mask[gi] = True
    return mask
        

# MultiGaussian I/O stuffs
def _parse_lammps_table(table_path: str):
    """
    Read LAMMPS pair_style table file.
    Lines format: idx r V [F]   or   r V [F]
    Returns
    -------
    r : np.ndarray
        Distances
    V : np.ndarray
        Potential values
    F : np.ndarray or None
        Forces if present in file, else None
    """
    r_list, v_list, f_list = [], [], []
    with open(table_path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                if len(parts) >= 4:      # idx r V F
                    float(parts[0]); float(parts[1]); float(parts[2]); float(parts[3])
                    # idx present
                    r_val = float(parts[1]); v_val = float(parts[2]); f_val = float(parts[3])
                    r_list.append(r_val); v_list.append(v_val); f_list.append(f_val)
                elif len(parts) == 3:    # idx r V   or   r V F
                    # try as (idx,r,V)
                    try:
                        r_val = float(parts[1]); v_val = float(parts[2])
                        r_list.append(r_val); v_list.append(v_val)
                    except Exception:
                        # try as (r,V,F)
                        r_val = float(parts[0]); v_val = float(parts[1]); f_val = float(parts[2])
                        r_list.append(r_val); v_list.append(v_val); f_list.append(f_val)
                elif len(parts) == 2:    # r V
                    r_val = float(parts[0]); v_val = float(parts[1])
                    r_list.append(r_val); v_list.append(v_val)
            except ValueError:
                continue

    if not r_list:
        raise ValueError(f"No numeric (r,V) rows found in {table_path}")

    r = np.asarray(r_list, dtype=float)
    V = np.asarray(v_list, dtype=float)
    F = np.asarray(f_list, dtype=float) if f_list else None

    # filter nan/inf, sort by r
    m = np.isfinite(r) & np.isfinite(V)
    if F is not None:
        m = m & np.isfinite(F)
    r, V = r[m], V[m]
    F = F[m] if F is not None else None
    idx = np.argsort(r)
    r, V = r[idx], V[idx]
    F = F[idx] if F is not None else None

    return r, V, F


def _gaussian_basis(r: np.ndarray, r0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Construct Gaussian basis matrix B, shape (N, K):
        B[:,k] = exp(-(r - r0_k)^2 / (2 sigma_k^2)) / (sigma_k * sqrt(2π))
    """
    r = r.reshape(-1, 1)
    r0 = r0.reshape(1, -1)
    sigma = sigma.reshape(1, -1)
    phi = np.exp(- (r - r0)**2 / (2.0 * sigma**2))
    return phi / (sigma * np.sqrt(2.0 * np.pi))


def _gaussian_basis_dr(r: np.ndarray, r0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Derivative of Gaussian basis with respect to r.
        d/dr[phi_k(r)] = [-(r - r0_k) / sigma_k^2] * phi_k(r)
    """
    B = _gaussian_basis(r, r0, sigma)
    r = r.reshape(-1, 1)
    r0 = r0.reshape(1, -1)
    sigma = sigma.reshape(1, -1)
    return (-(r - r0) / (sigma**2)) * B


def _model_from_params(r, params):
    A, r0, sig = _unpack_params(params)
    return _gaussian_basis(r, r0, sig) @ A


def _model_dr_from_params(r, params):
    A, r0, sig = _unpack_params(params)
    return _gaussian_basis_dr(r, r0, sig) @ A


def _init_grid(r: np.ndarray, V: np.ndarray, n_gauss: int):
    """
    Initial guess:
      - r0_k: equal spacing on [rmin, rmax]
      - sigma_k: total width / (2*n_gauss)
      - A_k: linear fit (r0, sigma)
    """
    rmin, rmax = float(r.min()), float(r.max())
    width = rmax - rmin if rmax > rmin else 1.0
    r0_init = np.linspace(rmin + 0.15*width, rmax - 0.15*width, n_gauss)
    sigma_init = np.full(n_gauss, max(width / (2.0 * n_gauss), 1e-3))

    B = _gaussian_basis(r, r0_init, sigma_init)  # (N,K)
    # small normalized hill
    lam = 1e-8
    A_init = np.linalg.lstsq(B.T @ B + lam * np.eye(n_gauss), B.T @ V, rcond=None)[0]
    return A_init, r0_init, sigma_init


def _pack_params(A: np.ndarray, r0: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    flatten (A, r0, sigma) as [A0,r0_0,sigma_0, A1,r0_1,sigma_1, ...]
    """
    n = len(A)
    out = np.empty(3*n, dtype=float)
    out[0::3] = A
    out[1::3] = r0
    out[2::3] = sigma
    return out


def _unpack_params(p: np.ndarray):
    """
    reversed _pack_params
    """
    A = p[0::3]
    r0 = p[1::3]
    sigma = p[2::3]
    return A, r0, sigma


def _solve_A_with_anchors(r, V, r0, sigma, anchors_r, w_data, w_c0, w_c1,
                          A_lower=None, A_upper=None):
    """
    Solve min || w_data*(B A - V) ||^2 + || w_c0*(Ba A) ||^2 + || w_c1*(Da A) ||^2
    with element-wise bounds on A (optional).
    """
    B  = _gaussian_basis(r, r0, sigma)
    Ba = _gaussian_basis(anchors_r, r0, sigma)
    Da = _gaussian_basis_dr(anchors_r, r0, sigma)

    M = np.vstack([w_data * B, w_c0 * Ba, w_c1 * Da])
    y = np.concatenate([w_data * V, np.zeros_like(anchors_r), np.zeros_like(anchors_r)])

    K = B.shape[1]
    lb = np.full(K, -np.inf) if A_lower is None else np.array(A_lower, dtype=float)
    ub = np.full(K,  np.inf) if A_upper is None else np.array(A_upper, dtype=float)

    # small ridge via Tikhonov can be emulated by augmenting rows
    res = lsq_linear(M, y, bounds=(lb, ub), method="trf", max_iter=5000, lsq_solver="exact")
    return res.x


def _make_cutoff_anchors(r: np.ndarray, cutoff: float, n_anchor: int, span: float) -> np.ndarray:
    """
    Generate anchor points in [cutoff - span, cutoff] (inclusive).
    If cutoff exceeds data max, use data max instead.
    """
    rc = float(cutoff) if np.isfinite(cutoff) else float(r.max())
    rmax = float(r.max())
    rc = min(rc, rmax)
    r_start = max(rc - span, r.min())
    if n_anchor <= 1 or r_start >= rc:
        return np.array([rc], dtype=float)
    return np.linspace(r_start, rc, n_anchor, dtype=float)


def FitMultiGaussianFromLmpTable(
    table_path: str,
    typ1: str,
    typ2: str,
    n_gauss: int,
    cutoff: float = np.inf,
    return_params: bool = False,
    use_scipy: bool = True,
    bounds: dict = None,
    init: np.ndarray = None,
    random_state: int = 0,
    # --- cutoff anchoring (kept) ---
    anchor_to_cutoff: bool = True,
    n_anchor: int = 3,
    anchor_span: float = 0.6,
    weight_data: float = 1.0,
    weight_c0: float = 10.0,
    weight_c1: float = 5.0,
    # --- repulsive gaussian constraint ---
    repulsive_index: int | None = 0,     # which component must be repulsive; None to disable
    repulsive_A_min: float = 1e-8,       # enforce A_k >= this
    repulsive_r0_max: float = 0.0,       # enforce r0_k <= this (≤0 for “at/inside contact”)
):
    """
    Fit a MultiGaussianPotential to LAMMPS table data (r, V[, F]) with
    optional cutoff anchoring and repulsive component constraints.

    The fitting procedure:
    1. Read r, V (and optionally F) from the LAMMPS table file.
    2. Build an initial guess for Gaussian parameters (A, r0, sigma).
    3. Solve a constrained linear least squares for amplitudes A,
       including optional cutoff anchors.
    4. Optionally refine all parameters (A, r0, sigma) using
       SciPy's nonlinear least_squares with bounds.

    Features
    --------
    • Cutoff anchoring: Enforces V(rc) ≈ 0 and dV/dr|rc ≈ 0 using
      additional "anchor points" near the cutoff radius.
    • Repulsive Gaussian constraint: Designate one Gaussian
      component as a short-range repulsive term with:
          A >= repulsive_A_min (positive amplitude)
          r0 <= repulsive_r0_max (usually ≤ 0, i.e. at/inside contact)

    Parameters
    ----------
    table_path : str
        Path to LAMMPS table file (lines: idx r V [F] or r V [F]).
    typ1, typ2 : str
        Particle type identifiers (for potential bookkeeping).
    n_gauss : int
        Number of Gaussian components in the expansion.
    cutoff : float
        Cutoff radius. Anchors and force zeroing are applied here.
    return_params : bool
        If True, also return the optimized parameter array.
    use_scipy : bool
        Whether to run nonlinear least_squares refinement after
        the initial linear solve.
    bounds : dict
        Parameter bounds, e.g. {"A": (amin, amax),
                                "r0": (rmin, rmax),
                                "sigma": (smin, smax)}.
        Defaults: A free, r0 ∈ (-∞, r.max), sigma ∈ [1e-6, ∞).
    init : np.ndarray, optional
        Initial parameter array (length 3*n_gauss). If None,
        a grid-based initialization is used.
    random_state : int
        Random seed for reproducibility.
    anchor_to_cutoff : bool
        If True, include cutoff anchors to enforce V(rc)=0, dV/dr|rc=0.
    n_anchor : int
        Number of anchor points in [cutoff - anchor_span, cutoff].
    anchor_span : float
        Width of the anchoring region before cutoff.
    weight_data : float
        Weight factor for data residuals.
    weight_c0 : float
        Weight factor for cutoff value (V=0) constraints.
    weight_c1 : float
        Weight factor for cutoff slope (dV/dr=0) constraints.
    repulsive_index : int or None
        Index of the Gaussian component to force as repulsive.
        If None, no repulsive constraint is applied.
    repulsive_A_min : float
        Minimum amplitude for the repulsive component (A >= this).
    repulsive_r0_max : float
        Maximum center for the repulsive component (r0 <= this).
        Typically set ≤ 0 to place repulsion at/inside contact.

    Returns
    -------
    pot : MultiGaussianPotential
        The fitted potential object.
    params : np.ndarray, optional
        The optimized parameter vector (if return_params=True).
    """
    rng = np.random.default_rng(random_state)
    r, V, F = _parse_lammps_table(table_path)

    # initial guess
    if init is None:
        A0, r00, s0 = _init_grid(r, V, n_gauss)
        p0 = _pack_params(A0, r00, s0)
    else:
        p0 = np.asarray(init, dtype=float)
        if p0.size != 3 * n_gauss:
            raise ValueError(f"init length must be {3*n_gauss}, got {p0.size}")

    # solve A
    A_lin, r0_lin, sigma_lin = _unpack_params(p0)
    if anchor_to_cutoff:
        anchors_r = _make_cutoff_anchors(r, cutoff, n_anchor, anchor_span)
    else:
        anchors_r = np.array([], dtype=float)

    K = n_gauss
    A_lower = np.full(K, -np.inf)
    if repulsive_index is not None and 0 <= repulsive_index < K:
        A_lower[repulsive_index] = max(0.0, repulsive_A_min)

    A_lin = _solve_A_with_anchors(
        r, V, r0_lin, sigma_lin, anchors_r,
        w_data=float(weight_data), w_c0=float(weight_c0), w_c1=float(weight_c1),
        A_lower=A_lower, A_upper=None,
    )
    p0 = _pack_params(A_lin, r0_lin, sigma_lin)

    # project initial guess onto repulsive constraints (if enabled) ---
    if repulsive_index is not None:
        k = int(repulsive_index)
        A0, r00, s0 = _unpack_params(p0)
        if 0 <= k < n_gauss:
            A0[k]  = max(A0[k],  repulsive_A_min)
            r00[k] = min(r00[k], repulsive_r0_max)
            p0 = _pack_params(A0, r00, s0)

    # refine with scipy
    if use_scipy:
        try:
            from scipy.optimize import least_squares

            anchors = anchors_r  # shape (Na,)
            def resid(params):
                # residual
                res_data = _model_from_params(r, params) - V
                # cutoff restraint
                if anchor_to_cutoff and anchors.size > 0:
                    res_c0 = _model_from_params(anchors, params)       # 目标=0
                    res_c1 = _model_dr_from_params(anchors, params)    # 目标=0
                    return np.concatenate([
                        weight_data * res_data,
                        weight_c0   * res_c0,
                        weight_c1   * res_c1
                    ])
                else:
                    return weight_data * res_data

            # bounds
            def _pair(v): return (-np.inf, np.inf) if v is None else v
            A_lb, A_ub = _pair(bounds["A"])   if bounds and "A"   in bounds else (-np.inf, np.inf)
            # r0_lb, r0_ub = _pair(bounds["r0"]) if bounds and "r0" in bounds else (r.min(), r.max())
            sig_lb, sig_ub = _pair(bounds["sigma"]) if bounds and "sigma" in bounds else (1e-6, np.inf)
            if bounds and "r0" in bounds:
                r0_lb, r0_ub = _pair(bounds["r0"])
            else:
                r0_lb, r0_ub = (-np.inf, r.max())   # or (-cutoff, r.max())

            lb = np.empty_like(p0);  ub = np.empty_like(p0)
            lb[0::3] = A_lb;   ub[0::3] = A_ub
            lb[1::3] = r0_lb;  ub[1::3] = r0_ub
            lb[2::3] = sig_lb; ub[2::3] = sig_ub

            # --- tighten bounds for the designated repulsive component ---
            if repulsive_index is not None and 0 <= int(repulsive_index) < n_gauss:
                k = int(repulsive_index)
                lb[3*k + 0] = max(lb[3*k + 0], repulsive_A_min)   # A_k ≥ A_min ≥ 0
                ub[3*k + 1] = min(ub[3*k + 1], 0.0)               # r0_k ≤ 0
                # (sigma bounds keep using global setting; add custom if you want)

            res = least_squares(resid, p0, bounds=(lb, ub), method="trf", max_nfev=3000, verbose=0)
            p_opt = res.x
        except Exception:
            p_opt = p0
    else:
        p_opt = p0

    pot = MultiGaussianPotential(typ1, typ2, n_gauss=n_gauss, cutoff=cutoff, init_params=p_opt)
    return (pot, p_opt.copy()) if return_params else pot


def ReadLmpFF(
        file: str,
        typ_sel: Optional[List[str]] = None, 
        n_gauss: Optional[int] = 16,
        bounds: Optional[Dict] = {"sigma": [0.1, 5]},
        repulsive_index: Optional[int] = None,
        repulsive_A_min: Optional[float] = 1e-4,
        repulsive_r0_max: Optional[float] = 0
) -> Dict[Tuple[str, str], BasePotential]:
    """
    Generalized reader for LAMMPS force field files using registered potential types.

    Supports both analytic pair styles (e.g., gauss/cut, lj/cut) and
    table-based potentials (".table"), where a MultiGaussianPotential
    is fitted to the table data.

    Parameters
    ----------
    file : str
        Path to a LAMMPS force field file containing `pair_coeff` lines.
    typ_sel : list of str, optional
        Only load specified pair styles (e.g., ["gauss/cut"]). If None,
        all registered styles are considered.
    n_gauss : int, optional
        Number of Gaussian components to use when fitting a .table file
        to a MultiGaussianPotential. Default = 16.
    bounds : dict, optional
        Bounds for parameters during fitting, e.g.
            {"sigma": (0.1, 5.0), "A": (amin, amax), "r0": (rmin, rmax)}.
        Default = {"sigma": [0.1, 5]}.
    repulsive_index : int or None, optional
        Index of the Gaussian component to enforce as a repulsive core.
        If None (default), no repulsive constraint is applied.
    repulsive_A_min : float, optional
        Minimum amplitude for the repulsive Gaussian (A >= this).
        Default = 1e-4.
    repulsive_r0_max : float, optional
        Maximum center location for the repulsive Gaussian (r0 <= this).
        Typically set to 0. Default = 0.

    Returns
    -------
    pair2potential : dict
        Mapping (type1, type2) → BasePotential object.
        If style is ".table", the potential is a fitted
        MultiGaussianPotential; otherwise, an analytic potential
        (GaussianPotential, LennardJonesPotential, etc.).

    Notes
    -----
    • For ".table" entries, this function calls
      FitMultiGaussianFromLmpTable() internally with the provided
      n_gauss, bounds, and repulsive constraints.
    • The cutoff is automatically taken as the maximum r in the table.
    • Anchor points are added near the cutoff to enforce V(rc) ≈ 0
      and dV/dr|rc ≈ 0.
    """
    pair2potential = {}

    with open(file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if "pair_coeff" in line:
            tmp = line.split()
            style = tmp[3]
            if style.find(".table") != -1: style = ".table"
            if typ_sel is None or style in typ_sel:
                if style in POTENTIAL_REGISTRY:
                    constructor = POTENTIAL_REGISTRY[style]
                    if style == ".table":
                        r, v, f = _parse_lammps_table(tmp[3])
                        pair2potential[(tmp[1], tmp[2])] = FitMultiGaussianFromLmpTable(
                            tmp[3], 
                            tmp[1], 
                            tmp[2],
                            n_gauss,
                            cutoff=np.max(r),
                            use_scipy=True,
                            bounds=bounds,
                            anchor_to_cutoff=True,        # anchor to 0 at cutoff
                            n_anchor=4,                   # 4 anchors within [rc - anchor_span, rc]
                            anchor_span=0.6,
                            weight_c0=20.0,               # constrant strength of U
                            weight_c1=10.0,               # constrant strength of dU/dr
                            # repulsive
                            repulsive_index=repulsive_index,          # or "auto"
                            repulsive_A_min=repulsive_A_min,       # tune to your scale (kcal/mol)
                            repulsive_r0_max=repulsive_r0_max,       # r0 ≤ 0 Å
                        )
                    else:
                        params = list(map(float, tmp[4:]))
                        pair2potential[(tmp[1], tmp[2])] = constructor(tmp[1], tmp[2], *params)

    return pair2potential


def WriteLmpTable(
    filename: str,
    r: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    comment: str = None,
    table_name: str = "Table1"
):
    """
    Write a LAMMPS-style pair_style table file from arrays of r, V(r), F(r).

    Parameters
    ----------
    filename : str
        Output file path.
    r : np.ndarray
        1D array of distances.
    V : np.ndarray
        1D array of potential energy values at r.
    F : np.ndarray
        1D array of forces (-dV/dr) at r.
    comment : str, optional
        Comment string to put at the top of the file.
    table_name : str, optional
        Label for the LAMMPS table (default "Table1").
    """
    r = np.asarray(r, dtype=float)
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=float)
    assert r.shape == V.shape == F.shape, "r, V, F must have the same shape"

    with open(filename, "w") as f:
        if comment is not None:
            for line in comment.splitlines():
                f.write(f"# {line}\n")

        npoints = len(r)
        f.write(f"{table_name}\n")
        f.write(f"N {npoints} R {r[0]:.6f} {r[-1]:.6f}\n")

        for i, (ri, vi, fi) in enumerate(zip(r, V, F), start=1):
            f.write(f"{i:6d}  {ri:16.8f}  {vi:16.8e}  {fi:16.8e}\n")


def WriteLmpFF(
    old_file: str,
    new_file: str,
    pair2potential: Dict[Tuple[str, str], BasePotential],
    typ_sel: Optional[List[str]] = None
):
    """
    Write updated parameters to a new LAMMPS-style force field file.

    This function updates the `pair_coeff` lines in a LAMMPS force field file
    based on the current parameters stored in `pair2potential`.

    Behavior differs depending on the pair style:

    • Analytic styles (e.g., "gauss/cut", "lj/cut"):
        The numeric parameters in the corresponding `pair_coeff` line are replaced
        with the updated values from the potential object.

    • Table style (".table"):
        The `pair_coeff` line in the file is left unchanged, but the associated
        table file itself is regenerated. The new table file contains:
            - column 1: index
            - column 2: r
            - column 3: V(r)   (recomputed from the fitted potential)
            - column 4: F(r)   (recomputed as -dV/dr)
        A header comment is also written to indicate the table source.

    Parameters
    ----------
    old_file : str
        Path to the original LAMMPS force field file (to be read and used as a template).
    new_file : str
        Path to the new file to write (will contain updated coefficients or table references).
    pair2potential : dict
        Mapping (type1, type2) → BasePotential object.
        For ".table" entries, this must be a MultiGaussianPotential (or similar) with
        `.value(r)` and `.force(r)` methods implemented.
    typ_sel : list of str, optional
        If provided, only update the specified pair styles. If None (default),
        all styles found in the file are considered.

    Notes
    -----
    • For ".table" styles, the table file specified in the original pair_coeff line
      is overwritten with new (r, V, F) values computed from the potential object.
    • The main LAMMPS force field file (`new_file`) is always written,
      even if only comments or table updates were applied.
    """
    L_new = FFParamArray(pair2potential)
    idx = 0
    with open(old_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "pair_coeff" in line:
            tmp = line.split()
            style = tmp[3]
            if style.find(".table") != -1: style = ".table"
            pair = (tmp[1], tmp[2])

            if typ_sel is None or style in typ_sel:
                if pair in pair2potential:
                    if style == ".table": # do not change the output line, update table file
                        r, v, f = _parse_lammps_table(tmp[3])
                        WriteLmpTable(tmp[3], r, pair2potential[pair].value(r), pair2potential[pair].force(r), f"# Table {tmp[3]}: id, r, potential, force", pair[0]+"-"+pair[1])
                    else:
                        n_param = pair2potential[pair].n_params()
                        tmp[4:4 + n_param] = map(str, L_new[idx:idx + n_param])
                        idx += n_param
                        lines[i] = "   ".join(tmp) + "\n"

    with open(new_file, "w") as f:
        f.writelines(lines)