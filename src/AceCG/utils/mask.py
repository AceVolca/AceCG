# AceCG/utils/mask.py
import numpy as np
import fnmatch, re
from typing import Dict, Tuple, Iterable, Optional, List
from ..potentials.base import BasePotential

from .ffio import FFParamIndexMap


def BuildGlobalMask(
    pair2potential: Dict[Tuple[str, str], BasePotential],
    init_mask: Optional[np.ndarray] = None,
    patterns: Optional[Dict[Tuple[str, str], Iterable[str]]] = None,
    mode: str = "freeze",
    strict: bool = False,
    case_sensitive: bool = True,
    global_patterns: Optional[Iterable[str]] = None,
    L0=False
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
    init_mask : np.ndarray of bool, optional
    	Initial mask to work on
		If None, then apply the default mask based on freeze/train mode
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

    l0_mask = [L0 for _ in range(len(pair2potential.items()))]
    if mode not in ("freeze","train"):
        raise ValueError(f"mode must be 'freeze' or 'train', got {mode}")

    default_train = (mode == "freeze")  # freeze mode: default True, train mode: default False
    mask = np.full(total, default_train, dtype=bool)
    
    if init_mask is not None:
        assert init_mask.shape == mask.shape
        mask = np.copy(init_mask)

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
    if L0:
        mask = np.append(mask, np.array(l0_mask))
    return mask


def DescribeMask(mask: np.ndarray, pair2potential: Dict[Tuple[str, str], BasePotential]) -> str:
    """
    Pretty-print a summary of which parameters are trainable vs frozen
    according to the given mask.

    Parameters
    ----------
    mask : np.ndarray of bool
        Global mask aligned with FFParamArray order (True=trainable, False=frozen).
    pair2potential : dict
        Mapping (type1,type2) -> BasePotential; must match order used in FFParamArray.

    Returns
    -------
    report : str
        A formatted string listing trainable/frozen parameters for each pair.
    """
    idx_map = FFParamIndexMap(pair2potential)  # [(pair, param_name), ...]
    assert len(mask) == len(idx_map), "Mask length mismatch."

    lines = []
    lines.append("=== Parameter Mask Summary ===")
    for (pair, pname), m in zip(idx_map, mask):
        status = "train" if m else "frozen"
        lines.append(f"{pair[0]}-{pair[1]:<4s} | {pname:<12s} : {status}")
    report = "\n".join(lines)
    return report