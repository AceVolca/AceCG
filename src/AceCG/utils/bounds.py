# AceCG/utils/bounds.py
from typing import Dict, Tuple, Optional, List, TypeVar
import numpy as np
import fnmatch, re
from ..potentials.base import BasePotential
from .ffio import FFParamIndexMap

Pattern = str
Bound = Tuple[Optional[float], Optional[float]]  # (lb, ub), None is nonbounded

P = TypeVar("P", bound=BasePotential)

def BuildGlobalBounds(
    pair2potential: Dict[Tuple[str, str], P],
    pair_bounds: Optional[Dict[Tuple[str, str], Dict[Pattern, Bound]]] = None,
    global_bounds: Optional[Dict[Pattern, Bound]] = None,
    case_sensitive: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-index lower/upper bounds aligned with FFParamArray order,
    using parameter *names* from each potential's `param_names()`.

    Pattern syntax
    --------------
    • Exact: "A_0"
    • Glob : "A_*", "*r0*", "sigma_[0-3]"
    • Regex: prefix with "re:", e.g. "re:^A_\\d+$", "re:.*(r0|sigma).*"

    Precedence
    ----------
    For a given parameter (by name):
      global patterns apply first  → then pair-specific patterns override.
    If multiple patterns match, the later wins (last-write-wins).

    Returns
    -------
    lb, ub : np.ndarray
        Arrays of shape (n_params,) with +-inf for 'no bound'.
    """
    # build pair to global offset in L_array
    offsets: Dict[Tuple[str, str], int] = {}
    total = 0
    for pair, pot in pair2potential.items():
        offsets[pair] = total
        total += pot.n_params()

    lb = np.full(total, -np.inf, dtype=float)
    ub = np.full(total,  np.inf, dtype=float)

    def norm(s: str) -> str:
        return s if case_sensitive else s.lower()

    def iter_patterns(pats: Optional[Dict[Pattern, Bound]]):
        if not pats: return []
        return list(pats.items())

    for pair, pot in pair2potential.items():
        base = offsets[pair]
        names: List[str] = list(pot.param_names())
        names_norm = [norm(n) for n in names]

        # apply global bounds，then pair-specific bounds
        merged: List[Tuple[Pattern, Bound]] = []
        merged.extend(iter_patterns(global_bounds))
        if pair_bounds and pair in pair_bounds:
            merged.extend(iter_patterns(pair_bounds[pair]))

        for pat, (lo, hi) in merged:
            use_regex = pat.startswith("re:")
            body = pat[3:] if use_regex else pat

            if use_regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                rgx = re.compile(body, flags=flags)
                hits = [i for i, nm in enumerate(names) if rgx.search(nm)]
            else:
                body_cmp = norm(body)
                hits = [i for i, nm in enumerate(names_norm) if fnmatch.fnmatch(nm, body_cmp)]

            for i in hits:
                g = base + i
                if lo is not None: lb[g] = lo
                if hi is not None: ub[g] = hi

    return lb, ub


def DescribeBounds(
    lb: np.ndarray,
    ub: np.ndarray,
    pair2potential: Dict[Tuple[str, str], BasePotential],
    only_bounded: bool = False,
    precision: int = 6,
) -> str:
    """
    Pretty-print per-parameter lower/upper bounds aligned with FFParamArray order.

    Parameters
    ----------
    lb, ub : np.ndarray
        Lower/upper bounds arrays (shape = total #params). Use +/-inf for 'no bound'.
    pair2potential : dict[(type1,type2) -> BasePotential]
        Potentials used to build FFParamArray; iteration order must match.
    only_bounded : bool, default False
        If True, show only parameters that have at least one finite bound.
    precision : int, default 6
        Decimal places when formatting finite bounds.

    Returns
    -------
    report : str
        Text table listing bounds per (pair, param_name).
    """
    assert lb.shape == ub.shape, "lb/ub shape mismatch"
    idx_map = FFParamIndexMap(pair2potential)  # [(pair, pname), ...]
    assert len(idx_map) == lb.size, "Bounds length does not match parameter count"

    def fmt(v: float) -> str:
        if np.isneginf(v): return "-inf"
        if np.isposinf(v): return "+inf"
        return f"{v:.{precision}f}"

    lines = []
    lines.append("=== Parameter Bounds Summary ===")
    current_pair: Optional[Tuple[str, str]] = None

    for i, ((t1t2, pname), lo, hi) in enumerate(zip(idx_map, lb, ub)):
        bounded = np.isfinite(lo) or np.isfinite(hi)
        if only_bounded and not bounded:
            continue
        if t1t2 != current_pair:
            current_pair = t1t2
            lines.append(f"\nPair {current_pair[0]}-{current_pair[1]}")
            lines.append("-" * 40)
            lines.append(f"{'param':<16} {'lower':>12} {'upper':>12}")
        lines.append(f"{pname:<16} {fmt(lo):>12} {fmt(hi):>12}")

    if len(lines) == 1:
        lines.append("(no parameters to display)")
    return "\n".join(lines)