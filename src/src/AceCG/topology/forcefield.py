"""Canonical forcefield container: InteractionKey → List[BasePotential].

No I/O methods live here — ReadLmpFF / WriteLmpFF stay in ``io.forcefield``.
"""

from __future__ import annotations

import copy
import fnmatch
import re
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np

from ..potentials.base import BasePotential
from .types import InteractionKey

Pattern = str
Bound = Tuple[Optional[float], Optional[float]]
MaskPatternMap = Mapping[InteractionKey, Iterable[Pattern]]
BoundPatternMap = Mapping[InteractionKey, Mapping[Pattern, Bound]]


class Forcefield(MutableMapping):
    """Dict-like container: ``InteractionKey → List[BasePotential]``.

    Parameters
    ----------
    data : dict or Forcefield, optional
        Initial mapping from interaction keys to lists of potentials. Passing
        another ``Forcefield`` creates a copy constructor path.

    Notes
    -----
    The container caches the global parameter vector, masks, bounds, and
    per-potential slices so trainers can operate on a single flat vector.

    >>> ff = Forcefield({key: [pot]})
    >>> ff[key]               # list of potentials
    >>> ff.param_array()      # flat parameter vector
    """

    __slots__ = (
        "_data", "_param",
        "_param_mask", "_key_mask", "_param_bounds_lb", "_param_bounds_ub",
        "_vp_types", "_virtual_mask", "_real_mask", "_real_virtual_mask",
        "_param_slices_cache", "_param_blocks_cache",
    )

    _CACHE_FIELDS = (
        "_param", "_param_mask", "_key_mask",
        "_param_bounds_lb", "_param_bounds_ub",
        "_vp_types", "_virtual_mask", "_real_mask", "_real_virtual_mask",
    )

    # ==================================================================
    # Public interface
    # ==================================================================

    def __init__(self, data: Optional[Dict[InteractionKey, List[BasePotential]]] = None):
        """Create a forcefield container.

        Parameters
        ----------
        data : dict[InteractionKey, list[BasePotential]] or Forcefield, optional
            Source forcefield data. ``None`` creates an empty container; a dict
            is deep-copied; a ``Forcefield`` uses the copy-constructor path.

        >>> ff = Forcefield({key: [pot1, pot2]})
        >>> ff2 = Forcefield(ff)
        """
        if data is None:
            self._data: Dict[InteractionKey, List[BasePotential]] = {}
        elif isinstance(data, Forcefield):
            self._data = dict(data._data)
            for f in self._CACHE_FIELDS:
                setattr(self, f, self._copy_val(getattr(data, f)))
            self._rebuild_structure_cache()
            return
        elif isinstance(data, dict):
            self._data = copy.deepcopy(data)
        else:
            raise TypeError(f"Expected dict or Forcefield, got {type(data).__name__}")
        for f in self._CACHE_FIELDS:
            setattr(self, f, None)
        self._rebuild_param_cache()
        self._rebuild_structure_cache()
        self._auto_detect_bounds()

    def __getitem__(self, key: InteractionKey) -> List[BasePotential]:
        """``ff[key]`` → list of potentials for *key*."""
        return self._data[key]

    def __setitem__(self, key: InteractionKey, value: List[BasePotential]) -> None:
        """``ff[key] = [pot]`` — insert or replace, incrementally updating caches."""
        if key in self._data:
            old_start, old_stop = self._key_param_range(key)
            self._data[key] = value
            new_params = self._collect_key_params(key)
            self._splice_caches(old_start, old_stop, new_params, key)
        else:
            insert_at = self.n_params()
            self._data[key] = value
            new_params = self._collect_key_params(key)
            self._splice_caches(insert_at, insert_at, new_params, key)
            if self._key_mask is not None:
                self._key_mask[key] = True

    def __delitem__(self, key: InteractionKey) -> None:
        """``del ff[key]`` — remove key and splice out its cache block."""
        start, stop = self._key_param_range(key)
        del self._data[key]
        self._splice_caches(start, stop, np.empty(0, dtype=np.float64), None)
        if self._key_mask is not None:
            self._key_mask.pop(key, None)

    def __iter__(self) -> Iterator[InteractionKey]:
        """Iterate over interaction keys.  ``for key in ff: ...``"""
        return iter(self._data)

    def __len__(self) -> int:
        """``len(ff)`` → number of interaction keys."""
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        """``key in ff`` → True if key is present."""
        return key in self._data

    def __bool__(self) -> bool:
        """``bool(ff)`` → False when empty."""
        return bool(self._data)

    def __repr__(self) -> str:
        return f"Forcefield({self._data!r})"

    def __copy__(self):
        """Support ``copy.copy(ff)`` — shallow copy (potentials are shared)."""
        return Forcefield(self)

    def __deepcopy__(self, memo):
        """Support ``copy.deepcopy(ff)`` — full deep copy."""
        new = object.__new__(Forcefield)
        memo[id(self)] = new
        new._data = copy.deepcopy(self._data, memo)
        for f in self._CACHE_FIELDS:
            setattr(new, f, self._copy_val(getattr(self, f)))
        new._rebuild_structure_cache()
        return new

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_potentials(self) -> Generator[Tuple[InteractionKey, BasePotential], None, None]:
        """Yield ``(key, potential)`` pairs, flattening list values.

        >>> for key, pot in ff.iter_potentials(): ...
        """
        for key, val in self._data.items():
            if isinstance(val, list):
                for pot in val:
                    yield key, pot
            else:
                yield key, val

    # ------------------------------------------------------------------
    # Parameter vector
    # ------------------------------------------------------------------

    def n_params(self) -> int:
        """Total scalar parameter count.  ``ff.n_params()`` → int."""
        if self._param is not None:
            return int(self._param.size)
        return sum(p.n_params() for _, p in self.iter_potentials())

    def param_array(self) -> np.ndarray:
        """Flat 1-D copy of all parameters.  ``ff.param_array()`` → ndarray."""
        if self._param is None:
            self._rebuild_param_cache()
        return self._param.copy()

    def param_index_map(self) -> List[Tuple[InteractionKey, str]]:
        """Map each parameter index to ``(key, param_name)``.

        >>> ff.param_index_map()  # [(key, 'epsilon'), (key, 'sigma'), ...]
        """
        index_map: List[Tuple[InteractionKey, str]] = []
        for key, pot in self.iter_potentials():
            for name in pot.param_names():
                index_map.append((key, name))
        return index_map

    def param_slices(self) -> List[Tuple[InteractionKey, int, slice]]:
        """Return ``(key, pot_index, slice)`` per potential in param-vector order.

        >>> for key, pi, sl in ff.param_slices(): params[sl]
        """
        return list(self._param_slices_cache)

    def param_blocks(self) -> Tuple[Tuple[InteractionKey, BasePotential, slice], ...]:
        """Return cached ``(key, potential, slice)`` blocks in param-vector order."""
        return self._param_blocks_cache

    def interaction_offsets(self) -> List[slice]:
        """Return a parameter ``slice`` for each potential (flattened).

        >>> offsets = ff.interaction_offsets()  # [slice(0,3), slice(3,6), ...]
        """
        return [sl for _, _, sl in self.param_blocks()]

    def update_params(self, L: np.ndarray) -> None:
        """Write a full parameter vector back into all potentials.

        Parameters
        ----------
        L : np.ndarray
            One-dimensional global parameter vector with length
            :meth:`n_params`.

        >>> ff.update_params(new_theta)
        """
        L = np.asarray(L, dtype=np.float64).reshape(-1)
        n_total = self.n_params()
        if L.shape != (n_total,):
            raise ValueError(f"Parameter vector shape must be ({n_total},), got {L.shape}")
        idx = 0
        for _, pot in self.iter_potentials():
            n = pot.n_params()
            pot.set_params(L[idx : idx + n])
            idx += n
        self._param = L.copy()

    # ------------------------------------------------------------------
    # Masks (L2 per-parameter & L1 per-key)
    # ------------------------------------------------------------------

    @property
    def param_mask(self) -> np.ndarray:
        """L2 bool mask ``(n_params,)``.  ``True`` = trainable.  Defaults all-True.

        >>> ff.param_mask = np.array([True, True, False])
        """
        n = self.n_params()
        if self._param_mask is not None and self._param_mask.shape == (n,):
            return self._param_mask
        return np.ones(n, dtype=bool)

    @param_mask.setter
    def param_mask(self, mask: np.ndarray) -> None:
        """Set the parameter-level trainability mask."""
        mask = np.asarray(mask, dtype=bool)
        n = self.n_params()
        if mask.shape != (n,):
            raise ValueError(f"param_mask shape must be ({n},), got {mask.shape}")
        self._param_mask = mask.copy()
        self._key_mask = self._derive_key_mask_from_l2(mask)

    @property
    def key_mask(self) -> Dict[InteractionKey, bool]:
        """L1 per-key mask ``{key: bool}``.  Synced bidirectionally with ``param_mask``.

        >>> ff.key_mask = {bond_key: True, angle_key: False}
        """
        if self._key_mask is not None and set(self._key_mask) == set(self._data):
            return self._key_mask
        self._key_mask = self._derive_key_mask_from_l2(self.param_mask)
        return self._key_mask

    @key_mask.setter
    def key_mask(self, km: Mapping[InteractionKey, bool]) -> None:
        """Set the interaction-level mask and propagate it to parameters."""
        self._validate_interaction_keyed_mapping("key_mask", km)
        n = self.n_params()
        mask = self._param_mask if self._param_mask is not None else np.ones(n, dtype=bool)
        mask = mask.copy()
        offset = 0
        for key, val in self._data.items():
            pots = val if isinstance(val, list) else [val]
            block_n = sum(p.n_params() for p in pots)
            if key in km:
                mask[offset:offset + block_n] = km[key]
            offset += block_n
        self.param_mask = mask

    def build_mask(
        self,
        init_mask: Optional[np.ndarray] = None,
        patterns: Optional[MaskPatternMap] = None,
        mode: str = "freeze",
        strict: bool = False,
        case_sensitive: bool = True,
        global_patterns: Optional[Iterable[str]] = None,
    ) -> np.ndarray:
        """Build and store a parameter-level mask from name patterns.

        Parameters
        ----------
        init_mask : np.ndarray, optional
            Starting mask. If omitted, the initial mask is all trainable for
            ``mode="freeze"`` and all frozen for ``mode="train"``.
        patterns : Mapping[InteractionKey, Iterable[str]], optional
            Per-interaction parameter name patterns. Glob patterns are used by
            default; prefix a pattern with ``"re:"`` to use a regular
            expression.
        mode : {"freeze", "train"}, default="freeze"
            Whether matching parameters should be frozen or trained.
        strict : bool, default=False
            If ``True``, raise when a pattern matches no parameters.
        case_sensitive : bool, default=True
            Whether glob/regex matching is case-sensitive.
        global_patterns : Iterable[str], optional
            Patterns applied to every interaction.

        Returns
        -------
        np.ndarray
            Boolean mask ordered like :meth:`param_array`.

        ``patterns`` must be keyed by ``InteractionKey``.

        >>> ff.build_mask(mode="freeze", global_patterns=["*k*"])
        """
        pair_offsets = {}
        total = 0
        for key, pot in self.iter_potentials():
            pair_offsets[id(pot)] = (key, total)
            total += pot.n_params()

        if mode not in ("freeze", "train"):
            raise ValueError(f"mode must be 'freeze' or 'train', got {mode}")

        default_train = mode == "freeze"
        mask = np.full(total, default_train, dtype=bool)

        if init_mask is not None:
            if init_mask.shape != mask.shape:
                raise ValueError("init_mask shape mismatch")
            mask = np.copy(init_mask)
        if patterns is not None:
            self._validate_interaction_keyed_mapping("patterns", patterns)

        def maybe_norm(value: str) -> str:
            return value if case_sensitive else value.lower()

        for key, pot in self.iter_potentials():
            base = pair_offsets[id(pot)][1]
            local_names = list(pot.param_names())
            norm_names = [maybe_norm(name) for name in local_names]

            pair_patterns: List[str] = []
            if global_patterns:
                pair_patterns.extend(global_patterns)
            if patterns and key in patterns:
                pair_patterns.extend(patterns[key])

            for pattern in pair_patterns:
                use_regex = pattern.startswith("re:")
                pattern_body = pattern[3:] if use_regex else pattern
                if use_regex:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    regex = re.compile(pattern_body, flags=flags)
                    hits = [i for i, name in enumerate(local_names) if regex.search(name)]
                else:
                    pattern_cmp = maybe_norm(pattern_body)
                    hits = [i for i, name in enumerate(norm_names) if fnmatch.fnmatch(name, pattern_cmp)]

                if not hits and strict:
                    raise KeyError(
                        f"No params matched pattern '{pattern}' for key {key}. "
                        f"Available: {local_names}"
                    )

                for local_idx in hits:
                    global_idx = base + local_idx
                    mask[global_idx] = mode == "train"

        self.param_mask = mask
        return mask

    def derive_l1_mask(self, l2_mask: Optional[np.ndarray] = None) -> Dict[InteractionKey, bool]:
        """Derive L1 key mask from an L2 mask.  No-arg returns cached ``key_mask``.

        >>> l1 = ff.derive_l1_mask()  # {key: True/False, ...}
        """
        if l2_mask is None:
            return dict(self.key_mask)
        return self._derive_key_mask_from_l2(l2_mask)

    def describe_mask(self, mask: Optional[np.ndarray] = None) -> str:
        """Pretty-print trainable vs frozen parameters.

        >>> print(ff.describe_mask())
        """
        if mask is None:
            mask = self.param_mask
        idx_map = self.param_index_map()
        if len(mask) != len(idx_map):
            raise ValueError("Mask length mismatch")
        lines = ["=== Parameter Mask Summary ==="]
        for (key, param_name), is_trainable in zip(idx_map, mask):
            key_str = key.label() if hasattr(key, "label") else str(key)
            status = "train" if is_trainable else "frozen"
            lines.append(f"{key_str:<10s} | {param_name:<12s} : {status}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # VP masks
    # ------------------------------------------------------------------

    def set_vp_masks(self, vp_types: Iterable[str]) -> None:
        """Classify parameters as virtual, real, or mixed.

        Parameters
        ----------
        vp_types : Iterable[str]
            Type labels treated as virtual-particle types. Each interaction is
            classified by how many of its type labels appear in this set.

        >>> ff.set_vp_masks(["VP"])
        >>> ff.virtual_mask  # True where all types are VP
        """
        vp_set = frozenset(vp_types)
        self._vp_types = vp_set
        n = self.n_params()
        virtual = np.zeros(n, dtype=bool)
        real = np.zeros(n, dtype=bool)
        mixed = np.zeros(n, dtype=bool)
        offset = 0
        for key, pot in self.iter_potentials():
            np_ = pot.n_params()
            n_vp = sum(1 for t in key.types if t in vp_set)
            if n_vp == len(key.types):
                virtual[offset:offset + np_] = True
            elif n_vp == 0:
                real[offset:offset + np_] = True
            else:
                mixed[offset:offset + np_] = True
            offset += np_
        self._virtual_mask = virtual
        self._real_mask = real
        self._real_virtual_mask = mixed

    @property
    def vp_types(self) -> Optional[frozenset]:
        """The VP type set passed to ``set_vp_masks()``, or ``None``."""
        return self._vp_types

    @property
    def virtual_mask(self) -> np.ndarray:
        """Bool mask — True for VP-only keys.  ``ff.virtual_mask`` → ndarray."""
        if self._virtual_mask is not None and self._virtual_mask.shape == (self.n_params(),):
            return self._virtual_mask
        return np.zeros(self.n_params(), dtype=bool)

    @property
    def real_mask(self) -> np.ndarray:
        """Bool mask — True for real-only keys.  ``ff.real_mask`` → ndarray."""
        if self._real_mask is not None and self._real_mask.shape == (self.n_params(),):
            return self._real_mask
        return np.ones(self.n_params(), dtype=bool)

    @property
    def real_virtual_mask(self) -> np.ndarray:
        """Bool mask — True for mixed (cross-term) keys."""
        if self._real_virtual_mask is not None and self._real_virtual_mask.shape == (self.n_params(),):
            return self._real_virtual_mask
        return np.zeros(self.n_params(), dtype=bool)

    @property
    def direct_active_mask(self) -> np.ndarray:
        """``~virtual_mask``: True for directly observable params."""
        return ~self.virtual_mask

    # ------------------------------------------------------------------
    # Parameter bounds
    # ------------------------------------------------------------------

    @property
    def param_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(lb, ub)`` arrays, shape ``(n_params,)``.  Defaults ±inf.

        >>> lb, ub = ff.param_bounds
        """
        n = self.n_params()
        lb = self._param_bounds_lb if self._param_bounds_lb is not None and self._param_bounds_lb.shape == (n,) else np.full(n, -np.inf)
        ub = self._param_bounds_ub if self._param_bounds_ub is not None and self._param_bounds_ub.shape == (n,) else np.full(n, np.inf)
        return lb, ub

    @param_bounds.setter
    def param_bounds(self, bounds: Tuple[np.ndarray, np.ndarray]) -> None:
        """Set lower and upper bound arrays for the full parameter vector."""
        lb, ub = bounds
        lb = np.asarray(lb, dtype=float)
        ub = np.asarray(ub, dtype=float)
        n = self.n_params()
        if lb.shape != (n,) or ub.shape != (n,):
            raise ValueError(f"param_bounds shape must be ({n},), got lb={lb.shape}, ub={ub.shape}")
        self._param_bounds_lb = lb.copy()
        self._param_bounds_ub = ub.copy()

    def apply_bounds(self, L: np.ndarray) -> np.ndarray:
        """Clamp a parameter vector into stored bounds.

        Parameters
        ----------
        L : np.ndarray
            Parameter vector ordered like :meth:`param_array`.

        Returns
        -------
        np.ndarray
            ``L`` clipped elementwise into ``[lower_bounds, upper_bounds]``.

        >>> L_clamped = ff.apply_bounds(L)
        """
        if self._param_bounds_lb is None and self._param_bounds_ub is None:
            return L
        Lc = L.copy()
        if self._param_bounds_lb is not None:
            Lc = np.maximum(Lc, self._param_bounds_lb)
        if self._param_bounds_ub is not None:
            Lc = np.minimum(Lc, self._param_bounds_ub)
        return Lc

    def build_bounds(
        self,
        pair_bounds: Optional[BoundPatternMap] = None,
        global_bounds: Optional[Dict[Pattern, Bound]] = None,
        case_sensitive: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build and store per-parameter bounds through name patterns.

        Parameters
        ----------
        pair_bounds : Mapping[InteractionKey, Mapping[str, tuple]], optional
            Per-interaction bounds. Inner keys are glob patterns or ``"re:"``
            regular expressions; values are ``(lower, upper)`` where ``None``
            means unbounded on that side.
        global_bounds : dict[str, tuple], optional
            Bounds patterns applied to every interaction.
        case_sensitive : bool, default=True
            Whether pattern matching is case-sensitive.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Lower and upper bound arrays ordered like :meth:`param_array`.

        ``pair_bounds`` must be keyed by ``InteractionKey``.

        >>> lb, ub = ff.build_bounds(global_bounds={"*k*": (0, None)})
        """
        offsets = {}
        total = 0
        for key, pot in self.iter_potentials():
            offsets[id(pot)] = (key, total)
            total += pot.n_params()
        if pair_bounds is not None:
            self._validate_interaction_keyed_mapping("pair_bounds", pair_bounds)

        lb = np.full(total, -np.inf, dtype=float)
        ub = np.full(total, np.inf, dtype=float)

        def norm(value: str) -> str:
            return value if case_sensitive else value.lower()

        def iter_patterns(pats: Optional[Mapping[Pattern, Bound]]):
            return [] if not pats else list(pats.items())

        for key, pot in self.iter_potentials():
            base = offsets[id(pot)][1]
            names: List[str] = list(pot.param_names())
            names_norm = [norm(name) for name in names]

            merged: List[Tuple[Pattern, Bound]] = []
            merged.extend(iter_patterns(global_bounds))
            if pair_bounds and key in pair_bounds:
                merged.extend(iter_patterns(pair_bounds[key]))

            for pattern, (lo, hi) in merged:
                use_regex = pattern.startswith("re:")
                body = pattern[3:] if use_regex else pattern
                if use_regex:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    regex = re.compile(body, flags=flags)
                    hits = [i for i, name in enumerate(names) if regex.search(name)]
                else:
                    body_cmp = norm(body)
                    hits = [i for i, name in enumerate(names_norm) if fnmatch.fnmatch(name, body_cmp)]

                for idx in hits:
                    global_idx = base + idx
                    if lo is not None:
                        lb[global_idx] = lo
                    if hi is not None:
                        ub[global_idx] = hi

        self.param_bounds = (lb, ub)
        return lb, ub

    def describe_bounds(
        self,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        only_bounded: bool = False,
        precision: int = 6,
    ) -> str:
        """Pretty-print per-parameter bounds.

        >>> print(ff.describe_bounds(only_bounded=True))
        """
        if lb is None or ub is None:
            lb, ub = self.param_bounds
        if lb.shape != ub.shape:
            raise ValueError("lb/ub shape mismatch")
        idx_map = self.param_index_map()
        if len(idx_map) != lb.size:
            raise ValueError("Bounds length does not match parameter count")

        def fmt(value: float) -> str:
            if np.isneginf(value):
                return "-inf"
            if np.isposinf(value):
                return "+inf"
            return f"{value:.{precision}f}"

        lines = ["=== Parameter Bounds Summary ==="]
        current_key = None
        for (key, param_name), lo, hi in zip(idx_map, lb, ub):
            bounded = np.isfinite(lo) or np.isfinite(hi)
            if only_bounded and not bounded:
                continue
            if key != current_key:
                current_key = key
                key_str = key.label() if hasattr(key, "label") else str(key)
                lines.append(f"\nKey {key_str}")
                lines.append("-" * 40)
                lines.append(f"{'param':<16} {'lower':>12} {'upper':>12}")
            lines.append(f"{param_name:<16} {fmt(lo):>12} {fmt(hi):>12}")
        if len(lines) == 1:
            lines.append("(no parameters to display)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Deep copy
    # ------------------------------------------------------------------

    def deepcopy(self) -> Forcefield:
        """Deep-copy all potentials and caches.  ``ff2 = ff.deepcopy()``."""
        out = Forcefield(copy.deepcopy(self._data))
        for f in self._CACHE_FIELDS:
            setattr(out, f, self._copy_val(getattr(self, f)))
        out._rebuild_structure_cache()
        return out

    # ==================================================================
    # Private helpers
    # ==================================================================

    @staticmethod
    def _copy_val(v):
        """Copy a cache value: ndarray→.copy(), dict→dict(), else passthrough."""
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            return v.copy()
        if isinstance(v, dict):
            return dict(v)
        return v

    @staticmethod
    def _validate_interaction_keyed_mapping(name: str, mapping: Mapping[object, object]) -> None:
        """Raise when a selector dict is keyed by legacy tuples instead of InteractionKey."""
        for key in mapping:
            if not isinstance(key, InteractionKey):
                raise TypeError(
                    f"{name} must be keyed by InteractionKey, got {type(key).__name__}: {key!r}"
                )

    def _key_param_range(self, target_key: InteractionKey) -> Tuple[int, int]:
        """Return ``(start, stop)`` for *target_key*'s block in the param vector."""
        offset = 0
        for key, val in self._data.items():
            pots = val if isinstance(val, list) else [val]
            n = sum(p.n_params() for p in pots)
            if key == target_key:
                return offset, offset + n
            offset += n
        raise KeyError(target_key)

    def _collect_key_params(self, key: InteractionKey) -> np.ndarray:
        """Concatenate params for *key* from its current potentials."""
        val = self._data[key]
        pots = val if isinstance(val, list) else [val]
        parts = [p.get_params() for p in pots]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)

    def _splice_caches(
        self,
        old_start: int,
        old_stop: int,
        new_params: np.ndarray,
        key: Optional[InteractionKey],
    ) -> None:
        """Replace ``[old_start:old_stop)`` in every cached array with *new_params*."""
        new_n = len(new_params)
        idx = np.arange(old_start, old_stop)

        def _splice(arr: Optional[np.ndarray], fill) -> Optional[np.ndarray]:
            if arr is None:
                return None
            arr = np.delete(arr, idx)
            if new_n > 0:
                arr = np.insert(arr, old_start, fill)
            return arr

        self._param = _splice(self._param, new_params)
        self._param_mask = _splice(self._param_mask, np.ones(new_n, dtype=bool))
        self._param_bounds_lb = _splice(self._param_bounds_lb, np.full(new_n, -np.inf))
        self._param_bounds_ub = _splice(self._param_bounds_ub, np.full(new_n, np.inf))

        if key is not None and new_n > 0:
            val = self._data[key]
            pots = val if isinstance(val, list) else [val]
            offset = 0
            intrinsic_bounds: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
            has_finite_intrinsic = False
            for pot in pots:
                pn = pot.n_params()
                if hasattr(pot, "param_bounds"):
                    pot_lb, pot_ub = pot.param_bounds()
                    pot_lb = np.asarray(pot_lb, dtype=float)
                    pot_ub = np.asarray(pot_ub, dtype=float)
                    if pot_lb.shape != (pn,) or pot_ub.shape != (pn,):
                        raise ValueError(
                            f"param_bounds shape mismatch for {type(pot).__name__}: "
                            f"expected {(pn,)}, got lb={pot_lb.shape}, ub={pot_ub.shape}"
                        )
                    intrinsic_bounds.append((offset, pn, pot_lb, pot_ub))
                    if np.any(np.isfinite(pot_lb)) or np.any(np.isfinite(pot_ub)):
                        has_finite_intrinsic = True
                offset += pn

            if has_finite_intrinsic and (self._param_bounds_lb is None or self._param_bounds_ub is None):
                n_total = self.n_params()
                if self._param_bounds_lb is None:
                    self._param_bounds_lb = np.full(n_total, -np.inf, dtype=float)
                if self._param_bounds_ub is None:
                    self._param_bounds_ub = np.full(n_total, np.inf, dtype=float)

            for offset, pn, pot_lb, pot_ub in intrinsic_bounds:
                if self._param_bounds_lb is not None:
                    self._param_bounds_lb[old_start + offset:old_start + offset + pn] = pot_lb
                if self._param_bounds_ub is not None:
                    self._param_bounds_ub[old_start + offset:old_start + offset + pn] = pot_ub

        if self._vp_types is not None and key is not None and new_n > 0:
            vp_set = self._vp_types
            n_vp = sum(1 for t in key.types if t in vp_set)
            virt_fill = np.full(new_n, n_vp == len(key.types), dtype=bool)
            real_fill = np.full(new_n, n_vp == 0, dtype=bool)
            mixed_fill = np.full(new_n, 0 < n_vp < len(key.types), dtype=bool)
        else:
            virt_fill = np.zeros(max(new_n, 0), dtype=bool)
            real_fill = np.ones(max(new_n, 0), dtype=bool)
            mixed_fill = np.zeros(max(new_n, 0), dtype=bool)
        self._virtual_mask = _splice(self._virtual_mask, virt_fill)
        self._real_mask = _splice(self._real_mask, real_fill)
        self._real_virtual_mask = _splice(self._real_virtual_mask, mixed_fill)
        if self._param_mask is not None or self._key_mask is not None:
            self._key_mask = self._derive_key_mask_from_l2(self.param_mask)
        self._rebuild_structure_cache()

    def _rebuild_param_cache(self) -> None:
        """Rebuild ``_param`` from scratch (used on init)."""
        if not self._data:
            self._param = np.empty(0, dtype=np.float64)
            return
        parts = [pot.get_params() for _, pot in self.iter_potentials()]
        self._param = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)

    def _rebuild_structure_cache(self) -> None:
        """Rebuild cached parameter slice and block metadata."""
        blocks: List[Tuple[InteractionKey, BasePotential, slice]] = []
        offset = 0
        for key, pot in self.iter_potentials():
            n = pot.n_params()
            sl = slice(offset, offset + n)
            blocks.append((key, pot, sl))
            offset += n
        pot_index_by_key: Dict[InteractionKey, int] = {}
        slices: List[Tuple[InteractionKey, int, slice]] = []
        for key, pot, sl in blocks:
            pi = pot_index_by_key.get(key, 0)
            slices.append((key, pi, sl))
            pot_index_by_key[key] = pi + 1
        self._param_blocks_cache = tuple(blocks)
        self._param_slices_cache = tuple(slices)

    def _derive_key_mask_from_l2(self, l2: np.ndarray) -> Dict[InteractionKey, bool]:
        """Derive L1 key mask from an L2 per-parameter mask."""
        l2 = np.asarray(l2, dtype=bool).reshape(-1)
        n_total = self.n_params()
        if l2.shape != (n_total,):
            raise ValueError(f"L2 mask shape must be ({n_total},), got {l2.shape}")
        km: Dict[InteractionKey, bool] = {}
        offset = 0
        for key, val in self._data.items():
            pots = val if isinstance(val, list) else [val]
            block_n = sum(p.n_params() for p in pots)
            km[key] = bool(np.any(l2[offset:offset + block_n]))
            offset += block_n
        return km

    def _auto_detect_bounds(self) -> None:
        """Scan potentials for intrinsic ``param_bounds()`` and store them."""
        lb_parts: List[np.ndarray] = []
        ub_parts: List[np.ndarray] = []
        has_finite = False
        for _, pot in self.iter_potentials():
            n = pot.n_params()
            if hasattr(pot, "param_bounds"):
                pot_lb, pot_ub = pot.param_bounds()
                pot_lb = np.asarray(pot_lb, dtype=float)
                pot_ub = np.asarray(pot_ub, dtype=float)
                lb_parts.append(pot_lb)
                ub_parts.append(pot_ub)
                if np.any(np.isfinite(pot_lb)) or np.any(np.isfinite(pot_ub)):
                    has_finite = True
            else:
                lb_parts.append(np.full(n, -np.inf))
                ub_parts.append(np.full(n, np.inf))
        if has_finite:
            self._param_bounds_lb = np.concatenate(lb_parts)
            self._param_bounds_ub = np.concatenate(ub_parts)
