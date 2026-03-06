"""Shared FM infrastructure: topology serialization, masking, design-matrix construction."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from MDAnalysis import Universe

from .bonded_projectors import FMInteraction, build_design_matrix, interaction_offsets


def tqdm_enabled() -> bool:
    """Check whether tqdm progress bars are enabled via ACECG_TQDM env var."""
    flag = str(os.environ.get("ACECG_TQDM", "1")).strip().lower()
    return flag not in {"0", "false", "no", "off"}


def set_worker_blas_threads(n_threads: int) -> None:
    """Best-effort thread cap for BLAS/OpenMP inside worker processes."""
    nt = str(max(int(n_threads), 1))
    os.environ["OPENBLAS_NUM_THREADS"] = nt
    os.environ["OMP_NUM_THREADS"] = nt
    os.environ["MKL_NUM_THREADS"] = nt
    os.environ["NUMEXPR_NUM_THREADS"] = nt
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=int(nt))
    except Exception:
        pass


def collect_topology_arrays(u: Universe) -> Dict[str, Any]:
    """Serialize topology attributes from a Universe for passing to worker processes."""
    out: Dict[str, Any] = {}
    if hasattr(u.atoms, "types"):
        out["types"] = np.asarray(u.atoms.types).astype(str)
    if hasattr(u.atoms, "resids"):
        out["resids"] = np.asarray(u.atoms.resids).astype(np.int64)
    if hasattr(u, "bonds") and len(u.bonds) > 0:
        out["bonds"] = np.asarray(u.bonds.indices, dtype=np.int64)
    if hasattr(u, "angles") and len(u.angles) > 0:
        out["angles"] = np.asarray(u.angles.indices, dtype=np.int64)
    if hasattr(u, "dihedrals") and len(u.dihedrals) > 0:
        out["dihedrals"] = np.asarray(u.dihedrals.indices, dtype=np.int64)
    alias = getattr(u, "_acecg_type_alias", None)
    if isinstance(alias, dict):
        out["_acecg_type_alias"] = {str(k): str(v) for k, v in alias.items()}
    return out


def attach_topology_arrays(u: Universe, arrays: Mapping[str, Any]) -> None:
    """Restore topology attributes on a Universe from serialized arrays."""
    if "types" in arrays and (not hasattr(u.atoms, "types")):
        u.add_TopologyAttr("types", np.asarray(arrays["types"]).astype(str))
    if "resids" in arrays and (not hasattr(u.atoms, "resids")):
        u.add_TopologyAttr("resids", np.asarray(arrays["resids"]).astype(np.int64))

    if "bonds" in arrays and (not hasattr(u, "bonds") or len(u.bonds) == 0):
        arr = np.asarray(arrays["bonds"], dtype=np.int64)
        if arr.size > 0:
            u.add_TopologyAttr("bonds", arr)
    if "angles" in arrays and (not hasattr(u, "angles") or len(u.angles) == 0):
        arr = np.asarray(arrays["angles"], dtype=np.int64)
        if arr.size > 0:
            u.add_TopologyAttr("angles", arr)
    if "dihedrals" in arrays and (not hasattr(u, "dihedrals") or len(u.dihedrals) == 0):
        arr = np.asarray(arrays["dihedrals"], dtype=np.int64)
        if arr.size > 0:
            u.add_TopologyAttr("dihedrals", arr)
    alias = arrays.get("_acecg_type_alias")
    if isinstance(alias, dict):
        setattr(u, "_acecg_type_alias", {str(k): str(v) for k, v in alias.items()})


def normalize_diagonal(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Row normalization (X / row_norm) used in Bayesian ridge solve."""
    scale = np.sqrt(np.square(X).sum(axis=1))
    return X / scale, scale


def normalize_strict(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Numerically safer row normalization used in strict mode."""
    scale = np.sqrt(np.square(X).sum(axis=1))
    scale = np.where(scale > 1.0e-16, scale, 1.0)
    return X / scale[:, None], scale


class FMTraceMixin:
    """Mixin providing trace/diagnostic logging for FM components."""

    _trace_enabled: bool = False
    _trace_t0: float = 0.0

    def _init_trace(self) -> None:
        trace_flag = str(os.environ.get("ACECG_TRACE_STEPS", "0")).strip().lower()
        self._trace_enabled = trace_flag not in {"0", "false", "no", "off"}
        self._trace_t0 = time.perf_counter()

    def _trace(self, msg: str) -> None:
        if not self._trace_enabled:
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dt = time.perf_counter() - self._trace_t0
        print(f"[{now}   + {dt:8.2f} s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# FMComponentMixin (merged from fm_base.py)
# ---------------------------------------------------------------------------

class FMComponentMixin(FMTraceMixin):
    """Mixin providing shared FM masking, filtering, and design-matrix construction.

    Subclasses must set:
        self.u: Universe
        self.interactions: List[FMInteraction]
        self.cutoff: float
        self.exclude: Any
        self.sel: str
    """

    _style_mask: Dict[str, bool]
    _type_mask: Dict[str, set]
    _param_mask: np.ndarray
    _offsets: List[slice]
    _n_params: int

    def _init_fm_component(self, interactions: Sequence[FMInteraction]) -> None:
        """Initialize mask state and offsets from interactions list."""
        self._style_mask = {s: True for s in ["pair", "bond", "angle", "dihedral", "nb3b"]}
        self._type_mask = {}
        self._offsets = interaction_offsets(interactions)
        self._n_params = 0 if not self._offsets else self._offsets[-1].stop
        self._param_mask = np.ones(self._n_params, dtype=bool)
        self._init_trace()

    def set_style_mask(self, style_mask: Mapping[str, bool]) -> None:
        for k, v in style_mask.items():
            self._style_mask[str(k)] = bool(v)

    def set_type_mask(self, type_mask: Mapping[str, Iterable[Tuple[str, ...]]]) -> None:
        self._type_mask = {str(k): {tuple(t) for t in vals} for k, vals in type_mask.items()}

    def set_param_mask(self, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self._n_params,):
            raise ValueError(f"param mask must have shape ({self._n_params},)")
        self._param_mask = mask.copy()

    def _active_interactions(self) -> List[FMInteraction]:
        active: List[FMInteraction] = []
        for it in self.interactions:
            if not self._style_mask.get(it.style, True):
                continue
            allowed = self._type_mask.get(it.style)
            if allowed is not None and tuple(it.types) not in allowed:
                continue
            active.append(it)
        return active

    def _build_A(self, frame: int) -> np.ndarray:
        self.u.trajectory[frame]
        A = build_design_matrix(
            self.u,
            self._active_interactions(),
            cutoff=self.cutoff,
            exclude=self.exclude,
            sel=self.sel,
        )
        if not np.all(self._param_mask):
            A[:, ~self._param_mask] = 0.0
        return A
