"""FM workflow utilities: config loading, interaction building, universe loading."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import MDAnalysis as mda
import numpy as np

from ..potentials.bspline import BSplinePotential
from ..utils.bonded_projectors import FMInteraction
from ..utils.topology_mscg import attach_topology_from_mscg_top


_STYLE_NTYPE = {
    "pair": 2,
    "bond": 2,
    "angle": 3,
    "dihedral": 4,
    "nb3b": 3,
}


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load an FM config JSON file."""
    p = Path(path)
    if p.suffix.lower() in {".json"}:
        return json.loads(p.read_text())
    raise ValueError(f"Unsupported config extension: {p.suffix}")



def build_bspline_potential(spec: Dict[str, Any]) -> BSplinePotential:
    """Build a BSplinePotential from an interaction spec dict."""
    types = tuple(str(x) for x in spec["types"])
    return BSplinePotential.from_order_spec(
        typ1=types[0],
        typ2=types[-1],
        minimum=float(spec["min"]),
        maximum=float(spec["max"]),
        resolution=float(spec["resolution"]),
        order=int(spec["order"]),
    )


def build_interactions(cfg: Dict[str, Any]) -> List[FMInteraction]:
    """Build FMInteraction list from config."""
    interactions: List[FMInteraction] = []
    for inter_spec in cfg["interactions"]:
        pot = build_bspline_potential(inter_spec)
        metadata = {k: v for k, v in inter_spec.items() if k not in {"style", "model", "types"}}
        interactions.append(
            FMInteraction(
                style=str(inter_spec["style"]),
                types=tuple(str(x) for x in inter_spec["types"]),
                potential=pot,
                metadata=metadata,
            )
        )
    return interactions


def load_window_universe(cfg: Dict[str, Any], window: int) -> mda.Universe:
    """Load an MDAnalysis Universe for a trajectory window from config."""
    traj_cfg = cfg["trajectory"]
    traj_pattern = str(traj_cfg["path_pattern"])
    traj_file = Path(traj_pattern.format(window=int(window)))
    if not traj_file.exists():
        raise FileNotFoundError(traj_file)

    fmt = str(traj_cfg.get("format", "LAMMPSDUMP"))
    topology_cfg = cfg.get("topology", {}) or {}
    lammps_data = topology_cfg.get("lammps_data")
    if lammps_data:
        data_file = Path(str(lammps_data))
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        u = mda.Universe(str(data_file), str(traj_file), format=fmt)
    else:
        u = mda.Universe(str(traj_file), format=fmt)
        warnings.warn(
            "No topology.lammps_data provided; MDAnalysis may guess masses (often 1.0). "
            "Provide topology.lammps_data in FM config to load per-type masses from LAMMPS data.",
            UserWarning,
            stacklevel=2,
        )

    top_path = topology_cfg.get("top_in")
    if top_path:
        attach_topology_from_mscg_top(u, top_path)
    return u


def frame_slice(cfg: Dict[str, Any], n_total: int) -> Tuple[int, int, int]:
    """Compute (start, end, every) from config and total frame count."""
    traj = cfg["trajectory"]
    start = int(traj.get("skip", 0))
    every = int(traj.get("every", 1))
    frames = int(traj.get("frames", 0))

    if frames <= 0:
        end = n_total
    else:
        end = min(n_total, start + frames * every)
    if start < 0 or start >= n_total:
        raise ValueError(f"Invalid start={start} for n_total={n_total}")
    if every <= 0:
        raise ValueError(f"Invalid every={every}")
    return start, end, every


def interaction_table_stem(style: str, types: Sequence[str]) -> str:
    """Generate LAMMPS table filename stem from interaction style and types."""
    t = "_".join(types)
    if style == "pair":
        return t
    if style == "bond":
        return f"{t}_bon"
    if style == "angle":
        return f"{t}_ang"
    if style == "dihedral":
        return f"{t}_dih"
    if style == "nb3b":
        return f"{t}_nb3b"
    raise ValueError(f"Unknown style: {style}")



def find_equilibrium(x: np.ndarray, force: np.ndarray) -> float:
    """Find equilibrium position (zero-crossing of force)."""
    x = np.asarray(x, dtype=float)
    f = np.asarray(force, dtype=float)
    for i in range(len(f) - 1):
        if f[i] >= 0.0 and f[i + 1] < 0.0:
            denom = f[i + 1] - f[i]
            if abs(denom) < 1.0e-15:
                return float(x[i])
            return float(x[i] - f[i] * (x[i + 1] - x[i]) / denom)
    return float(x[np.argmin(np.abs(f))])
