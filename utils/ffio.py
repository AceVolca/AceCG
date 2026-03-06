# AceCG/utils/ffio.py
"""Force field I/O: read/write LAMMPS force fields and FM B-spline table export."""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List, Sequence

from ..potentials.base import BasePotential
from ..potentials.gaussian import GaussianPotential
from ..potentials.lennardjones import LennardJonesPotential
from ..potentials.lennardjones96 import LennardJones96Potential
from ..potentials.lennardjones_soft import LennardJonesSoftPotential
from ..potentials.multi_gaussian import MultiGaussianPotential
from ..potentials.srlrgaussian import SRLRGaussianPotential
from ..potentials import POTENTIAL_REGISTRY

from ..fitters.base import TABLE_FITTERS


# ---------------------------------------------------------------------------
# Parameter flattening
# ---------------------------------------------------------------------------

def FFParamArray(pair2potential: Dict[Tuple[str, str], BasePotential]) -> np.ndarray:
    """Concatenate all potential parameters into a single 1D NumPy array."""
    return np.concatenate([pot.get_params() for pot in pair2potential.values()])


def FFParamIndexMap(pair2potential: Dict[Tuple[str, str], BasePotential]) -> List[Tuple[Tuple[str, str], str]]:
    """Create a map from parameter index to (pair_type, param_name)."""
    index_map = []
    for pair, pot in pair2potential.items():
        for name in pot.param_names():
            index_map.append((pair, name))
    return index_map


# ---------------------------------------------------------------------------
# LAMMPS table parsing
# ---------------------------------------------------------------------------

def _parse_lmp_table(table_path: str):
    """Read LAMMPS pair_style table file (internal).

    Returns (r, V, F) where F may be None if not present.
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
                    r_val = float(parts[1]); v_val = float(parts[2]); f_val = float(parts[3])
                    r_list.append(r_val); v_list.append(v_val); f_list.append(f_val)
                elif len(parts) == 3:    # idx r V   or   r V F
                    try:
                        r_val = float(parts[1]); v_val = float(parts[2])
                        r_list.append(r_val); v_list.append(v_val)
                    except Exception:
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


# ---------------------------------------------------------------------------
# LAMMPS table writing
# ---------------------------------------------------------------------------

def _write_lmp_table(
    filename: str,
    r: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    comment: str = "LAMMPS Table written by AceCG",
    table_name: str = "Table1"
):
    """Write a LAMMPS-style pair_style table file (internal)."""
    r = np.asarray(r, dtype=float)
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=float)
    assert r.shape == V.shape == F.shape, "r, V, F must have the same shape"

    with open(filename, "w") as f:
        if comment is not None:
            for line in comment.splitlines():
                f.write(f"# {line}\n")

        npoints = len(r)
        f.write(f"\n{table_name}\n")
        f.write(f"N {npoints} R {r[0]:.6f} {r[-1]:.6f}\n\n")

        for i, (ri, vi, fi) in enumerate(zip(r, V, F), start=1):
            f.write(f"{i:6d}  {ri:16.8f}  {vi:16.8e}  {fi:16.8e}\n")


def _write_lmp_table_bundle(
    outdir: str,
    tables: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """Write a set of LAMMPS tables in one call (internal)."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    files: Dict[str, str] = {}
    for stem, item in tables.items():
        table_file = out_path / f"{stem}.table"
        _write_lmp_table(
            filename=str(table_file),
            r=np.asarray(item["r"], dtype=float),
            V=np.asarray(item["V"], dtype=float),
            F=np.asarray(item["F"], dtype=float),
            comment=str(item.get("comment", "LAMMPS Table written by AceCG")),
            table_name=str(item.get("table_name", stem)),
        )
        files[str(stem)] = str(table_file)

    return files


# ---------------------------------------------------------------------------
# LAMMPS force field read/write (analytic + table)
# ---------------------------------------------------------------------------

def ReadLmpFF(
        file: str,
        pair_style: str,
        pair_typ_sel: Optional[List[str]] = None,
        cutoff: Optional[int] = None,
        global_var: Optional[dict] = None,
        table_fit: str = "multigaussian",
        table_fit_overrides: Optional[dict] = None,
) -> Dict[Tuple[str, str], BasePotential]:
    """Generalized reader for LAMMPS force field files using registered potential types."""
    assert pair_style is not None
    if pair_style != "hybrid":
        pair_typ_sel = None
        param_offset = 3
    else:
        param_offset = 4

    base_dir = os.path.dirname(os.path.abspath(file))
    pair2potential: Dict[Tuple[str, str], BasePotential] = {}

    with open(file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if "pair_coeff" in line:
            tmp = line.split()
            if pair_style == "hybrid":
                style = tmp[3]
            else:
                style = pair_style
            pair = (tmp[1], tmp[2])

            if pair_typ_sel is None or style in pair_typ_sel:
                if style == "table":
                    table_file = tmp[param_offset]
                    if not os.path.isabs(table_file):
                        table_file = os.path.join(base_dir, table_file)
                    fitter = TABLE_FITTERS.create(table_fit, **(table_fit_overrides or {}))
                    pot = fitter.fit(table_file, typ1=pair[0], typ2=pair[1])
                    pair2potential[pair] = pot
                else:
                    if style in POTENTIAL_REGISTRY:
                        constructor = POTENTIAL_REGISTRY[style]
                        params = list(map(float, tmp[param_offset:]))
                        if style == "double/gauss":
                            if cutoff is None: gauss_params, cutoff = params[:-1], params[-1]
                            else: gauss_params = params[:]
                            pair2potential[pair] = constructor(pair[0], pair[1], 2, cutoff, gauss_params)
                        elif style == "lj/cut/soft":
                            if cutoff is not None:
                                params.append(cutoff)
                            params.append(global_var["n"])
                            params.append(global_var["alpha"])
                            pair2potential[pair] = constructor(pair[0], pair[1], *params)
                        else:
                            if cutoff is not None:
                                params.append(cutoff)
                            pair2potential[pair] = constructor(pair[0], pair[1], *params)
    return pair2potential


def WriteLmpFF(
    old_file: str,
    new_file: str,
    pair2potential: Dict[Tuple[str, str], BasePotential],
    pair_style: str,
    pair_typ_sel: Optional[List[str]] = None
):
    """Write updated parameters to a new LAMMPS-style force field file."""
    assert pair_style is not None
    if pair_style != "hybrid":
        pair_typ_sel = None
        param_offset = 3
    else:
        param_offset = 4

    base_dir = os.path.dirname(os.path.abspath(old_file))

    L_new = FFParamArray(pair2potential)
    idx = 0
    with open(old_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "pair_coeff" in line:
            tmp = line.split()
            if pair_style == "hybrid":
                style = tmp[3]
            else:
                style = pair_style
            pair = (tmp[1], tmp[2])

            if pair_typ_sel is None or style in pair_typ_sel:
                if pair in pair2potential:
                    if style == "table":
                        table_file = tmp[param_offset]
                        if not os.path.isabs(table_file):
                            table_file = os.path.join(base_dir, table_file)

                        r, v, f = _parse_lmp_table(table_file)
                        _write_lmp_table(
                            table_file,
                            r, pair2potential[pair].value(r), pair2potential[pair].force(r),
                            f"Table {tmp[param_offset]}: id, r, potential, force", tmp[param_offset+1]
                        )
                    else:
                        n_param = pair2potential[pair].n_params()
                        tmp[param_offset:param_offset + n_param] = map(str, L_new[idx:idx + n_param])
                        idx += n_param
                        lines[i] = "   ".join(tmp) + "\n"

    with open(new_file, "w") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# FM B-spline table export (merged from fm_tables.py)
# ---------------------------------------------------------------------------

from ..utils.bonded_projectors import FMInteraction
from ..utils.table_padding import (
    constant_force_extrapolate,
    export_grid,
    integrate_force_to_potential,
)
from .fm_workflow import find_equilibrium, interaction_table_stem


def _eval_bspline_force_on_model_grid(
    spec: Dict[str, Any], interaction: FMInteraction
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate B-spline force on the model's interior grid."""
    model = str(spec.get("model", "")).lower()
    if model != "bspline":
        raise ValueError(f"Unsupported model in FM table export: {model}")
    pot = interaction.potential
    xmin = float(spec["min"])
    xmax = float(spec["max"])
    dx = float(spec["resolution"])
    x_model = np.arange(xmin + dx, xmax - dx + dx * 0.1, dx, dtype=float)
    if x_model.size < 3:
        x_model = np.linspace(xmin, xmax, max(3, int(round((xmax - xmin) / dx)) + 1), dtype=float)
    B_model = np.asarray(pot.basis_values(x_model), dtype=float)
    c = np.asarray(pot.get_params(), dtype=float).reshape(-1)
    return x_model, B_model @ c


def _fm_bspline_force_and_value(
    spec: Dict[str, Any],
    interaction: FMInteraction,
    x_out: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (V, F) for an FM B-spline interaction on the output grid.

    Uses constant-force extrapolation outside the model range.
    Style-specific handling: bonds are trimmed to the attractive well,
    pair potentials are shifted to zero at the cutoff, angles are floored at zero.
    """
    style = str(spec.get("style", "")).lower()
    model = str(spec.get("model", "")).lower()
    if model != "bspline":
        pot = interaction.potential
        return np.asarray(pot.value(x_out), dtype=float), np.asarray(pot.force(x_out), dtype=float)

    x_model, f_model = _eval_bspline_force_on_model_grid(spec, interaction)
    v_model = integrate_force_to_potential(x_model, f_model)

    if style == "bond":
        lo = 0
        hi = f_model.size - 1
        for i in range(f_model.size):
            if f_model[i] > 0.0:
                lo = i
                break
        for i in range(f_model.size - 1, -1, -1):
            if f_model[i] < 0.0:
                hi = i
                break
        if hi <= lo:
            lo, hi = 0, f_model.size - 1
        x_trim = x_model[lo : hi + 1]
        f_trim = f_model[lo : hi + 1]
        v_trim = integrate_force_to_potential(x_trim, f_trim)
        v_trim = v_trim - float(np.min(v_trim))
        v, f = constant_force_extrapolate(x_trim, v_trim, f_trim, x_out)
        v = np.maximum(v, 0.0)
        return v, f

    if style == "angle":
        v_model = v_model - float(np.min(v_model))
        v, f = constant_force_extrapolate(x_model, v_model, f_model, x_out)
        v = np.maximum(v, 0.0)
        return v, f

    if style == "pair":
        lo = 0
        for i in range(f_model.size):
            if f_model[i] > 0.0:
                lo = i
                break
        if lo >= f_model.size - 1:
            lo = max(0, f_model.size - 2)
        x_src = x_model[lo:]
        f_src = f_model[lo:]
        v_src = integrate_force_to_potential(x_src, f_src)
        v, f = constant_force_extrapolate(x_src, v_src, f_src, x_out)
    else:
        v, f = constant_force_extrapolate(x_model, v_model, f_model, x_out)

    if style == "pair":
        v = v - float(v[-1])
    else:
        v = v - float(np.min(v))
    return v, f


def build_forcefield_tables(
    cfg: Dict[str, Any],
    interactions: Sequence[FMInteraction],
) -> Dict[str, Any]:
    """Build table payload (r, V, F arrays) for all interactions.

    Returns a dict with 'tables' mapping table_stem -> {r, V, F, ...}.
    """
    payload: Dict[str, Any] = {"tables": {}}

    for inter, spec in zip(interactions, cfg["interactions"]):
        x = export_grid(spec)
        v, f = _fm_bspline_force_and_value(spec, inter, x)
        stem = interaction_table_stem(inter.style, inter.types)
        payload["tables"][stem] = {
            "style": str(inter.style),
            "types": [str(t) for t in inter.types],
            "r": np.asarray(x, dtype=float).tolist(),
            "V": np.asarray(v, dtype=float).tolist(),
            "F": np.asarray(f, dtype=float).tolist(),
            "min": float(x[0]),
            "max": float(x[-1]),
            "n": int(x.size),
            "eq": find_equilibrium(np.asarray(x, dtype=float), np.asarray(f, dtype=float)),
            "table_name": stem,
            "comment": f"AceCG FM export for {inter.style}:{':'.join(inter.types)}",
            "model_min": float(spec.get("min", x[0])),
            "model_max": float(spec.get("max", x[-1])),
        }
    return payload


def export_tables(
    cfg: Dict[str, Any],
    interactions: Sequence[FMInteraction],
    outdir: str | Path,
    *,
    table_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Export LAMMPS table files for all FM interactions.

    If table_payload is provided, writes from it directly.
    Otherwise builds the payload first.
    """
    if table_payload is None:
        table_payload = build_forcefield_tables(cfg=cfg, interactions=interactions)

    tables_raw = table_payload.get("tables", {})
    if not isinstance(tables_raw, dict):
        raise ValueError("forcefield table payload is missing 'tables' dictionary")

    bundle: Dict[str, Dict[str, Any]] = {}
    for stem, item in tables_raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"table payload for {stem!r} is not a dictionary")
        bundle[str(stem)] = {
            "r": np.asarray(item["r"], dtype=float),
            "V": np.asarray(item["V"], dtype=float),
            "F": np.asarray(item["F"], dtype=float),
            "comment": str(item.get("comment", f"AceCG FM export for {stem}")),
            "table_name": str(item.get("table_name", stem)),
        }

    out_path = Path(outdir)
    written = _write_lmp_table_bundle(str(out_path), bundle)

    manifest: Dict[str, Any] = {"tables": {}}
    for stem, item in tables_raw.items():
        entry = {k: v for k, v in dict(item).items() if k not in {"r", "V", "F"}}
        entry["file"] = str(written[str(stem)])
        manifest["tables"][str(stem)] = entry
    return manifest


def compare_table_files(
    reference_file: str | Path,
    candidate_file: str | Path,
    *,
    ngrid: int = 2000,
) -> Dict[str, float]:
    """Compare two LAMMPS table files on a common grid."""
    xr, vr, fr = _parse_lmp_table(str(reference_file))
    xc, vc, fc = _parse_lmp_table(str(candidate_file))

    if fr is None or fc is None:
        raise ValueError(f"Missing force column in table comparison: {reference_file} vs {candidate_file}")

    lo = max(float(np.min(xr)), float(np.min(xc)))
    hi = min(float(np.max(xr)), float(np.max(xc)))
    if hi <= lo:
        raise ValueError(f"No overlap in r-range between {reference_file} and {candidate_file}")

    x = np.linspace(lo, hi, int(ngrid), dtype=float)
    vr_i = np.interp(x, xr, vr)
    vc_i = np.interp(x, xc, vc)
    fr_i = np.interp(x, xr, fr)
    fc_i = np.interp(x, xc, fc)

    eq_r = find_equilibrium(x, fr_i)
    eq_c = find_equilibrium(x, fc_i)

    return {
        "max_abs_dV": float(np.max(np.abs(vc_i - vr_i))),
        "max_abs_dF": float(np.max(np.abs(fc_i - fr_i))),
        "eq_ref": float(eq_r),
        "eq_candidate": float(eq_c),
        "abs_dEQ": float(abs(eq_c - eq_r)),
    }
