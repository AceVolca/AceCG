"""LAMMPS table parsing, writing, and conversion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..topology.types import InteractionKey
from ..topology.forcefield import Forcefield


def _uniform_grid(xmin: float, xmax: float, dx: float) -> np.ndarray:
    xmin_f = float(xmin)
    xmax_f = float(xmax)
    dx_f = float(dx)
    n = int(round((xmax_f - xmin_f) / dx_f)) + 1
    if n < 2:
        n = 2
    return np.linspace(xmin_f, xmax_f, n, dtype=float)


def integrate_force_to_potential(x: np.ndarray, force: np.ndarray) -> np.ndarray:
    """Integrate a force table into a cutoff-anchored potential.

    Parameters
    ----------
    x : np.ndarray
        Monotonic grid coordinates.
    force : np.ndarray
        Force values on ``x``.

    Returns
    -------
    np.ndarray
        Potential values with the final grid point anchored at zero.
    """
    x = np.asarray(x, dtype=float).ravel()
    f = np.asarray(force, dtype=float).ravel()
    if x.size == 0:
        return np.empty(0, dtype=float)
    if x.size == 1:
        return np.zeros(1, dtype=float)

    dx = np.diff(x)
    trap = 0.5 * (f[:-1] + f[1:]) * dx
    u = np.empty_like(f)
    u[-1] = 0.0
    u[:-1] = np.cumsum(trap[::-1])[::-1]
    return u


def constant_force_extrapolate(
    x_model: np.ndarray,
    potential_model: np.ndarray,
    force_model: np.ndarray,
    x_out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate in-range and linearly extrapolate potential with boundary force."""
    xm = np.asarray(x_model, dtype=float).ravel()
    um = np.asarray(potential_model, dtype=float).ravel()
    fm = np.asarray(force_model, dtype=float).ravel()
    xo = np.asarray(x_out, dtype=float).ravel()
    if xm.size < 2:
        raise ValueError("x_model must contain at least two points")
    if xm.size != um.size or xm.size != fm.size:
        raise ValueError("x_model/potential_model/force_model must have identical lengths")

    u = np.empty_like(xo)
    f = np.empty_like(xo)
    lo = float(xm[0])
    hi = float(xm[-1])

    for i, xv in enumerate(xo):
        if xv <= lo:
            f[i] = fm[0]
            u[i] = um[0] + fm[0] * (lo - xv)
        elif xv >= hi:
            f[i] = fm[-1]
            u[i] = um[-1] + fm[-1] * (hi - xv)
        else:
            j = int(np.searchsorted(xm, xv))
            j = max(1, min(j, xm.size - 1))
            t = (xv - xm[j - 1]) / max(xm[j] - xm[j - 1], 1.0e-30)
            u[i] = um[j - 1] + t * (um[j] - um[j - 1])
            f[i] = fm[j - 1] + t * (fm[j] - fm[j - 1])

    return u, f


def export_grid(spec: Dict[str, Any]) -> np.ndarray:
    """Build a uniform output grid from an interaction spec dict."""
    dx = float(spec.get("table_resolution", spec["resolution"]))
    xmin = float(spec.get("table_min", spec["min"]))
    xmax = float(spec.get("table_max", spec["max"]))
    return _uniform_grid(xmin, xmax, dx)


def parse_lammps_table(table_path: str | Path):
    """Read a LAMMPS table file and return ``(r, V, F)``."""
    r_list, v_list, f_list = [], [], []
    with open(table_path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            try:
                if len(parts) >= 4:
                    float(parts[0])
                    float(parts[1])
                    float(parts[2])
                    float(parts[3])
                    r_list.append(float(parts[1]))
                    v_list.append(float(parts[2]))
                    f_list.append(float(parts[3]))
                elif len(parts) == 3:
                    try:
                        r_list.append(float(parts[1]))
                        v_list.append(float(parts[2]))
                    except Exception:
                        r_list.append(float(parts[0]))
                        v_list.append(float(parts[1]))
                        f_list.append(float(parts[2]))
                elif len(parts) == 2:
                    r_list.append(float(parts[0]))
                    v_list.append(float(parts[1]))
            except ValueError:
                continue

    if not r_list:
        raise ValueError(f"No numeric (r,V) rows found in {table_path}")

    r = np.asarray(r_list, dtype=float)
    v = np.asarray(v_list, dtype=float)
    f = np.asarray(f_list, dtype=float) if f_list else None

    mask = np.isfinite(r) & np.isfinite(v)
    if f is not None:
        mask &= np.isfinite(f)
    r = r[mask]
    v = v[mask]
    f = f[mask] if f is not None else None

    order = np.argsort(r)
    r = r[order]
    v = v[order]
    f = f[order] if f is not None else None
    return r, v, f


parse_lmp_table = parse_lammps_table


def write_lammps_table(
    filename: str | Path,
    r: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    comment: str = "LAMMPS Table written by AceCG",
    table_name: str = "Table1",
    table_style: str = "pair",
    eq: float | None = None,
    fp: Tuple[float, float] | None = None,
) -> None:
    """Write a LAMMPS-style pair, bond, or angle table file.

    Parameters
    ----------
    filename : str or Path
        Destination table path.
    r, V, F : np.ndarray
        Grid coordinates, potential values, and force values. All arrays must
        have the same shape.
    comment : str, default="LAMMPS Table written by AceCG"
        Comment text written above the table.
    table_name : str, default="Table1"
        LAMMPS table section name.
    table_style : {"pair", "bond", "angle"}, default="pair"
        Header format to write.
    eq : float, optional
        Equilibrium coordinate for bonded table headers.
    fp : tuple[float, float], optional
        Endpoint force derivatives for bonded table headers.
    """
    r = np.asarray(r, dtype=float)
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=float)
    if r.shape != V.shape or r.shape != F.shape:
        raise ValueError("r, V, F must have the same shape")

    style = str(table_style).lower()
    if style not in {"pair", "bond", "angle"}:
        raise ValueError(f"Unsupported LAMMPS table style: {table_style!r}")

    with open(filename, "w", encoding="utf-8") as f:
        if comment is not None:
            for line in comment.splitlines():
                f.write(f"# {line}\n")

        npoints = len(r)
        f.write(f"\n{table_name}\n")
        if style == "pair":
            f.write(f"N {npoints} R {r[0]:.6f} {r[-1]:.6f}\n\n")
        else:
            header_parts = [f"N {npoints}"]
            if fp is not None:
                header_parts.append(f"FP {float(fp[0]):.8e} {float(fp[1]):.8e}")
            if eq is not None:
                header_parts.append(f"EQ {float(eq):.8f}")
            f.write(" ".join(header_parts) + "\n\n")

        for i, (ri, vi, fi) in enumerate(zip(r, V, F), start=1):
            f.write(f"{i:6d}  {ri:16.8f}  {vi:16.8e}  {fi:16.8e}\n")


def write_lammps_table_bundle(
    outdir: str | Path,
    tables: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """Write a collection of LAMMPS table payloads.

    Parameters
    ----------
    outdir : str or Path
        Directory where ``*.table`` files are written.
    tables : dict
        Mapping from table stem to payload dictionaries containing ``r``,
        ``V``, ``F``, and optional style/header metadata.

    Returns
    -------
    dict[str, str]
        Mapping from table stem to written file path.
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    files: Dict[str, str] = {}
    for stem, item in tables.items():
        style = str(item.get("style", "pair")).lower()
        if style == "dihedral":
            raise ValueError(f"LAMMPS dihedral table export is not supported for {stem!r}")
        table_file = out_path / f"{stem}.table"
        write_lammps_table(
            filename=str(table_file),
            r=np.asarray(item["r"], dtype=float),
            V=np.asarray(item["V"], dtype=float),
            F=np.asarray(item["F"], dtype=float),
            comment=str(item.get("comment", "LAMMPS Table written by AceCG")),
            table_name=str(item.get("table_name", stem)),
            table_style=style,
            eq=float(item["eq"]) if item.get("eq") is not None else None,
            fp=tuple(item["fp"]) if item.get("fp") is not None else None,
        )
        files[str(stem)] = str(table_file)

    return files


def interaction_table_stem(style: str, types: Sequence[str]) -> str:
    """Return AceCG's default filename stem for an interaction table.

    Parameters
    ----------
    style : str
        Interaction style such as ``"pair"``, ``"bond"``, or ``"angle"``.
    types : Sequence[str]
        Interaction type labels.

    Returns
    -------
    str
        Stable table stem without extension.
    """
    joined = "_".join(types)
    if style == "pair":
        return joined
    if style == "bond":
        return f"{joined}_bon"
    if style == "angle":
        return f"{joined}_ang"
    if style == "dihedral":
        return f"{joined}_dih"
    if style == "nb3b":
        return f"{joined}_nb3b"
    raise ValueError(f"Unknown style: {style}")


def find_equilibrium(x: np.ndarray, force: np.ndarray) -> float:
    """Estimate the equilibrium coordinate from a force table.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional coordinate grid.
    force : np.ndarray
        Force values on ``x``.

    Returns
    -------
    float
        Coordinate of the minimum of the integrated potential, refined by a
        local quadratic fit when possible.
    """
    x = np.asarray(x, dtype=float)
    force_values = np.asarray(force, dtype=float)
    if x.ndim != 1 or force_values.ndim != 1 or x.size != force_values.size:
        raise ValueError("x and force must be one-dimensional arrays with identical lengths")
    if x.size == 0:
        raise ValueError("x and force must contain at least one value")
    if x.size == 1:
        return float(x[0])

    potential = integrate_force_to_potential(x, force_values)
    min_index = int(np.argmin(potential))
    if min_index == 0 or min_index == x.size - 1:
        return float(x[min_index])

    x_window = x[min_index - 1 : min_index + 2]
    potential_window = potential[min_index - 1 : min_index + 2]
    a, b, _ = np.polyfit(x_window, potential_window, deg=2)
    if abs(float(a)) < 1.0e-15:
        return float(x[min_index])

    vertex = float(-b / (2.0 * a))
    if x_window[0] <= vertex <= x_window[-1]:
        return vertex
    return float(x[min_index])


def _eval_bspline_force_on_model_grid(
    spec: Dict[str, Any],
    pot,
) -> Tuple[np.ndarray, np.ndarray]:
    model = str(spec.get("model", "")).lower()
    if model != "bspline":
        raise ValueError(f"Unsupported model in FM table export: {model}")
    xmin = float(spec["min"])
    xmax = float(spec["max"])
    dx = float(spec["resolution"])
    x_model = np.arange(xmin + dx, xmax - dx + dx * 0.1, dx, dtype=float)
    if x_model.size < 3:
        x_model = np.linspace(xmin, xmax, max(3, int(round((xmax - xmin) / dx)) + 1), dtype=float)
    B_model = np.asarray(pot.basis_values(x_model), dtype=float)
    c = np.asarray(pot.get_params(), dtype=float).reshape(-1)
    return x_model, B_model @ c


def estimate_table_fp(x: np.ndarray, y: np.ndarray) -> Tuple[float, float] | None:
    """Estimate endpoint derivatives for a tabulated curve.

    Parameters
    ----------
    x, y : np.ndarray
        One-dimensional grid and values.

    Returns
    -------
    tuple[float, float] or None
        Left and right finite-difference slopes, or ``None`` if they cannot be
        estimated.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return None
    dx_lo = float(x[1] - x[0])
    dx_hi = float(x[-1] - x[-2])
    if abs(dx_lo) < 1.0e-15 or abs(dx_hi) < 1.0e-15:
        return None
    return (
        float((y[1] - y[0]) / dx_lo),
        float((y[-1] - y[-2]) / dx_hi),
    )


def _fm_bspline_force_and_value(
    spec: Dict[str, Any],
    pot,
    x_out: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    style = str(spec.get("style", "")).lower()
    model = str(spec.get("model", "")).lower()
    if model != "bspline":
        return np.asarray(pot.value(x_out), dtype=float), np.asarray(pot.force(x_out), dtype=float)

    x_model, f_model = _eval_bspline_force_on_model_grid(spec, pot)
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
    forcefield: "Forcefield",
) -> Dict[str, Any]:
    """Build serializable table payloads from a forcefield and FM config.

    Parameters
    ----------
    cfg : dict
        Runtime FM configuration containing an ``"interactions"`` list.
    forcefield : Forcefield
        Forcefield whose potentials are evaluated on export grids.

    Returns
    -------
    dict
        Payload with a ``"tables"`` mapping ready for
        :func:`write_lammps_table_bundle`.
    """
    from ..potentials.base import IteratePotentials

    payload: Dict[str, Any] = {"tables": {}}

    for (key, pot), spec in zip(IteratePotentials(forcefield), cfg["interactions"]):
        x = export_grid(spec)
        v, f = _fm_bspline_force_and_value(spec, pot, x)
        style = str(key.style).lower()
        stem = interaction_table_stem(key.style, key.types)
        payload["tables"][stem] = {
            "style": style,
            "types": [str(t) for t in key.types],
            "r": np.asarray(x, dtype=float).tolist(),
            "V": np.asarray(v, dtype=float).tolist(),
            "F": np.asarray(f, dtype=float).tolist(),
            "min": float(x[0]),
            "max": float(x[-1]),
            "n": int(x.size),
            "eq": find_equilibrium(np.asarray(x, dtype=float), np.asarray(f, dtype=float)),
            "fp": estimate_table_fp(np.asarray(x, dtype=float), np.asarray(f, dtype=float))
            if style in {"bond", "angle"}
            else None,
            "table_name": stem,
            "comment": f"AceCG FM export for {key.style}:{':'.join(key.types)}",
            "model_min": float(spec.get("min", x[0])),
            "model_max": float(spec.get("max", x[-1])),
        }
    return payload


def export_tables(
    cfg: Dict[str, Any],
    forcefield: "Forcefield",
    outdir: str | Path,
    *,
    table_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Export forcefield tables and return a manifest.

    Parameters
    ----------
    cfg : dict
        Runtime FM configuration.
    forcefield : Forcefield
        Forcefield to evaluate.
    outdir : str or Path
        Destination directory for table files.
    table_payload : dict, optional
        Precomputed table payload. If omitted, it is built from ``cfg`` and
        ``forcefield``.

    Returns
    -------
    dict
        Manifest containing per-table metadata and written file paths.
    """
    if table_payload is None:
        table_payload = build_forcefield_tables(cfg=cfg, forcefield=forcefield)

    tables_raw = table_payload.get("tables", {})
    if not isinstance(tables_raw, dict):
        raise ValueError("forcefield table payload is missing 'tables' dictionary")

    bundle: Dict[str, Dict[str, Any]] = {}
    for stem, item in tables_raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"table payload for {stem!r} is not a dictionary")
        bundle[str(stem)] = {
            "style": str(item.get("style", "pair")),
            "r": np.asarray(item["r"], dtype=float),
            "V": np.asarray(item["V"], dtype=float),
            "F": np.asarray(item["F"], dtype=float),
            "comment": str(item.get("comment", f"AceCG FM export for {stem}")),
            "table_name": str(item.get("table_name", stem)),
            "eq": float(item["eq"]) if item.get("eq") is not None else None,
            "fp": tuple(item["fp"]) if item.get("fp") is not None else None,
        }

    out_path = Path(outdir)
    written = write_lammps_table_bundle(str(out_path), bundle)

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
    """Compare two LAMMPS table files on a common interpolation grid.

    Parameters
    ----------
    reference_file, candidate_file : str or Path
        Table files to compare.
    ngrid : int, default=2000
        Number of interpolation points in the overlapping coordinate range.

    Returns
    -------
    dict[str, float]
        Maximum absolute energy/force differences and equilibrium-coordinate
        diagnostics.
    """
    xr, vr, fr = parse_lammps_table(str(reference_file))
    xc, vc, fc = parse_lammps_table(str(candidate_file))

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


def _interaction_table_filename(style: str, types: Sequence[str]) -> str:
    style_key = str(style).lower()
    labels = tuple(str(item) for item in types)
    if style_key == "pair":
        return f"Pair_{labels[0]}-{labels[1]}.table"
    if style_key == "bond":
        return f"{labels[0]}_{labels[1]}_bon.table"
    if style_key == "angle":
        return f"{labels[0]}_{labels[1]}_{labels[2]}_ang.table"
    raise ValueError(f"Unsupported interaction style {style!r}")


def _table_name(style: str, types: Sequence[str]) -> str:
    filename = _interaction_table_filename(style, types)
    return filename[:-6] if filename.endswith(".table") else filename


def _baseline_pair_table_keyword(types: Sequence[str]) -> str:
    labels = tuple(str(item) for item in types)
    return f"{labels[0]}_{labels[1]}"


def _extend_table_constant_force_tail(
    x: np.ndarray,
    value: np.ndarray,
    force: np.ndarray,
    *,
    max_value: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.asarray(x, dtype=np.float64).copy()
    energy = np.asarray(value, dtype=np.float64).copy()
    grad = np.asarray(force, dtype=np.float64).copy()
    if grid.size == 0 or float(grid[-1]) >= float(max_value):
        return grid, energy, grad
    if grid.size < 2:
        raise ValueError("table grid must have at least two points to extend its tail")
    step = float(np.median(np.diff(grid)))
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("table grid must be strictly increasing")
    extension = np.arange(float(grid[-1]) + step, float(max_value) + 0.5 * step, step, dtype=np.float64)
    if extension.size == 0:
        return grid, energy, grad
    tail_force = np.full_like(extension, float(grad[-1]), dtype=np.float64)
    tail_energy = float(energy[-1]) - float(grad[-1]) * (extension - float(grid[-1]))
    return (
        np.concatenate([grid, extension]),
        np.concatenate([energy, tail_energy]),
        np.concatenate([grad, tail_force]),
    )


def cap_table_forces(
    table_path: str | Path,
    max_force: float = 100.0,
    scale: float = 1.0,
) -> None:
    """Clamp force values in an existing LAMMPS table file.

    Parameters
    ----------
    table_path : str or Path
        Table file to edit in place.
    max_force : float, default=100.0
        Absolute force cap.
    scale : float, default=1.0
        Optional multiplier applied before clipping.
    """
    max_f = abs(float(max_force))
    scale_f = float(scale)
    path = Path(table_path)
    raw = path.read_text(encoding="utf-8")

    header_lines: list[str] = []
    data_lines: list[str] = []
    table_name = ""
    in_header = True
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            if in_header:
                header_lines.append(line)
            continue
        if stripped.startswith("#"):
            header_lines.append(line)
            continue
        parts = stripped.split()
        if in_header:
            try:
                float(parts[0])
                in_header = False
                data_lines.append(line)
            except ValueError:
                if parts[0] == "N":
                    in_header = False
                else:
                    table_name = stripped
                continue
        else:
            if parts[0] == "N":
                continue
            data_lines.append(line)

    r_list, f_list = [], []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 4:
            try:
                r_list.append(float(parts[1]))
                f_list.append(float(parts[3]))
            except ValueError:
                continue

    if not r_list:
        return

    r = np.asarray(r_list, dtype=float)
    f = np.asarray(f_list, dtype=float)
    f_scaled = f * scale_f
    f_capped = np.clip(f_scaled, -max_f, max_f)
    v = integrate_force_to_potential(r, f_capped)

    write_lammps_table(
        filename=str(path),
        r=r,
        V=v,
        F=f_capped,
        comment="\n".join(l.lstrip("# ") for l in header_lines if l.strip().startswith("#"))
        or f"Force-capped table (max_force={max_f})",
        table_name=table_name,
        table_style="pair",
    )


__all__ = [
    "parse_lammps_table",
    "parse_lmp_table",
    "integrate_force_to_potential",
    "constant_force_extrapolate",
    "export_grid",
    "interaction_table_stem",
    "find_equilibrium",
    "build_forcefield_tables",
    "export_tables",
    "compare_table_files",
    "cap_table_forces",
    "write_lammps_table",
    "write_lammps_table_bundle",
    "estimate_table_fp",
]
