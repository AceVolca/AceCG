"""LAMMPS table parsing, writing, and conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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
    if spec.get("table_grid") is not None:
        grid = np.asarray(spec["table_grid"], dtype=float).reshape(-1)
        if grid.size < 2 or np.any(np.diff(grid) <= 0.0):
            raise ValueError("table_grid must contain strictly increasing coordinates")
        return grid
    dx = float(spec.get("table_resolution", spec["resolution"]))
    xmin = float(spec.get("table_min", spec["min"]))
    xmax = float(spec.get("table_max", spec["max"]))
    if str(spec.get("style", "")).strip().lower() == "dihedral":
        span = xmax - xmin
        n = int(round(span / dx))
        if n < 2 or not np.isclose(n * dx, span, rtol=0.0, atol=1.0e-8):
            raise ValueError(
                "dihedral table resolution must divide its output span exactly"
            )
        # Half-open cyclic grid: never emit both congruent seam endpoints.
        return xmin + np.arange(n, dtype=float) * (span / n)
    return _uniform_grid(xmin, xmax, dx)


@dataclass(frozen=True)
class LammpsTableSection:
    """One keyword-selected section from a LAMMPS table file."""

    keyword: str
    header_tokens: Tuple[str, ...]
    x: np.ndarray
    potential: np.ndarray
    force: Optional[np.ndarray]
    metadata: Dict[str, str] = field(default_factory=dict)


def _update_acecg_metadata(raw: str, metadata: Dict[str, str]) -> None:
    """Add metadata from one AceCG table comment to ``metadata`` in-place."""
    stripped = raw.strip()
    if not stripped.lower().startswith("# acecg-table "):
        return
    for token in stripped.split()[2:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        metadata[str(key)] = str(value)


def _read_lammps_table_sections(
    table_path: str | Path,
    *,
    table_styles: Optional[Mapping[str, str]] = None,
    default_table_style: str | None = None,
) -> list[LammpsTableSection]:
    path = Path(table_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: list[LammpsTableSection] = []
    pending_metadata: Dict[str, str] = {}
    index = 0
    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("#"):
            _update_acecg_metadata(raw, pending_metadata)
            index += 1
            continue
        tokens = stripped.split()
        numeric_row = (
            len(tokens) in {3, 4}
            and re.fullmatch(r"[+-]?\d+", tokens[0]) is not None
        )
        if numeric_row:
            try:
                [float(token) for token in tokens[1:]]
            except ValueError:
                numeric_row = False
        if numeric_row:
            raise ValueError(
                f"Numeric table row outside a declared section in {path}: "
                f"{stripped!r}"
            )

        keyword = tokens[0]
        header_index = index + 1
        while header_index < len(lines):
            header_text = lines[header_index].split("#", 1)[0].strip()
            if header_text:
                break
            header_index += 1
        if header_index >= len(lines):
            raise ValueError(
                f"Missing table header after section {keyword!r} in {path}"
            )
        header_tokens = tuple(header_text.split())
        n_positions = [
            position
            for position, token in enumerate(header_tokens)
            if token.upper() == "N"
        ]
        if not n_positions:
            raise ValueError(
                f"Missing N header in table section {keyword!r} of {path}"
            )
        if len(n_positions) != 1 or n_positions[0] + 1 >= len(header_tokens):
            raise ValueError(
                f"Invalid N header in table section {keyword!r} of {path}"
            )
        try:
            n_rows = int(header_tokens[n_positions[0] + 1])
        except ValueError as exc:
            raise ValueError(
                f"Invalid N header in table section {keyword!r} of {path}"
            ) from exc
        if n_rows < 1:
            raise ValueError(
                f"Table section {keyword!r} in {path} has invalid N={n_rows}"
            )

        rows: list[list[float]] = []
        row_index = header_index + 1
        while row_index < len(lines) and len(rows) < n_rows:
            row_text = lines[row_index].split("#", 1)[0].strip()
            row_index += 1
            if not row_text:
                continue
            parts = row_text.split()
            try:
                values = [float(token) for token in parts]
            except ValueError:
                break
            if len(values) not in {3, 4}:
                raise ValueError(
                    f"Expected 3 or 4 numeric columns in section {keyword!r} "
                    f"of {path}, got {len(values)}"
                )
            rows.append(values)
        if len(rows) != n_rows:
            raise ValueError(
                f"Table section {keyword!r} in {path} declares N={n_rows} "
                f"but contains {len(rows)} numeric rows"
            )
        next_index = row_index
        while next_index < len(lines):
            next_text = lines[next_index].split("#", 1)[0].strip()
            if next_text:
                break
            next_index += 1
        if next_index < len(lines):
            next_tokens = next_text.split()
            if (
                len(next_tokens) in {3, 4}
                and re.fullmatch(r"[+-]?\d+", next_tokens[0]) is not None
            ):
                try:
                    [float(token) for token in next_tokens[1:]]
                except ValueError:
                    pass
                else:
                    raise ValueError(
                        f"Table section {keyword!r} in {path} declares N={n_rows} "
                        "but contains additional numeric rows"
                    )

        widths = {len(row) for row in rows}
        if len(widths) != 1:
            raise ValueError(
                f"Table section {keyword!r} in {path} mixes row widths"
            )
        values = np.asarray(rows, dtype=float)
        x = values[:, 1]
        potential = values[:, 2]
        force = values[:, 3] if values.shape[1] == 4 else None
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(potential)):
            raise ValueError(f"Non-finite table values in section {keyword!r}")
        if force is not None and np.any(~np.isfinite(force)):
            raise ValueError(f"Non-finite force values in section {keyword!r}")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError(
                f"Coordinates in table section {keyword!r} must be strictly increasing"
            )
        if any(section.keyword == keyword for section in sections):
            raise ValueError(f"Duplicate table keyword {keyword!r} in {path}")
        section = LammpsTableSection(
            keyword=keyword,
            header_tokens=header_tokens,
            x=x,
            potential=potential,
            force=force,
            metadata=dict(pending_metadata),
        )
        declared_style = (
            table_styles.get(keyword, default_table_style)
            if table_styles is not None
            else default_table_style
        )
        if str(declared_style or "").strip().lower() == "dihedral":
            header_upper = {token.upper() for token in section.header_tokens}
            if "DEGREES" in header_upper and "RADIANS" in header_upper:
                raise ValueError(
                    f"Dihedral table {path} cannot specify DEGREES and RADIANS"
                )
            if "DEGREES" not in header_upper:
                converted_force = None
                if section.force is not None:
                    converted_force = np.asarray(section.force, dtype=float) * (
                        np.pi / 180.0
                    )
                section = LammpsTableSection(
                    keyword=section.keyword,
                    header_tokens=section.header_tokens,
                    x=np.degrees(np.asarray(section.x, dtype=float)),
                    potential=np.asarray(section.potential, dtype=float),
                    force=converted_force,
                    metadata=dict(section.metadata),
                )
        sections.append(section)
        pending_metadata.clear()
        index = row_index
    return sections


def read_lammps_table_section(
    table_path: str | Path,
    *,
    table_name: str | None = None,
    table_style: str | None = None,
) -> LammpsTableSection:
    """Read one keyword-selected LAMMPS table section."""
    style = str(table_style or "").strip().lower()
    sections = _read_lammps_table_sections(
        table_path,
        table_styles=(
            {str(table_name): style}
            if table_name is not None and style
            else None
        ),
        default_table_style=(style if table_name is None else None),
    )
    if not sections:
        raise ValueError(f"No LAMMPS table sections found in {table_path}")
    if table_name is None:
        if len(sections) != 1:
            names = [section.keyword for section in sections]
            raise ValueError(
                f"{table_path} contains multiple table sections {names}; "
                "a table_name is required"
            )
        section = sections[0]
    else:
        matches = [
            section for section in sections if section.keyword == str(table_name)
        ]
        if not matches:
            raise KeyError(
                f"Table keyword {table_name!r} was not found in {table_path}"
            )
        section = matches[0]
    return section


def parse_lammps_table(
    table_path: str | Path,
    table_name: str | None = None,
    table_style: str | None = None,
):
    """Read one LAMMPS table section and return ``(x, V, F)``."""
    section = read_lammps_table_section(
        table_path,
        table_name=table_name,
        table_style=table_style,
    )
    return section.x.copy(), section.potential.copy(), (
        None if section.force is None else section.force.copy()
    )


def _write_lammps_table_sections(
    path: str | Path,
    sections: Sequence[LammpsTableSection],
    *,
    comment: str | None = None,
) -> None:
    """Serialize a complete ordered section sequence to a new stage path."""
    destination = Path(path)
    seen: set[str] = set()
    prepared: list[
        tuple[LammpsTableSection, np.ndarray, np.ndarray, Optional[np.ndarray]]
    ] = []
    for section in sections:
        keyword = str(section.keyword)
        if not keyword or any(char.isspace() for char in keyword):
            raise ValueError(f"Invalid LAMMPS table keyword {keyword!r}")
        if keyword in seen:
            raise ValueError(f"Duplicate table keyword {keyword!r}")
        seen.add(keyword)
        x = np.asarray(section.x, dtype=float)
        potential = np.asarray(section.potential, dtype=float)
        force = (
            None
            if section.force is None
            else np.asarray(section.force, dtype=float)
        )
        if x.ndim != 1 or potential.ndim != 1 or x.shape != potential.shape:
            raise ValueError(
                f"Table section {keyword!r} coordinates and potential must be "
                "one-dimensional arrays with identical shapes"
            )
        if force is not None and (force.ndim != 1 or force.shape != x.shape):
            raise ValueError(
                f"Table section {keyword!r} force must match its coordinate shape"
            )
        if x.size < 1:
            raise ValueError(f"Table section {keyword!r} must contain at least one row")
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(potential)):
            raise ValueError(f"Non-finite table values in section {keyword!r}")
        if force is not None and np.any(~np.isfinite(force)):
            raise ValueError(f"Non-finite force values in section {keyword!r}")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError(
                f"Coordinates in table section {keyword!r} must be strictly increasing"
            )
        header_tokens = tuple(str(token) for token in section.header_tokens)
        n_positions = [
            position
            for position, token in enumerate(header_tokens)
            if token.upper() == "N"
        ]
        if len(n_positions) != 1 or n_positions[0] + 1 >= len(header_tokens):
            raise ValueError(f"Invalid N header in table section {keyword!r}")
        try:
            declared_rows = int(header_tokens[n_positions[0] + 1])
        except ValueError as exc:
            raise ValueError(f"Invalid N header in table section {keyword!r}") from exc
        if declared_rows != x.size:
            raise ValueError(
                f"Table section {keyword!r} declares N={declared_rows} "
                f"but contains {x.size} rows"
            )
        prepared.append((section, x, potential, force))

    created = False
    try:
        with destination.open("x", encoding="utf-8") as handle:
            created = True
            if comment is not None:
                for line in str(comment).splitlines():
                    handle.write(f"# {line}\n")
            for section, x, potential, force in prepared:
                if section.metadata:
                    payload = " ".join(
                        f"{key}={section.metadata[key]}"
                        for key in sorted(section.metadata)
                    )
                    handle.write(f"# ACECG-TABLE {payload}\n")
                handle.write(f"\n{section.keyword}\n")
                handle.write(" ".join(section.header_tokens) + "\n\n")
                if force is None:
                    for row, (coordinate, value) in enumerate(
                        zip(x, potential), start=1
                    ):
                        handle.write(
                            f"{row:6d}  {coordinate:16.8f}  {value:16.8e}\n"
                        )
                else:
                    for row, (coordinate, value, force_value) in enumerate(
                        zip(x, potential, force), start=1
                    ):
                        handle.write(
                            f"{row:6d}  {coordinate:16.8f}  {value:16.8e}  "
                            f"{force_value:16.8e}\n"
                        )
    except Exception as exc:
        if created and destination.exists():
            try:
                destination.unlink()
            except Exception as cleanup_exc:
                exc.add_note(f"Failed to remove table stage {destination}: {cleanup_exc}")
        raise


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
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a LAMMPS-style pair, bond, angle, or dihedral table file.

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
    table_style : {"pair", "bond", "angle", "dihedral"}, default="pair"
        Header format to write.
    eq : float, optional
        Equilibrium coordinate for bonded table headers.
    fp : tuple[float, float], optional
        Endpoint force derivatives for bonded table headers.
    """
    r = np.asarray(r, dtype=float)
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=float)
    if r.ndim != 1 or V.ndim != 1 or F.ndim != 1:
        raise ValueError("r, V, F must be one-dimensional arrays")
    if r.shape != V.shape or r.shape != F.shape:
        raise ValueError("r, V, F must have the same shape")

    style = str(table_style).lower()
    if style not in {"pair", "bond", "angle", "dihedral"}:
        raise ValueError(f"Unsupported LAMMPS table style: {table_style!r}")
    if np.any(~np.isfinite(r)) or np.any(~np.isfinite(V)) or np.any(~np.isfinite(F)):
        raise ValueError("LAMMPS table coordinates, potential, and force must be finite")
    if r.size < 2 or np.any(np.diff(r) <= 0.0):
        raise ValueError("LAMMPS table coordinates must be strictly increasing")
    if style == "dihedral":
        span = float(r[-1] - r[0])
        if span >= 360.0 - 1.0e-10:
            raise ValueError(
                "LAMMPS dihedral table angle span must be strictly less than 360 degrees"
            )
        wrapped = (r - r[0]) % 360.0
        if np.unique(np.round(wrapped, 10)).size != r.size:
            raise ValueError("LAMMPS dihedral table contains congruent angle entries")
    npoints = len(r)
    if style == "pair":
        header_tokens = (
            "N",
            str(npoints),
            "R",
            f"{r[0]:.6f}",
            f"{r[-1]:.6f}",
        )
    elif style in {"bond", "angle"}:
        header_parts = ["N", str(npoints)]
        if fp is not None:
            header_parts.extend(
                ("FP", f"{float(fp[0]):.8e}", f"{float(fp[1]):.8e}")
            )
        if eq is not None:
            header_parts.extend(("EQ", f"{float(eq):.8f}"))
        header_tokens = tuple(header_parts)
    else:
        header_tokens = ("N", str(npoints), "DEGREES")

    destination = Path(filename)
    stage = destination.with_name(f".{destination.name}.acecg-stage")
    section = LammpsTableSection(
        keyword=str(table_name),
        header_tokens=header_tokens,
        x=r,
        potential=V,
        force=F,
        metadata={str(key): str(value) for key, value in (metadata or {}).items()},
    )
    _write_lammps_table_sections(stage, (section,), comment=comment)
    try:
        os.replace(stage, destination)
    except Exception as exc:
        if stage.exists():
            try:
                stage.unlink()
            except Exception as cleanup_exc:
                exc.add_note(f"Failed to remove table stage {stage}: {cleanup_exc}")
        raise


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
    if style == "dihedral":
        return (
            np.asarray(pot.value(x_out), dtype=float),
            np.asarray(pot.force(x_out), dtype=float),
        )
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
        Payload with a ``"tables"`` mapping ready for :func:`export_tables`.
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
            "metadata": (
                pot.table_metadata()
                if hasattr(pot, "table_metadata")
                else {}
            ),
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

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    for stem, item in tables_raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"table payload for {stem!r} is not a dictionary")
        table_file = out_path / f"{stem}.table"
        write_lammps_table(
            filename=table_file,
            r=np.asarray(item["r"], dtype=float),
            V=np.asarray(item["V"], dtype=float),
            F=np.asarray(item["F"], dtype=float),
            comment=str(item.get("comment", f"AceCG FM export for {stem}")),
            table_name=str(item.get("table_name", stem)),
            table_style=str(item.get("style", "pair")),
            eq=float(item["eq"]) if item.get("eq") is not None else None,
            fp=tuple(item["fp"]) if item.get("fp") is not None else None,
            metadata=dict(item.get("metadata", {})),
        )
        written[str(stem)] = str(table_file)

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
    if style_key == "dihedral":
        return f"{'_'.join(labels)}_dih.table"
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


__all__ = [
    "LammpsTableSection",
    "parse_lammps_table",
    "read_lammps_table_section",
    "integrate_force_to_potential",
    "constant_force_extrapolate",
    "export_grid",
    "interaction_table_stem",
    "find_equilibrium",
    "build_forcefield_tables",
    "export_tables",
    "compare_table_files",
    "write_lammps_table",
    "estimate_table_fp",
]
