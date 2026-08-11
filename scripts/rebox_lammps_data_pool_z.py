#!/usr/bin/env python3
"""Rebox a pool of LAMMPS data files by bonded unwrapping in z only.

The intended use is a VP-grown DOPC init pool whose original z box is shorter
than the target XZ sampling box.  For each bonded component, the script unwraps
z coordinates using the original z periodic minimum image, then translates the
component just enough to fit in the expanded z box.  X and Y coordinates are
left unchanged, so components that cross x/y periodic boundaries keep doing so.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import math
import os
import re
import shutil
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SECTION_NAMES = {
    "Masses",
    "Atoms",
    "Velocities",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
    "Pair Coeffs",
    "Bond Coeffs",
    "Angle Coeffs",
    "Dihedral Coeffs",
    "Improper Coeffs",
}

FRAME_DATA_RE = re.compile(r"^frame_\d+\.data$")


@dataclass(frozen=True)
class AtomRecord:
    line_index: int
    atom_id: int
    x: float
    y: float
    z: float
    x_token: int
    y_token: int
    z_token: int
    ix_token: int | None
    iy_token: int | None
    iz_token: int | None
    tokens: tuple[str, ...]
    comment: str
    newline: str


@dataclass(frozen=True)
class ReboxStats:
    input: str
    output: str
    n_atoms: int
    n_bonds: int
    n_components: int
    changed_atoms: int
    crossing_components: int
    zlo_old: float
    zhi_old: float
    zlo_new: float
    zhi_new: float
    max_abs_z_displacement: float
    max_raw_bond_dz: float
    max_min_image_bond_length: float


def split_comment(line: str) -> tuple[str, str, str]:
    newline = "\n" if line.endswith("\n") else ""
    body = line[:-1] if newline else line
    if "#" not in body:
        return body, "", newline
    content, comment = body.split("#", 1)
    return content.rstrip(), " #" + comment, newline


def section_name_from_line(line: str) -> str | None:
    content = line.split("#", 1)[0].strip()
    if not content:
        return None
    for name in sorted(SECTION_NAMES, key=len, reverse=True):
        if content == name or content.startswith(name + " "):
            return name
    return None


def parse_box_bounds(lines: list[str]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for line in lines:
        content = line.split("#", 1)[0].strip()
        parts = content.split()
        if len(parts) >= 4 and parts[2:] == ["xlo", "xhi"]:
            bounds["x"] = (float(parts[0]), float(parts[1]))
        elif len(parts) >= 4 and parts[2:] == ["ylo", "yhi"]:
            bounds["y"] = (float(parts[0]), float(parts[1]))
        elif len(parts) >= 4 and parts[2:] == ["zlo", "zhi"]:
            bounds["z"] = (float(parts[0]), float(parts[1]))
    missing = sorted({"x", "y", "z"} - set(bounds))
    if missing:
        raise ValueError(f"Missing box bounds for dimensions: {', '.join(missing)}")
    return bounds["x"], bounds["y"], bounds["z"]


def atom_column_indices(atom_header: str, tokens: list[str]) -> tuple[int, int, int, int | None, int | None, int | None]:
    style = atom_header.split("#", 1)[1].strip().lower() if "#" in atom_header else ""
    if style.startswith("full"):
        xyz = (4, 5, 6)
    elif style.startswith("atomic"):
        xyz = (2, 3, 4)
    elif len(tokens) >= 7:
        xyz = (4, 5, 6)
    elif len(tokens) >= 5:
        xyz = (2, 3, 4)
    else:
        raise ValueError(f"Unsupported Atoms row with {len(tokens)} columns: {' '.join(tokens)}")
    image_start = xyz[2] + 1
    if len(tokens) >= image_start + 3:
        return (*xyz, image_start, image_start + 1, image_start + 2)
    return (*xyz, None, None, None)


def parse_atoms_and_bonds(lines: list[str]) -> tuple[list[AtomRecord], list[tuple[int, int]], str]:
    section: str | None = None
    atom_header = ""
    atoms: list[AtomRecord] = []
    bonds: list[tuple[int, int]] = []

    for idx, line in enumerate(lines):
        maybe_section = section_name_from_line(line)
        if maybe_section is not None:
            section = maybe_section
            if section == "Atoms":
                atom_header = line
            continue
        if section not in {"Atoms", "Bonds"}:
            continue
        content, comment, newline = split_comment(line)
        tokens = content.split()
        if not tokens or not tokens[0].lstrip("+-").isdigit():
            continue
        if section == "Atoms":
            x_idx, y_idx, z_idx, ix_idx, iy_idx, iz_idx = atom_column_indices(atom_header, tokens)
            atoms.append(
                AtomRecord(
                    line_index=idx,
                    atom_id=int(tokens[0]),
                    x=float(tokens[x_idx]),
                    y=float(tokens[y_idx]),
                    z=float(tokens[z_idx]),
                    x_token=x_idx,
                    y_token=y_idx,
                    z_token=z_idx,
                    ix_token=ix_idx,
                    iy_token=iy_idx,
                    iz_token=iz_idx,
                    tokens=tuple(tokens),
                    comment=comment,
                    newline=newline,
                )
            )
        elif section == "Bonds" and len(tokens) >= 4:
            bonds.append((int(tokens[2]), int(tokens[3])))
    if not atoms:
        raise ValueError("No atoms parsed from Atoms section")
    return atoms, bonds, atom_header


def minimum_image_delta(delta: float, box_length: float) -> float:
    half = 0.5 * box_length
    if delta > half:
        return delta - box_length
    if delta < -half:
        return delta + box_length
    return delta


def connected_components(atom_ids: set[int], bonds: Iterable[tuple[int, int]]) -> tuple[list[list[int]], dict[int, list[int]]]:
    adjacency: dict[int, list[int]] = {atom_id: [] for atom_id in atom_ids}
    for a, b in bonds:
        if a not in adjacency or b not in adjacency:
            continue
        adjacency[a].append(b)
        adjacency[b].append(a)

    components: list[list[int]] = []
    seen: set[int] = set()
    for atom_id in sorted(atom_ids):
        if atom_id in seen:
            continue
        queue: deque[int] = deque([atom_id])
        seen.add(atom_id)
        component: list[int] = []
        while queue:
            cur = queue.popleft()
            component.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(component)
    return components, adjacency


def unwrap_component_z(
    component: list[int],
    adjacency: dict[int, list[int]],
    z_by_id: dict[int, float],
    old_lz: float,
) -> dict[int, float]:
    root = component[0]
    unwrapped = {root: z_by_id[root]}
    queue: deque[int] = deque([root])
    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if nxt not in component or nxt in unwrapped:
                continue
            dz = minimum_image_delta(z_by_id[nxt] - z_by_id[cur], old_lz)
            unwrapped[nxt] = unwrapped[cur] + dz
            queue.append(nxt)
    return unwrapped


def best_component_shift(
    atom_ids: list[int],
    z_unwrapped: dict[int, float],
    z_original: dict[int, float],
    zlo_new: float,
    zhi_new: float,
) -> float:
    min_z = min(z_unwrapped[i] for i in atom_ids)
    max_z = max(z_unwrapped[i] for i in atom_ids)
    lower = zlo_new - min_z
    upper = zhi_new - max_z
    if lower > upper:
        span = max_z - min_z
        raise ValueError(
            f"Cannot fit bonded component with z span {span:.6f} into "
            f"target bounds {zlo_new:.6f} {zhi_new:.6f}"
        )
    if lower <= 0.0 <= upper:
        return 0.0

    candidates = {lower, upper}
    for atom_id in atom_ids:
        preferred = z_original[atom_id] - z_unwrapped[atom_id]
        candidates.add(min(max(preferred, lower), upper))

    def score(shift: float) -> tuple[int, float, float]:
        displacements = [abs((z_unwrapped[i] + shift) - z_original[i]) for i in atom_ids]
        moved = sum(disp > 1e-5 for disp in displacements)
        return moved, sum(displacements), abs(shift)

    return min(candidates, key=score)


def wrap_into_box(value: float, lo: float, hi: float) -> tuple[float, int]:
    length = hi - lo
    image = math.floor((value - lo) / length)
    wrapped = value - image * length
    if wrapped >= hi:
        wrapped -= length
        image += 1
    if wrapped < lo:
        wrapped += length
        image -= 1
    return wrapped, image


def rewrite_z_bound(line: str, zlo_new: float, zhi_new: float, precision: int) -> str:
    if "zlo" not in line or "zhi" not in line:
        return line
    content, comment, newline = split_comment(line)
    parts = content.split()
    if len(parts) >= 4 and parts[2:] == ["zlo", "zhi"]:
        return f"{zlo_new:.{precision}f} {zhi_new:.{precision}f} zlo zhi{comment}{newline}"
    return line


def format_float(value: float, precision: int) -> str:
    text = f"{value:.{precision}f}"
    if text == "-0." + "0" * precision:
        return "0." + "0" * precision
    return text


def rewrite_atom_line(atom: AtomRecord, z_new: float, z_image: int, precision: int, zero_z_image: bool) -> str:
    tokens = list(atom.tokens)
    if abs(z_new - atom.z) > 0.5 * 10 ** (-precision):
        tokens[atom.z_token] = format_float(z_new, precision)
    if zero_z_image and atom.iz_token is not None:
        tokens[atom.iz_token] = str(z_image)
    return " ".join(tokens) + atom.comment + atom.newline


def rebox_file(
    input_path: Path,
    output_path: Path,
    target_z_length: float,
    precision: int,
    overwrite: bool,
    zero_z_image: bool,
) -> ReboxStats:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}")
    lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
    (xlo, xhi), (ylo, yhi), (zlo_old, zhi_old) = parse_box_bounds(lines)
    atoms, bonds, _atom_header = parse_atoms_and_bonds(lines)
    old_lz = zhi_old - zlo_old
    if old_lz <= 0:
        raise ValueError(f"Invalid old z length in {input_path}: {old_lz}")
    if target_z_length < old_lz:
        raise ValueError(f"target z length {target_z_length} is smaller than input z length {old_lz}")
    z_center = 0.5 * (zlo_old + zhi_old)
    zlo_new = z_center - 0.5 * target_z_length
    zhi_new = z_center + 0.5 * target_z_length

    atom_by_id = {a.atom_id: a for a in atoms}
    z_original = {a.atom_id: a.z for a in atoms}
    atom_ids = set(atom_by_id)
    components, adjacency = connected_components(atom_ids, bonds)

    z_final: dict[int, float] = {}
    z_image: dict[int, int] = {}
    crossing_components = 0
    changed_atoms = 0
    max_abs_disp = 0.0

    for component in components:
        unwrapped = unwrap_component_z(component, adjacency, z_original, old_lz)
        if len(unwrapped) != len(component):
            missing = sorted(set(component) - set(unwrapped))
            raise ValueError(f"Disconnected unwrap state for component starting {component[0]}: {missing[:5]}")
        component_crosses = any(abs(unwrapped[i] - z_original[i]) > 1e-5 for i in component)
        if component_crosses:
            crossing_components += 1
        shift = best_component_shift(component, unwrapped, z_original, zlo_new, zhi_new)
        for atom_id in component:
            z_wrapped, image = wrap_into_box(unwrapped[atom_id] + shift, zlo_new, zhi_new)
            z_final[atom_id] = z_wrapped
            z_image[atom_id] = image
            disp = abs(z_wrapped - z_original[atom_id])
            if disp > 0.5 * 10 ** (-precision):
                changed_atoms += 1
            max_abs_disp = max(max_abs_disp, disp)

    max_raw_bond_dz = 0.0
    max_min_image_bond_length = 0.0
    lx = xhi - xlo
    ly = yhi - ylo
    lz_new = zhi_new - zlo_new
    for a, b in bonds:
        if a not in atom_by_id or b not in atom_by_id:
            continue
        aa = atom_by_id[a]
        bb = atom_by_id[b]
        dx = minimum_image_delta(bb.x - aa.x, lx)
        dy = minimum_image_delta(bb.y - aa.y, ly)
        dz_raw = z_final[b] - z_final[a]
        dz = minimum_image_delta(dz_raw, lz_new)
        max_raw_bond_dz = max(max_raw_bond_dz, abs(dz_raw))
        max_min_image_bond_length = max(max_min_image_bond_length, math.sqrt(dx * dx + dy * dy + dz * dz))

    output_lines = list(lines)
    for idx, line in enumerate(output_lines):
        output_lines[idx] = rewrite_z_bound(line, zlo_new, zhi_new, precision)
    for atom in atoms:
        output_lines[atom.line_index] = rewrite_atom_line(
            atom,
            z_final[atom.atom_id],
            z_image[atom.atom_id],
            precision,
            zero_z_image,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_path.write_text("".join(output_lines), encoding="utf-8")
    tmp_path.replace(output_path)

    return ReboxStats(
        input=str(input_path),
        output=str(output_path),
        n_atoms=len(atoms),
        n_bonds=len(bonds),
        n_components=len(components),
        changed_atoms=changed_atoms,
        crossing_components=crossing_components,
        zlo_old=zlo_old,
        zhi_old=zhi_old,
        zlo_new=zlo_new,
        zhi_new=zhi_new,
        max_abs_z_displacement=max_abs_disp,
        max_raw_bond_dz=max_raw_bond_dz,
        max_min_image_bond_length=max_min_image_bond_length,
    )


def rebox_worker(args: tuple[str, str, float, int, bool, bool]) -> ReboxStats:
    input_path, output_path, target_z_length, precision, overwrite, zero_z_image = args
    return rebox_file(
        Path(input_path),
        Path(output_path),
        target_z_length=target_z_length,
        precision=precision,
        overwrite=overwrite,
        zero_z_image=zero_z_image,
    )


def copy_or_link(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "none":
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown sidecar mode: {mode}")


def copy_sidecars(source_dir: Path, output_dir: Path, mode: str, overwrite: bool) -> int:
    if mode == "none":
        return 0
    copied = 0
    for src in source_dir.iterdir():
        if src.is_dir() or FRAME_DATA_RE.match(src.name):
            continue
        dst = output_dir / src.name
        copy_or_link(src, dst, mode, overwrite)
        copied += 1
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True, help="Glob for input LAMMPS data frames.")
    parser.add_argument("--output-dir", required=True, help="Directory for reboxed frames.")
    parser.add_argument("--target-z-length", type=float, default=100.0)
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N sorted frames.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--sidecars",
        choices=("none", "hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to copy non-data files and matching force arrays into the output directory.",
    )
    parser.add_argument(
        "--preserve-z-image-flags",
        action="store_true",
        help="Do not reset z image flags to the wrapped new-z-box image.",
    )
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = sorted(Path(path) for path in glob.glob(args.input_glob))
    if args.limit is not None:
        input_paths = input_paths[: args.limit]
    if not input_paths:
        raise SystemExit(f"No inputs matched {args.input_glob!r}")

    source_dirs = {path.parent for path in input_paths}
    if len(source_dirs) != 1:
        raise SystemExit("All input frames must be in one directory for sidecar copying")
    source_dir = next(iter(source_dirs))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_sidecars = copy_sidecars(source_dir, output_dir, args.sidecars, args.overwrite)
    worker_args = [
        (
            str(path),
            str(output_dir / path.name),
            args.target_z_length,
            args.precision,
            args.overwrite,
            not args.preserve_z_image_flags,
        )
        for path in input_paths
    ]

    if args.workers == 1:
        stats = [rebox_worker(item) for item in worker_args]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            stats = list(pool.map(rebox_worker, worker_args, chunksize=8))

    aggregate = {
        "n_files": len(stats),
        "copied_sidecars": copied_sidecars,
        "input_glob": args.input_glob,
        "output_dir": str(output_dir),
        "target_z_length": args.target_z_length,
        "zlo_new_min": min(s.zlo_new for s in stats),
        "zhi_new_max": max(s.zhi_new for s in stats),
        "changed_atoms_total": sum(s.changed_atoms for s in stats),
        "crossing_components_total": sum(s.crossing_components for s in stats),
        "max_abs_z_displacement": max(s.max_abs_z_displacement for s in stats),
        "max_raw_bond_dz": max(s.max_raw_bond_dz for s in stats),
        "max_min_image_bond_length": max(s.max_min_image_bond_length for s in stats),
        "files": [asdict(s) for s in stats],
    }

    summary_path = Path(args.summary_json) if args.summary_json else output_dir / "rebox_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in aggregate.items() if k != "files"}, indent=2, sort_keys=True))
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
