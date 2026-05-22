"""AceCG io trajectory implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import MDAnalysis as mda

from .logger import get_screen_logger


logger = get_screen_logger("trajectory")


# ── Type aliases ───────────────────────────────────────────────

FrameSpec = tuple[int, np.ndarray, np.ndarray]
"""(frame_id, positions, box) tuple for one trajectory frame."""

FrameMap = Mapping[int, Sequence[FrameSpec]]
"""Mapping from conditioning frame id → sequence of sample FrameSpecs."""


def load_dump_positions(trajectory_file: Path) -> np.ndarray:
    """Read positions from a LAMMPS dump file using numpy (no MDAnalysis).

    ~8x faster than MDAnalysis for position-only extraction.
    Assumes ``dump_modify sort id`` was used (atoms sorted by ID in
    each frame) and columns include ``x y z``.

    Returns
    -------
    positions : ndarray, shape (n_frames, n_atoms, 3)
    """
    with open(trajectory_file, "r") as f:
        all_lines = f.readlines()
    total_lines = len(all_lines)
    if total_lines < 9:
        raise ValueError(f"Dump file too short: {total_lines} lines")
    n_atoms = int(all_lines[3].strip())
    # TIMESTEP, N, BOX BOUNDS (3 lines), ATOMS header
    header_lines = 9
    lines_per_frame = header_lines + n_atoms
    if total_lines < lines_per_frame:
        raise ValueError(
            f"Dump file too short: {total_lines} lines, expected at least {lines_per_frame}"
        )
    n_frames = total_lines // lines_per_frame
    if total_lines % lines_per_frame != 0:
        raise ValueError(
            f"Dump file has {total_lines} lines, not divisible by {lines_per_frame} "
            f"(header={header_lines} + n_atoms={n_atoms})"
        )
    atoms_header = all_lines[header_lines - 1].strip()
    if not atoms_header.startswith("ITEM: ATOMS"):
        raise ValueError(f"Expected 'ITEM: ATOMS' at line {header_lines}, got: {atoms_header}")
    # skip "ITEM:" and "ATOMS"
    columns = atoms_header.split()[2:]
    try:
        col_x = columns.index("x")
        col_y = columns.index("y")
        col_z = columns.index("z")
    except ValueError:
        for prefix in ("xu", "xs"):
            if prefix in columns:
                col_x = columns.index(prefix)
                col_y = columns.index(prefix.replace("x", "y"))
                col_z = columns.index(prefix.replace("x", "z"))
                break
        else:
            raise ValueError(f"Cannot find position columns in dump header: {atoms_header}")
    atom_lines: List[str] = []
    for frame_idx in range(n_frames):
        start = frame_idx * lines_per_frame + header_lines
        atom_lines.extend(all_lines[start : start + n_atoms])
    data = np.loadtxt(atom_lines, dtype=np.float64, usecols=(col_x, col_y, col_z))
    return data.reshape(n_frames, n_atoms, 3)


def count_lammpstrj_frames_and_atoms(trj_path: Path) -> Tuple[int, int]:
    """
    Count number of frames and atoms per frame for a standard LAMMPS dump (lammpstrj).

    Assumptions:
      - Each frame starts with:
            ITEM: TIMESTEP
            <timestep>
            ITEM: NUMBER OF ATOMS
            <natoms>
            ITEM: BOX BOUNDS ...
            <3 lines>
            ITEM: ATOMS ...
            <natoms lines>
      - natoms is constant across frames (typical for most MD trajectories).

    Returns
    -------
    n_frames : int
    natoms   : int
    """
    n_frames = 0
    natoms: Optional[int] = None

    with trj_path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if line.startswith("ITEM: TIMESTEP"):
                # timestep value line
                if not f.readline():
                    break

                # NUMBER OF ATOMS
                line2 = f.readline()
                if not line2:
                    break
                if not line2.startswith("ITEM: NUMBER OF ATOMS"):
                    raise ValueError(
                        f"Unexpected format near frame {n_frames}: expected 'ITEM: NUMBER OF ATOMS', got: {line2.strip()}"
                    )
                nline = f.readline()
                if not nline:
                    break
                this_natoms = int(nline.strip())
                if natoms is None:
                    natoms = this_natoms
                elif this_natoms != natoms:
                    raise ValueError(
                        f"natoms changes across frames: first={natoms}, now={this_natoms} at frame {n_frames}"
                    )

                # BOX BOUNDS header + 3 lines
                boxhdr = f.readline()
                if not boxhdr:
                    break
                if not boxhdr.startswith("ITEM: BOX BOUNDS"):
                    raise ValueError(
                        f"Unexpected format near frame {n_frames}: expected 'ITEM: BOX BOUNDS', got: {boxhdr.strip()}"
                    )
                for _ in range(3):
                    if not f.readline():
                        raise ValueError("Unexpected EOF while reading BOX BOUNDS lines")

                # ATOMS header + natoms lines
                atomshdr = f.readline()
                if not atomshdr:
                    break
                if not atomshdr.startswith("ITEM: ATOMS"):
                    raise ValueError(
                        f"Unexpected format near frame {n_frames}: expected 'ITEM: ATOMS', got: {atomshdr.strip()}"
                    )
                for _ in range(this_natoms):
                    if not f.readline():
                        raise ValueError("Unexpected EOF while reading ATOMS lines")

                n_frames += 1

    if natoms is None:
        raise ValueError("No frames detected. Is this a valid lammpstrj?")

    return n_frames, natoms


def split_lammpstrj(
    trj_path: str | Path,
    n_parts: int,
    out_dir: str | Path,
    prefix: str = "part",
    digits: int = 3,
    dry_run: bool = False,
) -> List[Path]:
    """
    Split a LAMMPS lammpstrj (dump custom style) into `n_parts` approximately equal segments by frame count.

    Why text streaming:
      - O(1) memory, avoids MDAnalysis indexing overhead for huge trajectories.
      - Works even if you don't have topology files.

    Parameters
    ----------
    trj_path : str | Path
        Input lammpstrj file.
    n_parts : int
        Number of segments to split into (must be >= 1).
    out_dir : str | Path
        Output directory.
    prefix : str
        Output file prefix, default "part".
    digits : int
        Zero-padding digits for part index, default 3 -> part_000.lammpstrj.
    dry_run : bool
        If True, do not write files; just return planned output paths.

    Returns
    -------
    out_paths : List[Path]
        List of output file paths in order.

    Notes
    -----
    - Assumes a standard lammpstrj frame structure and constant natoms across frames.
    - If your dump format is unusual (variable natoms, additional sections), we can extend the parser.
    """
    trj_path = Path(trj_path)
    out_dir = Path(out_dir)
    if n_parts < 1:
        raise ValueError("n_parts must be >= 1")
    if not trj_path.exists():
        raise FileNotFoundError(trj_path)

    n_frames, _ = count_lammpstrj_frames_and_atoms(trj_path)

    # Distribute frames as evenly as possible across parts
    base = n_frames // n_parts
    rem = n_frames % n_parts
    frames_per_part = [base + (1 if i < rem else 0) for i in range(n_parts)]
    # If n_parts > n_frames, some parts will be 0 frames; that’s usually undesirable.
    # We'll still create empty files only if writing is requested.
    out_paths = [out_dir / f"{prefix}_{i:0{digits}d}.lammpstrj" for i in range(n_parts)]
    if dry_run:
        return out_paths

    out_dir.mkdir(parents=True, exist_ok=True)

    # Stream through input again, writing frame-by-frame
    part_idx = 0
    frames_written_in_part = 0
    target_in_part = frames_per_part[part_idx] if n_parts > 0 else 0

    out_f = out_paths[part_idx].open("w", encoding="utf-8")
    try:
        with trj_path.open("r", encoding="utf-8", errors="replace") as in_f:
            while True:
                line = in_f.readline()
                if not line:
                    break

                if not line.startswith("ITEM: TIMESTEP"):
                    # Skip any leading junk safely (shouldn't happen in normal dumps)
                    continue

                # Read the full frame into a small list (size ~ natoms lines, manageable)
                # ITEM: TIMESTEP
                frame_lines = [line]
                # timestep value line
                for _ in range(1):
                    nxt = in_f.readline()
                    if not nxt:
                        raise ValueError("Unexpected EOF after ITEM: TIMESTEP")
                    frame_lines.append(nxt)

                # NUMBER OF ATOMS header + value
                for _ in range(2):
                    nxt = in_f.readline()
                    if not nxt:
                        raise ValueError("Unexpected EOF while reading NUMBER OF ATOMS section")
                    frame_lines.append(nxt)
                natoms = int(frame_lines[-1].strip())

                # BOX BOUNDS header + 3 lines
                nxt = in_f.readline()
                if not nxt:
                    raise ValueError("Unexpected EOF while reading BOX BOUNDS header")
                frame_lines.append(nxt)
                for _ in range(3):
                    nxt = in_f.readline()
                    if not nxt:
                        raise ValueError("Unexpected EOF while reading BOX BOUNDS lines")
                    frame_lines.append(nxt)

                # ATOMS header + natoms lines
                nxt = in_f.readline()
                if not nxt:
                    raise ValueError("Unexpected EOF while reading ATOMS header")
                frame_lines.append(nxt)
                for _ in range(natoms):
                    nxt = in_f.readline()
                    if not nxt:
                        raise ValueError("Unexpected EOF while reading ATOMS lines")
                    frame_lines.append(nxt)

                # If current part has 0 target frames, advance to a non-zero one
                while part_idx < n_parts and target_in_part == 0:
                    out_f.close()
                    part_idx += 1
                    if part_idx >= n_parts:
                        break
                    out_f = out_paths[part_idx].open("w", encoding="utf-8")
                    frames_written_in_part = 0
                    target_in_part = frames_per_part[part_idx]

                if part_idx >= n_parts:
                    # More frames than expected (shouldn't happen); ignore remainder
                    break

                out_f.writelines(frame_lines)
                frames_written_in_part += 1

                if frames_written_in_part >= target_in_part:
                    out_f.close()
                    part_idx += 1
                    if part_idx >= n_parts:
                        break
                    out_f = out_paths[part_idx].open("w", encoding="utf-8")
                    frames_written_in_part = 0
                    target_in_part = frames_per_part[part_idx]
    finally:
        try:
            out_f.close()
        except Exception:
            pass

    return out_paths


def split_lammpstrj_mdanalysis(
    trj_path: str | Path,
    n_parts: int,
    out_dir: str | Path,
    *,
    topology: Optional[str | Path] = None,
    select: Optional[str] = None,
    prefix: str = "part",
    digits: int = 3,
    output_fields: Sequence[str] = ("id", "type", "x", "y", "z"),
    reorder_by: Optional[str] = "id",
    verbose: bool = True,
) -> List[Path]:
    """
    Split a LAMMPS lammpstrj using MDAnalysis, optionally writing only a subset of atoms.

    Parameters
    ----------
    trj_path : str | Path
        Input LAMMPS dump trajectory (.lammpstrj).
    n_parts : int
        Number of parts to split into (>=1).
    out_dir : str | Path
        Output directory.
    topology : str | Path, optional
        Optional topology file if needed by your setup (often not needed for dump readers).
        Examples: a LAMMPS data file, PDB, etc.
    select : str, optional
        MDAnalysis selection string, e.g.:
          - "protein"
          - "name CA"
          - "type 1 2 3"
          - "resid 10-50"
        If None, write all atoms.
    prefix : str
        Output prefix, default "part".
    digits : int
        Zero-padding digits, default 3.
    output_fields : Sequence[str]
        Fields to write in "ITEM: ATOMS ..." line. Typical: ("id","type","x","y","z").
        Must be compatible with the dump content MDAnalysis parses.
    reorder_by : str, optional
        If provided, reorder atoms each frame by this attribute (e.g. "id") before writing.
        Use None to keep current order.
    verbose : bool
        Print basic progress / summary.

    Returns
    -------
    out_paths : List[Path]
        Output file paths.

    Notes
    -----
    - This writer emits a standard lammpstrj with:
        ITEM: TIMESTEP
        ITEM: NUMBER OF ATOMS
        ITEM: BOX BOUNDS ...
        ITEM: ATOMS <output_fields...>
      and then the selected atoms.
    - If you need to preserve *exact* original extra columns or nonstandard dump sections,
      use the pure-text splitter.
    """
    trj_path = Path(trj_path)
    out_dir = Path(out_dir)
    if n_parts < 1:
        raise ValueError("n_parts must be >= 1")
    if not trj_path.exists():
        raise FileNotFoundError(trj_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = [out_dir / f"{prefix}_{i:0{digits}d}.lammpstrj" for i in range(n_parts)]

    # Build Universe
    if topology is None:
        u = mda.Universe(str(trj_path), format="LAMMPSDUMP")
    else:
        u = mda.Universe(str(topology), str(trj_path), format="LAMMPSDUMP")

    n_frames = len(u.trajectory)
    if n_frames == 0:
        raise ValueError("Trajectory has 0 frames.")

    # Even frame distribution
    base = n_frames // n_parts
    rem = n_frames % n_parts
    frames_per_part = [base + (1 if i < rem else 0) for i in range(n_parts)]

    if verbose:
        logger.info(
            "split_lammpstrj_mdanalysis frames=%d parts=%d frames_per_part=%s",
            n_frames,
            n_parts,
            frames_per_part,
        )

    # Atom selection
    ag_all = u.atoms
    ag = ag_all.select_atoms(select) if select else ag_all

    # Prepare output writers (plain text)
    def _write_frame_header(f, ts, natoms: int, box: np.ndarray, atoms_header: str):
        f.write("ITEM: TIMESTEP\n")
        # LAMMPS timestep is integer-like; MDAnalysis stores in ts.frame/ts.time depending on reader.
        # For dump, ts.data.get('timestep') is often present; fallback to frame index.
        timestep = ts.data.get("timestep", ts.frame)
        f.write(f"{int(timestep)}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{natoms}\n")

        # BOX BOUNDS: infer triclinic flags if present
        # MDAnalysis ts.dimensions = [lx, ly, lz, alpha, beta, gamma]
        # For LAMMPS dump, ts.triclinic_dimensions may exist, but not always.
        # We'll write orthorhombic bounds if we can't infer tilt.
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        # For safety, output as 0..L bounds from dimensions
        # (lx, ly, lz, alpha, beta, gamma)
        dims = ts.dimensions
        lx, ly, lz = float(dims[0]), float(dims[1]), float(dims[2])
        f.write(f"0.0 {lx:.8f}\n")
        f.write(f"0.0 {ly:.8f}\n")
        f.write(f"0.0 {lz:.8f}\n")

        f.write(atoms_header)

    # Writer for one frame
    def _write_atoms(f, ag_frame, fields: Sequence[str]):
        # Optional reorder
        if reorder_by:
            if hasattr(ag_frame, reorder_by):
                key = getattr(ag_frame, reorder_by)
                order = np.argsort(np.asarray(key))
                agw = ag_frame[order]
            else:
                agw = ag_frame
        else:
            agw = ag_frame

        # Build columns
        cols = []
        for fld in fields:
            if fld in ("x", "y", "z"):
                # positions
                pass
            else:
                if hasattr(agw, fld):
                    cols.append(np.asarray(getattr(agw, fld)))
                else:
                    raise ValueError(
                        f"Requested field '{fld}' not available on AtomGroup. "
                        f"Available examples: id, type, resid, resname, name, ..."
                    )

        # (N,3), in Angstrom
        pos = agw.positions
        # Compose line-by-line to avoid huge intermediate strings
        # Determine index mapping in fields
        # We'll handle x/y/z by reading pos
        for i in range(len(agw)):
            parts = []
            for fld in fields:
                if fld == "x":
                    parts.append(f"{pos[i,0]:.8f}")
                elif fld == "y":
                    parts.append(f"{pos[i,1]:.8f}")
                elif fld == "z":
                    parts.append(f"{pos[i,2]:.8f}")
                else:
                    val = getattr(agw, fld)[i]
                    # ids/types are usually ints
                    if isinstance(val, (np.integer, int)):
                        parts.append(str(int(val)))
                    else:
                        parts.append(str(val))
            f.write(" ".join(parts) + "\n")

    atoms_header = "ITEM: ATOMS " + " ".join(output_fields) + "\n"

    # Iterate and write parts
    frame_idx = 0
    for part_idx, n_in_part in enumerate(frames_per_part):
        out_path = out_paths[part_idx]
        with out_path.open("w", encoding="utf-8") as f:
            if verbose:
                logger.info(
                    "writing %s: %d frames, selection=%r atoms=%d",
                    out_path.name,
                    n_in_part,
                    select or "ALL",
                    len(ag),
                )
            for _ in range(n_in_part):
                ts = u.trajectory[frame_idx]
                # Update selection positions (selection is "live", but safest to reselect if topology changes; usually unnecessary)
                ag_frame = u.atoms.select_atoms(select) if select else u.atoms
                _write_frame_header(f, ts, len(ag_frame), ts.dimensions, atoms_header)
                _write_atoms(f, ag_frame, output_fields)
                frame_idx += 1

    return out_paths


# ── LAMMPS dump frame reader ───────────────────────────────────────────────

def _coerce_atom_field(values: Sequence[str]) -> np.ndarray:
    """Convert a dump column to int, float, or object dtype."""
    if not values:
        return np.empty(0, dtype=object)
    for dtype in (np.int64, np.float64):
        try:
            return np.asarray(values, dtype=dtype)
        except ValueError:
            continue
    return np.asarray(values, dtype=object)


def _read_requested_custom_atom_fields(
    trajectory_path: Path,
    frame_ids: Sequence[int],
    custom_atom_fields: Sequence[str],
) -> Dict[int, Dict[str, np.ndarray]]:
    requested = sorted({int(frame_id) for frame_id in frame_ids})
    requested_set = set(requested)
    field_names = tuple(dict.fromkeys(str(name) for name in custom_atom_fields))
    records: Dict[int, Dict[str, np.ndarray]] = {}
    frame_index = -1

    with trajectory_path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"Unexpected LAMMPS dump header {line.strip()!r}")

            if not handle.readline():
                raise ValueError("Unexpected EOF after ITEM: TIMESTEP")
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Expected ITEM: NUMBER OF ATOMS")
            n_atoms = int(handle.readline().strip())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Expected ITEM: BOX BOUNDS")
            for _ in range(3):
                if not handle.readline():
                    raise ValueError("Unexpected EOF while reading BOX BOUNDS")

            atoms_header = handle.readline().strip()
            if not atoms_header.startswith("ITEM: ATOMS "):
                raise ValueError("Expected ITEM: ATOMS header")
            columns = atoms_header.split()[2:]
            frame_index += 1
            keep = frame_index in requested_set
            if not keep:
                for _ in range(n_atoms):
                    if not handle.readline():
                        raise ValueError("Unexpected EOF while skipping ATOMS lines")
                continue

            missing = [name for name in field_names if name not in columns]
            if missing:
                raise KeyError(f"Missing requested custom atom fields {missing}")
            indices = {name: int(columns.index(name)) for name in field_names}
            atom_id_idx = int(columns.index("id")) if "id" in columns else -1
            data = {name: [] for name in field_names}
            atom_ids: List[int] = []

            for _ in range(n_atoms):
                parts = handle.readline().split()
                if atom_id_idx >= 0:
                    atom_ids.append(int(parts[atom_id_idx]))
                for name, idx in indices.items():
                    data[name].append(parts[idx])

            order = None
            if atom_ids and not np.array_equal(
                np.asarray(atom_ids, dtype=np.int64),
                np.arange(1, n_atoms + 1, dtype=np.int64),
            ):
                order = np.argsort(np.asarray(atom_ids, dtype=np.int64))

            record = {name: _coerce_atom_field(values) for name, values in data.items()}
            if order is not None:
                record = {name: values[order] for name, values in record.items()}
            records[frame_index] = record

            if len(records) == len(requested):
                break

    missing_frames = [frame_id for frame_id in requested if frame_id not in records]
    if missing_frames:
        raise KeyError(f"Missing requested frames in trajectory: {missing_frames}")
    return records


def read_lammpstrj_frames(
    trajectory_path,
    frame_ids: Sequence[int],
    *,
    custom_atom_fields: Optional[Sequence[str]] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Read positions, forces, and dimensions for requested frames from a LAMMPS dump.

    Parameters
    ----------
    trajectory_path : str or Path
        Path to the LAMMPS trajectory file.
    frame_ids : sequence of int
        Frame indices (0-based) to read.
    custom_atom_fields : sequence of str, optional
        Explicit nonstandard per-atom dump columns to read via the AceCG text
        parser. Standard positions/forces/box data still come from MDAnalysis.

    Returns
    -------
    dict
        ``{frame_id: {"positions": (N,3), "forces": (N,3), "dimensions": (6,), "timestep": int}}``
        with optional ``"atom_fields"`` for explicit custom-column requests.
    """
    requested = sorted({int(frame_id) for frame_id in frame_ids})
    if not requested:
        raise ValueError("frame_ids must not be empty")
    trajectory = Path(trajectory_path)
    custom_field_records: Dict[int, Dict[str, np.ndarray]] = {}
    if custom_atom_fields:
        custom_field_records = _read_requested_custom_atom_fields(
            trajectory,
            requested,
            custom_atom_fields,
        )

    u = mda.Universe(str(trajectory), format="LAMMPSDUMP")
    records: Dict[int, Dict[str, Any]] = {}
    for frame_id in requested:
        ts = u.trajectory[int(frame_id)]
        if not bool(getattr(ts, "has_forces", False)):
            raise KeyError("Trajectory is missing required force columns fx/fy/fz")
        records[frame_id] = {
            "frame_id": int(frame_id),
            "timestep": int(ts.data.get("step", ts.frame)),
            "positions": np.asarray(ts.positions, dtype=np.float64).copy(),
            "forces": np.asarray(ts.forces, dtype=np.float64).copy(),
            "dimensions": np.asarray(ts.dimensions, dtype=np.float64).copy(),
        }
        if custom_field_records:
            records[frame_id]["atom_fields"] = custom_field_records[frame_id]
    return {frame_id: records[frame_id] for frame_id in requested}


# ── Frame extraction helpers ───────────────────────────────────────────────

def iter_frames(
    universe: mda.Universe,
    *,
    start: int = 0,
    end: Optional[int] = None,
    every: int = 1,
    frame_ids: Optional[Sequence[int]] = None,
    include_forces: bool = False,
):
    """
    Iterator for frame extraction from an MDAnalysis Universe trajectory.
    Yield one engine frame at a time. 

    This is the
    canonical source for one-pass post-processing. It never stores more than
    one frame worth of positions / box / forces in memory.

    When *frame_ids* is given, seek to each listed frame index (O(1) random
    access in LAMMPSDUMP) and ignore *start*/*end*/*every*.
    """
    include_forces = bool(include_forces) and bool(
        getattr(universe.trajectory.ts, "has_forces", False)
    )

    if frame_ids is not None:
        indices = frame_ids
    else:
        if end is None:
            end = len(universe.trajectory)
        indices = range(start, end, every)

    for idx in indices:
        ts = universe.trajectory[idx]
        fid = ts.frame
        pos = universe.atoms.positions.copy().astype(np.float32)
        dim = universe.dimensions
        box = (
            dim.copy().astype(np.float32)
            if dim is not None
            else np.zeros(6, dtype=np.float32)
        )
        if include_forces:
            forces = ts.forces.copy().astype(np.float32).ravel()
        else:
            forces = None
        yield fid, pos, box, forces
