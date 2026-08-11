# AceCG/io/trajectory.py
"""Trajectory I/O: readers, splitters, and the canonical frame iterator.

Everything that turns a file on disk into per-frame arrays lives here, so a
workflow never has to know how MDAnalysis was opened or how frames were
distributed. Three layers, from lowest to highest:

``iter_frames``
    The one-frame-at-a-time iterator. Yields a :class:`FrameRecord` — a
    fixed-key frozen mapping, *not* a positional tuple, so a caller that wants
    only positions is unaffected when a new field is added.
``open_universe`` / ``select_frame_ids`` / ``partition_frame_ids``
    The three steps every MPI producer repeats: open, choose frames, split them
    into balanced contiguous per-rank slices.
:class:`MPITrajReader`
    Those steps as one plan object, including the rank-0 scan whose result
    (frame count + XDR offsets) is broadcast so no rank rescans the file.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import MDAnalysis as mda

from .logger import get_screen_logger


logger = get_screen_logger("trajectory")


GROMACS_KJ_PER_KCAL = 4.184
"""Exact kJ-per-kcal factor used at AceCG reference-force boundaries."""


def reference_force_scale_to_lammps_real(
    trajectory_format: Optional[str],
) -> float:
    """Scale MDAnalysis reference forces into AceCG/LAMMPS-real units.

    A TRR stores force in GROMACS ``kJ/(mol*nm)`` units.  MDAnalysis converts
    its length denominator to Angstrom but leaves the energy unit as kJ, so an
    FM consumer receives ``kJ/(mol*Angstrom)`` and must divide by 4.184.  Other
    currently supported force-bearing inputs are already authored in AceCG's
    native ``kcal/(mol*Angstrom)`` convention.

    This is deliberately an explicit consumer-side scale, not an automatic
    :func:`iter_frames` conversion: trajmap must preserve GROMACS units when it
    maps one TRR and writes another TRR.
    """
    normalized = "" if trajectory_format is None else str(trajectory_format)
    if normalized.strip().upper() == "TRR":
        return 1.0 / GROMACS_KJ_PER_KCAL
    return 1.0


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FrameSpec = tuple[int, np.ndarray, np.ndarray]
"""(frame_id, positions, box) tuple for one trajectory frame."""

FrameMap = Mapping[int, Sequence[FrameSpec]]
"""Mapping from conditioning frame id → sequence of sample FrameSpecs."""


# ---------------------------------------------------------------------------
# The frame record
# ---------------------------------------------------------------------------

FRAME_FIELDS: Tuple[str, ...] = (
    "frame_id",
    "positions",
    "box",
    "forces",
    "velocities",
    "time",
    "dt",
    "step",
)
"""Every key a :class:`FrameRecord` carries, in declaration order."""


@dataclass(frozen=True, eq=False)
class FrameRecord(Mapping):
    """One trajectory frame as a fixed-key, immutable mapping.

    The keys are always exactly :data:`FRAME_FIELDS`; a field that the caller
    did not ask :func:`iter_frames` for is ``None``. That is the whole contract:
    downstream code reads the fields it needs (``frame["positions"]`` or
    ``frame.positions``, both work) and never has to know how many other fields
    exist. The old ``(frame_id, positions, box, forces)`` tuple could not grow a
    field without touching every unpacking site, and made "not requested"
    indistinguishable from "position 3".

    ``forces``/``velocities`` are ``None`` both when they were not requested and
    when the trajectory cannot supply them, so a caller that requires them must
    check — see :meth:`require`.

    Attributes
    ----------
    frame_id : int
        0-based index of the frame within its (possibly chained) trajectory.
    positions : np.ndarray, shape (n_atoms, 3)
        Atom positions, Å.
    box : np.ndarray, shape (6,) or None
        ``[lx, ly, lz, alpha, beta, gamma]``. All-zero when the trajectory
        carries no cell.
    forces, velocities : np.ndarray or None
        Flat ``(3 * n_atoms,)`` arrays, matching what the compute kernels take.
    time, dt : float or None
        Frame time and inter-frame spacing in the trajectory's own units (ps for
        GROMACS containers).
    step : int or None
        Integrator step recorded by the writer, when it recorded one.
    """

    frame_id: int
    positions: np.ndarray
    box: Optional[np.ndarray] = None
    forces: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None
    time: Optional[float] = None
    dt: Optional[float] = None
    step: Optional[int] = None

    # ── Mapping interface ─────────────────────────────────────────────
    # Implemented so a FrameRecord *is* the promised frozen dict: `record[key]`,
    # `dict(record)`, `**record`, `"forces" in record` all work. Iteration walks
    # the keys, like any mapping, which also makes the removed 4-tuple unpacking
    # fail loudly instead of silently binding strings.

    def __getitem__(self, key: str) -> Any:
        if key not in FRAME_FIELDS:
            raise KeyError(
                f"{key!r} is not a frame field; FrameRecord always carries exactly "
                f"{list(FRAME_FIELDS)}."
            )
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(FRAME_FIELDS)

    def __len__(self) -> int:
        return len(FRAME_FIELDS)

    def require(self, *names: str) -> Tuple[Any, ...]:
        """Return the named fields, raising when any of them is ``None``.

        Use at the top of a consumer that cannot proceed without a field, so the
        error names the missing field instead of surfacing as ``NoneType`` deep
        inside a kernel.
        """
        missing = [name for name in names if self[name] is None]
        if missing:
            raise KeyError(
                f"frame {self.frame_id} carries no {missing}; the trajectory does "
                "not provide it, or iter_frames was not asked for it."
            )
        return tuple(self[name] for name in names)

    def replace(self, **changes: Any) -> "FrameRecord":
        """A copy with some fields replaced."""
        known = {field.name for field in dataclass_fields(self)}
        unknown = set(changes) - known
        if unknown:
            raise KeyError(f"unknown frame field(s) {sorted(unknown)}.")
        merged = {name: getattr(self, name) for name in known}
        merged.update(changes)
        return FrameRecord(**merged)


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
    header_lines = 9  # TIMESTEP, N, BOX BOUNDS (3 lines), ATOMS header
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
    columns = atoms_header.split()[2:]  # skip "ITEM:" and "ATOMS"
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
                frame_lines = [line]  # ITEM: TIMESTEP
                for _ in range(1):  # timestep value line
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
        dims = ts.dimensions  # (lx, ly, lz, alpha, beta, gamma)
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

        pos = agw.positions  # (N,3), in Angstrom
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


# ---------------------------------------------------------------------------
# Generic LAMMPS dump frame reader (migrated from cdfm_realdata.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Frame extraction helpers (moved from compute/postprocess.py)
# ---------------------------------------------------------------------------

def iter_frames(
    universe: mda.Universe,
    *,
    start: int = 0,
    end: Optional[int] = None,
    every: int = 1,
    frame_ids: Optional[Sequence[int]] = None,
    include_forces: bool = False,
    include_velocities: bool = False,
    include_time: bool = False,
    atom_indices: Optional[Sequence[int]] = None,
) -> Iterator[FrameRecord]:
    """Yield one :class:`FrameRecord` at a time from a Universe trajectory.

    The canonical frame source for every AceCG consumer — post-processing,
    VP growth, trajectory mapping, RDFs. It never holds more than one frame
    worth of arrays, so trajectory length does not bound memory.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Trajectory to read. Frames are visited by random access, which is O(1)
        for LAMMPSDUMP and for XDR readers with installed offsets.
    start, end, every : int
        Contiguous window ``range(start, end, every)``; ``end`` defaults to the
        trajectory length.
    frame_ids : sequence of int, optional
        Explicit frame indices. Takes precedence over the window.
    include_forces, include_velocities : bool, default False
        Also read forces / velocities, flattened to ``(3 * n_atoms,)``. Silently
        left at ``None`` when the trajectory carries none, so a caller that
        requires them should use :meth:`FrameRecord.require`.
    include_time : bool, default False
        Also read ``time``, ``dt`` and the writer's integrator ``step``. Cheap,
        but off by default so an unrequested field stays ``None``.
    atom_indices : sequence of int, optional
        Gather only these atoms, in this order, instead of the whole frame. The
        reader still pulls the full frame off disk — that is the container's
        business — but a consumer that needs a fraction of a solvated system stops
        copying the rest: a 12-bead lipid mapping of a 1.19-million-atom box reads
        599,040 atoms, so this halves the per-frame copy and the ``forces``
        payload. ``positions``/``forces``/``velocities`` then have this length,
        and index ``k`` of the record is ``atom_indices[k]`` of the frame.

    Yields
    ------
    FrameRecord
        Fixed-key frozen mapping; unrequested fields are ``None``.
    """
    include_forces = bool(include_forces) and bool(
        getattr(universe.trajectory.ts, "has_forces", False)
    )
    include_velocities = bool(include_velocities) and bool(
        getattr(universe.trajectory.ts, "has_velocities", False)
    )
    include_time = bool(include_time)

    if frame_ids is not None:
        indices: Iterable[int] = frame_ids
    else:
        if end is None:
            end = len(universe.trajectory)
        indices = range(start, end, every)

    take: Optional[np.ndarray] = None
    if atom_indices is not None:
        take = np.asarray(atom_indices, dtype=np.intp).ravel()
        n_atoms = int(universe.atoms.n_atoms)
        if take.size and (int(take.min()) < 0 or int(take.max()) >= n_atoms):
            raise ValueError(
                f"atom_indices spans [{int(take.min())}, {int(take.max())}] but the "
                f"trajectory has {n_atoms} atoms."
            )

    def _subset(values: np.ndarray) -> np.ndarray:
        """One per-atom vector field as float32, gathered if a subset was asked."""
        if take is None:
            return values.copy().astype(np.float32)
        return np.take(values, take, axis=0).astype(np.float32, copy=False)

    for idx in indices:
        ts = universe.trajectory[int(idx)]
        dim = universe.dimensions
        box = (
            dim.copy().astype(np.float32)
            if dim is not None
            else np.zeros(6, dtype=np.float32)
        )
        step: Optional[int] = None
        if include_time:
            raw_step = ts.data.get("step") if hasattr(ts, "data") else None
            if raw_step is None and hasattr(ts, "data"):
                raw_step = ts.data.get("timestep")
            step = None if raw_step is None else int(raw_step)
        yield FrameRecord(
            frame_id=int(ts.frame),
            positions=_subset(universe.atoms.positions),
            box=box,
            forces=_subset(ts.forces).ravel() if include_forces else None,
            velocities=_subset(ts.velocities).ravel() if include_velocities else None,
            time=float(getattr(ts, "time", 0.0)) if include_time else None,
            dt=float(getattr(ts, "dt", 0.0) or 0.0) if include_time else None,
            step=step,
        )


# ---------------------------------------------------------------------------
# Shared MPI trajectory reading
# ---------------------------------------------------------------------------
#
# MPIComputeEngine.run_post(), the VP grower and trajmap all had to answer the
# same four questions: how do I open this trajectory, which frames do I want,
# which of them are mine, and how do I read them. The answers below are the
# single implementation; `iter_frames` above answers the fourth.


def format_for_path(
    path: Optional[Union[str, Path]],
    *,
    explicit_format: Optional[str] = None,
) -> Optional[str]:
    """Canonical MDAnalysis format keyword from an override or file suffix.

    A non-blank explicit format takes precedence. Blank and ``auto`` request
    suffix inference. Unknown suffixes remain ``None`` so MDAnalysis can infer
    formats outside AceCG's known set.
    """
    if explicit_format is not None:
        text = str(explicit_format).strip()
        if text and text.lower() != "auto":
            aliases = {
                "lammpstrj": "LAMMPSDUMP",
                "lammpsdump": "LAMMPSDUMP",
                "lammps_dump": "LAMMPSDUMP",
                "dump": "LAMMPSDUMP",
                "xtc": "XTC",
                "dcd": "DCD",
                "h5md": "H5MD",
                "trr": "TRR",
                "xyz": "XYZ",
            }
            lowered = text.lower()
            if lowered in aliases:
                return aliases[lowered]
            if all(char.isalnum() or char in {"_", "-"} for char in text):
                return text.upper().replace("-", "_")
            return text
    if path is None:
        return None
    suffix = Path(path).suffix.lower()
    if suffix == ".data":
        return "DATA"
    if suffix in (".lammpstrj", ".dump"):
        return "LAMMPSDUMP"
    if suffix == ".trr":
        return "TRR"
    if suffix == ".xtc":
        return "XTC"
    if suffix == ".dcd":
        return "DCD"
    if suffix == ".h5md":
        return "H5MD"
    if suffix == ".xyz":
        return "XYZ"
    return None


def open_universe(
    topology: Optional[Union[str, Path]],
    trajectories: Sequence[Union[str, Path]] = (),
    *,
    trajectory_format: Optional[str] = None,
    topology_format: Optional[str] = None,
    offsets: Optional[Sequence[np.ndarray]] = None,
    **universe_kwargs: Any,
) -> mda.Universe:
    """Open an MDAnalysis Universe, filling in the AceCG conventions.

    Parameters
    ----------
    topology : str or Path, optional
        Topology file. ``None`` uses the first trajectory as its own topology,
        which XTC/TRR/LAMMPSDUMP all support, while still reading *every* segment
        as coordinates.
    trajectories : sequence of str or Path
        Trajectory segments in time order; MDAnalysis chains them.
    trajectory_format, topology_format : str, optional
        Explicit format keywords. Inferred from the suffixes when omitted
        (``.trr`` → ``TRR``, ``.xtc`` → ``XTC``, ``.data`` → ``DATA``,
        ``.lammpstrj``/``.dump`` → ``LAMMPSDUMP``).
    offsets : sequence of np.ndarray, optional
        Frame offsets from :func:`reader_offsets`, one array per XDR reader.
        Installing them skips the offset scan, which MDAnalysis otherwise
        repeats on every open of a trajectory it cannot write a sidecar beside.
    **universe_kwargs
        Passed through, e.g. ``atom_style="id resid type charge x y z"``.

    Returns
    -------
    MDAnalysis.Universe
    """
    paths = [str(path) for path in trajectories]
    trajectory_format = format_for_path(
        paths[0] if paths else None,
        explicit_format=trajectory_format,
    )
    if topology is None:
        if not paths:
            raise ValueError("open_universe needs a topology or at least one trajectory.")
        # `Universe(*paths)` would bind paths[0] as the *topology* and chain only
        # paths[1:] as coordinates, silently dropping the first segment's frames.
        # Naming paths[0] twice makes it the topology and keeps every segment in
        # the coordinate chain.
        universe = mda.Universe(
            paths[0], *paths, format=trajectory_format, **universe_kwargs
        )
    else:
        if topology_format is None:
            topology_format = format_for_path(topology)
        universe = mda.Universe(
            str(topology),
            *paths,
            format=trajectory_format,
            topology_format=topology_format,
            **universe_kwargs,
        )
    if offsets is not None:
        install_offsets(universe, offsets)
    return universe


def xdr_readers(universe: mda.Universe) -> List[Any]:
    """The XTC/TRR readers behind a universe, whether chained or single."""
    trajectory = universe.trajectory
    readers = list(getattr(trajectory, "readers", []) or [trajectory])
    return [reader for reader in readers if hasattr(reader, "_xdr")]


def reader_offsets(universe: mda.Universe) -> Optional[List[np.ndarray]]:
    """Frame offsets of every XDR reader behind ``universe``, when available.

    MDAnalysis persists these in a sidecar next to the trajectory, which fails
    for a read-only archive — it then rescans the whole file on *every* open.
    Collecting them once and broadcasting makes the scan happen once per run
    rather than once per rank.
    """
    readers = xdr_readers(universe)
    if not readers:
        return None
    try:
        return [np.asarray(reader._xdr.offsets).copy() for reader in readers]
    except AttributeError:
        return None


def install_offsets(universe: mda.Universe, offsets: Sequence[np.ndarray]) -> bool:
    """Install previously collected frame offsets. ``True`` when they took."""
    readers = xdr_readers(universe)
    if not readers or len(readers) != len(offsets):
        return False
    for reader, offset in zip(readers, offsets):
        try:
            reader._xdr.set_offsets(np.asarray(offset))
        except AttributeError:
            return False
    return True


def select_frame_ids(
    total_frames: int,
    *,
    start: int = 0,
    end: Optional[int] = None,
    every: int = 1,
    n_frames: int = 0,
    frame_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    """Resolve a frame selection against a known trajectory length.

    Precedence matches every AceCG config: an explicit ``frame_ids`` list wins
    over the ``start``/``every``/``n_frames`` window. ``n_frames = 0`` means "to
    the end"; ``end`` is an alternative upper bound for callers that have one.

    Raises
    ------
    ValueError
        If ``frame_ids`` names a frame outside the trajectory, or the selection
        is empty.
    """
    total = int(total_frames)
    if frame_ids is not None:
        selected = [int(fid) for fid in frame_ids]
        bad = [fid for fid in selected if fid < 0 or fid >= total]
        if bad:
            raise ValueError(
                f"frame_ids lists frame(s) {bad[:5]} outside the trajectory's "
                f"{total} frame(s)."
            )
    else:
        step = int(every)
        if step < 1:
            raise ValueError(f"every must be >= 1, got {step}.")
        first = int(start)
        if first < 0:
            raise ValueError(f"start must be >= 0, got {first}.")
        stop = total if end is None else min(total, int(end))
        requested = int(n_frames)
        if requested > 0:
            stop = min(stop, first + requested * step)
        selected = list(range(first, stop, step))
    if not selected:
        raise ValueError(
            f"frame selection is empty: the trajectory has {total} frame(s) and "
            f"start={start}, every={every}, n_frames={n_frames}, "
            f"frame_ids={'set' if frame_ids is not None else 'unset'}."
        )
    return selected


def partition_frame_ids(
    frame_ids: Sequence[int], *, size: int, rank: int
) -> Tuple[int, int, List[int]]:
    """Split a frame-id list into balanced *contiguous* per-rank slices.

    Contiguity is not incidental: it lets each rank write one trajectory segment
    that concatenates with its neighbours in rank order, and keeps sequential
    reads sequential.

    Returns
    -------
    local_count, local_offset, local_ids
        Slice length, its start index within ``frame_ids``, and the ids.
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}.")
    if not 0 <= rank < size:
        raise ValueError(f"rank must lie in [0, {size}), got {rank}.")
    base, remainder = divmod(len(frame_ids), size)
    local_count = base + (1 if rank < remainder else 0)
    local_offset = rank * base + min(rank, remainder)
    return (
        local_count,
        local_offset,
        [int(fid) for fid in frame_ids[local_offset : local_offset + local_count]],
    )


def map_ids_to_segments(
    frame_ids: Sequence[int],
    trajectory_files: Sequence[Union[str, Path]],
    segment_frame_counts: Sequence[int],
) -> Tuple[List[str], List[int], List[int]]:
    """Reduce a frame-id list to just the segments that hold it.

    A rank owning a contiguous slice of a 40-segment trajectory usually needs
    one or two segments, so opening only those turns a 40-file chained reader
    into a 1-file one.

    Returns
    -------
    paths : list of str
        The needed segments, in trajectory order.
    local_ids : list of int
        ``frame_ids`` re-expressed as indices into the chain of ``paths``.
    segment_numbers : list of int
        1-based indices of the needed segments, for reporting.
    """
    boundaries = np.cumsum(np.asarray(segment_frame_counts, dtype=np.int64))
    included: List[int] = []
    local_offset_of_segment: Dict[int, int] = {}
    running = 0
    local_ids: List[int] = []
    for raw_id in frame_ids:
        frame_id = int(raw_id)
        segment = int(np.searchsorted(boundaries, frame_id, side="right"))
        if segment >= len(segment_frame_counts):
            raise ValueError(
                f"frame {frame_id} lies past the {int(boundaries[-1])} frame(s) of "
                f"{len(segment_frame_counts)} segment(s)."
            )
        segment_start = 0 if segment == 0 else int(boundaries[segment - 1])
        if segment not in local_offset_of_segment:
            local_offset_of_segment[segment] = running
            included.append(segment)
            running += int(segment_frame_counts[segment])
        local_ids.append(local_offset_of_segment[segment] + frame_id - segment_start)
    return (
        [str(trajectory_files[index]) for index in included],
        local_ids,
        [index + 1 for index in included],
    )


UNIVERSE_STRATEGIES: Tuple[str, ...] = (
    "auto",
    "reopen",
    "broadcast",
    "local_segments",
)
"""Universe-construction strategies :class:`MPITrajReader` accepts."""


XDR_TRAJECTORY_FORMATS: Tuple[str, ...] = ("XTC", "TRR")
"""Formats whose frame offsets can be collected and re-installed by hand.

MDAnalysis keeps XDR offsets inside the C-level ``_xdr`` object, which is *not*
part of the reader's picklable state, so a broadcast Universe arrives without
them and rebuilds them per rank — from the sidecar when one can be written next
to the trajectory, and by rescanning the whole file when it cannot. Every other
format we use keeps its offsets in ordinary Python attributes (LAMMPSDUMP's
``_offsets``), which *do* travel with a pickle. That single difference is why the
two families want opposite strategies.
"""


def offsets_are_installable(
    trajectory_format: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
) -> bool:
    """``True`` when this input's frame offsets can be broadcast and installed.

    Decided from the explicit format keyword when there is one, else from the
    first segment's suffix. Statically knowable, so a strategy can be chosen
    before anything is opened.
    """
    resolved = format_for_path(path, explicit_format=trajectory_format)
    if resolved is None:
        return False
    return resolved in XDR_TRAJECTORY_FORMATS


@dataclass(frozen=True)
class TrajPlan:
    """The broadcastable result of scanning a trajectory once.

    Metadata only — no Universe, no frames. Small enough to ``bcast`` even for a
    100k-frame trajectory (the offsets dominate, at 8 bytes per frame).

    Attributes
    ----------
    total_frames : int
        Frames in the whole (possibly chained) trajectory.
    frame_ids : tuple of int
        Globally selected frames, in order.
    segment_frame_counts : tuple of int
        Frames per input segment, needed to reduce a slice to its segments.
    offsets : tuple of np.ndarray or None
        XDR frame offsets per reader, or ``None`` for non-XDR input.
    has_forces, has_velocities : bool or None
        Whether the first selected frame carries forces / velocities. Collected
        during the scan, where the trajectory is already open and positioned, so a
        caller can reject an impossible request without reopening anything.
        ``None`` when the scan did not open the trajectory.
    """

    total_frames: int
    frame_ids: Tuple[int, ...]
    segment_frame_counts: Tuple[int, ...]
    offsets: Optional[Tuple[np.ndarray, ...]] = None
    has_forces: Optional[bool] = None
    has_velocities: Optional[bool] = None


def broadcast_root_outcome(
    outcome: Optional[Tuple[Any, Optional[Exception]]], *, comm: Optional[Any]
) -> Any:
    """Broadcast an already-computed rank-0 ``(value, error)`` outcome."""
    if comm is not None and int(comm.Get_size()) > 1:
        outcome = comm.bcast(outcome, root=0)
    value, error = outcome
    if error is not None:
        raise error
    return value


def raise_if_rank_failed(local_error: Optional[Exception], *, comm: Optional[Any]) -> None:
    """Raise the lowest-rank error after one whole rank-local operation."""
    errors = [local_error]
    if comm is not None and int(comm.Get_size()) > 1:
        errors = comm.allgather(local_error)
    for error in errors:
        if error is not None:
            raise error


class MPITrajReader:
    """Plan and execute a distributed read of one logical trajectory.

    The four steps every MPI producer in AceCG needs, in one place: scan once on
    rank 0, broadcast the plan, hand each rank its contiguous slice, and open
    only what that rank must open.

    This object stores the plan — paths, formats, frame ids, per-rank slices,
    XDR offsets — plus, on rank 0 only, the live inspection Universe until the
    first local iteration can reuse it. The Universe is consumed by
    :meth:`iter_local`; it is never exposed as a caller-managed local-read
    protocol or added to :class:`TrajPlan`.

    Universe-construction strategies
    --------------------------------
    The three pre-existing AceCG readers differed only here, and the differences
    were measured rather than accidental, so they became a parameter instead of
    being collapsed:

    ``"reopen"``
        Every rank opens the full chain itself with the scan's frame offsets
        installed.
    ``"broadcast"``
        Rank 0 opens the chain and the Universe object is MPI-broadcast.
    ``"local_segments"``
        Every rank opens only the segments its own contiguous slice touches,
        turning a 50-file chained reader into a 1-file one, and installs just
        those segments' offsets.
    ``"auto"`` (default)
        Chosen from two things that actually change the cost: whether the format's
        offsets survive a pickle (see :func:`offsets_are_installable`), and
        whether there are more segments than ``broadcast_segment_limit``.

        =================== ======================= =========================
        format              few segments            many segments
        =================== ======================= =========================
        XTC / TRR           ``reopen``              ``local_segments``
        LAMMPSDUMP, other   ``broadcast``           ``local_segments``
        =================== ======================= =========================

        For XDR there is nothing to gain from broadcasting a Universe — the
        offsets do not come with it — so every rank reopens and installs the
        broadcast offsets instead. For LAMMPSDUMP the offsets *do* travel, so one
        text scan on rank 0 serves the whole job, which is what the VP grower
        measured. Past a handful of segments both families prefer opening only
        the local segments: holding 50 readers per rank to read two of them costs
        50 file handles and a chained-lookup penalty on every seek.

        A serial run always resolves to ``reopen``: there is no peer to broadcast
        to, and one rank owns every segment anyway.

    Parameters
    ----------
    trajectory_files : sequence of str or Path
        Segments in time order.
    topology : str or Path, optional
        Topology; may be omitted for XTC/TRR/LAMMPSDUMP.
    trajectory_format, topology_format : str, optional
        Explicit MDAnalysis format keywords.
    universe_kwargs : mapping, optional
        Extra keywords for every :func:`open_universe` call, e.g. ``atom_style``.
    comm : mpi4py communicator, optional
        ``None`` for serial.
    strategy : {"auto", "reopen", "broadcast", "local_segments"}, default "auto"
        Universe-construction strategy, see above. Set it explicitly only to
        override the format/segment-count reasoning.
    broadcast_segment_limit : int, default 2
        Segment count above which ``"auto"`` switches to ``"local_segments"``.
    prefer_offsets : bool, default True
        Collect XDR offsets during the scan and install them on every later
        open.

    Examples
    --------
    >>> reader = MPITrajReader(trajectory_files=["md.xtc"], comm=comm)  # doctest: +SKIP
    >>> plan = reader.scan(every=10)                                   # doctest: +SKIP
    >>> for frame in reader.iter_local(include_time=True):             # doctest: +SKIP
    ...     consume(frame["positions"], frame["time"])
    """

    def __init__(
        self,
        *,
        trajectory_files: Sequence[Union[str, Path]],
        topology: Optional[Union[str, Path]] = None,
        trajectory_format: Optional[str] = None,
        topology_format: Optional[str] = None,
        universe_kwargs: Optional[Mapping[str, Any]] = None,
        comm: Optional[Any] = None,
        strategy: str = "auto",
        broadcast_segment_limit: int = 2,
        prefer_offsets: bool = True,
    ) -> None:
        if strategy not in UNIVERSE_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {list(UNIVERSE_STRATEGIES)}, got {strategy!r}."
            )
        paths = [str(path) for path in trajectory_files]
        if not paths:
            raise ValueError("MPITrajReader needs at least one trajectory file.")
        self.trajectory_files: Tuple[str, ...] = tuple(paths)
        self.topology = None if topology is None else str(topology)
        self.trajectory_format = format_for_path(
            paths[0], explicit_format=trajectory_format
        )
        self.topology_format = (
            topology_format
            if topology_format is not None or self.topology is None
            else format_for_path(self.topology)
        )
        self.universe_kwargs: Dict[str, Any] = dict(universe_kwargs or {})
        self.comm = comm
        self.requested_strategy = strategy
        self.broadcast_segment_limit = int(broadcast_segment_limit)
        self.prefer_offsets = bool(prefer_offsets)
        self.plan: Optional[TrajPlan] = None
        self._root_inspection_universe: Optional[mda.Universe] = None
        self._opened_segment_numbers: Tuple[int, ...] = ()

    # ── rank/size ─────────────────────────────────────────────────────

    @property
    def rank(self) -> int:
        """This rank's id, ``0`` in serial."""
        return 0 if self.comm is None else int(self.comm.Get_rank())

    @property
    def size(self) -> int:
        """Communicator size, ``1`` in serial."""
        return 1 if self.comm is None else int(self.comm.Get_size())

    @property
    def installable_offsets(self) -> bool:
        """``True`` when this input's offsets can be broadcast and re-installed."""
        return offsets_are_installable(self.trajectory_format, self.trajectory_files[0])

    @property
    def strategy(self) -> str:
        """The resolved universe-construction strategy for this run's shape.

        See the class docstring for the ``"auto"`` table. Serial runs always come
        out as ``"reopen"``: there is nothing to broadcast, and one rank owns
        every segment.
        """
        if self.requested_strategy != "auto":
            return self.requested_strategy
        if self.size <= 1:
            return "reopen"
        if len(self.trajectory_files) > self.broadcast_segment_limit:
            return "local_segments"
        # Few segments: reopening pays only when the offsets can come along.
        return "reopen" if self.installable_offsets else "broadcast"

    # ── stage 1: scan once, broadcast the plan ────────────────────────

    def scan(
        self,
        *,
        start: int = 0,
        end: Optional[int] = None,
        every: int = 1,
        n_frames: int = 0,
        frame_ids: Optional[Sequence[int]] = None,
        count_segments: bool = True,
        segment_frame_counts: Optional[Sequence[int]] = None,
        inspection_universe: Optional[mda.Universe] = None,
    ) -> TrajPlan:
        """Learn the trajectory's shape on rank 0 and broadcast it.

        Parameters
        ----------
        start, end, every, n_frames, frame_ids
            Frame selection, resolved by :func:`select_frame_ids`.
        count_segments : bool, default True
            Also record per-segment frame counts, which :meth:`iter_local` needs
            for the ``"local_segments"`` strategy. Ignored when
            *segment_frame_counts* is given.
        segment_frame_counts : sequence of int, optional
            Frame count per segment, when the caller already knows it — from a
            universe it opened for another reason. Supplying it makes the scan
            open nothing itself. XDR offsets are unavailable unless
            *inspection_universe* is also supplied. For local-segment
            LAMMPSDUMP input the reader obtains these counts itself with the
            cheap text counter.
        inspection_universe : MDAnalysis.Universe, optional
            Rank 0's already-open full trajectory. Its shape, capabilities, and
            XDR offsets are inspected here, then the live Universe is retained
            only until :meth:`iter_local` reuses it for ``reopen`` or
            ``broadcast``. Ignored on non-root ranks.

        Returns
        -------
        TrajPlan
            Also stored on ``self.plan``.
        """
        self._root_inspection_universe = None
        outcome: Optional[Tuple[Optional[TrajPlan], Optional[Exception]]] = None
        if self.rank == 0:
            universe = inspection_universe
            try:
                if segment_frame_counts is not None:
                    counts = tuple(int(count) for count in segment_frame_counts)
                    if len(counts) != len(self.trajectory_files):
                        raise ValueError(
                            f"segment_frame_counts has {len(counts)} entries but there "
                            f"are {len(self.trajectory_files)} trajectory file(s)."
                        )
                    total_frames = int(sum(counts))
                    offsets = (
                        reader_offsets(universe)
                        if universe is not None and self.prefer_offsets
                        else None
                    )
                elif (
                    universe is None
                    and self.strategy == "local_segments"
                    and self.trajectory_format == "LAMMPSDUMP"
                    and not self.installable_offsets
                ):
                    counts = tuple(
                        int(count_lammpstrj_frames_and_atoms(Path(path))[0])
                        for path in self.trajectory_files
                    )
                    total_frames = int(sum(counts))
                    offsets = None
                else:
                    if universe is None:
                        universe = self.open_full()
                    total_frames = int(len(universe.trajectory))
                    offsets = reader_offsets(universe) if self.prefer_offsets else None
                    counts = (
                        self._segment_frame_counts(universe)
                        if count_segments
                        else (total_frames,)
                    )
                selected = select_frame_ids(
                    total_frames,
                    start=start,
                    end=end,
                    every=every,
                    n_frames=n_frames,
                    frame_ids=frame_ids,
                )
                # What the first selected frame can supply, taken while the
                # trajectory is already open and seeked: a caller that needs
                # forces or velocities can fail before local opens begin.
                has_forces: Optional[bool] = None
                has_velocities: Optional[bool] = None
                if universe is not None:
                    ts = universe.trajectory[int(selected[0])]
                    has_forces = bool(getattr(ts, "has_forces", False))
                    has_velocities = bool(getattr(ts, "has_velocities", False))
                plan = TrajPlan(
                    total_frames=total_frames,
                    frame_ids=tuple(selected),
                    segment_frame_counts=tuple(counts),
                    offsets=None if offsets is None else tuple(offsets),
                    has_forces=has_forces,
                    has_velocities=has_velocities,
                )
                if universe is not None and self.strategy in {"broadcast", "reopen"}:
                    self._root_inspection_universe = universe
                outcome = (plan, None)
            except Exception as exc:
                self._root_inspection_universe = None
                outcome = (None, exc)
        plan = broadcast_root_outcome(outcome, comm=self.comm)
        if plan is None:  # pragma: no cover - a successful outcome always has one
            raise RuntimeError("MPITrajReader.scan() produced no plan.")
        self.plan = plan
        return plan

    def _segment_frame_counts(self, universe: mda.Universe) -> Tuple[int, ...]:
        """Frames per input segment, from the already-open chained reader."""
        trajectory = universe.trajectory
        readers = getattr(trajectory, "readers", None)
        if readers:
            return tuple(int(len(reader)) for reader in readers)
        return (int(len(trajectory)),)

    # ── stage 2: this rank's slice ────────────────────────────────────

    def _require_plan(self) -> TrajPlan:
        if self.plan is None:
            raise RuntimeError("call MPITrajReader.scan() before using the plan.")
        return self.plan

    def local_slice(self, rank: Optional[int] = None) -> Tuple[int, int, List[int]]:
        """``(count, offset, ids)`` of a rank's contiguous slice."""
        plan = self._require_plan()
        return partition_frame_ids(
            plan.frame_ids, size=self.size, rank=self.rank if rank is None else int(rank)
        )

    @property
    def local_frame_ids(self) -> List[int]:
        """Global frame ids this rank owns."""
        return self.local_slice()[2]

    @property
    def opened_segment_numbers(self) -> Tuple[int, ...]:
        """One-based segment numbers opened by the last local-segment read."""
        return self._opened_segment_numbers

    # ── stage 3: open only what this rank reads ───────────────────────

    def open_full(self) -> mda.Universe:
        """Open the whole chain with the plan's offsets installed."""
        plan = self.plan
        return open_universe(
            self.topology,
            self.trajectory_files,
            trajectory_format=self.trajectory_format,
            topology_format=self.topology_format,
            offsets=None if plan is None else plan.offsets,
            **self.universe_kwargs,
        )

    def _open_local(self) -> Tuple[Optional[mda.Universe], List[int]]:
        """Open this rank's source and keep reader-local seek ids private."""
        plan = self._require_plan()
        global_ids = self.local_frame_ids
        strategy = self.strategy
        self._opened_segment_numbers = ()

        if strategy == "broadcast":
            outcome: Optional[
                Tuple[Optional[mda.Universe], Optional[Exception]]
            ] = None
            if self.rank == 0:
                try:
                    universe = self._root_inspection_universe
                    self._root_inspection_universe = None
                    if universe is None:
                        universe = self.open_full()
                    outcome = (universe, None)
                except Exception as exc:
                    outcome = (None, exc)
            universe = broadcast_root_outcome(outcome, comm=self.comm)
            return universe, list(global_ids)

        universe: Optional[mda.Universe] = None
        local_ids: List[int] = []
        segment_numbers: List[int] = []
        local_error: Optional[Exception] = None
        try:
            if global_ids:
                if strategy == "reopen":
                    universe = self._root_inspection_universe
                    self._root_inspection_universe = None
                    if universe is None:
                        universe = self.open_full()
                    local_ids = list(global_ids)
                else:
                    paths, local_ids, segment_numbers = map_ids_to_segments(
                        global_ids,
                        self.trajectory_files,
                        plan.segment_frame_counts,
                    )
                    universe = open_universe(
                        self.topology,
                        paths,
                        trajectory_format=self.trajectory_format,
                        topology_format=self.topology_format,
                        offsets=self._offsets_for_segments(segment_numbers),
                        **self.universe_kwargs,
                    )
        except Exception as exc:
            local_error = exc

        try:
            raise_if_rank_failed(local_error, comm=self.comm)
        except Exception:
            self._root_inspection_universe = None
            raise
        if strategy == "local_segments":
            self._opened_segment_numbers = tuple(int(v) for v in segment_numbers)
        return universe, local_ids

    def _offsets_for_segments(
        self, segment_numbers: Sequence[int]
    ) -> Optional[List[np.ndarray]]:
        """This rank's slice of the scanned offsets, in the order it opens them.

        Without this, ``local_segments`` would throw away the one thing the rank-0
        scan produced and make every rank rescan its own segments — which is the
        whole cost the scan exists to avoid.
        """
        plan = self.plan
        if plan is None or plan.offsets is None:
            return None
        if len(plan.offsets) != len(self.trajectory_files):
            # One offsets array per XDR reader; a mixed or partially-XDR chain does
            # not line up with the segment list, so install nothing rather than
            # install the wrong table.
            return None
        return [plan.offsets[int(number) - 1] for number in segment_numbers]

    # ── stage 4: read ─────────────────────────────────────────────────

    def iter_local(
        self,
        **frame_kwargs: Any,
    ) -> Iterator[FrameRecord]:
        """Open and iterate this rank's frames with global logical identity.

        Local Universe indices are an implementation detail. Each raw record is
        checked against the exact seek id used to read it, then only its
        ``frame_id`` is replaced with the matching global id from the selected
        logical trajectory.
        """
        global_ids = self.local_frame_ids
        universe, seek_ids = self._open_local()
        if len(seek_ids) != len(global_ids):
            raise RuntimeError(
                f"rank {self.rank}: reader mapped {len(global_ids)} global frame "
                f"id(s) onto {len(seek_ids)} reader-local seek id(s)."
            )
        if not global_ids:
            return iter(())
        if universe is None:  # pragma: no cover - a successful non-empty open has one
            raise RuntimeError(
                f"rank {self.rank}: local trajectory open returned no Universe for "
                f"{len(global_ids)} assigned frame(s)."
            )

        raw_records = iter_frames(universe, frame_ids=seek_ids, **frame_kwargs)

        def restore_global_ids() -> Iterator[FrameRecord]:
            produced = 0
            for global_id, seek_id in zip(global_ids, seek_ids):
                try:
                    record = next(raw_records)
                except StopIteration:
                    raise RuntimeError(
                        f"rank {self.rank}: local trajectory yielded {produced} "
                        f"frame(s); expected {len(global_ids)}."
                    ) from None
                if int(record.frame_id) != int(seek_id):
                    raise RuntimeError(
                        f"rank {self.rank}: reader-local frame order mismatch: "
                        f"expected {int(seek_id)}, got {int(record.frame_id)}."
                    )
                produced += 1
                yield record.replace(frame_id=int(global_id))
            try:
                extra = next(raw_records)
            except StopIteration:
                return
            raise RuntimeError(
                f"rank {self.rank}: local trajectory yielded more than "
                f"{len(global_ids)} frame(s); first extra reader-local id is "
                f"{int(extra.frame_id)}."
            )

        return restore_global_ids()
