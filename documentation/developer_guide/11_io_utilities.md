# 11 I/O Utilities Developer Reference

*Updated: 2026-05-24.*

> This chapter covers generic I/O utilities only. VP-specific `latent.settings` and table generation are documented in [10_vp_grower.md](10_vp_grower.md).

In the active repo, `AceCG.io` owns three categories of functionality:

- trajectory / frame extraction
- forcefield / table / coordinate serialization
- lightweight helpers such as screen logging

It does not own trainer semantics, workflow training loops, or compute reducer math.

---

## Core Modules

| File | Responsibility |
|---|---|
| `io/trajectory.py` | Trajectory splitting, random frame reading, one-pass frame iteration |
| `io/forcefield.py` | LAMMPS forcefield / mask I/O |
| `io/coordinates.py` | AA-to-CG coordinate mapping and sanity checks |
| `io/coordinates_writers.py` | `.gro` / `.pdb` / LAMMPS `.data` writing |
| `io/tables.py` | Table I/O and FM table-bundle export |
| `io/logger.py` | Screen logging and formatted timestamps |

---

## Trajectory API

The most important trajectory entry points are standalone functions in `trajectory.py`, not workflow helpers.

| Function | Purpose |
|---|---|
| `iter_frames(universe, ...)` | Canonical one-pass frame iterator for compute runtime and VP grower |
| `read_lammpstrj_frames(path, frame_ids, ...)` | Random-access read for explicit frame-id lists |
| `count_lammpstrj_frames_and_atoms(path)` | Lightweight scan of segment frame count and atom count |
| `split_lammpstrj(...)` | Text-parser-based splitting for large trajectories |
| `split_lammpstrj_mdanalysis(...)` | MDAnalysis-based splitting path |

Developer contract:

- `iter_frames()` is a first-class compute-layer entry point and returns `(frame_id, positions, box, forces)`.
- Explicit `frame_ids` override `start/end/every`.
- Whether forces are read is controlled by `include_forces` and by whether the trajectory contains force columns.

---

## Forcefield I/O

`io/forcefield.py` bridges runtime `Forcefield` objects and LAMMPS settings files.

| Function | Purpose |
|---|---|
| `ReadLmpFFMask()` | Parse authored forcefield mask files |
| `ReadLmpFFBounds()` | Parse authored parameter-bound files |
| `ReadLmpFF()` | Read a `Forcefield` from LAMMPS-style settings |
| `WriteLmpFF()` | Write the current `Forcefield` to new settings / table files |
| `resolve_source_table_entries()` | Resolve original table tokens and table names for FM source-table paths |

`ReadLmpFF()` has two important conventions:

- if `topology_arrays` is provided, bond/angle type ids are restored to canonical `InteractionKey` values
- `table` styles are not retained as raw tables; they are fitted into runtime potentials according to `table_fit`

`ReadLmpFFMask()` and `ReadLmpFFBounds()` require `topology_arrays`. Numeric
bonded coefficient entries such as `bond_coeff 4 ...` and `angle_coeff 7 ...`
are LAMMPS 1-based type ids; the readers translate them through the topology
maps into canonical interaction keys after configured type aliases are applied.
The returned specs therefore contain keys such as `bond:HG:VP`, not `bond:4`.

`WriteLmpFF()` is the corresponding write side:

- non-table terms are written by filling parameters back into settings
- table terms regenerate table files

It is therefore the canonical output path for workflow-exported runtime forcefield bundles.

---

## Coordinate Construction and `.data` Writing

Important high-level entry points:

| Function | Purpose |
|---|---|
| `build_CG_coords()` | Build CG beads from AA coordinates and mapping, optionally writing `.gro`, `.pdb`, or `.data` |
| `write_lammps_data()` | Write LAMMPS data files with optional bonds / angles / dihedrals |
| `write_gro()` / `write_pdb()` | Lightweight structure-file writing |

`build_CG_coords()` handles:

- reading YAML mapping or an in-memory mapping dict
- mapping sanity checks
- bead position and mass calculation from AA topology / trajectory
- deciding whether to write files through `outputs={...}`

`write_lammps_data()` is the lower-level writer. It currently supports:

- `atomic` / `full` atom styles
- optional bonds / angles / dihedrals and their type ids
- triclinic boxes
- image flags and wrapped coordinates

When coordinates, type ids, and topology arrays are already available, call `write_lammps_data()` directly. When performing AA-to-CG mapping, use `build_CG_coords()`.

---

## Table I/O

`io/tables.py` supports forcefield I/O, FM table export, and table-file comparison.

Important functions:

| Function | Purpose |
|---|---|
| `parse_lammps_table()` | Low-level table parser |
| `write_lammps_table()` | Low-level table writer |
| `build_forcefield_tables()` | Build in-memory payload from FM runtime spec plus `Forcefield` |
| `export_tables()` | Write a full FM table set to a directory and return a manifest |
| `compare_table_files()` | Compare reference and candidate table files |
| `cap_table_forces()` | Apply a hard cap to table forces |

For workflow code, the usual high-level entry point is `export_tables()`. `parse_lammps_table()` and `write_lammps_table()` are lower-level building blocks.

---

## Logger

`io/logger.py` is intentionally thin, but almost every workflow / scheduler / compute entry point uses it.

Common symbols:

| Symbol | Purpose |
|---|---|
| `get_screen_logger(name)` | Get a module-level `ScreenLogger` |
| `format_screen_message(...)` | Format messages consistently |
| `user_timestamp()` | User-readable timestamp |

This layer only handles lightweight console logging. It does not own structured event collection or file-log rotation.

---

## Usage Rules

To keep the I/O layer stable, follow these rules:

1. Prefer `iter_frames()` for one-pass runtime code; do not reimplement per-frame reading in workflows.
2. Prefer `WriteLmpFF()` for runtime forcefield export; do not hand-build LAMMPS settings in workflows.
3. Prefer `export_tables()` for FM table export so manifest and filename conventions stay consistent.
4. Keep VP-specific `latent.settings` and VP table builders in the separate [10_vp_grower.md](10_vp_grower.md) pipeline.
