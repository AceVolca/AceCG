# 12 TrajMap Developer Reference

*Updated: 2026-08-11.*

> This chapter covers the AA→CG trajectory-mapping transformation and the
> linear force-mapping method that can be fit on top of it. Training
> workflows are documented in [09_workflows.md](09_workflows.md); VP growth,
> the other one-shot trajectory transformation, is documented separately in
> [10_vp_grower.md](10_vp_grower.md).

TrajMap turns an all-atom (AA) trajectory into a coarse-grained (CG) one: a
mapping YAML plus an AA topology in, a CG topology and CG trajectory
segments out, with an optional linear force-mapping operator fit on the same
pass. Like VP growth it is not a `BaseWorkflow` subclass — no training loop,
no frame cache, no forcefield.

## Core ownership

| File | Responsibility |
|---|---|
| `workflows/trajmap.py` | `TrajMapWorkflow`, `acg-trajmap` CLI entry point; resolves config, compiles the mapping spec, drives the reader/mapper/writer pipeline, returns `TrajMapResult` |
| `io/trajmap.py` | `map_cg_trajectory()` — the concrete per-rank frame stream, staged writes, count-validated merge, and report-last completion record |
| `topology/cgmap.py` | `CGMapSpec` — compiles an OpenMSCG-style `cgyaml` mapping document into flat NumPy arrays once, so mapping a frame is pure array arithmetic |
| `topology/cgmap_builder.py` | Authors a mapping YAML from a bead table plus a real AA topology (the write side of `cgmap.py`); also parses GROMACS `.itp` topologies |
| `compute/cgmap.py` | `CGMapper` — the per-frame AA→CG kernel: preallocated-buffer NumPy operations, no Python-level loop over sites/molecules/atoms |
| `compute/force_mapping.py` | Distributed sufficient-statistics accumulation and fitting for the optional linear force-mapping operator |
| `io/force_operator.py` | Single-file persistence (read/write) for a fitted force operator |
| `configs/trajmap_config.py` | `TrajMapConfig` and its `[aa]` / `[mapping]` / `[force_mapping]` / `[run]` sections; standalone parser, separate from the main FM/REM/CDREM/CDFM config |

## Why this exists, and why it is not OpenMSCG

TrajMap reimplements OpenMSCG's `cgyaml`/`cgmap`/`mapper.py` mapping schema
rather than depending on OpenMSCG at runtime, for two reasons: OpenMSCG's
`Mapper.process` loops per-site in the interpreter (the dominant cost for a
large trajectory), and its reference implementation has several bugs that
real `map.yaml` files in this project's archives trigger. `CGMapSpec`
documents every deviation from OpenMSCG's own code inline, all deliberate,
all fixes verified against real mapping files:

- **Anchor is added exactly once.** OpenMSCG's `unpack_group` adds a nested
  group's anchor twice — once when it is passed down recursively, once again
  in the repeat loop — so every site inside a nested group with a non-zero
  parent anchor lands at twice the offset it should.
- Nested children may omit `anchor` / `repeat` / `offset` (default `0/1/0`);
  OpenMSCG raises `KeyError` on files that omit them, and real archive files
  do.
- `site-types` keys may be YAML-parsed as ints; lookups accept both the raw
  key and its string form.
- Integer `x-weight` lists are accepted (OpenMSCG's in-place `/=` raises
  `TypeError` on them).
- Duplicate atom indices inside one site **sum** (matching
  `Mapper.process`'s matmul); OpenMSCG's `Mapper.get_matrix` *overwrites*
  instead, so its own two code paths already disagree with each other — this
  project adopts the summing one.
- The caller's mapping dict is never mutated (OpenMSCG's `from_topology`
  mutates it in place).

Site ordering is preserved bit-for-bit against OpenMSCG: group → repeat →
site-within-unit.

## Call path

```text
TrajMapWorkflow.run()
  -> parse TrajMapConfig
  -> rank 0 opens the AA topology alone, compiles the mapping YAML into a
     CGMapSpec, validates it against the real atom count and masses
  -> MPITrajReader scans the trajectory once on rank 0, broadcasts the plan
     (frame count, selected frame ids, per-segment counts, XDR offsets)
     alongside the spec — no rank repeats the offset scan
  -> reader hands each rank a contiguous balanced slice; "auto" strategy
     decides what each rank opens (whole chain with broadcast offsets for
     one/two segments, or only the touched segments for a long list)
  -> each rank maps its slice with one CGMapper, streaming straight into its
     own segment file — nothing accumulates in memory beyond one AA frame
     plus mapper scratch
  -> optional: accumulate_force_map_statistics() / fit_force_map() attach a
     fitted linear force operator to the CGMapSpec
  -> rank 0 writes the CG topology (LAMMPS data, optional gro) from the first
     selected frame, optionally concatenates segments, writes a JSON report
  -> TrajMapResult
```

Per-rank output segments are a deliberate choice: gathering mapped frames to
rank 0 would serialize the write and route the whole trajectory through one
process. Contiguous per-rank segments let every rank write in parallel, and
because AceCG's own configs already accept a *list* of trajectory segments,
the unmerged form is directly consumable — merging is a convenience, not a
requirement.

## Scientific objects

| Object or operation | Meaning |
|---|---|
| `CGMapSpec` | Compiled mapping plan: flat NumPy arrays for site→atom indices, x-weight (position, row-normalized → COM if masses, COG if ones), f-weight (force, raw sum), and optional CG bonded topology |
| `CGMapper` | Per-frame kernel: `X_I = Σ_i w^x_{Ii} x_i`, `F_I = Σ_i w^f_{Ii} f_i`, over preallocated buffers |
| `MappedFrame` | One mapped `(n_sites, 3)` position/force frame |
| A *molecule* | One repeat unit of one top-level `system` group — an AA-side notion from `anchor`/`repeat`/`offset`, independent of whether CG bonded topology is known; matches the CG `resid` assignment `io.coordinates.build_CG_coords` already uses |

A site whose atoms straddle a periodic boundary must be made whole before
averaging, or the weighted mean lands at a meaningless point in the middle of
the box; `unwrap` selects the reference each atom is imaged against before
`CGMapper` runs.

Optional CG bonded topology comes from either a `cg-topology:` block in the
same mapping YAML (extra top-level keys OpenMSCG itself ignores, so the file
stays readable by it) or an OpenMSCG `top.in`/`cgtop` file via the existing
`topology/mscg.py` parser. Neither is required — a spec with no bonded
topology still maps trajectories.

## Linear force-mapping

`compute/force_mapping.py` and `io/force_operator.py` implement the
statistically optimal linear force-aggregation method of Kraemer, Durumeric,
Charron, Chen, Clementi & Noe, *Statistically Optimal Force Aggregation for
Coarse-Graining Molecular Dynamics*, J. Phys. Chem. Lett. 14(17), 3970–3979
(2023), <https://doi.org/10.1021/acs.jpclett.3c00444>. Given a molecule's AA
force samples and the CG coordinate map, it fits a linear operator that
aggregates AA forces onto CG sites more accurately than the fixed weights in
`x-weight`/`f-weight`, subject to a constraint that it reproduces the
coordinate map exactly (`W C^T = I`, checked at fit time). `[force_mapping]`
in `TrajMapConfig` selects the method (`fixed` / `constraint_aware_uniform` /
`optimal_linear`), scope (`auto` / `global` / `per_template`), and backend
(`auto` / `native` / `osqp` / `compare`); a fitted operator round-trips
through `io/force_operator.py` as a single file alongside the CG topology.

## Development rules

1. Keep the OpenMSCG-schema compilation math in `topology/cgmap.py`; keep any
   further OpenMSCG-schema deviation documented inline there, not silently.
2. Keep the per-frame mapping kernel in `compute/cgmap.py` as whole-array
   NumPy over preallocated buffers — no per-site or per-molecule Python loop.
3. Keep discovery, selection, global IDs, partitioning, and local opening in
   `MPITrajReader` (shared with VP growth); do not reimplement frame
   partitioning inside TrajMap.
4. Keep VP and TrajMap terminals separate: they share the reader spine and
   failure/publication rules, not scientific kernels or writers.
5. Attribute the linear force-mapping method's origin (see above) in any code
   or documentation that touches `compute/force_mapping.py` /
   `io/force_operator.py`.
