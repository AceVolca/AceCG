# 10 VP Grower Developer Reference

*Updated: 2026-08-10.*

> This chapter covers the one-shot VP trajectory transformation. Training
> workflows are documented in [09_workflows.md](09_workflows.md); the other
> one-shot trajectory transformation, TrajMap, is documented separately in
> [12_trajmap.md](12_trajmap.md).

VP growth turns a CG-only reference topology and trajectory into:

- a VP-augmented schema topology
- `latent.settings` and initial pair/bond/angle tables
- one `frame_*.data` per unique selected source frame
- optional aligned `frame_*.forces.npy`
- `timing.json` and the manifest-last `manifest.json` completion record

It does not inherit from `BaseWorkflow`, enter trainer/optimizer loops, or use
the task scheduler.

## Core ownership

| File | Responsibility |
|---|---|
| `workflows/vp_growth.py` | Resolve config-relative paths, construct the shared reader, call one terminal, return `VPGrowthResult` |
| `io/vp_growth.py` | Distributed VP transform, validation, staging, publication, timing, and ordered provenance |
| `io/trajectory.py` | Scan, global frame identity, MPI slices, loading strategy, and rank-local iteration |
| `topology/vpgrower.py` | Static template, single-frame VP placement, and concrete LAMMPS DATA writer |
| `io/vp_ffbuilder.py` | VP forcefield construction plus the shared settings/table inventory, renderer, and writers |
| `configs/vp_growth_config.py` | VP grower-specific config model and parser |

There is no `compute/vp_prepare.py` layer. It was an additional orchestration
and writer owner, not a scientific kernel, and has been deleted rather than
retained as a compatibility facade.

## Call path

```text
VPGrowthWorkflow.run()
  -> resolve paths
  -> MPITrajReader(strategy="auto", broadcast_segment_limit=2)
  -> grow_vp_trajectory(...)
       -> build VPTopologyTemplate from the static reference topology
       -> reader.scan()
       -> enumerate and preflight every final target
       -> stage schema topology and latent settings/tables
       -> reader.iter_local() exactly once per rank
       -> VPGrower.grow_frame() + concrete DATA/force writers
       -> one records/statistics gather
       -> validate order, coverage, files, and force arrays
       -> stage timing and ordered manifest
       -> publish exact targets, manifest last
  -> VPGrowthResult from the terminal's plain dictionary
```

The workflow owns no Universe open, scan, frame loop, MPI collective, writer,
merge, or serialization logic.

## Scientific objects

| Object or operation | Meaning |
|---|---|
| `VPTopologyTemplate` | Immutable atom/type/topology layout and real/VP index mappings |
| `VPGrownFrame` | One `(n_atoms, 3)` grown coordinate array plus box |
| `VPGrower.from_universe()` | Compile the static reference topology into the template |
| `VPGrower.grow_frame()` | Sole per-frame VP placement and clash-resolution kernel |
| `write_vp_data()` | Concrete template-plus-frame LAMMPS DATA writer |

The orientation seed is always
`orientation_seed_base + source_frame_id`. It never depends on rank, local
seek ID, selection index, or MPI size.

## Shared reader policy

`MPITrajReader` retains the VP-measured threshold:

- serial input reopens
- one or two non-XDR segments broadcast the live rank-0 Universe
- one or two XTC/TRR segments reopen with offsets scanned once on rank 0
- more than two segments open only the segments intersecting each rank's slice

For the last case, a LAMMPSDUMP chain is counted per file with the cheap text
counter inside `scan()`. The workflow does not discover or pass segment
counts. `iter_local()` is the only distributed-read entry and restores every
record's global logical source ID.

## Ordered duplicate semantics

Selection is an occurrence sequence, not a set. For a selection such as
`[5, 0, 2, 2]`:

- the manifest contains four records in that exact order
- source ID `2` has one physical DATA/force pair
- only the rank owning the first `2` occurrence grows and writes that pair
- both occurrence records point to that pair and use the same seed

The manifest record fields include `selection_index`, `source_frame_id`,
`orientation_seed`, `data`, and `forces`. No sorting or frame-ID dictionary
merge is allowed.

## Force contract

`include_forces = true` is strict. If scan knows the source has no forces, the
terminal fails before staging. If capability is unknown, every consumed
occurrence is checked and a missing force closes collectively after the local
loop. Emitted arrays are `float32` with shape `(n_real, 3)` and remain aligned
to their source global ID.

## Output inventory and publication

`vp_forcefield_inventory()` is the single source of relative settings/table
names used by renderer, writer, and terminal. Before the first writer, the
terminal enumerates:

- `vp_topology.data`
- latent settings and every configured pair/bond/angle table
- unique DATA and requested force files
- `timing.json`
- `manifest.json`

It rejects resolved duplicates, parent/child overlaps, staging overlap, and
non-overwrite collisions. Writers target one shared staging directory. After
all staged artifacts and force arrays validate, only enumerated final paths are
replaced; unrelated files survive overwrite. An old manifest is invalidated
before replacement starts, and the new manifest is published last. This is a
completion-marker protocol, not a claim of multi-file POSIX atomicity.

Typical output:

```text
output_dir/
  vp_topology.data
  latent.settings
  Pair_*.table
  VP_*_bon.table
  VP_*_ang.table
  frame_000000.data
  frame_000000.forces.npy
  timing.json
  manifest.json
```

## Development rules

1. Keep template and placement mathematics in `topology/vpgrower.py`.
2. Keep exactly one VP transform/output operation in `io/vp_growth.py`; do not
   add per-frame wrappers, workflow inheritance, sinks, managers, or result
   hierarchies.
3. Keep discovery, selection, global IDs, partitioning, and local opening in
   `MPITrajReader`.
4. Keep VP and TrajMap terminals separate: they share the reader spine and
   failure/publication rules, not scientific kernels or writers.
