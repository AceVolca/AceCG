# 00 AceCG Software Architecture

*Updated: 2026-05-06. Merged and expanded from draw.md (2026-04-03).*

> This is the top-level architecture document for AceCG and is the recommended first document to read before diving into the code.

---

## Executive Summary

AceCG is a coarse-graining force-field training system. It takes all-atom MD trajectories as input and optimizes CG force-field parameters through several training methods, including FM, noisy FM, mixed noisy FM, DSM, REM, noisy REM, CDREM, CDFM, and VP growth.

Core design principle: **each layer owns only its own concerns, exposes stable APIs upward, and depends only on the layer directly below it.**

---

## Layer Overview

```text
L6  Workflow
    workflows/base.py, sampling.py, rem.py, cdrem.py,
    fm.py, dsm.py, cdfm.py, vp_growth.py

L5  Scheduler / Task Runner
    schedulers/task_scheduler.py, task_runner.py,
    resource_pool.py, mpi_backend.py, profiler.py

L4  Trainers / Solvers / Optimizers
    trainers/analytic/{rem,mse,fm,cdrem,cdfm,l_zero,multi}.py
    solvers/fm_matrix.py
    optimizers/{adam,adamW,rmsprop,newton_raphson}.py

L3  Compute Runtime
    compute/mpi_engine.py
    compute/frame_geometry.py, energy.py, force.py
    compute/reducers.py, registry.py
    analysis/rdf.py

L2  I/O + Config
    io/trajectory.py, forcefield.py, coordinates.py
    io/tables.py, logger.py
    configs/parser.py, models.py, vp_config.py

L1  Topology / Forcefield
    topology/types.py, topology_array.py, forcefield.py
    topology/neighbor.py, mscg.py

L0  Potentials / Fitters / Samplers
    potentials/{harmonic,bspline,gaussian,gated,lj,...}.py
    fitters/{fit_bspline,fit_harmonic,fit_multi_gaussian}.py
    samplers/base.py, conditioned.py
```

---

## Layer Details

### L0: Potentials / Fitters / Samplers

This layer does not depend on any other AceCG module. It is the leaf layer of the system.

| Module | Responsibility |
|---|---|
| `potentials/base.py` | `BasePotential` abstract base class and `IteratePotentials` helper |
| `potentials/harmonic.py` | Harmonic bonded / angle potential |
| `potentials/bspline.py` | Force-basis B-spline potential; all parameters are linear |
| `potentials/gaussian.py` | Normalized Gaussian pair potential |
| `potentials/gated.py` | Hard-concrete gate wrapper for interaction sparsification experiments |
| `potentials/lj*.py` | LJ 12-6 / 9-6 / soft-core families |
| `potentials/multi_gaussian.py` | Normalized multi-Gaussian family |
| `fitters/fit_*.py` | Initial parameter fitting from RDF / distribution data |
| `samplers/base.py` | Base LAMMPS script staging and replica planning |
| `samplers/conditioned.py` | Conditioned sampling script support |

This layer does not own atom indexing, MPI, frame iteration, or trainer state.

Scalar force convention:

```text
F(r) = -dU/dr
```

The compute layer projects scalar forces back to Cartesian atom forces through the geometry direction vectors in `FrameGeometry`.

### L1: Topology / Forcefield

The core objects are:

| Class | File | Responsibility |
|---|---|---|
| `InteractionKey` | `topology/types.py` | Canonical interaction identifier, `NamedTuple(style, types)` |
| `TopologyArrays` | `topology/topology_array.py` | Frozen, MPI-broadcastable topology snapshot |
| `Forcefield` | `topology/forcefield.py` | `MutableMapping[InteractionKey, List[BasePotential]]` plus parameter-vector caches |

`InteractionKey` normalization rules:

| Style | Rule | Example |
|---|---|---|
| pair / bond | Lexicographic order: `(a, b) if a <= b` | `pair("C", "A")` -> `("A", "C")` |
| angle | Reverse if `a > c`: `(a, b, c) if a <= c else (c, b, a)` | `angle("Z", "Y", "A")` -> `("A", "Y", "Z")` |
| dihedral | Reverse if `(a, b) > (d, c)` | `dihedral("D", "C", "B", "A")` -> `("A", "B", "C", "D")` |

`TopologyArrays` is built once by `collect_topology_arrays(universe, ...)` and then broadcast to all MPI ranks. It contains atom-level arrays, bonded terms, exclusion lists, virtual-site classification, and instance-to-type mappings.

`Forcefield` is the source of truth for parameters:

- `param_array()` and `update_params(L)` control the complete flattened parameter vector.
- potential-local `param_mask` and `param_bounds` values carry trainability
  and bound metadata with each potential object.
- `key_mask`, `param_mask`, and `param_bounds` expose global, read-only dirty
  caches assembled from potential-local metadata.
- `deepcopy()` and `copy.deepcopy()` are safe; trainers and solvers own independent copies.

### L2: I/O + Config

| Module | Responsibility |
|---|---|
| `io/trajectory.py` | `iter_frames()`, the unified trajectory reader yielding `(frame_id, positions, box, forces)` |
| `io/forcefield.py` | `ReadLmpFF` / `WriteLmpFF`, LAMMPS force-field I/O |
| `io/coordinates.py` | CG coordinate mapping helpers |
| `io/coordinates_writers.py` | LAMMPS data-file writing, including topology support |
| `io/tables.py` | LAMMPS tabulated potential I/O |
| `io/logger.py` | Structured screen logging |
| `configs/parser.py` | `.acg` config parser |
| `configs/models.py` | Config dataclasses such as `SchedulerConfig` and `WorkflowConfig` |
| `configs/vp_config.py` | VP-related config models |

Runtime conventions:

- `iter_frames()` is the compute engine's unified trajectory entry point.
- Topology is explicitly supplied by each task spec; each `run_post()` rebuilds `TopologyArrays`.
- Force-field snapshots are passed through pickle files via `forcefield_path`.

See [11_io_utilities.md](11_io_utilities.md).

### L3: Compute Runtime

This is the numerical core of the system. See [03_compute.md](03_compute.md) and [04_mpi_runtime.md](04_mpi_runtime.md).

| File | Responsibility |
|---|---|
| `compute/mpi_engine.py` | `MPIComputeEngine`, `FrameCache`, `TrajectoryCache`; legacy aliases: `FrameObservables`, `TrajectoryObservablesCache` |
| `compute/frame_geometry.py` | `FrameGeometry`, an immutable per-frame geometry view |
| `compute/energy.py` | `energy()` kernel |
| `compute/force.py` | `force()` kernel |
| `compute/reducers.py` | Stateful pipeline reducers: init / consume / finalize |
| `compute/registry.py` | `build_default_engine()`, which registers core observables |
| `analysis/rdf.py` | RDF / PDF distribution analysis |

Stable public APIs:

```python
FrameGeometry = compute_frame_geometry(positions, box, topology_arrays, ...)
energy_dict   = energy(geom, forcefield, return_grad=True, ...)
force_dict    = force(geom, forcefield, return_grad=True, ...)
```

`FrameGeometry` is the only geometry container for both ordinary frames and
same-frame coordinate batches. `positions` may have shape
`(..., n_atoms, 3)`, and geometry values preserve those leading dimensions
with the interaction-term axis last. Interaction index arrays stay shared per
key.

There are no separate public `energy_batch()` or `force_batch()` kernels.
Potential evaluators, `compute_frame_geometry()`, `energy()`, and `force()`
all accept batched inputs through the ordinary APIs. `MPIComputeEngine.compute()`
is the single local frame-compute entry point; it flattens leading sample
dimensions, chunks them for memory when requested, manages pair-cache policy,
and folds outputs back into reducer-ready sufficient statistics.

This layer does not own scheduler policy, workflow orchestration, or trainer tallies across tasks.

### L4: Trainers / Solvers / Optimizers

| Module | Responsibility |
|---|---|
| `trainers/base.py` | `BaseTrainer` abstract contract |
| `trainers/analytic/rem.py` | `REMTrainerAnalytic`, energy-gradient batch consumer |
| `trainers/analytic/fm.py` | `FMTrainerAnalytic`, iterative FM batch consumer |
| `trainers/analytic/cdrem.py` | `CDREMTrainerAnalytic`, latent-variable REM |
| `trainers/analytic/cdfm.py` | `CDFMTrainerAnalytic`, CDFM aggregated by x |
| `trainers/analytic/mse.py` | `MSETrainerAnalytic`, PMF-matching MSE |
| `trainers/analytic/multi.py` | `MultiTrainerAnalytic`, multi-trainer composition |
| `solvers/fm_matrix.py` | `FMMatrixSolver`, exact FM matrix solve: OLS / ridge / Bayesian |
| `optimizers/adam.py` | Masked Adam |
| `optimizers/newton_raphson.py` | Second-order Newton-Raphson |

All trainers share the same step contract:

```python
out = trainer.step(batch, apply_update=True)
```

The workflow builds the batch. Trainers do not rebuild trajectory state.

### L5: Scheduler

See [08_schedulers.md](08_schedulers.md).

The scheduler has three layers:

```text
resource discovery -> greedy placement -> MPI command assembly
```

The workflow only needs:

```python
TaskScheduler.run_iteration(xz_tasks, zbx_tasks)  # returns list[TaskResult]
```

### L6: Workflows

See [09_workflows.md](09_workflows.md).

Workflows own iteration directories, checkpoints, task planning, force-field snapshots, and calls into trainers or solvers. They should not duplicate reducer math or own MPI frame slicing.

---

## End-to-End Training Flow

Typical REM / CDREM / CDFM flow:

```text
.acg config
  -> workflow parses config
  -> topology and forcefield snapshots are built
  -> scheduler launches simulation and post-processing tasks
  -> compute runtime writes reducer pickle payloads
  -> workflow builds trainer batches
  -> trainer.step(...) updates its owned forcefield and optimizer
  -> workflow propagates the new forcefield into the next iteration
```

Typical FM flow:

```text
.acg config
  -> FMWorkflow builds FM forcefield
  -> optional aa_ref.noise_* creates same-frame coordinate replicas
  -> task_runner.run_post(...) accumulates step_mode="fm" statistics
  -> either FMTrainerAnalytic.step(...) iterates
     or FMMatrixSolver.solve(...) performs a closed-form solve
  -> tables are exported for LAMMPS
```

Typical DSM flow:

```text
.acg config
  -> DSMWorkflow reuses FM interaction specs and trainable forcefield setup
  -> aa_ref.noise_* creates noisy coordinate replicas
  -> MPIComputeEngine.add_noise(target="dsm") creates score targets
  -> step_mode="dsm" accumulates FM-shaped sufficient statistics
  -> DSMWorkflow scales those statistics by beta^2
  -> FMTrainerAnalytic.step(...) performs the optimizer update
  -> DSM writes dsm_step_XXXX/ff/workflow_checkpoint.pkl for resume
```

---

## Complete Data Flow

```text
AA trajectory (.lammpstrj)
         |
         v
  io/trajectory.py
  iter_frames(trajectory, topology)
         |
         | (frame_id, positions, box, forces)
         v
  compute/mpi_engine.py
  MPIComputeEngine.run_post(spec)
    +-- rank 0: load forcefield, Universe, TopologyArrays, sel_indices
    +-- MPI broadcast: Universe, TopologyArrays, sel_indices
    +-- each rank receives a contiguous frame segment
    +-- iter_frames -> local frame sequence
    +-- optional spec["noise"] -> same-frame coordinate batch
    +-- engine.compute(...) folds batch axes into reducer payloads
    +-- for each step in spec["steps"]:
           |
           +-- step_mode="rem"       -> reducers.init_rem_state / consume_rem_frame
           +-- step_mode="fm"        -> reducers.init_fm_state / consume_fm_frame
           +-- step_mode="dsm"       -> FM reducer contract with synthetic score targets
           +-- step_mode="cdrem"     -> same reducer path as rem
           +-- step_mode="cdfm_zbx"  -> reducers.init_cdfm_zbx_state / ...
           |                           rank 0 computes y_eff first
           +-- step_mode="rdf"       -> analysis/rdf.py, outside one-pass pipeline
                    |
                    | local partials
                    v
         comm.reduce / comm.gather (rank 0)
                    |
                    v
         finalize_*_root -> pickle output
                    |
         +----------v-------------+
         | trainer batch dict     |
         +----------+-------------+
                    |
                    v
         trainer.step(batch)
           +-- compute gradient
           +-- optionally compute Hessian
           +-- optimizer.step(grad, hessian) -> delta_L
                    |
                    v
         forcefield.update_params(L + delta_L)
                    |
                    v
         save checkpoint
         next iteration
```

---

## Core Class Dependency Graph

```text
InteractionKey
    +-- TopologyArrays (owns bond_key_index, keys_bondtypes, ...)
    +-- Forcefield (key -> List[BasePotential])

BasePotential
    +-- Forcefield

TopologyArrays + Forcefield + positions/box
    +-- FrameGeometry (compute_frame_geometry)
            +-- energy(geom, ff) -> {energy, energy_grad, ...}
            +-- force(geom, ff)  -> {force, force_grad, fm_stats}
            +-- same APIs accept (..., n_atoms, 3) same-frame batches

MPIComputeEngine
    +-- owns: MPI comm, _registry
    +-- compute(request, frame, ...) -> local observable dict
    +-- run_post(spec) -> pickle files
            +-- uses reducers.py pipeline (init/consume/finalize)

reducers.py
    +-- input: engine.compute() result + topology + forcefield
    +-- output: local partial dicts for MPI reduce

trainer.step(batch)
    +-- input: workflow-built batch dict from pickle output
    +-- computes: gradient and optional Hessian
    +-- output: optimizer.step(grad) -> delta_L -> forcefield update

workflow / scheduler
    +-- orchestrate the above without absorbing their internals
```

---

## Suggested Code Reading Order

For a quick understanding of the current runtime, read in this order:

1. `topology/types.py` - `InteractionKey`
2. `topology/topology_array.py` - `TopologyArrays`, `collect_topology_arrays()`
3. `topology/forcefield.py` - `Forcefield` and parameter-vector APIs
4. `compute/frame_geometry.py` - `FrameGeometry`, `compute_frame_geometry()`
5. `compute/registry.py` - `build_default_engine()` and registered observables
6. `compute/mpi_engine.py` - `MPIComputeEngine.compute()` and `run_post()`
7. `compute/reducers.py` - init / consume / finalize pipeline API
8. `trainers/analytic/*.py` - trainer batch contracts and `step()` implementations
9. `solvers/fm_matrix.py` - matrix FM solver
10. `schedulers/task_runner.py` - what happens on the compute node
11. `workflows/base.py` - checkpoint and iteration-loop infrastructure

---

## Important Design Boundaries

| Relationship | Allowed | Forbidden |
|---|---|---|
| workflow -> compute | Call `MPIComputeEngine.run_post()` and consume pickle output | Call reducers directly or perform MPI reduce in workflow code |
| workflow -> trainer | Build a batch and call `trainer.step()` | Reproduce gradient calculations from inside trainers |
| trainer -> compute | None; trainers do not create compute engines | Call `run_post()` or create `MPIComputeEngine` |
| scheduler -> compute | Launch `mpi_engine` subprocesses through `task_runner` | Instantiate `MPIComputeEngine` directly |
| reducer -> compute | Call local `engine.compute()` | Read trajectories, broadcast with MPI, or write output files |
| all layers -> topology | Import `InteractionKey`, `TopologyArrays`, `Forcefield` | Reverse dependencies back into topology |

---

## Cross-Layer Rules

1. Potentials know only scalar coordinates and local parameters.
2. Topology owns atom identity, bonded structure, exclusions, and force-field masks.
3. Compute owns per-frame geometry, force/energy evaluation, and reducer accumulation.
4. Trainers and solvers consume prebuilt statistics; they do not read trajectories.
5. Scheduler owns process placement and launch mechanics only.
6. Workflows own orchestration and persistence, but not numerical kernels.
