# 04 AceCG MPI Runtime and Task Call Chain

*Updated: 2026-05-05. Merged and expanded from draw_mpi.md (2026-04-03).*

---

## Overview

This document describes:

- how a scheduled task moves from workflow to compute runtime
- what `MPIComputeEngine.run_post()` actually does
- the semantics of `step_mode="multi"`
- how noisy same-frame batches are generated and folded before reducers
- where MPI logic lives, and where it does not live
- the reducer pipeline design: init / consume / finalize

---

## End-to-End Call Chain

```text
workflow
    |
    v  builds task spec (dict)
scheduler builds task spec
    |
    v
task_runner.py starts as a subprocess
    |
    +-- Phase 1: subprocess.Popen(sim_launch.argv)  <- LAMMPS run
    |
    +-- Phase 2: MPI post, if spec has steps
              |
              +-- in-process: build_default_engine().run_post(spec)
              |
              +-- MPI:        python -m AceCG.compute.mpi_engine spec.json
                                  |
                                  v
                          mpi_engine.__main__
                          build_default_engine(comm=comm)
                          engine.run_post(spec)
```

---

## Stage 1: Workflow / Scheduler Boundary

Workflow and scheduler do only three things:

1. Build a valid task spec.
2. Launch the task.
3. Consume pickle outputs.

They should not own:

- trajectory broadcast
- MPI reduction
- frame chunking
- reducer internals

---

## Stage 2: `task_runner.py`

`task_runner.py` is the local wrapper on the compute node:

```text
task_runner.run(spec):
  simulation phase:
    subprocess.Popen(spec["sim_launch"]["argv"], env=...)
    wait for exit code

  optional post phase:
    if spec has "steps":
      in-process: build_default_engine().run_post(spec)
      MPI post:   subprocess.Popen(post_launch.argv, ...)
```

---

## Stage 3: `mpi_engine.__main__`

The entry point is intentionally thin:

```python
spec = json.load(sys.argv[1])
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None
engine = build_default_engine(comm=comm)
engine.run_post(spec)
```

There is no module-level `run_post()` shim.

---

## Stage 4: `build_default_engine()`

`build_default_engine()` creates an `MPIComputeEngine` and registers seven core observables:

| Name | Reduce mode | Description |
|---|---|---|
| `energy_grad` | gather | Per-frame energy gradient `dU/dtheta` |
| `energy_value` | gather | Per-frame scalar energy |
| `energy_hessian` | gather | Per-frame energy Hessian `d2U/dtheta2` |
| `energy_grad_outer` | gather | Per-frame gradient outer product |
| `force_grad` | gather | Per-frame force Jacobian `df/dtheta` |
| `force_value` | gather | Per-frame model forces |
| `fm_stats` | dict_sum | Per-frame FM sufficient statistics: `JtJ`, `Jty`, `y_sumsq` |

---

## Stage 5: `MPIComputeEngine.run_post(spec)`

This is the true MPI post-processing boundary. The logic is centralized in `compute/mpi_engine.py`.

### Spec Structure

```python
spec = {
    "work_dir":        str,
    "forcefield_path": str,
    "topology":        str,
    "trajectory":      str | list,
    "trajectory_format": str,
    "frame_start":     int,
    "frame_end":       int,
    "every":           int,
    "frame_ids":       list[int],
    "frame_weight":    list[float],
    "cutoff":          float,
    "sel":             str,
    "exclude_bonded":  str,
    "exclude_option":  str,
    "atom_type_name_aliases": dict,
    "vp_names":        list[str],
    "collect_observables": bool,
    "gather_observables":  bool,
    "noise":            dict,      # optional epoch-local AA-reference noise spec
    "steps": [
        {
            "step_mode":   str,
            "output_file": str,
            ...
        },
    ],
}
```

### `step_mode` Summary

| `step_mode` | Purpose | Main output keys |
|---|---|---|
| `rem` | REM / AA-side energy statistics | `energy_grad_avg`, `n_frames`, `weight_sum`, optional `d2U_avg` |
| `cdrem` | CDREM xz or zbx post-processing; uses the same reducer as REM | Same as `rem` |
| `fm` | Force-matching statistics | `JtJ`, `Jty`, `y_sumsq`, `Jtf`, `f_sumsq`, `fty`, `nframe` |
| `dsm` | DSM score-matching statistics; uses the FM reducer state with synthetic noise targets | Same as `fm` |
| `cdfm_zbx` | CDFM conditioned sampling replica; rank 0 first computes `y_eff` from `(init_config, init_force)` | `grad_direct`, `grad_reinforce`, `sse`, `obs_rows`, `n_samples` |
| `rdf` | RDF / PDF distributions; does not enter the one-pass pipeline | `{InteractionKey: distribution_array}` |

`cdfm_y_eff` has been removed from the active repository. `y_eff` preprocessing is now folded into the rank-0 setup logic of `cdfm_zbx`.

### High-Level Flow

```text
run_post(spec):
  rank 0 only:
    load forcefield pickle
    mda.Universe(topology, trajectory)
    collect_topology_arrays(universe, ...)
    sel_indices = universe.select_atoms(sel).indices
    total_frames = len(universe.trajectory)

  broadcast shared context:
    universe, topology_arrays, sel_indices, total_frames

  all ranks:
    assign frames, either contiguous chunks or balanced discrete_ids
    split steps into one_pass_steps and rdf_steps

  one-pass pipeline:
    for each local frame:
      if spec has noise:
        frame_batch, sample_weights = engine.add_noise(...)
        result = engine.compute(request, frame_batch, frame_weights=sample_weights, ...)
      else:
        result = engine.compute(request, frame)
      for step in one_pass_steps:
        consume_*_frame(state, result)

  MPI reduce:
    comm.reduce(local_partials, root=0)

  rank 0:
    finalize_*_root(state)
    pickle.dump(result, output_file)

  rdf_steps:
    run separately through analysis.rdf, outside the one-pass pipeline
```

---

## Frame Assignment Strategy

Two modes exist.

Contiguous chunking, the default:

```text
[frame_start, frame_end, every] -> n_selected frames
rank i receives: [local_offset, local_offset + local_count) * every
```

Special case: if `n_selected <= MPI size`, distributed slicing is disabled and rank 0 handles all frames.

Discrete frame selection:

- opt in with `spec["frame_ids"]`
- works for any post task requiring an explicit frame-id subset
- frame ids are split evenly across ranks

`aa_ref.noise_subsample_per_epoch` uses the same discrete-frame path. The
workflow resolves the epoch/stage seed, and `run_post()` draws a sorted,
no-replacement subset after `frame_start`, `frame_end`, and `every` have
selected the AA-reference frame universe.

---

## Reducer Pipeline API

`reducers.py` provides five functions for each `step_mode`:

```text
init_*_state(step) -> state dict
request_*(step) -> {need_energy_grad: bool, need_force_grad: bool, ...}
consume_*_frame(state, frame_result, ...) -> mutates state
local_partials_*(state) -> partial dict for comm.reduce
reduce_plan_*(step) -> {key: "sum" | "gather" | ...}
finalize_*_root(state) -> final result dict
```

Current reducers:

| Prefix | `step_mode` |
|---|---|
| `init_fm_state` / ... | `fm` |
| `init_fm_state` + `request_dsm` | `dsm` |
| `init_rem_state` / ... | `rem`, `cdrem` |
| `init_cdfm_zbx_state` / ... | `cdfm_zbx` |
| direct `analysis.rdf` path | `rdf` |

The `cdfm_zbx` reducer performs `y_eff` preprocessing before entering the main one-pass loop.

Benefits of this design:

- each step accumulates local state independently
- MPI reduce behavior is declared by `reduce_plan_*` and executed uniformly by the engine
- reducer functions are pure local math and perform no MPI or I/O
- same-frame noisy batches are already folded by `engine.compute()` into
  ordinary reducer payload keys before `consume_*_frame()` is called

---

## Observable Cache

`MPIComputeEngine.run_post()` supports optional observable caching for workflows that need per-frame geometry, such as RDF analysis or visualization:

```python
spec["collect_observables"] = True
spec["gather_observables"] = True
spec["observables_output_file"] = "traj_cache.pkl"
```

| Class | Location | Responsibility |
|---|---|---|
| `FrameCache` | `compute/mpi_engine.py` | Lightweight per-frame geometry digest: pair distances, bonds, angles, and related data |
| `TrajectoryCache` | `compute/mpi_engine.py` | Collection of many `FrameCache` objects |
| `geometry_to_observables()` | `compute/mpi_engine.py` | Extracts `FrameCache` from `FrameGeometry` |

The old names `FrameObservables` and `TrajectoryObservablesCache` remain as compatibility aliases. New code and docs should prefer `FrameCache` and `TrajectoryCache`.

Observable cache requests are valid for ordinary frames. They are rejected for
noisy same-frame batches because current cache semantics are per real
trajectory frame, not per noisy sample.

---

## MPI Logic Ownership

Inside `MPIComputeEngine.run_post()`:

- rank 0 loads shared context: forcefield, universe, topology, selection indices
- shared context is broadcast
- contiguous frame splitting is performed
- local frame extraction is performed

Inside the reducer loop, still owned by `run_post()`:

- for each one-pass step: `comm.reduce(local_partials, root=0)` or `comm.gather()`
- rank 0 calls `finalize_*_root()` and writes pickle output

Not inside reducers:

- trajectory reading
- MPI broadcast
- output-file writing

Not inside workflows:

- MPI frame slicing
- engine cache ownership
- reducer-side MPI math

---

## Legal Downstream Access Pattern

```text
workflow / scheduler
    |
    v  build task spec with steps
MPIComputeEngine.run_post(spec)
    |
    v  pickle result files
trainer.step(batch) or solver.solve(batch)
    |
    v  Forcefield update
```

Allowed:

- launch MPI post tasks
- read pickle outputs
- assemble trainer / solver batches

Strongly discouraged:

- calling reducers directly as the workflow's primary production interface
- implementing MPI frame splitting in workflow code
- owning cache invalidation in workflow code
- reproducing compute-side accumulation in workflow code because legacy helper functions exist

---

## Core Runtime Classes

| Class | Why it matters |
|---|---|
| `MPIComputeEngine` | Core of the task-scoped MPI runtime |
| `TrajectoryCache` | Per-rank per-frame observable storage, gatherable to rank 0 |
| `FrameCache` | Lightweight per-frame geometry digest |
| `TopologyArrays` | Frozen topology snapshot broadcast to workers |
| `Forcefield` | Snapshot of model parameters and masks |
| `FrameGeometry` | Immutable per-frame geometry object passed to energy / force kernels |

Once these six objects are understood, most runtime behavior becomes predictable.

`FrameGeometry` also covers same-frame coordinate batches. Interaction index
arrays remain shared per key while distances, vectors, angles, and dihedrals
preserve arbitrary leading sample dimensions.
