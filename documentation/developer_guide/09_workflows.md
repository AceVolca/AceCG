# 09 Workflow Module Developer Reference

*Updated: 2026-05-05.*

> This chapter covers training / orchestration workflows only. VP grower is documented separately in [10_vp_grower.md](10_vp_grower.md).

The workflow layer sits above scheduler, trainer, and solver. It connects config, topology, resources, and batch construction. It owns iteration directory layout, checkpoint semantics, task planning, and trainer / solver call timing. It does not own reducer math, MPI broadcast, frame slicing, or table/potential kernels.

---

## Core Files

| File | Responsibility |
|---|---|
| `workflows/base.py` | `BaseWorkflow`, CLI override parser, topology / optimizer / resource builders |
| `workflows/sampling.py` | `SamplingWorkflow`, AA stats, force-field staging, scheduler / sampler construction |
| `workflows/rem.py` | `REMWorkflow` |
| `workflows/cdrem.py` | `CDREMWorkflow` |
| `workflows/cdfm.py` | `CDFMWorkflow` |
| `workflows/fm.py` | `FMWorkflow` |
| `workflows/dsm.py` | `DSMWorkflow` |
| `workflows/__init__.py` | Public workflow class exports |

---

## Layer Boundaries

```text
config + topology + scheduler + trainer/solver
  -> workflow wires these objects together

workflow
  -> plans tasks, writes snapshots, reads post pickles,
     builds batches, calls trainer.step() / solver.solve()

compute / reducers / task_runner
  -> perform one post computation inside a task;
     workflow only consumes the result
```

Workflows should own:

- run directory layout and iteration naming
- task spec assembly
- checkpoint / resume
- trainer / solver selection and batch schema assembly

Workflows should not own:

- direct reducer calls as the main path
- `MPIComputeEngine` internals
- per-frame trajectory accumulation logic
- numerical details of force / energy Jacobians

---

## `BaseWorkflow`

`BaseWorkflow` is the root class for all training workflows. Construction does three things:

- parse and apply config overrides
- create `output_dir`
- build `TopologyArrays` from `system.topology_file`

Important shared helpers:

| Method | Purpose |
|---|---|
| `_run_workflow_cli()` | Shared CLI wrapper for `acg-rem`, `acg-fm`, `acg-dsm`, and related entry points |
| `_apply_config_overrides()` | Supports `--section.field value` style overrides |
| `_build_topology()` | Builds `TopologyArrays` from current config |
| `_build_optimizer()` | Selects optimizer from `training.optimizer` / `training.trainer` |
| `_build_resource_pool()` | Discovers MPI backend and CPU resources from environment and scheduler config |
| `_build_forcefield_mask()` | Compiles `[system] forcefield_mask` into runtime `param_mask` |
| `_build_forcefield_bounds()` | Compiles `[system] forcefield_bounds` into runtime `param_bounds` |

Important development rule: `BaseWorkflow.run()` remains abstract. Real training loops belong in concrete workflow classes.

---

## `SamplingWorkflow`

`SamplingWorkflow` is the shared base for REM, CDREM, and CDFM. It adds:

- `ReadLmpFF()` force-field loading
- combination of VP mask, user `forcefield_mask`, and user `forcefield_bounds`
- `TaskScheduler` and `BaseSampler` / `ConditionedSampler` construction
- unified `beta = 1 / (k_B T)` derivation
- AA reference-data strategy and workflow checkpoint I/O

Key helpers:

| Method | Purpose |
|---|---|
| `_build_forcefield()` | Build runtime `Forcefield` from LAMMPS settings |
| `_build_scheduler()` | Build `TaskScheduler` |
| `_build_sampler()` | Build free-sampling `BaseSampler` |
| `_build_aa_data_strategy()` | Choose constant AA stats or per-epoch recomputation for REM paths |
| `_run_aa_post()` | Run one REM-style post on the AA trajectory with the current forcefield |
| `_aa_noise_runtime_spec()` | Build epoch-local noisy AA-reference specs for REM AA-side post |
| `_snapshot_forcefield()` | Write a force-field pickle for MPI post |
| `_snapshot_optimizer()` | Save Adam/AdamW/RMSprop internal state |
| `_write_workflow_checkpoint()` | Save full resume state |
| `_load_workflow_checkpoint()` | Restore from the latest completed epoch |

`SamplingWorkflow` itself does not define a training loop. It only collects common behavior for workflows that run simulations.

When `aa_ref.noise_enabled = true`, `SamplingWorkflow` builds noisy
AA-reference statistics for REM through the ordinary `_run_aa_post()` path.
Noise stage is `epoch // aa_ref.noise_update_interval`; `noise_schedule` and
`noise_sigma_final` resolve the stage-local sigma; `noise_cache_policy =
stage` can reuse AA statistics when the active gradient convention is
cacheable. `aa_ref.noise_subsample_per_epoch` selects a sorted, no-replacement
frame subset after the usual AA frame slicing.

---

## Concrete Workflows

### `REMWorkflow`

`REMWorkflow` is the thinnest single-sampling iterator:

1. Export the current forcefield once per epoch.
2. Use `BaseSampler` to launch one free-CG xz task.
3. Read the `step_mode="rem"` `result.pkl`.
4. Build `REMTrainerAnalytic.make_batch(...)`.
5. Call `trainer.step(batch)` and write parameters back to the runtime forcefield.

AA-side statistics are selected by `_build_aa_data_strategy()`: linear
gauge-free paths may read cached stats directly, while nonlinear paths may
recompute per epoch.

Noisy REM is implemented on the AA positive-phase side. The AA post spec
receives `spec["noise"]`, and `MPIComputeEngine` computes noisy same-frame
batches before reducer accumulation. The CG xz sampling branch remains the
ordinary sampled-CG REM branch.

### `CDREMWorkflow`

`CDREMWorkflow` adds a `ConditionedSampler` on top of `SamplingWorkflow`:

- xz branch uses free-sampling `BaseSampler`
- zbx branch uses `ConditionedSampler`
- each epoch launches `1 + K` tasks

`_build_xz_task()` and `_build_zbx_task()` both use `step_mode="cdrem"` for post-processing. The lower reducer reuses the REM path. `_collect_cdrem_batch()` combines xz and zbx `result.pkl` files into a `CDREMBatch`.

### `CDFMWorkflow`

`CDFMWorkflow` is zbx-only and has no free xz sampling:

- `_build_sampler()` is explicitly overridden to `None`
- `init_config_pool` and `init_force_pool` are paired by frame id
- `_install_cdfm_mask()` narrows the default training mask to VP-only / mixed channels
- every zbx task uses `step_mode="cdfm_zbx"`

In the active repo, `y_eff` no longer has a separate `cdfm_y_eff` step. It is computed during the rank-0 preprocessing phase of `cdfm_zbx` from `(init_config, init_force)` and then broadcast to all ranks in that replica.

### `FMWorkflow`

`FMWorkflow` inherits directly from `BaseWorkflow` because it does not need the iterative sampler / scheduler layer.

Its two special pieces are:

- `_build_fm_forcefield()`, which builds the training forcefield from `training.fm_specs`
- `_run_post_accumulation()`, which directly calls `schedulers.task_runner.run_post(...)` to accumulate `step_mode="fm"` statistics over the AA trajectory

FM has two execution paths:

| Path | Condition | Behavior |
|---|---|---|
| trainer path | `fm_method=iterator`, or auto selects a first-order optimizer | Accumulate FM stats each round and call `FMTrainerAnalytic.step()` |
| solver path | `fm_method=solver`, or auto selects Newton / closed-form | Accumulate FM stats once and call `FMMatrixSolver.solve()` |

After solving, `_export_table_bundle()` exports the result as LAMMPS tables.

Noisy FM is a post-processing option on the AA reference path. If
`aa_ref.noise_enabled = true`, `_fm_noise_runtime_spec()` adds `spec["noise"]`
to the FM post spec. Coordinate replicas are same-frame batches handled by
`MPIComputeEngine.compute()`; reducers still see ordinary FM statistics.

Mixed noisy FM is enabled with `aa_ref.noise_force_mix_ratio = alpha`.
The original included frame, when present, keeps the unmodified reference-force
target. Noisy replicas use:

```text
(1 - alpha) * y_ref + alpha * (-delta / (beta * sigma^2))
```

This path requires `training.temperature` or `training.beta`, positive noise
sigma, and at least one noisy replica. It is supported only for
`training.method = fm`.

### `DSMWorkflow`

`DSMWorkflow` subclasses `FMWorkflow` but forces the iterative trainer path.
It uses `training.fm_specs` to build the trainable forcefield and requires
`aa_ref.noise_enabled = true`, positive resolved `noise_sigma`, and positive
`training.temperature` / `training.beta`.

Each epoch writes:

```text
dsm_step_XXXX/
  forcefield.pkl
  dsm_batch.pkl
  ff/
    optimizer_snapshot.pkl
    workflow_checkpoint.pkl
```

The post spec uses `step_mode="dsm"` plus a noise runtime spec with
`target="dsm"` and `beta`. The compute layer generates the score target
`-delta / (beta * sigma^2)`, and the workflow scales the returned FM-shaped
statistics by `beta^2` before calling `FMTrainerAnalytic.step()`.

To resume, set `training.start_epoch = N`; DSM restores
`dsm_step_{N-1}/ff/workflow_checkpoint.pkl` before running epoch `N`.

---

## Directory Conventions

Typical sampling workflow layout:

```text
iter_0000/
  ff/
    forcefield_snapshot.pkl
    optimizer_snapshot.pkl
    workflow_checkpoint.pkl
  xz/
  zbx/
```

FM workflow layout:

```text
fm_step_0000/
  forcefield.pkl
  fm_batch.pkl
```

DSM workflow layout:

```text
dsm_step_0000/
  forcefield.pkl
  dsm_batch.pkl
  ff/
    optimizer_snapshot.pkl
    workflow_checkpoint.pkl
```

The most important debugging files are:

- `ff/workflow_checkpoint.pkl`, the canonical resume entry point
- each replica's `result.pkl`, the direct compute post output

---

## Development Rules

When adding a workflow, prefer these rules:

1. Only workflows decide task count, task type, and batch schema.
2. Treat reducer output as input; do not copy reducer math into workflow code.
3. Checkpoints must save `Forcefield`, optimizer state, and workflow RNG state together.
4. One-shot VP grower data production should stay in the separate [10_vp_grower.md](10_vp_grower.md) pipeline, not inside training workflows.
