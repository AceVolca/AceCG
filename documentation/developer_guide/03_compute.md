# 03 Compute Module Developer Reference

*Updated: 2026-05-24.*

The compute layer is the task-scoped numerical runtime between trajectory/topology I/O and trainers.

It owns:

- `FrameGeometry`
- `MPIComputeEngine`, including `FrameCache` and `TrajectoryCache`; legacy aliases are `FrameObservables` and `TrajectoryObservablesCache`
- the `energy()` and `force()` kernels
- the stateful reducer pipeline in `reducers.py`

It does not own:

- scheduler policy
- workflow orchestration
- trainer tallies across tasks
- optimizer policy

---

## Core Files

| File | Responsibility |
|---|---|
| `compute/frame_geometry.py` | Immutable per-frame geometry view |
| `compute/mpi_engine.py` | Task runtime: MPI broadcast, frame extraction, post dispatch, observable cache |
| `compute/energy.py` | `energy()` kernel |
| `compute/force.py` | `force()` kernel |
| `compute/reducers.py` | Stateful pipeline reducers: init / consume / finalize per `step_mode` |
| `compute/registry.py` | `build_default_engine()`, which registers core observables |

---

## Three Stable Public APIs

```python
from AceCG.compute import compute_frame_geometry, energy, force

geom = compute_frame_geometry(positions, box, topology_arrays, interaction_mask=None)

energy_dict = energy(
    geom, forcefield,
    return_value=True,
    return_grad=True,
    return_hessian=False,
    return_grad_outer=False,
)

force_dict = force(
    geom, forcefield,
    return_value=True,
    return_grad=True,
    return_hessian=False,
    return_fm_stats=False,
)
```

All other compute symbols should be treated as internal reducer helpers, parameter helpers, or post-processing bridges, not stable public APIs.

All three APIs accept ordinary single-frame inputs and same-frame coordinate
batches. `positions` may have shape `(..., n_atoms, 3)`, and `box` may have
shape `(6,)` or `(..., 6)`. There are no separate public `energy_batch()` or
`force_batch()` kernels.

---

## `MPIComputeEngine`

`MPIComputeEngine` is the compute runtime object.

An engine instance owns:

- an MPI communicator, `comm`, which may be `None`
- a registry of observables
- `serial_threshold`, below which frame counts are handled serially

Two stable public entry points:

```python
engine = build_default_engine(comm=comm)

result = engine.compute(
    request,
    frame,                # (frame_id, positions, box, forces)
    topology_arrays,
    forcefield_snapshot,
    frame_weight=1.0,
    frame_weights=None,
    batch_size=None,
    neighbor_mode="shared",
    ...
)

engine.run_post(spec)
```

`engine.compute()` is local and does not perform MPI:

- accepts a localized `frame` tuple
- builds `FrameGeometry`
- evaluates registered observables according to `request`
- returns a dict, including `frame_cache` for reducer cache requests and
  `frame_observables` when `return_observables=True`

`engine.run_post(spec)` is the scheduled-task boundary:

- rank 0 loads the force-field snapshot, MDAnalysis Universe, and `TopologyArrays`
- shared context is broadcast to all ranks
- each rank receives a contiguous frame chunk or a split of `discrete_ids`
- one-pass steps call the reducer pipeline
- MPI reduce -> rank 0 finalize -> pickle output
- see [04_mpi_runtime.md](04_mpi_runtime.md)

### Same-Frame Batch Contract

`MPIComputeEngine.compute()` is the only local frame compute entry point for
both single frames and noisy same-frame batches.

For a batched frame:

- `positions` has shape `sample_shape + (n_atoms, 3)`
- `box` has shape `(6,)` or `sample_shape + (6,)`
- `reference_forces` may be unbatched or batched in the same leading shape
- `frame_weights` supplies per-sample weights and is normalized inside the call
- scalar `frame_weight` remains the outer per-base-frame reducer weight

The engine flattens `sample_shape`, processes the flat sample axis in
`batch_size` chunks when requested, and folds sample outputs into the same
single-frame payload keys consumed by reducers. Reducers do not receive
`batch_payload`, `*_batch`, or `consume_batch` inputs.

Pair-cache policy for noisy/same-frame batches is controlled by
`neighbor_mode`:

| Mode | Behavior |
|---|---|
| `shared` | Build one true-cutoff pair cache from the first sample and reuse it. This is the default fast path. |
| `skin` | Build one `cutoff + neighbor_skin` candidate cache from the supplied unnoised reference frame; energy/force kernels still mask each potential to its true cutoff. |
| `chunk` | Rebuild the pair cache once per `batch_size` chunk. With `batch_size = 1`, this is the exact high-noise path. |

Current frame-cache semantics are per real trajectory frame. Batched compute
rejects `FrameCache` / observable-cache requests.

---

## Observable Cache: `FrameCache` and `TrajectoryCache`

`run_post()` supports a per-frame geometry digest cache for downstream analysis:

```python
spec["collect_observables"] = True
spec["gather_observables"] = True
spec["observables_output_file"] = "cache.pkl"
```

| Class | Location | Contents |
|---|---|---|
| `FrameCache` | `compute/mpi_engine.py` | Lightweight geometry digest for one frame: pair distances, bond lengths, angles, dihedrals, box |
| `TrajectoryCache` | `compute/mpi_engine.py` | Collection of `FrameCache` objects for a trajectory |
| `geometry_to_observables()` | `compute/mpi_engine.py` | Extracts a `FrameCache` from `FrameGeometry` |

The old names `FrameObservables` and `TrajectoryObservablesCache` remain as compatibility aliases. New code and documentation should prefer `FrameCache` and `TrajectoryCache`.

This mechanism lets `analysis/rdf.py` access already-computed geometry without rereading the trajectory.

---

## Reducer Pipeline

`reducers.py` provides a five-function pipeline for each `step_mode`:

```python
state = init_*_state(step)
req = request_*(step)
consume_*_frame(state, frame_result)
partials = local_partials_*(state)
plan = reduce_plan_*(step)
result = finalize_*_root(state)
```

Current reducer families:

| `step_mode` | Init function |
|---|---|
| `fm` | `init_fm_state` |
| `dsm` | `init_fm_state` with DSM request flags |
| `rem` / `cdrem` | `init_rem_state` |
| `cdfm_zbx` | `init_cdfm_zbx_state` |
| `rdf` | Uses `analysis.rdf.interaction_distributions()` directly and does not enter the one-pass pipeline |

`cdfm_y_eff` has been deprecated. `y_eff` is now computed during the rank-0 preprocessing phase of `cdfm_zbx` from each replica's init configuration and paired `.forces.npy` reference-force file:

```text
y_eff = y_ref - f_theta_cg_only(R_init)
```

The mask is temporarily switched to CG-only during this calculation and restored afterward.

Reducer conventions:

- reducers perform local math and call `engine.compute()` for observables
- reducers do not read trajectories, perform MPI broadcast, or write output files
- MPI reduce is handled uniformly by `run_post()`
- reducers consume reducer-ready single-frame payloads even when
  `engine.compute()` internally processed a noisy coordinate batch

---

## Leaf Step Output Contracts

All scheduled compute outputs are pickle files. Each payload is a top-level dict whose values are ndarrays and simple scalars.

### `step_mode="rem"` / `"cdrem"`

| Key | Meaning |
|---|---|
| `energy_grad_avg` | Weighted average `<dU/dtheta>` |
| `n_frames` | Number of contributing frames |
| `weight_sum` | Total frame weight |
| `d2U_avg` | Optional weighted-average Hessian |
| `grad_outer_avg` | Optional gradient outer-product average |
| `energy_grad_frame` | Optional per-frame gradient stack |

For `step_mode="rem"`, energy-gradient requests use the gauge-free gradient
channel. This removes parameter-dependent bonded B-spline gauge shifts from
REM statistics. `step_mode="cdrem"` stays on the physical energy-gradient
channel unless a coordinate mask is configured, in which case the reducer uses
gauge-free gradients for the masked statistic.

REM-style steps may carry `energy_mask`, a JSON-friendly mapping keyed by
interaction label. The compute kernel applies this mask to geometry coordinates
before summing energy-gradient channels. The mask is a coordinate selector, not
a parameter trainability selector; `Forcefield.param_mask` is applied afterward
to all gradient channels. Masked runs also emit `energy_mask_diagnostics` with
active/total counts and per-interaction fractions.

When a step sets `need_aux_unmasked_energy_grad`, the reducer requests an
additional unmasked energy-gradient channel from the same frames. This channel
is used only by trainer optimizer-gradient policies such as `hybrid_aux` and
`unmasked`; ordinary masked REM/CDREM statistics do not need it.

### `step_mode="fm"`

| Key | Meaning |
|---|---|
| `JtJ` | Normalized weighted FM normal matrix |
| `Jty` | Normalized weighted FM right-hand-side vector |
| `y_sumsq` | Normalized weighted target-force squared norm |
| `Jtf` | Normalized weighted `J^T f` |
| `f_sumsq` | Normalized weighted model-force squared norm |
| `fty` | Normalized weighted `f^T y` |
| `nframe` | Number of contributing frames |
| `weight_sum` | Total frame weight |
| `n_atoms_obs` | Number of observed atoms per frame |

### `step_mode="dsm"`

DSM uses the FM reducer state and emits the same payload keys as
`step_mode="fm"`. The difference is the request contract: DSM asks for FM
statistics without requiring reference forces from the trajectory. The
synthetic score target is generated by `MPIComputeEngine.add_noise()` when the
runtime noise spec contains `target = "dsm"`. `DSMWorkflow` then multiplies the
FM-style statistics by `beta^2` before calling `FMTrainerAnalytic.step()`.

### `step_mode="cdfm_zbx"`

Required spec keys:

| Key | Meaning |
|---|---|
| `init_force_path` | Reference-force `.npy` paired with this replica init configuration; shape `(n_real, 3)` or `(3*n_real,)` |
| `init_frame_id` | Init-configuration frame id for `FrameCache.frame_idx`; defaults to 0 |
| `mode` | `"direct"` or `"reinforce"` |
| `beta` | Temperature factor for reinforce mode; optional for direct mode |

Output payload:

| Key | Meaning |
|---|---|
| `grad_direct` | Direct gradient contribution |
| `grad_reinforce` | REINFORCE gradient contribution |
| `sse` | Sum of squared error |
| `obs_rows` | Observation-row count |
| `n_samples` | Sample count |
| `rmse` | `sqrt(sse / obs_rows)` |

---

## `multi` Mode

When `spec["steps"]` contains multiple steps, all one-pass steps share a single trajectory traversal:

```text
run_post(spec with multiple steps):
  prepare shared context once
  extract local frames once
  for each frame:
    result = engine.compute(...)
    consume_step0_frame(state0, result)
    consume_step1_frame(state1, result)
    ...
  for each step:
    comm.reduce(local_partials)
    finalize_*_root()
    pickle output
```

Contract:

- all one-pass steps in `steps` must share the same trajectory, topology, and context
- if different trajectory or geometry context is needed, use a separate task
- `rdf` steps do not enter the one-pass pipeline; they run separately after one-pass steps finish

---

## Frame-Weight Semantics

`spec["frame_weight"]` always means task-local trajectory weights consumed inside the scheduled task.

It applies to all `step_mode` values.

It is not a cross-task weight such as CDREM `x_weight` or CDFM `x_weight`. Those belong to trainer batch schemas and are handled by workflows and trainers.

---

## Allowed Workflow Access Patterns

Do:

- build a task spec and pass it to `MPIComputeEngine.run_post()`
- use `multi` when several leaf reductions share the same trajectory, topology, and context
- consume pickle payloads to assemble trainer or solver batches
- treat `engine.compute()` only as a local observable helper

Do not:

- call reducers directly as the primary production API
- own MPI broadcast, MPI reduction, trajectory slicing, or cache lifetime in workflow code
- reimplement compute-side accumulation in workflow code just because legacy helper functions exist

---

## Required Task Spec Keys

| Key | Meaning |
|---|---|
| `work_dir` | Task working directory |
| `forcefield_path` | Force-field pickle path |
| `topology` | Topology file path |
| `trajectory` | Trajectory path as `str`, or path list as `list[str]` |
| `steps` | List of steps; each step must have `step_mode` and `output_file` |

Common optional keys:

| Key | Meaning |
|---|---|
| `trajectory_format` | MDAnalysis format string; default `LAMMPSDUMP` |
| `frame_start` / `frame_end` | Inclusive / exclusive frame range |
| `every` | Sampling stride; default 1 |
| `frame_ids` | Discrete frame selection |
| `frame_weight` | Task-local frame weights |
| `sel` | Atom-selection expression; default `"all"` |
| `cutoff` | Nonbonded cutoff |
| `exclude_option` | Pair-search exclusion mode |
| `exclude_bonded` | Bonded exclusion mask, such as `"111"` |
| `atom_type_name_aliases` | Atom type int -> name map |
| `vp_names` | Virtual-site names |
| `collect_observables` | Whether to collect `FrameCache` |
| `gather_observables` | Whether to gather caches to rank 0 |
| `observables_output_file` | Observable-cache pickle output path |
| `noise` | Epoch-local noisy AA-reference runtime spec for FM, REM, or DSM |

## Noisy AA-Reference Runtime Spec

Workflows pass noisy coordinate augmentation to `run_post()` through
`spec["noise"]`. Config parsing lives in `[aa_ref]` as flat `noise_*` fields,
but the compute layer receives one epoch-local runtime dict with the resolved
sigma and seed.

Common fields:

| Runtime key | Meaning |
|---|---|
| `samples_per_frame` | Number of noisy coordinate replicas |
| `sigma` | Resolved coordinate-noise standard deviation for this stage |
| `seed` | Base seed; frame id is mixed into the per-frame RNG seed |
| `selection` | Atom subset to perturb |
| `include_original` | Include the unperturbed frame as one sample |
| `wrap` | Wrap noisy coordinates back into the periodic box |
| `batch_size` | Maximum same-frame samples per compute chunk |
| `subsample_per_epoch` / `subsample_seed` | Sorted no-replacement frame subset for this epoch |
| `neighbor_mode` / `neighbor_skin` | Same-frame pair-cache policy |
| `force_mix_ratio` | Mixed noisy-FM score-target blend for noisy replicas |
| `target = "dsm"` / `beta` | DSM score-target mode |

Default noisy FM broadcasts the original reference-force target to coordinate
replicas. Noisy REM evaluates REM energy statistics on noisy coordinate
replicas. Mixed noisy FM uses `force_mix_ratio = alpha` on noisy replicas only:

```text
y = (1 - alpha) * y_ref + alpha * (-delta / (beta * sigma^2))
```

An included original frame keeps the unmodified `y_ref` target. DSM sets
`target = "dsm"` and uses only the synthetic score target
`-delta / (beta * sigma^2)`.
