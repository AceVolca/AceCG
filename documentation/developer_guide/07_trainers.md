# 07 Trainer Module Developer Reference

*Updated: 2026-05-05.*

The trainer layer sits above `compute/` and next to `solvers/`. It consumes workflow-built batch statistics and produces gradients, Hessians, update steps, and diagnostics. It does not own MPI execution, trajectory extraction, or reducer runtime state.

---

## Core Modules

| File | Responsibility |
|---|---|
| `trainers/base.py` | `BaseTrainer`, shared trainer contract |
| `trainers/analytic/rem.py` | REM trainer and batch schema |
| `trainers/analytic/mse.py` | PMF-matching MSE trainer based on PMFs, bin assignments, and per-frame energy gradients |
| `trainers/analytic/fm.py` | Iterative FM/DSM trainer consuming the standard FM reducer payload |
| `trainers/analytic/cdrem.py` | Latent-variable CDREM trainer |
| `trainers/analytic/cdfm.py` | CDFM gradient consumer with EM guardrail handling |
| `trainers/analytic/multi.py` | Meta-trainer that combines multiple child trainers |
| `trainers/autodiff/` | Placeholder package for future autodiff trainers |

`AceCG.trainers` currently re-exports analytic trainers only.

---

## Layer Boundaries

```text
topology + compute
  -> Forcefield-owned masks, reducer outputs, workflow-built batches

trainer
  -> consumes one batch, computes grad/update, mutates its own Forcefield + optimizer

workflow
  -> decides which batch to build, when to call trainer.step(), and how to log
```

A trainer should own:

- a private `Forcefield` copy
- a private optimizer instance
- batch-to-gradient mathematics
- trainer-local logs and diagnostics

A trainer should not own:

- `MPIComputeEngine`
- `run_post()`
- trajectory I/O or MDAnalysis objects
- reducer-side accumulation loops
- solver-style exact linear solves

---

## `BaseTrainer`

### Responsibilities

| Method | Meaning |
|---|---|
| `__init__(forcefield, optimizer, beta=None, logger=None)` | Deep-copy and own a `Forcefield` and `BaseOptimizer` |
| `get_params()` | Return the current full parameter vector |
| `update_forcefield(L_new)` | Write a full parameter vector into forcefield and optimizer |
| `clamp_and_update()` | Apply bounds to optimizer state, then synchronize |
| `get_param_names()` | Return ordered parameter labels |
| `get_param_bounds()` | Return `(lb, ub)` |
| `get_interaction_labels()` | Return interaction labels in forcefield order |
| `n_total_params()` | Total scalar parameter count |
| `active_interaction_mask()` | Derive the L1 interaction activity mask from the optimizer's L2 mask |
| `is_optimization_linear()` | Report whether all active channels are linear |
| `optimizer_accepts_hessian()` | Use the optimizer signature to determine whether Hessians are accepted |
| `step(batch, apply_update=True)` | Abstract single-step trainer entry point |

### Batch Contract

Every concrete trainer is batch-driven:

```python
out = trainer.step(batch, apply_update=True)
```

The workflow owns batch construction. Trainers should not rebuild trajectory state or recompute missing frame statistics on their own.

### Hessian Contract

`BaseTrainer.optimizer_accepts_hessian()` checks whether the optimizer's `step()` signature has a parameter named `hessian`. Optimizer authors who want second-order information must use that exact parameter name.

---

## Public Trainer Interface

`AceCG.trainers` exports:

| Symbol | Meaning |
|---|---|
| `REMTrainerAnalytic`, `REMBatch`, `REMOut` | REM trainer |
| `MSETrainerAnalytic`, `MSEBatch`, `MSEOut` | MSE / PMF-matching trainer |
| `FMTrainerAnalytic`, `FMBatch` | Iterative FM trainer |
| `CDREMTrainerAnalytic`, `CDREMBatch`, `CDREMOut` | Latent-variable REM trainer |
| `CDFMTrainerAnalytic`, `CDFMBatch` | Latent-variable FM trainer |
| `MultiTrainerAnalytic`, `MultiOut` | Composite meta-trainer |

Recommended usage:

- workflows call `TrainerClass.make_batch(...)` or an equivalent batch helper
- workflows pass the batch to `trainer.step(...)`
- trainers do not create compute engines, call `run_post()`, or run reducers

---

## Analytic Trainers

### `REMTrainerAnalytic`

Consumes REM energy statistics and computes:

$$\nabla = \beta \left(\langle dU / d\lambda \rangle_{\text{AA}} - \langle dU / d\lambda \rangle_{\text{CG}}\right)$$

AA-side statistics come from workflow-built batches, usually pickle output from `run_post(step_mode="rem")`. When the optimizer accepts Hessians, REM also consumes second-order statistics.

For REM, compute-side post-processing uses gauge-free energy gradients so
parameter-dependent bonded B-spline gauge shifts do not enter cached AA
statistics or REM gradient averages. CDREM keeps the physical energy-gradient
convention because it remains on the physical latent-energy path.

### `FMTrainerAnalytic`

Consumes the standard FM reducer payload:

| Key | Meaning |
|---|---|
| `JtJ` | Normalized weighted average `J_i^T J_i` |
| `Jty` | Normalized weighted average `J_i^T y_i` |
| `y_sumsq` | Normalized weighted average `y_i^T y_i` |
| `Jtf` | Normalized weighted average `J_i^T f_i` |
| `f_sumsq` | Normalized weighted average `f_i^T f_i` |
| `fty` | Normalized weighted average `f_i^T y_i` |
| `nframe` | Number of contributing frames |

Computes:

$$L = \frac{1}{2}\left(f_\text{sumsq} - 2\,fty + y_\text{sumsq}\right)$$

$$\nabla = Jtf - Jty,\quad H \approx JtJ$$

This is the iterative FM path. Exact closed-form solving lives in `FMMatrixSolver`, not in the trainer.

FM statistics are generated by `run_post(spec, step_mode="fm")`; the workflow reads the pickle output and builds a batch.

Noisy FM and mixed noisy FM still consume this same batch schema. The compute
layer folds same-frame noisy replicas into one reducer-ready FM statistics
payload per base frame. Mixed noisy FM changes only the target construction for
noisy replicas before the FM statistics are accumulated.

### DSM Through `FMTrainerAnalytic`

DSM does not have a separate trainer class. `DSMWorkflow` runs
`step_mode="dsm"`, which emits the same FM-shaped sufficient statistics while
using synthetic score targets generated from noisy coordinate displacements.
Before calling `FMTrainerAnalytic.step()`, the workflow scales the relevant
statistics by `beta^2`, making the reused FM objective equivalent to:

```text
0.5 * || beta * f_theta(x) + delta / sigma^2 ||^2
```

DSM therefore follows the iterative FM trainer path and rejects
`training.fm_method = solver`.

### `MSETrainerAnalytic`

Consumes AA/CG PMFs, CG frame-to-bin assignments, and per-frame CG energy gradients to build a gauge-fixed PMF-matching objective. It no longer relies on the old pair-distance derivative path. The `CG` field in `MSEBatch` is retained only for compatibility with older type signatures; the recommended and currently used input path is `energy_grad_frame`.

Let `s` denote a PMF bin, `F_AA(s)` the reference PMF, and `F_CG(s)` the current CG PMF. Because PMFs are defined only up to an arbitrary additive constant, the trainer first computes a gauge shift:

$$c = \frac{1}{N_\text{bin}} \sum_s \left(F_\text{CG}(s) - F_\text{AA}(s)\right)$$

Then it defines the gauge-fixed mismatch:

$$\Delta F(s) = F_\text{CG}(s) - c - F_\text{AA}(s)$$

The objective is:

$$L_\text{MSE} = \frac{1}{2} \sum_s \Delta F(s)^2$$

The derivative of the CG PMF is estimated from conditional energy gradients:

$$\partial F_{\text{CG}}(s)/\partial \lambda = \langle dU/d\lambda \rangle_{\text{CG}|s} - \langle dU/d\lambda \rangle_{\text{CG}}$$

After gauge fixing, $\sum_s \Delta F(s)=0$, so the global ensemble-average term cancels for the full bin sum. The current implementation accumulates the conditional-mean term over observed bins:

$$\nabla_\lambda L_\text{MSE}
  = \sum_{s \in \mathcal{S}_\text{obs}}
      \Delta F(s)\,\langle dU/d\lambda \rangle_{\text{CG}|s}$$

Here $\mathcal{S}_\text{obs}$ is the set of bins with at least one positive-weight CG frame. Empty bins still contribute to loss and the gauge shift, but not to the gradient, because their conditional averages cannot be estimated from the current frames.

#### MSE Batch Schema

| Key | Required | Meaning |
|---|---:|---|
| `pmf_AA` | yes | Reference PMF, shape `(n_bins,)` |
| `pmf_CG` | yes | Current CG PMF, shape `(n_bins,)` |
| `CG_bin_idx_frame` | yes | Bin index for each frame, shape `(n_frames,)`, values in `[0, n_bins)` |
| `energy_grad_frame` | yes | Per-frame CG energy gradients, shape `(n_frames, n_params)` |
| `frame_weight` | no | Nonnegative per-frame weights, shape `(n_frames,)`; uniform weights are used if omitted |
| `step_index` | no | Logging step, default `0` |

Conditional means are computed as:

$$
\langle dU/d\lambda \rangle_{\text{CG}|s}
=
\frac{\sum_{i: b_i=s} w_i\,g_i}{\sum_{i: b_i=s} w_i}
$$

where `b_i = CG_bin_idx_frame[i]` and `g_i = energy_grad_frame[i]`. If `frame_weight` is not provided, all `w_i = 1`. Weights must be nonnegative, and their total sum must be positive and finite.

#### MSE Usage

Workflows should build batches with `make_batch()`:

```python
batch = MSETrainerAnalytic.make_batch(
    pmf_AA=pmf_ref,
    pmf_CG=pmf_model,
    CG_bin_idx_frame=bin_idx_frame,
    energy_grad_frame=energy_grad_frame,
    frame_weight=frame_weight,      # optional
    step_index=iteration_index,
)
out = trainer.step(batch, apply_update=True)
```

`step()` returns the standard trainer output:

| Key | Meaning |
|---|---|
| `name` | Always `"MSE"` |
| `loss` | Gauge-fixed PMF mismatch objective |
| `grad` | MSE gradient, shape `(n_params,)` |
| `hessian` | Currently `None`; retained for interface uniformity |
| `update` | Optimizer-returned parameter update; `zeros_like(grad)` in dry-run mode |
| `meta` | Includes `gauge_shift`, `frame_weight_source`, `n_observed_bins`, `missing_bins`, `grad_norm`, and `update_norm` |

Use `apply_update=False` for dry runs, meta-trainers, or gradient debugging. In that mode the optimizer is not called and the trainer's forcefield is not mutated.

### `CDREMTrainerAnalytic`

Consumes latent-variable conditional and joint derivative statistics, then computes CDREM gradients and optional Hessians.

### `CDFMTrainerAnalytic`

Consumes by-`x` batch arrays:

- `grad_direct_by_x`
- `grad_reinforce_by_x`
- `sse_by_x`
- `n_samples_by_x`
- `obs_rows`
- optional `x_weight`
- `mode`

Inside `step()`, it tallies over `x`, applies guardrails, REINFORCE clipping, masks, and optimizer stepping. It does not build trajectory statistics on its own.

#### CDFM Data Channels

CDFM no longer uses a `cdfm_y_eff` preprocessing step. Each zbx replica declares two paired glob patterns in `.acg`:

```ini
[conditioning]
init_config_pool = conditioning/frame_*.data
init_force_pool  = conditioning/frame_*.forces.npy
mask_cg_only     = true
```

The two pools must match one-to-one by frame id. `init_force_pool` frame ids are extracted by `AceCG.configs.utils.extract_frame_id_from_force_file()`. If a filename contains multiple numbers, they must all agree, such as `frame_000035.forces.npy` or `frame_35_rep35.npy`.

During `CDFMWorkflow.__init__`, `_install_cdfm_mask()` calls `forcefield.build_mask(init_mask=init_mask)` according to `mask_cg_only`. When `True`, the current mask is ANDed with `~real_mask`, freezing all CG-only, non-VP terms. When `False`, the existing mask is preserved and CG-only channels may participate in CDFM gradient updates.

The CG-only force baseline `f_theta_cg_only(R_init)` is evaluated by the `cdfm_zbx` preprocessing block in `run_post()` by temporarily swapping to `real_mask` and restoring the mask afterward. It is not affected by `mask_cg_only`.

### `MultiTrainerAnalytic`

Combines multiple child trainers at the trainer layer. Two modes exist:

| Mode | Meaning |
|---|---|
| `update` | Each child trainer updates independently; the meta-trainer combines updates |
| `grad` | Each child trainer dry-runs and returns grad/Hessian; the meta-trainer performs one optimizer step |

`MultiTrainerAnalytic` is not a shared compute object. Shared reducer work and cache reuse must be handled below it, in workflow or compute.

---

## Workflow Contract

Workflows are responsible for:

1. selecting which trainer to use
2. building the correct batch dict
3. calling `trainer.step(batch, apply_update=...)`
4. deciding whether to switch to solver mode
5. logging and checkpoint policy

Minimal FM call pattern:

```python
batch = FMTrainerAnalytic.make_batch(
    JtJ=stats["JtJ"],
    Jty=stats["Jty"],
    y_sumsq=stats["y_sumsq"],
    Jtf=stats["Jtf"],
    f_sumsq=stats["f_sumsq"],
    fty=stats["fty"],
    nframe=stats["nframe"],
    step_index=iteration_index,
)
out = trainer.step(batch, apply_update=True)
```

Here `stats` is read from the pickle output of `run_post(spec)`.
