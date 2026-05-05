# 01 Potential Module Developer Reference

*Updated: 2026-05-05.*

Potentials are the lowest-level scalar interaction models in the modeling stack. Each potential evaluates one interaction coordinate and exposes parameter derivatives for `compute/`, trainers, and solvers. A potential does not know about MPI, `FrameGeometry`, atom ownership, or workflows.

---

## Core Modules

| File | Responsibility |
|---|---|
| `potentials/base.py` | `BasePotential` and `IteratePotentials()` |
| `potentials/harmonic.py` | Harmonic bonded / angle potential |
| `potentials/gaussian.py` | Normalized Gaussian pair potential |
| `potentials/lennardjones.py` | Lennard-Jones 12-6 pair potential |
| `potentials/lennardjones96.py` | Lennard-Jones 9-6 pair potential |
| `potentials/lennardjones_soft.py` | Soft-core LJ variant |
| `potentials/bspline.py` | Force-basis B-spline potential |
| `potentials/multi_gaussian.py` | Normalized multi-Gaussian family |
| `potentials/unnormalized_multi_gaussian.py` | Unnormalized LAMMPS-style multi-Gaussian |
| `potentials/srlrgaussian.py` | SR/LR Gaussian pair potential |

`AceCG.potentials.__init__` also defines `POTENTIAL_REGISTRY`, the mapping from LAMMPS-style names to concrete potential classes.

---

## Layer Boundaries

```text
topology / Forcefield
  -> key ordering, masks, bounds, parameter concatenation

potential
  -> value(r), force(r), derivatives with respect to local parameters

compute
  -> calls potentials through Forcefield and FrameGeometry,
     assembling per-frame energy, force, Jacobian, and Hessian data
```

A potential should own:

- the local parameter vector for one interaction model
- scalar evaluation rules
- parameter-derivative channels
- optional local bounds and linearity metadata

A potential should not own:

- atom indices
- MPI or frame iteration
- pair search
- Cartesian force assembly
- workflow or trainer state

---

## Public Potential Interface

`AceCG.potentials` exports:

| Symbol | Meaning |
|---|---|
| `BasePotential` | Abstract base class |
| `IteratePotentials` | Flattened `(key, potential)` iteration helper |
| `BSplinePotential` | Force-basis spline |
| `GaussianPotential` | Normalized Gaussian |
| `HarmonicPotential` | Harmonic potential |
| `LennardJonesPotential` | LJ 12-6 |
| `LennardJones96Potential` | LJ 9-6 |
| `LennardJonesSoftPotential` | Soft-core LJ |
| `MultiGaussianPotential` | Normalized multi-Gaussian |
| `UnnormalizedMultiGaussianPotential` | Unnormalized multi-Gaussian |
| `SRLRGaussianPotential` | SR/LR Gaussian |
| `POTENTIAL_REGISTRY` | Style-name to class lookup table |

---

## `BasePotential`

### Required Methods

| Method | Meaning |
|---|---|
| `value(r)` | Scalar potential energy at local coordinate `r` |
| `force(r)` | Scalar force, `F = -dU/dr` |

`r` is a local scalar coordinate array, not Cartesian coordinates:

- pair potential: pair distance
- bond potential: bond length
- angle potential: angle coordinate
- dihedral potential: dihedral coordinate

The input can have any shape. Potential evaluators must preserve the input
shape for scalar channels:

```python
r.shape              -> value(r).shape
r.shape              -> force(r).shape
r.shape + (n_params,) -> energy_grad(r).shape
r.shape + (n_params,) -> force_grad(r).shape
```

In batched compute, `r` usually has shape `sample_shape + (n_terms,)`. The
potential should treat every leading dimension as a same-frame batch dimension
and the last axis as the interaction-term axis.

### Common Metadata

Subclasses should populate:

| Field | Meaning |
|---|---|
| `_params` | Current parameter vector |
| `_param_names` | Parameter labels |
| `_dparam_names` | First energy-derivative channel names |
| `_d2param_names` | Second energy-derivative channel names |
| `_df_dparam_names` | First force-derivative channel names |
| `_param_linear_mask` | Per-parameter linearity mask |
| `_params_to_scale` | Parameters affected by `get_scaled_potential(z)` |

### Common Helpers

| Method | Meaning |
|---|---|
| `get_params()` / `set_params()` | Parameter access |
| `n_params()` | Number of local scalar parameters |
| `param_names()` | Parameter labels |
| `energy_grad(r)` | Stacked `dU/dtheta` channels |
| `force_grad(r)` | Stacked `dF/dtheta` channels |
| `energy_grad_sum(r)` | `dU/dtheta` summed over the last coordinate axis |
| `energy_grad_sum_by_sample(r, active=...)` | Per-sample summed gradients for `sample_shape + (n_terms,)` inputs |
| `gauge_free_energy_grad_sum(r)` | Summed gradients with parameter-dependent gauge shifts removed |
| `gauge_free_energy_grad_sum_by_sample(r, active=...)` | Per-sample gauge-free summed gradients |
| `basis_values(r)` | Per-parameter force-basis values |
| `basis_derivatives(r)` | Force-basis derivatives with respect to `r` |
| `is_param_linear()` | Linearity mask |
| `is_gauge_free_energy_grad_cacheable()` | Per-parameter mask for REM AA-cacheable gauge-free channels |
| `get_scaled_potential(z)` | Return a copy with selected parameters scaled |

---

## Scientific Contract

### Scalar Force Convention

All potentials return scalar force:

$$F(r) = -\frac{dU}{dr}$$

The compute layer projects scalar quantities back to Cartesian atom forces through the geometric direction vectors in `FrameGeometry`.

### Energy Derivative Channels

`compute.energy` builds:

- `energy`
- `energy_grad`, first derivative with respect to parameters
- `energy_hessian`, second derivative with respect to parameters
- `energy_grad_outer`, outer product of the full accumulated gradient vector
- `gauge_free_energy_grad` and `gauge_free_energy_grad_outer` for REM

These come from `value(r)`, `_dparam_names`, and `_d2param_names`.

For batched geometry, `compute.energy` asks potentials for per-sample reduced
gradients through `energy_grad_sum_by_sample()` whenever possible. This avoids
materializing dense `(..., n_terms, n_params)` arrays when a reducer only needs
summed energy-gradient statistics.

### Force Derivative Channels

`compute.force` and FM/CDFM build force Jacobians from `force_grad(r)` or `basis_values(r)`.

For performance, new potentials should provide analytic derivatives through `_df_dparam_names` or override `basis_values(r)` directly. If no derivative is provided, `BasePotential.force_grad()` falls back to finite differences. That fallback is correct but much slower and should only be treated as a development fallback.

`force_grad(r)` may return a sparse matrix for one-dimensional `r` when the
projector can consume it efficiently. For batched `r`, return a dense array
with shape `r.shape + (n_params,)`; the force kernel owns the sparse
same-frame batch accumulation path.

---

## Built-In Potential Families

### Analytic Parameterizations

| Class | Notes |
|---|---|
| `HarmonicPotential` | Two parameters, `k` and `r0`; `k` is linear, `r0` is nonlinear |
| `GaussianPotential` | Amplitude is linear; center and width are nonlinear |
| `LennardJonesPotential` | Epsilon is linear, sigma is nonlinear |
| `LennardJones96Potential` | Same high-level structure as LJ 12-6 |
| `LennardJonesSoftPotential` | Soft-core nonlinear LJ family |

These classes expose explicit analytic first and second derivatives.

### Force-Basis Potential

| Class | Notes |
|---|---|
| `BSplinePotential` | Coefficients directly parameterize force; all channels are linear |

`BSplinePotential` is special because:

- force is the direct model output
- energy is obtained by integrating basis functions
- all parameters are linear optimization channels
- dense and sparse basis access are available for FM/CDFM
- pair splines use the gauge `U(r_max) = 0`
- bonded splines use the gauge `min(U) = 0`
- `gauge_free_energy_grad_sum*()` omits parameter-dependent bonded gauge
  shifts so REM can cache/use gauge-free energy gradients

### Multi-Component Gaussian Family

| Class | Notes |
|---|---|
| `MultiGaussianPotential` | Normalized Gaussian mixture; amplitudes are linear, centers and widths are nonlinear |
| `UnnormalizedMultiGaussianPotential` | Unnormalized LAMMPS-style mixture |
| `SRLRGaussianPotential` | Gaussian family for short-range / long-range models |

These models use vectorized dynamic derivative dispatch. They expose second derivatives within each component; cross-component second derivatives are zero.

---

## `IteratePotentials(forcefield)`

Flattened iteration helper:

```python
for key, pot in IteratePotentials(forcefield):
    ...
```

Works for both:

- legacy `Dict[key, BasePotential]`
- current `Forcefield` / `Dict[key, List[BasePotential]]`

Use it when you want to iterate in flattened parameter order. Use `forcefield.items()` when you want a grouped-by-key view.

---

## Registry Contract

`POTENTIAL_REGISTRY` maps external style names to classes:

| Registry key | Class |
|---|---|
| `harmonic` | `HarmonicPotential` |
| `gauss/cut`, `gauss/wall` | `GaussianPotential` |
| `lj/cut` | `LennardJonesPotential` |
| `lj96/cut` | `LennardJones96Potential` |
| `lj/cut/soft` | `LennardJonesSoftPotential` |
| `table` | `MultiGaussianPotential` |
| `double/gauss` | `UnnormalizedMultiGaussianPotential` |
| `srlr_gauss` | `SRLRGaussianPotential` |

Register a new potential family here when it should be serializable from external styles.

---

## Interaction With `Forcefield`

Potentials do not manage global masks or bounds. That is the responsibility of `Forcefield`.

Potentials should provide:

- local parameter vectors
- local derivative channels
- optional `param_bounds()` for local bounds
- linearity information via `_param_linear_mask`

`Forcefield` builds:

- the global parameter vector
- per-parameter and per-key masks
- global bounds arrays
- parameter slices and offsets

---

## Rules for New Potentials

When adding a new potential:

1. Inherit from `BasePotential`.
2. Implement `value(r)` and `force(r)` with vectorized NumPy.
3. Populate `_params`, `_param_names`, `_dparam_names`, and `_param_linear_mask`.
4. Provide `_df_dparam_names` or override `basis_values(r)` for efficient FM support.
5. Provide `_d2param_names` if REM/CDREM Hessian support matters.
6. Preserve arbitrary leading batch dimensions and append parameter channels only at the end.
7. Override `energy_grad_sum_by_sample()` when dense term-by-parameter materialization would be expensive.
8. Override `gauge_free_energy_grad_sum*()` if the potential has a parameter-dependent additive gauge.
9. Provide `param_bounds()` when natural bounds exist.
10. Register it in `POTENTIAL_REGISTRY` if it is part of the public style set.

Avoid:

- embedding atom indices or topology logic in potentials
- allocating Python objects per sample
- relying on finite-difference `force_grad()` in production paths
- reading workflow or optimizer state from inside a potential

---

## Allowed Workflow Access Patterns

Do:

- build or serialize potentials through `Forcefield` helpers
- treat potentials as low-level scientific model objects
- let compute, trainer, and solver code consume them through `Forcefield`

Do not:

- directly call potential methods from workflow code to reproduce compute math
- mix topology indices or MPI logic into potential usage
- mutate potential parameters outside the owning `Forcefield`, trainer, or solver
