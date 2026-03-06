# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AceCG is a Python package for **bottom-up coarse-graining** of molecular systems. It supports Relative Entropy Minimization (REM), MSE-based PMF matching, and Force-Matching (FM) — both iterative (trainer-based) and linear (solver-based). FM implementation was ported from OpenMSCG (C/C++) into AceCG's native Python architecture.

## Environment & Cluster Rules

- **Login node only for lightweight tasks.** Heavy tests/benchmarks must run on compute nodes via `ssh midway3-0417` or SLURM.
- Conda setup: `module load python; source activate <conda_env>` (valid envs: `mscgtest`, `confdiff`)
- Python 3.8, MDAnalysis 2.4.3, NumPy 1.24.4, SciPy 1.10.1
- Import mode: PYTHONPATH-root (`import AceCG` or `from potentials import ...`)

## Commands

```bash
# Install (development)
pip install -e .

# Run tests
pytest
pytest -m "not slurm"               # skip SLURM-dependent tests

# FM full pipeline (on compute node)
bash tests_solver/slurm/run_full_pipeline.sh
ACECG_ENGINE=iterative bash tests_solver/slurm/run_full_pipeline.sh  # iterative mode
```

## Architecture

### Module Hierarchy

All active code lives at the repository root (not under `src/AceCG/`, which is stale/deprecated):

- **`potentials/`** — Analytic potential models. Base: `BasePotential` (abstract). Implementations: Gaussian, BSpline, LennardJones, MultiGaussian, etc. Each potential exposes `value(r)`, `force(r)`, parameter names, and FM derivative channels (`_dparam_dr_names`, `_d2param_dr_names`). Registry pattern via `POTENTIAL_REGISTRY` in `__init__.py`.
- **`optimizers/`** — Gradient-based optimizers. Base: `BaseOptimizer`. Implementations: Adam, AdamW, RMSprop, NewtonRaphson. Interface: `step(grad, hessian=None) -> update_vector`. Support parameter masks for selective training.
- **`trainers/`** — Training loops. Base: `BaseTrainer`. Key implementations:
  - `REMTrainerAnalytic` — Relative entropy minimization (ensemble reweighting)
  - `MSETrainerAnalytic` — PMF matching
  - `FMTrainerAnalytic` — Iterative force-matching (frame-batch via `trainer.step()`)
  - `MultiTrainerAnalytic` — Combines multiple trainers
- **`solvers/`** — Accumulator-style matrix solvers. Base: `BaseSolver`. Key: `FMMatrixSolver` (OpenMSCG-style normal equations: `XtX * coeff = XtY`).
- **`fitters/`** — Table-to-potential converters. Base: `BaseTableFitter`. Implementations: BSpline, MultiGaussian. Registry pattern for lookup.
- **`utils/`** — Shared utilities:
  - `neighbor.py` — Neighbor lists (`ComputeNeighborList`), bonded topology (`GetBondedInfo`)
  - `fm_projectors.py` — FM design matrix builders (`PairProjector`, `BondProjector`, `AngleProjector`, `DihedralProjector`, `NB3BProjector`, `build_design_matrix`)
  - `topology_mscg.py` — OpenMSCG `top.in` parser (`parse_mscg_top`, `attach_topology_from_mscg_top`)
  - `compute.py` — Analytical derivatives (`dUdL`, `Hessian`, `dUdLByFrame`)
  - `ffio.py` — LAMMPS force field I/O (`ReadLmpFF`, `WriteLmpFF`, `WriteLmpTable`)

### Key Design Patterns

- **Deep-copy isolation:** Trainers/solvers deep-copy potentials and optimizers to prevent shared state.
- **Parameter flattening:** `FFParamArray()` concatenates params from a dict of potentials into a 1D array.
- **Dictionary-based I/O:** `step()` methods take/return dicts for interface resilience.
- **Topology precedence:** Existing MDAnalysis bonds are authoritative; `top.in` attachment only fills missing attrs.
- **Pair exclusions:** Bonded (1-2, 1-3, 1-4) or resid-based, routed through `utils/neighbor.py`.

## Critical Rules

1. **Source of truth is root-level modules** — never add logic under `src/AceCG/`.
2. **Iterator integrity** — FM iterator tests must exercise `FMTrainerAnalytic.step()` end-to-end. Never bypass the iterator trainer by routing through the linear solver in iterator test paths.
3. **Reuse first** — before adding code, check if equivalent functionality exists in AceCG, MDAnalysis, NumPy, or SciPy.
4. **Refactor safety** — changes affecting floating-point results require before/after comparison with explicit thresholds.
5. **No CLI entrypoints** in this phase; API-first usage only.
6. **Workflow/SLURM scripts** stay under `tests_solver/tools` and `tests_solver/slurm`, not in the core API.

## Reference Paths

- Original AceCG (read-only): `/project2/gavoth/zhikunzhou/AceCG`
- OpenMSCG (reference): `/project2/gavoth/weizhixue/programs/OpenMSCG`
- OpenMSCG run example: `/project2/gavoth/zhikunzhou/projects/vcg_AceCG_ver/DOPC_phase_diagram/step1_FM_excl100`
- Canonical FM runs: `tests_solver/_runs/trueiter_mscgtest_fastnodes_20260305_112051` and `trueiter_confdiff_fastnodes_20260305_115305`
