# AceCG: A Comprehensive Engine for Coarse-Graining

AceCG is an MPI-enabled coarse-graining force-field training engine. It
currently supports force matching (FM), denoising score matching (DSM),
relative entropy minimization (REM), conditioned dual-sampling REM (CDREM),
conditioned direct force matching (CDFM), virtual-particle growth, and
boundary-prior force-field post-processing.

The installable package lives in `src/AceCG/`. Reusable `.acg` templates live
in `configs/`, and public developer documentation lives in `documentation/`.

## Quick Start

Install in editable mode from the repository root:

```bash
python -m pip install -e .
```

Run a lightweight test check:

```bash
PYTHONPATH=src python -m pytest tests -q
```

Use a compute node for MPI, LAMMPS, production runs, and long test suites.

The main CLI entry points are:

```text
acg-fm
acg-dsm
acg-rem
acg-cdrem
acg-cdfm
acg-vpgrower
acg-boundary-prior
```

Start with `documentation/developer_guide/00_architecture.md`, then read the
workflow, scheduler, and MPI runtime chapters for production runs.

## Release Notes

### 2026-06-15

- Added sampling-trajectory format plumbing for REM, CDREM, and CDFM
  post-processing, including explicit `sampling.trajectory_format` and
  automatic inference for common LAMMPS `dump xtc`, `dump dcd`, and
  `dump h5md` scripts.
- Added `sampling.archive_trajectory` for REM runs that need to retain sampled
  CG trajectories after successful post-processing.
- Made `sampling.init_config_pool` mutually exclusive with replay modes
  `latest` and `random`, with validation in both config parsing and sampler
  construction.
- Rejected coordinate-only FM reference formats such as XTC and DCD when
  reference forces are required.
- Updated OpenMPI and MPICH scheduler launch realization to prefer Slurm
  `srun --mpi=pmi2` whenever `SLURM_JOB_ID` is present, including local-host
  placements inside a Slurm allocation.
