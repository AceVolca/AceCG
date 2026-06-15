# AceCG: A Comprehensive Engine for Coarse-Graining

<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/102f6238-9eb4-4d3c-b555-50cac0b8df90" />

AceCG is a versatile and MPI-enabled coarse-graining force-field training engine.

Currently, we support:
- Force-Matching (FM)
- Denoising Score Matching (DSM)
- Relative Entropy Minimization (REM)
- Conditional-Derivative REM (CDREM)
- Conditional-Derivative Force Matching (CDFM)
- Virtual Particle workflows
- Boundary-prior force-field post-processing

The software is now built on LAMMPS. We will be adding OpenMM and Neural Network Potential Interfaces soon.

## Quick Start

The active package lives in `src/AceCG/`. Config templates live in `configs/`.
Public developer documentation lives in `documentation/`.

Start with the docs in `documentation/developer_guide`.

Use a compute node for MPI, LAMMPS, production runs, and long test suites.

`configs/templates` hold some templates for job running. Adjust accordingly to your specific needs and the current software architecture.

Install in editable mode:

```bash
python -m pip install -e .
```

Run a lightweight test check:

```bash
PYTHONPATH=src python -m pytest tests -q
```

Main CLI entry points:

```text
acg-fm
acg-dsm
acg-rem
acg-cdrem
acg-cdfm
acg-vpgrower
acg-boundary-prior
```

## Latest update

- 06/15/2026, Weizhi: Fixed some known issues in sampling workflow and trajectory handling:
  - Added `sampling.trajectory_format` for REM/CDREM/CDFM sampled trajectory post-processing.
  - Added automatic format inference for common LAMMPS `dump xtc`, `dump dcd`, and `dump h5md` scripts.
  - Added `sampling.archive_trajectory` for REM runs that need to retain sampled CG trajectories.
  - Made `sampling.init_config_pool` mutually exclusive with replay modes `latest` and `random`.
  - Rejected coordinate-only FM reference formats such as XTC and DCD when reference forces are required.
  - Updated OpenMPI and MPICH scheduler launch realization to prefer Slurm `srun --mpi=pmi2` whenever `SLURM_JOB_ID` is present.
- 06/02/2026, Weizhi: Source code cleanup, ready for pypi release
- 06/01/2026, Ace: bound & mask apply_spec() moved to forcefield class; WriteLmpFF supports multiple potentials of the same style for pairs
- 05/24/2026, Weizhi: Merged several new functionalities, including on-the-fly validation simulations (Weizhi), L0 gate utilities (Ace), mask & bound utilities (Ace & Weizhi), coordinate masks for REM diagnostics (Zhikun), etc.
- 05/06/2026, Ace: Added potential-local mask/bounds metadata (Ace, Weizhi), versioned Forcefield metadata caching (Weizhi), hard-concrete L0 gate utilities (Ace).
- 05/05/2026, Weizhi: Added DSM, noisy FM, mixed noisy FM, noisy REM, batch compute backends, and gauge-free gradient support (Zhikun).
- 04/24/2026, Weizhi: Synced all architectural updates to the current repo. See the developer guide for all details. AceCG is now with MPI CPU support, multitask scheduling, unified topology management, parallelized compute backends, config file parsing, command-line interface, etc.

## Developer Documentation

See `./documentation`. Last updated 06/15/2026 to reflect current changes.

## Known Issues

To be reported. No known issues for this 06/15/2026 release yet. Please feel free to test the code and raise issues.

## Developer Team

- [@Chengxi (Ace) Yang](https://github.com/AceVolca)
- [@Weizhi Xue](https://github.com/KJAdams2000)
- [@Zhikun Zhou](https://github.com/afakeoutstandingplyer)
- [@Curt Waltmann](https://github.com/waltmann1)
- [@Harper Smith](https://github.com/hesmithh)
- [@Thomas Qu](https://github.com/FreddyNietzky)
- [@Ivan Kuang](https://github.com/Miku-keai)
- [@Brian Faintich](https://github.com/brianfaintich)
