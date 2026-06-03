# AceCG: A Comprehensive Engine for Coarse-Graining

<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/102f6238-9eb4-4d3c-b555-50cac0b8df90" />

AceCG is a versatile and MPI-enabled coarse-graining force-field training engine.

Currently, we support:
- Force-Matching (FM)
- Denoising Score Matching (DSM)
- Relative Entropy Minimization (REM)
- Virtual Particle workflows

The software is now built on LAMMPS. We will be adding OpenMM and Neural Network Potential Interfaces soon.

## Quick Start

The active package lives in `src/AceCG/`. Config templates live in `configs/`.
Tracked experiment records and generated outputs live in `experiments/`.

Start with the docs in `documentation/developer_guide`.

Use a compute node for MPI, LAMMPS, production runs, and long test suites.

`configs/templates` hold some templates for job running. Adjust accordingly to your specific needs and the current software architecture.

## Latest update

- 06/02/2026, Weizhi: Source code cleanup, ready for pypi release
- 06/01/2026, Ace: bound & mask apply_spec() moved to forcefield class; WriteLmpFF supports multiple potentials of the same style for pairs
- 05/24/2026, Weizhi: Merged several new functionalities, including on-the-fly validation simulations (Weizhi), L0 gate utilities (Ace), mask & bound utilities (Ace & Weizhi), coordinate masks for REM diagnostics (Zhikun), etc.
- 05/06/2026, Ace: Added potential-local mask/bounds metadata (Ace, Weizhi), versioned Forcefield metadata caching (Weizhi), hard-concrete L0 gate utilities (Ace).
- 05/05/2026, Weizhi: Added DSM, noisy FM, mixed noisy FM, noisy REM, batch compute backends, and gauge-free gradient support (Zhikun).
- 04/24/2026, Weizhi: Synced all architectural updates to the current repo. See the developer guide for all details. AceCG is now with MPI CPU support, multitask scheduling, unified topology management, parallelized compute backends, config file parsing, command-line interface, etc.

## Known Issues

The 05/24 merge is too heavy. Code is not simple enough. Need to fix.

## Developer Team

- [@Chengxi (Ace) Yang](https://github.com/AceVolca)
- [@Weizhi Xue](https://github.com/KJAdams2000)
- [@Zhikun Zhou](https://github.com/afakeoutstandingplyer)
- [@Curt Waltmann](https://github.com/waltmann1)
- [@Harper Smith](https://github.com/hesmithh)
- [@Thomas Qu](https://github.com/FreddyNietzky)
- [@Ivan Kuang](https://github.com/Miku-keai)
- [@Brian Faintich](https://github.com/brianfaintich)
