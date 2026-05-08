# AceCG: A Comprehensive Engine for Coarse-Graining

AceCG is a versatile and MPI-enabled coarse-graining force-field training engine for workflows such as
FM, REM, Reweighted REM, MSE, CDREM and VP growth.

## Quick Start

The active package lives in `src/AceCG/`. Config templates live in `configs/`.
Tracked experiment records and generated outputs live in `experiments/`.

Start with the docs in `documentation/developer_guide`.

Use a compute node for MPI, LAMMPS, production runs, and long test suites.

`configs/templates` hold some templates for job running. Adjust accordingly to your specific needs and the current software architecture.

## Latest update

- 05/06/2026, Ace: Added potential-local mask/bounds metadata (Ace, Weizhi), versioned Forcefield metadata caching (Weizhi), hard-concrete L0 gate utilities (Ace).
- 05/05/2026, Weizhi: Added DSM, noisy FM, mixed noisy FM, noisy REM, batch compute backends, and gauge-free gradient support (Zhikun).
- 04/24/2026, Weizhi: Synced all architectural updates to the current repo. See the developer guide for all details. AceCG is now with MPI CPU support, multitask scheduling, unified topology management, parallelized compute backends, config file parsing, command-line interface, etc.

## Developer Team

- [@Chengxi (Ace) Yang](https://github.com/AceVolca)
- [@Weizhi Xue](https://github.com/KJAdams2000)
- [@Zhikun Zhou](https://github.com/afakeoutstandingplyer)
- [@Curt Waltmann](https://github.com/waltmann1)
- [@Harper Smith](https://github.com/hesmithh)
- [@Thomas Qu](https://github.com/FreddyNietzky)
- [@Ivan Kuang](https://github.com/Miku-keai)
- [@Brian Faintich](https://github.com/brianfaintich)
