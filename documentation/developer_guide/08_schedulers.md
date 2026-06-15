# 08 AceCG `schedulers/` Developer Reference

*Updated: 2026-06-15. Merged from the older scheduler design document.*

> Audience: developers who need to understand, debug, or extend the AceCG scheduler subsystem.

The scheduler layer translates workflow-produced `TaskSpec` objects into concrete MPI process launches. Within one iteration it streams multiple xz / zbx tasks onto the CPU resources available on the current machine or SLURM allocation.

It owns:

- discovery of compute resources: nodes and CPUs per node, via `ResourcePool`
- greedy CPU allocation through `LeasePool` and `Placer`
- MPI launch-command assembly through the `MpiBackend` family
- task subprocess launch, polling, signal forwarding, and timeout handling through `TaskScheduler`
- sim + post execution inside task processes through `task_runner`

It does not own:

- workflow orchestration, which lives in `workflows/`
- trainer, optimizer, or reducer semantics
- accumulation of training statistics across iterations or tasks

---

## 1. Core Files

| File | Responsibility |
|---|---|
| `schedulers/resource_pool.py` | Node/CPU discovery, `HostInventory`, `LeasePool`, `Placer`, `ResourcePool` |
| `schedulers/mpi_backend.py` | Data types `HostSlice` / `Placement` / `LaunchSpec`, plus Intel/OpenMPI/MPICH implementations |
| `schedulers/task_scheduler.py` | Streaming scheduler engine: `TaskScheduler.run_iteration()`, `_stream()`, process management |
| `schedulers/task_runner.py` | Compute-node sim + post executor and subprocess entry point |
| `schedulers/profiler.py` | Optional MPI preflight benchmark; side path only |

---

## 2. Call-Chain Overview

```text
Workflow
  -> _build_resource_pool()              # workflows/base.py
     -> ResourcePool.discover()          # auto-discover nodes/CPUs + choose MPI backend
  -> TaskScheduler(pool)
     -> run_iteration(xz_tasks, zbx_tasks)
        -> Placer.place(cores)           # lease CPUs from LeasePool
        -> backend.realize(placement)    # build MPI command line
        -> subprocess.Popen(...)         # launch task_runner
           -> task_runner.py main()      # on compute node: run simulation + post
```

---

## 3. Three-Layer Design

The scheduler is split into resource discovery, resource placement, and command assembly. Each layer is replaceable on its own:

```text
Layer 1: resource discovery
  HostInventory(hostname, cpu_ids)
  ResourcePool.discover()

Layer 2: greedy resource placement
  LeasePool      per-CPU free pool
  Placer.place() best-fit + multi-host placement
  Placement      tuple of HostSlice

Layer 3: vendor-specific command assembly
  MpiBackend.realize(placement, payload)
  LaunchSpec(argv, env_add, env_strip_prefixes)
```

Inputs and outputs are plain data classes: `HostInventory`, `Placement`, and `LaunchSpec`. Layers do not reach across boundaries.

- Switching MPI implementation should only require Layer 3 changes.
- Switching resource discovery, for example to YARN or Kubernetes, should only require Layer 1 changes.
- The greedy algorithm in Layer 2 is independent and can be tested separately.

---

## 4. Layer 1: Resource Discovery

### Discovery Priority

`ResourcePool.discover()` chooses `list[HostInventory]` in this order:

1. `explicit_hosts` from `.acg`, a manually specified node+CPU list; highest priority user override.
2. `scontrol show job -d` in a SLURM environment; no SSH needed.
3. `SLURM_NODELIST` plus SSH probing; each node runs `os.sched_getaffinity(0)`.
4. `localhost` fallback outside SLURM.

| Method | Latency | Reliability | Requirements |
|---|---|---|---|
| `scontrol show job -d` | <1s | High; reads SLURM DB | `SLURM_JOB_ID` and `scontrol` on PATH |
| SSH + `sched_getaffinity` | 5-10s per node | Medium; SSH may timeout | Passwordless SSH and Python 3 |
| `explicit_hosts` | 0s | Highest; user-provided | User knows the allocation |

### Three Launch Environments

Each backend dispatches inside `realize()` based on `SLURM_JOB_ID` and
`_is_local(placement)`. OpenMPI and MPICH prefer the Slurm path whenever
`SLURM_JOB_ID` is present, even for local-host placements inside a Slurm
allocation. Intel MPI also uses Slurm realization inside allocations according
to its configured launch mode.

| Environment | Trigger | Typical use |
|---|---|---|
| local | all cores are on the local host and no Slurm allocation is active | login-node smoke tests |
| SLURM | `SLURM_JOB_ID` exists | production `sbatch`, multi-node `sinteractive`, and local-host placements inside an allocation |
| multi-host SSH | no `SLURM_JOB_ID` and placement spans nodes | small clusters or private machines with passwordless SSH |

### `scontrol` Discovery Optimization

Earlier versions used `SLURM_NODELIST + SSH` to probe each node, which could take 60-120 seconds for 12 nodes. Switching to one `scontrol show job -d` query reduced latency below one second and removed SSH overhead.

Parsed format:

```text
Nodes=midway3-0022 CPU_IDs=16-21 Mem=22860 GRES=
Nodes=midway3-0095 CPU_IDs=7,10-13,18,46 Mem=26670 GRES=
```

`CPU_IDs` is a comma-separated range list parsed by `_parse_cpu_range()`.

### Why `explicit_hosts` Exists

Two situations can break `scontrol` discovery:

1. The shell has lost SLURM environment variables. This can happen when entering a compute node through `ssh compute_node 'python -m AceCG ...'`; `SLURM_JOB_ID` and `SLURM_NODELIST` disappear.
2. CPU allocation is fragmented rather than whole-node. In `sinteractive`, a job may receive 5-8 CPUs per node. SSH fallback sees the whole-node affinity because the SSH shell is not inside the job cgroup.

Manually specifying `explicit_hosts` in `.acg` is the most robust fallback. It lives in the `[scheduler]` section as a raw list-of-dicts, and the parser stores it in `scheduler.extras["explicit_hosts"]`:

```ini
[scheduler]
explicit_hosts = [
  {"hostname": "midway3-0022", "cpu_ids": [16, 17, 18, 19, 20, 21]},
  {"hostname": "midway3-0095", "cpu_ids": [7, 10, 11, 12, 13, 18, 46]},
]
```

---

## 5. Layer 2: Greedy Resource Placement

### Data Structures

- `LeasePool`: per-host `{host: sorted list of free cpu_ids}` plus `_total[host]`, which Intel MPMD uses through `host_total()`.
- `CpuLease`: one acquired block, `(host, cpu_ids)`.
- `Placer`: consumes `LeasePool + MpiBackend` and returns `PlacementResult(placement, leases)` for one task.

### `Placer.place()` Algorithm

```python
place(min_cores, preferred_cores, max_cores, single_host_only)
```

Two paths:

Single-host first: starting at `preferred_cores` and decreasing toward `min_cores`, if any host has at least `n` free cores, acquire `n` immediately.

Multi-host fallback: if single-host placement fails, `single_host_only=False`, and `backend.supports_multi_host=True`, use greedy multi-node placement.

- `whole_node=False`, the default for SSH bootstrap / srun paths: sort hosts by free-core count descending and take up to `target - collected` cores from each.
- The old whole-node constraint has been removed. The current Intel MPMD path uses per-segment pinning and no longer requires reserving full nodes.

### `_allocate_contiguous()`

The allocator prefers contiguous free CPU blocks, such as CPU 16-21, to avoid crossing NUMA domains. If no contiguous block exists, it falls back to the first `n` free cores.

---

## 6. Layer 3: MPI Command Assembly

### Data Flow

```text
Placement
payload_cmd  -> backend.realize() -> LaunchSpec
run_dir
```

- `Placement` is a group of `HostSlice` objects: host plus cpu ids.
- `payload_cmd` is the application command, such as `["lmp"]` or `["python", "-m", "AceCG.compute.mpi_engine"]`.
- `LaunchSpec` is directly consumable by `subprocess.Popen`: `(argv, env_add, env_strip_prefixes)`.

### Backend Selection

```python
pick_backend(mpirun_path, intel_launch_mode="mpmd")
```

`mpirun --version` is inspected to detect the MPI family:

| Output contains | Backend |
|---|---|
| `"intel"` / `"impi"` | `IntelMpiBackend` |
| `"open mpi"` | `OpenMpiBackend` |
| `"mpich"` / `"hydra"` | `MpichBackend` |
| other | `LocalMpirunBackend`, single-host only |

### Vendor Backend SLURM Paths

| Backend | SLURM command |
|---|---|
| IntelMPI (mpmd) | `mpirun -bootstrap slurm -n X -host A -env I_MPI_PIN_PROCESSOR_LIST ... : -n Y -host B ...` |
| IntelMPI (srun) | `srun --mpi=pmi2 --overlap --exact --distribution=arbitrary --cpu-bind=map_cpu:c0,c1,...` |
| OpenMPI | `srun --mpi=pmi2 --overlap --exact --distribution=arbitrary --cpu-bind=map_cpu:... + SLURM_HOSTFILE` |
| MPICH | Same OpenMPI SLURM path |

Three key invariants:

1. Per-rank CPU pinning: every SLURM path uses `map_cpu:c0,c1,c2,...` to pin rank `i` exactly to `cpu_ids[i]`. It does not rely on SLURM block/cyclic distribution.
2. `SLURM_HOSTFILE` plus `--distribution=arbitrary`: OpenMPI and MPICH use this pair to support arbitrary rank counts per node. The hostfile has one line per rank, and `arbitrary` makes `srun` follow that order.
3. `env_strip_prefixes=("PMI_", "SLURM_")`: before `_launch()`, parent process PMI / SLURM step-scoped variables are stripped, then clean `SLURM_JOB_ID`, `SLURM_CONF`, and `SLURM_HOSTFILE` are injected. Without stripping, nested `mpirun` / `srun` may believe it is still inside the parent step.

### Intel MPI: MPMD vs srun

This is the most subtle scheduler area.

MPMD mode, the default `intel_launch_mode = mpmd`:

- uses `mpirun -bootstrap slurm` plus MPMD colon syntax
- writes one segment per slice: `-n N -host H -env I_MPI_PIN_PROCESSOR_LIST cpu_ids`, joined with `:`
- Hydra starts the internal `srun` step and does not require `libpmi2.so`
- segments must be sorted in canonical `SLURM_JOB_NODELIST` order; otherwise rank placement and CPU pinning can drift
- `_realize_slurm_mpmd()` handles this with `_sort_slices_by_slurm_nodelist(...)`

srun mode, `intel_launch_mode = srun`:

- uses direct `srun --mpi=pmi2`
- Intel MPI's PMI2 client must be able to `dlopen libpmi2.so`
- `I_MPI_PMI_LIBRARY` should point to `libpmi2.so`, typically after `module load slurm/current`
- recent live scheduler runs have validated this path
- it is lighter than Hydra bootstrap and provides exact per-rank CPU binding

`.acg` config:

```ini
[scheduler]
mpirun_path = /software/intel/oneapi_hpc_2023.1/mpi/2021.9.0/bin/mpirun
intel_launch_mode = srun    # or "mpmd", the default
```

### `LaunchSpec.env_add` vs `ResourcePool.extra_env`

| Field | Source | Scope | Example |
|---|---|---|---|
| `LaunchSpec.env_add` | Backend-generated | One `realize()` call | `I_MPI_PIN_PROCESSOR_LIST`, `SLURM_HOSTFILE` |
| `ResourcePool.extra_env` | `.acg` `scheduler.extras.extra_env` | All task subprocesses | `FI_PROVIDER=tcp`, `LD_LIBRARY_PATH` patch |

Merge order:

```text
os.environ -> strip -> env_add -> extra_env
```

Therefore `extra_env` can override backend choices. It only enters subprocesses; the scheduler parent process cannot see it. Any logic for selecting a backend or launch mode must use config fields, not `extra_env`.

---

## 7. Streaming Task Scheduling

### `TaskScheduler.run_iteration(xz_tasks, zbx_tasks)`

```text
run_iteration(xz, zbx):
  1. Rebuild a fresh LeasePool from ResourcePool for this iteration.
  2. Merge xz_tasks + zbx_tasks into pending, with xz first.
  3. Fail fast if a single_host_only task's min_cores exceeds max free cores on any host.
  4. _fill(): for each pending task, call placer.place(); launch immediately if placed.
  5. Poll loop, proc.poll() every 0.5 s:
       completed -> release leases -> record TaskResult -> call _fill()
       xz failed -> _kill_all(active) and return early
       zbx failed -> discard that replica and continue
  6. After return, check min_success_zbx to decide whether the iteration failed.
```

### `_stream()` Flow

```text
pending = [xz_task, zbx_0, zbx_1, ..., zbx_K]
active = {}  # pid -> (proc, task, placement_result, t0)

_fill():
  for task in pending:
    pr = placer.place(...)
    if pr is not None:
      _launch(task, pr)
      active[pid] = (proc, task, pr, t0)
      pending.remove(task)

while active:
  for pid, (proc, task, pr, t0) in active:
    ret = proc.poll()
    if ret is None:
      if time.monotonic() - t0 > timeout:
        _timeout_kill(pgid)
      continue
    release leases(pr.leases)
    record TaskResult(task, ret, ...)
    del active[pid]
    if task.is_xz and ret != 0:
      _kill_all(active)
      return FAILED
    _fill()
  if no progress:
    sleep(0.5)
```

### Subprocess Launch Flow

```python
def _launch(task, pr):
    spec_dict = task.to_spec_dict(cpu_cores)

    sim_launch = backend.realize(placement, sim_cmd, run_dir)
    post_launch = backend.realize(post_placement, py_cmd, run_dir)

    env = os.environ - strip_prefixes + env_add + pool.extra_env

    subprocess.Popen(
        ["python", "-m", "AceCG.schedulers.task_runner", spec_json],
        start_new_session=True,
    )
```

### Process Management

Each task subprocess uses `start_new_session=True`, creating an independent process group. The scheduler tracks `pgid`:

- timeout: `os.killpg(pgid, SIGTERM)`, then `SIGKILL` after 5 seconds
- scheduler SIGTERM: `_sigterm_handler` kills all active process groups before forwarding to the previous handler
- `atexit`: kills process groups as a final guard against orphaned `mpirun`

---

## 8. `task_runner.py`

Runs as a subprocess on the compute node. It reads `task_spec.json` and executes:

1. Simulation phase: `subprocess.Popen(sim_launch.argv, env=...)`, usually LAMMPS.
2. Optional post-processing phase:
   - `mode="python"` for local Python functions such as bonded projectors or gradient calculations
   - `mode="mpi"` for MPI-parallel post-processing through `post_launch.argv`

It writes `timing.json`:

```json
{"sim_wall": 340.1, "post_wall": 17.4, "total_wall": 357.5}
```

---

## 9. Common Pitfalls and Debugging Guide

### Pitfall 1: Not Loading the LAMMPS Module

Symptoms: `srun --mpi=pmi2` reports `libpmi2.so: cannot open shared object file`, or Intel MPI cannot find `libfabric.so.1` or `libmpi.so.12`.

Root cause: `module load lammps/<ver>` prepends both PATH (`mpirun`, `lmp`) and `LD_LIBRARY_PATH` (`libmpi.so.*`, `libfabric.so.1`). Without it, `shutil.which("mpirun")` may find a conda MPICH shim while `lmp` was built against Intel MPI.

Fix:

```bash
module load lammps/29Aug2024
source /software/.../conda.sh
conda activate acg
hash -r
```

Avoid `ssh compute_node "source env.sh && python ..."` fire-and-forget launches. The SSH shell loses `SLURM_JOB_ID` / PMI environment.

### Pitfall 2: `srun --overlap` Without `--exact`

Symptom: concurrent `srun` steps wait for each other and throughput drops to one serial task.

Root cause: without `--exact`, each `srun` step claims all allocated task slots.

Fix: always use `srun --overlap --exact`.

### Pitfall 3: Orphaned Old Processes Corrupt Trajectories

Symptoms: `.lammpstrj` files become corrupt, MDAnalysis reports `IndexError`, and new tasks become unusually slow because old LAMMPS processes compete for the same cores.

Root cause: killing the parent process may leave Hydra-launched LAMMPS children alive. A rerun reusing the same `output_dir` can have old and new LAMMPS processes writing the same trajectory file.

Fix:

```bash
pkill -u $USER -f "task_runner.*cdrem_t5"
pkill -u $USER -f "lmp.*in\.(xz|zbx)\.lmp"
# or scancel $SLURM_JOB_ID to kill all srun steps in the job

rm -rf experiments/<level>/<run_id>/iter_0000/
```

Developer recommendation: before launching LAMMPS, `task_runner` should detect whether `.lammpstrj` or `sim.log` already exists in `run_dir` and either delete it or fail explicitly.

### Pitfall 4: `shutil.which` Is Expensive on HPC Filesystems

Symptom: each task realization takes 2-5 seconds and iteration scheduling overhead exceeds 10 seconds.

Fix: cache `_srun_path` and `_slurm_conf` in backend `__init__()` and use cached fields in `realize()`.

### Pitfall 5: Intel MPMD Segments Not Sorted by SLURM Node Order

Symptom: `mpirun -bootstrap slurm` rank numbers do not match expected CPU pinning.

Fix: `_realize_slurm_mpmd` must use:

```python
sorted_slices = _sort_slices_by_slurm_nodelist(placement.slices)
```

### Pitfall 6: `sinteractive --ntasks=N` Causes CPU Capping

Symptom: a `--ntasks=64` job has `CPUs/Task=1`, and step-level `--cpus-per-task=M` is silently capped to 1 by SLURM.

Fix: request the job as:

```bash
sinteractive --nodes=N --ntasks-per-node=M
```

### Pitfall 7: `dict_values` Is Not Pickleable in Python 3.13

Symptom: MPI broadcast fails in `topology_array.py`.

Fix: wrap with `list(d.values())`.

### General Debugging Steps

```bash
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NODELIST=$SLURM_NODELIST"
echo "which mpirun: $(which mpirun)"
mpirun --version

scontrol show job -d $SLURM_JOB_ID | grep -E "Nodes=|CPU_IDs="

python -c "
from AceCG.schedulers.resource_pool import ResourcePool
pool = ResourcePool.discover()
print(pool.hosts)
"

mpirun -n 4 bash -c 'cat /proc/$$/status | grep Cpus_allowed_list'

python -c "
from AceCG.schedulers.mpi_backend import pick_backend
b = pick_backend('mpirun')
print(type(b).__name__, b.supports_multi_host)
"
```

---

## 10. Adding a New MPI Backend

1. Add a class in `mpi_backend.py`:

   ```python
   class FooMpiBackend(MpiBackend):
       name = "foo"
       supports_multi_host = True

       def __init__(self, mpirun_path: str) -> None:
           self.mpirun_path = mpirun_path
           self._srun_path = shutil.which("srun")
           self._slurm_conf = _find_slurm_conf()

       def realize(self, placement, payload_cmd, run_dir) -> LaunchSpec:
           if os.environ.get("SLURM_JOB_ID"):
               return self._realize_slurm(placement, payload_cmd, run_dir)
           if _is_local(placement):
               return self._realize_local(placement, payload_cmd)
           return self._realize_ssh(placement, payload_cmd, run_dir)
   ```

2. Each `_realize_*` method must produce correct argv, include per-rank CPU pinning, set vendor-specific variables in `env_add`, and set `env_strip_prefixes` appropriately.

3. Add detection strings to `detect_mpi_family()` so `mpirun --version` can return `"foo"`.

4. Add dispatch in `pick_backend()`:

   ```python
   if family == "foo":
       return FooMpiBackend(mpirun_path)
   ```

5. Add unit tests, ideally with a stub backend whose `realize()` returns `echo <...>`, so scheduler flow can be tested without real MPI.

6. Add a production smoke test, such as `scripts/smoke_foo.sh`, that runs a few CDREM epochs and compares `grad_norm` curves against an existing backend.

---

## 11. Config Quick Reference

| `.acg` field | Type | Description |
|---|---|---|
| `scheduler.launcher` | deprecated | Ignored; backend is now selected by `scheduler.mpirun_path`, `scheduler.mpi_family`, or PATH |
| `scheduler.mpirun_path` | `str` | Absolute path to `mpirun`; empty means auto-detect |
| `scheduler.mpi_family` | `str` | Optional explicit backend override, such as `intel`, `openmpi`, or `mpich` |
| `scheduler.python_exe` | `str` | Python executable on compute nodes; default `"python"` |
| `scheduler.task_timeout` | `int` | Task timeout in seconds; must be set |
| `scheduler.min_success_zbx` | `int` | Minimum number of successful ZBX tasks; defaults to all |
| `scheduler.explicit_hosts` | `list[dict]` | Written in `[scheduler]`; stored at runtime in `scheduler.extras.explicit_hosts` |
| `scheduler.extra_env` | `dict` | Written in `[scheduler]`; stored in `scheduler.extras.extra_env` and injected into all task subprocesses |
| `scheduler.intel_launch_mode` | `str` | Written in `[scheduler]`; stored in `scheduler.extras.intel_launch_mode`; Intel MPI SLURM mode: `"mpmd"` or `"srun"` |
