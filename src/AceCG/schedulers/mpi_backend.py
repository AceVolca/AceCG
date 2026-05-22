"""MPI backend abstraction for the AceCG scheduler.

Separates vendor-specific MPI launcher assembly from resource discovery
and task scheduling.  Each backend implements ``realize()`` which turns
a ``Placement`` (host + core allocation) into a ``LaunchSpec`` (concrete
argv + environment).

Each backend supports three launch modes, selected automatically:

* **local** — single-host, shared-memory fabric (``mpirun``).
* **slurm** — multi-host inside a SLURM allocation (``srun --mpi=pmi2``).
* **ssh**   — multi-host without SLURM (``mpirun`` over SSH).

Backends
--------
IntelMpiBackend         — Intel MPI (Hydra MPMD default, direct srun optional)
OpenMpiBackend          — OpenMPI (srun for SLURM, rankfile for SSH)
MpichBackend            — MPICH (srun for SLURM, Hydra SSH for SSH)
LocalMpirunBackend      — any MPI, single-host only (generic fallback)
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


# ── Data objects ───────────────────────────────────────────────

@dataclass(frozen=True)
class HostSlice:
    """A block of CPU cores on one host, allocated to a task."""

    host: str
    cpu_ids: tuple[int, ...]
    # total CPUs on this host (0 = unknown)
    host_n_cpus: int = 0

    @property
    def n_cpus(self) -> int:
        """Return the number of CPU cores in this host slice."""
        return len(self.cpu_ids)


@dataclass(frozen=True)
class Placement:
    """Where a task will run: one or more host slices.

    ``n_ranks`` equals the total CPU count across all slices (one rank
    per core).
    """

    slices: tuple[HostSlice, ...]
    n_ranks: int

    @property
    def single_host(self) -> bool:
        """Return ``True`` when the placement uses exactly one host."""
        return len(self.slices) == 1

    @classmethod
    def from_host_cores(
        cls,
        host: str,
        cpu_ids: tuple[int, ...],
        n_ranks: int | None = None,
    ) -> Placement:
        """Convenience: build a single-host placement."""
        return cls(
            slices=(HostSlice(host=host, cpu_ids=cpu_ids),),
            n_ranks=n_ranks if n_ranks is not None else len(cpu_ids),
        )


@dataclass(frozen=True)
class LaunchSpec:
    """Everything the executor needs to ``Popen`` a task.

    ``env_strip_prefixes`` lists prefixes whose matching env vars should
    be removed before launching (e.g. ``("PMI_",)``).  ``env_add`` holds
    key-value pairs to inject.
    """

    argv: tuple[str, ...]
    env_add: dict[str, str] = field(default_factory=dict)
    env_strip_prefixes: tuple[str, ...] = ("PMI_",)
    cwd: str | None = None


# ── Backend base class ───────────────────────────────────────────────

class MpiBackend(ABC):
    """Base class every MPI backend must inherit from."""

    name: str = "unset"
    supports_multi_host: bool = False

    @abstractmethod
    def realize(
        self,
        placement: Placement,
        payload_cmd: list[str],
        run_dir: Path,
    ) -> LaunchSpec:
        """Turn *placement* + *payload_cmd* into a launchable spec.

        Parameters
        ----------
        placement
            Where the task will run.
        payload_cmd
            The application command (e.g. ``["lmp"]`` for LAMMPS).
        run_dir
            Task working directory.

        Returns
        -------
        LaunchSpec
            Ready-to-execute specification.

        Raises
        ------
        RuntimeError
            If the placement is unrealizable by this backend.
        """
        ...


def _is_local(placement: Placement) -> bool:
    """True when all slices sit on the local host."""
    if not placement.single_host:
        return False
    local = socket.gethostname().split(".")[0]
    return placement.slices[0].host in (local, "localhost", "127.0.0.1")


def _cpu_mask_hex(cpu_ids: tuple[int, ...]) -> str:
    """Build a hexadecimal CPU mask from core IDs for ``--cpu-bind``."""
    mask = 0
    for c in cpu_ids:
        mask |= 1 << c
    return f"0x{mask:x}"


# ── Helpers ───────────────────────────────────────────────

def _find_libpmi2() -> str | None:
    """Locate ``libpmi2.so`` via ``LD_LIBRARY_PATH``.

    Returns the absolute path if found, ``None`` otherwise.
    """
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if not d:
            continue
        candidate = os.path.join(d, "libpmi2.so")
        if os.path.isfile(candidate):
            return candidate
    return None


def _sort_slices_by_slurm_nodelist(
    slices: tuple[HostSlice, ...],
) -> list[HostSlice]:
    """Order slices by their position in ``SLURM_JOB_NODELIST``.

    Hydra's SLURM bootstrap delegates rank dispatch to ``srun``, and
    ``srun``'s block distribution walks the node list in the order given
    by ``SLURM_JOB_NODELIST`` — not hostname-sorted.  MPMD segments must
    follow the same order or per-segment env (e.g. ``I_MPI_PIN_...``)
    will land on the wrong node.

    Falls back to hostname sort if ``SLURM_JOB_NODELIST`` is unset or
    cannot be parsed.
    """
    nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
    if not nodelist:
        return sorted(slices, key=lambda s: s.host)
    try:
        from .resource_pool import _parse_slurm_nodelist
        ordered = _parse_slurm_nodelist(nodelist)
    except Exception:
        return sorted(slices, key=lambda s: s.host)
    index = {h.split(".")[0]: i for i, h in enumerate(ordered)}
    sentinel = len(index)

    def key(slice_: HostSlice) -> tuple[int, str]:
        host_short = slice_.host.split(".")[0]
        return (index.get(host_short, sentinel), host_short)

    return sorted(slices, key=key)


def _find_slurm_conf() -> str:
    """Locate ``slurm.conf`` for srun.

    Checks ``SLURM_CONF`` env var first, then probes relative to ``srun``
    (i.e. ``<srun_dir>/../etc/slurm.conf``).  Returns empty string if
    not found (srun may still work if the compiled-in default is valid).
    """
    val = os.environ.get("SLURM_CONF", "")
    if val:
        return val
    srun = shutil.which("srun")
    if srun:
        candidate = os.path.join(os.path.dirname(srun), "..", "etc", "slurm.conf")
        candidate = os.path.normpath(candidate)
        if os.path.isfile(candidate):
            return candidate
    return ""


# ── Intel MPI ───────────────────────────────────────────────

class IntelMpiBackend(MpiBackend):
    """Intel MPI backend — supports local, SLURM, and SSH launches.

    **Local** (single-host): ``mpirun`` with fork bootstrap and
    shared-memory fabric.  Per-rank CPU pinning via
    ``I_MPI_PIN_PROCESSOR_LIST``.

    **SLURM** (multi-host): uses Hydra SSH bootstrap
    (``mpirun -bootstrap ssh …``) with MPMD colon syntax and
    per-host CPU pinning.  This bypasses SLURM's srun entirely,
    avoiding both the PMI2 deadlock (srun --mpi=pmi2) and the
    srun step oversubscription (mpirun -bootstrap slurm).
    Requires passwordless SSH between compute nodes (standard in
    SLURM interactive allocations).

    NOTE: ``srun --mpi=pmi2`` deadlocks during MPI_Init on clusters
    where SLURM's PMI2 server is incompatible with Intel MPI's
    PMI2 client (Midway3/caslake).  ``mpirun -bootstrap slurm``
    over-allocates whole nodes and prevents concurrent tasks on
    shared nodes.  Both are retained but disabled.

    **SSH** (multi-host, no SLURM): same as SLURM path—Hydra SSH
    bootstrap with MPMD colon syntax.
    """

    name = "intel"
    supports_multi_host = True

    def __init__(
        self,
        mpirun_path: str,
        *,
        launch_mode: str = "mpmd",
    ) -> None:
        self.mpirun_path = mpirun_path
        # "mpmd" or "srun"
        self.launch_mode = launch_mode
        self._libpmi2_path: str | None = _find_libpmi2()
        self._srun_path: str | None = shutil.which("srun")
        self._slurm_conf: str = _find_slurm_conf()

    @property
    def _srun_bootstrap(self) -> bool:
        """SSH bootstrap does not need whole-node placement."""
        return False

    def realize(
        self,
        placement: Placement,
        payload_cmd: list[str],
        run_dir: Path,
    ) -> LaunchSpec:
        """Build the Intel MPI launch command for a placement."""
        if _is_local(placement):
            return self._realize_local(placement, payload_cmd)
        if os.environ.get("SLURM_JOB_ID"):
            return self._realize_slurm(placement, payload_cmd, run_dir)
        return self._realize_ssh(placement, payload_cmd, run_dir)

    # ── Local (single-host, shared-memory) ───────────────────────────────────────────────

    def _realize_local(
        self,
        placement: Placement,
        payload_cmd: list[str],
    ) -> LaunchSpec:
        cpu_list = ",".join(str(c) for c in placement.slices[0].cpu_ids)
        argv = (
            self.mpirun_path,
            "-np", str(placement.n_ranks),
            *payload_cmd,
        )
        return LaunchSpec(
            argv=argv,
            env_add={
                "I_MPI_FABRICS": "shm",
                "I_MPI_HYDRA_BOOTSTRAP": "fork",
                "I_MPI_PIN_PROCESSOR_LIST": cpu_list,
            },
        )

    # ── SLURM (multi-host) ───────────────────────────────────────────────

    def _realize_slurm(
        self,
        placement: Placement,
        payload_cmd: list[str],
        run_dir: Path,
    ) -> LaunchSpec:
        # Two working approaches for Intel MPI on SLURM:
        #   1. mpirun -bootstrap slurm (MPMD) — default, well-tested
        #   2. srun --mpi=pmi2 — simpler, selectable via config
        #
        # Controlled by self.launch_mode ("mpmd" or "srun").
        if self.launch_mode == "mpmd":
            return self._realize_slurm_mpmd(placement, payload_cmd)
        return self._realize_slurm_srun(placement, payload_cmd, run_dir)
    
    def _realize_slurm_srun(
        self,
        placement: Placement,
        payload_cmd: list[str],
        run_dir: Path,
    ) -> LaunchSpec:
        if self._srun_path is None:
            raise RuntimeError("srun not found on PATH")
        hostfile = run_dir / "srun_hostfile"
        lines: list[str] = []
        for s in placement.slices:
            lines.extend([s.host] * s.n_cpus)
        hostfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
        global_cpu_list: list[str] = []
        for s in placement.slices:
            global_cpu_list.extend(str(c) for c in s.cpu_ids)
        cpu_bind_str = ",".join(global_cpu_list)
        argv = (
            self._srun_path,
            "--overlap", "--exact",
            "--mpi=pmi2",
            "--distribution=arbitrary",
            f"--cpu-bind=map_cpu:{cpu_bind_str}",
            "-n", str(placement.n_ranks),
            *payload_cmd,
        )
        env_add: dict[str, str] = {
            "SLURM_HOSTFILE": str(hostfile),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", ""),
            "SLURM_CONF": self._slurm_conf,
        }
        if self._libpmi2_path:
            env_add["I_MPI_PMI_LIBRARY"] = self._libpmi2_path
        return LaunchSpec(
            argv=argv,
            env_add=env_add,
            env_strip_prefixes=("PMI_", "SLURM_"),
        )

    def _realize_slurm_mpmd(
        self,
        placement: Placement,
        payload_cmd: list[str],
    ) -> LaunchSpec:
        # MPMD colon syntax: each host-slice becomes a separate segment
        # with its own -n and -host.  Hydra's SLURM bootstrap ignores
        # -hostfile, so we must use this form for heterogeneous layouts.
        #   mpirun -bootstrap slurm
        #     -n 9 -host A -env I_MPI_PIN_PROCESSOR_LIST <cpus> <cmd>
        #     : -n 8 -host B -env I_MPI_PIN_PROCESSOR_LIST <cpus> <cmd>
        #
        # CRITICAL: segments MUST follow SLURM's canonical node order.
        # Hydra's SLURM bootstrap creates a single srun step for all
        # hosts, and SLURM does block distribution in the order given by
        # SLURM_JOB_NODELIST — NOT hostname-sorted.  If segments are out
        # of SLURM's order, rank counts get misassigned to the wrong
        # hosts, silently corrupting per-rank CPU pinning.  We read
        # SLURM_JOB_NODELIST here and sort slices by their position in
        # that list; unknown hosts (shouldn't happen) fall back to the
        # end in hostname order.
        #
        # Per-segment I_MPI_PIN_PROCESSOR_LIST pins each rank to the
        # exact allocated CPUs, enabling concurrent sub-node tasks
        # without overlap (no whole-node reservation needed).
        sorted_slices = _sort_slices_by_slurm_nodelist(placement.slices)
        argv: list[str] = [self.mpirun_path, "-bootstrap", "slurm"]
        for i, s in enumerate(sorted_slices):
            if i > 0:
                argv.append(":")
            cpu_list = ",".join(str(c) for c in s.cpu_ids)
            argv.extend([
                "-n", str(s.n_cpus),
                "-host", s.host,
                "-env", "I_MPI_PIN", "1",
                "-env", "I_MPI_PIN_PROCESSOR_LIST", cpu_list,
            ])
            argv.extend(payload_cmd)
        return LaunchSpec(
            argv=argv,
            env_add={},
            env_strip_prefixes=("PMI_",),
        )

    # ── SSH (multi-host, no SLURM) ───────────────────────────────────────────────

    def _realize_ssh(
        self,
        placement: Placement,
        payload_cmd: list[str],
        run_dir: Path,
    ) -> LaunchSpec:
        # Hydra SSH bootstrap with MPMD colon syntax.
        # Each segment gets per-host CPU pinning via -env so that
        # rank *i* on each host is pinned to cpu_ids[i] of that host.
        argv: list[str] = [self.mpirun_path, "-bootstrap", "ssh"]
        for i, s in enumerate(placement.slices):
            if i > 0:
                argv.append(":")
            cpu_list = ",".join(str(c) for c in s.cpu_ids)
            argv.extend([
                "-n", str(s.n_cpus),
                "-host", s.host,
                "-env", "I_MPI_PIN", "1",
                "-env", "I_MPI_PIN_PROCESSOR_LIST", cpu_list,
            ])
            argv.extend(payload_cmd)
        return LaunchSpec(argv=argv, env_add={})


# Backward-compat aliases.
IntelSlurmBackend = IntelMpiBackend
IntelHydraSlurmBackend = IntelMpiBackend


# ── OpenMPI ───────────────────────────────────────────────

class OpenMpiBackend(MpiBackend):
    """OpenMPI backend — supports local, SLURM, and SSH launches.

    **Local** (single-host): ``mpirun --mca btl self,vader`` for
    shared-memory, per-rank CPU pinning via ``--bind-to core
    --cpu-set``.

    **SLURM** (multi-host): ``srun --mpi=pmi2`` with
    ``SLURM_HOSTFILE`` + ``--distribution=arbitrary`` for heterogeneous
    per-node rank counts and ``--cpu-bind=map_cpu`` for per-rank CPU
    pinning.  OpenMPI's PLM SLURM plugin is broken on clusters where
    SLURM env is incomplete and inter-node SSH is disabled — verified
    on OpenMPI 4.1.2 and 5.0.2.

    **SSH** (multi-host, no SLURM): ``mpirun --rankfile`` for explicit
    per-rank host + core assignment.  Requires passwordless SSH between
    compute nodes (OpenMPI's default process launcher).
    """

    name = "openmpi"
    supports_multi_host = True

    def __init__(self, mpirun_path: str) -> None:
        self.mpirun_path = mpirun_path
        self._srun_path: str | None = shutil.which("srun")
        self._slurm_conf: str = _find_slurm_conf()

    def realize(
        self,
        placement: Placement,
        payload_cmd: list[str],
        run_dir: Path,
    ) -> LaunchSpec:
        """Build the OpenMPI launch command for a placement."""
        if _is_local(placement):
            return self._realize_local(placement, payload_cmd)
        if os.environ.get("SLURM_JOB_ID"):
            return self._realize_slurm(placement, payload_cmd, run_dir)
        return self._realize_ssh(placement, payload_cmd, run_dir)

    # ── Local (single-host, shared-memory) ───────────────────────────────────────────────

    def _realize_local(
        self,
        placement: Placement,
        payload_cmd: list[str],
    ) -> LaunchSpec:
        # Pin each rank to its allocated core.
        # --bind-to core + --cpu-set restricts to exactly cpu_ids.
        cpu_list = ",".join(str(c) for c in placement.slices[0].cpu_ids)
        argv = (
            self.mpirun_path,
            "--mca", "btl", "self,vader",
            "--bind-to", "core",
            "--cpu-set", cpu_list,
            "-np", str(placement.n_ranks),
            *payload_cmd,
        )
        return LaunchSpec(argv=argv, env_add={})

    # ── SLURM (multi-host) ───────────────────────────────────────────────

    def _realize_slurm(
        self,
        placement: Placement,
        payload_cmd: list[str],
        run_dir: Path,
    ) -> LaunchSpec:
        if self._srun_path is None:
            raise RuntimeError("srun not found on PATH")

        # SLURM hostfile: one hostname per rank — supports heterogeneous
        # per-node counts (e.g. 2 ranks on node A, 5 on node B).
        hostfile = run_dir / "srun_hostfile"
        lines: list[str] = []
        for s in placement.slices:
            lines.extend([s.host] * s.n_cpus)
        hostfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Per-rank CPU binding: build a global map_cpu list following
        # the hostfile rank order.  Each rank is pinned to its specific
        # core regardless of which node it lands on (heterogeneous-safe).
        global_cpu_list: list[str] = []
        for s in placement.slices:
            global_cpu_list.extend(str(c) for c in s.cpu_ids)
        cpu_bind_str = ",".join(global_cpu_list)
        argv = (
            self._srun_path,
            "--overlap", "--exact",
            "--mpi=pmi2",
            "--distribution=arbitrary",
            f"--cpu-bind=map_cpu:{cpu_bind_str}",
            "-n", str(placement.n_ranks),
            *payload_cmd,
        )
        # Strip ALL SLURM_ vars to remove step-scoped pollution,
        # then re-inject only what srun needs.
        return LaunchSpec(
            argv=argv,
            env_add={
                "SLURM_HOSTFILE": str(hostfile),
                "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", ""),
                "SLURM_CONF": self._slurm_conf,
            },
            env_strip_prefixes=("PMI_", "SLURM_"),
        )

    # ── SSH (multi-host, no SLURM) ───────────────────────────────────────────────

    def _realize_ssh(
        self,
        placement: Placement,
        payload_cmd: list[str],
        run_dir: Path,
    ) -> LaunchSpec:
        # OpenMPI rankfile: explicit per-rank host + CPU slot assignment.
        #   rank 0=hostA slot=0
        #   rank 1=hostA slot=1
        #   rank 2=hostB slot=15
        # OpenMPI SSHs to each host and pins each rank to its slot.
        rankfile = run_dir / "rankfile"
        lines: list[str] = []
        rank = 0
        for s in placement.slices:
            for c in s.cpu_ids:
                lines.append(f"rank {rank}={s.host} slot={c}")
                rank += 1
        rankfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
        argv = (
            self.mpirun_path,
            "--rankfile", str(rankfile),
            "-np", str(placement.n_ranks),
            *payload_cmd,
        )
        return LaunchSpec(argv=argv, env_add={})


# Backward-compat alias.
OpenMPISlurmBackend = OpenMpiBackend


# ── MPICH ───────────────────────────────────────────────

class MpichBackend(MpiBackend):
    """MPICH backend — supports local, SLURM, and SSH launches.

    **Local** (single-host): Hydra with ``-launcher fork`` and
    ``-bind-to user:`` for per-rank CPU pinning.

    **SLURM** (multi-host): ``srun --mpi=pmi2`` with per-rank CPU
    binding (``--cpu-bind=map_cpu``).  MPICH supports the PMI2 wire
    protocol natively, so ``srun`` spawns ranks directly.

    **SSH** (multi-host, no SLURM): Hydra with ``-launcher ssh`` and
    ``-f hostfile`` for host distribution, ``-bind-to user:`` for
    per-rank CPU pinning.  Requires passwordless SSH between compute
    nodes.
    """

    name = "mpich"
    supports_multi_host = True

    def __init__(self, mpirun_path: str) -> None:
        self.mpirun_path = mpirun_path
        self._srun_path: str | None = shutil.which("srun")
        self._slurm_conf: str = _find_slurm_conf()

    def realize(
        self,
        placement: Placement,
        payload_cmd: list[str],
        run_dir: Path,
    ) -> LaunchSpec:
        """Build the MPICH launch command for a placement."""
        if _is_local(placement):
            return self._realize_local(placement, payload_cmd)
        if os.environ.get("SLURM_JOB_ID"):
            return self._realize_slurm(placement, payload_cmd, run_dir)
        return self._realize_ssh(placement, payload_cmd, run_dir)

    # ── Local (single-host) ───────────────────────────────────────────────

    def _realize_local(
        self,
        placement: Placement,
        payload_cmd: list[str],
    ) -> LaunchSpec:
        # Pin each rank to its allocated core via Hydra's user binding.
        # -bind-to user:c0,c1,...  maps rank i → cpu_ids[i].
        cpu_list = ",".join(str(c) for c in placement.slices[0].cpu_ids)
        argv = (
            self.mpirun_path,
            "-launcher", "fork",
            "-bind-to", f"user:{cpu_list}",
            "-np", str(placement.n_ranks),
            *payload_cmd,
        )
        return LaunchSpec(argv=argv, env_add={})

    # ── SLURM (multi-host) ───────────────────────────────────────────────

    def _realize_slurm(
        self,
        placement: Placement,
        payload_cmd: list[str],
        run_dir: Path,
    ) -> LaunchSpec:
        if self._srun_path is None:
            raise RuntimeError("srun not found on PATH")
        # SLURM hostfile: one hostname per rank — supports heterogeneous
        # per-node counts (e.g. 2 ranks on node A, 5 on node B).
        hostfile = run_dir / "srun_hostfile"
        lines: list[str] = []
        for s in placement.slices:
            lines.extend([s.host] * s.n_cpus)
        hostfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Per-rank CPU binding: build a global map_cpu list following
        # the hostfile rank order.  Each rank is pinned to its specific
        # core regardless of which node it lands on (heterogeneous-safe).
        global_cpu_list: list[str] = []
        for s in placement.slices:
            global_cpu_list.extend(str(c) for c in s.cpu_ids)
        cpu_bind_str = ",".join(global_cpu_list)
        argv = (
            self._srun_path,
            "--overlap", "--exact",
            "--mpi=pmi2",
            "--distribution=arbitrary",
            f"--cpu-bind=map_cpu:{cpu_bind_str}",
            "-n", str(placement.n_ranks),
            *payload_cmd,
        )
        # Strip ALL SLURM_ vars to remove step-scoped pollution,
        # then re-inject only what srun needs.
        return LaunchSpec(
            argv=argv,
            env_add={
                "SLURM_HOSTFILE": str(hostfile),
                "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", ""),
                "SLURM_CONF": self._slurm_conf,
            },
            env_strip_prefixes=("PMI_", "SLURM_"),
        )

    # ── SSH (multi-host, no SLURM) ───────────────────────────────────────────────

    def _realize_ssh(
        self,
        placement: Placement,
        payload_cmd: list[str],
        run_dir: Path,
    ) -> LaunchSpec:
        # Hydra SSH launcher with machinefile.
        # Format: hostname:nprocs per line (block distribution).
        # The global cpu_list for -bind-to user: matches the rank order:
        # first N1 entries for host-1's cores, then N2 for host-2, etc.
        hostfile = run_dir / "ssh_hostfile"
        lines: list[str] = []
        for s in placement.slices:
            lines.append(f"{s.host}:{s.n_cpus}")
        hostfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
        global_cpu_list: list[str] = []
        for s in placement.slices:
            global_cpu_list.extend(str(c) for c in s.cpu_ids)
        cpu_bind_str = ",".join(global_cpu_list)
        argv = (
            self.mpirun_path,
            "-launcher", "ssh",
            "-f", str(hostfile),
            "-bind-to", f"user:{cpu_bind_str}",
            "-np", str(placement.n_ranks),
            *payload_cmd,
        )
        return LaunchSpec(argv=argv, env_add={})


# Backward-compat alias.
MPICHSlurmBackend = MpichBackend


# ── Local mpirun (any MPI, single-host only) ───────────────────────────────────────────────

class LocalMpirunBackend(MpiBackend):
    """Any MPI implementation, single-host shared-memory only.

    Falls back to the most portable flags per detected MPI family.
    """

    name = "local-mpirun"
    supports_multi_host = False

    def __init__(self, mpirun_path: str, mpi_family: str = "unknown") -> None:
        self.mpirun_path = mpirun_path
        self.mpi_family = mpi_family

    def realize(
        self,
        placement: Placement,
        payload_cmd: list[str],
        run_dir: Path,
    ) -> LaunchSpec:
        """Build a single-host generic ``mpirun`` launch command."""
        if not placement.single_host:
            raise RuntimeError(
                f"LocalMpirunBackend does not support multi-host placements "
                f"(got {len(placement.slices)} slices)."
            )
        cpu_list = ",".join(str(c) for c in placement.slices[0].cpu_ids)
        extra: list[str] = []
        env_add: dict[str, str] = {}
        if self.mpi_family == "openmpi":
            extra = ["--mca", "btl", "self,vader",
                      "--bind-to", "core", "--cpu-set", cpu_list]
        elif self.mpi_family == "intel":
            env_add = {"I_MPI_FABRICS": "shm", "I_MPI_HYDRA_BOOTSTRAP": "fork",
                       "I_MPI_PIN_PROCESSOR_LIST": cpu_list}
        elif self.mpi_family == "mpich":
            extra = ["-launcher", "fork", "-bind-to", f"user:{cpu_list}"]
        argv = (
            self.mpirun_path,
            *extra,
            "-np", str(placement.n_ranks),
            *payload_cmd,
        )
        return LaunchSpec(argv=argv, env_add=env_add)


# ── MPI family detection ───────────────────────────────────────────────

def _normalize_mpi_family(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower().replace("-", "").replace("_", "")
    if token in {"", "auto", "autodetect"}:
        return None
    aliases = {
        "intel": "intel",
        "intelmpi": "intel",
        "impi": "intel",
        "openmpi": "openmpi",
        "openrte": "openmpi",
        "ompi": "openmpi",
        "mpich": "mpich",
        "hydra": "mpich",
        "unknown": "unknown",
    }
    family = aliases.get(token)
    if family is None:
        raise ValueError(
            f"Unsupported MPI family override {value!r}; expected one of "
            "'intel', 'openmpi', 'mpich', or 'auto'."
        )
    return family


def _resolve_launcher_path(exe: str) -> str:
    if os.path.dirname(exe):
        return exe
    resolved = shutil.which(exe)
    return resolved or exe


def _read_text_prefix(path: str, limit: int = 4096) -> str:
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(limit)
    except OSError:
        return ""
    if b"\x00" in prefix:
        return ""
    return prefix.decode("utf-8", errors="ignore").lower()


def _probe_executable_output(exe: str, *argsets: tuple[str, ...]) -> str:
    for argset in argsets:
        try:
            result = subprocess.run(
                [exe, *argset],
                capture_output=True,
                text=True,
                errors="ignore",
                timeout=5,
            )
        except Exception:
            continue
        output = (result.stdout + result.stderr).strip()
        if output:
            return output.lower()
    return ""


def _detect_mpi_family_from_output(output: str) -> str | None:
    if not output:
        return None
    if "intel(r) mpi library" in output or "intel mpi library" in output:
        return "intel"
    if "open mpi" in output or "openrte" in output:
        return "openmpi"
    if "hydra build details" in output or "mpich version" in output:
        return "mpich"
    return None


def _detect_mpi_family_from_wrapper_text(text: str) -> str | None:
    if not text:
        return None
    if any(
        marker in text for marker in (
            "intel corporation",
            "i_mpi_mpirun",
            "i_mpi_hydra_bootstrap",
            "mpivars.sh",
        )
    ):
        return "intel"
    if "open mpi" in text or "openrte" in text or "orterun" in text:
        return "openmpi"
    return None


def _detect_mpi_family_from_siblings(exe: str) -> str | None:
    if not os.path.isabs(exe):
        return None
    bindir = os.path.dirname(exe)
    if not bindir:
        return None

    orterun = os.path.join(bindir, "orterun")
    if os.path.isfile(orterun) and os.access(orterun, os.X_OK):
        family = _detect_mpi_family_from_output(
            _probe_executable_output(orterun, ("--version",)),
        )
        if family is not None:
            return family
        return "openmpi"

    hydra = os.path.join(bindir, "mpiexec.hydra")
    if os.path.isfile(hydra) and os.access(hydra, os.X_OK):
        family = _detect_mpi_family_from_output(
            _probe_executable_output(hydra, ("-version",), ("--version",)),
        )
        if family is not None:
            return family
    return None


def detect_mpi_family(mpirun_path: str | None = None) -> str:
    """Return ``'intel'`` | ``'openmpi'`` | ``'mpich'`` | ``'unknown'``."""
    exe = mpirun_path or shutil.which("mpirun") or shutil.which("mpiexec")
    if exe is None:
        return "unknown"

    resolved = _resolve_launcher_path(exe)
    real_exe = os.path.realpath(resolved) if os.path.exists(resolved) else resolved

    family = _detect_mpi_family_from_wrapper_text(_read_text_prefix(real_exe))
    if family is not None:
        return family

    if os.path.basename(real_exe).lower() == "orterun":
        return "openmpi"

    family = _detect_mpi_family_from_siblings(real_exe)
    if family is not None:
        return family

    family = _detect_mpi_family_from_output(
        _probe_executable_output(real_exe, ("--version",), ("-version",)),
    )
    if family is not None:
        return family
    return "unknown"


def pick_backend(
    mpirun_path: str,
    *,
    in_slurm: bool | None = None,
    intel_launch_mode: str = "mpmd",
    mpi_family: str | None = None,
) -> MpiBackend:
    """Select the appropriate MPI backend.

    Each backend supports local, SLURM, and SSH launches internally.
    The ``in_slurm`` hint is used only as a last resort when the MPI
    family cannot be detected.

    Parameters
    ----------
    mpirun_path
        Path to the ``mpirun`` / ``mpiexec`` binary.
    in_slurm
        Whether we are inside a SLURM allocation.  Auto-detected from
        ``SLURM_JOB_ID`` if not specified.
    intel_launch_mode
        For Intel MPI on SLURM: ``"mpmd"`` (default, Hydra bootstrap)
        or ``"srun"`` (direct srun --mpi=pmi2).
    mpi_family
        Optional family override: ``"intel"``, ``"openmpi"``,
        ``"mpich"``, or ``"auto"`` / ``None`` for autodetection.
    """
    family = _normalize_mpi_family(mpi_family)
    if family is None:
        family = detect_mpi_family(mpirun_path)
    if in_slurm is None:
        in_slurm = bool(os.environ.get("SLURM_JOB_ID"))

    if family == "intel":
        return IntelMpiBackend(mpirun_path, launch_mode=intel_launch_mode)
    if family == "openmpi":
        return OpenMpiBackend(mpirun_path)
    if family == "mpich":
        return MpichBackend(mpirun_path)

    # Unknown MPI family: local-only fallback.
    return LocalMpirunBackend(mpirun_path, mpi_family=family)
