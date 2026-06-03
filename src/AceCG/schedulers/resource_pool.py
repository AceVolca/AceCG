"""Resource discovery and CPU lease management for the AceCG scheduler.

Provides three tiers:

1. **HostInventory** — discovered compute hosts with CPU ID lists
2. **CpuLease / LeasePool** — dynamic per-CPU allocation (replaces fixed-slot)
3. **ResourcePool** — discovery + backend construction
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import warnings
from dataclasses import dataclass

from .mpi_backend import MpiBackend, pick_backend


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HostInventory:
    """One compute host with its available CPU cores."""

    hostname: str
    cpu_ids: tuple[int, ...]

    @property
    def n_cpus(self) -> int:
        """Return the number of CPU cores available on this host."""
        return len(self.cpu_ids)


@dataclass(frozen=True)
class CpuLease:
    """A block of CPU cores leased to one task."""

    host: str
    cpu_ids: tuple[int, ...]

    @property
    def n_cpus(self) -> int:
        """Return the number of CPU cores in this lease."""
        return len(self.cpu_ids)

    def taskset_arg(self) -> str:
        """``taskset -c`` argument, e.g. ``'0-7'`` or ``'0,2,4,6'``."""
        ids = sorted(self.cpu_ids)
        if ids == list(range(ids[0], ids[0] + len(ids))):
            return f"{ids[0]}-{ids[-1]}"
        return ",".join(str(i) for i in ids)


# ---------------------------------------------------------------------------
# LeasePool
# ---------------------------------------------------------------------------

class LeasePool:
    """Dynamic CPU allocation pool across multiple hosts.

    Thread-unsafe — only used from the single-threaded scheduler poll loop.
    """

    def __init__(self, hosts: list[HostInventory]) -> None:
        self._free: dict[str, list[int]] = {
            h.hostname: sorted(h.cpu_ids) for h in hosts
        }
        self._total: dict[str, int] = {
            h.hostname: h.n_cpus for h in hosts
        }

    def acquire(self, cpu_cores: int, *, prefer_host: str | None = None) -> CpuLease:
        """Lease *cpu_cores* CPUs from the host with the most free cores.

        Raises ``RuntimeError`` if no single host has enough free cores.
        """
        candidates = [
            (h, cpus) for h, cpus in self._free.items()
            if len(cpus) >= cpu_cores
        ]
        if not candidates:
            summary = {h: len(c) for h, c in self._free.items()}
            raise RuntimeError(
                f"No host has {cpu_cores} free CPUs. Free: {summary}"
            )
        if prefer_host:
            preferred = [(h, c) for h, c in candidates if h == prefer_host]
            if preferred:
                candidates = preferred
        host, cpus = max(candidates, key=lambda x: len(x[1]))
        allocated = _allocate_contiguous(cpus, cpu_cores)
        for cid in allocated:
            cpus.remove(cid)
        return CpuLease(host=host, cpu_ids=tuple(allocated))

    def release(self, lease: CpuLease) -> None:
        """Return leased CPUs to the free pool."""
        self._free.setdefault(lease.host, []).extend(lease.cpu_ids)
        self._free[lease.host].sort()

    def free_total(self) -> int:
        """Return the total number of currently free CPU cores."""
        return sum(len(c) for c in self._free.values())

    def max_free_on_any_host(self) -> int:
        """Return the largest free-core count on any single host."""
        return max((len(c) for c in self._free.values()), default=0)

    def host_total(self, hostname: str) -> int:
        """Total CPUs for *hostname* (free + leased)."""
        return self._total.get(hostname, 0)


def _allocate_contiguous(free: list[int], n: int) -> list[int]:
    """Allocate *n* CPUs, preferring contiguous IDs.

    Strategy — all paths operate on the actual free-ID list (never fabricate
    ranges):

    1. Prefer the first window of ``n`` whose span equals ``n - 1`` (a fully
       contiguous block — tightest NUMA locality).
    2. If no such block exists (fragmented pool), pick the window with the
       **narrowest span** ``window[-1] - window[0]``.  This keeps the
       allocation as tightly clustered as the fragmentation allows, instead
       of blindly slicing the smallest-ID prefix and potentially straddling
       socket boundaries.

    Ties broken by smaller starting ID (deterministic, numa-friendly).
    """
    s = sorted(free)
    if len(s) < n:
        raise ValueError(
            f"Cannot allocate {n} CPUs from only {len(s)} free CPU IDs."
        )
    best_window: list[int] | None = None
    best_span: int | None = None
    for i in range(len(s) - n + 1):
        window = s[i : i + n]
        span = window[-1] - window[0]
        if span == n - 1:
            return window
        if best_span is None or span < best_span:
            best_span = span
            best_window = window
    return best_window if best_window is not None else s[:n]


# ---------------------------------------------------------------------------
# Placer
# ---------------------------------------------------------------------------

from .mpi_backend import HostSlice, Placement


class PlacementResult:
    """Result of a successful placement: a Placement + leases to track."""

    __slots__ = ("placement", "leases")

    def __init__(
        self,
        placement: Placement,
        leases: list[CpuLease],
    ) -> None:
        self.placement = placement
        self.leases = leases


class Placer:
    """Place a task onto a ``LeasePool``.

    Implements best-fit-decreasing host selection with contiguous core
    preference.  Supports multi-host fallback when the backend declares
    ``supports_multi_host`` and the request allows it.
    """

    def __init__(self, lease_pool: LeasePool, *, backend: MpiBackend) -> None:
        self.pool = lease_pool
        self.backend = backend

    def place(
        self,
        min_cores: int,
        preferred_cores: int,
        *,
        single_host_only: bool = True,
    ) -> PlacementResult | None:
        """Try to place a task; return a ``PlacementResult`` or ``None``.

        Single-host path: tries *preferred_cores* down to *min_cores*.
        Multi-host path (opt-in): greedy pack across hosts when no single
        host can satisfy *min_cores* and the backend supports it.
        """
        # --- single-host path (always tried first) ---
        for n in range(preferred_cores, min_cores - 1, -1):
            if self.pool.max_free_on_any_host() >= n:
                lease = self.pool.acquire(n)
                placement = Placement.from_host_cores(
                    lease.host, lease.cpu_ids, n_ranks=n,
                )
                return PlacementResult(placement, [lease])
        if single_host_only or not self.backend.supports_multi_host:
            return None

        # --- multi-host fallback ---
        # Greedy: pack across hosts by descending free-cores until we
        # reach at least min_cores (up to preferred_cores).
        total_free = self.pool.free_total()
        if total_free < min_cores:
            return None

        target = min(preferred_cores, total_free)
        leases: list[CpuLease] = []
        collected = 0
        host_free = sorted(
            self.pool._free.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )
        for hostname, free_cpus in host_free:
            if collected >= target:
                break
            avail = len(free_cpus)
            if avail == 0:
                continue
            take = min(avail, target - collected)
            lease = self.pool.acquire(take, prefer_host=hostname)
            leases.append(lease)
            collected += take

        if collected < min_cores:
            # Not enough even across hosts; release what we took
            for lease in leases:
                self.pool.release(lease)
            return None

        slices = tuple(
            HostSlice(host=l.host, cpu_ids=l.cpu_ids,
                      host_n_cpus=self.pool.host_total(l.host))
            for l in leases
        )
        placement = Placement(slices=slices, n_ranks=collected)
        return PlacementResult(placement, leases)


# ---------------------------------------------------------------------------
# ResourcePool
# ---------------------------------------------------------------------------

class ResourcePool:
    """Discover compute resources and select the MPI backend.

    Construct via the ``discover()`` classmethod for automatic detection,
    or directly for full control.  All vendor-specific MPI assembly lives
    in ``self.backend`` (an ``MpiBackend`` instance).
    """

    def __init__(
        self,
        hosts: list[HostInventory],
        *,
        sim_cmd: list[str],
        backend: MpiBackend,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.hosts = hosts
        self.sim_cmd = list(sim_cmd)
        self.backend = backend
        self.extra_env = dict(extra_env or {})

    @classmethod
    def discover(
        cls,
        *,
        sim_cmd: list[str],
        explicit_hosts: list[tuple[str, tuple[int, ...]]] | None = None,
        mpirun_path: str | None = None,
        mpi_family: str | None = None,
        extra_env: dict[str, str] | None = None,
        intel_launch_mode: str = "mpmd",
    ) -> ResourcePool:
        """Auto-discover compute resources and select the MPI backend.

        Discovery order for hosts:
        1. *explicit_hosts* (user override)
        2. ``scontrol show job -d`` (SLURM, no SSH needed)
        3. SLURM_NODELIST + per-host CPU discovery via SSH (fallback)
        4. localhost + ``os.sched_getaffinity(0)``
        """
        if explicit_hosts is not None:
            hosts = [
                HostInventory(h, tuple(sorted(cpus)))
                for h, cpus in explicit_hosts
            ]
        else:
            hosts = _discover_hosts()
        if not hosts:
            raise RuntimeError("No compute hosts discovered")

        mpirun = (
            mpirun_path
            or shutil.which("mpirun")
            or shutil.which("mpiexec")
        )
        if mpirun is None:
            local_hostname = socket.gethostname().split(".")[0]
            remote = [h for h in hosts if h.hostname != local_hostname]
            if remote:
                warnings.warn(
                    "MPI launcher not found on PATH and scheduler.mpirun_path "
                    "was not set.  Dropping "
                    f"{len(remote)} remote host(s) and continuing with local "
                    "host only; this can silently reduce a multi-node "
                    "allocation to one node.  Load the MPI/LAMMPS module or "
                    "set scheduler.mpirun_path to use the full allocation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                hosts = [h for h in hosts if h.hostname == local_hostname]
            if not hosts:
                raise RuntimeError(
                    "No usable hosts: remote hosts discovered but "
                    "no mpirun available for SSH-based launching"
                )
            mpirun = "mpirun"

        backend = pick_backend(
            mpirun,
            intel_launch_mode=intel_launch_mode,
            mpi_family=mpi_family,
        )

        return cls(
            hosts=hosts,
            sim_cmd=sim_cmd,
            backend=backend,
            extra_env=extra_env,
        )

    def build_lease_pool(self) -> LeasePool:
        """Create a fresh lease pool from this resource inventory."""
        return LeasePool(self.hosts)

    def __repr__(self) -> str:
        hosts = ", ".join(f"{h.hostname}({h.n_cpus})" for h in self.hosts)
        return f"ResourcePool([{hosts}], backend={self.backend.name!r})"


# ---------------------------------------------------------------------------
# Host discovery
# ---------------------------------------------------------------------------

def _discover_hosts() -> list[HostInventory]:
    """Auto-discover hosts from SLURM env or localhost.

    Discovery order:
    1. SLURM allocation: ``scontrol show job -d`` for per-node CPU IDs
    2. SLURM allocation fallback: ``SLURM_NODELIST`` + SSH probe
    3. Non-SLURM: localhost + ``os.sched_getaffinity(0)``
    """
    nodelist = os.environ.get("SLURM_NODELIST", "")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    local_hostname = socket.gethostname().split(".")[0]

    if not nodelist and not job_id:
        return [HostInventory(local_hostname, _discover_local_cpus())]

    # Preferred: scontrol gives exact per-node CPU IDs without SSH.
    # Works even when SLURM_NODELIST is absent (e.g. SSH into a
    # compute node where pam_slurm_adopt sets only SLURM_JOB_ID).
    hosts = _discover_hosts_scontrol()
    if hosts:
        return hosts

    if not nodelist:
        return [HostInventory(local_hostname, _discover_local_cpus())]

    # Fallback: parse SLURM_NODELIST + SSH probe per remote host.
    hostnames = _parse_slurm_nodelist(nodelist)
    local_cpus = _discover_local_cpus()
    hosts = []
    for hostname in hostnames:
        short = hostname.split(".")[0]
        if short == local_hostname or hostname == "localhost":
            hosts.append(HostInventory(short, local_cpus))
        else:
            remote = _discover_remote_cpus(hostname)
            if remote is None:
                warnings.warn(
                    f"Cannot discover CPUs on {hostname!r} via SSH; "
                    f"skipping host (use explicit_hosts to override)",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            hosts.append(HostInventory(short, remote))
    return hosts


def _discover_hosts_scontrol() -> list[HostInventory]:
    """Discover per-node CPU IDs via ``scontrol show job -d``.

    Two-tier parse, both driven from the same ``scontrol`` output:

    1. **Detail lines** ``Nodes=<h> CPU_IDs=<range> ...`` give exact
       per-node CPU allocations.  ``CPU_IDs=all`` is reconciled against
       the job's ``NumCPUs=`` field (single-node allocs on exclusive
       nodes sometimes report the literal string ``all``).
    2. **Fallback header fields** ``NodeList=`` + ``NumCPUs=`` /
       ``NumNodes=`` are used when the detail section is absent
       (some SLURM configurations suppress it when ``JOB_GRES=(null)``).

    Returns an empty list on any failure; caller falls through to SSH probing
    or localhost.  Node order is preserved in SLURM's intrinsic order so
    MPMD segment assembly can rely on it later.
    """
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id:
        return []
    scontrol = shutil.which("scontrol")
    if not scontrol:
        return []
    try:
        result = subprocess.run(
            [scontrol, "show", "job", job_id, "-d"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
    except Exception:
        return []

    text = result.stdout
    fields = _parse_scontrol_fields(text)

    detail_hosts = _scontrol_detail_hosts(text, fields)
    if detail_hosts:
        return detail_hosts

    return _scontrol_header_fallback_hosts(fields)


def _parse_scontrol_fields(text: str) -> dict[str, str]:
    """Collect ``key=value`` tokens across all lines of scontrol output.

    scontrol prints multi-token lines like ``NumNodes=1 NumCPUs=64 ...``;
    we flatten the whole blob into a flat key→value dict.  Later keys
    overwrite earlier ones (OK for header fields — they're unique).
    """
    fields: dict[str, str] = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, _, val = token.partition("=")
        if key and val:
            fields[key] = val
    return fields


def _scontrol_detail_hosts(
    text: str, fields: dict[str, str],
) -> list[HostInventory]:
    """Primary path: ``Nodes=<h> CPU_IDs=<range>`` detail lines."""
    hosts: list[HostInventory] = []
    num_cpus_total = _safe_int(fields.get("NumCPUs"))
    for line in text.splitlines():
        m = re.match(r"\s*Nodes=(\S+)\s+CPU_IDs=(\S+)", line)
        if not m:
            continue
        cpu_ids = _parse_cpu_range(m.group(2), fallback_n=num_cpus_total)
        if cpu_ids:
            for hostname in _parse_slurm_nodelist(m.group(1)):
                hosts.append(HostInventory(hostname.split(".")[0], tuple(sorted(cpu_ids))))
    return hosts


def _scontrol_header_fallback_hosts(
    fields: dict[str, str],
) -> list[HostInventory]:
    """Fallback: reconstruct from ``NodeList=`` + ``NumCPUs=``/``NumNodes=``.

    Used when scontrol omits the ``Nodes=... CPU_IDs=...`` detail block
    (seen on exclusive-node allocations where ``JOB_GRES=(null)`` skips
    the detailed accounting section).  We assume block distribution —
    ``NumCPUs`` split evenly across ``NumNodes`` hosts.
    """
    nodelist_raw = fields.get("NodeList") or fields.get("BatchHost")
    if not nodelist_raw or nodelist_raw == "(null)":
        return []
    hostnames = _parse_slurm_nodelist(nodelist_raw)
    if not hostnames:
        return []
    num_cpus_total = _safe_int(fields.get("NumCPUs"))
    num_nodes = _safe_int(fields.get("NumNodes")) or len(hostnames)
    if num_cpus_total is None or num_nodes <= 0:
        return []
    per_node = num_cpus_total // num_nodes
    if per_node <= 0:
        return []
    hosts: list[HostInventory] = []
    for hostname in hostnames:
        cpu_ids = tuple(range(per_node))
        hosts.append(HostInventory(hostname.split(".")[0], cpu_ids))
    return hosts


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_cpu_range(s: str, *, fallback_n: int | None = None) -> list[int]:
    """Parse a CPU range string like ``'16-21'`` or ``'36-37,39-42'``.

    ``'all'`` (or ``'(null)'``) is treated as "all of ``fallback_n`` cores
    starting at 0" — seen on exclusive single-node allocations.  Returns
    an empty list if the string is uninterpretable so callers can skip
    the node instead of crashing.
    """
    s = (s or "").strip()
    if not s:
        return []
    if s.lower() in {"all", "(null)", "null", "none"}:
        if fallback_n and fallback_n > 0:
            return list(range(fallback_n))
        return []
    result: list[int] = []
    try:
        for token in s.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                lo, hi = token.split("-", 1)
                result.extend(range(int(lo), int(hi) + 1))
            else:
                result.append(int(token))
    except ValueError:
        return []
    return result


def _discover_local_cpus() -> tuple[int, ...]:
    """Discover available CPU IDs on the local host."""
    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:
        return tuple(range(os.cpu_count() or 1))


def _discover_remote_cpus(hostname: str) -> tuple[int, ...] | None:
    """Discover CPUs on a remote host via SSH.  Returns None if SSH fails."""
    try:
        result = subprocess.run(
            [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                hostname, "python3", "-c",
                "import os,json; print(json.dumps(sorted(os.sched_getaffinity(0))))",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return tuple(json.loads(result.stdout.strip()))
    except Exception:
        pass
    # Fallback: nproc
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
             hostname, "nproc"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return tuple(range(int(result.stdout.strip())))
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# SLURM_NODELIST parser
# ---------------------------------------------------------------------------

def _parse_slurm_nodelist(nodelist: str) -> list[str]:
    """Expand ``SLURM_NODELIST`` into individual hostnames."""
    parts = _split_nodelist_top(nodelist.strip())
    hosts: list[str] = []
    for part in parts:
        hosts.extend(_expand_node_part(part))
    return hosts


def _split_nodelist_top(s: str) -> list[str]:
    """Split comma-separated nodelist at top level (depth=0)."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _expand_node_part(part: str) -> list[str]:
    """Expand one node-part (prefix + optional bracket range)."""
    m = re.match(r"^(.*?)\[([^\]]+)\](.*)$", part)
    if m is None:
        return [part]
    prefix, range_str, suffix = m.group(1), m.group(2), m.group(3)
    result: list[str] = []
    for token in range_str.split(","):
        if "-" in token:
            lo, hi = token.split("-", 1)
            width = len(lo)
            for i in range(int(lo), int(hi) + 1):
                result.append(f"{prefix}{str(i).zfill(width)}{suffix}")
        else:
            result.append(f"{prefix}{token}{suffix}")
    return result
