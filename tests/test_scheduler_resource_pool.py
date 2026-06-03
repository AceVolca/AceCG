"""Unit tests for resource_pool.py and mpi_backend.py (Phase 1)."""

import os
from pathlib import Path

import pytest

from AceCG.schedulers.resource_pool import (
    HostInventory,
    CpuLease,
    LeasePool,
    ResourcePool,
    _parse_slurm_nodelist,
)
from AceCG.schedulers.mpi_backend import (
    HostSlice,
    LaunchSpec,
    LocalMpirunBackend,
    IntelMpiBackend,
    OpenMpiBackend,
    MpichBackend,
    Placement,
    detect_mpi_family,
    pick_backend,
)


# ---------------------------------------------------------------------------
# SLURM_NODELIST parser
# ---------------------------------------------------------------------------

def test_parse_single_node():
    assert _parse_slurm_nodelist("midway3-0001") == ["midway3-0001"]


def test_parse_range():
    result = _parse_slurm_nodelist("midway3-[0001-0003]")
    assert result == ["midway3-0001", "midway3-0002", "midway3-0003"]


def test_parse_range_with_gap():
    result = _parse_slurm_nodelist("midway3-[0001-0003,0005]")
    assert result == ["midway3-0001", "midway3-0002", "midway3-0003", "midway3-0005"]


def test_parse_multiple_prefixes():
    result = _parse_slurm_nodelist("node[001-003],gpu[01-02]")
    assert result == ["node001", "node002", "node003", "gpu01", "gpu02"]


def test_parse_no_bracket():
    assert _parse_slurm_nodelist("midway3-0386") == ["midway3-0386"]


# ---------------------------------------------------------------------------
# LeasePool
# ---------------------------------------------------------------------------

def test_lease_pool_acquire_release():
    hosts = [HostInventory("h1", tuple(range(8)))]
    pool = LeasePool(hosts)
    assert pool.free_total() == 8
    lease = pool.acquire(4)
    assert lease.n_cpus == 4
    assert pool.free_total() == 4
    pool.release(lease)
    assert pool.free_total() == 8


def test_lease_pool_contiguous_preference():
    hosts = [HostInventory("h1", (0, 1, 2, 3, 5, 6, 7, 8))]
    pool = LeasePool(hosts)
    lease = pool.acquire(4)
    ids = sorted(lease.cpu_ids)
    assert ids[-1] - ids[0] == 3, "Should allocate a contiguous block"


def test_lease_pool_multi_host():
    hosts = [
        HostInventory("h1", tuple(range(4))),
        HostInventory("h2", tuple(range(8))),
    ]
    pool = LeasePool(hosts)
    lease = pool.acquire(6)
    assert lease.host == "h2", "Should pick host with most free cores"
    assert lease.n_cpus == 6


def test_lease_pool_no_capacity():
    hosts = [HostInventory("h1", tuple(range(4)))]
    pool = LeasePool(hosts)
    with pytest.raises(RuntimeError, match="No host has 8 free CPUs"):
        pool.acquire(8)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def test_placement_single_host():
    p = Placement.from_host_cores("h1", (0, 1, 2, 3))
    assert p.single_host
    assert p.n_ranks == 4
    assert p.slices[0].host == "h1"


def test_placement_multi_host():
    p = Placement(
        slices=(HostSlice("h1", (0, 1)), HostSlice("h2", (0, 1, 2))),
        n_ranks=5,
    )
    assert not p.single_host
    assert p.n_ranks == 5


# ---------------------------------------------------------------------------
# LocalMpirunBackend
# ---------------------------------------------------------------------------

def test_local_backend_openmpi(tmp_path):
    b = LocalMpirunBackend("/usr/bin/mpirun", mpi_family="openmpi")
    p = Placement.from_host_cores("localhost", tuple(range(4)))
    spec = b.realize(p, ["lmp"], tmp_path)
    assert spec.argv == (
        "/usr/bin/mpirun", "--mca", "btl", "self,vader",
        "--bind-to", "core", "--cpu-set", "0,1,2,3",
        "-np", "4", "lmp",
    )
    assert spec.env_add == {}


def test_local_backend_intel(tmp_path):
    b = LocalMpirunBackend("/usr/bin/mpirun", mpi_family="intel")
    p = Placement.from_host_cores("localhost", tuple(range(8)))
    spec = b.realize(p, ["lmp"], tmp_path)
    # No BTL flags, but env_add sets shm fabric + per-core pinning
    assert "-np" in spec.argv
    assert spec.argv[-1] == "lmp"
    assert spec.env_add["I_MPI_FABRICS"] == "shm"
    assert spec.env_add["I_MPI_PIN_PROCESSOR_LIST"] == "0,1,2,3,4,5,6,7"


def test_local_backend_mpich(tmp_path):
    b = LocalMpirunBackend("/usr/bin/mpirun", mpi_family="mpich")
    p = Placement.from_host_cores("localhost", tuple(range(4)))
    spec = b.realize(p, ["lmp"], tmp_path)
    assert "-launcher" in spec.argv
    assert "fork" in spec.argv
    assert "-bind-to" in spec.argv
    assert "user:0,1,2,3" in spec.argv


def test_local_backend_rejects_multi_host(tmp_path):
    b = LocalMpirunBackend("/usr/bin/mpirun")
    p = Placement(
        slices=(HostSlice("h1", (0,)), HostSlice("h2", (0,))),
        n_ranks=2,
    )
    with pytest.raises(RuntimeError, match="multi-host"):
        b.realize(p, ["lmp"], tmp_path)


# ---------------------------------------------------------------------------
# IntelMpiBackend
# ---------------------------------------------------------------------------

def test_intel_backend_single_host(tmp_path):
    b = IntelMpiBackend("/usr/bin/mpirun")
    p = Placement.from_host_cores("localhost", tuple(range(8)))
    spec = b.realize(p, ["lmp"], tmp_path)
    assert spec.argv == ("/usr/bin/mpirun", "-np", "8", "lmp")
    assert spec.env_add["I_MPI_FABRICS"] == "shm"
    assert spec.env_add["I_MPI_PIN_PROCESSOR_LIST"] == "0,1,2,3,4,5,6,7"


def test_intel_backend_cross_host_srun(monkeypatch, tmp_path):
    """Intel SLURM uses MPMD with per-segment CPU pinning.

    srun --mpi=pmi2 deadlocks during MPI_Init.  SSH bootstrap fails
    (no passwordless SSH between compute nodes).  SLURM bootstrap with
    per-segment I_MPI_PIN_PROCESSOR_LIST gives precise per-core control
    and allows concurrent sub-node tasks.
    """
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_CONF", "/etc/slurm/slurm.conf")
    b = IntelMpiBackend("/usr/bin/mpirun")
    b._libpmi2_path = "/software/slurm/lib/libpmi2.so"
    p = Placement(
        slices=(HostSlice("h1", (0, 1)), HostSlice("h2", (0, 1, 2))),
        n_ranks=5,
    )
    spec = b.realize(p, ["lmp"], tmp_path)
    # SLURM bootstrap with MPMD colon syntax
    assert spec.argv[0] == "/usr/bin/mpirun"
    assert "-bootstrap" in spec.argv
    assert "slurm" in spec.argv
    assert ":" in spec.argv
    assert "--mpi=pmi2" not in spec.argv
    # Per-host CPU pinning via -env
    argv = list(spec.argv)
    # Check that I_MPI_PIN_PROCESSOR_LIST appears per-host
    pin_indices = [i for i, x in enumerate(argv) if x == "I_MPI_PIN_PROCESSOR_LIST"]
    assert len(pin_indices) == 2  # one per host


def test_intel_backend_cross_host_mpmd_fallback(monkeypatch, tmp_path):
    """Intel SLURM uses MPMD with per-segment pinning regardless of libpmi2."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    b = IntelMpiBackend("/usr/bin/mpirun")
    b._libpmi2_path = None  # no libpmi2 - still uses SLURM MPMD
    p = Placement(
        slices=(HostSlice("h1", (0, 1)), HostSlice("h2", (0, 1, 2))),
        n_ranks=5,
    )
    spec = b.realize(p, ["lmp"], tmp_path)
    assert spec.argv[0] == "/usr/bin/mpirun"
    assert "-bootstrap" in spec.argv
    assert "slurm" in spec.argv
    # MPMD colon syntax with per-segment CPU pinning
    assert ":" in spec.argv
    argv = list(spec.argv)
    pin_indices = [i for i, x in enumerate(argv) if x == "I_MPI_PIN_PROCESSOR_LIST"]
    assert len(pin_indices) == 2  # one per host


def test_intel_backend_cross_host_ssh(monkeypatch, tmp_path):
    """No SLURM → SSH mode with Hydra SSH bootstrap + MPMD colon syntax."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    b = IntelMpiBackend("/usr/bin/mpirun")
    p = Placement(
        slices=(HostSlice("h1", (0, 1)), HostSlice("h2", (4, 5, 6))),
        n_ranks=5,
    )
    spec = b.realize(p, ["lmp"], tmp_path)
    assert spec.argv[0] == "/usr/bin/mpirun"
    assert "-bootstrap" in spec.argv
    assert "ssh" in spec.argv
    # MPMD colon syntax with per-host CPU pinning
    assert ":" in spec.argv
    argv = list(spec.argv)
    idx_colon = argv.index(":")
    seg0 = argv[3:idx_colon]   # after mpirun -bootstrap ssh
    # Segment 0: -n 2 -host h1 -env I_MPI_PIN 1 -env I_MPI_PIN_PROCESSOR_LIST 0,1 lmp
    assert "-n" in seg0 and "2" in seg0
    assert "-host" in seg0 and "h1" in seg0
    assert "-env" in seg0
    assert "0,1" in seg0
    assert "lmp" in seg0
    # Segment 1
    seg1 = argv[idx_colon + 1:]
    assert "-n" in seg1 and "3" in seg1
    assert "-host" in seg1 and "h2" in seg1
    assert "4,5,6" in seg1


# ---------------------------------------------------------------------------
# OpenMpiBackend
# ---------------------------------------------------------------------------

def test_openmpi_backend_single(tmp_path):
    b = OpenMpiBackend("/usr/bin/mpirun")
    p = Placement.from_host_cores("localhost", tuple(range(4)))
    spec = b.realize(p, ["lmp"], tmp_path)
    assert "--mca" in spec.argv
    assert "self,vader" in spec.argv
    assert "--bind-to" in spec.argv
    assert "core" in spec.argv
    assert "--cpu-set" in spec.argv
    assert "0,1,2,3" in spec.argv
    assert "-np" in spec.argv


def test_openmpi_backend_multi(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    b = OpenMpiBackend("/usr/bin/mpirun")
    p = Placement(
        slices=(HostSlice("h1", (0, 1)), HostSlice("h2", (0, 1, 2))),
        n_ranks=5,
    )
    spec = b.realize(p, ["lmp"], tmp_path)
    # Multi-host uses srun --mpi=pmi2 + SLURM_HOSTFILE
    assert "--mpi=pmi2" in spec.argv
    assert "--distribution=arbitrary" in spec.argv
    assert "-n" in spec.argv
    assert "5" in spec.argv
    assert "lmp" in spec.argv
    # Verify SLURM_HOSTFILE has heterogeneous per-node rank counts
    hostfile_path = spec.env_add["SLURM_HOSTFILE"]
    content = Path(hostfile_path).read_text()
    assert content.count("h1") == 2
    assert content.count("h2") == 3
    # Verify per-rank CPU binding from all slices (global cpu list)
    assert "--cpu-bind=map_cpu:0,1,0,1,2" in spec.argv


def test_openmpi_backend_multi_nonzero_cpuids(tmp_path, monkeypatch):
    """map_cpu list reflects actual cpu_ids, not 0..N-1."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    b = OpenMpiBackend("/usr/bin/mpirun")
    p = Placement(
        slices=(HostSlice("h1", (16, 17, 18, 19)),),
        n_ranks=4,
    )
    spec = b.realize(p, ["lmp"], tmp_path)
    assert "--cpu-bind=map_cpu:16,17,18,19" in spec.argv


def test_openmpi_backend_ssh(tmp_path, monkeypatch):
    """No SLURM → SSH mode with OpenMPI rankfile."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    b = OpenMpiBackend("/usr/bin/mpirun")
    p = Placement(
        slices=(HostSlice("h1", (0, 1)), HostSlice("h2", (4, 5, 6))),
        n_ranks=5,
    )
    spec = b.realize(p, ["lmp"], tmp_path)
    assert spec.argv[0] == "/usr/bin/mpirun"
    assert "--rankfile" in spec.argv
    assert "-np" in spec.argv
    assert "5" in spec.argv
    # Verify rankfile content
    argv = list(spec.argv)
    idx_rf = argv.index("--rankfile")
    rankfile_path = argv[idx_rf + 1]
    content = Path(rankfile_path).read_text()
    assert "rank 0=h1 slot=0" in content
    assert "rank 1=h1 slot=1" in content
    assert "rank 2=h2 slot=4" in content
    assert "rank 3=h2 slot=5" in content
    assert "rank 4=h2 slot=6" in content


# ---------------------------------------------------------------------------
# MpichBackend
# ---------------------------------------------------------------------------

def test_mpich_backend_single(tmp_path):
    b = MpichBackend("/usr/bin/mpirun")
    p = Placement.from_host_cores("localhost", tuple(range(4)))
    spec = b.realize(p, ["lmp"], tmp_path)
    assert "-launcher" in spec.argv
    assert "fork" in spec.argv
    assert "-bind-to" in spec.argv
    assert "user:0,1,2,3" in spec.argv


def test_mpich_backend_multi(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    b = MpichBackend("/usr/bin/mpirun")
    p = Placement(
        slices=(HostSlice("h1", (0, 1)), HostSlice("h2", (0, 1, 2))),
        n_ranks=5,
    )
    spec = b.realize(p, ["lmp"], tmp_path)
    # Multi-host uses srun --mpi=pmi2 + SLURM_HOSTFILE
    assert "--mpi=pmi2" in spec.argv
    assert "--distribution=arbitrary" in spec.argv
    assert "-n" in spec.argv
    assert "5" in spec.argv
    assert "lmp" in spec.argv
    # Verify SLURM_HOSTFILE has heterogeneous per-node rank counts
    hostfile_path = spec.env_add["SLURM_HOSTFILE"]
    content = Path(hostfile_path).read_text()
    assert content.count("h1") == 2
    assert content.count("h2") == 3
    # Verify per-rank CPU binding from all slices (global cpu list)
    assert "--cpu-bind=map_cpu:0,1,0,1,2" in spec.argv


def test_mpich_backend_multi_nonzero_cpuids(tmp_path, monkeypatch):
    """map_cpu list reflects actual cpu_ids, not 0..N-1."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    b = MpichBackend("/usr/bin/mpirun")
    p = Placement(
        slices=(HostSlice("h1", (32, 33, 34, 35)),),
        n_ranks=4,
    )
    spec = b.realize(p, ["lmp"], tmp_path)
    assert "--cpu-bind=map_cpu:32,33,34,35" in spec.argv


def test_mpich_backend_ssh(tmp_path, monkeypatch):
    """No SLURM → SSH mode with MPICH Hydra SSH launcher."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    b = MpichBackend("/usr/bin/mpirun")
    p = Placement(
        slices=(HostSlice("h1", (0, 1)), HostSlice("h2", (4, 5, 6))),
        n_ranks=5,
    )
    spec = b.realize(p, ["lmp"], tmp_path)
    assert spec.argv[0] == "/usr/bin/mpirun"
    assert "-launcher" in spec.argv
    assert "ssh" in spec.argv
    assert "-f" in spec.argv
    assert "-np" in spec.argv
    assert "5" in spec.argv
    # Verify hostfile content (Hydra machinefile: hostname:nprocs)
    argv = list(spec.argv)
    idx_f = argv.index("-f")
    hostfile_path = argv[idx_f + 1]
    content = Path(hostfile_path).read_text()
    assert "h1:2" in content
    assert "h2:3" in content
    # CPU binding: global list matching rank order
    assert "-bind-to" in spec.argv
    assert "user:0,1,4,5,6" in spec.argv


def test_intel_backend_multi_cpubind_heterogeneous(tmp_path, monkeypatch):
    """SLURM MPMD uses per-segment CPU pinning for heterogeneous nodes."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    b = IntelMpiBackend("/usr/bin/mpirun")
    b._libpmi2_path = None
    # Node A has 32 cores (using 32), node B has 31 total (using 16)
    p = Placement(
        slices=(
            HostSlice("A", tuple(range(32)), host_n_cpus=32),
            HostSlice("B", tuple(range(16)), host_n_cpus=31),
        ),
        n_ranks=48,
    )
    spec = b.realize(p, ["lmp"], tmp_path)
    # SLURM bootstrap with per-segment CPU pinning
    assert spec.argv[0] == "/usr/bin/mpirun"
    assert "-bootstrap" in spec.argv
    assert "slurm" in spec.argv
    argv = list(spec.argv)
    assert ":" in argv
    # Each segment has I_MPI_PIN_PROCESSOR_LIST with exact CPU list
    pin_indices = [i for i, x in enumerate(argv) if x == "I_MPI_PIN_PROCESSOR_LIST"]
    assert len(pin_indices) == 2  # one per host
    # First segment (A, sorted): 32 CPUs
    pin_val_a = argv[pin_indices[0] + 1]
    assert len(pin_val_a.split(",")) == 32
    # Second segment (B): 16 CPUs
    pin_val_b = argv[pin_indices[1] + 1]
    assert len(pin_val_b.split(",")) == 16


def test_intel_backend_mpmd_sorted_by_hostname(tmp_path, monkeypatch):
    """MPMD segments must be sorted by hostname for SLURM distribution."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    b = IntelMpiBackend("/usr/bin/mpirun")
    b._libpmi2_path = None
    # Placement given in DESCENDING CPU order (not hostname sorted)
    p = Placement(
        slices=(
            HostSlice("node-09", tuple(range(9))),
            HostSlice("node-08", tuple(range(8))),
            HostSlice("node-04", tuple(range(4))),
        ),
        n_ranks=21,
    )
    spec = b.realize(p, ["lmp"], tmp_path)
    argv = list(spec.argv)
    # SLURM bootstrap with hostname-sorted segments
    assert "slurm" in argv
    assert ":" in argv
    # Check that hosts appear in sorted order
    host_indices = [i for i, x in enumerate(argv) if x == "-host"]
    assert len(host_indices) == 3
    hosts = [argv[i + 1] for i in host_indices]
    assert hosts == ["node-04", "node-08", "node-09"]  # sorted


def test_mpich_backend_multi_heterogeneous_cpubind(tmp_path, monkeypatch):
    """srun path uses per-rank map_cpu for heterogeneous nodes."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    b = MpichBackend("/usr/bin/mpirun")
    p = Placement(
        slices=(
            HostSlice("A", tuple(range(32)), host_n_cpus=32),
            HostSlice("B", tuple(range(16)), host_n_cpus=31),
        ),
        n_ranks=48,
    )
    spec = b.realize(p, ["lmp"], tmp_path)
    assert "--mpi=pmi2" in spec.argv
    expected_cpus = ",".join(str(c) for c in range(32)) + "," + ",".join(str(c) for c in range(16))
    assert f"--cpu-bind=map_cpu:{expected_cpus}" in spec.argv


# ---------------------------------------------------------------------------
# ResourcePool construction
# ---------------------------------------------------------------------------

def test_resource_pool_direct_construction():
    backend = LocalMpirunBackend("mpirun", mpi_family="openmpi")
    pool = ResourcePool(
        hosts=[HostInventory("localhost", tuple(range(8)))],
        sim_cmd=["lmp"],
        backend=backend,
    )
    assert len(pool.hosts) == 1
    assert pool.backend is backend
    assert "localhost" in repr(pool)


def test_resource_pool_build_lease_pool():
    backend = LocalMpirunBackend("mpirun")
    pool = ResourcePool(
        hosts=[
            HostInventory("h1", tuple(range(8))),
            HostInventory("h2", tuple(range(8))),
        ],
        sim_cmd=["lmp"],
        backend=backend,
    )
    lp = pool.build_lease_pool()
    assert lp.free_total() == 16


def test_resource_pool_discover_warns_loudly_when_mpirun_missing_on_remote_hosts(monkeypatch):
    monkeypatch.setattr(
        "AceCG.schedulers.resource_pool.socket.gethostname",
        lambda: "localhost",
    )
    monkeypatch.setattr(
        "AceCG.schedulers.resource_pool._discover_hosts",
        lambda: [
            HostInventory("localhost", (0, 1, 2, 3)),
            HostInventory("remote-a", (0, 1, 2, 3)),
        ],
    )
    monkeypatch.setattr(
        "AceCG.schedulers.resource_pool.shutil.which",
        lambda exe: None,
    )
    monkeypatch.setattr(
        "AceCG.schedulers.resource_pool.pick_backend",
        lambda mpirun_path, **kwargs: LocalMpirunBackend(mpirun_path),
    )

    with pytest.warns(RuntimeWarning, match="Dropping 1 remote host"):
        pool = ResourcePool.discover(sim_cmd=["lmp"])

    assert isinstance(pool.backend, LocalMpirunBackend)
    assert [h.hostname for h in pool.hosts] == ["localhost"]


def test_resource_pool_discover_passes_mpi_family_override(monkeypatch):
    monkeypatch.setattr(
        "AceCG.schedulers.resource_pool._discover_hosts",
        lambda: [HostInventory("localhost", (0, 1, 2, 3))],
    )
    monkeypatch.setattr(
        "AceCG.schedulers.resource_pool.shutil.which",
        lambda exe: "/usr/bin/mpirun" if exe in {"mpirun", "mpiexec"} else None,
    )

    captured: dict[str, object] = {}

    def _fake_pick_backend(mpirun_path, *, intel_launch_mode="mpmd", mpi_family=None):
        captured["mpirun_path"] = mpirun_path
        captured["intel_launch_mode"] = intel_launch_mode
        captured["mpi_family"] = mpi_family
        return LocalMpirunBackend(mpirun_path, mpi_family="openmpi")

    monkeypatch.setattr("AceCG.schedulers.resource_pool.pick_backend", _fake_pick_backend)

    pool = ResourcePool.discover(sim_cmd=["lmp"], mpi_family="openmpi")

    assert isinstance(pool.backend, LocalMpirunBackend)
    assert captured == {
        "mpirun_path": "/usr/bin/mpirun",
        "intel_launch_mode": "mpmd",
        "mpi_family": "openmpi",
    }


# ---------------------------------------------------------------------------
# pick_backend
# ---------------------------------------------------------------------------

def test_pick_backend_fallback_local(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    b = pick_backend("/nonexistent/mpirun", in_slurm=False)
    assert isinstance(b, LocalMpirunBackend)


def test_detect_mpi_family_prefers_intel_sibling_probe(tmp_path):
    mpirun = tmp_path / "mpirun"
    mpirun.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  echo \"HYDRA build details:\"\n"
        "  exit 0\n"
        "fi\n",
        encoding="utf-8",
    )
    os.chmod(mpirun, 0o755)

    hydra = tmp_path / "mpiexec.hydra"
    hydra.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-version\" ] || [ \"$1\" = \"--version\" ]; then\n"
        "  echo \"Intel(R) MPI Library for Linux* OS, Version 2021.9\"\n"
        "  exit 0\n"
        "fi\n",
        encoding="utf-8",
    )
    os.chmod(hydra, 0o755)

    assert detect_mpi_family(str(mpirun)) == "intel"


def test_detect_mpi_family_openmpi_orterun_symlink(tmp_path):
    orterun = tmp_path / "orterun"
    orterun.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  echo \"orterun (OpenRTE) 4.1.2\"\n"
        "  exit 0\n"
        "fi\n",
        encoding="utf-8",
    )
    os.chmod(orterun, 0o755)

    mpirun = tmp_path / "mpirun"
    mpirun.symlink_to(orterun)

    assert detect_mpi_family(str(mpirun)) == "openmpi"


def test_pick_backend_uses_mpi_family_override(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(
        "AceCG.schedulers.mpi_backend.detect_mpi_family",
        lambda _path: (_ for _ in ()).throw(AssertionError("should not autodetect")),
    )

    b = pick_backend("/usr/bin/mpirun", in_slurm=True, mpi_family="openmpi")

    assert isinstance(b, OpenMpiBackend)


def test_pick_backend_rejects_invalid_mpi_family_override():
    with pytest.raises(ValueError, match="Unsupported MPI family override"):
        pick_backend("/usr/bin/mpirun", in_slurm=True, mpi_family="not-a-family")


def test_pick_backend_intel_slurm():
    b = pick_backend("/usr/bin/mpirun", in_slurm=True)
    # Can't guarantee detection without real mpirun, but verify it runs
    assert b is not None


# ---------------------------------------------------------------------------
# Placer — single-host placement
# ---------------------------------------------------------------------------

from AceCG.schedulers.resource_pool import Placer, PlacementResult


def _make_placer(hosts, *, supports_multi_host=False):
    """Build a Placer from a list of (hostname, n_cpus) tuples."""
    invs = [HostInventory(h, tuple(range(n))) for h, n in hosts]
    pool = LeasePool(invs)
    backend = LocalMpirunBackend("mpirun", mpi_family="openmpi")
    # Override supports_multi_host for testing
    backend.supports_multi_host = supports_multi_host
    return Placer(pool, backend=backend)


def test_placer_preferred_fits():
    placer = _make_placer([("h1", 8)])
    pr = placer.place(2, 4)
    assert pr is not None
    assert isinstance(pr, PlacementResult)
    assert pr.placement.n_ranks == 4
    assert pr.placement.single_host
    assert len(pr.leases) == 1
    assert pr.leases[0].n_cpus == 4


def test_placer_shrinks_to_min():
    placer = _make_placer([("h1", 3)])
    pr = placer.place(2, 4)
    assert pr is not None
    assert pr.placement.n_ranks == 3


def test_placer_no_capacity_returns_none():
    placer = _make_placer([("h1", 1)])
    pr = placer.place(2, 4)
    assert pr is None


def test_placer_deterministic():
    p1 = _make_placer([("h1", 8), ("h2", 8)])
    p2 = _make_placer([("h1", 8), ("h2", 8)])
    r1 = p1.place(4, 4)
    r2 = p2.place(4, 4)
    assert r1 is not None and r2 is not None
    assert r1.leases[0].host == r2.leases[0].host
    assert r1.leases[0].cpu_ids == r2.leases[0].cpu_ids


def test_placer_contiguous_preference():
    # Host with gaps: 0,1,4,5,6,7 — should pick contiguous block 4-7
    invs = [HostInventory("h1", (0, 1, 4, 5, 6, 7))]
    pool = LeasePool(invs)
    backend = LocalMpirunBackend("mpirun")
    placer = Placer(pool, backend=backend)
    pr = placer.place(4, 4)
    assert pr is not None
    assert sorted(pr.leases[0].cpu_ids) == [4, 5, 6, 7]


# ---------------------------------------------------------------------------
# Placer — multi-host fallback
# ---------------------------------------------------------------------------

def test_placer_multi_host_disabled_by_default():
    """single_host_only=True (default) prevents multi-host even if possible."""
    placer = _make_placer([("h1", 2), ("h2", 2)], supports_multi_host=True)
    pr = placer.place(3, 4, single_host_only=True)
    assert pr is None


def test_placer_multi_host_when_backend_no_support():
    """Backend without multi-host support keeps tasks pending."""
    placer = _make_placer([("h1", 2), ("h2", 2)], supports_multi_host=False)
    pr = placer.place(3, 4, single_host_only=False)
    assert pr is None


def test_placer_multi_host_packs_across_hosts():
    """Multi-host packing when no single host can satisfy min_cores."""
    placer = _make_placer([("h1", 3), ("h2", 3)], supports_multi_host=True)
    pr = placer.place(4, 6, single_host_only=False)
    assert pr is not None
    assert len(pr.leases) == 2
    assert pr.placement.n_ranks == 6
    assert not pr.placement.single_host
    hosts = {l.host for l in pr.leases}
    assert hosts == {"h1", "h2"}


def test_placer_multi_host_prefers_single():
    """Single-host placement is preferred even when multi-host is allowed."""
    placer = _make_placer([("h1", 8), ("h2", 4)], supports_multi_host=True)
    pr = placer.place(4, 4, single_host_only=False)
    assert pr is not None
    assert pr.placement.single_host
    assert len(pr.leases) == 1


def test_placer_multi_host_insufficient_total():
    """Multi-host fails when total free < min_cores."""
    placer = _make_placer([("h1", 2), ("h2", 1)], supports_multi_host=True)
    pr = placer.place(4, 6, single_host_only=False)
    assert pr is None


def test_placer_multi_host_releases_on_failure():
    """If multi-host can't reach min_cores, leases are released."""
    placer = _make_placer([("h1", 2), ("h2", 1)], supports_multi_host=True)
    initial_free = placer.pool.free_total()
    pr = placer.place(4, 6, single_host_only=False)
    assert pr is None
    assert placer.pool.free_total() == initial_free


def test_placer_multi_host_intel_partial_node(monkeypatch):
    """Intel SSH bootstrap does NOT require whole-node allocation.

    With SSH bootstrap, Intel MPI uses per-core CPU pinning via
    I_MPI_PIN_PROCESSOR_LIST, so sub-node concurrent tasks are safe.
    """
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    invs = [
        HostInventory("h1", tuple(range(4))),
        HostInventory("h2", tuple(range(3))),
        HostInventory("h3", tuple(range(5))),
    ]
    pool = LeasePool(invs)
    backend = IntelMpiBackend("/usr/bin/mpirun")
    backend._libpmi2_path = None
    placer = Placer(pool, backend=backend)
    pr = placer.place(6, 6, single_host_only=False)
    assert pr is not None
    total_leased = sum(l.n_cpus for l in pr.leases)
    assert total_leased >= 6


def test_placer_multi_host_intel_reuses_remaining_nodes(monkeypatch):
    """Two Intel tasks can share a pool without whole-node waste."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    invs = [
        HostInventory("n1", tuple(range(4))),   # 4
        HostInventory("n2", tuple(range(3))),   # 3
        HostInventory("n3", tuple(range(3))),   # 3
        HostInventory("n4", (0,)),              # 1
        HostInventory("n5", (0,)),              # 1 total=12
    ]
    pool = LeasePool(invs)
    backend = IntelMpiBackend("/usr/bin/mpirun")
    backend._libpmi2_path = None
    placer = Placer(pool, backend=backend)
    pr1 = placer.place(6, 6, single_host_only=False)
    assert pr1 is not None
    total1 = sum(l.n_cpus for l in pr1.leases)
    assert total1 >= 6
    pr2 = placer.place(6, 6, single_host_only=False)
    assert pr2 is not None
    total2 = sum(l.n_cpus for l in pr2.leases)
    assert total2 >= 6


def test_placer_multi_host_partial_node_openmpi():
    """Non-Intel backends can take partial nodes (no srun bootstrap)."""
    invs = [
        HostInventory("h1", tuple(range(4))),
        HostInventory("h2", tuple(range(3))),
        HostInventory("h3", tuple(range(5))),
    ]
    pool = LeasePool(invs)
    backend = OpenMpiBackend("/usr/bin/mpirun")
    backend.supports_multi_host = True
    placer = Placer(pool, backend=backend)
    # Request 6 cores
    pr = placer.place(6, 6, single_host_only=False)
    assert pr is not None
    # OpenMPI takes exactly 6 (partial node OK)
    assert pr.placement.n_ranks == 6
