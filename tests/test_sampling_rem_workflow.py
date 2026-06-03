import dataclasses
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from AceCG.configs.models import (
    ACGConfig,
    AARefConfig,
    AARefNoiseConfig,
    ConditioningConfig,
    SamplingConfig,
    SchedulerConfig,
    SystemConfig,
    TrainingConfig,
)
from AceCG.potentials.gaussian import GaussianPotential
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.samplers.base import BaseSampler
from AceCG.schedulers.resource_pool import ResourcePool
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey
from AceCG.workflows.base import BaseWorkflow
from AceCG.workflows.rem import REMWorkflow
from AceCG.workflows.sampling import AAStats, SamplingWorkflow


class _SamplingHarness(SamplingWorkflow):
    def _build_trainer(self):
        return object()

    def run(self):
        return None


class _DummySampler:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.cleaned = False
        self.state_counter = 0
        self.loaded_state = None

    def init_epoch(self, *, iteration_index: int, epoch_dir: Path, n_runs: int = 1):
        assert n_runs == 1
        run_dir = epoch_dir / "run_0000"
        run_dir.mkdir(parents=True, exist_ok=True)
        input_script = run_dir / "in.rem.lmp"
        input_script.write_text("# dummy rem input\n", encoding="utf-8")
        trajectory_path = run_dir / "traj.lammpstrj"
        plan = SimpleNamespace(
            run_id=0,
            run_dir=run_dir,
            input_script_path=input_script,
            trajectory_path=trajectory_path,
            frame_id=None,
            write_data_path=None,
        )
        return SimpleNamespace(replica_plans=[plan])

    def clean_epoch(self, state):
        self.cleaned = True
        self.state_counter += 1

    def state_dict(self):
        return {"state_counter": self.state_counter}

    def load_state_dict(self, state):
        self.loaded_state = dict(state)
        self.state_counter = int(state["state_counter"])


class _CapturingScheduler:
    def __init__(self):
        self.last_xz_tasks = None
        self.last_iter_dir = None
        self.loaded_state = None
        self.state_counter = 0

    def run_iteration(self, xz_tasks, zbx_tasks, *, iter_dir=None):
        assert not zbx_tasks
        self.last_xz_tasks = list(xz_tasks)
        self.last_iter_dir = iter_dir
        self.state_counter += 1
        task = xz_tasks[0]
        with open(task.post_spec["forcefield_path"], "rb") as fh:
            forcefield = pickle.load(fh)
        n_params = forcefield.n_params()
        output_path = Path(task.run_dir) / task.post_spec["steps"][0]["output_file"]
        with open(output_path, "wb") as fh:
            pickle.dump(
                {
                    "energy_grad_avg": np.zeros(n_params, dtype=np.float64),
                    "d2U_avg": np.zeros((n_params, n_params), dtype=np.float64),
                    "grad_outer_avg": np.zeros((n_params, n_params), dtype=np.float64),
                    "gradient_convention": "gauge_free",
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return SimpleNamespace(xz_ok=True)

    def state_dict(self):
        return {"state_counter": self.state_counter}

    def load_state_dict(self, state):
        self.loaded_state = dict(state)
        self.state_counter = int(state["state_counter"])


def _write_rem_script(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "read_data default.data",
                "dump traj all custom 1 traj.lammpstrj id type x y z",
                "run 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path.parent / "default.data").write_text("# default data\n", encoding="utf-8")


def _make_config(
    config_path: Path,
    *,
    optimizer="adam",
    output_dir="results/rem",
    forcefield_path="ff/system.settings",
    sampling_input="scripts/in.rem.lmp",
    init_config_pool=None,
    cutoff=7.5,
    trajectory_files=(),
    all_atom_data_path=None,
    sim_var=None,
    seed=0,
    start_epoch=0,
    n_epochs=1,
    aa_noise=None,
) -> ACGConfig:
    return ACGConfig(
        path=config_path,
        system=SystemConfig(
            topology_file="system/topology.data",
            forcefield_path=forcefield_path,
            pair_style="table",
            cutoff=cutoff,
        ),
        training=TrainingConfig(
            method="rem",
            optimizer=optimizer,
            temperature=300.0,
            n_epochs=n_epochs,
            start_epoch=start_epoch,
            output_dir=output_dir,
            seed=seed,
        ),
        sampling=SamplingConfig(
            input=sampling_input,
            engine_command="lmp",
            init_config_pool=init_config_pool,
            sim_var={} if sim_var is None else dict(sim_var),
        ),
        scheduler=SchedulerConfig(python_exe="python"),
        aa_ref=AARefConfig(
            trajectory_files=tuple(trajectory_files),
            trajectory_format="LAMMPSDUMP",
            all_atom_data_path=all_atom_data_path,
            noise=AARefNoiseConfig() if aa_noise is None else aa_noise,
        ),
        conditioning=ConditioningConfig(),
    )


def _linear_forcefield() -> Forcefield:
    forcefield = Forcefield(
        {
            InteractionKey(style="bond", types=("1",)): [
                HarmonicPotential("A", "B", k=2.0, r0=1.25)
            ]
        }
    )
    forcefield.build_mask(init_mask=np.array([True, False], dtype=bool))
    return forcefield


def _nonlinear_forcefield() -> Forcefield:
    return Forcefield(
        {
            InteractionKey.pair("A", "B"): [
                GaussianPotential("A", "B", A=1.5, r0=1.0, sigma=0.4, cutoff=5.0)
            ]
        }
    )


def _patch_workflow_basics(monkeypatch, forcefield: Forcefield, *, scheduler=None, sampler=None):
    monkeypatch.setattr(BaseWorkflow, "_build_topology", lambda self: {})
    monkeypatch.setattr(SamplingWorkflow, "_build_forcefield", lambda self: Forcefield(forcefield))
    monkeypatch.setattr(
        BaseWorkflow,
        "_build_resource_pool",
        lambda self, sim_cmd=None: SimpleNamespace(
            hosts=[SimpleNamespace(n_cpus=4)],
            mpirun_path="mpirun",
        ),
    )
    monkeypatch.setattr(
        SamplingWorkflow,
        "_build_scheduler",
        lambda self: scheduler if scheduler is not None else SimpleNamespace(),
    )
    monkeypatch.setattr(
        SamplingWorkflow,
        "_build_sampler",
        lambda self: sampler if sampler is not None else SimpleNamespace(),
    )


def test_build_resource_pool_passes_mpi_backend_options(monkeypatch, tmp_path):
    cfg = _make_config(tmp_path / "test.acg")
    cfg = dataclasses.replace(
        cfg,
        scheduler=SchedulerConfig(
            mpirun_path="/opt/mpi/bin/mpirun",
            mpi_family="intel",
            python_exe="python",
        ),
    )
    dummy = SimpleNamespace(config=cfg)
    sentinel = object()
    seen: dict[str, object] = {}

    def _fake_discover(cls, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(ResourcePool, "discover", classmethod(_fake_discover))

    pool = BaseWorkflow._build_resource_pool(dummy)

    assert pool is sentinel
    assert seen["mpirun_path"] == "/opt/mpi/bin/mpirun"
    assert seen["mpi_family"] == "intel"


def test_sampling_workflow_resolves_relative_input_pool_and_float_cutoff(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "test.acg"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("# config\n", encoding="utf-8")

    script_path = config_path.parent / "scripts" / "in.rem.lmp"
    _write_rem_script(script_path)
    init_pool_dir = config_path.parent / "pool"
    init_pool_dir.mkdir(parents=True, exist_ok=True)
    (init_pool_dir / "frame0.data").write_text("# pool frame\n", encoding="utf-8")

    captured = {}

    def _fake_read_lmpff(path, pair_style, pair_typ_sel=None, cutoff=None, table_fit=None, **kwargs):
        captured["path"] = path
        captured["cutoff"] = cutoff
        return _linear_forcefield()

    monkeypatch.setattr(BaseWorkflow, "_build_topology", lambda self: {})
    monkeypatch.setattr(
        BaseWorkflow,
        "_build_resource_pool",
        lambda self, sim_cmd=None: SimpleNamespace(hosts=[SimpleNamespace(n_cpus=4)]),
    )
    monkeypatch.setattr(SamplingWorkflow, "_build_scheduler", lambda self: SimpleNamespace())
    monkeypatch.setattr("AceCG.workflows.sampling.ReadLmpFF", _fake_read_lmpff)

    workflow = _SamplingHarness(
        _make_config(
            config_path,
            init_config_pool="pool/*.data",
            trajectory_files=("aa/reference.lammpstrj",),
        )
    )

    assert Path(captured["path"]) == (config_path.parent / "ff/system.settings").resolve()
    assert captured["cutoff"] == pytest.approx(7.5)
    assert workflow.output_dir == (config_path.parent / "results/rem").resolve()
    assert workflow.sampler._sim_input == script_path.resolve()
    assert workflow.sampler._init_pool is not None
    assert [record.path for record in workflow.sampler._init_pool] == [
        (init_pool_dir / "frame0.data").resolve()
    ]


def test_rem_builds_constant_aa_stats_from_relative_cache(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "test.acg"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("# config\n", encoding="utf-8")

    cache_path = config_path.parent / "cache" / "aa.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(
            {
                "energy_grad_AA": np.array([1.0, 2.0]),
                "gradient_convention": "gauge_free",
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    _patch_workflow_basics(monkeypatch, _linear_forcefield())

    workflow = REMWorkflow(
        _make_config(
            config_path,
            optimizer="adam",
            all_atom_data_path="cache/aa.pkl",
        )
    )

    assert isinstance(workflow.aa_data_strategy, AAStats)
    assert np.allclose(workflow.aa_data_strategy.energy_grad, np.array([1.0, 2.0]))
    assert workflow.aa_data_strategy.d2U is None
    assert workflow.aa_data_strategy.gradient_convention == "gauge_free"


def test_rem_rejects_cached_aa_stats_without_gauge_free_convention(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "test.acg"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("# config\n", encoding="utf-8")

    cache_path = config_path.parent / "cache" / "aa.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump({"energy_grad_AA": np.array([1.0, 2.0])}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    _patch_workflow_basics(monkeypatch, _linear_forcefield())

    with pytest.raises(KeyError, match="gradient_convention"):
        REMWorkflow(
            _make_config(
                config_path,
                optimizer="adam",
                all_atom_data_path="cache/aa.pkl",
            )
        )


def test_rem_rejects_cached_physical_aa_stats(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "test.acg"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("# config\n", encoding="utf-8")

    cache_path = config_path.parent / "cache" / "aa.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(
            {
                "energy_grad_AA": np.array([1.0, 2.0]),
                "gradient_convention": "physical",
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    _patch_workflow_basics(monkeypatch, _linear_forcefield())

    with pytest.raises(ValueError, match="expected 'gauge_free'"):
        REMWorkflow(
            _make_config(
                config_path,
                optimizer="adam",
                all_atom_data_path="cache/aa.pkl",
            )
        )


def test_rem_builds_dynamic_aa_strategy_from_resolved_trajectories(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "test.acg"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("# config\n", encoding="utf-8")

    captured_specs = []

    def _fake_run_post(spec, resource_pool, *, run_dir=None, python_exe=None):
        captured_specs.append(dict(spec))
        output_path = Path(spec["steps"][0]["output_file"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as fh:
            pickle.dump(
                {
                    "energy_grad_avg": np.array([3.0, 4.0, 5.0]),
                    "gradient_convention": "gauge_free",
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    _patch_workflow_basics(monkeypatch, _nonlinear_forcefield())
    monkeypatch.setattr("AceCG.workflows.sampling.run_post", _fake_run_post)

    workflow = REMWorkflow(
        _make_config(
            config_path,
            optimizer="adam",
            trajectory_files=("aa/reference.lammpstrj",),
            all_atom_data_path="cache/ignored.pkl",
        )
    )

    assert callable(workflow.aa_data_strategy)
    stats = workflow.aa_data_strategy(workflow.forcefield)

    assert np.allclose(stats.energy_grad, np.array([3.0, 4.0, 5.0]))
    assert stats.gradient_convention == "gauge_free"
    assert captured_specs
    assert captured_specs[0]["steps"][0]["step_mode"] == "rem"
    assert captured_specs[0]["trajectory"] == [
        str((config_path.parent / "aa/reference.lammpstrj").resolve())
    ]


def test_rem_noisy_aa_strategy_forwards_noise_spec(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "test.acg"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("# config\n", encoding="utf-8")

    captured_specs = []

    def _fake_run_post(spec, resource_pool, *, run_dir=None, python_exe=None):
        captured_specs.append(dict(spec))
        output_path = Path(spec["steps"][0]["output_file"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as fh:
            pickle.dump(
                {
                    "energy_grad_avg": np.array([3.0, 4.0, 5.0]),
                    "gradient_convention": "gauge_free",
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    noise = AARefNoiseConfig(
        enabled=True,
        samples_per_frame=2,
        sigma=0.10,
        sigma_final=0.02,
        schedule="cosine",
        update_interval=2,
        seed=11,
        include_original=True,
        batch_size=4,
        subsample_per_epoch=640,
    )
    _patch_workflow_basics(monkeypatch, _nonlinear_forcefield())
    monkeypatch.setattr("AceCG.workflows.sampling.run_post", _fake_run_post)

    workflow = REMWorkflow(
        _make_config(
            config_path,
            optimizer="adam",
            trajectory_files=("aa/reference.lammpstrj",),
            aa_noise=noise,
            n_epochs=4,
        )
    )
    stats = workflow.aa_data_strategy(workflow.forcefield, epoch=3)

    assert np.allclose(stats.energy_grad, np.array([3.0, 4.0, 5.0]))
    assert stats.gradient_convention == "gauge_free"
    assert captured_specs
    runtime_noise = captured_specs[0]["noise"]
    assert runtime_noise["samples_per_frame"] == 2
    assert runtime_noise["subsample_per_epoch"] == 640
    assert runtime_noise["subsample_seed"] == 14
    assert runtime_noise["sigma"] == pytest.approx(0.02)
    assert runtime_noise["seed"] == 12
    assert runtime_noise["include_original"] is True
    assert runtime_noise["batch_size"] == 4
    assert "beta" not in runtime_noise


def test_rem_requires_hessian_cache_when_optimizer_is_newton(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "test.acg"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("# config\n", encoding="utf-8")

    cache_path = config_path.parent / "cache" / "aa.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(
            {
                "energy_grad_AA": np.array([1.0, 2.0]),
                "gradient_convention": "gauge_free",
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    _patch_workflow_basics(monkeypatch, _linear_forcefield())

    with pytest.raises(KeyError, match="d2U_AA"):
        REMWorkflow(
            _make_config(
                config_path,
                optimizer=None,
                all_atom_data_path="cache/aa.pkl",
            )
        )


def test_rem_run_builds_canonical_xz_taskspec_and_sets_need_hessian(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "test.acg"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("# config\n", encoding="utf-8")

    cache_path = config_path.parent / "cache" / "aa.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(
            {
                "energy_grad_AA": np.array([0.0, 0.0]),
                "d2U_AA": np.eye(2, dtype=np.float64),
                "gradient_convention": "gauge_free",
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    scheduler = _CapturingScheduler()
    sampler = _DummySampler(tmp_path)
    _patch_workflow_basics(monkeypatch, _linear_forcefield(), scheduler=scheduler, sampler=sampler)
    def _fake_write_forcefield(self, ff_dir):
        ff_dir.mkdir(parents=True, exist_ok=True)
        target = ff_dir / "runtime.settings"
        target.write_text("# ff\n", encoding="utf-8")
        return target
    monkeypatch.setattr(
        SamplingWorkflow,
        "_write_forcefield",
        _fake_write_forcefield,
    )
    monkeypatch.setattr(
        "AceCG.workflows.sampling.run_post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Unexpected direct workflow run_post call")),
    )

    workflow = REMWorkflow(
        _make_config(
            config_path,
            optimizer=None,
            all_atom_data_path="cache/aa.pkl",
        )
    )
    result = workflow.run()

    assert result["epochs"] == 1
    task = scheduler.last_xz_tasks[0]
    assert Path(task.run_dir).is_absolute()
    assert task.sim_backend == "lammps"
    assert task.post_exec == {"mode": "mpi"}
    assert task.trajectory_files == ["traj.lammpstrj"]
    assert "trajectory_format" not in task.post_spec
    assert "post_mode" not in task.post_spec
    assert "perf_trace" not in task.post_spec
    assert len(task.post_spec["steps"]) == 1
    assert task.post_spec["steps"][0]["step_mode"] == "rem"
    assert task.post_spec["steps"][0]["need_hessian"] is True
    assert task.post_spec["steps"][0]["output_file"] == "result.pkl"
    assert sampler.cleaned is True


def test_rem_run_forwards_perf_trace_to_cg_post_spec(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "test.acg"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("# config\n", encoding="utf-8")

    cache_path = config_path.parent / "cache" / "aa.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(
            {
                "energy_grad_AA": np.array([0.0, 0.0]),
                "d2U_AA": np.eye(2, dtype=np.float64),
                "gradient_convention": "gauge_free",
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    scheduler = _CapturingScheduler()
    sampler = _DummySampler(tmp_path)
    _patch_workflow_basics(monkeypatch, _linear_forcefield(), scheduler=scheduler, sampler=sampler)

    def _fake_write_forcefield(self, ff_dir):
        ff_dir.mkdir(parents=True, exist_ok=True)
        target = ff_dir / "runtime.settings"
        target.write_text("# ff\n", encoding="utf-8")
        return target

    monkeypatch.setattr(
        SamplingWorkflow,
        "_write_forcefield",
        _fake_write_forcefield,
    )
    cfg = _make_config(
        config_path,
        optimizer=None,
        all_atom_data_path="cache/aa.pkl",
    )
    cfg = dataclasses.replace(
        cfg,
        sampling=dataclasses.replace(
            cfg.sampling,
            perf_trace=True,
            extras={"heartbeat_interval": 7, "perf_trace_all_ranks": True},
        ),
    )

    workflow = REMWorkflow(cfg)
    workflow.run()

    task = scheduler.last_xz_tasks[0]
    assert task.post_spec["perf_trace"] is True
    assert task.post_spec["heartbeat_interval"] == 7
    assert task.post_spec["perf_trace_all_ranks"] is True


def test_rem_resume_restores_sampler_state_and_workflow_rng(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "test.acg"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("# config\n", encoding="utf-8")

    cache_path = config_path.parent / "cache" / "aa.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(
            {
                "energy_grad_AA": np.array([0.0, 0.0]),
                "d2U_AA": np.eye(2, dtype=np.float64),
                "gradient_convention": "gauge_free",
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    seed = 17

    scheduler_1 = _CapturingScheduler()
    sampler_1 = _DummySampler(tmp_path)
    _patch_workflow_basics(monkeypatch, _linear_forcefield(), scheduler=scheduler_1, sampler=sampler_1)

    def _fake_write_forcefield(self, ff_dir):
        ff_dir.mkdir(parents=True, exist_ok=True)
        target = ff_dir / "runtime.settings"
        target.write_text("# ff\n", encoding="utf-8")
        return target

    monkeypatch.setattr(SamplingWorkflow, "_write_forcefield", _fake_write_forcefield)
    monkeypatch.setattr(
        "AceCG.workflows.sampling.run_post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Unexpected direct workflow run_post call")),
    )

    workflow_1 = REMWorkflow(
        _make_config(
            config_path,
            optimizer=None,
            all_atom_data_path="cache/aa.pkl",
            sim_var={"SEED": "{RANDOM}"},
            seed=seed,
            n_epochs=1,
        )
    )
    result_1 = workflow_1.run()

    assert result_1["epochs"] == 1
    assert scheduler_1.last_xz_tasks[0].sim_var["SEED"] == "{RANDOM}"
    checkpoint_path = (
        config_path.parent
        / "results/rem"
        / "iter_0000"
        / "ff"
        / "workflow_checkpoint.pkl"
    )
    assert checkpoint_path.exists()

    scheduler_2 = _CapturingScheduler()
    sampler_2 = _DummySampler(tmp_path)
    _patch_workflow_basics(monkeypatch, _linear_forcefield(), scheduler=scheduler_2, sampler=sampler_2)
    monkeypatch.setattr(SamplingWorkflow, "_write_forcefield", _fake_write_forcefield)
    monkeypatch.setattr(
        "AceCG.workflows.sampling.run_post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Unexpected direct workflow run_post call")),
    )

    workflow_2 = REMWorkflow(
        _make_config(
            config_path,
            optimizer=None,
            all_atom_data_path="cache/aa.pkl",
            sim_var={"SEED": "{RANDOM}"},
            seed=seed,
            start_epoch=1,
            n_epochs=2,
        )
    )
    result_2 = workflow_2.run()

    assert result_2["epochs"] == 1
    assert sampler_2.loaded_state == {"state_counter": 1}
    assert scheduler_2.loaded_state == {"state_counter": 1}
    assert scheduler_2.last_xz_tasks[0].sim_var["SEED"] == "{RANDOM}"
