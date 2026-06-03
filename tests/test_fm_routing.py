"""Tests for FM linearity detection and current workflow routing."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from AceCG.configs.models import (
    ACGConfig,
    AARefConfig,
    AARefNoiseConfig,
    ConditioningConfig,
    FMInteractionSpec,
    FMTrainingSpecs,
    SamplingConfig,
    SchedulerConfig,
    SystemConfig,
    TrainingConfig,
)
from AceCG.potentials.bspline import BSplinePotential
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey
from AceCG.trainers.base import BaseTrainer
from AceCG.workflows.dsm import DSMWorkflow
from AceCG.workflows.fm import FMWorkflow, _build_zero_potential


class DummyOptimizer:
    def __init__(self, params, mask=None):
        self.L = np.asarray(params, dtype=np.float64).copy()
        self.mask = (
            np.ones_like(self.L, dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool).copy()
        )

    def step(self, grad, hessian=None):
        return np.zeros_like(self.L)

    def set_params(self, params):
        self.L = np.asarray(params, dtype=np.float64).copy()


class DummyTrainer(BaseTrainer):
    def step(self, batch, apply_update=True):
        return {"name": "dummy", "grad": np.zeros_like(self.get_params())}


def _linear_forcefield():
    key = InteractionKey.pair("A", "B")
    pot = BSplinePotential(
        "A",
        "B",
        knots=np.array([0.0, 0.0, 2.0, 2.0], dtype=float),
        coefficients=np.array([0.0, 0.0], dtype=float),
        degree=1,
        cutoff=2.0,
    )
    return Forcefield({key: [pot]})


def _harmonic_forcefield():
    key = InteractionKey.bond("A", "B")
    pot = HarmonicPotential("A", "B", k=1.0, r0=0.5)
    return Forcefield({key: [pot]})


def _build_trainer(forcefield: Forcefield, *, param_mask=None):
    params = forcefield.param_array()
    optimizer = DummyOptimizer(params, mask=param_mask)
    return DummyTrainer(forcefield, optimizer, beta=None, logger=None)


def test_is_optimization_linear_true_for_linear_potential():
    trainer = _build_trainer(_linear_forcefield())
    assert trainer.is_optimization_linear() is True


def test_is_optimization_linear_false_for_active_nonlinear_parameter():
    trainer = _build_trainer(_harmonic_forcefield())
    assert trainer.is_optimization_linear() is False


def test_is_optimization_linear_ignores_masked_nonlinear_parameter():
    trainer = _build_trainer(
        _harmonic_forcefield(),
        param_mask=np.array([True, False], dtype=bool),
    )
    assert trainer.is_optimization_linear() is True


def test_build_zero_potential_uses_minimum_gauge_for_bond_specs():
    spec = FMInteractionSpec(
        style="bond",
        types=("A", "B"),
        model="bspline",
        model_size=4,
        domain=(1.0, 3.0),
        init_mode="authored_zero",
        resolution=0.1,
    )
    pot = _build_zero_potential(spec)
    assert pot.bonded is True


def _minimal_fm_config(
    tmp_path: Path,
    *,
    method="fm",
    fm_method="auto",
    optimizer="newton",
    temperature=None,
    n_epochs=1,
    start_epoch=0,
    aa_noise=None,
    trajectory_files=(),
    sampling=None,
):
    fm_spec = FMInteractionSpec(
        style="pair",
        types=("A", "B"),
        model="bspline",
        model_size=2,
        domain=(0.0, 2.0),
        init_mode="authored_zero",
        resolution=0.1,
    )
    return ACGConfig(
        path=tmp_path / "test.acg",
        system=SystemConfig(
            topology_file="topology.data",
            forcefield_path=None,
            forcefield_mask=None,
            pair_style=None,
            cutoff=None,
            exclude_bonded="111",
            exclude_option="resid",
            type_names=None,
            table_fit=None,
        ),
        training=TrainingConfig(
            method=method,
            para_path=None,
            fm_specs=FMTrainingSpecs(pair_specs=(fm_spec,)),
            fm_method=fm_method,
            solver_mode="ols",
            solver_ridge_alpha=0.0,
            optimizer=optimizer,
            trainer=None,
            lr=None,
            n_epochs=n_epochs,
            start_epoch=start_epoch,
            convergence_tol=0.0,
            output_dir=str(tmp_path / "results"),
            seed=0,
            temperature=temperature,
            need_hessian=False,
        ),
        sampling=SamplingConfig() if sampling is None else sampling,
        scheduler=SchedulerConfig(python_exe="python"),
        aa_ref=AARefConfig(
            trajectory_files=tuple(trajectory_files),
            trajectory_format="LAMMPSDUMP",
            skip_frames=0,
            every=1,
            n_frames=0,
            noise=AARefNoiseConfig() if aa_noise is None else aa_noise,
        ),
        vp=None,
        conditioning=ConditioningConfig(),
    )


def _patch_fmworkflow_basics(monkeypatch, *, forcefield):
    monkeypatch.setattr(FMWorkflow, "_build_topology", lambda self: object())
    monkeypatch.setattr(FMWorkflow, "_build_output_dir", lambda self: Path(self.config.training.output_dir))
    monkeypatch.setattr(
        FMWorkflow,
        "_build_resource_pool",
        lambda self, sim_cmd=None: type("Pool", (), {"hosts": [type("Host", (), {"n_cpus": 1})()]})(),
    )
    monkeypatch.setattr(FMWorkflow, "_build_fm_forcefield", lambda self: forcefield)
    monkeypatch.setattr(FMWorkflow, "_build_optimizer", lambda self, forcefield: "optimizer")
    monkeypatch.setattr(FMWorkflow, "_build_solver", lambda self: "solver")
    monkeypatch.setattr(FMWorkflow, "_build_trainer", lambda self: "trainer")


def test_fmworkflow_auto_routes_newton_to_solver(monkeypatch, tmp_path):
    _patch_fmworkflow_basics(monkeypatch, forcefield=_linear_forcefield())
    workflow = FMWorkflow(_minimal_fm_config(tmp_path, fm_method="auto", optimizer="newton"))
    assert workflow._use_solver is True
    assert workflow.trainer_or_solver == "solver"


def test_fmworkflow_auto_routes_non_newton_to_iterator(monkeypatch, tmp_path):
    _patch_fmworkflow_basics(monkeypatch, forcefield=_linear_forcefield())
    workflow = FMWorkflow(_minimal_fm_config(tmp_path, fm_method="auto", optimizer="adam"))
    assert workflow._use_solver is False
    assert workflow.trainer_or_solver == "trainer"


def test_fmworkflow_explicit_solver_and_iterator_modes(monkeypatch, tmp_path):
    _patch_fmworkflow_basics(monkeypatch, forcefield=_harmonic_forcefield())
    solver_workflow = FMWorkflow(_minimal_fm_config(tmp_path, fm_method="solver", optimizer="adam"))
    iterator_workflow = FMWorkflow(_minimal_fm_config(tmp_path, fm_method="iterator", optimizer="newton"))
    assert solver_workflow._use_solver is True
    assert solver_workflow.trainer_or_solver == "solver"
    assert iterator_workflow._use_solver is False
    assert iterator_workflow.trainer_or_solver == "trainer"


def test_trainable_export_forcefield_drops_frozen_overlay(monkeypatch, tmp_path):
    key = InteractionKey.pair("A", "B")
    trainable = BSplinePotential(
        "A",
        "B",
        knots=np.array([0.0, 0.0, 2.0, 2.0], dtype=float),
        coefficients=np.array([0.0, 0.0], dtype=float),
        degree=1,
        cutoff=2.0,
    )
    frozen = BSplinePotential(
        "A",
        "B",
        knots=np.array([0.0, 0.0, 2.0, 2.0], dtype=float),
        coefficients=np.array([1.0, 1.0], dtype=float),
        degree=1,
        cutoff=2.0,
    )
    frozen.param_mask = np.array([False, False], dtype=bool)
    ff = Forcefield({key: [trainable, frozen]})
    ff.key_mask = {key: True}
    _patch_fmworkflow_basics(monkeypatch, forcefield=ff)

    workflow = FMWorkflow(_minimal_fm_config(tmp_path, fm_method="iterator", optimizer="adam"))
    export_ff = workflow._trainable_export_forcefield()

    assert export_ff.n_params() == trainable.n_params()
    assert len(export_ff[key]) == 1
    np.testing.assert_allclose(export_ff[key][0].get_params(), trainable.get_params())


def test_dsmworkflow_always_uses_iterative_trainer(monkeypatch, tmp_path):
    _patch_fmworkflow_basics(monkeypatch, forcefield=_linear_forcefield())
    workflow = DSMWorkflow(
        _minimal_fm_config(
            tmp_path,
            method="dsm",
            fm_method="auto",
            optimizer="newton",
            temperature=300.0,
            aa_noise=AARefNoiseConfig(
                enabled=True,
                samples_per_frame=2,
                sigma=0.10,
            ),
            trajectory_files=("aa/reference.lammpstrj",),
        )
    )

    assert workflow._use_solver is False
    assert workflow.trainer_or_solver == "trainer"


def test_dsmworkflow_noise_runtime_spec_carries_beta_and_schedule(monkeypatch, tmp_path):
    _patch_fmworkflow_basics(monkeypatch, forcefield=_linear_forcefield())
    noise = AARefNoiseConfig(
        enabled=True,
        samples_per_frame=3,
        sigma=0.10,
        sigma_final=0.04,
        schedule="cosine",
        update_interval=2,
        seed=19,
        include_original=True,
        batch_size=2,
        subsample_per_epoch=12,
    )
    workflow = DSMWorkflow(
        _minimal_fm_config(
            tmp_path,
            method="dsm",
            temperature=300.0,
            n_epochs=4,
            aa_noise=noise,
            trajectory_files=("aa/reference.lammpstrj",),
        )
    )

    spec = workflow._dsm_noise_runtime_spec(step_index=3)

    assert spec["target"] == "dsm"
    assert spec["samples_per_frame"] == 3
    assert spec["sigma"] == pytest.approx(0.04)
    assert spec["seed"] == 20
    assert spec["subsample_seed"] == 22
    assert spec["beta"] == pytest.approx(1.0 / (0.001987204 * 300.0))
    assert spec["batch_size"] == 2


def test_dsmworkflow_scales_fm_statistics_by_beta_squared(monkeypatch, tmp_path):
    _patch_fmworkflow_basics(monkeypatch, forcefield=_linear_forcefield())
    workflow = DSMWorkflow(
        _minimal_fm_config(
            tmp_path,
            method="dsm",
            temperature=300.0,
            aa_noise=AARefNoiseConfig(
                enabled=True,
                samples_per_frame=1,
                sigma=0.10,
            ),
            trajectory_files=("aa/reference.lammpstrj",),
        )
    )
    batch = {
        "JtJ": np.eye(2),
        "Jty": np.ones(2),
        "Jtf": np.full(2, 2.0),
        "y_sumsq": 3.0,
        "f_sumsq": 4.0,
        "fty": 5.0,
        "nframe": 1,
    }

    workflow._scale_dsm_batch(batch)

    scale = workflow.beta * workflow.beta
    np.testing.assert_allclose(batch["JtJ"], np.eye(2) * scale)
    np.testing.assert_allclose(batch["Jty"], np.ones(2) * scale)
    np.testing.assert_allclose(batch["Jtf"], np.full(2, 2.0) * scale)
    assert batch["y_sumsq"] == pytest.approx(3.0 * scale)
    assert batch["f_sumsq"] == pytest.approx(4.0 * scale)
    assert batch["fty"] == pytest.approx(5.0 * scale)
    assert batch["nframe"] == 1


def test_dsm_resume_restores_forcefield_and_optimizer_state(monkeypatch, tmp_path):
    monkeypatch.setattr(FMWorkflow, "_build_topology", lambda self: object())
    monkeypatch.setattr(
        FMWorkflow,
        "_build_resource_pool",
        lambda self, sim_cmd=None: type("Pool", (), {"hosts": [type("Host", (), {"n_cpus": 1})()]})(),
    )
    monkeypatch.setattr(FMWorkflow, "_build_fm_forcefield", lambda self: _linear_forcefield())
    seen_optimizer_steps = []

    def _fake_run_post_accumulation(self, *, step_index=0):
        seen_optimizer_steps.append(int(self.trainer_or_solver.optimizer.t))
        return {
            "JtJ": np.eye(2, dtype=np.float64),
            "Jty": np.ones(2, dtype=np.float64),
            "Jtf": np.zeros(2, dtype=np.float64),
            "y_sumsq": 1.0,
            "f_sumsq": 0.0,
            "fty": 0.0,
            "nframe": 1,
        }

    monkeypatch.setattr(
        DSMWorkflow,
        "_run_post_accumulation",
        _fake_run_post_accumulation,
    )
    noise = AARefNoiseConfig(
        enabled=True,
        samples_per_frame=1,
        sigma=0.10,
    )

    workflow_1 = DSMWorkflow(
        _minimal_fm_config(
            tmp_path,
            method="dsm",
            optimizer="adam",
            temperature=300.0,
            n_epochs=1,
            aa_noise=noise,
            trajectory_files=("aa/reference.lammpstrj",),
        )
    )
    result_1 = workflow_1.run()

    assert result_1["epochs"] == 1
    checkpoint_path = (
        tmp_path
        / "results"
        / "dsm_step_0000"
        / "ff"
        / "workflow_checkpoint.pkl"
    )
    assert checkpoint_path.exists()
    with checkpoint_path.open("rb") as fh:
        payload = pickle.load(fh)
    assert payload["optimizer_state"]["t"] == 1
    np.testing.assert_allclose(
        payload["forcefield"].param_array(),
        workflow_1.forcefield.param_array(),
    )

    workflow_2 = DSMWorkflow(
        _minimal_fm_config(
            tmp_path,
            method="dsm",
            optimizer="adam",
            temperature=300.0,
            n_epochs=2,
            start_epoch=1,
            aa_noise=noise,
            trajectory_files=("aa/reference.lammpstrj",),
        )
    )
    result_2 = workflow_2.run()

    assert result_2["epochs"] == 1
    assert seen_optimizer_steps == [0, 1]
    assert (tmp_path / "results" / "dsm_step_0001" / "ff" / "workflow_checkpoint.pkl").exists()


def test_fmworkflow_noise_runtime_spec_resolves_stage_without_beta(monkeypatch, tmp_path):
    _patch_fmworkflow_basics(monkeypatch, forcefield=_linear_forcefield())
    noise = AARefNoiseConfig(
        enabled=True,
        samples_per_frame=3,
        sigma=0.10,
        sigma_final=0.04,
        schedule="cosine",
        update_interval=2,
        seed=19,
        include_original=True,
        batch_size=2,
        subsample_per_epoch=12,
    )
    workflow = FMWorkflow(
        _minimal_fm_config(
            tmp_path,
            temperature=300.0,
            n_epochs=4,
            aa_noise=noise,
            trajectory_files=("aa/reference.lammpstrj",),
        )
    )

    spec = workflow._fm_noise_runtime_spec(step_index=3)

    assert spec is not None
    assert spec["samples_per_frame"] == 3
    assert spec["sigma"] == pytest.approx(0.04)
    assert spec["seed"] == 20
    assert spec["subsample_seed"] == 22
    assert "beta" not in spec
    assert spec["batch_size"] == 2
    assert spec["subsample_per_epoch"] == 12
    assert "sigma_final" not in spec
    assert "schedule" not in spec


def test_fmworkflow_noise_runtime_spec_carries_beta_for_force_mix(monkeypatch, tmp_path):
    _patch_fmworkflow_basics(monkeypatch, forcefield=_linear_forcefield())
    noise = AARefNoiseConfig(
        enabled=True,
        samples_per_frame=3,
        sigma=0.10,
        seed=19,
        force_mix_ratio=0.25,
    )
    workflow = FMWorkflow(
        _minimal_fm_config(
            tmp_path,
            temperature=300.0,
            aa_noise=noise,
            trajectory_files=("aa/reference.lammpstrj",),
        )
    )

    spec = workflow._fm_noise_runtime_spec(step_index=0)

    assert spec is not None
    assert spec["force_mix_ratio"] == pytest.approx(0.25)
    assert spec["beta"] == pytest.approx(1.0 / (0.001987204 * 300.0))


def test_fmworkflow_run_post_accumulation_forwards_noise_spec(monkeypatch, tmp_path):
    _patch_fmworkflow_basics(monkeypatch, forcefield=_linear_forcefield())
    captured_specs = []

    def _fake_run_post(spec, resource_pool, *, run_dir=None, python_exe=None):
        captured_specs.append(dict(spec))
        output_path = Path(spec["steps"][0]["output_file"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as fh:
            import pickle

            pickle.dump(
                {
                    "JtJ": np.eye(2, dtype=np.float64),
                    "Jty": np.zeros(2, dtype=np.float64),
                    "y_sumsq": 0.0,
                    "nframe": 1,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    monkeypatch.setattr("AceCG.workflows.fm.run_post", _fake_run_post)
    noise = AARefNoiseConfig(
        enabled=True,
        samples_per_frame=2,
        sigma=0.10,
        sigma_final=0.04,
        schedule="cosine",
        update_interval=2,
        seed=19,
        include_original=True,
        batch_size=2,
    )
    workflow = FMWorkflow(
        _minimal_fm_config(
            tmp_path,
            n_epochs=4,
            aa_noise=noise,
            trajectory_files=("aa/reference.lammpstrj",),
        )
    )

    batch = workflow._run_post_accumulation(step_index=3)

    assert batch is not None
    assert captured_specs
    runtime_noise = captured_specs[0]["noise"]
    assert runtime_noise["samples_per_frame"] == 2
    assert runtime_noise["sigma"] == pytest.approx(0.04)
    assert runtime_noise["seed"] == 20
    assert runtime_noise["include_original"] is True
    assert runtime_noise["batch_size"] == 2
    assert "beta" not in runtime_noise


def test_fmworkflow_run_post_accumulation_forwards_perf_trace(monkeypatch, tmp_path):
    _patch_fmworkflow_basics(monkeypatch, forcefield=_linear_forcefield())
    captured_specs = []

    def _fake_run_post(spec, resource_pool, *, run_dir=None, python_exe=None):
        captured_specs.append(dict(spec))
        output_path = Path(spec["steps"][0]["output_file"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as fh:
            import pickle

            pickle.dump(
                {
                    "JtJ": np.eye(2, dtype=np.float64),
                    "Jty": np.zeros(2, dtype=np.float64),
                    "y_sumsq": 0.0,
                    "nframe": 1,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    monkeypatch.setattr("AceCG.workflows.fm.run_post", _fake_run_post)
    workflow = FMWorkflow(
        _minimal_fm_config(
            tmp_path,
            trajectory_files=("aa/reference.lammpstrj",),
            sampling=SamplingConfig(
                perf_trace=True,
                extras={"heartbeat_interval": 7, "perf_trace_all_ranks": True},
            ),
        )
    )

    batch = workflow._run_post_accumulation(step_index=0)

    assert batch is not None
    assert captured_specs
    assert captured_specs[0]["perf_trace"] is True
    assert captured_specs[0]["perf_trace_all_ranks"] is True
    assert captured_specs[0]["heartbeat_interval"] == 7
