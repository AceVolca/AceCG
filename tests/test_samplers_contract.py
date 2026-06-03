from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pytest

from AceCG.samplers.base import BaseSampler, InitConfigRecord
from AceCG.samplers.conditioned import ConditionedSampler


@dataclass(frozen=True)
class _FakeScriptInfo:
    input_path: Path
    init_data_path: Path
    trajectory_path: Path
    checkpoint_path: Path | None


def _patch_sampler_runtime(monkeypatch, script_info: _FakeScriptInfo) -> None:
    import AceCG.samplers.base as sampler_base

    monkeypatch.setattr(
        sampler_base,
        "parse_script",
        lambda script_path, backend="lammps": script_info,
    )

    def _fake_stage(sim_input: Path, run_dir: Path) -> Path:
        staged = run_dir / Path(sim_input).name
        staged.write_text("# staged\n", encoding="utf-8")
        return staged

    monkeypatch.setattr(sampler_base, "stage_lammps_input_tree", _fake_stage)


def test_base_sampler_replay_archive_round_trips_state(monkeypatch, tmp_path: Path) -> None:
    script_path = tmp_path / "sample.in"
    script_path.write_text("# placeholder\n", encoding="utf-8")
    init_cfg = tmp_path / "seed.data"
    init_cfg.write_text("seed data\n", encoding="utf-8")

    _patch_sampler_runtime(
        monkeypatch,
        _FakeScriptInfo(
            input_path=script_path,
            init_data_path=Path("inputs/init.data"),
            trajectory_path=Path("traj/out.lammpstrj"),
            checkpoint_path=Path("restart/restart.data"),
        ),
    )

    sampler = BaseSampler(
        sim_input=script_path,
        init_config_pool=[init_cfg],
        replay_mode="latest",
        rng=random.Random(0),
    )
    state = sampler.init_epoch(iteration_index=3, epoch_dir=tmp_path / "epoch_0003", n_runs=1)
    plan = state.replica_plans[0]

    assert plan.input_script_path.exists()
    assert plan.read_data_target.exists()
    assert plan.read_data_target.read_text(encoding="utf-8") == "seed data\n"

    assert plan.write_data_path is not None
    plan.write_data_path.parent.mkdir(parents=True, exist_ok=True)
    plan.write_data_path.write_text("restart snapshot\n", encoding="utf-8")

    sampler.clean_epoch(state)

    assert len(sampler.replay_pool) == 1
    archived = sampler.replay_pool[0]
    assert archived.parent == state.epoch_dir / "replay_pool"
    assert archived.read_text(encoding="utf-8") == "restart snapshot\n"

    restored = BaseSampler(
        sim_input=script_path,
        init_config_pool=[init_cfg],
        replay_mode="latest",
        rng=random.Random(0),
    )
    restored.load_state_dict(sampler.state_dict())
    assert restored.replay_pool == sampler.replay_pool


def test_base_sampler_rejects_runtime_paths_that_escape_replica_dirs(monkeypatch, tmp_path: Path) -> None:
    script_path = tmp_path / "sample.in"
    init_cfg = tmp_path / "seed.data"
    init_cfg.write_text("seed\n", encoding="utf-8")

    _patch_sampler_runtime(
        monkeypatch,
        _FakeScriptInfo(
            input_path=script_path,
            init_data_path=Path("inputs/init.data"),
            trajectory_path=Path("/abs/out.lammpstrj"),
            checkpoint_path=None,
        ),
    )

    with pytest.raises(ValueError, match="replica-local"):
        BaseSampler(sim_input=script_path, init_config_pool=[init_cfg])


def test_conditioned_sampler_requires_frame_ids(monkeypatch, tmp_path: Path) -> None:
    script_path = tmp_path / "sample.in"
    init_cfg = tmp_path / "seed.data"
    init_cfg.write_text("seed\n", encoding="utf-8")

    _patch_sampler_runtime(
        monkeypatch,
        _FakeScriptInfo(
            input_path=script_path,
            init_data_path=Path("inputs/init.data"),
            trajectory_path=Path("traj/out.lammpstrj"),
            checkpoint_path=Path("restart/restart.data"),
        ),
    )

    with pytest.raises(ValueError, match="frame_id"):
        ConditionedSampler(
            sim_input=script_path,
            init_config_pool=[InitConfigRecord(path=init_cfg, frame_id=None)],
        )


def test_conditioned_sampler_uses_unique_init_configs_and_force_paths(monkeypatch, tmp_path: Path) -> None:
    script_path = tmp_path / "sample.in"
    script_path.write_text("# placeholder\n", encoding="utf-8")
    cfg_a = tmp_path / "seed_a.data"
    cfg_b = tmp_path / "seed_b.data"
    cfg_a.write_text("A\n", encoding="utf-8")
    cfg_b.write_text("B\n", encoding="utf-8")
    force_a = tmp_path / "seed_a.force.npy"
    force_b = tmp_path / "seed_b.force.npy"
    force_a.write_text("fa\n", encoding="utf-8")
    force_b.write_text("fb\n", encoding="utf-8")

    _patch_sampler_runtime(
        monkeypatch,
        _FakeScriptInfo(
            input_path=script_path,
            init_data_path=Path("inputs/init.data"),
            trajectory_path=Path("traj/out.lammpstrj"),
            checkpoint_path=Path("restart/restart.data"),
        ),
    )

    sampler = ConditionedSampler(
        sim_input=script_path,
        init_config_pool=[
            InitConfigRecord(path=cfg_a, frame_id=10, force_path=force_a),
            InitConfigRecord(path=cfg_b, frame_id=20, force_path=force_b),
        ],
        require_force_path=True,
        rng=random.Random(0),
    )
    state = sampler.init_epoch(iteration_index=1, epoch_dir=tmp_path / "epoch", n_runs=2)

    assert {plan.frame_id for plan in state.replica_plans} == {10, 20}
    assert {plan.init_force_path for plan in state.replica_plans} == {force_a, force_b}
    assert {plan.run_dir.name for plan in state.replica_plans} == {"zbx_0000", "zbx_0001"}
    assert all(plan.read_data_target.exists() for plan in state.replica_plans)
