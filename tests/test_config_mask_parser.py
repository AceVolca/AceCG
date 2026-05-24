import pytest

from AceCG.configs import parse_acg_file
from AceCG.topology.types import InteractionKey


def _write_cdrem_config(cfg_path, *, conditioning_pattern):
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/topology.data",
                "forcefield_path = forcefield/real.settings",
                "pair_style = table",
                "",
                "[training]",
                "method = cdrem",
                "temperature = 300.0",
                "output_dir = results/cdrem",
                "",
                "[sampling]",
                "input = scripts/in.xz.lmp",
                "engine_command = lmp",
                "ncores = 4",
                "",
                "[conditioning]",
                "input = scripts/in.zbx.lmp",
                f"init_config_pool = {conditioning_pattern}",
                "n_samples = 2",
                "ncores_per_task = 2",
                "",
                "[scheduler]",
                "task_timeout = 30",
                "",
                "[aa_ref]",
                "trajectory_files = ['aa/reference.lammpstrj']",
                "",
                "[vp]",
                "vp_names = ['VP']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_dsm_config(cfg_path, *, extra_training=(), extra_aa_ref=()):
    lines = [
        "[system]",
        "topology_file = system/topology.data",
        "cutoff = 25.0",
        "",
        "[training]",
        "method = dsm",
        "temperature = 300.0",
        "optimizer = adam",
        "n_epochs = 4",
        "output_dir = results/dsm",
        *extra_training,
        "",
        "[training.fm_specs]",
        'pair_specs = [{"types": ["1", "1"], "n_coeffs": 8, "domain": [1.0, 5.0], "degree": 3, "max_force": 100.0}]',
        "",
        "[aa_ref]",
        "trajectory_files = ['aa/reference.lammpstrj']",
        "noise_enabled = true",
        "noise_samples_per_frame = 2",
        "noise_sigma = 0.10",
        "noise_update_interval = 2",
        "noise_subsample_per_epoch = 8",
        *extra_aa_ref,
    ]
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_fm_noise_config(cfg_path, *, extra_training=(), extra_aa_ref=()):
    lines = [
        "[system]",
        "topology_file = system/topology.data",
        "cutoff = 25.0",
        "",
        "[training]",
        "method = fm",
        "temperature = 300.0",
        "optimizer = newton",
        "n_epochs = 1",
        "output_dir = results/fm",
        *extra_training,
        "",
        "[training.fm_specs]",
        'pair_specs = [{"types": ["1", "1"], "n_coeffs": 8, "domain": [1.0, 5.0], "degree": 3, "max_force": 100.0}]',
        "",
        "[aa_ref]",
        "trajectory_files = ['aa/reference.lammpstrj']",
        "noise_enabled = true",
        "noise_samples_per_frame = 2",
        "noise_sigma = 0.10",
        *extra_aa_ref,
    ]
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_acg_file_reads_forcefield_mask_path(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "mask.settings").write_text(
        "pair_coeff A B table mask 0:2\n",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "test.acg"
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/topology.data",
                "forcefield_path = forcefield/real.settings",
                "forcefield_mask_path = forcefield/mask.settings",
                "pair_style = table",
                "",
                "[training]",
                "method = rem",
                "temperature = 300.0",
                "output_dir = results/rem",
                "",
                "[sampling]",
                "input = scripts/in.rem.lmp",
                "engine_command = lmp",
                "",
                "[aa_ref]",
                "trajectory_files = ['aa/cg_traj.lammpstrj']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = parse_acg_file(cfg_path)

    assert cfg.system.forcefield_path == "forcefield/real.settings"
    assert cfg.system.forcefield_mask_path == "forcefield/mask.settings"
    assert cfg.system.forcefield_mask is not None
    assert cfg.system.forcefield_mask.entries == (
        (InteractionKey.pair("A", "B"), "table", ("mask", "0:2")),
    )


def test_parse_acg_file_requires_sampling_engine_command_for_rem(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/topology.data",
                "forcefield_path = forcefield/real.settings",
                "pair_style = table",
                "",
                "[training]",
                "method = rem",
                "temperature = 300.0",
                "output_dir = results/rem",
                "",
                "[sampling]",
                "input = scripts/in.rem.lmp",
                "",
                "[aa_ref]",
                "trajectory_files = ['aa/cg_traj.lammpstrj']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        parse_acg_file(cfg_path)
    except ValueError as exc:
        assert "sampling.engine_command is required" in str(exc)
    else:
        raise AssertionError("Expected REM config without sampling.engine_command to fail.")


def test_parse_acg_file_accepts_validation_section(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/topology.data",
                "forcefield_path = forcefield/real.settings",
                "pair_style = table",
                "",
                "[training]",
                "method = rem",
                "temperature = 300.0",
                "output_dir = results/rem",
                "",
                "[sampling]",
                "input = scripts/in.rem.lmp",
                "engine_command = lmp",
                "",
                "[validation]",
                "input = scripts/in.validation.lmp",
                "engine_command = lmp",
                "ncores = 2",
                "num_epochs_per_validation = 5",
                'sim_var = {"SEED": "{RANDOM}"}',
                "",
                "[scheduler]",
                "task_timeout = 30",
                "",
                "[aa_ref]",
                "trajectory_files = ['aa/cg_traj.lammpstrj']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = parse_acg_file(cfg_path)

    assert cfg.validation.enabled is True
    assert cfg.validation.input == "scripts/in.validation.lmp"
    assert cfg.validation.engine_command == "lmp"
    assert cfg.validation.ncores == 2
    assert cfg.validation.num_epochs_per_validation == 5
    assert cfg.validation.sim_var == {"SEED": "{RANDOM}"}


@pytest.mark.parametrize("field", ["replay_mode", "archive_trajectory"])
def test_parse_acg_file_rejects_validation_replay_and_archive_controls(tmp_path, field):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/topology.data",
                "forcefield_path = forcefield/real.settings",
                "pair_style = table",
                "",
                "[training]",
                "method = rem",
                "temperature = 300.0",
                "output_dir = results/rem",
                "",
                "[sampling]",
                "input = scripts/in.rem.lmp",
                "engine_command = lmp",
                "",
                "[validation]",
                "input = scripts/in.validation.lmp",
                "engine_command = lmp",
                f"{field} = off",
                "",
                "[scheduler]",
                "task_timeout = 30",
                "",
                "[aa_ref]",
                "trajectory_files = ['aa/cg_traj.lammpstrj']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"validation\.{field}"):
        parse_acg_file(cfg_path)


def test_parse_acg_file_requires_validation_forcefield_template_for_source_free_fm(tmp_path):
    cfg_path = tmp_path / "test.acg"
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/topology.data",
                "pair_style = table",
                "cutoff = 25.0",
                "",
                "[training]",
                "method = fm",
                "optimizer = adam",
                "output_dir = results/fm",
                "",
                "[training.fm_specs]",
                'pair_specs = [{"types": ["1", "1"], "n_coeffs": 8, "domain": [1.0, 5.0], "degree": 3, "max_force": 100.0, "init_mode": "authored_zero"}]',
                "",
                "[aa_ref]",
                "trajectory_files = ['aa/reference.lammpstrj']",
                "",
                "[validation]",
                "input = scripts/in.validation.lmp",
                "engine_command = lmp",
                "",
                "[scheduler]",
                "task_timeout = 30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation.forcefield_template_path is required"):
        parse_acg_file(cfg_path)


def test_parse_acg_file_warns_on_deprecated_scheduler_launcher(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/topology.data",
                "forcefield_path = forcefield/real.settings",
                "pair_style = table",
                "",
                "[training]",
                "method = rem",
                "temperature = 300.0",
                "output_dir = results/rem",
                "",
                "[sampling]",
                "input = scripts/in.rem.lmp",
                "engine_command = lmp",
                "",
                "[scheduler]",
                "launcher = local",
                "",
                "[aa_ref]",
                "trajectory_files = ['aa/cg_traj.lammpstrj']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.warns(DeprecationWarning, match=r"scheduler\.launcher"):
        cfg = parse_acg_file(cfg_path)

    assert cfg.scheduler.launcher is None


def test_parse_acg_file_accepts_scheduler_mpi_family_override(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/topology.data",
                "forcefield_path = forcefield/real.settings",
                "pair_style = table",
                "",
                "[training]",
                "method = rem",
                "temperature = 300.0",
                "output_dir = results/rem",
                "",
                "[sampling]",
                "input = scripts/in.rem.lmp",
                "engine_command = lmp",
                "",
                "[scheduler]",
                "mpi_family = intel",
                "",
                "[aa_ref]",
                "trajectory_files = ['aa/cg_traj.lammpstrj']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = parse_acg_file(cfg_path)

    assert cfg.scheduler.mpi_family == "intel"


def test_parse_acg_file_accepts_sampling_perf_trace(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/topology.data",
                "forcefield_path = forcefield/real.settings",
                "pair_style = table",
                "",
                "[training]",
                "method = rem",
                "temperature = 300.0",
                "output_dir = results/rem",
                "",
                "[sampling]",
                "input = scripts/in.rem.lmp",
                "engine_command = lmp",
                "perf_trace = true",
                "",
                "[aa_ref]",
                "trajectory_files = ['aa/cg_traj.lammpstrj']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = parse_acg_file(cfg_path)

    assert cfg.sampling.perf_trace is True


def test_parse_acg_file_accepts_scheduler_extra_env(tmp_path, recwarn):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/topology.data",
                "forcefield_path = forcefield/real.settings",
                "pair_style = table",
                "",
                "[training]",
                "method = rem",
                "temperature = 300.0",
                "output_dir = results/rem",
                "",
                "[sampling]",
                "input = scripts/in.rem.lmp",
                "engine_command = lmp",
                "",
                "[scheduler]",
                'extra_env = {"OMP_NUM_THREADS": "1", "SOFTPATH": "/software"}',
                "",
                "[aa_ref]",
                "trajectory_files = ['aa/cg_traj.lammpstrj']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = parse_acg_file(cfg_path)

    assert cfg.scheduler.extras["extra_env"] == {
        "OMP_NUM_THREADS": "1",
        "SOFTPATH": "/software",
    }
    assert not any("extra_env" in str(w.message) for w in recwarn)


def test_parse_acg_file_accepts_aa_ref_noise_config(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/topology.data",
                "forcefield_path = forcefield/real.settings",
                "pair_style = table",
                "cutoff = 10.0",
                "",
                "[training]",
                "method = rem",
                "temperature = 300.0",
                "output_dir = results/rem",
                "",
                "[sampling]",
                "input = scripts/in.rem.lmp",
                "engine_command = lmp",
                "",
                "[aa_ref]",
                "trajectory_files = ['aa/cg_traj.lammpstrj']",
                "noise_enabled = true",
                "noise_samples_per_frame = 4",
                "noise_sigma = 0.05",
                "noise_sigma_final = 0.01",
                "noise_schedule = cosine",
                "noise_update_interval = 3",
                "noise_seed = 17",
                "noise_distribution = gaussian",
                "noise_selection = real",
                "noise_include_original = true",
                "noise_wrap = true",
                "noise_batch_size = 2",
                "noise_subsample_per_epoch = 640",
                "noise_cache_policy = stage",
                "noise_neighbor_mode = skin",
                "noise_neighbor_skin = 0.25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = parse_acg_file(cfg_path)
    noise = cfg.aa_ref.noise

    assert noise.enabled is True
    assert noise.samples_per_frame == 4
    assert noise.sigma == pytest.approx(0.05)
    assert noise.sigma_final == pytest.approx(0.01)
    assert noise.schedule == "cosine"
    assert noise.update_interval == 3
    assert noise.seed == 17
    assert noise.selection == "real"
    assert noise.include_original is True
    assert noise.wrap is True
    assert noise.batch_size == 2
    assert noise.subsample_per_epoch == 640
    assert noise.cache_policy == "stage"
    assert noise.neighbor_mode == "skin"
    assert noise.neighbor_skin == pytest.approx(0.25)

    runtime = noise.to_runtime_dict()
    assert set(runtime) == {
        "samples_per_frame",
        "sigma",
        "seed",
        "distribution",
        "selection",
        "include_original",
        "wrap",
        "batch_size",
        "subsample_per_epoch",
        "neighbor_mode",
        "neighbor_skin",
    }
    assert runtime["subsample_per_epoch"] == 640
    assert runtime["neighbor_mode"] == "skin"
    assert runtime["neighbor_skin"] == pytest.approx(0.25)


def test_parse_acg_file_rejects_negative_aa_ref_noise_subsample(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/topology.data",
                "forcefield_path = forcefield/real.settings",
                "pair_style = table",
                "",
                "[training]",
                "method = rem",
                "temperature = 300.0",
                "output_dir = results/rem",
                "",
                "[sampling]",
                "input = scripts/in.rem.lmp",
                "engine_command = lmp",
                "",
                "[aa_ref]",
                "trajectory_files = ['aa/reference.lammpstrj']",
                "noise_enabled = true",
                "noise_samples_per_frame = 1",
                "noise_sigma = 0.05",
                "noise_subsample_per_epoch = -1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="noise_subsample_per_epoch must be non-negative"):
        parse_acg_file(cfg_path)


def test_parse_acg_file_rejects_invalid_noise_neighbor_mode(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/topology.data",
                "forcefield_path = forcefield/real.settings",
                "pair_style = table",
                "",
                "[training]",
                "method = rem",
                "temperature = 300.0",
                "output_dir = results/rem",
                "",
                "[sampling]",
                "input = scripts/in.rem.lmp",
                "engine_command = lmp",
                "",
                "[aa_ref]",
                "trajectory_files = ['aa/reference.lammpstrj']",
                "noise_enabled = true",
                "noise_samples_per_frame = 1",
                "noise_sigma = 0.05",
                "noise_neighbor_mode = bad",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="noise_neighbor_mode must be one of"):
        parse_acg_file(cfg_path)


def test_parse_acg_file_rejects_negative_noise_neighbor_skin(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/topology.data",
                "forcefield_path = forcefield/real.settings",
                "pair_style = table",
                "",
                "[training]",
                "method = rem",
                "temperature = 300.0",
                "output_dir = results/rem",
                "",
                "[sampling]",
                "input = scripts/in.rem.lmp",
                "engine_command = lmp",
                "",
                "[aa_ref]",
                "trajectory_files = ['aa/reference.lammpstrj']",
                "noise_enabled = true",
                "noise_samples_per_frame = 1",
                "noise_sigma = 0.05",
                "noise_neighbor_skin = -0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="noise_neighbor_skin must be non-negative"):
        parse_acg_file(cfg_path)


def test_parse_acg_file_accepts_dsm_with_noise_and_fm_specs(tmp_path):
    cfg_path = tmp_path / "dsm.acg"
    _write_dsm_config(cfg_path)

    cfg = parse_acg_file(cfg_path)

    assert cfg.training.method == "dsm"
    assert cfg.training.fm_specs.pair_specs
    assert cfg.aa_ref.noise.enabled is True
    assert cfg.aa_ref.noise.subsample_per_epoch == 8


def test_parse_acg_file_reads_fm_noise_force_mix_ratio(tmp_path):
    cfg_path = tmp_path / "fm.acg"
    _write_fm_noise_config(cfg_path, extra_aa_ref=("noise_force_mix_ratio = 0.25",))

    cfg = parse_acg_file(cfg_path)

    assert cfg.training.method == "fm"
    assert cfg.aa_ref.noise.force_mix_ratio == pytest.approx(0.25)


@pytest.mark.parametrize("value", ["-0.01", "1.01"])
def test_parse_acg_file_rejects_noise_force_mix_ratio_out_of_range(tmp_path, value):
    cfg_path = tmp_path / "fm.acg"
    _write_fm_noise_config(cfg_path, extra_aa_ref=(f"noise_force_mix_ratio = {value}",))

    with pytest.raises(ValueError, match="noise_force_mix_ratio must be between 0 and 1"):
        parse_acg_file(cfg_path)


def test_parse_acg_file_rejects_noise_force_mix_ratio_without_temperature(tmp_path):
    cfg_path = tmp_path / "fm.acg"
    _write_fm_noise_config(cfg_path, extra_aa_ref=("noise_force_mix_ratio = 0.25",))
    text = cfg_path.read_text(encoding="utf-8").replace("temperature = 300.0\n", "")
    cfg_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="noise_force_mix_ratio requires training.temperature"):
        parse_acg_file(cfg_path)


def test_parse_acg_file_rejects_dsm_solver_mode(tmp_path):
    cfg_path = tmp_path / "dsm.acg"
    _write_dsm_config(cfg_path, extra_training=("fm_method = solver",))

    with pytest.raises(ValueError, match="DSM is iterative"):
        parse_acg_file(cfg_path)


def test_parse_acg_file_rejects_dsm_without_noise(tmp_path):
    cfg_path = tmp_path / "dsm.acg"
    _write_dsm_config(
        cfg_path,
        extra_aa_ref=(
            "noise_enabled = false",
        ),
    )

    with pytest.raises(ValueError, match="DSM requires aa_ref.noise_enabled=true"):
        parse_acg_file(cfg_path)


def test_parse_acg_file_rejects_cdrem_aa_ref_noise(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    conditioning_dir = tmp_path / "conditioning"
    conditioning_dir.mkdir()
    (conditioning_dir / "frame_000001.data").write_text("# frame 1\n", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    _write_cdrem_config(cfg_path, conditioning_pattern="conditioning/frame_*.data")
    text = cfg_path.read_text(encoding="utf-8")
    text = text.replace(
        "trajectory_files = ['aa/reference.lammpstrj']\n",
        "trajectory_files = ['aa/reference.lammpstrj']\n"
        "noise_enabled = true\n"
        "noise_samples_per_frame = 2\n"
        "noise_sigma = 0.05\n",
    )
    cfg_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="supported only for method='fm' or method='rem' or method='dsm'"):
        parse_acg_file(cfg_path)


def test_parse_acg_file_rejects_cdfm_aa_ref_noise(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    conditioning_dir = tmp_path / "conditioning"
    conditioning_dir.mkdir()
    (conditioning_dir / "frame_000001.data").write_text("# frame 1\n", encoding="utf-8")
    (conditioning_dir / "frame_000001.forces.npy").write_text("", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    _write_cdfm_config(
        cfg_path,
        init_config_pattern="conditioning/frame_*.data",
        init_force_pattern="conditioning/frame_*.forces.npy",
    )
    with cfg_path.open("a", encoding="utf-8") as fh:
        fh.write(
            "\n[aa_ref]\n"
            "noise_enabled = true\n"
            "noise_samples_per_frame = 2\n"
            "noise_sigma = 0.05\n"
        )

    with pytest.raises(ValueError, match="supported only for method='fm' or method='rem' or method='dsm'"):
        parse_acg_file(cfg_path)


def test_parse_acg_file_accepts_flat_integer_conditioning_pool_for_cdrem(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    conditioning_dir = tmp_path / "conditioning"
    conditioning_dir.mkdir()
    (conditioning_dir / "frame_000001.data").write_text("# frame 1\n", encoding="utf-8")
    (conditioning_dir / "frame_000123.data").write_text("# frame 123\n", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    _write_cdrem_config(cfg_path, conditioning_pattern="conditioning/frame_*.data")

    cfg = parse_acg_file(cfg_path)

    assert cfg.conditioning.init_config_pool == "conditioning/frame_*.data"


def test_parse_acg_file_rejects_nested_conditioning_pool_for_cdrem(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    nested_a = tmp_path / "conditioning" / "001"
    nested_b = tmp_path / "conditioning" / "002"
    nested_a.mkdir(parents=True)
    nested_b.mkdir(parents=True)
    (nested_a / "frame_000001.data").write_text("# frame 1\n", encoding="utf-8")
    (nested_b / "frame_000002.data").write_text("# frame 2\n", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    _write_cdrem_config(cfg_path, conditioning_pattern="conditioning/*/frame_*.data")

    with pytest.raises(ValueError, match="flat directory"):
        parse_acg_file(cfg_path)


def test_parse_acg_file_rejects_noncanonical_conditioning_filenames_for_cdrem(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    conditioning_dir = tmp_path / "conditioning"
    conditioning_dir.mkdir()
    (conditioning_dir / "subsample_01.data").write_text("# bad name\n", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    _write_cdrem_config(cfg_path, conditioning_pattern="conditioning/*.data")

    with pytest.raises(ValueError, match="frame_<integer>.data"):
        parse_acg_file(cfg_path)


def _write_cdfm_config(cfg_path, *, init_config_pattern, init_force_pattern, mask_cg_only=None):
    mask_line = (
        f"mask_cg_only = {str(mask_cg_only).lower()}"
        if mask_cg_only is not None
        else ""
    )
    lines = [
        "[system]",
        "topology_file = system/topology.data",
        "forcefield_path = forcefield/real.settings",
        "pair_style = table",
        "",
        "[training]",
        "method = cdfm",
        "temperature = 300.0",
        "output_dir = results/cdfm",
        "",
        "[sampling]",
        "ncores = 4",
        "",
        "[conditioning]",
        "input = scripts/in.zbx.lmp",
        f"init_config_pool = {init_config_pattern}",
    ]
    if init_force_pattern is not None:
        lines.append(f"init_force_pool = {init_force_pattern}")
    if mask_line:
        lines.append(mask_line)
    lines.extend(
        [
            "n_samples = 2",
            "ncores_per_task = 2",
            "",
            "[scheduler]",
            "task_timeout = 30",
            "",
            "[vp]",
            "vp_names = ['VP']",
        ]
    )
    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_acg_file_accepts_cdfm_init_force_pool_and_mask_cg_only(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    conditioning_dir = tmp_path / "conditioning"
    conditioning_dir.mkdir()
    (conditioning_dir / "frame_000001.data").write_text("# frame 1\n", encoding="utf-8")
    (conditioning_dir / "frame_000001.forces.npy").write_text("", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    _write_cdfm_config(
        cfg_path,
        init_config_pattern="conditioning/frame_*.data",
        init_force_pattern="conditioning/frame_*.forces.npy",
        mask_cg_only=False,
    )

    cfg = parse_acg_file(cfg_path)

    assert cfg.conditioning.init_config_pool == "conditioning/frame_*.data"
    assert cfg.conditioning.init_force_pool == "conditioning/frame_*.forces.npy"
    assert cfg.conditioning.mask_cg_only is False


def test_parse_acg_file_defaults_mask_cg_only_true_for_cdfm(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    conditioning_dir = tmp_path / "conditioning"
    conditioning_dir.mkdir()
    (conditioning_dir / "frame_000001.data").write_text("# frame 1\n", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    _write_cdfm_config(
        cfg_path,
        init_config_pattern="conditioning/frame_*.data",
        init_force_pattern="conditioning/frame_*.forces.npy",
    )

    cfg = parse_acg_file(cfg_path)

    assert cfg.conditioning.mask_cg_only is True


def test_parse_acg_file_rejects_cdfm_without_init_force_pool(tmp_path):
    forcefield_dir = tmp_path / "forcefield"
    forcefield_dir.mkdir()
    (forcefield_dir / "real.settings").write_text("", encoding="utf-8")
    conditioning_dir = tmp_path / "conditioning"
    conditioning_dir.mkdir()
    (conditioning_dir / "frame_000001.data").write_text("# frame 1\n", encoding="utf-8")
    cfg_path = tmp_path / "test.acg"
    _write_cdfm_config(
        cfg_path,
        init_config_pattern="conditioning/frame_*.data",
        init_force_pattern=None,
    )

    with pytest.raises(ValueError, match="init_force_pool"):
        parse_acg_file(cfg_path)


def test_parse_dsm_accepts_fixed_priors_and_authored_gauss_cut(tmp_path):
    cfg_path = tmp_path / "barnase_gauss.acg"
    cfg_path.write_text(
        "\n".join(
            [
                "[system]",
                "topology_file = system/BB_CG.data",
                "fixed_forcefield_path = rendered/fixed_priors.ff",
                "pair_style = hybrid/overlay soft 10.0 gauss/cut 30.0",
                "cutoff = 30.0",
                "",
                "[training]",
                "method = dsm",
                "fm_method = iterator",
                "optimizer = adamw weight_decay=1e-5",
                "lr = 1e-3",
                "n_epochs = 300",
                "temperature = 300.0",
                "output_dir = results",
                "",
                "[training.fm_specs]",
                "pair_specs = [",
                '  {"types": ["8", "42"], "model": "gauss/cut", "domain": [4.0, 30.0], "A": -3.0, "r0": 11.2, "sigma": 1.5, "cutoff": 30.0},',
                "  ]",
                "",
                "[aa_ref]",
                'trajectory_files = ["traj.lammpstrj"]',
                "trajectory_format = LAMMPSDUMP",
                "noise_enabled = true",
                "noise_samples_per_frame = 32",
                "noise_sigma = 0.2",
                "noise_sigma_final = 0.2",
                "noise_schedule = constant",
                "noise_batch_size = 32",
                "noise_neighbor_mode = chunk",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = parse_acg_file(cfg_path)
    spec = cfg.training.fm_specs.pair_specs[0]

    assert cfg.system.fixed_forcefield_path == "rendered/fixed_priors.ff"
    assert spec.model == "gauss/cut"
    assert spec.init_mode == "authored_direct"
    assert spec.model_size == 3
    assert spec.model_overrides == {
        "A": -3.0,
        "r0": 11.2,
        "sigma": 1.5,
        "cutoff": 30.0,
    }
    assert cfg.aa_ref.noise.samples_per_frame == 32
    assert cfg.aa_ref.noise.batch_size == 32
    assert cfg.aa_ref.noise.neighbor_mode == "chunk"
