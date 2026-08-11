"""The VP workflow is a thin path/configuration owner."""

from __future__ import annotations

import inspect

from AceCG.configs.vp_config import VPConfig
from AceCG.configs.vp_growth_config import VPGrowthAARef, VPGrowthConfig, VPGrowthRun
from AceCG.workflows import vp_growth as vp_growth_module
from AceCG.workflows.vp_growth import VPGrowthWorkflow


def test_workflow_resolves_paths_and_calls_one_terminal(tmp_path, monkeypatch):
    config = VPGrowthConfig(
        path=tmp_path / "config" / "run.acg",
        aa_ref=VPGrowthAARef(
            trajectory_files=("a.xtc", "b.xtc"),
            trajectory_format="XTC",
            ref_topo="cg.data",
        ),
        vp=VPConfig(),
        run=VPGrowthRun(output_dir="results"),
    )
    captured = {}

    def terminal(**kwargs):
        captured.update(kwargs)
        output = kwargs["output_dir"]
        return {
            "output_dir": output,
            "topology_path": output / "vp_topology.data",
            "latent_settings_path": output / "latent.settings",
            "timing_path": output / "timing.json",
            "manifest_path": output / "manifest.json",
            "n_selected": 4,
            "n_unique": 3,
        }

    monkeypatch.setattr(vp_growth_module, "grow_vp_trajectory", terminal)
    result = VPGrowthWorkflow(config).run()

    base = config.path.parent
    assert captured["output_dir"] == (base / "results").resolve()
    assert captured["reference_topology"] == (base / "cg.data").resolve()
    reader = captured["reader"]
    assert reader.trajectory_files == (
        str((base / "a.xtc").resolve()),
        str((base / "b.xtc").resolve()),
    )
    assert reader.requested_strategy == "auto"
    assert reader.broadcast_segment_limit == 2
    assert result.output_dir == (base / "results").resolve()
    assert result.n_selected == 4
    assert result.n_unique == 3


def test_workflow_run_does_not_own_reader_or_output_operations():
    source = inspect.getsource(VPGrowthWorkflow.run)
    for forbidden in (
        ".scan(",
        ".local_slice(",
        ".iter_local(",
        ".gather(",
        "write_vp_data",
        "write_latent_settings",
        "manifest.json",
        "timing.json",
    ):
        assert forbidden not in source
    assert source.count("grow_vp_trajectory(") == 1
