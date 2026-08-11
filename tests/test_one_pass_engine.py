from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from AceCG.analysis.rdf import (
    accumulate_distribution_frame,
    finalize_distribution_state,
    init_distribution_state,
)
from AceCG.compute.mpi_engine import build_default_engine
from AceCG.io.trajectory import FrameRecord
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.potentials.lennardjones import LennardJonesPotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey


def _bond_key():
    return InteractionKey.bond("A", "B")


def _make_topo(bond_key):
    return SimpleNamespace(
        n_atoms=2,
        bonds=np.array([[0, 1]], dtype=np.int64),
        bond_key_index=np.array([0], dtype=np.int32),
        keys_bondtypes=[bond_key],
        angles=np.empty((0, 3), dtype=np.int64),
        angle_key_index=np.empty(0, dtype=np.int32),
        keys_angletypes=[],
        dihedrals=np.empty((0, 4), dtype=np.int64),
        dihedral_key_index=np.empty(0, dtype=np.int32),
        keys_dihedraltypes=[],
        real_site_indices=np.array([0, 1], dtype=np.int64),
        atom_type_codes=np.array([1, 2], dtype=np.int32),
        atom_type_name_to_code={"A": 1, "B": 2},
        exclude_12=np.empty((0, 2), dtype=np.int32),
        exclude_13=np.empty((0, 2), dtype=np.int32),
        exclude_14=np.empty((0, 2), dtype=np.int32),
    )


def _make_forcefield():
    bk = _bond_key()
    pot = HarmonicPotential("A", "B", k=5.0, r0=4.0)
    return Forcefield({bk: [pot]}), bk


class _DummyTrajectory:
    def __len__(self):
        return 2

    def __getitem__(self, frame_id):
        assert int(frame_id) in {0, 1}
        return SimpleNamespace(has_forces=True, has_velocities=False)


class _DummyUniverse:
    def __init__(self):
        self.trajectory = _DummyTrajectory()

    def select_atoms(self, sel):
        assert sel == "all"
        return SimpleNamespace(indices=np.array([0, 1], dtype=np.int64))


def _expected_fm_payload(frames, topo, forcefield, ref_forces, frame_weights):
    engine = build_default_engine()
    partials = []
    weights = np.asarray(frame_weights, dtype=np.float64)
    for (frame_id, pos, box), ref_force, frame_weight in zip(frames, ref_forces, weights):
        payload = engine.compute(
            request={"fm_stats"},
            frame=(frame_id, pos, box, np.asarray(ref_force, dtype=np.float64)),
            topology_arrays=topo,
            forcefield_snapshot=forcefield,
            frame_weight=float(frame_weight),
        )
        partials.append(payload["fm_stats"])
    weight_sum = float(sum(float(partial["weight_sum"]) for partial in partials))
    JtJ = sum(np.asarray(partial["JtJ"], dtype=np.float64) for partial in partials)
    Jty = sum(np.asarray(partial["Jty"], dtype=np.float64) for partial in partials)
    Jtf = sum(np.asarray(partial["Jtf"], dtype=np.float64) for partial in partials)
    yty = sum(float(partial["yty"]) for partial in partials)
    ftf = sum(float(partial["ftf"]) for partial in partials)
    fTy = sum(float(partial["fTy"]) for partial in partials)
    n_frames = sum(int(partial["n_frames"]) for partial in partials)
    n_atoms_obs = max(int(partial["n_atoms_obs"]) for partial in partials)
    scale = 1.0 / weight_sum if weight_sum > 0.0 else 0.0
    return {
        "JtJ": JtJ * scale,
        "Jty": Jty * scale,
        "Jtf": Jtf * scale,
        "y_sumsq": yty * scale,
        "f_sumsq": ftf * scale,
        "fty": fTy * scale,
        "nframe": int(n_frames),
        "weight_sum": weight_sum,
        "n_atoms_obs": int(n_atoms_obs),
    }


def _expected_rem_payload(frames, topo, forcefield, frame_weights, *, need_hessian):
    engine = build_default_engine()
    weights = np.asarray(frame_weights, dtype=np.float64)
    grad_rows = []
    hessian_rows = []
    outer_rows = []
    request = {"energy_grad"}
    if need_hessian:
        request.update({"energy_hessian", "energy_grad_outer"})
    for frame_id, pos, box in frames:
        payload = engine.compute(
            request=request,
            frame=(frame_id, pos, box, None),
            topology_arrays=topo,
            forcefield_snapshot=forcefield,
        )
        grad_rows.append(np.asarray(payload["energy_grad"], dtype=np.float64))
        if need_hessian:
            hessian_rows.append(np.asarray(payload["energy_hessian"], dtype=np.float64))
            outer_rows.append(np.asarray(payload["energy_grad_outer"], dtype=np.float64))
    grad_stack = np.asarray(grad_rows, dtype=np.float64)
    grad_sum = np.tensordot(weights, grad_stack, axes=1)
    weight_sum = float(weights.sum())
    payload = {
        "energy_grad_avg": grad_sum / weight_sum,
        "n_frames": int(len(frames)),
        "weight_sum": weight_sum,
    }
    if need_hessian:
        hessian_stack = np.asarray(hessian_rows, dtype=np.float64)
        outer_stack = np.asarray(outer_rows, dtype=np.float64)
        payload["d2U_avg"] = np.tensordot(weights, hessian_stack, axes=1) / weight_sum
        payload["grad_outer_avg"] = np.tensordot(weights, outer_stack, axes=1) / weight_sum
    return payload


def _expected_cdfm_zbx(frames, topo, forcefield, y_eff, *, mode, beta=None):
    engine = build_default_engine()
    y_eff_arr = np.asarray(y_eff, dtype=np.float64).ravel()
    n_params = forcefield.n_params()
    J_sum = np.zeros((y_eff_arr.size, n_params), dtype=np.float64)
    f_sum = np.zeros(y_eff_arr.size, dtype=np.float64)
    gu_sum = np.zeros(n_params, dtype=np.float64)
    gu_f_sum = np.zeros((n_params, y_eff_arr.size), dtype=np.float64)
    for frame_id, pos, box in frames:
        req = {"force", "force_grad", "energy_grad"}
        local = engine.compute(
            request=req,
            frame=(frame_id, pos, box, None),
            topology_arrays=topo,
            forcefield_snapshot=forcefield,
        )
        J_sum += np.asarray(local["force_grad"], dtype=np.float64)
        force_value = np.asarray(local["force"], dtype=np.float64).ravel()
        energy_grad = np.asarray(local["energy_grad"], dtype=np.float64).ravel()
        f_sum += force_value
        gu_sum += energy_grad
        gu_f_sum += np.outer(energy_grad, force_value)
    weight_sum = float(len(frames))
    f_bar = f_sum / weight_sum
    error = f_bar - y_eff_arr
    grad_direct = (J_sum / weight_sum).T @ error
    grad_reinforce = np.zeros_like(grad_direct)
    if mode == "reinforce":
        assert beta is not None
        grad_reinforce = -float(beta) * (
            (gu_f_sum @ error) / weight_sum
            - float(np.dot(f_bar, error)) * (gu_sum / weight_sum)
        )
    return {
        "grad_direct": grad_direct,
        "grad_reinforce": grad_reinforce,
        "n_samples": len(frames),
    }


@pytest.mark.parametrize("explicit_format", ["TRR", "trr", None])
def test_one_pass_engine_streams_once_and_matches_single_mode(
    monkeypatch, tmp_path: Path, explicit_format
):
    forcefield, bond_key = _make_forcefield()
    topo = _make_topo(bond_key)
    ff_path = tmp_path / "forcefield.pkl"
    with ff_path.open("wb") as handle:
        pickle.dump(forcefield, handle, protocol=pickle.HIGHEST_PROTOCOL)

    frames = [
        (
            0,
            np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64),
            np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float64),
        ),
        (
            1,
            np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]], dtype=np.float64),
            np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64),
            np.array([0.5, 0.0, 0.0, -0.5, 0.0, 0.0], dtype=np.float64),
        ),
    ]
    flat_frames = [(frame_id, pos, box) for frame_id, pos, box, _ in frames]
    ref_forces = [force for _, _, _, force in frames]
    frame_weights = np.array([1.0, 2.0], dtype=np.float64)

    calls = {"iter_frames": 0, "universe_formats": []}

    def fake_universe(*args, **kwargs):
        calls["universe_formats"].append(kwargs.get("format"))
        return _DummyUniverse()

    monkeypatch.setattr("MDAnalysis.Universe", fake_universe)
    monkeypatch.setattr(
        "AceCG.topology.topology_array.collect_topology_arrays",
        lambda *args, **kwargs: topo,
    )

    def fake_iter_frames(universe, *, frame_ids, include_forces=False):
        del universe
        calls["iter_frames"] += 1
        assert include_forces is True
        for seek_id in frame_ids:
            frame_id, positions, box, force = frames[int(seek_id)]
            # MDAnalysis exposes TRR force as kJ/(mol*A); the FM target below
            # is expressed in AceCG/LAMMPS-real kcal/(mol*A).
            yield FrameRecord(
                frame_id=frame_id,
                positions=positions,
                box=box,
                forces=force * 4.184,
            )

    monkeypatch.setattr("AceCG.io.trajectory.iter_frames", fake_iter_frames)

    spec = {
        "post_mode": "one_pass",
        "work_dir": str(tmp_path),
        "forcefield_path": str(ff_path),
        "topology": "topology.data",
        "trajectory": ["traj.trr"],
        "frame_end": 2,
        "frame_weight": frame_weights.tolist(),
        "perf_trace": True,
        "step_index": 7,
        "steps": [
            {"step_mode": "fm", "name": "fm", "output_file": str(tmp_path / "fm.pkl")},
            {
                "step_mode": "rem",
                "name": "rem",
                "need_hessian": True,
                "output_file": str(tmp_path / "rem.pkl"),
            },
        ],
    }
    if explicit_format is not None:
        spec["trajectory_format"] = explicit_format

    engine = build_default_engine()
    engine.run_post(spec)

    assert calls["iter_frames"] == 1
    assert calls["universe_formats"] == ["TRR"]
    timing_payload = json.loads(
        (tmp_path / "mpi_post_timing.json").read_text(encoding="utf-8")
    )
    assert timing_payload["metadata"] == {
        "size": 1,
        "n_steps": 2,
        "need_reference_forces": True,
    }

    with (tmp_path / "fm.pkl").open("rb") as handle:
        fm_payload = pickle.load(handle)
    with (tmp_path / "rem.pkl").open("rb") as handle:
        rem_payload = pickle.load(handle)

    fm_expected = _expected_fm_payload(flat_frames, topo, forcefield, ref_forces, frame_weights)
    rem_expected = _expected_rem_payload(
        flat_frames,
        topo,
        forcefield,
        frame_weights,
        need_hessian=True,
    )

    np.testing.assert_allclose(fm_payload["JtJ"], fm_expected["JtJ"])
    np.testing.assert_allclose(fm_payload["Jty"], fm_expected["Jty"])
    np.testing.assert_allclose(fm_payload["Jtf"], fm_expected["Jtf"])
    assert fm_payload["y_sumsq"] == pytest.approx(fm_expected["y_sumsq"])
    assert fm_payload["f_sumsq"] == pytest.approx(fm_expected["f_sumsq"])
    assert fm_payload["fty"] == pytest.approx(fm_expected["fty"])
    assert fm_payload["nframe"] == fm_expected["nframe"]
    assert fm_payload["weight_sum"] == pytest.approx(fm_expected["weight_sum"])
    assert fm_payload["n_atoms_obs"] == fm_expected["n_atoms_obs"]
    assert fm_payload["step_index"] == 7
    np.testing.assert_allclose(rem_payload["energy_grad_avg"], rem_expected["energy_grad_avg"])
    np.testing.assert_allclose(rem_payload["d2U_avg"], rem_expected["d2U_avg"])
    np.testing.assert_allclose(rem_payload["grad_outer_avg"], rem_expected["grad_outer_avg"])


def test_one_pass_rdf_matches_direct_accumulator(monkeypatch, tmp_path: Path):
    forcefield, bond_key = _make_forcefield()
    pair_key = InteractionKey.pair("A", "B")
    topo = _make_topo(bond_key)
    ff_path = tmp_path / "forcefield.pkl"
    with ff_path.open("wb") as handle:
        pickle.dump(forcefield, handle)

    frames = [
        (0, np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]]), np.array([10, 10, 10, 90, 90, 90])),
        (1, np.array([[1.0, 1.0, 1.0], [4.0, 1.0, 1.0]]), np.array([10, 10, 10, 90, 90, 90])),
    ]
    monkeypatch.setattr("MDAnalysis.Universe", lambda *args, **kwargs: _DummyUniverse())
    topology_exclusions = []

    def collect_topology(*args, **kwargs):
        topology_exclusions.append(kwargs["exclude_option"])
        return topo

    monkeypatch.setattr(
        "AceCG.topology.topology_array.collect_topology_arrays",
        collect_topology,
    )
    calls = {"iter_frames": 0, "geometry": 0}

    def fake_iter_frames(universe, *, frame_ids, include_forces=False):
        del universe, include_forces
        calls["iter_frames"] += 1
        return iter(
            FrameRecord(
                frame_id=int(frame_id),
                positions=frames[int(frame_id)][1],
                box=frames[int(frame_id)][2],
                forces=None,
            )
            for frame_id in frame_ids
        )

    monkeypatch.setattr("AceCG.io.trajectory.iter_frames", fake_iter_frames)
    import AceCG.compute.mpi_engine as engine_module

    actual_geometry = engine_module.compute_frame_geometry

    def counted_geometry(*args, **kwargs):
        calls["geometry"] += 1
        return actual_geometry(*args, **kwargs)

    monkeypatch.setattr(engine_module, "compute_frame_geometry", counted_geometry)
    output_path = tmp_path / "rdf.pkl"
    build_default_engine().run_post(
        {
            "work_dir": str(tmp_path),
            "forcefield_path": str(ff_path),
            "topology": "topology.data",
            "trajectory": "traj.xtc",
            "trajectory_format": "XTC",
            "frame_weight": [1.0, 2.0],
            "cutoff": 5.0,
            "steps": [
                {
                    "step_mode": "rdf",
                    "interaction_keys": [pair_key.label()],
                    "nbins_pair": 5,
                    "output_file": str(output_path),
                }
            ],
        }
    )
    assert calls == {"iter_frames": 1, "geometry": 2}
    assert topology_exclusions == ["none"]

    state = init_distribution_state(
        topo,
        forcefield,
        interaction_keys=[pair_key],
        cutoff=5.0,
        nbins_pair=5,
        sel_indices=np.array([0, 1], dtype=np.int32),
    )
    direct_engine = build_default_engine()
    geometry_mask = {bond_key: True, pair_key: True}
    for (frame_id, positions, box), weight in zip(frames, (1.0, 2.0)):
        payload = direct_engine.compute(
            request={"frame_cache"},
            frame=(frame_id, positions, box, None),
            topology_arrays=topo,
            forcefield_snapshot=forcefield,
            geometry_mask=geometry_mask,
            pair_type_list=[pair_key],
            pair_cutoff=5.0,
            sel_indices=np.array([0, 1], dtype=np.int32),
            exclude_option="none",
        )
        accumulate_distribution_frame(state, payload["frame_cache"], frame_weight=weight)
    expected = finalize_distribution_state(state)[pair_key]
    with output_path.open("rb") as handle:
        actual = pickle.load(handle)[pair_key]
    np.testing.assert_allclose(actual.x, expected.x)
    np.testing.assert_allclose(actual.values, expected.values)
    np.testing.assert_allclose(actual.counts, expected.counts)
    np.testing.assert_allclose(actual.edges, expected.edges)
    assert actual.mode == expected.mode
    assert actual.variable == expected.variable
    assert actual.n_frames == expected.n_frames
    assert actual.weight_sum == pytest.approx(expected.weight_sum)
    assert actual.meta == expected.meta


@pytest.mark.parametrize(
    "field,value",
    [
        ("rdf_source", "cache"),
        ("frame_start", 0),
        ("frame_end", 1),
        ("every", 2),
        ("sel_indices", [0]),
        ("exclude_option", "none"),
    ],
)
def test_one_pass_rdf_rejects_removed_step_source_fields(field, value):
    with pytest.raises(ValueError, match="were removed"):
        build_default_engine().run_post(
            {"steps": [{"step_mode": "rdf", field: value}]}
        )


def test_one_pass_rdf_rejects_noise_before_frame_processing():
    with pytest.raises(ValueError, match="noise is not supported"):
        build_default_engine().run_post(
            {"noise": {"enabled": True}, "steps": [{"step_mode": "rdf"}]}
        )


def test_fm_and_multiple_rdf_steps_share_one_source_geometry_and_top_level_weights(
    monkeypatch, tmp_path: Path
):
    forcefield, bond_key = _make_forcefield()
    pair_key = InteractionKey.pair("A", "B")
    topology = _make_topo(bond_key)
    forcefield_path = tmp_path / "forcefield.pkl"
    with forcefield_path.open("wb") as handle:
        pickle.dump(forcefield, handle)
    frames = [
        (0, 2.0, 1.0),
        (1, 2.5, 0.5),
        (2, 3.0, 0.25),
    ]

    class ThreeFrameTrajectory:
        def __len__(self):
            return 3

        def __getitem__(self, frame_id):
            assert 0 <= int(frame_id) < 3
            return SimpleNamespace(has_forces=True, has_velocities=False)

    class ThreeFrameUniverse:
        trajectory = ThreeFrameTrajectory()

        def select_atoms(self, selection):
            assert selection == "all"
            return SimpleNamespace(indices=np.array([0, 1], dtype=np.int32))

    calls = {"frames": 0, "pairs": 0, "geometry": 0}
    monkeypatch.setattr("MDAnalysis.Universe", lambda *args, **kwargs: ThreeFrameUniverse())
    monkeypatch.setattr(
        "AceCG.topology.topology_array.collect_topology_arrays",
        lambda *args, **kwargs: topology,
    )

    def fake_iter_frames(universe, *, frame_ids, include_forces=False):
        del universe
        assert include_forces
        calls["frames"] += 1
        return iter(
            FrameRecord(
                frame_id=int(frame_id),
                positions=np.array([[0.0, 0.0, 0.0], [frames[int(frame_id)][1], 0.0, 0.0]]),
                box=np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0]),
                forces=np.array([frames[int(frame_id)][2], 0.0, 0.0, -frames[int(frame_id)][2], 0.0, 0.0]),
            )
            for frame_id in frame_ids
        )

    monkeypatch.setattr("AceCG.io.trajectory.iter_frames", fake_iter_frames)
    import AceCG.compute.mpi_engine as engine_module

    actual_pairs = engine_module.compute_pairs_by_type
    actual_geometry = engine_module.compute_frame_geometry

    def counted_pairs(*args, **kwargs):
        calls["pairs"] += 1
        return actual_pairs(*args, **kwargs)

    def counted_geometry(*args, **kwargs):
        calls["geometry"] += 1
        return actual_geometry(*args, **kwargs)

    monkeypatch.setattr(engine_module, "compute_pairs_by_type", counted_pairs)
    monkeypatch.setattr(engine_module, "compute_frame_geometry", counted_geometry)
    fm_path, rdf_short_path, rdf_long_path = (
        tmp_path / "fm.pkl",
        tmp_path / "rdf_short.pkl",
        tmp_path / "rdf_long.pkl",
    )
    build_default_engine().run_post(
        {
            "work_dir": str(tmp_path),
            "forcefield_path": str(forcefield_path),
            "topology": "topology.data",
            "trajectory": "traj.lammpstrj",
            "trajectory_format": "LAMMPSDUMP",
            "frame_start": 0,
            "frame_end": 3,
            "every": 2,
            "frame_weight": [2.0, 5.0],
            "exclude_option": "none",
            "steps": [
                {"step_mode": "fm", "output_file": str(fm_path)},
                {
                    "step_mode": "rdf",
                    "interaction_keys": [pair_key.label()],
                    "cutoff": 4.0,
                    "nbins_pair": 4,
                    "output_file": str(rdf_short_path),
                },
                {
                    "step_mode": "rdf",
                    "interaction_keys": [pair_key.label()],
                    "cutoff": 5.0,
                    "nbins_pair": 5,
                    "output_file": str(rdf_long_path),
                },
            ],
        }
    )
    assert calls == {"frames": 1, "pairs": 2, "geometry": 2}
    with fm_path.open("rb") as handle:
        fm_payload = pickle.load(handle)
    assert fm_payload["nframe"] == 2
    assert fm_payload["weight_sum"] == pytest.approx(7.0)
    for path, expected_bins in ((rdf_short_path, 4), (rdf_long_path, 5)):
        with path.open("rb") as handle:
            result = pickle.load(handle)[pair_key]
        assert result.n_frames == 2
        assert result.weight_sum == pytest.approx(7.0)
        assert result.counts.shape == (expected_bins,)


def test_rdf_duplicate_global_ids_keep_global_id_weight_semantics(monkeypatch, tmp_path: Path):
    forcefield, bond_key = _make_forcefield()
    pair_key = InteractionKey.pair("A", "B")
    topology = _make_topo(bond_key)
    forcefield_path = tmp_path / "forcefield.pkl"
    with forcefield_path.open("wb") as handle:
        pickle.dump(forcefield, handle)
    seen_frame_ids = []
    monkeypatch.setattr("MDAnalysis.Universe", lambda *args, **kwargs: _DummyUniverse())
    monkeypatch.setattr(
        "AceCG.topology.topology_array.collect_topology_arrays",
        lambda *args, **kwargs: topology,
    )

    def fake_iter_frames(universe, *, frame_ids, include_forces=False):
        del universe, include_forces
        def records():
            for frame_id in frame_ids:
                seen_frame_ids.append(int(frame_id))
                distance = 1.25 if int(frame_id) == 0 else 2.25
                yield FrameRecord(
                    frame_id=int(frame_id),
                    positions=np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
                    box=np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]),
                    forces=None,
                )
        return records()

    monkeypatch.setattr("AceCG.io.trajectory.iter_frames", fake_iter_frames)
    output_path = tmp_path / "rdf.pkl"
    build_default_engine().run_post(
        {
            "work_dir": str(tmp_path),
            "forcefield_path": str(forcefield_path),
            "topology": "topology.data",
            "trajectory": "traj.xtc",
            "trajectory_format": "XTC",
            "frame_ids": [1, 0, 1],
            "frame_weight": [1.0, 2.0, 3.0],
            "exclude_option": "none",
            "steps": [{
                "step_mode": "rdf",
                "interaction_keys": [pair_key.label()],
                "cutoff": 4.0,
                "nbins_pair": 4,
                "output_file": str(output_path),
            }],
        }
    )
    with output_path.open("rb") as handle:
        result = pickle.load(handle)[pair_key]
    assert seen_frame_ids == [1, 0, 1]
    assert result.n_frames == 3
    # Inline duplicate ids share their global-id slot: the final id-1 weight
    # applies to both occurrences, yielding 3 + 2 + 3 rather than 1 + 2 + 3.
    assert result.weight_sum == pytest.approx(8.0)
    np.testing.assert_allclose(result.counts, [0.0, 2.0, 6.0, 0.0])


def test_masked_explicit_rdf_key_does_not_change_fm_force_payload(monkeypatch, tmp_path: Path):
    bond_key = _bond_key()
    pair_key = InteractionKey.pair("A", "B")
    forcefield = Forcefield({
        bond_key: [HarmonicPotential("A", "B", k=5.0, r0=4.0)],
        pair_key: [LennardJonesPotential("A", "B", epsilon=0.5, sigma=2.0, cutoff=5.0)],
    })
    forcefield.key_mask = {bond_key: True, pair_key: False}
    topology = _make_topo(bond_key)
    forcefield_path = tmp_path / "forcefield.pkl"
    with forcefield_path.open("wb") as handle:
        pickle.dump(forcefield, handle)
    monkeypatch.setattr("MDAnalysis.Universe", lambda *args, **kwargs: _DummyUniverse())
    monkeypatch.setattr(
        "AceCG.topology.topology_array.collect_topology_arrays",
        lambda *args, **kwargs: topology,
    )

    def fake_iter_frames(universe, *, frame_ids, include_forces=False):
        del universe
        assert include_forces
        return iter(
            FrameRecord(
                frame_id=int(frame_id),
                positions=np.array([[0.0, 0.0, 0.0], [3.0 + int(frame_id), 0.0, 0.0]]),
                box=np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0]),
                forces=np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
            )
            for frame_id in frame_ids
        )

    monkeypatch.setattr("AceCG.io.trajectory.iter_frames", fake_iter_frames)
    base_path, mixed_path, rdf_path = tmp_path / "base.pkl", tmp_path / "mixed.pkl", tmp_path / "rdf.pkl"
    common = {
        "work_dir": str(tmp_path), "forcefield_path": str(forcefield_path),
        "topology": "topology.data", "trajectory": "traj.lammpstrj",
        "trajectory_format": "LAMMPSDUMP", "frame_end": 2, "exclude_option": "none",
    }
    build_default_engine().run_post({**common, "steps": [{"step_mode": "fm", "output_file": str(base_path)}]})
    build_default_engine().run_post({**common, "steps": [
        {"step_mode": "fm", "output_file": str(mixed_path)},
        {"step_mode": "rdf", "interaction_keys": [pair_key.label()], "cutoff": 5.0, "nbins_pair": 5, "output_file": str(rdf_path)},
    ]})
    with base_path.open("rb") as handle:
        base_payload = pickle.load(handle)
    with mixed_path.open("rb") as handle:
        mixed_payload = pickle.load(handle)
    for field in ("JtJ", "Jty", "Jtf"):
        np.testing.assert_allclose(mixed_payload[field], base_payload[field])
    for field in ("y_sumsq", "f_sumsq", "fty", "weight_sum"):
        assert mixed_payload[field] == pytest.approx(base_payload[field])
    assert mixed_payload["nframe"] == base_payload["nframe"]
    with rdf_path.open("rb") as handle:
        rdf_result = pickle.load(handle)[pair_key]
    assert rdf_result.n_frames == 2
    assert np.sum(rdf_result.counts) == pytest.approx(2.0)


def test_one_pass_engine_supports_cdfm_zbx(monkeypatch, tmp_path: Path):
    forcefield, bond_key = _make_forcefield()
    topo = _make_topo(bond_key)
    ff_path = tmp_path / "forcefield.pkl"
    with ff_path.open("wb") as handle:
        pickle.dump(forcefield, handle, protocol=pickle.HIGHEST_PROTOCOL)

    frames = [
        (
            0,
            np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64),
            np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64),
            None,
        ),
        (
            1,
            np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]], dtype=np.float64),
            np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64),
            None,
        ),
    ]
    flat_frames = [(frame_id, pos, box) for frame_id, pos, box, _ in frames]

    # Baseline single frame used by rank 0 to compute y_eff from init config.
    init_positions = frames[0][1].copy()
    init_box = frames[0][2].copy()

    baseline_req = {"force"}
    baseline_force = build_default_engine().compute(
        request=baseline_req,
        frame=(0, init_positions, init_box, None),
        topology_arrays=topo,
        forcefield_snapshot=forcefield,
    )["force"]
    baseline_force = np.asarray(baseline_force, dtype=np.float64).ravel()
    # Arbitrary per-site reference force, then back out y_eff so the test
    # can independently verify payload values against the handmade expected.
    y_ref = np.array([0.2, 0.0, 0.0, -0.2, 0.0, 0.0], dtype=np.float64)
    expected_y_eff = y_ref - baseline_force
    init_force_path = tmp_path / "frame_000000.forces.npy"
    np.save(init_force_path, y_ref.reshape(2, 3))

    class _InitUniverse:
        def __init__(self):
            self.atoms = SimpleNamespace(positions=init_positions.copy())
            self.dimensions = init_box.copy()

        def select_atoms(self, sel):  # pragma: no cover - not exercised
            raise AssertionError("init universe should not be asked to select_atoms")

    class _TrajUniverse:
        def __init__(self):
            self.trajectory = _DummyTrajectory()

        def select_atoms(self, sel):
            assert sel == "all"
            return SimpleNamespace(indices=np.array([0, 1], dtype=np.int64))

    universe_calls = {"n": 0}

    def _universe_factory(*args, **kwargs):
        # The engine constructs (1) the trajectory universe during
        # shared-context preparation, and (2) the single-frame init
        # universe inside the cdfm_zbx preprocessing block. Return
        # different stubs for each.
        universe_calls["n"] += 1
        if universe_calls["n"] == 1:
            return _TrajUniverse()
        return _InitUniverse()

    monkeypatch.setattr("MDAnalysis.Universe", _universe_factory)
    monkeypatch.setattr(
        "AceCG.topology.topology_array.collect_topology_arrays",
        lambda *args, **kwargs: topo,
    )
    def fake_iter_frames(universe, *, frame_ids, include_forces=False):
        del universe, include_forces
        return iter(
            FrameRecord(
                frame_id=int(seek_id),
                positions=frames[int(seek_id)][1],
                box=frames[int(seek_id)][2],
                forces=None,
            )
            for seek_id in frame_ids
        )

    monkeypatch.setattr("AceCG.io.trajectory.iter_frames", fake_iter_frames)

    spec = {
        "post_mode": "one_pass",
        "work_dir": str(tmp_path),
        "forcefield_path": str(ff_path),
        "topology": str(tmp_path / "init_config.data"),
        "trajectory": ["traj.lammpstrj"],
        "trajectory_format": "LAMMPSDUMP",
        "frame_end": 2,
        "steps": [
            {
                "step_mode": "cdfm_zbx",
                "init_force_path": str(init_force_path),
                "init_frame_id": 0,
                "output_file": str(tmp_path / "cdfm.pkl"),
                "mode": "reinforce",
                "beta": 0.5,
            },
        ],
    }

    engine = build_default_engine()
    engine.run_post(spec)

    with (tmp_path / "cdfm.pkl").open("rb") as handle:
        payload = pickle.load(handle)

    expected = _expected_cdfm_zbx(
        flat_frames,
        topo,
        forcefield,
        expected_y_eff,
        mode="reinforce",
        beta=0.5,
    )
    np.testing.assert_allclose(payload["grad_direct"], expected["grad_direct"])
    np.testing.assert_allclose(payload["grad_reinforce"], expected["grad_reinforce"])
    assert payload["n_samples"] == expected["n_samples"]
