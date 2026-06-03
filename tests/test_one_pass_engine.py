from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from AceCG.compute.mpi_engine import build_default_engine
from AceCG.potentials.harmonic import HarmonicPotential
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
    )


def _make_forcefield():
    bk = _bond_key()
    pot = HarmonicPotential("A", "B", k=5.0, r0=4.0)
    return Forcefield({bk: [pot]}), bk


class _DummyTrajectory:
    def __len__(self):
        return 2


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


def _expected_cdfm_zbx(frames, topo, forcefield, y_eff, *, mode):
    engine = build_default_engine()
    y_eff_arr = np.asarray(y_eff, dtype=np.float64).ravel()
    n_params = forcefield.n_params()
    J_sum = np.zeros((y_eff_arr.size, n_params), dtype=np.float64)
    f_sum = np.zeros(y_eff_arr.size, dtype=np.float64)
    for frame_id, pos, box in frames:
        req = {"force", "force_grad"}
        local = engine.compute(
            request=req,
            frame=(frame_id, pos, box, None),
            topology_arrays=topo,
            forcefield_snapshot=forcefield,
        )
        J_sum += np.asarray(local["force_grad"], dtype=np.float64)
        f_sum += np.asarray(local["force"], dtype=np.float64).ravel()
    weight_sum = float(len(frames))
    f_bar = f_sum / weight_sum
    error = f_bar - y_eff_arr
    grad_direct = (J_sum / weight_sum).T @ error
    grad_reinforce = np.zeros_like(grad_direct)
    if mode == "reinforce":
        raise AssertionError("reinforce mode is not used in this test helper")
    return {
        "grad_direct": grad_direct,
        "grad_reinforce": grad_reinforce,
        "n_samples": len(frames),
    }


def test_one_pass_engine_streams_once_and_matches_single_mode(monkeypatch, tmp_path: Path):
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

    calls = {"iter_frames": 0}

    monkeypatch.setattr(
        "MDAnalysis.Universe",
        lambda *args, **kwargs: _DummyUniverse(),
    )
    monkeypatch.setattr(
        "AceCG.topology.topology_array.collect_topology_arrays",
        lambda *args, **kwargs: topo,
    )

    def fake_iter_frames(universe, *, start=0, end=None, every=1, include_forces=False):
        del universe, start, end, every
        calls["iter_frames"] += 1
        assert include_forces is True
        for item in frames:
            yield item

    monkeypatch.setattr("AceCG.io.trajectory.iter_frames", fake_iter_frames)

    spec = {
        "post_mode": "one_pass",
        "work_dir": str(tmp_path),
        "forcefield_path": str(ff_path),
        "topology": "topology.data",
        "trajectory": ["traj.lammpstrj"],
        "trajectory_format": "LAMMPSDUMP",
        "frame_end": 2,
        "frame_weight": frame_weights.tolist(),
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

    engine = build_default_engine()
    engine.run_post(spec)

    assert calls["iter_frames"] == 1

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
    np.testing.assert_allclose(rem_payload["energy_grad_avg"], rem_expected["energy_grad_avg"])
    np.testing.assert_allclose(rem_payload["d2U_avg"], rem_expected["d2U_avg"])
    np.testing.assert_allclose(rem_payload["grad_outer_avg"], rem_expected["grad_outer_avg"])


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
    monkeypatch.setattr(
        "AceCG.io.trajectory.iter_frames",
        lambda universe, *, start=0, end=None, every=1, include_forces=False: iter(frames),
    )

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
                "mode": "direct",
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
        mode="direct",
    )
    np.testing.assert_allclose(payload["grad_direct"], expected["grad_direct"])
    np.testing.assert_allclose(payload["grad_reinforce"], expected["grad_reinforce"])
    assert payload["n_samples"] == expected["n_samples"]
