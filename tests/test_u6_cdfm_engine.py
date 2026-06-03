"""CDFM helper and one-pass engine tests."""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from AceCG.compute.mpi_engine import build_default_engine
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey


def _request(*names):
    return frozenset(names)


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
    bond_key = _bond_key()
    pot = HarmonicPotential("A", "B", k=5.0, r0=4.0)
    return Forcefield({bond_key: [pot]}), bond_key


def _frames():
    return [
        (
            0,
            np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64),
            np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64),
        ),
        (
            1,
            np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]], dtype=np.float64),
            np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64),
        ),
    ]


def _expected_cdfm_y_eff(*, forcefield, topo, refs):
    force_rows = []
    engine = build_default_engine()
    for frame_id, positions, box in _frames():
        payload = engine.compute(
            request=_request("force"),
            frame=(frame_id, positions, box, None),
            topology_arrays=topo,
            forcefield_snapshot=forcefield,
        )
        force_rows.append(np.asarray(payload["force"], dtype=np.float64))
    return np.vstack(
        [
            np.asarray(ref, dtype=np.float64).ravel() - np.asarray(model, dtype=np.float64).ravel()
            for model, ref in zip(force_rows, refs)
        ]
    )


def _expected_cdfm_zbx(*, forcefield, topo, y_eff, mode: str, beta: float | None = None):
    engine = build_default_engine()
    y_eff_arr = np.asarray(y_eff, dtype=np.float64).ravel()
    n_params = forcefield.n_params()
    J_sum = np.zeros((y_eff_arr.size, n_params), dtype=np.float64)
    f_sum = np.zeros(y_eff_arr.size, dtype=np.float64)
    gu_sum = np.zeros(n_params, dtype=np.float64)
    gu_f_sum = np.zeros((n_params, y_eff_arr.size), dtype=np.float64)
    n_samples = 0
    request = _request("force", "force_grad", "energy_grad")
    for frame_id, positions, box in _frames():
        payload = engine.compute(
            request=request,
            frame=(frame_id, positions, box, None),
            topology_arrays=topo,
            forcefield_snapshot=forcefield,
        )
        J_sum += np.asarray(payload["force_grad"], dtype=np.float64)
        f_vec = np.asarray(payload["force"], dtype=np.float64).ravel()
        f_sum += f_vec
        gu = np.asarray(payload["energy_grad"], dtype=np.float64).ravel()
        gu_sum += gu
        gu_f_sum += np.outer(gu, f_vec)
        n_samples += 1
    weight_sum = float(n_samples)
    f_bar = f_sum / weight_sum
    error = f_bar - y_eff_arr
    grad_direct = (J_sum / weight_sum).T @ error
    grad_reinforce = np.zeros_like(grad_direct)
    if mode == "reinforce":
        assert beta is not None
        gu_bar = gu_sum / weight_sum
        phi_bar = float(np.dot(f_bar, error))
        grad_reinforce = -float(beta) * (
            (gu_f_sum @ error) / weight_sum - phi_bar * gu_bar
        )
    return {
        "grad_direct": grad_direct,
        "grad_reinforce": grad_reinforce,
        "obs_rows": int(error.size),
        "n_samples": int(weight_sum),
    }


def test_cdfm_y_eff_shape_and_values():
    forcefield, bond_key = _make_forcefield()
    topo = _make_topo(bond_key)
    refs = [
        np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.5, 0.0, 0.0, -0.5, 0.0, 0.0], dtype=np.float64),
    ]
    y_eff = _expected_cdfm_y_eff(forcefield=forcefield, topo=topo, refs=refs)
    assert y_eff.shape == (2, 6)
    assert np.any(np.abs(y_eff) > 1.0e-12)


def test_cdfm_zbx_grads_direct():
    forcefield, bond_key = _make_forcefield()
    topo = _make_topo(bond_key)
    result = _expected_cdfm_zbx(
        forcefield=forcefield,
        topo=topo,
        y_eff=np.zeros(6, dtype=np.float64),
        mode="direct",
    )
    assert result is not None
    assert result["grad_direct"].shape == (2,)
    assert result["grad_reinforce"].shape == (2,)
    assert result["obs_rows"] == 6
    assert result["n_samples"] == 2


def test_cdfm_zbx_grads_reinforce_nonzero():
    forcefield, bond_key = _make_forcefield()
    topo = _make_topo(bond_key)
    result = _expected_cdfm_zbx(
        forcefield=forcefield,
        topo=topo,
        y_eff=np.zeros(6, dtype=np.float64),
        mode="reinforce",
        beta=0.7,
    )
    assert result is not None
    assert result["grad_reinforce"].shape == (2,)
    assert np.any(np.abs(result["grad_reinforce"]) > 1.0e-12)


def test_one_pass_engine_supports_cdrem_alias(monkeypatch, tmp_path: Path):
    forcefield, bond_key = _make_forcefield()
    topo = _make_topo(bond_key)
    ff_path = tmp_path / "forcefield.pkl"
    with ff_path.open("wb") as handle:
        pickle.dump(forcefield, handle, protocol=pickle.HIGHEST_PROTOCOL)

    class _DummyTrajectory:
        def __len__(self):
            return 2

    class _DummyUniverse:
        def __init__(self):
            self.trajectory = _DummyTrajectory()

        def select_atoms(self, sel):
            assert sel == "all"
            return SimpleNamespace(indices=np.array([0, 1], dtype=np.int64))

    monkeypatch.setattr("MDAnalysis.Universe", lambda *args, **kwargs: _DummyUniverse())
    monkeypatch.setattr(
        "AceCG.topology.topology_array.collect_topology_arrays",
        lambda *args, **kwargs: topo,
    )

    frames_with_forces = [
        (*frame, None) for frame in _frames()
    ]

    def fake_iter_frames(universe, *, start=0, end=None, every=1, include_forces=False):
        del universe, start, end, every, include_forces
        for item in frames_with_forces:
            yield item

    monkeypatch.setattr("AceCG.io.trajectory.iter_frames", fake_iter_frames)

    spec = {
        "post_mode": "one_pass",
        "work_dir": str(tmp_path),
        "forcefield_path": str(ff_path),
        "topology": "topology.data",
        "trajectory": ["traj.lammpstrj"],
        "trajectory_format": "LAMMPSDUMP",
        "steps": [
            {
                "step_mode": "cdrem",
                "name": "cdrem-alias",
                "need_hessian": True,
                "output_file": str(tmp_path / "cdrem.pkl"),
            }
        ],
    }
    build_default_engine().run_post(spec)
    with (tmp_path / "cdrem.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    assert "energy_grad_avg" in payload
    assert "d2U_avg" in payload
    assert "grad_outer_avg" in payload
