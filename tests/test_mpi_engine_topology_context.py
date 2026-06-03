from __future__ import annotations

import pickle
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pytest

from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.compute.mpi_engine import _selected_frame_ids_from_spec
from AceCG.compute.mpi_engine import MPIComputeEngine, build_default_engine
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.topology_array import collect_topology_arrays
from AceCG.topology.types import InteractionKey


def _make_lammps_like_universe() -> mda.Universe:
    u = mda.Universe.empty(
        3,
        n_residues=1,
        atom_resindex=np.array([0, 0, 0], dtype=np.int64),
        trajectory=True,
    )
    u.add_TopologyAttr("types", np.array(["1", "2", "3"], dtype=object))
    u.add_TopologyAttr("masses", np.array([1.0, 2.0, 0.0], dtype=np.float64))
    u.add_TopologyAttr("charges", np.zeros(3, dtype=np.float64))
    u.add_TopologyAttr("resids", np.array([1], dtype=np.int64))
    u.add_TopologyAttr("bonds", np.array([[0, 1], [1, 2]], dtype=np.int64))
    u.add_TopologyAttr("angles", np.array([[0, 1, 2]], dtype=np.int64))
    u.atoms.positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    u.dimensions = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0], dtype=np.float32)
    return u


def _test_forcefield() -> Forcefield:
    bond_key = InteractionKey.bond("A", "B")
    return Forcefield({bond_key: [HarmonicPotential("A", "B", k=5.0, r0=4.0)]})


def test_selected_frame_ids_subsamples_after_stride_selection():
    spec = {
        "frame_start": 0,
        "frame_end": 7500,
        "every": 5,
        "noise": {
            "subsample_per_epoch": 640,
            "seed": 20260502,
        },
    }

    selected = _selected_frame_ids_from_spec(spec, total_frames=7500)
    selected_again = _selected_frame_ids_from_spec(spec, total_frames=7500)

    assert selected == selected_again
    assert len(selected) == 640
    assert len(set(selected)) == 640
    assert selected == sorted(selected)
    assert all(frame_id % 5 == 0 for frame_id in selected)
    assert min(selected) >= 0
    assert max(selected) < 7500


def test_selected_frame_ids_caps_subsample_at_dataset_size():
    spec = {
        "frame_start": 10,
        "frame_end": 40,
        "every": 5,
        "noise": {
            "subsample_per_epoch": 99,
            "seed": 1,
        },
    }

    assert _selected_frame_ids_from_spec(spec, total_frames=100) == [
        10,
        15,
        20,
        25,
        30,
        35,
    ]


def test_selected_frame_ids_prefers_subsample_seed():
    base_spec = {
        "frame_start": 0,
        "frame_end": 100,
        "every": 1,
        "noise": {
            "subsample_per_epoch": 10,
            "seed": 3,
        },
    }
    with_subsample_seed = {
        **base_spec,
        "noise": {
            **base_spec["noise"],
            "subsample_seed": 4,
        },
    }

    assert _selected_frame_ids_from_spec(base_spec, total_frames=100) != (
        _selected_frame_ids_from_spec(with_subsample_seed, total_frames=100)
    )


def test_add_noise_dsm_target_builds_synthetic_reference_force():
    engine = MPIComputeEngine()
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        dtype=np.float64,
    )
    box = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0], dtype=np.float64)
    beta = 2.5
    sigma = 0.2

    frame_batch, weights = engine.add_noise(
        (7, positions, box, None),
        {
            "target": "dsm",
            "samples_per_frame": 2,
            "sigma": sigma,
            "beta": beta,
            "seed": 11,
            "include_original": True,
        },
        SimpleNamespace(real_site_indices=None),
    )

    frame_ids, noisy_positions, box_batch, reference_force = frame_batch
    displacement = noisy_positions - positions[None, :, :]
    np.testing.assert_array_equal(frame_ids, np.array([7, 7, 7], dtype=np.int64))
    np.testing.assert_allclose(box_batch, np.broadcast_to(box, (3, 6)))
    np.testing.assert_allclose(weights, np.full(3, 1.0 / 3.0))
    np.testing.assert_allclose(reference_force, -displacement / (beta * sigma * sigma))


def test_add_noise_force_mix_preserves_original_and_mixes_noisy_targets():
    engine = MPIComputeEngine()
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        dtype=np.float64,
    )
    reference_force = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float64,
    )
    box = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0], dtype=np.float64)
    beta = 2.5
    sigma = 0.2
    mix_ratio = 0.25

    frame_batch, _ = engine.add_noise(
        (7, positions, box, reference_force),
        {
            "samples_per_frame": 2,
            "sigma": sigma,
            "beta": beta,
            "force_mix_ratio": mix_ratio,
            "seed": 11,
            "include_original": True,
        },
        SimpleNamespace(real_site_indices=None),
    )

    _, noisy_positions, _, mixed_force = frame_batch
    displacement = noisy_positions - positions[None, :, :]
    score_force = -displacement / (beta * sigma * sigma)
    expected = np.broadcast_to(reference_force, mixed_force.shape).astype(np.float32).copy()
    expected[1:] = (1.0 - mix_ratio) * expected[1:] + mix_ratio * score_force[1:]

    np.testing.assert_allclose(mixed_force[0], reference_force)
    np.testing.assert_allclose(mixed_force, expected)


def test_add_noise_force_mix_rejects_missing_beta():
    engine = MPIComputeEngine()
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        dtype=np.float64,
    )
    reference_force = np.zeros((2, 3), dtype=np.float64)
    box = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0], dtype=np.float64)

    with pytest.raises(ValueError, match="requires beta"):
        engine.add_noise(
            (7, positions, box, reference_force),
            {
                "samples_per_frame": 1,
                "sigma": 0.2,
                "force_mix_ratio": 0.25,
            },
            SimpleNamespace(real_site_indices=None),
        )
    np.testing.assert_allclose(reference_force[0], 0.0)


def test_collect_topology_arrays_normalizes_aliases_and_exclusions():
    u = _make_lammps_like_universe()

    arrays = collect_topology_arrays(
        u,
        exclude_bonded="100",
        exclude_option="resid",
        atom_type_name_aliases={"1": "A", "2": "B", "3": "VP"},
        vp_names=("VP",),
    )

    assert arrays.names.tolist() == ["A", "B", "VP"]
    assert arrays.exclude_12.shape == (2, 2)
    assert arrays.exclude_13.shape == (0, 2)
    assert arrays.exclude_14.shape == (0, 2)
    assert arrays.excluded_nb_mode == "resid"
    assert arrays.excluded_nb.shape[0] == 0
    assert arrays.excluded_nb_all is True
    np.testing.assert_array_equal(
        arrays.virtual_site_indices,
        np.array([2], dtype=np.int64),
    )


def test_run_post_rem_forwards_topology_context_and_writes_pickle(monkeypatch, tmp_path):
    ff_path = tmp_path / "ff.pkl"
    with ff_path.open("wb") as handle:
        pickle.dump(_test_forcefield(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    captured = {}

    class DummyTrajectory:
        def __len__(self):
            return 2

    class DummyUniverse:
        def __init__(self):
            self.trajectory = DummyTrajectory()

        def select_atoms(self, sel):
            captured["sel"] = sel
            return SimpleNamespace(indices=np.array([0, 1], dtype=np.int64))

    monkeypatch.setattr(mda, "Universe", lambda *args, **kwargs: DummyUniverse())

    def fake_collect_topology_arrays(universe, *, exclude_bonded, exclude_option, atom_type_name_aliases, vp_names):
        captured["exclude_bonded"] = exclude_bonded
        captured["exclude_option"] = exclude_option
        captured["atom_type_name_aliases"] = atom_type_name_aliases
        captured["vp_names"] = vp_names
        return SimpleNamespace(real_site_indices=np.array([0, 1], dtype=np.int64))

    def fake_iter_frames(universe, *, start, end, every=1, include_forces):
        captured["include_forces"] = include_forces
        yield (
            0,
            np.zeros((2, 3), dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            None,
        )

    def fake_compute_frame_geometry(*args, **kwargs):
        return SimpleNamespace(
            n_atoms=2,
            real_site_indices=np.array([0, 1], dtype=np.int64),
            box=np.zeros(6, dtype=np.float64),
            positions=np.zeros((2, 3), dtype=np.float64),
        )

    def fake_energy(geom, ff, **kwargs):
        del geom, ff, kwargs
        grad = np.array([2.0, 4.0], dtype=np.float64)
        return {"energy_grad": grad, "gauge_free_energy_grad": grad}

    monkeypatch.setattr(
        "AceCG.topology.topology_array.collect_topology_arrays",
        fake_collect_topology_arrays,
    )
    monkeypatch.setattr(
        "AceCG.io.trajectory.iter_frames",
        fake_iter_frames,
    )
    monkeypatch.setattr(
        "AceCG.compute.mpi_engine.compute_frame_geometry",
        fake_compute_frame_geometry,
    )
    monkeypatch.setattr(
        "AceCG.compute.mpi_engine.energy",
        fake_energy,
    )

    spec = {
        "post_mode": "one_pass",
        "work_dir": str(tmp_path),
        "forcefield_path": str(ff_path),
        "topology": "top.data",
        "trajectory": "traj.lammpstrj",
        "cutoff": 12.0,
        "sel": "all",
        "exclude_option": "resid",
        "exclude_bonded": "101",
        "atom_type_name_aliases": {"1": "A", "2": "B"},
        "vp_names": ["VP"],
        "frame_weight": [2.0, 2.0],
        "steps": [{"step_mode": "rem", "output_file": "energy_grad.pkl"}],
    }

    build_default_engine().run_post(spec)

    assert captured["sel"] == "all"
    assert captured["exclude_bonded"] == "101"
    assert captured["exclude_option"] == "resid"
    assert captured["atom_type_name_aliases"] == {"1": "A", "2": "B"}
    assert captured["vp_names"] == ["VP"]
    assert captured["include_forces"] is False

    with (tmp_path / "energy_grad.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    np.testing.assert_allclose(payload["energy_grad_avg"], np.array([2.0, 4.0], dtype=np.float64))


def test_run_post_rem_reads_frame_weight_file(monkeypatch, tmp_path):
    ff_path = tmp_path / "ff.pkl"
    with ff_path.open("wb") as handle:
        pickle.dump(_test_forcefield(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    weight_path = tmp_path / "weights.npy"
    np.save(weight_path, np.array([1.0, 3.0], dtype=np.float64))

    class DummyTrajectory:
        def __len__(self):
            return 2

    class DummyUniverse:
        def __init__(self):
            self.trajectory = DummyTrajectory()

        def select_atoms(self, sel):
            return SimpleNamespace(indices=np.array([0, 1], dtype=np.int64))

    monkeypatch.setattr(mda, "Universe", lambda *args, **kwargs: DummyUniverse())
    monkeypatch.setattr(
        "AceCG.topology.topology_array.collect_topology_arrays",
        lambda *args, **kwargs: SimpleNamespace(real_site_indices=np.array([0, 1], dtype=np.int64)),
    )

    def fake_iter_frames(universe, *, start, end, every=1, include_forces):
        del universe, include_forces
        for frame_id in range(start, end, every):
            yield (
                frame_id,
                np.full((2, 3), float(frame_id), dtype=np.float64),
                np.zeros(6, dtype=np.float64),
                None,
            )

    def fake_compute_frame_geometry(positions, *args, **kwargs):
        return SimpleNamespace(
            marker=float(positions[0, 0]),
            n_atoms=2,
            real_site_indices=np.array([0, 1], dtype=np.int64),
            box=np.zeros(6, dtype=np.float64),
            positions=positions,
        )

    def fake_energy(geom, ff, **kwargs):
        del ff, kwargs
        base = 1.0 + 2.0 * float(geom.marker)
        grad = np.array([base, 2.0 * base], dtype=np.float64)
        return {"energy_grad": grad, "gauge_free_energy_grad": grad}

    monkeypatch.setattr("AceCG.io.trajectory.iter_frames", fake_iter_frames)
    monkeypatch.setattr(
        "AceCG.compute.mpi_engine.compute_frame_geometry",
        fake_compute_frame_geometry,
    )
    monkeypatch.setattr("AceCG.compute.mpi_engine.energy", fake_energy)

    spec = {
        "post_mode": "one_pass",
        "work_dir": str(tmp_path),
        "forcefield_path": str(ff_path),
        "topology": "top.data",
        "trajectory": "traj.lammpstrj",
        "frame_weight_file": weight_path.name,
        "steps": [{"step_mode": "rem", "output_file": "energy_grad.pkl"}],
    }

    build_default_engine().run_post(spec)

    with (tmp_path / "energy_grad.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    np.testing.assert_allclose(payload["energy_grad_avg"], np.array([2.5, 5.0], dtype=np.float64))
    assert payload["weight_sum"] == pytest.approx(4.0)


def test_multi_run_post_reads_trajectory_once(monkeypatch, tmp_path):
    ff_path = tmp_path / "ff.pkl"
    with ff_path.open("wb") as handle:
        pickle.dump(_test_forcefield(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    calls = {"iter_frames": 0}

    class DummyTrajectory:
        def __len__(self):
            return 4

    class DummyUniverse:
        def __init__(self):
            self.trajectory = DummyTrajectory()

        def select_atoms(self, sel):
            return SimpleNamespace(indices=np.array([0, 1], dtype=np.int64))

    monkeypatch.setattr(mda, "Universe", lambda *args, **kwargs: DummyUniverse())
    monkeypatch.setattr(
        "AceCG.topology.topology_array.collect_topology_arrays",
        lambda *args, **kwargs: SimpleNamespace(real_site_indices=np.array([0, 1], dtype=np.int64)),
    )

    def fake_iter_frames(universe, *, start, end, every=1, include_forces):
        calls["iter_frames"] += 1
        for frame_id in range(start, end, every):
            yield (
                frame_id,
                np.zeros((2, 3), dtype=np.float64),
                np.zeros(6, dtype=np.float64),
                np.zeros(6, dtype=np.float64),
            )

    def fake_compute_frame_geometry(*args, **kwargs):
        return SimpleNamespace(
            n_atoms=2,
            real_site_indices=np.array([0, 1], dtype=np.int64),
            box=np.zeros(6, dtype=np.float64),
            positions=np.zeros((2, 3), dtype=np.float64),
        )

    def fake_energy(geom, ff, **kwargs):
        grad = np.zeros(2, dtype=np.float64)
        out = {"energy_grad": grad, "gauge_free_energy_grad": grad}
        if kwargs.get("return_hessian"):
            out["energy_hessian"] = np.zeros((2, 2), dtype=np.float64)
        if kwargs.get("return_grad_outer"):
            out["energy_grad_outer"] = np.zeros((2, 2), dtype=np.float64)
        if kwargs.get("return_gauge_free_energy_grad_outer"):
            out["gauge_free_energy_grad_outer"] = np.zeros((2, 2), dtype=np.float64)
        return out

    def fake_force(geom, ff, **kwargs):
        return {
            "fm_stats": {
                "JtJ": np.zeros((2, 2), dtype=np.float64),
                "Jty": np.zeros(2, dtype=np.float64),
                "yty": 0.0,
                "Jtf": np.zeros(2, dtype=np.float64),
                "ftf": 0.0,
                "fTy": 0.0,
                "n_frames": 1,
                "weight_sum": 1.0,
                "n_atoms_obs": 2,
            }
        }

    monkeypatch.setattr("AceCG.io.trajectory.iter_frames", fake_iter_frames)
    monkeypatch.setattr(
        "AceCG.compute.mpi_engine.compute_frame_geometry",
        fake_compute_frame_geometry,
    )
    monkeypatch.setattr(
        "AceCG.compute.mpi_engine.energy",
        fake_energy,
    )
    monkeypatch.setattr(
        "AceCG.compute.mpi_engine.force",
        fake_force,
    )

    spec = {
        "post_mode": "one_pass",
        "work_dir": str(tmp_path),
        "forcefield_path": str(ff_path),
        "topology": "top.data",
        "trajectory": "traj.lammpstrj",
        "exclude_bonded": "111",
        "atom_type_name_aliases": {"1": "A"},
        "steps": [
            {"step_mode": "rem", "output_file": "rem.pkl"},
            {"step_mode": "fm", "output_file": "fm.pkl"},
        ],
    }

    build_default_engine().run_post(spec)

    assert calls["iter_frames"] == 1
    assert (tmp_path / "rem.pkl").exists()
    assert (tmp_path / "fm.pkl").exists()


def test_run_post_rejects_negative_frame_weight(monkeypatch, tmp_path):
    ff_path = tmp_path / "ff.pkl"
    with ff_path.open("wb") as handle:
        pickle.dump({}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    class DummyTrajectory:
        def __len__(self):
            return 2

    class DummyUniverse:
        def __init__(self):
            self.trajectory = DummyTrajectory()

        def select_atoms(self, sel):
            return SimpleNamespace(indices=np.array([0, 1], dtype=np.int64))

    monkeypatch.setattr(mda, "Universe", lambda *args, **kwargs: DummyUniverse())
    monkeypatch.setattr(
        "AceCG.topology.topology_array.collect_topology_arrays",
        lambda *args, **kwargs: SimpleNamespace(real_site_indices=np.array([0, 1], dtype=np.int64)),
    )

    spec = {
        "post_mode": "one_pass",
        "work_dir": str(tmp_path),
        "forcefield_path": str(ff_path),
        "topology": "top.data",
        "trajectory": "traj.lammpstrj",
        "frame_weight": [1.0, -1.0],
        "steps": [{"step_mode": "rem", "output_file": "energy_grad.pkl"}],
    }

    with pytest.raises(ValueError, match="frame_weight must be nonnegative with positive sum"):
        build_default_engine().run_post(spec)
