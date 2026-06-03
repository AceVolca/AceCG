"""Regression tests for matrix-path potential gradients and force dispatch."""

from __future__ import annotations

import numpy as np

from AceCG.compute.energy import energy
from AceCG.compute.force import force
from AceCG.compute.frame_geometry import FrameGeometry
from AceCG.potentials.base import BasePotential
from AceCG.potentials.bspline import BSplinePotential
from AceCG.potentials.multi_gaussian import MultiGaussianPotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey


PAIR_KEY = InteractionKey.pair("A", "B")


def _dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def _named_channel_stack(potential, names: list[str], values: np.ndarray) -> np.ndarray:
    r_flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.column_stack(
        [
            np.asarray(getattr(potential, name)(r_flat), dtype=np.float64).reshape(-1)
            for name in names
        ]
    )


def _finite_difference_force_grad(potential, values: np.ndarray) -> np.ndarray:
    r_arr = np.asarray(values, dtype=np.float64)
    params0 = potential.get_params()
    scale = np.maximum(1.0, np.abs(params0))
    jac = np.empty(r_arr.shape + (params0.size,), dtype=np.float64)
    try:
        for idx in range(params0.size):
            step = 1.0e-6 * scale[idx]
            delta = np.zeros_like(params0)
            delta[idx] = step
            potential.set_params(params0 + delta)
            plus = np.asarray(potential.force(r_arr), dtype=np.float64)
            potential.set_params(params0 - delta)
            minus = np.asarray(potential.force(r_arr), dtype=np.float64)
            jac[..., idx] = (plus - minus) / (2.0 * step)
    finally:
        potential.set_params(params0)
    return jac


def _pair_geometry(distance: float) -> FrameGeometry:
    positions = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=np.float64)
    return FrameGeometry(
        pair_distances={PAIR_KEY: np.array([distance], dtype=np.float64)},
        pair_indices={PAIR_KEY: (np.array([0], dtype=np.int64), np.array([1], dtype=np.int64))},
        pair_vectors={PAIR_KEY: np.array([[distance, 0.0, 0.0]], dtype=np.float64)},
        bond_distances={},
        bond_vectors={},
        bond_indices={},
        angle_values={},
        angle_indices={},
        dihedral_values={},
        dihedral_indices={},
        positions=positions,
        box=np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64),
        n_atoms=2,
        real_site_indices=None,
    )


def _pair_geometry_batch(distances: np.ndarray) -> FrameGeometry:
    distance_values = np.asarray(distances, dtype=np.float64).reshape(-1)
    positions = np.zeros((distance_values.size, 2, 3), dtype=np.float64)
    positions[:, 1, 0] = distance_values
    pair_vectors = np.zeros((distance_values.size, 1, 3), dtype=np.float64)
    pair_vectors[:, 0, 0] = distance_values
    return FrameGeometry(
        pair_distances={PAIR_KEY: distance_values.reshape(-1, 1)},
        pair_indices={PAIR_KEY: (np.array([0], dtype=np.int64), np.array([1], dtype=np.int64))},
        pair_vectors={PAIR_KEY: pair_vectors},
        bond_distances={},
        bond_vectors={},
        bond_indices={},
        angle_values={},
        angle_indices={},
        dihedral_values={},
        dihedral_indices={},
        positions=positions,
        box=np.broadcast_to(
            np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64),
            (distance_values.size, 6),
        ).copy(),
        n_atoms=2,
        real_site_indices=None,
    )


def _pair_geometry_batch_terms(distances: np.ndarray) -> FrameGeometry:
    distance_values = np.asarray(distances, dtype=np.float64)
    if distance_values.ndim != 2:
        raise ValueError("distances must be shaped (n_samples, n_terms)")
    n_samples, n_terms = distance_values.shape
    positions = np.zeros((n_samples, n_terms + 1, 3), dtype=np.float64)
    positions[:, 1:, 0] = distance_values
    pair_vectors = np.zeros(distance_values.shape + (3,), dtype=np.float64)
    pair_vectors[..., 0] = distance_values
    return FrameGeometry(
        pair_distances={PAIR_KEY: distance_values},
        pair_indices={
            PAIR_KEY: (
                np.zeros(n_terms, dtype=np.int64),
                np.arange(1, n_terms + 1, dtype=np.int64),
            )
        },
        pair_vectors={PAIR_KEY: pair_vectors},
        bond_distances={},
        bond_vectors={},
        bond_indices={},
        angle_values={},
        angle_indices={},
        dihedral_values={},
        dihedral_indices={},
        positions=positions,
        box=np.broadcast_to(
            np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0], dtype=np.float64),
            (n_samples, 6),
        ).copy(),
        n_atoms=n_terms + 1,
        real_site_indices=None,
    )


class _ForceGradOnlyPotential(BasePotential):
    def __init__(self):
        super().__init__()
        self.typ1 = "A"
        self.typ2 = "B"
        self.cutoff = 10.0
        self._params = np.array([1.0, 2.0], dtype=np.float64)
        self._param_names = ["p0", "p1"]
        self._dparam_names = ["dp0", "dp1"]
        self._d2param_names = [["zero", "zero"], ["zero", "zero"]]

    def value(self, r: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(r, dtype=np.float64))

    def force(self, r: np.ndarray) -> np.ndarray:
        r_arr = np.asarray(r, dtype=np.float64)
        return np.ones_like(r_arr)

    def force_grad(self, r: np.ndarray) -> np.ndarray:
        r_arr = np.asarray(r, dtype=np.float64).reshape(-1)
        return np.tile(np.array([[2.0, 3.0]], dtype=np.float64), (r_arr.size, 1))

    def basis_values(self, r: np.ndarray) -> np.ndarray:
        raise AssertionError("force kernel must not call basis_values()")

    def is_param_linear(self) -> np.ndarray:
        return np.array([True, True], dtype=bool)


class _EnergyGradSumOnlyPotential(BasePotential):
    def __init__(self):
        super().__init__()
        self.typ1 = "A"
        self.typ2 = "B"
        self.cutoff = 10.0
        self._params = np.array([1.0, 2.0], dtype=np.float64)
        self._param_names = ["p0", "p1"]
        self._dparam_names = ["dp0", "dp1"]
        self._d2param_names = [["boom", "boom"], ["boom", "boom"]]

    def value(self, r: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(r, dtype=np.float64))

    def force(self, r: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(r, dtype=np.float64))

    def energy_grad(self, r: np.ndarray) -> np.ndarray:
        raise AssertionError("energy kernel must not call energy_grad()")

    def energy_grad_sum(self, r: np.ndarray) -> np.ndarray:
        r_arr = np.asarray(r, dtype=np.float64).reshape(-1)
        return np.array([2.0 * r_arr.size, 3.0 * r_arr.size], dtype=np.float64)

    def boom(self, r: np.ndarray) -> np.ndarray:
        raise AssertionError("energy kernel must skip Hessian dispatch for fully linear potentials")

    def is_param_linear(self) -> np.ndarray:
        return np.array([True, True], dtype=bool)


def test_bspline_matrix_overrides_match_named_channels():
    potential = BSplinePotential.from_range(
        "A",
        "B",
        minimum=3.0,
        maximum=8.0,
        resolution=0.5,
        degree=3,
    )
    potential.set_params(np.linspace(0.1, 1.0, potential.n_params()))
    distances = np.linspace(3.1, 7.9, 9, dtype=np.float64)

    expected_energy = _named_channel_stack(potential, potential.dparam_names(), distances)
    actual_energy = np.asarray(potential.energy_grad(distances), dtype=np.float64)
    np.testing.assert_allclose(actual_energy, expected_energy, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        potential.energy_grad_sum(distances),
        expected_energy.sum(axis=0),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    expected_force = _named_channel_stack(potential, potential.df_dparam_names(), distances)
    actual_force = _dense(potential.force_grad(distances))
    np.testing.assert_allclose(actual_force, expected_force, rtol=1.0e-12, atol=1.0e-12)


def test_bspline_sparse_force_grad_survives_cutoff_row_zeroing():
    potential = BSplinePotential.from_range(
        "A",
        "B",
        minimum=3.0,
        maximum=8.0,
        resolution=0.5,
        degree=3,
    )
    force_grad = potential.force_grad(np.array([4.0, 9.0], dtype=np.float64))

    assert hasattr(force_grad, "tocoo")
    dense = _dense(force_grad)
    assert np.any(np.abs(dense[0]) > 0.0)
    np.testing.assert_allclose(dense[1], 0.0)


def test_batched_force_geometry_uses_bspline_sparse_pair_path_like_single_frames():
    potential = BSplinePotential.from_range(
        "A",
        "B",
        minimum=3.0,
        maximum=8.0,
        resolution=0.5,
        degree=3,
    )
    potential.set_params(np.linspace(0.1, 1.0, potential.n_params()))
    forcefield = Forcefield({PAIR_KEY: [potential]})
    distances = np.array([4.0, 4.5], dtype=np.float64)
    weights = np.array([0.4, 0.6], dtype=np.float64)
    reference_forces = np.zeros((distances.size, 6), dtype=np.float64)

    batch_result = force(
        _pair_geometry_batch(distances),
        forcefield,
        return_value=True,
        return_grad=True,
        return_fm_stats=True,
        reference_force=reference_forces,
        frame_weights=weights,
    )
    single_results = [
        force(
            _pair_geometry(float(distance)),
            forcefield,
            return_value=True,
            return_grad=True,
            return_fm_stats=True,
            reference_force=reference_forces[index],
            frame_weight=float(weights[index]),
        )
        for index, distance in enumerate(distances)
    ]

    np.testing.assert_allclose(
        batch_result["force"],
        np.stack([item["force"] for item in single_results]),
    )
    np.testing.assert_allclose(
        batch_result["force_grad"],
        np.stack([item["force_grad"] for item in single_results]),
    )
    partial = batch_result["fm_stats_sum"]
    sparse_partial = force(
        _pair_geometry_batch(distances),
        forcefield,
        return_fm_stats=True,
        reference_force=reference_forces,
        frame_weights=weights,
    )["fm_stats_sum"]
    for key in ("JtJ", "Jtf", "Jty"):
        np.testing.assert_allclose(
            partial[key],
            sum(item["fm_stats"][key] for item in single_results),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(sparse_partial[key], partial[key], rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(
        partial["ftf"],
        sum(item["fm_stats"]["ftf"] for item in single_results),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        partial["fTy"],
        sum(item["fm_stats"]["fTy"] for item in single_results),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        partial["yty"],
        sum(item["fm_stats"]["yty"] for item in single_results),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    for key in ("ftf", "fTy", "yty", "weight_sum"):
        np.testing.assert_allclose(sparse_partial[key], partial[key], rtol=1.0e-6, atol=1.0e-7)


def test_multigaussian_matrix_overrides_match_reference_with_100_peaks():
    rng = np.random.default_rng(20260409)
    n_gauss = 100
    params = np.empty(3 * n_gauss, dtype=np.float64)
    params[0::3] = rng.normal(loc=0.0, scale=0.7, size=n_gauss)
    params[1::3] = np.linspace(3.0, 23.0, n_gauss) + rng.normal(loc=0.0, scale=0.05, size=n_gauss)
    params[2::3] = rng.uniform(0.25, 1.25, size=n_gauss)
    potential = MultiGaussianPotential(
        "A",
        "B",
        n_gauss=n_gauss,
        cutoff=30.0,
        init_params=params,
    )
    distances = np.linspace(3.5, 22.5, 11, dtype=np.float64)

    expected_energy = _named_channel_stack(potential, potential.dparam_names(), distances)
    actual_energy = np.asarray(potential.energy_grad(distances), dtype=np.float64)
    np.testing.assert_allclose(actual_energy, expected_energy, rtol=1.0e-12, atol=1.0e-12)

    expected_force = _finite_difference_force_grad(potential, distances)
    actual_force = np.asarray(potential.force_grad(distances), dtype=np.float64)
    np.testing.assert_allclose(actual_force, expected_force, rtol=5.0e-6, atol=5.0e-8)


def test_force_kernel_uses_force_grad_contract_only():
    geometry = _pair_geometry(4.0)
    forcefield = Forcefield({PAIR_KEY: [_ForceGradOnlyPotential()]})

    result = force(
        geometry,
        forcefield,
        return_grad=True,
        return_fm_stats=True,
        reference_force=np.zeros(6, dtype=np.float64),
    )

    assert result["force_grad"].shape == (6, 2)
    assert result["fm_stats"]["JtJ"].shape == (2, 2)


def test_fm_stats_only_preserves_dense_force_grad_potential_path():
    distances = np.array([4.0, 5.0], dtype=np.float64)
    weights = np.array([0.25, 0.75], dtype=np.float64)
    reference_forces = np.zeros((2, 6), dtype=np.float64)
    forcefield = Forcefield({PAIR_KEY: [_ForceGradOnlyPotential()]})

    dense = force(
        _pair_geometry_batch(distances),
        forcefield,
        return_grad=True,
        return_fm_stats=True,
        reference_force=reference_forces,
        frame_weights=weights,
    )["fm_stats_sum"]
    sparse = force(
        _pair_geometry_batch(distances),
        forcefield,
        return_fm_stats=True,
        reference_force=reference_forces,
        frame_weights=weights,
    )["fm_stats_sum"]

    for key in ("JtJ", "Jtf", "Jty", "ftf", "fTy", "yty"):
        np.testing.assert_allclose(sparse[key], dense[key], rtol=1.0e-6, atol=1.0e-7)


def test_energy_kernel_uses_energy_grad_sum_and_skips_linear_hessian_dispatch():
    geometry = _pair_geometry(4.0)
    forcefield = Forcefield({PAIR_KEY: [_EnergyGradSumOnlyPotential()]})

    result = energy(
        geometry,
        forcefield,
        return_grad=True,
        return_hessian=True,
    )

    np.testing.assert_allclose(result["energy_grad"], np.array([2.0, 3.0], dtype=np.float64))
    np.testing.assert_allclose(result["energy_hessian"], np.zeros((2, 2), dtype=np.float64))


def test_batched_energy_kernel_uses_reduced_grad_sum_by_sample():
    distances = np.array([4.0, 11.0], dtype=np.float64)
    forcefield = Forcefield({PAIR_KEY: [_EnergyGradSumOnlyPotential()]})

    result = energy(
        _pair_geometry_batch(distances),
        forcefield,
        return_grad=True,
        return_grad_outer=True,
    )

    expected = np.array([[2.0, 3.0], [0.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(result["energy_grad"], expected)
    np.testing.assert_allclose(
        result["energy_grad_outer"],
        np.einsum("bi,bj->bij", expected, expected),
    )


def test_bspline_batched_energy_grad_sum_matches_single_frame_reductions():
    potential = BSplinePotential.from_range(
        "A",
        "B",
        minimum=3.0,
        maximum=8.0,
        resolution=0.5,
        degree=3,
    )
    potential.set_params(np.linspace(0.1, 1.0, potential.n_params()))
    distances = np.array(
        [
            [4.0, 4.5, 9.0],
            [3.5, 7.5, 4.2],
        ],
        dtype=np.float64,
    )
    expected = np.stack(
        [
            potential.energy_grad_sum(row[row <= potential.cutoff])
            for row in distances
        ],
        axis=0,
    )

    np.testing.assert_allclose(
        potential.energy_grad_sum(distances),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    result = energy(
        _pair_geometry_batch_terms(distances),
        Forcefield({PAIR_KEY: [potential]}),
        return_grad=True,
    )
    np.testing.assert_allclose(result["energy_grad"], expected, rtol=1.0e-12, atol=1.0e-12)
