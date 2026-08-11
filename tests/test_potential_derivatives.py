"""Tests for the Phase U potential derivative API."""

import numpy as np
import pytest

from AceCG.potentials import POTENTIAL_REGISTRY
from AceCG.potentials.base import BasePotential
from AceCG.potentials.bspline import BSplinePotential
from AceCG.potentials.dihedral_bspline import DihedralBSplinePotential
from AceCG.potentials.gaussian import GaussianPotential
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.potentials.lennardjones import LennardJonesPotential
from AceCG.potentials.lennardjones96 import LennardJones96Potential
from AceCG.potentials.lennardjones_soft import LennardJonesSoftPotential
from AceCG.potentials.multi_gaussian import MultiGaussianPotential
from AceCG.potentials.soft import SoftPotential
from AceCG.potentials.srlrgaussian import SRLRGaussianPotential
from AceCG.potentials.unnormalized_multi_gaussian import (
    UnnormalizedMultiGaussianPotential,
)


def build_bspline():
    pot = BSplinePotential(
        "A",
        "B",
        knots=np.array([0.0, 0.0, 2.0, 2.0], dtype=float),
        coefficients=np.array([1.5, -0.5], dtype=float),
        degree=1,
        cutoff=2.0,
    )
    return pot


def build_bspline_minimum_gauge():
    return BSplinePotential(
        "A",
        "B",
        knots=np.array([0.0, 0.0, 2.0, 2.0], dtype=float),
        coefficients=np.array([1.0, -1.0], dtype=float),
        degree=1,
        cutoff=2.0,
        bonded=True,
    )


def build_bspline_off_grid_minimum_gauge():
    return BSplinePotential(
        "A",
        "B",
        knots=np.array([0.0, 0.0, 2.0, 2.0], dtype=float),
        coefficients=np.array([0.7, -1.3], dtype=float),
        degree=1,
        cutoff=2.0,
        bonded=True,
    )


def build_dihedral_periodic():
    return DihedralBSplinePotential(
        ("A", "B", "C", "D"),
        minimum=-180.0,
        maximum=180.0,
        coefficients=np.array([0.7, -1.3, 0.4, 0.9, -0.2, 1.1], dtype=float),
        degree=3,
        boundary_mode="periodic",
    )


def build_dihedral_cutoff():
    return DihedralBSplinePotential(
        ("A", "B", "C", "D"),
        minimum=-120.0,
        maximum=100.0,
        coefficients=np.array([0.5, -0.8, 0.3, 1.2, -0.6], dtype=float),
        degree=3,
        boundary_mode="cutoff",
    )


def build_potential_cases():
    return [
        ("bspline", build_bspline(), np.array([True, True], dtype=bool)),
        (
            "dihedral_bspline_periodic",
            build_dihedral_periodic(),
            np.ones(6, dtype=bool),
        ),
        (
            "dihedral_bspline_cutoff",
            build_dihedral_cutoff(),
            np.ones(5, dtype=bool),
        ),
        (
            "harmonic",
            HarmonicPotential("A", "B", k=2.0, r0=1.25),
            np.array([True, False], dtype=bool),
        ),
        (
            "gaussian",
            GaussianPotential("A", "B", A=1.5, r0=1.0, sigma=0.4, cutoff=5.0),
            np.array([True, False, False], dtype=bool),
        ),
        (
            "multi_gaussian",
            MultiGaussianPotential(
                "A",
                "B",
                n_gauss=2,
                cutoff=5.0,
                init_params=np.array([1.0, 0.8, 0.3, -0.5, 1.4, 0.6], dtype=float),
            ),
            np.array([True, False, False, True, False, False], dtype=bool),
        ),
        (
            "unnormalized_multi_gaussian",
            UnnormalizedMultiGaussianPotential(
                "A",
                "B",
                n_gauss=2,
                cutoff=5.0,
                init_params=np.array([0.8, 0.7, 0.4, -1.1, 1.2, 0.5], dtype=float),
            ),
            np.array([True, False, False, True, False, False], dtype=bool),
        ),
        (
            "srlr_gaussian",
            SRLRGaussianPotential("A", "B", A=1.0, B=0.4, C=0.5, D=0.9, cutoff=5.0),
            np.array([True, False, True, False], dtype=bool),
        ),
        (
            "lennard_jones",
            LennardJonesPotential("A", "B", epsilon=0.8, sigma=1.1, cutoff=5.0),
            np.array([True, False], dtype=bool),
        ),
        (
            "lennard_jones96",
            LennardJones96Potential("A", "B", epsilon=0.8, sigma=1.1, cutoff=5.0),
            np.array([True, False], dtype=bool),
        ),
        (
            "lennard_jones_soft",
            LennardJonesSoftPotential(
                "A",
                "B",
                epsilon=0.8,
                sigma=1.1,
                lam=0.6,
                cutoff=5.0,
                n=2,
                alpha_LJ=0.5,
            ),
            np.array([True, False, False], dtype=bool),
        ),
        (
            "soft",
            SoftPotential("A", "B", A=25.0, cutoff=5.0),
            np.array([True, False], dtype=bool),
        ),
    ]


def finite_difference_force_grad(potential, r, rel_step=1.0e-6):
    r_arr = np.asarray(r, dtype=float)
    params0 = potential.get_params()
    scale = np.maximum(1.0, np.abs(params0))
    numeric = np.empty(r_arr.shape + (params0.size,), dtype=float)
    try:
        for idx in range(params0.size):
            step = rel_step * scale[idx]
            delta = np.zeros_like(params0)
            delta[idx] = step
            potential.set_params(params0 + delta)
            values_plus = np.asarray(potential.force(r_arr), dtype=float)
            potential.set_params(params0 - delta)
            values_minus = np.asarray(potential.force(r_arr), dtype=float)
            numeric[..., idx] = (values_plus - values_minus) / (2.0 * step)
    finally:
        potential.set_params(params0)
    return numeric


@pytest.mark.parametrize("name,potential,expected_mask", build_potential_cases())
def test_derivative_api_shapes_and_linearity_masks(name, potential, expected_mask):
    del name
    r = np.array([0.75, 1.0, 1.35], dtype=float)
    energy_grad = potential.energy_grad(r)
    force_grad = potential.force_grad(r)
    assert energy_grad.shape == (r.size, potential.n_params())
    assert force_grad.shape == (r.size, potential.n_params())
    assert np.array_equal(potential.is_param_linear(), expected_mask)


@pytest.mark.parametrize("name,potential,expected_mask", build_potential_cases())
def test_force_grad_matches_finite_difference(name, potential, expected_mask):
    del name, expected_mask
    r = np.array([0.75, 1.0, 1.35], dtype=float)
    force_grad = potential.force_grad(r)
    if hasattr(force_grad, "toarray"):
        force_grad = force_grad.toarray()
    force_grad = np.asarray(force_grad, dtype=float)
    numeric = finite_difference_force_grad(potential, r)
    np.testing.assert_allclose(force_grad, numeric, atol=1.0e-5, rtol=1.0e-4)


def build_batch_dimension_cases():
    """Cases for the batch-shape contract."""
    return build_potential_cases()


@pytest.mark.parametrize("name,potential,expected_mask", build_batch_dimension_cases())
def test_potential_evaluators_preserve_leading_batch_dimensions(name, potential, expected_mask):
    del name, expected_mask
    r = np.array(
        [
            [0.75, 1.0, 1.35],
            [0.85, 1.15, 1.45],
        ],
        dtype=float,
    )

    assert np.asarray(potential.value(r)).shape == r.shape
    assert np.asarray(potential.force(r)).shape == r.shape
    energy_grad = np.asarray(potential.energy_grad(r), dtype=float)
    force_grad = potential.force_grad(r)
    if hasattr(force_grad, "toarray"):
        force_grad = force_grad.toarray()
    force_grad = np.asarray(force_grad, dtype=float)

    assert energy_grad.shape == r.shape + (potential.n_params(),)
    assert force_grad.shape == r.shape + (potential.n_params(),)
    np.testing.assert_allclose(
        potential.energy_grad_sum(r),
        energy_grad.sum(axis=-2),
    )


def test_harmonic_energy_grad_matches_named_derivatives():
    potential = HarmonicPotential("A", "B", k=2.0, r0=1.25)
    r = np.array([0.75, 1.0, 1.35], dtype=float)
    expected = np.column_stack([potential.dk(r), potential.dr0(r)])
    assert np.allclose(potential.energy_grad(r), expected)


def test_harmonic_force_grad_matches_named_derivatives():
    potential = HarmonicPotential("A", "B", k=2.0, r0=1.25)
    r = np.array([0.75, 1.0, 1.35], dtype=float)
    expected = np.column_stack([potential.dkdr(r), potential.dr0dr(r)])
    assert np.allclose(potential.force_grad(r), expected)


def test_bspline_energy_grad_matches_basis_integrals():
    potential = build_bspline()
    r = np.array([0.25, 0.75, 1.5], dtype=float)
    assert np.allclose(potential.energy_grad(r), -potential.basis_integrals(r))


def test_bspline_cutoff_gauge_zero_at_cutoff():
    potential = build_bspline()
    values = potential.value(np.array([potential.cutoff], dtype=float))
    assert values[0] == pytest.approx(0.0)


def test_bspline_minimum_gauge_zero_at_energy_minimum():
    potential = build_bspline_minimum_gauge()
    r = np.linspace(0.0, 2.0, 129, dtype=float)
    values = potential.value(r)
    assert float(np.min(values)) == pytest.approx(0.0, abs=1.0e-12)
    assert float(values[np.argmin(values)]) == pytest.approx(0.0, abs=1.0e-12)


def test_bspline_minimum_gauge_finds_off_grid_stationary_minimum():
    potential = build_bspline_off_grid_minimum_gauge()
    r_min = potential._minimum_gauge_coordinate()

    assert r_min == pytest.approx(0.7, abs=1.0e-12)
    assert float(potential.value(np.array([0.7], dtype=float))[0]) == pytest.approx(
        0.0,
        abs=1.0e-12,
    )


def test_bspline_minimum_gauge_energy_grad_matches_finite_difference():
    potential = build_bspline_minimum_gauge()
    r = np.array([0.25, 0.75, 1.5], dtype=float)
    analytic = potential.energy_grad(r)
    params0 = potential.get_params()
    step = 1.0e-7
    numeric = np.empty_like(analytic)
    try:
        for idx in range(params0.size):
            delta = np.zeros_like(params0)
            delta[idx] = step
            potential.set_params(params0 + delta)
            values_plus = potential.value(r)
            potential.set_params(params0 - delta)
            values_minus = potential.value(r)
            numeric[:, idx] = (values_plus - values_minus) / (2.0 * step)
    finally:
        potential.set_params(params0)
    assert np.allclose(analytic, numeric, atol=1.0e-6, rtol=1.0e-5)


def test_bspline_gauge_free_gradient_drops_bonded_gauge_shift():
    potential = build_bspline_minimum_gauge()
    r = np.array([0.25, 0.75, 1.5], dtype=float)
    physical = potential.energy_grad_sum(r)
    gauge_free = potential.gauge_free_energy_grad_sum(r)
    shift = potential.basis_integrals(
        np.asarray([potential._minimum_gauge_coordinate()], dtype=float)
    )[0]

    assert not np.allclose(physical, gauge_free)
    np.testing.assert_allclose(physical - gauge_free, r.size * shift)
    np.testing.assert_allclose(
        potential.gauge_free_energy_grad_sum(r.reshape(1, -1))[0],
        gauge_free,
    )


def test_bspline_force_grad_matches_basis_values():
    potential = build_bspline()
    r = np.array([0.25, 0.75, 1.5], dtype=float)
    force_grad = potential.force_grad(r)
    if hasattr(force_grad, "toarray"):
        force_grad = force_grad.toarray()
    assert np.allclose(force_grad, potential.basis_values(r))


def test_base_force_grad_requires_explicit_potential_implementation():
    class BarePotential(BasePotential):
        def __init__(self):
            super().__init__()
            self._params = np.array([1.0], dtype=float)
            self._param_names = ["A"]
            self._dparam_names = ["dA"]
            self._d2param_names = [["dA_2"]]

        def value(self, r):
            return np.asarray(r, dtype=float)

        def force(self, r):
            return np.asarray(r, dtype=float)

        def is_param_linear(self):
            return np.array([True], dtype=bool)

    with pytest.raises(NotImplementedError, match="force_grad"):
        BarePotential().force_grad(np.array([1.0], dtype=float))


def test_base_basis_derivatives_requires_explicit_potential_implementation():
    potential = GaussianPotential("A", "B", A=1.5, r0=1.0, sigma=0.4, cutoff=5.0)

    with pytest.raises(NotImplementedError, match="basis_derivatives"):
        potential.basis_derivatives(np.array([1.0], dtype=float))


def test_harmonic_param_bounds_property_keeps_nonnegative_force_constant():
    potential = HarmonicPotential("A", "B", k=2.0, r0=1.25)

    lower, upper = potential.param_bounds
    np.testing.assert_allclose(lower, [0.0, -np.inf])
    np.testing.assert_allclose(upper, [np.inf, np.inf])

    potential.param_bounds = (
        np.array([-5.0, 0.5], dtype=float),
        np.array([10.0, 2.0], dtype=float),
    )
    lower, upper = potential.param_bounds
    np.testing.assert_allclose(lower, [0.0, 0.5])
    np.testing.assert_allclose(upper, [10.0, 2.0])


def test_soft_cutoff_is_an_optimizable_parameter_with_named_derivatives():
    potential = SoftPotential("A", "B", A=25.0, cutoff=5.0)
    r = np.array([0.75, 1.0, 1.35], dtype=float)

    params0 = potential.get_params()
    step = 1.0e-6

    numeric_grad = np.empty((r.size, params0.size), dtype=float)
    for idx in range(params0.size):
        delta = np.zeros_like(params0)
        delta[idx] = step
        potential.set_params(params0 + delta)
        values_plus = potential.value(r)
        potential.set_params(params0 - delta)
        values_minus = potential.value(r)
        numeric_grad[:, idx] = (values_plus - values_minus) / (2.0 * step)
    potential.set_params(params0)

    analytic_grad = np.column_stack([potential.dA(r), potential.drc(r)])
    assert np.allclose(analytic_grad, numeric_grad, atol=1.0e-6, rtol=1.0e-5)

    potential.set_params(np.array([25.0, 6.0], dtype=float))
    assert potential.cutoff == pytest.approx(6.0)
    assert potential.value(np.array([5.5], dtype=float))[0] != pytest.approx(0.0)

    potential.set_params(params0)
    numeric_mixed = np.empty(r.size, dtype=float)
    numeric_cutoff_second = np.empty(r.size, dtype=float)
    delta = np.array([0.0, step], dtype=float)
    potential.set_params(params0 + delta)
    mixed_plus = potential.dA(r)
    cutoff_plus = potential.drc(r)
    potential.set_params(params0 - delta)
    mixed_minus = potential.dA(r)
    cutoff_minus = potential.drc(r)
    potential.set_params(params0)
    numeric_mixed[:] = (mixed_plus - mixed_minus) / (2.0 * step)
    numeric_cutoff_second[:] = (cutoff_plus - cutoff_minus) / (2.0 * step)

    assert np.allclose(potential.dAdrc(r), numeric_mixed, atol=1.0e-6, rtol=1.0e-5)
    assert np.allclose(potential.dA_drc(r), potential.dAdrc(r))
    assert np.allclose(potential.drc_2(r), numeric_cutoff_second, atol=1.0e-5, rtol=1.0e-4)


def test_registered_potentials_override_base_linearity_method():
    covered_types = {type(potential) for _, potential, _ in build_potential_cases()}
    registry_types = set(POTENTIAL_REGISTRY.values())
    assert registry_types <= covered_types
    for pot_type in registry_types:
        assert pot_type.is_param_linear is not BasePotential.is_param_linear


def test_registered_potentials_override_base_force_grad_method():
    covered_types = {type(potential) for _, potential, _ in build_potential_cases()}
    registry_types = set(POTENTIAL_REGISTRY.values())
    assert registry_types <= covered_types
    for pot_type in registry_types:
        assert pot_type.force_grad is not BasePotential.force_grad


# ─────────────────────────────────────────────────────────────────────────────
# Dihedral parameter derivatives (F-003)
#
# The shared parametrization above samples only phi in [0.75, 1.35] degrees,
# which never approaches the +-180 seam or a cutoff endpoint. These tests sweep
# a dense grid across the whole model domain, including both sides of the seam,
# so a branch error in the periodic bank cannot hide between sample points.
# ─────────────────────────────────────────────────────────────────────────────


def finite_difference_energy_grad(potential, phi, rel_step=1.0e-6):
    phi_arr = np.asarray(phi, dtype=float)
    params0 = potential.get_params()
    scale = np.maximum(1.0, np.abs(params0))
    numeric = np.empty(phi_arr.shape + (params0.size,), dtype=float)
    try:
        for idx in range(params0.size):
            step = rel_step * scale[idx]
            delta = np.zeros_like(params0)
            delta[idx] = step
            potential.set_params(params0 + delta)
            plus = np.asarray(potential.value(phi_arr), dtype=float)
            potential.set_params(params0 - delta)
            minus = np.asarray(potential.value(phi_arr), dtype=float)
            numeric[..., idx] = (plus - minus) / (2.0 * step)
    finally:
        potential.set_params(params0)
    return numeric


@pytest.mark.parametrize(
    "builder,phi",
    [
        (build_dihedral_periodic, np.linspace(-179.9, 179.9, 181)),
        (build_dihedral_periodic, np.array([-180.0, -179.99, -90.0, 0.0, 90.0, 179.99])),
        (build_dihedral_cutoff, np.linspace(-119.9, 99.9, 151)),
        (build_dihedral_cutoff, np.array([-120.0, -119.99, -60.0, 0.0, 50.0, 99.99])),
    ],
)
def test_dihedral_force_grad_matches_finite_difference(builder, phi):
    potential = builder()
    analytic = np.asarray(potential.force_grad(phi), dtype=float)
    numeric = finite_difference_force_grad(potential, phi)
    np.testing.assert_allclose(analytic, numeric, atol=1.0e-6, rtol=1.0e-5)


@pytest.mark.parametrize(
    "builder,phi",
    [
        (build_dihedral_periodic, np.linspace(-179.9, 179.9, 181)),
        (build_dihedral_cutoff, np.linspace(-119.9, 99.9, 151)),
    ],
)
def test_dihedral_energy_grad_matches_finite_difference(builder, phi):
    potential = builder()
    analytic = np.asarray(potential.energy_grad(phi), dtype=float)
    numeric = finite_difference_energy_grad(potential, phi)
    np.testing.assert_allclose(analytic, numeric, atol=1.0e-6, rtol=1.0e-5)


@pytest.mark.parametrize(
    "builder,phi",
    [
        (build_dihedral_periodic, np.linspace(-179.5, 179.5, 121)),
        (build_dihedral_cutoff, np.linspace(-119.5, 99.5, 121)),
    ],
)
def test_dihedral_force_is_negative_energy_derivative(builder, phi):
    """force(phi) == -dU/dphi, checked against a central difference in phi."""
    potential = builder()
    step = 1.0e-4
    numeric = -(
        np.asarray(potential.value(phi + step), dtype=float)
        - np.asarray(potential.value(phi - step), dtype=float)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        np.asarray(potential.force(phi), dtype=float), numeric, atol=1.0e-6, rtol=1.0e-5
    )


@pytest.mark.parametrize(
    "builder,phi",
    [
        (build_dihedral_periodic, np.linspace(-179.5, 179.5, 97)),
        (build_dihedral_cutoff, np.linspace(-119.5, 99.5, 97)),
    ],
)
def test_dihedral_basis_derivatives_match_finite_difference(builder, phi):
    """``basis_derivatives`` is d/dphi of ``basis_values``.

    This is not a spare API: ``fitters/fit_bspline.py:182`` builds the
    zero-slope anchor rows of the fit design matrix out of it, so an error here
    silently biases every fitted dihedral force basis.
    """
    potential = builder()
    step = 1.0e-4
    numeric = (
        np.asarray(potential.basis_values(phi + step), dtype=float)
        - np.asarray(potential.basis_values(phi - step), dtype=float)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        np.asarray(potential.basis_derivatives(phi), dtype=float),
        numeric,
        atol=1.0e-8,
        rtol=1.0e-6,
    )


@pytest.mark.parametrize(
    "builder,end",
    [
        (build_dihedral_periodic, 180.0 - 1.0e-9),
        (build_dihedral_cutoff, 100.0),
    ],
)
def test_dihedral_force_bases_integrate_to_zero_over_the_model_domain(builder, end):
    """Every trainable force basis must have zero integral over a full turn.

    ``basis_integrals`` measures from the -180 seam, so its value at the far
    end of the model domain *is* the full-domain integral of each basis. The
    module docstring states this constraint is what makes the integrated energy
    single-valued at the LAMMPS cyclic seam; if it is lost, U picks up a
    parameter-dependent jump every time a dihedral crosses +-180.
    """
    potential = builder()
    full_turn = np.asarray(
        potential.basis_integrals(np.array([end], dtype=float)), dtype=float
    )[0]
    np.testing.assert_allclose(full_turn, np.zeros(potential.n_params()), atol=1.0e-8)


def test_dihedral_periodic_bank_is_continuous_across_the_seam():
    """The +180 limit must meet the -180 value, not merely alias onto it."""
    potential = build_dihedral_periodic()
    left = np.array([-180.0], dtype=float)
    # +180 normalizes onto -180 by convention, so comparing the two is
    # tautological. Approach the seam from below instead, which is the case a
    # broken periodic fold or a lost integral constraint would actually break.
    limit = np.array([180.0 - 1.0e-7], dtype=float)
    np.testing.assert_allclose(potential.value(left), potential.value(limit), atol=1.0e-6)
    np.testing.assert_allclose(potential.force(left), potential.force(limit), atol=1.0e-6)
    np.testing.assert_allclose(
        np.asarray(potential.force_grad(left), dtype=float),
        np.asarray(potential.force_grad(limit), dtype=float),
        atol=1.0e-6,
    )
    # The convention itself: +180 is represented only as -180.
    np.testing.assert_allclose(
        potential.normalize_angles(np.array([180.0, 540.0, -180.0], dtype=float)),
        np.array([-180.0, -180.0, -180.0], dtype=float),
    )


def test_dihedral_cutoff_bank_vanishes_outside_the_model_window():
    """Force, energy and every parameter derivative are zero outside the window.

    The class docstring promises this. Because the bases integrate to zero over
    the window, the energy is continuous at both edges rather than stepping.
    """
    potential = build_dihedral_cutoff()
    outside = np.array([-179.0, -130.0, 110.0, 179.0], dtype=float)
    np.testing.assert_allclose(potential.value(outside), np.zeros(outside.size), atol=1.0e-12)
    np.testing.assert_allclose(potential.force(outside), np.zeros(outside.size), atol=1.0e-12)
    np.testing.assert_allclose(
        np.asarray(potential.force_grad(outside), dtype=float),
        np.zeros((outside.size, potential.n_params())),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(potential.energy_grad(outside), dtype=float),
        np.zeros((outside.size, potential.n_params())),
        atol=1.0e-12,
    )
    edges = np.array([-120.0, 100.0], dtype=float)
    np.testing.assert_allclose(potential.value(edges), np.zeros(2), atol=1.0e-9)


@pytest.mark.parametrize("builder", [build_dihedral_periodic, build_dihedral_cutoff])
def test_dihedral_energy_grad_sum_matches_base_batch_contract(builder):
    """F-017: dihedral ``energy_grad_sum`` must sum over the last axis only.

    ``BasePotential.energy_grad_sum`` documents "summed over the last
    coordinate axis" and every registered potential must honour it, so a
    ``(n_frames, n_terms)`` input yields ``(n_frames, n_params)``.
    ``DihedralBSplinePotential.energy_grad_sum`` now delegates rank>1 input to
    ``energy_grad_sum_by_sample`` (the same pattern ``BSplinePotential`` uses)
    instead of flattening every axis.
    """
    potential = builder()
    phi = np.array([[-90.0, 0.0, 45.0], [-30.0, 10.0, 60.0]], dtype=float)
    n_params = potential.n_params()

    per_frame = np.asarray(potential.energy_grad_sum(phi), dtype=float)
    assert per_frame.shape == (2, n_params)

    per_sample = np.asarray(potential.energy_grad_sum_by_sample(phi), dtype=float)
    np.testing.assert_allclose(per_sample, per_frame, rtol=1.0e-12)

    reference = np.asarray(potential.energy_grad(phi), dtype=float).sum(axis=-2)
    np.testing.assert_allclose(per_frame, reference, rtol=1.0e-10, atol=1.0e-12)
