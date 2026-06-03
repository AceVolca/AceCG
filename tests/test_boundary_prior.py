from __future__ import annotations

import numpy as np

from AceCG.potentials.boundary_prior import BoundaryPriorPotential
from AceCG.potentials.gaussian import GaussianPotential
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey


def _gaussian() -> GaussianPotential:
    return GaussianPotential("A", "B", A=1.0, r0=2.0, sigma=0.5, cutoff=4.0)


def test_boundary_prior_forwards_property_style_mask_and_bounds() -> None:
    base = _gaussian()
    base.param_mask = np.array([True, False, True], dtype=bool)
    base.param_bounds = (
        np.array([0.0, -np.inf, 0.1], dtype=float),
        np.array([5.0, np.inf, 2.0], dtype=float),
    )

    wrapped = BoundaryPriorPotential(base, style="pair", lower=1.0)

    np.testing.assert_array_equal(wrapped.param_mask, [True, False, True])
    lb, ub = wrapped.param_bounds
    np.testing.assert_allclose(lb, [0.0, -np.inf, 0.1])
    np.testing.assert_allclose(ub, [5.0, np.inf, 2.0])

    wrapped.param_mask = np.array([False, True, True], dtype=bool)
    np.testing.assert_array_equal(base.param_mask, [False, True, True])

    wrapped.param_bounds = (
        np.array([-1.0, 0.0, 0.2], dtype=float),
        np.array([6.0, 4.0, 1.8], dtype=float),
    )
    base_lb, base_ub = base.param_bounds
    np.testing.assert_allclose(base_lb, [-1.0, 0.0, 0.2])
    np.testing.assert_allclose(base_ub, [6.0, 4.0, 1.8])


def test_boundary_prior_forcefield_cache_tracks_base_bounds_update() -> None:
    key = InteractionKey.pair("A", "B")
    wrapped = BoundaryPriorPotential(_gaussian(), style="pair", lower=1.0)
    forcefield = Forcefield({key: [wrapped]})

    lb_first, _ = forcefield.param_bounds
    forcefield[key][0].base.param_bounds = (
        np.array([-2.0, 0.0, 0.2], dtype=float),
        np.array([2.0, 4.0, 1.8], dtype=float),
    )
    lb_updated, ub_updated = forcefield.param_bounds

    assert lb_updated is not lb_first
    np.testing.assert_allclose(lb_updated, [-2.0, 0.0, 0.2])
    np.testing.assert_allclose(ub_updated, [2.0, 4.0, 1.8])


def test_boundary_prior_supports_method_style_base_bounds() -> None:
    base = HarmonicPotential("A", "B", k=1.0, r0=2.0)
    wrapped = BoundaryPriorPotential(base, style="bond", lower=1.0, upper=3.0)

    lb, ub = wrapped.param_bounds
    np.testing.assert_allclose(lb, [0.0, -np.inf])
    assert np.isposinf(ub).all()

    wrapped.param_bounds = (
        np.array([0.5, 1.0], dtype=float),
        np.array([5.0, 3.0], dtype=float),
    )
    lb_updated, ub_updated = wrapped.param_bounds
    np.testing.assert_allclose(lb_updated, [0.5, 1.0])
    np.testing.assert_allclose(ub_updated, [5.0, 3.0])
