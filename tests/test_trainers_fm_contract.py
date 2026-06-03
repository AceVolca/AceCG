"""Focused trainer-contract tests that avoid workflow imports."""

from __future__ import annotations

import inspect

import numpy as np

from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.trainers.base import BaseTrainer
from AceCG.trainers.analytic.fm import FMTrainerAnalytic


class DummyOptimizer:
    def __init__(self, params):
        self.L = np.asarray(params, dtype=np.float64).copy()
        self.mask = np.ones_like(self.L, dtype=bool)

    def set_params(self, params):
        self.L = np.asarray(params, dtype=np.float64).copy()

    def step(self, grad, hessian=None):
        return np.zeros_like(np.asarray(grad, dtype=np.float64))


def _make_forcefield():
    key = InteractionKey.bond("A", "B")
    pot = HarmonicPotential("A", "B", k=5.0, r0=4.0)
    return Forcefield({key: [pot]})


def test_base_trainer_step_uses_batch_contract():
    params = list(inspect.signature(BaseTrainer.step).parameters)
    assert params == ["self", "batch", "apply_update"]


def test_fm_trainer_uses_normalized_fm_statistics():
    forcefield = _make_forcefield()
    optimizer = DummyOptimizer(forcefield.param_array())
    trainer = FMTrainerAnalytic(forcefield, optimizer)

    JtJ = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=np.float64)
    Jty = np.array([0.4, -0.2], dtype=np.float64)
    Jtf = np.array([0.9, 0.7], dtype=np.float64)
    y_sumsq = 1.6
    f_sumsq = 2.4
    fty = 0.3

    out = trainer.step(
        {
            "JtJ": JtJ,
            "Jty": Jty,
            "y_sumsq": y_sumsq,
            "Jtf": Jtf,
            "f_sumsq": f_sumsq,
            "fty": fty,
            "nframe": 5,
        },
        apply_update=False,
    )

    np.testing.assert_allclose(out["grad"], Jtf - Jty)
    np.testing.assert_allclose(out["hessian"], JtJ)
    assert out["loss"] == 0.5 * (f_sumsq - 2.0 * fty + y_sumsq)
    np.testing.assert_allclose(out["update"], np.zeros_like(Jty))
