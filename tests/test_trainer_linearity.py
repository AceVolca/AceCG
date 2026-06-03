import numpy as np

from AceCG.optimizers import AdamMaskedOptimizer
from AceCG.potentials.gaussian import GaussianPotential
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey
from AceCG.trainers.base import BaseTrainer


class _DummyTrainer(BaseTrainer):
    def step(self, batch, apply_update=True):
        return {"batch": batch, "apply_update": apply_update}


def _build_trainer(forcefield: Forcefield, mask: np.ndarray) -> _DummyTrainer:
    forcefield = Forcefield(forcefield)
    forcefield.build_mask(init_mask=np.asarray(mask, dtype=bool))
    optimizer = AdamMaskedOptimizer(
        L=forcefield.param_array(),
        mask=forcefield.param_mask,
        lr=1.0e-3,
        seed=0,
    )
    return _DummyTrainer(forcefield=forcefield, optimizer=optimizer)


def test_is_optimization_linear_ignores_masked_nonlinear_channels():
    forcefield = Forcefield(
        {
            InteractionKey(style="bond", types=("1",)): [HarmonicPotential("A", "B", k=2.0, r0=1.25)],
        }
    )
    trainer = _build_trainer(forcefield, np.array([True, False], dtype=bool))
    assert trainer.is_optimization_linear() is True


def test_is_optimization_linear_detects_active_nonlinear_channels():
    forcefield = Forcefield(
        {
            InteractionKey(style="bond", types=("1",)): [HarmonicPotential("A", "B", k=2.0, r0=1.25)],
        }
    )
    trainer = _build_trainer(forcefield, np.array([True, True], dtype=bool))
    assert trainer.is_optimization_linear() is False


def test_is_optimization_linear_handles_mixed_forcefields():
    forcefield = Forcefield(
        {
            InteractionKey.pair("A", "B"): [
                GaussianPotential("A", "B", A=1.5, r0=1.0, sigma=0.4, cutoff=5.0)
            ],
            InteractionKey(style="bond", types=("2",)): [
                HarmonicPotential("A", "B", k=2.0, r0=1.25)
            ],
        }
    )
    trainer = _build_trainer(
        forcefield,
        np.array([True, False, False, True, False], dtype=bool),
    )
    assert trainer.is_optimization_linear() is True

    nonlinear_trainer = _build_trainer(
        forcefield,
        np.array([True, False, True, True, False], dtype=bool),
    )
    assert nonlinear_trainer.is_optimization_linear() is False
