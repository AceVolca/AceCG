from __future__ import annotations

import numpy as np

from AceCG.optimizers import AdamMaskedOptimizer
from AceCG.potentials.gaussian import GaussianPotential
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey
from AceCG.trainers.analytic.cdfm import CDFMTrainerAnalytic
from AceCG.trainers.analytic.fm import FMTrainerAnalytic
from AceCG.trainers.base import BaseTrainer


class _DummyTrainer(BaseTrainer):
    def step(self, batch, apply_update=True):
        return {"batch": batch, "apply_update": apply_update}


def _build_forcefield() -> tuple[Forcefield, InteractionKey, InteractionKey]:
    pair_key = InteractionKey.pair("A", "B")
    bond_key = InteractionKey(style="bond", types=("1",))
    forcefield = Forcefield(
        {
            pair_key: [GaussianPotential("A", "B", A=1.5, r0=1.0, sigma=0.4, cutoff=5.0)],
            bond_key: [HarmonicPotential("A", "B", k=2.0, r0=1.25)],
        }
    )
    forcefield.build_mask(init_mask=np.ones(forcefield.n_params(), dtype=bool))
    return forcefield, pair_key, bond_key


def _build_optimizer(forcefield: Forcefield, mask: np.ndarray) -> AdamMaskedOptimizer:
    return AdamMaskedOptimizer(
        L=forcefield.param_array(),
        mask=np.asarray(mask, dtype=bool),
        lr=1.0e-3,
        seed=0,
    )


def _offset_tuples(offsets):
    return [(sl.start, sl.stop, sl.step) for sl in offsets]


def test_base_trainer_accessors_follow_forcefield_and_optimizer_mask() -> None:
    forcefield, pair_key, bond_key = _build_forcefield()
    trainer = _DummyTrainer(
        forcefield=forcefield,
        optimizer=_build_optimizer(forcefield, np.array([True, False, False, False, False])),
    )

    np.testing.assert_allclose(trainer.get_params(), forcefield.param_array())
    assert trainer.n_total_params() == forcefield.n_params()
    assert trainer.get_interaction_labels() == [pair_key.label(), bond_key.label()]

    param_names = trainer.get_param_names()
    assert len(param_names) == forcefield.n_params()
    assert param_names[0].startswith(pair_key.label())
    assert param_names[-1].startswith(bond_key.label())

    lower, upper = trainer.get_param_bounds()
    assert lower.shape == (forcefield.n_params(),)
    assert upper.shape == (forcefield.n_params(),)

    active = trainer.active_interaction_mask()
    assert active[pair_key] is True
    assert active[bond_key] is False


def test_fm_and_cdfm_trainers_expose_current_offset_accessor() -> None:
    forcefield, _, _ = _build_forcefield()
    optimizer = _build_optimizer(forcefield, np.ones(forcefield.n_params(), dtype=bool))

    fm_trainer = FMTrainerAnalytic(forcefield=forcefield, optimizer=optimizer)
    cdfm_trainer = CDFMTrainerAnalytic(forcefield=forcefield, optimizer=optimizer)

    expected = _offset_tuples(forcefield.interaction_offsets())
    assert _offset_tuples(fm_trainer.get_offsets()) == expected
    assert _offset_tuples(cdfm_trainer.get_offsets()) == expected
    assert set(FMTrainerAnalytic.schema()) == {"batch", "return"}
    assert set(CDFMTrainerAnalytic.schema()) == {"batch", "return"}
