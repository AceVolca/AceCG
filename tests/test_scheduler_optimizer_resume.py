"""P0 test: optimizer state_dict round-trips through BaseWorkflow checkpoint."""

import numpy as np
import pytest

from AceCG.optimizers.adamW import AdamWMaskedOptimizer
from AceCG.optimizers.adam import AdamMaskedOptimizer


def test_adamw_state_dict_roundtrip():
    L = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    mask = np.array([True, False, True])
    opt = AdamWMaskedOptimizer(L, mask, lr=0.01, weight_decay=0.01)

    # Run a few steps
    for i in range(5):
        grad = np.random.default_rng(i).standard_normal(3)
        opt.step(grad)

    state = opt.state_dict()
    assert state["t"] == 5
    assert len(state["m"]) == 3

    # Recreate and restore
    opt2 = AdamWMaskedOptimizer(np.zeros(3), mask, lr=0.1)
    opt2.load_state_dict(state)
    assert opt2.t == 5
    assert opt2.lr == pytest.approx(0.01)
    assert np.allclose(opt2.L, opt.L)
    assert np.allclose(opt2.m, opt.m)
    assert np.allclose(opt2.v, opt.v)


def test_adam_state_dict_roundtrip():
    L = np.array([1.0, 2.0], dtype=np.float64)
    mask = np.array([True, True])
    opt = AdamMaskedOptimizer(L, mask, lr=0.02)
    for i in range(3):
        opt.step(np.ones(2) * (i + 1))

    state = opt.state_dict()
    opt2 = AdamMaskedOptimizer(np.zeros(2), mask, lr=0.5)
    opt2.load_state_dict(state)
    assert opt2.t == 3
    assert np.allclose(opt2.L, opt.L)


def test_adamw_amsgrad_roundtrip():
    L = np.array([1.0, 2.0], dtype=np.float64)
    mask = np.array([True, True])
    opt = AdamWMaskedOptimizer(L, mask, lr=0.01, amsgrad=True)
    opt.step(np.ones(2))

    state = opt.state_dict()
    assert state["vmax"] is not None
    opt2 = AdamWMaskedOptimizer(np.zeros(2), mask)
    opt2.load_state_dict(state)
    assert opt2.vmax is not None
    assert np.allclose(opt2.vmax, opt.vmax)
