from types import SimpleNamespace

import numpy as np
import pytest

from AceCG.optimizers import SGDMaskedOptimizer


def test_sgd_masked_step_and_state_roundtrip():
    params = np.array([1.0, 2.0, 3.0])
    mask = np.array([True, False, True])
    optimizer = SGDMaskedOptimizer(params, mask, lr=0.1)

    update = optimizer.step(np.array([2.0, 50.0, -4.0]))

    np.testing.assert_allclose(update, [-0.2, 0.0, 0.4])
    np.testing.assert_allclose(optimizer.L, [0.8, 2.0, 3.4])

    restored = SGDMaskedOptimizer(np.zeros(3), np.ones(3, dtype=bool), lr=1.0)
    restored.load_state_dict(optimizer.state_dict())
    np.testing.assert_allclose(restored.L, optimizer.L)
    np.testing.assert_array_equal(restored.mask, mask)
    assert restored.lr == 0.1
    assert restored.clip_grad_value is None
    assert restored.clip_grad_norm is None


def test_sgd_rejects_gradient_shape_mismatch():
    optimizer = SGDMaskedOptimizer(
        np.zeros(2), np.ones(2, dtype=bool), lr=0.1
    )

    with np.testing.assert_raises_regex(ValueError, "Gradient shape"):
        optimizer.step(np.zeros(3))


def test_sgd_clips_active_gradient_values_before_scaling():
    optimizer = SGDMaskedOptimizer(
        np.array([1.0, 2.0, 3.0]),
        np.array([True, False, True]),
        lr=0.1,
        clip_grad_value=1.0,
    )

    update = optimizer.step(np.array([20.0, 50.0, -0.5]))

    np.testing.assert_allclose(update, [-0.1, 0.0, 0.05])
    np.testing.assert_allclose(optimizer.L, [0.9, 2.0, 3.05])
    assert optimizer.state_dict()["clip_grad_value"] == 1.0


def test_sgd_clips_each_interaction_block_norm_without_flattening_components():
    optimizer = SGDMaskedOptimizer(
        np.zeros(6),
        np.array([True, True, True, True, False, True]),
        lr=0.1,
        clip_grad_norm=5.0,
        clip_blocks=[("pair:A:B", 0, 3), ("angle:A:B:C", 3, 6)],
    )

    update = optimizer.step(np.array([6.0, 8.0, 0.0, 0.6, 100.0, 0.8]))

    # The first block is scaled by 1/2 and preserves its 3:4 direction.  The
    # second block is below the threshold when the inactive entry is excluded.
    np.testing.assert_allclose(update, [-0.3, -0.4, 0.0, -0.06, 0.0, -0.08])
    assert optimizer.last_clip_stats == (
        {
            "label": "pair:A:B",
            "start": 0,
            "stop": 3,
            "raw_norm": 10.0,
            "effective_norm": 5.0,
            "scale": 0.5,
            "clipped": True,
        },
        {
            "label": "angle:A:B:C",
            "start": 3,
            "stop": 6,
            "raw_norm": 1.0,
            "effective_norm": 1.0,
            "scale": 1.0,
            "clipped": False,
        },
    )

    restored = SGDMaskedOptimizer(np.zeros(6), np.ones(6, dtype=bool), lr=1.0)
    restored.load_state_dict(optimizer.state_dict())
    assert restored.clip_grad_value is None
    assert restored.clip_grad_norm == 5.0
    assert restored.clip_blocks == (
        ("pair:A:B", 0, 3),
        ("angle:A:B:C", 3, 6),
    )


def test_sgd_block_norm_clip_rejects_invalid_or_ambiguous_configuration():
    with pytest.raises(ValueError, match="mutually exclusive"):
        SGDMaskedOptimizer(
            np.zeros(2),
            np.ones(2, dtype=bool),
            clip_grad_value=1.0,
            clip_grad_norm=1.0,
            clip_blocks=[("pair:A:A", 0, 2)],
        )

    with pytest.raises(ValueError, match="cover every active parameter"):
        SGDMaskedOptimizer(
            np.zeros(3),
            np.ones(3, dtype=bool),
            clip_grad_norm=1.0,
            clip_blocks=[("pair:A:A", 0, 2)],
        )


def test_workflow_builds_sgd_from_config_token():
    pytest.importorskip("MDAnalysis")
    from AceCG.workflows.base import BaseWorkflow

    training = SimpleNamespace(
        optimizer="sgd clip_grad_value=1.0",
        trainer=None,
        lr=1.0e-3,
        seed=17794,
    )
    workflow = SimpleNamespace(config=SimpleNamespace(training=training))
    forcefield = SimpleNamespace(
        param_array=lambda: np.array([1.0, 2.0]),
        param_mask=np.array([True, False]),
    )

    optimizer = BaseWorkflow._build_optimizer(workflow, forcefield)

    assert isinstance(optimizer, SGDMaskedOptimizer)
    assert optimizer.lr == 1.0e-3
    assert optimizer.clip_grad_value == 1.0
    np.testing.assert_array_equal(optimizer.mask, [True, False])


def test_workflow_routes_forcefield_blocks_to_sgd_norm_clipping():
    pytest.importorskip("MDAnalysis")
    from AceCG.workflows.base import BaseWorkflow

    training = SimpleNamespace(
        optimizer="sgd clip_grad_norm=2.0",
        trainer=None,
        lr=5.0e-3,
        seed=22689,
    )
    workflow = SimpleNamespace(config=SimpleNamespace(training=training))
    forcefield = SimpleNamespace(
        param_array=lambda: np.zeros(4),
        param_mask=np.ones(4, dtype=bool),
        param_blocks=lambda: [
            (SimpleNamespace(label=lambda: "pair:A:B"), object(), slice(0, 2)),
            (SimpleNamespace(label=lambda: "bond:A:B"), object(), slice(2, 4)),
        ],
    )

    optimizer = BaseWorkflow._build_optimizer(workflow, forcefield)

    assert isinstance(optimizer, SGDMaskedOptimizer)
    assert optimizer.clip_grad_value is None
    assert optimizer.clip_grad_norm == 2.0
    assert optimizer.clip_blocks == (
        ("pair:A:B", 0, 2),
        ("bond:A:B", 2, 4),
    )
