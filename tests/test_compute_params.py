"""Tests for Forcefield parameter-vector utilities: param_array, param_index_map,
build_mask, build_bounds, describe_mask, describe_bounds, derive_l1_mask."""

import pytest
import numpy as np

from AceCG.potentials.gaussian import GaussianPotential
from AceCG.potentials.harmonic import HarmonicPotential
from AceCG.topology.forcefield import Forcefield
from AceCG.topology.types import InteractionKey


# ---------------------------------------------------------------------------
# Mock potential (duck-typed; does not subclass BasePotential)
# ---------------------------------------------------------------------------

class MockPot:
    """Minimal mock: n_params(), get_params(), param_names()."""

    def __init__(self, params, names):
        self._params = np.array(params, dtype=float)
        self._names = list(names)

    def n_params(self) -> int:
        return len(self._params)

    def get_params(self) -> np.ndarray:
        return self._params.copy()

    def param_names(self):
        return list(self._names)


# 2-interaction fixture: ("A","B") → 3 params, ("B","C") → 2 params
@pytest.fixture
def p2p():
    pA = MockPot([1.0, 2.0, 3.0], ["c0", "c1", "c2"])
    pB = MockPot([4.0, 5.0], ["c0", "c1"])
    return Forcefield({
        InteractionKey.pair("A", "B"): pA,
        InteractionKey.pair("B", "C"): pB,
    })


# ---------------------------------------------------------------------------
# param_array (was FFParamArray)
# ---------------------------------------------------------------------------

def test_ffparamarray_length(p2p):
    arr = p2p.param_array()
    assert arr.shape == (5,)


def test_ffparamarray_values(p2p):
    arr = p2p.param_array()
    assert np.allclose(arr[:3], [1.0, 2.0, 3.0])
    assert np.allclose(arr[3:], [4.0, 5.0])


def test_ffparamarray_empty():
    arr = Forcefield({}).param_array()
    assert arr.shape == (0,)


def test_ffparamarray_is_copy(p2p):
    arr = p2p.param_array()
    arr[0] = 999.0
    # Original unchanged
    arr2 = p2p.param_array()
    assert arr2[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# param_index_map (was FFParamIndexMap)
# ---------------------------------------------------------------------------

def test_ffparamindexmap_length(p2p):
    idx_map = p2p.param_index_map()
    assert len(idx_map) == 5


def test_ffparamindexmap_structure(p2p):
    idx_map = p2p.param_index_map()
    # First 3 → ("A","B") with names c0,c1,c2
    for i, name in enumerate(["c0", "c1", "c2"]):
        assert idx_map[i][0] == InteractionKey.pair("A", "B")
        assert idx_map[i][1] == name
    # Last 2 → ("B","C") with names c0,c1
    for i, name in enumerate(["c0", "c1"]):
        assert idx_map[3 + i][0] == InteractionKey.pair("B", "C")
        assert idx_map[3 + i][1] == name


# ---------------------------------------------------------------------------
# build_mask (was BuildGlobalMask)
# ---------------------------------------------------------------------------

def test_buildglobalmask_default_all_true(p2p):
    mask = p2p.build_mask()
    assert mask.shape == (5,)
    assert np.all(mask)


def test_buildglobalmask_freeze_specific_pair(p2p):
    # Freeze ("B","C"): both its params → [True,True,True,False,False]
    mask = p2p.build_mask(
        patterns={InteractionKey.pair("B", "C"): ["c0", "c1"]},
        mode="freeze",
    )
    assert mask.shape == (5,)
    assert np.all(mask[:3])
    assert not mask[3]
    assert not mask[4]


def test_buildglobalmask_freeze_one_param(p2p):
    # Freeze only c0 of ("A","B")
    mask = p2p.build_mask(
        patterns={InteractionKey.pair("A", "B"): ["c0"]},
        mode="freeze",
    )
    assert not mask[0]   # c0 of ("A","B") frozen
    assert mask[1]       # c1 trainable
    assert mask[2]       # c2 trainable
    assert mask[3]       # ("B","C") trainable
    assert mask[4]


def test_buildglobalmask_train_mode(p2p):
    # Train mode: default frozen; train only c1 of ("A","B")
    mask = p2p.build_mask(
        patterns={InteractionKey.pair("A", "B"): ["c1"]},
        mode="train",
    )
    assert not mask[0]   # c0 frozen
    assert mask[1]       # c1 trainable
    assert not mask[2]   # c2 frozen
    assert not mask[3]   # ("B","C") frozen
    assert not mask[4]


def test_buildglobalmask_global_patterns(p2p):
    # Freeze all c0 params globally
    mask = p2p.build_mask(global_patterns=["c0"], mode="freeze")
    assert not mask[0]   # ("A","B") c0 frozen
    assert mask[1]       # ("A","B") c1 trainable
    assert mask[2]       # ("A","B") c2 trainable
    assert not mask[3]   # ("B","C") c0 frozen
    assert mask[4]       # ("B","C") c1 trainable


def test_buildmask_stores_on_param_mask(p2p):
    """build_mask should also set self.param_mask."""
    mask = p2p.build_mask(
        patterns={InteractionKey.pair("A", "B"): ["c0"]},
        mode="freeze",
    )
    assert np.array_equal(p2p.param_mask, mask)


def test_potential_local_param_mask_is_source_of_truth():
    pA = MockPot([1.0, 2.0, 3.0], ["c0", "c1", "c2"])
    pB = MockPot([4.0, 5.0], ["c0", "c1"])
    pA.param_mask = np.array([True, False, True], dtype=bool)
    pB.param_mask = np.array([False, True], dtype=bool)

    ff = Forcefield({
        InteractionKey.pair("A", "B"): pA,
        InteractionKey.pair("B", "C"): pB,
    })

    np.testing.assert_array_equal(
        ff.param_mask,
        np.array([True, False, True, False, True], dtype=bool),
    )


def test_param_mask_setter_writes_potential_local_masks(p2p):
    mask = np.array([False, True, False, True, False], dtype=bool)
    p2p.param_mask = mask

    np.testing.assert_array_equal(
        p2p[InteractionKey.pair("A", "B")].param_mask,
        np.array([False, True, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        p2p[InteractionKey.pair("B", "C")].param_mask,
        np.array([True, False], dtype=bool),
    )


def test_param_mask_setter_does_not_change_key_mask(p2p):
    p2p.param_mask = np.array([False, False, False, True, False], dtype=bool)
    assert p2p.key_mask == {
        InteractionKey.pair("A", "B"): True,
        InteractionKey.pair("B", "C"): True,
    }


def test_key_mask_setter_normalizes_interaction_mask_only(p2p):
    p2p.key_mask = {InteractionKey.pair("A", "B"): False}
    np.testing.assert_array_equal(
        p2p.param_mask,
        np.array([True, True, True, True, True], dtype=bool),
    )
    assert p2p.key_mask == {
        InteractionKey.pair("A", "B"): False,
        InteractionKey.pair("B", "C"): True,
    }


def test_param_mask_cache_reuses_readonly_array_for_versioned_potentials():
    key = InteractionKey.pair("A", "B")
    pot = GaussianPotential("A", "B", A=1.0, r0=2.0, sigma=0.5, cutoff=4.0)
    ff = Forcefield({key: [pot]})

    first = ff.param_mask
    second = ff.param_mask

    assert first is second
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0] = False


def test_param_mask_cache_refreshes_after_potential_local_update():
    key = InteractionKey.pair("A", "B")
    pot = GaussianPotential("A", "B", A=1.0, r0=2.0, sigma=0.5, cutoff=4.0)
    ff = Forcefield({key: [pot]})

    first = ff.param_mask
    ff[key][0].param_mask = np.array([True, False, True], dtype=bool)
    updated = ff.param_mask

    assert updated is not first
    np.testing.assert_array_equal(updated, [True, False, True])
    assert ff.key_mask == {key: True}


def test_duck_typed_param_mask_changes_are_conservatively_resynced(p2p):
    key = InteractionKey.pair("A", "B")

    _ = p2p.param_mask
    p2p[key].param_mask = np.array([False, True, False], dtype=bool)

    np.testing.assert_array_equal(
        p2p.param_mask,
        np.array([False, True, False, True, True], dtype=bool),
    )


# ---------------------------------------------------------------------------
# describe_mask (was DescribeMask)
# ---------------------------------------------------------------------------

def test_describemask_returns_string(p2p):
    p2p.build_mask()
    s = p2p.describe_mask()
    assert isinstance(s, str)
    assert len(s) > 0


def test_describemask_contains_status(p2p):
    p2p.build_mask()
    s = p2p.describe_mask()
    assert "train" in s


def test_describemask_length_mismatch(p2p):
    bad_mask = np.array([True, False])
    with pytest.raises(ValueError):
        p2p.describe_mask(bad_mask)


# ---------------------------------------------------------------------------
# build_bounds (was BuildGlobalBounds)
# ---------------------------------------------------------------------------

def test_buildglobalbounds_default_infinite(p2p):
    lb, ub = p2p.build_bounds()
    assert lb.shape == (5,)
    assert ub.shape == (5,)
    assert np.all(np.isneginf(lb))
    assert np.all(np.isposinf(ub))


def test_buildglobalbounds_pair_bounds(p2p):
    # Bound c0 and c1 of ("A","B") to [0, 10]
    lb, ub = p2p.build_bounds(
        pair_bounds={
            InteractionKey.pair("A", "B"): {"c0": (0.0, 10.0), "c1": (0.0, 10.0)},
        },
    )
    assert lb[0] == pytest.approx(0.0)
    assert ub[0] == pytest.approx(10.0)
    assert lb[1] == pytest.approx(0.0)
    assert ub[1] == pytest.approx(10.0)
    # c2 of ("A","B") unbounded
    assert np.isneginf(lb[2])
    assert np.isposinf(ub[2])
    # ("B","C") params unbounded
    assert np.isneginf(lb[3])
    assert np.isposinf(ub[3])


def test_buildglobalbounds_global_bounds(p2p):
    # Apply lower bound of -1 to all c0 params
    lb, ub = p2p.build_bounds(global_bounds={"c0": (-1.0, None)})
    assert lb[0] == pytest.approx(-1.0)   # ("A","B") c0
    assert np.isposinf(ub[0])
    assert lb[3] == pytest.approx(-1.0)   # ("B","C") c0
    assert np.isposinf(ub[3])
    # c1,c2 of ("A","B") and c1 of ("B","C") still infinite
    assert np.isneginf(lb[1])
    assert np.isneginf(lb[2])
    assert np.isneginf(lb[4])


def test_buildbounds_stores_on_param_bounds(p2p):
    """build_bounds should also set self.param_bounds."""
    lb, ub = p2p.build_bounds(global_bounds={"c0": (-1.0, 1.0)})
    stored_lb, stored_ub = p2p.param_bounds
    assert np.array_equal(lb, stored_lb)
    assert np.array_equal(ub, stored_ub)


def test_buildbounds_writes_potential_local_bounds(p2p):
    p2p.build_bounds(global_bounds={"c0": (-1.0, 1.0)})
    local_lb, local_ub = p2p[InteractionKey.pair("A", "B")].param_bounds

    np.testing.assert_allclose(local_lb, [-1.0, -np.inf, -np.inf])
    np.testing.assert_allclose(local_ub, [1.0, np.inf, np.inf])


def test_potential_local_bounds_are_buildbounds_base():
    pA = MockPot([1.0, 2.0, 3.0], ["c0", "c1", "c2"])
    pB = MockPot([4.0, 5.0], ["c0", "c1"])
    pA.param_bounds = (
        np.array([0.0, -np.inf, -np.inf], dtype=float),
        np.array([10.0, np.inf, np.inf], dtype=float),
    )

    ff = Forcefield({
        InteractionKey.pair("A", "B"): pA,
        InteractionKey.pair("B", "C"): pB,
    })
    lb, ub = ff.build_bounds(global_bounds={"c1": (-2.0, 2.0)})

    np.testing.assert_allclose(lb, [0.0, -2.0, -np.inf, -np.inf, -2.0])
    np.testing.assert_allclose(ub, [10.0, 2.0, np.inf, np.inf, 2.0])


def test_insert_bounded_potential_materializes_bounds_cache():
    ff = Forcefield({
        InteractionKey.pair("A", "B"): MockPot([1.0], ["c0"]),
    })
    lb0, ub0 = ff.param_bounds
    assert np.isneginf(lb0[0]) and np.isposinf(ub0[0])

    ff[InteractionKey.bond("B", "C")] = HarmonicPotential("B", "C", k=1.0, r0=2.0)

    lb, ub = ff.param_bounds
    np.testing.assert_allclose(lb, [-np.inf, 0.0, -np.inf])
    assert np.isposinf(ub).all()


def test_param_bounds_cache_reuses_readonly_arrays_for_versioned_potentials():
    key = InteractionKey.pair("A", "B")
    pot = GaussianPotential("A", "B", A=1.0, r0=2.0, sigma=0.5, cutoff=4.0)
    ff = Forcefield({key: [pot]})

    lb_first, ub_first = ff.param_bounds
    lb_second, ub_second = ff.param_bounds

    assert lb_first is lb_second
    assert ub_first is ub_second
    assert not lb_first.flags.writeable
    assert not ub_first.flags.writeable
    with pytest.raises(ValueError):
        lb_first[0] = -1.0


def test_param_bounds_cache_refreshes_after_potential_local_update():
    key = InteractionKey.pair("A", "B")
    pot = GaussianPotential("A", "B", A=1.0, r0=2.0, sigma=0.5, cutoff=4.0)
    ff = Forcefield({key: [pot]})

    lb_first, ub_first = ff.param_bounds
    ff[key][0].param_bounds = (
        np.array([-2.0, 0.0, 0.1], dtype=float),
        np.array([2.0, 4.0, 1.5], dtype=float),
    )
    lb_updated, ub_updated = ff.param_bounds

    assert lb_updated is not lb_first
    assert ub_updated is not ub_first
    np.testing.assert_allclose(lb_updated, [-2.0, 0.0, 0.1])
    np.testing.assert_allclose(ub_updated, [2.0, 4.0, 1.5])


def test_method_style_bounds_shadow_assignment_refreshes_cache():
    key = InteractionKey.bond("A", "B")
    pot = HarmonicPotential("A", "B", k=1.0, r0=2.0)
    ff = Forcefield({key: [pot]})

    lb_first, _ = ff.param_bounds
    ff[key][0].param_bounds = (
        np.array([0.5, 1.0], dtype=float),
        np.array([5.0, 3.0], dtype=float),
    )
    lb_updated, ub_updated = ff.param_bounds

    assert lb_updated is not lb_first
    np.testing.assert_allclose(lb_updated, [0.5, 1.0])
    np.testing.assert_allclose(ub_updated, [5.0, 3.0])


# ---------------------------------------------------------------------------
# describe_bounds (was DescribeBounds)
# ---------------------------------------------------------------------------

def test_describebounds_returns_string(p2p):
    p2p.build_bounds()
    s = p2p.describe_bounds()
    assert isinstance(s, str)
    assert len(s) > 0


def test_describebounds_contains_inf(p2p):
    p2p.build_bounds()
    s = p2p.describe_bounds()
    assert "inf" in s.lower() or "-inf" in s or "+inf" in s


def test_describebounds_only_bounded(p2p):
    p2p.build_bounds(
        pair_bounds={InteractionKey.pair("A", "B"): {"c0": (0.0, 5.0)}},
    )
    s = p2p.describe_bounds(only_bounded=True)
    assert isinstance(s, str)
    # Should show the bounded param
    assert "c0" in s


# ===========================================================================
# List[BasePotential] format tests
# ===========================================================================

@pytest.fixture
def p2p_list():
    """New-style Dict[K, List[BasePotential]] with one multi-pot key."""
    pA = MockPot([1.0, 2.0, 3.0], ["c0", "c1", "c2"])
    pB = MockPot([4.0, 5.0], ["c0", "c1"])
    pC = MockPot([6.0], ["c0"])
    return Forcefield({
        InteractionKey.pair("A", "B"): [pA],
        InteractionKey.pair("B", "C"): [pB, pC],
    })


def test_ffparamarray_list_format(p2p_list):
    arr = p2p_list.param_array()
    assert arr.shape == (6,)
    assert np.allclose(arr, [1, 2, 3, 4, 5, 6])


def test_ffparamindexmap_list_format(p2p_list):
    idx = p2p_list.param_index_map()
    assert len(idx) == 6
    # ("B","C") has 3 entries from two pots
    bc_entries = [e for e in idx if e[0] == InteractionKey.pair("B", "C")]
    assert len(bc_entries) == 3


def test_buildglobalmask_list_format(p2p_list):
    mask = p2p_list.build_mask()
    assert mask.shape == (6,)
    assert np.all(mask)


def test_buildglobalmask_freeze_list_format(p2p_list):
    # Freeze c0 of ("B","C") — applies to both pots in the list
    mask = p2p_list.build_mask(
        patterns={InteractionKey.pair("B", "C"): ["c0"]},
        mode="freeze",
    )
    assert mask[0] and mask[1] and mask[2]  # ("A","B") untouched
    assert not mask[3]  # ("B","C") pot0 c0 frozen
    assert mask[4]      # ("B","C") pot0 c1 trainable
    assert not mask[5]  # ("B","C") pot1 c0 frozen


# ===========================================================================
# derive_l1_mask (was DeriveL1Mask)
# ===========================================================================

def test_derive_l1_all_trainable(p2p):
    mask = np.ones(5, dtype=bool)
    l1 = p2p.derive_l1_mask(mask)
    assert l1[InteractionKey.pair("A", "B")] is True
    assert l1[InteractionKey.pair("B", "C")] is True


def test_derive_l1_all_frozen(p2p):
    mask = np.zeros(5, dtype=bool)
    l1 = p2p.derive_l1_mask(mask)
    assert l1[InteractionKey.pair("A", "B")] is False
    assert l1[InteractionKey.pair("B", "C")] is False


def test_derive_l1_partial_freeze(p2p):
    # Freeze ("B","C") entirely, keep ("A","B") partially active
    mask = np.array([True, False, False, False, False], dtype=bool)
    l1 = p2p.derive_l1_mask(mask)
    assert l1[InteractionKey.pair("A", "B")] is True
    assert l1[InteractionKey.pair("B", "C")] is False


def test_derive_l1_preserves_order(p2p):
    mask = np.ones(5, dtype=bool)
    l1 = p2p.derive_l1_mask(mask)
    assert list(l1.keys()) == [
        InteractionKey.pair("A", "B"),
        InteractionKey.pair("B", "C"),
    ]


def test_derive_l1_list_format(p2p_list):
    mask = np.zeros(6, dtype=bool)
    mask[5] = True  # Only last param (("B","C") pot1 c0) is trainable
    l1 = p2p_list.derive_l1_mask(mask)
    assert l1[InteractionKey.pair("A", "B")] is False
    assert l1[InteractionKey.pair("B", "C")] is True


def test_buildmask_rejects_tuple_keyed_patterns(p2p):
    with pytest.raises(TypeError, match="patterns must be keyed by InteractionKey"):
        p2p.build_mask(patterns={("A", "B"): ["c0"]}, mode="freeze")


def test_buildbounds_rejects_tuple_keyed_pair_bounds(p2p):
    with pytest.raises(TypeError, match="pair_bounds must be keyed by InteractionKey"):
        p2p.build_bounds(pair_bounds={("A", "B"): {"c0": (0.0, 5.0)}})


def test_key_mask_rejects_tuple_keyed_mapping(p2p):
    with pytest.raises(TypeError, match="key_mask must be keyed by InteractionKey"):
        p2p.key_mask = {("A", "B"): False}


def test_derive_l1_rejects_wrong_length(p2p):
    with pytest.raises(ValueError, match="L2 mask shape must be"):
        p2p.derive_l1_mask(np.array([True, False], dtype=bool))


def test_update_params_rejects_wrong_length_and_preserves_state(p2p):
    old = p2p.param_array()
    with pytest.raises(ValueError, match="Parameter vector shape must be"):
        p2p.update_params(np.array([1.0, 2.0], dtype=float))
    np.testing.assert_allclose(p2p.param_array(), old)


def test_replacing_key_rederives_l1_mask_from_spliced_l2(p2p):
    ab = InteractionKey.pair("A", "B")
    bc = InteractionKey.pair("B", "C")
    p2p.key_mask = {ab: False, bc: True}

    p2p[ab] = MockPot([9.0], ["d0"])

    np.testing.assert_array_equal(
        p2p.param_mask,
        np.array([True, True, True], dtype=bool),
    )
    assert p2p.key_mask == {ab: False, bc: True}


def test_derive_l1_empty():
    l1 = Forcefield({}).derive_l1_mask(np.array([], dtype=bool))
    assert l1 == {}
