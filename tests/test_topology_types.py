"""Tests for AceCG.topology.types.InteractionKey (§15.4 item A1)."""

import pytest

from AceCG.topology.types import InteractionKey


# ===========================================================================
# InteractionKey
# ===========================================================================

class TestInteractionKeyConstruction:
    def test_pair(self):
        k = InteractionKey.pair("A", "B")
        assert k.style == "pair"
        assert k.types == ("A", "B")

    def test_pair_canonical(self):
        assert InteractionKey.pair("B", "A") == InteractionKey.pair("A", "B")

    def test_bond(self):
        k = InteractionKey.bond("X", "Y")
        assert k.style == "bond"
        assert k.types == ("X", "Y")

    def test_bond_canonical(self):
        assert InteractionKey.bond("Y", "X") == InteractionKey.bond("X", "Y")

    def test_angle(self):
        k = InteractionKey.angle("A", "B", "C")
        assert k.style == "angle"
        assert k.types == ("A", "B", "C")

    def test_angle_canonical(self):
        assert InteractionKey.angle("C", "B", "A") == InteractionKey.angle("A", "B", "C")

    def test_dihedral(self):
        k = InteractionKey.dihedral("A", "B", "C", "D")
        assert k.style == "dihedral"
        assert k.types == ("A", "B", "C", "D")

    def test_dihedral_canonical(self):
        assert InteractionKey.dihedral("D", "C", "B", "A") == InteractionKey.dihedral("A", "B", "C", "D")


class TestInteractionKeyLabel:
    def test_pair_label(self):
        assert InteractionKey.pair("A", "B").label() == "pair:A:B"

    def test_angle_label(self):
        assert InteractionKey.angle("A", "B", "C").label() == "angle:A:B:C"


class TestInteractionKeyHashAndEquality:
    def test_hashable(self):
        k = InteractionKey.pair("A", "B")
        d = {k: 42}
        assert d[k] == 42

    def test_dict_key(self):
        k1 = InteractionKey.pair("A", "B")
        k2 = InteractionKey.pair("B", "A")
        d = {k1: "first"}
        assert d[k2] == "first"

    def test_different_styles_not_equal(self):
        kp = InteractionKey(style="pair", types=("A", "B"))
        kb = InteractionKey(style="bond", types=("A", "B"))
        assert kp != kb

    def test_set_dedup(self):
        s = {InteractionKey.pair("A", "B"), InteractionKey.pair("B", "A")}
        assert len(s) == 1
