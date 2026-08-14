import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bricks import PART_VOCAB, parse_bricks
from src.inventory.engine import Inventory, InventoryError, consume, report


class TestBasics:
    def test_from_parts(self):
        inv = Inventory.from_parts({"1x2": 4, "2x4": 2})
        assert inv.available("1x2") == 4
        assert inv.available("1x1") == 0
        assert inv.total() == 6

    def test_rejects_negative(self):
        with pytest.raises(InventoryError):
            Inventory.from_parts({"1x2": -1})

    def test_from_structure_merges_rotations(self):
        inv = Inventory.from_structure(parse_bricks("1x4 (0,0,0)\n4x1 (0,6,0)"))
        assert inv.available("1x4") == 2

    def test_deduct_and_exhaust(self):
        inv = Inventory.from_parts({"1x2": 1})
        inv.deduct("1x2")
        assert inv.available("1x2") == 0
        with pytest.raises(InventoryError):
            inv.deduct("1x2")

    def test_never_goes_negative(self):
        inv = Inventory.from_parts({"1x2": 2})
        for _ in range(2):
            inv.deduct("1x2")
        with pytest.raises(InventoryError):
            inv.deduct("1x2")
        assert all(v >= 0 for v in inv.counts.values())


class TestTransactions:
    def test_rollback_restores_exactly(self):
        inv = Inventory.from_parts({"1x2": 5, "2x4": 3})
        before = dict(inv.counts)
        inv.begin()
        inv.deduct("1x2")
        inv.deduct("1x2")
        inv.deduct("2x4")
        inv.rollback()
        assert dict(inv.counts) == before

    def test_commit_keeps_changes(self):
        inv = Inventory.from_parts({"1x2": 5})
        inv.begin()
        inv.deduct("1x2")
        inv.commit()
        assert inv.available("1x2") == 4

    def test_nested_rollback(self):
        inv = Inventory.from_parts({"1x2": 5})
        inv.begin()
        inv.deduct("1x2")
        inv.begin()
        inv.deduct("1x2")
        inv.rollback()          # inner only
        assert inv.available("1x2") == 4
        inv.rollback()          # outer
        assert inv.available("1x2") == 5

    def test_nested_commit_propagates_to_outer_log(self):
        inv = Inventory.from_parts({"1x2": 5})
        inv.begin()
        inv.begin()
        inv.deduct("1x2")
        inv.commit()
        inv.rollback()          # outer rollback must undo the committed inner
        assert inv.available("1x2") == 5

    def test_rollback_without_begin(self):
        with pytest.raises(InventoryError):
            Inventory.from_parts({"1x2": 1}).rollback()

    def test_consume_is_atomic(self):
        inv = Inventory.from_parts({"1x2": 1})
        before = dict(inv.counts)
        with pytest.raises(InventoryError):
            consume(inv, parse_bricks("1x2 (0,0,0)\n1x2 (0,4,0)"))
        assert dict(inv.counts) == before, "failed build must leave no trace"


class TestMaskState:
    def test_bitmask_roundtrip(self):
        inv = Inventory.from_parts({"1x1": 1, "2x4": 1})
        bits = inv.mask_state()
        assert bits == (1 << PART_VOCAB.index("1x1")) | (1 << PART_VOCAB.index("2x4"))

    def test_exhausted_part_drops_out(self):
        inv = Inventory.from_parts({"1x1": 1})
        assert "1x1" in inv.available_parts()
        inv.deduct("1x1")
        assert "1x1" not in inv.available_parts()
        assert inv.mask_state() == 0

    def test_state_space_is_256(self):
        """Only 2**8 masks exist, so they can all be precomputed offline."""
        assert len(PART_VOCAB) == 8
        assert Inventory.from_parts({p: 1 for p in PART_VOCAB}).mask_state() == 255


class TestMatching:
    def test_can_build_and_missing(self):
        inv = Inventory.from_parts({"1x2": 6, "2x2": 1})
        req = Counter({"1x2": 4, "2x2": 2})
        assert not inv.can_build(req)
        assert inv.missing(req) == {"2x2": 1}

    def test_exact_fit(self):
        req = Counter({"1x2": 4})
        assert Inventory.from_parts({"1x2": 4}).can_build(req)

    def test_report_three_lists(self):
        inv = Inventory.from_parts({"1x2": 5, "2x4": 2})
        r = report(inv, parse_bricks("1x2 (0,0,0)\n2x1 (0,4,0)\n2x4 (5,5,0)"))
        assert r["used"] == {"1x2": 2, "2x4": 1}
        assert r["remaining"] == {"1x2": 3, "2x4": 1}
        assert r["valid"]

    def test_report_flags_overdraw(self):
        inv = Inventory.from_parts({"1x2": 1})
        r = report(inv, parse_bricks("1x2 (0,0,0)\n1x2 (0,4,0)"))
        assert not r["valid"]
        assert r["overdrawn"] == {"1x2": 1}
