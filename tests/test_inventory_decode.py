"""Tests for the inventory gate.

These use a synthetic Slots so the base model never loads.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constraints.inventory_decode import (
    InventoryGate,
    _extents_for_mask,
    _widths_for,
)
from src.data.bricks import PART_VOCAB, canonical_part
from src.generation.brickgpt import MAX_DIM, TOKENS_PER_BRICK, Slots, sample
from src.inventory.engine import Inventory

# Token ids are arbitrary but distinct; 1..8 map to dims, 100.. to positions.
SLOTS = Slots(
    dims=list(range(1, MAX_DIM + 1)),
    posns=list(range(100, 120)),
    literal_x=200,
    literal_open=201,
    literal_comma=202,
    literal_close=203,
    eos=999,
)
DIM_TOKEN = {i: i for i in range(1, MAX_DIM + 1)}


def gate_for(parts: dict[str, int]) -> InventoryGate:
    return InventoryGate(SLOTS, Inventory.from_parts(parts))


class TestAvailabilityTables:
    def test_extents_cover_both_orientations(self):
        inv = Inventory.from_parts({"2x4": 1})
        assert _extents_for_mask(inv.mask_state()) == (2, 4)

    def test_width_narrows_once_h_is_fixed(self):
        m = Inventory.from_parts({"2x4": 1, "1x8": 1}).mask_state()
        assert _widths_for(m, 2) == (4,)
        assert _widths_for(m, 1) == (8,)
        assert _widths_for(m, 8) == (1,)

    def test_empty_stock_offers_nothing(self):
        assert _extents_for_mask(Inventory.from_parts({}).mask_state()) == ()

    def test_full_stock_offers_every_extent(self):
        m = Inventory.from_parts({p: 1 for p in PART_VOCAB}).mask_state()
        assert _extents_for_mask(m) == (1, 2, 4, 6, 8)

    def test_tables_cover_all_256_states(self):
        """The availability space is finite and small, so it is fully cached."""
        for mask in range(256):
            for h in _extents_for_mask(mask):
                assert _widths_for(mask, h), (mask, h)


class TestSlotGating:
    def test_slot0_offers_only_stocked_extents_plus_eos(self):
        g = gate_for({"2x4": 1})
        assert sorted(g.allowed(0, [])) == [2, 4, SLOTS.eos]

    def test_slot2_depends_on_sampled_h(self):
        g = gate_for({"2x4": 1, "1x2": 1})
        # h == 2 was sampled two steps back. w == 1 spells "2x1", the rotated
        # form of the 1x2 in stock, so it has to stay open alongside w == 4.
        assert sorted(g.allowed(2, [2, SLOTS.literal_x])) == [1, 4]
        # h == 1 rules out 2x4 entirely; only "1x2" remains reachable.
        assert sorted(g.allowed(2, [1, SLOTS.literal_x])) == [2]

    def test_slot2_excludes_widths_whose_part_is_absent(self):
        g = gate_for({"1x2": 1})
        assert 4 not in g.allowed(2, [1, SLOTS.literal_x])   # 1x4 not stocked

    def test_literal_slots_unaffected(self):
        g = gate_for({"2x4": 1})
        assert g.allowed(1, [2]) == [SLOTS.literal_x]
        assert g.allowed(3, [2, 200, 4]) == [SLOTS.literal_open]
        assert g.allowed(9, []) == [SLOTS.literal_close]

    def test_position_slots_are_bounded(self):
        g = gate_for({"2x4": 1})
        for slot in (4, 6, 8):
            assert g.allowed(slot, []) == SLOTS.posns
            assert len(g.allowed(slot, [])) == 20

    def test_exhausted_stock_leaves_only_eos(self):
        g = gate_for({"1x1": 1})
        g.on_brick(1, 1)
        assert g.allowed(0, [1, 200, 1]) == [SLOTS.eos]
        assert g.stop_reason == "inventory_exhausted"

    def test_slot2_raises_if_slot0_offered_an_impossible_h(self):
        g = gate_for({"2x4": 1})
        g.inventory.deduct("2x4")          # drained behind the gate's back
        with pytest.raises(RuntimeError, match="no in-stock width"):
            g.allowed(2, [2, SLOTS.literal_x])


class TestDeduction:
    def test_on_brick_decrements(self):
        g = gate_for({"2x4": 2})
        g.on_brick(2, 4)
        assert g.inventory.available("2x4") == 1

    def test_rotated_spelling_hits_the_same_counter(self):
        g = gate_for({"2x4": 2})
        g.on_brick(4, 2)
        assert g.inventory.available("2x4") == 1
        assert g.accepted == ["2x4"]

    def test_part_leaves_the_mask_when_it_runs_out(self):
        g = gate_for({"2x4": 1, "1x2": 1})
        assert 4 in g.allowed(0, [])
        g.on_brick(2, 4)
        assert 4 not in g.allowed(0, [])
        assert 2 in g.allowed(0, [])          # 1x2 still in stock

    def test_cannot_overdraw(self):
        from src.inventory.engine import InventoryError

        g = gate_for({"1x1": 1})
        g.on_brick(1, 1)
        with pytest.raises(InventoryError):
            g.on_brick(1, 1)

    def test_gate_never_offers_a_part_it_cannot_afford(self):
        """Walk the stock down and check the invariant at every step."""
        g = gate_for({p: 1 for p in PART_VOCAB})
        for _ in range(len(PART_VOCAB)):
            extents = [t for t in g.allowed(0, []) if t != SLOTS.eos]
            if not extents:
                break
            h = extents[0]
            w = g.allowed(2, [h, SLOTS.literal_x])[0]
            part = canonical_part(h, w)
            assert g.inventory.available(part) > 0, part
            g.on_brick(h, w)
        assert g.inventory.total() == 0


class TestSampling:
    """MPS multinomial can draw outside a sparse distribution's support, so
    sampling is restricted to the candidate list before normalising."""

    def test_only_ever_returns_an_allowed_token(self):
        torch.manual_seed(0)
        logits = torch.randn(1000)
        allowed = [5, 17, 999]
        for _ in range(200):
            assert sample(logits, allowed, 1.0) in allowed

    def test_respects_the_distribution(self):
        logits = torch.full((100,), -10.0)
        logits[7] = 10.0
        picks = [sample(logits, [7, 42], 0.6) for _ in range(50)]
        assert picks.count(7) > 45

    def test_single_candidate_is_deterministic(self):
        logits = torch.randn(100)
        assert all(sample(logits, [3], 0.6) == 3 for _ in range(20))

    def test_zero_temperature_does_not_divide_by_zero(self):
        logits = torch.randn(100)
        assert sample(logits, [1, 2, 3], 0.0) in (1, 2, 3)


def test_brick_is_ten_tokens():
    assert TOKENS_PER_BRICK == 10
