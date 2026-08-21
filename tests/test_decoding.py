"""Tests for the brick grammar used by constrained decoding.

Only the tokenizer is loaded (small, ungated); the base model is not, so these
run without a Llama licence.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bricks import WORLD
from src.generation.brickgpt import (
    MAX_DIM,
    TOKENS_PER_BRICK,
    Slots,
    build_prompt,
    parse_output,
)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("AvaLovelace/BrickGPT")
    except Exception as e:                      # offline with a cold cache
        pytest.skip(f"tokenizer unavailable: {type(e).__name__}")


@pytest.fixture(scope="module")
def slots(tok):
    return Slots.build(tok)


class TestSingleTokenAssumption:
    """The ten-slot schedule only works if each field is exactly one token.

    The reference implementation documents this as a warning; pinning it here
    means a tokenizer change fails loudly instead of silently corrupting the
    grammar.
    """

    def test_dimensions(self, tok):
        for i in range(1, MAX_DIM + 1):
            assert len(tok.encode(str(i), add_special_tokens=False)) == 1

    def test_coordinates(self, tok):
        for i in range(WORLD):
            assert len(tok.encode(str(i), add_special_tokens=False)) == 1

    def test_literals(self, tok):
        for s in ("x", " (", ",", ")\n"):
            assert len(tok.encode(s, add_special_tokens=False)) == 1, repr(s)


class TestSlots:
    def test_schedule_length(self):
        assert TOKENS_PER_BRICK == 10

    def test_eos_only_at_slot_zero(self, slots):
        assert slots.eos in slots.allowed(0)
        for s in range(1, TOKENS_PER_BRICK):
            assert slots.eos not in slots.allowed(s), f"slot {s} could truncate a brick"

    def test_dimension_slots(self, slots):
        assert set(slots.allowed(2)) == set(slots.dims)
        assert len(slots.dims) == MAX_DIM

    def test_position_slots(self, slots):
        for s in (4, 6, 8):
            assert set(slots.allowed(s)) == set(slots.posns)
        assert len(slots.posns) == WORLD

    def test_literal_slots(self, slots):
        assert slots.allowed(1) == [slots.literal_x]
        assert slots.allowed(3) == [slots.literal_open]
        assert slots.allowed(5) == slots.allowed(7) == [slots.literal_comma]
        assert slots.allowed(9) == [slots.literal_close]

    def test_every_slot_defined(self, slots):
        for s in range(TOKENS_PER_BRICK):
            assert slots.allowed(s)

    def test_rejects_bad_slot(self, slots):
        with pytest.raises(ValueError):
            slots.allowed(TOKENS_PER_BRICK)

    def test_positions_stop_at_world_edge(self, slots, tok):
        """20 is out of bounds and must not be offered."""
        assert tok.encode("20", add_special_tokens=False)[0] not in slots.posns


class TestParseOutput:
    def test_clean_output(self):
        bricks, bad = parse_output("2x4 (0,0,0)\n1x2 (4,0,0)\n")
        assert len(bricks) == 2 and bad == []

    def test_collects_prose(self):
        bricks, bad = parse_output("Here is a chair:\n2x4 (0,0,0)\nEnjoy!")
        assert len(bricks) == 1
        assert bad == ["Here is a chair:", "Enjoy!"]

    def test_truncated_line_is_not_a_brick(self):
        bricks, bad = parse_output("2x4 (0,0,0)\n1x2 (4,0")
        assert len(bricks) == 1 and bad == ["1x2 (4,0"]

    def test_rotated_spelling_survives(self):
        bricks, _ = parse_output("8x1 (0,0,0)")
        assert bricks[0].part == "1x8"


def test_prompt_lists_both_orientations():
    p = build_prompt("A chair.")
    for spelling in ("2x4", "4x2", "1x8", "8x1"):
        assert spelling in p
    assert "A chair." in p
