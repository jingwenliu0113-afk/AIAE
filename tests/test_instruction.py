import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bricks import PART_VOCAB, parse_bricks
from src.data.counterfactual import Sample
from src.data.instruction import (
    INVENTORY_HEADER,
    Example,
    build_prompt,
    decode_target,
    encode,
    format_inventory,
)

CAPTION = "A small red car."
INV = {"1x2": 8, "2x4": 4, "1x1": 10}


def listed_parts(prompt: str) -> set[str]:
    """Parts that appear as a `part: count` line in the inventory block."""
    if INVENTORY_HEADER not in prompt:
        return set()
    block = prompt.split(INVENTORY_HEADER, 1)[1].split("\n\n", 1)[0]
    return {ln.split(":")[0].strip() for ln in block.strip().splitlines() if ":" in ln}


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("AvaLovelace/BrickGPT")
    except Exception as e:  # offline with a cold cache
        pytest.skip(f"tokenizer unavailable: {type(e).__name__}")


def make_sample(**kw) -> Sample:
    base = dict(
        sample_id="s:0:control:exact", pair_id="s:0", role="control",
        variant="exact", object_id="o", structure_id="s", split="train",
        caption=CAPTION, caption_index=-1, inventory=dict(INV),
        used={"1x2": 2}, bricks_txt="1x2 (0,0,0)\n2x1 (0,4,0)",
        dropped_part=None, seed=0, solver_status="OPTIMAL", solve_seconds=0.1,
        n_bricks=2, n_cells=4,
    )
    return Sample(**{**base, **kw})


class TestInventoryBlock:
    def test_one_line_per_stocked_part(self):
        assert format_inventory(INV).splitlines() == ["1x1: 10", "1x2: 8", "2x4: 4"]

    def test_order_is_vocabulary_order_not_insertion_order(self):
        a = format_inventory({"2x4": 1, "1x1": 2})
        b = format_inventory({"1x1": 2, "2x4": 1})
        assert a == b

    def test_zero_quantities_are_omitted(self):
        assert "2x6" not in format_inventory({**INV, "2x6": 0})

    def test_empty_inventory_raises(self):
        with pytest.raises(ValueError):
            format_inventory({})

    def test_part_names_match_the_output_vocabulary(self):
        """The listed name is the string the model must type."""
        text = format_inventory({p: 1 for p in PART_VOCAB})
        for p in PART_VOCAB:
            assert f"{p}: 1" in text
        assert "brick_" not in text


class TestPromptComposition:
    def test_arms_differ_only_by_the_inventory_block(self):
        """Anything else varying would confound the A-E comparison."""
        plain = build_prompt(CAPTION)
        with_inv = build_prompt(CAPTION, INV)
        assert INVENTORY_HEADER not in plain
        assert INVENTORY_HEADER in with_inv
        # Removing the block must recover the arm-A prompt exactly.
        head, _, tail = with_inv.partition(INVENTORY_HEADER)
        rebuilt = head + tail.split("\n\n", 2)[2]
        assert rebuilt == plain

    def test_keeps_brickgpt_instruction_body(self):
        """The published checkpoint was trained against this wording."""
        for arm in (build_prompt(CAPTION), build_prompt(CAPTION, INV)):
            assert "Allowed brick dimensions are" in arm
            assert "All bricks are 1 unit tall." in arm
            assert "### Input:" in arm

    def test_caption_is_last(self):
        assert build_prompt(CAPTION, INV).rstrip().endswith(CAPTION)

    def test_inventory_precedes_the_caption(self):
        p = build_prompt(CAPTION, INV)
        assert p.index(INVENTORY_HEADER) < p.index("### Input:")

    def test_rule_sentence_present(self):
        assert "Do not use more of any brick" in build_prompt(CAPTION, INV)


class TestExample:
    def test_from_sample_carries_provenance(self):
        ex = Example.from_sample(make_sample())
        assert ex.split == "train" and ex.role == "control"
        assert ex.variant == "exact" and ex.object_id == "o"

    def test_target_is_the_brick_text(self):
        ex = Example.from_sample(make_sample())
        assert decode_target(ex.target) == parse_bricks("1x2 (0,0,0)\n2x1 (0,4,0)")

    def test_target_round_trips_rotations(self):
        ex = Example.from_sample(make_sample(bricks_txt="8x1 (0,0,0)"))
        bricks = decode_target(ex.target)
        assert bricks[0].part == "1x8" and bricks[0].h == 8

    def test_without_inventory_gives_arm_a_prompt(self):
        s = make_sample()
        assert Example.from_sample(s, with_inventory=False).prompt == build_prompt(
            s.caption
        )

    def test_target_never_contains_the_dropped_part(self):
        s = make_sample(role="counterfactual", dropped_part="2x6",
                        bricks_txt="1x2 (0,0,0)")
        ex = Example.from_sample(s)
        assert "2x6" not in ex.target
        # Check the listed quantities, not raw substrings: the rule sentence
        # names every part when it explains rotation sharing.
        assert "2x6" not in listed_parts(ex.prompt)


class TestEncoding:


    def test_prompt_is_masked_out_of_the_loss(self, tok):
        enc = encode(tok, Example.from_sample(make_sample()))
        n = enc["n_prompt_tokens"]
        assert enc["labels"][:n] == [-100] * n
        assert all(x != -100 for x in enc["labels"][n:])

    def test_lengths_line_up(self, tok):
        enc = encode(tok, Example.from_sample(make_sample()))
        assert len(enc["input_ids"]) == len(enc["labels"])
        assert len(enc["input_ids"]) == enc["n_prompt_tokens"] + enc["n_target_tokens"]

    def test_target_ends_with_eos(self, tok):
        enc = encode(tok, Example.from_sample(make_sample()))
        assert enc["input_ids"][-1] == tok.eos_token_id
        assert enc["labels"][-1] == tok.eos_token_id

    def test_unmasked_ids_decode_back_to_the_bricks(self, tok):
        s = make_sample()
        enc = encode(tok, Example.from_sample(s))
        kept = [i for i in enc["labels"] if i != -100]
        text = tok.decode(kept, skip_special_tokens=True)
        assert decode_target(text) == parse_bricks(s.bricks_txt)

    def test_inventory_block_cost_is_bounded(self, tok):
        """At most eight listed lines plus a fixed rule, so the block cannot
        blow up the budget. Per-row costs are reported in report 10."""
        a = encode(tok, Example.from_sample(make_sample(), with_inventory=False))
        b = encode(tok, Example.from_sample(
            make_sample(inventory={p: 12 for p in PART_VOCAB})))
        cost = b["n_prompt_tokens"] - a["n_prompt_tokens"]
        assert 0 < cost < 200

    def test_block_cost_grows_with_listed_parts(self, tok):
        one = encode(tok, Example.from_sample(make_sample(inventory={"1x1": 1})))
        eight = encode(tok, Example.from_sample(
            make_sample(inventory={p: 1 for p in PART_VOCAB})))
        assert eight["n_prompt_tokens"] > one["n_prompt_tokens"]


class TestRotationEquivalence:
    """A canonical inventory must legally cover a rotated target.

    The model is shown ``1x4`` and may emit ``4x1``; if the two drew on
    separate counts, or the prompt failed to say they do not, a legal build
    would look like a violation.
    """

    def test_rule_states_the_equivalence(self):
        """A general rule plus one example, not an enumeration.

        Enumerating all six pairs cost 52 more tokens and pushed the longest
        sequences past the 2048 budget without making the rule clearer.
        """
        rule = build_prompt(CAPTION, INV)
        assert "either order" in rule
        assert "same part as" in rule
        assert "same quantity" in rule

    def test_only_canonical_names_are_listed(self):
        """Rotated spellings never appear as their own line."""
        listed = listed_parts(build_prompt(CAPTION, {p: 3 for p in PART_VOCAB}))
        assert listed == set(PART_VOCAB)
        for rotated in ("4x1", "2x1", "6x1", "8x1", "4x2", "6x2"):
            assert rotated not in listed

    def test_rotated_target_is_covered_by_canonical_inventory(self):
        """1x4 in stock, 4x1 in the target: legal, and the counts agree."""
        s = make_sample(inventory={"1x4": 2}, used={"1x4": 2},
                        bricks_txt="4x1 (0,0,0)\n4x1 (0,2,0)")
        ex = Example.from_sample(s)
        bricks = decode_target(ex.target)
        assert [b.part for b in bricks] == ["1x4", "1x4"]
        assert [(b.h, b.w) for b in bricks] == [(4, 1), (4, 1)]
        assert "1x4: 2" in ex.prompt
        assert "4x1: " not in ex.prompt
        used = Counter(b.part for b in bricks)
        assert all(ex.inventory[p] >= n for p, n in used.items())

    def test_mixed_orientations_draw_on_one_count(self):
        s = make_sample(inventory={"1x2": 2}, used={"1x2": 2},
                        bricks_txt="1x2 (0,0,0)\n2x1 (4,0,0)")
        ex = Example.from_sample(s)
        used = Counter(b.part for b in decode_target(ex.target))
        assert used == {"1x2": 2}
        assert ex.inventory["1x2"] == 2

    def test_exhausting_stock_with_rotations_is_exact(self):
        """Both spellings together must not exceed the single quantity."""
        s = make_sample(inventory={"1x8": 2}, used={"1x8": 2},
                        bricks_txt="1x8 (0,0,0)\n8x1 (0,9,0)")
        used = Counter(b.part for b in decode_target(
            Example.from_sample(s).target))
        assert used["1x8"] == 2
