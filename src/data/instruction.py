"""Instruction format: Sample -> (prompt, target) for training and evaluation.

One template serves every arm. The inventory block is optional, and that is
the *only* thing that differs between the unconditioned and the
inventory-conditioned arms::

    A            build_prompt(caption)
    B, C, D, E   build_prompt(caption, inventory)

Anything else varying between arms would confound the comparison the workflow
requires, so the wording outside the block is fixed.

Two deviations from the literal spec in workflow section 9.7, both to protect
that comparison:

* The body is BrickGPT's own instruction verbatim, not a fresh ``### Request``
  preamble. It carries the allowed dimensions and the one-unit-tall rule that
  the published checkpoint was trained against; replacing it would push the
  un-finetuned arm off-distribution and flatter every other arm. The spec's
  ``### Request`` maps to BrickGPT's ``### Input:`` and ``### Output`` to the
  assistant turn.
* Parts are named ``1x1``, not ``brick_1x1``, reusing the model's existing size
  vocabulary rather than introducing a second naming scheme. This does not make
  the strings identical: the model may answer a listed ``1x4`` with ``4x1``, and
  the two spellings resolve through ``canonical_part`` to one inventory entry
  sharing one quantity -- which is what the prompt's rotation rule states.

The target ends with the tokenizer's EOS: a model that never emits it can only
ever stop by running out of budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - annotations only
    # `Sample` is used in one annotation, and `from __future__ import
    # annotations` means annotations are never evaluated. Importing it at
    # runtime pulled in src.data.counterfactual -> src.data.retile -> OR-Tools,
    # so report 16's measured child loaded a constraint solver it never calls
    # -- and the source manifest either had to cover all of it or be wrong
    # about what the child runs.
    from src.data.counterfactual import Sample
from src.generation.prompt import (
    INVENTORY_HEADER,
    INVENTORY_RULE,
    build_prompt,
    format_inventory,
    strip_inventory_block,
)

__all__ = [
    "INVENTORY_HEADER", "INVENTORY_RULE", "build_prompt", "format_inventory",
    "strip_inventory_block", "Example", "encode", "decode_target",
]


@dataclass
class Example:
    prompt: str
    target: str
    sample_id: str
    pair_id: str
    split: str
    role: str
    variant: str
    object_id: str
    dropped_part: str | None
    inventory: dict[str, int]

    @classmethod
    def from_sample(cls, s: Sample, *, with_inventory: bool = True) -> "Example":
        return cls(
            prompt=build_prompt(s.caption, s.inventory if with_inventory else None),
            target=s.bricks_txt + "\n",
            sample_id=s.sample_id,
            pair_id=s.pair_id,
            split=s.split,
            role=s.role,
            variant=s.variant,
            object_id=s.object_id,
            dropped_part=s.dropped_part,
            inventory=s.inventory,
        )


def encode(tokenizer, ex: Example) -> dict:
    """Tokenise into ``input_ids``/``labels`` with the prompt masked out.

    Labels are ``-100`` across the prompt so loss is taken only on the bricks;
    training on the prompt would spend capacity reproducing an instruction that
    is always supplied at inference.
    """
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": ex.prompt}],
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    target_ids = tokenizer.encode(ex.target, add_special_tokens=False)
    target_ids = target_ids + [tokenizer.eos_token_id]

    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
        "n_prompt_tokens": len(prompt_ids),
        "n_target_tokens": len(target_ids),
    }


def decode_target(text: str):
    """Parse a generated target back into bricks (round-trip check)."""
    from src.data.bricks import parse_bricks

    return parse_bricks(text.strip(), strict=False)
