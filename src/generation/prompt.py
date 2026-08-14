"""The one prompt builder, shared by training data and inference.

Both must produce byte-identical text: if the block a model was trained on
differs in any way from the block it is prompted with at evaluation time, the
A-E comparison measures that difference as well as the thing under test. So
there is a single implementation here and neither the data pipeline nor the
generator has its own.

The instruction body is BrickGPT's own wording. It carries the allowed
dimensions and the one-unit-tall rule the published checkpoint was trained
against; replacing it would push the un-finetuned arm off-distribution and
flatter every other arm.
"""

from __future__ import annotations

from src.data.bricks import PART_VOCAB

INSTRUCTION = (
    "Create a LEGO model of the input. Format your response as a list of bricks: "
    "<brick dimensions> <brick position>, where the brick position is (x,y,z).\n"
    "Allowed brick dimensions are 2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, "
    "6x1, 1x8, 8x1, 1x1, 2x2.\n"
    "All bricks are 1 unit tall.\n\n"
    "### Input:\n{caption}"
)

INVENTORY_HEADER = "### Available Parts"

#: Stated in the prompt because the model is shown one spelling and may emit
#: either -- 98.7% of targets use a rotated spelling somewhere. Without this
#: the listed ``1x4`` says nothing about whether ``4x1`` is allowed, or whether
#: the two draw on separate stock.
#:
#: Phrased as a general rule with one example rather than enumerating all six
#: rotation pairs. The enumeration cost 99 tokens against 47 here, which pushed
#: the longest sequences past the 2048 budget for no gain in clarity.
INVENTORY_RULE = (
    "Dimensions may be written in either order: 4x1 is the same part as 1x4 "
    "and draws on the same quantity.\n"
    "Use only the bricks listed above. Do not use more of any brick than the "
    "quantity given."
)

_MARKER = "### Input:"


def format_inventory(inventory: dict[str, int]) -> str:
    """One ``part: count`` line per stocked part, in vocabulary order.

    Fixed order rather than insertion order, so two inventories with the same
    contents render identically and sequence carries no signal.
    """
    lines = [f"{p}: {inventory[p]}" for p in PART_VOCAB if inventory.get(p, 0) > 0]
    if not lines:
        raise ValueError("inventory is empty")
    return "\n".join(lines)


def build_prompt(caption: str, inventory: dict[str, int] | None = None) -> str:
    """The full instruction. Omit ``inventory`` for the arm-A prompt.

    The inventory block is the only difference between the two forms.
    """
    head, _, tail = INSTRUCTION.format(caption=caption).partition(_MARKER)
    if inventory is None:
        return head + _MARKER + tail
    block = f"{INVENTORY_HEADER}\n{format_inventory(inventory)}\n\n{INVENTORY_RULE}\n\n"
    return head + block + _MARKER + tail


def strip_inventory_block(prompt: str) -> str:
    """Recover the arm-A prompt from a conditioned one."""
    if INVENTORY_HEADER not in prompt:
        return prompt
    head, _, rest = prompt.partition(INVENTORY_HEADER)
    return head + rest.split("\n\n", 2)[2]
