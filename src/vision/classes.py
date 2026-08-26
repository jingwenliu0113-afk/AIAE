"""The eight vision classes, derived rather than restated.

The project's part vocabulary lives in :mod:`src.data.bricks` and its LDraw
part files live in :mod:`src.rendering.ldr`.  A vision model needs a third
thing -- the label a public dataset uses for the same physical brick -- and the
temptation is to write a fresh table with all three columns in it.  That table
would be a second source of truth, and the first time somebody corrected one
of the eight rows in only one place, every downstream count would be wrong in
a way that looks like a model problem.

So the mapping here is *computed* from ``PART_TO_LDRAW``: the dataset labels
its directories with the official brick design number, and an LDraw part file
for a plain rectangular brick is that same number with ``.DAT`` after it.  If
a future edit changes ``PART_TO_LDRAW``, this module changes with it, and
:func:`check_contract` fails loudly if the derivation ever stops holding.

The vocabulary is fixed at these eight and does not grow.  A photograph of a
part outside the eight has to come back as *not one of ours* rather than as
the nearest of the eight, which is why :data:`UNKNOWN` exists as a value the
prediction schema can carry.
"""

from __future__ import annotations

from src.data.bricks import PART_INDEX, PART_VOCAB, canonical_part
from src.rendering.ldr import PART_TO_LDRAW

#: The label a prediction carries when the answer is "not one of the eight".
#: A value, not an exception: an operator correcting a detection has to be able
#: to say this about a box, and a report has to be able to count them.
UNKNOWN = "unknown"

#: The design number the public datasets label a directory with, per canonical
#: part.  Derived from the LDraw part file name, which is that number.
PART_TO_DESIGN: dict[str, str] = {
    part: PART_TO_LDRAW[part].rsplit(".", 1)[0] for part in PART_VOCAB
}

#: The reverse direction, which is what reading a dataset directory needs.
DESIGN_TO_PART: dict[str, str] = {
    design: part for part, design in PART_TO_DESIGN.items()
}

#: Model output order.  It is ``PART_VOCAB`` order and nothing else, so a
#: logit vector, a confusion matrix and an inventory table all index the same
#: way and a saved checkpoint cannot silently disagree with a report.
CLASS_ORDER: tuple[str, ...] = tuple(PART_VOCAB)

#: Index of each class in :data:`CLASS_ORDER`.
CLASS_INDEX: dict[str, int] = dict(PART_INDEX)

N_CLASSES = len(CLASS_ORDER)


class ClassError(ValueError):
    """A label, index or design number outside the eight-class contract."""


def check_contract() -> None:
    """Refuse a state where the three tables have drifted apart.

    Called by the model, the dataset reader and the tests.  A derivation that
    is never checked is a comment.
    """
    if tuple(PART_VOCAB) != CLASS_ORDER:
        raise ClassError(
            "CLASS_ORDER is no longer the part vocabulary; a vision model "
            "trained against the old order would be silently mislabelled")
    if set(PART_TO_LDRAW) != set(PART_VOCAB):
        raise ClassError(
            "PART_TO_LDRAW and PART_VOCAB name different parts; the vision "
            "class list cannot be derived from a mapping that disagrees with "
            "the vocabulary")
    if len(DESIGN_TO_PART) != len(PART_TO_DESIGN):
        raise ClassError(
            "two parts share a design number, so a dataset directory could "
            "not say which of them it holds")
    for part, design in PART_TO_DESIGN.items():
        if not design.isdigit():
            raise ClassError(
                f"design number {design!r} for {part} is not a number; the "
                "dataset labels its directories with the design number")
        if PART_TO_LDRAW[part] != f"{design}.DAT":
            raise ClassError(
                f"the LDraw part file for {part} is {PART_TO_LDRAW[part]!r}, "
                f"which is not {design!r} with .DAT after it; the derivation "
                "this module rests on no longer holds")
    if CLASS_INDEX != {name: i for i, name in enumerate(CLASS_ORDER)}:
        raise ClassError("CLASS_INDEX does not index CLASS_ORDER")


def design_numbers() -> tuple[str, ...]:
    """The eight dataset directory names, in class order."""
    return tuple(PART_TO_DESIGN[part] for part in CLASS_ORDER)


def part_of_design(design: str) -> str:
    """``"3001" -> "2x4"``.  Refuses a design number outside the eight."""
    if not isinstance(design, str):
        raise ClassError(f"a design number must be a string, not "
                         f"{type(design).__name__}")
    key = design.strip()
    if key not in DESIGN_TO_PART:
        raise ClassError(
            f"design number {design!r} is not one of the eight this project "
            f"covers: {' '.join(design_numbers())}")
    return DESIGN_TO_PART[key]


def design_of_part(part: str) -> str:
    """``"2x4" -> "3001"``.  Accepts either rotation spelling."""
    if not isinstance(part, str):
        raise ClassError(f"a part must be a string, not {type(part).__name__}")
    key = normalise_part(part)
    return PART_TO_DESIGN[key]


def normalise_part(part: str) -> str:
    """Canonicalise a part label, refusing anything outside the eight.

    ``4x1`` and ``1x4`` are one class, for the same reason they are one
    inventory item.  A vision class list with both would halve every count.
    """
    if not isinstance(part, str):
        raise ClassError(f"a part must be a string, not {type(part).__name__}")
    text = part.strip()
    head, sep, tail = text.partition("x")
    if not sep or not head.isdigit() or not tail.isdigit():
        raise ClassError(f"{part!r} is not a part name of the form HxW")
    canonical = canonical_part(int(head), int(tail))
    if canonical not in CLASS_INDEX:
        raise ClassError(
            f"{part!r} normalises to {canonical!r}, which is not one of the "
            f"eight: {' '.join(CLASS_ORDER)}")
    return canonical


def label_index(part: str) -> int:
    """Class index for a part label, after rotation normalisation."""
    return CLASS_INDEX[normalise_part(part)]


def index_label(index: int) -> str:
    """Part label for a class index."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise ClassError(f"a class index must be a whole number, not "
                         f"{type(index).__name__}")
    if not 0 <= index < N_CLASSES:
        raise ClassError(
            f"class index {index} is outside 0..{N_CLASSES - 1}")
    return CLASS_ORDER[index]
