"""Assign colours to a finished structure's brick slots, deterministically.

The structure comes first and is colourless.  This module then hands each of
its bricks a colour out of a ``(part, colour)`` stock, and the result is the
input to both the LDraw writer and the preview -- one assignment, two outputs,
so the file a person downloads is the picture they looked at.

Four properties, and each one is a refusal rather than a hope:

**It never exceeds stock.**  Every assignment is a deduction from the existing
:class:`~src.inventory.engine.Inventory`, whose ``deduct`` raises when a key is
empty.  No second counter is kept here, so there is no way for this module's
idea of the remaining stock to drift from the engine's.

**It refuses by name when a shape's colours do not add up.**  If a structure
needs seven ``2x4`` and the stock holds four red and two blue, the answer is
"``2x4``: needs 7, has 6" -- not six coloured bricks and one invented one, and
not a silent partial assignment.  The check runs over every shape before a
single brick is coloured, so a refusal leaves the stock untouched.

**It is deterministic and re-runnable.**  Bricks are coloured in a fixed order
-- bottom layer first, then by position -- and within a brick the colours are
tried in a fixed order: the operator's preferences first, in the order they
gave them, then the rest of the palette in table order.  The same structure and
the same stock always produce the same file.

**A preference is a preference, not a requirement.**  A preferred colour that
runs out is followed by the next one, and the report says how many bricks got a
preferred colour and how many did not.  A caller who needs "all red or
nothing" can read that number; the assignment does not decide it for them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.colour.palette import (COLOUR_ORDER, ColourError, check_contract,
                                colour, ldraw_code)
from src.data.bricks import Brick
from src.inventory.engine import Inventory, InventoryError


class AssignError(ColourError):
    """The structure cannot be coloured from this stock.  Nothing was changed."""


@dataclass(frozen=True)
class ColouredBrick:
    index: int
    part: str
    colour_id: str
    ldraw: int
    preferred: bool

    def as_dict(self) -> dict:
        return {"index": self.index, "part": self.part,
                "colour_id": self.colour_id, "ldraw": self.ldraw,
                "preferred": self.preferred}


@dataclass(frozen=True)
class Assignment:
    """A complete colouring, plus the stock arithmetic that produced it."""

    bricks: tuple[ColouredBrick, ...]
    stocked: dict[tuple[str, str], int]
    used: dict[tuple[str, str], int]
    remaining: dict[tuple[str, str], int]
    preferences: tuple[str, ...]
    order: tuple[int, ...] = field(repr=False, default=())

    @property
    def preferred_count(self) -> int:
        return sum(1 for brick in self.bricks if brick.preferred)

    @property
    def non_preferred_count(self) -> int:
        return len(self.bricks) - self.preferred_count

    def colours(self) -> dict[int, int]:
        """``{brick index: LDraw colour code}`` -- what the writers take."""
        return {brick.index: brick.ldraw for brick in self.bricks}

    def colour_ids(self) -> dict[int, str]:
        return {brick.index: brick.colour_id for brick in self.bricks}

    def check_within_stock(self) -> None:
        """Refuse an assignment that used more of anything than was stocked."""
        for key, count in self.used.items():
            have = self.stocked.get(key, 0)
            if count > have:
                part, colour_id = key
                raise AssignError(
                    f"the assignment used {count} {colour_id} {part} and only "
                    f"{have} were stocked; this is a defect in the assigner, "
                    "not an input problem")
        for key, left in self.remaining.items():
            if left < 0:
                part, colour_id = key
                raise AssignError(
                    f"{colour_id} {part} is overdrawn by {-left}")

    def as_dict(self) -> dict:
        return {
            "bricks": [brick.as_dict() for brick in self.bricks],
            "n_bricks": len(self.bricks),
            "preferences": list(self.preferences),
            "preferred_bricks": self.preferred_count,
            "non_preferred_bricks": len(self.bricks) - self.preferred_count,
            "stocked": _flatten(self.stocked),
            "used": _flatten(self.used),
            "remaining": {key: value for key, value
                          in _flatten(self.remaining).items() if value},
            "within_stock": True,
            "determinism": (
                "bricks are coloured bottom layer first then by position, and "
                "colours are tried preferences-first then in palette order; "
                "the same structure and stock always give the same result"),
        }


def _flatten(counts: dict[tuple[str, str], int]) -> dict[str, int]:
    return {f"{part}:{colour_id}": value
            for (part, colour_id), value in sorted(counts.items())}


def parse_colour_stock(spec: str) -> dict[tuple[str, str], int]:
    """``"2x4:red:6,2x4:blue:2"`` -> ``{("2x4", "red"): 6, ...}``.

    Part normalisation is the vision class module's, so ``4x1`` and ``1x4``
    name one shape here exactly as they do everywhere else, and the two
    spellings of one shape-and-colour may not both be given.
    """
    from src.vision.classes import normalise_part

    check_contract()
    if not isinstance(spec, str) or not spec.strip():
        raise AssignError(
            "a colour stock is required, as part:colour:count entries, "
            "e.g. '2x4:red:6,1x2:yellow:4'")
    out: dict[tuple[str, str], int] = {}
    spelled: dict[tuple[str, str], str] = {}
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        pieces = [piece.strip() for piece in item.split(":")]
        if len(pieces) != 3:
            raise AssignError(
                f"{item!r} is not part:colour:count")
        raw_part, raw_colour, raw_count = pieces
        # Both lookups raise their own module's error type. Re-raised as one
        # here so a caller -- the UI in particular -- has a single exception to
        # catch for "this stock string is not usable", instead of three.
        try:
            part = normalise_part(raw_part)
        except ValueError as exc:
            raise AssignError(f"{item!r}: {exc}") from None
        try:
            colour_id = colour(raw_colour).colour_id
        except ValueError as exc:
            raise AssignError(f"{item!r}: {exc}") from None
        if not raw_count.isascii() or not raw_count.isdigit():
            raise AssignError(
                f"{raw_count!r} is not a whole number of {raw_part} "
                f"{raw_colour}")
        count = int(raw_count)
        if count < 1:
            raise AssignError(f"{item!r} stocks nothing")
        key = (part, colour_id)
        if key in out:
            raise AssignError(
                f"{raw_part!r} and {spelled[key]!r} are the same shape "
                f"({part}) in the same colour and draw on one stock; give it "
                "once")
        out[key] = count
        spelled[key] = raw_part
    if not out:
        raise AssignError("a colour stock is required")
    return out


def shape_totals(stock: dict[tuple[str, str], int]) -> Counter[str]:
    """Total pieces per shape, across all its colours."""
    out: Counter[str] = Counter()
    for (part, _colour_id), count in stock.items():
        out[part] += count
    return out


def check_feasible(bricks, stock: dict[tuple[str, str], int]) -> None:
    """Refuse before colouring anything if a shape cannot be covered."""
    need = Counter(brick.part for brick in bricks)
    have = shape_totals(stock)
    short = {part: (count, have.get(part, 0))
             for part, count in need.items() if have.get(part, 0) < count}
    if short:
        lines = [f"{part}: needs {want}, has {got}"
                 for part, (want, got) in sorted(short.items())]
        raise AssignError(
            "the colour stock cannot cover this structure, so nothing was "
            "coloured: " + "; ".join(lines)
            + ". No colour is invented to close the gap")


def brick_order(bricks) -> tuple[int, ...]:
    """The fixed order bricks are coloured in: bottom layer, then position.

    A stated order is what makes the result reproducible, and bottom-first
    matches the assembly order, so the colours a person sees in step one are
    the colours assigned first.
    """
    return tuple(sorted(
        range(len(bricks)),
        key=lambda i: (bricks[i].z, bricks[i].x, bricks[i].y,
                       bricks[i].h, bricks[i].w, i)))


def assign(bricks, stock: dict[tuple[str, str], int], *,
           preferences=()) -> Assignment:
    """Colour every brick, or refuse and colour none.

    ``preferences`` are colour ids in the order the operator prefers them.  A
    preference is honoured while that colour has stock in the right shape, and
    the report says how often it could not be.
    """
    check_contract()
    bricks = list(bricks)
    if not bricks:
        raise AssignError("there is no structure to colour")
    for brick in bricks:
        if not isinstance(brick, Brick):
            raise AssignError(
                f"a brick must be a Brick, not {type(brick).__name__}")
    for key in stock:
        if not isinstance(key, tuple) or len(key) != 2:
            raise AssignError("a colour stock is keyed by (part, colour_id)")
        colour(key[1])
    wanted: list[str] = []
    for name in preferences:
        colour_id = colour(name).colour_id
        if colour_id not in wanted:
            wanted.append(colour_id)
    check_feasible(bricks, stock)

    engine = Inventory()
    engine.counts.update({key: int(value) for key, value in stock.items()})
    order = brick_order(bricks)
    tried_order = tuple(wanted) + tuple(
        name for name in COLOUR_ORDER if name not in wanted)

    coloured: list[ColouredBrick] = []
    engine.begin()
    try:
        for index in order:
            part = bricks[index].part
            chosen = None
            for colour_id in tried_order:
                if engine.available(part, colour_id) > 0:
                    engine.deduct(part, colour_id)
                    chosen = colour_id
                    break
            if chosen is None:
                # check_feasible already ruled this out for whole shapes, so
                # reaching here means the totals were right and the per-colour
                # arithmetic still failed -- which would be a defect in this
                # function rather than an input problem. Say which.
                raise AssignError(
                    f"no colour of {part} is left for brick {index} even "
                    "though the shape total was sufficient; this is a defect "
                    "in the assigner")
            coloured.append(ColouredBrick(
                index=index, part=part, colour_id=chosen,
                ldraw=ldraw_code(chosen), preferred=chosen in wanted))
    except (AssignError, InventoryError):
        engine.rollback()
        raise
    engine.commit()

    used: Counter[tuple[str, str]] = Counter(
        (brick.part, brick.colour_id) for brick in coloured)
    remaining = {key: int(stock.get(key, 0)) - used.get(key, 0)
                 for key in set(stock) | set(used)}
    result = Assignment(
        bricks=tuple(sorted(coloured, key=lambda b: b.index)),
        stocked={key: int(value) for key, value in stock.items()},
        used=dict(used), remaining=remaining,
        preferences=tuple(wanted), order=order)
    result.check_within_stock()
    if len(result.bricks) != len(bricks):
        raise AssignError(
            f"{len(result.bricks)} of {len(bricks)} bricks were coloured; a "
            "partial assignment is not returned")
    return result


def uniform_stock(bricks, colour_id: str) -> dict[tuple[str, str], int]:
    """Exactly enough of one colour to build this structure.

    The convenience case: a person who wants the whole thing in one colour, and
    the demonstration path that needs a stock which is certain to fit.
    """
    key = colour(colour_id).colour_id
    return {(part, key): count
            for part, count in Counter(b.part for b in bricks).items()}
