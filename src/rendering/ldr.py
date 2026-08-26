"""LDraw (.ldr) export.

The coordinate conversion and part ids are taken from the BrickGPT reference
implementation (``src/brickgpt/data/brick_structure.py``) so that files written
here are byte-identical to the official ones; ``tests/test_ldr.py`` pins that
with the golden vector from their own test suite.

The official package is not installed as a dependency: it requires ``bpy``
(Blender) which pins ``numpy<2`` and conflicts with the rest of this project,
and none of the Blender rendering path is needed here.

LDraw conventions that the numbers below encode:

* 1 stud = 20 LDU horizontally, 1 brick = 24 LDU tall.
* Y points *down*, so stacking upward means going negative.
* A brick's origin sits at the centre of its footprint, hence the ``+ h/2``.
"""

from __future__ import annotations

from pathlib import Path

from src.data.bricks import Brick

#: canonical part -> LDraw part file, from BrickGPT's brick_library.json
PART_TO_LDRAW: dict[str, str] = {
    "1x1": "3005.DAT",
    "1x2": "3004.DAT",
    "1x4": "3010.DAT",
    "1x6": "3009.DAT",
    "1x8": "3008.DAT",
    "2x2": "3003.DAT",
    "2x4": "3001.DAT",
    "2x6": "2456.DAT",
}

#: Orientation matrices; ori 0 is the short-side-along-x spelling.
_MATRIX = {
    0: "0 0 1 0 1 0 -1 0 0",
    1: "-1 0 0 0 1 0 0 0 -1",
}

LDU_STUD = 20
LDU_BRICK = 24
DEFAULT_COLOUR = 115


def brick_line(
    brick: Brick, *, colour: int = DEFAULT_COLOUR, base_height: float = 0
) -> str:
    """One Type-1 line, with no step marker.

    Split out from :func:`brick_to_ldr` so the assembly export can group
    several bricks under one ``0 STEP`` without a second copy of the
    coordinate conversion.  The conversion is the part that is pinned against
    the reference implementation, and there is exactly one of it.
    """
    x = (brick.x + brick.h * 0.5) * LDU_STUD
    z = (brick.y + brick.w * 0.5) * LDU_STUD
    y = (brick.z + base_height) * -LDU_BRICK
    ori = 1 if brick.h > brick.w else 0
    part = PART_TO_LDRAW[brick.part]
    return f"1 {colour} {x} {y} {z} {_MATRIX[ori]} {part}\n"


def brick_to_ldr(
    brick: Brick, *, colour: int = DEFAULT_COLOUR, base_height: float = 0
) -> str:
    """One Type-1 line plus a ``0 STEP`` marker."""
    return brick_line(brick, colour=colour,
                      base_height=base_height) + "0 STEP\n"


def to_ldr(
    bricks: list[Brick],
    *,
    colours: dict[int, int] | None = None,
    base_height: float = 0,
) -> str:
    """Serialise a structure.

    ``colours`` optionally maps brick index -> LDraw colour code, which is how
    the colour-assignment stage feeds its result in; without it everything is
    rendered in the single default colour, matching BrickGPT.
    """
    out = []
    for i, b in enumerate(bricks):
        colour = (colours or {}).get(i, DEFAULT_COLOUR)
        out.append(brick_to_ldr(b, colour=colour, base_height=base_height))
    return "".join(out)


def to_ldr_steps(
    steps: list[list[int]],
    bricks: list[Brick],
    *,
    colours: dict[int, int] | None = None,
    base_height: float = 0,
) -> str:
    """Serialise a structure with one ``0 STEP`` per assembly step.

    ``steps`` are lists of indices into ``bricks``, in build order.  This is
    what ``0 STEP`` is for: a viewer walks the file a step at a time, so the
    steps in the file have to be the steps a person would actually build.
    The default writer emits one marker per brick, which is a step per brick;
    this one takes the order from the assembly planner instead.

    Every brick must appear exactly once across the steps.  A missing or
    repeated index would produce a file that is not the structure it claims
    to be, so it is refused rather than written.
    """
    seen: list[int] = [index for step in steps for index in step]
    if sorted(seen) != list(range(len(bricks))):
        raise ValueError(
            f"the steps cover {len(seen)} index(es) for {len(bricks)} bricks; "
            "every brick must appear exactly once in exactly one step")
    out = []
    for step in steps:
        if not step:
            raise ValueError("an assembly step with no bricks is not a step")
        for index in step:
            out.append(brick_line(
                bricks[index], colour=(colours or {}).get(index,
                                                          DEFAULT_COLOUR),
                base_height=base_height))
        out.append("0 STEP\n")
    return "".join(out)


def write_ldr(path: str | Path, bricks: list[Brick], **kw) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_ldr(bricks, **kw), encoding="utf-8")
    return p
