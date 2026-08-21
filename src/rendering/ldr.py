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


def brick_to_ldr(
    brick: Brick, *, colour: int = DEFAULT_COLOUR, base_height: float = 0
) -> str:
    """One Type-1 line plus a ``0 STEP`` marker."""
    x = (brick.x + brick.h * 0.5) * LDU_STUD
    z = (brick.y + brick.w * 0.5) * LDU_STUD
    y = (brick.z + base_height) * -LDU_BRICK
    ori = 1 if brick.h > brick.w else 0
    part = PART_TO_LDRAW[brick.part]
    return f"1 {colour} {x} {y} {z} {_MATRIX[ori]} {part}\n0 STEP\n"


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


def write_ldr(path: str | Path, bricks: list[Brick], **kw) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_ldr(bricks, **kw), encoding="utf-8")
    return p
