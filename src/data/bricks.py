"""Parsing and normalisation for StableText2Brick structures.

Text format is one brick per line: ``hxw (x,y,z)``.

Every brick is exactly 1 unit tall.  ``h`` is the extent along the x axis and
``w`` the extent along y, so a brick occupies the cells

    [x, x+h) x [y, y+w) x {z}

That axis assignment is not stated in the dataset card; it is forced by the
bounds check (see ``scripts/01_eda.py`` -- the opposite assignment puts
thousands of bricks outside the 20x20x20 world).

The raw text contains both orientations of the same physical part (``1x4`` and
``4x1``).  Those are one inventory item, so ``canonical_part`` sorts the two
extents and orientation is tracked separately.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

WORLD = 20

#: Largest extent appearing in any part (1x8 / 8x1).
MAX_EXTENT = 8

#: The eight physical parts, written with the short extent first.
PART_VOCAB: tuple[str, ...] = (
    "1x1",
    "1x2",
    "1x4",
    "1x6",
    "1x8",
    "2x2",
    "2x4",
    "2x6",
)

PART_INDEX = {p: i for i, p in enumerate(PART_VOCAB)}

_LINE = re.compile(r"^\s*(\d+)x(\d+)\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)\s*$")


class ParseError(ValueError):
    """Raised when a line does not match the brick grammar."""


@dataclass(frozen=True, slots=True)
class Brick:
    h: int  # extent along x
    w: int  # extent along y
    x: int
    y: int
    z: int

    @property
    def part(self) -> str:
        """Canonical part id, orientation-independent."""
        return canonical_part(self.h, self.w)

    @property
    def rotated(self) -> bool:
        """True when the text form is the long-side-first spelling (``4x1``)."""
        return self.h > self.w

    @property
    def cells(self) -> list[tuple[int, int, int]]:
        return [
            (self.x + dx, self.y + dy, self.z)
            for dx in range(self.h)
            for dy in range(self.w)
        ]

    @property
    def footprint(self) -> set[tuple[int, int]]:
        """2-D cells covered in this brick's layer."""
        return {
            (self.x + dx, self.y + dy)
            for dx in range(self.h)
            for dy in range(self.w)
        }

    def in_bounds(self, world: int = WORLD) -> bool:
        return (
            0 <= self.x
            and 0 <= self.y
            and 0 <= self.z < world
            and self.x + self.h <= world
            and self.y + self.w <= world
        )

    def __str__(self) -> str:
        return f"{self.h}x{self.w} ({self.x},{self.y},{self.z})"


def canonical_part(h: int, w: int) -> str:
    """``(4, 1) -> "1x4"``.  Both orientations map to one inventory item."""
    lo, hi = (h, w) if h <= w else (w, h)
    return f"{lo}x{hi}"


def is_valid_part(part: str) -> bool:
    """Vocabulary check for generated output.

    Expects the canonical spelling, so callers must normalise first.  Note
    plausible-looking sizes such as ``2x8`` are *not* in the set -- the eight
    parts are not the full cross product of the extents that appear.
    """
    return part in PART_INDEX


def parse_line(line: str) -> Brick:
    m = _LINE.match(line)
    if not m:
        raise ParseError(f"unparseable brick line: {line!r}")
    h, w, x, y, z = (int(g) for g in m.groups())
    return Brick(h=h, w=w, x=x, y=y, z=z)


def parse_bricks(text: str, *, strict: bool = True) -> list[Brick]:
    """Parse a whole ``bricks`` field.

    With ``strict=False`` unparseable lines are skipped instead of raising,
    which is what the EDA pass wants so it can count them.
    """
    out: list[Brick] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            out.append(parse_line(line))
        except ParseError:
            if strict:
                raise
    return out


def required_inventory(bricks: list[Brick]) -> Counter[str]:
    """Parts needed to build this structure, keyed by canonical part id."""
    return Counter(b.part for b in bricks)


def occupancy(bricks: list[Brick]) -> dict[tuple[int, int, int], int]:
    """Map cell -> index of the brick occupying it.  Later bricks overwrite,
    so this is only meaningful together with :func:`find_collisions`."""
    grid: dict[tuple[int, int, int], int] = {}
    for i, b in enumerate(bricks):
        for c in b.cells:
            grid[c] = i
    return grid


def find_collisions(bricks: list[Brick]) -> list[tuple[int, int]]:
    """Pairs of brick indices sharing at least one cell."""
    seen: dict[tuple[int, int, int], int] = {}
    hits: set[tuple[int, int]] = set()
    for i, b in enumerate(bricks):
        for c in b.cells:
            j = seen.get(c)
            if j is not None:
                hits.add((j, i))
            else:
                seen[c] = i
    return sorted(hits)


def layers(bricks: list[Brick]) -> dict[int, list[Brick]]:
    out: dict[int, list[Brick]] = {}
    for b in bricks:
        out.setdefault(b.z, []).append(b)
    return out


def studs_connected(a: Brick, b: Brick) -> bool:
    """True when two bricks are actually joined.

    In LEGO two bricks side by side in the same layer touch but do **not**
    connect; a real joint is a stud/tube coupling between adjacent layers.
    Since every brick here is one unit tall and axis aligned, that reduces to
    "the layers differ by one and the 2-D footprints overlap".
    """
    if abs(a.z - b.z) != 1:
        return False
    return bool(a.footprint & b.footprint)


def connected_components(bricks: list[Brick], *, ground: bool = False) -> list[list[int]]:
    """Components under :func:`studs_connected` (union-find over brick indices).

    The project definition of "connected" is stud coupling alone, which is what
    ``ground=False`` (the default) computes: a baseplate is not a part, carries
    no inventory and is not written to the output, so it must not be what holds
    a model together.

    ``ground=True`` additionally joins every brick resting on ``z == 0``.  That
    is a *separate* question -- whether the structure is anchored -- and it is
    reported as its own metric.  It must never be substituted for ``connected``:
    it merges components that share no studs, so an assembly that would fall
    into pieces when lifted off the baseplate would pass.  The two criteria give
    materially different answers on real data; see
    scripts/08_corpus_structure_study.py for current figures.
    """
    n = len(bricks)
    parent = list(range(n + 1))     # last slot is the ground node
    GROUND = n

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Only bricks in adjacent layers can join, so bucket by z first.
    by_z: dict[int, list[int]] = {}
    for i, b in enumerate(bricks):
        by_z.setdefault(b.z, []).append(i)

    for z, here in by_z.items():
        above = by_z.get(z + 1, [])
        for i in here:
            if ground and bricks[i].z == 0:
                union(i, GROUND)
            for j in above:
                if bricks[i].footprint & bricks[j].footprint:
                    union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def is_connected(bricks: list[Brick], *, ground: bool = False) -> bool:
    """Whether the whole structure hangs together by studs alone.

    Defaults to the project definition; pass ``ground=True`` only to measure
    anchoring, never to decide acceptance.
    """
    if not bricks:
        return False
    return len(connected_components(bricks, ground=ground)) == 1


def touches_ground(bricks: list[Brick]) -> bool:
    return any(b.z == 0 for b in bricks)


def unsupported_bricks(bricks: list[Brick]) -> list[int]:
    """Bricks above the ground with nothing directly beneath them.

    Reported, not enforced: most corpus structures contain at least one, so
    rejecting them would throw away most of the data and shift the training
    distribution away from what the base model was fitted to.  A brick with no
    support below can still be held by one above it -- connectivity and support
    are different questions.  Figures, and the populations they belong to, are
    in scripts/08_corpus_structure_study.py; quote them from there rather than
    from memory.
    """
    by_z: dict[int, list[Brick]] = {}
    for b in bricks:
        by_z.setdefault(b.z, []).append(b)
    out = []
    for i, b in enumerate(bricks):
        if b.z == 0:
            continue
        if not any(b.footprint & x.footprint for x in by_z.get(b.z - 1, [])):
            out.append(i)
    return out


def format_bricks(bricks: list[Brick]) -> str:
    return "\n".join(str(b) for b in bricks)
