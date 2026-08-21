"""Inventory with transactional deduct/restore, for constrained decoding.

Keys are ``(part, colour)``.  The StableText2Brick track is colourless, so
colour defaults to ``None`` and every count is shape-only; the key shape is
already a pair so the colour post-processing stage can reuse this class
without a refactor.

Parts are canonical (``1x4``, never ``4x1``) -- see ``src.data.bricks``.  Half
of all bricks in the corpus use the rotated spelling, so anything that feeds
this class must normalise first or every count will be silently halved.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.data.bricks import PART_VOCAB, Brick, canonical_part

Key = tuple[str, str | None]


class InventoryError(RuntimeError):
    pass


@dataclass
class Inventory:
    """Mutable multiset of parts with an undo log."""

    counts: Counter[Key] = field(default_factory=Counter)
    _log: list[list[Key]] = field(default_factory=list, repr=False)

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_parts(cls, parts: dict[str, int]) -> "Inventory":
        inv = cls()
        for p, n in parts.items():
            if n < 0:
                raise InventoryError(f"negative quantity for {p}: {n}")
            if n:
                inv.counts[(p, None)] = n
        return inv

    @classmethod
    def from_structure(cls, bricks: list[Brick]) -> "Inventory":
        """Exact inventory: precisely what this structure consumes."""
        return cls.from_parts(dict(Counter(b.part for b in bricks)))

    def copy(self) -> "Inventory":
        return Inventory(counts=Counter(self.counts))

    # ---- queries ------------------------------------------------------------
    def available(self, part: str, colour: str | None = None) -> int:
        return self.counts.get((part, colour), 0)

    def has(self, part: str, colour: str | None = None) -> bool:
        return self.available(part, colour) > 0

    def available_parts(self) -> frozenset[str]:
        """Canonical parts with stock left -- the gate for decode masking."""
        return frozenset(p for (p, _c), n in self.counts.items() if n > 0)

    def mask_state(self) -> int:
        """Bitmask over PART_VOCAB of what is still in stock.

        Only 2**8 = 256 states exist, so the whole family of decode-time token
        masks can be precomputed offline and looked up by this integer.
        """
        avail = self.available_parts()
        bits = 0
        for i, p in enumerate(PART_VOCAB):
            if p in avail:
                bits |= 1 << i
        return bits

    def total(self) -> int:
        return sum(self.counts.values())

    # ---- mutation -----------------------------------------------------------
    def deduct(self, part: str, colour: str | None = None) -> None:
        key = (part, colour)
        if self.counts.get(key, 0) <= 0:
            raise InventoryError(f"out of stock: {part} ({colour})")
        self.counts[key] -= 1
        if self._log:
            self._log[-1].append(key)

    def restore(self, part: str, colour: str | None = None) -> None:
        self.counts[(part, colour)] += 1

    def can_build(self, required: Counter[str]) -> bool:
        return all(self.available(p) >= n for p, n in required.items())

    def missing(self, required: Counter[str]) -> Counter[str]:
        out: Counter[str] = Counter()
        for p, n in required.items():
            short = n - self.available(p)
            if short > 0:
                out[p] = short
        return out

    # ---- transactions -------------------------------------------------------
    def begin(self) -> None:
        """Open a savepoint.  Nestable."""
        self._log.append([])

    def commit(self) -> None:
        if not self._log:
            raise InventoryError("commit without begin")
        done = self._log.pop()
        if self._log:
            self._log[-1].extend(done)

    def rollback(self) -> None:
        """Undo every deduction since the matching :meth:`begin`."""
        if not self._log:
            raise InventoryError("rollback without begin")
        for key in reversed(self._log.pop()):
            self.counts[key] += 1

    # ---- reporting ----------------------------------------------------------
    def as_dict(self) -> dict[str, int]:
        return {p: n for (p, _c), n in sorted(self.counts.items()) if n}

    def __str__(self) -> str:
        return ", ".join(f"{p}:{n}" for p, n in self.as_dict().items()) or "(empty)"


def consume(inv: Inventory, bricks: list[Brick]) -> None:
    """Deduct a whole structure, rolling back entirely if it does not fit."""
    inv.begin()
    try:
        for b in bricks:
            inv.deduct(b.part)
    except InventoryError:
        inv.rollback()
        raise
    inv.commit()


def report(initial: Inventory, bricks: list[Brick]) -> dict:
    """initial / used / remaining, the three lists the UI has to show."""
    used = Counter(b.part for b in bricks)
    remaining = Counter(initial.as_dict())
    remaining.subtract(used)
    return {
        "initial": initial.as_dict(),
        "used": dict(sorted(used.items())),
        "remaining": {p: n for p, n in sorted(remaining.items()) if n},
        "valid": all(v >= 0 for v in remaining.values()),
        "overdrawn": {p: -n for p, n in sorted(remaining.items()) if n < 0},
    }
