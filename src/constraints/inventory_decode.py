"""Inventory-gated decoding -- the D/E arm's hard constraint.

Extends the brick grammar so a part that is out of stock cannot be emitted at
all.  Inventory violations become unreachable rather than detected: there is
nothing to reject and nothing to backtrack, because the tokens that would spell
an unavailable part are masked out before sampling.

The gate sits on two of the ten slots::

    slot 0   h   -> only extents that some in-stock part still uses
    slot 2   w   -> only extents w where canonical(h, w) is in stock

Both spellings of a part share one counter, so offering ``4`` at slot 0 means
``1x4`` *or* ``4x1`` may still be spelled; slot 2 then narrows it to whichever
actually has stock.  Stock is decremented when a brick completes, at the next
slot 0.

Only 2**8 = 256 availability states exist, so every slot-0 mask is precomputed
once and looked up by bitmask.  Slot 2 depends on the h just sampled, so it is
keyed by (bitmask, h) -- also a small, finite table.

What this does *not* cover: collision, support and connectivity depend on where
the brick lands, and connectivity is a property of the finished model, so they
stay with the rejection/rollback layer.  Bounds are already handled, since the
coordinate slots only ever offer 0-19.
"""

from __future__ import annotations

from functools import lru_cache

from src.data.bricks import MAX_EXTENT, PART_VOCAB, canonical_part
from src.generation.brickgpt import BrickGate, Slots
from src.inventory.engine import Inventory

#: Extents that appear in the vocabulary at all.
EXTENTS = tuple(range(1, MAX_EXTENT + 1))


@lru_cache(maxsize=None)
def _extents_for_mask(mask: int) -> tuple[int, ...]:
    """Extents usable as h, given which parts are in stock."""
    avail = {p for i, p in enumerate(PART_VOCAB) if mask >> i & 1}
    out = set()
    for part in avail:
        lo, hi = (int(v) for v in part.split("x"))
        out.add(lo)
        out.add(hi)
    return tuple(sorted(out))


@lru_cache(maxsize=None)
def _widths_for(mask: int, h: int) -> tuple[int, ...]:
    """Extents usable as w once h is fixed."""
    avail = {p for i, p in enumerate(PART_VOCAB) if mask >> i & 1}
    return tuple(w for w in EXTENTS if canonical_part(h, w) in avail)


class InventoryGate(BrickGate):
    """Brick grammar plus a hard stock gate.

    Stock is decremented as each brick completes, so the mask tightens as the
    pile runs down; when nothing is left, EOS becomes the only legal token and
    generation stops with ``inventory_exhausted``.
    """

    def __init__(self, slots: Slots, inventory: Inventory):
        super().__init__(slots)
        self.inventory = inventory
        self.accepted: list[str] = []
        #: Quantities as the prompt stated them, before any deduction.
        self.opening_inventory: dict[str, int] = inventory.as_dict()
        self._dim_token = {i + 1: tid for i, tid in enumerate(slots.dims)}
        self._dim_value = {tid: i + 1 for i, tid in enumerate(slots.dims)}

    def allowed(self, slot: int, out: list[int]) -> list[int]:
        if slot == 0:
            extents = _extents_for_mask(self.inventory.mask_state())
            if not extents:
                self.stop_reason = "inventory_exhausted"
                return [self.slots.eos]
            return [self._dim_token[h] for h in extents] + [self.slots.eos]
        if slot == 2:
            # h was sampled two steps ago, at slot 0.
            h = self._dim_value[out[-2]]
            widths = _widths_for(self.inventory.mask_state(), h)
            if not widths:
                raise RuntimeError(
                    f"no in-stock width for h={h}; slot 0 should not have offered it"
                )
            return [self._dim_token[w] for w in widths]
        return super().allowed(slot, out)

    def on_brick(self, h: int, w: int) -> None:
        part = canonical_part(h, w)
        self.inventory.deduct(part)
        self.accepted.append(part)


def generate_with_inventory(
    gpt,
    caption: str,
    inventory: Inventory,
    *,
    gate_cls: type[InventoryGate] = InventoryGate,
    **kw,
):
    """Run BrickGPT under a stock gate, with the matching prompt.

    The stock has to reach the model twice: once as the ``### Available Parts``
    block it reads, and once as the counter the gate enforces. Those must be
    the same numbers, so the opening quantities are snapshotted here before the
    gate starts spending them -- the gate mutates ``inventory`` in place, and
    handing the live object to the prompt builder would make what the model was
    told depend on when the string happened to be rendered.

    ``gate_cls`` lets a caller layer extra bookkeeping on the gate (the D-arm
    eval audits every sampled token) without rebuilding this pairing by hand.
    Constructing the gate at the call site is what let the two halves drift
    apart in the first place, so there is one path and subclasses go through it.

    Returns ``(Generation, gate)``. ``inventory`` is consumed; pass a copy if
    the caller still needs the opening position.
    """
    opening = inventory.as_dict()
    gate = gate_cls(gpt.slots, inventory)
    gate.opening_inventory = opening
    gen = gpt.generate(caption, inventory=opening, gate=gate, **kw)
    return gen, gate
