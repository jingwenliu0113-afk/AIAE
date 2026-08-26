"""Placement-gated decoding: collision and connectivity, frozen rules first.

**This module enforces collision. It does not enforce support and it does not
model physics.** ``stud_only_connected`` is a *connectivity* predicate --
adjacent-layer 2-D footprint overlap -- and nothing here may be described,
named, logged or reported as support or stability. Real stability analysis
needs a solver this project does not have (see PROJECT_STATUS, external
blockers); a connectivity count is not a weaker version of it, it is a
different question.

Nothing here claims collision or connectivity is a demonstrated cause of the
Core Success@K observed in Phase 2, and nothing here promises that gating them
raises it. Phase 2 measured marginal pass rates and did no causal
decomposition. The one relevant precedent points the other way:
``InventoryGate`` *lowered* the marginal ``in_bounds`` and ``collision_free``
rates (D against B, E against C), because constraining one axis moves the
others.

---------------------------------------------------------------------------
The frozen contract
---------------------------------------------------------------------------

**1. Collision: source and coordinate semantics.**
A brick occupies ``{(x+dx, y+dy, z) : 0 <= dx < h, 0 <= dy < w}`` -- the
``Brick.cells`` definition in :mod:`src.data.bricks`, reused, not restated.
``h`` extends along x, ``w`` along y, and every brick is one unit tall. Two
bricks collide when their cell sets intersect, which is exactly what
:func:`src.data.bricks.find_collisions` reports over a finished model. This
gate is the same predicate evaluated one brick earlier, so a model that
survives it has ``find_collisions() == []`` by construction.

**2. ``stud_only_connected``: exact semantics.**
Two bricks are connected when ``abs(a.z - b.z) == 1`` and their 2-D
footprints intersect (:func:`src.data.bricks.studs_connected`). The whole
model is connected when those edges alone leave one component
(``is_connected(bricks, ground=False)``). Same-layer side contact is **not**
connection. The ground is **not** a connector: ``ground=True`` answers a
different question and is never substituted here.

**3. Not support, not stability.** See the top of this docstring. Enforced by
a test that scans this module, its CLI surface and its test names for
``support``/``stability``.

**4. First brick, ground, layers.**
The first brick of a model has nothing to collide with, so collision never
constrains it. It is *not* required to sit at ``z == 0``: the project's
checkers report ``touches_ground`` separately and do not require it, and a
gate that required it would be enforcing a rule the scorer does not have.
Bricks in the same layer can collide (their cells coincide) but can never
connect. Bricks in adjacent layers can connect. Bricks two or more layers
apart can neither collide nor connect.

**5. Order of generation, and why connectivity is not enforced per brick.**
Collision is **monotone**: a placement that collides never stops colliding,
and one that does not can never start, because earlier bricks are never
moved. So the per-brick test and the final-model test agree exactly, and
gating it introduces no stricter semantics.

Connectivity is **not** monotone: a brick placed with nothing adjacent may be
joined later by a third brick. Requiring each new brick to touch the existing
structure would therefore be a *stricter* rule than the checker's, and it is
one this project has already ruled out -- CLAUDE.md, non-negotiable decision
4: two pillars joined at the end by a beam are a legal model. **This module
therefore never requires a new brick to connect.**

The only connectivity mode offered evaluates the checker's own final-model
predicate at **every slot 0 where EOS would otherwise be on offer**, before
anything is sampled -- not at the point the model asks to stop. It never
learns of such a point: withholding EOS happens while the candidate list is
being built, so a stop the gate prevented was never requested, and a stop it
allowed cannot be distinguished from any other continuation. Saying the
predicate runs "where the model asks to stop" would read in an observation
this layer does not make. The mode is off by default.

**6. What happens on a violation: masking, not rejection.**
Collision is decided before sampling, not after. At each slot the gate offers
only values that can still be completed into a legal placement, so an illegal
brick is unreachable rather than detected. There is **no rejection, no
resampling and no backtracking**, and the module deliberately has no such
mechanism. The mask sits on five slots -- 0, 2, 4, 6 and 8 -- and the last
one is a different kind of check from the other four:

- **Slots 0, 2, 4 and 6 are forward feasibility.** A value is offered only
  when some completion of the slots after it is still legal: an ``h`` some
  width can finish, a ``w`` some ``x`` can finish, an ``x`` some ``y`` can
  finish, a ``y`` some ``z`` can finish. That is what stops the decoder
  painting itself into a corner a backtrack would be needed to leave.
- **Slot 8 is the final coordinate mask.** ``z`` is the brick's last
  decision, so there is nothing further to look ahead to: the layers offered
  are exactly the ones the finished brick does not collide in. Feasibility at
  slot 6 is what guarantees that list is not empty. Here the mask settles the
  placement rather than preserving a choice for a later slot.

Completability at slot 0 is judged against the widths the *composing* gate
will actually offer at slot 2, not against ``w = 1``. ``w = 1`` is the most
permissive extent in the grammar, so it answers "does some width fit" -- but
under a stock gate it answers the wrong question, because ``w = 1`` may be
unpayable. An ``h`` is offered only when a width that is both spellable and
in stock also fits.

**7. Retry and backtrack limits.**
Not applicable, and therefore not implemented rather than set to zero. The
one bounded counter is connectivity's: :data:`MAX_EOS_DEFERRALS` is how many
times EOS may be **withheld from the candidate list**, and nothing else. It
is not a count of times the model asked to stop. This layer decides what may
be sampled; it never learns what would have been sampled from a list it did
not offer, so it cannot and does not claim the model ever chose EOS.

When that budget is spent the gate offers **EOS and nothing else**, and
termination is ``connectivity_unmet`` -- written in the same call that leaves
EOS as the only candidate. Writing the reason while bricks were still on
offer would be a lie waiting to happen: the decode loop keeps the first
reason a gate writes, so the model could go on, join its pieces and finish
connected under a reason saying it had not.

When no ``h`` can be completed at all the world is full and termination is
``space_exhausted``.

**8. Auditable counters.** :meth:`PlacementGate.counters` reports what this
layer actually did -- candidates masked per slot, bricks placed, EOS
deferrals. ``eos_deferrals`` is the number of times EOS was withheld from a
candidate list, per clause 7; reading it as stop attempts, or as evidence
the model would have stopped, reads in an observation this layer never
makes. The four counters this project has never implemented
(``candidate_rejections``, ``brick_retries``, ``previous_brick_backtracks``,
``physics_rollbacks``) stay ``None`` with ``implemented: False``, because
this layer performs no rejection, retry, backtrack or rollback. Reporting
them as ``0`` would claim they were counted and did not happen.

**9. Composition with InventoryGate, and the invariant that outranks
everything.** :class:`InventoryPlacementGate` applies stock first and
placement second, intersecting the two at slots 0 and 2. Stock is a veto:
placement may narrow what stock allows and may never widen it, so **no mode
can overspend the inventory**. If the intersection at slot 0 is empty the
gate stops rather than relaxing either constraint.

Slot-0 feasibility reads stock's own width table -- the same table slot 2
reads, not a second copy -- so an ``h`` survives slot 0 only when some width
that stock will still sell also fits in the space that is left.

**10. Opt-in.** Nothing on the existing decode path changes. The legacy
gates, entry points and control flow are untouched; a caller reaches this
layer only by naming one of its gate classes, and a gate constructed with
``enabled=False`` returns byte-identical candidate lists to its parent.

**11. Synthetic evidence only.** Every test here builds its own fixtures.
The Phase 2 160 cases are not read, and a test asserts that.

**12. No causal claim, no promise.** See the top of this docstring.
"""

from __future__ import annotations

from src.constraints.inventory_decode import InventoryGate, _widths_for
from src.data.bricks import WORLD, Brick
from src.generation.brickgpt import TOKENS_PER_BRICK, BrickGate, Slots

#: How many times connectivity may withhold EOS from the candidate list
#: before it gives up, forces EOS and records ``connectivity_unmet``. It
#: counts withheld offers, not stop attempts: nothing here observes what the
#: model would have sampled (contract clause 7). Bounded because refusing
#: forever is not a strategy: the model is under no obligation to place the
#: brick that would join the components, so an unbounded refusal spends the
#: token budget and stops on ``max_tokens`` with the disconnection intact and
#: the reason lost.
MAX_EOS_DEFERRALS = 8

#: Termination reasons this layer can add to :attr:`BrickGate.STOP_REASONS`.
PLACEMENT_STOP_REASONS = ("space_exhausted", "connectivity_unmet")

#: The counters this project has not built. Named here so the report shows
#: which were considered, and reported as null so nobody reads a 0 as a
#: measurement. This layer masks; it never rejects, retries or rolls back.
UNIMPLEMENTED_COUNTERS: tuple[str, ...] = (
    "candidate_rejections", "brick_retries",
    "previous_brick_backtracks", "physics_rollbacks",
)

CONNECTIVITY_MODES = ("off", "final_eos")


class PlacementRefused(RuntimeError):
    """A state this layer will not decode from. Nothing was sampled."""


class Occupancy:
    """Which cells are taken, and what can still be placed.

    Keyed by layer, because only same-layer cells can collide. That turns
    every feasibility question into a few small set intersections.
    """

    def __init__(self, world: int = WORLD):
        self.world = world
        self._layers: dict[int, set[tuple[int, int]]] = {}
        self.bricks: list[Brick] = []

    def add(self, brick: Brick) -> None:
        if not brick.in_bounds(self.world):
            raise PlacementRefused(f"{brick} is out of bounds")
        layer = self._layers.setdefault(brick.z, set())
        if layer & brick.footprint:
            raise PlacementRefused(f"{brick} collides with an earlier brick")
        layer |= brick.footprint
        self.bricks.append(brick)

    def _fits(self, h: int, w: int, x: int, y: int) -> bool:
        return 0 <= x and 0 <= y and x + h <= self.world and y + w <= self.world

    def _has_empty_layer(self) -> bool:
        """A layer with nothing in it takes any in-bounds footprint.

        The fast path, and the usual one: 80 bricks cannot fill 20 layers.
        Without it every slot-0 decision would scan the whole grid.
        """
        return len(self._layers) < self.world

    def free_z(self, h: int, w: int, x: int, y: int) -> list[int]:
        """Layers where this footprint lands without colliding."""
        if not self._fits(h, w, x, y):
            return []
        foot = {(x + dx, y + dy) for dx in range(h) for dy in range(w)}
        return [z for z in range(self.world)
                if not (self._layers.get(z, frozenset()) & foot)]

    def free_y(self, h: int, w: int, x: int) -> list[int]:
        if not (0 <= x and x + h <= self.world):
            return []
        if self._has_empty_layer():
            return [y for y in range(self.world) if y + w <= self.world]
        return [y for y in range(self.world) if self.free_z(h, w, x, y)]

    def free_x(self, h: int, w: int) -> list[int]:
        if h > self.world or w > self.world:
            return []
        if self._has_empty_layer():
            return [x for x in range(self.world) if x + h <= self.world]
        return [x for x in range(self.world) if self.free_y(h, w, x)]

    def any_placement(self, h: int, w: int) -> bool:
        return bool(self.free_x(h, w))


class PlacementRules:
    """The rules themselves, once, so the two gates cannot drift apart.

    Deliberately not a ``BrickGate``: composed into both gates rather than
    inherited by either. Two gates inheriting the same rules through
    different bases is how one of them quietly ends up with a different
    ``allowed`` and nobody notices until the numbers disagree.
    """

    def __init__(self, slots: Slots, *, enabled: bool, connectivity: str,
                 world: int, max_eos_deferrals: int):
        if connectivity not in CONNECTIVITY_MODES:
            raise PlacementRefused(
                f"connectivity={connectivity!r} is not one of "
                f"{list(CONNECTIVITY_MODES)}")
        if world < 1:
            raise PlacementRefused(f"world={world!r} has no cells")
        if max_eos_deferrals < 0:
            raise PlacementRefused(
                f"max_eos_deferrals={max_eos_deferrals!r} is negative")
        self.slots = slots
        self.enabled = enabled
        self.connectivity = connectivity
        self.world = world
        self.max_eos_deferrals = max_eos_deferrals
        self.occupancy = Occupancy(world)
        self._dim_value = {tid: i + 1 for i, tid in enumerate(slots.dims)}
        self._dim_token = {i + 1: tid for i, tid in enumerate(slots.dims)}
        self._pos_value = {tid: i for i, tid in enumerate(slots.posns)}
        #: Every width the grammar can spell. The default answer to "which
        #: widths may slot 2 offer", used when nothing else constrains it.
        self._all_widths = tuple(range(1, len(slots.dims) + 1))
        self.pending: Brick | None = None
        self.masked: dict[int, int] = {0: 0, 2: 0, 4: 0, 6: 0, 8: 0}
        self.bricks_placed = 0
        self.eos_deferrals = 0

    # -- reading the brick being spelled --------------------------------

    def decided(self, out: list[int], n: int) -> list[int]:
        """The values decided so far for the brick currently being spelled."""
        cur = out[len(out) - (len(out) % TOKENS_PER_BRICK):] if out else []
        vals = []
        for slot, tid in enumerate(cur):
            if slot in (0, 2):
                if tid not in self._dim_value:
                    raise PlacementRefused(
                        f"slot {slot} holds a token that is not an extent")
                vals.append(self._dim_value[tid])
            elif slot in (4, 6, 8):
                if tid not in self._pos_value:
                    raise PlacementRefused(
                        f"slot {slot} holds a token that is not a coordinate")
                vals.append(self._pos_value[tid])
        if len(vals) < n:
            raise PlacementRefused(
                f"{n} values were needed and {len(vals)} are decided")
        return vals[:n]

    # -- the mask -------------------------------------------------------

    def narrow(self, slot: int, out: list[int], base: list[int],
               stop, widths=None) -> list[int]:
        """Base candidates minus everything that cannot be completed.

        ``stop`` is called with a termination reason when the gate decides
        the model has to stop; the caller owns ``stop_reason``.

        ``widths`` is how the composing gate answers "which widths may slot 2
        offer for this h". It is a callable rather than a set because stock
        changes as the pile runs down, and slot 0 has to ask the question
        against the stock the model has *now*. Left out, every width in the
        grammar is spellable.
        """
        kept = self._narrow(slot, out, base, stop,
                            widths or self._every_width)
        self.masked[slot] = self.masked.get(slot, 0) + (len(base) - len(kept))
        if slot == TOKENS_PER_BRICK - 1:
            self.pending = self.read_pending(out)
        return kept

    def _every_width(self, h: int) -> tuple[int, ...]:
        return self._all_widths

    def _narrow(self, slot, out, base, stop, widths) -> list[int]:
        if slot == 0:
            return self._slot0(base, stop, widths)
        if slot == 2:
            (h,) = self.decided(out, 1)
            return self._slot2(h, base)
        if slot == 4:
            h, w = self.decided(out, 2)
            return self._keep_pos(base, self.occupancy.free_x(h, w))
        if slot == 6:
            h, w, x = self.decided(out, 3)
            return self._keep_pos(base, self.occupancy.free_y(h, w, x))
        if slot == 8:
            h, w, x, y = self.decided(out, 4)
            return self._keep_pos(base, self.occupancy.free_z(h, w, x, y))
        return list(base)

    def _completable(self, h: int, widths) -> bool:
        """Can this h be finished into a brick that fits and can be spelled?

        Asking ``any_placement(h, 1)`` is the tempting shortcut -- ``w = 1``
        is the most permissive extent, so an h that fails there fails
        everywhere. But the converse is what slot 0 needs, and it does not
        hold once something narrows the widths: under a stock gate ``w = 1``
        may be unpayable, and offering an h whose only payable widths do not
        fit hands slot 2 an empty list. So the question is asked against the
        widths that will actually be on offer.
        """
        return any(self.occupancy.any_placement(h, w) for w in widths(h))

    def _slot0(self, base: list[int], stop, widths) -> list[int]:
        eos = self.slots.eos
        kept = [t for t in base
                if t == eos or self._completable(self._dim_value[t], widths)]
        if not [t for t in kept if t != eos]:
            # Nothing that can be spelled also fits: the world is full as far
            # as this model is concerned. Stopping is the only move left, and
            # it gets its own reason rather than being folded into
            # normal_eos, which would say the model chose to stop.
            stop("space_exhausted")
            return [eos] if eos in base else []
        return self._defer_eos(kept, stop)

    def _slot2(self, h: int, base: list[int]) -> list[int]:
        kept = [t for t in base
                if self.occupancy.any_placement(h, self._dim_value[t])]
        if not kept:
            raise PlacementRefused(
                f"no width completes h={h}; slot 0 should not have offered it")
        return kept

    def _keep_pos(self, base: list[int], values: list[int]) -> list[int]:
        ok = set(values)
        kept = [t for t in base if self._pos_value[t] in ok]
        if not kept:
            raise PlacementRefused(
                "no coordinate completes this brick; an earlier slot should "
                "not have offered what led here")
        return kept

    # -- connectivity ---------------------------------------------------

    def _defer_eos(self, kept: list[int], stop) -> list[int]:
        """Hold EOS back while the model is in more than one piece.

        The checker's own final-model predicate, evaluated where the model
        asks to stop, and nowhere else. It is never applied per brick: a
        component joined by a later beam is a legal model, and requiring each
        brick to connect would make this gate stricter than the checker it
        is gating (CLAUDE.md, non-negotiable decision 4).
        """
        if self.connectivity != "final_eos":
            return kept
        from src.data.bricks import is_connected

        placed = self.occupancy.bricks
        if len(placed) < 2 or is_connected(placed, ground=False):
            return kept
        eos = self.slots.eos
        if self.eos_deferrals >= self.max_eos_deferrals:
            # The budget is spent, so stop here and stop for good. Writing
            # the reason while bricks were still on offer would let the model
            # build on, join its pieces and finish connected under a reason
            # saying it had not, because the decode loop keeps the first
            # reason a gate writes. Offering EOS alone makes the reason true
            # at the moment it is written and keeps it true afterwards.
            if eos not in kept:
                raise PlacementRefused(
                    "the deferral budget is spent and EOS is not on offer; "
                    "there is no state this layer could stop from")
            stop("connectivity_unmet")
            return [eos]
        without = [t for t in kept if t != eos]
        if not without:
            # Unreachable from _slot0, which stops with space_exhausted
            # before it gets here. Kept as a guard, and it takes the same
            # exit: withholding the only candidate would leave the decoder
            # nothing at all to sample.
            stop("connectivity_unmet")
            return kept
        self.eos_deferrals += 1
        return without

    # -- committing -----------------------------------------------------

    def read_pending(self, out: list[int]) -> Brick:
        h, w, x, y, z = self.decided(out, 5)
        return Brick(h=h, w=w, x=x, y=y, z=z)

    def commit(self, h: int, w: int) -> None:
        if self.pending is None:
            raise PlacementRefused(
                "a brick completed without its placement being read; the "
                "gate will not guess where it went")
        if (self.pending.h, self.pending.w) != (h, w):
            raise PlacementRefused(
                f"the completed brick says {h}x{w} and the placement read "
                f"{self.pending.h}x{self.pending.w}")
        self.occupancy.add(self.pending)
        self.pending = None
        self.bricks_placed += 1

    # -- audit ----------------------------------------------------------

    def counters(self) -> dict:
        out: dict = {
            "enabled": self.enabled,
            "connectivity": self.connectivity,
            "bricks_placed": self.bricks_placed,
            "eos_deferrals": self.eos_deferrals,
            "max_eos_deferrals": self.max_eos_deferrals,
            "candidates_masked": dict(self.masked),
            "candidates_masked_total": sum(self.masked.values()),
        }
        for name in UNIMPLEMENTED_COUNTERS:
            out[name] = {"value": None, "implemented": False}
        return out


class PlacementGate(BrickGate):
    """Collision as a mask, connectivity only where the checker asks.

    ``enabled=False`` makes this exactly its parent: the candidate lists come
    back token for token identical, so a caller can wire the gate in and
    leave the behaviour alone until it decides otherwise.
    """

    STOP_REASONS = BrickGate.STOP_REASONS + PLACEMENT_STOP_REASONS

    def __init__(self, slots: Slots, *, enabled: bool = False,
                 connectivity: str = "off", world: int = WORLD,
                 max_eos_deferrals: int = MAX_EOS_DEFERRALS):
        super().__init__(slots)
        self.rules = PlacementRules(
            slots, enabled=enabled, connectivity=connectivity, world=world,
            max_eos_deferrals=max_eos_deferrals)

    def _stop(self, reason: str) -> None:
        self.stop_reason = reason

    def allowed(self, slot: int, out: list[int]) -> list[int]:
        base = super().allowed(slot, out)
        if not self.rules.enabled:
            return base
        return self.rules.narrow(slot, out, base, self._stop)

    def on_brick(self, h: int, w: int) -> None:
        super().on_brick(h, w)
        if self.rules.enabled:
            self.rules.commit(h, w)

    def counters(self) -> dict:
        return self.rules.counters()


class InventoryPlacementGate(InventoryGate):
    """Stock first, placement second, and stock always wins.

    The order is the invariant. Placement narrows what stock allows and can
    never widen it, so no mode reachable from here can overspend the
    inventory. Once stock has ended the model, placement does not extend it
    and connectivity does not get to refuse the stop -- otherwise a
    connectivity deferral could talk the decoder into spelling a brick it
    cannot pay for.
    """

    STOP_REASONS = InventoryGate.STOP_REASONS + PLACEMENT_STOP_REASONS

    def __init__(self, slots: Slots, inventory, *, enabled: bool = False,
                 connectivity: str = "off", world: int = WORLD,
                 max_eos_deferrals: int = MAX_EOS_DEFERRALS):
        super().__init__(slots, inventory)
        self.rules = PlacementRules(
            slots, enabled=enabled, connectivity=connectivity, world=world,
            max_eos_deferrals=max_eos_deferrals)

    def _stop(self, reason: str) -> None:
        self.stop_reason = reason

    def allowed(self, slot: int, out: list[int]) -> list[int]:
        stock = super().allowed(slot, out)
        if not self.rules.enabled:
            return stock
        if self.stop_reason == "inventory_exhausted":
            return stock
        return self.rules.narrow(slot, out, stock, self._stop,
                                 widths=self._stock_widths)

    def _stock_widths(self, h: int) -> tuple[int, ...]:
        """Widths slot 2 will offer for this h, from stock's own table.

        The same lookup :class:`InventoryGate` uses at slot 2, called here
        rather than reimplemented: a second copy of "which widths are in
        stock" is how slot 0 and slot 2 come to disagree, and slot 2 has no
        way to recover when they do.
        """
        return _widths_for(self.inventory.mask_state(), h)

    def on_brick(self, h: int, w: int) -> None:
        super().on_brick(h, w)
        if self.rules.enabled:
            self.rules.commit(h, w)

    def counters(self) -> dict:
        out = self.rules.counters()
        out["inventory_opening"] = dict(self.opening_inventory)
        out["inventory_remaining"] = self.inventory.as_dict()
        out["parts_accepted"] = list(self.accepted)
        return out


def generate_raw_with_placement(gpt, caption: str, *, inventory=None,
                                enabled: bool = True,
                                connectivity: str = "off",
                                world: int = WORLD,
                                max_eos_deferrals: int = MAX_EOS_DEFERRALS,
                                **kw):
    """The opt-in entry point. Returns ``(RawGeneration, gate)``.

    With ``inventory`` this is the stock gate plus placement and reuses
    :func:`src.constraints.inventory_decode.generate_raw_with_inventory`
    verbatim, so the prompt and the counter are snapshotted by the same code
    that already does it. Without one it is placement alone.

    Nothing calls this by default. The existing decode path, its gates and
    its entry points are untouched; reaching this layer means naming it.
    """
    if inventory is None:
        gate = PlacementGate(gpt.slots, enabled=enabled,
                             connectivity=connectivity, world=world,
                             max_eos_deferrals=max_eos_deferrals)
        return gpt.generate_raw(caption, gate=gate, **kw), gate

    from src.constraints.inventory_decode import generate_raw_with_inventory

    def gate_cls(slots, inv):
        return InventoryPlacementGate(
            slots, inv, enabled=enabled, connectivity=connectivity,
            world=world, max_eos_deferrals=max_eos_deferrals)

    return generate_raw_with_inventory(gpt, caption, inventory,
                                       gate_cls=gate_cls, **kw)
