"""The collision/connectivity layer, and every way it must refuse.

Synthetic fixtures only. Nothing here reads the Phase 2 test split or the
160 cases materialised from it, and :class:`TestNoPhase2Data` asserts that
by scanning this file rather than by promising.

The layer masks; it never rejects, resamples or backtracks. So the tests are
about what the candidate lists contain at each of the ten slots, and about
the invariants that hold whatever the model then samples.
"""

import itertools
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constraints.inventory_decode import InventoryGate
from src.constraints.placement_decode import (
    CONNECTIVITY_MODES, MAX_EOS_DEFERRALS, UNIMPLEMENTED_COUNTERS,
    InventoryPlacementGate, Occupancy, PlacementGate, PlacementRefused,
    PlacementRules, generate_raw_with_placement)
from src.data.bricks import (WORLD, Brick, canonical_part, find_collisions,
                             is_connected, required_inventory,
                             studs_connected)
from src.generation.brickgpt import (TOKENS_PER_BRICK, BrickGate, BrickGPT,
                                     Slots, parse_output)
from src.inventory.engine import Inventory

# --------------------------------------------------------------------------
# A Slots with distinct, checkable ids and no tokenizer.
# --------------------------------------------------------------------------

DIM_BASE, POS_BASE = 1000, 2000


def stub_slots() -> Slots:
    return Slots(
        dims=[DIM_BASE + i for i in range(8)],
        posns=[POS_BASE + i for i in range(WORLD)],
        literal_x=90, literal_open=91, literal_comma=92,
        literal_close=93, eos=99,
    )


def dim(v: int) -> int:
    return DIM_BASE + v - 1


def pos(v: int) -> int:
    return POS_BASE + v


def spell(h, w, x, y, z) -> list[int]:
    s = stub_slots()
    return [dim(h), s.literal_x, dim(w), s.literal_open,
            pos(x), s.literal_comma, pos(y),
            s.literal_comma, pos(z), s.literal_close]


def drive(gate, bricks) -> None:
    """Walk a gate through complete bricks the way the decode loop does."""
    out: list[int] = []
    for b in bricks:
        toks = spell(b.h, b.w, b.x, b.y, b.z)
        for slot in range(TOKENS_PER_BRICK):
            allowed = gate.allowed(slot, out)
            assert toks[slot] in allowed, (
                f"{b} slot {slot}: {toks[slot]} was masked out")
            out.append(toks[slot])
        gate.on_brick(b.h, b.w)


def offer(gate, out, slot) -> list[int]:
    return gate.allowed(slot, out)


def prefix(h=None, w=None, x=None, y=None, z=None) -> list[int]:
    s = stub_slots()
    seq = []
    if h is not None:
        seq += [dim(h), s.literal_x]
    if w is not None:
        seq += [dim(w), s.literal_open]
    if x is not None:
        seq += [pos(x), s.literal_comma]
    if y is not None:
        seq += [pos(y), s.literal_comma]
    if z is not None:
        seq += [pos(z)]
    return seq


def on_gate(**kw) -> PlacementGate:
    kw.setdefault("enabled", True)
    return PlacementGate(stub_slots(), **kw)


# --------------------------------------------------------------------------


class TestLegalPlacementsAreAccepted:
    def test_a_clear_grid_offers_every_in_bounds_coordinate(self):
        g = on_gate()
        xs = offer(g, prefix(h=2, w=4), 4)
        assert [t - POS_BASE for t in xs] == list(range(WORLD - 2 + 1))

    def test_the_first_brick_is_never_constrained_by_collision(self):
        """Nothing to collide with, so only bounds may narrow it."""
        g = on_gate()
        assert set(offer(g, prefix(h=1, w=1), 4)) == set(stub_slots().posns)
        zs = offer(g, prefix(h=1, w=1, x=0, y=0), 8)
        assert [t - POS_BASE for t in zs] == list(range(WORLD))

    def test_a_tower_of_legal_bricks_walks_through(self):
        g = on_gate()
        drive(g, [Brick(2, 4, 0, 0, z) for z in range(5)])
        assert g.rules.bricks_placed == 5
        assert find_collisions(g.rules.occupancy.bricks) == []

    def test_the_first_brick_may_sit_off_the_ground(self):
        """touches_ground is reported by the checker, not required by it."""
        g = on_gate()
        drive(g, [Brick(2, 2, 3, 3, 7)])
        assert g.rules.occupancy.bricks[0].z == 7


class TestCollidingPlacementsAreUnreachable:
    def test_the_occupied_layer_is_removed_from_z(self):
        g = on_gate()
        drive(g, [Brick(2, 2, 0, 0, 0)])
        zs = [t - POS_BASE for t in offer(g, prefix(h=2, w=2, x=0, y=0), 8)]
        assert 0 not in zs
        assert zs == [z for z in range(WORLD) if z != 0]

    def test_a_partial_overlap_also_removes_the_layer(self):
        """One shared cell is a collision; it need not be the whole brick."""
        g = on_gate()
        drive(g, [Brick(2, 2, 0, 0, 0)])
        zs = [t - POS_BASE for t in offer(g, prefix(h=2, w=2, x=1, y=1), 8)]
        assert 0 not in zs

    def test_a_footprint_that_only_touches_is_still_offered(self):
        """Edge contact is not overlap. 2x2 at (0,0) and (2,0) abut."""
        g = on_gate()
        drive(g, [Brick(2, 2, 0, 0, 0)])
        zs = [t - POS_BASE for t in offer(g, prefix(h=2, w=2, x=2, y=0), 8)]
        assert 0 in zs

    def test_nothing_sampled_can_ever_collide(self):
        g = on_gate()
        drive(g, [Brick(2, 2, 0, 0, 0), Brick(2, 2, 2, 0, 0),
                  Brick(1, 8, 0, 2, 0), Brick(2, 4, 0, 0, 1)])
        assert find_collisions(g.rules.occupancy.bricks) == []

    def test_a_collision_reaching_the_occupancy_is_refused(self):
        """Belt and braces under the mask, in case a caller bypasses it."""
        occ = Occupancy()
        occ.add(Brick(2, 2, 0, 0, 0))
        with pytest.raises(PlacementRefused):
            occ.add(Brick(2, 2, 1, 1, 0))


class TestBoundsAndFootprintEdges:
    def test_x_stops_where_the_brick_would_leave_the_world(self):
        g = on_gate()
        xs = [t - POS_BASE for t in offer(g, prefix(h=8, w=1), 4)]
        assert max(xs) == WORLD - 8

    def test_y_stops_where_the_brick_would_leave_the_world(self):
        g = on_gate()
        ys = [t - POS_BASE for t in offer(g, prefix(h=1, w=6, x=0), 6)]
        assert max(ys) == WORLD - 6

    def test_the_last_legal_cell_is_still_offered(self):
        g = on_gate()
        xs = [t - POS_BASE for t in offer(g, prefix(h=1, w=1), 4)]
        assert xs[-1] == WORLD - 1

    def test_rotation_swaps_which_axis_is_limited(self):
        """1x8 and 8x1 are one inventory part and two different footprints."""
        g = on_gate()
        wide = [t - POS_BASE for t in offer(g, prefix(h=1, w=8, x=0), 6)]
        tall = [t - POS_BASE for t in offer(g, prefix(h=8, w=1, x=0), 6)]
        assert max(wide) == WORLD - 8
        assert max(tall) == WORLD - 1

    def test_an_out_of_bounds_brick_is_refused_by_the_occupancy(self):
        occ = Occupancy()
        with pytest.raises(PlacementRefused):
            occ.add(Brick(2, 2, WORLD - 1, 0, 0))


class TestLayerSemantics:
    def test_same_layer_bricks_never_connect(self):
        """CLAUDE.md: side contact in one layer is not a joint."""
        assert not studs_connected(Brick(2, 2, 0, 0, 0), Brick(2, 2, 2, 0, 0))

    def test_adjacent_layers_with_overlap_connect(self):
        assert studs_connected(Brick(2, 2, 0, 0, 0), Brick(2, 2, 1, 0, 1))

    def test_two_layers_apart_neither_collide_nor_connect(self):
        g = on_gate()
        drive(g, [Brick(2, 2, 0, 0, 0)])
        zs = [t - POS_BASE for t in offer(g, prefix(h=2, w=2, x=0, y=0), 8)]
        assert 2 in zs
        assert not studs_connected(Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 2))

    def test_adjacent_layers_do_not_collide(self):
        g = on_gate()
        drive(g, [Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 1)])
        assert find_collisions(g.rules.occupancy.bricks) == []


class TestConnectivityIsNotEnforcedPerBrick:
    """Non-negotiable decision 4: two pillars joined at the end are legal."""

    def test_a_disconnected_brick_is_accepted_while_building(self):
        g = on_gate(connectivity="final_eos")
        drive(g, [Brick(1, 1, 0, 0, 0), Brick(1, 1, 10, 10, 0)])
        assert g.rules.bricks_placed == 2
        assert not is_connected(g.rules.occupancy.bricks, ground=False)

    def test_two_pillars_joined_by_a_later_beam_are_reachable(self):
        g = on_gate(connectivity="final_eos")
        drive(g, [Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 3, 0),
                  Brick(1, 4, 0, 0, 1)])
        assert is_connected(g.rules.occupancy.bricks, ground=False)

    def test_off_mode_never_touches_eos(self):
        g = on_gate(connectivity="off")
        drive(g, [Brick(1, 1, 0, 0, 0), Brick(1, 1, 10, 10, 0)])
        assert stub_slots().eos in offer(g, [] , 0)
        assert g.rules.eos_deferrals == 0


class TestConnectivityAtEos:
    def test_eos_is_withheld_while_the_model_is_in_pieces(self):
        g = on_gate(connectivity="final_eos")
        drive(g, [Brick(1, 1, 0, 0, 0), Brick(1, 1, 10, 10, 0)])
        assert stub_slots().eos not in offer(g, [], 0)
        assert g.rules.eos_deferrals == 1

    def test_eos_returns_once_the_pieces_are_joined(self):
        g = on_gate(connectivity="final_eos")
        drive(g, [Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 3, 0),
                  Brick(1, 4, 0, 0, 1)])
        assert stub_slots().eos in offer(g, [], 0)

    def test_a_single_brick_may_always_stop(self):
        g = on_gate(connectivity="final_eos")
        drive(g, [Brick(1, 1, 0, 0, 0)])
        assert stub_slots().eos in offer(g, [], 0)

    def test_deferrals_are_bounded_and_then_it_gives_up(self):
        g = on_gate(connectivity="final_eos", max_eos_deferrals=3)
        drive(g, [Brick(1, 1, 0, 0, 0), Brick(1, 1, 10, 10, 0)])
        for i in range(3):
            assert stub_slots().eos not in offer(g, [], 0)
            assert g.rules.eos_deferrals == i + 1
        assert stub_slots().eos in offer(g, [], 0)
        assert g.stop_reason == "connectivity_unmet"
        assert g.rules.eos_deferrals == 3

    def test_zero_deferrals_never_withholds(self):
        g = on_gate(connectivity="final_eos", max_eos_deferrals=0)
        drive(g, [Brick(1, 1, 0, 0, 0), Brick(1, 1, 10, 10, 0)])
        assert stub_slots().eos in offer(g, [], 0)
        assert g.stop_reason == "connectivity_unmet"

    def test_an_unknown_mode_is_refused(self):
        for bad in ("per_brick", "strict", "", None, "Final_EOS"):
            with pytest.raises(PlacementRefused):
                PlacementGate(stub_slots(), enabled=True, connectivity=bad)
        assert CONNECTIVITY_MODES == ("off", "final_eos")


class TestSpaceExhaustion:
    def fill(self, gate, world) -> None:
        drive(gate, [Brick(1, 1, x, y, z)
                     for z in range(world)
                     for x in range(world)
                     for y in range(world)])

    def test_a_full_world_stops_with_its_own_reason(self):
        g = on_gate(world=2)
        self.fill(g, 2)
        assert offer(g, [], 0) == [stub_slots().eos]
        assert g.stop_reason == "space_exhausted"

    def test_space_exhausted_is_not_normal_eos(self):
        g = on_gate(world=2)
        self.fill(g, 2)
        assert g.stop_reason != "normal_eos"
        assert "space_exhausted" in PlacementGate.STOP_REASONS

    def test_a_nearly_full_world_still_offers_what_fits(self):
        g = on_gate(world=2)
        drive(g, [Brick(1, 1, x, y, 0) for x in range(2) for y in range(2)])
        hs = offer(g, [], 0)
        assert dim(1) in hs
        assert dim(2) in hs
        zs = [t - POS_BASE for t in offer(g, prefix(h=1, w=1, x=0, y=0), 8)]
        assert zs == [1]


class TestCountersAreExact:
    def test_bricks_placed_counts_each_brick_once(self):
        g = on_gate()
        drive(g, [Brick(2, 2, 0, 0, z) for z in range(4)])
        assert g.counters()["bricks_placed"] == 4

    def test_masked_candidates_are_counted_per_slot(self):
        g = on_gate()
        drive(g, [Brick(2, 2, 0, 0, 0)])
        before = dict(g.counters()["candidates_masked"])
        offer(g, prefix(h=2, w=2, x=0, y=0), 8)
        after = g.counters()["candidates_masked"]
        assert after[8] == before[8] + 1, "exactly one z was removed"

    def test_nothing_is_counted_when_disabled(self):
        g = PlacementGate(stub_slots(), enabled=False)
        drive(g, [Brick(2, 2, 0, 0, 0)])
        c = g.counters()
        assert c["candidates_masked_total"] == 0
        assert c["bricks_placed"] == 0
        assert c["enabled"] is False

    def test_unimplemented_counters_are_null_not_zero(self):
        """A 0 would say it was counted and did not happen."""
        g = on_gate()
        drive(g, [Brick(2, 2, 0, 0, 0)])
        c = g.counters()
        assert set(UNIMPLEMENTED_COUNTERS) == {
            "candidate_rejections", "brick_retries",
            "previous_brick_backtracks", "physics_rollbacks"}
        for name in UNIMPLEMENTED_COUNTERS:
            assert c[name] == {"value": None, "implemented": False}, name

    def test_eos_deferrals_are_not_double_counted(self):
        g = on_gate(connectivity="final_eos")
        drive(g, [Brick(1, 1, 0, 0, 0), Brick(1, 1, 10, 10, 0)])
        offer(g, [], 0)
        offer(g, [], 0)
        assert g.counters()["eos_deferrals"] == 2

    def test_the_layer_never_claims_a_rejection_it_did_not_make(self):
        """It masks. There is no rejection path to count."""
        source = Path(__file__).resolve().parents[1] / "src" / "constraints" \
            / "placement_decode.py"
        text = source.read_text(encoding="utf-8")
        for banned in ("def resample", "def backtrack", "def rollback",
                       "def reject_brick"):
            assert banned not in text


class TestDisabledIsByteIdenticalToLegacy:
    def test_every_slot_matches_the_parent_gate(self):
        legacy, gated = BrickGate(stub_slots()), PlacementGate(stub_slots())
        out: list[int] = []
        for b in [Brick(2, 4, 0, 0, 0), Brick(2, 4, 0, 0, 1)]:
            toks = spell(b.h, b.w, b.x, b.y, b.z)
            for slot in range(TOKENS_PER_BRICK):
                assert gated.allowed(slot, out) == legacy.allowed(slot, out)
                out.append(toks[slot])
            legacy.on_brick(b.h, b.w)
            gated.on_brick(b.h, b.w)

    def test_a_collision_is_reachable_when_disabled(self):
        """Legacy behaviour is preserved exactly, warts included."""
        g = PlacementGate(stub_slots(), enabled=False)
        out: list[int] = []
        for b in [Brick(2, 2, 0, 0, 0), Brick(2, 2, 0, 0, 0)]:
            toks = spell(b.h, b.w, b.x, b.y, b.z)
            for slot in range(TOKENS_PER_BRICK):
                assert toks[slot] in g.allowed(slot, out)
                out.append(toks[slot])
            g.on_brick(b.h, b.w)

    def test_disabled_never_changes_stop_reason(self):
        g = PlacementGate(stub_slots(), enabled=False, connectivity="final_eos")
        drive(g, [Brick(1, 1, 0, 0, 0), Brick(1, 1, 10, 10, 0)])
        g.allowed(0, [])
        assert g.stop_reason == "running"


class TestInventoryGateAloneIsUnchanged:
    def parts(self):
        return {"2x4": 2, "1x2": 1}

    def test_the_combined_gate_disabled_matches_the_stock_gate(self):
        a = InventoryGate(stub_slots(), Inventory.from_parts(self.parts()))
        b = InventoryPlacementGate(stub_slots(),
                                   Inventory.from_parts(self.parts()),
                                   enabled=False)
        out: list[int] = []
        for br in [Brick(2, 4, 0, 0, 0), Brick(2, 4, 0, 0, 1)]:
            toks = spell(br.h, br.w, br.x, br.y, br.z)
            for slot in range(TOKENS_PER_BRICK):
                assert b.allowed(slot, out) == a.allowed(slot, out)
                out.append(toks[slot])
            a.on_brick(br.h, br.w)
            b.on_brick(br.h, br.w)
        assert b.inventory.as_dict() == a.inventory.as_dict()

    def test_stock_exhaustion_still_ends_the_model(self):
        g = InventoryPlacementGate(stub_slots(),
                                   Inventory.from_parts({"1x1": 1}),
                                   enabled=True)
        drive(g, [Brick(1, 1, 0, 0, 0)])
        assert g.allowed(0, []) == [stub_slots().eos]
        assert g.stop_reason == "inventory_exhausted"

    def test_connectivity_cannot_override_stock_exhaustion(self):
        """A deferral here would spell a brick the model cannot pay for."""
        g = InventoryPlacementGate(stub_slots(),
                                   Inventory.from_parts({"1x1": 2}),
                                   enabled=True, connectivity="final_eos")
        drive(g, [Brick(1, 1, 0, 0, 0), Brick(1, 1, 10, 10, 0)])
        assert not is_connected(g.rules.occupancy.bricks, ground=False)
        assert g.allowed(0, []) == [stub_slots().eos]
        assert g.stop_reason == "inventory_exhausted"
        assert g.rules.eos_deferrals == 0


class TestCombinedModeNeverOverspends:
    def test_stock_is_never_exceeded(self):
        parts = {"2x4": 3, "1x2": 2}
        g = InventoryPlacementGate(stub_slots(), Inventory.from_parts(parts),
                                   enabled=True)
        drive(g, [Brick(2, 4, 0, 0, 0), Brick(2, 4, 0, 4, 0),
                  Brick(1, 2, 0, 8, 0)])
        used = Counter(g.accepted)
        for part, n in parts.items():
            assert used[part] <= n
        assert g.inventory.as_dict() == {"2x4": 1, "1x2": 1}

    def test_placement_only_narrows_what_stock_offered(self):
        parts = {"2x4": 2, "1x1": 4}
        stock = InventoryGate(stub_slots(), Inventory.from_parts(parts))
        both = InventoryPlacementGate(stub_slots(),
                                      Inventory.from_parts(parts),
                                      enabled=True)
        drive(both, [Brick(2, 4, 0, 0, 0)])
        drive(stock, [Brick(2, 4, 0, 0, 0)])
        for slot, out in ((0, []), (4, prefix(h=2, w=4)),
                          (8, prefix(h=2, w=4, x=0, y=0))):
            narrowed = set(both.allowed(slot, out))
            offered = set(stock.allowed(slot, out)) if slot in (0,) \
                else set(BrickGate(stub_slots()).allowed(slot, out))
            assert narrowed <= offered, f"slot {slot} widened the candidates"

    def test_the_result_is_both_payable_and_collision_free(self):
        parts = {"2x2": 4}
        g = InventoryPlacementGate(stub_slots(), Inventory.from_parts(parts),
                                   enabled=True)
        drive(g, [Brick(2, 2, 0, 0, 0), Brick(2, 2, 2, 0, 0),
                  Brick(2, 2, 0, 0, 1)])
        bricks = g.rules.occupancy.bricks
        assert find_collisions(bricks) == []
        assert required_inventory(bricks) <= Counter(parts)


class TestFailClosed:
    def test_a_brick_completing_without_a_placement_is_refused(self):
        g = on_gate()
        with pytest.raises(PlacementRefused):
            g.on_brick(2, 4)

    def test_a_placement_that_disagrees_with_the_brick_is_refused(self):
        g = on_gate()
        out: list[int] = []
        for slot, tok in enumerate(spell(2, 4, 0, 0, 0)):
            g.allowed(slot, out)
            out.append(tok)
        with pytest.raises(PlacementRefused):
            g.on_brick(1, 1)

    def test_a_truncated_brick_is_refused(self):
        rules = PlacementRules(stub_slots(), enabled=True, connectivity="off",
                               world=WORLD, max_eos_deferrals=1)
        with pytest.raises(PlacementRefused):
            rules.decided(prefix(h=2, w=4), 5)

    def test_a_non_extent_token_at_an_extent_slot_is_refused(self):
        rules = PlacementRules(stub_slots(), enabled=True, connectivity="off",
                               world=WORLD, max_eos_deferrals=1)
        with pytest.raises(PlacementRefused):
            rules.decided([pos(0), 90], 1)

    def test_a_non_coordinate_token_at_a_coordinate_slot_is_refused(self):
        rules = PlacementRules(stub_slots(), enabled=True, connectivity="off",
                               world=WORLD, max_eos_deferrals=1)
        with pytest.raises(PlacementRefused):
            rules.decided(prefix(h=2, w=4) + [dim(1), 92], 3)

    def test_a_degenerate_world_is_refused(self):
        for bad in (0, -1):
            with pytest.raises(PlacementRefused):
                PlacementGate(stub_slots(), enabled=True, world=bad)

    def test_a_negative_deferral_budget_is_refused(self):
        with pytest.raises(PlacementRefused):
            PlacementGate(stub_slots(), enabled=True, connectivity="final_eos",
                          max_eos_deferrals=-1)


class TestDeterminism:
    def sequence(self, seed_bricks):
        g = on_gate(connectivity="final_eos")
        seen = []
        out: list[int] = []
        for b in seed_bricks:
            toks = spell(b.h, b.w, b.x, b.y, b.z)
            for slot in range(TOKENS_PER_BRICK):
                seen.append(tuple(g.allowed(slot, out)))
                out.append(toks[slot])
            g.on_brick(b.h, b.w)
        return seen, g.counters()

    def test_the_same_bricks_give_the_same_candidate_lists(self):
        bricks = [Brick(2, 2, 0, 0, 0), Brick(2, 2, 5, 5, 0),
                  Brick(2, 2, 0, 0, 1)]
        a_seen, a_counts = self.sequence(bricks)
        b_seen, b_counts = self.sequence(bricks)
        assert a_seen == b_seen
        assert a_counts == b_counts

    def test_candidate_lists_are_ordered_not_set_valued(self):
        """Order decides which index multinomial draws; a set would not."""
        g = on_gate()
        xs = offer(g, prefix(h=1, w=1), 4)
        assert xs == sorted(xs)


class TestConnectivityIsNotDressedUpAsMore:
    """The naming rule, enforced rather than promised.

    The banned words are assembled from fragments so that this file scanning
    itself does not count as a violation of itself -- the literals below never
    appear whole in the source text.
    """

    BANNED = ("sup" "port", "stabil" "ity", "sta" "ble",
              "phys" "ics", "gra" "vity")

    def module_text(self) -> str:
        source = Path(__file__).resolve().parents[1] / "src" / "constraints" \
            / "placement_decode.py"
        return source.read_text(encoding="utf-8")

    def test_no_public_name_in_the_module_claims_more_than_connectivity(self):
        import src.constraints.placement_decode as mod

        for name in dir(mod):
            if name.startswith("_"):
                continue
            low = name.lower()
            for banned in self.BANNED:
                assert banned not in low, f"public name {name} claims {banned}"

    def test_no_class_or_function_name_here_claims_it_either(self):
        text = Path(__file__).read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("def test_", "class Test", "def ")):
                low = stripped.split("(")[0].lower()
                for banned in self.BANNED:
                    assert banned not in low, line

    def test_the_module_states_the_disclaimer_in_its_contract(self):
        text = " ".join(self.module_text().split())
        assert "does not enforce" in text
        assert "does not model" in text
        assert "different question" in text

    def test_connectivity_is_the_checkers_predicate_not_a_new_one(self):
        """Same function, same ground=False, no second definition."""
        a, b = Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 0, 1)
        assert studs_connected(a, b)
        assert is_connected([a, b], ground=False)
        assert "is_connected(placed, ground=False)" in self.module_text()


class TestNoPhase2Data:
    """Fixtures only. The banned paths are assembled from fragments so this
    file does not trip its own scan."""

    BANNED = ("instruct" "_inv_test", "instruct" "_inv_train",
              "core_eval" "_plan", "results" ".jsonl", "scores" ".json",
              "data/" "processed", "runs/" "core_eval")

    def test_this_file_reads_no_corpus_or_plan(self):
        text = Path(__file__).read_text(encoding="utf-8")
        for banned in self.BANNED:
            assert banned not in text, banned

    def test_the_module_reads_no_corpus_or_plan(self):
        source = Path(__file__).resolve().parents[1] / "src" / "constraints" \
            / "placement_decode.py"
        text = source.read_text(encoding="utf-8")
        for banned in self.BANNED:
            assert banned not in text, banned
        for reader in ("open(", "read_text", "read_bytes", "json.load"):
            assert reader not in text, reader


class TestNoClaimIsMadeAboutCoreSuccess:
    def test_the_module_disclaims_the_causal_reading(self):
        source = Path(__file__).resolve().parents[1] / "src" / "constraints" \
            / "placement_decode.py"
        flat = " ".join(source.read_text(encoding="utf-8").split())
        assert "did no causal decomposition" in flat
        assert "nothing here promises that gating them" in flat
        assert "demonstrated cause" in flat


# --------------------------------------------------------------------------
# Regressions. Each class below fails on the version of the module that
# shipped before this pass; the comment on each says which rule it broke.
# --------------------------------------------------------------------------


class TestSlotZeroOffersOnlyHeightsTheStockCanFinish:
    """The old rule asked "does h fit at w = 1", and w = 1 may be unpayable.

    With ``2x4`` the only part in stock and no room for anything four wide,
    slot 0 used to offer ``h = 2`` because a 2x1 would have fitted -- and
    then slot 2 had nothing in stock that fitted and raised.
    """

    def stock_gate(self, parts, world, prefill=()):
        g = InventoryPlacementGate(stub_slots(), Inventory.from_parts(parts),
                                   enabled=True, world=world)
        for b in prefill:
            g.rules.occupancy.add(b)
        return g

    def test_the_space_takes_a_2x1_but_the_stock_will_not_sell_one(self):
        """The premise, stated as an assertion rather than trusted."""
        free = Occupancy(2)
        assert free.any_placement(2, 1)
        assert not free.any_placement(2, 4)
        assert canonical_part(2, 4) == "2x4"

    def test_a_height_whose_only_payable_width_does_not_fit_is_withheld(self):
        g = self.stock_gate({"2x4": 1}, world=2)
        offered = g.allowed(0, [])
        assert dim(2) not in offered, "h=2 can only be spelled 2x4 here"
        assert dim(4) not in offered
        assert offered == [stub_slots().eos]
        assert g.stop_reason == "space_exhausted"

    def test_slot_2_is_never_handed_a_height_it_cannot_complete(self):
        g = self.stock_gate({"2x4": 1}, world=2)
        for th in g.allowed(0, []):
            if th == stub_slots().eos:
                continue
            h = th - DIM_BASE + 1
            assert g.allowed(2, prefix(h=h)), f"slot 2 empty for h={h}"

    def test_a_partly_filled_world_drops_the_height_that_stopped_fitting(self):
        """Every 2x2 square is blocked; a 2x1 still is not."""
        prefill = [Brick(1, 1, 1, 1, z) for z in range(3)]
        g = self.stock_gate({"2x2": 4}, world=3, prefill=prefill)
        assert g.rules.occupancy.any_placement(2, 1)
        assert not g.rules.occupancy.any_placement(2, 2)
        assert g.allowed(0, []) == [stub_slots().eos]
        assert g.stop_reason == "space_exhausted"

    def test_the_same_stock_is_offered_again_once_the_width_fits(self):
        g = self.stock_gate({"2x4": 1}, world=8)
        offered = g.allowed(0, [])
        assert dim(2) in offered and dim(4) in offered
        assert g.stop_reason == "running"

    def test_the_placement_only_gate_still_reaches_every_width(self):
        """No stock, so w = 1 really is spellable and h = 2 survives."""
        g = on_gate(world=3)
        for z in range(3):
            g.rules.occupancy.add(Brick(1, 1, 1, 1, z))
        assert dim(2) in g.allowed(0, [])
        assert g.stop_reason == "running"

    def test_stock_still_wins_when_placement_would_have_allowed_more(self):
        g = self.stock_gate({"1x1": 2}, world=4)
        offered = set(g.allowed(0, []))
        assert offered == {dim(1), stub_slots().eos}
        assert dim(2) not in offered


class TestTheDeferralBudgetEndsTheModelForGood:
    """The old rule wrote ``connectivity_unmet`` and kept offering bricks.

    The decode loop keeps the first reason a gate writes, so the model could
    build on, join its pieces and finish connected under a reason saying it
    had not.
    """

    def two_pieces(self, **kw):
        g = on_gate(connectivity="final_eos", **kw)
        drive(g, [Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 3, 0)])
        assert not is_connected(g.rules.occupancy.bricks, ground=False)
        return g

    def test_no_brick_candidate_survives_once_the_budget_is_spent(self):
        g = self.two_pieces(max_eos_deferrals=1)
        assert stub_slots().eos not in offer(g, [], 0)
        offered = offer(g, [], 0)
        assert offered == [stub_slots().eos]
        assert [t for t in offered if t in stub_slots().dims] == []
        assert g.stop_reason == "connectivity_unmet"

    def test_the_beam_that_would_rescue_it_is_masked_out(self):
        g = self.two_pieces(max_eos_deferrals=1)
        offer(g, [], 0)
        assert dim(1) not in offer(g, [], 0)

    def test_the_reason_is_written_only_where_it_is_true(self):
        for budget in range(0, 4):
            g = self.two_pieces(max_eos_deferrals=budget)
            last = None
            for _ in range(budget + 2):
                last = tuple(g.allowed(0, []))
            if g.stop_reason == "connectivity_unmet":
                assert last == (stub_slots().eos,), budget
                assert not is_connected(g.rules.occupancy.bricks,
                                        ground=False), budget

    def test_a_model_that_joins_up_in_time_keeps_a_clean_reason(self):
        g = self.two_pieces(max_eos_deferrals=3)
        assert stub_slots().eos not in offer(g, [], 0)   # withheld once
        drive(g, [Brick(1, 4, 0, 0, 1)])                 # and again at its slot 0
        assert is_connected(g.rules.occupancy.bricks, ground=False)
        assert stub_slots().eos in offer(g, [], 0)
        assert g.stop_reason == "running"
        assert g.rules.eos_deferrals == 2

    def test_the_counter_counts_withheld_offers_not_stop_attempts(self):
        g = self.two_pieces(max_eos_deferrals=2)
        offer(g, [], 0)
        offer(g, [], 0)
        assert g.counters()["eos_deferrals"] == 2
        offer(g, [], 0)
        assert g.counters()["eos_deferrals"] == 2, "the give-up is not a deferral"

    def test_the_contract_defines_the_counter_as_withheld_offers(self):
        source = Path(__file__).resolve().parents[1] / "src" / "constraints" \
            / "placement_decode.py"
        flat = " ".join(source.read_text(encoding="utf-8").split())
        assert "withheld from the candidate list" in flat
        assert "not a count of times the model asked to stop" in flat
        assert "EOS and nothing else" in flat


class TestEverySlotZeroCandidateHasACompletePath:
    """Small worlds and every stock subset, walked to the last coordinate.

    Contract clause 6 says each slot offers only values that can still be
    completed. This is that clause, checked by enumeration rather than by
    argument: every h that slot 0 offers, every w slot 2 offers for it, every
    coordinate after that, down to a brick that fits, does not collide and
    is one the stock can pay for.
    """

    POOL = ("1x1", "1x2", "2x2", "2x4")
    WORLDS = (2, 3)

    def prefills(self, world):
        every = [Brick(1, 1, x, y, z)
                 for z in range(world)
                 for x in range(world)
                 for y in range(world)]
        return ([], [Brick(1, 1, 0, 0, 0)], every[:world], every)

    def combos(self):
        for world in self.WORLDS:
            for size in range(1, len(self.POOL) + 1):
                for subset in itertools.combinations(self.POOL, size):
                    for prefill in self.prefills(world):
                        yield world, {part: 2 for part in subset}, prefill

    def walk(self, gate, world, parts):
        placed = gate.rules.occupancy.bricks
        for th in gate.allowed(0, []):
            if th == stub_slots().eos:
                continue
            h = th - DIM_BASE + 1
            widths = gate.allowed(2, prefix(h=h))
            assert widths, f"slot 0 offered h={h} and slot 2 had nothing"
            for tw in widths:
                w = tw - DIM_BASE + 1
                assert canonical_part(h, w) in parts
                xs = gate.allowed(4, prefix(h=h, w=w))
                assert xs, f"no x completes {h}x{w}"
                for tx in xs:
                    x = tx - POS_BASE
                    ys = gate.allowed(6, prefix(h=h, w=w, x=x))
                    assert ys, f"no y completes {h}x{w} at x={x}"
                    for ty in ys:
                        y = ty - POS_BASE
                        zs = gate.allowed(8, prefix(h=h, w=w, x=x, y=y))
                        assert zs, f"no z completes {h}x{w} at ({x},{y})"
                        for tz in zs:
                            z = tz - POS_BASE
                            brick = Brick(h=h, w=w, x=x, y=y, z=z)
                            assert brick.in_bounds(world)
                            same = [o for o in placed if o.z == z]
                            assert find_collisions(same + [brick]) == []

    def test_every_offered_height_finishes_a_legal_payable_brick(self):
        checked = 0
        for world, parts, prefill in self.combos():
            gate = InventoryPlacementGate(
                stub_slots(), Inventory.from_parts(parts),
                enabled=True, world=world)
            for brick in prefill:
                gate.rules.occupancy.add(brick)
            self.walk(gate, world, parts)
            checked += 1
        assert checked == len(self.WORLDS) * 15 * 4

    def test_a_full_world_offers_nothing_but_eos_whatever_the_stock(self):
        for world, parts, _ in self.combos():
            gate = InventoryPlacementGate(
                stub_slots(), Inventory.from_parts(parts),
                enabled=True, world=world)
            for brick in [Brick(1, 1, x, y, z)
                          for z in range(world)
                          for x in range(world)
                          for y in range(world)]:
                gate.rules.occupancy.add(brick)
            assert gate.allowed(0, []) == [stub_slots().eos]
            assert gate.stop_reason == "space_exhausted"


# --------------------------------------------------------------------------
# The entry point, driven through the real decode loop.
#
# ``PlannedGPT.generate_raw`` is :meth:`BrickGPT.generate_raw` itself, called
# on a stand-in object: the loop, the slot arithmetic, the brick accounting
# and the termination all come from the module under test. Only the weights
# and the tokenizer are replaced, so these are integration tests and still
# need no checkpoint, no tokenizer and no network.
# --------------------------------------------------------------------------


class StubTokenizer:
    """Turns the stub ids back into brick text the project's parser can read."""

    def __init__(self, slots: Slots):
        self.slots = slots
        self.literal = {slots.literal_x: "x", slots.literal_open: " (",
                        slots.literal_comma: ",", slots.literal_close: ")\n"}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        out = []
        for tid in ids:
            if tid == self.slots.eos:
                if not skip_special_tokens:
                    out.append("<eos>")
            elif tid in self.literal:
                out.append(self.literal[tid])
            elif DIM_BASE <= tid < DIM_BASE + len(self.slots.dims):
                out.append(str(tid - DIM_BASE + 1))
            elif POS_BASE <= tid < POS_BASE + len(self.slots.posns):
                out.append(str(tid - POS_BASE))
            else:
                raise AssertionError(f"{tid} is not in the stub vocabulary")
        return "".join(out)


class PlannedGPT:
    """Deterministic logits: the planned token if the gate allows it.

    Every token gets a logit ten thousand below the one before it, so after
    the softmax the planned token has probability exactly 1 when it survives
    the mask, and the lowest-numbered surviving token has probability exactly
    1 when it does not. No seed, no tolerance, no flaky draw.
    """

    VOCAB = POS_BASE + WORLD + 1

    def __init__(self, slots: Slots, plan: list[int]):
        self.slots = slots
        self.device = "cpu"
        self.tokenizer = StubTokenizer(slots)
        self.plan = list(plan)
        self.step = 0
        self.prompts: list[tuple[str, dict | None]] = []
        self.model = self._logits

    def encode(self, caption: str, inventory):
        self.prompts.append(
            (caption, None if inventory is None else dict(inventory)))
        return torch.tensor([[7, 8, 9]])

    def _logits(self, *, input_ids, past_key_values, use_cache):
        row = torch.arange(self.VOCAB, dtype=torch.float32) * -1e4 - 1e4
        if self.step < len(self.plan):
            row[self.plan[self.step]] = 0.0
        self.step += 1
        return SimpleNamespace(logits=row.view(1, 1, -1),
                               past_key_values=(self.step,))

    def generate_raw(self, *args, **kw):
        return BrickGPT.generate_raw(self, *args, **kw)


def plan_for(bricks, *, then_eos: bool = True) -> list[int]:
    toks: list[int] = []
    for b in bricks:
        toks += spell(b.h, b.w, b.x, b.y, b.z)
    if then_eos:
        toks.append(stub_slots().eos)
    return toks


def placed_of(gate) -> list[tuple[int, int, int, int, int]]:
    return [(b.h, b.w, b.x, b.y, b.z) for b in gate.rules.occupancy.bricks]


class TestTheEntryPointOnTheRealLoop:
    def test_a_planned_tower_decodes_parses_and_stops_normally(self):
        bricks = [Brick(2, 4, 0, 0, z) for z in range(3)]
        gpt = PlannedGPT(stub_slots(), plan_for(bricks))
        raw, gate = generate_raw_with_placement(
            gpt, "a tower", enabled=True, max_bricks=8)
        assert raw.text == "2x4 (0,0,0)\n2x4 (0,0,1)\n2x4 (0,0,2)\n"
        assert raw.termination == "normal_eos"
        assert raw.n_tokens == 31 and raw.seconds >= 0 and not raw.truncated
        parsed, unparsed = parse_output(raw.text)
        assert unparsed == [] and len(parsed) == 3
        assert placed_of(gate) == [(2, 4, 0, 0, z) for z in range(3)]
        assert isinstance(gate, PlacementGate)
        assert gpt.prompts == [("a tower", None)]

    def test_a_plan_that_repeats_a_brick_cannot_collide(self):
        b = Brick(2, 4, 0, 0, 0)
        gpt = PlannedGPT(stub_slots(), plan_for([b, b]))
        raw, gate = generate_raw_with_placement(
            gpt, "twice", enabled=True, max_bricks=4)
        assert raw.termination == "normal_eos"
        assert placed_of(gate) == [(2, 4, 0, 0, 0), (2, 4, 0, 0, 1)]
        assert find_collisions(gate.rules.occupancy.bricks) == []
        assert gate.counters()["candidates_masked"][8] >= 1

    def test_disabled_lets_the_repeat_through_exactly_as_before(self):
        b = Brick(2, 4, 0, 0, 0)
        gpt = PlannedGPT(stub_slots(), plan_for([b, b]))
        raw, gate = generate_raw_with_placement(
            gpt, "twice", enabled=False, max_bricks=4)
        assert raw.text == "2x4 (0,0,0)\n2x4 (0,0,0)\n"
        parsed, _ = parse_output(raw.text)
        assert find_collisions(parsed) != [], "legacy behaviour, warts included"
        counters = gate.counters()
        assert counters["candidates_masked_total"] == 0
        assert counters["bricks_placed"] == 0
        assert counters["enabled"] is False

    def test_the_token_budget_still_ends_the_run(self):
        bricks = [Brick(1, 1, 0, 0, z) for z in range(6)]
        gpt = PlannedGPT(stub_slots(), plan_for(bricks))
        raw, gate = generate_raw_with_placement(
            gpt, "tall", enabled=True, max_bricks=2)
        assert raw.termination == "max_bricks"
        assert raw.truncated and raw.n_tokens == 20
        assert gate.counters()["bricks_placed"] == 2

    def test_the_unimplemented_counters_stay_null_after_a_real_decode(self):
        gpt = PlannedGPT(stub_slots(), plan_for([Brick(1, 1, 0, 0, 0)]))
        _, gate = generate_raw_with_placement(gpt, "one", enabled=True)
        counters = gate.counters()
        for name in UNIMPLEMENTED_COUNTERS:
            assert counters[name] == {"value": None, "implemented": False}, name
        assert counters["max_eos_deferrals"] == MAX_EOS_DEFERRALS

    # -- with stock -----------------------------------------------------

    def test_the_prompt_and_the_counter_see_the_same_opening_stock(self):
        parts = {"2x4": 2}
        bricks = [Brick(2, 4, 0, 0, z) for z in range(3)]
        gpt = PlannedGPT(stub_slots(), plan_for(bricks))
        raw, gate = generate_raw_with_placement(
            gpt, "a wall", inventory=Inventory.from_parts(parts),
            enabled=True, max_bricks=8)
        caption, shown = gpt.prompts[0]
        assert caption == "a wall"
        assert shown == parts, "the model was told the opening position"
        assert gate.opening_inventory == parts, "and the gate enforced it"
        assert isinstance(gate, InventoryPlacementGate)
        assert raw.termination == "inventory_exhausted"
        assert placed_of(gate) == [(2, 4, 0, 0, 0), (2, 4, 0, 0, 1)]
        assert Counter(gate.accepted) == Counter({"2x4": 2})
        assert gate.inventory.as_dict() == {}
        assert required_inventory(gate.rules.occupancy.bricks) \
            <= Counter(parts)

    def test_the_combined_gate_on_the_loop_never_overspends(self):
        parts = {"2x2": 3, "1x1": 2}
        plan = plan_for([Brick(2, 2, 0, 0, 0)] * 6)
        gpt = PlannedGPT(stub_slots(), plan)
        raw, gate = generate_raw_with_placement(
            gpt, "blocks", inventory=Inventory.from_parts(parts),
            enabled=True, max_bricks=10)
        used = required_inventory(gate.rules.occupancy.bricks)
        assert used <= Counter(parts), used
        assert used["2x2"] == 3
        assert find_collisions(gate.rules.occupancy.bricks) == []
        assert raw.termination in PlacementGate.STOP_REASONS

    def test_a_stock_only_world_with_no_room_stops_without_a_brick(self):
        """Slot 0 has nothing payable that fits, so the loop stops at once."""
        gpt = PlannedGPT(stub_slots(), plan_for([Brick(2, 4, 0, 0, 0)]))
        raw, gate = generate_raw_with_placement(
            gpt, "no room", inventory=Inventory.from_parts({"2x4": 1}),
            enabled=True, world=2, max_bricks=4)
        assert raw.termination == "space_exhausted"
        assert raw.text == ""
        assert placed_of(gate) == []
        assert gate.inventory.as_dict() == {"2x4": 1}, "nothing was spent"

    # -- connectivity ---------------------------------------------------

    def test_the_reason_never_outlives_the_pieces_it_names(self):
        """The old gate wrote the reason, then let the beam in anyway.

        Plan: two pillars, then a beam that joins them. The budget is spent
        the moment the second pillar lands, so the beam must be unreachable
        and the model must stop while it really is in pieces.
        """
        pillars = [Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 3, 0)]
        beam = Brick(1, 4, 0, 0, 1)
        plan = plan_for(pillars, then_eos=False) + plan_for([beam])
        gpt = PlannedGPT(stub_slots(), plan)
        raw, gate = generate_raw_with_placement(
            gpt, "two pillars", enabled=True, connectivity="final_eos",
            max_eos_deferrals=0, max_bricks=12)
        assert raw.termination == "connectivity_unmet"
        assert placed_of(gate) == [(1, 1, 0, 0, 0), (1, 1, 0, 3, 0)]
        assert not is_connected(gate.rules.occupancy.bricks, ground=False)
        assert gate.counters()["eos_deferrals"] == 0

    def test_a_budget_that_still_allows_the_beam_ends_connected_and_clean(self):
        pillars = [Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 3, 0)]
        beam = Brick(1, 4, 0, 0, 1)
        plan = plan_for(pillars, then_eos=False) + plan_for([beam])
        gpt = PlannedGPT(stub_slots(), plan)
        raw, gate = generate_raw_with_placement(
            gpt, "two pillars", enabled=True, connectivity="final_eos",
            max_eos_deferrals=3, max_bricks=12)
        assert raw.termination == "normal_eos"
        assert is_connected(gate.rules.occupancy.bricks, ground=False)
        assert placed_of(gate) == [(1, 1, 0, 0, 0), (1, 1, 0, 3, 0),
                                   (1, 4, 0, 0, 1)]
        assert gate.counters()["eos_deferrals"] == 1

    def test_connectivity_off_is_the_default_and_stops_in_pieces(self):
        pillars = [Brick(1, 1, 0, 0, 0), Brick(1, 1, 0, 3, 0)]
        gpt = PlannedGPT(stub_slots(), plan_for(pillars))
        raw, gate = generate_raw_with_placement(
            gpt, "two pillars", enabled=True, max_bricks=12)
        assert raw.termination == "normal_eos"
        assert not is_connected(gate.rules.occupancy.bricks, ground=False)
        assert gate.counters()["eos_deferrals"] == 0
