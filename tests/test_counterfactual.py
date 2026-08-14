import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bricks import PART_VOCAB, parse_bricks
from src.data.counterfactual import (
    LOOSE_TAU,
    VARIANTS,
    GenerationError,
    make_pair,
    read_jsonl,
    write_jsonl,
)
from src.data.bricks import is_connected
from src.data.retile import occupancy_of
from src.data.splits import SplitManifest, assign, build

# A solid 5x7x2 block. Picked because both arms come out stud-connected: with
# layers solved independently, identical layers get identical tilings and stack
# into columns that never bridge, so most simple blocks fail the gate.
SOURCE = {
    "structure_id": "s1",
    "object_id": "o1",
    "captions": ["A short caption.", "A longer, more detailed caption."],
    "bricks": "\n".join(
        f"1x1 ({x},{y},{z})"
        for z in range(2)
        for x in range(5)
        for y in range(7)
    ),
}


@pytest.fixture(scope="module")
def pair():
    return make_pair(SOURCE, "train", seed=0)


class TestPairing:
    def test_emits_both_roles_in_all_variants(self, pair):
        assert len(pair) == 2 * len(VARIANTS) == 8
        roles = Counter(s.role for s in pair)
        assert roles == {"control": 4, "counterfactual": 4}
        assert {s.variant for s in pair} == set(VARIANTS)

    def test_pair_id_is_shared(self, pair):
        assert len({s.pair_id for s in pair}) == 1

    def test_sample_ids_unique(self, pair):
        assert len({s.sample_id for s in pair}) == len(pair)

    def test_caption_identical_across_the_pair(self, pair):
        """Same request, different stock -- the caption must not vary."""
        assert len({s.caption for s in pair}) == 1

    def test_both_roles_come_from_the_solver(self, pair):
        """Style is held fixed so stock is the only difference."""
        assert all(s.solver_status in ("OPTIMAL", "FEASIBLE") for s in pair)

    def test_all_checks_pass(self, pair):
        for s in pair:
            assert all(s.checks.values()), (s.sample_id, s.checks)


class TestCounterfactualSignal:
    def test_dropped_part_recorded_only_on_counterfactual(self, pair):
        for s in pair:
            if s.role == "counterfactual":
                assert s.dropped_part in PART_VOCAB
            else:
                assert s.dropped_part is None

    def test_never_drops_1x1(self, pair):
        assert all(s.dropped_part != "1x1" for s in pair)

    def test_target_avoids_the_dropped_part(self, pair):
        for s in pair:
            if s.role == "counterfactual":
                assert s.dropped_part not in s.used

    def test_dropped_part_absent_from_every_inventory_variant(self, pair):
        """Including the distractor variant, which otherwise adds unused parts."""
        for s in pair:
            if s.role == "counterfactual":
                assert s.dropped_part not in s.inventory, s.variant

    def test_the_two_targets_actually_differ(self, pair):
        control = next(s for s in pair if s.role == "control")
        cf = next(s for s in pair if s.role == "counterfactual")
        assert control.used != cf.used


class TestVoxelConsistency:
    def test_both_targets_reproduce_the_source_shape(self, pair):
        occ = occupancy_of(parse_bricks(SOURCE["bricks"]))
        for s in pair:
            assert occupancy_of(s.bricks) == occ

    def test_cell_count_matches_source(self, pair):
        occ = occupancy_of(parse_bricks(SOURCE["bricks"]))
        assert all(s.n_cells == len(occ) for s in pair)


class TestInventoryLegality:
    def test_target_fits_inventory(self, pair):
        for s in pair:
            used = Counter(b.part for b in s.bricks)
            for p, n in used.items():
                assert s.inventory.get(p, 0) >= n, (s.sample_id, p)

    def test_exact_variant_is_tight(self, pair):
        for s in pair:
            if s.variant == "exact":
                assert s.inventory == s.used

    def test_loose_variant_scales_by_tau(self, pair):
        import math

        for s in pair:
            if s.variant == "loose":
                for p, n in s.used.items():
                    assert s.inventory[p] == math.ceil(LOOSE_TAU * n)

    def test_distractor_adds_unused_parts(self, pair):
        for s in pair:
            if s.variant in ("distractor", "mixed"):
                extra = set(s.inventory) - set(s.used)
                assert extra, f"{s.variant} added nothing -- it is a relabelled exact"
                assert all(s.inventory[p] > 0 for p in extra)
                assert sorted(extra) == s.extra_parts

    def test_exact_and_loose_add_nothing(self, pair):
        for s in pair:
            if s.variant in ("exact", "loose"):
                assert set(s.inventory) == set(s.used)
                assert s.extra_parts == []

    def test_mixed_is_loose_plus_distractor(self, pair):
        import math

        for s in pair:
            if s.variant != "mixed":
                continue
            for p, n in s.used.items():
                assert s.inventory[p] == math.ceil(LOOSE_TAU * n)
            assert set(s.inventory) - set(s.used)

    def test_empty_distractor_pool_is_an_error(self):
        """A target using every permitted part cannot get a real distractor."""
        import random

        from src.data.counterfactual import _inventory_variants

        used = Counter({p: 1 for p in PART_VOCAB})
        with pytest.raises(GenerationError, match="no distractor pool"):
            _inventory_variants(used, forbidden=set(), rng=random.Random(0))


class TestConnectivityGate:
    """The gate is stud coupling alone.

    A baseplate is not a part, carries no inventory and is never written to the
    output, so it must not be what holds a model together.
    """

    def test_both_arms_are_single_component(self, pair):
        for s in pair:
            assert s.n_components == 1, s.sample_id
            assert s.checks["connected"]
            assert s.checks["touches_ground"]

    def test_gate_is_stud_only_not_ground(self, pair):
        """Ground anchoring is recorded but never substitutes for connection."""
        for s in pair:
            assert s.n_ground_components <= s.n_components
            assert "ground_anchored" not in s.checks

    def test_disconnected_control_is_rejected(self):
        """Two separate towers with nothing bridging them."""
        row = dict(
            SOURCE,
            structure_id="s_disc",
            bricks="2x2 (0,0,0)\n2x2 (0,0,1)\n2x2 (10,10,1)\n2x2 (10,10,2)",
        )
        with pytest.raises(GenerationError, match="disconnected"):
            make_pair(row, "train", seed=0)

    def test_ground_adjacency_alone_does_not_pass(self):
        """Two bricks resting side by side share no studs.

        Under the old ground-inclusive rule this shape passed; it must not.
        """
        row = dict(
            SOURCE,
            structure_id="s_ground",
            bricks="2x2 (0,0,0)\n2x2 (4,0,0)",
        )
        with pytest.raises(GenerationError, match="disconnected"):
            make_pair(row, "train", seed=0)

    def test_all_droppable_parts_are_tried(self, pair):
        """A single failed choice must not discard the whole source."""
        cf = next(s for s in pair if s.role == "counterfactual")
        assert cf.tried_parts
        assert cf.tried_parts[-1] == cf.dropped_part
        assert "1x1" not in cf.tried_parts

    def test_support_is_recorded_not_enforced(self, pair):
        """Support is a reported metric, never an acceptance condition.

        Rates and the populations they belong to live in
        scripts/08_corpus_structure_study.py.
        """
        for s in pair:
            assert isinstance(s.n_unsupported, int)

    def test_distractor_still_covers_the_target(self, pair):
        for s in pair:
            if s.variant == "distractor":
                for p, n in s.used.items():
                    assert s.inventory[p] >= n


class TestRotationNormalisation:
    def test_inventory_keys_are_canonical(self, pair):
        for s in pair:
            for p in list(s.inventory) + list(s.used):
                assert p in PART_VOCAB, p

    def test_targets_may_use_either_spelling(self, pair):
        """The text keeps orientation; only the inventory key is normalised."""
        for s in pair:
            for b in s.bricks:
                assert b.part in PART_VOCAB
                assert f"{b.h}x{b.w}" in s.bricks_txt


class TestProvenance:
    def test_inherits_source_ids_and_split(self, pair):
        for s in pair:
            assert s.object_id == "o1"
            assert s.structure_id == "s1"
            assert s.split == "train"

    def test_records_seed_and_solver_status(self, pair):
        assert all(s.seed == 0 and s.solver_status for s in pair)


class TestDeterminism:
    def test_same_seed_same_dropped_part(self):
        a = make_pair(SOURCE, "train", seed=0)
        b = make_pair(SOURCE, "train", seed=0)
        assert [s.dropped_part for s in a] == [s.dropped_part for s in b]
        assert [s.bricks_txt for s in a] == [s.bricks_txt for s in b]


class TestFailureIsLoud:
    def test_all_1x1_structure_raises(self):
        row = dict(SOURCE, bricks="1x1 (0,0,0)\n1x1 (2,0,0)", structure_id="s2")
        with pytest.raises(GenerationError):
            make_pair(row, "train", seed=0)


class TestRoundTrip:
    def test_jsonl(self, pair, tmp_path):
        p = write_jsonl(pair, tmp_path / "s.jsonl")
        back = read_jsonl(p)
        assert len(back) == len(pair)
        assert [s.sample_id for s in back] == [s.sample_id for s in pair]
        assert back[0].inventory == pair[0].inventory
        assert back[0].bricks_txt == pair[0].bricks_txt


class TestSplitLeakage:
    ROWS = {
        "train": [
            {"structure_id": f"s{i}", "object_id": f"o{i%50}"} for i in range(200)
        ],
        "test": [
            {"structure_id": f"t{i}", "object_id": f"p{i%20}"} for i in range(60)
        ],
    }

    def test_object_lands_in_exactly_one_split(self):
        m = build(self.ROWS)
        for oid in m.objects:
            splits = {
                m.split_of_structure(sid)
                for sid, o in m.structures.items()
                if o == oid
            }
            assert len(splits) == 1, oid

    def test_upstream_test_stays_test(self):
        m = build(self.ROWS)
        for r in self.ROWS["test"]:
            assert m.split_of_object(r["object_id"]) == "test"

    def test_val_only_carved_from_train(self):
        m = build(self.ROWS)
        val_objects = set(m.ids("val"))
        test_objects = {r["object_id"] for r in self.ROWS["test"]}
        assert not (val_objects & test_objects)

    def test_assignment_is_a_pure_function_of_the_id(self):
        """Stable under reordering, resampling, or adding rows later."""
        assert assign("abc", "train") == assign("abc", "train")
        shuffled = {k: list(reversed(v)) for k, v in self.ROWS.items()}
        assert build(self.ROWS).objects == build(shuffled).objects

    def test_conflicting_upstream_split_raises(self):
        rows = {
            "train": [{"structure_id": "a", "object_id": "shared"}],
            "test": [{"structure_id": "b", "object_id": "shared"}],
        }
        with pytest.raises(ValueError, match="more than one upstream"):
            build(rows)

    def test_derived_sample_inherits_split(self, pair):
        m = build(self.ROWS)
        oid = m.ids("train")[0]
        row = dict(SOURCE, object_id=oid)
        derived = make_pair(row, m.split_of_object(oid), seed=0)
        assert all(s.split == "train" for s in derived)


class TestManifestIsFrozen:
    def test_save_refuses_to_overwrite(self, tmp_path):
        m = build(TestSplitLeakage.ROWS)
        p = tmp_path / "m.json"
        m.save(p)
        with pytest.raises(FileExistsError, match="invalidate"):
            m.save(p)
        m.save(p, force=True)

    def test_round_trip(self, tmp_path):
        m = build(TestSplitLeakage.ROWS)
        p = m.save(tmp_path / "m.json")
        back = SplitManifest.load(p)
        assert back.objects == m.objects
        assert back.counts() == m.counts()
