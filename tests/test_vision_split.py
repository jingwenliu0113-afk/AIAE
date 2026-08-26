"""The vision split: groups, not images, and every leak refused.

The failure this guards against does not raise and does not look wrong. Split
per image instead of per capture group and two frames of one burst land on
opposite sides of the test boundary; the score that comes back is partly a
memory score, and it looks like a good result.

So the tests here are mostly about refusals: a record with no group, a group in
two strata, a stratum too small to divide, a manifest whose items do not match
the data, and a frozen file being overwritten.
"""

from __future__ import annotations

import json

import pytest

from src.vision.split import (DEFAULT_WEIGHTS, SALT, SPLITS, TEST, TRAIN,
                              VALIDATION, SplitError, SplitRecord, VisionSplit,
                              assign_groups)


def records(n_groups=12, per_group=3, strata=("a",), label="1x2"):
    out = []
    for stratum in strata:
        for group in range(n_groups):
            for item in range(per_group):
                out.append(SplitRecord(
                    item_id=f"{stratum}/g{group}/i{item}",
                    group=f"{stratum}/g{group}", stratum=stratum,
                    label=label))
    return out


class TestARecordNeedsProvenance:
    @pytest.mark.parametrize("field", ["item_id", "group", "stratum"])
    def test_an_empty_field_is_refused(self, field):
        kw = {"item_id": "a", "group": "g", "stratum": "s"}
        kw[field] = "   "
        with pytest.raises(SplitError, match="non-empty string"):
            SplitRecord(**kw)

    def test_the_refusal_says_why_it_will_not_guess(self):
        with pytest.raises(SplitError, match="will not guess"):
            SplitRecord(item_id="a", group="", stratum="s")


class TestTheBoundaryIsDrawnBetweenGroups:
    def test_no_group_is_split_across_two_sides(self):
        split = VisionSplit.build("x", records(n_groups=15))
        by_group = {}
        for item, group in split.items.items():
            by_group.setdefault(group, set()).add(split.groups[group])
        assert all(len(sides) == 1 for sides in by_group.values())

    def test_every_split_gets_something(self):
        split = VisionSplit.build("x", records(n_groups=9))
        counts = split.counts()
        assert all(counts[name] > 0 for name in SPLITS)
        groups = split.group_counts()
        assert all(groups[name] > 0 for name in SPLITS)

    def test_the_assignment_is_deterministic(self):
        first = VisionSplit.build("x", records(n_groups=20))
        again = VisionSplit.build("x", records(n_groups=20))
        assert first.groups == again.groups
        assert first.digest() == again.digest()

    def test_the_order_records_arrive_in_does_not_matter(self):
        rows = records(n_groups=20)
        forward = VisionSplit.build("x", rows)
        backward = VisionSplit.build("x", list(reversed(rows)))
        assert forward.groups == backward.groups

    def test_every_stratum_reaches_every_split(self):
        split = VisionSplit.build(
            "x", records(n_groups=6, strata=("1x1", "2x4", "2x6")))
        for stratum in ("1x1", "2x4", "2x6"):
            sides = {split.groups[group] for group, name
                     in split.strata.items() if name == stratum}
            assert sides == set(SPLITS), stratum

    def test_the_proportions_are_roughly_the_weights(self):
        split = VisionSplit.build("x", records(n_groups=60, per_group=1))
        counts = split.counts()
        total = sum(counts.values())
        for name, want in DEFAULT_WEIGHTS.items():
            assert abs(counts[name] / total - want) < 0.08, name


class TestAStratumTooSmallToDivideIsRefused:
    def test_two_groups_and_three_splits_is_refused(self):
        rows = [SplitRecord(item_id=f"i{i}", group=f"g{i % 2}", stratum="s")
                for i in range(6)]
        with pytest.raises(SplitError, match="capture group"):
            assign_groups(rows)

    def test_the_message_says_it_will_not_cut_a_group_in_half(self):
        rows = [SplitRecord(item_id=f"i{i}", group="g", stratum="s")
                for i in range(4)]
        with pytest.raises(SplitError, match="cut a group in half"):
            assign_groups(rows)

    def test_a_group_in_two_strata_is_refused(self):
        rows = [SplitRecord(item_id="a", group="g", stratum="one"),
                SplitRecord(item_id="b", group="g", stratum="two")]
        with pytest.raises(SplitError, match="appears in strata"):
            assign_groups(rows)


class TestWeightsAreChecked:
    @pytest.mark.parametrize("weights", [
        {TRAIN: 0.5, VALIDATION: 0.3, TEST: 0.3},
        {TRAIN: 0.7, VALIDATION: 0.3},
        {TRAIN: 1.0, VALIDATION: 0.0, TEST: 0.0},
        {TRAIN: 0.7, VALIDATION: 0.15, "holdout": 0.15},
    ])
    def test_a_bad_weight_set_is_refused(self, weights):
        with pytest.raises(SplitError):
            assign_groups(records(), weights=weights)


class TestDuplicatesAndEmptiness:
    def test_a_duplicated_item_is_refused(self):
        rows = records(n_groups=9)
        rows.append(rows[0])
        with pytest.raises(SplitError, match="appears twice"):
            VisionSplit.build("x", rows)

    def test_no_records_is_refused(self):
        with pytest.raises(SplitError, match="needs records"):
            VisionSplit.build("x", [])

    def test_a_non_record_is_refused(self):
        with pytest.raises(SplitError, match="must be a SplitRecord"):
            VisionSplit.build("x", [{"item_id": "a"}])


class TestLeakageChecking:
    def test_a_clean_split_passes(self):
        VisionSplit.build("x", records(n_groups=9)).check_no_leakage()

    def test_an_item_pointing_at_an_unknown_group_is_caught(self):
        split = VisionSplit.build("x", records(n_groups=9))
        broken = VisionSplit(
            dataset=split.dataset, weights=split.weights,
            groups=split.groups, items={**split.items, "ghost": "nowhere"},
            strata=split.strata, labels=split.labels)
        with pytest.raises(SplitError, match="which has no split"):
            broken.check_no_leakage()

    def test_an_empty_split_is_caught(self):
        split = VisionSplit.build("x", records(n_groups=9))
        collapsed = {group: TRAIN for group in split.groups}
        broken = VisionSplit(
            dataset=split.dataset, weights=split.weights, groups=collapsed,
            items=split.items, strata=split.strata, labels=split.labels)
        with pytest.raises(SplitError, match="not a split"):
            broken.check_no_leakage()

    def test_an_unknown_split_name_is_caught(self):
        split = VisionSplit.build("x", records(n_groups=9))
        groups = dict(split.groups)
        groups[next(iter(groups))] = "holdout"
        broken = VisionSplit(
            dataset=split.dataset, weights=split.weights, groups=groups,
            items=split.items, strata=split.strata, labels=split.labels)
        with pytest.raises(SplitError, match="unknown split"):
            broken.check_no_leakage()


class TestFreezingIsFreezing:
    def test_it_writes_and_reloads_identically(self, tmp_path):
        split = VisionSplit.build("x", records(n_groups=12))
        path, digest = split.freeze(tmp_path / "s.json")
        loaded = VisionSplit.load(path, expected_digest=digest)
        assert loaded.groups == split.groups
        assert loaded.items == split.items
        assert loaded.digest() == split.digest()

    def test_it_refuses_to_overwrite(self, tmp_path):
        split = VisionSplit.build("x", records(n_groups=12))
        split.freeze(tmp_path / "s.json")
        with pytest.raises(SplitError, match="already exists"):
            split.freeze(tmp_path / "s.json")

    def test_a_wrong_expected_digest_is_refused(self, tmp_path):
        split = VisionSplit.build("x", records(n_groups=12))
        path, _digest = split.freeze(tmp_path / "s.json")
        with pytest.raises(SplitError, match="not the expected"):
            VisionSplit.load(path, expected_digest="0" * 64)

    def test_a_single_edited_assignment_changes_the_digest(self, tmp_path):
        split = VisionSplit.build("x", records(n_groups=12))
        path, digest = split.freeze(tmp_path / "s.json")
        body = json.loads(path.read_text(encoding="utf-8"))
        group = next(g for g, side in body["groups"].items() if side == TRAIN)
        body["groups"][group] = TEST
        edited = tmp_path / "edited.json"
        edited.write_text(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")), encoding="utf-8")
        with pytest.raises(SplitError, match="not the expected"):
            VisionSplit.load(edited, expected_digest=digest)

    def test_a_different_salt_is_a_different_split(self, tmp_path):
        split = VisionSplit.build("x", records(n_groups=12))
        path, _digest = split.freeze(tmp_path / "s.json")
        body = json.loads(path.read_text(encoding="utf-8"))
        body["salt"] = "something-else"
        other = tmp_path / "other.json"
        other.write_text(json.dumps(body), encoding="utf-8")
        with pytest.raises(SplitError, match="different splits"):
            VisionSplit.load(other)

    def test_a_file_that_is_not_a_split_manifest_is_refused(self, tmp_path):
        target = tmp_path / "x.json"
        target.write_text(json.dumps({"kind": "something.else"}),
                          encoding="utf-8")
        with pytest.raises(SplitError, match="does not declare itself"):
            VisionSplit.load(target)

    def test_invalid_json_is_refused(self, tmp_path):
        target = tmp_path / "x.json"
        target.write_text("{ not json", encoding="utf-8")
        with pytest.raises(SplitError, match="not valid JSON"):
            VisionSplit.load(target)

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(SplitError, match="no vision split manifest"):
            VisionSplit.load(tmp_path / "absent.json")


class TestQueries:
    def test_split_of_item_and_items_in_agree(self):
        split = VisionSplit.build("x", records(n_groups=12))
        for name in SPLITS:
            for item in split.items_in(name):
                assert split.split_of_item(item) == name

    def test_an_unknown_item_is_refused(self):
        split = VisionSplit.build("x", records(n_groups=12))
        with pytest.raises(SplitError, match="not in this split"):
            split.split_of_item("nope")

    def test_an_unknown_split_name_is_refused(self):
        split = VisionSplit.build("x", records(n_groups=12))
        with pytest.raises(SplitError, match="not one of"):
            split.items_in("holdout")

    def test_label_counts_add_up(self):
        split = VisionSplit.build(
            "x", records(n_groups=9, strata=("1x1", "2x4")))
        counted = sum(sum(bucket.values())
                      for bucket in split.label_counts().values())
        assert counted == len(split.items)

    def test_the_salt_is_part_of_the_serialisation(self):
        split = VisionSplit.build("x", records(n_groups=9))
        assert split.as_dict()["salt"] == SALT
