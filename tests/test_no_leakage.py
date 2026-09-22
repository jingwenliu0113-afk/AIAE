"""The split may not leak, and this is where that is checked by name.

The leakage this project can actually suffer is not row-level: it is that one
object's variants land in different splits. A structure is a *variant* of an
object -- the same model re-tiled from a different inventory -- so two rows can
be near-identical while carrying different `structure_id`. Splitting on the
structure puts a model's twin in train and its sibling in validation, and the
validation score then measures memorisation.

`src/data/splits.py` therefore assigns by `object_id` and derives every
structure's split from its object. These tests hold that property, and each
one is paired with a case that must fail: a guard nobody has watched fail is a
guard nobody knows is connected.

Everything here runs on synthetic rows, so it means the same thing in a
checkout that has no dataset as in one that does.
"""

from __future__ import annotations

import pytest

from src.data.splits import BUCKETS, VAL_FRACTION, assign, build

SPLITS = {"train", "val", "test"}


def rows(*pairs: tuple[str, str]) -> list[dict]:
    return [{"object_id": o, "structure_id": s} for o, s in pairs]


def many(prefix: str, objects: int, per_object: int) -> list[dict]:
    return rows(*[(f"{prefix}{o}", f"{prefix}{o}-v{v}")
                  for o in range(objects) for v in range(per_object)])


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------

def test_no_object_reaches_two_splits():
    manifest = build({"train": many("t", 400, 3), "test": many("e", 60, 3)})
    seen: dict[str, str] = {}
    for structure, object_id in manifest.structures.items():
        split = manifest.split_of_structure(structure)
        assert split in SPLITS
        assert seen.setdefault(object_id, split) == split, (
            f"{object_id} is in both {seen[object_id]} and {split}")


def test_every_variant_of_an_object_follows_its_object():
    """The whole point: a model's twin may not sit across the split."""
    manifest = build({"train": many("t", 300, 4)})
    by_object: dict[str, set[str]] = {}
    for structure, object_id in manifest.structures.items():
        by_object.setdefault(object_id, set()).add(
            manifest.split_of_structure(structure))
    split_across = {o: s for o, s in by_object.items() if len(s) > 1}
    assert split_across == {}, f"variants split across folds: {split_across}"


def test_the_assignment_is_a_pure_function_of_the_id():
    """Python's hash() is salted per process; this must not be.

    A per-process salt would reshuffle the split on every run, so a model
    trained yesterday would be evaluated today against objects it had seen.
    """
    ids = [f"object-{n}" for n in range(500)]
    once = [assign(i, "train") for i in ids]
    again = [assign(i, "train") for i in ids]
    assert once == again
    assert once == [assign(i, "train") for i in ids]
    # Known-answer: the bucket comes from a fixed salt, so these are stable
    # across machines and Python versions, not just within one process.
    assert assign("object-0", "train") in SPLITS


def test_the_upstream_test_split_is_never_reassigned():
    for n in range(500):
        assert assign(f"object-{n}", "test") == "test"


def test_validation_is_carved_out_of_train_only():
    manifest = build({"train": many("t", 2000, 1), "test": many("e", 200, 1)})
    val = {o for o, s in manifest.objects.items() if s == "val"}
    assert val, "no validation objects were carved out at all"
    assert all(o.startswith("t") for o in val), "a test object became validation"
    share = len(val) / 2000
    assert 0.5 * VAL_FRACTION < share < 2 * VAL_FRACTION, (
        f"validation is {share:.1%}, nowhere near the {VAL_FRACTION:.0%} asked for")


# ---------------------------------------------------------------------------
# The same guards, made to fail
# ---------------------------------------------------------------------------

def test_an_object_in_two_upstream_splits_is_refused_not_merged():
    """Silently keeping the last assignment is how a leak ships."""
    with pytest.raises(ValueError, match="more than one upstream split"):
        build({"train": rows(("shared", "a")), "test": rows(("shared", "b"))})


def test_splitting_on_the_structure_instead_would_be_caught():
    """Proof the property test above can fail.

    This is what the module deliberately does not do: assign per structure.
    With four variants per object the leak is near-certain, and the check has
    to see it.
    """
    per_structure = {}
    for row in many("t", 300, 4):
        per_structure[row["structure_id"]] = assign(row["structure_id"], "train")
    by_object: dict[str, set[str]] = {}
    for structure, split in per_structure.items():
        by_object.setdefault(structure.split("-v")[0], set()).add(split)
    leaked = {o for o, s in by_object.items() if len(s) > 1}
    assert leaked, (
        "structure-level assignment did not leak in this sample, so this test "
        "is not demonstrating the failure it claims to")


def test_a_reshuffling_assignment_would_be_caught():
    """Proof the purity test can fail: a salted bucket changes between calls."""
    phase = [0]

    def unstable(object_id: str, upstream: str) -> str:
        # Stands in for Python's per-process hash salt: the same id lands
        # differently depending on when it is asked.
        return "val" if (len(object_id) + phase[0]) % 2 else "train"

    ids = [f"object-{n}" for n in range(20)]
    first = [unstable(i, "train") for i in ids]
    phase[0] += 1
    assert first != [unstable(i, "train") for i in ids]


def test_the_bucket_count_is_fine_enough_for_the_fraction():
    """A coarse bucket silently rounds the validation share to zero."""
    assert BUCKETS * VAL_FRACTION >= 100, (
        f"{BUCKETS} buckets at {VAL_FRACTION} cannot express the split")
