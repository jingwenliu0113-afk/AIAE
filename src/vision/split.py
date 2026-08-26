"""The vision split, frozen before anything is fitted.

This is a *new* split for a *new* test.  It has nothing to do with the 160
frozen Phase 2 cases, shares no data with them, and no number computed here
may be put beside one of theirs.

The rule the whole module exists to enforce: **a split boundary is drawn
between capture groups, never between individual photographs.**  Two frames of
one burst, two crops of one physical brick, two renders of one object differ by
less than the thing being measured.  Splitting per image puts near-duplicates
on both sides, and the test score that comes back is partly a memory score.
It looks like a good result and it is not one, which is the worst kind.

So every record has to arrive with a group, the split is a partition of
*groups*, and a record with no group is refused rather than assigned:

* if the provenance to build groups is not in the data, this module raises and
  the caller has to say so out loud;
* random per-image splitting is not offered as a fallback, because a fallback
  that hides the problem is the problem.

The assignment is deterministic -- a stable digest of the group id under a
fixed salt -- so the same records always produce the same split, on any
machine, without storing a random seed's meaning anywhere.

Test is opened once.  :meth:`VisionSplit.freeze` writes the manifest with its
own digest; the evaluation checks that digest before it reads a single test
image, so a split silently rebuilt after seeing a result cannot be passed off
as the one that was frozen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
SPLITS = (TRAIN, VALIDATION, TEST)

#: Group-level proportions.  Validation is what tuning may look at; test is
#: opened once, after everything is frozen.
DEFAULT_WEIGHTS = {TRAIN: 0.70, VALIDATION: 0.15, TEST: 0.15}

#: Fixed, and part of the split's identity.  Changing it produces a different
#: split, which is why it is a constant here and not an argument with a
#: default: a caller who could pass a salt could search for a good one.
SALT = "brickagain-vision-split-v1"


class SplitError(ValueError):
    """The records cannot be split without leaking, so nothing was split."""


@dataclass(frozen=True)
class SplitRecord:
    """One item to be split, with the provenance that decides where it goes.

    ``group`` is the capture group: a photographic session, a burst, a source
    object, a rendered instance.  ``stratum`` is what the split is balanced
    over, so that every class reaches every side; for a dataset whose groups
    span classes it is a single constant and the balancing is over nothing.
    """

    item_id: str
    group: str
    stratum: str
    label: str | None = None

    def __post_init__(self) -> None:
        for name in ("item_id", "group", "stratum"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SplitError(
                    f"{name} must be a non-empty string; a record without a "
                    "group cannot be split without risking a leak, and this "
                    "module will not guess one")


def _digest(value: str) -> str:
    return hashlib.sha256(f"{SALT}\0{value}".encode("utf-8")).hexdigest()


def _check_weights(weights: dict[str, float]) -> dict[str, float]:
    if set(weights) != set(SPLITS):
        raise SplitError(f"weights must name exactly {list(SPLITS)}")
    for name, value in weights.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise SplitError(f"weight for {name} is not a number")
        if not 0 < float(value) < 1:
            raise SplitError(
                f"weight for {name} is {value}; every split must get a "
                "positive share smaller than the whole")
    total = sum(float(v) for v in weights.values())
    if abs(total - 1.0) > 1e-9:
        raise SplitError(f"weights sum to {total}, not 1")
    return {name: float(weights[name]) for name in SPLITS}


def assign_groups(records, *, weights: dict[str, float] | None = None
                  ) -> dict[str, str]:
    """Partition the groups in ``records`` into the three splits.

    Within each stratum the groups are ordered by their salted digest and
    handed out to whichever split is furthest behind its target share of that
    stratum's *items*.  Weighting by items rather than by group count matters
    when groups differ in size, which they do: one photographic session can
    hold thirty frames and another three.

    A stratum with fewer groups than splits is refused.  Silently giving that
    stratum no test items would produce a per-class recall of zero that looks
    like a model failure.
    """
    shares = _check_weights(weights or DEFAULT_WEIGHTS)
    sizes: dict[str, int] = {}
    strata: dict[str, str] = {}
    for record in records:
        sizes[record.group] = sizes.get(record.group, 0) + 1
        seen = strata.setdefault(record.group, record.stratum)
        if seen != record.stratum:
            raise SplitError(
                f"group {record.group!r} appears in strata {seen!r} and "
                f"{record.stratum!r}; a group must belong to one stratum or "
                "balancing it means nothing")
    if not sizes:
        raise SplitError("there are no records to split")

    by_stratum: dict[str, list[str]] = {}
    for group, stratum in strata.items():
        by_stratum.setdefault(stratum, []).append(group)

    out: dict[str, str] = {}
    for stratum, groups in sorted(by_stratum.items()):
        if len(groups) < len(SPLITS):
            raise SplitError(
                f"stratum {stratum!r} has {len(groups)} capture group(s) and "
                f"{len(SPLITS)} splits to fill. Splitting it would either "
                "leave a split empty or cut a group in half; neither is "
                "acceptable, so this is refused rather than worked around")
        ordered = sorted(groups, key=lambda g: (_digest(g), g))
        total = sum(sizes[g] for g in ordered)
        target = {name: shares[name] * total for name in SPLITS}
        placed = {name: 0 for name in SPLITS}
        got = {name: 0 for name in SPLITS}
        # Every split gets one group first, so none can end up empty however
        # the sizes fall.
        for name, group in zip(SPLITS, ordered):
            out[group] = name
            placed[name] += sizes[group]
            got[name] += 1
        for group in ordered[len(SPLITS):]:
            name = min(SPLITS, key=lambda s: (placed[s] - target[s], s))
            out[group] = name
            placed[name] += sizes[group]
            got[name] += 1
    return out


@dataclass(frozen=True)
class VisionSplit:
    """A frozen assignment of groups to splits, plus what it is a split of."""

    dataset: str
    weights: dict[str, float]
    groups: dict[str, str]
    items: dict[str, str] = field(repr=False)
    strata: dict[str, str] = field(repr=False)
    labels: dict[str, str | None] = field(repr=False)
    note: str = ""

    # -- construction ----------------------------------------------------
    @classmethod
    def build(cls, dataset: str, records, *,
              weights: dict[str, float] | None = None,
              note: str = "") -> "VisionSplit":
        records = list(records)
        if not records:
            raise SplitError("a split needs records")
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, SplitRecord):
                raise SplitError(
                    f"a record must be a SplitRecord, not "
                    f"{type(record).__name__}")
            if record.item_id in seen:
                raise SplitError(
                    f"item {record.item_id!r} appears twice; a duplicated "
                    "item could otherwise be counted in two splits")
            seen.add(record.item_id)
        groups = assign_groups(records, weights=weights)
        return cls(
            dataset=str(dataset),
            weights=_check_weights(weights or DEFAULT_WEIGHTS),
            groups=dict(sorted(groups.items())),
            items={r.item_id: r.group for r in sorted(
                records, key=lambda r: r.item_id)},
            strata={r.group: r.stratum for r in sorted(
                records, key=lambda r: r.group)},
            labels={r.item_id: r.label for r in sorted(
                records, key=lambda r: r.item_id)},
            note=str(note))

    # -- queries ---------------------------------------------------------
    def split_of_item(self, item_id: str) -> str:
        group = self.items.get(item_id)
        if group is None:
            raise SplitError(f"{item_id!r} is not in this split manifest")
        return self.groups[group]

    def items_in(self, split: str) -> tuple[str, ...]:
        if split not in SPLITS:
            raise SplitError(f"{split!r} is not one of {list(SPLITS)}")
        return tuple(item for item, group in self.items.items()
                     if self.groups[group] == split)

    def counts(self) -> dict[str, int]:
        out = {name: 0 for name in SPLITS}
        for group in self.items.values():
            out[self.groups[group]] += 1
        return out

    def group_counts(self) -> dict[str, int]:
        out = {name: 0 for name in SPLITS}
        for split in self.groups.values():
            out[split] += 1
        return out

    def label_counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {name: {} for name in SPLITS}
        for item, group in self.items.items():
            label = self.labels.get(item)
            if label is None:
                continue
            bucket = out[self.groups[group]]
            bucket[label] = bucket.get(label, 0) + 1
        return {name: dict(sorted(bucket.items()))
                for name, bucket in out.items()}

    # -- the check that is the point of the module -------------------------
    def check_no_leakage(self) -> None:
        """Refuse a manifest that could put one group on two sides."""
        for item, group in self.items.items():
            if group not in self.groups:
                raise SplitError(
                    f"item {item!r} names group {group!r}, which has no split")
        assigned = set(self.groups.values())
        unknown = assigned - set(SPLITS)
        if unknown:
            raise SplitError(f"unknown split name(s): {sorted(unknown)}")
        missing = set(SPLITS) - assigned
        if missing:
            raise SplitError(
                f"no group was assigned to {sorted(missing)}; an empty split "
                "is not a split")
        # A group maps to exactly one split by construction, so the real
        # question is whether any *item* is reachable from two groups. It
        # cannot be, given the dict -- but a hand-edited manifest can, and a
        # hand-edited manifest is exactly what this check is for.
        for split in SPLITS:
            others = [s for s in SPLITS if s != split]
            here = set(self.items_in(split))
            for other in others:
                overlap = here & set(self.items_in(other))
                if overlap:
                    raise SplitError(
                        f"{len(overlap)} item(s) are in both {split} and "
                        f"{other}: {sorted(overlap)[:3]}")

    # -- persistence -----------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "kind": "brickagain.vision_split",
            "dataset": self.dataset,
            "salt": SALT,
            "weights": self.weights,
            "note": self.note,
            "boundary": (
                "groups, not images: a capture group is never split across "
                "train, validation and test"),
            "counts": self.counts(),
            "group_counts": self.group_counts(),
            "label_counts": self.label_counts(),
            "groups": self.groups,
            "items": self.items,
            "strata": self.strata,
            "labels": self.labels,
        }

    def digest(self) -> str:
        """SHA-256 of the canonical serialisation of this split."""
        return hashlib.sha256(canonical_bytes(self.as_dict())).hexdigest()

    def freeze(self, path: str | Path) -> tuple[Path, str]:
        """Write the manifest and return its path and digest.

        Refuses to overwrite.  A frozen split that can be rewritten in place
        is not frozen, and the one time it matters is after somebody has seen
        a test number they did not like.
        """
        target = Path(path)
        if target.exists():
            raise SplitError(
                f"{target} already exists. A frozen split is not overwritten: "
                "if this split is genuinely being replaced, move the old file "
                "aside deliberately so both remain on the record")
        self.check_no_leakage()
        body = canonical_bytes(self.as_dict())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        return target, hashlib.sha256(body).hexdigest()

    @classmethod
    def load(cls, path: str | Path, *,
             expected_digest: str | None = None) -> "VisionSplit":
        target = Path(path)
        if not target.is_file():
            raise SplitError(f"there is no vision split manifest at {target}")
        raw = target.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if expected_digest is not None and actual != expected_digest:
            raise SplitError(
                f"the vision split manifest digest is {actual}, not the "
                f"expected {expected_digest}; the split that was frozen is "
                "not the split on disk")
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SplitError(
                f"the vision split manifest is not valid JSON: {exc}") from exc
        if not isinstance(body, dict) or body.get(
                "kind") != "brickagain.vision_split":
            raise SplitError(
                "this file does not declare itself a brickagain vision split")
        if body.get("salt") != SALT:
            raise SplitError(
                f"the manifest was built under salt {body.get('salt')!r} and "
                f"this code uses {SALT!r}; the two are different splits")
        for name in ("weights", "groups", "items", "strata", "labels"):
            if not isinstance(body.get(name), dict):
                raise SplitError(f"the manifest field {name} is not an object")
        split = cls(dataset=str(body.get("dataset", "")),
                    weights=_check_weights(body["weights"]),
                    groups=dict(body["groups"]), items=dict(body["items"]),
                    strata=dict(body["strata"]), labels=dict(body["labels"]),
                    note=str(body.get("note", "")))
        split.check_no_leakage()
        return split


def canonical_bytes(body: dict) -> bytes:
    """One serialisation, so a digest is a property of content only."""
    return json.dumps(body, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
