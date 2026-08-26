"""Train-only comparison and the minimum F-pipeline baseline.

The source boundary is part of the implementation, not a comment: every
catalogue row must say ``split=train`` and the input filename must end in
``_train.jsonl``.  Its object and structure identifiers must also agree with
the frozen split manifest at its pinned digest.  A mixed, unknown or held-out
row is rejected before it can become an index.

Retrieval here is a deterministic lexical baseline.  It uses Unicode-normal
word and character features so Chinese text is tokenised without an external
service, but it is not a multilingual embedding model and must not be
called semantic retrieval.  Its purpose is to make the product path runnable
offline and to expose the retrieval/optimisation boundary honestly.

``compare_existing`` retrieves by caption, then re-ranks the retrieved set by
the exact inventory calculation.  ``run_f_pipeline`` retrieves the same
train-only shapes and asks the existing CP-SAT re-tiler to cover each shape
    within the operator's stock.  Solver success, exact-cover verification,
    ground contact and adjacent-layer connectivity are separate fields.  Neither
is support or stability, and neither function makes a physics claim.

Nothing in this module reads the frozen Phase 2 plan, cases, results or
scores.  Nothing it returns is a metric or evidence that one method improved.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.data.bricks import (PART_VOCAB, Brick, find_collisions,
                             format_bricks, is_connected, parse_bricks,
                             required_inventory, touches_ground)
from src.data.retile import occupancy_of, retile
from src.data.splits import MANIFEST_PATH, SplitManifest


# The object assignment is frozen evidence, not a file selected by the
# operator.  Tests replace both globals with a synthetic manifest; the CLI has
# no override flag and production always checks this digest before indexing.
FROZEN_SPLIT_MANIFEST_SHA256 = (
    "9f64bec66a0214e936b3c229577a3028bf26b4f9e610d1697ba7e79ccd626b1a")


class DeliveryError(ValueError):
    """The delivery request or its train-only catalogue is invalid."""


@dataclass(frozen=True)
class CatalogItem:
    """One canonical train structure; private identifiers stay internal."""

    catalog_id: str
    caption: str
    bricks: tuple[Brick, ...]
    required: dict[str, int]
    touches_ground: bool
    connected: bool
    structure_id: str = field(repr=False)
    object_id: str = field(repr=False)

    @property
    def n_bricks(self) -> int:
        return len(self.bricks)


@dataclass(frozen=True)
class TrainCatalog:
    source: Path
    sha256: str
    split_manifest_sha256: str
    items: tuple[CatalogItem, ...]


@dataclass(frozen=True)
class Comparison:
    item: CatalogItem
    lexical_score: float
    missing: dict[str, int]
    completion: float

    @property
    def buildable(self) -> bool:
        return (not self.missing and self.item.touches_ground
                and self.item.connected)

    @property
    def missing_total(self) -> int:
        return sum(self.missing.values())

    def as_dict(self) -> dict:
        return {
            "catalog_id": self.item.catalog_id,
            "caption": self.item.caption,
            "lexical_score": self.lexical_score,
            "retrieval_kind": "deterministic lexical baseline",
            "n_bricks": self.item.n_bricks,
            "required_inventory": dict(self.item.required),
            "fully_buildable": self.buildable,
            "missing_parts": dict(self.missing),
            "missing_total": self.missing_total,
            "inventory_completion": self.completion,
            "touches_ground": self.item.touches_ground,
            "stud_only_connected": self.item.connected,
            "static_structure_ready": (self.item.touches_ground
                                       and self.item.connected),
        }


@dataclass(frozen=True)
class ExistingComparison:
    retrieved: tuple[Comparison, ...]
    ranked: tuple[Comparison, ...]
    selected: Comparison | None
    excluded_same_object: int


@dataclass(frozen=True)
class PipelineAttempt:
    comparison: Comparison
    solver_status: str
    wall_seconds: float
    candidates: int
    solver_returned_tiling: bool
    exact_cover_verified: bool
    inventory_verified: bool
    collision_free: bool
    in_bounds: bool
    touches_ground: bool
    connected: bool
    failure: str | None
    bricks: tuple[Brick, ...] = field(default=(), repr=False)

    @property
    def delivery_ready(self) -> bool:
        return (
            self.solver_returned_tiling
            and self.exact_cover_verified
            and self.inventory_verified
            and self.collision_free
            and self.in_bounds
            and self.touches_ground
            and self.connected
        )

    def as_dict(self) -> dict:
        return {
            "catalog_id": self.comparison.item.catalog_id,
            "caption": self.comparison.item.caption,
            "lexical_score": self.comparison.lexical_score,
            "solver_status": self.solver_status,
            "wall_seconds": self.wall_seconds,
            "candidate_placements": self.candidates,
            "solver_returned_tiling": self.solver_returned_tiling,
            "exact_cover_verified": self.exact_cover_verified,
            "inventory_verified": self.inventory_verified,
            "collision_free": self.collision_free,
            "in_bounds": self.in_bounds,
            "touches_ground": self.touches_ground,
            "stud_only_connected": self.connected,
            "delivery_ready": self.delivery_ready,
            "failure": self.failure,
        }


@dataclass(frozen=True)
class PipelineResult:
    attempts: tuple[PipelineAttempt, ...]
    selected: PipelineAttempt | None
    excluded_same_object: int

    @property
    def status(self) -> str:
        if self.selected and self.selected.delivery_ready:
            return "success"
        if any(a.solver_returned_tiling for a in self.attempts):
            return "tiling_found_but_not_delivery_ready"
        return "no_valid_build"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _public_id(structure_id: str) -> str:
    return hashlib.sha256(structure_id.encode("utf-8")).hexdigest()[:16]


def load_train_catalog(path: str | Path) -> TrainCatalog:
    """Load only canonical ``control/exact`` structures from a train JSONL.

    The counterfactual corpus stores eight stock variants per pair.  Indexing
    all eight would present one geometry as eight independent works, so the
    canonical control/exact row is the catalogue item and inventory matching
    is derived again from its brick list.
    """
    source = Path(path)
    if not source.is_file():
        raise DeliveryError(f"train catalogue is not a file: {source}")
    if not source.name.endswith("_train.jsonl"):
        raise DeliveryError(
            "the catalogue filename must end in _train.jsonl; a delivery "
            "index may contain train rows only")

    manifest_path = Path(MANIFEST_PATH)
    if not manifest_path.is_file():
        raise DeliveryError(
            f"the frozen split manifest is unavailable: {manifest_path}")
    manifest_digest = _sha256(manifest_path)
    if manifest_digest != FROZEN_SPLIT_MANIFEST_SHA256:
        raise DeliveryError(
            "the frozen split manifest digest differs: expected "
            f"{FROZEN_SPLIT_MANIFEST_SHA256}, got {manifest_digest}")
    try:
        manifest = SplitManifest.load(manifest_path)
        if not isinstance(manifest.objects, dict) or not isinstance(
                manifest.structures, dict) or not isinstance(manifest.meta,
                                                              dict):
            raise TypeError(
                "objects, structures and meta must all be JSON objects")
        manifest.check_no_leakage()
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise DeliveryError(
            f"the frozen split manifest is invalid: {exc}") from exc

    items: list[CatalogItem] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DeliveryError(
                    f"invalid JSON in train catalogue line {line_no}: "
                    f"{exc.msg}") from exc
            if not isinstance(row, dict):
                raise DeliveryError(
                    f"train catalogue line {line_no} must be a JSON object; "
                    f"got {type(row).__name__}")
            if row.get("split") != "train":
                raise DeliveryError(
                    f"catalogue line {line_no} declares split="
                    f"{row.get('split')!r}; the index is train-only")
            oid = row.get("object_id")
            sid = row.get("structure_id")
            if not isinstance(oid, str) or not oid:
                raise DeliveryError(
                    f"train catalogue line {line_no} has no non-empty "
                    "object_id")
            if oid not in manifest.objects:
                raise DeliveryError(
                    f"train catalogue line {line_no} names object_id absent "
                    "from the frozen split manifest")
            assigned = manifest.objects[oid]
            if assigned != "train":
                raise DeliveryError(
                    f"train catalogue line {line_no} object_id is assigned "
                    f"to {assigned!r} by the frozen split manifest")
            if not isinstance(sid, str) or not sid:
                raise DeliveryError(
                    f"train catalogue line {line_no} has no non-empty "
                    "structure_id")
            manifest_oid = manifest.structures.get(sid)
            if manifest_oid != oid:
                reason = ("is absent from" if manifest_oid is None else
                          "belongs to another object_id in")
                raise DeliveryError(
                    f"train catalogue line {line_no} structure_id {reason} "
                    "the frozen split manifest")
            if row.get("role") != "control" or row.get("variant") != "exact":
                continue

            required_keys = ("object_id", "structure_id", "caption",
                             "bricks_txt")
            missing = [key for key in required_keys if key not in row]
            if missing:
                raise DeliveryError(
                    f"train catalogue line {line_no} is missing {missing}")
            for key in ("caption", "bricks_txt"):
                value = row[key]
                if not isinstance(value, str) or not value.strip():
                    raise DeliveryError(
                        f"train catalogue line {line_no} field {key} must be "
                        "a non-empty string")
            if sid in seen:
                raise DeliveryError(
                    f"duplicate canonical structure_id at line {line_no}")
            try:
                bricks = parse_bricks(row["bricks_txt"])
            except ValueError as exc:
                raise DeliveryError(
                    f"invalid bricks in train catalogue line {line_no}: "
                    f"{exc}") from exc
            if not bricks:
                raise DeliveryError(
                    f"train catalogue line {line_no} has no bricks")
            if any(b.part not in PART_VOCAB for b in bricks):
                raise DeliveryError(
                    f"train catalogue line {line_no} contains an unknown part")
            if any(not b.in_bounds() for b in bricks) or find_collisions(bricks):
                raise DeliveryError(
                    f"train catalogue line {line_no} is not valid geometry")
            grounded = touches_ground(bricks)
            connected = is_connected(bricks)
            if not grounded:
                raise DeliveryError(
                    f"train catalogue line {line_no} does not touch ground; "
                    "it is not a static delivery candidate")
            if not connected:
                raise DeliveryError(
                    f"train catalogue line {line_no} is not one component "
                    "under adjacent-layer connectivity")
            seen.add(sid)
            items.append(CatalogItem(
                catalog_id=_public_id(sid),
                caption=row["caption"].strip(),
                bricks=tuple(bricks),
                required=dict(sorted(required_inventory(bricks).items())),
                touches_ground=grounded,
                connected=connected,
                structure_id=sid,
                object_id=oid,
            ))

    if not items:
        raise DeliveryError("the train catalogue has no control/exact structures")
    items.sort(key=lambda item: item.catalog_id)
    return TrainCatalog(source=source, sha256=_sha256(source),
                        split_manifest_sha256=manifest_digest,
                        items=tuple(items))


_WORD = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK = re.compile(r"[\u3400-\u9fff]")


def _features(text: str) -> Counter[str]:
    normal = unicodedata.normalize("NFKC", text).casefold()
    words = _WORD.findall(normal)
    cjk = "".join(_CJK.findall(normal))
    compact = "".join(ch for ch in normal if ch.isalnum())
    features: Counter[str] = Counter("w:" + word for word in words)
    for source, prefix in ((cjk, "zh:"), (compact, "c:")):
        for n in (2, 3):
            features.update(prefix + source[i:i + n]
                            for i in range(max(0, len(source) - n + 1)))
    return features


def lexical_similarity(a: str, b: str) -> float:
    """Cosine similarity over deterministic Unicode lexical features."""
    left, right = _features(a), _features(b)
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    norm_l = math.sqrt(sum(value * value for value in left.values()))
    norm_r = math.sqrt(sum(value * value for value in right.values()))
    return dot / (norm_l * norm_r) if dot else 0.0


def inventory_evidence(required: dict[str, int],
                       inventory: dict[str, int]) -> dict:
    """Exactly what a stock is short of for a given requirement.

    Split out from :func:`_comparison` so the multilingual retrieval front end
    in :mod:`src.retrieval` computes missing parts and completion with *this*
    arithmetic rather than its own.  Two implementations of "how short are we"
    is how a page and a command line come to disagree about whether something
    can be built.
    """
    missing = {
        part: need - inventory.get(part, 0)
        for part, need in required.items()
        if need > inventory.get(part, 0)
    }
    total = sum(required.values())
    supplied = sum(min(need, inventory.get(part, 0))
                   for part, need in required.items())
    return {
        "missing": dict(sorted(missing.items())),
        "missing_total": sum(missing.values()),
        "required_total": total,
        "supplied_total": supplied,
        "completion": (supplied / total if total else 0.0),
    }


def _comparison(item: CatalogItem, score: float,
                inventory: dict[str, int]) -> Comparison:
    evidence = inventory_evidence(item.required, inventory)
    return Comparison(
        item=item, lexical_score=round(float(score), 8),
        missing=evidence["missing"], completion=evidence["completion"])


def retrieve(catalog: TrainCatalog, caption: str, inventory: dict[str, int],
             *, top_n: int = 10,
             exclude_object_id: str | None = None) -> tuple[tuple[Comparison, ...], int]:
    """Return caption-ranked train candidates and the same-object exclusion count."""
    if not isinstance(caption, str) or not caption.strip():
        raise DeliveryError("a non-empty caption is required")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise DeliveryError("top_n must be a positive whole number")
    if not inventory:
        raise DeliveryError("an inventory is required")
    if exclude_object_id is not None and (
            not isinstance(exclude_object_id, str)
            or not exclude_object_id.strip()):
        raise DeliveryError(
            "exclude_object_id must be a non-empty string when supplied")

    eligible = [item for item in catalog.items
                if item.object_id != exclude_object_id]
    excluded = len(catalog.items) - len(eligible)
    if not eligible:
        raise DeliveryError("same-object exclusion removed the whole catalogue")
    scored = [_comparison(item, lexical_similarity(caption, item.caption),
                          inventory)
              for item in eligible]
    scored.sort(key=lambda c: (-c.lexical_score, c.item.catalog_id))
    return tuple(scored[:top_n]), excluded


def compare_existing(catalog: TrainCatalog, caption: str,
                     inventory: dict[str, int], *, top_n: int = 10,
                     exclude_object_id: str | None = None) -> ExistingComparison:
    """Retrieve first, then re-rank that set by exact inventory evidence."""
    got, excluded = retrieve(
        catalog, caption, inventory, top_n=top_n,
        exclude_object_id=exclude_object_id)
    ranked = tuple(sorted(
        got,
        key=lambda c: (not c.buildable, c.missing_total,
                       -c.completion, -c.lexical_score, c.item.catalog_id)))
    selected = next((candidate for candidate in ranked if candidate.buildable),
                    None)
    return ExistingComparison(
        retrieved=got, ranked=ranked, selected=selected,
        excluded_same_object=excluded)


def _attempt(comparison: Comparison, inventory: dict[str, int], *,
             time_limit: float, seed: int) -> PipelineAttempt:
    source_bricks = list(comparison.item.bricks)
    target = occupancy_of(source_bricks)
    outcome = retile(
        target, allowed=frozenset(inventory), budget=dict(inventory),
        time_limit=time_limit, seed=seed, workers=1)
    if not outcome.ok:
        return PipelineAttempt(
            comparison=comparison, solver_status=outcome.status,
            wall_seconds=outcome.wall_seconds, candidates=outcome.candidates,
            solver_returned_tiling=False, exact_cover_verified=False,
            inventory_verified=False, collision_free=False, in_bounds=False,
            touches_ground=False, connected=False,
            failure=("solver timeout" if outcome.status == "UNKNOWN" else
                     "no tiling within the supplied inventory"))

    bricks = tuple(outcome.bricks or ())
    cells = [cell for brick in bricks for cell in brick.cells]
    exact = set(cells) == target and len(cells) == len(target)
    used = required_inventory(list(bricks))
    stock_ok = all(n <= inventory.get(part, 0)
                   for part, n in used.items())
    collision_free = not find_collisions(list(bricks))
    in_bounds = all(brick.in_bounds() for brick in bricks)
    grounded = touches_ground(list(bricks))
    connected = is_connected(list(bricks))
    failures = []
    if not exact:
        failures.append("exact cover verification failed")
    if not stock_ok:
        failures.append("inventory verification failed")
    if not collision_free:
        failures.append("collision verification failed")
    if not in_bounds:
        failures.append("bounds verification failed")
    if not grounded:
        failures.append("the tiling does not touch ground")
    if not connected:
        failures.append("adjacent-layer connectivity is not one component")
    return PipelineAttempt(
        comparison=comparison, solver_status=outcome.status,
        wall_seconds=outcome.wall_seconds, candidates=outcome.candidates,
        solver_returned_tiling=True, exact_cover_verified=exact,
        inventory_verified=stock_ok, collision_free=collision_free,
        in_bounds=in_bounds, touches_ground=grounded, connected=connected,
        failure="; ".join(failures) or None, bricks=bricks)


def run_f_pipeline(catalog: TrainCatalog, caption: str,
                   inventory: dict[str, int], *, top_n: int = 5,
                   time_limit: float = 2.0, seed: int = 0,
                   exclude_object_id: str | None = None) -> PipelineResult:
    """Lexically retrieve train shapes, then CP-SAT re-tile under stock.

    Candidates are tried in retrieval order.  The first independently
    verified, grounded and connected tiling is selected.  Infeasibility,
    timeout and a tiling that fails the checker remain distinguishable in
    ``attempts``.
    """
    if (isinstance(time_limit, bool)
            or not isinstance(time_limit, (int, float))
            or not math.isfinite(float(time_limit))
            or time_limit <= 0):
        raise DeliveryError(
            "time_limit must be a finite number greater than zero")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DeliveryError("seed must be a non-negative whole number")
    got, excluded = retrieve(
        catalog, caption, inventory, top_n=top_n,
        exclude_object_id=exclude_object_id)
    attempts: list[PipelineAttempt] = []
    selected = None
    for candidate in got:
        attempt = _attempt(candidate, inventory,
                           time_limit=float(time_limit), seed=seed)
        attempts.append(attempt)
        if attempt.delivery_ready:
            selected = attempt
            break
    return PipelineResult(
        attempts=tuple(attempts), selected=selected,
        excluded_same_object=excluded)


def selected_text(result: ExistingComparison | PipelineResult) -> str | None:
    """Brick grammar for the selected result, if one is delivery-ready."""
    if isinstance(result, ExistingComparison):
        return (format_bricks(list(result.selected.item.bricks))
                if result.selected else None)
    return (format_bricks(list(result.selected.bricks))
            if result.selected else None)
