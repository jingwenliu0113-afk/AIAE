"""Frozen train/validation/test assignment, keyed by ``object_id``.

Everything downstream -- LoRA data, retrieval index, F-pipeline shapes,
CP-SAT derived samples -- inherits its split from here.  Splitting per row
instead would leak: 18,790 objects have more than one structure, so the same
object would land on both sides.

The upstream train/test boundary is kept exactly as published (measured: zero
objects cross it).  Only validation is carved out, from train, by hashing the
object id -- so the assignment is stable under reordering, resampling, or
adding rows later, in a way that shuffling with a seed is not.

The manifest is written once and then treated as read-only.  ``build`` refuses
to overwrite unless explicitly forced, because silently regenerating it would
invalidate every experiment measured against the previous one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "splits" / "object_splits.json"

VAL_FRACTION = 0.05
HASH_SALT = "brickagain-v1"
BUCKETS = 10_000


def _bucket(object_id: str) -> int:
    """Stable hash bucket.  Python's hash() is salted per process; this is not."""
    h = hashlib.sha256(f"{HASH_SALT}:{object_id}".encode()).hexdigest()
    return int(h[:8], 16) % BUCKETS


def assign(object_id: str, upstream_split: str, val_fraction: float = VAL_FRACTION) -> str:
    """Which split an object belongs to.  Pure function of the id."""
    if upstream_split == "test":
        return "test"
    return "val" if _bucket(object_id) < val_fraction * BUCKETS else "train"


@dataclass
class SplitManifest:
    objects: dict[str, str]          # object_id -> train | val | test
    structures: dict[str, str]       # structure_id -> object_id
    meta: dict

    def split_of_object(self, object_id: str) -> str:
        return self.objects[object_id]

    def split_of_structure(self, structure_id: str) -> str:
        return self.objects[self.structures[structure_id]]

    def object_of_structure(self, structure_id: str) -> str:
        return self.structures[structure_id]

    def ids(self, split: str) -> list[str]:
        return sorted(o for o, s in self.objects.items() if s == split)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.objects.values():
            out[s] = out.get(s, 0) + 1
        return dict(sorted(out.items()))

    def check_no_leakage(self) -> None:
        """Every structure of an object must sit in that object's split."""
        for sid, oid in self.structures.items():
            if oid not in self.objects:
                raise ValueError(f"structure {sid} references unknown object {oid}")

    def save(self, path: Path = MANIFEST_PATH, *, force: bool = False) -> Path:
        if path.exists() and not force:
            raise FileExistsError(
                f"{path} already exists; regenerating it would invalidate every "
                f"experiment measured against it. Pass force=True to overwrite."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "meta": self.meta,
                    "counts": self.counts(),
                    "objects": self.objects,
                    "structures": self.structures,
                },
                indent=0,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path = MANIFEST_PATH) -> "SplitManifest":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(objects=d["objects"], structures=d["structures"], meta=d["meta"])


def build(rows_by_split: dict[str, list[dict]], *, val_fraction: float = VAL_FRACTION) -> SplitManifest:
    """``rows_by_split`` maps upstream split name -> rows with structure_id/object_id."""
    objects: dict[str, str] = {}
    structures: dict[str, str] = {}
    conflicts: list[str] = []

    for upstream, rows in rows_by_split.items():
        for r in rows:
            oid, sid = r["object_id"], r["structure_id"]
            structures[sid] = oid
            target = assign(oid, upstream, val_fraction)
            if oid in objects and objects[oid] != target:
                conflicts.append(oid)
            objects[oid] = target

    if conflicts:
        raise ValueError(
            f"{len(set(conflicts))} object_id appear in more than one upstream "
            f"split; the published split was assumed clean. e.g. {conflicts[:3]}"
        )

    m = SplitManifest(
        objects=objects,
        structures=structures,
        meta={
            "created": date.today().isoformat(),
            "salt": HASH_SALT,
            "val_fraction": val_fraction,
            "buckets": BUCKETS,
            "method": "sha256(salt:object_id) bucket; test kept from upstream",
            "source": "AvaLovelace/StableText2Brick",
        },
    )
    m.check_no_leakage()
    return m
