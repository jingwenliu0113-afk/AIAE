"""Generate the paired counterfactual dataset.

Usage: 04_build_counterfactual.py [--train N --val N --test N] [--max-bricks N]

A pair is kept only when *both* arms clear every gate: solver feasible, exact
voxel cover, inventory legal, dropped part absent from target and from all four
inventory framings, and -- the expensive one -- a single connected component
under stud coupling alone.  The baseplate is not a part, carries no inventory
and is never written out, so it never counts towards connection; it is reported
separately as an anchoring metric.

Support is measured and reported but not enforced: 93.3% of re-tiled targets
carry at least one brick with nothing directly below, as does 83.8% of the
corpus, so rejecting them would gut the dataset and shift it away from the base
model's distribution (data/reports/08_corpus_structure.md).

Every sample inherits object_id and split from its source row via the frozen
manifest, so derived data cannot cross the split boundary.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402

from src.data.bricks import parse_bricks  # noqa: E402
from src.data.counterfactual import (  # noqa: E402
    VARIANTS,
    GenerationError,
    make_pair,
    write_jsonl,
)
from src.data.splits import SplitManifest  # noqa: E402

WANT = {"train": 1200, "val": 200, "test": 200}
MAX_BRICKS = 150
MAX_ATTEMPT_FACTOR = 12     # measured yield is ~14.5%; see sizing in the report
SEED = 0
OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "data" / "reports"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> None:
    for split in WANT:
        flag = f"--{split}"
        if flag in sys.argv:
            WANT[split] = int(sys.argv[sys.argv.index(flag) + 1])


def main() -> None:
    parse_args()
    manifest = SplitManifest.load()
    ds = load_dataset("AvaLovelace/StableText2Brick")
    rows = [r for v in ds.values()
            for r in v.select_columns(
                ["structure_id", "object_id", "captions", "bricks"]).to_list()]

    by_split: dict[str, list[dict]] = defaultdict(list)
    too_big = 0
    for r in rows:
        if len(parse_bricks(r["bricks"])) > MAX_BRICKS:
            too_big += 1
            continue
        by_split[manifest.split_of_structure(r["structure_id"])].append(r)

    print(f"eligible sources (<= {MAX_BRICKS} bricks): "
          f"{ {k: len(v) for k, v in by_split.items()} } (excluded {too_big})")

    rng = random.Random(SEED)
    report: dict[str, dict] = {}
    t0 = time.time()

    for split, want in WANT.items():
        pool = list(by_split[split])
        rng.shuffle(pool)
        samples = []
        failures: Counter[str] = Counter()
        dropped: Counter[str] = Counter()
        extra_types: Counter[str] = Counter()
        extra_count: Counter[int] = Counter()
        unsupported_free = 0
        stud_single = 0
        ground_single = 0
        parts_tried: list[int] = []
        targets = 0
        attempted = 0
        limit = want * MAX_ATTEMPT_FACTOR

        for row in pool[:limit]:
            if len({s.pair_id for s in samples}) >= want:
                break
            attempted += 1
            try:
                pair = make_pair(row, split, seed=SEED)
            except GenerationError as e:
                failures[str(e).split(":")[0].strip()] += 1
                continue
            if any(not all(s.checks.values()) for s in pair):
                failures["verification"] += 1
                continue
            samples.extend(pair)
            dropped[pair[-1].dropped_part] += 1
            for s in pair:
                if s.variant in ("distractor", "mixed"):
                    extra_count[len(s.extra_parts)] += 1
                    extra_types.update(s.extra_parts)
            for role in ("control", "counterfactual"):
                one = next(s for s in pair if s.role == role)
                targets += 1
                unsupported_free += one.n_unsupported == 0
                stud_single += one.n_components == 1
                ground_single += one.n_ground_components == 1
            parts_tried.append(
                len(next(s for s in pair if s.role == "counterfactual").tried_parts)
            )

        path = write_jsonl(samples, OUT_DIR / f"counterfactual_{split}.jsonl")
        pairs = len({s.pair_id for s in samples})
        report[split] = {
            "attempted_sources": attempted,
            "pairs_kept": pairs,
            "samples": len(samples),
            "pair_yield": pairs / attempted if attempted else 0.0,
            "hit_target": pairs >= want,
            "failures": dict(failures.most_common()),
            "dropped_part_distribution": dict(dropped.most_common()),
            "distractor_extra_type_counts": dict(sorted(extra_count.items())),
            "distractor_extra_types": dict(extra_types.most_common()),
            "targets": targets,
            "fully_supported_targets": unsupported_free,
            "support_rate": unsupported_free / targets if targets else 0.0,
            "stud_single_component": stud_single,
            "ground_single_component": ground_single,
            "mean_parts_tried": (
                sum(parts_tried) / len(parts_tried) if parts_tried else 0.0
            ),
            "max_parts_tried": max(parts_tried) if parts_tried else 0,
            "objects": len({s.object_id for s in samples}),
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        print(f"  {split:6s} {pairs:5d}/{want} pairs from {attempted} attempts "
              f"({pairs/attempted:.1%} yield) -> {len(samples)} samples")

    objects_by_split = {
        s: {json.loads(l)["object_id"]
            for l in (OUT_DIR / f"counterfactual_{s}.jsonl").read_text().splitlines()}
        for s in WANT
    }
    overlap = {f"{a}|{b}": len(objects_by_split[a] & objects_by_split[b])
               for a in objects_by_split for b in objects_by_split if a < b}

    L = ["# Counterfactual dataset", ""]
    L.append(f"- seed {SEED}, max source bricks {MAX_BRICKS}, "
             f"attempt cap {MAX_ATTEMPT_FACTOR}x target")
    L.append(f"- variants per target: {', '.join(VARIANTS)}")
    L.append(f"- wall clock: {time.time()-t0:.0f}s")
    L.append("")
    L.append("**Acceptance gate**: both arms solver-feasible, exact voxel cover, "
             "inventory legal, dropped part absent everywhere, and a single "
             "connected component **under stud coupling alone**. The baseplate "
             "is not a part, carries no inventory and is never written out, so "
             "it never counts towards connection; it appears only as the "
             "ground-single metric below. Support is reported, not enforced.")
    L.append("")
    L.append("Rather than drop a source when one chosen part fails, every "
             "droppable part is tried in a seeded order until one yields a "
             "stud-connected counterfactual.")
    L += ["", "## Yield", "",
          "| split | pairs kept | target met | attempts | yield | samples | objects |",
          "|---|---:|:--:|---:|---:|---:|---:|"]
    for s, d in report.items():
        L.append(f"| {s} | {d['pairs_kept']} | {'yes' if d['hit_target'] else 'NO'} | "
                 f"{d['attempted_sources']} | {d['pair_yield']:.1%} | "
                 f"{d['samples']} | {d['objects']} |")

    L += ["", "## Failure reasons", "",
          "| split | " + " | ".join(
              sorted({k for d in report.values() for k in d["failures"]})) + " |"]
    reasons = sorted({k for d in report.values() for k in d["failures"]})
    L.append("|---|" + "---:|" * len(reasons))
    for s, d in report.items():
        L.append(f"| {s} | " + " | ".join(
            str(d["failures"].get(r, 0)) for r in reasons) + " |")

    L += ["", "## Components (stud coupling gates; baseplate is a metric)", "",
          "| split | targets | stud single | ground single | parts tried (mean/max) |",
          "|---|---:|---:|---:|---|"]
    for s, d in report.items():
        L.append(f"| {s} | {d['targets']} | {d['stud_single_component']} | "
                 f"{d['ground_single_component']} | "
                 f"{d['mean_parts_tried']:.1f} / {d['max_parts_tried']} |")
    L += ["", "Every kept target is stud-single by construction. Ground-single "
          "is shown to make explicit that the weaker criterion is not what "
          "was applied.", ""]
    L += ["", "## Support (reported, not gated)", "",
          "| split | targets | fully supported | rate |", "|---|---:|---:|---:|"]
    for s, d in report.items():
        L.append(f"| {s} | {d['targets']} | {d['fully_supported_targets']} | "
                 f"{d['support_rate']:.1%} |")

    L += ["", "## Distractor effectiveness", "",
          "Every distractor and mixed sample must add at least one part the "
          "target does not use.", "",
          "| split | extra-type counts | types added |", "|---|---|---|"]
    for s, d in report.items():
        L.append(f"| {s} | {d['distractor_extra_type_counts']} | "
                 f"{d['distractor_extra_types']} |")

    L += ["", "## Dropped part distribution", ""]
    parts = sorted({p for d in report.values() for p in d["dropped_part_distribution"]})
    L.append("| split | " + " | ".join(parts) + " |")
    L.append("|---|" + "---:|" * len(parts))
    for s, d in report.items():
        L.append(f"| {s} | " + " | ".join(
            str(d["dropped_part_distribution"].get(p, 0)) for p in parts) + " |")

    L += ["", "## Cross-split object overlap (must all be 0)", ""]
    for k, v in overlap.items():
        L.append(f"- `{k}`: **{v}**")

    L += ["", "## Files", "", "| file | bytes | sha256 |", "|---|---:|---|"]
    for s, d in report.items():
        L.append(f"| `{d['path']}` | {d['bytes']} | `{d['sha256']}` |")
    L.append("")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "04_counterfactual.md").write_text("\n".join(L), encoding="utf-8")
    (REPORT_DIR / "04_counterfactual.json").write_text(
        json.dumps({"report": report, "cross_split_overlap": overlap}, indent=2),
        encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
