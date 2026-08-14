"""Full audit of the generated dataset, recomputed from the files.

Every property is recomputed from the written JSONL instead of reading the
``checks`` field stored beside it, so a generator that verified itself wrongly
cannot pass on that basis.

This is *not* an independent reimplementation: the parser, the connectivity
predicate and the part vocabulary are the project's own, shared with the
generator. A bug inside those would be invisible to both. What it does catch is
a generator that mislabels, miscounts or fails to apply a check it claims to
have applied.

Writes data/reports/07_audit.md and exits non-zero if anything fails.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bricks import (  # noqa: E402
    PART_VOCAB,
    connected_components,
    is_connected,
    parse_bricks,
    touches_ground,
    unsupported_bricks,
)
from src.data.counterfactual import LOOSE_TAU, VARIANTS, read_jsonl  # noqa: E402
from src.data.retile import occupancy_of  # noqa: E402
from src.data.splits import SplitManifest  # noqa: E402

SPLITS = ("train", "val", "test")
OUT = ROOT / "data" / "reports"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    manifest = SplitManifest.load()
    failures: list[str] = []
    rows: list[str] = []
    per_split: dict[str, dict] = {}
    objects: dict[str, set[str]] = {}

    for split in SPLITS:
        path = ROOT / "data" / "processed" / f"counterfactual_{split}.jsonl"
        samples = read_jsonl(path)
        objects[split] = {s.object_id for s in samples}
        pairs: dict[str, list] = defaultdict(list)
        for s in samples:
            pairs[s.pair_id].append(s)

        stats = Counter()
        extra_hist = Counter()
        support_ok = 0
        targets = 0
        stud_single = 0
        ground_single = 0
        both_stud_connected = 0
        tried_hist = Counter()

        for pid, group in pairs.items():
            if len(group) != 2 * len(VARIANTS):
                failures.append(f"{split}/{pid}: {len(group)} samples, expected 8")
            if len({s.caption for s in group}) != 1:
                failures.append(f"{split}/{pid}: caption differs within pair")
            roles = Counter(s.role for s in group)
            if roles != {"control": 4, "counterfactual": 4}:
                failures.append(f"{split}/{pid}: roles {dict(roles)}")

            occs = set()
            for s in group:
                bricks = parse_bricks(s.bricks_txt)
                occ = occupancy_of(bricks)
                occs.add(frozenset(occ))

                # split provenance
                if manifest.split_of_object(s.object_id) != split:
                    failures.append(f"{split}/{s.sample_id}: split mismatch")

                # rotation normalisation
                for p in list(s.inventory) + list(s.used):
                    if p not in PART_VOCAB:
                        failures.append(f"{s.sample_id}: non-canonical part {p}")

                # inventory legality
                used = Counter(b.part for b in bricks)
                if dict(used) != s.used:
                    failures.append(f"{s.sample_id}: used field disagrees with bricks")
                for p, n in used.items():
                    if s.inventory.get(p, 0) < n:
                        failures.append(f"{s.sample_id}: overdraws {p}")

                # counterfactual arm
                if s.role == "counterfactual":
                    if s.dropped_part in used:
                        failures.append(f"{s.sample_id}: uses dropped {s.dropped_part}")
                    if s.dropped_part in s.inventory:
                        failures.append(f"{s.sample_id}: offered dropped part")
                    if s.dropped_part == "1x1":
                        failures.append(f"{s.sample_id}: dropped 1x1")

                # variant semantics
                extra = set(s.inventory) - set(s.used)
                if s.variant in ("distractor", "mixed"):
                    if not extra:
                        failures.append(f"{s.sample_id}: {s.variant} adds nothing")
                    extra_hist[len(extra)] += 1
                elif extra:
                    failures.append(f"{s.sample_id}: {s.variant} should add nothing")
                if s.variant in ("loose", "mixed"):
                    for p, n in s.used.items():
                        if s.inventory[p] != math.ceil(LOOSE_TAU * n):
                            failures.append(f"{s.sample_id}: loose scaling wrong")
                if s.variant == "exact" and s.inventory != s.used:
                    failures.append(f"{s.sample_id}: exact not tight")

                stats["samples"] += 1

            # Geometry is a property of the target, not of the inventory
            # framing: the four variants share one tiling, so counting them
            # separately would inflate every structural figure fourfold.
            for role in ("control", "counterfactual"):
                variants = [s for s in group if s.role == role]
                if len({v.bricks_txt for v in variants}) != 1:
                    failures.append(f"{split}/{pid}/{role}: variants differ in geometry")
                s = variants[0]
                bricks = parse_bricks(s.bricks_txt)
                n_stud = len(connected_components(bricks))
                n_ground = len(connected_components(bricks, ground=True))
                if n_stud != 1:
                    failures.append(f"{s.sample_id}: {n_stud} stud components")
                if n_stud != s.n_components:
                    failures.append(f"{s.sample_id}: n_components disagrees")
                if n_ground != s.n_ground_components:
                    failures.append(f"{s.sample_id}: n_ground_components disagrees")
                if n_ground > n_stud:
                    failures.append(f"{s.sample_id}: ground split more than studs")
                if not touches_ground(bricks):
                    failures.append(f"{s.sample_id}: floats")

                targets += 1
                stud_single += n_stud == 1
                ground_single += n_ground == 1
                support_ok += not unsupported_bricks(bricks)

            if len(occs) != 1:
                failures.append(f"{split}/{pid}: arms cover different voxels")

            ctrl = next(s for s in group if s.role == "control")
            cfx = next(s for s in group if s.role == "counterfactual")
            if ctrl.used == cfx.used:
                failures.append(f"{split}/{pid}: arms identical, no counterfactual")
            if is_connected(parse_bricks(ctrl.bricks_txt)) and is_connected(
                parse_bricks(cfx.bricks_txt)
            ):
                both_stud_connected += 1
            else:
                failures.append(f"{split}/{pid}: an arm is not stud-connected")
            tried_hist[len(cfx.tried_parts)] += 1

        per_split[split] = {
            "pairs": len(pairs),
            "samples": stats["samples"],
            "unique_targets": targets,
            "objects": len(objects[split]),
            "distractor_extra_hist": dict(sorted(extra_hist.items())),
            "support_rate": support_ok / targets if targets else 0.0,
            "stud_single_targets": stud_single,
            "ground_single_targets": ground_single,
            "pairs_both_stud_connected": both_stud_connected,
            "parts_tried_hist": dict(sorted(tried_hist.items())),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }

    overlaps = {f"{a}|{b}": len(objects[a] & objects[b])
                for a in SPLITS for b in SPLITS if a < b}
    for k, v in overlaps.items():
        if v:
            failures.append(f"cross-split overlap {k}: {v}")
    if any(0 in d["distractor_extra_hist"] for d in per_split.values()):
        failures.append("a distractor/mixed sample added zero parts")

    rows = ["# Dataset audit (recomputed from files)", ""]
    rows.append("Everything below is recalculated from the JSONL, not read from "
                "the stored `checks` field. It shares the project's parser and "
                "predicates with the generator, so it verifies that the checks "
                "were applied and labelled correctly -- not that those "
                "predicates are themselves right.")
    rows += ["", "Each pair contributes two distinct tilings (control and "
             "counterfactual); the four inventory framings reuse the same "
             "geometry, so samples = 4 x unique targets.", "",
             "| split | pairs | unique targets | samples | objects | bytes |",
             "|---|---:|---:|---:|---:|---:|"]
    for s, d in per_split.items():
        rows.append(f"| {s} | {d['pairs']} | **{d['unique_targets']}** | "
                    f"{d['samples']} | {d['objects']} | {d['bytes']} |")
    rows += ["", "| split | sha256 |", "|---|---|"]
    for s, d in per_split.items():
        rows.append(f"| {s} | `{d['sha256']}` |")
    rows += ["", "## Components", "",
             "Stud coupling is the gate. The baseplate column shows the weaker "
             "criterion that is deliberately *not* applied.", "",
             "| split | unique targets | stud single | ground single | pairs both stud-connected |",
             "|---|---:|---:|---:|---:|"]
    for s, d in per_split.items():
        rows.append(f"| {s} | {d['unique_targets']} | {d['stud_single_targets']} | "
                    f"{d['ground_single_targets']} | "
                    f"{d['pairs_both_stud_connected']}/{d['pairs']} |")
    rows += ["", "## Support (reported, not gated; over unique targets)", "",
             "| split | unique targets | fully supported |", "|---|---:|---:|"]
    for s, d in per_split.items():
        rows.append(f"| {s} | {d['unique_targets']} | {d['support_rate']:.1%} |")
    rows += ["", "## Droppable parts tried before success", ""]
    for s, d in per_split.items():
        rows.append(f"- {s}: {d['parts_tried_hist']}")
    rows += ["", "## Distractor extra-type histogram (0 must be absent)", ""]
    for s, d in per_split.items():
        rows.append(f"- {s}: {d['distractor_extra_hist']}")
    rows += ["", "## Cross-split object overlap", ""]
    for k, v in overlaps.items():
        rows.append(f"- `{k}`: **{v}**")
    rows += ["", "## Result", ""]
    rows.append(f"- checks failed: **{len(failures)}**")
    for f in failures[:20]:
        rows.append(f"  - {f}")
    rows.append("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "07_audit.md").write_text("\n".join(rows), encoding="utf-8")
    (OUT / "07_audit.json").write_text(
        json.dumps({"per_split": per_split, "overlaps": overlaps,
                    "failures": failures}, indent=2), encoding="utf-8")
    print("\n".join(rows))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
