"""EDA on StableText2Brick.

Answers the questions the project plan is actually load-bearing on:

1. Which axis do ``h`` and ``w`` map to?  (settled by bounds violations)
2. Which raw dimension spellings occur, and how many parts after normalising
   rotation?
3. How many structures share an ``object_id``, and -- the important one --
   do those variants differ in brick composition?  That is the natural supply
   of counterfactual (same caption, different inventory) training pairs.
4. Does any ``object_id`` leak across the train/test split?

Writes a markdown report to data/reports/01_eda.md.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402

from src.data.bricks import (  # noqa: E402
    PART_VOCAB,
    WORLD,
    Brick,
    canonical_part,
    find_collisions,
    parse_bricks,
    required_inventory,
)

OUT = ROOT / "data" / "reports"
OUT.mkdir(parents=True, exist_ok=True)

COLS = ["structure_id", "object_id", "category_id", "bricks", "captions"]


def load() -> dict[str, list[dict]]:
    ds = load_dataset("AvaLovelace/StableText2Brick")
    # stability_scores is 20*20*20 floats per row; never materialise it here.
    return {k: v.select_columns(COLS).to_list() for k, v in ds.items()}


def main() -> None:
    data = load()
    lines: list[str] = ["# StableText2Brick EDA", ""]
    lines.append(f"- train rows: {len(data['train'])}")
    lines.append(f"- test rows: {len(data['test'])}")
    lines.append("")

    all_rows = [(s, r) for s, rows in data.items() for r in rows]

    # ---- 1. axis assignment -------------------------------------------------
    # Try both readings and count bricks that fall outside the 20^3 world.
    oob_hx = oob_hy = 0
    raw_dims: Counter[str] = Counter()
    parse_fail = 0
    total_bricks = 0

    parsed: dict[str, list[Brick]] = {}
    for _split, r in all_rows:
        try:
            bricks = parse_bricks(r["bricks"], strict=True)
        except Exception:
            parse_fail += 1
            continue
        parsed[r["structure_id"]] = bricks
        for b in bricks:
            total_bricks += 1
            raw_dims[f"{b.h}x{b.w}"] += 1
            # reading A (what we implement): h along x, w along y
            if not (b.x + b.h <= WORLD and b.y + b.w <= WORLD):
                oob_hx += 1
            # reading B: h along y, w along x
            if not (b.x + b.w <= WORLD and b.y + b.h <= WORLD):
                oob_hy += 1

    lines += [
        "## 1. Axis assignment",
        "",
        f"- bricks parsed: {total_bricks}",
        f"- structures that failed to parse: {parse_fail}",
        f"- out-of-bounds under `h->x, w->y` (implemented): **{oob_hx}**",
        f"- out-of-bounds under `h->y, w->x`: **{oob_hy}**",
        "",
    ]

    # ---- 2. rotation / vocabulary ------------------------------------------
    canon: Counter[str] = Counter()
    for dim, n in raw_dims.items():
        h, w = (int(v) for v in dim.split("x"))
        canon[canonical_part(h, w)] += n

    lines += ["## 2. Dimension spellings", ""]
    lines.append(f"- distinct raw spellings: **{len(raw_dims)}**")
    lines.append(f"- distinct parts after rotation normalisation: **{len(canon)}**")
    lines.append("")
    lines.append("| raw spelling | count |")
    lines.append("|---|---:|")
    for dim, n in raw_dims.most_common():
        lines.append(f"| `{dim}` | {n} |")
    lines.append("")
    lines.append("| canonical part | count | share |")
    lines.append("|---|---:|---:|")
    for p, n in canon.most_common():
        lines.append(f"| `{p}` | {n} | {n / total_bricks:.1%} |")
    lines.append("")
    unexpected = set(canon) - set(PART_VOCAB)
    lines.append(f"- parts outside declared PART_VOCAB: {sorted(unexpected) or 'none'}")
    lines.append("")

    # ---- 3. structures per object_id ---------------------------------------
    by_obj: dict[str, list[str]] = defaultdict(list)
    split_of: dict[str, set[str]] = defaultdict(set)
    for split, r in all_rows:
        by_obj[r["object_id"]].append(r["structure_id"])
        split_of[r["object_id"]].add(split)

    per_obj = Counter(len(v) for v in by_obj.values())
    lines += ["## 3. Structures per object_id", ""]
    lines.append(f"- distinct object_id: **{len(by_obj)}**")
    lines.append(f"- mean structures per object: {len(all_rows)/len(by_obj):.2f}")
    lines.append("")
    lines.append("| structures | #objects |")
    lines.append("|---:|---:|")
    for k in sorted(per_obj):
        lines.append(f"| {k} | {per_obj[k]} |")
    lines.append("")

    # ---- 4. do variants differ in composition? -----------------------------
    multi = {o: sids for o, sids in by_obj.items() if len(sids) > 1}
    same_inv = 0
    diff_counts_only = 0
    diff_types = 0
    typeset_examples: list[tuple[str, list[str]]] = []

    for obj, sids in multi.items():
        invs = []
        for sid in sids:
            b = parsed.get(sid)
            if b is None:
                continue
            invs.append(required_inventory(b))
        if len(invs) < 2:
            continue
        uniq = {tuple(sorted(i.items())) for i in invs}
        if len(uniq) == 1:
            same_inv += 1
            continue
        typesets = {frozenset(i) for i in invs}
        if len(typesets) > 1:
            diff_types += 1
            if len(typeset_examples) < 5:
                typeset_examples.append(
                    (obj, [",".join(sorted(t)) for t in typesets])
                )
        else:
            diff_counts_only += 1

    lines += ["## 4. Counterfactual supply (the load-bearing number)", ""]
    lines.append(f"- object_id with >1 structure: **{len(multi)}**")
    lines.append(f"- ...whose variants have identical inventories: {same_inv}")
    lines.append(f"- ...differing in counts only: {diff_counts_only}")
    lines.append(
        f"- ...differing in which part *types* are used: **{diff_types}**"
    )
    lines.append("")
    if typeset_examples:
        lines.append("Examples of type-set differences:")
        lines.append("")
        for obj, ts in typeset_examples:
            lines.append(f"- `{obj}`")
            for t in ts:
                lines.append(f"  - {{{t}}}")
        lines.append("")

    # ---- 5. leakage ---------------------------------------------------------
    leaked = [o for o, s in split_of.items() if len(s) > 1]
    lines += ["## 5. Train/test leakage", ""]
    lines.append(f"- object_id appearing in both splits: **{len(leaked)}**")
    lines.append("")

    # ---- 6. size + sanity ---------------------------------------------------
    sizes = sorted(len(b) for b in parsed.values())
    n = len(sizes)
    def pct(p: float) -> int:
        return sizes[min(n - 1, int(p * n))]

    collide = sum(1 for b in parsed.values() if find_collisions(b))
    cats = Counter(r["category_id"] for _s, r in all_rows)

    lines += ["## 6. Structure size and sanity", ""]
    lines.append(f"- bricks per structure: min {sizes[0]}, p25 {pct(.25)}, "
                 f"median {pct(.5)}, p75 {pct(.75)}, p95 {pct(.95)}, max {sizes[-1]}")
    lines.append(f"- structures with internal collisions: **{collide}**")
    lines.append(f"- distinct category_id: {len(cats)}")
    lines.append("")
    lines.append("| category_id | rows |")
    lines.append("|---|---:|")
    for c, k in cats.most_common():
        lines.append(f"| `{c}` | {k} |")
    lines.append("")

    (OUT / "01_eda.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "rows": {k: len(v) for k, v in data.items()},
        "total_bricks": total_bricks,
        "parse_failures": parse_fail,
        "oob_h_to_x": oob_hx,
        "oob_h_to_y": oob_hy,
        "raw_spellings": dict(raw_dims.most_common()),
        "canonical_parts": dict(canon.most_common()),
        "objects": len(by_obj),
        "multi_structure_objects": len(multi),
        "variants_identical_inventory": same_inv,
        "variants_differ_counts_only": diff_counts_only,
        "variants_differ_types": diff_types,
        "leaked_objects": len(leaked),
        "structures_with_collisions": collide,
        "bricks_per_structure": {
            "min": sizes[0], "p50": pct(.5), "p95": pct(.95), "max": sizes[-1]
        },
    }
    (OUT / "01_eda.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2)[:4000])


if __name__ == "__main__":
    main()
