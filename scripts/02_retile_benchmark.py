"""Benchmark the CP-SAT re-tiling generator.

The plan assumed "20x20 solves instantly"; this measures it instead, and
breaks the result down by *which* part is dropped, because that turns out to
dominate feasibility far more than structure size does.

Writes data/reports/02_retile.md and .json.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402

from src.data.bricks import PART_VOCAB, parse_bricks, required_inventory  # noqa: E402
from src.data.retile import drop_part, occupancy_of, verify  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
TIME_LIMIT = 10.0
OUT = ROOT / "data" / "reports"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ds = load_dataset("AvaLovelace/StableText2Brick")
    rows = ds["train"].select_columns(["bricks"]).to_list()
    random.seed(0)
    samp = random.sample(rows, N)

    by_part: dict[str, list[dict]] = defaultdict(list)
    verify_failures = 0
    t0 = time.time()

    for i, r in enumerate(samp):
        bricks = parse_bricks(r["bricks"])
        occ = occupancy_of(bricks)
        used = required_inventory(bricks)
        for part in PART_VOCAB:
            if part not in used:
                continue          # dropping an unused part is a no-op
            t = time.time()
            res = drop_part(bricks, part, time_limit=TIME_LIMIT)
            wall = time.time() - t
            if res.ok:
                try:
                    verify(occ, res.bricks)
                except AssertionError:
                    verify_failures += 1
            by_part[part].append({
                "ok": res.ok,
                "status": res.status,
                "wall": wall,
                "n_in": len(bricks),
                "n_out": len(res.bricks or []),
                "cells": len(occ),
                "cand": res.candidates,
            })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{N}  ({time.time()-t0:.0f}s elapsed)", flush=True)

    lines = ["# CP-SAT re-tiling benchmark", ""]
    lines.append(f"- structures sampled: {N} (seed 0)")
    lines.append(f"- time limit per solve: {TIME_LIMIT}s")
    lines.append(f"- total solves: {sum(len(v) for v in by_part.values())}")
    lines.append(f"- wall clock: {time.time()-t0:.0f}s")
    lines.append(f"- exact-cover verification failures: **{verify_failures}**")
    lines.append("")
    lines.append("## Feasibility by dropped part")
    lines.append("")
    lines.append("| dropped | solves | feasible | median s | p95 s | max s | median bricks in->out |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    summary: dict[str, dict] = {}
    for part in PART_VOCAB:
        recs = by_part.get(part, [])
        if not recs:
            continue
        ok = [r for r in recs if r["ok"]]
        walls = sorted(r["wall"] for r in recs)
        def q(p: float) -> float:
            return walls[min(len(walls) - 1, int(p * len(walls)))]
        ratio = ""
        if ok:
            mi = sorted(r["n_in"] for r in ok)[len(ok) // 2]
            mo = sorted(r["n_out"] for r in ok)[len(ok) // 2]
            ratio = f"{mi} -> {mo}"
        lines.append(
            f"| `{part}` | {len(recs)} | **{len(ok)/len(recs):.0%}** | "
            f"{q(.5):.2f} | {q(.95):.2f} | {walls[-1]:.2f} | {ratio} |"
        )
        summary[part] = {
            "solves": len(recs),
            "feasible_rate": len(ok) / len(recs),
            "median_s": q(.5),
            "p95_s": q(.95),
            "max_s": walls[-1],
        }

    all_recs = [r for v in by_part.values() for r in v]
    statuses = Counter(r["status"] for r in all_recs)
    overall = sum(r["ok"] for r in all_recs) / len(all_recs)
    lines += ["", "## Overall", ""]
    lines.append(f"- feasible: **{overall:.1%}** of {len(all_recs)} solves")
    lines.append(f"- statuses: {dict(statuses)}")
    lines.append("")
    timeouts = [r for r in all_recs if r["status"] == "UNKNOWN"]
    lines.append(f"- hit the {TIME_LIMIT}s limit: **{len(timeouts)}**")
    lines.append("")

    (OUT / "02_retile.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "02_retile.json").write_text(
        json.dumps({
            "n_structures": N,
            "time_limit": TIME_LIMIT,
            "overall_feasible": overall,
            "verify_failures": verify_failures,
            "statuses": dict(statuses),
            "by_part": summary,
        }, indent=2),
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
