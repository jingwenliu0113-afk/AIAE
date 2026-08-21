"""Operational benchmark: is stagger worth running at our time budget?

This is **not** a clean measurement of what the stagger constraint does to
connectivity. Every condition gets the same 20s per shape, and the staggered
model is far harder, so it exhausts that budget on shapes the others finish.
What is being compared is therefore end-to-end behaviour under a fixed budget,
which is the question that decides whether it goes in the pipeline -- not the
intrinsic effect of the constraint.

Reading the numbers:

* Shapes the staggered condition does not finish come back ``UNKNOWN``: the
  solver ran out of time, which says nothing about whether a connected tiling
  exists. They are **not** disconnected results.
* The "of attempted" figure is a solved-and-connected yield, mixing solver
  success with connectivity. It is not a connectivity rate.
* Mean brick count is computed over each condition's own solved subset, and
  those subsets differ. It cannot support a claim that stagger doubles brick
  count; the harder shapes are simply missing from one column.

An earlier note quoted 48.3% against 23.3% here. That comparison was worse
still -- unstaggered from the per-layer solve, staggered from the joint solve
-- and is withdrawn.

Conditions, same shapes and seed throughout:

* ``per-layer``     -- the production path, layers solved independently
* ``joint``         -- one model over the whole structure, no stagger
* ``joint+stagger`` -- identical, plus the constraint forbidding a brick from
                       sitting on an identical footprint in the layer below

Sampling rule matches scripts/08: eligible rows (<= MAX_BRICKS bricks) sorted
by structure_id, seeded sample, first N taken.

Writes data/reports/09_stagger_ablation.md and .json.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402

from src.data.bricks import connected_components, is_connected, parse_bricks  # noqa: E402
from src.data.retile import occupancy_of, retile  # noqa: E402

SEED = 0
MAX_BRICKS = 150
N = 60
TIME_LIMIT = 20.0
OUT = ROOT / "data" / "reports"


def sample_rows(n: int) -> list[dict]:
    ds = load_dataset("AvaLovelace/StableText2Brick")
    rows = [r for v in ds.values()
            for r in v.select_columns(["structure_id", "bricks"]).to_list()]
    rows = [r for r in rows if len(parse_bricks(r["bricks"])) <= MAX_BRICKS]
    rows.sort(key=lambda r: r["structure_id"])
    return random.Random(SEED).sample(rows, 400)[:n]


CONDITIONS = {
    "per-layer": dict(),
    "joint": dict(budget={}),
    "joint+stagger": dict(budget={}, stagger=True),
}


def load_previous() -> dict:
    """Reuse a previous run's measurements instead of re-solving.

    The staggered condition takes ~1000s, so the write-up can be rebuilt
    without paying that again. Fields added after the run are derived from
    what was stored.
    """
    results = json.loads((OUT / "09_stagger_ablation.json").read_text())["results"]
    for d in results.values():
        d.setdefault(
            "rate_over_attempted",
            d["stud_connected"] / d["attempted"] if d["attempted"] else 0.0,
        )
    return results


def measure() -> dict:
    rows = sample_rows(N)
    results: dict[str, dict] = {}
    for name, kw in CONDITIONS.items():
        solved = conn = bricks_total = 0
        t = time.time()
        for r in rows:
            occ = occupancy_of(parse_bricks(r["bricks"]))
            res = retile(occ, seed=SEED, time_limit=TIME_LIMIT, **kw)
            if not res.ok:
                continue
            solved += 1
            bricks_total += len(res.bricks)
            conn += is_connected(res.bricks)
        secs = time.time() - t
        results[name] = {
            "attempted": len(rows),
            "solved": solved,
            "stud_connected": conn,
            "rate": conn / solved if solved else 0.0,
            "rate_over_attempted": conn / len(rows) if rows else 0.0,
            "mean_bricks": bricks_total / solved if solved else 0.0,
            "seconds": round(secs, 1),
        }
        print(f"  {name:14s} solved {solved}/{len(rows)}  "
              f"connected {conn}/{solved}  ({secs:.0f}s)")
    return results


def main() -> None:
    reuse = "--report-only" in sys.argv
    t0 = time.time()
    results = load_previous() if reuse else measure()

    fair = results["joint"], results["joint+stagger"]
    delta = fair[1]["rate"] - fair[0]["rate"]
    delta_attempted = fair[1]["rate_over_attempted"] - fair[0]["rate_over_attempted"]

    L = ["# Stagger ablation", ""]
    L.append(f"- seed {SEED}, {N} shapes, eligibility <= {MAX_BRICKS} bricks")
    L.append(f"- sampling rule identical to `scripts/08_corpus_structure_study.py`")
    L.append(f"- solver time limit {TIME_LIMIT}s, single worker")
    L.append("- measurements reused from a previous run"
             if reuse else f"- wall clock {time.time()-t0:.0f}s")
    L += ["", "| condition | solved | stud-connected | rate | mean bricks | seconds |",
          "|---|---:|---:|---:|---:|---:|"]
    for name, d in results.items():
        L.append(f"| `{name}` | {d['solved']}/{d['attempted']} | "
                 f"{d['stud_connected']} | {d['rate']:.1%} | "
                 f"{d['mean_bricks']:.1f} | {d['seconds']} |")
    unsolved = fair[0]["solved"] - fair[1]["solved"]
    L += ["", "## What these numbers do and do not say", "",
          f"`joint+stagger` left **{unsolved} of {N}** shapes unfinished at the "
          f"{TIME_LIMIT:.0f}s budget. Those come back `UNKNOWN` -- the solver "
          "ran out of time. They are **not** disconnected results, and nothing "
          "here shows a connected tiling does not exist for them.", "",
          "Two rates, neither of which is a connectivity rate on its own:", "",
          f"- over each condition's own solved subset: `joint` "
          f"{fair[0]['rate']:.1%}, `joint+stagger` {fair[1]['rate']:.1%} "
          f"({delta:+.1%}) -- different denominators",
          f"- solved **and** connected, over all {N} attempted: `joint` "
          f"{fair[0]['rate_over_attempted']:.1%}, `joint+stagger` "
          f"{fair[1]['rate_over_attempted']:.1%} ({delta_attempted:+.1%}) -- "
          "an end-to-end yield that mixes solver success with connectivity",
          "",
          f"Mean brick count ({fair[0]['mean_bricks']:.0f} vs "
          f"{fair[1]['mean_bricks']:.0f}) is computed over those same "
          "different solved subsets, with an unknown status mix. It does not "
          "support a claim that stagger inflates brick count; the shapes it "
          "could not finish are simply absent from its column.", ""]
    L += ["## Engineering conclusion", "",
          f"At a {TIME_LIMIT:.0f}s budget the staggered formulation solves "
          f"fewer shapes ({fair[1]['solved']}/{N} against "
          f"{fair[0]['solved']}/{N}) and costs "
          f"{fair[1]['seconds']/max(fair[0]['seconds'], 1e-9):.0f}x the wall "
          "clock for the same sample. That is sufficient to keep it out of the "
          "production path, and it is the only conclusion drawn here.", "",
          "Whether the constraint helps or hurts connectivity when given "
          "enough time is not answered by this benchmark. Settling it would "
          "need a budget large enough for both conditions to solve every "
          f"shape. Sample of {N} shapes.", ""]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "09_stagger_ablation.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "09_stagger_ablation.json").write_text(
        json.dumps({"seed": SEED, "n": N, "results": results}, indent=2),
        encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
