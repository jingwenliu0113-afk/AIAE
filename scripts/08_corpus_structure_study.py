"""Connectivity and support in the corpus, and in what we generate from it.

Support and connectivity figures had been quoted without saying which set of
structures they described, and a 400-structure corpus figure was then compared
against a 60-structure re-tiling figure as though they covered the same shapes.

Three populations are reported separately, and only the two that *are* the same
shapes may be compared:

* **source 400**       -- every sampled corpus tiling
* **paired source 60** -- the first 60 of those, as tiled by the corpus
* **paired retile 60** -- CP-SAT re-tilings of those same 60 shapes

Sampling rule: sort every eligible row by structure_id (stable, independent of
dataset order), then take a seeded random sample. Eligible means <= MAX_BRICKS
bricks, matching the dataset generator.

Writes data/reports/08_corpus_structure.md and .json.
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

from src.data.bricks import (  # noqa: E402
    connected_components,
    is_connected,
    parse_bricks,
    unsupported_bricks,
)
from src.data.retile import occupancy_of, retile  # noqa: E402

SEED = 0
MAX_BRICKS = 150
N_CONNECTIVITY = 400      # cheap: parsing only
N_RETILE = 60             # expensive: two CP-SAT solves each
OUT = ROOT / "data" / "reports"


def sample_rows(n: int) -> list[dict]:
    ds = load_dataset("AvaLovelace/StableText2Brick")
    rows = [r for v in ds.values()
            for r in v.select_columns(["structure_id", "bricks"]).to_list()]
    rows = [r for r in rows if len(parse_bricks(r["bricks"])) <= MAX_BRICKS]
    rows.sort(key=lambda r: r["structure_id"])       # stable order
    return random.Random(SEED).sample(rows, n)


def pct(a: int, b: int) -> str:
    return f"{a}/{b} = {a/b:.1%}" if b else "n/a"


def main() -> None:
    t0 = time.time()
    rows = sample_rows(N_CONNECTIVITY)
    parsed = [parse_bricks(r["bricks"]) for r in rows]

    def summarise(bricks_list: list) -> dict:
        return {
            "n": len(bricks_list),
            "stud_connected": sum(is_connected(b) for b in bricks_list),
            "ground_connected": sum(is_connected(b, ground=True) for b in bricks_list),
            "with_unsupported": sum(1 for b in bricks_list if unsupported_bricks(b)),
            "total_unsupported": sum(len(unsupported_bricks(b)) for b in bricks_list),
            "total_bricks": sum(len(b) for b in bricks_list),
            "median_stud_components": sorted(
                len(connected_components(b)) for b in bricks_list
            )[len(bricks_list) // 2] if bricks_list else 0,
        }

    paired_src = parsed[:N_RETILE]
    tiled = []
    unsolved = 0
    for b in paired_src:
        r = retile(occupancy_of(b), seed=SEED)
        if r.ok:
            tiled.append(r.bricks)
        else:
            unsolved += 1

    pops = {
        "source 400": summarise(parsed),
        "paired source 60": summarise(paired_src),
        "paired retile 60": summarise(tiled) | {"unsolved": unsolved},
    }

    src, paired, retiled = (
        pops["source 400"], pops["paired source 60"], pops["paired retile 60"]
    )
    d_unsup = retiled["with_unsupported"] - paired["with_unsupported"]

    L = ["# Corpus structure study", ""]
    L.append(f"- seed {SEED}; eligible rows sorted by `structure_id`, then sampled")
    L.append(f"- eligibility: <= {MAX_BRICKS} bricks (same as the generator)")
    L.append(f"- `paired source` and `paired retile` are the **same {N_RETILE} "
             f"shapes**; `source 400` is the wider sample and is *not* "
             f"comparable to them one-to-one")
    L.append(f"- wall clock: {time.time()-t0:.0f}s")
    L += ["", "## All three populations", "",
          "| population | n | stud connected | stud + baseplate | >=1 unsupported |",
          "|---|---:|---|---|---|"]
    for name, d in pops.items():
        L.append(f"| `{name}` | {d['n']} | "
                 f"**{pct(d['stud_connected'], d['n'])}** | "
                 f"{pct(d['ground_connected'], d['n'])} | "
                 f"{pct(d['with_unsupported'], d['n'])} |")
    L += ["", "## Support, on the shapes that can be compared", "",
          f"Restricted to the {N_RETILE} paired shapes:", "",
          f"- corpus tiling: {pct(paired['with_unsupported'], paired['n'])}",
          f"- CP-SAT re-tiling: {pct(retiled['with_unsupported'], retiled['n'])}",
          f"- difference: **{d_unsup:+d} of {N_RETILE} structures**", "",
          "That is a small difference on a small sample. It is an observation "
          "about this sample, not evidence that re-tiling causes unsupported "
          "bricks; no causal claim is made either way. Support is reported "
          "rather than gated in the generator.", ""]
    L += ["## Connectivity", "",
          "Stud coupling alone is the project definition and the acceptance "
          "gate. The baseplate column is an anchoring metric only: it merges "
          "components that share no studs, so a model held together only by it "
          "would come apart when lifted.",
          "",
          f"On the paired {N_RETILE}, corpus tilings are "
          f"{pct(paired['stud_connected'], paired['n'])} single-component "
          f"against {pct(retiled['stud_connected'], retiled['n'])} for our "
          "re-tilings.",
          "",
          "The current re-tiling formulation is associated with that drop, but "
          "the cause is not isolated. Solver strategy alone does not account "
          "for it: per-layer reaches 38.3% and a joint solve 33.3% on the same "
          "shapes (scripts/09_stagger_ablation.py), so switching to a joint "
          "model does not recover the gap. Candidate factors -- the "
          "fewest-bricks objective, tie-breaking between equally optimal "
          "tilings, and per-layer independence -- have not been separated.",
          ""]
    L += ["## Raw counts", ""]
    for name, d in pops.items():
        L.append(f"- `{name}`: {json.dumps(d)}")
    L.append("")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "08_corpus_structure.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "08_corpus_structure.json").write_text(
        json.dumps({"seed": SEED, "max_bricks": MAX_BRICKS,
                    "populations": pops}, indent=2),
        encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
