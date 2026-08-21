"""Render the counterfactual samples into instruction format.

Emits one JSONL per split with the prompt/target pair and its token counts, so
training never has to re-tokenise to know whether a row fits, and reports the
sequence-length distribution against candidate budgets.

Two arms are written from the same rows:

``inv``
    The conditioned prompt, carrying the inventory block. This is training
    data for the inventory-conditioned arms.

``noinv``
    The same rows with the block removed. This is **not** arm-A training data
    -- it is arm A's matched evaluation counterpart, so the unconditioned
    baseline is scored on exactly the same objects, captions and targets. With
    the block gone the four inventory framings of a target collapse to one
    prompt, so rows repeat. The duplicates are kept deliberately: each still
    carries the inventory that was hidden from the prompt, which is what a
    compliance metric has to be scored against. Report and evaluation must
    therefore weight by multiplicity, and the unique prompt-target count is
    reported alongside the row count.

Every row stores the structured inventory, so any row can be replayed back to
its build and its stock without consulting the counterfactual file.

Writes data/processed/instruct_{arm}_{split}.jsonl and
data/reports/10_instruction.md.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.counterfactual import read_jsonl  # noqa: E402
from src.data.instruction import Example, encode  # noqa: E402

SPLITS = ("train", "val", "test")
ARMS = {"inv": True, "noinv": False}      # with / without the inventory block
BUDGETS = (1024, 2048, 4096)

#: Rows longer than this are dropped -- by whole pair, never truncated.
MAX_CONTEXT = 2048

TOKENIZER = "AvaLovelace/BrickGPT"
#: Pinned so token counts and the prompt bytes cannot move under us.
TOKENIZER_REVISION = "19737def7bfe5950b2a466825ad7c6d74b7eafe3"
OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "data" / "reports"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def pctile(xs: list[int], q: float) -> int:
    return sorted(xs)[min(len(xs) - 1, int(q * len(xs)))]


def oversized_pairs(tok, split: str) -> tuple[set[str], dict]:
    """pair_ids with any row over budget, and what removing them costs.

    Dropped whole: both roles and all four framings go together, so the pairing
    structure and the arm alignment survive. Truncating a target instead would
    teach the model to stop mid-structure.

    Four different numbers describe one removal and they are easy to conflate,
    so each is counted separately rather than derived in prose:

    ``over_budget_trigger_rows``
        rows that actually measured over budget, per arm. These *trigger* the
        removal; they are not the extent of it.
    ``pairs_dropped``
        distinct pair_ids containing at least one such row.
    ``source_samples_removed``
        counterfactual samples in those pairs -- 2 roles x 4 framings each, so
        eight per pair however few rows tripped the check.
    ``instruction_rows_removed_total``
        the above written once per arm, which is what leaves the JSONL.
    """
    bad: set[str] = set()
    triggers: Counter[str] = Counter()
    samples = read_jsonl(OUT_DIR / f"counterfactual_{split}.jsonl")
    for s in samples:
        for arm, with_inv in ARMS.items():
            ex = Example.from_sample(s, with_inventory=with_inv)
            if len(encode(tok, ex)["input_ids"]) > MAX_CONTEXT:
                bad.add(s.pair_id)
                triggers[arm] += 1
    removed = [s for s in samples if s.pair_id in bad]
    counts = {
        "over_budget_trigger_rows": {a: triggers.get(a, 0) for a in ARMS},
        "pairs_dropped": len(bad),
        "source_samples_removed": len(removed),
        "instruction_rows_removed_total": len(removed) * len(ARMS),
        "removed_sample_ids": sorted(s.sample_id for s in removed),
    }
    return bad, counts


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    report: dict[str, dict] = {}
    prompt_tokens: dict[str, dict[str, int]] = {"inv": {}, "noinv": {}}
    t0 = time.time()

    dropped = {s: oversized_pairs(tok, s) for s in SPLITS}
    removals = {s: c for s, (_ids, c) in dropped.items()}
    for s, c in removals.items():
        print(f"  {s}: {sum(c['over_budget_trigger_rows'].values())} rows over "
              f"{MAX_CONTEXT} -> dropping {c['pairs_dropped']} pairs "
              f"= {c['source_samples_removed']} source samples "
              f"= {c['instruction_rows_removed_total']} instruction rows")

    for arm, with_inv in ARMS.items():
        for split in SPLITS:
            samples = [s for s in read_jsonl(OUT_DIR / f"counterfactual_{split}.jsonl")
                       if s.pair_id not in dropped[split][0]]
            rows = []
            totals: list[int] = []
            prompts: list[int] = []
            targets: list[int] = []
            for s in samples:
                ex = Example.from_sample(s, with_inventory=with_inv)
                enc = encode(tok, ex)
                n = len(enc["input_ids"])
                totals.append(n)
                prompts.append(enc["n_prompt_tokens"])
                targets.append(enc["n_target_tokens"])
                prompt_tokens[arm][ex.sample_id] = enc["n_prompt_tokens"]
                rows.append({
                    "sample_id": ex.sample_id,
                    "pair_id": ex.pair_id,
                    "object_id": ex.object_id,
                    "split": ex.split,
                    "role": ex.role,
                    "variant": ex.variant,
                    "dropped_part": ex.dropped_part,
                    "caption": s.caption,
                    # Kept on both arms: for noinv the prompt hides it, but a
                    # compliance metric still has to score against it.
                    "inventory": ex.inventory,
                    "used": s.used,
                    "prompt": ex.prompt,
                    "target": ex.target,
                    "n_prompt_tokens": enc["n_prompt_tokens"],
                    "n_target_tokens": enc["n_target_tokens"],
                    "n_tokens": n,
                })

            uniq = {(r["prompt"], r["target"]) for r in rows}
            groups: dict[tuple, list] = {}
            for r in rows:
                groups.setdefault((r["prompt"], r["target"]), []).append(r)
            mult = Counter(len(g) for g in groups.values())
            cross_object_dupes = sum(
                1 for g in groups.values()
                if len(g) > 1 and len({r["object_id"] for r in g}) > 1
            )

            path = OUT_DIR / f"instruct_{arm}_{split}.jsonl"
            path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                encoding="utf-8",
            )
            report[f"{arm}/{split}"] = {
                "rows": len(rows),
                "unique_prompt_target": len(uniq),
                "multiplicity": dict(sorted(mult.items())),
                "duplicate_groups_spanning_objects": cross_object_dupes,
                "objects": len({r["object_id"] for r in rows}),
                "prompt_median": pctile(prompts, .5),
                "target_median": pctile(targets, .5),
                "total_median": pctile(totals, .5),
                "total_p95": pctile(totals, .95),
                "total_max": max(totals),
                "fits": {str(b): sum(t <= b for t in totals) / len(totals)
                         for b in BUDGETS},
                "fits_count": {str(b): sum(t <= b for t in totals)
                               for b in BUDGETS},
                "over_budget": sum(t > MAX_CONTEXT for t in totals),
                "dropped_pairs": removals[split]["pairs_dropped"],
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            print(f"  {arm:6s} {split:6s} {len(rows):5d} rows  "
                  f"median {pctile(totals, .5):5d}  p95 {pctile(totals, .95):5d}")

    # Per-row paired difference. A difference of medians is not the median
    # difference, and here they disagree: the block's cost varies with how many
    # part lines an inventory has.
    deltas = sorted(
        prompt_tokens["inv"][sid] - prompt_tokens["noinv"][sid]
        for sid in prompt_tokens["inv"]
    )
    block_cost = {
        "n": len(deltas),
        "min": deltas[0],
        "median": deltas[len(deltas) // 2],
        "p95": deltas[min(len(deltas) - 1, int(0.95 * len(deltas)))],
        "max": deltas[-1],
    }

    L = ["# Instruction format", ""]
    L.append("One template for every arm; the inventory block is the only "
             "difference between the unconditioned and conditioned arms.")
    L.append("")
    L.append("`inv` is training data for the conditioned arms. `noinv` is "
             "**arm A's matched evaluation counterpart, not arm-A training "
             "data**: the same objects, captions and targets with the block "
             "removed, so the unconditioned baseline is scored on identical "
             "material. Its rows repeat because the four framings of a target "
             "share one unconditioned prompt; each duplicate still carries the "
             "inventory hidden from it, which compliance is scored against, so "
             "evaluation weights by multiplicity.")
    L.append("")
    L.append("Two documented deviations from the section 9.7 sketch, both to "
             "keep the arms comparable: the body is BrickGPT's own instruction "
             "rather than a fresh `### Request` preamble (it carries the "
             "allowed dimensions and one-unit-tall rule the checkpoint was "
             "trained against), and parts are named `1x1` rather than "
             "`brick_1x1`, reusing the model's existing size vocabulary. The "
             "spelling still need not match: the model may emit `4x1` against "
             "a listed `1x4`, and the canonical mapping draws both from the "
             "same quantity -- which is what the rotation rule states.")
    L += ["", f"- tokenizer: `{TOKENIZER}` @ `{TOKENIZER_REVISION}` (pinned)",
          f"- prompt tokens are masked out of the loss (`labels = -100`)",
          f"- target ends with EOS",
          f"- wall clock {time.time()-t0:.0f}s", ""]
    L += ["## Sequence lengths", "",
          "| arm / split | rows | prompt (median) | target (median) | total median | p95 | max |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for k, d in report.items():
        L.append(f"| `{k}` | {d['rows']} | {d['prompt_median']} | "
                 f"{d['target_median']} | {d['total_median']} | "
                 f"{d['total_p95']} | {d['total_max']} |")
    L += ["", "## Coverage by budget", "",
          f"Exact counts, not rounded rates. Pairs containing over-{MAX_CONTEXT} "
          "rows were removed before writing -- the whole pair, both roles and "
          "all four framings, not just the row that measured long. Targets are "
          "never truncated.", "",
          "| arm / split | " + " | ".join(
              f"<= {b}" for b in BUDGETS) + f" | over {MAX_CONTEXT} | pairs dropped |",
          "|---|" + "---|" * len(BUDGETS) + "---:|---:|"]
    for k, d in report.items():
        cells = " | ".join(
            f"{d['fits_count'][str(b)]}/{d['rows']} ({d['fits'][str(b)]:.2%})"
            for b in BUDGETS)
        L.append(f"| `{k}` | {cells} | **{d['over_budget']}** | "
                 f"{d['dropped_pairs']} |")
    L += ["", "### What was removed", "",
          "Four distinct counts, listed separately because they are easy to "
          "conflate. The trigger rows are what tripped the budget check; the "
          "pair is the unit actually removed, so the number of rows that leave "
          "the JSONL is several times larger than the number that measured "
          "long.", "",
          "| split | over-budget trigger rows | pairs dropped | source samples removed | instruction rows removed (both arms) |",
          "|---|---|---:|---:|---:|"]
    for s in SPLITS:
        c = removals[s]
        trig = ", ".join(f"{a} {n}" for a, n in c["over_budget_trigger_rows"].items())
        L.append(f"| `{s}` | {trig} | {c['pairs_dropped']} | "
                 f"{c['source_samples_removed']} | "
                 f"{c['instruction_rows_removed_total']} |")
    L += ["", "A pair is 2 roles x 4 inventory framings = 8 counterfactual "
          "samples, and each surviving sample is written once per arm.", ""]
    L += ["", "## Cost of the inventory block", "",
          "Per-row paired difference `inv - noinv` over all "
          f"{block_cost['n']} rows. A difference of medians would not answer "
          "this: the cost depends on how many part lines an inventory has.", "",
          "| min | median | p95 | max |", "|---:|---:|---:|---:|",
          f"| +{block_cost['min']} | +{block_cost['median']} | "
          f"+{block_cost['p95']} | +{block_cost['max']} |", ""]
    L += ["## Rows, unique prompts and multiplicity", "",
          "`inv` also shows a little repetition. That is not the framings "
          "collapsing: it is two `structure_id`s of the same object whose voxel "
          "occupancy is identical, so re-tiling returns the same target and the "
          "caption is the same object's. Every such group sits inside one "
          "object, so it cannot cross the split boundary -- the column below "
          "counts any group that does.", "",
          "`noinv` is arm A's matched evaluation counterpart, not arm-A "
          "training data. Removing the block collapses a target's four "
          "inventory framings to one prompt, so rows repeat; each duplicate "
          "still carries the inventory hidden from its prompt, which is what "
          "compliance is scored against. Evaluation weights by multiplicity.",
          "",
          "| arm / split | rows | unique prompt-target | multiplicity | dup groups spanning objects |",
          "|---|---:|---:|---|---:|"]
    for k, d in report.items():
        L.append(f"| `{k}` | {d['rows']} | {d['unique_prompt_target']} | "
                 f"{d['multiplicity']} | **{d['duplicate_groups_spanning_objects']}** |")
    L += ["", "## Files", "", "| file | sha256 |", "|---|---|"]
    for k, d in report.items():
        L.append(f"| `{d['path']}` | `{d['sha256']}` |")
    L.append("")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "10_instruction.md").write_text("\n".join(L), encoding="utf-8")
    (REPORT_DIR / "10_instruction.json").write_text(
        json.dumps({"report": report, "block_cost": block_cost,
                    "removals": removals}, indent=2),
        encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
