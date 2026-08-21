"""Audit the instruction data: prompts, tokens and inventory alignment.

Recomputed from the written JSONL. Checks the properties training and
evaluation will silently rely on:

* every prompt is exactly what the shared builder produces for its stored
  caption and inventory -- so what a model is trained on and what it is
  prompted with at evaluation cannot diverge
* stripping the block from an ``inv`` prompt reproduces its ``noinv`` twin
  byte for byte, and the two arms line up one-to-one by sample_id
* stored token counts match a fresh tokenisation at the pinned revision
* the target fits its inventory under rotation equivalence, counting ``4x1``
  against the ``1x4`` line
* rotated spellings never appear as their own inventory line
* whatever is missing against the counterfactual file is exactly a set of whole
  pairs -- both roles, all four framings, both arms. A budget check that fires
  on one row removes the pair it belongs to, and a half-removed pair would
  leave a control without its counterfactual (or vice versa), quietly breaking
  the comparison the dataset exists to support.

Writes data/reports/11_instruction_audit.md; exits non-zero on any failure.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.bricks import PART_VOCAB, parse_bricks  # noqa: E402
from src.data.counterfactual import read_jsonl  # noqa: E402
from src.data.instruction import Example, encode  # noqa: E402
from src.generation.prompt import (  # noqa: E402
    INVENTORY_HEADER,
    build_prompt,
    strip_inventory_block,
)

SPLITS = ("train", "val", "test")
OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "data" / "reports"
TOKENIZER = "AvaLovelace/BrickGPT"
TOKENIZER_REVISION = "19737def7bfe5950b2a466825ad7c6d74b7eafe3"
#: Must match scripts/10_build_instruction_data.py; the audit re-derives the
#: budget decision rather than importing the builder's verdict.
MAX_CONTEXT = 2048


def load(arm: str, split: str) -> list[dict]:
    path = OUT_DIR / f"instruct_{arm}_{split}.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def listed_parts(prompt: str) -> dict[str, int]:
    if INVENTORY_HEADER not in prompt:
        return {}
    block = prompt.split(INVENTORY_HEADER, 1)[1].split("\n\n", 1)[0]
    out = {}
    for line in block.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = int(v)
    return out


def check_removals_are_whole_pairs(
    split: str, inv: dict, noinv: dict, failures: list[str], tok
) -> dict:
    """Anything absent against the counterfactual file must be a whole pair.

    Rebuilt from the source file rather than from the build report, so a wrong
    report cannot certify itself. That extends to *why* a pair went: the
    removed samples are re-tokenised here and at least one of them has to
    actually exceed the budget. Otherwise a pair could be dropped for any
    reason at all -- or by accident -- and this check would still pass it.
    """
    source = read_jsonl(OUT_DIR / f"counterfactual_{split}.jsonl")
    by_pair: dict[str, set[str]] = {}
    for s in source:
        by_pair.setdefault(s.pair_id, set()).add(s.sample_id)
    all_ids = {s.sample_id for s in source}
    roles = {s.sample_id: (s.role, s.variant) for s in source}
    by_id = {s.sample_id: s for s in source}

    #: Every condition that would make a removal unsound. The summary column
    #: is derived from this list, so a new check is covered by the reported
    #: boolean the moment it is added here -- rather than the column silently
    #: continuing to describe only partial-pair failures.
    marks = len(failures)

    for arm, rows in (("inv", inv), ("noinv", noinv)):
        extra = set(rows) - all_ids
        if extra:
            failures.append(
                f"{split}/{arm}: {len(extra)} rows are not in the counterfactual file")

    missing_inv = all_ids - set(inv)
    missing_noinv = all_ids - set(noinv)
    if missing_inv != missing_noinv:
        failures.append(
            f"{split}: arms removed different samples "
            f"({len(missing_inv ^ missing_noinv)} differ) -- removal must be "
            "identical across arms")

    missing = missing_inv | missing_noinv
    touched = {sid.rsplit(":", 2)[0] for sid in missing}
    triggers: dict[str, int] = {}
    for pair in sorted(touched):
        whole = by_pair.get(pair)
        if whole is None:
            failures.append(f"{split}: removed pair {pair} is not in the source")
            continue
        kept = whole - missing
        if kept:
            failures.append(
                f"{split}: pair {pair} only partly removed -- "
                f"{len(kept)} of {len(whole)} samples survive "
                f"({sorted(roles[s] for s in kept)})")

        # Re-derive the reason: some row of this pair must really be too long.
        over = 0
        for sid in sorted(whole):
            for with_inv in (True, False):
                ex = Example.from_sample(by_id[sid], with_inventory=with_inv)
                if len(encode(tok, ex)["input_ids"]) > MAX_CONTEXT:
                    over += 1
        triggers[pair] = over
        if over == 0:
            failures.append(
                f"{split}: pair {pair} was removed but no row of it exceeds "
                f"{MAX_CONTEXT} tokens -- removal has no measured cause")

    return {
        "source_samples": len(all_ids),
        "missing_per_arm": {"inv": len(missing_inv), "noinv": len(missing_noinv)},
        "pairs_touched_by_removal": len(touched),
        "instruction_rows_removed_total": len(missing_inv) + len(missing_noinv),
        "over_budget_rows_in_removed_pairs": triggers,
        "removal_sound": len(failures) == marks,
    }


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    failures: list[str] = []
    stats: dict[str, dict] = {}

    for split in SPLITS:
        inv = {r["sample_id"]: r for r in load("inv", split)}
        noinv = {r["sample_id"]: r for r in load("noinv", split)}

        if set(inv) != set(noinv):
            failures.append(f"{split}: arms do not align by sample_id")

        removal = check_removals_are_whole_pairs(split, inv, noinv, failures, tok)

        rotated_seen = 0
        checked_tokens = 0
        for sid, r in inv.items():
            # 1. prompt is exactly what the builder produces
            if r["prompt"] != build_prompt(r["caption"], r["inventory"]):
                failures.append(f"{sid}: prompt is not the builder's output")
            # 2. stripping the block reproduces the noinv twin
            twin = noinv.get(sid)
            if twin is None:
                failures.append(f"{sid}: missing from noinv")
            else:
                if strip_inventory_block(r["prompt"]) != twin["prompt"]:
                    failures.append(f"{sid}: stripped prompt != noinv prompt")
                if twin["prompt"] != build_prompt(r["caption"]):
                    failures.append(f"{sid}: noinv prompt is not the arm-A form")
                if twin["target"] != r["target"]:
                    failures.append(f"{sid}: targets differ across arms")
                if twin["inventory"] != r["inventory"]:
                    failures.append(f"{sid}: noinv lost the hidden inventory")
                if INVENTORY_HEADER in twin["prompt"]:
                    failures.append(f"{sid}: noinv still carries the block")

            # 3. listed lines match the structured inventory, canonical only
            listed = listed_parts(r["prompt"])
            expected = {k: v for k, v in r["inventory"].items() if v > 0}
            if listed != expected:
                failures.append(f"{sid}: listed block != structured inventory")
            for p in listed:
                if p not in PART_VOCAB:
                    failures.append(f"{sid}: non-canonical part listed: {p}")

            # 4. target fits the inventory under rotation equivalence
            bricks = parse_bricks(r["target"].strip(), strict=False)
            used = Counter(b.part for b in bricks)
            if dict(used) != r["used"]:
                failures.append(f"{sid}: used field disagrees with target")
            for p, k in used.items():
                if r["inventory"].get(p, 0) < k:
                    failures.append(f"{sid}: target overdraws {p}")
            if any(b.rotated for b in bricks):
                rotated_seen += 1

            # 5. stored token counts reproduce -- every row, both arms
            checked_tokens += 1
            for arm_row in (r, twin):
                if arm_row is None:
                    continue
                ex = type("E", (), {
                    "prompt": arm_row["prompt"], "target": arm_row["target"]})()
                enc = encode(tok, ex)
                if enc["n_prompt_tokens"] != arm_row["n_prompt_tokens"]:
                    failures.append(f"{sid}: prompt token count drifted")
                if enc["n_target_tokens"] != arm_row["n_target_tokens"]:
                    failures.append(f"{sid}: target token count drifted")
                if len(enc["input_ids"]) != arm_row["n_tokens"]:
                    failures.append(f"{sid}: total token count drifted")
                if enc["input_ids"][-1] != tok.eos_token_id:
                    failures.append(f"{sid}: target does not end with EOS")
                n = enc["n_prompt_tokens"]
                if enc["labels"][:n] != [-100] * n:
                    failures.append(f"{sid}: prompt not masked out of loss")

        stats[split] = {
            "rows": len(inv),
            "targets_using_a_rotated_spelling": rotated_seen,
            "rows_retokenised": checked_tokens,
            "removal": removal,
        }

    L = ["# Instruction data audit", ""]
    L.append(f"Recomputed from the JSONL at tokenizer `{TOKENIZER}` @ "
             f"`{TOKENIZER_REVISION}`. **Every check covers every row of both "
             "arms** -- token counts included, re-derived rather than sampled.")
    L += ["", "| split | rows per arm | rows re-tokenised (both arms) | "
          "targets using a rotated spelling |",
          "|---|---:|---:|---:|"]
    total = 0
    for s, d in stats.items():
        total += d["rows_retokenised"] * 2
        L.append(f"| {s} | {d['rows']} | {d['rows_retokenised'] * 2} | "
                 f"{d['targets_using_a_rotated_spelling']} |")
    L.append(f"| **all** | | **{total}** | |")
    L += ["", "Rotated targets are the point of the rotation rule: their parts "
          "are counted against the canonical line (`4x1` against `1x4`), and "
          "the check above confirms none of them overdraws.", ""]

    L += ["## Removals against the counterfactual file", "",
          "Rebuilt from `counterfactual_{split}.jsonl`, not from the build "
          "report. Every sample absent from the instruction data must belong "
          "to a pair that is absent *entirely* -- both roles, all four "
          "framings, in both arms. A pair left half-removed would strand a "
          "control without its counterfactual.", "",
          "| split | source samples | missing (inv) | missing (noinv) | pairs touched | over-budget rows in them | removal sound |",
          "|---|---:|---:|---:|---:|---|---|"]
    for s, d in stats.items():
        r = d["removal"]
        trig = r["over_budget_rows_in_removed_pairs"]
        L.append(f"| {s} | {r['source_samples']} | {r['missing_per_arm']['inv']} | "
                 f"{r['missing_per_arm']['noinv']} | "
                 f"{r['pairs_touched_by_removal']} | "
                 f"{', '.join(str(v) for v in trig.values()) or '-'} | "
                 f"{'**yes**' if r['removal_sound'] else '**NO**'} |")
    L += ["", "`removal sound` covers every removal condition together: whole "
          "pairs only, identical across arms, no row invented, and at least "
          "one genuinely over-budget row per removed pair (re-tokenised here, "
          "not read from the build report). Any one of them failing turns the "
          "column.", ""]

    L += ["## Result", "", f"- checks failed: **{len(failures)}**"]
    for f in failures[:20]:
        L.append(f"  - {f}")
    L.append("")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "11_instruction_audit.md").write_text("\n".join(L), encoding="utf-8")
    (REPORT_DIR / "11_instruction_audit.json").write_text(
        json.dumps({"stats": stats, "failures": failures}, indent=2), encoding="utf-8")
    print("\n".join(L))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
