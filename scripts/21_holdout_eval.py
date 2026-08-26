#!/usr/bin/env python3
"""Held-out evaluation of the two finished arms. Mac only, no training.

H1 and H2 each ran 2,000 rows on the node. This scores both of them on data
neither of them ever saw, and it does exactly that -- it loads no optimizer,
takes no gradient, writes no checkpoint and touches no GPU pack.

**The criterion is frozen in this file, above the code that produces a
number.** Mean masked loss over the 320 frozen held-out rows; lower wins;
exactly equal is a draw. Training loss, seconds per row, peak VRAM and adapter
size are recorded and are *descriptive only* -- a criterion chosen after
seeing them would be a rationalisation, and the whole reason H1 and H2 were
written down before either existed was to make that impossible.

**The held-out rows are the ones the project already froze.** 40 whole pairs
from ``instruct_inv_val.jsonl`` by seeded shuffle of sorted pair ids, seed 0 --
the same selection ``scripts/13_lora_smoke.py`` has used since the smoke test,
which is why it is 320 rows and not a number chosen today. The test split is
not opened, and this file contains no path that could open it.

**Both arms are scored identically.** Same tokenizer and revision, same base
and published-adapter revisions, same bf16, same max_length, same row order,
same masking, same batch size. The only thing that differs is which adapter is
cold-loaded on top -- which is the comparison.

Usage::

  ./.venv/bin/python scripts/21_holdout_eval.py --arms-root DIR --out FILE
  ./.venv/bin/python scripts/21_holdout_eval.py --criterion     # prints and stops
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.session import sha256_file  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen before any number from this evaluation exists.
# ---------------------------------------------------------------------------

#: The held-out selection, unchanged from the smoke test that established it.
VAL_FILE = "instruct_inv_val.jsonl"
N_VAL_PAIRS = 40
VAL_ROWS = 320
ARMS = ("H1", "H2")

PRIMARY_CRITERION = (
    "mean masked validation loss over the 320 frozen held-out rows; the lower "
    "value wins; exactly equal is a draw"
)

DESCRIPTIVE_ONLY = (
    "training loss", "seconds per row", "peak VRAM", "adapter bytes",
    "trainable parameter count",
)

CRITERION_NOTE = (
    "Frozen in this file before the evaluation was run, and stated over one "
    "number so there is nothing to select afterwards. The descriptive "
    "readings are recorded because they are worth knowing and are not the "
    "criterion: a comparison that got to choose between mean loss, speed and "
    "memory after seeing all three would be choosing its own answer."
)


def winner(means: dict) -> str:
    """Which arm wins, from the frozen criterion alone.

    Pure and total: two numbers in, one of ``H1``/``H2``/``draw`` out. No
    tolerance, no rounding, no threshold. Equality means a draw because the
    criterion says so, not because the difference looked small.
    """
    missing = [a for a in ARMS if not isinstance(means.get(a), (int, float))]
    if missing:
        raise ValueError(f"no mean validation loss for {missing}")
    a, b = means["H1"], means["H2"]
    if a == b:
        return "draw"
    return "H1" if a < b else "H2"


def selection_digest(sample_ids) -> str:
    """Which rows, in which order. The pair count alone does not pin either."""
    h = hashlib.sha256()
    for sample_id in sample_ids:
        h.update(str(sample_id).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def code_provenance() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None

    return {"head": git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(git("status", "--porcelain"))}


def criterion_document() -> dict:
    """The frozen criterion, for a report to embed verbatim."""
    return {
        "primary_criterion": PRIMARY_CRITERION,
        "descriptive_only": list(DESCRIPTIVE_ONLY),
        "note": CRITERION_NOTE,
        "held_out": {"file": VAL_FILE, "pairs": N_VAL_PAIRS,
                     "rows": VAL_ROWS, "selection":
                     "whole pairs by seeded shuffle of sorted pair ids, "
                     "seed 0; every row of a chosen pair is included"},
        "test_split_read": False,
        "arms": list(ARMS),
    }


# ---------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------

def adapter_dir(arms_root: Path, arm: str) -> Path:
    """Named from the arm, never discovered. No glob, no listing, no latest."""
    if arm not in ARMS:
        raise ValueError(f"{arm!r} is not one of {list(ARMS)}")
    return Path(arms_root) / arm / "adapter"


def arm_evidence(arms_root: Path, arm: str) -> dict:
    path = Path(arms_root) / arm / f"{arm}_evidence.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(cfg):
    from src.training.lora import read_rows, sample_pairs

    path = ROOT / "data" / "processed" / VAL_FILE
    rows = sample_pairs(read_rows(path), n_pairs=N_VAL_PAIRS, seed=cfg.seed)
    if len(rows) != VAL_ROWS:
        raise SystemExit(
            f"the frozen held-out selection produced {len(rows)} rows, not "
            f"{VAL_ROWS}. The data or the selection changed; refusing to "
            "score two arms on something other than what was frozen.")
    return rows, sha256_file(path)


def score(arm: str, *, arms_root: Path, rows, encs, tok, cfg, device: str):
    """One arm's per-row masked loss over the held-out rows.

    ``load_finetuned`` is the project's one correct cold start: it checks the
    manifest, the pinned revisions and the adapter digest before it builds
    anything, and it lands the delta on the *merged* BrickGPT rather than on
    bare Llama.
    """
    import torch

    from src.training.lora import collate, load_finetuned

    directory = adapter_dir(arms_root, arm)
    blob = directory / "adapter_model.safetensors"
    t0 = time.time()
    model, info = load_finetuned(directory, dtype=torch.bfloat16,
                                 device=device, verify_digest=True)
    load_seconds = time.time() - t0

    model.eval()
    losses, sample_ids = [], []
    started = time.time()
    with torch.no_grad():
        for i, enc in enumerate(encs):
            batch = collate([enc], tok.eos_token_id)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            losses.append(float(model(**batch).loss.detach().item()))
            sample_ids.append(rows[i].sample_id)
            if (i + 1) % 40 == 0:
                print(f"  {arm}: {i + 1}/{len(encs)} rows, running mean "
                      f"{sum(losses) / len(losses):.6f}", flush=True)
    seconds = time.time() - started

    del model
    return {
        "arm": arm,
        "adapter_dir_name": directory.parent.name,
        "adapter_sha256": sha256_file(blob),
        "adapter_bytes": blob.stat().st_size,
        "manifest": info.get("manifest", {}).get("lora"),
        "manifest_adapter_sha256": info.get("manifest", {}).get(
            "adapter_sha256"),
        "rows": len(losses),
        "sample_ids": sample_ids,
        "selection_digest": selection_digest(sample_ids),
        "per_row_loss": losses,
        "mean_val_loss": sum(losses) / len(losses),
        "model_load_seconds": round(load_seconds, 3),
        "eval_seconds": round(seconds, 3),
        "seconds_per_row": round(seconds / len(losses), 6),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Held-out evaluation of the two finished arms. Mac only.")
    ap.add_argument("--arms-root", metavar="DIR",
                    help="the directory holding the returned H1/ and H2/ runs")
    ap.add_argument("--out", metavar="FILE",
                    help="where to write the evaluation record (write-once)")
    ap.add_argument("--criterion", action="store_true",
                    help="print the frozen criterion and stop")
    ap.add_argument("--device", default="mps")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.criterion:
        print(json.dumps(criterion_document(), indent=2, ensure_ascii=False))
        return 0
    if not args.arms_root or not args.out:
        print("--arms-root and --out are both required", file=sys.stderr)
        return 2

    from src.generation.brickgpt import load_tokenizer
    from src.training.lora import LoraConfig_, encode_row

    arms_root = Path(args.arms_root)
    code = code_provenance()
    print(f"code: HEAD {code['head']} dirty={code['working_tree_dirty']}",
          flush=True)
    print(f"criterion: {PRIMARY_CRITERION}", flush=True)

    # The two frozen configurations, only to assert that everything the
    # evaluation depends on is shared. Nothing here is trained.
    from src.training import hypotheses

    configs = {arm: hypotheses.config_for(arm) for arm in ARMS}
    shared = {}
    for field in ("seed", "max_length", "batch_size", "dtype",
                  "target_modules", "quantization"):
        values = {arm: configs[arm].as_dict()[field] for arm in ARMS}
        if len(set(map(str, values.values()))) != 1:
            print(f"the two arms differ in {field}: {values}; they cannot be "
                  "scored on one held-out set", file=sys.stderr)
            return 2
        shared[field] = values["H1"]
    cfg = configs["H1"]

    rows, data_sha = load_rows(cfg)
    tok = load_tokenizer()
    encs = [encode_row(tok, r, cfg.max_length) for r in rows]
    truncated = sum(e.truncated for e in encs)
    print(f"held-out: {len(rows)} rows, {truncated} truncated, "
          f"data sha {data_sha[:16]}...", flush=True)

    results = {}
    for arm in ARMS:
        print(f"--- {arm} ---", flush=True)
        results[arm] = score(arm, arms_root=arms_root, rows=rows, encs=encs,
                             tok=tok, cfg=cfg, device=args.device)
        print(f"  {arm} mean validation loss: "
              f"{results[arm]['mean_val_loss']:.6f}", flush=True)

    if results["H1"]["sample_ids"] != results["H2"]["sample_ids"]:
        print("the two arms were scored on different rows; the comparison is "
              "void", file=sys.stderr)
        return 2

    means = {arm: results[arm]["mean_val_loss"] for arm in ARMS}
    decided = winner(means)

    import platform
    from importlib.metadata import version

    import torch

    from src.model_ids import (ADAPTER, ADAPTER_REVISION, BASE_MODEL,
                               BASE_REVISION, TOKENIZER_REVISION)

    record = {
        "kind": "holdout_eval",
        "criterion": criterion_document(),
        "code": code,
        "held_out": {
            "file": VAL_FILE, "sha256": data_sha, "rows": len(rows),
            "pairs": N_VAL_PAIRS, "truncated_rows": truncated,
            "selection_digest": results["H1"]["selection_digest"],
            "sample_ids_identical_across_arms": True,
            "test_split_read": False,
        },
        "shared_settings": shared,
        "provenance": {
            "device": args.device, "dtype": "bfloat16",
            "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
            "published_adapter": ADAPTER,
            "published_adapter_revision": ADAPTER_REVISION,
            "tokenizer_revision": TOKENIZER_REVISION,
            "packages": {"python": platform.python_version(),
                         "torch": torch.__version__,
                         "transformers": version("transformers"),
                         "peft": version("peft")},
            "platform": platform.platform(),
        },
        "arms": {arm: {**results[arm],
                       "config": configs[arm].as_dict()} for arm in ARMS},
        "means": means,
        "winner": decided,
        "descriptive_only": {
            arm: {
                "training_seconds_per_row":
                    arm_evidence(arms_root, arm).get("seconds_per_row"),
                "training_peak_vram_gb":
                    arm_evidence(arms_root, arm).get("peak_vram_gb"),
                "training_final_window_loss":
                    (arm_evidence(arms_root, arm).get("windows") or [{}])[-1]
                    .get("loss"),
                "adapter_bytes": results[arm]["adapter_bytes"],
                "trainable_digest":
                    arm_evidence(arms_root, arm).get("trainable_digest"),
            } for arm in ARMS},
        "note": ("This scores two finished arms. It does not tune, retrain, "
                 "select a next round, or start final training."),
    }

    from src.training.session import write_once_json

    write_once_json(Path(args.out), record)
    print(json.dumps({"means": means, "winner": decided,
                      "criterion": PRIMARY_CRITERION,
                      "record": str(Path(args.out).name)},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
