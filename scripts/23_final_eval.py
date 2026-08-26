#!/usr/bin/env python3
"""Does the final run beat the arm it was based on? Mac only, no training.

``final_H2`` trained the selected configuration on the whole 9,584-row split.
The arm it was selected from, ``H2``, trained the same configuration on the
frozen 2,000-row pool. Both are adapters on the same merged BrickGPT, so the
question "which one is the project model" has an answer, and this scores both
to find it.

**The criterion is frozen in this file, above the code that produces a
number.** Mean masked loss over the same 320 frozen held-out rows; the lower
value becomes the project model; **exactly equal keeps the incumbent**, which
is the existing H2. A tie is not a coin toss and not a draw: replacing a model
requires beating it.

**The pipeline is not asserted to be identical -- it is demonstrated.** Both
candidates are scored in one process, over the same rows, in the same order,
with the same mask, the same tokenizer and revisions and the same dtype. The
held-out selection, its constants and its row loader are *imported* from
``scripts/21_holdout_eval.py`` rather than restated, so they cannot drift. And
the existing H2 is re-scored here rather than read from the earlier record:
if this run does not reproduce the number that record holds, the pipeline has
changed and the comparison is void.

The test split is not opened, nothing is trained, and no GPU pack is built.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.session import sha256_file  # noqa: E402


def holdout():
    """The frozen held-out selection, imported rather than restated."""
    path = ROOT / "scripts" / "21_holdout_eval.py"
    spec = importlib.util.spec_from_file_location("holdout_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Frozen before any number from this evaluation exists.
# ---------------------------------------------------------------------------

INCUMBENT = "existing_H2"
CHALLENGER = "final_H2"
CANDIDATES = (INCUMBENT, CHALLENGER)

PRIMARY_CRITERION = (
    "mean masked validation loss over the same 320 frozen held-out rows; the "
    "lower value becomes the project model; exactly equal keeps the incumbent "
    f"({INCUMBENT})"
)

DESCRIPTIVE_ONLY = (
    "training rows", "training loss", "seconds per row", "peak VRAM",
    "adapter bytes", "optimizer steps",
)

CRITERION_NOTE = (
    "A tie keeps the incumbent because replacing a model should require "
    "beating it, not merely matching it. Stated here before either number "
    "existed, over one quantity, so there is nothing to select afterwards."
)


def project_model(means: dict) -> str:
    """Which adapter is the project model. Pure, total, no tolerance.

    Strictly lower wins. Equal and worse both keep the incumbent, which is
    what makes this a promotion rule rather than a preference.
    """
    missing = [c for c in CANDIDATES if not isinstance(means.get(c), (int, float))]
    if missing:
        raise ValueError(f"no mean validation loss for {missing}")
    return CHALLENGER if means[CHALLENGER] < means[INCUMBENT] else INCUMBENT


def criterion_document() -> dict:
    h = holdout()
    return {
        "primary_criterion": PRIMARY_CRITERION,
        "descriptive_only": list(DESCRIPTIVE_ONLY),
        "note": CRITERION_NOTE,
        "candidates": list(CANDIDATES),
        "incumbent": INCUMBENT,
        "held_out": {"file": h.VAL_FILE, "pairs": h.N_VAL_PAIRS,
                     "rows": h.VAL_ROWS,
                     "selection": "the same frozen selection scripts/21 used"},
        "test_split_read": False,
        "trains_anything": False,
    }


# ---------------------------------------------------------------------------

def score_adapter(name: str, directory: Path, *, rows, encs, tok, device: str):
    """One adapter's per-row masked loss over the held-out rows."""
    import torch

    from src.training.lora import collate, load_finetuned

    directory = Path(directory)
    blob = directory / "adapter_model.safetensors"
    manifest_path = directory / "brickagain_manifest.json"
    t0 = time.time()
    model, info = load_finetuned(directory, dtype=torch.bfloat16, device=device,
                                 verify_digest=True)
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
                print(f"  {name}: {i + 1}/{len(encs)} rows, running mean "
                      f"{sum(losses) / len(losses):.6f}", flush=True)
    seconds = time.time() - started
    del model

    return {
        "candidate": name,
        "adapter_sha256": sha256_file(blob),
        "adapter_bytes": blob.stat().st_size,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_lora": json.loads(
            manifest_path.read_text(encoding="utf-8")).get("lora"),
        "manifest_adapter_sha256": info.get("manifest", {}).get(
            "adapter_sha256"),
        "rows": len(losses),
        "sample_ids": sample_ids,
        "selection_digest": holdout().selection_digest(sample_ids),
        "per_row_loss": losses,
        "mean_val_loss": sum(losses) / len(losses),
        "model_load_seconds": round(load_seconds, 3),
        "eval_seconds": round(seconds, 3),
        "seconds_per_row": round(seconds / len(losses), 6),
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Score final_H2 against the existing H2. Mac only.")
    ap.add_argument("--existing-h2-adapter", metavar="DIR")
    ap.add_argument("--final-h2-adapter", metavar="DIR")
    ap.add_argument("--reference-record", metavar="FILE",
                    help="the earlier holdout_eval.json, whose recorded H2 "
                         "mean this run must reproduce")
    ap.add_argument("--out", metavar="FILE")
    ap.add_argument("--criterion", action="store_true")
    ap.add_argument("--device", default="mps")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.criterion:
        print(json.dumps(criterion_document(), indent=2, ensure_ascii=False))
        return 0
    required = (args.existing_h2_adapter, args.final_h2_adapter,
                args.reference_record, args.out)
    if not all(required):
        print("--existing-h2-adapter, --final-h2-adapter, --reference-record "
              "and --out are all required", file=sys.stderr)
        return 2

    h = holdout()
    from src.generation.brickgpt import load_tokenizer
    from src.training import hypotheses
    from src.training.lora import encode_row

    code = h.code_provenance()
    print(f"code: HEAD {code['head']} dirty={code['working_tree_dirty']}",
          flush=True)
    print(f"criterion: {PRIMARY_CRITERION}", flush=True)

    cfg = hypotheses.config_for("H2")
    rows, data_sha = h.load_rows(cfg)
    tok = load_tokenizer()
    encs = [encode_row(tok, r, cfg.max_length) for r in rows]
    truncated = sum(e.truncated for e in encs)
    print(f"held-out: {len(rows)} rows, {truncated} truncated, "
          f"data sha {data_sha[:16]}...", flush=True)

    directories = {INCUMBENT: Path(args.existing_h2_adapter),
                   CHALLENGER: Path(args.final_h2_adapter)}
    results = {}
    for name in CANDIDATES:
        print(f"--- {name} ---", flush=True)
        results[name] = score_adapter(name, directories[name], rows=rows,
                                      encs=encs, tok=tok, device=args.device)
        print(f"  {name} mean validation loss: "
              f"{results[name]['mean_val_loss']:.6f}", flush=True)

    if results[INCUMBENT]["sample_ids"] != results[CHALLENGER]["sample_ids"]:
        print("the two adapters were scored on different rows; the comparison "
              "is void", file=sys.stderr)
        return 2

    # The pipeline, demonstrated rather than asserted. The incumbent was
    # scored once before, by a different script, in a different process; if it
    # does not come back to the same number now, something in the path from
    # rows to loss has changed and this comparison means nothing.
    reference = json.loads(Path(args.reference_record).read_text(
        encoding="utf-8"))
    recorded = reference["arms"]["H2"]["mean_val_loss"]
    got = results[INCUMBENT]["mean_val_loss"]
    reproduced = got == recorded
    print(f"incumbent reproduces its recorded mean: {reproduced} "
          f"({got!r} vs {recorded!r})", flush=True)
    if not reproduced:
        print("refusing to decide: the incumbent did not reproduce the mean "
              "the earlier record holds, so the two numbers below were not "
              "produced by the same pipeline.", file=sys.stderr)
        return 2
    if results[INCUMBENT]["selection_digest"] != \
            reference["held_out"]["selection_digest"]:
        print("refusing to decide: this run scored a different set of rows "
              "from the earlier record.", file=sys.stderr)
        return 2

    means = {c: results[c]["mean_val_loss"] for c in CANDIDATES}
    decided = project_model(means)

    import platform
    from importlib.metadata import version

    import torch

    from src.model_ids import (ADAPTER, ADAPTER_REVISION, BASE_MODEL,
                               BASE_REVISION, TOKENIZER_REVISION)

    record = {
        "kind": "final_eval",
        "criterion": criterion_document(),
        "code": code,
        "held_out": {
            "file": h.VAL_FILE, "sha256": data_sha, "rows": len(rows),
            "pairs": h.N_VAL_PAIRS, "truncated_rows": truncated,
            "selection_digest": results[INCUMBENT]["selection_digest"],
            "sample_ids_identical_across_candidates": True,
            "same_selection_as_reference_record": True,
            "test_split_read": False,
        },
        "reference_record": {
            "path_name": Path(args.reference_record).name,
            "recorded_incumbent_mean": recorded,
            "reproduced_here": got,
            "identical": reproduced,
        },
        "shared_settings": {
            "config": cfg.as_dict(), "device": args.device,
            "dtype": "bfloat16", "max_length": cfg.max_length,
            "batch_size": cfg.batch_size, "seed": cfg.seed,
        },
        "provenance": {
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
        "candidates": results,
        "means": means,
        "project_model": decided,
        "note": ("This scores two finished adapters. It trains nothing, tunes "
                 "nothing, builds no pack and publishes nothing."),
    }

    from src.training.session import write_once_json

    write_once_json(Path(args.out), record)
    print(json.dumps({"means": means, "project_model": decided,
                      "criterion": PRIMARY_CRITERION,
                      "record": Path(args.out).name},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
