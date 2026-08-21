"""Cold-start the saved adapter the correct way, and prove it is the right way.

A locally trained adapter is just weights on disk. Nothing in the directory
says they were fitted on top of the *merged* BrickGPT rather than on bare
Llama, so the obvious load -- ``BrickGPT(adapter=<path>)`` -- silently produces
a different model that still loads and still emits bricks. This script loads
the real checkpoint through ``load_finetuned`` in a fresh process and checks:

1. the call order was base -> published adapter -> merge -> local adapter
2. the manifest's digest matches the adapter on disk
3. a forward pass over one real row produces a finite loss
4. the wrong path (bare base + our adapter) gives a *different* loss, so the
   distinction is real rather than nominal
5. ``BrickGPT(adapter=<local path>)`` refuses outright

One short forward pass, no generation, no training.

Writes data/reports/13_lora_coldstart.json; exits non-zero on any failure.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.brickgpt import load_tokenizer  # noqa: E402
from src.training.lora import (  # noqa: E402
    BASE_MODEL,
    BASE_REVISION,
    LOAD_ORDER,
    LoraConfig_,
    collate,
    encode_row,
    load_finetuned,
    read_rows,
    sample_pairs,
)

OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "data" / "reports"
CKPT_DIR = ROOT / "artifacts" / "checkpoints" / "lora_smoke"

#: Matches the smoke run. bf16 on the loading path too, so the loss here is
#: comparable with the training-time numbers rather than a different precision.
DTYPE = torch.bfloat16


def _portable_refusal_message(message: str) -> str:
    """Keep a useful checkpoint path without publishing the machine path."""
    return message.replace(str(CKPT_DIR), CKPT_DIR.relative_to(ROOT).as_posix())


def _sanitization_note() -> dict:
    """Say, in the record itself, that the stored message is not verbatim.

    The rewrite happens as the report is written rather than afterwards, so
    nothing stored is being edited -- but the field still is not the literal
    text the exception carried, and a report that does not say so invites
    being quoted as though it were.
    """
    return {
        "applied_after_the_run": False,
        "fields": ["refusal_message"],
        "refusal_message": {
            "verbatim": False,
            "reason": ("the exception embeds the absolute filesystem path of "
                       "the checkpoint on the machine that ran it, which is "
                       "personal information and must not be published"),
            "transform": ("the absolute checkpoint path is replaced by its "
                          "repository-relative form as the report is written; "
                          "no other part of the message is altered"),
            "original_retained": False,
            "original_recoverable_from_this_file": False,
        },
        "note": ("Rewritten at write time rather than after the fact, so no "
                 "stored record was edited -- but the field is still not the "
                 "exception's literal text and must not be quoted as one."),
    }


def main() -> int:
    cfg = LoraConfig_()
    out: dict[str, object] = {"checkpoint": str(CKPT_DIR.relative_to(ROOT))}
    failures: list[str] = []

    if not (CKPT_DIR / "adapter_model.safetensors").exists():
        print(f"no checkpoint at {CKPT_DIR}; run 13_lora_smoke.py first",
              file=sys.stderr)
        return 1

    from importlib.metadata import version

    out["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "peft": version("peft"),
        "dtype": str(DTYPE),
        "device": "cpu",
    }

    tok = load_tokenizer()
    row = sample_pairs(read_rows(OUT_DIR / "instruct_inv_val.jsonl"),
                       n_pairs=1, seed=cfg.seed)[0]
    enc = encode_row(tok, row, cfg.max_length)
    batch = collate([enc], tok.eos_token_id)
    out["probe_row"] = {
        "sample_id": row.sample_id,
        "pair_id": row.pair_id,
        "role": row.role,
        "variant": row.variant,
        "n_total_tokens": len(enc.input_ids),
        "n_prompt_tokens": enc.n_prompt_tokens,
        "n_supervised_tokens": len(enc.input_ids) - enc.n_prompt_tokens,
        "truncated": enc.truncated,
    }

    # ---- 1-2: correct order, verified digest ------------------------------
    calls: list = []
    model, info = load_finetuned(CKPT_DIR, device="cpu", dtype=DTYPE,
                                 _calls=calls)
    order = [c[0] for c in calls]
    out["load_order_observed"] = order
    out["load_order_expected"] = list(LOAD_ORDER)
    if tuple(order) != LOAD_ORDER:
        failures.append(f"load order was {order}, expected {list(LOAD_ORDER)}")
    out["base_revision_used"] = calls[0][2]
    if calls[0][2] != BASE_REVISION:
        failures.append(f"base revision {calls[0][2]} != pinned {BASE_REVISION}")
    out["manifest"] = info["manifest"]

    # ---- 3: a real forward pass -------------------------------------------
    with torch.no_grad():
        right = model(**{k: v.to("cpu") for k, v in batch.items()}).loss.item()
    out["loss_correct_path"] = round(right, 4)
    if not (right == right and abs(right) != float("inf")):
        failures.append(f"loss is not finite: {right}")
    del model

    # ---- 4: the wrong path really is a different model --------------------
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    bare = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, dtype=DTYPE)
    wrong = PeftModel.from_pretrained(bare, str(CKPT_DIR)).eval()
    with torch.no_grad():
        wrong_loss = wrong(**{k: v.to("cpu") for k, v in batch.items()}).loss.item()
    out["loss_wrong_path_bare_base_plus_our_adapter"] = round(wrong_loss, 4)
    out["loss_difference"] = round(wrong_loss - right, 4)
    if abs(wrong_loss - right) < 1e-3:
        failures.append(
            "loading onto the bare base gives the same loss; the merge step "
            "is not doing anything and the guard proves nothing")
    del wrong, bare

    # ---- 5: the unsafe constructor refuses --------------------------------
    from src.generation.brickgpt import _refuse_locally_trained_adapter

    try:
        _refuse_locally_trained_adapter(str(CKPT_DIR))
        out["brickgpt_refuses_local_adapter"] = False
        failures.append("BrickGPT would accept the local adapter path")
    except ValueError as e:
        out["brickgpt_refuses_local_adapter"] = True
        out["refusal_message"] = _portable_refusal_message(str(e))
        out["sanitization"] = _sanitization_note()

    out["failures"] = failures
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "13_lora_coldstart.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    for k, v in out.items():
        if k not in ("manifest", "refusal_message"):
            print(f"  {k}: {v}")
    print(f"\nfailures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
