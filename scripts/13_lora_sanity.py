"""Pre-flight checks for the LoRA smoke test. Seconds, not minutes.

Every one of these can fail silently in a run that otherwise looks healthy:
loss falls, the adapter saves, and the result is meaningless. So they are
checked before any real training, on a handful of rows, and the answers are
written to the report:

1. prompt tokens are entirely masked out of the loss (-100)
2. the target *and* its EOS are the only supervised positions
3. no target is truncated by the length budget
4. only the expected LoRA tensors are trainable, and their gradients are
   non-zero after one backward pass
5. the adapter can be saved and reloaded to the same weights

Writes data/reports/13_lora_sanity.json; exits non-zero on any failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.brickgpt import load_tokenizer  # noqa: E402
from src.training.lora import (  # noqa: E402
    LoraConfig_,
    assert_only_lora_trainable,
    build_model,
    collate,
    encode_row,
    read_rows,
    sample_pairs,
)

OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "data" / "reports"
CKPT_DIR = ROOT / "artifacts" / "checkpoints"
N_ROWS = 8


def main() -> int:
    cfg = LoraConfig_()
    tok = load_tokenizer()
    rows = sample_pairs(read_rows(OUT_DIR / "instruct_inv_train.jsonl"),
                        n_pairs=1, seed=cfg.seed)[:N_ROWS]
    checks: dict[str, object] = {}
    failures: list[str] = []

    # ---- 1-3: masking and truncation, on real rows ------------------------
    encs = [encode_row(tok, r, cfg.max_length) for r in rows]
    eos = tok.eos_token_id

    prompt_masked = all(
        e.labels[:e.n_prompt_tokens] == [-100] * e.n_prompt_tokens for e in encs)
    supervised_is_target = all(
        e.labels[e.n_prompt_tokens:] == e.input_ids[e.n_prompt_tokens:]
        for e in encs)
    ends_with_eos = all(e.input_ids[-1] == eos and e.labels[-1] == eos
                        for e in encs)
    none_truncated = not any(e.truncated for e in encs)
    # A -100 must never appear after the prompt, or supervision would have a
    # hole in the middle of the target.
    no_holes = all(-100 not in e.labels[e.n_prompt_tokens:] for e in encs)

    checks["prompt_fully_masked"] = prompt_masked
    checks["supervised_positions_are_the_target"] = supervised_is_target
    checks["target_ends_with_eos_and_is_supervised"] = ends_with_eos
    checks["no_target_truncated"] = none_truncated
    checks["no_masked_hole_inside_the_target"] = no_holes
    checks["supervised_token_counts"] = [
        len(e.labels) - e.n_prompt_tokens for e in encs]
    checks["total_token_counts"] = [len(e.input_ids) for e in encs]

    for name, ok in (("prompt_fully_masked", prompt_masked),
                     ("supervised_positions_are_the_target", supervised_is_target),
                     ("target_ends_with_eos_and_is_supervised", ends_with_eos),
                     ("no_target_truncated", none_truncated),
                     ("no_masked_hole_inside_the_target", no_holes)):
        if not ok:
            failures.append(name)

    # ---- 4: trainability and real gradients -------------------------------
    model, info = build_model(cfg)
    checks["model"] = info
    try:
        trainable = assert_only_lora_trainable(model)
        checks["only_lora_trainable"] = True
    except RuntimeError as e:
        checks["only_lora_trainable"] = False
        failures.append(f"trainable set: {e}")
        trainable = []

    batch = collate(encs[:2], pad_id=eos)
    batch = {k: v.to(model.device) for k, v in batch.items()}
    model.train()
    out = model(**batch)
    out.loss.backward()

    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    have_grad = {n: (g is not None and float(g.abs().sum()) > 0)
                 for n, g in grads.items()}
    n_nonzero = sum(have_grad.values())
    checks["initial_loss"] = round(out.loss.detach().item(), 4)
    checks["trainable_tensors"] = len(trainable)
    checks["tensors_with_nonzero_grad"] = n_nonzero
    # lora_B initialises to zero, so its gradient is the one that proves the
    # path is live; lora_A can legitimately be zero on the first step.
    b_live = sum(1 for n, ok in have_grad.items() if "lora_B" in n and ok)
    checks["lora_B_tensors_with_nonzero_grad"] = b_live
    if n_nonzero == 0:
        failures.append("no LoRA parameter received a gradient")
    if b_live == 0:
        failures.append("no lora_B gradient: the adapter is not in the graph")
    if not torch.isfinite(out.loss):
        failures.append(f"loss is not finite: {out.loss}")

    # ---- 5: save and reload round-trip ------------------------------------
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    probe = CKPT_DIR / "sanity_probe"
    model.save_pretrained(probe)
    saved = sorted(p.name for p in probe.iterdir())
    checks["saved_files"] = saved

    # A real round trip: load the file back as a second adapter on the same
    # model and compare tensor for tensor. Comparing the safetensors keys to
    # parameter names by hand only tests the string surgery -- peft writes
    # `...lora_A.weight` but names the live tensor `...lora_A.default.weight`,
    # so a near-miss there reads as "no tensors matched" whatever the weights.
    model.load_adapter(str(probe), adapter_name="reloaded")
    params = dict(model.named_parameters())
    matched, mismatched = 0, []
    for name, p in params.items():
        if ".default." not in name or "lora_" not in name:
            continue
        twin = params.get(name.replace(".default.", ".reloaded."))
        if twin is None:
            mismatched.append(name)
            continue
        if torch.allclose(p.detach().float().cpu(), twin.detach().float().cpu()):
            matched += 1
        else:
            mismatched.append(name)

    checks["reload_method"] = "save_pretrained -> load_adapter, tensor compare"
    checks["reloaded_tensors_matching"] = matched
    checks["reloaded_tensors_mismatched"] = len(mismatched)
    checks["adapter_files"] = saved
    checks["adapter_ships_a_tokenizer"] = any("tokenizer" in f for f in saved)
    if matched == 0:
        failures.append("saved adapter has no tensor matching the live model")
    if mismatched:
        failures.append(f"{len(mismatched)} reloaded tensors differ from the model")

    checks["failures"] = failures
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "13_lora_sanity.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8")

    for k, v in checks.items():
        if k not in ("model", "supervised_token_counts", "total_token_counts"):
            print(f"  {k}: {v}")
    print(f"\nfailures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
