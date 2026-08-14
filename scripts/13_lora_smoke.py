"""LoRA smoke test: 2,000 rows, one pre-declared configuration.

**This is a plumbing test, not an experiment.** It answers "does a
reproducible fine-tune run end to end on this machine, from the right starting
point, with the loss mask doing what it claims" -- and nothing else. It does
not compare hyperparameters, does not select a checkpoint, never reads the
test split, and its generations are recorded as smoke observations rather than
as evidence about inventory compliance or generalisation.

One configuration is declared in src/training/lora.py and used as-is. The
2e-3-versus-lower-LR comparison the project owes itself is a separate round;
running both here and keeping the better one would turn a smoke test into a
selection with n=1.

Writes data/reports/13_lora_smoke.md/.json and an adapter under
artifacts/checkpoints/ (gitignored). Reads the instruction JSONL; never
rewrites it.
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.brickgpt import load_tokenizer  # noqa: E402
from src.training.lora import (  # noqa: E402
    BASE_MODEL,
    MANIFEST_NAME,
    LoraConfig_,
    sha256_file,
    write_manifest,
    assert_only_lora_trainable,
    build_model,
    check_no_object_overlap,
    collate,
    encode_row,
    read_rows,
    sample_pairs,
    split_stats,
)

OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "data" / "reports"
CKPT_DIR = ROOT / "artifacts" / "checkpoints" / "lora_smoke"

N_PAIRS = 250                 # 250 x 8 rows = 2,000
N_VAL_PAIRS = 40              # 320 rows, enough for a stable val loss
N_GEN = 4                     # fixed val prompts for before/after generation
GEN_MAX_NEW = 400


def marginal_rates(history: list[dict]) -> list[dict]:
    """Per-window seconds/row, from the cumulative elapsed times.

    The mean and median over the whole run describe a distribution; they do
    not show that the distribution moved. This does.
    """
    out, prev_row, prev_t = [], 0, 0.0
    for h in history:
        if "elapsed_seconds" not in h:
            continue
        rows = h["row"] - prev_row
        secs = h["elapsed_seconds"] - prev_t
        out.append({
            "row": h["row"],
            "seconds_per_row": round(secs / rows, 2) if rows else None,
            "train_loss": h["train_loss"],
        })
        prev_row, prev_t = h["row"], h["elapsed_seconds"]
    return out


def degradation(rates: list[dict], window: int = 4) -> dict:
    """How much slower the end of the run is than the beginning."""
    if len(rates) < 2 * window:
        return {}
    first = sum(r["seconds_per_row"] for r in rates[:window]) / window
    last = sum(r["seconds_per_row"] for r in rates[-window:]) / window
    return {
        "first_windows_seconds_per_row": round(first, 2),
        "last_windows_seconds_per_row": round(last, 2),
        "slowdown_factor": round(last / first, 1) if first else None,
        "worst_window_seconds_per_row": max(
            r["seconds_per_row"] for r in rates),
    }


#: Files whose contents decide what the run actually did. Digested up front so
#: the record survives edits made while a multi-hour run is in flight.
CODE_FILES = (
    "scripts/13_lora_smoke.py",
    "src/training/lora.py",
    "src/data/instruction.py",
    "src/generation/prompt.py",
    "src/model_ids.py",
)


def code_provenance() -> dict:
    """Capture the source state **before** anything loads or trains.

    Calling ``git rev-parse HEAD`` after a five-hour run records the commit
    that existed when the run *finished*, which is not the code that ran if
    anything was edited or committed meanwhile -- and on a long run something
    usually is. HEAD alone is also not enough: a dirty tree means HEAD does not
    describe the files on disk, so the working state is digested directly.
    """
    import subprocess

    def git(*args) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(ROOT), *args],
                                 capture_output=True, text=True, timeout=10)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    status = git("status", "--porcelain")
    return {
        "captured": "before model load and training",
        "head": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(status) if status is not None else None,
        "dirty_paths": sorted(
            line[3:] for line in (status or "").splitlines())[:20] or [],
        "file_sha256": {
            f: sha256_file(ROOT / f) for f in CODE_FILES
            if (ROOT / f).exists()
        },
    }


def training_order_digest(order: list[int], rows) -> str:
    """Digest over the order rows were actually fed to the model.

    Distinct from the selection digest, which covers *which* rows were chosen
    and their fixed listing order. The shuffle that decides the training order
    comes from ``torch.randperm``, so two runs can select identical rows and
    still present them differently -- a different curriculum from the same
    data. Recorded directly rather than left to be re-derived from the seed,
    because re-deriving it depends on the torch version's RNG behaviour.
    """
    import hashlib

    h = hashlib.sha256()
    for idx in order:
        h.update(rows[idx].sample_id.encode())
        h.update(b"\n")
    return h.hexdigest()


def run_provenance(manifest: dict) -> dict:
    """Everything needed to say which inputs and weights this run used.

    Digests over the data files and the saved adapter, the pinned revisions,
    and a digest over the selected sample order -- so a later reader can tell
    whether a report describes the checkpoint and data now on disk, rather
    than trusting that nothing moved.
    """
    files = {
        name: sha256_file(OUT_DIR / name)
        for name in ("instruct_inv_train.jsonl", "instruct_inv_val.jsonl")
    }
    return {
        "instruction_sha256": files,
        "base_model": manifest["base_model"],
        "base_revision": manifest["base_revision"],
        "published_adapter": manifest["published_adapter"],
        "published_adapter_revision": manifest["published_adapter_revision"],
        "tokenizer": manifest["tokenizer"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "adapter_sha256": manifest["adapter_sha256"],
        "manifest_sha256": sha256_file(CKPT_DIR / MANIFEST_NAME),
    }


def selection_digest(rows) -> str:
    """Digest over the chosen sample ids, in order.

    Pins *which* rows were trained on and in what order they were laid out,
    which the pair count alone does not.
    """
    import hashlib

    h = hashlib.sha256()
    for r in rows:
        h.update(r.sample_id.encode())
        h.update(b"\n")
    return h.hexdigest()


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def mps_gb() -> float:
    try:
        return torch.mps.current_allocated_memory() / (1024 ** 3)
    except Exception:
        return 0.0


def benchmark_workers(rows, tok, cfg) -> dict:
    """Decide DataLoader workers by measuring, not by guessing.

    Rows are encoded once up front and held in memory, so a worker process has
    almost nothing to do; raising the count would mostly buy process overhead.
    Measured rather than asserted, because that reasoning is only true while
    encoding stays precomputed.
    """
    from functools import partial

    from torch.utils.data import DataLoader

    encs = [encode_row(tok, r, cfg.max_length) for r in rows[:64]]
    # partial over the module-level collate, not a lambda: worker processes
    # pickle the collate_fn, and a closure fails to start them at all.
    fn = partial(collate, pad_id=tok.eos_token_id)
    out = {}
    for workers in (0, 2, 4):
        dl = DataLoader(encs, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=workers, collate_fn=fn)
        t = time.time()
        for _ in range(2):
            for _ in dl:
                pass
        out[str(workers)] = round(time.time() - t, 3)
    best = min(out, key=out.get)
    return {"seconds_for_two_passes": out, "chosen": int(best)}


@torch.no_grad()
def generate(model, tok, rows, *, max_new: int = GEN_MAX_NEW) -> list[dict]:
    """Greedy, unconstrained generation. Deterministic and gate-free.

    Greedy on purpose: sampling on this machine has a measured MPS defect on
    sparse distributions (report 06), and a smoke observation should not also
    be a sampling experiment. No inventory gate either -- the point is to see
    what the weights emit, not what the constraint layer would allow.
    """
    model.eval()
    out = []
    for r in rows:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": r.prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )["input_ids"].to(model.device)
        t = time.time()
        gen = model.generate(input_ids=ids, max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
        out.append({
            "sample_id": r.sample_id,
            "seconds": round(time.time() - t, 2),
            "n_new_tokens": int(gen.shape[1] - ids.shape[1]),
            "text": text,
        })
    return out


@torch.no_grad()
def eval_loss(model, tok, encs, cfg) -> float:
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(encs), cfg.batch_size):
        batch = collate(encs[i:i + cfg.batch_size], tok.eos_token_id)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        total += model(**batch).loss.detach().item()
        n += 1
    return total / max(n, 1)


def _write_report(*, cfg, model_info, data, env, perf, history, val0,
                  val1, before, after, bench, n_train_rows) -> list[str]:
    """Build the markdown. Shared by a fresh run and by --from-json."""
    train_encs = range(n_train_rows)          # only len() is used below

    L = ["# LoRA smoke test (2,000 rows)", ""]
    L.append("**A plumbing test, not an experiment.** It checks that a "
             "reproducible fine-tune runs end to end from the right starting "
             "point with a correct loss mask. It compares no hyperparameters, "
             "selects no checkpoint, and never reads the test split. The "
             "generations at the end are smoke observations: **no claim is "
             "made here about inventory compliance or generalisation**, which "
             "need the A-E protocol, multiple seeds and the hard-gate arms.")
    L += ["", "## Where training starts", "",
          "Not from bare Llama. BrickGPT is published as a LoRA adapter, so "
          "training a fresh adapter on the base model would discard the "
          "published checkpoint while still looking like fine-tuning.", "",
          f"- base model: `{model_info['base_model']}`",
          f"- published adapter: `{model_info['published_adapter']}` @ "
          f"`{model_info['published_adapter_revision']}` "
          f"(r={model_info['published_adapter_config']['r']}, "
          f"alpha={model_info['published_adapter_config']['alpha']})",
          f"- tokenizer: `AvaLovelace/BrickGPT` @ pinned revision, resolved "
          "separately from the adapter",
          f"- **start_from**: {model_info['start_from']}",
          f"- merge verified to change weights: "
          f"`{model_info['merge_changed_weights']}` (a fingerprint over "
          "q_proj/v_proj is compared before and after; the run aborts if the "
          "merge is a no-op)", ""]

    L += ["## Configuration (declared before the run)", "",
          "| setting | value |", "|---|---|"]
    for k, v in cfg.as_dict().items():
        L.append(f"| `{k}` | {v} |")
    L += ["",
          f"- trainable parameters: **{model_info['trainable_parameters']:,}** "
          f"of {model_info['total_parameters']:,} "
          f"({model_info['trainable_parameters']/model_info['total_parameters']:.3%})",
          f"- trainable tensors: {model_info['n_trainable_tensors']}",
          "- no quantisation: bf16 throughout. 4-bit was not used -- there is "
          "no dependable bitsandbytes path on Apple Silicon, and section 9.8 "
          "treats QLoRA as a memory option rather than a requirement.", ""]

    L += ["## Data", "",
          "| split | samples | pairs | objects |", "|---|---:|---:|---:|",
          f"| train | {data['train']['samples']} | {data['train']['pairs']} | "
          f"{data['train']['objects']} |",
          f"| val | {data['val']['samples']} | {data['val']['pairs']} | "
          f"{data['val']['objects']} |", "",
          f"- selection: {data['selection']}",
          f"- roles (train): {data['train']['roles']}",
          f"- variants (train): {data['train']['variants']}",
          f"- validation comes from the **val split only**; the test split is "
          f"not opened by this script (`test_split_read: "
          f"{data['test_split_read']}`)",
          f"- train/val `object_id` overlap: **{data['object_overlap']['shared']}** "
          f"({data['object_overlap']['train_objects']} vs "
          f"{data['object_overlap']['val_objects']} objects)",
          f"- rows truncated at max_length {cfg.max_length}: "
          f"**{data['truncated_rows']}** (longest row "
          f"{data['max_total_tokens']} tokens)", ""]

    L += ["## Loss", "",
          "| point | value |", "|---|---:|",
          f"| validation before training | {val0:.4f} |",
          f"| validation after training | {val1:.4f} |",
          f"| train, first logged window | {history[0]['train_loss']:.4f} |",
          f"| train, last logged window | {history[-1]['train_loss']:.4f} |", "",
          "Validation is computed with the same masking as training, over "
          f"{data['val']['samples']} held-out rows. Two points on a single "
          "short run are a pipeline signal, not evidence that the model "
          "learned the task.", "",
          "| epoch | rows seen | optimizer steps | train loss |",
          "|---:|---:|---:|---:|"]
    for h in history:
        L.append(f"| {h['epoch']} | {h['row']} | {h['optimizer_steps']} | "
                 f"{h['train_loss']:.4f} |")

    L += ["", "## Cost", ""]
    for k, v in perf.items():
        if k not in ("dataloader_workers", "marginal_rates", "degradation"):
            L.append(f"- `{k}`: {v}")

    rates, deg = perf["marginal_rates"], perf["degradation"]
    if deg:
        L += ["", "### The run gets progressively slower", "",
              f"**{deg['slowdown_factor']}x.** The first windows run at "
              f"{deg['first_windows_seconds_per_row']}s/row and the last at "
              f"{deg['last_windows_seconds_per_row']}s/row, with a worst window "
              f"of {deg['worst_window_seconds_per_row']}s/row. This is why the "
              f"mean ({perf['seconds_per_row_mean']}s) sits so far above the "
              f"median ({perf['seconds_per_row_median']}s), and why the "
              "progress line's ETA kept rising instead of falling: it "
              "extrapolates a rate that no longer holds.", "",
              "The trend is noisy rather than smooth -- the table below goes "
              "up and down by a factor of two or three between neighbouring "
              "windows -- so this is a rising, erratic cost with a severe "
              "tail, not a clean curve. Only the direction and the size of "
              "the endpoints are being claimed.", "",
              "Row length is ruled out: rows are shuffled, so long and short "
              "ones are spread evenly through the run.", "",
              "**Memory is not ruled out.** The two figures recorded are "
              f"`peak_process_rss_gb` = {perf['peak_process_rss_gb']} and "
              f"`mps_allocated_gb_end` = {perf['mps_allocated_gb_end']}, and "
              "they say only that the process's peak resident set was small "
              "and that PyTorch's *tracked* MPS allocation was small **at the "
              "moment the run ended**. Neither is a peak of MPS usage, and "
              "neither covers driver-side or IOKit allocations, unified-memory "
              "pressure, compaction, swap, or fragmentation inside the "
              "allocator -- an allocator that is degrading typically still "
              "reports a modest tracked total. So memory remains a live "
              "candidate alongside anything else. **The cause is not isolated "
              "here and this report claims none.** The next step is a short "
              "diagnostic over 100-200 rows recording current *and* driver MPS "
              "allocation, system memory pressure, sequence length and "
              "per-phase timing -- not another full run.", "",
              "| rows | seconds/row | train loss |", "|---:|---:|---:|"]
        for r in rates:
            if r["row"] % 250 == 0 or r["row"] in (50, len(train_encs)):
                L.append(f"| {r['row']} | {r['seconds_per_row']} | "
                         f"{r['train_loss']:.4f} |")
        L += ["",
              "**Consequence for the plan.** Section 9.8 asks whether local "
              "MPS is fast enough to be the real training environment. On this "
              f"evidence, not as it stands: 2,000 rows for one epoch took "
              f"{perf['train_seconds']/3600:.1f} hours, and the full "
              "inventory-conditioned set is 9,584 rows, which at three epochs "
              "is fourteen times this work. A flat 2.85s/row would put that "
              "near 23 hours; the observed curve would put it far higher. "
              "Isolating the slowdown -- or restarting the process "
              "periodically, or moving to the Kaggle GPU fallback the workflow "
              "already allows -- has to come before the A-E runs, and is a "
              "finding of this smoke test rather than a detail of it.", ""]
    L += ["",
          f"DataLoader workers were chosen by measurement, not raised on "
          f"principle: {bench['seconds_for_two_passes']} seconds for two "
          f"passes over 64 rows at 0/2/4 workers, so **{bench['chosen']}** was "
          "used. Rows are encoded once up front and held in memory, which "
          "leaves a worker process almost nothing to do.", ""]

    L += ["## Environment", ""]
    for k, v in env.items():
        L.append(f"- `{k}`: {v}")

    prov = data.get("provenance", {})
    if prov:
        L += ["", "## Provenance", "",
              "What this run read and what it produced, so a later reader can "
              "tell whether the report still describes the files on disk.", "",
              "| item | value |", "|---|---|"]
        for name, digest in prov.get("instruction_sha256", {}).items():
            L.append(f"| `{name}` sha256 | `{digest}` |")
        L += [f"| train selection digest | `{data.get('train_selection_digest')}` |",
              f"| val selection digest | `{data.get('val_selection_digest')}` |"]
        for ep, dig in (data.get("training_order_digest") or {}).items():
            L.append(f"| training-order digest (epoch {ep}) | `{dig}` |")
        L += [f"| manifest sha256 | `{prov.get('manifest_sha256')}` |",
              f"| base model | `{prov['base_model']}` @ "
              f"`{prov['base_revision']}` |",
              f"| published adapter | `{prov['published_adapter']}` @ "
              f"`{prov['published_adapter_revision']}` |",
              f"| tokenizer | `{prov['tokenizer']}` @ "
              f"`{prov['tokenizer_revision']}` |",
              f"| saved adapter sha256 | `{prov['adapter_sha256']}` |", ""]
        code = prov.get("code", {})
        if code.get("limitation"):
            L += ["### Code provenance: an unrecoverable gap", "",
                  "**The exact source that produced this run was never "
                  "recorded.** " + code["limitation"], ""]
        elif code:
            L += ["Source state, captured **before** the model loaded and "
                  "training began -- not read afterwards, when it would "
                  "describe whatever the tree had become by then:", "",
                  f"- HEAD at start: `{code.get('head')}`",
                  f"- working tree dirty: `{code.get('working_tree_dirty')}`"]
            if code.get("dirty_paths"):
                L.append(f"- uncommitted at start: "
                         f"{', '.join('`%s`' % p for p in code['dirty_paths'])}")
            L += ["", "| file | sha256 |", "|---|---|"]
            for f, dig in code.get("file_sha256", {}).items():
                L.append(f"| `{f}` | `{dig}` |")
            L.append("")
        L += ["**Selection digest and training-order digest are different "
              "things.** The selection digest covers *which* rows were chosen "
              "and their fixed listing order. The training-order digest covers "
              "the order they were actually fed to the model after "
              "`torch.randperm`: two runs can select identical rows and still "
              "present them in different orders, which is a different "
              "curriculum over the same data.", ""]
        if data.get("training_order_digest_note"):
            L += [f"*{data['training_order_digest_note']}*", ""]
        back = prov.get("backfilled_after_the_run", [])
        if back:
            L += ["### Backfilled, not measured during the run", "",
                  "These fields did not exist when this run executed and were "
                  "established afterwards from evidence. They are "
                  "reconstructions and are labelled as such rather than "
                  "presented as readings taken while training.", "",
                  "| field | how it got here |", "|---|---|"]
            for k in back:
                L.append(f"| `{k}` | **backfilled after the run** |")
            L += ["", prov.get("backfill_basis", ""), ""]

    L += ["", "## Generations before and after (smoke observation only)", "",
          f"{N_GEN} fixed prompts from the **val** split, greedy decoding, no "
          "inventory gate, {} new tokens max. Greedy because sampling on this "
          "machine has a measured MPS defect on sparse distributions (report "
          "06) and a smoke check should not also be a sampling experiment. "
          "Before-training output is the merged BrickGPT: a freshly "
          "initialised LoRA has `lora_B = 0` and is exactly the identity, so "
          "the two rows differ only by what training changed.".format(GEN_MAX_NEW),
          "",
          "**These are not evidence of compliance or quality.** They exist to "
          "show the generation path still produces brick syntax after "
          "training.", "",
          "Two neutral observations, recorded because the next round should "
          "look at them rather than because anything is concluded here. "
          "Every output on both sides parses as bricks, so the syntax survived "
          "training. And with one exception before training, none of these "
          f"generations emitted EOS inside {GEN_MAX_NEW} tokens -- they were "
          "cut off by the budget. Whether that is the prompt, the greedy "
          "decode, the short run, or nothing at all is not established by "
          "four samples.", ""]
    for b, a in zip(before, after):
        L += [f"### `{b['sample_id']}`", "",
              f"- before: {b['n_new_tokens']} tokens in {b['seconds']}s",
              f"- after: {a['n_new_tokens']} tokens in {a['seconds']}s", "",
              "```text", "BEFORE:", b["text"][:600].rstrip(), "",
              "AFTER:", a["text"][:600].rstrip(), "```", ""]

    L += ["## Checkpoint", "",
          f"- adapter: `artifacts/checkpoints/lora_smoke/` (gitignored; only "
          "the script, config, tests and this report are committed)",
          "- the published adapter is merged into the base and is **not** "
          "modified; what is saved is our delta alone",
          "- saved without tokenizer files, which is why the loader addresses "
          "tokenizer and adapter separately", ""]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # rstrip each line: several sections build text with trailing spaces before
    # a join, and `git diff --check` treats those as whitespace errors in a
    # committed file.
    body = "\n".join(line.rstrip() for line in L).rstrip("\n") + "\n"
    (REPORT_DIR / "13_lora_smoke.md").write_text(body, encoding="utf-8")
    return L


def _check_replayable(stored: dict) -> None:
    """Refuse to re-render if the inputs or the checkpoint have moved.

    Re-rendering exists so prose can be corrected without repeating 5.8 hours.
    That is only honest while the report still describes the files on disk: a
    changed instruction file or a retrained adapter would otherwise be
    described by the previous run's numbers under the new digests.
    """
    prov = stored.get("data", {}).get("provenance")
    if not prov:
        print("note: stored run predates the provenance block; nothing to "
              "check against", file=sys.stderr)
        return

    problems = []
    for name, was in prov.get("instruction_sha256", {}).items():
        path = OUT_DIR / name
        if not path.exists():
            problems.append(f"{name} is missing")
        elif (now := sha256_file(path)) != was:
            problems.append(f"{name} changed: {was[:12]}... -> {now[:12]}...")

    for label, key, path, note in (
        ("adapter", "adapter_sha256",
         CKPT_DIR / "adapter_model.safetensors", "retrained since this report"),
        ("manifest", "manifest_sha256", CKPT_DIR / MANIFEST_NAME,
         "the load contract changed since this report"),
    ):
        was = prov.get(key)
        if not was:
            continue
        if not path.exists():
            problems.append(f"the saved {label} is gone")
        elif (now := sha256_file(path)) != was:
            problems.append(
                f"{label} changed: {was[:12]}... -> {now[:12]}... ({note})")

    if problems:
        raise SystemExit(
            "cannot re-render: the stored run no longer describes what is on "
            "disk:\n  - " + "\n  - ".join(problems)
            + "\nRe-run the training instead of replaying.")


def render_only() -> int:
    """Rebuild the report from the stored run. No model, no training.

    Training this is 5.8 hours; correcting a sentence is not a reason to spend
    it. Only the prose and the derived tables are rebuilt -- every measured
    number comes from the stored JSON, which is not rewritten except to carry
    the same values through.
    """
    stored = json.loads((REPORT_DIR / "13_lora_smoke.json").read_text())
    _check_replayable(stored)
    _write_report(
        cfg=LoraConfig_(**{k: tuple(v) if isinstance(v, list) else v
                           for k, v in stored["config"].items()
                           if k in LoraConfig_.__dataclass_fields__}),
        model_info=stored["model"], data=stored["data"], env=stored["env"],
        perf=stored["performance"], history=stored["loss"]["history"],
        val0=stored["loss"]["val_before"], val1=stored["loss"]["val_after"],
        before=stored["generations"]["before"],
        after=stored["generations"]["after"],
        bench=stored["performance"]["dataloader_workers"],
        n_train_rows=stored["data"]["train"]["samples"],
    )
    print("re-rendered data/reports/13_lora_smoke.md from the stored run")
    return 0


def main() -> int:
    if "--from-json" in sys.argv[1:]:
        return render_only()

    cfg = LoraConfig_()
    torch.manual_seed(cfg.seed)
    t_start = time.time()

    # First, before the tokenizer, the model, or any data: the source state.
    code = code_provenance()
    print(f"code: HEAD {code['head']} dirty={code['working_tree_dirty']}",
          flush=True)

    tok = load_tokenizer()
    train_rows = sample_pairs(read_rows(OUT_DIR / "instruct_inv_train.jsonl"),
                              n_pairs=N_PAIRS, seed=cfg.seed)
    val_rows = sample_pairs(read_rows(OUT_DIR / "instruct_inv_val.jsonl"),
                            n_pairs=N_VAL_PAIRS, seed=cfg.seed)

    data = {
        "train": split_stats(train_rows),
        "val": split_stats(val_rows),
        "val_source": "instruct_inv_val.jsonl (val split only)",
        "test_split_read": False,
        "object_overlap": check_no_object_overlap(train_rows, val_rows),
        "selection": (
            f"{N_PAIRS} whole pairs by seeded shuffle of sorted pair ids "
            f"(seed {cfg.seed}); every row of a chosen pair is included"
        ),
        "train_selection_digest": selection_digest(train_rows),
        "val_selection_digest": selection_digest(val_rows),
    }
    print(f"train {data['train']['samples']} rows / {data['train']['pairs']} pairs "
          f"/ {data['train']['objects']} objects", flush=True)
    print(f"val   {data['val']['samples']} rows / {data['val']['pairs']} pairs "
          f"/ {data['val']['objects']} objects", flush=True)

    bench = benchmark_workers(train_rows, tok, cfg)
    print(f"dataloader workers: {bench}", flush=True)

    train_encs = [encode_row(tok, r, cfg.max_length) for r in train_rows]
    val_encs = [encode_row(tok, r, cfg.max_length) for r in val_rows]
    truncated = sum(e.truncated for e in train_encs + val_encs)
    data["truncated_rows"] = truncated
    data["max_total_tokens"] = max(len(e.input_ids) for e in train_encs + val_encs)
    if truncated:
        print(f"WARNING: {truncated} rows truncated", file=sys.stderr)

    model, model_info = build_model(cfg)
    assert_only_lora_trainable(model)
    print(f"trainable {model_info['trainable_parameters']:,} / "
          f"{model_info['total_parameters']:,}", flush=True)

    gen_rows = val_rows[:N_GEN]
    print("generating BEFORE ...", flush=True)
    before = generate(model, tok, gen_rows)

    val0 = eval_loss(model, tok, val_encs, cfg)
    print(f"val loss before: {val0:.4f}", flush=True)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)

    order = list(range(len(train_encs)))
    history: list[dict] = []
    step_times: list[float] = []
    tokens_seen = 0
    opt_steps = 0

    model.train()
    t_train = time.time()
    training_order: dict[int, str] = {}
    for epoch in range(cfg.epochs):
        rng = torch.Generator().manual_seed(cfg.seed + epoch)
        perm = torch.randperm(len(order), generator=rng).tolist()
        # Recorded as it happens. Re-deriving this later would depend on the
        # torch version's RNG staying identical, which is not a property to
        # rely on for a provenance record.
        training_order[epoch] = training_order_digest(perm, train_rows)
        opt.zero_grad()
        run_loss, run_n = 0.0, 0

        for i, idx in enumerate(perm, 1):
            t0 = time.time()
            batch = collate([train_encs[idx]], tok.eos_token_id)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / cfg.grad_accum).backward()
            run_loss += loss.detach().item()
            run_n += 1
            tokens_seen += int(batch["attention_mask"].sum())

            if i % cfg.grad_accum == 0:
                opt.step()
                opt.zero_grad()
                opt_steps += 1
            step_times.append(time.time() - t0)

            if i % 50 == 0 or i == len(perm):
                done = time.time() - t_train
                print(f"  epoch {epoch} {i}/{len(perm)}  "
                      f"loss {run_loss / run_n:.4f}  "
                      f"{done / i:.2f}s/row  eta {(len(perm) - i) * done / i:5.0f}s",
                      flush=True)
                history.append({
                    "epoch": epoch, "row": i, "optimizer_steps": opt_steps,
                    "train_loss": round(run_loss / run_n, 4),
                    # Elapsed is recorded per window so the *marginal* rate can
                    # be recovered later. Summary statistics alone hide a
                    # progressive slowdown: a median stays respectable while
                    # the tail crawls, and the run's own ETA keeps lying.
                    "elapsed_seconds": round(done, 1),
                })
                run_loss, run_n = 0.0, 0

    train_seconds = time.time() - t_train
    val1 = eval_loss(model, tok, val_encs, cfg)
    print(f"val loss after: {val1:.4f}", flush=True)

    print("generating AFTER ...", flush=True)
    after = generate(model, tok, gen_rows)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CKPT_DIR)
    (CKPT_DIR / "training_config.json").write_text(
        json.dumps({"config": cfg.as_dict(), "model": model_info}, indent=2))
    manifest = write_manifest(CKPT_DIR, model_info, cfg)
    data["training_order_digest"] = training_order
    data["provenance"] = {**run_provenance(manifest), "code": code}

    perf = {
        "train_seconds": round(train_seconds, 1),
        "total_seconds": round(time.time() - t_start, 1),
        "rows": len(train_encs),
        "optimizer_steps": opt_steps,
        "seconds_per_row_median": round(sorted(step_times)[len(step_times) // 2], 3),
        "seconds_per_row_mean": round(sum(step_times) / len(step_times), 3),
        "seconds_per_optimizer_step": round(train_seconds / max(opt_steps, 1), 3),
        "tokens_seen": tokens_seen,
        "tokens_per_second": round(tokens_seen / train_seconds, 1),
        "peak_process_rss_gb": round(rss_gb(), 2),
        "mps_allocated_gb_end": round(mps_gb(), 2),
        "dataloader_workers": bench,
        "marginal_rates": marginal_rates(history),
        "degradation": degradation(marginal_rates(history)),
    }
    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "peft": version("peft"),
        "device": model_info["device"],
    }

    _write_report(cfg=cfg, model_info=model_info, data=data, env=env,
                  perf=perf, history=history, val0=val0, val1=val1,
                  before=before, after=after, bench=bench,
                  n_train_rows=len(train_encs))
    (REPORT_DIR / "13_lora_smoke.json").write_text(json.dumps({
        "config": cfg.as_dict(), "model": model_info, "data": data,
        "env": env, "performance": perf,
        "loss": {"val_before": val0, "val_after": val1, "history": history},
        "generations": {"before": before, "after": after},
    }, indent=2), encoding="utf-8")
    print("\nwrote data/reports/13_lora_smoke.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
