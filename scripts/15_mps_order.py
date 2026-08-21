"""Report 15: run the two MPS conditions in separate processes, both orders.

Report 14 found that the arm which cleared the cache did not degrade -- and
that arm also ran second, in the same process, on a machine the first arm had
already loaded. Order was confounded with condition and n was 1, so "clearing
helped" and "running second helped" predict the same result.

This script separates them:

* every arm gets its own Python process, so no allocator, heap, model or
  optimizer state crosses between arms
* each condition runs twice, once early and once late
* the execution order is fixed and written down before anything starts:

      position 0   block B1   continuous
      position 1   block B1   empty_cache
      position 2   block B2   empty_cache
      position 3   block B2   continuous

  Each condition therefore has mean global position 1.5, which is what makes
  the contrast readable against a machine that drifts in one direction.

The plan is written atomically before the first child is spawned and its
digest is checked by every child and again at aggregation, so the order cannot
be adjusted after a result is visible.

**The parent does not believe the children.** A child records measurements; it
does not get to say whether it succeeded, whether the machine had recovered,
or whether its numbers may be compared. Exit status comes from the process
table, recovery from the parent's own polling, and eligibility is recomputed
from raw fields at aggregation time.

Nothing is trained in the sense of being kept: optimizer updates run, in
memory, on real training rows, and no checkpoint is written.

Modes:
    --calibrate      idle sampling only; loads no model
    --run            parent: plan, spawn four children, gate, aggregate
    --child          one condition in this process (spawned by the parent)
    --verify         fail-closed check of a stored experiment on disk
    --from-json      verify, then re-render the markdown from what is stored
    --recompute      re-derive every conclusion from the raw files and write
                     it beside the parent's aggregate, never over it
    --session-init   lay out a one-run-per-boot session: plan, thresholds and
                     an exact copy of the source, fixed before the first run
    --session-next   run the next child in this boot, then exit
    --session-finalize  turn a stopped session into its record, without
                     re-running anything; idempotent
    --session-status what the session has and has not done so far

The session modes are the approved way to run this experiment: one measured
run per boot, four boots, with the operator restarting the machine in between.
`--run`, the same-boot flow they replace, is withdrawn permanently and refuses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.brickgpt import load_tokenizer  # noqa: E402
from src.training.diagnostics import (  # noqa: E402
    DeviceOps,
    PhaseTimer,
    StopCondition,
    memory_sample,
    summarise_phases,
    system_memory,
    window_stats,
)
from src.training.lora import (  # noqa: E402
    LoraConfig_,
    assert_only_lora_trainable,
    build_model,
    collate,
    encode_row,
    read_rows,
    sample_pairs,
)
from src.training.preflight import (  # noqa: E402
    CALIBRATION_INTERVAL_SECONDS,
    CALIBRATION_SAMPLES,
    GATE_METRICS,
    RECOVERY_CONSECUTIVE_PASSES,
    RECOVERY_MAX_WAIT_SECONDS,
    RECOVERY_POLL_SECONDS,
    calibrate,
    evaluate_gate,
    preflight_sample,
    thresholds_from,
    wait_for_recovery,
)
from src.training.session import (  # noqa: E402
    EVENT_DIR,
    EVENT_FILE_RE,
    EVENT_FINISHED,
    EVENT_GATE_ATTEMPT,
    EVENT_KINDS,
    EVENT_PRE_SPAWN_ABORT,
    EVENT_RETRYABLE,
    EVENT_STARTED,
    append_event,
    boot_identity,
    exclusive_lock,
    manifest_digest,
    read_events,
    sha256_file,
    snapshot_sources,
    verify_sources,
    write_once_json,
)

OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "data" / "reports"
RUN_DIR = REPORT_DIR / "15_mps_order"

EXPERIMENT_SCHEMA = 3

MAX_ROWS = 200
WINDOW = 20
MEMORY_EVERY = 5
EMPTY_CACHE_EVERY = 10
PHASES = ("collate_h2d", "forward", "backward", "optimizer")

#: Declared before running, in this order, and checked against what actually
#: ran. Two blocks, reversed, so condition is crossed with position.
PLAN_SPEC = (
    ("B1", "continuous", 0),
    ("B1", "empty_cache", 1),
    ("B2", "empty_cache", 0),
    ("B2", "continuous", 1),
)

CONDITION_MAX_SECONDS = 30 * 60
EXPERIMENT_MAX_SECONDS = 2 * 60 * 60

#: A pre-selected engineering tolerance, fixed before any run. It is **not**
#: derived from report 14: that run stored its losses rounded to four decimal
#: places, so it cannot bound cross-process bf16 agreement at any finer
#: precision, and quoting a number from it would be inventing evidence. The
#: observed maximum difference is reported whatever this value is.
LOSS_TOLERANCE = 1e-3
LOSS_TOLERANCE_BASIS = (
    "Pre-selected engineering tolerance, fixed before any run. NOT derived "
    "from report 14, whose losses were stored rounded to four decimal places "
    "and therefore cannot bound cross-process bf16 agreement. Exceeding it "
    "does not invalidate the timing measurements, but the set may not then be "
    "described as having followed equivalent training trajectories.")

CODE_FILES = (
    "scripts/15_mps_order.py",
    "src/training/preflight.py",
    "src/training/session.py",
    "src/training/diagnostics.py",
    "src/training/lora.py",
    "src/data/instruction.py",
    "src/model_ids.py",
)

#: Stored timings are rounded (per-row to 6 decimals, per-call clears to 5,
#: totals to 3 or 4), so sums of the parts never land exactly on the recorded
#: whole. These bound the arithmetic, not the measurement: anything larger than
#: this is a bookkeeping error rather than rounding.
SUM_TOLERANCE_SECONDS = 0.01
CLEAR_SUM_TOLERANCE_SECONDS = 0.001


def row_span_tolerance(rows: int) -> float:
    """How much of the wall clock may sit outside the per-row spans.

    The loop does a little work between one row's span ending and the next
    beginning. It is tens of microseconds per row, and the point of bounding it
    is that the teardown clear -- seconds, on this machine -- cannot hide there.
    """
    return 0.02 + 0.002 * rows

#: A child states what it measured. These are conclusions, and conclusions are
#: the parent's job: exit status comes from the process table, recovery from
#: the parent's polling, eligibility from a recomputation over raw fields. A
#: child that writes any of them is rejected rather than trusted.
CHILD_FORBIDDEN_KEYS = ("exit_status", "recovery", "recovery_passed",
                        "eligible_for_paired_contrast", "complete",
                        "comparable")

CHILD_REQUIRED_KEYS = (
    "experiment_id", "run_id", "block_id", "condition", "position_in_block",
    "global_position", "plan_digest", "provenance", "preflight",
    "model_load_seconds", "rows_completed", "rows_requested", "stopped_early",
    "input_order_digest", "completed_input_digest", "model_compute_seconds",
    "end_to_end_seconds", "between_row_overhead_breakdown",
    "scheduled_empty_cache_every", "scheduled_empty_cache_calls",
    "scheduled_empty_cache_cost", "teardown_empty_cache_calls",
    "teardown_empty_cache_seconds", "phases", "windows", "memory", "per_row",
    "child_pid", "started_at", "finished_at",
)

REQUIRED_PROVENANCE = (
    "head", "working_tree_dirty", "code_sha256", "instruction_sha256",
    "selection_digest", "training_order_digest", "lora_config", "optimizer",
    "packages", "device", "dtype", "phases", "stop_conditions",
    "measurement_intervals", "base_model", "base_revision",
    "published_adapter_revision", "tokenizer_revision",
)

#: Everything that must be the same in every run, whichever arm it is. The
#: condition and its clear schedule are deliberately absent: those are the one
#: thing the experiment varies, and a comparison that demanded they match would
#: reject the experiment for running.
COMPARED_PROVENANCE_FIELDS = (
    "code_sha256", "head", "working_tree_dirty", "instruction_sha256",
    "selection_digest", "training_order_digest", "lora_config", "optimizer",
    "packages", "device", "dtype", "phases", "stop_conditions",
    "measurement_intervals", "base_model", "base_revision",
    "published_adapter", "published_adapter_revision", "tokenizer_revision",
)

#: Compared straight off the child record rather than out of its provenance.
COMPARED_RECORD_FIELDS = ("input_order_digest", "rows_requested",
                          "trainable_parameters")

REQUIRED_CALIBRATION_KEYS = ("kind", "loads_model", "samples", "stats",
                             "thresholds", "gate", "calibration_digest")

#: A gate record with a field missing is not a gate that passed quietly; it is
#: a gate whose result cannot be recomputed, and that fails.
REQUIRED_GATE_KEYS = ("passed", "polls", "waited_seconds",
                      "consecutive_passes_required", "reason")
REQUIRED_POLL_KEYS = ("poll", "elapsed_seconds", "sample", "passed",
                      "failed_metrics", "consecutive_passes")
REQUIRED_CHILD_SLOT_KEYS = ("run_id", "condition", "block_id",
                            "global_position", "exit_status", "gate",
                            "recovery_passed_recomputed", "not_run")

#: What a headline is allowed to rest on. Every one of these must hold; the
#: report says which failed when one does.
HEADLINE_REQUIREMENTS = (
    "all four runs completed",
    "all four runs eligible for the paired contrast",
    f"every pair shares all {MAX_ROWS} rows",
    "token and supervised-token counts identical across runs",
    "loss verdict within_tolerance",
    "both blocks agree on the direction",
    "every run measured the same code, data and settings",
    "every run honoured its treatment contract",
    "each run started from its own boot",
)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

#: One implementation of the write-once rule, shared with the session flow.
atomic_write_json = write_once_json


def digest_ids(ids) -> str:
    h = hashlib.sha256()
    for i in ids:
        h.update(str(i).encode())
        h.update(b"\n")
    return h.hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def digest_obj(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_under_root(path) -> Path:
    """Take a path from the command line and put it where the tree keeps it.

    Everything a session stores is recorded relative to the tree root, so a
    path that arrives relative to the shell's working directory has to be
    anchored before it can be stored -- otherwise the two disagree and the only
    symptom is a crash on the way out.
    """
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve()


def rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def build_plan(experiment_id: str) -> list[dict]:
    return [
        {"run_id": f"r{i + 1}", "block_id": block, "condition": condition,
         "position_in_block": in_block, "global_position": i,
         "experiment_id": experiment_id}
        for i, (block, condition, in_block) in enumerate(PLAN_SPEC)
    ]


def resolve_device(torch_mod) -> str:
    """Refuse to run anywhere but MPS.

    On CPU ``empty_cache()`` is an explicit no-op, so the treatment arm would
    schedule its clears, perform none, come out identical to the control, and
    be rejected later for a clear count that contradicts its own schedule --
    after the machine hours were spent.
    """
    try:
        available = bool(torch_mod.backends.mps.is_available())
    except Exception:
        available = False
    if not available:
        raise SystemExit(
            "MPS is not available, and this diagnostic is only about MPS. On "
            "CPU `empty_cache()` is a no-op, so the treatment arm would "
            "schedule clears it never performs and come out identical to the "
            "control. Stopping before the run rather than after it.")
    return "mps"


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def run_calibration(out_path: Path, *, samples=CALIBRATION_SAMPLES,
                    interval=CALIBRATION_INTERVAL_SECONDS,
                    sampler=preflight_sample, sleep=time.sleep) -> dict:
    """Idle sampling only. No model is loaded and no row is trained."""
    collected = []
    for i in range(samples):
        s = sampler()
        collected.append(s)
        print(f"  calibration {i + 1}/{samples}: "
              + ", ".join(f"{m}={s.get(m)}" for m in GATE_METRICS), flush=True)
        if i < samples - 1:
            sleep(interval)

    stats = calibrate(collected)
    record = {
        "schema_version": EXPERIMENT_SCHEMA,
        "kind": "preflight_calibration",
        "created_at": now_iso(),
        "loads_model": False,
        "samples_requested": samples,
        "interval_seconds": interval,
        "metrics": list(GATE_METRICS),
        "note": ("Idle machine sampling. This pass loads no model, which is a "
                 "property of this code path; whether anything else had "
                 "already loaded one is NOT knowable from these readings -- "
                 "swap, memory pressure, page counts and load average do not "
                 "record what ran before. Sampling a genuinely quiet machine "
                 "is an operating condition the operator arranges. No process "
                 "names, paths or command lines are recorded: the gate needs "
                 "to know whether the machine is busy, not what it is busy "
                 "with. `free_plus_inactive_gb` is what vm_stat allows adding "
                 "up; inactive pages are reclaimable, so it is neither free "
                 "nor available memory."),
        "scale_formula": "scale = 1.4826 * MAD",
        "samples": collected,
        "stats": stats,
        "thresholds": thresholds_from(stats),
        "gate": {"poll_seconds": RECOVERY_POLL_SECONDS,
                 "max_wait_seconds": RECOVERY_MAX_WAIT_SECONDS,
                 "consecutive_passes_required": RECOVERY_CONSECUTIVE_PASSES},
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    record["calibration_digest"] = digest_obj(
        {"stats": stats, "thresholds": record["thresholds"],
         "gate": record["gate"]})
    atomic_write_json(out_path, record)
    return record


# --------------------------------------------------------------------------
# child
# --------------------------------------------------------------------------

def capture_provenance(rows, perm, cfg, opt_spec, *, device: str) -> dict:
    """Recorded before the model loads. Report 13's gap cannot be closed; this
    is how it is not repeated."""
    from src.model_ids import (ADAPTER, ADAPTER_REVISION, BASE_MODEL,
                               BASE_REVISION, TOKENIZER_REVISION)

    def git(*a):
        try:
            r = subprocess.run(["git", "-C", str(ROOT), *a],
                               capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:
            return None

    status = git("status", "--porcelain")
    stop = StopCondition(max_seconds=CONDITION_MAX_SECONDS)
    return {
        "captured": "before model load",
        "head": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(status) if status is not None else None,
        "code_sha256": {f: sha256_file(ROOT / f) for f in CODE_FILES
                        if (ROOT / f).exists()},
        "instruction_sha256": {
            "instruct_inv_train.jsonl":
                sha256_file(OUT_DIR / "instruct_inv_train.jsonl")},
        "selection_digest": digest_ids(r.sample_id for r in rows),
        "training_order_digest": digest_ids(rows[i].sample_id for i in perm),
        "lora_config": cfg.as_dict(),
        "optimizer": opt_spec,
        "packages": {"python": platform.python_version(),
                     "torch": torch.__version__,
                     "transformers": version("transformers"),
                     "peft": version("peft")},
        "device": device,
        "dtype": cfg.dtype,
        "phases": list(PHASES),
        "stop_conditions": {"slow_row_seconds": stop.slow_row_seconds,
                            "slow_row_streak": stop.slow_row_streak,
                            "max_seconds": stop.max_seconds},
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "published_adapter": ADAPTER,
        "published_adapter_revision": ADAPTER_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "measurement_intervals": {"window": WINDOW,
                                  "memory_sample_every": MEMORY_EVERY,
                                  "empty_cache_every": EMPTY_CACHE_EVERY,
                                  "max_rows": MAX_ROWS},
    }


def _serialise_row(row: dict) -> dict:
    """Round the timings; leave the loss exactly as measured."""
    out = {}
    for k, v in row.items():
        out[k] = v if k == "loss" else (round(v, 6) if isinstance(v, float)
                                        else v)
    return out


def child_verify_source(session_dir: Path, plan_digest_expected: str) -> dict:
    """The child checks the code it is about to run, before it runs any of it.

    The parent checked the same thing a moment earlier, and that is not enough:
    between the parent's check and this process starting, the tree stayed
    writable. This runs before the tokenizer, before the rows and before the
    model, because after any of those the check would be describing something
    the process had already committed to.
    """
    session_dir = Path(session_dir)
    session_path = session_dir / "session.json"
    plan_path = session_dir / "plan.json"
    for p in (session_path, plan_path):
        if not p.exists():
            raise SystemExit(f"{p} does not exist, so this child cannot check "
                             "the source it was asked to run")
    session = json.loads(session_path.read_text())
    plan_doc = json.loads(plan_path.read_text())
    problems: list[str] = []
    if digest_obj(plan_doc.get("plan")) != plan_doc.get("plan_digest"):
        problems.append("the plan file's own digest does not match its plan")
    if plan_doc.get("plan_digest") != plan_digest_expected:
        problems.append("the plan file's digest is not the one this child was "
                        "told to run")
    if session.get("plan_digest") != plan_digest_expected:
        problems.append("the session and this child disagree on the plan "
                        "digest")
    manifest = session.get("source_manifest") or {}
    if manifest_digest(manifest) != session.get("source_manifest_digest"):
        problems.append("the source manifest does not match the digest stored "
                        "with it")
    problems += verify_sources(
        ROOT, manifest,
        session_dir / session.get("source_snapshot_dir", SNAPSHOT_DIR))
    if problems:
        raise SystemExit(
            "this child refuses to start:\n  - " + "\n  - ".join(problems)
            + "\nNothing has been loaded. The plan and the source it was made "
              "against must both still be what they were.")
    return {"files_verified": len(manifest.get("files") or {}),
            "source_manifest_digest": session.get("source_manifest_digest"),
            "verified_at": now_iso()}


def run_child(entry: dict, plan_digest_expected: str, out_path: Path,
              session_dir: Path) -> int:
    # Before the tokenizer, before the rows, before the model.
    source_check = child_verify_source(session_dir, plan_digest_expected)

    started_at = now_iso()
    condition = entry["condition"]
    if condition not in ("continuous", "empty_cache"):
        raise SystemExit(f"unknown condition {condition!r}")
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path}")

    device = resolve_device(torch)
    pre = preflight_sample()

    cfg = LoraConfig_()
    tok = load_tokenizer()
    rows = sample_pairs(read_rows(OUT_DIR / "instruct_inv_train.jsonl"),
                        n_pairs=250, seed=cfg.seed)
    encs = [encode_row(tok, r, cfg.max_length) for r in rows]

    # Rebuilt here, from the seed, rather than handed down by the parent: the
    # parent must have no way to change what a child trains on.
    rng = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(encs), generator=rng).tolist()[:MAX_ROWS]
    input_order_digest = digest_ids(rows[i].sample_id for i in perm)

    opt_spec = {"class": "AdamW", "lr": cfg.learning_rate,
                "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01}
    provenance = capture_provenance(rows, perm, cfg, opt_spec, device=device)

    dev = DeviceOps(device)
    torch.manual_seed(cfg.seed)
    t_load = time.perf_counter()
    model, info = build_model(cfg, device=device)
    model_load_seconds = time.perf_counter() - t_load
    assert_only_lora_trainable(model)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.01)
    model.train()

    clear_every = EMPTY_CACHE_EVERY if condition == "empty_cache" else None
    timer = PhaseTimer(sync=dev.sync)
    stop = StopCondition(max_seconds=CONDITION_MAX_SECONDS)
    memory: list[dict] = []
    stopped: str | None = None
    probe_seconds = 0.0
    t_begin = time.perf_counter()

    opt.zero_grad()
    for i, idx in enumerate(perm, 1):
        row_begin = time.perf_counter()
        e = encs[idx]
        timer.start()

        batch = collate([e], tok.eos_token_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        timer.phase("collate_h2d")

        loss = model(**batch).loss
        timer.phase("forward")

        (loss / cfg.grad_accum).backward()
        timer.phase("backward")

        if i % cfg.grad_accum == 0:
            opt.step()
            opt.zero_grad()
        timer.phase("optimizer")

        row = timer.end(row=i, sample_id=rows[idx].sample_id,
                        n_tokens=int(batch["attention_mask"].sum()),
                        n_supervised=int((batch["labels"] != -100).sum()),
                        loss=loss.detach().item())

        clear_seconds = 0.0
        if clear_every and i % clear_every == 0:
            clear_seconds = dev.empty_cache()

        probe = 0.0
        if i % MEMORY_EVERY == 0 or i == 1:
            t_probe = time.perf_counter()
            sample = {"row": i,
                      "elapsed_seconds": round(time.perf_counter() - t_begin, 2),
                      **memory_sample(torch, rss_bytes=rss_bytes()),
                      **system_memory()}
            probe = time.perf_counter() - t_probe
            sample["probe_seconds"] = round(probe, 4)
            memory.append(sample)
            probe_seconds += probe

        elapsed = time.perf_counter() - t_begin
        if (reason := stop.check(elapsed, row["total"])):
            stopped = reason
            print(f"  [{condition}] stopping at row {i}: {reason}", flush=True)
        elif i % WINDOW == 0:
            print(f"  [{condition}] {i}/{len(perm)}  {elapsed / i:.2f}s/row  "
                  f"loss {row['loss']:.4f}", flush=True)

        row["scheduled_empty_cache_seconds"] = clear_seconds
        row["memory_probe_seconds"] = probe
        row["end_to_end"] = time.perf_counter() - row_begin
        if stopped:
            break

    done = timer.rows
    total = time.perf_counter() - t_begin
    compute = sum(r["total"] for r in done)
    clear_cost = dev.scheduled_clear_cost()
    del model, opt
    teardown_seconds = dev.empty_cache(teardown=True)

    record = {
        "schema_version": EXPERIMENT_SCHEMA,
        "kind": "child",
        **{k: entry[k] for k in ("experiment_id", "run_id", "block_id",
                                 "condition", "position_in_block",
                                 "global_position")},
        "plan_digest": plan_digest_expected,
        "child_source_check": source_check,
        "child_pid": os.getpid(),
        "started_at": started_at,
        "finished_at": now_iso(),
        "provenance": provenance,
        "preflight": pre,
        "model_load_seconds": round(model_load_seconds, 3),
        "rows_completed": len(done),
        "rows_requested": len(perm),
        "stopped_early": stopped,
        "input_order_digest": input_order_digest,
        "completed_input_digest": digest_ids(r["sample_id"] for r in done),
        "end_to_end_seconds": round(total, 3),
        "model_compute_seconds": round(compute, 3),
        "between_row_overhead_seconds": round(total - compute, 3),
        "between_row_overhead_breakdown": {
            "scheduled_empty_cache_seconds": clear_cost["total_seconds"],
            "memory_probe_seconds": round(probe_seconds, 4),
            "unattributed_seconds": round(
                total - compute - clear_cost["total_seconds"] - probe_seconds,
                4)},
        "end_to_end_seconds_per_row": round(total / max(len(done), 1), 4),
        "model_compute_seconds_per_row": round(compute / max(len(done), 1), 4),
        "scheduled_empty_cache_every": clear_every,
        "scheduled_empty_cache_calls": dev.scheduled_empty_cache_calls,
        "scheduled_empty_cache_cost": clear_cost,
        "teardown_empty_cache_calls": dev.teardown_empty_cache_calls,
        "teardown_empty_cache_seconds": round(teardown_seconds, 5),
        "loss_decimals_stored": None,
        "phases": summarise_phases(done, PHASES),
        "windows": window_stats(done, WINDOW),
        "memory": memory,
        "per_row": [_serialise_row(r) for r in done],
        "trainable_parameters": info["trainable_parameters"],
    }
    atomic_write_json(out_path, record)
    print(f"wrote {out_path}", flush=True)
    return 0


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def validate_child(record: dict, entry: dict, plan_digest_expected: str,
                   ) -> list[str]:
    """Everything about one child that must hold, checked from raw fields."""
    problems: list[str] = []
    if record.get("schema_version") != EXPERIMENT_SCHEMA:
        problems.append(
            f"{entry['run_id']}: schema_version "
            f"{record.get('schema_version')!r}, expected {EXPERIMENT_SCHEMA}")
    if record.get("kind") != "child":
        problems.append(f"{entry['run_id']}: not a child record")

    for key in CHILD_FORBIDDEN_KEYS:
        if key in record:
            problems.append(
                f"{entry['run_id']}: child wrote `{key}`, which is a "
                "conclusion the parent must reach itself")
    for key in CHILD_REQUIRED_KEYS:
        if key not in record:
            problems.append(f"{entry['run_id']}: missing {key}")

    for key in ("run_id", "block_id", "condition", "position_in_block",
                "global_position", "experiment_id"):
        if record.get(key) != entry.get(key):
            problems.append(
                f"{entry['run_id']}: {key} is {record.get(key)!r} but the "
                f"declared plan says {entry.get(key)!r}")

    if record.get("plan_digest") != plan_digest_expected:
        problems.append(f"{entry['run_id']}: plan digest does not match the "
                        "plan written before the run")

    prov = record.get("provenance") or {}
    for key in REQUIRED_PROVENANCE:
        if key not in prov:
            problems.append(f"{entry['run_id']}: provenance missing {key}")
    if not isinstance(prov.get("working_tree_dirty"), bool):
        problems.append(f"{entry['run_id']}: working_tree_dirty is not a bool")
    if not prov.get("code_sha256"):
        problems.append(f"{entry['run_id']}: no code digests recorded")

    if record.get("loss_decimals_stored") is not None:
        problems.append(f"{entry['run_id']}: losses were stored rounded")

    rows = record.get("per_row") or []
    if len(rows) != record.get("rows_completed"):
        problems.append(f"{entry['run_id']}: rows_completed "
                        f"{record.get('rows_completed')} but stores "
                        f"{len(rows)}")
    if record.get("rows_completed", 0) > MAX_ROWS:
        problems.append(f"{entry['run_id']}: over the {MAX_ROWS}-row cap")

    ids = [r.get("sample_id") for r in rows]
    if all(ids) and digest_ids(ids) != record.get("completed_input_digest"):
        problems.append(f"{entry['run_id']}: per-row sample ids do not digest "
                        "to the stored completed_input_digest")
    if not all(ids):
        problems.append(f"{entry['run_id']}: some rows record no sample_id")

    every = record.get("scheduled_empty_cache_every")
    calls = record.get("scheduled_empty_cache_calls")
    expected = len(rows) // every if every else 0
    if calls != expected:
        problems.append(
            f"{entry['run_id']}: {calls} scheduled clears but the schedule "
            f"over {len(rows)} rows implies {expected}")
    if record.get("condition") == "continuous" and every is not None:
        problems.append(f"{entry['run_id']}: control arm records a clear "
                        "schedule")
    return problems


# --------------------------------------------------------------------------
# replay: calibration, gate, cross-run comparison, treatment contract
# --------------------------------------------------------------------------

def _slot_run_id(slot: dict) -> str:
    return slot.get("run_id") or (slot.get("entry") or {}).get("run_id", "?")


def _slot_record(slot: dict) -> dict | None:
    return slot.get("record")


def replay_calibration(calib: dict) -> dict:
    """Recompute the thresholds from the samples they were derived from.

    The stored thresholds are the numbers every run was judged against, and
    they are four floats in a file that stayed writable for the whole
    experiment. Recomputing them from the samples is the only way the file can
    be said to agree with itself; the digest is checked too, but a digest
    computed over already-edited numbers agrees with nothing.
    """
    problems: list[str] = []
    for key in REQUIRED_CALIBRATION_KEYS:
        if key not in (calib or {}):
            problems.append(f"calibration record is missing {key}")
    if (calib or {}).get("kind") != "preflight_calibration":
        problems.append("calibration file is not a calibration record")
    if (calib or {}).get("loads_model") is not False:
        problems.append("calibration does not record that it loaded no model")

    samples = (calib or {}).get("samples") or []
    if not samples:
        problems.append("calibration stores no samples, so nothing about its "
                        "thresholds can be rechecked")
    stats = calibrate(samples) if samples else {}
    thresholds = thresholds_from(stats) if stats else {}
    gate_policy = (calib or {}).get("gate") or {}
    digest = digest_obj({"stats": stats, "thresholds": thresholds,
                         "gate": gate_policy})

    if samples and stats != (calib or {}).get("stats"):
        problems.append("the stored calibration statistics do not follow from "
                        "the stored samples")
    if samples and thresholds != (calib or {}).get("thresholds"):
        problems.append("the stored thresholds do not follow from the stored "
                        "samples")
    if samples and digest != (calib or {}).get("calibration_digest"):
        problems.append("the calibration digest does not match the stats, "
                        "thresholds and gate policy stored with it")
    for metric in GATE_METRICS:
        if thresholds.get(metric) is None:
            problems.append(f"no calibrated threshold for {metric}")
    for key in ("poll_seconds", "max_wait_seconds",
                "consecutive_passes_required"):
        v = gate_policy.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            problems.append(f"gate policy {key} is {v!r}, not a positive "
                            "integer")
    return {"problems": problems, "stats": stats, "thresholds": thresholds,
            "gate_policy": gate_policy, "calibration_digest": digest,
            "samples_seen": len(samples)}


def replay_gate(slot: dict, thresholds: dict, gate_policy: dict) -> dict:
    """Re-judge every poll, then re-derive the streak, the wait and the verdict.

    The gate is the only evidence that two runs started from comparable
    machine states, and it is stored as a verdict next to the readings it was
    reached from. Reading the verdict back is not checking it. Each sample goes
    through ``evaluate_gate`` again, the streak is counted again, and the
    release or the timeout has to fall out of the polls rather than be asserted
    beside them.
    """
    run_id = _slot_run_id(slot)
    problems: list[str] = []
    for key in ("gate", "recovery_passed_recomputed"):
        if key not in slot:
            problems.append(
                f"{run_id}: the aggregate records no `{key}`, so whether the "
                "machine had recovered cannot be recomputed")

    gate = slot.get("gate")
    if not isinstance(gate, dict):
        return {"run_id": run_id, "passed": None, "polls": [],
                "waited_seconds": None, "timed_out": None,
                "problems": problems + [f"{run_id}: no gate record"]}
    for key in REQUIRED_GATE_KEYS:
        if key not in gate:
            problems.append(f"{run_id}: gate record is missing `{key}`")

    needed = gate_policy.get("consecutive_passes_required")
    poll_seconds = gate_policy.get("poll_seconds")
    max_wait = gate_policy.get("max_wait_seconds")
    if (gate.get("consecutive_passes_required") is not None
            and gate.get("consecutive_passes_required") != needed):
        problems.append(
            f"{run_id}: the gate ran with "
            f"{gate.get('consecutive_passes_required')} consecutive passes "
            f"required, but the calibration policy says {needed}")

    polls = gate.get("polls")
    if not isinstance(polls, list) or not polls:
        # A run nobody reached has no polls because no gate was run, which is
        # absence rather than a missing record. A run that was gated and has
        # no polls is the second thing, and fails.
        if slot.get("never_attempted") and slot.get("not_run"):
            return {"run_id": run_id, "passed": False, "polls": [],
                    "waited_seconds": None, "timed_out": False,
                    "released_at_poll": None, "never_attempted": True,
                    "problems": problems}
        problems.append(f"{run_id}: the gate recorded no polls")
        polls = []

    streak = 0
    released_at = None
    recomputed = []
    previous_elapsed = None
    for i, p in enumerate(polls, 1):
        if not isinstance(p, dict):
            problems.append(f"{run_id}: poll {i} is not a record")
            continue
        for key in REQUIRED_POLL_KEYS:
            if key not in p:
                problems.append(f"{run_id}: poll {i} is missing `{key}`")
        if p.get("poll") not in (None, i):
            problems.append(f"{run_id}: poll {i} is numbered {p.get('poll')}")

        sample = p.get("sample")
        if not isinstance(sample, dict):
            problems.append(f"{run_id}: poll {i} kept no readings, so its "
                            "verdict cannot be recomputed")
            continue
        verdict = evaluate_gate(sample, thresholds)
        if p.get("passed") != verdict["passed"]:
            problems.append(
                f"{run_id}: poll {i} is recorded as passed="
                f"{p.get('passed')}, but its own readings evaluate to "
                f"{verdict['passed']} against the calibrated thresholds")
        if sorted(p.get("failed_metrics") or []) != verdict["failed"]:
            problems.append(
                f"{run_id}: poll {i} lists failed metrics "
                f"{sorted(p.get('failed_metrics') or [])}, recomputed "
                f"{verdict['failed']}")
        streak = streak + 1 if verdict["passed"] else 0
        if p.get("consecutive_passes") != streak:
            problems.append(
                f"{run_id}: poll {i} records a streak of "
                f"{p.get('consecutive_passes')}, recomputed {streak}")

        elapsed = p.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            if previous_elapsed is not None:
                gap = elapsed - previous_elapsed
                if gap < 0:
                    problems.append(f"{run_id}: poll {i} is timestamped before "
                                    "the poll before it")
                elif isinstance(poll_seconds, (int, float)) and (
                        gap < poll_seconds * 0.9):
                    problems.append(
                        f"{run_id}: poll {i} came {round(gap, 2)}s after the "
                        f"one before it, sooner than the {poll_seconds}s the "
                        "policy sleeps between polls")
            previous_elapsed = elapsed

        recomputed.append({"poll": i, "passed": verdict["passed"],
                           "failed_metrics": verdict["failed"],
                           "consecutive_passes": streak,
                           "elapsed_seconds": elapsed})
        if released_at is None and isinstance(needed, int) and streak >= needed:
            released_at = i

    if released_at is not None and released_at != len(recomputed):
        problems.append(
            f"{run_id}: the gate kept polling after it had already been inside "
            f"the band {needed} times running (at poll {released_at} of "
            f"{len(recomputed)})")

    passed = released_at is not None
    waited = previous_elapsed
    timed_out = (not passed and isinstance(waited, (int, float))
                 and isinstance(max_wait, (int, float))
                 and waited + 0.005 >= max_wait)
    if not passed and polls and not timed_out:
        problems.append(
            f"{run_id}: the gate stopped after {waited}s without either "
            f"passing or reaching its own {max_wait}s deadline")
    if gate.get("passed") is not None and bool(gate.get("passed")) != passed:
        problems.append(
            f"{run_id}: the gate is recorded as passed={gate.get('passed')}, "
            f"but replaying its polls gives {passed}")
    if (isinstance(gate.get("waited_seconds"), (int, float))
            and isinstance(waited, (int, float))
            and abs(gate["waited_seconds"] - waited) > 0.05):
        problems.append(
            f"{run_id}: the gate records {gate['waited_seconds']}s waited, but "
            f"its last poll is at {waited}s")
    if passed and gate.get("reason") is not None:
        problems.append(f"{run_id}: the gate passed but records a reason for "
                        "stopping")
    if not passed and polls and not gate.get("reason"):
        problems.append(f"{run_id}: the gate did not pass but records no "
                        "reason")

    claimed = slot.get("recovery_passed_recomputed")
    if claimed is not None and bool(claimed) != passed:
        problems.append(
            f"{run_id}: the aggregate says the machine had "
            f"{'' if claimed else 'not '}recovered, replaying the polls says "
            f"{'' if passed else 'not '}recovered")
    if passed and slot.get("not_run"):
        problems.append(f"{run_id}: the gate passed, yet the run is recorded "
                        "as never started")
    if not passed and not slot.get("not_run"):
        problems.append(f"{run_id}: the gate did not pass, yet the run is not "
                        "recorded as skipped")

    return {"run_id": run_id, "passed": passed, "polls": recomputed,
            "waited_seconds": waited, "timed_out": timed_out,
            "released_at_poll": released_at, "problems": problems}


def check_child_preflight(record: dict | None, thresholds: dict,
                          run_id: str) -> dict:
    """The child's own reading, judged against the same band as the gate.

    The gate polls from the parent and then hands over; the child samples once
    more, in its own process, immediately before it loads anything. If that
    reading is outside the band the run started somewhere the gate would not
    have let it start, whatever the polls said a moment earlier.
    """
    if record is None:
        return {"run_id": run_id, "checked": False, "passed": None,
                "problems": []}
    sample = record.get("preflight")
    if not isinstance(sample, dict):
        return {"run_id": run_id, "checked": False, "passed": None,
                "problems": [f"{run_id}: the child recorded no preflight "
                             "reading of its own"]}
    verdict = evaluate_gate(sample, thresholds)
    problems = []
    if not verdict["passed"]:
        problems.append(
            f"{run_id}: the child's own preflight reading is outside the "
            f"calibrated band on {', '.join(verdict['failed'])}, so it started "
            "from a machine state the gate would have refused")
    return {"run_id": run_id, "checked": True, "passed": verdict["passed"],
            "failed_metrics": verdict["failed"], "sample": sample,
            "problems": problems}


def compare_children(children: list[dict]) -> dict:
    """Every field that must be identical across runs, compared across runs.

    The condition and its clear schedule are the one thing that varies. Nothing
    else may: not the code, not the rows, not the order, not the revisions, not
    the measurement intervals. Two runs that differ anywhere else are two
    different experiments reported as one.
    """
    measured = [(_slot_run_id(c), _slot_record(c)) for c in children
                if _slot_record(c) is not None]
    run_ids = [r for r, _ in measured]
    fields: dict[str, dict] = {}

    def collect(name: str, getter) -> None:
        values = {run: getter(rec) for run, rec in measured}
        distinct = {canonical(v) for v in values.values()}
        agrees = len(distinct) <= 1
        fields[name] = ({"agrees": True,
                         "value": next(iter(values.values()), None)}
                        if agrees else {"agrees": False, "values": values})

    for field in COMPARED_PROVENANCE_FIELDS:
        collect(field, lambda rec, f=field: (rec.get("provenance") or {}).get(f))
    for field in COMPARED_RECORD_FIELDS:
        collect(field, lambda rec, f=field: rec.get(f))

    disagreements = sorted(f for f, v in fields.items() if not v["agrees"])

    completed_declared: dict[str, dict] = {}
    for run, rec in measured:
        declared = rec.get("input_order_digest")
        completed = rec.get("completed_input_digest")
        completed_declared[run] = {
            "declared": declared, "completed": completed,
            "matches": bool(declared) and declared == completed,
            "rows_completed": rec.get("rows_completed"),
            "rows_requested": rec.get("rows_requested")}

    problems = [f"runs disagree on {f}" for f in disagreements]
    problems += [f"{run}: the rows it completed do not digest to the full "
                 "declared input order"
                 for run, v in completed_declared.items() if not v["matches"]]
    return {"compared_runs": run_ids,
            "comparable": len(run_ids) >= 2,
            "note": ("Only one run was measured, so nothing was compared "
                     "across runs." if len(run_ids) == 1 else
                     "No run was measured." if not run_ids else
                     "Every field below had to be identical in every measured "
                     "run; condition and clear schedule are excluded because "
                     "they are what the experiment varies."),
            "fields": fields,
            "disagreements": disagreements,
            "completed_declared_order": completed_declared,
            "problems": problems}


def code_on_disk_check(children: list[dict]) -> dict:
    """Is the code each run recorded still the code in the working tree?

    A digest recorded before the model loaded says what ran. It does not
    preserve it: the tree stays open for editing between runs, and once a file
    has moved on, the version that produced a measurement is gone unless
    something kept a copy. That is not a failure -- development continues --
    but a report that does not say it invites the reader to assume the run can
    be reproduced from what is on disk today.

    The session flow keeps a snapshot precisely so this answer stops being
    "gone"; runs made before it existed can only be described honestly.
    """
    out: dict[str, dict] = {}
    for slot in children:
        rec = _slot_record(slot)
        if rec is None:
            continue
        recorded = (rec.get("provenance") or {}).get("code_sha256") or {}
        same, differs, missing = [], [], []
        for rel, digest in sorted(recorded.items()):
            live = ROOT / rel
            if not live.exists():
                missing.append(rel)
            elif sha256_file(live) == digest:
                same.append(rel)
            else:
                differs.append({"file": rel, "ran_with": digest,
                                "on_disk_now": sha256_file(live)})
        out[_slot_run_id(slot)] = {
            "files_recorded": len(recorded), "unchanged": same,
            "changed_since_the_run": differs, "missing_from_the_tree": missing,
            "source_preserved": False if (differs or missing) else True}
    return out


def check_treatment_contract(record: dict | None, run_id: str) -> dict:
    """Did each arm do exactly what its condition says, and nothing else?

    Two failures matter here and neither shows up in a total. A control arm
    that cleared once has quietly become a treatment arm. A treatment arm whose
    clears landed somewhere other than every tenth row was not on the schedule
    the report describes. So the row numbers are checked, not just the count,
    and the per-call costs have to add up to the total they are summarised as.
    """
    if record is None:
        return {"run_id": run_id, "checked": False, "problems": []}
    condition = record.get("condition")
    rows = record.get("per_row") or []
    every = record.get("scheduled_empty_cache_every")
    calls = record.get("scheduled_empty_cache_calls")
    cost = record.get("scheduled_empty_cache_cost") or {}
    per_call = cost.get("per_call_seconds")
    problems: list[str] = []

    cleared_rows = [r.get("row") for r in rows
                    if (r.get("scheduled_empty_cache_seconds") or 0) > 0]
    row_numbers = [r.get("row") for r in rows]

    if condition == "continuous":
        expected_rows: list = []
        if every is not None:
            problems.append(f"{run_id}: the control arm records a clear "
                            f"schedule of every {every} rows")
        if calls:
            problems.append(f"{run_id}: the control arm performed {calls} "
                            "scheduled clears")
        if cleared_rows:
            problems.append(f"{run_id}: the control arm cleared at rows "
                            f"{cleared_rows[:5]}")
        if cost.get("total_seconds"):
            problems.append(f"{run_id}: the control arm records "
                            f"{cost.get('total_seconds')}s of scheduled clear "
                            "cost")
    elif condition == "empty_cache":
        if every != EMPTY_CACHE_EVERY:
            problems.append(f"{run_id}: the treatment arm records a schedule "
                            f"of every {every} rows, not every "
                            f"{EMPTY_CACHE_EVERY}")
        expected_rows = [n for n in row_numbers
                         if isinstance(n, int) and every and n % every == 0]
        if cleared_rows != expected_rows:
            missing = sorted(set(expected_rows) - set(cleared_rows))[:5]
            extra = sorted(set(cleared_rows) - set(expected_rows))[:5]
            problems.append(
                f"{run_id}: clears landed on {len(cleared_rows)} rows, the "
                f"schedule calls for {len(expected_rows)}"
                + (f"; missing at {missing}" if missing else "")
                + (f"; unscheduled at {extra}" if extra else ""))
        if calls != len(expected_rows):
            problems.append(f"{run_id}: {calls} clears counted, "
                            f"{len(expected_rows)} scheduled over "
                            f"{len(rows)} rows")
    else:
        expected_rows = []
        problems.append(f"{run_id}: unknown condition {condition!r}")

    if cost.get("calls") != calls:
        problems.append(f"{run_id}: the clear cost record counts "
                        f"{cost.get('calls')} calls, the run counts {calls}")
    if isinstance(per_call, list):
        if len(per_call) != (calls or 0):
            problems.append(f"{run_id}: {len(per_call)} per-call clear costs "
                            f"for {calls} clears")
        total = cost.get("total_seconds")
        if isinstance(total, (int, float)) and abs(
                sum(per_call) - total) > CLEAR_SUM_TOLERANCE_SECONDS:
            problems.append(f"{run_id}: per-call clear costs sum to "
                            f"{round(sum(per_call), 6)}, recorded total "
                            f"{total}")
        if per_call and cost.get("max_seconds") != max(per_call):
            problems.append(f"{run_id}: the largest per-call clear cost is "
                            f"{max(per_call)}, recorded max "
                            f"{cost.get('max_seconds')}")
    else:
        problems.append(f"{run_id}: no per-call clear costs recorded")

    row_clear_sum = sum(r.get("scheduled_empty_cache_seconds") or 0
                        for r in rows)
    if isinstance(cost.get("total_seconds"), (int, float)) and abs(
            row_clear_sum - cost["total_seconds"]) > SUM_TOLERANCE_SECONDS:
        problems.append(f"{run_id}: the per-row clear seconds sum to "
                        f"{round(row_clear_sum, 6)}, the run records "
                        f"{cost['total_seconds']}")

    # The teardown clear happens after the condition's clock has stopped. What
    # shows that is arithmetic: the timed spans and the recorded totals have to
    # add up without it, and the slack left over is far smaller than it is.
    teardown_calls = record.get("teardown_empty_cache_calls")
    teardown_seconds = record.get("teardown_empty_cache_seconds")
    if teardown_calls != 1:
        problems.append(f"{run_id}: {teardown_calls} teardown clears, expected "
                        "exactly one")
    if not isinstance(teardown_seconds, (int, float)) or teardown_seconds < 0:
        problems.append(f"{run_id}: teardown clear time is "
                        f"{teardown_seconds!r}")
    breakdown = record.get("between_row_overhead_breakdown") or {}
    if any("teardown" in k for k in breakdown):
        problems.append(f"{run_id}: the between-row overhead breakdown counts "
                        "a teardown clear")

    compute = record.get("model_compute_seconds")
    end_to_end = record.get("end_to_end_seconds")
    overhead = record.get("between_row_overhead_seconds")
    row_span_sum = sum(r.get("end_to_end") or 0 for r in rows)
    row_compute_sum = sum(r.get("total") or 0 for r in rows)
    gap = None
    if all(isinstance(v, (int, float)) for v in (compute, end_to_end, overhead)):
        if abs((end_to_end - compute) - overhead) > SUM_TOLERANCE_SECONDS:
            problems.append(
                f"{run_id}: end-to-end minus compute is "
                f"{round(end_to_end - compute, 4)}, recorded overhead "
                f"{overhead}")
        parts = [v for v in breakdown.values() if isinstance(v, (int, float))]
        if parts and abs(sum(parts) - overhead) > SUM_TOLERANCE_SECONDS:
            problems.append(f"{run_id}: the overhead breakdown sums to "
                            f"{round(sum(parts), 4)}, recorded {overhead}")
        if abs(row_compute_sum - compute) > SUM_TOLERANCE_SECONDS:
            problems.append(f"{run_id}: the per-row timed spans sum to "
                            f"{round(row_compute_sum, 4)}, recorded compute "
                            f"{compute}")
        gap = round(end_to_end - row_span_sum, 6)
        allowed = row_span_tolerance(len(rows))
        if gap < -SUM_TOLERANCE_SECONDS or gap > allowed:
            problems.append(
                f"{run_id}: {gap}s of the condition's clock falls outside the "
                f"per-row spans, more than the {round(allowed, 4)}s the loop "
                "itself accounts for")
        elif (isinstance(teardown_seconds, (int, float)) and teardown_seconds
                and gap >= teardown_seconds):
            problems.append(
                f"{run_id}: the {gap}s outside the per-row spans is as large "
                f"as the {teardown_seconds}s teardown clear, so the teardown "
                "cannot be shown to be outside the condition's clock")

    probe_sum = sum(r.get("memory_probe_seconds") or 0 for r in rows)
    recorded_probe = breakdown.get("memory_probe_seconds")
    if isinstance(recorded_probe, (int, float)) and abs(
            probe_sum - recorded_probe) > SUM_TOLERANCE_SECONDS:
        problems.append(f"{run_id}: per-row memory probes sum to "
                        f"{round(probe_sum, 4)}, recorded {recorded_probe}")

    return {"run_id": run_id, "condition": condition, "checked": True,
            "scheduled_every": every, "clears_expected_at": expected_rows,
            "clears_observed_at": cleared_rows,
            "clear_calls": calls,
            "clear_total_seconds": cost.get("total_seconds"),
            "clear_seconds_from_rows": round(row_clear_sum, 6),
            "teardown_calls": teardown_calls,
            "teardown_seconds": teardown_seconds,
            "seconds_outside_row_spans": gap,
            "teardown_outside_condition_clock": (
                None if gap is None or not isinstance(
                    teardown_seconds, (int, float))
                else gap < teardown_seconds),
            "problems": problems}


def block_directions(experiment: dict) -> dict:
    """Which arm was faster in each block, from the last window of each run."""
    kids = {_slot_run_id(c): _summarise(_slot_record(c))
            for c in experiment["children"]}
    plan_by_run = {p["run_id"]: p for p in experiment["plan"]}
    out: dict[str, dict] = {}
    for block in sorted({p["block_id"] for p in experiment["plan"]}):
        runs = [r for r, p in plan_by_run.items() if p["block_id"] == block]
        pick = lambda cond: next(                              # noqa: E731
            (r for r in runs if plan_by_run[r]["condition"] == cond), None)
        cv = (kids.get(pick("continuous")) or {}).get(
            "last_window_seconds_per_row")
        ev = (kids.get(pick("empty_cache")) or {}).get(
            "last_window_seconds_per_row")
        if cv is None or ev is None:
            direction = "incomplete"
        elif ev < cv:
            direction = "empty_cache faster"
        elif ev > cv:
            direction = "continuous faster"
        else:
            direction = "equal"
        out[block] = {"continuous_seconds_per_row": cv,
                      "empty_cache_seconds_per_row": ev,
                      "direction": direction}
    directions = [v["direction"] for v in out.values()]
    return {"blocks": out, "directions": directions,
            "consistent": len(set(directions)) == 1 and "incomplete"
            not in directions}


def headline_gate(experiment: dict, comparison: dict, contracts: dict,
                  losses: dict, verdicts: dict, directions: dict) -> dict:
    """What has to hold before the report is allowed to state a result.

    Every clause here is a way the contrast could be an artefact rather than an
    effect. None of them is a formality: report 14 satisfied none of them and
    still read as though it had found something.
    """
    plan = experiment["plan"]
    kids = experiment["children"]
    reasons: list[str] = []

    measured = [c for c in kids if _slot_record(c) is not None]
    if len(kids) != len(plan) or len(measured) != len(plan):
        reasons.append(f"{len(measured)} of {len(plan)} runs produced a report")
    elif any(c.get("exit_status") != 0 for c in kids):
        reasons.append("not every run exited cleanly")
    if experiment.get("stopped_reason"):
        reasons.append(f"the experiment stopped early: "
                       f"{experiment['stopped_reason']}")

    ineligible = sorted(r for r, v in verdicts.items()
                        if not v["eligible_for_paired_contrast"])
    if len(verdicts) != len(plan) or ineligible:
        reasons.append("not every run is eligible for the paired contrast"
                       + (f" ({', '.join(ineligible)})" if ineligible else ""))

    expected_pairs = len(plan) * (len(plan) - 1) // 2
    if losses.get("comparable_pairs") != expected_pairs:
        reasons.append(f"{losses.get('comparable_pairs')} comparable pairs, "
                       f"expected {expected_pairs}")
    elif any(p.get("shared_rows") != MAX_ROWS for p in losses.get("pairs", [])):
        reasons.append(f"not every pair shares all {MAX_ROWS} rows")
    if losses.get("token_count_mismatches_total"):
        reasons.append("token or supervised-token counts differ between runs")
    if losses.get("verdict") != "within_tolerance":
        reasons.append(f"the loss verdict is {losses.get('verdict')}")

    if not directions["consistent"]:
        reasons.append("the two blocks do not agree on the direction")
    if comparison.get("problems"):
        reasons.append("the runs did not all measure the same thing: "
                       + "; ".join(comparison["problems"][:3]))
    contract_problems = [p for c in contracts.values()
                         for p in c.get("problems", [])]
    if contract_problems:
        reasons.append("a run did not honour its treatment contract: "
                       + "; ".join(contract_problems[:3]))

    # Four runs in one boot is the design exp001 disproved. Distinct boots are
    # not a formality here; they are the only reason the runs are comparable.
    fingerprints = [c.get("boot_fingerprint") for c in measured]
    if len(set(fingerprints)) != len(plan) or not all(fingerprints):
        named = len([f for f in fingerprints if f])
        reasons.append(
            f"the runs did not each start from their own boot "
            f"({len(set(f for f in fingerprints if f))} distinct boots across "
            f"{named} runs that recorded one, out of {len(plan)})")

    return {"allowed": not reasons, "reasons": reasons,
            "distinct_boots": len(set(f for f in fingerprints if f)),
            "requirements": list(HEADLINE_REQUIREMENTS),
            "direction": (directions["directions"][0]
                          if directions["consistent"] else None)}


def analyse(experiment: dict, calib: dict) -> dict:
    """Re-derive every conclusion from raw fields, ignoring anything claimed.

    One path, used by the parent as it finishes, by ``--verify``, by
    ``--from-json`` and by ``--recompute``. A verifier that re-derives less
    than the writer did is a verifier that agrees with the writer by
    construction.
    """
    exp = dict(experiment)
    calibration = replay_calibration(calib)
    thresholds = calibration["thresholds"]
    problems = list(calibration["problems"])

    gates: dict[str, dict] = {}
    preflights: dict[str, dict] = {}
    for slot in exp["children"]:
        run_id = _slot_run_id(slot)
        for key in REQUIRED_CHILD_SLOT_KEYS:
            if key not in slot and key not in (slot.get("entry") or {}):
                problems.append(f"{run_id}: the aggregate records no `{key}`")
        gate = replay_gate(slot, thresholds, calibration["gate_policy"])
        gates[run_id] = gate
        problems += gate["problems"]
        slot["recovery_passed_replayed"] = gate["passed"]
        pre = check_child_preflight(_slot_record(slot), thresholds, run_id)
        preflights[run_id] = pre
        problems += pre["problems"]

    comparison = compare_children(exp["children"])
    contracts = {_slot_run_id(c): check_treatment_contract(
        _slot_record(c), _slot_run_id(c)) for c in exp["children"]}
    problems += comparison["problems"]
    problems += [p for c in contracts.values() for p in c.get("problems", [])]

    exp["calibration_replay"] = calibration
    exp["thresholds_recomputed"] = thresholds
    exp["gate_replay"] = gates
    exp["child_preflight_checks"] = preflights
    exp["code_on_disk"] = code_on_disk_check(exp["children"])
    exp["cross_child_comparison"] = comparison
    exp["treatment_contract"] = contracts
    exp["verdicts"] = recompute_verdicts(exp)
    exp["losses"] = compare_losses(exp["children"])
    exp["block_directions"] = block_directions(exp)
    exp["headline"] = headline_gate(exp, comparison, contracts, exp["losses"],
                                    exp["verdicts"], exp["block_directions"])
    exp["replay_problems"] = problems
    return exp


def check_thresholds_agree(agg: dict, plan_doc: dict, calib: dict,
                           recomputed: dict) -> list[str]:
    """Plan, aggregate and calibration have to be talking about one gate."""
    problems: list[str] = []
    stored_digest = (calib or {}).get("calibration_digest")
    for name, doc in (("plan", plan_doc), ("aggregate", agg)):
        if doc.get("calibration_digest") != stored_digest:
            problems.append(
                f"the {name} was written against calibration "
                f"{str(doc.get('calibration_digest'))[:12]}..., the "
                f"calibration file is "
                f"{str(stored_digest)[:12]}...")
    if agg.get("thresholds") != recomputed["thresholds"]:
        problems.append("the aggregate's thresholds are not the ones the "
                        "calibration samples produce")
    if agg.get("gate_policy") != recomputed["gate_policy"]:
        problems.append("the aggregate's gate policy differs from the "
                        "calibration's")
    return problems


def validate_experiment(agg: dict, plan_doc: dict, exp_dir: Path,
                        calib: dict, *,
                        terminal_outcomes: dict | None = None) -> list[str]:
    """Fail-closed check of a stored experiment before anything re-reads it.

    The aggregate references its children by path and digest rather than
    copying them, which keeps it small but means the aggregate alone proves
    nothing: a child file edited afterwards would leave the aggregate looking
    untouched. So every child is re-hashed against the digest recorded at the
    time, and then re-validated in full.
    """
    problems: list[str] = []
    if agg.get("schema_version") != EXPERIMENT_SCHEMA:
        problems.append(f"aggregate schema_version is "
                        f"{agg.get('schema_version')!r}, expected "
                        f"{EXPERIMENT_SCHEMA}")
    if agg.get("kind") != "experiment":
        problems.append("aggregate is not an experiment record")

    plan = plan_doc.get("plan")
    if not plan:
        return problems + ["plan file records no plan"]
    if digest_obj(plan) != plan_doc.get("plan_digest"):
        problems.append("the plan file's own digest does not match its plan")
    if agg.get("plan_digest") != plan_doc.get("plan_digest"):
        problems.append("aggregate and plan file disagree on the plan digest")
    if agg.get("plan") != plan:
        problems.append("aggregate carries a different plan from the plan file")

    children = agg.get("children") or []
    if len(children) > len(plan):
        problems.append(f"{len(children)} children recorded for a "
                        f"{len(plan)}-run plan")
    for i, c in enumerate(children[:len(plan)]):
        entry = plan[i]
        for key in ("run_id", "condition", "block_id", "global_position"):
            if c.get(key) != entry.get(key):
                problems.append(
                    f"child {i} records {key}={c.get(key)!r}, but the plan "
                    f"says {entry.get(key)!r}")
        if c.get("not_run"):
            if c.get("report_path") or c.get("report_sha256"):
                problems.append(f"{entry['run_id']}: never started, yet a "
                                "report is referenced")
            continue
        rel = c.get("report_path")
        if not rel:
            # A run that started and produced nothing is a real state, but only
            # when the journal says which of the four ways it happened. Without
            # that evidence -- a legacy experiment, or a session whose journal
            # says otherwise -- a missing report is still a broken record.
            outcome = (terminal_outcomes or {}).get(entry["run_id"])
            if outcome in TERMINAL_OUTCOMES:
                continue
            problems.append(f"{entry['run_id']}: ran but references no report")
            continue
        path = ROOT / rel
        if not path.exists():
            problems.append(f"{entry['run_id']}: child report {rel} is missing")
            continue
        if sha256_file(path) != c.get("report_sha256"):
            problems.append(
                f"{entry['run_id']}: child report has changed since the run "
                "(digest does not match the one recorded then)")
            continue
        try:
            record = json.loads(path.read_text())
        except Exception as exc:
            problems.append(f"{entry['run_id']}: report is unreadable ({exc})")
            continue
        problems += validate_child(record, entry, agg["plan_digest"])

    for i, c in enumerate(children[:len(plan)]):
        for key in REQUIRED_CHILD_SLOT_KEYS:
            if key not in c:
                problems.append(
                    f"{plan[i]['run_id']}: the aggregate records no `{key}`, "
                    "so the run cannot be replayed")

    problems += check_thresholds_agree(agg, plan_doc, calib,
                                       replay_calibration(calib))
    problems += _check_stored_losses(agg)

    # `complete` is a conclusion, so it has to follow from what is on disk.
    implied = (len(children) == len(plan)
               and all(c.get("exit_status") == 0 for c in children)
               and agg.get("stopped_reason") is None)
    if bool(agg.get("complete")) != implied:
        problems.append(
            f"aggregate says complete={agg.get('complete')}, but the children "
            f"and stop reason recorded imply {implied}")
    return problems


def _check_stored_losses(agg: dict) -> list[str]:
    """A stored loss verdict has to agree with the pairs stored beside it.

    "No pair to compare" and "compared and passed" are different states, and
    the first one written as the second is how an experiment that measured
    nothing comes to read as an experiment that agreed with itself.

    Only the current three-state shape is checked here. An aggregate written
    before that shape existed is not a contradiction, it is old; it is reported
    by ``stored_conclusion_notes`` and superseded by the recomputation rather
    than treated as tampering.
    """
    losses = agg.get("losses")
    if not isinstance(losses, dict) or "verdict" not in losses:
        return []
    problems = []
    pairs = losses.get("pairs")
    pairs = pairs if isinstance(pairs, list) else []
    if losses.get("comparable_pairs") != len(pairs):
        problems.append(
            f"the aggregate counts {losses.get('comparable_pairs')} comparable "
            f"pairs but stores {len(pairs)}")
    if not pairs:
        if losses.get("verdict") != "not_applicable":
            problems.append(
                f"the aggregate has no comparable pair yet records a loss "
                f"verdict of {losses.get('verdict')!r}")
        if losses.get("max_abs_loss_diff_overall") is not None:
            problems.append("the aggregate has no comparable pair yet records "
                            "a largest loss difference")
    elif losses.get("verdict") == "not_applicable":
        problems.append(f"the aggregate stores {len(pairs)} comparable pairs "
                        "yet records no loss verdict")
    return problems


def stored_conclusion_notes(agg: dict) -> list[str]:
    """Conclusions in the stored aggregate that the recomputation replaces.

    These are not measurements and nothing downstream reads them: every number
    the report states is derived again from the child records, which are pinned
    by digest. They are surfaced because deleting the record of what was once
    concluded would be a worse answer than printing it next to what replaced
    it.
    """
    notes: list[str] = []
    losses = agg.get("losses")
    if isinstance(losses, dict) and "verdict" not in losses:
        pairs = losses.get("pairs")
        n = len(pairs) if isinstance(pairs, list) else "an unknown number of"
        notes.append(
            f"the parent's aggregate stores the older two-state loss field "
            f"`within_tolerance`={losses.get('within_tolerance')!r} over {n} "
            "comparable pairs. With no pair to compare, that field reads as "
            "'compared and failed' when nothing was compared. It is superseded "
            "here by the three-state verdict, which is recomputed from the "
            "child records rather than read back.")
    return notes


def load_calibration(plan_doc: dict) -> tuple[Path | None, dict, list[str]]:
    """The calibration the plan names -- not whichever one is lying around.

    The thresholds are only meaningful attached to the samples they came from,
    and the plan is where that link was written down before the first run.
    """
    rel = plan_doc.get("calibration_path")
    if not rel:
        return None, {}, ["the plan names no calibration file, so the gate "
                          "thresholds cannot be rechecked"]
    path = ROOT / rel
    if not path.exists():
        return path, {}, [f"the calibration the plan names ({rel}) is missing"]
    try:
        return path, json.loads(path.read_text()), []
    except Exception as exc:
        return path, {}, [f"the calibration file is unreadable ({exc})"]


def load_experiment(exp_dir: Path) -> dict:
    """Rebuild the full experiment from disk, refusing anything inconsistent.

    Fail-closed throughout: a check that cannot be carried out is a check that
    did not pass. Nothing here trusts a stored conclusion, including the
    parent's own -- the thresholds are recomputed from the calibration samples,
    every gate poll is re-judged against them, and the verdicts are derived
    again from the raw fields.
    """
    exp_dir = Path(exp_dir)
    agg_path, plan_path = exp_dir / "aggregate.json", exp_dir / "plan.json"
    for p in (agg_path, plan_path):
        if not p.exists():
            raise SystemExit(f"{p} does not exist")
    agg = json.loads(agg_path.read_text())
    plan_doc = json.loads(plan_path.read_text())
    _, calib, calib_problems = load_calibration(plan_doc)

    # A one-run-per-boot session keeps its evidence in the journal, and whether
    # this *is* one is answered from the directory rather than from a flag in
    # the file being checked. The state machine runs first, because what it
    # proves -- which runs started, and how the ones without a report ended --
    # is what the aggregate is then allowed to say.
    journal = None
    if looks_like_a_session(exp_dir, agg):
        journal = replay_session_journal(exp_dir, agg)
        if journal["problems"]:
            raise SystemExit(
                "this session's journal does not support its aggregate:\n  - "
                + "\n  - ".join(journal["problems"])
                + "\nThe aggregate is a summary of these events; where they "
                  "disagree, the events are what happened.")

    problems = calib_problems + validate_experiment(
        agg, plan_doc, exp_dir, calib,
        terminal_outcomes=(journal or {}).get("terminal_outcomes"))
    if problems:
        raise SystemExit("cannot replay this experiment:\n  - "
                         + "\n  - ".join(problems)
                         + "\nRe-run it rather than re-rendering it.")

    plan = plan_doc["plan"]
    children = []
    for i, c in enumerate(agg["children"]):
        record = None
        if not c.get("not_run") and c.get("report_path"):
            record = json.loads((ROOT / c["report_path"]).read_text())
        children.append({**c, "entry": plan[i], "record": record})

    if journal is not None:
        # The headline rests on the fingerprints out of the events, never on
        # the summary in the aggregate.
        state = journal["state"]
        for slot in children:
            begin = state["started"].get(_slot_run_id(slot))
            slot["boot_fingerprint"] = (begin["body"].get("boot_fingerprint")
                                        if begin else None)
        agg = {**agg, "boot_fingerprints_source": "journal"}

    exp = analyse({**agg, "children": children}, calib)
    if journal is not None:
        exp["journal_replay"] = {
            "events_replayed": len(journal["state"]["events"]),
            "completed": journal["state"]["completed"],
            "terminal": journal["state"]["terminal"],
            "terminal_reason": journal["state"]["terminal_reason"],
            "boot_fingerprints": journal["boot_fingerprints"],
            "snapshot_verified_without_the_working_tree": True}
    if exp["replay_problems"]:
        raise SystemExit("cannot replay this experiment:\n  - "
                         + "\n  - ".join(exp["replay_problems"])
                         + "\nRe-run it rather than re-rendering it.")
    exp["superseded_conclusions"] = stored_conclusion_notes(agg)
    exp["calibration"] = {
        "path": plan_doc.get("calibration_path"),
        "sha256": (sha256_file(ROOT / plan_doc["calibration_path"])
                   if plan_doc.get("calibration_path")
                   and (ROOT / plan_doc["calibration_path"]).exists() else None),
        "digest": calib.get("calibration_digest")}
    return exp


def recompute_verdicts(experiment: dict) -> dict:
    """Derive every conclusion from raw fields, ignoring anything claimed.

    Ignoring the parent's conclusions too, not only the child's: the recovery
    verdict used here is the one replayed from the gate's own polls when there
    is one, and a run with no recomputed recovery verdict at all is ineligible
    rather than assumed fine.
    """
    children = experiment["children"]
    plan = experiment["plan"]
    contracts = experiment.get("treatment_contract") or {}

    ref = children[0]["record"] if children else None
    verdicts = {}
    for slot, entry in zip(children, plan):
        rec = slot["record"]
        reasons = []
        if slot["exit_status"] != 0:
            reasons.append(f"exit status {slot['exit_status']}")
        if rec is None:
            reasons.append("no child report")
        else:
            if rec.get("stopped_early"):
                reasons.append(f"stopped early: {rec['stopped_early']}")
            if rec.get("rows_completed") != MAX_ROWS:
                reasons.append(
                    f"completed {rec.get('rows_completed')} of {MAX_ROWS} rows")
            if rec.get("completed_input_digest") != rec.get(
                    "input_order_digest"):
                reasons.append("the rows it completed are not the full "
                               "declared input order")
            if ref is not None and rec.get("input_order_digest") != ref.get(
                    "input_order_digest"):
                reasons.append("input order differs from the first run")
            if ref is not None:
                for key in COMPARED_PROVENANCE_FIELDS:
                    if (rec["provenance"].get(key)
                            != ref["provenance"].get(key)):
                        reasons.append(f"provenance differs on {key}")
            reasons += contracts.get(entry["run_id"], {}).get("problems", [])

        recovered = slot.get("recovery_passed_replayed")
        if recovered is None:
            recovered = slot.get("recovery_passed_recomputed")
        if recovered is False:
            reasons.append("machine had not recovered before this run started")
        elif recovered is not True:
            reasons.append("no recomputed recovery verdict for this run")
        verdicts[entry["run_id"]] = {
            "eligible_for_paired_contrast": not reasons,
            "reasons": reasons,
        }
    return verdicts


def compare_losses(children: list[dict]) -> dict:
    """Every pairwise per-row loss difference, kept whole.

    Reported across all four runs rather than per block: dropping one block
    because its losses drifted, while keeping the other, would be choosing
    which evidence to look at after seeing it.
    """
    usable = [c for c in children if c["record"]]
    by_run = {c["record"]["run_id"]:
              {r["sample_id"]: r for r in c["record"]["per_row"]}
              for c in usable}
    names = sorted(by_run)
    pairs = []
    worst = 0.0
    token_mismatches = 0
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ra, rb = by_run[names[a]], by_run[names[b]]
            shared = sorted(set(ra) & set(rb))
            diffs = [abs(ra[s]["loss"] - rb[s]["loss"]) for s in shared]
            tok = sum(1 for s in shared
                      if ra[s]["n_tokens"] != rb[s]["n_tokens"]
                      or ra[s]["n_supervised"] != rb[s]["n_supervised"])
            token_mismatches += tok
            m = max(diffs) if diffs else None
            if m is not None:
                worst = max(worst, m)
            pairs.append({"a": names[a], "b": names[b],
                          "shared_rows": len(shared),
                          "max_abs_loss_diff": m,
                          "mean_abs_loss_diff": (
                              round(sum(diffs) / len(diffs), 12)
                              if diffs else None),
                          "rows_over_tolerance": sum(
                              1 for d in diffs if d > LOSS_TOLERANCE),
                          "token_count_mismatches": tok})
    # Three states, not two. With fewer than two completed runs there is no
    # comparison to pass or fail, and reporting that as "outside tolerance"
    # would invent a disagreement between runs that do not exist.
    if not pairs:
        verdict = "not_applicable"
    elif worst <= LOSS_TOLERANCE:
        verdict = "within_tolerance"
    else:
        verdict = "over_tolerance"
    return {"tolerance": LOSS_TOLERANCE,
            "tolerance_basis": LOSS_TOLERANCE_BASIS,
            "pairs": pairs,
            "comparable_pairs": len(pairs),
            "max_abs_loss_diff_overall": worst if pairs else None,
            "verdict": verdict,
            "token_count_mismatches_total": token_mismatches}


# --------------------------------------------------------------------------
# the withdrawn same-boot parent
# --------------------------------------------------------------------------

RUN_WITHDRAWN = (
    "`--run` is withdrawn permanently. It ran all four conditions inside one "
    "boot, waiting for the machine to come back inside the calibrated band "
    "between them, and exp001 is the evidence that it does not: after one "
    "200-row run the swap reading settled at 1.558GB against a 0.537GB "
    "threshold and stayed there for the whole 15-minute observation window. "
    "The experiment stopped at one run of four and no condition was ever "
    "compared. There is no flag that makes this flow correct, so there is no "
    "flag for it.\n"
    "\n"
    "The approved flow is one measured run per boot:\n"
    "  --session-init  --experiment-id <id>   once, against a fixed baseline\n"
    "  --session-next  --experiment-id <id>   once per boot, four times\n"
    "\n"
    "Between runs the operator restarts the machine. This tool does not "
    "restart it and must not: a tool that reboots the machine it is measuring "
    "has changed the thing it was measuring, and the restart is the whole "
    "reason the next run is comparable.\n"
    "\n"
    "Reading a finished experiment back is separate and read-only: --verify, "
    "--recompute and --from-json. exp001's records are kept exactly as they "
    "were.")


def run_parent(experiment_id: str, calibration_path: Path) -> int:
    raise SystemExit(RUN_WITHDRAWN)


def _strip_records(experiment: dict) -> dict:
    """The aggregate references children by digest; it does not copy them."""
    out = dict(experiment)
    out["children"] = [
        {k: v for k, v in c.items()
         if k not in ("record", "entry", "recovery_passed_replayed")}
        | {"run_id": c["entry"]["run_id"],
           "condition": c["entry"]["condition"],
           "block_id": c["entry"]["block_id"],
           "global_position": c["entry"]["global_position"],
           "summary": _summarise(c["record"])}
        for c in experiment["children"]]
    return out


def _summarise(rec: dict | None) -> dict | None:
    if not rec:
        return None
    w = rec["windows"]
    mem = rec["memory"]
    drv = [m["mps_driver_allocated_gb"] for m in mem
           if m.get("mps_driver_allocated_gb") is not None]
    cap = next((m["mps_recommended_max_gb"] for m in mem
                if m.get("mps_recommended_max_gb")), None)
    sw = [m["swap_used_gb"] for m in mem if m.get("swap_used_gb") is not None]
    pr = [m["memory_pressure_percent_free"] for m in mem
          if m.get("memory_pressure_percent_free") is not None]
    fpi = [m["free_plus_inactive_gb"] for m in mem
           if m.get("free_plus_inactive_gb") is not None]
    first = w[0]["seconds_per_row"] if w else None
    last = w[-1]["seconds_per_row"] if w else None
    return {
        "rows_completed": rec["rows_completed"],
        "stopped_early": rec["stopped_early"],
        "model_load_seconds": rec["model_load_seconds"],
        "model_compute_seconds": rec["model_compute_seconds"],
        "end_to_end_seconds": rec["end_to_end_seconds"],
        "model_compute_seconds_per_row": rec["model_compute_seconds_per_row"],
        "end_to_end_seconds_per_row": rec["end_to_end_seconds_per_row"],
        "first_window_seconds_per_row": first,
        "last_window_seconds_per_row": last,
        "degradation_ratio": (round(last / first, 3)
                              if first and last else None),
        "scheduled_empty_cache_calls": rec["scheduled_empty_cache_calls"],
        "scheduled_empty_cache_cost": rec["scheduled_empty_cache_cost"],
        "teardown_empty_cache_calls": rec["teardown_empty_cache_calls"],
        "teardown_empty_cache_seconds": rec["teardown_empty_cache_seconds"],
        "between_row_overhead_breakdown": rec["between_row_overhead_breakdown"],
        "driver_min_gb": min(drv) if drv else None,
        "driver_max_gb": max(drv) if drv else None,
        "driver_end_gb": drv[-1] if drv else None,
        "recommended_max_gb": cap,
        "samples_over_recommended_max": (
            sum(1 for v in drv if cap and v > cap) if drv else None),
        "memory_samples": len(drv),
        "swap_start_gb": sw[0] if sw else None,
        "swap_end_gb": sw[-1] if sw else None,
        "pressure_start_percent_free": pr[0] if pr else None,
        "pressure_min_percent_free": min(pr) if pr else None,
        "pressure_end_percent_free": pr[-1] if pr else None,
        "free_plus_inactive_min_gb": min(fpi) if fpi else None,
        "phase_forward_mean": rec["phases"].get("forward", {}).get(
            "mean_seconds"),
        "phase_backward_mean": rec["phases"].get("backward", {}).get(
            "mean_seconds"),
    }


# --------------------------------------------------------------------------
# one measured run per boot
# --------------------------------------------------------------------------

SNAPSHOT_DIR = "source_snapshot"

SESSION_POLICY = (
    "One measured run per boot. exp001 stopped because the recovery gate was "
    "sampled on an idle machine and, once a run had happened, the machine did "
    "not come back inside that band within the observation window. Waiting "
    "longer in the same boot is not what makes the next run comparable; "
    "starting from a fresh boot is. So the four runs are spread over four "
    "boots, judged against one fixed idle baseline, and this tool runs exactly "
    "one of them per invocation and then exits. It never restarts the machine: "
    "a tool that reboots what it is measuring has changed what it measured.")

SESSION_BASELINE_NOTE = (
    "These thresholds come from a single idle sampling pass and are held fixed "
    "for all four runs. That the machine had loaded no model when it was "
    "sampled is an operating condition the operator arranged, not a fact this "
    "tool can establish: it reads swap, memory pressure, page counts and load "
    "average, none of which record what ran before.")

#: A child of a session must carry `child_source_check`; see
#: `_check_child_source_check`, which judges its contents rather than its
#: presence. Kept apart from CHILD_REQUIRED_KEYS, which describes any child of
#: this report -- including the ones written before the session flow existed.

REQUIRED_EVENT_KEYS = ("schema_version", "kind", "event", "index", "run_id",
                       "condition", "block_id", "position_in_block",
                       "global_position", "experiment_id", "plan_digest",
                       "boot_fingerprint", "recorded_at")
REQUIRED_STARTED_KEYS = ("gate", "source_verified", "out_path",
                         "child_command_digest")
REQUIRED_FINISHED_KEYS = ("started_index", "started_event_digest",
                          "exit_status", "started_at", "finished_at",
                          "wall_seconds", "report_path", "report_sha256",
                          "outcome")
REQUIRED_ATTEMPT_KEYS = ("gate",)
REQUIRED_ABORT_KEYS = ("gate", "abort_reason", "source_problems")
REQUIRED_SOURCE_VERIFIED_KEYS = ("files", "source_manifest_digest",
                                 "checked_after_gate")
REQUIRED_CHILD_SOURCE_CHECK_KEYS = ("files_verified", "source_manifest_digest",
                                    "verified_at")

#: A finished measurement is either this, or the experiment is over.
OUTCOME_COMPLETED = "completed"

#: Two fields of an aggregate describe when it was written and what the working
#: tree looked like then, not what the experiment did. Finalising again is
#: allowed to disagree on those and on nothing else.
AGGREGATE_VOLATILE_FIELDS = ("created_at", "code_on_disk")

#: Where the script lives, as the plan records it. A literal rather than
#: ``__file__``: the digest below has to be recomputable from a stored session
#: without knowing where the checkout sits today.
SCRIPT_REL = "scripts/15_mps_order.py"


def child_invocation(paths: dict, plan_digest: str, entry: dict) -> dict:
    """One description of how a child is launched, used twice.

    The parent spawns from it and the validator re-derives its digest from the
    same fields, so "was this the command that ran" is a question with an
    answer. The digest covers the root-relative form, so moving the checkout
    does not invalidate a stored session.
    """
    out_path = Path(paths["dir"]) / f"{entry['run_id']}.json"
    spec = {"script": SCRIPT_REL,
            "plan": str(Path(paths["plan"]).relative_to(ROOT)),
            "plan_digest": plan_digest,
            "session_dir": str(Path(paths["dir"]).relative_to(ROOT)),
            "global_position": entry["global_position"],
            "run_id": entry["run_id"], "condition": entry["condition"],
            "out": str(out_path.relative_to(ROOT))}
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child",
           "--plan", str(paths["plan"]), "--plan-digest", plan_digest,
           "--session-dir", str(paths["dir"]),
           "--global-position", str(spec["global_position"]),
           "--run-id", spec["run_id"], "--condition", spec["condition"],
           "--out", str(out_path)]
    return {"spec": spec, "cmd": cmd, "digest": digest_obj(spec),
            "out_path": out_path, "out_rel": spec["out"]}


def session_paths(experiment_id: str) -> dict:
    d = RUN_DIR / experiment_id
    return {"dir": d, "plan": d / "plan.json", "session": d / "session.json",
            "snapshot": d / SNAPSHOT_DIR, "aggregate": d / "aggregate.json",
            "lock": RUN_DIR / f".{experiment_id}.lock",
            "markdown": REPORT_DIR / f"15_mps_order_{experiment_id}.md"}


def run_session_init(experiment_id: str, calibration_path: Path) -> int:
    """Fix the plan, the thresholds and the source, before the first run."""
    paths = session_paths(experiment_id)
    with exclusive_lock(paths["lock"], description=f"session {experiment_id}"):
        if paths["dir"].exists():
            # Laying out a session writes several files, so it has a window in
            # which it can die half-done. Distinguishing the two cases matters:
            # one holds measurements, the other holds nothing but copies of
            # files that are still in the tree.
            if paths["plan"].exists() or paths["session"].exists():
                raise SystemExit(
                    f"{paths['dir']} is an initialised session. Sessions are "
                    "never reopened and never overwritten -- choose a new "
                    "experiment id.")
            raise SystemExit(
                f"{paths['dir']} exists but holds no plan and no session, so "
                "an earlier --session-init did not finish. It contains only "
                "copies of source files that are still in the working tree "
                "and no measurement of any kind. Remove the directory "
                "deliberately and run --session-init again.")

        calib_path = resolve_under_root(calibration_path)
        if not calib_path.exists():
            raise SystemExit(f"{calib_path} does not exist. Sample an idle "
                             "baseline with --calibrate first.")
        calib = json.loads(calib_path.read_text())
        replay = replay_calibration(calib)
        if replay["problems"]:
            raise SystemExit("this calibration cannot be used:\n  - "
                             + "\n  - ".join(replay["problems"]))

        plan = build_plan(experiment_id)
        digest = digest_obj(plan)
        manifest = snapshot_sources(ROOT, CODE_FILES, paths["snapshot"])
        boot = boot_identity(experiment_id)

        atomic_write_json(paths["plan"], {
            "schema_version": EXPERIMENT_SCHEMA, "kind": "plan",
            "experiment_id": experiment_id, "created_at": now_iso(),
            "plan": plan, "plan_digest": digest,
            "calibration_path": str(calib_path.relative_to(ROOT)),
            "calibration_digest": calib["calibration_digest"],
            "max_rows": MAX_ROWS,
            "condition_max_seconds": CONDITION_MAX_SECONDS,
            "experiment_max_seconds": None,
            "loss_tolerance": LOSS_TOLERANCE,
            "loss_tolerance_basis": LOSS_TOLERANCE_BASIS,
            "one_run_per_boot": True,
        })
        atomic_write_json(paths["session"], {
            "schema_version": EXPERIMENT_SCHEMA, "kind": "session",
            "experiment_id": experiment_id, "created_at": now_iso(),
            "policy": SESSION_POLICY,
            "one_run_per_boot": True,
            "plan_digest": digest,
            "calibration_path": str(calib_path.relative_to(ROOT)),
            "calibration_digest": calib["calibration_digest"],
            "calibration_sha256": sha256_file(calib_path),
            "thresholds": replay["thresholds"],
            "gate_policy": replay["gate_policy"],
            "thresholds_baseline": "fixed idle baseline",
            "thresholds_baseline_note": SESSION_BASELINE_NOTE,
            "source_snapshot_dir": SNAPSHOT_DIR,
            "source_manifest": manifest,
            "source_manifest_digest": manifest_digest(manifest),
            "created_in_boot_fingerprint": boot["boot_fingerprint"],
            "boot_fingerprint_note": (
                "Hashed with the experiment id. The boot's own identifier is "
                "never written to disk, to a report or to the screen."),
        })
    print(f"session {experiment_id} laid out at {paths['dir']}")
    print(f"  plan digest {digest[:12]}... ({len(plan)} runs, "
          + ", ".join(p["condition"] for p in plan) + ")")
    print(f"  thresholds  {replay['thresholds']} (fixed idle baseline)")
    print(f"  source snapshot: {len(manifest['files'])} files, digest "
          f"{manifest_digest(manifest)[:12]}...")
    print(f"  boot fingerprint at init: {boot['boot_fingerprint']}")
    print(f"\nRun one child with:\n  {sys.executable} {Path(__file__).name} "
          f"--session-next --experiment-id {experiment_id}")
    return 0


def _session_preconditions(experiment_id: str, *,
                           check_working_tree: bool = True) -> dict:
    """Everything that must hold before a child may start in this boot.

    Every check here runs before the gate is polled and long before a child is
    spawned, because after the spawn the boot is spent and the answer to "was
    this session laid out the way it says" no longer changes anything.

    ``check_working_tree=False`` reads a finished session back: the snapshot
    still has to match the manifest, but the live tree is allowed to have moved
    on, because it always will.
    """
    paths = session_paths(experiment_id)
    for key in ("plan", "session"):
        if not paths[key].exists():
            raise SystemExit(f"{paths[key]} does not exist. Run "
                             "--session-init first.")
    plan_doc = json.loads(paths["plan"].read_text())
    session = json.loads(paths["session"].read_text())
    plan = plan_doc.get("plan") or []
    digest = digest_obj(plan)
    problems: list[str] = []

    # ---- the two documents are the documents they claim to be -------------
    for label, doc, kind in (("plan", plan_doc, "plan"),
                             ("session", session, "session")):
        if doc.get("schema_version") != EXPERIMENT_SCHEMA:
            problems.append(
                f"the {label} file records schema_version "
                f"{doc.get('schema_version')!r}, this tool writes "
                f"{EXPERIMENT_SCHEMA}")
        if doc.get("kind") != kind:
            problems.append(f"the {label} file is a {doc.get('kind')!r} "
                            f"record, not a {kind}")
        if doc.get("experiment_id") != experiment_id:
            problems.append(
                f"the {label} file belongs to experiment "
                f"{doc.get('experiment_id')!r}, this is {experiment_id!r}")
        if doc.get("one_run_per_boot") is not True:
            problems.append(
                f"the {label} file records one_run_per_boot="
                f"{doc.get('one_run_per_boot')!r}. This flow only runs "
                "sessions that were laid out for one measured run per boot.")
    if paths["dir"].name != experiment_id:
        problems.append(f"the session directory is {paths['dir'].name!r}, the "
                        f"experiment is {experiment_id!r}")
    if not plan:
        problems.append("the plan file records no plan")
    if digest != plan_doc.get("plan_digest"):
        problems.append("the plan file's own digest does not match its plan")
    if session.get("plan_digest") != plan_doc.get("plan_digest"):
        problems.append("the session and the plan disagree on the plan digest")

    # ---- both point at one calibration, and it is the one on disk ---------
    calib_rel = session.get("calibration_path")
    if plan_doc.get("calibration_path") != calib_rel:
        problems.append(
            f"the plan was laid out against calibration "
            f"{plan_doc.get('calibration_path')!r}, the session against "
            f"{calib_rel!r}")
    if plan_doc.get("calibration_digest") != session.get("calibration_digest"):
        problems.append("the plan and the session disagree on the calibration "
                        "digest")
    calib = {}
    if not calib_rel:
        problems.append("the session names no calibration file")
    else:
        calib_path = ROOT / calib_rel
        if not calib_path.exists():
            problems.append(f"the calibration this session was laid out "
                            f"against ({calib_rel}) is missing")
        else:
            calib = json.loads(calib_path.read_text())
            if not session.get("calibration_sha256"):
                problems.append("the session records no digest for the "
                                "calibration file")
            elif sha256_file(calib_path) != session["calibration_sha256"]:
                problems.append("the calibration file has changed since the "
                                "session was laid out")
    replay = replay_calibration(calib)
    problems += replay["problems"]
    if calib.get("calibration_digest") != session.get("calibration_digest"):
        problems.append("the calibration file's own digest is not the one the "
                        "session was laid out against")

    # ---- the band the four runs are judged against ------------------------
    if replay["thresholds"] != session.get("thresholds"):
        problems.append(
            "the thresholds the calibration samples produce are not the ones "
            "this session was laid out with, so the four runs would not be "
            "judged against one band")
    stored_policy = session.get("gate_policy")
    if stored_policy != replay["gate_policy"]:
        problems.append(
            "the session's gate policy is not the calibration's: "
            f"{stored_policy!r} against {replay['gate_policy']!r}. How long to "
            "wait and how many consecutive passes to require are part of the "
            "band, not settings to be adjusted per run.")

    manifest = session.get("source_manifest") or {}
    if manifest_digest(manifest) != session.get("source_manifest_digest"):
        problems.append("the source manifest does not match the digest stored "
                        "with it")
    problems += verify_sources(
        ROOT, manifest,
        paths["dir"] / session.get("source_snapshot_dir", SNAPSHOT_DIR),
        check_working_tree=check_working_tree)
    return {"paths": paths, "plan_doc": plan_doc, "plan": plan,
            "plan_digest": digest, "session": session, "calibration": calib,
            "experiment_id": experiment_id,
            "thresholds": session.get("thresholds"),
            "gate_policy": session.get("gate_policy"), "problems": problems}


def verify_session_source(ctx: dict) -> list[str]:
    """Re-read the tree and the snapshot. Called again after the gate."""
    session, paths = ctx["session"], ctx["paths"]
    manifest = session.get("source_manifest") or {}
    problems = []
    if manifest_digest(manifest) != session.get("source_manifest_digest"):
        problems.append("the source manifest does not match the digest stored "
                        "with it")
    return problems + verify_sources(
        ROOT, manifest,
        paths["dir"] / session.get("source_snapshot_dir", SNAPSHOT_DIR))


# --------------------------------------------------------------------------
# the journal, validated before anything is decided from it
# --------------------------------------------------------------------------

def _check_started_semantics(name: str, body: dict, entry: dict,
                             ctx: dict) -> list[str]:
    """What a started event claims about the source and the command it ran.

    Deleting one of these fields is caught by the required-key check. Editing
    one is not, and editing is the interesting case: a source check that says
    it looked at three files when the manifest has six, or an ``out_path``
    pointing somewhere the session never intended, describes a run nobody
    performed.
    """
    problems: list[str] = []
    session = ctx["session"]
    manifest = session.get("source_manifest") or {}
    n_files = len(manifest.get("files") or {})
    digest = session.get("source_manifest_digest")

    verified = body.get("source_verified")
    if not isinstance(verified, dict):
        problems.append(f"{name}: source_verified is not a record")
    else:
        for key in REQUIRED_SOURCE_VERIFIED_KEYS:
            if key not in verified:
                problems.append(f"{name}: source_verified is missing `{key}`")
        if verified.get("files") != n_files:
            problems.append(
                f"{name}: says it verified {verified.get('files')!r} files, "
                f"the manifest holds {n_files}")
        if verified.get("source_manifest_digest") != digest:
            problems.append(f"{name}: source_verified names a different "
                            "source manifest from the session's")
        if verified.get("checked_after_gate") is not True:
            problems.append(
                f"{name}: does not record that the source was re-checked "
                "after the gate. The gate can wait fifteen minutes, and the "
                "tree is writable for all of it.")

    invocation = child_invocation(ctx["paths"], ctx["plan_digest"], entry)
    if body.get("out_path") != invocation["out_rel"]:
        problems.append(
            f"{name}: out_path is {body.get('out_path')!r}, but this session "
            f"writes {entry['run_id']} to {invocation['out_rel']!r}")
    if body.get("child_command_digest") != invocation["digest"]:
        problems.append(
            f"{name}: the child command digest is not the one this session's "
            "own invocation produces")
    return problems


def _is_clean_exit(value) -> bool:
    """A process that exited zero, and nothing that merely compares equal to it.

    JSON has no integer type of its own, so ``false``, ``0.0`` and ``"0"`` all
    arrive as plausible-looking stand-ins, and Python agrees with two of them:
    ``False == 0`` and ``0.0 == 0``. An exit status is the small integer the
    kernel returned, so it is checked as one.
    """
    return (isinstance(value, int) and not isinstance(value, bool)
            and value == 0)


def _check_finished_outcome(name: str, body: dict) -> list[str]:
    """One outcome per way a measurement can end, and nothing else.

    The four are exhaustive because they are read off two facts the parent
    always has: what the process exited with, and whether a report exists. An
    outcome outside the set is not a fifth way for a run to end, it is a word
    someone wrote into the file, and the difference matters because everything
    downstream keys off it -- `completed` counts a run, the others end the
    experiment.

    They are also cross-checked rather than taken at face value. `no_report`
    beside a non-zero exit is the interesting case: it reads as "the run was
    fine, the report just isn't there", and it would hide a crash.
    """
    problems: list[str] = []
    outcome = body.get("outcome")
    exit_status = body.get("exit_status")
    rel, sha = body.get("report_path"), body.get("report_sha256")

    if bool(rel) != bool(sha):
        problems.append(
            f"{name}: records a report {'path with no digest' if rel else 'digest with no path'}"
            ". A report is its path and its digest together, or it is not "
            "there.")
    if outcome not in FINISHED_OUTCOMES:
        return problems + [
            f"{name}: outcome {outcome!r} is not one of "
            f"{', '.join(FINISHED_OUTCOMES)}"]

    if outcome == OUTCOME_COMPLETED:
        if not _is_clean_exit(exit_status):
            problems.append(f"{name}: outcome says completed, exit status is "
                            f"{exit_status!r}")
        if not rel or not sha:
            problems.append(f"{name}: outcome says completed with no report")
    elif outcome == "nonzero_exit":
        if (not isinstance(exit_status, int) or isinstance(exit_status, bool)
                or exit_status == 0):
            problems.append(
                f"{name}: outcome says the child exited non-zero, exit status "
                f"is {exit_status!r}")
    elif outcome == "timed_out":
        if exit_status is not None:
            problems.append(
                f"{name}: outcome says the child timed out, which leaves no "
                f"exit status, but one is recorded ({exit_status!r})")
    elif outcome == "no_report":
        if not _is_clean_exit(exit_status):
            problems.append(
                f"{name}: outcome says the child exited cleanly without "
                f"writing a report, but its exit status is {exit_status!r}. A "
                "child that failed did not merely forget to write.")
        if rel or sha:
            problems.append(f"{name}: outcome says no report, yet one is "
                            "referenced")
    return problems


def _check_child_source_check(run: str, record: dict, ctx: dict) -> list[str]:
    """The child's own pre-load check, judged rather than merely counted."""
    session = ctx["session"]
    manifest = session.get("source_manifest") or {}
    n_files = len(manifest.get("files") or {})
    check = record.get("child_source_check")
    if not isinstance(check, dict):
        problems = [f"{run}: child report is missing `child_source_check`, so "
                    "it did not check its own source before loading anything"]
        return problems
    problems = []
    for key in REQUIRED_CHILD_SOURCE_CHECK_KEYS:
        if key not in check:
            problems.append(f"{run}: child_source_check is missing `{key}`")
    if check.get("files_verified") != n_files:
        problems.append(
            f"{run}: the child says it verified {check.get('files_verified')!r} "
            f"files, the manifest holds {n_files}")
    if check.get("source_manifest_digest") != session.get(
            "source_manifest_digest"):
        problems.append(f"{run}: the child checked a different source manifest "
                        "from the session's")
    return problems


def validate_journal(events: list[dict], plan: list[dict], ctx: dict) -> dict:
    """Read the journal fail-closed, then say what state it leaves the session in.

    Every check here exists because the alternative is a session that quietly
    continues from a state nobody can account for. The two that matter most:

    * a ``measurement_started`` with no ``measurement_finished`` means a child
      began and the parent never came back. Whether it trained for an hour or
      died immediately is not recorded anywhere, so the honest reading is that
      the experiment is over -- not that the run may be tried again.
    * a boot fingerprint that appears against two measured runs means the
      second one did not start from a fresh machine, which is the entire point
      of the design.

    Failed gate attempts are kept and reported. They are not failures of the
    experiment: nothing was measured, so nothing was spent.
    """
    problems: list[str] = []
    plan_by_run = {p["run_id"]: p for p in plan}
    order = [p["run_id"] for p in plan]
    thresholds = ctx["thresholds"]
    gate_policy = ctx["gate_policy"]

    parsed: list[dict] = []
    for position, item in enumerate(events, 1):
        name, body = item["file_name"], item["body"]
        if "__unreadable__" in body:
            problems.append(f"{name}: unreadable ({body['__unreadable__']})")
            continue
        m = EVENT_FILE_RE.match(name)
        if not m:
            problems.append(f"{name}: not a journal event file name")
            continue
        if int(m.group("index")) != position:
            problems.append(
                f"{name}: journal index {m.group('index')} at position "
                f"{position}. Indices must run 1..N with nothing inserted, "
                "removed or renumbered.")
        for key in REQUIRED_EVENT_KEYS:
            if key not in body:
                problems.append(f"{name}: event is missing `{key}`")
        if body.get("event") != m.group("event"):
            problems.append(f"{name}: file says {m.group('event')!r}, the "
                            f"event says {body.get('event')!r}")
        if body.get("run_id") != m.group("run_id"):
            problems.append(f"{name}: file says run {m.group('run_id')!r}, "
                            f"the event says {body.get('run_id')!r}")
        if body.get("index") != position:
            problems.append(f"{name}: event records index "
                            f"{body.get('index')!r} at position {position}")
        if body.get("event") not in EVENT_KINDS:
            problems.append(f"{name}: unknown event {body.get('event')!r}")
            continue
        if body.get("kind") != "session_event":
            problems.append(f"{name}: not a session event record")
        if body.get("schema_version") != EXPERIMENT_SCHEMA:
            problems.append(f"{name}: schema_version "
                            f"{body.get('schema_version')!r}")
        if body.get("experiment_id") != ctx["experiment_id"]:
            problems.append(
                f"{name}: belongs to experiment "
                f"{body.get('experiment_id')!r}, this session is "
                f"{ctx['experiment_id']!r}. Events are not moved between "
                "sessions.")
        if body.get("plan_digest") != ctx["plan_digest"]:
            problems.append(f"{name}: written against a different plan")
        entry = plan_by_run.get(body.get("run_id"))
        if entry is None:
            problems.append(f"{name}: run {body.get('run_id')!r} is not in the "
                            "plan")
            continue
        for key in ("condition", "block_id", "position_in_block",
                    "global_position"):
            if body.get(key) != entry.get(key):
                problems.append(f"{name}: {key} is {body.get(key)!r}, the plan "
                                f"says {entry.get(key)!r}")
        parsed.append({**item, "position": position, "event": body["event"],
                       "run_id": body["run_id"], "body": body})

    # ---- state machine, run order, boots ---------------------------------
    started: dict[str, dict] = {}
    finished: dict[str, dict] = {}
    attempts: dict[str, list] = {r: [] for r in order}
    boots: dict[str, str] = {}
    completed: list[str] = []
    terminal_reason: str | None = None
    open_started: dict | None = None

    for item in parsed:
        body, name, run = item["body"], item["file_name"], item["run_id"]
        event = item["event"]
        entry = plan_by_run[run]
        attempts.setdefault(run, []).append(item)

        if open_started is not None and not (
                event == EVENT_FINISHED
                and run == open_started["run_id"]):
            problems.append(
                f"{name}: {open_started['file_name']} began a measurement that "
                "never finished, so nothing may follow it but that run's "
                "measurement_finished")

        expected_next = next((r for r in order if r not in completed), None)
        if run != expected_next:
            problems.append(
                f"{name}: {run} was attempted while the next planned run was "
                f"{expected_next!r}. The order is fixed before the first run "
                "and is never revisited.")

        if event == EVENT_GATE_ATTEMPT:
            for key in REQUIRED_ATTEMPT_KEYS:
                if key not in body:
                    problems.append(f"{name}: gate attempt is missing `{key}`")
            if body.get("gate", {}).get("passed"):
                problems.append(f"{name}: recorded as a gate attempt, but its "
                                "gate passed. A gate that passes either starts "
                                "a measurement or aborts before the spawn.")
        elif event == EVENT_PRE_SPAWN_ABORT:
            for key in REQUIRED_ABORT_KEYS:
                if key not in body:
                    problems.append(f"{name}: abort event is missing `{key}`")
            if not body.get("gate", {}).get("passed"):
                problems.append(
                    f"{name}: recorded as an abort after the gate passed, but "
                    "its gate did not pass. A gate that never released is a "
                    "gate attempt.")
            if not body.get("source_problems"):
                problems.append(f"{name}: aborted before the spawn without "
                                "recording what was wrong")
            if body.get("out_path") or body.get("child_command_digest"):
                problems.append(f"{name}: aborted before any child existed, "
                                "yet references one")
        elif event == EVENT_STARTED:
            for key in REQUIRED_STARTED_KEYS:
                if key not in body:
                    problems.append(f"{name}: started event is missing `{key}`")
            problems += _check_started_semantics(name, body, entry, ctx)
            if run in started:
                problems.append(
                    f"{name}: {run} already started at "
                    f"{started[run]['file_name']}. A measured run is attempted "
                    "once.")
            if not body.get("gate", {}).get("passed"):
                problems.append(f"{name}: a measurement began without its gate "
                                "passing")
            fp = body.get("boot_fingerprint")
            if not fp:
                problems.append(f"{name}: no boot fingerprint, so this run "
                                "cannot be shown to be in its own boot")
            elif fp in boots:
                problems.append(
                    f"{name}: this boot already measured {boots[fp]}. One "
                    "measured run per boot; restart between runs.")
            else:
                boots[fp] = run
            started[run] = item
            open_started = item
        elif event == EVENT_FINISHED:
            for key in REQUIRED_FINISHED_KEYS:
                if key not in body:
                    problems.append(f"{name}: finished event is missing "
                                    f"`{key}`")
            begin = started.get(run)
            if begin is None:
                problems.append(f"{name}: finished a measurement that never "
                                "started")
            else:
                if body.get("started_index") != begin["body"].get("index"):
                    problems.append(
                        f"{name}: points at journal index "
                        f"{body.get('started_index')!r}, but {run} started at "
                        f"{begin['body'].get('index')!r}")
                if body.get("started_event_digest") != begin["file_sha256"]:
                    problems.append(
                        f"{name}: the started event it references does not "
                        "hash to the digest recorded here")
                if body.get("boot_fingerprint") != begin["body"].get(
                        "boot_fingerprint"):
                    problems.append(f"{name}: finished in a different boot "
                                    "from the one it started in")
                if body.get("run_id") != begin["body"].get("run_id"):
                    problems.append(f"{name}: finished a different run from "
                                    "the one that started")
                started_out = begin["body"].get("out_path")
                if body.get("report_path") not in (None, started_out):
                    problems.append(
                        f"{name}: reports {body.get('report_path')!r}, but the "
                        f"measurement began against {started_out!r}")
                if (body.get("outcome") == OUTCOME_COMPLETED
                        and body.get("report_path") != started_out):
                    problems.append(
                        f"{name}: completed without writing the report the "
                        f"measurement began against ({started_out!r})")
            if run in finished:
                problems.append(f"{name}: {run} already finished")
            finished[run] = item
            open_started = None
            outcome_problems = _check_finished_outcome(name, body)
            problems += outcome_problems
            if not outcome_problems:
                if body.get("outcome") == OUTCOME_COMPLETED:
                    completed.append(run)
                elif terminal_reason is None:
                    terminal_reason = (
                        f"{run} started and did not complete "
                        f"({body.get('outcome')}); exit status "
                        f"{body.get('exit_status')!r}")

        gate = body.get("gate")
        if isinstance(gate, dict):
            replay = replay_gate({"run_id": f"{run} ({name})", "gate": gate,
                                  "recovery_passed_recomputed":
                                      bool(gate.get("passed")),
                                  "not_run": not gate.get("passed")},
                                 thresholds, gate_policy)
            problems += replay["problems"]

    if open_started is not None and terminal_reason is None:
        terminal_reason = (
            f"{open_started['run_id']} began a measurement at "
            f"{open_started['file_name']} and no finished event was ever "
            "written. What that child did is not recorded, so it cannot be "
            "retried and it cannot be counted.")

    # ---- the reports the journal points at -------------------------------
    # Every report, whatever the outcome beside it. A run that timed out may
    # still have written one, and a report nobody checked is a report that
    # could say anything.
    for run, item in finished.items():
        body = item["body"]
        rel = body.get("report_path")
        if not rel:
            continue
        path = ROOT / rel
        if not path.exists():
            problems.append(f"{item['file_name']}: report {rel} is missing")
            continue
        if sha256_file(path) != body.get("report_sha256"):
            problems.append(f"{item['file_name']}: report {rel} has changed "
                            "since the run")
            continue
        try:
            record = json.loads(path.read_text())
        except Exception as exc:
            problems.append(f"{item['file_name']}: report unreadable ({exc})")
            continue
        problems += validate_child(record, plan_by_run[run], ctx["plan_digest"])
        problems += _check_child_source_check(run, record, ctx)
        problems += check_treatment_contract(record, run)["problems"]

    return {"problems": problems, "events": parsed, "attempts": attempts,
            "started": started, "finished": finished, "completed": completed,
            "boots": boots, "open_started": open_started,
            "terminal": terminal_reason is not None,
            "terminal_reason": terminal_reason,
            "next": next((p for p in plan if p["run_id"] not in completed),
                         None)}


def replay_session_journal(exp_dir: Path, agg: dict) -> dict:
    """Rebuild the one-run-per-boot evidence from the events, not the summary.

    The aggregate says four runs came from four boots. That sentence is a
    conclusion, and the files it was drawn from are still on disk, so it is
    drawn again here: the state machine runs over the event files, the child
    slots are built again from the started and finished events, and every one
    of them has to be the one the aggregate recorded.

    The working tree is deliberately not checked. What makes a finished session
    durable is the snapshot taken when the plan was made; requiring the live
    tree to stay frozen would mean a session could only be replayed in a
    repository nobody had worked in since.
    """
    exp_dir = Path(exp_dir)
    problems: list[str] = []

    # The directory is the identity. The aggregate's own `experiment_id` is
    # checked against it *before* anything is read, and is never what decides
    # which session gets loaded -- an aggregate naming another experiment would
    # otherwise be replayed against that experiment's journal and agree with
    # it.
    experiment_id = exp_dir.name
    if agg.get("experiment_id") != experiment_id:
        return {"problems": [
            f"the aggregate in {experiment_id} says it belongs to experiment "
            f"{agg.get('experiment_id')!r}. Nothing is loaded on the strength "
            "of that: the directory names the experiment."], "state": None}
    if session_paths(experiment_id)["dir"] != exp_dir:
        return {"problems": [
            f"{exp_dir} is not where session {experiment_id} lives "
            f"({session_paths(experiment_id)['dir']})"], "state": None}

    session_path = exp_dir / "session.json"
    if not session_path.exists():
        return {"problems": [f"{session_path} does not exist, so the journal "
                             "cannot be replayed"], "state": None}
    if agg.get("one_run_per_boot") is not True:
        problems.append(
            "this directory holds a one-run-per-boot session, but the "
            f"aggregate records one_run_per_boot={agg.get('one_run_per_boot')!r}"
            ". A session is not read as a legacy experiment because a flag "
            "says so.")

    # Every one of these is how the aggregate says where its evidence is. A
    # missing one is not a reason to skip the check it enables.
    expected_session_rel = str(session_path.relative_to(ROOT))
    if agg.get("session_path") != expected_session_rel:
        problems.append(
            f"the aggregate names session file "
            f"{agg.get('session_path')!r}, this session's is "
            f"{expected_session_rel!r}")
    if not agg.get("session_sha256"):
        problems.append("the aggregate records no digest for session.json")
    elif sha256_file(session_path) != agg["session_sha256"]:
        problems.append("session.json has changed since the aggregate was "
                        "written")

    ctx = _session_preconditions(experiment_id, check_working_tree=False)
    problems += ctx["problems"]
    if problems:
        return {"problems": problems, "state": None}

    session = ctx["session"]
    if agg.get("source_snapshot_dir") != session.get("source_snapshot_dir",
                                                     SNAPSHOT_DIR):
        problems.append("the aggregate and session.json disagree on where the "
                        "source snapshot lives")
    if agg.get("source_manifest_digest") != session.get(
            "source_manifest_digest"):
        problems.append("the aggregate and session.json disagree on the source "
                        "manifest digest")

    events = read_events(exp_dir)
    listed = agg.get("journal_files")
    if not isinstance(listed, list):
        problems.append("the aggregate lists no journal files")
        listed = []
    names = [f.get("file_name") for f in listed]
    if len(names) != len(set(names)):
        problems.append("the aggregate lists the same journal event twice")
    recorded = {f.get("file_name"): f.get("sha256") for f in listed}
    on_disk = {e["file_name"]: e["file_sha256"] for e in events}
    for name in sorted(set(recorded) | set(on_disk)):
        if name not in on_disk:
            problems.append(f"journal event {name} is missing")
        elif name not in recorded:
            problems.append(f"journal event {name} was added after the "
                            "aggregate was written")
        elif recorded[name] != on_disk[name]:
            problems.append(f"journal event {name} has changed since the "
                            "aggregate was written")

    state = validate_journal(events, ctx["plan"], ctx)
    problems += state["problems"]
    if problems:
        return {"problems": problems, "state": state}

    # ---- the conclusions, derived again and compared ---------------------
    derived = session_experiment_state(ctx, state)
    unaccounted = sorted(set(derived) - set(SESSION_DERIVED_FIELDS))
    if unaccounted:
        problems.append(
            "the journal derives fields nothing compares: "
            + ", ".join(unaccounted)
            + ". Add them to SESSION_DERIVED_FIELDS with where they are "
              "checked, or they are conclusions no one is reading back.")
    if canonical([_attempt_summary(i) for i in state["events"]]) != canonical(
            agg.get("journal_events")):
        problems.append("the aggregate's journal summary is not what the "
                        "event files produce")
    if bool(agg.get("terminal")) != state["terminal"]:
        problems.append(f"the aggregate says terminal={agg.get('terminal')}, "
                        f"the journal says {state['terminal']}")
    if canonical(derived["boot_fingerprints"]) != canonical(
            agg.get("boot_fingerprints")):
        problems.append(
            "the aggregate's boot fingerprints are not the ones the journal "
            "records")
    if bool(agg.get("complete")) != derived["complete"]:
        problems.append(f"the aggregate says complete={agg.get('complete')}, "
                        f"the journal implies {derived['complete']}")
    if agg.get("stopped_reason") != derived["stopped_reason"]:
        problems.append("the aggregate's stop reason is not the one the "
                        "journal implies")
    if agg.get("elapsed_seconds") != derived["elapsed_seconds"]:
        problems.append(
            f"the aggregate records {agg.get('elapsed_seconds')!r} seconds "
            f"elapsed, the journal's finished events sum to "
            f"{derived['elapsed_seconds']!r}")

    stored = agg.get("children") or []
    if len(stored) != len(ctx["plan"]):
        problems.append(f"{len(stored)} child slots for a "
                        f"{len(ctx['plan'])}-run plan")
    else:
        for slot, expected in zip(stored, derived["children"]):
            run = expected["run_id"]
            for key in COMPARED_SESSION_CHILD_KEYS:
                if canonical(slot.get(key)) != canonical(expected.get(key)):
                    problems.append(
                        f"{run}: the aggregate records {key}="
                        f"{slot.get(key)!r}, the journal gives "
                        f"{expected.get(key)!r}")
    return {"problems": problems, "state": state, "derived": derived,
            "boot_fingerprints": derived["boot_fingerprints"], "ctx": ctx,
            "terminal_outcomes": derived["terminal_outcomes"]}


SESSION_MARKERS = ("one_run_per_boot", "session_path", "session_sha256",
                   "journal_files", "journal_events", "boot_fingerprints",
                   "source_manifest_digest", "source_snapshot_dir", "terminal")


def looks_like_a_session(exp_dir: Path, agg: dict) -> bool:
    """Is this directory a one-run-per-boot session?

    Answered from what is on disk and from any session field the aggregate
    carries -- never from ``one_run_per_boot`` alone. A boolean is exactly what
    an editor would flip to have a session read as a legacy experiment and skip
    the journal entirely, so the flag is checked *inside* the session path and
    a false one is a failure rather than a route around it.
    """
    exp_dir = Path(exp_dir)
    return ((exp_dir / "session.json").exists()
            or (exp_dir / EVENT_DIR).exists()
            or any(k in (agg or {}) for k in SESSION_MARKERS))


def read_journal(experiment_id: str, *,
                 check_working_tree: bool = True) -> tuple[dict, dict]:
    """Preconditions and journal together, both fail-closed.

    ``check_working_tree`` is the difference between starting a run and
    accounting for one. A child may only be spawned against a tree that still
    matches the plan; writing the record of a session that already crashed must
    not depend on that, or a crash would become unreportable the moment anyone
    edited a file.
    """
    ctx = _session_preconditions(experiment_id,
                                 check_working_tree=check_working_tree)
    if ctx["problems"]:
        raise SystemExit("this session cannot be read:\n  - "
                         + "\n  - ".join(ctx["problems"]))
    state = validate_journal(read_events(ctx["paths"]["dir"]), ctx["plan"], ctx)
    if state["problems"]:
        raise SystemExit(
            "this session's journal does not check out:\n  - "
            + "\n  - ".join(state["problems"])
            + "\nNothing is decided from a journal that cannot be read.")
    return ctx, state


def run_session_next(experiment_id: str) -> int:
    """Run the next planned child in this boot, then exit. No restarting."""
    paths = session_paths(experiment_id)
    with exclusive_lock(paths["lock"], description=f"session {experiment_id}"):
        ctx, state = read_journal(experiment_id)
        plan = ctx["plan"]

        if state["terminal"]:
            unfinalised = (
                "" if paths["aggregate"].exists() else
                f"\nIt has not been written up yet. Run --session-finalize "
                f"--experiment-id {experiment_id} to produce the incomplete "
                "aggregate and report for what did happen. That records the "
                "session; it does not resume it.")
            raise SystemExit(
                f"this experiment is over: {state['terminal_reason']}\n"
                "A measured run that started and did not complete cannot be "
                "repeated -- not in this boot and not in a later one -- "
                "because the machine it would repeat on is not the machine it "
                "started on. Begin again with a new experiment id and the full "
                f"{len(plan)}-run plan." + unfinalised)

        entry = state["next"]
        if entry is None:
            print(f"all {len(plan)} runs have completed.")
            return finalise_session(experiment_id, ctx=ctx, state=state)

        boot = boot_identity(experiment_id)
        if not boot["boot_fingerprint"]:
            raise SystemExit(
                f"this boot cannot be identified: {boot['reason']}. Refusing "
                "to run, because one measured run per boot cannot be enforced "
                "against a boot with no name.")
        if boot["boot_fingerprint"] in state["boots"]:
            done = state["completed"]
            raise SystemExit(
                f"a measured run has already happened in this boot "
                f"({state['boots'][boot['boot_fingerprint']]}). "
                f"{len(done)} of {len(plan)} runs are done "
                f"({', '.join(done) or 'none'}); the next is "
                f"{entry['run_id']} ({entry['condition']}). Restart the "
                "machine and run this command again. This tool will not "
                "restart it for you.")

        index = len(state["events"]) + 1
        print(f"\n=== {experiment_id} step {index}: {entry['run_id']} "
              f"({entry['condition']}, block {entry['block_id']}, position "
              f"{entry['global_position']}) ===", flush=True)
        print(f"  boot fingerprint {boot['boot_fingerprint']}", flush=True)
        print(f"  source verified: "
              f"{len(ctx['session']['source_manifest']['files'])} files "
              "unchanged", flush=True)

        base = {"schema_version": EXPERIMENT_SCHEMA, "kind": "session_event",
                "index": index, "experiment_id": experiment_id,
                "plan_digest": ctx["plan_digest"],
                "boot_fingerprint": boot["boot_fingerprint"],
                "boot_source": boot["source"],
                **{k: entry[k] for k in ("run_id", "condition", "block_id",
                                         "position_in_block",
                                         "global_position")}}

        gate = wait_for_recovery(ctx["thresholds"],
                                 poll_seconds=ctx["gate_policy"][
                                     "poll_seconds"],
                                 max_wait_seconds=ctx["gate_policy"][
                                     "max_wait_seconds"],
                                 needed_consecutive=ctx["gate_policy"][
                                     "consecutive_passes_required"])
        for p in gate["polls"]:
            print(f"  poll {p['poll']} @{p['elapsed_seconds']}s "
                  f"passed={p['passed']} streak={p['consecutive_passes']}"
                  + (f" failed={p['failed_metrics']}" if p["failed_metrics"]
                     else ""), flush=True)

        if not gate["passed"]:
            append_event(paths["dir"], index, entry["run_id"],
                         EVENT_GATE_ATTEMPT,
                         {**base, "event": EVENT_GATE_ATTEMPT,
                          "recorded_at": now_iso(), "gate": gate,
                          "reason": gate["reason"]})
            print(f"\nGATE: {gate['reason']}", flush=True)
            print("No child was started, so this boot has not been used up. "
                  "Leave the machine idle and run the same command again, or "
                  "restart it first.", flush=True)
            return 1

        # The gate has just spent up to fifteen minutes polling. The tree was
        # writable for all of it, so it is checked again here -- after the
        # wait, before the spawn.
        drift = verify_session_source(ctx)
        if drift:
            # Its own event, not a gate attempt: this gate *passed*, and a
            # journal that files it under "the gate never released" is a
            # journal that misdescribes the machine. Nothing was measured, so
            # the boot is untouched and this run may be attempted again.
            append_event(paths["dir"], index, entry["run_id"],
                         EVENT_PRE_SPAWN_ABORT,
                         {**base, "event": EVENT_PRE_SPAWN_ABORT,
                          "recorded_at": now_iso(), "gate": gate,
                          "abort_reason": "source changed while the gate was "
                                          "waiting",
                          "source_problems": drift})
            raise SystemExit(
                "the source changed while the gate was waiting:\n  - "
                + "\n  - ".join(drift)
                + "\nNo child was started and this boot is untouched: restore "
                  "the tree and run the same command again, in this boot or "
                  "another.")

        invocation = child_invocation(paths, ctx["plan_digest"], entry)
        out_path, cmd = invocation["out_path"], invocation["cmd"]

        # Written before the spawn. From here the boot is spent, whatever
        # happens next -- including this process being killed in the next
        # instruction.
        started_digest = append_event(
            paths["dir"], index, entry["run_id"], EVENT_STARTED,
            {**base, "event": EVENT_STARTED, "recorded_at": now_iso(),
             "gate": gate,
             "source_verified": {
                 "files": len(ctx["session"]["source_manifest"]["files"]),
                 "source_manifest_digest": ctx["session"][
                     "source_manifest_digest"],
                 "checked_after_gate": True},
             "out_path": invocation["out_rel"],
             "child_command_digest": invocation["digest"]})

        started_at = now_iso()
        t0 = time.monotonic()
        outcome = OUTCOME_COMPLETED
        try:
            returncode = subprocess.run(
                cmd, cwd=str(ROOT),
                timeout=CONDITION_MAX_SECONDS + 600).returncode
        except subprocess.TimeoutExpired:
            returncode, outcome = None, "timed_out"
        wall = time.monotonic() - t0

        report_sha = None
        if out_path.exists():
            report_sha = sha256_file(out_path)
        if returncode != 0 and outcome == OUTCOME_COMPLETED:
            outcome = "nonzero_exit"
        if report_sha is None and outcome == OUTCOME_COMPLETED:
            outcome = "no_report"

        append_event(
            paths["dir"], index + 1, entry["run_id"], EVENT_FINISHED,
            {**base, "index": index + 1, "event": EVENT_FINISHED,
             "recorded_at": now_iso(), "started_index": index,
             "started_event_digest": started_digest,
             "exit_status": returncode, "started_at": started_at,
             "finished_at": now_iso(), "wall_seconds": round(wall, 3),
             "report_path": (str(out_path.relative_to(ROOT))
                             if report_sha else None),
             "report_sha256": report_sha, "outcome": outcome})
        print(f"  {entry['run_id']} exit={returncode} wall={wall:.1f}s "
              f"outcome={outcome}", flush=True)

        if outcome != OUTCOME_COMPLETED:
            print(f"\nThis experiment is over: {entry['run_id']} started and "
                  f"did not complete ({outcome}). It cannot be retried in any "
                  "boot. Begin again with a new experiment id.", flush=True)
            # An experiment that stopped still has to produce its record. The
            # runs that did finish are measurements; leaving them unreported
            # would lose them to a failure that happened afterwards.
            finalise_session(experiment_id)
            return 1

        remaining = [p["run_id"] for p in plan
                     if p["run_id"] not in state["completed"] + [
                         entry["run_id"]]]
        if not remaining:
            return finalise_session(experiment_id)
        print(f"\nDone for this boot. {len(remaining)} runs left "
              f"({', '.join(remaining)}).", flush=True)
        print("Restart the machine, then run this command again. This tool "
              "will not restart it for you.", flush=True)
        return 0


def _attempt_summary(item: dict) -> dict:
    body = item["body"]
    gate = body.get("gate") or {}
    return {"index": body.get("index"), "event": body.get("event"),
            "file_name": item["file_name"], "recorded_at": body.get(
                "recorded_at"),
            "boot_fingerprint": body.get("boot_fingerprint"),
            # A gate that never released, and an abort before the spawn, both
            # left the machine as they found it. Only a measurement spends the
            # boot it happened in.
            "spent_the_boot": body.get("event") not in EVENT_RETRYABLE,
            "gate_passed": gate.get("passed"),
            "gate_polls": len(gate.get("polls") or []),
            "gate_waited_seconds": gate.get("waited_seconds"),
            "gate_reason": (gate.get("reason") or body.get("abort_reason")
                            or body.get("reason")),
            "outcome": body.get("outcome"),
            "exit_status": body.get("exit_status")}


#: Everything ``session_experiment_state`` derives, and where reading the
#: aggregate back checks it. The replay refuses a session whose derivation
#: produces a field this table does not account for: an unchecked conclusion in
#: a stored file is exactly what the whole replay path exists to prevent.
SESSION_DERIVED_FIELDS = {
    "children": "compared slot by slot against COMPARED_SESSION_CHILD_KEYS",
    "boot_fingerprints": "compared with the aggregate's",
    "complete": "compared with the aggregate's",
    "stopped_reason": "compared with the aggregate's",
    "elapsed_seconds": "compared with the aggregate's",
    "terminal_outcomes": ("follows from each child's outcome, which is "
                          "compared slot by slot"),
}

#: Every field of a session's child slot that the journal determines. The
#: aggregate stores them; reading it back derives them again and compares.
COMPARED_SESSION_CHILD_KEYS = (
    "run_id", "condition", "block_id", "global_position", "exit_status",
    "not_run", "never_attempted", "outcome", "report_path", "report_sha256",
    "started_at", "finished_at", "parent_observed_wall_seconds",
    "recovery_passed_recomputed", "boot_fingerprint", "attempts", "gate")

#: Outcomes that explain, from the journal, why a run that started has no
#: report. Only these excuse a missing report, and only for a session.
#: ``never_finished`` is not written by anything -- it is what the replay calls
#: a started event with no finished event beside it.
TERMINAL_OUTCOMES = ("nonzero_exit", "timed_out", "no_report",
                     "never_finished")

#: The closed set a finished event may record. Read off two facts the parent
#: always has: what the process exited with, and whether a report exists.
FINISHED_OUTCOMES = (OUTCOME_COMPLETED, "nonzero_exit", "timed_out",
                     "no_report")


def session_experiment_state(ctx: dict, state: dict) -> dict:
    """The child slots and the verdicts the journal implies, derived once.

    ``finalise_session`` builds the aggregate from this and reading one back
    compares against it, so there is no second place where "what the journal
    means" could be written down slightly differently.
    """
    plan = ctx["plan"]
    children = []
    for entry in plan:
        run = entry["run_id"]
        attempts = [_attempt_summary(i) for i in state["attempts"].get(run, [])]
        begin = state["started"].get(run)
        end = state["finished"].get(run)
        slot = {"entry": entry, "attempts": attempts,
                **{k: entry[k] for k in ("run_id", "condition", "block_id",
                                         "global_position")}}
        if begin is None:
            slot |= {
                "exit_status": None,
                "gate": next((i["body"]["gate"]
                              for i in reversed(state["attempts"].get(run, []))
                              if i["body"].get("gate")),
                             {"passed": False, "polls": [],
                              "waited_seconds": None,
                              "consecutive_passes_required": ctx[
                                  "gate_policy"][
                                      "consecutive_passes_required"],
                              "reason": "never attempted"}),
                "recovery_passed_recomputed": False,
                "started_at": None, "finished_at": None,
                "parent_observed_wall_seconds": None,
                "report_path": None, "report_sha256": None, "not_run": True,
                "never_attempted": not attempts, "outcome": None,
                "boot_fingerprint": None}
        else:
            body = end["body"] if end else {}
            slot |= {
                "exit_status": body.get("exit_status"),
                "gate": begin["body"]["gate"],
                "recovery_passed_recomputed": bool(
                    begin["body"]["gate"].get("passed")),
                "started_at": body.get("started_at"),
                "finished_at": body.get("finished_at"),
                "parent_observed_wall_seconds": body.get("wall_seconds"),
                "report_path": body.get("report_path"),
                "report_sha256": body.get("report_sha256"), "not_run": False,
                "never_attempted": False,
                "outcome": body.get("outcome") or "never_finished",
                "boot_fingerprint": begin["body"].get("boot_fingerprint")}
        children.append(slot)

    missing = [c["run_id"] for c in children if c["not_run"]]
    stopped = state["terminal_reason"]
    if stopped is None and missing:
        stopped = (f"{len(missing)} of {len(plan)} runs never ran: "
                   + ", ".join(missing))
    return {
        "children": children,
        "boot_fingerprints": [c["boot_fingerprint"] for c in children
                              if not c["not_run"] and c["boot_fingerprint"]],
        "terminal_outcomes": {c["run_id"]: c["outcome"] for c in children
                              if c["outcome"] in TERMINAL_OUTCOMES},
        "complete": (not missing and not state["terminal"]
                     and all(c["exit_status"] == 0 for c in children)),
        "stopped_reason": stopped,
        "elapsed_seconds": round(
            sum(c.get("parent_observed_wall_seconds") or 0
                for c in children if not c["not_run"]), 2),
    }


def finalise_session(experiment_id: str, *, ctx=None, state=None) -> int:
    """Turn a finished or stopped journal into its record.

    Refuses a session that can still continue, before writing anything. An
    aggregate for a session with runs left is a report of an experiment that
    has not happened yet, and once it exists the idempotence check will hold
    every later finalise to it.
    """
    if ctx is None or state is None:
        # Snapshot-only: finalising is how a session that crashed gets its
        # record, and by then the tree has usually moved on. Starting a run is
        # the operation that needs the live tree to still match.
        ctx, state = read_journal(experiment_id, check_working_tree=False)
    paths, plan = ctx["paths"], ctx["plan"]

    if not state["terminal"] and len(state["completed"]) != len(plan):
        done = state["completed"]
        raise SystemExit(
            f"{experiment_id} can still continue: {len(done)} of {len(plan)} "
            f"runs are done ({', '.join(done) or 'none'}) and nothing has "
            "gone terminal. Finalising now would write a report for an "
            "experiment that has not finished. Nothing was written.")

    derived = session_experiment_state(ctx, state)
    children = []
    for slot in derived["children"]:
        record = None
        if slot["report_path"]:
            record = json.loads((ROOT / slot["report_path"]).read_text())
        children.append({**slot, "record": record})

    experiment = {
        "schema_version": EXPERIMENT_SCHEMA, "kind": "experiment",
        "experiment_id": experiment_id, "created_at": now_iso(),
        "one_run_per_boot": True,
        "boot_fingerprints": derived["boot_fingerprints"],
        "journal_events": [_attempt_summary(i) for i in state["events"]],
        # The aggregate points at the journal rather than standing in for it.
        # Reading it back re-runs the state machine over these files, so the
        # boot evidence comes from the events and not from the summary above.
        "session_path": str(paths["session"].relative_to(ROOT)),
        "session_sha256": sha256_file(paths["session"]),
        "source_snapshot_dir": ctx["session"].get("source_snapshot_dir",
                                                  SNAPSHOT_DIR),
        "source_manifest_digest": ctx["session"].get("source_manifest_digest"),
        "journal_files": [{"file_name": i["file_name"],
                           "sha256": i["file_sha256"]}
                          for i in state["events"]],
        "terminal": state["terminal"],
        "plan": plan, "plan_digest": ctx["plan_digest"],
        "plan_path": str(paths["plan"].relative_to(ROOT)),
        "calibration_digest": ctx["session"]["calibration_digest"],
        "thresholds": ctx["thresholds"], "gate_policy": ctx["gate_policy"],
        "thresholds_baseline": ctx["session"].get("thresholds_baseline"),
        "thresholds_baseline_note": ctx["session"].get(
            "thresholds_baseline_note"),
        "children": children,
        "complete": derived["complete"],
        "stopped_reason": derived["stopped_reason"],
        "elapsed_seconds": derived["elapsed_seconds"],
    }
    experiment = analyse(experiment, ctx["calibration"])
    fresh = _strip_records(experiment)

    # Finalising twice must not rewrite the record, and must not quietly
    # accept a different one. Either the file on disk is what deriving it
    # again produces, or something changed that nobody accounted for.
    if paths["aggregate"].exists():
        stored = json.loads(paths["aggregate"].read_text())
        diffs = sorted(
            k for k in (set(stored) | set(fresh))
            - set(AGGREGATE_VOLATILE_FIELDS)
            if canonical(stored.get(k)) != canonical(fresh.get(k)))
        if diffs:
            raise SystemExit(
                f"{paths['aggregate']} already exists and is not what "
                "deriving it again produces. Fields that differ: "
                + ", ".join(diffs)
                + "\nRefusing to overwrite it and refusing to endorse it.")
        sha = sha256_file(paths["aggregate"])
        print(f"{paths['aggregate']} was already final and still derives to "
              f"itself (sha256 {sha[:12]}...); nothing was rewritten")
    else:
        sha = atomic_write_json(paths["aggregate"], fresh)
        print(f"wrote {paths['aggregate']}")
    _write_markdown(experiment, paths["markdown"],
                    source={"aggregate_path": str(
                        paths["aggregate"].relative_to(ROOT)),
                        "aggregate_sha256": sha,
                        "recomputed_after_the_run": False})
    print(f"wrote {paths['markdown']}")
    for problem in experiment["replay_problems"]:
        print(f"  replay problem: {problem}")
    if not experiment["complete"]:
        print(f"INCOMPLETE: {experiment['stopped_reason']}")
    return 0 if experiment["complete"] else 1


def run_session_finalize(experiment_id: str) -> int:
    """Produce the record for a session that stopped, without re-running it.

    The path out of a crash. A parent that died between starting a measurement
    and reporting it leaves a journal that says exactly that, and this turns it
    into an aggregate and a report that say it too. Nothing is retried: the
    experiment is over, and what it is owed is an honest record of how far it
    got.
    """
    paths = session_paths(experiment_id)
    with exclusive_lock(paths["lock"], description=f"session {experiment_id}"):
        return finalise_session(experiment_id)


def run_session_status(experiment_id: str) -> int:
    paths = session_paths(experiment_id)
    with exclusive_lock(paths["lock"], description=f"session {experiment_id}"):
        ctx, state = read_journal(experiment_id)
        plan = ctx["plan"]
        boot = boot_identity(experiment_id)
        print(f"session {experiment_id}: {len(state['completed'])} of "
              f"{len(plan)} runs measured")
        for entry in plan:
            run = entry["run_id"]
            attempts = state["attempts"].get(run, [])
            begin = state["started"].get(run)
            end = state["finished"].get(run)
            outcome = (end["body"].get("outcome") if end else
                       "started, never finished" if begin else "pending")
            print(f"  {run} {entry['condition']:<12} "
                  f"{outcome:<24} attempts={len(attempts)} boot="
                  f"{(begin['body'].get('boot_fingerprint') if begin else '-')}")
        print(f"this boot: {boot['boot_fingerprint']} "
              f"(via {boot['source']})")
        if state["terminal"]:
            print(f"  TERMINAL: {state['terminal_reason']}")
            print("  This experiment cannot continue. Start a new experiment "
                  "id with the full plan.")
        elif boot["boot_fingerprint"] in state["boots"]:
            print("  a measured run has already happened in this boot: "
                  "restart before the next one")
    return 0


# --------------------------------------------------------------------------
# recomputation, written beside the parent's aggregate and never over it
# --------------------------------------------------------------------------

RECOMPUTED_NAME = "aggregate.recomputed.json"

#: Conclusions in the parent's aggregate that the recomputation re-derives.
#: Any difference is reported rather than resolved: the recomputed file is the
#: one to read, and which one to believe is not a question a diff should be
#: allowed to answer silently.
RECOMPUTED_COMPARISON_KEYS = ("complete", "stopped_reason", "thresholds",
                              "gate_policy", "plan_digest",
                              "calibration_digest", "verdicts", "losses")


def differences_from_stored(stored: dict, recomputed: dict) -> list[dict]:
    out = []
    for key in RECOMPUTED_COMPARISON_KEYS:
        if canonical(stored.get(key)) != canonical(recomputed.get(key)):
            out.append({"field": key,
                        "in_parent_aggregate": stored.get(key),
                        "recomputed": recomputed.get(key)})
    return out


def build_recomputed(exp: dict, exp_dir: Path, stored: dict) -> dict:
    """Re-derive the whole experiment from the files the parent left behind.

    The parent wrote its aggregate as it stopped, in the same process that had
    just been through a gate timeout. This re-derives every conclusion from the
    plan, the calibration samples, the child reports and the gate polls, in a
    separate pass with nothing else in scope, and writes it to its own file.
    The parent's aggregate is left exactly as it was, digest included: a record
    of what was concluded at the time is not improved by being edited later.
    """
    exp_dir = Path(exp_dir)
    agg_path = exp_dir / "aggregate.json"
    plan_path = exp_dir / "plan.json"
    calib = exp.get("calibration") or {}
    sources = {
        "aggregate_path": str(agg_path.relative_to(ROOT)),
        "aggregate_sha256": sha256_file(agg_path),
        "plan_path": str(plan_path.relative_to(ROOT)),
        "plan_sha256": sha256_file(plan_path),
        "calibration_path": calib.get("path"),
        "calibration_sha256": calib.get("sha256"),
        "children": [
            {"run_id": _slot_run_id(c), "report_path": c.get("report_path"),
             "report_sha256": c.get("report_sha256"),
             "measured": _slot_record(c) is not None}
            for c in exp["children"]],
    }
    out = _strip_records(exp)
    out["kind"] = "experiment_recomputed"
    out["recomputed_after_the_run"] = True
    out["recomputed_at"] = now_iso()
    out["recomputed_from"] = sources
    out["recomputed_note"] = (
        "Derived from the plan, the calibration samples, the child reports and "
        "the gate polls recorded at the time. Thresholds are recomputed from "
        "the samples rather than read back, every poll is re-judged against "
        "them, and every verdict follows from raw fields. The parent's "
        "aggregate.json is unchanged and its digest is recorded above.")
    out["differences_from_parent_aggregate"] = differences_from_stored(
        stored, out)
    return out


def write_recomputed(experiment_id: str) -> int:
    exp_dir = RUN_DIR / experiment_id
    exp = load_experiment(exp_dir)
    stored = json.loads((exp_dir / "aggregate.json").read_text())
    recomputed = build_recomputed(exp, exp_dir, stored)
    out_path = exp_dir / RECOMPUTED_NAME
    sha = atomic_write_json(out_path, recomputed)
    _write_markdown(exp, REPORT_DIR / "15_mps_order.md",
                    source={"aggregate_path": str(out_path.relative_to(ROOT)),
                            "aggregate_sha256": sha,
                            "recomputed_after_the_run": True,
                            "parent_aggregate_path":
                                recomputed["recomputed_from"]["aggregate_path"],
                            "parent_aggregate_sha256":
                                recomputed["recomputed_from"][
                                    "aggregate_sha256"]})
    print(f"wrote {out_path}")
    print(f"  parent aggregate left untouched at "
          f"{recomputed['recomputed_from']['aggregate_path']} "
          f"(sha256 {recomputed['recomputed_from']['aggregate_sha256'][:12]}"
          "...)")
    diffs = recomputed["differences_from_parent_aggregate"]
    print(f"  differences from the parent's conclusions: {len(diffs)}")
    for d in diffs:
        print(f"    - {d['field']}")
    print(f"re-rendered {REPORT_DIR / '15_mps_order.md'} from {RECOMPUTED_NAME}")
    return 0


#: Two fields of the recomputation describe the moment it was made rather than
#: the experiment: when it ran, and what the working tree looked like then.
#: They are excluded from the agreement check and re-derived at render time, so
#: nothing is read back from the stored file either way.
RECOMPUTED_VOLATILE_FIELDS = ("recomputed_at", "code_on_disk")


def check_recomputed_matches(exp_dir: Path, exp: dict, stored_agg: dict,
                             ) -> list[str]:
    """A stored recomputation may only be cited if it still recomputes.

    The file is a derived artefact sitting in a writable tree, and citing it by
    name in the report is exactly the kind of endorsement that has to be
    earned. So it is rebuilt from the raw records here and compared field by
    field; a disagreement is a stop, not a warning, because the whole reason
    the report points at that file is that it is supposed to be the derivation.
    """
    path = exp_dir / RECOMPUTED_NAME
    if not path.exists():
        return [f"{RECOMPUTED_NAME} does not exist"]
    try:
        stored = json.loads(path.read_text())
    except Exception as exc:
        return [f"{RECOMPUTED_NAME} is unreadable ({exc})"]
    fresh = build_recomputed(exp, exp_dir, stored_agg)
    problems = []
    keys = set(stored) | set(fresh)
    for key in sorted(keys - set(RECOMPUTED_VOLATILE_FIELDS)):
        if canonical(stored.get(key)) != canonical(fresh.get(key)):
            problems.append(
                f"{RECOMPUTED_NAME}: `{key}` is not what recomputing it now "
                "produces")
    return problems


# --------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------

def _write_markdown(experiment: dict, path: Path, source: dict | None = None,
                    ) -> None:
    e = experiment
    kids = e["children"]
    directions = e.get("block_directions") or block_directions(e)
    comparison = e.get("cross_child_comparison") or compare_children(kids)
    contracts = e.get("treatment_contract") or {
        _slot_run_id(c): check_treatment_contract(_slot_record(c),
                                                  _slot_run_id(c))
        for c in kids}
    # Recomputed here rather than read back, even though `analyse` already
    # stored one: the gate turns on `complete` and `stopped_reason`, and a
    # headline rendered from a verdict reached before those were last touched
    # is a headline that describes a different experiment.
    headline = headline_gate(e, comparison, contracts, e["losses"],
                             e["verdicts"], directions)

    L = [f"# MPS ordering experiment ({MAX_ROWS} rows x 4 processes)", ""]
    L.append(
        "Report 14 could not separate `empty_cache` from `ran second`: both "
        "arms shared a process and the clearing arm went last. Here each arm "
        "runs in its own Python process and each condition runs twice, once "
        "early and once late, so condition is crossed with execution "
        "position.")
    if source:
        L += ["", "## Rendered from", ""]
        if source.get("recomputed_after_the_run"):
            L += [f"`{source['aggregate_path']}` "
                  f"(sha256 `{source['aggregate_sha256']}`) -- the "
                  "**recomputed** aggregate, re-derived after the run from the "
                  "plan, the calibration samples, the child reports and the "
                  "gate polls.", "",
                  f"The parent's own aggregate, "
                  f"`{source.get('parent_aggregate_path')}` (sha256 "
                  f"`{source.get('parent_aggregate_sha256')}`), is unchanged "
                  "and was not written to. Where the two differ it is listed "
                  "in `differences_from_parent_aggregate` in the recomputed "
                  "file; nothing below is read back from either, it is all "
                  "derived again from the raw records."]
            if source.get("recomputation_reverified"):
                L += ["", "That recomputed file was rebuilt from the raw "
                      "records before it was cited here and matched field for "
                      "field, apart from when it was made and what the working "
                      "tree looked like then -- both of which are re-derived "
                      "below rather than read from it."]
            for note in e.get("superseded_conclusions") or []:
                L += ["", f"- {note}"]
        else:
            L += [f"`{source['aggregate_path']}` "
                  f"(sha256 `{source['aggregate_sha256']}`), written by the "
                  "parent as the experiment finished."]
    L += ["", "## Declared before running", "",
          "| position | block | condition | run | in-block |",
          "|---:|---|---|---|---:|"]
    for p in e["plan"]:
        L.append(f"| {p['global_position']} | {p['block_id']} | "
                 f"`{p['condition']}` | {p['run_id']} | "
                 f"{p['position_in_block']} |")
    L += ["", f"Plan digest `{e['plan_digest']}`, written atomically before "
          "the first child was spawned and re-checked by every child and "
          "again here. Each condition has mean global position 1.5.", "",
          f"Calibration digest `{e['calibration_digest']}`.", ""]

    L += ["## Outcome", "",
          f"- complete: **{e['complete']}**",
          f"- stopped reason: `{e['stopped_reason']}`",
          f"- elapsed: {e['elapsed_seconds']}s", ""]

    L += ["## Per run", "",
          "`compute` is the summed timed regions; `end-to-end` adds the "
          "between-row work. They are never quoted against each other.", "",
          "| run | block | pos | condition | rows | first window s/row | "
          "last window s/row | degradation | compute s | end-to-end s | "
          "load s | exit |",
          "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in kids:
        s = _summarise(c["record"])
        en = c["entry"]
        if not s:
            L.append(f"| {en['run_id']} | {en['block_id']} | "
                     f"{en['global_position']} | `{en['condition']}` | "
                     "- | - | - | - | - | - | - | "
                     f"{c['exit_status']} |")
            continue
        L.append(
            f"| {en['run_id']} | {en['block_id']} | {en['global_position']} | "
            f"`{en['condition']}` | {s['rows_completed']} | "
            f"{s['first_window_seconds_per_row']} | "
            f"{s['last_window_seconds_per_row']} | "
            f"{s['degradation_ratio']}x | {s['model_compute_seconds']} | "
            f"{s['end_to_end_seconds']} | {s['model_load_seconds']} | "
            f"{c['exit_status']} |")

    L += ["", "### Clears and overhead", "",
          "| run | condition | scheduled clears | scheduled clear s (total / "
          "mean / max) | teardown clears | teardown s | probe s | "
          "unattributed s |",
          "|---|---|---:|---|---:|---:|---:|---:|"]
    for c in kids:
        s = _summarise(c["record"])
        en = c["entry"]
        if not s:
            continue
        cost = s["scheduled_empty_cache_cost"] or {}
        br = s["between_row_overhead_breakdown"] or {}
        L.append(f"| {en['run_id']} | `{en['condition']}` | "
                 f"{s['scheduled_empty_cache_calls']} | "
                 f"{cost.get('total_seconds')} / {cost.get('mean_seconds')} / "
                 f"{cost.get('max_seconds')} | "
                 f"{s['teardown_empty_cache_calls']} | "
                 f"{s['teardown_empty_cache_seconds']} | "
                 f"{br.get('memory_probe_seconds')} | "
                 f"{br.get('unattributed_seconds')} |")
    L += ["", "Teardown clears run after the condition's clock has stopped. "
          "They are counted and timed apart and are inside none of the "
          "figures above.", ""]

    L += ["### Memory", "",
          "`free+inactive` is what `vm_stat` allows adding up; inactive pages "
          "are reclaimable, so it is neither free nor available memory.", "",
          "| run | condition | driver min/max/end GB | over recommended max | "
          "swap start/end GB | pressure start/min/end %free | "
          "least free+inactive GB |",
          "|---|---|---|---:|---|---|---:|"]
    for c in kids:
        s = _summarise(c["record"])
        en = c["entry"]
        if not s:
            continue
        L.append(
            f"| {en['run_id']} | `{en['condition']}` | "
            f"{s['driver_min_gb']} / {s['driver_max_gb']} / "
            f"{s['driver_end_gb']} | "
            f"{s['samples_over_recommended_max']}/{s['memory_samples']} | "
            f"{s['swap_start_gb']} / {s['swap_end_gb']} | "
            f"{s['pressure_start_percent_free']} / "
            f"{s['pressure_min_percent_free']} / "
            f"{s['pressure_end_percent_free']} | "
            f"{s['free_plus_inactive_min_gb']} |")

    # Paired contrast by block, and position effect.
    by_run = {c["entry"]["run_id"]: _summarise(c["record"]) for c in kids}
    L += ["", "## Paired contrast, by block", "",
          "| block | continuous last-window s/row | empty_cache last-window "
          "s/row | direction |", "|---|---:|---:|---|"]
    for block, d in directions["blocks"].items():
        L.append(f"| {block} | {d['continuous_seconds_per_row']} | "
                 f"{d['empty_cache_seconds_per_row']} | {d['direction']} |")

    L += ["", "## Execution position", "",
          "| position | run | condition | last window s/row |",
          "|---:|---|---|---:|"]
    for p in e["plan"]:
        s = by_run.get(p["run_id"]) or {}
        L.append(f"| {p['global_position']} | {p['run_id']} | "
                 f"`{p['condition']}` | "
                 f"{s.get('last_window_seconds_per_row')} |")

    L += ["", "## Reading this", ""]
    if headline["allowed"] and directions["directions"][0] == (
            "empty_cache faster"):
        L.append(
            "**The direction is the same in both blocks.** With each arm in "
            "its own process and each condition run once early and once late, "
            "the arm that cleared the cache was the faster one in both "
            "orders. On this machine and this configuration that is a "
            "**repeatedly observed engineering mitigation**.")
    elif headline["allowed"]:
        L.append(
            "**Every precondition holds and the two blocks agree**, and what "
            f"they agree on is `{directions['directions'][0]}`. That is the "
            "result, whichever way it points.")
    elif e.get("complete") and not directions["consistent"]:
        L.append(
            "**The direction is not the same in both blocks.** The effect is "
            "entangled with execution position, and this design has not "
            "separated them. It does not license any statement about "
            "`empty_cache` helping.")
    elif e.get("complete"):
        L.append(
            "**The runs finished, but the result is not comparable.** What "
            "failed is listed below; until each of those holds, this is a set "
            "of measurements rather than a contrast.")
    else:
        L.append(
            "**The experiment did not complete, so nothing is compared.** The "
            "runs that finished are recorded above and are not to be read as "
            "a partial result: the contrast needs all four.")
    L += ["", "A headline is allowed only when all of these hold:", ""]
    for req in headline["requirements"]:
        L.append(f"- {req}")
    if headline["allowed"]:
        L += ["", "All of them do."]
    else:
        L += ["", "**Not allowed here.** What failed:", ""]
        for reason in headline["reasons"]:
            L.append(f"- {reason}")
    L += ["",
          "This says nothing about *why*. Retained cache, allocator "
          "fragmentation, unified-memory pressure and swap thrash are all "
          "consistent with these readings and none is separated here. Two "
          "runs per condition support an engineering judgement, not a "
          "variance estimate and not a causal claim.", ""]

    losses = e["losses"]
    L += ["## Losses", "",
          f"Pre-declared tolerance **{losses['tolerance']}** absolute.",
          "", losses["tolerance_basis"], "",
          f"- comparable pairs: **{losses['comparable_pairs']}**",
          f"- largest absolute per-row difference across all pairs: "
          f"**{losses['max_abs_loss_diff_overall']}**",
          f"- verdict: **{losses['verdict']}**",
          f"- token/supervised-token mismatches: "
          f"**{losses['token_count_mismatches_total']}**", ""]
    if losses["comparable_pairs"]:
        L += ["| pair | shared rows | max abs diff | mean abs diff | rows over "
              "tolerance | token mismatches |",
              "|---|---:|---:|---:|---:|---:|"]
        for p in losses["pairs"]:
            L.append(f"| {p['a']} vs {p['b']} | {p['shared_rows']} | "
                     f"{p['max_abs_loss_diff']} | {p['mean_abs_loss_diff']} | "
                     f"{p['rows_over_tolerance']} | "
                     f"{p['token_count_mismatches']} |")
    if losses["verdict"] == "not_applicable":
        L += ["Fewer than two runs produced a report, so there is no pair to "
              "compare and **no loss verdict exists**. This is not a run that "
              "passed the tolerance, and it is not one that failed it."]
    elif losses["verdict"] == "over_tolerance":
        L += ["**At least one pair exceeds the declared tolerance.** The "
              "whole set therefore may not be described as having followed "
              "equivalent training trajectories. Every run is kept and "
              "reported; no block is dropped on the strength of its own "
              "loss spread."]
    L += ["", "## Eligibility", "",
          "Derived from raw fields. A child does not state its own exit "
          "status, recovery or comparability, and the parent's stored verdict "
          "does not decide it here either: exit status comes from the process "
          "table, recovery from replaying the gate's own polls.", "",
          "| run | eligible | reasons |", "|---|---|---|"]
    for run_id, v in e["verdicts"].items():
        L.append(f"| {run_id} | {v['eligible_for_paired_contrast']} | "
                 + ("; ".join(v["reasons"]) or "-") + " |")

    L += ["", "## What was compared across runs", "",
          comparison["note"], ""]
    if comparison["comparable"]:
        L += ["| field | agrees |", "|---|---|"]
        for field, v in comparison["fields"].items():
            L.append(f"| `{field}` | {v['agrees']} |")
        L.append("")
    if comparison["compared_runs"]:
        L += ["| run | rows | the rows it ran are the full declared "
              "order |", "|---|---:|---|"]
        for run, v in comparison["completed_declared_order"].items():
            L.append(f"| {run} | {v['rows_completed']}/{v['rows_requested']} | "
                     f"{v['matches']} |")

    code = e.get("code_on_disk") or code_on_disk_check(kids)
    if code:
        L += ["", "### The source that ran", "",
              "What each run recorded before it loaded anything, checked "
              "against the working tree as this was written.", "",
              "| run | files | still on disk unchanged | changed since the "
              "run | source preserved |", "|---|---:|---:|---:|---|"]
        for run, v in code.items():
            L.append(f"| {run} | {v['files_recorded']} | "
                     f"{len(v['unchanged'])} | "
                     f"{len(v['changed_since_the_run'])} | "
                     f"{v['source_preserved']} |")
        changed = {run: v for run, v in code.items()
                   if v["changed_since_the_run"] or v["missing_from_the_tree"]}
        if changed:
            L += ["", "**Not every run can be reproduced from what is on disk "
                  "now.** The digests below identify the code that produced "
                  "the measurement; they do not preserve it, and no copy was "
                  "kept:", ""]
            for run, v in changed.items():
                for f in v["changed_since_the_run"]:
                    L.append(f"- {run} ran `{f['file']}` at "
                             f"`{f['ran_with'][:12]}...`, on disk now "
                             f"`{f['on_disk_now'][:12]}...`")
                for f in v["missing_from_the_tree"]:
                    L.append(f"- {run} ran `{f}`, which is no longer in the "
                             "tree")
    if comparison["disagreements"]:
        L += ["", "**Runs disagree on: "
              + ", ".join(f"`{d}`" for d in comparison["disagreements"])
              + ".** Only the condition and its clear schedule are allowed to "
              "differ between runs."]

    if e.get("one_run_per_boot"):
        L += ["", "## Boots and attempts", "",
              "One measured run per boot. Boots are identified by a hash "
              "keyed to this experiment id; the machine's own boot identifier "
              "is never written down. Every attempt is listed, including the "
              "ones where the gate never released -- those started no child "
              "and spent no boot.", "",
              "| run | boot | attempts | outcome |", "|---|---|---:|---|"]
        for c in kids:
            L.append(f"| {_slot_run_id(c)} | "
                     f"{c.get('boot_fingerprint') or '-'} | "
                     f"{len(c.get('attempts') or [])} | "
                     f"{c.get('outcome') or ('never ran' if c.get('not_run') else '-')} |")
        if e.get("terminal"):
            L += ["", f"**This experiment is over.** {e.get('stopped_reason')} "
                  "A measured run that started and did not complete cannot be "
                  "repeated in any boot; a new experiment id with the full "
                  "plan is the only way forward."]

    L += ["", "## Treatment contract", "",
          "`continuous` performs no scheduled clear at all. `empty_cache` "
          f"clears on every {EMPTY_CACHE_EVERY}th row and on no other row. "
          "The row numbers are checked, not only the count, and the per-call "
          "costs have to add up to the total they are reported as.", "",
          "| run | condition | clears | on the scheduled rows | clear s "
          "(rows / recorded) | teardown s | s outside the row spans | teardown "
          "outside the clock |",
          "|---|---|---:|---|---|---:|---:|---|"]
    for c in kids:
        run_id = _slot_run_id(c)
        k = contracts.get(run_id) or {}
        if not k.get("checked"):
            continue
        L.append(
            f"| {run_id} | `{k['condition']}` | {k['clear_calls']} | "
            f"{k['clears_observed_at'] == k['clears_expected_at']} | "
            f"{k['clear_seconds_from_rows']} / {k['clear_total_seconds']} | "
            f"{k['teardown_seconds']} | {k['seconds_outside_row_spans']} | "
            f"{k['teardown_outside_condition_clock']} |")
    contract_problems = [p for k in contracts.values()
                         for p in k.get("problems", [])]
    if contract_problems:
        L += ["", "**Contract failures:**", ""]
        L += [f"- {p}" for p in contract_problems]

    L += ["", "## Gate", "",
          "Thresholds come from one idle sampling pass, held fixed. That pass "
          "loads no model, which is a property of the code path; that nothing "
          "else had loaded one is an operating condition the operator "
          "arranged, and is **not** something these readings can establish. A "
          "run may not start until the machine has been inside the band on "
          f"{e['gate_policy']['consecutive_passes_required']} consecutive "
          f"polls {e['gate_policy']['poll_seconds']}s apart, within "
          f"{e['gate_policy']['max_wait_seconds']}s. Every threshold below is "
          "recomputed from the calibration samples, and every poll is judged "
          "against them again here rather than read back.", "",
          "| metric | threshold |", "|---|---:|"]
    for m, t in (e.get("thresholds_recomputed") or e["thresholds"]).items():
        L.append(f"| `{m}` | {t} |")
    L += ["", "| run | polls | waited s | passed (replayed) | child preflight "
          "inside the band |", "|---|---:|---:|---|---|"]
    replays = e.get("gate_replay") or {}
    preflights = e.get("child_preflight_checks") or {}
    for c in kids:
        run_id = _slot_run_id(c)
        g = c["gate"]
        r = replays.get(run_id) or {}
        pre = preflights.get(run_id) or {}
        L.append(f"| {run_id} | {len(g['polls'])} | {g['waited_seconds']} | "
                 f"{r.get('passed', g['passed'])} | "
                 f"{pre.get('passed') if pre.get('checked') else '-'} |")
    L += ["", "### Limits", "",
          "- One machine, one macOS and torch version, one sitting.",
          "- Thermal state is not measured, only balanced by the design.",
          "- swap and page cache are balanced across conditions, not reset.",
          "- Two runs per condition. That is an engineering signal, not a "
          "variance estimate.",
          "- A fresh process resets this process's allocator, heap, model and "
          "optimizer. It does not prove the OS or the driver returned to a "
          "comparable state; the gate is evidence about the OS-level "
          "readings, not about driver internals.", ""]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(line.rstrip() for line in L).rstrip("\n") + "\n",
                    encoding="utf-8")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--from-json", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--session-init", action="store_true")
    ap.add_argument("--session-next", action="store_true")
    ap.add_argument("--session-finalize", action="store_true")
    ap.add_argument("--session-status", action="store_true")
    ap.add_argument("--experiment-id")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--plan")
    ap.add_argument("--plan-digest")
    ap.add_argument("--session-dir")
    ap.add_argument("--global-position", type=int)
    ap.add_argument("--run-id")
    ap.add_argument("--condition")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    if args.calibrate:
        out = Path(args.out or RUN_DIR / "calibration.json")
        rec = run_calibration(out)
        print(f"\nwrote {out}")
        for m, s in rec["stats"].items():
            print(f"  {m}: median={s['median']} mad={s['mad']} "
                  f"scale={s['scale']} threshold={s['threshold']}")
        return 0

    if args.child:
        if not args.session_dir:
            raise SystemExit(
                "--child needs --session-dir: it verifies the plan digest and "
                "the source manifest against the session it belongs to before "
                "it reads anything else.")
        plan_doc = json.loads(Path(args.plan).read_text())
        plan = plan_doc["plan"]
        if digest_obj(plan) != args.plan_digest:
            raise SystemExit("plan digest does not match the plan file")
        if plan_doc.get("plan_digest") != args.plan_digest:
            raise SystemExit("plan file's own digest does not match")
        entry = plan[args.global_position]
        if (entry["run_id"] != args.run_id
                or entry["condition"] != args.condition):
            raise SystemExit(
                f"plan entry {args.global_position} is "
                f"{entry['run_id']}/{entry['condition']}, but this child was "
                f"told {args.run_id}/{args.condition}")
        return run_child(entry, args.plan_digest, Path(args.out),
                         Path(args.session_dir))

    if args.run:
        raise SystemExit(RUN_WITHDRAWN)

    if args.from_json:
        if not args.experiment_id:
            raise SystemExit("--from-json needs --experiment-id")
        exp_dir = RUN_DIR / args.experiment_id
        # `load_experiment` re-derives everything; nothing here is read back
        # from a stored conclusion. A recomputed aggregate may be cited only
        # after it has been shown to still recompute to itself.
        exp = load_experiment(exp_dir)
        agg_path = exp_dir / "aggregate.json"
        recomputed = exp_dir / RECOMPUTED_NAME
        source = {"aggregate_path": str(agg_path.relative_to(ROOT)),
                  "aggregate_sha256": sha256_file(agg_path),
                  "recomputed_after_the_run": False}
        if recomputed.exists():
            stored_agg = json.loads(agg_path.read_text())
            problems = check_recomputed_matches(exp_dir, exp, stored_agg)
            if problems:
                raise SystemExit(
                    "the stored recomputation does not match a fresh one:\n  "
                    "- " + "\n  - ".join(problems)
                    + f"\nRefusing to cite it. Delete {RECOMPUTED_NAME} and "
                      "run --recompute again, or read the report against the "
                      "plan, calibration, child reports and gate polls "
                      "directly.")
            source = {"aggregate_path": str(recomputed.relative_to(ROOT)),
                      "aggregate_sha256": sha256_file(recomputed),
                      "recomputed_after_the_run": True,
                      "recomputation_reverified": True,
                      "parent_aggregate_path": str(agg_path.relative_to(ROOT)),
                      "parent_aggregate_sha256": sha256_file(agg_path)}
        _write_markdown(exp, REPORT_DIR / "15_mps_order.md", source=source)
        print(f"re-rendered {REPORT_DIR / '15_mps_order.md'} from "
              f"{source['aggregate_path']} (thresholds recomputed from the "
              "calibration samples, every gate poll re-judged, children "
              "re-hashed against the digests recorded at run time"
              + (", stored recomputation re-derived and matched"
                 if source["recomputed_after_the_run"] else "") + ")")
        return 0

    if args.verify:
        if not args.experiment_id:
            raise SystemExit("--verify needs --experiment-id")
        exp = load_experiment(RUN_DIR / args.experiment_id)
        print(f"{args.experiment_id}: plan, calibration, gate polls, child "
              "digests and child records all check out")
        print(f"  thresholds recomputed from "
              f"{exp['calibration_replay']['samples_seen']} calibration "
              f"samples: {exp['thresholds_recomputed']}")
        for run_id, g in exp["gate_replay"].items():
            print(f"  {run_id}: {len(g['polls'])} polls replayed, "
                  f"passed={g['passed']}, waited {g['waited_seconds']}s")
        if "journal_replay" in exp:
            j = exp["journal_replay"]
            print(f"  journal replayed: {j['events_replayed']} events, "
                  f"completed {j['completed']}, terminal={j['terminal']}")
            print(f"  boot fingerprints from the journal: "
                  f"{j['boot_fingerprints']}")
        print(f"  headline allowed: {exp['headline']['allowed']}")
        for reason in exp["headline"]["reasons"]:
            print(f"    - {reason}")
        for note in exp["superseded_conclusions"]:
            print(f"  superseded: {note}")
        return 0

    if args.recompute:
        if not args.experiment_id:
            raise SystemExit("--recompute needs --experiment-id")
        return write_recomputed(args.experiment_id)

    if args.session_init:
        if not args.experiment_id:
            raise SystemExit("--session-init needs --experiment-id")
        return run_session_init(
            args.experiment_id,
            Path(args.calibration or RUN_DIR / "calibration.json"))

    if args.session_next:
        if not args.experiment_id:
            raise SystemExit("--session-next needs --experiment-id")
        return run_session_next(args.experiment_id)

    if args.session_finalize:
        if not args.experiment_id:
            raise SystemExit("--session-finalize needs --experiment-id")
        return run_session_finalize(args.experiment_id)

    if args.session_status:
        if not args.experiment_id:
            raise SystemExit("--session-status needs --experiment-id")
        return run_session_status(args.experiment_id)

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
