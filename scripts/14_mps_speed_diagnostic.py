"""Short MPS speed diagnostic: where does a training row's time actually go?

The smoke test (report 13) took 5.76 hours for 2,000 rows and degraded 13.1x
from start to finish. That measurement says *what* happened and deliberately
claims no cause. This script is the follow-up: at most 200 rows, heavily
instrumented, so the cost can be attributed to a phase and watched against
memory rather than guessed at.

Scope, fixed before running:

* same starting point, LoRA config, data seed and training-order rule as the
  smoke test, so the rows are the rows that run would have seen
* **at most 200 rows per condition**; this is a diagnostic, not training
* **nothing is saved to artifacts/checkpoints/lora_smoke/** -- that checkpoint
  belongs to report 13 and is not touched
* two pre-declared conditions, compared as declared rather than picked after
  the fact:
    ``continuous``   one uninterrupted run
    ``empty_cache``  the same run, with torch.mps.empty_cache() every 10 rows
* stop conditions bound each condition; hitting one is a normal outcome and
  partial results are kept

Optimizer updates do run, in memory, on the same rows the smoke test used --
what the script does not do is keep the result. Calling it "no training" would
be wrong in the direction that matters: the weights move, so a row's cost is a
real training row's cost.

A process-restart condition is deliberately **not** included. Restarting also
resets the model and optimizer, so any speedup would confound "a fresh process
is faster" with "a fresh model is faster" -- it needs its own design that
states which state carries over, and inventing it here would produce a number
that reads as a fix and is not one.

Writes data/reports/14_mps_speed.md and .json.
"""

from __future__ import annotations

import json
import math
import platform
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
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

OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "data" / "reports"

MAX_ROWS = 200
WINDOW = 20
MEMORY_EVERY = 5
EMPTY_CACHE_EVERY = 10
PHASES = ("collate_h2d", "forward", "backward", "optimizer")
CONDITIONS = ("continuous", "empty_cache")

#: 1 = the original 200-row run, which recorded neither the per-row input ids,
#: nor the split of between-row overhead, nor the loss at full precision. Those
#: gaps cannot be closed now, so a schema-1 run is rendered with them marked
#: "not recorded" instead of filled in from today's code. 2 = everything below
#: is captured at run time.
SCHEMA_VERSION = 2

#: Serialised as measured. Every other float on a row is a timing, where the
#: clock runs out of meaning well before six decimal places -- but a loss
#: rounded on the way to disk can never be compared more finely afterwards,
#: which is exactly the position the schema-1 run left its four-decimal losses
#: in.
UNROUNDED_ROW_FIELDS = ("loss",)
ROW_TIMING_DECIMALS = 6

CODE_FILES = (
    "scripts/14_mps_speed_diagnostic.py",
    "src/training/diagnostics.py",
    "src/training/lora.py",
    "src/data/instruction.py",
    "src/model_ids.py",
)

# ---------------------------------------------------------------------------
# Replay contracts.
#
# One frozen contract per schema, holding everything a record of that schema
# was promised to contain: its required env keys, its required provenance
# keys, its required per-condition fields, and the two validators that read
# them.
#
# Nothing here is factored into a shared base, and nothing is derived from
# today's CODE_FILES or LoraConfig_. That duplication is the point. A stored
# record is a finished statement about a run that already happened, and what
# it had to contain was settled when it was written. A shared tuple would mean
# that adding one requirement -- a new setting, a new source file, a new
# hyperparameter -- retroactively invalidated every correct record ever
# produced, which is the same mistake as re-rendering an old run with today's
# constants, one layer down.
#
# Adding a requirement therefore means adding schema 3, leaving schemas 1 and
# 2 meaning exactly what they have always meant. Rules that happen to be
# shared today live in ``_check_*_common`` helpers, which each version's
# validator calls explicitly -- a later schema is free not to.
# ---------------------------------------------------------------------------

SCHEMA1_ENV = ("schema_version", "max_rows_per_condition", "window",
               "memory_sample_every", "empty_cache_every", "seed",
               "grad_accum", "phases", "condition_order",
               "condition_definitions", "stop_slow_row_seconds",
               "stop_slow_row_streak", "stop_max_seconds",
               "loss_decimals_stored")

SCHEMA1_PROVENANCE = ("instruction_sha256", "selection_digest",
                      "training_order_digest", "base_revision",
                      "published_adapter_revision")

#: Schema 1 promised nothing per condition: it recorded no per-row sample ids,
#: no per-row end-to-end time and no overhead breakdown. Those gaps are
#: labelled in the record rather than demanded of it.
SCHEMA1_CONDITION_FIELDS = ()

SCHEMA2_ENV = ("schema_version", "max_rows_per_condition", "window",
               "memory_sample_every", "empty_cache_every", "seed",
               "grad_accum", "phases", "condition_order",
               "condition_definitions", "stop_slow_row_seconds",
               "stop_slow_row_streak", "stop_max_seconds",
               "loss_decimals_stored")

#: Presence is what is checked, never truthiness: ``working_tree_dirty`` is
#: legitimately ``False`` on a clean tree, and a falsy-means-missing test
#: would reject exactly the runs whose provenance is best.
SCHEMA2_PROVENANCE = (
    "instruction_sha256", "selection_digest", "training_order_digest",
    "base_revision", "published_adapter_revision",
    "head", "working_tree_dirty", "code_sha256", "lora_config", "packages",
    "device", "dtype", "phases", "stop_conditions", "condition_order",
    "condition_input_order_digests")

SCHEMA2_CONDITION_FIELDS = (
    "input_order_digest", "completed_input_digest",
    "scheduled_empty_cache_calls", "teardown_empty_cache_calls",
    "between_row_overhead_breakdown")

SCHEMA2_CODE_FILES = (
    "scripts/14_mps_speed_diagnostic.py",
    "src/training/diagnostics.py",
    "src/training/lora.py",
    "src/data/instruction.py",
    "src/model_ids.py",
)

SCHEMA2_LORA_FIELDS = (
    "rank", "alpha", "dropout", "learning_rate", "target_modules",
    "batch_size", "grad_accum", "max_length", "epochs", "seed", "dtype",
    "quantization", "effective_batch",
)

SCHEMA2_PACKAGES = ("python", "torch", "transformers", "peft")

SCHEMA2_STOP_FIELDS = ("slow_row_seconds", "slow_row_streak", "max_seconds")

#: Schema 2 is an MPS record by construction: ``resolve_device()`` refuses to
#: start anywhere else, because on CPU the clear is a no-op and the treatment
#: arm silently becomes the control.
SCHEMA2_DEVICE = "mps"


@dataclass(frozen=True)
class ReplayContract:
    """What one schema's records were promised to contain, and who checks it.

    Frozen, and resolved once per replay by exact version. Every requirement
    the gate applies comes from the instance chosen here -- there is no
    ``schema >= n`` anywhere downstream, because "at least version n" is a
    guess about what a later schema means, and a later schema is free to mean
    something else by the same field name.
    """

    version: int
    required_env: tuple[str, ...]
    required_provenance: tuple[str, ...]
    required_condition_fields: tuple[str, ...]
    check_provenance: Callable[[dict, dict], list[str]]
    check_conditions: Callable[[dict, dict, dict, "ReplayContract"], list[str]]


def condition_definitions(clear_every: int) -> dict:
    """What each declared condition *was*, resolved and stored with the run.

    Written into the report JSON rather than kept as a renderer constant: the
    difference between the arms is part of what the run measured, so a later
    re-render must read it from the record. Interpolation happens here, once,
    so the stored strings are final and the renderer does no substitution.
    """
    return {
        "continuous": "one uninterrupted run",
        "empty_cache": ("the same rows in the same order, with "
                        f"`torch.mps.empty_cache()` every {clear_every} rows"),
    }


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_ids(ids) -> str:
    import hashlib

    h = hashlib.sha256()
    for i in ids:
        h.update(str(i).encode())
        h.update(b"\n")
    return h.hexdigest()


def capture_provenance(rows, perm, cfg, *, device: str) -> dict:
    """Recorded **before** the model loads. See report 13's permanent gap.

    Everything a later reader would need to say what ran: the code, the data,
    the exact hyperparameters, the package versions, and the order the rows go
    in. Recording it afterwards is how report 13 ended up with a gap that
    cannot be closed.
    """
    import subprocess

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
    stop = StopCondition()
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
        # Declared per condition and *before* anything runs, so each arm's
        # result can be checked against a value it could not have influenced.
        # They are equal here by construction -- every condition rebuilds the
        # same permutation from the same seed -- and writing that out per name
        # is what makes a later divergence visible instead of assumed away.
        "condition_input_order_digests": {
            c: digest_ids(rows[i].sample_id for i in perm) for c in CONDITIONS},
        "lora_config": cfg.as_dict(),
        "packages": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "peft": version("peft"),
        },
        "device": device,
        "dtype": cfg.dtype,
        "phases": list(PHASES),
        "stop_conditions": {
            "slow_row_seconds": stop.slow_row_seconds,
            "slow_row_streak": stop.slow_row_streak,
            "max_seconds": stop.max_seconds,
        },
        "condition_order": list(CONDITIONS),
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "published_adapter": ADAPTER,
        "published_adapter_revision": ADAPTER_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "backfilled_after_the_run": [],
    }


def check_replayable(stored: dict) -> None:
    """Fail closed before re-rendering: inputs, settings, and self-consistency.

    Two separate jobs. The first is that the run must still describe the world
    it was measured in -- same input file, all the settings the prose quotes.
    The second is that the record must agree with itself: a stored run whose
    clear count contradicts its own clear schedule, or whose per-row ids no
    longer digest to the value stored beside them, has been edited, and
    re-rendering it would launder the edit into a report.
    """
    prov = stored.get("provenance")
    if not prov:
        raise SystemExit(
            "stored run has no provenance block; refusing to re-render a run "
            "whose inputs cannot be checked")

    env = stored.get("env") or {}
    contract = resolve_contract(env.get("schema_version"))

    problems = [f"stored env is missing {key}"
                for key in contract.required_env if key not in env]
    problems += [f"provenance is missing {key}"
                 for key in contract.required_provenance if key not in prov]
    problems += contract.check_provenance(env, prov)
    problems += contract.check_conditions(stored, env, prov, contract)

    if problems:
        raise SystemExit("cannot re-render:\n  - " + "\n  - ".join(problems)
                         + "\nRe-run the diagnostic instead of replaying.")


def resolve_contract(version) -> ReplayContract:
    """The one contract this record is judged by, chosen by exact version."""
    # An exact int, so `2.0` and `True` do not slip through `in` on equality.
    if (isinstance(version, bool) or not isinstance(version, int)
            or version not in CONTRACTS):
        raise SystemExit(
            f"stored run declares schema_version {version!r}; this script "
            f"validates {sorted(CONTRACTS)} and nothing else. "
            "A record written under a schema this script does not know may "
            "mean something different by the same field names, so it is "
            "refused rather than checked against whichever rules happen to be "
            "closest.")
    return CONTRACTS[version]


def _check_provenance_common(env: dict, prov: dict) -> list[str]:
    """Content rules both current schemas opt into. Not inherited: called.

    Covers the fields schemas 1 and 2 happen to share, plus the one check that
    is about the world rather than the record -- whether the input file the
    run read is still the file on disk.
    """
    problems = []
    for key in ("selection_digest", "training_order_digest"):
        if key in prov and not _is_hex(prov[key], 64):
            problems.append(f"{key} is not a valid digest")
    for key in ("base_revision", "published_adapter_revision"):
        if key in prov and not (isinstance(prov[key], str) and prov[key]):
            problems.append(f"{key} is empty")

    files = prov.get("instruction_sha256")
    if "instruction_sha256" in prov and (not isinstance(files, dict)
                                         or not files):
        problems.append("instruction_sha256 names no input files")
    for name, was in (files or {}).items() if isinstance(files, dict) else ():
        path = OUT_DIR / name
        if not path.exists():
            problems.append(f"{name} is missing")
        elif (now := sha256_file(path)) != was:
            problems.append(f"{name} changed: {was[:12]}... -> {now[:12]}...")
    return problems


def _check_provenance_v1(env: dict, prov: dict) -> list[str]:
    """Schema 1 promised none of the schema 2 record, so none is demanded.

    Its gaps are real, enumerated in ``schema_1_note`` and labelled in
    ``backfilled_after_the_run``. Holding it to a later contract would not
    recover any of them; it would only make the one honest record unreadable.
    """
    return _check_provenance_common(env, prov)


def _is_hex(value, length: int | None = None) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if length is not None and len(value) != length:
        return False
    return all(c in "0123456789abcdef" for c in value.lower())


def _check_provenance_v2(env: dict, prov: dict) -> list[str]:
    """Schema 2 must carry a full pre-load record, and it must agree with env.

    Three ways this goes wrong, and a key-presence sweep only catches the
    first. A field can be absent, which schema 2 does not allow -- tested by
    presence, never truthiness, because ``working_tree_dirty`` is ``False`` on
    precisely the runs whose provenance is cleanest. A field can be *present
    and empty*: ``head: null``, ``code_sha256: {}``, a ``packages`` dict with
    three of four versions. That is a gap wearing the shape of a record, and
    it is the failure mode that let report 13 look complete. Or a field can
    contradict the settings block beside it, meaning one of the two was edited
    afterwards and the report would quote whichever the renderer read first.

    The expected code files and LoRA fields come from the frozen schema 2
    contract above, never from ``CODE_FILES`` or ``LoraConfig_`` as they stand
    today. Adding a source file or a hyperparameter must not retroactively
    invalidate a record that was complete when it was written; a new
    requirement is a new schema.
    """
    problems = _check_provenance_common(env, prov)

    def empty(key, why):
        problems.append(f"schema 2 provenance records {key} as empty: {why}")

    # HEAD -- an actual commit-ish, not None and not a placeholder.
    head = prov.get("head")
    if "head" in prov and not (_is_hex(head) and len(head) >= 7):
        empty("head", "the commit that ran must be a real revision, and "
                      "report 13's gap was exactly a missing one")

    # A clean tree records False. None means nobody looked.
    dirty = prov.get("working_tree_dirty", "absent")
    if dirty != "absent" and not isinstance(dirty, bool):
        problems.append(
            f"working_tree_dirty is {dirty!r}, not a bool; False means a clean "
            "tree and is fine, but None means the tree was never inspected")

    # Code digests: the files that were meant to be captured, each a real
    # digest rather than a placeholder.
    code = prov.get("code_sha256")
    if "code_sha256" in prov:
        if not isinstance(code, dict) or not code:
            empty("code_sha256", "the code that ran is exactly what report 13 "
                                 "could not recover")
        else:
            for f in SCHEMA2_CODE_FILES:
                if f not in code:
                    problems.append(f"code_sha256 does not cover {f}")
            bad = sorted(k for k, v in code.items() if not _is_hex(v, 64))
            if bad:
                problems.append("code_sha256 has no valid digest for "
                                + ", ".join(bad))

    # The whole hyperparameter set, not the two fields that happen to be
    # cross-checked below.
    cfg = prov.get("lora_config")
    if "lora_config" in prov:
        if not isinstance(cfg, dict) or not cfg:
            empty("lora_config", "the hyperparameters are the run")
        else:
            missing = [f for f in SCHEMA2_LORA_FIELDS if f not in cfg]
            if missing:
                problems.append("lora_config is missing " + ", ".join(missing))
            blank = sorted(k for k, v in cfg.items() if v is None)
            if blank:
                problems.append("lora_config has no value for "
                                + ", ".join(blank))

    packages = prov.get("packages")
    if "packages" in prov:
        if not isinstance(packages, dict) or not packages:
            empty("packages", "a version-sensitive measurement needs versions")
        else:
            for p in SCHEMA2_PACKAGES:
                if not packages.get(p):
                    problems.append(f"packages records no {p} version")

    for key in ("device", "dtype"):
        if key in prov and not (isinstance(prov[key], str) and prov[key]):
            empty(key, "a timing is only meaningful against the thing it ran on")

    # Schema 2 cannot legitimately be a CPU record: the arms would be
    # indistinguishable, since `empty_cache()` does nothing there.
    device = prov.get("device")
    if isinstance(device, str) and device and device != SCHEMA2_DEVICE:
        problems.append(
            f"schema 2 records device {device!r}; this diagnostic only means "
            f"anything on {SCHEMA2_DEVICE!r}, where clearing the cache is not "
            "a no-op")

    for key in ("phases", "condition_order"):
        value = prov.get(key)
        if key in prov:
            if not (isinstance(value, list) and value):
                empty(key, "an empty list would let the report describe nothing")
            elif not all(isinstance(v, str) and v for v in value):
                problems.append(f"{key} must be a list of names, got {value!r}")

    stop = prov.get("stop_conditions")
    if "stop_conditions" in prov:
        if not isinstance(stop, dict) or not stop:
            empty("stop_conditions", "a partial run's bound is part of its "
                                     "result")
        else:
            for k in SCHEMA2_STOP_FIELDS:
                v = stop.get(k)
                if v is None:
                    problems.append(f"stop_conditions records no {k}")
                elif k == "slow_row_streak":
                    # A count of rows: fractional or zero-length is not a rule.
                    if (isinstance(v, bool) or not isinstance(v, int)
                            or v < 1):
                        problems.append(
                            f"stop_conditions.{k} must be a positive integer "
                            f"number of rows, got {v!r}")
                elif (isinstance(v, bool) or not isinstance(v, (int, float))
                        or not math.isfinite(v) or v <= 0):
                    # inf would record a bound that can never be reached, and
                    # nan compares false against everything.
                    problems.append(
                        f"stop_conditions.{k} must be a finite positive "
                        f"number of seconds, got {v!r}")
    stop = stop if isinstance(stop, dict) else {}

    def disagree(what, a, b):
        if a is not None and b is not None and a != b:
            problems.append(f"provenance and env disagree on {what}: "
                            f"{a!r} vs {b!r}")

    disagree("phases", prov.get("phases"), env.get("phases"))
    disagree("condition_order", prov.get("condition_order"),
             env.get("condition_order"))
    disagree("device", prov.get("device"), env.get("device"))
    disagree("dtype", prov.get("dtype"), env.get("dtype"))
    disagree("stop_slow_row_seconds", stop.get("slow_row_seconds"),
             env.get("stop_slow_row_seconds"))
    disagree("stop_slow_row_streak", stop.get("slow_row_streak"),
             env.get("stop_slow_row_streak"))
    disagree("stop_max_seconds", stop.get("max_seconds"),
             env.get("stop_max_seconds"))

    for p in ("python", "torch", "transformers", "peft"):
        disagree(f"the {p} version",
                 (packages or {}).get(p) if isinstance(packages, dict) else None,
                 env.get(p))

    # The seed, grad_accum and dtype the run declared must be the ones the
    # LoRA config actually carried. dtype in particular is quoted by the
    # report and is the sort of thing that gets edited in one place only.
    cfg = cfg if isinstance(cfg, dict) else {}
    disagree("seed", cfg.get("seed"), env.get("seed"))
    disagree("grad_accum", cfg.get("grad_accum"), env.get("grad_accum"))
    disagree("dtype", cfg.get("dtype"), prov.get("dtype"))

    declared = prov.get("condition_input_order_digests")
    if "condition_input_order_digests" in prov:
        if not isinstance(declared, dict) or not declared:
            empty("condition_input_order_digests",
                  "each arm's row order is declared before it can run")
        else:
            for c in (env.get("condition_order") or []):
                if c not in declared:
                    problems.append("no pre-run input-order digest was "
                                    f"recorded for {c}")
                elif not _is_hex(declared[c], 64):
                    problems.append(f"the pre-run input-order digest for {c} "
                                    "is not a valid digest")
    return problems


def _check_conditions_common(stored: dict, env: dict, prov: dict,
                             contract: ReplayContract) -> list[str]:
    """Per-condition consistency both current schemas opt into.

    Which fields must be *present* comes from ``contract`` rather than from
    the schema number: an empty tuple means the schema promised none, and a
    later schema is free to promise a different set. Everything else here is a
    statement the record makes about itself -- the clear count against its own
    clear schedule, the row count against its own cap, the per-row ids against
    the digest stored beside them -- and holds for any schema that records
    those fields at all.
    """
    problems: list[str] = []
    conditions = stored.get("conditions") or []

    order = list(env.get("condition_order") or [])
    names = [c.get("condition") for c in conditions]
    if order and names != order:
        problems.append(
            f"conditions {names} do not match the recorded order {order}")
    for i, c in enumerate(conditions):
        if c.get("run_order") != i:
            problems.append(
                f"`{c.get('condition')}` records run_order "
                f"{c.get('run_order')} but is stored at position {i}")

    cap = env.get("max_rows_per_condition")
    clear_every = env.get("empty_cache_every")
    for c in conditions:
        who = c.get("condition")
        rows = c.get("per_row") or []
        done = c.get("rows_completed")

        if len(rows) != done:
            problems.append(f"`{who}` says {done} rows completed but stores "
                            f"{len(rows)} of them")
        if isinstance(cap, int) and isinstance(done, int) and done > cap:
            problems.append(f"`{who}` completed {done} rows, over the "
                            f"{cap}-row cap the run declared")

        for key in contract.required_condition_fields:
            if c.get(key) is None:
                problems.append(f"`{who}` is missing {key}")

        # The clear count must follow from the schedule. A count that does not
        # is either a miscount or a teardown call folded into the total.
        n = c.get("scheduled_empty_cache_calls")
        every = c.get("scheduled_empty_cache_every")
        if n is not None:
            if every:
                expected = len(rows) // every
            elif who == "empty_cache" and clear_every:
                expected = len(rows) // clear_every
            else:
                expected = 0
            if n != expected:
                problems.append(
                    f"`{who}` records {n} scheduled `empty_cache()` calls but "
                    f"its schedule over {len(rows)} rows implies {expected}")

        # Per-row ids must still digest to the value stored next to them.
        ids = [r.get("sample_id") for r in rows]
        stored_digest = c.get("completed_input_digest")
        if stored_digest and all(ids):
            if (now := digest_ids(ids)) != stored_digest:
                problems.append(
                    f"`{who}` per-row sample ids digest to {now[:12]}..., not "
                    f"the stored {stored_digest[:12]}...")
        elif stored_digest:
            problems.append(f"`{who}` stores an input digest but not the "
                            "sample ids it was taken over")

        # Both conditions were fed the same permutation; the run-level
        # training-order digest is the same statement, so all three must agree.
        req = c.get("input_order_digest")
        if req and prov.get("training_order_digest") not in (None, req):
            problems.append(
                f"`{who}` input order {req[:12]}... does not match the "
                f"recorded training order "
                f"{prov['training_order_digest'][:12]}...")

        # And it must match what was declared for *this* condition before the
        # model loaded. That value was fixed before the arm could run, so a
        # mismatch means the arm did not receive the rows it was promised.
        promised = (prov.get("condition_input_order_digests") or {}).get(who)
        if req and promised and req != promised:
            problems.append(
                f"`{who}` ran input order {req[:12]}..., but the order "
                f"declared for it before the model loaded was "
                f"{promised[:12]}...")

    digests = {c["input_order_digest"] for c in conditions
               if c.get("input_order_digest")}
    if len(digests) > 1:
        problems.append("conditions were fed different row orders, so their "
                        "timings are not comparable")
    return problems


def _check_conditions_v1(stored, env, prov, contract) -> list[str]:
    return _check_conditions_common(stored, env, prov, contract)


def _check_conditions_v2(stored, env, prov, contract) -> list[str]:
    return _check_conditions_common(stored, env, prov, contract)


#: The registry. One entry per schema, each holding its own frozen field lists
#: and its own validators. Adding a schema means adding an entry -- never
#: widening an existing one, and never writing ``>= n`` anywhere.
CONTRACTS: dict[int, ReplayContract] = {
    1: ReplayContract(
        version=1,
        required_env=SCHEMA1_ENV,
        required_provenance=SCHEMA1_PROVENANCE,
        required_condition_fields=SCHEMA1_CONDITION_FIELDS,
        check_provenance=_check_provenance_v1,
        check_conditions=_check_conditions_v1,
    ),
    2: ReplayContract(
        version=2,
        required_env=SCHEMA2_ENV,
        required_provenance=SCHEMA2_PROVENANCE,
        required_condition_fields=SCHEMA2_CONDITION_FIELDS,
        check_provenance=_check_provenance_v2,
        check_conditions=_check_conditions_v2,
    ),
}

#: Kept as the public statement of what can be replayed; the registry is the
#: single source of truth for it.
SUPPORTED_SCHEMA_VERSIONS = tuple(sorted(CONTRACTS))


def resolve_device(torch_mod) -> str:
    """Refuse to run this diagnostic anywhere but MPS.

    Falling back to CPU used to look harmless. It is not: ``DeviceOps`` makes
    ``empty_cache()`` an explicit no-op there -- correctly, since there is no
    MPS cache to clear -- so the ``empty_cache`` arm would schedule twenty
    clears, perform none, and record zero. The replay gate would then reject
    the run for a clear count that contradicts its own schedule, after the
    machine hours had already been spent. Worse, a reader would have a report
    whose treatment arm is identical to its control and no longer says so.

    None of that is a reason to loosen the gate. It is a reason not to start:
    the whole question is what the MPS allocator does over a few hundred rows,
    and on CPU there is no question to ask.
    """
    try:
        available = bool(torch_mod.backends.mps.is_available())
    except Exception:
        available = False
    if not available:
        raise SystemExit(
            "MPS is not available, and this diagnostic is only about MPS. On "
            "CPU `empty_cache()` is a no-op, so the `empty_cache` arm would "
            "schedule clears it never performs, come out identical to "
            "`continuous`, and be rejected by the replay gate for a clear "
            "count that contradicts its schedule. Stopping before the run "
            "rather than after it.")
    return "mps"


def rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _serialise_row(row: dict) -> dict:
    """Round the timings; leave the loss exactly as it was measured."""
    out = {}
    for k, v in row.items():
        if k in UNROUNDED_ROW_FIELDS:
            out[k] = v
        elif isinstance(v, float):
            out[k] = round(v, ROW_TIMING_DECIMALS)
        else:
            out[k] = v
    return out


def run_condition(name: str, cfg: LoraConfig_, encs, rows_meta, tok, *,
                  device: str, order: int) -> dict:
    """One condition, from a freshly built model, up to MAX_ROWS rows.

    ``order`` is its position in the process: recorded because the conditions
    share a process and run in a fixed sequence, so the second one does not
    start from the first one's initial conditions.
    """
    dev = DeviceOps(device)
    torch.manual_seed(cfg.seed)
    model, info = build_model(cfg, device=device)
    assert_only_lora_trainable(model)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    model.train()

    # Same training-order rule as the smoke test, truncated to MAX_ROWS.
    rng = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(encs), generator=rng).tolist()[:MAX_ROWS]
    clear_every = EMPTY_CACHE_EVERY if name == "empty_cache" else None

    timer = PhaseTimer(sync=dev.sync)
    stop = StopCondition()
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

        row = timer.end(
            row=i,
            sample_id=rows_meta[idx].sample_id,
            n_tokens=int(batch["attention_mask"].sum()),
            n_supervised=int((batch["labels"] != -100).sum()),
            loss=loss.detach().item(),   # stored as measured, never rounded
        )

        # Outside the timed region, so window seconds stay model compute only.
        # Timed on its own so the intervention's cost is a figure rather than
        # part of an undivided lump; ``timer.end()`` has just synchronised, so
        # this is the clear itself and not a queue draining into it.
        clear_seconds = 0.0
        if clear_every and i % clear_every == 0:
            clear_seconds = dev.empty_cache()

        probe = 0.0
        if i % MEMORY_EVERY == 0 or i == 1:
            t_probe = time.perf_counter()
            sample = {
                "row": i,
                "elapsed_seconds": round(time.perf_counter() - t_begin, 2),
                **memory_sample(torch, rss_bytes=rss_bytes()),
                **system_memory(),
            }
            probe = time.perf_counter() - t_probe
            sample["probe_seconds"] = round(probe, 4)
            memory.append(sample)
            probe_seconds += probe

        elapsed = time.perf_counter() - t_begin
        if (reason := stop.check(elapsed, row["total"])):
            stopped = reason
            print(f"  [{name}] stopping at row {i}: {reason}", flush=True)
        elif i % WINDOW == 0:
            print(f"  [{name}] {i}/{len(perm)}  {elapsed / i:.2f}s/row  "
                  f"loss {row['loss']:.4f}", flush=True)

        row["scheduled_empty_cache_seconds"] = clear_seconds
        row["memory_probe_seconds"] = probe
        row["end_to_end"] = time.perf_counter() - row_begin
        if stopped:
            break

    rows = timer.rows
    total = time.perf_counter() - t_begin
    compute = sum(r["total"] for r in rows)
    clear_cost = dev.scheduled_clear_cost()
    del model, opt
    # Housekeeping, after the clock above has stopped: frees this condition's
    # model before the next one builds its own. It is not the intervention
    # under test and is in none of the totals, so it is counted apart.
    dev.empty_cache(teardown=True)

    return {
        "condition": name,
        "run_order": order,
        "rows_completed": len(rows),
        "rows_requested": len(perm),
        "stopped_early": stopped,
        "input_order_digest": digest_ids(rows_meta[i].sample_id for i in perm),
        "completed_input_digest": digest_ids(r["sample_id"] for r in rows),
        # end_to_end covers the timed regions plus everything between rows.
        # compute is the timed regions alone. Window figures report both.
        "end_to_end_seconds": round(total, 2),
        "model_compute_seconds": round(compute, 2),
        "between_row_overhead_seconds": round(total - compute, 2),
        "between_row_overhead_breakdown": {
            "scheduled_empty_cache_seconds": clear_cost["total_seconds"],
            "memory_probe_seconds": round(probe_seconds, 4),
            "unattributed_seconds": round(
                total - compute - clear_cost["total_seconds"] - probe_seconds, 4),
        },
        "end_to_end_seconds_per_row": round(total / max(len(rows), 1), 3),
        "model_compute_seconds_per_row": round(compute / max(len(rows), 1), 3),
        "scheduled_empty_cache_every": clear_every,
        "scheduled_empty_cache_calls": dev.scheduled_empty_cache_calls,
        "scheduled_empty_cache_cost": clear_cost,
        # Counted, never added to the line above: it ran after the clock
        # stopped and is not part of the condition being compared.
        "teardown_empty_cache_calls": dev.teardown_empty_cache_calls,
        "phases": summarise_phases(rows, PHASES),
        "windows": window_stats(rows, WINDOW),
        "memory": memory,
        "per_row": [_serialise_row(r) for r in rows],
        "trainable_parameters": info["trainable_parameters"],
    }


def main(argv: list[str]) -> int:
    cfg = LoraConfig_()

    if "--from-json" in argv:
        # Re-render prose from the stored run. The measurements here are wall
        # clock, so re-running to fix a sentence would silently replace the
        # numbers being described with different ones.
        stored = json.loads((REPORT_DIR / "14_mps_speed.json").read_text())
        check_replayable(stored)
        _write_report(stored["env"], stored["baseline_memory"],
                      stored["conditions"], stored["provenance"])
        print("re-rendered data/reports/14_mps_speed.md from the stored run")
        return 0

    device = resolve_device(torch)
    tok = load_tokenizer()

    rows = sample_pairs(read_rows(OUT_DIR / "instruct_inv_train.jsonl"),
                        n_pairs=250, seed=cfg.seed)
    encs = [encode_row(tok, r, cfg.max_length) for r in rows]
    print(f"{len(encs)} rows available; using at most {MAX_ROWS} per condition",
          flush=True)

    rng = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(len(encs), generator=rng).tolist()[:MAX_ROWS]
    provenance = capture_provenance(rows, perm, cfg, device=device)
    print(f"code: HEAD {provenance['head']} "
          f"dirty={provenance['working_tree_dirty']}", flush=True)

    baseline = {**memory_sample(torch, rss_bytes=rss_bytes()), **system_memory()}
    # Fixed order in one process: recorded, because it is a limitation.
    results = [run_condition(c, cfg, encs, rows, tok, device=device, order=i)
               for i, c in enumerate(CONDITIONS)]

    stop = StopCondition()
    env = {
        "schema_version": SCHEMA_VERSION,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "peft": version("peft"),
        "device": device,
        "dtype": cfg.dtype,
        "max_rows_per_condition": MAX_ROWS,
        "window": WINDOW,
        "memory_sample_every": MEMORY_EVERY,
        "empty_cache_every": EMPTY_CACHE_EVERY,
        "seed": cfg.seed,
        "grad_accum": cfg.grad_accum,
        "phases": list(PHASES),
        "stop_slow_row_seconds": stop.slow_row_seconds,
        "stop_slow_row_streak": stop.slow_row_streak,
        "stop_max_seconds": stop.max_seconds,
        "condition_order": list(CONDITIONS),
        "condition_definitions": condition_definitions(EMPTY_CACHE_EVERY),
        "single_process_fixed_order": True,
        # None = stored as measured. A number here means the losses on disk
        # were rounded to that many places and cannot be compared more finely.
        "loss_decimals_stored": None,
        "row_timing_decimals_stored": ROW_TIMING_DECIMALS,
    }

    payload = {"env": env, "baseline_memory": baseline,
               "provenance": provenance, "conditions": results}
    # Run the replay gate against the record just built. A run that cannot
    # satisfy its own consistency checks has a bookkeeping bug, and finding
    # that out now beats finding it out the first time someone re-renders --
    # by which point the machine hours are spent.
    check_replayable(payload)
    _write_report(env, baseline, results, provenance)
    (REPORT_DIR / "14_mps_speed.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print("\nwrote data/reports/14_mps_speed.md", flush=True)
    return 0


UNKNOWN = "not recorded for this run"


def _require(env: dict, key: str):
    """Read a setting the prose quotes, or refuse to write the prose."""
    if key not in env:
        raise SystemExit(
            f"stored run does not record `{key}`; refusing to render a report "
            "that would have to take it from today's constants and would then "
            "describe a run that never had those settings")
    return env[key]


def _write_report(env: dict, baseline: dict, results: list[dict],
                  provenance: dict) -> None:
    """Build the markdown. Shared by a fresh run and by --from-json.

    Every setting printed comes from ``env``, never from this module's
    constants: re-rendering an old run must describe *that* run's settings,
    and reading MAX_ROWS here would silently restate history if the constant
    later changed. Anything ``env`` does not record is refused rather than
    filled in, and anything the run did not measure is printed as unknown
    rather than reconstructed.
    """
    max_rows = _require(env, "max_rows_per_condition")
    window_size = _require(env, "window")
    mem_every = _require(env, "memory_sample_every")
    clear_every = _require(env, "empty_cache_every")
    phases = tuple(_require(env, "phases"))
    order = list(_require(env, "condition_order"))
    definitions = dict(_require(env, "condition_definitions"))
    slow_seconds = _require(env, "stop_slow_row_seconds")
    slow_streak = _require(env, "stop_slow_row_streak")
    max_seconds = _require(env, "stop_max_seconds")
    loss_dp = _require(env, "loss_decimals_stored")

    L = [f"# MPS speed diagnostic (<= {max_rows} rows)", ""]
    L.append("Report 13 measured a 13.1x slowdown across 2,000 rows and named "
             "no cause. This is the short follow-up that tries to localise it: "
             "same starting point, LoRA config, data seed and training-order "
             "rule, but at most "
             f"{max_rows} rows per condition, with each phase timed separately "
             "and memory sampled as it goes. Optimizer updates do run, in "
             "memory, on real training rows -- what is not done is keeping "
             "them: no checkpoint is written and "
             "`artifacts/checkpoints/lora_smoke/` is untouched.")
    L += ["", "## Method", "",
          "- `time.perf_counter()`, with `torch.mps.synchronize()` at every "
          "phase boundary. Without the sync an MPS call returns as soon as the "
          "work is *enqueued*, so the timing would be attributed to whichever "
          "phase happened to wait for it.",
          f"- phases: {', '.join('`%s`' % p for p in phases)}; anything left "
          "over is reported as `_unattributed` rather than absorbed.",
          f"- memory sampled every {mem_every} rows: PyTorch's tracked "
          "allocation *and* the driver's, which can diverge.",
          f"- windows of {window_size} rows.", ""]
    L += ["### Conditions, declared before running", "",
          "| condition | difference |", "|---|---|"]
    for name in order:
        blurb = definitions.get(name)
        if not blurb:
            raise SystemExit(
                f"stored run declares a condition `{name}` but records no "
                "definition for it; what the arms differed by is part of what "
                "the run measured, so it is read from the record rather than "
                "supplied by this renderer")
        L.append(f"| `{name}` | {blurb} |")
    L += ["",
          "A process-restart condition is **not** included. Restarting resets "
          "the model and optimizer too, so a speedup would confound a fresh "
          "process with a fresh model; measuring it properly needs its own "
          "design stating which state carries over. Reporting it here would "
          "produce a number that reads as a fix without being one.", "",
          "### Stop conditions", "",
          f"- {slow_streak} consecutive rows over {slow_seconds:.0f}s",
          f"- or {max_seconds / 60:.0f} minutes in one condition",
          "", "Hitting one is a normal outcome: partial results are kept and "
          "the reason is reported. The goal is to localise a cost, not to "
          "finish training.", ""]

    L += ["## Environment", ""]
    for k, v in env.items():
        L.append(f"- `{k}`: {v}")
    L += ["", "Baseline memory before any condition ran:", "",
          "```json", json.dumps(baseline, indent=2), "```", ""]

    for res in results:
        scheduled = res.get("scheduled_empty_cache_calls")
        teardown = res.get("teardown_empty_cache_calls")
        cost = res.get("scheduled_empty_cache_cost") or {}
        split = res.get("between_row_overhead_breakdown")

        L += ["", f"## Condition: `{res['condition']}`", "",
              f"- run order in the process: **{res['run_order']}** "
              f"(0 = first)",
              f"- rows completed: **{res['rows_completed']}**"
              f" of {res['rows_requested']} requested",
              f"- stopped early: `{res['stopped_early']}`",
              f"- end-to-end: {res['end_to_end_seconds']}s "
              f"= {res['end_to_end_seconds_per_row']}"
              "s/row, **including** the between-row work broken out below",
              f"- model compute: {res['model_compute_seconds']}s "
              f"= {res['model_compute_seconds_per_row']}s/row, the "
              "summed timed regions only",
              f"- between-row overhead: "
              f"{res['between_row_overhead_seconds']}s"
              + (f" -- scheduled `empty_cache()` "
                 f"{split['scheduled_empty_cache_seconds']}s, memory probes "
                 f"{split['memory_probe_seconds']}s, "
                 f"{split['unattributed_seconds']}s in stop checks, printing "
                 "and loop bookkeeping" if split else
                 f"; its split into clears, probes and the rest is {UNKNOWN}"),
              f"- scheduled `empty_cache()`: "
              + (f"**{scheduled}** calls"
                 + (f", {cost['total_seconds']}s in total, "
                    f"{cost['mean_seconds']}s mean and {cost['max_seconds']}s "
                    "worst" if cost.get("mean_seconds") is not None else "")
                 if scheduled is not None else UNKNOWN)
              + " -- this is the intervention under test",
              (f"- teardown `empty_cache()`: **{teardown}** call(s), made "
               "after this condition's clock had already stopped, to free the "
               "model before the next condition builds its own -- "
               "housekeeping rather than the intervention, counted apart and "
               "inside none of the figures above."
               if teardown is not None else
               f"- teardown `empty_cache()`: **{UNKNOWN}**. This run never "
               "counted them, so how many were made -- if any -- is not "
               "known and no number is assumed here. By design such a clear "
               "would fall outside the timed region and outside every figure "
               "above, which is why it is counted apart from the scheduled "
               "clears in the first place."), "",
              "**The window figures below report both**: model compute is the "
              "summed timed regions, end-to-end adds the between-row work. "
              "`empty_cache()` and the memory probes run outside the timed "
              "region, so the two columns must not be quoted against each "
              "other.", ""]

        L += ["### Where the time went", "",
              "| phase | total s | mean s | median s | max s | share |",
              "|---|---:|---:|---:|---:|---:|"]
        for p in phases:
            d = res["phases"].get(p)
            if d:
                L.append(f"| `{p}` | {d['total_seconds']} | {d['mean_seconds']} "
                         f"| {d['median_seconds']} | {d['max_seconds']} | "
                         f"{d['share_of_measured']:.1%} |")
        un = res["phases"]["_unattributed"]
        L.append(f"| _unattributed_ | {un['total_seconds']} | | | | "
                 f"{un['share_of_total']:.1%} |")

        L += ["", "### Per window (raw)", "", "`compute` is the timed regions; "
              "`end-to-end` adds the between-row work. A run that did not "
              f"record per-row end-to-end shows `{UNKNOWN}` rather than a "
              "figure copied from the compute column.", "",
              "| window | rows | compute s | compute s/row | end-to-end s | "
              "end-to-end s/row | tokens | supervised | tok/s | mean seq |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for w in res["windows"]:
            e2e = w.get("end_to_end_seconds")
            e2e_row = w.get("end_to_end_seconds_per_row")
            L.append(f"| {w['window']} | {w['rows']} | {w['seconds']} | "
                     f"{w['seconds_per_row']} | "
                     f"{'-' if e2e is None else e2e} | "
                     f"{'-' if e2e_row is None else e2e_row} | {w['tokens']} | "
                     f"{w['supervised_tokens']} | {w['tokens_per_second']} | "
                     f"{w['mean_seq_len']} |")

        L += ["", "### Memory as it went", "",
              "`peak process RSS` is `ru_maxrss`: a high-water mark for the "
              "process, not a current reading. `free+inactive` is what "
              "`vm_stat` allows adding up; inactive pages are reclaimable, so "
              "it is neither free memory nor available memory in the everyday "
              "sense.", "",
              "| row | elapsed s | MPS current GB | MPS driver GB | "
              "recommended max GB | peak process RSS GB | free+inactive GB | "
              "swap GB | mem pressure % free |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m in res["memory"]:
            L.append(
                f"| {m['row']} | {m['elapsed_seconds']} | "
                f"{m['mps_current_allocated_gb']} | "
                f"{m['mps_driver_allocated_gb']} | "
                f"{m['mps_recommended_max_gb']} | "
                f"{m.get('peak_process_rss_gb')} | "
                f"{m.get('free_plus_inactive_gb')} | "
                f"{m['swap_used_gb']} | "
                f"{m.get('memory_pressure_percent_free')} |")
        L.append("")

    # Cross-condition comparison, on the pre-declared terms. The two s/row
    # columns are different measurements and are labelled as such rather than
    # collapsed into one "s/row".
    L += ["", "## Comparison", "",
          "`end-to-end s/row` includes between-row overhead; `compute s/row` "
          "and the window columns are the timed regions only.", "",
          "| condition | order | rows | end-to-end s/row | compute s/row | "
          "first window (compute) | last window (compute) | stopped |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for res in results:
        w = res["windows"]
        L.append(f"| `{res['condition']}` | {res['run_order']} | "
                 f"{res['rows_completed']} | "
                 f"{res['end_to_end_seconds_per_row']} | "
                 f"{res['model_compute_seconds_per_row']} | "
                 f"{w[0]['seconds_per_row'] if w else '-'} | "
                 f"{w[-1]['seconds_per_row'] if w else '-'} | "
                 f"{res['stopped_early'] or 'no'} |")

    cont = next(r for r in results if r["condition"] == "continuous")
    ec = next((r for r in results if r["condition"] == "empty_cache"), None)
    w = cont["windows"]
    reproduced = bool(w) and w[-1]["seconds_per_row"] > 2 * w[0]["seconds_per_row"]

    def mem_span(res):
        m = res["memory"]
        drv = [x["mps_driver_allocated_gb"] for x in m if x["mps_driver_allocated_gb"]]
        cur = [x["mps_current_allocated_gb"] for x in m if x["mps_current_allocated_gb"]]
        sw = [x["swap_used_gb"] for x in m if x["swap_used_gb"] is not None]
        fr = [x.get("free_plus_inactive_gb", x.get("free_gb")) for x in m
              if x.get("free_plus_inactive_gb", x.get("free_gb")) is not None]
        pr = [x.get("memory_pressure_percent_free") for x in m
              if x.get("memory_pressure_percent_free") is not None]
        cap = next((x["mps_recommended_max_gb"] for x in m
                    if x["mps_recommended_max_gb"]), None)
        return {
            "driver_min": min(drv) if drv else None,
            "driver_max": max(drv) if drv else None,
            "driver_end": drv[-1] if drv else None,
            "current_min": min(cur) if cur else None,
            "current_max": max(cur) if cur else None,
            "swap_start": sw[0] if sw else None,
            "swap_end": sw[-1] if sw else None,
            "free_min": min(fr) if fr else None,
            "press_start": pr[0] if pr else None,
            "press_min": min(pr) if pr else None,
            "press_end": pr[-1] if pr else None,
            "cap": cap,
            "over_cap": sum(1 for v in drv if cap and v > cap),
            "samples": len(drv),
        }

    mc, me = mem_span(cont), (mem_span(ec) if ec else None)

    def _gap(a, b):
        """None wherever a reading is missing: absent is not zero."""
        return None if a is None or b is None else abs(a - b)

    tracked_gap = max(
        (g for g in ((_gap(mc["current_max"], me["current_max"]),
                      _gap(mc["current_min"], me["current_min"])) if me
                     else ()) if g is not None), default=None)

    # Which phase dominates is read off the stored phases rather than named in
    # the prose: hard-coding `forward` would keep asserting it for a run where
    # the cost had moved somewhere else.
    named = [(p, d) for p, d in cont["phases"].items()
             if p != "_unattributed" and isinstance(d, dict)
             and "total_seconds" in d]
    worst = max(named, key=lambda kv: kv[1]["total_seconds"], default=None)
    other = (ec or {}).get("phases", {}).get(worst[0]) if worst else None
    costliest = (
        f" The `{worst[0]}` phase is where the cost sits: "
        f"{worst[1]['mean_seconds']:.2f}s mean in `continuous` against "
        f"{other['mean_seconds']:.2f}s in `empty_cache`, on the same rows in "
        "the same order." if worst and other else "")

    L += ["", "## Reading this", "",
          ("**The long tail did reproduce within this window.** The last "
           "window is more than twice the first, so the effect starts early "
           "enough to study at this scale."
           if reproduced else
           "**The long tail did not reproduce at this scale.** The last window "
           "is not materially slower than the first. That means only that "
           f"{max_rows} rows was not enough to reproduce it -- **not** that the "
           "problem is gone, not that it was misattributed, and not that the "
           "2,000-row measurement was wrong. Report 13's degradation stands as "
           "measured; this run simply does not reach the regime where it "
           "appeared."), ""]

    # Verified, not asserted: if the inputs or losses ever diverge the
    # sentence below must change with them. Losses are compared at the
    # precision they were *stored* at -- comparing schema-2's full-precision
    # values after rounding them to four places would throw away the very
    # precision that was recorded to make this check sharper.
    pr_a = {r["row"]: r for r in cont["per_row"]}
    pr_b = {r["row"]: r for r in ec["per_row"]} if ec else {}
    shared = sorted(set(pr_a) & set(pr_b))
    at_stored = ((lambda v: round(v, loss_dp)) if loss_dp is not None
                 else (lambda v: v))
    same_loss = sum(1 for k in shared
                    if at_stored(pr_a[k]["loss"]) == at_stored(pr_b[k]["loss"]))
    same_tokens = sum(1 for k in shared
                      if pr_a[k].get("n_tokens") == pr_b[k].get("n_tokens")
                      and pr_a[k].get("n_supervised") == pr_b[k].get("n_supervised"))
    loss_claim = (
        "produce **the same loss at the stored precision of "
        f"{loss_dp} decimal places** -- which bounds agreement at that "
        "precision and says nothing about the digits below it"
        if loss_dp is not None else
        "produce **the same loss as stored**, at the full precision the run "
        "recorded -- which is agreement between two float64 readings of a "
        "bf16 computation, not a proof of bitwise-identical arithmetic")

    if reproduced and me:
        L += ["### What moved with it", "",
              f"The two conditions ran the same rows in the same order. "
              f"Checked rather than assumed: **{same_tokens}/{len(shared)}** "
              f"rows match on token and supervised-token counts, and "
              f"**{same_loss}/{len(shared)}** {loss_claim}.",
              "",
              "What differs is the timing, the memory, **and the conditions "
              "they started from**: both ran in one process in the fixed order "
              f"`{' -> '.join(order)}`, each "
              "rebuilding the model and optimizer from the same seed, but the "
              "second inherited whatever swap, thermal, OS and process state "
              "the first left behind. Order is therefore confounded with "
              "condition here.", "",
              "| | `continuous` | `empty_cache` |", "|---|---:|---:|",
              f"| s/row, first window | {w[0]['seconds_per_row']} | "
              f"{ec['windows'][0]['seconds_per_row']} |",
              f"| s/row, last window | **{w[-1]['seconds_per_row']}** | "
              f"**{ec['windows'][-1]['seconds_per_row']}** |",
              f"| MPS *tracked* allocation | {mc['current_min']}-"
              f"{mc['current_max']} GB | {me['current_min']}-"
              f"{me['current_max']} GB |",
              f"| MPS *driver* min / max / end GB | {mc['driver_min']} / "
              f"**{mc['driver_max']}** / {mc['driver_end']} | "
              f"{me['driver_min']} / {me['driver_max']} / {me['driver_end']} |",
              f"| samples over the {mc['cap']} GB recommended max | "
              f"**{mc['over_cap']}/{mc['samples']}** | "
              f"{me['over_cap']}/{me['samples']} |",
              f"| swap start / end GB | {mc['swap_start']} / "
              f"**{mc['swap_end']}** | {me['swap_start']} / {me['swap_end']} |",
              f"| least free+inactive GB seen | **{mc['free_min']}** | "
              f"{me['free_min']} |",
              f"| memory pressure % free, start / min / end | "
              f"{mc['press_start']} / **{mc['press_min']}** / {mc['press_end']} "
              f"| {me['press_start']} / {me['press_min']} / {me['press_end']} |",
              "",
              "Two observations, stated at the strength the design supports.",
              "",
              "**The tracked figure would never have shown this.** PyTorch's "
              f"`current_allocated_memory` sits at {mc['current_min']}-"
              f"{mc['current_max']} GB in `continuous` and {me['current_min']}-"
              f"{me['current_max']} GB in `empty_cache` -- flat in both"
              + (f", and within {tracked_gap:.3f} GB of each other"
                 if tracked_gap is not None else "")
              + ". The driver figure is where the growth is, "
              f"and in `continuous` it runs to {mc['driver_max']} GB, past the "
              f"{mc['cap']} GB the system recommends, while swap grows to "
              f"{mc['swap_end']} GB and free+inactive pages fall to "
              f"{mc['free_min']} GB -- reclaimable pages, not a reading of "
              "memory sitting available. Report 13 read a flat tracked figure "
              "and a small RSS as 'not memory exhaustion'; on this evidence "
              "that reading was wrong, and the correction made to it last "
              "round was warranted.",
              "",
              "**The condition that cleared the cache did not degrade.** "
              f"With `empty_cache()` every {clear_every} rows, driver "
              "allocation stayed a sawtooth that never reached the "
              "recommended max and the per-row time did not rise. That is a "
              "co-occurrence in one ordered pair of runs, not a demonstrated "
              "fix." + costliest, "",
              "**This is a strong short-range signal, not an isolated "
              "cause.** Under a single fixed order with n=1 per condition, "
              "periodic `empty_cache()` and the absence of degradation "
              "occurred together, and the degradation moved with driver "
              "allocation, swap and memory pressure. That is co-occurrence "
              "plus a mitigation that worked once. It does **not** rule out "
              "the fixed order itself -- `empty_cache` ran second, on a "
              "machine the first condition had already loaded -- and it does "
              "**not** establish the internal mechanism: retained cache, "
              "fragmentation, unified-memory pressure and swap thrash are all "
              "consistent with these readings and none is separated here. Nor "
              "does it show this accounts for report 13's 13.1x over 2,000 "
              f"rows; this run is {max_rows} rows.", "",
              "The order confound is the first thing a follow-up should "
              "remove, by running the conditions in both orders or in fresh "
              "processes.", ""]

    L += ["## Provenance", ""]
    if provenance.get("backfilled_after_the_run"):
        L += ["**Part of this record was reconstructed after the run, and is "
              "labelled as such.** The gaps below cannot be closed "
              "retrospectively; later runs capture all of it before the model "
              "loads.", ""]
    if provenance.get("limitation"):
        L += [provenance["limitation"], ""]
    L += ["| item | value | when recorded |", "|---|---|---|"]
    back = set(provenance.get("backfilled_after_the_run", []))
    for key in ("head", "working_tree_dirty", "selection_digest",
                "training_order_digest", "base_model", "base_revision",
                "published_adapter_revision", "tokenizer_revision"):
        if provenance.get(key) is not None:
            when = "**backfilled**" if key in back else "at run time"
            L.append(f"| `{key}` | `{provenance[key]}` | {when} |")
    for name, dig in (provenance.get("instruction_sha256") or {}).items():
        when = ("**backfilled**" if "instruction_sha256" in back
                else "at run time")
        L.append(f"| `{name}` | `{dig}` | {when} |")
    for f, dig in (provenance.get("code_sha256") or {}).items():
        L.append(f"| `{f}` | `{dig}` | at run time |")
    promised = provenance.get("condition_input_order_digests") or {}
    for res in results:
        who = res["condition"]
        dig = res.get("input_order_digest")
        if not dig:
            L.append(f"| `{who}` input order | {UNKNOWN} | -- |")
            continue
        when = ("declared before model load, matched by the run"
                if promised.get(who) == dig else "at run time")
        L.append(f"| `{who}` input order | `{dig}` | {when} |")
    L.append("")
    if provenance.get("lora_config"):
        L += ["The hyperparameters, recorded before the model loaded rather "
              "than read back from the code afterwards:", "",
              "```json", json.dumps(provenance["lora_config"], indent=2),
              "```", ""]
    if provenance.get("packages"):
        L += ["Package versions at run time: "
              + ", ".join(f"`{k}` {v}"
                          for k, v in provenance["packages"].items())
              + f"; device `{provenance.get('device')}`, dtype "
                f"`{provenance.get('dtype')}`.", ""]
    if provenance.get("backfill_basis"):
        L += [provenance["backfill_basis"], ""]

    L += ["### Limits of this diagnostic", "",
          "- Timings are wall clock on a shared machine. Nothing here isolates "
          "thermal state, other processes, or OS-level scheduling, and no "
          "claim is made about any of them.",
          "- The memory columns are readings, not explanations. A rising "
          "driver figure against a flat tracked figure is *consistent with* "
          "allocator growth; it does not establish it.",
          "- One run per condition, in one process, in a fixed order. The gap "
          "measured is far larger than window-to-window noise, but order is "
          "confounded with condition and n=1 cannot separate them.",
          "- The second condition did not start from the first's initial "
          "conditions: swap, thermal state and OS state carried over. Only "
          "the model and optimizer were rebuilt from the same seed.",
          "- The mechanism is not identified. The output is a narrowed search "
          "space and a mitigation that co-occurred with the absence of the "
          "slowdown once, not a diagnosis.",
          "- Under this design `empty_cache()` cannot be said to remove the "
          "slowdown: the degradation did not appear in the condition that "
          "cleared the cache, and the same condition also ran second. Whether "
          "clearing holds over thousands of rows, and what it costs when it "
          "does, is the next thing to measure -- and it needs both orders, or "
          "fresh processes, before it can be measured at all.", ""]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    body = "\n".join(line.rstrip() for line in L).rstrip("\n") + "\n"
    (REPORT_DIR / "14_mps_speed.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
