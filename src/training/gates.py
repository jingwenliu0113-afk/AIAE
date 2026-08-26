"""Three gates the node must pass, and a ledger that survives being interrupted.

The Taichung machine has never trained anything. Before either frozen
hypothesis is allowed to run on it, three gates have to pass, each answering
something the one before it cannot:

``gate_8``
    Does the machinery exist at all? A real load, forward, backward, optimizer
    step, save, and a **cold** load of what was saved. Eight rows because
    ``grad_accum`` is eight: it is the smallest run in which the optimizer is
    exercised even once, and a "successful" run that never stepped the
    optimizer is the exact failure this gate is for.

``gate_100``
    What does it cost and does it stay sane? Speed, peak VRAM, loss and
    stability.

``gate_500``
    Does it survive interruption? Checkpoint part-way, stop on purpose,
    resume, and then prove that no row was skipped, no row was trained on
    twice, and the weights it continued from are the weights it stopped on.

Two design decisions are worth stating plainly, because both could reasonably
have gone the other way.

**No speed threshold.** ``gate_100`` records seconds per row and refuses to
run without a reading, but it does not judge the reading against a number.
There is no prior measurement on this GPU, and this project's own rule --
written into ``src/training/preflight.py`` -- is that a threshold invented
before the calibration exists is a rationalisation, not a gate. Producing the
first honest measurement *is* what gate 100 is for. Peak VRAM is thresholded,
because 16 GB is a physical fact rather than an opinion, and divergence is
thresholded, because a loss ten times worse than it started is not a slow
learner.

**A checkpoint is the weights, the generator and the optimizer, never a
subset.** ``optimizer.state_dict()`` holds Adam's moments and its step counts
and not a single model parameter. Restoring only that puts the moments beside
weights that were rebuilt from scratch, and the run continues from a point it
was never at -- while the ledger stays contiguous, the order matches, the
provenance is unchanged and every invariant below reports green, because the
ledger records which rows were measured and never what they were measured
against. The generator is the same story one step further out: dropout draws
on every forward pass, so a resume holding the right weights and the right
moments still restarts that stream at the seed and measures different rows.
So every checkpoint carries the trainable tensors, the generator state, their
digests, and the value the live model digested to when it was written; and the
resume re-computes that value after loading, because a load that silently did
nothing leaves a file that still matches its own hash.

**The ledger is append-only and self-describing.** Interruption is where a
training loop tells its most convincing lie: the run finishes, the loss looks
fine, and the optimizer saw row 250 twice and row 251 never. So every entry
records which attempt wrote it and which checkpoint that attempt resumed from,
and the invariants are checked against that rather than against a line count.
Rows measured after the last checkpoint are **re-executed** on resume, because
their optimizer effect was never saved -- and the discarded attempt stays in
the file, visible, rather than being quietly rewritten away.

Nothing in this module imports torch. The real run supplies a device through
:class:`GateDeps`; :class:`FakeGateDeps` supplies a deterministic one, and both
go through the same code below -- a runner that is only ever exercised through
a stand-in is a runner nobody has tested.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.training import pack
from src.training.gpu_node import REQUIRED_DEVICE
from src.training.longrun import ChildDeps, canonical_json, digest_obj, finite
from src.training.lora import LoraConfig_
from src.training.session import (copy_once, now_iso, sha256_file,
                                  write_once_json)

PLAN_NAME = "plan.json"
LEDGER_NAME = "ledger.jsonl"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_STATE = "state.json"
OPTIMIZER_NAME = "optimizer.pt"

#: The trainable half of the model, saved beside the optimizer state.
#:
#: ``optimizer.state_dict()`` contains Adam's moments and its step counts and
#: not one model parameter. A checkpoint holding only that restores the
#: moments onto weights that were rebuilt from scratch -- the run continues,
#: every ledger invariant stays green, and the weights are somewhere the run
#: never was. So the trainable tensors are checkpointed too.
#:
#: Only the trainable ones. The base is 1B frozen parameters that are
#: identical in every checkpoint and are already pinned by the dependency
#: digest; copying them into each one would cost gigabytes to store bytes
#: nothing can change. ``safetensors`` because it stores tensors and has no
#: mechanism for storing anything else, so a checkpoint cannot become a way
#: to execute code on the node.
MODEL_STATE_NAME = "model_state.safetensors"

#: The generator, saved beside them both.
#:
#: ``lora_dropout`` is 0.05 and the model is in train mode, so every forward
#: pass draws a mask. A resume that restores the weights and Adam's moments
#: into a fresh process still begins its dropout stream at the seed rather
#: than where the rows already measured left it -- so it measures different
#: losses from its first row onward and ends on different weights, while the
#: weights it started from and the moments it started with are both exactly
#: right. The position of a run is three things, and two of them is as
#: unresumable as one.
RNG_STATE_NAME = "rng_state.pt"

#: The card's usable capacity, from the node spec. A run that exceeds it does
#: not slow down, it dies -- which is why this is the one reading gate 100
#: judges rather than merely records.
MAX_PEAK_VRAM_GB = 15.0

#: A loss this many times worse than it started is a diverged run, not a slow
#: one. Declared here, before any number from this machine exists, precisely
#: so it cannot be chosen after seeing one.
DIVERGENCE_FACTOR = 10.0

#: Deliberately ``None``. See the module docstring: there is no calibration
#: for this device, and gate 100 exists to produce the first one.
SPEED_THRESHOLD_SECONDS_PER_ROW = None
SPEED_THRESHOLD_REASON = (
    "no speed has ever been calibrated on this GPU, and a threshold chosen "
    "before the calibration exists is a rationalisation rather than a gate. "
    "Gate 100 records seconds per row and refuses to run without a reading; "
    "judging that reading against a number is what the calibration it "
    "produces is for.")

WINDOW = 20


@dataclass(frozen=True)
class Gate:
    name: str
    rows: int
    checkpoint_every: int
    proves: tuple[str, ...]


_ACCUM = LoraConfig_().grad_accum

GATES: dict[str, Gate] = {
    "gate_8": Gate(
        name="gate_8", rows=8, checkpoint_every=8,
        proves=("a real model load",
                "a real forward pass on every row",
                "a real backward pass on every row",
                "at least one real optimizer step",
                "a real save of the adapter",
                "a real cold load of what was saved, matching its digest")),
    "gate_100": Gate(
        name="gate_100", rows=100, checkpoint_every=8 * _ACCUM,
        proves=("seconds per row, recorded",
                "peak VRAM, against the card's capacity",
                "a finite loss on every row",
                "stability: the loss did not diverge")),
    "gate_500": Gate(
        name="gate_500", rows=500, checkpoint_every=8 * _ACCUM,
        proves=("a checkpoint taken part-way through",
                "a deliberate stop",
                "a resume from that checkpoint",
                "the trainable weights restored and re-digested",
                "the generator restored, so the stream continues",
                "optimizer state, progress and input order restored",
                "no row skipped and no row trained on twice")),
}


class GateRefused(RuntimeError):
    """The gate stopped rather than running something it could not vouch for."""


class DeliberateStop(RuntimeError):
    """The stop gate 500 is built around. Not a failure: the point."""

    def __init__(self, position: int):
        super().__init__(f"stopped on purpose after row {position}")
        self.position = position


# ---------------------------------------------------------------------------
# The plan: what this run is, frozen before the first row.
# ---------------------------------------------------------------------------

def order_digest(order) -> str:
    """The input order, as one value. Reused for the plan and the checkpoint."""
    return digest_obj([int(i) for i in order])


def write_plan(run_dir, *, gate: str, order, pack_digest: str, config: dict,
               provenance: dict, expected_dependency_digest: str,
               allocator_config: str, determinism: dict) -> dict:
    """Freeze the run. Write-once, so a resume cannot be re-planned.

    ``expected_dependency_digest`` has no default because it is provenance,
    not a preference. Re-checking the dependencies on every resume is not the
    same as freezing them: a resume that swaps the cache *and* carries the new
    digest agrees with itself perfectly, and would splice a run that began on
    one tokenizer onto one that continued on another. What the run was planned
    against has to be written down once, at the start, where nothing later can
    supply it.
    """
    body = {
        "gate": gate,
        "rows": len(list(order)),
        "order": [int(i) for i in order],
        "order_digest": order_digest(order),
        "pack_digest": pack_digest,
        "dependency_digest": expected_dependency_digest,
        # Runtime provenance. Both are read before this process could have
        # influenced them, and both change the numbers: the same work
        # reserved 15.477 GB under the native segment policy and 7.635 GB
        # under expandable segments, and both report backend "native".
        "allocator_config": allocator_config,
        "determinism": determinism,
        "config": config,
        "provenance": provenance,
        "created_at": now_iso(),
    }
    write_once_json(Path(run_dir) / PLAN_NAME, body)
    return body


def read_plan(run_dir) -> dict | None:
    path = Path(run_dir) / PLAN_NAME
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return body if isinstance(body, dict) else None


# ---------------------------------------------------------------------------
# The ledger: append-only, self-describing, and checked rather than counted.
# ---------------------------------------------------------------------------

LEDGER_FIELDS = ("attempt", "position", "index", "resumed_from", "loss",
                 "sample_id", "tokens", "supervised_tokens", "seconds")


def append_ledger(run_dir, entry: dict) -> None:
    path = Path(run_dir) / LEDGER_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(canonical_json(entry) + "\n")
        fh.flush()


def read_ledger(run_dir) -> list[dict]:
    path = Path(run_dir) / LEDGER_NAME
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                out.append({"__unreadable__": line[:80]})
    return out


def ledger_digest(entries) -> str:
    return digest_obj([{k: e.get(k) for k in LEDGER_FIELDS} for e in entries])


def effective_ledger(entries) -> dict[int, dict]:
    """The latest attempt at each position -- what actually stands.

    The raw file keeps every attempt, including work a resume discarded. This
    is the view the "no gaps, no duplicates" invariant is about.
    """
    out: dict[int, dict] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        pos, attempt = e.get("position"), e.get("attempt")
        if not _is_int(pos) or not _is_int(attempt):
            continue
        prev = out.get(pos)
        if prev is None or attempt >= prev.get("attempt", 0):
            out[pos] = e
    return out


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def ledger_problems(entries, *, order, declared_rows: int | None = None,
                    checkpoint_positions=()) -> list[str]:
    """Everything wrong with this ledger, in sentences.

    Deliberately not a boolean. "The run is invalid" is not actionable; "row
    251 was measured by attempt 1 and again by attempt 2, which resumed from
    checkpoint 256" is.
    """
    order = list(order or [])
    checkpoints = set(int(p) for p in checkpoint_positions)
    problems: list[str] = []

    clean = []
    for i, e in enumerate(entries or []):
        if not isinstance(e, dict):
            problems.append(f"ledger line {i + 1} is not an object")
            continue
        missing = [k for k in ("attempt", "position", "index", "resumed_from")
                   if not _is_int(e.get(k))]
        if missing:
            problems.append(
                f"ledger line {i + 1} has no usable {missing}; a row that "
                "cannot be placed cannot be checked, and skipping it would "
                "make the gap invisible")
            continue
        clean.append(e)

    if declared_rows is not None and not clean:
        problems.append(
            f"the ledger records no usable row, but {declared_rows} were "
            "declared")
        return problems

    by_attempt: dict[int, list[dict]] = {}
    for e in clean:
        by_attempt.setdefault(e["attempt"], []).append(e)

    attempts = sorted(by_attempt)
    if attempts and attempts != list(range(1, len(attempts) + 1)):
        problems.append(
            f"attempts are {attempts}; they must run 1..n with none missing, "
            "or a whole attempt has disappeared from the record")

    previous_max = None
    for attempt in attempts:
        rows = sorted(by_attempt[attempt], key=lambda e: e["position"])
        resumes = {e["resumed_from"] for e in rows}
        if len(resumes) > 1:
            problems.append(
                f"attempt {attempt} records more than one resume point "
                f"{sorted(resumes)}; an attempt starts once")
            continue
        resumed_from = rows[0]["resumed_from"]
        if resumed_from and resumed_from not in checkpoints and checkpoints:
            problems.append(
                f"attempt {attempt} resumed from {resumed_from}, which is not "
                f"a checkpoint {sorted(checkpoints)}. Resuming from a position "
                "no checkpoint saved means the optimizer state does not "
                "correspond to the rows the ledger claims are done.")
        positions = [e["position"] for e in rows]
        if len(set(positions)) != len(positions):
            repeated = sorted({p for p in positions if positions.count(p) > 1})
            problems.append(
                f"attempt {attempt} measured row(s) {repeated} twice within "
                "one attempt")
        expected = list(range(resumed_from + 1, resumed_from + 1 + len(set(positions))))
        if sorted(set(positions)) != expected:
            problems.append(
                f"attempt {attempt} resumed from {resumed_from} and covers "
                f"{_span(sorted(set(positions)))}; it must run contiguously "
                f"from {resumed_from + 1}")
        for e in rows:
            if e["position"] <= resumed_from:
                problems.append(
                    f"attempt {attempt} measured row {e['position']}, which "
                    f"resumed_from {resumed_from} says was already saved; "
                    "re-training a row the optimizer state already includes "
                    "applies its gradient twice")
        if previous_max is not None and resumed_from > previous_max:
            problems.append(
                f"attempt {attempt} resumed from {resumed_from} but the "
                f"previous attempt only reached {previous_max}; rows "
                f"{previous_max + 1}..{resumed_from} were never measured by "
                "anything")
        previous_max = max(previous_max or 0, max(positions))

    seen_sample: dict[int, str] = {}
    for e in clean:
        pos, index = e["position"], e["index"]
        if order:
            if pos - 1 >= len(order):
                problems.append(
                    f"row {pos} is beyond the {len(order)} rows the plan's "
                    "order declares")
            elif order[pos - 1] != index:
                problems.append(
                    f"row {pos} used pool index {index}, but the plan's order "
                    f"puts {order[pos - 1]} there; the input order was not "
                    "preserved across the resume")
        sample = e.get("sample_id")
        if not sample:
            problems.append(f"row {pos} records no sample_id")
        elif index in seen_sample and seen_sample[index] != sample:
            problems.append(
                f"pool index {index} was sample {seen_sample[index]!r} and is "
                f"now {sample!r}; the same index became a different row")
        elif index not in seen_sample:
            seen_sample[index] = sample
        loss = e.get("loss")
        if not finite(loss):
            problems.append(
                f"row {pos} recorded a loss of {loss!r}, which is not finite")

    effective = effective_ledger(clean)
    if effective:
        positions = sorted(effective)
        contiguous = list(range(1, max(positions) + 1))
        missing = sorted(set(contiguous) - set(positions))
        if missing:
            problems.append(
                f"row(s) {_span(missing)} are missing from the effective "
                "ledger; nothing measured them")
    if declared_rows is not None:
        got = sorted(effective)
        want = list(range(1, declared_rows + 1))
        if got != want:
            short = sorted(set(want) - set(got))
            extra = sorted(set(got) - set(want))
            if short:
                problems.append(
                    f"{len(short)} of {declared_rows} rows were never "
                    f"measured: {_span(short)}")
            if extra:
                problems.append(
                    f"rows {_span(extra)} were measured but are outside the "
                    f"declared {declared_rows}")
    return problems


def _span(values) -> str:
    values = list(values)
    if len(values) <= 6:
        return ", ".join(str(v) for v in values)
    return f"{values[0]}..{values[-1]} ({len(values)} rows)"


def duplicate_positions(entries) -> list[int]:
    """Positions measured more than once *within* a single attempt."""
    seen: dict[tuple[int, int], int] = {}
    for e in entries or []:
        if isinstance(e, dict) and _is_int(e.get("attempt")) \
                and _is_int(e.get("position")):
            key = (e["attempt"], e["position"])
            seen[key] = seen.get(key, 0) + 1
    return sorted({pos for (_, pos), n in seen.items() if n > 1})


def missing_positions(entries, declared_rows: int) -> list[int]:
    effective = effective_ledger(entries)
    return sorted(set(range(1, declared_rows + 1)) - set(effective))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def checkpoint_dir(run_dir, position: int) -> Path:
    return Path(run_dir) / CHECKPOINT_DIR / f"{int(position):06d}"


def write_checkpoint(run_dir, *, position: int, attempt: int, plan: dict,
                     optimizer_sha256: str | None, model_state: dict | None,
                     rng_state: dict | None, trainable_digest: str | None,
                     ledger_entries, provenance: dict) -> dict:
    """Publish one checkpoint, once, and never rewrite it.

    What it has to carry is what a resume has to be able to check: where the
    run had got to, what the **weights** and the optimizer state hash to,
    what the trainable tensors digest to at this row, what the input order
    was, which pack produced it, which dependency bytes it ran against, and
    enough provenance that a run continued days later can be shown to be the
    same run rather than assumed to be.

    ``model_state`` and ``trainable_digest`` have no defaults. A checkpoint
    that could be written without them is one that will be, and the resume it
    produces looks exactly like a correct one from the outside: contiguous
    positions, no duplicates, the planned order, unchanged provenance, and
    weights the run never had.

    The dependency digest is taken **from the plan**, never re-read here. A
    checkpoint recording what the cache looked like at the moment it was
    written would agree with a cache that had already been swapped, which is
    the opposite of what a checkpoint is for.
    """
    body = {
        "position": int(position),
        "attempt": int(attempt),
        "gate": plan.get("gate"),
        "order_digest": plan.get("order_digest"),
        "pack_digest": plan.get("pack_digest"),
        "dependency_digest": plan.get("dependency_digest"),
        "allocator_config": plan.get("allocator_config"),
        "determinism": plan.get("determinism"),
        "optimizer_sha256": optimizer_sha256,
        # The file, and what the live model digested to when it was written.
        # The file's own digest says the bytes did not rot; only this says
        # the load put them back where they came from.
        "model_state": model_state,
        "rng_state": rng_state,
        "trainable_digest": trainable_digest,
        "ledger_digest": ledger_digest(ledger_entries),
        "rows_in_ledger": len(list(ledger_entries)),
        "provenance": provenance,
        "written_at": now_iso(),
    }
    write_once_json(checkpoint_dir(run_dir, position) / CHECKPOINT_STATE, body)
    return body


def read_checkpoints(run_dir) -> list[dict]:
    base = Path(run_dir) / CHECKPOINT_DIR
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir()):
        state = d / CHECKPOINT_STATE
        if not state.is_file():
            continue
        try:
            body = json.loads(state.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(body, dict):
            body = dict(body)
            body["dir"] = str(d)
            out.append(body)
    # By position, not by name and not by mtime: a directory listing sorts
    # lexically and a resumed run's newest checkpoint is not the newest file.
    out.sort(key=lambda b: b.get("position") if _is_int(b.get("position")) else -1)
    return out


def latest_checkpoint(run_dir) -> dict | None:
    found = read_checkpoints(run_dir)
    return found[-1] if found else None


# ---------------------------------------------------------------------------
# Resuming: verify everything first, then continue
# ---------------------------------------------------------------------------

def dependency_digest_problems(expected, *, checker=None) -> list[str]:
    """Does this machine's cache hold the bytes the build machine had?

    The node's preflight asks the same question and phrases the answer as a
    gate verdict; a run asks it again here, because ``run_gate`` and
    ``resume_problems`` are reachable without a preflight and a guarantee that
    only holds when somebody remembers to run something is not a guarantee.
    Both go through :func:`~src.training.gpu_node.preflight`'s reasoning, and
    through the one digest in
    :func:`~src.training.longrun.dependency_digest`.
    """
    from src.training.gpu_node import preflight_dependency_problems

    return preflight_dependency_problems(expected, checker=checker)


def runtime_problems(allocator_config, determinism) -> list[str]:
    """Is this runtime provenance usable and strict? Shape only, no comparison."""
    from src.training.gpu_node import allocator_config_problems
    from src.training.gpu_node import determinism_problems as _det

    return list(allocator_config_problems(allocator_config)) + list(
        _det(determinism))


def _frozen_runtime_problems(plan: dict, ckpt: dict, *, allocator_config,
                             determinism) -> list[str]:
    """The plan's runtime provenance, against this invocation and the checkpoint."""
    problems = []
    frozen_alloc = plan.get("allocator_config", _MISSING)
    if frozen_alloc is _MISSING:
        problems.append(
            "the plan records no allocator_config, so nothing says which "
            "allocator produced the rows already measured. A run planned "
            "before that was written down cannot be continued.")
    else:
        problems += [f"the plan's allocator config is unusable: {p}"
                     for p in _alloc_problems(frozen_alloc)]
        if allocator_config != frozen_alloc:
            problems.append(
                f"this invocation inherited allocator config "
                f"{allocator_config!r} but the run was planned under "
                f"{frozen_alloc!r}. Continuing would join rows measured under "
                "two different allocators into one run.")
        if ckpt.get("allocator_config", _MISSING) is _MISSING:
            problems.append("the checkpoint records no allocator_config")
        elif ckpt.get("allocator_config") != frozen_alloc:
            problems.append(
                "the checkpoint's allocator_config does not match the plan's")

    frozen_det = plan.get("determinism", _MISSING)
    if frozen_det is _MISSING:
        problems.append(
            "the plan records no determinism settings, so the seed, the TF32 "
            "state and whether deterministic kernels were in force are all "
            "unknown for the rows already measured")
    else:
        problems += [f"the plan's determinism record is unusable: {p}"
                     for p in _det_problems(frozen_det)]
        if determinism != frozen_det:
            differing = sorted(
                k for k in set(frozen_det) | set(determinism or {})
                if (frozen_det or {}).get(k) != (determinism or {}).get(k))
            problems.append(
                f"the determinism settings differ from the plan's in "
                f"{differing}; the rows already measured were produced under "
                "the other ones")
        if ckpt.get("determinism", _MISSING) is _MISSING:
            problems.append("the checkpoint records no determinism settings")
        elif ckpt.get("determinism") != frozen_det:
            problems.append(
                "the checkpoint's determinism settings do not match the plan's")
    return problems


def _alloc_problems(value):
    from src.training.gpu_node import allocator_config_problems

    return allocator_config_problems(value)


def _det_problems(value):
    from src.training.gpu_node import determinism_problems

    return determinism_problems(value)


def _frozen_dependency_problems(plan: dict, ckpt: dict, *, expected,
                                checker=None) -> list[str]:
    """The plan's dependency digest, against everything that must equal it.

    Four comparisons, and the frozen value is one side of every one of them.
    That is the point: ``dependency_digest_problems`` already establishes that
    what the operator carried matches what this machine can see, and two
    values that agree with each other say nothing about the run they are
    joining. Only the plan knows what attempt 1 actually used.
    """
    frozen = plan.get("dependency_digest", _MISSING)
    if frozen is _MISSING:
        return ["the plan records no dependency_digest, so nothing says which "
                "tokenizer, base model and adapter bytes this run began "
                "against. A run planned before that was written down cannot "
                "be vouched for now; start it again rather than continue it."]
    problems = pack.expected_digest_problems(
        frozen, what="dependency digest frozen in the plan")
    if problems:
        return problems

    if expected != frozen:
        problems.append(
            f"the dependency digest carried for this resume "
            f"({str(expected)[:16]}...) is not the {frozen[:16]}... this run "
            "was planned against. Whichever is right, continuing would join "
            "two different dependency sets into one run.")

    recomputed = _recomputed_dependency_digest(checker)
    if recomputed != frozen:
        problems.append(
            f"this machine's dependencies now digest to "
            f"{str(recomputed)[:16]}..., not the {frozen[:16]}... this run "
            "was planned against. The cache changed between the stop and the "
            "resume; the rows already measured cannot be continued on top of "
            "different bytes.")

    if ckpt.get("dependency_digest", _MISSING) is _MISSING:
        problems.append(
            "the checkpoint records no dependency_digest, so it cannot be "
            "shown to belong to this run's dependency set")
    elif ckpt.get("dependency_digest") != frozen:
        problems.append(
            f"the checkpoint's dependency_digest "
            f"({str(ckpt.get('dependency_digest'))[:16]}...) does not match "
            f"the plan's {frozen[:16]}...")
    return problems


def _recomputed_dependency_digest(checker=None):
    """What this machine's cache digests to now, or ``None`` if unreadable."""
    from src.training.gpu_node import recomputed_dependency_digest

    return recomputed_dependency_digest(checker=checker)


def _digest_field_problems(value, *, what: str) -> list[str]:
    """Is a digest the checkpoint recorded about itself usable? Format only.

    :func:`~src.training.pack.expected_digest_problems` owns the one rule
    about what a digest looks like, and it is called here rather than
    restated. Its wording for a missing value is not: that text is about a
    digest carried separately from the build machine, and these are the run's
    own record of where it had got to.
    """
    if not isinstance(value, str):
        return [f"the checkpoint records no {what} ({value!r}), so nothing "
                "says what a resume should end up holding"]
    return [f"the checkpoint's {what} is unusable: {problem}"
            for problem in pack.expected_digest_problems(value, what=what)]


def _blob_problems(ckpt: dict, *, field: str, expected_name: str, what: str,
                   consequence: str) -> list[str]:
    """One checkpointed file: recorded, named honestly, present, and intact.

    The name is read **off disk**, so it is input rather than a constant. It
    is compared against the only name this code writes rather than joined onto
    the checkpoint directory and trusted -- a state file naming
    ``../../something`` would otherwise make a resume read whatever it liked.
    """
    state = ckpt.get(field)
    if not isinstance(state, dict):
        return [f"the checkpoint records no {what}. {consequence}"]

    name = state.get("name")
    if name != expected_name:
        return [f"the checkpoint's {what} is named {str(name)[:40]!r}; this "
                f"code writes {expected_name!r} and nothing else, and a path "
                "taken from a file on disk is input, not a constant"]

    problems = _digest_field_problems(state.get("sha256"),
                                      what=f"{what} digest")
    blob = Path(ckpt.get("dir", "")) / name
    if not blob.is_file():
        return problems + [
            f"the checkpoint's {what} ({expected_name}) is missing. "
            f"{consequence}"]

    expected = state.get("sha256")
    if isinstance(expected, str) and sha256_file(blob) != expected:
        problems.append(
            f"the checkpoint's {what} does not match the digest recorded "
            "beside it")
    size = state.get("bytes")
    if _is_int(size) and blob.stat().st_size != size:
        problems.append(
            f"the checkpoint's {what} is not the size recorded beside it")
    return problems


def restorable_state_problems(ckpt: dict) -> list[str]:
    """Can everything this checkpoint claims to hold actually be restored?

    Three things define where a run had got to, and the ledger records none of
    them: the trainable weights, the optimizer state, and the generator. This
    covers the first and the third; the optimizer is checked beside them in
    :func:`resume_problems` because it was already checked there.

    Asked before the optimizer, in the order the resume restores them in. A
    check that runs in a different order from the thing it guards eventually
    guards nothing.
    """
    problems = _digest_field_problems(ckpt.get("trainable_digest"),
                                      what="trainable digest")
    problems += _blob_problems(
        ckpt, field="model_state", expected_name=MODEL_STATE_NAME,
        what="model state",
        consequence=(
            "optimizer.state_dict() holds no model parameter at all, so "
            "resuming would put Adam's moments onto weights rebuilt from "
            "scratch and continue a run that never happened -- while every "
            "ledger invariant stayed green."))
    problems += _blob_problems(
        ckpt, field="rng_state", expected_name=RNG_STATE_NAME,
        what="rng state",
        consequence=(
            "Dropout draws on every forward pass, so a resume without the "
            "generator restarts that stream from the seed: the same weights "
            "measure different rows, and the run ends somewhere it would not "
            "have ended had it never been interrupted."))
    return problems


#: Tells "this writer had no field" apart from "it had one and it was null".
#: ``dict.get`` collapses them, and here they mean different things: the first
#: is a plan written before the digest was frozen, the second is a plan that
#: knew about it and recorded nothing.
_MISSING = object()


def resume_problems(run_dir, *, pack_dir, expected_pack_digest,
                    expected_dependency_digest, allocator_config, determinism,
                    verifier=None, dependency_checker=None) -> list[str]:
    """Everything that must hold before a resumed run measures another row.

    The manifest is checked **file by file** rather than by its digest alone.
    A manifest says the set of files is the set the manifest describes; it does
    not say that the copy on this disk still is, and it cannot say it is the
    set the build machine produced -- every digest inside it is computed from
    it. Between a stop and a resume the tree sat on a machine somebody uses,
    possibly for days, so both questions are asked: does the pack still match
    its own manifest, and does that manifest match the value carried here.
    """
    verifier = verifier or pack.verify
    problems = pack.trusted_digest_problems(pack_dir, expected_pack_digest)
    # Re-checked on every resume, never carried over from the attempt that
    # started the run. The cache lives on a machine somebody uses and a resume
    # can be days later; trusting the earlier reading would trust it from
    # before the gap that made re-taking it worthwhile.
    problems += dependency_digest_problems(
        expected_dependency_digest, checker=dependency_checker)
    problems += runtime_problems(allocator_config, determinism)
    problems += list(verifier(pack_dir))

    plan = read_plan(run_dir)
    if plan is None:
        return problems + [
            f"{PLAN_NAME} is missing or unreadable in {Path(run_dir).name}; "
            "there is nothing that says what this run was supposed to be"]

    manifest, manifest_problems = pack.read_manifest(pack_dir)
    if manifest_problems:
        problems += manifest_problems
    elif manifest.get("pack_digest") != plan.get("pack_digest"):
        problems.append(
            f"pack_digest {str(manifest.get('pack_digest'))[:12]}... is not "
            f"the {str(plan.get('pack_digest'))[:12]}... this run was planned "
            "against; this is a different pack")

    if order_digest(plan.get("order") or []) != plan.get("order_digest"):
        problems.append(
            "the plan's order no longer digests to its recorded "
            "order_digest; the input order has been edited")

    ckpt = latest_checkpoint(run_dir)
    if ckpt is None:
        return problems + [
            "there is no checkpoint to resume from; a run that never "
            "checkpointed can only be started again from the beginning"]

    if ckpt.get("order_digest") != plan.get("order_digest"):
        problems.append(
            "the checkpoint's order_digest does not match the plan's; it was "
            "taken against a different input order")
    if ckpt.get("pack_digest") != plan.get("pack_digest"):
        problems.append(
            "the checkpoint's pack_digest does not match the plan's")
    problems += _frozen_dependency_problems(
        plan, ckpt, expected=expected_dependency_digest,
        checker=dependency_checker)
    problems += _frozen_runtime_problems(
        plan, ckpt, allocator_config=allocator_config,
        determinism=determinism)

    # Weights and generator before optimizer, the order the resume uses.
    problems += restorable_state_problems(ckpt)

    state = Path(ckpt["dir"]) / OPTIMIZER_NAME
    expected = ckpt.get("optimizer_sha256")
    if expected is None:
        problems.append(
            "the checkpoint records no optimizer digest, so resuming would "
            "restore state nothing can vouch for")
    elif not state.is_file():
        problems.append(
            f"the checkpoint's {OPTIMIZER_NAME} is missing; without the "
            "optimizer state a resume restarts Adam's moments from zero, "
            "which is a different optimisation")
    elif sha256_file(state) != expected:
        problems.append(
            f"the checkpoint's {OPTIMIZER_NAME} does not match the digest "
            "recorded beside it")

    entries = read_ledger(run_dir)
    checkpoints = [c["position"] for c in read_checkpoints(run_dir)
                   if _is_int(c.get("position"))]
    problems += ledger_problems(entries, order=plan.get("order") or [],
                                checkpoint_positions=checkpoints)

    upto = [e for e in entries if _is_int(e.get("position"))
            and e["position"] <= ckpt["position"]]
    if ledger_digest(upto) != ckpt.get("ledger_digest"):
        problems.append(
            "the ledger up to the checkpoint does not digest to what the "
            "checkpoint recorded; it has been appended to, truncated or "
            "edited since")

    rows = plan.get("rows")
    if _is_int(rows):
        # Two different ways of being finished, and the ledger is the one that
        # matters. Checkpoints land on a fixed stride, so the last rows of a
        # completed run are usually *past* the final checkpoint -- asking only
        # whether the checkpoint reached the end would let a finished run be
        # resumed and measure its last rows a second time.
        done = sorted(effective_ledger(entries))
        if done and done[-1] >= rows:
            problems.append(
                f"the ledger already covers all {rows} rows; this run is "
                "finished and resuming it would measure rows it has already "
                "measured")
        elif ckpt["position"] >= rows:
            problems.append(
                f"the checkpoint is at row {ckpt['position']} of {rows}; this "
                "run is finished and has nothing to resume")
    return problems


def resume_point(run_dir, *, pack_dir, expected_pack_digest,
                 expected_dependency_digest, allocator_config, determinism,
                 verifier=None, dependency_checker=None) -> dict:
    """Where a resume would start, or refuse and say why."""
    problems = resume_problems(
        run_dir, pack_dir=pack_dir,
        expected_pack_digest=expected_pack_digest,
        expected_dependency_digest=expected_dependency_digest,
        allocator_config=allocator_config, determinism=determinism,
        verifier=verifier, dependency_checker=dependency_checker)
    if problems:
        raise GateRefused("refusing to resume:\n  - " + "\n  - ".join(problems))
    ckpt = latest_checkpoint(run_dir)
    entries = read_ledger(run_dir)
    attempts = [e["attempt"] for e in entries if _is_int(e.get("attempt"))]
    return {"resume_from": ckpt["position"],
            "next_position": ckpt["position"] + 1,
            "attempt": (max(attempts) if attempts else 0) + 1,
            "checkpoint_dir": ckpt["dir"],
            # What the weights must digest to once they are loaded back.
            # Verified above as a file; this is what proves the *load* worked
            # and not merely that the bytes are intact.
            "trainable_digest": ckpt.get("trainable_digest"),
            "model_state": ckpt.get("model_state") or {},
            "provenance": ckpt.get("provenance") or {}}


# ---------------------------------------------------------------------------
# What the runner needs from outside itself
# ---------------------------------------------------------------------------

class GateDeps(ChildDeps):
    """Report 16's child contract, plus what a gate additionally needs.

    ``load(rows=...)`` returns everything :class:`~src.training.longrun.ChildDeps`
    returns -- ``order``, ``step``, ``provenance``, ``sample_ids``,
    ``model_load_seconds``, ``teardown`` -- and four more, because a gate saves,
    reloads and measures where report 16 only measured:

    ``save_adapter(dir)``
        writes the adapter and returns ``{"path", "sha256", "bytes"}``.
    ``save_model_state(path)`` / ``load_model_state(path)``
        the trainable tensors. Separate from the adapter save, which is the
        run's *output*; this is the run's *position*, and it is written at
        every checkpoint rather than once at the end.
    ``save_rng_state(path)`` / ``load_rng_state(path)``
        the generator. ``lora_dropout`` is 0.05 and the model is in train
        mode, so a resume that restarts the stream at the seed measures
        different rows from exactly the right weights.
    ``save_optimizer(path)`` / ``load_optimizer(path)``
        the optimizer state, which holds Adam's moments and not one model
        parameter -- which is why the two pairs above exist.
    ``trainable_digest()``
        one value over every trainable tensor's contents. Taken at each
        checkpoint and re-taken after a restore, so a load that quietly did
        nothing is a refusal rather than a continuation.
    ``peak_memory()``
        ``{"peak_vram_gb": ...}``, or ``None`` where it cannot be read.

    Extending the existing contract rather than declaring a new one is
    deliberate: the two runners then agree about what a step is, and a change
    to that meaning has one place to be made.
    """

    def load(self, *, rows: int) -> dict:
        raise NotImplementedError


class FakeGateDeps(GateDeps):
    """A deterministic stand-in: no torch, no device, no dataset, no network.

    It is in this module rather than in the tests on purpose. The runner is
    only trustworthy if the path the fake drives is the path the real one
    drives, and a fake that lives in a test file drifts from the contract the
    moment the contract changes.
    """

    def __init__(self, *, rows: int, seed: int = 0, cfg=None):
        self.rows = rows
        self.seed = seed
        # The configuration this stand-in speaks for. Only used to write the
        # adapter manifest, which the real loader writes from the config it
        # was built with -- a fake that skipped it could not be used to test
        # the check that the manifest and the weights agree.
        self.cfg = cfg
        self.loaded_optimizer_from: str | None = None
        self.loaded_model_state_from: str | None = None
        self.optimizer_steps = 0
        self.provenance_override: dict | None = None
        # The weights, kept deliberately apart from ``optimizer_steps``.
        #
        # A fake whose trainable digest is a function of the optimizer counter
        # cannot tell a checkpoint that restored only the optimizer apart from
        # one that restored everything -- restoring the counter would restore
        # the digest -- and every test about that distinction would pass
        # against the broken checkpoint it was written to catch. So these move
        # on *every* row, and they accumulate rather than decay: a state that
        # forgets where it came from after fifty rows is a state a wrong
        # resume converges back onto.
        self.weights = {"w": 0.0, "rows_trained": 0}
        self.loaded_rng_state_from: str | None = None
        # The generator, standing in for the dropout mask a real forward pass
        # draws. Advanced once per row and folded into the weights, so a
        # resume that restored the weights and the optimizer and not this
        # still ends somewhere else -- which is the only way a test can show
        # that restoring two of the three is not enough.
        self.rng = {"state": 12345, "draws": 0}

    def _provenance(self) -> dict:
        base = {
            "device": "cuda",
            "dtype": LoraConfig_().dtype,
            "lora_config": LoraConfig_().as_dict(),
            "optimizer": "AdamW(lr=0.0001, betas=(0.9,0.999), wd=0.01)",
            "quantization": LoraConfig_().quantization,
            # Carried because the gate suite requires it of every run, and a
            # stand-in that does not satisfy a contract cannot be used to test
            # it. ``max_rows`` is the one provenance value that legitimately
            # differs between the six runs, so it is the one they are each
            # pinned against individually.
            "measurement_intervals": {"window": WINDOW, "max_rows": self.rows},
        }
        base.update(self.provenance_override or {})
        return base

    def load(self, *, rows: int) -> dict:
        order = [(self.seed * 7919 + i * 31) % 2000 for i in range(rows)]
        sample_ids = [f"row-{i:05d}" for i in order]
        holder = {"loss": 2.0}

        def step(index: int, position: int) -> dict:
            # A loss that falls and then flattens, plus what the weights have
            # become. Deterministic, so a resumed run measuring the same row
            # twice produces the same number and the duplicate is invisible in
            # the values -- which is exactly why the ledger, not the losses, is
            # what the invariant is checked on.
            #
            # It depends on the weights so that continuing from the wrong ones
            # shows up in the losses too, not only in the final digest.
            # One draw per row, exactly where a real forward pass takes one.
            self.rng["state"] = (self.rng["state"] * 1103515245 + 12345) % (
                2 ** 31)
            self.rng["draws"] += 1
            noise = (self.rng["state"] % 1000) * 1e-9
            loss = (0.5 + 1.5 * math.exp(-position / 120.0)
                    + self.weights["w"] + noise)
            holder["loss"] = loss
            # Every row moves the weights, whether or not it crossed an
            # accumulation boundary, and the draw moves them too. Rounded so
            # the value round-trips through the checkpoint exactly.
            self.weights["w"] = round(
                self.weights["w"] + ((index % 7) - 3) * 1e-4 + noise, 12)
            self.weights["rows_trained"] += 1
            if position % _ACCUM == 0:
                self.optimizer_steps += 1
            return {"loss": loss, "tokens": 64 + (index % 17),
                    "supervised_tokens": 32 + (index % 11),
                    "sample_id": sample_ids[position - 1]}

        def save_adapter(dest) -> dict:
            dest = Path(dest)
            dest.mkdir(parents=True, exist_ok=True)
            blob = dest / "adapter_model.json"
            blob.write_text(canonical_json(
                {"rows": rows, "steps": self.optimizer_steps,
                 "weights": self.weights}), encoding="utf-8")
            # Written beside the weights, as production does. What a reader
            # needs from it is which LoRA shape these weights are, so that is
            # what a stand-in has to carry too.
            cfg = self.cfg if self.cfg is not None else LoraConfig_()
            (dest / "brickagain_manifest.json").write_text(
                canonical_json({"lora": {"r": cfg.rank, "alpha": cfg.alpha,
                                         "target_modules":
                                             list(cfg.target_modules)}}),
                encoding="utf-8")
            return {"path": blob.name, "sha256": sha256_file(blob),
                    "bytes": blob.stat().st_size}

        def save_model_state(path) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(canonical_json(self.weights), encoding="utf-8")

        def load_model_state(path) -> None:
            body = json.loads(Path(path).read_text(encoding="utf-8"))
            self.weights = {"w": float(body["w"]),
                            "rows_trained": int(body["rows_trained"])}
            self.loaded_model_state_from = str(path)

        def save_rng_state(path) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(canonical_json(self.rng), encoding="utf-8")

        def load_rng_state(path) -> None:
            body = json.loads(Path(path).read_text(encoding="utf-8"))
            self.rng = {"state": int(body["state"]),
                        "draws": int(body["draws"])}
            self.loaded_rng_state_from = str(path)

        def save_optimizer(path) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(canonical_json(
                {"steps": self.optimizer_steps, "moment": holder["loss"]}),
                encoding="utf-8")

        def load_optimizer(path) -> None:
            # Deliberately touches nothing but the optimizer. This is the
            # whole distinction the gate 500 resume has to make.
            body = json.loads(Path(path).read_text(encoding="utf-8"))
            self.optimizer_steps = int(body.get("steps", 0))
            self.loaded_optimizer_from = str(path)

        def cold_load(saved: dict, adapter_dir) -> dict:
            blob = Path(adapter_dir) / saved["path"]
            got = sha256_file(blob) if blob.is_file() else None
            return {"loaded": blob.is_file(), "sha256": got,
                    "matches_saved": got == saved.get("sha256")}

        return {
            "order": order,
            "step": step,
            "provenance": self._provenance(),
            # Carried because the production loader carries it and the final
            # run refuses without it. A stand-in that omitted the reading
            # could not be used to test the check that demands it.
            "data_source": "pool",
            "truncated_rows": 0,
            "max_total_tokens": 64,
            "sample_ids": sample_ids,
            "model_load_seconds": 1.25,
            "teardown": lambda: 0.0,
            "save_adapter": save_adapter,
            "save_model_state": save_model_state,
            "load_model_state": load_model_state,
            "save_rng_state": save_rng_state,
            "load_rng_state": load_rng_state,
            "save_optimizer": save_optimizer,
            "load_optimizer": load_optimizer,
            "cold_load": cold_load,
            # Over the weights and nothing else -- not the row count, not the
            # optimizer counter. See ``__init__``.
            "trainable_digest": lambda: hashlib.sha256(
                canonical_json(self.weights).encode()).hexdigest(),
            "peak_memory": lambda: {"peak_vram_gb": 9.1,
                                    "peak_allocated_gb": 7.4,
                                    "allocator_backend": "native",
                                    "inactive_split_bytes_current": 0.5,
                                    "inactive_split_bytes_peak": 1.2,
                                    "num_alloc_retries": 0,
                                    "num_ooms": 0},
        }


def trainable_named(model) -> list:
    """The tensors a LoRA run owns, in one fixed order.

    Shared by the digest, the save and the restore so the three cannot drift.
    If the save stored one set and the digest covered another, a checkpoint
    would verify against itself perfectly and restore something else -- which
    is the failure this whole mechanism exists to prevent, reintroduced one
    level down.

    Sorted by name because ``named_parameters`` order is a property of how the
    module tree was built, and a digest that depends on that is a digest of
    the construction rather than of the weights.
    """
    return [(name, param)
            for name, param in sorted(model.named_parameters(),
                                      key=lambda kv: kv[0])
            if param.requires_grad]


def _safetensors_io():
    """Save and load a plain name-to-tensor mapping, with no pickle involved.

    Imported here rather than at module scope: this module is imported by
    tests that never touch a tensor, and by the pack audit on a machine that
    is not the node.
    """
    def save(mapping, path):
        from safetensors.torch import save_file

        save_file(mapping, str(path))

    def load(path):
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")

    return save, load


class ProductionGateDeps(GateDeps):
    """The real one, assembled from report 16's loader rather than beside it.

    The expensive, dangerous half -- tokenizer, frozen pool, base weights, the
    published adapter, the merge, our adapter, AdamW, the per-row step -- is
    :class:`~src.training.longrun.ProductionChildDeps`, unchanged and already
    tested. Rewriting it here would give the project two definitions of "which
    weights was this", and the day they disagreed nothing would say so.

    What this class adds is the four things a gate does and a measurement does
    not: save the adapter, save and restore the optimizer state, cold-load
    what was saved, and read peak VRAM. ``base`` and ``torch_mod`` are
    injectable so the wrapper can be driven without a GPU; production supplies
    neither and gets the real ones.

    **This path has never executed.** Its first execution is gate 8 on the
    node, which is what gate 8 is for.
    """

    def __init__(self, *, device: str = "cuda", base=None, torch_mod=None,
                 tensor_io=None, cfg=None, source: str = "pool"):
        if device != REQUIRED_DEVICE:
            raise GateRefused(
                f"ProductionGateDeps was asked for device {device!r}. The node "
                f"runs {REQUIRED_DEVICE!r} and there is no fallback: a run on "
                "another device is a different experiment, and the numbers "
                "would be compared against ones it is not comparable with.")
        self.device = device
        self._base = base
        self._torch = torch_mod
        self._tensor_io = tensor_io
        # Passed straight through to the base loader. See its ``__init__``:
        # the only value that ever arrives here is a frozen hypothesis
        # configuration, from ``hypotheses.config_for`` and nowhere else.
        self._cfg = cfg
        # Likewise: ``pool`` for every measurement so far, ``full_train`` only
        # for the final run.
        self._source = source

    def load(self, *, rows: int) -> dict:
        base = self._base
        if base is None:
            from src.training.longrun import ProductionChildDeps

            base = ProductionChildDeps(device=self.device, cfg=self._cfg,
                                       source=self._source)
        loaded = dict(base.load(rows=rows))

        torch_mod = self._torch
        if torch_mod is None:
            import torch as torch_mod  # noqa: PLC0415

        holder = loaded.get("holder")
        if not isinstance(holder, dict) or "model" not in holder:
            raise GateRefused(
                "the base loader did not expose the model it built, so the "
                "adapter cannot be saved and the optimizer state cannot be "
                "written. A gate that cannot save has nothing to cold-load.")

        def save_adapter(dest) -> dict:
            from src.training.lora import write_manifest

            dest = Path(dest)
            dest.mkdir(parents=True, exist_ok=True)
            holder["model"].save_pretrained(str(dest))
            # Written beside the weights, not inferred later: a directory of
            # LoRA tensors does not say it was fitted on top of a *merged*
            # BrickGPT, and loading it the obvious way produces a model that
            # runs and is wrong.
            #
            # From the configuration this loader was *built* with, never from
            # ``LoraConfig_()``. The manifest is what ``load_finetuned``
            # validates the adapter against, so writing the default beside an
            # H2 adapter would produce a directory whose weights are rank 32
            # and whose manifest says rank 16 -- and the cold load would then
            # be checking the wrong shape, or passing and leaving a
            # permanently mislabelled checkpoint behind.
            write_manifest(dest, loaded.get("provenance") or {}, config)
            blob = dest / "adapter_model.safetensors"
            return {"path": blob.name,
                    "sha256": sha256_file(blob) if blob.is_file() else None,
                    "bytes": blob.stat().st_size if blob.is_file() else None}

        save_tensors, load_tensors = self._tensor_io or _safetensors_io()
        # The one configuration this loader speaks for. Resolved once, so the
        # adapter manifest, the model that was built and the plan cannot
        # disagree about which run this is.
        config = self._cfg if self._cfg is not None else LoraConfig_()

        def save_model_state(path) -> None:
            """The trainable tensors, on CPU, in a format that carries no code.

            Only the trainable ones: the base is 1B frozen parameters that are
            byte-identical in every checkpoint and are already pinned by the
            dependency digest, so writing them into each one would cost
            gigabytes to store what nothing can change. On CPU so the file
            does not encode which device wrote it, and cloned so no tensor in
            it is a view of another.
            """
            model = holder.get("model")
            if model is None:
                raise GateRefused(
                    "there is no model to checkpoint. A checkpoint holding "
                    "only the optimizer restores Adam's moments onto weights "
                    "rebuilt from scratch, which continues a different run.")
            named = trainable_named(model)
            if not named:
                raise GateRefused(
                    "no trainable tensor was found to checkpoint; a LoRA run "
                    "with nothing requiring grad has not trained anything")
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_tensors({name: param.detach().to("cpu").contiguous().clone()
                          for name, param in named}, path)

        def load_model_state(path) -> None:
            """Put them back, in place, or refuse.

            Copied into the existing parameters rather than swapped in as new
            ones: the optimizer restored immediately afterwards refers to the
            parameter objects this model was built with, and replacing them
            would leave Adam's moments attached to tensors nothing steps.

            The set has to match exactly. Restoring a subset would leave the
            rest at their freshly initialised values -- a partial resume that
            reports success, which is the failure mode one layer down from the
            one this file is about.
            """
            model = holder.get("model")
            if model is None:
                raise GateRefused("there is no model to restore into")
            state = load_tensors(path)
            named = dict(trainable_named(model))
            missing = sorted(set(named) - set(state))
            unexpected = sorted(set(state) - set(named))
            if missing or unexpected:
                raise GateRefused(
                    f"the checkpoint holds {len(state)} trainable tensors and "
                    f"this model has {len(named)}: {len(missing)} are missing "
                    f"from the checkpoint and {len(unexpected)} are not in "
                    "the model. Restoring a subset would leave the rest at "
                    "their freshly initialised values.")
            with torch_mod.no_grad():
                for name, param in named.items():
                    incoming = state[name]
                    if tuple(incoming.shape) != tuple(param.shape):
                        raise GateRefused(
                            f"{name} is {tuple(incoming.shape)} in the "
                            f"checkpoint and {tuple(param.shape)} in this "
                            "model")
                    param.copy_(incoming.to(param.device, param.dtype))

        def save_rng_state(path) -> None:
            """Both generators, so the dropout stream continues rather than restarts.

            CPU as well as CUDA: the CPU generator is what seeds anything
            built later, and saving one of the two would leave a resume half
            in the stream and half at the beginning of it.

            ``get_rng_state`` returns ByteTensors, so this file is tensors and
            nothing else, and it is read back with ``weights_only=True``.
            """
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            cuda = getattr(torch_mod, "cuda", None)
            torch_mod.save({"cpu": torch_mod.get_rng_state(),
                            "cuda": list(cuda.get_rng_state_all())},
                           str(path))

        def load_rng_state(path) -> None:
            state = torch_mod.load(str(path), map_location="cpu",
                                   weights_only=True)
            cuda = getattr(torch_mod, "cuda", None)
            devices = cuda.device_count()
            saved = list(state.get("cuda") or [])
            if len(saved) != devices:
                raise GateRefused(
                    f"the checkpoint holds generator state for {len(saved)} "
                    f"CUDA device(s) and this machine has {devices}. The "
                    "stream cannot be continued onto a different number of "
                    "devices; start the run again rather than continue it.")
            torch_mod.set_rng_state(state["cpu"])
            cuda.set_rng_state_all(saved)

        def save_optimizer(path) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            optimizer = holder.get("optimizer")
            if optimizer is None:
                raise GateRefused(
                    "there is no optimizer to save; a checkpoint without one "
                    "restores Adam's moments from zero, which is a different "
                    "optimisation rather than a continuation")
            torch_mod.save(optimizer.state_dict(), str(path))

        def load_optimizer(path) -> None:
            state = torch_mod.load(str(path), map_location=self.device,
                                   weights_only=True)
            optimizer = holder.get("optimizer")
            if optimizer is None:
                raise GateRefused("there is no optimizer to restore into")
            optimizer.load_state_dict(state)

        def cold_load(saved: dict, adapter_dir) -> dict:
            """Build a model from what is on disk, having released the old one.

            :func:`src.training.lora.load_finetuned` is the loader that already
            knows the one correct order and refuses to guess: it checks the
            manifest, the pinned revisions and the adapter digest before it
            builds anything.
            """
            from src.training.lora import load_finetuned

            blob = Path(adapter_dir) / saved["path"]
            digest = sha256_file(blob) if blob.is_file() else None
            try:
                model, info = load_finetuned(
                    Path(adapter_dir), device=self.device, verify_digest=True)
            except Exception as exc:  # the refusal is the finding
                from src.training.longrun import _portable

                return {"loaded": False, "sha256": digest,
                        "matches_saved": False,
                        "reason": _portable(f"{type(exc).__name__}: {exc}")}
            del model
            return {"loaded": True, "sha256": digest,
                    "matches_saved": digest == saved.get("sha256"),
                    "load_order": info.get("load_order"), "reason": None}

        def trainable_digest() -> str:
            """One value over every trainable tensor's contents.

            The repeatability criterion is stated over three things -- input
            order, per-row loss, and the final weights -- because the first
            two can agree while the third does not: anything after the last
            measured row still moves the parameters.

            Sorted by name so the traversal order cannot change the answer,
            moved to CPU and widened to float32 because bfloat16 widens
            exactly and a raw byte view would make the digest depend on the
            storage layout rather than the values.
            """
            import hashlib

            model = holder.get("model")
            if model is None:
                raise GateRefused(
                    "the model has already been released, so the trainable "
                    "tensors cannot be digested. This has to be taken before "
                    "teardown.")
            h = hashlib.sha256()
            named = trainable_named(model)
            counted = 0
            for name, param in named:
                counted += 1
                h.update(name.encode("utf-8"))
                h.update(b"\0")
                tensor = param.detach().to("cpu", torch_mod.float32).contiguous()
                h.update(tensor.numpy().tobytes())
            if counted == 0:
                raise GateRefused(
                    "no trainable tensor was found to digest; a LoRA run with "
                    "nothing requiring grad has not trained anything")
            return h.hexdigest()

        def peak_memory() -> dict:
            """Two headline readings, and the allocator numbers that explain them.

            Reserved is what the process took from the driver and is what has
            to fit in the card; allocated is what was live. When they diverge
            -- gate 100 reserved 15.477 GB while 7.150 GB was live -- the pair
            alone cannot say whether the allocator was merely holding freed
            blocks or was fragmenting and retrying. ``num_alloc_retries`` and
            ``num_ooms`` settle that: an allocator that never retried was
            never under pressure. ``inactive_split_bytes`` says how much of
            the cache is split fragments rather than whole reusable blocks.

            Everything here is **observed**. Nothing is judged, and no
            allocator setting is read from or written to the environment: a
            diagnostic that reconfigures what it measures has measured
            something else.
            """
            def read(fn):
                try:
                    value = fn()
                except Exception:
                    return None
                return (round(value / 1024 ** 3, 3)
                        if isinstance(value, (int, float)) else None)

            def plain(fn):
                try:
                    value = fn()
                except Exception:
                    return None
                return value if isinstance(value, str) else None

            cuda = getattr(torch_mod, "cuda", None)
            try:
                stats = cuda.memory_stats()
            except Exception:
                stats = None
            if not isinstance(stats, dict):
                stats = None

            def stat(key, *, as_gb: bool):
                # ``None`` for absent, never 0. "The allocator never retried"
                # and "this build does not report retries" are opposite
                # findings, and a zero would read as the first.
                if stats is None or key not in stats:
                    return None
                value = stats[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return None
                return round(value / 1024 ** 3, 3) if as_gb else value

            return {
                "peak_vram_gb": read(getattr(cuda, "max_memory_reserved",
                                             lambda: None)),
                "peak_allocated_gb": read(getattr(cuda, "max_memory_allocated",
                                                  lambda: None)),
                "allocator_backend": plain(
                    getattr(cuda, "get_allocator_backend", lambda: None)),
                "inactive_split_bytes_current": stat(
                    "inactive_split_bytes.all.current", as_gb=True),
                "inactive_split_bytes_peak": stat(
                    "inactive_split_bytes.all.peak", as_gb=True),
                "num_alloc_retries": stat("num_alloc_retries", as_gb=False),
                "num_ooms": stat("num_ooms", as_gb=False),
            }

        def teardown() -> float:
            """Drop the model and the optimizer, then clear the CUDA cache.

            The base loader's teardown is report 16's, and report 16 runs on
            MPS: it calls ``torch.mps.empty_cache()``, which on this device
            raises. Overridden here rather than made conditional there, so the
            Mac's measured path keeps the exact teardown it was measured with.

            Order is load-bearing and is report 15's: the references go first,
            *then* the clear. Clearing while a merged 1B model, an adapter and
            Adam's two moment tensors are still referenced measures how much
            cannot be freed rather than how long freeing takes.
            """
            import time

            # References first, unconditionally, and outside anything that
            # can raise: whatever happens to the clear, the model must not be
            # left on the device.
            holder.pop("model", None)
            holder.pop("optimizer", None)
            t0 = time.perf_counter()
            cuda = getattr(torch_mod, "cuda", None)
            if cuda is None or not cuda.is_available():
                raise GateRefused(
                    "teardown could not reach a CUDA device to clear. This "
                    "class only ever runs on CUDA, so a device that has "
                    "disappeared by teardown is a broken run, not a tidy-up "
                    "detail.")
            # Deliberately not wrapped. A clear that silently failed leaves
            # the card holding memory the next measurement will inherit, and
            # lets a run report a peak that was never released -- which is
            # precisely the kind of quiet wrongness the peak reading exists
            # to make visible.
            cuda.empty_cache()
            return time.perf_counter() - t0

        loaded.update({"teardown": teardown,
                       "trainable_digest": trainable_digest,
                       "save_adapter": save_adapter,
                       "save_model_state": save_model_state,
                       "load_model_state": load_model_state,
                       "save_rng_state": save_rng_state,
                       "load_rng_state": load_rng_state,
                       "save_optimizer": save_optimizer,
                       "load_optimizer": load_optimizer,
                       "cold_load": cold_load,
                       "peak_memory": peak_memory})
        return loaded


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

#: ``teardown`` is in this list rather than merely called. It is used after
#: the last row -- which is the worst possible moment to discover a loader
#: does not supply it, with every row already measured and the model still on
#: the device.
_LOADED_REQUIRED = ("order", "step", "provenance", "sample_ids",
                    "model_load_seconds", "teardown", "save_adapter",
                    "save_model_state", "load_model_state",
                    "save_rng_state", "load_rng_state",
                    "save_optimizer", "load_optimizer", "cold_load",
                    "peak_memory", "trainable_digest")


def link_support_problems(directory, *, linker=os.link,
                          make_dirs: bool = True) -> list[str]:
    """Can this filesystem do the one thing every guarantee here rests on?

    ``write_once_json`` and ``copy_once`` publish by ``os.link``, because it
    creates the name or raises ``EEXIST`` with no window in between. Looking
    first and renaming has that window. So on a filesystem without hard links
    there is no safe way to publish at all -- and finding that out at the
    first checkpoint means finding it out with rows already measured.

    On this node the realistic cause is keeping the pack on the Windows
    filesystem rather than inside WSL2, which would also make the run too slow
    to be worth measuring. Checked here, against the directory the ledger and
    the checkpoints will actually live in, rather than in the preflight: the
    preflight writes nothing, and a check that writes belongs where the
    writing does.
    """
    directory = Path(directory)
    if make_dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return [f"cannot create the run directory ({exc})"]
    if not directory.is_dir():
        return [f"the run directory {directory.name!r} does not exist and was "
                "not created"]
    fd, probe = None, None
    try:
        fd, probe = tempfile.mkstemp(dir=str(directory), prefix=".linkprobe.")
        os.close(fd)
        fd = None
        target = probe + ".link"
        try:
            linker(probe, target)
        except OSError as exc:
            return [f"this filesystem cannot create a hard link ({exc}). "
                    "Every write-once guarantee here -- the manifest, the "
                    "plan, each checkpoint -- publishes by os.link, and there "
                    "is no other way to create a name that cannot overwrite "
                    "something. Put the run somewhere that can."]
        with contextlib.suppress(OSError):
            os.unlink(target)
    except OSError as exc:
        return [f"cannot write into the run directory ({exc})"]
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        if probe is not None:
            with contextlib.suppress(OSError):
                os.unlink(probe)
    return []


def _loader_problems(loaded, *, rows: int) -> list[str]:
    if not isinstance(loaded, dict):
        return [f"the loader returned {type(loaded).__name__}, not the "
                "mapping the contract describes"]
    problems = [f"the loader returned no {k!r}" for k in _LOADED_REQUIRED
                if k not in loaded]
    order = loaded.get("order")
    if not isinstance(order, list) or len(order) != rows:
        problems.append(
            f"the loader's order holds {len(order) if isinstance(order, list) else '?'} "
            f"entries, not the {rows} this gate declares")
    if not callable(loaded.get("step")):
        problems.append("the loader's 'step' is not callable")
    return problems


def _provenance_drift(before: dict, after: dict) -> list[str]:
    problems = []
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            problems.append(
                f"{key} was {before.get(key)!r} when this run started and is "
                f"{after.get(key)!r} now")
    return problems


def run_gate(name: str, *, deps: GateDeps, run_dir, pack_dir,
             expected_pack_digest, expected_dependency_digest,
             allocator_config, determinism,
             stop_after: int | None = None, resume: bool = False,
             verifier=None, dependency_checker=None, clock=None,
             spec: Gate | None = None, config: dict | None = None,
             loader_check=None) -> dict:
    """Run one gate, or refuse before measuring anything.

    ``expected_pack_digest`` and ``expected_dependency_digest`` have no
    defaults. Both are values the build machine printed and carried here
    separately, and each checks something the thing being checked cannot
    establish about itself: a pack rewritten wholesale agrees with its own
    manifest, and a cache of correctly-named files says nothing about what is
    inside them. A trust check with a default is one that gets omitted, and
    omitting either fails open.

    ``stop_after`` is how gate 500's deliberate stop is performed: the run
    raises :class:`DeliberateStop` after that row, having checkpointed
    normally on the way. It is not an error path bolted on for a test -- an
    interruption that only ever happens by accident is one nobody has a
    recovery procedure for.
    """
    import time

    # ``spec`` exists for runs that use this machinery and are not gates --
    # the two frozen hypothesis arms, which need the same plan, the same
    # ledger, the same checkpoint contract and the same resume, and which no
    # verdict here judges. It cannot be used to make a *gate* smaller: a name
    # that is a gate always runs the frozen gate specification, whatever is
    # passed.
    if name in GATES:
        if spec is not None and spec != GATES[name]:
            raise GateRefused(
                f"{name!r} is a gate, and its specification is frozen at "
                f"{GATES[name]}. A gate that could be handed a different row "
                "count or checkpoint stride would be a gate in name only.")
        gate = GATES[name]
    elif spec is None:
        raise GateRefused(
            f"{name!r} is not one of the gates {sorted(GATES)} and no run "
            "specification was supplied, so there is nothing that says how "
            "long this run is or when it checkpoints")
    elif spec.name != name:
        raise GateRefused(
            f"the run is called {name!r} and the specification supplied is "
            f"for {spec.name!r}")
    else:
        gate = spec
    clock = clock or time.perf_counter
    run_dir = Path(run_dir)

    # Trust and integrity first, and both before anything on disk is created.
    # ``link_support_problems`` makes the run directory, and a refusal that has
    # already made one has already started a run -- the next thing anybody does
    # with a half-made run directory is resume it.
    problems = pack.trusted_digest_problems(pack_dir, expected_pack_digest)
    problems += dependency_digest_problems(expected_dependency_digest,
                                           checker=dependency_checker)
    problems += runtime_problems(allocator_config, determinism)
    problems += list((verifier or pack.verify)(pack_dir))
    if problems:
        raise GateRefused("refusing to run the gate:\n  - "
                          + "\n  - ".join(problems))

    manifest, manifest_problems = pack.read_manifest(pack_dir)
    if manifest_problems:
        raise GateRefused("refusing to run the gate:\n  - "
                          + "\n  - ".join(manifest_problems))

    problems = link_support_problems(run_dir)
    if problems:
        raise GateRefused("refusing to run the gate:\n  - "
                          + "\n  - ".join(problems))

    existing = read_plan(run_dir)
    point = None
    if resume:
        point = resume_point(
            run_dir, pack_dir=pack_dir,
            expected_pack_digest=expected_pack_digest,
            expected_dependency_digest=expected_dependency_digest,
            allocator_config=allocator_config, determinism=determinism,
            verifier=verifier, dependency_checker=dependency_checker)
        plan = existing
        start, attempt = point["next_position"], point["attempt"]
        resumed_from = point["resume_from"]
    else:
        if existing is not None:
            raise GateRefused(
                f"{run_dir.name} already holds a plan for {existing.get('gate')!r}. "
                "Starting fresh over it would abandon its ledger and its "
                "checkpoints without saying so; resume it, or use a new "
                "directory.")
        plan, start, attempt, resumed_from = None, 1, 1, 0

    model_state_restored = rng_state_restored = False
    loaded = deps.load(rows=gate.rows)
    loader_problems = _loader_problems(loaded, rows=gate.rows)
    if loader_problems:
        raise GateRefused("the loader did not honour its contract:\n  - "
                          + "\n  - ".join(loader_problems))

    # A caller's own question about what the loader produced, asked here
    # because here is after the load and before the plan is written -- which
    # is the only moment at which the answer can still stop the run without
    # anything having been measured or recorded.
    if loader_check is not None:
        extra = list(loader_check(loaded))
        if extra:
            raise GateRefused("the loader did not honour its contract:\n  - "
                              + "\n  - ".join(extra))

    provenance = loaded["provenance"]
    if plan is None:
        plan = write_plan(run_dir, gate=name, order=loaded["order"],
                          pack_digest=manifest.get("pack_digest"),
                          config=config if config is not None
                          else LoraConfig_().as_dict(),
                          provenance=provenance,
                          expected_dependency_digest=expected_dependency_digest,
                          allocator_config=allocator_config,
                          determinism=determinism)
    else:
        drift = _provenance_drift(plan.get("provenance") or {}, provenance)
        if drift:
            raise GateRefused(
                "refusing to resume: the provenance has changed since this "
                "run was planned, so continuing it would silently splice two "
                "different runs together:\n  - " + "\n  - ".join(drift))
        if list(loaded["order"]) != list(plan.get("order") or []):
            raise GateRefused(
                "refusing to resume: the loader produced a different input "
                "order from the one this run was planned against")
        if resumed_from:
            # The point resolved above, not resolved again: re-running the
            # whole resume check here would verify a second time and, worse,
            # would be a second place for "which checkpoint" to be decided.
            saved_at = Path(point["checkpoint_dir"])
            # Weights first. The optimizer's state refers to the parameters it
            # is attached to; restoring it onto a model whose tensors are
            # about to be replaced puts Adam's moments beside weights they
            # were never computed for.
            loaded["load_model_state"](saved_at / MODEL_STATE_NAME)
            restored = loaded["trainable_digest"]()
            if restored != point["trainable_digest"]:
                raise GateRefused(
                    "refusing to resume: after restoring the checkpoint's "
                    f"model state the trainable tensors digest to "
                    f"{str(restored)[:16]}..., not the "
                    f"{str(point['trainable_digest'])[:16]}... recorded when "
                    "it was written. The file matched its own digest, so what "
                    "failed is the load, not the bytes: these are not the "
                    "weights the run stopped on.")
            model_state_restored = True
            loaded["load_optimizer"](saved_at / OPTIMIZER_NAME)
            # Last, next to what consumes it: the first row of this attempt
            # must draw the mask the interrupted attempt would have drawn.
            loaded["load_rng_state"](saved_at / RNG_STATE_NAME)
            rng_state_restored = True

    order = list(loaded["order"])
    step = loaded["step"]
    forward = backward = 0
    started = clock()
    per_row: list[dict] = []
    stopped_at = None

    try:
        for position in range(start, gate.rows + 1):
            t0 = clock()
            measured = step(order[position - 1], position)
            elapsed = clock() - t0
            forward += 1
            backward += 1
            entry = {
                "attempt": attempt,
                "position": position,
                "index": int(order[position - 1]),
                "resumed_from": resumed_from,
                "loss": float(measured["loss"]),
                "sample_id": measured["sample_id"],
                "tokens": int(measured["tokens"]),
                "supervised_tokens": int(measured["supervised_tokens"]),
                "seconds": round(elapsed, 6),
            }
            append_ledger(run_dir, entry)
            per_row.append(entry)

            if position % gate.checkpoint_every == 0 and position < gate.rows:
                _take_checkpoint(run_dir, position=position, attempt=attempt,
                                 plan=plan, loaded=loaded,
                                 provenance=provenance)
            if stop_after is not None and position >= stop_after:
                stopped_at = position
                raise DeliberateStop(position)
    finally:
        if stopped_at is None and stop_after is None:
            pass

    total = clock() - started
    adapter_dir = run_dir / "adapter"
    saved = loaded["save_adapter"](adapter_dir)
    # Before teardown, which drops the model: after it there is nothing left
    # to digest. The repeatability criterion is stated over three things --
    # input order, per-row loss, and the final trainable tensor content --
    # and this is the third.
    trainable = loaded["trainable_digest"]()
    # Teardown *before* the cold load, so the cold load is cold. Reloading
    # while the trained model is still resident proves the file parses; it
    # does not prove the machine can build a model from it, which is the
    # thing gate 8 is for.
    loaded["teardown"]()
    cold = loaded["cold_load"](saved, adapter_dir)
    memory = loaded["peak_memory"]() or {}

    entries = read_ledger(run_dir)
    checkpoints = [c["position"] for c in read_checkpoints(run_dir)]
    # Where the run was interrupted is a property of the *run*, not of the
    # attempt that finished it: the stop happened in the attempt before this
    # one, and a resumed attempt that reported ``stopped_at: None`` would make
    # a stop-and-resume indistinguishable from a run that was never stopped.
    prior = [e["position"] for e in entries
             if _is_int(e.get("attempt")) and _is_int(e.get("position"))
             and e["attempt"] == attempt - 1]
    if stopped_at is None and prior:
        stopped_at = max(prior)
    problems = ledger_problems(entries, order=order,
                               declared_rows=gate.rows,
                               checkpoint_positions=checkpoints)
    effective = effective_ledger(entries)
    losses = [e.get("loss") for e in effective.values()]
    windows = _windows(effective)

    return {
        "gate": name,
        "rows_declared": gate.rows,
        "rows_completed": len(effective),
        "attempts": attempt,
        "resumed_from": resumed_from or None,
        "stopped_at": stopped_at,
        "checkpoints": checkpoints,
        # Both digests, from the plan rather than from this moment. This is
        # what travels back to the Mac, and a result that cannot name which
        # pack and which tokenizer, base model and adapter bytes produced it
        # is a number without provenance.
        "pack_digest": plan.get("pack_digest"),
        "dependency_digest": plan.get("dependency_digest"),
        "allocator_config": plan.get("allocator_config"),
        "determinism": plan.get("determinism"),
        "trainable_digest": trainable,
        # Read from the loader and passed through unjudged. What the rows were
        # taken from, and whether any of them did not fit, are properties of
        # the data and the tokenizer rather than of the run -- and the run is
        # the only thing that can write them down.
        "data_source": loaded.get("data_source"),
        "truncated_rows": loaded.get("truncated_rows"),
        "max_total_tokens": loaded.get("max_total_tokens"),
        # Three restorations, reported apart. Any two of them without the
        # third is the failure this triple exists to make visible: a resume
        # can do it while looking, from every other field here, correct.
        "model_state_restored": model_state_restored,
        "rng_state_restored": rng_state_restored,
        "optimizer_state_restored": bool(resumed_from),
        "model_load_seconds": loaded["model_load_seconds"],
        "forward_rows": len(effective),
        "backward_rows": len(effective),
        "optimizer_steps": len(effective) // _ACCUM,
        "measured_this_attempt": len(per_row),
        "seconds_per_row": (round(total / len(per_row), 6) if per_row else None),
        "peak_vram_gb": memory.get("peak_vram_gb"),
        # Recorded, never judged. Reserved is what has to fit in the card and
        # is what the threshold is on; allocated is what was actually live.
        # With only reserved, "the run needs 15.5 GB" and "the caching
        # allocator grew to 15.5 GB" are the same number and call for
        # opposite responses. ``None`` when it could not be read, so an
        # absent reading stays distinguishable from a low one.
        "peak_allocated_gb": memory.get("peak_allocated_gb"),
        # Observed only. No verdict reads any of these; the bound stays on
        # reserved at MAX_PEAK_VRAM_GB, declared before this machine produced
        # a number. They exist to say *why* reserved and allocated diverged.
        "allocator_backend": memory.get("allocator_backend"),
        "inactive_split_bytes_current": memory.get("inactive_split_bytes_current"),
        "inactive_split_bytes_peak": memory.get("inactive_split_bytes_peak"),
        "num_alloc_retries": memory.get("num_alloc_retries"),
        "num_ooms": memory.get("num_ooms"),
        "loss_first_window": windows[0]["loss"] if windows else None,
        "loss_last_window": windows[-1]["loss"] if windows else None,
        "windows": windows,
        "losses_finite": all(finite(v) for v in losses) if losses else False,
        "saved": saved,
        "cold_load": cold,
        "ledger_problems": problems,
        "duplicate_positions": duplicate_positions(entries),
        "missing_positions": missing_positions(entries, gate.rows),
        "provenance": provenance,
        "speed_threshold": SPEED_THRESHOLD_SECONDS_PER_ROW,
        "speed_threshold_reason": SPEED_THRESHOLD_REASON,
    }


def _take_checkpoint(run_dir, *, position, attempt, plan, loaded, provenance):
    """Weights, optimizer, generator, and the record of all three.

    Every blob is written into a staging directory and then linked into the
    checkpoint by :func:`~src.training.session.copy_once`, which creates the
    name or refuses because it is taken. Writing them straight to their final
    names would let a second checkpoint at the same row overwrite the first,
    and ``state.json`` being write-once would then leave a record describing
    weights that were replaced underneath it -- worse than no checkpoint,
    because it still verifies.
    """
    target = checkpoint_dir(run_dir, position)
    staging = Path(tempfile.mkdtemp(dir=str(run_dir), prefix=".ckpt."))
    try:
        model_tmp = staging / MODEL_STATE_NAME
        loaded["save_model_state"](model_tmp)
        if not model_tmp.is_file():
            raise GateRefused(
                f"the loader's save_model_state wrote no {MODEL_STATE_NAME}; "
                "a checkpoint without the trainable tensors cannot be "
                "resumed from, only restarted")
        optimizer_tmp = staging / OPTIMIZER_NAME
        loaded["save_optimizer"](optimizer_tmp)
        if not optimizer_tmp.is_file():
            raise GateRefused(
                f"the loader's save_optimizer wrote no {OPTIMIZER_NAME}")
        rng_tmp = staging / RNG_STATE_NAME
        loaded["save_rng_state"](rng_tmp)
        if not rng_tmp.is_file():
            raise GateRefused(
                f"the loader's save_rng_state wrote no {RNG_STATE_NAME}; "
                "without it a resume restarts the dropout stream from the "
                "seed and measures different rows from the same weights")
        # Taken from the live model once the files are on disk and before
        # any of them is published, so it describes the weights that were
        # saved. This is the value the resume re-computes after loading them.
        trainable = loaded["trainable_digest"]()
        model_bytes = model_tmp.stat().st_size
        rng_bytes = rng_tmp.stat().st_size

        target.mkdir(parents=True, exist_ok=True)
        model_sha = copy_once(model_tmp, target / MODEL_STATE_NAME)
        optimizer_sha = copy_once(optimizer_tmp, target / OPTIMIZER_NAME)
        rng_sha = copy_once(rng_tmp, target / RNG_STATE_NAME)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    upto = [e for e in read_ledger(run_dir)
            if _is_int(e.get("position")) and e["position"] <= position]
    write_checkpoint(run_dir, position=position, attempt=attempt, plan=plan,
                     optimizer_sha256=optimizer_sha,
                     model_state={"name": MODEL_STATE_NAME,
                                  "sha256": model_sha,
                                  "bytes": model_bytes},
                     rng_state={"name": RNG_STATE_NAME,
                                "sha256": rng_sha,
                                "bytes": rng_bytes},
                     trainable_digest=trainable,
                     ledger_entries=upto, provenance=provenance)


def _windows(effective: dict[int, dict]) -> list[dict]:
    positions = sorted(effective)
    out = []
    for start in range(0, len(positions), WINDOW):
        chunk = [effective[p] for p in positions[start:start + WINDOW]]
        if not chunk:
            continue
        seconds = [e.get("seconds") for e in chunk if finite(e.get("seconds"))]
        losses = [e.get("loss") for e in chunk if finite(e.get("loss"))]
        out.append({
            "window": start // WINDOW,
            "rows": f"{positions[start]}-{positions[min(start + WINDOW, len(positions)) - 1]}",
            "n_rows": len(chunk),
            "seconds_per_row": (round(sum(seconds) / len(seconds), 6)
                                if seconds else None),
            "loss": round(sum(losses) / len(losses), 6) if losses else None,
        })
    return out


# ---------------------------------------------------------------------------
# Verdicts. Fail closed: no evidence is not a pass.
# ---------------------------------------------------------------------------

def gate_problems(name: str, evidence: dict | None) -> list[str]:
    """Why this evidence does not clear the gate. Empty means it does."""
    gate = GATES[name]
    e = evidence or {}
    problems: list[str] = []

    rows = e.get("rows_completed")
    if not _is_int(rows):
        problems.append("rows_completed was not recorded")
    elif rows != gate.rows:
        problems.append(
            f"{rows} rows were completed, not the {gate.rows} this gate "
            "declares")

    for key in ("ledger_problems",):
        recorded = e.get(key)
        if recorded is None:
            problems.append(f"{key} was not recorded, so nothing checked the "
                            "row-by-row record")
        elif recorded:
            problems += [f"ledger: {p}" for p in recorded]

    if e.get("losses_finite") is not True:
        problems.append("not every row recorded a finite loss")

    if name == "gate_8":
        problems += _gate_8_problems(e)
    elif name == "gate_100":
        problems += _gate_100_problems(e)
    elif name == "gate_500":
        problems += _gate_500_problems(e)
    return problems


def _gate_8_problems(e: dict) -> list[str]:
    problems = []
    load_seconds = e.get("model_load_seconds")
    if not finite(load_seconds) or load_seconds <= 0:
        problems.append(
            f"model_load_seconds is {load_seconds!r}; a run that did not "
            "measurably load a model did not load one")
    for key in ("forward_rows", "backward_rows"):
        value = e.get(key)
        if not _is_int(value) or value != GATES["gate_8"].rows:
            problems.append(
                f"{key} is {value!r}, not {GATES['gate_8'].rows}")
    steps = e.get("optimizer_steps")
    if not _is_int(steps) or steps < 1:
        problems.append(
            f"optimizer_steps is {steps!r}; eight rows at grad_accum "
            f"{_ACCUM} must step the optimizer at least once, and a run that "
            "never stepped it trained nothing")
    saved = e.get("saved")
    if not isinstance(saved, dict) or not saved.get("sha256"):
        problems.append(
            "no adapter was saved with a digest, so there is nothing a cold "
            "load could be checked against")
    cold = e.get("cold_load")
    if not isinstance(cold, dict):
        problems.append("no cold load was performed")
    else:
        if cold.get("loaded") is not True:
            problems.append("the cold load did not find what was saved")
        if cold.get("matches_saved") is not True:
            problems.append(
                "the cold load's digest does not match what was saved; the "
                "adapter on disk is not the adapter that was written")
    return problems


def _gate_100_problems(e: dict) -> list[str]:
    problems = []
    speed = e.get("seconds_per_row")
    if not finite(speed) or speed <= 0:
        problems.append(
            f"seconds_per_row is {speed!r}. There is no threshold on this "
            "reading, but there is no gate without the reading either.")
    vram = e.get("peak_vram_gb")
    if not finite(vram):
        problems.append(
            "peak VRAM could not be read, and an unread reading is not a "
            "healthy one")
    elif vram > MAX_PEAK_VRAM_GB:
        problems.append(
            f"peak VRAM {vram} GB exceeds the {MAX_PEAK_VRAM_GB} GB this card "
            "has to give; the next longer run does not slow down, it dies")
    first, last = e.get("loss_first_window"), e.get("loss_last_window")
    if not finite(first) or not finite(last):
        problems.append(
            f"the first and last window losses are {first!r} and {last!r}; "
            "both must be finite for stability to mean anything")
    elif first > 0 and last > first * DIVERGENCE_FACTOR:
        problems.append(
            f"the loss diverged: {first} at the start, {last} at the end, "
            f"more than {DIVERGENCE_FACTOR}x worse")
    if not e.get("windows"):
        problems.append("no windows were recorded, so no trend was measured")
    return problems


def _gate_500_problems(e: dict) -> list[str]:
    problems = []
    checkpoints = e.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) < 1:
        problems.append(
            "no checkpoint was taken, so nothing was there to resume from")
    if not _is_int(e.get("stopped_at")):
        problems.append(
            "no deliberate stop was recorded; a resume that was never "
            "preceded by a stop has not been demonstrated")
    if not _is_int(e.get("resumed_from")):
        problems.append("no resume point was recorded")
    if e.get("model_state_restored") is not True:
        problems.append(
            "the trainable model state was not restored; a resume that "
            "rebuilds the adapter from scratch continues from weights the run "
            "never had, however tidy the ledger looks")
    if e.get("optimizer_state_restored") is not True:
        problems.append(
            "the optimizer state was not restored; a resume that starts Adam's "
            "moments from zero is a different optimisation, not a continuation")
    if e.get("rng_state_restored") is not True:
        problems.append(
            "the generator was not restored; with dropout on, a resume that "
            "restarts the stream at the seed measures different rows from the "
            "same weights and ends somewhere the uninterrupted run would not")
    attempts = e.get("attempts")
    if not _is_int(attempts) or attempts < 2:
        problems.append(
            f"attempts is {attempts!r}; a stop-and-resume takes at least two")
    dupes = e.get("duplicate_positions")
    if dupes is None:
        problems.append("duplicate_positions was not computed")
    elif dupes:
        problems.append(f"row(s) {_span(dupes)} were trained on twice")
    missing = e.get("missing_positions")
    if missing is None:
        problems.append("missing_positions was not computed")
    elif missing:
        problems.append(f"row(s) {_span(missing)} were never trained on")
    return problems


def verdict(name: str, evidence: dict | None) -> str:
    """``passed`` or ``failed``. There is no third answer and no default."""
    return "passed" if not gate_problems(name, evidence) else "failed"


def summary() -> dict:
    """The three gates as a document, for a report to embed verbatim."""
    return {
        "gates": {name: {"rows": g.rows,
                         "checkpoint_every": g.checkpoint_every,
                         "proves": list(g.proves)}
                  for name, g in GATES.items()},
        "max_peak_vram_gb": MAX_PEAK_VRAM_GB,
        "divergence_factor": DIVERGENCE_FACTOR,
        "speed_threshold": SPEED_THRESHOLD_SECONDS_PER_ROW,
        "speed_threshold_reason": SPEED_THRESHOLD_REASON,
    }
