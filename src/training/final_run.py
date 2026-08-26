"""The final training run: H2, the whole split, once, from a clean start.

H2 was selected on held-out loss over the 320 frozen validation rows. This is
the run that produces the model, and it differs from the two arms in three
ways that all have to be *enforced* rather than intended.

**It reads the whole training split.** 9,584 rows -- 1,198 whole pairs -- not
the frozen 2,000-row pool every measurement in this project has used until
now. The shape is declared here and checked against the file, so a split that
changed size is a refusal rather than a quietly different experiment.

**It starts clean.** Base, then the published BrickGPT adapter, then the
merge, then a *freshly initialised* LoRA. Not H2's weights, not H2's optimizer
state, not H2's generator. That is not a promise: a run that restored anything
records it in its own evidence, so the three restoration flags being false is
a property checked here. There is no resume, no stop, no ``--from-adapter``
and no second attempt, because none of those exist in this module or in the
script that drives it.

**Truncation stops it.** ``max_length`` is 2,048 and a truncated row trains on
a target that stops mid-structure and reports a perfectly ordinary loss --
silent by construction. The reading comes from the loader, is checked after
the load and *before the plan is written*, and any truncated row at all ends
the run before a single row is measured.

What it shares with everything before it is the machinery: the same six-role
gate suite has to verify, the same write-once plan, the same append-only
ledger, the same checkpoint contract carrying weights, optimizer state and
generator, and the same operational validator discipline -- a failure leaves
immutable failure evidence and no ordinary evidence file at all.

It declares no winner and starts nothing else. It produces one replayable
measurement and stops.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.training import gate_suite, gates, hypotheses, pack
from src.training.longrun import DATA_SOURCES, _portable
from src.training.lora import LoraConfig_
from src.training.session import now_iso, write_once_json

#: Selected on held-out loss, and frozen here so nothing downstream re-decides.
SELECTED_ARM = "H2"

#: The whole training split. Read from the loader's own declaration rather
#: than restated, so there is one place that says how big it is.
FULL_TRAIN_PAIRS = DATA_SOURCES["full_train"]["pairs"]
FULL_TRAIN_ROWS = DATA_SOURCES["full_train"]["rows"]
DATA_SOURCE = "full_train"
FINAL_EPOCHS = 1

RUN_NAME = "final_H2"

#: Every 512 rows, which is 64 optimizer steps.
#:
#: The gates checkpoint every 64 rows because gate 500 exists to be
#: interrupted and the cost of re-executing a few rows is the thing being
#: measured. This run is not that. At roughly a tenth of a second per row the
#: whole 9,584 rows take about fifteen minutes, so a checkpoint every 64 rows
#: would write some six gigabytes of rank-32 adapter, optimizer and generator
#: state in order to save, at worst, six seconds of recomputation. 512 rows
#: bounds a crash at about forty-eight seconds of lost work and costs eighteen
#: checkpoints. It is a multiple of ``grad_accum``, so no checkpoint lands
#: mid-accumulation with gradients nothing will apply.
FINAL_CHECKPOINT_EVERY = 64 * LoraConfig_().grad_accum

FAILURE_NAME = f"{RUN_NAME}_failure.json"

FINAL_PROVES = (
    "one epoch over the whole 9,584-row training split",
    "the selected arm's frozen rank, alpha and learning rate",
    "a freshly initialised adapter on the merged BrickGPT",
    "no truncated row",
    "the final trainable tensors, digested",
)


class FinalRunRefused(RuntimeError):
    """Stopped rather than start something that could not be vouched for."""


class FinalRunFailed(RuntimeError):
    """It ran and did not produce a measurement anybody can use."""


def frozen_config() -> LoraConfig_:
    """The selected arm's configuration, from the declaration and nowhere else."""
    return hypotheses.config_for(SELECTED_ARM)


def expected_optimizer_steps() -> int:
    """9,584 rows at ``grad_accum`` 8 is 1,198 steps. Derived, not typed."""
    return FULL_TRAIN_ROWS // LoraConfig_().grad_accum


def final_spec() -> gates.Gate:
    return gates.Gate(name=RUN_NAME, rows=FULL_TRAIN_ROWS,
                      checkpoint_every=FINAL_CHECKPOINT_EVERY,
                      proves=FINAL_PROVES)


def evidence_path(run_dir) -> Path:
    return gate_suite.evidence_path(run_dir, RUN_NAME)


def failure_path(run_dir) -> Path:
    return Path(run_dir) / FAILURE_NAME


def write_failure(run_dir, reason: str, *, problems=None,
                  measured: dict | None = None) -> Path:
    """Publish why this run stopped, once, and never rewrite it."""
    body = {"run": RUN_NAME, "arm": SELECTED_ARM,
            "reason": _portable(str(reason)),
            "problems": [_portable(str(p)) for p in (problems or [])],
            "written_at": now_iso()}
    if measured is not None:
        body["measured"] = measured
    write_once_json(failure_path(run_dir), body)
    return failure_path(run_dir)


def truncation_problems(loaded) -> list[str]:
    """Did every row fit? Asked of the loader, before the plan is written.

    ``None`` and absent are refusals rather than zero. "No row was truncated"
    and "nobody counted" are opposite findings, and only one of them is a
    reason to keep going.
    """
    value = (loaded or {}).get("truncated_rows")
    if isinstance(value, bool) or not isinstance(value, int):
        return ["the loader reported no truncated-row count, so nothing says "
                "whether the whole split fits in max_length. An uncounted "
                "reading is not a zero."]
    if value:
        return [f"{value} row(s) were truncated at max_length "
                f"{frozen_config().max_length}. A truncated row trains on a "
                "target that stops mid-structure and reports an ordinary "
                "loss, so this is silent unless it is refused here."]
    return []


def operational_problems(evidence: dict | None) -> list[str]:
    """Everything a finished final run has to be before its evidence exists.

    Absolute values throughout: the row count is the split's, the step count
    is derived from it, and the configuration is the declared arm's. Nothing
    is read from the evidence's own idea of what it was supposed to be.
    """
    e = evidence or {}
    problems: list[str] = []

    for field in ("rows_declared", "rows_completed"):
        value = e.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            problems.append(f"{field} is {value!r}, not a count")
        elif value != FULL_TRAIN_ROWS:
            problems.append(
                f"{field} is {value}, not the {FULL_TRAIN_ROWS} rows the whole "
                "training split holds")

    steps = e.get("optimizer_steps")
    want = expected_optimizer_steps()
    if not isinstance(steps, int) or isinstance(steps, bool):
        problems.append(f"optimizer_steps is {steps!r}, not a count")
    elif steps != want:
        problems.append(
            f"optimizer_steps is {steps}, not the {want} that "
            f"{FULL_TRAIN_ROWS} rows at grad_accum {LoraConfig_().grad_accum} "
            "must produce")

    if e.get("losses_finite") is not True:
        problems.append("not every row recorded a finite loss")

    recorded = e.get("ledger_problems")
    if recorded is None:
        problems.append("ledger_problems was not recorded")
    elif recorded:
        problems += [f"ledger: {p}" for p in recorded]

    problems += truncation_problems(e)

    # Fresh, and checkably so. A run that restored weights, optimizer state or
    # the generator continued something; this one is required not to have.
    for field in ("model_state_restored", "rng_state_restored",
                  "optimizer_state_restored"):
        if e.get(field) is not False:
            problems.append(
                f"{field} is {e.get(field)!r}; the final run starts from a "
                "freshly initialised adapter on the merged base and inherits "
                "nothing from the arms")
    attempts = e.get("attempts")
    if attempts != 1:
        problems.append(
            f"attempts is {attempts!r}; the final run is one attempt. A second "
            "one would mean it was interrupted and continued, and what to do "
            "about that is a person's decision.")

    cfg = frozen_config().as_dict()
    if e.get("config") != cfg:
        differing = sorted(k for k in set(cfg) | set(e.get("config") or {})
                           if (e.get("config") or {}).get(k) != cfg.get(k))
        problems.append(
            f"the recorded config differs from {SELECTED_ARM}'s frozen one in "
            f"{differing}")
    if e.get("epochs") != FINAL_EPOCHS:
        problems.append(
            f"epochs is {e.get('epochs')!r}, not the declared {FINAL_EPOCHS}")

    saved = e.get("saved")
    saved_sha = saved.get("sha256") if isinstance(saved, dict) else None
    if not isinstance(saved, dict):
        problems.append("no adapter was saved")
    else:
        problems += [f"the saved adapter's digest is unusable: {p}"
                     for p in pack.expected_digest_problems(
                         saved_sha, what="adapter digest")]

    cold = e.get("cold_load")
    if not isinstance(cold, dict):
        problems.append("no cold load was performed")
    else:
        if cold.get("loaded") is not True:
            problems.append(
                "the cold load did not rebuild a model from what was saved"
                + (f" ({cold.get('reason')})" if cold.get("reason") else ""))
        if cold.get("matches_saved") is not True:
            problems.append("the cold load's digest does not match what was "
                            "saved")
        cold_sha = cold.get("sha256")
        if not isinstance(cold_sha, str) or not cold_sha:
            problems.append("the cold load recorded no digest of what it read")
        elif isinstance(saved_sha, str) and cold_sha != saved_sha:
            problems.append(
                f"the cold load read {cold_sha[:16]}... where the adapter was "
                f"saved as {str(saved_sha)[:16]}...")

    problems += [f"the final trainable digest is unusable: {p}"
                 for p in pack.expected_digest_problems(
                     e.get("trainable_digest"), what="trainable digest")]
    return problems


def run_final(*, deps_factory, run_dir, pack_dir, gate_runs,
              expected_pack_digest, expected_dependency_digest,
              allocator_config, determinism, verifier=None,
              dependency_checker=None, clock=None) -> dict:
    """Verify the suite, then train once. No resume, no retry, no inheritance.

    ``deps_factory`` is called with the frozen configuration **after** the
    unlock and never before, so nothing is built until the run is allowed.
    """
    cfg = frozen_config()
    run_dir = Path(run_dir)

    for existing in (evidence_path(run_dir), failure_path(run_dir)):
        if existing.exists():
            raise FinalRunRefused(
                f"{run_dir.name} already holds {existing.name}. The final run "
                "happens once in a directory: a failure record is immutable "
                "and a finished run is the record. Starting again is a "
                "decision somebody makes deliberately, in a new directory.")

    hypotheses.require_unlocked(
        SELECTED_ARM, runs=gate_runs,
        expected_pack_digest=expected_pack_digest,
        expected_dependency_digest=expected_dependency_digest,
        allocator_config=allocator_config, determinism=determinism)

    enriched = None
    try:
        deps = deps_factory(cfg)
        evidence = gates.run_gate(
            RUN_NAME, deps=deps, run_dir=run_dir, pack_dir=pack_dir,
            expected_pack_digest=expected_pack_digest,
            expected_dependency_digest=expected_dependency_digest,
            allocator_config=allocator_config, determinism=determinism,
            resume=False, verifier=verifier,
            dependency_checker=dependency_checker, clock=clock,
            spec=final_spec(), config=cfg.as_dict(),
            loader_check=truncation_problems)
        enriched = _enriched(cfg, evidence, run_dir, deps_loaded=None)
        problems = operational_problems(enriched)
        if problems:
            raise FinalRunFailed(
                "the final run did not produce a usable measurement:\n  - "
                + "\n  - ".join(problems))
        enriched["operational_problems"] = []
    except BaseException as exc:
        write_failure(run_dir, f"{type(exc).__name__}: {exc}",
                      problems=(operational_problems(enriched)
                                if enriched is not None else []),
                      measured=enriched)
        if isinstance(exc, gates.GateRefused) and "truncat" in str(exc):
            raise FinalRunFailed(str(exc)) from exc
        raise

    write_once_json(evidence_path(run_dir), enriched)
    return enriched


def _enriched(cfg: LoraConfig_, evidence: dict, run_dir: Path,
              deps_loaded) -> dict:
    plan = gates.read_plan(run_dir) or {}
    effective = gates.effective_ledger(gates.read_ledger(run_dir))
    positions = sorted(effective)
    out = dict(evidence)
    out.pop("speed_threshold", None)
    out.pop("speed_threshold_reason", None)
    out.update({
        "run": RUN_NAME,
        "arm": SELECTED_ARM,
        "config": cfg.as_dict(),
        "epochs": FINAL_EPOCHS,
        "order_digest": plan.get("order_digest"),
        "sample_ids": [effective[p].get("sample_id") for p in positions],
        "per_row_loss": [effective[p].get("loss") for p in positions],
        "adapter": evidence.get("saved"),
    })
    return out


def summary() -> dict:
    """What the final run is, for a report to embed verbatim."""
    return {
        "run": RUN_NAME,
        "selected_arm": SELECTED_ARM,
        "config": frozen_config().as_dict(),
        "rows": FULL_TRAIN_ROWS,
        "pairs": FULL_TRAIN_PAIRS,
        "epochs": FINAL_EPOCHS,
        "expected_optimizer_steps": expected_optimizer_steps(),
        "checkpoint_every": FINAL_CHECKPOINT_EVERY,
        "data_source": DATA_SOURCE,
        "starts_from": "base -> published adapter -> merge -> fresh LoRA",
        "inherits": "nothing",
        "resumable": False,
        "declares_a_winner": False,
    }
