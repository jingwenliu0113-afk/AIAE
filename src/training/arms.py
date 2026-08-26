"""Turning a frozen hypothesis into a run, and refusing to before it is time.

H1 and H2 are declared in :mod:`src.training.hypotheses` and are settings, not
runs. This is the only thing in the project that turns one into the other, and
almost everything that can go wrong about it goes wrong before the first row:
the wrong pack, dependency bytes that changed, an allocator inherited from a
different shell, a gate suite that does not exist or does not agree, or a
configuration that somebody nudged on a command line.

So the order here is the design.

1. The six-role gate suite is proved, through
   :func:`src.training.hypotheses.require_unlocked`, which re-derives every
   verdict from evidence and binds all six to the values carried in.
2. Only then is a loader built. That is why this takes a ``deps_factory``
   rather than a loader: a loader is a model, and a model is a spent boot.
   Nothing can be built before the unlock because there is nothing to build
   it with until the unlock has produced the configuration.
3. Only then does the runner touch disk. ``run_gate`` performs its own trust
   checks again and creates the run directory after them.

Three things this module deliberately does not do.

**It has no verdict.** ``gate_problems`` does not know these names and nothing
here invents a threshold. The arms exist to be compared with each other, by a
person, after both have run; a runner that announced a winner would be
choosing the success criterion after seeing the numbers, which is the exact
failure the whole frozen-declaration arrangement exists to prevent.

**It has no second checkpoint format.** The plan, the ledger, the write-once
checkpoint carrying weights, optimizer state and generator, and the resume
that re-digests what it restored are :mod:`src.training.gates`'s, unchanged.
A weaker contract for the longer, more expensive runs would be exactly
backwards.

**It runs one arm.** There is no loop over the two and no retry. If an arm
fails, immutable failure evidence is written and the exception propagates;
deciding what to do next is a person's job, and starting the second arm on
the wreckage of the first would produce two runs that are not comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.training import gate_suite, gates, hypotheses, pack
from src.training.longrun import _portable
from src.training.lora import LoraConfig_
from src.training.session import now_iso, sha256_file, write_once_json

ARM_NAMES: tuple[str, ...] = ("H1", "H2")

#: The order the two arms run in, and how many times each runs. Frozen here
#: for the same reason the configurations are: an order chosen later is a
#: variable nobody controlled. The machine is warmer for the second arm
#: whichever one it is, so *which* one is second has to be decided before any
#: number exists rather than after.
ARM_ORDER: tuple[str, ...] = ("H1", "H2")
RUNS_PER_ARM = 1

#: One epoch over the frozen pool. Read from the declaration rather than
#: restated, so the row count cannot differ between what is declared and what
#: is run.
ARM_ROWS = hypotheses.ROWS

#: The same stride gate 500 uses, and for the same reason: a checkpoint
#: mid-accumulation saves gradients nothing will apply.
CHECKPOINT_EVERY = 8 * LoraConfig_().grad_accum

FAILURE_SUFFIX = "_failure.json"

ARM_PROVES = (
    "one epoch over the frozen 2,000-row pool",
    "the declared rank, alpha and learning rate and nothing else",
    "a per-row loss, recorded and never judged here",
    "the final trainable tensors, digested",
)


class ArmRefused(RuntimeError):
    """The runner stopped rather than start something it could not vouch for."""


class ArmFailed(RuntimeError):
    """The arm ran and did not produce a measurement anybody can use."""


def expected_optimizer_steps() -> int:
    """How many optimizer steps one arm must take.

    Derived from the frozen row count and the configured accumulation, not
    typed in: 2,000 rows at ``grad_accum`` 8 is 250 steps. A number written
    here by hand would be a second opinion about the accumulation.
    """
    return ARM_ROWS // LoraConfig_().grad_accum


def preceding_arm(name: str) -> str | None:
    """Which arm must have finished before this one may start."""
    index = ARM_ORDER.index(name)
    return ARM_ORDER[index - 1] if index else None


def _adapter_problems(run_dir: Path, arm: str, evidence: dict,
                      cfg: LoraConfig_) -> list[str]:
    """The adapter on disk, against what the evidence says it is.

    Two separate questions, and neither answers the other: the file has to
    hash to what was recorded, and the manifest beside it has to describe the
    shape this arm was fitted with. A rank-32 adapter with a manifest saying
    16 loads and is wrong, which is exactly what ``load_finetuned`` reads that
    manifest to prevent.
    """
    problems: list[str] = []
    saved = evidence.get("adapter") or evidence.get("saved")
    if not isinstance(saved, dict) or not isinstance(saved.get("path"), str):
        return [f"{arm}'s evidence does not say which adapter file it wrote"]

    name = saved["path"]
    if "/" in name or "\\" in name or name in ("", ".", ".."):
        return [f"{arm}'s adapter is named {name[:40]!r}; a path taken from a "
                "file on disk is input, not a constant"]

    blob = run_dir / "adapter" / name
    if not blob.is_file():
        problems.append(f"{arm}'s adapter ({name}) is not on disk, so there is "
                        "nothing to replay from")
    else:
        got = sha256_file(blob)
        if got != saved.get("sha256"):
            problems.append(
                f"{arm}'s adapter hashes to {got[:16]}..., not the "
                f"{str(saved.get('sha256'))[:16]}... its evidence records")

    manifest = run_dir / "adapter" / "brickagain_manifest.json"
    if not manifest.is_file():
        return problems + [
            f"{arm}'s adapter has no brickagain_manifest.json, so nothing "
            "beside the weights says which LoRA shape they are"]
    try:
        body = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return problems + [f"{arm}'s adapter manifest is unreadable"]
    lora = (body or {}).get("lora")
    if not isinstance(lora, dict):
        return problems + [f"{arm}'s adapter manifest records no lora block"]
    want = {"r": cfg.rank, "alpha": cfg.alpha,
            "target_modules": list(cfg.target_modules)}
    for key, value in want.items():
        if lora.get(key) != value:
            problems.append(
                f"{arm}'s adapter manifest says {key}={lora.get(key)!r} where "
                f"{arm} is {value!r}; the weights and the manifest beside them "
                "describe different runs")
    return problems


def _record_problems(run_dir: Path, arm: str, evidence: dict,
                     carried: dict) -> list[str]:
    """The plan and the ledger, against the evidence and against each other."""
    problems: list[str] = []
    plan = gates.read_plan(run_dir)
    if plan is None:
        return [f"{arm} has no readable {gates.PLAN_NAME}, so nothing says "
                "what that run was supposed to be"]
    if plan.get("gate") != arm:
        problems.append(f"{arm}'s plan is for {plan.get('gate')!r}")

    for field, value in carried.items():
        for where, body in (("plan", plan), ("evidence", evidence)):
            got = body.get(field, _MISSING)
            if got is _MISSING:
                problems.append(
                    f"{arm}'s {where} records no {field}, so nothing says "
                    "which one that run was produced under")
            elif got != value:
                problems.append(
                    f"{arm}'s {where} {field} is not the one carried in for "
                    "this run; the two arms would not be comparable")

    if gates.order_digest(plan.get("order") or []) != plan.get("order_digest"):
        problems.append(f"{arm}'s plan order no longer digests to its recorded "
                        "order_digest; the input order has been edited")
    if evidence.get("order_digest") != plan.get("order_digest"):
        problems.append(f"{arm}'s evidence and plan disagree about the input "
                        "order")

    entries = gates.read_ledger(run_dir)
    if not entries:
        return problems + [f"{arm} has no readable {gates.LEDGER_NAME}, so "
                           "the summary cannot be checked against anything"]
    problems += [f"{arm} ledger: {p}" for p in gates.ledger_problems(
        entries, order=plan.get("order") or [], declared_rows=ARM_ROWS,
        checkpoint_positions=[c["position"]
                              for c in gates.read_checkpoints(run_dir)
                              if isinstance(c.get("position"), int)])]

    effective = gates.effective_ledger(entries)
    positions = sorted(effective)
    if evidence.get("sample_ids") != [effective[p].get("sample_id")
                                      for p in positions]:
        problems.append(
            f"{arm}'s recorded sample_ids are not the ones its ledger holds")
    if evidence.get("per_row_loss") != [effective[p].get("loss")
                                        for p in positions]:
        problems.append(
            f"{arm}'s recorded per-row loss is not what its ledger holds")
    return problems


def predecessor_problems(name: str, *, previous_run_dir, expected_pack_digest,
                         expected_dependency_digest, allocator_config,
                         determinism) -> list[str]:
    """Is it this arm's turn, and did the one before it really finish?

    The order is frozen, so H2 is only ever second -- and second means *after
    a complete H1 on this pack*, not merely later in the day. What used to
    stand in for that was H1's own evidence saying
    ``operational_problems: []``: two keys, typed by anyone, naming no pack,
    no dataset, no configuration and no measurement.

    So nothing stored is believed. H1's evidence, plan, ledger and adapter are
    re-read; :func:`operational_problems` is re-run over them; the frozen
    configuration, the row count, the step count and the epoch count are
    checked against the declaration; and every one of the four carried values
    this H2 is being run against has to be the one H1 was produced under. Two
    arms bound to different packs are two experiments, not a comparison.

    The predecessor is named explicitly by the caller. Nothing here goes
    looking for it: a runner that could find its own predecessor could find
    the wrong one.
    """
    if name not in ARM_ORDER:
        return [f"{name!r} is not one of the frozen arms {list(ARM_ORDER)}"]
    previous = preceding_arm(name)
    if previous is None:
        if previous_run_dir is not None:
            return [f"{name} is first in the frozen order {list(ARM_ORDER)} "
                    "and has no predecessor, so it was given a run directory "
                    "for one that does not exist"]
        return []
    if previous_run_dir is None:
        return [f"{name} runs after {previous} and no {previous} run "
                "directory was given. The order is frozen; running the second "
                "arm without the first is not the comparison that was "
                "declared."]

    run_dir = Path(previous_run_dir)
    # Failure markers first, and for either arm: a failed arm stops the
    # sequence, and what to do next is a person's decision rather than this
    # function's.
    marked = [arm for arm in ARM_NAMES if failure_path(run_dir, arm).exists()]
    if marked:
        return [f"{run_dir.name} holds a failure record for {marked}. A "
                "failed arm stops the sequence: the second arm does not start "
                "on the wreckage of the first."]

    path = evidence_path(run_dir, previous)
    if not path.is_file():
        return [f"{run_dir.name} holds no {path.name}, so {previous} has not "
                f"finished and {name} is not due to start"]
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [f"{path.name} is unreadable, so nothing says {previous} "
                "finished"]
    if not isinstance(evidence, dict) or evidence.get("arm") != previous:
        return [f"{path.name} records arm "
                f"{(evidence or {}).get('arm')!r}, not {previous}"]

    # Re-run, never read. A stored empty list is the writer's opinion of its
    # own work.
    problems = [f"{previous}: {p}" for p in operational_problems(evidence)]

    cfg = frozen_config(previous)
    if evidence.get("config") != cfg.as_dict():
        differing = sorted(
            k for k in set(cfg.as_dict()) | set(evidence.get("config") or {})
            if (evidence.get("config") or {}).get(k) != cfg.as_dict().get(k))
        problems.append(
            f"{previous}'s recorded config differs from the frozen one in "
            f"{differing}; whatever ran, it was not the declared arm")
    if evidence.get("epochs") != cfg.epochs:
        problems.append(
            f"{previous} recorded {evidence.get('epochs')!r} epochs, not the "
            f"declared {cfg.epochs}")

    carried = {"pack_digest": expected_pack_digest,
               "dependency_digest": expected_dependency_digest,
               "allocator_config": allocator_config,
               "determinism": determinism}
    problems += _record_problems(run_dir, previous, evidence, carried)
    problems += _adapter_problems(run_dir, previous, evidence, cfg)
    return problems


#: Tells "no such field" apart from "the field is there and is null".
_MISSING = object()


#: Everything a finished arm has to be before its evidence is published.
#: Checked in one place, and checked *before* anything is written, because a
#: file that looks like a measurement and is not one is worse than no file.
def operational_problems(evidence: dict | None) -> list[str]:
    """Why this arm did not produce a usable measurement. Empty means it did.

    ``run_gate`` returns evidence whatever happened -- a cold load that could
    not rebuild the adapter, a row whose loss was NaN, a ledger with a hole in
    it, four hundred rows instead of two thousand. Each of those is a file
    that reads like a result. Deciding here, once, is what stops the deciding
    from being done later by whoever happens to open it.

    Every failing check is reported, not just the first: an arm that went
    wrong in three ways is worth knowing about in three ways, and fixing them
    one boot at a time is expensive.
    """
    e = evidence or {}
    problems: list[str] = []

    # Against the frozen length, never against the length this evidence
    # declares for itself. ``rows_declared`` came out of the same file as
    # ``rows_completed``: reading the target from the thing being measured
    # means a run that declares eight rows and completes eight rows agrees
    # with itself perfectly and is not the arm.
    for field in ("rows_declared", "rows_completed"):
        value = e.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            problems.append(f"{field} is {value!r}, not a count")
        elif value != ARM_ROWS:
            problems.append(
                f"{field} is {value}, not the {ARM_ROWS} rows an arm is. A "
                "short run is not a shorter version of the same measurement.")
    want_rows = ARM_ROWS

    if e.get("losses_finite") is not True:
        problems.append(
            "not every row recorded a finite loss, so the arm measured "
            "something that cannot be compared with anything")

    recorded = e.get("ledger_problems")
    if recorded is None:
        problems.append("ledger_problems was not recorded, so nothing checked "
                        "the row-by-row record")
    elif recorded:
        problems += [f"ledger: {p}" for p in recorded]

    steps = e.get("optimizer_steps")
    want_steps = expected_optimizer_steps()
    if not isinstance(steps, int) or isinstance(steps, bool):
        problems.append(f"optimizer_steps is {steps!r}, not a count")
    elif steps != want_steps:
        problems.append(
            f"optimizer_steps is {steps}, not the {want_steps} that "
            f"{want_rows} rows at grad_accum {LoraConfig_().grad_accum} must "
            "produce")

    saved = e.get("saved")
    saved_sha = saved.get("sha256") if isinstance(saved, dict) else None
    if not isinstance(saved, dict):
        problems.append(
            "no adapter was saved, so there is nothing the cold load could be "
            "checked against and nothing to replay from")
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
            problems.append(
                "the cold load's digest does not match what was saved; the "
                "adapter on disk is not the adapter that was written")
        # Compared here as well, rather than trusting the boolean beside it.
        # ``matches_saved`` is a claim the same writer made; the two digests
        # are the thing the claim is about.
        cold_sha = cold.get("sha256")
        if not isinstance(cold_sha, str) or not cold_sha:
            problems.append(
                "the cold load recorded no digest of what it read, so nothing "
                "says it read the adapter that was saved")
        elif isinstance(saved_sha, str) and cold_sha != saved_sha:
            problems.append(
                f"the cold load read {cold_sha[:16]}... where the adapter was "
                f"saved as {str(saved_sha)[:16]}...")

    problems += [f"the final trainable digest is unusable: {p}"
                 for p in pack.expected_digest_problems(
                     e.get("trainable_digest"), what="trainable digest")]
    return problems


def frozen_config(name: str) -> LoraConfig_:
    """The arm's configuration, from the declaration and from nowhere else."""
    if name not in ARM_NAMES:
        raise ArmRefused(
            f"{name!r} is not one of the frozen arms {list(ARM_NAMES)}. There "
            "are two, they were written down before any number from them "
            "existed, and a third would be a condition nobody declared.")
    return hypotheses.config_for(name)


def shared_seed() -> int:
    """The seed both arms run under, or a refusal.

    The determinism settings have to be applied before an arm is chosen -- the
    preflight and the verifier come first, and applying them afterwards would
    mean the verifier ran under different settings from the run. That is only
    sound because the two arms share a seed, so this asserts it rather than
    assuming it.
    """
    seeds = {name: hypotheses.config_for(name).seed for name in ARM_NAMES}
    if len(set(seeds.values())) != 1:
        raise ArmRefused(
            f"the arms declare different seeds {seeds}. Seed is one of the "
            "fields held identical between them, and determinism is applied "
            "before an arm is chosen; two seeds would make the settings the "
            "verifier ran under depend on a choice made after it.")
    return next(iter(seeds.values()))


def arm_spec(name: str) -> gates.Gate:
    """How long the run is and when it checkpoints. Not a gate."""
    if name not in ARM_NAMES:
        raise ArmRefused(f"{name!r} is not one of {list(ARM_NAMES)}")
    return gates.Gate(name=name, rows=ARM_ROWS,
                      checkpoint_every=CHECKPOINT_EVERY, proves=ARM_PROVES)


def evidence_path(run_dir, arm: str) -> Path:
    return gate_suite.evidence_path(run_dir, arm)


def failure_path(run_dir, arm: str) -> Path:
    return Path(run_dir) / f"{arm}{FAILURE_SUFFIX}"


def write_failure(run_dir, arm: str, reason: str, *, problems=None,
                  measured: dict | None = None) -> Path:
    """Publish why this arm stopped, once, and never rewrite it.

    Write-once like everything else here. A second failure at the same path is
    a second story about one run, and the first one is the one that happened.

    What was measured before it stopped goes in here rather than into an
    ordinary evidence file. The distinction is the point: ``H*_evidence.json``
    means "this arm produced a measurement", and a flag inside it saying
    otherwise is a thing people read past.
    """
    body = {"arm": arm, "reason": _portable(str(reason)),
            "problems": [_portable(str(p)) for p in (problems or [])],
            "written_at": now_iso()}
    if measured is not None:
        body["measured"] = measured
    write_once_json(failure_path(run_dir, arm), body)
    return failure_path(run_dir, arm)


def child_argv(*, arm, run_dir, pack_dir, expected_pack_digest,
               expected_dependency_digest, gate_runs, resume: bool = False,
               data_root=None, previous_run_dir=None) -> list[str]:
    """The command line for one arm, built in one place.

    Built here rather than in the script so that what a parent would spawn and
    what the child can parse are the same list, and a test can show it by
    feeding this straight back into the script's own parser.
    """
    argv = ["--arm", str(arm), "--run-dir", str(run_dir),
            "--pack-dir", str(pack_dir),
            "--expected-pack-digest", str(expected_pack_digest),
            "--expected-dependency-digest", str(expected_dependency_digest)]
    if data_root is not None:
        argv += ["--data-root", str(data_root)]
    for role in gate_suite.ROLES:
        argv += [f"--{role.replace('_', '-')}", str(gate_runs[role])]
    if previous_run_dir is not None:
        argv += ["--previous-arm-run-dir", str(previous_run_dir)]
    if resume:
        argv.append("--resume")
    return argv


def run_arm(name: str, *, deps_factory, run_dir, pack_dir, gate_runs,
            expected_pack_digest, expected_dependency_digest,
            allocator_config, determinism, previous_run_dir=None,
            resume: bool = False, stop_after: int | None = None,
            verifier=None, dependency_checker=None, clock=None) -> dict:
    """Prove the suite, then run one arm. In that order, always.

    ``deps_factory`` is called with the frozen configuration **after** the
    unlock and never before. A loader is a built model; taking one as an
    argument would mean the caller had already spent the boot by the time this
    function decided whether it was allowed to.
    """
    cfg = frozen_config(name)
    run_dir = Path(run_dir)

    ordering = predecessor_problems(
        name, previous_run_dir=previous_run_dir,
        expected_pack_digest=expected_pack_digest,
        expected_dependency_digest=expected_dependency_digest,
        allocator_config=allocator_config, determinism=determinism)
    if ordering:
        raise ArmRefused(f"refusing to run {name}:\n  - "
                         + "\n  - ".join(ordering))

    # A directory that already recorded a failure is not a directory to try
    # again in. The failure is immutable and the run that produced it is part
    # of the record; starting over on top of it would leave one directory
    # describing two runs.
    for arm in ARM_NAMES:
        if failure_path(run_dir, arm).exists():
            raise ArmRefused(
                f"{run_dir.name} already holds a failure record for {arm}. "
                "Failure evidence is immutable and a failed arm stops the "
                "sequence; use a new directory, deliberately, once somebody "
                "has read why the last one stopped.")

    # The gate suite, first, and through the declaration's own lock so that
    # "which runs count" has one definition. Raises HypothesisLocked, listing
    # every reason, before anything below happens.
    unlocked = hypotheses.require_unlocked(
        name, runs=gate_runs,
        expected_pack_digest=expected_pack_digest,
        expected_dependency_digest=expected_dependency_digest,
        allocator_config=allocator_config, determinism=determinism)
    if unlocked.as_dict() != cfg.as_dict():
        raise ArmRefused(
            "the configuration the lock returned is not the one declared for "
            f"{name}; something between the declaration and here changed it")

    if evidence_path(run_dir, name).exists():
        raise ArmRefused(
            f"{run_dir.name} already holds evidence for {name}. Each arm gets "
            "its own directory, written once: reusing one would put two "
            "measurements where a replay expects one.")
    for other in ARM_NAMES:
        if other != name and evidence_path(run_dir, other).exists():
            raise ArmRefused(
                f"{run_dir.name} already holds evidence for {other}. The two "
                "arms do not share a directory -- they do not share a "
                "process, a model, an optimizer or a generator either, and a "
                "shared directory is how that would stop being true.")

    enriched = None
    try:
        # Inside the guard: building the loader *is* building the model, and a
        # model that dies on the way up -- out of memory, a driver that went
        # away -- is a failure of this arm, not an absence of one.
        deps = deps_factory(cfg)
        evidence = gates.run_gate(
            name, deps=deps, run_dir=run_dir, pack_dir=pack_dir,
            expected_pack_digest=expected_pack_digest,
            expected_dependency_digest=expected_dependency_digest,
            allocator_config=allocator_config, determinism=determinism,
            stop_after=stop_after, resume=resume, verifier=verifier,
            dependency_checker=dependency_checker, clock=clock,
            spec=arm_spec(name), config=cfg.as_dict())
        enriched = _enriched(name, cfg, evidence, run_dir)
        problems = operational_problems(enriched)
        if problems:
            raise ArmFailed(
                f"{name} ran and did not produce a usable measurement:\n  - "
                + "\n  - ".join(problems))
        enriched["operational_problems"] = []
    except gates.DeliberateStop:
        # Not a failure: the checkpoint is on disk and --resume continues it.
        raise
    except BaseException as exc:
        write_failure(run_dir, name, f"{type(exc).__name__}: {exc}",
                      problems=(operational_problems(enriched)
                                if enriched is not None else []),
                      measured=enriched)
        raise

    # Published only now, and only once. Everything above had to hold first,
    # so the existence of this file is itself the claim that it did.
    write_once_json(evidence_path(run_dir, name), enriched)
    return enriched


def _enriched(name: str, cfg: LoraConfig_, evidence: dict,
              run_dir: Path) -> dict:
    """The runner's evidence: what a replay needs, and no judgement.

    The per-row losses and sample ids are lifted out of the effective ledger
    rather than accumulated alongside it, so the summary and the row-by-row
    record cannot disagree.
    """
    plan = gates.read_plan(run_dir) or {}
    effective = gates.effective_ledger(gates.read_ledger(run_dir))
    positions = sorted(effective)
    out = dict(evidence)
    out.pop("speed_threshold", None)
    out.pop("speed_threshold_reason", None)
    out.update({
        "arm": name,
        "rows_declared": evidence.get("rows_declared"),
        "config": cfg.as_dict(),
        "epochs": cfg.epochs,
        "order_digest": plan.get("order_digest"),
        "sample_ids": [effective[p].get("sample_id") for p in positions],
        "per_row_loss": [effective[p].get("loss") for p in positions],
        "adapter": evidence.get("saved"),
    })
    return out


def summary() -> dict:
    """What running an arm would require, for a report to embed verbatim."""
    return {
        "arms": {name: hypotheses.config_for(name).as_dict()
                 for name in ARM_NAMES},
        "rows": ARM_ROWS,
        "epochs": hypotheses.EPOCHS,
        "checkpoint_every": CHECKPOINT_EVERY,
        "required_roles": list(gate_suite.ROLES),
        "declares_a_winner": False,
        "why": ("The two arms are measured, never compared here. A runner "
                "that announced a winner would be choosing the success "
                "criterion after seeing the numbers."),
    }
