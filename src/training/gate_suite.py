"""The formal gate suite: six named runs, checked against each other.

Three gates passing is not the claim that unlocks H1 and H2. Each gate answers
one question about one run; what an unlock needs is that **six specific runs**
exist, that they were all produced by the same pack against the same
dependency bytes under the same allocator and the same determinism settings,
and that the ones which are supposed to agree do agree exactly.

The six roles, by name:

``gate_8``
    the machinery exists.
``gate_100_r1`` / ``gate_100_r2`` / ``gate_100_r3``
    three independent runs whose input order, sample ids, hundred per-row
    losses and final trainable tensors are identical. Repeatability is a
    property of a *set* of runs; no single run has it.
``gate_500_resumed``
    stopped on purpose and resumed.
``gate_500_uninterrupted_control``
    the same 500 rows without the interruption. Its job is to be compared
    against, so it must **fail** gate 500's own verdict -- a control that
    passed was interrupted, which makes it not a control.

Three decisions worth stating plainly.

**Roles are supplied, never discovered.** No glob, no directory listing, no
sort, no ``mtime``, no "the latest one". Each of those turns "which runs prove
this" into "which runs happen to be lying around", and the difference matters
most in exactly the case the suite exists for: a directory holding a failed
attempt beside a successful one. The caller says which run plays which part,
and is wrong in a way somebody can read.

**Stored verdicts are re-derived, not believed.** Every evidence file is run
back through :func:`~src.training.gates.gate_problems`, and the recomputed
answer must also match the one stored beside it. A file that says ``passed``
is a file. The same goes for ``ledger_problems`` inside the evidence: the
ledger itself is re-checked against the plan's order, because a run that
reported a clean ledger and recorded a different one is precisely the failure
a stored summary cannot show.

**Agreement is recomputed from the ledgers.** The repeatability claim and the
interrupted-versus-uninterrupted claim are both stated over per-row values, so
they are read from the per-row record. Comparing two evidence summaries would
compare two summaries.

Nothing here imports torch, builds a model, opens a socket or reads the
dataset. It reads JSON that runs already wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.training import gates, pack
from src.training.longrun import measurement_interval_problems
from src.training.lora import LoraConfig_

#: The six, in the order a report should list them. Membership is exact: a
#: role missing, a role repeated and a role nobody declared are all refusals.
ROLES: tuple[str, ...] = (
    "gate_8",
    "gate_100_r1",
    "gate_100_r2",
    "gate_100_r3",
    "gate_500_resumed",
    "gate_500_uninterrupted_control",
)

GATE_100_ROLES: tuple[str, ...] = ("gate_100_r1", "gate_100_r2", "gate_100_r3")
GATE_500_RESUMED = "gate_500_resumed"
GATE_500_CONTROL = "gate_500_uninterrupted_control"

#: Which gate each role plays. The role is not the gate: three roles play
#: ``gate_100`` and two play ``gate_500``, and the evidence has to say the same.
ROLE_GATE: dict[str, str] = {
    "gate_8": "gate_8",
    "gate_100_r1": "gate_100",
    "gate_100_r2": "gate_100",
    "gate_100_r3": "gate_100",
    GATE_500_RESUMED: "gate_500",
    GATE_500_CONTROL: "gate_500",
}

EVIDENCE_SUFFIX = "_evidence.json"

#: What every one of the six must carry, and must carry the *same* value for.
#: Read from the plan and from the evidence, and both compared against the
#: value the caller carried in -- not against each other, which would only
#: establish that the run agreed with itself.
BINDING: tuple[str, ...] = ("pack_digest", "dependency_digest",
                            "allocator_config", "determinism")

#: The evidence fields whose only job is to say the run was interrupted. The
#: control is allowed to fail on these and on nothing else, which is checked
#: by substituting them and requiring the rest to come back clean.
INTERRUPTION_FIELDS: tuple[str, ...] = (
    "stopped_at", "resumed_from", "attempts", "model_state_restored",
    "rng_state_restored", "optimizer_state_restored")

#: Tells "no such field" apart from "the field is there and is null".
_MISSING = object()


def evidence_path(run_dir, gate: str) -> Path:
    """Where a gate's evidence lives. One definition, read and written.

    The node's entry point writes through this and the verifier reads through
    it, so the name cannot drift between the two -- and, because the gate name
    comes from the role, the file is *named* rather than found.
    """
    return Path(run_dir) / f"{gate}{EVIDENCE_SUFFIX}"


def read_evidence(run_dir, gate: str) -> tuple[dict | None, str | None]:
    """``(evidence, stored_verdict)``, or ``(None, None)`` if unreadable."""
    path = evidence_path(run_dir, gate)
    if not path.is_file():
        return None, None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(body, dict) or not isinstance(body.get("evidence"), dict):
        return None, None
    stored = body.get("verdict")
    return body["evidence"], stored if isinstance(stored, str) else None


def re_executed_span(run_dir) -> list[int]:
    """The rows this run measured more than once, from the ledger.

    Computed from the record rather than from ``resumed_from`` and
    ``stopped_at``: those two say what the run *reported*, and the span is
    used to check the run against itself. The verifier then requires the two
    accounts to agree, which is the point of having both.
    """
    attempts: dict[int, set] = {}
    for entry in gates.read_ledger(run_dir):
        if not isinstance(entry, dict):
            continue
        position, attempt = entry.get("position"), entry.get("attempt")
        if isinstance(position, int) and isinstance(attempt, int):
            attempts.setdefault(position, set()).add(attempt)
    return sorted(p for p, seen in attempts.items() if len(seen) > 1)


# ---------------------------------------------------------------------------
# Reading one role
# ---------------------------------------------------------------------------

def _role_problems(role: str, run_dir) -> tuple[dict | None, list[str]]:
    """Everything one role has to be before it can be compared with the others."""
    gate = ROLE_GATE[role]
    run_dir = Path(run_dir)
    def say(text: str) -> str:
        # Every problem carries its role. A suite refusal that does not name
        # which of the six was wrong sends the reader to look at all of them.
        return f"{role}: {text}"

    if not run_dir.is_dir():
        return None, [say(f"{run_dir.name!r} is not a directory")]

    plan = gates.read_plan(run_dir)
    if plan is None:
        return None, [say(f"{gates.PLAN_NAME} is missing or unreadable, so "
                          "nothing says what this run was supposed to be")]
    if plan.get("gate") != gate:
        return None, [say(f"the plan is for {plan.get('gate')!r}, but this "
                          f"role plays {gate!r}")]

    evidence, stored = read_evidence(run_dir, gate)
    if evidence is None:
        return None, [say(f"{evidence_path(run_dir, gate).name} is missing or "
                          "unreadable")]
    if evidence.get("gate") != gate:
        return None, [say(f"the evidence names gate {evidence.get('gate')!r}, "
                          f"but this role plays {gate!r}")]

    problems: list[str] = []

    # The verdict, re-derived. Both directions matter: a stored ``passed`` on
    # failing evidence is the obvious forgery, and a stored ``failed`` on
    # passing evidence means the file was edited too -- either way what is on
    # disk is not what the run produced.
    recomputed = gates.verdict(gate, evidence)
    if stored is None:
        problems.append(say("the evidence file records no verdict"))
    elif stored != recomputed:
        problems.append(say(
            f"the stored verdict is {stored!r} but this evidence recomputes to "
            f"{recomputed!r}; the file has been edited since the run wrote it"))

    # The ledger, re-checked against the plan rather than against the
    # evidence's own summary of it.
    entries = gates.read_ledger(run_dir)
    checkpoints = [c["position"] for c in gates.read_checkpoints(run_dir)
                   if isinstance(c.get("position"), int)]
    ledger_problems = gates.ledger_problems(
        entries, order=plan.get("order") or [],
        declared_rows=gates.GATES[gate].rows,
        checkpoint_positions=checkpoints)
    problems += [say(f"ledger: {p}") for p in ledger_problems]

    return ({"role": role, "gate": gate, "dir": run_dir, "plan": plan,
             "evidence": evidence, "stored_verdict": stored,
             "verdict": recomputed, "entries": entries,
             "effective": gates.effective_ledger(entries)}, problems)


# ---------------------------------------------------------------------------
# Binding: all six against the values the caller carried in
# ---------------------------------------------------------------------------

def _carried_problems(*, expected_pack_digest, expected_dependency_digest,
                      allocator_config, determinism) -> list[str]:
    problems = pack.expected_digest_problems(
        expected_pack_digest, what="pack digest")
    problems += pack.expected_digest_problems(
        expected_dependency_digest, what="dependency digest")
    problems += gates.runtime_problems(allocator_config, determinism)
    return [f"the value carried in for this check is unusable: {p}"
            for p in problems]


def _binding_problems(record, carried: dict) -> list[str]:
    role = record["role"]
    problems = []
    for where, body in (("plan", record["plan"]),
                        ("evidence", record["evidence"])):
        for field in BINDING:
            got = body.get(field, _MISSING)
            if got is _MISSING:
                problems.append(
                    f"{role}: the {where} records no {field}, so nothing says "
                    "which one this run was produced under")
            elif got != carried[field]:
                problems.append(
                    f"{role}: the {where}'s {field} is not the one carried in "
                    "for this check; this run belongs to a different pack, "
                    "dependency set or runtime")
    return problems


def _provenance_for_comparison(provenance: dict) -> dict:
    """Provenance minus the one field that is supposed to differ.

    ``measurement_intervals.max_rows`` is the gate's own row count -- 8 for
    gate 8, 500 for gate 500 -- so requiring it to be identical across the six
    would require the six to be the same gate. It is not ignored: it is pinned
    to the gate's declared size instead, which says more than "the same as the
    others" does.
    """
    out = dict(provenance or {})
    intervals = out.get("measurement_intervals")
    if isinstance(intervals, dict) and "max_rows" in intervals:
        out["measurement_intervals"] = {k: v for k, v in intervals.items()
                                        if k != "max_rows"}
    return out


def _row_count_problems(record) -> list[str]:
    """The one provenance field exempt from the identity check, pinned instead.

    ``measurement_intervals.max_rows`` is left out of the cross-run comparison
    because it is the gate's own row count and is *supposed* to differ. That
    exemption is only safe if the value is required and checked here. Pinning
    it "if it happens to be present" is the same as not pinning it: a run
    whose provenance says nothing then agrees with every other run and with
    every row count, which is precisely the reading the exemption was granted
    on the strength of.

    Both copies are checked. The plan's and the evidence's are two files.
    """
    role, gate = record["role"], record["gate"]
    declared = gates.GATES[gate].rows
    problems = []
    for where, body in (("plan", record["plan"]),
                        ("evidence", record["evidence"])):
        provenance = body.get("provenance")
        if not isinstance(provenance, dict):
            problems.append(f"{role}: the {where} records no provenance object")
            continue
        if "measurement_intervals" not in provenance:
            problems.append(
                f"{role}: the {where}'s provenance records no "
                "measurement_intervals, so it never says how many rows this "
                "run was to measure -- and max_rows is the one field the "
                "cross-run provenance comparison exempts, on the "
                "understanding that it is pinned here instead")
            continue
        problems += [
            f"{role}: in the {where}'s provenance, {p}"
            for p in measurement_interval_problems(
                provenance["measurement_intervals"], declared_rows=declared)]
    return problems


def _provenance_problems(records: dict) -> list[str]:
    problems = []
    for record in records.values():
        problems += _row_count_problems(record)
        plan_prov = record["plan"].get("provenance")
        if not isinstance(plan_prov, dict) or not plan_prov:
            problems.append(f"{record['role']}: the plan records no provenance")
            continue
        if record["evidence"].get("provenance") != plan_prov:
            problems.append(
                f"{record['role']}: the evidence's provenance is not the one "
                "its own plan froze")

    # Compared against gate_8's, because the suite has to agree with itself
    # and one of them has to be the reference. Which one is arbitrary; that it
    # is the same one for every comparison is not.
    reference = records.get("gate_8")
    if reference is None:
        return problems
    want = _provenance_for_comparison(reference["plan"].get("provenance") or {})
    for role in ROLES:
        record = records.get(role)
        if record is None or role == "gate_8":
            continue
        got = _provenance_for_comparison(record["plan"].get("provenance") or {})
        if got != want:
            differing = sorted(k for k in set(want) | set(got)
                               if want.get(k) != got.get(k))
            problems.append(
                f"{role}: the provenance differs from gate_8's in {differing}; "
                "the six runs must have been produced by one machine in one "
                "state, or they are not a suite")
    return problems


# ---------------------------------------------------------------------------
# Agreement between runs
# ---------------------------------------------------------------------------

def expected_optimizer_steps(gate: str) -> int:
    """How many times this gate's rows must step the optimizer.

    Derived from the gate's declared length and the accumulation the project
    actually configures, not typed in. 500 rows at ``grad_accum`` 8 is 62
    steps; a number written here by hand would be a second opinion about the
    accumulation, and the day it changed one of the two would be wrong.
    """
    return gates.GATES[gate].rows // LoraConfig_().grad_accum


def _optimizer_step_problems(record) -> list[str]:
    """A count, not merely an agreement.

    Requiring only that the two gate 500 runs report the same number accepts
    any number they both report. A shared bug produces exactly that: two runs,
    one wrong answer, perfect agreement. So each is checked against what 500
    rows at this accumulation must produce.
    """
    role, gate = record["role"], record["gate"]
    want = expected_optimizer_steps(gate)
    got = record["evidence"].get("optimizer_steps", _MISSING)
    if got is _MISSING:
        return [f"{role}: the evidence records no optimizer_steps"]
    if isinstance(got, bool) or not isinstance(got, int):
        return [f"{role}: optimizer_steps is {got!r}, a "
                f"{type(got).__name__} rather than a whole number of steps"]
    if got != want:
        return [f"{role}: optimizer_steps is {got}, not the {want} that "
                f"{gates.GATES[gate].rows} rows at grad_accum "
                f"{LoraConfig_().grad_accum} must produce"]
    return []


def _series(record) -> dict:
    """The per-row record, as three lists, from the effective ledger."""
    effective = record["effective"]
    positions = sorted(effective)
    return {
        "positions": positions,
        "order": [effective[p].get("index") for p in positions],
        "sample_ids": [effective[p].get("sample_id") for p in positions],
        "losses": [effective[p].get("loss") for p in positions],
    }


def _agreement_problems(records: dict, roles, *, rows: int,
                        what: str) -> list[str]:
    """Order, sample ids, per-row losses and final weights, all exact."""
    present = [records[r] for r in roles if r in records]
    if len(present) < 2:
        return []
    problems = []
    series = {record["role"]: _series(record) for record in present}
    for role, s in series.items():
        if s["positions"] != list(range(1, rows + 1)):
            problems.append(
                f"{role}: the effective ledger covers "
                f"{len(s['positions'])} rows, not the {rows} {what} compares")
    if problems:
        return problems

    reference = present[0]["role"]
    for record in present[1:]:
        role = record["role"]
        for key, label in (("order", "input order"),
                           ("sample_ids", "sample_id"),
                           ("losses", "per-row loss")):
            if series[role][key] != series[reference][key]:
                first = next(
                    (i + 1 for i, (a, b) in enumerate(
                        zip(series[reference][key], series[role][key]))
                     if a != b), None)
                problems.append(
                    f"{what}: {role} and {reference} differ in {label}, first "
                    f"at row {first}. These runs are required to be identical "
                    "row by row; a difference here is the finding, not a "
                    "tolerance to widen.")

    digests = {record["role"]: record["evidence"].get("trainable_digest")
               for record in present}
    if len(set(digests.values())) != 1:
        problems.append(
            f"{what}: the final trainable_digest differs across "
            f"{sorted(digests)}; identical per-row losses can still end on "
            "different weights, which is why this is checked separately")
    for role, digest in digests.items():
        problems += [f"{role}: {p}" for p in pack.expected_digest_problems(
            digest, what="trainable digest")]
    return problems


def _control_problems(records: dict) -> list[str]:
    """The control must have been uninterrupted, and must fail only for that."""
    record = records.get(GATE_500_CONTROL)
    if record is None:
        return []
    role, evidence = record["role"], record["evidence"]
    problems = []

    expected = {"stopped_at": None, "resumed_from": None, "attempts": 1,
                "model_state_restored": False, "rng_state_restored": False,
                "optimizer_state_restored": False}
    for field, want in expected.items():
        if evidence.get(field, _MISSING) != want:
            problems.append(
                f"{role}: {field} is {evidence.get(field, _MISSING)!r}, not "
                f"{want!r}. The control's whole value is that it was never "
                "interrupted; one that was is a second interrupted run wearing "
                "the control's name.")
    if record["verdict"] != "failed":
        problems.append(
            f"{role}: this run's gate_500 verdict is {record['verdict']!r}. "
            "The control must fail gate 500 -- the gate asks whether a run "
            "survives interruption, and this one was never interrupted. A "
            "control that passes it passed by being something else.")

    # Why it failed. Substitute the six fields that exist only to record an
    # interruption and recompute: whatever is left is a reason that has
    # nothing to do with being a control, and no such reason is acceptable.
    as_if = dict(evidence)
    as_if.update({"stopped_at": 1, "resumed_from": 1, "attempts": 2,
                  "model_state_restored": True, "rng_state_restored": True,
                  "optimizer_state_restored": True})
    remaining = gates.gate_problems("gate_500", as_if)
    problems += [
        f"{role}: this run fails gate 500 for a reason other than never "
        f"having been interrupted -- {p}" for p in remaining]
    return problems


def _resumed_problems(records: dict) -> list[str]:
    """The resume, checked against the run's own record of it.

    Two accounts of the same interruption have to agree: the evidence says
    where it resumed from and where it stopped, and the ledger says which rows
    were measured twice. The interval between them is the only part of the run
    where the restore can be seen at all -- attempt 2 walked into those rows
    from the checkpoint, attempt 1 walked into them from having just trained
    them, and if the weights, the optimizer and the generator all came back
    the two produce identical losses.

    Every row of that interval must be present in **both** attempts. Comparing
    only the rows that happen to appear on both sides makes an interval with
    one side missing compare cleanly over nothing.
    """
    record = records.get(GATE_500_RESUMED)
    if record is None:
        return []
    role, evidence = record["role"], record["evidence"]
    problems = []
    if record["verdict"] != "passed":
        problems.append(
            f"{role}: this run's gate_500 verdict is {record['verdict']!r}:\n"
            + "\n".join(f"      - {p}" for p in
                        gates.gate_problems("gate_500", evidence)))

    # Exactly two attempts, and both accounts of that must say two. A third
    # attempt is not more evidence: it means the run was interrupted again,
    # and which pair of attempts the re-run interval belongs to stops being a
    # question with one answer.
    attempts = {e["attempt"] for e in record["entries"]
                if isinstance(e, dict) and isinstance(e.get("attempt"), int)
                and not isinstance(e.get("attempt"), bool)}
    if attempts != {1, 2}:
        problems.append(
            f"{role}: the ledger records attempts {sorted(attempts)}; the "
            "resumed gate 500 is one stop and one resume, so it is exactly "
            "attempts 1 and 2. Anything else is a different run shape than "
            "the one this role stands for.")
        return problems
    declared = evidence.get("attempts", _MISSING)
    if isinstance(declared, bool) or not isinstance(declared, int) \
            or declared != 2:
        problems.append(
            f"{role}: the evidence says attempts is {declared!r}, but the "
            "ledger records two. The two accounts of the same run disagree.")
        return problems

    span = re_executed_span(record["dir"])
    resumed_from = evidence.get("resumed_from")
    stopped_at = evidence.get("stopped_at")
    for name, value in (("resumed_from", resumed_from),
                        ("stopped_at", stopped_at)):
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(
                f"{role}: the evidence's {name} is {value!r}, so there is no "
                "interval to check the resume over")
    if problems:
        return problems

    want = list(range(resumed_from + 1, stopped_at + 1))
    if not want:
        problems.append(
            f"{role}: the run reports resuming from {resumed_from} and "
            f"stopping at {stopped_at}, which is no rows at all; nothing was "
            "re-executed, so the resume has not been demonstrated")
        return problems
    if span != want:
        problems.append(
            f"{role}: the ledger shows "
            f"{('rows ' + str(span[0]) + '..' + str(span[-1])) if span else 'no row'} "
            f"measured more than once, but the run reports resuming from "
            f"{resumed_from} and stopping at {stopped_at}, which is rows "
            f"{want[0]}..{want[-1]}")
        return problems

    by_attempt: dict[int, dict[int, dict]] = {}
    for entry in record["entries"]:
        if isinstance(entry, dict) and isinstance(entry.get("attempt"), int) \
                and isinstance(entry.get("position"), int):
            by_attempt.setdefault(entry["attempt"], {})[entry["position"]] = entry
    first, second = by_attempt.get(1, {}), by_attempt.get(2, {})

    uncovered = [p for p in want if p not in first or p not in second]
    if uncovered:
        problems.append(
            f"{role}: rows {uncovered[0]}..{uncovered[-1]} "
            f"({len(uncovered)} of {len(want)}) of the re-executed interval "
            "are missing from one of the two attempts. A row only one attempt "
            "measured says nothing about whether the resume put the run back "
            "where it stopped, and skipping it would compare the interval "
            "over whatever is left.")
        return problems

    differing = [p for p in want
                 if first[p].get("loss") != second[p].get("loss")]
    if differing:
        problems.append(
            f"{role}: rows {differing[0]}..{differing[-1]} "
            f"({len(differing)} of {len(want)}) were re-executed after the "
            "resume and measured a different loss than the attempt that was "
            "discarded. The resume did not put the run back where it stopped.")
    return problems


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------

def _shape_problems(runs) -> list[str]:
    if not isinstance(runs, dict):
        return [f"the runs argument is a {type(runs).__name__}, not a mapping "
                "of role to run directory"]
    problems = []
    for role in ROLES:
        if role not in runs:
            problems.append(
                f"{role}: no run directory was supplied for this role. The "
                "six roles are named, not discovered; a suite missing one is "
                "not a suite with five.")
    for role in sorted(runs):
        if role not in ROLE_GATE:
            problems.append(
                f"{role}: is not one of the six roles {list(ROLES)}")

    resolved: dict[str, str] = {}
    for role in ROLES:
        if role not in runs:
            continue
        try:
            key = str(Path(runs[role]).resolve())
        except (OSError, TypeError, ValueError):
            problems.append(f"{role}: {runs[role]!r} is not a usable path")
            continue
        if key in resolved:
            problems.append(
                f"{resolved[key]} and {role} were given the same directory. "
                "One run cannot play two roles: the agreement checks would "
                "compare it with itself and pass by construction.")
        else:
            resolved[key] = role
    return problems


def suite_problems(runs, *, expected_pack_digest, expected_dependency_digest,
                   allocator_config, determinism) -> list[str]:
    """Everything that stops these six runs from unlocking an arm.

    Empty means the suite is proved. Deliberately a list of sentences rather
    than a boolean: "not unlocked" is not actionable, and "gate_100_r2 and
    gate_100_r1 differ in per-row loss, first at row 44" is.

    The four carried values have no defaults. Each is something the build
    machine printed or the operator exported, brought here by a route the runs
    did not travel; a check whose reference value can be omitted is a check
    that gets omitted.
    """
    carried = {"pack_digest": expected_pack_digest,
               "dependency_digest": expected_dependency_digest,
               "allocator_config": allocator_config,
               "determinism": determinism}
    problems = _carried_problems(
        expected_pack_digest=expected_pack_digest,
        expected_dependency_digest=expected_dependency_digest,
        allocator_config=allocator_config, determinism=determinism)
    problems += _shape_problems(runs)
    if problems:
        return problems

    records: dict[str, dict] = {}
    for role in ROLES:
        record, found = _role_problems(role, runs[role])
        problems += found
        if record is not None:
            records[role] = record
    if problems:
        return problems

    for role in ROLES:
        problems += _binding_problems(records[role], carried)
    problems += _provenance_problems(records)

    for role in ROLES:
        record = records[role]
        if role == GATE_500_CONTROL:
            continue          # judged by _control_problems, which knows why
        if record["verdict"] != "passed":
            problems.append(
                f"{role}: this run's {record['gate']} verdict is "
                f"{record['verdict']!r}:\n"
                + "\n".join(f"      - {p}" for p in gates.gate_problems(
                    record["gate"], record["evidence"])))

    problems += _agreement_problems(
        records, GATE_100_ROLES, rows=gates.GATES["gate_100"].rows,
        what="the gate_100 repeatability criterion")
    problems += _agreement_problems(
        records, (GATE_500_RESUMED, GATE_500_CONTROL),
        rows=gates.GATES["gate_500"].rows,
        what="the interrupted-versus-uninterrupted comparison")

    for role in (GATE_500_RESUMED, GATE_500_CONTROL):
        if role in records:
            problems += _optimizer_step_problems(records[role])

    problems += _control_problems(records)
    problems += _resumed_problems(records)
    return problems


def summary() -> dict:
    """What the suite requires, for a report to embed verbatim."""
    return {
        "roles": list(ROLES),
        "role_gate": dict(ROLE_GATE),
        "binding": list(BINDING),
        "evidence_file": f"<gate>{EVIDENCE_SUFFIX}",
    }
