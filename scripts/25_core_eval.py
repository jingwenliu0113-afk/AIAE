#!/usr/bin/env python3
"""Core acceptance for arms B, C, D and E: one contract, four modes.

The contract itself is :mod:`src.eval.acceptance`, frozen before any number
from this run exists. This file is the way to reach it, and the modes
correspond to the two machines the work is split across.

``--contract``
    Print the frozen contract and stop. Reads nothing, writes nothing, runs
    anywhere.

``--materialize`` -- **Mac only**
    Open the test split once, turn 20 whole pairs into 160 cases and write the
    plan. Guarded three ways: an explicit
    ``--open-test-after-codex-approval`` flag, the file's SHA-256 checked
    against the value the contract pins, and a write-once destination. It
    prints the case count and the plan digest and nothing else -- a
    materialiser that echoed a caption would be a materialiser that leaked one
    into a terminal log.

``--run`` -- **execution node, WSL2 with CUDA only**
    Decode one *step* of the frozen schedule. The step number is the whole
    order: step 0 must be sealed before step 1 starts, a sealed step cannot be
    run again, and a step whose cells partly exist needs ``--resume``. Before
    anything loads it runs the node preflight against the two digests the
    operator carried by hand, checks the adapter against the digests in the
    plan, and warms the device on a caption that is not in the test split.
    Then, before the first cell, it writes an immutable attempt record that
    every cell it goes on to write will name; when the step is complete it
    writes the completion record listing every attempt. A step whose cells
    are complete but which was never sealed is sealed without decoding
    anything. It parses nothing and scores nothing, and there is no CPU or
    MPS fallback. Each step is its own process and loads the model from cold.

``--verify`` / ``--score`` -- **Mac only**
    Both require ``--evidence`` and both carried digests; neither has a path
    that skips them. ``--verify`` says whether the results are the grid the
    plan predetermined and whether the attempt-and-completion chain holds;
    ``--score`` additionally re-derives the scorer source manifest and refuses
    unless it is the one the plan was approved with, then applies the
    deterministic checks and writes the per-arm summaries and both paired
    contrasts.

Usage::

  ./.venv/bin/python scripts/25_core_eval.py --contract
  ./.venv/bin/python scripts/25_core_eval.py --materialize \\
      --out gpu_plans/core_eval_plan.json --open-test-after-codex-approval
  python scripts/25_core_eval.py --run --step 0 --plan PLAN --results FILE \\
      --evidence DIR --expected-pack-digest SHA \\
      --expected-dependency-digest SHA
  ./.venv/bin/python scripts/25_core_eval.py --verify --plan PLAN \\
      --results FILE --evidence DIR --expected-pack-digest SHA \\
      --expected-dependency-digest SHA
  ./.venv/bin/python scripts/25_core_eval.py --score --plan PLAN \\
      --results FILE --evidence DIR --expected-pack-digest SHA \\
      --expected-dependency-digest SHA --out FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import acceptance  # noqa: E402
from src.eval.acceptance import (ARM_ORDER, N_STEPS, SETTINGS,  # noqa: E402
                                 PlanRefused, ResultsRefused)

#: Without this the materialiser does nothing. The test split is opened once
#: in the life of this project and the flag is how that becomes a decision
#: somebody made rather than a default somebody inherited.
TEST_GUARD = "--open-test-after-codex-approval"


def _refuse(problems, headline: str) -> int:
    print(headline, file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 2


def _mac_guard(mode: str) -> list[str]:
    return acceptance.mac_only_problems(mode)


def _dedupe(problems) -> list:
    """Same sentence from two validators is one problem, not two."""
    seen, out = set(), []
    for problem in problems:
        if problem in seen:
            continue
        seen.add(problem)
        out.append(problem)
    return out


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def mode_contract() -> int:
    document = acceptance.contract_document(ROOT)
    print(json.dumps(document, indent=2, ensure_ascii=False))
    print(f"\ncontract_digest: {acceptance.contract_digest()}")
    print("scorer_source_manifest_digest: "
          f"{document['scoring']['source_manifest_digest']}")
    return 0


#: The three plan fields a reissue is allowed to touch. Everything else --
#: the source, the cases, the prompts, the inventories, the arms, the
#: settings, the final model, the schedule, the seeds, K and the contract
#: digest -- is compared field by field and must come through untouched.
REISSUABLE: tuple[str, ...] = ("scorer_source_manifest",
                               "scorer_source_manifest_digest", "plan_digest")


def _reissue_mac_guard() -> list[str]:
    """A reissue writes a plan, so it belongs on the machine plans come from.

    ``acceptance.mac_only_problems`` is not reused here on purpose: its mode
    list is part of the frozen contract's vocabulary, and widening it to admit
    a repair mode would edit that vocabulary to make a repair convenient.
    """
    import platform

    system = platform.system()
    if system != acceptance.MAC_SYSTEM:
        return [f"--reissue-plan runs on the Mac only, and this is "
                f"{system!r}. A plan is issued where the test split lives, "
                "even when the reissue does not open it."]
    return []


def mode_reissue_plan(args) -> int:
    """Re-issue an existing plan against the current scorer source. Mac only.

    The plan pins the scorer it will be measured against, so a fix to any
    scorer module invalidates a materialised plan -- by design. The blunt
    remedy is to materialise again, and materialising opens the test split.
    This mode is the narrow alternative: it reads the plan already on disk,
    swaps only the scorer manifest and the digests that follow from it, and
    proves every other field came through identical. **It never opens the
    test split**, because it never needs a case it does not already have.

    Fail-closed at every step. The caller must state, from a source other
    than the file, both the digest and the SHA-256 of the plan being
    reissued: the digest says it is the plan the contract knows, the SHA-256
    says the bytes on disk are the ones that digest was taken over. Either
    missing, either wrong, any frozen field moving, or the output already
    existing, and nothing is written.
    """
    problems = _reissue_mac_guard()
    if problems:
        return _refuse(problems, "refusing to reissue here:")

    if not (args.plan and args.out):
        print("--reissue-plan needs --plan (the plan to reissue) and --out "
              "(a new path; the plan in place is never overwritten)",
              file=sys.stderr)
        return 2
    if not (args.expected_plan_digest and args.expected_plan_sha256):
        print("--reissue-plan needs --expected-plan-digest and "
              "--expected-plan-sha256, both carried from outside the file. A "
              "plan that authenticates itself authenticates any plan put in "
              "its place.", file=sys.stderr)
        return 2

    source, out = Path(args.plan), Path(args.out)
    if out.exists():
        print(f"refusing to reissue onto {out}, which already exists",
              file=sys.stderr)
        return 2
    if out.resolve() == source.resolve():
        print("refusing to reissue a plan onto itself", file=sys.stderr)
        return 2
    if not source.is_file():
        print(f"there is no plan at {source}", file=sys.stderr)
        return 2

    raw = source.read_bytes()
    got_sha = hashlib.sha256(raw).hexdigest()
    if got_sha != args.expected_plan_sha256:
        return _refuse(
            [f"{source} hashes to {got_sha}, and the SHA-256 carried in was "
             f"{args.expected_plan_sha256}"],
            "refusing to reissue these bytes:")

    try:
        old = acceptance.read_plan(source)
    except PlanRefused as exc:
        print(f"refusing to reissue a plan that does not verify: {exc}",
              file=sys.stderr)
        return 2

    checks = []
    if old.get("plan_digest") != args.expected_plan_digest:
        checks.append(f"the plan records plan_digest "
                      f"{old.get('plan_digest')!r} and the value carried in "
                      f"was {args.expected_plan_digest!r}")
    if acceptance.plan_digest(old) != old.get("plan_digest"):
        checks.append("the plan's recorded plan_digest is not the digest of "
                      "its own contents")
    if checks:
        return _refuse(checks, "refusing to reissue this plan:")

    new = dict(old)
    new["scorer_source_manifest"] = acceptance.scorer_manifest(ROOT)
    new["scorer_source_manifest_digest"] = acceptance.scorer_manifest_digest(ROOT)
    new["plan_digest"] = acceptance.plan_digest(new)

    # Field by field, both directions, over the whole plan. Not a spot check
    # of the fields thought about while writing this.
    moved = sorted((set(old) | set(new)) - set(REISSUABLE))
    drifted = [f for f in moved if old.get(f) != new.get(f)]
    if drifted:
        return _refuse([f"{f} differs between the plan and its reissue"
                        for f in drifted],
                       "refusing to reissue: a frozen field moved:")
    if set(old) != set(new):
        return _refuse(["the reissue does not carry the same field names"],
                       "refusing to reissue:")

    problems = acceptance.plan_problems(new)
    if problems:
        return _refuse(problems, "the reissued plan does not verify:")

    acceptance.write_plan(out, new)
    print(json.dumps({
        "reissued_from": str(source),
        "out": str(out),
        "unchanged_fields": len(moved),
        "old_plan_digest": old["plan_digest"],
        "new_plan_digest": new["plan_digest"],
        "old_scorer_source_manifest_digest":
            old["scorer_source_manifest_digest"],
        "new_scorer_source_manifest_digest":
            new["scorer_source_manifest_digest"],
        "contract_digest": new["contract_digest"],
        "cases": len(new["cases"]),
        "test_split_opened": False,
    }, indent=2))
    return 0


def mode_materialize(args) -> int:
    problems = _mac_guard("materialize")
    if problems:
        return _refuse(problems, "refusing to materialise here:")
    if not args.open_test_after_codex_approval:
        print(f"refusing to open {acceptance.TEST_FILE} without {TEST_GUARD}. "
              "The test split is opened once, deliberately, after review.",
              file=sys.stderr)
        return 2
    out = Path(args.out or (ROOT / acceptance.PLAN_PATH))
    try:
        body = acceptance.materialize_plan(ROOT)
        acceptance.write_plan(out, body)
    except PlanRefused as exc:
        print(f"refusing to materialise the plan: {exc}", file=sys.stderr)
        return 2
    # Count and digest only. Nothing from a case is printed.
    print(json.dumps({"cases": len(body["cases"]),
                      "pairs": body["source"]["pairs"],
                      "plan_digest": body["plan_digest"],
                      "contract_digest": body["contract_digest"]}, indent=2))
    return 0


def _load_plan(args):
    """The plan, validated, or ``None`` when the caller did not name one."""
    if not args.plan:
        return None
    return acceptance.read_plan(Path(args.plan))


def _carried_digest_problems(args) -> list[str]:
    """Both values, or neither stage runs. No defaults, ever."""
    from src.training import pack as pack_module

    return ([f"--expected-pack-digest: {p}" for p in
             pack_module.expected_digest_problems(args.expected_pack_digest)]
            + [f"--expected-dependency-digest: {p}" for p in
               pack_module.expected_digest_problems(
                   args.expected_dependency_digest,
                   what="dependency digest")])


def mode_run(args) -> int:
    import torch

    from src.training import gpu_node
    from src.training.session import now_iso, write_once_json

    probe_reading = gpu_node.probe()
    problems = acceptance.node_only_problems("run", probe_reading)
    if problems:
        return _refuse(problems, "refusing to run here:")

    if args.step is None:
        print("--step is required; the schedule is frozen and the step number "
              "is the order", file=sys.stderr)
        return 2
    plan = _load_plan(args)
    if plan is None or not args.results or not args.evidence:
        print("--plan, --results and --evidence are all required",
              file=sys.stderr)
        return 2
    problems = _carried_digest_problems(args)
    if problems:
        return _refuse(problems, "refusing to run without both carried "
                                 "digests:")

    results = Path(args.results)
    evidence = Path(args.evidence)
    evidence.mkdir(parents=True, exist_ok=True)

    # Before anything else touches the file. ``step_problems`` reaches
    # ``known_keys`` and so ``read_cells``, which raises on a line that was
    # half written -- exactly what a process killed mid-append leaves behind.
    # The traceback that produced came after the arguments were accepted and
    # said nothing about what to do; this says it, and stops.
    problems = acceptance.results_problems(results)
    if problems:
        return _refuse(problems, "refusing to read these results:")

    problems = acceptance.step_problems(results, plan, args.step,
                                        resume=args.resume,
                                        evidence_dir=evidence)
    if problems:
        return _refuse(problems, "refusing to take this step:")

    # Every finished predecessor, replayed in full -- not "its completion
    # file is present", which is what this used to be. A predecessor whose
    # attempts contradict its completion, or whose preflight failed, is not
    # a finished predecessor, and step N would be measured on top of it.
    # Before the preflight, the model and the warm-up, so that noticing costs
    # nothing.
    problems = acceptance.predecessor_problems(
        evidence, plan, args.step, results_path=results,
        expected_pack_digest=args.expected_pack_digest,
        expected_dependency_digest=args.expected_dependency_digest)
    if problems:
        return _refuse(problems[:20],
                       "refusing to start: an earlier step's evidence does "
                       "not hold:")

    group, name = acceptance.step(args.step)
    state = acceptance.step_state(results, evidence, plan, args.step)

    # The cells are all there and only the closing record is missing. Read,
    # check, write the completion -- and decode nothing. A step in this state
    # is a step whose last attempt died between its final cell and its seal,
    # and re-measuring 320 cells to recover a bookkeeping file would throw
    # away the measurements it was meant to protect.
    if state == acceptance.STEP_UNSEALED:
        return _seal(args, plan, results, evidence, args.step,
                     decoded_here=0)

    preflight = gpu_node.preflight(
        probe=probe_reading, pack_dir=args.pack_dir,
        expected_pack_digest=args.expected_pack_digest,
        expected_dependency_digest=args.expected_dependency_digest)
    if not preflight["passed"]:
        return _refuse(
            [f"{k}: {preflight['checks'][k]['detail']}"
             for k in preflight["failed"]],
            "the node preflight did not pass, so nothing here starts:")

    try:
        device = acceptance.resolve_device(torch)
    except acceptance.DeviceRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2

    adapter_dir = None
    if acceptance.ARMS[name].model != acceptance.PUBLIC_MODEL:
        if not args.adapter_dir:
            print(f"step {args.step} runs {acceptance.FINAL_MODEL} and needs "
                  "--adapter-dir; the weights are not in the pack",
                  file=sys.stderr)
            return 2
        adapter_dir = Path(args.adapter_dir)
        problems = acceptance.plan_final_adapter_problems(plan, adapter_dir)
        if problems:
            return _refuse(problems, "refusing to load these weights:")

    todo = [c for c in acceptance.step_cells(plan, args.step)
            if c not in acceptance.known_keys(results)]
    interface, info = acceptance.build_interface(
        name, device=device, adapter_dir=adapter_dir)

    print(f"step {args.step} ({group}/{name}): {len(todo)} cells on {device}, "
          f"plan {plan['plan_digest'][:16]}...", flush=True)
    warmup = acceptance.warm_up(interface, name)
    print(f"  warm-up: {warmup['generations']} generations, excluded",
          flush=True)

    # Written here: after the preflight and the warm-up, before the first
    # cell. An attempt that dies at cell 200 then leaves 200 cells that each
    # name a record saying exactly which machine, pack and preflight produced
    # them -- which is what the single end-of-step record could never do.
    attempt_index = acceptance.next_attempt_index(evidence, args.step)
    attempt = acceptance.build_attempt_evidence(
        index=args.step, attempt_index=attempt_index, plan=plan,
        probe_reading=probe_reading, preflight_result=preflight,
        pack_digest=args.expected_pack_digest,
        dependency_digest=args.expected_dependency_digest,
        warmup=warmup, adapter_dir=adapter_dir,
        cells_missing_at_start=len(todo), started_at=now_iso())
    write_once_json(acceptance.attempt_path(evidence, args.step,
                                            attempt_index), attempt)
    digest = acceptance.attempt_digest(attempt)
    print(f"  attempt {attempt['attempt_id']} ({digest[:16]}...)", flush=True)

    cases = acceptance.case_index(plan)
    known = acceptance.known_keys(results)
    written = 0
    for i, (_digest, case_id, _arm, seed) in enumerate(todo, 1):
        row = acceptance.run_case(interface, cases[case_id], name, seed,
                                  plan_digest_value=plan["plan_digest"],
                                  step_index=args.step, group=group,
                                  attempt_id=attempt["attempt_id"],
                                  attempt_digest=digest)
        try:
            acceptance.append_cell(results, row, known=known)
        except ResultsRefused as exc:
            print(f"stopping: {exc}", file=sys.stderr)
            return 2
        written += 1
        if i % 40 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} cells", flush=True)

    return _seal(args, plan, results, evidence, args.step,
                 decoded_here=written, load_order=info.get("load_order"))


def _seal(args, plan, results, evidence, index: int, *, decoded_here: int,
          load_order=None) -> int:
    """Close a step: check the whole chain, then write the record.

    The check comes first and is the same one ``--verify`` runs, minus the
    completion this is about to create. Sealing used to check almost nothing
    -- it read the attempt files, called ``attempt_reference`` on whatever
    came back, and wrote the record. A file that parsed and was missing a
    field produced a ``KeyError`` after the cells were measured; a failed
    preflight or an adapter that should not have been there produced a
    completion that vouched for both.

    Derived from what is on disk rather than from what this process happens
    to remember, so the same code closes a step that ran in one go and a step
    that took three attempts across two days.
    """
    from src.training.session import now_iso, write_once_json

    group, name = acceptance.step(index)

    problems = acceptance.step_chain_problems(
        evidence, plan, index, results_path=results,
        expected_pack_digest=args.expected_pack_digest,
        expected_dependency_digest=args.expected_dependency_digest,
        require_completion=False)
    if problems:
        return _refuse(problems[:20], "refusing to seal this step:")

    rows = acceptance.read_cells(results)
    mine = [r for r in rows if r.get("step_index") == index
            and r.get("arm") == name]
    counted: dict = {}
    for row in mine:
        counted[row.get("attempt_id")] = counted.get(row.get("attempt_id"),
                                                     0) + 1

    references = []
    for path in acceptance.existing_attempts(evidence, index):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:        # already reported above
            return _refuse([f"{path.name} is not readable ({exc})"],
                           "refusing to seal this step:")
        references.append(acceptance.attempt_reference(
            body, counted.get(body.get("attempt_id"), 0)))

    completion = acceptance.build_step_completion(
        index=index, plan=plan, attempts=references, cells_recorded=len(mine),
        sealed_at=now_iso(), sealed_without_decoding=decoded_here == 0)
    problems = acceptance.completion_problems(completion, plan, index=index)
    if problems:
        return _refuse(problems, "refusing to seal this step:")
    write_once_json(acceptance.completion_path(evidence, index), completion)

    print(json.dumps({"step": index, "group": group, "arm": name,
                      "cells_decoded_here": decoded_here,
                      "cells_recorded": completion["cells_recorded"],
                      "attempts": [r["attempt_id"] for r in references],
                      "sealed_without_decoding":
                          completion["sealed_without_decoding"],
                      "load_order": load_order}, indent=2))
    return 0


def _log_dir_problems(log_dir, pack_dir) -> list[str]:
    """Logs live outside the pack, and the pack says why.

    A file written inside the pack is a file the manifest does not list, and
    the node's own preflight refuses a pack that grew one. That is not a
    quibble -- it already happened: the first attempt at a continuous run put
    ``logs/step0.log`` under the pack root and the very next preflight
    stopped the run, correctly.
    """
    if not log_dir:
        return ["--run-all needs --log-dir, and it has no default: a default "
                "would eventually be inside the pack"]
    log, pack = Path(log_dir).resolve(), Path(pack_dir or ".").resolve()
    if log == pack or pack in log.parents:
        return [f"{log} is inside the pack at {pack}. Logs go outside it: a "
                "pack that grows an unmanifested file stops verifying, and "
                "the preflight that notices is the one guarding the run."]
    return []


def _step_argv(args, index: int, *, resume: bool) -> list[str]:
    """The official single-step command line, built from the frozen schedule.

    The adapter is decided here and only here. ``C`` and ``E`` are the
    fine-tuned arms and must carry it; ``B`` and ``D`` are the published
    model and must not, because a public arm that quietly loaded final_H2
    would produce a B - C of zero and look like a result.
    """
    _group, arm = acceptance.step(index)
    argv = [sys.executable, str(Path(__file__).resolve()),
            "--run", "--step", str(index),
            "--plan", str(args.plan or (ROOT / acceptance.PLAN_PATH)),
            "--results", str(args.results),
            "--evidence", str(args.evidence),
            "--pack-dir", str(args.pack_dir),
            "--expected-pack-digest", args.expected_pack_digest,
            "--expected-dependency-digest", args.expected_dependency_digest]
    if acceptance.ARMS[arm].model != acceptance.PUBLIC_MODEL:
        argv += ["--adapter-dir", args.adapter_dir]
    if resume:
        argv.append("--resume")
    return argv


def mode_run_all(args) -> int:
    """Steps 0 to 7, in the frozen order, on the node. WSL2 CUDA only.

    This exists because the alternative was a driver on the Mac holding an
    ssh connection open for six hours, and a laptop lid is not a scheduling
    primitive: the previous batch stalled for seven hours when the process
    watching it was culled, and stopped again when a step's ssh died. Here
    the loop lives beside the runner it drives, inside the pack, digested by
    the manifest like everything else.

    It adds no validation of its own. Each step is the official
    ``--run --step N`` in its own process -- its own cold load, its own
    preflight, its own predecessor replay -- and what this mode checks after
    it is what the pack already knows how to check:
    ``step_chain_problems(require_completion=True)``, the same function the
    sealer and ``--verify`` use. A driver that scored a step with its own
    idea of "finished" would be a second opinion nobody reviewed.
    """
    # Imported here, as ``mode_run`` does: the Mac-only modes must not
    # pay for a module that reads the GPU.
    from src.training import gpu_node

    problems = acceptance.node_only_problems("run", gpu_node.probe())
    if problems:
        return _refuse(problems, "refusing to run the schedule here:")

    problems = _carried_digest_problems(args)
    if not (args.results and args.evidence):
        problems.append("--run-all needs --results and --evidence")
    if not args.adapter_dir:
        problems.append("--run-all needs --adapter-dir: half the schedule is "
                        "the fine-tuned arm, and the loader is told where "
                        "final_H2 is rather than finding it")
    problems += _log_dir_problems(args.log_dir, args.pack_dir)
    if problems:
        return _refuse(problems, "refusing to run the schedule:")

    plan_path = Path(args.plan or (ROOT / acceptance.PLAN_PATH))
    try:
        plan = acceptance.read_plan(plan_path)
    except PlanRefused as exc:
        print(f"refusing to run against this plan: {exc}", file=sys.stderr)
        return 2

    results, evidence = Path(args.results), Path(args.evidence)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    for index in range(N_STEPS):
        group, arm = acceptance.step(index)
        state = acceptance.step_state(results, evidence, plan, index)
        where = f"step {index} ({group}/{arm})"

        if state == acceptance.STEP_SEALED:
            print(f"{where}: sealed already, skipping", flush=True)
            continue

        # PARTIAL fills the gaps and never replaces a cell. UNSEALED needs no
        # resume: the runner's own seal branch writes the closing record and
        # decodes nothing, which is the whole point of that state existing.
        resume = state == acceptance.STEP_PARTIAL
        print(f"{where}: {state}"
              + (", resuming" if resume else "")
              + (f", adapter {args.adapter_dir}"
                 if acceptance.ARMS[arm].model != acceptance.PUBLIC_MODEL
                 else ", no adapter"),
              flush=True)

        log = log_dir / f"step{index}.log"
        with log.open("ab") as handle:
            completed = subprocess.run(_step_argv(args, index, resume=resume),
                                       stdout=handle, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, cwd=ROOT)
        if completed.returncode != 0:
            return _refuse(
                [f"{where} exited {completed.returncode}; its output is in "
                 f"{log}"],
                "stopping: a step did not succeed, so the next one does not "
                "start:")

        if not acceptance.completion_path(evidence, index).is_file():
            return _refuse(
                [f"{where} exited 0 and wrote no completion record"],
                "stopping: a step reported success it cannot show:")

        problems = acceptance.step_chain_problems(
            evidence, plan, index,
            expected_pack_digest=args.expected_pack_digest,
            expected_dependency_digest=args.expected_dependency_digest,
            results_path=results, require_completion=True)
        if problems:
            return _refuse(problems[:20],
                           f"stopping: {where} sealed and its chain does not "
                           "verify:")
        print(f"{where}: sealed and verified", flush=True)

    print(f"all {N_STEPS} steps sealed and verified", flush=True)
    return 0


def mode_verify(args) -> int:
    problems = _mac_guard("verify")
    if problems:
        return _refuse(problems, "refusing to verify here:")
    plan = _load_plan(args)
    if plan is None or not args.results or not args.evidence:
        print("--plan, --results and --evidence are all required",
              file=sys.stderr)
        return 2
    problems = _carried_digest_problems(args)
    if problems:
        return _refuse(problems, "refusing to verify without both carried "
                                 "digests:")

    arms = tuple(args.arms.split(",")) if args.arms else ARM_ORDER
    problems = _dedupe(
        acceptance.validate_results(Path(args.results), plan, arms=arms)
        + acceptance.evidence_problems(
            Path(args.evidence), plan, results_path=Path(args.results),
            expected_pack_digest=args.expected_pack_digest,
            expected_dependency_digest=args.expected_dependency_digest,
            arms=arms))
    print(json.dumps({"plan_digest": plan["plan_digest"],
                      "contract_digest": acceptance.contract_digest(),
                      "arms": list(arms),
                      "steps": N_STEPS,
                      "expected_cells": len(acceptance.expected_cells(
                          plan, arms=arms)),
                      "evidence_manifest_digest":
                          acceptance.evidence_manifest_digest(
                              Path(args.evidence), plan, arms=arms),
                      "problems": len(problems)}, indent=2))
    for problem in problems:
        print(f"  - {problem}")
    return 0 if not problems else 1


def mode_score(args) -> int:
    from src.eval import scoring
    from src.training.session import write_once_json

    problems = _mac_guard("score")
    if problems:
        return _refuse(problems, "refusing to score here:")
    if not args.plan or not args.results or not args.out or not args.evidence:
        print("--plan, --results, --evidence and --out are all required. "
              "A score with no evidence beside it is a number with no "
              "provenance: nothing would say which machine produced the "
              "cells, under which pack, or whether the preflight passed.",
              file=sys.stderr)
        return 2
    problems = _carried_digest_problems(args)
    if problems:
        return _refuse(problems, "refusing to score without both carried "
                                 "digests:")

    plan = _load_plan(args)
    arms = tuple(args.arms.split(",")) if args.arms else ARM_ORDER

    # Before a single cell is read: is the scorer on this machine the one
    # this plan was approved with?
    problems = acceptance.plan_scorer_problems(plan, ROOT)
    if problems:
        return _refuse(problems, "refusing to score with a different scorer:")

    problems = _dedupe(
        acceptance.validate_results(Path(args.results), plan, arms=arms)
        + acceptance.evidence_problems(
            Path(args.evidence), plan, results_path=Path(args.results),
            expected_pack_digest=args.expected_pack_digest,
            expected_dependency_digest=args.expected_dependency_digest,
            arms=arms))
    if problems:
        return _refuse(problems[:20],
                       "refusing to score results that do not verify:")

    cases = acceptance.case_index(plan)
    rows = acceptance.read_cells(Path(args.results))
    scores = [scoring.score_row(r, cases[r["case_id"]]) for r in rows]

    record = scoring.score_record(scores, k=SETTINGS.k, arms=arms, root=ROOT)
    record["contract"] = acceptance.contract_document(ROOT)
    record["contract_digest"] = acceptance.contract_digest()
    record["plan_digest"] = plan["plan_digest"]
    record["schedule"] = plan["schedule"]
    record["plan_scorer_source_manifest_digest"] = plan[
        "scorer_source_manifest_digest"]
    record["evidence_manifest"] = acceptance.evidence_manifest(
        Path(args.evidence), plan, arms=arms)
    record["evidence_manifest_digest"] = acceptance.evidence_manifest_digest(
        Path(args.evidence), plan, arms=arms)
    write_once_json(Path(args.out), record)

    print(json.dumps({
        "core_success_at_k": {
            name: record["per_arm"][name]["overall"]["core_success_at_k"]
            for name in arms},
        "contrasts": [
            {"contrast": c["contrast"],
             "overall_core_success_delta":
                 c["strata"][0]["metrics"]["core_success_at_k"]}
            for c in record["contrasts"]["contrasts"]],
        "scorer_source_manifest_digest":
            record["scorer_source_manifest_digest"],
        "evidence_manifest_digest": record["evidence_manifest_digest"],
    }, indent=2))
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Core acceptance for arms B, C, D and E.")
    ap.add_argument("--contract", action="store_true",
                    help="print the frozen contract and stop")
    ap.add_argument("--materialize", action="store_true",
                    help=f"Mac only; build the frozen plan, needs {TEST_GUARD}")
    ap.add_argument("--reissue-plan", action="store_true",
                    help="Mac only; re-issue an existing plan against "
                         "the current scorer source. Never opens the "
                         "test split")
    ap.add_argument(TEST_GUARD, action="store_true",
                    help="the deliberate act of opening the test split")
    ap.add_argument("--run", action="store_true",
                    help="WSL2 CUDA only; decode one step of the schedule")
    ap.add_argument("--run-all", action="store_true",
                    help="WSL2 CUDA only; run steps 0..7 of the frozen "
                         "schedule, each in its own process")
    ap.add_argument("--verify", action="store_true", help="Mac only")
    ap.add_argument("--score", action="store_true", help="Mac only")
    ap.add_argument("--step", type=int, metavar="N",
                    help=f"which step of the frozen schedule, 0..{N_STEPS - 1}")
    ap.add_argument("--arms", metavar="B,C,D,E",
                    help="which arms a verify or score should expect")
    ap.add_argument("--plan", metavar="FILE")
    ap.add_argument("--results", metavar="FILE")
    ap.add_argument("--evidence", metavar="DIR",
                    help="the write-once attempt and completion records; "
                         "required by --run, --verify and --score alike")
    ap.add_argument("--out", metavar="FILE")
    ap.add_argument("--pack-dir", metavar="DIR", default=".",
                    help="the pack this node is running, for its preflight")
    ap.add_argument("--expected-pack-digest", metavar="SHA256",
                    help="the value the build machine printed, carried here "
                         "by a route the pack did not travel")
    ap.add_argument("--expected-dependency-digest", metavar="SHA256",
                    help="likewise, for the tokenizer, base model and "
                         "published adapter this machine holds")
    ap.add_argument("--expected-plan-digest", metavar="SHA256",
                    help="with --reissue-plan, the plan_digest of the "
                         "plan being reissued, carried from outside it")
    ap.add_argument("--expected-plan-sha256", metavar="SHA256",
                    help="with --reissue-plan, the SHA-256 of that "
                         "plan's bytes, likewise carried in")
    ap.add_argument("--adapter-dir", metavar="DIR",
                    help=f"where {acceptance.FINAL_MODEL} lives; it does not "
                         "travel in the pack and is checked against the "
                         "digests in the plan")
    ap.add_argument("--log-dir", metavar="DIR",
                    help="with --run-all, where each step's output goes. "
                         "Must be outside the pack, and has no default")
    ap.add_argument("--resume", action="store_true",
                    help="fill only the cells of this step that are missing; "
                         "never re-run or replace one that is there")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    chosen = [f for f in ("contract", "materialize", "reissue_plan",
                          "run", "run_all", "verify", "score")
              if getattr(args, f)]
    if len(chosen) != 1:
        print("pass exactly one of --contract, --materialize, "
              "--reissue-plan, --run, --run-all, --verify, --score",
              file=sys.stderr)
        return 2
    return {
        "contract": lambda: mode_contract(),
        "materialize": lambda: mode_materialize(args),
        "reissue_plan": lambda: mode_reissue_plan(args),
        "run": lambda: mode_run(args),
        "run_all": lambda: mode_run_all(args),
        "verify": lambda: mode_verify(args),
        "score": lambda: mode_score(args),
    }[chosen[0]]()


if __name__ == "__main__":
    raise SystemExit(main())
