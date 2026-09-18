#!/usr/bin/env python3
"""Phase 3C: the way to reach the contract in :mod:`src.eval.phase3c`.

The contract is frozen there, before any number from this run exists. This
file is only the door, and the modes correspond to the two machines the work
is split across.

``--contract``
    Print the frozen contract and stop. Reads nothing, writes nothing, runs
    anywhere.

``--audit`` -- **Mac only**
    Run the group-level isolation audit and print it. Opens the test split's
    ids and captions to compute exclusions, writes nothing, and never opens
    ``val``. Useful before ``--materialize`` because it answers "are there
    enough independent pairs" without committing to a plan.

``--materialize`` -- **Mac only**
    Open the test split once, apply the three exclusions, draw 20 pairs and
    write a whole *generation* -- plan, case membership, isolation audit and
    a snapshot of the contract -- into ``data/phase3c/frozen/<generation>/``.
    Guarded four ways: an explicit ``--open-test-after-codex-approval`` flag,
    the split's SHA-256 checked against the value the contract pins,
    ``plan_leak_problems`` over the whole plan, and a write-once destination
    that refuses all four if any one exists. It prints counts and digests and
    nothing else -- a materialiser that echoed a caption would be a
    materialiser that leaked one into a terminal log.

``--authorize`` -- **Mac only**
    Bind the plan to the pack that will carry it, the dependency set the node
    will resolve, and the digest of every module the generation path can
    reach -- the two gates included. Written once, beside the plan and
    outside any run directory. **No step may run without it**: the node
    recomputes all of it and refuses if anything differs, and the Mac refuses
    to score a run whose manifest does not carry this document's digest.

    The gap it closes: before it existed the manifest merely *recorded* a
    pack digest and a dependency digest, and nothing compared them to
    anything, so any two well-formed hex strings were accepted on both
    machines -- and nothing bound the gate source at all.

``--run`` -- **execution node, WSL2 with CUDA only**
    Decode one *step* of the frozen schedule; there are six, one per
    (group, arm). Before anything loads it runs the node preflight against
    the two digests the operator carried by hand, checks the adapter against
    the digests in the plan, checks the observed environment field by field
    against the one the contract pins, and warms the device on a caption in
    no split. Step 0 additionally writes the execution manifest, once, before
    its first cell. Samples are appended cell by cell, so a step that dies at
    cell 200 leaves 200 rows that each name the plan and the manifest that
    produced them. It parses nothing and scores nothing, and there is no CPU
    or MPS fallback.

``--seal`` -- **execution node**
    Close the run: refuse unless all six members exist and account for every
    expected row, then write ``seal.json``. There is no partial seal. The
    ``seal_digest`` it prints is what the operator carries to the Mac by a
    route other than the directory, because a digest travelling inside the
    thing it authenticates authenticates nothing.

``--verify`` / ``--score`` / ``--report`` -- **Mac only**
    ``--verify`` needs ``--carried-seal-digest`` and has no path that skips
    it: it re-hashes every sealed member here, checks the grid against the
    plan cell by cell, and writes the receipt. ``--score`` re-derives the
    scorer source manifest, refuses unless it is the one the plan was
    approved with, then scores every stored sample from scratch -- the node's
    own summary is never read. ``--report`` renders what ``--score`` derived
    and derives nothing of its own.

Usage::

  ./.venv/bin/python scripts/59_phase3c.py --contract
  ./.venv/bin/python scripts/59_phase3c.py --audit
  ./.venv/bin/python scripts/59_phase3c.py --materialize \\
      --out-dir data/phase3c/frozen/gen03 --open-test-after-codex-approval
  ./.venv/bin/python scripts/59_phase3c.py --stage
  ./.venv/bin/python scripts/59_phase3c.py --authorize \\
      --pack-manifest DIR/pack_manifest.json \\
      --expected-pack-digest SHA --expected-dependency-digest SHA
  python scripts/59_phase3c.py --run --step 0 --plan PLAN \\
      --authorization GRANT --out-dir DIR \\
      --expected-pack-digest SHA --expected-dependency-digest SHA \\
      --adapter-dir DIR
  python scripts/59_phase3c.py --seal --plan PLAN --authorization GRANT \\
      --out-dir DIR
  ./.venv/bin/python scripts/59_phase3c.py --verify --plan PLAN \\
      --authorization GRANT --out-dir DIR --carried-seal-digest SHA
  ./.venv/bin/python scripts/59_phase3c.py --score --plan PLAN \\
      --authorization GRANT --out-dir DIR --carried-seal-digest SHA
  ./.venv/bin/python scripts/59_phase3c.py --report --plan PLAN \\
      --authorization GRANT --out-dir DIR
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import phase3c  # noqa: E402
from src.eval.acceptance import PlanRefused  # noqa: E402

#: Without this the materialiser does nothing. The test split is opened
#: deliberately, after review, and the flag is how that becomes a decision
#: somebody made rather than a default somebody inherited.
TEST_GUARD = "--open-test-after-codex-approval"

#: The Mac-only stages of *this* phase. ``acceptance.mac_only_problems`` is
#: not widened to hold them: its mode list is part of Phase 2's frozen
#: contract vocabulary, and editing that vocabulary so a later phase fits is
#: exactly the retroactive rewrite the project forbids. The guard is
#: reimplemented here, against the same constant, so the two phases share a
#: definition of "the Mac" without sharing a mutable list.
MAC_ONLY_MODES: tuple[str, ...] = ("audit", "materialize", "authorize",
                                   "stage", "verify", "score", "report")
MAC_SYSTEM = "Darwin"


def _refuse(problems, headline: str) -> int:
    print(headline, file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 2


def _mac_guard(mode: str, *, system=None) -> list[str]:
    if mode not in MAC_ONLY_MODES:
        return [f"{mode!r} is not one of the Mac-only stages "
                f"{list(MAC_ONLY_MODES)}"]
    system = platform.system() if system is None else system
    if system != MAC_SYSTEM:
        return [f"--{mode} runs on the Mac only, and this is {system!r}. The "
                "execution node decodes; it does not open the test split and "
                "it does not score."]
    return []


def _plan_path(args) -> Path:
    return Path(args.plan) if args.plan else ROOT / phase3c.PLAN_PATH


def _authorization_path(args) -> Path:
    """Where the grant is, explicit or beside the plan.

    This read ``args.authorization`` and returned ``args.grant``, which is
    not a parse target: every command that passed ``--authorization``
    explicitly died with ``AttributeError`` before any check ran, and the
    only reason no test caught it is that the tests called the helpers
    rather than the CLI. There are argparse-level tests now.
    """
    if args.authorization:
        return Path(args.authorization)
    return _plan_path(args).with_name(
        Path(phase3c.AUTHORIZATION_PATH).name)


def _load_plan(args, *, check_scorer=True):
    return phase3c.read_plan(_plan_path(args), root=ROOT,
                             check_scorer=check_scorer)


def _load_authorization(args, plan, *, check_sources: bool):
    return phase3c.read_authorization(_authorization_path(args), plan,
                                      root=ROOT,
                                      check_sources=check_sources)


def _carried_digest_problems(args, grant) -> list[str]:
    """Well-formed *and* the authorised values. Shape alone proves nothing.

    This used to check the shape only, because there was nothing on this
    machine holding the value they were supposed to equal. The
    authorization holds them now, so the comparison is the point and the
    shape check is only there to give a clearer message first.
    """
    from src.training import pack as pack_module

    problems = ([f"--expected-pack-digest: {p}" for p in
                 pack_module.expected_digest_problems(
                     args.expected_pack_digest)]
                + [f"--expected-dependency-digest: {p}" for p in
                   pack_module.expected_digest_problems(
                       args.expected_dependency_digest,
                       what="dependency digest")])
    problems.extend(phase3c.carried_digest_problems(
        grant,
        pack_digest=args.expected_pack_digest,
        dependency_digest=args.expected_dependency_digest))
    return problems


# ---------------------------------------------------------------------------
# modes that read nothing
# ---------------------------------------------------------------------------

def mode_contract() -> int:
    document = phase3c.contract_document(ROOT)
    print(json.dumps(document, indent=2, ensure_ascii=False))
    print(f"\ncontract_digest: {phase3c.contract_digest(ROOT)}")
    print(f"scorer_source_manifest_digest: "
          f"{phase3c.scorer_manifest_digest(ROOT)}")
    return 0


# ---------------------------------------------------------------------------
# Mac: the audit and the plan
# ---------------------------------------------------------------------------

def mode_audit(args) -> int:
    problems = _mac_guard("audit")
    if problems:
        return _refuse(problems, "refusing to audit here:")
    try:
        audit = phase3c.isolation_audit(ROOT)
    except PlanRefused as exc:
        print(f"refusing to audit: {exc}", file=sys.stderr)
        return 2
    # Pair ids, object ids and counts. No caption and no inventory is
    # printed: the audit exists to prove separation, not to display rows.
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


def mode_materialize(args) -> int:
    problems = _mac_guard("materialize")
    if problems:
        return _refuse(problems, "refusing to materialise here:")
    if not args.open_test_after_codex_approval:
        print(f"refusing to open {phase3c.TEST_FILE} without {TEST_GUARD}. "
              "The test split is opened deliberately, after review.",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir or (ROOT / phase3c.ARCHIVE_DIR))
    try:
        body, membership, audit = phase3c.materialize_plan(ROOT)
        written = phase3c.write_plan_set(out_dir, body, membership, audit,
                                         root=ROOT)
    except PlanRefused as exc:
        print(f"refusing to materialise the plan: {exc}", file=sys.stderr)
        return 2

    # Counts and digests only. Nothing from a case is printed.
    print(json.dumps({
        "cases": len(body["cases"]),
        "pairs": body["source"]["pairs"],
        "contract_digest": body["contract_digest"],
        "plan_digest": body["plan_digest"],
        "case_membership_digest": body["case_membership_digest"],
        "audit_digest": body["audit_digest"],
        "scorer_source_manifest_digest":
            body["scorer_source_manifest_digest"],
        "generation": phase3c.GENERATION,
        "supersedes": [e["generation"] for e in phase3c.SUPERSEDED],
        "written": {k: str(v) for k, v in written.items()},
        "next": ("--authorize, once the pack is built; no step may run "
                 "before an execution authorization exists"),
    }, indent=2))
    return 0


def mode_authorize(args) -> int:
    """Bind the plan to a pack, a dependency set and a generation source.

    Run on the Mac, after ``18_gpu_pack.py --build`` and
    ``--dependencies``, and before anything reaches the node. The two
    digests are the ones those tools printed; they are typed in here rather
    than read back out of the pack, because a value read from the thing it
    describes authenticates nothing.
    """
    from src.training.session import now_iso, write_once_json

    problems = _mac_guard("authorize")
    if problems:
        return _refuse(problems, "refusing to authorise here:")
    if not args.expected_pack_digest or not args.expected_dependency_digest:
        print("--expected-pack-digest and --expected-dependency-digest are "
              "both required; they are what this document exists to bind",
              file=sys.stderr)
        return 2
    try:
        plan = _load_plan(args)
    except PlanRefused as exc:
        return _refuse([str(exc)], "refusing to authorise:")

    path = _authorization_path(args)

    # The evidence for the pack digest comes first: the grant *binds* its
    # digest, so it has to exist before the grant can name it. Archived
    # before the "this grant already exists" branch too, so a rerun cannot
    # leave the archive with a grant and no evidence.
    # ``--pack-manifest`` points at the ``pack_manifest.json`` the build
    # wrote; without it the authorised digest would be a number with no path
    # back to the bytes it describes, which is why gen02 is superseded.
    if not args.pack_manifest:
        print("--pack-manifest is required: it is the file table the pack "
              "digest was taken over, and a digest nobody can re-derive "
              "binds nothing", file=sys.stderr)
        return 2
    manifest_file = Path(args.pack_manifest)
    if not manifest_file.is_file():
        print(f"{manifest_file} is not here", file=sys.stderr)
        return 2
    try:
        evidence = phase3c.build_pack_evidence(
            json.loads(manifest_file.read_text()),
            pack_digest=args.expected_pack_digest)
    except PlanRefused as exc:
        return _refuse([str(exc)], "refusing to authorise:")
    problems = phase3c.pack_evidence_problems(
        evidence, pack_digest=args.expected_pack_digest)
    if problems:
        return _refuse(problems, "the pack evidence does not hold:")

    # Which plan actually travelled, read out of the file table rather than
    # out of the pointer that chose it. The pointer is data and data can be
    # stale; this is the check a stale pointer cannot pass, because it
    # compares the carried bytes against the archive's own.
    problems = phase3c.pack_carries_this_generation_problems(evidence)
    if problems:
        return _refuse(problems,
                       "the pack does not carry this generation's plan:")
    evidence_path = path.with_name(Path(phase3c.PACK_EVIDENCE_PATH).name)
    if evidence_path.exists():
        existing = json.loads(evidence_path.read_text())
        if existing.get("evidence_digest") != evidence["evidence_digest"]:
            return _refuse(
                [f"{evidence_path} already holds different pack evidence"],
                "refusing to replace archived pack evidence:")
    else:
        write_once_json(evidence_path, evidence)

    try:
        body = phase3c.build_authorization(
            plan=plan, pack_digest=args.expected_pack_digest,
            dependency_digest=args.expected_dependency_digest,
            pack_evidence_digest=evidence["evidence_digest"],
            authorized_at=now_iso(), root=ROOT)
    except PlanRefused as exc:
        return _refuse([str(exc)], "refusing to authorise:")

    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("authorization_digest") != \
                body["authorization_digest"]:
            return _refuse(
                [f"{path} already authorises "
                 f"{str(existing.get('authorization_digest'))[:16]}... and "
                 f"this would authorise {body['authorization_digest'][:16]}"
                 "..."],
                "refusing to replace a grant that already exists:")
        print(f"{path} already holds this exact authorization; nothing "
              "rewritten")
        return 0

    problems = phase3c.authorization_problems(body, plan, root=ROOT)
    if problems:
        return _refuse(problems,
                       "the authorization this run produces does not hold:")

    write_once_json(path, body)
    print(json.dumps({
        "pack_evidence_digest": evidence["evidence_digest"],
        "pack_evidence_files": evidence["n_files"],
        "authorization_digest": body["authorization_digest"],
        "plan_digest": body["plan_digest"],
        "pack_digest": body["pack_digest"],
        "dependency_digest": body["dependency_digest"],
        "pack_evidence_digest": body["pack_evidence_digest"],
        "generation_source_manifest_digest":
            body["generation_source_manifest_digest"],
        "gate_sources": body["gate_sources"],
        "files_pinned": len(body["generation_source_manifest"]),
        "written": str(path),
    }, indent=2))
    return 0


def mode_stage(args) -> int:
    """Copy the plan out of the archive, for the pack to carry.

    The **plan only**. The grant is not staged and is not in the pack: it
    names that pack's own digest, so shipping it inside would put the value
    and the thing it authenticates in one parcel. It travels by hand and the
    node is pointed at it with ``--authorization``.

    Separate from ``--authorize`` on purpose: authorising is a judgement
    about what may run, and staging is the moment a file leaves this
    machine's archive for a machine that will execute it. Running them
    together would hide the second decision inside the first.
    """
    problems = _mac_guard("stage")
    if problems:
        return _refuse(problems, "refusing to stage here:")
    try:
        written = phase3c.stage_for_node(
            ROOT, archive_dir=Path(args.out_dir) if args.out_dir else None)
    except PlanRefused as exc:
        return _refuse([str(exc)], "refusing to stage:")
    problems = phase3c.staged_copy_problems(
        ROOT, archive_dir=Path(args.out_dir) if args.out_dir else None)
    if problems:
        return _refuse(problems, "the staged copies do not hold:")
    print(json.dumps(written, indent=2))
    print("\nBuild the pack now; this is the Phase 3C file it carries. The "
          "grant is not staged and does not travel in the pack.")
    return 0


# ---------------------------------------------------------------------------
# Node: the decode
# ---------------------------------------------------------------------------

def mode_run(args) -> int:
    import torch

    from src.eval import acceptance
    from src.training import gpu_node
    from src.training.session import now_iso, write_once_json

    probe_reading = gpu_node.probe()
    problems = acceptance.node_only_problems("run", probe_reading)
    if problems:
        return _refuse(problems, "refusing to run here:")

    if args.step is None:
        print("--step is required; the schedule is frozen and the step "
              "number is the order", file=sys.stderr)
        return 2
    if not args.out_dir:
        print("--out-dir is required", file=sys.stderr)
        return 2
    try:
        plan = _load_plan(args, check_scorer=False)
        # check_sources=True: this *is* the generation machine, so the
        # authorised source digests are recomputed here against the tree
        # that is about to decode. An edited gate stops the run now.
        grant = _load_authorization(args, plan, check_sources=True)
    except PlanRefused as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2

    problems = _carried_digest_problems(args, grant)
    if problems:
        return _refuse(problems,
                       "refusing to run: the carried digests are not the "
                       "authorised ones:")

    out_dir = Path(args.out_dir)
    member = out_dir / phase3c.samples_member(args.step)
    member.parent.mkdir(parents=True, exist_ok=True)

    # The environment the contract pinned, checked against what this machine
    # actually is, before a single weight loads. A run that decoded first and
    # compared afterwards would have burned the GPU time before finding out
    # the numbers were not comparable to anything.
    observed = phase3c.observed_environment(probe_reading)
    problems = phase3c.environment_problems(observed)
    if problems:
        return _refuse(problems,
                       "this machine is not the one the contract pins:")

    preflight = gpu_node.preflight(
        probe=probe_reading, pack_dir=args.pack_dir,
        expected_pack_digest=args.expected_pack_digest,
        expected_dependency_digest=args.expected_dependency_digest)
    if not preflight["passed"]:
        return _refuse(
            [f"{k}: {preflight['checks'][k]['detail']}"
             for k in preflight["failed"]],
            "the node preflight did not pass, so nothing here starts:")

    if not args.adapter_dir:
        print(f"every arm runs {phase3c.FINAL_MODEL} and needs "
              "--adapter-dir; the weights are not in the pack",
              file=sys.stderr)
        return 2
    adapter_dir = Path(args.adapter_dir)
    problems = acceptance.plan_final_adapter_problems(plan, adapter_dir)
    if problems:
        return _refuse(problems, "refusing to load these weights:")

    # Written once, before the first cell of the first step, and read back
    # and compared on every later step. A run whose manifest changed halfway
    # is a run in two environments.
    manifest_path = out_dir / phase3c.MANIFEST_NAME
    if not manifest_path.exists():
        manifest = phase3c.build_execution_manifest(
            plan=plan, grant=grant, observed=observed,
            started_at=now_iso(),
            recomputed_sources=phase3c.generation_source_manifest(ROOT))
        write_once_json(manifest_path, manifest)
    try:
        manifest = phase3c.read_execution_manifest(out_dir)
    except PlanRefused as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    problems = phase3c.manifest_problems(manifest, plan, grant)
    if problems:
        return _refuse(problems, "the execution manifest does not hold:")

    try:
        device = acceptance.resolve_device(torch)
    except acceptance.DeviceRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2

    group, name = phase3c.step(args.step)
    # Resume, bound to the manifest. ``known_keys`` alone would count a row
    # written under a different execution as done and skip the cell, so the
    # rows already here are judged against this manifest first and a
    # disagreement stops the step rather than silently shrinking it.
    if member.is_file():
        problems = phase3c.row_binding_problems(
            out_dir, plan, manifest_digest=manifest["manifest_digest"])
        problems = [p for p in problems
                    if p.startswith(phase3c.samples_member(args.step))]
        if problems:
            return _refuse(problems,
                           "refusing to resume: rows already here were not "
                           "produced under this execution manifest:")
    done = phase3c.known_keys(member)
    todo = [c for c in phase3c.step_cells(plan, args.step) if c not in done]
    if not todo:
        print(f"step {args.step} ({group}/{name}) is complete: "
              f"{len(done)} cells")
        return 0

    # Phase 3C's own door, reading Phase 3C's own arms. Handing 'A', 'B' or
    # 'C' to ``acceptance`` resolves them against Phase 2's registry, where
    # 'A' does not exist and 'B' and 'C' mean different weights and no gate.
    #
    # The load and the check that it is the right load are one try block: a
    # door that refuses to open and a door that opened onto the wrong weights
    # are the same answer here, and both must come before the warm-up and
    # before the first cell so that a wrong step writes nothing at all rather
    # than a member's worth of rows claiming weights they did not come from.
    try:
        interface, load_info = phase3c.build_interface(
            name, device=device, adapter_dir=adapter_dir)
        identity = phase3c.verified_model_identity(load_info, adapter_dir)
    except PlanRefused as exc:
        return _refuse(
            [str(exc)],
            "the weights that loaded are not the ones this arm runs, so "
            "nothing is measured:")

    print(f"step {args.step} ({group}/{name}): {len(todo)} cells on {device}, "
          f"plan {plan['plan_digest'][:16]}...", flush=True)
    # The same gate the measured cells use: warming an ungated decode and
    # then measuring a gated one warms the wrong work.
    #
    # And checked, here, before the first cell. The warm-up is the one part
    # of a step that is decoded and then thrown away: no digest covers it, no
    # sample row records it, and nothing downstream re-derives it. A step
    # whose warm-up ran a different gate from its cells would seal, verify
    # and score without a single check noticing -- so the check has to be
    # here or nowhere.
    try:
        warmup = phase3c.warm_up(interface, name)
    except Exception as exc:                     # noqa: BLE001 - fail closed
        return _refuse(
            [f"{type(exc).__name__}: {exc}"],
            "the warm-up did not complete, so no cell of this step is "
            "measured against a known warm state:")
    problems = phase3c.warm_up_problems(warmup, name)
    if problems:
        return _refuse(
            problems,
            "the warm-up that ran is not the warm-up this arm's contract "
            "names, so nothing is measured:")
    print(f"  warm-up: {warmup['generations']} generations through gate "
          f"{warmup['gate']}, excluded", flush=True)

    cases = phase3c.case_index(plan)
    written = 0
    for i, (_digest, case_id, _arm, seed) in enumerate(todo, 1):
        # A gate that is not this arm's is a refusal, not a traceback. The
        # warm-up already checked the object the decode builds, so reaching
        # this means the gate changed mid-step; the step stops here with the
        # rows it had rather than exiting on an unhandled exception, and the
        # member is short so --seal refuses it.
        try:
            row = phase3c.run_case(
                interface, cases[case_id], name, seed,
                plan_digest_value=plan["plan_digest"],
                manifest_digest=manifest["manifest_digest"],
                step_index=args.step, group=group, model=identity)
        except PlanRefused as exc:
            return _refuse(
                [str(exc)],
                f"cell {i} of this step did not decode through this arm's "
                "gate, so the step stops rather than measuring it:")
        try:
            phase3c.append_cell(member, row, known=done)
        except PlanRefused as exc:
            print(f"stopping: {exc}", file=sys.stderr)
            return 2
        written += 1
        if i % 40 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} cells", flush=True)

    problems = phase3c.step_problems(
        phase3c.read_rows(member), plan, args.step,
        manifest_digest=manifest["manifest_digest"])
    if problems:
        return _refuse(problems[:20],
                       f"step {args.step} finished and does not hold:")
    # ``done`` is the set ``append_cell`` has been updating, so it already
    # counts everything written in this process as well as everything
    # resumed. Adding ``written`` to it reported a 320-cell step as 640.
    print(f"step {args.step} ({group}/{name}) complete: {written} decoded, "
          f"{len(done)} cells")
    return 0


def mode_seal(args) -> int:
    from src.training.session import write_once_json

    if not args.out_dir:
        print("--out-dir is required", file=sys.stderr)
        return 2
    try:
        plan = _load_plan(args, check_scorer=False)
        grant = _load_authorization(args, plan, check_sources=True)
    except PlanRefused as exc:
        print(f"refusing to seal: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    # Through the checked reader: a manifest whose stated digest does not
    # cover its own contents is refused here rather than sealed and carried
    # to the Mac, and the value every row is bound to is the recomputed one.
    try:
        manifest = phase3c.read_execution_manifest(out_dir)
    except PlanRefused as exc:
        print(f"refusing to seal: {exc}", file=sys.stderr)
        return 2
    manifest_digest = manifest["manifest_digest"]

    # Every step, checked against the plan, before anything is sealed. This
    # is where a partial run is refused: ``build_seal`` needs all six members
    # and ``seal_problems`` needs every expected row.
    problems = list(phase3c.manifest_problems(manifest, plan, grant))
    for index in range(phase3c.N_STEPS):
        path = out_dir / phase3c.samples_member(index)
        if not path.is_file():
            problems.append(f"{phase3c.samples_member(index)} is missing; a "
                            "seal covers three complete arms or none")
            continue
        problems.extend(
            f"step {index}: {p}"
            for p in phase3c.step_problems(
                phase3c.read_rows(path), plan, index,
                manifest_digest=manifest_digest))
    if problems:
        return _refuse(problems[:20], "refusing to seal this run:")

    try:
        seal = phase3c.build_seal(out_dir, manifest)
    except PlanRefused as exc:
        print(f"refusing to seal: {exc}", file=sys.stderr)
        return 2
    problems = phase3c.seal_problems(seal, plan)
    if problems:
        return _refuse(problems, "the seal this run produces does not hold:")

    write_once_json(out_dir / phase3c.SEAL_NAME, seal)
    print(json.dumps({"seal_digest": seal["seal_digest"],
                      "manifest_digest": seal["manifest_digest"],
                      "members": len(seal["members"]),
                      "rows": sum(m["rows"]
                                  for m in seal["members"].values())},
                     indent=2))
    print("\nCarry seal_digest to the Mac by a route other than this "
          "directory. --verify has no path that skips it.")
    return 0


# ---------------------------------------------------------------------------
# Mac: the receipt, the scores, the report
# ---------------------------------------------------------------------------

def _sealed(args):
    """``(plan, grant, seal, out_dir)``. No partial answers.

    ``check_sources=False``: this is the Mac, which deliberately is not the
    generation machine, so its copy of the decode path is not expected to
    hash to the authorised values. What the Mac checks instead is that the
    *node* recomputed them and that the manifest carries the result -- which
    is what ``manifest_chain_problems`` does.
    """
    out_dir = Path(args.out_dir)
    plan = _load_plan(args)
    grant = _load_authorization(args, plan, check_sources=False)
    seal_path = out_dir / phase3c.SEAL_NAME
    if not seal_path.is_file():
        raise PlanRefused(f"{phase3c.SEAL_NAME} is not in {out_dir}")
    return plan, grant, json.loads(seal_path.read_text()), out_dir


def mode_verify(args) -> int:
    from src.training.session import now_iso, write_once_json

    problems = _mac_guard("verify")
    if problems:
        return _refuse(problems, "refusing to verify here:")
    if not args.out_dir or not args.carried_seal_digest:
        print("--out-dir and --carried-seal-digest are both required; the "
              "carried digest is the whole point of the receipt",
              file=sys.stderr)
        return 2
    try:
        plan, grant, seal, out_dir = _sealed(args)
    except PlanRefused as exc:
        print(f"refusing to verify: {exc}", file=sys.stderr)
        return 2

    receipt = phase3c.build_receipt(out_dir, seal, args.carried_seal_digest,
                                    plan=plan, grant=grant,
                                    verified_at=now_iso())
    # The receipt is written whether or not it is clean: a refusal that
    # leaves no record is a refusal nobody can audit. The exit code, not the
    # existence of the file, is what says the run may be quoted.
    receipt_path = out_dir / phase3c.RECEIPT_NAME
    if not receipt_path.exists():
        write_once_json(receipt_path, receipt)
    else:
        existing = json.loads(receipt_path.read_text())
        # By identity, not by digest. ``receipt_digest`` covers
        # ``verified_at``, so comparing digests made a second verification of
        # an unchanged directory refuse its own unchanged conclusion the
        # moment the clock ticked. What must not change is the judgement.
        if phase3c.receipt_identity(existing) != \
                phase3c.receipt_identity(receipt):
            return _refuse(
                [f"{phase3c.RECEIPT_NAME} already records a different "
                 f"verification of this run"],
                "refusing to overwrite a receipt with a different one:")
        receipt = existing
        print(f"{receipt_path} already holds this exact verification; "
              "nothing rewritten")

    print(json.dumps({"verified": receipt["verified"],
                      "receipt_digest": receipt["receipt_digest"],
                      "seal_digest": receipt["seal_digest"],
                      "problems": receipt["problems"][:20]}, indent=2))
    return 0 if receipt["verified"] else 2


def mode_score(args) -> int:
    from src.training.session import write_once_json

    problems = _mac_guard("score")
    if problems:
        return _refuse(problems, "refusing to score here:")
    if not args.out_dir or not args.carried_seal_digest:
        print("--out-dir and --carried-seal-digest are both required",
              file=sys.stderr)
        return 2
    try:
        plan, grant, seal, out_dir = _sealed(args)
    except PlanRefused as exc:
        print(f"refusing to score: {exc}", file=sys.stderr)
        return 2

    # The receipt is re-derived rather than read. An existing-result path
    # that trusted the stored receipt would trust a file inside the
    # directory it authenticates.
    problems = phase3c.score_preconditions(
        out_dir, plan, seal, carried_seal_digest=args.carried_seal_digest,
        grant=grant, root=ROOT)
    if problems:
        return _refuse(problems[:20], "refusing to score this run:")

    scores_by_arm = phase3c.rescore(out_dir, plan)
    record = phase3c.score_record(scores_by_arm, plan=plan,
                                  k=phase3c.SETTINGS.k, root=ROOT)
    out = Path(args.out) if args.out else out_dir / phase3c.SCORES_NAME
    if out.exists():
        existing = json.loads(out.read_text())
        if phase3c.scores_identity(existing) != phase3c.scores_identity(record):
            return _refuse(
                ["the scores on disk are not the scores this run derives"],
                "refusing to overwrite an existing score record:")
        print(f"{out} already holds this exact record; nothing rewritten")
    else:
        write_once_json(out, record)

    overall = {name: record["per_arm"][name]["overall"]
               for name in phase3c.ARM_ORDER}
    print(json.dumps({
        "draws": record["draws"], "cases": record["cases"],
        "primary_contrast": record["primary_contrast"],
        # ``wilson`` writes ``value``, not ``rate``; this printed a
        # ``KeyError`` instead of a summary and nothing exercised it,
        # because no test ran --score through the CLI. One does now.
        "rates": {name: {k: v["value"] for k, v in
                         overall[name]["rates"].items()}
                  for name in phase3c.ARM_ORDER},
    }, indent=2))
    return 0


def mode_report(args) -> int:
    from src.eval import phase3c_report

    problems = _mac_guard("report")
    if problems:
        return _refuse(problems, "refusing to report here:")
    if not args.out_dir:
        print("--out-dir is required", file=sys.stderr)
        return 2
    # The grant and the seal, not just the plan. A report is published over
    # a chain that is re-derived here from the bytes; a stored
    # ``verified: true`` is not evidence to it, and there is no path that
    # renders one without a grant.
    try:
        plan, grant, seal, out_dir = _sealed(args)
    except PlanRefused as exc:
        print(f"refusing to report: {exc}", file=sys.stderr)
        return 2

    archive_dir = _plan_path(args).parent
    try:
        written = phase3c_report.write_report(
            out_dir, plan, grant=grant, seal=seal,
            out=Path(args.out) if args.out else None,
            root=ROOT, archive_dir=archive_dir)
    except PlanRefused as exc:
        print(f"refusing to report: {exc}", file=sys.stderr)
        return 2

    # The members, re-read from disk after writing. The chain above was
    # already re-derived inside ``write_report``; deriving the whole score
    # record a second time here would cost minutes and prove nothing new.
    # What this adds is that the bytes that landed are the bytes rendered.
    problems = phase3c_report.published_member_problems(
        Path(args.out).parent if args.out else out_dir, plan,
        out_dir=out_dir)
    if problems:
        return _refuse(problems[:20],
                       "the report was written and does not re-derive:")
    print(json.dumps({**{k: str(v) for k, v in written.items()},
                      "verified": True}, indent=2))
    return 0


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Phase 3C: does PlacementGate change legality")
    ap.add_argument("--contract", action="store_true",
                    help="print the frozen contract and stop")
    ap.add_argument("--audit", action="store_true",
                    help="Mac only; print the isolation audit, write nothing")
    ap.add_argument("--materialize", action="store_true", help="Mac only")
    ap.add_argument("--stage", action="store_true",
                    help="Mac only; copy the plan and grant out of the "
                         "archive for the pack to carry")
    ap.add_argument("--authorize", action="store_true",
                    help="Mac only; bind the plan to a pack, a dependency "
                         "set and the generation source")
    ap.add_argument(TEST_GUARD, action="store_true",
                    help="required by --materialize")
    ap.add_argument("--run", action="store_true",
                    help="execution node only, one step")
    ap.add_argument("--seal", action="store_true",
                    help="execution node; close a complete run")
    ap.add_argument("--verify", action="store_true", help="Mac only")
    ap.add_argument("--score", action="store_true", help="Mac only")
    ap.add_argument("--report", action="store_true", help="Mac only")
    ap.add_argument("--step", type=int, metavar="N",
                    help=f"0..{phase3c.N_STEPS - 1}")
    ap.add_argument("--plan", metavar="FILE")
    ap.add_argument("--authorization", metavar="FILE",
                    help="defaults to the authorization beside the plan")
    ap.add_argument("--out-dir", metavar="DIR",
                    help="where samples, manifest, seal and receipt live")
    ap.add_argument("--out", metavar="FILE")
    ap.add_argument("--pack-dir", metavar="DIR", default=".",
                    help="node only; the pack being executed")
    ap.add_argument("--pack-manifest", metavar="FILE",
                    help="Mac only, --authorize; the pack_manifest.json the "
                         "build wrote, archived as the evidence for the "
                         "pack digest")
    ap.add_argument("--expected-pack-digest", metavar="SHA256",
                    help="carried by hand, not read from the pack")
    ap.add_argument("--expected-dependency-digest", metavar="SHA256",
                    help="carried by hand, not read from the pack")
    ap.add_argument("--carried-seal-digest", metavar="SHA256",
                    help="carried from the node by a route other than the "
                         "directory it authenticates")
    ap.add_argument("--adapter-dir", metavar="DIR",
                    help="node only; the final adapter")
    return ap


MODES: tuple[str, ...] = ("contract", "audit", "materialize", "authorize",
                          "stage", "run", "seal", "verify", "score",
                          "report")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    chosen = [m for m in MODES if getattr(args, m.replace("-", "_"))]
    if len(chosen) != 1:
        print(f"choose exactly one of {['--' + m for m in MODES]}",
              file=sys.stderr)
        return 2
    mode = chosen[0]
    if mode == "contract":
        return mode_contract()
    return {
        "audit": mode_audit, "materialize": mode_materialize,
        "authorize": mode_authorize, "stage": mode_stage,
        "run": mode_run, "seal": mode_seal,
        "verify": mode_verify, "score": mode_score, "report": mode_report,
    }[mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
