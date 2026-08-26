#!/usr/bin/env python3
"""The execution node's entry point. Runs inside WSL2 Ubuntu on the GPU node.

Everything here fails closed. The preflight has to pass before a gate runs,
and there is no flag that skips it, softens it, or lets a failed check through
as a warning -- which also means this script cannot run a gate on the Mac, not
because it checks for a Mac but because a Mac has no CUDA device, no WSL2
kernel and no RTX 5070 Ti. The lock is the machine, not a promise.

Every mode that trusts the pack requires **two** carried values, neither with
a default and neither skippable:

``--expected-pack-digest``
    what ``18_gpu_pack.py --build`` printed. A manifest digests only itself,
    so a pack rewritten wholesale agrees with itself perfectly -- checking it
    against itself proves arithmetic.
``--expected-dependency-digest``
    what ``18_gpu_pack.py --dependencies`` printed. The pack carries no model:
    the tokenizer, the base weights and the published adapter come from this
    machine's own cache, and a file of the correct *name* resolves whatever is
    inside it. This binds their contents to the ones the Mac had.

Both travel by a route the things they describe did not, and neither is stored
in the pack manifest -- a digest kept beside the files it authenticates is
rewritten by whoever rewrites them.

Modes, and what each does to the disk:

  --preflight                  reads the machine, the pack and this machine's
                               dependency cache, and prints the dependency
                               digest it recomputed so it can be compared by
                               eye. Writes nothing.
  --verify-pack                the manifest, file by file. Writes nothing.
  --summary                    the gates and the two frozen arms. Writes nothing.
  --self-test --run-dir DIR    drives the whole gate pipeline with a
                               deterministic stand-in: no model, no device, no
                               dataset, no network. Writes only inside DIR.
  --gate NAME --run-dir DIR    the real thing. Loads the model, trains, saves,
                               cold-loads. Requires the preflight to pass.
  --resume --run-dir DIR       continues an interrupted real run, after
                               verifying every packed file and the checkpoint.
  --stop-after N               with --gate: stop deliberately after row N,
                               having checkpointed on the way. This is how gate
                               500's interruption is performed, and it is a
                               first-class mode rather than a test hook: an
                               interruption that only ever happens by accident
                               is one nobody has a recovery procedure for.

H1 and H2 are printed by ``--summary`` and cannot be run from here at all.
They are frozen settings, not runs. They become runnable only after the six
named runs of the formal gate suite -- gate 8, three gate 100 repeats, the
resumed gate 500 and its uninterrupted control -- exist on this node and are
shown to agree by :mod:`src.training.gate_suite`, which is a decision for the
person reading those results and not for this script. Three gates passing is
not the same claim: it says nothing about repeatability, about the two 500-row
runs matching, or about all six having come from one pack.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import gates, gpu_node, hypotheses, pack  # noqa: E402
from src.training.longrun import dependency_digest  # noqa: E402
from src.training.lora import LoraConfig_  # noqa: E402

PACK_ROOT = Path(__file__).resolve().parents[1]


def _print_preflight(result: dict) -> None:
    for name in sorted(result["checks"]):
        check = result["checks"][name]
        mark = "ok  " if check["passed"] else "FAIL"
        print(f"  [{mark}] {name}: {check['detail']}")
    # Printed whatever the verdict, and printed in full: the operator's job
    # here is to compare it against what the Mac reported, and a truncated
    # value cannot be compared.
    print(f"\nthis machine's dependency_digest: {result['dependency_digest']}")
    print("  verdict: "
          + ("matches the carried value"
             if result["checks"]["dependency_digest"]["passed"]
             else "DOES NOT match the carried value"))
    print(f"\npassed: {result['passed']}")
    if result["failed"]:
        print(f"failed: {', '.join(result['failed'])}")


def _preflight(pack_dir: Path, data_root: Path | None,
               expected_pack_digest, expected_dependency_digest) -> dict:
    return gpu_node.preflight(
        probe=gpu_node.probe(), pack_dir=pack_dir,
        expected_pack_digest=expected_pack_digest,
        expected_dependency_digest=expected_dependency_digest,
        data_root=data_root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--verify-pack", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--gate", choices=sorted(gates.GATES))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-dir", metavar="DIR")
    ap.add_argument("--stop-after", type=int, metavar="N")
    ap.add_argument("--pack-dir", metavar="DIR", default=str(PACK_ROOT))
    ap.add_argument("--data-root", metavar="DIR", default=str(PACK_ROOT))
    ap.add_argument("--expected-pack-digest", metavar="SHA256",
                    help="the pack_digest the build machine printed, carried "
                         "here separately. Required by --preflight, --gate, "
                         "--resume and --self-test; 64 lowercase hex.")
    ap.add_argument("--expected-dependency-digest", metavar="SHA256",
                    help="the dependency_digest `18_gpu_pack.py "
                         "--dependencies` printed, carried here separately. "
                         "Required by --preflight, --gate and --resume; "
                         "64 lowercase hex.")
    args = ap.parse_args(argv)

    pack_dir = Path(args.pack_dir)
    data_root = Path(args.data_root)
    digest = args.expected_pack_digest
    dep_digest = args.expected_dependency_digest

    needs_digest = (args.preflight or args.gate or args.resume
                    or args.self_test)
    needs_dependency = args.preflight or args.gate or args.resume
    if needs_digest:
        problems = pack.expected_digest_problems(digest)
        if needs_dependency:
            problems += pack.expected_digest_problems(
                dep_digest, what="dependency digest")
        if problems:
            for problem in problems:
                print(problem, file=sys.stderr)
            print("\nPass --expected-pack-digest with the value "
                  "`18_gpu_pack.py --build` printed on the Mac, and "
                  "--expected-dependency-digest with the value "
                  "`18_gpu_pack.py --dependencies` printed. There is no flag "
                  "that skips either: without them this node would be "
                  "checking the pack against numbers the pack itself "
                  "supplied, and its model cache against nothing at all.",
                  file=sys.stderr)
            return 2

    if args.summary:
        print(json.dumps({"node": gpu_node.NODE_SPEC,
                          "gates": gates.summary(),
                          "hypotheses": hypotheses.summary(),
                          "hypotheses_runnable_here": False,
                          "why": ("H1 and H2 are frozen settings. They become "
                                  "runnable when the six named runs of the "
                                  "formal gate suite exist on this node and "
                                  "agree with each other, and that decision "
                                  "is not this script's to make. Printing the "
                                  "declaration is not running it.")},
                         indent=2, ensure_ascii=False))
        return 0

    if args.verify_pack:
        problems = pack.verify(pack_dir, data_root=data_root)
        for problem in problems:
            print(problem)
        print(f"\n{len(problems)} problem(s)")
        return 1 if problems else 0

    if args.preflight:
        # Once. It resolves and digests every pinned dependency, so running it
        # twice to print it and then judge it would read the base model's
        # weights off disk a second time for nothing.
        result = _preflight(pack_dir, data_root, digest, dep_digest)
        _print_preflight(result)
        return 0 if result["passed"] else 2

    if args.self_test:
        if not args.run_dir:
            print("--self-test needs --run-dir", file=sys.stderr)
            return 2
        return _self_test(Path(args.run_dir), pack_dir, digest)

    if args.gate or args.resume:
        return _real_run(args, pack_dir, data_root, digest, dep_digest)

    ap.print_help()
    return 0


def _self_test(run_dir: Path, pack_dir: Path, digest: str) -> int:
    """The whole pipeline, with the model, device and dataset stood in for.

    Worth running on the node before anything expensive: it exercises the
    plan, the ledger, the checkpoints, a deliberate stop and a resume against
    the *real* pack on the *real* filesystem, which is where a permissions
    problem or a filesystem without hard links shows up. What it cannot tell
    you is anything about the GPU.
    """
    print("self-test: 500-row pipeline with a deterministic stand-in\n")
    # The dependency binding is stood in for here, deliberately. This mode
    # exercises the plan, the ledger, the checkpoints and the resume against
    # the real filesystem; it says nothing about the GPU and nothing about the
    # model cache, and pretending otherwise by demanding a real dependency
    # digest would make a pipeline check look like a trust check.
    stand_in = {"repositories": [], "instruction_pool": {}}
    dep_checker = lambda: {"ok": True, "problems": [], "evidence": stand_in}
    dep_value = dependency_digest(stand_in)
    # The runtime provenance a real run reads from the machine. Stood in for
    # here for the same reason the dependencies are: this mode touches no
    # device, so reporting the machine's real allocator state would dress a
    # pipeline check up as a measurement.
    alloc = "expandable_segments:True"
    det = {"use_deterministic_algorithms": True, "warn_only": False,
           "cudnn_benchmark": False, "cudnn_deterministic": True,
           "cublas_workspace_config": ":4096:8",
           "tf32_matmul_allowed": False, "tf32_cudnn_allowed": False,
           "seed": LoraConfig_().seed}
    try:
        try:
            gates.run_gate("gate_500", deps=gates.FakeGateDeps(rows=500),
                           run_dir=run_dir, pack_dir=pack_dir,
                           expected_pack_digest=digest,
                           expected_dependency_digest=dep_value,
                           dependency_checker=dep_checker,
                           allocator_config=alloc, determinism=det,
                           stop_after=200)
        except gates.DeliberateStop as stop:
            print(f"  stopped on purpose after row {stop.position}")
        point = gates.resume_point(run_dir, pack_dir=pack_dir,
                                   expected_pack_digest=digest,
                                   expected_dependency_digest=dep_value,
                                   dependency_checker=dep_checker,
                                   allocator_config=alloc, determinism=det)
        print(f"  resuming from checkpoint {point['resume_from']} "
              f"as attempt {point['attempt']}")
        evidence = gates.run_gate("gate_500", deps=gates.FakeGateDeps(rows=500),
                                  run_dir=run_dir, pack_dir=pack_dir,
                                  expected_pack_digest=digest,
                                  expected_dependency_digest=dep_value,
                                  dependency_checker=dep_checker,
                                  allocator_config=alloc, determinism=det,
                                  resume=True)
    except gates.GateRefused as exc:
        print(exc, file=sys.stderr)
        return 2

    problems = gates.gate_problems("gate_500", evidence)
    print(f"  rows completed : {evidence['rows_completed']}")
    print(f"  duplicates     : {evidence['duplicate_positions'] or 'none'}")
    print(f"  missing        : {evidence['missing_positions'] or 'none'}")
    print(f"  weights back   : {evidence['model_state_restored']}")
    print(f"  generator back : {evidence['rng_state_restored']}")
    print(f"  optimizer back : {evidence['optimizer_state_restored']}")
    print(f"\nverdict: {gates.verdict('gate_500', evidence)}")
    for problem in problems:
        print(f"  - {problem}")
    print("\nNote: this says nothing about the GPU. It says the pipeline, the "
          "ledger, the checkpoints and the resume work on this filesystem.")
    return 0 if not problems else 1


def _real_run(args, pack_dir: Path, data_root: Path, digest: str,
              dep_digest: str) -> int:
    if not args.run_dir:
        print("--gate and --resume need --run-dir", file=sys.stderr)
        return 2
    run_dir = Path(args.run_dir)

    result = _preflight(pack_dir, data_root, digest, dep_digest)
    if not result["passed"]:
        print("refusing to run: the node did not pass preflight.\n",
              file=sys.stderr)
        _print_preflight(result)
        print("\nThere is no flag that skips these checks. A run that starts "
              "on the wrong device, the wrong hardware, an unverified pack or "
              "a dataset that does not match its pin is not the run that was "
              "planned, and its numbers would be compared against ones it is "
              "not comparable with.", file=sys.stderr)
        return 2

    # The allocator config was inherited before this process started; the
    # preflight has already refused if it was absent, disabled or in conflict.
    # Read, never written.
    import os

    alloc, alloc_problems = gpu_node.allocator_config_from_env(dict(os.environ))
    if alloc_problems:
        for problem in alloc_problems:
            print(problem, file=sys.stderr)
        return 2

    # Strict determinism, before the model is built. Not warn_only, and
    # nothing here catches the RuntimeError an operation without a
    # deterministic kernel raises: a silent downgrade would answer the
    # repeatability question wrong while looking like it answered it right.
    import torch

    try:
        det = gpu_node.apply_determinism(torch, seed=LoraConfig_().seed,
                                         env=dict(os.environ))
    except RuntimeError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    print(f"allocator config : {alloc}")
    print(f"determinism      : {json.dumps(det, sort_keys=True)}")

    name = args.gate or (gates.read_plan(run_dir) or {}).get("gate")
    if name not in gates.GATES:
        print(f"cannot tell which gate {run_dir} is; pass --gate",
              file=sys.stderr)
        return 2

    try:
        evidence = gates.run_gate(
            name, deps=gates.ProductionGateDeps(device=gpu_node.REQUIRED_DEVICE),
            run_dir=run_dir, pack_dir=pack_dir,
            expected_pack_digest=digest,
            expected_dependency_digest=dep_digest,
            allocator_config=alloc, determinism=det, resume=args.resume,
            stop_after=args.stop_after)
    except gates.DeliberateStop as stop:
        print(f"stopped on purpose after row {stop.position}. The checkpoint "
              f"and ledger are in {run_dir}; resume with --resume.")
        return 0
    except gates.GateRefused as exc:
        print(exc, file=sys.stderr)
        return 2

    verdict = gates.verdict(name, evidence)
    # Named through the verifier, so the writer and the reader cannot drift
    # apart. This script never reads evidence back and never unlocks anything;
    # sharing the name is the whole of the relationship.
    from src.training import gate_suite

    out = gate_suite.evidence_path(run_dir, name)
    from src.training.session import write_once_json

    write_once_json(out, {"verdict": verdict, "evidence": evidence})
    print(json.dumps({"gate": name, "verdict": verdict,
                      "rows_completed": evidence["rows_completed"],
                      "seconds_per_row": evidence["seconds_per_row"],
                      "peak_vram_gb": evidence["peak_vram_gb"],
                      # Printed rather than only filed: this is the value the
                      # repeatability criterion is stated over, and reading it
                      # off the console is how two runs get compared without
                      # anyone deciding which fields to compare.
                      "trainable_digest": evidence["trainable_digest"],
                      "model_state_restored": evidence["model_state_restored"],
                      "rng_state_restored": evidence["rng_state_restored"],
                      "optimizer_state_restored":
                          evidence["optimizer_state_restored"],
                      "problems": gates.gate_problems(name, evidence)},
                     indent=2, ensure_ascii=False))
    return 0 if verdict == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
