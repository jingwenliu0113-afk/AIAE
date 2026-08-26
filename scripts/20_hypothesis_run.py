#!/usr/bin/env python3
"""Run one frozen hypothesis arm on the execution node. One arm, one process.

H1 and H2 are declared in ``src/training/hypotheses.py``. This script is the
only way either becomes a run, and it is built so that the interesting
decisions were all made somewhere else:

* the configuration comes from ``hypotheses.config_for`` and nothing on this
  command line can change it. There is no ``--rank``, no ``--lr``, no
  ``--dtype``, no ``--rows``, no ``--seed``, no ``--config``, and no
  ``--force``, ``--unlock`` or ``--skip-preflight``;
* the six gate runs are six explicit arguments. Nothing is globbed, sorted,
  listed or chosen by being the newest -- which run plays which role is a
  thing the operator says out loud;
* the pack digest and the dependency digest are carried in and have no
  defaults, because each checks something the thing being checked cannot
  establish about itself;
* the allocator configuration is read from the live environment, never
  written by this process, and the determinism settings are applied here and
  **read back** rather than copied from a gate's plan. Provenance taken from
  the run it is supposed to be describing describes nothing.

Modes::

  --summary                    prints the frozen declaration and stops.
                               Loads no model.
  --verify                     preflight, then the six-role suite verifier.
                               Loads no model. Answers "would an arm start?"
  --arm H1|H2 --run-dir DIR    runs that arm, after all of the above passes.
  --resume                     continues an interrupted arm from its last
                               checkpoint, re-verifying everything first.

Order, before a model is built or anything is written:

1. the node preflight, in full, which also verifies the pack file by file and
   this machine's dependency bytes against the carried digest;
2. the allocator configuration, read from the environment this process
   inherited;
3. the determinism settings, applied in this process and read back;
4. the six-role gate suite, through ``hypotheses.require_unlocked``.

Any of them failing stops the run before the loader is called.

Nothing here decides which arm won. It produces two replayable measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import arms, gate_suite, gates, gpu_node, pack  # noqa: E402
from src.training.longrun import dependency_digest  # noqa: E402

PACK_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run one frozen hypothesis arm on the execution node.")
    ap.add_argument("--arm", choices=arms.ARM_NAMES)
    ap.add_argument("--run-dir", metavar="DIR")
    ap.add_argument("--previous-arm-run-dir", metavar="DIR",
                    help="the run directory of the arm before this one in the "
                         "frozen order. Required for H2 and refused for H1: "
                         "the order is H1 then H2, once each, and H2 means "
                         "after a complete H1 rather than merely later.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--pack-dir", metavar="DIR", default=str(PACK_ROOT))
    ap.add_argument("--data-root", metavar="DIR", default=str(PACK_ROOT))
    ap.add_argument("--expected-pack-digest", metavar="SHA256",
                    help="the pack_digest the build machine printed, carried "
                         "here separately. Required; 64 lowercase hex.")
    ap.add_argument("--expected-dependency-digest", metavar="SHA256",
                    help="the dependency_digest the build machine printed, "
                         "carried here separately. Required; 64 lowercase hex.")
    for role in gate_suite.ROLES:
        ap.add_argument(f"--{role.replace('_', '-')}", metavar="DIR",
                        help=f"the run directory that played {role}")
    return ap


def _roles_from(args) -> dict:
    """The six, by name. No discovery of any kind."""
    return {role: getattr(args, role) for role in gate_suite.ROLES}


def _missing_roles(runs) -> list[str]:
    return [role for role, value in runs.items() if not value]


def main() -> int:
    args = build_parser().parse_args()

    if args.summary:
        print(json.dumps({"node": gpu_node.NODE_SPEC,
                          "arms": arms.summary()["arms"],
                          "contract": arms.summary(),
                          "arms_runnable_here": False,
                          "why": ("An arm runs only when this script is given "
                                  "one arm, a new run directory, both carried "
                                  "digests and the six gate runs, and only "
                                  "after preflight and the suite verifier "
                                  "pass. Printing the declaration is not "
                                  "running it.")},
                         indent=2, ensure_ascii=False))
        return 0

    for name, value in (("--expected-pack-digest", args.expected_pack_digest),
                        ("--expected-dependency-digest",
                         args.expected_dependency_digest)):
        problems = pack.expected_digest_problems(value, what=f"{name} value")
        if problems:
            print(f"{name} is required and must be 64 lowercase hex: "
                  + "; ".join(problems), file=sys.stderr)
            return 2

    runs = _roles_from(args)
    missing = _missing_roles(runs)
    if missing:
        print("every one of the six gate roles must be given a run directory; "
              f"missing {missing}", file=sys.stderr)
        return 2

    pack_dir = Path(args.pack_dir)
    data_root = Path(args.data_root)

    # 1. The node preflight, in full. It verifies the pack file by file, this
    #    machine's dependency bytes against the carried digest, the device,
    #    the dtype, the offline pins and the allocator configuration -- all
    #    without loading a tensor or initialising the device.
    result = gpu_node.preflight(
        probe=gpu_node.probe(), pack_dir=pack_dir, data_root=data_root,
        expected_pack_digest=args.expected_pack_digest,
        expected_dependency_digest=args.expected_dependency_digest)
    for name in sorted(result["checks"]):
        check = result["checks"][name]
        print(f"  [{'ok  ' if check['passed'] else 'FAIL'}] {name}: "
              f"{check['detail']}")
    if not result["passed"]:
        print("\nrefusing to run: the node did not pass preflight. There is "
              "no flag that skips it.", file=sys.stderr)
        return 2

    # 2. The allocator, read from what this process inherited. Never written
    #    here: a run that configured its own allocator could not be told from
    #    one that inherited a different setting.
    allocator_config, alloc_problems = gpu_node.allocator_config_from_env(
        dict(os.environ))
    if alloc_problems:
        for problem in alloc_problems:
            print(problem, file=sys.stderr)
        return 2

    # 3. Determinism, applied in *this* process and read back. Not copied from
    #    a gate's plan: what a gate ran under is not evidence about what this
    #    process is running under.
    import torch  # noqa: PLC0415

    try:
        determinism = gpu_node.apply_determinism(
            torch, seed=arms.shared_seed(), env=dict(os.environ))
    except RuntimeError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    print(f"allocator config : {allocator_config}")
    print(f"determinism      : {json.dumps(determinism, sort_keys=True)}")

    carried = {"expected_pack_digest": args.expected_pack_digest,
               "expected_dependency_digest": args.expected_dependency_digest,
               "allocator_config": allocator_config,
               "determinism": determinism}

    # 4. The six-role gate suite.
    suite = gate_suite.suite_problems({role: Path(p) for role, p in runs.items()},
                                      **carried)
    if suite:
        print("\nthe formal gate suite does not verify:", file=sys.stderr)
        for problem in suite:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    print("formal gate suite: verified (six roles)")

    if args.verify:
        print("\n--verify stops here. No model was loaded and nothing was "
              "written.")
        return 0

    if not args.arm or not args.run_dir:
        print("--arm and --run-dir are both required to run an arm",
              file=sys.stderr)
        return 2

    previous = (Path(args.previous_arm_run_dir)
                if args.previous_arm_run_dir else None)
    ordering = arms.predecessor_problems(args.arm, previous_run_dir=previous,
                                         **carried)
    if ordering:
        print(f"refusing to run {args.arm}:", file=sys.stderr)
        for problem in ordering:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    run_dir = Path(args.run_dir)
    try:
        evidence = arms.run_arm(
            args.arm,
            deps_factory=lambda cfg: gates.ProductionGateDeps(
                device=gpu_node.REQUIRED_DEVICE, cfg=cfg),
            run_dir=run_dir, pack_dir=pack_dir,
            gate_runs={role: Path(p) for role, p in runs.items()},
            previous_run_dir=previous, resume=args.resume, **carried)
    except gates.DeliberateStop as stop:
        print(f"stopped after row {stop.position}; resume with --resume.")
        return 0
    except (arms.ArmRefused, gates.GateRefused) as exc:
        print(exc, file=sys.stderr)
        return 2
    except arms.ArmFailed as exc:
        # The failure record is already on disk, written by the runner and
        # write-once. Nothing here publishes ordinary evidence for it, and
        # nothing here starts the other arm.
        print(exc, file=sys.stderr)
        print(f"\nfailure evidence: "
              f"{arms.failure_path(run_dir, args.arm).name}. The sequence "
              "stops here.", file=sys.stderr)
        return 1

    # ``run_arm`` published the evidence itself, once, after the operational
    # checks passed. This prints; it does not decide.
    print(json.dumps({"arm": evidence["arm"],
                      "rows_completed": evidence["rows_completed"],
                      "optimizer_steps": evidence["optimizer_steps"],
                      "losses_finite": evidence["losses_finite"],
                      "seconds_per_row": evidence["seconds_per_row"],
                      "peak_vram_gb": evidence["peak_vram_gb"],
                      "trainable_digest": evidence["trainable_digest"],
                      "ledger_problems": evidence["ledger_problems"],
                      "operational_problems": evidence["operational_problems"],
                      "evidence_file": arms.evidence_path(
                          run_dir, args.arm).name,
                      "note": "no winner is declared here"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
