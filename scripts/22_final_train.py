#!/usr/bin/env python3
"""Final training run on the execution node. H2, the whole split, once.

H2 was selected on held-out loss over the 320 frozen validation rows. This
script is the only way that selection becomes a trained model, and like the
arm runner before it, the interesting decisions were all made elsewhere:

* the configuration is ``hypotheses.config_for("H2")`` and nothing on this
  command line can change it -- no ``--rank``, no ``--lr``, no ``--epochs``,
  no ``--seed``, no ``--rows``, no ``--force``, no ``--unlock``;
* it starts from a freshly initialised adapter on the merged BrickGPT. There
  is no ``--resume``, no ``--from-adapter`` and no ``--stop-after``, so
  continuing H2's weights, optimizer state or generator is not something this
  script can be asked to do;
* the six gate runs are six explicit arguments, and the pack and dependency
  digests are carried in with no defaults;
* the allocator configuration is read from the live environment and the
  determinism settings are applied here and read back.

Order, before a model is built or anything is written:

1. the node preflight, in full;
2. the allocator configuration, inherited;
3. the determinism settings, applied and read back;
4. the six-role gate suite, through ``hypotheses.require_unlocked``;
5. the truncated-row count, checked after the load and before the plan.

Any of them failing stops the run. A failure leaves immutable failure
evidence and no ordinary evidence file, and nothing is retried.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import final_run, gate_suite, gates, gpu_node, pack  # noqa: E402

PACK_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Final training run: the selected arm, whole split, once.")
    ap.add_argument("--run-dir", metavar="DIR")
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


def main() -> int:
    args = build_parser().parse_args()

    if args.summary:
        print(json.dumps({"node": gpu_node.NODE_SPEC,
                          "final_run": final_run.summary(),
                          "runnable_here": False,
                          "why": ("The final run starts when this script is "
                                  "given a new run directory, both carried "
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

    runs = {role: getattr(args, role) for role in gate_suite.ROLES}
    missing = [role for role, value in runs.items() if not value]
    if missing:
        print("every one of the six gate roles must be given a run directory; "
              f"missing {missing}", file=sys.stderr)
        return 2

    pack_dir = Path(args.pack_dir)
    result = gpu_node.preflight(
        probe=gpu_node.probe(), pack_dir=pack_dir, data_root=Path(args.data_root),
        expected_pack_digest=args.expected_pack_digest,
        expected_dependency_digest=args.expected_dependency_digest)
    for name in sorted(result["checks"]):
        check = result["checks"][name]
        print(f"  [{'ok  ' if check['passed'] else 'FAIL'}] {name}: "
              f"{check['detail']}")
    if not result["passed"]:
        print("\nrefusing to run: the node did not pass preflight.",
              file=sys.stderr)
        return 2

    allocator_config, alloc_problems = gpu_node.allocator_config_from_env(
        dict(os.environ))
    if alloc_problems:
        for problem in alloc_problems:
            print(problem, file=sys.stderr)
        return 2

    import torch  # noqa: PLC0415

    try:
        determinism = gpu_node.apply_determinism(
            torch, seed=final_run.frozen_config().seed, env=dict(os.environ))
    except RuntimeError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    print(f"allocator config : {allocator_config}")
    print(f"determinism      : {json.dumps(determinism, sort_keys=True)}")

    carried = {"expected_pack_digest": args.expected_pack_digest,
               "expected_dependency_digest": args.expected_dependency_digest,
               "allocator_config": allocator_config,
               "determinism": determinism}

    suite = gate_suite.suite_problems(
        {role: Path(p) for role, p in runs.items()}, **carried)
    if suite:
        print("\nthe formal gate suite does not verify:", file=sys.stderr)
        for problem in suite:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    print("formal gate suite: verified (six roles)")
    print(f"final run        : {json.dumps(final_run.summary(), sort_keys=True)}")

    if args.verify:
        print("\n--verify stops here. No model was loaded and nothing was "
              "written.")
        return 0

    if not args.run_dir:
        print("--run-dir is required to run", file=sys.stderr)
        return 2

    run_dir = Path(args.run_dir)
    try:
        evidence = final_run.run_final(
            deps_factory=lambda cfg: gates.ProductionGateDeps(
                device=gpu_node.REQUIRED_DEVICE, cfg=cfg,
                source=final_run.DATA_SOURCE),
            run_dir=run_dir, pack_dir=pack_dir,
            gate_runs={role: Path(p) for role, p in runs.items()}, **carried)
    except (final_run.FinalRunRefused, gates.GateRefused) as exc:
        print(exc, file=sys.stderr)
        return 2
    except final_run.FinalRunFailed as exc:
        print(exc, file=sys.stderr)
        print(f"\nfailure evidence: {final_run.failure_path(run_dir).name}. "
              "Nothing is retried.", file=sys.stderr)
        return 1

    print(json.dumps({"run": evidence["run"], "arm": evidence["arm"],
                      "rows_completed": evidence["rows_completed"],
                      "optimizer_steps": evidence["optimizer_steps"],
                      "truncated_rows": evidence["truncated_rows"],
                      "losses_finite": evidence["losses_finite"],
                      "seconds_per_row": evidence["seconds_per_row"],
                      "peak_vram_gb": evidence["peak_vram_gb"],
                      "trainable_digest": evidence["trainable_digest"],
                      "operational_problems": evidence["operational_problems"],
                      "evidence_file": final_run.evidence_path(run_dir).name,
                      "note": "no winner is declared here"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
