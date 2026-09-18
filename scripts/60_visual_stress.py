#!/usr/bin/env python3
"""V1 synthetic visual stress test: the way to reach the frozen contract.

The contract is :mod:`src.eval.visual_stress`, frozen before any image is
scored. This is the door.

``--manifest``
    Build the frozen manifest -- the scenes, the condition matrix, the ground
    truth, every image's pinned digest, the acceptance and rejection rules,
    the metric list and the source digests -- and write it write-once. It
    prints digests and counts, never an image.

``--render``
    Write every image the manifest names. Existing files are left alone and
    counted; nothing is overwritten.

``--run``
    Analyse every accepted image through ``src.ui.photo.analyse_photo`` -- the
    interface's own entry point, so the UI flow is exercised rather than
    described -- and write the per-image results. An image whose stored
    pixels do not reproduce the manifest's digest is rejected and not scored.

    The **manifest** decides which checkpoint and which device are used.
    ``--checkpoint`` and ``--device`` are compared to it and refused when
    they differ; they are not overrides. Passing them is how a mistake
    becomes a loud refusal instead of a run against weights nobody recorded.

``--summarise``
    Recompute every reported number from the per-image results and write the
    summary. Rerunning it on unchanged results rewrites nothing; running it
    against changed results is refused rather than overwritten.

``--report``
    Render the summary. Derives nothing, and refuses to write a report that
    calls a software render a photograph.

``--all``
    The five in order, which is the only order they work in.

Usage::

  ./.venv/bin/python scripts/60_visual_stress.py --all \\
      --out-dir runs/visual_stress/cv --method cv-baseline
  ./.venv/bin/python scripts/60_visual_stress.py --all \\
      --out-dir runs/visual_stress/learned --method transfer-resnet18 \\
      --checkpoint runs/vision/classifier
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import visual_stress as vs  # noqa: E402
from src.eval.acceptance import PlanRefused  # noqa: E402


def _refuse(problems, headline: str) -> int:
    print(headline, file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 2


def _manifest_path(args) -> Path:
    return Path(args.out_dir) / vs.MANIFEST_NAME


def _load_manifest(args) -> dict:
    path = _manifest_path(args)
    if not path.is_file():
        raise PlanRefused(f"{path} is not here; run --manifest first")
    body = json.loads(path.read_text())
    problems = vs.manifest_problems(body, root=ROOT)
    if problems:
        raise PlanRefused("this manifest does not hold:\n  - "
                          + "\n  - ".join(problems))
    return body


def mode_manifest(args) -> int:
    path = _manifest_path(args)
    try:
        body = vs.build_manifest(method=args.method, root=ROOT,
                                 checkpoint=args.checkpoint,
                                 device=args.device)
    except PlanRefused as exc:
        return _refuse([str(exc)], "refusing to build the manifest:")
    problems = vs.manifest_problems(body, root=ROOT)
    if problems:
        return _refuse(problems, "the manifest this run produces is wrong:")
    try:
        written = vs.write_once_json_checked(path, body, what="manifest")
    except PlanRefused as exc:
        return _refuse([str(exc)], "refusing to replace a frozen manifest:")
    recognition = body["recognition"]
    print(json.dumps({
        "manifest_digest": body["manifest_digest"],
        "source_manifest_digest": body["source_manifest_digest"],
        "method": recognition["method"],
        "checkpoint": recognition["checkpoint"],
        "checkpoint_digest": (
            (recognition["checkpoint_manifest"] or {}).get(
                "checkpoint_digest")),
        "device_requested": recognition["device_requested"],
        "device_resolved":
            recognition["inference_environment"]["device_resolved"],
        "scenes": body["scenes"], "n_images": body["n_images"],
        "conditions": len(body["condition_ids"]),
        "tier": body["tier"],
        "v2_v3": {k: body["tiers"][k] for k in
                  ("v2_validated_ai_variations", "v3_pure_ai_hard_cases")},
        "written": written,
    }, indent=2))
    return 0


def mode_render(args) -> int:
    try:
        manifest = _load_manifest(args)
    except PlanRefused as exc:
        return _refuse([str(exc)], "refusing to render:")
    counts = vs.render_images(Path(args.out_dir), manifest)
    accepted, rejected = vs.partition(Path(args.out_dir), manifest)
    print(json.dumps({**counts, "accepted": len(accepted),
                      "rejected": len(rejected),
                      "rejected_reasons": [r["reason"] for r in rejected[:5]]},
                     indent=2))
    return 0 if not rejected else 2


def mode_run(args) -> int:
    try:
        manifest = _load_manifest(args)
    except PlanRefused as exc:
        return _refuse([str(exc)], "refusing to run:")
    out = Path(args.out_dir) / vs.RESULTS_NAME
    # ``--checkpoint`` and ``--device`` are passed so a mismatch is refused
    # loudly; ``run`` takes the values it uses from the manifest.
    try:
        record = vs.run(Path(args.out_dir), manifest,
                        checkpoint=args.checkpoint, device=args.device,
                        root=ROOT)
    except PlanRefused as exc:
        return _refuse([str(exc)],
                       "refusing to run against an unpinned runtime:")
    # Whole-record comparison. Comparing ``results`` alone let the accepted
    # list, the rejected list and the safe-rejection probe change unnoticed.
    try:
        vs.write_once_json_checked(out, record, what="result set")
    except PlanRefused as exc:
        return _refuse([str(exc)],
                       "refusing to overwrite an existing result set:")
    print(json.dumps({"scored": len(record["results"]),
                      "accepted": len(record["accepted"]),
                      "rejected": len(record["rejected"]),
                      "all_probes_refused":
                          record["safe_rejection_shared_input_layer"][
                              "all_refused_by_name"]},
                     indent=2))
    return 0


def mode_summarise(args) -> int:
    try:
        manifest = _load_manifest(args)
    except PlanRefused as exc:
        return _refuse([str(exc)], "refusing to summarise:")
    results_path = Path(args.out_dir) / vs.RESULTS_NAME
    if not results_path.is_file():
        return _refuse([f"{results_path} is not here"],
                       "refusing to summarise:")
    results = json.loads(results_path.read_text())
    if results.get("manifest_digest") != manifest["manifest_digest"]:
        return _refuse(["the results are for a different manifest"],
                       "refusing to summarise:")
    summary = vs.summarise(results, manifest)
    out = Path(args.out_dir) / vs.SUMMARY_NAME
    try:
        vs.write_once_json_checked(out, summary, what="summary")
    except PlanRefused as exc:
        return _refuse([str(exc)],
                       "refusing to overwrite an existing summary:")
    overall = summary["overall"]
    print(json.dumps({
        "summary_digest": summary["summary_digest"],
        "scored": summary["n_scored"], "rejected": summary["n_rejected"],
        "inventory_exact_match": overall["inventory_exact_match"]["value"],
        "oracle_correction_path_valid":
            overall["oracle_correction_path_valid"]["value"],
        "part_errors": overall["part_errors"],
        "colour_errors": overall["colour_errors"],
        "false_positives": overall["false_positives"],
        "false_negatives": overall["false_negatives"],
    }, indent=2))
    return 0


def mode_report(args) -> int:
    from src.eval import visual_stress_report

    try:
        written = visual_stress_report.write_report(Path(args.out_dir))
    except PlanRefused as exc:
        return _refuse([str(exc)], "refusing to report:")
    print(json.dumps(written, indent=2, default=str))
    return 0


def mode_verify(args) -> int:
    """Re-derive everything in a run directory and say what does not match."""
    from src.eval import visual_stress_report

    try:
        outcome = visual_stress_report.verify(Path(args.out_dir), root=ROOT)
    except PlanRefused as exc:
        return _refuse([str(exc)], "refusing to verify:")
    print(json.dumps(outcome, indent=2))
    return 0 if outcome["verified"] else 2


def mode_all(args) -> int:
    for step in (mode_manifest, mode_render, mode_run, mode_summarise,
                 mode_report, mode_verify):
        code = step(args)
        if code != 0:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="V1 synthetic visual stress test")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="re-derive every artefact here and compare it, "
                         "including re-running the pinned model over every "
                         "accepted image and comparing its answers")
    ap.add_argument("--all", action="store_true",
                    help="the five above, in the only order that works")
    ap.add_argument("--out-dir", metavar="DIR", default=vs.DEFAULT_DIR)
    ap.add_argument("--method", metavar="NAME", default=vs.METHOD_CV,
                    choices=list(vs.METHODS))
    ap.add_argument("--checkpoint", metavar="DIR",
                    help=f"required by --method {vs.METHOD_LEARNED}")
    ap.add_argument("--device", metavar="NAME")
    return ap


MODES: tuple[str, ...] = ("manifest", "render", "run", "summarise", "report",
                          "verify", "all")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    chosen = [m for m in MODES if getattr(args, m)]
    if len(chosen) != 1:
        print(f"choose exactly one of {['--' + m for m in MODES]}",
              file=sys.stderr)
        return 2
    if args.method == vs.METHOD_LEARNED and not args.checkpoint:
        print(f"--method {vs.METHOD_LEARNED} needs --checkpoint",
              file=sys.stderr)
        return 2
    return {"manifest": mode_manifest, "render": mode_render,
            "run": mode_run, "summarise": mode_summarise,
            "report": mode_report, "verify": mode_verify,
            "all": mode_all}[chosen[0]](args)


if __name__ == "__main__":
    raise SystemExit(main())
