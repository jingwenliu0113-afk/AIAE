#!/usr/bin/env python3
"""The frozen vision test: classification, detection, and the CV comparison.

    --validation     score the validation split; may be run any number of times
    --test           score the frozen test split. Opened once, on purpose
    --classification / --detection   which task; default both
    --checkpoint DIR the fitted classifier. Without it only the CV baseline
                     is scored, and the report says so

Everything about the test path is arranged so that a number cannot be produced
twice with different code and quoted as one result:

**The digests are required, not optional.**  ``--test`` refuses to run without
``--expected-data-digest`` and ``--expected-split-digest``.  A frozen test whose
data or split could have moved is not frozen.

**The output is written once.**  A test report is written with ``O_EXCL``; a
second run has to be given a different destination, which leaves both on the
record instead of overwriting the first.

**Real and synthetic never merge.**  Every classification metric is computed
per population and printed as two blocks.  In this archive renders outnumber
photographs nine to one, so a pooled figure would be a figure about renders
wearing a label that says otherwise.

**The two methods see the same images.**  The traditional-CV baseline and the
fitted network are scored on the same frozen items in the same order, by the
same code in :mod:`src.vision.metrics`.  Wall-clock time is recorded for both,
because "the CV baseline is faster" is a real finding and so is what it costs.

The detection population's ground-truth boxes carry no per-brick class, so its
per-class count error is reported as unavailable rather than computed against a
label this project would have had to invent.  The renders in that archive *do*
name their class in the filename, so they are scored separately and fully.

Scope: these numbers describe this public archive, these eight classes and
these capture conditions. Nothing here generalises to an arbitrary pile of
bricks, and no figure here may be placed beside a Phase 2 result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision import datasets
from src.vision.classes import CLASS_ORDER, UNKNOWN
from src.vision.cv_baseline import classify_array
from src.vision.detect import detect
from src.vision.metrics import (SCORE_CLASSIFIER_CONFIDENCE, Box,
                                classification_report, detection_report,
                                most_confused)
from src.vision.preprocess import ImageError, read_image
from src.vision.schema import LOW_CONFIDENCE, METHOD_CV, METHOD_LEARNED
from src.vision.split import TEST, VALIDATION, VisionSplit

RAW = ROOT / "data/raw/vision"
DEFAULT_OUT = ROOT / "runs/vision/eval"

EXIT_OK, EXIT_PROBLEM, EXIT_REFUSED = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--validation", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--classification", action="store_true")
    p.add_argument("--detection", action="store_true")
    p.add_argument("--checkpoint",
                   help="a fitted classifier directory; without it only the "
                        "CV baseline is scored")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"))
    p.add_argument("--raw-root", default=str(RAW))
    p.add_argument("--out", help="write the report here; refuses to overwrite")
    p.add_argument("--expected-data-digest", action="append",
                   metavar="KEY=SHA256")
    p.add_argument("--expected-split-digest", action="append",
                   metavar="KEY=SHA256")
    p.add_argument("--limit", type=int,
                   help="score only the first N items of each population; "
                        "for a quick check, and recorded in the report")
    p.add_argument("--json", action="store_true")
    return p


def _pairs(values) -> dict[str, str]:
    out = {}
    for pair in values or []:
        key, _, digest = pair.partition("=")
        if not digest:
            raise SystemExit(f"expected KEY=SHA256, got {pair!r}")
        out[key] = digest
    return out


def load(key: str, split_name: str, *, raw_root, data_digest=None,
         split_digest=None):
    manifest = datasets.read_manifest(RAW / f"{key}_manifest.json",
                                      expected_digest=data_digest)
    split = VisionSplit.load(RAW / f"{key}_split.json",
                             expected_digest=split_digest)
    split.check_no_leakage()
    records = datasets.records_from_manifest(manifest)
    chosen = [record for record in records
              if split.split_of_item(record.member) == split_name]
    root = Path(raw_root) / key
    return manifest, split, chosen, root


# ---------------------------------------------------------------------------
# Single-brick classification
# ---------------------------------------------------------------------------

def classify_population(records, root, *, predict, limit=None):
    """Run one classifier over one population and time it."""
    truth, predictions = [], []
    started = time.monotonic()
    skipped = []
    chosen = records[:limit] if limit else records
    for record in chosen:
        try:
            image = read_image(root / record.member)
        except ImageError as exc:
            skipped.append({"member": record.member, "reason": str(exc)})
            continue
        truth.append(record.part)
        predictions.append(predict(image.rgb))
    return truth, predictions, {
        "seconds": round(time.monotonic() - started, 3),
        "images": len(truth),
        "seconds_per_image": (round((time.monotonic() - started)
                                    / max(1, len(truth)), 5)),
        "unreadable": skipped,
    }


def classification_block(records, root, *, predict, method, limit=None) -> dict:
    out = {"method": method, "populations": {}}
    for population in sorted({record.population for record in records}):
        subset = [record for record in records
                  if record.population == population]
        subset.sort(key=lambda record: record.member)
        truth, predictions, timing = classify_population(
            subset, root, predict=predict, limit=limit)
        if not truth:
            out["populations"][population] = {
                "n": 0, "reason": "no readable image in this population"}
            continue
        report = classification_report(truth, predictions,
                                      population=population)
        report["timing"] = timing
        report["most_confused"] = most_confused(report)
        out["populations"][population] = report
    return out


# ---------------------------------------------------------------------------
# Multi-brick detection
# ---------------------------------------------------------------------------

def truth_boxes(record, root, *, per_class: bool):
    """Ground-truth boxes for one detection image."""
    label_path = root / datasets.label_for(record.member)
    annotation = datasets.parse_voc(label_path.read_text(encoding="utf-8"))
    boxes = []
    for box in annotation.boxes:
        label = record.part if (per_class and record.part) else None
        boxes.append(Box(box.x0, box.y0, box.x1, box.y1, label=label))
    return boxes, annotation


def detection_block(records, root, *, classify=None, limit=None) -> dict:
    out = {}
    for population in sorted({record.population for record in records}):
        subset = sorted((record for record in records
                         if record.population == population),
                        key=lambda record: record.member)
        if limit:
            subset = subset[:limit]
        # Only the renders carry a per-brick class, from their own filename.
        per_class = population == datasets.POPULATION_SYNTHETIC
        images = []
        started = time.monotonic()
        skipped = []
        diagnostics = {"merged_or_split": 0, "empty_predictions": 0,
                       "unknown_labels": 0, "low_confidence": 0}
        for record in subset:
            try:
                image = read_image(root / record.member)
                truth, annotation = truth_boxes(record, root,
                                                per_class=per_class)
            except (ImageError, datasets.DatasetError, OSError) as exc:
                skipped.append({"member": record.member, "reason": str(exc)})
                continue
            result = detect(image.rgb, classify=classify)
            predicted = [d.as_box() for d in result.detections]
            if not predicted:
                diagnostics["empty_predictions"] += 1
            if len(predicted) != len(truth):
                diagnostics["merged_or_split"] += 1
            diagnostics["unknown_labels"] += sum(
                1 for d in result.detections if d.label == UNKNOWN)
            diagnostics["low_confidence"] += len(
                result.low_confidence_items())
            images.append((truth, predicted))
        if not images:
            out[population] = {"images": 0,
                               "reason": "no readable image", "skipped": skipped}
            continue
        # Every predicted box comes out of the shared deterministic stage one
        # and carries the stage-two classifier's confidence as its score. Both
        # facts are declared, because they decide what the average precision
        # below is a measurement of -- and what it is not.
        report = detection_report(
            images, population=population, per_class_truth=per_class,
            score_semantics=SCORE_CLASSIFIER_CONFIDENCE,
            stage_one_shared=True)
        report["images"] = len(images)
        report["timing"] = {
            "seconds": round(time.monotonic() - started, 3),
            "seconds_per_image": round(
                (time.monotonic() - started) / len(images), 5)}
        report["behaviour"] = diagnostics
        report["unreadable"] = skipped
        out[population] = report
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _percent(value) -> str:
    return "   n/a" if value != value else f"{value * 100:6.2f}%"


def print_report(report: dict) -> None:
    print("=" * 76)
    print(f"BrickAgain vision evaluation — split: {report['split']}")
    print("=" * 76)
    print(report["scope"])
    print()
    for key, block in report.get("classification", {}).items():
        if not isinstance(block, dict):
            print(f"--- classification / {key} ---")
            print(f"  {block}")
            print()
            continue
        print(f"--- classification / {key} ---")
        for population, body in block["populations"].items():
            if not body.get("n"):
                print(f"  {population}: {body.get('reason')}")
                continue
            print(f"  {population:10s} n={body['n']:5d}  "
                  f"acc={_percent(body['accuracy'])}  "
                  f"forced={_percent(body['forced_top1_accuracy'])}  "
                  f"top3={_percent(body['top3_accuracy'])}  "
                  f"macroF1={_percent(body['macro_f1'])}  "
                  f"coverage={_percent(body['coverage'])}  "
                  f"{body['timing']['seconds_per_image'] * 1000:.1f} ms/img")
            for part in CLASS_ORDER:
                row = body["per_class"][part]
                print(f"      {part:4s} support={row['support']:5d} "
                      f"P={_percent(row['precision'])} "
                      f"R={_percent(row['recall'])} "
                      f"F1={_percent(row['f1'])} "
                      f"abstained={row['abstained']}")
            worst = body["most_confused"][:4]
            if worst:
                print("      most confused: " + ", ".join(
                    f"{cell['truth']}->{cell['predicted']}×{cell['count']}"
                    for cell in worst))
        print()
    for key, block in report.get("detection", {}).items():
        if not isinstance(block, dict):
            continue
        print(f"--- detection / {key} ---")
        for population, body in block.items():
            if not body.get("images"):
                print(f"  {population}: {body.get('reason')}")
                continue
            headline = body.get(
                "class_agnostic_ap50_by_classifier_confidence")
            label = "class-agnostic AP@50 (by classifier confidence)"
            if headline is None:
                headline = body["average_precision_50"]
                label = "AP@50"
            print(f"  {population:10s} images={body['images']:4d}  "
                  f"{label}={_percent(headline)}  "
                  f"P={_percent(body['precision'])}  "
                  f"R={_percent(body['recall'])}  "
                  f"count MAE={body['count_absolute_error_mean']:.3f}  "
                  f"signed={body['count_error_mean']:+.3f}")
            if body.get("not_a_detector_comparison"):
                print("      " + body["not_a_detector_comparison"])
            if body["per_class_count_mae"] is None:
                print(f"      per-class count error and 8-class mAP@50: "
                      f"unavailable — {body['per_class_unavailable_reason']}")
            else:
                print("      per-class count MAE: " + "  ".join(
                    f"{part}:{value:.2f}"
                    for part, value in body["per_class_count_mae"].items()))
        print()
    if report.get("comparison"):
        print("--- traditional CV against the fitted network, same items ---")
        for line in report["comparison"]["lines"]:
            print("  " + line)
        print()
    for note in report["notes"]:
        print(f"* {note}")


def comparison(report: dict) -> dict | None:
    """Side-by-side lines for the two classification methods."""
    blocks = report.get("classification") or {}
    if METHOD_CV not in blocks or METHOD_LEARNED not in blocks:
        return None
    lines = []
    rows = {}
    for population in sorted(set(blocks[METHOD_CV]["populations"])
                             & set(blocks[METHOD_LEARNED]["populations"])):
        cv = blocks[METHOD_CV]["populations"][population]
        learned = blocks[METHOD_LEARNED]["populations"][population]
        if not cv.get("n") or not learned.get("n"):
            continue
        if cv["n"] != learned["n"]:
            lines.append(
                f"{population}: refusing to compare — {cv['n']} items scored "
                f"by the CV baseline against {learned['n']} by the network")
            continue
        rows[population] = {
            "n": cv["n"],
            "cv_accuracy": cv["accuracy"],
            "learned_accuracy": learned["accuracy"],
            "cv_forced_top1": cv["forced_top1_accuracy"],
            "learned_forced_top1": learned["forced_top1_accuracy"],
            "cv_coverage": cv["coverage"],
            "learned_coverage": learned["coverage"],
            "cv_macro_f1": cv["macro_f1"],
            "learned_macro_f1": learned["macro_f1"],
            "cv_top3": cv["top3_accuracy"],
            "learned_top3": learned["top3_accuracy"],
            "cv_ms_per_image": cv["timing"]["seconds_per_image"] * 1000,
            "learned_ms_per_image":
                learned["timing"]["seconds_per_image"] * 1000,
        }
        lines.append(
            f"{population:10s} n={cv['n']:5d}\n"
            f"      accuracy (abstention counts as wrong): "
            f"CV {_percent(cv['accuracy'])} vs network "
            f"{_percent(learned['accuracy'])}\n"
            f"      forced top-1 (threshold ignored):      "
            f"CV {_percent(cv['forced_top1_accuracy'])} vs network "
            f"{_percent(learned['forced_top1_accuracy'])}\n"
            f"      coverage (how often it answered):      "
            f"CV {_percent(cv['coverage'])} vs network "
            f"{_percent(learned['coverage'])}\n"
            f"      top-3:                                 "
            f"CV {_percent(cv['top3_accuracy'])} vs network "
            f"{_percent(learned['top3_accuracy'])}\n"
            f"      macro F1:                              "
            f"CV {_percent(cv['macro_f1'])} vs network "
            f"{_percent(learned['macro_f1'])}\n"
            f"      speed:                                 "
            f"CV {rows[population]['cv_ms_per_image']:.1f} ms vs network "
            f"{rows[population]['learned_ms_per_image']:.1f} ms per image")
    if not rows and not lines:
        return None
    return {
        "rows": rows, "lines": lines, "same_items": True,
        "note": ("both methods were run over the same frozen items in the "
                 "same order by the same scoring code. Neither figure is "
                 "hidden because the other is better, and the speed column "
                 "does not excuse an accuracy column"),
        "read_both_columns": (
            "the two accuracy rows measure different things and both are "
            "needed. One confidence threshold is shared by both methods -- a "
            "per-method threshold tuned on the data the methods are compared "
            "on would make the comparison about the thresholds. The CV "
            "baseline's scores are diffuse, so under that shared rule it "
            "declines most items and its accuracy row is dominated by the "
            "abstentions; its forced top-1 row is the like-for-like number. "
            "That it abstains this much is itself a finding about the method "
            "on this data, not a presentation problem to be tuned away."),
    }


def write_once(path: Path, body: dict) -> None:
    """Write a report, refusing to overwrite one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(body, indent=2, ensure_ascii=False,
                         sort_keys=True).encode("utf-8")
    try:
        handle = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise SystemExit(
            f"refused: {path} already exists. A frozen-test report is not "
            "overwritten; give a different --out so both stay on the record")
    with os.fdopen(handle, "wb") as file:
        file.write(payload)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.validation == args.test:
        print("choose exactly one of --validation and --test",
              file=sys.stderr)
        return EXIT_REFUSED
    split_name = TEST if args.test else VALIDATION
    keys = []
    if args.classification or not (args.classification or args.detection):
        keys.append(datasets.CLASSIFICATION.key)
    if args.detection or not (args.classification or args.detection):
        keys.append(datasets.DETECTION.key)
    data_digests = _pairs(args.expected_data_digest)
    split_digests = _pairs(args.expected_split_digest)

    if args.test:
        missing = [key for key in keys
                   if key not in data_digests or key not in split_digests]
        if missing:
            print(f"refused: --test requires --expected-data-digest and "
                  f"--expected-split-digest for {missing}. A frozen test whose "
                  "data or split could have moved is not frozen.",
                  file=sys.stderr)
            return EXIT_REFUSED

    model = manifest = device = None
    if args.checkpoint:
        from src.vision.model import load as load_model
        from src.vision.model import predict_arrays

        model, manifest, device = load_model(args.checkpoint,
                                             device=args.device)

    report: dict = {
        "kind": "brickagain.vision_eval",
        "split": split_name,
        "limit": args.limit,
        "checkpoint": args.checkpoint,
        "checkpoint_manifest": ({
            "weights_sha256": manifest["weights"]["sha256"],
            "data_manifest_sha256": manifest["data_manifest_sha256"],
            "split_manifest_sha256": manifest.get("split_manifest_sha256"),
            "code_sha256": manifest["code_sha256"],
            "selected_epoch": manifest["selected_epoch"],
            "selection_criterion": manifest["selection_criterion"],
            "config": manifest["config"],
            "dependencies": manifest["dependencies"],
            "fitted_on_device": manifest["device"],
        } if manifest else None),
        "inference_device": device,
        "low_confidence_threshold": LOW_CONFIDENCE,
        "scope": (
            "this public archive, these eight classes and these capture "
            "conditions. It does not generalise to an arbitrary pile of "
            "bricks, and no figure here may be placed beside a Phase 2 "
            "result."),
        "notes": [
            "an abstention counts as wrong and is never folded into a class; "
            "coverage is reported beside accuracy",
            "real photographs and renders are scored separately and never "
            "pooled",
            "connectivity, grounding and stability are not part of this "
            "evaluation; it measures image recognition only",
        ],
        "classification": {},
        "detection": {},
    }

    try:
        for key in keys:
            manifest_body, split, records, root = load(
                key, split_name, raw_root=args.raw_root,
                data_digest=data_digests.get(key),
                split_digest=split_digests.get(key))
            report.setdefault("data", {})[key] = {
                "data_manifest_sha256": datasets.manifest_digest(
                    manifest_body),
                "split_manifest_sha256": split.digest(),
                "items": len(records),
            }
            if key == datasets.CLASSIFICATION.key:
                labelled = [record for record in records if record.part]
                report["classification"][METHOD_CV] = classification_block(
                    labelled, root, predict=classify_array, method=METHOD_CV,
                    limit=args.limit)
                if model is not None:
                    def predict_one(rgb, _model=model, _device=device):
                        from src.vision.model import predict_arrays

                        return predict_arrays([_model][0], [rgb],
                                              device=_device)[0]

                    report["classification"][METHOD_LEARNED] = \
                        classification_block(
                            labelled, root, predict=predict_one,
                            method=METHOD_LEARNED, limit=args.limit)
                else:
                    report["classification"]["learned_absent_reason"] = (
                        "no --checkpoint was given, so only the traditional "
                        "CV baseline was scored. This is not a comparison")
            else:
                report["detection"][METHOD_CV] = detection_block(
                    records, root, classify=None, limit=args.limit)
                if model is not None:
                    from src.vision.detect import learned_classifier

                    report["detection"][METHOD_LEARNED] = detection_block(
                        records, root,
                        classify=learned_classifier(model, device),
                        limit=args.limit)
        report["comparison"] = comparison(report)
    except (datasets.DatasetError, OSError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    if args.out:
        write_once(Path(args.out), report)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print_report(report)
        if args.out:
            print(f"\nwritten to {args.out}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
