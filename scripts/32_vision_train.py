#!/usr/bin/env python3
"""Fit the eight-class head on the public single-brick archive.

    --fetch-backbone   download the pinned backbone once; the only mode that
                       touches the network
    --check            report whether a strict-offline fit would work
    --smoke            a few steps on a small subset, to prove the code path
                       runs. Explicitly labelled a smoke: it is not a result
    --run              the real fit
    --selection-record write the machine-checkable record of how the returned
                       checkpoint was fitted and which epoch was kept
    --verify-selection re-derive that record from the artefacts and report
                       every disagreement

The real fit belongs on the CUDA node.  This project's compute split sends
model training to the Windows RTX 5070 Ti and keeps data preparation, the
frozen split, the acceptance code and the reports on the Mac; a Mac run of
``--run`` is possible and is what ``--smoke`` uses a small version of, but the
official fit is the one the node performs from a private pack.

Whatever runs it, the same three things are true: the test split is never
opened, epoch selection is on validation only, and the checkpoint carries the
seed, configuration, data digest, split digest, code digest, dependency
versions and per-epoch log without which a result cannot be checked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision import datasets
from src.vision.model import (Augmentation, ModelError, TrainConfig,
                              check_backbone_cache, device_report,
                              resolve_device, suggested_batch_size,
                              tuning_for)
from src.vision.model_ids import CLASSIFIER_BACKBONE
from src.vision.selection import RECORD_FILE, SelectionError
from src.vision.selection import build as build_selection
from src.vision.selection import read as read_selection
from src.vision.selection import verify as verify_selection
from src.vision.selection import write as write_selection
from src.vision.split import TRAIN, VALIDATION, VisionSplit
from src.vision.train import (DEFAULT_REAL_WEIGHT, TrainError,
                              load_split_items, run)

RAW = ROOT / "data/raw/vision"
DEFAULT_OUT = ROOT / "runs/vision/classifier"

EXIT_OK, EXIT_PROBLEM, EXIT_REFUSED = 0, 1, 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fetch-backbone", action="store_true",
                   help="download the pinned backbone (network); one-off")
    p.add_argument("--check", action="store_true",
                   help="report offline readiness and split sizes")
    p.add_argument("--smoke", action="store_true",
                   help="a few steps on a small subset; not a result")
    p.add_argument("--run", action="store_true", help="the real fit")
    p.add_argument("--selection-record", action="store_true",
                   help="write selection_record.json beside the checkpoint")
    p.add_argument("--verify-selection", action="store_true",
                   help="re-derive the selection record and report problems")
    p.add_argument("--data-manifest",
                   default=str(RAW / "classification_manifest.json"))
    p.add_argument("--split", default=str(RAW / "classification_split.json"))
    p.add_argument("--raw-root", default=str(RAW))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--expected-data-digest")
    p.add_argument("--expected-split-digest")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int,
                   help="default: derived from the device's reported memory "
                        "(128 on a 16 GB card, 64 on 8 GB, 32 otherwise)")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trainable-stages", type=int, default=1)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--real-weight", type=float, default=DEFAULT_REAL_WEIGHT)
    p.add_argument("--steps-per-epoch", type=int)
    p.add_argument("--no-augmentation", action="store_true")
    p.add_argument("--deterministic", action="store_true",
                   help="turn off bfloat16 autocast, TF32 and cuDNN "
                        "autotuning, trading throughput for a bit-identical "
                        "re-run on the same device")
    p.add_argument("--full-backbone", action="store_true",
                   help="fit every parameter instead of the head and the last "
                        "stage; slower and, on a few hundred real "
                        "photographs, likely to overfit")
    p.add_argument("--json", action="store_true")
    return p


def config_from(args) -> TrainConfig:
    batch = (args.batch_size if args.batch_size is not None
             else suggested_batch_size(resolve_device(args.device)))
    return TrainConfig(
        epochs=args.epochs, batch_size=batch,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        seed=args.seed, freeze_backbone=not args.full_backbone,
        trainable_stages=args.trainable_stages,
        label_smoothing=args.label_smoothing,
        augmentation=Augmentation(enabled=not args.no_augmentation))


def fetch_backbone() -> dict:
    """Download the pinned backbone.  The one mode that uses the network."""
    from huggingface_hub import snapshot_download

    pin = CLASSIFIER_BACKBONE
    snapshot_download(pin.repo, revision=pin.revision,
                      allow_patterns=list(pin.files))
    return check_backbone_cache()


def check(args) -> dict:
    manifest = datasets.read_manifest(args.data_manifest)
    split = VisionSplit.load(args.split)
    split.check_no_leakage()
    excluded: list[dict] = []
    items = load_split_items(manifest, split,
                             allowed_splits=(TRAIN, VALIDATION),
                             raw_root=args.raw_root, excluded=excluded)
    populations: dict[str, dict[str, int]] = {}
    for name, entries in items.items():
        bucket: dict[str, int] = {}
        for item in entries:
            bucket[item.population] = bucket.get(item.population, 0) + 1
        populations[name] = dict(sorted(bucket.items()))
    device = resolve_device(args.device)
    return {
        "backbone": check_backbone_cache(),
        "device_that_would_be_used": device,
        "device_report": device_report(device),
        "tuning_that_would_be_used": tuning_for(
            device, deterministic=args.deterministic).as_dict(),
        "batch_size_that_would_be_used": (
            args.batch_size if args.batch_size is not None
            else suggested_batch_size(device)),
        "data_manifest_sha256": datasets.manifest_digest(manifest),
        "split_manifest_sha256": split.digest(),
        "items": {name: len(entries) for name, entries in items.items()},
        "by_population": populations,
        "excluded_unreadable": excluded,
        "test_split_not_loaded": True,
        "note": ("the test split is not loaded by this script in any mode; "
                 "only scripts/33_vision_eval.py opens it, once, against a "
                 "frozen digest"),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    chosen = [name for name in ("fetch_backbone", "check", "smoke", "run",
                                "selection_record", "verify_selection")
              if getattr(args, name)]
    if len(chosen) != 1:
        print("choose exactly one of --fetch-backbone, --check, --smoke, "
              "--run, --selection-record, --verify-selection",
              file=sys.stderr)
        return EXIT_REFUSED
    try:
        if args.fetch_backbone:
            report = fetch_backbone()
        elif args.check:
            report = check(args)
        elif args.selection_record:
            path, digest = write_selection(args.out)
            report = {"mode": "selection-record", "path": str(path),
                      "sha256": digest,
                      "record": build_selection(args.out)}
        elif args.verify_selection:
            path = Path(args.out) / RECORD_FILE
            if not path.is_file():
                print(f"refused: there is no {path}; run --selection-record "
                      "first", file=sys.stderr)
                return EXIT_REFUSED
            # Read through the module's own reader: a record that is not
            # readable JSON is a named refusal here rather than a decoder
            # traceback on the operator's terminal.
            stored = read_selection(args.out)
            problems = verify_selection(stored, args.out)
            report = {"mode": "verify-selection", "path": str(path),
                      "problems": problems,
                      "cross_configuration_selection":
                          stored.get("cross_configuration_selection"),
                      "cross_configuration_reason":
                          stored.get("cross_configuration_reason")}
            if problems:
                print(json.dumps(report, indent=2, ensure_ascii=False,
                                 sort_keys=True))
                return EXIT_PROBLEM
        else:
            config = config_from(args)
            if args.smoke:
                config = TrainConfig(
                    epochs=min(2, args.epochs), batch_size=8,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay, seed=args.seed,
                    freeze_backbone=True, trainable_stages=0,
                    label_smoothing=args.label_smoothing,
                    augmentation=config.augmentation)
            report = run(
                data_manifest_path=args.data_manifest, split_path=args.split,
                out_dir=args.out, config=config, raw_root=args.raw_root,
                device=args.device,
                steps_per_epoch=(args.steps_per_epoch
                                 if args.steps_per_epoch is not None
                                 else (6 if args.smoke else None)),
                real_weight=args.real_weight,
                expected_data_digest=args.expected_data_digest,
                expected_split_digest=args.expected_split_digest,
                deterministic=args.deterministic,
                progress=(None if args.json else
                          lambda line: print(f"  {line}", flush=True)))
            report["mode"] = "smoke" if args.smoke else "run"
            if args.smoke:
                report["smoke_warning"] = (
                    "this is a code-path smoke on a few steps. It is not a "
                    "result, not a fine-tune and may not be quoted as one")
    except (TrainError, ModelError, SelectionError,
            datasets.DatasetError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
