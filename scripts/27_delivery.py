#!/usr/bin/env python3
"""Minimum non-UI delivery: stock + brief -> comparison or F-pipeline.

Both modes are CPU-only and offline.  They load a catalogue whose rows are
required to declare ``split=train`` and agree with the frozen object-level
split manifest; no decoder, model weights, Phase 2 case, or frozen result is
read.

``compare`` retrieves existing train works with a deterministic lexical
baseline, then re-ranks the retrieved set using exact missing-part counts.
``f-pipeline`` retrieves train shapes and sends them to the existing CP-SAT
re-tiler under the supplied inventory.

The output is a delivery demonstration, not a formal evaluation and not a
metric.  Lexical similarity is not multilingual semantic retrieval.  The
ground-contact and adjacent-layer connectivity checks are static geometry,
not support, stability or physics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.bricks import parse_bricks
from src.delivery.pipeline import (DeliveryError, ExistingComparison,
                                   PipelineResult, compare_existing,
                                   load_train_catalog, run_f_pipeline,
                                   selected_text)
from src.demo.showcase import (ShowcaseError, format_report,
                               inspect_supplied, parse_inventory, write_ldraw)
from src.rendering.preview import (PreviewError, validate_preview_path,
                                   write_preview)

DEFAULT_CATALOG = ROOT / "data/processed/counterfactual_train.jsonl"
EXIT_OK, EXIT_NO_RESULT, EXIT_REFUSED = 0, 1, 2

DELIVERY_CHECKS = (
    "parse_success", "known_parts", "type_compliance", "inventory_valid",
    "in_bounds", "collision_free", "stud_only_connected", "touches_ground",
    "ldraw_serializable",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BrickAgain minimum non-UI delivery. Produces no metric.")
    p.add_argument("--mode", required=True, choices=("compare", "f-pipeline"))
    p.add_argument("--caption", required=True, help="requested work")
    p.add_argument("--inventory", required=True,
                   help="manual stock, e.g. '2x4:10,1x2:8'")
    p.add_argument("--catalog", default=str(DEFAULT_CATALOG),
                   help="train-only counterfactual JSONL")
    p.add_argument("--top-n", type=int, default=5,
                   help="caption-ranked train shapes to consider")
    p.add_argument("--exclude-object-id",
                   help="evaluation safeguard: exclude this query object")
    p.add_argument("--time-limit", type=float,
                   help="CP-SAT seconds per candidate (f-pipeline only)")
    p.add_argument("--seed", type=int,
                   help="deterministic CP-SAT seed (f-pipeline only)")
    p.add_argument("--ldr", metavar="FILE", help="write selected result .ldr")
    p.add_argument("--preview", metavar="FILE",
                   help="write selected result CPU 3-D preview (.png/.svg)")
    p.add_argument("--json", action="store_true", help="print JSON")
    return p


def _comparison_dict(result: ExistingComparison) -> dict:
    return {
        "status": ("buildable_existing_work_found" if result.selected else
                   "no_buildable_existing_work_in_retrieved_set"),
        "excluded_same_object_count": result.excluded_same_object,
        "retrieved_caption_order": [c.as_dict() for c in result.retrieved],
        "inventory_reranked": [c.as_dict() for c in result.ranked],
        "selected_catalog_id": (result.selected.item.catalog_id
                                if result.selected else None),
    }


def _pipeline_dict(result: PipelineResult) -> dict:
    return {
        "status": result.status,
        "excluded_same_object_count": result.excluded_same_object,
        "attempts": [attempt.as_dict() for attempt in result.attempts],
        "selected_catalog_id": (
            result.selected.comparison.item.catalog_id
            if result.selected else None),
    }


def _make_report(mode: str, caption: str, inventory: dict[str, int],
                 result: ExistingComparison | PipelineResult) -> dict | None:
    text = selected_text(result)
    if text is None:
        return None
    if isinstance(result, ExistingComparison):
        selected_id = result.selected.item.catalog_id  # type: ignore[union-attr]
    else:
        selected_id = result.selected.comparison.item.catalog_id  # type: ignore[union-attr]
    report = inspect_supplied(
        caption, inventory, text,
        origin=f"train-only:{mode}:{selected_id}", termination=None)
    ready = all(report["checks"][name] is True for name in DELIVERY_CHECKS)
    report["delivery"] = {
        "method": mode,
        "selected_catalog_id": selected_id,
        "checks_used": list(DELIVERY_CHECKS),
        "static_delivery_ready": ready,
        "note": (
            "This is a per-item deterministic delivery check, not a metric. "
            "Termination is not applicable because no decoder ran, so the "
            "showcase Core verdict remains n/a."),
    }
    return report


def make_payload(args) -> tuple[dict, dict | None]:
    if args.mode == "compare" and (args.time_limit is not None
                                   or args.seed is not None):
        bad = "--time-limit" if args.time_limit is not None else "--seed"
        raise DeliveryError(
            f"{bad} configures CP-SAT and does not apply to --mode compare")
    inventory = parse_inventory(args.inventory)
    catalog = load_train_catalog(args.catalog)
    if args.mode == "compare":
        result: ExistingComparison | PipelineResult = compare_existing(
            catalog, args.caption, inventory, top_n=args.top_n,
            exclude_object_id=args.exclude_object_id)
        body = _comparison_dict(result)
    else:
        result = run_f_pipeline(
            catalog, args.caption, inventory, top_n=args.top_n,
            time_limit=args.time_limit if args.time_limit is not None else 2.0,
            seed=args.seed if args.seed is not None else 0,
            exclude_object_id=args.exclude_object_id)
        body = _pipeline_dict(result)

    report = _make_report(args.mode, args.caption, inventory, result)
    payload = {
        "kind": "brickagain.minimum_delivery",
        "notice": (
            "Offline delivery demonstration only. It measures nothing and "
            "is not comparable to the frozen Phase 2 evaluation."),
        "method": {
            "name": args.mode,
            "retrieval": "deterministic lexical baseline",
            "retrieval_limit": (
                "not a multilingual embedding model and not evidence of "
                "semantic quality"),
            "shape_source": "train-only catalogue",
            "model_loaded": False,
            "phase_3c": "not authorised and not run",
        },
        "request": {"caption": args.caption, "inventory": inventory,
                    "top_n": args.top_n,
                    "same_object_exclusion_requested":
                        args.exclude_object_id is not None},
        "catalog": {"file": catalog.source.name, "split": "train",
                    "sha256": catalog.sha256,
                    "split_manifest_sha256": catalog.split_manifest_sha256,
                    "canonical_structures": len(catalog.items)},
        "result": body,
        "showcase": report,
        "outputs": {"ldraw": None, "preview": None},
    }
    return payload, report


def _format_summary(payload: dict) -> str:
    lines = [
        "=" * 72,
        "BrickAgain minimum delivery",
        "=" * 72,
        payload["notice"],
        "",
        f"mode       : {payload['method']['name']}",
        "retrieval  : deterministic lexical baseline",
        f"limitation : {payload['method']['retrieval_limit']}",
        f"catalogue  : {payload['catalog']['file']} "
        f"({payload['catalog']['canonical_structures']} canonical train "
        "structures)",
        f"catalog SHA: {payload['catalog']['sha256']}",
        f"split SHA  : {payload['catalog']['split_manifest_sha256']}",
        "",
        f"status     : {payload['result']['status']}",
    ]
    if payload["method"]["name"] == "compare":
        lines += ["", "existing-work comparison:"]
        for rank, row in enumerate(payload["result"]["inventory_reranked"], 1):
            missing = (", ".join(f"{p}:{n}" for p, n in
                                  row["missing_parts"].items()) or "none")
            lines.append(
                f"  {rank}. {row['catalog_id']}  lexical="
                f"{row['lexical_score']:.4f}  buildable="
                f"{row['fully_buildable']}  missing={missing}\n"
                f"     {row['caption']}")
    else:
        lines += ["", "F-pipeline attempts:"]
        for rank, row in enumerate(payload["result"]["attempts"], 1):
            lines.append(
                f"  {rank}. {row['catalog_id']}  lexical="
                f"{row['lexical_score']:.4f}  solver={row['solver_status']}  "
                f"grounded={row['touches_ground']}  "
                f"connected={row['stud_only_connected']}  "
                f"ready={row['delivery_ready']}"
                + (f"  reason={row['failure']}" if row["failure"] else ""))
    lines += [
        "",
        "No decoder ran. Connectivity means adjacent-layer footprint "
        "overlap; it is not support, stability or physics.",
    ]
    return "\n".join(lines)


def refuse_same_output_path(args) -> None:
    """Reject before retrieval or either writer can create a partial output."""
    if args.ldr is None or args.preview is None:
        return
    ldraw = Path(args.ldr).expanduser().resolve(strict=False)
    preview = Path(args.preview).expanduser().resolve(strict=False)
    if ldraw == preview:
        raise DeliveryError(
            "--ldr and --preview resolve to the same output path; use two "
            "different files so neither output overwrites the other")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        refuse_same_output_path(args)
        if args.preview:
            validate_preview_path(args.preview)
        payload, report = make_payload(args)
        if report is not None and report["delivery"]["static_delivery_ready"]:
            if args.ldr:
                payload["outputs"]["ldraw"] = str(write_ldraw(report, args.ldr))
            if args.preview:
                payload["outputs"]["preview"] = str(write_preview(
                    args.preview, parse_bricks(report["result"]["text"]),
                    title=args.caption))
    except (DeliveryError, ShowcaseError, PreviewError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(_format_summary(payload))
        if report is not None:
            print("\n" + format_report(report))
        if args.ldr and payload["outputs"]["ldraw"]:
            print(f"LDraw written to {payload['outputs']['ldraw']}")
        if args.preview and payload["outputs"]["preview"]:
            print(f"3-D preview written to {payload['outputs']['preview']}")
        if ((args.ldr or args.preview)
                and not payload["outputs"]["ldraw"]
                and not payload["outputs"]["preview"]):
            print("No output file was written because no selected result "
                  "passed the static delivery checks.")

    ready = bool(report and report["delivery"]["static_delivery_ready"])
    return EXIT_OK if ready else EXIT_NO_RESULT


if __name__ == "__main__":
    raise SystemExit(main())
