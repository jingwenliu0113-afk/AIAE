"""Read-only delivery summary over already sealed aggregate evidence.

This module does not score a generation, open a case, or recompute Phase 2.
It first checks the four sealed file digests.  It hashes the plan and raw
result stream as opaque bytes, then reads only the already materialised score
aggregate and project-model pointer.

The purpose is closure, not a new result: it makes explicit that Core
Success@4 exists, while Structural, Semantic and Full Success@K were never
materialised under a frozen contract.  Likewise, the stored stock variants
are strata from one fixed run, not a pre-frozen three-axis sweep.  Missing
evidence is reported as missing instead of being retroactively manufactured.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class EvidenceError(ValueError):
    """Sealed evidence is missing or differs from its recorded digest."""


FROZEN_FILES = {
    "plan": ("gpu_plans/core_eval_plan.json",
             "e0303a5b0f5815a25090773bc84eb0d91122ee20c6a8ba064481b9626ad226c3"),
    "results": ("runs/core_eval/results.jsonl",
                "7fda6aec3b02ed767ff4f23858a9164c6334f6a57d36acf8c76609bd2c8c6bd1"),
    "scores": ("runs/core_eval/scores.json",
               "730b8a8f67527163c881de87a6c0a4a93033c179240bad4163dee797c279e2b3"),
    "project_model": ("runs/project_model.json",
                      "84ac7235b98be87c65bf3bbcad79852ed47e6cb5710ff33f220f9862f7d7474a"),
}

EXPECTED_CORE = {
    "B": (6, 160),
    "C": (8, 160),
    "D": (47, 160),
    "E": (26, 160),
}

SCORER_MANIFEST_DIGEST = (
    "baa952cfb199adfe35c85ada518d445b3e1bfbed8da2690a6b00c4cb112cc9cd")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_paths(root: Path) -> dict[str, Path]:
    paths = {}
    for name, (relative, expected) in FROZEN_FILES.items():
        path = root / relative
        if not path.is_file():
            raise EvidenceError(
                f"sealed {name} evidence is unavailable: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise EvidenceError(
                f"sealed {name} digest differs: expected {expected}, got "
                f"{actual}; nothing was summarised")
        paths[name] = path
    return paths


def sealed_delivery_summary(root: str | Path) -> dict:
    """Verify sealed bytes and extract only their stored aggregate values."""
    root = Path(root)
    paths = _verified_paths(root)
    scores = json.loads(paths["scores"].read_text(encoding="utf-8"))
    project = json.loads(paths["project_model"].read_text(encoding="utf-8"))

    if scores.get("kind") != "core_eval_scores" or scores.get("k") != 4:
        raise EvidenceError("the sealed score aggregate has an unknown schema")
    if scores.get("cases") != 160 or scores.get("draws") != 2560:
        raise EvidenceError("the sealed score aggregate has unexpected scope")
    if scores.get("scorer_source_manifest_digest") != SCORER_MANIFEST_DIGEST:
        raise EvidenceError("the scorer source manifest digest differs")
    if project.get("model") != "final_H2":
        raise EvidenceError("the sealed project model is not final_H2")

    core = {}
    inventory = {}
    for arm, expected in EXPECTED_CORE.items():
        stored = scores["core_success_at_k"]["by_arm"][arm]["overall"]
        observed = (stored["numerator"], stored["denominator"])
        if observed != expected:
            raise EvidenceError(
                f"stored Core Success@4 for {arm} differs: {observed}")
        core[arm] = {
            "numerator": stored["numerator"],
            "denominator": stored["denominator"],
            "value": stored["value"],
        }
        inv = scores["per_arm"][arm]["overall"]["rates"]["inventory_valid"]
        inventory[arm] = {
            "numerator": inv["numerator"],
            "denominator": inv["denominator"],
            "value": inv["value"],
            "count_overflow_amount_total":
                scores["per_arm"][arm]["overall"]
                      ["count_overflow_amount_total"],
        }

    return {
        "kind": "brickagain.delivery_evidence_closure",
        "notice": (
            "Read-only extraction from a sealed aggregate. No case was "
            "opened, no score was recomputed, and this is not a new result."),
        "verified_sha256": {
            name: FROZEN_FILES[name][1] for name in FROZEN_FILES},
        "scope": {
            "cases": 160,
            "draws": 2560,
            "k": 4,
            "limits": (
                "One fixed execution on cases frozen in advance; no "
                "pre-frozen inference test, significance claim or "
                "generalisation claim."),
        },
        "project_model": "final_H2",
        "core_success_at_4": core,
        "inventory_valid": inventory,
        "success_families": {
            "core_success_at_4": "available as the sealed aggregate above",
            "structural_success_at_k": (
                "not separately materialised; Core Success@4 is not "
                "retroactively renamed"),
            "semantic_success_at_k": (
                "not materialised; no frozen semantic threshold or human "
                "evaluation exists"),
            "full_success_at_k": (
                "not materialised because semantic success is unavailable"),
        },
        "inventory_axes": {
            "stored_strata": list(scores.get("strata", [])),
            "three_axis_report": (
                "not materialised: exact/loose/distractor/mixed are strata "
                "from one fixed run, not a pre-frozen tau/rho/removal sweep"),
        },
        "phase_3_placement": (
            "rules and implementation reviewed; never formally evaluated, "
            "so no metric is attached"),
        "research_track": "closed; missing experiments are not a backlog",
    }


def format_evidence_summary(summary: dict) -> str:
    lines = [
        "BrickAgain sealed evidence closure",
        summary["notice"],
        "",
        "Core Success@4 (stored aggregate; not recomputed):",
    ]
    for arm, row in summary["core_success_at_4"].items():
        lines.append(
            f"  {arm}: {row['numerator']}/{row['denominator']} "
            f"({row['value']:.2%})")
    lines += [
        "",
        "Success-family closure:",
    ]
    for name, value in summary["success_families"].items():
        lines.append(f"  {name}: {value}")
    lines += [
        "",
        "Inventory-axis closure:",
        "  " + summary["inventory_axes"]["three_axis_report"],
        "",
        "Phase 3 placement:",
        "  " + summary["phase_3_placement"],
        "",
        "Research track: " + summary["research_track"],
    ]
    return "\n".join(lines)
