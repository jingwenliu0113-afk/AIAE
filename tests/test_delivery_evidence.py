"""The delivery evidence report verifies and extracts; it never re-scores."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.delivery import evidence
from src.delivery.evidence import (EvidenceError, FROZEN_FILES,
                                   format_evidence_summary,
                                   sealed_delivery_summary, sha256_file)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/28_delivery_evidence.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("m28", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sealed_root(tmp_path, monkeypatch):
    """A tiny aggregate with the same schema, safe for the public snapshot."""
    core = {arm: {"overall": {"numerator": n, "denominator": d,
                              "value": n / d}}
            for arm, (n, d) in evidence.EXPECTED_CORE.items()}
    per_arm = {
        arm: {"overall": {
            "rates": {"inventory_valid": {
                "numerator": (640 if arm in "DE" else 1),
                "denominator": 640,
                "value": (1.0 if arm in "DE" else 1 / 640)}},
            "count_overflow_amount_total": (0 if arm in "DE" else 9),
        }} for arm in "BCDE"
    }
    scores = {
        "kind": "core_eval_scores", "k": 4, "cases": 160,
        "draws": 2560,
        "scorer_source_manifest_digest": evidence.SCORER_MANIFEST_DIGEST,
        "strata": ["overall", "role=control", "role=counterfactual",
                   "variant=exact", "variant=loose", "variant=distractor",
                   "variant=mixed"],
        "core_success_at_k": {"by_arm": core},
        "per_arm": per_arm,
    }
    bodies = {
        "plan": b"opaque plan fixture\n",
        "results": b"opaque result fixture\n",
        "scores": json.dumps(scores).encode(),
        "project_model": json.dumps({"model": "final_H2"}).encode(),
    }
    frozen = {}
    for name, (relative, _) in FROZEN_FILES.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bodies[name])
        frozen[name] = (relative, sha256_file(path))
    monkeypatch.setattr(evidence, "FROZEN_FILES", frozen)
    return tmp_path


def test_the_four_real_sealed_digests_are_pinned_in_source():
    assert FROZEN_FILES == {
        "plan": ("gpu_plans/core_eval_plan.json",
                 "e0303a5b0f5815a25090773bc84eb0d91122ee20c6a8ba064481b9626ad226c3"),
        "results": ("runs/core_eval/results.jsonl",
                    "7fda6aec3b02ed767ff4f23858a9164c6334f6a57d36acf8c76609bd2c8c6bd1"),
        "scores": ("runs/core_eval/scores.json",
                   "730b8a8f67527163c881de87a6c0a4a93033c179240bad4163dee797c279e2b3"),
        "project_model": ("runs/project_model.json",
                          "84ac7235b98be87c65bf3bbcad79852ed47e6cb5710ff33f220f9862f7d7474a"),
    }


def test_the_four_sealed_digests_are_verified_before_summary(sealed_root):
    summary = sealed_delivery_summary(sealed_root)
    assert summary["verified_sha256"] == {
        key: digest for key, (_, digest) in evidence.FROZEN_FILES.items()}


def test_only_the_aggregate_and_model_pointer_are_parsed(sealed_root,
                                                         monkeypatch):
    calls = []
    original = Path.read_text

    def watched(path, *args, **kwargs):
        calls.append(path.relative_to(sealed_root).as_posix())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", watched)
    sealed_delivery_summary(sealed_root)
    assert calls == ["runs/core_eval/scores.json", "runs/project_model.json"]


def test_the_stored_core_values_are_not_renamed_or_extended(sealed_root):
    summary = sealed_delivery_summary(sealed_root)
    assert {arm: (row["numerator"], row["denominator"])
            for arm, row in summary["core_success_at_4"].items()} == {
                "B": (6, 160), "C": (8, 160),
                "D": (47, 160), "E": (26, 160)}
    families = summary["success_families"]
    assert "not separately materialised" in families["structural_success_at_k"]
    assert "not materialised" in families["semantic_success_at_k"]
    assert "not materialised" in families["full_success_at_k"]


def test_the_inventory_strata_are_not_called_a_three_axis_sweep(sealed_root):
    summary = sealed_delivery_summary(sealed_root)
    assert summary["inventory_axes"]["stored_strata"] == [
        "overall", "role=control", "role=counterfactual",
        "variant=exact", "variant=loose", "variant=distractor",
        "variant=mixed"]
    assert "not a pre-frozen" in summary["inventory_axes"]["three_axis_report"]


def test_missing_private_evidence_is_a_refusal(tmp_path):
    with pytest.raises(EvidenceError, match="unavailable"):
        sealed_delivery_summary(tmp_path)


def test_the_text_version_names_every_missing_evidence_family(sealed_root):
    text = format_evidence_summary(sealed_delivery_summary(sealed_root))
    assert "not recomputed" in text
    assert "structural_success_at_k" in text
    assert "semantic_success_at_k" in text
    assert "full_success_at_k" in text
    assert "never formally evaluated" in text


def test_the_command_emits_the_same_read_only_json(cli, sealed_root,
                                                   monkeypatch, capsys):
    summary = sealed_delivery_summary(sealed_root)
    monkeypatch.setattr(cli, "sealed_delivery_summary", lambda root: summary)
    assert cli.main(["--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["kind"] == "brickagain.delivery_evidence_closure"
    assert body["project_model"] == "final_H2"
    assert body["scope"]["draws"] == 2560


def test_the_source_never_imports_the_scorer_or_case_plan_reader():
    body = (ROOT / "src/delivery/evidence.py").read_text(encoding="utf-8")
    assert "src.eval.scoring" not in body
    assert "score_generation" not in body
    assert "per_draw" not in body
