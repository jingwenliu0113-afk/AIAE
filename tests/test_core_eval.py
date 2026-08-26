"""Core acceptance, tested without a test split, a model, a GPU or a network.

Every fixture here is synthetic. Nothing in this file opens
``data/processed/instruct_inv_test.jsonl``, resolves a checkpoint, or reaches
the hub -- which is the only way the contract can be checked *before* the run
it governs rather than after it.

The loaders are injected, so "arm C went through ``load_finetuned`` with
``verify_digest=True`` and ``local_files_only=True``" is an assertion about a
recorded call rather than a hope about a code path. The device probe is
injected for the same reason: "``--run`` refuses to start off WSL2" has to be
provable on a Mac.
"""

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bricks import is_connected, parse_bricks
from src.eval import acceptance, scoring
from src.training import pack
from src.training.longrun import canonical_json

ROOT = Path(__file__).resolve().parents[1]

CAPTIONS = ("A small chair.", "A red car.", "A short bridge.", "A tall tower.")
INVENTORIES = (
    {"1x2": 6, "2x4": 4},
    {"1x1": 5, "2x2": 3, "1x4": 2},
    {"2x4": 8, "1x8": 2},
    {"1x2": 3, "2x6": 4, "1x1": 9},
)
ROLES = acceptance.ROLES
VARIANTS = acceptance.VARIANTS


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def synthetic_rows(n_pairs: int = 25) -> list[dict]:
    """25 pairs of 8, shaped like the real split and carrying no real data."""
    rows = []
    for p in range(n_pairs):
        for r, role in enumerate(ROLES):
            for v, variant in enumerate(VARIANTS):
                i = r * len(VARIANTS) + v
                caption = CAPTIONS[(p + i) % len(CAPTIONS)]
                inventory = dict(INVENTORIES[(p + i) % len(INVENTORIES)])
                rows.append({
                    "sample_id": f"s{p:03d}_{i}",
                    "pair_id": f"p{p:03d}",
                    "object_id": f"o{p:03d}",
                    "split": "test",
                    "role": role,
                    "variant": variant,
                    "dropped_part": None,
                    "caption": caption,
                    "inventory": inventory,
                    "used": {"2x4": 1},
                    "prompt": f"prompt for {caption}",
                    "target": "2x4 (0,0,0)\n2x4 (0,0,1)\n",
                    "n_prompt_tokens": 100,
                    "n_target_tokens": 21,
                    "n_tokens": 121,
                })
    return rows


@pytest.fixture
def fake_split(tmp_path, monkeypatch):
    """A stand-in test split, and the contract re-pinned to its digest."""
    path = tmp_path / "data" / "processed" / "instruct_inv_test.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                              for r in synthetic_rows()) + "\n",
                    encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(acceptance, "EXPECTED_TEST_SHA256", digest)
    return tmp_path


def tiny_plan(label=""):
    """A full-size plan -- 20 pairs of 8 -- built from nothing.

    Full size on purpose: the validator refuses a plan that is not exactly 160
    cases in 20 pairs of two roles by four variants, and a fixture that dodged
    that would be testing a validator this project does not have. ``label``
    changes the captions, which changes the digest, which is how a second,
    different plan is made.
    """
    cases = []
    for p in range(acceptance.N_PAIRS):
        for r, role in enumerate(ROLES):
            for v, variant in enumerate(VARIANTS):
                i = r * len(VARIANTS) + v
                cases.append(acceptance.build_case({
                    "sample_id": f"c{p:02d}_{i}", "pair_id": f"q{p:02d}",
                    "role": role, "variant": variant,
                    "caption": f"A thing {label}{p}-{i}.",
                    "inventory": {"2x4": 4},
                }))
    body = {
        "kind": acceptance.PLAN_KIND,
        "schema_version": acceptance.PLAN_SCHEMA_VERSION,
        "contract_version": acceptance.CONTRACT_VERSION,
        "contract_digest": acceptance.contract_digest(),
        "settings_digest": acceptance.settings_digest(),
        "source": acceptance.source_document(acceptance.EXPECTED_TEST_SHA256,
                                             len(cases)),
        "arms": {n: acceptance.ARMS[n].as_dict() for n in acceptance.ARM_ORDER},
        "settings": acceptance.SETTINGS.as_dict(),
        "final_model": acceptance.final_model_document(),
        "scorer_source_manifest": acceptance.scorer_manifest(ROOT),
        "scorer_source_manifest_digest": acceptance.scorer_manifest_digest(ROOT),
        "cases": cases,
        "carries": list(acceptance.CASE_FIELDS),
        "note": "synthetic",
    }
    body["schedule"] = acceptance.plan_schedule(body)
    body["plan_digest"] = acceptance.plan_digest(body)
    return body


# --- evidence helpers ------------------------------------------------------

PACK_DIGEST = "1" * 64
DEPENDENCY_DIGEST = "2" * 64


def node_probe(**over):
    reading = {"os_system": "Linux", "wsl2": True,
               "wsl_evidence": "kernel release names a Microsoft build",
               "torch_cuda_build": "12.8", "cuda_available": True,
               "gpu_name": "NVIDIA GeForce RTX 5070 Ti", "vram_total_gb": 15.9,
               "system_ram_gb": 31.2, "allocator_backend": "native",
               "offline_env": {"HF_HUB_OFFLINE": "1",
                               "TRANSFORMERS_OFFLINE": "1",
                               "HF_HUB_DISABLE_TELEMETRY": "1"}}
    reading.update(over)
    return reading


def preflight_stub(passed=True, drop=(), fail=()):
    checks = {name: {"passed": name not in fail, "detail": "..."}
              for name in acceptance.REQUIRED_PREFLIGHT_GATES
              if name not in drop}
    return {"passed": passed and not fail, "checks": checks,
            "failed": sorted(fail)}


def warmup_stub(**over):
    body = {"generations": acceptance.WARMUP["generations"],
            "seeds": list(acceptance.WARMUP["seeds"]),
            "caption_is_from_the_test_split": False,
            "seconds": [3.1, 2.9],
            "excluded_from_every_reported_number": True,
            "policy": acceptance.WARMUP["policy"]}
    body.update(over)
    return body


def attempt_body(plan, index, attempt_index=0, *, missing=320,
                 preflight=None, warmup=None, **over):
    _group, name = acceptance.step(index)
    adapter = "/somewhere/adapter" \
        if acceptance.ARMS[name].model != acceptance.PUBLIC_MODEL else None
    body = acceptance.build_attempt_evidence(
        index=index, attempt_index=attempt_index, plan=plan,
        probe_reading=node_probe(),
        preflight_result=preflight or preflight_stub(),
        pack_digest=PACK_DIGEST, dependency_digest=DEPENDENCY_DIGEST,
        warmup=warmup or warmup_stub(), adapter_dir=adapter,
        cells_missing_at_start=missing,
        started_at="2026-08-22T10:00:00+00:00")
    body.update(over)
    return body


_ATTEMPTS: dict = {}


def attempts_for(plan):
    """One attempt body per step, cached so a fixture is cheap to reuse."""
    key = plan["plan_digest"]
    if key not in _ATTEMPTS:
        _ATTEMPTS[key] = {i: attempt_body(plan, i)
                          for i in range(acceptance.N_STEPS)}
    return _ATTEMPTS[key]


# --- results helpers -------------------------------------------------------

def placement(plan):
    """``(case_id, arm) -> (step_index, group)`` from the frozen schedule."""
    out = {}
    for index in range(acceptance.N_STEPS):
        group, name = acceptance.STEP_ORDER[index]
        for case in acceptance.step_cases(plan, index):
            out[(case["case_id"], name)] = (index, group)
    return out


def cell_for(plan, case, arm, seed, *, where=None, attempt=None, **over):
    index, group = (where or placement(plan))[(case["case_id"], arm)]
    attempt = attempt or attempts_for(plan)[index]
    row = {
        "plan_digest": plan["plan_digest"], "case_id": case["case_id"],
        "arm": arm, "seed": seed, "step_index": index, "group": group,
        "attempt_id": attempt["attempt_id"],
        "attempt_digest": acceptance.attempt_digest(attempt),
        "raw_text": "2x4 (0,0,0)\n2x4 (0,0,1)\n",
        "n_tokens": 21, "seconds": 1.5, "termination": "normal_eos",
        "truncated": False,
        "gate": acceptance.gate_ledger(acceptance.ARMS[arm],
                                       case["inventory"], None),
        "prompt_sha256": case["prompt_sha256"],
        "inventory_digest": case["inventory_digest"],
        "contract_digest": acceptance.contract_digest(),
        "settings_digest": acceptance.settings_digest(),
        "model": acceptance.model_identity(arm),
    }
    if acceptance.ARMS[arm].gate != acceptance.GATE_NONE:
        row["gate"]["accepted_parts"] = ["2x4", "2x4"]
        row["gate"]["remaining_inventory"] = {"2x4": 2}
    row.update(over)
    return row


def fill_step(path, plan, index, known=None, *, where=None, attempt=None,
              cells=None):
    known = acceptance.known_keys(path) if known is None else known
    where = where or placement(plan)
    cases = {c["case_id"]: c for c in plan["cases"]}
    for _d, case_id, arm, seed in (cells if cells is not None
                                   else acceptance.step_cells(plan, index)):
        acceptance.append_cell(
            path, cell_for(plan, cases[case_id], arm, seed, where=where,
                           attempt=attempt),
            known=known)
    return acceptance.STEP_ORDER[index][1]


def fill(path, plan, arms=("B",)):
    """Every step of the frozen schedule whose arm is wanted, in order."""
    known = acceptance.known_keys(path)
    where = placement(plan)
    for index in range(acceptance.N_STEPS):
        if acceptance.STEP_ORDER[index][1] in arms:
            fill_step(path, plan, index, known, where=where)


def seal_step(evidence_dir, plan, index, results, *, attempts=None,
              sealed_without_decoding=False):
    """Write the attempt records and the completion for one step."""
    from src.training.session import write_once_json

    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    bodies = attempts if attempts is not None else [attempts_for(plan)[index]]
    rows = [r for r in acceptance.read_cells(results)
            if r.get("step_index") == index]
    counted = {}
    for row in rows:
        counted[row["attempt_id"]] = counted.get(row["attempt_id"], 0) + 1
    references = []
    for body in bodies:
        path = acceptance.attempt_path(evidence_dir, index,
                                       body["attempt_index"])
        if not path.exists():
            write_once_json(path, body)
        references.append(acceptance.attempt_reference(
            body, counted.get(body["attempt_id"], 0)))
    completion = acceptance.build_step_completion(
        index=index, plan=plan, attempts=references, cells_recorded=len(rows),
        sealed_at="2026-08-22T11:00:00+00:00",
        sealed_without_decoding=sealed_without_decoding)
    write_once_json(acceptance.completion_path(evidence_dir, index),
                    completion)
    return completion


def seal_all(evidence_dir, plan, results, arms=acceptance.ARM_ORDER):
    for index in range(acceptance.N_STEPS):
        if acceptance.STEP_ORDER[index][1] in arms:
            seal_step(evidence_dir, plan, index, results)
    return Path(evidence_dir)


def run_and_seal(tmp_path, plan, arms=acceptance.ARM_ORDER):
    """A results file and a matching evidence directory, both complete."""
    results = Path(tmp_path) / "results.jsonl"
    fill(results, plan, arms=arms)
    evidence = seal_all(Path(tmp_path) / "evidence", plan, results, arms=arms)
    return results, evidence


def rewrite(path, rows):
    path.write_text("".join(canonical_json(r) + "\n" for r in rows),
                    encoding="utf-8")


def tokens_for(n_bricks: int, *, eos: bool = True) -> int:
    return n_bricks * 10 + (1 if eos else 0)


class TestTheAttemptRecord:
    """Written after the preflight and the warm-up, before the first cell."""

    def test_a_good_attempt_has_no_problems(self):
        plan = tiny_plan()
        for index in range(acceptance.N_STEPS):
            assert acceptance.attempt_problems(
                attempt_body(plan, index), plan,
                expected_pack_digest=PACK_DIGEST,
                expected_dependency_digest=DEPENDENCY_DIGEST) == [], index

    def problems(self, body, plan):
        return acceptance.attempt_problems(
            body, plan, expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST)

    def test_it_binds_the_pack_digest_carried_by_hand(self):
        plan = tiny_plan()
        body = attempt_body(plan, 0, pack_digest="9" * 64)
        assert any("not the digest carried from the build machine" in p
                   for p in self.problems(body, plan))

    def test_it_binds_the_dependency_digest_too(self):
        plan = tiny_plan()
        body = attempt_body(plan, 0, dependency_digest="9" * 64)
        assert any("dependencies" in p for p in self.problems(body, plan))

    def test_a_missing_expected_digest_is_a_refusal_not_a_default(self):
        plan = tiny_plan()
        problems = acceptance.attempt_problems(
            attempt_body(plan, 0), plan, expected_pack_digest=None,
            expected_dependency_digest=None)
        assert any("pack digest" in p for p in problems)
        assert any("dependency digest" in p for p in problems)

    def test_a_preflight_that_did_not_pass_is_a_refusal(self):
        plan = tiny_plan()
        body = attempt_body(plan, 0, preflight=preflight_stub(fail=("cuda",)))
        assert any("did not pass" in p for p in self.problems(body, plan))

    def test_it_records_the_number_of_cells_it_set_out_to_fill(self):
        plan = tiny_plan()
        body = attempt_body(plan, 0, missing=17)
        assert body["cells_missing_at_start"] == 17
        assert self.problems(body, plan) == []

    def test_a_step_labelled_with_the_wrong_arm_is_refused(self):
        plan = tiny_plan()
        body = attempt_body(plan, 0, arm="E")
        assert any("frozen schedule" in p for p in self.problems(body, plan))

    def test_an_attempt_id_that_is_not_its_own_index_is_refused(self):
        plan = tiny_plan()
        body = attempt_body(plan, 0, attempt_id="something-else")
        assert any("is not the id step" in p for p in self.problems(body, plan))

    def test_evidence_for_another_plan_is_refused(self):
        plan = tiny_plan()
        body = attempt_body(plan, 0,
                            plan_digest=tiny_plan(label="x ")["plan_digest"])
        assert any("different plan" in p for p in self.problems(body, plan))

    def test_an_extra_or_missing_field_is_refused(self):
        plan = tiny_plan()
        body = attempt_body(plan, 0)
        body["surprise"] = 1
        assert any("attempt fields" in p for p in self.problems(body, plan))

    def test_a_step_that_did_not_run_on_wsl2_or_cuda_is_refused(self):
        plan = tiny_plan()
        for field, value, phrase in (("wsl2", False, "did not run on WSL2"),
                                     ("cuda_available", False,
                                      "CUDA available"),
                                     ("torch_cuda_build", None,
                                      "no CUDA build")):
            body = attempt_body(plan, 0)
            body["platform"] = {**body["platform"], field: value}
            assert any(phrase in p for p in self.problems(body, plan)), field

    def test_a_different_device_or_dtype_is_refused(self):
        plan = tiny_plan()
        for field, value in (("device", "cpu"), ("dtype", "float16")):
            body = attempt_body(plan, 0)
            body["platform"] = {**body["platform"], field: value}
            assert any("records device" in p
                       for p in self.problems(body, plan)), field

    def test_the_gpu_and_memory_readings_are_recorded(self):
        body = attempt_body(tiny_plan(), 0)
        assert body["platform"]["gpu_name"] == "NVIDIA GeForce RTX 5070 Ti"
        assert body["platform"]["vram_total_gb"] == 15.9
        assert body["platform"]["system_ram_gb"] == 31.2

    def test_package_provenance_is_required(self):
        plan = tiny_plan()
        for name in ("python", "torch", "transformers", "peft"):
            body = attempt_body(plan, 0)
            body["provenance"] = {**body["provenance"], name: None}
            assert any(f"no {name} version" in p
                       for p in self.problems(body, plan)), name

    def test_the_warm_up_must_be_declared_and_excluded(self):
        plan = tiny_plan()
        body = attempt_body(plan, 0, warmup=warmup_stub(
            excluded_from_every_reported_number=False))
        assert any("excluded" in p for p in self.problems(body, plan))

    def test_a_warm_up_of_the_wrong_size_is_refused(self):
        plan = tiny_plan()
        body = attempt_body(plan, 0, warmup=warmup_stub(generations=1))
        assert any("warm-up generations" in p
                   for p in self.problems(body, plan))

    def test_a_warm_up_taken_from_the_test_split_is_refused(self):
        plan = tiny_plan()
        body = attempt_body(plan, 0, warmup=warmup_stub(
            caption_is_from_the_test_split=True))
        assert any("outside the test split" in p
                   for p in self.problems(body, plan))

    def test_a_final_arm_attempt_must_record_the_three_adapter_digests(self):
        plan = tiny_plan()
        index = acceptance.STEP_ORDER.index(("even", "C"))
        body = attempt_body(plan, index)
        assert body["adapter"]["files"] == acceptance.FINAL_ADAPTER_SHA256
        body["adapter"] = None
        assert any("records no adapter" in p for p in self.problems(body, plan))


class TestEveryPreflightGateMustBePresentAndTrue:
    """An absent gate used to satisfy itself. Deleting one must now fail."""

    def problems(self, body, plan):
        return acceptance.attempt_problems(
            body, plan, expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST)

    def test_the_required_set_is_the_declared_ten(self):
        assert set(acceptance.REQUIRED_PREFLIGHT_GATES) == {
            "platform", "cuda", "torch_cuda_build", "gpu_model", "vram",
            "system_ram", "offline", "allocator_config", "pack",
            "dependencies"}
        assert len(acceptance.REQUIRED_PREFLIGHT_GATES) == 10

    @pytest.mark.parametrize("gate", acceptance.REQUIRED_PREFLIGHT_GATES)
    def test_deleting_a_gate_fails(self, gate):
        plan = tiny_plan()
        body = attempt_body(plan, 0, preflight=preflight_stub(drop=(gate,)))
        problems = self.problems(body, plan)
        assert any(f"does not contain the gate {gate!r}" in p
                   for p in problems), (gate, problems)

    @pytest.mark.parametrize("gate", acceptance.REQUIRED_PREFLIGHT_GATES)
    def test_a_gate_that_is_false_fails(self, gate):
        plan = tiny_plan()
        body = attempt_body(plan, 0, preflight=preflight_stub(fail=(gate,)))
        assert any(f"{gate!r} did not pass" in p
                   for p in self.problems(body, plan)), gate

    @pytest.mark.parametrize("gate", acceptance.REQUIRED_PREFLIGHT_GATES)
    def test_a_gate_that_is_merely_truthy_fails(self, gate):
        plan = tiny_plan()
        body = attempt_body(plan, 0)
        body["preflight"]["checks"][gate] = "yes"
        assert any(f"{gate!r} did not pass" in p
                   for p in self.problems(body, plan)), gate

    def test_an_empty_check_table_fails_every_gate(self):
        plan = tiny_plan()
        body = attempt_body(plan, 0)
        body["preflight"]["checks"] = {}
        problems = self.problems(body, plan)
        for gate in acceptance.REQUIRED_PREFLIGHT_GATES:
            assert any(gate in p for p in problems), gate

    def test_every_required_gate_is_one_the_real_preflight_emits(self):
        """A gate name the preflight never produces can only be absent."""
        from src.training import gpu_node

        result = gpu_node.preflight(
            probe=node_probe(), pack_dir="/nowhere",
            expected_pack_digest="3" * 64,
            expected_dependency_digest="4" * 64,
            verifier=lambda dest, data_root=None: ["stand-in"],
            dependency_checker=lambda: {"ok": False, "problems": ["stand-in"],
                                        "evidence": None})
        for gate in acceptance.REQUIRED_PREFLIGHT_GATES:
            assert gate in result["checks"], gate


class TestTheCompletionRecord:
    def completion(self, plan, index=0, **over):
        body = acceptance.build_step_completion(
            index=index, plan=plan,
            attempts=[acceptance.attempt_reference(
                attempts_for(plan)[index], 320)],
            cells_recorded=320, sealed_at="2026-08-22T11:00:00+00:00",
            sealed_without_decoding=False)
        body.update(over)
        return body

    def test_a_good_completion_has_no_problems(self):
        plan = tiny_plan()
        for index in range(acceptance.N_STEPS):
            assert acceptance.completion_problems(
                self.completion(plan, index), plan, index=index) == [], index

    def test_it_lists_every_attempt(self):
        plan = tiny_plan()
        body = self.completion(plan)
        assert [a["attempt_id"] for a in body["attempts"]] == [
            acceptance.attempt_id_for(0, 0)]

    def test_a_completion_with_no_attempts_is_refused(self):
        plan = tiny_plan()
        body = self.completion(plan, attempts=[])
        assert any("lists no attempts" in p for p in
                   acceptance.completion_problems(body, plan, index=0))

    def test_the_attempts_must_account_for_every_cell(self):
        plan = tiny_plan()
        body = self.completion(plan)
        body["attempts"][0]["cells_written"] = 319
        assert any("account for 319 cells" in p for p in
                   acceptance.completion_problems(body, plan, index=0))

    def test_an_incomplete_step_may_not_be_sealed(self):
        plan = tiny_plan()
        body = self.completion(plan, cells_recorded=319)
        body["attempts"][0]["cells_written"] = 319
        assert any("a step is closed when it is complete" in p for p in
                   acceptance.completion_problems(body, plan, index=0))

    def test_two_attempts_sharing_an_id_are_refused(self):
        plan = tiny_plan()
        body = self.completion(plan)
        body["attempts"] = [dict(body["attempts"][0], cells_written=160),
                            dict(body["attempts"][0], cells_written=160)]
        assert any("share an id" in p for p in
                   acceptance.completion_problems(body, plan, index=0))

    def test_it_says_whether_sealing_decoded_anything(self):
        plan = tiny_plan()
        body = self.completion(plan, sealed_without_decoding="maybe")
        assert any("whether sealing decoded" in p for p in
                   acceptance.completion_problems(body, plan, index=0))

    def test_a_completion_for_another_plan_or_step_is_refused(self):
        plan = tiny_plan()
        assert any("different plan" in p for p in
                   acceptance.completion_problems(
                       self.completion(plan, plan_digest="z" * 64), plan,
                       index=0))
        assert any("frozen schedule says" in p for p in
                   acceptance.completion_problems(self.completion(plan, 0),
                                                  plan, index=1))


class TestTheEvidenceChain:
    def test_a_complete_run_verifies(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan)
        assert acceptance.evidence_problems(
            evidence, plan, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST) == []

    def test_no_evidence_directory_is_a_refusal(self, tmp_path):
        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        fill(results, plan, arms=("B",))
        problems = acceptance.evidence_problems(
            None, plan, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST, arms=("B",))
        assert any("no evidence directory" in p for p in problems)

    def test_an_unsealed_step_is_reported(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        acceptance.completion_path(evidence, 0).unlink()
        problems = acceptance.evidence_problems(
            evidence, plan, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST, arms=("B",))
        assert any("never sealed" in p for p in problems), problems

    def test_a_completion_naming_an_absent_attempt_is_reported(self,
                                                               tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        acceptance.attempt_path(evidence, 0, 0).unlink()
        problems = acceptance.evidence_problems(
            evidence, plan, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST, arms=("B",))
        assert any("no such attempt record" in p for p in problems), problems

    def test_an_attempt_the_completion_omits_is_reported(self, tmp_path):
        from src.training.session import write_once_json

        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        write_once_json(acceptance.attempt_path(evidence, 0, 1),
                        attempt_body(plan, 0, 1, missing=0))
        problems = acceptance.evidence_problems(
            evidence, plan, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST, arms=("B",))
        assert any("does not list it" in p for p in problems), problems

    def test_a_rewritten_attempt_no_longer_matches_its_digest(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        path = acceptance.attempt_path(evidence, 0, 0)
        body = json.loads(path.read_text(encoding="utf-8"))
        body["platform"]["gpu_name"] = "something else"
        path.unlink()
        path.write_text(json.dumps(body), encoding="utf-8")
        problems = acceptance.evidence_problems(
            evidence, plan, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST, arms=("B",))
        assert any("'attempt_digest'" in p for p in problems), problems

    def test_a_cell_naming_an_unlisted_attempt_is_reported(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        rows = acceptance.read_cells(results)
        rows[0]["attempt_id"] = "invented"
        rewrite(results, rows)
        problems = acceptance.evidence_problems(
            evidence, plan, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST, arms=("B",))
        assert any("which this step's completion does not list" in p
                   for p in problems), problems

    def test_a_cell_count_the_completion_disagrees_with_is_reported(self,
                                                                    tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        path = acceptance.completion_path(evidence, 0)
        body = json.loads(path.read_text(encoding="utf-8"))
        body["attempts"][0]["cells_written"] = 100
        path.unlink()
        path.write_text(json.dumps(body), encoding="utf-8")
        problems = acceptance.evidence_problems(
            evidence, plan, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST, arms=("B",))
        assert any("'cells_written'" in p for p in problems), problems

    def test_a_cell_whose_attempt_digest_is_wrong_is_reported(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        rows = acceptance.read_cells(results)
        rows[0]["attempt_digest"] = "0" * 64
        rewrite(results, rows)
        problems = acceptance.evidence_problems(
            evidence, plan, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST, arms=("B",))
        assert any("not the digest of the attempt it names" in p
                   for p in problems), problems

    def test_the_evidence_manifest_covers_every_file_it_rested_on(self,
                                                                  tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        manifest = acceptance.evidence_manifest(evidence, plan, arms=("B",))
        assert set(manifest) == {
            "step_00_even_B.json", "step_00_even_B_attempt_00.json",
            "step_06_odd_B.json", "step_06_odd_B_attempt_00.json"}
        assert all(len(v) == 64 for v in manifest.values())
        before = acceptance.evidence_manifest_digest(evidence, plan,
                                                     arms=("B",))
        acceptance.completion_path(evidence, 0).unlink()
        assert acceptance.evidence_manifest_digest(
            evidence, plan, arms=("B",)) != before


class TestAResumeOpensANewAttempt:
    def partial(self, tmp_path, plan, done=100):
        results = tmp_path / "results.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        first = attempt_body(plan, 0, 0, missing=320)
        from src.training.session import write_once_json

        write_once_json(acceptance.attempt_path(evidence, 0, 0), first)
        fill_step(results, plan, 0, attempt=first,
                  cells=acceptance.step_cells(plan, 0)[:done])
        return results, evidence, first

    def test_the_cells_of_the_dead_attempt_keep_its_id(self, tmp_path):
        plan = tiny_plan()
        results, evidence, first = self.partial(tmp_path, plan)
        rows = acceptance.read_cells(results)
        assert {r["attempt_id"] for r in rows} == {first["attempt_id"]}
        assert len(rows) == 100

    def test_the_new_attempt_gets_the_next_index(self, tmp_path):
        plan = tiny_plan()
        _results, evidence, _first = self.partial(tmp_path, plan)
        assert acceptance.next_attempt_index(evidence, 0) == 1
        assert acceptance.attempt_id_for(0, 1).endswith("attempt01")

    def test_old_cells_are_never_re_attributed(self, tmp_path):
        from src.training.session import write_once_json

        plan = tiny_plan()
        results, evidence, first = self.partial(tmp_path, plan)
        second = attempt_body(plan, 0, 1, missing=220)
        write_once_json(acceptance.attempt_path(evidence, 0, 1), second)
        fill_step(results, plan, 0, attempt=second,
                  cells=acceptance.step_cells(plan, 0)[100:])
        rows = acceptance.read_cells(results)
        counted = {}
        for row in rows:
            counted[row["attempt_id"]] = counted.get(row["attempt_id"], 0) + 1
        assert counted == {first["attempt_id"]: 100, second["attempt_id"]: 220}

    def test_the_completion_lists_both_attempts(self, tmp_path):
        from src.training.session import write_once_json

        plan = tiny_plan()
        results, evidence, first = self.partial(tmp_path, plan)
        second = attempt_body(plan, 0, 1, missing=220)
        write_once_json(acceptance.attempt_path(evidence, 0, 1), second)
        fill_step(results, plan, 0, attempt=second,
                  cells=acceptance.step_cells(plan, 0)[100:])
        completion = seal_step(evidence, plan, 0, results,
                               attempts=[first, second])
        assert [a["attempt_id"] for a in completion["attempts"]] == [
            first["attempt_id"], second["attempt_id"]]
        assert [a["cells_written"] for a in completion["attempts"]] == \
            [100, 220]
        assert acceptance.completion_problems(completion, plan, index=0) == []

    def test_the_two_attempt_chain_verifies_end_to_end(self, tmp_path):
        from src.training.session import write_once_json

        plan = tiny_plan()
        results, evidence, first = self.partial(tmp_path, plan)
        second = attempt_body(plan, 0, 1, missing=220)
        write_once_json(acceptance.attempt_path(evidence, 0, 1), second)
        fill_step(results, plan, 0, attempt=second,
                  cells=acceptance.step_cells(plan, 0)[100:])
        seal_step(evidence, plan, 0, results, attempts=[first, second])
        problems = acceptance.evidence_problems(
            evidence, plan, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST, arms=("B",))
        assert any("step 6" in p for p in problems)     # not run yet
        assert not any("step 0" in p for p in problems), problems


class TestSealingWithoutDecoding:
    def test_a_step_with_every_cell_and_no_completion_is_unsealed(self,
                                                                  tmp_path):
        from src.training.session import write_once_json

        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        write_once_json(acceptance.attempt_path(evidence, 0, 0),
                        attempts_for(plan)[0])
        fill_step(results, plan, 0)
        assert acceptance.step_state(results, evidence, plan, 0) == \
            acceptance.STEP_UNSEALED

    def test_that_state_is_not_a_refusal(self, tmp_path):
        from src.training.session import write_once_json

        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        write_once_json(acceptance.attempt_path(evidence, 0, 0),
                        attempts_for(plan)[0])
        fill_step(results, plan, 0)
        assert acceptance.step_problems(results, plan, 0, resume=False,
                                        evidence_dir=evidence) == []

    def test_once_sealed_the_step_is_refused(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        assert any("complete and sealed" in p for p in
                   acceptance.step_problems(results, plan, 0, resume=False,
                                            evidence_dir=evidence))

    def test_sealing_records_that_it_decoded_nothing(self, tmp_path):
        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        fill_step(results, plan, 0)
        completion = seal_step(tmp_path / "evidence", plan, 0, results,
                               sealed_without_decoding=True)
        assert completion["sealed_without_decoding"] is True
        assert acceptance.completion_problems(completion, plan, index=0) == []

    def test_an_unsealed_earlier_step_blocks_the_next_one(self, tmp_path):
        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        fill_step(results, plan, 0)
        problems = acceptance.step_problems(results, plan, 1, resume=False,
                                            evidence_dir=evidence)
        assert any("Seal it before starting another step" in p
                   for p in problems), problems

    def test_a_sealed_earlier_step_lets_the_next_one_start(self, tmp_path):
        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        fill_step(results, plan, 0)
        evidence = tmp_path / "evidence"
        seal_step(evidence, plan, 0, results)
        assert acceptance.step_problems(results, plan, 1, resume=False,
                                        evidence_dir=evidence) == []


# ---------------------------------------------------------------------------
# The pack boundary
# ---------------------------------------------------------------------------

class TestWhatTravelsAndWhatDoesNot:
    def test_the_entry_point_is_on_the_allowlist(self):
        assert "scripts/25_core_eval.py" in pack.PACK_ALLOW
        assert pack.classify("scripts/25_core_eval.py")[0] == "include"

    def test_this_suite_travels_with_it(self):
        assert "tests/test_core_eval.py" in pack.PACKED_TEST_SUITES
        assert "tests/test_core_eval.py" in pack.PACK_ALLOW
        assert pack.classify("tests/test_core_eval.py")[0] == "include"

    def test_the_plan_travels_at_the_location_the_contract_names(self):
        assert acceptance.PLAN_PATH in pack.PACK_ALLOW
        assert pack.classify(acceptance.PLAN_PATH)[0] == "include"

    def test_the_plan_is_not_somewhere_the_verifier_ignores(self):
        """Under runs/ it would be a manifest entry nothing walks."""
        assert not acceptance.PLAN_PATH.startswith("runs/")
        assert not any(pack._matches(acceptance.PLAN_PATH, pattern)
                       for pattern in pack.VERIFY_IGNORE)

    def test_the_test_split_never_travels(self):
        verdict, _reason = pack.classify(acceptance.TEST_FILE)
        assert verdict == "exclude"
        assert acceptance.TEST_FILE not in pack.REQUIRED_DATA

    def test_the_final_weights_never_travel(self):
        for rel in ("runs/gpu_returns/x/runs/final_H2/adapter/"
                    "adapter_model.safetensors",
                    "artifacts/checkpoints/final_H2/adapter_model.safetensors"):
            assert pack.classify(rel)[0] == "exclude", rel

    def test_the_project_model_pointer_never_travels(self):
        """Which is why its digests are in the contract instead."""
        assert pack.classify("runs/project_model.json")[0] == "exclude"

    def test_the_scoring_module_travels_with_the_source_tree(self):
        for rel in ("src/eval/acceptance.py", "src/eval/scoring.py"):
            assert pack.classify(rel)[0] == "include", rel


class TestTheContractDocument:
    def document(self):
        return acceptance.contract_document(ROOT)

    def test_it_states_the_four_arms_and_their_settings(self):
        doc = self.document()
        assert list(doc["arms"]) == ["B", "C", "D", "E"]
        assert doc["settings_are_identical_across_arms"] is True
        assert doc["shared_settings"]["seeds"] == [0, 1, 2, 3]

    def test_it_states_the_frozen_selection(self):
        selection = self.document()["selection"]
        assert selection["pairs"] == 20
        assert selection["rows_per_pair"] == 8
        assert selection["cases"] == 160
        assert selection["seed"] == 0
        assert selection["selector"] == "src.training.lora.sample_pairs"
        assert selection["expected_sha256"] == acceptance.EXPECTED_TEST_SHA256

    def test_it_states_the_schedule_and_what_the_runner_refuses(self):
        schedule = self.document()["schedule"]
        assert len(schedule["steps"]) == 8
        assert schedule["cells_per_step"] == 320
        assert any("out of order" in r for r in schedule["runner_refuses"])

    def test_it_states_which_machine_runs_which_stage(self):
        platforms = self.document()["platforms"]
        assert "WSL2" in platforms["run"]
        for mode in ("materialize", "verify", "score"):
            assert platforms[mode] == "Mac only"

    def test_it_carries_the_whole_metric_specification(self):
        assert self.document()["metrics"] == acceptance.METRIC_SPEC

    def test_it_names_the_fields_a_plan_may_carry(self):
        doc = self.document()
        assert set(doc["plan_carries"]) == set(acceptance.CASE_FIELDS)
        assert "target" in doc["plan_refuses"]

    def test_it_names_the_final_model_and_its_digests(self):
        block = self.document()["final_model"]
        assert block["name"] == "final_H2"
        assert block["adapter_files"] == acceptance.FINAL_ADAPTER_SHA256


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------

class TestTheEntryPointGuardsEveryStage:
    """The CLI, loaded by path. ``scripts/`` is not a package."""

    def cli(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "brickagain_core_eval_cli", ROOT / "scripts" / "25_core_eval.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def mac_cli(self, monkeypatch):
        """The CLI with only the platform boundary lifted.

        ``--verify`` and ``--score`` are Mac-only, and the node that runs
        this pack is Linux, so on the node the guard returns before any of
        the logic below it runs. The nine tests using this helper are about
        that logic -- the evidence chain, the carried digests, the complete
        grid, the scorer manifest -- and a guard that refuses first turns
        every one of them into another test of the guard, green on the node
        while proving nothing it claims to prove.

        The guard itself stays tested, and by tests that do not use this
        helper: ``test_verify_and_score_refuse_off_the_mac`` here, and
        ``TestOnlyTheMacMayMaterializeVerifyOrScore`` over
        ``acceptance.mac_only_problems`` directly.
        Something still has to show that Linux refuses the real thing.

        Patching this module's ``_mac_guard`` rather than
        ``acceptance.mac_only_problems`` keeps the substitution to one
        function and one call site, so a second caller appearing later is
        not disarmed by a decision taken here.
        """
        module = self.cli()
        monkeypatch.setattr(module, "_mac_guard", lambda mode: [])
        return module

    def test_materialize_without_the_guard_writes_nothing(self, tmp_path):
        out = tmp_path / "plan.json"
        assert self.cli().main(["--materialize", "--out", str(out)]) == 2
        assert not out.exists()

    def test_the_guard_is_the_flag_the_contract_names(self):
        module = self.cli()
        assert module.TEST_GUARD == "--open-test-after-codex-approval"
        assert module.TEST_GUARD in module.build_parser().format_help()

    def test_exactly_one_mode_may_be_chosen(self):
        module = self.cli()
        assert module.main([]) == 2
        assert module.main(["--verify", "--score"]) == 2

    def test_the_contract_mode_reads_and_writes_nothing(self, capsys):
        assert self.cli().main(["--contract"]) == 0
        printed = capsys.readouterr().out
        assert acceptance.contract_digest() in printed
        assert acceptance.scorer_manifest_digest(ROOT) in printed

    def test_run_refuses_on_this_machine_before_it_reads_a_plan(self,
                                                               tmp_path):
        """The Mac is not the node; nothing about the plan matters yet."""
        import platform

        if platform.system() != "Darwin":            # pragma: no cover
            pytest.skip("not the development machine")
        module = self.cli()
        results = tmp_path / "r.jsonl"
        assert module.main(["--run", "--step", "0", "--plan",
                            str(tmp_path / "absent.json"), "--results",
                            str(results), "--evidence", str(tmp_path)]) == 2
        assert not results.exists()

    def test_run_requires_a_step_number(self, tmp_path, monkeypatch):
        module = self.cli()
        monkeypatch.setattr(acceptance, "node_only_problems",
                            lambda mode, probe: [])
        assert module.main(["--run", "--plan", str(tmp_path / "p.json"),
                            "--results", str(tmp_path / "r.jsonl"),
                            "--evidence", str(tmp_path)]) == 2

    def test_run_requires_both_carried_digests(self, tmp_path, monkeypatch):
        module = self.cli()
        plan = tiny_plan()
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        monkeypatch.setattr(acceptance, "node_only_problems",
                            lambda mode, probe: [])
        base = ["--run", "--step", "0", "--plan", str(plan_path), "--results",
                str(tmp_path / "r.jsonl"), "--evidence", str(tmp_path / "e")]
        assert module.main(base) == 2
        assert module.main(base + ["--expected-pack-digest",
                                   PACK_DIGEST]) == 2

    def test_verify_and_score_refuse_off_the_mac(self, tmp_path, monkeypatch):
        module = self.cli()
        monkeypatch.setattr(acceptance, "mac_only_problems",
                            lambda mode, **kw: [f"--{mode} runs on the Mac "
                                                "only, and this is 'Linux'"])
        for mode in ("--verify", "--score"):
            assert module.main([mode, "--plan", str(tmp_path / "p.json"),
                                "--results", str(tmp_path / "r.jsonl"),
                                "--evidence", str(tmp_path),
                                "--out", str(tmp_path / "o.json")]) == 2

    def test_the_completion_file_name_says_which_step_it_is(self, tmp_path):
        assert acceptance.completion_path(tmp_path, 0).name == \
            "step_00_even_B.json"
        assert acceptance.completion_path(tmp_path, 5).name == \
            "step_05_odd_E.json"
        assert acceptance.attempt_path(tmp_path, 5, 2).name == \
            "step_05_odd_E_attempt_02.json"

    def prepared(self, tmp_path, arms=acceptance.ARM_ORDER):
        plan = tiny_plan()
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        results, evidence = run_and_seal(tmp_path, plan, arms=arms)
        return plan, plan_path, results, evidence

    def digests(self):
        return ["--expected-pack-digest", PACK_DIGEST,
                "--expected-dependency-digest", DEPENDENCY_DIGEST]

    def test_verify_passes_with_a_complete_grid_and_its_evidence(
            self, tmp_path, monkeypatch):
        module = self.mac_cli(monkeypatch)
        _plan, plan_path, results, evidence = self.prepared(tmp_path)
        assert module.main(["--verify", "--plan", str(plan_path), "--results",
                            str(results), "--evidence", str(evidence)]
                           + self.digests()) == 0

    def test_verify_requires_evidence(self, tmp_path, monkeypatch):
        module = self.mac_cli(monkeypatch)
        _plan, plan_path, results, _evidence = self.prepared(tmp_path)
        assert module.main(["--verify", "--plan", str(plan_path), "--results",
                            str(results)] + self.digests()) == 2

    def test_verify_requires_both_carried_digests(self, tmp_path,
                                                  monkeypatch):
        module = self.mac_cli(monkeypatch)
        _plan, plan_path, results, evidence = self.prepared(tmp_path)
        assert module.main(["--verify", "--plan", str(plan_path), "--results",
                            str(results), "--evidence", str(evidence)]) == 2

    def test_a_missing_step_completion_is_reported(self, tmp_path,
                                                   monkeypatch):
        module = self.mac_cli(monkeypatch)
        _plan, plan_path, results, evidence = self.prepared(tmp_path)
        acceptance.completion_path(evidence, 3).unlink()
        assert module.main(["--verify", "--plan", str(plan_path), "--results",
                            str(results), "--evidence", str(evidence)]
                           + self.digests()) == 1

    def test_score_writes_the_contrasts_and_refuses_to_overwrite(
            self, tmp_path, monkeypatch):
        module = self.mac_cli(monkeypatch)
        _plan, plan_path, results, evidence = self.prepared(tmp_path)
        out = tmp_path / "scores.json"
        assert module.main(["--score", "--plan", str(plan_path), "--results",
                            str(results), "--evidence", str(evidence),
                            "--out", str(out)] + self.digests()) == 0
        body = json.loads(out.read_text(encoding="utf-8"))
        assert [c["contrast"] for c in body["contrasts"]["contrasts"]] == \
            ["B-C", "D-E"]
        assert body["draws"] == 2560
        assert body["scorer_source_manifest_digest"] == \
            acceptance.scorer_manifest_digest(ROOT)
        assert body["plan_scorer_source_manifest_digest"] == \
            body["scorer_source_manifest_digest"]
        assert body["evidence_manifest_digest"] == \
            acceptance.evidence_manifest_digest(evidence, _plan)
        assert len(body["evidence_manifest"]) == 16
        with pytest.raises(SystemExit):
            module.main(["--score", "--plan", str(plan_path), "--results",
                         str(results), "--evidence", str(evidence),
                         "--out", str(out)] + self.digests())

    def test_score_refuses_an_incomplete_grid(self, tmp_path,
                                              monkeypatch):
        module = self.mac_cli(monkeypatch)
        plan = tiny_plan()
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        out = tmp_path / "scores.json"
        assert module.main(["--score", "--plan", str(plan_path), "--results",
                            str(results), "--evidence", str(evidence),
                            "--out", str(out)] + self.digests()) == 2
        assert not out.exists()

    def test_score_refuses_without_evidence_and_without_digests(
            self, tmp_path, monkeypatch):
        module = self.mac_cli(monkeypatch)
        _plan, plan_path, results, evidence = self.prepared(tmp_path)
        out = tmp_path / "scores.json"
        without_evidence = ["--score", "--plan", str(plan_path), "--results",
                            str(results), "--out", str(out)] + self.digests()
        assert module.main(without_evidence) == 2
        assert not out.exists()
        without_digests = ["--score", "--plan", str(plan_path), "--results",
                           str(results), "--evidence", str(evidence),
                           "--out", str(out)]
        assert module.main(without_digests) == 2
        assert not out.exists()
        for partial in (["--expected-pack-digest", PACK_DIGEST],
                        ["--expected-dependency-digest", DEPENDENCY_DIGEST]):
            assert module.main(without_digests + partial) == 2
            assert not out.exists()

    def test_score_refuses_a_malformed_carried_digest(self, tmp_path,
                                                      monkeypatch):
        module = self.mac_cli(monkeypatch)
        _plan, plan_path, results, evidence = self.prepared(tmp_path)
        out = tmp_path / "scores.json"
        assert module.main(["--score", "--plan", str(plan_path), "--results",
                            str(results), "--evidence", str(evidence),
                            "--out", str(out), "--expected-pack-digest",
                            "not-a-digest", "--expected-dependency-digest",
                            DEPENDENCY_DIGEST]) == 2
        assert not out.exists()

    def test_score_refuses_when_the_scorer_is_not_the_plans(
            self, tmp_path, monkeypatch):
        module = self.mac_cli(monkeypatch)
        plan, plan_path, results, evidence = self.prepared(tmp_path)
        plan["scorer_source_manifest"]["src/eval/scoring.py"] = "a" * 64
        plan["scorer_source_manifest_digest"] = acceptance.digest_obj(
            plan["scorer_source_manifest"])
        plan["plan_digest"] = acceptance.plan_digest(plan)
        plan_path.unlink()
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        out = tmp_path / "scores.json"
        assert module.main(["--score", "--plan", str(plan_path), "--results",
                            str(results), "--evidence", str(evidence),
                            "--out", str(out)] + self.digests()) == 2
        assert not out.exists()

    def test_no_mode_skips_the_evidence(self):
        """There is no branch left that scores without provenance."""
        import ast

        source = (ROOT / "scripts" / "25_core_eval.py").read_text(
            encoding="utf-8")
        tree = ast.parse(source)
        for name in ("mode_verify", "mode_score"):
            node = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and n.name == name)
            body = ast.get_source_segment(source, node) or ""
            assert "evidence_problems" in body, name
            assert "_carried_digest_problems" in body, name
            assert "if args.evidence or" not in body, name


# ---------------------------------------------------------------------------
# The two-layer split, exercised without any real weights
# ---------------------------------------------------------------------------

class FakeTokenizerForSlots:
    """Every literal the grammar needs, one id each. No vocabulary file."""

    def __init__(self):
        self.pieces = ([str(i) for i in range(1, 9)]
                       + [str(i) for i in range(20)]
                       + ["x", " (", ",", ")\n"])
        self.ids = {}
        for piece in self.pieces:
            self.ids.setdefault(piece, len(self.ids) + 10)
        self.back = {v: k for k, v in self.ids.items()}
        self.eos_token_id = 9

    def encode(self, s, add_special_tokens=False):
        return [self.ids[s]]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.back.get(i, "") for i in ids)

    def apply_chat_template(self, messages, **kw):
        import torch

        return {"input_ids": torch.tensor([[1, 2, 3]])}


class ScriptedLogits:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def __call__(self, input_ids=None, past_key_values=None, use_cache=True):
        import torch

        want = self.script[self.calls] if self.calls < len(self.script) else None
        self.calls += 1
        logits = torch.full((1, 1, 4096), -1e4)
        if want is not None:
            logits[0, 0, want] = 1e4
        return type("Out", (), {"logits": logits,
                                "past_key_values": None})()

    def eval(self):
        return self

    def parameters(self):
        import torch

        yield torch.zeros(1)


def one_brick_script(slots, *, repeats=1):
    body = [slots.dims[1], slots.literal_x, slots.dims[3],
            slots.literal_open, slots.posns[0], slots.literal_comma,
            slots.posns[0], slots.literal_comma, slots.posns[0],
            slots.literal_close]
    return body * repeats + [slots.eos]


def make_scripted(repeats=1):
    from src.generation.brickgpt import BrickGPT, Slots

    tok = FakeTokenizerForSlots()
    slots = Slots.build(tok)
    model = ScriptedLogits(one_brick_script(slots, repeats=repeats))
    return BrickGPT.from_loaded(model, tok, device="cpu")


@pytest.fixture
def scripted():
    return make_scripted()


@pytest.fixture
def scripted_pair():
    """An interface with enough script for a warm-up plus one measured cell."""
    return make_scripted(repeats=1), tiny_plan()


class TestTheGpuLayerAndTheParserLayer:
    def test_from_loaded_wraps_weights_without_resolving_any(self, scripted):
        assert scripted.device == "cpu"
        assert scripted.slots.eos == 9

    def test_from_loaded_refuses_a_device_the_weights_are_not_on(self):
        from src.generation.brickgpt import BrickGPT

        with pytest.raises(ValueError, match="refusing to generate"):
            BrickGPT.from_loaded(ScriptedLogits([]), FakeTokenizerForSlots(),
                                 device="cuda")

    def test_the_raw_layer_reports_text_tokens_seconds_and_termination(
            self, scripted):
        raw = scripted.generate_raw("a chair", inventory={"2x4": 2},
                                    max_bricks=5, temperature=0.6)
        assert raw.text == "2x4 (0,0,0)\n"
        assert raw.n_tokens == 11
        assert raw.termination == "normal_eos"
        assert raw.seconds >= 0
        assert not hasattr(raw, "bricks")

    def test_generate_still_returns_the_same_thing_it_always_did(self,
                                                                 scripted):
        gen = scripted.generate("a chair", inventory={"2x4": 2}, max_bricks=5,
                                temperature=0.6)
        assert gen.text == "2x4 (0,0,0)\n"
        assert gen.n_tokens == 11
        assert gen.unparsed == []
        assert [str(b) for b in gen.bricks] == ["2x4 (0,0,0)"]

    def test_the_two_layers_agree_on_every_shared_field(self, scripted):
        raw = scripted.generate_raw("a chair", inventory={"2x4": 2},
                                    max_bricks=5, temperature=0.6)
        gen = make_scripted().generate("a chair", inventory={"2x4": 2},
                                       max_bricks=5, temperature=0.6)
        assert (gen.text, gen.n_tokens, gen.truncated) == \
               (raw.text, raw.n_tokens, raw.truncated)

    def test_the_gated_pairing_has_a_raw_form_that_shares_one_path(self):
        import inspect

        from src.constraints import inventory_decode

        for name in ("generate_with_inventory", "generate_raw_with_inventory"):
            source = inspect.getsource(getattr(inventory_decode, name))
            assert "_gated(" in source, name

    def test_a_gated_run_reports_a_ledger_and_no_parse(self, scripted):
        from src.constraints.inventory_decode import generate_raw_with_inventory
        from src.inventory.engine import Inventory

        raw, gate = generate_raw_with_inventory(
            scripted, "a chair", Inventory.from_parts({"2x4": 2}),
            max_bricks=5, temperature=0.6)
        assert gate.accepted == ["2x4"]
        assert gate.opening_inventory == {"2x4": 2}
        assert raw.termination == "normal_eos"
        ledger = acceptance.gate_ledger(acceptance.ARMS["D"],
                                        gate.opening_inventory, gate)
        assert ledger["accepted_parts"] == ["2x4"]
        assert ledger["remaining_inventory"] == {"2x4": 1}
        assert acceptance.counter_problems(ledger["counters"]) == []

    def test_an_ungated_arm_reports_no_accepted_parts_rather_than_none_used(
            self):
        ledger = acceptance.gate_ledger(acceptance.ARMS["B"], {"2x4": 2}, None)
        assert ledger["accepted_parts"] is None
        assert ledger["remaining_inventory"] is None

    def test_the_termination_vocabulary_is_the_decoders_own(self):
        from src.generation.brickgpt import BrickGate

        assert set(acceptance.TERMINATIONS) == set(BrickGate.STOP_REASONS)


class TestRunCaseProducesOneCell:
    def test_it_records_raw_text_and_never_a_brick_list(self, scripted_pair):
        interface, plan = scripted_pair
        case = plan["cases"][0]
        row = acceptance.run_case(
            interface, case, "B", 0,
            settings=acceptance.Settings(max_bricks=5, max_tokens=50),
            plan_digest_value=plan["plan_digest"], step_index=0, group="even")
        assert set(row) == set(acceptance.RESULT_FIELDS)
        assert row["raw_text"] == "2x4 (0,0,0)\n"
        assert "bricks" not in row
        assert row["gate"]["gate"] == acceptance.GATE_NONE
        assert row["step_index"] == 0 and row["group"] == "even"

    def test_the_prompt_it_built_is_the_prompt_the_case_pins(self,
                                                             scripted_pair):
        interface, plan = scripted_pair
        case = plan["cases"][0]
        row = acceptance.run_case(
            interface, case, "B", 0,
            settings=acceptance.Settings(max_bricks=5, max_tokens=50),
            plan_digest_value=plan["plan_digest"])
        assert row["prompt_sha256"] == case["prompt_sha256"]
        assert row["inventory_digest"] == case["inventory_digest"]

    def test_the_weights_it_records_come_from_the_contract(self,
                                                           scripted_pair):
        interface, plan = scripted_pair
        row = acceptance.run_case(
            interface, plan["cases"][0], "B", 0,
            settings=acceptance.Settings(max_bricks=5, max_tokens=50),
            plan_digest_value=plan["plan_digest"])
        assert row["model"] == acceptance.model_identity("B")
        assert row["model"]["adapter_files"] is None


class TestTheWarmUpRunsAndIsThrownAway:
    def test_it_decodes_the_frozen_caption_and_returns_only_evidence(self):
        interface = make_scripted(repeats=1)
        warmup = acceptance.warm_up(interface, "B")
        assert warmup["generations"] == 2
        assert warmup["seeds"] == [9001, 9002]
        assert warmup["excluded_from_every_reported_number"] is True
        assert warmup["caption_is_from_the_test_split"] is False
        assert len(warmup["seconds"]) == 2
        assert "raw_text" not in warmup and "case_id" not in warmup

    def test_it_writes_no_cell(self, tmp_path):
        interface = make_scripted(repeats=1)
        path = tmp_path / "r.jsonl"
        acceptance.warm_up(interface, "B")
        assert acceptance.read_cells(path) == []

    def test_a_gated_arm_warms_through_the_gate_too(self):
        interface = make_scripted(repeats=1)
        warmup = acceptance.warm_up(interface, "D")
        assert warmup["generations"] == 2


# ---------------------------------------------------------------------------
# The arm matrix
# ---------------------------------------------------------------------------

class TestTheArmMatrix:
    def test_there_are_exactly_four_arms(self):
        assert acceptance.ARM_ORDER == ("B", "C", "D", "E")
        assert set(acceptance.ARMS) == set(acceptance.ARM_ORDER)

    def test_b_and_d_are_the_published_model_and_c_and_e_are_final_h2(self):
        assert acceptance.ARMS["B"].model == acceptance.PUBLIC_MODEL
        assert acceptance.ARMS["D"].model == acceptance.PUBLIC_MODEL
        assert acceptance.ARMS["C"].model == acceptance.FINAL_MODEL
        assert acceptance.ARMS["E"].model == acceptance.FINAL_MODEL

    def test_the_public_arms_load_merged_and_the_final_arms_verify_digests(self):
        for name in ("B", "D"):
            assert acceptance.ARMS[name].loader.endswith("load_merged_brickgpt")
        for name in ("C", "E"):
            assert "load_finetuned" in acceptance.ARMS[name].loader
            assert "verify_digest=True" in acceptance.ARMS[name].loader

    def test_only_d_and_e_carry_a_hard_gate(self):
        assert acceptance.ARMS["B"].gate == acceptance.GATE_NONE
        assert acceptance.ARMS["C"].gate == acceptance.GATE_NONE
        assert acceptance.ARMS["D"].gate == acceptance.GATE_INVENTORY
        assert acceptance.ARMS["E"].gate == acceptance.GATE_INVENTORY
        assert acceptance.ARMS["D"].gate == acceptance.ARMS["E"].gate

    def test_every_arm_uses_the_inventory_prompt(self):
        assert {a.prompt_form for a in acceptance.ARMS.values()} == \
            {"inventory"}

    def test_the_gate_the_contract_names_is_the_one_that_exists(self):
        module, _, cls = acceptance.GATE_INVENTORY.rpartition(".")
        gate = __import__(module, fromlist=[cls])
        assert getattr(gate, cls).__name__ == "InventoryGate"

    def test_the_two_contrasts_are_frozen_with_their_direction(self):
        assert acceptance.CONTRASTS == (("B", "C"), ("D", "E"))
        assert acceptance.contrast_name("B", "C") == "B-C"


class TestEveryArmSharesEverySetting:
    def test_the_four_settings_objects_are_one_object(self):
        rendered = {canonical_json(acceptance.settings_for(name))
                    for name in acceptance.ARM_ORDER}
        assert len(rendered) == 1, rendered

    def test_the_frozen_values_are_the_declared_ones(self):
        s = acceptance.SETTINGS
        assert s.seeds == (0, 1, 2, 3)
        assert s.k == 4 == len(s.seeds)
        assert s.temperature == 0.6
        assert s.max_bricks == 80
        assert s.max_tokens == 800
        assert s.device == "cuda"
        assert s.dtype == "bfloat16"

    def test_k_cannot_disagree_with_the_seed_list(self):
        assert acceptance.Settings(seeds=(0, 1, 2)).k == 3

    def test_the_revisions_are_the_shared_pinned_ones(self):
        from src import model_ids

        s = acceptance.SETTINGS
        assert s.base_revision == model_ids.BASE_REVISION
        assert s.published_adapter_revision == model_ids.ADAPTER_REVISION
        assert s.tokenizer_revision == model_ids.TOKENIZER_REVISION

    def test_the_contract_digest_is_stable(self):
        assert acceptance.contract_digest() == acceptance.contract_digest()
        assert len(acceptance.contract_digest()) == 64


# ---------------------------------------------------------------------------
# The canonical metric specification
# ---------------------------------------------------------------------------

class TestTheMetricSpecIsCanonicalAndFrozen:
    REQUIRED = (
        "parse_success", "known_parts", "type_compliance",
        "count_overflow_amount", "count_overflow_rate",
        "macro_count_overflow_rate", "micro_count_overflow_rate",
        "inventory_valid", "in_bounds", "collision_free",
        "stud_only_connected", "touches_ground", "ldraw_serializable",
        "termination_accepted", "deterministic_core_success",
        "core_success_at_k", "unsupported_brick_count",
        "unsupported_brick_rate", "seconds", "seconds_summary",
        "paired_seconds_delta", "contrast_delta",
    )

    def test_every_reported_quantity_has_a_written_definition(self):
        for name in self.REQUIRED:
            assert name in acceptance.METRIC_SPEC, name
            assert acceptance.METRIC_SPEC[name].get("definition"), name

    def test_every_rate_states_its_numerator_and_denominator(self):
        for name, spec in acceptance.METRIC_SPEC.items():
            if spec.get("type") == "rate":
                assert spec.get("numerator"), name
                assert spec.get("denominator"), name

    def test_both_overflow_rates_exist_and_differ_in_denominator(self):
        macro = acceptance.METRIC_SPEC["macro_count_overflow_rate"]
        micro = acceptance.METRIC_SPEC["micro_count_overflow_rate"]
        assert macro["denominator"] != micro["denominator"]
        assert macro["scope"] == micro["scope"] == "arm"

    def test_the_spec_is_inside_the_contract_digest(self, monkeypatch):
        before = acceptance.contract_digest()
        altered = dict(acceptance.METRIC_SPEC)
        altered["count_overflow_rate"] = {
            **altered["count_overflow_rate"], "denominator": "something else"}
        monkeypatch.setattr(acceptance, "METRIC_SPEC_DIGEST",
                            acceptance.digest_obj(altered))
        assert acceptance.contract_digest() != before

    def test_the_spec_digest_is_the_digest_of_the_spec(self):
        assert acceptance.METRIC_SPEC_DIGEST == \
            acceptance.digest_obj(acceptance.METRIC_SPEC)

    def test_the_core_success_checks_are_named_once(self):
        assert acceptance.CORE_SUCCESS_CHECKS == (
            "parse_success", "known_parts", "type_compliance",
            "inventory_valid", "in_bounds", "collision_free",
            "stud_only_connected", "touches_ground", "ldraw_serializable",
            "termination_accepted")

    def test_the_scorer_computes_exactly_those_checks(self):
        out = scoring.score_generation("2x4 (0,0,0)\n", inventory={"2x4": 1},
                                       n_tokens=11, termination="normal_eos")
        assert set(out["checks"]) == set(acceptance.CORE_SUCCESS_CHECKS) | {
            "deterministic_core_success"}

    def test_the_quantile_method_is_named_not_borrowed(self):
        assert "R type 7" in acceptance.QUANTILE_METHOD
        assert acceptance.QUANTILE_PROBS == (0.05, 0.25, 0.5, 0.75, 0.95)

    def test_the_seconds_definition_excludes_the_model_load(self):
        spec = acceptance.METRIC_SPEC["seconds"]
        assert "model load" in spec["excludes"]
        assert "decode loop" in spec["definition"]
        assert "loads from cold" in spec["definition"]

    def test_the_required_preflight_gates_are_in_the_contract_digest(
            self, monkeypatch):
        before = acceptance.contract_digest()
        monkeypatch.setattr(acceptance, "REQUIRED_PREFLIGHT_GATES",
                            acceptance.REQUIRED_PREFLIGHT_GATES[:-1])
        assert acceptance.contract_digest() != before


class TestTheFrozenQuantile:
    def test_it_interpolates_linearly_between_ranks(self):
        assert acceptance.quantile([1, 2, 3, 4], 0.5) == 2.5
        assert acceptance.quantile([1, 2, 3, 4], 0.25) == 1.75
        assert acceptance.quantile([1, 2, 3, 4], 0.75) == 3.25

    def test_the_endpoints_are_the_endpoints(self):
        assert acceptance.quantile([5, 1, 9], 0.0) == 1
        assert acceptance.quantile([5, 1, 9], 1.0) == 9

    def test_an_empty_sample_is_null_not_zero(self):
        assert acceptance.quantile([], 0.5) is None
        assert acceptance.quantiles([])["p50"] is None

    def test_one_value_is_that_value_at_every_probability(self):
        for p in acceptance.QUANTILE_PROBS:
            assert acceptance.quantile([7.5], p) == 7.5

    def test_it_does_not_depend_on_input_order(self):
        assert acceptance.quantile([3, 1, 2], 0.5) == \
            acceptance.quantile([1, 2, 3], 0.5)

    def test_it_matches_numpys_default_linear_method(self):
        numpy = pytest.importorskip("numpy")
        sample = [0.4, 1.9, 2.2, 7.5, 7.6, 11.0, 11.1]
        for p in acceptance.QUANTILE_PROBS:
            assert acceptance.quantile(sample, p) == pytest.approx(
                float(numpy.quantile(sample, p, method="linear")))


class TestTheScorerSourceManifest:
    def test_it_names_the_parser_checker_ldraw_and_scorer(self):
        assert set(acceptance.SCORER_SOURCES) == {
            "src/data/bricks.py", "src/rendering/ldr.py",
            "src/generation/brickgpt.py", "src/eval/scoring.py",
            "src/eval/acceptance.py"}

    def test_every_named_module_is_on_disk_and_digested(self):
        manifest = acceptance.scorer_manifest(ROOT)
        assert set(manifest) == set(acceptance.SCORER_SOURCES)
        assert all(isinstance(v, str) and len(v) == 64
                   for v in manifest.values())

    def test_a_changed_module_is_reported_not_ignored(self):
        recorded = dict(acceptance.scorer_manifest(ROOT))
        recorded["src/eval/scoring.py"] = "a" * 64
        problems = acceptance.scorer_manifest_problems(recorded, ROOT)
        assert any("produced by different code" in p for p in problems)

    def test_an_absent_module_is_null_rather_than_a_digest_of_nothing(
            self, tmp_path):
        assert set(acceptance.scorer_manifest(tmp_path).values()) == {None}

    def test_a_record_naming_something_else_is_refused(self):
        recorded = dict(acceptance.scorer_manifest(ROOT))
        recorded["src/eval/oracle.py"] = "b" * 64
        assert any("does not count as scorer source" in p
                   for p in acceptance.scorer_manifest_problems(recorded, ROOT))

    def test_it_is_recorded_in_the_contract_document(self):
        scoring_block = acceptance.contract_document(ROOT)["scoring"]
        assert scoring_block["source_manifest_digest"] == \
            acceptance.scorer_manifest_digest(ROOT)

    def test_a_score_record_carries_it(self):
        record = scoring.score_record([], k=4, arms=("B",), root=ROOT)
        assert record["scorer_source_manifest_digest"] == \
            acceptance.scorer_manifest_digest(ROOT)


class TestThePlanPinsTheApprovedScorer:
    def test_the_plan_carries_the_manifest_and_its_digest(self):
        plan = tiny_plan()
        assert plan["scorer_source_manifest"] == \
            acceptance.scorer_manifest(ROOT)
        assert plan["scorer_source_manifest_digest"] == \
            acceptance.scorer_manifest_digest(ROOT)

    def test_it_is_inside_plan_digest(self):
        plan = tiny_plan()
        before = plan["plan_digest"]
        plan["scorer_source_manifest"]["src/eval/scoring.py"] = "c" * 64
        assert acceptance.plan_digest(plan) != before

    def test_a_manifest_naming_the_wrong_modules_is_refused(self):
        plan = tiny_plan()
        plan["scorer_source_manifest"].pop("src/rendering/ldr.py")
        plan["scorer_source_manifest_digest"] = acceptance.digest_obj(
            plan["scorer_source_manifest"])
        plan["plan_digest"] = acceptance.plan_digest(plan)
        assert any("counts as scorer source" in p
                   for p in acceptance.plan_problems(plan))

    def test_a_digest_the_manifest_does_not_produce_is_refused(self):
        plan = tiny_plan()
        plan["scorer_source_manifest_digest"] = "d" * 64
        plan["plan_digest"] = acceptance.plan_digest(plan)
        assert any("its own manifest does not produce" in p
                   for p in acceptance.plan_problems(plan))

    def test_plan_problems_does_not_compare_against_this_machine(self):
        """The node reads the same plan and never scores anything."""
        plan = tiny_plan()
        plan["scorer_source_manifest"] = {
            rel: "e" * 64 for rel in acceptance.SCORER_SOURCES}
        plan["scorer_source_manifest_digest"] = acceptance.digest_obj(
            plan["scorer_source_manifest"])
        plan["plan_digest"] = acceptance.plan_digest(plan)
        assert acceptance.plan_problems(plan) == []

    def test_score_time_does_compare_against_this_machine(self):
        plan = tiny_plan()
        assert acceptance.plan_scorer_problems(plan, ROOT) == []
        plan["scorer_source_manifest"]["src/eval/scoring.py"] = "f" * 64
        problems = acceptance.plan_scorer_problems(plan, ROOT)
        assert any("not the scorer on" in p for p in problems)

    def test_a_digest_that_drifted_alone_is_caught_too(self):
        plan = tiny_plan()
        plan["scorer_source_manifest_digest"] = "0" * 64
        assert any("this machine's is" in p
                   for p in acceptance.plan_scorer_problems(plan, ROOT))


# ---------------------------------------------------------------------------
# Materialising the plan
# ---------------------------------------------------------------------------

class TestThePlanTakesWholePairs:
    def test_twenty_pairs_of_eight_make_one_hundred_and_sixty_cases(
            self, fake_split):
        plan = acceptance.materialize_plan(fake_split)
        assert len(plan["cases"]) == acceptance.N_CASES == 160
        pairs = {}
        for case in plan["cases"]:
            pairs[case["pair_id"]] = pairs.get(case["pair_id"], 0) + 1
        assert len(pairs) == 20
        assert set(pairs.values()) == {8}

    def test_no_pair_is_ever_split(self, fake_split):
        plan = acceptance.materialize_plan(fake_split)
        chosen = {c["pair_id"] for c in plan["cases"]}
        ids = {c["sample_id"] for c in plan["cases"]}
        for pair in chosen:
            want = {f"{pair.replace('p', 's')}_{i}" for i in range(8)}
            assert want <= ids, pair

    def test_every_pair_is_two_roles_by_four_variants(self, fake_split):
        plan = acceptance.materialize_plan(fake_split)
        by_pair = {}
        for case in plan["cases"]:
            by_pair.setdefault(case["pair_id"], set()).add(
                (case["role"], case["variant"]))
        want = {(r, v) for r in ROLES for v in VARIANTS}
        for pair, got in by_pair.items():
            assert got == want, pair

    def test_the_selection_is_the_frozen_one(self, fake_split):
        assert acceptance.SELECTION_SEED == 0
        assert acceptance.N_PAIRS == 20
        a = acceptance.materialize_plan(fake_split)
        b = acceptance.materialize_plan(fake_split)
        assert a["plan_digest"] == b["plan_digest"]
        assert [c["case_id"] for c in a["cases"]] == \
               [c["case_id"] for c in b["cases"]]

    def test_a_file_with_a_different_digest_is_refused(self, fake_split,
                                                       monkeypatch):
        monkeypatch.setattr(acceptance, "EXPECTED_TEST_SHA256", "0" * 64)
        with pytest.raises(acceptance.PlanRefused, match="hashes to"):
            acceptance.materialize_plan(fake_split)

    def test_an_absent_file_is_refused_rather_than_invented(self, tmp_path):
        with pytest.raises(acceptance.PlanRefused, match="not on this machine"):
            acceptance.materialize_plan(tmp_path)

    def test_the_plan_verifies_against_the_contract(self, fake_split):
        plan = acceptance.materialize_plan(fake_split)
        assert acceptance.plan_problems(plan) == []

    def test_the_plan_carries_the_final_model_and_its_three_digests(
            self, fake_split):
        plan = acceptance.materialize_plan(fake_split)
        block = plan["final_model"]
        assert block["name"] == "final_H2"
        assert block["adapter_files"] == acceptance.FINAL_ADAPTER_SHA256
        assert block["weights_travel_in_the_pack"] is False
        assert block["base_revision"] == acceptance.SETTINGS.base_revision

    def test_the_plan_carries_the_frozen_schedule(self, fake_split):
        plan = acceptance.materialize_plan(fake_split)
        schedule = plan["schedule"]
        assert [(s["group"], s["arm"]) for s in schedule["steps"]] == \
            list(acceptance.STEP_ORDER)
        assert len(schedule["groups"]["even"]) == 10
        assert len(schedule["groups"]["odd"]) == 10

    def test_the_plan_carries_the_approved_scorer(self, fake_split):
        plan = acceptance.materialize_plan(fake_split)
        assert plan["scorer_source_manifest_digest"] == \
            acceptance.scorer_manifest_digest(acceptance.ROOT)


class TestNoAnswerLeavesTheMac:
    def test_a_case_carries_exactly_the_six_fields_and_two_digests(
            self, fake_split):
        plan = acceptance.materialize_plan(fake_split)
        for case in plan["cases"]:
            assert set(case) == set(acceptance.CASE_FIELDS)

    def test_no_target_reference_or_used_column_reaches_the_plan(
            self, fake_split):
        plan = acceptance.materialize_plan(fake_split)
        blob = json.dumps(plan, ensure_ascii=False)
        for banned in ("target", "used", "object_id", "bricks_txt"):
            assert f'"{banned}"' not in blob, banned
        assert acceptance.plan_leak_problems(plan) == []

    def test_a_target_smuggled_into_a_case_is_caught_by_name(self, fake_split):
        plan = acceptance.materialize_plan(fake_split)
        plan["cases"][0]["target"] = "whatever"
        problems = acceptance.plan_leak_problems(plan)
        assert any("answer field" in p for p in problems), problems

    def test_a_target_smuggled_under_an_innocent_name_is_caught_by_parsing(
            self, fake_split):
        """The field-name list alone fails open; the parser does not."""
        plan = acceptance.materialize_plan(fake_split)
        plan["cases"][0]["caption"] = "2x4 (0,0,0)\n2x4 (0,0,1)"
        problems = acceptance.plan_leak_problems(plan)
        assert any("parses as one or more bricks" in p for p in problems), \
            problems

    def test_a_real_caption_does_not_parse_as_a_brick(self):
        for caption in CAPTIONS:
            assert parse_bricks(caption, strict=False) == []

    def test_the_plan_is_written_once(self, fake_split, tmp_path):
        plan = acceptance.materialize_plan(fake_split)
        out = tmp_path / "plan.json"
        acceptance.write_plan(out, plan)
        with pytest.raises(SystemExit):
            acceptance.write_plan(out, plan)


class TestPlanProblemsChecksEveryPart:
    """One negative test per check. A validator is only as good as its list."""

    def plan(self):
        return tiny_plan()

    def reseal(self, plan):
        plan["plan_digest"] = acceptance.plan_digest(plan)
        return plan

    def test_a_good_plan_has_no_problems(self):
        assert acceptance.plan_problems(self.plan()) == []

    def test_a_non_object_is_refused(self):
        assert acceptance.plan_problems(["not", "a", "plan"])

    def test_an_extra_top_level_field_is_refused(self):
        plan = self.reseal({**self.plan(), "surprise": 1})
        assert any("top-level fields" in p
                   for p in acceptance.plan_problems(plan))

    def test_a_missing_top_level_field_is_refused(self):
        plan = self.plan()
        plan.pop("note")
        assert any("top-level fields" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_a_wrong_kind_is_refused(self):
        plan = self.reseal({**self.plan(), "kind": "something.else"})
        assert any("kind is" in p for p in acceptance.plan_problems(plan))

    def test_a_wrong_schema_version_is_refused(self):
        plan = self.reseal({**self.plan(), "schema_version": 99})
        assert any("schema_version" in p
                   for p in acceptance.plan_problems(plan))

    def test_a_wrong_contract_version_is_refused(self):
        plan = self.reseal({**self.plan(), "contract_version": 99})
        assert any("contract_version" in p
                   for p in acceptance.plan_problems(plan))

    def test_a_plan_from_another_contract_is_refused(self):
        plan = self.reseal({**self.plan(), "contract_digest": "a" * 64})
        assert any("contract" in p for p in acceptance.plan_problems(plan))

    def test_a_settings_digest_that_is_not_ours_is_refused(self):
        plan = self.reseal({**self.plan(), "settings_digest": "b" * 64})
        assert any("settings digest" in p
                   for p in acceptance.plan_problems(plan))

    def test_a_drifted_arm_definition_is_refused(self):
        plan = self.plan()
        plan["arms"]["C"] = {**plan["arms"]["C"], "gate": "none-ish"}
        assert any("arm definitions" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_a_missing_arm_is_refused(self):
        plan = self.plan()
        plan["arms"].pop("E")
        assert any("arm definitions" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_a_drifted_settings_block_is_refused(self):
        plan = self.plan()
        plan["settings"] = {**plan["settings"], "temperature": 0.9}
        assert any("settings block" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_a_drifted_final_model_block_is_refused(self):
        plan = self.plan()
        plan["final_model"] = {**plan["final_model"],
                               "adapter_files": {"adapter_model.safetensors":
                                                 "c" * 64}}
        assert any("final_model" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_a_drifted_carries_list_is_refused(self):
        plan = self.plan()
        plan["carries"] = ["case_id"]
        assert any("carries" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_incomplete_source_metadata_is_refused(self):
        for field in ("selector", "selection", "seed", "pairs",
                      "rows_per_pair", "roles", "variants"):
            plan = self.plan()
            plan["source"].pop(field)
            assert any("source block" in p
                       for p in acceptance.plan_problems(self.reseal(plan))), \
                field

    def test_a_source_naming_another_file_digest_is_refused(self):
        plan = self.plan()
        plan["source"]["sha256"] = "d" * 64
        assert any("does not pin" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_the_wrong_number_of_cases_is_refused(self):
        plan = self.plan()
        plan["cases"] = plan["cases"][:-8]
        assert any("cases, not 160" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_a_case_with_an_extra_field_is_refused(self):
        plan = self.plan()
        plan["cases"][0]["extra"] = 1
        assert any("not exactly" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_an_empty_identifier_is_refused(self):
        for field in ("case_id", "sample_id", "pair_id", "caption"):
            plan = self.plan()
            plan["cases"][0][field] = "   "
            problems = acceptance.plan_problems(self.reseal(plan))
            assert any("non-empty string" in p for p in problems), field

    def test_case_id_and_sample_id_must_be_the_same_identifier(self):
        plan = self.plan()
        plan["cases"][0]["sample_id"] = "somewhere-else"
        assert any("case_id and sample_id differ" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_an_unknown_role_is_refused(self):
        plan = self.plan()
        plan["cases"][0]["role"] = "neither"
        assert any("role 'neither'" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_an_unknown_variant_is_refused(self):
        plan = self.plan()
        plan["cases"][0]["variant"] = "sideways"
        assert any("variant 'sideways'" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_a_non_canonical_inventory_is_refused(self):
        plan = self.plan()
        plan["cases"][0]["inventory"] = {"2x4": 4, "1x2": 0}
        assert any("canonical form" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_a_part_outside_the_vocabulary_is_refused(self):
        plan = self.plan()
        plan["cases"][0]["inventory"] = {"2x8": 4}
        assert any("outside the vocabulary" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_a_doctored_prompt_digest_is_refused(self):
        plan = self.plan()
        plan["cases"][0]["prompt_sha256"] = "f" * 64
        assert any("prompt digest" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_a_doctored_inventory_digest_is_refused(self):
        plan = self.plan()
        plan["cases"][0]["inventory_digest"] = "e" * 64
        assert any("inventory digest" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_two_cases_sharing_an_id_are_refused(self):
        plan = self.plan()
        plan["cases"][1] = dict(plan["cases"][0])
        assert any("share a case_id" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_a_pair_that_is_eight_of_the_wrong_eight_is_refused(self):
        plan = self.plan()
        for case in plan["cases"][:8]:
            case["variant"] = "exact"
            case["prompt_sha256"] = acceptance.prompt_sha256(
                case["caption"], case["inventory"])
        problems = acceptance.plan_problems(self.reseal(plan))
        assert any("roles x" in p for p in problems), problems

    def test_a_pair_of_the_wrong_size_is_refused(self):
        plan = self.plan()
        plan["cases"][0]["pair_id"] = "q19"
        problems = acceptance.plan_problems(self.reseal(plan))
        assert any("cases, not 8" in p for p in problems), problems

    def test_a_rewritten_schedule_is_refused(self):
        plan = self.plan()
        plan["schedule"]["group_arm_order"]["even"] = ["B", "C", "D", "E"]
        assert any("schedule" in p
                   for p in acceptance.plan_problems(self.reseal(plan)))

    def test_a_plan_digest_that_does_not_recompute_is_refused(self):
        plan = self.plan()
        plan["cases"][0]["caption"] = "A different thing."
        plan["cases"][0]["prompt_sha256"] = acceptance.prompt_sha256(
            "A different thing.", plan["cases"][0]["inventory"])
        assert any("plan_digest" in p for p in acceptance.plan_problems(plan))

    def test_read_plan_refuses_rather_than_returns(self, tmp_path):
        plan = self.plan()
        plan["kind"] = "not.this"
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        with pytest.raises(acceptance.PlanRefused):
            acceptance.read_plan(path)


# ---------------------------------------------------------------------------
# The frozen schedule
# ---------------------------------------------------------------------------

class TestTheFrozenSchedule:
    def test_there_are_eight_steps_covering_every_arm_twice(self):
        assert acceptance.N_STEPS == 8
        arms = [a for _g, a in acceptance.STEP_ORDER]
        assert sorted(arms) == sorted(list(acceptance.ARM_ORDER) * 2)

    def test_one_group_runs_bdce_and_the_other_ceBD(self):
        assert acceptance.GROUP_ARM_ORDER["even"] == ("B", "D", "C", "E")
        assert acceptance.GROUP_ARM_ORDER["odd"] == ("C", "E", "B", "D")

    def test_neither_contrast_always_runs_the_same_arm_first(self):
        for a, b in acceptance.CONTRASTS:
            first = {}
            for group, order in acceptance.GROUP_ARM_ORDER.items():
                first[group] = a if order.index(a) < order.index(b) else b
            assert set(first.values()) == {a, b}, (a, b, first)

    def test_the_pairs_split_by_index_parity_into_ten_and_ten(self):
        plan = tiny_plan()
        schedule = acceptance.plan_schedule(plan)
        assert len(schedule["groups"]["even"]) == 10
        assert len(schedule["groups"]["odd"]) == 10
        pairs = acceptance.ordered_pair_ids(plan)
        assert schedule["groups"]["even"] == pairs[0::2]
        assert schedule["groups"]["odd"] == pairs[1::2]

    def test_every_step_covers_ten_pairs_of_eight_cases_at_four_seeds(self):
        plan = tiny_plan()
        for index in range(acceptance.N_STEPS):
            assert len(acceptance.step_cells(plan, index)) == 10 * 8 * 4

    def test_the_eight_steps_partition_the_whole_grid(self):
        plan = tiny_plan()
        seen = []
        for index in range(acceptance.N_STEPS):
            seen += acceptance.step_cells(plan, index)
        assert len(seen) == 160 * 4 * 4 == 2560
        assert len(set(seen)) == len(seen)
        assert set(seen) == set(acceptance.expected_cells(plan))

    def test_expected_cells_comes_out_in_schedule_order(self):
        plan = tiny_plan()
        arms_in_order = []
        for _d, _c, arm, _s in acceptance.expected_cells(plan):
            if not arms_in_order or arms_in_order[-1] != arm:
                arms_in_order.append(arm)
        assert arms_in_order == [a for _g, a in acceptance.STEP_ORDER]

    def test_there_are_eight_model_loads_one_per_step(self):
        """Each step is its own process, so adjacency saves nothing.

        An earlier version of this test asserted three, from counting runs of
        equal weights in ``STEP_ORDER``. That count is only meaningful inside
        one process, and no process spans two steps.
        """
        document = acceptance.contract_document(ROOT)["schedule"]
        assert "one per step, eight in total" in document["model_loads"]
        assert "no model is cached across steps" in document["model_loads"]
        assert "not part of the reported seconds" in document["model_loads"]

    def test_nothing_caches_a_model_between_steps(self):
        import ast

        for rel in ("src/eval/acceptance.py", "scripts/25_core_eval.py"):
            source = (ROOT / rel).read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and \
                        node.name in ("build_interface", "mode_run"):
                    body = ast.get_source_segment(source, node) or ""
                    for banned in ("lru_cache", "global ", "_MODEL_CACHE"):
                        assert banned not in body, (rel, node.name, banned)

    def test_a_step_number_outside_the_schedule_is_refused(self):
        for bad in (-1, 8, 99, "0", 1.0, True):
            with pytest.raises(ValueError):
                acceptance.step(bad)


class TestTheWarmUpPolicy:
    def test_it_is_frozen_with_a_caption_from_outside_the_test_split(self):
        assert acceptance.WARMUP["generations"] == 2
        assert acceptance.WARMUP["seeds"] == (9001, 9002)
        assert acceptance.WARMUP["recorded_as_measurement"] is False

    def test_its_seeds_are_not_measured_seeds(self):
        assert not set(acceptance.WARMUP["seeds"]) & \
            set(acceptance.SETTINGS.seeds)

    def test_its_caption_is_not_any_case_caption(self):
        captions = {c["caption"] for c in tiny_plan()["cases"]}
        assert acceptance.WARMUP["caption"] not in captions

    def test_it_is_part_of_the_contract_digest(self, monkeypatch):
        before = acceptance.contract_digest()
        monkeypatch.setattr(acceptance, "WARMUP",
                            {**acceptance.WARMUP, "generations": 5})
        assert acceptance.contract_digest() != before

    def test_the_schedule_records_it_for_every_step(self):
        schedule = acceptance.plan_schedule(tiny_plan())
        assert schedule["warmup"]["generations"] == 2
        assert schedule["warmup"]["seeds"] == [9001, 9002]


class TestTheRunnerRefusesToSkipOrRepeat:
    def test_step_zero_may_start_on_an_empty_file(self, tmp_path):
        plan = tiny_plan()
        assert acceptance.step_problems(tmp_path / "r.jsonl", plan, 0,
                                        resume=False,
                                        evidence_dir=tmp_path) == []

    def test_a_later_step_may_not_start_before_an_earlier_one(self, tmp_path):
        plan = tiny_plan()
        problems = acceptance.step_problems(tmp_path / "r.jsonl", plan, 1,
                                            resume=False,
                                            evidence_dir=tmp_path)
        assert any("step 0 (even/B) is 320 cells short" in p
                   for p in problems), problems

    def test_the_next_step_may_start_once_the_previous_one_is_sealed(
            self, tmp_path):
        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        fill_step(results, plan, 0)
        evidence = tmp_path / "evidence"
        seal_step(evidence, plan, 0, results)
        assert acceptance.step_problems(results, plan, 1, resume=False,
                                        evidence_dir=evidence) == []

    def test_a_sealed_step_is_never_run_again(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        for resume in (False, True):
            assert any("complete and sealed" in p for p in
                       acceptance.step_problems(results, plan, 0,
                                                resume=resume,
                                                evidence_dir=evidence))

    def test_a_partly_done_step_needs_resume(self, tmp_path):
        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        fill_step(results, plan, 0,
                  cells=acceptance.step_cells(plan, 0)[:1])
        assert any("Pass --resume" in p for p in
                   acceptance.step_problems(results, plan, 0, resume=False,
                                            evidence_dir=evidence))
        assert acceptance.step_problems(results, plan, 0, resume=True,
                                        evidence_dir=evidence) == []

    def test_a_cell_from_a_later_step_existing_first_is_refused(self, tmp_path):
        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        fill_step(results, plan, 5, cells=acceptance.step_cells(plan, 5)[:1])
        problems = acceptance.step_problems(results, plan, 0, resume=False,
                                            evidence_dir=tmp_path)
        assert any("comes after this one" in p for p in problems), problems

    def test_an_out_of_range_step_is_refused_without_reading_anything(
            self, tmp_path):
        plan = tiny_plan()
        assert acceptance.step_problems(tmp_path / "nope.jsonl", plan, 9,
                                        resume=False, evidence_dir=tmp_path)

    def test_resume_only_names_the_cells_that_are_missing(self, tmp_path):
        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        fill_step(results, plan, 0)
        missing = acceptance.missing_cells(results, plan, arms=("B",))
        assert set(missing) == set(acceptance.step_cells(plan, 6))


# ---------------------------------------------------------------------------
# Which machine may do which thing
# ---------------------------------------------------------------------------

class FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class FakeTorch:
    def __init__(self, *, cuda=False, mps=False):
        self.cuda = FakeCuda(cuda)
        self.mps_available = mps
        self.bfloat16 = "bfloat16-dtype"


class TestThereIsNoFallbackDevice:
    def test_cuda_present_resolves_to_cuda(self):
        assert acceptance.resolve_device(FakeTorch(cuda=True)) == "cuda"

    def test_no_cuda_is_a_refusal_not_a_cpu_run(self):
        with pytest.raises(acceptance.DeviceRefused, match="no CPU or MPS"):
            acceptance.resolve_device(FakeTorch(cuda=False))

    def test_mps_does_not_stand_in_for_cuda(self):
        with pytest.raises(acceptance.DeviceRefused):
            acceptance.resolve_device(FakeTorch(cuda=False, mps=True))


class TestOnlyTheNodeMayRun:
    def test_wsl2_with_cuda_passes(self):
        assert acceptance.node_only_problems("run", node_probe()) == []

    def test_a_mac_is_refused(self):
        problems = acceptance.node_only_problems(
            "run", node_probe(os_system="Darwin"))
        assert any("WSL2 Ubuntu" in p for p in problems), problems

    def test_bare_linux_is_refused(self):
        problems = acceptance.node_only_problems(
            "run", node_probe(wsl2=False,
                              wsl_evidence="kernel release does not name a "
                                           "Microsoft build"))
        assert any("not WSL2" in p for p in problems), problems

    def test_an_unreadable_platform_is_refused_rather_than_assumed(self):
        assert acceptance.node_only_problems("run", node_probe(os_system=None))
        assert acceptance.node_only_problems("run", node_probe(wsl2=None))

    def test_a_cpu_only_torch_is_refused(self):
        assert any("no CUDA build" in p for p in acceptance.node_only_problems(
            "run", node_probe(torch_cuda_build=None)))

    def test_cuda_unavailable_is_refused(self):
        assert any("different experiment" in p
                   for p in acceptance.node_only_problems(
                       "run", node_probe(cuda_available=False)))

    def test_an_empty_reading_is_refused_in_every_direction(self):
        assert len(acceptance.node_only_problems("run", {})) >= 3


class TestOnlyTheMacMayMaterializeVerifyOrScore:
    def test_darwin_passes(self):
        for mode in acceptance.MAC_ONLY_MODES:
            assert acceptance.mac_only_problems(mode, system="Darwin") == []

    def test_linux_is_refused_for_each_of_them(self):
        for mode in acceptance.MAC_ONLY_MODES:
            problems = acceptance.mac_only_problems(mode, system="Linux")
            assert any("runs on the Mac only" in p for p in problems), mode

    def test_the_three_modes_are_exactly_these(self):
        assert acceptance.MAC_ONLY_MODES == ("materialize", "verify", "score")
        assert acceptance.NODE_ONLY_MODES == ("run",)

    def test_a_mode_that_is_not_mac_only_is_refused_by_name(self):
        assert acceptance.mac_only_problems("run", system="Darwin")

    def test_this_machine_is_the_mac(self):
        import platform

        if platform.system() != "Darwin":            # pragma: no cover
            pytest.skip("not the development machine")
        for mode in acceptance.MAC_ONLY_MODES:
            assert acceptance.mac_only_problems(mode) == []


# ---------------------------------------------------------------------------
# Loaders and the final adapter
# ---------------------------------------------------------------------------

class FakeModel:
    def __init__(self, label):
        self.label = label
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        return self


def recording_loaders():
    calls = []

    def tokenizer(name, revision, **kw):
        calls.append(("tokenizer", name, revision, kw))
        return f"tokenizer:{name}"

    def merged(**kw):
        calls.append(("merged", kw))
        return FakeModel("merged"), {"load_order": ["base", "published_adapter",
                                                    "merge"]}

    def finetuned(ckpt, **kw):
        calls.append(("finetuned", str(ckpt), kw))
        return FakeModel("finetuned"), {"load_order": list(
            ("base", "published_adapter", "merge", "local_adapter"))}

    def interface(model, tok, device=None):
        calls.append(("interface", model.label, tok, device))
        return ("interface", model, tok, device)

    return calls, {"tokenizer": tokenizer, "merged": merged,
                   "finetuned": finetuned, "interface": interface}


class TestTheOnlyDoorToALoadedArm:
    def test_the_public_arms_go_through_the_merged_loader(self):
        for name in ("B", "D"):
            calls, loaders = recording_loaders()
            acceptance.build_interface(name, device="cuda", loaders=loaders,
                                       dtype="bfloat16-dtype")
            kinds = [c[0] for c in calls]
            assert "merged" in kinds and "finetuned" not in kinds, name

    def test_the_final_arms_go_through_load_finetuned_with_verify_digest(self):
        for name in ("C", "E"):
            calls, loaders = recording_loaders()
            acceptance.build_interface(name, device="cuda",
                                       adapter_dir="/somewhere/adapter",
                                       loaders=loaders, dtype="bfloat16-dtype")
            call = next(c for c in calls if c[0] == "finetuned")
            assert call[2]["verify_digest"] is True, name
            assert call[2]["device"] == "cuda", name
            assert "merged" not in [c[0] for c in calls], name

    def test_a_final_arm_without_an_adapter_directory_refuses(self):
        _calls, loaders = recording_loaders()
        with pytest.raises(ValueError, match="adapter directory"):
            acceptance.build_interface("C", device="cuda", loaders=loaders,
                                       dtype="bfloat16-dtype")

    def test_the_interface_is_built_from_already_loaded_weights(self):
        calls, loaders = recording_loaders()
        acceptance.build_interface("B", device="cuda", loaders=loaders,
                                   dtype="bfloat16-dtype")
        assert [c[0] for c in calls].count("interface") == 1

    def test_the_final_arms_never_reach_brickgpt_with_a_local_adapter(self,
                                                                     tmp_path):
        from src.generation.brickgpt import _refuse_locally_trained_adapter
        from src.model_ids import LOCAL_ADAPTER_MANIFEST

        (tmp_path / LOCAL_ADAPTER_MANIFEST).write_text("{}")
        with pytest.raises(ValueError, match="load_finetuned"):
            _refuse_locally_trained_adapter(str(tmp_path))


class TestStrictOfflineIsExplicit:
    def test_the_contract_declares_it(self):
        assert acceptance.SETTINGS.local_files_only is True

    def test_every_arm_passes_it_to_every_loader(self):
        for name in acceptance.ARM_ORDER:
            calls, loaders = recording_loaders()
            acceptance.build_interface(
                name, device="cuda", adapter_dir="/somewhere/adapter",
                loaders=loaders, dtype="bfloat16-dtype")
            for call in calls:
                if call[0] == "tokenizer":
                    assert call[3]["local_files_only"] is True, name
                elif call[0] == "merged":
                    assert call[1]["local_files_only"] is True, name
                elif call[0] == "finetuned":
                    assert call[2]["local_files_only"] is True, name

    def test_the_loaders_can_actually_take_the_flag(self):
        import inspect

        from src.generation.brickgpt import load_tokenizer
        from src.training.lora import load_finetuned, load_merged_brickgpt

        for fn in (load_tokenizer, load_merged_brickgpt, load_finetuned):
            assert "local_files_only" in inspect.signature(fn).parameters, fn


def adapter_directory(tmp_path, *, contents=None):
    contents = contents or {}
    directory = tmp_path / "adapter"
    directory.mkdir()
    for name in acceptance.FINAL_ADAPTER_FILES:
        (directory / name).write_text(contents.get(name, f"bytes of {name}"),
                                      encoding="utf-8")
    return directory


def pinned_for(directory):
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in acceptance.FINAL_ADAPTER_FILES}


class TestTheThreeDigestsAreInTheContract:
    def test_all_three_files_are_named_with_a_digest_each(self):
        assert set(acceptance.FINAL_ADAPTER_FILES) == {
            "adapter_model.safetensors", "brickagain_manifest.json",
            "adapter_config.json"}
        for name in acceptance.FINAL_ADAPTER_FILES:
            digest = acceptance.FINAL_ADAPTER_SHA256[name]
            assert isinstance(digest, str) and len(digest) == 64, name

    def test_they_are_the_digests_the_project_model_record_names(self):
        pointer = ROOT / "runs" / "project_model.json"
        if not pointer.is_file():                     # pragma: no cover
            # One prefix, so a public snapshot can enumerate exactly which
            # skips are allowed: this one is absent evidence, not a defect,
            # and it has to say which of the two it is.
            pytest.skip("artifact-only: the project model pointer is not in "
                        "this tree")
        body = json.loads(pointer.read_text(encoding="utf-8"))
        assert body["model"] == acceptance.FINAL_MODEL
        recorded = {k: v["sha256"]
                    for k, v in body["adapter"]["files"].items()}
        assert recorded == acceptance.FINAL_ADAPTER_SHA256

    def test_a_matching_directory_passes(self, tmp_path):
        directory = adapter_directory(tmp_path)
        assert acceptance.final_adapter_problems(
            directory, expected=pinned_for(directory)) == []

    @pytest.mark.parametrize("name", acceptance.FINAL_ADAPTER_FILES)
    def test_a_changed_file_is_refused(self, tmp_path, name):
        directory = adapter_directory(tmp_path)
        expected = pinned_for(directory)
        (directory / name).write_text("different bytes", encoding="utf-8")
        problems = acceptance.final_adapter_problems(directory,
                                                     expected=expected)
        assert any(name in p and "hashes to" in p for p in problems), problems

    @pytest.mark.parametrize("name", acceptance.FINAL_ADAPTER_FILES)
    def test_a_missing_file_is_refused(self, tmp_path, name):
        directory = adapter_directory(tmp_path)
        expected = pinned_for(directory)
        (directory / name).unlink()
        assert any("missing" in p for p in
                   acceptance.final_adapter_problems(directory,
                                                     expected=expected))

    def test_the_real_contract_digests_refuse_a_stand_in_directory(self,
                                                                   tmp_path):
        assert len(acceptance.final_adapter_problems(
            adapter_directory(tmp_path))) == 3

    def test_the_node_checks_through_the_plan_not_a_side_channel(self,
                                                                 tmp_path):
        problems = acceptance.plan_final_adapter_problems(
            tiny_plan(), adapter_directory(tmp_path))
        assert any("hashes to" in p for p in problems)

    def test_a_plan_whose_final_model_block_drifted_is_not_used_to_check(
            self, tmp_path):
        plan = tiny_plan()
        plan["final_model"] = {**plan["final_model"], "adapter_files": {}}
        assert acceptance.plan_final_adapter_problems(
            plan, adapter_directory(tmp_path)) == [
                "the plan's final_model block is not the frozen one; refusing "
                "to check the weights against digests that already drifted"]

    def test_no_code_here_reads_the_project_model_pointer(self):
        import ast

        for rel in ("src/eval/acceptance.py", "src/eval/scoring.py",
                    "scripts/25_core_eval.py"):
            tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value,
                                                                 str):
                    assert "project_model.json" not in node.value, rel


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------

class TestParseSuccess:
    def test_an_empty_generation_is_a_parse_failure(self):
        out = scoring.score_generation("", inventory={"2x4": 2}, n_tokens=1,
                                       termination="normal_eos")
        assert out["parse"]["n_bricks"] == 0
        assert out["parse"]["parse_success"] is False
        assert out["deterministic_core_success"] is False

    def test_an_unparsed_line_is_a_parse_failure(self):
        text = "2x4 (0,0,0)\nI am afraid I cannot do that\n"
        out = scoring.score_generation(text, inventory={"2x4": 2},
                                       n_tokens=tokens_for(1),
                                       termination="normal_eos")
        assert out["parse"]["n_unparsed_lines"] == 1
        assert out["parse"]["parse_success"] is False

    def test_tokens_and_whole_bricks_must_agree(self):
        text = "2x4 (0,0,0)\n2x4 (0,0,1)\n"
        ok = scoring.score_generation(text, inventory={"2x4": 2},
                                      n_tokens=tokens_for(2),
                                      termination="normal_eos")
        assert ok["parse"]["parse_success"] is True
        bad = scoring.score_generation(text, inventory={"2x4": 2},
                                       n_tokens=tokens_for(3),
                                       termination="normal_eos")
        assert bad["parse"]["tokens_match_complete_bricks"] is False
        assert bad["parse"]["parse_success"] is False

    def test_a_token_count_that_stops_inside_a_brick_is_reported(self):
        bricks, note = scoring.expected_complete_bricks(15, "normal_eos")
        assert bricks is None
        assert "inside a brick" in note

    def test_a_budget_stop_lands_on_a_brick_boundary(self):
        assert scoring.expected_complete_bricks(800, "max_tokens") == (80, None)


class TestPartsAndInventory:
    def test_a_part_outside_the_vocabulary_is_caught(self):
        out = scoring.score_generation("2x8 (0,0,0)\n", inventory={"2x4": 1},
                                       n_tokens=tokens_for(1),
                                       termination="normal_eos")
        assert out["parts"]["known_parts"] is False
        assert out["parts"]["unknown_parts"] == ["2x8"]
        assert out["inventory"]["type_compliance"] is False
        assert out["ldraw"]["serializable"] is False
        assert out["deterministic_core_success"] is False

    def test_type_compliance_is_about_which_parts_not_how_many(self):
        out = scoring.score_generation("1x1 (0,0,0)\n", inventory={"2x4": 5},
                                       n_tokens=tokens_for(1),
                                       termination="normal_eos")
        assert out["inventory"]["type_compliance"] is False
        assert out["inventory"]["type_violations"] == ["1x1"]

    def test_the_overflow_formula_is_the_declared_one(self):
        text = "".join(f"1x2 (0,{2 * i},0)\n" for i in range(5))
        text += "".join(f"2x4 (5,{4 * j},0)\n" for j in range(3))
        out = scoring.score_generation(text, inventory={"1x2": 2, "2x4": 4},
                                       n_tokens=tokens_for(8),
                                       termination="normal_eos")
        inv = out["inventory"]
        assert inv["used"] == {"1x2": 5, "2x4": 3}
        assert inv["count_overflow_amount"] == 3
        assert inv["count_overflow_rate"] == pytest.approx(3 / 8)
        assert inv["used_total"] == 8
        assert inv["inventory_valid"] is False
        assert inv["type_compliance"] is True

    def test_an_empty_generation_has_a_zero_overflow_rate_not_a_zero_divide(
            self):
        out = scoring.score_generation("", inventory={"2x4": 1}, n_tokens=1,
                                       termination="normal_eos")
        assert out["inventory"]["count_overflow_rate"] == 0.0

    def test_inventory_valid_is_reported_apart_from_the_two_others(self):
        out = scoring.score_generation("2x4 (0,0,0)\n", inventory={"2x4": 1},
                                       n_tokens=tokens_for(1),
                                       termination="normal_eos")
        for key in ("type_compliance", "count_overflow_amount",
                    "count_overflow_rate", "inventory_valid"):
            assert key in out["inventory"], key


class TestGeometry:
    def test_a_brick_over_the_edge_is_caught(self):
        out = scoring.score_generation("2x4 (19,0,0)\n", inventory={"2x4": 1},
                                       n_tokens=tokens_for(1),
                                       termination="normal_eos")
        assert out["geometry"]["in_bounds"] is False
        assert out["geometry"]["out_of_bounds_indices"] == [0]
        assert out["deterministic_core_success"] is False

    def test_two_bricks_in_the_same_cell_are_caught(self):
        out = scoring.score_generation("2x4 (0,0,0)\n2x4 (0,0,0)\n",
                                       inventory={"2x4": 2},
                                       n_tokens=tokens_for(2),
                                       termination="normal_eos")
        assert out["geometry"]["collision_free"] is False
        assert out["deterministic_core_success"] is False


class TestConnectivityIsStudsAlone:
    TEXT = "2x4 (0,0,0)\n2x4 (10,10,0)\n"

    def test_ground_is_not_what_holds_a_model_together(self):
        out = scoring.score_generation(self.TEXT, inventory={"2x4": 2},
                                       n_tokens=tokens_for(2),
                                       termination="normal_eos")
        assert out["connectivity"]["stud_only_connected"] is False
        assert out["connectivity"]["n_components"] == 2
        assert out["deterministic_core_success"] is False

    def test_the_ground_true_answer_is_the_one_not_used(self):
        bricks = parse_bricks(self.TEXT)
        assert is_connected(bricks, ground=True) is True
        assert is_connected(bricks, ground=False) is False

    def test_the_criterion_is_recorded_beside_the_answer(self):
        out = scoring.score_generation(self.TEXT, inventory={"2x4": 2},
                                       n_tokens=tokens_for(2),
                                       termination="normal_eos")
        assert "ground=False" in out["connectivity"]["criterion"]
        assert "ground=False" in \
            acceptance.METRIC_SPEC["stud_only_connected"]["definition"]

    def test_touching_the_ground_is_its_own_reading_and_is_required(self):
        out = scoring.score_generation("2x4 (0,0,3)\n2x4 (0,0,4)\n",
                                       inventory={"2x4": 2},
                                       n_tokens=tokens_for(2),
                                       termination="normal_eos")
        assert out["connectivity"]["stud_only_connected"] is True
        assert out["connectivity"]["touches_ground"] is False
        assert out["deterministic_core_success"] is False


class TestSupportIsDescriptiveOnly:
    def test_it_is_counted_and_not_called_stability(self):
        out = scoring.score_generation(
            "2x4 (0,0,0)\n2x4 (0,0,1)\n1x1 (0,0,3)\n",
            inventory={"2x4": 2, "1x1": 1}, n_tokens=tokens_for(3),
            termination="normal_eos")
        assert out["support"]["unsupported_brick_count"] == 1
        assert out["support"]["unsupported_brick_rate"] == pytest.approx(1 / 3)
        assert "Not a physics result" in out["support"]["note"]

    def test_it_does_not_decide_core_success(self):
        out = scoring.score_generation(
            "2x4 (0,0,0)\n2x4 (0,0,1)\n2x4 (0,0,2)\n", inventory={"2x4": 3},
            n_tokens=tokens_for(3), termination="normal_eos")
        assert out["deterministic_core_success"] is True

    def test_the_spec_says_descriptive_only(self):
        for name in ("unsupported_brick_count", "unsupported_brick_rate"):
            assert "escriptive only" in \
                acceptance.METRIC_SPEC[name]["definition"].lower()


class TestTerminationDecidesCoreSuccessToo:
    TEXT = "2x4 (0,0,0)\n2x4 (0,0,1)\n"

    @pytest.mark.parametrize("termination", ["normal_eos",
                                             "inventory_exhausted"])
    def test_the_two_accepted_reasons_pass(self, termination):
        out = scoring.score_generation(self.TEXT, inventory={"2x4": 2},
                                       n_tokens=tokens_for(2),
                                       termination=termination)
        assert out["deterministic_core_success"] is True

    @pytest.mark.parametrize("termination", ["max_bricks", "max_tokens"])
    def test_a_budget_stop_is_not_a_core_success(self, termination):
        out = scoring.score_generation(self.TEXT, inventory={"2x4": 2},
                                       n_tokens=tokens_for(2, eos=False),
                                       termination=termination)
        assert out["termination_accepted"] is False
        assert out["deterministic_core_success"] is False


class TestTheFourCountersAreNullAndNotZero:
    def test_all_four_are_declared(self):
        assert acceptance.UNIMPLEMENTED_COUNTERS == (
            "candidate_rejections", "brick_retries",
            "previous_brick_backtracks", "physics_rollbacks")

    def test_each_is_null_and_says_it_is_unimplemented(self):
        counters = acceptance.unimplemented_counters()
        for name in acceptance.UNIMPLEMENTED_COUNTERS:
            assert counters[name] == {"value": None, "implemented": False}
        assert acceptance.counter_problems(counters) == []

    def test_a_zero_is_refused_because_nothing_counted_it(self):
        counters = acceptance.unimplemented_counters()
        counters["brick_retries"] = {"value": 0, "implemented": False}
        assert any("0 would be a measurement" in p
                   for p in acceptance.counter_problems(counters))

    def test_a_counter_claiming_to_be_implemented_is_refused(self):
        counters = acceptance.unimplemented_counters()
        counters["physics_rollbacks"] = {"value": None, "implemented": True}
        assert any("no rejection or rollback layer" in p
                   for p in acceptance.counter_problems(counters))

    def test_the_scorer_reports_them_the_same_way(self):
        out = scoring.score_generation("2x4 (0,0,0)\n", inventory={"2x4": 1},
                                       n_tokens=tokens_for(1),
                                       termination="normal_eos")
        assert acceptance.counter_problems(out["counters"]) == []

    def test_no_retry_or_backtrack_machinery_was_added(self):
        for name in acceptance.UNIMPLEMENTED_COUNTERS:
            assert acceptance.METRIC_SPEC.get(name) is None


# ---------------------------------------------------------------------------
# Aggregation, contrasts and timing
# ---------------------------------------------------------------------------

def draw(case_id, arm, seed, success, *, role="control", variant="exact",
         seconds=1.0, overflow_rate=0.0, overflow_amount=0, used_total=4):
    """One scored draw, with only the fields the aggregators read."""
    return {
        "case_id": case_id, "pair_id": case_id[:3], "role": role,
        "variant": variant, "arm": arm, "seed": seed, "seconds": seconds,
        "n_tokens": 21, "step_index": 0, "group": "even",
        "checks": {name: success for name in scoring.DRAW_BOOLEANS},
        "inventory": {"count_overflow_rate": overflow_rate,
                      "count_overflow_amount": overflow_amount,
                      "used_total": used_total},
        "support": {"unsupported_brick_rate": 0.0},
        "termination": "normal_eos",
        "deterministic_core_success": success,
    }


def four_seeds(case_id, arm, successes, **kw):
    return [draw(case_id, arm, s, s in successes, **kw) for s in range(4)]


class TestCoreSuccessAtFour:
    def test_one_success_in_four_counts_the_case(self):
        out = scoring.core_success_at_k(four_seeds("c1", "D", {2}), k=4,
                                        arms=("D",))
        overall = out["by_arm"]["D"]["overall"]
        assert overall["numerator"] == 1
        assert overall["denominator"] == 1
        assert overall["value"] == 1.0

    def test_four_failures_do_not(self):
        out = scoring.core_success_at_k(four_seeds("c1", "D", set()), k=4,
                                        arms=("D",))
        assert out["by_arm"]["D"]["overall"]["value"] == 0.0

    def test_a_case_missing_a_seed_is_incomplete_not_scored(self):
        scores = four_seeds("c1", "D", {0})[:3]
        overall = scoring.core_success_at_k(
            scores, k=4, arms=("D",))["by_arm"]["D"]["overall"]
        assert overall["denominator"] == 0
        assert overall["incomplete_cases"] == ["c1"]
        assert overall["value"] is None

    def test_the_rate_is_over_cases_not_draws(self):
        scores = four_seeds("c1", "E", {0}) + four_seeds("c2", "E", set())
        overall = scoring.core_success_at_k(
            scores, k=4, arms=("E",))["by_arm"]["E"]["overall"]
        assert overall["denominator"] == 2
        assert overall["value"] == 0.5

    def test_k_comes_from_the_contract(self):
        assert acceptance.SETTINGS.k == 4

    def test_it_is_reported_per_role_and_per_variant_too(self):
        scores = (four_seeds("c1", "B", {0}, role="control", variant="exact")
                  + four_seeds("c2", "B", set(), role="counterfactual",
                               variant="mixed"))
        by_arm = scoring.core_success_at_k(scores, k=4, arms=("B",))["by_arm"]
        assert by_arm["B"]["role=control"]["value"] == 1.0
        assert by_arm["B"]["role=counterfactual"]["value"] == 0.0
        assert by_arm["B"]["variant=exact"]["value"] == 1.0
        assert by_arm["B"]["variant=loose"]["value"] is None


class TestMacroAndMicroOverflowAreBothReported:
    def scores(self, arm="B"):
        return [draw("c1", arm, 0, False, overflow_rate=1.0,
                     overflow_amount=1, used_total=1),
                draw("c2", arm, 0, True, overflow_rate=0.0,
                     overflow_amount=0, used_total=99)]

    def test_the_macro_rate_is_the_unweighted_mean_of_draw_rates(self):
        summary = scoring.per_arm_summary(self.scores(), arms=("B",),
                                          k=4)["B"]["overall"]
        assert summary["macro_count_overflow_rate"]["value"] == 0.5
        assert summary["macro_count_overflow_rate"]["denominator"] == 2

    def test_the_micro_rate_pools_over_bricks(self):
        micro = scoring.per_arm_summary(
            self.scores(), arms=("B",),
            k=4)["B"]["overall"]["micro_count_overflow_rate"]
        assert micro["numerator"] == 1
        assert micro["denominator"] == 100
        assert micro["value"] == pytest.approx(0.01)

    def test_they_differ_and_neither_stands_in_for_the_other(self):
        summary = scoring.per_arm_summary(self.scores(), arms=("B",),
                                          k=4)["B"]["overall"]
        assert summary["macro_count_overflow_rate"]["value"] != \
            summary["micro_count_overflow_rate"]["value"]

    def test_an_empty_stratum_is_null_rather_than_zero(self):
        summary = scoring.per_arm_summary(self.scores(), arms=("B",),
                                          k=4)["B"]["variant=loose"]
        assert summary["draws"] == 0
        assert summary["macro_count_overflow_rate"]["value"] is None
        assert summary["micro_count_overflow_rate"]["value"] is None


class TestEveryRateCarriesItsDenominator:
    def test_per_arm_boolean_rates_do(self):
        rates = scoring.per_arm_summary(four_seeds("c1", "B", {0, 1}),
                                        arms=("B",),
                                        k=4)["B"]["overall"]["rates"]
        assert rates["parse_success"] == {"numerator": 2, "denominator": 4,
                                          "value": 0.5}

    def test_a_rate_over_nothing_is_null_not_zero(self):
        rates = scoring.per_arm_summary([], arms=("B",),
                                        k=4)["B"]["overall"]["rates"]
        assert rates["parse_success"] == {"numerator": 0, "denominator": 0,
                                          "value": None}


class TestThePairedContrasts:
    def scores(self):
        out = []
        for case, role, variant in (("c01", "control", "exact"),
                                    ("c02", "counterfactual", "mixed")):
            out += four_seeds(case, "B", {0}, role=role, variant=variant,
                              seconds=2.0)
            out += four_seeds(case, "C", {0, 1, 2, 3}, role=role,
                              variant=variant, seconds=1.5)
            out += four_seeds(case, "D", {0}, role=role, variant=variant,
                              seconds=2.5)
            out += four_seeds(case, "E", {0, 1}, role=role, variant=variant,
                              seconds=1.0)
        return out

    def test_both_contrasts_are_produced_directly(self):
        out = scoring.paired_comparisons(self.scores(), k=4)
        assert [c["contrast"] for c in out["contrasts"]] == ["B-C", "D-E"]

    def test_the_direction_is_a_minus_b_and_is_stated(self):
        first = scoring.paired_comparisons(self.scores(), k=4)["contrasts"][0]
        assert first["direction"] == "every delta is value(B) - value(C)"
        metric = first["strata"][0]["metrics"]["deterministic_core_success"]
        assert metric["a_value"] == 0.25
        assert metric["b_value"] == 1.0
        assert metric["delta"] == pytest.approx(-0.75)

    def test_every_comparison_shows_both_raw_values_and_both_denominators(self):
        out = scoring.paired_comparisons(self.scores(), k=4)
        for contrast in out["contrasts"]:
            for stratum in contrast["strata"]:
                for name, metric in stratum["metrics"].items():
                    assert set(metric) >= {
                        "a_value", "b_value", "delta", "a_numerator",
                        "a_denominator", "b_numerator", "b_denominator",
                        "denominators_match"}, name

    def test_overall_role_and_variant_strata_are_all_present(self):
        keys = [s["stratum"]["key"] for s in scoring.paired_comparisons(
            self.scores(), k=4)["contrasts"][0]["strata"]]
        assert keys[0] == "overall"
        for role in ROLES:
            assert f"role={role}" in keys
        for variant in VARIANTS:
            assert f"variant={variant}" in keys

    def test_core_success_at_k_is_contrasted_with_its_case_denominator(self):
        metric = scoring.paired_comparisons(self.scores(), k=4)[
            "contrasts"][0]["strata"][0]["metrics"]["core_success_at_k"]
        assert metric["a_denominator"] == 2
        assert metric["b_denominator"] == 2
        assert metric["a_value"] == 1.0 and metric["b_value"] == 1.0
        assert metric["delta"] == 0.0

    def test_both_overflow_rates_are_contrasted(self):
        metrics = scoring.paired_comparisons(self.scores(), k=4)[
            "contrasts"][0]["strata"][0]["metrics"]
        assert "macro_count_overflow_rate" in metrics
        assert "micro_count_overflow_rate" in metrics

    def test_a_stratum_with_no_draws_gives_a_null_delta_not_zero(self):
        strata = scoring.paired_comparisons(self.scores(), k=4)[
            "contrasts"][0]["strata"]
        empty = next(s for s in strata
                     if s["stratum"]["key"] == "variant=loose")
        assert empty["a_draws"] == empty["b_draws"] == 0
        assert empty["metrics"]["deterministic_core_success"]["delta"] is None
        assert empty["metrics"]["deterministic_core_success"]["a_value"] is None

    def test_no_ratio_or_significance_is_produced(self):
        """Checked over keys: the note names these words to disown them."""
        def keys(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    yield k
                    yield from keys(v)
            elif isinstance(node, list):
                for v in node:
                    yield from keys(v)

        out = scoring.paired_comparisons(self.scores(), k=4)
        for key in keys(out):
            lowered = str(key).lower()
            for banned in ("p_value", "pvalue", "significan", "confidence",
                           "ratio", "percent", "relative"):
                assert banned not in lowered, key
        assert "no significance claim" in out["note"]
        assert "absolute difference" in \
            acceptance.METRIC_SPEC["contrast_delta"]["definition"]


class TestTiming:
    def scores(self):
        out = []
        for i, seconds in enumerate((1.0, 2.0, 3.0, 4.0)):
            out.append(draw("c01", "B", i, True, seconds=seconds))
            out.append(draw("c01", "C", i, True, seconds=seconds - 0.5))
        return out

    def test_every_cell_keeps_its_raw_value(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "r.jsonl"
        acceptance.append_cell(path, cell_for(plan, plan["cases"][0], "B", 0,
                                              seconds=1.2345678901234567))
        assert acceptance.read_cells(path)[0]["seconds"] == 1.2345678901234567

    def test_no_duration_is_rounded_anywhere(self):
        import ast

        for rel in ("src/eval/acceptance.py", "src/eval/scoring.py"):
            source = (ROOT / rel).read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "round"):
                    continue
                rounded = ast.get_source_segment(source, node.args[0]) or ""
                assert "second" not in rounded.lower(), (rel, rounded)

    def test_run_case_stores_the_measured_float(self, scripted_pair):
        interface, plan = scripted_pair
        row = acceptance.run_case(
            interface, plan["cases"][0], "B", 0,
            settings=acceptance.Settings(max_bricks=5, max_tokens=50),
            plan_digest_value=plan["plan_digest"])
        assert isinstance(row["seconds"], float)
        assert row["seconds"] == float(row["seconds"])

    def test_the_per_arm_summary_reports_n_total_mean_and_quantiles(self):
        summary = scoring.per_arm_summary(self.scores(), arms=("B",),
                                          k=4)["B"]["overall"]["seconds"]
        assert summary["n"] == 4
        assert summary["total"] == 10.0
        assert summary["mean"] == 2.5
        assert summary["min"] == 1.0 and summary["max"] == 4.0
        assert summary["quantiles"]["p50"] == 2.5
        assert "R type 7" in summary["quantile_method"]

    def test_the_paired_delta_is_per_cell_not_a_difference_of_means(self):
        out = scoring.paired_seconds(self.scores(), "B", "C", "overall", None)
        assert out["pairs_compared"] == 4
        assert out["mean"] == pytest.approx(0.5)
        assert out["median"] == pytest.approx(0.5)
        assert out["quantiles"]["p50"] == pytest.approx(0.5)

    def test_a_cell_only_one_arm_measured_is_excluded_and_counted(self):
        scores = [s for s in self.scores()
                  if not (s["arm"] == "C" and s["seed"] == 3)]
        out = scoring.paired_seconds(scores, "B", "C", "overall", None)
        assert out["pairs_compared"] == 3
        assert out["unpaired_a"] == 1
        assert out["unpaired_b"] == 0

    def test_the_paired_delta_appears_in_every_stratum_of_every_contrast(self):
        out = scoring.paired_comparisons(self.scores(), k=4)
        for contrast in out["contrasts"]:
            for stratum in contrast["strata"]:
                assert "paired_seconds_delta" in stratum
                assert "quantile_method" in stratum["paired_seconds_delta"]

    def test_mean_seconds_is_also_contrasted_directly(self):
        metrics = scoring.paired_comparisons(self.scores(), k=4)[
            "contrasts"][0]["strata"][0]["metrics"]
        assert metrics["mean_seconds"]["a_value"] == 2.5
        assert metrics["mean_seconds"]["b_value"] == 2.0
        assert metrics["mean_seconds"]["delta"] == pytest.approx(0.5)


class TestNothingIsCalledWhatItIsNot:
    def all_keys(self, node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from self.all_keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from self.all_keys(v)

    def check(self, body):
        for key in self.all_keys(body):
            lowered = str(key).lower()
            for term in acceptance.FORBIDDEN_METRIC_TERMS:
                assert term.replace(" ", "_") not in lowered, (key, term)

    def test_the_scorer_produces_no_forbidden_metric_name(self):
        self.check(scoring.score_generation(
            "2x4 (0,0,0)\n", inventory={"2x4": 1}, n_tokens=tokens_for(1),
            termination="normal_eos"))

    def test_no_summary_or_contrast_is_named_one_of_them_either(self):
        rows = four_seeds("c01", "B", {0}) + four_seeds("c01", "C", {1})
        self.check(scoring.score_record(rows, k=4, arms=("B", "C"), root=ROOT))

    def test_the_forbidden_names_appear_only_as_a_declaration(self):
        self.check(acceptance.contract_document(ROOT))

    def test_the_contract_says_which_names_are_refused(self):
        refused = acceptance.contract_document(ROOT)["scoring"]["not_reported"]
        for term in ("stability", "semantic_success", "full_success",
                     "render_quality"):
            assert term in refused


# ---------------------------------------------------------------------------
# Results: append-only, no clobber, resume only fills gaps
# ---------------------------------------------------------------------------

class TestResultsAreAppendOnly:
    def test_a_cell_is_written_once(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        row = cell_for(plan, plan["cases"][0], "B", 0)
        acceptance.append_cell(path, row)
        with pytest.raises(acceptance.ResultsRefused, match="already recorded"):
            acceptance.append_cell(path, row)

    def test_a_rewritten_cell_is_refused_even_with_different_text(self,
                                                                 tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        acceptance.append_cell(path, cell_for(plan, plan["cases"][0], "B", 0))
        with pytest.raises(acceptance.ResultsRefused):
            acceptance.append_cell(path, cell_for(plan, plan["cases"][0], "B",
                                                  0, raw_text="1x1 (0,0,0)\n"))
        assert len(acceptance.read_cells(path)) == 1

    def test_an_incomplete_row_is_refused_before_it_is_written(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        row = cell_for(plan, plan["cases"][0], "B", 0)
        row.pop("termination")
        with pytest.raises(acceptance.ResultsRefused, match="termination"):
            acceptance.append_cell(path, row)
        assert not path.exists()

    def test_a_row_without_its_step_or_attempt_is_refused(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        for field in ("step_index", "attempt_id", "attempt_digest"):
            row = cell_for(plan, plan["cases"][0], "B", 0)
            row.pop(field)
            with pytest.raises(acceptance.ResultsRefused, match=field):
                acceptance.append_cell(path, row)

    def test_the_same_cell_under_a_different_arm_is_a_different_cell(self,
                                                                    tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        for arm in ("B", "D"):
            acceptance.append_cell(path, cell_for(plan, plan["cases"][0], arm,
                                                  0))
        assert len(acceptance.read_cells(path)) == 2


class TestTheValidator:
    def test_a_complete_grid_verifies(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B", "D"))
        assert acceptance.validate_results(path, plan, arms=("B", "D")) == []

    def test_the_whole_four_arm_grid_verifies(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=acceptance.ARM_ORDER)
        assert len(acceptance.read_cells(path)) == 2560
        assert acceptance.validate_results(path, plan) == []

    def test_a_missing_cell_is_refused(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        rewrite(path, acceptance.read_cells(path)[:-1])
        assert any("never measured" in p for p in
                   acceptance.validate_results(path, plan, arms=("B",)))

    def test_a_duplicate_cell_is_refused(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        with path.open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(acceptance.read_cells(path)[0]) + "\n")
        assert any("recorded 2 times" in p for p in
                   acceptance.validate_results(path, plan, arms=("B",)))

    def test_a_cell_no_plan_predetermined_is_refused(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        stray = cell_for(plan, plan["cases"][0], "B", 0)
        stray["seed"] = 9
        with path.open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(stray) + "\n")
        assert any("not one of [0, 1, 2, 3]" in p for p in
                   acceptance.validate_results(path, plan, arms=("B",)))

    def test_a_row_attributed_to_the_wrong_step_is_refused(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        rows = acceptance.read_cells(path)
        rows[0]["step_index"] = 6
        rows[0]["group"] = "odd"
        rewrite(path, rows)
        assert any("frozen schedule puts it in step" in p for p in
                   acceptance.validate_results(path, plan, arms=("B",)))

    def test_a_row_with_no_attempt_is_refused(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        rows = acceptance.read_cells(path)
        rows[0]["attempt_id"] = ""
        rows[1]["attempt_digest"] = "short"
        rewrite(path, rows)
        problems = acceptance.validate_results(path, plan, arms=("B",))
        assert any("names no attempt" in p for p in problems)
        assert any("attempt_digest is" in p for p in problems)

    def test_an_arm_that_ran_the_wrong_gate_is_refused(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        rows = acceptance.read_cells(path)
        rows[0]["gate"]["gate"] = acceptance.GATE_INVENTORY
        rewrite(path, rows)
        assert any("gate" in p for p in
                   acceptance.validate_results(path, plan, arms=("B",)))

    def test_a_settings_drift_is_refused(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        rows = acceptance.read_cells(path)
        rows[0]["settings_digest"] = "b" * 64
        rewrite(path, rows)
        assert any("different settings" in p for p in
                   acceptance.validate_results(path, plan, arms=("B",)))

    def test_a_contract_drift_is_refused(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        rows = acceptance.read_cells(path)
        rows[0]["contract_digest"] = "c" * 64
        rewrite(path, rows)
        assert any("different contract" in p for p in
                   acceptance.validate_results(path, plan, arms=("B",)))

    def test_a_prompt_digest_that_is_not_the_plans_is_refused(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        rows = acceptance.read_cells(path)
        rows[0]["prompt_sha256"] = "c" * 64
        rewrite(path, rows)
        assert any("not the prompt the plan pins" in p for p in
                   acceptance.validate_results(path, plan, arms=("B",)))

    def test_an_inventory_digest_that_is_not_the_plans_is_refused(self,
                                                                  tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        rows = acceptance.read_cells(path)
        rows[0]["inventory_digest"] = "d" * 64
        rewrite(path, rows)
        assert any("not the inventory the plan pins" in p for p in
                   acceptance.validate_results(path, plan, arms=("B",)))

    def test_an_adapter_digest_that_does_not_match_is_refused(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("C",))
        rows = acceptance.read_cells(path)
        rows[0]["model"] = {
            **rows[0]["model"],
            "adapter_files": {n: "e" * 64
                              for n in acceptance.FINAL_ADAPTER_FILES}}
        rewrite(path, rows)
        assert any("not the weights arm C is defined to run" in p
                   for p in acceptance.validate_results(path, plan,
                                                        arms=("C",)))

    def test_a_public_arm_claiming_a_local_adapter_is_refused(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        rows = acceptance.read_cells(path)
        rows[0]["model"] = {**rows[0]["model"],
                            "adapter_files": dict(
                                acceptance.FINAL_ADAPTER_SHA256)}
        rewrite(path, rows)
        assert any("not the weights arm B is defined to run" in p
                   for p in acceptance.validate_results(path, plan,
                                                        arms=("B",)))

    def test_an_incomplete_result_row_is_refused(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        rows = acceptance.read_cells(path)
        rows[0]["raw_text"] = None
        rows[1]["n_tokens"] = 0
        rows[2]["termination"] = "gave_up"
        rows[3]["seconds"] = -1.0
        rewrite(path, rows)
        problems = acceptance.validate_results(path, plan, arms=("B",))
        assert any("raw_text is not text" in p for p in problems)
        assert any("n_tokens is 0" in p for p in problems)
        assert any("termination 'gave_up'" in p for p in problems)
        assert any("seconds is -1.0" in p for p in problems)

    def test_a_truncated_line_is_reported_rather_than_raised(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"plan_digest": "abc", "case_')
        assert any("stopped mid-append" in p for p in
                   acceptance.validate_results(path, plan, arms=("B",)))

    def test_results_belonging_to_another_plan_are_refused(self, tmp_path):
        plan = tiny_plan()
        other = tiny_plan(label="other ")
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        rows = acceptance.read_cells(path)
        rows[0]["plan_digest"] = other["plan_digest"]
        rewrite(path, rows)
        assert any("belongs to plan" in p for p in
                   acceptance.validate_results(path, plan, arms=("B",)))

    def test_a_plan_that_does_not_verify_stops_the_results_being_scored(
            self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "results.jsonl"
        fill(path, plan, arms=("B",))
        plan["kind"] = "not.this"
        assert any("kind is" in p for p in
                   acceptance.validate_results(path, plan, arms=("B",)))


# ---------------------------------------------------------------------------
# One validator, three callers
# ---------------------------------------------------------------------------

class TestTheSingleStepChainValidator:
    """``--verify``, the sealer and the runner all ask this one question."""

    def chain(self, evidence, plan, index, results, **kw):
        kw.setdefault("expected_pack_digest", PACK_DIGEST)
        kw.setdefault("expected_dependency_digest", DEPENDENCY_DIGEST)
        return acceptance.step_chain_problems(
            evidence, plan, index, results_path=results, **kw)

    def test_a_complete_sealed_step_has_no_problems(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        assert self.chain(evidence, plan, 0, results) == []

    def test_a_complete_unsealed_step_has_none_either_in_the_seal_form(
            self, tmp_path):
        from src.training.session import write_once_json

        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        write_once_json(acceptance.attempt_path(evidence, 0, 0),
                        attempts_for(plan)[0])
        fill_step(results, plan, 0)
        assert self.chain(evidence, plan, 0, results,
                          require_completion=False) == []

    def test_the_seal_form_refuses_a_completion_that_already_exists(self,
                                                                    tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        assert any("a step is sealed once" in p for p in
                   self.chain(evidence, plan, 0, results,
                              require_completion=False))

    def test_it_checks_the_step_is_complete(self, tmp_path):
        from src.training.session import write_once_json

        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        write_once_json(acceptance.attempt_path(evidence, 0, 0),
                        attempts_for(plan)[0])
        fill_step(results, plan, 0, cells=acceptance.step_cells(plan, 0)[:10])
        problems = self.chain(evidence, plan, 0, results,
                              require_completion=False)
        assert any("310 of 320 cells are missing" in p
                   for p in problems), problems

    def test_it_checks_the_rows_themselves(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        rows = acceptance.read_cells(results)
        rows[0]["termination"] = "gave_up"
        rewrite(results, rows)
        assert any("termination 'gave_up'" in p for p in
                   self.chain(evidence, plan, 0, results))

    def test_it_checks_a_cell_that_claims_a_step_it_is_not_in(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        rows = acceptance.read_cells(results)
        rows[-1]["step_index"] = 0
        rows[-1]["group"] = "even"
        rewrite(results, rows)
        assert any("claims this step and is not one of its cells" in p
                   for p in self.chain(evidence, plan, 0, results))

    def test_it_checks_a_step_with_no_attempt_records_at_all(self, tmp_path):
        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        fill_step(results, plan, 0)
        assert any("no attempt records" in p for p in
                   self.chain(evidence, plan, 0, results,
                              require_completion=False))

    def test_it_survives_an_unreadable_attempt_file(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        path = acceptance.attempt_path(evidence, 0, 0)
        path.unlink()
        path.write_text("{not json", encoding="utf-8")
        problems = self.chain(evidence, plan, 0, results)
        assert any("not readable" in p for p in problems), problems

    def test_it_survives_an_attempt_file_that_is_not_an_object(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        path = acceptance.attempt_path(evidence, 0, 0)
        path.unlink()
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert any("not an object" in p for p in
                   self.chain(evidence, plan, 0, results))

    @pytest.mark.parametrize("field", acceptance.ATTEMPT_FIELDS)
    def test_a_missing_attempt_field_is_a_sentence_not_an_exception(
            self, tmp_path, field):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        path = acceptance.attempt_path(evidence, 0, 0)
        body = json.loads(path.read_text(encoding="utf-8"))
        body.pop(field)
        path.unlink()
        path.write_text(json.dumps(body), encoding="utf-8")
        problems = self.chain(evidence, plan, 0, results)
        assert problems, field
        assert all(isinstance(p, str) for p in problems), field

    def test_a_failed_preflight_is_reported_by_the_seal_form_too(self,
                                                                 tmp_path):
        from src.training.session import write_once_json

        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        broken = attempt_body(plan, 0, preflight=preflight_stub(fail=("cuda",)))
        write_once_json(acceptance.attempt_path(evidence, 0, 0), broken)
        fill_step(results, plan, 0, attempt=broken)
        assert any("did not pass" in p for p in
                   self.chain(evidence, plan, 0, results,
                              require_completion=False))

    def test_the_carried_digests_are_checked_by_every_form(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        for kw in ({"require_completion": True},
                   {"require_completion": False}):
            problems = self.chain(evidence, plan, 0, results,
                                  expected_pack_digest="9" * 64, **kw)
            assert any("carried from the build machine" in p
                       for p in problems), kw


class TestTheCompletionEntryIsDerivedNotWritten:
    #: ``attempt_id`` is the key the entry is matched by, so drifting it does
    #: not produce a mismatched entry -- it produces an entry for an attempt
    #: that does not exist, which is checked separately below.
    COMPARED = tuple(f for f in acceptance.ATTEMPT_REFERENCE_FIELDS
                     if f != "attempt_id")

    @pytest.mark.parametrize("field", COMPARED)
    def test_every_field_of_a_reference_must_match_its_attempt(self, tmp_path,
                                                               field):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        path = acceptance.completion_path(evidence, 0)
        body = json.loads(path.read_text(encoding="utf-8"))
        original = body["attempts"][0][field]
        body["attempts"][0][field] = (original + 1
                                      if isinstance(original, int)
                                      else "drifted")
        path.unlink()
        path.write_text(json.dumps(body), encoding="utf-8")
        problems = acceptance.evidence_problems(
            evidence, plan, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST, arms=("B",))
        assert any(f"'{field}'" in p and "differs from the attempt record" in p
                   for p in problems), (field, problems)

    def test_a_drifted_attempt_id_leaves_an_entry_for_nothing(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        path = acceptance.completion_path(evidence, 0)
        body = json.loads(path.read_text(encoding="utf-8"))
        body["attempts"][0]["attempt_id"] = "invented"
        path.unlink()
        path.write_text(json.dumps(body), encoding="utf-8")
        problems = acceptance.evidence_problems(
            evidence, plan, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST, arms=("B",))
        assert any("no such attempt record is here" in p for p in problems)
        assert any("does not list it" in p for p in problems)
        assert any("does not list" in p and "320 cells name" in p
                   for p in problems)

    def test_an_entry_with_an_extra_field_is_refused(self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        path = acceptance.completion_path(evidence, 0)
        body = json.loads(path.read_text(encoding="utf-8"))
        body["attempts"][0]["note"] = "hello"
        path.unlink()
        path.write_text(json.dumps(body), encoding="utf-8")
        assert any("not\n" not in p and "fields" in p for p in
                   acceptance.evidence_problems(
                       evidence, plan, results_path=results,
                       expected_pack_digest=PACK_DIGEST,
                       expected_dependency_digest=DEPENDENCY_DIGEST,
                       arms=("B",)))

    def test_the_reference_a_good_run_writes_is_exactly_the_derived_one(
            self, tmp_path):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        body = json.loads(acceptance.completion_path(
            evidence, 0).read_text(encoding="utf-8"))
        attempt = json.loads(acceptance.attempt_path(
            evidence, 0, 0).read_text(encoding="utf-8"))
        assert body["attempts"][0] == acceptance.attempt_reference(attempt, 320)


class TestAPublicArmRecordsNoAdapter:
    def test_b_and_d_attempts_carry_none(self):
        plan = tiny_plan()
        for index, (_group, name) in enumerate(acceptance.STEP_ORDER):
            if acceptance.ARMS[name].model == acceptance.PUBLIC_MODEL:
                assert attempt_body(plan, index)["adapter"] is None, index

    @pytest.mark.parametrize("index", [i for i, (_g, a)
                                       in enumerate(acceptance.STEP_ORDER)
                                       if acceptance.ARMS[a].model
                                       == acceptance.PUBLIC_MODEL])
    def test_a_public_arm_that_records_an_adapter_is_refused(self, index):
        plan = tiny_plan()
        body = attempt_body(plan, index)
        body["adapter"] = {"directory_name": "adapter",
                           "files": dict(acceptance.FINAL_ADAPTER_SHA256),
                           "checked_against": "the plan's final_model block"}
        problems = acceptance.attempt_problems(
            body, plan, expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST)
        assert any("take no local delta" in p for p in problems), problems

    def test_the_final_arms_still_require_one(self):
        plan = tiny_plan()
        for index, (_group, name) in enumerate(acceptance.STEP_ORDER):
            if acceptance.ARMS[name].model == acceptance.PUBLIC_MODEL:
                continue
            body = attempt_body(plan, index)
            body["adapter"] = None
            assert any("records no adapter" in p for p in
                       acceptance.attempt_problems(
                           body, plan, expected_pack_digest=PACK_DIGEST,
                           expected_dependency_digest=DEPENDENCY_DIGEST))


class TestAPredecessorIsReplayedNotAssumed:
    def test_a_good_predecessor_lets_the_next_step_start(self, tmp_path):
        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        fill_step(results, plan, 0)
        seal_step(evidence, plan, 0, results)
        assert acceptance.predecessor_problems(
            evidence, plan, 1, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST) == []

    def test_step_zero_has_no_predecessors(self, tmp_path):
        assert acceptance.predecessor_problems(
            tmp_path, tiny_plan(), 0, results_path=tmp_path / "none.jsonl",
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST) == []

    def test_a_forged_completion_beside_no_attempts_is_caught(self, tmp_path):
        """The old check was 'the completion file exists'. It does."""
        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        fill_step(results, plan, 0)
        forged = acceptance.build_step_completion(
            index=0, plan=plan,
            attempts=[{"attempt_id": "invented", "attempt_digest": "0" * 64,
                       "cells_written": 320,
                       "started_at": "2026-08-22T10:00:00+00:00",
                       "pack_digest": PACK_DIGEST,
                       "dependency_digest": DEPENDENCY_DIGEST}],
            cells_recorded=320, sealed_at="2026-08-22T11:00:00+00:00",
            sealed_without_decoding=False)
        acceptance.completion_path(evidence, 0).write_text(
            json.dumps(forged), encoding="utf-8")
        assert acceptance.completion_path(evidence, 0).is_file()
        problems = acceptance.predecessor_problems(
            evidence, plan, 1, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST)
        assert any("no attempt records" in p for p in problems), problems
        assert any("no such attempt record is here" in p
                   for p in problems), problems

    def test_a_predecessor_with_a_failed_preflight_is_caught(self, tmp_path):
        from src.training.session import write_once_json

        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        broken = attempt_body(plan, 0, preflight=preflight_stub(fail=("pack",)))
        write_once_json(acceptance.attempt_path(evidence, 0, 0), broken)
        fill_step(results, plan, 0, attempt=broken)
        seal_step(evidence, plan, 0, results, attempts=[broken])
        assert any("did not pass" in p for p in
                   acceptance.predecessor_problems(
                       evidence, plan, 1, results_path=results,
                       expected_pack_digest=PACK_DIGEST,
                       expected_dependency_digest=DEPENDENCY_DIGEST))

    def test_every_predecessor_is_replayed_not_only_the_last(self, tmp_path):
        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        for index in range(3):
            fill_step(results, plan, index)
            seal_step(evidence, plan, index, results)
        acceptance.attempt_path(evidence, 0, 0).unlink()
        problems = acceptance.predecessor_problems(
            evidence, plan, 3, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST)
        assert any(p.startswith("step 0 ") for p in problems), problems

    def test_a_predecessor_missing_cells_is_caught(self, tmp_path):
        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        fill_step(results, plan, 0)
        seal_step(evidence, plan, 0, results)
        rewrite(results, acceptance.read_cells(results)[:-1])
        assert any("cells are missing" in p for p in
                   acceptance.predecessor_problems(
                       evidence, plan, 1, results_path=results,
                       expected_pack_digest=PACK_DIGEST,
                       expected_dependency_digest=DEPENDENCY_DIGEST))


# ---------------------------------------------------------------------------
# The sealer and the runner, driven through the entry point
# ---------------------------------------------------------------------------

class TestSealingRefusesInsteadOfRaising:
    """Every refusal here is exit 2 with sentences, and writes no completion."""

    def cli(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "brickagain_core_eval_cli", ROOT / "scripts" / "25_core_eval.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def unsealed(self, tmp_path, *, attempt=None, cells=None):
        """A step with its cells written and no completion yet."""
        from src.training.session import write_once_json

        plan = tiny_plan()
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        results = tmp_path / "results.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        body = attempt or attempts_for(plan)[0]
        write_once_json(acceptance.attempt_path(evidence, 0, 0), body)
        fill_step(results, plan, 0, attempt=body, cells=cells)
        return plan, plan_path, results, evidence

    def run(self, module, plan_path, results, evidence, monkeypatch, step=0):
        """``--run`` with the node checks stubbed; nothing loads a model."""
        monkeypatch.setattr(acceptance, "node_only_problems",
                            lambda mode, probe: [])
        return module.main(["--run", "--step", str(step), "--plan",
                            str(plan_path), "--results", str(results),
                            "--evidence", str(evidence),
                            "--expected-pack-digest", PACK_DIGEST,
                            "--expected-dependency-digest",
                            DEPENDENCY_DIGEST])

    def test_a_healthy_step_seals_without_decoding(self, tmp_path,
                                                   monkeypatch):
        module = self.cli()
        _plan, plan_path, results, evidence = self.unsealed(tmp_path)
        assert self.run(module, plan_path, results, evidence, monkeypatch) == 0
        body = json.loads(acceptance.completion_path(
            evidence, 0).read_text(encoding="utf-8"))
        assert body["sealed_without_decoding"] is True
        assert body["cells_recorded"] == 320
        assert len(acceptance.read_cells(results)) == 320

    @pytest.mark.parametrize("field", acceptance.ATTEMPT_FIELDS)
    def test_a_missing_attempt_field_refuses_without_a_traceback(
            self, tmp_path, monkeypatch, field):
        module = self.cli()
        _plan, plan_path, results, evidence = self.unsealed(tmp_path)
        path = acceptance.attempt_path(evidence, 0, 0)
        body = json.loads(path.read_text(encoding="utf-8"))
        body.pop(field)
        path.unlink()
        path.write_text(json.dumps(body), encoding="utf-8")
        assert self.run(module, plan_path, results, evidence,
                        monkeypatch) == 2, field
        assert not acceptance.completion_path(evidence, 0).exists(), field
        assert len(acceptance.read_cells(results)) == 320, field

    def test_an_unreadable_attempt_file_refuses(self, tmp_path, monkeypatch):
        module = self.cli()
        _plan, plan_path, results, evidence = self.unsealed(tmp_path)
        path = acceptance.attempt_path(evidence, 0, 0)
        path.unlink()
        path.write_text("{ truncated", encoding="utf-8")
        assert self.run(module, plan_path, results, evidence,
                        monkeypatch) == 2
        assert not acceptance.completion_path(evidence, 0).exists()

    def test_a_wrong_carried_digest_refuses_to_seal(self, tmp_path,
                                                    monkeypatch):
        module = self.cli()
        _plan, plan_path, results, evidence = self.unsealed(tmp_path)
        monkeypatch.setattr(acceptance, "node_only_problems",
                            lambda mode, probe: [])
        assert module.main(["--run", "--step", "0", "--plan", str(plan_path),
                            "--results", str(results), "--evidence",
                            str(evidence), "--expected-pack-digest", "9" * 64,
                            "--expected-dependency-digest",
                            DEPENDENCY_DIGEST]) == 2
        assert not acceptance.completion_path(evidence, 0).exists()

    def test_a_failed_preflight_refuses_to_seal(self, tmp_path, monkeypatch):
        module = self.cli()
        plan = tiny_plan()
        broken = attempt_body(plan, 0, preflight=preflight_stub(fail=("vram",)))
        _plan, plan_path, results, evidence = self.unsealed(tmp_path,
                                                            attempt=broken)
        assert self.run(module, plan_path, results, evidence,
                        monkeypatch) == 2
        assert not acceptance.completion_path(evidence, 0).exists()

    @pytest.mark.parametrize("gate", acceptance.REQUIRED_PREFLIGHT_GATES)
    def test_a_deleted_preflight_gate_refuses_to_seal(self, tmp_path,
                                                      monkeypatch, gate):
        module = self.cli()
        plan = tiny_plan()
        broken = attempt_body(plan, 0, preflight=preflight_stub(drop=(gate,)))
        _plan, plan_path, results, evidence = self.unsealed(tmp_path,
                                                            attempt=broken)
        assert self.run(module, plan_path, results, evidence,
                        monkeypatch) == 2, gate
        assert not acceptance.completion_path(evidence, 0).exists(), gate

    def test_a_public_arm_with_an_adapter_refuses_to_seal(self, tmp_path,
                                                          monkeypatch):
        module = self.cli()
        plan = tiny_plan()
        wrong = attempt_body(plan, 0)
        wrong["adapter"] = {"directory_name": "adapter",
                            "files": dict(acceptance.FINAL_ADAPTER_SHA256),
                            "checked_against": "the plan's final_model block"}
        _plan, plan_path, results, evidence = self.unsealed(tmp_path,
                                                            attempt=wrong)
        assert self.run(module, plan_path, results, evidence,
                        monkeypatch) == 2
        assert not acceptance.completion_path(evidence, 0).exists()

    def test_a_cell_naming_an_attempt_with_no_record_refuses_to_seal(
            self, tmp_path, monkeypatch):
        module = self.cli()
        _plan, plan_path, results, evidence = self.unsealed(tmp_path)
        rows = acceptance.read_cells(results)
        rows[0]["attempt_id"] = "invented"
        rewrite(results, rows)
        assert self.run(module, plan_path, results, evidence,
                        monkeypatch) == 2
        assert not acceptance.completion_path(evidence, 0).exists()

    def test_a_cell_with_a_stale_attempt_digest_refuses_to_seal(
            self, tmp_path, monkeypatch):
        module = self.cli()
        _plan, plan_path, results, evidence = self.unsealed(tmp_path)
        rows = acceptance.read_cells(results)
        rows[0]["attempt_digest"] = "0" * 64
        rewrite(results, rows)
        assert self.run(module, plan_path, results, evidence,
                        monkeypatch) == 2
        assert not acceptance.completion_path(evidence, 0).exists()


class TestTheRunnerReplaysItsPredecessors:
    def cli(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "brickagain_core_eval_cli", ROOT / "scripts" / "25_core_eval.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def prepared(self, tmp_path):
        plan = tiny_plan()
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        results = tmp_path / "results.jsonl"
        evidence = tmp_path / "evidence"
        fill_step(results, plan, 0)
        seal_step(evidence, plan, 0, results)
        return plan, plan_path, results, evidence

    def start_step_one(self, module, plan_path, results, evidence,
                       monkeypatch):
        monkeypatch.setattr(acceptance, "node_only_problems",
                            lambda mode, probe: [])
        called = []
        monkeypatch.setattr(acceptance, "build_interface",
                            lambda *a, **kw: called.append("loaded"))
        monkeypatch.setattr(acceptance, "warm_up",
                            lambda *a, **kw: called.append("warmed"))
        code = module.main(["--run", "--step", "1", "--plan", str(plan_path),
                            "--results", str(results), "--evidence",
                            str(evidence), "--expected-pack-digest",
                            PACK_DIGEST, "--expected-dependency-digest",
                            DEPENDENCY_DIGEST])
        return code, called

    def test_a_forged_predecessor_completion_stops_the_next_step(
            self, tmp_path, monkeypatch):
        """The file is there and says the right things. Nothing wrote it."""
        module = self.cli()
        plan, plan_path, results, evidence = self.prepared(tmp_path)
        acceptance.attempt_path(evidence, 0, 0).unlink()
        code, called = self.start_step_one(module, plan_path, results,
                                           evidence, monkeypatch)
        assert code == 2
        assert called == [], "nothing may load or warm before this refusal"
        assert acceptance.completion_path(evidence, 0).is_file()

    def test_a_predecessor_with_a_failed_preflight_stops_the_next_step(
            self, tmp_path, monkeypatch):
        from src.training.session import write_once_json

        module = self.cli()
        plan = tiny_plan()
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        results = tmp_path / "results.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        broken = attempt_body(plan, 0,
                              preflight=preflight_stub(fail=("offline",)))
        write_once_json(acceptance.attempt_path(evidence, 0, 0), broken)
        fill_step(results, plan, 0, attempt=broken)
        seal_step(evidence, plan, 0, results, attempts=[broken])
        code, called = self.start_step_one(module, plan_path, results,
                                           evidence, monkeypatch)
        assert code == 2
        assert called == []

    def test_a_predecessor_whose_cells_went_missing_stops_the_next_step(
            self, tmp_path, monkeypatch):
        module = self.cli()
        _plan, plan_path, results, evidence = self.prepared(tmp_path)
        rewrite(results, acceptance.read_cells(results)[:-1])
        code, called = self.start_step_one(module, plan_path, results,
                                           evidence, monkeypatch)
        assert code == 2
        assert called == []

    def test_a_sound_predecessor_gets_past_the_replay(self, tmp_path,
                                                      monkeypatch):
        """It stops at the preflight instead, which is the next gate."""
        module = self.cli()
        _plan, plan_path, results, evidence = self.prepared(tmp_path)
        monkeypatch.setattr(acceptance, "node_only_problems",
                            lambda mode, probe: [])
        reached = []
        from src.training import gpu_node

        monkeypatch.setattr(gpu_node, "preflight",
                            lambda **kw: reached.append("preflight") or {
                                "passed": False, "failed": ["pack"],
                                "checks": {"pack": {"passed": False,
                                                    "detail": "stand-in"}}})
        code = module.main(["--run", "--step", "1", "--plan", str(plan_path),
                            "--results", str(results), "--evidence",
                            str(evidence), "--expected-pack-digest",
                            PACK_DIGEST, "--expected-dependency-digest",
                            DEPENDENCY_DIGEST])
        assert code == 2
        assert reached == ["preflight"], "the replay must not have refused"

    def test_the_replay_runs_before_anything_is_spent(self):
        """Order is the point: refusing after a load has cost the load."""
        order = call_order_in("mode_run")
        replay = order.index("predecessor_problems")
        for later in ("preflight", "build_interface", "warm_up", "run_case"):
            assert replay < order.index(later), later


# ---------------------------------------------------------------------------
# A damaged results file fails closed, everywhere
# ---------------------------------------------------------------------------

DAMAGE = {
    "truncated": '{"plan_digest": "abc", "case_',
    "unparsable": "not json at all",
    "not_an_object": "[1, 2, 3]",
}


def damage(path, kind):
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(DAMAGE[kind] + "\n")


def call_order_in(function_name: str, path="scripts/25_core_eval.py"):
    """The attribute names called inside one function, in source order.

    Read from the syntax tree rather than by searching the text: the comments
    in ``mode_run`` name the very functions these tests order, and a substring
    search finds the prose before the call.
    """
    import ast

    source = (ROOT / path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == function_name)
    out = []
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        name = (func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None)
        if name:
            out.append((call.lineno, call.col_offset, name))
    return [name for _line, _col, name in sorted(out)]


class TestResultsAreReadFailClosed:
    def test_results_problems_names_each_kind(self, tmp_path):
        plan = tiny_plan()
        for kind in DAMAGE:
            path = tmp_path / f"{kind}.jsonl"
            fill_step(path, plan, 0, cells=acceptance.step_cells(plan, 0)[:2])
            damage(path, kind)
            assert acceptance.results_problems(path), kind

    def test_an_absent_file_is_not_damage(self, tmp_path):
        """Step 0 starts on a tree with no results file at all."""
        assert acceptance.results_problems(tmp_path / "never-written.jsonl") \
            == []

    def test_a_fresh_tree_may_start_step_zero(self, tmp_path, monkeypatch):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "brickagain_core_eval_cli", ROOT / "scripts" / "25_core_eval.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        plan = tiny_plan()
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        reached = []
        from src.training import gpu_node

        monkeypatch.setattr(acceptance, "node_only_problems",
                            lambda mode, probe: [])
        monkeypatch.setattr(gpu_node, "preflight",
                            lambda **kw: reached.append("preflight") or {
                                "passed": False, "failed": ["pack"],
                                "checks": {"pack": {"passed": False,
                                                    "detail": "stand-in"}}})
        code = module.main(["--run", "--step", "0", "--plan", str(plan_path),
                            "--results", str(tmp_path / "results.jsonl"),
                            "--evidence", str(tmp_path / "evidence"),
                            "--expected-pack-digest", PACK_DIGEST,
                            "--expected-dependency-digest",
                            DEPENDENCY_DIGEST])
        assert code == 2
        assert reached == ["preflight"], "the read guard must not have refused"

    def test_a_clean_file_has_no_read_problems(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "r.jsonl"
        fill_step(path, plan, 0, cells=acceptance.step_cells(plan, 0)[:2])
        assert acceptance.results_problems(path) == []

    @pytest.mark.parametrize("kind", sorted(DAMAGE))
    def test_predecessor_problems_does_not_discard_them(self, tmp_path, kind):
        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        fill_step(results, plan, 0)
        seal_step(evidence, plan, 0, results)
        damage(results, kind)
        problems = acceptance.predecessor_problems(
            evidence, plan, 1, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST)
        assert any(p.startswith("results: ") for p in problems), (kind,
                                                                  problems)

    @pytest.mark.parametrize("kind", sorted(DAMAGE))
    def test_the_seal_form_of_the_chain_validator_reports_them(self, tmp_path,
                                                               kind):
        from src.training.session import write_once_json

        plan = tiny_plan()
        results = tmp_path / "r.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        write_once_json(acceptance.attempt_path(evidence, 0, 0),
                        attempts_for(plan)[0])
        fill_step(results, plan, 0)
        damage(results, kind)
        problems = acceptance.step_chain_problems(
            evidence, plan, 0, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST,
            require_completion=False)
        assert problems, kind

    @pytest.mark.parametrize("kind", sorted(DAMAGE))
    def test_evidence_problems_reports_them_once(self, tmp_path, kind):
        plan = tiny_plan()
        results, evidence = run_and_seal(tmp_path, plan, arms=("B",))
        damage(results, kind)
        problems = acceptance.evidence_problems(
            evidence, plan, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST, arms=("B",))
        named = [p for p in problems if p.startswith("results: ")]
        assert len(named) == 1, (kind, named)

    def test_known_keys_still_raises_so_nobody_skips_the_guard(self,
                                                               tmp_path):
        """Silently skipping a bad line would let a run re-append a cell."""
        plan = tiny_plan()
        path = tmp_path / "r.jsonl"
        fill_step(path, plan, 0, cells=acceptance.step_cells(plan, 0)[:2])
        damage(path, "truncated")
        with pytest.raises(ValueError):
            acceptance.known_keys(path)


class TestTheRunnerRefusesADamagedResultsFile:
    def cli(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "brickagain_core_eval_cli", ROOT / "scripts" / "25_core_eval.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def sealed_step_zero(self, tmp_path):
        plan = tiny_plan()
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        results = tmp_path / "results.jsonl"
        evidence = tmp_path / "evidence"
        fill_step(results, plan, 0)
        seal_step(evidence, plan, 0, results)
        return plan, plan_path, results, evidence

    def spend_nothing(self, monkeypatch):
        """Record any attempt to load, warm or preflight; none may happen."""
        from src.training import gpu_node

        spent = []
        monkeypatch.setattr(acceptance, "node_only_problems",
                            lambda mode, probe: [])
        monkeypatch.setattr(gpu_node, "preflight",
                            lambda **kw: spent.append("preflight") or {
                                "passed": True, "failed": [], "checks": {}})
        monkeypatch.setattr(acceptance, "build_interface",
                            lambda *a, **kw: spent.append("loaded"))
        monkeypatch.setattr(acceptance, "warm_up",
                            lambda *a, **kw: spent.append("warmed"))
        monkeypatch.setattr(acceptance, "run_case",
                            lambda *a, **kw: spent.append("decoded"))
        return spent

    @pytest.mark.parametrize("kind", sorted(DAMAGE))
    def test_a_damaged_file_stops_step_one_before_anything_is_spent(
            self, tmp_path, monkeypatch, kind):
        module = self.cli()
        _plan, plan_path, results, evidence = self.sealed_step_zero(tmp_path)
        damage(results, kind)
        before = results.read_text(encoding="utf-8")
        spent = self.spend_nothing(monkeypatch)
        code = module.main(["--run", "--step", "1", "--plan", str(plan_path),
                            "--results", str(results), "--evidence",
                            str(evidence), "--expected-pack-digest",
                            PACK_DIGEST, "--expected-dependency-digest",
                            DEPENDENCY_DIGEST])
        assert code == 2, kind
        assert spent == [], (kind, spent)
        assert results.read_text(encoding="utf-8") == before, kind
        assert not acceptance.completion_path(evidence, 1).exists(), kind

    @pytest.mark.parametrize("kind", sorted(DAMAGE))
    def test_a_damaged_file_stops_a_seal_and_writes_no_completion(
            self, tmp_path, monkeypatch, kind):
        from src.training.session import write_once_json

        module = self.cli()
        plan = tiny_plan()
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        results = tmp_path / "results.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        write_once_json(acceptance.attempt_path(evidence, 0, 0),
                        attempts_for(plan)[0])
        fill_step(results, plan, 0)
        damage(results, kind)
        before = results.read_text(encoding="utf-8")
        spent = self.spend_nothing(monkeypatch)
        code = module.main(["--run", "--step", "0", "--plan", str(plan_path),
                            "--results", str(results), "--evidence",
                            str(evidence), "--expected-pack-digest",
                            PACK_DIGEST, "--expected-dependency-digest",
                            DEPENDENCY_DIGEST])
        assert code == 2, kind
        assert spent == [], (kind, spent)
        assert not acceptance.completion_path(evidence, 0).exists(), kind
        assert results.read_text(encoding="utf-8") == before, kind

    def test_the_guard_runs_before_step_problems_can_raise(self):
        order = call_order_in("mode_run")
        guard = order.index("results_problems")
        for later in ("step_problems", "predecessor_problems", "preflight",
                      "build_interface", "warm_up", "run_case"):
            assert guard < order.index(later), later


# ---------------------------------------------------------------------------
# A row must sit where the frozen schedule puts it
# ---------------------------------------------------------------------------

class TestSchedulePlacementIsCheckedBySelfAndByCellKey:
    def cli(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "brickagain_core_eval_cli", ROOT / "scripts" / "25_core_eval.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def unsealed(self, tmp_path):
        """Step 0 with all its cells written and no completion yet."""
        from src.training.session import write_once_json

        plan = tiny_plan()
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        results = tmp_path / "results.jsonl"
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        write_once_json(acceptance.attempt_path(evidence, 0, 0),
                        attempts_for(plan)[0])
        fill_step(results, plan, 0)
        return plan, plan_path, results, evidence

    def chain(self, evidence, plan, results, index=0):
        return acceptance.step_chain_problems(
            evidence, plan, index, results_path=results,
            expected_pack_digest=PACK_DIGEST,
            expected_dependency_digest=DEPENDENCY_DIGEST,
            require_completion=False)

    def seal(self, module, plan_path, results, evidence, monkeypatch, step=0):
        from src.training import gpu_node

        spent = []
        monkeypatch.setattr(acceptance, "node_only_problems",
                            lambda mode, probe: [])
        monkeypatch.setattr(gpu_node, "preflight",
                            lambda **kw: spent.append("preflight") or {
                                "passed": True, "failed": [], "checks": {}})
        monkeypatch.setattr(acceptance, "build_interface",
                            lambda *a, **kw: spent.append("loaded"))
        code = module.main(["--run", "--step", str(step), "--plan",
                            str(plan_path), "--results", str(results),
                            "--evidence", str(evidence),
                            "--expected-pack-digest", PACK_DIGEST,
                            "--expected-dependency-digest",
                            DEPENDENCY_DIGEST])
        return code, spent

    def test_a_healthy_step_is_placed_correctly(self, tmp_path):
        plan, _pp, results, evidence = self.unsealed(tmp_path)
        assert self.chain(evidence, plan, results) == []

    def test_a_complete_step_with_the_wrong_group_is_not_sealed(
            self, tmp_path, monkeypatch):
        """Every cell is there; one of them says it ran in the other group."""
        module = self.cli()
        plan, plan_path, results, evidence = self.unsealed(tmp_path)
        rows = acceptance.read_cells(results)
        rows[0]["group"] = "odd"
        rewrite(results, rows)
        problems = self.chain(evidence, plan, results)
        assert any("group 'odd'" in p and "frozen schedule" in p
                   for p in problems), problems
        code, spent = self.seal(module, plan_path, results, evidence,
                                monkeypatch)
        assert code == 2
        assert not acceptance.completion_path(evidence, 0).exists()
        assert spent == []

    def test_every_cell_of_the_step_may_carry_the_wrong_group(self, tmp_path,
                                                              monkeypatch):
        module = self.cli()
        plan, plan_path, results, evidence = self.unsealed(tmp_path)
        rows = acceptance.read_cells(results)
        for row in rows:
            row["group"] = "odd"
        rewrite(results, rows)
        code, _spent = self.seal(module, plan_path, results, evidence,
                                 monkeypatch)
        assert code == 2
        assert not acceptance.completion_path(evidence, 0).exists()

    def test_a_row_claiming_this_step_with_the_wrong_arm_is_refused(
            self, tmp_path, monkeypatch):
        """Caught by the claim, not by the cell key -- a key carries the arm."""
        module = self.cli()
        plan, plan_path, results, evidence = self.unsealed(tmp_path)
        stray = cell_for(plan, plan["cases"][0], "D", 0)
        stray["step_index"] = 0
        stray["group"] = "even"
        with Path(results).open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(stray) + "\n")
        problems = self.chain(evidence, plan, results)
        assert any("arm 'D'" in p and "frozen schedule" in p
                   for p in problems), problems
        code, _spent = self.seal(module, plan_path, results, evidence,
                                 monkeypatch)
        assert code == 2
        assert not acceptance.completion_path(evidence, 0).exists()

    def test_a_cell_of_this_step_filed_under_another_is_caught_here(
            self, tmp_path):
        """Selecting only rows that claim the step would miss this one."""
        plan, _pp, results, evidence = self.unsealed(tmp_path)
        rows = acceptance.read_cells(results)
        rows[0]["step_index"] = 6
        rows[0]["group"] = "odd"
        rewrite(results, rows)
        problems = self.chain(evidence, plan, results)
        assert any("step_index 6" in p and "group 'odd'" in p
                   for p in problems), problems
        assert any("1 of 320 cells are missing" in p
                   for p in problems), problems

    def test_a_row_claiming_this_step_that_is_not_one_of_its_cells(
            self, tmp_path):
        plan, _pp, results, evidence = self.unsealed(tmp_path)
        stray = cell_for(plan, plan["cases"][0], "B", 0)
        stray["step_index"] = 0
        stray["group"] = "even"
        stray["seed"] = 9
        with Path(results).open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(stray) + "\n")
        assert any("claims this step and is not one of its cells" in p
                   for p in self.chain(evidence, plan, results))

    @pytest.mark.parametrize("index", range(acceptance.N_STEPS))
    def test_the_expected_placement_is_the_frozen_one_for_every_step(
            self, index):
        plan = tiny_plan()
        group, name = acceptance.STEP_ORDER[index]
        for case in acceptance.step_cases(plan, index):
            assert placement(plan)[(case["case_id"], name)] == (index, group)

    def test_a_wrongly_placed_predecessor_stops_the_next_step(self, tmp_path,
                                                              monkeypatch):
        module = self.cli()
        plan = tiny_plan()
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        results = tmp_path / "results.jsonl"
        evidence = tmp_path / "evidence"
        fill_step(results, plan, 0)
        seal_step(evidence, plan, 0, results)
        rows = acceptance.read_cells(results)
        rows[0]["group"] = "odd"
        rewrite(results, rows)
        code, spent = self.seal(module, plan_path, results, evidence,
                                monkeypatch, step=1)
        assert code == 2
        assert spent == [], "nothing may load before a placement refusal"


class TestPlanReissue:
    """Repairing a scorer must not require opening the test split again.

    The plan pins the scorer it will be measured against, so fixing any
    scorer module invalidates a materialised plan. That is the contract
    working, not failing -- but the blunt remedy, materialising again, opens
    the test split, and the test split is opened once in the life of this
    project. ``--reissue-plan`` is the narrow alternative: the cases already
    on disk, the scorer manifest replaced, everything else proved identical.
    """

    def cli(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "brickagain_core_eval_cli", ROOT / "scripts" / "25_core_eval.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def mac_cli(self, monkeypatch):
        module = self.cli()
        monkeypatch.setattr(module, "_reissue_mac_guard", lambda: [])
        return module

    def aged(self, tmp_path):
        """A plan on disk whose scorer manifest is not this machine's."""
        plan = tiny_plan()
        plan["scorer_source_manifest"] = {
            rel: "f" * 64 for rel in acceptance.SCORER_SOURCES}
        plan["scorer_source_manifest_digest"] = acceptance.digest_obj(
            plan["scorer_source_manifest"])
        plan["plan_digest"] = acceptance.plan_digest(plan)
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return plan, path, sha

    def argv(self, path, out, plan, sha, **over):
        args = {"--plan": str(path), "--out": str(out),
                "--expected-plan-digest": plan["plan_digest"],
                "--expected-plan-sha256": sha}
        args.update(over)
        argv = ["--reissue-plan"]
        for k, v in args.items():
            if v is not None:
                argv += [k, v]
        return argv

    def test_it_swaps_the_scorer_manifest_and_nothing_else(self, tmp_path,
                                                           monkeypatch):
        module = self.mac_cli(monkeypatch)
        old, path, sha = self.aged(tmp_path)
        out = tmp_path / "candidate.json"
        assert module.main(self.argv(path, out, old, sha)) == 0

        new = json.loads(out.read_text(encoding="utf-8"))
        assert new["scorer_source_manifest"] == acceptance.scorer_manifest(ROOT)
        assert new["scorer_source_manifest_digest"] == \
            acceptance.scorer_manifest_digest(ROOT)
        assert new["plan_digest"] == acceptance.plan_digest(new)
        assert new["plan_digest"] != old["plan_digest"]

        for field in set(old) - set(module.REISSUABLE):
            assert new[field] == old[field], field
        assert set(new) == set(old)

    def test_every_frozen_field_survives_by_name(self, tmp_path, monkeypatch):
        """Named one by one, so a field silently dropped is a failure here."""
        module = self.mac_cli(monkeypatch)
        old, path, sha = self.aged(tmp_path)
        out = tmp_path / "candidate.json"
        assert module.main(self.argv(path, out, old, sha)) == 0
        new = json.loads(out.read_text(encoding="utf-8"))

        for field in ("source", "cases", "arms", "settings", "settings_digest",
                      "final_model", "schedule", "contract_digest",
                      "contract_version", "schema_version", "kind", "carries",
                      "note"):
            assert new[field] == old[field], field
        assert new["settings"]["seeds"] == old["settings"]["seeds"]
        assert new["settings"]["k"] == old["settings"]["k"]
        assert [c["case_id"] for c in new["cases"]] == \
            [c["case_id"] for c in old["cases"]]
        assert [c["caption"] for c in new["cases"]] == \
            [c["caption"] for c in old["cases"]]
        assert [c["inventory"] for c in new["cases"]] == \
            [c["inventory"] for c in old["cases"]]
        assert [c["prompt_sha256"] for c in new["cases"]] == \
            [c["prompt_sha256"] for c in old["cases"]]

    def test_it_never_reads_the_test_split(self, tmp_path, monkeypatch):
        """Not "it is not supposed to" but "the read would have been seen"."""
        module = self.mac_cli(monkeypatch)
        old, path, sha = self.aged(tmp_path)
        out = tmp_path / "candidate.json"

        opened = []
        real = Path.read_text

        def watched(self, *a, **kw):
            if acceptance.TEST_FILE in str(self):
                opened.append(str(self))
            return real(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", watched)
        real_bytes = Path.read_bytes

        def watched_bytes(self, *a, **kw):
            if acceptance.TEST_FILE in str(self):
                opened.append(str(self))
            return real_bytes(self, *a, **kw)

        monkeypatch.setattr(Path, "read_bytes", watched_bytes)
        assert module.main(self.argv(path, out, old, sha)) == 0
        assert opened == []

    def test_a_wrong_carried_digest_writes_nothing(self, tmp_path, monkeypatch):
        module = self.mac_cli(monkeypatch)
        old, path, sha = self.aged(tmp_path)
        out = tmp_path / "candidate.json"
        assert module.main(self.argv(path, out, old, sha,
                                     **{"--expected-plan-digest": "0" * 64})) == 2
        assert not out.exists()

    def test_a_wrong_carried_sha_writes_nothing(self, tmp_path, monkeypatch):
        module = self.mac_cli(monkeypatch)
        old, path, sha = self.aged(tmp_path)
        out = tmp_path / "candidate.json"
        assert module.main(self.argv(path, out, old, "0" * 64)) == 2
        assert not out.exists()

    def test_both_carried_values_are_required(self, tmp_path, monkeypatch):
        module = self.mac_cli(monkeypatch)
        old, path, sha = self.aged(tmp_path)
        out = tmp_path / "candidate.json"
        for drop in ("--expected-plan-digest", "--expected-plan-sha256"):
            assert module.main(self.argv(path, out, old, sha,
                                         **{drop: None})) == 2
            assert not out.exists()

    def test_it_refuses_to_overwrite_its_output(self, tmp_path, monkeypatch):
        module = self.mac_cli(monkeypatch)
        old, path, sha = self.aged(tmp_path)
        out = tmp_path / "candidate.json"
        out.write_text("{}", encoding="utf-8")
        assert module.main(self.argv(path, out, old, sha)) == 2
        assert out.read_text(encoding="utf-8") == "{}"

    def test_it_refuses_to_reissue_onto_the_plan_in_place(self, tmp_path,
                                                          monkeypatch):
        module = self.mac_cli(monkeypatch)
        old, path, sha = self.aged(tmp_path)
        before = path.read_bytes()
        assert module.main(self.argv(path, path, old, sha)) == 2
        assert path.read_bytes() == before

    def test_a_tampered_case_is_refused_by_the_carried_sha(self, tmp_path,
                                                           monkeypatch):
        """Editing a case changes the bytes, and the bytes are checked."""
        module = self.mac_cli(monkeypatch)
        old, path, sha = self.aged(tmp_path)
        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["cases"][0]["caption"] = "something else"
        path.write_text(json.dumps(tampered), encoding="utf-8")
        out = tmp_path / "candidate.json"
        assert module.main(self.argv(path, out, old, sha)) == 2
        assert not out.exists()

    def test_a_plan_lying_about_its_own_digest_is_refused(self, tmp_path,
                                                          monkeypatch):
        module = self.mac_cli(monkeypatch)
        plan = tiny_plan()
        plan["plan_digest"] = "9" * 64
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        out = tmp_path / "candidate.json"
        assert module.main(self.argv(path, out, plan, sha)) == 2
        assert not out.exists()

    def test_the_mode_is_mac_only_and_says_so(self, tmp_path):
        module = self.cli()
        import platform

        if platform.system() != "Darwin":            # pragma: no cover
            pytest.skip("not the development machine")
        assert module._reissue_mac_guard() == []

    def test_the_guard_refuses_off_the_mac(self, tmp_path, monkeypatch):
        module = self.cli()
        old, path, sha = self.aged(tmp_path)
        out = tmp_path / "candidate.json"
        monkeypatch.setattr(
            module, "_reissue_mac_guard",
            lambda: ["--reissue-plan runs on the Mac only, and this is 'Linux'"])
        assert module.main(self.argv(path, out, old, sha)) == 2
        assert not out.exists()

    def test_it_is_one_mode_among_the_others(self):
        module = self.cli()
        assert module.main(["--reissue-plan", "--materialize"]) == 2


class TestRunAllDrivesTheScheduleOnTheNode:
    """The continuous runner, and every way it must refuse to continue.

    It exists because the driver used to live on the Mac holding an ssh
    connection open for hours, and a laptop lid is not a scheduling
    primitive: one batch stalled for seven hours when the watching process
    was culled, and stopped again when a step's ssh died. Moving the loop
    into the pack puts it under the same manifest as the runner it drives.

    What matters in these tests is that it adds no judgement of its own: a
    step counts as finished when the pack's own
    ``step_chain_problems(require_completion=True)`` says so, and when it
    does not, nothing further starts.
    """

    def cli(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "brickagain_core_eval_cli", ROOT / "scripts" / "25_core_eval.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def node_cli(self, monkeypatch):
        module = self.cli()
        monkeypatch.setattr(acceptance, "node_only_problems",
                            lambda mode, probe: [])
        return module

    def args(self, tmp_path, plan_path, **over):
        out = {"plan": str(plan_path),
               "results": str(tmp_path / "results.jsonl"),
               "evidence": str(tmp_path / "evidence"),
               "pack_dir": str(tmp_path / "pack"),
               "log_dir": str(tmp_path / "logs"),
               "adapter_dir": str(tmp_path / "adapter"),
               "expected_pack_digest": PACK_DIGEST,
               "expected_dependency_digest": DEPENDENCY_DIGEST}
        out.update(over)
        return types.SimpleNamespace(**out)

    def planned(self, tmp_path):
        plan = tiny_plan()
        path = tmp_path / "plan.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        (tmp_path / "pack").mkdir(exist_ok=True)
        return plan, path

    def spy(self, module, monkeypatch, *, codes=None, on_call=None):
        """Record every step subprocess, and answer with a scripted code."""
        seen = []

        def fake_run(argv, **kw):
            seen.append(argv)
            if on_call is not None:
                on_call(len(seen) - 1, argv)
            code = 0 if codes is None else codes[len(seen) - 1]
            return types.SimpleNamespace(returncode=code)

        monkeypatch.setattr(module.subprocess, "run", fake_run)
        return seen

    def step_of(self, argv):
        return int(argv[argv.index("--step") + 1])

    # -- ordering ---------------------------------------------------------

    def test_it_walks_the_frozen_order_once_each(self, tmp_path, monkeypatch):
        module = self.node_cli(monkeypatch)
        plan, path = self.planned(tmp_path)
        args = self.args(tmp_path, path)

        def seal(_i, argv):
            index = self.step_of(argv)
            acceptance.completion_path(Path(args.evidence), index).parent.mkdir(
                parents=True, exist_ok=True)
            acceptance.completion_path(Path(args.evidence), index).write_text(
                "{}", encoding="utf-8")

        seen = self.spy(module, monkeypatch, on_call=seal)
        monkeypatch.setattr(acceptance, "step_chain_problems",
                            lambda *a, **k: [])
        assert module.mode_run_all(args) == 0
        assert [self.step_of(a) for a in seen] == list(range(acceptance.N_STEPS))

    # -- fail-fast --------------------------------------------------------

    def test_a_non_zero_step_stops_the_schedule(self, tmp_path, monkeypatch):
        module = self.node_cli(monkeypatch)
        _plan, path = self.planned(tmp_path)
        args = self.args(tmp_path, path)
        codes = [0] * acceptance.N_STEPS
        codes[0] = 2
        seen = self.spy(module, monkeypatch, codes=codes)
        assert module.mode_run_all(args) == 2
        assert len(seen) == 1, "step 1 must not start after step 0 failed"

    def test_exit_zero_without_a_completion_stops_the_schedule(
            self, tmp_path, monkeypatch):
        """Success it cannot show is not success."""
        module = self.node_cli(monkeypatch)
        _plan, path = self.planned(tmp_path)
        args = self.args(tmp_path, path)
        seen = self.spy(module, monkeypatch)
        assert module.mode_run_all(args) == 2
        assert len(seen) == 1

    def test_a_chain_that_does_not_verify_stops_the_schedule(
            self, tmp_path, monkeypatch):
        module = self.node_cli(monkeypatch)
        _plan, path = self.planned(tmp_path)
        args = self.args(tmp_path, path)

        def seal(_i, argv):
            p = acceptance.completion_path(Path(args.evidence),
                                           self.step_of(argv))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}", encoding="utf-8")

        seen = self.spy(module, monkeypatch, on_call=seal)
        monkeypatch.setattr(acceptance, "step_chain_problems",
                            lambda *a, **k: ["the chain does not hold"])
        assert module.mode_run_all(args) == 2
        assert len(seen) == 1

    def test_the_completion_check_is_the_packs_own_validator(
            self, tmp_path, monkeypatch):
        """Called with require_completion=True, not a local reimplementation."""
        module = self.node_cli(monkeypatch)
        _plan, path = self.planned(tmp_path)
        args = self.args(tmp_path, path)

        def seal(_i, argv):
            p = acceptance.completion_path(Path(args.evidence),
                                           self.step_of(argv))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}", encoding="utf-8")

        self.spy(module, monkeypatch, on_call=seal)
        calls = []
        monkeypatch.setattr(acceptance, "step_chain_problems",
                            lambda *a, **k: calls.append(k) or [])
        assert module.mode_run_all(args) == 0
        assert len(calls) == acceptance.N_STEPS
        assert all(c["require_completion"] is True for c in calls)
        assert all(c["expected_pack_digest"] == PACK_DIGEST for c in calls)

    # -- skip / resume / seal --------------------------------------------

    def test_a_sealed_step_is_skipped_and_not_rerun(self, tmp_path,
                                                    monkeypatch):
        module = self.node_cli(monkeypatch)
        _plan, path = self.planned(tmp_path)
        args = self.args(tmp_path, path)

        def seal(_i, argv):
            p = acceptance.completion_path(Path(args.evidence),
                                           self.step_of(argv))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}", encoding="utf-8")

        evidence = Path(args.evidence)
        evidence.mkdir(parents=True, exist_ok=True)
        acceptance.completion_path(evidence, 0).write_text("{}",
                                                           encoding="utf-8")
        seen = self.spy(module, monkeypatch, on_call=seal)
        monkeypatch.setattr(acceptance, "step_chain_problems",
                            lambda *a, **k: [])
        assert module.mode_run_all(args) == 0
        assert 0 not in [self.step_of(a) for a in seen]
        assert [self.step_of(a) for a in seen] == list(range(1,
                                                             acceptance.N_STEPS))

    def test_a_partial_step_resumes_and_others_do_not(self, tmp_path,
                                                      monkeypatch):
        module = self.node_cli(monkeypatch)
        plan, path = self.planned(tmp_path)
        args = self.args(tmp_path, path)
        monkeypatch.setattr(
            acceptance, "step_state",
            lambda *a, **k: acceptance.STEP_PARTIAL if a[3] == 0
            else acceptance.STEP_UNSTARTED)

        def seal(_i, argv):
            p = acceptance.completion_path(Path(args.evidence),
                                           self.step_of(argv))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}", encoding="utf-8")

        seen = self.spy(module, monkeypatch, on_call=seal)
        monkeypatch.setattr(acceptance, "step_chain_problems",
                            lambda *a, **k: [])
        assert module.mode_run_all(args) == 0
        assert "--resume" in seen[0]
        assert all("--resume" not in a for a in seen[1:])

    def test_cells_complete_unsealed_is_not_resumed(self, tmp_path,
                                                    monkeypatch):
        """It needs a closing record, not 320 more decodes."""
        module = self.node_cli(monkeypatch)
        _plan, path = self.planned(tmp_path)
        args = self.args(tmp_path, path)
        monkeypatch.setattr(
            acceptance, "step_state",
            lambda *a, **k: acceptance.STEP_UNSEALED if a[3] == 0
            else acceptance.STEP_UNSTARTED)

        def seal(_i, argv):
            p = acceptance.completion_path(Path(args.evidence),
                                           self.step_of(argv))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}", encoding="utf-8")

        seen = self.spy(module, monkeypatch, on_call=seal)
        monkeypatch.setattr(acceptance, "step_chain_problems",
                            lambda *a, **k: [])
        assert module.mode_run_all(args) == 0
        assert "--resume" not in seen[0]

    # -- adapter routing --------------------------------------------------

    def test_only_the_fine_tuned_arms_carry_the_adapter(self, tmp_path,
                                                        monkeypatch):
        """A public arm that quietly loaded final_H2 makes B - C zero."""
        module = self.node_cli(monkeypatch)
        _plan, path = self.planned(tmp_path)
        args = self.args(tmp_path, path)

        def seal(_i, argv):
            p = acceptance.completion_path(Path(args.evidence),
                                           self.step_of(argv))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}", encoding="utf-8")

        seen = self.spy(module, monkeypatch, on_call=seal)
        monkeypatch.setattr(acceptance, "step_chain_problems",
                            lambda *a, **k: [])
        assert module.mode_run_all(args) == 0

        for argv in seen:
            index = self.step_of(argv)
            _group, arm = acceptance.step(index)
            carries = "--adapter-dir" in argv
            if arm in ("C", "E"):
                assert carries, f"step {index} is {arm} and must load final_H2"
                assert argv[argv.index("--adapter-dir") + 1] == args.adapter_dir
            else:
                assert not carries, f"step {index} is {arm} and must not"

    def test_the_public_steps_are_exactly_0_1_6_7(self, tmp_path,
                                                  monkeypatch):
        module = self.node_cli(monkeypatch)
        _plan, path = self.planned(tmp_path)
        args = self.args(tmp_path, path)

        def seal(_i, argv):
            p = acceptance.completion_path(Path(args.evidence),
                                           self.step_of(argv))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}", encoding="utf-8")

        seen = self.spy(module, monkeypatch, on_call=seal)
        monkeypatch.setattr(acceptance, "step_chain_problems",
                            lambda *a, **k: [])
        module.mode_run_all(args)
        public = [self.step_of(a) for a in seen if "--adapter-dir" not in a]
        assert public == [0, 1, 6, 7]

    def test_it_refuses_without_an_adapter_directory(self, tmp_path,
                                                     monkeypatch):
        module = self.node_cli(monkeypatch)
        _plan, path = self.planned(tmp_path)
        seen = self.spy(module, monkeypatch)
        assert module.mode_run_all(
            self.args(tmp_path, path, adapter_dir=None)) == 2
        assert seen == []

    # -- log placement ----------------------------------------------------

    def test_a_log_directory_inside_the_pack_is_refused(self, tmp_path,
                                                        monkeypatch):
        """This already happened, and the preflight was right to stop it."""
        module = self.node_cli(monkeypatch)
        _plan, path = self.planned(tmp_path)
        pack = tmp_path / "pack"
        seen = self.spy(module, monkeypatch)
        for inside in (pack, pack / "logs", pack / "a" / "b"):
            assert module.mode_run_all(
                self.args(tmp_path, path, log_dir=str(inside))) == 2
        assert seen == []

    def test_a_log_directory_is_required(self, tmp_path, monkeypatch):
        module = self.node_cli(monkeypatch)
        _plan, path = self.planned(tmp_path)
        seen = self.spy(module, monkeypatch)
        assert module.mode_run_all(
            self.args(tmp_path, path, log_dir=None)) == 2
        assert seen == []

    def test_a_log_directory_beside_the_pack_is_accepted(self, tmp_path,
                                                         monkeypatch):
        module = self.node_cli(monkeypatch)
        _plan, path = self.planned(tmp_path)
        assert module._log_dir_problems(str(tmp_path / "logs"),
                                        str(tmp_path / "pack")) == []

    # -- carried digests and platform -------------------------------------

    def test_both_carried_digests_are_required(self, tmp_path, monkeypatch):
        module = self.node_cli(monkeypatch)
        _plan, path = self.planned(tmp_path)
        seen = self.spy(module, monkeypatch)
        for drop in ("expected_pack_digest", "expected_dependency_digest"):
            assert module.mode_run_all(
                self.args(tmp_path, path, **{drop: None})) == 2
        assert seen == []

    def test_it_refuses_off_the_node(self, tmp_path, monkeypatch):
        module = self.cli()
        _plan, path = self.planned(tmp_path)
        monkeypatch.setattr(acceptance, "node_only_problems",
                            lambda mode, probe: ["this is not the node"])
        seen = self.spy(module, monkeypatch)
        assert module.mode_run_all(self.args(tmp_path, path)) == 2
        assert seen == []

    def test_it_uses_no_helper_outside_the_pack(self):
        """The failure this mode replaces was a helper nobody had reviewed."""
        source = (ROOT / "scripts" / "25_core_eval.py").read_text(
            encoding="utf-8")
        assert "check_step" not in source
        assert "scratchpad" not in source
