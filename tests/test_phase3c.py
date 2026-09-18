"""Phase 3C: what the pipeline refuses, and what it re-derives.

No model runs here and none needs to. Every check in this suite is about the
machinery *around* the decode -- whether a partial run can be sealed, whether
a tampered sample survives a receipt, whether a fabricated score can pass as
a derived one, whether the case list can be widened after the fact -- and all
of those are answerable from synthetic samples that carry the same fields a
real one does.

The samples are built by :func:`synth_run`, which writes a complete,
internally consistent run directory for the frozen plan: six steps, three
arms, 160 cases, four seeds, 1,920 cells. It is deliberately *not* a
plausible model output. The brick text is a deterministic function of the
case's own inventory, so the scorer produces stable numbers and a test can
say what changing one byte does, which is the only thing being asserted.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pytest

from src.eval import phase3c, phase3c_report
from src.eval.acceptance import PlanRefused, canonical_inventory, digest_obj

ROOT = Path(__file__).resolve().parents[1]

PLAN_PATH = ROOT / phase3c.PLAN_PATH
CARRIED = "carried"

#: A stand-in, not a reading of any clock. The public-snapshot audit refuses
#: a second-precision timestamp in a published file, correctly: one identifies
#: when a particular machine did a particular thing. Nothing here depends on
#: the value, only on it being the same on every run.
STAMP = "2026-01-01T00:00Z"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

#: The public tree carries this suite but not ``gpu_plans/`` or
#: ``data/processed/``, both of which are withheld. A test that needs either
#: skips with this prefix, which is the project's declared vocabulary for
#: "skipped for want of unpublished evidence" -- an undeclared skip in the
#: public tree is a defect, and the snapshot suite checks both directions.
ARTIFACT_ONLY = "artifact-only:"


#: Where the ``plan`` fixture's generation lives, so a test that needs the
#: *archive* -- ``archive_problems`` re-digests the membership and the audit
#: against the plan -- looks in the same place the plan came from, whether
#: that is the real freeze or a directory materialised for this run. Set by
#: the fixture below; read only from a test that already depends on it.
_ARCHIVE_FOR_TESTS: Path | None = None


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    """The frozen plan if it is here, otherwise one materialised for the test.

    Deliberately not "skip unless frozen". These tests check the machinery,
    not the archive, and making them depend on a written generation had two
    costs: they could not run before the freeze -- which is exactly when the
    machinery needs checking -- and a change to this module would have to be
    frozen before it could be tested, which is the wrong order.

    The fallback still needs the test split, so in the public tree, where the
    split is withheld, it skips as before.
    """
    global _ARCHIVE_FOR_TESTS
    if PLAN_PATH.is_file():
        _ARCHIVE_FOR_TESTS = PLAN_PATH.parent
        return phase3c.read_plan(PLAN_PATH, root=ROOT)
    if not (ROOT / phase3c.TEST_FILE).is_file():
        pytest.skip(f"{ARTIFACT_ONLY} neither {phase3c.PLAN_PATH} nor "
                    f"{phase3c.TEST_FILE} is published")
    body, membership, audit = phase3c.materialize_plan(ROOT)
    # ``<root>/frozen/<generation>/`` in shape, because the supersession
    # record is written beside the archive rather than inside it.
    out = (tmp_path_factory.mktemp("phase3c_archive")
           / phase3c.GENERATION)
    out.mkdir()
    phase3c.write_plan_set(out, body, membership, audit, root=ROOT)
    materialised = phase3c.read_plan(out / Path(phase3c.PLAN_PATH).name,
                                     root=ROOT)
    # A complete archive, because that is what the report chain requires:
    # plan set, contract snapshot, supersession record, grant, and the pack
    # evidence the grant binds.
    _write_grant_set(out, materialised)
    _ARCHIVE_FOR_TESTS = out
    return materialised


def _write_grant_set(archive: Path, plan: dict) -> None:
    """The grant and the evidence it binds, written into a test archive."""
    from src.training.session import write_once_json

    evidence = synth_pack_evidence()
    write_once_json(archive / Path(phase3c.PACK_EVIDENCE_PATH).name,
                    evidence)
    write_once_json(archive / Path(phase3c.AUTHORIZATION_PATH).name,
                    synth_authorization(plan))


def _needs_the_test_split():
    if not (ROOT / phase3c.TEST_FILE).is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {phase3c.TEST_FILE} is not published")


def synth_text(case: dict, name: str, seed: int) -> str:
    """A generation that spends the case's own stock, deterministically.

    Arm A is allowed to overspend -- one extra brick of the first stocked
    part beyond its quantity -- so ``inventory_valid`` separates the arms the
    way the real contrast is expected to, and a test that asserts "B and C
    are 1.0 by construction" is asserting against data where A is not.
    """
    inventory = canonical_inventory(case["inventory"])
    stocked = [p for p, n in sorted(inventory.items()) if n > 0]
    if not stocked:
        return ""
    part = stocked[0]
    take = min(inventory[part], 1 + (seed % 2))
    if name == "A":
        take = inventory[part] + 1
    h, w = (int(x) for x in part.split("x"))
    return "".join(f"{h}x{w} ({i * w},0,{i})\n" for i in range(take))


def real_gate(name: str, inventory: dict):
    """The gate arm ``name`` actually decodes through, really constructed.

    Not a stand-in. ``phase3c.observed_gate`` checks the exact class of the
    object a decode returned, so a row built for arm B has to be built with
    the class arm B's decode builds -- which is the whole point of the check
    that gen08 was missing.
    """
    from src.constraints.inventory_decode import InventoryGate
    from src.constraints.placement_decode import InventoryPlacementGate
    from src.generation.brickgpt import MAX_DIM, Slots, WORLD
    from src.inventory.engine import Inventory

    spec = phase3c.arm(name)
    if spec.gate == phase3c.GATE_NONE:
        return None
    # The token ids are never read here: ``gate_ledger`` asks the gate what
    # it accepted and what it has left, not what it would allow next.
    slots = Slots(dims=list(range(100, 100 + MAX_DIM)),
                  posns=list(range(200, 200 + WORLD)), literal_x=1,
                  literal_open=2, literal_comma=3, literal_close=4, eos=5)
    stock = Inventory.from_parts(dict(inventory))
    if spec.gate == phase3c.GATE_INVENTORY:
        gate = InventoryGate(slots, stock)
    else:
        gate = InventoryPlacementGate(slots, stock, enabled=True,
                                      connectivity=spec.connectivity)
    # ``_gated`` snapshots the opening quantities onto the gate before it
    # decodes; ``InventoryPlacementGate.counters`` reads that back.
    gate.opening_inventory = dict(inventory)
    return gate


def synth_row(case: dict, name: str, seed: int, *, plan: dict,
              manifest_digest: str, step_index: int, group: str) -> dict:
    text = synth_text(case, name, seed)
    n_bricks = len([ln for ln in text.splitlines() if ln.strip()])
    from src.generation.prompt import build_prompt

    inventory = canonical_inventory(case["inventory"])
    return {
        "plan_digest": plan["plan_digest"],
        "case_id": case["case_id"],
        "arm": name,
        "seed": seed,
        "step_index": step_index,
        "group": group,
        "manifest_digest": manifest_digest,
        "raw_text": text,
        "n_tokens": n_bricks * phase3c.TOKENS_PER_BRICK + phase3c.EOS_TOKENS,
        "seconds": 1.0 + seed * 0.1,
        "termination": "normal_eos",
        "truncated": False,
        "gate": phase3c.gate_ledger(phase3c.arm(name), inventory,
                                    real_gate(name, inventory)),
        "prompt_sha256": phase3c.sha256_text(
            build_prompt(case["caption"], inventory)),
        "inventory_digest": digest_obj(inventory),
        "contract_digest": plan["contract_digest"],
        "settings_digest": plan["settings_digest"],
        "model": phase3c.model_identity(),
    }


def synth_environment() -> dict:
    """A probe reading that is exactly the machine the contract pins."""
    pinned = phase3c.EXECUTION_ENVIRONMENT
    return phase3c.observed_environment(
        {"os_system": pinned["os_system"], "wsl2": pinned["wsl2"],
         "gpu_name": pinned["gpu_name"],
         "torch_cuda_build": pinned["torch_cuda_build"],
         "offline_env": dict(pinned["offline_env"]),
         "vram_total_gb": 15.9, "system_ram_gb": 31.2},
        versions={k: pinned[k] for k in
                  ("python", "torch", "transformers", "peft", "accelerate")})


def _synth_pack_table() -> dict:
    """A structurally real pack manifest table, over one real file.

    Not a stand-in digest. ``build_pack_evidence`` refuses a table that does
    not recompute to the digest being authorised, so a test archive can only
    hold evidence that genuinely recomputes -- which means the fixtures
    exercise the same arithmetic the freeze does.
    """
    from src.training import pack as pack_module
    from src.training.session import manifest_digest, sha256_file

    rel = "src/eval/phase3c.py"
    files = {rel: {"sha256": sha256_file(ROOT / rel),
                   "bytes": (ROOT / rel).stat().st_size,
                   "snapshot_name": rel}}
    requirements: dict = {}
    table = {
        "schema_version": pack_module.SCHEMA_VERSION,
        "kind": pack_module.KIND,
        "root_relative": True,
        "files": files,
        "files_digest": manifest_digest({"files": files}),
        "data_requirements": requirements,
        "data_digest": pack_module.data_digest(requirements),
    }
    table["pack_digest"] = pack_module.pack_digest(table)
    return table


PACK_TABLE = _synth_pack_table()
PACK_DIGEST = PACK_TABLE["pack_digest"]
DEPENDENCY_DIGEST = "1" * 64


def synth_pack_evidence() -> dict:
    return phase3c.build_pack_evidence(PACK_TABLE, pack_digest=PACK_DIGEST)


PACK_EVIDENCE_DIGEST = synth_pack_evidence()["evidence_digest"]


def synth_authorization(plan: dict) -> dict:
    """The grant a real run would have, built from this tree.

    Its source manifest is the real one, so ``check_sources=True`` holds
    here exactly as it would on the node -- which is what lets a test edit
    one source digest and watch the whole chain refuse.
    """
    return phase3c.build_authorization(
        plan=plan, pack_digest=PACK_DIGEST,
        dependency_digest=DEPENDENCY_DIGEST,
        pack_evidence_digest=PACK_EVIDENCE_DIGEST,
        authorized_at=STAMP, root=ROOT)


def synth_run(out_dir: Path, plan: dict, *, steps=None,
              seal_it: bool = True, grant=None) -> dict:
    """A complete run directory. ``steps`` limits it, to make a partial one."""
    from src.training.session import write_once_json

    out_dir = Path(out_dir)
    grant = grant or synth_authorization(plan)
    manifest = phase3c.build_execution_manifest(
        plan=plan, grant=grant,
        observed=synth_environment(), started_at=STAMP,
        recomputed_sources=phase3c.generation_source_manifest(ROOT))
    write_once_json(out_dir / phase3c.MANIFEST_NAME, manifest)

    cases = phase3c.case_index(plan)
    for index in (range(phase3c.N_STEPS) if steps is None else steps):
        group, name = phase3c.step(index)
        member = out_dir / phase3c.samples_member(index)
        known: set = set()
        for _digest, case_id, arm_name, seed in phase3c.step_cells(plan,
                                                                   index):
            phase3c.append_cell(member, synth_row(
                cases[case_id], arm_name, seed, plan=plan,
                manifest_digest=manifest["manifest_digest"],
                step_index=index, group=group), known=known)
    if not seal_it:
        return {"manifest": manifest, "grant": grant}
    seal = phase3c.build_seal(out_dir, manifest)
    write_once_json(out_dir / phase3c.SEAL_NAME, seal)
    return {"manifest": manifest, "seal": seal, "grant": grant}


@pytest.fixture(scope="module")
def grant(plan):
    """The execution authorization, under the name the code uses for it."""
    return synth_authorization(plan)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, plan, grant):
    out = tmp_path_factory.mktemp("phase3c_run")
    built = synth_run(out, plan, grant=grant)
    return out, built["seal"], built["manifest"]


# ---------------------------------------------------------------------------
# the frozen contract itself
# ---------------------------------------------------------------------------

def test_the_three_arms_differ_only_in_the_gate():
    seen = {(a.model, a.loader, a.prompt_form)
            for a in phase3c.ARMS.values()}
    assert len(seen) == 1, "the arms disagree on something other than the gate"
    assert {a.gate for a in phase3c.ARMS.values()} == {
        phase3c.GATE_NONE, phase3c.GATE_INVENTORY,
        phase3c.GATE_INVENTORY_PLACEMENT}
    assert phase3c.PRIMARY_CONTRAST == ("C", "B")


def test_the_contract_pins_the_node_not_this_machine():
    assert phase3c.EXECUTION_ENVIRONMENT["os_system"] == "Linux"
    assert phase3c.EXECUTION_ENVIRONMENT["device"] == "cuda"
    assert phase3c.environment_problems({"os_system": "Darwin"}), \
        "a Mac must not pass the environment the contract pins"


def test_the_scorer_manifest_covers_this_module():
    assert "src/eval/phase3c.py" in phase3c.SCORER_SOURCES
    assert "src/eval/scoring.py" in phase3c.SCORER_SOURCES


def test_the_selection_never_opens_the_val_split(monkeypatch):
    """val access rejection, by watching the filesystem rather than the text.

    A string scan would fail on this project's own honesty: the audit records
    ``val_file`` so a reader can see *which* file it is claiming not to have
    opened. What matters is whether anything opens it, so every open under
    the audit and the selection is recorded and the val split must not be
    among them.
    """
    import builtins
    import io

    _needs_the_test_split()
    opened: list[str] = []
    real_open, real_path_open = builtins.open, Path.open

    def watched_open(file, *a, **kw):
        opened.append(str(file))
        return real_open(file, *a, **kw)

    def watched_path_open(self, *a, **kw):
        opened.append(str(self))
        return real_path_open(self, *a, **kw)

    monkeypatch.setattr(builtins, "open", watched_open)
    monkeypatch.setattr(Path, "open", watched_path_open)
    monkeypatch.setattr(io, "open", watched_open, raising=False)
    try:
        audit = phase3c.isolation_audit(ROOT)
    finally:
        monkeypatch.undo()

    val = Path(audit["checks"]["val_file"]).name
    offenders = [p for p in opened if val in p]
    assert offenders == [], f"the audit opened {offenders}"
    assert any("instruct_inv_test" in p for p in opened), (
        "the watcher saw no test-split read at all, so it is not watching "
        "what it claims to watch")
    assert audit["checks"]["val_opened"] is False


def test_the_token_correction_is_a_no_op_on_phase_2_terminations():
    """The correction exists for arm C's two new terminations and no others.

    Run against the real scorer rather than a hand-made dict: the claim is
    that Phase 2's four terminations come out of
    ``correct_token_accounting`` exactly as ``score_generation`` left them,
    and only a real scorer output can say that.
    """
    from src.eval.scoring import score_generation

    inventory = {"2x4": 4, "1x1": 4}
    text = "2x4 (0,0,0)\n2x4 (0,0,1)\n"
    for termination in ("normal_eos", "inventory_exhausted", "max_bricks",
                        "max_tokens"):
        n_tokens = 2 * phase3c.TOKENS_PER_BRICK + (
            phase3c.EOS_TOKENS
            if termination in ("normal_eos", "inventory_exhausted") else 0)
        scored = score_generation(text, inventory=inventory,
                                  n_tokens=n_tokens, termination=termination)
        corrected = phase3c.correct_token_accounting(
            scored, n_tokens=n_tokens, termination=termination)
        assert corrected == scored, (
            f"the correction changed a {termination} draw, and it exists "
            "only for space_exhausted and connectivity_unmet")


def test_the_correction_rescues_arm_cs_two_new_terminations():
    """The other half: on those two, Phase 2's rule would have been wrong."""
    from src.eval.scoring import score_generation

    inventory = {"2x4": 4}
    text = "2x4 (0,0,0)\n2x4 (0,0,1)\n"
    for termination in ("space_exhausted", "connectivity_unmet"):
        # These spend a token on EOS, so the draw ends on 10n + 1 while being
        # an unaccepted termination -- the coincidence Phase 2's rule relied
        # on and arm C breaks.
        n_tokens = 2 * phase3c.TOKENS_PER_BRICK + phase3c.EOS_TOKENS
        scored = score_generation(text, inventory=inventory,
                                  n_tokens=n_tokens, termination=termination)
        corrected = phase3c.correct_token_accounting(
            scored, n_tokens=n_tokens, termination=termination)
        assert corrected["checks"]["parse_success"] is True
        assert scored["checks"]["parse_success"] is False, (
            "if the Phase 2 rule already accepted these, the correction "
            "would be unnecessary and this contract would be overstating it")


def test_forbidden_metric_terms_appear_in_no_source_we_publish():
    for name in ("src/eval/phase3c.py", "src/eval/phase3c_report.py",
                 "scripts/59_phase3c.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        # The tuple itself is the one place the words may appear.
        body = text.replace(repr(phase3c.FORBIDDEN_METRIC_TERMS), "")
        for term in phase3c.FORBIDDEN_METRIC_TERMS:
            occurrences = body.lower().count(term.lower())
            allowed = 1 if name.endswith("phase3c.py") else 0
            assert occurrences <= allowed, \
                f"{name} uses {term!r} {occurrences} times"


# ---------------------------------------------------------------------------
# case isolation: contamination, group leakage, val
# ---------------------------------------------------------------------------

def _archive(plan, name: str) -> dict:
    """A sibling of whichever plan the fixture produced."""
    if PLAN_PATH.is_file():
        return json.loads((PLAN_PATH.parent / name).read_text())
    pytest.skip(f"{ARTIFACT_ONLY} {phase3c.ARCHIVE_DIR} is not published")


def test_the_audit_excludes_by_group_and_not_only_by_pair(plan):
    audit = _archive(plan, Path(phase3c.AUDIT_PATH).name)
    counts = audit["exclusions"]["counts"]
    assert counts["by_pair_id"] == 20
    assert counts["by_group"] > 0, (
        "a group-level exclusion that removes nothing is not evidence of "
        "group-level isolation")
    assert audit["checks"]["val_opened"] is False
    assert audit["checks"]["every_eligible_object_is_a_test_object"] is True


def test_no_selected_pair_shares_an_object_with_phase_2(plan):
    audit = _archive(plan, Path(phase3c.AUDIT_PATH).name)
    excluded = set(audit["exclusions"]["by_pair_id"]) | set(
        audit["exclusions"]["by_group"]) | set(
        audit["exclusions"]["by_caption"])
    assert not (set(plan["selection"]["pairs"]) & excluded)


def test_the_selected_pairs_are_distinct_objects(plan):
    membership = _archive(plan, Path(phase3c.MEMBERSHIP_PATH).name)
    assert membership["membership_digest"] == plan["case_membership_digest"]
    assert len(plan["selection"]["pairs"]) == phase3c.N_PAIRS
    assert len(plan["cases"]) == phase3c.N_CASES


def test_the_plan_carries_no_answer(plan):
    from src.eval.acceptance import plan_leak_problems

    assert plan_leak_problems(plan) == []
    for case in plan["cases"]:
        assert set(case) <= set(phase3c.CASE_FIELDS)


def test_a_plan_whose_membership_was_widened_is_refused(plan, tmp_path):
    """wrong case membership: the digest is over the case list."""
    body = json.loads(json.dumps(plan))
    body["cases"] = body["cases"][:-1]
    assert phase3c.plan_problems(body, root=ROOT), \
        "a plan with 159 cases must not validate"


def test_a_case_from_outside_the_membership_is_refused(plan, run_dir):
    """case contamination: a row naming a case the plan never listed."""
    out_dir, _seal, manifest = run_dir
    rows = phase3c.read_rows(out_dir / phase3c.samples_member(0))
    stranger = dict(rows[0], case_id="not-a-case-in-this-plan")
    problems = phase3c.row_problems(stranger, phase3c.case_index(plan),
                                    plan=plan, step_index=0)
    assert any("is not a case in this plan" in p for p in problems)


# ---------------------------------------------------------------------------
# the run: missing arms, partial seals, wrong manifests
# ---------------------------------------------------------------------------

def test_a_run_missing_one_arm_cannot_be_sealed(plan, tmp_path):
    """missing arm + partial seal, in one refusal."""
    out = tmp_path / "partial"
    # Every step except the two that run arm C.
    steps = [i for i in range(phase3c.N_STEPS) if phase3c.step(i)[1] != "C"]
    built = synth_run(out, plan, steps=steps, seal_it=False)
    with pytest.raises(PlanRefused) as exc:
        phase3c.build_seal(out, built["manifest"])
    assert "is not here" in str(exc.value)


def test_a_seal_that_omits_a_member_is_refused(plan, run_dir):
    out_dir, seal, _manifest = run_dir
    partial = json.loads(json.dumps(seal))
    dropped = phase3c.samples_member(phase3c.N_STEPS - 1)
    partial["members"].pop(dropped)
    partial["seal_digest"] = digest_obj(
        {k: v for k, v in partial.items() if k != "seal_digest"})
    problems = phase3c.seal_problems(partial, plan)
    assert any(dropped in p for p in problems)
    assert any("rows" in p for p in problems)


def test_a_manifest_for_another_plan_is_refused(plan, run_dir,
                                                grant):
    """wrong manifest."""
    _out, _seal, manifest = run_dir
    wrong = dict(manifest, plan_digest="0" * 64)
    problems = phase3c.manifest_problems(wrong, plan, grant)
    assert any("plan_digest" in p for p in problems)


def test_a_manifest_claiming_the_wrong_adapter_is_refused(plan, run_dir,
                                                          grant):
    """wrong model / adapter."""
    _out, _seal, manifest = run_dir
    wrong = dict(manifest, adapter_sha256={"adapter_model.safetensors":
                                           "0" * 64})
    wrong["manifest_digest"] = digest_obj(
        {k: v for k, v in wrong.items() if k != "manifest_digest"})
    problems = phase3c.manifest_problems(wrong, plan, grant)
    assert any("adapter" in p for p in problems)


def test_a_manifest_from_the_wrong_machine_is_refused(plan, run_dir,
                                                      grant):
    """wrong environment."""
    _out, _seal, manifest = run_dir
    observed = dict(manifest["environment_observed"], gpu_name="RTX 4090")
    wrong = dict(manifest, environment_observed=observed)
    wrong["manifest_digest"] = digest_obj(
        {k: v for k, v in wrong.items() if k != "manifest_digest"})
    problems = phase3c.manifest_problems(wrong, plan, grant)
    assert any("gpu_name" in p for p in problems)


def test_a_cell_cannot_be_measured_twice(plan, tmp_path):
    """conflicting overwrite, at the level of one cell."""
    cases = phase3c.case_index(plan)
    _digest, case_id, name, seed = phase3c.step_cells(plan, 0)[0]
    path = tmp_path / "samples.jsonl"
    row = synth_row(cases[case_id], name, seed, plan=plan,
                    manifest_digest="0" * 64, step_index=0, group="even")
    known: set = set()
    phase3c.append_cell(path, row, known=known)
    with pytest.raises(PlanRefused):
        phase3c.append_cell(path, dict(row, seconds=99.0), known=known)


def test_a_half_written_last_line_is_refused(tmp_path):
    path = tmp_path / "cut.jsonl"
    path.write_text('{"case_id": "a"}\n{"case_id": "b"', encoding="utf-8")
    with pytest.raises(PlanRefused) as exc:
        phase3c.read_rows(path)
    assert "cut off" in str(exc.value)


# ---------------------------------------------------------------------------
# the seal, the receipt and what tampering does to them
# ---------------------------------------------------------------------------

def test_a_clean_run_verifies(plan, run_dir, grant):
    out_dir, seal, _manifest = run_dir
    receipt = phase3c.build_receipt(out_dir, seal, seal["seal_digest"],
                                    plan=plan, grant=grant,
                                    verified_at=STAMP)
    assert receipt["problems"] == [], receipt["problems"]
    assert receipt["verified"] is True
    assert phase3c.receipt_problems(
        receipt, out_dir, plan=plan, seal=seal,
        grant=grant) == []


def test_a_tampered_sample_fails_the_receipt(plan, run_dir, tmp_path,
                                             grant):
    """tampered samples."""
    out_dir, seal, _manifest = run_dir
    copy = tmp_path / "tampered"
    _copy_tree(out_dir, copy)
    member = copy / phase3c.samples_member(0)
    rows = phase3c.read_rows(member)
    rows[0]["seconds"] = 0.001
    _write_rows(member, rows)
    receipt = phase3c.build_receipt(copy, seal, seal["seal_digest"],
                                    plan=plan, grant=grant,
                                    verified_at=STAMP)
    assert receipt["verified"] is False
    assert any("hashes to" in p for p in receipt["problems"])


def test_a_carried_digest_that_disagrees_fails_the_receipt(plan, run_dir,
                                                           grant):
    """seal / receipt mismatch."""
    out_dir, seal, _manifest = run_dir
    receipt = phase3c.build_receipt(out_dir, seal, "f" * 64, plan=plan,
                                    grant=grant,
                                    verified_at=STAMP)
    assert receipt["verified"] is False
    assert any("carried" in p for p in receipt["problems"])


def test_a_file_the_seal_does_not_name_is_reported(plan, run_dir, tmp_path):
    """scratch / formal path overlap: a stray file in a sealed directory."""
    out_dir, seal, _manifest = run_dir
    copy = tmp_path / "stray"
    _copy_tree(out_dir, copy)
    (copy / "samples" / "scratch.jsonl").write_text("{}\n", encoding="utf-8")
    _found, problems = phase3c.rehash(copy, seal)
    assert any("does not name" in p for p in problems)


def test_a_reseal_over_a_different_run_is_refused(plan, run_dir, tmp_path):
    """conflicting overwrite, at the level of the seal file."""
    from src.training.session import write_once_json

    out_dir, seal, _manifest = run_dir
    with pytest.raises(SystemExit) as exc:
        write_once_json(out_dir / phase3c.SEAL_NAME, dict(seal))
    assert "refusing to overwrite" in str(exc.value)


# ---------------------------------------------------------------------------
# the execution grant, and the counter-examples
# ---------------------------------------------------------------------------
#
# Each of these was accepted before the authorization existed. The manifest
# recorded a pack digest and a dependency digest and nothing compared them to
# anything, and nothing bound the gate source at all -- so a run produced by
# an edited InventoryPlacementGate, against a pack nobody authorised, scored
# clean. Every test here drives a real refusal path rather than asserting a
# message: the receipt must come back unverified, or the scorer must refuse.


def _receipt(out_dir, seal, plan, grant, carried=None):
    return phase3c.build_receipt(
        out_dir, seal, carried or seal["seal_digest"], plan=plan,
        grant=grant, verified_at=STAMP)


def _rebuilt(auth: dict, **changes) -> dict:
    """An authorization edited and re-digested, so it is self-consistent.

    Self-consistency is the point: a tampered grant that failed its own
    digest check would be caught by arithmetic. These are caught because the
    manifest was written against a different grant.
    """
    body = {**{k: v for k, v in auth.items()
               if k != "authorization_digest"}, **changes}
    body["authorization_digest"] = digest_obj(body)
    return body


def test_a_different_pack_digest_fails_the_receipt(plan, run_dir,
                                                   grant):
    """The exact failure the old design allowed: any well-formed digest."""
    out_dir, seal, _manifest = run_dir
    other = _rebuilt(grant, pack_digest="a" * 64)
    receipt = _receipt(out_dir, seal, plan, other)
    assert receipt["verified"] is False
    assert any("pack_digest" in p for p in receipt["problems"]), \
        receipt["problems"]


def test_a_different_dependency_digest_fails_the_receipt(plan, run_dir,
                                                         grant):
    out_dir, seal, _manifest = run_dir
    other = _rebuilt(grant, dependency_digest="b" * 64)
    receipt = _receipt(out_dir, seal, plan, other)
    assert receipt["verified"] is False
    assert any("dependency_digest" in p for p in receipt["problems"])


def test_an_edited_gate_source_fails_the_receipt(plan, run_dir,
                                                 grant):
    """The layer under test, swapped. This had no defence at all before."""
    out_dir, seal, _manifest = run_dir
    gate = "src/constraints/placement_decode.py"
    sources = dict(grant["generation_source_manifest"])
    sources[gate] = "c" * 64
    other = _rebuilt(
        grant,
        gate_sources={**grant["gate_sources"], gate: "c" * 64},
        generation_source_manifest=sources,
        generation_source_manifest_digest=digest_obj(sources))
    receipt = _receipt(out_dir, seal, plan, other)
    assert receipt["verified"] is False
    assert any("generation_source_manifest_digest" in p
               or "gate_sources" in p for p in receipt["problems"])


def test_an_edited_gate_source_on_the_node_stops_the_run(plan,
                                                         grant):
    """The node's own check: the tree it is about to decode with."""
    gate = "src/constraints/inventory_decode.py"
    sources = dict(grant["generation_source_manifest"])
    sources[gate] = "d" * 64
    other = _rebuilt(
        grant,
        gate_sources={**grant["gate_sources"], gate: "d" * 64},
        generation_source_manifest=sources,
        generation_source_manifest_digest=digest_obj(sources))
    problems = phase3c.authorization_problems(other, plan, root=ROOT,
                                              check_sources=True)
    assert any(gate in p for p in problems), problems


def test_a_source_file_that_appeared_is_refused(plan, grant):
    """Neither direction is allowed to drift, not only the digests."""
    sources = dict(grant["generation_source_manifest"])
    sources.pop("src/constraints/placement_decode.py")
    other = _rebuilt(
        grant,
        gate_sources={k: v for k, v in grant["gate_sources"].items()
                      if k != "src/constraints/placement_decode.py"},
        generation_source_manifest=sources,
        generation_source_manifest_digest=digest_obj(sources))
    problems = phase3c.generation_source_problems(sources, ROOT)
    assert any("is on this machine and is not in the authorised" in p
               for p in problems), problems


def test_a_carried_pack_digest_that_is_not_authorised_stops_the_run(
        grant):
    problems = phase3c.carried_digest_problems(
        grant, pack_digest="e" * 64,
        dependency_digest=grant["dependency_digest"])
    assert any("pack_digest carried here" in p for p in problems), problems


def test_a_missing_carried_digest_stops_the_run(grant):
    problems = phase3c.carried_digest_problems(
        grant, pack_digest=None,
        dependency_digest=grant["dependency_digest"])
    assert any("was not given" in p for p in problems)


def test_a_deleted_execution_manifest_fails_the_receipt(plan, run_dir,
                                                        tmp_path,
                                                        grant):
    """The one file saying which pack and gate produced the samples."""
    out_dir, seal, _manifest = run_dir
    copy = tmp_path / "no_manifest"
    _copy_tree(out_dir, copy)
    (copy / phase3c.MANIFEST_NAME).unlink()
    receipt = _receipt(copy, seal, plan, grant)
    assert receipt["verified"] is False
    assert any(phase3c.MANIFEST_NAME in p for p in receipt["problems"])


def test_a_tampered_execution_manifest_fails_the_receipt(plan, run_dir,
                                                         tmp_path,
                                                         grant):
    """Edited and re-digested, so only the seal's record of it disagrees."""
    out_dir, seal, _manifest = run_dir
    copy = tmp_path / "tampered_manifest"
    _copy_tree(out_dir, copy)
    path = copy / phase3c.MANIFEST_NAME
    body = json.loads(path.read_text())
    body["pack_digest"] = "f" * 64
    body["manifest_digest"] = digest_obj(
        {k: v for k, v in body.items() if k != "manifest_digest"})
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    receipt = _receipt(copy, seal, plan, grant)
    assert receipt["verified"] is False
    assert any("hashes to" in p or "pack_digest" in p
               or "different execution manifest" in p
               for p in receipt["problems"]), receipt["problems"]


def test_a_manifest_whose_bytes_moved_at_all_fails_the_receipt(
        plan, run_dir, tmp_path, grant):
    """Even a reformat: the seal records its size and SHA, and now spends them."""
    out_dir, seal, _manifest = run_dir
    copy = tmp_path / "reformatted_manifest"
    _copy_tree(out_dir, copy)
    path = copy / phase3c.MANIFEST_NAME
    body = json.loads(path.read_text())
    path.write_text(json.dumps(body) + "\n", encoding="utf-8")
    receipt = _receipt(copy, seal, plan, grant)
    assert receipt["verified"] is False
    assert any(phase3c.MANIFEST_NAME in p for p in receipt["problems"])


def test_scoring_refuses_an_unauthorised_pack(plan, run_dir, grant):
    """The scorer, not only the receipt, is fail-closed on this."""
    out_dir, seal, _manifest = run_dir
    other = _rebuilt(grant, pack_digest="a" * 64)
    problems = phase3c.score_preconditions(
        out_dir, plan, seal, carried_seal_digest=seal["seal_digest"],
        grant=other, root=ROOT)
    assert any("pack_digest" in p for p in problems), problems


def test_scoring_refuses_a_deleted_execution_manifest(plan, run_dir,
                                                      tmp_path,
                                                      grant):
    out_dir, seal, _manifest = run_dir
    copy = tmp_path / "score_no_manifest"
    _copy_tree(out_dir, copy)
    (copy / phase3c.MANIFEST_NAME).unlink()
    problems = phase3c.score_preconditions(
        copy, plan, seal, carried_seal_digest=seal["seal_digest"],
        grant=grant, root=ROOT)
    assert any(phase3c.MANIFEST_NAME in p for p in problems)


def test_a_stored_receipt_that_no_longer_holds_is_caught(plan, run_dir,
                                                         tmp_path,
                                                         grant):
    """existing-result: a receipt is re-checked, never believed."""
    out_dir, seal, _manifest = run_dir
    copy = tmp_path / "stale_receipt"
    _copy_tree(out_dir, copy)
    receipt = _receipt(copy, seal, plan, grant)
    assert receipt["verified"] is True
    (copy / phase3c.MANIFEST_NAME).unlink()
    problems = phase3c.receipt_problems(receipt, copy, plan=plan, seal=seal,
                                        grant=grant)
    assert problems, "a receipt whose manifest vanished must not still hold"


def test_the_authorization_pins_both_gates(grant):
    for gate in phase3c.GATE_SOURCES:
        assert gate in grant["gate_sources"]
        assert grant["gate_sources"][gate] == \
            grant["generation_source_manifest"][gate]


def test_the_authorization_covers_the_whole_decode_path(grant):
    """Not a hand-kept list: the entry point's real import closure."""
    sources = set(grant["generation_source_manifest"])
    assert phase3c.GENERATION_ENTRY_POINT in sources
    for required in ("src/eval/phase3c.py", "src/generation/brickgpt.py",
                     "src/generation/prompt.py", "src/eval/acceptance.py",
                     "src/training/lora.py"):
        assert required in sources, required


def test_an_authorization_for_another_plan_is_refused(plan, grant):
    other = _rebuilt(grant, plan_digest="0" * 64)
    problems = phase3c.authorization_problems(other, plan, root=ROOT,
                                              check_sources=False)
    assert any("plan_digest" in p for p in problems)


def test_an_authorization_whose_digest_does_not_cover_it_is_refused(
        plan, grant):
    tampered = dict(grant, pack_digest="9" * 64)
    problems = phase3c.authorization_problems(tampered, plan, root=ROOT,
                                              check_sources=False)
    assert any("authorization_digest does not cover" in p for p in problems)


def test_an_authorization_cannot_name_a_non_digest(plan):
    """All three bound digests, each refused in turn."""
    good = {"pack_digest": PACK_DIGEST,
            "dependency_digest": DEPENDENCY_DIGEST,
            "pack_evidence_digest": PACK_EVIDENCE_DIGEST}
    for field in sorted(good):
        for bad in ("", "not-a-digest", "0" * 63, "g" * 64):
            with pytest.raises(PlanRefused):
                phase3c.build_authorization(
                    plan=plan, **{**good, field: bad},
                    authorized_at=STAMP, root=ROOT)


def test_a_grant_that_names_other_pack_evidence_is_refused(plan, tmp_path):
    """The grant binds the evidence; an archive holding another is refused.

    gen03 archived the file table beside the grant and nothing tied the two
    together, so a grant and an evidence document could describe different
    packs and each hold perfectly on its own.
    """
    from src.training.session import write_once_json

    root = tmp_path / "frozen"
    _copy_tree(_ARCHIVE_FOR_TESTS.parent, root)
    archive = root / phase3c.GENERATION
    assert phase3c.archive_problems(plan, archive, root=ROOT) == []

    grant_path = archive / Path(phase3c.AUTHORIZATION_PATH).name
    bent = _rebuilt(json.loads(grant_path.read_text()),
                    pack_evidence_digest="d" * 64)
    grant_path.unlink()
    write_once_json(grant_path, bent)
    problems = phase3c.archive_problems(plan, archive, root=ROOT)
    assert any("pack evidence" in p for p in problems), problems


def test_a_missing_authorization_file_is_refused(plan, tmp_path):
    with pytest.raises(PlanRefused) as exc:
        phase3c.read_authorization(tmp_path / "nothing.json", plan,
                                   root=ROOT)
    assert "authorised before it starts" in str(exc.value)


# ---------------------------------------------------------------------------
# the archive, the staged copy, and the generation boundary
# ---------------------------------------------------------------------------

def _archive_present() -> bool:
    return (ROOT / phase3c.PLAN_PATH).is_file()


def test_the_staged_copy_is_the_archives_bytes():
    """A copy that drifted would be a second truth nobody chose."""
    if not _archive_present():
        pytest.skip(f"{ARTIFACT_ONLY} {phase3c.ARCHIVE_DIR} is not published")
    problems = phase3c.staged_copy_problems(ROOT)
    assert problems == [], problems


def test_staging_over_a_different_copy_is_refused(tmp_path):
    from src.training.session import copy_once

    if not _archive_present():
        pytest.skip(f"{ARTIFACT_ONLY} {phase3c.ARCHIVE_DIR} is not published")
    fake_root = tmp_path / "tree"
    (fake_root / "gpu_plans").mkdir(parents=True)
    staged = fake_root / phase3c.NODE_PLAN_PATH
    staged.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PlanRefused) as exc:
        phase3c.stage_for_node(fake_root,
                               archive_dir=ROOT / phase3c.ARCHIVE_DIR)
    assert "never rewritten" in str(exc.value)


def test_staging_before_authorising_leaves_the_grant_unstaged(tmp_path):
    """The grant names this pack's digest, so it cannot travel inside it."""
    if not _archive_present():
        pytest.skip(f"{ARTIFACT_ONLY} {phase3c.ARCHIVE_DIR} is not published")
    from src.training import pack as pack_module

    included = {e["path"] for e in pack_module.manifest(ROOT)["include"]}
    assert phase3c.NODE_PLAN_PATH in included
    assert phase3c.NODE_PLAN_PATH.replace("_plan", "_execution_authorization") \
        not in included, (
        "the authorization names the digest of this pack; shipping it "
        "inside the pack puts the value and the thing it authenticates in "
        "one parcel")


def test_the_archive_never_travels_in_a_pack():
    from src.training import pack as pack_module

    included = [e["path"] for e in pack_module.manifest(ROOT)["include"]]
    under_data = [rel for rel in included if rel.startswith("data/")]
    assert under_data == [], under_data


def test_every_superseded_generation_is_recorded_and_not_deleted():
    assert phase3c.SUPERSEDED, "a superseded generation must stay on record"
    names = [e["generation"] for e in phase3c.SUPERSEDED]
    assert names == sorted(names, reverse=True), names
    assert phase3c.GENERATION not in names
    assert names[0] < phase3c.GENERATION
    for entry in phase3c.SUPERSEDED:
        assert entry["superseded_because"]
        assert entry["not_deleted_because"]
        # gen01's files are in gpu_plans/, which is gitignored, so their
        # presence is a property of this disk rather than of the repository.
        # What must hold everywhere is that neither is the current one.
        assert entry["plan"] not in (phase3c.PLAN_PATH,
                                     phase3c.NODE_PLAN_PATH)


def test_gen02_is_superseded_for_an_unreproducible_pack_digest():
    """The named reason, not a generic one, and the files still there.

    gen02's grant bound a pack digest whose bytes were never archived. That
    is the specific defect this generation exists to not repeat, so the
    record says it and the current generation carries the evidence gen02
    lacked.
    """
    entry = next(e for e in phase3c.SUPERSEDED if e["generation"] == "gen02")
    assert "cannot be reproduced" in entry["superseded_because"]
    assert entry["pack_digest_it_names"]
    old_archive = ROOT / "data/phase3c/frozen/gen02"
    if old_archive.is_dir():
        for name in ("plan.json", "execution_authorization.json",
                     "case_membership.json", "isolation_audit.json",
                     "contract_at_freeze.json"):
            assert (old_archive / name).is_file(), name
        stored = json.loads(
            (old_archive / "execution_authorization.json").read_text())
        assert stored["authorization_digest"] == entry["authorization_digest"]
        assert stored["pack_digest"] == entry["pack_digest_it_names"]
        # And it is not reachable as *the* grant any more.
        assert phase3c.AUTHORIZATION_PATH != \
            "data/phase3c/frozen/gen02/execution_authorization.json"


#: gen02's five archived files, byte for byte, as they were committed. Not
#: a convenience: "superseded, not edited" is a claim about bytes, and the
#: only way to hold it is to have written the bytes down. Every one of these
#: is also inside gen02's own execution_authorization, which is why nothing
#: here can be "corrected" -- correcting it would mean writing a new value
#: into a write-once document.
GEN02_FILES: dict = {
    "case_membership.json":
        "a4a4a48b264911d8a58bd0fe5c9c0ab81a713e1ad21d4be65ea37a0bf368a266",
    "contract_at_freeze.json":
        "451f741c6b80f0fab586235618e825a0b78ac6b2baf5df1f63584bc09b4f0481",
    "execution_authorization.json":
        "f3e9375abf811b685bdcb97a09402f34f74b177a20276c59fb50b448bb5e4182",
    "isolation_audit.json":
        "472747b342417d9507ff04e67e0ef7bcff668d4fef26aaaf1e54d4582aa61731",
    "plan.json":
        "f18159db403d46d54159cb514a4617bf020807a5139090b3d6d55f0b60f095d1",
}


def test_the_superseded_archives_are_byte_for_byte_what_they_were():
    """gen02 is not edited. Its bytes are pinned here, so a test says so."""
    from src.training.session import sha256_file

    directory = ROOT / "data/phase3c/frozen/gen02"
    if not directory.is_dir():
        pytest.skip(f"{ARTIFACT_ONLY} the gen02 archive is not published")
    found = sorted(p.name for p in directory.iterdir() if p.is_file())
    assert found == sorted(GEN02_FILES), found
    for name, digest in sorted(GEN02_FILES.items()):
        assert sha256_file(directory / name) == digest, name
    # And it is not what any current path points at.
    for path in (phase3c.PLAN_PATH, phase3c.AUTHORIZATION_PATH,
                 phase3c.PACK_EVIDENCE_PATH,
                 phase3c.CONTRACT_SNAPSHOT_PATH):
        assert "gen02" not in path, path


def test_the_gen01_plan_still_says_what_the_record_says_it_says():
    entry = next(e for e in phase3c.SUPERSEDED if e["generation"] == "gen01")
    path = ROOT / entry["plan"]
    if not path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {entry['plan']} is not published")
    stored = json.loads(path.read_text())
    assert stored["plan_digest"] == entry["plan_digest"]
    assert stored["scorer_source_manifest_digest"] == \
        entry["scorer_source_manifest_digest"]
    # gen01 and gen02 share a plan_digest; gen03 does not, because its
    # contract gained two acceptance criteria and contract_digest is inside
    # plan_digest. The record says so; this is the arithmetic behind it.
    assert entry["plan_digest_shared_with"].startswith("gen02")
    if PLAN_PATH.is_file():
        current = json.loads(PLAN_PATH.read_text())
        assert stored["plan_digest"] != current["plan_digest"]
        assert stored["contract_digest"] != current["contract_digest"]


def _attempts_are_published() -> bool:
    """Whether the attempted executions this record names are in this tree.

    ``supersession_problems`` reads them, and they live under
    ``data/phase3c/``, which the public tree withholds along with the archive
    they belong to. Nothing there can authorise or run a generation anyway,
    so the check is skipped rather than loosened: a record whose evidence is
    absent must stay a refusal wherever the evidence could have been.
    """
    return all((ROOT / entry["attempted_execution"] / "index.json").is_file()
               for entry in phase3c.SUPERSEDED
               if entry.get("attempted_execution"))


def test_the_supersession_record_is_versioned_and_names_both(tmp_path):
    record = phase3c.build_supersession_record()
    assert record["current"] == phase3c.GENERATION
    assert [e["generation"] for e in record["not_executable"]] == \
        [e["generation"] for e in phase3c.SUPERSEDED]
    if _attempts_are_published():
        assert phase3c.supersession_problems(record) == []
    assert phase3c.GENERATION in phase3c.SUPERSESSION_PATH
    body = {k: v for k, v in record.items() if k != "supersession_digest"}
    assert record["supersession_digest"] == digest_obj(body)

    stored = ROOT / phase3c.SUPERSESSION_PATH
    if stored.is_file():
        assert json.loads(stored.read_text()) == record
    # A record naming another generation as current is refused.
    other = {**record, "current": "gen99"}
    other["supersession_digest"] = digest_obj(
        {k: v for k, v in other.items() if k != "supersession_digest"})
    assert phase3c.supersession_problems(other)


def test_the_current_generation_carries_a_contract_snapshot():
    if not _archive_present():
        pytest.skip(f"{ARTIFACT_ONLY} {phase3c.ARCHIVE_DIR} is not published")
    snapshot = json.loads(
        (ROOT / phase3c.CONTRACT_SNAPSHOT_PATH).read_text())
    assert snapshot["generation"] == phase3c.GENERATION
    assert snapshot["contract_digest"] == phase3c.contract_digest(ROOT)
    assert snapshot["contract"] == phase3c.contract_document(ROOT)
    assert [s["generation"] for s in snapshot["superseded"]] == \
        [e["generation"] for e in phase3c.SUPERSEDED]
    assert phase3c.contract_snapshot_problems(
        snapshot, phase3c.read_plan(PLAN_PATH, root=ROOT)) == []


def test_the_archived_grant_authorises_the_archived_plan():
    if not (ROOT / phase3c.AUTHORIZATION_PATH).is_file():
        pytest.skip(f"{ARTIFACT_ONLY} the grant is not published")
    stored_plan = phase3c.read_plan(ROOT / phase3c.PLAN_PATH, root=ROOT)
    stored = phase3c.read_authorization(ROOT / phase3c.AUTHORIZATION_PATH,
                                        stored_plan, root=ROOT)
    assert stored["plan_digest"] == stored_plan["plan_digest"]
    assert phase3c.authorization_problems(stored, stored_plan, root=ROOT,
                                          check_sources=True) == []


# ---------------------------------------------------------------------------
# scoring: fabrication, idempotence, existing-result corruption
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scored(plan, run_dir, grant):
    out_dir, seal, _manifest = run_dir
    problems = phase3c.score_preconditions(
        out_dir, plan, seal, carried_seal_digest=seal["seal_digest"],
        grant=grant, root=ROOT)
    assert problems == [], problems
    by_arm = phase3c.rescore(out_dir, plan)
    return phase3c.score_record(by_arm, plan=plan, k=phase3c.SETTINGS.k,
                                root=ROOT)


def test_every_cell_is_scored_and_the_grid_is_complete(scored):
    expected = phase3c.N_CASES * phase3c.SETTINGS.k * len(phase3c.ARM_ORDER)
    assert scored["draws"] == expected
    assert scored["cases"] == phase3c.N_CASES
    for name in phase3c.ARM_ORDER:
        overall = scored["per_arm"][name]["overall"]
        assert overall["draws"] == phase3c.N_CASES * phase3c.SETTINGS.k
        assert overall["cases"] == phase3c.N_CASES


def test_the_synthetic_arms_separate_the_way_the_scorer_should(scored):
    """Not a finding about the model. A check that the scorer reads the data.

    ``synth_text`` makes arm A overspend and arms B and C not, so a scorer
    that ignored ``raw_text`` would show the three arms identical here.
    """
    rate = {name: scored["per_arm"][name]["overall"]["rates"]
            ["inventory_valid"]["value"] for name in phase3c.ARM_ORDER}
    assert rate["A"] == 0.0
    assert rate["B"] == 1.0 and rate["C"] == 1.0


def test_scoring_the_same_directory_twice_is_idempotent(plan, run_dir):
    """rerun idempotence."""
    out_dir, _seal, _manifest = run_dir
    first = phase3c.score_record(phase3c.rescore(out_dir, plan), plan=plan,
                                 k=phase3c.SETTINGS.k, root=ROOT)
    second = phase3c.score_record(phase3c.rescore(out_dir, plan), plan=plan,
                                  k=phase3c.SETTINGS.k, root=ROOT)
    assert phase3c.scores_identity(first) == phase3c.scores_identity(second)


def test_a_fabricated_summary_from_the_node_is_never_read(plan, run_dir,
                                                          tmp_path):
    """fabricated summary: a node-written summary changes no number."""
    out_dir, _seal, _manifest = run_dir
    copy = tmp_path / "with_summary"
    _copy_tree(out_dir, copy)
    (copy / "run_summary.json").write_text(
        json.dumps({"core_success_at_4": 0.99}), encoding="utf-8")
    record = phase3c.score_record(phase3c.rescore(copy, plan), plan=plan,
                                  k=phase3c.SETTINGS.k, root=ROOT)
    baseline = phase3c.score_record(phase3c.rescore(out_dir, plan), plan=plan,
                                    k=phase3c.SETTINGS.k, root=ROOT)
    assert (phase3c.scores_identity(record)
            == phase3c.scores_identity(baseline))


def test_a_self_consistent_but_fabricated_score_is_caught(plan, run_dir):
    """self-consistent but fabricated score.

    A record whose per-arm block was rewritten and whose own digests were
    recomputed to match is still not the record the samples imply, and
    ``scores_identity`` is what says so.
    """
    out_dir, _seal, _manifest = run_dir
    real = phase3c.score_record(phase3c.rescore(out_dir, plan), plan=plan,
                                k=phase3c.SETTINGS.k, root=ROOT)
    fake = json.loads(json.dumps(real))
    fake["per_arm"]["C"]["overall"]["core_success_at_4"]["value"] = 0.99
    assert phase3c.scores_identity(fake) != phase3c.scores_identity(real)


def test_scoring_refuses_when_the_scorer_is_not_the_plans(plan, run_dir,
                                                          grant):
    """existing-result corruption, on the scorer side."""
    out_dir, seal, _manifest = run_dir
    wrong = dict(plan, scorer_source_manifest_digest="0" * 64)
    problems = phase3c.score_preconditions(
        out_dir, wrong, seal, carried_seal_digest=seal["seal_digest"],
        grant=grant, root=ROOT)
    assert any("not the scorer this plan was approved with" in p
               for p in problems)


def test_scoring_refuses_a_directory_whose_samples_changed(plan, run_dir,
                                                           tmp_path,
                                                           grant):
    """existing-result corruption, on the samples side."""
    out_dir, seal, _manifest = run_dir
    copy = tmp_path / "corrupt"
    _copy_tree(out_dir, copy)
    member = copy / phase3c.samples_member(1)
    rows = phase3c.read_rows(member)
    # Appended rather than replaced: a fixed replacement can happen to equal
    # what the row already said, and a tampering test that tampers with
    # nothing passes for the wrong reason.
    rows[0]["raw_text"] = rows[0]["raw_text"] + "1x1 (40,40,40)\n"
    _write_rows(member, rows)
    problems = phase3c.score_preconditions(
        copy, plan, seal, carried_seal_digest=seal["seal_digest"],
        grant=grant, root=ROOT)
    assert any("hashes to" in p for p in problems)


def test_a_row_from_another_plan_stops_the_scorer(plan, run_dir, tmp_path):
    out_dir, _seal, _manifest = run_dir
    copy = tmp_path / "foreign"
    _copy_tree(out_dir, copy)
    member = copy / phase3c.samples_member(2)
    rows = phase3c.read_rows(member)
    rows[0]["plan_digest"] = "0" * 64
    _write_rows(member, rows)
    with pytest.raises(PlanRefused) as exc:
        phase3c.rescore(copy, plan)
    assert "different plan" in str(exc.value)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def _published(tmp_path, name, plan, run_dir, scored, grant, *,
               carried=None):
    """A run directory with scores and a receipt, ready to report over.

    Any run document a previous test left in the shared ``run_dir`` is
    removed from the copy first, so this builds the state it says it does
    rather than inheriting one.
    """
    from src.training.session import write_once_json

    out_dir, seal, _manifest = run_dir
    copy = tmp_path / name
    _copy_tree(out_dir, copy)
    for stale in (phase3c.SCORES_NAME, phase3c.RECEIPT_NAME,
                  *phase3c.PUBLISHED_NAMES):
        (copy / stale).unlink(missing_ok=True)
    write_once_json(copy / phase3c.SCORES_NAME, scored)
    receipt = phase3c.build_receipt(
        copy, seal, carried or seal["seal_digest"], plan=plan, grant=grant,
        verified_at=STAMP)
    write_once_json(copy / phase3c.RECEIPT_NAME, receipt)
    return copy, seal


@pytest.fixture(scope="module")
def published(tmp_path_factory, plan, run_dir, scored, grant):
    """One fully published report directory, built once.

    Publishing re-derives the whole score record, which is the expensive
    thing in this suite. The tests that need many *variations* of a
    published directory copy this one and check
    ``published_member_problems``; a separate test proves ``verify`` calls
    that function, so the cheap checks are not checking a different rule.
    """
    out = tmp_path_factory.mktemp("phase3c_published")
    copy, seal = _published(out, "published", plan, run_dir, scored, grant)
    written = phase3c_report.write_report(
        copy, plan, grant=grant, seal=seal, root=ROOT,
        archive_dir=_ARCHIVE_FOR_TESTS)
    return copy, seal, written


def test_the_report_renders_and_names_no_physical_claim(published, plan,
                                                        scored, grant):
    copy, seal, written = published
    body = written["report"].read_text(encoding="utf-8")
    assert phase3c_report.forbidden_terms_in(body) == []
    assert "geometric" in body
    assert scored["plan_digest"] in body
    assert "No physical assembly was attempted." in body

    successes = json.loads(written[phase3c_report.SUCCESS_INDEX_NAME]
                           .read_text())
    failures = json.loads(written[phase3c_report.FAILURE_INDEX_NAME]
                          .read_text())
    assert successes["n"] + failures["n"] == phase3c.N_CASES
    known = {c["case_id"] for c in plan["cases"]}
    for entry in successes["cases"] + failures["cases"]:
        assert entry["case_id"] in known
        assert entry["samples"], "an index entry must point at its samples"

    outcome = phase3c_report.verify(copy, plan, grant=grant, seal=seal,
                                    root=ROOT,
                                    archive_dir=_ARCHIVE_FOR_TESTS)
    assert outcome["verified"], outcome["problems"]
    # Idempotent: the same content is accepted, nothing is rewritten.
    again = phase3c_report.write_report(
        copy, plan, grant=grant, seal=seal, root=ROOT,
        archive_dir=_ARCHIVE_FOR_TESTS)
    assert again == written


def test_the_reproduce_document_is_one_of_the_published_members(published):
    copy, _seal, _written = published
    for name in phase3c_report.PUBLISHED_MEMBERS:
        assert (copy / name).is_file(), name


def test_deleting_any_published_member_is_caught(published, plan,
                                                 tmp_path):
    """Three of four documents is not a published report."""
    source, _seal, _written = published
    for name in phase3c_report.PUBLISHED_MEMBERS:
        copy = tmp_path / f"missing_{name}"
        _copy_tree(source, copy)
        (copy / name).unlink()
        problems = phase3c_report.published_member_problems(copy, plan)
        assert any(name in p and "not here" in p for p in problems), problems


def test_verify_fails_when_a_published_member_is_missing(published, plan,
                                                         grant, tmp_path):
    """The wiring: the cheap member check is what ``verify`` calls.

    Run once, over the full chain, so the four cheap assertions above are
    known to be checking the rule ``verify`` enforces rather than a
    parallel one.
    """
    source, seal, _written = published
    copy = tmp_path / "verify_missing"
    _copy_tree(source, copy)
    (copy / phase3c_report.REPRODUCE_NAME).unlink()
    outcome = phase3c_report.verify(copy, plan, grant=grant, seal=seal,
                                    root=ROOT,
                                    archive_dir=_ARCHIVE_FOR_TESTS)
    assert not outcome["verified"]
    assert any(phase3c_report.REPRODUCE_NAME in p and "not here" in p
               for p in outcome["problems"]), outcome["problems"]


def test_republishing_different_content_is_refused(published, plan, grant,
                                                   tmp_path):
    """Write-once, member by member. An edited artefact is not rewritten."""
    source, seal, _written = published
    for name, edit in (
            (phase3c_report.REPORT_NAME, "text"),
            (phase3c_report.REPRODUCE_NAME, "text"),
            (phase3c_report.SUCCESS_INDEX_NAME, "json"),
            (phase3c_report.FAILURE_INDEX_NAME, "json")):
        copy = tmp_path / f"clobber_{name}"
        _copy_tree(source, copy)
        path = copy / name
        if edit == "text":
            path.write_text(path.read_text() + "\nand one more claim\n",
                            encoding="utf-8")
        else:
            body = json.loads(path.read_text())
            body["n"] = (body["n"] or 0) + 1
            path.write_text(json.dumps(body, indent=2) + "\n")
        # The cheap check sees it, and so does a republish.
        assert any(name in p for p in
                   phase3c_report.published_member_problems(copy, plan))
        with pytest.raises(PlanRefused) as exc:
            phase3c_report.write_report(copy, plan, grant=grant, seal=seal,
                                        root=ROOT,
                                        archive_dir=_ARCHIVE_FOR_TESTS)
        assert "already holds a different" in str(exc.value), name


def test_a_report_over_an_unverified_receipt_is_refused(plan, run_dir,
                                                        scored, tmp_path,
                                                        grant):
    copy, seal = _published(tmp_path, "unverified", plan, run_dir, scored,
                            grant, carried="f" * 64)
    with pytest.raises(PlanRefused) as exc:
        phase3c_report.write_report(copy, plan, grant=grant, seal=seal,
                                    root=ROOT,
                                    archive_dir=_ARCHIVE_FOR_TESTS)
    assert "does not hold" in str(exc.value)
    assert not (copy / phase3c_report.REPORT_NAME).exists()


def test_the_receipt_is_a_whole_expected_record_not_a_subset(plan, run_dir,
                                                             grant, tmp_path):
    """Every field, rebuilt from the plan, the grant, the seal and the bytes.

    The old check named the fields it compared, so a field nobody thought of
    was a field nothing checked. Here each field of the expected record is
    edited in turn -- with ``receipt_digest`` recomputed so the document
    stays internally consistent -- and every one of them must be refused.
    """
    from src.training.session import write_once_json

    out_dir, seal, _manifest = run_dir
    copy = tmp_path / "whole_record"
    _copy_tree(out_dir, copy)
    for stale in (phase3c.RECEIPT_NAME, phase3c.SCORES_NAME,
                  *phase3c.PUBLISHED_NAMES):
        (copy / stale).unlink(missing_ok=True)
    good = phase3c.build_receipt(copy, seal, seal["seal_digest"], plan=plan,
                                 grant=grant, verified_at=STAMP)
    assert phase3c.receipt_problems(good, copy, plan=plan, seal=seal,
                                    grant=grant) == []
    write_once_json(copy / phase3c.RECEIPT_NAME, good)

    swaps = {
        "kind": "brickagain.something_else",
        "contract_digest": "a" * 64,
        "plan_digest": "b" * 64,
        "authorization_digest": "c" * 64,
        "pack_digest": "d" * 64,
        "pack_evidence_digest": "e" * 64,
        "dependency_digest": "f" * 64,
        "adapter_sha256": {},
        "gate_sources": {},
        "generation_source_manifest_digest": "1" * 64,
        "seal_digest": "2" * 64,
        "carried_seal_digest": "3" * 64,
        "manifest_digest": "4" * 64,
        "members": {},
        "verified_on": "the node",
    }
    for field, value in swaps.items():
        assert field in good, f"{field} is not in the expected record"
        bent = _rebuilt_receipt(good, **{field: value})
        problems = phase3c.receipt_problems(bent, copy, plan=plan, seal=seal,
                                            grant=grant)
        assert problems, f"{field} was changed and nothing objected"

    # A field the expected record does not have is itself a problem.
    bent = _rebuilt_receipt(good, note="looks fine")
    assert any("which a receipt for this run does not have" in p
               for p in phase3c.receipt_problems(bent, copy, plan=plan,
                                                 seal=seal, grant=grant))


def test_a_receipt_whose_member_table_was_edited_is_refused(plan, run_dir,
                                                            grant, tmp_path):
    """``members`` is the actual rehash, not what the receipt claims."""
    out_dir, seal, _manifest = run_dir
    copy = tmp_path / "bent_members"
    _copy_tree(out_dir, copy)
    good = phase3c.build_receipt(copy, seal, seal["seal_digest"], plan=plan,
                                 grant=grant, verified_at=STAMP)
    members = json.loads(json.dumps(good["members"]))
    first = sorted(members)[0]
    members[first] = {**members[first], "size": members[first]["size"] + 1}
    bent = _rebuilt_receipt(good, members=members)
    problems = phase3c.receipt_problems(bent, copy, plan=plan, seal=seal,
                                        grant=grant)
    assert any("members" in p for p in problems), problems


def test_verifying_twice_across_a_clock_tick_is_idempotent(plan, run_dir,
                                                           grant, tmp_path,
                                                           capsys):
    """The regression: ``verified_at`` moved and the second run refused.

    Two verifications of an unchanged directory differ only in when they
    happened. Comparing them by ``receipt_digest`` -- which covers the
    timestamp -- made the second one call its own unchanged conclusion "a
    different receipt". This drives the real CLI twice, with a real clock
    between, and both must exit 0 with the first receipt still on disk.
    """
    import time

    from src.training.session import now_iso

    cli = _cli()
    source, seal, _manifest = run_dir
    out_dir = tmp_path / "idempotent"
    _copy_tree(source, out_dir)
    for stale in (phase3c.SCORES_NAME, phase3c.RECEIPT_NAME,
                  *phase3c.PUBLISHED_NAMES):
        (out_dir / stale).unlink(missing_ok=True)
    plan_path, grant_path = _staged(tmp_path, plan, grant)
    argv = ["--plan", str(plan_path), "--authorization", str(grant_path),
            "--verify", "--out-dir", str(out_dir),
            "--carried-seal-digest", seal["seal_digest"]]

    assert cli.main(argv) == 0, capsys.readouterr()
    first = (out_dir / phase3c.RECEIPT_NAME).read_bytes()
    stored = json.loads(first)

    time.sleep(1.1)                       # past this clock's resolution
    assert phase3c.build_receipt(
        out_dir, seal, seal["seal_digest"], plan=plan, grant=grant,
        verified_at=now_iso())["verified_at"] != stored["verified_at"]

    assert cli.main(argv) == 0, capsys.readouterr()
    assert (out_dir / phase3c.RECEIPT_NAME).read_bytes() == first, \
        "the second verification rewrote a receipt it agreed with"


def test_the_contract_snapshot_must_hold_the_contract_it_names(plan):
    """A snapshot naming one contract and holding another used to pass."""
    snapshot = phase3c.build_contract_snapshot(plan, root=ROOT)
    assert phase3c.contract_snapshot_problems(snapshot, plan) == []

    bent = {**snapshot, "contract": {**snapshot["contract"], "kind": "other"}}
    bent["snapshot_digest"] = digest_obj(
        {k: v for k, v in bent.items() if k != "snapshot_digest"})
    problems = phase3c.contract_snapshot_problems(bent, plan)
    assert any("contract it holds digests to" in p for p in problems), \
        problems

    # And with the payload digest updated too: now it is honest about what
    # it holds and dishonest about which contract that is.
    bent["payload_digest"] = digest_obj(bent["contract"])
    bent["snapshot_digest"] = digest_obj(
        {k: v for k, v in bent.items() if k != "snapshot_digest"})
    assert any("the plan was built on" in p
               for p in phase3c.contract_snapshot_problems(bent, plan))


def test_the_archive_closure_is_all_six_documents(plan, tmp_path):
    """Grant, evidence, contract snapshot and supersession are required."""
    if _ARCHIVE_FOR_TESTS is None:
        pytest.skip(f"{ARTIFACT_ONLY} there is no archive to check")
    root = tmp_path / "frozen"
    _copy_tree(_ARCHIVE_FOR_TESTS.parent, root)
    archive = root / phase3c.GENERATION
    assert phase3c.archive_problems(plan, archive, root=ROOT) == []

    for name, fragment in (
            (Path(phase3c.AUTHORIZATION_PATH).name, "authorised"),
            (Path(phase3c.PACK_EVIDENCE_PATH).name, "no path back"),
            (Path(phase3c.CONTRACT_SNAPSHOT_PATH).name, "is not in")):
        again = tmp_path / f"missing_{name}"
        _copy_tree(root, again)
        (again / phase3c.GENERATION / name).unlink()
        problems = phase3c.archive_problems(
            plan, again / phase3c.GENERATION, root=ROOT)
        assert any(fragment in p for p in problems), (name, problems)

    again = tmp_path / "missing_supersession"
    _copy_tree(root, again)
    (again / Path(phase3c.SUPERSESSION_PATH).name).unlink()
    assert any("may run" in p for p in phase3c.archive_problems(
        plan, again / phase3c.GENERATION, root=ROOT))


def test_the_score_record_carries_its_own_digest(plan, scored):
    """And identity covers everything but that digest."""
    assert phase3c._is_digest(scored["scores_digest"])
    assert scored["scores_digest"] == digest_obj(
        phase3c.scores_identity(scored))
    assert "scores_digest" not in phase3c.scores_identity(scored)
    assert phase3c.scores_problems(scored, scored) == []

    # Fields the old eleven-field identity left out are now covered.
    for field in ("seeds", "arms", "primary_contrast", "uncertainty",
                  "scorer_source_manifest_digest", "note"):
        bent = {**scored, field: "changed"}
        bent["scores_digest"] = digest_obj(phase3c.scores_identity(bent))
        problems = phase3c.scores_problems(bent, scored)
        assert any(field in p for p in problems), (field, problems)


def test_a_reformatted_published_member_is_refused(published, plan, grant,
                                                   tmp_path):
    """Same object, different bytes. A published artefact is bytes."""
    source, seal, _written = published
    copy = tmp_path / "reformatted"
    _copy_tree(source, copy)
    path = copy / phase3c_report.SUCCESS_INDEX_NAME
    body = json.loads(path.read_text())
    path.unlink()
    path.write_text(json.dumps(body, indent=4, sort_keys=True) + "\n",
                    encoding="utf-8")

    assert json.loads(path.read_text()) == body, "the object is unchanged"
    problems = phase3c_report.published_member_problems(copy, plan)
    assert any("different bytes" in p for p in problems), problems
    with pytest.raises(PlanRefused) as exc:
        phase3c_report.write_report(copy, plan, grant=grant, seal=seal,
                                    root=ROOT,
                                    archive_dir=_ARCHIVE_FOR_TESTS)
    assert "reformatted" in str(exc.value) or "different bytes" in str(
        exc.value)


def test_a_blocked_member_stops_all_four_from_being_written(plan, run_dir,
                                                            scored, tmp_path,
                                                            grant):
    """No half-publication: one member in the way and none is written."""
    copy, seal = _published(tmp_path, "half", plan, run_dir, scored, grant)
    blocker = copy / phase3c_report.FAILURE_INDEX_NAME
    blocker.write_text('{"kind": "something else"}\n', encoding="utf-8")

    with pytest.raises(PlanRefused) as exc:
        phase3c_report.write_report(copy, plan, grant=grant, seal=seal,
                                    root=ROOT,
                                    archive_dir=_ARCHIVE_FOR_TESTS)
    assert "none is written" in str(exc.value)
    for name in phase3c_report.PUBLISHED_MEMBERS:
        if name == phase3c_report.FAILURE_INDEX_NAME:
            continue
        assert not (copy / name).exists(), name


def _rebuilt_receipt(receipt: dict, **changes) -> dict:
    """A receipt with fields changed and its own digest made to agree."""
    body = {**receipt, **changes}
    body["receipt_digest"] = digest_obj(
        {k: v for k, v in body.items() if k != "receipt_digest"})
    return body


def test_a_fabricated_verified_receipt_publishes_nothing(plan, run_dir,
                                                         scored, tmp_path,
                                                         grant):
    """``verified: true`` beside a non-empty problem list and a bad seal.

    This is the fail-open the report had: it read that one field and
    believed it. Here the flag says clean, the receipt's own list says it is
    not, the seal digest it names is not the seal's, and the digest over the
    whole thing is recomputed so nothing is internally inconsistent.
    """
    from src.training.session import write_once_json

    out_dir, seal, _manifest = run_dir
    copy = tmp_path / "fabricated"
    _copy_tree(out_dir, copy)
    write_once_json(copy / phase3c.SCORES_NAME, scored)
    receipt = phase3c.build_receipt(copy, seal, seal["seal_digest"],
                                    plan=plan, grant=grant,
                                    verified_at=STAMP)
    receipt["problems"] = ["a member did not re-hash"]
    receipt["seal_digest"] = "e" * 64
    receipt["carried_seal_digest"] = "e" * 64
    receipt["verified"] = True
    receipt["receipt_digest"] = digest_obj(
        {k: v for k, v in receipt.items() if k != "receipt_digest"})
    write_once_json(copy / phase3c.RECEIPT_NAME, receipt)

    assert phase3c.receipt_problems(receipt, copy, plan=plan, seal=seal,
                                    grant=grant)
    with pytest.raises(PlanRefused):
        phase3c_report.write_report(copy, plan, grant=grant, seal=seal,
                                    root=ROOT,
                                    archive_dir=_ARCHIVE_FOR_TESTS)
    for name in phase3c_report.PUBLISHED_MEMBERS:
        assert not (copy / name).exists(), name


def test_a_tampered_receipt_rehashed_still_fails_the_chain(plan, run_dir,
                                                           scored, tmp_path,
                                                           grant):
    """Recomputing ``receipt_digest`` does not make the chain hold."""
    from src.training.session import write_once_json

    out_dir, seal, _manifest = run_dir
    copy = tmp_path / "rehashed_receipt"
    _copy_tree(out_dir, copy)
    write_once_json(copy / phase3c.SCORES_NAME, scored)
    receipt = phase3c.build_receipt(copy, seal, seal["seal_digest"],
                                    plan=plan, grant=grant,
                                    verified_at=STAMP)
    assert receipt["verified"]
    receipt["pack_digest"] = "b" * 64
    receipt["receipt_digest"] = digest_obj(
        {k: v for k, v in receipt.items() if k != "receipt_digest"})
    write_once_json(copy / phase3c.RECEIPT_NAME, receipt)

    problems = phase3c.receipt_problems(receipt, copy, plan=plan, seal=seal,
                                        grant=grant)
    assert any("pack_digest" in p for p in problems), problems
    with pytest.raises(PlanRefused):
        phase3c_report.write_report(copy, plan, grant=grant, seal=seal,
                                    root=ROOT,
                                    archive_dir=_ARCHIVE_FOR_TESTS)


def test_a_report_over_fabricated_scores_is_refused(plan, run_dir, scored,
                                                    tmp_path, grant):
    """The scores are recomputed from the samples, not read and formatted."""
    from src.training.session import write_once_json

    out_dir, seal, _manifest = run_dir
    copy = tmp_path / "bent_scores"
    _copy_tree(out_dir, copy)
    bent = json.loads(json.dumps(scored))
    entry = bent["per_arm"]["C"]["overall"]["rates"][
        "stud_only_connected"]
    assert entry["value"] != 1.0, "pick a rate this data does not already hit"
    entry["value"] = 1.0
    entry["numerator"] = entry["denominator"]
    write_once_json(copy / phase3c.SCORES_NAME, bent)
    receipt = phase3c.build_receipt(copy, seal, seal["seal_digest"],
                                    plan=plan, grant=grant,
                                    verified_at=STAMP)
    write_once_json(copy / phase3c.RECEIPT_NAME, receipt)
    with pytest.raises(PlanRefused) as exc:
        phase3c_report.write_report(copy, plan, grant=grant, seal=seal,
                                    root=ROOT,
                                    archive_dir=_ARCHIVE_FOR_TESTS)
    assert "not what these samples derive" in str(exc.value)


def test_a_report_that_would_misname_a_geometric_check_is_refused():
    body = "stud_only_connected shows the model is physically stable."
    assert phase3c_report.forbidden_terms_in(body) == ["physically stable"]


# ---------------------------------------------------------------------------
# the CLI's guards
# ---------------------------------------------------------------------------

def _cli():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "phase3c_cli", ROOT / "scripts" / "59_phase3c.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _staged(tmp_path, plan, grant):
    """A whole archive, plus a grant carried to a path of its own.

    The archive is copied because ``--report`` re-digests the membership
    and the audit against the plan. The grant is deliberately *outside* it,
    under a name the defaulting rule would never find, because that is the
    path that used to raise ``AttributeError`` before any check ran.
    """
    from src.training.session import write_once_json

    if _ARCHIVE_FOR_TESTS is None:
        pytest.skip(f"{ARTIFACT_ONLY} there is no archive to stage")
    # The whole layout, not just the generation directory: the supersession
    # record lives beside it, and ``archive_problems`` requires it.
    root = tmp_path / "frozen"
    where = root / phase3c.GENERATION
    _copy_tree(_ARCHIVE_FOR_TESTS.parent, root)
    # The archive keeps its grant, because that is what a real Mac archive
    # holds and what ``archive_problems`` requires. What is *carried* is a
    # copy under a name the defaulting rule would never find, so pointing
    # ``--authorization`` at it exercises the explicit path -- the one that
    # used to raise before any check ran.
    grant_path = tmp_path / "carried_grant.json"
    write_once_json(grant_path, grant)
    return where / Path(phase3c.PLAN_PATH).name, grant_path


def test_an_explicit_authorization_reaches_the_checks_not_an_attribute_error(
        plan, grant, run_dir, tmp_path, capsys):
    """Every formal mode, through argparse, with ``--authorization`` given.

    The regression: ``_authorization_path`` tested ``args.authorization``
    and returned ``args.grant``, which argparse never defines. So each of
    these commands died with ``AttributeError`` the moment a path was named
    explicitly -- the exact form the node and the reproduce document use --
    and the suite did not notice because it called the helpers directly.

    Each command below must reach a *semantic* verdict: exit 0 where the
    chain holds, or a named refusal where it does not. An ``AttributeError``
    would propagate out of ``main`` and fail the test rather than return a
    code, which is the assertion.
    """
    cli = _cli()
    source, seal, _manifest = run_dir
    # A copy: these three modes write into the directory, and ``run_dir`` is
    # module-scoped. A test that publishes into the shared fixture leaves
    # every later test reading a directory this one changed.
    out_dir = tmp_path / "cli_run"
    _copy_tree(source, out_dir)
    for stale in (phase3c.SCORES_NAME, phase3c.RECEIPT_NAME,
                  *phase3c.PUBLISHED_NAMES):
        (out_dir / stale).unlink(missing_ok=True)
    plan_path, grant_path = _staged(tmp_path, plan, grant)
    common = ["--plan", str(plan_path), "--authorization", str(grant_path)]

    # --verify: the whole Mac chain, over a clean directory.
    code = cli.main([*common, "--verify", "--out-dir", str(out_dir),
                     "--carried-seal-digest", seal["seal_digest"]])
    assert code == 0, capsys.readouterr()

    # --score: same, and it re-derives the receipt rather than reading it.
    code = cli.main([*common, "--score", "--out-dir", str(out_dir),
                     "--carried-seal-digest", seal["seal_digest"]])
    assert code == 0, capsys.readouterr()

    # --report: the full re-derivation, over the directory the two above
    # just published into. It re-verifies what it wrote before returning 0.
    code = cli.main([*common, "--report", "--out-dir", str(out_dir)])
    assert code == 0, capsys.readouterr()
    for name in phase3c_report.PUBLISHED_MEMBERS:
        assert (out_dir / name).is_file(), name


def test_an_explicit_authorization_that_is_wrong_refuses_by_name(
        plan, grant, run_dir, tmp_path, capsys):
    """The other half: a named grant that does not hold is a refusal."""
    from src.training.session import write_once_json

    cli = _cli()
    out_dir, seal, _manifest = run_dir
    plan_path, _ = _staged(tmp_path, plan, grant)
    bent = _rebuilt(grant, pack_digest="c" * 64)
    bad = tmp_path / "wrong_grant.json"
    write_once_json(bad, bent)

    for mode, extra in (("--verify", ["--carried-seal-digest",
                                      seal["seal_digest"]]),
                        ("--score", ["--carried-seal-digest",
                                     seal["seal_digest"]]),
                        ("--report", [])):
        code = cli.main(["--plan", str(plan_path), "--authorization",
                         str(bad), mode, "--out-dir", str(out_dir), *extra])
        assert code == 2, mode
        # ``--verify`` reports a dirty receipt on stdout and still exits 2;
        # the other two refuse on stderr. Both are named refusals, so the
        # assertion is over what the command said, not where it said it.
        captured = capsys.readouterr()
        said = (captured.out + captured.err).lower()
        assert ("authoris" in said or "pack_digest" in said
                or "grant:" in said), (mode, said[:400])


def test_a_missing_explicit_authorization_refuses_rather_than_defaults(
        plan, run_dir, tmp_path, capsys):
    """A named path that is not there is a refusal, not a fallback."""
    cli = _cli()
    out_dir, seal, _manifest = run_dir
    plan_path, _ = _staged(tmp_path, plan, synth_authorization(plan))
    code = cli.main(["--plan", str(plan_path), "--authorization",
                     str(tmp_path / "nothing.json"), "--verify",
                     "--out-dir", str(out_dir),
                     "--carried-seal-digest", seal["seal_digest"]])
    assert code == 2
    assert "is not here" in capsys.readouterr().err


def test_seal_and_run_accept_an_explicit_authorization(plan, grant, tmp_path,
                                                       capsys):
    """The node's two modes, parsed. Neither may raise before it checks.

    ``--seal`` is checked to a named refusal here rather than to success:
    the directory is deliberately incomplete. ``--run`` refuses earlier
    still, on the machine guard, which is the correct answer on a Mac and is
    reached only if the argument parsing worked.
    """
    cli = _cli()
    plan_path, grant_path = _staged(tmp_path, plan, grant)
    empty = tmp_path / "empty_run"
    empty.mkdir()

    code = cli.main(["--seal", "--plan", str(plan_path), "--authorization",
                     str(grant_path), "--out-dir", str(empty)])
    assert code == 2
    assert phase3c.MANIFEST_NAME in capsys.readouterr().err

    code = cli.main(["--run", "--step", "0", "--plan", str(plan_path),
                     "--authorization", str(grant_path),
                     "--out-dir", str(empty),
                     "--expected-pack-digest", PACK_DIGEST,
                     "--expected-dependency-digest", DEPENDENCY_DIGEST,
                     "--adapter-dir", str(tmp_path)])
    assert code == 2
    assert "refusing to run here" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# every row is bound to the execution manifest that is actually here
# ---------------------------------------------------------------------------

def _rewrite_one_row(out_dir: Path, step: int, **changes) -> None:
    """Change one field of one row and rewrite the member canonically."""
    path = out_dir / phase3c.samples_member(step)
    rows = phase3c.read_rows(path)
    rows[0] = {**rows[0], **changes}
    path.unlink()
    _write_rows(path, rows)


def test_one_row_naming_another_manifest_fails_even_after_a_full_rehash(
        plan, run_dir, tmp_path, grant, scored):
    """The named adversarial case: 1,920 rows, one field, everything rehashed.

    A complete run is copied, exactly one row's ``manifest_digest`` is
    replaced, and then *every recomputable digest downstream is recomputed*:
    the member's SHA-256 and size, a fresh seal over it, a fresh carried
    digest, and a fresh receipt. Nothing is internally inconsistent any
    more -- which is precisely the attack, because the field the row states
    about itself was never compared to the manifest in the directory.

    The receipt must not verify, and neither scoring nor reporting may
    produce anything.
    """
    from src.training.session import write_once_json

    out_dir, _seal, manifest = run_dir
    copy = tmp_path / "rebound"
    _copy_tree(out_dir, copy)
    for name in (phase3c.SEAL_NAME, phase3c.RECEIPT_NAME,
                 phase3c.SCORES_NAME):
        (copy / name).unlink(missing_ok=True)
    _rewrite_one_row(copy, 0, manifest_digest="a" * 64)

    # Everything recomputable, recomputed.
    reseal = phase3c.build_seal(copy, manifest)
    assert phase3c.seal_problems(reseal, plan) == []
    write_once_json(copy / phase3c.SEAL_NAME, reseal)
    receipt = phase3c.build_receipt(copy, reseal, reseal["seal_digest"],
                                    plan=plan, grant=grant,
                                    verified_at=STAMP)

    assert not receipt["verified"]
    assert any("execution manifest" in p for p in receipt["problems"]), \
        receipt["problems"]

    assert phase3c.score_preconditions(
        copy, plan, reseal, carried_seal_digest=reseal["seal_digest"],
        grant=grant, root=ROOT)
    with pytest.raises(PlanRefused):
        phase3c.rescore(copy, plan)

    write_once_json(copy / phase3c.RECEIPT_NAME, receipt)
    write_once_json(copy / phase3c.SCORES_NAME, scored)
    with pytest.raises(PlanRefused):
        phase3c_report.write_report(copy, plan, grant=grant, seal=reseal,
                                    root=ROOT,
                                    archive_dir=_ARCHIVE_FOR_TESTS)
    assert not (copy / phase3c_report.REPORT_NAME).exists()


def test_a_row_whose_manifest_digest_is_not_a_digest_is_refused(plan,
                                                                run_dir,
                                                                tmp_path):
    out_dir, _seal, manifest = run_dir
    copy = tmp_path / "not_a_digest"
    _copy_tree(out_dir, copy)
    _rewrite_one_row(copy, 1, manifest_digest="")
    rows = phase3c.read_rows(copy / phase3c.samples_member(1))
    problems = phase3c.step_problems(
        rows, plan, 1, manifest_digest=manifest["manifest_digest"])
    assert any("not a SHA-256" in p for p in problems), problems


def test_swapping_the_manifest_for_another_run_is_refused(plan, run_dir,
                                                          tmp_path, grant):
    """The rows are unchanged; the manifest under them is a different one."""
    from src.training.session import write_once_json

    out_dir, seal, _manifest = run_dir
    copy = tmp_path / "swapped_manifest"
    _copy_tree(out_dir, copy)
    (copy / phase3c.MANIFEST_NAME).unlink()
    other = phase3c.build_execution_manifest(
        plan=plan, grant=grant, observed=synth_environment(),
        started_at="2026-02-02T00:00Z",
        recomputed_sources=phase3c.generation_source_manifest(ROOT))
    assert other["manifest_digest"] != seal["manifest_digest"]
    write_once_json(copy / phase3c.MANIFEST_NAME, other)

    problems = phase3c.manifest_chain_problems(copy, seal, plan=plan,
                                               grant=grant)
    assert any("different execution manifest" in p for p in problems)
    assert any("name an execution manifest other than" in p
               for p in problems), problems


def test_the_row_binding_is_checked_without_a_manifest_argument(plan,
                                                                run_dir):
    """``rescore`` reads the manifest itself; no caller supplies a digest."""
    out_dir, _seal, manifest = run_dir
    assert phase3c.read_execution_manifest(out_dir)["manifest_digest"] == \
        manifest["manifest_digest"]
    scores = phase3c.rescore(out_dir, plan)
    assert sum(len(v) for v in scores.values()) == \
        phase3c.N_CASES * phase3c.SETTINGS.k * len(phase3c.ARM_ORDER)


# ---------------------------------------------------------------------------
# the pack, the archive and the evidence for the digest a grant names
# ---------------------------------------------------------------------------

def test_the_pack_carries_this_generations_staged_plan():
    """The binding, in its third design.

    It was a typed allowlist entry, and the suite asserted the typing. That
    made a Phase 3C bump edit ``src/training/pack.py`` -- a file the V1
    visual run's frozen source manifest pins -- and repairing V1 edits a file
    the Phase 3C pack carries. gen05 was frozen inside that cycle.

    So the allowlist names no plan at all now, and the binding is between
    ``phase3c`` and the pointer document it writes. What this test guards is
    that the two still agree and that no generation string has crept back
    into ``pack.py``.
    """
    from src.training import pack as pack_module

    assert not [p for p in pack_module.PACK_ALLOW
                if p.startswith("gpu_plans/phase3c_")], \
        "the allowlist names a Phase 3C plan again, and the cycle with it"

    if not (ROOT / phase3c.NODE_PLAN_PATH).is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {phase3c.NODE_PLAN_PATH} is not staged")
    resolved, problems = pack_module.staged_phase3c_plan(ROOT)
    assert problems == [], problems
    assert resolved == phase3c.NODE_PLAN_PATH


def test_the_archived_pack_evidence_recomputes_the_authorised_digest():
    """A digest nobody can re-derive binds nothing; gen02 is why.

    Historical source identity and live-tree drift are deliberately separate
    now.  The former is re-derived from an exact archive; the latter remains a
    checked disclosure instead of being mistaken for the historical bytes.
    """
    path = ROOT / phase3c.PACK_EVIDENCE_PATH
    grant_path = ROOT / phase3c.AUTHORIZATION_PATH
    if not path.is_file() or not grant_path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {phase3c.PACK_EVIDENCE_PATH} is not "
                    "published")
    evidence = json.loads(path.read_text())
    grant = json.loads(grant_path.read_text())
    assert phase3c.pack_evidence_problems(
        evidence, pack_digest=grant["pack_digest"]) == []
    import importlib.util

    script = ROOT / "scripts/71_evidence_chain.py"
    spec = importlib.util.spec_from_file_location("evidence_chain_v2", script)
    evidence_v2 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(evidence_v2)
    binding = json.loads((ROOT / evidence_v2.SOURCE_BINDING).read_text())
    declared = json.loads((ROOT / evidence_v2.LIVE_DRIFT).read_text())
    assert evidence_v2.historical_snapshot_problems(
        binding, ROOT / evidence_v2.SOURCE_ARCHIVE, root=ROOT
    ) == []
    assert evidence_v2.live_drift_problems(ROOT, binding, declared) == []


def test_pack_evidence_that_does_not_recompute_is_refused():
    evidence = {
        "kind": "brickagain.phase3c_pack_evidence",
        "generation": phase3c.GENERATION,
        "pack_digest": "d" * 64,
        "files_digest": "e" * 64,
        "data_digest": "f" * 64,
        "n_files": 1,
        "how_to_recheck": "-",
        "pack_manifest": {"schema_version": 1, "kind": "x",
                          "files": {"a.py": {"sha256": "0" * 64,
                                             "bytes": 1,
                                             "snapshot_name": "a.py"}},
                          "files_digest": "e" * 64,
                          "data_digest": "f" * 64,
                          "pack_digest": "d" * 64},
    }
    evidence["evidence_digest"] = digest_obj(
        {k: v for k, v in evidence.items() if k != "evidence_digest"})
    problems = phase3c.pack_evidence_problems(evidence,
                                              pack_digest="d" * 64)
    assert any("files_digest does not cover" in p for p in problems), problems
    assert any("recomputes to" in p for p in problems), problems


def test_the_archive_backs_the_plan_it_is_asked_about(plan, tmp_path):
    """``archive_problems`` re-digests the membership and the audit."""
    if _ARCHIVE_FOR_TESTS is None:
        pytest.skip(f"{ARTIFACT_ONLY} there is no archive to check")
    assert phase3c.archive_problems(plan, _ARCHIVE_FOR_TESTS,
                                    root=ROOT) == []

    bent = tmp_path / "bent_archive"
    _copy_tree(_ARCHIVE_FOR_TESTS, bent)
    audit = bent / Path(phase3c.AUDIT_PATH).name
    body = json.loads(audit.read_text())
    body["note_added_afterwards"] = "widened"
    audit.unlink()
    audit.write_text(json.dumps(body, indent=2) + "\n")
    problems = phase3c.archive_problems(plan, bent, root=ROOT)
    assert any("audit_digest" in p and "contents digest to" in p
               for p in problems), problems

    # And the other direction: a document whose self-digest is recomputed
    # after the edit, so it is internally consistent and is still not the
    # one the plan was built against.
    body["audit_digest"] = digest_obj(
        {k: v for k, v in body.items() if k != "audit_digest"})
    audit.unlink()
    audit.write_text(json.dumps(body, indent=2) + "\n")
    problems = phase3c.archive_problems(plan, bent, root=ROOT)
    assert any("the plan's audit_digest is" in p for p in problems), problems

    (bent / Path(phase3c.MEMBERSHIP_PATH).name).unlink()
    assert any("case_membership.json is not in" in p
               for p in phase3c.archive_problems(plan, bent, root=ROOT))


# ---------------------------------------------------------------------------
# the documents' commands, parsed
# ---------------------------------------------------------------------------

def _commands_in(text: str, script: str) -> list[list[str]]:
    """Every ``59_phase3c.py`` invocation in a markdown document, as argv.

    Line continuations are joined, placeholders are left as the literal
    words they are -- argparse does not care what a value means -- and shell
    constructs that are not this script are skipped.
    """
    import shlex

    joined, buffer = [], ""
    for line in text.splitlines():
        stripped = line.strip()
        if buffer:
            buffer += " " + stripped
        elif script in stripped:
            buffer = stripped
        else:
            continue
        if buffer.endswith("\\"):
            buffer = buffer[:-1]
            continue
        joined.append(buffer)
        buffer = ""

    # Shell loop variables, given a value argparse can typecheck. The
    # documents run these inside ``for step in 0 1 2 3 4 5``, so 0 is a
    # value the loop actually takes rather than one invented here.
    shell_values = {"$step": "0", "${step}": "0"}

    out = []
    for command in joined:
        try:
            words = shlex.split(command)
        except ValueError:
            continue
        if script not in words:
            continue
        argv = [shell_values.get(w, w)
                for w in words[words.index(script) + 1:]]
        out.append(argv)
    return out


DOCUMENTS: tuple[str, ...] = ("PHASE3C_NODE_RUN.md",)


def test_every_documented_command_parses(published, plan):
    """The commands in the node document and in ``reproduce.md`` are real.

    The previous reproduce document told the reader to materialise with
    ``--out``, which is not a flag, and to run the node against the private
    archive path a pack may not carry, with no ``--authorization`` anywhere.
    None of that is catchable by reading; it is catchable by parsing.
    """
    cli = _cli()
    parser = cli.build_parser()

    texts = {}
    for name in DOCUMENTS:
        path = ROOT / name
        if path.is_file():
            texts[name] = path.read_text(encoding="utf-8")

    _copy, _seal, written = published
    texts[phase3c_report.REPRODUCE_NAME] = \
        written["reproduce"].read_text(encoding="utf-8")

    seen = 0
    for name, text in texts.items():
        for argv in _commands_in(text, "scripts/59_phase3c.py"):
            seen += 1
            args = parser.parse_args(argv)          # raises SystemExit if not
            chosen = [m for m in cli.MODES if getattr(args, m)]
            assert len(chosen) == 1, (name, argv)
            # And every path it names resolves against a real definition.
            if args.plan:
                assert args.plan in (phase3c.PLAN_PATH,
                                     phase3c.NODE_PLAN_PATH), (name, argv)
    assert seen >= 8, seen


def test_the_node_document_points_the_node_at_the_staged_plan():
    path = ROOT / "PHASE3C_NODE_RUN.md"
    if not path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {path.name} is not published")
    text = path.read_text(encoding="utf-8")
    for argv in _commands_in(text, "scripts/59_phase3c.py"):
        if "--run" in argv or "--seal" in argv:
            assert phase3c.NODE_PLAN_PATH in argv, argv
            assert "--authorization" in argv, argv
    assert phase3c.NODE_PLAN_PATH in text
    assert phase3c.GENERATION in text
    # The archive never travels, so no *node* command names it. The Mac
    # section legitimately does: the archive is the record, and the staged
    # copy exists only so a pack can carry one.
    for argv in _commands_in(text, "scripts/59_phase3c.py"):
        if "--run" in argv or "--seal" in argv:
            assert phase3c.PLAN_PATH not in argv, argv
        if "--verify" in argv or "--score" in argv or "--report" in argv:
            assert phase3c.PLAN_PATH in argv, argv
            assert "--authorization" in argv, argv


#: The documents that carry a history of frozen generations, and therefore
#: carry digests belonging to generations that are no longer current.
HISTORY_DOCUMENTS: tuple[str, ...] = ("PROJECT_STATUS.md",
                                      "RESEARCH_RECORD.md")


def test_no_document_pairs_a_plan_file_with_another_generations_digest():
    """A history that quotes the current generation's digest is not history.

    The concrete failure this catches: a blanket search-and-replace over the
    documents, run to refresh the *current* generation's digests, rewrote
    the ones inside a table describing gen01 -- leaving a row that named
    gen01's file beside a later generation's ``plan_digest``. Nothing in the
    prose was wrong; the number was.

    So every markdown row that names a Phase 3C plan file and a
    ``plan_digest`` in the same row is checked against that file on disk.
    """
    import re

    if not any((ROOT / name).is_file() for name in HISTORY_DOCUMENTS):
        pytest.skip(f"{ARTIFACT_ONLY} the history documents are not "
                    "published")
    row = re.compile(
        r"\|\s*`(?P<path>[\w/]*phase3c[\w/]*plan\.json)`\s*\|"
        r"[^|]*\|\s*`plan_digest (?P<digest>[0-9a-f]{16,64})")
    checked = 0
    for name in HISTORY_DOCUMENTS:
        doc = ROOT / name
        if not doc.is_file():
            continue
        for match in row.finditer(doc.read_text(encoding="utf-8")):
            path = ROOT / match.group("path")
            if not path.is_file():
                continue
            stored = json.loads(path.read_text())["plan_digest"]
            quoted = match.group("digest")
            assert stored.startswith(quoted), (
                f"{name} pairs {match.group('path')} with plan_digest "
                f"{quoted}..., and that file's plan_digest is "
                f"{stored[:len(quoted)]}...")
            checked += 1
    assert checked, "no document row paired a plan file with a plan_digest"


VERIFICATION_DIR = "data/reports/phase3c_verification"


def test_the_verification_log_is_a_real_directory_with_a_real_index():
    """A report that cites its raw output has to cite something that exists.

    The round-59 report said the raw outputs were "kept in ``verify_final/``".
    That was a session scratch directory: real on the machine that wrote it,
    absent from the repository, and therefore not evidence anybody could
    check. The logs live here now, and this test is what keeps the citation
    honest -- the index must name files that exist, and every file it names
    must record the command that produced it and that command's exit code.
    """
    directory = ROOT / VERIFICATION_DIR
    if not directory.is_dir():
        pytest.skip(f"{ARTIFACT_ONLY} {VERIFICATION_DIR} is not published")
    index_path = directory / "index.json"
    assert index_path.is_file(), f"{VERIFICATION_DIR} has no index.json"
    index = json.loads(index_path.read_text())
    assert index["kind"] == "brickagain.verification_log"
    assert index["runs"], "the index records no runs"

    named = set()
    for run in index["runs"]:
        for field in ("name", "command", "exit_code", "log", "summary"):
            assert field in run, (run.get("name"), field)
        log = directory / run["log"]
        assert log.is_file(), run["log"]
        named.add(run["log"])
        assert isinstance(run["exit_code"], int)
        # The summary line the index quotes must be in the log it names.
        assert run["summary"].strip() in log.read_text(encoding="utf-8"), \
            run["name"]
    on_disk = {p.name for p in directory.iterdir()
               if p.is_file() and p.name != "index.json"}
    assert on_disk == named, sorted(on_disk ^ named)


def test_the_verification_log_carries_no_personal_path():
    """The logs are redacted, and the needles are built rather than written.

    Spelled out, the two strings below would themselves be a personal path
    and an account name in a published file, which the snapshot audit
    refuses -- correctly, since it cannot tell a test's needle from the
    thing it looks for.
    """
    home = str(Path.home())
    needles = ("/" + "Users" + "/", Path(home).name, home)
    directory = ROOT / VERIFICATION_DIR
    if not directory.is_dir():
        pytest.skip(f"{ARTIFACT_ONLY} {VERIFICATION_DIR} is not published")
    for path in sorted(directory.iterdir()):
        text = path.read_text(encoding="utf-8", errors="replace")
        for needle in needles:
            assert needle not in text, (path.name, needle[:6])


def _known_digests() -> dict:
    """Every 64-hex value any archived artefact declares, by 16-hex prefix."""
    import re

    out: dict = {}
    roots = [ROOT / "data/phase3c/frozen", ROOT / "gpu_plans",
             ROOT / "data/reports/60_visual_stress"]
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            try:
                body = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            stack = [body]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    for value in node.values():
                        if isinstance(value, str) and re.fullmatch(
                                r"[0-9a-f]{64}", value):
                            out.setdefault(value[:16], set()).add(value)
                        elif isinstance(value, (dict, list)):
                            stack.append(value)
                elif isinstance(node, list):
                    stack.extend(v for v in node
                                 if isinstance(v, (dict, list)))
    return out


def test_no_document_quotes_a_digest_that_only_starts_like_a_real_one():
    """The splice this catches, exactly.

    A search-and-replace written to refresh 16-character *prefixes* was run
    over documents that also carry full 64-character digests. Where a full
    digest began with a prefix being replaced, its first sixteen characters
    were swapped and the remaining forty-eight left alone -- producing a
    value that looks like a digest, starts like a real one, and is nothing.

    So: any full digest in these documents whose first sixteen characters
    match a real artefact's must *be* that artefact's.
    """
    import re

    known = _known_digests()
    if not known:
        pytest.skip(f"{ARTIFACT_ONLY} no archived artefacts to check against")
    spliced: list[str] = []
    for name in HISTORY_DOCUMENTS + ("PORTFOLIO.md", "PHASE3C_NODE_RUN.md"):
        doc = ROOT / name
        if not doc.is_file():
            continue
        for value in set(re.findall(r"\b[0-9a-f]{64}\b",
                                    doc.read_text(encoding="utf-8"))):
            matches = known.get(value[:16])
            if matches and value not in matches:
                spliced.append(f"{name}: {value} starts like "
                               f"{sorted(matches)[0]} and is not it")
    assert spliced == [], spliced


def test_the_documents_do_not_claim_the_grant_is_staged():
    """``--stage`` copies the plan. Saying otherwise would be a false map."""
    for name in DOCUMENTS + ("PORTFOLIO.md",):
        path = ROOT / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert "把封存裡的 plan 與 grant 複製" not in text, name
    from src.training import pack as pack_module

    included = [e["path"] for e in pack_module.manifest(ROOT)["include"]]
    assert phase3c.AUTHORIZATION_PATH not in included
    assert not any(p.endswith("execution_authorization.json")
                   for p in included)


def test_the_materialiser_refuses_without_the_flag(capsys):
    cli = _cli()
    assert cli.main(["--materialize", "--out-dir", "/dev/null"]) == 2
    assert "open-test-after-codex-approval" in capsys.readouterr().err


def test_scoring_refuses_off_the_mac():
    cli = _cli()
    assert cli._mac_guard("score", system="Linux")
    assert cli._mac_guard("score", system="Darwin") == []


def test_verify_refuses_without_a_carried_digest(capsys):
    cli = _cli()
    assert cli.main(["--verify", "--out-dir", "/tmp"]) == 2
    assert "carried-seal-digest" in capsys.readouterr().err


def test_exactly_one_mode(capsys):
    cli = _cli()
    assert cli.main(["--contract", "--score"]) == 2
    assert cli.main([]) == 2


@pytest.mark.skipif(platform.system() != "Darwin",
                    reason="the contract mode is machine-independent but the "
                           "digest it prints is checked against this tree")
def test_the_contract_mode_prints_the_frozen_digest(capsys):
    cli = _cli()
    assert cli.main(["--contract"]) == 0
    assert phase3c.contract_digest(ROOT) in capsys.readouterr().out


# ---------------------------------------------------------------------------

def _copy_tree(src: Path, dst: Path) -> None:
    import shutil

    shutil.copytree(src, dst)


def _write_rows(path: Path, rows) -> None:
    from src.eval.acceptance import canonical_json

    path.write_text("".join(canonical_json(r) + "\n" for r in rows),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# The formal run path: which weights load, which gate warms, and what a row
# is allowed to claim it ran on
# ---------------------------------------------------------------------------
#
# gen04's node run reached the node, passed every carried digest, every
# pinned environment field and the whole preflight, wrote its execution
# manifest, and then died on ``KeyError: 'A' is not one of
# ['B', 'C', 'D', 'E']``. The runner had handed Phase 3C's arm names to
# Phase 2's registry. Only arm A is absent from that registry, so only arm A
# failed loudly: arm B would have resolved to the *published* model with *no*
# gate, and arm C to the fine-tuned model with *no* gate, while the measured
# cells went on using Phase 3C's gates. The run would have finished and
# answered a different question.
#
# Nothing caught it because the only ``--run`` test above refuses earlier, on
# a deliberately incomplete directory, so the line that builds the model had
# never been executed by any test.
#
# These tests execute it. They cross the machine guard, the environment
# comparison, the node preflight, the adapter check, the carried digests, the
# authorization and the execution manifest -- every one running its real
# comparison logic -- and reach the loader, the warm-up and the cells. What is
# injected is only what a Mac cannot have: the node's own probe, a CUDA
# device, the dependency cache reading, and the weights. There is no
# production flag that skips a guard, and none is added to make this run.

#: A decode that satisfies the token arithmetic: one brick, ``10n + 1``
#: tokens, a termination the contract accepts. Deliberately not plausible
#: model output -- what is being asserted is which code path produced it.
class _FakeRaw:
    text = "1x2 (0,0,0)\n"
    n_tokens = phase3c.TOKENS_PER_BRICK + phase3c.EOS_TOKENS
    seconds = 0.001
    termination = "normal_eos"
    truncated = False


# There is deliberately no stand-in gate here any more.
#
# There used to be one: a ``_FakeGate`` carrying ``accepted``, ``inventory``
# and a ``counters()`` method, substituted for both gated entry points so the
# run path could be exercised without a model. It implemented everything
# ``gate_ledger`` asks for, which made it a *superset* of the real
# ``InventoryGate`` -- and the real one has no ``counters()``. So the suite
# proved the ledger worked against an object built to satisfy it, gen08 was
# frozen, the node ran, and step 1 died on ``AttributeError`` after step 0
# had already written 320 cells.
#
# The substitution now happens one layer lower, at the interface's
# ``generate_raw``, so ``generate_raw_with_inventory`` and
# ``generate_raw_with_placement`` run for real and build the real gate
# classes. Anything reintroducing a hand-written gate object here would put
# that hole straight back.


def _good_probe(**overrides) -> dict:
    """A reading of the node the contract pins. Judged, never trusted."""
    base = {
        "os_system": "Linux", "wsl2": True,
        "wsl_evidence": "kernel release names a Microsoft build",
        "torch_version": "2.13.0+cu130", "torch_cuda_build": "13.0",
        "cuda_available": True, "device_count": 1,
        "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
        "vram_total_gb": 15.9, "system_ram_gb": 31.2,
        "offline_env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "HF_HUB_DISABLE_TELEMETRY": "1"},
        "alloc_env": {"PYTORCH_ALLOC_CONF": "expandable_segments:True",
                      "PYTORCH_CUDA_ALLOC_CONF": None},
        "cublas_workspace_config": ":4096:8", "allocator_backend": "native",
    }
    base.update(overrides)
    return base


#: Synthetic dependency evidence. Nothing is read from this machine: the
#: node's answer is bound to *these* bytes, so the test has to own them.
_DEP_EVIDENCE = {
    "schema_version": 1, "kind": "longrun_dependency_preflight",
    "network_used": False, "tensors_loaded": False,
    "device_initialised": False,
    "repositories": [
        {"repo_id": "Vendor/Tok", "revision": "a" * 40,
         "files": [{"name": "tokenizer.json", "bytes": 10,
                    "sha256": "1" * 64}]},
    ],
    "instruction_pool": {"path": "data/processed/instruct_inv_train.jsonl",
                         "sha256": "5" * 64},
}


def _finetuned_info(ckpt) -> dict:
    """What ``load_finetuned`` reports when it has actually run."""
    from src.training.lora import LOAD_ORDER

    return {
        "base_model": phase3c.SETTINGS.base_model,
        "base_revision": phase3c.SETTINGS.base_revision,
        "published_adapter": phase3c.SETTINGS.published_adapter,
        "published_adapter_revision":
            phase3c.SETTINGS.published_adapter_revision,
        "merge_changed_weights": True,
        "load_order": list(LOAD_ORDER),
        "local_adapter": str(ckpt),
    }


def _published_info() -> dict:
    """What ``load_merged_brickgpt`` reports: no local adapter, no merge key
    of its own beyond the merge itself. This is the shape gen04's arm B would
    have carried."""
    return {
        "base_model": phase3c.SETTINGS.base_model,
        "base_revision": phase3c.SETTINGS.base_revision,
        "published_adapter": phase3c.SETTINGS.published_adapter,
        "published_adapter_revision":
            phase3c.SETTINGS.published_adapter_revision,
        "merge_changed_weights": True,
        "load_order": ["base", "published_adapter", "merge"],
    }


#: "Leave the real warm-up in place." A sentinel rather than ``None``,
#: because ``None`` is itself one of the malformed records under test.
_REAL = object()


class _RunPath:
    """One prepared node run, with every guard live and the weights injected.

    ``calls`` records what the run actually did: which loader ran, and which
    decode path each generation went through. Warm-up generations are the
    ones at :data:`phase3c.WARMUP` seeds; the cells are at the plan's.
    """

    def __init__(self, calls, out_dir, adapter, member):
        self.calls = calls
        self.out_dir = out_dir
        self.adapter = adapter
        self.member = member

    def loaders_used(self):
        return [c[1] for c in self.calls if c[0] == "load"]

    def decodes(self, *, warmup: bool):
        seeds = set(phase3c.WARMUP["seeds"])
        return [c for c in self.calls
                if c[0] == "decode" and ((c[2] in seeds) == warmup)]

    def gates(self, *, warmup: bool):
        return sorted({c[1] for c in self.decodes(warmup=warmup)})

    def gate_classes(self, *, warmup: bool):
        """The class name of the object each decode actually built.

        Recorded from ``type(gate)`` inside the interface, so it is what the
        shipped gated entry points constructed and not what the arm spec
        says they should have.
        """
        return sorted({c[3] for c in self.decodes(warmup=warmup)})


@pytest.fixture()
def run_path(plan, tmp_path, monkeypatch):
    """A factory: prepare a real node run, then execute one step of it."""
    from src.eval import acceptance as acc
    from src.training import gpu_node, pack as pack_module
    from src.training.longrun import dependency_digest
    from src.training.session import sha256_file, write_once_json

    def make(*, step: int = 0, info=None, loader="finetuned",
             adapter_bytes=None, arms=None, warmup=_REAL):
        """``warmup`` substitutes what :func:`phase3c.warm_up` returns.

        A dict is returned in its place, an exception instance is raised
        from it, and :data:`_REAL` leaves the real one running. Everything
        else in the run -- the guards, the loader, the cells -- is untouched,
        because what is under test is what ``mode_run`` does with a warm-up
        record it cannot trust.
        """
        calls: list = []

        # -- the weights ------------------------------------------------
        adapter = tmp_path / "adapter"
        adapter.mkdir(exist_ok=True)
        contents = {"adapter_model.safetensors": b"not real tensors\n",
                    "brickagain_manifest.json": b'{"load_order": []}\n',
                    "adapter_config.json": b'{"r": 32}\n'}
        contents.update(adapter_bytes or {})
        for stem, blob in contents.items():
            (adapter / stem).write_bytes(blob)
        digests = {stem: sha256_file(adapter / stem) for stem in
                   ("adapter_model.safetensors", "brickagain_manifest.json",
                    "adapter_config.json")}
        # The real adapter check runs; the digests it checks against are the
        # test's own, because no test can produce files hashing to the
        # frozen ones.
        monkeypatch.setattr(acc, "FINAL_ADAPTER_SHA256", digests)
        monkeypatch.setattr(phase3c, "FINAL_ADAPTER_SHA256", digests)
        if arms is not None:
            monkeypatch.setattr(phase3c, "ARMS", arms)

        # -- the plan, re-digested under those weights ------------------
        body = dict(plan)
        body["final_model"] = phase3c.final_model_document()
        body["contract_digest"] = phase3c.contract_digest(ROOT)
        body["plan_digest"] = phase3c.plan_digest(body)
        staged = tmp_path / "staged_plan.json"
        write_once_json(staged, body)

        # -- a real pack, verified for real -----------------------------
        src = tmp_path / "src_tree"
        for rel, text in {"requirements.txt": "torch\n",
                          "src/__init__.py": "",
                          "src/training/__init__.py": ""}.items():
            p = src / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        for rel in pack_module.REQUIRED_DATA:
            p = src / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('{"sample_id":"x"}\n', encoding="utf-8")
        pack_dir = tmp_path / "pack"
        pack_module.build(pack_dir, root=src)
        table, problems = pack_module.read_manifest(pack_dir)
        assert not problems, problems
        pack_digest = table["pack_digest"]
        evidence = phase3c.build_pack_evidence(table, pack_digest=pack_digest)
        dep_digest = dependency_digest(_DEP_EVIDENCE)

        grant = phase3c.build_authorization(
            plan=body, pack_digest=pack_digest,
            dependency_digest=dep_digest,
            pack_evidence_digest=evidence["evidence_digest"],
            authorized_at=STAMP, root=ROOT)
        grant_path = tmp_path / "carried_grant.json"
        write_once_json(grant_path, grant)

        # -- what a Mac cannot be ---------------------------------------
        monkeypatch.setattr(gpu_node, "probe", lambda **kw: _good_probe())
        monkeypatch.setattr(gpu_node, "_dependency_check",
                            lambda checker=None: {
                                "check": {"passed": True,
                                          "detail": "injected"},
                                "evidence": _DEP_EVIDENCE})
        monkeypatch.setattr(acc, "resolve_device", lambda torch_mod: "cuda")
        monkeypatch.setattr(phase3c, "_platform_python", lambda: "3.12.3")
        import torch
        monkeypatch.setattr(torch, "__version__", "2.13.0+cu130")

        # -- the weights, and every decode door -------------------------
        from src.constraints.inventory_decode import InventoryGate
        from src.constraints.placement_decode import InventoryPlacementGate
        from src.generation.brickgpt import MAX_DIM, Slots, WORLD

        class FakeIface:
            """A model that does not decode, wired below the gates.

            ``slots`` is here because the real gated entry points build
            ``gate_cls(gpt.slots, inventory)`` from it. Substituting at this
            level -- rather than replacing the entry points -- is what puts
            the real ``InventoryGate`` and ``InventoryPlacementGate`` in
            front of ``gate_ledger``, which is where gen08 failed.
            """

            slots = Slots(dims=list(range(100, 100 + MAX_DIM)),
                          posns=list(range(200, 200 + WORLD)), literal_x=1,
                          literal_open=2, literal_comma=3, literal_close=4,
                          eos=5)

            def generate_raw(self, caption, inventory=None, gate=None, **kw):
                # Recorded from the object that actually arrived, so the
                # call log cannot say a gate ran that did not.
                if gate is None:
                    seen = phase3c.GATE_NONE
                elif type(gate) is InventoryPlacementGate:
                    seen = phase3c.GATE_INVENTORY_PLACEMENT
                elif type(gate) is InventoryGate:
                    seen = phase3c.GATE_INVENTORY
                else:
                    seen = f"unexpected:{type(gate).__name__}"
                calls.append(("decode", seen, kw.get("seed"),
                              type(gate).__name__ if gate is not None
                              else None))
                return _FakeRaw()

        def finetuned(ckpt, **kw):
            calls.append(("load", "finetuned"))
            return "MODEL", (info if info is not None
                             else _finetuned_info(ckpt))

        def merged(*a, **kw):
            # ``*a`` because this stands in for the finetuned slot in the
            # substitution test, where it is called with the adapter path.
            calls.append(("load", "merged"))
            return "PUBLISHED", (info if info is not None
                                 else _published_info())

        monkeypatch.setattr(acc, "default_loaders", lambda: {
            "tokenizer": lambda *a, **k: "TOK",
            "merged": merged,
            "finetuned": merged if loader == "merged" else finetuned,
            "interface": lambda m, t, device=None: FakeIface(),
        })

        # ``generate_raw_with_inventory`` and ``generate_raw_with_placement``
        # are deliberately NOT substituted. They run, they build the real
        # gate for the arm, and that real gate is what reaches the warm-up
        # check and every measured row's ledger.

        if warmup is not _REAL:
            def substitute(_interface, _name):
                calls.append(("warm_up", "substituted"))
                if isinstance(warmup, BaseException):
                    raise warmup
                return warmup
            monkeypatch.setattr(phase3c, "warm_up", substitute)

        out_dir = tmp_path / f"run_step{step}"
        code = _cli().main([
            "--run", "--step", str(step),
            "--plan", str(staged), "--authorization", str(grant_path),
            "--out-dir", str(out_dir), "--pack-dir", str(pack_dir),
            "--expected-pack-digest", pack_digest,
            "--expected-dependency-digest", dep_digest,
            "--adapter-dir", str(adapter),
        ])
        return code, _RunPath(calls, out_dir, adapter,
                              out_dir / phase3c.samples_member(step))

    return make


#: ``(step, arm, the gate that arm's spec names)``. One step per arm, taken
#: from the frozen schedule rather than restated.
_STEP_ARM_GATE = tuple(
    (i, phase3c.step(i)[1], phase3c.ARMS[phase3c.step(i)[1]].gate)
    for i in (0, 1, 2))


class TestTheRunPathReachesTheLoader:
    """The line gen04 died on, executed."""

    def test_arm_a_no_longer_dies_in_phase_2s_registry(self, run_path):
        """The exact gen04 failure: 'A' is not one of ['B','C','D','E']."""
        code, run = run_path(step=0)
        assert code == 0
        assert run.member.is_file()
        assert len(phase3c.read_rows(run.member)) == 320

    @pytest.mark.parametrize("step,arm,gate", _STEP_ARM_GATE)
    def test_every_arm_loads_final_h2_through_the_finetuned_loader(
            self, run_path, step, arm, gate):
        """A, B and C are one model and one adapter. Only the gate differs."""
        code, run = run_path(step=step)
        assert code == 0, arm
        assert run.loaders_used() == ["finetuned"], arm
        assert "merged" not in run.loaders_used(), arm

    @pytest.mark.parametrize("step,arm,gate", _STEP_ARM_GATE)
    def test_the_warm_up_runs_this_arms_own_gate(self, run_path, step, arm,
                                                 gate):
        """gen04 warmed arm C ungated and then measured it gated."""
        code, run = run_path(step=step)
        assert code == 0, arm
        assert run.gates(warmup=True) == [gate], arm
        assert len(run.decodes(warmup=True)) == len(phase3c.WARMUP["seeds"])

    @pytest.mark.parametrize("step,arm,gate", _STEP_ARM_GATE)
    def test_the_cells_use_the_same_gate_the_warm_up_used(
            self, run_path, step, arm, gate):
        code, run = run_path(step=step)
        assert code == 0, arm
        assert run.gates(warmup=False) == run.gates(warmup=True) == [gate]
        recorded = {r["gate"]["gate"] for r in phase3c.read_rows(run.member)}
        assert recorded == {gate}, arm

    @pytest.mark.parametrize("step,arm,gate", _STEP_ARM_GATE)
    def test_a_row_names_the_weights_that_actually_loaded(
            self, run_path, step, arm, gate):
        code, run = run_path(step=step)
        assert code == 0, arm
        rows = phase3c.read_rows(run.member)
        expected = phase3c.observed_model_identity(
            _finetuned_info(run.adapter), run.adapter)
        assert expected == phase3c.model_identity()
        for row in rows:
            assert row["model"] == expected, arm
            assert row["model"]["model"] == phase3c.FINAL_MODEL


class TestTheRunPathFailsClosed:
    """Every substitution gen04 could have made, refused before a cell."""

    def _refused(self, code, run):
        assert code == 2
        # Not "empty": absent. A refusal that left a member behind would be a
        # half-published step.
        assert not run.member.exists()

    def test_arm_b_cannot_silently_load_the_published_model(self, run_path):
        """The substitution gen04's runner would have made for arm B."""
        code, run = run_path(step=1, loader="merged",
                             info=_published_info())
        self._refused(code, run)
        assert run.loaders_used() == ["merged"]

    def test_a_load_record_without_a_local_adapter_is_refused(self, run_path):
        code, run = run_path(step=0, info=_published_info())
        self._refused(code, run)

    def test_a_load_of_some_other_adapter_directory_is_refused(
            self, run_path, tmp_path):
        info = _finetuned_info(tmp_path / "somewhere_else")
        code, run = run_path(step=0, info=info)
        self._refused(code, run)

    def test_a_delta_that_did_not_land_on_the_merged_adapter_is_refused(
            self, run_path, tmp_path):
        info = _finetuned_info(tmp_path / "adapter")
        info["load_order"] = ["base", "local_adapter"]
        code, run = run_path(step=0, info=info)
        self._refused(code, run)

    def test_a_merge_that_changed_no_weight_is_refused(self, run_path,
                                                       tmp_path):
        info = _finetuned_info(tmp_path / "adapter")
        info["merge_changed_weights"] = False
        code, run = run_path(step=0, info=info)
        self._refused(code, run)

    def test_a_different_base_revision_is_refused(self, run_path, tmp_path):
        info = _finetuned_info(tmp_path / "adapter")
        info["base_revision"] = "0" * 40
        code, run = run_path(step=0, info=info)
        self._refused(code, run)

    def test_an_adapter_whose_bytes_changed_is_refused(self, run_path,
                                                       monkeypatch):
        """The digests are checked against the directory, not against a
        value copied out of the contract."""
        code, run = run_path(step=0)
        assert code == 0
        # Same harness, weights rewritten after the contract was pinned.
        (run.adapter / "adapter_model.safetensors").write_bytes(b"swapped\n")
        problems = phase3c.loaded_identity_problems(
            _finetuned_info(run.adapter), run.adapter)
        assert any("adapter_model.safetensors" in p for p in problems)

    def test_a_drifted_gate_is_refused_before_anything_loads(self, run_path):
        """An arm table that no longer matches the plan stops the step."""
        drifted = dict(phase3c.ARMS)
        drifted["C"] = phase3c.Arm(
            "C", phase3c.FINAL_MODEL, phase3c.LOADER_FINAL,
            phase3c.PROMPT_FORM, phase3c.GATE_NONE, None, "drifted")
        code, run = run_path(step=2, arms=drifted)
        self._refused(code, run)
        assert run.loaders_used() == []


def test_the_cli_has_no_flag_that_skips_a_guard():
    """There is no ``--skip-preflight``, and adding one must fail here."""
    parser = _cli().build_parser()
    options = {s for action in parser._actions for s in action.option_strings}
    for forbidden in ("--skip-preflight", "--skip-manifest", "--skip-guards",
                      "--no-preflight", "--force", "--skip-authorization",
                      "--skip-adapter", "--insecure"):
        assert forbidden not in options, forbidden
    assert not [o for o in options if "skip" in o or "bypass" in o]


# ---------------------------------------------------------------------------
# The staged plan travels by pointer, so a generation bump does not edit
# src/training/pack.py
# ---------------------------------------------------------------------------
#
# The cycle these tests close. Until gen05 the current generation's staged
# plan was a literal inside ``PACK_ALLOW``, so a Phase 3C bump edited
# ``src/training/pack.py`` -- which is inside the V1 visual run's import
# closure, whose frozen ``source_manifest`` pins that file's SHA-256. Every
# Phase 3C bump therefore invalidated V1's stored runs; and re-freezing V1 to
# repair that edits ``src/eval/visual_stress.py``, which the Phase 3C pack
# carries, which invalidates the Phase 3C pack digest and the authorization
# bound to it. Each repair broke the other, and gen05 was frozen inside the
# cycle without ever running.
#
# The generation is data now. What has to stay true is that it is *only*
# data: that no generation string is left in ``pack.py``, that the pointer
# selects exactly one plan, and that every way of getting the pointer wrong
# refuses rather than shipping a superseded plan.

def _pointer(tmp_path, **overrides):
    """A staged tree: one plan, one pointer, and whatever is overridden."""
    from src.training import pack as pack_module
    from src.eval.acceptance import canonical_json

    root = tmp_path
    (root / "gpu_plans").mkdir(parents=True, exist_ok=True)
    plan_rel = overrides.pop("plan_rel", "gpu_plans/phase3c_gen09_plan.json")
    (root / plan_rel).write_text('{"kind": "test"}\n', encoding="utf-8")
    from src.training.session import sha256_file

    body = {
        "kind": pack_module.PHASE3C_POINTER_KIND,
        "generation": "gen09",
        "staged_plan": plan_rel,
        "plan_sha256": sha256_file(root / plan_rel),
    }
    body.update(overrides)
    (root / pack_module.PHASE3C_STAGED_POINTER).write_text(
        canonical_json(body) + "\n", encoding="utf-8")
    return root, plan_rel


class TestTheStagedPlanTravelsByPointer:
    def test_pack_py_carries_no_generation_string_at_all(self):
        """The literal that used to move. If it comes back, so does the cycle."""
        import re

        source = (ROOT / "src" / "training" / "pack.py").read_text(
            encoding="utf-8")
        stale = re.findall(r"phase3c_gen\d+_plan\.json", source)
        assert stale == [], (
            "a per-generation Phase 3C plan name is back in pack.py; bumping "
            f"the generation would edit this file again: {stale}")

    def test_the_allowlist_names_no_phase3c_plan(self):
        from src.training import pack as pack_module

        assert not [p for p in pack_module.PACK_ALLOW
                    if "phase3c" in p and "plan" in p]

    def test_a_generation_bump_changes_the_pointer_and_nothing_in_pack_py(
            self, tmp_path, monkeypatch):
        """The whole point, asserted directly: bump, and pack.py is untouched."""
        from src.training import pack as pack_module
        from src.training.session import sha256_file

        before = sha256_file(ROOT / "src" / "training" / "pack.py")
        root, _ = _pointer(tmp_path)
        resolved, problems = pack_module.staged_phase3c_plan(root)
        assert problems == [] and resolved.endswith("gen09_plan.json")

        # "Bump": a different generation, staged and pointed at.
        (root / "gpu_plans" / "phase3c_gen10_plan.json").write_text(
            '{"kind": "test2"}\n', encoding="utf-8")
        _pointer(tmp_path, plan_rel="gpu_plans/phase3c_gen10_plan.json",
                 generation="gen10")
        resolved, problems = pack_module.staged_phase3c_plan(root)
        assert problems == [], problems
        assert resolved == "gpu_plans/phase3c_gen10_plan.json"
        assert sha256_file(ROOT / "src" / "training" / "pack.py") == before

    def test_only_the_pointed_plan_is_included(self, tmp_path):
        """Superseded staged copies stay on disk and stay out of the pack."""
        from src.training import pack as pack_module

        root, plan_rel = _pointer(tmp_path)
        for other in ("gpu_plans/phase3c_gen07_plan.json",
                      "gpu_plans/phase3c_gen08_plan.json"):
            (root / other).write_text("{}\n", encoding="utf-8")
        staged, _ = pack_module.staged_phase3c_plan(root)
        assert pack_module.classify(plan_rel, staged_plan=staged)[0] == \
            "include"
        for other in ("gpu_plans/phase3c_gen07_plan.json",
                      "gpu_plans/phase3c_gen08_plan.json"):
            verdict, reason = pack_module.classify(other, staged_plan=staged)
            assert verdict == "exclude", (other, reason)
            assert "does not name" in reason

    def test_with_no_pointer_resolved_no_plan_is_included(self):
        from src.training import pack as pack_module

        verdict, reason = pack_module.classify(
            "gpu_plans/phase3c_gen09_plan.json", staged_plan=None)
        assert verdict == "exclude", reason

    @pytest.mark.parametrize("overrides,needle", [
        ({"kind": "something.else"}, "kind"),
        ({"generation": "gen10"}, "names"),
        ({"plan_sha256": "0" * 64}, "not the same document"),
        ({"plan_sha256": "not-a-digest"}, "not a SHA-256"),
        ({"staged_plan": "gpu_plans/core_eval_plan.json"}, "names"),
        ({"generation": ""}, "no generation"),
    ])
    def test_a_pointer_that_does_not_hold_resolves_to_nothing(
            self, tmp_path, overrides, needle):
        from src.training import pack as pack_module

        root, _ = _pointer(tmp_path, **overrides)
        resolved, problems = pack_module.staged_phase3c_plan(root)
        assert resolved is None
        assert any(needle in p for p in problems), (problems, needle)

    def test_a_missing_pointer_refuses(self, tmp_path):
        from src.training import pack as pack_module

        (tmp_path / "gpu_plans").mkdir()
        resolved, problems = pack_module.staged_phase3c_plan(tmp_path)
        assert resolved is None
        assert any("is not here" in p for p in problems), problems

    def test_a_pointer_naming_a_plan_that_is_not_here_refuses(self, tmp_path):
        from src.training import pack as pack_module

        root, plan_rel = _pointer(tmp_path)
        (root / plan_rel).unlink()
        resolved, problems = pack_module.staged_phase3c_plan(root)
        assert resolved is None
        assert any("not here" in p for p in problems), problems

    def test_unreadable_json_refuses(self, tmp_path):
        from src.training import pack as pack_module

        root, _ = _pointer(tmp_path)
        (root / pack_module.PHASE3C_STAGED_POINTER).write_text(
            "{not json", encoding="utf-8")
        resolved, problems = pack_module.staged_phase3c_plan(root)
        assert resolved is None
        assert any("readable JSON" in p for p in problems), problems


class TestThisTreesPointerIsThisGeneration:
    def test_the_pointer_names_this_generation_and_its_plan(self):
        if not (ROOT / phase3c.NODE_PLAN_PATH).is_file():
            pytest.skip(f"{ARTIFACT_ONLY} {phase3c.NODE_PLAN_PATH} is not "
                        "staged")
        assert phase3c.staged_pointer_problems(ROOT) == []

    def test_the_pointer_document_is_built_from_the_staged_bytes(self):
        from src.training.session import sha256_file

        if not (ROOT / phase3c.NODE_PLAN_PATH).is_file():
            pytest.skip(f"{ARTIFACT_ONLY} {phase3c.NODE_PLAN_PATH} is not "
                        "staged")
        body = phase3c.staged_pointer_document(ROOT)
        assert body["generation"] == phase3c.GENERATION
        assert body["staged_plan"] == phase3c.NODE_PLAN_PATH
        assert body["plan_sha256"] == sha256_file(ROOT
                                                  / phase3c.NODE_PLAN_PATH)


class TestTheAuthorizationChecksWhatActuallyTravelled:
    """A stale pointer cannot be authorised, whatever it said at build time."""

    def _evidence(self, files):
        return {"pack_manifest": {"files": files}}

    def test_this_generations_plan_at_the_archives_bytes_passes(self):
        from src.training.session import sha256_file

        archived = ROOT / phase3c.PLAN_PATH
        if not archived.is_file():
            pytest.skip(f"{ARTIFACT_ONLY} {phase3c.PLAN_PATH} is not frozen")
        evidence = self._evidence({
            phase3c.NODE_PLAN_PATH: {"sha256": sha256_file(archived)},
            "src/eval/phase3c.py": {"sha256": "0" * 64},
        })
        assert phase3c.pack_carries_this_generation_problems(evidence) == []

    def test_a_superseded_plan_in_the_pack_is_refused(self):
        evidence = self._evidence(
            {"gpu_plans/phase3c_gen01_plan.json": {"sha256": "0" * 64}})
        problems = phase3c.pack_carries_this_generation_problems(evidence)
        assert any("only that one may travel" in p for p in problems), problems

    def test_two_plans_in_the_pack_are_refused(self):
        evidence = self._evidence({
            phase3c.NODE_PLAN_PATH: {"sha256": "0" * 64},
            "gpu_plans/phase3c_gen01_plan.json": {"sha256": "1" * 64},
        })
        problems = phase3c.pack_carries_this_generation_problems(evidence)
        assert any("only that one may travel" in p for p in problems), problems

    def test_no_plan_in_the_pack_is_refused(self):
        problems = phase3c.pack_carries_this_generation_problems(
            self._evidence({"src/eval/phase3c.py": {"sha256": "0" * 64}}))
        assert problems

    def test_this_generations_path_carrying_other_bytes_is_refused(self):
        archived = ROOT / phase3c.PLAN_PATH
        if not archived.is_file():
            pytest.skip(f"{ARTIFACT_ONLY} {phase3c.PLAN_PATH} is not frozen")
        evidence = self._evidence(
            {phase3c.NODE_PLAN_PATH: {"sha256": "0" * 64}})
        problems = phase3c.pack_carries_this_generation_problems(evidence)
        assert any("nobody archived" in p for p in problems), problems


# ---------------------------------------------------------------------------
# The warm-up is checked before the first cell, because nothing checks it
# afterwards
# ---------------------------------------------------------------------------
#
# A warm-up is decoded and thrown away. No digest covers it, no sample row
# records it, the seal does not name it and the receipt does not re-derive
# it. So a step whose warm-up ran a different gate from its cells -- exactly
# what gen04's runner would have done for arm C -- would seal, verify and
# score with nothing anywhere noticing. The check has to happen between the
# warm-up returning and the first cell being written, or it cannot happen at
# all, and it has to leave the member *absent* when it fires.

def _good_warmup(name: str) -> dict:
    body = phase3c.expected_warm_up(name)
    body["seconds"] = [0.001] * body["generations"]
    return body


class TestTheWarmUpMustBeTheOneTheContractNames:
    def _refused(self, code, run):
        assert code == 2
        assert not run.member.exists(), \
            "a refused warm-up left a member behind"

    def test_the_real_warm_up_passes_its_own_check(self, run_path):
        """The check is not vacuous: the genuine article satisfies it."""
        code, run = run_path(step=2)
        assert code == 0
        assert phase3c.warm_up_problems(_good_warmup("C"), "C") == []

    @pytest.mark.parametrize("step,arm", [(0, "A"), (1, "B"), (2, "C")])
    def test_a_warm_up_reporting_another_arms_gate_is_refused(
            self, run_path, step, arm):
        """The gen04 substitution, injected directly at the warm-up."""
        other = "A" if arm != "A" else "C"
        bad = _good_warmup(arm)
        bad["gate"] = phase3c.ARMS[other].gate
        code, run = run_path(step=step, warmup=bad)
        self._refused(code, run)

    def test_a_warm_up_naming_another_arm_is_refused(self, run_path):
        bad = _good_warmup("C")
        bad["arm"] = "B"
        code, run = run_path(step=2, warmup=bad)
        self._refused(code, run)

    def test_a_warm_up_with_the_wrong_connectivity_is_refused(self, run_path):
        """Arm C's collision mask without its EOS layer is not arm C."""
        bad = _good_warmup("C")
        bad["connectivity"] = None
        code, run = run_path(step=2, warmup=bad)
        self._refused(code, run)

    def test_a_warm_up_at_other_seeds_is_refused(self, run_path):
        bad = _good_warmup("C")
        bad["seeds"] = [1, 2]
        code, run = run_path(step=2, warmup=bad)
        self._refused(code, run)

    def test_a_warm_up_of_the_wrong_size_is_refused(self, run_path):
        bad = _good_warmup("C")
        bad["generations"] = 3
        code, run = run_path(step=2, warmup=bad)
        self._refused(code, run)

    def test_a_warm_up_whose_seconds_do_not_match_its_count_is_refused(
            self, run_path):
        """It says two and timed one, so it did not decode what it claims."""
        bad = _good_warmup("C")
        bad["seconds"] = [0.001]
        code, run = run_path(step=2, warmup=bad)
        self._refused(code, run)

    def test_a_warm_up_taken_from_the_test_split_is_refused(self, run_path):
        bad = _good_warmup("C")
        bad["caption_is_from_the_test_split"] = True
        code, run = run_path(step=2, warmup=bad)
        self._refused(code, run)

    def test_a_warm_up_that_is_not_excluded_is_refused(self, run_path):
        """Its seconds would then be in a reported number."""
        bad = _good_warmup("C")
        bad["excluded_from_every_reported_number"] = False
        code, run = run_path(step=2, warmup=bad)
        self._refused(code, run)

    def test_a_warm_up_under_a_different_policy_is_refused(self, run_path):
        bad = _good_warmup("C")
        bad["policy"] = "warmed once, sometimes"
        code, run = run_path(step=2, warmup=bad)
        self._refused(code, run)

    @pytest.mark.parametrize("field", list(phase3c.WARMUP_FIELDS))
    def test_an_incomplete_warm_up_record_is_refused(self, run_path, field):
        bad = _good_warmup("C")
        bad.pop(field)
        code, run = run_path(step=2, warmup=bad)
        self._refused(code, run)

    def test_a_warm_up_carrying_an_unnamed_field_is_refused(self, run_path):
        """Something other than warm_up produced it."""
        bad = _good_warmup("C")
        bad["measured"] = True
        code, run = run_path(step=2, warmup=bad)
        self._refused(code, run)

    @pytest.mark.parametrize("record", [None, [], "warmed", 2])
    def test_a_warm_up_that_is_not_a_record_at_all_is_refused(
            self, run_path, record):
        code, run = run_path(step=2, warmup=record)
        self._refused(code, run)

    @pytest.mark.parametrize("error", [
        RuntimeError("CUDA out of memory during warm-up"),
        ValueError("the gate refused every token"),
        MemoryError(),
    ])
    def test_a_warm_up_that_fails_is_refused_rather_than_skipped(
            self, run_path, error):
        """An exception is not "no warm-up needed"; it is an unknown state."""
        code, run = run_path(step=2, warmup=error)
        self._refused(code, run)

    def test_an_interrupted_warm_up_propagates_and_writes_nothing(
            self, run_path, tmp_path):
        """Deliberately *not* swallowed.

        ``mode_run`` catches ``Exception``, not ``BaseException``, so an
        operator's Ctrl-C leaves the process rather than being turned into a
        named refusal that reads like an orderly stop. What matters for this
        suite is the other half: it happens before the first cell, so the
        member is absent either way.

        Checked against **this run's own out-dir**. It used to assert that
        ``runs/phase3c`` did not exist in the repository, which was a claim
        about the tree rather than about the step: once gen09 ran to
        completion and its out-dir was kept, the assertion failed while the
        behaviour it was written for was still correct.
        """
        with pytest.raises(KeyboardInterrupt):
            run_path(step=2, warmup=KeyboardInterrupt())
        # The same path ``run_path`` gives the step.
        out = tmp_path / "run_step2"
        # The manifest is written before the warm-up, so it is here; the
        # member is not, because the interrupt landed before the first cell.
        assert (out / "execution_manifest.json").is_file()
        assert not (out / phase3c.samples_member(2)).exists()
        assert not (out / "seal.json").exists()
        assert not (out / "receipt.json").exists()

    def test_a_refused_warm_up_decodes_no_cell(self, run_path):
        """The member is absent, and the gate was never asked for a cell."""
        bad = _good_warmup("C")
        bad["gate"] = phase3c.GATE_NONE
        code, run = run_path(step=2, warmup=bad)
        self._refused(code, run)
        assert run.decodes(warmup=False) == []
        assert ("warm_up", "substituted") in run.calls


# ---------------------------------------------------------------------------
# The failed attempt is read, not cited
# ---------------------------------------------------------------------------
#
# A failed attempt is the one artefact with no digest chain of its own.
# There are no cells, so there is no seal, so there is no receipt and nothing
# re-derives anything. What it has is an index naming every file it holds
# with that file's size and SHA-256, and a read-only listing of the node
# directory taken afterwards -- and neither is worth anything alone, because
# an index can be edited to agree with itself and a listing is only text.
#
# So they are checked against each other, against the bytes on disk, and
# against the frozen generation the attempt ran under; and the supersession
# entry that names the attempt binds its index's own SHA-256, so pointing at
# a different directory is not a matching string but a refusal.
#
# Every case below works on a temporary copy. No frozen byte is touched to
# test what happens when a frozen byte moves.

ATTEMPT = ("data/phase3c/failed_attempts/"
           "gen04_runner_defect_20260830T070331Z")


@pytest.fixture()
def attempt(tmp_path):
    """A writable copy of the real failed attempt, plus the tree it names."""
    import shutil

    source = ROOT / ATTEMPT
    if not (source / "index.json").is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {ATTEMPT} is not published")
    root = tmp_path / "tree"
    (root / ATTEMPT).parent.mkdir(parents=True)
    shutil.copytree(source, root / ATTEMPT)
    archive = ROOT / phase3c.ARCHIVE_ROOT / "gen04"
    if archive.is_dir():
        (root / phase3c.ARCHIVE_ROOT).mkdir(parents=True, exist_ok=True)
        shutil.copytree(archive, root / phase3c.ARCHIVE_ROOT / "gen04")
    return root, root / ATTEMPT


def _reindex(directory: Path, index: dict) -> None:
    """Rewrite the index and recompute its own digest, as a forger would."""
    import hashlib

    from src.eval.acceptance import canonical_json

    index.pop("index_digest", None)
    index["index_digest"] = hashlib.sha256(
        canonical_json(index).encode("utf-8")).hexdigest()
    (directory / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def _index(directory: Path) -> dict:
    return json.loads((directory / "index.json").read_text())


class TestTheFailedAttemptHolds:
    def test_the_real_one_verifies(self, attempt):
        root, directory = attempt
        assert phase3c.failed_attempt_problems(directory, root=root) == []

    def test_it_verifies_against_a_relocated_tree(self, attempt):
        """Root-aware: the check is not reading the repository behind itself."""
        root, directory = attempt
        assert str(root) != str(ROOT)
        assert phase3c.failed_attempt_problems(directory, root=root) == []


class TestTheFailedAttemptFailsClosed:
    def _refused(self, root, directory, needle=None):
        problems = phase3c.failed_attempt_problems(directory, root=root)
        assert problems, "a tampered attempt verified"
        if needle:
            assert any(needle in p for p in problems), problems

    # -- the claim itself ------------------------------------------------
    def test_a_nonzero_cell_count_is_refused(self, attempt):
        root, directory = attempt
        index = _index(directory)
        index["proof_no_cell_was_written"]["total_cells_written"] = 1
        _reindex(directory, index)
        self._refused(root, directory, "cells were written")

    def test_claiming_it_may_be_cited_is_refused(self, attempt):
        root, directory = attempt
        index = _index(directory)
        index["citable_as_a_result"] = True
        _reindex(directory, index)
        self._refused(root, directory, "citable_as_a_result")

    def test_an_outcome_that_is_not_failed_is_refused(self, attempt):
        root, directory = attempt
        index = _index(directory)
        index["outcome"] = "succeeded"
        _reindex(directory, index)
        self._refused(root, directory, "outcome")

    # -- the sample members ----------------------------------------------
    @pytest.mark.parametrize("state", ["zero_byte", "populated"])
    def test_a_member_recorded_as_anything_but_absent_is_refused(
            self, attempt, state):
        root, directory = attempt
        index = _index(directory)
        index["proof_no_cell_was_written"]["expected_members"][0]["state"] = \
            state
        _reindex(directory, index)
        self._refused(root, directory)

    def test_an_unknown_member_state_is_refused(self, attempt):
        root, directory = attempt
        index = _index(directory)
        index["proof_no_cell_was_written"]["expected_members"][2]["state"] = \
            "probably_absent"
        _reindex(directory, index)
        self._refused(root, directory, "not one of")

    def test_an_absent_member_carrying_a_digest_is_refused(self, attempt):
        root, directory = attempt
        index = _index(directory)
        index["proof_no_cell_was_written"]["expected_members"][1].update(
            {"bytes": 0, "sha256": "0" * 64})
        _reindex(directory, index)
        self._refused(root, directory, "does not have")

    def test_a_planted_zero_byte_sample_is_refused(self, attempt):
        """The distinction the whole zero-cell claim rests on.

        A zero-byte file is something a process created and left empty.
        Fabricating one to 'show' that no cell was written destroys exactly
        the evidence it pretends to supply, so the listing has to be the
        thing that answers, and it has to say absent.
        """
        root, directory = attempt
        planted = directory / "node" / "samples"
        planted.mkdir(parents=True, exist_ok=True)
        (planted / phase3c.samples_member(0).split("/")[-1]).write_bytes(b"")
        self._refused(root, directory, "does not name it")

    def test_a_listing_that_found_a_populated_member_is_refused(self, attempt):
        root, directory = attempt
        index = _index(directory)
        listing = directory / index["node"]["observation"]["listing"]
        listing.write_text(
            listing.read_text().replace("step_02_even_C ABSENT -",
                                        "step_02_even_C POPULATED 4096"),
            encoding="utf-8")
        entry = next(e for e in index["evidence_files"]
                     if e["path"].endswith(listing.name))
        import hashlib
        blob = listing.read_bytes()
        entry.update(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        _reindex(directory, index)
        self._refused(root, directory, "not absent")

    def test_a_listing_that_probes_too_few_members_is_refused(self, attempt):
        root, directory = attempt
        index = _index(directory)
        listing = directory / index["node"]["observation"]["listing"]
        listing.write_text(
            listing.read_text().replace("step_05_odd_A ABSENT -\n", ""),
            encoding="utf-8")
        entry = next(e for e in index["evidence_files"]
                     if e["path"].endswith(listing.name))
        import hashlib
        blob = listing.read_bytes()
        entry.update(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        _reindex(directory, index)
        self._refused(root, directory)

    # -- a seal, a receipt, scores or a report ---------------------------
    @pytest.mark.parametrize("name", ["seal", "receipt", "scores", "report"])
    def test_claiming_one_of_the_four_exists_is_refused(self, attempt, name):
        root, directory = attempt
        index = _index(directory)
        index["not_created_for_this_attempt"][name] = True
        _reindex(directory, index)
        self._refused(root, directory, name)

    @pytest.mark.parametrize("name", ["seal.json", "receipt.json",
                                      "scores.json"])
    def test_one_of_the_four_appearing_on_disk_is_refused(self, attempt, name):
        root, directory = attempt
        (directory / "node" / name).write_text("{}\n", encoding="utf-8")
        self._refused(root, directory)

    def test_a_listing_that_found_a_seal_is_refused(self, attempt):
        root, directory = attempt
        index = _index(directory)
        listing = directory / index["node"]["observation"]["listing"]
        text = listing.read_text()
        head, sep, tail = text.partition("### COMMAND: sh -c find ")
        assert sep, "the listing has no seal probe to alter"
        block, sep2, rest = tail.partition("### EXIT_CODE:")
        listing.write_text(
            head + sep + block.replace("\n0\n", "\n1\n") + sep2 + rest,
            encoding="utf-8")
        entry = next(e for e in index["evidence_files"]
                     if e["path"].endswith(listing.name))
        import hashlib
        blob = listing.read_bytes()
        entry.update(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        _reindex(directory, index)
        self._refused(root, directory, "must hold none")

    # -- the evidence table ----------------------------------------------
    def test_an_evidence_file_whose_bytes_moved_is_refused(self, attempt):
        root, directory = attempt
        (directory / "node" / "driver.log").write_text("edited\n",
                                                       encoding="utf-8")
        self._refused(root, directory, "digests to")

    def test_an_evidence_entry_with_a_rewritten_digest_is_refused(
            self, attempt):
        """The table is checked against the bytes, not against itself."""
        root, directory = attempt
        index = _index(directory)
        index["evidence_files"][0]["sha256"] = "0" * 64
        _reindex(directory, index)
        self._refused(root, directory)

    def test_a_file_the_index_does_not_name_is_refused(self, attempt):
        root, directory = attempt
        (directory / "node" / "extra.txt").write_text("x\n", encoding="utf-8")
        self._refused(root, directory, "does not name it")

    def test_a_named_file_that_is_gone_is_refused(self, attempt):
        root, directory = attempt
        (directory / "node" / "driver_nohup.log").unlink()
        self._refused(root, directory, "is not here")

    def test_an_index_digest_that_does_not_cover_the_record_is_refused(
            self, attempt):
        root, directory = attempt
        index = _index(directory)
        index["statement"] = "everything was fine"
        (directory / "index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        self._refused(root, directory, "index_digest")

    # -- the manifest identity -------------------------------------------
    def test_a_manifest_whose_own_digest_was_recomputed_is_still_refused(
            self, attempt):
        """Forged coherently: the manifest agrees with itself and not with
        the generation it claims to have run under."""
        root, directory = attempt
        path = directory / "node" / "execution_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["pack_digest"] = "0" * 64
        body = {k: v for k, v in manifest.items() if k != "manifest_digest"}
        from src.eval.acceptance import digest_obj
        manifest["manifest_digest"] = digest_obj(body)
        path.write_text(json.dumps(manifest, indent=2) + "\n",
                        encoding="utf-8")
        index = _index(directory)
        entry = next(e for e in index["evidence_files"]
                     if e["path"].endswith("execution_manifest.json"))
        import hashlib
        blob = path.read_bytes()
        entry.update(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        index["digests_this_attempt_ran_under"]["pack_digest"] = "0" * 64
        index["digests_this_attempt_ran_under"]["execution_manifest_digest"] \
            = manifest["manifest_digest"]
        _reindex(directory, index)
        self._refused(root, directory, "archived grant")

    def test_a_manifest_digest_that_does_not_cover_it_is_refused(
            self, attempt):
        root, directory = attempt
        path = directory / "node" / "execution_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["plan_digest"] = "1" * 64
        path.write_text(json.dumps(manifest, indent=2) + "\n",
                        encoding="utf-8")
        index = _index(directory)
        entry = next(e for e in index["evidence_files"]
                     if e["path"].endswith("execution_manifest.json"))
        import hashlib
        blob = path.read_bytes()
        entry.update(bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        _reindex(directory, index)
        self._refused(root, directory, "digests to")

    def test_an_index_that_misreports_what_it_ran_under_is_refused(
            self, attempt):
        root, directory = attempt
        index = _index(directory)
        index["digests_this_attempt_ran_under"]["authorization_digest"] = \
            "2" * 64
        _reindex(directory, index)
        self._refused(root, directory, "authorization_digest")

    def test_a_missing_listing_is_refused(self, attempt):
        root, directory = attempt
        index = _index(directory)
        index["node"]["observation"].pop("listing")
        _reindex(directory, index)
        self._refused(root, directory, "listing")


class TestTheSupersessionReadsTheAttempt:
    def test_this_generations_record_holds(self):
        if not _attempts_are_published():
            pytest.skip(f"{ARTIFACT_ONLY} the attempted executions this "
                        "record names are not published")
        record = phase3c.build_supersession_record()
        assert phase3c.supersession_problems(record) == []

    def test_an_entry_naming_a_path_with_no_digest_is_refused(self):
        record = phase3c.build_supersession_record()
        for entry in record["not_executable"]:
            entry.pop("attempted_execution_index_sha256", None)
        problems = phase3c.attempted_execution_problems(record)
        assert any("a path is not a binding" in p for p in problems), problems

    def test_an_attempt_and_its_binding_cannot_both_disappear(
            self, plan, tmp_path):
        record = phase3c.build_supersession_record()
        entry = next(e for e in record["not_executable"]
                     if e["generation"] == "gen04")
        del entry["attempted_execution"]
        del entry["attempted_execution_index_sha256"]
        record["supersession_digest"] = digest_obj(
            {k: v for k, v in record.items()
             if k != "supersession_digest"})

        problems = phase3c.supersession_problems(record, root=ROOT)
        assert any("not what this module records" in p for p in problems), \
            problems

        assert _ARCHIVE_FOR_TESTS is not None
        frozen = tmp_path / "frozen"
        _copy_tree(_ARCHIVE_FOR_TESTS.parent, frozen)
        archive = frozen / phase3c.GENERATION
        supersession = frozen / Path(phase3c.SUPERSESSION_PATH).name
        supersession.write_text(json.dumps(record, indent=2) + "\n",
                                 encoding="utf-8")
        problems = phase3c.archive_problems(
            plan, archive, root=ROOT)
        assert any("not what this module records" in p for p in problems), \
            problems

    def test_an_entry_bound_to_other_bytes_is_refused(self):
        if not _attempts_are_published():
            pytest.skip(f"{ARTIFACT_ONLY} the attempted executions this "
                        "record names are not published")
        record = phase3c.build_supersession_record()
        for entry in record["not_executable"]:
            if "attempted_execution_index_sha256" in entry:
                entry["attempted_execution_index_sha256"] = "3" * 64
        problems = phase3c.attempted_execution_problems(record)
        assert any("digests to" in p for p in problems), problems

    def test_an_entry_naming_a_directory_that_is_not_there_is_refused(
            self, tmp_path):
        record = phase3c.build_supersession_record()
        problems = phase3c.attempted_execution_problems(record, root=tmp_path)
        assert any("no index.json there" in p for p in problems), problems

    def test_an_entry_claiming_cells_or_a_seal_is_refused(self, attempt):
        root, _directory = attempt
        record = phase3c.build_supersession_record()
        for entry in record["not_executable"]:
            if entry.get("attempted_execution"):
                entry["cells_produced"] = 320
                entry["sealed"] = True
        problems = phase3c.attempted_execution_problems(record, root=root)
        assert any("cells" in p for p in problems), problems
        assert any("seal" in p for p in problems), problems

    def test_a_tampered_attempt_fails_the_supersession_too(self, attempt):
        """The binding is end to end: the record reads the directory."""
        root, directory = attempt
        index = _index(directory)
        index["proof_no_cell_was_written"]["total_cells_written"] = 1920
        _reindex(directory, index)
        problems = phase3c.attempted_execution_problems(
            phase3c.build_supersession_record(), root=root)
        assert problems


# ---------------------------------------------------------------------------
# The ledger, against the gates a decode really builds
# ---------------------------------------------------------------------------
#
# gen08 reached the node and died here. ``gate_ledger`` called
# ``gate.counters()`` on whatever object it was handed: arm A returns before
# the call, arm C's ``InventoryPlacementGate`` has the method, and arm B's
# ``InventoryGate`` does not. Step 0 wrote 320 cells; step 1 failed on its
# first.
#
# It survived review because the suite's stand-in gate implemented
# ``counters()`` -- a superset of the real interface. So every case here
# constructs the class the arm's own decode constructs, and there is no
# hand-written gate object anywhere in this file.

def _slots():
    from src.generation.brickgpt import MAX_DIM, Slots, WORLD
    return Slots(dims=list(range(100, 100 + MAX_DIM)),
                 posns=list(range(200, 200 + WORLD)), literal_x=1,
                 literal_open=2, literal_comma=3, literal_close=4, eos=5)


_LEDGER_INVENTORY = {"1x2": 4, "2x2": 4, "2x4": 4}


class TestTheLedgerUsesTheRealGate:
    """Arms A, B and C, each with the object its own decode returns."""

    def test_arm_a_reports_no_gate_and_no_stock(self):
        ledger = phase3c.gate_ledger(phase3c.arm("A"), _LEDGER_INVENTORY, None)
        assert ledger["gate"] == phase3c.GATE_NONE
        assert ledger["connectivity"] is None
        assert ledger["accepted_parts"] is None
        assert ledger["remaining_inventory"] is None
        assert ledger["counters"] == \
            phase3c.acceptance.unimplemented_counters()

    def test_arm_b_has_no_counters_and_still_gets_a_correct_ledger(self):
        """The exact gen08 failure, as a passing case.

        ``InventoryGate`` has no ``counters()``. That is not a defect in the
        gate: arm B has no placement layer, so it has no placement counters.
        The ledger has to say so without inventing them.
        """
        from src.constraints.inventory_decode import InventoryGate

        assert not hasattr(InventoryGate, "counters")
        gate = real_gate("B", _LEDGER_INVENTORY)
        assert type(gate) is InventoryGate
        ledger = phase3c.gate_ledger(phase3c.arm("B"), _LEDGER_INVENTORY, gate)

        assert ledger["gate"] == phase3c.GATE_INVENTORY
        assert ledger["connectivity"] is None
        assert ledger["accepted_parts"] == []
        assert ledger["remaining_inventory"] == _LEDGER_INVENTORY
        # Unimplemented, not zero. "The layer masked nothing" and "there is
        # no layer" are different statements and only one of them is true.
        assert ledger["counters"] == \
            phase3c.acceptance.unimplemented_counters()
        assert "candidates_masked_total" not in ledger["counters"]

    def test_arm_c_reports_the_real_placement_counters(self):
        from src.constraints.placement_decode import InventoryPlacementGate

        gate = real_gate("C", _LEDGER_INVENTORY)
        assert type(gate) is InventoryPlacementGate
        ledger = phase3c.gate_ledger(phase3c.arm("C"), _LEDGER_INVENTORY, gate)

        assert ledger["gate"] == phase3c.GATE_INVENTORY_PLACEMENT
        assert ledger["connectivity"] == phase3c.CONNECTIVITY_MODE
        assert ledger["remaining_inventory"] == _LEDGER_INVENTORY
        # Real, from the gate: present and numeric, not the unimplemented
        # block arm B carries.
        assert ledger["counters"]["candidates_masked_total"] == 0
        assert ledger["counters"]["connectivity"] == phase3c.CONNECTIVITY_MODE
        assert ledger["counters"] != \
            phase3c.acceptance.unimplemented_counters()

    def test_the_ledger_reads_stock_the_gate_actually_spent(self):
        """Not a copy of the opening quantities."""
        gate = real_gate("B", _LEDGER_INVENTORY)
        gate.inventory.deduct("2x4")
        ledger = phase3c.gate_ledger(phase3c.arm("B"), _LEDGER_INVENTORY, gate)
        assert ledger["opening_inventory"]["2x4"] == 4
        assert ledger["remaining_inventory"]["2x4"] == 3


class TestTheLedgerFailsClosedOnTheWrongGate:
    def _refused(self, name, gate, needle):
        with pytest.raises(PlanRefused) as caught:
            phase3c.gate_ledger(phase3c.arm(name), _LEDGER_INVENTORY, gate)
        assert needle in str(caught.value), str(caught.value)

    def test_arm_a_handed_a_gate_is_refused(self):
        self._refused("A", real_gate("B", _LEDGER_INVENTORY), "names no gate")

    @pytest.mark.parametrize("name", ["B", "C"])
    def test_a_gated_arm_handed_no_gate_is_refused(self, name):
        self._refused(name, None, "returned no gate")

    def test_arm_b_handed_arm_cs_gate_is_refused(self):
        """``InventoryPlacementGate`` subclasses ``InventoryGate``.

        An ``isinstance`` check would accept it here, and arm B would be
        measured with the placement layer running -- so ``C - B`` would be
        the difference between two gated arms.
        """
        from src.constraints.inventory_decode import InventoryGate
        from src.constraints.placement_decode import InventoryPlacementGate

        assert issubclass(InventoryPlacementGate, InventoryGate)
        gate = real_gate("C", _LEDGER_INVENTORY)
        assert isinstance(gate, InventoryGate)
        self._refused("B", gate, "InventoryPlacementGate")

    def test_arm_c_handed_arm_bs_gate_is_refused(self):
        self._refused("C", real_gate("B", _LEDGER_INVENTORY), "InventoryGate")

    def test_arm_c_whose_gate_lost_its_counters_is_refused(self, monkeypatch):
        """Not a fall back to the unimplemented block."""
        from src.constraints.placement_decode import InventoryPlacementGate

        gate = real_gate("C", _LEDGER_INVENTORY)
        monkeypatch.delattr(InventoryPlacementGate, "counters")
        self._refused("C", gate, "does not report counters")

    def test_arm_c_whose_counters_raise_is_refused(self, monkeypatch):
        from src.constraints.placement_decode import InventoryPlacementGate

        gate = real_gate("C", _LEDGER_INVENTORY)

        def boom(self):
            raise RuntimeError("no")

        monkeypatch.setattr(InventoryPlacementGate, "counters", boom)
        self._refused("C", gate, "could not report its counters")

    def test_arm_c_whose_counters_are_not_a_record_is_refused(self,
                                                              monkeypatch):
        from src.constraints.placement_decode import InventoryPlacementGate

        gate = real_gate("C", _LEDGER_INVENTORY)
        monkeypatch.setattr(InventoryPlacementGate, "counters",
                            lambda self: ["not", "a", "record"])
        self._refused("C", gate, "not a record")

    def test_arm_c_in_the_wrong_connectivity_mode_is_refused(self):
        """The mode comes from the gate, not from the spec that asked."""
        from src.constraints.placement_decode import InventoryPlacementGate
        from src.inventory.engine import Inventory

        gate = InventoryPlacementGate(
            _slots(), Inventory.from_parts(dict(_LEDGER_INVENTORY)),
            enabled=True, connectivity="off")
        gate.opening_inventory = dict(_LEDGER_INVENTORY)
        self._refused("C", gate, "its gate is in 'off'")

    def test_a_counters_interrupt_still_propagates(self, monkeypatch):
        """``except Exception`` and deliberately not ``BaseException``."""
        from src.constraints.placement_decode import InventoryPlacementGate

        gate = real_gate("C", _LEDGER_INVENTORY)

        def interrupted(self):
            raise KeyboardInterrupt()

        monkeypatch.setattr(InventoryPlacementGate, "counters", interrupted)
        with pytest.raises(KeyboardInterrupt):
            phase3c.gate_ledger(phase3c.arm("C"), _LEDGER_INVENTORY, gate)


class TestTheRunPathBuildsTheRealGate:
    """The real gate classes reach the warm-up and the first measured row."""

    @pytest.mark.parametrize("step,arm,gate", _STEP_ARM_GATE)
    def test_the_gate_that_reaches_the_row_is_the_real_class(
            self, run_path, step, arm, gate):
        code, run = run_path(step=step)
        assert code == 0, arm
        rows = phase3c.read_rows(run.member)
        assert len(rows) == 320, arm
        for row in rows:
            assert row["gate"]["gate"] == gate, arm

    def test_arm_b_writes_stock_and_no_invented_placement_counters(
            self, run_path):
        """Step 1 is the step gen08 died on."""
        code, run = run_path(step=1)
        assert code == 0
        row = phase3c.read_rows(run.member)[0]
        assert row["gate"]["gate"] == phase3c.GATE_INVENTORY
        assert row["gate"]["connectivity"] is None
        assert row["gate"]["remaining_inventory"] is not None
        assert row["gate"]["accepted_parts"] is not None
        assert row["gate"]["counters"] == \
            phase3c.acceptance.unimplemented_counters()

    def test_arm_c_writes_real_placement_counters(self, run_path):
        code, run = run_path(step=2)
        assert code == 0
        row = phase3c.read_rows(run.member)[0]
        assert row["gate"]["gate"] == phase3c.GATE_INVENTORY_PLACEMENT
        assert row["gate"]["connectivity"] == phase3c.CONNECTIVITY_MODE
        assert "candidates_masked_total" in row["gate"]["counters"]

    def test_a_run_whose_gate_lost_its_counters_writes_no_member(
            self, run_path, monkeypatch):
        """A refusal mid-step leaves nothing published for that step."""
        from src.constraints.placement_decode import InventoryPlacementGate

        monkeypatch.delattr(InventoryPlacementGate, "counters")
        code, run = run_path(step=2)
        assert code != 0
        assert not run.member.exists()
        assert not (run.out_dir / "seal.json").exists()
        assert not (run.out_dir / "receipt.json").exists()

    @pytest.mark.parametrize("step,arm,expected", [
        (0, "A", None),
        (1, "B", "InventoryGate"),
        (2, "C", "InventoryPlacementGate"),
    ])
    def test_the_object_the_decode_built_is_the_shipped_class(
            self, run_path, step, arm, expected):
        """Recorded from ``type(gate)``, warm-up and cells alike.

        The substitution that hid gen08 returned a hand-written stand-in
        implementing ``counters()`` -- a superset of the real
        ``InventoryGate``. Nothing here can pass unless the shipped gated
        entry points ran and built the shipped class.
        """
        code, run = run_path(step=step)
        assert code == 0, arm
        assert run.gate_classes(warmup=True) == [expected], arm
        assert run.gate_classes(warmup=False) == [expected], arm


# ---------------------------------------------------------------------------
# The partial attempt: gen08 reached the node and stopped part way
# ---------------------------------------------------------------------------
#
# gen04's record claims "no cell exists" and every member absent. gen08's
# claims "step 0 wrote exactly its 320 cells and steps 1-5 never started",
# which is a strictly harder thing to hold: the 320 rows have to BE the 320
# cells the plan predetermined for that step, bound to that run's execution
# manifest. So the member is read and judged by the same ``step_problems`` a
# real run is judged by, and the five absent members have to be absent
# rather than empty.
#
# Every case below works on a temporary copy. No frozen byte is touched.

PARTIAL_ATTEMPT = ("data/phase3c/failed_attempts/"
                   "gen08_gate_ledger_defect_20260831T044425Z")
PARTIAL_MEMBER = "node/samples/step_00_even_A.jsonl"


@pytest.fixture()
def partial(tmp_path):
    """A writable copy of the gen08 attempt, plus the archive it names."""
    import shutil

    source = ROOT / PARTIAL_ATTEMPT
    if not (source / "index.json").is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {PARTIAL_ATTEMPT} is not published")
    root = tmp_path / "tree"
    (root / PARTIAL_ATTEMPT).parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, root / PARTIAL_ATTEMPT)
    archive = ROOT / phase3c.ARCHIVE_ROOT / "gen08"
    if not archive.is_dir():
        pytest.skip(f"{ARTIFACT_ONLY} gen08's archive is not published")
    (root / phase3c.ARCHIVE_ROOT).mkdir(parents=True, exist_ok=True)
    shutil.copytree(archive, root / phase3c.ARCHIVE_ROOT / "gen08")
    return root, root / PARTIAL_ATTEMPT


def _reseal(directory: Path) -> dict:
    """Re-file and re-digest the whole index, as a careful forger would.

    Not just ``index_digest``: the evidence table AND every member entry's
    own size and SHA-256, so a test that changes bytes is answered by the
    check that reads those bytes rather than by a size that stopped matching.
    """
    index = _index(directory)
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "index.json":
            blob = path.read_bytes()
            files.append({"path": str(path.relative_to(directory)),
                          "bytes": len(blob),
                          "sha256": hashlib.sha256(blob).hexdigest()})
    index["evidence_files"] = files
    for entry in index.get(phase3c.PARTIAL_PROOF, {}).get("members", []):
        rel = entry.get("evidence_path")
        if not rel:
            continue
        blob = (directory / rel).read_bytes()
        entry["bytes"] = len(blob)
        entry["sha256"] = hashlib.sha256(blob).hexdigest()
    _reindex(directory, index)
    return index


def _member_entry(index: dict, step: int = 0) -> dict:
    return index[phase3c.PARTIAL_PROOF]["members"][step]


class TestThePartialAttemptHolds:
    def test_the_real_one_verifies(self, partial):
        root, directory = partial
        assert phase3c.failed_attempt_problems(directory, root=root) == []

    def test_it_verifies_against_a_relocated_tree(self, partial):
        root, directory = partial
        assert str(root) != str(ROOT)
        assert phase3c.failed_attempt_problems(directory, root=root) == []

    def test_it_records_step_zero_populated_and_the_rest_absent(self,
                                                                partial):
        _root, directory = partial
        proof = _index(directory)[phase3c.PARTIAL_PROOF]
        states = [(e["member"], e["state"]) for e in proof["members"]]
        assert states[0] == (phase3c.samples_member(0), "populated")
        assert [s for _m, s in states[1:]] == ["absent"] * 5
        assert proof["total_cells_written"] == 320
        assert proof["total_cells_planned"] == 1920

    def test_the_five_absent_members_are_absent_on_disk(self, partial):
        _root, directory = partial
        for step in range(1, phase3c.N_STEPS):
            stem = Path(phase3c.samples_member(step)).name
            assert not (directory / "node" / "samples" / stem).exists()

    def test_the_schema_two_record_still_verifies_unchanged(self, attempt):
        """gen04's bytes are held to every check they were written under."""
        root, directory = attempt
        assert _index(directory)["schema_version"] == 2
        assert phase3c.failed_attempt_problems(directory, root=root) == []

    def test_the_cell_count_is_read_from_either_schema(self, attempt,
                                                       partial):
        assert phase3c.failed_attempt_cells(_index(attempt[1])) == 0
        assert phase3c.failed_attempt_cells(_index(partial[1])) == 320


class TestThePartialAttemptFailsClosed:
    def _refused(self, root, directory, needle=None):
        problems = phase3c.failed_attempt_problems(directory, root=root)
        assert problems, "a tampered partial attempt verified"
        if needle:
            assert any(needle in p for p in problems), problems

    def _rows(self, directory):
        return (directory / PARTIAL_MEMBER).read_text().splitlines(True)

    def _write(self, directory, lines):
        (directory / PARTIAL_MEMBER).write_text("".join(lines))

    # -- the member's own bytes ------------------------------------------
    def test_altered_evidence_bytes_are_refused(self, partial):
        """Without re-filing: the evidence table answers this one."""
        root, directory = partial
        (directory / "node" / "driver.log").write_text("rewritten\n")
        self._refused(root, directory, "digests to")

    def test_a_declared_size_that_does_not_match_is_refused(self, partial):
        root, directory = partial
        index = _index(directory)
        _member_entry(index)["bytes"] = 1
        _reindex(directory, index)
        self._refused(root, directory, "bytes")

    # -- the rows, with every size and digest re-filed --------------------
    def test_a_missing_row_is_refused(self, partial):
        root, directory = partial
        self._write(directory, self._rows(directory)[:-1])
        index = _reseal(directory)
        _member_entry(index)["cells"] = 319
        index[phase3c.PARTIAL_PROOF]["total_cells_written"] = 319
        _reseal(directory)
        self._refused(root, directory, "must produce are absent")

    def test_a_duplicated_row_is_refused(self, partial):
        root, directory = partial
        lines = self._rows(directory)
        self._write(directory, lines[:-1] + [lines[0]])
        _reseal(directory)
        self._refused(root, directory, "appear more than once")

    def test_an_extra_row_is_refused(self, partial):
        root, directory = partial
        lines = self._rows(directory)
        row = json.loads(lines[0])
        row["seed"] = 99
        self._write(directory, lines + [json.dumps(row) + "\n"])
        index = _reseal(directory)
        _member_entry(index)["cells"] = 321
        index[phase3c.PARTIAL_PROOF]["total_cells_written"] = 321
        _reseal(directory)
        self._refused(root, directory)

    def test_a_row_moved_to_another_arm_is_refused(self, partial):
        root, directory = partial
        lines = self._rows(directory)
        row = json.loads(lines[0])
        row["arm"] = "B"
        lines[0] = json.dumps(row) + "\n"
        self._write(directory, lines)
        _reseal(directory)
        self._refused(root, directory)

    def test_a_row_bound_to_another_manifest_is_refused(self, partial):
        root, directory = partial
        lines = self._rows(directory)
        row = json.loads(lines[0])
        row["manifest_digest"] = "0" * 64
        lines[0] = json.dumps(row) + "\n"
        self._write(directory, lines)
        _reseal(directory)
        self._refused(root, directory)

    def test_a_corrupt_row_is_refused(self, partial):
        root, directory = partial
        lines = self._rows(directory)
        lines[3] = "{not json\n"
        self._write(directory, lines)
        _reseal(directory)
        self._refused(root, directory)

    def test_a_wrong_cell_count_is_refused(self, partial):
        root, directory = partial
        index = _index(directory)
        _member_entry(index)["cells"] = 319
        index[phase3c.PARTIAL_PROOF]["total_cells_written"] = 319
        _reindex(directory, index)
        self._refused(root, directory, "holds 320 rows")

    # -- the member set --------------------------------------------------
    def test_a_fabricated_zero_byte_step_one_is_refused(self, partial):
        """A zero-byte member is not an absent one, and the listing knows."""
        root, directory = partial
        (directory / "node" / "samples" / "step_01_even_B.jsonl").write_text("")
        index = _index(directory)
        _member_entry(index, 1).update(
            {"state": "zero_byte", "bytes": 0, "cells": 0,
             "sha256": hashlib.sha256(b"").hexdigest(),
             "evidence_path": "node/samples/step_01_even_B.jsonl"})
        index[phase3c.PARTIAL_PROOF]["entries_in_samples_dir"] = 2
        _reindex(directory, index)
        _reseal(directory)
        self._refused(root, directory, "the listing found it")

    def test_a_planted_member_no_entry_claims_is_refused(self, partial):
        """Filing it in the evidence table is not accounting for it.

        The table answers "is every file here named?"; it does not answer
        "is every sample file a member this attempt actually wrote?".
        """
        root, directory = partial
        (directory / "node" / "samples" / "step_03_odd_C.jsonl").write_text(
            '{"case_id":"x"}\n')
        _reseal(directory)
        self._refused(root, directory, "no member entry accounts for")

    def test_a_populated_member_re_declared_absent_is_refused(self, partial):
        root, directory = partial
        index = _index(directory)
        _member_entry(index).update(
            {"state": "absent", "bytes": None, "sha256": None, "cells": None,
             "evidence_path": None})
        index[phase3c.PARTIAL_PROOF]["total_cells_written"] = 0
        index[phase3c.PARTIAL_PROOF]["entries_in_samples_dir"] = 0
        _reindex(directory, index)
        self._refused(root, directory, "no member is populated")

    def test_an_absent_member_carrying_a_digest_is_refused(self, partial):
        root, directory = partial
        index = _index(directory)
        _member_entry(index, 1)["sha256"] = "4" * 64
        _reindex(directory, index)
        self._refused(root, directory, "an absent file has none of these")

    def test_members_out_of_schedule_order_are_refused(self, partial):
        root, directory = partial
        index = _index(directory)
        members = index[phase3c.PARTIAL_PROOF]["members"]
        members[1], members[2] = members[2], members[1]
        _reindex(directory, index)
        self._refused(root, directory, "the schedule has")

    # -- the surrounding documents ---------------------------------------
    def test_a_fabricated_seal_is_refused(self, partial):
        root, directory = partial
        (directory / "node" / "seal.json").write_text('{"kind":"seal"}\n')
        _reseal(directory)
        self._refused(root, directory, "must not exist")

    def test_a_claimed_seal_is_refused(self, partial):
        root, directory = partial
        index = _index(directory)
        index["not_created_for_this_attempt"]["seal"] = True
        _reindex(directory, index)
        self._refused(root, directory, "says a seal exists")

    def test_an_altered_execution_manifest_is_refused(self, partial):
        root, directory = partial
        path = directory / "node" / "execution_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["pack_digest"] = "0" * 64
        path.write_text(json.dumps(manifest, indent=2))
        _reseal(directory)
        self._refused(root, directory, "manifest_digest")

    def test_an_altered_listing_is_refused(self, partial):
        root, directory = partial
        listing = directory / _index(directory)["node"]["observation"][
            "listing"]
        listing.write_text(listing.read_text().replace(
            "step_01_even_B ABSENT -", "step_01_even_B POPULATED 4096"))
        _reseal(directory)
        self._refused(root, directory, "the listing found it")

    def test_an_index_edited_and_re_digested_is_still_refused(self, partial):
        """Recomputing its own digest is not evidence of anything."""
        root, directory = partial
        index = _index(directory)
        index[phase3c.PARTIAL_PROOF]["total_cells_written"] = 1920
        _reindex(directory, index)
        self._refused(root, directory, "account for")

    def test_carrying_both_outcome_blocks_is_refused(self, partial):
        root, directory = partial
        index = _index(directory)
        index[phase3c.ZERO_CELL_PROOF] = {"total_cells_written": 0}
        _reindex(directory, index)
        self._refused(root, directory, "an attempt has one outcome")

    def test_the_wrong_schema_for_the_block_is_refused(self, partial):
        root, directory = partial
        index = _index(directory)
        index["schema_version"] = 2
        _reindex(directory, index)
        self._refused(root, directory, "schema 2 records")


class TestTheSupersessionBindsThePartialAttempt:
    def test_a_cell_count_that_disagrees_with_the_attempt_is_refused(
            self, partial):
        """The count is checked against the record, not against zero."""
        root, _directory = partial
        record = phase3c.build_supersession_record()
        entry = next(e for e in record["not_executable"]
                     if e["generation"] == "gen08")
        assert entry["cells_produced"] == 320
        entry["cells_produced"] = 0
        problems = phase3c.attempted_execution_problems(record, root=root)
        assert any("recorded 320" in p for p in problems), problems

    def test_a_coherently_rewritten_member_is_caught_by_the_binding(
            self, partial):
        """The one thing the directory cannot check about itself.

        ``raw_text`` is the model's output: no digest inside the attempt
        constrains what it says, so an edit that also re-files every size,
        digest and count passes ``failed_attempt_problems``. What catches it
        is one level up -- the supersession entry binds the index's own
        SHA-256, and re-filing the index changes it.
        """
        root, directory = partial
        lines = (directory / PARTIAL_MEMBER).read_text().splitlines(True)
        row = json.loads(lines[5])
        row["raw_text"] = "2x4 (0,0,0)\n"
        lines[5] = json.dumps(row) + "\n"
        (directory / PARTIAL_MEMBER).write_text("".join(lines))
        _reseal(directory)

        assert phase3c.failed_attempt_problems(directory, root=root) == []
        problems = phase3c.attempted_execution_problems(
            phase3c.build_supersession_record(), root=root)
        assert any("digests to" in p for p in problems), problems
