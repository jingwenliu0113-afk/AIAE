"""V1: what the visual stress pipeline refuses, and what it re-derives.

The renderer is the thing most worth testing here, because everything else
rests on it: if the same scene and condition can produce two different
images, the ground truth stops being attached to anything and every number
downstream is about an image nobody can reproduce.

The rest is the same discipline as Phase 3C -- provenance, tampering, a
missing image, a wrong label, an existing result, idempotence -- applied to
images instead of samples. And two tests exist purely to keep the honesty
claims honest: that no summary reports a tier it did not run, and that the
report cannot be written with the words that would turn a software render
into a photograph.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval import visual_stress as vs
from src.eval import visual_stress_report as vsr
from src.eval.acceptance import PlanRefused
from src.vision import synthetic

ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_ONLY = "artifact-only:"


# ---------------------------------------------------------------------------
# the renderer
# ---------------------------------------------------------------------------

def test_the_same_scene_and_condition_render_the_same_bytes():
    scene = synthetic.scenes(2)[0]
    for condition in synthetic.condition_matrix():
        first = synthetic.image_digest(scene, condition, seed=7)
        second = synthetic.image_digest(scene, condition, seed=7)
        assert first == second, synthetic.condition_id(condition)


def test_a_different_condition_renders_different_bytes():
    scene = synthetic.scenes(1)[0]
    digests = {synthetic.condition_id(c):
               synthetic.image_digest(scene, c, seed=7)
               for c in synthetic.condition_matrix()}
    assert len(set(digests.values())) == len(digests), (
        "two conditions produced identical images, so one of them is not "
        "doing anything: " + str(digests))


def test_every_image_survives_the_png_round_trip(tmp_path):
    """The digest the manifest pins must be the digest a stored file has.

    Two quality conditions crop to a multiple of their block size, so the
    canvas the layout predicts is not the image that comes out. Deriving the
    digest's size from the layout rather than from the pixels made those
    images look tampered with; this is the test that would have caught it.
    """
    for scene in synthetic.scenes(3):
        for condition in synthetic.condition_matrix():
            array = synthetic.render(scene, condition,
                                     seed=vs.RENDER_SEED)
            path = tmp_path / "x.png"
            path.unlink(missing_ok=True)
            synthetic.write_png(path, array)
            assert vs.stored_digest(path) == synthetic.image_digest(
                scene, condition, seed=vs.RENDER_SEED), (
                f"{scene.scene_id}/{synthetic.condition_id(condition)}")


def test_the_ground_truth_is_the_scene_not_a_reading_of_the_image():
    scene = synthetic.scenes(4)[3]
    truth = scene.ground_truth()
    counted: dict[str, int] = {}
    for p in scene.placements:
        counted[p.part] = counted.get(p.part, 0) + 1
    assert truth["inventory"] == dict(sorted(counted.items()))
    assert truth["n_bricks"] == len(scene.placements)


def test_a_quarter_turn_does_not_change_what_the_part_is():
    """Decision 2: ``1x2`` and ``2x1`` are one inventory line, not two."""
    flat = synthetic.Placement("1x2", "red", 0, 0, turn=0)
    turned = synthetic.Placement("1x2", "red", 0, 0, turn=1)
    assert flat.extents() != turned.extents()
    assert flat.part == turned.part == "1x2"


def test_the_condition_matrix_moves_one_axis_at_a_time():
    matrix = synthetic.condition_matrix()
    assert matrix[0] == synthetic.BASELINE
    for condition in matrix[1:]:
        moved = [k for k, v in condition.items()
                 if v != synthetic.BASELINE[k]]
        assert len(moved) == 1, condition


def test_an_unknown_condition_value_is_refused():
    scene = synthetic.scenes(1)[0]
    with pytest.raises(synthetic.SceneError):
        synthetic.render(scene, {**synthetic.BASELINE,
                                 "lighting": "candlelight"})


def test_every_part_in_the_vocabulary_appears_in_some_scene():
    from src.data.bricks import PART_VOCAB

    seen = {p.part for s in synthetic.scenes(vs.N_SCENES,
                                             seed=vs.RENDER_SEED)
            for p in s.placements}
    assert seen == set(PART_VOCAB)


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------

def test_the_manifest_pins_the_code_that_produced_it():
    manifest = vs.build_manifest(method=vs.METHOD_CV, root=ROOT)
    assert vs.manifest_problems(manifest, root=ROOT) == []
    moved = json.loads(json.dumps(manifest))
    moved["source_manifest"]["src/vision/synthetic.py"] = "0" * 64
    problems = vs.manifest_problems(moved, root=ROOT)
    assert any("not the code this manifest pins" in p for p in problems)


def test_a_manifest_whose_digest_does_not_cover_it_is_refused():
    manifest = vs.build_manifest(method=vs.METHOD_CV, root=ROOT)
    tampered = dict(manifest, n_images=9999)
    problems = vs.manifest_problems(tampered, root=ROOT)
    assert any("manifest_digest does not cover" in p for p in problems)


def test_the_manifest_method_must_be_one_the_ui_has():
    from src.ui.full import RECOGNISE_METHODS

    assert vs.METHODS == RECOGNISE_METHODS
    with pytest.raises(PlanRefused):
        vs.build_manifest(method="whatever", root=ROOT)


def test_the_manifest_declares_v2_and_v3_as_not_run():
    manifest = vs.build_manifest(method=vs.METHOD_CV, root=ROOT)
    assert manifest["tier"] == "V1"
    assert manifest["tiers"]["v2_validated_ai_variations"] == "not run"
    assert manifest["tiers"]["v3_pure_ai_hard_cases"] == "not run"
    assert manifest["provenance"]["ai_model"] is None
    assert manifest["provenance"]["ai_seed"] is None


def test_the_manifest_names_a_v2_rejection_rule_it_never_applied():
    """The schema states the rule now, not when V2 finally runs.

    A rejection concept that only appeared once it was needed would be a
    schema written after the result it describes.
    """
    manifest = vs.build_manifest(method=vs.METHOD_CV, root=ROOT)
    assert any("V2 only, not run" in rule
               for rule in manifest["rejection_criteria"])


# ---------------------------------------------------------------------------
# a small run, end to end
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """Two scenes' worth, rendered, analysed and summarised for real."""
    out = tmp_path_factory.mktemp("visual") / "run"
    manifest = vs.build_manifest(method=vs.METHOD_CV, root=ROOT)
    keep = [c for c in manifest["cases"]
            if c["scene_id"] in ("scene_00", "scene_01")]
    manifest = dict(manifest, cases=keep, n_images=len(keep))
    # Re-digest: a sliced manifest whose digest still covered the whole set
    # would be exactly the tampered manifest the checks now refuse.
    manifest["manifest_digest"] = vs.digest_of(
        {k: v for k, v in manifest.items() if k != "manifest_digest"})
    vs.render_images(out, manifest)
    record = vs.run(out, manifest)
    summary = vs.summarise(record, manifest)
    return out, manifest, record, summary


def test_a_clean_run_accepts_every_image(small_run):
    _out, manifest, record, _summary = small_run
    assert len(record["accepted"]) == manifest["n_images"]
    assert record["rejected"] == []
    assert len(record["results"]) == manifest["n_images"]


def test_a_tampered_image_is_rejected_and_not_scored(small_run, tmp_path):
    """tampering: one pixel changed, and the image leaves the formal rate."""
    out, manifest, _record, _summary = small_run
    import shutil

    copy = tmp_path / "tampered"
    shutil.copytree(out, copy)
    entry = manifest["cases"][0]
    path = copy / entry["member"]
    from PIL import Image

    array = np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.uint8)
    array = array.copy()
    array[0, 0] = ((array[0, 0].astype(int) + 40) % 256).astype(np.uint8)
    Image.fromarray(array).save(str(path))

    accepted, rejected = vs.partition(copy, manifest)
    assert entry["image_id"] in {r["image_id"] for r in rejected}
    assert entry["image_id"] not in {a["image_id"] for a in accepted}
    record = vs.run(copy, manifest)
    assert entry["image_id"] not in {r["image_id"] for r in record["results"]}


def test_a_missing_image_is_rejected_by_name(small_run, tmp_path):
    out, manifest, _record, _summary = small_run
    import shutil

    copy = tmp_path / "missing"
    shutil.copytree(out, copy)
    entry = manifest["cases"][1]
    (copy / entry["member"]).unlink()
    _accepted, rejected = vs.partition(copy, manifest)
    reasons = {r["image_id"]: r["reason"] for r in rejected}
    assert entry["image_id"] in reasons
    assert "not rendered or stored" in reasons[entry["image_id"]]


def test_a_wrong_label_would_change_the_result(small_run):
    """wrong label: the truth is not read from the result being scored."""
    out, manifest, _record, _summary = small_run
    entry = manifest["cases"][0]
    scene = {s.scene_id: s for s in synthetic.scenes(vs.N_SCENES,
                                                     seed=vs.RENDER_SEED)}[
        entry["scene_id"]]
    analysis = vs.analyse_one(out, entry, method=vs.METHOD_CV)
    honest = vs.score_image(entry, analysis, scene)

    wrong_scene = synthetic.Scene(
        scene_id=scene.scene_id,
        placements=tuple(
            synthetic.Placement("2x6", p.colour_id, p.row, p.col, p.turn)
            for p in scene.placements))
    wrong_entry = dict(entry, ground_truth=wrong_scene.ground_truth())
    wrong = vs.score_image(wrong_entry, analysis, wrong_scene)
    assert wrong["true_inventory"] != honest["true_inventory"]
    assert (wrong["inventory_exact_match"]
            != honest["inventory_exact_match"]
            or wrong["count_errors"] != honest["count_errors"])


def test_the_summary_is_re_derived_from_the_per_image_results(small_run):
    _out, manifest, record, summary = small_run
    again = vs.summarise(record, manifest)
    assert vs.summary_identity(again) == vs.summary_identity(summary)


def test_a_changed_per_image_result_changes_the_summary(small_run):
    """A summary that ignored its inputs would pass every other test here."""
    _out, manifest, record, summary = small_run
    edited = json.loads(json.dumps(record))
    for row in edited["results"]:
        row["inventory_exact_match"] = True
    changed = vs.summarise(edited, manifest)
    assert (vs.summary_identity(changed)
            != vs.summary_identity(summary))
    assert changed["overall"]["inventory_exact_match"]["value"] == 1.0


def test_summarising_twice_is_idempotent(small_run):
    _out, manifest, record, _summary = small_run
    first = vs.summarise(record, manifest)
    second = vs.summarise(record, manifest)
    assert first["summary_digest"] == second["summary_digest"]


def test_the_per_axis_baseline_cell_holds_only_baseline_images(small_run):
    """Otherwise the unchanged column is an average over every change."""
    _out, manifest, _record, summary = small_run
    for axis, values in summary["by_axis"].items():
        base = values[manifest["baseline"][axis]]
        assert base["images"] == vs.N_SCENES - 6 or base["images"] == 2, (
            f"{axis} baseline covers {base['images']} images")
        for other_axis, other_values in summary["by_axis"].items():
            if other_axis == axis:
                continue
            assert (base["images"]
                    == other_values[manifest["baseline"][other_axis]][
                        "images"]), (
                "every axis's baseline cell must be the same set of images")


def test_the_summary_reports_no_tier_it_did_not_run(small_run):
    _out, _manifest, _record, summary = small_run
    assert summary["tier"] == "V1"
    assert summary["tiers"]["v2_validated_ai_variations"] == "not run"
    assert summary["tiers"]["v3_pure_ai_hard_cases"] == "not run"
    text = json.dumps(summary, ensure_ascii=False).lower()
    for banned in ("real photograph", "in-the-wild", "physically stable"):
        assert banned not in text.replace(
            summary["note"].lower(), "").replace(
            summary["tiers"]["not_claimed"].lower(), "")


def test_the_ui_flow_is_the_one_the_interface_uses(small_run):
    """Not a reimplementation: the same entry point, the same corrections."""
    out, manifest, _record, _summary = small_run
    from src.ui.full import analyse_photo

    entry = manifest["cases"][0]
    direct = analyse_photo((out / entry["member"]).read_bytes(),
                           mode="multi", method=vs.METHOD_CV)
    through = vs.analyse_one(out, entry, method=vs.METHOD_CV)
    assert [i.box for i in direct.items] == [i.box for i in through.items]


def test_the_oracle_correction_path_is_well_formed(small_run):
    """The plumbing holds when an oracle supplies the answers.

    Not a usability result: the corrections are read out of the ground
    truth, so what this shows is that the edits apply and the adoption
    arrives, never that a person would find the errors to correct.
    """
    _out, _manifest, record, _summary = small_run
    assert all(r["oracle_correction_path_valid"] for r in record["results"])
    for row in record["results"]:
        assert row["oracle_corrected_spec"], row["image_id"]


def test_the_pipeline_refuses_rubbish_by_name(small_run):
    _out, _manifest, record, _summary = small_run
    probes = record["safe_rejection_shared_input_layer"]
    assert probes["all_refused_by_name"] is True
    for name, entry in probes.items():
        if not isinstance(entry, dict):
            continue
        assert entry["refused"], f"{name} was not refused: {entry}"


# ---------------------------------------------------------------------------
# provenance, write-once, and what tampering does to each artefact
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# the execution source closure
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# the three attacks: a self-consistent forgery is still a forgery
# ---------------------------------------------------------------------------
#
# Every one of these rebuilds *all* downstream artefacts and *all* self
# digests, so nothing in the directory disagrees with anything else. That is
# the point: before gen04 the only anchor was internal consistency, and a
# result set written by hand passed every check because every check derived
# from the rows.


def _stage_run(tmp_path, name):
    """A published gen04 run directory, copied somewhere writable."""
    import shutil

    source = ROOT / "runs" / "visual_stress" / f"{vs.GENERATION}_cv"
    if not (source / vs.RESULTS_NAME).is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {source.name} is not published")
    dest = tmp_path / name
    shutil.copytree(source, dest)
    return dest


def _reseal(directory: Path, results: dict, manifest: dict) -> None:
    """Rewrite results, summary, report and both indices, all digests fresh.

    The attacker's job, done properly. Anything left stale here would be
    caught by an internal check and would prove nothing.
    """
    results["results_digest"] = vs.digest_of(vs.results_identity(results))
    summary = vs.summarise(results, manifest)
    successes, failures = vs.case_indices(results)
    body = vsr.render(summary, results, manifest)
    for name, payload in (
            (vs.RESULTS_NAME, results),
            (vs.SUMMARY_NAME, summary),
            (vs.SUCCESS_INDEX_NAME, {
                "kind": "brickagain.visual_stress_success_cases",
                "tier": "V1",
                "manifest_digest": manifest["manifest_digest"],
                "summary_digest": summary["summary_digest"],
                "criterion": "the adopted inventory equals the scene exactly",
                "n": len(successes), "cases": successes}),
            (vs.FAILURE_INDEX_NAME, {
                "kind": "brickagain.visual_stress_failure_cases",
                "tier": "V1",
                "manifest_digest": manifest["manifest_digest"],
                "summary_digest": summary["summary_digest"],
                "criterion": "the adopted inventory differs from the scene",
                "n": len(failures), "cases": failures}),
    ):
        path = directory / name
        path.unlink(missing_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n",
                        encoding="utf-8")
    report = directory / vs.REPORT_NAME
    report.unlink(missing_ok=True)
    report.write_text(body, encoding="utf-8")


def test_a_forged_runtime_in_the_results_is_refused(tmp_path):
    """Attack 1: claim a different model produced these answers.

    Only ``results`` changes; the images, the manifest and the summary's
    arithmetic are untouched, and every digest downstream is rebuilt.
    """
    directory = _stage_run(tmp_path, "forged_runtime")
    manifest = json.loads((directory / vs.MANIFEST_NAME).read_text())
    results = json.loads((directory / vs.RESULTS_NAME).read_text())
    assert vsr.verify(directory, root=ROOT, replay=False)["verified"]

    results["runtime"] = {**results["runtime"],
                          "checkpoint": "runs/vision/some_other_model",
                          "checkpoint_digest": "b" * 64,
                          "device_resolved": "cuda"}
    _reseal(directory, results, manifest)

    outcome = vsr.verify(directory, root=ROOT, replay=False)
    assert not outcome["verified"]
    assert any("runtime" in p or "results" in p
               for p in outcome["problems"]), outcome["problems"]


def test_dropping_an_accepted_image_is_refused(tmp_path):
    """Attack 2: quietly shrink the denominator.

    One image leaves the accepted list and its row leaves the results. Every
    rate is then honestly recomputed over what is left, and every digest is
    rebuilt, so the directory is perfectly self-consistent -- and describes
    a run over 143 images that claims to be a run over 144.
    """
    directory = _stage_run(tmp_path, "dropped_image")
    manifest = json.loads((directory / vs.MANIFEST_NAME).read_text())
    results = json.loads((directory / vs.RESULTS_NAME).read_text())
    victim = next(r["image_id"] for r in results["results"]
                  if not r["inventory_exact_match"])

    results["accepted"] = [a for a in results["accepted"]
                           if a["image_id"] != victim]
    results["results"] = [r for r in results["results"]
                          if r["image_id"] != victim]
    _reseal(directory, results, manifest)

    stored = json.loads((directory / vs.RESULTS_NAME).read_text())
    # Self-consistent: its own digest covers it, and the summary matches.
    assert stored["results_digest"] == vs.digest_of(
        vs.results_identity(stored))
    assert vs.summarise(stored, manifest)["summary_digest"] == \
        json.loads((directory / vs.SUMMARY_NAME).read_text())["summary_digest"]

    outcome = vsr.verify(directory, root=ROOT, replay=False)
    assert not outcome["verified"]
    assert any("accept" in p for p in outcome["problems"]), \
        outcome["problems"]


def test_flipping_an_exact_match_to_true_is_refused(tmp_path):
    """Attack 3: make one image succeed, consistently.

    The row's ``inventory_exact_match`` becomes true, and the fields that
    would have to agree with it -- the predicted inventory, the part and
    colour error lists and their counts -- are edited to match. Then every
    downstream artefact and every digest is rebuilt. Nothing internal
    disagrees.

    What refuses it is the replay: the pinned model, re-run on the same
    stored bytes, does not return that row.
    """
    _needs_checkpoint()
    directory = _stage_run(tmp_path, "flipped_match")
    manifest = json.loads((directory / vs.MANIFEST_NAME).read_text())
    results = json.loads((directory / vs.RESULTS_NAME).read_text())

    row = next(r for r in results["results"]
               if not r["inventory_exact_match"])
    victim = row["image_id"]
    row["inventory_exact_match"] = True
    row["predicted_inventory"] = dict(row["true_inventory"])
    row["count_errors"] = {}
    row["part_errors"] = []
    row["n_part_errors"] = 0
    row["colour_errors"] = []
    row["n_colour_errors"] = 0
    row["false_positives"] = 0
    row["false_negatives"] = 0
    _reseal(directory, results, manifest)

    # The forgery is arithmetically perfect: the counts match the lists,
    # the exact-match flag matches the two inventories, the true-brick count
    # is the scene's. Every model-free layer accepts it, which is the point.
    rebuilt = json.loads((directory / vs.RESULTS_NAME).read_text())
    assert vs.row_metric_problems(rebuilt, manifest) == []
    assert vs.results_problems(rebuilt, manifest) == []
    assert vsr.verify(directory, root=ROOT, replay=False)["verified"], \
        vsr.verify(directory, root=ROOT, replay=False)["problems"]

    # Only the replay refuses, because only the replay asks the model.
    problems = vs.replay_problems(
        directory, manifest,
        json.loads((directory / vs.RESULTS_NAME).read_text()),
        root=ROOT, images=[victim])
    assert any("do not reproduce their stored row" in p
               for p in problems), problems
    outcome = vsr.verify(directory, root=ROOT, replay=True)
    assert not outcome["verified"]
    assert outcome["replayed"]


def test_the_replay_accepts_the_run_it_actually_produced(tmp_path):
    """The other direction: an untouched directory replays clean.

    A check that refuses everything is not a check. This is the positive
    control, over a handful of images so it stays cheap.
    """
    directory = _stage_run(tmp_path, "honest")
    manifest = json.loads((directory / vs.MANIFEST_NAME).read_text())
    results = json.loads((directory / vs.RESULTS_NAME).read_text())
    sample = [a["image_id"] for a in results["accepted"][:6]]
    assert vs.replay_problems(directory, manifest, results, root=ROOT,
                              images=sample) == []


def test_the_summary_names_the_result_set_it_came_from(tmp_path):
    """A summary paired with a different result set is refused."""
    directory = _stage_run(tmp_path, "swapped_results")
    summary_path = directory / vs.SUMMARY_NAME
    summary = json.loads(summary_path.read_text())
    assert summary["results_digest"] == json.loads(
        (directory / vs.RESULTS_NAME).read_text())["results_digest"]

    summary["results_digest"] = "c" * 64
    summary["summary_digest"] = vs.digest_of(vs.summary_identity(summary))
    summary_path.unlink()
    summary_path.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
    outcome = vsr.verify(directory, root=ROOT, replay=False)
    assert not outcome["verified"]
    assert any("different per-image result set" in p
               for p in outcome["problems"]), outcome["problems"]


def test_the_source_manifest_is_the_computed_closure_not_a_list():
    """A list is a promise; a closure is a measurement.

    The old manifest pinned nine hand-written files. The closure is far
    wider, and five of the files it adds -- metrics, ldr, palette, bricks,
    session -- can each change a reported number.

    ``src/ui/app.py`` used to be in here too and is deliberately no longer
    required. It was reached only because the recogniser imported ``UiError``
    from it; the module itself holds the two-page interface, which this run
    never executes and which cannot move a score. Moving the recogniser into
    ``src/ui/photo.py`` and ``UiError`` into ``src/ui/errors.py`` dropped it,
    along with the interface's notice text -- whose correction had twice made
    a rendered corpus unverifiable.

    The floor is expressed against the hand-written list rather than as a
    literal, because a literal goes stale on exactly the refactors that make
    the closure more precise, and then reads as a regression.
    """
    names = set(vs.execution_sources(ROOT))
    assert vs.EXECUTION_ENTRY_POINT in names
    assert len(names) > 3 * len(vs.SOURCE_FILES), len(names)
    for required in ("src/vision/metrics.py", "src/ui/photo.py",
                     "src/rendering/ldr.py", "src/colour/palette.py",
                     "src/data/bricks.py", "src/training/session.py",
                     "scripts/60_visual_stress.py"):
        assert required in names, required
    for declared in (vs.SOURCE_FILES + vs.LEARNED_SOURCE_FILES
                     + vs.REQUIRED_IN_CLOSURE):
        assert declared in names, declared
    # The floor is a check on the measurement, not a substitute for it:
    # the closure has to be strictly wider than everything named by hand.
    named = (set(vs.SOURCE_FILES) | set(vs.LEARNED_SOURCE_FILES)
             | set(vs.REQUIRED_IN_CLOSURE))
    assert named < names, sorted(names - named)
    assert set(vs.source_manifest(ROOT)) == names


def test_editing_a_previously_unpinned_dependency_moves_the_digest(tmp_path):
    """``src/vision/metrics.py`` is the named example. It used to be free.

    The whole tree is copied so nothing here writes into the repository;
    one module the old list omitted is edited, and both the source manifest
    digest and the run manifest digest must move.
    """
    _needs_checkpoint()
    tree = _tree_copy(tmp_path)
    before = vs.build_manifest(method=vs.METHOD_CV, root=tree)

    target = tree / "src" / "vision" / "metrics.py"
    target.write_text(target.read_text() + "\n# a change to the matcher\n",
                      encoding="utf-8")
    after = vs.build_manifest(method=vs.METHOD_CV, root=tree)

    assert before["source_manifest_digest"] != after["source_manifest_digest"]
    assert before["manifest_digest"] != after["manifest_digest"]
    # And the earlier manifest no longer holds against the edited tree.
    problems = vs.manifest_problems(before, root=tree)
    assert any("src/vision/metrics.py" in p for p in problems), problems


def test_a_source_file_that_vanished_fails_closed(tmp_path):
    """A closure that *shrank* is caught by the floor, not tolerated.

    Deleting a module makes the static closure stop reaching it, so a fresh
    manifest would honestly describe a smaller tree and nothing would say
    anything. :data:`vs.REQUIRED_IN_CLOSURE` is why that is a refusal: it
    is checked against the measurement, and is not what produces it.
    """
    tree = _tree_copy(tmp_path)
    manifest = vs.build_manifest(method=vs.METHOD_CV, root=tree)
    (tree / "src" / "vision" / "metrics.py").unlink()
    with pytest.raises(PlanRefused) as exc:
        vs.execution_sources(tree)
    assert "src/vision/metrics.py" in str(exc.value)
    problems = vs.manifest_problems(manifest, root=tree)
    assert any("cannot be computed here" in p for p in problems), problems


def test_a_source_file_that_appeared_fails_closed(tmp_path):
    """A module the closure reaches that the manifest never named."""
    tree = _tree_copy(tmp_path)
    manifest = vs.build_manifest(method=vs.METHOD_CV, root=tree)
    extra = tree / "src" / "vision" / "extra_step.py"
    extra.write_text("VALUE = 1\n", encoding="utf-8")
    reached = tree / "src" / "vision" / "detect.py"
    reached.write_text("from src.vision import extra_step  # noqa: F401\n"
                       + reached.read_text(), encoding="utf-8")

    now = set(vs.execution_sources(tree))
    assert "src/vision/extra_step.py" in now
    problems = vs.manifest_problems(manifest, root=tree)
    assert any("files differ" in p for p in problems), problems


def test_a_declared_module_the_closure_cannot_reach_is_refused(tmp_path,
                                                               monkeypatch):
    tree = _tree_copy(tmp_path)
    monkeypatch.setattr(vs, "SOURCE_FILES",
                        vs.SOURCE_FILES + ("src/vision/net.py",))
    with pytest.raises(PlanRefused) as exc:
        vs.execution_sources(tree)
    assert "src/vision/net.py" in str(exc.value)


def _tree_copy(tmp_path) -> Path:
    """The parts of the tree the closure reads, copied somewhere writable."""
    import shutil

    tree = tmp_path / "tree"
    (tree / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / vs.EXECUTION_ENTRY_POINT,
                 tree / vs.EXECUTION_ENTRY_POINT)
    shutil.copytree(ROOT / "src", tree / "src")
    for name in ("runs",):
        source = ROOT / name
        if source.is_dir():
            (tree / name).symlink_to(source)
    return tree


# ---------------------------------------------------------------------------
# the learned runtime is the manifest's, not the command line's
# ---------------------------------------------------------------------------

def test_a_checkpoint_other_than_the_pinned_one_is_refused(tmp_path):
    """manifest pins A, the run is handed B."""
    _needs_checkpoint()
    import shutil

    other = tmp_path / "other_checkpoint"
    shutil.copytree(CHECKPOINT, other)
    manifest = vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT,
                                 checkpoint=str(CHECKPOINT))
    problems = vs.runtime_problems(manifest, checkpoint=str(other), root=ROOT)
    assert any("the manifest pins" in p for p in problems), problems
    with pytest.raises(PlanRefused):
        vs.run(tmp_path, manifest, checkpoint=str(other), root=ROOT)


def test_a_device_other_than_the_pinned_one_is_refused(tmp_path):
    """manifest pins one device, the run is handed another."""
    _needs_checkpoint()
    manifest = vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT,
                                 checkpoint=str(CHECKPOINT), device="mps")
    assert manifest["recognition"]["device_requested"] == "mps"
    problems = vs.runtime_problems(manifest, device="cpu", root=ROOT)
    assert any("device 'cpu'" in p and "mps" in p for p in problems), problems
    with pytest.raises(PlanRefused):
        vs.run(tmp_path, manifest, device="cpu", root=ROOT)


def test_checkpoint_bytes_that_moved_are_refused(tmp_path):
    _needs_checkpoint()
    import shutil

    staged = tmp_path / "staged"
    shutil.copytree(CHECKPOINT, staged)
    manifest = vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT,
                                 checkpoint=str(staged))
    assert vs.runtime_problems(manifest, root=ROOT) == []
    weights = staged / "vision_head.pt"
    weights.write_bytes(weights.read_bytes() + b"\x00")
    problems = vs.runtime_problems(manifest, root=ROOT)
    assert any("different model" in p for p in problems), problems


def test_a_selection_record_whose_own_checks_failed_is_refused(tmp_path):
    """Digest correct, audit failed. Hashing it was all the old code did."""
    _needs_checkpoint()
    import shutil

    staged = tmp_path / "unaudited"
    shutil.copytree(CHECKPOINT, staged)
    record_path = staged / "selection_record.json"
    record = json.loads(record_path.read_text())
    record["checks"]["selected_epoch_is_the_argmin_of_the_epoch_log"] = False
    record_path.write_text(json.dumps(record, indent=2) + "\n")

    # The manifest is built *after* the edit, so its digest covers the
    # failed record: nothing here is internally inconsistent.
    manifest = vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT,
                                 checkpoint=str(staged))
    problems = vs.checkpoint_problems(
        staged, manifest["recognition"]["checkpoint_manifest"])
    assert any("own checks did not pass" in p for p in problems), problems
    assert vs.manifest_problems(manifest, root=ROOT) != []
    assert vs.runtime_problems(manifest, root=ROOT) != []


def test_a_selection_record_that_names_problems_is_refused(tmp_path):
    _needs_checkpoint()
    import shutil

    staged = tmp_path / "problematic"
    shutil.copytree(CHECKPOINT, staged)
    record_path = staged / "selection_record.json"
    record = json.loads(record_path.read_text())
    record["problems"] = ["the weights digest did not match"]
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    problems = vs.checkpoint_problems(staged)
    assert any("names problems of its own" in p for p in problems), problems


def test_a_selection_record_whose_epoch_disagrees_is_refused(tmp_path):
    _needs_checkpoint()
    import shutil

    staged = tmp_path / "wrong_epoch"
    shutil.copytree(CHECKPOINT, staged)
    record_path = staged / "selection_record.json"
    record = json.loads(record_path.read_text())
    record["epoch_selection"]["selected_epoch_rederived"] = 3
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    problems = vs.checkpoint_problems(staged)
    assert any("re-derives 3" in p for p in problems), problems


def test_a_run_summary_that_disagrees_with_the_manifest_is_refused(tmp_path):
    _needs_checkpoint()
    import shutil

    staged = tmp_path / "bad_summary"
    shutil.copytree(CHECKPOINT, staged)
    path = staged / "run_summary.json"
    summary = json.loads(path.read_text())
    summary["selected_epoch"] = 99
    path.write_text(json.dumps(summary, indent=2) + "\n")
    problems = vs.checkpoint_problems(staged)
    assert any("selected_epoch" in p for p in problems), problems


def test_the_published_result_sets_record_the_runtime_they_used(tmp_path):
    """What the answers came from, stated, and taken from the manifest."""
    _needs_checkpoint()
    manifest = vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT,
                                 checkpoint=str(CHECKPOINT))
    assert vs.runtime_problems(manifest, checkpoint=str(CHECKPOINT),
                               device=None, root=ROOT) == []

    directory = ROOT / "runs" / "visual_stress" / f"{vs.GENERATION}_learned"
    results = directory / vs.RESULTS_NAME
    if not results.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {directory.name} is not published")
    record = json.loads(results.read_text())
    runtime = record["runtime"]
    stored = json.loads((directory / vs.MANIFEST_NAME).read_text())
    recognition = stored["recognition"]
    assert runtime["checkpoint"] == recognition["checkpoint"]
    assert runtime["checkpoint_digest"] == \
        recognition["checkpoint_manifest"]["checkpoint_digest"]
    assert runtime["device_resolved"] == \
        recognition["inference_environment"]["device_resolved"]
    # And it is part of what makes two result sets the same result.
    assert "runtime" in vs.results_identity(record)


CHECKPOINT = ROOT / "runs" / "vision" / "classifier"


def _needs_checkpoint():
    if not (CHECKPOINT / "vision_head.pt").is_file():
        pytest.skip(f"{ARTIFACT_ONLY} the fitted checkpoint is not published")


def test_a_learned_manifest_must_name_its_checkpoint():
    """A method name is not a model. Two runs against different weights
    used to produce two manifests that agreed."""
    with pytest.raises(PlanRefused) as exc:
        vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT)
    assert "needs a checkpoint" in str(exc.value)


def test_the_cv_manifest_may_not_name_a_checkpoint():
    _needs_checkpoint()
    with pytest.raises(PlanRefused) as exc:
        vs.build_manifest(method=vs.METHOD_CV, root=ROOT,
                          checkpoint=str(CHECKPOINT))
    assert "loads no checkpoint" in str(exc.value)


def test_the_learned_manifest_pins_the_whole_checkpoint():
    _needs_checkpoint()
    manifest = vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT,
                                 checkpoint=str(CHECKPOINT))
    pinned = manifest["recognition"]["checkpoint_manifest"]
    assert set(pinned["files"]) == set(vs.CHECKPOINT_FILES)
    for name, entry in pinned["files"].items():
        assert entry["sha256"] == vs.sha256_file(CHECKPOINT / name)
        assert entry["bytes"] > 0
    for field in ("code_sha256", "data_manifest_sha256",
                  "split_manifest_sha256", "selected_epoch",
                  "selection_criterion", "seed", "fitted_on_device",
                  "dependencies", "preprocessing", "class_order"):
        assert pinned[field] is not None, field
    assert vs.manifest_problems(manifest, root=ROOT) == []


def test_the_learned_manifest_pins_the_inference_environment():
    _needs_checkpoint()
    manifest = vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT,
                                 checkpoint=str(CHECKPOINT), device="cpu")
    env = manifest["recognition"]["inference_environment"]
    assert env["device_requested"] == "cpu"
    assert env["os_system"] and env["python"] and env["torch"]
    assert manifest["recognition"]["device_requested"] == "cpu"
    # The fit ran elsewhere; the manifest must not conflate the two.
    fitted = manifest["recognition"]["checkpoint_manifest"]["fitted_on_device"]
    assert fitted == "cuda"


def test_the_device_is_part_of_the_manifest_identity():
    _needs_checkpoint()
    a = vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT,
                          checkpoint=str(CHECKPOINT), device="cpu")
    b = vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT,
                          checkpoint=str(CHECKPOINT), device="mps")
    assert a["manifest_digest"] != b["manifest_digest"]


def test_the_checkpoint_is_part_of_the_manifest_identity(tmp_path):
    _needs_checkpoint()
    import shutil

    other = tmp_path / "other_checkpoint"
    shutil.copytree(CHECKPOINT, other)
    (other / "vision_head.pt").write_bytes(b"different weights")
    a = vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT,
                          checkpoint=str(CHECKPOINT))
    b = vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT,
                          checkpoint=str(other))
    assert a["manifest_digest"] != b["manifest_digest"]


def test_a_swapped_checkpoint_fails_the_manifest(tmp_path):
    """The counter-example: the weights moved and the manifest did not."""
    _needs_checkpoint()
    import shutil

    staged = tmp_path / "checkpoint"
    shutil.copytree(CHECKPOINT, staged)
    manifest = vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT,
                                 checkpoint=str(staged))
    assert vs.manifest_problems(manifest, root=ROOT) == []
    (staged / "vision_head.pt").write_bytes(b"swapped")
    problems = vs.manifest_problems(manifest, root=ROOT)
    assert any("different model" in p for p in problems), problems
    # And the cross-check catches it a second, independent way: the run
    # summary still names the weights the fit produced.
    assert any("weights_sha256" in p for p in problems), problems


def test_one_closure_covers_both_recognisers_and_says_which_runs():
    """The pinned set is the closure; the *method* is what narrows.

    It used to be the other way round -- the CV manifest pinned nine files
    and the learned one thirteen -- which meant each manifest left the other
    branch's code unpinned. A static closure reaches both branches, so both
    are digested, and which subset a method executes is stated by name.
    """
    _needs_checkpoint()
    cv = vs.build_manifest(method=vs.METHOD_CV, root=ROOT)
    learned = vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT,
                                checkpoint=str(CHECKPOINT))
    assert set(cv["source_manifest"]) == set(learned["source_manifest"])
    for name in vs.LEARNED_SOURCE_FILES:
        assert name in cv["source_manifest"]
        assert name in learned["execution_source"]["method_source_files"]
        assert name not in cv["execution_source"]["method_source_files"]
    # The two manifests still differ: the model, not the file list.
    assert cv["manifest_digest"] != learned["manifest_digest"]


def test_an_edited_learned_module_fails_the_learned_manifest_only(tmp_path):
    _needs_checkpoint()
    learned = vs.build_manifest(method=vs.METHOD_LEARNED, root=ROOT,
                                checkpoint=str(CHECKPOINT))
    moved = json.loads(json.dumps(learned))
    moved["source_manifest"]["src/vision/model.py"] = "0" * 64
    problems = vs.manifest_problems(moved, root=ROOT)
    assert any("not the code this manifest pins" in p for p in problems)


# --- write-once, per artefact ---------------------------------------------

@pytest.fixture
def published(tmp_path, small_run):
    """A complete published directory: manifest, results, summary, report."""
    import shutil

    out, manifest, record, summary = small_run
    copy = tmp_path / "published"
    shutil.copytree(out, copy)
    for name, payload in ((vs.MANIFEST_NAME, manifest),
                          (vs.RESULTS_NAME, record),
                          (vs.SUMMARY_NAME, summary)):
        (copy / name).write_text(json.dumps(payload), encoding="utf-8")
    vsr.write_report(copy)
    return copy, manifest, record, summary


def test_a_published_directory_verifies(published):
    copy, _manifest, _record, _summary = published
    outcome = vsr.verify(copy)
    assert outcome["verified"] is True, outcome["problems"]


@pytest.mark.parametrize("name", ["report", "success", "failure"])
def test_rewriting_a_published_artefact_is_refused(published, name):
    """These three used plain writes; a second run replaced them silently."""
    copy, _manifest, record, _summary = published
    if name == "report":
        path = copy / vs.REPORT_NAME
        path.write_text(path.read_text(encoding="utf-8") + "\nedited\n",
                        encoding="utf-8")
    else:
        which = (vs.SUCCESS_INDEX_NAME if name == "success"
                 else vs.FAILURE_INDEX_NAME)
        body = json.loads((copy / which).read_text())
        body["n"] = 999
        (copy / which).write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(PlanRefused) as exc:
        vsr.write_report(copy)
    assert "not rewritten" in str(exc.value)


def test_rewriting_a_published_artefact_with_identical_bytes_is_accepted(
        published):
    """Idempotence: a rerun that changes nothing rewrites nothing."""
    copy, _manifest, _record, _summary = published
    before = {p.name: vs.sha256_file(p) for p in copy.glob("*.json")}
    before[vs.REPORT_NAME] = vs.sha256_file(copy / vs.REPORT_NAME)
    written = vsr.write_report(copy)
    assert all(entry["rewritten"] is False for entry in written.values()), \
        written
    after = {p.name: vs.sha256_file(p) for p in copy.glob("*.json")}
    after[vs.REPORT_NAME] = vs.sha256_file(copy / vs.REPORT_NAME)
    assert before == after


# --- tampering, per artefact ----------------------------------------------

def test_a_tampered_summary_fails_verify(published):
    copy, _manifest, _record, summary = published
    body = json.loads(json.dumps(summary))
    body["overall"]["inventory_exact_match"]["value"] = 0.99
    (copy / vs.SUMMARY_NAME).write_text(json.dumps(body), encoding="utf-8")
    outcome = vsr.verify(copy)
    assert outcome["verified"] is False
    assert any("not what the results imply" in p
               for p in outcome["problems"])


def test_a_tampered_accepted_list_fails_verify(published):
    """Comparing ``results`` alone let this through."""
    copy, _manifest, record, _summary = published
    body = json.loads(json.dumps(record))
    body["accepted"] = body["accepted"][:-1]
    (copy / vs.RESULTS_NAME).write_text(json.dumps(body), encoding="utf-8")
    outcome = vsr.verify(copy)
    assert outcome["verified"] is False


def test_a_tampered_rejected_list_fails_verify(published):
    copy, _manifest, record, _summary = published
    body = json.loads(json.dumps(record))
    body["rejected"] = [{"image_id": "invented", "member": "x",
                         "reason": "invented"}]
    (copy / vs.RESULTS_NAME).write_text(json.dumps(body), encoding="utf-8")
    outcome = vsr.verify(copy)
    assert outcome["verified"] is False


def test_a_tampered_safe_rejection_probe_fails_verify(published):
    """A run claiming four refusals it did not get."""
    copy, _manifest, record, _summary = published
    body = json.loads(json.dumps(record))
    probes = body["safe_rejection_shared_input_layer"]
    probes["not_an_image"] = {"refused": False, "by": None,
                              "message": "an inventory was produced"}
    probes["all_refused_by_name"] = True
    (copy / vs.RESULTS_NAME).write_text(json.dumps(body), encoding="utf-8")
    outcome = vsr.verify(copy)
    assert outcome["verified"] is False


def test_tampered_summary_metadata_fails_verify(published):
    """The tier declaration and the blocker are part of the identity."""
    copy, _manifest, _record, summary = published
    body = json.loads(json.dumps(summary))
    body["tiers"] = dict(body["tiers"],
                         v2_validated_ai_variations="complete")
    (copy / vs.SUMMARY_NAME).write_text(json.dumps(body), encoding="utf-8")
    outcome = vsr.verify(copy)
    assert outcome["verified"] is False


def test_a_tampered_manifest_fails_verify(published):
    copy, manifest, _record, _summary = published
    body = json.loads(json.dumps(manifest))
    body["n_images"] = 9999
    (copy / vs.MANIFEST_NAME).write_text(json.dumps(body), encoding="utf-8")
    outcome = vsr.verify(copy)
    assert outcome["verified"] is False


def test_a_tampered_report_fails_verify(published):
    copy, _manifest, _record, _summary = published
    path = copy / vs.REPORT_NAME
    path.write_text(path.read_text(encoding="utf-8").replace(
        "0.0%", "99.9%"), encoding="utf-8")
    outcome = vsr.verify(copy)
    assert outcome["verified"] is False
    assert any("stored report" in p for p in outcome["problems"])


def test_a_tampered_index_fails_verify(published):
    copy, _manifest, _record, _summary = published
    path = copy / vs.FAILURE_INDEX_NAME
    body = json.loads(path.read_text())
    body["cases"] = body["cases"][:-1]
    body["n"] = len(body["cases"])
    path.write_text(json.dumps(body), encoding="utf-8")
    outcome = vsr.verify(copy)
    assert outcome["verified"] is False
    assert any(vs.FAILURE_INDEX_NAME in p for p in outcome["problems"])


def test_a_deleted_index_fails_verify(published):
    copy, _manifest, _record, _summary = published
    (copy / vs.SUCCESS_INDEX_NAME).unlink()
    outcome = vsr.verify(copy)
    assert outcome["verified"] is False


def test_a_tampered_image_fails_verify(published):
    """The rejected count and the images on disk have to agree."""
    copy, manifest, _record, _summary = published
    from PIL import Image

    entry = manifest["cases"][0]
    path = copy / entry["member"]
    array = np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.uint8)
    array = array.copy()
    array[0, 0] = ((array[0, 0].astype(int) + 40) % 256).astype(np.uint8)
    Image.fromarray(array).save(str(path))
    outcome = vsr.verify(copy)
    assert outcome["verified"] is False


# --- the honest names ------------------------------------------------------

def test_the_correction_metric_is_named_for_what_it_measures(small_run):
    """It is an oracle path check, not a human success rate."""
    _out, _manifest, record, summary = small_run
    assert "oracle_correction_path_valid" in summary["overall"]
    assert "correction_flow_completed" not in summary["overall"]
    for row in record["results"]:
        assert "oracle_correction_path_valid" in row
        assert "correction_flow_completed" not in row


def test_the_safe_rejection_field_says_it_is_the_shared_layer(small_run):
    _out, _manifest, record, _summary = small_run
    assert "safe_rejection_shared_input_layer" in record
    assert "safe_rejection" not in record


def test_the_docstring_disclaims_the_oracle_correction_metric():
    """The name is not enough; the module has to say what it is not."""
    text = (ROOT / "src" / "eval" / "visual_stress.py").read_text(
        encoding="utf-8")
    low = " ".join(text.lower().split())
    assert "not a usability result and not a human success rate" in low
    assert "never as \"users succeed\"" in low
    assert "shared input layer" in low
    assert "no person was involved" in low


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def test_the_report_renders_and_claims_no_photograph(small_run, tmp_path):
    import shutil

    out, manifest, record, summary = small_run
    copy = tmp_path / "report"
    shutil.copytree(out, copy)
    (copy / vs.MANIFEST_NAME).write_text(json.dumps(manifest),
                                         encoding="utf-8")
    (copy / vs.RESULTS_NAME).write_text(json.dumps(record), encoding="utf-8")
    (copy / vs.SUMMARY_NAME).write_text(json.dumps(summary), encoding="utf-8")

    written = vsr.write_report(copy)
    body = (copy / vs.REPORT_NAME).read_text(encoding="utf-8")
    assert written["report"]["rewritten"] is True
    assert vs.forbidden_terms_in(body) == []
    assert "軟體渲染圖" in body
    assert "不是照片" in body
    assert summary["manifest_digest"] in body
    assert "V2 與 V3 沒有執行" in body

    successes = json.loads((copy / vs.SUCCESS_INDEX_NAME).read_text())
    failures = json.loads((copy / vs.FAILURE_INDEX_NAME).read_text())
    assert successes["n"] + failures["n"] == len(record["results"])
    known = {c["image_id"] for c in manifest["cases"]}
    for entry in successes["cases"] + failures["cases"]:
        assert entry["image_id"] in known
        assert entry["member"], "an index entry must point at its image"


def test_a_report_over_a_fabricated_summary_is_refused(small_run, tmp_path):
    """existing-result corruption: the summary must match its own inputs."""
    import shutil

    out, manifest, record, summary = small_run
    copy = tmp_path / "fabricated"
    shutil.copytree(out, copy)
    fake = json.loads(json.dumps(summary))
    fake["overall"]["inventory_exact_match"]["value"] = 0.99
    (copy / vs.MANIFEST_NAME).write_text(json.dumps(manifest),
                                         encoding="utf-8")
    (copy / vs.RESULTS_NAME).write_text(json.dumps(record), encoding="utf-8")
    (copy / vs.SUMMARY_NAME).write_text(json.dumps(fake), encoding="utf-8")
    with pytest.raises(PlanRefused) as exc:
        vsr.write_report(copy)
    assert "not what the per-image results imply" in str(exc.value)


def test_the_forbidden_scan_catches_the_claim():
    body = "這些 real photographs 顯示 100% 準確率。"
    assert "real photographs" in vs.forbidden_terms_in(body)
    assert vs.forbidden_terms_in("這是一份軟體渲染圖的結果。") == []


def test_a_report_containing_the_claim_is_refused_not_published(small_run,
                                                                tmp_path,
                                                                monkeypatch):
    """The real defence: the scan runs on the rendered text before it lands.

    Scanning the *sources* instead would be worse than useless -- this
    module's own docstrings say "not real-photograph accuracy", which is the
    honest sentence, and a scan that failed on it would push the denial out
    of the code. What must not exist is a published report making the claim,
    so that is what is tested: a renderer that emits it produces no file.
    """
    import shutil

    out, manifest, record, summary = small_run
    copy = tmp_path / "overclaim"
    shutil.copytree(out, copy)
    for name, payload in ((vs.MANIFEST_NAME, manifest),
                          (vs.RESULTS_NAME, record),
                          (vs.SUMMARY_NAME, summary)):
        (copy / name).write_text(json.dumps(payload), encoding="utf-8")

    real_render = vsr.render
    monkeypatch.setattr(
        vsr, "render",
        lambda *a, **k: real_render(*a, **k)
        + "\n這證明了 real photograph 準確率。\n")
    with pytest.raises(PlanRefused) as exc:
        vsr.write_report(copy)
    assert "may not claim" in str(exc.value)
    assert not (copy / vs.REPORT_NAME).exists(), (
        "a refused report must leave no file behind")


# ---------------------------------------------------------------------------
# the CLI's guards
# ---------------------------------------------------------------------------

def _cli():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "visual_cli", ROOT / "scripts" / "60_visual_stress.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_learned_method_needs_a_checkpoint(capsys, tmp_path):
    cli = _cli()
    # A real temporary directory rather than a literal path: the public
    # snapshot's audit reads an absolute path in a published file as a
    # personal path, and it is right to.
    assert cli.main(["--manifest", "--method", vs.METHOD_LEARNED,
                     "--out-dir", str(tmp_path / "nowhere")]) == 2
    assert "--checkpoint" in capsys.readouterr().err


def test_exactly_one_mode(capsys):
    cli = _cli()
    assert cli.main(["--manifest", "--report"]) == 2
    assert cli.main([]) == 2


def test_a_run_without_a_manifest_is_refused(capsys, tmp_path):
    cli = _cli()
    assert cli.main(["--run", "--out-dir", str(tmp_path)]) == 2
    assert "run --manifest first" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# the stored run, if it is in this tree
# ---------------------------------------------------------------------------

STORED = ROOT / "runs" / "visual_stress"
REPORTS = ROOT / "data" / "reports" / "60_visual_stress"


@pytest.mark.parametrize("name", ["cv", "learned"])
def test_the_stored_run_verifies_in_full(name):
    """Everything, re-derived: manifest, summary, report, both indices."""
    directory = STORED / f"{vs.GENERATION}_{name}"
    if not (directory / vs.SUMMARY_NAME).is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {directory.name} is not published")
    outcome = vsr.verify(directory)
    assert outcome["verified"] is True, outcome["problems"]


@pytest.mark.parametrize("name", ["cv", "learned"])
def test_the_stored_images_still_hash_to_what_the_manifest_pins(name):
    directory = STORED / f"{vs.GENERATION}_{name}"
    if not (directory / vs.MANIFEST_NAME).is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {directory.name} is not published")
    manifest = json.loads((directory / vs.MANIFEST_NAME).read_text())
    _accepted, rejected = vs.partition(directory, manifest)
    assert rejected == [], rejected


def test_every_superseded_generation_is_recorded_and_kept():
    """gen01 and gen02 stay on disk and stay unusable, both on purpose."""
    assert vs.SUPERSEDED
    names = [e["generation"] for e in vs.SUPERSEDED]
    # Newest first, and every one of them still on disk with the digests
    # this module records. Written generically so the next bump does not
    # need this test edited -- only the list it reads.
    assert names == sorted(names, reverse=True), names
    assert vs.GENERATION not in names
    assert names[0] < vs.GENERATION
    for entry in vs.SUPERSEDED:
        assert entry["superseded_because"] and entry["not_deleted_because"]
        assert "measurements_unchanged" in entry
        prefix = "" if entry["generation"] == "gen01" \
            else f"{entry['generation']}_"
        for method, key in (("cv", "cv_manifest_digest"),
                            ("learned", "learned_manifest_digest")):
            path = REPORTS / f"{prefix}{method}__visual_manifest.json"
            if not path.is_file():
                continue
            assert json.loads(path.read_text())["manifest_digest"] == \
                entry[key], path.name
    entry = next(e for e in vs.SUPERSEDED if e["generation"] == "gen01")
    old = REPORTS / "cv__summary.json"
    if not old.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} the gen01 reports are not published")
    stored = json.loads(old.read_text())
    assert stored["manifest_digest"] == entry["cv_manifest_digest"]
    # It predates the rename, which is why it cannot be re-derived by this
    # code -- and why it is superseded rather than corrected in place.
    assert "correction_flow_completed" in stored["overall"]
    assert "oracle_correction_path_valid" not in stored["overall"]


# -- the generation is data, and that is what keeps two archives apart -------
#
# ``src/eval/visual_stress.py`` is one of the 60 files the Phase 3C pack
# carries, and Phase 3C gen09's pack evidence pins every one by SHA-256. While
# the generation was a literal in that module, bumping V1 edited a file that
# archive pins -- the mirror of the cycle ``src/training/pack.py`` already cut
# by moving the Phase 3C generation into ``gpu_plans/phase3c_staged.json``.
# These tests pin both halves of the fix so neither can be quietly undone.


def test_the_generation_is_declared_in_data_not_in_the_module():
    """A literal here re-creates the cycle that retired gen04 and gen05."""
    source = (ROOT / "src" / "eval" / "visual_stress.py").read_text(
        encoding="utf-8")
    for literal in ('GENERATION = "gen', "GENERATION = 'gen"):
        assert literal not in source, (
            f"the generation is a literal again ({literal!r}); bumping it "
            "then edits a file the Phase 3C pack pins")
    assert 'GENERATION: str = _DECLARATION["current"]' in source
    assert vs.GENERATION == json.loads(
        (ROOT / vs.GENERATION_DECLARATION).read_text(encoding="utf-8")
    )["current"]


def test_the_declaration_is_outside_every_pack_that_pins_this_module():
    """Otherwise the generation only moved house and the cycle is intact."""
    from src.training import pack as training_pack

    packed = {e["path"] for e in training_pack.manifest(ROOT)["include"]}
    assert "src/eval/visual_stress.py" in packed, (
        "this test's premise is gone: the module is no longer packed, so the "
        "coupling it guards against no longer exists and this should be "
        "reconsidered rather than deleted")
    assert vs.GENERATION_DECLARATION not in packed, (
        f"{vs.GENERATION_DECLARATION} travels in the same pack as the module "
        "it was extracted from, so bumping the generation still moves the "
        "pack digest")
    assert not any(p.startswith("data/") for p in packed), sorted(
        p for p in packed if p.startswith("data/"))


@pytest.mark.parametrize("break_it,expected", [
    (lambda d: d.pop("kind"), "kind"),
    (lambda d: d.__setitem__("current", ""), "current"),
    (lambda d: d["recogniser"].__setitem__("source_files", []), "source_files"),
    (lambda d: d["superseded"].insert(
        0, {"generation": d["current"]}), "superseded"),
])
def test_a_broken_declaration_is_refused_rather_than_half_read(
        tmp_path, break_it, expected):
    """A partial read would silently pin the wrong files."""
    body = json.loads(
        (ROOT / vs.GENERATION_DECLARATION).read_text(encoding="utf-8"))
    break_it(body)
    target = tmp_path / vs.GENERATION_DECLARATION
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(vs.PlanRefused) as excinfo:
        vs._read_generation_declaration(tmp_path)
    assert expected in str(excinfo.value), str(excinfo.value)
