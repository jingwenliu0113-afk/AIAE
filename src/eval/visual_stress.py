"""V1: the synthetic visual stress test, frozen before it produces a number.

The question is narrow on purpose: **does the recognition path and the UI
flow around it hold up when the image conditions change?** Not whether it
works on photographs -- nothing here is a photograph -- and not whether a
model built from these numbers would assemble anything.

---------------------------------------------------------------------------
The three tiers, and why only one of them is here
---------------------------------------------------------------------------

===  ==============================================  ====================
tier  what it is                                      counts towards
===  ==============================================  ====================
V1   deterministic renders with a constructed truth   the formal accuracy
V2   AI variations of V1, each checked for drift      the formal accuracy
V3   pure AI hard cases, ground truth unreliable      stress only, never
===  ==============================================  ====================

**Only V1 exists.** This project has no authorised, reproducible,
version-pinnable image generation mechanism: no diffusion weights, no image
API client, nothing in either requirements file. Producing V2 or V3 would
mean adding a credential, sending data to an unauthorised service,
downloading a large model, or fabricating images and calling them generated
-- and all four are refused. :data:`V2_V3_BLOCKED` records that, the report
prints it, and a test asserts no summary claims a tier it did not run.

---------------------------------------------------------------------------
What V1 is and is not
---------------------------------------------------------------------------

The images are **software renders**: flat top-down drawings of studded
rectangles, drawn by arithmetic in :mod:`src.vision.synthetic`. Aspect, stud
pitch and colour are right, because those are what the recogniser measures.
Everything else about a photograph -- optics, real materials, real light, a
real table -- is absent.

So a rate here says *the recogniser handled this rendered condition this
often*. It does not say what would happen to a photograph, and the report
refuses to render if it contains the words that would claim otherwise.

The ground truth is not labelled, it is **constructed**: the scene says a
2x4 in orange is at row 1 column 1, and the renderer draws that. There is no
labelling error to account for, which is exactly why V1 and not V3 is the
quantitative tier.

---------------------------------------------------------------------------
What is measured, per image
---------------------------------------------------------------------------

Everything goes through :func:`src.ui.photo.analyse_photo` -- the same
function the interface calls -- so the UI flow is exercised rather than
described. Then:

* **inventory exact match**: adopted parts equal the scene's parts, exactly.
* **part errors**: per brick, matched by box overlap against the truth.
* **colour errors**: the colour the reader returned against the one drawn.
* **count errors**: per part, adopted count minus true count.
* **false positives**: detections matching no true brick.
* **false negatives**: true bricks no detection matched.
* **UI load**: whether ``analyse_photo`` returned editable items at all.
* **oracle correction path valid**: whether the *data structures* survive
  being corrected. The ground truth is injected automatically -- every item
  is edited to the right part and colour, surplus items are deleted, missing
  ones are added -- and the result is put through
  :func:`src.ui.corrections.adopt` and
  :func:`~src.ui.corrections.inventory_spec`. What it establishes is that
  the correction path is *well-formed*: the edits apply, the adoption
  arrives at the scene's inventory, and the spec string parses.

  **It is not a usability result and not a human success rate.** No person
  was involved, nobody had to find the errors, and the oracle never had to
  decide anything. A real person would have to notice that a 1x1 is missing
  entirely -- and the false-negative figures say that is exactly what this
  path does not show them. Read as "the plumbing holds", never as "users
  succeed".
* **safe rejection**: whether four deliberately malformed inputs are refused
  by name rather than silently producing an inventory. This tests the
  **shared input layer** -- ``decode_image`` and the ``analyse_photo``
  guards, which run before any recogniser is chosen -- so it is one result
  about a common path, not a separate test of each model. Both runs report
  it and both report the same thing, necessarily.

---------------------------------------------------------------------------
What a manifest pins, and what it decides
---------------------------------------------------------------------------

The manifest is written before any image is scored and is the authority for
two things a run used to be free to contradict.

**The code.** ``source_manifest`` is the SHA-256 of every module in the
*import closure* of :data:`EXECUTION_ENTRY_POINT` -- computed, not listed.
The previous design named nine files by hand while the closure is 55, so
``src/vision/metrics.py``, ``src/ui/app.py``, ``src/rendering/ldr.py``,
``src/colour/palette.py``, ``src/data/bricks.py`` and
``src/training/session.py`` could all change a reported number without
moving a digest.

**The model and the device.** :func:`run` takes its checkpoint and its
device from the manifest. ``--checkpoint`` and ``--device`` still exist and
are still compared -- a mismatch is a named refusal rather than a silent
override -- but they do not decide anything. A learned checkpoint is
admitted only if its selection record's own audit holds, its stated epoch is
the one it re-derives, and its run summary agrees with the fitting manifest
beside it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.eval.acceptance import PlanRefused, digest_obj
from src.training.session import sha256_file
from src.vision import synthetic

ROOT = Path(__file__).resolve().parents[2]

KIND = "brickagain.visual_stress"
CONTRACT_VERSION = 1
MANIFEST_KIND = "brickagain.visual_stress_manifest"
RESULTS_KIND = "brickagain.visual_stress_results"
SUMMARY_KIND = "brickagain.visual_stress_summary"

DEFAULT_DIR = "runs/visual_stress"
MANIFEST_NAME = "visual_manifest.json"
RESULTS_NAME = "per_image_results.json"
SUMMARY_NAME = "summary.json"
REPORT_NAME = "visual_stress_report.md"
SUCCESS_INDEX_NAME = "success_cases.json"
FAILURE_INDEX_NAME = "failure_cases.json"
IMAGES_DIR = "images"

#: The recognition path under test, in the UI's own vocabulary rather than a
#: second one -- a private set of names here would let this module accept a
#: method the interface does not have. ``cv-baseline`` is the default because
#: it needs no checkpoint and no torch, and a baseline that cannot run
#: without a fitted model stops being reproducible the day that model moves.
METHOD_CV = "cv-baseline"
METHOD_LEARNED = "transfer-resnet18"
METHODS: tuple[str, ...] = (METHOD_CV, METHOD_LEARNED)

#: The scene count and the render seed, frozen. Changing either changes every
#: image, so both are in the manifest digest.
N_SCENES = 8
RENDER_SEED = 20260829

#: The entry point whose import closure *is* this run's execution source.
#:
#: The gap this closes. The manifest used to pin a hand-written list of nine
#: modules, and the real closure from this entry point is 55: ``src/ui/app.py``
#: (which ``src/ui/full.py`` imports), ``src/vision/metrics.py``,
#: ``src/rendering/ldr.py``, ``src/colour/palette.py``, ``src/data/bricks.py``
#: and ``src/training/session.py`` among them. Every one of those can change a
#: reported number -- ``metrics.py`` is the matching arithmetic -- and none of
#: them moved the manifest digest. A list is a promise somebody will remember
#: to update it; this is a measurement.
EXECUTION_ENTRY_POINT = "scripts/60_visual_stress.py"

#: What ``run_summary.json`` must state, so it can be cross-checked against
#: the fitting manifest beside it rather than only hashed. A summary that
#: names different weights, a different epoch or different data than the
#: manifest it sits next to describes a different fit.
RUN_SUMMARY_PINS: tuple[str, ...] = (
    "checkpoint", "class_order", "data_manifest_sha256", "device", "epochs",
    "selected_epoch", "split_manifest_sha256", "train_items",
    "validation_items", "weights_sha256",
)

#: What a learned checkpoint must contain for a run to name it. Every one is
#: hashed into the manifest: a rate produced against different weights than
#: the manifest says is a rate about a different model.
CHECKPOINT_FILES: tuple[str, ...] = (
    "brickagain_vision_manifest.json",
    "run_summary.json",
    "selection_record.json",
    "vision_head.pt",
)

#: Box overlap at which a detection is taken to be about a given true brick.
MATCH_IOU = 0.30


#: Words a V1 report may not use about itself. Scanned before it is written.
FORBIDDEN_REPORT_TERMS: tuple[str, ...] = (
    "real photograph", "real photographs", "photographic accuracy",
    "in the wild", "in-the-wild", "physically verified",
    "assembly verified", "physically stable",
)

#: The named blocker, recorded rather than worked around.
V2_V3_BLOCKED: dict = {
    "v2_validated_ai_variations": "not run",
    "v3_pure_ai_hard_cases": "not run",
    "reason": (
        "this project has no authorised, reproducible, version-pinnable AI "
        "image generation mechanism. There is no diffusion checkpoint, no "
        "image-generation API client and no such dependency in "
        "requirements.txt or requirements-vision.txt. Producing either tier "
        "would require adding a credential, sending data to an unauthorised "
        "service, downloading a large model, or fabricating images -- all of "
        "which are refused."),
    "what_would_unblock_it": (
        "an image generation mechanism the user authorises, whose model, "
        "version, prompt, seed and parameters can be pinned, and which "
        "permits pinning the source image digest of every variation"),
    "not_claimed": (
        "no accuracy figure in this run covers V2 or V3, and none of these "
        "images may be presented as a photograph"),
}


#: Where the generation, the supersession record and the recogniser
#: declaration live. **Not literals in this module, deliberately.**
#:
#: This file is one of the 60 the Phase 3C pack carries, and Phase 3C gen09's
#: pack evidence pins every one of them by SHA-256. While the generation was a
#: literal here, bumping it edited a file that archive pins -- so re-freezing
#: V1 invalidated Phase 3C, exactly as bumping Phase 3C used to invalidate V1
#: by editing ``src/training/pack.py``. ``pack.py`` cut that direction by
#: moving the Phase 3C generation into ``gpu_plans/phase3c_staged.json``; this
#: cuts the mirror direction. ``data/**`` is not in the pack, so a V1 bump now
#: rewrites a JSON file and leaves every packed module byte-identical.
#:
#: The read is at import and fails closed: without a well-formed declaration
#: there is no generation, and every function here needs one.
#: The record that already existed for this. A second file declaring the
#: same fact is two truths, which is what this project refuses everywhere
#: else; the presentation tooling reads this one by name.
GENERATION_DECLARATION = "data/reports/60_visual_stress/GENERATIONS.json"
GENERATION_KIND = "brickagain.visual_stress_generations"


def _read_generation_declaration(root=None) -> dict:
    """The declaration, or a refusal naming what is wrong with it.

    Fails closed in every direction: absent, unreadable, the wrong kind,
    missing a field, a generation that is also listed as superseded, or a
    recogniser list that is not a list of strings. A caller never gets a
    partial answer, because a partial one would silently pin the wrong files.
    """
    path = Path(root or ROOT) / GENERATION_DECLARATION
    if not path.is_file():
        raise PlanRefused(
            f"{GENERATION_DECLARATION} is not here; the V1 generation and the "
            "recogniser declaration live there and nothing can run without "
            "them")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PlanRefused(
            f"{GENERATION_DECLARATION} is not readable JSON ({exc})") from None
    if not isinstance(body, dict) or body.get("kind") != GENERATION_KIND:
        raise PlanRefused(
            f"{GENERATION_DECLARATION} says kind {body.get('kind')!r} and "
            f"this reader wants {GENERATION_KIND!r}")

    problems = []
    generation = body.get("current")
    if not isinstance(generation, str) or not generation:
        problems.append("current is missing or not a string")
    recogniser = body.get("recogniser")
    if not isinstance(recogniser, dict):
        problems.append("recogniser is missing or not an object")
        recogniser = {}
    for field in ("source_files", "learned_source_files",
                  "required_in_closure"):
        value = recogniser.get(field)
        if not isinstance(value, list) or not value or not all(
                isinstance(v, str) and v for v in value):
            problems.append(f"recogniser.{field} is not a non-empty list of "
                            "paths")
    superseded = body.get("superseded")
    if not isinstance(superseded, list):
        problems.append("superseded is missing or not a list")
        superseded = []
    for entry in superseded:
        if not isinstance(entry, dict) or not entry.get("generation"):
            problems.append("a superseded entry names no generation")
            continue
        # The live generation cannot also be a retired one. That confusion is
        # what lets a superseded run be published under current authority.
        if entry["generation"] == generation:
            problems.append(
                f"{generation!r} is the current generation and is also listed "
                "as superseded")
    if problems:
        raise PlanRefused(
            f"{GENERATION_DECLARATION} does not hold:\n  - "
            + "\n  - ".join(problems))
    return body


_DECLARATION = _read_generation_declaration()

#: This run's generation. Artefacts are versioned rather than replaced,
#: because each time the manifest gained something it had been missing, a
#: summary written under the old shape stopped describing a run anybody
#: could check -- and overwriting it would destroy the only evidence of what
#: the previous attempt actually pinned.
GENERATION: str = _DECLARATION["current"]

SUPERSEDED: tuple[dict, ...] = tuple(_DECLARATION["superseded"])

#: The modules that *are* the recogniser under test. A file here that the
#: closure does not reach is a refusal: the manifest would then not be
#: describing the recogniser.
SOURCE_FILES: tuple[str, ...] = tuple(
    _DECLARATION["recogniser"]["source_files"])

#: The extra modules the *learned* path reaches and the CV path does not.
LEARNED_SOURCE_FILES: tuple[str, ...] = tuple(
    _DECLARATION["recogniser"]["learned_source_files"])

#: Names the closure must reach even though no list here spells out why each
#: one matters. Every one is a module an earlier hand-written list omitted
#: while it could still change a reported number.
REQUIRED_IN_CLOSURE: tuple[str, ...] = tuple(
    _DECLARATION["recogniser"]["required_in_closure"])


def execution_sources(root=None) -> tuple[str, ...]:
    """Every local module this run can reach, computed from the entry point.

    A static import closure, the same measurement the pack uses, so a direct,
    indirect or conditional dependency cannot escape by not being on
    anybody's list. The entry point itself is included; so is every module
    the render, partition, detect, score, report and verify paths reach.

    Fails closed two ways: a file in :data:`SOURCE_FILES` or
    :data:`LEARNED_SOURCE_FILES` that the closure does not reach is a
    refusal, because the closure would then not be describing the recogniser
    under test, and a name the closure reaches that is not a readable file
    here is a refusal too.
    """
    from src.training import pack as pack_module

    base = Path(root or ROOT)
    closure = pack_module.import_closure(
        base, entry_points=(EXECUTION_ENTRY_POINT,))
    names = sorted(set(closure) | {EXECUTION_ENTRY_POINT})
    declared = (set(SOURCE_FILES) | set(LEARNED_SOURCE_FILES)
                | set(REQUIRED_IN_CLOSURE))
    missing = sorted(declared - set(names))
    if missing:
        raise PlanRefused(
            f"{missing} are the modules this test is about and the import "
            f"closure of {EXECUTION_ENTRY_POINT} does not reach them; a "
            "source manifest that omitted them would not describe the "
            "recogniser under test")
    unreadable = [n for n in names if not (base / n).is_file()]
    if unreadable:
        raise PlanRefused(
            f"{unreadable} are in the closure and are not files here")
    return tuple(names)


def source_manifest(root=None, *, method: str = METHOD_CV) -> dict:
    """The digest of every module in this run's execution closure.

    Method-independent by design. The learned path reaches four modules the
    CV path does not *execute*, but a static closure reaches both, and
    pinning only what one branch runs would leave the other branch's code
    free to change under a manifest that still matched. Which subset a
    method actually executes is recorded separately, by name, in
    :func:`method_source_files`.
    """
    base = Path(root or ROOT)
    del method                       # kept for callers; the closure is one set
    return {name: sha256_file(base / name)
            for name in execution_sources(base)}


def method_source_files(method: str) -> list[str]:
    """The named modules a given recogniser's answers actually pass through.

    A statement about which branch runs, not the pinning set. Everything
    here is inside :func:`execution_sources` and is digested by it.
    """
    names = set(SOURCE_FILES)
    if method == METHOD_LEARNED:
        names |= set(LEARNED_SOURCE_FILES)
    return sorted(names)


def checkpoint_manifest(checkpoint) -> dict:
    """Every file in a learned checkpoint, hashed, plus what it says it is.

    The weights, the fitting manifest, the selection record and the run
    summary. Without this a manifest naming ``transfer-resnet18`` described a
    method rather than a model: the same word covers any checkpoint anybody
    points at, including one fitted on different data or selected by a
    different rule.
    """
    directory = Path(checkpoint)
    if not directory.is_dir():
        raise PlanRefused(f"{directory} is not a checkpoint directory")
    files = {}
    for name in CHECKPOINT_FILES:
        path = directory / name
        if not path.is_file():
            raise PlanRefused(
                f"{directory} is missing {name}; a learned run names the "
                "weights, the fitting manifest, the selection record and the "
                "run summary, or it names nothing")
        files[name] = {"sha256": sha256_file(path),
                       "bytes": path.stat().st_size}
    fitted = json.loads(
        (directory / "brickagain_vision_manifest.json").read_text())
    selection = json.loads((directory / "selection_record.json").read_text())
    summary = json.loads((directory / "run_summary.json").read_text())
    body = {
        "directory": directory.name,
        "files": files,
        "kind": fitted.get("kind"),
        "backbone": fitted.get("backbone"),
        "class_order": fitted.get("class_order"),
        "code_sha256": fitted.get("code_sha256"),
        "config": fitted.get("config"),
        "data_manifest_sha256": fitted.get("data_manifest_sha256"),
        "split_manifest_sha256": fitted.get("split_manifest_sha256"),
        "dependencies": fitted.get("dependencies"),
        "fitted_on_device": fitted.get("device"),
        "preprocessing": fitted.get("preprocessing"),
        "seed": fitted.get("seed"),
        "selected_epoch": fitted.get("selected_epoch"),
        "selection_criterion": fitted.get("selection_criterion"),
        "selection_record_kind": selection.get("kind"),
        "selection_checks": dict(selection.get("checks") or {}),
        "selection_problems": list(selection.get("problems") or []),
        "selection_epoch_stated": (selection.get("epoch_selection") or {}
                                   ).get("selected_epoch_stated"),
        "selection_epoch_rederived": (selection.get("epoch_selection") or {}
                                      ).get("selected_epoch_rederived"),
        "selection_criterion_recorded": (selection.get("epoch_selection")
                                         or {}).get("criterion"),
        "selection_digests": dict(selection.get("digests") or {}),
        "run_summary": {k: summary.get(k) for k in RUN_SUMMARY_PINS},
        "boundary": fitted.get("boundary"),
    }
    body["checkpoint_digest"] = digest_obj(body)
    return body


def checkpoint_problems(checkpoint, recorded=None, *, root=None) -> list[str]:
    """Whether this checkpoint may produce a number, and is the pinned one.

    Two questions, deliberately together, because answering only the first
    is what the old code did. It hashed the four files and stopped: a
    selection record whose own audit had failed, or which named an epoch its
    log did not support, hashed exactly as well as one that held.

    So the record's own verdict is read as a verdict. ``checks`` must all be
    true, ``problems`` must be empty, and the epoch it states must be the
    epoch it re-derived. Then, if ``recorded`` is given, the whole thing is
    compared to what the manifest pinned -- files, digests, config, class
    order, epoch, criterion and code/data/split provenance, by comparing
    ``checkpoint_digest``, which covers all of them.
    """
    problems: list[str] = []
    path = Path(checkpoint)
    if not path.is_absolute():
        path = Path(root or ROOT) / path
    try:
        actual = checkpoint_manifest(path)
    except (PlanRefused, OSError, ValueError) as exc:
        return [f"the checkpoint cannot be read: {exc}"]

    checks = actual.get("selection_checks") or {}
    if not checks:
        problems.append("the selection record states no checks; a record "
                        "that audits nothing is not an audit")
    failed = sorted(k for k, v in checks.items() if v is not True)
    if failed:
        problems.append(
            f"the selection record's own checks did not pass: {failed}. "
            "These weights were not selected the way the record says they "
            "were, and a rate from them describes an unknown model.")
    if actual.get("selection_problems"):
        problems.append(
            f"the selection record names problems of its own: "
            f"{actual['selection_problems'][:5]}")
    stated = actual.get("selection_epoch_stated")
    rederived = actual.get("selection_epoch_rederived")
    if stated is None or rederived is None:
        problems.append("the selection record does not say which epoch it "
                        "kept, or does not re-derive it")
    elif stated != rederived:
        problems.append(
            f"the selection record states epoch {stated} and re-derives "
            f"{rederived}")
    if actual.get("selected_epoch") not in (None, stated):
        problems.append(
            f"the fitting manifest names epoch {actual.get('selected_epoch')} "
            f"and the selection record states {stated}")
    # The run summary, cross-checked against the fitting manifest rather
    # than merely hashed. run_summary.json carries no ``kind``; what makes
    # it evidence is that it names the same weights, the same epoch and the
    # same data and split as the manifest beside it.
    summary = actual.get("run_summary") or {}
    for field, want in (("weights_sha256",
                         (actual.get("files") or {}).get(
                             "vision_head.pt", {}).get("sha256")),
                        ("selected_epoch", actual.get("selected_epoch")),
                        ("class_order", actual.get("class_order")),
                        ("data_manifest_sha256",
                         actual.get("data_manifest_sha256")),
                        ("split_manifest_sha256",
                         actual.get("split_manifest_sha256")),
                        ("device", actual.get("fitted_on_device"))):
        got = summary.get(field)
        if got is None:
            problems.append(f"run_summary.json does not record {field}")
        elif want is not None and got != want:
            problems.append(
                f"run_summary.json says {field} is {str(got)[:24]!r} and the "
                f"fitting manifest says {str(want)[:24]!r}")

    if recorded is not None:
        if not isinstance(recorded, dict):
            problems.append("the manifest records no checkpoint manifest")
        elif actual["checkpoint_digest"] != recorded.get("checkpoint_digest"):
            problems.append(
                f"the checkpoint here digests to "
                f"{actual['checkpoint_digest'][:16]}... and the manifest "
                f"pins {str(recorded.get('checkpoint_digest'))[:16]}...; "
                "these would be numbers about a different model")
    return problems


def inference_environment(device=None) -> dict:
    """Where the inference ran, read rather than assumed.

    Not the environment the checkpoint was *fitted* in -- that is inside
    ``checkpoint_manifest`` -- but the one that produced these numbers. The
    two differ here: the fit ran on CUDA and this scoring runs on the Mac.
    """
    import platform

    def _version(name: str):
        try:
            return __import__(name).__version__
        except Exception:
            return None

    resolved = device
    if resolved is None:
        try:
            from src.vision.model import resolve_device

            resolved = resolve_device(None)
        except Exception:
            resolved = None
    return {
        "device_requested": device,
        "device_resolved": resolved,
        "os_system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "numpy": _version("numpy"),
        "PIL": _version("PIL"),
    }


def image_id(scene, condition: dict) -> str:
    return f"{scene.scene_id}/{synthetic.condition_id(condition)}"


def image_member(scene, condition: dict) -> str:
    safe = synthetic.condition_id(condition).replace("=", "-").replace(
        "+", "_")
    return f"{IMAGES_DIR}/{scene.scene_id}__{safe}.png"


def cases(root=None) -> list[dict]:
    """Every (scene, condition) pair, with its ground truth and digest.

    The image digest is computed from the pixels the renderer produces, not
    from a file: it is what a stored PNG must reproduce, which is how a
    swapped image is caught.
    """
    out = []
    for scene in synthetic.scenes(N_SCENES, seed=RENDER_SEED):
        for condition in synthetic.condition_matrix():
            out.append({
                "image_id": image_id(scene, condition),
                "member": image_member(scene, condition),
                "scene_id": scene.scene_id,
                "condition": dict(condition),
                "condition_id": synthetic.condition_id(condition),
                "scene_digest": scene.digest(),
                "image_digest": synthetic.image_digest(
                    scene, condition, seed=RENDER_SEED),
                "ground_truth": scene.ground_truth(),
            })
    return out


def build_manifest(*, method: str, root=None, checkpoint=None,
                   device=None) -> dict:
    """The frozen manifest, including *which model* a learned run used.

    ``checkpoint`` is required for the learned method and refused for the CV
    one. Before this, a learned manifest named a method and nothing else, so
    two runs against different weights produced two manifests that agreed --
    and a rate could be quoted against a checkpoint nobody could identify.
    """
    if method not in METHODS:
        raise PlanRefused(f"{method!r} is not one of {list(METHODS)}")
    if method == METHOD_LEARNED and not checkpoint:
        raise PlanRefused(
            f"{METHOD_LEARNED} needs a checkpoint; a manifest that named the "
            "method alone would describe any weights anybody pointed at")
    if method == METHOD_CV and checkpoint:
        raise PlanRefused(
            f"{METHOD_CV} loads no checkpoint; naming one would record a "
            "dependency this run does not have")
    entries = cases(root)
    body = {
        "kind": MANIFEST_KIND,
        "contract_version": CONTRACT_VERSION,
        "generation": GENERATION,
        "superseded": [dict(s) for s in SUPERSEDED],
        "tier": "V1",
        "tiers": V2_V3_BLOCKED,
        "provenance": {
            "renderer": "src.vision.synthetic.render",
            "kind": "deterministic software render",
            "not_a_photograph": (
                "every image is drawn by arithmetic from a stated scene; none "
                "is a photograph and none may be presented as one"),
            "render_seed": RENDER_SEED,
            "stud_px": synthetic.STUD_PX,
            "ai_model": None,
            "ai_model_version": None,
            "ai_prompt": None,
            "ai_seed": None,
        },
        "recognition": {
            "method": method,
            "entry_point": "src.ui.photo.analyse_photo",
            "mode": "multi",
            "checkpoint": (str(checkpoint) if checkpoint else None),
            "checkpoint_manifest": (checkpoint_manifest(checkpoint)
                                    if checkpoint else None),
            "device_requested": device,
            "inference_environment": inference_environment(device),
            "learned_sources_included": method == METHOD_LEARNED,
        },
        "scenes": N_SCENES,
        "conditions": {k: list(v) for k, v in synthetic.CONDITIONS.items()},
        "baseline": dict(synthetic.BASELINE),
        "condition_ids": [synthetic.condition_id(c)
                          for c in synthetic.condition_matrix()],
        "n_images": len(entries),
        "acceptance": ACCEPTANCE,
        "rejection_criteria": REJECTION_CRITERIA,
        "metrics": list(METRICS),
        "forbidden_report_terms": list(FORBIDDEN_REPORT_TERMS),
        "execution_source": {
            "entry_point": EXECUTION_ENTRY_POINT,
            "computed_by": ("static import closure over the local src tree, "
                            "src.training.pack.import_closure"),
            "n_files": len(source_manifest(root)),
            "method_source_files": method_source_files(method),
            "note": ("the pinned set is the whole closure and does not "
                     "depend on the method; method_source_files says which "
                     "of it this recogniser executes"),
        },
        "source_manifest": source_manifest(root, method=method),
        "cases": entries,
    }
    body["source_manifest_digest"] = digest_obj(body["source_manifest"])
    body["manifest_digest"] = digest_obj(
        {k: v for k, v in body.items() if k != "manifest_digest"})
    return body


#: What makes an image admissible to the formal rate. For V1 these are
#: mechanical, because the ground truth is constructed rather than labelled.
#: They are stated anyway: V2 would use the same field with a rule about
#: whether the generator changed the geometry, and a schema that only grew a
#: rejection concept when it was needed would be a schema written after the
#: result.
ACCEPTANCE: tuple[str, ...] = (
    "the stored image reproduces the digest the manifest pins",
    "the ground truth is the scene the renderer was given, unmodified",
    "the renderer's source digest matches the manifest",
)

REJECTION_CRITERIA: tuple[str, ...] = (
    "the image does not reproduce its pinned digest",
    "the scene digest does not match the manifest",
    "the image could not be rendered or stored",
    "(V2 only, not run) the generator altered brick geometry, count, colour "
    "or part, so the ground truth no longer describes the image",
)

METRICS: tuple[str, ...] = (
    "inventory_exact_match", "part_errors", "colour_errors", "count_errors",
    "false_positives", "false_negatives", "ui_loaded",
    "oracle_correction_path_valid", "safe_rejection_shared_input_layer",
    "per_condition", "rejected_images",
)


def manifest_problems(manifest, *, root=None) -> list[str]:
    problems: list[str] = []
    if not isinstance(manifest, dict):
        return [f"the manifest is a {type(manifest).__name__}"]
    if manifest.get("kind") != MANIFEST_KIND:
        problems.append(f"kind is {manifest.get('kind')!r}")
    if manifest.get("tier") != "V1":
        problems.append(f"tier is {manifest.get('tier')!r}, not V1")
    if manifest.get("recognition", {}).get("method") not in METHODS:
        problems.append("the recognition method is not one of "
                        f"{list(METHODS)}")
    entries = manifest.get("cases") or []
    if len(entries) != manifest.get("n_images"):
        problems.append("n_images does not match the case list")
    recognition = manifest.get("recognition") or {}
    method = recognition.get("method")
    try:
        here = source_manifest(root, method=method)
    except PlanRefused as exc:
        problems.append(f"the execution source cannot be computed here: "
                        f"{exc}")
        here = None
    if here is not None and manifest.get("source_manifest") != here:
        recorded = manifest.get("source_manifest") or {}
        differing = sorted(
            name for name in set(recorded) | set(here)
            if recorded.get(name) != here.get(name))
        problems.append(
            "the code on this machine is not the code this manifest pins, so "
            "a number derived here would not be the number it describes; "
            f"{len(differing)} files differ, first {differing[:5]}")
    if method == METHOD_LEARNED:
        checkpoint = recognition.get("checkpoint")
        recorded = recognition.get("checkpoint_manifest")
        if not checkpoint or not recorded:
            problems.append(
                "a learned manifest must name its checkpoint and hash it")
        else:
            # The whole verdict, not just the digest: a selection record
            # whose own checks failed hashes exactly as well as one that
            # held, and hashing it was all this used to do.
            problems.extend(checkpoint_problems(checkpoint, recorded,
                                                root=root))
    elif recognition.get("checkpoint") is not None:
        problems.append(f"{METHOD_CV} named a checkpoint")

    declared = manifest.get("execution_source") or {}
    if declared.get("entry_point") != EXECUTION_ENTRY_POINT:
        problems.append(
            f"the manifest's execution entry point is "
            f"{declared.get('entry_point')!r}, not {EXECUTION_ENTRY_POINT!r}")
    if declared.get("n_files") != len(manifest.get("source_manifest") or {}):
        problems.append(
            "execution_source.n_files does not match the source manifest")
    if declared.get("method_source_files") != method_source_files(method):
        problems.append(
            "the manifest names a different set of modules as the ones this "
            "recogniser executes")
    body = {k: v for k, v in manifest.items() if k != "manifest_digest"}
    if manifest.get("manifest_digest") != digest_obj(body):
        problems.append("manifest_digest does not cover the manifest")
    return problems


# ---------------------------------------------------------------------------
# Rendering the frozen set to disk, write-once
# ---------------------------------------------------------------------------

def render_images(out_dir, manifest: dict) -> dict:
    """Write every image the manifest names. Refuses to overwrite one."""
    out_dir = Path(out_dir)
    scene_by_id = {s.scene_id: s
                   for s in synthetic.scenes(N_SCENES, seed=RENDER_SEED)}
    written, skipped = [], []
    for entry in manifest["cases"]:
        path = out_dir / entry["member"]
        if path.exists():
            skipped.append(entry["member"])
            continue
        scene = scene_by_id[entry["scene_id"]]
        array = synthetic.render(scene, entry["condition"], seed=RENDER_SEED)
        path.parent.mkdir(parents=True, exist_ok=True)
        synthetic.write_png(path, array)
        written.append(entry["member"])
    return {"written": len(written), "already_present": len(skipped)}


def stored_digest(path) -> str:
    """The digest of a stored PNG's *pixels*, comparable to the manifest's.

    Not the file's SHA-256: PNG encoders differ by version and a re-encode
    that changed no pixel would look like tampering. What is pinned is the
    image, so the image is what is hashed.
    """
    import numpy as np
    from PIL import Image

    with Image.open(str(path)) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    return synthetic.digest_array(array)


def partition(out_dir, manifest: dict) -> tuple[list, list]:
    """``(accepted, rejected)``, each entry naming its reason."""
    out_dir = Path(out_dir)
    accepted, rejected = [], []
    for entry in manifest["cases"]:
        path = out_dir / entry["member"]
        if not path.is_file():
            rejected.append({"image_id": entry["image_id"],
                             "member": entry["member"],
                             "reason": "the image was not rendered or stored"})
            continue
        try:
            digest = stored_digest(path)
        except Exception as exc:                     # noqa: BLE001
            rejected.append({"image_id": entry["image_id"],
                             "member": entry["member"],
                             "reason": f"the image could not be read: {exc}"})
            continue
        if digest != entry["image_digest"]:
            rejected.append({
                "image_id": entry["image_id"], "member": entry["member"],
                "reason": (f"the stored image hashes to {digest[:16]}... and "
                           f"the manifest pins "
                           f"{entry['image_digest'][:16]}...")})
            continue
        accepted.append({"image_id": entry["image_id"],
                         "member": entry["member"]})
    return accepted, rejected


# ---------------------------------------------------------------------------
# Scoring one image
# ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / float(area_a + area_b - inter)


def truth_boxes(scene, condition: dict) -> list[dict]:
    """Where each true brick is, in the pixels of the rendered image.

    Under a tilt the boxes move, and the tilt is a per-row horizontal scale,
    so the true box is recomputed through the same transform rather than
    assumed unchanged -- otherwise every tilted image would look like a
    detector failure that is really a bookkeeping failure.
    """
    width, height = synthetic.canvas_size(scene, condition)
    out = []
    for p, x, y in synthetic.layout(scene, condition):
        across, down = p.extents()
        x0 = int(round(x * synthetic.STUD_PX))
        y0 = int(round(y * synthetic.STUD_PX))
        x1 = x0 + int(round(across * synthetic.STUD_PX))
        y1 = y0 + int(round(down * synthetic.STUD_PX))
        box = _tilt_box((x0, y0, x1, y1), condition["viewpoint"], width,
                        height)
        out.append({"part": p.part, "colour_id": p.colour_id, "box": box})
    return out


def _tilt_box(box, kind: str, width: int, height: int):
    if kind == "top":
        return tuple(int(v) for v in box)
    strength = {"tilt_small": 0.16, "tilt_large": 0.34}[kind]
    centre = (width - 1) / 2.0
    x0, y0, x1, y1 = box

    def at(row, x):
        factor = 1.0 - strength * (1.0 - row / max(1, height - 1))
        return centre + (x - centre) * factor

    xs = [at(y0, x0), at(y0, x1), at(y1, x0), at(y1, x1)]
    return (int(round(min(xs))), int(y0), int(round(max(xs))), int(y1))


def score_image(entry: dict, analysis, scene) -> dict:
    """One image's verdict, from the UI's own analysis object."""
    from src.ui.corrections import adopt, inventory_spec

    truth = truth_boxes(scene, entry["condition"])
    items = [i for i in analysis.items if not i.deleted]
    adopted = adopt(items)

    matched_truth: dict[int, int] = {}
    matched_item: dict[int, int] = {}
    for ti, t in enumerate(truth):
        best, best_iou = None, 0.0
        for ii, item in enumerate(items):
            if ii in matched_item:
                continue
            score = _iou(t["box"], item.adopted_box)
            if score > best_iou:
                best, best_iou = ii, score
        if best is not None and best_iou >= MATCH_IOU:
            matched_truth[ti] = best
            matched_item[best] = ti

    part_errors, colour_errors = [], []
    for ti, ii in sorted(matched_truth.items()):
        item = items[ii]
        if item.adopted_part != truth[ti]["part"]:
            part_errors.append({"true": truth[ti]["part"],
                                "predicted": item.adopted_part})
        if item.adopted_colour != truth[ti]["colour_id"]:
            colour_errors.append({"true": truth[ti]["colour_id"],
                                  "predicted": item.adopted_colour})

    true_inventory = dict(entry["ground_truth"]["inventory"])
    count_errors = {}
    for part in sorted(set(true_inventory) | set(adopted.parts)):
        delta = adopted.parts.get(part, 0) - true_inventory.get(part, 0)
        if delta:
            count_errors[part] = delta

    corrected, corrected_spec = _correction_flow(items, truth, true_inventory)

    return {
        "image_id": entry["image_id"],
        "scene_id": entry["scene_id"],
        "condition_id": entry["condition_id"],
        "condition": dict(entry["condition"]),
        "n_true_bricks": len(truth),
        "n_detected": len(items),
        "ui_loaded": True,
        "inventory_exact_match": adopted.parts == true_inventory,
        "predicted_inventory": dict(adopted.parts),
        "true_inventory": true_inventory,
        "count_errors": count_errors,
        "part_errors": part_errors,
        "n_part_errors": len(part_errors),
        "colour_errors": colour_errors,
        "n_colour_errors": len(colour_errors),
        "false_positives": len(items) - len(matched_item),
        "false_negatives": len(truth) - len(matched_truth),
        "unresolved": list(adopted.unresolved),
        # Named for what it measures. It was "correction_flow_completed",
        # which read as a claim about people completing a flow; nobody was
        # asked to do anything and the answers were injected from the
        # ground truth.
        "oracle_correction_path_valid": corrected,
        "oracle_corrected_spec": corrected_spec,
        "abstentions": sum(1 for i in items
                           if i.adopted_part == synthetic_unknown()),
        "mean_confidence": (
            round(sum(float(i.predicted_confidence) for i in items)
                  / len(items), 4) if items else None),
        # top-1-correct even where the classifier declined to commit. An
        # abstention and a wrong answer are different failures: the first
        # sends the crop to a person, the second puts a wrong brick in the
        # inventory, and a report that fused them would hide which one this
        # path actually has.
        "top1_correct_among_matched": _top1_correct(items, truth,
                                                    matched_truth),
        "diagnostics": _plain(
            {k: v for k, v in (analysis.diagnostics or {}).items()
             if k in ("proposals", "kept_boxes", "dropped_thin",
                      "empty_reason", "suppressed")}),
    }


def synthetic_unknown() -> str:
    from src.vision.classes import UNKNOWN

    return UNKNOWN


def _top1_correct(items, truth, matched_truth) -> int:
    """How often the classifier's first choice was right, abstention aside."""
    hits = 0
    for ti, ii in matched_truth.items():
        item = items[ii]
        top = (item.predicted_top3 or (None,))[0]
        if top == truth[ti]["part"]:
            hits += 1
    return hits


def _plain(value):
    """numpy scalars out, JSON in. A diagnostic that cannot be stored is
    a diagnostic that stops the run at the last step."""
    import numpy as np

    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _correction_flow(items, truth, true_inventory) -> tuple[bool, str | None]:
    """Does the correction path hold when an oracle supplies the answers?

    The edits go through the project's own correction dataclass and its own
    ``adopt``/``inventory_spec``, so a path that could not produce a usable
    inventory fails here. What it does **not** establish is that a person
    would get there: the corrections below are read straight out of the
    ground truth, in order, with no searching and no judgement.
    """
    from src.ui.corrections import add_item, adopt, inventory_spec

    edited = []
    for index, item in enumerate(items):
        if index < len(truth):
            edited.append(_replace(item, edited_part=truth[index]["part"],
                                   edited_colour=truth[index]["colour_id"],
                                   edited_count=1))
        else:
            edited.append(_replace(item, deleted=True))
    for index in range(len(items), len(truth)):
        edited = list(add_item(edited, part=truth[index]["part"], count=1,
                               colour_id=truth[index]["colour_id"]))
    result = adopt(edited)
    if result.parts != true_inventory:
        return False, None
    return True, inventory_spec(result.parts)


def _replace(item, **changes):
    import dataclasses

    return dataclasses.replace(item, **changes)


# ---------------------------------------------------------------------------
# Running the whole set
# ---------------------------------------------------------------------------

def analyse_one(out_dir, entry: dict, *, method: str, checkpoint=None,
                device=None):
    from src.ui.photo import analyse_photo

    path = Path(out_dir) / entry["member"]
    return analyse_photo(path.read_bytes(), mode="multi", method=method,
                         checkpoint=checkpoint, device=device)


def runtime_problems(manifest: dict, *, checkpoint=None, device=None,
                     root=None) -> list[str]:
    """Whether this process will run the model and device the manifest pins.

    The fail-open this closes: ``run`` took ``checkpoint`` and ``device``
    straight off the command line and passed them to ``analyse_photo``.
    Nothing compared either to the manifest, so a manifest pinning one
    checkpoint could be scored by another, and a manifest recording MPS
    could be scored on CPU, with the digest chain agreeing throughout.

    Now the manifest is the authority. A caller may still *state* the two --
    that is how the command line stays usable and how a mistake is caught
    loudly rather than absorbed -- but a stated value that differs from the
    pinned one is a refusal, not an override.
    """
    problems: list[str] = []
    recognition = manifest.get("recognition") or {}
    method = recognition.get("method")
    pinned_checkpoint = recognition.get("checkpoint")

    if checkpoint is not None and str(checkpoint) != str(pinned_checkpoint):
        problems.append(
            f"this run was asked for checkpoint {str(checkpoint)!r} and the "
            f"manifest pins {str(pinned_checkpoint)!r}; the manifest decides")
    if method == METHOD_LEARNED:
        if not pinned_checkpoint:
            problems.append("a learned manifest that names no checkpoint "
                            "cannot say which model produced a number")
        else:
            problems.extend(checkpoint_problems(
                pinned_checkpoint, recognition.get("checkpoint_manifest"),
                root=root))
    elif pinned_checkpoint is not None:
        problems.append(f"{METHOD_CV} loads no checkpoint and this manifest "
                        "names one")

    pinned_request = recognition.get("device_requested")
    if device is not None and device != pinned_request:
        problems.append(
            f"this run was asked for device {device!r} and the manifest pins "
            f"{pinned_request!r}")
    pinned_env = recognition.get("inference_environment") or {}
    here = inference_environment(pinned_request)
    for field in ("device_resolved", "os_system", "machine", "python",
                  "torch", "numpy", "PIL"):
        if here.get(field) != pinned_env.get(field):
            problems.append(
                f"the inference environment differs from the manifest's: "
                f"{field} is {here.get(field)!r} here and the manifest "
                f"records {pinned_env.get(field)!r}")
    return problems


def run(out_dir, manifest: dict, *, checkpoint=None, device=None,
        root=None) -> dict:
    """Score every accepted image, against the model the manifest pins.

    ``checkpoint`` and ``device`` are not overrides. They are compared to
    the manifest by :func:`runtime_problems` and then discarded: what is
    handed to the recogniser is what the manifest says, always.
    """
    out_dir = Path(out_dir)
    method = manifest["recognition"]["method"]
    problems = runtime_problems(manifest, checkpoint=checkpoint,
                                device=device, root=root)
    if problems:
        raise PlanRefused(
            "refusing to score against a runtime the manifest does not "
            "pin:\n  - " + "\n  - ".join(problems))
    # From the manifest, never from the argument.
    checkpoint = manifest["recognition"]["checkpoint"]
    device = manifest["recognition"]["device_requested"]
    accepted, rejected = partition(out_dir, manifest)
    accepted_ids = {a["image_id"] for a in accepted}
    scene_by_id = {s.scene_id: s
                   for s in synthetic.scenes(N_SCENES, seed=RENDER_SEED)}

    results = []
    for entry in manifest["cases"]:
        if entry["image_id"] not in accepted_ids:
            continue
        analysis = analyse_one(out_dir, entry, method=method,
                               checkpoint=checkpoint, device=device)
        results.append(score_image(entry, analysis,
                                   scene_by_id[entry["scene_id"]]))
    record = {
        "kind": RESULTS_KIND,
        "contract_version": CONTRACT_VERSION,
        "generation": manifest.get("generation"),
        "manifest_digest": manifest["manifest_digest"],
        "source_manifest_digest": manifest["source_manifest_digest"],
        "method": method,
        "accepted": accepted,
        "rejected": rejected,
        "results": results,
        "runtime": {
            "checkpoint": checkpoint,
            "checkpoint_digest": (
                (manifest["recognition"].get("checkpoint_manifest") or {})
                .get("checkpoint_digest")),
            "device_requested": device,
            "device_resolved": manifest["recognition"][
                "inference_environment"]["device_resolved"],
            "taken_from": ("the manifest; --checkpoint and --device are "
                           "compared to it and are not overrides"),
        },
        "safe_rejection_shared_input_layer": safe_rejection_probe(),
    }
    record["results_digest"] = digest_obj(
        {k: v for k, v in record.items() if k != "results_digest"})
    return record


def row_metric_problems(record: dict, manifest: dict) -> list[str]:
    """Everything a row must say about itself, checked without a model.

    The replay is the anchor for *what the recogniser returned*; this is the
    cheaper, model-free layer beneath it: a row's counts must agree with the
    lists they count, its exact-match flag must agree with the two
    inventories it reports, and its true-brick count must be the scene's.

    None of that needs weights, so it holds in a tree where the checkpoint
    is not published -- which is where a run's own arithmetic would
    otherwise go unchecked entirely.
    """
    problems: list[str] = []
    truth_by_id = {c["image_id"]: c for c in manifest.get("cases") or []}
    for row in record.get("results") or []:
        image_id = row.get("image_id")
        where = f"{image_id}"
        for count, listed in (("n_part_errors", "part_errors"),
                              ("n_colour_errors", "colour_errors")):
            items = row.get(listed)
            if not isinstance(items, list):
                problems.append(f"{where}: {listed} is not a list")
            elif row.get(count) != len(items):
                problems.append(
                    f"{where}: {count} is {row.get(count)} and {listed} "
                    f"holds {len(items)}")

        predicted = row.get("predicted_inventory")
        true = row.get("true_inventory")
        if isinstance(predicted, dict) and isinstance(true, dict):
            exact = predicted == true
            if bool(row.get("inventory_exact_match")) != exact:
                problems.append(
                    f"{where}: inventory_exact_match is "
                    f"{row.get('inventory_exact_match')!r} and the two "
                    "inventories it reports "
                    + ("agree" if exact else "differ"))
            deltas = {part: predicted.get(part, 0) - true.get(part, 0)
                      for part in sorted(set(predicted) | set(true))
                      if predicted.get(part, 0) != true.get(part, 0)}
            if row.get("count_errors") != deltas:
                problems.append(
                    f"{where}: count_errors is not the difference between "
                    "the inventories it reports")

        entry = truth_by_id.get(image_id)
        if entry is not None:
            truth = entry.get("ground_truth") or {}
            expected = truth.get("n_bricks")
            if expected is not None and row.get("n_true_bricks") != expected:
                problems.append(
                    f"{where}: n_true_bricks is {row.get('n_true_bricks')} "
                    f"and the manifest's scene has {expected}")
            want = truth.get("inventory")
            if isinstance(want, dict) and isinstance(true, dict) \
                    and true != want:
                problems.append(
                    f"{where}: the row's true_inventory is not the scene's")
        matched = (row.get("n_true_bricks") or 0) - (
            row.get("false_negatives") or 0)
        if (row.get("top1_correct_among_matched") or 0) > max(0, matched):
            problems.append(
                f"{where}: more bricks are called top-1 correct than were "
                "matched at all")
        if (row.get("false_negatives") or 0) > (row.get("n_true_bricks") or 0):
            problems.append(
                f"{where}: more true bricks are missed than exist")
    return problems


def replay_problems(out_dir, manifest: dict, record: dict, *,
                    root=None, images=None) -> list[str]:
    """Re-run the pinned recogniser and compare its answers to the stored ones.

    **This is the independent anchor for what the recogniser predicted.**
    Every other check in this file is internal: the summary re-derives from
    the rows, the indices re-derive from the rows, the digests cover the
    rows. All of that holds perfectly for a result set whose rows were
    written by hand, because nothing outside the directory says what the
    model actually returned for a given image.

    So this loads the checkpoint the manifest pins, on the device it pins,
    puts the stored image bytes back through ``src.ui.photo.analyse_photo``,
    re-scores the result with :func:`score_image`, and compares the whole
    row. Determinism is what makes that a check rather than a second
    sample: the renderer is arithmetic, the weights are fixed, and there is
    no sampling anywhere in the path.

    ``images`` limits the replay to named ``image_id``s; it exists for
    tests, and no verification path uses it.
    """
    out_dir = Path(out_dir)
    problems = runtime_problems(manifest, root=root)
    if problems:
        return [f"cannot replay: {p}" for p in problems]

    method = manifest["recognition"]["method"]
    checkpoint = manifest["recognition"]["checkpoint"]
    device = manifest["recognition"]["device_requested"]
    scene_by_id = {sc.scene_id: sc
                   for sc in synthetic.scenes(N_SCENES, seed=RENDER_SEED)}
    stored = {row["image_id"]: row for row in record.get("results") or []}
    by_id = {c["image_id"]: c for c in manifest.get("cases") or []}

    wanted = [a["image_id"] for a in record.get("accepted") or []]
    if images is not None:
        wanted = [i for i in wanted if i in set(images)]

    differing: list[str] = []
    for image_id in wanted:
        entry = by_id.get(image_id)
        if entry is None:
            problems.append(f"{image_id} is accepted and is not in the "
                            "manifest")
            continue
        row = stored.get(image_id)
        if row is None:
            problems.append(f"{image_id} is accepted and has no stored row")
            continue
        try:
            analysis = analyse_one(out_dir, entry, method=method,
                                   checkpoint=checkpoint, device=device)
        except Exception as exc:                      # noqa: BLE001
            problems.append(f"{image_id} could not be re-analysed: "
                            f"{type(exc).__name__}: {str(exc)[:120]}")
            continue
        rebuilt = score_image(entry, analysis, scene_by_id[entry["scene_id"]])
        if rebuilt != row:
            fields = sorted(k for k in set(rebuilt) | set(row)
                            if rebuilt.get(k) != row.get(k))
            differing.append(f"{image_id} ({', '.join(fields[:4])})")

    if differing:
        problems.append(
            f"{len(differing)} of {len(wanted)} accepted images do not "
            f"reproduce their stored row when the pinned model is re-run, "
            f"first {differing[0]}")
    return problems


def safe_rejection_probe() -> dict:
    """Does the shared input layer refuse rubbish rather than inventing stock?

    Four inputs that are not images. A pipeline that answered any of them
    with an inventory would be a pipeline that answers anything with one.

    **This is one result about a common path, not a per-model test.**
    ``decode_image`` and the ``analyse_photo`` guards run before a
    recogniser is selected, so a malformed input never reaches either
    model. Both manifests report this and both necessarily report the same
    thing; the field is named to say so.
    """
    from src.ui.errors import UiError
    from src.ui.photo import analyse_photo

    probes = {
        "empty_bytes": b"",
        "not_an_image": b"this is not an image, it is a sentence",
        "truncated_png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 24,
        "zip_bomb_header": b"PK\x03\x04" + b"\x00" * 64,
    }
    out = {}
    for name, payload in probes.items():
        try:
            analyse_photo(payload, mode="multi", method=METHOD_CV)
        except UiError as exc:
            out[name] = {"refused": True, "by": "UiError",
                         "message": str(exc)[:160]}
        except Exception as exc:                     # noqa: BLE001
            out[name] = {"refused": False,
                         "by": type(exc).__name__,
                         "message": str(exc)[:160],
                         "note": ("refused, but not by the named path; an "
                                  "uncaught exception is not a refusal the "
                                  "interface can show a person")}
        else:
            out[name] = {"refused": False, "by": None,
                         "message": "an inventory was produced from this"}
    out["all_refused_by_name"] = all(v["refused"] for v in out.values()
                                     if isinstance(v, dict))
    return out


# ---------------------------------------------------------------------------
# The summary, re-derived from per-image results
# ---------------------------------------------------------------------------

def _rate(hits: int, n: int) -> dict:
    return {"numerator": hits, "denominator": n,
            "value": (hits / n) if n else None}


def summarise(results_record: dict, manifest: dict) -> dict:
    """Every reported number, recomputed from the per-image list.

    Nothing is carried through from the run: given the same per-image
    results this produces the same summary, and given different ones it
    produces different ones. That is what makes the summary checkable.
    """
    rows = results_record["results"]
    n = len(rows)

    def overall(rows_subset) -> dict:
        m = len(rows_subset)
        return {
            "images": m,
            "inventory_exact_match": _rate(
                sum(1 for r in rows_subset if r["inventory_exact_match"]), m),
            "ui_loaded": _rate(
                sum(1 for r in rows_subset if r["ui_loaded"]), m),
            "oracle_correction_path_valid": _rate(
                sum(1 for r in rows_subset
                    if r["oracle_correction_path_valid"]), m),
            "true_bricks": sum(r["n_true_bricks"] for r in rows_subset),
            "detected": sum(r["n_detected"] for r in rows_subset),
            "part_errors": sum(r["n_part_errors"] for r in rows_subset),
            "colour_errors": sum(r["n_colour_errors"] for r in rows_subset),
            "false_positives": sum(r["false_positives"]
                                   for r in rows_subset),
            "false_negatives": sum(r["false_negatives"]
                                   for r in rows_subset),
            "images_with_a_count_error": _rate(
                sum(1 for r in rows_subset if r["count_errors"]), m),
            # Abstentions and wrong answers are different failures and are
            # counted apart. An abstention sends the crop to a person, which
            # the correction flow then resolves; a confident wrong label puts
            # a brick in the inventory that is not on the table. A single
            # error rate would fuse the two and hide which one this path has.
            "abstentions": sum(r.get("abstentions") or 0
                               for r in rows_subset),
            "top1_correct_among_matched": _rate(
                sum(r.get("top1_correct_among_matched") or 0
                    for r in rows_subset),
                sum(r["n_true_bricks"] - r["false_negatives"]
                    for r in rows_subset)),
            "mean_confidence": (
                round(sum(r["mean_confidence"] for r in rows_subset
                          if r.get("mean_confidence") is not None)
                      / max(1, sum(1 for r in rows_subset
                                   if r.get("mean_confidence") is not None)),
                      4) if rows_subset else None),
        }

    by_condition = {}
    for condition_id in manifest["condition_ids"]:
        subset = [r for r in rows if r["condition_id"] == condition_id]
        by_condition[condition_id] = overall(subset)

    # One axis at a time, against the baseline -- and the baseline cell holds
    # *only* the baseline images. Selecting every row whose axis equals its
    # baseline value would sweep in every other axis's variations too, so the
    # "unchanged" column would silently be an average over all the changes
    # and every axis would look milder than it is.
    baseline = manifest["baseline"]
    by_axis: dict = {}
    for axis, values in manifest["conditions"].items():
        by_axis[axis] = {}
        for value in values:
            subset = [r for r in rows
                      if r["condition"][axis] == value
                      and all(r["condition"][other] == baseline[other]
                              for other in baseline if other != axis)]
            if subset:
                by_axis[axis][value] = overall(subset)

    by_scene = {}
    for scene_id in sorted({r["scene_id"] for r in rows}):
        by_scene[scene_id] = overall([r for r in rows
                                      if r["scene_id"] == scene_id])

    confusions: dict[str, int] = {}
    for row in rows:
        for err in row["part_errors"]:
            key = f"{err['true']}->{err['predicted']}"
            confusions[key] = confusions.get(key, 0) + 1
    colour_confusions: dict[str, int] = {}
    for row in rows:
        for err in row["colour_errors"]:
            key = f"{err['true']}->{err['predicted']}"
            colour_confusions[key] = colour_confusions.get(key, 0) + 1

    body = {
        "kind": SUMMARY_KIND,
        "contract_version": CONTRACT_VERSION,
        "generation": manifest.get("generation"),
        "tier": "V1",
        "tiers": manifest["tiers"],
        "manifest_digest": manifest["manifest_digest"],
        "source_manifest_digest": manifest["source_manifest_digest"],
        # The summary names the exact per-image record it was derived from.
        # Without it a summary could be paired with a different result set
        # that happened to imply the same numbers -- or, more usefully to an
        # attacker, a result set whose *unsummarised* fields differed.
        "results_digest": results_record.get("results_digest"),
        "method": results_record["method"],
        "n_images_in_manifest": manifest["n_images"],
        "n_accepted": len(results_record["accepted"]),
        "n_rejected": len(results_record["rejected"]),
        "rejected": results_record["rejected"],
        "n_scored": n,
        "overall": overall(rows),
        "by_condition": by_condition,
        "by_axis": by_axis,
        "by_scene": by_scene,
        "part_confusions": dict(sorted(confusions.items(),
                                       key=lambda kv: -kv[1])),
        "colour_confusions": dict(sorted(colour_confusions.items(),
                                         key=lambda kv: -kv[1])),
        "safe_rejection_shared_input_layer":
            results_record["safe_rejection_shared_input_layer"],
        "note": (
            "V1 only. Every image is a deterministic software render, not a "
            "photograph. These rates describe robustness to rendered image "
            "conditions and say nothing about real-photograph accuracy, "
            "in-the-wild behaviour, physical assembly or stability."),
    }
    body["summary_digest"] = digest_obj(
        {k: v for k, v in body.items() if k != "summary_digest"})
    return body


def summary_identity(summary) -> dict:
    """Everything that makes two summaries the same result.

    Deliberately *not* a chosen subset any more. It was, and the omitted
    fields -- ``rejected``, ``safe_rejection``, the tier declaration, the
    source digest -- are exactly the ones a rewrite would target: a run that
    dropped a rejected image, or claimed all four rubbish probes were
    refused when one was not, produced a summary that compared equal.
    """
    if not isinstance(summary, dict):
        return {"not_a_summary": type(summary).__name__}
    return {k: v for k, v in summary.items() if k != "summary_digest"}


def results_identity(record) -> dict:
    """The same, for a per-image result set.

    ``results`` alone was compared before, which let the accepted list, the
    rejected list and the safe-rejection probe change with nothing noticing.
    Only ``results_digest`` is excluded, because it is the digest of this
    mapping and including it would make the comparison circular.
    """
    if not isinstance(record, dict):
        return {"not_a_result_set": type(record).__name__}
    return {k: v for k, v in record.items() if k != "results_digest"}


def results_problems(record, manifest: dict) -> list[str]:
    """Whether a stored result set is internally sound and this manifest's.

    Everything here is checkable without re-running a recogniser: the
    record's own digest, the manifest it names, the accepted/rejected
    partition recomputed from the images on disk, and a one-to-one
    correspondence between the accepted list and the scored rows. A row for
    an image nobody accepted, or an accepted image with no row, is a hole in
    the denominator of every rate the summary reports.
    """
    problems: list[str] = []
    if not isinstance(record, dict):
        return [f"the result set is a {type(record).__name__}"]
    if record.get("kind") != RESULTS_KIND:
        problems.append(f"kind is {record.get('kind')!r}")
    for field in ("manifest_digest", "source_manifest_digest"):
        if record.get(field) != manifest.get(field):
            problems.append(f"the results' {field} is not this manifest's")
    recognition = manifest.get("recognition") or {}
    if record.get("method") != recognition.get("method"):
        problems.append("the results name a different recognition method")

    # The runtime block, against what the manifest pins. Without this a
    # result set could name any checkpoint and any device as the source of
    # its answers, rebuild every downstream digest, and hold.
    runtime = record.get("runtime")
    if not isinstance(runtime, dict):
        problems.append("the result set records no runtime")
    else:
        expected = {
            "checkpoint": recognition.get("checkpoint"),
            "checkpoint_digest": (
                (recognition.get("checkpoint_manifest") or {})
                .get("checkpoint_digest")),
            "device_requested": recognition.get("device_requested"),
            "device_resolved": (recognition.get("inference_environment")
                                or {}).get("device_resolved"),
        }
        for field, want in expected.items():
            if runtime.get(field) != want:
                problems.append(
                    f"the results say {field} was {runtime.get(field)!r} and "
                    f"the manifest pins {want!r}")

    accepted = record.get("accepted")
    rows = record.get("results")
    if not isinstance(accepted, list) or not isinstance(rows, list):
        return problems + ["the result set has no accepted list or no rows"]
    accepted_ids = [a.get("image_id") for a in accepted]
    row_ids = [r.get("image_id") for r in rows]
    if len(set(accepted_ids)) != len(accepted_ids):
        problems.append("the accepted list names an image twice")
    if len(set(row_ids)) != len(row_ids):
        problems.append("a scored row appears twice")
    unscored = sorted(set(accepted_ids) - set(row_ids))
    unaccepted = sorted(set(row_ids) - set(accepted_ids))
    if unscored:
        problems.append(
            f"{len(unscored)} accepted images have no scored row, first "
            f"{unscored[0]!r}")
    if unaccepted:
        problems.append(
            f"{len(unaccepted)} scored rows are for images this run did not "
            f"accept, first {unaccepted[0]!r}")

    rejected = record.get("rejected") or []
    known = {c["image_id"] for c in manifest.get("cases") or []}
    stray = sorted({*accepted_ids, *(r.get("image_id") for r in rejected)}
                   - known)
    if stray:
        problems.append(f"the result set names images the manifest does not: "
                        f"{stray[:3]}")
    if len(accepted_ids) + len(rejected) != len(known):
        problems.append(
            f"{len(accepted_ids)} accepted plus {len(rejected)} rejected is "
            f"not the manifest's {len(known)} images")

    if not isinstance(record.get("results_digest"), str) or \
            len(record.get("results_digest") or "") != 64:
        problems.append("results_digest is not a SHA-256")
    elif record["results_digest"] != digest_obj(results_identity(record)):
        problems.append("results_digest does not cover the result set")
    return problems


def digest_of(body) -> str:
    """One definition of "these bytes", for artefacts without their own."""
    return digest_obj(body)


def write_once_json_checked(path, body, *, what: str) -> dict:
    """Write, or accept an identical existing file, or refuse.

    The third branch is the one that matters and the one the report and the
    two indices did not have: they were written with ``write_text``, so a
    second run silently replaced them. Now a rerun over unchanged inputs
    rewrites nothing and a rerun over changed inputs stops.
    """
    from src.training.session import write_once_json

    path = Path(path)
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except ValueError as exc:
            raise PlanRefused(f"{path} exists and is not valid JSON ({exc}); "
                              "refusing to replace it") from exc
        if digest_of(existing) != digest_of(body):
            raise PlanRefused(
                f"{path} already holds a different {what}. A published "
                "artefact is not rewritten; a rerun that changes it is a "
                "different result and needs a new directory.")
        return {"path": str(path), "rewritten": False,
                "sha256": sha256_file(path)}
    write_once_json(path, body)
    return {"path": str(path), "rewritten": True, "sha256": sha256_file(path)}


def write_once_text_checked(path, text: str, *, what: str) -> dict:
    """The same discipline for the rendered report."""
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise PlanRefused(
                f"{path} already holds a different {what}. A published "
                "artefact is not rewritten; a rerun that changes it is a "
                "different result and needs a new directory.")
        return {"path": str(path), "rewritten": False,
                "sha256": sha256_file(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "rewritten": True, "sha256": sha256_file(path)}


def case_indices(results_record: dict) -> tuple[list, list]:
    """Success and failure indices, by whole-inventory exact match."""
    successes, failures = [], []
    for row in results_record["results"]:
        entry = {
            "image_id": row["image_id"],
            "member": next((a["member"] for a in results_record["accepted"]
                            if a["image_id"] == row["image_id"]), None),
            "scene_id": row["scene_id"],
            "condition_id": row["condition_id"],
            "true_inventory": row["true_inventory"],
            "predicted_inventory": row["predicted_inventory"],
            "count_errors": row["count_errors"],
            "n_part_errors": row["n_part_errors"],
            "n_colour_errors": row["n_colour_errors"],
            "false_positives": row["false_positives"],
            "false_negatives": row["false_negatives"],
            "oracle_correction_path_valid":
                row["oracle_correction_path_valid"],
        }
        (successes if row["inventory_exact_match"] else failures).append(entry)
    return successes, failures


def forbidden_terms_in(text: str) -> list[str]:
    low = text.lower()
    return [t for t in FORBIDDEN_REPORT_TERMS if t.lower() in low]


__all__ = [
    "KIND", "MANIFEST_KIND", "RESULTS_KIND", "SUMMARY_KIND", "DEFAULT_DIR",
    "GENERATION", "SUPERSEDED",
    "MANIFEST_NAME", "RESULTS_NAME", "SUMMARY_NAME", "REPORT_NAME",
    "SUCCESS_INDEX_NAME", "FAILURE_INDEX_NAME", "IMAGES_DIR",
    "METHOD_CV", "METHOD_LEARNED", "METHODS", "N_SCENES", "RENDER_SEED",
    "SOURCE_FILES", "V2_V3_BLOCKED", "FORBIDDEN_REPORT_TERMS", "MATCH_IOU",
    "ACCEPTANCE", "REJECTION_CRITERIA", "METRICS",
    "source_manifest", "cases", "build_manifest", "manifest_problems",
    "render_images", "stored_digest", "partition", "truth_boxes",
    "score_image", "analyse_one", "run", "safe_rejection_probe",
    "summarise", "summary_identity", "results_identity", "results_problems",
    "digest_of", "replay_problems", "row_metric_problems",
    "case_indices", "forbidden_terms_in", "image_id", "image_member",
    "write_once_json_checked", "write_once_text_checked",
    "checkpoint_manifest", "checkpoint_problems", "inference_environment",
    "CHECKPOINT_FILES", "RUN_SUMMARY_PINS", "LEARNED_SOURCE_FILES",
    "EXECUTION_ENTRY_POINT", "REQUIRED_IN_CLOSURE", "execution_sources",
    "method_source_files",
    "runtime_problems",
]
