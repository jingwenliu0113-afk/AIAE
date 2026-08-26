"""The full interface's work, as plain functions over plain data.

:mod:`src.ui.app` holds the two-page version and keeps working exactly as it
did.  This module adds what the full version needs, and it adds it the same
way: no HTTP, no HTML, every decision delegated to the module that already owns
it.

The delegations are the point, so they are listed rather than left to be
discovered:

* what counts as a deliverable structure -- ``scripts/27_delivery.py``'s
  ``DELIVERY_CHECKS``, read, never re-listed;
* what a valid inventory is -- ``src.demo.showcase.parse_inventory``, including
  rotation normalisation and the refusal to sum two spellings of one part;
* what the checks say -- ``src.eval.scoring.score_generation``, called;
* what the project model is -- ``runs/project_model.json`` verified by
  ``scripts/24_project_model.py``'s own verifier;
* colours -- :mod:`src.colour.assign`, which deducts from the existing
  inventory engine and never keeps a second counter;
* build order -- :mod:`src.assembly.order`, which re-verifies every step.

Three methods reach the same result shape: retrieval over existing works, the
minimum F-pipeline, and one demonstration decode with the project model.  A
page that renders all three from one shape cannot describe one of them with
another one's caveats.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from src.colour.assign import AssignError, Assignment, assign, parse_colour_stock
from src.colour.palette import ColourError
from src.data.bricks import parse_bricks
from src.delivery.pipeline import DeliveryError
from src.ui import corrections as corrections_module
from src.ui.app import UiError
from src.ui.corrections import CorrectionError, CorrectedInventory
from src.vision.classes import UNKNOWN

#: The three method names the full interface offers.
METHOD_RAG = "rag"
METHOD_PIPELINE = "f-pipeline"
METHOD_PROJECT = "final-h2"
FULL_METHODS = (METHOD_RAG, METHOD_PIPELINE, METHOD_PROJECT)

METHOD_LABELS = {
    METHOD_RAG: "RAG 既有作品檢索",
    METHOD_PIPELINE: "最低 F-pipeline",
    METHOD_PROJECT: "正式模型 final_H2 展示",
}

#: The two photograph modes.  They are different tasks with different stated
#: assumptions, so the operator chooses which one this photograph is.
PHOTO_SINGLE = "single"
PHOTO_MULTI = "multi"
PHOTO_MODES = (PHOTO_SINGLE, PHOTO_MULTI)

PHOTO_MODE_LABELS = {
    PHOTO_SINGLE: "單顆積木照片",
    PHOTO_MULTI: "多顆積木照片（平鋪、少遮擋）",
}

#: The recognition method for a photograph.
RECOGNISE_CV = "cv-baseline"
RECOGNISE_LEARNED = "transfer-resnet18"
RECOGNISE_METHODS = (RECOGNISE_CV, RECOGNISE_LEARNED)

CAPTURE_ASSUMPTION_ZH = (
    "影像辨識的成立條件是：積木平鋪、儘量不重疊、背景單純、光線穩定。"
    "不符合這些條件時，相鄰積木會被併成一個框，反光強的積木會被切成兩個框；"
    "這些失敗會如實出現在計數誤差裡，不會被藏起來。")

RECOGNITION_LIMIT_ZH = (
    "辨識結果只是**建議庫存**，不是量測結果。低信心與未辨識的項目必須由人工"
    "決定；本介面不會替它們挑一個最接近的類別。")


@dataclass(frozen=True)
class PhotoAnalysis:
    """What one photograph produced, before anybody corrected it."""

    mode: str
    method: str
    items: tuple
    diagnostics: dict
    width: int
    height: int
    colour_readings: dict = field(default_factory=dict, repr=False)
    colour_problems: dict = field(default_factory=dict, repr=False)

    @property
    def found(self) -> int:
        return len(self.items)

    @property
    def unidentified(self) -> int:
        return sum(1 for item in self.items
                   if item.predicted_part == UNKNOWN)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "method": self.method,
            "found": self.found,
            "unidentified": self.unidentified,
            "width": self.width,
            "height": self.height,
            "image": {"width": self.width, "height": self.height},
            "diagnostics": self.diagnostics,
            "colour_problems": dict(self.colour_problems),
            "assumption": CAPTURE_ASSUMPTION_ZH,
            "limit": RECOGNITION_LIMIT_ZH,
        }


def analyse_photo(image_bytes: bytes, *, mode: str,
                  method: str = RECOGNISE_CV, checkpoint=None,
                  device=None) -> PhotoAnalysis:
    """Detect and label the bricks in one photograph, and read their colours.

    ``single`` runs the single-brick classifier over the whole image and
    reports one item; ``multi`` runs the two-stage detector.  Both come back as
    editable items, because the correction interface does not care which task
    produced them.

    A colour that cannot be read is recorded as a problem for that item rather
    than defaulted: a crop with too little surface to measure has no colour,
    and saying so is what puts it in front of a person.
    """
    from src.colour.recognise import RecogniseError, read_colour
    from src.vision.detect import detect
    from src.vision.preprocess import ImageError, decode_image

    if mode not in PHOTO_MODES:
        raise UiError(f"照片模式 {mode!r} 不在 {list(PHOTO_MODES)} 之內")
    if method not in RECOGNISE_METHODS:
        raise UiError(f"辨識方法 {method!r} 不在 {list(RECOGNISE_METHODS)} 之內")
    try:
        loaded = decode_image(image_bytes)
    except ImageError as exc:
        raise UiError(f"這張圖片被拒絕：{exc}") from None

    classify = None
    if method == RECOGNISE_LEARNED:
        classify = _learned_classifier(checkpoint, device)

    if mode == PHOTO_SINGLE:
        prediction = (classify(loaded.rgb) if classify is not None
                      else _cv_whole_image(loaded.rgb))
        from src.vision.detect import Detection, DetectionResult

        box = (0, 0, loaded.width, loaded.height)
        result = DetectionResult(
            detections=(Detection(box=box, prediction=prediction),),
            diagnostics={"mode": PHOTO_SINGLE,
                         "note": "the whole image is treated as one brick"},
            width=loaded.width, height=loaded.height, scale=1.0)
    else:
        result = detect(loaded.rgb, classify=classify)

    # The mask is computed once over the whole image and then sliced per box.
    # Re-segmenting each crop would be wrong in a way that is easy to miss: a
    # tight crop of a brick is almost all brick, so the border-based background
    # estimate samples the brick itself and every colour comes back as the
    # background's. That produced "white" for a red brick before this.
    from src.vision.segment import foreground_mask

    readings = {}
    problems = {}
    whole_mask, _threshold = foreground_mask(
        loaded.rgb.astype("float32"))
    for index, detection in enumerate(result.detections):
        x0, y0, x1, y1 = detection.box
        crop = loaded.rgb[y0:y1, x0:x1]
        mask = whole_mask[y0:y1, x0:x1]
        try:
            readings[index] = read_colour(crop, mask)
        except RecogniseError as exc:
            problems[index] = str(exc)
    items = corrections_module.items_from_detection(result, colours=readings)
    return PhotoAnalysis(
        mode=mode, method=method, items=items,
        diagnostics=result.diagnostics, width=loaded.width,
        height=loaded.height, colour_readings=readings,
        colour_problems=problems)


def _cv_whole_image(rgb):
    from src.vision.cv_baseline import classify_array

    return classify_array(rgb)


def _learned_classifier(checkpoint, device):
    """A classify callable from a fitted checkpoint, or a refusal saying why."""
    if not checkpoint:
        raise UiError(
            "選了學習模型，但沒有提供已訓練的 checkpoint 目錄。"
            "請改用傳統 CV baseline，或以 --checkpoint 指定 checkpoint。")
    target = Path(checkpoint)
    if not target.is_dir():
        raise UiError(f"找不到 checkpoint 目錄：{target}")
    try:
        from src.vision.model import ModelError, load, predict_arrays
    except ImportError as exc:
        raise UiError(
            "學習模型需要 torch 與 transformers；請安裝釘版的 vision "
            f"requirements，或改用傳統 CV baseline（{exc}）") from None
    try:
        model, _manifest, resolved = load(target, device=device)
    except ModelError as exc:
        raise UiError(f"這個 checkpoint 無法載入：{exc}") from None

    def classify(crop):
        return predict_arrays(model, [crop], device=resolved)[0]

    return classify


# ---------------------------------------------------------------------------
# Corrections
# ---------------------------------------------------------------------------

def apply_corrections(items, edits, *, width: int, height: int):
    """Apply operator edits, refusing anything outside the eight or the image."""
    try:
        return corrections_module.apply_edits(items, edits, width=width,
                                              height=height)
    except CorrectionError as exc:
        raise UiError(str(exc)) from None


def add_correction(items, *, part, count, colour_id, width, height):
    try:
        return corrections_module.add_item(
            items, part=part, count=count, colour_id=colour_id,
            width=width, height=height)
    except CorrectionError as exc:
        raise UiError(str(exc)) from None


def adopted(items) -> CorrectedInventory:
    """The adopted stock, through the project's own normalisation."""
    try:
        return corrections_module.adopt(items)
    except (CorrectionError, ColourError) as exc:
        raise UiError(str(exc)) from None


def adopted_spec(inventory: CorrectedInventory) -> str:
    """The adopted stock as the string ``parse_inventory`` validates.

    Deliberately a string: every inventory refusal in this project lives in
    that one function, and a corrected photograph goes through it exactly as a
    typed inventory does rather than beside it.
    """
    if not inventory.parts:
        raise UiError(
            "修正後的庫存是空的：所有項目都被刪除，或都還是未辨識。"
            "請至少指定一個零件的類別與數量。")
    return corrections_module.inventory_spec(inventory.parts)


# ---------------------------------------------------------------------------
# The three methods
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FullResult:
    """One result, in the shape all three methods produce."""

    method: str
    provenance: dict
    status: str
    evidence: dict = field(default_factory=dict, repr=False)
    text: str | None = None
    report: dict | None = field(default=None, repr=False)
    explanation: dict | None = field(default=None, repr=False)

    @property
    def ready(self) -> bool:
        return bool(self.report
                    and self.report.get("delivery", {}).get(
                        "static_delivery_ready"))

    @property
    def bricks(self) -> list:
        return parse_bricks(self.text) if self.text else []

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "method_label": METHOD_LABELS[self.method],
            "provenance": self.provenance,
            "status": self.status,
            "static_delivery_ready": self.ready,
            "n_bricks": len(self.bricks),
            "evidence": self.evidence,
        }


def run_rag(*, caption: str, inventory: dict[str, int], index_dir,
            catalog_path, top_n: int = 10, exclude_object_id=None,
            device: str = "cpu") -> FullResult:
    """Semantic retrieval, then the exact arithmetic, then the explanation."""
    from src.delivery.pipeline import load_train_catalog
    from src.demo.showcase import inspect_supplied
    from src.retrieval import embed as embed_module
    from src.retrieval import index as index_module
    from src.retrieval.explain import explain_result
    from src.retrieval.nlp import NlpError, extract
    from src.retrieval.search import SearchError, search
    from src.ui.app import load_delivery

    try:
        conditions = extract(caption)
    except NlpError as exc:
        raise UiError(str(exc)) from None
    try:
        catalog = load_train_catalog(catalog_path)
        embedder = embed_module.load(device=device)
        loaded = index_module.load(
            index_dir, expected_identity_digest=embedder.identity_digest(),
            expected_catalog_sha256=catalog.sha256,
            expected_split_manifest_sha256=catalog.split_manifest_sha256)
        result = search(loaded, catalog, embedder, conditions, inventory,
                        top_n=top_n, exclude_object_id=exclude_object_id)
    except (embed_module.EmbedError, index_module.IndexError_, SearchError,
            DeliveryError) as exc:
        raise UiError(str(exc)) from None

    explanation = explain_result(result, inventory)
    report = None
    text = None
    if result.selected is not None:
        from src.data.bricks import format_bricks

        text = format_bricks(list(result.selected.item.bricks))
        report = inspect_supplied(
            caption, inventory, text,
            origin=f"train-only:rag:{result.selected.item.catalog_id}",
            termination=None)
        checks = tuple(load_delivery().DELIVERY_CHECKS)
        report["delivery"] = {
            "method": METHOD_RAG,
            "selected_catalog_id": result.selected.item.catalog_id,
            "checks_used": list(checks),
            "static_delivery_ready": all(
                report["checks"].get(name) is True for name in checks),
            "note": ("the same nine static checks the delivery command line "
                     "uses. No decoder ran, so termination is not "
                     "applicable and stays null."),
        }
    return FullResult(
        method=METHOD_RAG,
        provenance={
            "retrieval": "multilingual sentence embedding, exact cosine",
            "embedding": loaded.embedding,
            "index_documents": loaded.size,
            "catalogue_file": Path(catalog.source).name,
            "catalogue_sha256": catalog.sha256,
            "split_manifest_sha256": catalog.split_manifest_sha256,
            "identity_digest": loaded.identity_digest,
            "model_loaded": "embedding only; no generation model",
            "phase_3c": "not authorised and not run",
        },
        status=result.status, evidence=result.as_dict(), text=text,
        report=report, explanation=explanation)


def run_pipeline(*, caption: str, inventory_spec_text: str, catalog_path,
                 top_n: int = 5, time_limit: float = 2.0,
                 seed: int = 0) -> FullResult:
    """The existing minimum F-pipeline, through the delivery payload itself."""
    import argparse

    from src.ui.app import load_delivery

    module = load_delivery()
    namespace = argparse.Namespace(
        mode="f-pipeline", caption=caption, inventory=inventory_spec_text,
        catalog=str(catalog_path), top_n=top_n, exclude_object_id=None,
        time_limit=time_limit, seed=seed)
    payload, report = module.make_payload(namespace)
    if payload["method"]["model_loaded"] is not False:
        raise UiError(
            "the delivery payload reports a loaded model on the F-pipeline "
            "path, which must never load one")
    return FullResult(
        method=METHOD_PIPELINE,
        provenance={**payload["method"], **payload["catalog"],
                    "cpsat": {"time_limit": time_limit, "seed": seed}},
        status=payload["result"]["status"],
        evidence=payload["result"],
        text=(report["result"]["text"] if report else None),
        report=report)


def run_project_model(*, caption: str, inventory: dict[str, int],
                      placement: bool = False, connectivity: str = "off",
                      device: str = "mps", seed: int = 0,
                      temperature: float = 0.6, max_bricks: int = 24,
                      root=None) -> FullResult:
    """One demonstration decode with the archived project model."""
    from src.ui.model_entry import (DEMONSTRATION_NOTICE, ModelEntryError,
                                    PLACEMENT_NOTICE, decode)

    try:
        decoded = decode(caption, inventory, placement=placement,
                         connectivity=connectivity, device=device, seed=seed,
                         temperature=temperature, max_bricks=max_bricks,
                         root=root)
    except ModelEntryError as exc:
        raise UiError(str(exc)) from None
    body = decoded.as_dict()
    return FullResult(
        method=METHOD_PROJECT,
        provenance={
            **body["identity"],
            "device": decoded.device,
            "seed": decoded.seed,
            "temperature": decoded.temperature,
            "placement_gate": decoded.placement,
            "connectivity": body["connectivity"],
            "placement_notice": (PLACEMENT_NOTICE if decoded.placement
                                 else None),
            "demonstration_notice": DEMONSTRATION_NOTICE,
            "phase_3c": "not authorised and not run",
            "seconds": body["seconds"],
        },
        status=("decoded_and_delivery_ready" if decoded.ready
                else "decoded_but_not_delivery_ready"),
        evidence=body, text=decoded.report["result"]["text"],
        report=decoded.report)


# ---------------------------------------------------------------------------
# Finishing: colours, build order, and the two outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finished:
    """A deliverable result with its colours, its build order and its files."""

    assignment: Assignment | None
    plan: object
    step_previews: tuple
    ldraw: str
    preview: bytes = field(repr=False)
    preview_width: int = 0
    preview_height: int = 0
    colour_problem: str | None = None
    plan_problem: str | None = None
    preview_media_type: str = "image/png"

    @property
    def colour_source(self) -> str:
        """Where the drawn colours came from: an assignment, or the shape key.

        The distinction is the whole of the consistency claim.  With an
        assignment, the file and both kinds of image carry the same colours.
        Without one, they do not -- the file carries the LDraw default and the
        images carry a per-shape legend -- and saying otherwise would be a
        claim nobody could check without opening the file.
        """
        return "assignment" if self.assignment else "part-key"

    def as_dict(self) -> dict:
        return {
            "colours": (self.assignment.as_dict() if self.assignment
                        else None),
            "colour_problem": self.colour_problem,
            "assembly": (self.plan.as_dict() if self.plan else None),
            "assembly_problem": self.plan_problem,
            "step_previews": len(self.step_previews),
            "colour_source": self.colour_source,
            "same_structure": (
                "the LDraw file, the 3-D preview and every step image are "
                "produced from one brick list and one colour assignment; the "
                "images are drawn in that assignment's palette values, so the "
                "picture and the download carry the same colours"
                if self.assignment else
                "the LDraw file, the 3-D preview and every step image are "
                "produced from one brick list. No colour stock was given, so "
                "there is no assignment: the file is written in the LDraw "
                "default colour and the images use a per-shape legend. They "
                "are the same structure, not the same colours"),
        }


def finish(result: FullResult, *, colour_stock: str | None = None,
           preferences=(), max_per_step: int = 1,
           title: str | None = None, step_previews: bool = True) -> Finished:
    """Assign colours, order the build, and render both outputs.

    Everything is derived from one brick list, so the file a person downloads,
    the image they looked at and the steps they will follow cannot disagree.

    A colour stock that cannot cover the structure, or a structure with no
    legal build order, is reported as a named problem on the page -- the rest
    of the result is still produced, because "your red 2x4s ran out" should not
    take the preview away.
    """
    from src.assembly.order import AssemblyError
    from src.assembly.order import plan as build_plan
    from src.assembly.order import to_ldr as steps_to_ldr
    from src.assembly.order import write_step_previews
    from src.demo.showcase import ShowcaseError
    from src.rendering.preview import PreviewError, write_preview

    if not result.ready or not result.text:
        raise UiError(
            "沒有通過靜態交付檢查的結構，因此不產生配色、組裝步驟、預覽或下載。")
    bricks = result.bricks

    assignment = None
    colour_problem = None
    if colour_stock:
        try:
            assignment = assign(bricks, parse_colour_stock(colour_stock),
                                preferences=preferences)
        except (AssignError, ColourError) as exc:
            colour_problem = str(exc)
    colours = assignment.colours() if assignment else None

    plan = None
    plan_problem = None
    try:
        plan = build_plan(bricks, max_per_step=max_per_step)
    except AssemblyError as exc:
        plan_problem = str(exc)

    with tempfile.TemporaryDirectory(prefix="brickagain-full-") as tmp:
        scratch = Path(tmp)
        try:
            if plan is not None:
                ldraw = steps_to_ldr(plan, colours=colours)
            else:
                from src.rendering.ldr import to_ldr

                ldraw = to_ldr(bricks, colours=colours)
        except (ShowcaseError, ValueError) as exc:
            raise UiError(f"LDraw 無法產生：{exc}") from None
        try:
            preview = write_preview(scratch / "preview.png", bricks,
                                    title=title,
                                    colours=colours).read_bytes()
        except PreviewError as exc:
            raise UiError(f"預覽無法產生：{exc}") from None
        images: list[bytes] = []
        if plan is not None and step_previews:
            written = write_step_previews(plan, scratch / "steps", title=title,
                                          colours=colours)
            images = [path.read_bytes() for path in written]

    from src.ui.app import png_size

    width, height = png_size(preview)
    return Finished(assignment=assignment, plan=plan,
                    step_previews=tuple(images), ldraw=ldraw,
                    preview=preview, preview_width=width,
                    preview_height=height, colour_problem=colour_problem,
                    plan_problem=plan_problem)
