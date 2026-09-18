"""The brick recogniser: one photograph in, editable items out.

Lifted out of :mod:`src.ui.full` unchanged.  ``scripts/62_visual_stress_pbr.py``
calls ``analyse_photo`` as its recognition entry point, and both the V2 PBR
corpus and the visual-stress runs pin that script's static import closure by
SHA-256.  While this lived in ``full.py``, the closure also covered
``src/ui/full.py``, ``src/ui/model_entry.py``, ``src/demo/showcase.py`` and
``src/ui/app.py`` -- four modules the recogniser never executes, three of them
carrying the interface's notice text.

Correcting a notice therefore moved a rendered archive's source manifest and
made it unverifiable.  That happened twice.  The manifests should pin the
recogniser, because a change to it really would move the scores; they should
not pin a sentence about Phase 3C.

Nothing here is new code.  :mod:`src.ui.full` re-exports every name, so its own
callers are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.ui import corrections as corrections_module
from src.ui.errors import UiError
from src.vision.classes import UNKNOWN


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
