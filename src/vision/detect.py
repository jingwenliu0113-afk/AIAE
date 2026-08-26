"""Multi-brick detection as two stages, each of which can be checked alone.

**Stage one finds bricks and does not name them.**  Deterministic segmentation
against the background produces one box per foreground blob.  It is a
single-class detector -- "brick" -- and it is evaluated as one, against the
public archive's boxes, which are also single-class.

**Stage two names each box.**  Every crop goes to a single-brick classifier:
either the traditional-CV one or the fine-tuned network.  Swapping stage two
and holding stage one fixed is what makes the CV/learned comparison a
comparison of classifiers rather than of two whole pipelines.

The consequence is worth stating in the negative, because a report can get it
wrong: because stage one is shared, the two configurations emit **the same
boxes**.  Their detection precision, recall and count error are therefore
equal by construction, and any difference in an average precision computed
over them comes from the confidence that ranked the boxes, not from finding
them better.  There is no detector comparison here to make, and none is
claimed.

That split is a deliberate choice over training a detector.  The public
multi-brick archive labels boxes ``brick`` with no per-brick class at all, so
there is nothing to fit an eight-class detector against; inventing class labels
for those boxes in order to have a detector to train would be inventing the
ground truth.  The two-stage route uses each dataset for what it actually
labels.

**The stated capture assumption is part of the result.**  Stage one needs a
plain background and bricks that do not overlap much.  Off that condition it
merges touching bricks into one box and splits a strongly specular one into
two, and both failures are visible in the signed count error rather than hidden
by it.  ``diagnostics`` carries what was dropped and why, so an image where
thirty bricks became four boxes says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.vision.classes import UNKNOWN
from src.vision.cv_baseline import classify_component
from src.vision.metrics import Box, iou
from src.vision.preprocess import crop_box, fit_long_side
from src.vision.schema import METHOD_CV, Prediction
from src.vision.segment import Component, components

#: Two boxes overlapping more than this are the same brick found twice.
NMS_IOU = 0.55

#: A box this much inside another is a part of it, not a brick of its own --
#: the case a plain IoU test misses, because a small box inside a large one has
#: a low IoU however completely it is contained.
CONTAINMENT = 0.86

#: A box narrower or shorter than this fraction of the image is not one of the
#: bricks these photographs were composed to show.
MIN_BOX_SIDE_FRACTION = 0.012

#: How many boxes one image may yield.  The archive's most crowded photograph
#: holds 32 bricks; the cap is well above that so it never silently truncates a
#: real scene, and it does stop a badly segmented image from producing
#: thousands.
MAX_BOXES = 96


@dataclass(frozen=True)
class Detection:
    """One found brick: where it is, what it was called, how sure that was."""

    box: tuple[int, int, int, int]
    prediction: Prediction
    component: Component = field(repr=False, default=None)

    @property
    def label(self) -> str:
        return self.prediction.label

    @property
    def confidence(self) -> float:
        return self.prediction.confidence

    @property
    def low_confidence(self) -> bool:
        return self.prediction.low_confidence

    def as_box(self) -> Box:
        """This detection as a scored box for the metrics.

        The score is the *classifier's* confidence in the label, not an
        objectness from stage one -- stage one is deterministic segmentation
        and produces no confidence at all.  Anything that ranks these boxes is
        therefore ranking by stage two, which is why
        :func:`~src.vision.metrics.detection_report` has to be told
        :data:`~src.vision.metrics.SCORE_CLASSIFIER_CONFIDENCE` and refuses to
        publish the result as detection average precision.
        """
        x0, y0, x1, y1 = self.box
        return Box(x0, y0, x1, y1, label=self.label,
                   score=self.prediction.confidence)

    def as_dict(self) -> dict:
        return {"box": list(self.box), **self.prediction.as_dict()}


@dataclass(frozen=True)
class DetectionResult:
    """Everything one image produced, including what it failed to produce."""

    detections: tuple[Detection, ...]
    diagnostics: dict
    width: int
    height: int
    scale: float

    @property
    def count(self) -> int:
        return len(self.detections)

    def counts_by_part(self) -> dict[str, int]:
        """Per-class counts, with abstentions in their own ``unknown`` entry.

        An abstention is never folded into a class.  A caller building an
        inventory from a photograph has to see that four boxes were found and
        one of them was not identified, because that one is exactly what the
        operator has to correct.
        """
        out: dict[str, int] = {}
        for detection in self.detections:
            out[detection.label] = out.get(detection.label, 0) + 1
        return dict(sorted(out.items()))

    def low_confidence_items(self) -> tuple[int, ...]:
        return tuple(i for i, d in enumerate(self.detections)
                     if d.low_confidence)

    def as_dict(self) -> dict:
        counts = self.counts_by_part()
        return {
            "image": {"width": self.width, "height": self.height,
                      "analysis_scale": round(self.scale, 5)},
            "found": self.count,
            "counts_by_part": counts,
            "unknown": counts.get(UNKNOWN, 0),
            "low_confidence_indices": list(self.low_confidence_items()),
            "detections": [d.as_dict() for d in self.detections],
            "diagnostics": self.diagnostics,
            "assumption": (
                "flat, mostly non-overlapping bricks on a plain background. "
                "Off that condition touching bricks merge into one box and a "
                "specular one can split into two"),
        }


def suppress(boxes, *, nms_iou: float = NMS_IOU,
             containment: float = CONTAINMENT) -> tuple[list[int], dict]:
    """Drop duplicate boxes, keeping the largest, and say what was dropped.

    Ordered by area rather than by score.  The scores here come from stage
    *two*, and stage two has not run yet when duplicates are removed; area is
    the only ordering available at this point and it is the right one for
    segmentation output, where a duplicate is usually a fragment of a bigger
    blob.
    """
    order = sorted(range(len(boxes)),
                   key=lambda i: (-boxes[i].area, boxes[i].x0, boxes[i].y0))
    kept: list[int] = []
    dropped_overlap = dropped_contained = 0
    for index in order:
        candidate = boxes[index]
        redundant = False
        for other in kept:
            if iou(boxes[other], candidate) >= nms_iou:
                dropped_overlap += 1
                redundant = True
                break
            inside = _containment(candidate, boxes[other])
            if inside >= containment:
                dropped_contained += 1
                redundant = True
                break
        if not redundant:
            kept.append(index)
    kept.sort(key=lambda i: (boxes[i].y0, boxes[i].x0))
    return kept, {"dropped_overlapping": dropped_overlap,
                  "dropped_contained": dropped_contained}


def _containment(inner: Box, outer: Box) -> float:
    left = max(inner.x0, outer.x0)
    top = max(inner.y0, outer.y0)
    right = min(inner.x1, outer.x1)
    bottom = min(inner.y1, outer.y1)
    if right <= left or bottom <= top:
        return 0.0
    return ((right - left) * (bottom - top)) / inner.area


def propose(rgb: np.ndarray, *, long_side: int | None = None
            ) -> tuple[list[Component], np.ndarray, float, dict]:
    """Stage one: boxes for every plausible brick, with the rejects counted.

    The image is analysed downscaled, so the cost is bounded whatever came in,
    and the boxes are scaled back to the original pixels afterwards.  The scale
    factor is returned rather than applied silently, because a report has to be
    able to say what the boxes are in.
    """
    from src.vision.preprocess import CV_LONG_SIDE

    original = np.asarray(rgb, dtype=np.float32)
    target = CV_LONG_SIDE if long_side is None else long_side
    small = fit_long_side(original, target)
    scale = max(original.shape[0], original.shape[1]) / max(
        small.shape[0], small.shape[1])
    found, diagnostics = components(small, limit=MAX_BOXES)
    floor = max(2.0, MIN_BOX_SIDE_FRACTION * max(small.shape[0],
                                                 small.shape[1]))
    too_thin = 0
    kept: list[Component] = []
    for component in found:
        x0, y0, x1, y1 = component.box
        if (x1 - x0) < floor or (y1 - y0) < floor:
            too_thin += 1
            continue
        kept.append(component)
    diagnostics = {**diagnostics, "dropped_thin": too_thin,
                   "analysis_long_side": target,
                   "proposals": len(kept)}
    return kept, small, scale, diagnostics


def detect(rgb: np.ndarray, *, classify=None, long_side: int | None = None
           ) -> DetectionResult:
    """Find bricks and label them.

    ``classify`` takes an RGB crop and returns a
    :class:`~src.vision.schema.Prediction`.  Left out, the traditional-CV
    classifier runs on the component that stage one already measured, which is
    both cheaper and more faithful: the label is then computed from the same
    pixels and the same measurements as the box.

    An image with no foreground comes back with no detections and a diagnostic
    saying so.  That is a valid answer -- an empty tray is an empty tray -- and
    it is not an error.
    """
    original = np.asarray(rgb, dtype=np.float32)
    proposals, small, scale, diagnostics = propose(original,
                                                   long_side=long_side)
    small_boxes = [Box(*component.box) for component in proposals]
    kept, suppression = suppress(small_boxes)
    diagnostics = {**diagnostics, **suppression, "kept_boxes": len(kept)}

    detections: list[Detection] = []
    for index in kept:
        component = proposals[index]
        if classify is None:
            prediction = classify_component(small, component)
        else:
            prediction = classify(crop_box(small, component.box, pad=2))
        x0, y0, x1, y1 = component.box
        box = (int(round(x0 * scale)), int(round(y0 * scale)),
               max(int(round(x0 * scale)) + 1, int(round(x1 * scale))),
               max(int(round(y0 * scale)) + 1, int(round(y1 * scale))))
        detections.append(Detection(box=box, prediction=prediction,
                                    component=component))
    detections.sort(key=lambda d: (d.box[1], d.box[0]))
    if not detections:
        diagnostics = {**diagnostics, "empty_reason": (
            "no foreground component survived the area and shape filters; the "
            "image is reported as holding no bricks rather than as a failure")}
    return DetectionResult(
        detections=tuple(detections), diagnostics=diagnostics,
        width=int(original.shape[1]), height=int(original.shape[0]),
        scale=float(scale))


def learned_classifier(model, device: str, *, batch_size: int = 32):
    """A ``classify`` callable backed by the fine-tuned network.

    Returned as a closure so :func:`detect` never has to know whether a torch
    model exists.  One crop at a time: a detection image holds tens of boxes,
    not thousands, and batching them would buy little for the extra state.
    """
    from src.vision.model import predict_arrays

    def classify(crop) -> Prediction:
        return predict_arrays(model, [crop], device=device,
                             batch_size=batch_size)[0]

    return classify


def counts_to_inventory(result: DetectionResult) -> tuple[dict[str, int], dict]:
    """Turn a detection result into a stock dictionary and its caveats.

    Abstentions and anything not one of the eight are excluded from the stock
    and reported instead.  A photograph is not an inventory until a person has
    looked at what the model could not name -- so what comes back here is a
    *proposal*, and the fields that say how much of it is uncertain travel with
    it rather than beside it.
    """
    counts = result.counts_by_part()
    unknown = counts.pop(UNKNOWN, 0)
    caveats = {
        "unidentified_boxes": unknown,
        "low_confidence_boxes": len(result.low_confidence_items()),
        "needs_review": bool(unknown or result.low_confidence_items()),
        "note": ("boxes the classifier declined to name are not counted into "
                 "any part. This is a proposed inventory for a person to "
                 "correct, not a measured one"),
    }
    return {part: n for part, n in counts.items() if n > 0}, caveats
