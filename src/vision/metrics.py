"""Classification and detection metrics, computed one way.

Every number the vision work reports comes from here, so the traditional CV
baseline and the learned classifier are scored by the same code on the same
frozen items.  Two scoring functions is how a comparison becomes a comparison
of the scoring functions.

Two conventions are load bearing:

* **An abstention is not a correct answer and not a wrong class.**  A
  low-confidence prediction reports ``unknown``, and ``unknown`` is counted in
  its own row of the confusion matrix.  Folding it into the top-1 class would
  turn "the method declined" into "the method was right", and dropping those
  items from the denominator would let a method score well by answering only
  the easy ones.  Coverage is therefore reported beside accuracy, always.
* **Synthetic and real are never pooled.**  :func:`classification_report`
  scores one population.  The caller runs it twice and prints two blocks; a
  single averaged figure over renders and photographs would be dominated by
  whichever is more numerous, which in the public set is renders by nine to
  one.
* **A detection score has to say what it is.**  Average precision integrates a
  precision-recall curve swept by *ranking* the predictions, so the number it
  produces is a statement about whatever quantity did the ranking.  Ranked by
  an objectness score it is a localisation metric.  Ranked by a classifier's
  confidence over boxes some other stage localised, it is a statement about
  that classifier's confidence calibration and says nothing about the boxes.
  :func:`detection_report` therefore requires the caller to declare which, and
  publishes the number under a key that names it.  See
  :data:`SCORE_SEMANTICS`.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.vision.classes import CLASS_ORDER, UNKNOWN

#: IoU at which a detection counts as matching a ground-truth box.
IOU_MATCH = 0.5

#: The score attached to each predicted box is a localisation confidence: it
#: rises with how sure the detector is that *there is an object here*.  Average
#: precision over such scores is a detection metric.
SCORE_OBJECTNESS = "objectness"

#: The score is a classifier's confidence in the label it gave a box that some
#: earlier stage produced.  Average precision over these scores measures how
#: well that confidence orders the earlier stage's proposals -- useful, and not
#: a measurement of localisation.  Two pipelines that share stage one and
#: differ only in stage two produce *identical* boxes, so their precision,
#: recall and count error are equal by construction and only this ordering
#: differs; the difference is a difference between classifiers.
SCORE_CLASSIFIER_CONFIDENCE = "classifier_confidence"

SCORE_SEMANTICS: tuple[str, ...] = (SCORE_OBJECTNESS,
                                    SCORE_CLASSIFIER_CONFIDENCE)


def _safe(numerator: float, denominator: float) -> float:
    """A rate with an empty denominator is absent, not zero.

    Returning 0.0 for "no items of this class" reads as "the method got none
    of them right", which is a different statement.
    """
    return float(numerator) / float(denominator) if denominator else float("nan")


# ---------------------------------------------------------------------------
# Single-brick classification
# ---------------------------------------------------------------------------

def confusion(truth, predicted, *, classes=CLASS_ORDER) -> dict:
    """Confusion counts keyed ``truth -> predicted``, with an unknown column.

    Rows are the eight true classes.  Columns are the eight plus ``unknown``,
    because a method is allowed to decline and the matrix has to show how
    often it did and on what.
    """
    order = tuple(classes)
    columns = order + (UNKNOWN,)
    matrix = {t: {p: 0 for p in columns} for t in order}
    for actual, guess in zip(truth, predicted):
        if actual not in matrix:
            raise ValueError(
                f"true label {actual!r} is not one of the classes; a "
                "confusion matrix cannot have a row it was not given")
        if guess not in columns:
            raise ValueError(
                f"predicted label {guess!r} is neither a class nor {UNKNOWN!r}")
        matrix[actual][guess] += 1
    return matrix


def classification_report(truth, predictions, *, population: str,
                          classes=CLASS_ORDER) -> dict:
    """Score one population of single-brick predictions.

    ``predictions`` are :class:`~src.vision.schema.Prediction` objects, so the
    report reads both the acted-on ``label`` -- which may abstain -- and the
    ranked ``top3``, and never has to reconstruct either.
    """
    order = tuple(classes)
    truth = list(truth)
    predictions = list(predictions)
    if len(truth) != len(predictions):
        raise ValueError(
            f"{len(truth)} true labels against {len(predictions)} predictions")
    if not truth:
        raise ValueError("an empty population has no metrics; say so instead")

    labels = [p.label for p in predictions]
    top1 = [p.candidates[0].part for p in predictions]
    matrix = confusion(truth, labels, classes=order)

    correct = sum(1 for t, p in zip(truth, labels) if t == p)
    abstained = sum(1 for p in labels if p == UNKNOWN)
    top1_forced = sum(1 for t, p in zip(truth, top1) if t == p)
    top3 = sum(1 for t, p in zip(truth, predictions) if t in p.top_k(3))

    per_class = {}
    macro_f1_terms = []
    for name in order:
        support = sum(matrix[name].values())
        tp = matrix[name][name]
        fp = sum(matrix[other][name] for other in order if other != name)
        fn = support - tp
        precision = _safe(tp, tp + fp)
        recall = _safe(tp, support)
        if precision == precision and recall == recall and (
                precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if support else float("nan")
        per_class[name] = {
            "support": support, "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "abstained": matrix[name][UNKNOWN],
            "precision": precision, "recall": recall, "f1": f1,
        }
        if support:
            macro_f1_terms.append(f1)

    return {
        "population": population,
        "n": len(truth),
        "accuracy": _safe(correct, len(truth)),
        "accuracy_note": (
            "an abstention counts as wrong; coverage is reported separately "
            "so a method cannot look accurate by answering only easy items"),
        "coverage": _safe(len(truth) - abstained, len(truth)),
        "abstentions": abstained,
        "forced_top1_accuracy": _safe(top1_forced, len(truth)),
        "forced_top1_note": (
            "the same predictions with the confidence threshold ignored; "
            "reported so the threshold's effect is visible, not as the "
            "headline"),
        "top3_accuracy": _safe(top3, len(truth)),
        "macro_f1": (sum(macro_f1_terms) / len(macro_f1_terms)
                     if macro_f1_terms else float("nan")),
        "macro_f1_note": "mean over classes with at least one true item",
        "per_class": per_class,
        "confusion": matrix,
        "classes": list(order),
    }


def most_confused(report: dict, *, limit: int = 8) -> list[dict]:
    """The largest off-diagonal cells, for the error analysis."""
    out = []
    for actual, row in report["confusion"].items():
        for guess, count in row.items():
            if guess != actual and count:
                out.append({"truth": actual, "predicted": guess,
                            "count": count})
    out.sort(key=lambda cell: (-cell["count"], cell["truth"],
                               cell["predicted"]))
    return out[:limit]


# ---------------------------------------------------------------------------
# Multi-brick detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Box:
    """An axis-aligned box with exclusive far edges, and what it claims."""

    x0: int
    y0: int
    x1: int
    y1: int
    label: str | None = None
    score: float = 1.0

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(
                f"box ({self.x0},{self.y0},{self.x1},{self.y1}) has no area")

    @property
    def area(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def as_dict(self) -> dict:
        return {"box": [self.x0, self.y0, self.x1, self.y1],
                "label": self.label, "score": round(float(self.score), 6)}


def iou(a: Box, b: Box) -> float:
    left = max(a.x0, b.x0)
    top = max(a.y0, b.y0)
    right = min(a.x1, b.x1)
    bottom = min(a.y1, b.y1)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    return overlap / (a.area + b.area - overlap)


def match_boxes(truth, predicted, *, threshold: float = IOU_MATCH):
    """Greedy highest-score-first matching, one truth box per prediction.

    Greedy by descending score is the convention every detection benchmark
    uses, and it is what makes a second box on the same brick a false positive
    rather than a second true positive.  Returns ``(pairs, unmatched_truth,
    unmatched_predicted)`` as index lists.
    """
    truth = list(truth)
    predicted = list(predicted)
    order = sorted(range(len(predicted)),
                   key=lambda i: (-predicted[i].score, i))
    taken: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for p in order:
        best, best_iou = None, threshold
        for t in range(len(truth)):
            if t in taken:
                continue
            value = iou(truth[t], predicted[p])
            if value >= best_iou:
                best, best_iou = t, value
        if best is not None:
            taken.add(best)
            pairs.append((best, p))
    matched_pred = {p for _t, p in pairs}
    return (sorted(pairs),
            [t for t in range(len(truth)) if t not in taken],
            [p for p in range(len(predicted)) if p not in matched_pred])


def average_precision(records, n_truth: int) -> float:
    """Average precision by the all-point interpolation of the PR curve.

    ``records`` are ``(score, is_true_positive)`` over the whole population.
    All-point rather than the 11-point interpolation, and stated here, because
    the two give different numbers for the same detector and a report that
    does not say which is not reproducible.
    """
    if n_truth <= 0:
        return float("nan")
    ordered = sorted(records, key=lambda r: (-r[0],  not r[1]))
    tp = fp = 0
    points: list[tuple[float, float]] = []
    for _score, hit in ordered:
        if hit:
            tp += 1
        else:
            fp += 1
        points.append((tp / n_truth, tp / (tp + fp)))
    # Make precision monotone from the right, then integrate over recall.
    best = 0.0
    envelope: list[tuple[float, float]] = []
    for recall, precision in reversed(points):
        best = max(best, precision)
        envelope.append((recall, best))
    envelope.reverse()
    area = 0.0
    previous = 0.0
    for recall, precision in envelope:
        area += (recall - previous) * precision
        previous = recall
    return float(area)


def per_class_average_precision(images, *, classes=CLASS_ORDER,
                                threshold: float = IOU_MATCH) -> dict:
    """Average precision per class, and their mean: a real mAP.

    This is what "mAP@50 over eight classes" means, and it needs ground-truth
    boxes that carry a class.  Matching is done *within* a class, so a box
    correctly placed and wrongly named is a false positive for the name it
    claimed and a miss for the name it should have had -- which is the whole
    difference between an eight-class mAP and a single-class one.

    It is written here because the public multi-brick archive cannot support
    it: those boxes are labelled ``brick`` and nothing else.  Anything that
    needs a real mAP later has an implementation to call rather than a
    single-class number to relabel.
    """
    order = tuple(classes)
    per_class: dict[str, dict] = {}
    for name in order:
        records: list[tuple[float, bool]] = []
        n_truth = 0
        for truth, predicted in images:
            truth_of_class = [box for box in truth if box.label == name]
            predicted_of_class = [box for box in predicted
                                  if box.label == name]
            n_truth += len(truth_of_class)
            pairs, _missed, _extra = match_boxes(
                truth_of_class, predicted_of_class, threshold=threshold)
            hit = {p for _t, p in pairs}
            for i, box in enumerate(predicted_of_class):
                records.append((float(box.score), i in hit))
        per_class[name] = {
            "truth_boxes": n_truth,
            "predicted_boxes": len(records),
            "average_precision": average_precision(records, n_truth),
        }
    scored = [body["average_precision"] for body in per_class.values()
              if body["truth_boxes"]]
    return {
        "iou_threshold": threshold,
        "per_class": per_class,
        "classes_with_truth": len(scored),
        "mean_average_precision": (sum(scored) / len(scored) if scored
                                   else float("nan")),
        "mean_note": (
            "mean over classes that have at least one true box; a class with "
            "no ground truth has no average precision and is not counted as "
            "zero"),
    }


def truth_has_classes(images, *, classes=CLASS_ORDER) -> bool:
    """Whether every ground-truth box carries one of the known classes."""
    order = set(classes)
    seen = False
    for truth, _predicted in images:
        for box in truth:
            seen = True
            if box.label not in order:
                return False
    return seen


def detection_report(images, *, population: str, score_semantics: str,
                     threshold: float = IOU_MATCH,
                     per_class_truth: bool = True,
                     stage_one_shared: bool = False) -> dict:
    """Score detection over a population of images.

    ``images`` is a sequence of ``(truth_boxes, predicted_boxes)``.

    ``score_semantics`` has no default and must be one of
    :data:`SCORE_SEMANTICS`.  It decides what the average precision is called,
    because average precision is a statement about whatever ranked the
    predictions:

    * :data:`SCORE_OBJECTNESS` -- the scores rank boxes by how sure the
      detector is that a brick is there, so the number is a detection metric
      and is published as ``average_precision_50``;
    * :data:`SCORE_CLASSIFIER_CONFIDENCE` -- the scores are a *classifier's*
      confidence in the label it put on a box some earlier stage found.  The
      curve then measures how well that confidence orders the earlier stage's
      proposals.  It is published as
      ``class_agnostic_ap50_by_classifier_confidence`` and
      ``average_precision_50`` is ``None``, so a later reader cannot lift the
      familiar key and call the number detection average precision.

    ``stage_one_shared`` says the boxes came from a stage that is identical
    across the methods being compared.  Precision, recall and count error are
    then properties of that shared stage and are *equal by construction*
    between the methods, which the report states rather than leaving to be
    noticed.

    ``per_class_truth`` says whether the ground truth carries a class per box.
    In the public multi-brick set it does not -- the boxes are labelled
    ``brick`` and nothing more -- so per-class counting is *reported as
    unavailable* rather than computed against a label that was invented.  That
    is the honest reading of that dataset and it is recorded in the output, not
    only in prose.  Asking for per-class truth that is not there is refused,
    because a silent all-zero per-class table reads like a measurement.
    """
    if score_semantics not in SCORE_SEMANTICS:
        raise ValueError(
            f"score_semantics must be one of {list(SCORE_SEMANTICS)}, not "
            f"{score_semantics!r}. A detection average precision is a "
            "statement about whatever quantity ranked the predictions, so it "
            "cannot be computed without saying what that quantity was")
    images = list(images)
    if per_class_truth and not truth_has_classes(images):
        raise ValueError(
            "per_class_truth=True was requested, but the ground-truth boxes "
            "do not all carry one of the known classes. Pass "
            "per_class_truth=False so the per-class figures are reported as "
            "unavailable, rather than computed against labels that are not "
            "there")

    total_truth = 0
    records: list[tuple[float, bool]] = []
    matched = 0
    predicted_total = 0
    count_errors: list[int] = []
    per_class_abs_error: dict[str, list[int]] = {c: [] for c in CLASS_ORDER}
    label_pairs: list[tuple[str, str]] = []
    empty_images = 0

    for truth, predicted in images:
        truth = list(truth)
        predicted = list(predicted)
        total_truth += len(truth)
        predicted_total += len(predicted)
        if not truth:
            empty_images += 1
        pairs, _missed, extra = match_boxes(truth, predicted,
                                            threshold=threshold)
        matched += len(pairs)
        hit = {p for _t, p in pairs}
        for i, box in enumerate(predicted):
            records.append((float(box.score), i in hit))
        count_errors.append(len(predicted) - len(truth))
        if per_class_truth:
            for t, p in pairs:
                if truth[t].label and predicted[p].label:
                    label_pairs.append((truth[t].label, predicted[p].label))
            for name in CLASS_ORDER:
                want = sum(1 for b in truth if b.label == name)
                got = sum(1 for b in predicted if b.label == name)
                per_class_abs_error[name].append(abs(got - want))

    ap = average_precision(records, total_truth)
    report = {
        "population": population,
        "images": len(images),
        "iou_threshold": threshold,
        "truth_boxes": total_truth,
        "predicted_boxes": predicted_total,
        "matched_boxes": matched,
        "empty_truth_images": empty_images,
        "precision": _safe(matched, predicted_total),
        "recall": _safe(matched, total_truth),
        "score_semantics": score_semantics,
        "stage_one_shared": bool(stage_one_shared),
        "count_error_mean": (sum(count_errors) / len(count_errors)
                             if count_errors else float("nan")),
        "count_absolute_error_mean": (
            sum(abs(e) for e in count_errors) / len(count_errors)
            if count_errors else float("nan")),
        "count_error_note": (
            "predicted boxes minus true boxes, per image; the signed mean "
            "shows whether the detector splits or merges bricks"),
    }
    if score_semantics == SCORE_OBJECTNESS:
        report["average_precision_50"] = ap
        report["class_agnostic_ap50_by_classifier_confidence"] = None
        report["average_precision_note"] = (
            "all-point interpolation, single class 'brick', ranked by the "
            "detector's own objectness. With one class, mAP@50 and AP@50 are "
            "the same number -- and it is a one-class number, not an "
            "eight-class mAP")
    else:
        report["average_precision_50"] = None
        report["class_agnostic_ap50_by_classifier_confidence"] = ap
        report["average_precision_note"] = (
            "all-point interpolation over a single class 'brick', ranked by "
            "the CLASSIFIER's confidence in the label it gave each box. It "
            "measures how well that confidence orders the localiser's "
            "proposals; it is not a localisation metric and it is not an "
            "eight-class mAP. average_precision_50 is null on purpose so this "
            "number cannot be read as detection AP")
    report["not_a_detector_comparison"] = (
        "the boxes come from a stage that is the same for every method "
        "compared here, so precision, recall and the count errors are equal "
        "between them by construction and only the score ordering differs. "
        "Any difference in the class-agnostic average precision is a "
        "difference between classifiers' confidence, not between detectors"
        if stage_one_shared and score_semantics == SCORE_CLASSIFIER_CONFIDENCE
        else None)
    if per_class_truth:
        report["per_class_count_mae"] = {
            name: (sum(values) / len(values) if values else float("nan"))
            for name, values in per_class_abs_error.items()}
        report["mean_average_precision_50"] = per_class_average_precision(
            images, threshold=threshold)
        if label_pairs:
            report["matched_label_confusion"] = confusion(
                [t for t, _ in label_pairs], [p for _, p in label_pairs])
    else:
        report["per_class_count_mae"] = None
        report["mean_average_precision_50"] = None
        report["per_class_unavailable_reason"] = (
            "the ground-truth boxes in this population carry no per-brick "
            "class, so neither a per-class count error nor an eight-class "
            "mAP@50 can be computed. Both are reported as unavailable rather "
            "than computed against a label this project would have had to "
            "invent")
    return report
