"""Two-stage detection: boxes, then labels, and the cases that go wrong.

The detector's stated assumption is a plain background and bricks that do not
overlap much. These fixtures are drawn to be exactly that -- and then to break
it deliberately, because the interesting behaviour is what happens off the
assumption: touching bricks merge into one box, a duplicate is suppressed, an
empty tray is an empty tray rather than an error, and a box the classifier
cannot name is counted as unnamed instead of being filed under a class.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.vision.classes import CLASS_ORDER, UNKNOWN
from src.vision.detect import (CONTAINMENT, MAX_BOXES, NMS_IOU, Detection,
                               counts_to_inventory, detect, propose, suppress)
from src.vision.metrics import Box
from src.vision.schema import METHOD_CV, from_scores


def canvas(width=400, height=300, ground=(242, 242, 240)):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = ground
    return image


def brick(image, x0, y0, x1, y1, colour=(30, 90, 200)):
    image[y0:y1, x0:x1] = colour
    return image


class TestStageOneFindsBricks:
    def test_three_separated_bricks_give_three_boxes(self):
        image = canvas()
        brick(image, 20, 30, 100, 90)
        brick(image, 150, 30, 230, 90, colour=(200, 40, 40))
        brick(image, 280, 30, 360, 90, colour=(40, 160, 60))
        result = detect(image)
        assert result.count == 3

    def test_an_empty_tray_is_not_an_error(self):
        result = detect(canvas())
        assert result.count == 0
        assert "empty_reason" in result.diagnostics
        assert result.counts_by_part() == {}

    def test_the_boxes_are_in_original_pixel_coordinates(self):
        image = canvas(1200, 900)
        brick(image, 100, 200, 500, 400)
        result = detect(image)
        assert result.count == 1
        x0, y0, x1, y1 = result.detections[0].box
        assert 60 <= x0 <= 140 and 160 <= y0 <= 240
        assert 460 <= x1 <= 540 and 360 <= y1 <= 440
        assert result.width == 1200 and result.height == 900

    def test_the_diagnostics_say_what_was_dropped(self):
        image = canvas()
        brick(image, 20, 30, 100, 90)
        # a speck far below the area floor
        image[200:202, 200:202] = (10, 10, 10)
        _boxes, _small, _scale, diagnostics = propose(image)
        assert diagnostics["dropped_too_small"] >= 1
        assert "proposals" in diagnostics

    def test_two_touching_bricks_merge_and_the_count_error_shows_it(self):
        """Off the stated assumption. The failure is visible, not hidden."""
        image = canvas()
        brick(image, 20, 30, 100, 90)
        brick(image, 100, 30, 180, 90)
        result = detect(image)
        assert result.count == 1, "touching bricks merge under this method"

    def test_a_box_limit_is_reported_rather_than_silent(self):
        image = canvas(800, 600)
        for index in range(30):
            left = 10 + (index % 10) * 78
            top = 10 + (index // 10) * 190
            brick(image, left, top, left + 50, top + 120)
        _boxes, _small, _scale, diagnostics = propose(image)
        assert diagnostics["proposals"] <= MAX_BOXES


class TestDuplicateBoxes:
    def test_a_heavily_overlapping_box_is_suppressed(self):
        boxes = [Box(0, 0, 100, 100), Box(2, 2, 102, 102)]
        kept, report = suppress(boxes)
        assert len(kept) == 1
        assert report["dropped_overlapping"] == 1

    def test_a_box_contained_in_another_is_suppressed(self):
        """A small box inside a large one has a low IoU however completely it
        is contained, so the IoU test alone misses this case."""
        boxes = [Box(0, 0, 200, 200), Box(90, 90, 110, 110)]
        assert len(suppress(boxes)[0]) == 1
        kept, report = suppress(boxes)
        assert report["dropped_contained"] == 1

    def test_two_separate_boxes_both_survive(self):
        boxes = [Box(0, 0, 50, 50), Box(100, 100, 150, 150)]
        assert len(suppress(boxes)[0]) == 2

    def test_the_largest_is_the_one_kept(self):
        boxes = [Box(10, 10, 30, 30), Box(0, 0, 100, 100)]
        kept, _report = suppress(boxes)
        assert boxes[kept[0]].area == 100 * 100

    def test_the_thresholds_are_the_module_constants(self):
        assert 0 < NMS_IOU < 1 and 0 < CONTAINMENT <= 1


class TestStageTwoLabels:
    def test_the_default_classifier_labels_every_box(self):
        image = canvas()
        brick(image, 20, 30, 100, 90)
        brick(image, 150, 30, 230, 90)
        result = detect(image)
        assert all(d.prediction.method == METHOD_CV for d in result.detections)
        assert all(len(d.prediction.candidates) == 3
                   for d in result.detections)

    def test_a_supplied_classifier_is_used(self):
        image = canvas()
        brick(image, 20, 30, 100, 90)
        called = []

        def always_2x4(crop):
            called.append(crop.shape)
            scores = [0.0] * 8
            scores[CLASS_ORDER.index("2x4")] = 0.95
            return from_scores(METHOD_CV, scores)

        result = detect(image, classify=always_2x4)
        assert called
        assert result.counts_by_part() == {"2x4": 1}

    def test_an_unnamed_box_is_counted_as_unknown_not_as_a_class(self):
        image = canvas()
        brick(image, 20, 30, 100, 90)

        def undecided(_crop):
            return from_scores(METHOD_CV, [0.125] * 8)

        result = detect(image, classify=undecided)
        counts = result.counts_by_part()
        assert counts == {UNKNOWN: 1}
        assert result.low_confidence_items() == (0,)

    def test_the_result_serialises_with_its_assumption_stated(self):
        image = canvas()
        brick(image, 20, 30, 100, 90)
        body = detect(image).as_dict()
        assert "non-overlapping" in body["assumption"]
        assert body["found"] == 1
        assert "diagnostics" in body


class TestTurningDetectionsIntoAProposedInventory:
    def _result(self, labels):
        image = canvas(600, 200)
        for index, _label in enumerate(labels):
            left = 20 + index * 120
            brick(image, left, 40, left + 90, 150)
        order = iter(labels)

        def label(_crop):
            want = next(order)
            scores = [0.0] * 8
            if want == UNKNOWN:
                return from_scores(METHOD_CV, [0.125] * 8)
            scores[CLASS_ORDER.index(want)] = 0.95
            return from_scores(METHOD_CV, scores)

        return detect(image, classify=label)

    def test_named_boxes_become_stock(self):
        result = self._result(["2x4", "2x4", "1x2"])
        stock, caveats = counts_to_inventory(result)
        assert stock == {"1x2": 1, "2x4": 2}
        assert caveats["needs_review"] is False

    def test_an_unnamed_box_is_excluded_and_reported(self):
        result = self._result(["2x4", UNKNOWN])
        stock, caveats = counts_to_inventory(result)
        assert stock == {"2x4": 1}
        assert caveats["unidentified_boxes"] == 1
        assert caveats["needs_review"] is True
        assert "not counted into any part" in caveats["note"]

    def test_the_proposal_is_labelled_a_proposal(self):
        result = self._result(["2x4"])
        _stock, caveats = counts_to_inventory(result)
        assert "not a measured one" in caveats["note"]


class TestDetectionRecords:
    def test_a_detection_exposes_its_box_and_label(self):
        scores = [0.0] * 8
        scores[CLASS_ORDER.index("1x6")] = 0.8
        detection = Detection(box=(1, 2, 11, 22),
                              prediction=from_scores(METHOD_CV, scores))
        assert detection.label == "1x6"
        assert detection.as_box().label == "1x6"
        assert detection.as_dict()["box"] == [1, 2, 11, 22]

    def test_a_low_confidence_detection_reports_unknown(self):
        detection = Detection(box=(0, 0, 4, 4),
                              prediction=from_scores(METHOD_CV, [0.125] * 8))
        assert detection.low_confidence
        assert detection.label == UNKNOWN
        assert detection.as_box().label == UNKNOWN
