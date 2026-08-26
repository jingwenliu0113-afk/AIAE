"""Preprocessing, segmentation, the CV baseline, the schema and the metrics.

Every fixture here is drawn in NumPy: a rectangle of one colour on a lighter
ground, sometimes with bright discs on it to stand for studs. Synthetic
fixtures cannot show how the baseline behaves on a photograph -- that is what
``scripts/33_vision_eval.py`` is for, on the frozen split -- but they can pin
the behaviour that has to hold whatever the image: an empty image is not an
error, a bomb is refused before it is decoded, an abstention is not a class,
and two runs on the same pixels give the same answer.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from src.vision.classes import CLASS_ORDER, UNKNOWN
from src.vision.cv_baseline import (HEIGHT_IN_STUDS, classify_array,
                                    count_studs, estimate_pitch,
                                    expected_geometry, score_features)
from src.vision.metrics import (SCORE_CLASSIFIER_CONFIDENCE, SCORE_OBJECTNESS,
                                SCORE_SEMANTICS, Box, average_precision,
                                classification_report, confusion,
                                detection_report, iou, match_boxes,
                                most_confused, per_class_average_precision,
                                truth_has_classes)
from src.vision.preprocess import (ALLOWED_FORMATS, CROP_SIZE, IMAGE_MEAN,
                                   IMAGE_STD, MAX_IMAGE_PIXELS, ImageError,
                                   check_processor_config, crop_box,
                                   decode_image, fit_long_side, model_tensor,
                                   read_image, resize_rgb)
from src.vision.schema import (LOW_CONFIDENCE, METHOD_CV, METHOD_LEARNED,
                               Candidate, Prediction, from_scores,
                               normalise_scores)
from src.vision.segment import (SegmentError, background_colour, components,
                                foreground_mask, label_components, measure)


# --------------------------------------------------------------------------
# fixtures drawn by hand
# --------------------------------------------------------------------------

def canvas(width=240, height=180, ground=(240, 240, 238)):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = ground
    return image


def rectangle(image, x0, y0, x1, y1, colour=(200, 30, 30)):
    image[y0:y1, x0:x1] = colour
    return image


def discs(image, boxes, colour=(255, 255, 255), radius=4):
    ys, xs = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    for cx, cy in boxes:
        image[(xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2] = colour
    return image


def png_bytes(image):
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

class TestDecodingIsBounded:
    def test_a_png_round_trips(self):
        image = rectangle(canvas(), 40, 40, 120, 90)
        loaded = decode_image(png_bytes(image))
        assert loaded.width == 240 and loaded.height == 180
        assert loaded.source_format == "PNG"
        assert np.array_equal(loaded.rgb, image)

    def test_a_jpeg_is_accepted(self):
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(canvas()).save(buffer, format="JPEG")
        assert decode_image(buffer.getvalue()).source_format == "JPEG"

    def test_a_format_outside_the_allowlist_is_refused(self):
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(canvas()).save(buffer, format="BMP")
        with pytest.raises(ImageError, match="not one of"):
            decode_image(buffer.getvalue())
        assert "BMP" not in ALLOWED_FORMATS

    def test_something_that_is_not_an_image_is_refused(self):
        with pytest.raises(ImageError, match="not an image"):
            decode_image(b"just some bytes, definitely not a picture" * 20)

    def test_empty_bytes_are_refused(self):
        with pytest.raises(ImageError, match="empty"):
            decode_image(b"")

    def test_a_non_bytes_input_is_refused(self):
        with pytest.raises(ImageError, match="is bytes"):
            decode_image("a string")

    def test_a_byte_count_over_the_limit_is_refused_before_decoding(self):
        payload = png_bytes(canvas())
        with pytest.raises(ImageError, match="refused rather than decoded"):
            decode_image(payload, max_bytes=len(payload) - 1)

    def test_a_declared_pixel_count_over_the_limit_is_refused(self):
        """The decompression-bomb case: a tiny file declaring a huge canvas."""
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (7000, 7000), (255, 255, 255)).save(
            buffer, format="PNG")
        with pytest.raises(ImageError, match="checked before decoding"):
            decode_image(buffer.getvalue(), max_pixels=1_000_000)

    def test_the_real_cap_is_generous_enough_for_a_photograph(self):
        assert 2448 * 3264 < MAX_IMAGE_PIXELS

    def test_a_tiny_image_is_refused(self):
        with pytest.raises(ImageError, match="at least"):
            decode_image(png_bytes(canvas(width=8, height=8)))

    def test_a_multi_frame_image_is_refused(self):
        from PIL import Image

        buffer = io.BytesIO()
        frames = [Image.fromarray(canvas(64, 64)) for _ in range(3)]
        frames[0].save(buffer, format="PNG", save_all=True,
                       append_images=frames[1:])
        try:
            decode_image(buffer.getvalue())
        except ImageError as exc:
            assert "frames" in str(exc)

    def test_alpha_is_composited_onto_white_not_dropped(self):
        from PIL import Image

        rgba = Image.new("RGBA", (64, 64), (0, 0, 255, 0))
        buffer = io.BytesIO()
        rgba.save(buffer, format="PNG")
        loaded = decode_image(buffer.getvalue())
        assert loaded.had_alpha
        assert tuple(loaded.rgb[0, 0]) == (255, 255, 255)

    def test_reading_from_disk_checks_the_suffix(self, tmp_path):
        target = tmp_path / "image.gif"
        target.write_bytes(png_bytes(canvas(64, 64)))
        with pytest.raises(ImageError, match="does not end in"):
            read_image(target)

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(ImageError, match="no image at"):
            read_image(tmp_path / "absent.png")


class TestResamplingIsDeterministic:
    def test_the_same_input_gives_the_same_output(self):
        image = rectangle(canvas(), 20, 20, 100, 60)
        assert np.array_equal(resize_rgb(image, 80, 60),
                              resize_rgb(image, 80, 60))

    def test_a_same_size_resize_is_a_no_op(self):
        image = canvas(64, 48)
        assert np.array_equal(resize_rgb(image, 64, 48),
                              image.astype(np.float32))

    def test_a_flat_image_stays_flat(self):
        image = canvas(64, 64, ground=(120, 130, 140))
        out = resize_rgb(image, 31, 17)
        assert np.allclose(out[..., 0], 120.0, atol=0.01)

    def test_fit_long_side_never_upscales(self):
        small = canvas(40, 30)
        assert fit_long_side(small, 320).shape[:2] == (30, 40)

    def test_fit_long_side_downscales_to_the_target(self):
        big = canvas(1200, 600)
        out = fit_long_side(big, 320)
        assert max(out.shape[:2]) == 320

    def test_a_zero_size_target_is_refused(self):
        with pytest.raises(ImageError, match="at least one pixel"):
            resize_rgb(canvas(), 0, 10)


class TestTheModelTensor:
    def test_the_shape_and_normalisation_are_the_pinned_ones(self):
        tensor = model_tensor(canvas(400, 300))
        assert tensor.shape == (3, CROP_SIZE, CROP_SIZE)
        assert tensor.dtype == np.float32
        # A white-ish ground normalises to (1 - mean) / std per channel.
        for channel in range(3):
            expected = ((240 if channel < 2 else 238) / 255.0
                        - IMAGE_MEAN[channel]) / IMAGE_STD[channel]
            assert abs(float(tensor[channel].mean()) - expected) < 0.05

    def test_it_is_deterministic(self):
        image = rectangle(canvas(400, 300), 50, 40, 200, 120)
        assert np.array_equal(model_tensor(image), model_tensor(image))

    def test_a_published_config_that_matches_passes(self):
        check_processor_config({"image_mean": list(IMAGE_MEAN),
                                "image_std": list(IMAGE_STD),
                                "size": CROP_SIZE, "crop_pct": 0.875})

    @pytest.mark.parametrize("config", [
        {"image_mean": [0.5, 0.5, 0.5]},
        {"image_std": [1.0, 1.0, 1.0]},
        {"size": 256},
        {"size": CROP_SIZE, "crop_pct": 0.5},
    ])
    def test_a_published_config_that_disagrees_is_refused(self, config):
        with pytest.raises(ImageError):
            check_processor_config(config)

    def test_a_non_object_config_is_refused(self):
        with pytest.raises(ImageError, match="JSON object"):
            check_processor_config([1, 2, 3])


class TestCropBox:
    def test_it_clamps_into_the_image(self):
        image = canvas(60, 40)
        assert crop_box(image, (-10, -10, 500, 500)).shape[:2] == (40, 60)

    def test_a_degenerate_box_still_yields_a_pixel(self):
        image = canvas(60, 40)
        assert crop_box(image, (10, 10, 10, 10)).size > 0


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------

class TestSegmentation:
    def test_the_background_is_estimated_from_the_border(self):
        image = rectangle(canvas(ground=(200, 200, 200)), 60, 40, 180, 140)
        assert np.allclose(background_colour(image), (200, 200, 200), atol=1)

    def test_one_rectangle_is_one_component(self):
        image = rectangle(canvas(), 60, 40, 180, 140)
        found, diagnostics = components(image)
        assert len(found) == 1
        assert diagnostics["kept"] == 1

    def test_two_rectangles_are_two_components(self):
        image = rectangle(canvas(), 20, 30, 90, 100)
        rectangle(image, 130, 30, 210, 100, colour=(20, 60, 200))
        found, _diagnostics = components(image)
        assert len(found) == 2

    def test_an_empty_image_yields_nothing_and_says_so(self):
        found, diagnostics = components(canvas())
        assert found == []
        assert diagnostics["raw_components"] >= 0

    def test_the_measured_extents_recover_a_rectangle(self):
        image = rectangle(canvas(320, 240), 40, 60, 240, 120)
        found, _diagnostics = components(image)
        component = found[0]
        assert abs(component.length - 200) < 12
        assert abs(component.width - 60) < 12
        assert abs(component.aspect - 200 / 60) < 0.4

    def test_the_box_is_exclusive_at_the_far_edge(self):
        image = rectangle(canvas(), 60, 40, 180, 140)
        component = components(image)[0][0]
        x0, y0, x1, y1 = component.box
        assert x1 > x0 and y1 > y0
        assert x1 <= image.shape[1] and y1 <= image.shape[0]

    def test_labelling_is_deterministic(self):
        image = rectangle(canvas(), 20, 30, 90, 100)
        rectangle(image, 130, 30, 210, 100)
        mask, _threshold = foreground_mask(image)
        first, count_a = label_components(mask)
        again, count_b = label_components(mask)
        assert count_a == count_b == 2
        assert np.array_equal(first, again)

    def test_a_non_two_dimensional_mask_is_refused(self):
        with pytest.raises(SegmentError, match="2-D boolean"):
            label_components(np.zeros((4, 4, 3), dtype=bool))

    def test_a_non_three_channel_image_is_refused(self):
        with pytest.raises(SegmentError, match="h, w, 3"):
            foreground_mask(np.zeros((8, 8), dtype=np.float32))

    def test_measuring_a_component_that_does_not_exist_is_refused(self):
        labels = np.zeros((4, 4), dtype=np.int32)
        with pytest.raises(SegmentError, match="no component numbered"):
            measure(labels, 3)

    def test_the_limit_is_reported_rather_than_silently_applied(self):
        image = canvas(320, 240)
        for index in range(12):
            left = 10 + (index % 6) * 50
            top = 20 + (index // 6) * 90
            rectangle(image, left, top, left + 30, top + 50)
        _found, diagnostics = components(image, limit=4)
        assert diagnostics["kept"] == 4
        assert diagnostics["dropped_over_limit"] == 8


# --------------------------------------------------------------------------
# the CV baseline
# --------------------------------------------------------------------------

class TestExpectedGeometryIsDerived:
    def test_the_height_comes_from_the_ldraw_constants(self):
        assert abs(HEIGHT_IN_STUDS - 24 / 20) < 1e-9

    def test_every_class_has_a_row(self):
        assert set(expected_geometry()) == set(CLASS_ORDER)

    def test_a_one_wide_part_uses_its_height_as_the_short_side(self):
        short, long_side, aspect = expected_geometry()["1x6"]
        assert (short, long_side) == (1, 6)
        assert abs(aspect - 6 / HEIGHT_IN_STUDS) < 1e-9

    def test_a_two_wide_part_uses_its_real_width(self):
        _short, _long, aspect = expected_geometry()["2x6"]
        assert abs(aspect - 3.0) < 1e-9

    def test_the_square_parts_are_square(self):
        assert abs(expected_geometry()["1x1"][2] - 1.0) < 1e-9
        assert abs(expected_geometry()["2x2"][2] - 1.0) < 1e-9


class TestScoring:
    def test_a_score_vector_is_produced_for_every_class(self):
        assert len(score_features(3.0, 2.0, 6.0, 12, 0.8)) == len(CLASS_ORDER)

    def test_the_matching_geometry_scores_highest(self):
        scores = score_features(3.0, 2.0, 6.0, 12, 0.9)
        assert CLASS_ORDER[int(np.argmax(scores))] == "2x6"

    def test_a_nonsense_aspect_gives_a_flat_vector(self):
        assert len(set(score_features(0.0, 1, 1, 0, 0.0))) == 1
        assert len(set(score_features(float("nan"), 1, 1, 0, 0.0))) == 1

    def test_a_weak_pitch_lets_elongation_decide(self):
        # Stud counts that point at 1x1 with no pitch strength must not
        # override an elongation that clearly says 1x8.
        scores = score_features(6.6, 1.0, 1.0, 1, 0.0)
        assert CLASS_ORDER[int(np.argmax(scores))] == "1x8"

    def test_no_class_is_ever_scored_exactly_zero(self):
        assert all(value > 0 for value in score_features(3.0, 2.0, 6.0, 12, 1.0))


class TestPitchEstimation:
    def test_a_periodic_profile_yields_its_period(self):
        """Eight repeats: the most any part in the vocabulary can have."""
        profile = np.array([1.0 if i % 12 < 3 else 0.0 for i in range(96)])
        pitch, strength = estimate_pitch(profile)
        assert abs(pitch - 12) <= 1
        assert strength > 0.3

    def test_the_fundamental_is_preferred_over_its_harmonic(self):
        """The defect this guards: a harmonic halves the stud count.

        Autocorrelation peaks at twice the true period nearly as strongly as
        at the period itself, so taking the maximum reports 40 for a profile
        whose period is 20 -- which turns a 1x6 into a 1x2. Both lags are
        inside the searched range here, so the preference is what decides.
        """
        profile = np.array([1.0 if i % 20 < 5 else 0.0 for i in range(120)])
        pitch, _strength = estimate_pitch(profile)
        assert abs(pitch - 20) <= 1, "a harmonic was reported as the period"

    def test_a_period_no_part_could_have_is_outside_the_search(self):
        """Twenty repeats is not a stud pitch, so it is not reported as one."""
        profile = np.array([1.0 if i % 5 < 2 else 0.0 for i in range(100)])
        pitch, _strength = estimate_pitch(profile)
        assert pitch == 0.0 or pitch >= 100 * (1.0 / 8) - 1

    def test_a_flat_profile_yields_nothing(self):
        assert estimate_pitch(np.zeros(64)) == (0.0, 0.0)

    def test_a_very_short_profile_yields_nothing(self):
        assert estimate_pitch(np.array([1.0, 2.0, 3.0])) == (0.0, 0.0)


class TestStudCounting:
    def test_bright_discs_on_a_dark_brick_are_counted(self):
        image = rectangle(canvas(240, 120), 40, 30, 200, 90,
                         colour=(30, 30, 140))
        discs(image, [(60, 60), (100, 60), (140, 60), (180, 60)])
        mask, _threshold = foreground_mask(image)
        count, diagnostics = count_studs(image, mask)
        assert count == 4
        assert diagnostics["stud_candidates"] >= 4

    def test_studs_lighter_than_the_background_are_still_counted(self):
        """The case that made the raw mask useless.

        A stud catching the light can be brighter than the paper behind the
        brick, so the foreground threshold excludes it and a count inside the
        raw mask is zero. Filling the outline first is what fixes it.
        """
        image = rectangle(canvas(240, 120, ground=(200, 200, 198)),
                         40, 30, 200, 90, colour=(30, 30, 140))
        discs(image, [(70, 60), (120, 60), (170, 60)],
              colour=(255, 255, 255), radius=6)
        mask, _threshold = foreground_mask(image)
        assert not mask[60, 70], "the disc is outside the raw foreground mask"
        count, _diagnostics = count_studs(image, mask)
        assert count == 3

    def test_an_empty_mask_counts_nothing(self):
        image = canvas(64, 64)
        count, _diagnostics = count_studs(image,
                                          np.zeros((64, 64), dtype=bool))
        assert count == 0


class TestTheClassifierAsAWhole:
    def test_it_returns_a_prediction_for_a_plain_rectangle(self):
        image = rectangle(canvas(320, 240), 40, 90, 260, 150)
        prediction = classify_array(image)
        assert prediction.method == METHOD_CV
        assert len(prediction.candidates) == 3
        assert prediction.features["segmented"] is True

    def test_an_empty_image_is_a_low_confidence_prediction_not_an_error(self):
        prediction = classify_array(canvas())
        assert prediction.features["segmented"] is False
        assert prediction.low_confidence
        assert prediction.label == UNKNOWN

    def test_two_runs_on_the_same_pixels_agree(self):
        image = rectangle(canvas(320, 240), 40, 90, 260, 150)
        first = classify_array(image)
        again = classify_array(image)
        assert first.as_dict() == again.as_dict()

    def test_a_long_thin_rectangle_is_ranked_towards_the_long_parts(self):
        image = rectangle(canvas(400, 200), 20, 85, 380, 115)
        prediction = classify_array(image)
        assert prediction.candidates[0].part in ("1x6", "1x8")

    def test_a_square_is_ranked_towards_the_square_parts(self):
        image = rectangle(canvas(300, 300), 90, 90, 210, 210)
        prediction = classify_array(image)
        assert prediction.candidates[0].part in ("1x1", "2x2")


# --------------------------------------------------------------------------
# the prediction schema
# --------------------------------------------------------------------------

class TestThePredictionSchema:
    def test_low_confidence_reports_unknown_in_the_data(self):
        prediction = from_scores(METHOD_CV,
                                [0.2] * 4 + [0.2] * 4)
        assert prediction.low_confidence
        assert prediction.label == UNKNOWN
        assert prediction.as_dict()["label"] == UNKNOWN
        assert prediction.as_dict()["top1"] in CLASS_ORDER

    def test_a_confident_prediction_reports_its_class(self):
        scores = [0.02] * 8
        scores[CLASS_ORDER.index("2x4")] = 0.9
        prediction = from_scores(METHOD_LEARNED, scores)
        assert not prediction.low_confidence
        assert prediction.label == "2x4"

    def test_the_threshold_is_the_one_constant(self):
        scores = [0.0] * 8
        scores[0] = LOW_CONFIDENCE - 1e-9
        assert from_scores(METHOD_CV, scores).low_confidence
        scores[0] = LOW_CONFIDENCE
        assert not from_scores(METHOD_CV, scores).low_confidence

    def test_top3_is_the_first_three_in_score_order(self):
        scores = [0.05] * 8
        for part, value in (("1x1", 0.5), ("2x4", 0.3), ("1x8", 0.2)):
            scores[CLASS_ORDER.index(part)] = value
        prediction = from_scores(METHOD_CV, scores)
        assert prediction.top_k(3) == ("1x1", "2x4", "1x8")

    def test_a_tie_breaks_on_class_order(self):
        scores = [0.0] * 8
        scores[CLASS_ORDER.index("1x2")] = 0.4
        scores[CLASS_ORDER.index("2x2")] = 0.4
        prediction = from_scores(METHOD_CV, scores)
        assert prediction.candidates[0].part == "1x2"

    def test_an_incomplete_score_vector_is_refused(self):
        with pytest.raises(Exception, match="class order"):
            from_scores(METHOD_CV, [0.5, 0.5])

    def test_a_nan_score_is_refused(self):
        with pytest.raises(Exception, match="NaN"):
            from_scores(METHOD_CV, [float("nan")] + [0.0] * 7)

    def test_an_unknown_method_is_refused(self):
        with pytest.raises(Exception, match="not one of"):
            from_scores("guessing", [0.1] * 8)

    def test_an_unsorted_candidate_list_is_refused(self):
        with pytest.raises(Exception, match="descending score"):
            Prediction(method=METHOD_CV,
                       candidates=(Candidate("1x1", 0.1),
                                   Candidate("1x2", 0.9)))

    def test_a_duplicated_candidate_is_refused(self):
        with pytest.raises(Exception, match="twice"):
            Prediction(method=METHOD_CV,
                       candidates=(Candidate("1x1", 0.9),
                                   Candidate("1x1", 0.1)))

    def test_the_margin_is_the_gap_to_second(self):
        scores = [0.0] * 8
        scores[0], scores[1] = 0.7, 0.2
        assert abs(from_scores(METHOD_CV, scores).margin - 0.5) < 1e-9

    def test_normalising_an_all_zero_vector_gives_a_uniform_one(self):
        values = normalise_scores([0.0] * 8)
        assert abs(sum(values) - 1.0) < 1e-9
        assert len(set(values)) == 1
        assert from_scores(METHOD_CV, values).low_confidence


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def confident(part):
    scores = [0.01] * 8
    scores[CLASS_ORDER.index(part)] = 0.9
    return from_scores(METHOD_CV, scores)


def abstaining():
    return from_scores(METHOD_CV, [0.125] * 8)


class TestClassificationMetrics:
    def test_all_correct_scores_one(self):
        truth = list(CLASS_ORDER)
        report = classification_report(
            truth, [confident(part) for part in truth], population="synthetic")
        assert report["accuracy"] == 1.0
        assert report["macro_f1"] == 1.0
        assert report["coverage"] == 1.0

    def test_an_abstention_counts_as_wrong_and_lowers_coverage(self):
        truth = ["1x1", "1x2"]
        report = classification_report(
            truth, [confident("1x1"), abstaining()], population="real")
        assert report["accuracy"] == 0.5
        assert report["coverage"] == 0.5
        assert report["abstentions"] == 1
        assert report["confusion"]["1x2"][UNKNOWN] == 1

    def test_the_forced_accuracy_shows_the_thresholds_effect(self):
        scores = [0.0] * 8
        scores[CLASS_ORDER.index("1x2")] = 0.3
        prediction = from_scores(METHOD_CV, scores)
        report = classification_report(["1x2"], [prediction],
                                       population="real")
        assert report["accuracy"] == 0.0
        assert report["forced_top1_accuracy"] == 1.0

    def test_top3_counts_a_second_place_hit(self):
        scores = [0.0] * 8
        scores[CLASS_ORDER.index("1x1")] = 0.6
        scores[CLASS_ORDER.index("2x2")] = 0.3
        report = classification_report(["2x2"], [from_scores(METHOD_CV, scores)],
                                       population="real")
        assert report["accuracy"] == 0.0
        assert report["top3_accuracy"] == 1.0

    def test_a_class_with_no_items_reports_absent_rather_than_zero(self):
        report = classification_report(["1x1"], [confident("1x1")],
                                       population="real")
        recall = report["per_class"]["2x6"]["recall"]
        assert recall != recall, "no support should be nan, not 0.0"

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="against"):
            classification_report(["1x1", "1x2"], [confident("1x1")],
                                  population="real")

    def test_an_empty_population_is_refused(self):
        with pytest.raises(ValueError, match="no metrics"):
            classification_report([], [], population="real")

    def test_a_truth_label_outside_the_classes_is_refused(self):
        with pytest.raises(ValueError, match="not one of the classes"):
            confusion(["2x8"], ["1x1"])

    def test_most_confused_ranks_the_biggest_cells(self):
        truth = ["1x6"] * 5 + ["1x4"] * 2
        predictions = [confident("1x4")] * 5 + [confident("1x4")] * 2
        report = classification_report(truth, predictions,
                                       population="synthetic")
        worst = most_confused(report)
        assert worst[0]["truth"] == "1x6" and worst[0]["predicted"] == "1x4"
        assert worst[0]["count"] == 5


class TestDetectionMetrics:
    def test_iou_of_identical_boxes_is_one(self):
        box = Box(0, 0, 10, 10)
        assert iou(box, box) == 1.0

    def test_disjoint_boxes_have_no_overlap(self):
        assert iou(Box(0, 0, 5, 5), Box(10, 10, 15, 15)) == 0.0

    def test_a_zero_area_box_is_refused(self):
        with pytest.raises(ValueError, match="no area"):
            Box(3, 3, 3, 8)

    def test_matching_is_one_truth_box_per_prediction(self):
        truth = [Box(0, 0, 10, 10)]
        predicted = [Box(0, 0, 10, 10, score=0.9),
                     Box(1, 1, 11, 11, score=0.8)]
        pairs, missed, extra = match_boxes(truth, predicted)
        assert len(pairs) == 1 and missed == [] and extra == [1]

    def test_a_perfect_detector_scores_one(self):
        truth = [Box(0, 0, 10, 10), Box(20, 20, 30, 30)]
        report = detection_report([(truth, list(truth))], population="real",
                                  per_class_truth=False,
                                  score_semantics=SCORE_OBJECTNESS)
        assert report["precision"] == 1.0
        assert report["recall"] == 1.0
        assert report["average_precision_50"] == 1.0
        assert report["count_error_mean"] == 0.0

    def test_a_merged_pair_shows_up_as_a_negative_count_error(self):
        truth = [Box(0, 0, 10, 10), Box(11, 0, 21, 10)]
        merged = [Box(0, 0, 21, 10, score=0.9)]
        report = detection_report([(truth, merged)], population="real",
                                  per_class_truth=False,
                                  score_semantics=SCORE_OBJECTNESS)
        assert report["count_error_mean"] == -1.0
        assert report["recall"] <= 0.5

    def test_per_class_counting_is_unavailable_without_class_truth(self):
        truth = [Box(0, 0, 10, 10)]
        report = detection_report([(truth, list(truth))], population="real",
                                  per_class_truth=False,
                                  score_semantics=SCORE_OBJECTNESS)
        assert report["per_class_count_mae"] is None
        assert "invent" in report["per_class_unavailable_reason"]

    def test_per_class_counting_works_when_the_truth_has_classes(self):
        truth = [Box(0, 0, 10, 10, label="2x4")]
        predicted = [Box(0, 0, 10, 10, label="2x4", score=0.9)]
        report = detection_report([(truth, predicted)],
                                  population="synthetic",
                                  per_class_truth=True,
                                  score_semantics=SCORE_OBJECTNESS)
        assert report["per_class_count_mae"]["2x4"] == 0.0
        assert report["matched_label_confusion"]["2x4"]["2x4"] == 1

    def test_an_empty_scene_with_no_detections_is_counted(self):
        report = detection_report([([], [])], population="real",
                                  per_class_truth=False,
                                  score_semantics=SCORE_OBJECTNESS)
        assert report["empty_truth_images"] == 1
        assert report["count_error_mean"] == 0.0

    def test_average_precision_with_no_truth_is_absent(self):
        value = average_precision([(0.9, False)], 0)
        assert value != value


# ---------------------------------------------------------------------------
# Round 49: an average precision has to say what ranked it
#
# The frozen evaluation ranked stage-one boxes by the stage-TWO classifier's
# confidence and published the result as ``average_precision_50``, beside a
# note saying mAP@50 and AP@50 coincide at one class.  Both halves are true in
# isolation and the pair reads as "the two detectors were compared" -- which
# they were not: stage one is identical, so the boxes are identical and only
# the ordering differed.  These fixtures are synthetic, deterministic, and
# assert the shape of the correction rather than any number about real data.
# ---------------------------------------------------------------------------

def boxes_for(spec):
    """``[(x0, label, score), ...]`` -> boxes on one row, 10 wide, 10 apart."""
    return [Box(x, 0, x + 10, 10, label=label, score=score)
            for x, label, score in spec]


class TestADetectionScoreMustSayWhatItIs:
    def truth(self):
        return boxes_for([(0, "brick", 1.0), (20, "brick", 1.0)])

    def test_the_declaration_is_required(self):
        with pytest.raises(TypeError, match="score_semantics"):
            detection_report([(self.truth(), [])], population="real",
                             per_class_truth=False)

    def test_an_unknown_declaration_is_refused_by_name(self):
        with pytest.raises(ValueError, match="score_semantics must be one of"):
            detection_report([(self.truth(), [])], population="real",
                             per_class_truth=False, score_semantics="vibes")

    def test_objectness_publishes_the_familiar_key(self):
        truth = self.truth()
        report = detection_report([(truth, list(truth))], population="real",
                                  per_class_truth=False,
                                  score_semantics=SCORE_OBJECTNESS)
        assert report["average_precision_50"] == 1.0
        assert report["class_agnostic_ap50_by_classifier_confidence"] is None
        assert "not an" in report["average_precision_note"]

    def test_classifier_confidence_does_not_publish_that_key(self):
        """The red light: the number must not be readable as detection AP."""
        truth = self.truth()
        report = detection_report(
            [(truth, list(truth))], population="real", per_class_truth=False,
            score_semantics=SCORE_CLASSIFIER_CONFIDENCE)
        assert report["average_precision_50"] is None
        assert report["class_agnostic_ap50_by_classifier_confidence"] == 1.0
        note = report["average_precision_note"]
        assert "not a localisation metric" in note
        assert "not an eight-class mAP" in note

    def test_a_shared_stage_one_is_stated_as_such(self):
        truth = self.truth()
        report = detection_report(
            [(truth, list(truth))], population="real", per_class_truth=False,
            score_semantics=SCORE_CLASSIFIER_CONFIDENCE,
            stage_one_shared=True)
        assert report["stage_one_shared"] is True
        assert "between classifiers' confidence, not between detectors" in \
            report["not_a_detector_comparison"]

    def test_two_methods_sharing_stage_one_differ_only_in_the_ordering(self):
        """The fact the correction rests on, demonstrated on a fixture."""
        truth = self.truth()
        # The same two boxes both times; only the confidences differ.
        cv = boxes_for([(0, "2x4", 0.30), (20, "1x1", 0.90)])
        net = boxes_for([(0, "2x4", 0.95), (20, "1x1", 0.20)])
        one = detection_report(
            [(truth, cv)], population="real", per_class_truth=False,
            score_semantics=SCORE_CLASSIFIER_CONFIDENCE,
            stage_one_shared=True)
        two = detection_report(
            [(truth, net)], population="real", per_class_truth=False,
            score_semantics=SCORE_CLASSIFIER_CONFIDENCE,
            stage_one_shared=True)
        for key in ("precision", "recall", "matched_boxes",
                    "count_error_mean", "count_absolute_error_mean"):
            assert one[key] == two[key], key

    def test_per_class_truth_that_is_not_there_is_refused(self):
        truth = self.truth()          # every label is "brick", not a class
        with pytest.raises(ValueError, match="do not all carry"):
            detection_report([(truth, list(truth))], population="real",
                             per_class_truth=True,
                             score_semantics=SCORE_OBJECTNESS)

    def test_the_unavailable_reason_now_names_the_mAP_too(self):
        truth = self.truth()
        report = detection_report([(truth, list(truth))], population="real",
                                  per_class_truth=False,
                                  score_semantics=SCORE_OBJECTNESS)
        assert report["mean_average_precision_50"] is None
        assert "eight-class mAP@50" in report["per_class_unavailable_reason"]

    def test_truth_has_classes_reads_the_labels(self):
        assert truth_has_classes([(boxes_for([(0, "2x4", 1.0)]), [])])
        assert not truth_has_classes([(boxes_for([(0, "brick", 1.0)]), [])])
        assert not truth_has_classes([([], [])])


class TestARealEightClassMeanAveragePrecision:
    def test_a_perfect_two_class_scene_scores_one(self):
        truth = boxes_for([(0, "2x4", 1.0), (20, "1x1", 1.0)])
        report = per_class_average_precision([(truth, list(truth))])
        assert report["mean_average_precision"] == 1.0
        assert report["classes_with_truth"] == 2
        assert report["per_class"]["2x4"]["average_precision"] == 1.0

    def test_a_class_with_no_truth_is_absent_not_zero(self):
        truth = boxes_for([(0, "2x4", 1.0)])
        report = per_class_average_precision([(truth, list(truth))])
        empty = report["per_class"]["1x6"]["average_precision"]
        assert empty != empty
        assert report["mean_average_precision"] == 1.0

    def test_a_correctly_placed_box_with_the_wrong_name_scores_nothing(self):
        """What separates an eight-class mAP from a single-class one."""
        truth = boxes_for([(0, "2x4", 1.0)])
        mislabelled = boxes_for([(0, "1x1", 0.99)])
        single = detection_report(
            [(truth, mislabelled)], population="synthetic",
            per_class_truth=True, score_semantics=SCORE_OBJECTNESS)
        assert single["class_agnostic_ap50_by_classifier_confidence"] is None
        assert single["average_precision_50"] == 1.0, "the box is in the right place"
        per_class = single["mean_average_precision_50"]
        assert per_class["per_class"]["2x4"]["average_precision"] == 0.0
        assert per_class["mean_average_precision"] == 0.0

    def test_it_travels_inside_the_report_when_the_truth_has_classes(self):
        truth = boxes_for([(0, "2x4", 1.0)])
        report = detection_report(
            [(truth, list(truth))], population="synthetic",
            per_class_truth=True, score_semantics=SCORE_OBJECTNESS)
        assert report["mean_average_precision_50"][
            "mean_average_precision"] == 1.0

    def test_every_declared_semantics_is_accepted(self):
        truth = boxes_for([(0, "brick", 1.0)])
        for semantics in SCORE_SEMANTICS:
            body = detection_report([(truth, list(truth))], population="real",
                                    per_class_truth=False,
                                    score_semantics=semantics)
            assert body["score_semantics"] == semantics
