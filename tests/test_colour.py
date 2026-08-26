"""Colour: the palette contract, recognition, and the assignment invariants.

The assignment invariants are the ones that matter, and each is a refusal:

* no colour is used more than the stock holds;
* a shape whose colours do not add up is refused **by name**, with nothing
  coloured, rather than partly coloured or filled in with an invented colour;
* the same structure and stock always give the same file;
* the LDraw file and the preview take the same assignment.

The recognition side is smaller: it reads a crop's colour against the palette
and abstains when two entries are nearly equidistant, which is what sends a
crop to a person instead of guessing between two greys.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from src.colour import palette
from src.colour.assign import (AssignError, assign, brick_order, check_feasible,
                               parse_colour_stock, shape_totals, uniform_stock)
from src.colour.palette import (COLOUR_ORDER, DEFAULT_COLOUR_ID, PALETTE,
                                ColourError, check_contract, colour,
                                dataset_colour_id, ldraw_code)
from src.colour.recognise import (LOW_CONFIDENCE, MIN_PIXELS, RecogniseError,
                                  delta_e, palette_lab, read_colour,
                                  rgb_to_hsv, rgb_to_lab, usable_pixels)
from src.data.bricks import parse_bricks
from src.rendering.ldr import DEFAULT_COLOUR, to_ldr


# --------------------------------------------------------------------------
# the palette
# --------------------------------------------------------------------------

class TestThePaletteContract:
    def test_it_holds_as_shipped(self):
        check_contract()

    def test_the_ldraw_default_colour_is_in_the_palette(self):
        """Otherwise an export with no assignment uses a colour this project
        cannot name."""
        assert DEFAULT_COLOUR in {entry.ldraw for entry in PALETTE}
        assert ldraw_code(DEFAULT_COLOUR_ID) == DEFAULT_COLOUR

    def test_ids_and_ldraw_codes_are_unique(self):
        assert len({entry.colour_id for entry in PALETTE}) == len(PALETTE)
        assert len({entry.ldraw for entry in PALETTE}) == len(PALETTE)

    def test_every_entry_has_a_chinese_label(self):
        assert all(entry.label_zh for entry in PALETTE)

    def test_every_channel_is_in_range(self):
        for entry in PALETTE:
            assert all(0 <= channel <= 255 for channel in entry.rgb)

    def test_the_hex_form_round_trips(self):
        for entry in PALETTE:
            assert entry.hex == "#{:02X}{:02X}{:02X}".format(*entry.rgb)

    def test_a_colour_outside_the_palette_is_refused(self):
        with pytest.raises(ColourError, match="not one of the"):
            colour("chartreuse")

    def test_a_non_string_colour_is_refused(self):
        with pytest.raises(ColourError, match="must be a string"):
            colour(4)

    def test_lookup_is_case_and_space_insensitive(self):
        assert colour("  RED ").colour_id == "red"

    def test_a_missing_default_colour_is_caught(self, monkeypatch):
        trimmed = tuple(e for e in PALETTE if e.ldraw != DEFAULT_COLOUR)
        monkeypatch.setattr(palette, "PALETTE", trimmed)
        monkeypatch.setattr(palette, "BY_ID",
                            {e.colour_id: e for e in trimmed})
        monkeypatch.setattr(palette, "BY_LDRAW", {e.ldraw: e for e in trimmed})
        with pytest.raises(ColourError, match="LDraw default colour"):
            check_contract()

    def test_a_dataset_name_mapping_to_nothing_is_caught(self, monkeypatch):
        monkeypatch.setattr(palette, "DATASET_COLOUR_NAMES",
                            {"Invented": "not_a_colour"})
        with pytest.raises(ColourError, match="not in the palette"):
            check_contract()


class TestTheDatasetColourMappingIsDeliberatelyPartial:
    def test_the_unambiguous_names_map(self):
        assert dataset_colour_id("Bright Red") == "red"
        assert dataset_colour_id("Dark Stone Grey") == "dark_bluish_grey"
        assert dataset_colour_id("Brick Yellow") == "tan"

    def test_an_ambiguous_name_maps_to_nothing_rather_than_a_guess(self):
        """A nearest-neighbour guess here would turn a recognition score into
        a score of the guess."""
        assert dataset_colour_id("Bright Purple") is None
        assert dataset_colour_id("Medium Nougat") is None
        assert dataset_colour_id("Sand Yellow") is None

    def test_a_non_string_maps_to_nothing(self):
        assert dataset_colour_id(None) is None


# --------------------------------------------------------------------------
# colour spaces and recognition
# --------------------------------------------------------------------------

class TestColourSpaces:
    def test_white_is_lightness_one_hundred(self):
        lab = rgb_to_lab(np.array([255.0, 255.0, 255.0]))
        assert abs(float(lab[0]) - 100.0) < 0.2
        assert abs(float(lab[1])) < 0.2 and abs(float(lab[2])) < 0.2

    def test_black_is_lightness_zero(self):
        lab = rgb_to_lab(np.array([0.0, 0.0, 0.0]))
        assert abs(float(lab[0])) < 0.2

    def test_hue_of_pure_red_is_zero(self):
        hsv = rgb_to_hsv(np.array([255.0, 0.0, 0.0]))
        assert abs(float(hsv[0])) < 0.01
        assert abs(float(hsv[1]) - 1.0) < 0.01

    def test_hue_of_pure_green_is_one_hundred_and_twenty(self):
        assert abs(float(rgb_to_hsv(np.array([0.0, 255.0, 0.0]))[0]) - 120) < 0.01

    def test_grey_has_no_saturation(self):
        assert float(rgb_to_hsv(np.array([128.0, 128.0, 128.0]))[1]) == 0.0

    def test_the_palette_in_lab_has_one_row_per_colour(self):
        assert palette_lab().shape == (len(PALETTE), 3)

    def test_delta_e_to_itself_is_zero(self):
        for entry in PALETTE:
            assert delta_e(entry.colour_id, entry.rgb) < 1e-6


def crop(colour_rgb, ground=(245, 245, 243), size=80, inset=18):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = ground
    image[inset:size - inset, inset:size - inset] = colour_rgb
    return image


class TestReadingAColour:
    @pytest.mark.parametrize("colour_id", [
        "red", "blue", "yellow", "green", "orange", "purple", "black",
        "medium_blue", "reddish_brown", "tan",
    ])
    def test_a_palette_colour_is_read_back(self, colour_id):
        entry = colour(colour_id)
        reading = read_colour(crop(entry.rgb))
        assert reading.colour_id == colour_id
        assert not reading.low_confidence
        assert reading.label == colour_id

    def test_the_reading_reports_the_pixels_it_used(self):
        reading = read_colour(crop((200, 30, 30)))
        assert reading.pixels_used > 0
        assert reading.pixels_used <= reading.pixels_total

    def test_it_carries_three_candidates_with_distances(self):
        body = read_colour(crop((200, 30, 30))).as_dict()
        assert len(body["candidates"]) == 3
        assert body["candidates"][0]["delta_e"] <= \
            body["candidates"][1]["delta_e"]

    def test_a_colour_between_two_palette_entries_abstains(self):
        """Halfway between the two bluish greys: a person decides, not this."""
        first = np.array(colour("light_bluish_grey").rgb, dtype=float)
        second = np.array(colour("dark_bluish_grey").rgb, dtype=float)
        midpoint = tuple(int(v) for v in (first + second) / 2)
        reading = read_colour(crop(midpoint))
        assert reading.low_confidence
        assert reading.label is None

    def test_a_specular_highlight_does_not_turn_a_brick_white(self):
        image = crop((30, 60, 190), size=120, inset=24)
        image[50:70, 50:70] = (255, 255, 255)
        assert read_colour(image).colour_id in ("blue", "medium_blue")

    def test_a_crop_with_nothing_in_it_is_refused(self):
        flat = np.full((60, 60, 3), 240, dtype=np.uint8)
        with pytest.raises(RecogniseError, match="not enough surface"):
            read_colour(flat)

    def test_a_non_image_is_refused(self):
        with pytest.raises(RecogniseError, match="h, w, 3"):
            read_colour(np.zeros((8, 8), dtype=np.uint8))

    def test_a_mask_of_the_wrong_size_is_refused(self):
        with pytest.raises(RecogniseError, match="does not match"):
            usable_pixels(crop((10, 10, 10)), np.ones((4, 4), dtype=bool))

    def test_the_threshold_is_the_one_constant(self):
        assert 0 < LOW_CONFIDENCE < 1
        assert MIN_PIXELS > 0


# --------------------------------------------------------------------------
# assignment
# --------------------------------------------------------------------------

STRUCTURE = "2x4 (0,0,0)\n2x4 (0,0,1)\n1x2 (4,0,0)\n2x6 (0,0,2)"


class TestParsingAColourStock:
    def test_it_reads_part_colour_count(self):
        assert parse_colour_stock("2x4:red:6,1x2:blue:2") == {
            ("2x4", "red"): 6, ("1x2", "blue"): 2}

    def test_rotation_spellings_are_one_shape(self):
        assert parse_colour_stock("4x2:red:3") == {("2x4", "red"): 3}

    def test_both_rotation_spellings_of_one_entry_are_refused(self):
        with pytest.raises(AssignError, match="same shape"):
            parse_colour_stock("2x4:red:3,4x2:red:2")

    @pytest.mark.parametrize("spec", [
        "", "   ", "2x4:red", "2x4:red:0", "2x4:red:-1", "2x4:red:x",
        "2x9:red:1", "2x4:chartreuse:1", "2x4:red:1:2",
    ])
    def test_a_malformed_entry_is_refused(self, spec):
        """One exception type for every kind of bad stock string.

        The part lookup and the colour lookup each raise their own module's
        error; a caller that had to catch three types would eventually catch
        two and let the third reach the browser as a defect page.
        """
        with pytest.raises(AssignError):
            parse_colour_stock(spec)

    def test_a_non_ascii_digit_is_refused(self):
        with pytest.raises(AssignError, match="not a whole number"):
            parse_colour_stock("2x4:red:٣")

    def test_shape_totals_sum_across_colours(self):
        stock = parse_colour_stock("2x4:red:3,2x4:blue:2,1x2:black:1")
        assert shape_totals(stock) == Counter({"2x4": 5, "1x2": 1})


class TestAssignmentNeverExceedsStock:
    def test_a_uniform_stock_colours_everything(self):
        bricks = parse_bricks(STRUCTURE)
        result = assign(bricks, uniform_stock(bricks, "red"))
        assert len(result.bricks) == len(bricks)
        assert set(result.colour_ids().values()) == {"red"}
        result.check_within_stock()

    def test_no_colour_is_used_more_than_it_was_stocked(self):
        bricks = parse_bricks(STRUCTURE)
        stock = parse_colour_stock("2x4:red:1,2x4:blue:1,1x2:black:1,"
                                   "2x6:yellow:1")
        result = assign(bricks, stock)
        for key, used in result.used.items():
            assert used <= stock[key]
        assert all(value >= 0 for value in result.remaining.values())

    def test_the_remaining_count_is_exact(self):
        bricks = parse_bricks("2x4 (0,0,0)")
        result = assign(bricks, parse_colour_stock("2x4:red:5"))
        assert result.remaining[("2x4", "red")] == 4

    def test_a_shape_short_of_stock_is_refused_by_name(self):
        bricks = parse_bricks(STRUCTURE)
        stock = parse_colour_stock("2x4:red:1,1x2:black:1,2x6:yellow:1")
        with pytest.raises(AssignError, match="2x4: needs 2, has 1"):
            assign(bricks, stock)

    def test_a_refusal_colours_nothing(self):
        bricks = parse_bricks(STRUCTURE)
        stock = parse_colour_stock("2x4:red:1")
        with pytest.raises(AssignError):
            assign(bricks, stock)

    def test_the_refusal_says_it_invents_nothing(self):
        bricks = parse_bricks(STRUCTURE)
        with pytest.raises(AssignError, match="No colour is invented"):
            assign(bricks, parse_colour_stock("2x4:red:1"))

    def test_check_feasible_names_every_short_shape(self):
        bricks = parse_bricks(STRUCTURE)
        with pytest.raises(AssignError) as caught:
            check_feasible(bricks, parse_colour_stock("2x4:red:1"))
        message = str(caught.value)
        assert "2x4" in message and "1x2" in message and "2x6" in message

    def test_an_empty_structure_is_refused(self):
        with pytest.raises(AssignError, match="no structure"):
            assign([], parse_colour_stock("2x4:red:1"))

    def test_a_non_brick_is_refused(self):
        with pytest.raises(AssignError, match="must be a Brick"):
            assign(["2x4"], parse_colour_stock("2x4:red:1"))

    def test_a_stock_keyed_wrongly_is_refused(self):
        bricks = parse_bricks("2x4 (0,0,0)")
        with pytest.raises(AssignError, match="keyed by"):
            assign(bricks, {"2x4": 1})


class TestAssignmentIsDeterministic:
    def test_two_runs_give_the_same_colours(self):
        bricks = parse_bricks(STRUCTURE)
        stock = parse_colour_stock("2x4:red:2,1x2:blue:1,2x6:yellow:1")
        first = assign(bricks, stock)
        again = assign(bricks, stock)
        assert first.colours() == again.colours()
        assert first.as_dict() == again.as_dict()

    def test_the_order_is_bottom_layer_first(self):
        bricks = parse_bricks(STRUCTURE)
        order = brick_order(bricks)
        layers = [bricks[index].z for index in order]
        assert layers == sorted(layers)

    def test_the_ldraw_file_is_identical_across_runs(self):
        bricks = parse_bricks(STRUCTURE)
        stock = parse_colour_stock("2x4:red:2,1x2:blue:1,2x6:yellow:1")
        first = to_ldr(bricks, colours=assign(bricks, stock).colours())
        again = to_ldr(bricks, colours=assign(bricks, stock).colours())
        assert first == again


class TestPreferences:
    def test_a_preferred_colour_is_used_first(self):
        bricks = parse_bricks("2x4 (0,0,0)\n2x4 (0,0,1)")
        stock = parse_colour_stock("2x4:red:2,2x4:blue:2")
        result = assign(bricks, stock, preferences=["blue"])
        assert set(result.colour_ids().values()) == {"blue"}
        assert result.preferred_count == 2

    def test_running_out_of_a_preference_falls_through_and_is_reported(self):
        bricks = parse_bricks("2x4 (0,0,0)\n2x4 (0,0,1)\n2x4 (0,0,2)")
        stock = parse_colour_stock("2x4:blue:1,2x4:red:2")
        result = assign(bricks, stock, preferences=["blue"])
        assert result.preferred_count == 1
        assert result.as_dict()["non_preferred_bricks"] == 2

    def test_preferences_are_honoured_in_the_order_given(self):
        bricks = parse_bricks("2x4 (0,0,0)\n2x4 (0,0,1)")
        stock = parse_colour_stock("2x4:red:1,2x4:blue:1,2x4:yellow:1")
        result = assign(bricks, stock, preferences=["yellow", "blue"])
        assert result.bricks[0].colour_id == "yellow"

    def test_a_preference_outside_the_palette_is_refused(self):
        bricks = parse_bricks("2x4 (0,0,0)")
        with pytest.raises(ColourError, match="not one of the"):
            assign(bricks, parse_colour_stock("2x4:red:1"),
                   preferences=["chartreuse"])

    def test_a_repeated_preference_is_collapsed(self):
        bricks = parse_bricks("2x4 (0,0,0)")
        result = assign(bricks, parse_colour_stock("2x4:red:1"),
                        preferences=["red", "red"])
        assert result.preferences == ("red",)


class TestTheAssignmentFeedsBothWriters:
    def test_the_ldraw_codes_come_from_the_palette(self):
        bricks = parse_bricks("2x4 (0,0,0)")
        result = assign(bricks, parse_colour_stock("2x4:red:1"))
        assert result.colours() == {0: ldraw_code("red")}
        assert to_ldr(bricks, colours=result.colours()).startswith(
            f"1 {ldraw_code('red')} ")

    def test_the_serialised_form_states_its_determinism(self):
        bricks = parse_bricks("2x4 (0,0,0)")
        body = assign(bricks, parse_colour_stock("2x4:red:1")).as_dict()
        assert "always give the same result" in body["determinism"]
        assert body["within_stock"] is True

    def test_the_remaining_block_omits_zeroes_but_keeps_the_rest(self):
        bricks = parse_bricks("2x4 (0,0,0)")
        body = assign(bricks, parse_colour_stock("2x4:red:1,1x2:blue:3")
                      ).as_dict()
        assert "2x4:red" not in body["remaining"]
        assert body["remaining"]["1x2:blue"] == 3
