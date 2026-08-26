"""The eight-class contract: derived, checked, and refusing everything else.

These tests exist because the failure this module is built to prevent is
silent. If the vision class list drifts from the part vocabulary or from the
LDraw part files, nothing raises: photographs are classified into labels that
no longer mean the same inventory item, and every count downstream is wrong in
a way that looks like a model problem.
"""

from __future__ import annotations

import pytest

from src.data.bricks import PART_VOCAB
from src.rendering.ldr import PART_TO_LDRAW
from src.vision import classes
from src.vision.classes import (CLASS_INDEX, CLASS_ORDER, DESIGN_TO_PART,
                                N_CLASSES, PART_TO_DESIGN, UNKNOWN, ClassError,
                                check_contract, design_numbers,
                                design_of_part, index_label, label_index,
                                normalise_part, part_of_design)


class TestTheListIsDerivedAndNotRestated:
    def test_the_contract_holds_as_shipped(self):
        check_contract()

    def test_class_order_is_the_part_vocabulary(self):
        assert CLASS_ORDER == tuple(PART_VOCAB)

    def test_the_eight_design_numbers_are_the_contract_ones(self):
        assert set(design_numbers()) == {
            "3005", "3004", "3010", "3009", "3008", "3003", "3001", "2456"}

    def test_each_design_number_is_the_ldraw_file_without_its_suffix(self):
        for part, design in PART_TO_DESIGN.items():
            assert PART_TO_LDRAW[part] == f"{design}.DAT"

    def test_the_mapping_is_one_to_one(self):
        assert len(DESIGN_TO_PART) == len(PART_TO_DESIGN) == N_CLASSES

    def test_the_specific_pairs_the_contract_names(self):
        for part, design in (("1x1", "3005"), ("1x2", "3004"), ("1x4", "3010"),
                             ("1x6", "3009"), ("1x8", "3008"), ("2x2", "3003"),
                             ("2x4", "3001"), ("2x6", "2456")):
            assert design_of_part(part) == design
            assert part_of_design(design) == part


class TestADriftedTableIsRefused:
    """Each of these would silently mislabel every prediction."""

    def test_a_reordered_class_list_is_caught(self, monkeypatch):
        monkeypatch.setattr(classes, "CLASS_ORDER",
                            tuple(reversed(CLASS_ORDER)))
        with pytest.raises(ClassError, match="no longer the part vocabulary"):
            check_contract()

    def test_a_part_missing_from_the_ldraw_table_is_caught(self, monkeypatch):
        trimmed = {k: v for k, v in PART_TO_LDRAW.items() if k != "2x6"}
        monkeypatch.setattr(classes, "PART_TO_LDRAW", trimmed)
        with pytest.raises(ClassError, match="different parts"):
            check_contract()

    def test_two_parts_sharing_a_design_number_are_caught(self, monkeypatch):
        clashing = dict(PART_TO_DESIGN)
        clashing["2x6"] = clashing["2x4"]
        monkeypatch.setattr(classes, "PART_TO_DESIGN", clashing)
        monkeypatch.setattr(
            classes, "DESIGN_TO_PART",
            {design: part for part, design in clashing.items()})
        with pytest.raises(ClassError, match="share a design number"):
            check_contract()

    def test_a_part_file_that_is_not_its_number_is_caught(self, monkeypatch):
        broken = dict(PART_TO_LDRAW)
        broken["1x1"] = "brick1x1.dat"
        monkeypatch.setattr(classes, "PART_TO_LDRAW", broken)
        monkeypatch.setattr(
            classes, "PART_TO_DESIGN",
            {part: broken[part].rsplit(".", 1)[0] for part in PART_VOCAB})
        with pytest.raises(ClassError, match="is not a number"):
            check_contract()

    def test_a_class_index_that_does_not_index_the_order_is_caught(
            self, monkeypatch):
        monkeypatch.setattr(classes, "CLASS_INDEX",
                            {name: 0 for name in CLASS_ORDER})
        with pytest.raises(ClassError, match="does not index"):
            check_contract()


class TestRotationIsOneClass:
    """The project's non-negotiable rule, restated where a label enters."""

    @pytest.mark.parametrize("spelling,canonical", [
        ("4x1", "1x4"), ("1x4", "1x4"), ("8x1", "1x8"), ("6x2", "2x6"),
        ("2x2", "2x2"), (" 4x2 ", "2x4"),
    ])
    def test_both_spellings_are_one_class(self, spelling, canonical):
        assert normalise_part(spelling) == canonical
        assert label_index(spelling) == CLASS_INDEX[canonical]

    def test_a_plausible_size_outside_the_eight_is_refused(self):
        # 2x8 looks like it belongs and does not: the eight parts are not the
        # cross product of the extents that appear.
        with pytest.raises(ClassError, match="not one of the"):
            normalise_part("2x8")

    def test_a_non_part_string_is_refused(self):
        with pytest.raises(ClassError, match="form HxW"):
            normalise_part("brick")

    def test_a_non_string_is_refused(self):
        with pytest.raises(ClassError, match="must be a string"):
            normalise_part(24)


class TestIndexAndLabelRoundTrip:
    def test_every_index_maps_back(self):
        for index in range(N_CLASSES):
            assert label_index(index_label(index)) == index

    @pytest.mark.parametrize("bad", [-1, N_CLASSES, 999])
    def test_an_index_outside_the_range_is_refused(self, bad):
        with pytest.raises(ClassError, match="outside"):
            index_label(bad)

    def test_a_boolean_is_not_an_index(self):
        with pytest.raises(ClassError, match="whole number"):
            index_label(True)

    def test_a_design_number_outside_the_eight_is_refused(self):
        with pytest.raises(ClassError, match="not one of the eight"):
            part_of_design("3062")


class TestUnknownIsAValueNotAnException:
    def test_unknown_is_not_one_of_the_classes(self):
        assert UNKNOWN not in CLASS_ORDER
        assert UNKNOWN not in CLASS_INDEX

    def test_unknown_cannot_be_normalised_into_a_class(self):
        with pytest.raises(ClassError):
            normalise_part(UNKNOWN)
