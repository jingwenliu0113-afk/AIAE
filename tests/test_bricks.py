import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bricks import (
    PART_VOCAB,
    Brick,
    ParseError,
    canonical_part,
    connected_components,
    find_collisions,
    is_connected,
    is_valid_part,
    parse_bricks,
    parse_line,
    required_inventory,
    studs_connected,
)


class TestParsing:
    def test_basic(self):
        b = parse_line("2x6 (13,12,0)")
        assert (b.h, b.w, b.x, b.y, b.z) == (2, 6, 13, 12, 0)

    def test_roundtrip(self):
        for s in ["1x1 (0,0,0)", "8x1 (7,11,3)", "2x4 (10,10,19)"]:
            assert str(parse_line(s)) == s

    def test_rejects_garbage(self):
        for s in ["1x (0,0,0)", "2x6 (1,2)", "brick", "2x6 1,2,3"]:
            with pytest.raises(ParseError):
                parse_line(s)

    def test_skips_blank_lines(self):
        assert len(parse_bricks("1x1 (0,0,0)\n\n1x2 (1,0,0)\n")) == 2

    def test_non_strict_skips_bad(self):
        assert len(parse_bricks("1x1 (0,0,0)\nJUNK\n", strict=False)) == 1


class TestRotationNormalisation:
    """Half the corpus uses the long-side-first spelling; both are one part."""

    def test_canonical_is_orientation_free(self):
        assert canonical_part(1, 4) == canonical_part(4, 1) == "1x4"
        assert canonical_part(2, 6) == canonical_part(6, 2) == "2x6"

    def test_symmetric_parts(self):
        assert canonical_part(1, 1) == "1x1"
        assert canonical_part(2, 2) == "2x2"

    def test_brick_part_property(self):
        assert parse_line("4x1 (0,0,0)").part == "1x4"
        assert parse_line("1x4 (0,0,0)").part == "1x4"

    def test_rotated_flag(self):
        assert parse_line("4x1 (0,0,0)").rotated
        assert not parse_line("1x4 (0,0,0)").rotated

    def test_inventory_merges_orientations(self):
        inv = required_inventory(parse_bricks("1x4 (0,0,0)\n4x1 (0,5,0)"))
        assert inv == {"1x4": 2}, "rotations must share one inventory slot"


class TestGeometry:
    def test_axis_assignment(self):
        """h runs along x, w along y."""
        b = parse_line("4x1 (1,17,0)")
        assert b.footprint == {(1, 17), (2, 17), (3, 17), (4, 17)}

    def test_cell_count(self):
        assert len(parse_line("2x6 (0,0,0)").cells) == 12

    def test_in_bounds(self):
        assert parse_line("4x1 (16,0,0)").in_bounds()
        assert not parse_line("4x1 (17,0,0)").in_bounds()
        assert not parse_line("1x4 (0,17,0)").in_bounds()
        assert not parse_line("1x1 (0,0,20)").in_bounds()

    def test_collision_detected(self):
        assert find_collisions(parse_bricks("2x2 (0,0,0)\n2x2 (1,1,0)")) == [(0, 1)]

    def test_no_collision_when_adjacent(self):
        assert find_collisions(parse_bricks("2x2 (0,0,0)\n2x2 (2,0,0)")) == []

    def test_no_collision_across_layers(self):
        assert find_collisions(parse_bricks("2x2 (0,0,0)\n2x2 (0,0,1)")) == []


class TestConnectivity:
    def test_side_by_side_is_not_connected(self):
        """Touching in the same layer is not a LEGO joint."""
        a, b = parse_bricks("2x2 (0,0,0)\n2x2 (2,0,0)")
        assert not studs_connected(a, b)
        assert len(connected_components([a, b])) == 2

    def test_stacked_overlap_is_connected(self):
        a, b = parse_bricks("2x2 (0,0,0)\n2x2 (1,0,1)")
        assert studs_connected(a, b)
        assert len(connected_components([a, b])) == 1

    def test_stacked_without_overlap_is_not_connected(self):
        a, b = parse_bricks("2x2 (0,0,0)\n2x2 (5,5,1)")
        assert not studs_connected(a, b)

    def test_two_layers_apart_is_not_connected(self):
        a, b = parse_bricks("2x2 (0,0,0)\n2x2 (0,0,2)")
        assert not studs_connected(a, b)

    def test_two_columns_joined_by_a_beam(self):
        """The build order that incremental-connectivity would wrongly reject."""
        bricks = parse_bricks(
            "2x2 (0,0,0)\n"      # column A, x in [0,2)
            "2x2 (6,0,0)\n"      # column B, x in [6,8), disconnected at this point
            "8x1 (0,0,1)"        # beam spanning x in [0,8) above both
        )
        assert len(connected_components(bricks[:2])) == 2
        assert len(connected_components(bricks)) == 1


class TestPartVocabulary:
    def test_known_parts(self):
        assert is_valid_part("1x4") and is_valid_part("2x6")

    def test_rejects_out_of_vocab(self):
        # 2x8 looks plausible but is not one of the eight parts.
        assert not is_valid_part("2x8")
        assert not is_valid_part("3x3")

    def test_rejects_noncanonical_spelling(self):
        assert not is_valid_part("4x1"), "must be normalised before checking"

    def test_every_vocab_entry_is_canonical(self):
        for p in PART_VOCAB:
            h, w = (int(v) for v in p.split("x"))
            assert canonical_part(h, w) == p


class TestGroundIsNotConnection:
    """Regression: the baseplate must not merge stud-separate components.

    It is not a part, holds no inventory and is never written out, so a model
    that only hangs together through it would come apart when lifted.
    """

    SIDE_BY_SIDE = "2x2 (0,0,0)\n2x2 (4,0,0)"

    def test_two_ground_bricks_are_two_components(self):
        bricks = parse_bricks(self.SIDE_BY_SIDE)
        assert len(connected_components(bricks)) == 2
        assert not is_connected(bricks)

    def test_default_is_stud_only(self):
        import inspect

        assert inspect.signature(is_connected).parameters["ground"].default is False
        assert (
            inspect.signature(connected_components).parameters["ground"].default
            is False
        )

    def test_ground_flag_merges_them_but_is_opt_in(self):
        bricks = parse_bricks(self.SIDE_BY_SIDE)
        assert len(connected_components(bricks, ground=True)) == 1
        assert is_connected(bricks, ground=True)

    def test_bridging_brick_makes_one_component(self):
        """Only a stud connection may join them."""
        bricks = parse_bricks(self.SIDE_BY_SIDE + "\n6x2 (0,0,1)")
        assert len(connected_components(bricks)) == 1
        assert is_connected(bricks)

    def test_bridge_must_actually_overlap_both(self):
        """A brick above only one of them leaves the other separate."""
        bricks = parse_bricks(self.SIDE_BY_SIDE + "\n2x2 (0,0,1)")
        assert len(connected_components(bricks)) == 2

    def test_adjacent_but_not_overlapping_stays_split(self):
        bricks = parse_bricks("2x2 (0,0,0)\n2x2 (2,0,0)")
        assert len(connected_components(bricks)) == 2
