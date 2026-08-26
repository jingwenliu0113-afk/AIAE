"""Build order: the rule, the per-step re-verification, and what it refuses.

Three properties are the reason this module exists rather than the order being
whatever the parser happened to return:

* **Every step is legal.** A brick is placed only when it rests on the ground
  or when something already placed one layer below shares its footprint.
* **An intermediate step may be in pieces.** Two towers joined later by a beam
  is a shape this project explicitly supports, so connectivity is required of
  the finished structure and never of a step.
* **A structure that cannot be built is refused.** A brick held only from
  above has no legal position in any sequence, and the answer is its index --
  not a plausible-looking order nobody could follow.
"""

from __future__ import annotations

import pytest

from src.assembly.order import (DEFAULT_MAX_PER_STEP, AssemblyError, plan,
                                step_descriptions, to_ldr,
                                unplaceable_from_below, write_step_previews)
from src.colour.assign import assign, uniform_stock
from src.data.bricks import find_collisions, is_connected, parse_bricks
from src.rendering.ldr import to_ldr_steps

TOWER = "2x4 (0,0,0)\n2x4 (0,0,1)\n2x4 (0,0,2)"

#: Two grounded columns joined at the top by a beam. The case the rules were
#: written to allow, and the one a global-connectivity rule would forbid.
#:
#: The beam is spelled ``8x1`` on purpose: it has to run along x to reach both
#: columns, and it also exercises the rotation normalisation -- it is the part
#: ``1x8`` in the inventory and the parts list.
BRIDGE = ("1x2 (0,0,0)\n1x2 (0,0,1)\n"
          "1x2 (5,0,0)\n1x2 (5,0,1)\n"
          "8x1 (0,0,2)")

#: A brick with nothing under it anywhere: held from above only.
FLOATING = "2x4 (0,0,0)\n2x4 (0,0,1)\n1x2 (10,10,0)\n1x2 (10,10,2)"


class TestTheOrderIsLegal:
    def test_a_tower_is_built_bottom_up(self):
        result = plan(parse_bricks(TOWER))
        assert result.n_steps == 3
        assert result.every_step_valid
        assert result.ready
        layers = [result.bricks[index].z for index in result.order]
        assert layers == [0, 1, 2]

    def test_every_addition_is_grounded_or_supported(self):
        result = plan(parse_bricks(BRIDGE))
        for step in result.steps:
            assert step.grounded_additions or step.supported_additions
            assert len(step.added) == len(step.grounded_additions) + len(
                step.supported_additions)

    def test_two_columns_and_a_beam_is_orderable(self):
        result = plan(parse_bricks(BRIDGE))
        assert result.ready
        assert result.final_connected

    def test_an_intermediate_step_may_be_in_several_pieces(self):
        result = plan(parse_bricks(BRIDGE))
        assert max(step.components for step in result.steps) >= 2, \
            "the two columns should exist separately before the beam"
        assert result.steps[-1].components == 1

    def test_connectivity_is_required_of_the_end_and_not_of_a_step(self):
        result = plan(parse_bricks(BRIDGE))
        assert result.every_step_valid
        assert any(step.components > 1 for step in result.steps)
        assert result.final_connected

    def test_the_order_is_a_permutation_of_every_brick(self):
        bricks = parse_bricks(BRIDGE)
        result = plan(bricks)
        assert sorted(result.order) == list(range(len(bricks)))

    def test_it_is_deterministic(self):
        bricks = parse_bricks(BRIDGE)
        assert plan(bricks).order == plan(bricks).order

    def test_several_bricks_per_step_is_allowed(self):
        result = plan(parse_bricks(BRIDGE), max_per_step=2)
        assert result.max_per_step == 2
        assert max(len(step.added) for step in result.steps) <= 2
        assert result.every_step_valid


class TestPerStepVerification:
    def test_the_accumulated_structure_is_re_checked_every_step(self):
        result = plan(parse_bricks(BRIDGE))
        for step in result.steps:
            accumulated = result.prefix(step.number)
            assert step.total_bricks == len(accumulated)
            assert step.collision_free == (not find_collisions(accumulated))
            assert step.in_bounds == all(b.in_bounds() for b in accumulated)

    def test_the_cumulative_parts_list_adds_up(self):
        result = plan(parse_bricks(BRIDGE))
        last = result.steps[-1].cumulative_parts
        assert sum(last.values()) == len(result.bricks)

    def test_stock_is_tracked_down_to_zero(self):
        bricks = parse_bricks(TOWER)
        result = plan(bricks, stock={"2x4": 3})
        assert result.steps[-1].stock_remaining == {"2x4": 0}
        assert all(step.within_stock for step in result.steps)

    def test_a_stock_that_cannot_cover_the_structure_is_refused(self):
        with pytest.raises(AssemblyError, match="needs 3, has 2"):
            plan(parse_bricks(TOWER), stock={"2x4": 2})

    def test_the_prefix_of_zero_steps_is_the_empty_structure(self):
        result = plan(parse_bricks(TOWER))
        assert result.prefix(0) == []

    def test_the_prefix_of_every_step_is_the_whole_structure(self):
        result = plan(parse_bricks(TOWER))
        assert len(result.prefix(result.n_steps)) == len(result.bricks)

    @pytest.mark.parametrize("bad", [-1, 99])
    def test_a_prefix_outside_the_range_is_refused(self, bad):
        result = plan(parse_bricks(TOWER))
        with pytest.raises(AssemblyError, match="outside"):
            result.prefix(bad)


class TestWhatIsRefused:
    def test_a_structure_with_nothing_on_the_ground_is_refused(self):
        with pytest.raises(AssemblyError, match="rests on the ground"):
            plan(parse_bricks("2x4 (0,0,3)\n2x4 (0,0,4)"))

    def test_a_brick_held_only_from_above_is_named(self):
        with pytest.raises(AssemblyError, match="held from above"):
            plan(parse_bricks(FLOATING))

    def test_the_refusal_gives_the_indices(self):
        with pytest.raises(AssemblyError) as caught:
            plan(parse_bricks(FLOATING))
        assert "indices" in str(caught.value)

    def test_unplaceable_from_below_finds_the_same_brick(self):
        assert unplaceable_from_below(parse_bricks(FLOATING)) == [3]

    def test_an_empty_structure_is_refused(self):
        with pytest.raises(AssemblyError, match="no structure"):
            plan([])

    def test_a_non_brick_is_refused(self):
        with pytest.raises(AssemblyError, match="must be a Brick"):
            plan(["2x4 (0,0,0)"])

    @pytest.mark.parametrize("bad", [0, -1, True])
    def test_a_bad_step_limit_is_refused(self, bad):
        with pytest.raises(AssemblyError, match="positive whole number"):
            plan(parse_bricks(TOWER), max_per_step=bad)

    def test_the_default_step_limit_is_one(self):
        assert DEFAULT_MAX_PER_STEP == 1


class TestNoPhysicsClaim:
    def test_the_serialised_plan_says_connectivity_is_not_support(self):
        body = plan(parse_bricks(TOWER)).as_dict()
        note = body["not_a_physics_claim"]
        assert "not support" in note
        assert "centre of mass" in note

    def test_the_rule_is_stated_in_the_output(self):
        body = plan(parse_bricks(BRIDGE)).as_dict()
        assert "shares part of its footprint" in body["rule"]
        assert "several pieces" in body["rule"]


class TestLdrawSteps:
    def test_one_step_marker_per_step(self):
        result = plan(parse_bricks(BRIDGE))
        text = to_ldr(result)
        assert text.count("0 STEP") == result.n_steps

    def test_grouping_two_bricks_per_step_halves_the_markers(self):
        result = plan(parse_bricks(BRIDGE), max_per_step=2)
        assert to_ldr(result).count("0 STEP") == result.n_steps
        assert result.n_steps < len(result.bricks)

    def test_the_lines_are_in_build_order(self):
        result = plan(parse_bricks(TOWER))
        heights = [line.split()[3] for line in to_ldr(result).splitlines()
                   if line.startswith("1 ")]
        assert heights == ["0", "-24", "-48"]

    def test_the_colours_from_the_assigner_are_used(self):
        bricks = parse_bricks(TOWER)
        result = plan(bricks)
        colours = assign(bricks, uniform_stock(bricks, "red")).colours()
        text = to_ldr(result, colours=colours)
        from src.colour.palette import ldraw_code

        assert all(line.split()[1] == str(ldraw_code("red"))
                   for line in text.splitlines() if line.startswith("1 "))

    def test_steps_that_miss_a_brick_are_refused(self):
        bricks = parse_bricks(TOWER)
        with pytest.raises(ValueError, match="exactly once"):
            to_ldr_steps([[0], [1]], bricks)

    def test_steps_that_repeat_a_brick_are_refused(self):
        bricks = parse_bricks(TOWER)
        with pytest.raises(ValueError, match="exactly once"):
            to_ldr_steps([[0], [1], [2], [2]], bricks)

    def test_an_empty_step_is_refused(self):
        bricks = parse_bricks(TOWER)
        with pytest.raises(ValueError, match="not a step"):
            to_ldr_steps([[0], [], [1, 2]], bricks)

    def test_the_default_writer_is_unchanged(self):
        """The per-brick marker behaviour the golden vector pins."""
        from src.rendering.ldr import to_ldr as plain

        assert plain(parse_bricks(TOWER)).count("0 STEP") == 3


class TestStepPreviewsAndDescriptions:
    def test_one_preview_per_step_is_written(self, tmp_path):
        result = plan(parse_bricks(TOWER))
        written = write_step_previews(result, tmp_path, title="tower")
        assert len(written) == result.n_steps
        assert all(path.is_file() and path.stat().st_size > 0
                   for path in written)

    def test_the_previews_are_numbered_in_order(self, tmp_path):
        result = plan(parse_bricks(BRIDGE))
        written = write_step_previews(result, tmp_path)
        assert [path.name for path in written] == sorted(
            path.name for path in written)

    def test_svg_previews_are_written_too(self, tmp_path):
        result = plan(parse_bricks(TOWER))
        written = write_step_previews(result, tmp_path, suffix=".svg")
        assert all(path.suffix == ".svg" for path in written)

    def test_one_description_per_step(self):
        result = plan(parse_bricks(BRIDGE))
        lines = step_descriptions(result)
        assert len(lines) == result.n_steps
        assert all(f"第 {i} 步" in line
                   for i, line in enumerate(lines, 1))

    def test_a_description_names_the_parts_and_the_position(self):
        result = plan(parse_bricks(TOWER))
        first = step_descriptions(result)[0]
        assert "2x4" in first and "(0,0,0)" in first and "放在地面" in first

    def test_a_stacked_brick_says_which_layer_it_sits_on(self):
        result = plan(parse_bricks(TOWER))
        assert "疊在第 1 層" in step_descriptions(result)[1]

    def test_the_descriptions_report_the_running_component_count(self):
        result = plan(parse_bricks(BRIDGE))
        lines = step_descriptions(result)
        assert any("2 個子結構" in line for line in lines)


class TestThePlanAgreesWithTheStructure:
    def test_the_final_verdicts_match_the_checker(self):
        bricks = parse_bricks(BRIDGE)
        result = plan(bricks)
        assert result.final_collision_free == (not find_collisions(bricks))
        assert result.final_connected == is_connected(bricks)

    def test_a_disconnected_structure_is_ordered_but_not_ready(self):
        """Two towers that are never joined: every step is legal and the
        finished thing is still two pieces, so it is not deliverable."""
        bricks = parse_bricks("2x4 (0,0,0)\n2x4 (0,0,1)\n"
                              "2x4 (10,10,0)\n2x4 (10,10,1)")
        result = plan(bricks)
        assert result.every_step_valid
        assert not result.final_connected
        assert not result.ready


# ---------------------------------------------------------------------------
# Round 49: a step image is drawn in the assignment the file was written with
#
# ``write_step_previews`` took a ``colours`` argument and never passed it on,
# so every step image was drawn from the per-shape key while the LDraw file
# carried the assignment.  Each test below reads the colours out of the SVG
# the writer produced, so accepting the argument is not enough to pass.
# ---------------------------------------------------------------------------

import re

from src.colour.palette import BY_LDRAW
from src.rendering.preview import PART_COLOURS


def fills_of(path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {value.lower()
            for value in re.findall(r"fill:\s*(#[0-9a-fA-F]{6})", text)}


class TestStepImagesCarryTheAssignedColours:
    def make(self):
        bricks = parse_bricks(TOWER)
        result = plan(bricks)
        stock = {("2x4", "red"): 2, ("2x4", "blue"): 1}
        assignment = assign(bricks, stock, preferences=["red"])
        return result, assignment

    def test_every_step_image_uses_the_assignment(self, tmp_path):
        result, assignment = self.make()
        written = write_step_previews(result, tmp_path, suffix=".svg",
                                      colours=assignment.colours())
        assert len(written) == result.n_steps
        used = set()
        for path in written:
            used |= fills_of(path)
        for code in set(assignment.colours().values()):
            assert BY_LDRAW[code].hex.lower() in used
        assert PART_COLOURS["2x4"].lower() not in used or (
            PART_COLOURS["2x4"].lower() in {
                BY_LDRAW[code].hex.lower()
                for code in assignment.colours().values()})

    def test_the_last_step_image_matches_the_ldraw_file_colour_for_colour(
            self, tmp_path):
        """The consistency claim, checked against both artefacts."""
        result, assignment = self.make()
        colours = assignment.colours()
        written = write_step_previews(result, tmp_path, suffix=".svg",
                                      colours=colours)
        text = to_ldr(result, colours=colours)
        codes = {int(line.split()[1]) for line in text.splitlines()
                 if line.startswith("1 ")}
        in_image = fills_of(written[-1])
        for code in codes:
            assert BY_LDRAW[code].hex.lower() in in_image

    def test_a_step_prefix_is_recoloured_for_its_own_numbering(self,
                                                               tmp_path):
        """Step one holds brick 0 only; it must not take brick 1's colour."""
        bricks = parse_bricks(TOWER)
        result = plan(bricks, max_per_step=1)
        colours = {0: 4, 1: 1, 2: 2}
        written = write_step_previews(result, tmp_path, suffix=".svg",
                                      colours=colours)
        first = fills_of(written[0])
        placed = result.prefix_indices(1)
        assert len(placed) == 1
        assert BY_LDRAW[colours[placed[0]]].hex.lower() in first
        for index in (0, 1, 2):
            if index != placed[0]:
                assert BY_LDRAW[colours[index]].hex.lower() not in first

    def test_prefix_indices_and_prefix_agree(self):
        result = plan(parse_bricks(BRIDGE))
        for number in range(result.n_steps + 1):
            indices = result.prefix_indices(number)
            assert [result.bricks[i] for i in indices] == result.prefix(number)

    def test_prefix_indices_refuses_a_step_outside_the_plan(self):
        result = plan(parse_bricks(TOWER))
        with pytest.raises(AssemblyError, match="outside"):
            result.prefix_indices(result.n_steps + 1)
        with pytest.raises(AssemblyError, match="whole number"):
            result.prefix_indices(True)

    def test_a_partial_assignment_is_refused_before_anything_is_drawn(
            self, tmp_path):
        result = plan(parse_bricks(TOWER))
        with pytest.raises(AssemblyError, match="have none"):
            write_step_previews(result, tmp_path, colours={0: 4})
        assert not list(tmp_path.iterdir())

    def test_without_colours_the_shape_key_is_still_used(self, tmp_path):
        result = plan(parse_bricks(TOWER))
        written = write_step_previews(result, tmp_path, suffix=".svg")
        assert PART_COLOURS["2x4"].lower() in fills_of(written[-1])
