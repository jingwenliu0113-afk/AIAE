import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bricks import PART_VOCAB, parse_bricks
from src.rendering.ldr import PART_TO_LDRAW, to_ldr, write_ldr


class TestGoldenVector:
    """Pinned against BrickGPT's own tests/test_brick_structure.py.

    Output must stay byte-identical to the reference implementation, since the
    whole point of not installing their package is that we match it exactly.
    """

    EXPECTED = (
        "1 115 20.0 0 60.0 0 0 1 0 1 0 -1 0 0 2456.DAT\n0 STEP\n"
        "1 115 60.0 0 60.0 0 0 1 0 1 0 -1 0 0 2456.DAT\n0 STEP\n"
    )

    def test_matches_reference_output(self):
        assert to_ldr(parse_bricks("2x6 (0,0,0)\n2x6 (2,0,0)")) == self.EXPECTED


class TestConversion:
    def test_origin_is_footprint_centre(self):
        line = to_ldr(parse_bricks("2x4 (0,0,0)")).splitlines()[0].split()
        assert line[2] == "20.0"      # (0 + 2/2) * 20
        assert line[4] == "40.0"      # (0 + 4/2) * 20

    def test_y_goes_negative_upward(self):
        """LDraw Y points down, so higher layers are more negative."""
        ys = [
            to_ldr(parse_bricks(f"2x2 (0,0,{z})")).splitlines()[0].split()[3]
            for z in (0, 1, 2)
        ]
        assert ys == ["0", "-24", "-48"]

    def test_orientation_matrix_switches(self):
        flat = to_ldr(parse_bricks("2x6 (0,0,0)")).splitlines()[0]
        turned = to_ldr(parse_bricks("6x2 (0,0,0)")).splitlines()[0]
        assert "0 0 1 0 1 0 -1 0 0" in flat
        assert "-1 0 0 0 1 0 0 0 -1" in turned

    def test_rotated_spelling_uses_same_part_file(self):
        a = to_ldr(parse_bricks("1x4 (0,0,0)")).splitlines()[0].split()[-1]
        b = to_ldr(parse_bricks("4x1 (0,0,0)")).splitlines()[0].split()[-1]
        assert a == b == "3010.DAT"

    def test_step_after_every_brick(self):
        text = to_ldr(parse_bricks("1x1 (0,0,0)\n1x1 (2,0,0)\n1x1 (4,0,0)"))
        assert text.count("0 STEP") == 3

    def test_custom_colour(self):
        text = to_ldr(parse_bricks("1x1 (0,0,0)\n1x1 (2,0,0)"), colours={1: 4})
        assert text.splitlines()[0].split()[1] == "115"
        assert text.splitlines()[2].split()[1] == "4"


class TestPartMapping:
    def test_every_vocab_part_is_mapped(self):
        assert set(PART_TO_LDRAW) == set(PART_VOCAB)

    def test_part_files_are_distinct(self):
        assert len(set(PART_TO_LDRAW.values())) == len(PART_TO_LDRAW)


def test_write_ldr(tmp_path):
    p = write_ldr(tmp_path / "out" / "m.ldr", parse_bricks("2x2 (0,0,0)"))
    assert p.exists()
    assert p.read_text().startswith("1 115 ")
