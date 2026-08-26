"""The colour identities: one table, used by stock, assignment and LDraw.

Colour enters this project in three places -- a person's stock, an assignment
onto a finished structure, and the LDraw file that comes out -- and all three
have to mean the same thing by "red".  So there is one table and everything
reads it.

**It is deliberately small.**  Twenty colours a person can plausibly own in
quantity, each with the LDraw colour code from the standard configuration and
the sRGB value that goes with it.  A larger table would mean guessing at codes
and hexes, and a wrong entry here would show up as a wrong LDraw file rather
than as an error, so the table stops where the confidence does.  A colour
outside it is refused by name.

**The recognition side does not extend it.**  The public render archive labels
its images with LEGO's own internal colour names, forty-three of them, and only
some of those correspond unambiguously to an entry here.  The mapping in
:data:`DATASET_COLOUR_NAMES` covers the unambiguous ones and nothing else;
evaluation over the rest reports them as outside the mapped subset rather than
forcing each to a nearest neighbour and calling the result agreement.

``LDraw code 115`` is in the table because it is already
:data:`src.rendering.ldr.DEFAULT_COLOUR` -- the colour every uncoloured export
has been written in so far.  Keeping it means an assignment that leaves a brick
alone produces the same bytes as before.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.rendering.ldr import DEFAULT_COLOUR


class ColourError(ValueError):
    """A colour id, name or value outside this table."""


@dataclass(frozen=True)
class Colour:
    """One colour: this project's id, the LDraw code, and the sRGB value."""

    colour_id: str
    ldraw: int
    rgb: tuple[int, int, int]
    label_zh: str

    @property
    def hex(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)

    def as_dict(self) -> dict:
        return {"colour_id": self.colour_id, "ldraw": self.ldraw,
                "rgb": list(self.rgb), "hex": self.hex,
                "label_zh": self.label_zh}


#: The table.  ``colour_id`` is what a stock dictionary and the UI use; the
#: LDraw code and sRGB value are the standard configuration's.
PALETTE: tuple[Colour, ...] = (
    Colour("black", 0, (0x05, 0x13, 0x1D), "黑"),
    Colour("blue", 1, (0x00, 0x55, 0xBF), "藍"),
    Colour("green", 2, (0x23, 0x78, 0x41), "綠"),
    Colour("red", 4, (0xC9, 0x1A, 0x09), "紅"),
    Colour("brown", 6, (0x58, 0x39, 0x27), "棕"),
    Colour("light_grey", 7, (0x9B, 0xA1, 0x9D), "淺灰"),
    Colour("dark_grey", 8, (0x6D, 0x6E, 0x5C), "深灰"),
    Colour("bright_green", 10, (0x4B, 0x9F, 0x4A), "亮綠"),
    Colour("pink", 13, (0xFC, 0x97, 0xAC), "粉紅"),
    Colour("yellow", 14, (0xF2, 0xCD, 0x37), "黃"),
    Colour("white", 15, (0xFF, 0xFF, 0xFF), "白"),
    Colour("tan", 19, (0xE4, 0xCD, 0x9E), "淺沙"),
    Colour("purple", 22, (0x81, 0x00, 0x7B), "紫"),
    Colour("orange", 25, (0xFE, 0x8A, 0x18), "橘"),
    Colour("lime", 27, (0xBB, 0xE9, 0x0B), "萊姆綠"),
    Colour("reddish_brown", 70, (0x58, 0x2A, 0x12), "紅棕"),
    Colour("light_bluish_grey", 71, (0xA0, 0xA5, 0xA9), "淺藍灰"),
    Colour("dark_bluish_grey", 72, (0x6C, 0x6E, 0x68), "深藍灰"),
    Colour("medium_blue", 73, (0x5A, 0x93, 0xDB), "中藍"),
    Colour("medium_lime", 115, (0xC7, 0xD2, 0x3C), "中萊姆綠"),
)

BY_ID: dict[str, Colour] = {colour.colour_id: colour for colour in PALETTE}
BY_LDRAW: dict[int, Colour] = {colour.ldraw: colour for colour in PALETTE}

#: The order a report, a table or a UI lists colours in.
COLOUR_ORDER: tuple[str, ...] = tuple(colour.colour_id for colour in PALETTE)

#: The colour an assignment falls back to when a caller asks for none.  It is
#: the LDraw default this project has always written, so "no colours" and "the
#: default colour" produce identical files.
DEFAULT_COLOUR_ID = BY_LDRAW[DEFAULT_COLOUR].colour_id

#: LEGO's own colour names, as the public render archives spell them, mapped to
#: this table where the correspondence is not in doubt.  Deliberately partial:
#: several of the archive's names ("Bright Purple", "Sand Yellow", "Medium
#: Nougat") have no unambiguous entry here, and guessing at them would turn a
#: recognition score into a score of the guess.
DATASET_COLOUR_NAMES: dict[str, str] = {
    "Black": "black",
    "White": "white",
    "Bright Red": "red",
    "Bright Blue": "blue",
    "Bright Yellow": "yellow",
    "Bright Orange": "orange",
    "Bright Green": "bright_green",
    "Dark Green": "green",
    "Brick Yellow": "tan",
    "Reddish Brown": "reddish_brown",
    "Dark Stone Grey": "dark_bluish_grey",
    "Medium Stone Grey": "light_bluish_grey",
}


def check_contract() -> None:
    """Refuse a table that could not be written to LDraw unambiguously."""
    if len(BY_ID) != len(PALETTE):
        raise ColourError("two palette entries share a colour_id")
    if len(BY_LDRAW) != len(PALETTE):
        raise ColourError("two palette entries share an LDraw code")
    if DEFAULT_COLOUR not in BY_LDRAW:
        raise ColourError(
            f"the LDraw default colour {DEFAULT_COLOUR} is not in the palette; "
            "an export with no assignment would use a colour this project "
            "cannot name")
    for colour in PALETTE:
        if not colour.colour_id.replace("_", "").isalnum():
            raise ColourError(
                f"colour id {colour.colour_id!r} is not a plain identifier")
        if any(not 0 <= channel <= 255 for channel in colour.rgb):
            raise ColourError(f"{colour.colour_id} has a channel outside 0-255")
    for name, colour_id in DATASET_COLOUR_NAMES.items():
        if colour_id not in BY_ID:
            raise ColourError(
                f"dataset colour name {name!r} maps to {colour_id!r}, which is "
                "not in the palette")


def colour(colour_id: str) -> Colour:
    """Look up one colour, refusing anything outside the table."""
    if not isinstance(colour_id, str):
        raise ColourError(
            f"a colour id must be a string, not {type(colour_id).__name__}")
    key = colour_id.strip().lower()
    if key not in BY_ID:
        raise ColourError(
            f"{colour_id!r} is not one of the {len(PALETTE)} colours this "
            f"project handles: {' '.join(COLOUR_ORDER)}")
    return BY_ID[key]


def ldraw_code(colour_id: str) -> int:
    return colour(colour_id).ldraw


def dataset_colour_id(name: str) -> str | None:
    """The palette id for a dataset colour name, or ``None`` when unmapped.

    ``None`` rather than a nearest match on purpose: an unmapped name has to be
    countable as unmapped in a report.
    """
    if not isinstance(name, str):
        return None
    return DATASET_COLOUR_NAMES.get(name.strip())
