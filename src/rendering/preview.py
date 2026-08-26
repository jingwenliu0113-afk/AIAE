"""CPU-only 3-D preview for a parsed BrickAgain structure.

This is a geometric inspection aid, not a photorealistic render and not a
physics or stability analysis.  It draws the same axis-aligned cuboids that
the parser and deterministic checker read.  No model, network, Blender or GPU
is involved.

Two things it will not do quietly:

**It will not colour a brick differently from the file.**  When a caller passes
a colour assignment, the cuboids are drawn in *that* assignment's sRGB values,
taken from the same palette the LDraw writer takes its codes from.  A partial
mapping is refused rather than filled in, because a preview that agrees with
the download for nine bricks out of ten is worse than one that refuses: the
disagreement is invisible.  With no assignment the drawing falls back to the
per-shape key in :data:`PART_COLOURS`, and callers are expected to say so
rather than claim the image matches the file's colours.

**It will not draw a character it has no glyph for.**  Matplotlib's default
family has no CJK coverage, so a Chinese caption comes out as a row of empty
boxes -- text that looks like text and carries nothing.  The heading is
therefore drawn in the first installed family that covers it, and if no
installed family does, the characters that cannot be drawn are removed and the
image says how many were dropped.  A stated omission is recoverable; tofu is
not.
"""

from __future__ import annotations

import textwrap
from functools import lru_cache
from pathlib import Path

from src.data.bricks import Brick, find_collisions


class PreviewError(ValueError):
    """The requested preview cannot be produced."""


#: The per-shape key used when no colour assignment is supplied.  It is a
#: legend for reading geometry, not a claim about the exported file's colours.
PART_COLOURS = {
    "1x1": "#F2CD37",
    "1x2": "#FF9ECD",
    "1x4": "#58AB41",
    "1x6": "#00A0D8",
    "1x8": "#0055BF",
    "2x2": "#F47B30",
    "2x4": "#C91A09",
    "2x6": "#6C6E68",
}

#: Bricks in a collision are drawn in this instead of their own colour, so an
#: invalid structure stays visibly invalid.  It overrides an assignment, and
#: :func:`brick_facecolours` reports that it did.
COLLISION_COLOUR = "#D000FF"

#: Families tried, in order, for text the default family cannot draw.  Named
#: rather than discovered so the choice is reproducible and reviewable; the
#: list spans the packaged Linux fonts, macOS and Windows, and membership is
#: never assumed -- each candidate is checked for the glyphs actually needed.
CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Noto Sans CJK TC",
    "Noto Sans CJK SC",
    "Source Han Sans TC",
    "Source Han Sans SC",
    "Sarasa Gothic TC",
    "WenQuanYi Zen Hei",
    "PingFang TC",
    "PingFang SC",
    "Heiti TC",
    "Hiragino Sans",
    "Arial Unicode MS",
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "SimHei",
)

#: The family every heading falls back to for Latin text.  It ships with
#: Matplotlib, so it is the one family that is always present.
BASE_FONT = "DejaVu Sans"

#: Characters the writer adds to a heading itself.  Probed along with the
#: caption, because a machine that cannot draw the ellipsis this module appends
#: would put a box there for a character the caller never wrote.
ADDED_CHARACTERS = " …—"

DEFAULT_HEADING = "BrickAgain 3-D geometric preview"


def validate_preview_path(path: str | Path) -> Path:
    """Validate the output format without creating a directory or file."""
    out = Path(path)
    if out.suffix.lower() not in (".png", ".svg"):
        raise PreviewError("the 3-D preview path must end in .png or .svg")
    return out


# ---------------------------------------------------------------------------
# Colour: what is actually drawn, and where it came from
# ---------------------------------------------------------------------------

def _hex_for_ldraw(code: int) -> str:
    """The sRGB value the palette gives an LDraw colour code."""
    from src.colour.palette import BY_LDRAW

    entry = BY_LDRAW.get(int(code))
    if entry is None:
        raise PreviewError(
            f"LDraw colour {code} is not in this project's palette, so the "
            "preview cannot draw the colour the file would carry. Refused "
            "rather than substituted")
    return entry.hex


def brick_facecolours(bricks, *, colours: dict[int, int] | None = None
                      ) -> tuple[list[str], dict]:
    """The exact face colour every brick is drawn in, plus its provenance.

    ``colours`` maps a brick's index to an LDraw colour code -- the same
    mapping :func:`src.rendering.ldr.to_ldr` is given -- so one assignment
    drives the file and the image.  It must cover every brick: a mapping with
    a hole would put an unassigned brick in a shape colour beside assigned
    ones, and nothing in the image would say which was which.

    Returned rather than drawn here so the choice can be asserted on directly
    instead of being inferred from an image.
    """
    bricks = list(bricks)
    collided = {i for pair in find_collisions(bricks) for i in pair}
    if colours is None:
        source = "part-key"
        base = [PART_COLOURS[brick.part] for brick in bricks]
    else:
        if not isinstance(colours, dict):
            raise PreviewError(
                "a colour assignment is a {brick index: LDraw code} mapping")
        missing = [i for i in range(len(bricks)) if i not in colours]
        if missing:
            raise PreviewError(
                f"the colour assignment covers {len(colours)} of "
                f"{len(bricks)} bricks; indices {missing[:8]} have no colour. "
                "A partial assignment is refused, because a preview that "
                "silently mixed assigned and default colours would disagree "
                "with the LDraw file without showing it")
        strange = sorted(
            (repr(key) for key in colours
             if isinstance(key, bool) or not isinstance(key, int)),
            key=str)
        if strange:
            raise PreviewError(
                f"the colour assignment is keyed by {strange[:8]}, which are "
                "not brick indices; it must be {brick index: LDraw code}")
        extra = sorted(key for key in colours
                       if not 0 <= key < len(bricks))
        if extra:
            raise PreviewError(
                f"the colour assignment names brick index(es) {extra[:8]} "
                f"that are not in this structure of {len(bricks)} bricks")
        source = "assignment"
        base = [_hex_for_ldraw(colours[i]) for i in range(len(bricks))]
    drawn = [COLLISION_COLOUR if i in collided else value
             for i, value in enumerate(base)]
    return drawn, {
        "colour_source": source,
        "collision_overrides": sorted(collided),
        "note": (
            "every face colour above is what was drawn. Bricks in a collision "
            "are drawn in the collision colour whatever their assignment, so "
            "an invalid structure stays visibly invalid"
            if collided else
            "every face colour above is what was drawn"),
    }


# ---------------------------------------------------------------------------
# Text: a heading that is drawn or is said to be missing, never tofu
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _font_file(family: str) -> str | None:
    """The file Matplotlib would use for ``family``, or ``None`` if absent."""
    try:
        from matplotlib.font_manager import FontProperties, findfont
    except ImportError:  # pragma: no cover - environment boundary
        return None
    try:
        return findfont(FontProperties(family=family),
                        fallback_to_default=False)
    except Exception:
        # findfont raises its own error type when nothing matches, and the
        # question here is only "is this family usable"; an unusable family is
        # not an error, it is the common case on a machine without it.
        return None


@lru_cache(maxsize=256)
def _covered(family: str, text: str) -> frozenset:
    """The characters of ``text`` this family has a glyph for."""
    path = _font_file(family)
    if path is None:
        return frozenset()
    try:
        from matplotlib.ft2font import FT2Font

        font = FT2Font(path)
    except Exception:  # pragma: no cover - environment boundary
        return frozenset()
    return frozenset(
        character for character in set(text)
        if character.isspace() or font.get_char_index(ord(character)))


def resolve_font(text: str) -> dict:
    """Pick the families to draw ``text`` in, and say what cannot be drawn.

    :data:`BASE_FONT` is always last, so Latin text is unaffected by whichever
    CJK family a machine happens to have.  The candidate chosen is the first
    that covers everything :data:`BASE_FONT` cannot; failing that, the one that
    covers the most, so a partly-installed machine still draws what it can.
    """
    text = str(text)
    base = _covered(BASE_FONT, text)
    outstanding = set(text) - base
    if not outstanding:
        return {"families": (BASE_FONT,), "cjk_family": None,
                "undrawable": (), "reason": "the base family covers the text"}

    best_family, best_cover = None, frozenset()
    for family in CJK_FONT_CANDIDATES:
        cover = _covered(family, text) & outstanding
        if len(cover) > len(best_cover):
            best_family, best_cover = family, cover
        if len(cover) == len(outstanding):
            break

    undrawable = tuple(sorted(outstanding - best_cover))
    if best_family is None:
        return {
            "families": (BASE_FONT,), "cjk_family": None,
            "undrawable": tuple(sorted(outstanding)),
            "reason": ("no installed family covers the characters the base "
                       "family cannot draw"),
        }
    return {
        "families": (best_family, BASE_FONT), "cjk_family": best_family,
        "undrawable": undrawable,
        "reason": (f"{best_family} covers the remaining characters"
                   if not undrawable else
                   f"{best_family} covers most of the remaining characters"),
    }


def safe_heading(title: str | None) -> dict:
    """A heading that will render, plus what had to be dropped to get one.

    The degradation is explicit on purpose.  Removing a character silently
    would make a truncated caption look deliberate; a box glyph would make an
    absent font look like content.  So the undrawable characters come out and
    the image carries a line saying how many did.
    """
    heading = (title or DEFAULT_HEADING).strip() or DEFAULT_HEADING
    font = resolve_font(heading + ADDED_CHARACTERS)
    dropped = set(font["undrawable"]) - set(ADDED_CHARACTERS)
    note = None
    if dropped:
        kept = "".join(c for c in heading if c not in dropped).strip()
        kept = " ".join(kept.split())
        count = sum(1 for c in heading if c in dropped)
        note = (f"[{count} character(s) omitted: no font on this machine can "
                f"draw them]")
        heading = kept or DEFAULT_HEADING
        # Re-resolve: what is left is drawable by construction, and the note
        # itself is ASCII, so the base family is enough.
        font = resolve_font(heading + ADDED_CHARACTERS)
    wrapped = textwrap.fill(
        heading, width=72, max_lines=2, placeholder=" …",
        break_long_words=True, break_on_hyphens=False)
    return {"heading": wrapped, "note": note,
            "families": list(font["families"]),
            "font_family": font["cjk_family"] or BASE_FONT,
            "undrawable": list(font["undrawable"])}


def _faces(b: Brick) -> list[list[tuple[float, float, float]]]:
    """Six faces of one one-layer brick cuboid."""
    x0, x1 = float(b.x), float(b.x + b.h)
    y0, y1 = float(b.y), float(b.y + b.w)
    z0, z1 = float(b.z), float(b.z + 1)
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def write_preview(path: str | Path, bricks: list[Brick], *,
                  title: str | None = None, dpi: int = 150,
                  colours: dict[int, int] | None = None) -> Path:
    """Write a PNG or SVG 3-D cuboid preview and return its path.

    ``colours`` is the same ``{brick index: LDraw code}`` mapping the LDraw
    writer takes.  Given one, the image is drawn in the assignment's colours,
    so the picture and the download agree; left out, the per-shape key is used
    and the image is a geometry legend rather than a colour proof.

    Bricks involved in a collision are coloured magenta whatever their
    assignment, so an invalid input remains visibly invalid instead of being
    smoothed into a plausible image.  The file suffix must be ``.png`` or
    ``.svg``; both are written by the CPU Agg canvas.
    """
    if not bricks:
        raise PreviewError("a 3-D preview needs at least one parsed brick")
    unknown = sorted({b.part for b in bricks if b.part not in PART_COLOURS})
    if unknown:
        raise PreviewError(
            f"the 3-D preview does not know these parts: {unknown}")
    if any(not b.in_bounds() for b in bricks):
        raise PreviewError(
            "the 3-D preview refuses out-of-bounds bricks; inspect the text "
            "report for their indices")

    out = validate_preview_path(path)
    facecolours, _provenance = brick_facecolours(bricks, colours=colours)

    # Local imports keep parsing, scoring and LDraw usable on installations
    # that deliberately omit the optional visual stack.
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise PreviewError(
            "Matplotlib is required for a 3-D preview; install the pinned "
            "requirements or omit --preview") from exc

    fig = Figure(figsize=(8, 6), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")

    for brick, colour in zip(bricks, facecolours):
        poly = Poly3DCollection(
            _faces(brick), facecolors=colour, edgecolors="#202020",
            linewidths=0.45, alpha=0.86)
        ax.add_collection3d(poly)

    x0 = min(b.x for b in bricks)
    x1 = max(b.x + b.h for b in bricks)
    y0 = min(b.y for b in bricks)
    y1 = max(b.y + b.w for b in bricks)
    z0 = min(b.z for b in bricks)
    z1 = max(b.z + 1 for b in bricks)
    pad = 0.5
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    ax.set_zlim(max(0, z0 - pad), z1 + pad)
    ax.set_box_aspect((max(x1 - x0, 1), max(y1 - y0, 1),
                       max(z1 - z0, 1)))
    ax.view_init(elev=27, azim=-52)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("layer z")
    caption = safe_heading(title)
    lines = [caption["heading"]]
    if caption["note"]:
        lines.append(caption["note"])
    lines.append("Geometric preview only — no physics or stability analysis")
    fig.suptitle("\n".join(lines), fontsize=11,
                 fontfamily=list(caption["families"]))
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, format=out.suffix.lower().lstrip("."))
    except OSError as exc:
        raise PreviewError(
            f"the 3-D preview could not be written to {out}: {exc}") from exc
    return out
