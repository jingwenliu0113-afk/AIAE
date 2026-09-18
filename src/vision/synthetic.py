"""V1: deterministic renders with a ground truth, and nothing pretending.

Every image this module produces is drawn by arithmetic from a scene the
caller states. There is no renderer with its own defaults, no lighting model
with a random seed inside it, and no image file read from disk: given the
same scene and the same code, the same bytes come out, on any machine, and a
test asserts that by rendering twice and comparing SHA-256.

**These are software renders and may never be described as photographs.**
They are flat top-down drawings of studded rectangles -- close enough in
shape, aspect and stud pitch for the recogniser to have something real to
measure, and nowhere near a photograph. What they can measure is *robustness
to image conditions*: the same scene under a different background, a
different light, a shadow, a tilt, an occlusion or a worse camera, with the
ground truth held fixed by construction because the scene never changed.

What they cannot measure is real-photograph accuracy, and no number derived
from them may be reported as one.

---------------------------------------------------------------------------
The scene
---------------------------------------------------------------------------

A scene is a list of placements: a part from :data:`PART_VOCAB`, a colour
from the project palette, a position in millimetre-free "stud" units, and a
quarter-turn. The ground truth is the scene -- the parts, their counts and
their colours are what the caller asked for, not what a labeller thought it
saw -- which is the whole reason V1 is the quantitative baseline and V3 is
not.

Placements are laid out on a grid with a gap, so bricks do not touch unless
an occlusion condition deliberately makes them. Two bricks that touch merge
into one foreground component, and a merged component is a *detector*
outcome the report should be able to attribute to the occlusion rather than
to the classifier.

---------------------------------------------------------------------------
The conditions
---------------------------------------------------------------------------

Six axes, frozen in :data:`CONDITIONS`, each with a baseline value that is
the neutral one. The matrix is not the cross product: it is the baseline
plus one axis moved at a time, so a difference in the result has one cause.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from src.colour.palette import COLOUR_ORDER, colour
from src.data.bricks import PART_VOCAB

#: Pixels per stud at full quality. Chosen so a 1x1 is 44 px across, which
#: leaves the stud circle enough pixels to survive the downscale the
#: detector applies before it measures pitch.
STUD_PX = 44

#: The gap between bricks on the layout grid, in studs. One full stud, so an
#: unoccluded scene never merges two components by accident.
GAP_STUDS = 1.0

#: The margin around the whole layout, in studs.
MARGIN_STUDS = 1.5

#: Stud circle radius as a fraction of the stud pitch, and the fraction of
#: the top face's brightness the stud is drawn at.
STUD_RADIUS = 0.30
STUD_HIGHLIGHT = 1.14
STUD_SHADE = 0.80

#: The bevel: the fraction of a stud the darker edge band occupies.
BEVEL = 0.12
BEVEL_SHADE = 0.72


class SceneError(ValueError):
    """A scene that cannot be rendered, refused by name."""


@dataclass(frozen=True)
class Placement:
    """One brick on the table: what it is, what colour, where, which way."""

    part: str
    colour_id: str
    #: Grid cell, not pixels. The layout turns these into pixels.
    row: int
    col: int
    #: 0 or 1. A quarter turn swaps the part's two extents on the image and
    #: does *not* change what the part is -- ``1x2`` turned is still ``1x2``,
    #: which is the rotation normalisation the project fixed in decision 2.
    turn: int = 0

    def extents(self) -> tuple[int, int]:
        """``(studs across, studs down)`` after the turn."""
        h, w = (int(v) for v in self.part.split("x"))
        return (w, h) if self.turn % 2 == 0 else (h, w)


@dataclass(frozen=True)
class Scene:
    """A scene and its ground truth, which are the same object."""

    scene_id: str
    placements: tuple[Placement, ...]
    #: Columns in the layout grid. Rows follow from the placement count.
    columns: int = 4

    def inventory(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.placements:
            counts[p.part] = counts.get(p.part, 0) + 1
        return dict(sorted(counts.items()))

    def colour_inventory(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.placements:
            key = f"{p.part}:{p.colour_id}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def ground_truth(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "n_bricks": len(self.placements),
            "inventory": self.inventory(),
            "colour_inventory": self.colour_inventory(),
            "placements": [
                {"part": p.part, "colour_id": p.colour_id, "row": p.row,
                 "col": p.col, "turn": p.turn} for p in self.placements],
        }

    def digest(self) -> str:
        import json

        return hashlib.sha256(
            json.dumps(self.ground_truth(), sort_keys=True,
                       separators=(",", ":")).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The conditions, frozen
# ---------------------------------------------------------------------------

#: Every axis, its baseline value first. The matrix below is the baseline
#: plus one axis moved, never two, so a result has one cause.
CONDITIONS: dict[str, tuple[str, ...]] = {
    "background": ("plain", "grey", "textured", "dark", "cluttered"),
    "lighting": ("even", "gradient", "dim", "harsh"),
    "shadow": ("none", "soft", "hard"),
    "viewpoint": ("top", "tilt_small", "tilt_large"),
    "occlusion": ("none", "partial", "heavy"),
    "quality": ("full", "downscaled", "noisy", "blurred", "compressed"),
}

BASELINE: dict[str, str] = {k: v[0] for k, v in CONDITIONS.items()}


def condition_matrix() -> list[dict]:
    """The baseline, then every single-axis variation. Order is frozen."""
    out = [dict(BASELINE)]
    for axis in CONDITIONS:
        for value in CONDITIONS[axis][1:]:
            out.append({**BASELINE, axis: value})
    return out


def condition_id(condition: dict) -> str:
    moved = sorted(k for k, v in condition.items() if v != BASELINE[k])
    if not moved:
        return "baseline"
    return "+".join(f"{k}={condition[k]}" for k in moved)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    """A generator seeded only from the caller's integer.

    Never from the clock, the process or the machine. Two runs of this module
    with the same seed produce the same pixels, and a test asserts it.
    """
    return np.random.default_rng(seed)


def layout(scene: Scene, condition: dict) -> list[tuple[Placement, float,
                                                        float]]:
    """``(placement, x_studs, y_studs)`` for every brick, top-left origin.

    Occlusion is applied here rather than at draw time: it is a fact about
    where the bricks are, and a scene whose bricks overlap is a scene whose
    detector will see fewer components than there are bricks. That is the
    outcome being measured, so it must be visible in the geometry.
    """
    cells: list[tuple[Placement, float, float]] = []
    col_w: dict[int, float] = {}
    row_h: dict[int, float] = {}
    for p in scene.placements:
        across, down = p.extents()
        col_w[p.col] = max(col_w.get(p.col, 0.0), float(across))
        row_h[p.row] = max(row_h.get(p.row, 0.0), float(down))

    x_of: dict[int, float] = {}
    running = MARGIN_STUDS
    for col in sorted(col_w):
        x_of[col] = running
        running += col_w[col] + GAP_STUDS
    y_of: dict[int, float] = {}
    running = MARGIN_STUDS
    for row in sorted(row_h):
        y_of[row] = running
        running += row_h[row] + GAP_STUDS

    shift = {"none": 0.0, "partial": 0.55, "heavy": 1.15}[
        condition["occlusion"]]
    for index, p in enumerate(scene.placements):
        x, y = x_of[p.col], y_of[p.row]
        if shift and index % 2 == 1:
            # Every second brick slides back toward its left neighbour, so an
            # occluded scene has a predictable number of merge candidates
            # rather than a random one.
            x -= shift + GAP_STUDS
        cells.append((p, x, y))
    return cells


def canvas_size(scene: Scene, condition: dict) -> tuple[int, int]:
    cells = layout(scene, condition)
    right = max(x + p.extents()[0] for p, x, _y in cells)
    bottom = max(y + p.extents()[1] for p, _x, y in cells)
    left = min(x for _p, x, _y in cells)
    width = int(round((right - min(0.0, left) + MARGIN_STUDS) * STUD_PX))
    height = int(round((bottom + MARGIN_STUDS) * STUD_PX))
    return width, height


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _background(width: int, height: int, kind: str, seed: int) -> np.ndarray:
    rng = _rng(seed)
    if kind == "plain":
        base = np.full((height, width, 3), 246.0)
    elif kind == "grey":
        base = np.full((height, width, 3), 168.0)
    elif kind == "dark":
        base = np.full((height, width, 3), 38.0)
    elif kind == "textured":
        base = np.full((height, width, 3), 205.0)
        # A deterministic wood-like grain: low-frequency sinusoids, no noise.
        yy = np.linspace(0, 8 * math.pi, height)[:, None]
        xx = np.linspace(0, 3 * math.pi, width)[None, :]
        grain = 14.0 * np.sin(yy + 0.6 * np.sin(xx))
        base += grain[:, :, None]
        base[:, :, 2] -= 18.0
        base[:, :, 1] -= 8.0
    elif kind == "cluttered":
        base = np.full((height, width, 3), 232.0)
        # Blobs the detector may propose as foreground. Deterministic from
        # the seed, and deliberately not brick-shaped: a false positive here
        # is the recogniser failing to reject a non-brick, which is one of
        # the things this baseline is for.
        for _ in range(9):
            cx = int(rng.integers(0, width))
            cy = int(rng.integers(0, height))
            r = int(rng.integers(STUD_PX // 2, STUD_PX * 2))
            shade = float(rng.integers(120, 190))
            y0, y1 = max(0, cy - r), min(height, cy + r)
            x0, x1 = max(0, cx - r), min(width, cx + r)
            if y1 <= y0 or x1 <= x0:
                continue
            gy = np.arange(y0, y1)[:, None] - cy
            gx = np.arange(x0, x1)[None, :] - cx
            disc = (gy ** 2 + gx ** 2) <= r ** 2
            base[y0:y1, x0:x1][disc] = shade
    else:
        raise SceneError(f"unknown background {kind!r}")
    return base


def _light(image: np.ndarray, kind: str) -> np.ndarray:
    height, width = image.shape[:2]
    if kind == "even":
        return image
    if kind == "dim":
        return image * 0.55
    if kind == "gradient":
        ramp = np.linspace(1.25, 0.62, width)[None, :, None]
        return image * ramp
    if kind == "harsh":
        yy = np.linspace(-1.0, 1.0, height)[:, None]
        xx = np.linspace(-1.0, 1.0, width)[None, :]
        falloff = 1.55 - 0.95 * np.sqrt(yy ** 2 + xx ** 2)
        return image * np.clip(falloff, 0.35, 1.6)[:, :, None]
    raise SceneError(f"unknown lighting {kind!r}")


def _draw_brick(image: np.ndarray, p: Placement, x: float, y: float,
                condition: dict) -> None:
    across, down = p.extents()
    rgb = np.asarray(colour(p.colour_id).rgb, dtype=np.float64)
    x0 = int(round(x * STUD_PX))
    y0 = int(round(y * STUD_PX))
    x1 = x0 + int(round(across * STUD_PX))
    y1 = y0 + int(round(down * STUD_PX))
    h, w = image.shape[:2]
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w, x1), min(h, y1)
    if x1c <= x0c or y1c <= y0c:
        return

    image[y0c:y1c, x0c:x1c] = rgb
    # The bevel: a darker band inside the edge, which is what gives the
    # component a measurable boundary against a similar background.
    band = max(1, int(round(BEVEL * STUD_PX)))
    dark = rgb * BEVEL_SHADE
    image[y0c:min(y0c + band, y1c), x0c:x1c] = dark
    image[max(y0c, y1c - band):y1c, x0c:x1c] = dark
    image[y0c:y1c, x0c:min(x0c + band, x1c)] = dark
    image[y0c:y1c, max(x0c, x1c - band):x1c] = dark

    # The studs. Their pitch is exactly one stud, which is what
    # ``estimate_pitch`` measures and what makes a 2x4 distinguishable from
    # a 1x8 of the same area.
    radius = STUD_RADIUS * STUD_PX
    for sy in range(down):
        for sx in range(across):
            cx = x0 + (sx + 0.5) * STUD_PX
            cy = y0 + (sy + 0.5) * STUD_PX
            ry0 = max(0, int(math.floor(cy - radius)))
            ry1 = min(h, int(math.ceil(cy + radius)) + 1)
            rx0 = max(0, int(math.floor(cx - radius)))
            rx1 = min(w, int(math.ceil(cx + radius)) + 1)
            if ry1 <= ry0 or rx1 <= rx0:
                continue
            gy = np.arange(ry0, ry1)[:, None] + 0.5 - cy
            gx = np.arange(rx0, rx1)[None, :] + 0.5 - cx
            dist = np.sqrt(gy ** 2 + gx ** 2)
            disc = dist <= radius
            ring = disc & (dist > radius * 0.72)
            image[ry0:ry1, rx0:rx1][disc] = np.clip(rgb * STUD_HIGHLIGHT,
                                                    0, 255)
            image[ry0:ry1, rx0:rx1][ring] = rgb * STUD_SHADE


def _shadow(image: np.ndarray, cells, kind: str) -> None:
    if kind == "none":
        return
    offset, strength = {"soft": (0.18, 0.86), "hard": (0.34, 0.62)}[kind]
    h, w = image.shape[:2]
    dx = int(round(offset * STUD_PX))
    for p, x, y in cells:
        across, down = p.extents()
        x0 = int(round(x * STUD_PX)) + dx
        y0 = int(round(y * STUD_PX)) + dx
        x1 = x0 + int(round(across * STUD_PX))
        y1 = y0 + int(round(down * STUD_PX))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        image[y0:y1, x0:x1] *= strength


def _tilt(image: np.ndarray, kind: str) -> np.ndarray:
    """A vertical foreshortening, which is what a tilt does to a top view.

    Implemented as a per-row horizontal scale about the centre, sampled with
    nearest neighbour so the result is exact integer arithmetic rather than
    an interpolation whose rounding could differ between BLAS builds.
    """
    if kind == "top":
        return image
    strength = {"tilt_small": 0.16, "tilt_large": 0.34}[kind]
    h, w = image.shape[:2]
    out = np.empty_like(image)
    centre = (w - 1) / 2.0
    for row in range(h):
        # Rows near the top are squeezed most: the far edge of the table.
        factor = 1.0 - strength * (1.0 - row / max(1, h - 1))
        src = centre + (np.arange(w) - centre) / factor
        idx = np.clip(np.rint(src).astype(int), 0, w - 1)
        out[row] = image[row][idx]
    return out


def _quality(image: np.ndarray, kind: str, seed: int) -> np.ndarray:
    if kind == "full":
        return image
    if kind == "downscaled":
        # Integer 3x box average then nearest-neighbour back up: exact, and
        # it destroys the stud pitch the way a low-resolution camera does.
        f = 3
        h, w = image.shape[0] // f * f, image.shape[1] // f * f
        small = image[:h, :w].reshape(h // f, f, w // f, f, 3).mean((1, 3))
        return np.repeat(np.repeat(small, f, 0), f, 1)
    if kind == "noisy":
        rng = _rng(seed + 977)
        return image + rng.normal(0.0, 15.0, image.shape)
    if kind == "blurred":
        out = image.astype(np.float64)
        k = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        k = k / k.sum()
        for axis in (0, 1):
            padded = np.apply_along_axis(
                lambda v: np.convolve(v, k, mode="same"), axis, out)
            out = padded
        return out
    if kind == "compressed":
        # Posterise to 5 bits and quantise 4x4 blocks to their mean: the two
        # artefacts a low-quality JPEG actually produces, without depending
        # on an encoder whose output varies by library version.
        out = np.floor(image / 8.0) * 8.0
        f = 4
        h, w = out.shape[0] // f * f, out.shape[1] // f * f
        block = out[:h, :w].reshape(h // f, f, w // f, f, 3).mean((1, 3))
        out[:h, :w] = np.repeat(np.repeat(block, f, 0), f, 1)
        return out
    raise SceneError(f"unknown quality {kind!r}")


def render(scene: Scene, condition: dict, *, seed: int = 0) -> np.ndarray:
    """One image, as a float RGB array in 0..255. Deterministic, always."""
    for axis, allowed in CONDITIONS.items():
        if condition.get(axis) not in allowed:
            raise SceneError(f"{axis}={condition.get(axis)!r} is not one of "
                             f"{list(allowed)}")
    width, height = canvas_size(scene, condition)
    image = _background(width, height, condition["background"], seed)
    cells = layout(scene, condition)
    _shadow(image, cells, condition["shadow"])
    for p, x, y in cells:
        _draw_brick(image, p, x, y, condition)
    image = _light(image, condition["lighting"])
    image = _tilt(image, condition["viewpoint"])
    image = _quality(image, condition["quality"], seed)
    return np.clip(image, 0.0, 255.0)


def as_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0, 255).astype(np.uint8)


def render_bytes(scene: Scene, condition: dict, *, seed: int = 0) -> bytes:
    """The image as uint8 bytes, which is what gets hashed and stored."""
    return as_uint8(render(scene, condition, seed=seed)).tobytes()


def digest_array(array: np.ndarray) -> str:
    """The one definition of an image's digest: its own shape, its own bytes.

    Taken from the array rather than from :func:`canvas_size`, because two
    of the quality conditions do not preserve the canvas size -- the
    downscale crops to a multiple of its block before averaging. Reading the
    size from the layout instead of from the pixels made the digest of those
    images disagree with the digest of the file they were written to, which
    looked exactly like tampering and was arithmetic.
    """
    payload = (f"{array.shape[1]}x{array.shape[0]}|".encode()
               + as_uint8(array).tobytes())
    return hashlib.sha256(payload).hexdigest()


def image_digest(scene: Scene, condition: dict, *, seed: int = 0) -> str:
    return digest_array(render(scene, condition, seed=seed))


def write_png(path, array: np.ndarray) -> None:
    """Write a PNG through Pillow. Only the storage step needs a library."""
    from PIL import Image

    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(str(path))


# ---------------------------------------------------------------------------
# The frozen scenes
# ---------------------------------------------------------------------------

#: The colours the scenes draw from, in a fixed order. A subset of the
#: palette: eight well-separated hues, so a colour error is a real confusion
#: and not two neighbouring greys that no camera would separate either.
SCENE_COLOURS: tuple[str, ...] = ("red", "blue", "yellow", "green", "white",
                                  "black", "orange", "light_grey")


def scenes(n: int = 8, *, seed: int = 20260829) -> tuple[Scene, ...]:
    """The frozen scene set. Same list every time, from one integer.

    Sizes rise across the set -- 3 bricks to 10 -- so the report can say
    whether accuracy falls with scene density, and every part in
    ``PART_VOCAB`` appears in at least one scene.
    """
    rng = _rng(seed)
    parts = list(PART_VOCAB)
    out = []
    for i in range(n):
        count = 3 + i
        chosen = [parts[(i + j) % len(parts)] for j in range(count)]
        placements = []
        for j, part in enumerate(chosen):
            placements.append(Placement(
                part=part,
                colour_id=SCENE_COLOURS[(i + j) % len(SCENE_COLOURS)],
                row=j // 3, col=j % 3,
                turn=int(rng.integers(0, 2))))
        out.append(Scene(scene_id=f"scene_{i:02d}",
                         placements=tuple(placements)))
    return tuple(out)


__all__ = [
    "STUD_PX", "CONDITIONS", "BASELINE", "SCENE_COLOURS", "SceneError",
    "Placement", "Scene", "condition_matrix", "condition_id", "layout",
    "canvas_size", "render", "render_bytes", "image_digest", "write_png",
    "scenes",
]
