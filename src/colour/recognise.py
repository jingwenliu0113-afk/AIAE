"""Colour from a crop: pick the pixels that are the brick, then match.

Shape and colour are separate problems here, and this is the colour half.  It
takes a crop that a detector already isolated, decides which of its pixels are
actually brick surface, and matches their average to the palette.

**Choosing the pixels is most of the work.**  A crop of a brick contains
background at the corners, a specular highlight on the studs that is nearly
white whatever the brick is, and a shadowed side that is nearly black.  Averaging
all of it turns every brick grey.  So the pixels are filtered:

* far enough from the crop's own border colour to be foreground at all;
* not in the brightest tail -- that is the highlight, and it is the studs'
  reflection of the lamp rather than the plastic;
* not in the darkest tail -- that is shadow and the gap between studs.

**Matching happens in CIELAB, not RGB.**  Euclidean distance in sRGB does not
correspond to how different two colours look, and the palette has several pairs
(the two greys, the two browns) that are close in RGB and easy to separate
perceptually.  The HSV view is computed too and returned as a feature, because
hue is what makes a saturated-colour decision legible in a report -- but the
decision is the CIELAB one.

**Confidence is a margin, and low confidence means a person decides.**  The
score is how much closer the best palette entry is than the second, so a brick
halfway between two greys comes back uncertain rather than arbitrarily grey, and
the UI puts it in front of the operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.colour.palette import (COLOUR_ORDER, PALETTE, ColourError,
                                check_contract, colour)

#: Below this the colour is not offered as an answer and the crop is handed to
#: a person.  It is a normalised margin, so it does not depend on the units of
#: the distance.
LOW_CONFIDENCE = 0.40

#: Fraction of the brightest foreground pixels dropped as specular highlight.
HIGHLIGHT_FRACTION = 0.22

#: Fraction of the darkest foreground pixels dropped as shadow.
SHADOW_FRACTION = 0.18

#: A crop with fewer usable pixels than this has nothing to measure.
MIN_PIXELS = 24

#: sRGB -> linear -> XYZ, D65.  Written out rather than imported because
#: neither scikit-image nor OpenCV is in this project's pinned environment.
_XYZ_FROM_LINEAR = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Undo the sRGB transfer function.  Values in ``[0, 1]``."""
    array = np.asarray(rgb, dtype=np.float64)
    return np.where(array <= 0.04045, array / 12.92,
                    ((array + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in ``[0, 255]`` to CIELAB, D65, last axis of size three."""
    array = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = srgb_to_linear(array)
    xyz = linear @ _XYZ_FROM_LINEAR.T
    scaled = xyz / _WHITE_D65
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(scaled > epsilon, np.cbrt(scaled),
                 (kappa * scaled + 16.0) / 116.0)
    lightness = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([lightness, a, b], axis=-1)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """sRGB in ``[0, 255]`` to HSV with hue in degrees and s/v in ``[0, 1]``."""
    array = np.asarray(rgb, dtype=np.float64) / 255.0
    high = array.max(axis=-1)
    low = array.min(axis=-1)
    span = high - low
    hue = np.zeros_like(high)
    with np.errstate(invalid="ignore", divide="ignore"):
        red, green, blue = array[..., 0], array[..., 1], array[..., 2]
        hue = np.where(
            span == 0, 0.0,
            np.where(high == red, ((green - blue) / span) % 6.0,
                     np.where(high == green, (blue - red) / span + 2.0,
                              (red - green) / span + 4.0)))
    hue = hue * 60.0
    saturation = np.where(high == 0, 0.0, span / np.where(high == 0, 1, high))
    return np.stack([hue, saturation, high], axis=-1)


def palette_lab() -> np.ndarray:
    """The palette as CIELAB rows, in :data:`COLOUR_ORDER`."""
    check_contract()
    return rgb_to_lab(np.array([entry.rgb for entry in PALETTE],
                               dtype=np.float64))


@dataclass(frozen=True)
class ColourReading:
    """What one crop's colour was measured to be, and how sure that is."""

    colour_id: str
    confidence: float
    mean_rgb: tuple[int, int, int]
    mean_hsv: tuple[float, float, float]
    mean_lab: tuple[float, float, float]
    candidates: tuple[tuple[str, float], ...]
    pixels_used: int
    pixels_total: int

    @property
    def low_confidence(self) -> bool:
        return self.confidence < LOW_CONFIDENCE

    @property
    def label(self) -> str | None:
        """The colour to act on, or ``None`` when a person should decide."""
        return None if self.low_confidence else self.colour_id

    def as_dict(self) -> dict:
        return {
            "colour_id": self.colour_id,
            "label": self.label,
            "confidence": round(self.confidence, 5),
            "low_confidence": self.low_confidence,
            "mean_rgb": list(self.mean_rgb),
            "mean_hsv": [round(v, 3) for v in self.mean_hsv],
            "mean_lab": [round(v, 3) for v in self.mean_lab],
            "candidates": [{"colour_id": name, "delta_e": round(value, 3)}
                           for name, value in self.candidates],
            "pixels_used": self.pixels_used,
            "pixels_total": self.pixels_total,
        }


class RecogniseError(ColourError):
    """The crop has nothing measurable in it."""


def usable_pixels(rgb: np.ndarray, mask: np.ndarray | None = None
                  ) -> tuple[np.ndarray, dict]:
    """The pixels that are brick surface, with the exclusions counted."""
    array = np.asarray(rgb, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise RecogniseError("a colour reading needs an (h, w, 3) crop")
    flat = array.reshape(-1, 3)
    total = int(flat.shape[0])

    if mask is not None:
        keep = np.asarray(mask, dtype=bool).reshape(-1)
        if keep.shape[0] != total:
            raise RecogniseError("the mask does not match the crop")
        flat = flat[keep]
    else:
        from src.vision.segment import foreground_mask

        keep, _threshold = foreground_mask(array.astype(np.float32))
        flat = flat[keep.reshape(-1)]
    after_foreground = int(flat.shape[0])
    if after_foreground < MIN_PIXELS:
        raise RecogniseError(
            f"only {after_foreground} foreground pixel(s) in this crop; there "
            f"is not enough surface to read a colour from (minimum "
            f"{MIN_PIXELS})")

    value = flat.max(axis=1)
    order = np.argsort(value, kind="stable")
    low = int(round(after_foreground * SHADOW_FRACTION))
    high = int(round(after_foreground * HIGHLIGHT_FRACTION))
    middle = order[low:after_foreground - high] if (
        after_foreground - high - low >= MIN_PIXELS) else order
    used = flat[middle]
    return used, {"pixels_total": total,
                  "pixels_foreground": after_foreground,
                  "pixels_used": int(used.shape[0]),
                  "dropped_shadow": low if middle is not order else 0,
                  "dropped_highlight": high if middle is not order else 0,
                  "trim_skipped": middle is order}


def read_colour(rgb: np.ndarray, mask: np.ndarray | None = None
                ) -> ColourReading:
    """Measure one crop's colour against the palette."""
    used, counts = usable_pixels(rgb, mask)
    mean_rgb = used.mean(axis=0)
    lab = rgb_to_lab(mean_rgb)
    hsv = rgb_to_hsv(mean_rgb)
    distances = np.linalg.norm(palette_lab() - lab, axis=1)
    order = np.argsort(distances, kind="stable")
    best, second = float(distances[order[0]]), float(distances[order[1]])
    # A margin normalised by the pair's own separation: "twice as close to red
    # as to orange" is confident whatever the absolute distances are, and an
    # absolute threshold would be a threshold on lighting.
    confidence = 0.0 if second <= 0 else min(1.0, (second - best) / second)
    return ColourReading(
        colour_id=COLOUR_ORDER[int(order[0])],
        confidence=float(confidence),
        mean_rgb=tuple(int(round(v)) for v in mean_rgb),
        mean_hsv=tuple(float(v) for v in hsv),
        mean_lab=tuple(float(v) for v in lab),
        candidates=tuple((COLOUR_ORDER[int(i)], float(distances[i]))
                         for i in order[:3]),
        pixels_used=counts["pixels_used"],
        pixels_total=counts["pixels_total"])


def delta_e(colour_id: str, rgb) -> float:
    """CIELAB distance from a measured colour to a palette entry."""
    entry = colour(colour_id)
    return float(np.linalg.norm(
        rgb_to_lab(np.asarray(rgb, dtype=np.float64))
        - rgb_to_lab(np.array(entry.rgb, dtype=np.float64))))
