"""The traditional-CV classifier: measure the silhouette, find the stud pitch.

Nothing is learned here and nothing is fitted.  The expected values come from
the parts themselves -- a ``2x4`` is two studs by four and twice as long as it
is wide -- so the decision table is *derived* from ``PART_VOCAB`` rather than
written down beside it.

Three measurements, in the order they matter:

**The silhouette's elongation.**  Second moments of the foreground component,
turned into a length and a width.  Better than an axis-aligned bounding box,
which grows with the rotation angle of a brick lying at forty-five degrees.

**The stud pitch.**  A brick is a periodic object: studs on top, tube openings
underneath, ridges along the side.  Projecting a gradient-magnitude image onto
the brick's own principal axis produces a profile whose period is the stud
pitch, and the pitch divides the length and the width into stud counts -- which
is what actually names the part.  This is the measurement that separates
``1x1`` from ``2x2`` and ``1x2`` from ``2x4``, pairs that elongation cannot tell
apart at all.  The autocorrelation peak height comes back as a strength, and
when it is weak the pitch is not trusted and elongation carries the decision.

**A stud blob count**, kept as a reported feature rather than a deciding one.
It assumes a top-lit view of the studs, and much of the public archive is
photographed and rendered from below or nearly edge-on, so it is informative
about the image and not reliable about the part.

**The stated assumption, and what happens off it.**  This method wants a brick
whose repeated structure is visible against a plain background.  The public
single-brick archive is *not* mostly that: many of its members are rendered
from underneath, showing the hollow tubes, and some are nearly edge-on, so the
long-axis period is foreshortened or invisible.  That is a real limit of the
method on this data, it is reported as one, and the features that produced each
decision travel with it in ``Prediction.features`` so a confusion can be read
back to the measurement that caused it.
"""

from __future__ import annotations

import math

import numpy as np

from src.data.bricks import MAX_EXTENT
from src.rendering.ldr import LDU_BRICK, LDU_STUD
from src.vision.classes import CLASS_ORDER, check_contract
from src.vision.preprocess import fit_long_side
from src.vision.schema import METHOD_CV, Prediction, from_scores, normalise_scores
from src.vision.segment import Component, components, foreground_mask

#: How forgiving the elongation term is, in log units.  0.30 means a measured
#: 2.6:1 still gives a 2:1 part most of its score, which is the sort of error a
#: tilted photograph produces.
ASPECT_TOLERANCE = 0.30

#: How forgiving the stud-count term is, in studs per side.
EXTENT_TOLERANCE = 0.85

#: Weight of the stud-count term relative to elongation, at full pitch
#: strength.
EXTENT_WEIGHT = 1.2

#: Weight of the blob count in the decision, and it is **zero**.
#:
#: Not an oversight.  A sweep over these four constants on the *validation*
#: split -- never the test split -- found every non-zero value made the
#: baseline worse, on both real photographs and renders, by three to eight
#: points.  The reason is in the data rather than in the code: the archive's
#: members are largely rendered or photographed from below or nearly edge-on,
#: so a bright blob is as often a tube opening or a specular edge as a stud.
#: The count is still measured and still travels in ``features``, because it is
#: informative about the image; it just does not get a vote.
STUD_WEIGHT = 0.0

#: How forgiving the blob-count term would be, in studs, if it were weighted.
STUD_TOLERANCE = 2.5

#: Autocorrelation peak height below which the pitch is not believed.  Below
#: it the extent term is scaled down and elongation decides.  Chosen by the
#: same validation sweep.
MIN_PITCH_STRENGTH = 0.20

#: The pitch is searched between these fractions of the profile length, and
#: both ends are derived rather than chosen.
#:
#: The lower end is ``1 / MAX_EXTENT``: no part in the vocabulary has more than
#: eight studs along a side, so a period shorter than an eighth of the brick is
#: not a stud pitch and searching for one only finds noise. Getting this wrong
#: is not harmless -- at 0.09 the search started below the eighth and happily
#: reported a period no part could have.
#:
#: The upper end is a half: a period longer than half the profile cannot repeat
#: even once, so there is nothing for the autocorrelation to see.
PITCH_MIN_FRACTION = 1.0 / MAX_EXTENT
PITCH_MAX_FRACTION = 0.55

#: Smallest believable pitch in pixels, after downscaling.  Below this the
#: profile is measuring JPEG blocks.
MIN_PITCH_PIXELS = 4.0

#: How close to the best autocorrelation peak a shorter lag has to be before
#: it is preferred as the fundamental.  Autocorrelation peaks at twice the true
#: period almost as strongly as at the period itself, so taking the maximum
#: lands on the harmonic and halves the stud count -- naming a 1x8 a 1x4.
HARMONIC_TOLERANCE = 0.86

#: A stud blob must cover at least this fraction of the brick's area to count,
#: which drops speckle, and at most this fraction, which drops a single
#: blown-out highlight spanning the whole face.
STUD_MIN_AREA_FRACTION = 0.004
STUD_MAX_AREA_FRACTION = 0.16

#: How far above the brick's own median luminance, in units of its own spread,
#: a pixel has to be to be a candidate stud pixel.
STUD_BRIGHTNESS_SIGMA = 0.55

#: A brick's own height, in stud widths.  Derived from the LDraw constants
#: already pinned against the reference implementation -- 24 LDU tall and 20
#: LDU per stud -- rather than written here as a number.
#:
#: It matters because of what the measurements showed.  A ``1x6`` brick seen at
#: an angle does not present a 6:1 silhouette: it is one stud wide and 1.2
#: studs *tall*, so its short extent is set by its height rather than by its
#: width, and the silhouette is nearer 5:1.  A ``2x6`` is two studs wide, so
#: its height changes nothing and it does present 3:1 -- which is exactly what
#: was measured. Correcting the expected value with the brick's real geometry
#: is not fitting to the data; it is using a dimension the project already
#: knows.
HEIGHT_IN_STUDS = LDU_BRICK / LDU_STUD


def expected_geometry() -> dict[str, tuple[int, int, float]]:
    """``part -> (short studs, long studs, silhouette aspect)``.

    Every number comes from the part name and from the pinned LDraw
    dimensions, so correcting ``PART_VOCAB`` or those constants changes this
    table with them and cannot leave a stale row behind.

    The aspect is the *silhouette* aspect, not the plan aspect: each extent is
    at least the brick's own height, because a brick photographed from any
    angle other than straight down projects its height into the outline.
    """
    check_contract()
    out: dict[str, tuple[int, int, float]] = {}
    for part in CLASS_ORDER:
        short, _, long_side = part.partition("x")
        a, b = int(short), int(long_side)
        seen_short = max(float(a), HEIGHT_IN_STUDS)
        seen_long = max(float(b), HEIGHT_IN_STUDS)
        out[part] = (a, b, seen_long / seen_short)
    return out


def _luminance(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb, dtype=np.float32)
    return (0.2126 * array[..., 0] + 0.7152 * array[..., 1]
            + 0.0722 * array[..., 2])


def gradient_magnitude(luminance: np.ndarray) -> np.ndarray:
    """Central-difference gradient magnitude, written out.

    A Sobel kernel from a library would do the same thing; three lines here
    keep the whole feature path readable and independent of what is installed.
    """
    array = np.asarray(luminance, dtype=np.float32)
    dy = np.zeros_like(array)
    dx = np.zeros_like(array)
    dy[1:-1, :] = (array[2:, :] - array[:-2, :]) * 0.5
    dx[:, 1:-1] = (array[:, 2:] - array[:, :-2]) * 0.5
    return np.sqrt(dx * dx + dy * dy)


def axis_profiles(rgb: np.ndarray, mask: np.ndarray, component: Component
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Gradient energy projected onto the component's own two axes.

    Rotating the coordinates rather than the image keeps the projection exact
    and avoids a second resampling step: every foreground pixel contributes its
    gradient magnitude to one bin of each profile.
    """
    array = np.asarray(mask, dtype=bool)
    energy = gradient_magnitude(_luminance(rgb))
    ys, xs = np.nonzero(array)
    if ys.size == 0:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)
    cx, cy = component.centroid
    cos = math.cos(component.angle)
    sin = math.sin(component.angle)
    along = (xs - cx) * cos + (ys - cy) * sin
    across = -(xs - cx) * sin + (ys - cy) * cos
    weights = energy[ys, xs].astype(np.float64)

    def project(values: np.ndarray, extent: float) -> np.ndarray:
        bins = max(4, int(round(extent)))
        low = float(values.min())
        high = float(values.max())
        if high <= low:
            return np.zeros(bins, dtype=np.float64)
        index = np.clip(((values - low) / (high - low) * (bins - 1)).astype(int),
                        0, bins - 1)
        out = np.zeros(bins, dtype=np.float64)
        np.add.at(out, index, weights)
        counts = np.zeros(bins, dtype=np.float64)
        np.add.at(counts, index, 1.0)
        return out / np.maximum(counts, 1.0)

    return project(along, component.length), project(across, component.width)


def estimate_pitch(profile: np.ndarray) -> tuple[float, float]:
    """Dominant period of a 1-D profile, and how strongly it is present.

    The profile is detrended and normalised first, so the answer depends on the
    ridges rather than on how bright the brick is.  The strength is the height
    of the best autocorrelation peak, in ``[0, 1]``; a flat profile scores zero
    and the caller then knows not to use the pitch.
    """
    values = np.asarray(profile, dtype=np.float64)
    if values.size < 8:
        return 0.0, 0.0
    centred = values - values.mean()
    norm = float(np.dot(centred, centred))
    if norm <= 0:
        return 0.0, 0.0
    low = max(int(round(values.size * PITCH_MIN_FRACTION)), 2)
    high = int(round(values.size * PITCH_MAX_FRACTION))
    if high <= low:
        return 0.0, 0.0
    scores: dict[int, float] = {}
    best_lag, best_score = 0, 0.0
    for lag in range(low, high + 1):
        overlap = float(np.dot(centred[:-lag], centred[lag:]) / norm)
        scores[lag] = overlap
        if overlap > best_score:
            best_lag, best_score = lag, overlap
    if best_lag == 0 or best_score <= 0:
        return 0.0, 0.0
    # The fundamental, not one of its harmonics. Autocorrelation peaks just as
    # strongly at twice the true period, and taking the maximum lands on the
    # harmonic often enough to halve a stud count -- which names the wrong
    # part. So the smallest lag that is nearly as good as the best is the
    # answer, and the strength reported is that lag's own.
    fundamental = min(lag for lag, value in scores.items()
                      if value >= HARMONIC_TOLERANCE * best_score)
    if fundamental < MIN_PITCH_PIXELS:
        return 0.0, 0.0
    return float(fundamental), max(0.0, min(1.0, scores[fundamental]))


def fill_rows(mask: np.ndarray) -> np.ndarray:
    """Fill each row between its first and last set pixel.

    A brick's silhouette is convex enough for this to be its filled outline,
    and filling matters: a stud catching the light is often *lighter than the
    background*, so the threshold that finds the brick excludes its own studs
    and a stud count inside the raw mask comes back zero. Filling the outline
    first puts them back inside it.
    """
    array = np.asarray(mask, dtype=bool)
    out = np.zeros_like(array)
    for row in range(array.shape[0]):
        set_pixels = np.flatnonzero(array[row])
        if set_pixels.size:
            out[row, set_pixels[0]:set_pixels[-1] + 1] = True
    return out


def count_studs(rgb: np.ndarray, mask: np.ndarray) -> tuple[int, dict]:
    """Count bright blobs inside a brick mask, with the rejects recorded."""
    from src.vision.segment import label_components

    array = fill_rows(mask)
    area = int(array.sum())
    if area <= 0:
        return 0, {"stud_candidates": 0, "stud_rejected_small": 0,
                   "stud_rejected_large": 0}
    luminance = _luminance(rgb)
    inside = luminance[array]
    median = float(np.median(inside))
    spread = float(np.percentile(inside, 84) - median)
    if spread <= 0:
        spread = max(1.0, float(inside.std()))
    bright = array & (luminance >= median + STUD_BRIGHTNESS_SIGMA * spread)
    labels, count = label_components(bright)
    floor = max(1, int(round(area * STUD_MIN_AREA_FRACTION)))
    ceiling = max(floor + 1, int(round(area * STUD_MAX_AREA_FRACTION)))
    kept = small = large = 0
    if count:
        sizes = np.bincount(labels.reshape(-1), minlength=count + 1)[1:]
        for size in sizes.tolist():
            if size < floor:
                small += 1
            elif size > ceiling:
                large += 1
            else:
                kept += 1
    return kept, {"stud_candidates": int(count), "stud_rejected_small": small,
                  "stud_rejected_large": large}


def extents_from_pitch(component: Component, pitch: float
                       ) -> tuple[float, float]:
    """Stud counts along each axis, from one pitch estimate.

    One pitch for both axes on purpose: a stud grid is square, so a pitch
    measured along the long side divides the short side too.  Measuring the two
    independently would let a foreshortened short side invent its own scale.
    """
    if pitch <= 0:
        return 0.0, 0.0
    long_studs = component.length / pitch
    short_studs = component.width / pitch
    return (max(1.0, min(float(MAX_EXTENT), short_studs)),
            max(1.0, min(float(MAX_EXTENT), long_studs)))


def score_features(aspect: float, short_studs: float, long_studs: float,
                   studs: int, pitch_strength: float) -> list[float]:
    """Per-class score from the measurements, in class order.

    Every term is an exponential decay, so no class is ever scored at exactly
    zero: measurements that fit nothing land on a flat vector, which
    :func:`~src.vision.schema.normalise_scores` turns into a uniform
    distribution and the schema reports as low confidence.  That is the honest
    outcome for "these features do not describe any of the eight".
    """
    if not math.isfinite(aspect) or aspect <= 0:
        return [1.0] * len(CLASS_ORDER)
    # Below the strength floor the pitch is noise; between the floor and one it
    # is scaled, so a marginal period nudges the decision instead of deciding
    # it.
    trust = 0.0 if pitch_strength < MIN_PITCH_STRENGTH else min(
        1.0, (pitch_strength - MIN_PITCH_STRENGTH) / (1.0 - MIN_PITCH_STRENGTH))
    blobs = min(float(studs), MAX_EXTENT * MAX_EXTENT)
    out = []
    for part, (want_short, want_long, want_aspect) in \
            expected_geometry().items():
        penalty = abs(math.log(aspect / want_aspect)) / ASPECT_TOLERANCE
        if trust > 0:
            extent_error = (abs(short_studs - want_short)
                            + abs(long_studs - want_long))
            penalty += (EXTENT_WEIGHT * trust
                        * extent_error / EXTENT_TOLERANCE)
        penalty += (STUD_WEIGHT * abs(blobs - want_short * want_long)
                    / STUD_TOLERANCE)
        out.append(math.exp(-penalty))
    return out


def _measure(small: np.ndarray, mask: np.ndarray, component: Component,
             diagnostics: dict) -> Prediction:
    x0, y0, x1, y1 = component.box
    window = np.zeros_like(mask)
    window[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    along, _across = axis_profiles(small, window, component)
    pitch, strength = estimate_pitch(along)
    short_studs, long_studs = extents_from_pitch(component, pitch)
    studs, stud_diagnostics = count_studs(small, window)
    features = {
        "segmented": True,
        "aspect": round(component.aspect, 4),
        "length_px": round(component.length, 2),
        "width_px": round(component.width, 2),
        "pitch_px": round(pitch, 3),
        "pitch_strength": round(strength, 4),
        "studs_long": round(long_studs, 3),
        "studs_short": round(short_studs, 3),
        "studs": studs,
        "fill": round(component.fill, 4),
        "area": component.area,
        **stud_diagnostics,
        **diagnostics,
    }
    return from_scores(
        METHOD_CV,
        normalise_scores(score_features(component.aspect, short_studs,
                                       long_studs, studs, strength)),
        features=features)


def classify_array(rgb: np.ndarray) -> Prediction:
    """Classify one image believed to hold a single brick.

    The largest foreground component is taken as the brick.  When nothing is
    segmented at all, a uniform score vector comes back rather than an
    exception: "no brick found" has to be a low-confidence prediction the
    correction interface can show, not a crash in the middle of a batch.
    """
    small = fit_long_side(np.asarray(rgb, dtype=np.float32))
    found, diagnostics = components(small, limit=8)
    if not found:
        # Normalised, so the confidence is 1/8 and the schema reports it as low
        # confidence. An unnormalised flat vector would come back as a
        # perfectly confident 1x1, which is the opposite of what happened.
        return from_scores(
            METHOD_CV, normalise_scores([1.0] * len(CLASS_ORDER)),
            features={"segmented": False, "reason": "no foreground component",
                      **diagnostics})
    mask, _threshold = foreground_mask(small)
    return _measure(small, mask, found[0],
                    {**diagnostics, "components": len(found)})


def classify_component(rgb: np.ndarray, component: Component) -> Prediction:
    """Classify a crop a detector already isolated.

    The component's measurements are reused verbatim, so a box and its label
    are measured from the same pixels; re-segmenting the crop would answer a
    different question.
    """
    array = np.asarray(rgb, dtype=np.float32)
    mask, _threshold = foreground_mask(array)
    return _measure(array, mask, component, {"from_detector": True})
