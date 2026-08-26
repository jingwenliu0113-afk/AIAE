"""Foreground segmentation and connected components, in NumPy only.

The public datasets photograph and render bricks against a plain light
background, which is the one condition under which a threshold is an honest
segmenter rather than a wish.  This module does exactly that much and says so:
it estimates the background from the border, thresholds against it, labels the
components, and measures each one.  It does not detect bricks in a cluttered
scene and must not be described as though it did.

Written with no OpenCV, SciPy or scikit-image.  Partly because none of them is
in this project's pinned environment, and partly because a labelling pass
whose behaviour is defined by twenty lines here cannot drift when a wheel is
rebuilt.  The run-based labelling is linear in the number of foreground runs,
which for a 320-pixel image is a few hundred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Foreground must differ from the estimated background by at least this much,
#: on a 0-255 scale, in the channel-max sense.  Below it, a JPEG's own noise
#: would be segmented as objects.
MIN_CONTRAST = 26.0

#: A component smaller than this fraction of the image is dropped.  It is a
#: fraction rather than a pixel count so the same value holds at every input
#: size after :func:`~src.vision.preprocess.fit_long_side`.
MIN_AREA_FRACTION = 0.0025

#: A component larger than this fraction is the background itself leaking in
#: through a threshold that found nothing.
MAX_AREA_FRACTION = 0.96

#: How wide a border strip the background estimate is taken from.
BORDER_FRACTION = 0.06


class SegmentError(ValueError):
    """The image cannot be segmented under this module's stated assumption."""


@dataclass(frozen=True)
class Component:
    """One foreground blob and the scale-free measurements of it."""

    area: int
    box: tuple[int, int, int, int]          # x0, y0, x1, y1 (exclusive ends)
    centroid: tuple[float, float]
    length: float                            # principal extent, pixels
    width: float                             # secondary extent, pixels
    angle: float                             # principal axis, radians
    fill: float                              # area / (length * width)

    @property
    def aspect(self) -> float:
        return self.length / self.width if self.width > 0 else 0.0

    @property
    def box_area(self) -> int:
        x0, y0, x1, y1 = self.box
        return max(0, x1 - x0) * max(0, y1 - y0)

    def as_dict(self) -> dict:
        return {"area": self.area, "box": list(self.box),
                "centroid": [round(v, 3) for v in self.centroid],
                "length": round(self.length, 3), "width": round(self.width, 3),
                "aspect": round(self.aspect, 4), "fill": round(self.fill, 4)}


def background_colour(rgb: np.ndarray,
                      border_fraction: float = BORDER_FRACTION) -> np.ndarray:
    """Median colour of a border strip, as the background estimate.

    The median rather than the mean: a brick touching one edge shifts a mean
    and barely moves a median, and a brick touching an edge is common in the
    detection photographs.
    """
    array = np.asarray(rgb, dtype=np.float32)
    height, width = array.shape[:2]
    band_y = max(1, int(round(height * border_fraction)))
    band_x = max(1, int(round(width * border_fraction)))
    strips = [array[:band_y].reshape(-1, 3), array[-band_y:].reshape(-1, 3),
              array[:, :band_x].reshape(-1, 3),
              array[:, -band_x:].reshape(-1, 3)]
    return np.median(np.concatenate(strips, axis=0), axis=0)


def foreground_mask(rgb: np.ndarray, *, min_contrast: float = MIN_CONTRAST
                    ) -> tuple[np.ndarray, float]:
    """Boolean mask of pixels far enough from the background estimate.

    The distance is the largest absolute per-channel difference rather than a
    Euclidean one, so a brick that differs strongly in a single channel -- a
    saturated red on a grey card -- is not averaged into the background.

    Returns the mask and the threshold actually used, because the threshold is
    data dependent and a report that cannot say what it was cannot be checked.
    """
    array = np.asarray(rgb, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise SegmentError("segmentation expects an (h, w, 3) array")
    background = background_colour(array)
    distance = np.max(np.abs(array - background), axis=2)
    # Otsu on the distance image, then floored at the contrast minimum. Otsu
    # adapts to a dark brick on white and a light brick on grey alike; the
    # floor is what stops it from splitting an empty image down the middle of
    # its own noise and calling one half foreground.
    threshold = max(float(min_contrast), _otsu(distance))
    return distance >= threshold, threshold


def _otsu(values: np.ndarray, bins: int = 64) -> float:
    """Otsu's threshold over a non-negative image, on ``bins`` buckets."""
    flat = values.reshape(-1)
    top = float(flat.max())
    if top <= 0:
        return 0.0
    counts, edges = np.histogram(flat, bins=bins, range=(0.0, top))
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_low = np.cumsum(counts)
    weight_high = total - weight_low
    sum_low = np.cumsum(counts * centres)
    sum_total = sum_low[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_low = sum_low / weight_low
        mean_high = (sum_total - sum_low) / weight_high
        between = weight_low * weight_high * (mean_low - mean_high) ** 2
    between = np.nan_to_num(between, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return float(centres[int(np.argmax(between))])


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Four-connected labelling by row runs and union-find.

    Returns a label image where ``0`` is background and labels are ``1..n``
    renumbered in raster order of first appearance, so two runs on the same
    pixels always produce the same numbering.
    """
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise SegmentError("labelling expects a 2-D boolean mask")
    height, width = array.shape

    parent: list[int] = [0]

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    runs_by_row: list[list[tuple[int, int, int]]] = []
    for y in range(height):
        row = array[y]
        # Run starts and ends from the transitions of the padded row.
        padded = np.concatenate(([False], row, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        runs = []
        for k in range(0, len(edges), 2):
            start, stop = int(edges[k]), int(edges[k + 1])
            parent.append(len(parent))
            runs.append((start, stop, len(parent) - 1))
        runs_by_row.append(runs)
        if y:
            for start, stop, tag in runs:
                for pstart, pstop, ptag in runs_by_row[y - 1]:
                    if pstart < stop and start < pstop:
                        union(tag, ptag)

    renumber: dict[int, int] = {}
    labels = np.zeros((height, width), dtype=np.int32)
    for y, runs in enumerate(runs_by_row):
        for start, stop, tag in runs:
            root = find(tag)
            index = renumber.get(root)
            if index is None:
                index = renumber[root] = len(renumber) + 1
            labels[y, start:stop] = index
    return labels, len(renumber)


def measure(labels: np.ndarray, index: int) -> Component:
    """Second-moment measurements of one labelled component.

    ``length`` and ``width`` come from the eigenvalues of the covariance of
    the member pixels: for a filled rectangle of side ``a`` along an axis the
    variance is ``a**2 / 12``, so ``sqrt(12 * variance)`` recovers the side.
    That is a better estimate of a rotated brick's extent than its axis
    aligned bounding box, which grows with the rotation angle.
    """
    ys, xs = np.nonzero(labels == index)
    if ys.size == 0:
        raise SegmentError(f"there is no component numbered {index}")
    area = int(ys.size)
    cx, cy = float(xs.mean()), float(ys.mean())
    dx = xs - cx
    dy = ys - cy
    if area > 1:
        cov = np.array([[float((dx * dx).mean()), float((dx * dy).mean())],
                        [float((dx * dy).mean()), float((dy * dy).mean())]])
        values, vectors = np.linalg.eigh(cov)
        values = np.clip(values, 0.0, None)
        major, minor = float(values[1]), float(values[0])
        axis = vectors[:, 1]
        angle = float(np.arctan2(axis[1], axis[0]))
    else:
        major = minor = 1.0 / 12.0
        angle = 0.0
    length = max(1.0, float(np.sqrt(12.0 * major)))
    width = max(1.0, float(np.sqrt(12.0 * minor)))
    if width > length:
        length, width = width, length
    return Component(
        area=area,
        box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
        centroid=(cx, cy), length=length, width=width, angle=angle,
        fill=min(1.0, area / (length * width)))


def components(rgb: np.ndarray, *,
               min_area_fraction: float = MIN_AREA_FRACTION,
               max_area_fraction: float = MAX_AREA_FRACTION,
               limit: int = 64) -> tuple[list[Component], dict]:
    """Segment and measure, largest first, with the decisions recorded.

    The second return value is the diagnostic block: the threshold used, how
    many raw components were found and how many were dropped by each rule.  A
    detector that reports two boxes on a photograph of thirty bricks has to be
    able to say that it dropped twenty-eight and why.
    """
    array = np.asarray(rgb, dtype=np.float32)
    mask, threshold = foreground_mask(array)
    labels, count = label_components(mask)
    total = array.shape[0] * array.shape[1]
    floor = max(1, int(round(total * min_area_fraction)))
    ceiling = int(round(total * max_area_fraction))

    kept: list[Component] = []
    too_small = too_large = 0
    for index in range(1, count + 1):
        component = measure(labels, index)
        if component.area < floor:
            too_small += 1
            continue
        if component.area > ceiling:
            too_large += 1
            continue
        kept.append(component)
    kept.sort(key=lambda c: (-c.area, c.box))
    dropped_over_limit = max(0, len(kept) - limit)
    diagnostics = {
        "threshold": round(threshold, 3),
        "background_rgb": [round(v, 2) for v in background_colour(array)],
        "raw_components": count,
        "dropped_too_small": too_small,
        "dropped_too_large": too_large,
        "dropped_over_limit": dropped_over_limit,
        "kept": min(len(kept), limit),
        "foreground_fraction": round(float(mask.mean()), 5),
    }
    return kept[:limit], diagnostics
