"""One way in for every image, whoever supplied it.

Every image this project looks at -- a dataset member, a file on disk, an
upload from the browser -- arrives through :func:`decode_image`, so the limits
are in one place and cannot be bypassed by using a different entry point.  The
UI's upload guard, the dataset reader and the evaluation all call the same
function with the same caps.

The caps are not tidiness.  An image header is attacker-controlled data:

* a 40 KB PNG can declare 60,000 x 60,000 pixels and expand to fourteen
  gigabytes when decoded, so the pixel count is checked against the header
  *before* the pixels are produced;
* a decoder handed an unexpected format is a decoder running code paths this
  project never tests, so the format is an allowlist of two;
* a multi-frame image would have the caller silently look at frame one of
  many, so more than one frame is refused rather than truncated.

Preprocessing is written out here rather than delegated to a processor object
downloaded beside the weights.  The values are pinned constants, and
:func:`check_processor_config` compares them against the published
configuration when it is available -- so the pinning is checked rather than
merely asserted, and a silent upstream change to the normalisation cannot
shift every prediction while every file still looks right.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

#: The two formats the public datasets use, and the two the UI accepts.
ALLOWED_FORMATS = ("PNG", "JPEG")

#: Suffixes that may name an image, lowercase.  Checked in addition to the
#: decoded format, because a caller that trusts a suffix is a caller that can
#: be handed a renamed file.
ALLOWED_SUFFIXES = (".png", ".jpg", ".jpeg")

#: An upload larger than this is refused before it is decoded.  The dataset's
#: own photographs are 2448x3264 JPEGs of roughly two megabytes.
MAX_IMAGE_BYTES = 24 * 1024 * 1024

#: Total pixels, checked against the header.  A 6000x6000 photograph passes;
#: a decompression bomb declaring tens of thousands on a side does not.
MAX_IMAGE_PIXELS = 40_000_000

#: The smallest image with enough to look at.  Below this the CV features are
#: measuring noise.
MIN_IMAGE_SIDE = 16

#: Longest side the CV analysis works at.  Downscaling first makes the
#: connected-component pass bounded in time regardless of input size, and the
#: features it computes -- aspect ratio, stud spacing -- are scale free.
CV_LONG_SIDE = 320

# --- the pinned preprocessing for the learned classifier -------------------

#: Shorter side after resizing, then a centre crop of :data:`CROP_SIZE`.  These
#: are the pinned backbone's published values -- ``size`` 224 at ``crop_pct``
#: 0.875 is a 256-pixel short side and a 224 crop -- written here so a run is
#: reproducible from this file alone.
RESIZE_SHORT_SIDE = 256
CROP_SIZE = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

#: The published configuration asks for bicubic resampling and this project
#: resamples bilinearly, in :func:`resize_rgb`, for a reason worth stating
#: rather than burying: the same twenty lines of arithmetic have to run on the
#: CUDA node that fits the head and on the Mac that serves it, and two builds
#: of an imaging library do not guarantee the same bicubic output. The
#: deviation is applied identically during fine-tuning and at inference, so it
#: is part of the model rather than a mismatch inside it -- but it does mean
#: the head is fitted to bilinear inputs and would be slightly off if anyone
#: later fed it bicubic ones.
PUBLISHED_RESAMPLE = 3
RESAMPLE_NOTE = ("bilinear, implemented in src.vision.preprocess.resize_rgb; "
                 "the published configuration asks for bicubic and the "
                 "deviation is applied identically at fit and inference time")


class ImageError(ValueError):
    """The image is refused.  Nothing was decoded, or nothing usable was."""


@dataclass(frozen=True)
class LoadedImage:
    """Decoded pixels plus what was decided about them on the way in.

    ``rgb`` is ``uint8`` with shape ``(height, width, 3)``.  Alpha is
    composited onto white rather than dropped: the renders in the public
    detection set are cut-outs, and dropping alpha would leave a black
    background that no photograph in the set has.
    """

    rgb: np.ndarray
    width: int
    height: int
    source_format: str
    had_alpha: bool

    @property
    def pixels(self) -> int:
        return self.width * self.height


def _open(data: bytes):
    """Open with PIL, with the header-level checks done before decoding."""
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise ImageError(
            "Pillow is required to read an image; install the pinned vision "
            "requirements") from exc

    # PIL's own bomb guard raises a warning-turned-error at a threshold of its
    # own. This project sets its own limit and checks the header itself, so the
    # message a caller gets names this project's rule rather than PIL's.
    try:
        handle = Image.open(io.BytesIO(data))
    except UnidentifiedImageError as exc:
        raise ImageError("this is not an image in a format this project "
                         f"reads: {list(ALLOWED_FORMATS)}") from exc
    except Exception as exc:                      # noqa: BLE001 - see message
        raise ImageError(f"the image could not be opened: {exc}") from exc
    return handle


def decode_image(data: bytes, *, max_bytes: int = MAX_IMAGE_BYTES,
                 max_pixels: int = MAX_IMAGE_PIXELS) -> LoadedImage:
    """Decode bytes into RGB pixels, or refuse and decode nothing."""
    if not isinstance(data, (bytes, bytearray)):
        raise ImageError(f"an image is bytes, not {type(data).__name__}")
    if not data:
        raise ImageError("the image is empty")
    if len(data) > max_bytes:
        raise ImageError(
            f"the image is {len(data)} bytes, over the {max_bytes} limit; it "
            "is refused rather than decoded")

    handle = _open(data)
    fmt = (handle.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise ImageError(
            f"format {fmt or 'unknown'!r} is not one of "
            f"{list(ALLOWED_FORMATS)}")
    width, height = handle.size
    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        raise ImageError(
            f"the image is {width}x{height}; both sides must be at least "
            f"{MIN_IMAGE_SIDE} pixels")
    if width * height > max_pixels:
        raise ImageError(
            f"the image header declares {width}x{height} = {width * height} "
            f"pixels, over the {max_pixels} limit. The header is checked "
            "before decoding, so a declared size this large costs nothing to "
            "refuse")
    frames = getattr(handle, "n_frames", 1)
    if frames != 1:
        raise ImageError(
            f"the image has {frames} frames; a caller looking at frame one of "
            "many is a caller looking at the wrong thing")

    had_alpha = handle.mode in ("RGBA", "LA", "PA") or "transparency" in (
        handle.info or {})
    try:
        if had_alpha:
            from PIL import Image

            rgba = handle.convert("RGBA")
            flat = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            flat.alpha_composite(rgba)
            pixels = np.asarray(flat.convert("RGB"), dtype=np.uint8)
        else:
            pixels = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    except Exception as exc:                      # noqa: BLE001 - see message
        raise ImageError(f"the image could not be decoded: {exc}") from exc
    finally:
        handle.close()

    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ImageError("the decoded image is not three-channel RGB")
    return LoadedImage(rgb=pixels, width=int(pixels.shape[1]),
                       height=int(pixels.shape[0]), source_format=fmt,
                       had_alpha=bool(had_alpha))


def read_image(path, **kw) -> LoadedImage:
    """Decode an image from disk, checking the suffix as well as the bytes."""
    from pathlib import Path

    target = Path(path)
    if target.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ImageError(
            f"{target.name!r} does not end in one of "
            f"{list(ALLOWED_SUFFIXES)}")
    if not target.is_file():
        raise ImageError(f"there is no image at {target}")
    return decode_image(target.read_bytes(), **kw)


# ---------------------------------------------------------------------------
# Resampling, written here so both paths are deterministic and identical
# ---------------------------------------------------------------------------

def resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resize with the sample grid written out.

    Own implementation rather than a library call, for one reason that matters
    later: the learned classifier's inputs have to be byte-identical between
    the machine that trained it and the machine that runs it, and two versions
    of an imaging library do not guarantee that.  Twenty lines of arithmetic
    do.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ImageError("resize expects an (h, w, 3) array")
    if width < 1 or height < 1:
        raise ImageError("a resize target must be at least one pixel a side")
    src_h, src_w = rgb.shape[:2]
    if (src_h, src_w) == (height, width):
        return rgb.astype(np.float32)

    source = rgb.astype(np.float32)
    # Half-pixel centres: without the 0.5 offsets the output is shifted by
    # half a pixel per axis, which is invisible on inspection and shows up as
    # a small constant accuracy loss.
    ys = (np.arange(height, dtype=np.float64) + 0.5) * src_h / height - 0.5
    xs = (np.arange(width, dtype=np.float64) + 0.5) * src_w / width - 0.5
    ys = np.clip(ys, 0, src_h - 1)
    xs = np.clip(xs, 0, src_w - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = (ys - y0).astype(np.float32)[:, None, None]
    wx = (xs - x0).astype(np.float32)[None, :, None]

    top = source[y0][:, x0] * (1 - wx) + source[y0][:, x1] * wx
    bottom = source[y1][:, x0] * (1 - wx) + source[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def fit_long_side(rgb: np.ndarray, long_side: int = CV_LONG_SIDE) -> np.ndarray:
    """Downscale so the longer side is ``long_side``; never upscale."""
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= long_side:
        return rgb.astype(np.float32)
    scale = long_side / longest
    return resize_rgb(rgb, max(1, round(width * scale)),
                      max(1, round(height * scale)))


def model_tensor(rgb: np.ndarray) -> np.ndarray:
    """Resize, centre crop and normalise into ``(3, 224, 224)`` float32.

    Returns a plain NumPy array.  The caller turns it into a torch tensor, so
    this module -- and everything that only needs the CV path -- keeps working
    on a machine with no torch installed.
    """
    height, width = rgb.shape[:2]
    scale = RESIZE_SHORT_SIDE / min(height, width)
    resized = resize_rgb(rgb, max(CROP_SIZE, round(width * scale)),
                         max(CROP_SIZE, round(height * scale)))
    rh, rw = resized.shape[:2]
    top = (rh - CROP_SIZE) // 2
    left = (rw - CROP_SIZE) // 2
    crop = resized[top:top + CROP_SIZE, left:left + CROP_SIZE]
    scaled = crop / 255.0
    mean = np.asarray(IMAGE_MEAN, dtype=np.float32)
    std = np.asarray(IMAGE_STD, dtype=np.float32)
    return np.transpose((scaled - mean) / std, (2, 0, 1)).astype(np.float32)


def check_processor_config(config: dict) -> None:
    """Compare a published preprocessor config against the pinned values.

    Called when the model directory ships one.  A mismatch is refused: the
    pinned numbers exist so the preprocessing is readable from this file, and
    an upstream change that silently disagrees with them would move every
    prediction while every file still looked correct.
    """
    if not isinstance(config, dict):
        raise ImageError("a preprocessor configuration must be a JSON object")
    mean = config.get("image_mean")
    std = config.get("image_std")
    if mean is not None and tuple(float(v) for v in mean) != IMAGE_MEAN:
        raise ImageError(
            f"the published image_mean {mean} is not the pinned {IMAGE_MEAN}")
    if std is not None and tuple(float(v) for v in std) != IMAGE_STD:
        raise ImageError(
            f"the published image_std {std} is not the pinned {IMAGE_STD}")
    crop = config.get("crop_size", config.get("size"))
    if isinstance(crop, dict):
        crop = crop.get("height") or crop.get("width") or crop.get(
            "shortest_edge")
    if crop is not None and int(crop) != CROP_SIZE:
        raise ImageError(
            f"the published crop size {crop} is not the pinned {CROP_SIZE}")
    pct = config.get("crop_pct")
    if pct is not None:
        implied = round(CROP_SIZE / float(pct))
        if implied != RESIZE_SHORT_SIDE:
            raise ImageError(
                f"the published crop_pct {pct} implies a {implied}-pixel short "
                f"side, not the pinned {RESIZE_SHORT_SIDE}")


def crop_box(rgb: np.ndarray, box, *, pad: int = 0) -> np.ndarray:
    """Cut a box out of an image, clamped to its bounds.

    ``box`` is ``(x0, y0, x1, y1)`` with ``x1``/``y1`` exclusive, matching the
    detection boxes everywhere else in this package.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    height, width = rgb.shape[:2]
    x0 = max(0, min(width - 1, x0 - pad))
    y0 = max(0, min(height - 1, y0 - pad))
    x1 = max(x0 + 1, min(width, x1 + pad))
    y1 = max(y0 + 1, min(height, y1 + pad))
    return rgb[y0:y1, x0:x1]
