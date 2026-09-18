"""Scene-linear EXR to the one 8-bit image the recogniser is allowed to see.

Blender 5.2 defaults its view transform to AgX, and 4.x used Filmic before it.
A PNG written by Blender is therefore a function of the Blender version as much
as of the scene: the same geometry, the same seed and the same light produce
systematically different pixels across a major release, and that difference is
tone mapping rather than rendering. So Blender writes scene-linear EXR with the
transform explicitly off, and the conversion to 8-bit sRGB happens here, where
every step is stated and pinned.

Two digests come out of this, and they answer different questions. The digest
of the *decoded* pixels is what a re-render is compared against: OpenEXR's ZIP
is lossless but its compressed bytes move with the library version, so a
container digest would fail on an upgrade that changed no pixel. The digest of
the *stored bytes* is what an archived file is checked against: once a file is
filed, any change to it at all is tampering, and there is nothing to be
tolerant about.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.vision.pbr.contract import ContractError, parse_decimal, pixel_digest


def read_exr(path) -> np.ndarray:
    """The EXR at ``path`` as ``(H, W, 4)`` float32, R G B A.

    Half floats are widened to float32, which is exact, so the array is the
    same numbers the renderer wrote and the canonical digest over it does not
    depend on the width the file happened to use.
    """
    import OpenEXR

    with OpenEXR.File(str(path)) as handle:
        channels = handle.channels()
        header = dict(handle.header())
        if "RGBA" in channels:
            pixels = channels["RGBA"].pixels
        elif all(k in channels for k in "RGBA"):
            pixels = np.stack([channels[k].pixels for k in "RGBA"], axis=-1)
        else:
            raise ContractError(
                f"{path} has channels {sorted(channels)}; this track writes "
                "RGBA and the reader will not guess a mapping")
    array = np.ascontiguousarray(np.asarray(pixels, dtype=np.float32))
    if array.ndim != 3 or array.shape[2] != 4:
        raise ContractError(
            f"{path} decoded to {array.shape}; expected (H, W, 4)")

    # The file stores its channels alphabetically -- A, B, G, R -- and the
    # grouped "RGBA" key hands them back in RGBA order. That reorder happens
    # inside a third-party library, so this asserts it rather than trusting it.
    # Alpha is uniformly 1.0 in every render this track produces and R is not,
    # which makes an all-ones last plane a free and exact test of the
    # permutation: any rotation of the channels puts colour data where this
    # check looks for ones.
    alpha = array[:, :, 3]
    if not np.all(alpha == 1.0):
        raise ContractError(
            f"{path}: the fourth plane is not uniformly 1.0 (min "
            f"{float(alpha.min())!r}, max {float(alpha.max())!r}). Either the "
            "render was written with a transparent film -- which this track "
            "pins off -- or the channels came back in an order other than "
            "R,G,B,A and the colour planes are being read as each other")

    # The file says what space it is in; the pipeline pins scene-linear and
    # this reads the file's own answer rather than assuming it.
    declared = str(header.get("colorInteropID", "") or "")
    if declared and "lin_" not in declared:
        raise ContractError(
            f"{path} declares colour space {declared!r}. This reader treats "
            "the pixels as scene-linear radiance, so a file carrying a "
            "display encoding would be silently converted twice: once by "
            "Blender on write and once by linear_to_srgb here")
    return array


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """The standard piecewise sRGB transfer function.

    Not ``x ** (1/2.2)``: the approximation is wrong by several 8-bit codes
    near black, which is exactly where a dark brick against a light table is
    decided.
    """
    low = values <= 0.0031308
    out = np.empty_like(values)
    out[low] = 12.92 * values[low]
    high = ~low
    out[high] = 1.055 * np.power(values[high], 1.0 / 2.4) - 0.055
    return out


def to_canonical_png(exr_path, png_path, png_spec: dict) -> dict:
    """Write the canonical PNG and report what the conversion had to do.

    The clip counts are returned rather than swallowed. A render whose highlight
    goes above 1.0 loses information here, and how much it lost is part of the
    record: a stress condition that clips half the studs is a condition whose
    result has an explanation.
    """
    array = read_exr(exr_path)
    non_finite = int((~np.isfinite(array)).sum())
    if non_finite:
        raise ContractError(
            f"{exr_path} holds {non_finite} non-finite pixel components. The "
            "render is treated as failed rather than clamped: NaN compares "
            "unequal to itself, so a clamped NaN would make every later "
            "comparison undefined")

    rgb = array[:, :, :3] if png_spec["drop_alpha"] else array
    gain = parse_decimal(png_spec["exposure_gain"])
    rgb = rgb * gain
    low, high = (parse_decimal(v) for v in png_spec["clip_range"])
    below = int((rgb < low).sum())
    above = int((rgb > high).sum())
    clipped = np.clip(rgb, low, high)

    if png_spec["transfer"] != "srgb_piecewise_oetf":
        raise ContractError(f"unknown transfer {png_spec['transfer']!r}")
    encoded = linear_to_srgb(clipped) * float(png_spec["scale"])
    if png_spec["rounding"] != "half_to_even":
        raise ContractError(f"unknown rounding {png_spec['rounding']!r}")
    # numpy's rint is half-to-even, which is the tie rule stated in the record.
    eight_bit = np.clip(np.rint(encoded), 0, 255).astype(np.uint8)

    from PIL import Image

    image = Image.fromarray(eight_bit, mode="RGB")
    target = Path(png_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # No ancillary chunks and no timestamp: a PNG that records when it was
    # written is a PNG whose bytes change when nothing else did.
    image.save(target, format="PNG", optimize=False, compress_level=6)

    stored = target.read_bytes()
    return {
        "exr_path": str(exr_path),
        "png_path": str(target),
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "linear_pixel_digest": pixel_digest(array),
        "png_bytes": len(stored),
        "png_sha256": hashlib.sha256(stored).hexdigest(),
        "uint8_digest": hashlib.sha256(
            b"brickagain.pbr_uint8\x00"
            + int(eight_bit.shape[0]).to_bytes(4, "little")
            + int(eight_bit.shape[1]).to_bytes(4, "little")
            + int(eight_bit.shape[2]).to_bytes(4, "little")
            + eight_bit.tobytes()).hexdigest(),
        "clipped_below": below,
        "clipped_above": above,
        "non_finite": non_finite,
        "linear_max": float(array[:, :, :3].max()),
        "linear_min": float(array[:, :, :3].min()),
    }
