"""The canonical scene record, and the rules that make it re-derivable.

The authority for a V2 render is this file's ``scene.json``, not a ``.blend``.
A ``.blend`` is a binary that records whatever the builder happened to do; a
canonical JSON record is something an independent reader can recompute from
V1's own constants and compare byte for byte.  So the writer here reads
:mod:`src.vision.synthetic` and :mod:`src.rendering.ldr` directly -- never a
transcribed copy of their numbers -- resolves every value, and writes the
result.  The Blender builder then consumes only this file and imports nothing
from the project, which is why no module of ours can hide inside Blender's
reach where an import-closure manifest would miss it.

Two scales live here and must not be confused.  ``STUD_PX`` is a *projection*
ratio: 44 pixels per stud, the scale V1's arithmetic renderer draws at and the
scale the recogniser measures pitch against.  ``LDU_STUD`` is a *physical*
size: 20 LDraw units per stud, the geometry the mesh is built from.  Treating
44 as a length, or 20 as a pixel count, is a modelling error rather than a
rounding one, so the schema keeps them in separate blocks and the camera block
records the derivation that connects them.

Floats are the other way this record could stop being reproducible.  Every
length is an integer count of LDU and every angle an integer count of
millidegrees, which is exact because V1's own positions are: a margin of 1.5
studs is 30 LDU, a gap of 1.0 is 20, and the two occlusion shifts of 0.55 and
1.15 studs are 11 and 23.  The few quantities that genuinely are not integers
-- colours, roughness, a focal length -- are stored as decimal *strings* in one
fixed format, so serialising the record twice cannot produce two different sets
of bytes and therefore cannot produce two different digests.
"""

from __future__ import annotations

import hashlib
import math
import json
import re

#: Bumped when the meaning of a field changes, not when a value does.
SCHEMA_VERSION = 1

KIND = "brickagain.pbr_scene"

#: Physical geometry, read from the module the LDraw golden vectors pin.
LDU_PER_SCENE_UNIT = 1

#: The two camera kinds this track renders. ``ortho`` is the one that pairs
#: with V1 -- same projection, so the contrast against V1 isolates the
#: rendering domain rather than mixing a projection change into it.
CAMERA_KINDS = ("ortho", "perspective")

#: Canonical decimal strings: exactly six places, no exponent, no leading
#: plus, no leading zeros beyond a single ``0``, and no negative zero. The
#: format is narrow on purpose. A validator that accepts several spellings of
#: the same number accepts several digests for the same scene.
_DECIMAL = re.compile(r"^-?(0|[1-9][0-9]*)\.[0-9]{6}$")

#: Prepended to the pixel bytes before hashing, so an array reshaped to the
#: same byte count cannot collide with the original.
PIXEL_HEADER_MAGIC = b"brickagain.pbr_pixels\x00"


class ContractError(ValueError):
    """A scene record that cannot be used, refused by name."""


# ---------------------------------------------------------------------------
# canonical decimal strings
# ---------------------------------------------------------------------------

def decimal_str(value: float) -> str:
    """``value`` in the one spelling this schema accepts.

    Negative zero is written as ``"0.000000"``: ``-0.0`` and ``0.0`` are the
    same number, and letting both spellings through would let one scene have
    two digests.
    """
    text = f"{float(value):.6f}"
    if text == "-0.000000":
        return "0.000000"
    return text


def is_canonical_decimal(text) -> bool:
    """Whether ``text`` is the one accepted spelling.

    The regex alone lets ``"-0.000000"`` through, which is the same number as
    ``"0.000000"`` written two ways -- so it is rejected here rather than in
    the pattern, where a reader could miss that the pattern is deliberately
    incomplete.
    """
    if not isinstance(text, str) or not _DECIMAL.match(text):
        return False
    return not (text.startswith("-") and set(text) <= set("-0."))


def parse_decimal(text) -> float:
    if not is_canonical_decimal(text):
        raise ContractError(
            f"{text!r} is not a canonical decimal string; the format is "
            "exactly six decimal places, no exponent, no leading plus or "
            "zero, and no negative zero")
    return float(text)


# ---------------------------------------------------------------------------
# canonical serialisation
# ---------------------------------------------------------------------------

def canonical_bytes(body: dict) -> bytes:
    """The one byte sequence that stands for ``body``.

    ``sort_keys`` removes insertion order, the compact separators remove
    whitespace, and ``ensure_ascii=False`` with UTF-8 removes the escaping
    choice. Two processes that build the same record produce the same bytes.
    """
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def scene_digest(body: dict) -> str:
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def pixel_header(height: int, width: int, channels: int) -> bytes:
    return (PIXEL_HEADER_MAGIC
            + int(height).to_bytes(4, "little")
            + int(width).to_bytes(4, "little")
            + int(channels).to_bytes(4, "little")
            + b"float32-le\x00")


def canonical_pixels(array) -> bytes:
    """Header plus the float32 little-endian bytes of ``array``.

    The digest is taken over *decoded* pixels rather than over the EXR
    container. OpenEXR's ZIP is lossless but its compressed bytes move with
    the library version, so a container digest would fail on a Blender upgrade
    that changed nothing about the image.
    """
    import numpy as np

    if array.ndim != 3:
        raise ContractError(
            f"pixels are (H, W, C); this array has {array.ndim} dimensions")
    body = np.ascontiguousarray(array.astype("<f4"))
    height, width, channels = body.shape
    return pixel_header(height, width, channels) + body.tobytes()


def pixel_digest(array) -> str:
    return hashlib.sha256(canonical_pixels(array)).hexdigest()


# ---------------------------------------------------------------------------
# the writer: V1's own constants, resolved
# ---------------------------------------------------------------------------

def _v1() -> tuple:
    """V1's renderer, its LDraw constants and the palette, imported not copied.

    Transcribing ``STUD_PX`` or a part footprint into this module would put the
    same number in two places, and the pairing against V1 is only meaningful
    while the two agree. So they are read, and a test compares what this
    writer exports against what those modules hold.
    """
    from src.colour import palette
    from src.rendering import ldr
    from src.vision import synthetic

    return synthetic, ldr, palette


def canvas_centre_ldu(width_px: int, height_px: int) -> tuple[str, str]:
    """Half the canvas, in LDU, as canonical decimal strings.

    One stud is 44 px and 20 LDU, so half a canvas of ``w`` pixels is
    ``w * 5 / 22`` LDU. That is a whole number only when the canvas is a
    multiple of 22 px, which V1's unoccluded canvases are and its occluded ones
    are not -- the two occlusion shifts move a brick by 0.55 and 1.15 studs and
    the canvas is rounded to whole pixels afterwards.

    So the centre is *not* subtracted from the bricks. The bricks stay at exact
    integer LDU in a frame whose origin is the canvas corner, and this offset
    is carried by the camera instead, where a single value rounded to six
    decimal places is wrong by at most 5e-7 LDU -- about 1e-6 px, three orders
    of magnitude under the tolerance the projection check runs at.
    """
    return (decimal_str(width_px * 5 / 22), decimal_str(height_px * 5 / 22))


#: One brick's height, in LDU. Not typed in: read from LDraw's own constant by
#: :func:`_brick_top_z` at import time, so a record that disagrees with
#: ``src/rendering/ldr.py`` cannot be written.
def _brick_top_z() -> int:
    from src.rendering import ldr
    return int(ldr.LDU_BRICK)


BRICK_TOP_Z = _brick_top_z()


def _ldu(studs: float, what: str) -> int:
    """``studs`` as whole LDU, or a refusal naming what was not exact."""
    exact = studs * 20
    rounded = round(exact)
    if abs(exact - rounded) > 1e-9:
        raise ContractError(
            f"{what} is {studs} studs = {exact} LDU, which is not a whole "
            "number; V1's own positions are all exact multiples of 1 LDU, so "
            "this is a layout this schema cannot record exactly")
    return int(rounded)


def brick_record(index: int, placement, x_studs: float, y_studs: float,
                 palette) -> dict:
    """One brick, with every derived position resolved to integer LDU.

    ``y`` is negated: V1's layout grows downward like a raster, and Blender's
    ``+Y`` under a top-down camera at ``rotation=(0,0,0)`` is image-*up*.
    Getting that sign wrong produces a scene that renders and is vertically
    mirrored, which is why the projection check compares directed vectors
    against absolute raster coordinates rather than distances.

    The origin here is the canvas corner, not its centre. Centring would put a
    non-integer offset into every brick for the occluded conditions; the camera
    carries it instead.
    """
    across, down = placement.extents()
    x0 = _ldu(x_studs, f"brick {index} x")
    y0 = -_ldu(y_studs + down, f"brick {index} y+down")
    width_ldu = across * 20
    depth_ldu = down * 20
    top = BRICK_TOP_Z

    studs = [[x0 + 10 + 20 * i, y0 + 10 + 20 * j, top]
             for j in range(down) for i in range(across)]
    corners = [[x0, y0, top], [x0 + width_ldu, y0, top],
               [x0 + width_ldu, y0 + depth_ldu, top], [x0, y0 + depth_ldu, top]]
    rgb = palette.colour(placement.colour_id).rgb
    return {
        "brick_id": f"b{index:02d}",
        "part": placement.part,
        "turn": int(placement.turn),
        "footprint_studs": [across, down],
        "grid_row_col": [int(placement.row), int(placement.col)],
        "translation_scene_units": [x0, y0, 0],
        "extent_scene_units": [width_ldu, depth_ldu, top],
        "rotation_z_millideg": 0,
        "height_ldu": top,
        "colour_id": placement.colour_id,
        "colour_srgb_uint8": [int(v) for v in rgb],
        "colour_linear": [decimal_str(srgb_to_linear_scalar(v)) for v in rgb],
        "stud_positions_scene_units": studs,
        "top_face_corners_scene_units": corners,
    }


def srgb_to_linear_scalar(u8: int) -> float:
    """One 8-bit sRGB channel as scene-linear.

    The piecewise standard transfer function, not a gamma 2.2 approximation:
    the approximation is wrong by enough near black to move a dark brick's
    colour, and this value ends up in the material.
    """
    c = u8 / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb_scalar(value: float) -> float:
    return (12.92 * value if value <= 0.0031308
            else 1.055 * value ** (1 / 2.4) - 0.055)


#: Pinned render and material values. They are here rather than in the builder
#: because the builder must be able to reproduce a scene from the record alone,
#: and a value the builder supplies from its own source is a value the record
#: does not pin.
PINNED_MESH = {
    "stud_diameter_ldu": 12,
    "stud_height_ldu": 4,
    "cylinder_segments": 32,
    "bevel_width_ldu": 1,
    "bevel_segments": 2,
    "shade_smooth": False,
    "underside_recess": False,
}

PINNED_MATERIAL = {
    "node": "ShaderNodeBsdfPrincipled",
    "metallic": "0.000000",
    "roughness": "0.350000",
    "ior": "1.530000",
    "transmission_weight": "0.000000",
    "subsurface_weight": "0.000000",
    "coat_weight": "0.000000",
    "sheen_weight": "0.000000",
    "emission_strength": "0.000000",
    "alpha": "1.000000",
    # Blender's own default is 0.5 and this is pinned *to* that default: a
    # value the builder does not set is a value Blender chooses, and "the
    # default today" is not a record. On a brick this term is the PBR effect
    # the contrast is for; the table overrides it to 0 (see
    # CONDITION_BACKGROUND).
    "specular_ior_level": "0.500000",
}

#: The even base. Colour 0.5 at strength 2.0 is a uniform dome of radiance
#: 1.0, under which the table's albedo of 0.921582 renders at 0.9236 linear --
#: 246 in 8-bit sRGB, which is exactly the grey V1's ``plain`` background is
#: drawn at. Measured, not assumed: the four corners of the frame agree to
#: within a ratio of 1.013 on the calibration cell.
#:
#: The uniformity and clip figures quoted for this dome are measured on the
#: calibration cell and are *not* corpus figures: with 3 to 10 bricks standing
#: on it the table's own ambient occlusion takes the corner-to-corner ratio to
#: 1.04-1.12, and the baseline clips 0.5 per cent of components -- almost all
#: of them on white top faces, whose albedo of 1.0 lands on the clip point
#: under any exposure that puts the table on V1's 246. V1 draws white at 255
#: too, so both arms are at the ceiling there; what V2 loses is the shading
#: detail inside the white brick, and the per-cell clip counts travel with
#: every image so that loss is visible rather than averaged away.
#:
#: This is what makes a *world-only* baseline the faithful analogue of V1's
#: ``lighting=even, shadow=none``. A close area light was tried first and was
#: wrong twice over: it blew out the frame, and it laid a strong gradient
#: across a condition whose whole point is that the illumination is flat.
PINNED_WORLD = {
    "colour_linear": ["0.500000", "0.500000", "0.500000"],
    "strength": "2.000000",
    "hdri": None,
}

#: Read from Blender 5.2.1 rather than assumed. ``sampling_pattern`` defaults
#: to ``AUTOMATIC``, whose meaning is free to change between versions, so the
#: named pattern is pinned instead. ``film_transparent`` moved from
#: ``scene.cycles`` to ``scene.render`` in 5.x and is recorded where it now
#: lives.
PINNED_RENDER = {
    "engine": "CYCLES",
    "device": "CPU",
    "samples": 256,
    "use_adaptive_sampling": False,
    "use_denoising": False,
    "time_limit": "0.000000",
    "seed": 0,
    "use_animated_seed": False,
    "sampling_pattern": "TABULATED_SOBOL",
    "use_light_tree": True,
    "max_bounces": 8,
    "diffuse_bounces": 4,
    "glossy_bounces": 4,
    "transmission_bounces": 8,
    "volume_bounces": 0,
    "transparent_max_bounces": 8,
    "sample_clamp_direct": "0.000000",
    "sample_clamp_indirect": "10.000000",
    "caustics_reflective": False,
    "caustics_refractive": False,
    "pixel_filter_type": "BLACKMAN_HARRIS",
    "filter_width": "1.500000",
    "film_transparent": False,
    "use_motion_blur": False,
    "use_dof": False,
    # Read out of Blender 5.2.1 and pinned because each of these changes the
    # pixels and none of them was being set. ``film_exposure`` is a direct
    # per-pixel multiplier; ``light_sampling_threshold`` and ``blur_glossy``
    # change what the sampler does; ``sample_offset`` and
    # ``scrambling_distance`` change which samples are taken, and so the noise
    # at a fixed seed. Pinning a value to its current default is not a no-op:
    # it is the difference between a record and a hope.
    "film_exposure": "1.000000",
    "light_sampling_threshold": "0.010000",
    "blur_glossy": "1.000000",
    "min_light_bounces": 0,
    "min_transparent_bounces": 0,
    "use_fast_gi": False,
    "sample_offset": 0,
    "scrambling_distance": "1.000000",
}

#: The dome's own sampling. ``AUTOMATIC`` is refused for the same reason
#: ``sampling_pattern`` is: its meaning is free to move between versions. For a
#: *uniform* dome there is no structure to importance-sample, so ``MANUAL`` at a
#: stated resolution is both deterministic and equivalent.
PINNED_WORLD_SAMPLING = {
    "sampling_method": "MANUAL",
    "sample_map_resolution": 1024,
    "max_bounces": 1024,
    "use_shadows": True,
}

#: The area light's own sampling and shape terms. ``normalize`` matters most:
#: it decides whether ``power_watts`` is per-lamp or per-unit-area, and the
#: soft/hard pair differs by a factor of 100 in area, so an unpinned
#: ``normalize`` would silently change what the pair means.
PINNED_LIGHT_SAMPLING = {
    "spread_degrees": "180.000000",
    "normalize": True,
    "use_multiple_importance_sampling": True,
    "max_bounces": 1024,
    "is_portal": False,
}

#: The EXR is written without a view transform so it holds scene-linear data.
#: Blender 5.2 defaults ``view_transform`` to ``AgX``; leaving it there would
#: bake a tone map into the file and make the pixels a function of the Blender
#: version rather than of the scene.
PINNED_OUTPUT = {
    "file_format": "OPEN_EXR",
    "color_depth": "16",
    "exr_codec": "ZIP",
    "color_mode": "RGBA",
    "scene_view_transform": "Standard",
    "scene_look": "None",
    "scene_exposure": "0.000000",
    "scene_gamma": "1.000000",
    "image_color_management": "OVERRIDE",
    "image_view_transform": "Raw",
    "image_look": "None",
    "image_exposure": "0.000000",
    "image_gamma": "1.000000",
    "display_device": "sRGB",
    # Under ``image_color_management = OVERRIDE`` it is the *image* settings,
    # not the scene's, that govern the written file. The scene-level device was
    # being pinned and read back while the one that applies was neither.
    "image_display_device": "sRGB",
}

PINNED_PNG = {
    "exposure_gain": "1.000000",
    "drop_alpha": True,
    "clip_range": ["0.000000", "1.000000"],
    "transfer": "srgb_piecewise_oetf",
    "scale": 255,
    "rounding": "half_to_even",
    "dtype": "uint8",
}


def camera_records(width_px: int, height_px: int,
                   offset: tuple[str, str]) -> dict:
    """Both cameras, with the derivation from 44 px/stud resolved.

    ``sensor_fit`` is stated rather than left at ``AUTO``: ``AUTO`` picks the
    horizontal or vertical axis by whichever output dimension is larger, and
    V1's canvases include both landscape (1012x748) and portrait (660x1100),
    so the same formula would silently produce the wrong scale for half of
    them.

    The perspective distance follows from wanting 44 px/stud *at the brick top
    plane*, not at the table. With ``sensor_width == lens`` the horizontal
    field spans exactly the camera distance, so the distance to the registered
    plane is the canvas width in scene units and the camera sits 24 units
    further back than that -- one brick height.

    Registering the top plane rather than the table is what lets one scorer
    measure both cameras. V1's truth boxes are the brick footprints at 44
    px/stud; every brick in V1's scenes is one layer, so every top face is at
    z=24, and a pinhole images a single plane as a pure uniform scale about the
    optical axis. Registering that plane therefore puts every brick top exactly
    where V1's own truth box is, under both cameras. The earlier design
    registered the table instead, which magnified every brick top by 5.5 to
    12.2 per cent against a truth box that had not moved: measured over the
    eight perspective cells that gave a mean IoU of 0.513 between the projected
    top face and V1's box, with 7 of 52 bricks below the 0.3 matching
    threshold. The arm was measuring its own registration error.

    What the perspective camera still changes -- and what the arm measures --
    is that the side faces become visible and the table plane is *minified* by
    ``span / (span + 24)``, 4.96 to 9.84 per cent across the corpus.
    """
    # The canvas width in scene units. NOT integer division: 44 px/stud and 20
    # LDU/stud make this ``width_px * 5 / 11``, which is whole only when the
    # canvas is a multiple of 11 px. Every unoccluded canvas is; the occluded
    # ones are not, and truncating there put a 0.32 per cent scale error into
    # ``ortho_scale`` -- about 1 px at the frame edge. ``project`` does not read
    # ``ortho_scale``, so only Blender's own projection disagreed, and only the
    # whole-corpus Stage B check saw it.
    span_value = width_px * 5 / 11
    span = decimal_str(span_value)
    #: The perspective camera stands one brick height further back so that the
    #: plane it registers is the brick top at z=24 rather than the table.
    persp_distance = span_value + BRICK_TOP_Z
    cx, cy = float(offset[0]), float(offset[1])
    shared = {
        "sensor_fit": "HORIZONTAL",
        "shift_x": "0.000000",
        "shift_y": "0.000000",
        "clip_start": "1.000000",
        "clip_end": "2000.000000",
        "canvas_offset_scene_units": [offset[0], offset[1]],
        "rotation_euler_millideg": [0, 0, 0],
        "use_dof": False,
    }
    table_scale = span_value / persp_distance
    return {
        "ortho": {**shared, "type": "ORTHO", "ortho_scale": span,
                  "location_scene_units": [offset[0], decimal_str(-cy), span],
                  "reference_plane_z": 0,
                  "px_per_stud_at_reference_plane": 44,
                  "px_per_stud_everywhere": 44},
        "perspective": {
            **shared, "type": "PERSP",
            "location_scene_units": [offset[0], decimal_str(-cy),
                                     decimal_str(persp_distance)],
            "sensor_width_mm": "36.000000", "lens_mm": "36.000000",
            "horizontal_fov_millideg": 53130,
            "reference_plane_z": BRICK_TOP_Z,
            "px_per_stud_at_reference_plane": 44,
            "note_depth_scaling": (
                f"44 px/stud holds at z={BRICK_TOP_Z}, the brick top plane, "
                "which is where V1's truth boxes are. A plane at z scales by "
                "(d - z_ref)/(d - z) with d="
                f"{decimal_str(persp_distance)}; the table at z=0 therefore "
                f"renders at {decimal_str(44 * table_scale)} px/stud, "
                f"{decimal_str(100 * (1 - table_scale))} per cent smaller "
                "than under the orthographic camera. That minification and "
                "the newly visible side faces are the perspective-camera "
                "effect and are not corrected out."),
        },
    }


#: Illumination per condition, calibrated by measurement rather than chosen.
#:
#: Blender's watts are tied to the scene unit scale and one unit here is one
#: LDU -- 0.4 mm -- so a scene 184 mm across is 460 units across and a light
#: 3000 units up is, to Cycles' falloff, 3 km away. The powers are therefore
#: enormous; that is a unit consequence.
#:
#: Every row is rendered and measured by ``--mode calibrate``, which archives
#: the images and the derivation and whose ``calibration_problems`` refuses if
#: any recorded figure has moved. The light powers are *solved*, not tuned: the
#: table is Lambertian and lit by a dome plus one area light, so its rendered
#: radiance is linear in each, and two renders -- power 0 and a probe -- give
#: the power that lands the table on V1's own grey.
#:
#: The measured figures are named for the cell they were measured on, because
#: scene_00 is not the corpus: it holds red, blue and yellow bricks and no
#: white one, so its clip rate is 0 while the corpus baseline clips 0.5 per
#: cent almost entirely on white top faces. A clip figure measured here cannot
#: stand for the corpus and no longer claims to.
#:
#: The first
#: attempt put a directional light into every non-baseline condition at one
#: fixed power and clipped **83.8 per cent** of ``lighting=dim`` and **87.8 per
#: cent** of ``shadow=soft`` -- so those two conditions were measuring
#: over-exposure, and the colour errors they produced (47 and 38 of 154) were
#: a clipping artefact. Adding a light without taking the world down cannot
#: work: the world alone is already calibrated to full scale.
#:
#: ``world`` is the uniform dome's strength at colour 0.5; the *plain* table's
#: albedo of 0.921582 under it renders at ``0.4618 * strength``. The two
#: measured fields are that plain table on the calibration scene -- one brick
#: on an empty table -- so they are the illumination's own figures and not a
#: prediction for a different background or a fuller scene. The real corpus is
#: measured per cell and its clip counts travel with each image.
CONDITION_LIGHTING = {
    # V1's baseline is lighting=even, shadow=none -- flat, no cast shadow. The
    # dome alone is the analogue; a directional light would add a gradient and
    # a shadow to a condition that has neither.
    "even/none": {"world": "2.000000", "light": None,
                  "measured_table_srgb_on_scene_00": [244, 243, 242],
                  "measured_clip_percent_on_scene_00": "0.000000",
                  "measured_corner_uniformity_on_scene_00": "1.013000"},
    # V1 dims by multiplying its 8-bit image by 0.55, so its background lands
    # at 246 * 0.55 = 135. Matching that grey is what makes the pair
    # comparable in the quantity the segmenter thresholds on. The mechanisms
    # still differ -- a gamma-space multiply is not a linear-space dimming --
    # and that difference is part of what the contrast measures.
    "dim/none": {"world": "0.525000", "light": None,
                 "measured_table_srgb_on_scene_00": [134, 133, 133],
                 "measured_clip_percent_on_scene_00": "0.000000",
                 "measured_corner_uniformity_on_scene_00": "1.013000"},
    # A cast shadow needs a directional source, and the world comes down to
    # make room for it. Soft: a large source. Hard: a small one, whose power is
    # solved to the same table grey. What changes is not only the shadow edge:
    # measured on the calibration cell the corner-to-corner uniformity goes
    # 1.013 (baseline) -> 1.029 (soft) -> 1.041 (hard), so the frame gets less
    # even as the source gets smaller. That is inherent to a directional
    # source and is stated rather than claimed away.
    "even/soft": {"world": "1.400000",
                  "light": {"size_scene_units": 2000, "power_watts": "47407660.000000"},
                  "measured_table_srgb_on_scene_00": [247, 246, 245],
                  "measured_clip_percent_on_scene_00": "0.313674",
                  "measured_corner_uniformity_on_scene_00": "1.028800"},
    "even/hard": {"world": "1.200000",
                  "light": {"size_scene_units": 200, "power_watts": "55440008.000000"},
                  "measured_table_srgb_on_scene_00": [247, 246, 245],
                  "measured_clip_percent_on_scene_00": "1.403080",
                  "measured_corner_uniformity_on_scene_00": "1.040900"},
}

#: The table colour each ``background`` value renders at, taken from V1's own
#: fills so the pair differs in the renderer and not in the grey.
#:
#: These are rendered values, and holding them requires the table to be a
#: *Lambertian* backdrop -- ``specular_ior_level`` 0 -- rather than a full
#: Principled dielectric. Under a uniform dome of radiance 1.0 a dielectric
#: surface reflects a near-constant specular floor of about 0.0246 linear on
#: top of its albedo, so writing V1's fill in as the albedo renders it as
#: ``albedo * (1 - F) + F``. Measured on the earlier corpus that put the
#: *plain* table at 244-245 (V1: 246) and *grey* at 169 (V1: 168), which is
#: within a code or two, and *dark* at **59** against V1's **38** -- 55 per
#: cent brighter. The ``background`` axis was then differing in the renderer
#: *and* in the grey, which is the one thing a paired condition may not do.
#: The floor cannot be cancelled by lowering the albedo either: V1's ``dark``
#: is linear 0.0185, below the floor itself, so no albedo renders there.
#:
#: The bricks keep the full Principled material. The specular floor on a brick
#: is a PBR effect the contrast is *for*; on the backdrop it is a confound.
CONDITION_BACKGROUND = {"plain": 246, "grey": 168, "dark": 38}

LIGHT_BASE = {
    "type": "AREA",
    "shape": "SQUARE",
    "location_scene_units": [1200, 1200, 3000],
    "aims_at_scene_centre": True,
    "colour_linear": ["1.000000", "1.000000", "1.000000"],
    "use_shadow": True,
    **PINNED_LIGHT_SAMPLING,
}


def lighting_key(conditions: dict) -> str:
    return f"{conditions['lighting']}/{conditions['shadow']}"


def illumination(conditions: dict) -> dict:
    """The calibrated world strength and light for this condition, or a refusal.

    Refused rather than defaulted: a condition whose illumination has not been
    measured would render, and would be reported beside conditions that were
    calibrated, with nothing saying which was which.
    """
    key = lighting_key(conditions)
    if key not in CONDITION_LIGHTING:
        raise ContractError(
            f"lighting/shadow combination {key!r} has no calibrated "
            f"illumination; the calibrated set is "
            f"{sorted(CONDITION_LIGHTING)}. A condition is absent until it "
            "has been rendered and measured")
    return CONDITION_LIGHTING[key]


def lights_for(conditions: dict) -> list:
    spec = illumination(conditions)["light"]
    return [] if spec is None else [{**LIGHT_BASE, **spec}]


def world_for(conditions: dict) -> dict:
    return {**PINNED_WORLD, **PINNED_WORLD_SAMPLING,
            "strength": illumination(conditions)["world"]}


def table_for(conditions: dict, extent: int) -> dict:
    background = conditions["background"]
    if background not in CONDITION_BACKGROUND:
        raise ContractError(
            f"background {background!r} is not one of "
            f"{sorted(CONDITION_BACKGROUND)}; a textured or cluttered table "
            "is declared absent rather than approximated")
    value = CONDITION_BACKGROUND[background]
    return {
        "plane_z": 0,
        "edge_scene_units": 4 * extent,
        "colour_srgb_uint8": [value, value, value],
        "colour_linear": [decimal_str(srgb_to_linear_scalar(value))] * 3,
        "roughness": "0.800000",
        "metallic": "0.000000",
        # Lambertian: see CONDITION_BACKGROUND. Without this the dome's
        # specular reflection lifts `dark` from 38 to 59.
        "specular_ior_level": "0.000000",
    }


def build_scene(scene_index: int, condition_id: str, *,
                scene_seed: int | None = None) -> dict:
    """The canonical record for one V1 scene under one V1 condition."""
    synthetic, ldr, palette = _v1()
    from src.training.session import sha256_file

    from pathlib import Path
    root = Path(__file__).resolve().parents[3]

    # ``seed`` is keyword-only on ``scenes``, so its default lives in
    # ``__kwdefaults__``; ``__defaults__`` holds the positional ``n`` and
    # reading it here silently built a different set of scenes.
    seed = (synthetic.scenes.__kwdefaults__["seed"] if scene_seed is None
            else scene_seed)
    scenes = synthetic.scenes(seed=seed)
    if not 0 <= scene_index < len(scenes):
        raise ContractError(
            f"scene index {scene_index} is outside 0..{len(scenes) - 1}")
    scene = scenes[scene_index]
    conditions = _condition(synthetic, condition_id)
    width_px, height_px = synthetic.canvas_size(scene, conditions)
    offset = canvas_centre_ldu(width_px, height_px)
    cells = synthetic.layout(scene, conditions)

    bricks = [brick_record(i, p, x, y, palette)
              for i, (p, x, y) in enumerate(cells)]
    extent = int(round(max(width_px, height_px) * 5 / 11))

    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "provenance": {
            "scene_id": scene.scene_id,
            "scene_index": scene_index,
            "scene_seed": int(seed),
            "condition_id": condition_id,
            "condition": dict(sorted(conditions.items())),
            "v1_source_digests": {
                rel: sha256_file(root / rel) for rel in (
                    "src/vision/synthetic.py",
                    "src/rendering/ldr.py",
                    "src/colour/palette.py",
                )},
            "writer_source_digest": sha256_file(
                root / "src/vision/pbr/contract.py"),
        },
        "units": {
            "ldu_per_scene_unit": LDU_PER_SCENE_UNIT,
            "mm_per_ldu": "0.400000",
            "ldu_stud": int(ldr.LDU_STUD),
            "ldu_brick": int(ldr.LDU_BRICK),
        },
        "raster": {
            "width_px": width_px,
            "height_px": height_px,
            "target_px_per_stud": int(synthetic.STUD_PX),
            "pixel_center_convention": "half_integer",
            "row_zero": "top",
            "axis_map": {"scene_plus_x": "raster_plus_u",
                         "scene_plus_y": "raster_minus_v",
                         "scene_plus_z": "toward_camera"},
        },
        "scene_centre_scene_units": [0, 0, 0],
        "canvas_offset_scene_units": list(offset),
        "bricks": bricks,
        "table": table_for(conditions, extent),
        "world": world_for(conditions),
        "lights": lights_for(conditions),
        "illumination_calibration": illumination(conditions),
        "mesh": dict(PINNED_MESH),
        "material": dict(PINNED_MATERIAL),
        "cameras": camera_records(width_px, height_px, offset),
        "render": dict(PINNED_RENDER),
        "output": dict(PINNED_OUTPUT),
        "png": dict(PINNED_PNG),
    }
    return body


#: Axes V1 varies that this writer does not. They are refused by name rather
#: than accepted and ignored. ``background`` and ``lighting``/``shadow`` have
#: their own refusals at the point of use; these two have no point of use at
#: all, which is exactly why the silent-acceptance had to be closed here.
UNWRITTEN_AXES = {
    "viewpoint": ("a tilted camera needs its own registration and its own "
                  "ground truth, and neither the record nor the projection "
                  "carries a tilt: rotation_euler_millideg is [0,0,0] and "
                  "the layout is read from V1 unrotated. The perspective "
                  "*camera* arm is a different question and is implemented"),
    "quality": ("downscaling, noise, blur and JPEG are post-process steps V1 "
                "applies to its 8-bit image. V2 renders scene-linear EXR and "
                "converts once, under a pinned transfer function; a "
                "post-process on top of that is a second pipeline this "
                "generation does not pin. render_sampling_noise is a "
                "*render* noise condition and is not this axis"),
}


def _condition(synthetic, condition_id: str) -> dict:
    """V1's own condition dictionary for ``condition_id``."""
    if condition_id == "baseline":
        return dict(synthetic.BASELINE)
    axis, _, value = condition_id.partition("=")
    if not value or axis not in synthetic.CONDITIONS:
        raise ContractError(
            f"condition {condition_id!r} is not 'baseline' or one of "
            f"{sorted(synthetic.CONDITIONS)}=value")
    if value not in synthetic.CONDITIONS[axis]:
        raise ContractError(
            f"{value!r} is not one of {list(synthetic.CONDITIONS[axis])} "
            f"for axis {axis!r}")
    if axis in UNWRITTEN_AXES and value != synthetic.BASELINE[axis]:
        raise ContractError(
            f"axis {axis!r} is not implemented by this writer: {UNWRITTEN_AXES[axis]}. "
            f"{condition_id!r} is refused rather than rendered, because "
            "nothing downstream reads that axis and the record it would "
            "produce is byte-identical to baseline apart from its own "
            "provenance -- so it would render, score and be reported beside "
            "conditions that really were varied, with nothing saying which "
            "was which")
    return {**synthetic.BASELINE, axis: value}


# ---------------------------------------------------------------------------
# the record against the scene that rendered, in both directions
# ---------------------------------------------------------------------------

#: How far a read-back value may sit from the value the record asked for.
#: Blender stores most of these as 32-bit floats, so a decimal string written
#: at six places comes back a few 1e-7 away; anything larger is a value that
#: did not take.
RECEIPT_TOLERANCE = 1e-4

#: Degrees a light's measured aim may sit from the direction it declares.
#:
#: Not arbitrarily small. Blender stores the rotation as 32-bit floats and the
#: aim is recovered through ``acos`` of a dot product near 1, which amplifies a
#: relative error of eps into an angle of sqrt(2*eps): the float32 round trip
#: alone measures 0.0099 degrees. This is set well above that noise floor and
#: three orders of magnitude below the defect it exists to catch, which was
#: 58.992 degrees.
AIM_TOLERANCE_DEGREES = 0.05


def _allowance(want: float, tol: float) -> float:
    """How far a float32 round trip can move ``want``.

    An absolute tolerance alone is wrong at both ends of the range. Blender
    stores these as 32-bit floats, whose spacing at 4.7e7 -- the calibrated
    light powers -- is 4.0, so an exact-looking integer comes back one or two
    away for reasons that have nothing to do with whether the value took. One
    ULP is added to the absolute floor rather than widening the floor, so a
    roughness of 0.35 is still held to 1e-4.
    """
    return tol + abs(want) * 2.0 ** -23


def _near(want, got, tol: float = RECEIPT_TOLERANCE) -> bool:
    """Compare a record value against a read-back one, by kind.

    Decimal strings compare numerically against floats: the record's canonical
    form is a string and Blender's is a float, and comparing their reprs would
    reject every value that took correctly.
    """
    if isinstance(want, str) and is_canonical_decimal(want):
        try:
            value = parse_decimal(want)
            return abs(value - float(got)) <= _allowance(value, tol)
        except (TypeError, ValueError):
            return False
    if isinstance(want, bool) or isinstance(got, bool):
        return bool(want) is bool(got)
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        return abs(float(want) - float(got)) <= _allowance(float(want), tol)
    if isinstance(want, (list, tuple)) and isinstance(got, (list, tuple)):
        return (len(want) == len(got)
                and all(_near(a, b, tol) for a, b in zip(want, got)))
    return want == got


def receipt_problems(body: dict, receipt: dict) -> list[str]:
    """Everywhere the scene Blender built disagrees with the record.

    This is the check the receipt existed for and did not have. Six
    generations wrote a receipt carrying every resolved light rotation and
    nothing read it, and the whole time every shadow-condition light was
    aimed 58.992 degrees away from the scene centre while the record said
    ``aims_at_scene_centre: true``. A field written and never compared is
    documentation, not evidence, so this function is deliberately exhaustive
    over the blocks the record pins: a value the record states and this
    function does not read is a value nothing checks.

    Returns a list of human-readable problems. Empty means the scene graph
    that rendered matches the record that asked for it, field by field.
    """
    out: list[str] = []
    resolved = receipt.get("resolved")
    if not isinstance(resolved, dict):
        out.append("receipt has no 'resolved' block")
        return out

    def check(where: str, want, got):
        if not _near(want, got):
            out.append(f"{where}: record says {want!r}, Blender resolved {got!r}")

    for key, want in sorted(body["render"].items()):
        if key in ("engine", "device", "use_dof"):
            continue
        if key not in resolved:
            out.append(f"render.{key} is pinned by the record and absent from "
                       "the receipt, so nothing checks it")
            continue
        check(f"render.{key}", want, resolved[key])
    check("render.engine", body["render"]["engine"], resolved.get("engine"))

    for key, want in sorted(body["output"].items()):
        alias = {"scene_view_transform": "scene_view_transform",
                 "image_view_transform": "image_view_transform"}.get(key, key)
        if alias not in resolved:
            out.append(f"output.{key} is pinned by the record and absent from "
                       "the receipt, so nothing checks it")
            continue
        check(f"output.{key}", want, resolved[alias])

    check("raster.resolution",
          [body["raster"]["width_px"], body["raster"]["height_px"]],
          resolved.get("resolution"))
    check("raster.resolution_percentage", 100,
          resolved.get("resolution_percentage"))

    kind = receipt.get("camera_kind")
    spec = body["cameras"].get(kind)
    cam = resolved.get("camera") or {}
    if spec is None:
        out.append(f"receipt names camera {kind!r}, which the record does not "
                   f"carry: {sorted(body['cameras'])}")
    else:
        check("camera.type", spec["type"], cam.get("type"))
        check("camera.sensor_fit", spec["sensor_fit"], cam.get("sensor_fit"))
        check("camera.shift", [spec["shift_x"], spec["shift_y"]],
              cam.get("shift"))
        check("camera.clip", [spec["clip_start"], spec["clip_end"]],
              cam.get("clip"))
        check("camera.location", spec["location_scene_units"],
              cam.get("location"))
        check("camera.use_dof", spec["use_dof"], cam.get("use_dof"))
        if spec["type"] == "ORTHO":
            check("camera.ortho_scale", spec["ortho_scale"],
                  cam.get("ortho_scale"))
        else:
            check("camera.lens", spec["lens_mm"], cam.get("lens"))
            check("camera.sensor_width", spec["sensor_width_mm"],
                  cam.get("sensor_width"))
        want_rot = [v / 1000.0 for v in spec["rotation_euler_millideg"]]
        got_rot = [math.degrees(v) for v in (cam.get("rotation_euler") or [])]
        check("camera.rotation_euler_degrees", want_rot, got_rot)

    table, got_table = body["table"], resolved.get("table") or {}
    check("table.colour_linear", table["colour_linear"],
          got_table.get("base_color_linear"))
    check("table.edge_scene_units", table["edge_scene_units"],
          got_table.get("edge_scene_units"))
    material = got_table.get("material") or {}
    for key, want in sorted({**body["material"], **{
            k: table[k] for k in ("roughness", "metallic",
                                  "specular_ior_level")}}.items()):
        if key == "node":
            continue
        if key not in material:
            out.append(f"table.material.{key} is pinned and absent from the "
                       "receipt")
            continue
        check(f"table.material.{key}", want, material[key])

    got_bricks = receipt.get("bricks") or []
    if len(got_bricks) != len(body["bricks"]):
        out.append(f"the record has {len(body['bricks'])} bricks and the "
                   f"receipt {len(got_bricks)}")
    else:
        for want_brick, got_brick in zip(body["bricks"], got_bricks):
            name = want_brick["brick_id"]
            check(f"brick {name}.id", name, got_brick.get("brick_id"))
            check(f"brick {name}.declared_id", name,
                  got_brick.get("declared_brick_id"))
            check(f"brick {name}.colour_linear", want_brick["colour_linear"],
                  got_brick.get("base_color_linear"))
            got_material = got_brick.get("material") or {}
            for key, want in sorted(body["material"].items()):
                if key == "node":
                    continue
                if key not in got_material:
                    out.append(f"brick {name}.material.{key} is pinned and "
                               "absent from the receipt")
                    continue
                check(f"brick {name}.material.{key}", want, got_material[key])
            if not want_brick["footprint_studs"]:
                continue

    want_lights, got_lights = body["lights"], resolved.get("lights") or []
    if len(want_lights) != len(got_lights):
        out.append(f"the record asks for {len(want_lights)} lights and the "
                   f"receipt reports {len(got_lights)}")
    else:
        for i, (want_light, got_light) in enumerate(zip(want_lights, got_lights)):
            for key, field in (("type", "type"), ("shape", "shape"),
                               ("size_scene_units", "size"),
                               ("power_watts", "energy"),
                               ("colour_linear", "color"),
                               ("use_shadow", "use_shadow"),
                               ("spread_degrees", "spread_degrees"),
                               ("normalize", "normalize"),
                               ("use_multiple_importance_sampling",
                                "use_multiple_importance_sampling"),
                               ("max_bounces", "max_bounces"),
                               ("is_portal", "is_portal"),
                               ("location_scene_units", "location")):
                check(f"light{i}.{key}", want_light[key], got_light.get(field))
            if want_light["aims_at_scene_centre"]:
                residual = got_light.get("aim_residual_degrees")
                if residual is None:
                    out.append(f"light{i} declares it aims at the scene centre "
                               "and the receipt does not measure the aim")
                elif abs(float(residual)) > AIM_TOLERANCE_DEGREES:
                    out.append(
                        f"light{i} declares aims_at_scene_centre and the built "
                        f"rotation aims {float(residual):.6f} degrees away "
                        "from it")

    world, got_world = body["world"], resolved.get("world") or {}
    check("world.colour_linear", world["colour_linear"], got_world.get("colour"))
    for key in ("strength", "sampling_method", "sample_map_resolution",
                "max_bounces", "use_shadows"):
        if key not in got_world:
            out.append(f"world.{key} is pinned and absent from the receipt")
            continue
        check(f"world.{key}", world[key], got_world[key])

    if not receipt.get("geometry_digest"):
        out.append("the receipt carries no geometry digest")
    return out


def scene_digest_problems(receipt: dict, scene_bytes: bytes) -> list[str]:
    """Whether the receipt was built from these exact record bytes."""
    declared = receipt.get("scene_digest_declared")
    actual = hashlib.sha256(scene_bytes).hexdigest()
    if declared != actual:
        return [f"the receipt says it was built from scene bytes "
                f"{declared!r} and these bytes are {actual!r}"]
    return []
