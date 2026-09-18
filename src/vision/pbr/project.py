"""Where a scene point lands in the raster, with sign and offset.

The check this module exists for is not "is the scale about right". A scale
error, an axis swap, a sign flip and a half-canvas offset all produce images
that look plausible and pair wrongly against V1, so what is compared is the
*absolute* raster coordinate of named control points, both components, against
V1's own arithmetic. A projection that agrees on distances and disagrees on
direction fails here.

Nothing in this module touches Blender. It is the expectation; the builder's
own ``world_to_camera_view`` is the measurement, and the two are compared.
"""

from __future__ import annotations

from src.vision.pbr.contract import ContractError, parse_decimal


def px_per_ldu(body: dict) -> float:
    """Pixels per LDU: the projection ratio over the physical one.

    44 px/stud and 20 LDU/stud give 2.2 px/LDU. The two are kept apart in the
    record because one is a pixel count and the other a length.
    """
    return body["raster"]["target_px_per_stud"] / body["units"]["ldu_stud"]


def _centre(body: dict) -> tuple[float, float]:
    """The image centre in raster coordinates, half-integer convention."""
    return body["raster"]["width_px"] / 2.0, body["raster"]["height_px"] / 2.0


def depth_scale(body: dict, camera_kind: str, z: float) -> float:
    """How much a plane at ``z`` scales, relative to the registered plane.

    ``(d - z_ref) / (d - z)``, not ``z / d``: the scale of a plane under a
    pinhole is the ratio of camera distances. The reference is read from the
    camera record rather than assumed to be the table, because the perspective
    camera registers the brick top plane at ``z = 24`` -- that is what puts
    every brick top on V1's own truth box and lets one scorer measure both
    cameras. Assuming ``z_ref = 0`` here while the camera registers 24 would
    reintroduce the whole registration error in the *expectation* instead of
    in the render, which is worse: the check would then agree with the wrong
    picture.
    """
    camera = body["cameras"][camera_kind]
    z_ref = float(camera["reference_plane_z"])
    if camera_kind == "ortho":
        return 1.0
    d = parse_decimal(camera["location_scene_units"][2])
    if z >= d:
        raise ContractError(
            f"a point at z={z} is at or behind a camera at z={d}")
    return (d - z_ref) / (d - z)


def project(body: dict, camera_kind: str, point) -> tuple[float, float]:
    """``(u, v)`` for a scene point, in absolute raster pixels.

    The scene origin is the canvas *corner*, so the camera's canvas offset is
    subtracted here. That is where the offset lives for a reason: the occluded
    conditions have canvases that are not a multiple of 22 px, so a centred
    frame would put a non-terminating decimal into every brick position
    instead of into one camera value.
    """
    if camera_kind not in body["cameras"]:
        raise ContractError(
            f"camera {camera_kind!r} is not one of {sorted(body['cameras'])}")
    x, y, z = (float(v) for v in point)
    offset = body["cameras"][camera_kind]["canvas_offset_scene_units"]
    ox, oy = parse_decimal(offset[0]), parse_decimal(offset[1])
    cu, cv = _centre(body)
    ratio = px_per_ldu(body) * depth_scale(body, camera_kind, z)
    return cu + (x - ox) * ratio, cv - (y + oy) * ratio


def top_face_polygon(body: dict, camera_kind: str, brick: dict):
    """The brick's top face, projected, as four ``(u, v)`` corners."""
    return [project(body, camera_kind, c)
            for c in brick["top_face_corners_scene_units"]]


def top_face_aabb(body: dict, camera_kind: str, brick: dict
                  ) -> tuple[float, float, float, float]:
    """``(u0, v0, u1, v1)`` around the projected top face.

    An axis-aligned box because that is what ``Detection.box`` is; comparing a
    quadrilateral against a box would make the IoU depend on the comparison
    rather than on the detection.
    """
    poly = top_face_polygon(body, camera_kind, brick)
    us = [u for u, _ in poly]
    vs = [v for _, v in poly]
    return min(us), min(vs), max(us), max(vs)


def v1_footprint(placement, x_studs: float, y_studs: float, stud_px: int
                 ) -> tuple[float, float, float, float]:
    """What V1's own renderer draws for this brick, in raster pixels.

    V1 places a brick's top-left corner at ``(x, y)`` studs and fills
    ``across`` by ``down`` studs at ``stud_px`` pixels each. This is the
    independent expectation the projected footprint has to reproduce; it is
    computed from V1's layout and constant, not from the scene record.
    """
    across, down = placement.extents()
    return (x_studs * stud_px, y_studs * stud_px,
            (x_studs + across) * stud_px, (y_studs + down) * stud_px)


def iou(a, b) -> float:
    """Intersection over union of two ``(u0, v0, u1, v1)`` boxes."""
    au0, av0, au1, av1 = a
    bu0, bv0, bu1, bv1 = b
    iu0, iv0 = max(au0, bu0), max(av0, bv0)
    iu1, iv1 = min(au1, bu1), min(av1, bv1)
    if iu1 <= iu0 or iv1 <= iv0:
        return 0.0
    inter = (iu1 - iu0) * (iv1 - iv0)
    area_a = max(0.0, au1 - au0) * max(0.0, av1 - av0)
    area_b = max(0.0, bu1 - bu0) * max(0.0, bv1 - bv0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
