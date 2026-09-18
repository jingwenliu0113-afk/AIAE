"""Builds and renders one canonical scene. Runs inside Blender, only.

This module imports ``bpy`` and nothing from this project. That is the point:
``scripts/62_visual_stress_pbr.py`` calls Blender as a subprocess, so an
ordinary AST import closure started from that script can never reach this file,
and a module of ours that this file imported would sit outside every manifest.
Taking the whole scene from ``scene.json`` removes the possibility rather than
testing for it -- there is nothing here to miss.

The record is the authority and this file is a consumer of it. Every value used
is read from the JSON; none is defaulted from Blender and none is a constant
typed in here. What Blender resolves those values *to* is read back afterwards
and written to a receipt, so the scene graph that rendered can be compared
field by field against the record that asked for it, in both directions.
"""

import hashlib
import json
import math
import sys

import bmesh
import bpy


def _f(text):
    """A canonical decimal string as a float. Refuses anything else."""
    if not isinstance(text, str):
        raise ValueError(f"expected a decimal string, got {type(text).__name__}")
    return float(text)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _box(bm, x0, y0, z0, x1, y1, z1):
    verts = [bm.verts.new((x, y, z))
             for x, y, z in ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0),
                             (x0, y1, z0), (x0, y0, z1), (x1, y0, z1),
                             (x1, y1, z1), (x0, y1, z1))]
    for a, b, c, d in ((0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1),
                       (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)):
        bm.faces.new((verts[a], verts[b], verts[c], verts[d]))
    return verts


def _cylinder(bm, cx, cy, z0, z1, radius, segments):
    ring0, ring1 = [], []
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        ring0.append(bm.verts.new((x, y, z0)))
        ring1.append(bm.verts.new((x, y, z1)))
    for i in range(segments):
        j = (i + 1) % segments
        bm.faces.new((ring0[i], ring0[j], ring1[j], ring1[i]))
    bm.faces.new(tuple(reversed(ring0)))
    bm.faces.new(tuple(ring1))


def brick_mesh(brick, mesh_spec, name):
    """One brick: a bevelled body box plus unbevelled studs on its top.

    The bevel is not decoration. Without it the body edges are perfectly
    sharp, produce no edge highlight, and the largest visible difference
    between this renderer and V1's arithmetic shading disappears -- which is
    part of what the rendering-domain contrast is meant to measure. Studs are
    left sharp: a 1-unit bevel on a 12-unit cylinder would eat the stud.
    """
    x0, y0, _z = brick["translation_scene_units"]
    width, depth, height = brick["extent_scene_units"]
    bm = bmesh.new()
    body = _box(bm, x0, y0, 0.0, x0 + width, y0 + depth, height)
    if mesh_spec["bevel_width_ldu"] and mesh_spec["bevel_segments"]:
        edges = {e for v in body for e in v.link_edges}
        bmesh.ops.bevel(bm, geom=list(edges), offset=float(
            mesh_spec["bevel_width_ldu"]), offset_type="OFFSET",
            segments=int(mesh_spec["bevel_segments"]), profile=0.5,
            affect="EDGES", clamp_overlap=True)
    radius = mesh_spec["stud_diameter_ldu"] / 2.0
    for sx, sy, sz in brick["stud_positions_scene_units"]:
        _cylinder(bm, float(sx), float(sy), float(sz),
                  float(sz) + mesh_spec["stud_height_ldu"], radius,
                  int(mesh_spec["cylinder_segments"]))
    bm.normal_update()
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    if not mesh_spec["shade_smooth"]:
        for poly in mesh.polygons:
            poly.use_smooth = False
    return mesh


def _bsdf(obj):
    """The Principled node of an object's first material."""
    tree = obj.data.materials[0].node_tree
    return next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")


def _aim(light_obj):
    """The world-space direction the light's local -Z points along."""
    return tuple((light_obj.matrix_world.to_quaternion()
                  @ __import__("mathutils").Vector((0.0, 0.0, -1.0))))


def _aim_residual(light_obj) -> float:
    """Degrees between the light's aim and the direction to the origin."""
    import mathutils

    aim = mathutils.Vector(_aim(light_obj)).normalized()
    want = (-mathutils.Vector(light_obj.location)).normalized()
    return math.degrees(math.acos(max(-1.0, min(1.0, aim.dot(want)))))


def geometry_digest(objects) -> str:
    """SHA-256 over sorted vertex coordinates and face vertex counts.

    Quantised to 1e-4 of a unit before hashing: the coordinates are built from
    integers and a bevel, and hashing raw doubles would make the digest depend
    on the last bit of a cosine rather than on the shape.
    """
    parts = []
    for obj in sorted(objects, key=lambda o: o.name):
        mesh = obj.data
        verts = sorted(tuple(round(c, 4) for c in v.co) for v in mesh.vertices)
        faces = sorted(len(p.vertices) for p in mesh.polygons)
        parts.append(json.dumps([obj.name, verts, faces], sort_keys=True,
                                separators=(",", ":")))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# materials, table, world, light, camera
# ---------------------------------------------------------------------------

_PRINCIPLED = {
    "metallic": "Metallic", "roughness": "Roughness", "ior": "IOR",
    "transmission_weight": "Transmission Weight",
    "subsurface_weight": "Subsurface Weight", "coat_weight": "Coat Weight",
    "sheen_weight": "Sheen Weight", "emission_strength": "Emission Strength",
    "alpha": "Alpha", "specular_ior_level": "Specular IOR Level",
}


def principled(name, linear_rgb, spec, overrides=None):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    node = next(n for n in material.node_tree.nodes
                if n.type == "BSDF_PRINCIPLED")
    node.inputs["Base Color"].default_value = (*[_f(v) for v in linear_rgb], 1.0)
    values = {**{k: v for k, v in spec.items() if k in _PRINCIPLED},
              **(overrides or {})}
    for key, socket in _PRINCIPLED.items():
        if key in values:
            node.inputs[socket].default_value = _f(values[key])
    return material


def build(body):
    """The whole scene, from the record. Returns the objects it made."""
    scene = bpy.context.scene
    made = {"bricks": []}

    mesh_spec = body["mesh"]
    for brick in body["bricks"]:
        mesh = brick_mesh(brick, mesh_spec, brick["brick_id"])
        obj = bpy.data.objects.new(brick["brick_id"], mesh)
        obj.data.materials.append(principled(
            f"mat_{brick['brick_id']}", brick["colour_linear"],
            body["material"]))
        scene.collection.objects.link(obj)
        made["bricks"].append(obj)

    table = body["table"]
    edge = float(table["edge_scene_units"])
    bm = bmesh.new()
    _box(bm, -edge / 2, -edge / 2, -1.0, edge / 2, edge / 2, 0.0)
    tmesh = bpy.data.meshes.new("table")
    bm.to_mesh(tmesh)
    bm.free()
    for poly in tmesh.polygons:
        poly.use_smooth = False
    tobj = bpy.data.objects.new("table", tmesh)
    tobj.data.materials.append(principled(
        "mat_table", table["colour_linear"], body["material"],
        overrides={"roughness": table["roughness"],
                   "metallic": table["metallic"],
                   "specular_ior_level": table["specular_ior_level"]}))
    scene.collection.objects.link(tobj)
    made["table"] = tobj

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (
        *[_f(v) for v in body["world"]["colour_linear"]], 1.0)
    bg.inputs["Strength"].default_value = _f(body["world"]["strength"])
    world.cycles.sampling_method = body["world"]["sampling_method"]
    world.cycles.sample_map_resolution = int(
        body["world"]["sample_map_resolution"])
    world.cycles.max_bounces = int(body["world"]["max_bounces"])
    world.cycles.use_shadows = bool(body["world"]["use_shadows"])
    scene.world = world

    made["lights"] = []
    for index, spec in enumerate(body["lights"]):
        data = bpy.data.lights.new(f"light{index}", type=spec["type"])
        data.shape = spec["shape"]
        data.size = float(spec["size_scene_units"])
        data.energy = _f(spec["power_watts"])
        data.color = tuple(_f(v) for v in spec["colour_linear"])
        data.use_shadow = bool(spec["use_shadow"])
        data.spread = math.radians(_f(spec["spread_degrees"]))
        data.normalize = bool(spec["normalize"])
        data.cycles.use_multiple_importance_sampling = bool(
            spec["use_multiple_importance_sampling"])
        data.cycles.max_bounces = int(spec["max_bounces"])
        data.cycles.is_portal = bool(spec["is_portal"])
        obj = bpy.data.objects.new(f"light{index}", data)
        obj.location = tuple(float(v) for v in spec["location_scene_units"])
        if spec["aims_at_scene_centre"]:
            # Point -Z at the origin. Derived, then written to the receipt as
            # the resolved quaternion *and* as the residual angle between the
            # aim it produced and the direction to the origin, so the aim is
            # measured rather than described. The residual is what makes this
            # checkable: the axis below was ``(-fy, fx, 0)`` for six
            # generations -- the negation of the correct one, which rotates by
            # -angle and pointed every shadow-condition light 58.992 degrees
            # away from the scene centre while the record said it aimed at it.
            # A quaternion in a receipt nobody compares is not evidence.
            x, y, z = obj.location
            distance = math.sqrt(x * x + y * y + z * z)
            obj.rotation_mode = "QUATERNION"
            forward = (-x / distance, -y / distance, -z / distance)
            axis = (forward[1], -forward[0], 0.0)
            axis_len = math.hypot(axis[0], axis[1])
            angle = math.acos(max(-1.0, min(1.0, -forward[2])))
            if axis_len < 1e-12:
                obj.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            else:
                half = angle / 2.0
                s = math.sin(half) / axis_len
                obj.rotation_quaternion = (math.cos(half), axis[0] * s,
                                           axis[1] * s, 0.0)
        scene.collection.objects.link(obj)
        made["lights"].append(obj)
    return made


def camera(body, kind):
    spec = body["cameras"][kind]
    data = bpy.data.cameras.new(f"cam_{kind}")
    data.type = spec["type"]
    data.sensor_fit = spec["sensor_fit"]
    data.shift_x = _f(spec["shift_x"])
    data.shift_y = _f(spec["shift_y"])
    data.clip_start = _f(spec["clip_start"])
    data.clip_end = _f(spec["clip_end"])
    data.dof.use_dof = bool(spec["use_dof"])
    if spec["type"] == "ORTHO":
        data.ortho_scale = _f(spec["ortho_scale"])
    else:
        data.sensor_width = _f(spec["sensor_width_mm"])
        data.lens = _f(spec["lens_mm"])
    obj = bpy.data.objects.new(f"cam_{kind}", data)
    obj.location = tuple(_f(v) for v in spec["location_scene_units"])
    obj.rotation_euler = tuple(math.radians(v / 1000.0)
                               for v in spec["rotation_euler_millideg"])
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.scene.camera = obj
    return obj


def configure(body, device):
    """Render and output settings, entirely from the record."""
    scene = bpy.context.scene
    render, out = body["render"], body["output"]
    scene.render.engine = render["engine"]
    cycles = scene.cycles
    cycles.device = device
    cycles.samples = int(render["samples"])
    cycles.use_adaptive_sampling = bool(render["use_adaptive_sampling"])
    cycles.use_denoising = bool(render["use_denoising"])
    cycles.time_limit = _f(render["time_limit"])
    cycles.seed = int(render["seed"])
    cycles.use_animated_seed = bool(render["use_animated_seed"])
    cycles.sampling_pattern = render["sampling_pattern"]
    cycles.use_light_tree = bool(render["use_light_tree"])
    cycles.max_bounces = int(render["max_bounces"])
    cycles.diffuse_bounces = int(render["diffuse_bounces"])
    cycles.glossy_bounces = int(render["glossy_bounces"])
    cycles.transmission_bounces = int(render["transmission_bounces"])
    cycles.volume_bounces = int(render["volume_bounces"])
    cycles.transparent_max_bounces = int(render["transparent_max_bounces"])
    cycles.sample_clamp_direct = _f(render["sample_clamp_direct"])
    cycles.sample_clamp_indirect = _f(render["sample_clamp_indirect"])
    cycles.caustics_reflective = bool(render["caustics_reflective"])
    cycles.caustics_refractive = bool(render["caustics_refractive"])
    cycles.pixel_filter_type = render["pixel_filter_type"]
    cycles.filter_width = _f(render["filter_width"])
    cycles.film_exposure = _f(render["film_exposure"])
    cycles.light_sampling_threshold = _f(render["light_sampling_threshold"])
    cycles.blur_glossy = _f(render["blur_glossy"])
    cycles.min_light_bounces = int(render["min_light_bounces"])
    cycles.min_transparent_bounces = int(render["min_transparent_bounces"])
    cycles.use_fast_gi = bool(render["use_fast_gi"])
    cycles.sample_offset = int(render["sample_offset"])
    cycles.scrambling_distance = _f(render["scrambling_distance"])
    scene.render.film_transparent = bool(render["film_transparent"])
    scene.render.use_motion_blur = bool(render["use_motion_blur"])
    scene.render.resolution_x = int(body["raster"]["width_px"])
    scene.render.resolution_y = int(body["raster"]["height_px"])
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0

    scene.view_settings.view_transform = out["scene_view_transform"]
    scene.view_settings.look = out["scene_look"]
    scene.view_settings.exposure = _f(out["scene_exposure"])
    scene.view_settings.gamma = _f(out["scene_gamma"])
    scene.display_settings.display_device = out["display_device"]

    settings = scene.render.image_settings
    settings.file_format = out["file_format"]
    settings.color_depth = out["color_depth"]
    settings.exr_codec = out["exr_codec"]
    settings.color_mode = out["color_mode"]
    settings.color_management = out["image_color_management"]
    settings.view_settings.view_transform = out["image_view_transform"]
    settings.view_settings.look = out["image_look"]
    settings.view_settings.exposure = _f(out["image_exposure"])
    settings.view_settings.gamma = _f(out["image_gamma"])
    settings.display_settings.display_device = out["image_display_device"]


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------

CONTROL_POINTS = {"P_C": (0, 0, 0), "P_X": (20, 0, 0), "P_Y": (0, 20, 0),
                  "P_Z": (0, 0, 24), "P_XZ": (20, 0, 24)}


def control_points(body, cam_obj):
    """``world_to_camera_view`` on the control points, as absolute raster.

    Blender's own projection rather than ours, so the comparison is between
    what we believe ``sensor_fit`` and the aspect ratio mean and what Blender
    actually does with them.
    """
    from bpy_extras.object_utils import world_to_camera_view

    scene = bpy.context.scene
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    out = {}
    for name, point in CONTROL_POINTS.items():
        ndc = world_to_camera_view(scene, cam_obj,
                                   __import__("mathutils").Vector(point))
        out[name] = [ndc.x * width, (1.0 - ndc.y) * height]
    return out


def receipt(body, made, cam_obj, camera_kind, device, seconds):
    """What the scene graph resolved to, read back rather than restated."""
    scene = bpy.context.scene
    cycles = scene.cycles
    settings = scene.render.image_settings
    bricks = []
    for obj, spec in zip(made["bricks"], body["bricks"]):
        mesh = obj.data
        xs = [v.co.x for v in mesh.vertices]
        ys = [v.co.y for v in mesh.vertices]
        zs = [v.co.z for v in mesh.vertices]
        node = next(n for n in obj.data.materials[0].node_tree.nodes
                    if n.type == "BSDF_PRINCIPLED")
        bricks.append({
            "brick_id": obj.name,
            "declared_brick_id": spec["brick_id"],
            "vertices": len(mesh.vertices),
            "polygons": len(mesh.polygons),
            "bounds_min": [round(min(xs), 4), round(min(ys), 4),
                           round(min(zs), 4)],
            "bounds_max": [round(max(xs), 4), round(max(ys), 4),
                           round(max(zs), 4)],
            "base_color_linear": [round(v, 6) for v in
                                  node.inputs["Base Color"].default_value[:3]],
            "material": {key: round(node.inputs[socket].default_value, 6)
                         for key, socket in sorted(_PRINCIPLED.items())},
            "roughness": round(node.inputs["Roughness"].default_value, 6),
            "metallic": round(node.inputs["Metallic"].default_value, 6),
            "smooth_faces": sum(1 for p in mesh.polygons if p.use_smooth),
            "vertices": len(mesh.vertices),
            "polygons": len(mesh.polygons),
        })
    return {
        "kind": "brickagain.pbr_receipt",
        "scene_digest_declared": body.get("_scene_digest"),
        "camera_kind": camera_kind,
        "device_requested": device,
        "seconds": round(seconds, 3),
        "blender": {"version": bpy.app.version_string,
                    "build_hash": (bpy.app.build_hash.decode()
                                   if isinstance(bpy.app.build_hash, bytes)
                                   else str(bpy.app.build_hash))},
        "bricks": bricks,
        "geometry_digest": geometry_digest(
            made["bricks"] + [made["table"]]),
        "resolved": {
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "resolution_percentage": scene.render.resolution_percentage,
            "pixel_aspect": [scene.render.pixel_aspect_x,
                             scene.render.pixel_aspect_y],
            "engine": scene.render.engine,
            "device": cycles.device,
            "samples": cycles.samples,
            "use_adaptive_sampling": cycles.use_adaptive_sampling,
            "use_denoising": cycles.use_denoising,
            "time_limit": cycles.time_limit,
            "seed": cycles.seed,
            "use_animated_seed": cycles.use_animated_seed,
            "sampling_pattern": cycles.sampling_pattern,
            "use_light_tree": cycles.use_light_tree,
            "max_bounces": cycles.max_bounces,
            "diffuse_bounces": cycles.diffuse_bounces,
            "glossy_bounces": cycles.glossy_bounces,
            "transmission_bounces": cycles.transmission_bounces,
            "volume_bounces": cycles.volume_bounces,
            "transparent_max_bounces": cycles.transparent_max_bounces,
            "sample_clamp_direct": cycles.sample_clamp_direct,
            "sample_clamp_indirect": cycles.sample_clamp_indirect,
            "caustics_reflective": cycles.caustics_reflective,
            "caustics_refractive": cycles.caustics_refractive,
            "pixel_filter_type": cycles.pixel_filter_type,
            "filter_width": cycles.filter_width,
            "film_exposure": round(cycles.film_exposure, 6),
            "light_sampling_threshold": round(cycles.light_sampling_threshold, 6),
            "blur_glossy": round(cycles.blur_glossy, 6),
            "min_light_bounces": cycles.min_light_bounces,
            "min_transparent_bounces": cycles.min_transparent_bounces,
            "use_fast_gi": cycles.use_fast_gi,
            "sample_offset": cycles.sample_offset,
            "scrambling_distance": round(cycles.scrambling_distance, 6),
            "film_transparent": scene.render.film_transparent,
            "use_motion_blur": scene.render.use_motion_blur,
            "scene_view_transform": scene.view_settings.view_transform,
            "scene_look": scene.view_settings.look,
            "scene_exposure": scene.view_settings.exposure,
            "scene_gamma": scene.view_settings.gamma,
            "display_device": scene.display_settings.display_device,
            "image_color_management": settings.color_management,
            "image_view_transform": settings.view_settings.view_transform,
            "image_look": settings.view_settings.look,
            "image_exposure": settings.view_settings.exposure,
            "image_gamma": settings.view_settings.gamma,
            "image_display_device": settings.display_settings.display_device,
            "file_format": settings.file_format,
            "color_depth": settings.color_depth,
            "exr_codec": settings.exr_codec,
            "color_mode": settings.color_mode,
            "threads_mode": scene.render.threads_mode,
            "threads": scene.render.threads,
            "camera": {
                "type": cam_obj.data.type,
                "sensor_fit": cam_obj.data.sensor_fit,
                "ortho_scale": round(cam_obj.data.ortho_scale, 6),
                "lens": round(cam_obj.data.lens, 6),
                "sensor_width": round(cam_obj.data.sensor_width, 6),
                "shift": [cam_obj.data.shift_x, cam_obj.data.shift_y],
                "clip": [cam_obj.data.clip_start, cam_obj.data.clip_end],
                "use_dof": cam_obj.data.dof.use_dof,
                "location": [round(v, 6) for v in cam_obj.location],
                "rotation_euler": [round(v, 9) for v in cam_obj.rotation_euler],
            },
            "table": {
                "vertices": len(made["table"].data.vertices),
                "polygons": len(made["table"].data.polygons),
                # The `background` axis varies exactly one value, and the
                # receipt did not record it: a doctored or mis-set table colour
                # was invisible to every check.
                "base_color_linear": [
                    round(v, 6) for v in
                    _bsdf(made["table"]).inputs["Base Color"].default_value[:3]],
                "material": {
                    key: round(_bsdf(made["table"]).inputs[socket].default_value, 6)
                    for key, socket in sorted(_PRINCIPLED.items())},
                "bounds_z": [round(min(v.co.z for v in made["table"].data.vertices), 4),
                             round(max(v.co.z for v in made["table"].data.vertices), 4)],
                "edge_scene_units": round(
                    max(v.co.x for v in made["table"].data.vertices)
                    - min(v.co.x for v in made["table"].data.vertices), 4),
            },
            "lights": [{
                "type": o.data.type, "shape": o.data.shape,
                "size": round(o.data.size, 6),
                "energy": round(o.data.energy, 6),
                "color": [round(v, 6) for v in o.data.color],
                "use_shadow": o.data.use_shadow,
                "spread": round(o.data.spread, 9),
                "location": [round(v, 6) for v in o.location],
                "rotation_mode": o.rotation_mode,
                "rotation_quaternion": [round(v, 9)
                                        for v in o.rotation_quaternion],
                "spread_degrees": round(math.degrees(o.data.spread), 6),
                "normalize": o.data.normalize,
                "use_multiple_importance_sampling":
                    o.data.cycles.use_multiple_importance_sampling,
                "max_bounces": o.data.cycles.max_bounces,
                "is_portal": o.data.cycles.is_portal,
                # The aim, measured from the built rotation rather than
                # restated from the request: the local -Z the light actually
                # points along, and the angle between it and the direction to
                # the scene centre. Zero means the record's
                # ``aims_at_scene_centre`` is true of the scene that rendered.
                "aim_direction": [round(v, 9) for v in _aim(o)],
                "aim_residual_degrees": round(_aim_residual(o), 6),
            } for o in made["lights"]],
            "world": {
                "colour": [round(v, 6) for v in
                           bpy.context.scene.world.node_tree
                           .nodes["Background"].inputs["Color"]
                           .default_value[:3]],
                "strength": round(bpy.context.scene.world.node_tree
                                  .nodes["Background"].inputs["Strength"]
                                  .default_value, 6),
                "sampling_method": bpy.context.scene.world.cycles.sampling_method,
                "sample_map_resolution":
                    bpy.context.scene.world.cycles.sample_map_resolution,
                "max_bounces": bpy.context.scene.world.cycles.max_bounces,
                "use_shadows": bpy.context.scene.world.cycles.use_shadows,
            },
        },
        "control_points_raster": control_points(body, cam_obj),
    }


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    options = dict(zip(argv[::2], argv[1::2]))
    scene_path = options["--scene"]
    camera_kind = options["--camera"]
    device = options.get("--device", "CPU")
    exr_path = options["--exr"]
    receipt_path = options["--receipt"]

    raw = open(scene_path, "rb").read()
    body = json.loads(raw.decode("utf-8"))
    body["_scene_digest"] = hashlib.sha256(raw).hexdigest()

    if device != "CPU":
        import addon_utils
        addon_utils.enable("cycles", default_set=True, persistent=True)
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        try:
            prefs.refresh_devices()
        except Exception:
            pass
        for entry in prefs.devices:
            entry.use = entry.type == "METAL"

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    made = build(body)
    cam_obj = camera(body, camera_kind)
    configure(body, device)

    import time
    bpy.context.scene.render.filepath = exr_path
    started = time.time()
    bpy.ops.render.render(write_still=True)
    seconds = time.time() - started

    record = receipt(body, made, cam_obj, camera_kind, device, seconds)
    with open(receipt_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True, indent=1)
    print(f"###BUILT### {camera_kind} {device} {seconds:.3f}s")


if __name__ == "__main__":
    main()
