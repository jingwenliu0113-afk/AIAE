"""The V2 PBR track: a physically-based renderer over V1's own scenes.

V1 draws its scenes arithmetically -- an orthographic top-down projection at a
fixed 44 pixels per stud, with backgrounds, shadows and viewpoints synthesised
by array maths.  That makes its ground truth *constructed*: the scene says
"red 2x4 at row 1 column 1" and the renderer draws exactly that, so there is no
annotation error to subtract.  It also means the images have no real light
transport: no cast shadows, no specular highlight on a stud, no edge highlight,
and no perspective.

This package renders the same scenes through Cycles, keeping the constructed
ground truth and changing only the rendering domain.  The authority is a
canonical ``scene.json`` written by :mod:`src.vision.pbr.contract` from V1's
own constants; the Blender builder consumes that file and imports nothing from
this project, so there is no module of ours inside Blender's reach that an
import-closure manifest could miss.

Nothing here is on the Phase 3C pack allowlist: ``src/vision/**`` is denied by
name, so adding these modules cannot change ``pack_digest`` or reach the
execution node.
"""
