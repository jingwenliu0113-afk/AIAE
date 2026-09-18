#!/usr/bin/env python3
"""V2: the PBR visual stress track, from projection check to report.

The V1 visual stress run draws its scenes arithmetically. This one renders the
same scenes through Cycles and measures the same recogniser on the result, so
the contrast isolates the rendering domain rather than the scene, the
inventory or the recogniser.

Five modes, in the order they must happen:

``--check-projection``
    Stages A and C of the projection check, in pure Python. Stage A pins five
    control points to absolute raster coordinates with sign; Stage C takes
    every brick of every scene from V1's own layout through the writer and the
    camera and requires the orthographic footprint to reproduce what V1's
    renderer draws, exactly. Nothing is rendered: if this fails, rendering
    would only produce a corpus that pairs wrongly.

``--pilot``
    The replay pilot. Two cameras by two backends by one warm-up and three
    measured renders, so within-backend determinism is measured rather than
    assumed, and the warm-up -- which on Metal is a kernel compile taking over
    a minute -- is kept out of the comparison. Stage B compares Blender's own
    ``world_to_camera_view`` against this project's projection, and the
    feasibility check runs the real recogniser on the two distinct images.
    Produces no result that may be cited.

``--freeze``
    Writes the frozen contract: the condition set, the scene digests, the
    source manifests, the runtime record and the authorisation that binds
    them. Nothing after this may change without a new generation.

``--run``
    The formal corpus, against the frozen contract.

``--score`` / ``--report``
    Derived from the archived images, never from a summary the run wrote.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.pbr import colour as pbr_colour            # noqa: E402
from src.vision.pbr import contract as pbr                 # noqa: E402
from src.vision.pbr import project as pbr_project          # noqa: E402

#: v2gen09 supersedes v2gen08. Nothing in v2gen08 was found wrong: its own
#: checks all held and its scores stand. What moved was the tree underneath
#: it -- ``analyse_photo`` lived in ``src/ui/full.py`` and raised a ``UiError``
#: defined in ``src/ui/app.py``, so this script's import closure covered the
#: interface's notice text as well as the recogniser. Correcting a sentence
#: about Phase 3C therefore moved ``evaluation_source_manifest_digest`` and
#: made a rendered corpus unverifiable, for a change to code the run never
#: executes. v2gen09 is the re-derivation on the tree where the recogniser has
#: its own module, so a notice can be corrected without retiring a corpus.
#:
#: The renders are byte-reproducible on the CPU backend, so this re-derives
#: rather than re-measures -- but that is checked by the archive's own replay,
#: not claimed here.
GENERATION = "v2gen11"

BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
BUILDER = ROOT / "src/vision/pbr/builder_blender.py"

#: The scenes V1 uses, and therefore the scenes this pairs against.
SCENE_INDICES = tuple(range(8))

#: Conditions rendered for the paired contrast. Every one of these is a V1
#: condition id, so the pairing key is V1's own. Four of V1's axes are absent
#: and the report says so rather than leaving a reader to count: ``background``
#: textured and cluttered would need a texture and clutter objects this track
#: has not defined, and ``viewpoint`` and ``quality`` are post-processes V1
#: applies to its own base image, so putting them here would compare a
#: post-process to a render.
PAIRED_CONDITIONS = (
    "baseline",
    "background=grey",
    "background=dark",
    "shadow=soft",
    "shadow=hard",
    "lighting=dim",
    "occlusion=partial",
    "occlusion=heavy",
)

#: Conditions with no V1 counterpart. Reported inside V2 only, never paired.
V2_ONLY_CONDITIONS = ("render_sampling_noise",)

#: The perspective arm. Baseline geometry only: its contrast is against V2's
#: own orthographic render, not against V1.
PERSPECTIVE_CONDITIONS = ("baseline",)

#: Decided by measurement in discovery, not chosen. Three measured CPU renders
#: of one scene produced byte-identical canonical pixels; three Metal renders
#: produced three different digests, differing in 6 to 13 pixels of 307,200 by
#: one float16 least significant bit. So CPU is the formal backend and GPU is
#: recorded as non-replayable.
FORMAL_BACKEND = "CPU"

PILOT_SCENE = 6
PILOT_REPEATS = 3
IOU_FLOOR = 0.25
MAX_RECOGNISE_SECONDS = 120.0


def plain(value):
    """``value`` with numpy scalars and arrays turned into JSON types.

    The recogniser's diagnostics carry numpy floats -- a threshold it measured
    -- and one of those in a record is a record that cannot be written. So the
    coercion happens once, here, on the way to disk, rather than being
    remembered at every call site.
    """
    import numpy as np

    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    return value


class Refused(RuntimeError):
    """A step that will not run, with the reason in the message."""


# ---------------------------------------------------------------------------
# runtime identity
# ---------------------------------------------------------------------------

def tree_manifest(directory: Path) -> tuple[dict, str]:
    """Every file under ``directory``, by relative path and SHA-256.

    An executable's own hash misses the bundled Python, numpy, Cycles kernels
    and OCIO configuration that decide what a render is, so the whole bundle
    is measured.
    """
    from src.training.session import sha256_file

    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.is_symlink():
            files[str(path.relative_to(directory))] = sha256_file(path)
    body = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return files, hashlib.sha256(body.encode("utf-8")).hexdigest()


def runtime_record() -> dict:
    """Blender and the machine, to the depth that changes a render.

    macOS does not version its Metal driver separately, so the OS build is
    what stands for it. Cycles compiles Metal kernels at run time and caches
    them outside the bundle, which is why the OS build has to be here at all:
    the same Blender on a different macOS can compile a different kernel.
    """
    if not BLENDER.is_file():
        raise Refused(f"{BLENDER} is not installed")
    # BLENDER is .../Blender.app/Contents/MacOS/Blender, so parents[1] is
    # already Contents. This read `app / "Contents"` for seven generations --
    # /Applications/Blender.app/Contents/Contents, which does not exist -- so
    # rglob yielded nothing, blender_bundle_files was 0 and
    # blender_bundle_digest was sha256("{}") over 0 of 6498 files. The
    # executable's own hash was still recorded, so the runtime was not
    # unpinned; but the bundled Python, Cycles kernels and OCIO configuration
    # this manifest exists to cover were not measured at all, and
    # runtime_digest -- the value mode_run enforces -- was computed over the
    # vacuous field.
    files, digest = tree_manifest(BLENDER.parents[1])
    version = subprocess.run([str(BLENDER), "--version"], capture_output=True,
                             text=True, timeout=120).stdout.strip()
    from src.training.session import sha256_file

    dmg = ROOT / "artifacts/runtime/blender-5.2.1-macos-arm64.dmg"
    record = {
        "blender_version_output": version,
        "blender_executable_sha256": sha256_file(BLENDER),
        "blender_bundle_files": len(files),
        "blender_bundle_digest": digest,
        "blender_dmg_present": dmg.is_file(),
        "blender_dmg_sha256": sha256_file(dmg) if dmg.is_file() else None,
        "blender_dmg_source": ("https://download.blender.org/release/"
                               "Blender5.2/blender-5.2.1-macos-arm64.dmg"),
        "blender_checksum_source": ("https://download.blender.org/release/"
                                    "Blender5.2/blender-5.2.1.sha256"),
        "macos_version": platform.mac_ver()[0],
        "macos_build": subprocess.run(["sw_vers", "-buildVersion"],
                                      capture_output=True, text=True).stdout.strip(),
        "machine": platform.machine(),
        "hardware_model": subprocess.run(
            ["sysctl", "-n", "hw.model"], capture_output=True,
            text=True).stdout.strip(),
        "cpu_brand": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
            text=True).stdout.strip(),
        "note_metal_driver": ("macOS does not version the Metal driver "
                              "separately; the OS build stands for it"),
    }
    record["runtime_digest"] = pbr.scene_digest(record)
    return record


def source_manifests() -> dict:
    """The renderer directory, and the evaluation import closure.

    Two manifests because two things must be pinned in different ways. The
    renderer directory is measured *by directory*: ``builder_blender.py`` is
    launched as a subprocess and never imported, so an import closure cannot
    reach it and a file added there would travel unrecorded. The evaluation
    side is measured by import closure from this script, which is how the rest
    of the project measures execution sources.
    """
    from src.training import pack
    from src.training.session import sha256_file

    renderer_dir = ROOT / "src/vision/pbr"
    renderer = {str(p.relative_to(ROOT)): sha256_file(p)
                for p in sorted(renderer_dir.rglob("*.py"))}
    # ``import_closure`` returns the modules an entry point *reaches*, not the
    # entry point itself -- so this file, which holds the condition set, the
    # scorer call and the contrast logic, would have gone unpinned by either
    # manifest. The entry points are added by name.
    #
    # Only *this* script. The V3 probe used to be pinned here too, which
    # coupled two generations that share nothing but a hashing helper: a fix to
    # the V3 script moved V2's evaluation manifest, and a fix to a module V2
    # needs moved V3's. Each track pins its own entry point, the same way
    # ``scripts/64_pbr_archive.py`` is deliberately outside both so a verifier
    # can be improved without voiding a rendered corpus.
    entry_points = ("scripts/62_visual_stress_pbr.py",)
    closure = pack.import_closure(root=ROOT, entry_points=entry_points)
    evaluation = {rel: sha256_file(ROOT / rel)
                  for rel in sorted(set(closure) | set(entry_points))}

    # Everything the Blender-only module imports from this project must be in
    # one of the two manifests. It imports nothing today; the check is what
    # keeps that true.
    builder_closure = pack.import_closure(
        root=ROOT, entry_points=("src/vision/pbr/builder_blender.py",))
    unmanifested = sorted(set(builder_closure) - set(renderer) - set(evaluation))

    return {
        "renderer_source_manifest": renderer,
        "renderer_source_manifest_digest": pbr.scene_digest(renderer),
        "evaluation_source_manifest": evaluation,
        "evaluation_source_manifest_digest": pbr.scene_digest(evaluation),
        "builder_project_imports": sorted(builder_closure),
        "builder_imports_unmanifested": unmanifested,
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def scene_body(scene_index: int, condition_id: str) -> dict:
    """The canonical record, with the V2-only conditions applied."""
    if condition_id == "render_sampling_noise":
        body = pbr.build_scene(scene_index, "baseline")
        body["render"] = {**body["render"], "samples": 16}
        body["provenance"] = {**body["provenance"],
                              "condition_id": "render_sampling_noise",
                              "v2_only": True,
                              "v2_only_basis": "baseline",
                              "v2_only_change": "samples 256 -> 16"}
        return body
    body = pbr.build_scene(scene_index, condition_id)
    body["provenance"] = {**body["provenance"], "v2_only": False}
    return body


def condition_table(condition_id: str, scene_index: int) -> dict:
    """The record for one cell.

    The per-condition table colour, world strength and light now live in
    :mod:`src.vision.pbr.contract`, where the calibration that produced them is
    recorded beside them. This function is the V2-only conditions and nothing
    else -- a second place that adjusted the physical setup was how a
    directional light ended up in ``lighting=dim``.
    """
    return scene_body(scene_index, condition_id)


def render_one(body: dict, camera: str, device: str, out_dir: Path,
               stem: str) -> dict:
    """One render, plus the canonical PNG and the receipt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_path = out_dir / f"{stem}.scene.json"
    raw = pbr.canonical_bytes(body)
    scene_path.write_bytes(raw)
    exr = out_dir / f"{stem}.exr"
    receipt = out_dir / f"{stem}.receipt.json"
    started = time.time()
    proc = subprocess.run(
        [str(BLENDER), "-b", "--factory-startup", "--python", str(BUILDER),
         "--", "--scene", str(scene_path), "--camera", camera,
         "--device", device, "--exr", str(exr), "--receipt", str(receipt)],
        capture_output=True, text=True, timeout=3600, cwd=str(ROOT))
    wall = time.time() - started
    if proc.returncode != 0 or not exr.is_file():
        raise Refused(
            f"render {stem} exited {proc.returncode}\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    png = pbr_colour.to_canonical_png(exr, out_dir / f"{stem}.png", body["png"])
    return {
        "stem": stem,
        "camera": camera,
        "device": device,
        "scene_digest": hashlib.sha256(raw).hexdigest(),
        "scene_bytes": len(raw),
        "wall_seconds": round(wall, 3),
        "exit_code": proc.returncode,
        "image": png,
        "receipt": json.loads(receipt.read_text()),
    }


# ---------------------------------------------------------------------------
# scoring, through V1's own scorer
# ---------------------------------------------------------------------------

def score_row(scene_index: int, condition_id: str, png_path: Path) -> dict:
    """One image's verdict, computed by *V1's* scorer.

    Reusing :func:`src.eval.visual_stress.score_image` rather than writing a
    second scorer is what makes the pairing exact. Two scorers agreeing on a
    definition today is two scorers that can disagree later, and a contrast
    between arithmetic and PBR rendering measured by two different rulers is
    not a contrast between the renderers.
    """
    from src.eval import visual_stress as vs
    from src.ui.photo import PHOTO_MULTI, RECOGNISE_CV, analyse_photo
    from src.vision import synthetic

    scene = synthetic.scenes()[scene_index]
    basis = "baseline" if condition_id in V2_ONLY_CONDITIONS else condition_id
    condition = pbr._condition(synthetic, basis)
    entry = {"image_id": f"{scene.scene_id}/{condition_id}",
             "scene_id": scene.scene_id, "condition_id": condition_id,
             "condition": condition, "member": png_path.name,
             "ground_truth": scene.ground_truth()}
    started = time.time()
    analysis = analyse_photo(png_path.read_bytes(), mode=PHOTO_MULTI,
                             method=RECOGNISE_CV)
    seconds = time.time() - started
    row = vs.score_image(entry, analysis, scene)
    row["recognise_seconds"] = round(seconds, 3)
    row["colour_problems"] = len(analysis.colour_problems)
    row["scorer"] = "src.eval.visual_stress.score_image"
    row["colour_offsets"] = colour_offsets(scene_index, condition_id, png_path)
    return row


def colour_offsets(scene_index: int, condition_id: str, png_path: Path) -> list:
    """What each brick's top face renders at, against its palette value.

    The mechanism behind the colour-error contrast, measured rather than
    asserted. Under a uniform dome a dielectric reflects a near-constant
    specular floor on top of its albedo, so a rendered surface sits at roughly
    ``albedo * (1 - F) + F``: bright channels barely move and dark ones are
    lifted a long way. V1 by contrast fills each brick with the exact palette
    RGB. Without this measurement "V2's colour errors are 2.3x V1's" invites the
    reading "PBR shading is harder to read", when the effect is dominated by an
    additive floor that a different environment would change. The table is
    Lambertian for exactly this reason; a brick is not, because a brick's
    shading is what the contrast is *for*.
    """
    import numpy as np
    from PIL import Image

    basis = "baseline" if condition_id in V2_ONLY_CONDITIONS else condition_id
    body = pbr.build_scene(scene_index, basis)
    image = np.asarray(Image.open(png_path).convert("RGB")).astype(int)
    height, width = image.shape[:2]
    out = []
    for brick in body["bricks"]:
        u0, v0, u1, v1 = pbr_project.top_face_aabb(body, "ortho", brick)
        # Inset by the bevel so the patch is the flat top face and not its
        # rounded edge: the record's corners describe the unbevelled box.
        inset = float(body["mesh"]["bevel_width_ldu"]) * (
            body["raster"]["target_px_per_stud"] / body["units"]["ldu_stud"])
        a = max(0, int(round(v0 + inset)))
        b = min(height, int(round(v1 - inset)))
        c = max(0, int(round(u0 + inset)))
        d = min(width, int(round(u1 - inset)))
        if b <= a or d <= c:
            continue
        patch = image[a:b, c:d].reshape(-1, 3)
        median = [int(v) for v in np.median(patch, axis=0)]
        palette = [int(v) for v in brick["colour_srgb_uint8"]]
        out.append({
            "brick_id": brick["brick_id"],
            "colour_id": brick["colour_id"],
            "palette_srgb": palette,
            "rendered_median_srgb": median,
            "offset_codes": [m - p for m, p in zip(median, palette)],
            "fraction_saturated": round(
                float((patch == 255).all(axis=1).mean()), 6),
        })
    return out


def v1_rows() -> dict:
    """V1's own per-image rows, keyed by ``image_id``, for the pairing."""
    path = (ROOT / "data/reports/60_visual_stress"
            / "cv__per_image_results.json")
    body = json.loads(path.read_text())
    rows = body if isinstance(body, list) else body.get(
        "results", body.get("images"))
    return {row["image_id"]: row for row in rows}


PAIRED_METRICS = ("n_detected", "false_negatives", "false_positives",
                  "abstentions", "n_part_errors", "n_colour_errors",
                  "top1_correct_among_matched", "mean_confidence")

#: Already a per-image mean. Summing one produces a number with no meaning --
#: "the total of eighty averages" -- so these are reported as a mean of means
#: and the column header says so.
MEAN_METRICS = ("mean_confidence",)

#: Which direction is *better* for each paired metric. Without this the
#: worse/better counts are a raw sign count, and for the three metrics where
#: higher is better the two column headers are simply swapped -- which is what
#: the earlier generations published: "V2 worse" counted the cells where V2
#: detected *more* bricks, classified *more* correctly and was *more*
#: confident.
HIGHER_IS_BETTER = {
    "n_detected": True,
    "false_negatives": False,
    "false_positives": False,
    "abstentions": False,
    "n_part_errors": False,
    "n_colour_errors": False,
    "top1_correct_among_matched": True,
    "mean_confidence": True,
}


def paired_contrast(v2_rows: list, v1_by_id: dict) -> dict:
    """``V2 - V1`` on the conditions both ran, case by case.

    Paired on ``image_id``, which is V1's own key, and only over the
    intersection: a V2-only condition has no counterpart and pooling it in
    would make the delta a statement about a set V1 never saw.
    """
    pairs, missing = [], []
    for row in v2_rows:
        if row["condition_id"] in V2_ONLY_CONDITIONS or row["camera"] != "ortho":
            continue
        other = v1_by_id.get(row["image_id"])
        if other is None:
            missing.append(row["image_id"])
            continue
        pairs.append((row, other))
    out = {"n_pairs": len(pairs), "unpaired_v2_image_ids": missing,
           "metrics": {}}
    for name in PAIRED_METRICS:
        deltas = [float(a[name]) - float(b[name]) for a, b in pairs]
        v2_total = sum(float(a[name]) for a, _b in pairs)
        v1_total = sum(float(b[name]) for _a, b in pairs)
        divisor = len(pairs) if name in MEAN_METRICS else 1
        higher_is_better = HIGHER_IS_BETTER[name]
        n_higher = sum(1 for d in deltas if d > 0)
        n_lower = sum(1 for d in deltas if d < 0)
        out["metrics"][name] = {
            "is_already_a_mean": name in MEAN_METRICS,
            "higher_is_better": higher_is_better,
            "v2_total": round(v2_total / divisor, 4),
            "v1_total": round(v1_total / divisor, 4),
            "mean_delta": round(sum(deltas) / len(deltas), 6) if deltas else None,
            # Direction-free counts, so a reader can check the labelled ones.
            "n_v2_higher": n_higher,
            "n_v2_lower": n_lower,
            "n_v2_worse": n_lower if higher_is_better else n_higher,
            "n_v2_better": n_higher if higher_is_better else n_lower,
            "n_equal": sum(1 for d in deltas if d == 0),
            # What a per-condition reading of the same deltas looks like. A net
            # of +2 built from +25 and -23 is not "barely moved", and the
            # earlier report had no field that could tell the two apart.
            "sum_abs_delta": round(sum(abs(d) for d in deltas), 6),
            "n_changed": sum(1 for d in deltas if d != 0),
        }
    exact_v2 = sum(1 for a, _b in pairs if a["inventory_exact_match"])
    exact_v1 = sum(1 for _a, b in pairs if b["inventory_exact_match"])
    out["inventory_exact_match"] = {"v2": exact_v2, "v1": exact_v1,
                                    "denominator": len(pairs)}
    return out


def camera_contrast(v2_rows: list) -> dict:
    """``perspective - ortho`` inside V2, on the cells that ran both."""
    by_key = {}
    for row in v2_rows:
        by_key.setdefault((row["scene_id"], row["condition_id"]), {})[
            row["camera"]] = row
    pairs = [(v["perspective"], v["ortho"]) for v in by_key.values()
             if "perspective" in v and "ortho" in v]
    out = {"n_pairs": len(pairs), "metrics": {}}
    for name in PAIRED_METRICS:
        deltas = [float(a[name]) - float(b[name]) for a, b in pairs]
        divisor = len(pairs) if (name in MEAN_METRICS and pairs) else 1
        higher_is_better = HIGHER_IS_BETTER[name]
        n_higher = sum(1 for d in deltas if d > 0)
        n_lower = sum(1 for d in deltas if d < 0)
        out["metrics"][name] = {
            "is_already_a_mean": name in MEAN_METRICS,
            "higher_is_better": higher_is_better,
            "mean_delta": round(sum(deltas) / len(deltas), 6) if deltas else None,
            "persp_total": round(
                sum(float(a[name]) for a, _b in pairs) / divisor, 4),
            "ortho_total": round(
                sum(float(b[name]) for _a, b in pairs) / divisor, 4),
            "n_persp_higher": n_higher,
            "n_persp_lower": n_lower,
            "n_persp_worse": n_lower if higher_is_better else n_higher,
            "n_persp_better": n_higher if higher_is_better else n_lower,
            "n_equal": sum(1 for d in deltas if d == 0),
        }
    out.update(camera_geometry())
    return out


def camera_geometry() -> dict:
    """The perspective camera's registration and minification, derived.

    Derived from the records and V1's own layout rather than quoted, because
    the camera distance is per-scene: the earlier report hardcoded the figure
    for scene_06, which is the widest canvas and therefore the *least* affected
    cell, and presented it as the effect.
    """
    from src.eval import visual_stress as vs
    from src.vision import synthetic

    worst, minified = 0.0, []
    condition = pbr._condition(synthetic, "baseline")
    for index in SCENE_INDICES:
        body = pbr.build_scene(index, "baseline")
        scene = synthetic.scenes()[index]
        truth = vs.truth_boxes(scene, condition)
        for entry, brick in zip(truth, body["bricks"]):
            want = tuple(float(v) for v in entry["box"])
            for camera in ("ortho", "perspective"):
                got = pbr_project.top_face_aabb(body, camera, brick)
                worst = max(worst, max(abs(a - b) for a, b in zip(got, want)))
        span = pbr.parse_decimal(body["cameras"]["ortho"]["ortho_scale"])
        distance = pbr.parse_decimal(
            body["cameras"]["perspective"]["location_scene_units"][2])
        minified.append(100.0 * (1.0 - span / distance))
    return {
        "registration_px": round(worst, 6),
        "table_minification_percent_min": round(min(minified), 3),
        "table_minification_percent_max": round(max(minified), 3),
        "note_registration": (
            "the largest disagreement, over every brick and both cameras, "
            "between the projected top-face box and V1's own truth box"),
    }


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

#: Control points in the scene frame, whose origin is the canvas *corner*.
#: Named by where they land rather than by their coordinates, because the
#: expectation is computed from V1's 44 px/stud and the corner convention --
#: not by calling the projection this is meant to check. A stage that
#: recomputes the thing it is testing passes whatever the code does.
def control_points(body: dict) -> dict:
    """``{name: (point, {camera: (u, v)})}`` with the expectation stated.

    ``+Y`` is image-up and the origin is the top-left corner, so the canvas
    occupies negative ``Y``. A point at ``+Y`` would be above the frame, which
    is why the interior points are at negative ``Y`` and why getting the sign
    wrong here cannot pass silently.
    """
    width = body["raster"]["width_px"]
    height = body["raster"]["height_px"]
    stud_px = body["raster"]["target_px_per_stud"]
    ldu_stud = body["units"]["ldu_stud"]
    ratio = stud_px / ldu_stud                       # 2.2 px per LDU
    offset = body["cameras"]["ortho"]["canvas_offset_scene_units"]
    cx, cy = pbr.parse_decimal(offset[0]), pbr.parse_decimal(offset[1])
    distance = pbr.parse_decimal(
        body["cameras"]["perspective"]["location_scene_units"][2])
    top = body["units"]["ldu_brick"]
    # The perspective camera registers the brick top plane at 44 px/stud, so
    # the *table* is the plane that moves: it sits `top` further from the
    # camera and shrinks about the optical axis by the ratio of the distances.
    table = (distance - top) / distance

    def at(x, y, z=0):
        return (x, y, z)

    cu, cv = width / 2.0, height / 2.0

    def shrunk(u, v):
        """A table-plane expectation under the perspective camera."""
        return (cu + (u - cu) * table, cv + (v - cv) * table)

    # Table-plane points: identical under ortho, minified under perspective.
    table_points = {
        "corner":         (at(0, 0),               (0.0, 0.0)),
        "one_stud_right": (at(ldu_stud, 0),        (stud_px, 0.0)),
        "one_stud_down":  (at(0, -ldu_stud),       (0.0, stud_px)),
        "one_stud_both":  (at(ldu_stud, -ldu_stud), (stud_px, stud_px)),
        "centre":         (at(cx, -cy),            (cu, cv)),
    }
    out = {}
    for name, (point, (u, v)) in table_points.items():
        out[name] = (point, {"ortho": (u, v), "perspective": shrunk(u, v)})
    # Brick-top points: 44 px/stud under *both* cameras, which is the whole
    # reason one scorer can measure both. On the optical axis a lift moves
    # nothing; one stud off it moves nothing either, now that the plane the
    # lift lands on is the registered one -- and that is the falsifiable part:
    # if the camera stood at `span` instead of `span + 24` these would be out
    # by 5.5 to 12.2 per cent.
    out["centre_lifted"] = (at(cx, -cy, top), {"ortho": (cu, cv),
                                               "perspective": (cu, cv)})
    out["lifted_one_stud_right"] = (
        at(cx + ldu_stud, -cy, top),
        {"ortho": (cu + stud_px, cv), "perspective": (cu + stud_px, cv)})
    out["lifted_corner"] = (at(0, 0, top), {"ortho": (0.0, 0.0),
                                            "perspective": (0.0, 0.0)})
    out["lifted_one_stud_both"] = (
        at(ldu_stud, -ldu_stud, top),
        {"ortho": (stud_px, stud_px), "perspective": (stud_px, stud_px)})
    return out


def stage_a(body: dict) -> list:
    """Control points against absolute raster coordinates, with sign."""
    problems = []
    for name, (point, expected) in control_points(body).items():
        for camera, (want_u, want_v) in expected.items():
            got_u, got_v = pbr_project.project(body, camera, point)
            for axis, got, want in (("u", got_u, want_u), ("v", got_v, want_v)):
                if abs(got - want) > 0.001:
                    problems.append(
                        f"stage A {camera} {name}.{axis}: {got:.6f} vs "
                        f"{want:.6f}")
    return problems


def stage_c(scene_index: int, condition_id: str) -> tuple[list, list]:
    """Every brick, V1 spec through the writer to the raster footprint."""
    from src.vision import synthetic

    problems, rows = [], []
    body = condition_table(condition_id, scene_index)
    basis = "baseline" if condition_id in V2_ONLY_CONDITIONS else condition_id
    condition = pbr._condition(synthetic, basis)
    scene = synthetic.scenes()[scene_index]
    cells = synthetic.layout(scene, condition)
    if len(cells) != len(body["bricks"]):
        problems.append(
            f"{scene.scene_id}/{condition_id}: V1 lays out {len(cells)} "
            f"bricks and the record holds {len(body['bricks'])}")
        return problems, rows
    width, height = body["raster"]["width_px"], body["raster"]["height_px"]
    seen = set()
    for index, (brick, (placement, x, y)) in enumerate(zip(body["bricks"], cells)):
        tag = f"{scene.scene_id}/{condition_id}/b{index:02d}"
        across, down = placement.extents()
        for field, got, want in (
                ("part", brick["part"], placement.part),
                ("footprint", tuple(brick["footprint_studs"]), (across, down)),
                ("grid", tuple(brick["grid_row_col"]), (placement.row, placement.col)),
                ("colour_id", brick["colour_id"], placement.colour_id),
                ("turn", brick["turn"], placement.turn)):
            if got != want:
                problems.append(f"{tag} {field}: {got!r} vs V1 {want!r}")
        studs = brick["stud_positions_scene_units"]
        if len(studs) != across * down:
            problems.append(
                f"{tag}: {len(studs)} stud positions for a "
                f"{across}x{down} footprint")
        if len({tuple(s) for s in studs}) != len(studs):
            problems.append(f"{tag}: repeated stud position")
        x0, y0, _z = brick["translation_scene_units"]
        w_ldu, d_ldu, h_ldu = brick["extent_scene_units"]
        want_corners = [[x0, y0, h_ldu], [x0 + w_ldu, y0, h_ldu],
                        [x0 + w_ldu, y0 + d_ldu, h_ldu], [x0, y0 + d_ldu, h_ldu]]
        if brick["top_face_corners_scene_units"] != want_corners:
            problems.append(f"{tag}: top face corners are not the extent")
        got_box = pbr_project.top_face_aabb(body, "ortho", brick)
        want_box = pbr_project.v1_footprint(placement, x, y,
                                            body["raster"]["target_px_per_stud"])
        if any(abs(g - w) > 0.001 for g, w in zip(got_box, want_box)):
            problems.append(
                f"{tag}: ortho footprint {got_box} does not reproduce V1's "
                f"{want_box}")
        u0, v0, u1, v1 = got_box
        outside = (u0 < -0.001 or v0 < -0.001
                   or u1 > width + 0.001 or v1 > height + 0.001)
        # Not a failure. V1's occlusion shift moves a brick left and
        # ``canvas_size`` counts the negative extent into the *width* without
        # moving the drawing origin, so V1 itself draws that brick partly off
        # the left edge and its own truth box has a negative coordinate. A
        # check that refused this would be refusing V1's geometry.
        # The perspective camera registers the brick top plane, so this box has
        # to reproduce V1's footprint too. The earlier check compared the
        # perspective box against `got_box` scaled about the centre by
        # `depth_scale` -- the same arithmetic `project` had just done, so its
        # residual was 1.1e-13 against a 0.01 px tolerance and it could not
        # fail. Comparing against V1's own footprint instead makes the camera
        # distance, the reference plane and the depth formula all falsifiable:
        # a reference plane of 0 puts this box 5.5 to 12.2 per cent out.
        persp_box = pbr_project.top_face_aabb(body, "perspective", brick)
        if any(abs(g - w) > 0.001 for g, w in zip(persp_box, want_box)):
            problems.append(
                f"{tag}: perspective footprint {persp_box} does not reproduce "
                f"V1's {want_box}; the brick top plane is what this camera "
                "registers")
        # ... and the table plane must be minified by exactly the ratio of the
        # camera distances. If the reference plane were the table this would be
        # 1.0, so the two checks together pin which plane is registered.
        span = pbr.parse_decimal(body["cameras"]["ortho"]["ortho_scale"])
        distance = pbr.parse_decimal(
            body["cameras"]["perspective"]["location_scene_units"][2])
        foot = [x0, y0, 0]
        ortho_uv = pbr_project.project(body, "ortho", foot)
        persp_uv = pbr_project.project(body, "perspective", foot)
        centre_u, centre_v = width / 2.0, height / 2.0
        want_uv = (centre_u + (ortho_uv[0] - centre_u) * span / distance,
                   centre_v + (ortho_uv[1] - centre_v) * span / distance)
        if any(abs(g - w) > 0.001 for g, w in zip(persp_uv, want_uv)):
            problems.append(
                f"{tag}: the table plane projects to {persp_uv} and the ratio "
                f"of camera distances says {want_uv}")
        # V1's scorer rounds its truth boxes to whole pixels, and the occlusion
        # shifts are fractional studs, so the unrounded ideal above and the box
        # the scorer actually uses are not always the same box. Recorded rather
        # than refused: the rounding is V1's, and the size of the gap belongs
        # in the report.
        rounded = (int(round(want_box[0])), int(round(want_box[1])),
                   int(round(want_box[0])) + int(round((want_box[2] - want_box[0]))),
                   int(round(want_box[1])) + int(round((want_box[3] - want_box[1]))))
        rounding_px = max(abs(a - b) for a, b in zip(want_box, rounded))
        key = f"{brick['part']}:{brick['colour_id']}"
        seen.add((index, key))
        rows.append({"tag": tag, "outside_canvas": bool(outside),
                     "ortho_aabb": [round(v, 3) for v in got_box],
                     "v1_footprint": [round(v, 3) for v in want_box],
                     "v1_scorer_rounded_box": list(rounded),
                     "v1_rounding_px": round(rounding_px, 3),
                     "perspective_aabb": [round(v, 3) for v in persp_box]})
    got_inv = {}
    for brick in body["bricks"]:
        key = f"{brick['part']}:{brick['colour_id']}"
        got_inv[key] = got_inv.get(key, 0) + 1
    if dict(sorted(got_inv.items())) != scene.colour_inventory():
        problems.append(
            f"{scene.scene_id}/{condition_id}: colour inventory disagrees "
            "with V1")
    return problems, rows


def mode_check_projection(args) -> int:
    problems, checked = [], 0
    conditions = PAIRED_CONDITIONS + V2_ONLY_CONDITIONS
    for scene_index in SCENE_INDICES:
        body = condition_table("baseline", scene_index)
        problems.extend(stage_a(body))
        for condition_id in conditions:
            found, rows = stage_c(scene_index, condition_id)
            problems.extend(found)
            checked += len(rows)
    print(json.dumps({
        "mode": "check-projection",
        "scenes": len(SCENE_INDICES),
        "conditions": len(conditions),
        "bricks_checked": checked,
        "problems": problems,
        "ok": not problems,
    }, indent=1))
    return 0 if not problems else 1


def mode_pilot(args) -> int:
    """The replay pilot: determinism, Stage B, and a feasibility check.

    Produces nothing citable. Its whole job is to answer three questions
    before a corpus is rendered: does a backend reproduce its own pixels, does
    Blender agree with this project about where a point lands, and does the
    recogniser run at all on a physically-rendered image.
    """
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise Refused(f"{out_dir} is not empty; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    body = condition_table("baseline", PILOT_SCENE)
    problems = list(stage_a(body))
    stage_c_problems, _rows = stage_c(PILOT_SCENE, "baseline")
    problems.extend(stage_c_problems)
    if problems:
        (out_dir / "pilot.json").write_text(json.dumps(
            plain({"stage_a_c_problems": problems, "rendered": False}),
            indent=1))
        print(json.dumps({"mode": "pilot", "q2": "FAIL",
                          "problems": problems[:20]}, indent=1))
        return 1

    renders = []
    for camera in ("ortho", "perspective"):
        for device in ("CPU", "GPU"):
            for repeat in range(PILOT_REPEATS + 1):
                stem = f"{camera}_{device}_r{repeat}"
                record = render_one(body, camera, device, out_dir, stem)
                record["warm_up"] = repeat == 0
                record["repeat"] = repeat
                renders.append(record)

    # Stage B: Blender's own projection against ours, on the builder's points.
    stage_b_problems = []
    for record in renders:
        if record["warm_up"]:
            continue
        got = record["receipt"]["control_points_raster"]
        for name, point in {"P_C": (0, 0, 0), "P_X": (20, 0, 0),
                            "P_Y": (0, 20, 0), "P_Z": (0, 0, 24),
                            "P_XZ": (20, 0, 24)}.items():
            want = pbr_project.project(body, record["camera"], point)
            for axis, index in (("u", 0), ("v", 1)):
                if abs(got[name][index] - want[index]) > 0.001:
                    stage_b_problems.append(
                        f"stage B {record['camera']} {record['device']} "
                        f"{name}.{axis}: {got[name][index]:.6f} vs "
                        f"{want[index]:.6f}")

    # Q1: within-backend determinism on the canonical decoded pixels.
    determinism = {}
    for camera in ("ortho", "perspective"):
        for device in ("CPU", "GPU"):
            measured = [r for r in renders if r["camera"] == camera
                        and r["device"] == device and not r["warm_up"]]
            digests = [r["image"]["linear_pixel_digest"] for r in measured]
            determinism[f"{camera}/{device}"] = {
                "digests": digests,
                "exact": len(set(digests)) == 1,
                "warm_up_seconds": next(
                    r["wall_seconds"] for r in renders
                    if r["camera"] == camera and r["device"] == device
                    and r["warm_up"]),
                "measured_seconds": [r["wall_seconds"] for r in measured],
            }
    cpu_exact = all(v["exact"] for k, v in determinism.items()
                    if k.endswith("/CPU"))
    gpu_exact = all(v["exact"] for k, v in determinism.items()
                    if k.endswith("/GPU"))
    if cpu_exact and gpu_exact:
        backend = "either; a formal one must still be chosen and not mixed"
    elif cpu_exact:
        backend = "CPU"
    elif gpu_exact:
        backend = "GPU"
    else:
        backend = None

    # Q3: the real recogniser on the two distinct images.
    feasibility = {}
    for camera in ("ortho", "perspective"):
        first = next(r for r in renders if r["camera"] == camera
                     and r["device"] == FORMAL_BACKEND and r["repeat"] == 1)
        feasibility[camera] = feasibility_check(
            body, camera, out_dir / f"{first['stem']}.png")

    q1 = backend is not None
    q2 = not stage_b_problems
    q3 = all(v["pass"] for v in feasibility.values())
    record = {
        "kind": "brickagain.pbr_pilot",
        "not_citable": ("This pilot produces no V2 result. Its images may not "
                        "enter a corpus and its numbers may not be cited."),
        "generation_under_test": GENERATION,
        "scene_index": PILOT_SCENE,
        "renders": len(renders),
        "measured_renders": sum(1 for r in renders if not r["warm_up"]),
        "q1_determinism": determinism,
        "q1_formal_backend": backend,
        "q2_stage_b_problems": stage_b_problems,
        "q3_feasibility": feasibility,
        "verdict": {"q1": q1, "q2": q2, "q3": q3,
                    "pass": bool(q1 and q2 and q3)},
        "runtime": runtime_record(),
        "sources": source_manifests(),
        "render_records": [{k: v for k, v in r.items() if k != "receipt"}
                           for r in renders],
    }
    (out_dir / "pilot.json").write_text(
        json.dumps(plain(record), indent=1, sort_keys=True))
    print(json.dumps({"mode": "pilot", "verdict": record["verdict"],
                      "formal_backend": backend,
                      "determinism": {k: v["exact"] for k, v in
                                      determinism.items()},
                      "stage_b_problems": stage_b_problems[:6],
                      "feasibility": {k: v["pass"] for k, v in
                                      feasibility.items()}}, indent=1))
    return 0 if record["verdict"]["pass"] else 1


def feasibility_check(body: dict, camera: str, png_path: Path) -> dict:
    """Does the recogniser run at all on a physically-rendered image?

    Deliberately not a measurement. Every gate here is a way of being
    obviously broken -- crashing, timing out, finding nothing, saturating the
    proposal stage, landing nowhere near a brick, or failing to read any colour
    at all. Passing means "not obviously broken", which is what has to be true
    before a corpus is worth rendering. It does not mean the corpus is good.
    """
    from src.ui.photo import PHOTO_MULTI, RECOGNISE_CV, analyse_photo
    from src.vision.detect import MAX_BOXES

    started = time.time()
    error = None
    try:
        analysis = analyse_photo(png_path.read_bytes(), mode=PHOTO_MULTI,
                                 method=RECOGNISE_CV)
    except Exception as exc:                      # noqa: BLE001 - reported
        return {"pass": False, "stop": ["S1 unhandled exception"],
                "error": f"{type(exc).__name__}: {exc}"}
    seconds = time.time() - started

    truth = [(brick, pbr_project.top_face_aabb(body, camera, brick))
             for brick in body["bricks"]]
    ious, matched = [], []
    for item in analysis.items:
        best = max(((pbr_project.iou(item.box, box), brick)
                    for brick, box in truth), key=lambda pair: pair[0])
        ious.append(best[0])
        if best[0] >= IOU_FLOOR:
            matched.append((item, best[1]))
    best_iou = max(ious) if ious else 0.0
    saturated = (analysis.diagnostics.get("proposals", 0)
                 + analysis.diagnostics.get("dropped_thin", 0)) >= MAX_BOXES
    unreadable = sum(1 for item, _b in matched
                     if item.index in analysis.colour_problems)

    stop = []
    if seconds > MAX_RECOGNISE_SECONDS:
        stop.append(f"S2 recognise took {seconds:.1f}s")
    if not analysis.items:
        stop.append("S3 no detections")
    if saturated:
        stop.append(f"S4 proposal stage saturated at {MAX_BOXES}")
    if best_iou < IOU_FLOOR:
        stop.append(f"S5 best IoU {best_iou:.3f} below {IOU_FLOOR}")
    if matched and unreadable == len(matched):
        stop.append("S7 every matched item's colour was unreadable")
    return {
        "pass": not stop,
        "stop": stop,
        "seconds": round(seconds, 3),
        "n_items": len(analysis.items),
        "n_true_bricks": len(truth),
        "diagnostics": dict(analysis.diagnostics),
        "best_iou": round(best_iou, 4),
        "n_matched_at_floor": len(matched),
        "colour_unreadable_among_matched": unreadable,
        "recorded_only": {
            "part_correct_among_matched": sum(
                1 for i, b in matched if i.predicted_part == b["part"]),
            "colour_correct_among_matched": sum(
                1 for i, b in matched if i.predicted_colour == b["colour_id"]),
            "part_low_confidence": sum(
                1 for i in analysis.items
                if (i.predicted_confidence or 0) < 0.45),
            "colour_low_confidence": sum(
                1 for i in analysis.items
                if (i.predicted_colour_confidence or 0) < 0.40),
        },
        "note": ("Pass means not obviously broken on one scene and one "
                 "condition. It is not a rate and it does not say the corpus "
                 "is usable."),
    }


# ---------------------------------------------------------------------------
# freeze, run, score, report
# ---------------------------------------------------------------------------

FROZEN_DIR = ROOT / "data/phase_v2/frozen"
RESULTS_DIR = ROOT / "data/phase_v2/results"


def membership() -> list:
    """Every cell of the corpus, named before anything is rendered."""
    from src.vision import synthetic

    scenes = synthetic.scenes()
    cells = []
    for camera, conditions in (("ortho", PAIRED_CONDITIONS + V2_ONLY_CONDITIONS),
                               ("perspective", PERSPECTIVE_CONDITIONS)):
        for condition_id in conditions:
            for scene_index in SCENE_INDICES:
                scene = scenes[scene_index]
                body = condition_table(condition_id, scene_index)
                cells.append({
                    "cell_id": f"{camera}/{condition_id}/{scene.scene_id}",
                    "camera": camera,
                    "condition_id": condition_id,
                    "scene_id": scene.scene_id,
                    "scene_index": scene_index,
                    "paired_with_v1": (camera == "ortho"
                                       and condition_id in PAIRED_CONDITIONS),
                    "image_id": f"{scene.scene_id}/{condition_id}",
                    "width_px": body["raster"]["width_px"],
                    "height_px": body["raster"]["height_px"],
                    "n_true_bricks": len(body["bricks"]),
                    "scene_digest": pbr.scene_digest(body),
                })
    return cells


def mode_freeze(args) -> int:
    """Write the frozen contract, once, and refuse to rewrite it."""
    directory = FROZEN_DIR / GENERATION
    if directory.exists() and not args.force:
        raise Refused(
            f"{directory} already exists. A frozen generation is write-once; "
            "a change needs a new generation, not an overwrite")
    problems, checked = [], 0
    for scene_index in SCENE_INDICES:
        problems.extend(stage_a(condition_table("baseline", scene_index)))
        for condition_id in PAIRED_CONDITIONS + V2_ONLY_CONDITIONS:
            found, rows = stage_c(scene_index, condition_id)
            problems.extend(found)
            checked += len(rows)
    if problems:
        raise Refused(
            f"the projection check has {len(problems)} problems; nothing is "
            f"frozen. First: {problems[0]}")

    cells = membership()
    sources = source_manifests()
    if sources["builder_imports_unmanifested"]:
        raise Refused(
            "the Blender builder imports project modules that neither "
            "manifest covers: "
            f"{sources['builder_imports_unmanifested']}")
    runtime = runtime_record()
    contract = {
        "kind": "brickagain.pbr_contract",
        "generation": GENERATION,
        "scenes": list(SCENE_INDICES),
        "paired_conditions": list(PAIRED_CONDITIONS),
        "v2_only_conditions": list(V2_ONLY_CONDITIONS),
        "perspective_conditions": list(PERSPECTIVE_CONDITIONS),
        "formal_backend": FORMAL_BACKEND,
        "match_iou": _match_iou(),
        "scorer": "src.eval.visual_stress.score_image",
        "recogniser_entry": "src.ui.photo.analyse_photo(mode='multi', "
                            "method='cv-baseline')",
        "n_cells": len(cells),
        "bricks_projection_checked": checked,
        "absent_v1_axes": {
            "background=textured":
                "V1 fills its canvas with a procedural sine texture. The PBR "
                "analogue is a table material with that texture mapped on it, "
                "which needs a UV layout and a texture node this track has "
                "not defined. Absent rather than approximated.",
            "background=cluttered":
                "V1 scatters random discs over its canvas. The PBR analogue "
                "is clutter geometry on the table, which would add objects "
                "the scene record does not describe and whose ground truth "
                "would not be constructed. Absent rather than invented.",
            "lighting=gradient":
                "V1 multiplies a horizontal ramp over the finished image. The "
                "PBR analogue is an off-centre light, which is renderable but "
                "needs its own power calibration against the world-only "
                "baseline; not calibrated in this generation.",
            "lighting=harsh":
                "V1 multiplies a steep radial falloff. The PBR analogue is a "
                "small bright source, which this generation defines for "
                "shadow=hard but has not separately calibrated as a lighting "
                "condition; running both from one calibration would make the "
                "two axes the same condition under two names.",
            "viewpoint=tilt_small":
                "V1 applies a per-row horizontal shear to the finished image. "
                "A PBR camera rotation is a different operation, so pairing "
                "the two would compare a shear against a projection change. "
                "Applying V1's own shear to a PBR base image would pair "
                "exactly and is the right way to run this axis, but it is a "
                "post-process over a render rather than a render, so it is "
                "left to a later generation rather than mixed in here.",
            "viewpoint=tilt_large":
                "The same shear at a larger angle, absent for the same "
                "reason as viewpoint=tilt_small.",
            "quality=downscaled":
                "V1 resamples the finished image down and back up. Like the "
                "viewpoint axis this is a post-process, and running it here "
                "would put a post-processed image in a set whose other cells "
                "are renders.",
            "quality=noisy":
                "V1 adds fixed Gaussian pixel noise to the finished image. "
                "That is a third mechanism, distinct from both sensor noise "
                "and the Monte Carlo estimator noise this track's "
                "render_sampling_noise condition produces, so it must not be "
                "paired with either. Absent here; render_sampling_noise is "
                "reported inside V2 only.",
            "quality=blurred":
                "V1 convolves the finished image. A post-process, absent for "
                "the same reason as quality=downscaled.",
            "quality=compressed":
                "V1 round-trips the finished image through JPEG. A "
                "post-process, absent for the same reason as "
                "quality=downscaled.",
        },
        "claims_forbidden": [
            "V2 - V1 bounds nothing about performance on photographs",
            # Two, not three. The earlier contract said three and the report
            # reprinted it, above a table with two rows: there are two
            # contrasts -- V2 ortho against V1, paired by condition, and V2
            # perspective against V2 ortho -- plus one V2-only condition that
            # is never paired.
            "the two contrasts are not additive and neither isolates a single "
            "mechanism",
            "the perspective contrast includes side faces becoming visible "
            "and the table plane being minified, not distortion alone",
            "44 px/stud holds at the registered plane, which is the brick top "
            "at z=24; the table at z=0 is minified under the perspective "
            "camera",
            "the colour-error contrast is not a claim about PBR shading in "
            "general: an ambient specular floor lifts dark channels by a "
            "measured amount and that mechanism is reported with it",
        ],
    }
    body = {
        "kind": "brickagain.pbr_authorization",
        "generation": GENERATION,
        "contract": contract,
        "contract_digest": pbr.scene_digest(contract),
        "membership": cells,
        "membership_digest": pbr.scene_digest(cells),
        "sources": sources,
        "runtime": runtime,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(ROOT)).stdout.strip(),
        "git_status_short": subprocess.run(
            ["git", "status", "--short"], capture_output=True, text=True,
            cwd=str(ROOT)).stdout.strip().splitlines(),
    }
    body["authorization_digest"] = pbr.scene_digest(body)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "authorization.json").write_bytes(
        json.dumps(plain(body), indent=1, sort_keys=True).encode("utf-8"))
    print(json.dumps({
        "mode": "freeze", "generation": GENERATION,
        "n_cells": len(cells),
        "bricks_projection_checked": checked,
        "contract_digest": body["contract_digest"],
        "membership_digest": body["membership_digest"],
        "authorization_digest": body["authorization_digest"],
        "runtime_digest": runtime["runtime_digest"],
        "renderer_source_manifest_digest":
            sources["renderer_source_manifest_digest"],
        "evaluation_source_manifest_digest":
            sources["evaluation_source_manifest_digest"],
        "git_commit": body["git_commit"],
    }, indent=1))
    return 0


def _match_iou() -> float:
    from src.eval import visual_stress as vs

    return float(vs.MATCH_IOU)


def load_authorization() -> dict:
    path = FROZEN_DIR / GENERATION / "authorization.json"
    if not path.is_file():
        raise Refused(f"{path} does not exist; freeze first")
    body = json.loads(path.read_text())
    recomputed = pbr.scene_digest(
        {k: v for k, v in body.items() if k != "authorization_digest"})
    if recomputed != body["authorization_digest"]:
        raise Refused(
            f"{path} does not recompute: {recomputed} vs "
            f"{body['authorization_digest']}")
    return body


def mode_run(args) -> int:
    """The formal corpus, against the frozen authorisation."""
    grant = load_authorization()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise Refused(f"{out_dir} is not empty; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    live = source_manifests()
    for key in ("renderer_source_manifest_digest",
                "evaluation_source_manifest_digest"):
        if live[key] != grant["sources"][key]:
            raise Refused(
                f"{key} is {live[key]} and the authorisation binds "
                f"{grant['sources'][key]}. An execution-relevant change needs "
                "a new generation, not a run against a stale grant")
    live_runtime = runtime_record()
    if live_runtime["runtime_digest"] != grant["runtime"]["runtime_digest"]:
        raise Refused(
            "the runtime digest does not match the authorisation; Blender or "
            "the machine changed and this needs a new generation")

    records = []
    started = time.time()
    for index, cell in enumerate(grant["membership"], 1):
        body = condition_table(cell["condition_id"], cell["scene_index"])
        if pbr.scene_digest(body) != cell["scene_digest"]:
            raise Refused(
                f"{cell['cell_id']}: the scene record no longer matches the "
                "frozen membership")
        stem = cell["cell_id"].replace("/", "__")
        record = render_one(body, cell["camera"], FORMAL_BACKEND, out_dir, stem)
        # The record against the scene that rendered, in both directions. This
        # is what the receipt existed for: six generations wrote one and no
        # code read it, and every shadow-condition light was aimed 58.992
        # degrees away from the scene centre the whole time. A cell that
        # disagrees with its own record is refused here rather than scored.
        problems = pbr.receipt_problems(body, record["receipt"])
        problems += pbr.scene_digest_problems(
            record["receipt"], (out_dir / f"{stem}.scene.json").read_bytes())
        if problems:
            raise Refused(
                f"{stem}: the scene Blender built disagrees with the record "
                f"that asked for it: {problems}")
        record["cell_id"] = cell["cell_id"]
        record["condition_id"] = cell["condition_id"]
        record["scene_id"] = cell["scene_id"]
        record["scene_index"] = cell["scene_index"]
        record["image_id"] = cell["image_id"]
        record["paired_with_v1"] = cell["paired_with_v1"]
        records.append(record)
        print(f"[{index}/{len(grant['membership'])}] {cell['cell_id']} "
              f"{record['wall_seconds']}s", flush=True)
    total = time.time() - started

    manifest = {
        "kind": "brickagain.pbr_run",
        "generation": GENERATION,
        "authorization_digest": grant["authorization_digest"],
        "backend": FORMAL_BACKEND,
        "n_cells": len(records),
        "total_seconds": round(total, 1),
        "cells": [{k: v for k, v in r.items() if k != "receipt"}
                  for r in records],
        "receipts": {r["cell_id"]: r["receipt"] for r in records},
        "runtime": live_runtime,
    }
    manifest["run_digest"] = pbr.scene_digest(
        {k: v for k, v in manifest.items() if k != "receipts"})
    (out_dir / "run.json").write_text(
        json.dumps(plain(manifest), indent=1, sort_keys=True))
    print(json.dumps({"mode": "run", "n_cells": len(records),
                      "total_seconds": manifest["total_seconds"],
                      "run_digest": manifest["run_digest"]}, indent=1))
    return 0


def mode_score(args) -> int:
    """Scores, derived from the archived images by V1's own scorer."""
    grant = load_authorization()
    out_dir = Path(args.out_dir)
    run = json.loads((out_dir / "run.json").read_text())
    if run["authorization_digest"] != grant["authorization_digest"]:
        raise Refused("the run was not produced under this authorisation")

    from src.training.session import sha256_file

    rows = []
    for cell in run["cells"]:
        png = out_dir / f"{cell['stem']}.png"
        stored = sha256_file(png)
        if stored != cell["image"]["png_sha256"]:
            raise Refused(
                f"{png} hashes to {stored} and the run recorded "
                f"{cell['image']['png_sha256']}. An archived image that has "
                "changed is tampering, not tolerance")
        row = score_row(cell["scene_index"], cell["condition_id"], png)
        row["cell_id"] = cell["cell_id"]
        row["camera"] = cell["camera"]
        row["scene_id"] = cell["scene_id"]
        row["paired_with_v1"] = cell["paired_with_v1"]
        row["png_sha256"] = stored
        row["linear_pixel_digest"] = cell["image"]["linear_pixel_digest"]
        row["clipped_above"] = cell["image"]["clipped_above"]
        rows.append(row)

    by_condition = {}
    for row in rows:
        key = f"{row['camera']}/{row['condition_id']}"
        bucket = by_condition.setdefault(key, {
            "n": 0, "true_bricks": 0, "detected": 0, "false_negatives": 0,
            "false_positives": 0, "abstentions": 0, "part_errors": 0,
            "colour_errors": 0, "inventory_exact_match": 0,
            "top1_correct_among_matched": 0, "matched": 0})
        bucket["n"] += 1
        bucket["true_bricks"] += row["n_true_bricks"]
        bucket["detected"] += row["n_detected"]
        bucket["false_negatives"] += row["false_negatives"]
        bucket["false_positives"] += row["false_positives"]
        bucket["abstentions"] += row["abstentions"]
        bucket["part_errors"] += row["n_part_errors"]
        bucket["colour_errors"] += row["n_colour_errors"]
        bucket["inventory_exact_match"] += int(row["inventory_exact_match"])
        bucket["top1_correct_among_matched"] += row["top1_correct_among_matched"]
        # This was initialised to 0 and never incremented, so scores.json
        # published "matched": 0 for every condition beside
        # top1_correct_among_matched values of 15 to 35 -- a denominator of
        # zero next to its own numerator.
        bucket["matched"] += row["n_detected"] - row["false_positives"]
        bucket["paired_with_v1"] = bool(row["paired_with_v1"])

    by_colour = {}
    for row in rows:
        if row["camera"] != "ortho" or row["condition_id"] in V2_ONLY_CONDITIONS:
            continue
        for entry in row["colour_offsets"]:
            bucket = by_colour.setdefault(entry["colour_id"], {
                "n": 0, "palette_srgb": entry["palette_srgb"],
                "offset_sum": [0, 0, 0], "saturated_sum": 0.0})
            bucket["n"] += 1
            bucket["offset_sum"] = [a + b for a, b in
                                    zip(bucket["offset_sum"], entry["offset_codes"])]
            bucket["saturated_sum"] += entry["fraction_saturated"]
    colour_offset_by_colour = {
        name: {
            "n_bricks": b["n"],
            "palette_srgb": b["palette_srgb"],
            "mean_offset_codes": [round(v / b["n"], 2) for v in b["offset_sum"]],
            "worst_channel_offset_codes": round(max(b["offset_sum"]) / b["n"], 2),
            "mean_fraction_saturated": round(b["saturated_sum"] / b["n"], 6),
        } for name, b in sorted(by_colour.items())}

    v1 = v1_rows()
    scores = {
        "kind": "brickagain.pbr_scores",
        "generation": GENERATION,
        "colour_offset_by_colour": colour_offset_by_colour,
        "authorization_digest": grant["authorization_digest"],
        "run_digest": run["run_digest"],
        "scorer": "src.eval.visual_stress.score_image",
        "match_iou": _match_iou(),
        "n_rows": len(rows),
        "rows": rows,
        "by_condition": by_condition,
        "paired_contrast_v2_minus_v1": paired_contrast(rows, v1),
        "camera_contrast_perspective_minus_ortho": camera_contrast(rows),
        "v2_only_conditions": list(V2_ONLY_CONDITIONS),
        "claims_forbidden": grant["contract"]["claims_forbidden"],
    }
    scores["scores_digest"] = pbr.scene_digest(
        {k: v for k, v in scores.items() if k != "rows"})
    (out_dir / "scores.json").write_text(
        json.dumps(plain(scores), indent=1, sort_keys=True))
    print(json.dumps({
        "mode": "score", "n_rows": len(rows),
        "scores_digest": scores["scores_digest"],
        "paired": scores["paired_contrast_v2_minus_v1"]["n_pairs"],
        "camera_pairs": scores["camera_contrast_perspective_minus_ortho"]["n_pairs"],
    }, indent=1))
    return 0


def mode_report(args) -> int:
    out_dir = Path(args.out_dir)
    scores = json.loads((out_dir / "scores.json").read_text())
    grant = load_authorization()
    text = render_report(scores, grant)
    path = out_dir / "pbr_report.md"
    path.write_text(text, encoding="utf-8")
    print(json.dumps({"mode": "report", "path": str(path),
                      "bytes": len(text.encode("utf-8"))}, indent=1))
    return 0


def render_report(scores: dict, grant: dict) -> str:
    paired = scores["paired_contrast_v2_minus_v1"]
    camera = scores["camera_contrast_perspective_minus_ortho"]
    lines = [
        f"# V2 PBR visual stress: {scores['generation']}",
        "",
        "Every figure here is derived from the archived canonical PNGs by "
        "`src.eval.visual_stress.score_image` -- V1's own scorer, so the "
        "paired contrast is measured by one ruler rather than two.",
        "",
        f"* authorisation `{grant['authorization_digest'][:16]}...`",
        f"* run `{scores['run_digest'][:16]}...`",
        f"* scores `{scores['scores_digest'][:16]}...`",
        f"* backend **{grant['contract']['formal_backend']}**, chosen because "
        "three measured CPU renders were byte-identical and three Metal "
        "renders were not",
        f"* {scores['n_rows']} images, {paired['n_pairs']} of them paired "
        "against V1",
        "",
        "## PBR rendering-domain effect (V2 ortho - V1, paired)",
        "",
        "Rows marked *(mean)* are already per-image averages, so the two "
        "columns are means of those rather than sums -- the total of eighty "
        "averages is not a quantity.",
        "",
        "| quantity | V2 | V1 | mean delta | V2 worse | V2 better | equal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, block in paired["metrics"].items():
        label = f"`{name}` *(mean)*" if block["is_already_a_mean"] else f"`{name}`"
        lines.append(
            f"| {label} | {block['v2_total']} | {block['v1_total']} | "
            f"{block['mean_delta']} | {block['n_v2_worse']} | "
            f"{block['n_v2_better']} | {block['n_equal']} |")
    exact = paired["inventory_exact_match"]
    lines += [
        "",
        f"`inventory_exact_match`: V2 {exact['v2']}/{exact['denominator']}, "
        f"V1 {exact['v1']}/{exact['denominator']}.",
        "",
        "**This contrast is a set of changes, not one mechanism.** Light "
        "transport, real cast shadow, bevelled edge highlights, mesh "
        "rasterisation, the antialiasing filter, the colour pipeline and "
        "sampling noise all change together. The name is for the whole set.",
        "",
        "## perspective-camera effect (V2 perspective - V2 ortho)",
        "",
        f"{camera['n_pairs']} paired cells.",
        "",
        "| quantity | perspective | ortho | mean delta |",
        "|---|---:|---:|---:|",
    ]
    for name, block in camera["metrics"].items():
        label = f"`{name}` *(mean)*" if block["is_already_a_mean"] else f"`{name}`"
        lines.append(f"| {label} | {block['persp_total']} | "
                     f"{block['ortho_total']} | {block['mean_delta']} |")
    lines += [
        "",
        "This contrast includes non-parallel rays, **side faces becoming "
        "visible** on off-centre bricks, occlusion relations changing, and the "
        "table plane being minified. It is not distortion alone.",
        "",
        "The perspective camera registers the **brick top plane** at z=24, "
        "where V1's own truth boxes are, so one scorer measures both cameras "
        f"and every brick top lands on V1's box to {camera['registration_px']} "
        "px. The table at z=0 is therefore minified instead: "
        f"{camera['table_minification_percent_min']} to "
        f"{camera['table_minification_percent_max']} per cent across the eight "
        "cells, since the camera distance is per-scene. An earlier design "
        "registered the *table* and magnified every brick top by 5.5 to 12.2 "
        "per cent against a truth box that had not moved, which put the mean "
        "IoU between the projected top face and V1's box at 0.513 and pushed "
        "7 of 52 bricks below the 0.3 matching threshold: the arm was "
        "measuring its own registration error.",
        "",
        "## by condition",
        "",
        "`paired` marks the rows that make up the "
        f"{paired['n_pairs']}-cell contrast above. The other rows are not part "
        "of it: `render_sampling_noise` is V2-only and deliberately never "
        "paired, and every `perspective/` row belongs to the second contrast.",
        "",
        "| cell | paired | n | true | detected | matched | FN | FP | abstentions | part err | colour err | exact |",
        "|---|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(scores["by_condition"]):
        b = scores["by_condition"][key]
        lines.append(
            f"| `{key}` | {'yes' if b.get('paired_with_v1') else 'no'} | "
            f"{b['n']} | {b['true_bricks']} | {b['detected']} | "
            f"{b['matched']} | "
            f"{b['false_negatives']} | {b['false_positives']} | "
            f"{b['abstentions']} | {b['part_errors']} | {b['colour_errors']} | "
            f"{b['inventory_exact_match']} |")
    lines += [
        "",
        "## why the colour errors move",
        "",
        "Measured on the paired ortho cells: what each brick's top face renders "
        "at, against the exact palette RGB V1 fills it with. A dielectric under "
        "a uniform dome carries a near-constant specular floor on top of its "
        "albedo, so bright channels barely move and dark ones are lifted. This "
        "is the mechanism behind the colour contrast, and it is a property of "
        "*this environment* as much as of PBR shading: a different dome would "
        "move it.",
        "",
        "| colour | palette | mean offset (codes) | worst channel | saturated |",
        "|---|---|---|---:|---:|",
    ]
    for name, block in scores["colour_offset_by_colour"].items():
        lines.append(
            f"| `{name}` | {tuple(block['palette_srgb'])} | "
            f"{tuple(block['mean_offset_codes'])} | "
            f"{block['worst_channel_offset_codes']} | "
            f"{100 * block['mean_fraction_saturated']:.1f}% |")
    lines += [
        "",
        "## what this does not say",
        "",
    ]
    for claim in scores["claims_forbidden"]:
        lines.append(f"* {claim}")
    lines += [
        "* V2-only conditions "
        f"({', '.join(scores['v2_only_conditions'])}) are reported inside V2 "
        "and never paired against V1.",
        "",
        "## V1 axes absent from this generation",
        "",
    ]
    for axis, why in sorted(grant["contract"]["absent_v1_axes"].items()):
        lines.append(f"* `{axis}` -- {why}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# calibration: derived, archived, and re-checkable
# ---------------------------------------------------------------------------

#: The cell the illumination is calibrated on. A real corpus scene rather than
#: a toy, so the measurements are about the images that get scored.
CALIBRATION_SCENE = 0

#: How wide a border strip counts as "table", in pixels. The strip is outside
#: every brick footprint by construction: V1's canvas has a 1.5-stud margin.
BORDER_PX = 6


def measure_render(body: dict, exr_path: Path, png_path: Path,
                   png_report: dict) -> dict:
    """What one render actually came out at, in linear and in 8-bit.

    Every number ``CONDITION_LIGHTING`` records is produced here. They used to
    be typed into the contract from ad-hoc development renders: no script
    derived them, no artefact held the images, and no test compared them --
    while sitting inside ``illumination_calibration`` and therefore inside
    every scene digest. Two of them were wrong (0.000000 clip on the two
    conditions that clip the most), and nothing could have said so.
    """
    import numpy as np
    from PIL import Image

    linear = pbr_colour.read_exr(exr_path)[:, :, :3]
    eight = np.asarray(Image.open(png_path).convert("RGB")).astype(int)
    height, width = eight.shape[:2]
    w = BORDER_PX
    strip = np.concatenate([
        eight[:w].reshape(-1, 3), eight[-w:].reshape(-1, 3),
        eight[:, :w].reshape(-1, 3), eight[:, -w:].reshape(-1, 3)])
    lin_strip = np.concatenate([
        linear[:w].reshape(-1, 3), linear[-w:].reshape(-1, 3),
        linear[:, :w].reshape(-1, 3), linear[:, -w:].reshape(-1, 3)])

    patch = 24
    corners = [linear[:patch, :patch], linear[:patch, -patch:],
               linear[-patch:, :patch], linear[-patch:, -patch:]]
    means = [float(c.mean()) for c in corners]

    inside = np.zeros((height, width), dtype=bool)
    for brick in body["bricks"]:
        u0, v0, u1, v1 = pbr_project.top_face_aabb(body, "ortho", brick)
        inside[max(0, int(v0)):int(v1) + 1, max(0, int(u0)):int(u1) + 1] = True
    clipped = linear > 1.0
    total_clipped = int(clipped.sum())
    on_bricks = int((clipped & inside[:, :, None]).sum())

    return {
        "table_median_srgb": [int(v) for v in np.median(strip, axis=0)],
        "table_median_linear": round(float(np.median(lin_strip)), 6),
        "corner_mean_linear": [round(v, 6) for v in means],
        "corner_uniformity_ratio": round(max(means) / min(means), 4),
        "clip_percent": round(100.0 * png_report["clipped_above"]
                              / (height * width * 3), 6),
        "clipped_components": int(png_report["clipped_above"]),
        "clipped_components_on_bricks": on_bricks,
        "clipped_components_on_table": total_clipped - on_bricks,
        "linear_max": round(float(linear.max()), 6),
        "linear_min": round(float(linear.min()), 6),
        "resolution": [width, height],
    }


def _render_measured(body: dict, out_dir: Path, stem: str) -> dict:
    record = render_one(body, "ortho", FORMAL_BACKEND, out_dir, stem)
    measured = measure_render(body, out_dir / f"{stem}.exr",
                              out_dir / f"{stem}.png", record["image"])
    return {"stem": stem, "seconds": record["wall_seconds"],
            "receipt_problems": pbr.receipt_problems(body, record["receipt"]),
            "measured": measured}


def mode_calibrate(args) -> int:
    """Derive the illumination the contract records, by measurement.

    The table is Lambertian and lit only by the dome in the no-shadow
    conditions, so its rendered radiance is *linear* in the dome strength: one
    render fixes the constant and the required strength follows. Under a shadow
    condition the light adds a second linear term, so two renders -- power 0
    and a probe power -- fix both, and the power that lands the table on V1's
    own grey is solved rather than tuned. A solve is re-runnable; a tuning is a
    number somebody once typed.
    """
    out_dir = Path(args.out_dir) / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = pbr.srgb_to_linear_scalar(pbr.CONDITION_BACKGROUND["plain"])
    results = {"kind": "brickagain.pbr_calibration",
               "generation": GENERATION,
               "scene_index": CALIBRATION_SCENE,
               "target_plain_table_linear": round(target, 6),
               "target_plain_table_srgb": pbr.CONDITION_BACKGROUND["plain"],
               "conditions": {}, "solves": {}}

    for condition_id in ("baseline", "lighting=dim", "background=grey",
                         "background=dark", "shadow=soft", "shadow=hard"):
        body = pbr.build_scene(CALIBRATION_SCENE, condition_id)
        stem = f"confirm__{condition_id.replace('=', '-')}"
        entry = _render_measured(body, out_dir, stem)
        entry["world_strength"] = body["world"]["strength"]
        entry["lights"] = [{"size_scene_units": l["size_scene_units"],
                            "power_watts": l["power_watts"]}
                           for l in body["lights"]]
        results["conditions"][condition_id] = entry
        print(json.dumps({"calibrated": condition_id,
                          "table_srgb": entry["measured"]["table_median_srgb"],
                          "clip_percent": entry["measured"]["clip_percent"],
                          "receipt_problems": entry["receipt_problems"]}),
              flush=True)

    results["bare_table"] = {}
    for background, condition_id in (("plain", "baseline"),
                                     ("grey", "background=grey"),
                                     ("dark", "background=dark")):
        body = pbr.build_scene(CALIBRATION_SCENE, condition_id)
        bare = json.loads(json.dumps(body))
        bare["bricks"] = []
        stem = f"bare__{background}"
        entry = _render_measured(bare, out_dir, stem)
        entry["v1_fill_srgb"] = pbr.CONDITION_BACKGROUND[background]
        with_bricks = results["conditions"][condition_id]["measured"][
            "table_median_srgb"]
        entry["cost_of_ambient_occlusion_codes"] = [
            int(a - b) for a, b in
            zip(entry["measured"]["table_median_srgb"], with_bricks)]
        results["bare_table"][background] = entry
        print(json.dumps({"bare_table": background,
                          "table_srgb": entry["measured"]["table_median_srgb"],
                          "v1_fill": entry["v1_fill_srgb"],
                          "occlusion_cost_codes":
                              entry["cost_of_ambient_occlusion_codes"]}),
              flush=True)

    for condition_id, probe in (("shadow=soft", 70_000_000.0),
                                ("shadow=hard", 96_000_000.0)):
        body = pbr.build_scene(CALIBRATION_SCENE, condition_id)
        dark = json.loads(json.dumps(body))
        dark["lights"] = []
        a = _render_measured(dark, out_dir, f"solve__{condition_id.replace('=', '-')}__p0")
        probed = json.loads(json.dumps(body))
        probed["lights"][0]["power_watts"] = pbr.decimal_str(probe)
        b = _render_measured(probed, out_dir,
                             f"solve__{condition_id.replace('=', '-')}__probe")
        base = a["measured"]["table_median_linear"]
        gain = (b["measured"]["table_median_linear"] - base) / probe
        solved = (target - base) / gain if gain else None
        results["solves"][condition_id] = {
            "world_strength": body["world"]["strength"],
            "table_linear_at_zero_power": base,
            "probe_power_watts": pbr.decimal_str(probe),
            "table_linear_at_probe": b["measured"]["table_median_linear"],
            "linear_per_watt": gain,
            "solved_power_watts": (pbr.decimal_str(solved)
                                   if solved is not None else None),
            "recorded_power_watts": body["lights"][0]["power_watts"],
            "renders": [a, b],
        }
        print(json.dumps({"solved": condition_id,
                          "solved_power_watts": results["solves"][condition_id]
                          ["solved_power_watts"],
                          "recorded_power_watts": body["lights"][0]["power_watts"]}),
              flush=True)

    path = out_dir / "calibration.json"
    path.write_text(json.dumps(plain(results), indent=1, sort_keys=True) + "\n")
    print(json.dumps({"mode": "calibrate", "path": str(path),
                      "digest": pbr.scene_digest(plain(results))}, indent=1))
    return 0


def calibration_problems(calibration: dict) -> list:
    """Where the contract's recorded illumination disagrees with measurement."""
    problems = []
    for condition_id, entry in sorted(calibration["conditions"].items()):
        from src.vision import synthetic
        conditions = pbr._condition(synthetic, condition_id)
        row = pbr.CONDITION_LIGHTING[pbr.lighting_key(conditions)]
        want_srgb = row["measured_table_srgb_on_scene_00"]
        got = entry["measured"]["table_median_srgb"]
        background = conditions["background"]
        # A regression check: the contract records what the render measured, so
        # any later change that moves a pixel shows up here. Only for the plain
        # table, because CONDITION_LIGHTING is keyed by lighting/shadow and the
        # background variants share the even/none row.
        if background == "plain":
            if max(abs(a - b) for a, b in zip(want_srgb, got)) > 1:
                problems.append(
                    f"{condition_id}: the contract records the table at sRGB "
                    f"{want_srgb} and it renders at {got}")
        want_clip = pbr.parse_decimal(row["measured_clip_percent_on_scene_00"])
        got_clip = entry["measured"]["clip_percent"]
        if abs(got_clip - want_clip) > 0.05:
            problems.append(
                f"{condition_id}: the contract records {want_clip} per cent "
                f"clipped on the calibration scene and it clips {got_clip}")
        if entry["receipt_problems"]:
            problems.append(f"{condition_id}: {entry['receipt_problems']}")
    # The design check, on the quantity it is actually about: a bare table,
    # with no bricks to occlude the dome, has to render at V1's own canvas
    # fill. Exact to a code, not to a tolerance -- a Lambertian surface under a
    # uniform dome renders at its albedo, and that is the whole reason the
    # table is Lambertian. Before that change this check failed by 21 codes on
    # `dark`, and the `background` axis was differing in the renderer *and* in
    # the grey. What the bricks then cost is measured separately below and
    # reported rather than tolerated.
    for background, entry in sorted(calibration["bare_table"].items()):
        expect = pbr.CONDITION_BACKGROUND[background]
        got = entry["measured"]["table_median_srgb"]
        if max(abs(v - expect) for v in got) > 1:
            problems.append(
                f"bare table, background={background}: V1 fills its canvas "
                f"with {expect} and the table renders at {got}")
    for condition_id, solve in sorted(calibration["solves"].items()):
        recorded = pbr.parse_decimal(solve["recorded_power_watts"])
        solved = pbr.parse_decimal(solve["solved_power_watts"])
        if abs(recorded - solved) / max(1.0, abs(solved)) > 0.01:
            problems.append(
                f"{condition_id}: the contract records {recorded} watts and "
                f"the two-point solve says {solved}")
    return problems


MODES = {
    "calibrate": mode_calibrate,
    "check-projection": mode_check_projection,
    "pilot": mode_pilot,
    "freeze": mode_freeze,
    "run": mode_run,
    "score": mode_score,
    "report": mode_report,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--out-dir", default=str(ROOT / "runs/pbr" / GENERATION))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        return MODES[args.mode](args)
    except Refused as exc:
        print(json.dumps({"refused": str(exc)}, indent=1), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
