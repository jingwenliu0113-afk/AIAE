"""The V2 PBR track, checked against the artefacts and V1's own constants.

Three shapes of check, because they fail differently.

* **The record's rules.** A canonical decimal string has exactly one spelling
  and a schema is a closed set of fields, so the validator is driven by a table
  of things it must refuse. Every row is a way a record could be written twice
  and hash differently.
* **The chain to V1.** V1's layout, its 44 px/stud and its palette are read,
  never transcribed, and the projected footprint of every brick of every scene
  under every condition has to reproduce what V1's own renderer draws. This is
  the check that would catch a mirrored scene, a swapped axis or a half-canvas
  offset -- all of which render fine and pair wrongly.
* **The isolation.** ``src/vision/**`` is denied to the Phase 3C pack by name.
  Adding this track must not change ``pack_digest``, because gen09 has already
  executed against it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ONLY = "artifact-only:"

from src.vision.pbr import contract as pbr           # noqa: E402
from src.vision.pbr import project as pbr_project    # noqa: E402


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location(
        "pbr_cli", ROOT / "scripts" / "62_visual_stress_pbr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def v3cli():
    spec = importlib.util.spec_from_file_location(
        "v3_cli", ROOT / "scripts" / "63_v3_diffusion_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def archive_cli():
    """The post-hoc verifier, which is outside every frozen manifest."""
    spec = importlib.util.spec_from_file_location(
        "pbr_archive_cli", ROOT / "scripts" / "64_pbr_archive.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def v3archive_cli():
    spec = importlib.util.spec_from_file_location(
        "v3_archive_cli", ROOT / "scripts" / "65_v3_archive.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# canonical decimal strings
# ---------------------------------------------------------------------------

ACCEPTED = ("0.000000", "1.000000", "-1.000000", "-0.000001", "0.100000",
            "10.000000", "460.000000", "0.921582")

REFUSED = (
    ("1.0", "fewer than six decimal places"),
    ("1.5", "fewer than six decimal places"),
    ("1.5000000", "more than six decimal places"),
    ("1", "no decimal point"),
    ("1e0", "exponent"),
    ("1E0", "exponent"),
    ("1.000000e0", "exponent"),
    ("+1.000000", "leading plus"),
    ("01.000000", "leading zero"),
    ("-0.000000", "negative zero is the same number written twice"),
    (" 1.000000", "leading space"),
    ("1.000000 ", "trailing space"),
    ("NaN", "not a number"),
    ("Infinity", "not finite"),
    ("-Infinity", "not finite"),
    ("", "empty"),
    (".000000", "no integer part"),
    ("1.", "no fraction digits"),
)


@pytest.mark.parametrize("text", ACCEPTED)
def test_the_accepted_spellings_are_accepted(text):
    assert pbr.is_canonical_decimal(text)
    assert pbr.parse_decimal(text) == float(text)


@pytest.mark.parametrize("text,why", REFUSED)
def test_a_non_canonical_decimal_is_refused(text, why):
    assert not pbr.is_canonical_decimal(text), why
    with pytest.raises(pbr.ContractError):
        pbr.parse_decimal(text)


@pytest.mark.parametrize("value", [1.0, 1, 0, -1, None, True, [1.0], {"a": 1}])
def test_a_non_string_is_refused_whatever_it_holds(value):
    assert not pbr.is_canonical_decimal(value)


def test_negative_zero_is_written_as_zero():
    assert pbr.decimal_str(-0.0) == "0.000000"
    assert pbr.decimal_str(0.0) == "0.000000"


def test_the_written_form_is_always_accepted_back():
    for value in (0.0, -0.0, 1.0, -1.5, 0.921582, 460.0, 1e-7, -1e-7):
        assert pbr.is_canonical_decimal(pbr.decimal_str(value))


# ---------------------------------------------------------------------------
# canonical serialisation
# ---------------------------------------------------------------------------

def test_the_same_record_serialises_to_the_same_bytes():
    one = pbr.build_scene(0, "baseline")
    two = pbr.build_scene(0, "baseline")
    assert pbr.canonical_bytes(one) == pbr.canonical_bytes(two)
    assert pbr.scene_digest(one) == pbr.scene_digest(two)


def test_key_order_does_not_change_the_digest():
    body = pbr.build_scene(0, "baseline")
    shuffled = dict(reversed(list(body.items())))
    assert pbr.scene_digest(shuffled) == pbr.scene_digest(body)


def test_the_record_round_trips_through_json():
    body = pbr.build_scene(3, "baseline")
    again = json.loads(pbr.canonical_bytes(body).decode("utf-8"))
    assert again == body
    assert pbr.canonical_bytes(again) == pbr.canonical_bytes(body)


def test_a_changed_value_changes_the_digest():
    body = pbr.build_scene(0, "baseline")
    before = pbr.scene_digest(body)
    body["bricks"][0]["translation_scene_units"][0] += 1
    assert pbr.scene_digest(body) != before


def test_the_pixel_digest_separates_shapes_with_the_same_bytes():
    numpy = pytest.importorskip("numpy")
    flat = numpy.zeros((2, 6, 4), dtype=numpy.float32)
    tall = numpy.zeros((4, 3, 4), dtype=numpy.float32)
    assert flat.nbytes == tall.nbytes
    assert pbr.pixel_digest(flat) != pbr.pixel_digest(tall)


def test_the_pixel_digest_refuses_an_array_that_is_not_an_image():
    numpy = pytest.importorskip("numpy")
    with pytest.raises(pbr.ContractError):
        pbr.pixel_digest(numpy.zeros((4, 4), dtype=numpy.float32))


def test_half_and_single_precision_hash_the_same_when_the_values_are_equal():
    numpy = pytest.importorskip("numpy")
    half = numpy.full((2, 2, 4), 0.5, dtype=numpy.float16)
    single = numpy.full((2, 2, 4), 0.5, dtype=numpy.float32)
    assert pbr.pixel_digest(half) == pbr.pixel_digest(single)


# ---------------------------------------------------------------------------
# the chain to V1
# ---------------------------------------------------------------------------

def test_the_writer_reads_v1s_constants_rather_than_a_copy():
    """Every number the record shares with V1 is V1's, by value.

    Not by parsing source text -- a regex over a module is a test that breaks
    when a comment moves and passes when a constant is redefined below the
    line it matched. The modules are imported and the values compared.
    """
    from src.colour import palette
    from src.rendering import ldr
    from src.vision import synthetic

    body = pbr.build_scene(6, "baseline")
    assert body["raster"]["target_px_per_stud"] == synthetic.STUD_PX
    assert body["units"]["ldu_stud"] == ldr.LDU_STUD
    assert body["units"]["ldu_brick"] == ldr.LDU_BRICK

    scene = synthetic.scenes()[6]
    for brick, placement in zip(body["bricks"], scene.placements):
        assert brick["part"] == placement.part
        assert brick["turn"] == placement.turn
        assert tuple(brick["footprint_studs"]) == placement.extents()
        assert brick["colour_id"] == placement.colour_id
        assert tuple(brick["colour_srgb_uint8"]) == tuple(
            palette.colour(placement.colour_id).rgb)


def test_every_scene_colour_converts_to_the_linear_value_the_record_uses():
    from src.colour import palette
    from src.vision import synthetic

    for colour_id in synthetic.SCENE_COLOURS:
        rgb = palette.colour(colour_id).rgb
        expected = [pbr.decimal_str(pbr.srgb_to_linear_scalar(v)) for v in rgb]
        assert all(pbr.is_canonical_decimal(v) for v in expected)
        # And the transfer function is the piecewise standard, not gamma 2.2.
        for value, text in zip(rgb, expected):
            channel = value / 255.0
            want = (channel / 12.92 if channel <= 0.04045
                    else ((channel + 0.055) / 1.055) ** 2.4)
            assert abs(float(text) - want) < 5e-7


def test_the_transfer_functions_are_inverses():
    for u8 in range(0, 256, 7):
        linear = pbr.srgb_to_linear_scalar(u8)
        back = pbr.linear_to_srgb_scalar(linear) * 255.0
        assert abs(back - u8) < 1e-4


def test_v1s_own_positions_are_all_exact_whole_ldu():
    """The occlusion shifts included: 0.55 studs is 11 LDU, 1.15 is 23."""
    from src.vision import synthetic

    for scene in synthetic.scenes():
        for condition_id in ("baseline", "occlusion=partial", "occlusion=heavy"):
            condition = pbr._condition(synthetic, condition_id)
            for placement, x, y in synthetic.layout(scene, condition):
                across, down = placement.extents()
                for studs in (x, y + down):
                    assert abs(studs * 20 - round(studs * 20)) < 1e-9


def test_a_layout_that_is_not_whole_ldu_is_refused():
    with pytest.raises(pbr.ContractError):
        pbr._ldu(0.001, "a position that is not whole LDU")


def test_the_canvas_offset_is_a_canonical_decimal_pair():
    for width, height in ((1012, 748), (618, 836), (484, 220)):
        offset = pbr.canvas_centre_ldu(width, height)
        assert all(pbr.is_canonical_decimal(v) for v in offset)
        assert abs(pbr.parse_decimal(offset[0]) - width * 5 / 22) < 5e-7


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------

def test_the_scene_origin_is_the_canvas_corner(cli):
    body = pbr.build_scene(6, "baseline")
    u, v = pbr_project.project(body, "ortho", (0, 0, 0))
    assert abs(u) < 1e-6 and abs(v) < 1e-6


def test_plus_y_is_image_up_not_down():
    """The sign that would mirror every scene and still render."""
    body = pbr.build_scene(6, "baseline")
    _u0, v0 = pbr_project.project(body, "ortho", (0, 0, 0))
    _u1, v1 = pbr_project.project(body, "ortho", (0, 20, 0))
    assert v1 < v0


def test_one_stud_is_the_projection_ratio_not_the_physical_one():
    body = pbr.build_scene(6, "baseline")
    u0, _v = pbr_project.project(body, "ortho", (0, 0, 0))
    u1, _v = pbr_project.project(body, "ortho", (20, 0, 0))
    assert abs((u1 - u0) - 44.0) < 1e-6


def test_the_orthographic_camera_has_no_depth_scaling():
    body = pbr.build_scene(6, "baseline")
    low = pbr_project.project(body, "ortho", (100, -100, 0))
    high = pbr_project.project(body, "ortho", (100, -100, 24))
    assert low == pytest.approx(high, abs=1e-9)


def test_the_perspective_camera_registers_the_brick_top_plane():
    """The scale is 1.0 at z=24, and the table is what moves.

    Registering the table instead put every brick top 5.5 to 12.2 per cent
    outside a truth box that had not moved. The camera therefore stands one
    brick height further back than the span, and the *table* is minified.
    """
    body = pbr.build_scene(6, "baseline")
    span = pbr.parse_decimal(body["cameras"]["ortho"]["ortho_scale"])
    distance = pbr.parse_decimal(
        body["cameras"]["perspective"]["location_scene_units"][2])
    assert span == 460.0
    assert distance == span + pbr.BRICK_TOP_Z == 484.0
    assert body["cameras"]["perspective"]["reference_plane_z"] == 24
    assert pbr_project.depth_scale(body, "perspective", 24) == 1.0
    table = pbr_project.depth_scale(body, "perspective", 0)
    assert abs(table - 460.0 / 484.0) < 1e-12
    assert abs(44.0 * table - 41.81818181818182) < 1e-9


def test_the_registered_plane_is_read_and_not_assumed():
    """A record that registers the table has to scale the table by 1.0.

    The formula and the record have to agree, and the only way to check that is
    to change the record and require the formula to follow. If ``depth_scale``
    hardcoded z=0 this test would still pass and the previous one would fail;
    if it hardcoded z=24 this one fails.
    """
    body = pbr.build_scene(6, "baseline")
    body["cameras"]["perspective"]["reference_plane_z"] = 0
    assert pbr_project.depth_scale(body, "perspective", 0) == 1.0
    assert pbr_project.depth_scale(body, "perspective", 24) > 1.0


def test_a_point_at_or_behind_the_camera_is_refused():
    body = pbr.build_scene(6, "baseline")
    distance = pbr.parse_decimal(
        body["cameras"]["perspective"]["location_scene_units"][2])
    with pytest.raises(pbr.ContractError):
        pbr_project.depth_scale(body, "perspective", distance)


def test_an_unknown_camera_is_refused():
    body = pbr.build_scene(6, "baseline")
    with pytest.raises(pbr.ContractError):
        pbr_project.project(body, "isometric", (0, 0, 0))


def test_the_projection_check_holds_for_every_scene_and_condition(cli):
    """Stage A and Stage C over the whole frozen condition set.

    468 bricks. Stage C is the strongest single check in this track: it takes
    each brick from V1's layout through the writer, the record and the camera,
    and requires the orthographic footprint to reproduce what V1's own
    renderer draws, to a thousandth of a pixel.
    """
    problems, checked = [], 0
    conditions = cli.PAIRED_CONDITIONS + cli.V2_ONLY_CONDITIONS
    for scene_index in cli.SCENE_INDICES:
        problems.extend(cli.stage_a(cli.condition_table("baseline", scene_index)))
        for condition_id in conditions:
            found, rows = cli.stage_c(scene_index, condition_id)
            problems.extend(found)
            checked += len(rows)
    assert problems == []
    assert checked == 468


def test_stage_a_catches_a_flipped_y_axis(cli):
    """The check has to fail when the thing it checks is broken."""
    body = pbr.build_scene(6, "baseline")
    body["raster"]["axis_map"]["scene_plus_y"] = "raster_plus_v"
    original = pbr_project.project

    def mirrored(record, camera, point):
        u, v = original(record, camera, point)
        return u, record["raster"]["height_px"] - v

    pbr_project.project = mirrored
    try:
        assert cli.stage_a(body) != []
    finally:
        pbr_project.project = original


def test_stage_c_catches_a_brick_moved_by_one_stud(cli, monkeypatch):
    real = cli.condition_table

    def shifted(condition_id, scene_index):
        body = real(condition_id, scene_index)
        brick = body["bricks"][0]
        brick["translation_scene_units"][0] += 20
        x0, y0, _z = brick["translation_scene_units"]
        w, d, h = brick["extent_scene_units"]
        brick["top_face_corners_scene_units"] = [
            [x0, y0, h], [x0 + w, y0, h], [x0 + w, y0 + d, h], [x0, y0 + d, h]]
        return body

    monkeypatch.setattr(cli, "condition_table", shifted)
    problems, _rows = cli.stage_c(6, "baseline")
    assert any("does not reproduce V1" in p for p in problems)


# ---------------------------------------------------------------------------
# isolation from the Phase 3C pack
# ---------------------------------------------------------------------------

def test_this_track_is_denied_to_the_phase3c_pack():
    """gen09 has already executed against a 60-file pack.

    Everything this track adds lives under ``src/vision/**``, which the pack
    denies by name, or in ``scripts/`` where entries are listed one at a time
    and these are not listed. A module placed in ``src/eval/`` instead would be
    *included* -- and would ship a Blender dependency to an execution node
    whose contract forbids it from building data at all.
    """
    from src.training import pack

    for rel in ("src/vision/pbr/__init__.py", "src/vision/pbr/contract.py",
                "src/vision/pbr/project.py", "src/vision/pbr/colour.py",
                "src/vision/pbr/builder_blender.py",
                "scripts/62_visual_stress_pbr.py",
                "scripts/63_v3_diffusion_probe.py"):
        verdict, reason = pack.classify(rel)
        assert verdict == "exclude", f"{rel} would travel: {reason}"


TRACK_PATHS = ("src/vision/pbr/", "scripts/62_visual_stress_pbr.py",
               "scripts/63_v3_diffusion_probe.py", "scripts/64_pbr_archive.py")


def test_nothing_from_this_track_reached_the_pack():
    """The invariant is that the pack did not change, not that it is 60 files.

    A count is tree-dependent: the published tree has no ``data/`` and no
    ``gpu_plans/``, so three of the sixty are absent there and an assertion on
    sixty fails inside the snapshot for a reason that has nothing to do with
    this track. What has to hold in both trees is that none of this track's
    files travels -- gen09 has already executed against the pack it built, and
    a file added to it would change ``pack_digest``.
    """
    from src.training import pack

    travelling = [entry["path"] for entry in pack.manifest()["include"]]
    for prefix in TRACK_PATHS:
        offenders = [rel for rel in travelling if rel.startswith(prefix)]
        assert offenders == [], f"{prefix} reached the pack: {offenders}"


def test_a_module_under_src_eval_would_have_travelled():
    """The counterfactual, so the reason for the placement stays visible."""
    from src.training import pack

    verdict, reason = pack.classify("src/eval/visual_stress_pbr.py")
    assert verdict == "include"
    assert "src/**/*.py" in reason


def test_the_blender_module_imports_nothing_from_this_project(cli):
    """The structural half of the manifest argument.

    ``scripts/62`` launches Blender as a subprocess and never imports the
    builder, so an AST closure from the script cannot reach it. Rather than
    testing for a module the closure misses, the builder is built so there is
    nothing to miss -- it consumes ``scene.json`` and imports only ``bpy``,
    ``bmesh`` and the standard library.
    """
    sources = cli.source_manifests()
    assert sources["builder_project_imports"] == []
    assert sources["builder_imports_unmanifested"] == []


def test_both_manifests_cover_what_they_claim(cli):
    sources = cli.source_manifests()
    renderer = sources["renderer_source_manifest"]
    on_disk = {str(p.relative_to(ROOT))
               for p in sorted((ROOT / "src/vision/pbr").rglob("*.py"))}
    assert set(renderer) == on_disk, "the directory manifest is not the directory"
    assert "scripts/62_visual_stress_pbr.py" not in renderer
    evaluation = sources["evaluation_source_manifest"]
    # src/ui/photo.py, not src/ui/full.py: the recogniser moved into its own
    # module so this manifest stops covering the interface's notice text,
    # whose correction twice made this corpus unverifiable.
    for expected in ("src/eval/visual_stress.py", "src/ui/photo.py",
                     "src/vision/detect.py", "src/colour/recognise.py"):
        assert expected in evaluation, f"{expected} is not pinned"


def test_the_orchestrating_scripts_are_themselves_pinned(cli):
    """The gap that voided v2gen01 and v2gen02.

    ``import_closure`` returns the modules an entry point *reaches*, not the
    entry point itself. So the script holding the condition set, the scorer
    call and the contrast logic was in neither manifest: editing it changed
    what a generation measured and no digest moved. The entry point is now
    named.

    Only this track's entry point. The V3 probe used to be pinned here too,
    which coupled two generations that share nothing but a hashing helper, and
    the post-hoc verifiers are outside on purpose so a verifier can be improved
    without voiding a rendered corpus.
    """
    evaluation = cli.source_manifests()["evaluation_source_manifest"]
    assert "scripts/62_visual_stress_pbr.py" in evaluation
    for outside in ("scripts/63_v3_diffusion_probe.py",
                    "scripts/64_pbr_archive.py", "scripts/65_v3_archive.py"):
        assert outside not in evaluation, (
            f"{outside} is pinned by V2's manifest, which couples a fix to it "
            "to the validity of a rendered corpus")


def test_editing_the_orchestrating_script_moves_the_manifest_digest(cli,
                                                                    tmp_path):
    """The defect was silent, so the check has to be about the digest moving."""
    from src.training import pack
    from src.training.session import sha256_file

    entry_points = ("scripts/62_visual_stress_pbr.py",
                    "scripts/63_v3_diffusion_probe.py")
    closure = pack.import_closure(root=ROOT, entry_points=entry_points)
    # The entry points are not in the closure -- that is the whole point.
    for entry in entry_points:
        assert entry not in closure, (
            f"{entry} is in its own closure now; the manifest would have "
            "covered it without being told to and this test is stale")
    without = pbr.scene_digest(
        {rel: sha256_file(ROOT / rel) for rel in sorted(closure)})
    with_entries = cli.source_manifests()["evaluation_source_manifest_digest"]
    assert without != with_entries


def test_the_renderer_manifest_notices_a_new_file(cli, tmp_path):
    """A directory manifest exists to catch exactly this."""
    sources = cli.source_manifests()
    before = sources["renderer_source_manifest_digest"]
    planted = ROOT / "src/vision/pbr/_planted_for_a_test.py"
    planted.write_text("# planted\n", encoding="utf-8")
    try:
        after = cli.source_manifests()["renderer_source_manifest_digest"]
    finally:
        planted.unlink()
    assert after != before
    assert cli.source_manifests()["renderer_source_manifest_digest"] == before


# ---------------------------------------------------------------------------
# the frozen generation, and the things that must not pass
# ---------------------------------------------------------------------------

def _grant(cli):
    path = cli.FROZEN_DIR / cli.GENERATION / "authorization.json"
    if not path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {cli.GENERATION} is not frozen")
    return json.loads(path.read_text())


def test_the_authorisation_recomputes(cli):
    grant = _grant(cli)
    body = {k: v for k, v in grant.items() if k != "authorization_digest"}
    assert pbr.scene_digest(body) == grant["authorization_digest"]
    assert pbr.scene_digest(grant["contract"]) == grant["contract_digest"]
    assert pbr.scene_digest(grant["membership"]) == grant["membership_digest"]


def test_an_edited_authorisation_is_refused(cli):
    grant = _grant(cli)
    grant["contract"]["formal_backend"] = "GPU"
    body = {k: v for k, v in grant.items() if k != "authorization_digest"}
    assert pbr.scene_digest(body) != grant["authorization_digest"]


def test_the_frozen_membership_still_derives_from_the_writer(cli):
    """Every cell's scene digest, recomputed from the source, not re-read."""
    grant = _grant(cli)
    for cell in grant["membership"]:
        body = cli.condition_table(cell["condition_id"], cell["scene_index"])
        assert pbr.scene_digest(body) == cell["scene_digest"], cell["cell_id"]
        assert body["raster"]["width_px"] == cell["width_px"]
        assert len(body["bricks"]) == cell["n_true_bricks"]


def test_the_membership_names_every_cell_once(cli):
    grant = _grant(cli)
    ids = [c["cell_id"] for c in grant["membership"]]
    assert len(ids) == len(set(ids))
    assert len(ids) == grant["contract"]["n_cells"]


def test_the_paired_set_is_the_intersection_with_v1(cli):
    grant = _grant(cli)
    paired = [c for c in grant["membership"] if c["paired_with_v1"]]
    assert paired, "nothing is paired"
    for cell in paired:
        assert cell["camera"] == "ortho"
        assert cell["condition_id"] in cli.PAIRED_CONDITIONS
    for cell in grant["membership"]:
        if cell["condition_id"] in cli.V2_ONLY_CONDITIONS:
            assert not cell["paired_with_v1"]
        if cell["camera"] == "perspective":
            assert not cell["paired_with_v1"]


def test_every_paired_cell_has_a_v1_row(cli):
    grant = _grant(cli)
    v1 = cli.v1_rows()
    for cell in grant["membership"]:
        if cell["paired_with_v1"]:
            assert cell["image_id"] in v1, cell["image_id"]


def test_the_absent_v1_axes_are_named_with_reasons(cli):
    grant = _grant(cli)
    absent = grant["contract"]["absent_v1_axes"]
    assert absent, "a corpus that drops axes has to say which"
    for axis, why in absent.items():
        assert why and len(why) > 10, axis
    covered = set(cli.PAIRED_CONDITIONS) | set(absent)
    from src.vision import synthetic
    for axis, values in synthetic.CONDITIONS.items():
        for value in values:
            key = f"{axis}={value}"
            if key in ("background=plain", "lighting=even", "shadow=none",
                       "viewpoint=top", "occlusion=none", "quality=full"):
                continue          # these *are* the baseline
            assert key in covered, f"{key} is neither run nor declared absent"


# ---------------------------------------------------------------------------
# the archived images, and tampering with them
# ---------------------------------------------------------------------------

ARCHIVE = ROOT / "data/phase_v2/results"


def _archive(cli):
    directory = ARCHIVE / cli.GENERATION
    if not (directory / "index.json").is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {cli.GENERATION} is not archived")
    return directory


def _index(cli):
    return json.loads((_archive(cli) / "index.json").read_text())


def _run(cli):
    path = ARCHIVE / cli.GENERATION / "run.json"
    if not path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {cli.GENERATION} has not run")
    return json.loads(path.read_text())


def test_every_archived_image_still_hashes_to_what_the_run_recorded(cli):
    from src.training.session import sha256_file

    run = _run(cli)
    out_dir = _archive(cli)
    for cell in run["cells"]:
        png = out_dir / "images" / f"{cell['stem']}.png"
        assert png.is_file(), cell["cell_id"]
        assert sha256_file(png) == cell["image"]["png_sha256"], cell["cell_id"]


def test_an_edited_image_is_refused_by_the_scorer(cli, tmp_path):
    """A filed image that changed is tampering, not tolerance."""
    run = _run(cli)
    out_dir = _archive(cli)
    cell = run["cells"][0]
    source = out_dir / "images" / f"{cell['stem']}.png"
    from PIL import Image

    image = Image.open(source).convert("RGB")
    pixels = image.load()
    pixels[0, 0] = tuple((v + 7) % 256 for v in pixels[0, 0])
    edited = tmp_path / source.name
    image.save(edited, format="PNG")
    from src.training.session import sha256_file

    assert sha256_file(edited) != cell["image"]["png_sha256"]


def test_the_run_was_produced_under_the_authorisation_it_names(cli):
    run = _run(cli)
    grant = _grant(cli)
    assert run["authorization_digest"] == grant["authorization_digest"]
    assert run["backend"] == grant["contract"]["formal_backend"]
    assert run["n_cells"] == grant["contract"]["n_cells"]


def test_every_receipt_resolved_to_what_the_record_asked_for(cli):
    """The scene graph that rendered, against the record that asked for it."""
    run = _run(cli)
    for cell_id, receipt in run["receipts"].items():
        resolved = receipt["resolved"]
        assert resolved["engine"] == "CYCLES", cell_id
        assert resolved["device"] == run["backend"], cell_id
        assert resolved["use_adaptive_sampling"] is False, cell_id
        assert resolved["use_denoising"] is False, cell_id
        assert resolved["sampling_pattern"] == "TABULATED_SOBOL", cell_id
        assert resolved["seed"] == 0, cell_id
        assert resolved["film_transparent"] is False, cell_id
        assert resolved["use_motion_blur"] is False, cell_id
        assert resolved["camera"]["use_dof"] is False, cell_id
        assert resolved["camera"]["sensor_fit"] == "HORIZONTAL", cell_id
        # The whole reason the EXR is written by Blender at all.
        assert resolved["image_color_management"] == "OVERRIDE", cell_id
        assert resolved["image_view_transform"] == "Raw", cell_id
        assert resolved["scene_view_transform"] == "Standard", cell_id
        assert resolved["exr_codec"] == "ZIP", cell_id
        assert resolved["color_mode"] == "RGBA", cell_id


def test_no_receipt_left_a_face_smooth_shaded(cli):
    run = _run(cli)
    for cell_id, receipt in run["receipts"].items():
        for brick in receipt["bricks"]:
            assert brick["smooth_faces"] == 0, f"{cell_id} {brick['brick_id']}"


def test_every_receipt_agrees_with_the_projection_on_the_control_points(cli):
    """Stage B, over the whole corpus rather than the pilot's one scene."""
    run = _run(cli)
    points = {"P_C": (0, 0, 0), "P_X": (20, 0, 0), "P_Y": (0, 20, 0),
              "P_Z": (0, 0, 24), "P_XZ": (20, 0, 24)}
    for cell in run["cells"]:
        receipt = run["receipts"][cell["cell_id"]]
        body = cli.condition_table(cell["condition_id"], cell["scene_index"])
        for name, point in points.items():
            want = pbr_project.project(body, cell["camera"], point)
            got = receipt["control_points_raster"][name]
            assert abs(got[0] - want[0]) < 0.001, f"{cell['cell_id']} {name}.u"
            assert abs(got[1] - want[1]) < 0.001, f"{cell['cell_id']} {name}.v"


def test_no_render_produced_a_non_finite_pixel(cli):
    run = _run(cli)
    for cell in run["cells"]:
        assert cell["image"]["non_finite"] == 0, cell["cell_id"]


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------

def _scores(cli):
    path = ARCHIVE / cli.GENERATION / "scores.json"
    if not path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {cli.GENERATION} has no scores")
    return json.loads(path.read_text())


def test_the_scores_were_computed_by_v1s_own_scorer(cli):
    scores = _scores(cli)
    assert scores["scorer"] == "src.eval.visual_stress.score_image"
    from src.eval import visual_stress as vs
    assert scores["match_iou"] == float(vs.MATCH_IOU)
    for row in scores["rows"]:
        assert row["scorer"] == "src.eval.visual_stress.score_image"


def test_the_scores_recompute_from_the_archived_images(cli):
    """A sample re-derived, not re-read.

    Three cells rather than eighty: the scorer runs the whole recogniser, and
    the point is that the stored number is reproducible, which three cells
    establish as well as eighty and ten seconds faster.
    """
    scores = _scores(cli)
    out_dir = _archive(cli)
    for row in scores["rows"][:3]:
        cell = next(c for c in _run(cli)["cells"]
                    if c["cell_id"] == row["cell_id"])
        again = cli.score_row(cell["scene_index"], cell["condition_id"],
                              out_dir / "images" / f"{cell['stem']}.png")
        for field in ("n_true_bricks", "n_detected", "false_negatives",
                      "false_positives", "abstentions", "n_part_errors",
                      "n_colour_errors", "top1_correct_among_matched",
                      "inventory_exact_match"):
            assert again[field] == row[field], f"{row['cell_id']} {field}"


def test_the_paired_contrast_uses_only_cells_v1_also_ran(cli):
    scores = _scores(cli)
    paired = scores["paired_contrast_v2_minus_v1"]
    assert paired["unpaired_v2_image_ids"] == []
    v1 = cli.v1_rows()
    n_expected = sum(1 for row in scores["rows"]
                     if row["paired_with_v1"] and row["image_id"] in v1)
    assert paired["n_pairs"] == n_expected
    assert paired["n_pairs"] == len(cli.PAIRED_CONDITIONS) * len(cli.SCENE_INDICES)


def test_a_v2_only_condition_never_reaches_the_paired_contrast(cli):
    scores = _scores(cli)
    for row in scores["rows"]:
        if row["condition_id"] in cli.V2_ONLY_CONDITIONS:
            assert not row["paired_with_v1"]
    assert scores["v2_only_conditions"] == list(cli.V2_ONLY_CONDITIONS)


def test_the_camera_contrast_is_inside_v2_only(cli):
    scores = _scores(cli)
    camera = scores["camera_contrast_perspective_minus_ortho"]
    assert camera["n_pairs"] == len(cli.PERSPECTIVE_CONDITIONS) * len(
        cli.SCENE_INDICES)


def test_the_scores_digest_covers_the_contrasts(cli):
    scores = _scores(cli)
    body = {k: v for k, v in scores.items()
            if k not in ("rows", "scores_digest")}
    assert pbr.scene_digest(body) == scores["scores_digest"]
    body["paired_contrast_v2_minus_v1"]["n_pairs"] += 1
    assert pbr.scene_digest(body) != scores["scores_digest"]


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def _report(cli):
    path = ARCHIVE / cli.GENERATION / "pbr_report.md"
    if not path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {cli.GENERATION} has no report")
    return path.read_text(encoding="utf-8")


REPORT_MUST_SAY = (
    "V2 - V1 bounds nothing about performance on photographs",
    "not additive",
    "side faces becoming visible",
    # The registered plane is the brick top, so this is where 44 px/stud holds.
    # The earlier wording -- "holds at the table plane only" -- was true of a
    # camera that magnified every brick top against an unmoved truth box.
    "44 px/stud holds at the registered plane",
    "the table plane being minified",
    "never paired",
    # The mechanism has to travel with the colour contrast, or "2.3x V1's
    # colour errors" reads as a claim about PBR shading in general.
    "an ambient specular floor lifts dark channels",
    "why the colour errors move",
    # Which rows are in the 64 and which are not.
    "marks the rows that make up the",
    # The disclosure itself, kept as a positive check. A forbidden-substring
    # test for "distortion alone" cannot tell the claim from the denial of it,
    # and matched the report's own disclaimer.
    "not distortion alone",
)

REPORT_MUST_NOT_SAY = (
    "photorealistic accuracy",
    "equivalent to photographs",
    "proves the recogniser",
    "is purely a projection change",
    # The figure for the least-affected cell, quoted as the effect.
    "renders at 46.422 px/stud",
)


@pytest.mark.parametrize("claim", REPORT_MUST_SAY)
def test_the_report_keeps_its_disclosures(cli, claim):
    assert claim in _report(cli), claim


@pytest.mark.parametrize("claim", REPORT_MUST_NOT_SAY)
def test_the_report_makes_no_claim_it_cannot_support(cli, claim):
    assert claim not in _report(cli), claim


def test_the_report_quotes_the_figures_the_scores_hold(cli):
    report = _report(cli)
    scores = _scores(cli)
    paired = scores["paired_contrast_v2_minus_v1"]
    assert str(paired["n_pairs"]) in report
    for name, block in paired["metrics"].items():
        assert f"`{name}`" in report
        assert str(block["v2_total"]) in report, name


def test_the_report_names_the_backend_and_why(cli):
    report = _report(cli)
    grant = _grant(cli)
    assert grant["contract"]["formal_backend"] in report
    assert "byte-identical" in report
    assert "Metal" in report


def test_the_report_lists_every_absent_axis(cli):
    report = _report(cli)
    for axis in _grant(cli)["contract"]["absent_v1_axes"]:
        assert f"`{axis}`" in report, axis


# ---------------------------------------------------------------------------
# V3: two ledgers, and what may not be said about them
# ---------------------------------------------------------------------------

def _v3_plan(v3cli):
    path = v3cli.FROZEN_DIR / v3cli.GENERATION / "plan.json"
    if not path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {v3cli.GENERATION} is not frozen")
    return json.loads(path.read_text())


def test_the_v3_plan_recomputes(v3cli):
    plan = _v3_plan(v3cli)
    body = {k: v for k, v in plan.items() if k != "authorization_digest"}
    assert pbr.scene_digest(body) == plan["authorization_digest"]


def test_the_v3_plan_fixes_the_denominator_before_anything_is_generated(v3cli):
    plan = _v3_plan(v3cli)
    assert plan["n_attempts"] == plan["n_prompts"] * plan["n_seeds"]
    assert plan["n_attempts"] == 96
    ids = [a["attempt_id"] for a in plan["attempts"]]
    assert len(ids) == len(set(ids)) == 96


def test_the_v3_plan_forbids_retrying_a_failure(v3cli):
    plan = _v3_plan(v3cli)
    assert "not retried" in plan["retry_rule"]


def test_the_v3_prompts_claim_subject_alignment_and_no_more(v3cli):
    plan = _v3_plan(v3cli)
    assert "subject alignment only" in plan["alignment_claim"]
    assert "not a ground truth" in plan["alignment_claim"]
    tags = [a["scene_tag"] for a in plan["attempts"]]
    assert set(tags) == {f"scene_{i:02d}" for i in range(8)}


def test_the_v3_plan_names_the_licence_by_its_bytes(v3cli):
    plan = _v3_plan(v3cli)
    model = plan["model"]
    assert model["licence_name"] == "CreativeML Open RAIL++-M"
    assert model["licence_files"]["LICENSE.md"]["sha256"]
    assert model["licence_files"]["LICENSE.md"]["bytes"] > 1000
    assert model["weight_files"], "a model with no weights pinned"


V3_FORBIDDEN = ("accuracy", "miss rate", "silent-failure rate",
                "confidence interval", "p-value", "power",
                "out-of-distribution detection ability")


@pytest.mark.parametrize("claim", V3_FORBIDDEN)
def test_the_v3_plan_forbids_the_claims_it_cannot_support(v3cli, claim):
    assert claim in _v3_plan(v3cli)["forbidden"], claim


def _v3_outcome(v3cli):
    path = ARCHIVE / v3cli.GENERATION / "generation_outcome.json"
    if not path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {v3cli.GENERATION} has not run")
    return json.loads(path.read_text())


def _v3_distribution(v3cli):
    path = ARCHIVE / v3cli.GENERATION / "response_mode_distribution.json"
    if not path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {v3cli.GENERATION} has no distribution")
    return json.loads(path.read_text())


def test_the_generation_ledger_accounts_for_every_attempt(v3cli):
    outcome = _v3_outcome(v3cli)
    plan = _v3_plan(v3cli)
    assert outcome["denominator"] == 96
    assert len(outcome["attempts"]) == 96
    assert sum(outcome["counts"].values()) == 96
    planned = {a["attempt_id"] for a in plan["attempts"]}
    assert {a["attempt_id"] for a in outcome["attempts"]} == planned


def test_no_attempt_was_dropped_or_replaced(v3cli):
    outcome = _v3_outcome(v3cli)
    for attempt in outcome["attempts"]:
        assert attempt["outcome"] in ("produced", "generator_failed",
                                      "generator_timeout")
        if attempt["outcome"] != "produced":
            assert "error" in attempt, attempt["attempt_id"]


def test_the_two_ledgers_have_different_denominators_and_say_so(v3cli):
    outcome = _v3_outcome(v3cli)
    distribution = _v3_distribution(v3cli)
    assert distribution["denominator_n_produced"] == outcome["counts"]["produced"]
    assert "not 96" in distribution["denominator_note"]
    assert sum(distribution["counts"].values()) == \
        distribution["denominator_n_produced"]


def test_every_produced_image_reached_the_recogniser(v3cli):
    outcome = _v3_outcome(v3cli)
    distribution = _v3_distribution(v3cli)
    produced = {a["attempt_id"] for a in outcome["attempts"]
                if a["outcome"] == "produced"}
    scored = {r["attempt_id"] for r in distribution["rows"]}
    assert scored == produced, "an image was produced and not looked at"


def test_the_response_modes_are_exhaustive_and_exclusive(v3cli):
    distribution = _v3_distribution(v3cli)
    allowed = set(_v3_plan(v3cli)["response_modes"]) | {"recognizer_crash"}
    for row in distribution["rows"]:
        assert row["mode"] in allowed, row["attempt_id"]


def test_the_distribution_is_reported_per_prompt_as_well_as_pooled(v3cli):
    distribution = _v3_distribution(v3cli)
    per_prompt = distribution["per_prompt"]
    assert len(per_prompt) >= 1
    assert sum(b["n_produced"] for b in per_prompt.values()) == \
        distribution["denominator_n_produced"]


def test_the_v3_record_carries_no_rate_it_cannot_support(v3cli):
    """No field name that would read as an accuracy, anywhere in the record."""
    text = json.dumps(_v3_distribution(v3cli)).lower()
    for word in ("accuracy", "recall", "precision", "miss_rate",
                 "silent_failure", "p_value", "confidence_interval"):
        # The forbidden list itself names them; the counts must not.
        occurrences = text.count(word)
        allowed = json.dumps(_v3_plan(v3cli)["forbidden"]).lower().count(
            word.replace("_", " ")) + json.dumps(
            _v3_distribution(v3cli)["forbidden"]).lower().count(
            word.replace("_", " "))
        assert occurrences <= allowed, f"{word} appears as a measurement"


# ---------------------------------------------------------------------------
# the record against the scene that rendered
#
# Every test below exists because the corresponding defect survived a complete
# generation -- rendered, scored, reported, archived and passed by that day's
# verifier. The receipt carried the evidence and no code read it. So each one
# injects the defect and requires a refusal, rather than checking that a good
# artefact looks good.
# ---------------------------------------------------------------------------

def _calibration_dir():
    """The archived calibration, which is tracked evidence.

    Preferred over ``runs/``: the run directory is gitignored, so a test that
    read only from there would pass on this machine and have nothing to read
    anywhere else. That is the shape of the finding this whole archive closes.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pbr_cli_gen", ROOT / "scripts" / "62_visual_stress_pbr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tracked = ROOT / "data/phase_v2/calibration" / module.GENERATION
    if (tracked / "calibration.json").is_file():
        return tracked
    import glob

    matches = sorted(glob.glob("runs/pbr_calib*/calibration/calibration.json"))
    return Path(matches[-1]).parent if matches else None


def _receipt_for(condition_id="shadow=hard", scene_index=0):
    """A real receipt from the calibration renders, with its record."""
    directory = _calibration_dir()
    stem = f"confirm__{condition_id.replace('=', '-')}"
    path = directory / f"{stem}.receipt.json" if directory else None
    if path is None or not path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY}: no calibration receipt on disk")
    return (pbr.build_scene(scene_index, condition_id),
            json.loads(path.read_text()))


def test_a_real_receipt_agrees_with_its_record_field_by_field(archive_cli):
    body, receipt = _receipt_for()
    assert pbr.receipt_problems(body, receipt) == []
    assert archive_cli.extra_receipt_problems(body, receipt) == []


def test_the_writer_is_byte_identical_to_what_the_grant_pins(cli):
    """The discipline the extra checks exist to keep.

    A check discovered after a freeze goes in the verifier, not in the writer,
    because the writer's bytes are what the authorisation binds. This asserts
    that no such check has crept back into the frozen module.
    """
    import hashlib

    grant = _grant(cli)
    for relative, want in grant["sources"]["renderer_source_manifest"].items():
        live = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert live == want, (
            f"{relative} has moved since {cli.GENERATION} was frozen. A "
            "verifier improvement belongs in scripts/64_pbr_archive.py; a "
            "writer change opens a new generation")


def test_a_light_aimed_away_from_the_scene_centre_is_caught():
    """The defect that survived six generations.

    The rotation axis was the negation of ``cross((0,0,-1), forward)``, which
    rotates by -angle: the light kept its downward tilt and pointed its
    horizontal component the other way, 58.992 degrees off. The receipt
    recorded the resolved quaternion the whole time.
    """
    body, receipt = _receipt_for()
    receipt = json.loads(json.dumps(receipt))
    assert receipt["resolved"]["lights"], "shadow=hard must carry a light"
    receipt["resolved"]["lights"][0]["aim_residual_degrees"] = 58.992
    problems = pbr.receipt_problems(body, receipt)
    assert any("58.992" in p and "aims_at_scene_centre" in p for p in problems)


def test_a_light_whose_aim_is_not_measured_at_all_is_caught():
    body, receipt = _receipt_for()
    receipt = json.loads(json.dumps(receipt))
    receipt["resolved"]["lights"][0].pop("aim_residual_degrees")
    assert any("does not measure the aim" in p
               for p in pbr.receipt_problems(body, receipt))


def test_a_table_colour_that_did_not_take_is_caught():
    """``background`` varies exactly one value and the receipt now holds it."""
    body, receipt = _receipt_for("background=dark")
    receipt = json.loads(json.dumps(receipt))
    receipt["resolved"]["table"]["base_color_linear"] = [0.5, 0.5, 0.5]
    assert any("table.colour_linear" in p
               for p in pbr.receipt_problems(body, receipt))


def test_a_brick_material_socket_that_did_not_take_is_caught():
    body, receipt = _receipt_for("baseline")
    receipt = json.loads(json.dumps(receipt))
    receipt["bricks"][0]["material"]["roughness"] = 0.9
    assert any("material.roughness" in p
               for p in pbr.receipt_problems(body, receipt))


def test_a_pinned_value_missing_from_the_receipt_is_caught():
    """A field the record pins and the receipt omits is a field nothing checks."""
    body, receipt = _receipt_for("baseline")
    receipt = json.loads(json.dumps(receipt))
    receipt["resolved"].pop("film_exposure")
    assert any("film_exposure" in p and "absent" in p
               for p in pbr.receipt_problems(body, receipt))


def test_the_receipt_is_tied_to_the_exact_record_bytes():
    _body, receipt = _receipt_for("baseline")
    assert pbr.scene_digest_problems(receipt, b"not the scene") != []


def test_the_float32_allowance_is_one_ulp_and_not_a_wide_band():
    """Large values need room; small ones must not get any.

    The light powers are around 4.7e7, where float32 spacing is 4.0, so an
    exact integer comes back one or two away. A flat tolerance wide enough for
    that would swallow a roughness of 0.35 changing to 0.36.
    """
    assert pbr._near("47407660.000000", 47407661.0)
    assert not pbr._near("47407660.000000", 47407700.0)
    assert pbr._near("0.350000", 0.35000005)
    assert not pbr._near("0.350000", 0.36)


# ---------------------------------------------------------------------------
# the illumination, derived rather than typed in
# ---------------------------------------------------------------------------

def _calibration():
    directory = _calibration_dir()
    path = directory / "calibration.json" if directory else None
    if path is None or not path.is_file():
        pytest.skip(f"{ARTIFACT_ONLY}: no calibration on disk")
    return json.loads(path.read_text())


def test_the_recorded_illumination_is_what_the_render_measures(cli):
    assert cli.calibration_problems(_calibration()) == []


def test_a_bare_table_renders_at_v1s_own_canvas_fill():
    """Exact, not tolerant: a Lambertian surface under a uniform dome renders
    at its albedo, which is the whole reason the table is Lambertian."""
    calibration = _calibration()
    for background, entry in calibration["bare_table"].items():
        want = pbr.CONDITION_BACKGROUND[background]
        got = entry["measured"]["table_median_srgb"]
        assert max(abs(v - want) for v in got) <= 1, (
            f"background={background} renders at {got} and V1 fills {want}")


def test_a_dielectric_table_would_be_caught(cli):
    """The defect: V1's fill written in as an albedo renders 55 per cent high
    on ``dark``, because the dome's specular floor sits on top of it."""
    calibration = json.loads(json.dumps(_calibration()))
    calibration["bare_table"]["dark"]["measured"]["table_median_srgb"] = [59, 59, 59]
    problems = cli.calibration_problems(calibration)
    assert any("background=dark" in p and "59" in p for p in problems)


def test_a_light_power_that_drifted_from_the_solve_is_caught(cli):
    calibration = json.loads(json.dumps(_calibration()))
    calibration["solves"]["shadow=hard"]["solved_power_watts"] = "96000000.000000"
    assert any("shadow=hard" in p and "two-point solve" in p
               for p in cli.calibration_problems(calibration))


def test_the_clip_figures_are_named_for_the_cell_they_were_measured_on():
    """They are not corpus figures and the field names no longer imply it."""
    for row in pbr.CONDITION_LIGHTING.values():
        assert "measured_clip_percent_on_scene_00" in row
        assert "measured_plain_table_renders_at_srgb" not in row
        assert "measured_clip_percent_on_the_calibration_scene" not in row


# ---------------------------------------------------------------------------
# axes the writer does not implement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("condition_id", ["viewpoint=tilt_small",
                                          "viewpoint=tilt_large",
                                          "quality=downscaled",
                                          "quality=noisy",
                                          "quality=blurred",
                                          "quality=compressed"])
def test_an_axis_the_writer_ignores_is_refused_by_name(condition_id):
    """Accepted and silently ignored for seven generations.

    ``_condition`` returned V1's dictionary with the axis changed and nothing
    downstream read it, so ``build_scene`` produced a record byte-identical to
    baseline apart from its own provenance. No cell used these, so nothing was
    measured wrongly; a condition that renders as baseline while claiming to be
    a stress condition is the defect.
    """
    with pytest.raises(pbr.ContractError) as caught:
        pbr.build_scene(6, condition_id)
    assert "not implemented by this writer" in str(caught.value)


def test_every_condition_the_writer_accepts_changes_the_record():
    """The general form of the check above, so a future axis cannot slip in."""
    baseline = pbr.build_scene(6, "baseline")
    reference = pbr.scene_digest(
        {k: v for k, v in baseline.items() if k != "provenance"})
    from src.vision import synthetic

    for axis, values in synthetic.CONDITIONS.items():
        for value in values:
            condition_id = f"{axis}={value}"
            try:
                body = pbr.build_scene(6, condition_id)
            except pbr.ContractError:
                continue
            digest = pbr.scene_digest(
                {k: v for k, v in body.items() if k != "provenance"})
            if value == synthetic.BASELINE[axis]:
                assert digest == reference, condition_id
            else:
                assert digest != reference, (
                    f"{condition_id} is accepted and produces a record "
                    "identical to baseline, so it would render, score and be "
                    "reported as a stress condition it is not")


# ---------------------------------------------------------------------------
# the contrast columns
# ---------------------------------------------------------------------------

def test_the_worse_and_better_columns_know_which_direction_is_better(cli):
    """They did not, for seven generations.

    ``n_v2_worse`` counted ``delta > 0`` for all eight metrics, so on
    ``n_detected``, ``top1_correct_among_matched`` and ``mean_confidence`` --
    where higher is better -- the two published column headers were swapped.
    """
    v1 = {"x/a": {"n_detected": 5, "false_negatives": 1, "false_positives": 0,
                  "abstentions": 1, "n_part_errors": 1, "n_colour_errors": 1,
                  "top1_correct_among_matched": 2, "mean_confidence": 0.4,
                  "inventory_exact_match": False}}
    row = {"image_id": "x/a", "condition_id": "baseline", "camera": "ortho",
           "scene_id": "scene_00", "n_detected": 6, "false_negatives": 2,
           "false_positives": 0, "abstentions": 2, "n_part_errors": 2,
           "n_colour_errors": 2, "top1_correct_among_matched": 3,
           "mean_confidence": 0.5, "inventory_exact_match": False}
    metrics = cli.paired_contrast([row], v1)["metrics"]
    # V2 detected one *more*: higher, and better.
    assert metrics["n_detected"]["n_v2_higher"] == 1
    assert metrics["n_detected"]["n_v2_better"] == 1
    assert metrics["n_detected"]["n_v2_worse"] == 0
    # V2 made one *more* colour error: higher, and worse.
    assert metrics["n_colour_errors"]["n_v2_higher"] == 1
    assert metrics["n_colour_errors"]["n_v2_worse"] == 1
    assert metrics["n_colour_errors"]["n_v2_better"] == 0
    for name, block in metrics.items():
        assert "higher_is_better" in block, name


def test_every_paired_metric_declares_its_polarity(cli):
    assert set(cli.HIGHER_IS_BETTER) == set(cli.PAIRED_METRICS)


def test_the_contrast_reports_the_spread_and_not_only_the_net(cli):
    """A net of +2 built from +25 and -23 is not "barely moved"."""
    v1 = {"a": {"n_colour_errors": 5}, "b": {"n_colour_errors": 5}}
    base = {"condition_id": "baseline", "camera": "ortho",
            "scene_id": "scene_00", "inventory_exact_match": False}
    rows = [{**base, "image_id": "a", "n_colour_errors": 15},
            {**base, "image_id": "b", "n_colour_errors": -5}]
    for row in rows:
        for name in cli.PAIRED_METRICS:
            row.setdefault(name, 0)
    for key in ("a", "b"):
        for name in cli.PAIRED_METRICS:
            v1[key].setdefault(name, 0)
        v1[key]["inventory_exact_match"] = False
    block = cli.paired_contrast(rows, v1)["metrics"]["n_colour_errors"]
    assert block["mean_delta"] == 0.0
    assert block["sum_abs_delta"] == 20.0
    assert block["n_changed"] == 2


# ---------------------------------------------------------------------------
# the runtime manifest
# ---------------------------------------------------------------------------

def test_the_blender_bundle_manifest_covers_the_bundle(cli):
    """It covered 0 of 6498 files: the path had ``Contents`` twice."""
    if not cli.BLENDER.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} Blender is not installed")
    files, digest = cli.tree_manifest(cli.BLENDER.parents[1])
    assert len(files) > 1000, (
        f"the bundle manifest covers {len(files)} files; the defect looked "
        "exactly like this, with 0")
    import hashlib
    empty = hashlib.sha256(b"{}").hexdigest()
    assert digest != empty, "the digest is sha256('{}') over nothing"


def test_a_manifest_over_a_missing_directory_is_visibly_empty(cli):
    """What the defect produced, so the shape is recognisable."""
    files, digest = cli.tree_manifest(cli.BLENDER.parents[1] / "Contents")
    import hashlib
    assert files == {}
    assert digest == hashlib.sha256(b"{}").hexdigest()


# ---------------------------------------------------------------------------
# the colour pipeline
# ---------------------------------------------------------------------------

def _write_exr(path, rgba, attributes=None):
    """A real EXR, so the reader's guards are exercised and not mirrored."""
    import OpenEXR

    header = {"compression": OpenEXR.ZIP_COMPRESSION, **(attributes or {})}
    handle = OpenEXR.File(header, {"RGBA": OpenEXR.Channel(rgba)})
    handle.write(str(path))
    return path


def test_the_exr_reader_accepts_a_well_formed_file(tmp_path):
    import numpy as np

    from src.vision.pbr import colour as pbr_colour

    rgba = np.zeros((4, 4, 4), dtype=np.float32)
    rgba[:, :, 0] = 0.25
    rgba[:, :, 3] = 1.0
    array = pbr_colour.read_exr(_write_exr(tmp_path / "ok.exr", rgba))
    assert array.shape == (4, 4, 4)
    assert float(array[0, 0, 0]) == pytest.approx(0.25, abs=1e-3)


def test_the_exr_reader_refuses_a_channel_permutation(tmp_path):
    """Alpha is uniformly 1.0 in every render here and R is not, so an
    all-ones fourth plane is a free and exact test of the reorder a
    third-party library performs on the file's alphabetical A,B,G,R order.

    Written as a real file and read by the real reader: a test that mirrors the
    guard in its own body proves only that the mirror works.
    """
    import numpy as np

    from src.vision.pbr import colour as pbr_colour

    rgba = np.zeros((4, 4, 4), dtype=np.float32)
    rgba[:, :, 0] = 0.25
    rgba[:, :, 3] = 1.0
    rotated = np.ascontiguousarray(np.roll(rgba, 1, axis=2))
    with pytest.raises(pbr.ContractError) as caught:
        pbr_colour.read_exr(_write_exr(tmp_path / "rolled.exr", rotated))
    assert "not uniformly 1.0" in str(caught.value)


def test_the_exr_reader_refuses_a_transparent_film(tmp_path):
    """The other thing a non-uniform alpha means: this track pins
    ``film_transparent`` off, so an alpha below 1 is a scene that was not the
    pinned one."""
    import numpy as np

    from src.vision.pbr import colour as pbr_colour

    rgba = np.ones((4, 4, 4), dtype=np.float32)
    rgba[2, 2, 3] = 0.0
    with pytest.raises(pbr.ContractError):
        pbr_colour.read_exr(_write_exr(tmp_path / "alpha.exr", rgba))


def test_the_exr_reader_refuses_a_file_that_is_not_scene_linear(tmp_path):
    """The file says what space it is in, and the reader reads that rather
    than assuming it."""
    import numpy as np

    from src.vision.pbr import colour as pbr_colour

    rgba = np.zeros((4, 4, 4), dtype=np.float32)
    rgba[:, :, 3] = 1.0
    path = _write_exr(tmp_path / "srgb.exr", rgba,
                      {"colorInteropID": "srgb_display"})
    with pytest.raises(pbr.ContractError) as caught:
        pbr_colour.read_exr(path)
    assert "colour space" in str(caught.value)


def test_the_exr_reader_accepts_the_space_blender_writes(tmp_path):
    import numpy as np

    from src.vision.pbr import colour as pbr_colour

    rgba = np.zeros((4, 4, 4), dtype=np.float32)
    rgba[:, :, 3] = 1.0
    path = _write_exr(tmp_path / "lin.exr", rgba,
                      {"colorInteropID": "lin_rec709_scene"})
    assert pbr_colour.read_exr(path).shape == (4, 4, 4)


def test_a_real_archived_exr_declares_scene_linear():
    import glob

    matches = sorted(glob.glob("data/phase_v2/results/*/linear_sample/*.exr"))
    if not matches:
        pytest.skip(f"{ARTIFACT_ONLY}: no archived EXR on disk")
    import OpenEXR

    with OpenEXR.File(matches[0]) as handle:
        header = dict(handle.header())
    assert "lin_" in str(header.get("colorInteropID", "")), (
        f"{matches[0]} does not declare a scene-linear space")


def test_the_canonical_png_round_trips_from_the_archived_exr():
    """The link ``--mode verify`` did not close: the EXR hashed to its digest
    and the PNG hashed to its own, and nothing showed they were one picture."""
    import glob

    matches = sorted(glob.glob("data/phase_v2/results/*/linear_sample/*.exr"))
    if not matches:
        pytest.skip(f"{ARTIFACT_ONLY}: no archived EXR on disk")
    import tempfile

    from src.vision.pbr import colour as pbr_colour

    exr = Path(matches[0])
    scene = exr.parent.parent / f"scenes/{exr.stem}.scene.json"
    png = exr.parent.parent / f"images/{exr.stem}.png"
    if not (scene.is_file() and png.is_file()):
        pytest.skip(f"{ARTIFACT_ONLY}: the archive is incomplete")
    body = json.loads(scene.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        again = pbr_colour.to_canonical_png(exr, Path(tmp) / "a.png", body["png"])
    import hashlib
    assert again["png_sha256"] == hashlib.sha256(png.read_bytes()).hexdigest()


def test_the_array_and_scalar_transfer_functions_agree():
    """Two implementations of the same curve, one of them untested until now."""
    import numpy as np

    from src.vision.pbr import colour as pbr_colour

    values = np.linspace(0.0, 1.0, 100001, dtype=np.float64)
    array = pbr_colour.linear_to_srgb(values)
    scalar = np.array([pbr.linear_to_srgb_scalar(float(v)) for v in values])
    assert float(np.abs(array - scalar).max()) == 0.0


# ---------------------------------------------------------------------------
# can a forger pass --mode verify?
#
# The earlier verifier read its own coverage out of the artefacts under test:
# it iterated ``run["cells"]``, re-scored those, re-rendered the report from
# ``scores.json``'s own aggregate blocks and printed ``verified: true``. A
# forger who trimmed the cell list, doctored the rows it no longer covered and
# recomputed the index digests passed. These tests are the forgeries.
# ---------------------------------------------------------------------------

def _forgeable_archive(archive_cli, tmp_path, monkeypatch):
    """A private copy of the archive, with verify pointed at it."""
    import shutil

    source = Path("data/phase_v2/results") / archive_cli.GENERATION
    if not (source / "index.json").is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {archive_cli.GENERATION} is not archived")
    target = tmp_path / "results"
    shutil.copytree(source, target / archive_cli.GENERATION)
    monkeypatch.setattr(archive_cli, "RESULTS_DIR", target)
    return target / archive_cli.GENERATION


def _reindex(directory):
    """Recompute every digest the index claims, as a forger would."""
    from src.training.session import sha256_file

    index = json.loads((directory / "index.json").read_text())
    for relative, entry in index["files"].items():
        path = directory / relative
        if path.is_file():
            entry["bytes"] = path.stat().st_size
            entry["sha256"] = sha256_file(path)
    index["index_digest"] = pbr.scene_digest(
        {k: v for k, v in index.items() if k != "index_digest"})
    (directory / "index.json").write_text(
        json.dumps(index, indent=1, sort_keys=True))
    return index


def _capture(archive_cli, args):
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        archive_cli.mode_verify(args)
    return json.loads(buffer.getvalue())


def test_a_trimmed_cell_list_is_caught(archive_cli, tmp_path, monkeypatch):
    """The forgery the earlier verifier passed.

    Trim ``run["cells"]`` to three, leave ``n_cells: 80``, point
    ``linear_sample_cells`` at those three, recompute the index digests. The
    coverage now comes from the digest-checked authorisation instead.
    """
    directory = _forgeable_archive(archive_cli, tmp_path, monkeypatch)
    run = json.loads((directory / "run.json").read_text())
    run["cells"] = run["cells"][:3]
    (directory / "run.json").write_text(json.dumps(run, indent=1, sort_keys=True))
    index = json.loads((directory / "index.json").read_text())
    index["linear_sample_cells"] = [c["cell_id"] for c in run["cells"]][:1]
    (directory / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    _reindex(directory)

    result = _capture(archive_cli, argparse.Namespace(
        out_dir=str(directory), force=False))
    assert result["verified"] is False
    assert any("frozen membership" in p for p in result["problems"])


def test_a_swapped_image_is_caught(archive_cli, tmp_path, monkeypatch):
    """Two cells' images exchanged, then the index recomputed over them.

    The index then agrees with the files and the files with the index; what
    disagrees is the digest the *run* recorded when it scored them.
    """
    directory = _forgeable_archive(archive_cli, tmp_path, monkeypatch)
    images = sorted((directory / "images").glob("*.png"))
    first, second = images[0], images[1]
    keep = first.read_bytes()
    first.write_bytes(second.read_bytes())
    second.write_bytes(keep)
    _reindex(directory)

    result = _capture(archive_cli, argparse.Namespace(
        out_dir=str(directory), force=False))
    assert result["verified"] is False
    assert any("the run recorded" in p for p in result["problems"])


def test_a_doctored_receipt_is_caught(archive_cli, tmp_path, monkeypatch):
    """A light aimed away from a centre its record claims it aims at."""
    directory = _forgeable_archive(archive_cli, tmp_path, monkeypatch)
    run = json.loads((directory / "run.json").read_text())
    touched = 0
    # The receipts live in their own dict, not on the cells. Reading them off
    # the cells is what made the verifier's own receipt check do nothing while
    # reporting that it had compared 80 of them.
    for cell_id, receipt in (run.get("receipts") or {}).items():
        lights = receipt.get("resolved", {}).get("lights")
        if lights:
            lights[0]["aim_residual_degrees"] = 58.992
            touched += 1
            break
    if not touched:
        pytest.skip(f"{ARTIFACT_ONLY}: no cell in this archive carries a light")
    (directory / "run.json").write_text(json.dumps(run, indent=1, sort_keys=True))
    _reindex(directory)

    result = _capture(archive_cli, argparse.Namespace(
        out_dir=str(directory), force=False))
    assert result["verified"] is False
    assert any("58.992" in p for p in result["problems"])


def test_a_doctored_scene_record_is_caught(archive_cli, tmp_path, monkeypatch):
    """The membership pins each scene record's digest and nothing read it."""
    directory = _forgeable_archive(archive_cli, tmp_path, monkeypatch)
    scenes = sorted((directory / "scenes").glob("*.scene.json"))
    body = json.loads(scenes[0].read_text())
    body["table"]["colour_srgb_uint8"] = [200, 200, 200]
    scenes[0].write_bytes(pbr.canonical_bytes(body))
    _reindex(directory)

    result = _capture(archive_cli, argparse.Namespace(
        out_dir=str(directory), force=False))
    assert result["verified"] is False
    assert any("the membership pins" in p for p in result["problems"])


def test_a_substituted_authorisation_copy_is_caught(archive_cli, tmp_path,
                                                    monkeypatch):
    """The archived grant was pinned only to its own claimed hash."""
    directory = _forgeable_archive(archive_cli, tmp_path, monkeypatch)
    grant = json.loads((directory / "authorization.json").read_text())
    grant["note_added_by_a_forger"] = "this did not travel with the run"
    (directory / "authorization.json").write_text(
        json.dumps(grant, indent=1, sort_keys=True))
    _reindex(directory)

    result = _capture(archive_cli, argparse.Namespace(
        out_dir=str(directory), force=False))
    assert result["verified"] is False
    assert any("byte-identical" in p for p in result["problems"])


def test_verifying_a_superseded_generation_is_refused(archive_cli, tmp_path,
                                                      monkeypatch):
    """``verified: true`` said nothing about whether the thing had been voided."""
    directory = _forgeable_archive(archive_cli, tmp_path, monkeypatch)
    frozen = tmp_path / "frozen"
    import shutil
    shutil.copytree(Path("data/phase_v2/frozen"), frozen)
    record = json.loads((frozen / "supersession.json").read_text())
    record["superseded"].append({"generation": archive_cli.GENERATION,
                                 "may_be_cited": False})
    (frozen / "supersession.json").write_text(json.dumps(record, indent=1))
    monkeypatch.setattr(archive_cli, "FROZEN_DIR", frozen)

    result = _capture(archive_cli, argparse.Namespace(
        out_dir=str(directory), force=False))
    assert result["verified"] is False
    assert any("superseded" in p for p in result["problems"])


def test_verify_names_what_it_actually_checked(archive_cli):
    """A verifier that prints only a verdict cannot be audited."""
    source = Path("scripts/64_pbr_archive.py").read_text()
    for key in ("cells_from_frozen_membership",
                "scene_records_against_membership",
                "images_against_the_run_record",
                "receipts_compared_to_their_records",
                "rows_rescored", "aggregates_recomputed_from_rows",
                "exr_reconverted_to_png", "supersession_read"):
        assert key in source, f"verify does not report {key}"


# ---------------------------------------------------------------------------
# the V3 probe now has a re-derivation path
# ---------------------------------------------------------------------------

def test_the_v3_verifier_reproduces_the_archived_distribution(v3archive_cli):
    """The probe had no verify mode at all: its whole result was a
    self-consistent claim.

    The verifier restates the distribution body, because the probe computes it
    inline and the probe is pinned inside the frozen plan. Reproducing the
    digest from the archived rows is what shows the restatement is faithful --
    any field that disagreed would move it. The full re-derivation, which runs
    the live recogniser over all 96 retained images, is ``--mode verify``.
    """
    directory = Path("data/phase_v2/results") / v3archive_cli.GENERATION
    plan = Path("data/phase_v2/frozen") / v3archive_cli.GENERATION / "plan.json"
    if not (directory / "response_mode_distribution.json").is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {v3archive_cli.GENERATION} is not archived")
    distribution = json.loads(
        (directory / "response_mode_distribution.json").read_text())
    frozen = json.loads(plan.read_text())
    again = v3archive_cli.summarise(distribution["rows"], frozen,
                                    distribution["outcome_digest"])
    assert again["distribution_digest"] == distribution["distribution_digest"]


def test_a_doctored_v3_row_moves_the_distribution_digest(v3archive_cli):
    directory = Path("data/phase_v2/results") / v3archive_cli.GENERATION
    plan = Path("data/phase_v2/frozen") / v3archive_cli.GENERATION / "plan.json"
    if not (directory / "response_mode_distribution.json").is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {v3archive_cli.GENERATION} is not archived")
    distribution = json.loads(
        (directory / "response_mode_distribution.json").read_text())
    frozen = json.loads(plan.read_text())
    rows = json.loads(json.dumps(distribution["rows"]))
    rows[0]["mode"] = "reject"
    again = v3archive_cli.summarise(rows, frozen,
                                    distribution["outcome_digest"])
    assert again["distribution_digest"] != distribution["distribution_digest"]


def test_the_v3_verifier_is_outside_the_frozen_plan(v3archive_cli):
    """The coupling that voided six V2 generations, not repeated here."""
    plan = Path("data/phase_v2/frozen") / v3archive_cli.GENERATION / "plan.json"
    if not plan.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {v3archive_cli.GENERATION} is not frozen")
    manifest = json.loads(plan.read_text())["source_manifest"]["files"]
    assert "scripts/65_v3_archive.py" not in manifest
    assert "scripts/63_v3_diffusion_probe.py" in manifest


def test_the_v3_verifier_derives_the_recogniser_closure(v3archive_cli):
    """Derived from the code, not written down.

    The first version carried a hand-written nine-entry tuple. The real closure
    of ``src.ui.full.analyse_photo`` is 49 of the plan's 56 files: the list
    missed 40, ``src/vision/metrics.py`` and ``src/vision/classes.py`` among
    them, and drift in any of those was reported and then ignored.
    """
    closure = v3archive_cli.recogniser_closure()
    assert v3archive_cli.RECOGNISER_ENTRY_POINT in closure
    for missed in ("src/vision/metrics.py", "src/vision/classes.py",
                   "src/vision/model.py", "src/eval/scoring.py",
                   "src/training/pack.py"):
        assert missed in closure, (
            f"{missed} is reachable from the recogniser and the closure "
            "does not contain it")
    assert len(closure) > 40, len(closure)
    # The tool that derives the closure is itself inside it, so drift in the
    # derivation is fatal by the same rule.
    assert "src/training/pack.py" in closure
    assert not hasattr(v3archive_cli, "RECOGNISER_FILES"), (
        "the hardcoded list is back")


def test_every_non_decisive_exemption_carries_a_checkable_reason(v3archive_cli):
    plan = Path("data/phase_v2/frozen") / v3archive_cli.GENERATION / "plan.json"
    if not plan.is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {v3archive_cli.GENERATION} is not frozen")
    manifest = json.loads(plan.read_text())["source_manifest"]["files"]
    closure = v3archive_cli.recogniser_closure()
    assert v3archive_cli.NON_DECISIVE, "an empty exemption set is suspicious"
    for rel, reason in v3archive_cli.NON_DECISIVE.items():
        assert rel in manifest, f"{rel} is excused and the plan does not pin it"
        assert rel not in closure, (
            f"{rel} is excused on the grounds that the recogniser does not "
            "reach it, and it does")
        assert len(reason) > 80, f"{rel}'s reason is too short to check"


def _v3_verify(v3archive_cli, frozen_dir=None, images_dir=None):
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        v3archive_cli.mode_verify(argparse.Namespace(images_dir=images_dir))
    return json.loads(buffer.getvalue())


def test_drift_in_a_dependency_the_old_list_missed_is_fatal(v3archive_cli,
                                                            tmp_path,
                                                            monkeypatch):
    """The injection the old classifier would have waved through.

    ``src/vision/metrics.py`` is reachable from ``analyse_photo`` and was absent
    from the hand-written list, so its drift used to be filed under
    ``source_drift_outside_the_recogniser`` and ignored.
    """
    import shutil

    source = Path("data/phase_v2/frozen")
    if not (source / v3archive_cli.GENERATION / "plan.json").is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {v3archive_cli.GENERATION} is not frozen")
    frozen = tmp_path / "frozen"
    shutil.copytree(source, frozen)
    plan_path = frozen / v3archive_cli.GENERATION / "plan.json"
    plan = json.loads(plan_path.read_text())
    victim = "src/vision/metrics.py"
    assert victim in plan["source_manifest"]["files"]
    plan["source_manifest"]["files"][victim] = "0" * 64
    plan_path.write_text(json.dumps(plan, indent=1, sort_keys=True))
    monkeypatch.setattr(v3archive_cli, "FROZEN_DIR", frozen)

    result = _v3_verify(v3archive_cli)
    assert result["verified"] is False
    assert victim in result["checked"]["source_drift_fatal"]
    assert any("Drift is fatal unless" in p for p in result["problems"])
    # And it must not have re-derived anything: a distribution computed with
    # code the plan does not bind is not evidence about the plan.
    assert "distribution_digest_reproduced" not in result["checked"]


def test_a_void_exemption_is_caught(v3archive_cli, tmp_path, monkeypatch):
    """An exemption whose proof stops holding has to stop excusing.

    Here the excused file is declared to be a recogniser dependency; the
    exemption says the recogniser does not reach it, so it must be refused
    rather than silently continue to excuse.
    """
    import shutil

    source = Path("data/phase_v2/frozen")
    if not (source / v3archive_cli.GENERATION / "plan.json").is_file():
        pytest.skip(f"{ARTIFACT_ONLY} {v3archive_cli.GENERATION} is not frozen")
    frozen = tmp_path / "frozen"
    shutil.copytree(source, frozen)
    monkeypatch.setattr(v3archive_cli, "FROZEN_DIR", frozen)
    excused = sorted(v3archive_cli.NON_DECISIVE)[0]
    monkeypatch.setattr(v3archive_cli, "recogniser_closure",
                        lambda root=None: {excused})

    result = _v3_verify(v3archive_cli)
    assert result["verified"] is False
    assert any("The exemption is void" in p for p in result["problems"])


def test_a_brick_built_at_the_wrong_size_is_caught(archive_cli):
    """The bounds the record implies, which nothing compared until now.

    ``geometry_digest`` is recorded and is blind to connectivity; the
    projection check reads the *record*, not the mesh. So a mesh built at the
    wrong size, with a wrong bevel width or with a stud missing was invisible
    to every check in the track.
    """
    body, receipt = _receipt_for("baseline")
    receipt = json.loads(json.dumps(receipt))
    receipt["bricks"][0]["bounds_max"][0] += 20
    assert any("bounds_max" in p
               for p in archive_cli.extra_receipt_problems(body, receipt))


def test_studs_missing_from_the_top_of_a_brick_are_caught(archive_cli):
    """The stud height is what raises ``bounds_max.z`` above the body box."""
    body, receipt = _receipt_for("baseline")
    receipt = json.loads(json.dumps(receipt))
    receipt["bricks"][0]["bounds_max"][2] = body["bricks"][0][
        "extent_scene_units"][2]
    assert any("bounds_max" in p
               for p in archive_cli.extra_receipt_problems(body, receipt))


def test_a_table_at_the_wrong_height_is_caught(archive_cli):
    body, receipt = _receipt_for("baseline")
    receipt = json.loads(json.dumps(receipt))
    receipt["resolved"]["table"]["bounds_z"] = [-1.0, 5.0]
    assert any("table.plane_z" in p
               for p in archive_cli.extra_receipt_problems(body, receipt))


def test_verify_counts_report_work_done_and_not_candidates(archive_cli):
    """The defect this round introduced and then found in its own code.

    ``receipts_compared_to_their_records`` reported 80 while comparing 0: the
    receipts live in their own top-level dict, not on the cells, and the count
    was the number of *candidates*. Every count in verify now reports what was
    actually done, and reports a problem when that is short of the membership.
    """
    source = Path("scripts/64_pbr_archive.py").read_text()
    for phrase in ("were compared to the run's own digests",
                   "were hashed against the membership",
                   "receipts were compared to their",
                   "were re-converted and compared to their PNG",
                   "rows re-derived, so the"):
        assert phrase in source, f"verify cannot report a short count: {phrase}"


def test_verify_validates_the_out_dir_it_is_given(archive_cli, tmp_path,
                                                  monkeypatch):
    """The flag was accepted and ignored, while ``how_to_recheck`` passes it.

    Following the archive's own instruction therefore exercised a parameter
    that did nothing.
    """
    directory = _forgeable_archive(archive_cli, tmp_path, monkeypatch)
    result = _capture(archive_cli, argparse.Namespace(
        out_dir="runs/pbr/somewhere-else", force=False))
    assert result["verified"] is False
    assert any("--out-dir names" in p for p in result["problems"])


def test_verify_checks_the_index_counts_against_the_file_list(
        archive_cli, tmp_path, monkeypatch):
    """Index fields that were pure claims, all checkable from data in hand."""
    directory = _forgeable_archive(archive_cli, tmp_path, monkeypatch)
    index = json.loads((directory / "index.json").read_text())
    index["images_archived"] = 3
    index["index_digest"] = pbr.scene_digest(
        {k: v for k, v in index.items() if k != "index_digest"})
    (directory / "index.json").write_text(json.dumps(index, indent=1,
                                                     sort_keys=True))
    result = _capture(archive_cli, argparse.Namespace(
        out_dir=str(directory), force=False))
    assert result["verified"] is False
    assert any("images_archived" in p for p in result["problems"])


def test_a_forced_overwrite_names_what_it_replaced(archive_cli):
    """``--force`` is a real overwrite of evidence, so it leaves a trace."""
    source = Path("scripts/64_pbr_archive.py").read_text()
    assert "forced_overwrite_of" in source
    assert '"overwrote_index_digest": overwrote' in source


def test_the_deferred_defects_are_named_in_the_documents():
    """A known defect left in place has to be written down, not remembered."""
    for name in ("PORTFOLIO.md", "PROJECT_STATUS.md"):
        path = ROOT / name
        if not path.is_file():
            pytest.skip(f"{ARTIFACT_ONLY} {name} is not in this tree")
        text = path.read_text(encoding="utf-8")
        assert "drop_alpha=False" in text, name
        assert "half_to_even" in text, name
        assert "geometry_digest" in text, name


def test_the_calibration_is_tracked_evidence_and_not_only_a_run_directory(
        archive_cli, cli):
    """``runs/`` is gitignored, so evidence that lives only there is not kept.

    This is the same gap Phase 3C's gen09 had to close after the fact, and the
    reason the illumination figures could sit inside every scene digest with no
    artefact behind them.
    """
    # v2gen09--v2gen11 re-derived the corpus after source-closure changes; they
    # did not re-run or rename v2gen08's illumination calibration.  Verify the
    # explicitly referenced archive rather than pretending it was measured in
    # the current generation.
    tracked = (ROOT / "data/phase_v2/calibration" /
               archive_cli.CALIBRATION_GENERATION)
    if not tracked.is_dir():
        pytest.skip(
            f"{ARTIFACT_ONLY} {archive_cli.CALIBRATION_GENERATION} calibration "
            "is not archived")
    index = json.loads((tracked / "index.json").read_text())
    assert cli.GENERATION == "v2gen11"
    assert archive_cli.CALIBRATION_GENERATION == "v2gen08"
    assert index["generation"] == archive_cli.CALIBRATION_GENERATION
    assert index["n_files"] == len(index["files"])
    body = json.loads((tracked / "calibration.json").read_text())
    assert pbr.scene_digest(body) == index["calibration_digest"]
    # Every render it reports has its scene record and its receipt beside it.
    stems = [n[: -len(".receipt.json")] for n in index["files"]
             if n.endswith(".receipt.json")]
    assert len(stems) >= len(body["conditions"])
    for stem in stems:
        assert f"{stem}.scene.json" in index["files"], stem
        assert f"{stem}.png" in index["files"], stem
    assert index["exr_not_archived"], "the EXRs are named as absent, not silent"


def test_the_archived_calibration_verifies(archive_cli, cli):
    tracked = (ROOT / "data/phase_v2/calibration" /
               archive_cli.CALIBRATION_GENERATION)
    if not tracked.is_dir():
        pytest.skip(
            f"{ARTIFACT_ONLY} {archive_cli.CALIBRATION_GENERATION} calibration "
            "is not archived")
    result = _capture_calibration(archive_cli)
    assert result["problems"] == []
    assert result["verified"] is True
    assert result["generation"] == archive_cli.CALIBRATION_GENERATION
    assert result["contract_generation"] == cli.GENERATION
    assert result["checked"]["receipts_compared"] >= len(
        pbr.CONDITION_LIGHTING)


def _capture_calibration(archive_cli):
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        archive_cli.mode_verify_calibration(argparse.Namespace())
    return json.loads(buffer.getvalue())


def test_a_calibration_whose_numbers_moved_is_refused(archive_cli, cli,
                                                      tmp_path, monkeypatch):
    """Filing a calibration that disagrees with the contract is refused, so a
    stale figure cannot be archived as if it were current."""
    tracked = (ROOT / "data/phase_v2/calibration" /
               archive_cli.CALIBRATION_GENERATION)
    if not tracked.is_dir():
        pytest.skip(
            f"{ARTIFACT_ONLY} {archive_cli.CALIBRATION_GENERATION} calibration "
            "is not archived")
    body = json.loads((tracked / "calibration.json").read_text())
    body["generation"] = cli.GENERATION
    body["bare_table"]["dark"]["measured"]["table_median_srgb"] = [59, 59, 59]
    source = tmp_path / "calibration"
    source.mkdir()
    (source / "calibration.json").write_text(json.dumps(body))
    with pytest.raises(archive_cli.Refused):
        archive_cli.mode_archive_calibration(
            argparse.Namespace(out_dir=str(tmp_path), force=True))


def test_the_archive_can_be_replayed_and_says_what_a_replay_is(archive_cli):
    """The claim the CPU backend was chosen for, and never demonstrated.

    ``--mode verify`` re-decodes the sampled EXRs and re-derives their PNGs,
    which shows the archive is internally consistent; it does not show the
    render is reproducible. ``--mode replay`` runs Blender again from the
    archived scene record and compares the *decoded* pixels.

    The replay itself runs Blender, so it is not in the suite. This asserts the
    mode exists, compares the right things, and is honest about its scope: six
    replayed cells is not eighty.
    """
    source = Path("scripts/64_pbr_archive.py").read_text()
    assert "def mode_replay" in source
    assert "replay" in archive_cli.MODES
    for field in ("linear_pixel_digest_matches", "png_sha256_matches",
                  "uint8_digest_matches", "clipped_above_matches"):
        assert field in source, field
    # The claim string is wrapped across source lines, so check the parts that
    # survive wrapping rather than a literal that only exists at runtime.
    for part in ("a replay of the cells named here", "whole corpus"):
        assert part in source, (
            "a replay of some cells must not read as a replay of the "
            f"generation: {part!r} is missing")


def test_every_brick_is_one_layer_at_the_registered_plane():
    """The assumption the whole perspective registration rests on.

    Registering the brick top plane puts every top face on V1's own truth box
    *because* a pinhole images a single plane as a pure uniform scale about the
    optical axis -- which only helps if every top face is on one plane. V1's
    scenes are flat layouts, so they are; if a future scene stacked a brick, the
    registration would hold for one layer and silently miss the other, and the
    perspective arm would be measuring registration error again.
    """
    heights, bases, checked = set(), set(), 0
    for index in range(8):
        for condition in ("baseline", "background=grey", "background=dark",
                          "lighting=dim", "shadow=soft", "shadow=hard",
                          "occlusion=partial", "occlusion=heavy"):
            body = pbr.build_scene(index, condition)
            for brick in body["bricks"]:
                heights.add(brick["extent_scene_units"][2])
                bases.add(brick["translation_scene_units"][2])
                checked += 1
    assert checked == 416
    assert heights == {pbr.BRICK_TOP_Z}, (
        f"a brick top is not at z={pbr.BRICK_TOP_Z}: {sorted(heights)}; the "
        "perspective camera registers one plane and this scene has more")
    assert bases == {0}
