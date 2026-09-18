#!/usr/bin/env python3
"""Archives a V2 PBR generation into a tracked tree, and re-derives it.

Separate from ``scripts/62_visual_stress_pbr.py`` on purpose. That script
decides what is rendered and how it is scored, so its bytes are inside the
generation's source manifest and editing it has to open a new generation. This
one only copies finished bytes and re-derives what they claim, which cannot
change what a generation measured -- and three generations were voided in a row
because the two jobs shared one manifest and every fix to the archiver
invalidated a corpus that had already been rendered.

So the split is the fix, not a convenience: a post-hoc verifier has to be
improvable without re-rendering the thing it verifies. Phase 3C reached the
same shape, with ``scripts/61_phase3c_results.py`` sitting beside
``scripts/59_phase3c.py`` rather than inside it.

``--mode verify`` re-derives rather than re-reads. The index's own bookkeeping
is checked first -- every file present, every size and SHA-256 as claimed, and
nothing on disk the index does not account for -- and then the run is put back
through the frozen scorer, the report is re-rendered from the scores it claims
to describe, and the sampled linear EXRs are re-decoded and re-hashed. Checking
that an archive was filed honestly is not checking that its numbers came from
the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util                                     # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "pbr_pipeline", ROOT / "scripts" / "62_visual_stress_pbr.py")
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)

from src.vision.pbr import colour as pbr_colour           # noqa: E402
from src.vision.pbr import contract as pbr                # noqa: E402

GENERATION = pipeline.GENERATION
#: The illumination constants used by the current PBR contract were measured
#: and archived with v2gen08.  v2gen09--v2gen11 re-derived the same pixels
#: after source-closure changes; they did not re-run or relabel that
#: calibration.  Keep the two identities separate so the old archive's
#: ``how_to_recheck`` command continues to verify the evidence that actually
#: produced the constants.
CALIBRATION_GENERATION = "v2gen08"
FROZEN_DIR = pipeline.FROZEN_DIR
RESULTS_DIR = pipeline.RESULTS_DIR
Refused = pipeline.Refused
plain = pipeline.plain
load_authorization = pipeline.load_authorization
score_row = pipeline.score_row
render_report = pipeline.render_report
paired_contrast = pipeline.paired_contrast
camera_contrast = pipeline.camera_contrast
v1_rows = pipeline.v1_rows
source_manifests = pipeline.source_manifests


def archiver_identity() -> dict:
    """This script's own digest, recorded at archive time.

    Not part of the generation's manifest -- it cannot be, or improving the
    archiver would void the corpus. Recorded instead, so a reader can see
    which archiver produced an index and re-run that version if they want the
    bytes to match.
    """
    from src.training.session import sha256_file

    return {
        "archiver": "scripts/64_pbr_archive.py",
        "archiver_sha256": sha256_file(Path(__file__)),
        "pipeline": "scripts/62_visual_stress_pbr.py",
        "pipeline_sha256": sha256_file(
            ROOT / "scripts/62_visual_stress_pbr.py"),
        "note": ("the pipeline digest here is informational. The digest that "
                 "binds a generation is the one inside its authorisation"),
    }


def _relative(path: Path) -> str:
    """``path`` relative to the repository root, whether or not it was given so."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _calibration_generation(args) -> str:
    """Resolve a calibration archive without allowing path traversal."""
    value = getattr(args, "calibration_generation", None)
    value = CALIBRATION_GENERATION if value is None else value
    if not isinstance(value, str) or re.fullmatch(r"v2gen\d{2}", value) is None:
        raise Refused(f"invalid calibration generation {value!r}")
    return value


REPLAY_AUDIT_CELLS = 3


def mode_archive(args) -> int:
    """Copy the run into a tracked, write-once tree with an index.

    ``runs/`` is in ``.gitignore``, so a commit that only records the
    conclusion records no evidence -- the same gap Phase 3C's gen09 had to
    close after the fact. The canonical PNGs are what the scores are derived
    from, so all of them travel. The linear EXRs are 103 MB and only needed to
    re-render-and-compare, so a named sample travels and the rest are pinned by
    digest alone; the report may then say "sampled replay audit" and not
    "every image was re-rendered".
    """
    grant = load_authorization()
    out_dir = Path(args.out_dir)
    run = json.loads((out_dir / "run.json").read_text())
    if run["authorization_digest"] != grant["authorization_digest"]:
        raise Refused("the run was not produced under this authorisation")
    scores = json.loads((out_dir / "scores.json").read_text())
    report = (out_dir / "pbr_report.md")
    if not report.is_file():
        raise Refused("there is no report to archive")

    target = RESULTS_DIR / GENERATION
    overwrote = None
    if target.exists():
        if not args.force:
            raise Refused(f"{target} exists; the archive is write-once")
        # --force is a real overwrite of evidence, so it leaves a trace: what
        # was there is named in the new index rather than disappearing.
        previous = target / "index.json"
        if previous.is_file():
            overwrote = json.loads(previous.read_text()).get("index_digest")
        print(json.dumps({"forced_overwrite_of": _relative(target),
                          "previous_index_digest": overwrote}), flush=True)
    (target / "images").mkdir(parents=True, exist_ok=True)
    (target / "scenes").mkdir(parents=True, exist_ok=True)
    (target / "linear_sample").mkdir(parents=True, exist_ok=True)

    from src.training.session import sha256_file

    sampled = [c["cell_id"] for c in run["cells"][:REPLAY_AUDIT_CELLS]]
    files = {}

    def take(source: Path, relative: str):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files[relative] = {"bytes": destination.stat().st_size,
                           "sha256": sha256_file(destination)}

    for cell in run["cells"]:
        stem = cell["stem"]
        take(out_dir / f"{stem}.png", f"images/{stem}.png")
        take(out_dir / f"{stem}.scene.json", f"scenes/{stem}.scene.json")
        if cell["cell_id"] in sampled:
            take(out_dir / f"{stem}.exr", f"linear_sample/{stem}.exr")
    for name in ("run.json", "scores.json", "pbr_report.md"):
        take(out_dir / name, name)
    take(FROZEN_DIR / GENERATION / "authorization.json", "authorization.json")

    index = {
        "kind": "brickagain.pbr_result_index",
        "generation": GENERATION,
        "authorization_digest": grant["authorization_digest"],
        "run_digest": run["run_digest"],
        "scores_digest": scores["scores_digest"],
        "n_files": len(files),
        "files": files,
        "n_cells": len(run["cells"]),
        "images_archived": sum(1 for k in files if k.startswith("images/")),
        "linear_sample_cells": sampled,
        "linear_not_archived": [c["cell_id"] for c in run["cells"]
                                if c["cell_id"] not in sampled],
        "linear_pixel_digests": {c["cell_id"]: c["image"]["linear_pixel_digest"]
                                 for c in run["cells"]},
        "run_out_dir_on_the_mac": _relative(out_dir),
        "replay_claim": (
            f"{len(sampled)} of {len(run['cells'])} cells keep their linear "
            "EXR, so a re-render can be compared for those. Every cell's "
            "linear pixel digest is recorded, but a digest is not the pixels: "
            "the honest claim is a sampled replay audit, not that all "
            f"{len(run['cells'])} were re-rendered"),
        "how_to_recheck": ("./.venv/bin/python scripts/64_pbr_archive.py "
                           f"--mode verify --out-dir {_relative(out_dir)}"),
        "archiver_identity": archiver_identity(),
        "overwrote_index_digest": overwrote,
    }
    index["index_digest"] = pbr.scene_digest(
        {k: v for k, v in index.items() if k != "index_digest"})
    (target / "index.json").write_bytes(
        json.dumps(plain(index), indent=1, sort_keys=True).encode("utf-8"))
    print(json.dumps({"mode": "archive", "generation": GENERATION,
                      "target": _relative(target),
                      "n_files": len(files),
                      "megabytes": round(sum(
                          v["bytes"] for v in files.values()) / 1e6, 1),
                      "index_digest": index["index_digest"]}, indent=1))
    return 0


def extra_receipt_problems(body: dict, receipt: dict) -> list:
    """Comparisons this verifier adds on top of the frozen writer's.

    ``pbr.receipt_problems`` lives in ``src/vision/pbr/contract.py``, which the
    renderer manifest pins, so improving it moves the digest the authorisation
    binds and voids an already-rendered corpus. That is the coupling this file
    exists to break: a post-hoc verifier must be improvable without re-running
    the thing it verifies. So checks discovered *after* a freeze go here, and
    the writer stays byte-identical to what ran.

    (One visible cost of that discipline: the frozen ``receipt_problems``
    carries a dead ``if not want_brick["footprint_studs"]: continue`` at the
    end of its brick loop. Removing it would change the bytes the grant pins.
    It goes in the next generation, not in this one.)

    What is added:

    * The brick bounds the record implies. ``geometry_digest`` is recorded and
      is blind to connectivity and winding; the projection check reads the
      *record*, not the mesh. So a brick built at the wrong size, with a wrong
      bevel width, or with a stud missing was invisible to every check in the
      track. The bevel is inset, so the box is unchanged in x and y, and the
      studs raise the top by ``stud_height_ldu``.
    * The table's own height, which the ``plane_z`` field states and nothing
      compared.
    """
    out = []

    def check(where, want, got):
        if not pbr._near(want, got):
            out.append(f"{where}: record says {want!r}, Blender resolved {got!r}")

    resolved = receipt.get("resolved") or {}
    table, got_table = body["table"], resolved.get("table") or {}
    if "bounds_z" in got_table:
        check("table.plane_z", [-1.0, float(table["plane_z"])],
              got_table["bounds_z"])
    else:
        out.append("table.plane_z is stated by the record and the receipt "
                   "records no table bounds")

    mesh = body["mesh"]
    got_bricks = receipt.get("bricks") or []
    if len(got_bricks) == len(body["bricks"]):
        for want_brick, got_brick in zip(body["bricks"], got_bricks):
            name = want_brick["brick_id"]
            x0, y0, _z = want_brick["translation_scene_units"]
            width, depth, top = want_brick["extent_scene_units"]
            check(f"brick {name}.bounds_min", [x0, y0, 0],
                  got_brick.get("bounds_min"))
            check(f"brick {name}.bounds_max",
                  [x0 + width, y0 + depth, top + mesh["stud_height_ldu"]],
                  got_brick.get("bounds_max"))
    return out


def mode_verify(args) -> int:
    """Re-derive the archived result rather than re-read what it filed.

    "Re-derive" has to mean the frozen record decides what gets checked. The
    earlier version read its own coverage out of the artefacts under test: it
    iterated ``run["cells"]``, re-scored those, re-rendered the report from
    ``scores.json``'s own aggregate blocks, and printed ``verified: true``. A
    forger who trimmed the cell list, doctored the rows it no longer covered
    and recomputed the index digests passed. So the membership now comes from
    the digest-checked authorisation, the aggregates are recomputed from the
    rows, the images are tied to the digests the run recorded for them, and the
    receipts are compared against the scene records field by field.
    """
    target = RESULTS_DIR / GENERATION
    index_path = target / "index.json"
    if not index_path.is_file():
        raise Refused(f"{index_path} does not exist")
    index = json.loads(index_path.read_text())
    problems = []
    checked = {}

    recomputed = pbr.scene_digest(
        {k: v for k, v in index.items() if k != "index_digest"})
    if recomputed != index["index_digest"]:
        problems.append(f"index: {recomputed} vs {index['index_digest']}")

    from src.training.session import sha256_file

    on_disk = {str(p.relative_to(target)) for p in sorted(target.rglob("*"))
               if p.is_file() and p.name != "index.json"}
    claimed = set(index["files"])
    for extra in sorted(on_disk - claimed):
        problems.append(f"{extra} is on disk and no index entry accounts for it")
    for missing in sorted(claimed - on_disk):
        problems.append(f"{missing} is claimed and not on disk")
    for relative, entry in sorted(index["files"].items()):
        path = target / relative
        if not path.is_file():
            continue
        if path.stat().st_size != entry["bytes"]:
            problems.append(f"{relative} is {path.stat().st_size} bytes and "
                            f"the index says {entry['bytes']}")
        stored = sha256_file(path)
        if stored != entry["sha256"]:
            problems.append(f"{relative} hashes to {stored} and the index says "
                            f"{entry['sha256']}")

    # The flag was accepted and ignored, while index.json's how_to_recheck
    # passes it -- so following the archive's own instruction exercised a
    # parameter that did nothing. It now has to agree with the run directory
    # the archive says it came from.
    if getattr(args, "out_dir", None):
        given = _relative(Path(args.out_dir))
        recorded = index.get("run_out_dir_on_the_mac")
        if recorded and given != recorded:
            problems.append(
                f"--out-dir names {given!r} and the archive was filed from "
                f"{recorded!r}")

    grant = load_authorization()
    if index["authorization_digest"] != grant["authorization_digest"]:
        problems.append("the archive names a different authorisation")

    # Index fields that are pure claims, all checkable from data already here.
    on_disk_images = sum(1 for k in index["files"] if k.startswith("images/"))
    for field, want in (("n_files", len(index["files"])),
                        ("images_archived", on_disk_images)):
        if index.get(field) != want:
            problems.append(f"index says {field} is {index.get(field)} and the "
                            f"file list says {want}")

    # The archived grant against the frozen one, by bytes. The index's own
    # claimed digest is not evidence about the copy sitting beside it.
    frozen_grant = FROZEN_DIR / GENERATION / "authorization.json"
    archived_grant = target / "authorization.json"
    if archived_grant.is_file() and frozen_grant.is_file():
        if archived_grant.read_bytes() != frozen_grant.read_bytes():
            problems.append(
                "the archived authorization.json is not byte-identical to the "
                f"frozen {_relative(frozen_grant)}")

    # This generation has to be the live one. Nothing read this record before,
    # so "verified: true" said nothing about whether the thing verified had
    # since been voided.
    supersession = FROZEN_DIR / "supersession.json"
    if supersession.is_file():
        record = json.loads(supersession.read_text())
        voided = {entry["generation"] for entry in record["superseded"]}
        if GENERATION in voided:
            problems.append(
                f"{GENERATION} is recorded as superseded in "
                f"{_relative(supersession)} and may not be cited")
        checked["supersession_read"] = sorted(voided)

    # The live code against the manifests the authorisation binds. Everything
    # below re-derives *with the live code*, so a drifted tree would re-derive
    # a different answer and call the disagreement a defect in the archive.
    live = source_manifests()
    drifted = []
    for name in ("renderer_source_manifest_digest",
                 "evaluation_source_manifest_digest"):
        if live[name] != grant["sources"][name]:
            drifted.append(name)
            problems.append(
                f"{name}: the live tree is {live[name]} and the authorisation "
                f"binds {grant['sources'][name]}, so this re-derivation is not "
                "the one the grant covers")
    if drifted:
        # Stop here rather than re-derive. Re-scoring with code the grant does
        # not bind produces a number, and comparing that number to the archive
        # tests the wrong thing: a disagreement would be reported as a defect
        # in the evidence when it is a difference in the code. Which files
        # moved is named, so the reader can see the size of the drift.
        moved = {}
        for manifest in ("renderer_source_manifest",
                         "evaluation_source_manifest"):
            was = grant["sources"].get(manifest, {})
            now = live[manifest]
            moved[manifest] = sorted(
                {rel for rel in set(was) | set(now)
                 if was.get(rel) != now.get(rel)})
        checked["source_drift"] = moved
        print(json.dumps({"mode": "verify", "generation": GENERATION,
                          "directory": _relative(target),
                          "checked": checked,
                          "problems": problems, "verified": False}, indent=1))
        return 1

    run_path, scores_path = target / "run.json", target / "scores.json"
    if not (run_path.is_file() and scores_path.is_file()):
        problems.append("run.json or scores.json is missing; nothing to "
                        "re-derive")
    else:
        run = json.loads(run_path.read_text())
        scores = json.loads(scores_path.read_text())

        # Each record's own exclusion set, not a guess. run_digest covers the
        # cells and leaves out the receipts -- which are therefore anchored by
        # the index's hash of run.json and by the field-by-field comparison
        # below, not by run_digest -- and scores_digest covers everything but
        # the rows, which the re-scoring below covers instead.
        for name, body, omit in (("run", run, ("receipts",)),
                                 ("scores", scores, ("rows",))):
            key = f"{name}_digest"
            again = pbr.scene_digest(
                {k: v for k, v in body.items() if k not in (key,) + omit})
            if again != body[key]:
                problems.append(f"{key} recomputes to {again} and the record "
                                f"says {body[key]}")

        # Coverage from the frozen membership, not from the record under test.
        want = {cell["cell_id"] for cell in grant["membership"]}
        got_run = {cell["cell_id"] for cell in run["cells"]}
        got_scores = {row["cell_id"] for row in scores["rows"]}
        if index.get("n_cells") != len(want):
            problems.append(
                f"index says n_cells is {index.get('n_cells')} and the frozen "
                f"membership names {len(want)}")
        if got_run != want:
            problems.append(
                f"run.json covers {len(got_run)} cells and the frozen "
                f"membership names {len(want)}; missing "
                f"{sorted(want - got_run)[:5]}, unexpected "
                f"{sorted(got_run - want)[:5]}")
        if got_scores != want:
            problems.append(
                f"scores.json covers {len(got_scores)} cells and the frozen "
                f"membership names {len(want)}")
        checked["cells_from_frozen_membership"] = len(want)

        # Each archived scene record against the digest the membership pins.
        hashed = 0
        for cell in grant["membership"]:
            # The stem is the cell id with the separator swapped, the same
            # formula the run uses; the membership pins the cell id, which is
            # the thing that has to match.
            stem = cell["cell_id"].replace("/", "__")
            path = target / f"scenes/{stem}.scene.json"
            if not path.is_file():
                problems.append(f"scenes/{stem}.scene.json is not "
                                "archived and the membership names it")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashed += 1
            if digest != cell["scene_digest"]:
                problems.append(
                    f"{cell['cell_id']}: the archived scene record hashes to "
                    f"{digest} and the membership pins {cell['scene_digest']}")
        # The number hashed, not the number of candidates. A count that reports
        # what it was *asked* to check reads as coverage it may not have.
        checked["scene_records_against_membership"] = hashed
        if hashed != len(grant["membership"]):
            problems.append(f"{hashed} of {len(grant['membership'])} scene "
                            "records were hashed against the membership")

        # Each archived image against the digest the *run* recorded for it, not
        # against the index's own claim about the file beside it.
        by_id = {cell["cell_id"]: cell for cell in run["cells"]}
        compared_images = 0
        for cell_id in sorted(want & set(by_id)):
            cell = by_id[cell_id]
            png = target / f"images/{cell['stem']}.png"
            if not png.is_file():
                problems.append(f"images/{cell['stem']}.png is missing")
                continue
            stored = sha256_file(png)
            compared_images += 1
            if stored != cell["image"]["png_sha256"]:
                problems.append(
                    f"{cell_id}: the archived PNG hashes to {stored} and the "
                    f"run recorded {cell['image']['png_sha256']}")
        checked["images_against_the_run_record"] = compared_images
        if compared_images != len(want):
            problems.append(f"{compared_images} of {len(want)} archived images "
                            "were compared to the run's own digests")

        # Each receipt against its scene record, field by field. This is the
        # check that would have caught a light aimed 58.992 degrees away from
        # the centre of a scene whose record said it aimed at it.
        # The receipts live in their own top-level dict, not on the cells: the
        # run strips them off before hashing so run_digest covers the cells
        # without covering the receipts. Reading them off the cells found
        # nothing, and the count reported the number of *candidates* rather
        # than the number compared -- so it said 80 while comparing 0. The
        # count is now what was actually compared, and a cell whose receipt is
        # missing is a problem rather than a silent skip.
        receipts = run.get("receipts") or {}
        receipt_problems, compared = 0, 0
        for cell_id in sorted(want & set(by_id)):
            cell = by_id[cell_id]
            scene_path = target / f"scenes/{cell['stem']}.scene.json"
            receipt = receipts.get(cell_id)
            if receipt is None:
                problems.append(f"{cell_id}: the run carries no receipt, so "
                                "nothing compares it to its record")
                continue
            if not scene_path.is_file():
                problems.append(f"{cell_id}: no archived scene record to "
                                "compare the receipt against")
                continue
            raw = scene_path.read_bytes()
            body = json.loads(raw.decode("utf-8"))
            found = pbr.receipt_problems(body, receipt)
            found += pbr.scene_digest_problems(receipt, raw)
            found += extra_receipt_problems(body, receipt)
            compared += 1
            for entry in found:
                receipt_problems += 1
                problems.append(f"{cell_id}: {entry}")
        checked["receipts_compared_to_their_records"] = compared
        checked["receipt_disagreements"] = receipt_problems
        if compared != len(want):
            problems.append(
                f"{compared} of {len(want)} receipts were compared to their "
                "records")

        # The rows, re-derived from the archived images by the live scorer.
        rows_by_id = {row["cell_id"]: row for row in scores["rows"]}
        rederived = []
        for cell_id in sorted(want & set(by_id) & set(rows_by_id)):
            cell, row = by_id[cell_id], rows_by_id[cell_id]
            png = target / f"images/{cell['stem']}.png"
            if not png.is_file():
                continue
            again = score_row(cell["scene_index"], cell["condition_id"], png)
            again["cell_id"] = cell_id
            again["camera"] = cell["camera"]
            again["scene_id"] = cell["scene_id"]
            again["paired_with_v1"] = cell["paired_with_v1"]
            rederived.append(again)
            for field in ("n_true_bricks", "n_detected", "false_negatives",
                          "false_positives", "abstentions", "n_part_errors",
                          "n_colour_errors", "top1_correct_among_matched",
                          "inventory_exact_match", "mean_confidence"):
                if again[field] != row[field]:
                    problems.append(
                        f"scores: {cell_id} {field} re-derives to "
                        f"{again[field]} and the record says {row[field]}")
        checked["rows_rescored"] = len(rederived)

        # The aggregates, recomputed from the rows. Nothing recomputed these
        # before, and they are the numbers the report and both documents quote.
        if len(rederived) == len(want):
            again_paired = paired_contrast(rederived, v1_rows())
            again_camera = camera_contrast(rederived)
            for name, got, want_block in (
                    ("paired_contrast_v2_minus_v1", again_paired,
                     scores["paired_contrast_v2_minus_v1"]),
                    ("camera_contrast_perspective_minus_ortho", again_camera,
                     scores["camera_contrast_perspective_minus_ortho"])):
                if plain(got) != plain(want_block):
                    problems.append(
                        f"{name} recomputes from the rows to something other "
                        "than what scores.json publishes")
        checked["aggregates_recomputed_from_rows"] = (
            len(rederived) == len(want))
        if len(rederived) != len(want):
            problems.append(
                f"only {len(rederived)} of {len(want)} rows re-derived, so the "
                "aggregates the report quotes were not recomputed")

        # The disclaimers against the frozen contract, not against a copy the
        # scorer made of them.
        if scores.get("claims_forbidden") != grant["contract"]["claims_forbidden"]:
            problems.append(
                "scores.json's claims_forbidden is not the frozen contract's")
        for field, want_value in (("authorization_digest",
                                   grant["authorization_digest"]),
                                  ("run_digest", run["run_digest"])):
            if scores.get(field) != want_value:
                problems.append(
                    f"scores.json names {field} {scores.get(field)!r} and it "
                    f"should be {want_value!r}")

        # The report, re-rendered from the scores it claims to describe.
        expected = render_report(scores, grant)
        stored = (target / "pbr_report.md").read_text(encoding="utf-8")
        if expected != stored:
            problems.append(
                "report: pbr_report.md is not what the renderer produces from "
                "these scores")

        # The linear sample: re-decoded, re-hashed, and -- the link that was
        # open -- converted again to a PNG so the archived EXR and the archived
        # PNG are shown to be the same picture. The pinned colour pipeline was
        # the one step verify did not close.
        reconverted = 0
        for cell_id in index["linear_sample_cells"]:
            cell = by_id.get(cell_id)
            if cell is None:
                problems.append(f"{cell_id} is sampled and not in the run")
                continue
            exr = target / f"linear_sample/{cell['stem']}.exr"
            if not exr.is_file():
                problems.append(f"linear_sample/{cell['stem']}.exr is missing")
                continue
            array = pbr_colour.read_exr(exr)
            digest = pbr.pixel_digest(array)
            if digest != index["linear_pixel_digests"][cell_id]:
                problems.append(
                    f"linear: {cell_id} decodes to {digest} and the index says "
                    f"{index['linear_pixel_digests'][cell_id]}")
            scene_path = target / f"scenes/{cell['stem']}.scene.json"
            if scene_path.is_file():
                body = json.loads(scene_path.read_text())
                with tempfile.TemporaryDirectory() as tmp:
                    again = pbr_colour.to_canonical_png(
                        exr, Path(tmp) / "again.png", body["png"])
                reconverted += 1
                # Both digests. png_sha256 moves with the PNG encoder;
                # uint8_digest is the encoder-independent one the colour module
                # computes for exactly that reason and that nothing read.
                for field in ("uint8_digest",):
                    if field in cell["image"] and again.get(field) != cell["image"][field]:
                        problems.append(
                            f"colour: {cell_id} re-converting gives {field} "
                            f"{again.get(field)} and the run recorded "
                            f"{cell['image'][field]}")
                if again["png_sha256"] != cell["image"]["png_sha256"]:
                    problems.append(
                        f"colour: {cell_id} re-converting the archived EXR "
                        f"gives PNG {again['png_sha256']} and the run recorded "
                        f"{cell['image']['png_sha256']}")
        checked["exr_reconverted_to_png"] = reconverted
        if reconverted != len(index["linear_sample_cells"]):
            problems.append(
                f"{reconverted} of {len(index['linear_sample_cells'])} sampled "
                "EXRs were re-converted and compared to their PNG")

        # The archiver that filed this, against the manifest entry that gives
        # the claim meaning.
        identity = index.get("archiver_identity") or {}
        live_identity = archiver_identity()
        if identity.get("sha256") != live_identity.get("sha256"):
            problems.append(
                "the archive was filed by a different version of this script: "
                f"{identity.get('sha256')} vs {live_identity.get('sha256')}")

    print(json.dumps({"mode": "verify", "generation": GENERATION,
                      "directory": _relative(target),
                      "checked": checked,
                      "problems": problems, "verified": not problems},
                     indent=1))
    return 0 if not problems else 1


def mode_archive_calibration(args) -> int:
    """Put the illumination calibration into the tracked tree.

    The finding this closes: the contract's measured illumination figures were
    typed in from development renders, sitting inside every scene digest, while
    no script derived them, no artefact held the images and no test compared
    them -- and two of them were wrong. ``--mode calibrate`` derives them; this
    files the derivation so the numbers are checkable rather than asserted.

    The EXRs do not travel: the PNGs are what ``measure_render`` reads for the
    8-bit figures, the receipts carry what Blender resolved, and the scene
    records say what was asked for. The linear maxima and clip counts are in
    ``calibration.json`` itself.
    """
    from src.training.session import sha256_file

    source = Path(args.out_dir) / "calibration"
    body_path = source / "calibration.json"
    if not body_path.is_file():
        raise Refused(f"{body_path} does not exist; run --mode calibrate first")
    calibration = json.loads(body_path.read_text())

    if calibration.get("generation") != GENERATION:
        raise Refused(
            "the calibration belongs to generation "
            f"{calibration.get('generation')!r}, not {GENERATION!r}; archived "
            "evidence must not be relabelled")

    problems = pipeline.calibration_problems(calibration)
    if problems:
        raise Refused("the calibration does not agree with the contract, so "
                      f"it is not filed: {problems}")

    target = ROOT / "data/phase_v2/calibration" / GENERATION
    if target.exists() and not args.force:
        raise Refused(f"{target} exists; the archive is write-once")
    target.mkdir(parents=True, exist_ok=True)
    files = {}
    for path in sorted(source.iterdir()):
        if path.suffix == ".exr" or not path.is_file():
            continue
        shutil.copy2(path, target / path.name)
        files[path.name] = {"bytes": path.stat().st_size,
                            "sha256": sha256_file(target / path.name)}
    index = {
        "kind": "brickagain.pbr_calibration_index",
        "generation": GENERATION,
        "calibration_digest": pbr.scene_digest(calibration),
        "n_files": len(files),
        "files": files,
        "exr_not_archived": sorted(p.name for p in source.glob("*.exr")),
        "claim": ("the illumination figures the contract records, derived by "
                  "measurement rather than typed in. The light powers are "
                  "solved from two renders each, not tuned"),
        "how_to_recheck": ("./.venv/bin/python scripts/64_pbr_archive.py "
                           "--mode verify-calibration "
                           f"--calibration-generation {GENERATION}"),
        "archiver_identity": archiver_identity(),
    }
    index["index_digest"] = pbr.scene_digest(
        {k: v for k, v in index.items() if k != "index_digest"})
    (target / "index.json").write_bytes(
        json.dumps(plain(index), indent=1, sort_keys=True).encode("utf-8"))
    print(json.dumps({"mode": "archive-calibration", "generation": GENERATION,
                      "target": _relative(target), "n_files": len(files),
                      "megabytes": round(sum(v["bytes"] for v in files.values())
                                         / 1e6, 2),
                      "index_digest": index["index_digest"]}, indent=1))
    return 0


def mode_verify_calibration(args) -> int:
    """Re-derive the calibration's verdict from the archived files."""
    generation = _calibration_generation(args)
    target = ROOT / "data/phase_v2/calibration" / generation
    index_path = target / "index.json"
    if not index_path.is_file():
        raise Refused(f"{index_path} does not exist")
    index = json.loads(index_path.read_text())
    problems = []

    from src.training.session import sha256_file

    recomputed = pbr.scene_digest(
        {k: v for k, v in index.items() if k != "index_digest"})
    if recomputed != index["index_digest"]:
        problems.append(f"index: {recomputed} vs {index['index_digest']}")
    if index.get("generation") != generation:
        problems.append(
            f"index generation is {index.get('generation')!r}, expected "
            f"{generation!r}")
    on_disk = {p.name for p in target.iterdir()
               if p.is_file() and p.name != "index.json"}
    for extra in sorted(on_disk - set(index["files"])):
        problems.append(f"{extra} is on disk and unaccounted for")
    for missing in sorted(set(index["files"]) - on_disk):
        problems.append(f"{missing} is claimed and not on disk")
    for name, entry in sorted(index["files"].items()):
        path = target / name
        if path.is_file() and sha256_file(path) != entry["sha256"]:
            problems.append(f"{name} hashes to {sha256_file(path)} and the "
                            f"index says {entry['sha256']}")

    calibration = json.loads((target / "calibration.json").read_text())
    if pbr.scene_digest(calibration) != index["calibration_digest"]:
        problems.append("calibration.json does not match the digest the index "
                        "records for it")
    # The verdict, re-derived against the live contract.
    problems += pipeline.calibration_problems(calibration)
    # And each receipt against its scene record, the same comparison the run
    # makes -- so the calibration renders are held to the same standard.
    compared = 0
    for name in sorted(index["files"]):
        if not name.endswith(".receipt.json"):
            continue
        stem = name[: -len(".receipt.json")]
        scene_path = target / f"{stem}.scene.json"
        if not scene_path.is_file():
            problems.append(f"{stem}: no archived scene record")
            continue
        raw = scene_path.read_bytes()
        body = json.loads(raw.decode("utf-8"))
        receipt = json.loads((target / name).read_text())
        found = pbr.receipt_problems(body, receipt)
        found += pbr.scene_digest_problems(receipt, raw)
        found += extra_receipt_problems(body, receipt)
        compared += 1
        problems += [f"{stem}: {entry}" for entry in found]

    print(json.dumps({"mode": "verify-calibration", "generation": generation,
                      "contract_generation": GENERATION,
                      "directory": _relative(target),
                      "checked": {"receipts_compared": compared},
                      "problems": problems, "verified": not problems},
                     indent=1))
    return 0 if not problems else 1


def mode_replay(args) -> int:
    """Re-render archived cells and compare the pixels, not a digest of them.

    ``--mode verify`` re-decodes the sampled EXRs and re-derives their PNGs,
    which shows the archive is internally consistent. It does not show the
    render is reproducible: for that, Blender has to run again from the
    archived scene record and land on the same pixels. That is the claim the
    CPU backend was chosen for -- three measured CPU renders were
    byte-identical where three Metal renders were not -- and it had never been
    demonstrated against a formal generation.

    Compared on the *decoded* pixel digest rather than the file bytes. OpenEXR's
    ZIP is lossless but its compressed bytes move with the library version, so a
    container digest would fail on an upgrade that changed no pixel.
    """
    target = RESULTS_DIR / GENERATION
    run = json.loads((target / "run.json").read_text())
    grant = load_authorization()
    if run["authorization_digest"] != grant["authorization_digest"]:
        raise Refused("the archived run was not produced under this grant")

    if getattr(args, "cell", None):
        wanted = [c for c in run["cells"] if c["cell_id"] == args.cell]
        if not wanted:
            raise Refused(f"{args.cell!r} is not a cell of {GENERATION}")
        cells = wanted
    else:
        cells = run["cells"][: int(args.cells)]
    out_dir = Path(tempfile.mkdtemp(prefix="pbr_replay_"))
    rows, problems = [], []
    for cell in cells:
        stem = cell["stem"]
        scene_path = target / f"scenes/{stem}.scene.json"
        raw = scene_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != cell["scene_digest"]:
            problems.append(f"{cell['cell_id']}: the archived scene record is "
                            "not the one the run rendered")
            continue
        body = json.loads(raw.decode("utf-8"))
        record = pipeline.render_one(body, cell["camera"], run["backend"],
                                     out_dir, stem)
        again = record["image"]
        row = {
            "cell_id": cell["cell_id"],
            "seconds": record["wall_seconds"],
            "linear_pixel_digest_matches":
                again["linear_pixel_digest"] == cell["image"]["linear_pixel_digest"],
            "png_sha256_matches":
                again["png_sha256"] == cell["image"]["png_sha256"],
            "uint8_digest_matches":
                again.get("uint8_digest") == cell["image"].get("uint8_digest"),
            "clipped_above_matches":
                again["clipped_above"] == cell["image"]["clipped_above"],
        }
        receipt_found = pbr.receipt_problems(body, record["receipt"])
        receipt_found += extra_receipt_problems(body, record["receipt"])
        row["receipt_problems"] = receipt_found
        rows.append(row)
        for field in ("linear_pixel_digest_matches", "png_sha256_matches",
                      "uint8_digest_matches", "clipped_above_matches"):
            if not row[field]:
                problems.append(f"{cell['cell_id']}: {field} is false")
        problems += [f"{cell['cell_id']}: {entry}" for entry in receipt_found]
        print(json.dumps({"replayed": cell["cell_id"],
                          "seconds": record["wall_seconds"],
                          "pixels_identical": row["linear_pixel_digest_matches"]}),
              flush=True)

    print(json.dumps({"mode": "replay", "generation": GENERATION,
                      "backend": run["backend"],
                      "n_replayed": len(rows),
                      "n_cells_in_the_generation": len(run["cells"]),
                      "rows": rows,
                      "claim": ("re-rendered from the archived scene record by "
                                "the same backend on the same machine. This is "
                                "a replay of the cells named here, not of the "
                                "whole corpus"),
                      "problems": problems,
                      "replayed_identically": not problems}, indent=1))
    return 0 if not problems else 1


MODES = {"archive": mode_archive, "verify": mode_verify,
         "archive-calibration": mode_archive_calibration,
         "verify-calibration": mode_verify_calibration,
         "replay": mode_replay}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--out-dir",
                        default=str(ROOT / "runs/pbr" / GENERATION))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cells", default=2,
                        help="how many archived cells to re-render")
    parser.add_argument("--cell", default=None,
                        help="replay one named cell instead of the first N")
    parser.add_argument(
        "--calibration-generation", default=CALIBRATION_GENERATION,
        help=("calibration archive to verify; defaults to the generation that "
              "supplies the current contract's illumination constants"))
    args = parser.parse_args(argv)
    try:
        return MODES[args.mode](args)
    except Refused as exc:
        print(json.dumps({"refused": str(exc)}, indent=1), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
