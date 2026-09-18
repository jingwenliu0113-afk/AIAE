"""Verifies the archived V3 probe. Separate from the probe on purpose.

``scripts/63_v3_diffusion_probe.py`` is pinned inside ``v3gen01``'s frozen
plan, so improving a verifier that lived inside it would move the digest the
plan binds and void an already-generated corpus -- the coupling that voided six
V2 generations in a row before the archiver was split out. This file is
therefore outside the plan by design: it may be improved after the fact, and it
says so in what it prints.

What it can and cannot close:

* It re-derives the response-mode distribution from the retained images by
  running the *live* recogniser, and reproduces ``distribution_digest``. That
  is the probe's whole result, so this is a real re-derivation and not a
  re-read.
* It cannot re-derive the *images*: diffusion on MPS is not bit-reproducible
  across library versions, which is why the generation ledger records a digest
  per attempt instead. Every retained image is checked against the digest the
  generation recorded for it, so a substituted image is caught even though a
  re-generation is not possible.
* The plan pins ``src/vision/pbr/contract.py`` because the probe borrows its
  canonical-JSON hashing. That module has since moved for reasons that have
  nothing to do with V3, so the drift is reported by name rather than hidden:
  the recogniser closure is what decides this result, and it is checked
  separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "v3_probe", ROOT / "scripts/63_v3_diffusion_probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

from src.vision.pbr import contract as pbr                  # noqa: E402

GENERATION = probe.GENERATION
RESULTS_DIR = ROOT / "data/phase_v2/results"
FROZEN_DIR = ROOT / "data/phase_v2/frozen"

#: The recogniser's entry point. Its closure is *derived* from the code below,
#: not listed here.
#:
#: An earlier version of this file carried a hand-written nine-entry tuple and
#: called it the recogniser, treating drift anywhere else as merely
#: reportable. That was wrong twice over: the real closure of
#: ``src.ui.full.analyse_photo`` is 49 of the plan's 56 files -- the list
#: missed 40 of them, including ``src/vision/metrics.py`` and
#: ``src/vision/classes.py`` -- and "not in my list" is not the same claim as
#: "cannot change the answer". Drift is now fatal by default.
RECOGNISER_ENTRY_POINT = "src/ui/full.py"

#: Frozen files whose drift is *not* fatal. Each needs a reason a reader can
#: check, and each reason is machine-checked below: the file must be absent
#: from the recogniser's derived closure, or the exemption is void and the
#: drift becomes fatal again. An exemption that cannot fail is not an
#: exemption, it is a hole.
NON_DECISIVE = {
    "src/vision/pbr/contract.py": (
        "the probe imports it for one thing: scene_digest, the canonical-JSON "
        "hashing helper it uses to hash the plan body and the distribution "
        "body. It reads no image, proposes no box and reads no colour, so it "
        "cannot move a response mode. Proven, not asserted: this verifier "
        "requires it to be absent from the recogniser's own import closure"),
}


def recogniser_closure(root: Path | None = None) -> set:
    """Every module the recogniser reaches, derived from the source.

    A static import closure over-approximates what actually executes -- a
    module imported on a branch this run never takes is still in it. That is
    the direction to err in: over-approximating makes more drift fatal, not
    less.

    ``src/training/pack.py``, which computes this, is itself inside the closure
    and inside the frozen plan, so drift in the tool that derives the closure
    is itself fatal.
    """
    from src.training import pack

    base = Path(root or ROOT)
    reached = pack.import_closure(root=base,
                                  entry_points=(RECOGNISER_ENTRY_POINT,))
    return set(reached) | {RECOGNISER_ENTRY_POINT}


class Refused(RuntimeError):
    """Raised rather than continuing with a weaker claim."""


def verifier_identity() -> dict:
    from src.training.session import sha256_file

    return {"path": "scripts/65_v3_archive.py",
            "sha256": sha256_file(Path(__file__)),
            "note": ("outside the frozen plan by design, so it can be "
                     "improved without voiding a generated corpus")}


def recognise_one(raw: bytes, attempt_id: str, scene_tag: str) -> dict:
    """One image through the live recogniser, in the probe's own row shape.

    The probe computes this inline inside ``mode_recognise`` and the probe is
    pinned inside the frozen plan, so it cannot be refactored into a shared
    helper without moving the digest the plan binds. The shape is therefore
    restated here, and ``--mode verify`` reproducing ``distribution_digest`` is
    what shows the restatement is faithful: any field that disagreed would move
    the digest.
    """
    import time

    from src.ui.full import PHOTO_MULTI, RECOGNISE_CV, analyse_photo

    started = time.time()
    try:
        analysis = analyse_photo(raw, mode=PHOTO_MULTI, method=RECOGNISE_CV)
    except Exception as exc:                      # noqa: BLE001 - recorded
        return {"attempt_id": attempt_id, "scene_tag": scene_tag,
                "mode": "recognizer_crash",
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "seconds": round(time.time() - started, 3)}
    seconds = time.time() - started
    return {
        "attempt_id": attempt_id,
        "scene_tag": scene_tag,
        "mode": probe.response_mode(analysis, seconds),
        "seconds": round(seconds, 3),
        "n_items": len(analysis.items),
        "n_colour_problems": len(analysis.colour_problems),
        "mean_part_confidence": round(sum(
            (i.predicted_confidence or 0.0) for i in analysis.items)
            / len(analysis.items), 4) if analysis.items else None,
        "proposed_inventory_size": sum(
            1 for i in analysis.items
            if i.predicted_part and i.predicted_part != "unknown"),
    }


def summarise(rows: list, frozen: dict, outcome_digest: str) -> dict:
    """The distribution body, in the probe's own shape and order."""
    modes = frozen["response_modes"] + ["recognizer_crash"]
    per_prompt = {}
    for row in rows:
        bucket = per_prompt.setdefault(row["scene_tag"],
                                      {"n_produced": 0,
                                       **{m: 0 for m in modes}})
        bucket["n_produced"] += 1
        bucket[row["mode"]] = bucket.get(row["mode"], 0) + 1
    body = {
        "kind": "brickagain.v3_response_mode_distribution",
        "generation": GENERATION,
        "authorization_digest": frozen["authorization_digest"],
        "outcome_digest": outcome_digest,
        "denominator_n_produced": len(rows),
        "denominator_note": ("n_produced, not 96. The generation ledger keeps "
                             "the fixed denominator of 96 attempts; this one "
                             "is over the images that exist"),
        "counts": {mode: sum(1 for r in rows if r["mode"] == mode)
                   for mode in modes},
        "per_prompt": per_prompt,
        "rows": rows,
        "forbidden": frozen["forbidden"],
        "note": ("These are counts of what the recogniser did on a fixed "
                 "corpus. They are computed, not estimated: the corpus is the "
                 "population, so no interval belongs on them. Nothing here "
                 "says whether a confident answer was correct"),
    }
    body["distribution_digest"] = pbr.scene_digest(
        {k: v for k, v in body.items() if k != "rows"})
    return body


def mode_verify(args) -> int:
    """Re-derive the distribution from the retained images."""
    target = RESULTS_DIR / GENERATION
    index_path = target / "index.json"
    if not index_path.is_file():
        raise Refused(f"{index_path} does not exist")
    index = json.loads(index_path.read_text())
    plan_path = FROZEN_DIR / GENERATION / "plan.json"
    frozen = json.loads(plan_path.read_text())
    problems, checked = [], {}

    from src.training.session import sha256_file

    recomputed = pbr.scene_digest(
        {k: v for k, v in index.items() if k != "index_digest"})
    if recomputed != index["index_digest"]:
        problems.append(f"index: {recomputed} vs {index['index_digest']}")

    on_disk = {str(p.relative_to(target)) for p in sorted(target.rglob("*"))
               if p.is_file() and p.name != "index.json"}
    for extra in sorted(on_disk - set(index["files"])):
        problems.append(f"{extra} is on disk and no index entry accounts for it")
    for missing in sorted(set(index["files"]) - on_disk):
        problems.append(f"{missing} is claimed and not on disk")
    for relative, entry in sorted(index["files"].items()):
        path = target / relative
        if not path.is_file():
            continue
        stored = sha256_file(path)
        if stored != entry["sha256"]:
            problems.append(f"{relative} hashes to {stored} and the index "
                            f"says {entry['sha256']}")

    # The archived plan against the frozen one, by bytes.
    archived_plan = target / "plan.json"
    if archived_plan.is_file() and archived_plan.read_bytes() != plan_path.read_bytes():
        problems.append("the archived plan.json is not byte-identical to the "
                        "frozen one")

    # The two independently derived digests of each sampled image, which the
    # archive held side by side and never compared.
    compared = 0
    sampled = index.get("images_sampled") or []
    for attempt_id in sampled:
        relative = f"images_sample/{attempt_id.replace('/', '__')}.png"
        digest = (index.get("image_digests") or {}).get(attempt_id)
        if relative not in index["files"]:
            problems.append(f"{attempt_id} is listed as sampled and "
                            f"{relative} has no index entry")
            continue
        if digest is None:
            problems.append(f"{attempt_id} is sampled and the generation "
                            "ledger recorded no digest for it")
            continue
        compared += 1
        if index["files"][relative]["sha256"] != digest:
            problems.append(
                f"{attempt_id}: the archived copy hashes to "
                f"{index['files'][relative]['sha256']} and the generation "
                f"ledger recorded {digest}")
    # The number compared, not the number listed.
    checked["sampled_images_against_the_generation_ledger"] = compared
    if compared != len(sampled):
        problems.append(f"{compared} of {len(sampled)} sampled images were "
                        "compared to the generation ledger")

    # Source drift. Fail-closed: every drifted file the plan pins is treated as
    # able to change the answer, unless it is named in NON_DECISIVE *and* that
    # exemption's proof still holds.
    manifest = frozen["source_manifest"]["files"]
    moved = sorted(rel for rel, want in manifest.items()
                   if not (ROOT / rel).is_file()
                   or sha256_file(ROOT / rel) != want)
    closure = recogniser_closure()
    checked["recogniser_closure_size"] = len(closure)
    checked["frozen_files_in_the_recogniser_closure"] = len(
        set(manifest) & closure)
    checked["frozen_files_outside_it"] = sorted(set(manifest) - closure)

    for rel, reason in sorted(NON_DECISIVE.items()):
        if rel not in manifest:
            problems.append(
                f"{rel} is excused as non-decisive and the plan does not pin "
                "it, so the exemption is about nothing")
        elif rel in closure:
            problems.append(
                f"{rel} is excused as non-decisive on the grounds that the "
                "recogniser does not reach it, and the recogniser DOES reach "
                "it. The exemption is void")

    excused = [rel for rel in moved
               if rel in NON_DECISIVE and rel not in closure]
    decisive = [rel for rel in moved if rel not in excused]
    checked["source_drift_excused"] = [
        {"path": rel, "why": NON_DECISIVE[rel]} for rel in excused]
    checked["source_drift_fatal"] = decisive
    if decisive:
        problems.append(
            "files the frozen plan pins have moved, so a re-derived "
            "distribution is not the plan's. Drift is fatal unless the file is "
            "named in NON_DECISIVE with a reason this verifier can check: "
            f"{decisive}")

    # The distribution, re-derived by running the live recogniser over the
    # retained images.
    distribution = json.loads(
        (target / "response_mode_distribution.json").read_text())
    images_dir = Path(args.images_dir) if args.images_dir else (
        ROOT / "runs/v3" / GENERATION / "images")
    if not decisive and images_dir.is_dir():
        outcome = json.loads((target / "generation_outcome.json").read_text())
        attempts = {a["attempt_id"]: a for a in outcome["attempts"]
                    if a.get("png_sha256")}
        rows, missing = [], []
        for row in distribution["rows"]:
            attempt = attempts.get(row["attempt_id"])
            # The attempt id carries a slash and the filename does not.
            path = images_dir / f"{row['attempt_id'].replace('/', '__')}.png"
            if attempt is None or not path.is_file():
                missing.append(row["attempt_id"])
                continue
            raw = path.read_bytes()
            stored = hashlib.sha256(raw).hexdigest()
            if stored != attempt["png_sha256"]:
                problems.append(
                    f"{row['attempt_id']}: the retained image hashes to "
                    f"{stored} and the generation recorded "
                    f"{attempt['png_sha256']}")
                continue
            again = recognise_one(raw, row["attempt_id"], row["scene_tag"])
            rows.append(again)
            if again["mode"] != row["mode"]:
                problems.append(
                    f"{row['attempt_id']}: re-deriving gives mode "
                    f"{again['mode']!r} and the record says {row['mode']!r}")
        checked["rows_rederived"] = len(rows)
        checked["rows_whose_image_is_no_longer_on_disk"] = missing
        if missing:
            problems.append(
                f"{len(missing)} of {len(distribution['rows'])} images are no "
                "longer on disk, so the distribution cannot be fully "
                "re-derived; only the 8 sampled images travel with the archive")
        elif rows:
            again = summarise(rows, frozen, distribution["outcome_digest"])
            if again["distribution_digest"] != distribution["distribution_digest"]:
                problems.append(
                    "the re-derived distribution digest is "
                    f"{again['distribution_digest']} and the record says "
                    f"{distribution['distribution_digest']}")
            else:
                checked["distribution_digest_reproduced"] = (
                    distribution["distribution_digest"])
    elif decisive:
        # Deliberately not re-derived. A distribution computed with code the
        # plan does not bind is not evidence about the plan, and reporting it
        # beside the plan's own digest would invite the comparison.
        problems.append(
            "nothing was re-derived: the drift above means a re-derivation "
            "would not be the plan's")
    else:
        problems.append(
            f"{images_dir} is not a directory, so nothing could be "
            "re-derived; pass --images-dir")

    # The claims the plan forbids have to still be absent from the record.
    text = json.dumps(distribution, ensure_ascii=False)
    for claim in frozen["forbidden"]:
        if claim in text and claim not in json.dumps(distribution["forbidden"]):
            problems.append(f"the record makes a claim the plan forbids: {claim!r}")

    print(json.dumps({"mode": "verify", "generation": GENERATION,
                      "directory": str(target.relative_to(ROOT)),
                      "verifier": verifier_identity(),
                      "checked": checked,
                      "problems": problems, "verified": not problems},
                     indent=1, ensure_ascii=False))
    return 0 if not problems else 1


MODES = {"verify": mode_verify}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--images-dir", default=None)
    args = parser.parse_args(argv)
    try:
        return MODES[args.mode](args)
    except Refused as exc:
        print(json.dumps({"refused": str(exc)}, indent=1))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
