#!/usr/bin/env python3
"""V3: a fixed local diffusion checkpoint as an out-of-distribution probe.

V3 is not a measurement of accuracy and cannot be made into one. A diffusion
model asked for three orange 2x4 bricks does not produce three orange 2x4
bricks; it produces something nearby, with the wrong stud count, a drifted
colour, or an extra brick. Scoring that would need either hand annotation --
which reintroduces the annotation error V1's constructed ground truth removed
-- or a claim that the prompt is the ground truth, which it is not.

So V3 reports two things and keeps them in separate ledgers, because they
answer different questions and pooling them makes both uninterpretable:

``generation_outcome``
    Denominator 96, fixed: eight prompts by twelve seeds, named before the
    model was loaded. Every attempt is accounted for, including the ones that
    produced nothing. A failed attempt is not replaced.

``response_mode_distribution``
    Denominator ``n_produced``, stated in the record. What the recogniser
    *did* -- refused, was unsure, was confident, crashed, timed out. Both the
    numerator and the denominator are observable without ground truth, which
    is why this one is a legitimate number.

What may not be reported from here: accuracy, miss rate, silent-failure rate,
a confidence interval, a p-value, or a claim about out-of-distribution
detection ability. The corpus is fixed, so a proportion over it is computed
rather than estimated and there is no sampling error to put an interval on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vision.pbr import contract as pbr            # noqa: E402

GENERATION = "v3gen01"

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_VARIANT = "fp16"
MODEL_LICENSE = "CreativeML Open RAIL++-M"
LICENSE_FILES = ("LICENSE.md", "README.md")

STEPS = 30
GUIDANCE = 7.5
WIDTH = 1024
HEIGHT = 1024
SEEDS = tuple(range(12))
NEGATIVE_PROMPT = ""

#: Eight prompts, one per V2 scene, aligned on *subject* only. The alignment
#: claim is deliberately weak: these describe the same kind of thing as the
#: matching scene -- a top-down view of that many bricks in those colours --
#: and nothing more. They are not paired images, the counts and colours are not
#: ground truth, and no accuracy follows from them.
PROMPTS = (
    ("scene_00", "top-down photograph of 3 loose LEGO bricks on a plain white "
                 "table, one red, one blue, one yellow, studs facing up"),
    ("scene_01", "top-down photograph of 4 loose LEGO bricks on a plain white "
                 "table, blue green white and yellow, studs facing up"),
    ("scene_02", "top-down photograph of 5 loose LEGO bricks on a plain white "
                 "table, black green orange white and yellow, studs facing up"),
    ("scene_03", "top-down photograph of 6 loose LEGO bricks on a plain white "
                 "table, black green grey orange red and white, studs facing up"),
    ("scene_04", "top-down photograph of 7 loose LEGO bricks on a plain white "
                 "table, mixed colours, studs facing up, evenly spaced"),
    ("scene_05", "top-down photograph of 8 loose LEGO bricks on a plain white "
                 "table, eight different colours, studs facing up, evenly spaced"),
    ("scene_06", "top-down photograph of 9 loose LEGO bricks on a plain white "
                 "table, mixed colours and sizes, studs facing up, evenly spaced"),
    ("scene_07", "top-down photograph of 10 loose LEGO bricks on a plain white "
                 "table, mixed colours and sizes, studs facing up, evenly spaced"),
)

FROZEN_DIR = ROOT / "data/phase_v2/frozen"
MAX_RECOGNISE_SECONDS = 120.0
PART_CONFIDENCE_FLOOR = 0.45


class Refused(RuntimeError):
    """A step that will not run, with the reason in the message."""


def model_record() -> dict:
    """The checkpoint's identity, and the licence bytes beside it.

    A model family name binds nothing -- gen02 of Phase 3C was voided for
    pinning a digest nobody could re-derive. So the licence file's own bytes
    are hashed and kept, not just its name.
    """
    from src.training.session import sha256_file

    licence_dir = ROOT / "artifacts/runtime/v3_model"
    licences = {}
    for name in LICENSE_FILES:
        path = licence_dir / name
        if not path.is_file():
            raise Refused(
                f"{path} is missing. The licence bytes are recorded before the "
                "weights are used, not after")
        licences[name] = {"bytes": path.stat().st_size,
                          "sha256": sha256_file(path)}

    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(MODEL_ID, allow_patterns=["*.json"],
                                  local_files_only=True))
    weights = {}
    for path in sorted(local.rglob("*.safetensors")):
        weights[str(path.relative_to(local))] = {
            "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if not weights:
        raise Refused(f"no safetensors under {local}")
    record = {
        "model_id": MODEL_ID,
        "variant": MODEL_VARIANT,
        "licence_name": MODEL_LICENSE,
        "licence_files": licences,
        "licence_source": f"https://huggingface.co/{MODEL_ID}/resolve/main/",
        "weight_files": weights,
        "n_weight_files": len(weights),
    }
    record["model_digest"] = pbr.scene_digest(record)
    return record


def source_manifest() -> dict:
    """This script and everything it imports from the project.

    ``import_closure`` returns what an entry point *reaches* and not the entry
    point, so this file -- which holds the prompts, the seeds and the response
    mode rule -- is added by name. Leaving it out is the defect that voided
    two V2 generations.
    """
    from src.training import pack
    from src.training.session import sha256_file

    entry = "scripts/63_v3_diffusion_probe.py"
    closure = pack.import_closure(root=ROOT, entry_points=(entry,))
    files = {rel: sha256_file(ROOT / rel)
             for rel in sorted(set(closure) | {entry})}
    return {"files": files, "digest": pbr.scene_digest(files)}


def plan() -> dict:
    """The frozen plan: prompts, seeds, settings. Written before any weight."""
    attempts = [{"attempt_id": f"{tag}/seed{seed:02d}", "scene_tag": tag,
                 "prompt": prompt, "seed": seed}
                for tag, prompt in PROMPTS for seed in SEEDS]
    body = {
        "kind": "brickagain.v3_plan",
        "generation": GENERATION,
        "model_id": MODEL_ID,
        "variant": MODEL_VARIANT,
        "steps": STEPS,
        "guidance_scale": pbr.decimal_str(GUIDANCE),
        "width": WIDTH,
        "height": HEIGHT,
        "negative_prompt": NEGATIVE_PROMPT,
        "scheduler": "pipeline default, recorded at run time",
        "n_prompts": len(PROMPTS),
        "n_seeds": len(SEEDS),
        "n_attempts": len(attempts),
        "attempts": attempts,
        "retry_rule": ("none. An attempt that fails is recorded as "
                       "generator_failed and is not retried or replaced"),
        "alignment_claim": ("subject alignment only. Not image pairing, not a "
                            "ground truth, and no accuracy follows"),
        "response_modes": ["reject", "low_confidence", "high_confidence",
                           "recognizer_crash", "recognizer_timeout"],
        "forbidden": ["accuracy", "miss rate", "silent-failure rate",
                      "confidence interval", "p-value", "power",
                      "out-of-distribution detection ability"],
        "part_confidence_floor": pbr.decimal_str(PART_CONFIDENCE_FLOOR),
        "source_manifest": source_manifest(),
    }
    body["plan_digest"] = pbr.scene_digest(body)
    return body


def mode_freeze(args) -> int:
    directory = FROZEN_DIR / GENERATION
    if directory.exists() and not args.force:
        raise Refused(f"{directory} exists; a frozen plan is write-once")
    body = plan()
    body["model"] = model_record()
    body["authorization_digest"] = pbr.scene_digest(body)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plan.json").write_text(
        json.dumps(body, indent=1, sort_keys=True))
    print(json.dumps({"mode": "freeze", "generation": GENERATION,
                      "n_attempts": body["n_attempts"],
                      "plan_digest": body["plan_digest"],
                      "model_digest": body["model"]["model_digest"],
                      "licence": body["model"]["licence_name"],
                      "authorization_digest": body["authorization_digest"]},
                     indent=1))
    return 0


def load_plan() -> dict:
    path = FROZEN_DIR / GENERATION / "plan.json"
    if not path.is_file():
        raise Refused(f"{path} does not exist; freeze first")
    body = json.loads(path.read_text())
    recomputed = pbr.scene_digest(
        {k: v for k, v in body.items() if k != "authorization_digest"})
    if recomputed != body["authorization_digest"]:
        raise Refused(f"{path} does not recompute")
    return body


def response_mode(analysis, seconds: float) -> str:
    """What the recogniser did, without reference to what was in the image.

    Mutually exclusive and exhaustive over the produced images, and every
    branch is observable: a refusal is an empty result, unsure is a result
    whose every label is the abstention, and confident is anything else.
    """
    if seconds > MAX_RECOGNISE_SECONDS:
        return "recognizer_timeout"
    if not analysis.items:
        return "reject"
    if all((item.predicted_confidence or 0.0) < PART_CONFIDENCE_FLOOR
           for item in analysis.items):
        return "low_confidence"
    return "high_confidence"


def mode_run(args) -> int:
    frozen = load_plan()
    live = model_record()
    if live["model_digest"] != frozen["model"]["model_digest"]:
        raise Refused("the checkpoint does not match the frozen plan")
    live_sources = source_manifest()
    if live_sources["digest"] != frozen["source_manifest"]["digest"]:
        raise Refused(
            f"the source manifest is {live_sources['digest']} and the plan "
            f"binds {frozen['source_manifest']['digest']}. An "
            "execution-relevant change needs a new generation")
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise Refused(f"{out_dir} is not empty; pass --force")
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, variant=MODEL_VARIANT,
        use_safetensors=True)
    pipe = pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)
    scheduler = type(pipe.scheduler).__name__

    from src.training.session import sha256_file

    outcomes = []
    for index, attempt in enumerate(frozen["attempts"], 1):
        path = out_dir / "images" / f"{attempt['attempt_id'].replace('/', '__')}.png"
        record = {"attempt_id": attempt["attempt_id"],
                  "scene_tag": attempt["scene_tag"], "seed": attempt["seed"]}
        started = time.time()
        try:
            generator = torch.Generator(device="cpu").manual_seed(attempt["seed"])
            image = pipe(prompt=attempt["prompt"],
                         negative_prompt=NEGATIVE_PROMPT or None,
                         num_inference_steps=STEPS,
                         guidance_scale=GUIDANCE, width=WIDTH, height=HEIGHT,
                         generator=generator).images[0]
            image.save(path, format="PNG")
            record.update({"outcome": "produced", "png_path": str(path),
                           "png_sha256": sha256_file(path),
                           "png_bytes": path.stat().st_size})
        except Exception as exc:                  # noqa: BLE001 - recorded
            record.update({"outcome": "generator_failed",
                           "error": f"{type(exc).__name__}: {exc}"[:500]})
        record["seconds"] = round(time.time() - started, 3)
        outcomes.append(record)
        print(f"[{index}/{len(frozen['attempts'])}] {attempt['attempt_id']} "
              f"{record['outcome']} {record['seconds']}s", flush=True)

    body = {
        "kind": "brickagain.v3_generation_outcome",
        "generation": GENERATION,
        "authorization_digest": frozen["authorization_digest"],
        "scheduler_resolved": scheduler,
        "device": "mps",
        "torch_version": torch.__version__,
        "denominator": len(frozen["attempts"]),
        "counts": {mode: sum(1 for r in outcomes if r["outcome"] == mode)
                   for mode in ("produced", "generator_failed",
                                "generator_timeout")},
        "attempts": outcomes,
        "note": ("Every attempt is here. A failed attempt is not retried and "
                 "not replaced; the denominator stays 96"),
    }
    body["outcome_digest"] = pbr.scene_digest(
        {k: v for k, v in body.items() if k != "attempts"})
    (out_dir / "generation_outcome.json").write_text(
        json.dumps(body, indent=1, sort_keys=True))
    print(json.dumps({"mode": "run", "counts": body["counts"],
                      "outcome_digest": body["outcome_digest"]}, indent=1))
    return 0


def mode_recognise(args) -> int:
    frozen = load_plan()
    out_dir = Path(args.out_dir)
    outcome = json.loads((out_dir / "generation_outcome.json").read_text())
    if outcome["authorization_digest"] != frozen["authorization_digest"]:
        raise Refused("the generation was not produced under this plan")

    from src.training.session import sha256_file
    from src.ui.full import PHOTO_MULTI, RECOGNISE_CV, analyse_photo

    rows = []
    for attempt in outcome["attempts"]:
        if attempt["outcome"] != "produced":
            continue
        path = Path(attempt["png_path"])
        stored = sha256_file(path)
        if stored != attempt["png_sha256"]:
            raise Refused(f"{path} has changed since it was filed")
        started = time.time()
        try:
            analysis = analyse_photo(path.read_bytes(), mode=PHOTO_MULTI,
                                     method=RECOGNISE_CV)
        except Exception as exc:                  # noqa: BLE001 - recorded
            rows.append({"attempt_id": attempt["attempt_id"],
                         "scene_tag": attempt["scene_tag"],
                         "mode": "recognizer_crash",
                         "error": f"{type(exc).__name__}: {exc}"[:500],
                         "seconds": round(time.time() - started, 3)})
            continue
        seconds = time.time() - started
        rows.append({
            "attempt_id": attempt["attempt_id"],
            "scene_tag": attempt["scene_tag"],
            "mode": response_mode(analysis, seconds),
            "seconds": round(seconds, 3),
            "n_items": len(analysis.items),
            "n_colour_problems": len(analysis.colour_problems),
            "mean_part_confidence": round(sum(
                (i.predicted_confidence or 0.0) for i in analysis.items)
                / len(analysis.items), 4) if analysis.items else None,
            "proposed_inventory_size": sum(
                1 for i in analysis.items
                if i.predicted_part and i.predicted_part != "unknown"),
        })

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
        "outcome_digest": outcome["outcome_digest"],
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
    (out_dir / "response_mode_distribution.json").write_text(
        json.dumps(body, indent=1, sort_keys=True))
    print(json.dumps({"mode": "recognise",
                      "denominator_n_produced": len(rows),
                      "counts": body["counts"],
                      "distribution_digest": body["distribution_digest"]},
                     indent=1))
    return 0


RESULTS_DIR = ROOT / "data/phase_v2/results"
SAMPLE_PER_PROMPT = 1


def mode_archive(args) -> int:
    """Copy the ledgers into a tracked tree, with a sample of the images.

    Ninety-six 1024-pixel diffusion images are too large to track, and unlike
    V2's corpus they are not what a number is derived from -- the response mode
    ledger is derived from what the recogniser *did*, and that is re-derivable
    from any one image. So every image's digest is recorded, one image per
    prompt travels so a reader can see what the corpus looks like, and the
    index says plainly that the rest are pinned by digest and not archived.
    """
    frozen = load_plan()
    out_dir = Path(args.out_dir)
    outcome = json.loads((out_dir / "generation_outcome.json").read_text())
    distribution = json.loads(
        (out_dir / "response_mode_distribution.json").read_text())
    target = RESULTS_DIR / GENERATION
    if target.exists() and not args.force:
        raise Refused(f"{target} exists; the archive is write-once")
    (target / "images_sample").mkdir(parents=True, exist_ok=True)

    import shutil

    from src.training.session import sha256_file

    files = {}

    def take(source: Path, relative: str):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files[relative] = {"bytes": destination.stat().st_size,
                           "sha256": sha256_file(destination)}

    taken_per_prompt = {}
    sampled = []
    for attempt in outcome["attempts"]:
        if attempt["outcome"] != "produced":
            continue
        tag = attempt["scene_tag"]
        if taken_per_prompt.get(tag, 0) >= SAMPLE_PER_PROMPT:
            continue
        taken_per_prompt[tag] = taken_per_prompt.get(tag, 0) + 1
        name = Path(attempt["png_path"]).name
        take(Path(attempt["png_path"]), f"images_sample/{name}")
        sampled.append(attempt["attempt_id"])

    for name in ("generation_outcome.json", "response_mode_distribution.json"):
        take(out_dir / name, name)
    take(FROZEN_DIR / GENERATION / "plan.json", "plan.json")

    index = {
        "kind": "brickagain.v3_result_index",
        "generation": GENERATION,
        "authorization_digest": frozen["authorization_digest"],
        "outcome_digest": outcome["outcome_digest"],
        "distribution_digest": distribution["distribution_digest"],
        "n_files": len(files),
        "files": files,
        "n_attempts": outcome["denominator"],
        "n_produced": outcome["counts"]["produced"],
        "images_sampled": sampled,
        "image_digests": {a["attempt_id"]: a.get("png_sha256")
                          for a in outcome["attempts"]},
        "run_out_dir_on_the_mac": str(out_dir.relative_to(ROOT)),
        "archive_claim": (
            f"{len(sampled)} of {outcome['counts']['produced']} images travel, "
            "one per prompt. Every image's SHA-256 is recorded. The response "
            "mode ledger is a record of what the recogniser did and does not "
            "need the pixels to be re-read; a claim that all images were "
            "re-examined would not be supported by this archive"),
    }
    index["index_digest"] = pbr.scene_digest(
        {k: v for k, v in index.items() if k != "index_digest"})
    (target / "index.json").write_bytes(
        json.dumps(index, indent=1, sort_keys=True).encode("utf-8"))
    print(json.dumps({"mode": "archive", "generation": GENERATION,
                      "target": str(target.relative_to(ROOT)),
                      "n_files": len(files),
                      "megabytes": round(sum(v["bytes"] for v in files.values())
                                         / 1e6, 1),
                      "index_digest": index["index_digest"]}, indent=1))
    return 0


MODES = {"freeze": mode_freeze, "run": mode_run,
         "recognise": mode_recognise, "archive": mode_archive}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--out-dir", default=str(ROOT / "runs/v3" / GENERATION))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        return MODES[args.mode](args)
    except Refused as exc:
        print(json.dumps({"refused": str(exc)}, indent=1), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
