"""The fine-tuning loop, and the record it leaves behind.

Two things this module refuses to do, and they are the reason it exists rather
than the training being a few lines in a script:

**It never reads the test split.**  :func:`load_split_items` takes the split
name it is allowed to read, and :func:`fit` is only ever given ``train`` and
``validation``.  Selection between epochs is on validation loss and validation
accuracy; the test images are not opened, so there is no path by which a
hyper-parameter could be chosen with knowledge of them.

**It records everything a re-run needs.**  Per-epoch loss and accuracy on both
sides, the selected epoch and why, the seed, the configuration, the
augmentation, the data manifest digest, the split manifest digest, the code
digest, the dependency versions and the device.  A checkpoint without those is
a checkpoint whose result cannot be checked, and :func:`~src.vision.model.save`
refuses to write one.

The real photographs are outnumbered by renders roughly nine to one in the
public archive, which would let a model score well on the population that does
not matter.  ``real_weight`` addresses that by sampling: a real photograph is
drawn more often than a render, by a factor stated in the configuration and
recorded in the manifest.  It is a training-time choice only -- the reports
score real and synthetic separately and never pool them.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.vision import datasets
from src.vision.classes import CLASS_ORDER, label_index
from src.vision.model import (Augmentation, DeviceTuning, ModelError,
                              TrainConfig, apply_tuning, augment,
                              autocast_context, build_model, device_report,
                              forward_logits, freeze_parameters,
                              prepare_model, resolve_device, save, softmax,
                              suggested_batch_size, tuning_for)
from src.vision.preprocess import ImageError, model_tensor, read_image
from src.vision.split import TEST, TRAIN, VALIDATION, VisionSplit

#: How much more often a real photograph is drawn than a render.  A factor,
#: not a hard balance: at 1.0 the renders dominate and the model is fitted to
#: the population that is not the deliverable; at a very high value the real
#: photographs are seen so often that the renders stop contributing at all.
DEFAULT_REAL_WEIGHT = 6.0


class TrainError(RuntimeError):
    """Training cannot start, or cannot be trusted to have been correct."""


@dataclass(frozen=True)
class Item:
    """One training or validation image: where it is and what it is."""

    path: Path
    member: str
    part: str
    population: str

    @property
    def target(self) -> int:
        return label_index(self.part)


def readable_problem(path: Path) -> str | None:
    """Why this image cannot be used, or ``None`` when it can.

    Header only: PIL opens lazily, so this reads a few dozen bytes per file
    and a pass over the whole archive costs seconds. Checked once up front
    rather than per batch, so every batch stays full and the exclusions are a
    number in the manifest instead of a surprise in epoch three.
    """
    from src.vision.preprocess import MIN_IMAGE_SIDE

    try:
        from PIL import Image

        with Image.open(path) as handle:
            width, height = handle.size
            fmt = (handle.format or "").upper()
    except Exception as exc:                    # noqa: BLE001 - see message
        return f"cannot be opened: {exc}"
    from src.vision.preprocess import ALLOWED_FORMATS

    if fmt not in ALLOWED_FORMATS:
        return f"format {fmt or 'unknown'} is not one of {list(ALLOWED_FORMATS)}"
    if min(width, height) < MIN_IMAGE_SIDE:
        return (f"{width}x{height} is below the {MIN_IMAGE_SIDE}px minimum "
                "side this pipeline works at")
    return None


def load_split_items(data_manifest: dict, split: VisionSplit, *,
                     allowed_splits, raw_root,
                     excluded: list | None = None) -> dict[str, list[Item]]:
    """Items for the named splits only, minus any the pipeline cannot read.

    ``allowed_splits`` has no default.  A function that could be called with
    no argument and return everything is a function that will eventually be
    handed the test split by accident.

    An image the pipeline cannot use is excluded and appended to ``excluded``
    rather than aborting the run or being dropped quietly.  The public archive
    contains two renders of thirteen and fourteen pixels a side, which no
    amount of resizing makes into evidence; failing an eight-epoch fit over
    two files out of twenty-four thousand is the wrong failure, and so is not
    saying they were left out.
    """
    allowed = tuple(allowed_splits)
    if not allowed:
        raise TrainError("name at least one split to load")
    for name in allowed:
        if name not in (TRAIN, VALIDATION, TEST):
            raise TrainError(f"{name!r} is not a split")
    records = datasets.records_from_manifest(data_manifest)
    root = Path(raw_root) / data_manifest["source"]["key"]
    out: dict[str, list[Item]] = {name: [] for name in allowed}
    for record in records:
        if not record.part:
            continue
        try:
            side = split.split_of_item(record.member)
        except Exception as exc:                # noqa: BLE001 - see message
            raise TrainError(
                f"{record.member} is not in the frozen split: {exc}") from exc
        if side not in out:
            continue
        path = root / record.member
        problem = readable_problem(path)
        if problem is not None:
            if excluded is not None:
                excluded.append({"member": record.member, "split": side,
                                 "part": record.part,
                                 "population": record.population,
                                 "reason": problem})
            continue
        out[side].append(Item(path=path, member=record.member,
                              part=record.part, population=record.population))
    for name, items in out.items():
        if not items:
            raise TrainError(f"the {name} split has no labelled items")
        items.sort(key=lambda item: item.member)
    return out


def sample_weights(items, *, real_weight: float = DEFAULT_REAL_WEIGHT
                   ) -> np.ndarray:
    """Per-item sampling weight, larger for real photographs."""
    if not math.isfinite(real_weight) or real_weight <= 0:
        raise TrainError("real_weight must be a finite positive number")
    weights = np.array(
        [real_weight if item.population == datasets.POPULATION_REAL else 1.0
         for item in items], dtype=np.float64)
    return weights / weights.sum()


def _load_one(item: Item, index: int, *, spec: Augmentation, seed: int,
              epoch: int):
    try:
        image = read_image(item.path)
    except ImageError as exc:
        raise TrainError(f"{item.member} could not be read: {exc}") from exc
    pixels = augment(image.rgb, spec, seed=seed, epoch=epoch, index=index)
    return model_tensor(pixels), item.target


def _load_batch(items, indices, *, spec: Augmentation, seed: int, epoch: int,
                pool=None):
    """Decode and augment one batch, in parallel when a pool is given.

    Parallel because this is the bottleneck rather than the arithmetic: a
    batch of 128 JPEGs is a hundred-odd megabytes of decode and resize on a
    six-core CPU, and a card that finishes a ResNet-18 step in a few
    milliseconds spends the rest of its time idle waiting for it. Pillow's
    decode and NumPy's resample both release the GIL, so threads overlap for
    real. Order is preserved, so the batch a given seed produces does not
    depend on which thread finished first.
    """
    jobs = list(indices)
    load = lambda index: _load_one(items[index], index, spec=spec, seed=seed,
                                   epoch=epoch)
    results = list(pool.map(load, jobs)) if pool is not None else [
        load(index) for index in jobs]
    tensors = [tensor for tensor, _target in results]
    targets = [target for _tensor, target in results]
    return np.stack(tensors), np.array(targets, dtype=np.int64)


def evaluate(model, items, *, device: str, batch_size: int = 32,
             tuning: DeviceTuning | None = None, pool=None) -> dict:
    """Loss and accuracy over a split, with the populations kept apart.

    No augmentation, no shuffling: the same items in the same order every
    time, so two epochs' validation numbers are comparable to each other.
    """
    import torch

    settings = tuning or tuning_for(device)
    total_loss = 0.0
    seen = 0
    correct = 0
    by_population: dict[str, dict[str, int]] = {}
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    order = list(range(len(items)))
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            batch, targets = _load_batch(
                items, chunk, spec=Augmentation(enabled=False), seed=0,
                epoch=0, pool=pool)
            with autocast_context(settings, device):
                logits = forward_logits(
                    model, batch, device,
                    channels_last=settings.channels_last)
            labels = torch.from_numpy(targets).to(device)
            # The loss is read in float32 whatever the forward ran in: a
            # validation loss compared across epochs at bfloat16 precision has
            # about three decimal digits, and epoch selection turns on it.
            total_loss += float(loss_fn(logits.float(), labels).item())
            predicted = logits.argmax(dim=-1).to("cpu").numpy()
            for index, guess in zip(chunk, predicted.tolist()):
                bucket = by_population.setdefault(
                    items[index].population, {"n": 0, "correct": 0})
                bucket["n"] += 1
                hit = int(guess) == items[index].target
                bucket["correct"] += int(hit)
                correct += int(hit)
            seen += len(chunk)
    out = {"n": seen, "loss": total_loss / seen if seen else float("nan"),
           "accuracy": correct / seen if seen else float("nan")}
    out["by_population"] = {
        name: {"n": bucket["n"],
               "accuracy": bucket["correct"] / bucket["n"]
               if bucket["n"] else float("nan")}
        for name, bucket in sorted(by_population.items())}
    return out


def fit(train_items, validation_items, *, config: TrainConfig,
        device: str | None = None, steps_per_epoch: int | None = None,
        real_weight: float = DEFAULT_REAL_WEIGHT,
        local_files_only: bool = True, cache_dir=None,
        tuning: DeviceTuning | None = None,
        progress=None) -> tuple[object, list[dict], int, str, str, dict]:
    """Fine-tune the head, selecting the best epoch on validation.

    Returns ``(model, epoch_log, selected_epoch, criterion, device)``.  The
    returned model carries the selected epoch's parameters: the best state is
    kept in memory and restored at the end, so the checkpoint written is the
    epoch that was chosen rather than the last one that ran.
    """
    import torch

    resolved = resolve_device(device)
    settings = tuning if tuning is not None else tuning_for(resolved)
    applied = apply_tuning(settings, resolved)
    torch.manual_seed(config.seed)
    if resolved == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model = prepare_model(
        build_model(local_files_only=local_files_only, cache_dir=cache_dir),
        resolved, settings)
    frozen = freeze_parameters(model, config)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing)

    weights = sample_weights(train_items, real_weight=real_weight)
    per_epoch = steps_per_epoch or max(
        1, len(train_items) // config.batch_size)
    best_state = None
    best_loss = float("inf")
    selected = 0
    log: list[dict] = []

    pool = ThreadPoolExecutor(max_workers=settings.loader_workers)
    for epoch in range(1, config.epochs + 1):
        model.train()
        rng = np.random.default_rng([config.seed, epoch])
        running = 0.0
        counted = 0
        started = time.monotonic()
        for step in range(per_epoch):
            indices = rng.choice(len(train_items), size=config.batch_size,
                                 replace=True, p=weights)
            batch, targets = _load_batch(
                train_items, indices.tolist(), spec=config.augmentation,
                seed=config.seed, epoch=epoch, pool=pool)
            with autocast_context(settings, resolved):
                logits = forward_logits(
                    model, batch, resolved,
                    channels_last=settings.channels_last)
                loss = loss_fn(logits,
                               torch.from_numpy(targets).to(resolved))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            running += float(loss.item()) * len(indices)
            counted += len(indices)
            if progress is not None and (step + 1) % 20 == 0:
                progress(f"epoch {epoch} step {step + 1}/{per_epoch} "
                         f"loss {running / counted:.4f}")
        model.eval()
        validation = evaluate(model, validation_items, device=resolved,
                              batch_size=config.batch_size, tuning=settings,
                              pool=pool)
        entry = {
            "epoch": epoch,
            "train_loss": running / counted if counted else float("nan"),
            "train_steps": per_epoch,
            "train_images_drawn": counted,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_by_population": validation["by_population"],
            "seconds": round(time.monotonic() - started, 3),
        }
        log.append(entry)
        if progress is not None:
            progress(f"epoch {epoch}: train {entry['train_loss']:.4f}  "
                     f"val {entry['validation_loss']:.4f}  "
                     f"val acc {entry['validation_accuracy']:.4f}")
        # Strictly lower, so an exactly equal epoch keeps the earlier one --
        # the same tie-break rule the generation track's model selection uses.
        if entry["validation_loss"] < best_loss:
            best_loss = entry["validation_loss"]
            selected = epoch
            best_state = {name: tensor.detach().to("cpu").clone()
                          for name, tensor in model.state_dict().items()}

    pool.shutdown(wait=True)
    if best_state is None or selected == 0:
        raise TrainError("no epoch produced a usable validation loss")
    model.load_state_dict(best_state, strict=True)
    criterion = ("lowest mean validation cross-entropy over the frozen "
                 "validation split; an exactly equal epoch keeps the earlier "
                 "one. The test split was not read during fitting")
    log.append({"frozen_parameters": frozen,
                "device_report": device_report(resolved),
                "applied_tuning": applied})
    return model, log, selected, criterion, resolved, applied


def run(*, data_manifest_path, split_path, out_dir, config: TrainConfig,
        raw_root, device: str | None = None,
        steps_per_epoch: int | None = None,
        real_weight: float = DEFAULT_REAL_WEIGHT,
        expected_data_digest: str | None = None,
        expected_split_digest: str | None = None,
        local_files_only: bool = True, cache_dir=None,
        deterministic: bool = False, progress=None) -> dict:
    """One whole fitting run, from frozen manifests to a written checkpoint."""
    manifest = datasets.read_manifest(data_manifest_path,
                                      expected_digest=expected_data_digest)
    data_digest = datasets.manifest_digest(manifest)
    split = VisionSplit.load(split_path, expected_digest=expected_split_digest)
    split.check_no_leakage()
    split_digest = split.digest()

    excluded: list[dict] = []
    items = load_split_items(manifest, split,
                             allowed_splits=(TRAIN, VALIDATION),
                             raw_root=raw_root, excluded=excluded)
    settings = tuning_for(resolve_device(device), deterministic=deterministic)
    model, log, selected, criterion, resolved, applied = fit(
        items[TRAIN], items[VALIDATION], config=config, device=device,
        steps_per_epoch=steps_per_epoch, real_weight=real_weight,
        local_files_only=local_files_only, cache_dir=cache_dir,
        tuning=settings, progress=progress)

    written = save(
        model, out_dir, config=config, epoch_log=log,
        selected_epoch=selected, selection_criterion=criterion,
        data_manifest_sha256=data_digest, device=resolved,
        split_manifest_sha256=split_digest,
        tuning=settings, applied_tuning=applied,
        notes=("fitted on the public single-brick archive: real photographs "
               f"weighted {real_weight}x against renders during sampling. "
               "Real and synthetic are scored separately and never pooled. "
               "The test split was not opened by this run. "
               f"{len(excluded)} image(s) were excluded as unreadable by this "
               "pipeline; they are listed in run_summary.json"))
    summary = {
        "checkpoint": str(Path(out_dir)),
        "device": resolved,
        "device_report": device_report(resolved),
        "tuning": {**settings.as_dict(), "applied": applied},
        "batch_size": config.batch_size,
        "selected_epoch": selected,
        "epochs": config.epochs,
        "train_items": len(items[TRAIN]),
        "validation_items": len(items[VALIDATION]),
        "excluded_unreadable": excluded,
        "excluded_unreadable_count": len(excluded),
        "real_weight": real_weight,
        "data_manifest_sha256": data_digest,
        "split_manifest_sha256": split_digest,
        "weights_sha256": written["weights"]["sha256"],
        "class_order": list(CLASS_ORDER),
        "epoch_log": [entry for entry in log if "epoch" in entry],
    }
    (Path(out_dir) / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8")
    return summary
