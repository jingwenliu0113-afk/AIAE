"""The learned eight-class classifier: an ImageNet backbone plus a new head.

Fitting is done on the CUDA node; inference happens on the Mac.  That split is
the reason almost everything here is about making a checkpoint self-describing.
A set of weights whose preprocessing, class order, seed, data manifest and
dependency versions are not written down beside it cannot be re-run, and a
result from it cannot be checked -- so :func:`save` refuses to write a
checkpoint without all of them, and :func:`load` refuses to read one whose
weight digest or class order has moved.

``torch`` and ``transformers`` are imported inside the functions that need
them.  The CV baseline, the metrics, the split and the whole delivery path have
to keep importing and testing on a machine with neither installed, and a
module-level import here would take that away from them.

Nothing in this module reaches the generation track.  It does not import
:mod:`src.generation.brickgpt`, :mod:`src.training.lora` or anything that
loads ``final_H2``; a brick-image classifier and a text-to-structure decoder
are different models and this file must not be able to blur them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.vision.classes import CLASS_ORDER, N_CLASSES, check_contract
from src.vision.model_ids import CLASSIFIER_BACKBONE, VISION_MANIFEST
from src.vision.preprocess import (CROP_SIZE, IMAGE_MEAN, IMAGE_STD,
                                   RESAMPLE_NOTE, RESIZE_SHORT_SIDE,
                                   check_processor_config, model_tensor)
from src.vision.schema import METHOD_LEARNED, Prediction, from_scores

#: The file the fitted parameters go in.
WEIGHTS_FILE = "vision_head.pt"

DEVICES = ("cpu", "mps", "cuda")


class ModelError(RuntimeError):
    """The model, its checkpoint or its environment is not usable."""


# ---------------------------------------------------------------------------
# Making the accelerator earn its keep, without hard-coding one card
# ---------------------------------------------------------------------------

#: VRAM thresholds, in GiB, and the batch size each one gets.  A ResNet-18 at
#: 224 pixels under bfloat16 autocast holds a batch of 128 in roughly four
#: gigabytes of activations, so a sixteen-gigabyte card has ample headroom and
#: a smaller one steps down rather than failing at the first backward pass.
BATCH_BY_VRAM_GIB: tuple[tuple[float, int], ...] = (
    (14.0, 128),
    (7.0, 64),
    (0.0, 32),
)

#: Batch size on anything that is not a discrete accelerator.  Apple's unified
#: memory is large but its bandwidth is shared with everything else, and a CPU
#: run is a smoke rather than a fit, so neither wants a big batch.
BATCH_DEFAULT = 32

#: Image loading is JPEG decode plus a resize, both of which release the GIL
#: inside Pillow and NumPy, so threads genuinely overlap.  Held below the
#: thread count so the training process itself still gets a core.
def suggested_loader_workers() -> int:
    import os

    return max(2, min(8, (os.cpu_count() or 4) - 2))


@dataclass(frozen=True)
class DeviceTuning:
    """What to switch on for a given accelerator, and what that costs.

    Every field is recorded in the checkpoint manifest.  That matters more than
    usual here: bfloat16 autocast, TF32 matmuls and cuDNN autotuning all change
    the arithmetic, so a run is reproducible on the same device with the same
    settings and is *not* bit-identical to a run with different ones.  Writing
    them down is what makes the difference visible instead of mysterious.

    ``deterministic=True`` turns off the three that trade exactness for speed,
    for when a bit-identical re-run matters more than throughput.
    """

    autocast_dtype: str | None = None
    channels_last: bool = False
    tf32: bool = False
    cudnn_benchmark: bool = False
    loader_workers: int = 2
    deterministic: bool = False

    def as_dict(self) -> dict:
        return {
            "autocast_dtype": self.autocast_dtype,
            "channels_last": self.channels_last,
            "tf32": self.tf32,
            "cudnn_benchmark": self.cudnn_benchmark,
            "loader_workers": self.loader_workers,
            "deterministic": self.deterministic,
            "note": (
                "bfloat16 autocast, TF32 matmuls and cuDNN autotuning change "
                "the arithmetic. A run reproduces on the same device with the "
                "same settings; it is not bit-identical to a run with "
                "different ones, which is why these are recorded rather than "
                "assumed"),
        }


def tuning_for(device: str, *, deterministic: bool = False) -> DeviceTuning:
    """Pick the settings from what the device actually reports.

    Queried rather than assumed.  The node this project fits on is a Blackwell
    card with fifth-generation tensor cores and native bfloat16, but the code
    asks ``torch`` whether bfloat16 is supported instead of naming the card, so
    the same path is correct on the next one.
    """
    if device not in DEVICES:
        raise ModelError(f"device={device!r} is not one of {list(DEVICES)}")
    workers = suggested_loader_workers()
    if deterministic or device != "cuda":
        # No autocast off CUDA: MPS bfloat16 is not the same kernel set, and a
        # CPU run is a smoke where speed is not the point.
        return DeviceTuning(loader_workers=workers,
                            deterministic=deterministic)
    import torch

    return DeviceTuning(
        autocast_dtype=("bfloat16" if torch.cuda.is_bf16_supported()
                        else "float16"),
        channels_last=True,
        tf32=True,
        cudnn_benchmark=True,
        loader_workers=workers,
        deterministic=False,
    )


def apply_tuning(tuning: DeviceTuning, device: str) -> dict:
    """Set the torch flags and report what was actually set.

    Reporting what was set rather than what was asked for: a flag that a build
    of torch ignores would otherwise appear in the manifest as though it had
    taken effect.
    """
    import torch

    applied = {"device": device}
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tuning.tf32)
        torch.backends.cudnn.allow_tf32 = bool(tuning.tf32)
        torch.backends.cudnn.benchmark = bool(tuning.cudnn_benchmark)
        applied["matmul_allow_tf32"] = bool(
            torch.backends.cuda.matmul.allow_tf32)
        applied["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        applied["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    applied["autocast_dtype"] = tuning.autocast_dtype
    applied["channels_last"] = bool(tuning.channels_last)
    applied["loader_workers"] = int(tuning.loader_workers)
    return applied


def autocast_context(tuning: DeviceTuning, device: str):
    """An autocast context, or a no-op one when autocast is off.

    A context manager either way, so the training loop has one code path
    instead of a conditional around its inner loop.
    """
    import contextlib

    import torch

    if not tuning.autocast_dtype or device != "cuda":
        return contextlib.nullcontext()
    dtype = getattr(torch, tuning.autocast_dtype, None)
    if dtype is None:
        raise ModelError(
            f"autocast_dtype={tuning.autocast_dtype!r} is not a torch dtype")
    return torch.autocast(device_type="cuda", dtype=dtype)


def device_report(device: str) -> dict:
    """What the accelerator says about itself, for the manifest.

    A result produced on an unnamed device is a result nobody can repeat, and
    the numbers that matter -- memory, tensor-core generation, whether bfloat16
    is native -- are exactly the ones that decide the settings above.
    """
    import torch

    out = {"device": device, "torch": torch.__version__}
    if device == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        out.update({
            "name": properties.name,
            "capability": f"{properties.major}.{properties.minor}",
            "total_memory_mib": properties.total_memory // (1024 * 1024),
            "multi_processor_count": properties.multi_processor_count,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "cudnn_version": torch.backends.cudnn.version(),
            "arch_list": list(torch.cuda.get_arch_list()),
        })
    elif device == "mps":
        out["name"] = "Apple Metal Performance Shaders"
    else:
        import platform

        out["name"] = platform.processor() or "cpu"
    return out


def suggested_batch_size(device: str) -> int:
    """A batch size the device can actually hold, from its reported memory."""
    if device != "cuda":
        return BATCH_DEFAULT
    import torch

    if not torch.cuda.is_available():
        return BATCH_DEFAULT
    gigabytes = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    for threshold, batch in BATCH_BY_VRAM_GIB:
        if gigabytes >= threshold:
            return batch
    return BATCH_DEFAULT


# ---------------------------------------------------------------------------
# Deterministic augmentation, in NumPy, so the CUDA node and the Mac agree
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Augmentation:
    """What may be done to a training image, and by how much.

    Every parameter is drawn from a generator seeded by ``(seed, epoch,
    index)``, so the exact sequence of augmentations is a function of the
    configuration and can be reproduced without storing it.  Rotation is
    limited to multiples of ninety degrees plus a small jitter: a brick's class
    is invariant to which way round it lies, and a big free rotation would
    teach the head that a ``1x4`` seen end-on is a ``1x1``.
    """

    horizontal_flip: bool = True
    vertical_flip: bool = True
    quarter_turns: bool = True
    brightness: float = 0.18
    contrast: float = 0.14
    enabled: bool = True

    def as_dict(self) -> dict:
        return {"horizontal_flip": self.horizontal_flip,
                "vertical_flip": self.vertical_flip,
                "quarter_turns": self.quarter_turns,
                "brightness": self.brightness, "contrast": self.contrast,
                "enabled": self.enabled}


def augment(rgb: np.ndarray, spec: Augmentation, *, seed: int, epoch: int,
            index: int) -> np.ndarray:
    """Apply the augmentation for one item in one epoch, reproducibly."""
    if not spec.enabled:
        return rgb
    rng = np.random.default_rng(
        [int(seed) & 0xFFFFFFFF, int(epoch) & 0xFFFFFFFF,
         int(index) & 0xFFFFFFFF])
    out = np.asarray(rgb)
    if spec.quarter_turns:
        out = np.rot90(out, int(rng.integers(0, 4)))
    if spec.horizontal_flip and bool(rng.integers(0, 2)):
        out = out[:, ::-1]
    if spec.vertical_flip and bool(rng.integers(0, 2)):
        out = out[::-1]
    out = out.astype(np.float32)
    if spec.brightness:
        shift = float(rng.uniform(-spec.brightness, spec.brightness)) * 255.0
        out = out + shift
    if spec.contrast:
        factor = 1.0 + float(rng.uniform(-spec.contrast, spec.contrast))
        mean = float(out.mean())
        out = (out - mean) * factor + mean
    return np.clip(out, 0.0, 255.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """Everything that decides what a fitted head is.

    Written into the manifest verbatim.  ``freeze_backbone`` is the one choice
    worth explaining: with a few hundred real photographs, fitting eleven
    million backbone parameters overfits, so the default fits the head and the
    last stage only.  It is a configuration value rather than a hard rule so
    the alternative can be run and compared on validation.
    """

    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    freeze_backbone: bool = True
    trainable_stages: int = 1
    label_smoothing: float = 0.05
    augmentation: Augmentation = field(default_factory=Augmentation)

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ModelError("epochs must be at least one")
        if self.batch_size < 1:
            raise ModelError("batch_size must be at least one")
        if not 0 < self.learning_rate < 1:
            raise ModelError("learning_rate must be between zero and one")
        if self.weight_decay < 0:
            raise ModelError("weight_decay may not be negative")
        if self.seed < 0:
            raise ModelError("seed must not be negative")
        if not 0 <= self.label_smoothing < 0.5:
            raise ModelError("label_smoothing must be in [0, 0.5)")
        if self.trainable_stages < 0:
            raise ModelError("trainable_stages may not be negative")

    def as_dict(self) -> dict:
        return {"epochs": self.epochs, "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay, "seed": self.seed,
                "freeze_backbone": self.freeze_backbone,
                "trainable_stages": self.trainable_stages,
                "label_smoothing": self.label_smoothing,
                "augmentation": self.augmentation.as_dict()}


def preprocessing_record() -> dict:
    """The pinned preprocessing, for the manifest.

    A checkpoint whose preprocessing is not recorded is a checkpoint that can
    be served with different inputs from the ones it was fitted on, and no
    file would disagree.
    """
    return {"resize_short_side": RESIZE_SHORT_SIDE, "crop": CROP_SIZE,
            "image_mean": list(IMAGE_MEAN), "image_std": list(IMAGE_STD),
            "resample": RESAMPLE_NOTE,
            "implemented_by": "src.vision.preprocess.model_tensor"}


def dependency_versions() -> dict:
    """The versions a result was produced under, read at run time."""
    out = {}
    for name in ("torch", "transformers", "numpy", "PIL"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            out[name] = "absent"
    return out


def resolve_device(requested: str | None = None) -> str:
    """Pick a device, refusing a name this project does not support."""
    import torch

    if requested is not None:
        if requested not in DEVICES:
            raise ModelError(
                f"device={requested!r} is not one of {list(DEVICES)}")
        if requested == "cuda" and not torch.cuda.is_available():
            raise ModelError("cuda was asked for and is not available here")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ModelError("mps was asked for and is not available here")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Building and running the network
# ---------------------------------------------------------------------------

def build_model(*, local_files_only: bool = True, cache_dir=None):
    """Load the pinned backbone and attach a fresh eight-class head.

    ``ignore_mismatched_sizes`` is required and is exactly what is wanted: the
    published head classifies a thousand ImageNet categories and is discarded.
    The backbone weights are the pinned revision and are not.
    """
    check_contract()
    import torch
    from transformers import AutoConfig, AutoModelForImageClassification

    pin = CLASSIFIER_BACKBONE
    kw = dict(revision=pin.revision, local_files_only=local_files_only)
    if cache_dir is not None:
        kw["cache_dir"] = str(cache_dir)
    try:
        config = AutoConfig.from_pretrained(pin.repo, **kw)
        config.num_labels = N_CLASSES
        config.id2label = {i: name for i, name in enumerate(CLASS_ORDER)}
        config.label2id = {name: i for i, name in enumerate(CLASS_ORDER)}
        model = AutoModelForImageClassification.from_pretrained(
            pin.repo, config=config, ignore_mismatched_sizes=True, **kw)
    except OSError as exc:
        raise ModelError(
            f"the pinned backbone {pin.repo}@{pin.revision} is not in the "
            "local cache. Fetch it once with scripts/32_vision_train.py "
            f"--fetch-backbone, then every later run is offline: {exc}"
        ) from exc
    if getattr(model.config, "num_labels", None) != N_CLASSES:
        raise ModelError(
            f"the built model has {model.config.num_labels} labels, not "
            f"{N_CLASSES}")
    with torch.no_grad():
        pass
    return model


def freeze_parameters(model, config: TrainConfig) -> dict:
    """Freeze what the configuration says to freeze, and report what it did.

    Returns the trainable/frozen parameter counts, which go in the manifest:
    "the backbone was frozen" is a claim, and the count is the evidence.
    """
    trainable = 0
    frozen = 0
    head_names = ("classifier",)
    stage_prefixes = _late_stage_prefixes(model, config.trainable_stages)
    for name, parameter in model.named_parameters():
        keep = name.startswith(head_names) or any(
            name.startswith(prefix) for prefix in stage_prefixes)
        if config.freeze_backbone and not keep:
            parameter.requires_grad_(False)
            frozen += parameter.numel()
        else:
            parameter.requires_grad_(True)
            trainable += parameter.numel()
    if trainable == 0:
        raise ModelError(
            "no parameter is trainable; the configuration froze the head as "
            "well as the backbone and the fit would do nothing")
    if config.freeze_backbone and config.trainable_stages > 0 \
            and not stage_prefixes:
        raise ModelError(
            f"trainable_stages={config.trainable_stages} was asked for and no "
            "backbone stage could be found to unfreeze. Rather than quietly "
            "training the head alone, this is refused: a linear probe and a "
            "fine-tune are different models and must not be confused.")
    return {"trainable_parameters": trainable, "frozen_parameters": frozen,
            "trainable_stages_requested": config.trainable_stages,
            "trainable_prefixes": sorted(stage_prefixes) + list(head_names)}


def _late_stage_prefixes(model, count: int) -> tuple[str, ...]:
    """Names of the last ``count`` backbone stages, whatever they are called.

    Discovered from the module tree rather than hard-coded, so this does not
    silently freeze everything if the backbone's attribute names differ from
    what a reader assumed.

    Matched on the *last two* path segments rather than on a fixed depth. The
    first version required exactly three segments, and this backbone's stages
    are at ``resnet.encoder.stages.N`` -- four. So it matched nothing, returned
    an empty tuple, and every run silently trained the classifier head alone
    however many stages the configuration asked for. A depth assumption is
    exactly the kind of thing that fails without raising.
    """
    if count <= 0:
        return ()
    stages = []
    for name, _module in model.named_modules():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] == "stages" and parts[-1].isdigit():
            stages.append((int(parts[-1]), name))
    if not stages:
        return ()
    stages.sort()
    return tuple(name for _index, name in stages[-count:])


def forward_logits(model, batch, device: str, *,
                   channels_last: bool = False):
    """Logits for a stacked batch of preprocessed tensors.

    ``channels_last`` matters on tensor cores: a convolution reads NHWC
    natively, and feeding it NCHW makes cuDNN transpose every activation. The
    model has to be in the same layout, which :func:`prepare_model` does once.
    """
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(batch)).to(
        device, non_blocking=True)
    if channels_last:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return model(pixel_values=tensor).logits


def prepare_model(model, device: str, tuning: "DeviceTuning"):
    """Move a model to its device in the layout the tuning asked for."""
    import torch

    model = model.to(device)
    if tuning.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def softmax(logits) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(values)
    return (exponent / exponent.sum(axis=-1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# Saving and loading, with the manifest that makes a run checkable
# ---------------------------------------------------------------------------

REQUIRED_MANIFEST_KEYS = (
    "kind", "class_order", "backbone", "config", "preprocessing", "seed",
    "dependencies", "data_manifest_sha256", "code_sha256", "weights",
    "epoch_log", "selected_epoch", "selection_criterion", "device",
    "device_report", "tuning",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_digest(paths=None) -> str:
    """A digest over the modules that decide what this model is.

    Not the whole repository: a digest that changes when an unrelated test is
    edited tells a reader nothing.  These five files are the ones whose content
    changes what a checkpoint means.
    """
    root = Path(__file__).resolve().parents[2]
    names = paths or ("src/vision/model.py", "src/vision/preprocess.py",
                      "src/vision/classes.py", "src/vision/model_ids.py",
                      "src/vision/schema.py")
    digest = hashlib.sha256()
    for name in sorted(names):
        target = root / name
        if not target.is_file():
            raise ModelError(f"the code digest needs {name} and it is missing")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(target.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def save(model, directory, *, config: TrainConfig, epoch_log: list,
         selected_epoch: int, selection_criterion: str,
         data_manifest_sha256: str, device: str,
         split_manifest_sha256: str | None = None,
         tuning: "DeviceTuning | None" = None,
         applied_tuning: dict | None = None,
         notes: str = "") -> dict:
    """Write the fitted parameters and the manifest that describes them."""
    import torch

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    if not isinstance(data_manifest_sha256, str) or len(
            data_manifest_sha256) != 64:
        raise ModelError(
            "a checkpoint must record the digest of the data manifest it was "
            "fitted on; without it the result cannot be tied to its data")
    if not epoch_log:
        raise ModelError("a checkpoint must carry its per-epoch log")

    weights_path = target / WEIGHTS_FILE
    # Contiguous in the default layout, whatever layout the fit ran in: a
    # channels_last checkpoint would still load, but its tensors would carry a
    # memory format that has nothing to do with the weights and everything to
    # do with the machine that produced them.
    state = {name: tensor.detach().to("cpu").contiguous()
             for name, tensor in model.state_dict().items()}
    torch.save(state, weights_path)
    payload = weights_path.read_bytes()

    manifest = {
        "kind": "brickagain.vision_classifier",
        "class_order": list(CLASS_ORDER),
        "backbone": CLASSIFIER_BACKBONE.as_dict(),
        "config": config.as_dict(),
        "preprocessing": preprocessing_record(),
        "seed": config.seed,
        "dependencies": dependency_versions(),
        "data_manifest_sha256": data_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "code_sha256": code_digest(),
        "weights": {"file": WEIGHTS_FILE, "sha256": _sha256_bytes(payload),
                    "bytes": len(payload)},
        "epoch_log": epoch_log,
        "selected_epoch": selected_epoch,
        "selection_criterion": selection_criterion,
        "device": device,
        "device_report": device_report(device),
        "tuning": {**(tuning or DeviceTuning()).as_dict(),
                   "applied": applied_tuning or {}},
        "notes": notes,
        "boundary": (
            "an image classifier fitted on public LEGO photographs and "
            "renders. It is unrelated to the generation track's final_H2 "
            "adapter and no number from it may be placed beside a Phase 2 "
            "result"),
    }
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ModelError(f"the manifest is missing {missing}")
    (target / VISION_MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8")
    return manifest


def read_manifest(directory) -> dict:
    """Read and check a checkpoint manifest without loading the weights."""
    target = Path(directory) / VISION_MANIFEST
    if not target.is_file():
        raise ModelError(
            f"there is no {VISION_MANIFEST} in {directory}; a directory of "
            "weights with no manifest is not a checkpoint this project loads")
    try:
        manifest = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"{target} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get(
            "kind") != "brickagain.vision_classifier":
        raise ModelError(f"{target} does not declare itself a vision "
                         "classifier manifest")
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ModelError(f"{target} is missing {missing}")
    if list(manifest["class_order"]) != list(CLASS_ORDER):
        raise ModelError(
            f"the checkpoint was fitted with class order "
            f"{manifest['class_order']} and this code uses {list(CLASS_ORDER)}; "
            "every prediction would be mislabelled")
    return manifest


def load(directory, *, device: str | None = None,
         local_files_only: bool = True, cache_dir=None,
         verify_digest: bool = True):
    """Load a fitted checkpoint, verifying its digest and its preprocessing."""
    import torch

    manifest = read_manifest(directory)
    target = Path(directory)
    weights_path = target / manifest["weights"]["file"]
    if not weights_path.is_file():
        raise ModelError(f"the manifest names {weights_path}, which is missing")
    payload = weights_path.read_bytes()
    if verify_digest:
        actual = _sha256_bytes(payload)
        if actual != manifest["weights"]["sha256"]:
            raise ModelError(
                f"{weights_path.name} hashes to {actual}, not the "
                f"{manifest['weights']['sha256']} its manifest records")
    recorded = manifest["preprocessing"]
    if (recorded.get("crop") != CROP_SIZE
            or list(recorded.get("image_mean", [])) != list(IMAGE_MEAN)
            or list(recorded.get("image_std", [])) != list(IMAGE_STD)):
        raise ModelError(
            "the checkpoint's recorded preprocessing is not the preprocessing "
            "this code applies; the head would be served inputs it was not "
            "fitted on")
    model = build_model(local_files_only=local_files_only, cache_dir=cache_dir)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(state, strict=True)
    if getattr(incompatible, "missing_keys", None):
        raise ModelError(
            f"the checkpoint is missing parameters: {incompatible.missing_keys}")
    resolved = resolve_device(device)
    tuning = tuning_for(resolved)
    apply_tuning(tuning, resolved)
    model = prepare_model(model, resolved, tuning).eval()
    return model, manifest, resolved


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_arrays(model, images, *, device: str, batch_size: int = 32,
                   tuning: "DeviceTuning | None" = None
                   ) -> list[Prediction]:
    """Classify a list of RGB arrays, returning one prediction each."""
    settings = tuning or tuning_for(device)
    out: list[Prediction] = []
    batch: list[np.ndarray] = []
    for image in images:
        batch.append(model_tensor(np.asarray(image)))
        if len(batch) == batch_size:
            out.extend(_run_batch(model, batch, device, settings))
            batch = []
    if batch:
        out.extend(_run_batch(model, batch, device, settings))
    return out


def _run_batch(model, batch, device: str,
               tuning: "DeviceTuning") -> list[Prediction]:
    import torch

    with torch.no_grad(), autocast_context(tuning, device):
        logits = forward_logits(model, np.stack(batch), device,
                                channels_last=tuning.channels_last)
        # Softmax in float32 whatever the forward pass ran in: a probability
        # vector read out of bfloat16 has about three decimal digits, and the
        # low-confidence threshold is compared against it.
        probabilities = softmax(logits.detach().float().to("cpu").numpy())
    return [from_scores(METHOD_LEARNED, row,
                        features={"source": "softmax over pinned backbone"})
            for row in probabilities]


def check_backbone_cache(*, cache_dir=None) -> dict:
    """Report whether a strict-offline load would succeed, and what is missing.

    Called by the training script before it does anything expensive, and by
    the UI before it offers the learned method, so a missing download is a
    sentence rather than a stack trace from inside a library.
    """
    from huggingface_hub import try_to_load_from_cache

    pin = CLASSIFIER_BACKBONE
    present = {}
    for name in pin.files:
        try:
            hit = try_to_load_from_cache(
                pin.repo, name, revision=pin.revision,
                cache_dir=str(cache_dir) if cache_dir else None)
        except Exception:                       # noqa: BLE001 - cache probing
            hit = None
        present[name] = bool(hit) and hit is not None and str(hit) != "_"
    processor = None
    for name in pin.files:
        if name == "preprocessor_config.json" and present.get(name):
            try:
                hit = try_to_load_from_cache(
                    pin.repo, name, revision=pin.revision,
                    cache_dir=str(cache_dir) if cache_dir else None)
                processor = json.loads(Path(str(hit)).read_text("utf-8"))
                check_processor_config(processor)
            except Exception as exc:            # noqa: BLE001 - see message
                raise ModelError(
                    "the published preprocessor configuration does not match "
                    f"the pinned values: {exc}") from exc
    return {"repo": pin.repo, "revision": pin.revision,
            "licence": pin.licence, "files": present,
            "complete": all(present.values()),
            "processor_checked": processor is not None}
