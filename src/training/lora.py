"""LoRA smoke-test plumbing: sampling, collation, and where training starts.

Scope: this module exists to make a *reproducible* smoke test possible. It is
not the A-E experiment, and nothing here selects a model or tunes anything.

Where training starts, and why it matters
-----------------------------------------

BrickGPT is published as a LoRA adapter over ``meta-llama/Llama-3.2-1B-Instruct``
(r=32, alpha=16, q_proj/v_proj). Training "a LoRA on Llama-3.2-1B" would
therefore throw the published checkpoint away and start from a base model that
has never seen a brick, while still being describable as fine-tuning -- the
most expensive kind of silent mistake, because the run succeeds and the numbers
merely disappoint.

:func:`build_model` takes the explicit route: load the base, apply the
published adapter, **merge it into the weights**, then attach a fresh adapter
of our own on top. So

* the starting point provably contains BrickGPT (``merged_adapter`` is
  recorded, and :func:`assert_starts_from_brickgpt` checks the weights moved);
* the new adapter carries our own rank and alpha rather than inheriting the
  published ones;
* what gets saved is only our delta, and the published adapter is never
  modified.

The alternative -- continuing to train the published adapter in place -- would
pin us to its hyperparameters and produce a checkpoint whose provenance is a
mixture of theirs and ours. Recorded either way in the report as
``start_from``; the one thing that must never happen is for it to go
unrecorded.

Sampling
--------

Whole pairs only. A pair is 2 roles x 4 inventory framings, and the roles are
two builds of one shape: taking half a pair would put a control in training
whose counterfactual is absent, which is precisely the contrast the dataset was
built to teach. Selection is by seeded shuffle over sorted pair ids, so the
subset is a function of the seed alone.

Validation comes from the ``val`` split only. The test split is not read by
this module at all -- not for early stopping, not for checkpoint selection,
not for a sanity peek.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Shared with the inference path via src.model_ids, so the trained and
# untrained arms cannot end up on different base weights.
from src.model_ids import (  # noqa: E402
    ADAPTER as PUBLISHED_ADAPTER,
    ADAPTER_REVISION as PUBLISHED_REVISION,
    BASE_MODEL,
    BASE_REVISION,
    LOCAL_ADAPTER_MANIFEST as MANIFEST_NAME,
    TOKENIZER,
    TOKENIZER_REVISION,
)

# MANIFEST_NAME comes from src.model_ids: it is written beside every saved
# adapter. A local adapter directory is just weights -- nothing in it says the
# weights were fitted on top of a *merged* BrickGPT rather than on bare Llama,
# so loading it the obvious way (``BrickGPT(adapter=<path>)``) produces a model
# that runs, generates, and is wrong. The manifest makes the order checkable.

#: The only correct order. Each step's output is the next step's input.
LOAD_ORDER = ("base", "published_adapter", "merge", "local_adapter")

#: One pre-declared configuration, fixed before the run and not revisited
#: afterwards. The 2e-3 vs lower-LR comparison the project owes itself is a
#: separate round with its own report; picking the nicer of two runs after
#: seeing both would make this a selection, not a smoke test.
@dataclass(frozen=True)
class LoraConfig_:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    learning_rate: float = 1e-4
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    batch_size: int = 1
    grad_accum: int = 8
    max_length: int = 2048
    epochs: int = 1
    seed: int = 0
    #: bf16 throughout. No 4-bit: bitsandbytes has no dependable Apple Silicon
    #: path, and section 9.8 is explicit that QLoRA is a memory option rather
    #: than a requirement -- taking it here would trade reproducibility for a
    #: saving 48 GB does not need.
    dtype: str = "bfloat16"
    quantization: str = "none (bf16); 4-bit deliberately not used on MPS"

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["target_modules"] = list(d["target_modules"])
        d["effective_batch"] = self.effective_batch
        return d


@dataclass
class Row:
    sample_id: str
    pair_id: str
    object_id: str
    role: str
    variant: str
    prompt: str
    target: str
    n_tokens: int


def read_rows(path: Path) -> list[Row]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        r = json.loads(line)
        out.append(Row(
            sample_id=r["sample_id"], pair_id=r["pair_id"],
            object_id=r["object_id"], role=r["role"], variant=r["variant"],
            prompt=r["prompt"], target=r["target"], n_tokens=r["n_tokens"],
        ))
    return out


def sample_pairs(rows: list[Row], n_pairs: int, seed: int) -> list[Row]:
    """Take ``n_pairs`` whole pairs, deterministically.

    Shuffles *sorted* pair ids so the result depends on the seed and not on
    the order rows happen to sit in the file. Every row of a chosen pair comes
    along; no pair is ever split.
    """
    by_pair: dict[str, list[Row]] = {}
    for r in rows:
        by_pair.setdefault(r.pair_id, []).append(r)

    ids = sorted(by_pair)
    if n_pairs > len(ids):
        raise ValueError(f"asked for {n_pairs} pairs, file holds {len(ids)}")
    rng = random.Random(seed)
    rng.shuffle(ids)
    chosen = sorted(ids[:n_pairs])

    picked = [r for pid in chosen for r in sorted(by_pair[pid],
                                                  key=lambda r: r.sample_id)]
    sizes = {len(by_pair[pid]) for pid in chosen}
    if sizes != {8}:
        raise ValueError(f"pairs are not all 8 rows: sizes seen {sorted(sizes)}")
    return picked


def split_stats(rows: list[Row]) -> dict:
    return {
        "samples": len(rows),
        "pairs": len({r.pair_id for r in rows}),
        "objects": len({r.object_id for r in rows}),
        "roles": dict(sorted({
            role: sum(1 for r in rows if r.role == role)
            for role in {r.role for r in rows}}.items())),
        "variants": dict(sorted({
            v: sum(1 for r in rows if r.variant == v)
            for v in {r.variant for r in rows}}.items())),
    }


def check_no_object_overlap(train: list[Row], val: list[Row]) -> dict:
    """train and val must not share an object_id.

    The split manifest already guarantees this, which is exactly why it is
    worth re-checking here: a sampling bug would otherwise be invisible.
    """
    a = {r.object_id for r in train}
    b = {r.object_id for r in val}
    shared = a & b
    if shared:
        raise ValueError(
            f"{len(shared)} object_id(s) appear in both train and val "
            f"(e.g. {sorted(shared)[0]})")
    return {"train_objects": len(a), "val_objects": len(b), "shared": 0}


# ---- encoding ---------------------------------------------------------------

@dataclass
class Encoded:
    input_ids: list[int]
    labels: list[int]
    n_prompt_tokens: int
    truncated: bool


def encode_row(tok, row: Row, max_length: int) -> Encoded:
    """Prompt masked to -100; target and its EOS are the only supervision.

    Delegates to :func:`src.data.instruction.encode`, the same function that
    produced the stored token counts and that the instruction audit re-derives
    all 25,568 rows against. A second implementation here would be a second
    thing to keep in step, and the mask is exactly where a silent divergence
    would be most expensive.
    """
    from src.data.instruction import encode as encode_example

    ex = type("E", (), {"prompt": row.prompt, "target": row.target})()
    enc = encode_example(tok, ex)
    ids, labels = enc["input_ids"], enc["labels"]
    return Encoded(ids[:max_length], labels[:max_length],
                   enc["n_prompt_tokens"], truncated=len(ids) > max_length)


def collate(batch: list[Encoded], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad; padding is masked out of both attention and the loss."""
    width = max(len(e.input_ids) for e in batch)
    ids, labels, mask = [], [], []
    for e in batch:
        pad = width - len(e.input_ids)
        ids.append(e.input_ids + [pad_id] * pad)
        labels.append(e.labels + [-100] * pad)
        mask.append([1] * len(e.input_ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


# ---- model ------------------------------------------------------------------

def load_merged_brickgpt(*, dtype=torch.bfloat16, local_files_only: bool = False,
                         _calls: list | None = None):
    """Steps 1-3 of :data:`LOAD_ORDER`: base, published adapter, merge.

    Shared by training and by inference so the two cannot diverge on the one
    thing that decides what a saved delta means. ``_calls`` records the call
    sequence for tests, which is the only way to check an *order* rather than
    an end state.

    ``local_files_only`` is passed explicitly rather than read from the
    environment. Report 16's measured child sets it, so a run either resolves
    every weight from the local cache or refuses -- it never depends on
    whether the operator remembered to export ``HF_HUB_OFFLINE``, and it never
    reaches the network halfway through a boot it has already spent.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    record = _calls if _calls is not None else []

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, dtype=dtype,
        local_files_only=local_files_only)
    record.append(("base", BASE_MODEL, BASE_REVISION))
    before = _weight_fingerprint(base)

    published = PeftModel.from_pretrained(
        base, PUBLISHED_ADAPTER, revision=PUBLISHED_REVISION,
        local_files_only=local_files_only)
    record.append(("published_adapter", PUBLISHED_ADAPTER, PUBLISHED_REVISION))

    merged = published.merge_and_unload()
    record.append(("merge", None, None))
    after = _weight_fingerprint(merged)

    if before == after:
        raise RuntimeError(
            "merging the published adapter did not change any weight; "
            "training would silently start from bare Llama")

    return merged, {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "published_adapter": PUBLISHED_ADAPTER,
        "published_adapter_revision": PUBLISHED_REVISION,
        "merge_changed_weights": True,
        "load_order": list(LOAD_ORDER),
    }


def write_manifest(ckpt_dir: Path, info: dict, cfg: "LoraConfig_") -> dict:
    """Record what a saved adapter must be loaded on top of.

    Without this the directory is indistinguishable from an adapter trained on
    bare Llama, and the wrong load path fails silently rather than loudly.
    """
    ckpt_dir = Path(ckpt_dir)
    adapter = ckpt_dir / "adapter_model.safetensors"
    manifest = {
        "load_order": list(LOAD_ORDER),
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "published_adapter": PUBLISHED_ADAPTER,
        "published_adapter_revision": PUBLISHED_REVISION,
        "tokenizer": PUBLISHED_ADAPTER,
        "tokenizer_revision": PUBLISHED_REVISION,
        "adapter_sha256": sha256_file(adapter) if adapter.exists() else None,
        "lora": {"r": cfg.rank, "alpha": cfg.alpha,
                 "target_modules": list(cfg.target_modules)},
        "warning": (
            "This adapter was fitted on top of the MERGED BrickGPT weights. "
            "Applying it directly to the base model produces a model that "
            "loads and generates and is wrong. Load it with load_finetuned()."
        ),
    }
    (ckpt_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def sha256_file(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_finetuned(
    ckpt_dir: Path,
    *,
    dtype=torch.bfloat16,
    device: str = "mps",
    verify_digest: bool = True,
    local_files_only: bool = False,
    _calls: list | None = None,
):
    """Cold-start a locally trained adapter, in the one order that is correct.

    ``base -> published BrickGPT adapter -> merge -> local adapter``.

    The obvious alternative, ``BrickGPT(adapter=<local path>)``, skips the
    middle two steps and lands our delta on bare Llama. That model loads
    without complaint and emits plausible-looking bricks, so the mistake shows
    up as disappointing numbers rather than as an error -- which is why this
    function exists and why the manifest is checked rather than trusted.

    ``local_files_only`` is passed explicitly, never read from the
    environment, for the reason :func:`load_merged_brickgpt` gives: a measured
    run either resolves every weight from the local cache or refuses, and
    "did the operator export HF_HUB_OFFLINE" is not a property a measurement
    should depend on. It defaults to ``False`` because the existing callers
    are ordinary Mac evaluations; the core-acceptance runner passes ``True``.
    """
    from peft import PeftModel

    ckpt_dir = Path(ckpt_dir)
    manifest_path = ckpt_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{ckpt_dir} has no {MANIFEST_NAME}; refusing to guess what these "
            "weights were fitted on top of")
    manifest = json.loads(manifest_path.read_text())

    if tuple(manifest.get("load_order", ())) != LOAD_ORDER:
        raise ValueError(
            f"manifest load_order {manifest.get('load_order')} is not "
            f"{list(LOAD_ORDER)}; this loader cannot honour it")
    for key, expected in (("base_revision", BASE_REVISION),
                          ("published_adapter_revision", PUBLISHED_REVISION)):
        if manifest.get(key) != expected:
            raise ValueError(
                f"manifest {key}={manifest.get(key)!r} but this code pins "
                f"{expected!r}; the delta would sit on different weights")

    if verify_digest and manifest.get("adapter_sha256"):
        actual = sha256_file(ckpt_dir / "adapter_model.safetensors")
        if actual != manifest["adapter_sha256"]:
            raise ValueError(
                f"adapter_model.safetensors digest {actual[:12]}... does not "
                f"match the manifest ({manifest['adapter_sha256'][:12]}...); "
                "the checkpoint changed after it was written")

    merged, info = load_merged_brickgpt(dtype=dtype,
                                        local_files_only=local_files_only,
                                        _calls=_calls)
    # The checkpoint is a directory on this machine, so nothing about it can
    # reach the hub -- but the flag is passed anyway, because the value that
    # matters is the one carried through to the base and published adapter
    # above, and a reader checking "is this call strictly offline" should be
    # able to see one answer rather than two.
    model = PeftModel.from_pretrained(merged, str(ckpt_dir),
                                      local_files_only=local_files_only)
    if _calls is not None:
        _calls.append(("local_adapter", str(ckpt_dir), None))
    model.to(device).eval()
    return model, {**info, "local_adapter": str(ckpt_dir),
                   "manifest": manifest}


def build_model(cfg: LoraConfig_, *, device: str = "mps",
                local_files_only: bool = False):
    """Base -> published BrickGPT adapter -> merge -> fresh trainable adapter.

    Returns ``(model, info)``; ``info`` records exactly what was loaded and
    merged so the report can state where training started rather than imply it.
    """
    from peft import LoraConfig, get_peft_model

    merged, base_info = load_merged_brickgpt(
        dtype=getattr(torch, cfg.dtype), local_files_only=local_files_only)

    lora = LoraConfig(
        r=cfg.rank, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
        target_modules=list(cfg.target_modules), bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(merged, lora)
    model.to(device)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    info = {
        **base_info,
        "published_adapter_config": {"r": 32, "alpha": 16,
                                     "target_modules": ["q_proj", "v_proj"]},
        "start_from": (
            "published BrickGPT adapter merged into the base weights, then a "
            "new LoRA attached on top; the published adapter is not modified "
            "and is not what gets saved"
        ),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_modules": sorted({n.rsplit(".", 1)[0] for n in trainable}),
        "n_trainable_tensors": len(trainable),
        "dtype": cfg.dtype,
        "device": device,
    }
    return model, info


def _weight_fingerprint(model) -> float:
    """Cheap scalar over the projections the published adapter touches."""
    total = 0.0
    for name, p in model.named_parameters():
        if name.endswith(("q_proj.weight", "v_proj.weight")):
            total += float(p.detach().float().abs().sum())
    return round(total, 3)


def assert_only_lora_trainable(model) -> list[str]:
    """Every trainable tensor must be a LoRA tensor, and there must be some."""
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("nothing is trainable")
    stray = [n for n in trainable if "lora_" not in n]
    if stray:
        raise RuntimeError(f"non-LoRA parameters are trainable: {stray[:5]}")
    return trainable
