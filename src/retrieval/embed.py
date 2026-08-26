"""Sentence embeddings from the pinned multilingual model.

The delivery path's existing retrieval is a deterministic lexical baseline.  It
tokenises Chinese without an external service and it is honest about what it is,
but a Chinese request and an English caption share almost no characters, so it
can only match them by accident.  This module is the part that makes the
Chinese-in, English-catalogue case work on meaning.

Four things are fixed here because a retrieval index and the queries run
against it have to be produced the same way or the cosines are measured against
the wrong geometry:

**The model and its revision.**  From :mod:`src.vision.model_ids`.  An index
built under one revision and queried under another is not an index.

**The prefixes.**  This model family requires ``query: `` and ``passage: ``;
without them the vectors are worse and nothing reports it.  The two strings are
constants, and :func:`embed` will not run without being told which side it is
embedding.

**Mean pooling over the attention mask, then L2 normalisation.**  Pooling over
padding is a common and silent mistake -- it drags every short caption towards
the same vector.  Normalising afterwards is what makes a dot product a cosine,
so the index can be a matrix multiply.

**The device, and it defaults to the CPU.**  Not for speed: for reproducibility.
The catalogue is a few thousand captions, so the CPU is fast enough, and a
vector computed on one backend and compared against vectors computed on another
can reorder near-ties.  The device used is written into the index manifest, and
a query on a different one is reported rather than assumed harmless.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from src.vision.model_ids import (E5_PASSAGE_PREFIX, E5_QUERY_PREFIX,
                                  TEXT_EMBEDDING)

KIND_QUERY = "query"
KIND_PASSAGE = "passage"
KINDS = (KIND_QUERY, KIND_PASSAGE)

#: Tokens per text.  Captions in the catalogue are one or two sentences and
#: requests are shorter; a fixed cap keeps the vectors reproducible and stops a
#: pasted essay from becoming an expensive query.
MAX_TOKENS = 192

#: Texts per forward pass.
BATCH = 32

#: Vector width this model produces.  Checked rather than trusted, because a
#: silently different width would build an index that cannot be queried.
DIMENSION = 384


class EmbedError(RuntimeError):
    """The embedding model is unavailable, or was used inconsistently."""


@dataclass
class Embedder:
    """A loaded model, its identity, and the settings vectors depend on."""

    model: object
    tokenizer: object
    device: str
    repo: str
    revision: str
    dimension: int

    def identity(self) -> dict:
        return {"repo": self.repo, "revision": self.revision,
                "licence": TEXT_EMBEDDING.licence, "device": self.device,
                "dimension": self.dimension, "max_tokens": MAX_TOKENS,
                "pooling": "mean over the attention mask, then L2 normalised",
                "query_prefix": E5_QUERY_PREFIX,
                "passage_prefix": E5_PASSAGE_PREFIX}

    def identity_digest(self) -> str:
        """A digest over everything that changes the vectors.

        The device is *not* in it: a run on another backend must produce a
        comparable index, and the manifest records the device separately so a
        mismatch can be reported without invalidating the index outright.
        """
        parts = [self.repo, self.revision, str(self.dimension),
                 str(MAX_TOKENS), E5_QUERY_PREFIX, E5_PASSAGE_PREFIX,
                 "mean-masked-l2"]
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def load(*, device: str = "cpu", local_files_only: bool = True,
         cache_dir=None) -> Embedder:
    """Load the pinned embedding model, or say exactly what is missing."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment boundary
        raise EmbedError(
            "torch and transformers are required for multilingual retrieval; "
            "install the pinned vision requirements, or use the lexical "
            "baseline which needs neither") from exc

    pin = TEXT_EMBEDDING
    kw = dict(revision=pin.revision, local_files_only=local_files_only)
    if cache_dir is not None:
        kw["cache_dir"] = str(cache_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(pin.repo, **kw)
        model = AutoModel.from_pretrained(pin.repo, **kw)
    except OSError as exc:
        raise EmbedError(
            f"the pinned embedding model {pin.repo}@{pin.revision} is not in "
            "the local cache. Fetch it once with scripts/34_rag_index.py "
            f"--fetch-model; every later run is offline: {exc}") from exc
    model = model.to(device).eval()
    width = int(getattr(model.config, "hidden_size", 0))
    if width != DIMENSION:
        raise EmbedError(
            f"{pin.repo} produces {width}-dimensional vectors and this code "
            f"expects {DIMENSION}")
    return Embedder(model=model, tokenizer=tokenizer, device=device,
                    repo=pin.repo, revision=pin.revision, dimension=width)


def _prefix(kind: str) -> str:
    if kind == KIND_QUERY:
        return E5_QUERY_PREFIX
    if kind == KIND_PASSAGE:
        return E5_PASSAGE_PREFIX
    raise EmbedError(f"kind must be one of {list(KINDS)}, not {kind!r}")


def embed(embedder: Embedder, texts, *, kind: str,
          batch: int = BATCH) -> np.ndarray:
    """Embed texts as an ``(n, dimension)`` float32 array of unit vectors.

    ``kind`` has no default.  Embedding a query with the passage prefix is a
    mistake that costs quality and reports nothing, so the caller has to say
    which side this is.
    """
    import torch

    prefix = _prefix(kind)
    items = [str(text) for text in texts]
    if not items:
        raise EmbedError("there is nothing to embed")
    if any(not text.strip() for text in items):
        raise EmbedError("an empty text cannot be embedded")

    out = np.zeros((len(items), embedder.dimension), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(items), batch):
            chunk = [prefix + text for text in items[start:start + batch]]
            encoded = embedder.tokenizer(
                chunk, padding=True, truncation=True, max_length=MAX_TOKENS,
                return_tensors="pt")
            encoded = {key: value.to(embedder.device)
                       for key, value in encoded.items()}
            hidden = embedder.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out[start:start + len(chunk)] = pooled.to("cpu").numpy()
    return out


def check_cache(*, cache_dir=None) -> dict:
    """Whether a strict-offline load would work, and what is missing if not."""
    from huggingface_hub import try_to_load_from_cache

    pin = TEXT_EMBEDDING
    present = {}
    for name in pin.files:
        try:
            hit = try_to_load_from_cache(
                pin.repo, name, revision=pin.revision,
                cache_dir=str(cache_dir) if cache_dir else None)
        except Exception:                       # noqa: BLE001 - cache probing
            hit = None
        present[name] = hit is not None and str(hit) != "_"
    # The tokenizer needs either the fast tokenizer JSON or the sentencepiece
    # model, not necessarily both, so completeness is not simply "all present".
    core = ("config.json", "model.safetensors", "tokenizer_config.json")
    tokenizer_ok = present.get("tokenizer.json") or present.get(
        "sentencepiece.bpe.model")
    return {"repo": pin.repo, "revision": pin.revision,
            "licence": pin.licence, "files": present,
            "complete": all(present.get(name) for name in core)
            and bool(tokenizer_ok)}
