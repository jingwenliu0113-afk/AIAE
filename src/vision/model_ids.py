"""The vision and retrieval model identities, pinned in one place.

:mod:`src.model_ids` does this for the generation track and says why: two code
paths that resolve *different* weights measure that difference as well as the
thing under test, and nothing about either path looks wrong.  The same applies
to a classifier fitted on a CUDA node and served on a Mac, and to an embedding
index built once and queried later.

Each entry carries four things, and all four are required rather than nice to
have:

``repo``
    The Hugging Face repository.
``revision``
    The exact commit. A tag or ``main`` is not a pin: an upstream push moves
    the weights under a fitted head, and the head is then wrong in a way no
    file records.
``licence``
    Recorded so the compatibility question is answered in the repository and
    not from memory. The backbone here is Apache-2.0 and the embedding model
    is MIT; a repository whose card says only ``other`` was passed over for
    that reason rather than adapted around.
``files``
    What must be present locally for a strict-offline run to work, so a
    missing file is named before a load fails somewhere inside a library.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PinnedModel:
    repo: str
    revision: str
    licence: str
    files: tuple[str, ...]
    purpose: str

    def as_dict(self) -> dict:
        return {"repo": self.repo, "revision": self.revision,
                "licence": self.licence, "files": list(self.files),
                "purpose": self.purpose}


#: The image-classification backbone the eight-class head is fitted on top of.
#: ResNet-18: 11.7M parameters, ImageNet-1k, Apache-2.0, and one of the three
#: candidates the plan names. Small enough that final inference stays on the
#: Mac, which is the requirement that ruled out anything larger.
CLASSIFIER_BACKBONE = PinnedModel(
    repo="microsoft/resnet-18",
    revision="65a5785d9156231087c481e0c7dd33a5ff6f7e3e",
    licence="apache-2.0",
    files=("config.json", "model.safetensors", "preprocessor_config.json"),
    purpose="ImageNet-pretrained backbone for the eight-class brick head",
)

#: The multilingual sentence embedding for retrieval. MIT, 384 dimensions,
#: twelve layers, and genuinely multilingual -- which is the point: the queries
#: are Chinese and the captions are English, so a monolingual model would be
#: measuring nothing. Loaded through ``AutoModel`` with mean pooling rather
#: than through ``sentence-transformers``, which is not in this project's
#: pinned environment.
TEXT_EMBEDDING = PinnedModel(
    repo="intfloat/multilingual-e5-small",
    revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    licence="mit",
    files=("config.json", "model.safetensors", "tokenizer.json",
           "tokenizer_config.json", "special_tokens_map.json",
           "sentencepiece.bpe.model"),
    purpose="multilingual caption and query embedding for retrieval",
)

#: This family requires these prefixes; without them the vectors are worse in
#: a way no error reports. Written here because the index and the query path
#: must use the same two strings or every cosine is measured against the wrong
#: geometry.
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

#: Marker file written beside every locally fitted vision head, so a loader
#: can recognise one without importing the training code.
VISION_MANIFEST = "brickagain_vision_manifest.json"

#: Marker file written beside a built retrieval index.
RETRIEVAL_MANIFEST = "brickagain_retrieval_manifest.json"

PINNED = {
    "classifier_backbone": CLASSIFIER_BACKBONE,
    "text_embedding": TEXT_EMBEDDING,
}

__all__ = [
    "PinnedModel", "CLASSIFIER_BACKBONE", "TEXT_EMBEDDING", "PINNED",
    "E5_QUERY_PREFIX", "E5_PASSAGE_PREFIX", "VISION_MANIFEST",
    "RETRIEVAL_MANIFEST",
]
