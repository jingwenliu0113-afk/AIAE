"""A local vector index: a matrix, a manifest, and four ways to refuse.

Deliberately not FAISS or Chroma.  The catalogue is a few thousand captions of
384 dimensions -- a single ``float32`` matrix multiply, well under a
millisecond -- and an approximate-nearest-neighbour library would add a
dependency, a build step and a source of non-determinism to buy nothing.  Exact
cosine over a normalised matrix returns the same ranking every time, which is
what a demonstration that has to be re-runnable needs.

The index is only as trustworthy as its provenance, so the manifest carries all
of it and loading checks all of it:

* the **embedding model and revision**, because vectors from a different
  revision are not comparable to these;
* the **catalogue digest and the frozen split manifest digest**, so an index
  cannot outlive the data it was built from;
* the **vector file digest**, so a swapped or truncated matrix is refused;
* the **shape and the row order**, so a document list and a matrix that have
  drifted apart cannot be queried as though they matched;
* the **object identifier of every row**, because that is what the same-object
  exclusion is applied to.  A missing or empty ``object_id`` does not make the
  exclusion noisy, it makes it silently do nothing: the excluded work comes
  back, ranked first, and nothing in the output says the guard was off.  So a
  mapping that is absent, incomplete, empty-valued or disagrees with the
  catalogue is refused at load, not defaulted to ``""``.

**Train only, and the exclusion is part of the search.**  The documents come
from the same loader the delivery path uses, which refuses any row that is not
``split=train`` and checks every row against the frozen object-level split
manifest.  ``exclude_object_id`` is applied inside :meth:`VectorIndex.search`
before ranking, not to its output, so a caller cannot forget to apply it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MANIFEST_KIND = "brickagain.retrieval_index"
VECTORS_FILE = "vectors.npy"
MANIFEST_FILE = "index_manifest.json"


class IndexError_(ValueError):
    """The index, its manifest or a query is not usable."""


@dataclass(frozen=True)
class Document:
    """One indexed work: the text that was embedded, plus its evidence.

    ``object_id`` is kept so the same-object exclusion can be applied, and it
    is the one field never written into an explanation or a page: the public
    identifier is ``catalog_id``, which is a digest of the structure id.
    """

    catalog_id: str
    caption: str
    n_bricks: int
    required: dict[str, int]
    touches_ground: bool
    connected: bool
    object_id: str = field(repr=False, default="")

    def embed_text(self) -> str:
        """What is actually embedded for this document.

        The caption, plus the two facts a request most often states that a
        caption does not: how many bricks the work uses and which parts. A
        request for "a small car under thirty bricks" then has something to
        match on beyond the word "car", and the extra text is generated from
        the structure rather than written by hand.
        """
        parts = " ".join(f"{part}x{count}"
                         for part, count in sorted(self.required.items()))
        return (f"{self.caption} "
                f"[{self.n_bricks} bricks; parts: {parts}]")

    def as_dict(self) -> dict:
        return {"catalog_id": self.catalog_id, "caption": self.caption,
                "n_bricks": self.n_bricks, "required": dict(self.required),
                "touches_ground": self.touches_ground,
                "stud_only_connected": self.connected}


@dataclass(frozen=True)
class Hit:
    """One document the embedding search returned, at its semantic position.

    ``semantic_rank`` is named rather than called ``rank`` because there is a
    second ordering downstream -- the inventory and static-structure re-rank --
    and the two are different facts about the same work.  A single ``rank``
    field is what let a re-ranked position be printed under the words "semantic
    rank"; two names cannot be confused for one another.
    """

    document: Document
    score: float
    semantic_rank: int

    def as_dict(self) -> dict:
        return {"semantic_rank": self.semantic_rank,
                "semantic_score": round(self.score, 6),
                **self.document.as_dict()}


@dataclass(frozen=True)
class VectorIndex:
    """Documents, their unit vectors, and the provenance of both."""

    documents: tuple[Document, ...] = field(repr=False)
    vectors: np.ndarray = field(repr=False)
    embedding: dict
    catalog_sha256: str
    split_manifest_sha256: str
    identity_digest: str
    build_device: str

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise IndexError_("the vector matrix must be two-dimensional")
        if self.vectors.shape[0] != len(self.documents):
            raise IndexError_(
                f"{self.vectors.shape[0]} vector(s) for "
                f"{len(self.documents)} document(s); a matrix and a document "
                "list that have drifted apart would return the wrong work for "
                "every query")
        if self.vectors.dtype != np.float32:
            raise IndexError_("the vector matrix must be float32")
        norms = np.linalg.norm(self.vectors, axis=1)
        if norms.size and not np.allclose(norms, 1.0, atol=1e-3):
            raise IndexError_(
                "the vectors are not unit length, so a dot product is not a "
                "cosine and the ranking would be by magnitude")

    @property
    def size(self) -> int:
        return len(self.documents)

    def check_against_catalog(self, catalog) -> None:
        """Refuse an index whose object ids are not the catalogue's.

        The manifest check at load time proves the mapping is complete and
        non-empty; this proves it is *right*.  An index carrying plausible but
        wrong object ids would exclude the wrong work -- and, worse, would not
        exclude the requested one -- with every digest still agreeing.
        """
        by_id = {item.catalog_id: item for item in catalog.items}
        wrong = []
        for document in self.documents:
            item = by_id.get(document.catalog_id)
            if item is None:
                raise IndexError_(
                    f"the index holds catalog_id {document.catalog_id!r}, "
                    "which is not in the loaded catalogue")
            if item.object_id != document.object_id:
                wrong.append(document.catalog_id)
        if wrong:
            raise IndexError_(
                f"{len(wrong)} row(s) carry an object_id the catalogue does "
                f"not agree with, the first being {wrong[0]!r}; the "
                "same-object exclusion would be applied to the wrong work")

    def search(self, query: np.ndarray, *, top_n: int = 10,
               exclude_object_id: str | None = None) -> tuple[Hit, ...]:
        """Exact cosine search, with the same-object exclusion applied first.

        Ties break on ``catalog_id`` so the ranking is total and reproducible.
        The returned hits are in descending score order and their
        ``semantic_rank`` counts from one, so rank and score are monotone
        together and neither can be read off the other's ordering later.
        """
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise IndexError_("top_n must be a positive whole number")
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.vectors.shape[1]:
            raise IndexError_(
                f"the query has {vector.shape[0]} dimensions and the index "
                f"has {self.vectors.shape[1]}")
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise IndexError_("the query vector is zero")
        vector = vector / norm

        eligible = [i for i, document in enumerate(self.documents)
                    if exclude_object_id is None
                    or document.object_id != exclude_object_id]
        if not eligible:
            raise IndexError_(
                "the same-object exclusion removed every document from the "
                "index")
        scores = self.vectors[eligible] @ vector
        order = sorted(range(len(eligible)),
                       key=lambda k: (-float(scores[k]),
                                      self.documents[eligible[k]].catalog_id))
        return tuple(
            Hit(document=self.documents[eligible[k]],
                score=float(scores[k]), semantic_rank=rank)
            for rank, k in enumerate(order[:top_n], 1))

    # -- persistence -----------------------------------------------------
    def manifest(self, *, vectors_sha256: str) -> dict:
        return {
            "kind": MANIFEST_KIND,
            "embedding": self.embedding,
            "identity_digest": self.identity_digest,
            "build_device": self.build_device,
            "catalog_sha256": self.catalog_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "documents": len(self.documents),
            "dimension": int(self.vectors.shape[1]),
            "vectors": {"file": VECTORS_FILE, "sha256": vectors_sha256},
            "rows": [document.catalog_id for document in self.documents],
            "index_kind": (
                "exact cosine over a normalised float32 matrix, computed with "
                "NumPy. Not an approximate index: the ranking is the same on "
                "every run"),
            "boundary": (
                "train split only, checked row by row against the frozen "
                "object-level split manifest by the catalogue loader. The "
                "public identifier is catalog_id; object_id is kept for the "
                "same-object exclusion and is never published"),
            "not_a_metric": (
                "building an index measures nothing. No retrieval quality "
                "claim follows from this file"),
        }

    def save(self, directory) -> tuple[Path, str]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        vectors_path = target / VECTORS_FILE
        # ``allow_pickle`` stays off on both sides: a vector file is numbers.
        with vectors_path.open("wb") as handle:
            np.save(handle, self.vectors, allow_pickle=False)
        payload = vectors_path.read_bytes()
        manifest = self.manifest(
            vectors_sha256=hashlib.sha256(payload).hexdigest())
        manifest["document_rows"] = [document.as_dict()
                                     for document in self.documents]
        manifest["object_rows"] = {document.catalog_id: document.object_id
                                   for document in self.documents}
        body = json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")
        (target / MANIFEST_FILE).write_bytes(body)
        return target, hashlib.sha256(body).hexdigest()


def check_object_rows(objects, catalog_ids) -> dict[str, str]:
    """Refuse an object-id mapping that would turn the exclusion off.

    Exact, in both directions.  Every document row must have an entry, every
    entry must be a non-empty string, and no entry may name a row that is not
    there.  ``None`` is not "no exclusion needed", it is a manifest that
    cannot support one, and it is refused with that said.
    """
    wanted = list(catalog_ids)
    if objects is None:
        raise IndexError_(
            "the manifest carries no object_rows. Without them the "
            "same-object exclusion would silently match nothing and the work "
            "a caller asked to exclude would be returned as a recommendation")
    if not isinstance(objects, dict):
        raise IndexError_(
            "object_rows must be a {catalog_id: object_id} mapping, not "
            f"{type(objects).__name__}")
    missing = [key for key in wanted if key not in objects]
    if missing:
        raise IndexError_(
            f"{len(missing)} of {len(wanted)} document row(s) have no "
            f"object_id, the first being {missing[0]!r}; a partial mapping "
            "would exclude some works and silently not others")
    extra = sorted(set(objects) - set(wanted))
    if extra:
        raise IndexError_(
            f"object_rows names {len(extra)} catalog_id(s) that are not "
            f"document rows, the first being {extra[0]!r}; the mapping and "
            "the index do not describe the same set of works")
    empty = [key for key in wanted
             if not isinstance(objects[key], str) or not objects[key].strip()]
    if empty:
        raise IndexError_(
            f"{len(empty)} object_id(s) are empty or not text, the first "
            f"being for {empty[0]!r}; an empty object_id never matches, so "
            "the exclusion would pass over that work")
    return {key: str(objects[key]) for key in wanted}


def load(directory, *, expected_manifest_sha256: str | None = None,
         expected_identity_digest: str | None = None,
         expected_catalog_sha256: str | None = None,
         expected_split_manifest_sha256: str | None = None) -> VectorIndex:
    """Load an index, refusing every mismatch rather than warning about it."""
    target = Path(directory)
    manifest_path = target / MANIFEST_FILE
    if not manifest_path.is_file():
        raise IndexError_(f"there is no {MANIFEST_FILE} in {target}")
    raw = manifest_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and actual != \
            expected_manifest_sha256:
        raise IndexError_(
            f"the index manifest digest is {actual}, not the expected "
            f"{expected_manifest_sha256}")
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IndexError_(f"{manifest_path} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("kind") != MANIFEST_KIND:
        raise IndexError_(f"{manifest_path} is not a retrieval index manifest")
    if expected_identity_digest is not None and manifest.get(
            "identity_digest") != expected_identity_digest:
        raise IndexError_(
            "the index was built by a different embedding configuration "
            f"({manifest.get('identity_digest')}) than the one loaded now "
            f"({expected_identity_digest}); the vectors are not comparable")
    if expected_catalog_sha256 is not None and manifest.get(
            "catalog_sha256") != expected_catalog_sha256:
        raise IndexError_(
            f"the index was built from catalogue {manifest.get('catalog_sha256')}"
            f" and the catalogue now present is {expected_catalog_sha256}")
    if expected_split_manifest_sha256 is not None and manifest.get(
            "split_manifest_sha256") != expected_split_manifest_sha256:
        raise IndexError_(
            "the index was built against frozen split manifest "
            f"{manifest.get('split_manifest_sha256')} and the split manifest "
            f"now in force is {expected_split_manifest_sha256}; the "
            "train-only boundary this index was built under is not the "
            "boundary that applies now")

    vectors_path = target / manifest["vectors"]["file"]
    if not vectors_path.is_file():
        raise IndexError_(f"the manifest names {vectors_path}, which is missing")
    payload = vectors_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest["vectors"]["sha256"]:
        raise IndexError_(
            f"{vectors_path.name} hashes to {digest}, not the "
            f"{manifest['vectors']['sha256']} its manifest records")
    with vectors_path.open("rb") as handle:
        vectors = np.load(handle, allow_pickle=False)
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    rows = manifest.get("document_rows")
    if not isinstance(rows, list):
        raise IndexError_("the manifest carries no document rows")
    objects = check_object_rows(manifest.get("object_rows"),
                                [row["catalog_id"] for row in rows])
    documents = tuple(
        Document(catalog_id=row["catalog_id"], caption=row["caption"],
                 n_bricks=int(row["n_bricks"]),
                 required=dict(row["required"]),
                 touches_ground=bool(row["touches_ground"]),
                 connected=bool(row["stud_only_connected"]),
                 object_id=objects[row["catalog_id"]])
        for row in rows)
    if [document.catalog_id for document in documents] != list(
            manifest.get("rows", [])):
        raise IndexError_(
            "the manifest's row order and its document rows disagree; the "
            "matrix could not be matched to the documents")
    return VectorIndex(
        documents=documents, vectors=vectors,
        embedding=dict(manifest["embedding"]),
        catalog_sha256=str(manifest["catalog_sha256"]),
        split_manifest_sha256=str(manifest["split_manifest_sha256"]),
        identity_digest=str(manifest["identity_digest"]),
        build_device=str(manifest.get("build_device", "")))


def documents_from_catalog(catalog) -> tuple[Document, ...]:
    """Turn a loaded train catalogue into index documents.

    Takes the object the delivery path's own loader returns, so the train-only
    rule, the frozen split-manifest check and the canonical-row selection are
    all the ones already in place rather than a second reading of the file.
    """
    return tuple(Document(
        catalog_id=item.catalog_id, caption=item.caption,
        n_bricks=item.n_bricks, required=dict(item.required),
        touches_ground=item.touches_ground, connected=item.connected,
        object_id=item.object_id) for item in catalog.items)


def build(catalog, embedder, *, batch: int = 32) -> VectorIndex:
    """Embed a whole train catalogue into an index."""
    from src.retrieval.embed import KIND_PASSAGE, embed

    documents = documents_from_catalog(catalog)
    if not documents:
        raise IndexError_("the catalogue has no documents to index")
    vectors = embed(embedder, [document.embed_text()
                               for document in documents],
                    kind=KIND_PASSAGE, batch=batch)
    return VectorIndex(
        documents=documents, vectors=vectors,
        embedding=embedder.identity(), catalog_sha256=catalog.sha256,
        split_manifest_sha256=catalog.split_manifest_sha256,
        identity_digest=embedder.identity_digest(),
        build_device=embedder.device)
