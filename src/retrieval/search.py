"""Semantic search, then the arithmetic, then an explanation of the arithmetic.

The order matters and is the whole design:

1. the request's structured conditions are extracted (:mod:`src.retrieval.nlp`);
2. the embedding index returns semantic candidates, train-only, with the
   same-object exclusion applied inside the search;
3. the *exact* inventory calculation and the static structural checks re-rank
   them -- ``touches_ground`` and ``stud_only_connected``, from the same
   catalogue loader the delivery path uses;
4. the explanation is generated from the numbers step three produced.

Step three is what stops a good semantic match from being called buildable.  A
work that ranks first on meaning and is four bricks short is reported as four
bricks short, at the top of the list, and it is not selected.  The selected
work is the highest-ranked one that actually passes -- or none, said plainly.

**The explanation is not written by a model.**  It is assembled from the
evidence fields, so every sentence in it corresponds to a number a reader can
check, and there is no path by which it can describe a completion rate the
calculation did not produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.delivery.pipeline import CatalogItem, TrainCatalog, inventory_evidence
from src.retrieval.index import Hit, IndexError_, VectorIndex
from src.retrieval.nlp import Conditions, filter_hits, unapplied_conditions

RETRIEVAL_KIND = "multilingual sentence embedding, exact cosine"


class SearchError(ValueError):
    """The search cannot be run, or its inputs do not agree."""


@dataclass(frozen=True)
class Candidate:
    """One retrieved work with the evidence that decides whether it is usable."""

    hit: Hit
    item: CatalogItem = field(repr=False)
    missing: dict[str, int]
    missing_total: int
    completion: float
    required_total: int

    @property
    def touches_ground(self) -> bool:
        return self.item.touches_ground

    @property
    def connected(self) -> bool:
        return self.item.connected

    @property
    def static_ready(self) -> bool:
        return self.touches_ground and self.connected

    @property
    def buildable(self) -> bool:
        """Enough parts *and* the static structural conditions.

        Both halves, always.  A work with every part present that does not
        touch the ground is not something a person can build as described, and
        calling it buildable because the stock matched would be the exact
        overclaim this project refuses.
        """
        return not self.missing and self.static_ready

    def as_dict(self, *, rerank_rank: int | None = None) -> dict:
        """The candidate's evidence.

        ``semantic_rank`` is the embedding search's position and is carried
        from the hit unchanged.  ``rerank_rank`` is the position after the
        inventory and static-structure sort and is only present when the
        caller is listing the re-ranked order -- the two orderings are
        different facts, and a reader has to be able to see both and tell
        which is which.
        """
        body = {
            "semantic_rank": self.hit.semantic_rank,
            "catalog_id": self.item.catalog_id,
            "caption": self.item.caption,
            "semantic_score": round(self.hit.score, 6),
            "retrieval_kind": RETRIEVAL_KIND,
            "n_bricks": self.item.n_bricks,
            "required_inventory": dict(self.item.required),
            "missing_parts": dict(self.missing),
            "missing_total": self.missing_total,
            "inventory_completion": round(self.completion, 6),
            "inventory_sufficient": not self.missing,
            "touches_ground": self.touches_ground,
            "stud_only_connected": self.connected,
            "static_structure_ready": self.static_ready,
            "fully_buildable": self.buildable,
            "rank_note": (
                "semantic_rank is the embedding order; rerank_rank, where "
                "present, is the order after the exact inventory calculation "
                "and the static structural checks. Neither is derived from "
                "the other"),
        }
        if rerank_rank is not None:
            body["rerank_rank"] = int(rerank_rank)
        return body


@dataclass(frozen=True)
class SearchResult:
    """Everything one request produced, including why nothing was selected."""

    conditions: Conditions
    retrieved: tuple[Candidate, ...]
    ranked: tuple[Candidate, ...]
    selected: Candidate | None
    rejected_by_conditions: tuple[dict, ...]
    excluded_same_object: int
    index_documents: int

    @property
    def status(self) -> str:
        if self.selected is not None:
            return "buildable_existing_work_found"
        if not self.retrieved:
            return "no_semantic_candidate"
        return "no_buildable_existing_work_in_retrieved_set"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "retrieval": RETRIEVAL_KIND,
            "conditions": self.conditions.as_dict(),
            "conditions_not_applied_to_retrieval":
                unapplied_conditions(self.conditions),
            "index_documents": self.index_documents,
            "excluded_same_object_count": self.excluded_same_object,
            "rejected_by_conditions": list(self.rejected_by_conditions),
            "semantic_order": [c.as_dict() for c in self.retrieved],
            "inventory_reranked": [
                c.as_dict(rerank_rank=position)
                for position, c in enumerate(self.ranked, 1)],
            "selected_catalog_id": (self.selected.item.catalog_id
                                    if self.selected else None),
            "boundary": (
                "train split only; a semantic rank is not a promise that a "
                "work can be built. Nothing here is a retrieval quality "
                "measurement"),
        }


def _candidate(hit: Hit, item: CatalogItem,
               inventory: dict[str, int]) -> Candidate:
    evidence = inventory_evidence(item.required, inventory)
    return Candidate(
        hit=hit, item=item, missing=evidence["missing"],
        missing_total=evidence["missing_total"],
        completion=evidence["completion"],
        required_total=evidence["required_total"])


def search(index: VectorIndex, catalog: TrainCatalog, embedder,
           conditions: Conditions, inventory: dict[str, int], *,
           top_n: int = 10, exclude_object_id: str | None = None
           ) -> SearchResult:
    """Run one request end to end and return the evidence for every candidate.

    The index and the catalogue have to be the same catalogue, built under the
    same frozen split, with the same object ids.  All three are checked here
    rather than trusted: an index built from an older file would return
    ``catalog_id`` values this catalogue does not have, an index built under a
    different split manifest would carry works this one excludes, and an index
    whose object ids disagree would apply the same-object exclusion to the
    wrong work while every digest still agreed.
    """
    from src.retrieval.embed import KIND_QUERY, embed

    if index.catalog_sha256 != catalog.sha256:
        raise SearchError(
            f"the index was built from catalogue {index.catalog_sha256} and "
            f"the loaded catalogue is {catalog.sha256}; they are not the same "
            "data and the search is refused")
    if index.split_manifest_sha256 != catalog.split_manifest_sha256:
        raise SearchError(
            "the index was built against frozen split manifest "
            f"{index.split_manifest_sha256} and the catalogue was loaded "
            f"against {catalog.split_manifest_sha256}; the train-only "
            "boundary is not the same on both sides and the search is "
            "refused")
    try:
        index.check_against_catalog(catalog)
    except IndexError_ as exc:
        raise SearchError(str(exc)) from None
    if index.identity_digest != embedder.identity_digest():
        raise SearchError(
            "the index and this embedder disagree about the model, revision, "
            "prefixes or pooling; the vectors are not comparable and the "
            "search is refused")
    if not inventory:
        raise SearchError("an inventory is required")

    by_id = {item.catalog_id: item for item in catalog.items}
    vector = embed(embedder, [conditions.text], kind=KIND_QUERY)[0]
    # Over-fetch so the metadata conditions have something to filter without
    # silently shrinking the candidate list the caller asked for.
    hits = index.search(vector, top_n=max(top_n * 4, top_n),
                        exclude_object_id=exclude_object_id)
    kept, rejected = filter_hits(hits, conditions)

    retrieved: list[Candidate] = []
    for hit in kept[:top_n]:
        item = by_id.get(hit.document.catalog_id)
        if item is None:
            raise SearchError(
                f"the index returned catalog_id {hit.document.catalog_id!r}, "
                "which is not in the loaded catalogue")
        retrieved.append(_candidate(hit, item, inventory))

    ranked = tuple(sorted(
        retrieved,
        key=lambda c: (not c.buildable, c.missing_total, -c.completion,
                       -c.hit.score, c.item.catalog_id)))
    selected = next((c for c in ranked if c.buildable), None)
    excluded = (len(index.documents) - len(
        [d for d in index.documents
         if exclude_object_id is None or d.object_id != exclude_object_id]))
    return SearchResult(
        conditions=conditions, retrieved=tuple(retrieved), ranked=ranked,
        selected=selected,
        rejected_by_conditions=tuple(
            {"catalog_id": hit.document.catalog_id, "reason": reason}
            for hit, reason in rejected),
        excluded_same_object=excluded,
        index_documents=len(index.documents))
