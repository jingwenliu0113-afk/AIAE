"""Multilingual retrieval over the train-only catalogue, and its explanations.

Replaces the deterministic lexical baseline in :mod:`src.delivery.pipeline`
with a pinned multilingual sentence embedding, so a Chinese request can reach
an English caption on meaning rather than on shared characters. The lexical
baseline is kept, not deleted: it is what still runs when the embedding model
is absent, and the two are never reported as the same thing.

The boundaries are the ones the delivery path already had, enforced again
here because a new retrieval front end is exactly where they would be lost:

* **train split only.** The index is built from the same catalogue loader,
  which refuses a row that is not ``split=train`` and checks every row against
  the frozen object-level split manifest at its pinned digest.
* **no same-object retrieval for a held-out query.** The exclusion is a
  parameter of the search, not an afterthought applied to the results.
* **a recommendation is not a promise.** Semantic rank is followed by the
  exact inventory calculation and the static structural checks, and the
  explanation is generated from those numbers rather than from the model.
"""
