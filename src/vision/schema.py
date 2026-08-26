"""One prediction shape, whichever method produced it.

The traditional CV baseline and the learned classifier are different in every
respect except what they have to hand back.  Giving them one return type is
what makes the comparison in ``scripts/33_vision_eval.py`` a comparison rather
than two reports side by side, and it is what lets the UI show a correction
form that does not care which method ran.

Three decisions live here because they must not be made twice:

* **Low confidence is a state, not a rendering.**  A prediction below
  :data:`LOW_CONFIDENCE` reports ``label`` as
  :data:`~src.vision.classes.UNKNOWN` and ``low_confidence`` as true, in the
  data.  A consumer that never reads the flag still cannot mistake a coin flip
  for an answer.
* **Top-3 is the same ordering as top-1.**  Ties break on class order, so two
  runs on the same pixels produce the same list.
* **A score is a score, not a probability of being right.**  The field is
  named ``score`` and the method is recorded beside it; the CV baseline's
  number and a softmax are not the same quantity and are never averaged.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.vision.classes import CLASS_ORDER, UNKNOWN, ClassError, normalise_part

#: Below this the top-1 answer is not offered as an answer.  One threshold for
#: both methods: a per-method threshold tuned on the same data the methods are
#: compared on would make the comparison about the thresholds.
LOW_CONFIDENCE = 0.45

#: How many candidates a prediction carries.  Three is what the correction UI
#: shows, and what Top-3 accuracy is computed over.
TOP_K = 3

METHOD_CV = "cv-baseline"
METHOD_LEARNED = "transfer-resnet18"
METHODS = (METHOD_CV, METHOD_LEARNED)


@dataclass(frozen=True)
class Candidate:
    """One class and the score the method gave it."""

    part: str
    score: float

    def as_dict(self) -> dict:
        return {"part": self.part, "score": round(float(self.score), 6)}


@dataclass(frozen=True)
class Prediction:
    """What every classifier in this project returns.

    ``label`` is the answer to act on.  It is the top candidate's part when
    the method was confident enough, and :data:`~src.vision.classes.UNKNOWN`
    when it was not -- so a caller that reads only ``label`` cannot use a
    guess as though it were a decision.
    """

    method: str
    candidates: tuple[Candidate, ...]
    features: dict = None                       # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ClassError(
                f"method={self.method!r} is not one of {list(METHODS)}")
        if not self.candidates:
            raise ClassError("a prediction needs at least one candidate")
        seen = set()
        for candidate in self.candidates:
            part = normalise_part(candidate.part)
            if part in seen:
                raise ClassError(
                    f"{part} appears twice in one prediction; a candidate "
                    "list with a duplicate would double-count it in Top-3")
            seen.add(part)
        scores = [c.score for c in self.candidates]
        if any(s != s or s < 0 for s in scores):          # s != s catches NaN
            raise ClassError("a candidate score must be a non-negative number")
        if scores != sorted(scores, reverse=True):
            raise ClassError(
                "candidates must be ordered by descending score; an unsorted "
                "list makes Top-1 and Top-3 disagree about what came first")
        object.__setattr__(self, "features", dict(self.features or {}))

    @property
    def top(self) -> Candidate:
        return self.candidates[0]

    @property
    def confidence(self) -> float:
        return float(self.candidates[0].score)

    @property
    def margin(self) -> float:
        """Top-1 minus top-2.  Zero when only one candidate was produced."""
        if len(self.candidates) < 2:
            return float(self.candidates[0].score)
        return float(self.candidates[0].score - self.candidates[1].score)

    @property
    def low_confidence(self) -> bool:
        return self.confidence < LOW_CONFIDENCE

    @property
    def label(self) -> str:
        """The answer to act on, or ``unknown`` when there is not one."""
        return UNKNOWN if self.low_confidence else self.candidates[0].part

    def top_k(self, k: int = TOP_K) -> tuple[str, ...]:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ClassError("k must be a positive whole number")
        return tuple(c.part for c in self.candidates[:k])

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "label": self.label,
            "top1": self.candidates[0].part,
            "confidence": round(self.confidence, 6),
            "margin": round(self.margin, 6),
            "low_confidence": self.low_confidence,
            "top3": list(self.top_k()),
            "candidates": [c.as_dict() for c in self.candidates],
            "features": dict(self.features or {}),
        }


def from_scores(method: str, scores, *, features: dict | None = None,
                keep: int = TOP_K) -> Prediction:
    """Build a prediction from one score per class, in class order.

    ``scores`` must be exactly :data:`~src.vision.classes.CLASS_ORDER` long.
    Passing a dict is refused rather than filled with zeros: a missing class
    is a wiring mistake, and quietly scoring it zero would hide it.
    """
    values = [float(v) for v in scores]
    if len(values) != len(CLASS_ORDER):
        raise ClassError(
            f"{len(values)} scores for {len(CLASS_ORDER)} classes; the vector "
            "must be in class order and complete")
    if any(v != v for v in values):
        raise ClassError("a score vector may not contain NaN")
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ClassError("keep must be a positive whole number")
    kept = order[:min(keep, len(order))]
    return Prediction(
        method=method,
        candidates=tuple(Candidate(CLASS_ORDER[i], values[i]) for i in kept),
        features=features)


def normalise_scores(raw) -> list[float]:
    """Scale a non-negative score vector so it sums to one.

    Used by the CV baseline, whose numbers are distances turned into
    similarities and have no natural scale.  An all-zero vector comes back
    uniform rather than dividing by zero -- which is the honest answer: the
    features said nothing, so every class is equally unsupported and the
    result lands below :data:`LOW_CONFIDENCE`.
    """
    values = [max(0.0, float(v)) for v in raw]
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [v / total for v in values]
