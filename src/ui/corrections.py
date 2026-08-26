"""Operator corrections to a detection, with three values kept apart.

The requirement this module exists for is that a corrected inventory must never
be mistaken for a measured one.  So every item carries three things:

``predicted``
    What the model said, unchanged, for ever.  A correction never overwrites
    it.
``edited``
    What the operator changed, if anything, field by field.
``adopted``
    The value actually used downstream.

A page, a report and the inventory engine all read ``adopted``; the other two
are what make the number auditable afterwards.  An inventory built from
corrected detections says how many of its items a person changed, and that
count travels with it.

**There is no bypass.**  Every adopted part label goes through
:func:`src.vision.classes.normalise_part`, so ``4x1`` and ``1x4`` are one item
here exactly as they are everywhere else, and the stock dictionary handed on is
built by the same rotation-normalising path the command line uses.  A
correction cannot introduce a part outside the eight, a negative count, or a
colour outside the palette.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from src.colour.palette import ColourError, colour
from src.vision.classes import UNKNOWN, ClassError, normalise_part
from src.vision.schema import Prediction

#: The largest count one corrected item may carry.  The world holds 8,000
#: cells, so a single part in a greater quantity cannot change any outcome --
#: the same bound the numeric form fields use, for the same reason.
MAX_COUNT = 20 ** 3

#: The most items one photograph's correction list may hold.  Above the
#: archive's most crowded image by a wide margin, and low enough that a
#: runaway form cannot build an unbounded list.
MAX_ITEMS = 128

SOURCE_MODEL = "model"
SOURCE_OPERATOR = "operator"
SOURCE_MIXED = "model+operator"


class CorrectionError(ValueError):
    """A correction the interface refuses, with a reason a reader can act on."""


@dataclass(frozen=True)
class DetectedItem:
    """One detection and whatever a person did to it.

    ``box`` is ``(x0, y0, x1, y1)`` with exclusive far edges, in the
    coordinates of the image as uploaded.  ``predicted_part`` may be
    :data:`~src.vision.classes.UNKNOWN`: that is what the classifier reports
    when it declines, and it is exactly the case an operator is here to
    resolve.
    """

    index: int
    box: tuple[int, int, int, int]
    predicted_part: str
    predicted_confidence: float
    predicted_top3: tuple[str, ...] = ()
    predicted_colour: str | None = None
    predicted_colour_confidence: float | None = None
    edited_part: str | None = None
    edited_count: int | None = None
    edited_colour: str | None = None
    edited_box: tuple[int, int, int, int] | None = None
    added_by_operator: bool = False
    deleted: bool = False

    # -- the adopted values ----------------------------------------------
    @property
    def adopted_part(self) -> str:
        return self.edited_part if self.edited_part is not None \
            else self.predicted_part

    @property
    def adopted_count(self) -> int:
        return self.edited_count if self.edited_count is not None else 1

    @property
    def adopted_colour(self) -> str | None:
        return self.edited_colour if self.edited_colour is not None \
            else self.predicted_colour

    @property
    def adopted_box(self) -> tuple[int, int, int, int]:
        return self.edited_box if self.edited_box is not None else self.box

    @property
    def changed_fields(self) -> tuple[str, ...]:
        out = []
        if self.edited_part is not None and \
                self.edited_part != self.predicted_part:
            out.append("part")
        if self.edited_count is not None and self.edited_count != 1:
            out.append("count")
        if self.edited_colour is not None and \
                self.edited_colour != self.predicted_colour:
            out.append("colour")
        if self.edited_box is not None and self.edited_box != self.box:
            out.append("box")
        return tuple(out)

    @property
    def source(self) -> str:
        if self.added_by_operator:
            return SOURCE_OPERATOR
        return SOURCE_MIXED if self.changed_fields else SOURCE_MODEL

    @property
    def counts_towards_stock(self) -> bool:
        """Whether this item contributes to the adopted inventory.

        A deleted item does not, and neither does one still labelled
        ``unknown``: an unidentified box is not stock until a person says what
        it is.
        """
        return not self.deleted and self.adopted_part != UNKNOWN

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "source": self.source,
            "deleted": self.deleted,
            "added_by_operator": self.added_by_operator,
            "changed_fields": list(self.changed_fields),
            "predicted": {
                "part": self.predicted_part,
                "confidence": round(float(self.predicted_confidence), 6),
                "top3": list(self.predicted_top3),
                "colour": self.predicted_colour,
                "colour_confidence": (
                    None if self.predicted_colour_confidence is None
                    else round(float(self.predicted_colour_confidence), 6)),
                "box": list(self.box),
            },
            "edited": {
                "part": self.edited_part,
                "count": self.edited_count,
                "colour": self.edited_colour,
                "box": list(self.edited_box) if self.edited_box else None,
            },
            "adopted": {
                "part": self.adopted_part,
                "count": self.adopted_count,
                "colour": self.adopted_colour,
                "box": list(self.adopted_box),
                "counts_towards_stock": self.counts_towards_stock,
            },
        }


@dataclass(frozen=True)
class CorrectedInventory:
    """The adopted stock, and how much of it a person decided."""

    items: tuple[DetectedItem, ...]
    parts: dict[str, int]
    colour_parts: dict[tuple[str, str], int]
    unresolved: tuple[int, ...]

    @property
    def total(self) -> int:
        return sum(self.parts.values())

    @property
    def edited_items(self) -> int:
        return sum(1 for item in self.items if item.changed_fields
                   or item.added_by_operator or item.deleted)

    @property
    def fully_coloured(self) -> bool:
        """Whether every counted item carries a colour.

        The colour assigner needs a ``(part, colour)`` stock, so a partly
        coloured correction cannot be used for assignment -- and that is
        reported rather than silently filled with a default.
        """
        return bool(self.parts) and sum(self.colour_parts.values()) == \
            self.total

    def as_dict(self) -> dict:
        return {
            "parts": dict(self.parts),
            "total": self.total,
            "colour_parts": {f"{part}:{name}": count for (part, name), count
                             in sorted(self.colour_parts.items())},
            "fully_coloured": self.fully_coloured,
            "items": [item.as_dict() for item in self.items],
            "counted_items": sum(1 for item in self.items
                                 if item.counts_towards_stock),
            "deleted_items": sum(1 for item in self.items if item.deleted),
            "operator_added_items": sum(1 for item in self.items
                                        if item.added_by_operator),
            "edited_items": self.edited_items,
            "unresolved_items": list(self.unresolved),
            "provenance": (
                "every item keeps the model's prediction, the operator's edit "
                "and the adopted value separately. The stock below is the "
                "adopted values only, and an item still labelled unknown is "
                "not counted into any part"),
        }


def items_from_detection(result, *, colours=None) -> tuple[DetectedItem, ...]:
    """Turn a :class:`~src.vision.detect.DetectionResult` into editable items.

    ``colours`` is an optional per-detection
    :class:`~src.colour.recognise.ColourReading`; a low-confidence reading is
    carried as the prediction with its confidence, so the page can put it in
    front of a person rather than adopting it.
    """
    out = []
    for index, detection in enumerate(result.detections):
        reading = (colours or {}).get(index)
        out.append(DetectedItem(
            index=index,
            box=tuple(int(v) for v in detection.box),
            predicted_part=detection.label,
            predicted_confidence=detection.confidence,
            predicted_top3=tuple(detection.prediction.top_k(3)),
            predicted_colour=(reading.label if reading is not None else None),
            predicted_colour_confidence=(reading.confidence
                                         if reading is not None else None)))
    return tuple(out)


def _box(raw, *, width: int, height: int, name: str
         ) -> tuple[int, int, int, int]:
    values = list(raw)
    if len(values) != 4:
        raise CorrectionError(f"{name} 必須是四個數字 (x0, y0, x1, y1)")
    try:
        x0, y0, x1, y1 = (int(value) for value in values)
    except (TypeError, ValueError):
        raise CorrectionError(f"{name} 只接受整數座標") from None
    if x0 < 0 or y0 < 0:
        raise CorrectionError(f"{name} 的座標不可為負")
    if x1 > width or y1 > height:
        raise CorrectionError(
            f"{name} 超出圖片範圍（圖片是 {width}×{height}）")
    if x1 <= x0 or y1 <= y0:
        raise CorrectionError(f"{name} 沒有面積：右下角必須大於左上角")
    return (x0, y0, x1, y1)


def apply_edits(items, edits, *, width: int, height: int
                ) -> tuple[DetectedItem, ...]:
    """Apply a mapping of ``index -> {field: value}`` and return new items.

    Nothing is mutated: the model's prediction is preserved by construction
    because an edit produces a new record with the ``edited_*`` fields set and
    the ``predicted_*`` fields copied.
    """
    if len(items) > MAX_ITEMS:
        raise CorrectionError(
            f"這張照片有 {len(items)} 個項目，超過上限 {MAX_ITEMS}")
    by_index = {item.index: item for item in items}
    out = dict(by_index)
    for index, changes in (edits or {}).items():
        if index not in by_index:
            raise CorrectionError(f"沒有編號 {index} 的偵測項目")
        item = out[index]
        updates: dict = {}
        if "delete" in changes:
            updates["deleted"] = bool(changes["delete"])
        if changes.get("part") is not None:
            raw = str(changes["part"]).strip()
            if raw == UNKNOWN:
                updates["edited_part"] = UNKNOWN
            else:
                try:
                    updates["edited_part"] = normalise_part(raw)
                except ClassError as exc:
                    raise CorrectionError(f"項目 {index} 的零件：{exc}") from None
        if changes.get("count") is not None:
            try:
                count = int(changes["count"])
            except (TypeError, ValueError):
                raise CorrectionError(
                    f"項目 {index} 的數量必須是整數") from None
            if count < 0:
                raise CorrectionError(f"項目 {index} 的數量不可為負")
            if count > MAX_COUNT:
                raise CorrectionError(
                    f"項目 {index} 的數量 {count} 超過上限 {MAX_COUNT}")
            updates["edited_count"] = count
        if changes.get("colour") is not None:
            raw = str(changes["colour"]).strip()
            if raw:
                try:
                    updates["edited_colour"] = colour(raw).colour_id
                except ColourError as exc:
                    raise CorrectionError(
                        f"項目 {index} 的顏色：{exc}") from None
        if changes.get("box") is not None:
            updates["edited_box"] = _box(changes["box"], width=width,
                                         height=height,
                                         name=f"項目 {index} 的框")
        out[index] = replace(item, **updates)
    return tuple(out[index] for index in sorted(out))


def add_item(items, *, part: str, count: int = 1, colour_id: str | None = None,
             box=None, width: int = 0, height: int = 0
             ) -> tuple[DetectedItem, ...]:
    """Append an item the operator added, with no model prediction behind it.

    Its ``predicted_part`` is ``unknown`` and its confidence is zero, which is
    the truthful record: no model said anything about this one.
    """
    if len(items) >= MAX_ITEMS:
        raise CorrectionError(
            f"已經有 {len(items)} 個項目，超過上限 {MAX_ITEMS}，無法再新增")
    try:
        canonical = normalise_part(part)
    except ClassError as exc:
        raise CorrectionError(f"新增項目的零件：{exc}") from None
    if count < 1:
        raise CorrectionError("新增項目的數量至少是 1")
    if count > MAX_COUNT:
        raise CorrectionError(f"新增項目的數量超過上限 {MAX_COUNT}")
    resolved = None
    if colour_id:
        try:
            resolved = colour(colour_id).colour_id
        except ColourError as exc:
            raise CorrectionError(f"新增項目的顏色：{exc}") from None
    placed = ((0, 0, 1, 1) if box is None
              else _box(box, width=width, height=height, name="新增項目的框"))
    index = (max((item.index for item in items), default=-1) + 1)
    return tuple(items) + (DetectedItem(
        index=index, box=placed, predicted_part=UNKNOWN,
        predicted_confidence=0.0, predicted_top3=(),
        edited_part=canonical, edited_count=count, edited_colour=resolved,
        added_by_operator=True),)


def adopt(items) -> CorrectedInventory:
    """Build the adopted stock from a list of corrected items.

    The part totals go through the same normalisation the rest of the project
    uses, and the ``(part, colour)`` totals are only built for items that
    actually carry a colour -- a partly coloured correction produces a partial
    colour stock and :attr:`CorrectedInventory.fully_coloured` says so, rather
    than a default colour being substituted for the ones nobody chose.
    """
    parts: dict[str, int] = {}
    coloured: dict[tuple[str, str], int] = {}
    unresolved: list[int] = []
    for item in items:
        if item.deleted:
            continue
        if item.adopted_part == UNKNOWN:
            unresolved.append(item.index)
            continue
        part = normalise_part(item.adopted_part)
        count = item.adopted_count
        if count <= 0:
            continue
        parts[part] = parts.get(part, 0) + count
        if item.adopted_colour:
            key = (part, colour(item.adopted_colour).colour_id)
            coloured[key] = coloured.get(key, 0) + count
    return CorrectedInventory(
        items=tuple(items), parts=dict(sorted(parts.items())),
        colour_parts=dict(sorted(coloured.items())),
        unresolved=tuple(unresolved))


def inventory_spec(stock: dict[str, int]) -> str:
    """``{"2x4": 3}`` -> ``"2x4:3"`` in vocabulary order.

    The string form, because that is what
    :func:`src.demo.showcase.parse_inventory` takes and that function is where
    every inventory refusal in this project already lives.  Building the
    string keeps a corrected photograph on exactly the same validated path as
    a typed inventory instead of beside it.
    """
    from src.data.bricks import PART_VOCAB

    return ",".join(f"{part}:{stock[part]}" for part in PART_VOCAB
                    if stock.get(part, 0) > 0)


def colour_stock_spec(stock: dict[tuple[str, str], int]) -> str:
    """``{("2x4", "red"): 3}`` -> ``"2x4:red:3"``, for the colour assigner."""
    from src.data.bricks import PART_VOCAB

    ordered = sorted(stock.items(),
                     key=lambda pair: (PART_VOCAB.index(pair[0][0]),
                                       pair[0][1]))
    return ",".join(f"{part}:{name}:{count}"
                    for (part, name), count in ordered if count > 0)
