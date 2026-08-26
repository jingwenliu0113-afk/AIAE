"""Structured conditions out of a Chinese request, with the leftovers named.

A request like

    我想做一台 30 顆以內的藍色小車，顏色可以替換

carries five separable conditions -- a category, a brick budget, a colour
preference, whether substitution is allowed, and which mode to run -- and the
rest of the sentence is what the semantic search is for.  This module extracts
the five and hands the rest on unchanged.

**The rule that matters: nothing is guessed silently.**  A clause the extractor
recognises as a *condition* but cannot resolve is returned in
:attr:`Conditions.unresolved` with the text that caused it, and the caller has
to show it.  A number it cannot attach to anything, a colour word outside the
palette, a mode word it does not know -- each becomes a named item rather than
a default quietly substituted for what the person asked.  A retrieval that
ignored "30 顆以內" and returned a ninety-brick model would look like a
retrieval failure; saying "this condition was not applied" is the difference
between a wrong answer and an answer with a stated gap.

**It is a rule-based extractor and says so.**  No model runs.  It handles the
phrasings the project's own examples and tests use, in Traditional and
Simplified Chinese and in English, and it reports what it did not understand
instead of pretending to general language understanding.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from src.colour.palette import COLOUR_ORDER, BY_ID

MODE_EXISTING = "existing"
MODE_GENERATE = "generate"
MODE_PIPELINE = "f-pipeline"
MODE_EITHER = "existing_or_generate"
MODES = (MODE_EXISTING, MODE_GENERATE, MODE_PIPELINE, MODE_EITHER)

#: Categories the extractor recognises, with the words that select them.  A
#: small, stated list: a category the catalogue does not distinguish is not
#: worth pretending to extract.
CATEGORY_WORDS: dict[str, tuple[str, ...]] = {
    "vehicle": ("車", "汽車", "小車", "卡車", "貨車", "巴士", "火車",
                "車輛", "car", "truck", "bus", "train", "vehicle"),
    "building": ("房", "房子", "建築", "屋", "塔", "橋", "牆",
                 "house", "building", "tower", "bridge", "wall"),
    "animal": ("動物", "狗", "貓", "鳥", "馬", "魚",
               "animal", "dog", "cat", "bird", "horse", "fish"),
    "furniture": ("家具", "椅", "桌", "床", "櫃",
                  "chair", "table", "bed", "furniture"),
    "figure": ("人", "人物", "機器人", "figure", "robot", "person"),
    "plant": ("樹", "花", "植物", "tree", "flower", "plant"),
}

#: Colour words, mapped onto the palette ids.  Every palette colour is
#: reachable by its Chinese label; the extra spellings are the ones a person
#: actually types.
COLOUR_WORDS: dict[str, str] = {
    "紅": "red", "紅色": "red", "red": "red",
    "藍": "blue", "藍色": "blue", "blue": "blue",
    "綠": "green", "綠色": "green", "green": "green",
    "黃": "yellow", "黃色": "yellow", "yellow": "yellow",
    "黑": "black", "黑色": "black", "black": "black",
    "白": "white", "白色": "white", "white": "white",
    "橘": "orange", "橘色": "orange", "橙": "orange", "橙色": "orange",
    "orange": "orange",
    "粉紅": "pink", "粉色": "pink", "pink": "pink",
    "紫": "purple", "紫色": "purple", "purple": "purple",
    "棕": "brown", "棕色": "brown", "咖啡色": "brown", "brown": "brown",
    "紅棕": "reddish_brown", "reddish brown": "reddish_brown",
    "淺灰": "light_grey", "light grey": "light_grey",
    "深灰": "dark_grey", "dark grey": "dark_grey",
    "淺藍灰": "light_bluish_grey", "light bluish grey": "light_bluish_grey",
    "深藍灰": "dark_bluish_grey", "dark bluish grey": "dark_bluish_grey",
    "中藍": "medium_blue", "medium blue": "medium_blue",
    "亮綠": "bright_green", "bright green": "bright_green",
    "萊姆綠": "lime", "lime": "lime",
    "中萊姆綠": "medium_lime", "medium lime": "medium_lime",
    "淺沙": "tan", "沙色": "tan", "tan": "tan",
}

#: Words that mean "at most this many bricks".
_LIMIT = re.compile(
    r"(\d{1,4})\s*(?:顆|塊|片|个|個|pieces?|bricks?|parts?)?\s*"
    r"(?:以內|以内|以下|之內|之内|內|以里|最多|上限|max|maximum|"
    r"or fewer|or less|at most)")
_LIMIT_PREFIX = re.compile(
    r"(?:最多|不超過|不超过|上限|至多|no more than|at most|under|"
    r"fewer than|less than)\s*(\d{1,4})")

#: Any bare number with a piece word, used to notice a budget that was stated
#: without a limiting word so it can be reported rather than applied.
_BARE_COUNT = re.compile(r"(\d{1,4})\s*(?:顆|塊|片|个|個|pieces?|bricks?)")

_SUBSTITUTE_YES = ("可以替換", "可以替换", "可替換", "可替换", "顏色可以換",
                   "颜色可以换", "換色", "换色", "不限顏色", "不限颜色",
                   "颜色不限", "顏色不限", "substitute", "any colour",
                   "any color", "colour flexible", "color flexible")
_SUBSTITUTE_NO = ("不可替換", "不可替换", "不能換色", "不能换色",
                  "必須是", "必须是", "only", "strictly", "exact colour",
                  "exact color")

_MODE_WORDS: dict[str, tuple[str, ...]] = {
    MODE_EXISTING: ("既有", "現有", "现有", "找現成", "找现成", "推薦",
                    "推荐", "existing", "recommend", "search"),
    MODE_GENERATE: ("生成", "產生", "产生", "自己做", "新做", "generate",
                    "create new"),
    MODE_PIPELINE: ("重新鋪", "重新铺", "重鋪", "重铺", "re-tile", "retile",
                    "pipeline", "f-pipeline"),
}

#: A request longer than this is not read for conditions; the cap matches the
#: interface's own caption limit so the two agree.
MAX_REQUEST_CHARS = 2000


class NlpError(ValueError):
    """The request cannot be read at all."""


@dataclass(frozen=True)
class Unresolved:
    """One condition the extractor saw and could not apply."""

    field: str
    text: str
    reason: str

    def as_dict(self) -> dict:
        return {"field": self.field, "text": self.text, "reason": self.reason}


@dataclass(frozen=True)
class Conditions:
    """The structured request.  ``None`` means "not stated", never "default"."""

    text: str
    normalised: str
    category: str | None = None
    max_parts: int | None = None
    preferred_colours: tuple[str, ...] = ()
    allow_colour_substitution: bool | None = None
    mode: str | None = None
    unresolved: tuple[Unresolved, ...] = field(default_factory=tuple)

    @property
    def has_unresolved(self) -> bool:
        return bool(self.unresolved)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "category": self.category,
            "max_parts": self.max_parts,
            "preferred_colours": list(self.preferred_colours),
            "allow_colour_substitution": self.allow_colour_substitution,
            "mode": self.mode,
            "unresolved": [item.as_dict() for item in self.unresolved],
            "extractor": (
                "rule-based, not a language model. It handles the phrasings "
                "this project's examples and tests use and reports what it "
                "did not understand rather than defaulting silently"),
        }

    def describe_zh(self) -> list[str]:
        """One readable line per extracted condition, for the interface."""
        out = []
        if self.category:
            out.append(f"類別：{self.category}")
        if self.max_parts is not None:
            out.append(f"最大零件數：{self.max_parts}")
        if self.preferred_colours:
            names = "、".join(BY_ID[c].label_zh for c in self.preferred_colours)
            out.append(f"偏好顏色：{names}")
        if self.allow_colour_substitution is not None:
            out.append("允許替代顏色" if self.allow_colour_substitution
                       else "不允許替代顏色")
        if self.mode:
            out.append(f"模式：{self.mode}")
        if not out:
            out.append("沒有抽取到任何結構化條件；整段文字只用於語意檢索")
        return out


def normalise(text: str) -> str:
    """NFKC and case folding, so full-width digits and Latin match too."""
    return unicodedata.normalize("NFKC", text).casefold()


#: ``顏色`` is the word "colour" itself, not a colour.  Without this, every
#: sentence that mentions colour at all reports an unresolved colour.
_GENERIC_COLOUR_HEADS = ("顏", "颜", "配", "單", "单", "多", "雙", "双", "同")

#: How many characters before ``色`` may belong to a colour name (``淺藍灰色``).
_COLOUR_LOOKBACK = 3


def _colours(normal: str) -> tuple[tuple[str, ...], list[Unresolved]]:
    """Colour preferences, and colour-shaped words the palette does not have.

    Matching runs longest word first so ``淺藍灰`` is not read as ``藍``.  The
    unresolved pass then walks every ``色`` and asks whether *any* suffix
    ending there is a colour this project knows; only when none is does it
    report, and it skips the places where ``色`` is part of the word "colour"
    rather than the name of one.  Getting that wrong in the lenient direction
    would fill the page with phantom conditions; getting it wrong the other way
    would drop a real one, so both cases are tested.
    """
    found: list[str] = []
    matched_spans: list[tuple[int, int]] = []
    for word in sorted(COLOUR_WORDS, key=len, reverse=True):
        start = normal.find(word)
        if start < 0:
            continue
        end = start + len(word)
        if any(a <= start and end <= b for a, b in matched_spans):
            continue
        matched_spans.append((start, end))
        colour_id = COLOUR_WORDS[word]
        if colour_id not in found:
            found.append(colour_id)

    unresolved: list[Unresolved] = []
    seen_text: set[str] = set()
    for position, character in enumerate(normal):
        if character != "色":
            continue
        if position and normal[position - 1] in _GENERIC_COLOUR_HEADS:
            continue
        resolved = False
        for back in range(_COLOUR_LOOKBACK, 0, -1):
            start = position - back
            if start < 0:
                continue
            if (normal[start:position + 1] in COLOUR_WORDS
                    or normal[start:position] in COLOUR_WORDS):
                resolved = True
                break
        if resolved:
            continue
        phrase = normal[max(0, position - 1):position + 1]
        if phrase in seen_text:
            continue
        seen_text.add(phrase)
        unresolved.append(Unresolved(
            field="preferred_colours", text=phrase,
            reason=(f"{phrase} 不在本專案的 {len(COLOUR_ORDER)} 色調色盤內；"
                    "沒有替它挑一個最接近的顏色")))
    ordered = [c for c in COLOUR_ORDER if c in found]
    return tuple(ordered), unresolved


def _max_parts(normal: str) -> tuple[int | None, list[Unresolved]]:
    unresolved: list[Unresolved] = []
    for pattern in (_LIMIT, _LIMIT_PREFIX):
        match = pattern.search(normal)
        if match:
            value = int(match.group(1))
            if value < 1:
                unresolved.append(Unresolved(
                    field="max_parts", text=match.group(0),
                    reason="零件上限必須至少是 1"))
                return None, unresolved
            return value, unresolved
    bare = _BARE_COUNT.search(normal)
    if bare:
        unresolved.append(Unresolved(
            field="max_parts", text=bare.group(0),
            reason=(f"看到「{bare.group(0)}」但沒有「以內」「最多」這類限制詞，"
                    "無法判斷這是上限、下限還是剛好；因此沒有套用任何零件數條件")))
    return None, unresolved


def _category(normal: str) -> str | None:
    hits: list[tuple[int, str]] = []
    for name, words in CATEGORY_WORDS.items():
        for word in words:
            position = normal.find(normalise(word))
            if position >= 0:
                hits.append((position, name))
                break
    if not hits:
        return None
    hits.sort()
    return hits[0][1]


def _substitution(normal: str) -> bool | None:
    for phrase in _SUBSTITUTE_NO:
        if normalise(phrase) in normal:
            return False
    for phrase in _SUBSTITUTE_YES:
        if normalise(phrase) in normal:
            return True
    return None


def _mode(normal: str) -> tuple[str | None, list[Unresolved]]:
    hits = []
    for name, words in _MODE_WORDS.items():
        for word in words:
            if normalise(word) in normal:
                hits.append(name)
                break
    if not hits:
        return None, []
    if len(set(hits)) > 1:
        return None, [Unresolved(
            field="mode", text="、".join(sorted(set(hits))),
            reason=("同一句話裡出現多種方法的字眼，無法判斷要用哪一個；"
                    "請在介面上直接選擇方法"))]
    return hits[0], []


def extract(text: str) -> Conditions:
    """Read a request into conditions, naming whatever could not be applied."""
    if not isinstance(text, str):
        raise NlpError(f"a request must be a string, not {type(text).__name__}")
    stripped = text.strip()
    if not stripped:
        raise NlpError("the request is empty")
    if len(stripped) > MAX_REQUEST_CHARS:
        raise NlpError(
            f"the request is {len(stripped)} characters, over the "
            f"{MAX_REQUEST_CHARS} limit")
    normal = normalise(stripped)

    colours, colour_problems = _colours(normal)
    limit, limit_problems = _max_parts(normal)
    mode, mode_problems = _mode(normal)
    unresolved = tuple(colour_problems + limit_problems + mode_problems)
    return Conditions(
        text=stripped, normalised=normal, category=_category(normal),
        max_parts=limit, preferred_colours=colours,
        allow_colour_substitution=_substitution(normal), mode=mode,
        unresolved=unresolved)


def filter_hits(hits, conditions: Conditions):
    """Apply the extracted metadata conditions to a ranked hit list.

    Returns ``(kept, rejected)`` where each rejected entry carries the reason.
    Only the conditions that can be checked against catalogue metadata are
    applied here -- the brick budget.  Colour is not: the structure track is
    colourless, so a colour preference is an assignment preference and is
    passed to the colour assigner rather than used to reject a work.  Category
    is not either: the catalogue has no category field, and filtering on a
    category this module inferred from the request would be filtering on a
    guess.
    """
    kept, rejected = [], []
    for hit in hits:
        if (conditions.max_parts is not None
                and hit.document.n_bricks > conditions.max_parts):
            rejected.append((hit, (
                f"{hit.document.n_bricks} 塊超過要求的 "
                f"{conditions.max_parts} 塊上限")))
            continue
        kept.append(hit)
    return kept, rejected


def unapplied_conditions(conditions: Conditions) -> list[dict]:
    """Conditions that were understood but do not act on retrieval.

    Reported so a page never implies a condition changed the ranking when it
    did not.  This is the same honesty rule as ``unresolved``, for the opposite
    case: understood, and deliberately not applied here.
    """
    out = []
    if conditions.category:
        out.append({"field": "category", "value": conditions.category,
                    "reason": ("目錄沒有類別欄位；類別只作為語意檢索文字的一"
                               "部分，不用來過濾候選")})
    if conditions.preferred_colours:
        out.append({"field": "preferred_colours",
                    "value": list(conditions.preferred_colours),
                    "reason": ("結構軌是無色的；偏好顏色交給配色器，"
                               "不用來排除任何作品")})
    if conditions.allow_colour_substitution is not None:
        out.append({"field": "allow_colour_substitution",
                    "value": conditions.allow_colour_substitution,
                    "reason": ("配色器一律在庫存內指派，偏好顏色不足時會退到"
                               "其他顏色並如實回報，不因這個旗標改變檢索")})
    return out
