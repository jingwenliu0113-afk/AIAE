"""One validated request out of the full form, and one dispatch into a method.

The two-page interface validated its form in :mod:`src.ui.app`, and every
numeric rule it settled on -- half-width digits only, a nine-digit length cap,
a per-part ceiling of the world's own cell count, decimals that cannot spell
``nan`` -- is reused here by calling those functions rather than by writing
them again.  A second copy of "what a number may look like" is how one entry
point ends up accepting what another refuses.

What this module adds is the part the full form has and the two-page one did
not: three methods instead of two, a colour stock, a build-step limit, an
optional placement gate, and an inventory that may have come from a corrected
photograph instead of a keyboard.

**An inapplicable field is refused by name.**  A CP-SAT time limit submitted
against the retrieval method, a placement gate against anything but the
project model -- each is named and refused.  A form that silently discards a
setting hands back a result page that looks like it honoured one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.demo.showcase import ShowcaseError, parse_inventory
from src.ui.app import (DEFAULT_SEED, DEFAULT_TIME_LIMIT,
                        MAX_CAPTION_CHARS, MAX_INVENTORY_SPEC_CHARS,
                        MAX_PART_COUNT, MAX_TIME_LIMIT, MAX_TOP_N, UiError,
                        _check_spec_counts, _decimal, _one, _short, _whole,
                        inventory_spec_from_grid)
from src.ui.full import (FULL_METHODS, METHOD_LABELS, METHOD_PIPELINE,
                         METHOD_PROJECT, METHOD_RAG)
from src.data.bricks import PART_VOCAB

#: The most bricks one assembly step may add.  A step a person cannot follow
#: is not a step; the planner accepts any positive limit and this is what the
#: form will take.
MAX_PER_STEP = 8

#: Bricks one demonstration decode may place.  Small: a person is waiting.
MAX_DECODE_BRICKS = 40

#: Fields that belong to CP-SAT and therefore only to the F-pipeline.
PIPELINE_ONLY = ("time_limit",)

#: Fields that belong to the decoder and therefore only to the project model.
PROJECT_ONLY = ("placement",)


class RequestError(UiError):
    """A submitted form the full interface refuses, with a reason."""


@dataclass(frozen=True)
class FullRequest:
    """One validated submission of the full form."""

    method: str
    caption: str
    inventory: dict[str, int]
    inventory_spec: str
    inventory_origin: str
    grid: dict[str, int]
    colour_stock: str
    preferences: tuple[str, ...]
    top_n: int
    time_limit: float | None
    seed: int
    max_per_step: int
    placement: bool
    photo_handle: str | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"method": self.method, "caption": self.caption,
                "inventory": dict(self.inventory),
                "inventory_spec": self.inventory_spec,
                "inventory_origin": self.inventory_origin,
                "colour_stock": self.colour_stock,
                "preferences": list(self.preferences),
                "top_n": self.top_n, "time_limit": self.time_limit,
                "seed": self.seed, "max_per_step": self.max_per_step,
                "placement": self.placement}


def build_request(fields, *, server) -> FullRequest:
    """Validate the full form.  Raises :class:`RequestError` on refusal."""
    method = _one(fields, "method").strip() or METHOD_RAG
    if method not in FULL_METHODS:
        raise RequestError(f"方法 {_short(method)} 不在 {list(FULL_METHODS)} 之內")

    caption = _one(fields, "caption").strip()
    if not caption:
        raise RequestError("文字需求不可空白：請描述要組的作品")
    if len(caption) > MAX_CAPTION_CHARS:
        raise RequestError(
            f"文字需求長度 {len(caption)} 超過上限 {MAX_CAPTION_CHARS} 字")

    grid: dict[str, int] = {}
    for part in PART_VOCAB:
        raw = _one(fields, f"qty_{part}").strip()
        if not raw:
            continue
        count = _whole(raw, f"{part} 的數量", minimum=0, maximum=MAX_PART_COUNT)
        if count:
            grid[part] = count

    advanced = _one(fields, "inventory_spec").strip()
    if len(advanced) > MAX_INVENTORY_SPEC_CHARS:
        raise RequestError(
            f"庫存字串長度 {len(advanced)} 超過上限 "
            f"{MAX_INVENTORY_SPEC_CHARS} 字")
    photo_handle = _one(fields, "photo_handle").strip() or None
    if advanced and grid:
        raise RequestError(
            "庫存字串與八格數量同時填寫；兩者是同一份庫存的兩種輸入方式，"
            "請只用其中一種，本介面不自行猜測該相加或覆蓋")
    if not advanced and not grid:
        raise RequestError(
            "庫存不可空白：請在八種零件中至少填一個數量，"
            "或改用庫存字串（例如 2x4:10,1x2:8），"
            "或先上傳照片並修正辨識結果")

    spec = advanced or inventory_spec_from_grid(grid)
    if advanced:
        _check_spec_counts(advanced)
    try:
        inventory = parse_inventory(spec)
    except ShowcaseError as exc:
        raise RequestError(str(exc)) from None
    origin = ("修正後的照片辨識結果" if photo_handle
              else ("庫存字串" if advanced else "八格手動輸入"))

    colour_stock = _one(fields, "colour_stock").strip()
    if len(colour_stock) > MAX_INVENTORY_SPEC_CHARS:
        raise RequestError(
            f"顏色庫存字串長度 {len(colour_stock)} 超過上限 "
            f"{MAX_INVENTORY_SPEC_CHARS} 字")
    preferences = _preferences_from(caption)

    top_n = _whole(_one(fields, "top_n") or "10", "Top-N", minimum=1,
                   maximum=MAX_TOP_N)
    max_per_step = _whole(_one(fields, "max_per_step") or "1",
                          "組裝每步最多幾顆", minimum=1, maximum=MAX_PER_STEP)
    raw_seed = _one(fields, "seed").strip()
    seed = (_whole(raw_seed, "seed", minimum=0, maximum=2 ** 31 - 1)
            if raw_seed else DEFAULT_SEED)

    placement = bool(_one(fields, "placement").strip())
    raw_limit = _one(fields, "time_limit").strip()
    time_limit = None

    if method == METHOD_PIPELINE:
        time_limit = (_decimal(raw_limit, "time limit",
                               maximum=MAX_TIME_LIMIT)
                      if raw_limit else DEFAULT_TIME_LIMIT)
        if placement:
            raise RequestError(
                "placement gate 屬於解碼器，不適用於「"
                f"{METHOD_LABELS[METHOD_PIPELINE]}」；請取消勾選或改選 "
                f"{METHOD_LABELS[METHOD_PROJECT]}。本介面不靜默忽略它")
    elif method == METHOD_PROJECT:
        if raw_limit:
            raise RequestError(
                "time limit 設定 CP-SAT，不適用於「"
                f"{METHOD_LABELS[METHOD_PROJECT]}」；請清空該欄位。"
                "本介面不靜默忽略它")
        if not server.allow_project_model:
            raise RequestError(
                "這次啟動沒有開放正式模型入口，因此不會載入任何權重。")
    else:
        if raw_limit:
            raise RequestError(
                "time limit 設定 CP-SAT，不適用於「"
                f"{METHOD_LABELS[METHOD_RAG]}」；請清空該欄位。"
                "本介面不靜默忽略它")
        if placement:
            raise RequestError(
                "placement gate 屬於解碼器，不適用於「"
                f"{METHOD_LABELS[METHOD_RAG]}」；請取消勾選。"
                "本介面不靜默忽略它")
        if raw_seed:
            raise RequestError(
                "檢索是確定性的，seed 對「"
                f"{METHOD_LABELS[METHOD_RAG]}」沒有作用；請清空該欄位。"
                "本介面不靜默忽略它")

    return FullRequest(
        method=method, caption=caption, inventory=inventory,
        inventory_spec=spec, inventory_origin=origin, grid=dict(grid),
        colour_stock=colour_stock, preferences=preferences, top_n=top_n,
        time_limit=time_limit, seed=seed, max_per_step=max_per_step,
        placement=placement, photo_handle=photo_handle)


def _preferences_from(caption: str) -> tuple[str, ...]:
    """Colour preferences, read out of the request itself.

    The request already states them -- "a blue car" -- so asking for them a
    second time in a separate field would be asking the operator to repeat
    themselves and would let the two disagree.
    """
    from src.retrieval.nlp import NlpError, extract

    try:
        return extract(caption).preferred_colours
    except NlpError:
        return ()


def execute(request: FullRequest, *, server):
    """Dispatch one request and finish it: colours, build order, outputs.

    Returns ``(result, finished_or_None, inventory_spec, inventory_origin)``.
    ``finished`` is ``None`` when there is no deliverable structure -- which is
    a normal outcome, not an error, and the page says so.
    """
    from src.ui import full as full_module

    if request.method == METHOD_RAG:
        if not server.index_dir:
            raise RequestError(
                "這次啟動沒有提供 RAG 索引目錄，因此無法做語意檢索。"
                "請先用 scripts/34_rag_index.py --build 建立索引，"
                "再以 --index 指定它。")
        result = full_module.run_rag(
            caption=request.caption, inventory=request.inventory,
            index_dir=server.index_dir, catalog_path=server.catalog,
            top_n=request.top_n, device="cpu")
    elif request.method == METHOD_PIPELINE:
        result = full_module.run_pipeline(
            caption=request.caption, inventory_spec_text=request.inventory_spec,
            catalog_path=server.catalog, top_n=request.top_n,
            time_limit=request.time_limit or DEFAULT_TIME_LIMIT,
            seed=request.seed)
    else:
        result = full_module.run_project_model(
            caption=request.caption, inventory=request.inventory,
            placement=request.placement,
            connectivity=("any" if request.placement else "off"),
            device=(server.device or "mps"), seed=request.seed,
            # A fixed cap, not the inventory total. Tying it to the stock made
            # the cap fire at exactly the moment the inventory ran out, so the
            # run stopped for the wrong recorded reason -- max_bricks instead
            # of inventory_exhausted, which is the one the scorer accepts. The
            # hard inventory gate is the guarantee; this is only a timeout.
            max_bricks=MAX_DECODE_BRICKS,
            root=server.project_root)

    finished = None
    if result.ready:
        finished = full_module.finish(
            result, colour_stock=request.colour_stock or None,
            preferences=request.preferences,
            max_per_step=request.max_per_step, title=request.caption)
    return result, finished, request.inventory_spec, request.inventory_origin
