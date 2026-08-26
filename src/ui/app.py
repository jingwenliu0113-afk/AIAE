"""Request handling for the two-page UI, with no HTTP and no HTML in it.

Everything here is a plain function over plain data, so the whole UI can be
tested without a socket and without a click.  The HTTP layer in
:mod:`src.ui.server` does nothing but move bytes between a browser and these
functions.

Three boundaries are enforced here rather than described:

* **No second judgement.**  The verdict on whether a result may be delivered
  is read from ``report["delivery"]["static_delivery_ready"]``, which is
  computed by ``scripts/27_delivery.py``.  This module never recomputes it,
  and never assembles a report of its own.
* **No model.**  Nothing in this package imports a generation, training or
  weight-loading entry point, and no code path here can reach one.  The
  delivery payload records ``model_loaded: False`` and this module refuses to
  publish anything that says otherwise.
* **Nothing durable.**  The LDraw text and the preview image live in memory,
  produced inside a temporary directory that is removed before the function
  returns.  The UI writes nothing into ``artifacts/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import secrets
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence

from src.data.bricks import PART_VOCAB, WORLD, parse_bricks
from src.delivery.pipeline import DeliveryError
from src.demo.showcase import (ShowcaseError, parse_inventory, write_ldraw)
from src.rendering.preview import PreviewError, write_preview

ROOT = Path(__file__).resolve().parents[2]

#: The delivery command line is the single source of the payload and of the
#: static-delivery verdict.  It is loaded by path because its filename starts
#: with a digit and is therefore not an importable module name.
DELIVERY_CLI = ROOT / "scripts/27_delivery.py"

MODE_COMPARE = "compare"
MODE_PIPELINE = "f-pipeline"
MODES = (MODE_COMPARE, MODE_PIPELINE)

#: Labels for the two modes, and the boundary each one carries.
MODE_LABELS = {
    MODE_COMPARE: "既有作品比對",
    MODE_PIPELINE: "最低 F-pipeline",
}

#: The CP-SAT controls.  They belong to the F-pipeline and to nothing else.
PIPELINE_ONLY_FIELDS = ("time_limit", "seed")

#: Defaults matching ``scripts/27_delivery.py``.
DEFAULT_TOP_N = 5
DEFAULT_TIME_LIMIT = 2.0
DEFAULT_SEED = 0

#: A localhost form is still a parser boundary.  These caps exist so a
#: malformed or runaway submission is refused by name rather than absorbed.
MAX_CAPTION_CHARS = 2000
MAX_INVENTORY_SPEC_CHARS = 400
MAX_TOP_N = 50
MAX_TIME_LIMIT = 60.0

#: Half-width 0-9 and nothing else.  ``str.isdigit`` is true for Arabic-Indic,
#: Devanagari and a dozen other scripts, and ``int`` converts them happily, so
#: a quantity could arrive in a script no other part of the report echoes --
#: and two different strings would name the same stock.  A number here is
#: refused unless it is written the one way the reports are written.
_ASCII_WHOLE = re.compile(r"\A[0-9]+\Z")

#: The same rule for a decimal.  It also refuses ``nan``, ``inf`` and
#: exponent forms outright, so those never reach ``float`` at all.
_ASCII_DECIMAL = re.compile(r"\A[0-9]+(?:\.[0-9]+)?\Z")

#: A number longer than this is not a typo to be interpreted.  It also keeps
#: every value far inside CPython's integer-parsing limit: past roughly 4,300
#: digits ``int()`` raises a ValueError that is not one of this module's
#: refusals, which would surface as a defect page for what is really bad
#: input.  Refusing on length makes it a controlled 400.
MAX_NUMBER_CHARS = 9

#: No structure can use more of one part than the world has cells, so a stock
#: larger than that cannot change any outcome -- it can only make a form that
#: accepts nonsense look like it understood something.
MAX_PART_COUNT = WORLD ** 3

RETRIEVAL_LIMIT_ZH = (
    "目前是確定性詞彙檢索（Unicode 詞彙 baseline），不是多語 embedding，"
    "也不是正式語意檢索成效證據。")
CONNECTIVITY_LIMIT_ZH = (
    "連通性是相鄰層 footprint 交集，不是物理支撐，也不是穩定性分析；"
    "接觸地面同樣不是穩定性結果。")
NOT_A_METRIC_ZH = (
    "本頁任何數字都不是指標，也不可與已封存的 Phase 2 評估比較。")
NO_MODEL_ZH = (
    "本介面不生成、不載入權重、不啟用 Phase 3 placement gate，"
    "也不執行正式評估。")


class UiError(ValueError):
    """A submitted form the UI refuses, with a message a reader can act on.

    Distinct from :class:`~src.delivery.pipeline.DeliveryError` and
    :class:`~src.demo.showcase.ShowcaseError` only in where it was raised;
    all three are rendered the same way, and none of them ever reaches the
    browser as a traceback.
    """


#: Every exception class the UI treats as "refused, and here is why".  Anything
#: outside this tuple is a defect, and the server says so instead of guessing.
REFUSALS = (UiError, DeliveryError, ShowcaseError, PreviewError)


# ---------------------------------------------------------------------------
# The delivery command line, loaded once
# ---------------------------------------------------------------------------

_DELIVERY: ModuleType | None = None


def load_delivery(path: str | Path | None = None) -> ModuleType:
    """Import ``scripts/27_delivery.py`` and cache it.

    The UI reuses that module's ``make_payload`` verbatim.  That is the whole
    point: the page a person reads and the command a reviewer runs go through
    one implementation, so they cannot disagree about what is deliverable.
    """
    global _DELIVERY
    if path is None and _DELIVERY is not None:
        return _DELIVERY
    target = Path(path) if path is not None else DELIVERY_CLI
    if not target.is_file():
        raise UiError(f"the delivery command line is missing: {target}")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "brickagain_delivery_cli", target)
    if spec is None or spec.loader is None:
        raise UiError(f"the delivery command line could not be loaded: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attribute in ("make_payload", "DELIVERY_CHECKS", "DEFAULT_CATALOG"):
        if not hasattr(module, attribute):
            raise UiError(
                f"{target.name} does not expose {attribute}; the UI will not "
                "substitute its own copy of the delivery contract")
    if path is None:
        _DELIVERY = module
    return module


def default_catalog(delivery: ModuleType | None = None) -> Path:
    """The catalogue path the delivery command line already defaults to."""
    return Path((delivery or load_delivery()).DEFAULT_CATALOG)


def delivery_checks(delivery: ModuleType | None = None) -> tuple[str, ...]:
    """The static delivery checks, read from the delivery command line."""
    return tuple((delivery or load_delivery()).DELIVERY_CHECKS)


# ---------------------------------------------------------------------------
# Page one: the submitted form
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UiRequest:
    """One validated submission, ready to hand to the delivery payload."""

    mode: str
    caption: str
    inventory: dict[str, int]
    inventory_spec: str
    grid: dict[str, int]
    advanced_spec: str
    top_n: int
    time_limit: float | None
    seed: int | None

    @property
    def is_pipeline(self) -> bool:
        return self.mode == MODE_PIPELINE

    def as_namespace(self, catalog: str | Path) -> argparse.Namespace:
        """The exact argument surface ``make_payload`` reads.

        ``exclude_object_id`` is fixed at ``None``.  It is an evaluation
        safeguard for a query object, and this UI has no evaluation to guard:
        offering it here would invite exactly the held-out-case use the UI is
        forbidden to make.
        """
        return argparse.Namespace(
            mode=self.mode,
            caption=self.caption,
            inventory=self.inventory_spec,
            catalog=str(catalog),
            top_n=self.top_n,
            exclude_object_id=None,
            time_limit=self.time_limit,
            seed=self.seed,
        )


def _one(fields: Mapping[str, Sequence[str]], name: str) -> str:
    """A single value for ``name``, or ``""``. Repeated fields are refused."""
    values = fields.get(name) or []
    if len(values) > 1:
        raise UiError(f"欄位 {name} 被送出 {len(values)} 次；請只填一次")
    return values[0] if values else ""


def _short(value: str, limit: int = 60) -> str:
    """A value fit to quote back inside a refusal, however it arrived."""
    text = value if len(value) <= limit else value[:limit] + "…"
    return repr(text)


def _whole(raw: str, name: str, *, minimum: int, maximum: int) -> int:
    """A half-width whole number, bounded in both length and value."""
    text = raw.strip()
    if not text:
        raise UiError(f"{name} 不可空白")
    if not _ASCII_WHOLE.match(text):
        raise UiError(
            f"{name} 只接受半形 0-9 的整數，收到 {_short(text)}")
    if len(text) > MAX_NUMBER_CHARS:
        raise UiError(
            f"{name} 的位數 {len(text)} 超過上限 {MAX_NUMBER_CHARS}；"
            "這不是可以被解讀的輸入")
    value = int(text)
    if value < minimum or value > maximum:
        raise UiError(f"{name} 必須介於 {minimum} 與 {maximum} 之間，收到 {value}")
    return value


def _decimal(raw: str, name: str, *, maximum: float) -> float:
    """A half-width decimal greater than zero, bounded in length and value."""
    text = raw.strip()
    if not _ASCII_DECIMAL.match(text) or len(text) > MAX_NUMBER_CHARS + 4:
        raise UiError(
            f"{name} 只接受半形數字寫成的有限數字（例如 2 或 2.5），"
            f"收到 {_short(text)}")
    value = float(text)
    if value <= 0:
        raise UiError(f"{name} 必須大於 0")
    if value > maximum:
        raise UiError(f"{name} 上限為 {maximum:g}，收到 {value:g}")
    return value


def _check_spec_counts(spec: str) -> None:
    """Bound the numeric *shape* of a hand-written stock string.

    :func:`~src.demo.showcase.parse_inventory` keeps every decision that is
    properly its own: the eight-part vocabulary, rotation normalisation, and
    the refusal to sum two spellings of one part.  What it does not do is
    bound a count -- it accepts any ``str.isdigit`` run of any length, and a
    long enough one makes ``int()`` raise a bare ValueError that is not one of
    this UI's refusals and would surface as a defect page.  So the shape is
    checked here and the meaning is still decided there.
    """
    for chunk in spec.split(","):
        part, separator, count = chunk.partition(":")
        if not separator:
            continue                     # parse_inventory names this one
        _whole(count, f"{_short(part.strip()) if part.strip() else '零件'} "
                      "的數量", minimum=0, maximum=MAX_PART_COUNT)


def inventory_spec_from_grid(grid: Mapping[str, int]) -> str:
    """``{"2x4": 3}`` -> ``"2x4:3"`` in vocabulary order.

    The string is then handed to :func:`src.demo.showcase.parse_inventory`,
    which is where rotation normalisation and every inventory refusal already
    live.  Building the string instead of the dict is deliberate: it keeps the
    UI on the same validated path as the command line rather than beside it.
    """
    return ",".join(f"{part}:{grid[part]}" for part in PART_VOCAB
                    if grid.get(part, 0) > 0)


def parse_form(fields: Mapping[str, Sequence[str]]) -> UiRequest:
    """Validate a page-one submission. Raises :class:`UiError` on refusal.

    Nothing inapplicable is dropped.  A CP-SAT control submitted against the
    comparison mode is named and refused, because a form that silently
    discards a setting hands back a result page that looks like it honoured
    one.
    """
    mode = _one(fields, "mode").strip() or MODE_COMPARE
    if mode not in MODES:
        raise UiError(f"模式 {_short(mode)} 不在 {list(MODES)} 之內")

    caption = _one(fields, "caption").strip()
    if not caption:
        raise UiError("文字需求不可空白：請描述要組的作品")
    if len(caption) > MAX_CAPTION_CHARS:
        raise UiError(
            f"文字需求長度 {len(caption)} 超過上限 {MAX_CAPTION_CHARS} 字")

    grid: dict[str, int] = {}
    for part in PART_VOCAB:
        raw = _one(fields, f"qty_{part}").strip()
        if not raw:
            continue
        count = _whole(raw, f"{part} 的數量", minimum=0,
                       maximum=MAX_PART_COUNT)
        if count:
            grid[part] = count

    advanced = _one(fields, "inventory_spec").strip()
    if len(advanced) > MAX_INVENTORY_SPEC_CHARS:
        raise UiError(
            f"庫存字串長度 {len(advanced)} 超過上限 "
            f"{MAX_INVENTORY_SPEC_CHARS} 字")
    if advanced and grid:
        raise UiError(
            "庫存字串與八格數量同時填寫；兩者是同一份庫存的兩種輸入方式，"
            "請只用其中一種，本介面不自行猜測該相加或覆蓋")
    if not advanced and not grid:
        raise UiError(
            "庫存不可空白：請在八種零件中至少填一個數量，"
            "或改用庫存字串（例如 2x4:10,1x2:8）")

    spec = advanced or inventory_spec_from_grid(grid)
    if advanced:
        _check_spec_counts(advanced)
    # parse_inventory owns rotation normalisation and every inventory refusal.
    inventory = parse_inventory(spec)

    top_n = _whole(_one(fields, "top_n") or str(DEFAULT_TOP_N),
                   "Top-N", minimum=1, maximum=MAX_TOP_N)

    time_limit: float | None = None
    seed: int | None = None
    if mode == MODE_COMPARE:
        for name in PIPELINE_ONLY_FIELDS:
            if _one(fields, name).strip():
                raise UiError(
                    f"{name} 設定 CP-SAT，不適用於「{MODE_LABELS[MODE_COMPARE]}」；"
                    "請清空該欄位或改選最低 F-pipeline。本介面不靜默忽略它")
    else:
        raw_limit = _one(fields, "time_limit").strip()
        time_limit = (_decimal(raw_limit, "time limit", maximum=MAX_TIME_LIMIT)
                      if raw_limit else DEFAULT_TIME_LIMIT)
        raw_seed = _one(fields, "seed").strip()
        seed = (_whole(raw_seed, "seed", minimum=0, maximum=2 ** 31 - 1)
                if raw_seed else DEFAULT_SEED)

    return UiRequest(
        mode=mode, caption=caption, inventory=inventory, inventory_spec=spec,
        grid=dict(grid), advanced_spec=advanced, top_n=top_n,
        time_limit=time_limit, seed=seed)


# ---------------------------------------------------------------------------
# Running one request
# ---------------------------------------------------------------------------

#: PNG signature followed by the IHDR chunk header.
_PNG_PREFIX = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"


def png_size(data: bytes) -> tuple[int, int]:
    """Intrinsic pixel size of a PNG, read from its own IHDR.

    The page reserves this much space for the preview so the layout does not
    jump when the image arrives.  Measuring the bytes rather than repeating a
    figure size from :mod:`src.rendering.preview` means the reservation cannot
    drift away from the image it is reserving for.
    """
    if not data.startswith(_PNG_PREFIX) or len(data) < 33:
        raise UiError("the preview is not a PNG this UI can measure")
    return (int.from_bytes(data[16:20], "big"),
            int.from_bytes(data[20:24], "big"))


@dataclass(frozen=True)
class Artifacts:
    """The two outputs, in memory, both derived from one brick list."""

    bricks_text: str
    ldraw: str
    preview: bytes
    preview_width: int
    preview_height: int
    preview_media_type: str = "image/png"


@dataclass(frozen=True)
class UiResult:
    request: UiRequest
    payload: dict
    report: dict | None
    artifacts: Artifacts | None
    handle: str | None

    @property
    def ready(self) -> bool:
        """The delivery command line's verdict, never the UI's own."""
        return bool(self.report
                    and self.report["delivery"]["static_delivery_ready"])


def _artifacts(report: dict, caption: str) -> Artifacts:
    """LDraw text and preview bytes for the one selected brick list.

    Both come from ``report["result"]["text"]`` and nothing else, so the file
    a person downloads and the image they looked at are the same structure by
    construction rather than by coincidence.  ``write_ldraw`` is called for
    real, into a temporary directory, so the bytes served are the bytes the
    command line writes -- including its refusal when a structure would not
    serialise.
    """
    text = report["result"]["text"]
    bricks = parse_bricks(text)
    with tempfile.TemporaryDirectory(prefix="brickagain-ui-") as tmp:
        scratch = Path(tmp)
        ldraw = write_ldraw(report, scratch / "model.ldr").read_bytes()
        preview = write_preview(
            scratch / "preview.png", bricks, title=caption).read_bytes()
    width, height = png_size(preview)
    return Artifacts(bricks_text=text, ldraw=ldraw.decode("utf-8"),
                     preview=preview, preview_width=width,
                     preview_height=height)


def run_request(request: UiRequest, *, catalog: str | Path | None = None,
                delivery: ModuleType | None = None,
                store: "ResultStore | None" = None) -> UiResult:
    """Run one submission through the delivery command line's own payload.

    Refusals propagate as :class:`UiError`, ``DeliveryError``,
    ``ShowcaseError`` or ``PreviewError``.  A run that completes without a
    deliverable result is *not* an error: it returns a result whose
    :attr:`UiResult.ready` is ``False``, carrying the per-candidate evidence
    that says why.
    """
    module = delivery or load_delivery()
    target = Path(catalog) if catalog is not None else default_catalog(module)
    payload, report = module.make_payload(request.as_namespace(target))
    if payload["method"]["model_loaded"] is not False:
        raise UiError(
            "the delivery payload reports a loaded model; this UI refuses to "
            "display a result that claims a decode it must never run")

    artifacts = None
    handle = None
    if report is not None and report["delivery"]["static_delivery_ready"]:
        artifacts = _artifacts(report, request.caption)
        if store is not None:
            handle = store.put(artifacts)
    return UiResult(request=request, payload=payload, report=report,
                    artifacts=artifacts, handle=handle)


# ---------------------------------------------------------------------------
# Where the two downloadable things live: memory, briefly
# ---------------------------------------------------------------------------

class ResultStore:
    """A small, bounded, in-process map from handle to one result's outputs.

    Bounded on purpose.  A long session should not accumulate every preview
    it ever drew, and nothing here is meant to survive the process: the UI has
    no store on disk, so closing it leaves nothing behind to clean up.
    """

    def __init__(self, limit: int = 8) -> None:
        if limit < 1:
            raise ValueError("the result store needs room for one result")
        self.limit = limit
        self._items: "OrderedDict[str, Artifacts]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def put(self, artifacts: Artifacts) -> str:
        handle = secrets.token_urlsafe(16)
        self._items[handle] = artifacts
        self._items.move_to_end(handle)
        while len(self._items) > self.limit:
            self._items.popitem(last=False)
        return handle

    def get(self, handle: str) -> Artifacts | None:
        return self._items.get(handle)
