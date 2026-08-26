"""The project model entry point: verify the pointer, then decode once.

This is the one place in the interface that loads weights, and it is
deliberately narrow.

**It verifies before it loads.**  ``runs/project_model.json`` names the
adapter and records a digest for every file in it.  The pointer's own verifier
-- ``scripts/24_project_model.py``'s ``verify_problems`` -- is loaded and run,
so this module has no second opinion about what a valid pointer is.  A single
mismatched digest means no decode happens.

**It cannot retrain, tune or reselect.**  There is no code path here that
writes a checkpoint, changes a hyper-parameter or chooses between models.  The
model is whatever the pointer names, at the digests the pointer records.

**The placement gate is opt-in and off by default.**  Turning it on is
possible and carries a fixed notice: that gate has never been formally
evaluated, Phase 3C is not authorised, and switching it on is not evidence that
anything improved -- in either direction.

**It is a demonstration, not an evaluation.**  One caption, one inventory, one
decode.  No batch, no Success@K, no frozen case, no metric.
:data:`DEMONSTRATION_NOTICE` says so and travels with every result.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

POINTER = ROOT / "runs/project_model.json"
PROJECT_MODEL_CLI = ROOT / "scripts/24_project_model.py"

#: Bounds on one demonstration decode.  Small on purpose: this is a
#: demonstration and a person is waiting for the page.
MAX_BRICKS = 60
MAX_TOKENS = 700
DEFAULT_TEMPERATURE = 0.6
MAX_TEMPERATURE = 1.5

CONNECTIVITY_MODES = ("off", "any", "all")

DEMONSTRATION_NOTICE = (
    "這是單次展示解碼：一個需求、一份庫存、一次生成。"
    "不是批次、不是 Success@K、不是 Phase 3C，也不是任何評估。"
    "本頁任何數字都不可與已封存的 Phase 2 結果並列。")

PLACEMENT_NOTICE = (
    "placement gate 已開啟。**這個閘門從未經過正式評估**："
    "Phase 3C 未獲授權，從未以它計算過任何指標，"
    "開啟它不能作為任何項目改善的證據，反方向也不行。"
    "唯一相關的前例指向另一個方向——InventoryGate 曾使 in_bounds 與 "
    "collision_free 的邊際通過率下降。")

NO_RESULT_NOTICE = (
    "這次解碼沒有產生通過靜態交付檢查的結構，因此沒有預覽也沒有下載。"
    "沒有通過檢查的結構不會被畫成一張看起來合法的圖片，也不會提供下載。")


class ModelEntryError(RuntimeError):
    """The project model cannot be used, and the message says why."""


@dataclass(frozen=True)
class ModelIdentity:
    """What the pointer says the project model is, after verification."""

    name: str
    adapter_path: str
    files: dict[str, dict]
    lora: dict
    revisions: dict
    selected_by: dict
    verified: bool
    problems: tuple[str, ...] = field(default_factory=tuple)

    @property
    def adapter_digest(self) -> str | None:
        entry = self.files.get("adapter_model.safetensors") or {}
        return entry.get("sha256")

    def as_dict(self) -> dict:
        return {
            "model": self.name,
            "adapter_path": self.adapter_path,
            "adapter_sha256": self.adapter_digest,
            "files": {name: {"sha256": body.get("sha256"),
                             "bytes": body.get("bytes")}
                      for name, body in sorted(self.files.items())},
            "lora": dict(self.lora),
            "revisions": dict(self.revisions),
            "selected_by": {
                "criterion": (self.selected_by or {}).get("criterion"),
                "record_sha256": (self.selected_by or {}).get("sha256"),
                "means": (self.selected_by or {}).get("means"),
            },
            "verified": self.verified,
            "problems": list(self.problems),
            "not_retrained": (
                "this interface loads the adapter the pointer names at the "
                "digests it records. It has no path that retrains, tunes or "
                "reselects a model"),
        }


def _cli():
    """``scripts/24_project_model.py``, loaded by path and not reimplemented."""
    if not PROJECT_MODEL_CLI.is_file():
        raise ModelEntryError(
            f"{PROJECT_MODEL_CLI.name} is missing, so the project model "
            "pointer cannot be verified; this interface will not load weights "
            "it cannot check")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "brickagain_project_model_cli", PROJECT_MODEL_CLI)
    if spec is None or spec.loader is None:
        raise ModelEntryError(
            f"{PROJECT_MODEL_CLI.name} could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "verify_problems"):
        raise ModelEntryError(
            f"{PROJECT_MODEL_CLI.name} does not expose verify_problems; this "
            "interface will not substitute its own idea of a valid pointer")
    return module


def identity(*, pointer=None, root=None) -> ModelIdentity:
    """Read and verify the pointer.  Loads no weights.

    ``root`` relocates the whole lookup, pointer included. A root that moved
    the adapter search but left the pointer at this repository's own path
    would report "the adapter is missing" for a tree that simply has no
    pointer -- two different situations with two different answers.
    """
    base = Path(root) if root is not None else ROOT
    if pointer is not None:
        target = Path(pointer)
    else:
        target = base / "runs/project_model.json"
    if not target.is_file():
        raise ModelEntryError(
            f"{target.name} is not here, so there is no project model to "
            "load. It is written by scripts/24_project_model.py in the "
            "private research tree and is not published; a public checkout "
            "has no runs/ directory and therefore no project model.")
    try:
        body = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelEntryError(
            f"{target.name} is not valid JSON: {exc}") from exc
    if not isinstance(body, dict) or body.get("kind") != "project_model":
        raise ModelEntryError(
            f"{target.name} does not declare itself a project_model record")
    problems = tuple(_cli().verify_problems(body, root=base))
    adapter = body.get("adapter") or {}
    return ModelIdentity(
        name=str(body.get("model", "")),
        adapter_path=str(adapter.get("path", "")),
        files=dict(adapter.get("files") or {}),
        lora=dict(body.get("lora") or {}),
        revisions=dict(body.get("revisions") or {}),
        selected_by=dict(body.get("selected_by") or {}),
        verified=not problems, problems=problems)


@dataclass(frozen=True)
class Decoded:
    """One demonstration decode and everything needed to read it honestly."""

    identity: ModelIdentity
    report: dict = field(repr=False)
    placement: bool
    connectivity: str
    seed: int
    temperature: float
    device: str
    seconds: float

    @property
    def ready(self) -> bool:
        """Whether this structure passes the static delivery checks.

        Read from the same nine checks ``scripts/27_delivery.py`` uses, so the
        interface has no second opinion about what is deliverable.
        """
        return bool(self.report.get("delivery", {}).get(
            "static_delivery_ready"))

    def as_dict(self) -> dict:
        return {
            "identity": self.identity.as_dict(),
            "placement_gate": self.placement,
            "connectivity": self.connectivity if self.placement else None,
            "placement_notice": PLACEMENT_NOTICE if self.placement else None,
            "seed": self.seed,
            "temperature": self.temperature,
            "device": self.device,
            "seconds": round(self.seconds, 3),
            "termination": self.report["result"]["termination"],
            "n_tokens": self.report["result"]["n_tokens"],
            "n_bricks": self.report["result"]["n_bricks"],
            "static_delivery_ready": self.ready,
            "demonstration_notice": DEMONSTRATION_NOTICE,
            "no_result_notice": None if self.ready else NO_RESULT_NOTICE,
        }


def decode(caption: str, inventory: dict[str, int], *, placement: bool = False,
           connectivity: str = "off", device: str = "mps", seed: int = 0,
           temperature: float = DEFAULT_TEMPERATURE,
           max_bricks: int = MAX_BRICKS, max_tokens: int = MAX_TOKENS,
           root=None) -> Decoded:
    """Decode one structure with the project model, then score it.

    The decode itself is :func:`src.demo.showcase.generate` with
    ``model="project"``, which loads through the same two loaders the frozen
    arms use.  A second spelling of that load order is how a demonstration ends
    up showing a model nobody trained, so there is not one here.

    The scoring is ``scripts/27_delivery.py``'s ``DELIVERY_CHECKS`` applied to
    the decoded report, so "deliverable" means the same thing on this page as
    it does on the command line.
    """
    import time

    from src.demo.showcase import MODEL_PROJECT, ShowcaseError, generate

    if connectivity not in CONNECTIVITY_MODES:
        raise ModelEntryError(
            f"connectivity={connectivity!r} is not one of "
            f"{list(CONNECTIVITY_MODES)}")
    if not placement and connectivity != "off":
        raise ModelEntryError(
            "connectivity configures the placement gate, and the placement "
            "gate was not asked for")
    if not isinstance(temperature, (int, float)) or isinstance(
            temperature, bool) or not 0 < float(temperature) <= MAX_TEMPERATURE:
        raise ModelEntryError(
            f"temperature must be greater than zero and at most "
            f"{MAX_TEMPERATURE}")
    if not 1 <= int(max_bricks) <= MAX_BRICKS:
        raise ModelEntryError(
            f"max_bricks must be between 1 and {MAX_BRICKS} for a "
            "demonstration decode")

    who = identity(root=root)
    if not who.verified:
        raise ModelEntryError(
            "the project model pointer does not describe this tree, so no "
            "weights were loaded: " + "; ".join(who.problems[:4]))

    started = time.monotonic()
    try:
        report = generate(
            caption, inventory, model=MODEL_PROJECT, placement=placement,
            connectivity=connectivity, device=device, seed=seed,
            temperature=float(temperature), max_bricks=int(max_bricks),
            max_tokens=int(max_tokens), local_files_only=True, root=root)
    except ShowcaseError as exc:
        raise ModelEntryError(f"the decode was refused: {exc}") from None
    except (OSError, RuntimeError) as exc:
        raise ModelEntryError(
            f"the project model could not be loaded or run: {exc}") from None
    seconds = time.monotonic() - started

    report["delivery"] = _delivery_verdict(report)
    if report["provenance"]["mode"] != "decoded":
        raise ModelEntryError(
            "the decode returned a report that does not declare itself "
            "decoded; this interface will not present it as one")
    return Decoded(identity=who, report=report, placement=placement,
                   connectivity=connectivity, seed=seed,
                   temperature=float(temperature), device=device,
                   seconds=seconds)


def _delivery_verdict(report: dict) -> dict:
    """Apply the delivery command line's own nine static checks."""
    from src.ui.app import load_delivery

    checks = tuple(load_delivery().DELIVERY_CHECKS)
    ready = all(report["checks"].get(name) is True for name in checks)
    return {
        "method": "project-model-demonstration",
        "checks_used": list(checks),
        "static_delivery_ready": ready,
        "note": ("the same nine static checks the delivery command line uses. "
                 "A decode ran, so termination is available and is included "
                 "in the report -- but the delivery verdict is the nine "
                 "static checks, exactly as on the command line."),
    }
