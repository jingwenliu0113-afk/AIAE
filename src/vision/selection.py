"""What can be verified about how this checkpoint was chosen, and what cannot.

The round that produced ``runs/vision/classifier`` ran several configurations
on the CUDA node and returned **one** of them.  The returned artefacts -- the
checkpoint manifest and the run summary -- describe that configuration
completely: its settings, its per-epoch validation log, the machine, the
dependency versions, and the digests of the data, the split and the code.  From
them a reader can re-derive, without trusting any prose, which epoch was
selected and why.

What they do not contain is the other configurations.  Their summaries stayed
on the node and were never returned, so there is nothing on this machine to
check a cross-configuration claim against.  That claim is therefore
**withdrawn** here rather than repeated: :func:`build` writes
``cross_configuration_selection: null`` beside the reason, and :func:`verify`
refuses a record that fills it in.

This is not a "the evidence is probably fine" note.  A selection claim with no
artefact behind it is the same kind of statement as an accuracy figure with no
run behind it, and the project's rule for those is that they do not get made.

**Verification is total, not sampled.**  :func:`verify` rebuilds the whole
record from the manifest, the summary and the weights file, and compares every
field of the stored copy against it -- including the ones a reader is least
likely to look at, which is exactly where an edit would go.  A field the
rebuild does not produce is refused as an extra, and a field the rebuild
produces that the stored record lacks is refused as missing.  Comparing a
chosen subset would leave the rest editable, and a record that can be edited
in the parts nobody compares is not evidence.

**Every malformed input is a named refusal.**  A validation loss that is text,
a NaN, an infinity, an epoch that is a string, a log that is not a list, a
manifest with a missing key: each raises :class:`SelectionError` saying which
field and why.  None of them reaches a caller as a traceback, because the
caller here is a command-line verifier and a stack trace is not an answer.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

RECORD_FILE = "selection_record.json"
RECORD_KIND = "brickagain.vision_selection_record"
SUMMARY_FILE = "run_summary.json"

#: The rule that was fixed before the fit ran, restated here so the record can
#: check the checkpoint against it instead of quoting it.
EPOCH_CRITERION = ("lowest validation loss; an exact tie keeps the earlier "
                   "epoch")

WITHDRAWN = (
    "An earlier report said this checkpoint was chosen from among several "
    "configurations by a pre-frozen criterion. The per-configuration "
    "summaries that would show it were produced on the node and were not "
    "returned, so no artefact on this machine supports it. The claim is "
    "withdrawn. What is supported, and is checked below, is the choice of "
    "epoch WITHIN this run: it is re-derived from the epoch log that "
    "travelled with the checkpoint")

BOUNDARY = (
    "this record describes how one checkpoint was fitted and which of its own "
    "epochs was kept. It is not a result, not an accuracy, and not a "
    "comparison with anything")

#: Every field :func:`build` produces.  Named so :func:`verify` can say which
#: one is missing rather than only that something is, and so a new field cannot
#: be added to the builder without the comparison learning about it.
RECORD_FIELDS: tuple[str, ...] = (
    "kind", "checkpoint", "configuration", "seed", "epochs_run", "epoch_log",
    "epoch_selection", "run_provenance", "trainable_parameters",
    "environment", "digests", "checks", "problems",
    "cross_configuration_selection", "cross_configuration_reason", "boundary",
)


class SelectionError(ValueError):
    """The selection record cannot be built, or does not check out."""


def _no_constants(literal: str):
    """Refuse ``NaN``/``Infinity`` at the JSON boundary rather than later.

    Python's decoder accepts all three by default and hands back a float that
    compares false against itself.  A digest field or a loss that arrived that
    way would then flow through the record and out into a report, so it is
    stopped where it enters.
    """
    raise SelectionError(
        f"the file contains the JSON constant {literal!r}; a selection record "
        "is built only from finite numbers")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise SelectionError(f"{path} is missing")
    try:
        body = json.loads(path.read_text(encoding="utf-8"),
                          parse_constant=_no_constants)
    except SelectionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectionError(f"{path} is not readable JSON: {exc}") from None
    if not isinstance(body, dict):
        raise SelectionError(f"{path} is not a JSON object")
    return body


def _mapping(value, where: str) -> dict:
    if not isinstance(value, dict):
        raise SelectionError(
            f"{where} must be an object, not {type(value).__name__}")
    return value


def _need(body: dict, key: str, where: str):
    if key not in body:
        raise SelectionError(f"{where} has no {key!r}")
    return body[key]


def _whole_number(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectionError(
            f"{where} must be a whole number, not {type(value).__name__}")
    return value


def _finite(value, where: str) -> float:
    """A number that is a number: not text, not a bool, not NaN, not ±Inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(
            f"{where} must be a number, not {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise SelectionError(
            f"{where} is {value!r}, which is not finite; a selection "
            "cannot be derived from it")
    return number


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(record) -> dict:
    """The record as plain JSON types, so a comparison is a comparison.

    A stored record has been through a file: its tuples are lists and its keys
    are strings.  Normalising both sides through the same encoder means a
    difference reported here is a difference in content rather than in how the
    two copies happened to be built.
    """
    try:
        text = json.dumps(record, sort_keys=True, ensure_ascii=False,
                          allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SelectionError(
            f"the record cannot be written as JSON: {exc}") from None
    return json.loads(text)


def split_log(epoch_log) -> tuple[list[dict], list[dict]]:
    """Separate the per-epoch rows from the run's provenance rows.

    The log the node returned ends with a row that carries no epoch: the
    applied tuning, the device report, and which parameters were actually left
    trainable.  It is kept -- it is the evidence that the requested stages were
    unfrozen rather than silently ignored -- but it is not an epoch and it is
    not offered to the selection.

    A row that is not an object at all is refused rather than filtered out.
    Dropping it silently would let a log of ``[1, 2, 3]`` verify.
    """
    if epoch_log is None:
        raise SelectionError("the checkpoint carries no epoch log")
    if not isinstance(epoch_log, list):
        raise SelectionError(
            "the epoch log must be a list, not "
            f"{type(epoch_log).__name__}")
    epochs: list[dict] = []
    provenance: list[dict] = []
    for position, row in enumerate(epoch_log):
        if not isinstance(row, dict):
            raise SelectionError(
                f"epoch log row {position} is a {type(row).__name__}, not an "
                "object; the log cannot be read")
        (epochs if "epoch" in row else provenance).append(row)
    return epochs, provenance


def best_epoch(epoch_log) -> int:
    """Re-derive the selected epoch from the log, by the fixed criterion.

    Written as its own function so the record's claim about which epoch won is
    a computation over the numbers on disk rather than a field copied from the
    manifest that stated it.
    """
    rows, _provenance = split_log(epoch_log)
    if not rows:
        raise SelectionError("the epoch log is empty; no epoch can be chosen")
    best: tuple[int, float] | None = None
    seen: set[int] = set()
    for row in rows:
        number = _whole_number(row["epoch"], "an epoch log row's 'epoch'")
        if number < 1:
            raise SelectionError(
                f"epoch {number} is not a positive epoch number")
        if number in seen:
            raise SelectionError(
                f"epoch {number} appears twice in the log; the selection "
                "would depend on which copy was read")
        seen.add(number)
        if "validation_loss" not in row:
            raise SelectionError(
                f"epoch {number} has no validation_loss; the selection cannot "
                "be re-derived from this log")
        loss = _finite(row["validation_loss"],
                       f"epoch {number}'s validation_loss")
        if best is None or loss < best[1]:
            best = (number, loss)
    return best[0]


def build(directory, *, code_digest=None) -> dict:
    """Assemble the record from the returned artefacts, checking as it goes."""
    from src.vision.model import VISION_MANIFEST, ModelError
    from src.vision.model import code_digest as compute_code_digest

    target = Path(directory)
    manifest = _read_json(target / VISION_MANIFEST)
    summary = _read_json(target / SUMMARY_FILE)
    where = str(target / VISION_MANIFEST)

    weights = _mapping(_need(manifest, "weights", where), f"{where}: weights")
    weights_name = _need(weights, "file", f"{where}: weights")
    if not isinstance(weights_name, str) or not weights_name:
        raise SelectionError(f"{where}: weights.file is not a file name")
    weights_path = target / weights_name
    if weights_path.parent != target or not weights_path.is_file():
        raise SelectionError(
            f"the manifest names {weights_name!r}, which is not a file in "
            f"{target}")
    actual_weights = sha256_file(weights_path)
    actual_bytes = weights_path.stat().st_size

    derived = best_epoch(manifest.get("epoch_log"))
    log, provenance = split_log(manifest.get("epoch_log"))
    stated = _whole_number(_need(manifest, "selected_epoch", where),
                           f"{where}: selected_epoch")

    if code_digest is None:
        try:
            actual_code = compute_code_digest()
        except ModelError as exc:
            raise SelectionError(str(exc)) from None
    else:
        actual_code = str(code_digest)

    configuration = _mapping(_need(manifest, "config", where),
                             f"{where}: config")

    checks = {
        "weights_digest_matches_manifest":
            actual_weights == weights.get("sha256"),
        "weights_size_matches_manifest":
            actual_bytes == weights.get("bytes"),
        "selected_epoch_is_the_argmin_of_the_epoch_log": derived == stated,
        "code_digest_matches_this_tree":
            actual_code == manifest.get("code_sha256"),
        "summary_and_manifest_agree_on_the_weights":
            summary.get("weights_sha256") == weights.get("sha256"),
        "summary_and_manifest_agree_on_the_data":
            summary.get("data_manifest_sha256")
            == manifest.get("data_manifest_sha256"),
        "summary_and_manifest_agree_on_the_split":
            summary.get("split_manifest_sha256")
            == manifest.get("split_manifest_sha256"),
        "summary_and_manifest_agree_on_the_selected_epoch":
            summary.get("selected_epoch") == stated,
    }

    record = {
        "kind": RECORD_KIND,
        "checkpoint": target.name,
        "configuration": configuration,
        "seed": manifest.get("seed"),
        "epochs_run": len(log),
        "epoch_log": log,
        "epoch_selection": {
            "criterion": EPOCH_CRITERION,
            "manifest_criterion": manifest.get("selection_criterion"),
            "selected_epoch_stated": stated,
            "selected_epoch_rederived": derived,
            "validation_losses": {
                str(row["epoch"]): row["validation_loss"] for row in log},
            "note": (
                "re-derived from the epoch log on disk rather than read from "
                "the field that states it, so a manifest that named the wrong "
                "epoch would fail this record rather than pass it"),
        },
        "run_provenance": provenance,
        "trainable_parameters": next(
            (row["frozen_parameters"] for row in provenance
             if "frozen_parameters" in row), None),
        "environment": {
            "device": manifest.get("device"),
            "device_report": manifest.get("device_report"),
            "dependencies": manifest.get("dependencies"),
            "tuning": manifest.get("tuning"),
        },
        "digests": {
            "weights_sha256": actual_weights,
            "weights_bytes": actual_bytes,
            "code_sha256": actual_code,
            "data_manifest_sha256": manifest.get("data_manifest_sha256"),
            "split_manifest_sha256": manifest.get("split_manifest_sha256"),
            "backbone": manifest.get("backbone"),
        },
        "checks": checks,
        "problems": sorted(name for name, ok in checks.items() if not ok),
        "cross_configuration_selection": None,
        "cross_configuration_reason": WITHDRAWN,
        "boundary": BOUNDARY,
    }
    unnamed = sorted(set(record) - set(RECORD_FIELDS))
    if unnamed:
        raise SelectionError(
            f"the builder produced field(s) {unnamed} that RECORD_FIELDS does "
            "not name, so verification would not compare them")
    return canonical(record)


def differences(stored: dict, fresh: dict) -> list[str]:
    """Every way the stored record and a rebuilt one disagree, by name.

    Both directions, and no field exempt.  A field only the stored copy has is
    as much a disagreement as a value that moved: it is content nothing on disk
    produced, sitting inside a document whose whole purpose is to be checkable.
    """
    problems: list[str] = []
    for name in sorted(set(fresh) - set(stored)):
        problems.append(
            f"the record is missing {name!r}, which a record rebuilt from the "
            "artefacts contains")
    for name in sorted(set(stored) - set(fresh)):
        problems.append(
            f"the record carries {name!r}, which is not a field a rebuilt "
            "record has; it was added to the file rather than derived")
    for name in sorted(set(stored) & set(fresh)):
        if stored[name] != fresh[name]:
            problems.append(
                f"the record's {name!r} does not match the artefacts on disk")
    return problems


def verify(record, directory, *, code_digest=None) -> list[str]:
    """Re-derive the record from the artefacts and list every disagreement."""
    if not isinstance(record, dict) or record.get("kind") != RECORD_KIND:
        return ["the file is not a BrickAgain vision selection record"]
    problems: list[str] = []
    if record.get("cross_configuration_selection") is not None:
        problems.append(
            "cross_configuration_selection is filled in. No artefact on this "
            "machine can support a cross-configuration claim, so a record "
            "that makes one is refused rather than trusted")
    try:
        stored = canonical(record)
        fresh = build(directory, code_digest=code_digest)
    except SelectionError as exc:
        return problems + [str(exc)]
    return problems + differences(stored, fresh)


def write(directory, *, code_digest=None) -> tuple[Path, str]:
    """Write the record beside the checkpoint; return its path and digest."""
    body = build(directory, code_digest=code_digest)
    if body["problems"]:
        raise SelectionError(
            "the artefacts do not check out, so no selection record is "
            f"written: {body['problems']}")
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False,
                         indent=2, allow_nan=False).encode("utf-8") + b"\n"
    out = Path(directory) / RECORD_FILE
    out.write_bytes(payload)
    return out, hashlib.sha256(payload).hexdigest()


def read(directory) -> dict:
    """Read a stored record, refusing an unreadable one by name."""
    return _read_json(Path(directory) / RECORD_FILE)
