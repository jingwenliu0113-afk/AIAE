"""Report 16: the long-run session, its metrics, its verdicts and its replay.

Report 15 answered "does clearing the cache help, once you stop confounding it
with running second". Report 16 asks a narrower question report 15 cannot
reach: does the mitigation still hold at 2,000 rows, and does its cost grow?
All three runs use ``empty_cache``, so nothing here can say anything about a
control arm, and the verdict machinery is built to make that impossible to
write down by accident.

Four rules shape most of this file:

* the constants come from the approved design JSON and are embedded verbatim.
  This module's own constants are cross-checked against that file and a
  mismatch is a hard error, so a second set of numbers can never quietly win;
* a metric that cannot be computed is ``None`` with a reason. A metric that
  genuinely measured zero is ``0.0`` -- zero is a legitimate reading, and only
  an *uncomputable* one has to be null;
* a tool failure produces no verdict at all, because ``not_applicable`` means
  "this measurement cannot decide" while a tool failure means "this was not a
  measurement";
* nothing reads back a conclusion. Every verdict is recomputed from the raw
  per-row array and the watchdog's own log, and stored values are compared
  against the recomputation in both directions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.training.preflight import GATE_METRICS
from src.training.session import (boot_identity, copy_once, exclusive_lock,
                                  manifest_digest, now_iso, sha256_file,
                                  snapshot_sources, verify_sources,
                                  write_once_json)
from src.training.watchdog import (IDENTITY_FIELDS, SafetySpec,
                                   is_safety_reason, is_tool_failure,
                                   read_stop_request, replay_watchdog_log,
                                   replay_watchdog_semantics,
                                   watchdog_terminal_problems)

ROOT = Path(__file__).resolve().parents[2]
DESIGN_JSON = ROOT / "data" / "reports" / "16_longrun_design.json"
REPORT_DIR = ROOT / "data" / "reports" / "16_longrun"

#: Every repository-local file the measured child executes. Not a hand-kept
#: list of "the interesting ones": report 16's first real run died inside
#: ``src/generation/brickgpt.py``, which was not here -- so the snapshot could
#: not say what code had produced the failure, and would not have been able to
#: say what code produced the numbers either. A test walks the child's actual
#: import closure and fails if anything in it is missing from this tuple.
CODE_FILES = ("scripts/16_longrun.py",
              "src/__init__.py",
              "src/model_ids.py",
              "src/data/__init__.py",
              "src/data/bricks.py",
              "src/data/instruction.py",
              "src/generation/__init__.py",
              "src/generation/brickgpt.py",
              "src/generation/prompt.py",
              "src/training/__init__.py",
              "src/training/diagnostics.py",
              "src/training/longrun.py",
              "src/training/lora.py",
              "src/training/preflight.py",
              "src/training/session.py",
              "src/training/watchdog.py")

# ---------------------------------------------------------------------------
# Constants. Declared here for readability and checked against the design file
# at import-adjacent time by build_plan(); the design file is the authority.
# ---------------------------------------------------------------------------

LENGTHS = (500, 1000, 2000)
RUN_IDS = ("b1", "b2", "b3")
CONDITION = "empty_cache"
POOL_PAIRS, ROWS_PER_PAIR, POOL_ROWS, SEED = 250, 8, 2000, 0

#: The two row sources this loader can read, and nothing else.
#:
#: ``pool`` is the frozen 250-pair, 2,000-row sample every measurement in the
#: project so far has used -- reports 15 and 16, the gates, and both hypothesis
#: arms. ``full_train`` is the whole training split, which only the final run
#: reads. Both are declared here with the shape they must have and are checked
#: against the file, so a split that changed size is a refusal rather than a
#: quietly shorter run.
DATA_SOURCES: dict[str, dict] = {
    "pool": {"pairs": POOL_PAIRS, "rows": POOL_ROWS},
    "full_train": {"pairs": 1198, "rows": 9584},
}
WINDOW, SECONDARY_AGGREGATION, MEMORY_EVERY, EMPTY_CACHE_EVERY = 20, 100, 5, 10
MIN_ROWS = {"D100": 200, "D20": 40, "Dmax": 120, "clear_growth": 400}
CLEAR_GROWTH_MIN_CALLS, GROWTH_SEGMENT = 40, 20
Q1_HOLDS_D100, Q1_HOLDS_DMAX = 1.5, 3.0
Q1_FAILS_D100, Q1_FAILS_DMAX = 3.0, 6.0
Q2_STABLE_GROWTH, Q2_STABLE_SHARE = 1.5, 0.03
Q2_SCALES_GROWTH, Q2_SCALES_SHARE = 3.0, 0.05
REPEATABLE_MAX, NOT_REPEATABLE_MIN = 1.10, 1.30
LOSS_TOLERANCE = 0.001

Q1_RUN_VALUES = ("holds", "fails", "indeterminate", "not_applicable")
Q2_RUN_VALUES = ("stable", "scales", "indeterminate", "not_applicable")
Q1_PLAN_VALUES = ("holds_to_2000", "holds_to_1000", "holds_to_500", "fails",
                  "indeterminate", "not_applicable")
Q2_PLAN_VALUES = ("stable", "scales", "indeterminate", "not_applicable")

#: Section 4.6 of the design, in one place. The child writes exactly these,
#: single-run replay requires exactly these, and the cross-run comparison
#: compares exactly these. Three copies of a list is three chances for one of
#: them to quietly fall behind.
PROVENANCE_FIELDS = ("code_sha256", "instruction_sha256", "selection_digest",
                     "training_order_digest", "lora_config", "optimizer",
                     "packages", "device", "dtype", "phases",
                     "stop_conditions", "measurement_intervals", "base_model",
                     "base_revision", "published_adapter",
                     "published_adapter_revision", "tokenizer_revision",
                     "trainable_parameters")

#: Kept as an alias so older call sites read the same list, not another one.
CHILD_PROVENANCE_REQUIRED = PROVENANCE_FIELDS

#: Section 4.6 again: every field above is identical across the three runs,
#: and the declared length k is the one exception. k lives inside
#: ``measurement_intervals`` as ``max_rows``, so that single key is compared
#: against the run's own ``declared_rows`` rather than against the other runs.
#:
#: The alternative -- dropping ``max_rows`` from provenance so the blocks match
#: trivially -- would make the comparison pass by no longer recording the one
#: number that legitimately differs. An exception that is written down is
#: checkable; one that is achieved by not storing anything is not.
CROSS_RUN_VARIES = {"measurement_intervals": ("max_rows",)}


def provenance_problems(provenance, *, declared_rows) -> list[str]:
    """The provenance block, checked once, for the loader and for the replay.

    Round 21 left two definitions of what a good provenance block is. The
    loader asked only whether it was a mapping; the replay required all of
    section 4.6's fields *and* that ``measurement_intervals.max_rows`` be this
    run's declared length. So a child could load a model, measure every row,
    write its report -- and only then would anybody discover that the report
    could never be verified, with the boot already spent and unrepeatable.

    One function, called from both ends, returning the same sentences.
    """
    if not isinstance(provenance, dict) or not provenance:
        return ["provenance is empty or is not an object, so none of the "
                "fields it should carry are there"]
    problems = []
    missing = [k for k in PROVENANCE_FIELDS if k not in provenance]
    if missing:
        problems.append(f"provenance is missing {missing}")
    if "measurement_intervals" in provenance:
        # Absence is already reported by the ``missing`` check above, so this
        # only asks whether the block that is there says the right thing.
        problems += measurement_interval_problems(
            provenance["measurement_intervals"], declared_rows=declared_rows)
    return problems


def measurement_interval_problems(intervals, *, declared_rows) -> list[str]:
    """Does this ``measurement_intervals`` block declare this run's length?

    Split out because it has two callers with different ideas about whether
    the block may be absent: this module's provenance check reports absence
    through its own required-fields list, while the gate suite requires it
    outright. What "a good block" means is one fact, and it is this one.

    ``max_rows`` is required to be a strict ``int``. ``100.0 == 100`` and
    ``True == 1`` in Python, so a plain comparison accepts a float and a
    boolean as row counts -- and a provenance field that accepts ``True`` is
    not recording anything.
    """
    if not isinstance(intervals, dict):
        return [f"measurement_intervals is {type(intervals).__name__}, not an "
                "object"]
    if "max_rows" not in intervals:
        return ["measurement_intervals records no max_rows, so the one value "
                "allowed to differ between runs is not written down"]
    value = intervals["max_rows"]
    if isinstance(value, bool) or not isinstance(value, int):
        return [f"measurement_intervals.max_rows is {value!r}, a "
                f"{type(value).__name__} rather than a whole number of rows"]
    if value != declared_rows:
        return [f"measurement_intervals.max_rows is {value!r}, not this run's "
                f"declared {declared_rows!r}"]
    return []


EVENT_GATE_ATTEMPT = "gate_attempt"
EVENT_PRE_SPAWN_ABORT = "pre_spawn_abort"
EVENT_WATCHDOG_LAUNCH_FAILED = "watchdog_launch_failed"
EVENT_STARTED = "measurement_started"
EVENT_FINISHED = "measurement_finished"
EVENT_PLAN_ARM_CANCELLED = "plan_arm_cancelled"
EVENT_RETRYABLE = (EVENT_GATE_ATTEMPT, EVENT_PRE_SPAWN_ABORT,
                   EVENT_WATCHDOG_LAUNCH_FAILED)
FINISHED_OUTCOMES = ("completed", "nonzero_exit", "timed_out", "no_report")

RULE_R1 = (
    "R1: after the measurement_finished event of a run whose outcome is "
    "'completed' and which had no tool failure, the parent recomputes that "
    "run's Q1 verdict from the child's per_row array and watchdog.jsonl. If "
    "the verdict is 'fails', every not-yet-executed run is cancelled by rule. "
    "'holds', 'indeterminate' and 'not_applicable' all continue.")

# ---------------------------------------------------------------------------
# Experiment identity. A directory name is built from this, so it is checked
# against a whitelist rather than sanitised: "../../etc" has no safe reading.
# ---------------------------------------------------------------------------

SAFE_ID = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,62}\Z")


#: What the measured child's environment is pinned to, by the parent when it
#: spawns and by the child itself on entry. Both, deliberately: "did the
#: operator export HF_HUB_OFFLINE" is not a property a measurement may depend
#: on, and a child launched by hand for debugging must behave the same way.
PRODUCTION_OFFLINE_ENV = {"HF_HUB_OFFLINE": "1",
                          "TRANSFORMERS_OFFLINE": "1",
                          "HF_HUB_DISABLE_TELEMETRY": "1"}


def offline_environment(base: dict | None = None) -> dict:
    """``base`` (default: this process's environment) with offline pinned."""
    return {**(dict(base) if base is not None else dict(os.environ)),
            **PRODUCTION_OFFLINE_ENV}


def enforce_offline_environment() -> dict:
    """Pin this process's environment. Called by the child before it loads."""
    os.environ.update(PRODUCTION_OFFLINE_ENV)
    return dict(PRODUCTION_OFFLINE_ENV)


class OfflineNotGuaranteed(RuntimeError):
    """This process cannot be made strictly offline any more.

    Not a network failure: nothing has been reached for. It is raised *before*
    the tokenizer, the model or the adapter, because the alternative is a
    measured run that quietly depends on the hub being up.
    """


#: Modules that read the offline flag exactly once, when they are imported,
#: and keep the answer for the life of the process. ``transformers`` does not
#: get its own entry: ``transformers.utils.hub.is_offline_mode()`` returns
#: this same constant, so checking it twice would only look thorough.
OFFLINE_FROZEN_MODULES = (("huggingface_hub.constants", "HF_HUB_OFFLINE"),)


def offline_freeze_problems(*, modules=None) -> list[str]:
    """Has anything already decided this process is online?

    Round 20 moved :func:`enforce_offline_environment` ahead of ``load``'s
    imports, which makes ``load`` strictly offline *when it performs the first
    import*. It cannot help when the caller imported ``peft`` first: by then
    ``huggingface_hub`` has read ``HF_HUB_OFFLINE`` and setting the variable
    afterwards changes nothing -- measured in round 20, where the adapter load
    still called ``file_exists()`` on the hub. So the honest move is to say so
    and stop, rather than to load something that may have been fetched.

    ``modules`` is injectable so the guard can be driven without arranging a
    real import; production reads ``sys.modules``.
    """
    seen = sys.modules if modules is None else modules
    problems = []
    for name, attribute in OFFLINE_FROZEN_MODULES:
        module = seen.get(name)
        if module is None:
            continue  # never imported: the next import will read the pin
        if not getattr(module, attribute, False):
            problems.append(
                f"{name} was imported before {attribute} was set, so it has "
                f"frozen {attribute} as false for the life of this process; "
                "setting it now cannot change that, and peft would still "
                "reach the hub. Set HF_HUB_OFFLINE=1 before importing "
                "anything, or run the measured child, which does.")
    return problems


#: The files each pinned repository must already have locally. A tokenizer
#: needs its three; a model needs a config and at least one weight shard.
PINNED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                          "special_tokens_map.json")
PINNED_MODEL_FILES = ("config.json",)
PINNED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
PINNED_WEIGHT_CANDIDATES = ("model.safetensors", "pytorch_model.bin")


def _resolve_cached(repo_id: str, filename: str, revision: str) -> str | None:
    """Where a pinned file sits locally, or ``None``. Never fetches."""
    from huggingface_hub import try_to_load_from_cache

    # Returns a path, or None (not cached), or a sentinel meaning "the hub
    # told us this file does not exist". Only a real, present path counts.
    got = try_to_load_from_cache(repo_id, filename, revision=revision)
    if not isinstance(got, (str, Path)):
        return None
    return str(got) if Path(got).exists() else None


def _repository_evidence(repo_id: str, revision: str, required, *,
                         any_of=()) -> dict:
    """Portable evidence that one pinned repository resolves locally.

    Only what travels: the repo id, the revision, file names and their sizes
    and digests. Never the cache path -- it contains a home directory, and a
    report is a document other people read.
    """
    files, problems = [], []
    for name in required:
        path = _resolve_cached(repo_id, name, revision)
        if path is None:
            problems.append(f"{repo_id}@{revision[:12]} is missing {name} in "
                            "the local cache")
            continue
        files.append({"name": name, "bytes": Path(path).stat().st_size,
                      "sha256": sha256_file(path)})
    if any_of:
        found = None
        for name in any_of:
            path = _resolve_cached(repo_id, name, revision)
            if path is not None:
                found = {"name": name, "bytes": Path(path).stat().st_size,
                         "sha256": sha256_file(path)}
                break
        if found is None:
            problems.append(f"{repo_id}@{revision[:12]} has none of "
                            f"{list(any_of)} in the local cache")
        else:
            files.append(found)
    return {"repo_id": repo_id, "revision": revision,
            "files": sorted(files, key=lambda f: f["name"]),
            "problems": problems}


def dependency_preflight() -> dict:
    """Resolve every pinned dependency read-only, before anything is spent.

    Report 16's first measured run passed the gate, wrote
    ``measurement_started`` -- which spends the boot -- spawned the child, and
    only then discovered that the tokenizer could not be resolved. The boot
    was gone, and a run that starts without finishing makes the whole
    experiment terminal incomplete. The dependency was knowable a quarter of
    an hour earlier, for free.

    So: no network, no tensors, no MPS. Just "is every pinned file already on
    this disk", answered before the gate is polled.
    """
    from src.model_ids import (ADAPTER, ADAPTER_REVISION, BASE_MODEL,
                               BASE_REVISION, TOKENIZER, TOKENIZER_REVISION)

    repos = [
        _repository_evidence(TOKENIZER, TOKENIZER_REVISION,
                             PINNED_TOKENIZER_FILES),
        _repository_evidence(BASE_MODEL, BASE_REVISION, PINNED_MODEL_FILES,
                             any_of=PINNED_WEIGHT_CANDIDATES),
        _repository_evidence(ADAPTER, ADAPTER_REVISION, PINNED_ADAPTER_FILES),
    ]
    problems = [p for r in repos for p in r["problems"]]
    pool = ROOT / "data" / "processed" / "instruct_inv_train.jsonl"
    if not pool.exists():
        problems.append("the instruction pool "
                        "data/processed/instruct_inv_train.jsonl is missing")
    evidence = {
        "schema_version": 1, "kind": "longrun_dependency_preflight",
        "network_used": False, "tensors_loaded": False, "device_initialised": False,
        "repositories": [{k: v for k, v in r.items() if k != "problems"}
                         for r in repos],
        "instruction_pool": {
            "path": "data/processed/instruct_inv_train.jsonl",
            "sha256": sha256_file(pool) if pool.exists() else None},
    }
    return {"ok": not problems, "problems": problems, "evidence": evidence}


#: The dependency digest's own identity, mixed into the value so a digest of
#: this evidence can never collide with a digest of something else shaped
#: like it.
DEPENDENCY_DIGEST_KIND = "brickagain.dependency_digest"
DEPENDENCY_DIGEST_VERSION = 1


def dependency_digest(evidence) -> str:
    """One canonical value over *which bytes* the pinned dependencies are.

    :func:`dependency_preflight` already answers "is every pinned file here".
    It cannot answer "is it the same file", and the difference matters: a
    ``tokenizer.json`` of the correct name resolves perfectly well whatever is
    inside it, and a run against a vocabulary nobody compared is a run whose
    numbers are not comparable with anything.

    So this digests the portable evidence -- repository id, pinned revision,
    and every required file's name, size and digest, plus the instruction
    pool's repository-relative path and digest -- and nothing else. Three
    consequences, each deliberate:

    * **order cannot change it.** Repositories are sorted by their own
      canonical form and files by name and digest, so a hub that enumerates
      differently on two machines still produces one value. The two entries
      that share a repository id -- the tokenizer and the published adapter
      come from the same repo -- stay separate entries rather than being
      merged by id, because their file sets are what distinguish them;
    * **incidental fields cannot change it.** ``network_used`` and the rest
      describe the *reading*, not the dependencies. Including them would make
      the digest depend on how it was taken;
    * **absent evidence still digests.** An empty or missing block produces a
      value, and one no populated cache can match. A comparison that raised
      instead would turn a refusal into a crash.

    Deliberately *not* stored in the pack manifest: a digest kept beside the
    files it authenticates is rewritten by whoever rewrites them. Like
    ``pack_digest``, it is carried to the node by a separate route.
    """
    body = evidence if isinstance(evidence, dict) else {}
    repositories = []
    for repo in body.get("repositories") or []:
        if not isinstance(repo, dict):
            continue
        files = [{"name": f.get("name"), "bytes": f.get("bytes"),
                  "sha256": f.get("sha256")}
                 for f in (repo.get("files") or []) if isinstance(f, dict)]
        files.sort(key=lambda f: (str(f["name"]), str(f["sha256"])))
        repositories.append({"repo_id": repo.get("repo_id"),
                             "revision": repo.get("revision"),
                             "files": files})
    repositories.sort(key=canonical_json)
    pool = body.get("instruction_pool") or {}
    return digest_obj({
        "kind": DEPENDENCY_DIGEST_KIND,
        "schema_version": DEPENDENCY_DIGEST_VERSION,
        "repositories": repositories,
        "instruction_pool": {"path": pool.get("path"),
                             "sha256": pool.get("sha256")},
    })


#: What a run leaves behind when the child dies without writing a report.
FAILURE_EVIDENCE_VERSION = 1
FAILURE_EVIDENCE_KIND = "longrun_failure_evidence"
FAILURE_STAGES = ("spawn", "source_check", "preflight", "dependency",
                  "model_load", "training", "teardown", "report_write",
                  "unknown")

#: The evidence's whole vocabulary. Checked as a set on the way in *and* on
#: the way out: a field nobody validates is a field anybody can add, and an
#: extra key in a re-signed file is exactly how a story gets edited after the
#: fact without breaking a single digest.
FAILURE_EVIDENCE_FIELDS = ("schema_version", "kind", "experiment_id", "run_id",
                           "stage", "exception_type", "summary", "written_at")


#: Provider-issued keys. Round 19 stopped every one of these at the first
#: hyphen or underscore, which is where every *modern* key format starts:
#: ``sk-proj-...``, ``sk-ant-api03-...``, ``github_pat_11...``, ``xoxb-...``.
#: A prefix followed by four characters is not a redaction.
_SECRET_RE = re.compile(
    r"\b(?:hf_|sk-|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|glpat-|npm_|dop_v1_"
    r"|xox[abprs]-|AKIA|ASIA)[A-Za-z0-9_-]{6,}")

#: Account identities, which are personal even without a secret attached.
_ORG_RE = re.compile(r"\borg-[A-Za-z0-9_-]{4,}")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_AUTH_VALUE_RE = re.compile(r"(?i)\b(?:bearer|basic)\s+\S+")

#: ``key = value``, with an explicit separator and a non-empty value. This is
#: the *high-confidence* form and it is shared with the replay: naming a
#: credential is not disclosing one, so "password was not configured" and
#: "no api key is required" have to survive, while ``password=...`` must not.
_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token"
    r"|client[-_ ]?secret|private[-_ ]?key|secret[-_ ]?key|auth[-_ ]?token"
    r"|password|passwd|secret|token|authorization|credentials?)\b"
    r"\s*[:=]\s*\S+")

#: Wider, and writer-only: a traceback that prints a credential field without
#: a separator ("token for user ...") is still worth dropping on the way out,
#: but refusing it on the way back in would refuse honest evidence.
_CREDENTIAL_KEYWORD_RE = re.compile(
    r"(?i)\b(?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token"
    r"|client[-_ ]?secret|private[-_ ]?key|secret[-_ ]?key|auth[-_ ]?token"
    r"|password|passwd|secret|token|authorization|credentials?)\b"
    r"\s*[:=]?\s*\S*")

#: Absolute paths belonging to a person or a removable volume. ``/Users`` and
#: ``/home`` were the only two round 19 knew; a macOS temp directory lives
#: under ``/private/var/folders``, an external disk under ``/Volumes``, and a
#: Windows profile under ``C:\Users``. ``/var/log`` is deliberately absent:
#: it names no one.
_PERSONAL_PATH_RE = re.compile(
    r"""(?xi)
    (?: [a-z]:[\\/] | / )
    (?: users | home | root | private | volumes | mnt | media | srv
      | var[\\/]folders | tmp )
    [\\/] [^\s"'<>|`]*
    """)

#: What the replay refuses to publish, whoever signed the file. Deliberately
#: only the high-confidence shapes: a bare word like "password" in a sentence
#: is not a leak, and refusing it would refuse honest evidence.
_LEAK_CHECKS = ((_SECRET_RE, "credential"), (_ORG_RE, "credential"),
                (_EMAIL_RE, "credential"), (_AUTH_VALUE_RE, "credential"),
                (_CREDENTIAL_ASSIGNMENT_RE, "credential"),
                (_PERSONAL_PATH_RE, "absolute path"))

#: What each label is replaced with on the way out. The writer substitutes
#: every pattern in :data:`_LEAK_CHECKS` -- iterated from that table rather
#: than listed again -- so "what the replay refuses" and "what the writer
#: removes" cannot drift into being two different sets.
_LEAK_REPLACEMENT = {"credential": "<redacted>", "absolute path": "<path>"}


def leaked_identifiers(text) -> list[str]:
    """Which kinds of personal identifier survive in ``text``.

    Shared by the redactor's own tests and by the replay, so the two cannot
    drift into disagreeing about what counts as clean.
    """
    found = []
    for pattern, label in _LEAK_CHECKS:
        if pattern.search(str(text or "")) and label not in found:
            found.append(label)
    return found


def _portable(text: str, limit: int = 800) -> str:
    """Strip anything that identifies this machine or its accounts.

    Failure evidence is published; a traceback is not. Absolute paths carry a
    home directory, and a hub error can quote a token. What survives is the
    shape of the failure, which is the part anybody can act on -- so the
    repository- and home-relative remainders are kept as ``<repo>/...`` and
    ``<home>/...`` rather than blanked.
    """
    out = str(text or "")
    # These two first: ROOT itself lives under a personal path, and the
    # generic rule below would swallow the useful relative tail with it.
    for base in (str(ROOT), str(Path.home())):
        out = out.replace(base, "<repo>" if base == str(ROOT) else "<home>")
    for pattern, label in _LEAK_CHECKS:
        out = pattern.sub(_LEAK_REPLACEMENT[label], out)
    # Strictly wider than the table above, and writer-only.
    out = _CREDENTIAL_KEYWORD_RE.sub("<redacted>", out)
    out = " ".join(out.split())
    return out[:limit]


def write_failure_evidence(paths, run: str, *, experiment_id: str, stage: str,
                           exception_type: str, summary: str) -> dict:
    """Publish why a run died, once, immutably, and safe to read.

    Before this, a child that crashed left an exit code and nothing else, so
    the reason had to be reconstructed from a terminal scrollback -- which is
    not evidence, does not survive, and cannot be replayed.
    """
    written = {
        "schema_version": FAILURE_EVIDENCE_VERSION,
        "kind": FAILURE_EVIDENCE_KIND,
        "experiment_id": experiment_id,
        "run_id": run,
        "stage": stage if stage in FAILURE_STAGES else "unknown",
        # A blank reason is not a reason. Whatever the exception offered, the
        # file says something the replay will accept as an account.
        "exception_type": _portable(exception_type, 120) or "UnknownError",
        "summary": _portable(summary) or "the child gave no message",
        "written_at": now_iso(),
    }
    body = {field: written[field] for field in FAILURE_EVIDENCE_FIELDS}
    path = Path(paths["dir"]) / f"{run}.failure.json"
    write_once_json(path, body)
    return {"path": path, "body": body,
            "sha256": sha256_file(path)}


#: Distinguishes "this writer had no field" from "this writer had one and it
#: was null". ``dict.get`` collapses the two, and they mean opposite things:
#: the first is a session written before failure evidence existed, the second
#: is a session that knew about it and recorded nothing.
_ABSENT = object()


#: Extended ISO-8601, with a time *and* an offset. ``fromisoformat`` alone is
#: too generous: it accepts a bare date, a naive datetime and the basic
#: format, and a stamp with no zone cannot be placed against a journal whose
#: every other timestamp is UTC-aware.
_ISO_TIMESTAMP_RE = re.compile(
    r"\A\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?"
    r"(?:Z|[+-]\d{2}:\d{2})\Z")


def _is_iso_timestamp(value) -> bool:
    if not isinstance(value, str) or not _ISO_TIMESTAMP_RE.match(value):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def failure_evidence_problems(session_dir, run: str, finished_body: dict,
                              *, outcome=None) -> list[str]:
    """What the finished event claims about the reason, checked against it.

    Three separate questions, and round 18 only answered the middle one:

    * did this writer even have the field? A session written before it existed
      makes no claim, and a claim nobody made cannot have failed. exp001's b1
      is one of those;
    * does the file still hash to what was recorded?
    * and does the file *mean* anything? A digest only proves nobody edited
      the bytes since. Someone who rewrites the evidence and re-chains the
      journal breaks no digest at all, so the body is validated on its own
      terms -- closed schema, matching identity, a declared stage, a real
      timestamp, and a reason that is not blank.
    """
    session_dir = Path(session_dir)
    recorded = (finished_body or {}).get("failure_evidence_sha256", _ABSENT)
    path = session_dir / f"{run}.failure.json"
    if recorded is _ABSENT:
        # Written before the field existed, so no claim was made -- and a
        # claim nobody made cannot have failed. exp001's b1 is one of these.
        # But a failure file sitting there unpinned is a different thing: no
        # event vouches for it, so anybody could have dropped it in.
        if path.exists():
            return [f"{path.name} exists but no finished event pins it: this "
                    "session predates failure evidence, so nothing here can "
                    "vouch for that file"]
        return []
    if recorded is None:
        problems = []
        if outcome == "nonzero_exit":
            problems.append(
                "the child exited non-zero and the finished event records no "
                "failure evidence: a run that died has to say why")
        if path.exists():
            problems.append(
                f"{path.name} exists but the finished event records no "
                "failure evidence digest for it")
        return problems
    if not is_sha256(recorded):
        return [f"the finished event records failure evidence digest "
                f"{recorded!r}, which is not a sha-256"]
    if not path.exists():
        return [f"the finished event records failure evidence, but "
                f"{path.name} does not exist"]
    if sha256_file(path) != recorded:
        return [f"{path.name} does not match the failure evidence digest the "
                "finished event recorded"]
    try:
        body = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001
        return [f"{path.name} (failure evidence) cannot be read ({exc})"]
    if not isinstance(body, dict):
        return [f"{path.name} (failure evidence) is not an object"]

    name = f"{path.name} (failure evidence)"
    problems = []
    extra = sorted(set(body) - set(FAILURE_EVIDENCE_FIELDS))
    missing = sorted(set(FAILURE_EVIDENCE_FIELDS) - set(body))
    if extra:
        problems.append(f"{name} carries unexpected field(s) "
                        f"{', '.join(extra)}")
    if missing:
        problems.append(f"{name} is missing {', '.join(missing)}")
    if body.get("schema_version") != FAILURE_EVIDENCE_VERSION:
        problems.append(f"{name} declares schema version "
                        f"{body.get('schema_version')!r}, not "
                        f"{FAILURE_EVIDENCE_VERSION}")
    if body.get("kind") != FAILURE_EVIDENCE_KIND:
        problems.append(f"{path.name} is not failure evidence")
    if body.get("experiment_id") != session_dir.name:
        problems.append(f"{name} names experiment "
                        f"{body.get('experiment_id')!r}, not "
                        f"{session_dir.name!r}")
    if body.get("run_id") != run:
        problems.append(f"{name} names run {body.get('run_id')!r}")
    if body.get("stage") not in FAILURE_STAGES:
        problems.append(f"{name} records stage {body.get('stage')!r}")
    if not _is_iso_timestamp(body.get("written_at")):
        problems.append(f"{name} records written_at "
                        f"{body.get('written_at')!r}, which is not a "
                        "timestamp")
    for field in ("exception_type", "summary"):
        value = body.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{name} records a blank {field}")
    # Scanned as the strings themselves, not as JSON: a Windows path arrives
    # here with its backslashes doubled once it has been serialised, and the
    # pattern that has to catch it is the one the redactor uses.
    scanned = "\n".join(v for v in body.values() if isinstance(v, str))
    for leak in leaked_identifiers(scanned):
        problems.append(f"{name} still contains {leak} material: the child "
                        "redacts before it writes, so this file was not "
                        "written by the child")
    return problems


def session_status_rows(st: dict) -> dict:
    """What each run's line should say. One place, so the CLI cannot drift.

    A terminal experiment used to print its unmeasured runs as ``pending``,
    which reads as an invitation: they are not pending, they are unreachable
    in every future boot.
    """
    terminal = bool(st.get("terminal"))
    rows = {}
    for entry in st["plan"]["runs"]:
        run = entry["run_id"]
        if run in st["completed"]:
            rows[run] = "measured"
        elif run in (st.get("terminal") or []):
            rows[run] = "terminal"
        elif run in st["cancelled"]:
            rows[run] = "cancelled"
        elif terminal:
            rows[run] = "blocked (experiment terminal)"
        else:
            rows[run] = "pending"
    return rows


def session_next_hint(st: dict) -> str:
    """What may actually be run next, said plainly."""
    if st.get("terminal"):
        return "none: the experiment is terminal incomplete"
    if st.get("cancelled"):
        return "none: rule R1 cancelled the remaining arms"
    return st.get("next_run") or "none: every planned run has been measured"


def safe_experiment_id(experiment_id) -> str:
    if not isinstance(experiment_id, str) or not SAFE_ID.match(experiment_id):
        raise ValueError(
            f"experiment id {experiment_id!r} is not a safe name: it must "
            "match [a-z0-9][a-z0-9_-]{0,62}. Anything else could escape the "
            "reports directory once it becomes a path.")
    if experiment_id in (".", "..") or "/" in experiment_id \
            or "\\" in experiment_id:
        raise ValueError(f"experiment id {experiment_id!r} is a path, not a name")
    return experiment_id


def session_paths(experiment_id: str, *, root: Path | None = None) -> dict:
    eid = safe_experiment_id(experiment_id)
    base = Path(root) if root is not None else REPORT_DIR
    d = (base / eid).resolve()
    if base.resolve() not in d.parents:
        raise ValueError(f"{d} would sit outside {base}")
    return {"dir": d, "plan": d / "plan.json", "session": d / "session.json",
            "calibration": d / "calibration.json", "events": d,
            "snapshot": d / "source_snapshot", "aggregate": d / "aggregate.json",
            "lock": d / ".lock", "root": base}


# ---------------------------------------------------------------------------
# Float comparison
# ---------------------------------------------------------------------------


def same_value(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def same_sum(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def digest_obj(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The design file is the authority
# ---------------------------------------------------------------------------


def design_sha256(path=DESIGN_JSON) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_constants(path=DESIGN_JSON) -> dict:
    design = json.loads(Path(path).read_text())
    frozen = design.get("frozen_constants")
    if not isinstance(frozen, dict):
        raise ValueError(f"{path} has no frozen_constants block, so there is "
                         "nothing approved to build a plan from")
    return frozen


def check_constants_match_design(path=DESIGN_JSON) -> list[str]:
    """Refuse to run on two sets of numbers.

    The module constants above exist so the code reads clearly. The design
    file is what was approved. If they ever disagree, the honest outcome is a
    hard failure -- not silently preferring one of them, which is how a
    reviewed threshold turns into an unreviewed one.
    """
    f = load_frozen_constants(path)
    problems = []

    def cmp(name, mine, theirs):
        if mine != theirs:
            problems.append(f"{name}: module has {mine!r}, the approved design "
                            f"has {theirs!r}")

    cmp("lengths", list(LENGTHS), f["lengths"])
    cmp("run_ids", list(RUN_IDS), f["run_ids"])
    cmp("condition", CONDITION, f["condition"])
    cmp("pool_pairs", POOL_PAIRS, f["pool_pairs"])
    cmp("rows_per_pair", ROWS_PER_PAIR, f["rows_per_pair"])
    cmp("pool_rows", POOL_ROWS, f["pool_rows"])
    cmp("seed", SEED, f["seed"])
    cmp("window", WINDOW, f["window"])
    cmp("secondary_aggregation", SECONDARY_AGGREGATION, f["secondary_aggregation"])
    cmp("memory_every", MEMORY_EVERY, f["memory_every"])
    cmp("empty_cache_every", EMPTY_CACHE_EVERY, f["empty_cache_every"])
    cmp("min_rows", MIN_ROWS, f["min_rows"])
    cmp("clear_growth_min_calls", CLEAR_GROWTH_MIN_CALLS, f["clear_growth_min_calls"])
    cmp("growth_segment", GROWTH_SEGMENT, f["growth_segment"])
    cmp("loss_tolerance", LOSS_TOLERANCE, f["loss_tolerance"])
    b = f["bands"]
    for name, mine, key in (
            ("Q1_HOLDS_D100", Q1_HOLDS_D100, "Q1_holds_D100_max"),
            ("Q1_HOLDS_DMAX", Q1_HOLDS_DMAX, "Q1_holds_Dmax_max"),
            ("Q1_FAILS_D100", Q1_FAILS_D100, "Q1_fails_D100_min"),
            ("Q1_FAILS_DMAX", Q1_FAILS_DMAX, "Q1_fails_Dmax_min"),
            ("Q2_STABLE_GROWTH", Q2_STABLE_GROWTH, "Q2_stable_growth_max"),
            ("Q2_STABLE_SHARE", Q2_STABLE_SHARE, "Q2_stable_share_max"),
            ("Q2_SCALES_GROWTH", Q2_SCALES_GROWTH, "Q2_scales_growth_min"),
            ("Q2_SCALES_SHARE", Q2_SCALES_SHARE, "Q2_scales_share_min"),
            ("REPEATABLE_MAX", REPEATABLE_MAX, "repeatable_max"),
            ("NOT_REPEATABLE_MIN", NOT_REPEATABLE_MIN, "not_repeatable_min")):
        cmp(name, mine, b[key])
    spec = SafetySpec.from_dict(f["safety"])
    if spec != SafetySpec():
        problems.append("the SafetySpec defaults differ from the approved "
                        "design's safety block")
    if list(f["lengths"]) != [500, 1000, 2000]:
        problems.append(f"the approved lengths are {f['lengths']}, but report "
                        "16 is defined as the 500/1000/2000 prefix plan")
    return problems


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def build_plan(experiment_id: str, *, design_path=DESIGN_JSON) -> dict:
    """Build the immutable plan from the approved design file.

    Lengths are not a parameter. Report 16 *is* the 500/1000/2000 prefix plan;
    a different set of lengths is a different experiment and needs its own
    review, not a keyword argument.
    """
    eid = safe_experiment_id(experiment_id)
    problems = check_constants_match_design(design_path)
    if problems:
        raise ValueError("the module constants disagree with the approved "
                         "design file, so no plan may be built:\n  - "
                         + "\n  - ".join(problems))
    frozen = load_frozen_constants(design_path)
    lengths = list(frozen["lengths"])
    run_ids = list(frozen["run_ids"])

    entries = [{"run_id": rid, "declared_rows": k,
                "condition": frozen["condition"], "position": i}
               for i, (rid, k) in enumerate(zip(run_ids, lengths))]
    plan = {
        "schema_version": 1,
        "kind": "longrun_plan",
        "experiment_id": eid,
        "created_at": now_iso(),
        "design_file": str(Path(design_path).relative_to(ROOT)),
        "design_sha256": design_sha256(design_path),
        "frozen_constants": frozen,   # embedded verbatim, not referenced
        "runs": entries,
        "conditional_stop_rule": {"rule_id": "R1", "text": RULE_R1},
    }
    plan["plan_digest"] = digest_obj({k: v for k, v in plan.items()
                                      if k != "plan_digest"})
    return plan


def plan_digest(plan: dict) -> str:
    return digest_obj({k: v for k, v in plan.items() if k != "plan_digest"})


def prefix_lengths(plan: dict) -> list[int]:
    return [e["declared_rows"] for e in plan["runs"]]


def plan_spec(plan: dict) -> SafetySpec:
    return SafetySpec.from_dict(plan["frozen_constants"]["safety"])


# ---------------------------------------------------------------------------
# Windows and degradation metrics
# ---------------------------------------------------------------------------


def _seconds(per_row: list[dict]) -> list[float]:
    return [float(r["compute_seconds"]) for r in per_row]


def windows_from_per_row(per_row: list[dict], size: int = WINDOW) -> list[dict]:
    secs = _seconds(per_row)
    out = []
    for start in range(0, len(secs) - size + 1, size):
        chunk = secs[start:start + size]
        out.append({"window": len(out), "first_row": per_row[start]["row"],
                    "n_rows": size, "seconds": sum(chunk),
                    "seconds_per_row": sum(chunk) / size})
    return out


def _na(n_rows: int, need: int, why: str | None = None) -> str:
    return (f"not_applicable: {why}" if why
            else f"not_applicable: rows_completed={n_rows} < {need}")


def compute_metrics(per_row: list[dict], *, stop_reason: str | None = None) -> dict:
    """D100, D20 and Dmax, with an explicit reason whenever one is missing.

    A ratio of 0.0 is a real answer -- it means the tail measured zero -- and
    is returned as 0.0. Only a metric that cannot be computed at all becomes
    ``None``, and then it always carries a reason.
    """
    secs = _seconds(per_row)
    n = len(secs)
    out: dict = {"rows_completed": n}
    why = (f"{stop_reason} at row {per_row[-1]['row']}" if stop_reason and per_row
           else None)

    def ratio(numer, denom, key, need):
        if n < need:
            out[key], out[f"{key}_reason"] = None, _na(n, need, why)
            return
        if denom == 0:
            out[key] = None
            out[f"{key}_reason"] = ("not_applicable: the baseline window "
                                    "measured zero seconds, so a ratio is "
                                    "undefined")
            return
        out[key], out[f"{key}_reason"] = numer / denom, None

    head = sum(secs[:100]) / 100 if n >= 100 else None
    ratio(sum(secs[-100:]) / 100 if n >= 200 else 0.0,
          head if head is not None else 0.0, "D100", MIN_ROWS["D100"])
    if n >= MIN_ROWS["D20"]:
        w = windows_from_per_row(per_row, WINDOW)
        ratio(w[-1]["seconds_per_row"], w[0]["seconds_per_row"], "D20",
              MIN_ROWS["D20"])
    else:
        out["D20"], out["D20_reason"] = None, _na(n, MIN_ROWS["D20"], why)
    if n >= MIN_ROWS["Dmax"]:
        w = windows_from_per_row(per_row, WINDOW)
        ratio(max(x["seconds_per_row"] for x in w), head, "Dmax",
              MIN_ROWS["Dmax"])
    else:
        out["Dmax"], out["Dmax_reason"] = None, _na(n, MIN_ROWS["Dmax"], why)
    return out


def mean_seconds_per_row(per_row: list[dict], a: int, b: int) -> float | None:
    rows = {r["row"]: float(r["compute_seconds"]) for r in per_row}
    if not all(i in rows for i in range(a, b + 1)):
        return None
    return sum(rows[i] for i in range(a, b + 1)) / (b - a + 1)


def repeatability(values: list[float | None]) -> dict:
    usable = [v for v in values if v is not None]
    if len(usable) < 2:
        return {"R": None, "verdict": "not_applicable",
                "reason": "not_applicable: fewer than two runs cover this span"}
    lo, hi = min(usable), max(usable)
    if lo <= 0:
        return {"R": None, "verdict": "not_applicable",
                "reason": "not_applicable: a non-positive mean"}
    r = hi / lo
    verdict = ("repeatable" if r <= REPEATABLE_MAX else
               "not_repeatable" if r >= NOT_REPEATABLE_MIN else "indeterminate")
    return {"R": r, "verdict": verdict, "reason": None, "n": len(usable)}


# ---------------------------------------------------------------------------
# The treatment contract
# ---------------------------------------------------------------------------


def expected_clear_rows(k: int, every: int = EMPTY_CACHE_EVERY) -> list[int]:
    """``[10, 20, ..., k]`` -- bounded by this run's k, not by 2,000."""
    return list(range(every, k + 1, every))


def clear_contract(per_row: list[dict], *, declared_rows: int,
                   recorded_total: float | None,
                   per_call: list[dict] | None,
                   end_to_end_seconds: float | None,
                   every: int = EMPTY_CACHE_EVERY) -> dict:
    problems: list[str] = []
    observed = [r["row"] for r in per_row if r.get("cleared")]
    completed = len(per_row)
    expected = [row for row in expected_clear_rows(declared_rows, every)
                if row <= completed]
    if observed != expected:
        problems.append(f"clears happened at {observed[:8]}… but the schedule "
                        f"says {expected[:8]}…")

    from_rows = sum(float(r.get("clear_seconds") or 0.0) for r in per_row
                    if r.get("cleared"))
    calls = [float(c["seconds"]) for c in (per_call or [])]
    if per_call is not None:
        if [c["row"] for c in per_call] != observed:
            problems.append("the per-call rows do not match the rows that "
                            "actually cleared")
        if not same_sum(sum(calls), from_rows):
            problems.append(f"per-call seconds sum to {sum(calls)} but the "
                            f"row spans give {from_rows}")
    if recorded_total is not None and not same_sum(from_rows, recorded_total):
        problems.append(f"the recorded clear total {recorded_total} does not "
                        f"match the row spans {from_rows}")

    growth, growth_reason = None, None
    if len(calls) >= CLEAR_GROWTH_MIN_CALLS:
        first = sum(calls[:GROWTH_SEGMENT]) / GROWTH_SEGMENT
        last = sum(calls[-GROWTH_SEGMENT:]) / GROWTH_SEGMENT
        if first == 0:
            growth_reason = ("not_applicable: the first 20 calls measured zero "
                             "seconds, so a ratio is undefined")
        else:
            growth = last / first
    else:
        growth_reason = (f"not_applicable: clear_calls={len(calls)} < "
                         f"{CLEAR_GROWTH_MIN_CALLS}")

    share, share_reason = None, None
    if end_to_end_seconds is None:
        share_reason = "not_applicable: no end-to-end seconds recorded"
    elif end_to_end_seconds == 0:
        share_reason = ("not_applicable: end-to-end measured zero seconds, so "
                        "a share is undefined")
    else:
        share = from_rows / end_to_end_seconds

    return {"problems": problems, "clears_observed_at": observed,
            "clears_expected_at": expected, "clear_calls": len(calls),
            "clear_seconds_from_rows": from_rows,
            "clear_growth": growth, "clear_growth_reason": growth_reason,
            "clear_share": share, "clear_share_reason": share_reason}


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------


def clock_nesting_problems(clocks: dict) -> list[str]:
    problems = []
    load = clocks.get("model_load_seconds")
    cond = clocks.get("condition_clock_seconds")
    proc = clocks.get("process_clock_seconds")
    for name, value in (("model_load_seconds", load),
                        ("condition_clock_seconds", cond),
                        ("process_clock_seconds", proc)):
        if not finite(value):
            problems.append(f"{name} is missing or not a finite number, so "
                            "the clocks cannot be checked")
    if problems:
        return problems
    if cond > proc:
        problems.append(f"condition_clock {cond}s exceeds process_clock {proc}s, "
                        "but it is a sub-interval of it")
    if load + cond > proc + 1e-6:
        problems.append(f"model_load {load}s + condition_clock {cond}s exceeds "
                        f"process_clock {proc}s")
    if "gate_wait_seconds" in clocks:
        problems.append("gate wait belongs to the parent, before the child "
                        "exists; it must not be recorded among the child's "
                        "clocks")
    return problems


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def q1_run(metrics: dict, *, safety_reason: str | None = None,
           tool_failure: str | None = None) -> dict:
    if tool_failure:
        return {"verdict": None, "reason": f"tool_failure: {tool_failure}"}
    if safety_reason:
        if not is_safety_reason(safety_reason):
            return {"verdict": None, "reason": f"tool_failure: {safety_reason}"}
        return {"verdict": "fails", "reason": f"safety_stop:{safety_reason}"}

    d100, dmax = metrics.get("D100"), metrics.get("Dmax")
    if d100 is not None and d100 >= Q1_FAILS_D100:
        return {"verdict": "fails", "reason": f"D100={d100:.4f} >= {Q1_FAILS_D100}"}
    if dmax is not None and dmax >= Q1_FAILS_DMAX:
        return {"verdict": "fails", "reason": f"Dmax={dmax:.4f} >= {Q1_FAILS_DMAX}"}
    if d100 is None or dmax is None:
        missing = "D100" if d100 is None else "Dmax"
        return {"verdict": "not_applicable",
                "reason": metrics.get(f"{missing}_reason")
                or f"not_applicable: {missing} could not be computed"}
    if d100 <= Q1_HOLDS_D100 and dmax <= Q1_HOLDS_DMAX:
        return {"verdict": "holds", "reason": None}
    return {"verdict": "indeterminate",
            "reason": f"D100={d100:.4f}, Dmax={dmax:.4f} fall in the middle band"}


def q2_run(contract: dict, *, safety_reason: str | None = None,
           tool_failure: str | None = None) -> dict:
    if tool_failure:
        return {"verdict": None, "reason": f"tool_failure: {tool_failure}",
                "descriptive_only": True}
    growth, share = contract.get("clear_growth"), contract.get("clear_share")
    descriptive = bool(safety_reason)
    if growth is None:
        return {"verdict": "not_applicable",
                "reason": contract.get("clear_growth_reason")
                or "not_applicable: clear_growth could not be computed",
                "descriptive_only": True}
    if growth >= Q2_SCALES_GROWTH or (share is not None and share >= Q2_SCALES_SHARE):
        return {"verdict": "scales", "reason": None, "descriptive_only": descriptive}
    if growth <= Q2_STABLE_GROWTH and share is not None and share <= Q2_STABLE_SHARE:
        return {"verdict": "stable", "reason": None, "descriptive_only": descriptive}
    return {"verdict": "indeterminate", "reason": None,
            "descriptive_only": descriptive}


def q1_plan(run_verdicts: dict[str, str | None], lengths: dict[str, int]) -> dict:
    holds = [lengths[r] for r, v in run_verdicts.items() if v == "holds"]
    fails = sorted(lengths[r] for r, v in run_verdicts.items() if v == "fails")
    k_star = max(holds) if holds else None
    k_fail = fails[0] if fails else None
    if k_fail is not None:
        value = "fails"
    elif k_star == 2000:
        value = "holds_to_2000"
    elif k_star == 1000:
        value = "holds_to_1000"
    elif k_star == 500:
        value = "holds_to_500"
    else:
        decided = [v for v in run_verdicts.values() if v is not None]
        value = "indeterminate" if "indeterminate" in decided else "not_applicable"
    return {"value": value, "k_star": k_star, "k_fail": k_fail}


def q2_plan(run_verdicts: dict[str, str | None]) -> dict:
    contributing = [v for v in run_verdicts.values()
                    if v in ("stable", "scales", "indeterminate")]
    if not contributing:
        return {"value": "not_applicable", "contributing": 0}
    if "scales" in contributing:
        value = "scales"
    elif "indeterminate" in contributing:
        value = "indeterminate"
    else:
        value = "stable"
    return {"value": value, "contributing": len(contributing)}


# ---------------------------------------------------------------------------
# Rule R1
# ---------------------------------------------------------------------------


def r1_should_cancel(*, outcome: str, tool_failure: str | None,
                     q1_verdict: str | None, q1_reason: str | None) -> dict:
    if outcome != "completed":
        return {"cancel": False,
                "why": f"outcome is {outcome!r}, so this is terminal "
                       "incomplete, not an R1 cancellation"}
    if tool_failure:
        return {"cancel": False,
                "why": f"tool failure {tool_failure!r} produces no Q1 verdict, "
                       "so R1 has nothing to act on"}
    if q1_verdict != "fails":
        return {"cancel": False, "why": f"Q1 is {q1_verdict!r}, not 'fails'"}
    reason = q1_reason or ""
    if not (reason.startswith("safety_stop:") or reason.startswith("D100=")
            or reason.startswith("Dmax=")):
        return {"cancel": False,
                "why": f"a 'fails' verdict caused by {reason!r} is not a real "
                       "safety trip or a computable D100/Dmax failure"}
    return {"cancel": True, "why": reason}


def plan_arm_cancelled_event(*, experiment_id: str, digest: str, run_id: str,
                             outcome: str, verdict: dict, cancelled: list[str],
                             report_sha256: str | None = None,
                             watchdog_sha256: str | None = None) -> dict:
    """The cancellation names the exact evidence it was recomputed from.

    Without the watchdog digest, "R1 fired because a safety threshold tripped"
    is a claim with nothing behind it: the log that recorded the trip could be
    replaced, or absent, and the replay would have nothing to disagree with.
    """
    return {"rule_id": "R1", "triggered_by_run": run_id,
            "triggering_run_outcome": outcome,
            "recomputed_verdict": verdict,
            "verdict_source": "recomputed by the parent from the child's "
                              "per_row and watchdog.jsonl",
            "report_sha256": report_sha256,
            "watchdog_sha256": watchdog_sha256,
            "cancelled_runs": list(cancelled)}


# ---------------------------------------------------------------------------
# Prefix consistency
# ---------------------------------------------------------------------------


def prefix_consistency(reports: list[dict]) -> dict:
    """Compare shared prefixes row for row.

    A missing loss is not a pass. Two runs whose losses cannot be compared
    have not been shown to agree, and "we could not check" must never render
    as "checked and fine".
    """
    if len(reports) < 2:
        return {"verdict": "not_applicable",
                "reason": "only one run exists; nothing to compare",
                "problems": []}
    problems: list[str] = []
    ordered = sorted(reports, key=lambda r: len(r["per_row"]))
    base = ordered[0]
    for other in ordered[1:]:
        n = len(base["per_row"])
        if len(other["per_row"]) < n:
            problems.append(f"{other['run_id']} is shorter than {base['run_id']}")
            continue
        for i in range(n):
            a, b = base["per_row"][i], other["per_row"][i]
            if a.get("row") != b.get("row"):
                problems.append(f"row index {i}: {a.get('row')} vs {b.get('row')}")
                continue
            for field in ("sample_id", "tokens", "supervised_tokens"):
                if a.get(field) != b.get(field):
                    problems.append(f"row {a['row']}: {field} differs between "
                                    f"{base['run_id']} and {other['run_id']}")
            la, lb = a.get("loss"), b.get("loss")
            if not finite(la) or not finite(lb):
                problems.append(
                    f"row {a['row']}: a loss is missing or not finite in "
                    f"{base['run_id'] if not finite(la) else other['run_id']}, "
                    "so the prefix cannot be shown to agree")
            elif abs(la - lb) > LOSS_TOLERANCE:
                problems.append(
                    f"row {a['row']}: loss differs by {abs(la - lb):.6f} "
                    f"between {base['run_id']} and {other['run_id']}, over the "
                    f"pre-declared tolerance {LOSS_TOLERANCE}")
    return {"verdict": "passed" if not problems else "failed",
            "reason": None, "problems": problems}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def headline_full_B(*, runs: list[dict], boots: list[str], prefix: dict,
                    q1: dict, q2: dict, cross_run_identical: bool,
                    contract_problems: list[str]) -> dict:
    reasons: list[str] = []
    if len(runs) != len(RUN_IDS):
        reasons.append(f"{len(runs)} of {len(RUN_IDS)} runs are present")
    not_completed = [r["run_id"] for r in runs if r.get("outcome") != "completed"]
    if not_completed:
        reasons.append(f"these runs did not complete: {not_completed}")
    tooling = [r["run_id"] for r in runs if r.get("tool_failure")]
    if tooling:
        reasons.append(f"tool failures in {tooling}")
    tripped = [r["run_id"] for r in runs if r.get("safety_reason")]
    if tripped:
        reasons.append(f"safety thresholds tripped in {tripped}")
    if len(set(boots)) != len(RUN_IDS) or len(boots) != len(RUN_IDS):
        reasons.append(f"{len(set(boots))} distinct boots across {len(boots)} runs")
    if not cross_run_identical:
        reasons.append("cross-run provenance fields differ")
    if prefix.get("verdict") != "passed":
        reasons.append(f"prefix consistency is {prefix.get('verdict')!r}")
    if contract_problems:
        reasons.append(f"treatment contract problems: {contract_problems}")
    if q1.get("value") != "holds_to_2000":
        reasons.append(f"Q1_plan is {q1.get('value')!r}, not 'holds_to_2000'")
    if q2.get("value") != "stable":
        reasons.append(f"Q2_plan is {q2.get('value')!r}, not 'stable'")
    return {"kind": "headline_full_B", "allowed": not reasons, "reasons": reasons,
            "Q1_plan": q1.get("value"), "Q2_plan": q2.get("value"),
            "adoption_supported": not reasons}


def early_stop_finding(*, run_id: str, k: int, metrics: dict, q1: dict,
                       q2: dict) -> dict:
    reason = q1.get("reason") or ""
    if not (reason.startswith("safety_stop:") or reason.startswith("D100=")
            or reason.startswith("Dmax=")):
        raise ValueError("an early_stop_finding may only be produced by a real "
                         f"safety trip or a computable D100/Dmax failure, not "
                         f"by {reason!r}")
    payload = {"kind": "early_stop_finding", "triggered_by_run": run_id, "k": k,
               "Q1_run": q1.get("verdict"), "reason": reason,
               "metrics": {key: metrics.get(key) for key in
                           ("D100", "D100_reason", "D20", "D20_reason",
                            "Dmax", "Dmax_reason")},
               "Q2_run": q2.get("verdict"), "Q2_is_descriptive_only": True,
               "prefix_consistency": "not_applicable",
               "prefix_consistency_reason": "only one run exists; nothing to "
                                            "compare",
               "adoption_supported": False}
    # Zero is a legitimate reading. What is not legitimate is a value sitting
    # next to a reason that says it could not be computed.
    for key in ("D100", "D20", "Dmax"):
        if payload["metrics"].get(f"{key}_reason") and \
                payload["metrics"].get(key) is not None:
            raise ValueError(f"{key} carries a not_applicable reason and a "
                             f"value ({payload['metrics'][key]!r}); an "
                             "uncomputable metric must be null")
    return payload


# ---------------------------------------------------------------------------
# Launch sequence
# ---------------------------------------------------------------------------


@dataclass
class LaunchOutcome:
    ok: bool
    stage: str
    boot_consumed: bool
    event: str | None = None
    reason: str | None = None
    terminal: bool = False
    detail: str | None = None
    cleanup: list[str] | None = None


def launch_sequence(*, gate_ok: bool, sources_ok: bool, start_watchdog,
                    write_started, spawn_child, hand_identity, await_armed,
                    cleanup=None) -> LaunchOutcome:
    """gate → sources → watchdog ready → started → spawn → identity → armed.

    Two properties matter most. The watchdog must be ready *before*
    ``measurement_started`` is written, because writing that event is what
    spends the boot -- a watchdog that failed to start is retryable, a boot
    spent on an unwatched run is not. And every failure after a child exists
    runs ``cleanup``, because the worst outcome of a tool failure is a child
    that keeps training with nobody watching and nobody recording.
    """
    cleanup = cleanup or (lambda: [])
    if not gate_ok:
        return LaunchOutcome(False, "gate", False, event=EVENT_GATE_ATTEMPT,
                             reason="the gate did not release")
    if not sources_ok:
        return LaunchOutcome(False, "sources", False, event=EVENT_PRE_SPAWN_ABORT,
                             reason="the source snapshot no longer matches")

    ready = start_watchdog()
    if not ready.get("ready"):
        return LaunchOutcome(False, "watchdog_ready", False,
                             event=EVENT_WATCHDOG_LAUNCH_FAILED,
                             reason=ready.get("reason") or "watchdog not ready",
                             detail="retryable in the same boot",
                             cleanup=cleanup())

    write_started()  # *** the boot is consumed here ***

    child = spawn_child()
    if not child.get("spawned"):
        return LaunchOutcome(False, "spawn", True, reason="spawn_failed",
                             terminal=True, cleanup=cleanup(),
                             detail="measurement_started was already written, "
                                    "so the boot is spent")
    handed = hand_identity(child)
    if not handed.get("ok"):
        return LaunchOutcome(False, "identity", True, reason="identity_mismatch",
                             terminal=True, detail=handed.get("reason"),
                             cleanup=cleanup())
    armed = await_armed()
    if not armed.get("armed"):
        return LaunchOutcome(False, "armed", True, reason="armed_timeout",
                             terminal=True, detail=armed.get("reason"),
                             cleanup=cleanup())
    return LaunchOutcome(True, "running", True)


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

CHILD_SCHEMA_VERSION = 1
CHILD_KIND = "longrun_child"

#: Section 4.7's schema, in one closed list. Every key here is written by
#: :func:`run_child` and required by :func:`replay_child`, and nothing outside
#: it may appear in a report.
#:
#: Both halves matter. A report that simply omits the number it got wrong has
#: to fail, so absence is a problem in its own right rather than something the
#: recomputation quietly skips over. And a report carrying a field nobody
#: declared is carrying a number no replay recomputes, which is how a
#: conclusion gets smuggled in beside the evidence -- the aggregate comparison
#: has refused unknown keys since round three, and the child report is the
#: document every verdict is actually derived from.
CHILD_SCHEMA_KEYS = (
    "schema_version", "kind", "experiment_id", "run_id", "declared_rows",
    "condition", "plan_digest", "nonce", "pool_pairs", "pool_rows",
    "child_source_check", "preflight", "child_pid", "child_pgid",
    "child_start_identity", "started_at", "finished_at", "rows_requested",
    "rows_completed", "stopped_early", "tool_failure", "input_order_digest",
    "completed_input_digest", "provenance", "per_row", "memory", "metrics",
    "scheduled_empty_cache_every", "scheduled_empty_cache_cost",
    "teardown_empty_cache_calls", "teardown_empty_cache_seconds",
    "end_to_end_seconds", "model_compute_seconds",
    "between_row_overhead_breakdown", "clocks", "float_storage")

#: An alias, so the "must be present" loop and the "nothing else" check can
#: never fall out of step with each other.
CHILD_REQUIRED = CHILD_SCHEMA_KEYS

#: The blocks that are objects, and what has to be inside them. A present but
#: empty block is the same evasion as a missing one, one level down.
CHILD_NESTED_REQUIRED = {
    "child_source_check": ("plan_digest", "source_manifest_digest",
                           "files_verified"),
    "preflight": GATE_METRICS,
    "float_storage": ("seconds_rounded", "loss_rounded"),
    "between_row_overhead_breakdown": ("scheduled_empty_cache_seconds",
                                       "memory_probe_seconds",
                                       "unattributed_seconds"),
    "scheduled_empty_cache_cost": ("calls", "total_seconds", "per_call"),
    "clocks": ("model_load_seconds", "condition_clock_seconds",
               "process_clock_seconds"),
    "provenance": PROVENANCE_FIELDS,
}

# Section 4.8 7b's four fields (``IDENTITY_FIELDS``) are imported from
# src.training.watchdog, beside the ``ChildIdentity`` that writes them, so the
# launch record, the child report and the watchdog log cannot end up comparing
# three different sets of keys.

#: The whole of ``stopped_early`` when it is not null (§4.7). ``reason`` is the
#: design's ``condition`` field, renamed only because the report already has a
#: top-level ``condition`` meaning the arm; every verdict function reads
#: ``reason``, so storing both would be two names for one fact.
STOPPED_EARLY_FIELDS = ("reason", "rule", "row", "sampled_values",
                        "condition_clock_seconds", "process_clock_seconds",
                        "requested_by", "stop_request_sha256")

REQUESTED_BY = ("watchdog", "child")


GATE_POLICY = {"consecutive_passes_required": 3, "poll_interval_seconds": 30,
               "timeout_seconds": 900}


def validate_calibration(calib: dict) -> list[str]:
    """A calibration is only usable if all three parts are there and agree.

    Samples without thresholds cannot gate anything; thresholds without
    samples cannot be recomputed, so nobody can tell whether they were derived
    or typed in; and a policy is what turns a single passing reading into
    three consecutive ones.
    """
    from src.training.preflight import GATE_SPEC, calibrate, thresholds_from

    problems = []
    samples = calib.get("samples")
    if not isinstance(samples, list) or not samples:
        problems.append("the calibration has no samples, so its thresholds "
                        "cannot be recomputed")
    stored = calib.get("thresholds")
    if not isinstance(stored, dict) or not stored:
        problems.append("the calibration has no thresholds")
    policy = calib.get("policy")
    if not isinstance(policy, dict):
        problems.append("the calibration has no gate policy")
    elif {k: policy.get(k) for k in GATE_POLICY} != GATE_POLICY:
        problems.append(f"the gate policy is {policy!r}, not the fixed "
                        f"{GATE_POLICY!r}")
    if problems:
        return problems
    missing = [k for k in GATE_SPEC if k not in stored]
    if missing:
        problems.append(f"the thresholds are missing {missing}")
    recomputed = thresholds_from(calibrate(samples))
    if recomputed != stored:
        problems.append("the stored thresholds do not recompute from the "
                        "samples")
    return problems


def replay_gate_polls(gate: dict, thresholds: dict, policy: dict) -> list[str]:
    """Recompute every poll: streak, passed, waited and the timeout."""
    from src.training.preflight import evaluate_gate

    problems = []
    polls = gate.get("polls")
    if not isinstance(polls, list) or not polls:
        return ["the gate record has no polls to replay"]
    streak, passed_at = 0, None
    for i, poll in enumerate(polls):
        for key in ("index", "elapsed_seconds", "sample", "passed"):
            if key not in poll:
                problems.append(f"gate poll {i} is missing {key!r}")
        if problems:
            return problems
        if poll["index"] != i:
            problems.append(f"gate poll at position {i} claims index "
                            f"{poll['index']}")
        again = evaluate_gate(poll["sample"], thresholds)
        if again["passed"] != poll["passed"]:
            problems.append(f"gate poll {i} replays to {again['passed']}, not "
                            f"{poll['passed']}")
        streak = streak + 1 if again["passed"] else 0
        if poll.get("streak") is not None and poll["streak"] != streak:
            problems.append(f"gate poll {i} records streak {poll['streak']} but "
                            f"replay computes {streak}")
        if streak >= policy["consecutive_passes_required"] and passed_at is None:
            passed_at = i
    released = passed_at is not None
    if bool(gate.get("passed")) != released:
        problems.append(f"the gate claims passed={gate.get('passed')} but "
                        f"replaying its polls gives {released}")
    if released:
        waited = polls[passed_at]["elapsed_seconds"]
        if gate.get("waited_seconds") is not None and \
                not same_sum(gate["waited_seconds"], waited):
            problems.append(f"the gate claims it waited {gate['waited_seconds']}s "
                            f"but its polls say {waited}s")
        if waited > policy["timeout_seconds"]:
            problems.append(f"the gate released after {waited}s, past the "
                            f"{policy['timeout_seconds']}s timeout")
    return problems


def load_usable_calibration(calibration_path) -> dict:
    """Read and validate a calibration before anything exists to clean up.

    ``None`` used to reach ``Path(None)`` and raise TypeError from inside a
    session directory that had already been created; a missing file and
    unparseable JSON did the same thing one line later. The failure is the
    same in every case -- there is no calibration -- so it is found in one
    place, and found early.
    """
    if calibration_path is None:
        raise ValueError(
            "no calibration was given. --session-init needs the "
            "calibration.json its gate thresholds were derived from.")
    path = Path(calibration_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    try:
        calib = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} cannot be read as JSON ({exc})") from exc
    if not isinstance(calib, dict):
        raise ValueError(f"{path} is not a calibration object")
    bad = validate_calibration(calib)
    if bad:
        raise ValueError("the calibration cannot be used:\n  - "
                         + "\n  - ".join(bad))
    return calib


def session_init(experiment_id: str, *, calibration_path, root=None,
                 design_path=DESIGN_JSON, code_files=CODE_FILES) -> dict:
    """Create the immutable session. Nothing exists until everything passes.

    The order is the whole point. The directory used to be created first, so
    a calibration that was missing, unreadable or simply absent left an empty
    experiment behind -- and an empty experiment has spent its id, because
    sessions are never reopened. So every precondition that can be checked
    without writing is checked without writing: the id, the design and the
    plan built from it, the calibration, and the existence of every source
    file the snapshot will need. Only then does anything appear on disk.
    """
    paths = session_paths(experiment_id, root=root)      # 1. a safe id
    if paths["dir"].exists():
        raise FileExistsError(
            f"{paths['dir']} already exists. Sessions are never reopened: a "
            "new experiment id is the only way forward.")
    plan = build_plan(experiment_id, design_path=design_path)  # 2. design
    calib = load_usable_calibration(calibration_path)          # 3. calibration
    missing = [rel for rel in code_files if not (ROOT / rel).exists()]
    if missing:                                                # 4. sources
        raise FileNotFoundError(
            "the source snapshot cannot be taken, so this plan would have no "
            f"record of the code it was made against: {', '.join(missing)}")

    # 5. Everything above passed, so the experiment may exist.
    paths["dir"].mkdir(parents=True)
    write_once_json(paths["plan"], plan)
    copy_once(calibration_path, paths["calibration"])
    manifest = snapshot_sources(ROOT, code_files, paths["snapshot"])
    session = {
        "schema_version": 1, "kind": "longrun_session",
        "gate_policy": dict(GATE_POLICY),
        "thresholds": calib["thresholds"],
        "experiment_id": plan["experiment_id"],
        "created_at": now_iso(),
        "one_run_per_boot": True,
        "plan_digest": plan["plan_digest"],
        "design_sha256": plan["design_sha256"],
        "calibration_sha256": sha256_file(paths["calibration"]),
        "source_manifest": manifest,
        "source_manifest_digest": manifest_digest(manifest),
    }
    write_once_json(paths["session"], session)
    return {"paths": paths, "plan": plan, "session": session}


def load_session(experiment_id: str, *, root=None) -> dict:
    paths = session_paths(experiment_id, root=root)
    problems = []
    if not paths["dir"].exists():
        return {"problems": [f"{paths['dir']} does not exist"], "paths": paths}
    for key in ("plan", "session"):
        if not paths[key].exists():
            problems.append(f"{paths[key].name} is missing")
    if problems:
        return {"problems": problems, "paths": paths}
    plan = json.loads(paths["plan"].read_text())
    session = json.loads(paths["session"].read_text())
    if plan_digest(plan) != plan.get("plan_digest"):
        problems.append("plan.json does not hash to its own recorded digest")
    if session.get("plan_digest") != plan.get("plan_digest"):
        problems.append("session.json records a different plan digest")
    if plan.get("experiment_id") != paths["dir"].name:
        problems.append("plan.json names another experiment than its directory")
    return {"problems": problems, "paths": paths, "plan": plan,
            "session": session, "events": read_journal(paths["dir"])}


def validate_journal(events: list[dict], plan: dict) -> list[str]:
    """Read the journal fail-closed before anything is derived from it.

    Everything here exists because the alternative is a session that continues
    from a state nobody can account for: a renumbered file, a body that names
    another experiment, a finished event pointing at somebody else's start.
    """
    problems: list[str] = []
    order = [e["run_id"] for e in plan["runs"]]
    by_digest = {e["file_sha256"]: e for e in events}
    started, finished, seen_index = {}, {}, []

    for i, ev in enumerate(events):
        name, body = ev["file_name"], ev.get("body") or {}
        where = f"event {name}"
        if ev["index"] is None or ev["run_id"] is None or ev["event"] is None:
            problems.append(f"{where} does not follow NN-run-event.json")
            continue
        seen_index.append(ev["index"])
        if ev["index"] != i + 1:
            problems.append(f"{where} is numbered {ev['index']} but sits at "
                            f"position {i + 1}: the journal was renumbered, "
                            "reordered or has a gap")
        for key in EVENT_BODY_REQUIRED:
            if key not in body:
                problems.append(f"{where} body is missing {key!r}")
        if body.get("schema_version") != JOURNAL_SCHEMA:
            problems.append(f"{where} has schema_version "
                            f"{body.get('schema_version')!r}")
        if body.get("event") != ev["event"]:
            problems.append(f"{where} body says event {body.get('event')!r}")
        if body.get("index") != ev["index"]:
            problems.append(f"{where} body says index {body.get('index')!r}")
        if body.get("run_id") != ev["run_id"]:
            problems.append(f"{where} body says run {body.get('run_id')!r}")
        if body.get("experiment_id") != plan["experiment_id"]:
            problems.append(f"{where} names experiment "
                            f"{body.get('experiment_id')!r}")
        if body.get("plan_digest") != plan["plan_digest"]:
            problems.append(f"{where} carries plan digest "
                            f"{str(body.get('plan_digest'))[:16]}…")
        if ev["event"] not in EVENT_KINDS:
            problems.append(f"{where} is not a known journal kind")
        if ev["run_id"] not in order:
            problems.append(f"{where} names a run outside the plan")
        want_prev = events[i - 1]["file_sha256"] if i else None
        if body.get("previous_event_digest") != want_prev:
            problems.append(f"{where} points back at "
                            f"{str(body.get('previous_event_digest'))[:16]}… but "
                            f"the previous event hashes to {str(want_prev)[:16]}…")

        if ev["event"] == EVENT_STARTED:
            if ev["run_id"] in started:
                problems.append(f"{ev['run_id']} has two measurement_started "
                                "events")
            started[ev["run_id"]] = ev
        elif ev["event"] == EVENT_FINISHED:
            if ev["run_id"] in finished:
                problems.append(f"{ev['run_id']} has two measurement_finished "
                                "events")
            finished[ev["run_id"]] = ev
            if ev["run_id"] not in started:
                problems.append(f"{ev['run_id']} finished without starting")
                continue
            begin = started[ev["run_id"]]
            if body.get("started_index") != begin["index"]:
                problems.append(f"{where} records started_index "
                                f"{body.get('started_index')!r}, not "
                                f"{begin['index']}")
            digest = body.get("started_event_digest")
            if digest != begin["file_sha256"]:
                named = by_digest.get(digest)
                extra = (f" (that digest is {named['file_name']})" if named
                         else "")
                problems.append(f"{where} points back at another started event"
                                f"{extra}")

    if len(set(seen_index)) != len(seen_index):
        problems.append("the journal has duplicate event indices")

    measured = [r for r in order if r in started]
    if measured != [r for r in order if r in started]:
        problems.append("measured runs are out of plan order")
    positions = [order.index(r) for r in
                 [ev["run_id"] for ev in events if ev["event"] == EVENT_STARTED]]
    if positions != sorted(positions):
        problems.append("runs were measured out of the order the plan declares")
    return problems


def session_preconditions(experiment_id: str, *, root=None,
                          check_working_tree: bool = True) -> list[str]:
    """Everything that must already be true before a run may be attempted.

    Run inside the lock and *before* the gate. Polling the gate first would
    mean spending up to fifteen minutes of the operator's evening on a session
    we were never going to be allowed to extend -- and, worse, writing a
    gate_attempt event about a session whose own records do not add up.
    """
    st = session_state(experiment_id, root=root)
    if "plan" not in st:
        return st.get("problems", [f"no session {experiment_id!r}"])
    problems = list(st["problems"])
    paths, plan, session = st["paths"], st["plan"], st["session"]

    # calibration: unchanged, complete, and its thresholds still recompute
    calib: dict = {}
    if not paths["calibration"].exists():
        problems.append("calibration.json is missing")
    else:
        if sha256_file(paths["calibration"]) != session.get("calibration_sha256"):
            problems.append("calibration.json has changed since init")
        calib = json.loads(paths["calibration"].read_text())
        problems += validate_calibration(calib)
        if calib.get("thresholds") != session.get("thresholds"):
            problems.append("session.json records different thresholds from "
                            "the calibration")
    if session.get("gate_policy") != GATE_POLICY:
        problems.append(f"the session's gate policy is "
                        f"{session.get('gate_policy')!r}, not {GATE_POLICY!r}")

    # plan and session identity
    if plan_digest(plan) != plan.get("plan_digest"):
        problems.append("plan.json does not hash to its recorded digest")
    if session.get("plan_digest") != plan.get("plan_digest"):
        problems.append("session.json records another plan digest")
    if plan.get("experiment_id") != paths["dir"].name:
        problems.append("plan.json names another experiment than its directory")

    # the frozen source snapshot
    problems += verify_sources(ROOT, session["source_manifest"],
                               paths["snapshot"],
                               check_working_tree=check_working_tree)
    if manifest_digest(session["source_manifest"]) != \
            session.get("source_manifest_digest"):
        problems.append("the source manifest does not hash to its digest")

    # the gate every earlier attempt recorded, replayed poll by poll
    problems += gate_replay_problems(st["events"],
                                     (calib.get("thresholds") or {}))

    # every run already finished has to still stand up
    spec = plan_spec(plan)
    for entry in plan["runs"]:
        run = entry["run_id"]
        if run not in st["finished"]:
            continue
        body = st["finished"][run].get("body") or {}
        started_body = (st["started"].get(run) or {}).get("body") or {}
        rp = paths["dir"] / f"{run}.json"
        wpath = paths["dir"] / f"{run}.watchdog.jsonl"
        if body.get("outcome") != "completed":
            continue
        if not rp.exists():
            problems.append(f"{run} completed but {rp.name} is gone")
            continue
        # A digest that is merely absent used to pass here: the checks were
        # `if body.get(...)`, so a finished event that recorded no digest was
        # indistinguishable from one whose digest still matched. For a run the
        # journal calls completed, the digests are mandatory evidence.
        if not body.get("report_sha256"):
            problems.append(f"{run} completed but its finished event records "
                            "no report digest")
        elif sha256_file(rp) != body["report_sha256"]:
            problems.append(f"{rp.name} no longer matches its recorded digest")
        if not body.get("watchdog_sha256"):
            problems.append(f"{run} completed but its finished event records "
                            "no watchdog log digest")
        if not wpath.exists():
            problems.append(f"{run} completed but {wpath.name} is gone")
        elif body.get("watchdog_sha256") and \
                sha256_file(wpath) != body["watchdog_sha256"]:
            problems.append(f"{wpath.name} no longer matches its digest")
        # The same validator the pre-finish gate uses, so a session cannot
        # pass one and fail the other.
        problems += [f"{run}: {p}" for p in completed_run_evidence(
            paths, run, spec=spec, plan=plan, entry=entry, session=session,
            started_body=started_body, finished_body=body)["problems"]]
    return problems


def gate_replay_problems(events: list[dict], thresholds: dict) -> list[str]:
    """Every event that carries a gate record has to replay against the bands.

    Shared by the precondition replay and by ``--verify`` so the two cannot
    drift: a session that would fail verification afterwards must not be
    allowed to spend another boot first.
    """
    problems: list[str] = []
    for ev in events:
        if ev.get("event") not in (EVENT_STARTED, EVENT_GATE_ATTEMPT):
            continue
        gate = (ev.get("body") or {}).get("gate")
        if not isinstance(gate, dict):
            problems.append(f"{ev.get('file_name')} records no gate to replay")
        elif thresholds:
            problems += [f"{ev.get('file_name')}: {p}" for p in
                         replay_gate_polls(gate, thresholds, GATE_POLICY)]
    return problems


def launch_identity_problems(session_dir, run: str, started_body: dict, *,
                             report: dict | None = None,
                             finished_body: dict | None = None) -> list[str]:
    """Section 4.8 7b: one process, described twice, in agreement.

    The launch record is the parent's account of who the child was; the child
    report is the child's own. Requiring the file to *exist* -- which is all
    the previous round did -- proves only that somebody wrote a file: every
    field in it could name a different process and nothing would notice.

    So all four fields are compared across the two accounts, and the finished
    event has to carry the launch record's digest. Without that digest the
    record is a note rather than evidence, and the comparison could be made to
    agree afterwards by editing whichever side was inconvenient.
    """
    path = Path(session_dir) / f"{run}.launch.json"
    if not path.exists():
        return [f"{path.name} is missing, so nothing independent records which "
                "process was measured"]
    try:
        body = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:  # noqa: BLE001
        return [f"{path.name} cannot be read ({exc})"]
    problems = [f"{path.name} is missing {key!r}" for key in IDENTITY_FIELDS
                if key not in body]
    if body.get("run_id") != run:
        problems.append(f"{path.name} names run {body.get('run_id')!r}")

    if finished_body is not None:
        recorded = finished_body.get("launch_sha256")
        if not recorded:
            problems.append(f"the finished event records no launch digest, so "
                            f"{path.name} could be rewritten without anything "
                            "noticing")
        elif sha256_file(path) != recorded:
            problems.append(f"{path.name} does not match the launch digest the "
                            "finished event recorded")

    want = started_body.get("nonce")
    if want is None:
        problems.append("the started event recorded no launch nonce for the "
                        "launch record to be checked against")
    elif body.get("nonce") != want:
        problems.append(f"{path.name} carries another launch's nonce")

    if report is not None:
        for field in IDENTITY_FIELDS:
            mine, theirs = report.get(field), body.get(field)
            if mine != theirs:
                problems.append(
                    f"{field} is {mine!r} in the child's report and {theirs!r} "
                    f"in {path.name}: the launch record and the process that "
                    "was measured do not describe the same thing")
    return problems


def child_source_check_problems(report: dict, *, session: dict,
                                plan: dict) -> list[str]:
    """What the child says it verified, against what there was to verify.

    ``child_source_check`` is the child's own statement that it re-derived the
    plan digest and the source manifest before loading anything. Storing that
    statement without ever comparing it with the session is the child marking
    its own homework: any three plausible values would pass.
    """
    block = report.get("child_source_check")
    if not isinstance(block, dict) or not block:
        return ["child_source_check is empty or is not an object, so the child "
                "recorded no evidence that it checked its own source"]
    problems = []
    if block.get("plan_digest") != plan.get("plan_digest"):
        problems.append(
            f"child_source_check.plan_digest is "
            f"{str(block.get('plan_digest'))[:16]}…, not this session's plan")
    if block.get("source_manifest_digest") != \
            session.get("source_manifest_digest"):
        problems.append("child_source_check.source_manifest_digest is not the "
                        "one session.json records")
    files = (session.get("source_manifest") or {}).get("files") or {}
    if block.get("files_verified") != len(files):
        problems.append(f"child_source_check.files_verified is "
                        f"{block.get('files_verified')!r}, but the manifest "
                        f"covers {len(files)} files")
    return problems


def session_state(experiment_id: str, *, root=None) -> dict:
    """What the journal says, derived once and used by everything."""
    loaded = load_session(experiment_id, root=root)
    if loaded.get("problems") or "plan" not in loaded:
        return loaded
    plan, events = loaded["plan"], loaded["events"]
    started, finished, cancelled, attempts = {}, {}, [], {}
    boots = {}
    problems = list(loaded["problems"]) + validate_journal(events, plan)
    for ev in events:
        kind, run, body = ev.get("event"), ev.get("run_id"), ev.get("body") or {}
        if kind not in EVENT_KINDS:
            problems.append(f"event file {ev.get('file_name')} is not a known "
                            "journal kind")
        if body.get("experiment_id") not in (None, plan["experiment_id"]):
            problems.append(f"event {ev.get('index')} names another experiment")
        if kind in EVENT_RETRYABLE:
            attempts.setdefault(run, []).append(ev)
        elif kind == EVENT_STARTED:
            if run in started:
                problems.append(f"{run} has two measurement_started events")
            started[run] = ev
            boots[run] = body.get("boot_fingerprint")
        elif kind == EVENT_FINISHED:
            if run not in started:
                problems.append(f"{run} finished without starting")
            finished[run] = ev
        elif kind == EVENT_PLAN_ARM_CANCELLED:
            cancelled += list(body.get("cancelled_runs") or [])
    seen = [b for b in boots.values() if b]
    if len(set(seen)) != len(seen):
        problems.append("two measured runs share a boot fingerprint")
    terminal = [r for r in started if r not in finished]
    for run, ev in finished.items():
        fin = ev.get("body") or {}
        if fin.get("outcome") != "completed" or fin.get("tool_failure"):
            terminal.append(run)
    order = [e["run_id"] for e in plan["runs"]]
    completed = [r for r in order if r in finished and r not in terminal]
    remaining = [r for r in order if r not in started and r not in cancelled]
    return {"problems": problems, "paths": loaded["paths"], "plan": plan,
            "session": loaded["session"], "events": events,
            "started": started, "finished": finished,
            "cancelled": sorted(set(cancelled)), "attempts": attempts,
            "boots": boots, "terminal": sorted(set(terminal)),
            "completed": completed, "remaining": remaining,
            "next_run": remaining[0] if remaining else None,
            "complete": len(completed) == len(order),
            "complete_by_rule": bool(cancelled) and not terminal}


def session_status(experiment_id: str, *, root=None) -> dict:
    st = session_state(experiment_id, root=root)
    if "plan" not in st:
        return st
    st["this_boot"] = boot_identity(st["plan"]["experiment_id"])["boot_fingerprint"]
    st["boot_already_used"] = st["this_boot"] in set(
        b for b in st["boots"].values() if b)
    return st


# Report 16 has two journal kinds report 15 never needed, and report 15's
# writer rejects anything outside its own four. Rather than widen a module
# exp002's replay still executes, report 16 keeps its own writer over the same
# no-clobber primitive.
EVENT_KINDS = (EVENT_GATE_ATTEMPT, EVENT_PRE_SPAWN_ABORT,
               EVENT_WATCHDOG_LAUNCH_FAILED, EVENT_STARTED, EVENT_FINISHED,
               EVENT_PLAN_ARM_CANCELLED)
EVENT_FILE_RE = re.compile(
    r"\A(?P<index>\d{2,})-(?P<run>[a-z0-9_-]+)-(?P<event>[a-z_]+)\.json\Z")


JOURNAL_SCHEMA = 1
EVENT_BODY_REQUIRED = ("schema_version", "kind", "event", "index",
                       "experiment_id", "plan_digest", "run_id", "written_at",
                       "previous_event_digest")


def append_session_event(paths: dict, run_id: str, event: str, body: dict, *,
                         plan: dict) -> dict:
    """One file per event, published once, never rewritten, chained.

    Every event carries its own index, the experiment and plan it belongs to,
    and the digest of the event before it. Any one of those alone can be
    forged; together they make a journal you cannot reorder, renumber or
    quietly extend.
    """
    if event not in EVENT_KINDS:
        raise ValueError(f"unknown journal event {event!r}")
    events_dir = paths["dir"] / "events"
    events_dir.mkdir(exist_ok=True)
    existing = read_journal(paths["dir"])
    index = len(existing) + 1
    envelope = {
        "schema_version": JOURNAL_SCHEMA, "kind": "longrun_event",
        "event": event, "index": index,
        "experiment_id": plan["experiment_id"],
        "plan_digest": plan["plan_digest"], "run_id": run_id,
        "written_at": now_iso(),
        "previous_event_digest": existing[-1]["file_sha256"] if existing else None,
        **body,
    }
    name = f"{index:02d}-{run_id}-{event}.json"
    path = events_dir / name
    write_once_json(path, envelope)
    digest = sha256_file(path)
    return {"path": path, "file_name": name, "index": index,
            "sha256": digest, "file_sha256": digest, "body": envelope}


def read_journal(session_dir) -> list[dict]:
    """Every event on disk, in filename order, with its own digest.

    The digest is computed here rather than trusted from inside the file: a
    later event points back at an earlier one, and that link is only worth
    anything if the digest covers what is actually on disk.
    """
    d = Path(session_dir) / "events"
    if not d.exists():
        return []
    out = []
    for path in sorted(d.glob("*.json")):
        m = EVENT_FILE_RE.match(path.name)
        try:
            body = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            body = {"__unreadable__": str(exc)}
        out.append({"file_name": path.name, "file": path.name,
                    "file_sha256": sha256_file(path),
                    "index": int(m.group("index")) if m else None,
                    "run_id": m.group("run") if m else None,
                    "event": m.group("event") if m else None,
                    "body": body})
    return out


def session_finalize(experiment_id: str, *, root=None) -> dict:
    """Write the aggregate. Refuses while the session could still continue."""
    paths_ = session_paths(experiment_id, root=root)
    if not paths_["dir"].exists():
        return {"problems": [f"{paths_['dir']} does not exist"]}
    with exclusive_lock(paths_["lock"],
                        description=f"report 16 finalize {experiment_id}"):
        return _session_finalize_locked(experiment_id, root=root)


def _session_finalize_locked(experiment_id: str, *, root=None) -> dict:
    st = session_state(experiment_id, root=root)
    if "plan" not in st:
        return st
    if not (st["complete"] or st["terminal"] or st["cancelled"]):
        return {"problems": [
            f"{experiment_id} can still continue: "
            f"{st['remaining']} have not run. Finalising now would record a "
            "conclusion the journal does not support."], "state": st}
    agg = build_aggregate(st)
    paths = st["paths"]
    if paths["aggregate"].exists():
        stored = json.loads(paths["aggregate"].read_text())
        volatile = ("created_at",)
        a = {k: v for k, v in stored.items() if k not in volatile}
        b = {k: v for k, v in agg.items() if k not in volatile}
        if a != b:
            return {"problems": ["the stored aggregate disagrees with a fresh "
                                 "derivation from the journal"], "state": st}
        return {"problems": [], "aggregate": stored, "state": st,
                "idempotent": True}
    write_once_json(paths["aggregate"], agg)
    return {"problems": [], "aggregate": agg, "state": st}


def build_aggregate(st: dict) -> dict:
    """Derive the whole session's verdicts from evidence that was checked.

    The aggregate is the document a reader is handed, so it is the last place
    a weaker definition of ``completed`` can hide. This used to replay the
    child report and nothing else: no digest against the finished event, no
    launch identity, no stop request. Three reports rewritten to name another
    experiment agreed with each other, replayed clean, and produced Q1
    "holds" three times and a headline saying adoption was supported -- while
    ``--verify`` on the same session named nine problems.
    """
    plan, paths = st["plan"], st["paths"]
    spec = plan_spec(plan)
    lengths = {e["run_id"]: e["declared_rows"] for e in plan["runs"]}
    reports, q1s, q2s, runs, contract_problems = [], {}, {}, [], []
    for entry in plan["runs"]:
        run = entry["run_id"]
        rp = paths["dir"] / f"{run}.json"
        if not rp.exists():
            continue
        report = json.loads(rp.read_text())
        reports.append(report)
        body = (st["finished"].get(run) or {}).get("body") or {}
        started_body = (st["started"].get(run) or {}).get("body") or {}
        wpath = paths["dir"] / f"{run}.watchdog.jsonl"
        wsha = body.get("watchdog_sha256")
        if body.get("outcome") == "completed":
            # The same validator the pre-finish gate, the precondition replay,
            # --verify and rule R1 use. Anything less is a second definition
            # of completed, and a second definition is how a run passes one
            # gate and fails another with the boot already spent.
            evidence = completed_run_evidence(
                paths, run, spec=spec, plan=plan, entry=entry,
                session=st["session"], started_body=started_body,
                finished_body=body)
            problems = evidence["problems"]
            replay = evidence["replay"] or {"metrics": {}, "contract": {}}
            tool = body.get("tool_failure") or evidence["tool_failure"]
        else:
            # A terminal-incomplete run is judged the way it always was: it
            # has no ending to record, so demanding one would turn a single
            # cause into two failures.
            if wsha and not wpath.exists():
                contract_problems.append(
                    f"{run}: the finished event records a watchdog log digest "
                    "but the log is gone")
            replay = replay_child(
                report, watchdog_path=wpath if wpath.exists() else None,
                watchdog_sha256=wsha, spec=spec, require_terminal=False)
            problems = replay["problems"]
            # Evidence that does not replay is not a measurement, so it gets
            # no verdict -- the same rule a tool failure gets, for the same
            # reason.
            tool = body.get("tool_failure") or (
                "report_invalid" if problems else None)
        contract_problems += [f"{run}: {p}" for p in problems]
        safety = (report.get("stopped_early") or {}).get("reason")
        q1 = q1_run(replay["metrics"], safety_reason=safety, tool_failure=tool)
        q2 = q2_run(replay["contract"], safety_reason=safety, tool_failure=tool)
        q1s[run], q2s[run] = q1["verdict"], q2["verdict"]
        runs.append({"run_id": run, "outcome": body.get("outcome"),
                     "tool_failure": tool, "safety_reason": safety,
                     "Q1_run": q1["verdict"], "Q1_reason": q1["reason"],
                     "Q2_run": q2["verdict"], "metrics": replay["metrics"]})
    # Named, not just counted: "cross-run provenance fields differ" tells
    # nobody which field, and the headline's reason list is what a reader has.
    contract_problems += cross_run_problems(reports)
    prefix = prefix_consistency(reports)
    q1p = q1_plan(q1s, {r: lengths[r] for r in q1s})
    q2p = q2_plan(q2s)
    boots = [st["boots"].get(e["run_id"]) for e in plan["runs"]
             if st["boots"].get(e["run_id"])]
    head = headline_full_B(runs=runs, boots=boots, prefix=prefix, q1=q1p,
                           q2=q2p, cross_run_identical=cross_run_identical(reports),
                           contract_problems=contract_problems)
    agg = {"schema_version": 1, "kind": "longrun_aggregate",
           "experiment_id": plan["experiment_id"], "created_at": now_iso(),
           "plan_digest": plan["plan_digest"],
           "design_sha256": plan["design_sha256"],
           "one_run_per_boot": True,
           "boot_fingerprints": boots,
           "journal_events": len(st["events"]),
           "runs": runs, "cancelled_runs": st["cancelled"],
           "terminal": bool(st["terminal"]),
           "complete": st["complete"],
           "state": ("complete_by_rule" if st["cancelled"] and not st["terminal"]
                     else "terminal_incomplete" if st["terminal"]
                     else "complete" if st["complete"] else "incomplete"),
           "prefix_consistency": prefix,
           "Q1_plan": q1p, "Q2_plan": q2p,
           "headline": head, "replay_problems": contract_problems}
    return agg


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _nonneg(x) -> bool:
    return finite(x) and x >= 0


#: ``field -> (predicate, description)``. Present-and-wrong is as fatal as
#: absent: a report claiming 999999 seconds of compute or 999 teardown clears
#: is not a schema violation, it is a measurement nobody looked at.
_SCALAR_RULES = {
    "experiment_id": (lambda v: isinstance(v, str) and bool(v), "a non-empty string"),
    "run_id": (lambda v: isinstance(v, str) and bool(v), "a non-empty string"),
    "plan_digest": (lambda v: isinstance(v, str) and len(v) == 64,
                    "a 64-character digest"),
    "nonce": (lambda v: isinstance(v, str) and bool(v), "a non-empty string"),
    "child_start_identity": (lambda v: isinstance(v, str) and bool(v),
                             "a non-empty string"),
    "child_pid": (lambda v: _is_int(v) and v > 0, "a positive integer"),
    "child_pgid": (lambda v: _is_int(v) and v > 0, "a positive integer"),
    "declared_rows": (lambda v: _is_int(v) and v > 0, "a positive integer"),
    "rows_requested": (lambda v: _is_int(v) and v > 0, "a positive integer"),
    "rows_completed": (lambda v: _is_int(v) and v >= 0,
                       "a non-negative integer"),
    "pool_pairs": (_is_int, "an integer"),
    "pool_rows": (_is_int, "an integer"),
    "scheduled_empty_cache_every": (lambda v: _is_int(v) and v > 0,
                                    "a positive integer"),
    "teardown_empty_cache_calls": (lambda v: v == 1,
                                   "exactly 1: the child clears once, after "
                                   "the condition clock has stopped"),
    "teardown_empty_cache_seconds": (_nonneg, "a non-negative number"),
    "end_to_end_seconds": (_nonneg, "a non-negative number"),
    "model_compute_seconds": (_nonneg, "a non-negative number"),
    "started_at": (lambda v: isinstance(v, str) and bool(v), "a timestamp"),
    "finished_at": (lambda v: isinstance(v, str) and bool(v), "a timestamp"),
}


def _scalar_type_problems(report: dict) -> list[str]:
    problems = []
    for field, (ok, description) in _SCALAR_RULES.items():
        if field not in report:
            continue  # already reported as missing
        if not ok(report[field]):
            problems.append(f"{field} is {report[field]!r}; it has to be "
                            f"{description}")
    clocks = report.get("clocks")
    if isinstance(clocks, dict):
        for key in ("model_load_seconds", "condition_clock_seconds",
                    "process_clock_seconds"):
            if key in clocks and not _nonneg(clocks[key]):
                problems.append(f"clocks.{key} is {clocks[key]!r}; a duration "
                                "is a non-negative finite number")
    return problems


def _preflight_problems(pre) -> list[str]:
    """The four gated readings are numbers, or ``None`` for an unread probe.

    ``None`` is honest -- the gate treats an unreadable metric as a failure --
    but a string is not a reading at all, and storing one would let a report
    look like it had preflight data when nobody measured anything.
    """
    if not isinstance(pre, dict):
        return []
    problems = []
    for metric in GATE_METRICS:
        if metric not in pre:
            continue  # the missing-key case is reported by the nested check
        value = pre[metric]
        if value is not None and not finite(value):
            problems.append(f"preflight.{metric} is {value!r}; a gated reading "
                            "is a number, or null when the probe could not be "
                            "read")
    return problems


def _row_problems(per_row: list[dict]) -> list[str]:
    """Per-row values that are impossible rather than merely surprising."""
    problems = []
    last_monotonic = None
    for r in per_row:
        row = r.get("row")
        where = f"row {row}"
        compute, e2e = r.get("compute_seconds"), r.get("end_to_end_seconds")
        for name, value in (("compute_seconds", compute),
                            ("end_to_end_seconds", e2e)):
            if finite(value) and value < 0:
                problems.append(f"{where}: {name} is {value!r}; time does not "
                                "run backwards inside a row")
        if finite(compute) and finite(e2e) and e2e + 1e-9 < compute:
            problems.append(f"{where}: end_to_end {e2e}s is less than the "
                            f"compute {compute}s it contains")
        clear = r.get("clear_seconds")
        if clear is not None and finite(clear) and clear < 0:
            problems.append(f"{where}: clear_seconds is {clear!r}")
        for name in ("tokens", "supervised_tokens"):
            value = r.get(name)
            if not _is_int(value) or value < 0:
                problems.append(f"{where}: {name} is {value!r}; it has to be a "
                                "non-negative integer")
        tokens, supervised = r.get("tokens"), r.get("supervised_tokens")
        if _is_int(tokens) and tokens <= 0:
            problems.append(f"{where}: tokens is {tokens!r}; a measured row "
                            "has at least one token")
        if _is_int(tokens) and _is_int(supervised) and supervised > tokens:
            problems.append(f"{where}: supervised_tokens {supervised} exceeds "
                            f"tokens {tokens}; the mask is a subset")
        if not finite(r.get("loss")):
            problems.append(f"{where}: loss is {r.get('loss')!r}; it has to be "
                            "a finite number")
        mono = r.get("monotonic")
        if mono is not None:
            if not _nonneg(mono):
                problems.append(f"{where}: monotonic is {mono!r}")
            elif last_monotonic is not None and mono + 1e-9 < last_monotonic:
                problems.append(f"{where}: monotonic {mono} goes backwards "
                                f"after {last_monotonic}")
            if _nonneg(mono):
                last_monotonic = mono
    return problems


def _memory_problems(memory, rows_completed: int) -> list[str]:
    if not isinstance(memory, list):
        return []
    problems = []
    for entry in memory:
        row = entry.get("row")
        if not _is_int(row) or not (1 <= row <= max(rows_completed, 0)):
            problems.append(f"a memory sample names row {row!r}, which is not "
                            f"one of the {rows_completed} rows that were "
                            "measured")
        probe = entry.get("probe_seconds")
        if probe is not None and not _nonneg(probe):
            problems.append(f"memory sample at row {row!r} records "
                            f"probe_seconds {probe!r}")
    return problems


def _stopped_early_problems(stopped, report: dict,
                            rows_completed: int) -> list[str]:
    """The conditional half of section 4.7's schema.

    ``null`` is a complete answer. Anything else has to be the whole block,
    with a row inside the run and two clocks that sit inside the run's own.
    """
    if stopped is None:
        return []
    if not isinstance(stopped, dict):
        return [f"stopped_early is {stopped!r}; it is either null or the whole "
                "block"]
    problems = [f"stopped_early is missing {field!r}"
                for field in STOPPED_EARLY_FIELDS if field not in stopped]
    if stopped.get("requested_by") not in REQUESTED_BY and \
            "requested_by" in stopped:
        problems.append(f"stopped_early.requested_by is "
                        f"{stopped.get('requested_by')!r}, outside "
                        f"{REQUESTED_BY}")
    row = stopped.get("row")
    if "row" in stopped:
        if not _is_int(row) or row < 1:
            problems.append(f"stopped_early.row is {row!r}")
        elif row != rows_completed:
            problems.append(f"stopped_early.row is {row} but the run recorded "
                            f"{rows_completed} rows; a run stops at the row it "
                            "last completed")
    clocks = report.get("clocks") or {}
    for key in ("condition_clock_seconds", "process_clock_seconds"):
        if key not in stopped:
            continue
        value = stopped[key]
        if not _nonneg(value):
            problems.append(f"stopped_early.{key} is {value!r}; a duration is "
                            "a non-negative finite number")
            continue
        outer = clocks.get(key)
        if finite(outer) and value > outer + 1e-6:
            problems.append(
                f"stopped_early.{key} is {value}s, past the run's own {outer}s: "
                "the stop is inside the run, not after it")
    if "reason" in stopped and not (isinstance(stopped["reason"], str)
                                    and stopped["reason"]):
        problems.append(f"stopped_early.reason is {stopped.get('reason')!r}")
    return problems


def _derived_problems(report: dict, per_row: list[dict],
                      contract_total: float, probe_total: float) -> list[str]:
    """Recompute the sums the report states rather than reading them back."""
    problems = []
    compute = sum(float(r.get("compute_seconds") or 0.0) for r in per_row)
    e2e = sum(float(r.get("end_to_end_seconds") or 0.0) for r in per_row)
    for field, want in (("model_compute_seconds", compute),
                        ("end_to_end_seconds", e2e)):
        got = report.get(field)
        if got is not None and finite(got) and not same_sum(got, want):
            problems.append(f"{field} is {got}, but the per-row array sums to "
                            f"{want}")
    breakdown = report.get("between_row_overhead_breakdown")
    if isinstance(breakdown, dict):
        expected = {
            "scheduled_empty_cache_seconds": contract_total,
            "memory_probe_seconds": probe_total,
            "unattributed_seconds": max(
                0.0, e2e - compute - contract_total - probe_total)}
        for field, want in expected.items():
            got = breakdown.get(field)
            if got is None:
                continue  # the missing-key case is reported by the nested check
            if not _nonneg(got):
                problems.append(f"between_row_overhead_breakdown.{field} is "
                                f"{got!r}")
            elif not same_sum(got, want):
                problems.append(
                    f"between_row_overhead_breakdown.{field} is {got}, but "
                    f"recomputing it from per_row gives {want}")
    return problems


CROSS_RUN_FIELDS = PROVENANCE_FIELDS


def _without(value, keys):
    if not isinstance(value, dict):
        return value
    return {k: v for k, v in value.items() if k not in keys}


def cross_run_problems(reports: list[dict]) -> list[str]:
    """Every declared field present in every run, and equal across them.

    Absence is not agreement. A field missing from all three reports used to
    pass this check because nothing disagreed -- which is how a provenance
    field could be dropped everywhere and never noticed.

    The declared length k is the single exception the design allows
    (:data:`CROSS_RUN_VARIES`), and it is not waived: the key is required to
    be present in every run and to equal that run's own ``declared_rows``. A
    b2 whose ``max_rows`` says 500 measured a different prefix from the one
    the plan asked for, and that has to be as loud as any other disagreement.
    """
    if not reports:
        return []
    problems: list[str] = []
    provs = [r.get("provenance") or {} for r in reports]
    for field in PROVENANCE_FIELDS:
        absent = [r.get("run_id") for r, p in zip(reports, provs) if field not in p]
        if absent:
            problems.append(f"provenance field {field!r} is missing from "
                            f"{absent}, so the runs have not been shown to agree")
            continue
        varies = CROSS_RUN_VARIES.get(field, ())
        values = [_without(p[field], varies) for p in provs]
        if any(value != values[0] for value in values[1:]):
            problems.append(f"provenance field {field!r} differs across runs, "
                            "and section 4.6 requires it to be identical")
        for key in varies:
            for report, prov in zip(reports, provs):
                run = report.get("run_id")
                block = prov[field]
                if not isinstance(block, dict) or key not in block:
                    problems.append(
                        f"{run}: {field}.{key} is the one value allowed to "
                        "differ between runs, so it has to be recorded")
                elif block[key] != report.get("declared_rows"):
                    problems.append(
                        f"{run}: {field}.{key} is {block[key]!r}, not this "
                        f"run's declared {report.get('declared_rows')!r}")
    return problems


def cross_run_identical(reports: list[dict]) -> bool:
    return not cross_run_problems(reports)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def child_identity_problems(report: dict, entry: dict, plan: dict,
                            started_body: dict) -> list[str]:
    """Is this report the run the plan asked for, from the launch we recorded?

    A report is a file in a directory. Nothing about being called ``b1.json``
    makes it b1's report, so every field that says which run it is gets
    checked against the plan entry and the started event.
    """
    problems = []
    for field, want in (("experiment_id", plan["experiment_id"]),
                        ("run_id", entry["run_id"]),
                        ("condition", entry["condition"]),
                        ("declared_rows", entry["declared_rows"]),
                        ("plan_digest", plan["plan_digest"])):
        if report.get(field) != want:
            problems.append(f"the report's {field} is {report.get(field)!r}, "
                            f"but the plan says {want!r}")
    want_nonce = started_body.get("nonce")
    if want_nonce is None:
        problems.append("the started event recorded no launch nonce, so the "
                        "report cannot be tied to a launch")
    elif report.get("nonce") != want_nonce:
        problems.append("the report carries another launch's nonce")
    prov = report.get("provenance") or {}
    missing = [f for f in CHILD_PROVENANCE_REQUIRED if f not in prov]
    if missing:
        problems.append(f"provenance is missing {missing}")
    return problems


def replay_child(report: dict, *, watchdog_path=None,
                 watchdog_sha256: str | None = None,
                 spec: SafetySpec | None = None,
                 stop_request: dict | None = None,
                 launch_nonce: str | None = None,
                 require_terminal: bool = False) -> dict:
    """Recompute everything about one run, and check it both ways.

    Reading a stored value back is not verification. So every stored metric is
    compared against the recomputation, *and* the recomputation is checked for
    fields the report failed to store or stored that nobody should have --
    a report that simply omits the number it got wrong must not pass.
    """
    spec = spec or SafetySpec()
    problems: list[str] = []
    for key in CHILD_REQUIRED:
        if key not in report:
            problems.append(f"the report is missing {key!r}")
    unknown = sorted(set(report) - set(CHILD_SCHEMA_KEYS))
    if unknown:
        problems.append(f"the report carries fields nobody declared: {unknown}. "
                        "A field replay does not recompute is a number nobody "
                        "checked")
    if report.get("schema_version") != CHILD_SCHEMA_VERSION:
        problems.append(f"schema_version is {report.get('schema_version')!r}, "
                        f"not {CHILD_SCHEMA_VERSION}")
    if report.get("kind") != CHILD_KIND:
        problems.append(f"kind is {report.get('kind')!r}, not {CHILD_KIND!r}")
    if report.get("condition") != CONDITION:
        problems.append(f"condition is {report.get('condition')!r}; every run "
                        f"of plan B is {CONDITION!r}, so a report saying "
                        "otherwise did not measure this experiment")

    # A block that is present but empty is the same evasion as a missing one.
    for block, keys in CHILD_NESTED_REQUIRED.items():
        if block not in report:
            continue  # already reported above, once
        if block == "provenance":
            continue  # checked below, by the validator the loader also uses
        value = report[block]
        if not isinstance(value, dict) or not value:
            problems.append(f"{block} is empty or is not an object, so none of "
                            "the fields it should carry are there")
            continue
        missing = [k for k in keys if k not in value]
        if missing:
            problems.append(f"{block} is missing {missing}")
    storage = report.get("float_storage")
    if isinstance(storage, dict):
        for key in ("seconds_rounded", "loss_rounded"):
            if key in storage and storage[key] is not False:
                problems.append(
                    f"float_storage.{key} is {storage[key]!r}. Section 4.7 "
                    "stores seconds and losses exactly as measured: once they "
                    "are rounded, neither the summation identities nor the "
                    "0.001 loss tolerance mean what they were declared to mean")

    problems += _scalar_type_problems(report)
    problems += _preflight_problems(report.get("preflight"))

    per_row = report.get("per_row") or []
    if not per_row:
        problems.append("the report has no per_row array, so nothing can be "
                        "recomputed")
        return {"problems": problems, "metrics": {}, "contract": {},
                "watchdog": {"problems": [], "records": []}}

    rows = [r.get("row") for r in per_row]
    if rows != list(range(1, len(rows) + 1)):
        problems.append("per_row rows are not the contiguous sequence "
                        f"1..{len(rows)}")
    for r in per_row:
        for field in ("compute_seconds", "end_to_end_seconds"):
            if not finite(r.get(field)):
                problems.append(f"row {r.get('row')}: {field} is not a finite "
                                f"number ({r.get(field)!r})")
        if r.get("cleared") and not finite(r.get("clear_seconds")):
            problems.append(f"row {r.get('row')}: cleared but clear_seconds is "
                            f"{r.get('clear_seconds')!r}")
        if not r.get("cleared") and r.get("clear_seconds") is not None:
            problems.append(f"row {r.get('row')}: not cleared yet carries "
                            "clear_seconds")

    stopped = report.get("stopped_early")
    metrics = compute_metrics(per_row,
                              stop_reason=(stopped or {}).get("reason"))
    stored_metrics = report.get("metrics")
    if stored_metrics is not None:
        problems += _compare_metrics(stored_metrics, metrics)

    if report.get("rows_completed") != len(per_row):
        problems.append(f"rows_completed is {report.get('rows_completed')} but "
                        f"per_row holds {len(per_row)} rows")
    if stopped is None and report.get("declared_rows") != len(per_row):
        problems.append(f"the run was not stopped early yet completed "
                        f"{len(per_row)} of {report.get('declared_rows')} rows")
    if stopped is None and report.get("completed_input_digest") != \
            report.get("input_order_digest"):
        problems.append("the completed input digest differs from the declared "
                        "order even though the run was not stopped early")

    # Section 4.6 keeps input_order_digest and completed_input_digest at the
    # top level. Keeping them there is fine; leaving them out of the replay is
    # not -- a digest nobody recomputes is a label, and per_row is the only
    # place the identity of each measured row is actually written down.
    recomputed = _digest_ids(r.get("sample_id") for r in per_row)
    if report.get("completed_input_digest") != recomputed:
        problems.append("completed_input_digest does not recompute from the "
                        "sample ids in per_row")
    if stopped is None and report.get("input_order_digest") != recomputed:
        problems.append("input_order_digest does not recompute from per_row, "
                        "and this run measured its whole declared order")

    # The pool is a frozen constant of the design, not a per-run parameter.
    for key, want in (("pool_pairs", POOL_PAIRS), ("pool_rows", POOL_ROWS)):
        if key in report and report[key] != want:
            problems.append(f"{key} is {report[key]!r}, not the declared "
                            f"{want}")
    if isinstance(report.get("pool_pairs"), int) and \
            isinstance(report.get("pool_rows"), int) and \
            report["pool_pairs"] * ROWS_PER_PAIR != report["pool_rows"]:
        problems.append(f"pool_pairs {report['pool_pairs']} x {ROWS_PER_PAIR} "
                        f"rows is not pool_rows {report['pool_rows']}")
    if "rows_requested" in report and \
            report["rows_requested"] != report.get("declared_rows"):
        problems.append(f"rows_requested is {report['rows_requested']!r} but "
                        f"the run declares {report.get('declared_rows')!r}")

    if "provenance" in report:
        problems += provenance_problems(
            report["provenance"], declared_rows=report.get("declared_rows"))

    problems += _row_problems(per_row)
    problems += _memory_problems(report.get("memory"), len(per_row))
    problems += _stopped_early_problems(stopped, report, len(per_row))

    cost = report.get("scheduled_empty_cache_cost") or {}
    contract = clear_contract(
        per_row, declared_rows=report.get("declared_rows", len(per_row)),
        recorded_total=cost.get("total_seconds"), per_call=cost.get("per_call"),
        end_to_end_seconds=report.get("end_to_end_seconds"))
    problems += contract["problems"]
    problems += _derived_problems(
        report, per_row,
        sum(float(r.get("clear_seconds") or 0.0) for r in per_row
            if r.get("cleared")),
        sum(float(m.get("probe_seconds") or 0.0)
            for m in (report.get("memory") or [])))
    if cost.get("calls") is not None and cost["calls"] != contract["clear_calls"]:
        problems.append(f"the report claims {cost['calls']} clear calls but "
                        f"{contract['clear_calls']} are recorded")
    problems += clock_nesting_problems(report.get("clocks") or {})

    if not (report.get("provenance") or {}):
        problems.append("the report has no provenance block")

    watchdog = {"problems": [], "records": []}
    if watchdog_path is not None:
        watchdog = replay_watchdog_log(watchdog_path,
                                       expected_sha256=watchdog_sha256)
        problems += watchdog["problems"]
        if not watchdog["problems"]:
            if require_terminal:
                # Only asked of a run the journal calls completed: a run that
                # was SIGKILLed or that crashed has no ending to record, and
                # demanding one there would turn every terminal-incomplete
                # run into two failures with one cause.
                problems += watchdog_terminal_problems(
                    watchdog["records"], spec,
                    identity={f: report.get(f) for f in IDENTITY_FIELDS})
            sem = replay_watchdog_semantics(
                watchdog["records"], spec,
                stop_request=stop_request,
                # Stated, never inferred from a reason that may have been
                # deleted along with the block it lived in. A report with no
                # stopped_early is a report saying this run was not stopped,
                # and the log has to agree with that too.
                claimed_stop=report.get("stopped_early") is not None,
                nonce=launch_nonce if launch_nonce is not None
                else report.get("nonce"),
                # The third account. The launch record and the report are the
                # parent's and the child's; the watchdog is the only party
                # that was actually watching, so rewriting the other two
                # together no longer agrees with itself (§4.8 7b).
                identity={f: report.get(f) for f in IDENTITY_FIELDS},
                claimed_reason=(stopped or {}).get("reason"))
            problems += sem["problems"]
            watchdog["semantics"] = sem
            if stopped is not None and not sem["sigterm"]:
                problems.append("the child reports it was stopped by the "
                                "watchdog, but the watchdog log records no "
                                "sigterm")
    return {"problems": problems, "metrics": metrics, "contract": contract,
            "watchdog": watchdog}


METRIC_KEYS = ("D100", "D20", "Dmax")


def _compare_metrics(stored: dict, recomputed: dict) -> list[str]:
    """Both directions, and zero is a legitimate value.

    The rule is not "a metric may never be zero". It is that a metric which
    could not be computed must be null and carry a reason, and a metric which
    could be computed must match. A run whose tail genuinely measured zero
    seconds reports 0.0, and that passes.
    """
    problems = []
    allowed = set(METRIC_KEYS) | {f"{k}_reason" for k in METRIC_KEYS} \
        | {"rows_completed"}
    extra = set(stored) - allowed
    if extra:
        problems.append(f"the stored metrics carry unexpected keys "
                        f"{sorted(extra)}")
    for key in METRIC_KEYS:
        if key not in stored:
            problems.append(f"the stored metrics omit {key}")
            continue
        want, got = recomputed.get(key), stored.get(key)
        want_reason = recomputed.get(f"{key}_reason")
        got_reason = stored.get(f"{key}_reason")
        if want is None:
            if got is not None:
                problems.append(f"{key} cannot be recomputed ({want_reason}) "
                                f"but the report stores {got!r}")
            if got_reason != want_reason:
                problems.append(f"{key}_reason is {got_reason!r} but the "
                                f"recomputation gives {want_reason!r}")
        else:
            if got is None:
                problems.append(f"{key} recomputes to {want} but the report "
                                "stores null; a computable metric may not be "
                                "recorded as missing")
            elif not same_value(got, want):
                problems.append(f"{key} recomputes to {want} but the report "
                                f"stored {got}")
            if got_reason is not None:
                problems.append(f"{key} is computable yet carries the reason "
                                f"{got_reason!r}")
    return problems


# ---------------------------------------------------------------------------
# --verify and --from-json
# ---------------------------------------------------------------------------


def verify_experiment(experiment_id: str, *, root=None,
                      design_path=DESIGN_JSON) -> dict:
    """Replay a finished session end to end, recomputing everything.

    Section 4.8 of the design, in order: calibration and gate, the plan and
    session digests, the source snapshot, the journal and its boots, every
    child, every watchdog log, rule R1 and the runs it cancelled, then the
    aggregate, the plan-level verdicts and the headline.
    """
    st = session_state(experiment_id, root=root)
    if "plan" not in st:
        return {"problems": st.get("problems", ["no session"]), "notes": [],
                "state": None}
    problems = list(st["problems"])
    notes: list[str] = []
    paths, plan, session = st["paths"], st["plan"], st["session"]

    # 1. calibration: complete, self-consistent, and unchanged since init
    calib = None
    if paths["calibration"].exists():
        calib = json.loads(paths["calibration"].read_text())
        if sha256_file(paths["calibration"]) != session.get("calibration_sha256"):
            problems.append("calibration.json has changed since init")
        problems += validate_calibration(calib)
        if calib.get("thresholds") and session.get("thresholds") \
                and calib["thresholds"] != session["thresholds"]:
            problems.append("session.json records different thresholds from "
                            "the calibration it was built from")
    else:
        problems.append("calibration.json is missing")
    policy = session.get("gate_policy")
    if policy != GATE_POLICY:
        problems.append(f"the session's gate policy is {policy!r}, not the "
                        f"fixed {GATE_POLICY!r}")

    # 2. plan and session digests, and the design they came from
    if plan_digest(plan) != plan["plan_digest"]:
        problems.append("plan.json does not hash to its recorded digest")
    if plan.get("design_sha256") != design_sha256(design_path):
        # Informational, and therefore not a problem. The plan embeds the
        # constants verbatim and is itself under a digest, so the copy on disk
        # cannot change what any run was judged by. Calling this fatal while
        # documenting it as informational was the contradiction; the design
        # says informational, so it is reported as a note.
        notes.append("the design file on disk is not the one the plan was "
                     "built from. This is informational: the plan's embedded "
                     "frozen_constants are authoritative and are covered by "
                     "plan_digest.")
    if plan.get("frozen_constants", {}).get("lengths") != [500, 1000, 2000]:
        problems.append("the plan's lengths are not the approved 500/1000/2000")

    # 3. the source snapshot
    problems += verify_sources(ROOT, session["source_manifest"],
                               paths["snapshot"], check_working_tree=False)
    if manifest_digest(session["source_manifest"]) != \
            session.get("source_manifest_digest"):
        problems.append("the source manifest does not hash to its digest")

    # 4. journal, gate polls and boots
    problems += gate_replay_problems(st["events"],
                                     (calib or {}).get("thresholds") or {})
    boots = [b for b in st["boots"].values() if b]
    if len(set(boots)) != len(boots):
        problems.append("two measured runs share a boot")

    # 5. every child and its watchdog log
    spec = plan_spec(plan)
    reports = []
    for entry in plan["runs"]:
        run = entry["run_id"]
        rp = paths["dir"] / f"{run}.json"
        body = (st["finished"].get(run) or {}).get("body") or {}
        started_body = (st["started"].get(run) or {}).get("body") or {}
        if not rp.exists():
            if body.get("outcome") == "completed":
                problems.append(f"{run}: the journal says it completed, but "
                                f"{rp.name} does not exist")
            problems += [f"{run}: {p}" for p in failure_evidence_problems(
                paths["dir"], run, body, outcome=body.get("outcome"))]
            continue
        report = json.loads(rp.read_text())
        reports.append(report)
        if not body.get("report_sha256"):
            problems.append(f"{run}: the finished event records no report digest")
        elif sha256_file(rp) != body["report_sha256"]:
            problems.append(f"{run}.json does not match the digest the finished "
                            "event recorded")
        wpath = paths["dir"] / f"{run}.watchdog.jsonl"
        problems += [f"{run}: {p}" for p in failure_evidence_problems(
            paths["dir"], run, body, outcome=body.get("outcome"))]
        if body.get("outcome") == "completed":
            # One validator, shared with the pre-finish gate and with the
            # precondition replay.
            problems += [f"{run}: {p}" for p in completed_run_evidence(
                paths, run, spec=spec, plan=plan, entry=entry, session=session,
                started_body=started_body, finished_body=body)["problems"]]
        else:
            problems += [f"{run}: {p}" for p in child_identity_problems(
                report, entry, plan, started_body)]
        wsha = body.get("watchdog_sha256")
        if body.get("outcome") == "completed":
            if not wsha:
                problems.append(f"{run}: a completed run records no watchdog "
                                "log digest")
            if not wpath.exists():
                problems.append(f"{run}: a completed run has no watchdog log")
        if wsha and not wpath.exists():
            problems.append(f"{run}: the finished event records a watchdog log "
                            "digest but the log is gone")
        if body.get("outcome") != "completed":
            # A completed run's stop evidence is the shared validator's job,
            # above; doing it again here would be a second definition, and
            # this one used to be the weaker of the two. What is left is the
            # terminal-incomplete path, which still needs the request body to
            # replay the log against.
            stop_request = None
            if report.get("stopped_early") is not None:
                sp = paths["dir"] / f"{run}.stop_request.json"
                if not sp.exists():
                    problems.append(f"{run}: the child says it was stopped, "
                                    "but the stop request it authenticated is "
                                    "gone")
                else:
                    recorded = body.get("stop_request_sha256")
                    if recorded is not None and (
                            not is_sha256(recorded)
                            or sha256_file(sp) != recorded):
                        problems.append(f"{run}: the stop request does not "
                                        "match the digest the finished event "
                                        "recorded")
                    got = read_stop_request(paths["dir"], prefix=run,
                                            nonce=started_body.get("nonce"))
                    if not got["accepted"]:
                        problems.append(f"{run}: the stop request does not "
                                        f"authenticate ({got.get('detail')})")
                    else:
                        stop_request = got["body"]
            out = replay_child(report,
                               watchdog_path=wpath if wpath.exists() else None,
                               watchdog_sha256=wsha, spec=spec,
                               stop_request=stop_request,
                               launch_nonce=started_body.get("nonce"))
            problems += [f"{run}: {p}" for p in out["problems"]]

    # 6. R1 and the runs it cancelled
    for ev in st["events"]:
        if ev.get("event") != EVENT_PLAN_ARM_CANCELLED:
            continue
        body = ev.get("body") or {}
        run = body.get("triggered_by_run")
        rp = paths["dir"] / f"{run}.json"
        if not rp.exists():
            problems.append(f"R1 names {run}, whose report is missing")
            continue
        report = json.loads(rp.read_text())
        fin = (st["finished"].get(run) or {}).get("body") or {}
        out = replay_child(report, spec=spec)
        q1 = q1_run(out["metrics"],
                    safety_reason=(report.get("stopped_early") or {}).get("reason"),
                    tool_failure=fin.get("tool_failure"))
        decision = r1_should_cancel(outcome=fin.get("outcome", "completed"),
                                    tool_failure=fin.get("tool_failure"),
                                    q1_verdict=q1["verdict"],
                                    q1_reason=q1["reason"])
        if not decision["cancel"]:
            problems.append(f"R1 cancelled runs after {run}, but replay says "
                            f"it should not have: {decision['why']}")
        wsha = body.get("watchdog_sha256")
        wpath = paths["dir"] / f"{run}.watchdog.jsonl"
        if q1["reason"] and q1["reason"].startswith("safety_stop:"):
            if not wsha:
                problems.append(f"R1 fired on a safety stop in {run} but names "
                                "no watchdog log to have recomputed it from")
            if not wpath.exists():
                problems.append(f"R1 must be replayable against {wpath.name}, "
                                "which is gone")
            elif wsha and sha256_file(wpath) != wsha:
                problems.append(f"{wpath.name} does not match the digest R1 "
                                "recorded")
        stored = body.get("recomputed_verdict") or {}
        fresh = {"Q1": q1["verdict"], "reason": q1["reason"],
                 "D100": out["metrics"].get("D100"),
                 "Dmax": out["metrics"].get("Dmax")}
        if stored != fresh:
            problems.append(f"R1's recorded verdict {stored} does not match a "
                            f"fresh recomputation {fresh}")
        order = [e["run_id"] for e in plan["runs"]]
        expected = [r for r in order[order.index(run) + 1:]]
        if list(body.get("cancelled_runs") or []) != expected:
            problems.append(f"R1 cancelled {body.get('cancelled_runs')} but the "
                            f"plan's remaining runs were {expected}")
        for r in expected:
            if r in st["started"]:
                problems.append(f"{r} was cancelled yet has a started event")

    # 7. the aggregate, the plan verdicts and the headline
    aggregate = None
    if paths["aggregate"].exists():
        aggregate = json.loads(paths["aggregate"].read_text())
        fresh = build_aggregate(st)
        # Every derived field, not a chosen few. Comparing a shortlist means
        # anything outside it can be edited without the replay noticing, which
        # is the same as not storing it under a digest at all.
        volatile = {"created_at"}
        keys = (set(aggregate) | set(fresh)) - volatile
        for key in sorted(keys):
            if key not in aggregate:
                problems.append(f"the stored aggregate omits {key}")
            elif key not in fresh:
                problems.append(f"the stored aggregate carries an unknown "
                                f"field {key}")
            elif aggregate[key] != fresh[key]:
                problems.append(f"the stored aggregate's {key} does not match a "
                                "fresh derivation from the journal")
    return {"problems": problems, "notes": notes, "state": st,
            "aggregate": aggregate, "reports": reports}


def render_from_json(experiment_id: str, *, root=None) -> dict:
    """Render only after a full verification passes.

    ``--from-json`` is not an alias for ``--verify``: it produces the document
    people read. A renderer that draws whatever is on disk would happily
    typeset a tampered aggregate, so it refuses to draw anything until the
    same replay ``--verify`` performs has come back clean.
    """
    out = verify_experiment(experiment_id, root=root)
    if out["problems"]:
        return {"rendered": False, "problems": out["problems"],
                "reason": "verification failed, so nothing is rendered"}
    if out.get("aggregate") is None:
        return {"rendered": False,
                "problems": ["there is no aggregate to render yet"],
                "reason": "the session has not been finalised"}
    agg = out["aggregate"]
    lines = [f"# Report 16 -- {agg['experiment_id']}", "",
             f"state: {agg['state']}",
             f"Q1_plan: {agg['Q1_plan']['value']}",
             f"Q2_plan: {agg['Q2_plan']['value']}",
             f"headline allowed: {agg['headline']['allowed']}"]
    return {"rendered": True, "problems": [], "markdown": "\n".join(lines),
            "aggregate": agg}


# ---------------------------------------------------------------------------
# session_next: the real orchestration.
#
# The CLI refuses to reach this. It exists, tested, so that what approval
# would unlock is reviewable now rather than written afterwards in a hurry.
# Every external dependency is injected, which is also how the integration
# test drives a real child and a real watchdog without a model.
# ---------------------------------------------------------------------------


def session_next(experiment_id: str, *, root=None, gate, spawn_watchdog,
                 spawn_child, hand_identity, await_armed, supervise_fn,
                 collect, boot=None, verify_live_sources=True,
                 code_files=CODE_FILES, run_r1=True,
                 dependency_preflight=dependency_preflight) -> dict:
    """One measured run, one boot, under one non-blocking exclusive lock.

    The lock is taken *before* the journal is read, not after. Reading first
    and locking second leaves a window in which two processes both see the
    same next run, both pass the boot check and both spend the boot.
    """
    paths = session_paths(experiment_id, root=root)
    if not paths["dir"].exists():
        return {"ok": False, "problems": [f"{paths['dir']} does not exist"]}
    with exclusive_lock(paths["lock"], description=f"report 16 {experiment_id}"):
        return _session_next_locked(
            experiment_id, root=root, gate=gate, spawn_watchdog=spawn_watchdog,
            spawn_child=spawn_child, hand_identity=hand_identity,
            await_armed=await_armed, supervise_fn=supervise_fn, collect=collect,
            boot=boot, verify_live_sources=verify_live_sources, run_r1=run_r1,
            dependency_preflight=dependency_preflight)


def _session_next_locked(experiment_id, *, root, gate, spawn_watchdog,
                         spawn_child, hand_identity, await_armed, supervise_fn,
                         collect, boot, verify_live_sources, run_r1,
                         dependency_preflight=dependency_preflight) -> dict:
    # Everything the decision rests on is read here, inside the lock: the
    # journal, the next run, the boot and whether the session is already over.
    st = session_state(experiment_id, root=root)
    if "plan" not in st:
        return {"ok": False, "problems": st.get("problems", ["no session"])}
    if st["problems"]:
        return {"ok": False, "problems": st["problems"]}
    paths, plan = st["paths"], st["plan"]

    if st["terminal"]:
        return {"ok": False, "terminal": True, "problems": [
            f"{experiment_id} is terminal incomplete ({st['terminal']}). A "
            "measured run started and did not finish, so this experiment is "
            "over in every boot. Use --session-finalize to record it, then a "
            "new experiment id."]}
    if st["cancelled"]:
        return {"ok": False, "problems": [
            f"rule R1 cancelled {st['cancelled']}; use --session-finalize"]}
    if not st["next_run"]:
        return {"ok": False, "problems": ["every planned run has been measured; "
                                          "use --session-finalize"]}

    fingerprint = (boot or boot_identity(plan["experiment_id"]))["boot_fingerprint"]
    if fingerprint in {b for b in st["boots"].values() if b}:
        return {"ok": False, "problems": [
            "a measured run has already happened in this boot; restart before "
            "the next one. This tool does not restart the machine and must "
            "not: a tool that reboots what it is measuring has changed it."]}

    # Before the gate, before any event, before anything that could spend a
    # boot: replay what the session already claims about itself.
    pre = session_preconditions(experiment_id, root=root,
                                check_working_tree=verify_live_sources)
    if pre:
        return {"ok": False, "boot_consumed": False, "preconditions_failed": True,
                "problems": pre}

    # Before the gate, before the watchdog, before measurement_started -- and
    # therefore before this boot is spent. exp001's b1 discovered a missing
    # dependency after all three, from inside a child that had already cost
    # the experiment its only chance at b1.
    pre_deps = dependency_preflight()
    if not pre_deps.get("ok"):
        return {"ok": False, "boot_consumed": False, "retryable": True,
                "dependency_preflight_failed": True,
                "reason": "dependency_preflight",
                "evidence": pre_deps.get("evidence"),
                "problems": pre_deps.get("problems") or
                ["a pinned dependency could not be resolved locally"]}

    run = st["next_run"]
    entry = next(e for e in plan["runs"] if e["run_id"] == run)
    if True:
        gate_result = gate()
        if not gate_result.get("passed"):
            append_session_event(paths, run, EVENT_GATE_ATTEMPT,
                                 {"gate": gate_result,
                                  "boot_fingerprint": fingerprint}, plan=plan)
            return {"ok": False, "boot_consumed": False,
                    "problems": ["the gate did not release"]}

        live = verify_sources(ROOT, st["session"]["source_manifest"],
                              paths["snapshot"],
                              check_working_tree=verify_live_sources)
        if live:
            append_session_event(paths, run, EVENT_PRE_SPAWN_ABORT,
                                 {"gate": gate_result, "drift": live,
                                  "boot_fingerprint": fingerprint}, plan=plan)
            return {"ok": False, "boot_consumed": False, "problems": live}

        from src.training.watchdog import new_nonce
        state = {"watchdog": None, "child": None, "nonce": new_nonce(),
                 "started_event": None}

        def _cleanup():
            from src.training.watchdog import reap
            return reap(state["child"], state["watchdog"])

        def _start_watchdog():
            out = spawn_watchdog(run=run, paths=paths)
            state["watchdog"] = out.get("proc")
            return out

        def _write_started():
            state["started_event"] = append_session_event(
                paths, run, EVENT_STARTED,
                {"declared_rows": entry["declared_rows"],
                 "condition": entry["condition"],
                 "nonce": state["nonce"],
                 "boot_fingerprint": fingerprint, "gate": gate_result,
                 "started_at": now_iso()}, plan=plan)

        def _spawn_child():
            out = spawn_child(run=run, paths=paths, entry=entry,
                              nonce=state["nonce"], plan=plan)
            state["child"] = out.get("proc")
            return out

        outcome = launch_sequence(
            gate_ok=True, sources_ok=True, start_watchdog=_start_watchdog,
            write_started=_write_started, spawn_child=_spawn_child,
            hand_identity=lambda c: hand_identity(c, paths=paths, run=run),
            await_armed=await_armed, cleanup=_cleanup)

        if not outcome.ok:
            if not outcome.boot_consumed:
                append_session_event(paths, run, outcome.event,
                                     {"reason": outcome.reason,
                                      "detail": outcome.detail,
                                      "boot_fingerprint": fingerprint},
                                     plan=plan)
                return {"ok": False, "boot_consumed": False,
                        "retryable": True, "reason": outcome.reason,
                        "cleanup": outcome.cleanup, "problems": [outcome.detail]}
            _finish(paths, run, plan, fingerprint, outcome="no_report",
                    tool_failure=outcome.reason, detail=outcome.detail,
                    started_event=state["started_event"])
            return {"ok": False, "boot_consumed": True, "terminal": True,
                    "reason": outcome.reason, "cleanup": outcome.cleanup,
                    "problems": [outcome.detail]}

        watched = supervise_fn()
        if watched.get("stopped"):
            cleanup = _cleanup()
            _finish(paths, run, plan, fingerprint, outcome="no_report",
                    tool_failure=watched["reason"],
                    detail="supervision stopped the run",
                    started_event=state["started_event"])
            return {"ok": False, "boot_consumed": True, "terminal": True,
                    "reason": watched["reason"], "cleanup": cleanup,
                    "problems": [f"tool failure: {watched['reason']}"]}

        result = collect(run=run, paths=paths)
        _cleanup()
        qualified = qualify_outcome(
            paths, run, claimed=result.get("outcome"),
            exit_status=result.get("exit_status"),
            report_sha256=result.get("report_sha256"),
            watchdog_sha256=result.get("watchdog_sha256"),
            tool_failure=result.get("tool_failure"),
            spec=plan_spec(plan))
        _finish(paths, run, plan, fingerprint, outcome=qualified["outcome"],
                tool_failure=qualified.get("tool_failure")
                or result.get("tool_failure"),
                report_sha256=result.get("report_sha256"),
                watchdog_sha256=result.get("watchdog_sha256"),
                exit_status=result.get("exit_status"),
                detail="; ".join(qualified["problems"]) or None,
                started_event=state["started_event"])
        ok = qualified["outcome"] == "completed" and \
            not (qualified.get("tool_failure") or result.get("tool_failure"))
        r1 = None
        if ok and run_r1:
            # The rule is part of the plan, so it runs here rather than
            # waiting for somebody to remember to invoke it.
            r1 = _apply_rule_r1_locked(experiment_id, root=root)
        return {"ok": ok, "boot_consumed": True,
                "outcome": qualified["outcome"],
                "tool_failure": qualified.get("tool_failure")
                or result.get("tool_failure"),
                "r1": r1, "problems": qualified["problems"]}


def _is_clean_exit(value) -> bool:
    """A clean exit is the integer 0. ``False`` is not, and neither is "0"."""
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def is_sha256(value) -> bool:
    """A digest is 64 lower-case hex characters. Nothing else qualifies.

    ``None``, ``""`` and ``False`` are not "no digest to compare against";
    they are a run whose evidence cannot be checked, which is the same answer
    as a digest that does not match.
    """
    return isinstance(value, str) and bool(SHA256_RE.match(value))


def stop_evidence_problems(session_dir, run: str, *, report: dict,
                           finished_body: dict | None,
                           started_body: dict) -> dict:
    """The safety-stop invariant, checked in both directions.

    One stop leaves four independent traces: the child's ``stopped_early``
    block, the finished event's ``stop_request_sha256``, the stop request
    file on disk, and the watchdog log that asked for it. The previous
    version compared them only when both sides happened to be filled in --
    ``if want``, ``if claimed`` -- so ``None``, ``""`` and ``False`` all read
    as "nothing to compare", and a run whose report said nothing had happened
    was never looked at for traces at all.

    Three shapes came out of that. A finished event whose stop digest was
    nulled still fired R1. A report whose stop digest was blanked still
    verified. And a report whose whole ``stopped_early`` block was deleted --
    with the authenticated request, the finished event's digest, the
    threshold trip and the SIGTERM all left in place -- replayed clean, which
    on a run whose per-row times hold turns a stopped run into a passing one.

    So both directions are stated here. If the report claims a stop, all
    three digests must be present, well-formed and identical, and the request
    must authenticate and say the same thing the report does. If it claims
    none, nothing anywhere may say otherwise. The watchdog's half of the
    invariant lives in :func:`replay_watchdog_semantics`, which is told what
    the report claims rather than left to infer it.

    Returns the authenticated stop request body when there is one, because
    the watchdog replay needs it and reading it twice is how two readers come
    to disagree.
    """
    d = Path(session_dir)
    sp = d / f"{run}.stop_request.json"
    problems: list[str] = []
    stopped = report.get("stopped_early")
    fin_digest = (finished_body or {}).get("stop_request_sha256")
    nonce = (started_body or {}).get("nonce")

    if stopped is None:
        # Nothing anywhere may say a stop was asked for.
        if fin_digest is not None:
            problems.append(
                f"the report records no stopped_early, but the finished event "
                f"records a stop request digest ({str(fin_digest)[:16]}…): "
                "one of the two is wrong about whether this run was stopped")
        if sp.exists():
            got = read_stop_request(d, prefix=run, nonce=nonce)
            if got.get("accepted"):
                problems.append(
                    f"{sp.name} authenticates against this launch, but the "
                    "report records no stopped_early: a stop that was asked "
                    "for and honoured is not a stop that was recorded")
            else:
                problems.append(
                    f"{sp.name} exists for a run whose report records no "
                    f"stopped_early ({got.get('detail') or got.get('reason')})")
        return {"problems": problems, "stop_body": None}

    if not isinstance(stopped, dict):
        # ``_stopped_early_problems`` already says it is neither null nor the
        # whole block; there is no digest here to compare.
        return {"problems": problems, "stop_body": None}

    report_digest = stopped.get("stop_request_sha256")
    accounts = (("the finished event", fin_digest),
                ("the report's stopped_early", report_digest))
    for who, value in accounts:
        if not is_sha256(value):
            problems.append(
                f"the report says it stopped early, but {who} records "
                f"stop_request_sha256 {value!r}, which is not a sha-256 "
                "digest: a stop request nothing names by digest is a stop "
                "nobody can check")
    if not sp.exists():
        problems.append(
            f"the report says it stopped early, but {sp.name} does not "
            "exist: there is nothing that asked it to")
        return {"problems": problems, "stop_body": None}

    got = read_stop_request(d, prefix=run, nonce=nonce)
    if not got.get("accepted"):
        problems.append(
            f"the stop request does not authenticate "
            f"({got.get('detail') or got.get('reason')})")
        return {"problems": problems, "stop_body": None}
    on_disk = got["sha256"]
    if sha256_file(sp) != on_disk:
        problems.append(
            f"{sp.name} hashes to one thing as bytes and another as text; "
            "the stop request cannot be named by a single digest")
    for who, value in accounts:
        if value != on_disk:
            problems.append(
                f"{who} records the stop request as {str(value)[:16]}…, but "
                f"{sp.name} hashes to {on_disk[:16]}…")

    body = got["body"]
    for field in ("reason", "rule"):
        if stopped.get(field) != body.get(field):
            problems.append(
                f"stopped_early.{field} is {stopped.get(field)!r} but the stop "
                f"request says {body.get(field)!r}: the run and the request "
                "that stopped it do not describe the same stop")
    if body.get("nonce") != nonce:
        problems.append("the stop request carries another launch's nonce")
    return {"problems": problems, "stop_body": body}


def finished_eligibility_problems(finished_body: dict | None) -> list[str]:
    """What the finished event itself has to say for a run to be completed.

    The artefacts can all be in order while the event that files them says
    something else. A finished event carrying ``exit_status: 1`` -- with the
    journal chain repaired so nothing else notices -- replayed clean and
    still fired rule R1, because this half of the definition lived inside
    :func:`qualify_outcome` and nowhere else, and R1 calls the shared
    validator directly.

    ``False`` is not a clean exit and neither is ``"0"``. A digest that is
    absent is not a digest that matched: the comparisons below are guarded by
    truthiness, so without this the missing case falls through to "nothing
    wrong here".
    """
    if finished_body is None:
        return ["there is no finished event to say what this run did"]
    problems = []
    outcome = finished_body.get("outcome")
    if outcome != "completed":
        problems.append(f"the finished event records outcome {outcome!r}, not "
                        "'completed'")
    failure = finished_body.get("tool_failure")
    if failure is not None:
        problems.append(f"the finished event records tool failure {failure!r}, "
                        "so this run is not a measurement")
    if not _is_clean_exit(finished_body.get("exit_status")):
        problems.append(f"exit status {finished_body.get('exit_status')!r} is "
                        "not the integer 0")
    for name, key in (("report", "report_sha256"),
                      ("watchdog log", "watchdog_sha256")):
        if not finished_body.get(key):
            problems.append(f"no {name} digest was recorded")
    return problems


#: Tool-failure names in the order the evidence is judged. The first group
#: that produces a problem names the failure, so a run is filed under the
#: thing that is actually wrong with it rather than under whichever check
#: happened to be written last.
EVIDENCE_ORDER = ("no_report", "identity_mismatch", "stop_request_rejected",
                  "watchdog_unsealed", "report_invalid")


def _evidence_result(groups: dict, report, replay) -> dict:
    problems = [p for key in EVIDENCE_ORDER for p in groups[key]]
    return {"problems": problems,
            "tool_failure": next((k for k in EVIDENCE_ORDER if groups[k]), None),
            "report": report, "replay": replay}


def completed_run_evidence(paths, run: str, *, spec: SafetySpec | None = None,
                           plan: dict | None = None, entry: dict | None = None,
                           session: dict | None = None,
                           started_body: dict | None = None,
                           finished_body: dict | None = None) -> dict:
    """The single definition of "this run may be called completed".

    One function, three callers: :func:`qualify_outcome` before the finished
    event is written, and the two post-hoc replays afterwards. Two definitions
    of ``completed`` is exactly how a run passes the gate and fails
    ``--verify`` -- and by then the boot is spent, the journal says completed,
    and rule R1 may already have cancelled the remaining arms on the strength
    of it.

    The earlier version of this checked digests and the watchdog log and never
    replayed the report, which is the document every verdict is computed from.
    A report whose ``model_compute_seconds`` said 999999 and whose per-row
    array said otherwise was recorded as completed; an invalid
    ``stopped_early`` claiming a safety trip did not merely pass, it fired R1.

    The finished event is part of the evidence, not the thing the evidence is
    weighed against: its outcome, its tool_failure, its exit status and its
    two digests are checked here, so R1 and the aggregate cannot be handed a
    run the pre-finish gate would have refused. The two digests are read from
    that event and from nowhere else, so there is one account of what was
    recorded rather than two that can be made to differ.

    Anything not supplied is read from the session, so a caller cannot weaken
    the check by leaving an argument out.
    """
    d = Path(paths["dir"])
    if plan is None and (d / "plan.json").exists():
        plan = json.loads((d / "plan.json").read_text())
    if session is None and (d / "session.json").exists():
        session = json.loads((d / "session.json").read_text())
    if started_body is None:
        started_body = next(
            (ev.get("body") or {} for ev in read_journal(d)
             if ev.get("event") == EVENT_STARTED and ev.get("run_id") == run),
            {})
    if finished_body is None:
        finished_body = next(
            (ev.get("body") or {} for ev in read_journal(d)
             if ev.get("event") == EVENT_FINISHED and ev.get("run_id") == run),
            None)
    plan, session = plan or {}, session or {}
    if entry is None:
        entry = next((e for e in plan.get("runs") or []
                      if e.get("run_id") == run), None)
    if spec is None:
        try:
            spec = plan_spec(plan)
        except Exception:  # noqa: BLE001 - a plan we cannot read is caught below
            spec = SafetySpec()

    groups: dict[str, list[str]] = {key: [] for key in EVIDENCE_ORDER}
    groups["no_report"] += finished_eligibility_problems(finished_body)
    report_sha256 = (finished_body or {}).get("report_sha256")
    watchdog_sha256 = (finished_body or {}).get("watchdog_sha256")

    rp = d / f"{run}.json"
    if not rp.exists():
        groups["no_report"].append(f"{rp.name} does not exist")
        return _evidence_result(groups, None, None)
    try:
        report = json.loads(rp.read_text())
    except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001
        groups["no_report"].append(f"{rp.name} cannot be read ({exc})")
        return _evidence_result(groups, None, None)
    if report_sha256 and sha256_file(rp) != report_sha256:
        groups["no_report"].append(f"{rp.name} does not match the digest that "
                                   "was recorded for it")

    # who this run is, according to the plan and the launch it came from
    if entry is None:
        groups["report_invalid"].append(f"the plan declares no run {run!r}")
    else:
        groups["report_invalid"] += child_identity_problems(
            report, entry, plan, started_body)
    if session:
        groups["report_invalid"] += child_source_check_problems(
            report, session=session, plan=plan)
    groups["identity_mismatch"] += launch_identity_problems(
        d, run, started_body, report=report, finished_body=finished_body)

    wpath = d / f"{run}.watchdog.jsonl"
    if not wpath.exists():
        groups["watchdog_unsealed"].append(
            f"{wpath.name} does not exist, so nothing recorded how the run "
            "was supervised")
    elif watchdog_sha256 and sha256_file(wpath) != watchdog_sha256:
        groups["watchdog_unsealed"].append(
            f"{wpath.name} does not match the digest that was recorded for it")

    # Both directions of the stop invariant: a report may not claim a stop
    # the request does not support, and it may not stay silent about one the
    # rest of the session recorded.
    stop = stop_evidence_problems(d, run, report=report,
                                  finished_body=finished_body,
                                  started_body=started_body)
    groups["stop_request_rejected"] += stop["problems"]
    stop_body = stop["stop_body"]

    replay = replay_child(report,
                          watchdog_path=wpath if wpath.exists() else None,
                          watchdog_sha256=watchdog_sha256, spec=spec,
                          stop_request=stop_body,
                          launch_nonce=started_body.get("nonce"),
                          require_terminal=True)
    for problem in replay["problems"]:
        key = "watchdog_unsealed" if "watchdog" in problem \
            or "sigterm" in problem or "child_exit_observed" in problem \
            else "report_invalid"
        groups[key].append(problem)
    return _evidence_result(groups, report, replay)


def qualify_outcome(paths, run: str, *, claimed: str, exit_status,
                    report_sha256, watchdog_sha256, tool_failure=None,
                    spec: SafetySpec | None = None) -> dict:
    """Decide the outcome from the evidence, not from what the caller claims.

    ``completed`` is the only outcome that lets a run become a verdict, so it
    is the only one that has to be earned: a genuine integer zero exit, a
    report that exists and hashes to what was recorded, a watchdog log that
    does the same, and -- the part a matching digest cannot supply -- a log
    that reaches the end of the run.

    All of that is :func:`completed_run_evidence`, and it has to happen
    *here*, before the finished event is written, because the finished event
    is what records the outcome -- and because rule R1 acts on it immediately
    afterwards. A check that only runs in ``--verify`` describes a mistake
    after the boot is spent, the journal says completed, and the remaining
    arms may already have been cancelled.
    """
    if claimed != "completed":
        return {"outcome": claimed, "problems": []}
    # The finished event does not exist yet, so the evidence is judged
    # against the event that is about to be written. That is what keeps this
    # one definition rather than two: the same function is asked the same
    # question before and after the fact, from the same fields.
    prospective = {"outcome": claimed, "tool_failure": tool_failure,
                   "exit_status": exit_status,
                   "report_sha256": report_sha256,
                   "watchdog_sha256": watchdog_sha256,
                   "launch_sha256": _launch_digest(paths, run),
                   "stop_request_sha256": _stop_request_digest(paths, run)}
    evidence = completed_run_evidence(paths, run, spec=spec,
                                      finished_body=prospective)
    if evidence["problems"]:
        return {"outcome": "no_report", "problems": evidence["problems"],
                "tool_failure": evidence["tool_failure"]}
    return {"outcome": "completed", "problems": []}


def _finish(paths, run, plan, fingerprint, *, outcome, tool_failure=None,
            report_sha256=None, watchdog_sha256=None, exit_status=None,
            detail=None, started_event=None) -> dict:
    if outcome not in FINISHED_OUTCOMES:
        raise ValueError(f"{outcome!r} is not one of {FINISHED_OUTCOMES}")
    begin = started_event or next(
        (e for e in read_journal(paths["dir"])
         if e["event"] == EVENT_STARTED and e["run_id"] == run), None)
    if begin is None:
        raise ValueError(f"{run} has no measurement_started to point back at")
    return append_session_event(paths, run, EVENT_FINISHED, {
        "boot_fingerprint": fingerprint, "outcome": outcome,
        "tool_failure": tool_failure, "exit_status": exit_status,
        "report_sha256": report_sha256, "watchdog_sha256": watchdog_sha256,
        # The launch record is the parent's only independent account of which
        # process was measured, so it is pinned here like every other piece of
        # evidence rather than left as a file anyone could rewrite.
        "launch_sha256": _launch_digest(paths, run),
        "stop_request_sha256": _stop_request_digest(paths, run),
        # A run that died without a report used to leave only an exit code.
        # If the child published why, the digest is pinned here like every
        # other artefact, so the reason cannot be edited afterwards.
        "failure_evidence_sha256": _failure_evidence_digest(paths, run),
        "detail": detail, "finished_at": now_iso(),
        "started_index": begin["index"],
        "started_event_digest": begin["file_sha256"]}, plan=plan)


def _stop_request_digest(paths, run: str) -> str | None:
    path = paths["dir"] / f"{run}.stop_request.json"
    return sha256_file(path) if path.exists() else None


def _failure_evidence_digest(paths, run: str) -> str | None:
    path = paths["dir"] / f"{run}.failure.json"
    return sha256_file(path) if path.exists() else None


def _launch_digest(paths, run: str) -> str | None:
    path = paths["dir"] / f"{run}.launch.json"
    return sha256_file(path) if path.exists() else None


def apply_rule_r1(experiment_id: str, *, root=None) -> dict:
    """Recompute the last completed run's Q1 and cancel by rule if it failed."""
    paths = session_paths(experiment_id, root=root)
    with exclusive_lock(paths["lock"], description=f"report 16 r1 {experiment_id}"):
        return _apply_rule_r1_locked(experiment_id, root=root)


def _apply_rule_r1_locked(experiment_id: str, *, root=None) -> dict:
    st = session_state(experiment_id, root=root)
    if "plan" not in st or not st["completed"]:
        return {"fired": False, "why": "no completed run to judge"}
    run = st["completed"][-1]
    paths, plan = st["paths"], st["plan"]
    rp = paths["dir"] / f"{run}.json"
    if not rp.exists():
        return {"fired": False, "why": f"{run} has no report"}
    report = json.loads(rp.read_text())
    fin = st["finished"][run].get("body") or {}
    started_body = (st["started"].get(run) or {}).get("body") or {}
    wpath = paths["dir"] / f"{run}.watchdog.jsonl"
    # R1 acts on a verdict, and a verdict is computed from evidence. A rule
    # that trusts whatever the journal calls completed is a rule that can be
    # handed a forged premise: an invalid stopped_early claiming a safety trip
    # used to cancel b2 and b3 on the strength of a block that does not even
    # carry the row it stopped at.
    evidence = completed_run_evidence(
        paths, run, spec=plan_spec(plan), plan=plan, session=st["session"],
        started_body=started_body, finished_body=fin)
    if evidence["problems"]:
        return {"fired": False, "evidence_failure": True,
                "tool_failure": evidence["tool_failure"],
                "why": "this run's evidence does not replay, so there is no "
                       "verdict for a rule to act on",
                "problems": evidence["problems"]}
    out = evidence["replay"]
    q1 = q1_run(out["metrics"],
                safety_reason=(report.get("stopped_early") or {}).get("reason"),
                tool_failure=fin.get("tool_failure"))
    decision = r1_should_cancel(outcome=fin.get("outcome"),
                                tool_failure=fin.get("tool_failure"),
                                q1_verdict=q1["verdict"], q1_reason=q1["reason"])
    if not decision["cancel"]:
        return {"fired": False, "why": decision["why"]}
    order = [e["run_id"] for e in plan["runs"]]
    cancelled = order[order.index(run) + 1:]
    if not cancelled:
        return {"fired": False, "why": "nothing left to cancel"}
    body = plan_arm_cancelled_event(
        experiment_id=plan["experiment_id"], digest=plan["plan_digest"],
        run_id=run, outcome=fin.get("outcome"),
        verdict={"Q1": q1["verdict"], "reason": q1["reason"],
                 "D100": out["metrics"].get("D100"),
                 "Dmax": out["metrics"].get("Dmax")},
        cancelled=cancelled,
        report_sha256=fin.get("report_sha256"),
        watchdog_sha256=fin.get("watchdog_sha256")
        or (sha256_file(wpath) if wpath.exists() else None))
    append_session_event(paths, run, EVENT_PLAN_ARM_CANCELLED, body, plan=plan)
    return {"fired": True, "cancelled": cancelled, "verdict": q1}


# ---------------------------------------------------------------------------
# Child-side helpers, so a dummy child in a test uses the same contract
# ---------------------------------------------------------------------------


def write_progress(path, *, row: int, row_elapsed_seconds: float,
                   condition_clock_seconds: float,
                   process_clock_seconds: float) -> None:
    """Append one progress line. Durations only -- instants do not travel."""
    line = json.dumps({"row": row, "row_elapsed_seconds": row_elapsed_seconds,
                       "condition_clock_seconds": condition_clock_seconds,
                       "process_clock_seconds": process_clock_seconds},
                      ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# The two worker entry points. Both are real; neither is reachable, because
# the CLI refuses --child and --watchdog-worker alongside the session flags.
# ---------------------------------------------------------------------------


def run_watchdog_worker(*, experiment_dir: Path, run: str, handshake_fd: int,
                        heartbeat_fd: int, spec: SafetySpec | None = None,
                        probe=None, max_polls: int = 10 ** 9) -> int:
    """Receive the child's identity, then poll and enforce until it is over."""
    import time

    from src.training.watchdog import (ChildIdentity, LineReader, WatchdogLog,
                                       _signal_pgid, default_probe,
                                       observed_pgid, process_start_identity,
                                       read_progress_tail, watchdog_loop)

    spec = spec or SafetySpec()
    probe = probe or default_probe
    os.write(heartbeat_fd, b'{"ready": true}\n')
    reader = LineReader(handshake_fd)
    identity, deadline = None, time.monotonic() + spec.ready_timeout_seconds
    while identity is None and time.monotonic() < deadline:
        for msg in reader.poll():
            identity = ChildIdentity.from_dict(msg)
        time.sleep(0.01)
    if identity is None:
        return 3
    os.write(heartbeat_fd, b'{"armed": true}\n')

    def alive() -> bool:
        try:
            os.kill(identity.pid, 0)
            return True
        except OSError:
            return False

    log = WatchdogLog(experiment_dir / f"{run}.watchdog.jsonl")
    watchdog_loop(
        log, identity, spec, probe=probe, clock=time.monotonic,
        child_alive=alive, signaller=_signal_pgid,
        heartbeat=lambda beat: os.write(
            heartbeat_fd, (json.dumps(beat) + "\n").encode("utf-8")),
        observed_start=lambda: process_start_identity(identity.pid),
        observed_pgid_fn=lambda: observed_pgid(identity.pid),
        directory=experiment_dir, prefix=run,
        read_progress=lambda: read_progress_tail(
            experiment_dir / f"{run}.progress.jsonl"),
        wall_clock=now_iso, sleep=time.sleep, max_polls=max_polls)
    log.seal()
    return 0


#: What ``ChildDeps.load`` must hand back. Round 19 wrapped every stage the
#: child *runs* and left this seam bare: ``run_child`` indexed straight into
#: the mapping, so a loader returning ``None``, or a dict without ``order``,
#: died with an unpublished ``AttributeError`` or ``KeyError`` -- with the
#: model it had just built still on the device.
CHILD_LOAD_REQUIRED = ("order", "step", "sample_ids", "provenance",
                       "model_load_seconds")

#: Callable if supplied at all. ``step`` is required as well; the other three
#: have defaults, but a non-callable one is a broken loader, not an absent
#: option, and silently substituting the default would hide that.
CHILD_LOAD_CALLABLES = ("step", "teardown", "clear", "probe")


def _is_finite_seconds(value) -> bool:
    """A real, finite, non-negative duration. ``True`` is not one of those."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0


def child_load_problems(loaded, *, rows: int) -> list[str]:
    """Check the loader's return value before a single row is measured.

    Round 20 asked only whether the keys were present, which let ``step=None``
    through to die at row one, and let ``sample_ids``, ``provenance`` and
    ``model_load_seconds`` through entirely unexamined -- they are first
    touched when the report is built, by which point the run has happened and
    the numbers have nowhere to go. Every field is checked for what it has to
    be, and all of it happens with the model already built and therefore still
    to be released.
    """
    if not isinstance(loaded, dict):
        return [f"the loader returned {type(loaded).__name__}, not a mapping"]
    problems = [f"the loader returned no {field!r}"
                for field in CHILD_LOAD_REQUIRED if field not in loaded]
    for field in CHILD_LOAD_CALLABLES:
        if field not in loaded:
            continue
        value = loaded[field]
        # ``teardown``/``clear``/``probe`` have documented defaults, so a null
        # one means "not supplied". ``step`` is what does the measuring: a
        # null one is a broken loader, not an omission.
        if value is None and field not in CHILD_LOAD_REQUIRED:
            continue
        if not callable(value):
            problems.append(f"the loader's {field!r} is "
                            f"{type(value).__name__}, not callable")

    if "order" in loaded:
        order = loaded["order"]
        if not isinstance(order, (list, tuple)):
            problems.append(f"the loader's 'order' is "
                            f"{type(order).__name__}, not a list or tuple")
        elif len(order) != rows:
            problems.append(f"the order holds {len(order)} rows, not {rows}")
        else:
            # Every entry indexes the frozen pool. A float or a string raises
            # at row one; a bool silently reads row 0 or 1; a negative index
            # silently measures the wrong end of the pool and the report would
            # look entirely ordinary.
            bad = next((i for i, v in enumerate(order)
                        if isinstance(v, bool) or not isinstance(v, int)
                        or v < 0), None)
            if bad is not None:
                problems.append(
                    f"the loader's 'order' holds {type(order[bad]).__name__} "
                    f"{order[bad]!r} at position {bad}; every entry has to be "
                    "a non-negative pool index")
            else:
                # ``POOL_ROWS`` is one past the end, so production would raise
                # IndexError there -- after the model had loaded. A value just
                # inside the pool but repeated raises nothing at all: one row
                # gets measured twice and another never, and every number in
                # the report looks ordinary.
                over = sorted({v for v in order if v >= POOL_ROWS})
                if over:
                    problems.append(
                        f"the loader's 'order' names pool row(s) {over[:3]}, "
                        f"but the pool holds {POOL_ROWS} rows (0 to "
                        f"{POOL_ROWS - 1})")
                seen, repeated = set(), []
                for value in order:
                    if value in seen and value not in repeated:
                        repeated.append(value)
                    seen.add(value)
                if repeated:
                    problems.append(
                        f"the loader's 'order' repeats pool row(s) "
                        f"{sorted(repeated)[:3]}: {len(order) - len(seen)} "
                        "duplicate entries, so a row would be measured twice "
                        "and another not at all")

    if "sample_ids" in loaded:
        ids = loaded["sample_ids"]
        if not isinstance(ids, (list, tuple)):
            problems.append(f"the loader's 'sample_ids' is "
                            f"{type(ids).__name__}, not a list or tuple")
        elif len(ids) != rows:
            # This is what `input_order_digest` is computed from, so a wrong
            # length is a provenance claim about rows that were not measured.
            problems.append(f"the loader's 'sample_ids' holds {len(ids)} "
                            f"entries, not {rows}")
        else:
            bad = next((i for i, v in enumerate(ids)
                        if not isinstance(v, str) or not v), None)
            if bad is not None:
                problems.append(
                    f"the loader's 'sample_ids' holds "
                    f"{type(ids[bad]).__name__} {ids[bad]!r} at position "
                    f"{bad}; every entry has to be a non-empty string")
            elif len(set(ids)) != len(ids):
                problems.append(
                    f"the loader's 'sample_ids' holds "
                    f"{len(ids) - len(set(ids))} duplicate id(s); "
                    "input_order_digest is a hash of exactly these, so "
                    "duplicates let two different orders hash the same")

    if "provenance" in loaded:
        prov = loaded["provenance"]
        if not isinstance(prov, dict):
            problems.append(f"the loader's 'provenance' is "
                            f"{type(prov).__name__}, not a mapping")
        else:
            # Verbatim from the validator the replay uses, so the child cannot
            # publish a report that the replay was always going to refuse.
            problems += provenance_problems(prov, declared_rows=rows)

    if "model_load_seconds" in loaded \
            and not _is_finite_seconds(loaded["model_load_seconds"]):
        problems.append(
            f"the loader's 'model_load_seconds' is "
            f"{loaded['model_load_seconds']!r}, which is not a finite "
            "non-negative number of seconds")
    return problems


class ChildDeps:
    """Everything the child needs from outside itself.

    Production supplies a tokenizer, the frozen row pool, the model, the LoRA
    adapter and an optimizer. Tests supply fakes. Either way the code below is
    the same code -- which is the point: a child that is only ever exercised
    through a stand-in is a child nobody has tested.

    ``load`` returns ``{order, step, provenance, sample_ids,
    model_load_seconds, teardown, clear, probe}``. ``step(index, position)``
    takes the pool index of the row and its 1-based position in this run: the
    optimizer boundary is a property of the position, not of the row.
    """

    def load(self, *, rows: int) -> dict:
        raise NotImplementedError


def prepare_training(*, torch_mod, cfg, device: str, build, assert_trainable):
    """Seed, build, check, optimise, train mode -- report 15's order exactly.

    The seed goes in before the model is built because LoRA dropout draws from
    the global stream while the adapter is created; seeding afterwards would
    leave b1, b2 and b3 on three different dropout sequences and make their
    shared prefix disagree for a reason that has nothing to do with the cache.

    ``model.train()`` matters for the same reason from the other direction: an
    adapter left in eval mode applies no dropout at all, which is a different
    computation from the one exp002 measured.
    """
    import time

    torch_mod.manual_seed(cfg.seed)
    t_load = time.perf_counter()
    model, info = build(cfg, device=device)
    load_seconds = time.perf_counter() - t_load
    assert_trainable(model)
    optimizer = torch_mod.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    model.train()
    optimizer.zero_grad()
    return {"model": model, "optimizer": optimizer, "info": info,
            "model_load_seconds": load_seconds}


def make_training_step(*, holder: dict, encs, sample_ids, collate_fn, pad_id,
                       device: str, grad_accum: int, to_device=None):
    """One measured row, with report 15's arithmetic.

    Three things are easy to get subtly wrong and expensive to notice later:

    * the loss that is *stored* is the undivided one, while the loss that is
      *backpropagated* is divided by ``grad_accum``. Storing the divided value
      would make report 16's per-row losses incomparable with exp002's, which
      is the one cross-report check section 4.8.2 asks for;
    * the optimizer steps once per ``grad_accum`` rows, not once per row. An
      optimizer that steps every row trains a different model, so the losses
      diverge from row nine onwards;
    * the token counts come from the collated batch. ``Encoded`` carries ids
      and labels and no counts, so ``attention_mask.sum()`` and
      ``(labels != -100).sum()`` are where the numbers actually live.

    ``holder`` rather than closed-over locals so :func:`make_teardown` can drop
    the model and the optimizer before the teardown clear is measured.
    """
    to_device = to_device or (lambda t: t.to(device))

    def step(index: int, position: int) -> dict:
        model, optimizer = holder["model"], holder["optimizer"]
        batch = {k: to_device(v) for k, v in
                 collate_fn([encs[index]], pad_id).items()}
        loss = model(**batch).loss
        (loss / grad_accum).backward()
        if position % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        return {"loss": float(loss.detach().item()),
                "tokens": int(batch["attention_mask"].sum()),
                "supervised_tokens": int((batch["labels"] != -100).sum()),
                "sample_id": sample_ids[index]}

    return step


def make_teardown(torch_mod, holder: dict):
    """Drop the model and the optimizer, *then* measure the teardown clear.

    Report 15 does ``del model, opt`` before its teardown clear for a reason:
    a clear measured while a merged 1B model, an adapter and Adam's two moment
    tensors are still referenced is measuring how much cannot be freed, not
    how long freeing takes.
    """
    def teardown() -> float:
        holder.pop("model", None)
        holder.pop("optimizer", None)
        return _empty_cache(torch_mod)

    return teardown


class ProductionChildDeps(ChildDeps):
    """The real thing: tokenizer, dataset, base model, adapter, AdamW.

    Imports live inside ``load`` so that importing this module never pulls in
    torch, and so the locked CLI can be exercised in tests without a model
    being anywhere near the process. They also name the modules that actually
    export these functions: ``read_rows`` is in ``src.training.lora`` and
    ``load_tokenizer`` in ``src.generation.brickgpt``, and an ImportError here
    would arrive after a boot had already been spent.
    """

    def __init__(self, *, device: str = "mps", cfg=None,
                 source: str = "pool"):
        self.device = device
        # Which rows to read. ``pool`` is the default and is what every
        # measurement so far used; ``full_train`` is the whole split and is
        # read only by the final run. Named rather than sized: a caller that
        # could pass a row count could pass a different experiment.
        if source not in DATA_SOURCES:
            raise ValueError(
                f"{source!r} is not one of {sorted(DATA_SOURCES)}; this loader "
                "reads the frozen pool or the whole training split, and "
                "nothing in between")
        self.source = source
        # The configuration to build with. ``None`` means the project default,
        # which is what every measurement so far has used. It is injectable
        # only because H1 and H2 are two *frozen* configurations that differ in
        # rank, alpha and learning rate, and a loader that could not be told
        # which one to build would need a second loader -- which is how the
        # two arms would end up differing in something nobody chose.
        #
        # This is not an override hole: the only caller that passes anything
        # is :mod:`src.training.arms`, and it gets the value from
        # :func:`src.training.hypotheses.config_for` and from nowhere else.
        # No command line reaches this.
        self._cfg = cfg

    def load(self, *, rows: int) -> dict:
        # First statement in the function, ahead of every import below, and
        # that ordering is load-bearing rather than tidy. ``huggingface_hub``
        # reads ``HF_HUB_OFFLINE`` once, at *its own* import time, and
        # ``peft.load_peft_weights`` calls ``huggingface_hub.file_exists()``
        # on a path that ignores ``local_files_only`` entirely. Measured in
        # round 19: pinned before the imports, the whole load makes zero
        # network calls; pinned after them, the adapter load still reaches for
        # the hub. The parent also spawns the child with
        # ``offline_environment()`` and ``run_child`` pins it again at entry,
        # so a measured run has three independent guarantees -- but this one
        # is what makes calling ``load()`` directly safe too.
        enforce_offline_environment()
        # The pin above is necessary and, on its own, not sufficient: if the
        # caller already imported the hub while it was online, the flag is
        # frozen and no assignment here can thaw it. Refuse now -- before the
        # tokenizer, before the model, before the adapter, and before any
        # network call -- rather than load bytes that may have been fetched.
        frozen = offline_freeze_problems()
        if frozen:
            raise OfflineNotGuaranteed(_portable(" ".join(frozen)))

        import platform
        from importlib.metadata import version

        import torch

        from src.generation.brickgpt import load_tokenizer
        from src.model_ids import (ADAPTER, ADAPTER_REVISION, BASE_MODEL,
                                   BASE_REVISION, TOKENIZER_REVISION)
        from src.training.lora import (LoraConfig_, assert_only_lora_trainable,
                                       build_model, collate, encode_row,
                                       read_rows, sample_pairs)

        cfg = self._cfg if self._cfg is not None else LoraConfig_()
        # Strictly local, stated rather than inherited: a measured run either
        # resolves every byte from this machine's cache or refuses. It does
        # not discover halfway through a spent boot that a name has to be
        # looked up.
        tok = load_tokenizer(local_files_only=True)
        shape = DATA_SOURCES[self.source]
        pool = sample_pairs(read_rows(ROOT / "data" / "processed"
                                      / "instruct_inv_train.jsonl"),
                            n_pairs=shape["pairs"], seed=cfg.seed)
        if len(pool) != shape["rows"]:
            raise ValueError(
                f"the {self.source} source holds {len(pool)} rows, not the "
                f"declared {shape['rows']}")
        encs = [encode_row(tok, r, cfg.max_length) for r in pool]
        # Counted here, where the encoder is, and handed to the caller.
        # Truncation is silent by construction -- a truncated row trains on a
        # target that stops mid-structure and reports a perfectly ordinary
        # loss -- so the reading has to travel with the loader rather than be
        # recomputed by whoever remembers to.
        truncated_rows = sum(1 for e in encs if e.truncated)
        max_total_tokens = max(len(e.input_ids) for e in encs)

        # One permutation over the whole pool, computed once from the frozen
        # seed, and only then truncated. b1/b2/b3 are prefixes of the same
        # order, and that is only true if nothing is reshuffled per length.
        rng = torch.Generator().manual_seed(cfg.seed)
        full_order = torch.randperm(len(encs), generator=rng).tolist()
        order = full_order[:rows]

        prepared = prepare_training(
            torch_mod=torch, cfg=cfg, device=self.device,
            build=lambda c, *, device: build_model(c, device=device,
                                                   local_files_only=True),
            assert_trainable=assert_only_lora_trainable)
        info = prepared["info"]
        holder = {"model": prepared["model"],
                  "optimizer": prepared["optimizer"]}
        step = make_training_step(
            holder=holder, encs=encs,
            sample_ids=[r.sample_id for r in pool], collate_fn=collate,
            pad_id=tok.eos_token_id, device=self.device,
            grad_accum=cfg.grad_accum)

        provenance = {
            "code_sha256": {rel: sha256_file(ROOT / rel) for rel in CODE_FILES
                            if (ROOT / rel).exists()},
            "instruction_sha256": {"instruct_inv_train.jsonl": sha256_file(
                ROOT / "data" / "processed" / "instruct_inv_train.jsonl")},
            "selection_digest": _digest_ids(r.sample_id for r in pool),
            # The whole 2,000-row permutation, not this run's prefix of it:
            # section 4.6 requires the three runs to agree on this field, and
            # what they agree on is the order the pool was permuted into
            # before anybody chose a length. The per-run prefix is recorded
            # separately, as input_order_digest.
            "training_order_digest": _digest_ids(pool[i].sample_id
                                                 for i in full_order),
            "lora_config": cfg.as_dict(),
            "optimizer": {"class": "AdamW", "lr": cfg.learning_rate,
                          "betas": [0.9, 0.999], "eps": 1e-8,
                          "weight_decay": 0.01,
                          "grad_accum": cfg.grad_accum},
            "packages": {"python": platform.python_version(),
                         "torch": torch.__version__,
                         "transformers": version("transformers"),
                         "peft": version("peft")},
            "device": self.device, "dtype": cfg.dtype,
            "phases": ["collate_h2d", "forward", "backward", "optimizer"],
            "stop_conditions": plan_safety_summary(),
            "measurement_intervals": {"window": WINDOW,
                                      "memory_every": MEMORY_EVERY,
                                      "empty_cache_every": EMPTY_CACHE_EVERY,
                                      "max_rows": rows},
            "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
            "published_adapter": info.get("published_adapter", ADAPTER),
            "published_adapter_revision": info.get(
                "published_adapter_revision", ADAPTER_REVISION),
            "tokenizer_revision": TOKENIZER_REVISION,
            "trainable_parameters": info.get("trainable_parameters"),
        }
        return {"order": order, "step": step, "provenance": provenance,
                "data_source": self.source,
                "truncated_rows": truncated_rows,
                "max_total_tokens": max_total_tokens,
                "sample_ids": [pool[i].sample_id for i in order],
                "model_load_seconds": prepared["model_load_seconds"],
                "teardown": make_teardown(torch, holder),
                "clear": lambda: _empty_cache(torch),
                "probe": _driver_probe(torch),
                # Additive, and only for callers that have to reach the model
                # itself -- the gate runner saves the adapter and the optimizer
                # state, which nothing here measures. Report 16 ignores it:
                # ``child_load_problems`` checks the fields it requires and is
                # indifferent to extra ones. The alternative was a second
                # loader with a second copy of the provenance block above, and
                # two places for "which weights was this" to be defined is
                # exactly one too many.
                "holder": holder}


class FakeChildDeps(ChildDeps):
    """A stand-in with the same shape, for tests. Loads nothing.

    It follows the same rules the production loader follows, because a fake
    that is allowed to be sloppy about the full-pool permutation or about
    ``max_rows`` is a fake that hides exactly the cross-run failures those
    rules exist to prevent.
    """

    def __init__(self, *, pool_rows: int = POOL_ROWS, seed: int = SEED,
                 row_seconds: float = 0.0, stop_at_row: int | None = None,
                 loss: float = 0.5):
        self.pool_rows = pool_rows
        self.seed = seed
        self.row_seconds = row_seconds
        self.stop_at_row = stop_at_row
        self.loss = loss
        self.released = False

    def load(self, *, rows: int) -> dict:
        import random
        import time

        rng = random.Random(self.seed)
        full_order = list(range(self.pool_rows))
        rng.shuffle(full_order)
        order = full_order[:rows]
        holder = {"model": object(), "optimizer": object()}

        def step(index: int, position: int) -> dict:
            if holder.get("model") is None:
                raise AssertionError("a row was measured after teardown")
            if self.row_seconds:
                time.sleep(self.row_seconds)
            # Derived from the row, not from the position, so two runs sharing
            # a prefix must agree on them the way real token counts do.
            return {"loss": self.loss, "tokens": 100 + index % 7,
                    "supervised_tokens": 70 + index % 5,
                    "sample_id": f"s{index}"}

        def teardown() -> float:
            holder.pop("model", None)
            holder.pop("optimizer", None)
            self.released = True
            return 0.0

        provenance = {f: ({} if f.endswith(("sha256", "digest", "config",
                                            "optimizer", "packages",
                                            "conditions"))
                          else "fake") for f in PROVENANCE_FIELDS}
        provenance["selection_digest"] = _digest_ids(
            f"s{i}" for i in range(self.pool_rows))
        provenance["training_order_digest"] = _digest_ids(
            f"s{i}" for i in full_order)
        provenance["measurement_intervals"] = {
            "window": WINDOW, "memory_every": MEMORY_EVERY,
            "empty_cache_every": EMPTY_CACHE_EVERY, "max_rows": rows}
        return {"order": order, "step": step, "provenance": provenance,
                "sample_ids": [f"s{i}" for i in order],
                "model_load_seconds": 0.0, "teardown": teardown,
                "clear": lambda: 0.0, "probe": lambda: {}}


def _digest_ids(ids) -> str:
    h = hashlib.sha256()
    for i in ids:
        h.update(str(i).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _mps_available(torch) -> bool:
    """Is there an MPS *backend*, not merely an MPS attribute?

    ``torch.mps`` and ``torch.mps.empty_cache`` exist on every build,
    including CUDA ones -- they are present and they raise. The node found
    this the only way it could be found: the Mac has a real MPS backend, so
    the attribute check is accidentally correct here and wrong everywhere
    else.
    """
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        # A torch that cannot answer is treated as "no". An unavailable
        # answer is not a yes, and guessing yes is what raised on the node.
        return False


def _empty_cache(torch) -> float:
    import time

    t0 = time.perf_counter()
    if _mps_available(torch) and hasattr(getattr(torch, "mps", None),
                                         "empty_cache"):
        torch.mps.empty_cache()
    return time.perf_counter() - t0


def _driver_probe(torch):
    def probe() -> dict:
        from src.training.watchdog import default_probe

        out = dict(default_probe())
        if hasattr(torch, "mps"):
            for name, fn in (("mps_current_allocated_gb",
                              getattr(torch.mps, "current_allocated_memory", None)),
                             ("mps_driver_allocated_gb",
                              getattr(torch.mps, "driver_allocated_memory", None)),
                             ("mps_recommended_max_gb",
                              getattr(torch.mps, "recommended_max_memory", None))):
                if fn is not None:
                    out[name] = round(fn() / 1024 ** 3, 3)
        return out

    return probe


def plan_safety_summary() -> dict:
    spec = SafetySpec()
    return {k: getattr(spec, k) for k in
            ("slow_row_seconds", "slow_row_streak", "max_seconds",
             "process_max_seconds")}


def _own_start_identity() -> str | None:
    """This process's start time, read the same way the parent reads it."""
    from src.training.watchdog import process_start_identity

    return process_start_identity(os.getpid())


def child_verify_source(experiment_dir, plan_digest_expected) -> dict:
    """The child re-derives the plan digest and the source, before it loads.

    The parent checked the same thing a moment earlier, and that is not
    enough: the working tree stayed writable in between. More importantly the
    digest arrives on the command line, and argv is an input, not evidence --
    a child that copied it into its report would produce a field that agrees
    with itself no matter what ``plan.json`` says.

    So the digest is recomputed from the plan on disk and compared three ways
    (plan against itself, plan against argv, session against argv), the
    manifest is rehashed, and every snapshot copy is compared with the live
    file. All of it happens before the tokenizer, before the rows and before
    the model, because after any of those the check would be describing
    something this process had already committed to.
    """
    experiment_dir = Path(experiment_dir)
    if not isinstance(plan_digest_expected, str) or not plan_digest_expected:
        raise SystemExit(
            "this child was given no plan digest to check against. It will "
            "not load a tokenizer, a dataset or a model on trust.")
    session_path = experiment_dir / "session.json"
    plan_path = experiment_dir / "plan.json"
    for path in (session_path, plan_path):
        if not path.exists():
            raise SystemExit(f"{path} does not exist, so this child cannot "
                             "check the source it was asked to run")
    try:
        session = json.loads(session_path.read_text())
        plan_doc = json.loads(plan_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"this child cannot read its own session: {exc}") from exc

    problems: list[str] = []
    if plan_digest(plan_doc) != plan_doc.get("plan_digest"):
        problems.append("plan.json does not hash to the digest recorded in it")
    if plan_doc.get("plan_digest") != plan_digest_expected:
        problems.append("plan.json's digest is not the one this child was told "
                        "to run")
    if session.get("plan_digest") != plan_digest_expected:
        problems.append("session.json and this child disagree on the plan "
                        "digest")
    if plan_doc.get("experiment_id") != experiment_dir.name:
        problems.append("plan.json names another experiment than its directory")
    manifest = session.get("source_manifest") or {}
    if manifest_digest(manifest) != session.get("source_manifest_digest"):
        problems.append("the source manifest does not match the digest stored "
                        "with it")
    problems += verify_sources(ROOT, manifest,
                               experiment_dir / "source_snapshot")
    if problems:
        raise SystemExit(
            "this child refuses to start:\n  - " + "\n  - ".join(problems)
            + "\nNothing has been loaded. The plan and the source it was made "
              "against must both still be what they were.")
    return {"files_verified": len(manifest.get("files") or {}),
            "source_manifest_digest": session.get("source_manifest_digest"),
            "plan_digest": plan_doc["plan_digest"],
            "verified_at": now_iso()}


def _publish_child_failure(experiment_dir, run: str, *, stage: str,
                           exc: BaseException) -> None:
    """Write the failure evidence, and never let doing so mask the failure."""
    try:
        experiment_dir = Path(experiment_dir)
        write_failure_evidence(
            {"dir": experiment_dir}, run,
            experiment_id=experiment_dir.name, stage=stage,
            exception_type=type(exc).__name__, summary=str(exc))
    except Exception:  # noqa: BLE001 - the original exception is the news
        pass


def run_child(*, experiment_dir: Path, run: str, rows: int, nonce: str,
              plan_digest_value: str | None = None, deps: ChildDeps | None = None,
              plan_digest: str | None = None, preflight=None) -> int:
    """The measured child: one prefix of the frozen order, measured row by row.

    Unreachable from the CLI, which refuses ``--child`` alongside the session
    flags. It is written in full, and tested through this same function with
    injected dependencies, so that approval does not also require somebody to
    write the part that actually measures.

    Order of business: verify the plan and the frozen source *before* the
    tokenizer, the rows or the model -- see :func:`child_verify_source` --
    then, per row, time the step, clear the cache on every tenth row, record
    analysis data and a progress line every fifth, and check for a stop
    request at the row boundary; never inside the timed region, and never
    inside a signal handler.
    """
    import signal
    import time

    from src.training.watchdog import StopFlag, read_stop_request

    # Pinned here as well as by the parent that spawns this process: a child
    # started by hand for debugging must load exactly what a measured child
    # loads, and neither may depend on the shell it inherited.
    enforce_offline_environment()

    experiment_dir = Path(experiment_dir)
    flag = StopFlag()
    try:
        signal.signal(signal.SIGTERM, flag.handler)
    except ValueError:
        pass  # not the main thread: tests may call this directly

    # Bound once the deps have loaded; before that there is nothing to drop.
    teardown_fn = None
    torn_down = False

    def release() -> float:
        nonlocal torn_down
        if torn_down or teardown_fn is None:
            return 0.0
        torn_down = True
        return teardown_fn()

    # Every stage below is wrapped the same way: publish why, release the
    # device, re-raise the original. A child that dies leaves no report, and
    # used to leave no reason either -- exp001's b1 exited 1 and the only
    # account of why lived in a terminal scrollback. Round 18 wrapped the
    # model load alone, which left a failed row, a failed clear, a failed
    # teardown and a failed write with exactly the same silence.
    def die(stage: str, exc: BaseException) -> None:
        """Publish first, then clean up. Neither may mask the other.

        Publishing comes first on purpose: ``write_once_json`` refuses a
        second write, so if the best-effort release fails too, the reason
        already on disk is the one that ended the run rather than the one
        cleanup ran into.
        """
        _publish_child_failure(experiment_dir, run, stage=stage, exc=exc)
        try:
            release()
        except BaseException:  # noqa: BLE001 - the original is the news
            pass

    # Before the tokenizer, before the rows, before the model.
    try:
        source_check = child_verify_source(experiment_dir,
                                           plan_digest_value or plan_digest)
    except BaseException as exc:  # noqa: BLE001 - re-raised below
        die("source_check", exc)
        raise
    digest = source_check["plan_digest"]

    # Deliberately outside the published stages: this is a caller mistake, not
    # a run that died, and it must not leave a file claiming a run failed.
    report_path = experiment_dir / f"{run}.json"
    if report_path.exists():
        raise FileExistsError(f"{report_path} already exists; a measured run "
                              "is never written twice")
    progress_path = experiment_dir / f"{run}.progress.jsonl"

    # The four gated readings, taken by the child itself. The parent's gate
    # record says what the machine looked like before this process existed;
    # this says what it looked like once it did.
    if preflight is None:
        from src.training.preflight import preflight_sample
        preflight = preflight_sample
    try:
        pre = preflight()
    except BaseException as exc:  # noqa: BLE001 - re-raised below
        die("preflight", exc)
        raise
    started_at = now_iso()

    # ``--child`` injects nothing, so this line is the one that decides what
    # gets measured. Round five refactored the production deps and left it
    # out; every test passed because every test brought its own.
    deps = deps or ProductionChildDeps()

    process_t0 = time.perf_counter()
    try:
        loaded = deps.load(rows=rows)
    except BaseException as exc:  # noqa: BLE001 - re-raised below
        die("model_load", exc)
        raise
    # From here on there is a model on the device, so every failure path has
    # to drop it. ``release`` is idempotent because the failure paths and the
    # measured teardown can both reach it, and a second ``empty_cache`` would
    # be measured as if it were the first.
    #
    # Bound before the contract is checked, and only if it is real: the whole
    # point of checking here is that the model already exists, so a loader
    # that broke its contract still has to be unwound. A ``teardown`` that is
    # itself the broken field is not a teardown and must not be called.
    if isinstance(loaded, dict) and callable(loaded.get("teardown")):
        teardown_fn = loaded["teardown"]
    contract = child_load_problems(loaded, rows=rows)
    if contract:
        broken = ValueError("the loader did not honour the child contract:"
                            "\n  - " + "\n  - ".join(contract))
        die("dependency", broken)
        raise broken
    order = loaded["order"]
    step = loaded["step"]
    clear = loaded.get("clear") or (lambda: 0.0)
    probe = loaded.get("probe") or (lambda: {})

    per_row, memory, per_call = [], [], []
    stopped, tool_failure = None, None
    condition_t0 = time.perf_counter()

    try:
        for position, index in enumerate(order, start=1):
            row_t0 = time.perf_counter()
            result = step(index, position)
            compute = time.perf_counter() - row_t0

            cleared = position % EMPTY_CACHE_EVERY == 0
            clear_seconds = None
            if cleared:
                clear_seconds = clear()
                per_call.append({"row": position, "seconds": clear_seconds})
            probe_seconds = 0.0
            if position == 1 or position % MEMORY_EVERY == 0:
                probe_t0 = time.perf_counter()
                sample = probe()
                probe_seconds = time.perf_counter() - probe_t0
                memory.append({"row": position, **sample,
                               "probe_seconds": probe_seconds,
                               "monotonic": time.perf_counter() - condition_t0})
            end_to_end = time.perf_counter() - row_t0

            per_row.append({"row": position, "compute_seconds": compute,
                            "end_to_end_seconds": end_to_end,
                            "monotonic": time.perf_counter() - condition_t0,
                            "wall_clock": now_iso(),
                            "sample_id": result["sample_id"],
                            "tokens": result["tokens"],
                            "supervised_tokens": result["supervised_tokens"],
                            "loss": result["loss"], "cleared": cleared,
                            "clear_seconds": clear_seconds})
            if position == 1 or position % MEMORY_EVERY == 0:
                write_progress(progress_path, row=position,
                               row_elapsed_seconds=end_to_end,
                               condition_clock_seconds=time.perf_counter() - condition_t0,
                               process_clock_seconds=time.perf_counter() - process_t0)

            # The row boundary is the only place a stop is honoured, and the only
            # place the request is read: a signal handler may not touch a file.
            if flag.is_set() or (getattr(deps, "stop_at_row", None) == position):
                got = read_stop_request(experiment_dir, prefix=run, nonce=nonce)
                if got["accepted"]:
                    stopped = {"reason": got["body"]["reason"],
                               "rule": got["body"].get("rule"), "row": position,
                               "sampled_values": got["body"].get("sampled_values"),
                               # Both clocks as they stood at the row boundary
                               # where the stop was honoured, so replay can place
                               # the stop inside the run rather than take its word
                               # that it happened (§4.7).
                               "condition_clock_seconds":
                                   time.perf_counter() - condition_t0,
                               "process_clock_seconds":
                                   time.perf_counter() - process_t0,
                               "requested_by": "watchdog",
                               "stop_request_sha256": got["sha256"]}
                    break
                tool_failure = got["reason"]
                break
    except BaseException as exc:  # noqa: BLE001 - re-raised below
        die("training", exc)
        raise

    condition_seconds = time.perf_counter() - condition_t0
    teardown_t0 = time.perf_counter()
    try:
        release()
    except BaseException as exc:  # noqa: BLE001 - re-raised below
        die("teardown", exc)
        raise
    teardown_seconds = time.perf_counter() - teardown_t0
    process_seconds = time.perf_counter() - process_t0

    try:
        metrics = compute_metrics(per_row, stop_reason=(stopped or {}).get("reason"))
        contract_total = sum(c["seconds"] for c in per_call)
        probe_total = sum(m["probe_seconds"] for m in memory)
        end_to_end_total = sum(r["end_to_end_seconds"] for r in per_row)
        compute_total = sum(r["compute_seconds"] for r in per_row)

        report = {
            "schema_version": CHILD_SCHEMA_VERSION, "kind": CHILD_KIND,
            "experiment_id": experiment_dir.name,
            "run_id": run, "declared_rows": rows, "condition": CONDITION,
            "plan_digest": digest, "nonce": nonce,
            "pool_pairs": POOL_PAIRS, "pool_rows": POOL_ROWS,
            "child_source_check": source_check, "preflight": pre,
            # The child's own account of which process this was. The parent wrote
            # the same four fields into the launch record before the first row;
            # replay compares the two, because a record nobody compares with the
            # process it describes could name anything at all (§4.8 7b).
            "child_pid": os.getpid(), "child_pgid": os.getpgrp(),
            "child_start_identity": _own_start_identity(),
            "started_at": started_at, "finished_at": now_iso(),
            "rows_requested": rows, "rows_completed": len(per_row),
            "stopped_early": stopped, "tool_failure": tool_failure,
            "input_order_digest": _digest_ids(loaded["sample_ids"]),
            "completed_input_digest": _digest_ids(r["sample_id"] for r in per_row),
            "provenance": loaded["provenance"],
            "per_row": per_row, "memory": memory,
            "metrics": {k: metrics.get(k) for k in
                        ("D100", "D100_reason", "D20", "D20_reason", "Dmax",
                         "Dmax_reason")},
            "scheduled_empty_cache_every": EMPTY_CACHE_EVERY,
            "scheduled_empty_cache_cost": {
                "calls": len(per_call), "total_seconds": contract_total,
                "mean_seconds": (contract_total / len(per_call)) if per_call else None,
                "max_seconds": max((c["seconds"] for c in per_call), default=None),
                "per_call": per_call},
            "teardown_empty_cache_calls": 1,
            "teardown_empty_cache_seconds": teardown_seconds,
            "end_to_end_seconds": end_to_end_total,
            "model_compute_seconds": compute_total,
            "between_row_overhead_breakdown": {
                "scheduled_empty_cache_seconds": contract_total,
                "memory_probe_seconds": probe_total,
                "unattributed_seconds": max(
                    0.0, end_to_end_total - compute_total - contract_total
                    - probe_total)},
            "clocks": {"model_load_seconds": loaded["model_load_seconds"],
                       "condition_clock_seconds": condition_seconds,
                       "process_clock_seconds": process_seconds},
            "float_storage": {"seconds_rounded": False, "loss_rounded": False},
        }
        write_once_json(report_path, report)
    except BaseException as exc:  # noqa: BLE001 - re-raised below
        die("report_write", exc)
        raise
    return 0 if tool_failure is None else 4
