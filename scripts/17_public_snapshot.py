#!/usr/bin/env python3
"""The public/private boundary, written down once and used by both sides.

The public repository is built from an **allowlist**, never by mirroring this
working tree and never by filtering it afterwards. A denylist alone fails open:
a new file nobody thought about is published by default, and the first time
anybody notices is after it is already fetchable by commit SHA and by
``refs/pull/*``.

So every path gets one of three verdicts, and ``exclude`` is what an unlisted
path receives:

``include``
    Aggregate results, source, tests and documentation. Nothing per-record,
    nothing identifying.
``exclude``
    Raw and processed data, the frozen split manifest, per-run session
    evidence, per-record report JSON, weights, caches, locks.
``hold``
    On the allowlist by kind, but carrying identifiers that need a decision
    before publication. A held file is **not** copied by ``--build``: the
    default is to publish less, and to say exactly what is being held and
    why, rather than to publish and hope.

This module writes nothing outside the destination given to ``--build``, and
never touches the private tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Denials, checked first. A path matching any of these is excluded no matter
# what the allowlist says.
# ---------------------------------------------------------------------------

PRIVATE_DENY: tuple[tuple[str, str], ...] = (
    (".git/**", "version control internals"),
    (".venv/**", "local virtualenv"),
    ("**/__pycache__/**", "byte-compiled cache"),
    ("**/*.pyc", "byte-compiled cache"),
    ("**/.DS_Store", "desktop metadata"),
    ("**/.lock", "per-machine session lock"),
    ("**/.*.lock", "per-machine session lock"),
    (".env", "credentials"),
    (".env.*", "credentials"),
    (".hf_cache/**", "hugging face cache"),
    (".hf_home/**", "hugging face cache"),

    ("data/raw/**", "raw dataset"),
    ("data/processed/**", "processed per-record dataset"),
    ("data/splits/**",
     "frozen split manifest: 28,259 object hashes and 47,389 structure UUIDs"),

    ("data/reports/15_mps_order/exp001/**", "per-run session evidence"),
    ("data/reports/15_mps_order/exp002/**", "per-run session evidence"),
    ("data/reports/16_longrun/**", "per-run session evidence"),

    ("data/reports/05_d_arm.json", "24 per-generation records with raw output"),
    ("data/reports/10_instruction.json", "removed_sample_ids: 16 dataset ids"),
    ("data/reports/11_instruction_audit.json", "per-sample id keys"),
    ("data/reports/12_f_oracle.json", "1,600 per-task records, 200 UUIDs"),
    ("data/reports/13_lora_coldstart.json", "checkpoint paths and load order"),
    ("data/reports/13_lora_sanity.json", "64 per-module names"),
    ("data/reports/13_lora_smoke.json", "per-record loss history and ids"),
    ("data/reports/14_mps_speed.json", "200 per-row records"),

    ("artifacts/checkpoints/**", "model weights"),
    ("artifacts/renders/**", "local renders"),
    ("**/*.safetensors", "model weights"),
    ("**/*.ckpt", "model weights"),
    ("**/*.pt", "model weights"),
    ("**/*.pth", "model weights"),
    ("**/*.bin", "model weights"),
    ("**/*.gguf", "model weights"),
    ("**/*.h5", "model weights"),
    ("**/*.msgpack", "model weights"),

    ("CLAUDE.md", "internal collaborator guide; not part of the public surface"),

    # The release gate, not part of the released thing. It spawns full-suite
    # subprocesses and asserts against a tree that *has* the private
    # evidence, so inside the snapshot it can only fail -- and shipping a
    # test that needs `--ignore` to stay green teaches every reader that
    # ignoring a failing test here is normal. It runs in the private tree,
    # before a build, which is where a gate belongs.
    ("tests/test_public_snapshot.py",
     "the private release gate; it verifies this boundary rather than "
     "being part of what the boundary publishes"),
)


def _decided_denials() -> tuple[tuple[str, str], ...]:
    """The round 23 decisions, expressed as ordinary denials."""
    return tuple((path, f"withheld: {why}") for path, why in
                 DECIDED_AGAINST.items())


# ---------------------------------------------------------------------------
# What may be published. Order does not matter: any match is enough.
# ---------------------------------------------------------------------------

PUBLIC_ALLOW: tuple[str, ...] = (
    ".gitignore",
    "requirements.txt",
    "README.md",
    "LICENSE",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "THIRD_PARTY_NOTICES.md",
    "BRICKAGAIN_PROJECT_WORKFLOW.md",

    "src/**/*.py",
    "scripts/*.py",
    "tests/*.py",

    "artifacts/ldraw/**/*.ldr",

    # Markdown is listed one file at a time. `data/reports/*.md` published a
    # file the moment it was written, which is the opposite of what an
    # allowlist is for: three of the reports it swept in carry dataset object
    # hashes, a sample id, or a boot fingerprint.
    "data/reports/02_retile.md",
    "data/reports/04_counterfactual.md",
    "data/reports/06_mps_multinomial.md",
    "data/reports/07_audit.md",
    "data/reports/08_corpus_structure.md",
    "data/reports/09_stagger_ablation.md",
    "data/reports/10_instruction.md",
    "data/reports/11_instruction_audit.md",
    "data/reports/12_f_oracle.md",

    "data/reports/01_eda.json",
    "data/reports/02_retile.json",
    "data/reports/04_counterfactual.json",
    "data/reports/06_mps_multinomial.json",
    "data/reports/07_audit.json",
    "data/reports/08_corpus_structure.json",
    "data/reports/09_stagger_ablation.json",
    "data/reports/16_longrun_calibration.json",
    "data/reports/16_longrun_design.json",
    "data/reports/16_watchdog_microbenchmark.json",
    "data/reports/15_mps_order/calibration.json",
)


# ---------------------------------------------------------------------------
# Decided. Round 23 resolved all five holds as *exclude*: no boot fingerprint,
# no dataset identifier and no raw process id is published in this version.
# The mechanism stays, because the next document that needs a decision must
# not be publishable by default -- and ``build`` refuses while anything is
# still held.
# ---------------------------------------------------------------------------

HOLD: dict[str, str] = {}

#: Withheld with the evidence that decided it. These are ordinary denials
#: now; the record is here so the decision is not re-litigated from scratch.
DECIDED_AGAINST: dict[str, str] = {
    "PROJECT_STATUS.md":
        "one raw child process id in the exp001 timeline, and a running "
        "account of an unpublished private tree",
    "PUBLIC_RELEASE_CHECKLIST.md":
        "describes the cleanup of the public repository itself, including "
        "what is already exposed and where it remains reachable",
    "data/reports/01_eda.md":
        "4 raw dataset object hashes",
    "data/reports/05_d_arm.md":
        "per-generation narrative for 24 runs",
    "data/reports/13_lora_smoke.md":
        "1 dataset sample id in 4 headings, with the generated output",
    "data/reports/14_mps_speed.md":
        "per-row narrative of the 200-row record that is itself withheld",
    "data/reports/15_mps_order.md":
        "cites digests of exp001 evidence that is not published",
    "data/reports/15_mps_order_exp002.md":
        "4 boot fingerprints in the per-run table",
    "data/reports/16_longrun_design.md":
        "1 boot fingerprint and one raw child process id on the 7.12 timeline",
}


def _to_regex(pattern: str) -> re.Pattern:
    """Real glob semantics, because ``fnmatch`` does not have them.

    ``fnmatch`` turns every ``*`` into ``.*``, which matches a path separator.
    So ``src/**/*.py`` silently required a subdirectory and skipped
    ``src/model_ids.py`` -- a module every arm imports. An allowlist that
    quietly drops files is worse than no allowlist: the build looks clean and
    the result does not run.
    """
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append(r"(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(r".*")
            i += 2
        elif pattern[i] == "*":
            out.append(r"[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append(r"[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile(r"\A" + "".join(out) + r"\Z")


_COMPILED: dict[str, re.Pattern] = {}


def _matches(rel: str, pattern: str) -> bool:
    rx = _COMPILED.get(pattern)
    if rx is None:
        rx = _COMPILED[pattern] = _to_regex(pattern)
    return bool(rx.match(rel))


def classify(rel: str) -> tuple[str, str]:
    """``(verdict, reason)`` for one repository-relative path."""
    rel = str(rel)
    for pattern, why in PRIVATE_DENY + _decided_denials():
        if _matches(rel, pattern):
            return "exclude", f"denied by {pattern!r}: {why}"
    if rel in HOLD:
        return "hold", HOLD[rel]
    for pattern in PUBLIC_ALLOW:
        if _matches(rel, pattern):
            return "include", f"allowed by {pattern!r}"
    return "exclude", "not on the allowlist"


def manifest(root: Path | None = None) -> dict[str, list]:
    """Every file in the tree, sorted into the three verdicts."""
    root = Path(root or ROOT)
    out: dict[str, list] = {"include": [], "hold": [], "exclude": []}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        if rel.split("/")[0] in (".git", ".venv"):
            continue
        verdict, reason = classify(rel)
        out[verdict].append({"path": rel, "reason": reason,
                             "bytes": path.stat().st_size})
    return out


# ---------------------------------------------------------------------------
# Scans, run against whatever is about to be published.
# ---------------------------------------------------------------------------

#: Every scan hit that has been reviewed, keyed by kind and by a digest of
#: the **whole, untruncated** matched text, with the number of times it
#: occurs. Reviewed one at a time: a per-file or per-kind allowance would
#: let a new token into an already-approved file without anything changing,
#: and hashing a truncated match would let two different long secrets share
#: one identity. Adding a secret adds a key, repeating one changes a count,
#: and removing one leaves a stale approval -- all three fail the audit.
#: `--scan` shows the real text.
APPROVED_HITS: dict[str, dict[str, int]] = {
    # the report's own render timestamp, already public. Carries a +08:00 offset
    # and nothing else, and the re-render refusal is stated against it.
    'data/reports/12_f_oracle.md': {
        'precise-timestamp|eb04900cc865a488': 1,
    },
    # the calibration's created_at, in UTC. A threshold with no record of when it
    # was taken is a number.
    'data/reports/15_mps_order/calibration.json': {
        'precise-timestamp|541c2e94a5f963e9': 1,
    },
    # carried over verbatim from the archived calibration above.
    'data/reports/16_longrun_calibration.json': {
        'precise-timestamp|541c2e94a5f963e9': 1,
    },
    # this module's own scan patterns and the prefixes they name. Each path here is
    # a bare prefix inside a regex, not a location on anyone's disk.
    'scripts/17_public_snapshot.py': {
        'credential|5bb72c4d662356cd': 1,
        'personal-path|0fe68e7a8a473efa': 1,
        'personal-path|172e42f7f6aa8c93': 1,
        'personal-path|2d47a09e17242cbc': 1,
        'personal-path|3b122b9d4638c4dd': 1,
        'personal-path|407bb1063433b62d': 1,
        'personal-path|4491e4c3e7fe9be6': 1,
        'personal-path|46dc943e3424a225': 2,
        'personal-path|9b7818f86bf06aee': 1,
        'personal-path|a9e5ed9db2e44a54': 1,
        'personal-path|ac351c7174c8f47a': 1,
        'personal-path|bbc72803f9370979': 1,
    },
    # the hub download helper's function name; the provider-prefix rule matches it
    # and there is no key.
    'src/generation/brickgpt.py': {
        'credential|7dac63c8688451a0': 2,
    },
    # comments and regex sources describing what the redactor covers.
    'src/training/longrun.py': {
        'credential-assignment|4c0e654601aa6dd5': 1,
        'credential|483d236ef0a5d02c': 1,
        'credential|5bb72c4d662356cd': 1,
        'personal-path|3305df425d859fc4': 1,
    },
    # an ordinary English phrase in a docstring; the authorization-scheme rule
    # matches the word and the two that follow it.
    'tests/test_diagnostics.py': {
        'bearer|1276be3ad2325127': 1,
    },
    # synthetic fixtures only: fixed AAAABBBB/0000 literals, RFC 2606 and RFC 6761
    # reserved domains, invented process ids, invented timestamps, and assertions
    # that a marker is absent from redacted output. The paths are invented ones
    # under a fictional `someone`, or bare prefixes inside an assertion.
    'tests/test_longrun.py': {
        'bearer|770e4260ec4f945c': 1,
        'bearer|9df27a4624bf06ed': 2,
        'bearer|d430213632188743': 1,
        'bearer|f5b05a1067ad38cc': 1,
        'credential-assignment|0a8a804750a0e07c': 2,
        'credential-assignment|1481752a00b673ae': 1,
        'credential-assignment|19073fcd066b055f': 1,
        'credential-assignment|194710d49d7a0adf': 3,
        'credential-assignment|2543b9ce0ababe11': 1,
        'credential-assignment|39944ece80918fb4': 1,
        'credential-assignment|427bd14b83633eb7': 1,
        'credential-assignment|734f6a4d4cc56a6c': 2,
        'credential-assignment|877bd9039096fd89': 2,
        'credential-assignment|8da8aca683087aea': 1,
        'credential-assignment|a33cb97bf5c9e84a': 1,
        'credential-assignment|b42a14cc55082543': 1,
        'credential-assignment|bcc60e3147f22019': 1,
        'credential-assignment|bf4a3c89c34f135b': 1,
        'credential-assignment|da71e2bd5bdff40b': 1,
        'credential-assignment|f252500c561d4756': 1,
        'credential-assignment|ffe6cd14bf6682d4': 1,
        'credential|537f4d24258f0b78': 1,
        'credential|66796f09af4f00e1': 1,
        'credential|70c61658d5ad2636': 1,
        'credential|aecf65aefd672173': 3,
        'credential|c0011fa40fb5b2a8': 1,
        'credential|d361a2a1399d8c0b': 4,
        'credential|e9f610bf9e97b500': 1,
        'credential|f2affa795645b203': 1,
        'email|72497f475e4f76d0': 4,
        'email|d2f1927a621c536b': 3,
        'organization-id|2d7d0a06885ef0e4': 2,
        'organization-id|2f75218a0e380d39': 2,
        'personal-path|0fe68e7a8a473efa': 1,
        'personal-path|172e42f7f6aa8c93': 2,
        'personal-path|190b1e33aa09e964': 1,
        'personal-path|298f55db1d1b1f23': 1,
        'personal-path|2d47a09e17242cbc': 1,
        'personal-path|3b122b9d4638c4dd': 4,
        'personal-path|411a8d1af1f5c783': 1,
        'personal-path|4491e4c3e7fe9be6': 2,
        'personal-path|46dc943e3424a225': 4,
        'personal-path|5c2406d3c1e5548a': 1,
        'personal-path|60dabd422c5ee707': 1,
        'personal-path|7e76815d9b778c02': 1,
        'personal-path|7ff21fb52c7b97dc': 1,
        'personal-path|8208125467e3580a': 2,
        'personal-path|853b60c84c714116': 1,
        'personal-path|9b7818f86bf06aee': 1,
        'personal-path|a9e5ed9db2e44a54': 1,
        'personal-path|e44992995b0ce9f3': 1,
        'personal-path|e8749bf845243699': 1,
        'personal-path|fdf6f40dda050cc3': 1,
        'precise-timestamp|08625871dfe05606': 1,
        'precise-timestamp|6161affe69650269': 3,
        'precise-timestamp|79da750b6b924d53': 1,
        'precise-timestamp|8f9b85cb26b01958': 4,
        'precise-timestamp|c55d85bc617e7558': 1,
        'raw-pid|003b52b7c84da2c5': 1,
        'raw-pid|444a76db3fc92c4d': 1,
        'raw-pid|50c1f918086dbd35': 3,
        'raw-pid|5beb68d9066e683a': 1,
        'raw-pid|8898363e87d76bb7': 3,
        'raw-pid|8e9b834aeae50be8': 1,
        'raw-pid|cd4aa047ffe18819': 1,
    },
    # one assertion that a home-directory prefix is absent from a written report.
    'tests/test_lora.py': {
        'personal-path|46dc943e3424a225': 1,
    },
    # invented timestamps in gate and journal fixtures.
    'tests/test_mps_order.py': {
        'precise-timestamp|04e3d25947edc224': 3,
        'precise-timestamp|3a0c4546edba4087': 1,
        'precise-timestamp|a9fee6255ba06578': 1,
    },
}


SCANS: tuple[tuple[str, str], ...] = (
    # Provider-issued keys. Every prefix that has been seen in the wild, and
    # bodies that may carry hyphens and underscores -- stopping at the first
    # one is what let `sk-proj-...` through with four characters redacted.
    ("credential",
     r"\b(?:hf_|sk-|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|glpat-|npm_"
     r"|dop_v1_|xox[abprs]-|AKIA|ASIA|ya29\.|AIza)[A-Za-z0-9_.-]{6,}"),

    # `key = value`, with a separator and a non-empty value. Naming a
    # credential is not disclosing one, so a bare word does not match.
    ("credential-assignment",
     r"(?i)\b(?:api[-_ ]?key|access[-_ ]?token|refresh[-_ ]?token"
     r"|id[-_ ]?token|client[-_ ]?secret|client[-_ ]?id|private[-_ ]?key"
     r"|secret[-_ ]?key|auth[-_ ]?token|session[-_ ]?token|password|passwd"
     r"|secret|token|authorization|credentials?)\b\s*[:=]\s*\S+"),

    # Matches the scheme and the value after it. The word alone is not
    # enough, and ordinary prose does occasionally trip this; those hits
    # are approved individually rather than by loosening the rule.
    ("bearer", r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{4,}"),
    ("email", r"\b[A-Za-z0-9._+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+\b"),
    ("organization-id", r"\borg-[A-Za-z0-9_-]{4,}"),
    ("private-key-block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),

    # Absolute paths belonging to a person, a temp area or a removable
    # volume, on either platform.
    # The prefix *and* everything after it. Matching only the prefix meant
    # `/Users/` and `/Users/someone/private.txt` were the same finding, so
    # approving a mention of the prefix approved every real path under it.
    ("personal-path",
     r"(?:[A-Za-z]:\\Users\\|/Users/|/home/|/root/|/private/|/Volumes/"
     r"|/var/folders/|/mnt/|/media/|/srv/|/tmp/)"
     r"[^\s\"'`<>|,)\]}]*"),

    # Machine and process identity.
    ("raw-pid", r"(?i)\b(?:PID|child_pid|child_pgid|pgid|ppid)\b\s*[:=]?\s*\d{2,}"),
    ("boot-fingerprint",
     r"(?i)(?:boot[_ -]?fingerprint|fingerprint)\W{0,4}[0-9a-f]{32}\b"),
    ("hex32", r"\b[0-9a-f]{32}\b"),
    ("dataset-uuid",
     r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),

    # A wall-clock instant precise to the second with an offset says when a
    # particular machine did something.
    ("precise-timestamp",
     r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})"),
)


#: The audit's unit of approval: not ``(kind, path)`` -- which would let a new
#: token into an already-approved file unnoticed -- but the exact matched text
#: and how many times it occurs. Adding a secret adds a key; repeating one
#: changes a count; either fails.
def hit_key(kind: str, match: str) -> str:
    """The audit's key: the kind, and a digest of the exact matched text.

    A digest rather than the text itself, for one practical reason: this file
    is published, and a table of literal secret-shaped strings would be a
    table of new scan hits. Hashing keeps the approval exact -- a different
    token is a different key -- without writing the tokens down again. The
    text is still readable through ``--scan``, which reads the files.
    """
    return f"{kind}|{hashlib.sha256(match.encode()).hexdigest()[:16]}"


def hit_counter(paths, root=None) -> dict[str, dict[str, int]]:
    """``{path: {key: count}}`` for everything the scans find."""
    out: dict[str, dict[str, int]] = {}
    for hit in scan(paths, root=root):
        key = hit_key(hit["kind"], hit["match"])
        out.setdefault(hit["path"], {})
        out[hit["path"]][key] = out[hit["path"]].get(key, 0) + 1
    return out


def audit(paths, root=None) -> list[str]:
    """Every scan hit that is not approved, exactly as it appears.

    Run before a build copies anything. An unapproved hit is not a warning:
    the build refuses, because a file that leaks is not improved by being
    published alongside a note about it.
    """
    problems = []
    found = hit_counter(paths, root=root)
    seen = {hit_key(h["kind"], h["match"]): h for h in scan(paths, root=root)}

    def describe(key: str) -> str:
        hit = seen.get(key)
        return f"{key} ({hit['kind']} {hit['display']!r} at line {hit['line']})" \
            if hit else key

    # Only the paths under audit. Judging the whole approval table against a
    # subset would report every other file as stale.
    under_audit = {str(rel) for rel in paths}
    for path in sorted((set(found) | set(APPROVED_HITS)) & under_audit):
        got = found.get(path, {})
        want = APPROVED_HITS.get(path, {})
        for key in sorted(set(got) | set(want)):
            g, w = got.get(key, 0), want.get(key, 0)
            if g == w:
                continue
            if w == 0:
                problems.append(f"{path}: unapproved {describe(key)} x{g}")
            elif g == 0:
                problems.append(
                    f"{path}: approved {key!r} x{w} no longer occurs; the "
                    "approval is stale and must be removed")
            else:
                problems.append(
                    f"{path}: {key!r} occurs {g} times, {w} approved")
    return problems


def scan(paths, root: Path | None = None) -> list[dict]:
    """Every scan hit, with enough context to classify it one by one.

    ``match`` is the exact, untruncated text -- the approval's identity.
    ``display`` is a shortened form for printing, and is never hashed.
    """
    root = Path(root or ROOT)
    hits = []
    for rel in paths:
        p = root / rel
        try:
            text = p.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for name, pattern in SCANS:
            rx = re.compile(pattern)
            for i, line in enumerate(text.splitlines(), 1):
                # Every match on the line, not the first. Two secrets on one
                # line used to count as one, so hiding the second cost
                # nothing.
                for m in rx.finditer(line):
                    # `match` is the whole thing, always: it is what the
                    # approval is keyed on. Truncating it first made every
                    # secret sharing an 80-character head one approval, so
                    # approving one approved all of them. `display` is the
                    # abbreviated form, for reading only.
                    found = m.group(0)
                    hits.append({"kind": name, "path": rel, "line": i,
                                 "match": found,
                                 "display": (found[:77] + "..."
                                             if len(found) > 80 else found),
                                 "context": line.strip()[:120]})
    return hits


class BuildRefused(RuntimeError):
    """The build stopped before copying anything."""


def destination_problems(dest, root=None) -> list[str]:
    """Is this a safe place to write a public snapshot into?

    Checked here rather than in ``main`` so that calling ``build()`` directly
    -- which is what the tests and any future tooling do -- gets the same
    answer as the command line. A guard that only exists in the CLI is a
    guard that the next caller does not have.
    """
    root = Path(root or ROOT).resolve()
    given = Path(dest).expanduser().absolute()
    real = given.resolve()
    problems = []

    # A symlink anywhere in the destination path means the bytes land
    # somewhere other than the name suggests -- including, possibly, inside
    # the private tree.
    if str(given) != str(real):
        problems.append(
            f"{given} resolves to {real}: the destination must not reach its "
            "location through a symbolic link")

    if real == root:
        problems.append("the destination is the source tree itself")
    elif root in real.parents:
        problems.append(f"{real} is inside the source tree")
    elif real in root.parents:
        problems.append(
            f"{real} contains the source tree; building here would put a "
            "public snapshot around the private one")

    if real.exists():
        if not real.is_dir():
            problems.append(f"{real} exists and is not a directory")
        else:
            existing = sorted(x.name for x in real.iterdir())
            if existing:
                problems.append(
                    f"{real} is not empty ({len(existing)} entries, first: "
                    f"{existing[0]!r}). A public snapshot goes into a "
                    "dedicated empty directory: anything already there would "
                    "survive the copy and be published with it")
    return problems


def source_problems(paths, root=None) -> list[str]:
    """Nothing on the allowlist may be a symbolic link.

    A link is copied by ``copy2`` as its target's bytes, so an allowlisted
    name could stand in for any file on the machine -- including one the
    denylist covers.
    """
    root = Path(root or ROOT)
    problems = []
    for rel in paths:
        src = root / rel
        if src.is_symlink():
            problems.append(
                f"{rel} is a symbolic link to {os.readlink(src)!r}; an "
                "allowlisted path must be a real file")
    return problems


def build(dest, root=None) -> list[str]:
    """Copy every ``include`` path into ``dest``, or refuse and copy nothing.

    Four things are checked first, and all four refuse rather than warn:
    nothing may still be held, the destination has to be a dedicated empty
    place that is not the source tree or an ancestor of it, no allowlisted
    path may be a symbolic link, and every scan hit in what is about to be
    copied has to be individually approved. There is no way to switch any of
    them off, and adding one would be the whole failure.
    """
    root = Path(root or ROOT)
    m = manifest(root)
    included = [e["path"] for e in m["include"]]

    problems = []
    if m["hold"]:
        problems += [f"{e['path']} is still held: {e['reason']}"
                     for e in m["hold"]]
    problems += destination_problems(dest, root)
    problems += source_problems(included, root)
    # Unconditional. A build that can be told not to audit is a build that
    # eventually is, and the one time it matters is the time somebody was in
    # a hurry.
    problems += audit(included, root)
    if problems:
        raise BuildRefused("refusing to build:\n  - " + "\n  - ".join(problems))

    dest = Path(dest).expanduser().absolute()
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for rel in included:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target, follow_symlinks=False)
        copied.append(rel)
    return copied


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true",
                    help="print the manifest as JSON")
    ap.add_argument("--summary", action="store_true",
                    help="print counts and the held entries")
    ap.add_argument("--scan", action="store_true",
                    help="scan every includable path and print each hit")
    ap.add_argument("--audit", action="store_true",
                    help="report every scan hit that is not approved")
    ap.add_argument("--build", metavar="DEST",
                    help="copy the included paths into DEST (creates nothing "
                         "else, and never writes into the private tree)")
    args = ap.parse_args(argv)

    if args.build:
        try:
            copied = build(Path(args.build))
        except BuildRefused as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"copied {len(copied)} files into "
              f"{Path(args.build).expanduser().absolute()}")
        return 0

    m = manifest()
    if args.list:
        print(json.dumps(m, indent=2, ensure_ascii=False))
        return 0
    if args.scan:
        for hit in scan([e["path"] for e in m["include"]]):
            print(f"{hit['kind']:22s} {hit['path']}:{hit['line']}  "
                  f"{hit['match']}")
        return 0
    if args.audit:
        problems = audit([e["path"] for e in m["include"]])
        for problem in problems:
            print(problem)
        print(f"\n{len(problems)} unapproved hit(s)")
        return 1 if problems else 0

    print(f"include : {len(m['include'])}")
    print(f"hold    : {len(m['hold'])}")
    print(f"exclude : {len(m['exclude'])}")
    for entry in m["hold"]:
        print(f"\nHELD  {entry['path']}\n      {entry['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
