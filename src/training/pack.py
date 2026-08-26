"""The private training pack: what may be handed to the execution node.

``scripts/17_public_snapshot.py`` decides what may be published to the world.
This decides something narrower and different: what may be copied onto the
Taichung Windows machine so it can execute a training run. The node is not the
public, so the two lists are not the same list -- but the failure mode is, and
so is the defence.

**Default deny.** Every path gets a verdict, and an unlisted path is
``exclude``. A denylist alone fails open: the next file nobody thought about
travels by default, and the first time anybody notices is after it is on
another machine. The categories the brief names -- raw and processed data,
per-record reports, session evidence, weights, checkpoints, credentials,
addresses, personal absolute paths and organization ids -- are written down as
denials *as well*, so that the reason a thing did not travel is legible rather
than merely true.

**One definition of an identifier, not two.** The patterns, the key function
and the approval table all come from module 17. This module selects a subset
of its *kinds* -- the four the brief names, plus the shapes that carry them --
and nothing else. A test asserts every selected kind still exists there, so
renaming one cannot leave this side quietly scanning for nothing.

**One definition of a manifest, not two.** The manifest is the shape
``src/training/session.py`` already uses: ``{"files": {rel: {"sha256",
"bytes", "snapshot_name"}}}``. That is deliberate -- it means
:func:`session.manifest_digest` is the pack's digest and
:func:`session.verify_sources` is the pack's file-by-file verifier, rather
than two more functions that mean almost the same thing. The one difference is
``snapshot_name``: report 15 flattens ``a/b.py`` to ``a__b.py`` because it only
ever reads those copies, while a pack has to be *importable*, so it keeps the
tree and records that it did.

**The data does not travel in the pack.** Not one processed or raw byte. What
travels is what the data must hash to, so the node can prove it is training on
the same rows without this module ever moving them. A digest of a dataset is
not a dataset.

This module writes nothing outside the destination given to :func:`build`, and
never writes into the private tree.
"""

from __future__ import annotations

import importlib.util
import json
import os
import posixpath
import re
from pathlib import Path

from src.training.session import (copy_once, manifest_digest, now_iso,
                                  sha256_file, verify_sources,
                                  write_once_json)

ROOT = Path(__file__).resolve().parents[2]

#: The public-boundary module. Loaded by path because ``scripts/`` is not a
#: package; loaded once because its regexes are compiled on first use.
SNAPSHOT_SCRIPT = ROOT / "scripts" / "17_public_snapshot.py"

MANIFEST_NAME = "pack_manifest.json"
SCHEMA_VERSION = 1
KIND = "brickagain.training_pack"

_M17 = None


def snapshot_module():
    """Module 17, imported by path and cached."""
    global _M17
    if _M17 is None:
        spec = importlib.util.spec_from_file_location(
            "brickagain_public_snapshot", SNAPSHOT_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _M17 = mod
    return _M17


# ---------------------------------------------------------------------------
# Denials, checked first. A path matching any of these does not travel, no
# matter what the allowlist says.
# ---------------------------------------------------------------------------

PACK_DENY: tuple[tuple[str, str], ...] = (
    (".git/**", "version control internals"),
    (".venv/**", "a virtualenv built for this machine's architecture"),
    (".pytest_cache/**", "local test cache"),
    ("**/__pycache__/**", "byte-compiled cache"),
    ("**/*.pyc", "byte-compiled cache"),
    ("**/.DS_Store", "desktop metadata"),
    ("**/.lock", "per-machine session lock"),
    ("**/.*.lock", "per-machine session lock"),
    (".env", "credentials"),
    (".env.*", "credentials"),
    (".hf_cache/**", "hugging face cache, including any stored token"),
    (".hf_home/**", "hugging face cache, including any stored token"),
    ("**/.netrc", "credentials"),
    ("**/*.pem", "private key material"),
    ("**/*.key", "private key material"),

    # Data. None of it travels in the pack; the manifest carries its digests.
    ("data/raw/**", "raw dataset"),
    ("data/processed/**", "processed per-record dataset"),
    ("data/splits/**", "the frozen split manifest is a per-object identifier list"),

    # Per-run session evidence and per-record reports.
    ("data/reports/**", "per-run session evidence and per-record reports"),

    # The project model included. ``scripts/25_core_eval.py --run`` loads
    # final_H2 from a path given on the command line and checks the three
    # digests ``runs/project_model.json`` records before it reads a weight;
    # the weights themselves are 13.6MB of trained delta and are not a thing
    # a pack carries.
    ("artifacts/checkpoints/**", "model weights"),
    ("artifacts/renders/**", "local renders"),
    ("**/*.safetensors", "model weights"),
    ("**/*.ckpt", "checkpoint"),
    ("**/*.pt", "checkpoint"),
    ("**/*.pth", "checkpoint"),
    ("**/*.bin", "model weights"),
    ("**/*.gguf", "model weights"),
    ("**/*.h5", "model weights"),
    ("**/*.msgpack", "model weights"),
    ("**/optimizer.pt", "optimizer state belongs to a run, not to a pack"),

    # Documents that carry this machine's own absolute paths, its running
    # account of an unpublished tree, or both.
    ("CLAUDE.md",
     "the collaborator guide names this machine's project directory, which is "
     "a personal absolute path"),
    ("PROJECT_STATUS.md",
     "a running account of the private tree, with a raw child process id"),
    ("PUBLIC_RELEASE_CHECKLIST.md",
     "describes the public repository's own cleanup"),

    ("tests/test_public_snapshot.py",
     "the private release gate; it spawns the full suite against a tree that "
     "has the private evidence, so inside a pack it can only fail"),

    # The full-version tracks. None of them is reachable from anything this
    # pack executes -- ``NON_PACKED_SUBTREES`` below is checked against the
    # actual import closure by ``tests/test_pack.py``, so this is a measured
    # exclusion rather than a hopeful one. They are denied because a training
    # pack should carry the training code and nothing else: ``src/vision``
    # contains the project's only module that opens an outbound connection,
    # and the interface subtree carries an HTTP server. Neither has any
    # business on a node whose whole job is to run a fit from a read-only
    # payload.
    *((f"{subtree}/**", reason) for subtree, reason in (
        ("src/vision",
         "the image track, including src/vision/net.py -- the one module in "
         "this project that makes an outbound HTTP request. Nothing the node "
         "runs imports it"),
        ("src/retrieval",
         "the retrieval track: an embedding model, an index and a search. "
         "Nothing the node runs imports it"),
        ("src/colour", "colour assignment; not reachable from the fit"),
        ("src/assembly", "build ordering; not reachable from the fit"),
        ("src/ui", "the interface, including an HTTP server"),
        ("src/demo", "the demonstration entry point"),
        ("src/delivery", "the delivery pipeline"),
    )),
    ("src/rendering/preview.py",
     "the Matplotlib preview; not reachable from the fit and it would pull an "
     "optional visual stack onto the node"),
)

#: The ``src`` subtrees the pack deliberately does not carry, named once so
#: the denial table above and the test that measures it read the same list.
NON_PACKED_SUBTREES: tuple[str, ...] = (
    "src/vision", "src/retrieval", "src/colour", "src/assembly",
    "src/ui", "src/demo", "src/delivery",
)

#: The entry points whose import closure the payload has to cover. Both the
#: allowlist's meaning and the test that checks it are defined by this list:
#: everything these reach must travel, and the denied subtrees must not be
#: among what they reach.
PACK_ENTRY_POINTS: tuple[str, ...] = (
    "scripts/17_public_snapshot.py",
    "scripts/18_gpu_pack.py",
    "scripts/19_gpu_gate.py",
    "scripts/20_hypothesis_run.py",
    "scripts/22_final_train.py",
    "scripts/25_core_eval.py",
    "tests/test_pack.py",
    "tests/test_gpu_node.py",
    "tests/test_gates.py",
    "tests/test_gate_suite.py",
    "tests/test_arms.py",
    "tests/test_final_run.py",
    "tests/test_core_eval.py",
)


def import_closure(root=None, entry_points=PACK_ENTRY_POINTS) -> set[str]:
    """Every ``src`` module the pack's entry points can reach, by static read.

    Deliberately a static read rather than an import: importing the closure
    would need torch, a dataset and a device, and the question here is which
    files have to travel, which is answerable from the source alone.  A module
    reached only through ``importlib`` would be missed, so the modules that do
    that -- this one, and module 17 -- name their targets in the allowlist by
    hand and are covered by the pack's own build-and-verify.
    """
    import ast

    base = Path(root or ROOT)

    def module_file(name: str) -> Path | None:
        direct = base / (name.replace(".", "/") + ".py")
        if direct.is_file():
            return direct
        package = base / name.replace(".", "/") / "__init__.py"
        return package if package.is_file() else None

    def imports_of(path: Path) -> set[str]:
        found: set[str] = set()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names
                             if alias.name.startswith("src"))
            elif isinstance(node, ast.ImportFrom) and not node.level:
                if node.module and node.module.startswith("src"):
                    found.add(node.module)
                    for alias in node.names:
                        child = f"{node.module}.{alias.name}"
                        if module_file(child):
                            found.add(child)
        return found

    pending: list[str] = []
    for entry in entry_points:
        path = base / entry
        if path.is_file():
            pending.extend(imports_of(path))
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = module_file(name)
        if path is not None:
            pending.extend(imports_of(path))
    return {str(module_file(name).relative_to(base))
            for name in seen if module_file(name)}


# ---------------------------------------------------------------------------
# What the node needs in order to execute, and nothing else.
# ---------------------------------------------------------------------------

PACK_ALLOW: tuple[str, ...] = (
    "requirements.txt",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "GPU_NODE.md",

    # All of it, minus the subtrees denied above. The entry point's import
    # closure reaches widely across the training package, and a hand-kept
    # subset of *that* is a pack that classifies cleanly and does not import,
    # so the allowlist stays broad and the exclusions are stated as denials.
    # The two are reconciled by measurement rather than by belief:
    # ``tests/test_pack.py`` computes the real closure from
    # ``PACK_ENTRY_POINTS`` and fails if anything the node runs has begun to
    # import a denied subtree, or if anything in the closure stops travelling.
    "src/**/*.py",

    # Scripts are named one at a time, in the other direction. The node
    # executes; it does not rebuild the dataset. Shipping the dataset builders
    # would hand it everything it needs to become a second source of data,
    # which is exactly what the project forbids.
    "scripts/18_gpu_pack.py",
    "scripts/19_gpu_gate.py",
    "scripts/20_hypothesis_run.py",
    "scripts/22_final_train.py",

    # The core-acceptance entry point. Its ``--run`` mode is the node's whole
    # job for B/C/D/E; its ``--materialize`` mode is not usable here and is
    # not meant to be, because the test split it would read is not pinned by
    # ``REQUIRED_DATA`` and therefore never arrives. That is the guarantee:
    # not that the node has been asked nicely, but that the file is absent.
    "scripts/25_core_eval.py",

    # The boundary definitions themselves. This module reads its glob
    # semantics, its identifier patterns and its approval table from module
    # 17, so a pack without module 17 is a pack whose own verifier raises
    # FileNotFoundError the first time the node runs it -- which is how this
    # entry was found. Shipping it also makes the guarantee stronger rather
    # than weaker: the file is digested in the manifest like every other, so
    # the node's definition of "what counts as a credential" is provably the
    # Mac's rather than a copy that has to be trusted. It carries no secret
    # by construction, being itself a published file.
    "scripts/17_public_snapshot.py",

    # The six suites that are self-contained: they load no model, read no
    # dataset and reach no network, so they mean the same thing on the node as
    # they do here. The rest of the suite asserts against evidence the pack
    # deliberately does not carry, and a test that can only skip teaches the
    # operator that skipping is normal.
    #
    # ``test_gate_suite.py`` and ``test_arms.py`` travel for the same reason
    # their modules do: the node is where an unlock would be decided and where
    # an arm would run, and shipping the decision procedure and the runner
    # without the tests that say what they refuse would put the checks that
    # gate H1 and H2 on the machine with nothing there to contradict them.
    "tests/test_pack.py",
    "tests/test_gpu_node.py",
    "tests/test_gates.py",
    "tests/test_gate_suite.py",
    "tests/test_arms.py",
    "tests/test_final_run.py",
    "tests/test_core_eval.py",

    # The materialised core-acceptance plan, at the one location the contract
    # names. It carries six fields per case -- sample_id, pair_id, role,
    # variant, caption, inventory -- and no target, no reference bricks and no
    # used-parts column; ``acceptance.plan_leak_problems`` refuses those by
    # name *and* by running the project's brick parser over every string in
    # it, and the plan is not written if either check fires.
    #
    # It is deliberately not under ``runs/``. That tree is in VERIFY_IGNORE
    # as node run output, so a plan living there would be a file the pack's
    # own verifier walks past -- swappable after the pack was audited, with
    # every digest still agreeing. Here it is a manifest entry like any other.
    "gpu_plans/core_eval_plan.json",
)

#: The self-contained suites, named once. Both the allowlist above and the
#: test that guards it read this rather than each keeping its own copy.
PACKED_TEST_SUITES: tuple[str, ...] = (
    "tests/test_pack.py",
    "tests/test_gpu_node.py",
    "tests/test_gates.py",
    "tests/test_gate_suite.py",
    "tests/test_arms.py",
    "tests/test_final_run.py",
    "tests/test_core_eval.py",
)


#: Named, digested, and left behind. The node must have these files and they
#: must hash to this, but the pack does not carry them: they are the processed
#: per-record dataset, and moving them is a separate, deliberate act.
#:
#: **Only what the packed code can actually read.** The validation split was
#: pinned here once and no module in the pack ever opened it -- the gate path
#: (``longrun.ProductionChildDeps``) reads the training pool and nothing else,
#: and the two scripts that do read the val split are deliberately not in the
#: pack. The consequence was not academic: a pin nothing reads still has to be
#: transferred to the execution node, still sits there, and is still checked on
#: every verify, so an over-broad contract quietly widened what left this
#: machine. A test walks the packed modules and fails if anything here is
#: unreadable by all of them.
REQUIRED_DATA: tuple[str, ...] = (
    "data/processed/instruct_inv_train.jsonl",
)


#: The identifier kinds this boundary refuses. A subset of module 17's
#: ``SCANS`` -- the four the brief names, plus the two shapes that carry a
#: credential without looking like one.
#:
#: The kinds left out are left out on purpose. ``dataset-uuid``, ``hex32``,
#: ``raw-pid``, ``boot-fingerprint`` and ``precise-timestamp`` are
#: *publication* concerns: they identify a record or a machine to a stranger.
#: The node is not a stranger, it is the user's own second machine, and
#: auditing for them here would refuse builds for a reason the boundary does
#: not have -- and would need an approval table of its own beside module 17's,
#: which is the one thing this module exists to avoid.
PACK_SCAN_KINDS: tuple[str, ...] = (
    "credential",
    "credential-assignment",
    "bearer",
    "email",
    "organization-id",
    "private-key-block",
    "personal-path",
)


#: Present in a built pack without being in the manifest, because the node
#: creates them by running it. Everything else that appears is an extra file
#: and is refused: "it is only a note" is how a pack stops being the pack that
#: was audited.
VERIFY_IGNORE: tuple[str, ...] = (
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.DS_Store",
    "runs/**",

    # The dataset, which the node keeps at the same relative location the
    # repository uses so the code finds it unchanged. It is *not* an extra
    # file: the pack carries nothing under ``data/`` by construction, and the
    # two files that matter are pinned by digest in ``data_requirements`` and
    # checked by :func:`data_problems`. Refusing them here would mean a
    # correctly-assembled node could never verify, and the fix somebody
    # reached for would be to stop verifying.
    "data/**",
)


# ---------------------------------------------------------------------------
# Trust. A manifest is a set of digests computed over itself, so anybody who
# can write it can make every one of them agree. What makes a digest mean
# something is that it was carried here by a different route.
# ---------------------------------------------------------------------------

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


#: Why a carried digest is the only kind worth checking against. Shared by
#: both callers because the reasoning is the same one twice: a value stored
#: with the thing it authenticates is rewritten by whoever rewrites that thing.
_CARRIED_REASON = (
    "checking a thing against a value it supplied itself proves arithmetic "
    "and nothing else; the value this is compared against has to be the one "
    "the build machine printed, carried separately.")


def expected_digest_problems(value, *, what: str = "pack digest") -> list[str]:
    """Is this a usable digest to check against? Format only.

    ``what`` names which digest is missing or malformed. One validator rather
    than two, because "64 lowercase hex characters" is one fact, and two
    copies of it would eventually disagree about whether uppercase counts.
    """
    if value is None:
        return [f"no expected {what} was given. " + _CARRIED_REASON]
    if isinstance(value, bool) or not isinstance(value, str):
        return [f"the expected {what} is a {type(value).__name__}, not "
                "a string"]
    if not _HEX64.match(value):
        shown = value if len(value) <= 24 else value[:21] + "..."
        return [f"the expected {what} {shown!r} is not 64 lowercase "
                "hexadecimal characters"]
    return []


def trusted_digest_problems(dest, expected) -> list[str]:
    """Does the pack at ``dest`` match the digest carried from the build?

    The format is checked before the directory is touched, so a typo is a
    refusal rather than a read.
    """
    problems = expected_digest_problems(expected)
    if problems:
        return problems
    body, manifest_problems = read_manifest(dest)
    if body is None:
        return manifest_problems
    actual = body.get("pack_digest")
    if actual != expected:
        return [f"this pack declares pack_digest "
                f"{str(actual)[:16]}..., which is not the "
                f"{expected[:16]}... carried from the build machine. Every "
                "digest inside a manifest can be recomputed by whoever "
                "rewrote it, so a pack that is perfectly consistent with "
                "itself and inconsistent with this value is exactly the case "
                "this refuses."]
    return []


# ---------------------------------------------------------------------------
# Path closure. Every name below is resolved against the pack root, so a name
# that can leave the pack root is an instruction to read an unaudited file.
# ---------------------------------------------------------------------------

#: Segments that mean something other than "a directory of this name".
_UNSAFE_SEGMENTS = frozenset({"", ".", ".."})

_DRIVE_RE = re.compile(r"\A[A-Za-z]:")


def path_problems(rel, *, field: str = "path") -> list[str]:
    """Is this a normalised, repository-relative POSIX path?

    Anything else is refused rather than repaired. A manifest is not a place
    to be forgiving about what a name means: the entry says which bytes to
    check *and* where to find them, and every way of writing a name that
    resolves somewhere unexpected is a way of separating those two.
    """
    if isinstance(rel, bool) or not isinstance(rel, str):
        return [f"{field} {rel!r} is a {type(rel).__name__}, not a string"]
    if not rel or rel.strip() != rel:
        return [f"{field} {rel!r} is empty or padded with whitespace"]
    if "\\" in rel:
        return [f"{field} {rel!r} contains a backslash; manifest paths are "
                "POSIX and repository-relative"]
    if rel.startswith("/"):
        return [f"{field} {rel!r} is an absolute path"]
    if _DRIVE_RE.match(rel):
        return [f"{field} {rel!r} names a drive"]
    if rel.endswith("/"):
        return [f"{field} {rel!r} names a directory, not a file"]
    for segment in rel.split("/"):
        if segment in _UNSAFE_SEGMENTS:
            return [f"{field} {rel!r} has a {segment!r} segment; a manifest "
                    "path may not be relative to anything but the pack root"]
        if segment.strip() != segment:
            return [f"{field} {rel!r} has a segment padded with whitespace"]
    if posixpath.normpath(rel) != rel:
        return [f"{field} {rel!r} is not in normalised form"]
    return []


def manifest_path_problems(body) -> list[str]:
    """Every name the manifest carries, checked before any of them is used."""
    problems: list[str] = []
    files = (body or {}).get("files")
    if not isinstance(files, dict):
        problems.append("the manifest's file table is not an object")
    else:
        for rel in sorted(files, key=str):
            problems += path_problems(rel, field="manifest entry")
            entry = files[rel]
            if not isinstance(entry, dict):
                problems.append(f"{rel!r} maps to a "
                                f"{type(entry).__name__}, not an object")
                continue
            if "snapshot_name" not in entry:
                problems.append(
                    f"{rel!r} records no snapshot_name. Report 15 flattens "
                    "its snapshot names and a pack keeps the tree; an entry "
                    "that says neither cannot be resolved safely by either "
                    "reader.")
                continue
            name = entry["snapshot_name"]
            problems += path_problems(name, field=f"{rel!r} snapshot_name")
            if name != rel:
                problems.append(
                    f"{rel!r} has snapshot_name {name!r}. In a pack they are "
                    "the same string by construction, because the pack has to "
                    "import; a divergence is an entry whose digest describes "
                    "one file and whose location describes another.")

    requirements = (body or {}).get("data_requirements")
    if not isinstance(requirements, dict):
        problems.append("the manifest's data_requirements is not an object")
    else:
        got, want = set(requirements), set(REQUIRED_DATA)
        if got != want:
            extra = sorted(got - want)
            missing = sorted(want - got)
            problems.append(
                f"data_requirements names {sorted(got)}, not exactly the "
                f"declared {sorted(want)}"
                + (f"; unexpected {extra}" if extra else "")
                + (f"; absent {missing}" if missing else ""))
        for rel in sorted(got, key=str):
            problems += path_problems(rel, field="data requirement")
    return problems


def symlink_problems(root, rels) -> list[str]:
    """No manifest-named file, and no directory on its way, may be a link.

    A link is followed by every reader here -- ``sha256_file`` opens it,
    ``verify_sources`` digests what it points at -- so an allowlisted name can
    stand for any file on the machine while every digest still matches. The
    parent components matter as much as the leaf: one symbolic directory
    relocates everything beneath it at once.
    """
    root = Path(root)
    problems, seen = [], set()
    for rel in rels:
        if not isinstance(rel, str):
            continue
        parts = rel.split("/")
        for depth in range(1, len(parts) + 1):
            partial = "/".join(parts[:depth])
            if partial in seen:
                continue
            seen.add(partial)
            if (root / partial).is_symlink():
                what = "file" if depth == len(parts) else "parent directory"
                problems.append(
                    f"{partial} is a symbolic link, and it is the {what} of a "
                    "manifest entry. Every digest here would be taken of "
                    "whatever it points at, which is not what the manifest "
                    "says was packed.")
    return problems


def containment_problems(root, rels) -> list[str]:
    """After every link and every ``..`` is resolved, is it still inside?"""
    root = Path(root).resolve()
    problems = []
    for rel in rels:
        if not isinstance(rel, str):
            continue
        try:
            resolved = (Path(root) / rel).resolve()
        except OSError as exc:
            problems.append(f"{rel} could not be resolved ({exc})")
            continue
        if resolved != root and root not in resolved.parents:
            problems.append(
                f"{rel} resolves outside the pack. A manifest entry names "
                "what is in the pack; a name that leaves it is a name for "
                "something nobody audited.")
    return problems


def byte_problems(root, files: dict) -> list[str]:
    """The recorded size, against the file. Checked, not merely recorded."""
    root = Path(root)
    problems = []
    for rel in sorted(files, key=str):
        entry = files[rel] if isinstance(files.get(rel), dict) else {}
        recorded = entry.get("bytes")
        if isinstance(recorded, bool) or not isinstance(recorded, int) \
                or recorded < 0:
            problems.append(
                f"{rel}: the manifest records bytes {recorded!r}, which is "
                "not a size")
            continue
        path = root / rel
        if not path.is_file():
            continue
        actual = path.stat().st_size
        if actual != recorded:
            problems.append(
                f"{rel}: is {actual} bytes on disk and the manifest records "
                f"{recorded}")
    return problems


def approved_hits() -> dict:
    """Module 17's approval table. Not a copy of it."""
    return snapshot_module().APPROVED_HITS


def _matches(rel: str, pattern: str) -> bool:
    return snapshot_module()._matches(str(rel), pattern)


def classify(rel: str) -> tuple[str, str]:
    """``(verdict, reason)`` for one repository-relative path."""
    rel = str(rel)
    for pattern, why in PACK_DENY:
        if _matches(rel, pattern):
            return "exclude", f"denied by {pattern!r}: {why}"
    for pattern in PACK_ALLOW:
        if _matches(rel, pattern):
            return "include", f"allowed by {pattern!r}"
    return "exclude", "not on the allowlist"


def _prunable_dirs() -> tuple[tuple[str, str], ...]:
    """Directories the walk does not descend into, derived from the denials.

    Derived rather than listed again: a second list would be a second place
    for ``data/raw`` to stop being denied. Only the unambiguous shape counts
    -- a literal prefix followed by ``/**`` -- so a pattern with a wildcard in
    its prefix is still walked and classified file by file.
    """
    out = []
    for pattern, why in PACK_DENY:
        if pattern.endswith("/**") and "*" not in pattern[:-3]:
            out.append((pattern[:-3], why))
    return tuple(out)


def manifest(root: Path | None = None) -> dict[str, list]:
    """Every file in the tree, sorted into the two verdicts.

    Wholly-denied directories are pruned rather than walked, and recorded as
    one entry saying so. Walking ``data/raw`` to list every file we are about
    to refuse would take minutes and add nothing a reviewer wants to read.
    """
    root = Path(root or ROOT)
    pruned = _prunable_dirs()
    out: dict[str, list] = {"include": [], "exclude": []}
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = str(Path(dirpath).relative_to(root))
        rel_dir = "" if rel_dir == "." else rel_dir
        keep = []
        for name in sorted(dirnames):
            child = f"{rel_dir}/{name}" if rel_dir else name
            if name == "__pycache__":
                out["exclude"].append({"path": f"{child}/",
                                       "reason": "byte-compiled cache",
                                       "bytes": None, "pruned": True})
                continue
            hit = next((why for prefix, why in pruned if child == prefix), None)
            if hit is not None:
                out["exclude"].append({"path": f"{child}/",
                                       "reason": f"denied whole: {hit}",
                                       "bytes": None, "pruned": True})
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in sorted(filenames):
            rel = f"{rel_dir}/{name}" if rel_dir else name
            verdict, reason = classify(rel)
            path = root / rel
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            out[verdict].append({"path": rel, "reason": reason, "bytes": size})
    out["include"].sort(key=lambda e: e["path"])
    out["exclude"].sort(key=lambda e: e["path"])
    return out


# ---------------------------------------------------------------------------
# The identifier audit, run against what is about to be copied.
# ---------------------------------------------------------------------------

def pack_audit(paths, root=None, *, approved=None) -> list[str]:
    """Every identifier hit in ``paths`` that is not individually approved.

    The scan, the key and the approvals are module 17's; what is different
    here is which *kinds* count. Both directions are reported: an unapproved
    hit, and an approval that no longer matches anything -- the second because
    a stale approval is a sentence about a file that is no longer true.
    """
    m17 = snapshot_module()
    approved = m17.APPROVED_HITS if approved is None else approved
    kinds = set(PACK_SCAN_KINDS)

    hits = [h for h in m17.scan(paths, root=root) if h["kind"] in kinds]
    found: dict[str, dict[str, int]] = {}
    for hit in hits:
        key = m17.hit_key(hit["kind"], hit["match"])
        found.setdefault(hit["path"], {})
        found[hit["path"]][key] = found[hit["path"]].get(key, 0) + 1
    seen = {m17.hit_key(h["kind"], h["match"]): h for h in hits}

    def describe(key: str) -> str:
        hit = seen.get(key)
        if not hit:
            return key
        return (f"{key} ({hit['kind']} {hit['display']!r} at line "
                f"{hit['line']})")

    def in_scope(key: str) -> bool:
        return key.split("|", 1)[0] in kinds

    under_audit = {str(rel) for rel in paths}
    problems = []
    for path in sorted((set(found) | set(approved)) & under_audit):
        got = found.get(path, {})
        want = {k: v for k, v in approved.get(path, {}).items() if in_scope(k)}
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


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

class PackRefused(RuntimeError):
    """The pack stopped before copying anything."""


def data_requirements(root: Path | None = None) -> dict[str, dict]:
    """What the node's dataset must hash to, or a recorded absence.

    A file that is not here gets ``sha256: None`` and a reason, never a
    plausible-looking digest of nothing. The two must stay distinguishable:
    "the digest was never taken" and "the digest matches" are opposite
    answers, and collapsing them is how an unpinned dataset gets into a run.
    """
    root = Path(root or ROOT)
    out: dict[str, dict] = {}
    for rel in REQUIRED_DATA:
        path = root / rel
        if path.is_file():
            out[rel] = {"sha256": sha256_file(path),
                        "bytes": path.stat().st_size, "reason": None}
        else:
            out[rel] = {"sha256": None, "bytes": None,
                        "reason": ("absent from the tree the pack was built "
                                   "from, so this pack pins nothing for it")}
    return out


def data_digest(requirements: dict) -> str:
    from src.training.longrun import digest_obj

    return digest_obj({k: v.get("sha256") for k, v in requirements.items()})


def pack_digest(body: dict) -> str:
    """One value standing for the whole pack: its files and its data pins."""
    from src.training.longrun import digest_obj

    return digest_obj({"schema_version": body.get("schema_version"),
                       "kind": body.get("kind"),
                       "files_digest": body.get("files_digest"),
                       "data_digest": body.get("data_digest")})


def build(dest, root=None) -> dict:
    """Copy every included path into ``dest``, or refuse and copy nothing.

    Four refusals, none of which can be switched off: the destination has to
    be a dedicated empty directory that is neither the source tree nor an
    ancestor of it and is not reached through a symlink; no allowlisted path
    may be a symbolic link; every identifier hit in what is about to be copied
    has to be individually approved; and nothing under ``data/`` or
    ``artifacts/`` may have reached the include list at all.

    The last one is belt and braces over the denials above, and it stays:
    it is the assertion that would catch an editing mistake in the deny table
    itself, which is the table every other guarantee here rests on.
    """
    root = Path(root or ROOT)
    m17 = snapshot_module()
    m = manifest(root)
    included = [e["path"] for e in m["include"]]

    problems: list[str] = []
    problems += m17.destination_problems(dest, root)
    problems += m17.source_problems(included, root)
    problems += pack_audit(included, root)
    for rel in included:
        top = rel.split("/", 1)[0]
        if top in ("data", "artifacts"):
            problems.append(
                f"{rel} reached the include list; no path under {top}/ may "
                "travel in a pack")
    if problems:
        raise PackRefused("refusing to build the pack:\n  - "
                          + "\n  - ".join(problems))

    dest = Path(dest).expanduser().absolute()
    dest.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    for rel in included:
        copy_once(root / rel, dest / rel)
        src = root / rel
        files[rel] = {"sha256": sha256_file(src),
                      "bytes": src.stat().st_size,
                      # The path itself, because a pack has to import. Report
                      # 15 flattens; both say which, in the same field.
                      "snapshot_name": rel}

    requirements = data_requirements(root)
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "created_at": now_iso(),
        "root_relative": True,
        "files": files,
        "files_digest": manifest_digest({"files": files}),
        "data_requirements": requirements,
        "data_digest": data_digest(requirements),
    }
    body["pack_digest"] = pack_digest(body)
    write_once_json(dest / MANIFEST_NAME, body)
    return body


# ---------------------------------------------------------------------------
# Verifying
# ---------------------------------------------------------------------------

def read_manifest(dest) -> tuple[dict | None, list[str]]:
    path = Path(dest) / MANIFEST_NAME
    if not path.is_file():
        # The directory name only. This sentence ends up in preflight
        # evidence, which travels, and an absolute path names the account it
        # was built under.
        return None, [f"{MANIFEST_NAME} is missing from "
                      f"{path.parent.name!r}; an unverifiable pack is not a "
                      "pack"]
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, [f"{MANIFEST_NAME} could not be read as JSON ({exc}); "
                      "an unreadable manifest is a refusal, not an empty one"]
    if not isinstance(body, dict):
        return None, [f"{MANIFEST_NAME} is not an object"]
    return body, []


def _ignored(rel: str) -> bool:
    return any(_matches(rel, pattern) for pattern in VERIFY_IGNORE)


def verify(dest, *, data_root=None) -> list[str]:
    """Everything wrong with the pack at ``dest``, file by file.

    ``data_root`` is opt-in: a pack in transit is verified without the dataset
    beside it, and the node passes its own root once it has one. Asking for
    the data by default would make the ordinary check impossible to run at the
    one moment it matters most, which is before anything has been unpacked.
    """
    dest = Path(dest)
    body, problems = read_manifest(dest)
    if body is None:
        return problems

    if body.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"manifest schema_version is {body.get('schema_version')!r}, not "
            f"{SCHEMA_VERSION}; this reader cannot speak for it")
    if body.get("kind") != KIND:
        problems.append(f"manifest kind is {body.get('kind')!r}, not {KIND!r}")

    files = body.get("files")
    if not isinstance(files, dict) or not files:
        problems.append("the manifest records no files")
        return problems

    # Path closure before anything resolves a name. Every check after this
    # line opens ``dest / <name from the manifest>``, so a name that can leave
    # the pack must be refused rather than resolved: the failure it produces
    # otherwise is a *pass*, against a file one directory up that happens to
    # have the right bytes.
    path_issues = manifest_path_problems(body)
    if path_issues:
        return problems + path_issues

    named = sorted(files)
    reach = symlink_problems(dest, named) + containment_problems(dest, named)
    if reach:
        # Returned here rather than accumulated: the checks below would follow
        # exactly the links this just refused.
        return problems + reach

    # The existing verifier, not a second one. ``check_working_tree=False``
    # because on the node the pack *is* the tree: there is no earlier copy to
    # compare against, and asking for one would refuse every real pack.
    problems += verify_sources(dest, {"files": files}, dest,
                               check_working_tree=False)
    problems += byte_problems(dest, files)

    stated = body.get("files_digest")
    recomputed = manifest_digest({"files": files})
    if stated != recomputed:
        problems.append(
            f"files_digest is {stated!r} but the file table digests to "
            f"{recomputed!r}: the manifest disagrees with itself")

    stated_pack = body.get("pack_digest")
    recomputed_pack = pack_digest(body)
    if stated_pack != recomputed_pack:
        problems.append(
            f"pack_digest is {stated_pack!r} but recomputes to "
            f"{recomputed_pack!r}")

    expected = set(files) | {MANIFEST_NAME}
    for path in sorted(dest.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(dest))
        if rel in expected or _ignored(rel):
            continue
        problems.append(
            f"{rel} is present in the pack and not in its manifest; a pack "
            "with an extra file is not the pack that was audited")

    if data_root is not None:
        problems += data_problems(body, data_root)
    return problems


def data_problems(body: dict, data_root) -> list[str]:
    """The dataset beside the pack, checked against what the pack pinned."""
    data_root = Path(data_root)
    problems = []
    requirements = body.get("data_requirements")
    if not isinstance(requirements, dict) or not requirements:
        return ["the manifest pins no dataset, so nothing about the data on "
                "this machine can be checked"]
    for rel in sorted(requirements):
        # Same closure as the file table: this name is about to be joined to
        # a root and opened.
        unsafe = path_problems(rel, field="data requirement")
        if unsafe:
            problems += unsafe
            continue
        entry = requirements[rel] or {}
        expected = entry.get("sha256")
        path = data_root / rel
        if expected is None:
            problems.append(
                f"{rel}: this pack recorded no digest ({entry.get('reason')}), "
                "so no file on this machine can satisfy it. Rebuild the pack "
                "on a tree that has the dataset.")
            continue
        if not path.is_file():
            problems.append(f"{rel}: required by the pack and missing here")
        elif sha256_file(path) != expected:
            problems.append(
                f"{rel}: does not match the digest the pack pinned "
                f"(expected {expected[:12]}..., found "
                f"{sha256_file(path)[:12]}...)")
    return problems
