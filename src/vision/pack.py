"""The private vision pack: what may go to the CUDA node to fit a classifier.

Separate from :mod:`src.training.pack`, and separate on purpose.  That module's
central guarantee is that **no dataset byte travels**: the node already has the
instruction pool, so the pack carries digests and nothing else.  Fitting an
image classifier is the opposite situation -- the node has no images at all,
and the images *are* what has to travel.  Widening ``data/raw/**`` in the
existing pack to allow that would trade away an audited boundary for an
unrelated job, so this is a second boundary with its own answer rather than a
loosened first one.

What it shares with that module is every *definition*, which is the part that
must not fork:

* the identifier patterns and the approval table come from
  ``scripts/17_public_snapshot.py``;
* the manifest is the shape :mod:`src.training.session` already uses, so
  ``manifest_digest`` and ``verify_sources`` are this pack's digest and
  verifier rather than two more functions that mean nearly the same thing;
* the reason a carried digest is the only kind worth checking is the same
  reason, quoted from the same place.

What travels:

* the source modules the node's entry points actually import -- the measured
  closure, listed one file at a time -- and the three scripts the node needs
  to fit and check;
* the frozen classification data manifest and split manifest;
* the classification archive's selected members -- the eight classes only,
  about 207 MB, which is what fitting needs.

What does not, and each is a denial rather than an omission:

* **the detection images.**  Stage one of detection is deterministic
  segmentation and stage two is this classifier; nothing about detection is
  fitted, so its 553 MB has no reason to leave this machine.
* **weights of any kind**, including the generation track's ``final_H2``.  A
  vision pack has no business carrying a text-to-structure adapter.
* the processed text corpus, the frozen object split, ``PROJECT_STATUS.md``,
  ``CLAUDE.md``, the reference copy under ``BrickNet-master/``, credentials,
  caches and anything carrying a personal absolute path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

from src.training.session import manifest_digest, sha256_file, verify_sources

ROOT = Path(__file__).resolve().parents[2]

SNAPSHOT_SCRIPT = ROOT / "scripts" / "17_public_snapshot.py"

MANIFEST_NAME = "vision_pack_manifest.json"
KIND = "brickagain.vision_pack"
SCHEMA_VERSION = 1

_M17 = None


def snapshot_module():
    """Module 17, imported by path and cached.  One definition of a secret."""
    global _M17
    if _M17 is None:
        spec = importlib.util.spec_from_file_location(
            "brickagain_public_snapshot_for_vision", SNAPSHOT_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _M17 = module
    return _M17


class VisionPackRefused(RuntimeError):
    """The pack was not built.  Nothing was copied."""


# ---------------------------------------------------------------------------
# Denials first.  An unlisted path is excluded anyway; these are the ones
# whose reason for not travelling has to be legible rather than merely true.
# ---------------------------------------------------------------------------

#: ``src`` subtrees this pack deliberately does not carry.  Each is empty of
#: anything the entry points reach -- which ``tests/test_vision_pack.py``
#: measures -- and each is denied by name below rather than left to fall off
#: the end of the allowlist, so a reader sees the reason.
NON_PACKED_SUBTREES: tuple[tuple[str, str], ...] = (
    ("src/ui",
     "the interface, including an HTTP server. The node fits a model from a "
     "read-only payload and serves nothing"),
    ("src/retrieval",
     "the retrieval track: an embedding model, a vector index and a search. "
     "Nothing about fitting an image classifier reaches it"),
    ("src/delivery", "the delivery pipeline; not reachable from the fit"),
    ("src/colour", "colour assignment; not reachable from the fit"),
    ("src/assembly", "build ordering; not reachable from the fit"),
    ("src/demo", "the demonstration entry point"),
    ("src/generation", "the text-to-structure generation track"),
    ("src/eval", "the generation track's acceptance and scoring"),
    ("src/constraints", "the generation track's decoders"),
    ("src/inventory", "the generation track's inventory engine"),
)

#: Single modules denied by name inside subtrees that do travel.  The rest of
#: those subtrees is excluded by simply not being on the allowlist; these are
#: named because each one is worth a reader's attention.
NON_PACKED_MODULES: tuple[tuple[str, str], ...] = (
    ("src/vision/net.py",
     "the only module in this project that opens an outbound HTTP connection. "
     "The node does not acquire data -- the images travel inside this pack -- "
     "so nothing it runs imports this, and it does not go"),
    ("src/vision/detect.py",
     "multi-brick detection is evaluated on the Mac and is never fitted, so "
     "the node has no use for it"),
    ("src/rendering/preview.py",
     "the Matplotlib preview; not reachable from the fit, and it would pull "
     "an optional visual stack onto the node"),
    ("src/training/longrun.py",
     "the generation track's long-run driver; nothing here imports it"),
    ("src/training/lora.py", "the generation track's adapter code"),
    ("src/training/gates.py", "the generation track's gate decisions"),
    ("src/training/pack.py",
     "the *training* pack's boundary. This pack has its own, and shipping the "
     "other one would put a second answer on the node"),
)


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

    # Weights. Every kind, including the generation track's project model:
    # fitting an image classifier has no use for a text-to-structure adapter,
    # and a pack that could carry one is a pack that eventually does.
    ("artifacts/checkpoints/**", "model weights"),
    ("artifacts/renders/**", "local renders"),
    ("runs/**", "run output, including the project model pointer and adapters"),
    ("**/*.safetensors", "model weights"),
    ("**/*.ckpt", "checkpoint"),
    ("**/*.pt", "checkpoint"),
    ("**/*.pth", "checkpoint"),
    ("**/*.bin", "model weights"),
    ("**/*.gguf", "model weights"),
    ("**/*.h5", "model weights"),
    ("**/*.msgpack", "model weights"),

    # The text track's data and its frozen object split. Neither is used to
    # fit a vision classifier, and the split manifest is a list of 28,259
    # object hashes and 47,389 structure identifiers.
    ("data/processed/**", "processed per-record text dataset"),
    ("data/splits/**",
     "the frozen object split manifest is a per-object identifier list"),
    ("data/reports/**", "per-run session evidence and per-record reports"),
    ("gpu_plans/**", "the frozen core-acceptance plan is not this job"),

    # The detection archive. Detection is not fitted, so its images have no
    # reason to leave this machine.
    ("data/raw/vision/detection/**",
     "the detection archive is only evaluated, and evaluation stays on the "
     "Mac; nothing about detection is fitted"),
    ("data/raw/vision/detection_manifest.json",
     "belongs to the detection archive, which does not travel"),
    ("data/raw/vision/detection_split.json",
     "belongs to the detection archive, which does not travel"),
    ("data/raw/vision/detection_split.superseded-1.json",
     "a superseded split, kept on the record here and not sent anywhere"),
    ("data/raw/vision/_cd_cache/**",
     "a cache of archive central directories; the node does not read archives"),

    # Documents carrying this machine's own paths or a running account of an
    # unpublished tree.
    ("CLAUDE.md",
     "the collaborator guide names this machine's project directory, which is "
     "a personal absolute path"),
    ("PROJECT_STATUS.md",
     "a running account of the private tree, with a raw child process id"),
    ("PUBLIC_RELEASE_CHECKLIST.md",
     "describes the public repository's own cleanup"),
    ("BrickNet-master/**",
     "an untouched reference copy the operator downloaded; out of scope this "
     "round and not this project's to redistribute"),

    ("tests/test_public_snapshot.py",
     "the private release gate; it spawns the full suite against a tree that "
     "has the private evidence, so inside a pack it can only fail"),

    # The tracks this pack has no part of. Each subtree is empty of anything
    # the entry points reach -- measured, not assumed -- and is denied by name
    # so the exclusion carries a reason rather than being an absence.
    *((f"{subtree}/**", why) for subtree, why in NON_PACKED_SUBTREES),
    *NON_PACKED_MODULES,
)


# ---------------------------------------------------------------------------
# The import closure: what the node actually needs, measured rather than
# assumed.
# ---------------------------------------------------------------------------

#: The files the node runs.  Everything the payload carries out of ``src`` has
#: to be reachable from one of these, and the test that guards the allowlist
#: computes that reachability from this list rather than trusting it.
PACK_ENTRY_POINTS: tuple[str, ...] = (
    "scripts/32_vision_train.py",
    "scripts/36_vision_pack.py",
    "scripts/17_public_snapshot.py",
    "tests/test_vision_classes.py",
    "tests/test_vision_source.py",
    "tests/test_vision_split.py",
    "tests/test_vision_cv.py",
    "tests/test_vision_model.py",
    "tests/test_vision_pack.py",
)

#: Files reached only through a dynamic import, which a static reader cannot
#: see.  Listed one at a time, with the module that loads it and why, because
#: an unlisted dynamic import is a pack that passes its own closure test and
#: then fails to run on the node.
DYNAMIC_IMPORTS: tuple[tuple[str, str, str], ...] = (
    ("scripts/17_public_snapshot.py", "src/vision/pack.py",
     "loaded by path with importlib.util so the node's idea of what counts as "
     "a credential is provably the Mac's rather than a second copy"),
)

def _module_name(path: str) -> tuple[str, bool] | tuple[None, None]:
    """``src/vision/model.py`` -> ``("src.vision.model", False)``.

    The second value says whether the file is a package's ``__init__``, which
    is what decides how a relative import inside it resolves.
    """
    if not path.endswith(".py"):
        return None, None
    parts = path[:-3].split("/")
    if parts[-1] == "__init__":
        return ".".join(parts[:-1]), True
    return ".".join(parts), False


def _relative_base(module: str | None, is_package: bool,
                   level: int) -> str | None:
    """The package a relative import counts from, or ``None`` if it climbs out.

    One dot is the importing module's own package; each further dot goes one
    level up.  A module's package is its parent; a package's ``__init__`` is
    its own package, which is why that distinction is carried this far.
    """
    if module is None:
        return None
    parts = module.split(".")
    base = parts if is_package else parts[:-1]
    if level - 1 > len(base):
        return None
    root = base[:len(base) - (level - 1)]
    return ".".join(root) if root else None


def import_closure(root=None, entry_points=None
                   ) -> tuple[set[str], set[str]]:
    """Every ``src`` file the entry points reach, plus the package markers.

    Returned as ``(modules, markers)``.  A module is reached by a static read
    of the imports; a marker is an ``__init__.py`` on the path to a reached
    module, which Python needs present even though nothing names it.

    Relative imports are resolved against the importing module rather than
    skipped, and a relative import that cannot be resolved is a refusal: a
    silent skip there would drop a real dependency out of the payload and the
    node would find out by failing to import.

    A static read cannot see a dynamic import, so those are declared in
    :data:`DYNAMIC_IMPORTS` and checked against the allowlist separately.
    """
    import ast

    base = Path(root or ROOT)
    # Resolved here rather than in the signature so the constant is read at
    # call time: a default bound at definition time cannot be replaced, and a
    # test that cannot vary the entry points cannot show what happens when
    # there are none.
    entry_points = (PACK_ENTRY_POINTS if entry_points is None
                    else tuple(entry_points))

    def module_file(name: str) -> Path | None:
        direct = base / (name.replace(".", "/") + ".py")
        if direct.is_file():
            return direct
        package = base / name.replace(".", "/") / "__init__.py"
        return package if package.is_file() else None

    def imports_of(path: Path, rel: str) -> set[str]:
        module, is_package = _module_name(rel)
        found: set[str] = set()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            raise VisionPackRefused(
                f"{rel} cannot be read for its imports: {exc}") from None
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names
                             if alias.name.split(".")[0] == "src")
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            # ``from X import a, b`` names one module and then some things
            # inside it. The module is the dependency; a name inside it is a
            # second dependency only when it is itself a module. Treating an
            # alias as the module -- which an earlier version did for the
            # relative form -- resolves ``from .two import value`` to
            # ``pkg.value``, which does not exist, and drops ``two.py`` out of
            # the payload with nothing saying so.
            if node.level:
                base = _relative_base(module, bool(is_package), node.level)
                if base is None:
                    dots = "." * node.level
                    raise VisionPackRefused(
                        f"{rel} has a relative import "
                        f"({dots}{node.module or ''}) that cannot be "
                        "resolved, so the pack cannot know what it depends on")
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module
            if not target or target.split(".")[0] != "src":
                continue
            found.add(target)
            for alias in node.names:
                if alias.name == "*":
                    # A star import names no submodule; the module it came
                    # from is already recorded above.
                    continue
                child = f"{target}.{alias.name}"
                if module_file(child):
                    found.add(child)
        return found

    pending: list[str] = []
    for entry in entry_points:
        path = base / entry
        if path.is_file():
            pending.extend(imports_of(path, entry))
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = module_file(name)
        if path is not None:
            pending.extend(
                imports_of(path, path.relative_to(base).as_posix()))

    modules = {module_file(name).relative_to(base).as_posix()
               for name in seen if module_file(name)}
    markers: set[str] = set()
    for rel in modules:
        parts = rel.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            marker = "/".join(parts[:depth] + ["__init__.py"])
            if (base / marker).is_file():
                markers.add(marker)
    return modules, markers - modules


# ---------------------------------------------------------------------------
# What the node needs to fit the classifier, and nothing else.
# ---------------------------------------------------------------------------

#: The source files the payload carries, written out rather than globbed.
#: ``src/**/*.py`` carried all 83 modules in the project, most of which the
#: node never imports; this is the measured closure of
#: :data:`PACK_ENTRY_POINTS` plus the package markers Python needs to resolve
#: it, and ``tests/test_vision_pack.py`` recomputes it and fails on any drift
#: in either direction.
PACK_SOURCE_MODULES: tuple[str, ...] = (
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/bricks.py",
    "src/rendering/__init__.py",
    "src/rendering/ldr.py",
    "src/training/__init__.py",
    "src/training/session.py",
    "src/vision/__init__.py",
    "src/vision/classes.py",
    "src/vision/cv_baseline.py",
    "src/vision/datasets.py",
    "src/vision/metrics.py",
    "src/vision/model.py",
    "src/vision/model_ids.py",
    "src/vision/pack.py",
    "src/vision/preprocess.py",
    "src/vision/schema.py",
    "src/vision/segment.py",
    "src/vision/selection.py",
    "src/vision/source.py",
    "src/vision/split.py",
    "src/vision/train.py",
)

PACK_ALLOW: tuple[str, ...] = (
    "requirements.txt",
    "requirements-vision.txt",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "GPU_NODE.md",
    "VISION.md",

    # The measured closure, one file at a time. Not a glob: see
    # PACK_SOURCE_MODULES above for why, and tests/test_vision_pack.py for the
    # check that keeps this list honest.
    *PACK_SOURCE_MODULES,

    # Scripts named one at a time, in the other direction. The node fits and
    # verifies; it does not acquire data, freeze a split or build an index.
    "scripts/32_vision_train.py",
    "scripts/36_vision_pack.py",

    # The boundary definition itself, so the node's idea of what counts as a
    # credential is provably the Mac's rather than a copy to be trusted. It is
    # a published file and carries no secret by construction.
    "scripts/17_public_snapshot.py",

    # The self-contained vision suites: they load no weights, reach no network
    # and read only fixtures they build themselves, so they mean the same on
    # the node as here.
    "tests/test_vision_classes.py",
    "tests/test_vision_source.py",
    "tests/test_vision_split.py",
    "tests/test_vision_cv.py",
    "tests/test_vision_model.py",
    "tests/test_vision_pack.py",

    # The frozen manifests for the archive that does travel. Not under a
    # denied prefix, and each is a manifest entry like any other file.
    "data/raw/vision/classification_manifest.json",
    "data/raw/vision/classification_split.json",

    # The images themselves. This is the difference from the training pack:
    # the node cannot fit a classifier on digests.
    "data/raw/vision/classification/**",
)


def closure_problems(root=None) -> list[str]:
    """Does the declared payload still equal the measured closure?

    Run by :func:`pack_audit` and again by :func:`build`, not only by the test
    suite.  A pack is built by an operator at a terminal, and the drift this
    catches -- a new import that nothing carries, or a carried file nothing
    imports -- is exactly the kind that would otherwise be found on the node.
    """
    try:
        modules, markers = import_closure(root=root)
    except VisionPackRefused as exc:
        return [str(exc)]
    if not modules:
        return ["the import closure came back empty, so the reader is broken; "
                "an empty closure would let the allowlist say anything"]
    declared = set(PACK_SOURCE_MODULES)
    reached = modules | markers
    problems = []
    for rel in sorted(reached - declared):
        problems.append(
            f"{rel} is imported by something the node runs and is not in "
            "PACK_SOURCE_MODULES, so it would not travel")
    for rel in sorted(declared - reached):
        problems.append(
            f"{rel} is in PACK_SOURCE_MODULES and nothing the node runs "
            "imports it")
    for rel in sorted(declared):
        verdict, reason = classify(rel)
        if verdict != "include":
            problems.append(
                f"{rel} is declared as payload and classifies as {verdict} "
                f"({reason})")
    return problems

#: The self-contained suites, named once so the allowlist and the test that
#: guards it read the same list.
PACKED_TEST_SUITES: tuple[str, ...] = (
    "tests/test_vision_classes.py",
    "tests/test_vision_source.py",
    "tests/test_vision_split.py",
    "tests/test_vision_cv.py",
    "tests/test_vision_model.py",
    "tests/test_vision_pack.py",
)

#: The identifier kinds this boundary refuses.  The same subset of module 17's
#: ``SCANS`` the training pack uses, and for the same reason: the node is the
#: operator's own second machine, so publication-only kinds -- dataset uuids,
#: raw process ids, boot fingerprints, precise timestamps -- are not this
#: boundary's concern and auditing for them here would need a second approval
#: table beside module 17's.
PACK_SCAN_KINDS: tuple[str, ...] = (
    "credential",
    "credential-assignment",
    "bearer",
    "email",
    "organization-id",
    "private-key-block",
    "personal-path",
)

#: Only text files are scanned for identifiers.  A JPEG is not text, and
#: decoding two hundred megabytes of it as UTF-8 to look for the word
#: "password" would be theatre.  The images are the *subject* of this pack and
#: their content is a photograph of a brick; what protects them is that they
#: never enter git or the public snapshot, which is a separate boundary that
#: has its own test.
SCANNED_SUFFIXES = (".py", ".md", ".txt", ".json", ".jsonl", ".cfg", ".toml",
                    ".yaml", ".yml", "")

#: Present in a built pack without being in the manifest, because the node
#: creates them by running it.
VERIFY_IGNORE: tuple[str, ...] = (
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.DS_Store",
    "runs/**",
)


def _matches(rel: str, pattern: str) -> bool:
    return snapshot_module()._matches(rel, pattern)


def classify(rel: str) -> tuple[str, str]:
    """``(verdict, reason)`` for one repository-relative path."""
    rel = str(rel)
    for pattern, why in PACK_DENY:
        if _matches(rel, pattern):
            return "exclude", f"denied by {pattern!r}: {why}"
    for pattern in PACK_ALLOW:
        if _matches(rel, pattern):
            return "include", f"allowed by {pattern!r}"
    return "exclude", "not on the vision pack allowlist"


#: Directories not walked into at all.  Walking ``.venv`` and the detection
#: archive to classify every file one at a time costs minutes and decides
#: nothing: both are denied wholesale above.
_PRUNE = ("/.git", "/.venv", "/.pytest_cache", "/__pycache__",
          "/BrickNet-master", "/data/raw/vision/detection",
          "/data/raw/vision/_cd_cache", "/data/processed", "/data/reports",
          "/artifacts/checkpoints", "/artifacts/renders", "/runs")


def manifest_paths(root: Path | None = None) -> dict[str, list]:
    """Every file in the tree, sorted into include and exclude."""
    root = Path(root or ROOT)
    out: dict[str, list] = {"include": [], "exclude": []}
    for base, directories, files in os.walk(root):
        relative_base = "/" + str(Path(base).relative_to(root)).replace(
            os.sep, "/")
        if relative_base == "/.":
            relative_base = ""
        directories[:] = [
            name for name in sorted(directories)
            if not any((relative_base + "/" + name).startswith(prune)
                       for prune in _PRUNE)]
        for name in sorted(files):
            rel = str(Path(base, name).relative_to(root)).replace(os.sep, "/")
            verdict, reason = classify(rel)
            entry = {"path": rel, "reason": reason}
            if verdict == "include":
                entry["bytes"] = (Path(base) / name).stat().st_size
            out[verdict].append(entry)
    return out


def scanned(paths) -> list[str]:
    """Which included paths are text this boundary reads for identifiers."""
    return [rel for rel in paths
            if Path(rel).suffix.lower() in SCANNED_SUFFIXES]


def pack_audit(paths, root=None) -> list[str]:
    """Unapproved identifier hits in the text files about to travel."""
    module = snapshot_module()
    wanted = set(PACK_SCAN_KINDS)
    missing = wanted - {name for name, _pattern in module.SCANS}
    if missing:
        raise VisionPackRefused(
            f"these scan kinds are no longer defined in module 17: "
            f"{sorted(missing)}. A boundary scanning for nothing is worse "
            "than no boundary")
    text_paths = scanned(paths)
    found = module.hit_counter(text_paths, root=root)
    approved = module.APPROVED_HITS
    problems = closure_problems(root)
    for path in sorted(found):
        for key, count in sorted(found[path].items()):
            if key.split("|", 1)[0] not in wanted:
                continue
            allowed = (approved.get(path) or {}).get(key, 0)
            if count > allowed:
                problems.append(
                    f"{path}: {count} hit(s) of {key}, {allowed} approved by "
                    "the shared table")
    return problems


def destination_problems(dest, root=None) -> list[str]:
    """Is this a safe, empty, non-nested place to write a pack into?"""
    root = Path(root or ROOT).resolve()
    given = Path(dest).expanduser().absolute()
    real = given.resolve()
    problems = []
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
            f"{real} contains the source tree; building here would put a pack "
            "around the private tree")
    if real.exists():
        if not real.is_dir():
            problems.append(f"{real} exists and is not a directory")
        elif sorted(x.name for x in real.iterdir()):
            existing = sorted(x.name for x in real.iterdir())
            problems.append(
                f"{real} is not empty ({len(existing)} entries, first: "
                f"{existing[0]!r}). A pack goes into a dedicated empty "
                "directory: anything already there would survive the copy and "
                "travel with it")
    return problems


def symlink_problems(paths, root=None) -> list[str]:
    """Nothing on the allowlist may be a symbolic link.

    ``copy2`` would copy the target's bytes, so an allowlisted name could
    stand in for any file on this machine -- including one the denylist covers.
    """
    root = Path(root or ROOT)
    return [f"{rel} is a symbolic link to {os.readlink(root / rel)!r}; an "
            "allowlisted path must be a real file"
            for rel in paths if (root / rel).is_symlink()]


def required_present(root=None) -> list[str]:
    """The manifests and images a fitting run cannot start without."""
    root = Path(root or ROOT)
    problems = []
    for rel in ("data/raw/vision/classification_manifest.json",
                "data/raw/vision/classification_split.json"):
        if not (root / rel).is_file():
            problems.append(
                f"{rel} is missing; build it with scripts/30_vision_data.py "
                "--extract and scripts/31_vision_split.py --freeze before "
                "packing")
    return problems


def build(dest, root=None) -> dict:
    """Copy the included paths into ``dest``, or refuse and copy nothing.

    Six checks first, and all six refuse rather than warn: the destination has
    to be a dedicated empty place outside the tree, the required manifests
    have to exist, no allowlisted path may be a symlink, every identifier hit
    in the text that travels has to be individually approved by the shared
    table, the data manifest and split manifest have to agree with each other,
    and the declared source payload has to equal the measured import closure.
    There is no flag to switch any of them off.
    """
    root = Path(root or ROOT)
    listing = manifest_paths(root)
    included = [entry["path"] for entry in listing["include"]]

    problems = destination_problems(dest, root)
    problems += required_present(root)
    problems += symlink_problems(included, root)
    problems += pack_audit(included, root)
    problems += data_agreement_problems(root)
    problems += closure_problems(root)
    if problems:
        raise VisionPackRefused(
            "refusing to build:\n  - " + "\n  - ".join(problems))

    destination = Path(dest).expanduser().absolute()
    destination.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    for rel in included:
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target, follow_symlinks=False)
        files[rel] = {"sha256": sha256_file(root / rel),
                      "bytes": (root / rel).stat().st_size,
                      "snapshot_name": rel}

    body = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "job": ("fit the eight-class brick image classifier on the public "
                "single-brick archive, on CUDA, from the frozen data and "
                "split manifests carried here"),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files.values()),
        "data_manifest_sha256": files[
            "data/raw/vision/classification_manifest.json"]["sha256"],
        "split_manifest_sha256": files[
            "data/raw/vision/classification_split.json"]["sha256"],
        "images": sum(1 for rel in files
                      if rel.startswith("data/raw/vision/classification/")),
        "carries": (
            "source, three scripts, six self-contained vision suites, the two "
            "frozen manifests, and the eight-class members of the "
            "classification archive"),
        "does_not_carry": (
            "any weights including the generation track's final_H2, the "
            "detection archive, the processed text corpus, the frozen object "
            "split, per-run evidence, credentials, or any document carrying a "
            "personal absolute path"),
        "node_rules": (
            "the node fits and returns. It does not modify the project, use "
            "git, download anything, hold credentials, or become a second "
            "source of code or data. See GPU_NODE.md"),
    }
    digest = manifest_digest(body)
    body["pack_digest"] = digest
    (destination / MANIFEST_NAME).write_text(
        json.dumps(body, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8")
    return body


def data_agreement_problems(root=None) -> list[str]:
    """The split has to be a split of the data that is travelling."""
    from src.vision import datasets
    from src.vision.split import VisionSplit

    root = Path(root or ROOT)
    manifest_file = root / "data/raw/vision/classification_manifest.json"
    split_file = root / "data/raw/vision/classification_split.json"
    if not manifest_file.is_file() or not split_file.is_file():
        return []
    problems = []
    try:
        manifest = datasets.read_manifest(manifest_file)
        split = VisionSplit.load(split_file)
        split.check_no_leakage()
    except Exception as exc:                    # noqa: BLE001 - see message
        return [f"the frozen manifests are unusable: {exc}"]
    members = {row["member"] for row in manifest["records"]}
    missing = sorted(set(split.items) - members)
    extra = sorted(members - set(split.items))
    if missing:
        problems.append(
            f"{len(missing)} split item(s) are not in the data manifest, "
            f"first {missing[:3]}")
    if extra:
        problems.append(
            f"{len(extra)} data manifest record(s) are not in the split, "
            f"first {extra[:3]}")
    for member in sorted(members)[:1] if members else []:
        if not (root / "data/raw/vision/classification" / member).is_file():
            problems.append(
                f"the manifest names {member}, which is not on disk; extract "
                "the archive before packing")
    return problems


def read_manifest(dest) -> tuple[dict | None, list[str]]:
    """Read a built pack's manifest, refusing a shape it should not have."""
    path = Path(dest) / MANIFEST_NAME
    if not path.is_file():
        return None, [f"{MANIFEST_NAME} is missing from {dest}"]
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{MANIFEST_NAME} is not valid JSON: {exc}"]
    if not isinstance(body, dict) or body.get("kind") != KIND:
        return None, [f"{MANIFEST_NAME} does not declare itself a {KIND}"]
    if not isinstance(body.get("files"), dict) or not body["files"]:
        return None, [f"{MANIFEST_NAME} records no files"]
    return body, []


def _ignored(rel: str) -> bool:
    return any(_matches(rel, pattern) for pattern in VERIFY_IGNORE)


def verify(dest, *, expected_digest: str | None = None,
           root=None, check_working_tree: bool = True) -> list[str]:
    """Check a built pack file by file, and against a carried digest.

    ``expected_digest`` is checked because a manifest is a set of digests
    computed over itself: anybody who can rewrite the pack can make every one
    of them agree.  The value has to arrive by a different route -- the same
    reasoning the training pack states, and for the same reason.
    """
    destination = Path(dest)
    body, problems = read_manifest(destination)
    if body is None:
        return problems

    digest = manifest_digest(body)
    if body.get("pack_digest") != digest:
        problems.append(
            f"the manifest records pack_digest {body.get('pack_digest')} and "
            f"its own files hash to {digest}")
    if expected_digest is None:
        problems.append(
            "no expected pack digest was given. Checking a thing against a "
            "value it supplied itself proves arithmetic and nothing else; the "
            "value this is compared against has to be the one the build "
            "machine printed, carried separately")
    elif expected_digest != digest:
        problems.append(
            f"the pack hashes to {digest}, not the expected {expected_digest}")

    problems += verify_sources(
        Path(root or ROOT), body, destination,
        check_working_tree=check_working_tree)

    known = set(body["files"])
    for base, directories, files in os.walk(destination):
        directories[:] = [name for name in directories
                          if name not in ("__pycache__", ".git")]
        for name in files:
            rel = str(Path(base, name).relative_to(destination)).replace(
                os.sep, "/")
            if rel == MANIFEST_NAME or rel in known or _ignored(rel):
                continue
            problems.append(
                f"{rel} is in the pack and not in its manifest; a pack with an "
                "extra file is not the pack that was audited")
    return problems
