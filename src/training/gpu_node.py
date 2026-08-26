"""The execution node's preflight: fail closed, or do not start.

The Taichung machine runs Windows 11 with an RTX 5070 Ti, and the training
runs inside WSL2 Ubuntu because that is where a CUDA PyTorch build actually
lives. It is an **execution node**: it runs a pack the Mac built, checked file
by file against that pack's own manifest and then against a digest carried here
by a separate route, and it is not a second place where this project is
developed. Both sentences are checks below, not conventions.

Every check answers from a reading. Three consequences follow, and they are
the whole design:

* **an unreadable value is a failure.** ``None`` means the probe could not be
  taken, and a gate that cannot be evaluated has not been satisfied. This is
  report 15's rule (``src/training/preflight.py``) applied to a different
  machine, and the result has the same shape -- ``passed``, ``checks``,
  ``failed`` -- so there is one vocabulary for "the machine was not ready";
* **there is no fallback.** If CUDA is unavailable the answer is no. Not the
  CPU, not a smaller dtype, not 4-bit. A run that silently changes device or
  precision has quietly become a different experiment, and its numbers will be
  compared against ones from the experiment it stopped being;
* **the pack is the only thing it runs.** Verified file by file against its
  own manifest before anything is loaded, with the dataset checked against the
  digests the pack pinned. A drifted byte is a refusal.

The probe reads and reports; the preflight judges. They are separate so that
every judgement in the second can be driven from an injected reading in a test,
without a GPU, a model or a network anywhere near the process.
"""

from __future__ import annotations

import platform as _platform
from pathlib import Path

from src.training import pack
from src.training.longrun import (PRODUCTION_OFFLINE_ENV, _portable,
                                  dependency_digest, dependency_preflight)
from src.training.lora import LoraConfig_

#: Report 16's own dependency preflight, not a second implementation of it.
#: It resolves the pinned tokenizer, base model and published adapter from the
#: local cache, reads no network, loads no tensor and initialises no device --
#: which is exactly what this node needs to know before it spends anything.
#: Report 16 learned the cost of finding out later: its first measured run
#: passed the gate, wrote ``measurement_started``, spawned the child, and only
#: then discovered the tokenizer could not be resolved. The boot was gone.
DEFAULT_DEPENDENCY_CHECKER = dependency_preflight

#: The node, as the collaborator guide records it. Written down so the
#: preflight can refuse a machine that is not this one: a run that lands on
#: different hardware is not the run that was planned, and the cheapest moment
#: to notice is before the model loads.
NODE_SPEC = {
    "platform": "Windows 11 with WSL2 Ubuntu",
    "cpu": "AMD Ryzen 5 7600",
    "gpu": "NVIDIA GeForce RTX 5070 Ti",
    "vram_gb": 16,
    "system_ram_gb": 32,
    "role": "execution node only; never a second development source",
}

#: Matched as a substring of the reported device name. A model number rather
#: than the full string because vendors punctuate it differently between
#: driver versions, and a check that breaks on a driver update is a check
#: somebody turns off.
EXPECTED_GPU = "5070 Ti"

#: 16 GB nominal. The driver reports a little less than the box says, and a
#: floor at the nominal figure would refuse the very card it is written for.
MIN_VRAM_GB = 15.0

#: 32 GB nominal, same reasoning.
MIN_SYSTEM_RAM_GB = 30.0

REQUIRED_DEVICE = "cuda"

#: From the one configuration that already carries the project's dtype
#: decision and its reasons, so this cannot become a second opinion on it.
REQUIRED_DTYPE = LoraConfig_().dtype
REQUIRED_QUANTIZATION = "none"

#: The same three variables the measured child on the Mac is pinned to. Named
#: from that module rather than repeated, so "offline" means one thing.
REQUIRED_OFFLINE_ENV = PRODUCTION_OFFLINE_ENV


# ---------------------------------------------------------------------------
# Allocator configuration. Provenance, not a preference.
# ---------------------------------------------------------------------------

#: PyTorch renamed the variable; both names are still read, and a run that
#: sets one to one thing and the other to another has no single answer to
#: "which allocator produced this number".
ALLOC_ENV_PRIMARY = "PYTORCH_ALLOC_CONF"
ALLOC_ENV_ALIAS = "PYTORCH_CUDA_ALLOC_CONF"

#: What every measured run on this node must have had in its environment
#: *before it started*. Measured here: with the native segment policy this
#: node reserved 15.477 GB while 7.150 GB was live; with expandable segments
#: it reserved 7.635 GB for the same work. A caching allocator reads its
#: configuration once, at first use, so a program that sets this for itself
#: has already lost -- and could not afterwards be told apart from one that
#: inherited it. It is required, checked before the model loads, and never
#: written by anything here.
REQUIRED_ALLOC_OPTIONS = (("expandable_segments", "True"),)

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def normalize_alloc_conf(raw) -> str | None:
    """One canonical spelling of an allocator config, or ``None``.

    Sorted by key, whitespace stripped, keys lowercased and booleans written
    one way, so ``expandable_segments:true`` and ``expandable_segments:True``
    compare equal -- otherwise the alias check would report a conflict between
    two spellings of the same thing.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    options = {}
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        key, sep, value = chunk.partition(":")
        if not sep:
            return None
        key, value = key.strip().lower(), value.strip()
        low = value.lower()
        if low in _TRUE:
            value = "True"
        elif low in _FALSE:
            value = "False"
        options[key] = value
    if not options:
        return None
    return ",".join(f"{k}:{options[k]}" for k in sorted(options))


def allocator_config_from_env(env) -> tuple[str | None, list[str]]:
    """The normalized config this process inherited, and what is wrong with it."""
    env = env or {}
    primary = normalize_alloc_conf(env.get(ALLOC_ENV_PRIMARY))
    alias = normalize_alloc_conf(env.get(ALLOC_ENV_ALIAS))
    raw_primary = env.get(ALLOC_ENV_PRIMARY)
    raw_alias = env.get(ALLOC_ENV_ALIAS)

    if raw_primary and primary is None:
        return None, [f"{ALLOC_ENV_PRIMARY} is set to something this reader "
                      "cannot parse as an allocator config"]
    if raw_alias and alias is None:
        return None, [f"{ALLOC_ENV_ALIAS} is set to something this reader "
                      "cannot parse as an allocator config"]
    if primary and alias and primary != alias:
        return None, [
            f"{ALLOC_ENV_PRIMARY} says {primary!r} and {ALLOC_ENV_ALIAS} says "
            f"{alias!r}. The two names are aliases and they conflict; which "
            "one the allocator honoured is a property of the torch build, so "
            "there is no answer here that is true of the run."]
    config = primary or alias
    if config is None:
        return None, [
            f"neither {ALLOC_ENV_PRIMARY} nor {ALLOC_ENV_ALIAS} is set. The "
            "allocator reads its configuration once, before this process can "
            "influence it, so it has to be exported before launch -- nothing "
            "here will supply it, because a run that configured itself could "
            "not be told from one that inherited a different setting."]
    return config, allocator_config_problems(config)


def allocator_config_problems(config) -> list[str]:
    """Is this normalized config the one every measured run must have had?"""
    if not isinstance(config, str) or not config.strip():
        return ["no allocator configuration was recorded"]
    options = dict(chunk.partition(":")[::2] for chunk in config.split(",")
                   if ":" in chunk)
    problems = []
    for key, wanted in REQUIRED_ALLOC_OPTIONS:
        got = options.get(key)
        if got is None:
            problems.append(
                f"the allocator config {config!r} does not set {key!r}")
        elif got != wanted:
            problems.append(
                f"the allocator config sets {key}:{got}, not {key}:{wanted}")
    return problems


# ---------------------------------------------------------------------------
# Determinism. Strict, or it is not a determinism mode.
# ---------------------------------------------------------------------------

CUBLAS_WORKSPACE_ENV = "CUBLAS_WORKSPACE_CONFIG"

#: The two values cuBLAS accepts for deterministic reductions. Like the
#: allocator config this is read at library init, so it has to be exported
#: before launch.
CUBLAS_WORKSPACE_VALUES = (":4096:8", ":16:8")


def determinism_env_problems(env) -> list[str]:
    value = (env or {}).get(CUBLAS_WORKSPACE_ENV)
    if not value:
        return [f"{CUBLAS_WORKSPACE_ENV} is not set; cuBLAS chooses a "
                "non-deterministic reduction without it, and it is read at "
                f"library initialisation so it must be exported before launch. "
                f"Use one of {list(CUBLAS_WORKSPACE_VALUES)}."]
    if value not in CUBLAS_WORKSPACE_VALUES:
        return [f"{CUBLAS_WORKSPACE_ENV} is {value!r}, which is not one of "
                f"{list(CUBLAS_WORKSPACE_VALUES)}"]
    return []


def apply_determinism(torch_mod, *, seed: int, env=None) -> dict:
    """Turn strict determinism on, and report what is actually in effect.

    ``warn_only`` is never requested and never tolerated. It converts "this
    operation has no deterministic implementation" into a log line, which
    answers the repeatability question wrong while looking like it answered
    it right. An operation without a deterministic kernel must raise where it
    is called, and nothing here catches that.

    Every value returned is *read back* rather than assumed. Asking for a
    setting and recording the request would record an intention; what a run
    needs frozen is what was true.
    """
    if env is None:
        import os

        env = dict(os.environ)
    problems = determinism_env_problems(env)
    if problems:
        raise RuntimeError("refusing to enable determinism: "
                           + "; ".join(problems))

    torch_mod.use_deterministic_algorithms(True, warn_only=False)
    torch_mod.backends.cudnn.benchmark = False
    torch_mod.backends.cudnn.deterministic = True
    torch_mod.manual_seed(seed)

    def read(fn, default=None):
        try:
            return fn()
        except Exception:
            return default

    enabled = read(torch_mod.are_deterministic_algorithms_enabled)
    warn_only = read(
        torch_mod.is_deterministic_algorithms_warn_only_enabled, None)
    if enabled is not True:
        raise RuntimeError(
            "deterministic algorithms were requested and torch reports them "
            f"as {enabled!r}. A mode that is asked for and not in effect is "
            "worse than no mode: every number it produces looks deterministic.")
    if warn_only is not False:
        raise RuntimeError(
            f"torch reports warn_only={warn_only!r}. warn_only downgrades a "
            "missing deterministic kernel to a warning, so the run would "
            "continue non-deterministically and say so only in passing.")

    return {
        "use_deterministic_algorithms": True,
        "warn_only": False,
        "cudnn_benchmark": bool(torch_mod.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch_mod.backends.cudnn.deterministic),
        "cublas_workspace_config": env.get(CUBLAS_WORKSPACE_ENV),
        "tf32_matmul_allowed": bool(
            read(lambda: torch_mod.backends.cuda.matmul.allow_tf32, None)),
        "tf32_cudnn_allowed": bool(
            read(lambda: torch_mod.backends.cudnn.allow_tf32, None)),
        "seed": int(seed),
    }


DETERMINISM_FIELDS = ("use_deterministic_algorithms", "warn_only",
                      "cudnn_benchmark", "cudnn_deterministic",
                      "cublas_workspace_config", "tf32_matmul_allowed",
                      "tf32_cudnn_allowed", "seed")


def determinism_problems(settings) -> list[str]:
    """Is this a complete, strict determinism record?"""
    if not isinstance(settings, dict):
        return [f"the determinism record is a {type(settings).__name__}, "
                "not an object"]
    problems = [f"the determinism record has no {f!r}"
                for f in DETERMINISM_FIELDS if f not in settings]
    if problems:
        return problems
    if settings["use_deterministic_algorithms"] is not True:
        problems.append("deterministic algorithms were not in effect")
    if settings["warn_only"] is not False:
        problems.append("warn_only was in effect, so a missing deterministic "
                        "kernel would have been a warning rather than a stop")
    if settings["cudnn_benchmark"] is not False:
        problems.append("cudnn.benchmark was on; it picks kernels by timing, "
                        "which is a run-to-run choice")
    if settings["cudnn_deterministic"] is not True:
        problems.append("cudnn.deterministic was off")
    if settings["cublas_workspace_config"] not in CUBLAS_WORKSPACE_VALUES:
        problems.append(
            f"cublas_workspace_config is "
            f"{settings['cublas_workspace_config']!r}")
    if not isinstance(settings["seed"], int) or isinstance(settings["seed"], bool):
        problems.append(f"seed is {settings['seed']!r}, not an integer")
    return problems


# ---------------------------------------------------------------------------
# Reading the machine
# ---------------------------------------------------------------------------

def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def probe(*, torch_mod=..., proc_version=..., meminfo=..., env=...,
          os_system=...) -> dict:
    """One reading of everything the preflight judges, and nothing else.

    Deliberately narrow, for the reason report 15's sampler is narrow: the
    gate needs to know whether this is the right machine and whether it is
    ready, not what else is running on it or where anything lives. No path, no
    hostname, no account, no process name is read or reported.

    Every argument defaults to reading the real machine and can be injected.
    A value that cannot be read comes back ``None`` -- never a guess, and
    never a default that happens to look healthy.
    """
    if torch_mod is ...:
        torch_mod = _try(lambda: __import__("torch"))
    if proc_version is ...:
        proc_version = _read("/proc/version")
    if meminfo is ...:
        meminfo = _read("/proc/meminfo")
    if env is ...:
        import os

        env = dict(os.environ)
    if os_system is ...:
        os_system = _try(_platform.system)

    out: dict = {
        "os_system": os_system or None,
        "wsl2": None,
        "wsl_evidence": None,
        "torch_version": None,
        "torch_cuda_build": None,
        "cuda_available": None,
        "device_count": None,
        "gpu_name": None,
        "vram_total_gb": None,
        "system_ram_gb": None,
        "offline_env": {k: (env or {}).get(k) for k in REQUIRED_OFFLINE_ENV},
        # The allocator configuration this process inherited, exactly as it
        # was found. Judged below; never written.
        "alloc_env": {k: (env or {}).get(k)
                      for k in (ALLOC_ENV_PRIMARY, ALLOC_ENV_ALIAS)},
        "cublas_workspace_config": (env or {}).get(CUBLAS_WORKSPACE_ENV),
        "allocator_backend": None,
    }

    if proc_version is not None:
        # WSL2 kernels are built by Microsoft and say so in the release
        # string. This is the reading every WSL detection uses; what matters
        # here is that *unreadable* stays None below rather than becoming
        # False, because "not WSL" and "could not tell" are different answers.
        out["wsl2"] = "microsoft" in proc_version.lower()
        out["wsl_evidence"] = ("kernel release names a Microsoft build"
                               if out["wsl2"] else
                               "kernel release does not name a Microsoft build")

    if meminfo:
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                kb = _try(lambda: int(line.split()[1]))
                if kb is not None:
                    out["system_ram_gb"] = round(kb / (1024 ** 2), 2)
                break

    if torch_mod is not None:
        out["torch_version"] = _try(lambda: str(torch_mod.__version__))
        out["torch_cuda_build"] = _try(
            lambda: getattr(torch_mod.version, "cuda", None))
        out["cuda_available"] = _try(lambda: bool(torch_mod.cuda.is_available()))
        if out["cuda_available"]:
            out["device_count"] = _try(lambda: int(torch_mod.cuda.device_count()))
            out["gpu_name"] = _try(lambda: str(torch_mod.cuda.get_device_name(0)))
            total = _try(
                lambda: torch_mod.cuda.get_device_properties(0).total_memory)
            if isinstance(total, (int, float)):
                out["vram_total_gb"] = round(total / (1024 ** 3), 2)
            out["allocator_backend"] = _try(
                lambda: str(torch_mod.cuda.get_allocator_backend()))
    return out


# ---------------------------------------------------------------------------
# Judging it
# ---------------------------------------------------------------------------

def _check(passed: bool, detail: str) -> dict:
    return {"passed": bool(passed), "detail": detail}


def _unreadable(what: str) -> dict:
    return _check(False, f"{what} could not be read, and a check that cannot "
                         "be evaluated has not been satisfied")


def execution_node_problems(pack_dir) -> list[str]:
    """Why this directory would make the node a development source.

    A pack is a copy that is executed, not a repository. If it has version
    control in it
    the node can commit, branch and edit -- and the moment it does, there are
    two versions of this project and no way to say which produced a number.
    """
    pack_dir = Path(pack_dir)
    problems = []
    for name in (".git", ".hg", ".svn"):
        if (pack_dir / name).exists():
            problems.append(
                f"{name} is present in the pack directory. A pack is a "
                "copy that is executed, not a repository that is developed in: "
                "with version control here the node can diverge from the Mac "
                "and nothing afterwards can say which tree produced a result.")
    return problems


def preflight(*, probe: dict, pack_dir, expected_pack_digest,
              expected_dependency_digest,
              data_root=None,
              requested_device: str = REQUIRED_DEVICE,
              requested_dtype: str = REQUIRED_DTYPE,
              requested_quantization: str = REQUIRED_QUANTIZATION,
              verifier=None, dependency_checker=None) -> dict:
    """Judge one probe against this node's contract. Fails closed throughout.

    ``expected_pack_digest`` and ``expected_dependency_digest`` have no
    defaults, deliberately. Both are values the build machine printed, carried
    here by routes the things they describe did not travel, and both check
    something the node cannot establish by reading what it already has: every
    digest inside a manifest is computed from that manifest, and a cache full
    of correctly-named files says nothing about what is in them. A trust check
    with a default is a trust check somebody forgets to pass, and it fails
    open when they do.

    They stay two values rather than one. The pack and the dependencies are
    rebuilt on different occasions and travel separately, and folding them
    into a single number would mean re-carrying both whenever either moved.

    Returns report 15's gate shape -- ``{"passed", "checks", "failed"}`` --
    because it answers report 15's question about a different machine, and two
    shapes for one answer is two things for a caller to get right.
    """
    p = probe or {}
    verifier = verifier or pack.verify
    checks: dict[str, dict] = {}

    system, wsl2 = p.get("os_system"), p.get("wsl2")
    if system is None:
        checks["platform"] = _unreadable("the operating system")
    elif system != "Linux":
        # Checked before the kernel string, because on a Mac there is no
        # /proc/version to read and "the kernel release could not be read" is
        # a confusing way to say "this is the wrong machine".
        checks["platform"] = _check(
            False, f"this is {system}, and the node runs inside WSL2 Ubuntu. "
                   "The Mac is the development source and does not execute "
                   "packs; the execution node does not develop.")
    elif wsl2 is None:
        checks["platform"] = _unreadable("the kernel release")
    elif not wsl2:
        checks["platform"] = _check(
            False, "this is Linux but not WSL2: " + str(
                p.get("wsl_evidence") or "the kernel release does not name a "
                                         "Microsoft build"))
    else:
        checks["platform"] = _check(True, "Linux under WSL2, as declared")

    build = p.get("torch_cuda_build")
    checks["torch_cuda_build"] = (
        _check(True, f"torch is built against CUDA {build}") if build
        else _check(False, "this torch has no CUDA build (torch.version.cuda "
                           "is unset), so it can only run on the CPU. There "
                           "is no CPU fallback here."))

    available = p.get("cuda_available")
    if available is None:
        checks["cuda"] = _unreadable("CUDA availability")
    else:
        checks["cuda"] = _check(
            available, "CUDA is available" if available else
            "CUDA is not available. The run does not fall back to the CPU: a "
            "run on another device is a different experiment.")

    count = p.get("device_count")
    if count is None:
        checks["device_count"] = _unreadable("the CUDA device count")
    else:
        checks["device_count"] = _check(
            count >= 1, f"{count} CUDA device(s) visible")

    name = p.get("gpu_name")
    if name is None:
        checks["gpu_model"] = _unreadable("the CUDA device name")
    else:
        checks["gpu_model"] = _check(
            EXPECTED_GPU in name,
            f"device reports {name!r}" if EXPECTED_GPU in name else
            f"device reports {name!r}, which is not the declared "
            f"{NODE_SPEC['gpu']!r}. A run on different hardware is not the "
            "run that was planned.")

    vram = p.get("vram_total_gb")
    if vram is None:
        checks["vram"] = _unreadable("total VRAM")
    else:
        checks["vram"] = _check(
            vram >= MIN_VRAM_GB,
            f"{vram} GB VRAM, floor {MIN_VRAM_GB} GB")

    ram = p.get("system_ram_gb")
    if ram is None:
        checks["system_ram"] = _unreadable("total system memory")
    else:
        checks["system_ram"] = _check(
            ram >= MIN_SYSTEM_RAM_GB,
            f"{ram} GB system memory, floor {MIN_SYSTEM_RAM_GB} GB")

    checks["requested_device"] = _check(
        requested_device == REQUIRED_DEVICE,
        f"device {requested_device!r}" if requested_device == REQUIRED_DEVICE
        else f"device {requested_device!r} was requested; this node runs "
             f"{REQUIRED_DEVICE!r} and nothing else")

    checks["dtype"] = _check(
        requested_dtype == REQUIRED_DTYPE,
        f"dtype {requested_dtype!r}" if requested_dtype == REQUIRED_DTYPE
        else f"dtype {requested_dtype!r} was requested, not the frozen "
             f"{REQUIRED_DTYPE!r}. Changing precision changes the run.")

    checks["quantization"] = _check(
        requested_quantization == REQUIRED_QUANTIZATION,
        "no quantization" if requested_quantization == REQUIRED_QUANTIZATION
        else f"quantization {requested_quantization!r} was requested; the "
             "frozen configuration is unquantized and switching is not a "
             "memory tweak, it is a different model")

    offline = p.get("offline_env") or {}
    missing = sorted(k for k, v in REQUIRED_OFFLINE_ENV.items()
                     if offline.get(k) != v)
    checks["offline"] = _check(
        not missing,
        "the offline environment is pinned" if not missing else
        f"{missing} are not pinned; the run must not be able to reach the hub")

    node_problems = execution_node_problems(pack_dir)
    checks["execution_node_only"] = _check(
        not node_problems,
        "the pack directory is a copy that is executed, not a repository"
        if not node_problems else " ".join(node_problems))

    pack_problems = verifier(pack_dir, data_root=data_root)
    checks["pack"] = _check(
        not pack_problems,
        "the pack verifies file by file against its manifest"
        if not pack_problems else
        f"{len(pack_problems)} problem(s), first: {pack_problems[0]}")

    digest_problems = pack.trusted_digest_problems(pack_dir,
                                                   expected_pack_digest)
    checks["expected_digest"] = _check(
        not digest_problems,
        "the pack matches the digest carried from the build machine"
        if not digest_problems else digest_problems[0])

    alloc_config, alloc_problems = allocator_config_from_env(
        p.get("alloc_env") or {})
    checks["allocator_config"] = _check(
        not alloc_problems,
        f"allocator config {alloc_config!r} was inherited from the "
        "environment" if not alloc_problems else alloc_problems[0])

    dependency = _dependency_check(dependency_checker)
    checks["dependencies"] = dependency["check"]

    # Recomputed here, from this machine's own reading, and compared against
    # the carried value. Deliberately a separate verdict from ``dependencies``:
    # "every pinned file is present" and "every pinned file is the one the
    # Mac had" are different facts, and a single check reporting both would
    # make a content drift read like a missing file.
    recomputed = dependency_digest(dependency["evidence"])
    digest_problems = pack.expected_digest_problems(
        expected_dependency_digest, what="dependency digest")
    if not digest_problems and not dependency["check"]["passed"]:
        digest_problems = [
            "the dependencies could not be resolved, so there is nothing to "
            "compare against the carried dependency digest"]
    elif not digest_problems and recomputed != expected_dependency_digest:
        digest_problems = [
            f"this machine's dependencies digest to {recomputed[:16]}..., "
            f"which is not the {str(expected_dependency_digest)[:16]}... "
            "carried from the build machine. Every pinned file resolved, so "
            "this is not an absence: it is a different file under the same "
            "name."]
    checks["dependency_digest"] = _check(
        not digest_problems,
        f"this machine's dependencies digest to {recomputed[:16]}..., "
        "matching the value carried from the build machine"
        if not digest_problems else digest_problems[0])

    failed = sorted(k for k, v in checks.items() if not v["passed"])
    return {"passed": not failed, "checks": checks, "failed": failed,
            "pack_problems": pack_problems,
            "dependency_evidence": dependency["evidence"],
            # The normalized config, for the plan to freeze and every later
            # stage to compare against. The backend is reported beside it and
            # deliberately cannot stand in for it: "native" is a fact about
            # which allocator, not about how it was configured, and the two
            # runs that differ by 8 GB of reserved memory both say "native".
            "allocator_config": alloc_config,
            "allocator_backend": p.get("allocator_backend"),
            # Printed by the CLI so the operator can compare by eye against
            # what the Mac reported.
            "dependency_digest": recomputed,
            "node_spec": dict(NODE_SPEC)}


def recomputed_dependency_digest(*, checker=None):
    """What this machine's cache digests to, or ``None`` if it cannot be read.

    ``None`` rather than a raised exception or a placeholder digest: the
    callers compare it against a frozen value, and an unreadable cache has to
    come out *unequal* to every real one rather than crashing the comparison
    or accidentally matching.
    """
    dependency = _dependency_check(checker)
    if not dependency["check"]["passed"]:
        return None
    return dependency_digest(dependency["evidence"])


def preflight_dependency_problems(expected, *, checker=None) -> list[str]:
    """The dependency binding, as sentences rather than as a gate verdict.

    :func:`preflight` phrases this as two checks because an operator reading a
    preflight wants "present" and "the same" answered separately. A runner
    wants a list of reasons to refuse. Same reading, same digest, one
    implementation -- so the two can never come to different conclusions about
    the same cache.
    """
    problems = pack.expected_digest_problems(expected,
                                             what="dependency digest")
    if problems:
        return problems
    dependency = _dependency_check(checker)
    if not dependency["check"]["passed"]:
        return [dependency["check"]["detail"]]
    recomputed = dependency_digest(dependency["evidence"])
    if recomputed != expected:
        return [f"this machine's dependencies digest to {recomputed[:16]}..., "
                f"which is not the {str(expected)[:16]}... carried from the "
                "build machine. Every pinned file resolved, so this is not an "
                "absence: it is a different file under the same name."]
    return []


def _dependency_check(checker=None) -> dict:
    """Can every pinned dependency be resolved from this machine's cache?

    Fails closed in all three directions a checker can disappoint: it raised,
    it returned something this reader cannot judge, or it reported problems.
    An unavailable answer is not a good one -- and the whole point of running
    it here is that gate 8 is a terrible place to learn the tokenizer is
    absent.
    """
    checker = checker or DEFAULT_DEPENDENCY_CHECKER
    try:
        result = checker()
    except Exception as exc:
        return {"check": _check(False, _portable(
            f"the dependency preflight could not run: {type(exc).__name__}: "
            f"{exc}", 300)), "evidence": None}
    if not isinstance(result, dict) or "ok" not in result \
            or not isinstance(result.get("problems"), list):
        return {"check": _check(False,
                                "the dependency preflight returned something "
                                "this reader cannot judge, and an answer that "
                                "cannot be read is not a pass"),
                "evidence": None}
    evidence = result.get("evidence")
    if result["ok"] is not True:
        detail = "; ".join(str(x) for x in result["problems"])
        return {"check": _check(False, _portable(
            detail or "the dependency preflight reported failure without "
                      "saying why", 400)), "evidence": evidence}
    return {"check": _check(
        True, "every pinned tokenizer, base model and adapter file resolves "
              "from this machine's local cache, and the instruction pool is "
              "present -- with no network call, no tensor loaded and no "
              "device initialised"), "evidence": evidence}
