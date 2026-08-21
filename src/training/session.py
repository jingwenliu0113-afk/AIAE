"""One measured run per boot: the state that has to survive a restart.

Report 15's first attempt stopped after one run. The recovery gate had been
calibrated on a machine that had never loaded the model in that boot, and once
a run had happened the swap reading settled well above the calibrated band and
stayed there. Waiting longer inside the same boot did not bring it back inside
the observation window, so the design that assumed it would is the design that
has to change: spend one boot per measured run, and sample the idle baseline
once.

That makes the experiment something that gets interrupted by definition -- the
machine restarts between runs, possibly days apart, and a crash is a normal
outcome rather than an exception -- which is what this module holds:

* **an exclusive lock**, so two invocations cannot both decide they are the
  next run
* **a publish primitive that cannot clobber**, built on ``os.link`` rather than
  on looking first and renaming afterwards
* **an identifier for the boot that never stores the boot's own name**, so a
  repeat can be refused without the machine's boot UUID being written down
* **an exact copy of the source the plan was made against**, so four runs
  spread over four boots can be shown to have executed the same code rather
  than assumed to have
* **an append-only journal of immutable events**, so a run that started and
  never reported leaves a record saying exactly that

Nothing here restarts anything. A restart is the operator's to perform, and a
tool that reboots the machine it is measuring is a tool that has changed the
thing it was measuring.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

EVENT_DIR = "events"

#: A gate that never released. No child was spawned, so the boot is untouched.
EVENT_GATE_ATTEMPT = "gate_attempt"
#: The gate released, and then something else stopped the run before any child
#: existed -- in practice, the source moved while the gate was polling. It is
#: its own event rather than a gate attempt because its gate *passed*, and a
#: record that says otherwise is a record that lies about the machine. Like a
#: gate attempt it spends no boot: nothing was measured.
EVENT_PRE_SPAWN_ABORT = "pre_spawn_abort"
#: Written after the gate releases and **before** the child is spawned. This is
#: the event that consumes the boot: if the parent dies in the next second, the
#: journal still says a measurement began here.
EVENT_STARTED = "measurement_started"
#: Written once the child is over, whatever the outcome, and pinned to the
#: started event by that file's digest.
EVENT_FINISHED = "measurement_finished"

EVENT_KINDS = (EVENT_GATE_ATTEMPT, EVENT_PRE_SPAWN_ABORT, EVENT_STARTED,
               EVENT_FINISHED)

#: Events after which the same run may be attempted again. Neither spawned a
#: child, so neither spent the boot.
EVENT_RETRYABLE = (EVENT_GATE_ATTEMPT, EVENT_PRE_SPAWN_ABORT)

EVENT_FILE_RE = re.compile(
    r"^(?P<index>\d{2,})-(?P<run_id>[A-Za-z0-9_]+)-(?P<event>[a-z_]+)\.json$")


# --------------------------------------------------------------------------
# publishing: atomic, and unable to clobber
# --------------------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _publish(tmp: str, path: Path) -> None:
    """Give the finished content its final name, or fail because it is taken.

    ``os.link`` is the primitive that does both at once: it either creates the
    name or raises ``EEXIST``, with no window in between. Asking ``exists()``
    and then renaming has that window -- two processes can both see nothing and
    both rename, and the second silently replaces the first.
    """
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise SystemExit(
            f"refusing to overwrite {path}: a record for this step already "
            "exists, and replacing it would silently swap one measurement for "
            "another. Move or remove it deliberately if that is the intent.")
    except OSError as exc:
        raise SystemExit(
            f"cannot publish {path} atomically ({exc}). This filesystem does "
            "not support hard links, and every other way of creating that name "
            "can overwrite something. Refusing rather than racing.")


def write_once_json(path, obj) -> str:
    """Write JSON to a name that must not already exist, atomically.

    The content lands in a uniquely-named temporary file in the same directory
    -- never a shared ``.tmp`` beside the target, which two processes would
    write over each other before either published -- is flushed to disk, and is
    then linked into place.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        _publish(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
    _fsync_dir(path.parent)
    return sha256_file(path)


def copy_once(src, dst) -> str:
    """The same discipline for a file copy, so a snapshot cannot be rewritten."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent),
                               prefix=f".{dst.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as out, src.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        _publish(tmp, dst)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
    _fsync_dir(dst.parent)
    return sha256_file(dst)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# exclusive access
# --------------------------------------------------------------------------

@contextlib.contextmanager
def exclusive_lock(path, *, description: str):
    """Hold the session against every other process, or refuse to start.

    Non-blocking on purpose. Two invocations of the next run must not queue up
    behind each other and then both proceed: by the time the second acquired
    the lock the first would have consumed the boot, and the second would be
    working from state it read before that happened.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(
                f"another process is already working on {description}. Only "
                "one may read the journal, decide what runs next and spawn it, "
                "because two that read the same state would both decide the "
                "same thing. Wait for it to finish.")
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# --------------------------------------------------------------------------
# boot identity, without recording the boot's name
# --------------------------------------------------------------------------

def _sysctl(name: str) -> str | None:
    try:
        r = subprocess.run(["sysctl", "-n", name], capture_output=True,
                           text=True, timeout=10)
        out = r.stdout.strip()
        return out if r.returncode == 0 and out else None
    except Exception:
        return None


def _fingerprint(experiment_id: str, raw: str) -> str:
    return hashlib.sha256(
        f"15_mps_order:{experiment_id}\x00{raw}".encode()).hexdigest()[:32]


def boot_identity(experiment_id: str, sysctl=_sysctl) -> dict:
    """Name this boot to *this experiment* only, and never write down the boot.

    The raw value never leaves this function. What the journal needs is whether
    two runs happened in the same boot, which a hash answers; the boot session
    UUID itself is a machine identifier, it is stable for as long as the
    machine stays up, and there is no reason for it to end up in a report that
    gets read by someone else. Hashing with the experiment id as domain
    separation also means fingerprints from two experiments cannot be lined up
    against each other.

    If neither source can be read the answer is ``None`` and the caller must
    refuse: one measured run per boot cannot be enforced against a boot that
    cannot be told apart from the last one.
    """
    raw = sysctl("kern.bootsessionuuid")
    if raw:
        return {"boot_fingerprint": _fingerprint(experiment_id, raw),
                "source": "kern.bootsessionuuid", "detected_at": now_iso(),
                "reason": None}
    raw = sysctl("kern.boottime")
    if raw:
        m = re.search(r"sec\s*=\s*(\d+)", raw)
        seconds = m.group(1) if m else raw
        return {"boot_fingerprint": _fingerprint(experiment_id,
                                                 f"boottime:{seconds}"),
                "source": "kern.boottime", "detected_at": now_iso(),
                "reason": None}
    return {"boot_fingerprint": None, "source": None, "detected_at": now_iso(),
            "reason": ("neither kern.bootsessionuuid nor kern.boottime could "
                       "be read, so this boot cannot be told apart from the "
                       "previous one")}


# --------------------------------------------------------------------------
# source snapshot
# --------------------------------------------------------------------------

def _snapshot_name(rel: str) -> str:
    return rel.replace("/", "__")


def snapshot_sources(root, files, dest) -> dict:
    """Copy the source the plan was made against, and digest every file.

    The digest alone would say the code changed; it would not say what it used
    to be. Between the first run and the fourth the working tree stays open for
    editing, and the honest answer to "was this the same code" is a copy of the
    code, kept next to the plan.
    """
    root, dest = Path(root), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for rel in files:
        src = root / rel
        if not src.exists():
            raise SystemExit(f"cannot snapshot {rel}: it does not exist")
        copy_once(src, dest / _snapshot_name(rel))
        manifest[rel] = {"sha256": sha256_file(src),
                         "bytes": src.stat().st_size,
                         "snapshot_name": _snapshot_name(rel)}
    return {"created_at": now_iso(), "root_relative": True, "files": manifest}


def manifest_digest(manifest: dict) -> str:
    """One value standing for the whole snapshot, for a child to check against."""
    files = (manifest or {}).get("files") or {}
    return hashlib.sha256(json.dumps(
        {k: v.get("sha256") for k, v in files.items()},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_sources(root, manifest: dict, dest, *,
                   check_working_tree: bool = True) -> list[str]:
    """Check the snapshot -- and, before a run, the live tree -- file by file.

    Both sides are checked before a child is spawned: a digest that matches the
    live file proves nothing if the snapshot it is compared against was
    rewritten at the same time.

    ``check_working_tree=False`` is for reading a finished session back later.
    The snapshot is what makes the run durable; requiring the working tree to
    stay frozen for ever would mean a session could only be replayed by a tree
    that had stopped being worked in, which is not a property any repository
    has.
    """
    root, dest = Path(root), Path(dest)
    problems: list[str] = []
    files = (manifest or {}).get("files")
    if not files:
        return ["source manifest records no files"]
    for rel, entry in sorted(files.items()):
        expected = entry.get("sha256")
        if not expected:
            problems.append(f"{rel}: manifest records no digest")
            continue
        if check_working_tree:
            live = root / rel
            if not live.exists():
                problems.append(f"{rel}: missing from the working tree")
            elif sha256_file(live) != expected:
                problems.append(
                    f"{rel}: has changed since the plan was made "
                    f"(expected {expected[:12]}..., found "
                    f"{sha256_file(live)[:12]}...)")
        name = entry.get("snapshot_name") or _snapshot_name(rel)
        copy = dest / name
        if not copy.exists():
            problems.append(f"{rel}: its snapshot copy is missing")
        elif sha256_file(copy) != expected:
            problems.append(f"{rel}: its snapshot copy does not match the "
                            "manifest digest")
    return problems


# --------------------------------------------------------------------------
# append-only journal of immutable events
# --------------------------------------------------------------------------

def event_file_name(index: int, run_id: str, event: str) -> str:
    return f"{index:02d}-{run_id}-{event}.json"


def event_path(session_dir, index: int, run_id: str, event: str) -> Path:
    return (Path(session_dir) / EVENT_DIR
            / event_file_name(index, run_id, event))


def append_event(session_dir, index: int, run_id: str, event: str,
                 payload: dict) -> str:
    """One file per event, published once, never rewritten."""
    if event not in EVENT_KINDS:
        raise SystemExit(f"unknown journal event {event!r}")
    return write_once_json(event_path(session_dir, index, run_id, event),
                           payload)


def read_events(session_dir) -> list[dict]:
    """Every event on disk, in filename order, with its own digest.

    The digest is read here rather than trusted from inside the file: a
    finished event points back at the started event by digest, and that link is
    only worth anything if the digest is computed over what is actually there.
    """
    d = Path(session_dir) / EVENT_DIR
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            body = json.loads(p.read_text())
        except Exception as exc:
            body = {"__unreadable__": str(exc)}
        out.append({"file_name": p.name, "file_sha256": sha256_file(p),
                    "body": body})
    return out
