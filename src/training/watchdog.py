"""Out-of-process safety enforcement for report 16's long runs.

Report 15 measured 200 rows and finished in five minutes; a stall could be
noticed afterwards. Report 16 measures up to 2,000 rows -- 45 minutes for the
treatment, hours for a control -- and the thing being protected is the
machine, not the measurement. That changes what "safety check" has to mean.

The obvious design, probing the OS between rows inside the child, cannot work:
a foreground probe only runs at row boundaries, and the old 2,000-row record
had windows averaging 95 s/row, so the gap between two checks is whatever the
slowest row happens to be. A background thread inside the child would fix the
cadence and ruin the measurement, which is the one thing this design exists to
protect.

So enforcement lives here, in a separate process:

* the child records what it sees every five rows, for analysis and replay, and
  appends a progress line every row so somebody outside can see it moving;
* this watchdog polls the OS every five seconds, judges every threshold, and
  can stop the child;
* the parent watches both, live, and stops the run if this process dies or
  stops sending heartbeats -- including the case where the first heartbeat
  never arrives at all.

Two rules run through the whole file. Durations cross process boundaries;
instants do not, because two monotonic clocks have different origins. And a
failure of *this* machinery is never a measurement result: it is a tool
failure, and the difference is enforced by two closed sets below.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import secrets
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Two closed sets. One says something about the machine, the other says
# something about us. Writing a tool failure down as a measurement result
# would turn a bug in this file into a finding about MPS.
# ---------------------------------------------------------------------------

SAFETY_REASONS = (
    "swap_used_gb",
    "swap_growth_gb_per_probe",
    "memory_pressure_percent_free",
    "free_plus_inactive_gb",
    "slow_row_seconds",
    "max_seconds",
    "process_max_seconds",
)

TOOL_FAILURES = (
    "probe_unavailable",
    "watchdog_died",
    "heartbeat_gap",
    "heartbeat_never_arrived",
    "identity_mismatch",
    "stop_request_rejected",
    "no_report",
    "nonzero_exit",
    "timed_out",
    "spawn_failed",
    "armed_timeout",
    "progress_stalled",
    "progress_invalid",
    #: The watchdog log has no valid terminal record, so nobody can say the
    #: watchdog was still there when the child finished.
    "watchdog_unsealed",
    #: The child report does not replay: schema, sums, metrics or the
    #: stopped_early block do not survive recomputation.
    "report_invalid",
)

#: The last record a watchdog writes: it saw the child go. A log without one
#: is a log that never says how the run ended, and a run whose ending nobody
#: recorded cannot be a completed run (§4.5.3).
CHILD_EXIT_OBSERVED = "child_exit_observed"

ACTIONS = (None, "poll", "sigterm", "sigkill", "identity_mismatch",
           CHILD_EXIT_OBSERVED)


def is_safety_reason(reason) -> bool:
    return isinstance(reason, str) and reason in SAFETY_REASONS


def is_tool_failure(reason) -> bool:
    return isinstance(reason, str) and reason in TOOL_FAILURES


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SafetySpec:
    """The frozen safety limits.

    These protect the machine. They are deliberately not chosen so that any
    particular arm can finish: at 2,000 rows a control arm is expected to trip
    ``swap_used_gb``, and that is the accepted trade rather than a flaw.
    """

    swap_used_gb_max: float = 12.0
    swap_growth_gb_per_probe_max: float = 2.0
    memory_pressure_percent_free_min: int = 5
    memory_pressure_consecutive_probes: int = 5
    free_plus_inactive_gb_min: float = 0.30
    free_plus_inactive_consecutive_probes: int = 5
    slow_row_seconds: float = 120.0
    slow_row_streak: int = 3
    max_seconds: float = 28800.0
    process_max_seconds: float = 30600.0

    poll_seconds: float = 5.0
    poll_max_gap_seconds: float = 30.0
    probe_failure_streak_max: int = 3
    heartbeat_gap_seconds: float = 30.0
    first_heartbeat_deadline_seconds: float = 30.0
    parent_supervision_interval_seconds: float = 1.0
    grace_seconds: float = 120.0
    ready_timeout_seconds: float = 30.0
    armed_timeout_seconds: float = 30.0
    #: How many consecutive polls may see the SAME row before we call the
    #: child wedged. Progress that never arrives at all is a different case
    #: -- the child may simply still be loading -- and is bounded by
    #: process_max_seconds instead.
    progress_stall_polls: int = 60

    def as_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @staticmethod
    def from_dict(d: dict) -> "SafetySpec":
        fields = SafetySpec.__dataclass_fields__
        unknown = set(d) - set(fields)
        if unknown:
            raise ValueError(f"unknown safety fields {sorted(unknown)}")
        return SafetySpec(**{k: v for k, v in d.items()})


# ---------------------------------------------------------------------------
# Identity. A bare PID is not an identity: PIDs are reused, and a watchdog that
# signals a reused PID kills somebody else's process.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChildIdentity:
    pid: int
    pgid: int
    nonce: str
    start_identity: str

    def as_dict(self) -> dict:
        return {"child_pid": self.pid, "child_pgid": self.pgid,
                "nonce": self.nonce, "child_start_identity": self.start_identity}

    @staticmethod
    def from_dict(d: dict) -> "ChildIdentity":
        missing = [k for k in ("child_pid", "child_pgid", "nonce",
                               "child_start_identity") if k not in d]
        if missing:
            raise ValueError(f"child identity is missing {missing}")
        return ChildIdentity(int(d["child_pid"]), int(d["child_pgid"]),
                             str(d["nonce"]), str(d["child_start_identity"]))


def new_nonce() -> str:
    """One per launch, never reused. Ties a stop request to this run only."""
    return secrets.token_hex(16)


def process_start_identity(pid: int, *, runner=None) -> str | None:
    """The process's start time, so 'same PID' and 'same process' differ.

    ``None`` means the process is gone or unreadable, which counts as a
    mismatch everywhere below: we never signal on a guess.
    """
    runner = runner or (lambda cmd: subprocess.run(
        cmd, capture_output=True, text=True, timeout=5))
    try:
        out = runner(["ps", "-p", str(pid), "-o", "lstart="])
    except Exception:  # noqa: BLE001 - an unreadable ps is a mismatch
        return None
    if getattr(out, "returncode", 1) != 0:
        return None
    text = (out.stdout or "").strip()
    return text or None


def observed_pgid(pid: int, *, getpgid=os.getpgid) -> int | None:
    try:
        return getpgid(pid)
    except (ProcessLookupError, PermissionError, OSError):
        return None


def identity_check(identity: ChildIdentity, *, nonce: str,
                   observed_start=None, observed_pgid_value=None) -> dict:
    """All four fields, every time, immediately before signalling.

    Checking only the start identity was the earlier gap: a process can keep
    its start time and still be the wrong target if the group or the launch
    nonce disagree, and a stale nonce is exactly how an old launch's stop
    request could reach a new child.
    """
    problems = []
    if nonce != identity.nonce:
        problems.append("nonce does not match this launch")
    if observed_start is None:
        problems.append("the process start identity could not be read")
    elif observed_start != identity.start_identity:
        problems.append("the process start identity differs: the PID was reused")
    if observed_pgid_value is None:
        problems.append("the process group could not be read")
    elif int(observed_pgid_value) != identity.pgid:
        problems.append("the process group differs from the one we launched")
    return {"ok": not problems, "problems": problems}


def write_launch_record(directory, *, prefix: str, identity: ChildIdentity,
                        experiment_id: str, run_id: str) -> str:
    """The parent's independent copy of who the child is."""
    body = {"experiment_id": experiment_id, "run_id": run_id,
            **identity.as_dict()}
    text = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=1)
    fd = os.open(Path(directory) / f"{prefix}.launch.json",
                 os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_launch_nonce(directory, prefix: str) -> str | None:
    path = Path(directory) / f"{prefix}.launch.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("nonce")
    except (json.JSONDecodeError, OSError):
        return None


def _signal_pgid(sig: str, pgid: int, *, killpg=os.killpg) -> None:
    killpg(pgid, signal.SIGTERM if sig == "SIGTERM" else signal.SIGKILL)


def reap(*procs, term_grace: float = 2.0, sleep=time.sleep,
         killpg=os.killpg) -> list[str]:
    """Terminate and collect processes, leaving nothing running behind.

    Called on every failure path after a child exists. An orphaned child that
    keeps training while the parent has already given up is the worst possible
    outcome of a tool failure: nobody is watching it and nobody will record
    what it did.
    """
    notes = []
    for proc in procs:
        if proc is None:
            continue
        if proc.poll() is not None:
            notes.append(f"pid {proc.pid} had already exited")
            continue
        try:
            killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        deadline = time.monotonic() + term_grace
        while proc.poll() is None and time.monotonic() < deadline:
            sleep(0.05)
        if proc.poll() is None:
            try:
                killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            notes.append(f"pid {proc.pid} needed SIGKILL")
        else:
            notes.append(f"pid {proc.pid} exited on SIGTERM")
        try:
            proc.wait(timeout=2)
        except Exception:  # noqa: BLE001
            notes.append(f"pid {proc.pid} could not be reaped")
    return notes


# ---------------------------------------------------------------------------
# The durable log. This is the enforcement record, so it is held to the same
# standard as the journal: created exclusively, chained, fsynced, sealed.
# ---------------------------------------------------------------------------


def _digest_line(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


class WatchdogLog:
    """Append-only JSONL with a per-record hash chain.

    ``O_CREAT | O_EXCL`` on the final path, deliberately: no temp-then-rename,
    no truncation. If the destination is already there the launch fails rather
    than quietly overwriting an earlier run's evidence.
    """

    def __init__(self, path):
        self.path = Path(path)
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        self._fh = os.fdopen(fd, "w", encoding="utf-8")
        self._seq = 0
        self._prev: str | None = None
        self._sealed = False
        self._identity: dict | None = None

    @property
    def seq(self) -> int:
        return self._seq

    def arm(self, identity: "ChildIdentity") -> None:
        """Stamp every record with the process this log is an account of.

        The launch record is the parent's account of who the child was and the
        child's report is its own; rewriting both together used to agree with
        itself, because the one party that was actually watching the process
        wrote nothing down about it. Now it does, on every enforcement record,
        so a stop order says which process group it was about.
        """
        if self._seq:
            raise RuntimeError("this log has already been written to; a log "
                               "cannot change which process it describes")
        self._identity = dict(identity.as_dict())

    def append(self, record: dict) -> dict:
        if self._sealed:
            raise RuntimeError("the watchdog log is sealed; appending after "
                               "the finished event is exactly what replay "
                               "refuses to accept")
        row = {"seq": self._seq, "prev_digest": self._prev,
               **(self._identity or {}), **record}
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        self._fh.write(line + "\n")
        # Both, every time. flush() moves it out of Python; fsync() moves it
        # out of the OS cache. A SIGKILL between them leaves a truncated last
        # line, which replay treats as corruption rather than guessing.
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._prev = _digest_line(line)
        self._seq += 1
        return row

    def seal(self) -> str:
        if not self._sealed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._sealed = True
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self._sealed:
            self.seal()
        return False


def replay_watchdog_log(path, *, expected_sha256: str | None = None) -> dict:
    """Structural replay: is this file intact and in order?

    Semantic replay -- did the right things happen, in the right order, for
    the right reasons -- is :func:`replay_watchdog_semantics` below.
    """
    path = Path(path)
    problems: list[str] = []
    if not path.exists():
        return {"problems": [f"{path} does not exist"], "records": []}

    raw = path.read_bytes()
    if expected_sha256 is not None:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256:
            problems.append(
                f"{path.name} sha256 is {actual[:16]}… but the finished event "
                f"recorded {expected_sha256[:16]}…; the file changed after it "
                "was sealed")

    text = raw.decode("utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    else:
        problems.append(f"{path.name} does not end with a newline, so its "
                        "last record is truncated")

    records, digests = [], []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name} line {i + 1} is not complete JSON "
                            f"({exc.msg}); a truncated line is never skipped")
            return {"problems": problems, "records": []}
        digests.append(_digest_line(line))

    seqs = [r.get("seq") for r in records]
    if any(not isinstance(s, int) or isinstance(s, bool) for s in seqs):
        problems.append("every record needs an integer seq")
        return {"problems": problems, "records": records}
    if len(set(seqs)) != len(seqs):
        problems.append(f"{path.name} has duplicate seq values")
    if sorted(seqs) != list(range(len(seqs))):
        problems.append(f"{path.name} seq values are not the complete range "
                        f"0..{len(seqs) - 1}; a record was inserted or deleted")
    if seqs != sorted(seqs):
        problems.append(f"{path.name} records are out of seq order")

    for i, rec in enumerate(records):
        want = None if i == 0 else digests[i - 1]
        if rec.get("prev_digest") != want:
            problems.append(
                f"{path.name} seq {rec.get('seq')} has prev_digest "
                f"{str(rec.get('prev_digest'))[:16]}… but the previous line "
                f"hashes to {str(want)[:16]}…; the chain is broken, so a "
                "record was inserted, removed or rewritten")
            break

    return {"problems": problems, "records": records}


#: The four fields that say which process a record is about. Named here
#: because :meth:`ChildIdentity.as_dict` writes them; report 16 imports the
#: same tuple so the launch record, the child report and this log cannot end
#: up comparing three different sets of keys.
IDENTITY_FIELDS = ("nonce", "child_pid", "child_pgid", "child_start_identity")

RECORD_FIELDS = ("seq", "prev_digest", "monotonic", "wall_clock",
                 "swap_used_gb", "memory_pressure_percent_free",
                 "free_plus_inactive_gb", "failed", "failure_streak",
                 "violations", "action", "progress", "heartbeat_seq",
                 *IDENTITY_FIELDS)

_NUMERIC = ("monotonic", "swap_used_gb", "free_plus_inactive_gb")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) \
        and math.isfinite(x)


def identity_agreement_problems(records: list[dict],
                                identity: dict) -> list[str]:
    """Every record names the same process, and it is the expected one.

    Checked per record rather than once: a log is a chain of enforcement
    decisions, and a spliced-in record that named a different process would
    otherwise ride along on the agreement of its neighbours.
    """
    problems: list[str] = []
    for field in IDENTITY_FIELDS:
        want = identity.get(field)
        missing = [r.get("seq") for r in records if field not in r]
        if missing:
            problems.append(
                f"the watchdog log records no {field} at seq {missing[:5]}: "
                "there is nothing to check the launch record and the child "
                "report against")
            continue
        seen = {r.get(field) for r in records}
        if len(seen) > 1:
            problems.append(f"the watchdog log gives {len(seen)} different "
                            f"values for {field}; it is an account of one "
                            "process")
        if want is not None and any(v != want for v in seen):
            problems.append(
                f"the watchdog log's {field} is {sorted(seen, key=repr)!r} but "
                f"the run it is filed under has {want!r}: the process that was "
                "watched is not the process that was measured")
    return problems


#: A watchdog that reached either of these has stopped supervising: it killed
#: the child, or it refused to signal because the process was not the one it
#: was armed against. Neither is a state you can then observe a clean exit
#: from, so a terminal record after one is a forgery.
ABANDONMENT_ACTIONS = ("sigkill", "identity_mismatch")


def watchdog_terminal_problems(records: list[dict], spec: SafetySpec, *,
                               identity: dict | None = None) -> list[str]:
    """Did this log say how the run ended -- once, at the end, in time?

    The hole this closes: a watchdog that died in its first second leaves a
    short log, the child runs on and exits before the parent's next look, and
    supervision reports nothing wrong because by then there is nothing to
    supervise. Every digest matches, every record is intact, and the run gets
    filed as completed on the strength of a log covering none of it.

    A log is not evidence of supervision because it exists. It is evidence
    because it reaches the end of the run, and the terminal record is what
    says it did.

    And a terminal record is not evidence because it is there. It has to say
    *whose* exit it saw: the same four fields as every record before it and as
    the run it is filed under. Counting the record and never reading it let a
    terminal record naming an entirely different PID close a run.
    """
    problems: list[str] = []
    if not records:
        return ["the watchdog log is empty, so nothing records how the run "
                "ended"]
    marked = [i for i, r in enumerate(records)
              if r.get("action") == CHILD_EXIT_OBSERVED]
    if not marked:
        return [f"the watchdog log has no {CHILD_EXIT_OBSERVED} record: "
                "nothing says the watchdog was still watching when the child "
                "finished, so this run was not supervised to the end"]
    if len(marked) > 1:
        problems.append(f"the watchdog log holds {len(marked)} "
                        f"{CHILD_EXIT_OBSERVED} records; a child exits once")
    index = marked[-1]
    if index != len(records) - 1:
        problems.append(
            f"{CHILD_EXIT_OBSERVED} sits at seq {records[index].get('seq')} "
            f"with {len(records) - 1 - index} record(s) after it; it is the "
            "last thing a watchdog writes before it seals the log")
    terminal = records[index]
    for field in IDENTITY_FIELDS:
        if field not in terminal:
            problems.append(f"the {CHILD_EXIT_OBSERVED} record carries no "
                            f"{field}, so it names no process")
            continue
        if identity is not None and terminal[field] != identity.get(field):
            problems.append(
                f"the {CHILD_EXIT_OBSERVED} record's {field} is "
                f"{terminal[field]!r}, but this run's is "
                f"{identity.get(field)!r}: the exit that was observed is not "
                "the exit of the process that was measured")
        disagreeing = sorted({r[field] for r in records[:index]
                              if field in r and r[field] != terminal[field]},
                             key=repr)
        if disagreeing:
            problems.append(
                f"the {CHILD_EXIT_OBSERVED} record's {field} is "
                f"{terminal[field]!r} but earlier records in the same log say "
                f"{disagreeing!r}")
    abandoned = [r.get("seq") for r in records[:index]
                 if r.get("action") in ABANDONMENT_ACTIONS]
    if abandoned:
        actions = sorted({r.get("action") for r in records[:index]
                          if r.get("action") in ABANDONMENT_ACTIONS})
        problems.append(
            f"the log records {actions} at seq {abandoned} before its "
            f"{CHILD_EXIT_OBSERVED}: a watchdog that killed the child, or that "
            "refused to signal because the process was not the one it was "
            "armed against, did not go on to watch it exit cleanly")
    if index == 0:
        problems.append(
            f"the {CHILD_EXIT_OBSERVED} record is the whole log: the watchdog "
            "recorded an ending without ever having polled")
        return problems
    before, after = records[index - 1].get("monotonic"), \
        terminal.get("monotonic")
    if not (_finite(before) and _finite(after)):
        problems.append(f"the {CHILD_EXIT_OBSERVED} record has no usable "
                        "monotonic, so the final supervision interval cannot "
                        "be measured")
    elif not 0.0 <= after - before <= spec.poll_max_gap_seconds:
        # Both sides. A one-sided `> max_gap` test accepted a terminal record
        # dated *before* the poll it follows, because a negative interval is
        # never greater than thirty.
        problems.append(
            f"the {CHILD_EXIT_OBSERVED} record's monotonic sits "
            f"{after - before:.3f}s from the previous poll, outside "
            f"[0, {spec.poll_max_gap_seconds}]: the exit is observed on the "
            "next poll, not before the one before it and not an hour later")
    return problems


def replay_watchdog_semantics(records: list[dict], spec: SafetySpec, *,
                              stop_request: dict | None = None,
                              nonce: str | None = None,
                              identity: dict | None = None,
                              claimed_reason: str | None = None,
                              claimed_stop: bool | None = None) -> dict:
    """Recompute every judgement the watchdog claims to have made.

    Structure alone is not enough. A well-formed log can still say the run was
    stopped for a reason its own numbers do not support, or record a clean
    poll on the very sample that should have tripped a threshold. Both
    directions are checked here: what should have stopped did, and what should
    not have stop did not.

    ``claimed_stop`` is the third direction, and it is deliberately
    three-valued. ``claimed_reason`` cannot carry it: a report that simply
    drops its ``stopped_early`` block has no reason left to compare, and
    "nothing to compare" is exactly how a stop gets hidden -- the log still
    records the trip and the SIGTERM, the report says the run finished
    normally, and Q1 comes out of the per-row times as though nothing had
    happened. So a caller that knows what the report claims says so:
    ``True`` it claims a stop, ``False`` it claims none, ``None`` this log is
    being replayed on its own and no report is being cross-checked.
    :func:`~src.training.longrun.replay_child` always passes a real boolean.
    """
    problems: list[str] = []
    if identity is not None and records:
        problems += identity_agreement_problems(records, identity)
    state = SafetyState(spec=spec)
    expected_reason: str | None = None
    expected_at: int | None = None
    sigterm_at_mono: float | None = None
    sigkill_at_mono: float | None = None
    observed_sigterm: dict | None = None
    last_mono: float | None = None
    heartbeats = 0

    for rec in records:
        extra = set(rec) - set(RECORD_FIELDS)
        if extra:
            problems.append(f"seq {rec.get('seq')} has unknown fields "
                            f"{sorted(extra)}")
        for key in _NUMERIC:
            if rec.get(key) is not None and not _finite(rec[key]):
                problems.append(f"seq {rec.get('seq')} has a non-finite "
                                f"{key}: {rec[key]!r}")
        if rec.get("action") not in ACTIONS:
            problems.append(f"seq {rec.get('seq')} has action "
                            f"{rec.get('action')!r}, outside {ACTIONS}")
        mono = rec.get("monotonic")
        if not _finite(mono):
            problems.append(f"seq {rec.get('seq')} has no usable monotonic")
            continue
        if last_mono is not None and mono < last_mono:
            problems.append(f"seq {rec.get('seq')} goes backwards in time "
                            f"({mono} after {last_mono})")
        last_mono = mono
        if rec.get("heartbeat_seq") is not None:
            heartbeats += 1

        if rec.get("action") == "poll":
            sample = None
            if rec.get("failed") is None:
                sample = {k: rec.get(k) for k in
                          ("swap_used_gb", "memory_pressure_percent_free",
                           "free_plus_inactive_gb")}
            verdict = state.observe(sample, mono, progress=rec.get("progress"))
            if rec.get("failure_streak") != state.failure_streak:
                problems.append(
                    f"seq {rec['seq']} records failure_streak "
                    f"{rec.get('failure_streak')} but replay computes "
                    f"{state.failure_streak}")
            want = state.violations()
            got = dict(rec.get("violations") or {})
            for key, value in want.items():
                if got.get(key, 0) != value:
                    problems.append(
                        f"seq {rec['seq']} records {key} streak "
                        f"{got.get(key)} but replay computes {value}")
            if verdict["reason"] is not None and expected_reason is None:
                expected_reason, expected_at = verdict["reason"], rec["seq"]
        elif rec.get("action") == "sigterm":
            observed_sigterm = rec
            sigterm_at_mono = mono
            if expected_reason is None:
                problems.append(
                    f"seq {rec['seq']} sent SIGTERM, but replaying the polls "
                    "up to that point trips no threshold: a run was stopped "
                    "that should not have been")
        elif rec.get("action") == "sigkill":
            sigkill_at_mono = mono
            if sigterm_at_mono is None:
                problems.append(f"seq {rec['seq']} sent SIGKILL without a "
                                "preceding SIGTERM")

    if expected_reason is not None and observed_sigterm is None:
        problems.append(
            f"replay trips {expected_reason!r} at seq {expected_at}, but the "
            "log records no SIGTERM: a run continued that should have stopped")

    if claimed_reason is not None and expected_reason is not None \
            and claimed_reason != expected_reason:
        problems.append(f"the run claims it stopped for {claimed_reason!r} but "
                        f"replay trips {expected_reason!r}")

    if claimed_stop is False:
        if expected_reason is not None:
            problems.append(
                f"replay trips {expected_reason!r} at seq {expected_at}, but "
                "the report records no stopped_early: a stop that happened is "
                "not the same thing as a stop that was recorded")
        if observed_sigterm is not None:
            problems.append(
                f"seq {observed_sigterm.get('seq')} sent SIGTERM, but the "
                "report records no stopped_early")

    if sigkill_at_mono is not None and sigterm_at_mono is not None:
        waited = sigkill_at_mono - sigterm_at_mono
        if waited + 1e-9 < spec.grace_seconds:
            problems.append(
                f"SIGKILL came {waited:.3f}s after SIGTERM, inside the "
                f"{spec.grace_seconds}s grace period")

    if stop_request is not None:
        if nonce is not None and stop_request.get("nonce") != nonce:
            problems.append("the stop request carries another launch's nonce")
        if expected_reason is not None and \
                stop_request.get("reason") != expected_reason:
            problems.append(
                f"the stop request says {stop_request.get('reason')!r} but "
                f"replay trips {expected_reason!r}")

    return {"problems": problems, "expected_reason": expected_reason,
            "expected_at_seq": expected_at, "heartbeats": heartbeats,
            "sigterm": observed_sigterm is not None,
            "sigkill": sigkill_at_mono is not None}


# ---------------------------------------------------------------------------
# The safety state machine
# ---------------------------------------------------------------------------


@dataclass
class SafetyState:
    """Consecutive counters, kept here so replay can rebuild them exactly.

    A failed poll neither advances nor resets a violation streak. Reading
    nothing is not evidence of safety and it is not evidence of danger, so it
    must not quietly clear a streak that was building.
    """

    spec: SafetySpec = field(default_factory=SafetySpec)
    pressure_streak: int = 0
    free_streak: int = 0
    failure_streak: int = 0
    slow_row_streak: int = 0
    last_swap: float | None = None
    last_monotonic: float | None = None
    first_monotonic: float | None = None
    progress_seen: bool = False
    progress_missing_polls: int = 0
    progress_stall_polls: int = 0
    _row: int | None = field(default=None, repr=False)
    _row_max_elapsed: float = field(default=0.0, repr=False)

    def _fold_progress(self, progress: dict | None) -> str | None:
        """Row-level limits, judged from durations the child reports.

        Instants cannot cross a process boundary -- two monotonic clocks have
        different origins -- but elapsed seconds can, so the child reports how
        long it has been inside the current row and how long its two clocks
        have run.

        Two absences are deliberately different. Progress that has *never*
        arrived is normal for a while: the child may still be loading a model,
        and no row has begun. That case is bounded by ``process_max_seconds``
        on the watchdog's own clock, not by a rule here. Progress that arrived
        and then stopped advancing is a wedged child, and that is a tool
        failure with its own name.
        """
        if not progress:
            self.progress_missing_polls += 1
            return None
        self.progress_seen = True
        row = progress.get("row")
        # A row number is an int, it starts at 1, and it never goes backwards.
        # Anything else means the file we are reading is not the progress file
        # we think it is, and guessing which reading to believe is worse than
        # stopping.
        if not isinstance(row, int) or isinstance(row, bool) or row < 1:
            return "progress_invalid"
        if self._row is not None and row < self._row:
            return "progress_invalid"
        elapsed = progress.get("row_elapsed_seconds")
        if True:
            if self._row is None:
                self._row, self._row_max_elapsed = row, 0.0
                self.progress_stall_polls = 0
            elif row > self._row:
                # The previous row is finished: judge it, start the next, and
                # reset the stall counter. Forward motion is the whole point.
                if self._row_max_elapsed > self.spec.slow_row_seconds:
                    self.slow_row_streak += 1
                else:
                    self.slow_row_streak = 0
                self._row, self._row_max_elapsed = row, 0.0
                self.progress_stall_polls = 0
            else:
                # Same row again. Only this branch accumulates.
                self.progress_stall_polls += 1
            if self.progress_stall_polls >= self.spec.progress_stall_polls:
                return "progress_stalled"
            if _finite(elapsed):
                self._row_max_elapsed = max(self._row_max_elapsed, float(elapsed))
                if self._row_max_elapsed > self.spec.slow_row_seconds and \
                        self.slow_row_streak + 1 >= self.spec.slow_row_streak:
                    return "slow_row_seconds"
        if self.slow_row_streak >= self.spec.slow_row_streak:
            return "slow_row_seconds"
        cond = progress.get("condition_clock_seconds")
        if _finite(cond) and cond >= self.spec.max_seconds:
            return "max_seconds"
        return None

    def observe(self, sample: dict | None, monotonic: float,
                progress: dict | None = None) -> dict:
        """Fold one poll in. ``sample is None`` means the probe failed."""
        gap = None if self.last_monotonic is None else monotonic - self.last_monotonic
        late = gap is not None and gap > self.spec.poll_max_gap_seconds
        self.last_monotonic = monotonic
        if self.first_monotonic is None:
            self.first_monotonic = monotonic
        # The process ceiling is enforced from OUR elapsed time, never from a
        # number the child publishes. A child stuck before it ever writes a
        # progress line would otherwise run for ever: there would be nothing
        # to read, and a rule that reads it could never fire.
        own_elapsed = monotonic - self.first_monotonic

        if own_elapsed >= self.spec.process_max_seconds:
            # Checked before anything else can return early. A run that has
            # been alive past the ceiling has to stop even if this particular
            # poll also failed or arrived late.
            self.failure_streak = self.failure_streak + 1 if sample is None \
                else 0
            return {"reason": "process_max_seconds",
                    "failed": None if sample is not None else "probe_error",
                    "violations": self.violations()}
        if sample is None or late:
            # A late poll is the watchdog's own health check. It polls on a
            # timer and never waits for a row boundary, so a slow row cannot
            # produce one: a gap this large means the machine is in trouble,
            # or we are.
            self.failure_streak += 1
            reason = ("probe_unavailable"
                      if self.failure_streak >= self.spec.probe_failure_streak_max
                      else None)
            return {"reason": reason,
                    "failed": "late_poll" if late else "probe_error",
                    "violations": self.violations()}

        self.failure_streak = 0
        swap = sample.get("swap_used_gb")
        pressure = sample.get("memory_pressure_percent_free")
        free_inactive = sample.get("free_plus_inactive_gb")

        reason = None
        if _finite(swap) and swap >= self.spec.swap_used_gb_max:
            reason = "swap_used_gb"
        elif (_finite(swap) and self.last_swap is not None
              and swap - self.last_swap >= self.spec.swap_growth_gb_per_probe_max):
            reason = "swap_growth_gb_per_probe"

        if _finite(pressure) and pressure <= self.spec.memory_pressure_percent_free_min:
            self.pressure_streak += 1
        else:
            self.pressure_streak = 0
        if _finite(free_inactive) and free_inactive <= self.spec.free_plus_inactive_gb_min:
            self.free_streak += 1
        else:
            self.free_streak = 0

        progress_reason = self._fold_progress(progress)
        if own_elapsed >= self.spec.process_max_seconds:
            progress_reason = "process_max_seconds"

        if reason is None:
            if self.pressure_streak >= self.spec.memory_pressure_consecutive_probes:
                reason = "memory_pressure_percent_free"
            elif self.free_streak >= self.spec.free_plus_inactive_consecutive_probes:
                # Secondary on purpose. The 200-row control sat at 0.405 GB
                # with no harm done, so a floor above that would refuse a
                # state the machine is known to tolerate.
                reason = "free_plus_inactive_gb"
            elif progress_reason is not None:
                reason = progress_reason

        if _finite(swap):
            self.last_swap = swap
        return {"reason": reason, "failed": None, "violations": self.violations()}

    def violations(self) -> dict:
        return {"memory_pressure": self.pressure_streak,
                "free_plus_inactive": self.free_streak,
                "slow_row": self.slow_row_streak,
                "progress_stall": self.progress_stall_polls,
                "progress_missing": self.progress_missing_polls}


# ---------------------------------------------------------------------------
# Stop requests. The signal only says "stop"; this file says why, and the
# child refuses one it cannot authenticate.
# ---------------------------------------------------------------------------


def write_stop_request(directory, *, prefix: str, reason: str, rule: str | None,
                       nonce: str, monotonic: float, wall_clock: str,
                       sampled_values: dict | None = None) -> dict:
    directory = Path(directory)
    body = {"reason": reason, "rule": rule, "nonce": nonce,
            "monotonic": monotonic, "wall_clock": wall_clock,
            "sampled_values": sampled_values or {}}
    text = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=1)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    for name, payload in ((f"{prefix}.stop_request.json", text),
                          (f"{prefix}.stop_request.sha256", digest + "\n")):
        fd = os.open(directory / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
    return {"body": body, "sha256": digest}


def read_stop_request(directory, *, prefix: str, nonce: str) -> dict:
    """Authenticate a stop request.

    Both checks matter. The digest says the file was not edited between being
    written and being read; the nonce says it belongs to *this* launch and not
    to a stale file left behind by an earlier one.
    """
    directory = Path(directory)
    req = directory / f"{prefix}.stop_request.json"
    sha = directory / f"{prefix}.stop_request.sha256"
    if not req.exists() or not sha.exists():
        return {"accepted": False, "reason": "stop_request_rejected",
                "detail": "the stop request or its digest is missing"}
    text = req.read_text(encoding="utf-8")
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual != sha.read_text(encoding="utf-8").strip():
        return {"accepted": False, "reason": "stop_request_rejected",
                "detail": "the stop request does not match its recorded digest"}
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return {"accepted": False, "reason": "stop_request_rejected",
                "detail": "the stop request is not valid JSON"}
    if body.get("nonce") != nonce:
        return {"accepted": False, "reason": "stop_request_rejected",
                "detail": "the stop request carries another launch's nonce"}
    return {"accepted": True, "body": body, "sha256": actual}


# ---------------------------------------------------------------------------
# The signal-handler contract
# ---------------------------------------------------------------------------


class StopFlag:
    """All a SIGTERM handler is allowed to do.

    Writing JSON from inside a signal handler is not async-signal-safe: the
    handler can land in the middle of the very allocator or file object it
    would need, and the failure shows up exactly when the machine is already
    in trouble. So the handler sets a boolean and returns, and the child does
    the real work at the next row boundary, where it can fail cleanly.
    """

    __slots__ = ("_set",)

    def __init__(self) -> None:
        self._set = False

    def handler(self, signum, frame) -> None:  # noqa: ARG002 - signal ABI
        self._set = True

    def is_set(self) -> bool:
        return self._set


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def watchdog_loop(log: WatchdogLog, identity: ChildIdentity, spec: SafetySpec,
                  *, probe, clock, child_alive, signaller, heartbeat,
                  observed_start, directory, prefix: str = "run",
                  read_progress=lambda: None, wall_clock=lambda: "",
                  observed_pgid_fn=None, launch_nonce=None,
                  sleep=lambda _s: None, max_polls: int = 10 ** 9) -> dict:
    """Poll, judge, and if needed stop the child.

    The grace period is measured on *this* process's monotonic clock -- not the
    parent's, whose origin differs, and not wall clock, which can step under
    NTP at the worst possible moment.
    """
    observed_pgid_fn = observed_pgid_fn or (lambda: observed_pgid(identity.pid))
    # The nonce has to come from somewhere other than the identity we are
    # checking. Comparing identity.nonce with itself always passes and proves
    # nothing; the launch record the parent wrote is the independent copy.
    launch_nonce = launch_nonce or (lambda: read_launch_nonce(directory, prefix))
    # Stamped from the identity this loop actually enforces against, so the
    # log cannot describe one process while the signals go to another.
    log.arm(identity)
    state = SafetyState(spec=spec)
    stopped: dict | None = None
    sigterm_at: float | None = None
    beat = 0

    for _ in range(max_polls):
        now = clock()
        if not child_alive():
            # The last thing this loop does, and the only place it is written:
            # the watchdog saw the child go. A run whose ending nobody
            # recorded cannot be a completed run, because a watchdog that died
            # in the first second leaves a log indistinguishable from one that
            # watched to the end (§4.5.3).
            log.append({"monotonic": now, "wall_clock": wall_clock(),
                        "swap_used_gb": None,
                        "memory_pressure_percent_free": None,
                        "free_plus_inactive_gb": None,
                        "failed": None,
                        "failure_streak": state.failure_streak,
                        "violations": state.violations(),
                        "action": CHILD_EXIT_OBSERVED,
                        "progress": read_progress(),
                        "heartbeat_seq": beat})
            break

        try:
            sample = probe()
        except Exception:  # noqa: BLE001 - any probe failure is one failure
            sample = None
        progress = read_progress()
        verdict = state.observe(sample, now, progress=progress)
        record = {"monotonic": now, "wall_clock": wall_clock(),
                  "failed": verdict["failed"],
                  "failure_streak": state.failure_streak,
                  "violations": verdict["violations"],
                  "action": "poll", "progress": progress,
                  "heartbeat_seq": beat}
        record.update({k: (sample or {}).get(k) for k in
                       ("swap_used_gb", "memory_pressure_percent_free",
                        "free_plus_inactive_gb")})
        log.append(record)
        heartbeat({"seq": beat, "monotonic": now,
                   "reason": verdict["reason"]})
        beat += 1

        if stopped is None and verdict["reason"] is not None:
            reason = verdict["reason"]
            check = identity_check(identity, nonce=launch_nonce(),
                                   observed_start=observed_start(),
                                   observed_pgid_value=observed_pgid_fn())
            if not check["ok"]:
                # The PID is right and the process is not. Signal nothing.
                log.append({"monotonic": clock(), "wall_clock": wall_clock(),
                            "failed": "identity_mismatch",
                            "failure_streak": state.failure_streak,
                            "violations": verdict["violations"],
                            "action": "identity_mismatch", "progress": progress,
                            "heartbeat_seq": beat})
                return {"stopped": True, "reason": "identity_mismatch",
                        "is_tool_failure": True, "sigkilled": False,
                        "detail": check["problems"]}
            write_stop_request(directory, prefix=prefix, reason=reason,
                               rule=reason, nonce=identity.nonce,
                               monotonic=now, wall_clock=wall_clock(),
                               sampled_values=sample or {})
            signaller("SIGTERM", identity.pgid)
            sigterm_at = clock()
            log.append({"monotonic": sigterm_at, "wall_clock": wall_clock(),
                        "failed": verdict["failed"],
                        "failure_streak": state.failure_streak,
                        "violations": verdict["violations"],
                        "action": "sigterm", "progress": progress,
                        "heartbeat_seq": beat})
            stopped = {"reason": reason,
                       "is_tool_failure": is_tool_failure(reason)}

        if sigterm_at is not None and clock() - sigterm_at >= spec.grace_seconds:
            check = identity_check(identity, nonce=launch_nonce(),
                                   observed_start=observed_start(),
                                   observed_pgid_value=observed_pgid_fn())
            if check["ok"]:
                signaller("SIGKILL", identity.pgid)
                log.append({"monotonic": clock(), "wall_clock": wall_clock(),
                            "failed": None,
                            "failure_streak": state.failure_streak,
                            "violations": verdict["violations"],
                            "action": "sigkill", "progress": progress,
                            "heartbeat_seq": beat})
                return {**stopped, "stopped": True, "sigkilled": True}
            return {"stopped": True, "reason": "identity_mismatch",
                    "is_tool_failure": True, "sigkilled": False,
                    "detail": check["problems"]}

        sleep(spec.poll_seconds)

    if stopped is None:
        return {"stopped": False, "reason": None, "is_tool_failure": False,
                "sigkilled": False}
    return {**stopped, "stopped": True, "sigkilled": False}


# ---------------------------------------------------------------------------
# Parent-side supervision
# ---------------------------------------------------------------------------


def supervise(*, child_alive, watchdog_alive, poll_heartbeat, clock,
              spec: SafetySpec, on_stop, sleep=lambda _s: None,
              max_iterations: int = 10 ** 9) -> dict:
    """Watch child, watchdog and heartbeat together, live.

    Freshness is measured from when *this* process received a heartbeat, not
    from a timestamp inside it: a watchdog that has wedged can still be
    emitting stale numbers, and a clock written by the sender proves nothing
    about when it arrived.

    The first heartbeat gets its own deadline. Without one, a watchdog that
    armed and then immediately wedged would look "fresh" forever, because
    there would be no last-received time to age.

    The child is checked *first*, and that ordering is the whole of it. At the
    end of a normal run the child exits and the watchdog, seeing it gone,
    exits too; by the time the parent looks, both are usually already reaped.
    Asking "is the watchdog alive" first turned that into ``watchdog_died`` --
    a tool failure, terminal incomplete, a spent boot -- on the strength of
    which process the kernel happened to reap first. A child that has finished
    is not a supervision failure: the run is over, and what it produced is
    decided by ``collect`` from the exit status, the report, the watchdog log
    and their digests. Only a watchdog that vanishes from under a *running*
    child leaves anything unwatched.
    """
    started = clock()
    last_received: float | None = None
    for _ in range(max_iterations):
        now = clock()
        if poll_heartbeat():
            last_received = clock()
        if not child_alive():
            return {"stopped": False, "reason": None, "is_tool_failure": False}
        if not watchdog_alive():
            on_stop("watchdog_died")
            return {"stopped": True, "reason": "watchdog_died",
                    "is_tool_failure": True}
        if last_received is None:
            if now - started > spec.first_heartbeat_deadline_seconds:
                on_stop("heartbeat_never_arrived")
                return {"stopped": True, "reason": "heartbeat_never_arrived",
                        "is_tool_failure": True}
        elif now - last_received > spec.heartbeat_gap_seconds:
            on_stop("heartbeat_gap")
            return {"stopped": True, "reason": "heartbeat_gap",
                    "is_tool_failure": True}
        sleep(spec.parent_supervision_interval_seconds)
    return {"stopped": False, "reason": None, "is_tool_failure": False}


# ---------------------------------------------------------------------------
# Real IPC helpers, kept small so the integration test can use the real thing
# ---------------------------------------------------------------------------


def write_line(fd: int, obj: dict) -> None:
    os.write(fd, (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


class LineReader:
    """Non-blocking newline-delimited JSON from a pipe."""

    def __init__(self, fd: int):
        self.fd = fd
        os.set_blocking(fd, False)
        self._buf = b""

    def poll(self) -> list[dict]:
        out = []
        try:
            chunk = os.read(self.fd, 65536)
        except BlockingIOError:
            chunk = b""
        except OSError as exc:
            if exc.errno != errno.EAGAIN:
                raise
            chunk = b""
        if chunk:
            self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


def read_progress_tail(path) -> dict | None:
    """Last complete progress line, or None. Partial lines are ignored."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(text.split("\n")):
        if not line.strip():
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# A probe that reads the machine, and a microbenchmark for its cost.
# ---------------------------------------------------------------------------


def default_probe() -> dict:
    """Read the three OS-level numbers the safety rules use.

    ``free_plus_inactive_gb`` is what ``vm_stat`` lets you add up. Inactive
    pages are reclaimable, so it is neither free memory nor available memory,
    and it is a secondary signal here for exactly that reason.
    """
    from src.training.preflight import preflight_sample

    sample = preflight_sample()
    return {k: sample.get(k) for k in ("swap_used_gb",
                                       "memory_pressure_percent_free",
                                       "free_plus_inactive_gb")}


def microbenchmark(n: int = 20, *, probe=default_probe, cadence: float = 5.0,
                   cycles: int = 0, sleep=time.sleep) -> dict:
    """Measure what one probe costs, and how far a 5 s cadence drifts.

    A tool cost measurement. It is not a report 16 experiment, it creates no
    session and it loads nothing.
    """
    wall, cpu = [], []
    for _ in range(n):
        w0, c0 = time.perf_counter(), time.process_time()
        probe()
        wall.append(time.perf_counter() - w0)
        cpu.append(time.process_time() - c0)

    drift = []
    if cycles:
        target = time.monotonic()
        for _ in range(cycles):
            target += cadence
            probe()
            sleep(max(0.0, target - time.monotonic()))
            drift.append(time.monotonic() - target)

    def stats(xs):
        if not xs:
            return {"n": 0, "mean": None, "min": None, "max": None}
        s = sorted(xs)
        return {"n": len(xs), "mean": round(sum(xs) / len(xs), 6),
                "min": round(s[0], 6), "max": round(s[-1], 6),
                "median": round(s[len(s) // 2], 6)}

    return {"kind": "watchdog_microbenchmark",
            "note": "tool cost only; not a report 16 experiment, no session, "
                    "no model",
            "probe_wall_seconds": stats(wall),
            "probe_cpu_seconds": stats(cpu),
            "cadence_seconds": cadence,
            "cadence_drift_seconds": stats(drift)}
