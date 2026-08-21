#!/usr/bin/env python3
"""Report 16 -- long-run verification tool. The entry points are open.

The four executing entry points were refused until the user approved
unlocking them. Approval was always specified as the deletion of the
refusals and nothing else, so that is what happened: the wiring behind them
is the wiring that was already there and already tested.

**Report 16 has produced no measurement.** One session exists on disk --
``exp001`` -- and it is sealed ``terminal_incomplete``: its single run
started, the child exited 1 before writing a report, and by the
one-run-per-boot rule that ends the experiment in every future boot. Its
``runs`` array is empty. So ``--verify`` has now run against real report 16
data and returned no problems, but what it replayed was one failure, not one
measurement. Any further run needs a new experiment id and a new boot.

Modes, and what each of them does to the disk:

* read-only, writes nothing: ``--verify``, ``--session-status``,
  ``--from-json``, ``--show-plan``;
* writes one file, and only inside ``data/reports/16_longrun_tools/``, and
  only if that exact name is free: ``--microbenchmark --out-name``;
* creates or extends a session: ``--session-init``, ``--session-next``,
  ``--session-finalize``;
* spawned by ``--session-next``, not meant to be typed: ``--child``,
  ``--watchdog-worker``.

One measured run per boot, enforced by the journal rather than by good
intentions: ``--session-next`` spends this boot's fingerprint when it writes
``measurement_started``, and a run that starts without finishing makes the
whole experiment terminal incomplete. This tool does not restart the machine
and must not: a tool that reboots what it is measuring has changed it.

There is still no flag and no environment variable that changes what this
tool does. What it may do is what the source says it does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import longrun  # noqa: E402
from src.training import watchdog as wd  # noqa: E402

REPORT_DIR = longrun.REPORT_DIR
# Tool output has its own directory. A benchmark must never be able to name an
# arbitrary path: --out used to take one, which meant a typo could overwrite a
# report.
TOOL_DIR = ROOT / "data" / "reports" / "16_longrun_tools"

SAFE_OUT_NAME = "abcdefghijklmnopqrstuvwxyz0123456789_-."


def resolve_out_name(name: str) -> Path:
    """A name, never a path, and only inside the tool directory."""
    if not name or any(c not in SAFE_OUT_NAME for c in name.lower()):
        raise ValueError(f"{name!r} is not a plain file name")
    if not name.endswith(".json") or "/" in name or ".." in name:
        raise ValueError(f"{name!r} must be a plain *.json file name")
    TOOL_DIR.mkdir(parents=True, exist_ok=True)
    return TOOL_DIR / name


def cmd_show_plan(args) -> int:
    plan = longrun.build_plan(args.experiment_id or "preview")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print(f"\nplan_digest {plan['plan_digest']}", file=sys.stderr)
    print("preview only: nothing was written and no boot was consumed",
          file=sys.stderr)
    return 0


def _no_session(experiment_id: str) -> int:
    """Say what is true of this experiment id, and nothing wider.

    "nothing has been run yet" was a global claim printed by a per-experiment
    lookup: correct only while no session exists at all, and quietly wrong
    from the first one onwards.
    """
    print(f"no report 16 session {experiment_id!r} exists on disk.")
    print(f"--session-init --experiment-id {experiment_id} would create it.")
    return 1


def cmd_verify(args) -> int:
    paths = longrun.session_paths(args.experiment_id)
    if not paths["dir"].exists():
        return _no_session(args.experiment_id)
    out = longrun.verify_experiment(args.experiment_id)
    print(f"{args.experiment_id}: replay complete")
    for problem in out["problems"]:
        print(f"  problem: {problem}")
    if not out["problems"]:
        print("  problems: []")
    return 0 if not out["problems"] else 1


def cmd_from_json(args) -> int:
    paths = longrun.session_paths(args.experiment_id)
    if not paths["dir"].exists():
        return _no_session(args.experiment_id)
    out = longrun.render_from_json(args.experiment_id)
    if not out["rendered"]:
        print(f"nothing rendered: {out['reason']}", file=sys.stderr)
        for problem in out["problems"]:
            print(f"  problem: {problem}", file=sys.stderr)
        return 1
    print(out["markdown"])
    return 0


def cmd_session_finalize(args) -> int:
    paths = longrun.session_paths(args.experiment_id)
    if not paths["dir"].exists():
        return _no_session(args.experiment_id)
    out = longrun.session_finalize(args.experiment_id)
    for problem in out.get("problems", []):
        print(f"  problem: {problem}", file=sys.stderr)
    return 0 if not out.get("problems") else 1


def cmd_session_status(args) -> int:
    paths = longrun.session_paths(args.experiment_id)
    if not paths["dir"].exists():
        return _no_session(args.experiment_id)
    st = longrun.session_status(args.experiment_id)
    rows = longrun.session_status_rows(st)
    for entry in st["plan"]["runs"]:
        run = entry["run_id"]
        print(f"  {run} k={entry['declared_rows']:<5} {rows[run]}")
    # `next_run` is the state machine's cursor, not a suggestion. On a
    # terminal experiment it still points at the next unstarted arm, and
    # printing that as `pending` reads as an invitation to run something that
    # is unreachable in every future boot.
    print(f"next run: {longrun.session_next_hint(st)}")
    print(f"this boot: {st['this_boot']}"
          + ("  (already used)" if st["boot_already_used"] else ""))
    return 0


def cmd_microbenchmark(args) -> int:
    """Tool cost only. No session, no model, no experiment."""
    result = wd.microbenchmark(n=args.samples, cycles=args.cycles)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out_name:
        from src.training.session import write_once_json

        path = resolve_out_name(args.out_name)
        write_once_json(path, result)   # atomic, and fails if it exists
        print(f"wrote {path.relative_to(ROOT)}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# The wiring a measured run actually uses. It was written out in full while
# the entry points were still refused, so that approval would be the deletion
# of the refusals rather than an afternoon of writing the interesting part in
# a hurry. It has run against a real model: exp001's b1 on 2026-08-20, which
# died loading the tokenizer, and a throwaway eight-row child in round 21,
# which completed. Neither produced a measurement this experiment can use.
# ---------------------------------------------------------------------------


def _child_argv(paths, run, entry, nonce, plan):
    return [sys.executable, str(Path(__file__).resolve()), "--child",
            "--experiment-dir", str(paths["dir"]), "--run", run,
            "--rows", str(entry["declared_rows"]), "--nonce", nonce,
            "--plan-digest", plan["plan_digest"]]


def _watchdog_argv(paths, run, handshake_fd, heartbeat_fd):
    return [sys.executable, str(Path(__file__).resolve()), "--watchdog-worker",
            "--experiment-dir", str(paths["dir"]), "--run", run,
            "--handshake-fd", str(handshake_fd),
            "--heartbeat-fd", str(heartbeat_fd)]


class Launcher:
    """Holds the pipes and processes for one measured run."""

    def __init__(self, spec):
        self.spec = spec
        self.watchdog = None
        self.child = None
        self.hs_w = None
        self.hb_r = None
        self.reader = None
        self.beats = 0

    def gate(self, thresholds, policy):
        """Poll until the machine is back inside the calibrated band.

        Two translations happen here, and both matter. The gate policy and
        ``wait_for_recovery`` name the same three settings differently, and
        passing the policy's names straight through raised TypeError -- at the
        one moment nothing may fail, since the gate runs before the boot is
        spent but after the operator has rebooted the machine for this run.
        The call below is the translation; it is not repeated here, because a
        list of argument names in prose is a list that goes stale.

        The second translation is the poll shape: ``wait_for_recovery``
        numbers its polls from one, while section 4.8 replays them by
        zero-based index and streak. Storing the raw shape would leave
        ``--verify`` unable to recompute a single poll of the gate that let
        this run start.
        """
        from src.training.preflight import wait_for_recovery

        out = wait_for_recovery(
            thresholds,
            needed_consecutive=policy["consecutive_passes_required"],
            poll_seconds=policy["poll_interval_seconds"],
            max_wait_seconds=policy["timeout_seconds"])
        polls = [{"index": i, "elapsed_seconds": p["elapsed_seconds"],
                  "sample": p["sample"], "passed": p["passed"],
                  "streak": p["consecutive_passes"],
                  "failed_metrics": p.get("failed_metrics")}
                 for i, p in enumerate(out.get("polls") or [])]
        passed = bool(out.get("passed"))
        return {
            "passed": passed, "polls": polls,
            # Taken from the poll that released the gate rather than from a
            # second clock reading, so replay's recomputation is exact instead
            # of merely close.
            "waited_seconds": (polls[-1]["elapsed_seconds"]
                               if polls and passed else out.get("waited_seconds")),
            "consecutive_passes_required": policy["consecutive_passes_required"],
            "reason": out.get("reason")}

    def spawn_watchdog(self, *, run, paths):
        import subprocess

        hs_r, self.hs_w = os.pipe()
        self.hb_r, hb_w = os.pipe()
        try:
            self.watchdog = subprocess.Popen(
                _watchdog_argv(paths, run, hs_r, hb_w),
                pass_fds=(hs_r, hb_w), start_new_session=True)
        except OSError as exc:
            return {"ready": False, "reason": f"watchdog failed to start: {exc}"}
        finally:
            os.close(hs_r)
            os.close(hb_w)
        self.reader = wd.LineReader(self.hb_r)
        deadline = time.monotonic() + self.spec.ready_timeout_seconds
        while time.monotonic() < deadline:
            for msg in self.reader.poll():
                if msg.get("ready"):
                    return {"ready": True, "proc": self.watchdog}
            if self.watchdog.poll() is not None:
                return {"ready": False, "reason": "the watchdog exited before "
                                                  "it was ready"}
            time.sleep(0.05)
        return {"ready": False, "reason": "the watchdog never said it was ready",
                "proc": self.watchdog}

    def spawn_child(self, *, run, paths, entry, nonce, plan):
        import subprocess

        try:
            self.child = subprocess.Popen(
                _child_argv(paths, run, entry, nonce, plan),
                # The measured child never reaches the network. Pinned by the
                # parent as well as by the child itself, so a run does not
                # depend on what the operator's shell happened to export.
                env=longrun.offline_environment(),
                start_new_session=True)
        except OSError as exc:
            return {"spawned": False, "reason": str(exc)}
        return {"spawned": True, "proc": self.child, "pid": self.child.pid,
                "nonce": nonce}

    def hand_identity(self, child, *, paths, run):
        proc = child["proc"]
        start = wd.process_start_identity(proc.pid)
        pgid = wd.observed_pgid(proc.pid)
        if start is None or pgid is None:
            return {"ok": False, "reason": "the child vanished before its "
                                           "identity could be read"}
        identity = wd.ChildIdentity(pid=proc.pid, pgid=pgid,
                                    nonce=child["nonce"], start_identity=start)
        wd.write_launch_record(paths["dir"], prefix=run, identity=identity,
                               experiment_id=paths["dir"].name, run_id=run)
        wd.write_line(self.hs_w, identity.as_dict())
        self.identity = identity
        return {"ok": True}

    def await_armed(self):
        deadline = time.monotonic() + self.spec.armed_timeout_seconds
        while time.monotonic() < deadline:
            for msg in self.reader.poll():
                if msg.get("armed"):
                    return {"armed": True}
                self.beats += 1
            if self.watchdog.poll() is not None:
                return {"armed": False, "reason": "the watchdog exited before "
                                                  "it armed"}
            time.sleep(0.05)
        return {"armed": False, "reason": "the watchdog never armed"}

    def supervise(self):
        def poll_heartbeat():
            got = self.reader.poll()
            self.beats += len(got)
            return bool(got)

        return wd.supervise(
            child_alive=lambda: self.child.poll() is None,
            watchdog_alive=lambda: self.watchdog.poll() is None,
            poll_heartbeat=poll_heartbeat, clock=time.monotonic,
            spec=self.spec,
            on_stop=lambda reason: wd.reap(self.child, self.watchdog),
            sleep=time.sleep)

    def collect(self, *, run, paths):
        from src.training.session import sha256_file

        rc = self.child.wait(timeout=60) if self.child else None
        try:
            self.watchdog.wait(timeout=30)
        except Exception:  # noqa: BLE001
            wd.reap(self.watchdog)
        for fd in (self.hs_w, self.hb_r):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        rp = paths["dir"] / f"{run}.json"
        wp = paths["dir"] / f"{run}.watchdog.jsonl"
        return {"outcome": "completed" if rc == 0 else "nonzero_exit",
                "exit_status": rc,
                "report_sha256": sha256_file(rp) if rp.exists() else None,
                "watchdog_sha256": sha256_file(wp) if wp.exists() else None}


def cmd_session_init(args) -> int:
    out = longrun.session_init(args.experiment_id,
                               calibration_path=args.calibration)
    print(f"initialised {out['paths']['dir']}")
    print(f"plan_digest {out['plan']['plan_digest']}")
    return 0


def cmd_session_next(args) -> int:
    state = longrun.session_state(args.experiment_id)
    spec = longrun.plan_spec(state["plan"])
    calib = json.loads(state["paths"]["calibration"].read_text())
    launcher = Launcher(spec)
    out = longrun.session_next(
        args.experiment_id,
        gate=lambda: launcher.gate(calib["thresholds"], longrun.GATE_POLICY),
        spawn_watchdog=launcher.spawn_watchdog,
        spawn_child=launcher.spawn_child,
        hand_identity=launcher.hand_identity,
        await_armed=launcher.await_armed,
        supervise_fn=launcher.supervise,
        collect=launcher.collect)
    for problem in out.get("problems", []):
        print(f"  problem: {problem}", file=sys.stderr)
    return 0 if out.get("ok") else 1


def cmd_child(args) -> int:
    """The measured child. Loads the model and trains.

    Spawned by ``--session-next`` with a nonce and a plan digest. It
    re-derives the plan and the source snapshot before it loads anything, and
    refuses a stop request that does not match its own launch.
    """
    from src.training.longrun import run_child

    return run_child(experiment_dir=Path(args.experiment_dir), run=args.run,
                     rows=args.rows, nonce=args.nonce,
                     plan_digest=args.plan_digest)


def cmd_watchdog_worker(args) -> int:
    from src.training.longrun import run_watchdog_worker

    return run_watchdog_worker(experiment_dir=Path(args.experiment_dir),
                               run=args.run, handshake_fd=args.handshake_fd,
                               heartbeat_fd=args.heartbeat_fd)


#: What each mode cannot run without, keyed by its argparse destination.
#: Checked in one place, before dispatch, so a missing argument is argparse's
#: own exit 2 rather than a TypeError three frames down -- by which point
#: --session-init had already created the directory and spent the experiment
#: id, because sessions are never reopened.
MODE_REQUIREMENTS = {
    "session_init": ("experiment_id", "calibration"),
    "session_next": ("experiment_id",),
    "session_finalize": ("experiment_id",),
    "verify": ("experiment_id",),
    "session_status": ("experiment_id",),
    "from_json": ("experiment_id",),
    "child": ("experiment_dir", "run", "rows", "nonce", "plan_digest"),
    "watchdog_worker": ("experiment_dir", "run", "handshake_fd",
                        "heartbeat_fd"),
    # --show-plan builds a preview plan for whatever id it is given, or none;
    # --microbenchmark measures this tool and never touches a session.
    "show_plan": (),
    "microbenchmark": (),
}

MODES = tuple(MODE_REQUIREMENTS)


def _flag(dest: str) -> str:
    return "--" + dest.replace("_", "-")


def selected_mode(args) -> str | None:
    """Which mode was asked for. The parser guarantees at most one."""
    return next((mode for mode in MODES if getattr(args, mode, False)), None)


def check_arguments(parser, args) -> None:
    """Every mode's arguments, in one place, before anything is dispatched.

    ``is None`` rather than falsiness: ``--rows 0`` and ``--heartbeat-fd 0``
    are values, not absences, and a check that cannot tell them apart is a
    check that refuses the wrong command lines.
    """
    mode = selected_mode(args)
    if mode is None:
        return
    missing = [_flag(name) for name in MODE_REQUIREMENTS[mode]
               if getattr(args, name, None) is None]
    if missing:
        parser.error(f"{_flag(mode)} requires {', '.join(missing)}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-id")
    # One mode per invocation. `--verify --session-next` used to dispatch to
    # session-next, because main() read the modes in a fixed order: a command
    # line that reads like a read-only replay would spend a boot, and a
    # measured run that starts without finishing makes the whole experiment
    # terminal incomplete. Precedence is the wrong answer to that -- two modes
    # on one command line is a mistake, not a ranking.
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--session-init", action="store_true",
                       help="create the session, its immutable plan and its "
                            "source snapshot")
    modes.add_argument("--session-next", action="store_true",
                       help="run the next measured run; spends this boot")
    modes.add_argument("--session-finalize", action="store_true",
                       help="record a finished, terminal or cancelled session")
    modes.add_argument("--verify", action="store_true",
                       help="read-only replay")
    modes.add_argument("--session-status", action="store_true",
                       help="read-only")
    modes.add_argument("--from-json", action="store_true",
                       help="render, but only after a full verification passes")
    modes.add_argument("--show-plan", action="store_true",
                       help="print the plan that would be built; writes nothing")
    modes.add_argument("--microbenchmark", action="store_true",
                       help="measure probe cost and cadence drift; no session")
    modes.add_argument("--child", action="store_true",
                       help="internal: the measured child, spawned by "
                            "--session-next")
    modes.add_argument("--watchdog-worker", action="store_true",
                       help="internal: the out-of-process watchdog, spawned by "
                            "--session-next")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--cycles", type=int, default=0)
    ap.add_argument("--calibration",
                    help="calibration.json for --session-init")
    ap.add_argument("--experiment-dir")
    ap.add_argument("--run")
    ap.add_argument("--rows", type=int)
    ap.add_argument("--nonce")
    ap.add_argument("--plan-digest")
    ap.add_argument("--handshake-fd", type=int)
    ap.add_argument("--heartbeat-fd", type=int)
    ap.add_argument("--out-name",
                    help="file name (not a path) under "
                         "data/reports/16_longrun_tools/; never overwrites")
    return ap


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # The parser has already refused two modes at once. This refuses a mode
    # without the arguments it needs, and it happens here so that every path
    # below can assume they are present -- there is no dispatch that discovers
    # a missing argument by crashing on it.
    check_arguments(parser, args)

    # Exactly one of these can be true, so the order below is presentation,
    # not precedence.
    if args.session_init:
        return cmd_session_init(args)
    if args.session_next:
        return cmd_session_next(args)
    # These two are spawned by --session-next with a nonce and a plan digest.
    # Typing them by hand is possible and pointless: the child re-derives the
    # plan and the source before it loads anything, and the watchdog will not
    # arm without an identity handed to it over the handshake pipe.
    if args.child:
        return cmd_child(args)
    if args.watchdog_worker:
        return cmd_watchdog_worker(args)

    if args.microbenchmark:
        return cmd_microbenchmark(args)
    if args.show_plan:
        return cmd_show_plan(args)
    if not args.experiment_id:
        build_parser().print_help()
        return 0
    if args.verify:
        return cmd_verify(args)
    if args.from_json:
        return cmd_from_json(args)
    if args.session_status:
        return cmd_session_status(args)
    if args.session_finalize:
        return cmd_session_finalize(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
