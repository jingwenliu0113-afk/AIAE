"""Report 16 tests. No model, no tokenizer, no calibration, no real session.

Everything that matters about this design is a state machine: what order
things start in, which failures spend a boot, which failures may become a
verdict, and what replay refuses to accept. All of that is testable with
fakes, and most of it is tested that way here.

The parts that cannot honestly be faked -- process groups, signals, pipes,
heartbeats, a child that ignores SIGTERM -- get real subprocesses instead.
Those children compute nothing; they exist to be supervised, stopped and
reaped, which is the behaviour under test.

The execution lock is verified through the real CLI, never around it.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import importlib.util
import inspect
import pathlib
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import longrun as lr  # noqa: E402
from src.training import watchdog as wd  # noqa: E402

CLI = ROOT / "scripts" / "16_longrun.py"
LOCK_MESSAGE = "Report 16 尚未獲 Codex 執行核准；目前只允許唯讀模式與測試"


#: The one report 16 session that exists *in the private research tree*.
#: exp001 spent a boot on 2026-08-20, its child died loading the tokenizer,
#: and it was finalised as terminal incomplete. It is evidence now, not
#: scratch space -- and it is deliberately absent from any public export, so
#: nothing below may require it in order to check a behaviour.
EXP001 = "exp001"

EXP001_DIR = lr.REPORT_DIR / EXP001
EXP002_DIR = ROOT / "data" / "reports" / "15_mps_order" / "exp002"

#: Every skip in this file that exists only because per-run evidence is not
#: published carries this prefix, and nothing else does. The public-snapshot
#: tests enumerate skips by this string, so a behavioural test that started
#: skipping would be visible rather than absorbed into a total.
ARTIFACT_ONLY = "artifact-only:"

needs_exp001 = pytest.mark.skipif(
    not (EXP001_DIR / "aggregate.json").exists(),
    reason=f"{ARTIFACT_ONLY} exp001's evidence is not in this tree")
needs_exp002 = pytest.mark.skipif(
    not (EXP002_DIR / "session.json").exists(),
    reason=f"{ARTIFACT_ONLY} exp002's evidence is not in this tree")


def report_16_sessions():
    if not lr.REPORT_DIR.exists():
        return []
    return sorted(p.name for p in lr.REPORT_DIR.iterdir() if p.is_dir())


@pytest.fixture
def sessions_unchanged():
    """The set of report 16 sessions is the same after as it was before.

    These tests used to assert ``report_16_sessions() == [EXP001]``, which
    checks two unrelated things at once: that the command created nothing,
    and that this particular tree happens to hold exactly one session. Only
    the first is the behaviour under test. The second is false in a public
    snapshot, where hard-coding it would turn a behavioural check into a
    skip -- which is exactly the failure mode this round exists to prevent.
    """
    before = report_16_sessions()
    yield before
    after = report_16_sessions()
    assert after == before, (
        f"the session set changed: {before} -> {after}")


def load_cli():
    spec = importlib.util.spec_from_file_location("m16", CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DERIVATION = ROOT / "scripts" / "16_calibration_from_report15.py"


def load_derivation():
    spec = importlib.util.spec_from_file_location("m16calib", DERIVATION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def make_rows(n, *, seconds=1.0, every=lr.EMPTY_CACHE_EVERY, clear_seconds=0.27,
              tail_seconds=None, tail_from=None):
    rows = []
    for i in range(n):
        row = 1 + i
        sec = seconds
        if tail_from is not None and row >= tail_from and tail_seconds is not None:
            sec = tail_seconds
        cleared = every is not None and row % every == 0
        rows.append({"row": row, "compute_seconds": sec,
                     "end_to_end_seconds": sec + (clear_seconds if cleared else 0),
                     "sample_id": f"s{row}", "tokens": 100,
                     "supervised_tokens": 70, "loss": 0.5, "cleared": cleared,
                     "clear_seconds": clear_seconds if cleared else None})
    return rows


PROVENANCE = {"code_sha256": {"a": "b"}, "packages": {"torch": "2.13.0"},
              "device": "mps", "dtype": "bfloat16"}


def fake_provenance(k, *, filler="x"):
    """Every section 4.6 field, with the one value allowed to differ set to k.

    A fake that leaves ``measurement_intervals`` empty is a fake that never
    exercises the one field the design says may vary between runs, which is
    precisely where a cross-run comparison can go wrong.
    """
    prov = {f: ({} if f.endswith(("sha256", "digest", "config", "optimizer",
                                  "packages", "conditions"))
                else filler) for f in lr.PROVENANCE_FIELDS}
    prov["measurement_intervals"] = {"window": lr.WINDOW,
                                     "memory_every": lr.MEMORY_EVERY,
                                     "empty_cache_every": lr.EMPTY_CACHE_EVERY,
                                     "max_rows": k}
    return prov


def make_report(n, *, run_id="b1", declared=None, seconds=1.0, tail_seconds=None,
                tail_from=None, stopped=None, clear_seconds=0.27,
                with_metrics=False, nonce="n", experiment_id="exp016a"):
    rows = make_rows(n, seconds=seconds, tail_seconds=tail_seconds,
                     tail_from=tail_from, clear_seconds=clear_seconds)
    calls = [{"row": r["row"], "seconds": r["clear_seconds"]}
             for r in rows if r["cleared"]]
    total = sum(c["seconds"] for c in calls)
    e2e = sum(r["end_to_end_seconds"] for r in rows)
    compute = sum(r["compute_seconds"] for r in rows)
    k = declared or n
    ids = [r["sample_id"] for r in rows]
    order_ids = ids + [f"s{i}" for i in range(len(rows) + 1, k + 1)]
    prov = fake_provenance(k)
    prov.update(PROVENANCE)
    rep = {"schema_version": lr.CHILD_SCHEMA_VERSION, "kind": lr.CHILD_KIND,
           "experiment_id": experiment_id,
           "run_id": run_id, "declared_rows": k,
           "condition": lr.CONDITION,
           "plan_digest": "d" * 64, "nonce": nonce,
           "pool_pairs": lr.POOL_PAIRS, "pool_rows": lr.POOL_ROWS,
           "rows_requested": k,
           "child_source_check": {"files_verified": 2,
                                  "plan_digest": "d" * 64,
                                  "source_manifest_digest": "m" * 64},
           "preflight": {"swap_used_gb": 0.0,
                         "memory_pressure_percent_free": 95,
                         "free_plus_inactive_gb": 8.0,
                         "normalized_load_1m": 0.1},
           "child_pid": 4242, "child_pgid": 4242,
           "child_start_identity": "T0",
           "started_at": "2026-08-16T00:00:00Z",
           "finished_at": "2026-08-16T00:10:00Z",
           "tool_failure": None, "model_compute_seconds": compute,
           "per_row": rows, "memory": [], "stopped_early": stopped,
           "rows_completed": n, "end_to_end_seconds": e2e,
           "input_order_digest": lr._digest_ids(order_ids),
           "completed_input_digest": lr._digest_ids(ids),
           "provenance": prov,
           "scheduled_empty_cache_every": lr.EMPTY_CACHE_EVERY,
           "scheduled_empty_cache_cost": {"calls": len(calls),
                                          "total_seconds": total,
                                          "per_call": calls},
           "teardown_empty_cache_calls": 1,
           "teardown_empty_cache_seconds": 0.5,
           "between_row_overhead_breakdown": {
               "scheduled_empty_cache_seconds": total,
               "memory_probe_seconds": 0.0,
               "unattributed_seconds": max(0.0, e2e - compute - total)},
           "float_storage": {"seconds_rounded": False, "loss_rounded": False},
           "clocks": {"model_load_seconds": 2.2,
                      "condition_clock_seconds": compute,
                      "process_clock_seconds": compute + 2.2 + 1.0}}
    # ``metrics`` is part of the schema, so it is always written; the
    # ``with_metrics`` flag survives only because a few callers read it as
    # documentation of what the test is about.
    m = lr.compute_metrics(rows, stop_reason=(stopped or {}).get("reason"))
    rep["metrics"] = {key: m.get(key) for key in
                      ("D100", "D100_reason", "D20", "D20_reason",
                       "Dmax", "Dmax_reason")}
    return rep


class FakeClock:
    def __init__(self, step=5.0):
        self.t, self.step = 0.0, step

    def __call__(self):
        return self.t

    def advance(self, dt=None):
        self.t += self.step if dt is None else dt


def sample(swap=0.0, pressure=60, free=5.0):
    return {"swap_used_gb": swap, "memory_pressure_percent_free": pressure,
            "free_plus_inactive_gb": free}


def ident(**over):
    base = dict(pid=4242, pgid=4242, nonce=wd.new_nonce(), start_identity="T0")
    base.update(over)
    return wd.ChildIdentity(**base)


def clean_loop_args(tmp_path, identity=None, **over):
    """Also writes the parent's launch record, because the watchdog reads the
    nonce from there rather than from the identity it is checking."""
    identity = identity or ident()
    path = pathlib.Path(tmp_path) / "b1.launch.json"
    if not path.exists():
        wd.write_launch_record(tmp_path, prefix="b1", identity=identity,
                               experiment_id="exp016a", run_id="b1")
    base = dict(probe=lambda: sample(), clock=FakeClock(),
                child_alive=lambda: True, signaller=lambda s, t: None,
                heartbeat=lambda b: None, observed_start=lambda: "T0",
                observed_pgid_fn=lambda: 4242, directory=tmp_path,
                prefix="b1", max_polls=1)
    base.update(over)
    return base


# ===========================================================================
# Constants come from the approved design file
# ===========================================================================


def test_module_constants_match_the_approved_design():
    assert lr.check_constants_match_design() == []


def test_a_plan_cannot_be_built_when_the_constants_disagree(tmp_path, monkeypatch):
    design = json.loads(lr.DESIGN_JSON.read_text())
    design["frozen_constants"]["bands"]["Q1_holds_D100_max"] = 99.0
    alt = tmp_path / "design.json"
    alt.write_text(json.dumps(design))
    with pytest.raises(ValueError, match="disagree with the approved design"):
        lr.build_plan("exp016a", design_path=alt)


def test_a_design_without_frozen_constants_is_refused(tmp_path):
    alt = tmp_path / "design.json"
    alt.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ValueError, match="no frozen_constants"):
        lr.build_plan("exp016a", design_path=alt)


def test_lengths_are_not_a_parameter():
    import inspect
    assert "lengths" not in inspect.signature(lr.build_plan).parameters


def test_a_design_with_other_lengths_is_refused(tmp_path):
    design = json.loads(lr.DESIGN_JSON.read_text())
    design["frozen_constants"]["lengths"] = [100, 200, 300]
    alt = tmp_path / "design.json"
    alt.write_text(json.dumps(design))
    with pytest.raises(ValueError, match="500/1000/2000"):
        lr.build_plan("exp016a", design_path=alt)


def test_the_plan_embeds_the_constants_verbatim():
    p = lr.build_plan("exp016a")
    frozen = lr.load_frozen_constants()
    assert p["frozen_constants"] == frozen
    assert p["design_sha256"] == lr.design_sha256()


def test_plan_digest_is_stable_and_excludes_itself():
    p = lr.build_plan("exp016a")
    assert p["plan_digest"] == lr.plan_digest(p)


def test_rule_r1_text_is_inside_the_plan_digest():
    p = lr.build_plan("exp016a")
    t = json.loads(json.dumps(p))
    t["conditional_stop_rule"]["text"] = "R1: anything goes"
    assert lr.plan_digest(t) != p["plan_digest"]


def test_tampering_with_an_embedded_threshold_changes_the_digest():
    p = lr.build_plan("exp016a")
    t = json.loads(json.dumps(p))
    t["frozen_constants"]["safety"]["swap_used_gb_max"] = 99.0
    assert lr.plan_digest(t) != p["plan_digest"]


def test_the_plan_is_the_500_1000_2000_prefix_plan():
    assert lr.prefix_lengths(lr.build_plan("exp016a")) == [500, 1000, 2000]


# ===========================================================================
# Experiment identity: no path traversal
# ===========================================================================


@pytest.mark.parametrize("bad", ["../escape", "a/b", "..", ".", "", "A-Upper",
                                 "with space", "x" * 64, "/abs", "a\\b",
                                 "tri..ck/../../etc"])
def test_unsafe_experiment_ids_are_refused(bad):
    with pytest.raises(ValueError):
        lr.safe_experiment_id(bad)


@pytest.mark.parametrize("good", ["exp016a", "b", "a-b_c", "x0"])
def test_safe_experiment_ids_are_accepted(good):
    assert lr.safe_experiment_id(good) == good


def test_session_paths_stay_under_the_reports_directory(tmp_path):
    p = lr.session_paths("exp016a", root=tmp_path)
    assert p["dir"].parent == tmp_path.resolve()


def test_session_paths_refuse_traversal(tmp_path):
    with pytest.raises(ValueError):
        lr.session_paths("../../etc", root=tmp_path)


# ===========================================================================
# Session lifecycle on tmp_path
# ===========================================================================


def calibration_file(tmp_path):
    path = tmp_path / "calibration.json"
    samples = [{"swap_used_gb": 0.0, "memory_pressure_percent_free": 95,
                "free_plus_inactive_gb": 8.0, "normalized_load_1m": 0.1}
               for _ in range(10)]
    from src.training.preflight import calibrate, thresholds_from
    stats = calibrate(samples)
    path.write_text(json.dumps({"samples": samples, "stats": stats,
                                "thresholds": thresholds_from(stats),
                                "policy": dict(lr.GATE_POLICY)}))
    return path


def init(tmp_path, eid="exp016a"):
    global THRESHOLDS
    out = lr.session_init(eid, calibration_path=calibration_file(tmp_path),
                          root=tmp_path / "reports",
                          code_files=("src/training/longrun.py",
                                      "src/training/watchdog.py"))
    THRESHOLDS = json.loads(out["paths"]["calibration"].read_text())["thresholds"]
    return out


def test_session_init_writes_plan_session_snapshot_and_calibration(tmp_path):
    out = init(tmp_path)
    p = out["paths"]
    assert p["plan"].exists() and p["session"].exists()
    assert p["calibration"].exists() and p["snapshot"].is_dir()
    assert out["session"]["one_run_per_boot"] is True


def test_session_init_refuses_to_reopen(tmp_path):
    init(tmp_path)
    with pytest.raises(FileExistsError, match="never reopened"):
        init(tmp_path)


def test_plan_and_session_are_no_clobber(tmp_path):
    out = init(tmp_path)
    from src.training.session import write_once_json
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        write_once_json(out["paths"]["plan"], {"tampered": True})


def test_a_tampered_plan_fails_to_load(tmp_path):
    out = init(tmp_path)
    plan = json.loads(out["paths"]["plan"].read_text())
    plan["runs"][0]["declared_rows"] = 999
    out["paths"]["plan"].write_text(json.dumps(plan))
    assert any("does not hash" in p
               for p in lr.load_session("exp016a",
                                        root=tmp_path / "reports")["problems"])


def test_a_plan_naming_another_experiment_is_refused(tmp_path):
    out = init(tmp_path)
    plan = json.loads(out["paths"]["plan"].read_text())
    plan["experiment_id"] = "somewhere-else"
    plan["plan_digest"] = lr.plan_digest(plan)
    out["paths"]["plan"].write_text(json.dumps(plan))
    problems = lr.load_session("exp016a", root=tmp_path / "reports")["problems"]
    assert any("names another experiment" in p for p in problems)


def test_session_status_reports_pending_runs(tmp_path):
    init(tmp_path)
    st = lr.session_status("exp016a", root=tmp_path / "reports")
    assert st["completed"] == [] and st["next_run"] == "b1"
    assert st["this_boot"] and st["boot_already_used"] is False


def test_finalize_refuses_while_the_session_can_continue(tmp_path):
    init(tmp_path)
    out = lr.session_finalize("exp016a", root=tmp_path / "reports")
    assert any("can still continue" in p for p in out["problems"])


# ===========================================================================
# One run per boot, journal, terminal incomplete
# ===========================================================================


def gate_ok():
    return gate_record(THRESHOLDS)


def _started_body(paths, run):
    for ev in lr.read_journal(paths["dir"]):
        if ev["event"] == lr.EVENT_STARTED and ev["run_id"] == run:
            return ev["body"]
    return {}


def fake_collect(run, paths, *, terminal=True, **report_kw):
    """A collector that produces the evidence a completed run needs.

    It reads the launch nonce and the declared length from the journal, the
    way a real collector reads them from the run it just supervised.
    """
    plan = json.loads((paths["dir"] / "plan.json").read_text())
    entry = next(e for e in plan["runs"] if e["run_id"] == run)
    started = _started_body(paths, run)
    k = entry["declared_rows"]
    rep = make_report(k, run_id=run, declared=k, with_metrics=True,
                      nonce=started.get("nonce", "n"),
                      experiment_id=paths["dir"].name, **report_kw)
    rep["plan_digest"] = plan["plan_digest"]
    rep["condition"] = entry["condition"]
    rep["provenance"] = fake_provenance(k)
    rep["child_source_check"] = real_source_check(paths, plan)
    copy_identity(rep, paths, run)
    rp = paths["dir"] / f"{run}.json"
    rp.write_text(json.dumps(rep))
    log = wd.WatchdogLog(paths["dir"] / f"{run}.watchdog.jsonl")
    log.arm(launch_identity(paths, run))
    state = wd.SafetyState()
    v = state.observe(sample(), 0.0, progress=None)
    log.append({"monotonic": 0.0, "wall_clock": "Z", **sample(),
                "failed": v["failed"], "failure_streak": state.failure_streak,
                "violations": v["violations"], "action": "poll",
                "progress": None, "heartbeat_seq": 0})
    if terminal:
        log.append({"monotonic": 1.0, "wall_clock": "Z",
                    "swap_used_gb": None, "memory_pressure_percent_free": None,
                    "free_plus_inactive_gb": None,
                    "failed": None, "failure_streak": state.failure_streak,
                    "violations": v["violations"],
                    "action": wd.CHILD_EXIT_OBSERVED,
                    "progress": None, "heartbeat_seq": 1})
    return {"outcome": "completed", "exit_status": 0,
            "report_sha256": hashlib.sha256(rp.read_bytes()).hexdigest(),
            "watchdog_sha256": log.seal()}


def collector(*, terminal=True, **report_kw):
    """The shared collector, shaped per test.

    Tests used to hand-roll their own copies of this; every helper change
    then had to be made in several places, and the copies quietly fell
    behind -- which is how a fake ends up more permissive than the thing it
    stands in for.
    """
    def collect(*, run, paths):
        return fake_collect(run, paths, terminal=terminal, **report_kw)
    return collect


def launch_identity(paths, run):
    """The identity the parent recorded, as the watchdog would receive it."""
    body = json.loads((paths["dir"] / f"{run}.launch.json").read_text())
    return wd.ChildIdentity.from_dict(body)


def real_source_check(paths, plan):
    """What a child that really checked the source would have recorded."""
    session = json.loads((paths["dir"] / "session.json").read_text())
    return {"files_verified": len(session["source_manifest"]["files"]),
            "source_manifest_digest": session["source_manifest_digest"],
            "plan_digest": plan["plan_digest"],
            "verified_at": "2026-08-16T00:00:00Z"}


def copy_identity(report, paths, run):
    """Make the report name the process the launch record names.

    A real child reports its own pid, pgid and start time, and the parent
    wrote the same four values a moment earlier; a fake whose two accounts
    disagree by construction would make every identity test pass for the
    wrong reason.
    """
    launch = json.loads((paths["dir"] / f"{run}.launch.json").read_text())
    for field in lr.IDENTITY_FIELDS:
        report[field] = launch[field]
    return report


def fake_hand_identity(child, *, paths, run):
    """Write the launch record a real hand-off writes.

    The parent's independent account of who the child was is evidence the
    session is judged on later, so a fake that skips it would let the
    precondition replay pass on a session no real launch could produce.
    """
    identity = wd.ChildIdentity(pid=os.getpid(), pgid=os.getpgid(0),
                                nonce=child["nonce"], start_identity="T0")
    wd.write_launch_record(paths["dir"], prefix=run, identity=identity,
                           experiment_id=paths["dir"].name, run_id=run)
    return {"ok": True}


def next_args(tmp_path, **over):
    base = dict(
        gate=gate_ok,
        spawn_watchdog=lambda run, paths: {"ready": True, "proc": None},
        spawn_child=lambda run, paths, entry, nonce, plan: {
            "spawned": True, "proc": None, "pid": 1, "nonce": nonce},
        hand_identity=fake_hand_identity,
        await_armed=lambda: {"armed": True},
        supervise_fn=lambda: {"stopped": False},
        collect=fake_collect,
        boot={"boot_fingerprint": "boot-1"},
        # Injected, so the suite does not depend on what happens to be in this
        # machine's hub cache. The real preflight has its own tests.
        dependency_preflight=lambda: {"ok": True, "problems": [],
                                      "evidence": {"repositories": []}},
        verify_live_sources=False)
    base.update(over)
    base["root"] = tmp_path / "reports"
    return base


def test_a_measured_run_writes_started_then_finished(tmp_path):
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(tmp_path))
    assert out["ok"]
    st = lr.session_state("exp016a", root=tmp_path / "reports")
    kinds = [e["event"] for e in st["events"]]
    assert kinds == [lr.EVENT_STARTED, lr.EVENT_FINISHED]
    assert st["completed"] == ["b1"] and st["next_run"] == "b2"


def test_the_same_boot_cannot_measure_twice(tmp_path):
    init(tmp_path)
    lr.session_next("exp016a", **next_args(tmp_path))
    out = lr.session_next("exp016a", **next_args(tmp_path))
    assert not out["ok"]
    assert any("already happened in this boot" in p for p in out["problems"])


def test_a_different_boot_may_measure_the_next_run(tmp_path):
    init(tmp_path)
    lr.session_next("exp016a", **next_args(tmp_path))
    out = lr.session_next("exp016a",
                          **next_args(tmp_path, boot={"boot_fingerprint": "boot-2"}))
    assert out["ok"]


def test_a_failed_gate_consumes_no_boot(tmp_path):
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(
        tmp_path, gate=lambda: {"passed": False, "polls": []}))
    assert out["boot_consumed"] is False
    st = lr.session_state("exp016a", root=tmp_path / "reports")
    assert [e["event"] for e in st["events"]] == [lr.EVENT_GATE_ATTEMPT]
    assert st["next_run"] == "b1"


def test_a_watchdog_that_never_becomes_ready_consumes_no_boot(tmp_path):
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(
        tmp_path, spawn_watchdog=lambda run, paths: {"ready": False,
                                                     "reason": "timeout"}))
    assert out["boot_consumed"] is False and out["retryable"]
    st = lr.session_state("exp016a", root=tmp_path / "reports")
    assert [e["event"] for e in st["events"]] == [lr.EVENT_WATCHDOG_LAUNCH_FAILED]
    assert st["next_run"] == "b1" and not st["terminal"]


def test_a_child_that_fails_to_spawn_is_terminal(tmp_path):
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(
        tmp_path, spawn_child=lambda run, paths, entry, nonce, plan: {"spawned": False}))
    assert out["boot_consumed"] and out["terminal"]
    st = lr.session_state("exp016a", root=tmp_path / "reports")
    assert st["terminal"] == ["b1"]
    again = lr.session_next("exp016a",
                            **next_args(tmp_path, boot={"boot_fingerprint": "b2"}))
    assert not again["ok"] and again["terminal"]


def test_a_tool_failure_during_supervision_is_terminal_not_a_verdict(tmp_path):
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(
        tmp_path, supervise_fn=lambda: {"stopped": True,
                                        "reason": "watchdog_died",
                                        "is_tool_failure": True}))
    assert out["terminal"] and out["reason"] == "watchdog_died"
    st = lr.session_state("exp016a", root=tmp_path / "reports")
    body = st["finished"]["b1"]["body"]
    assert body["outcome"] == "no_report"
    assert body["tool_failure"] == "watchdog_died"
    assert st["completed"] == []


def test_every_failure_after_a_child_exists_runs_cleanup(tmp_path):
    init(tmp_path)
    reaped = []

    class P:
        pid = 1234

        def poll(self):
            return 0

    out = lr.session_next("exp016a", **next_args(
        tmp_path,
        spawn_child=lambda run, paths, entry, nonce, plan: {
            "spawned": True, "proc": P()},
        hand_identity=lambda c, paths, run: {"ok": False, "reason": "gone"}))
    assert out["terminal"] and out["cleanup"] is not None


def test_finished_outcomes_are_a_closed_set(tmp_path):
    out = init(tmp_path)
    with pytest.raises(ValueError, match="not one of"):
        lr._finish(out["paths"], "b1", out["plan"], "boot",
                   outcome="something_else")


# ===========================================================================
# Watchdog identity: all four fields
# ===========================================================================


def test_identity_check_requires_all_four_fields():
    i = ident(pid=10, pgid=10, nonce="N", start_identity="T0")
    assert wd.identity_check(i, nonce="N", observed_start="T0",
                             observed_pgid_value=10)["ok"]
    assert not wd.identity_check(i, nonce="OTHER", observed_start="T0",
                                 observed_pgid_value=10)["ok"]
    assert not wd.identity_check(i, nonce="N", observed_start="T1",
                                 observed_pgid_value=10)["ok"]
    assert not wd.identity_check(i, nonce="N", observed_start="T0",
                                 observed_pgid_value=11)["ok"]
    assert not wd.identity_check(i, nonce="N", observed_start=None,
                                 observed_pgid_value=10)["ok"]
    assert not wd.identity_check(i, nonce="N", observed_start="T0",
                                 observed_pgid_value=None)["ok"]


def test_a_matching_start_identity_is_not_enough_on_its_own(tmp_path):
    """The gap Codex found: checking only lstart lets a regrouped or
    stale-nonce target through."""
    sent = []
    i = ident()
    log = wd.WatchdogLog(tmp_path / "w.jsonl")
    out = wd.watchdog_loop(log, i, wd.SafetySpec(), **clean_loop_args(
        tmp_path, identity=i, probe=lambda: sample(swap=99.0),
        signaller=lambda s, t: sent.append(s),
        observed_start=lambda: "T0", observed_pgid_fn=lambda: 9999))
    assert sent == []
    assert out["reason"] == "identity_mismatch" and out["is_tool_failure"]


def test_nonce_is_not_reused():
    assert wd.new_nonce() != wd.new_nonce()


def test_signals_target_the_process_group(tmp_path):
    sent = []
    i = ident(pgid=999)
    log = wd.WatchdogLog(tmp_path / "w.jsonl")
    wd.watchdog_loop(log, i, wd.SafetySpec(grace_seconds=1e9),
                     **clean_loop_args(tmp_path, identity=i,
                                       probe=lambda: sample(swap=99.0),
                                       signaller=lambda s, t: sent.append((s, t)),
                                       observed_pgid_fn=lambda: 999))
    assert sent == [("SIGTERM", 999)]


# ===========================================================================
# Every safety threshold is actually enforced
# ===========================================================================


def test_swap_ceiling():
    assert wd.SafetyState().observe(sample(swap=12.0), 0.0)["reason"] == "swap_used_gb"


def test_swap_growth():
    s = wd.SafetyState()
    s.observe(sample(swap=1.0), 0.0)
    assert s.observe(sample(swap=3.0), 5.0)["reason"] == "swap_growth_gb_per_probe"


def test_pressure_needs_five_consecutive_polls():
    s = wd.SafetyState()
    for i in range(4):
        assert s.observe(sample(pressure=4), i * 5.0)["reason"] is None
    assert s.observe(sample(pressure=4), 20.0)["reason"] == "memory_pressure_percent_free"


def test_free_plus_inactive_is_secondary_and_below_what_the_control_survived():
    s = wd.SafetyState()
    for i in range(5):
        r = s.observe(sample(free=0.405), i * 5.0)
    assert r["reason"] is None
    s2 = wd.SafetyState()
    for i in range(5):
        r2 = s2.observe(sample(free=0.2), i * 5.0)
    assert r2["reason"] == "free_plus_inactive_gb"


def test_slow_rows_trip_after_three_consecutive_over_120s():
    s = wd.SafetyState()
    reason = None
    t = 0.0
    for row in (1, 2, 3, 4):
        for _ in range(3):
            reason = s.observe(sample(), t, progress={
                "row": row, "row_elapsed_seconds": 130.0,
                "condition_clock_seconds": t,
                "process_clock_seconds": t})["reason"]
            t += 5.0
            if reason:
                break
        if reason:
            break
    assert reason == "slow_row_seconds"


def test_fast_rows_never_trip_the_slow_row_rule():
    s = wd.SafetyState()
    for row in range(1, 20):
        r = s.observe(sample(), row * 5.0, progress={
            "row": row, "row_elapsed_seconds": 1.0,
            "condition_clock_seconds": row, "process_clock_seconds": row})
    assert r["reason"] is None


def test_condition_clock_ceiling_is_enforced():
    s = wd.SafetyState()
    r = s.observe(sample(), 0.0, progress={"row": 1, "row_elapsed_seconds": 1.0,
                                           "condition_clock_seconds": 28800.0,
                                           "process_clock_seconds": 28900.0})
    assert r["reason"] == "max_seconds"


def test_process_clock_ceiling_uses_the_watchdogs_own_elapsed_time():
    """Not the child's number: a wedged child publishes nothing to read."""
    s = wd.SafetyState()
    s.observe(sample(), 0.0, progress=None)
    r = s.observe(sample(), 30600.0, progress=None)
    assert r["reason"] == "process_max_seconds"


def test_a_failed_poll_neither_advances_nor_resets_a_streak():
    s = wd.SafetyState()
    for i in range(4):
        s.observe(sample(pressure=4), i * 5.0)
    s.observe(None, 20.0)
    assert s.pressure_streak == 4
    assert s.observe(sample(pressure=4), 25.0)["reason"] == "memory_pressure_percent_free"


def test_three_consecutive_poll_failures_are_probe_unavailable():
    s = wd.SafetyState()
    assert s.observe(None, 0.0)["reason"] is None
    assert s.observe(None, 5.0)["reason"] is None
    assert s.observe(None, 10.0)["reason"] == "probe_unavailable"


def test_a_late_poll_counts_as_a_failure():
    s = wd.SafetyState()
    s.observe(sample(), 0.0)
    assert s.observe(sample(), 40.0)["failed"] == "late_poll"


def test_probe_unavailable_is_a_tool_failure_not_a_safety_reason():
    assert wd.is_tool_failure("probe_unavailable")
    assert not wd.is_safety_reason("probe_unavailable")
    assert wd.is_safety_reason("slow_row_seconds")
    assert wd.is_safety_reason("process_max_seconds")


# ===========================================================================
# Parent supervision, including the first heartbeat that never arrives
# ===========================================================================


def test_parent_stops_when_the_watchdog_dies_mid_run():
    stops = []
    out = wd.supervise(child_alive=lambda: True, watchdog_alive=lambda: False,
                       poll_heartbeat=lambda: True, clock=lambda: 1.0,
                       spec=wd.SafetySpec(), on_stop=stops.append,
                       max_iterations=5)
    assert stops == ["watchdog_died"] and out["is_tool_failure"]


def test_parent_measures_freshness_from_when_it_received_a_heartbeat():
    clock = FakeClock(step=1.0)
    received = {"n": 0}

    def poll():
        # One heartbeat at the start, then silence.
        if received["n"] == 0:
            received["n"] += 1
            return True
        return False

    stops = []
    out = wd.supervise(child_alive=lambda: True, watchdog_alive=lambda: True,
                       poll_heartbeat=poll, clock=clock, spec=wd.SafetySpec(),
                       on_stop=stops.append,
                       sleep=lambda s: clock.advance(1.0), max_iterations=100)
    assert stops == ["heartbeat_gap"] and out["reason"] == "heartbeat_gap"


def test_a_first_heartbeat_that_never_arrives_has_its_own_deadline():
    clock = FakeClock(step=1.0)
    stops = []
    out = wd.supervise(child_alive=lambda: True, watchdog_alive=lambda: True,
                       poll_heartbeat=lambda: False, clock=clock,
                       spec=wd.SafetySpec(), on_stop=stops.append,
                       sleep=lambda s: clock.advance(1.0), max_iterations=100)
    assert stops == ["heartbeat_never_arrived"]
    assert out["reason"] == "heartbeat_never_arrived"


def test_fresh_heartbeats_keep_the_run_going():
    clock = FakeClock(step=1.0)
    out = wd.supervise(child_alive=lambda: clock.t < 10,
                       watchdog_alive=lambda: True, poll_heartbeat=lambda: True,
                       clock=clock, spec=wd.SafetySpec(),
                       on_stop=lambda r: None,
                       sleep=lambda s: clock.advance(1.0), max_iterations=50)
    assert out["stopped"] is False


# ===========================================================================
# Stop requests and the signal-handler contract
# ===========================================================================


def test_signal_handler_only_sets_a_flag():
    flag = wd.StopFlag()
    flag.handler(15, None)
    assert flag.is_set()
    assert wd.StopFlag.__slots__ == ("_set",)
    assert not hasattr(flag, "__dict__")


def test_a_stop_request_needs_a_matching_digest_and_nonce(tmp_path):
    wd.write_stop_request(tmp_path, prefix="b1", reason="swap_used_gb",
                          rule="swap_used_gb", nonce="abc", monotonic=1.0,
                          wall_clock="Z")
    assert wd.read_stop_request(tmp_path, prefix="b1", nonce="abc")["accepted"]
    assert not wd.read_stop_request(tmp_path, prefix="b1", nonce="x")["accepted"]


def test_an_edited_stop_request_is_rejected(tmp_path):
    wd.write_stop_request(tmp_path, prefix="b1", reason="swap_used_gb", rule="r",
                          nonce="abc", monotonic=1.0, wall_clock="Z")
    (tmp_path / "b1.stop_request.json").write_text('{"reason": "nothing"}')
    out = wd.read_stop_request(tmp_path, prefix="b1", nonce="abc")
    assert not out["accepted"] and out["reason"] == "stop_request_rejected"


def test_a_missing_stop_request_is_rejected(tmp_path):
    out = wd.read_stop_request(tmp_path, prefix="b1", nonce="abc")
    assert not out["accepted"]


def test_grace_expiry_escalates_to_sigkill_on_the_watchdogs_own_clock(tmp_path):
    sent, clock = [], FakeClock(step=60.0)
    i = ident()
    log = wd.WatchdogLog(tmp_path / "w.jsonl")
    out = wd.watchdog_loop(log, i, wd.SafetySpec(grace_seconds=120.0),
                           **clean_loop_args(
                               tmp_path, identity=i,
                               probe=lambda: sample(swap=99.0),
                               clock=clock, signaller=lambda s, t: sent.append(s),
                               sleep=lambda s: clock.advance(), max_polls=6))
    assert sent == ["SIGTERM", "SIGKILL"] and out["sigkilled"]


def test_no_sigkill_before_the_grace_period(tmp_path):
    sent, clock = [], FakeClock(step=10.0)
    i = ident()
    log = wd.WatchdogLog(tmp_path / "w.jsonl")
    wd.watchdog_loop(log, i, wd.SafetySpec(grace_seconds=120.0),
                     **clean_loop_args(tmp_path, identity=i,
                                       probe=lambda: sample(swap=99.0),
                                       clock=clock,
                                       signaller=lambda s, t: sent.append(s),
                                       sleep=lambda s: clock.advance(),
                                       max_polls=5))
    assert sent == ["SIGTERM"]


# ===========================================================================
# Watchdog log: structure
# ===========================================================================


def write_log(tmp_path, n=3, name="w.jsonl", spec=None):
    log = wd.WatchdogLog(tmp_path / name)
    state = wd.SafetyState(spec=spec or wd.SafetySpec())
    for i in range(n):
        s = sample()
        v = state.observe(s, float(i * 5))
        log.append({"monotonic": float(i * 5), "wall_clock": "Z", **s,
                    "failed": v["failed"], "failure_streak": state.failure_streak,
                    "violations": v["violations"], "action": "poll",
                    "progress": None, "heartbeat_seq": i})
    return log, log.seal()


def test_log_is_created_exclusively(tmp_path):
    write_log(tmp_path)
    with pytest.raises(FileExistsError):
        wd.WatchdogLog(tmp_path / "w.jsonl")


def test_a_clean_log_replays(tmp_path):
    _, sha = write_log(tmp_path)
    out = wd.replay_watchdog_log(tmp_path / "w.jsonl", expected_sha256=sha)
    assert out["problems"] == [] and len(out["records"]) == 3


def test_a_truncated_last_line_is_rejected_not_skipped(tmp_path):
    write_log(tmp_path)
    p = tmp_path / "w.jsonl"
    p.write_text(p.read_text()[:-12])
    assert wd.replay_watchdog_log(p)["problems"]


def test_a_rewritten_record_breaks_the_chain(tmp_path):
    write_log(tmp_path)
    p = tmp_path / "w.jsonl"
    lines = p.read_text().splitlines()
    body = json.loads(lines[1])
    body["swap_used_gb"] = 99.0
    lines[1] = json.dumps(body, ensure_ascii=False, sort_keys=True)
    p.write_text("\n".join(lines) + "\n")
    assert any("chain is broken" in x for x in wd.replay_watchdog_log(p)["problems"])


def test_a_deleted_record_is_rejected(tmp_path):
    write_log(tmp_path)
    p = tmp_path / "w.jsonl"
    lines = p.read_text().splitlines()
    del lines[1]
    p.write_text("\n".join(lines) + "\n")
    assert wd.replay_watchdog_log(p)["problems"]


def test_an_inserted_record_is_rejected(tmp_path):
    write_log(tmp_path)
    p = tmp_path / "w.jsonl"
    lines = p.read_text().splitlines()
    lines.insert(1, json.dumps({"seq": 9, "prev_digest": "x", "action": "poll"}))
    p.write_text("\n".join(lines) + "\n")
    assert wd.replay_watchdog_log(p)["problems"]


def test_duplicate_and_out_of_order_seq_are_rejected(tmp_path):
    write_log(tmp_path)
    p = tmp_path / "w.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines + [lines[-1]]) + "\n")
    assert any("duplicate seq" in x for x in wd.replay_watchdog_log(p)["problems"])
    lines[0], lines[2] = lines[2], lines[0]
    p.write_text("\n".join(lines) + "\n")
    assert any("out of seq order" in x
               for x in wd.replay_watchdog_log(p)["problems"])


def test_appending_after_the_seal_breaks_the_whole_file_digest(tmp_path):
    _, sha = write_log(tmp_path)
    p = tmp_path / "w.jsonl"
    with p.open("a") as fh:
        fh.write(json.dumps({"seq": 3, "prev_digest": None}) + "\n")
    assert any("changed after it" in x
               for x in wd.replay_watchdog_log(p, expected_sha256=sha)["problems"])


def test_a_sealed_log_refuses_further_appends(tmp_path):
    log, _ = write_log(tmp_path)
    with pytest.raises(RuntimeError, match="sealed"):
        log.append({"action": "poll"})


# ===========================================================================
# Watchdog log: semantics
# ===========================================================================


def semantic_records(samples, *, spec=None, action_at=None, sigterm=True,
                     grace=120.0, progress=None):
    spec = spec or wd.SafetySpec()
    state = wd.SafetyState(spec=spec)
    records, prev, seq = [], None, 0

    def add(rec):
        nonlocal prev, seq
        row = {"seq": seq, "prev_digest": prev, **rec}
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        prev = hashlib.sha256(line.encode()).hexdigest()
        seq += 1
        records.append(row)

    t = 0.0
    tripped_at = None
    for s in samples:
        v = state.observe(s, t, progress=progress)
        add({"monotonic": t, "wall_clock": "Z",
             **(s or {"swap_used_gb": None, "memory_pressure_percent_free": None,
                      "free_plus_inactive_gb": None}),
             "failed": v["failed"], "failure_streak": state.failure_streak,
             "violations": v["violations"], "action": "poll",
             "progress": progress, "heartbeat_seq": seq})
        if v["reason"] and tripped_at is None:
            tripped_at = t
            if sigterm:
                add({"monotonic": t, "wall_clock": "Z",
                     "swap_used_gb": None, "memory_pressure_percent_free": None,
                     "free_plus_inactive_gb": None, "failed": None,
                     "failure_streak": state.failure_streak,
                     "violations": v["violations"], "action": "sigterm",
                     "progress": progress, "heartbeat_seq": seq})
        t += 5.0
    return records, tripped_at


def test_semantics_accept_a_run_that_stopped_when_it_should():
    recs, _ = semantic_records([sample(), sample(swap=99.0)])
    out = wd.replay_watchdog_semantics(recs, wd.SafetySpec(),
                                       claimed_reason="swap_used_gb")
    assert out["problems"] == [] and out["expected_reason"] == "swap_used_gb"


def test_semantics_reject_a_run_that_should_have_stopped_and_did_not():
    recs, _ = semantic_records([sample(), sample(swap=99.0)], sigterm=False)
    out = wd.replay_watchdog_semantics(recs, wd.SafetySpec())
    assert any("should have stopped" in p for p in out["problems"])


def test_semantics_reject_a_stop_that_no_threshold_supports():
    recs, _ = semantic_records([sample(), sample()])
    recs.append({"seq": len(recs), "prev_digest": recs[-1]["prev_digest"],
                 "monotonic": 99.0, "wall_clock": "Z", "action": "sigterm",
                 "failed": None, "failure_streak": 0,
                 "violations": {"memory_pressure": 0, "free_plus_inactive": 0,
                                "slow_row": 0},
                 "swap_used_gb": 0.0, "memory_pressure_percent_free": 60,
                 "free_plus_inactive_gb": 5.0, "progress": None,
                 "heartbeat_seq": 0})
    out = wd.replay_watchdog_semantics(recs, wd.SafetySpec())
    assert any("should not have been" in p for p in out["problems"])


def test_semantics_reject_a_tampered_streak_counter():
    recs, _ = semantic_records([sample(pressure=4)] * 2)
    recs[1]["violations"]["memory_pressure"] = 99
    out = wd.replay_watchdog_semantics(recs, wd.SafetySpec())
    assert any("streak" in p for p in out["problems"])


def test_semantics_reject_a_claimed_reason_the_numbers_do_not_support():
    recs, _ = semantic_records([sample(), sample(swap=99.0)])
    out = wd.replay_watchdog_semantics(recs, wd.SafetySpec(),
                                       claimed_reason="free_plus_inactive_gb")
    assert any("replay trips" in p for p in out["problems"])


def test_semantics_reject_time_going_backwards():
    recs, _ = semantic_records([sample(), sample()])
    recs[1]["monotonic"] = -5.0
    assert any("backwards in time" in p for p in
               wd.replay_watchdog_semantics(recs, wd.SafetySpec())["problems"])


def test_semantics_reject_a_non_finite_number():
    recs, _ = semantic_records([sample(), sample()])
    recs[1]["swap_used_gb"] = float("inf")
    assert any("non-finite" in p for p in
               wd.replay_watchdog_semantics(recs, wd.SafetySpec())["problems"])


def test_semantics_reject_an_unknown_action():
    recs, _ = semantic_records([sample()])
    recs[0]["action"] = "explode"
    assert any("outside" in p for p in
               wd.replay_watchdog_semantics(recs, wd.SafetySpec())["problems"])


def test_semantics_reject_unknown_fields():
    recs, _ = semantic_records([sample()])
    recs[0]["surprise"] = 1
    assert any("unknown fields" in p for p in
               wd.replay_watchdog_semantics(recs, wd.SafetySpec())["problems"])


def test_semantics_reject_a_sigkill_inside_the_grace_period():
    recs, _ = semantic_records([sample(), sample(swap=99.0)])
    last = recs[-1]
    recs.append({**last, "seq": last["seq"] + 1, "action": "sigkill",
                 "monotonic": last["monotonic"] + 5.0})
    out = wd.replay_watchdog_semantics(recs, wd.SafetySpec())
    assert any("inside the" in p and "grace" in p for p in out["problems"])


def test_semantics_reject_a_sigkill_with_no_sigterm():
    recs, _ = semantic_records([sample()])
    last = recs[-1]
    recs.append({**last, "seq": last["seq"] + 1, "action": "sigkill",
                 "monotonic": 500.0})
    assert any("without a preceding SIGTERM" in p for p in
               wd.replay_watchdog_semantics(recs, wd.SafetySpec())["problems"])


def test_semantics_check_the_stop_request_nonce_and_reason():
    recs, _ = semantic_records([sample(), sample(swap=99.0)])
    out = wd.replay_watchdog_semantics(
        recs, wd.SafetySpec(), nonce="N",
        stop_request={"nonce": "OTHER", "reason": "swap_used_gb"})
    assert any("another launch's nonce" in p for p in out["problems"])
    out2 = wd.replay_watchdog_semantics(
        recs, wd.SafetySpec(), nonce="N",
        stop_request={"nonce": "N", "reason": "free_plus_inactive_gb"})
    assert any("stop request says" in p for p in out2["problems"])


def test_semantics_count_heartbeats():
    recs, _ = semantic_records([sample()] * 3)
    assert wd.replay_watchdog_semantics(recs, wd.SafetySpec())["heartbeats"] == 3


# ===========================================================================
# Metrics, zero semantics, contract, clocks
# ===========================================================================


@pytest.mark.parametrize("n,d100,d20,dmax", [
    (0, False, False, False), (1, False, False, False),
    (39, False, False, False), (40, False, True, False),
    (119, False, True, False), (120, False, True, True),
    (199, False, True, True), (200, True, True, True),
    (399, True, True, True), (400, True, True, True),
    (500, True, True, True),
])
def test_metric_availability_boundaries(n, d100, d20, dmax):
    m = lr.compute_metrics(make_rows(n))
    assert (m["D100"] is not None) is d100
    assert (m["D20"] is not None) is d20
    assert (m["Dmax"] is not None) is dmax
    for key, ok in (("D100", d100), ("D20", d20), ("Dmax", dmax)):
        if not ok:
            assert m[key] is None and m[f"{key}_reason"].startswith("not_applicable:")
        else:
            assert m[f"{key}_reason"] is None


def test_zero_is_a_legitimate_measurement():
    """A tail that really measured zero reports 0.0, not null."""
    rows = make_rows(200, seconds=1.0)
    for r in rows[100:]:
        r["compute_seconds"] = 0.0
    m = lr.compute_metrics(rows)
    assert m["D100"] == 0.0 and m["D100_reason"] is None


def test_only_an_undefined_ratio_becomes_null():
    rows = make_rows(200, seconds=0.0)
    m = lr.compute_metrics(rows)
    assert m["D100"] is None
    assert "measured zero seconds" in m["D100_reason"]


def test_a_stopped_run_records_the_stop_as_the_reason():
    m = lr.compute_metrics(make_rows(37), stop_reason="swap_used_gb")
    assert "swap_used_gb at row 37" in m["D100_reason"]


def test_d100_is_the_last_hundred_over_the_first_hundred():
    rows = make_rows(200, seconds=1.0, tail_seconds=2.0, tail_from=101)
    assert lr.compute_metrics(rows)["D100"] == pytest.approx(2.0)


def test_the_schedule_is_bounded_by_k_not_by_two_thousand():
    assert lr.expected_clear_rows(500)[-1] == 500
    assert lr.expected_clear_rows(2000)[-1] == 2000


def test_a_clean_contract_has_no_problems():
    r = make_report(500)
    out = lr.clear_contract(r["per_row"], declared_rows=500,
                            recorded_total=r["scheduled_empty_cache_cost"]["total_seconds"],
                            per_call=r["scheduled_empty_cache_cost"]["per_call"],
                            end_to_end_seconds=r["end_to_end_seconds"])
    assert out["problems"] == [] and out["clear_calls"] == 50


def test_a_clear_on_the_wrong_row_is_caught():
    r = make_report(100)
    r["per_row"][14]["cleared"] = True
    r["per_row"][14]["clear_seconds"] = 0.27
    out = lr.clear_contract(r["per_row"], declared_rows=100, recorded_total=None,
                            per_call=None, end_to_end_seconds=1.0)
    assert any("schedule says" in p for p in out["problems"])


def test_clear_growth_needs_forty_calls():
    rows = make_rows(400)
    calls = [{"row": r["row"], "seconds": 0.27} for r in rows if r["cleared"]]
    assert lr.clear_contract(rows, declared_rows=400, recorded_total=None,
                             per_call=calls,
                             end_to_end_seconds=1.0)["clear_growth"] == pytest.approx(1.0)
    rows = make_rows(390)
    calls = [{"row": r["row"], "seconds": 0.27} for r in rows if r["cleared"]]
    assert lr.clear_contract(rows, declared_rows=390, recorded_total=None,
                             per_call=calls,
                             end_to_end_seconds=1.0)["clear_growth"] is None


def test_float_tolerances_separate_identity_from_summation_order():
    assert lr.same_value(1.0, 1.0 + 1e-12)
    assert not lr.same_value(1.0, 1.0 + 1e-6)
    assert lr.same_sum(sum([0.1] * 30), sum(reversed([0.1] * 30)))


def test_clock_nesting():
    assert lr.clock_nesting_problems({"model_load_seconds": 2.0,
                                      "condition_clock_seconds": 100.0,
                                      "process_clock_seconds": 103.0}) == []
    assert lr.clock_nesting_problems({"model_load_seconds": 2.0,
                                      "condition_clock_seconds": 200.0,
                                      "process_clock_seconds": 100.0})
    assert any("belongs to the parent" in x for x in lr.clock_nesting_problems(
        {"model_load_seconds": 1.0, "condition_clock_seconds": 10.0,
         "process_clock_seconds": 12.0, "gate_wait_seconds": 300.0}))
    assert any("not a finite" in x for x in lr.clock_nesting_problems(
        {"model_load_seconds": float("nan"), "condition_clock_seconds": 1.0,
         "process_clock_seconds": 2.0}))


# ===========================================================================
# replay_child, both directions
# ===========================================================================


def test_replay_of_a_clean_report_has_no_problems():
    assert lr.replay_child(make_report(500, with_metrics=True))["problems"] == []


def test_replay_rejects_a_missing_required_field():
    r = make_report(500)
    del r["provenance"]
    assert any("missing 'provenance'" in p for p in lr.replay_child(r)["problems"])


def test_replay_rejects_stored_metrics_that_omit_a_key():
    r = make_report(500, with_metrics=True)
    del r["metrics"]["D100"]
    assert any("omit D100" in p for p in lr.replay_child(r)["problems"])


def test_replay_rejects_stored_metrics_with_extra_keys():
    r = make_report(500, with_metrics=True)
    r["metrics"]["D999"] = 1.0
    assert any("unexpected keys" in p for p in lr.replay_child(r)["problems"])


def test_replay_rejects_a_computable_metric_stored_as_null():
    r = make_report(500, with_metrics=True)
    r["metrics"]["D100"] = None
    assert any("may not be recorded as missing" in p
               for p in lr.replay_child(r)["problems"])


def test_replay_rejects_an_uncomputable_metric_stored_with_a_value():
    r = make_report(50, with_metrics=True)
    r["metrics"]["D100"] = 1.0
    assert any("cannot be recomputed" in p for p in lr.replay_child(r)["problems"])


def test_replay_rejects_a_reason_that_does_not_match():
    r = make_report(50, with_metrics=True)
    r["metrics"]["D100_reason"] = "not_applicable: because I said so"
    assert any("_reason is" in p for p in lr.replay_child(r)["problems"])


def test_replay_rejects_a_computable_metric_carrying_a_reason():
    r = make_report(500, with_metrics=True)
    r["metrics"]["D100_reason"] = "not_applicable: nope"
    assert any("yet carries the reason" in p
               for p in lr.replay_child(r)["problems"])


def test_replay_accepts_a_metric_that_genuinely_measured_zero():
    r = make_report(200)
    for row in r["per_row"][100:]:
        row["compute_seconds"] = 0.0
    r["end_to_end_seconds"] = sum(x["end_to_end_seconds"] for x in r["per_row"])
    compute = sum(x["compute_seconds"] for x in r["per_row"])
    r["model_compute_seconds"] = compute
    clear_total = sum(x["clear_seconds"] or 0.0 for x in r["per_row"])
    r["between_row_overhead_breakdown"] = {
        "scheduled_empty_cache_seconds": clear_total,
        "memory_probe_seconds": 0.0,
        "unattributed_seconds": max(
            0.0, r["end_to_end_seconds"] - compute - clear_total)}
    r["clocks"] = {"model_load_seconds": 2.2, "condition_clock_seconds": compute,
                   "process_clock_seconds": compute + 3.2}
    m = lr.compute_metrics(r["per_row"])
    r["metrics"] = {k: m.get(k) for k in
                    ("D100", "D100_reason", "D20", "D20_reason", "Dmax",
                     "Dmax_reason")}
    assert r["metrics"]["D100"] == 0.0
    assert lr.replay_child(r)["problems"] == []


def test_replay_rejects_non_contiguous_rows():
    r = make_report(50)
    r["per_row"][10]["row"] = 999
    assert any("contiguous" in p for p in lr.replay_child(r)["problems"])


def test_replay_rejects_a_non_finite_row_time():
    r = make_report(50)
    r["per_row"][3]["compute_seconds"] = float("nan")
    assert any("not a finite number" in p for p in lr.replay_child(r)["problems"])


def test_replay_rejects_a_rows_completed_mismatch():
    r = make_report(50)
    r["rows_completed"] = 49
    assert any("rows_completed is 49" in p for p in lr.replay_child(r)["problems"])


def test_replay_rejects_a_short_run_that_claims_it_was_not_stopped():
    r = make_report(50, declared=500)
    assert any("was not stopped early yet completed" in p
               for p in lr.replay_child(r)["problems"])


def test_replay_rejects_a_digest_mismatch_on_a_complete_run():
    r = make_report(50)
    r["completed_input_digest"] = "z" * 64
    assert any("completed input digest differs" in p
               for p in lr.replay_child(r)["problems"])


def test_replay_rejects_a_clear_seconds_on_an_uncleared_row():
    r = make_report(50)
    r["per_row"][2]["clear_seconds"] = 0.1
    assert any("not cleared yet carries" in p for p in lr.replay_child(r)["problems"])


def test_replay_rejects_a_call_count_that_disagrees():
    r = make_report(50)
    r["scheduled_empty_cache_cost"]["calls"] = 99
    assert any("claims 99 clear calls" in p for p in lr.replay_child(r)["problems"])


def test_replay_rejects_bad_clocks():
    r = make_report(50)
    r["clocks"]["process_clock_seconds"] = 1.0
    assert lr.replay_child(r)["problems"]


def test_replay_rejects_a_claimed_stop_the_watchdog_never_recorded(tmp_path):
    r = make_report(500, with_metrics=False)
    r["stopped_early"] = {"reason": "swap_used_gb", "row": 500}
    _, sha = write_log(tmp_path)
    out = lr.replay_child(r, watchdog_path=tmp_path / "w.jsonl",
                          watchdog_sha256=sha)
    assert any("records no sigterm" in p for p in out["problems"])


# ===========================================================================
# Prefix consistency
# ===========================================================================


def test_prefix_consistency_is_not_applicable_with_one_run():
    assert lr.prefix_consistency([make_report(500)])["verdict"] == "not_applicable"


def test_prefix_consistency_passes_on_true_prefixes():
    a, b = make_report(500, run_id="b1"), make_report(1000, run_id="b2")
    assert lr.prefix_consistency([a, b])["verdict"] == "passed"


def test_prefix_consistency_catches_a_diverging_loss():
    a, b = make_report(500, run_id="b1"), make_report(1000, run_id="b2")
    b["per_row"][3]["loss"] = 0.9
    assert lr.prefix_consistency([a, b])["verdict"] == "failed"


def test_a_missing_loss_is_never_a_pass():
    a, b = make_report(500, run_id="b1"), make_report(1000, run_id="b2")
    b["per_row"][3]["loss"] = None
    out = lr.prefix_consistency([a, b])
    assert out["verdict"] == "failed"
    assert any("cannot be shown to agree" in p for p in out["problems"])


def test_a_non_finite_loss_is_never_a_pass():
    a, b = make_report(500, run_id="b1"), make_report(1000, run_id="b2")
    b["per_row"][3]["loss"] = float("nan")
    assert lr.prefix_consistency([a, b])["verdict"] == "failed"


# ===========================================================================
# Verdicts, plan aggregation, R1, outputs
# ===========================================================================


def test_a_tool_failure_produces_no_verdict():
    assert lr.q1_run({}, tool_failure="watchdog_died")["verdict"] is None
    assert lr.q2_run({}, tool_failure="watchdog_died")["verdict"] is None


def test_a_real_safety_trip_makes_q1_fail_even_with_null_metrics():
    out = lr.q1_run(lr.compute_metrics(make_rows(37)),
                    safety_reason="slow_row_seconds")
    assert out["verdict"] == "fails" and out["reason"] == "safety_stop:slow_row_seconds"


def test_a_tool_failure_arriving_as_a_safety_reason_is_still_not_fails():
    out = lr.q1_run(lr.compute_metrics(make_rows(500)),
                    safety_reason="probe_unavailable")
    assert out["verdict"] is None and out["reason"].startswith("tool_failure:")


def test_q1_bands():
    assert lr.q1_run(lr.compute_metrics(make_rows(500)))["verdict"] == "holds"
    assert lr.q1_run(lr.compute_metrics(
        make_rows(500, tail_seconds=5.0, tail_from=401)))["verdict"] == "fails"
    assert lr.q1_run(lr.compute_metrics(
        make_rows(500, tail_seconds=2.0, tail_from=401)))["verdict"] == "indeterminate"
    assert lr.q1_run(lr.compute_metrics(make_rows(50)))["verdict"] == "not_applicable"


def test_q2_keeps_a_real_value_on_a_safety_trip_but_marks_it_descriptive():
    rows = make_rows(500)
    calls = [{"row": r["row"], "seconds": 0.27} for r in rows if r["cleared"]]
    c = lr.clear_contract(rows, declared_rows=500, recorded_total=None,
                          per_call=calls, end_to_end_seconds=1e6)
    out = lr.q2_run(c, safety_reason="swap_used_gb")
    assert out["verdict"] == "stable" and out["descriptive_only"] is True


LEN = {"b1": 500, "b2": 1000, "b3": 2000}


def test_q1_plan_enum_is_closed():
    for verdicts, expected in [
        ({"b1": "holds", "b2": "holds", "b3": "holds"}, "holds_to_2000"),
        ({"b1": "holds", "b2": "holds", "b3": "indeterminate"}, "holds_to_1000"),
        ({"b1": "holds", "b2": "indeterminate", "b3": None}, "holds_to_500"),
        ({"b1": "fails", "b2": None, "b3": None}, "fails"),
        ({"b1": "indeterminate", "b2": None, "b3": None}, "indeterminate"),
        ({"b1": None, "b2": None, "b3": None}, "not_applicable"),
    ]:
        out = lr.q1_plan(verdicts, LEN)
        assert out["value"] == expected and out["value"] in lr.Q1_PLAN_VALUES


def test_q2_plan_only_aggregates_computable_runs():
    assert lr.q2_plan({"b1": "not_applicable"})["value"] == "not_applicable"
    out = lr.q2_plan({"b1": "not_applicable", "b2": "stable", "b3": "stable"})
    assert out["value"] == "stable" and out["contributing"] == 2


@pytest.mark.parametrize("tool", ["probe_unavailable", "watchdog_died",
                                  "heartbeat_gap", "heartbeat_never_arrived",
                                  "identity_mismatch", "stop_request_rejected"])
def test_r1_never_fires_on_a_tool_failure(tool):
    out = lr.r1_should_cancel(outcome="completed", tool_failure=tool,
                              q1_verdict=None, q1_reason=f"tool_failure: {tool}")
    assert not out["cancel"] and "no Q1 verdict" in out["why"]


@pytest.mark.parametrize("outcome", ["nonzero_exit", "timed_out", "no_report"])
def test_r1_never_fires_on_a_non_completed_outcome(outcome):
    assert not lr.r1_should_cancel(outcome=outcome, tool_failure=None,
                                   q1_verdict="fails",
                                   q1_reason="safety_stop:swap_used_gb")["cancel"]


def test_r1_fires_on_a_real_safety_trip_and_on_a_metric_failure():
    assert lr.r1_should_cancel(outcome="completed", tool_failure=None,
                               q1_verdict="fails",
                               q1_reason="safety_stop:swap_used_gb")["cancel"]
    assert lr.r1_should_cancel(outcome="completed", tool_failure=None,
                               q1_verdict="fails",
                               q1_reason="D100=4.2000 >= 3.0")["cancel"]


def full_b_args(**over):
    runs = [{"run_id": r, "outcome": "completed", "tool_failure": None,
             "safety_reason": None} for r in lr.RUN_IDS]
    base = dict(runs=runs, boots=["a", "b", "c"],
                prefix={"verdict": "passed", "problems": []},
                q1={"value": "holds_to_2000"}, q2={"value": "stable"},
                cross_run_identical=True, contract_problems=[])
    base.update(over)
    return base


def test_headline_full_b_needs_everything():
    assert lr.headline_full_B(**full_b_args())["allowed"]
    for over in (dict(boots=["a", "a", "b"]),
                 dict(prefix={"verdict": "not_applicable", "problems": []}),
                 dict(q1={"value": "holds_to_1000"}),
                 dict(q2={"value": "scales"}),
                 dict(cross_run_identical=False),
                 dict(contract_problems=["x"])):
        assert not lr.headline_full_B(**full_b_args(**over))["allowed"]


def test_headline_full_b_refuses_tool_failures_and_safety_trips():
    runs = [{"run_id": r, "outcome": "completed", "tool_failure": None,
             "safety_reason": None} for r in lr.RUN_IDS]
    runs[1]["tool_failure"] = "probe_unavailable"
    assert not lr.headline_full_B(**full_b_args(runs=runs))["allowed"]
    runs[1]["tool_failure"] = None
    runs[0]["safety_reason"] = "swap_used_gb"
    assert not lr.headline_full_B(**full_b_args(runs=runs))["allowed"]


def test_early_stop_finding_never_claims_a_passed_prefix():
    m = lr.compute_metrics(make_rows(137), stop_reason="swap_used_gb")
    out = lr.early_stop_finding(run_id="b1", k=500, metrics=m,
                                q1={"verdict": "fails",
                                    "reason": "safety_stop:swap_used_gb"},
                                q2={"verdict": "not_applicable"})
    assert out["prefix_consistency"] == "not_applicable"
    assert out["adoption_supported"] is False and out["metrics"]["D100"] is None


def test_early_stop_finding_refuses_a_tool_failure_reason():
    with pytest.raises(ValueError, match="real safety trip"):
        lr.early_stop_finding(run_id="b1", k=500,
                              metrics=lr.compute_metrics(make_rows(137)),
                              q1={"verdict": None,
                                  "reason": "tool_failure: probe_unavailable"},
                              q2={"verdict": None})


def test_early_stop_finding_refuses_a_value_next_to_a_not_applicable_reason():
    m = lr.compute_metrics(make_rows(137), stop_reason="swap_used_gb")
    m["D100"] = 0.0   # a value beside a reason that says it is uncomputable
    with pytest.raises(ValueError, match="must be null"):
        lr.early_stop_finding(run_id="b1", k=500, metrics=m,
                              q1={"verdict": "fails",
                                  "reason": "safety_stop:swap_used_gb"},
                              q2={"verdict": "not_applicable"})


# ===========================================================================
# verify and from-json
# ===========================================================================


GOOD_SAMPLE = {"swap_used_gb": 0.0, "memory_pressure_percent_free": 95,
               "free_plus_inactive_gb": 8.0, "normalized_load_1m": 0.1}


def gate_record(thresholds, n=3):
    from src.training.preflight import evaluate_gate
    polls, streak = [], 0
    for i in range(n):
        judged = evaluate_gate(GOOD_SAMPLE, thresholds)
        streak = streak + 1 if judged["passed"] else 0
        polls.append({"index": i, "elapsed_seconds": i * 30.0,
                      "sample": dict(GOOD_SAMPLE), "passed": judged["passed"],
                      "streak": streak})
    return {"passed": True, "polls": polls,
            "waited_seconds": polls[-1]["elapsed_seconds"]}


def stop_block(*, row, reason="swap_used_gb", condition=1.0, process=2.0):
    """The whole of section 4.7's stopped_early, the way a real child writes it."""
    return {"reason": reason, "rule": reason, "row": row,
            "sampled_values": sample(swap=99.0),
            "condition_clock_seconds": condition,
            "process_clock_seconds": process,
            "requested_by": "watchdog", "stop_request_sha256": "s" * 64}


def place_run(paths, plan, run, k, *, boot, stop=None, seconds=1.0,
              tail_seconds=None, tail_from=None, outcome="completed",
              watchdog_records=3, safety_trip=False, terminal=True):
    """Write one run's evidence and its two journal events, properly chained."""
    nonce = wd.new_nonce()
    entry = next(e for e in plan["runs"] if e["run_id"] == run)
    rep = make_report(k, run_id=run, declared=k, with_metrics=True,
                      stopped=stop, seconds=seconds, tail_seconds=tail_seconds,
                      tail_from=tail_from, nonce=nonce,
                      experiment_id=plan["experiment_id"])
    rep["plan_digest"] = plan["plan_digest"]
    rep["condition"] = entry["condition"]
    rep["provenance"] = fake_provenance(k)
    rep["child_source_check"] = real_source_check(paths, plan)

    wd.write_launch_record(paths["dir"], prefix=run,
                           identity=wd.ChildIdentity(pid=1, pgid=1, nonce=nonce,
                                                     start_identity="T0"),
                           experiment_id=plan["experiment_id"], run_id=run)
    copy_identity(rep, paths, run)
    rp = paths["dir"] / f"{run}.json"
    rp.write_text(json.dumps(rep))
    log = wd.WatchdogLog(paths["dir"] / f"{run}.watchdog.jsonl")
    log.arm(launch_identity(paths, run))
    state = wd.SafetyState()
    samples = [sample() for _ in range(watchdog_records)]
    if safety_trip:
        samples[-1] = sample(swap=99.0)
    tripped = False
    for i, s in enumerate(samples):
        v = state.observe(s, float(i * 5))
        log.append({"monotonic": float(i * 5), "wall_clock": "Z", **s,
                    "failed": v["failed"], "failure_streak": state.failure_streak,
                    "violations": v["violations"], "action": "poll",
                    "progress": None, "heartbeat_seq": i})
        if v["reason"] and not tripped:
            tripped = True
            wd.write_stop_request(paths["dir"], prefix=run, reason=v["reason"],
                                  rule=v["reason"], nonce=nonce,
                                  monotonic=float(i * 5), wall_clock="Z",
                                  sampled_values=s)
            log.append({"monotonic": float(i * 5), "wall_clock": "Z", **s,
                        "failed": None, "failure_streak": state.failure_streak,
                        "violations": v["violations"], "action": "sigterm",
                        "progress": None, "heartbeat_seq": i})
    if terminal:
        log.append({"monotonic": float(len(samples) * 5), "wall_clock": "Z",
                    "swap_used_gb": None, "memory_pressure_percent_free": None,
                    "free_plus_inactive_gb": None, "failed": None,
                    "failure_streak": state.failure_streak,
                    "violations": v["violations"],
                    "action": wd.CHILD_EXIT_OBSERVED, "progress": None,
                    "heartbeat_seq": len(samples)})
    wsha = log.seal()

    # A real child cites the digest of the stop request it authenticated, so
    # the fake has to as well: a placeholder here would let the report claim a
    # stop that the request on disk does not support.
    sp = paths["dir"] / f"{run}.stop_request.json"
    if isinstance(rep.get("stopped_early"), dict) and sp.exists():
        rep["stopped_early"]["stop_request_sha256"] = hashlib.sha256(
            sp.read_bytes()).hexdigest()
        rp.write_text(json.dumps(rep))

    started = lr.append_session_event(paths, run, lr.EVENT_STARTED, {
        "declared_rows": k, "condition": entry["condition"], "nonce": nonce,
        "boot_fingerprint": boot, "gate": gate_record(THRESHOLDS),
        "started_at": lr.now_iso()}, plan=plan)
    lr._finish(paths, run, plan, boot, outcome=outcome, exit_status=0,
               report_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),
               watchdog_sha256=wsha, started_event=started)
    return {"nonce": nonce, "report": rep, "watchdog_sha256": wsha}


THRESHOLDS = None


def finished_session(tmp_path, *, rows=(500, 1000, 2000)):
    global THRESHOLDS
    out = init(tmp_path)
    THRESHOLDS = json.loads(out["paths"]["calibration"].read_text())["thresholds"]
    for i, (run, k) in enumerate(zip(lr.RUN_IDS, rows)):
        place_run(out["paths"], out["plan"], run, k, boot=f"boot-{i}")
    return out


def stopped_session(tmp_path):
    """b1 hits a real safety threshold and stops early."""
    global THRESHOLDS
    out = init(tmp_path)
    THRESHOLDS = json.loads(out["paths"]["calibration"].read_text())["thresholds"]
    place_run(out["paths"], out["plan"], "b1", 500, boot="boot-0",
              stop=stop_block(row=500), tail_seconds=5.0, tail_from=401,
              safety_trip=True)
    return out


def test_verify_passes_on_a_clean_session(tmp_path):
    finished_session(tmp_path)
    out = lr.verify_experiment("exp016a", root=tmp_path / "reports")
    assert out["problems"] == []


def test_verify_catches_a_tampered_child_report(tmp_path):
    out = finished_session(tmp_path)
    rp = out["paths"]["dir"] / "b1.json"
    rep = json.loads(rp.read_text())
    rep["metrics"]["D100"] = 9.9
    rp.write_text(json.dumps(rep))
    problems = lr.verify_experiment("exp016a",
                                    root=tmp_path / "reports")["problems"]
    assert any("does not match the digest" in p for p in problems)


def test_verify_catches_a_shared_boot(tmp_path):
    out = finished_session(tmp_path)
    events = sorted((out["paths"]["dir"] / "events").glob(
        "*measurement_started.json"))
    body = json.loads(events[1].read_text())
    body["boot_fingerprint"] = "boot-0"
    events[1].write_text(json.dumps(body))
    problems = lr.verify_experiment("exp016a",
                                    root=tmp_path / "reports")["problems"]
    assert any("share a boot" in p for p in problems)


def test_verify_catches_a_broken_source_snapshot(tmp_path):
    out = finished_session(tmp_path)
    snap = next(out["paths"]["snapshot"].glob("*"))
    snap.write_text("tampered")
    assert lr.verify_experiment("exp016a", root=tmp_path / "reports")["problems"]


def test_verify_catches_thresholds_that_do_not_recompute(tmp_path):
    out = finished_session(tmp_path)
    calib = json.loads(out["paths"]["calibration"].read_text())
    calib["thresholds"]["swap_used_gb"] = 99.0
    out["paths"]["calibration"].write_text(json.dumps(calib))
    problems = lr.verify_experiment("exp016a",
                                    root=tmp_path / "reports")["problems"]
    assert any("calibration.json has changed" in p for p in problems)


def test_from_json_refuses_to_render_when_verification_fails(tmp_path):
    out = finished_session(tmp_path)
    snap = next(out["paths"]["snapshot"].glob("*"))
    snap.write_text("tampered")
    rendered = lr.render_from_json("exp016a", root=tmp_path / "reports")
    assert rendered["rendered"] is False
    assert "verification failed" in rendered["reason"]


def test_from_json_is_not_an_alias_for_verify(tmp_path):
    finished_session(tmp_path)
    fin = lr.session_finalize("exp016a", root=tmp_path / "reports")
    assert fin["problems"] == []
    rendered = lr.render_from_json("exp016a", root=tmp_path / "reports")
    assert rendered["rendered"] is True and "Q1_plan" in rendered["markdown"]


def test_finalize_is_idempotent(tmp_path):
    finished_session(tmp_path)
    a = lr.session_finalize("exp016a", root=tmp_path / "reports")
    b = lr.session_finalize("exp016a", root=tmp_path / "reports")
    assert b.get("idempotent") and a["aggregate"]["headline"] == \
        b["aggregate"]["headline"]


def test_verify_catches_an_aggregate_that_disagrees_with_the_journal(tmp_path):
    finished_session(tmp_path)
    lr.session_finalize("exp016a", root=tmp_path / "reports")
    paths = lr.session_paths("exp016a", root=tmp_path / "reports")
    agg = json.loads(paths["aggregate"].read_text())
    assert agg["Q1_plan"]["value"] == "holds_to_2000"
    agg["Q1_plan"]["value"] = "holds_to_1000"   # a claim the journal denies
    paths["aggregate"].write_text(json.dumps(agg))
    problems = lr.verify_experiment("exp016a",
                                    root=tmp_path / "reports")["problems"]
    assert any("fresh derivation" in p for p in problems)


def test_a_clean_three_run_session_allows_the_headline(tmp_path):
    finished_session(tmp_path)
    out = lr.session_finalize("exp016a", root=tmp_path / "reports")
    agg = out["aggregate"]
    assert agg["Q1_plan"]["value"] == "holds_to_2000"
    assert agg["headline"]["allowed"] is True
    assert agg["prefix_consistency"]["verdict"] == "passed"


def test_r1_cancels_and_the_headline_is_refused(tmp_path):
    out = stopped_session(tmp_path)
    fired = lr.apply_rule_r1("exp016a", root=sess(tmp_path))
    assert fired["fired"] and fired["cancelled"] == ["b2", "b3"]
    fin = lr.session_finalize("exp016a", root=sess(tmp_path))
    agg = fin["aggregate"]
    assert agg["state"] == "complete_by_rule"
    assert agg["headline"]["allowed"] is False
    assert agg["prefix_consistency"]["verdict"] == "not_applicable"
    assert lr.verify_experiment("exp016a", root=sess(tmp_path))["problems"] == []


def test_r1_does_not_fire_after_a_tool_failure(tmp_path):
    out = init(tmp_path)
    paths, plan = out["paths"], out["plan"]
    place_run(paths, plan, "b1", 500, boot="boot-0", tail_seconds=5.0,
              tail_from=401)
    # Rewrite the finished event's outcome the way a tool failure would.
    fin = [e for e in events_of(paths) if "measurement_finished" in e.name][0]
    body = json.loads(fin.read_text())
    body["tool_failure"] = "watchdog_died"
    fin.write_text(json.dumps(body))
    assert not lr.apply_rule_r1("exp016a", root=sess(tmp_path))["fired"]


def test_r1_runs_automatically_after_a_finished_run(tmp_path):
    """The rule is part of the plan, so nobody has to remember to invoke it."""
    init(tmp_path)

    out = lr.session_next("exp016a", **next_args(
        tmp_path, collect=collector(tail_seconds=5.0, tail_from=401)))
    assert out["ok"] and out["r1"] and out["r1"]["fired"]
    st = lr.session_state("exp016a", root=sess(tmp_path))
    assert st["cancelled"] == ["b2", "b3"]


# ===========================================================================
# Real subprocesses: process groups, signals, pipes, heartbeats
# ===========================================================================


DUMMY_CHILD = textwrap.dedent('''
    import json, os, signal, sys, time
    sys.path.insert(0, {root!r})
    from src.training import longrun as lr
    from src.training import watchdog as wd

    directory, run, nonce = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = int(sys.argv[4]); ignore_term = sys.argv[5] == "ignore"
    flag = wd.StopFlag()
    signal.signal(signal.SIGTERM, flag.handler)
    progress = os.path.join(directory, run + ".progress.jsonl")
    t0 = time.perf_counter()
    per_row, stopped = [], None
    for i in range(1, rows + 1):
        r0 = time.perf_counter()
        time.sleep(0.005)
        sec = time.perf_counter() - r0
        cleared = i % 10 == 0
        per_row.append({{"row": i, "compute_seconds": sec,
                        "end_to_end_seconds": sec, "sample_id": "s%d" % i,
                        "tokens": 100, "supervised_tokens": 70, "loss": 0.5,
                        "cleared": cleared,
                        "clear_seconds": 0.001 if cleared else None}})
        lr.write_progress(progress, row=i, row_elapsed_seconds=sec,
                          condition_clock_seconds=time.perf_counter() - t0,
                          process_clock_seconds=time.perf_counter() - t0)
        if flag.is_set() and not ignore_term:
            got = wd.read_stop_request(directory, prefix=run, nonce=nonce)
            if got["accepted"]:
                stopped = {{"reason": got["body"]["reason"],
                           "rule": got["body"].get("rule"), "row": i,
                           "sampled_values": got["body"].get("sampled_values"),
                           "condition_clock_seconds": time.perf_counter() - t0,
                           "process_clock_seconds": time.perf_counter() - t0,
                           "requested_by": "watchdog",
                           "stop_request_sha256": got["sha256"]}}
                break
    condition_seconds = time.perf_counter() - t0
    if ignore_term:
        while True:
            time.sleep(0.05)
    compute = sum(r["compute_seconds"] for r in per_row)
    ids = [r["sample_id"] for r in per_row]
    order_ids = ids + ["s%d" % i for i in range(len(per_row) + 1, rows + 1)]
    clear_total = sum(r["clear_seconds"] or 0 for r in per_row)
    prov = {{f: ({{}} if f.endswith(("sha256", "digest", "config", "optimizer",
                                   "packages", "conditions"))
                else "x") for f in lr.PROVENANCE_FIELDS}}
    prov["measurement_intervals"] = {{"window": lr.WINDOW,
                                     "memory_every": lr.MEMORY_EVERY,
                                     "empty_cache_every": lr.EMPTY_CACHE_EVERY,
                                     "max_rows": rows}}
    report = {{"schema_version": lr.CHILD_SCHEMA_VERSION, "kind": lr.CHILD_KIND,
              "experiment_id": os.path.basename(directory.rstrip("/")),
              "run_id": run, "declared_rows": rows, "condition": "empty_cache",
              "plan_digest": "d" * 64, "nonce": nonce, "per_row": per_row,
              "memory": [], "pool_pairs": lr.POOL_PAIRS,
              "pool_rows": lr.POOL_ROWS, "rows_requested": rows,
              "child_source_check": {{"files_verified": 0,
                                     "plan_digest": "d" * 64,
                                     "source_manifest_digest": "m" * 64}},
              "preflight": {{"swap_used_gb": 0.0,
                            "memory_pressure_percent_free": 95,
                            "free_plus_inactive_gb": 8.0,
                            "normalized_load_1m": 0.1}},
              "child_pid": os.getpid(), "child_pgid": os.getpgrp(),
              "child_start_identity": wd.process_start_identity(os.getpid()),
              "started_at": "2026-08-16T00:00:00Z",
              "finished_at": "2026-08-16T00:10:00Z", "tool_failure": None,
              "model_compute_seconds": compute,
              "stopped_early": stopped, "rows_completed": len(per_row),
              "end_to_end_seconds": sum(r["end_to_end_seconds"] for r in per_row),
              "input_order_digest": lr._digest_ids(order_ids),
              "completed_input_digest": lr._digest_ids(ids),
              "provenance": prov,
              "metrics": {{k: lr.compute_metrics(
                  per_row,
                  stop_reason=(stopped or {{}}).get("reason")).get(k)
                  for k in ("D100", "D100_reason", "D20", "D20_reason",
                            "Dmax", "Dmax_reason")}},
              "scheduled_empty_cache_every": lr.EMPTY_CACHE_EVERY,
              "scheduled_empty_cache_cost": {{
                  "calls": sum(1 for r in per_row if r["cleared"]),
                  "total_seconds": clear_total,
                  "per_call": [{{"row": r["row"], "seconds": r["clear_seconds"]}}
                              for r in per_row if r["cleared"]]}},
              "teardown_empty_cache_calls": 1,
              "teardown_empty_cache_seconds": 0.0,
              "between_row_overhead_breakdown": {{
                  "scheduled_empty_cache_seconds": clear_total,
                  "memory_probe_seconds": 0.0,
                  "unattributed_seconds": 0.0}},
              "float_storage": {{"seconds_rounded": False,
                                "loss_rounded": False}},
              "clocks": {{"model_load_seconds": 0.0,
                         "condition_clock_seconds": condition_seconds,
                         "process_clock_seconds": time.perf_counter() - t0}}}}
    with open(os.path.join(directory, run + ".json"), "w") as fh:
        json.dump(report, fh)
''')

DUMMY_WATCHDOG = textwrap.dedent('''
    import json, os, sys, time
    sys.path.insert(0, {root!r})
    from src.training import watchdog as wd

    directory, run = sys.argv[1], sys.argv[2]
    handshake_fd, heartbeat_fd = int(sys.argv[3]), int(sys.argv[4])
    trip_at = int(sys.argv[5])
    spec = wd.SafetySpec(poll_seconds=0.05, grace_seconds=float(sys.argv[6]),
                         poll_max_gap_seconds=1e9)
    os.write(heartbeat_fd, b'{{"ready": true}}\\n')
    reader = wd.LineReader(handshake_fd)
    identity = None
    deadline = time.monotonic() + 10
    while identity is None and time.monotonic() < deadline:
        for msg in reader.poll():
            identity = wd.ChildIdentity.from_dict(msg)
        time.sleep(0.01)
    if identity is None:
        sys.exit(3)
    os.write(heartbeat_fd, b'{{"armed": true}}\\n')
    calls = {{"n": 0}}

    def probe():
        calls["n"] += 1
        swap = 99.0 if calls["n"] >= trip_at else 0.0
        return {{"swap_used_gb": swap, "memory_pressure_percent_free": 60,
                "free_plus_inactive_gb": 5.0}}

    def alive():
        try:
            os.kill(identity.pid, 0)
            return True
        except OSError:
            return False

    log = wd.WatchdogLog(os.path.join(directory, run + ".watchdog.jsonl"))
    out = wd.watchdog_loop(
        log, identity, spec, probe=probe, clock=time.monotonic,
        child_alive=alive,
        signaller=lambda sig, pg: wd._signal_pgid(sig, pg),
        heartbeat=lambda b: os.write(heartbeat_fd,
                                     (json.dumps(b) + "\\n").encode()),
        observed_start=lambda: wd.process_start_identity(identity.pid),
        observed_pgid_fn=lambda: wd.observed_pgid(identity.pid),
        directory=directory, prefix=run,
        read_progress=lambda: wd.read_progress_tail(
            os.path.join(directory, run + ".progress.jsonl")),
        wall_clock=lambda: "", sleep=time.sleep, max_polls=4000)
    sha = log.seal()
    with open(os.path.join(directory, run + ".watchdog.result.json"), "w") as fh:
        json.dump({{"result": out, "sha256": sha}}, fh)
''')


def _run_real(tmp_path, *, rows=60, trip_at=3, ignore_term=False, grace=0.4):
    """Start a real watchdog and a real child, in real process groups."""
    d = tmp_path / "live"
    d.mkdir()
    child_py = d / "child.py"
    child_py.write_text(DUMMY_CHILD.format(root=str(ROOT)))
    wd_py = d / "wd.py"
    wd_py.write_text(DUMMY_WATCHDOG.format(root=str(ROOT)))

    hs_r, hs_w = os.pipe()
    hb_r, hb_w = os.pipe()
    nonce = wd.new_nonce()
    watchdog = subprocess.Popen(
        [sys.executable, str(wd_py), str(d), "b1", str(hs_r), str(hb_w),
         str(trip_at), str(grace)],
        pass_fds=(hs_r, hb_w), start_new_session=True)
    os.close(hs_r)
    os.close(hb_w)
    reader = wd.LineReader(hb_r)

    ready, deadline = False, time.monotonic() + 15
    while not ready and time.monotonic() < deadline:
        for msg in reader.poll():
            if msg.get("ready"):
                ready = True
        time.sleep(0.01)
    assert ready, "watchdog never became ready"

    child = subprocess.Popen(
        [sys.executable, str(child_py), str(d), "b1", nonce, str(rows),
         "ignore" if ignore_term else "obey"],
        start_new_session=True)
    identity = wd.ChildIdentity(
        pid=child.pid, pgid=os.getpgid(child.pid), nonce=nonce,
        start_identity=wd.process_start_identity(child.pid) or "?")
    wd.write_launch_record(d, prefix="b1", identity=identity,
                           experiment_id="exp016a", run_id="b1")
    wd.write_line(hs_w, identity.as_dict())

    armed, deadline = False, time.monotonic() + 15
    beats = []
    while not armed and time.monotonic() < deadline:
        for msg in reader.poll():
            if msg.get("armed"):
                armed = True
            else:
                beats.append(msg)
        time.sleep(0.01)
    assert armed, "watchdog never armed"
    return {"dir": d, "child": child, "watchdog": watchdog, "reader": reader,
            "identity": identity, "hs_w": hs_w, "hb_r": hb_r, "beats": beats}


def _finish_real(live, *, timeout=30):
    child, watchdog, reader = live["child"], live["watchdog"], live["reader"]
    deadline = time.monotonic() + timeout
    beats = live["beats"]
    while time.monotonic() < deadline:
        beats += reader.poll()
        if child.poll() is not None and watchdog.poll() is not None:
            break
        time.sleep(0.02)
    notes = wd.reap(child, watchdog)
    os.close(live["hs_w"])
    os.close(live["hb_r"])
    return {"beats": beats, "notes": notes,
            "child_rc": child.returncode, "watchdog_rc": watchdog.returncode}


def test_integration_real_processes_stop_on_a_real_threshold(tmp_path):
    live = _run_real(tmp_path, rows=400, trip_at=3, grace=5.0)
    out = _finish_real(live)
    d = live["dir"]

    assert (d / "b1.stop_request.json").exists()
    assert (d / "b1.watchdog.jsonl").exists()
    report = json.loads((d / "b1.json").read_text())
    assert report["stopped_early"], "the child should have stopped at a boundary"
    assert report["stopped_early"]["reason"] == "swap_used_gb"
    assert report["rows_completed"] < 400

    got = wd.read_stop_request(d, prefix="b1", nonce=live["identity"].nonce)
    assert got["accepted"]

    structural = wd.replay_watchdog_log(d / "b1.watchdog.jsonl")
    assert structural["problems"] == []
    sem = wd.replay_watchdog_semantics(
        structural["records"],
        wd.SafetySpec(poll_seconds=0.05, grace_seconds=5.0,
                      poll_max_gap_seconds=1e9),
        stop_request=got["body"], nonce=live["identity"].nonce,
        claimed_reason="swap_used_gb")
    assert sem["problems"] == [] and sem["sigterm"] and not sem["sigkill"]
    assert sem["heartbeats"] > 0
    assert len(out["beats"]) > 0, "the parent received real heartbeats"

    replay = lr.replay_child(report, watchdog_path=d / "b1.watchdog.jsonl",
                             spec=wd.SafetySpec(poll_seconds=0.05,
                                                grace_seconds=5.0,
                                                poll_max_gap_seconds=1e9))
    assert [p for p in replay["problems"]
            if "not stopped early" not in p] == []


def test_integration_a_child_that_ignores_sigterm_is_sigkilled(tmp_path):
    live = _run_real(tmp_path, rows=100000, trip_at=3, ignore_term=True,
                     grace=0.4)
    out = _finish_real(live, timeout=40)
    d = live["dir"]
    result = json.loads((d / "b1.watchdog.result.json").read_text())
    assert result["result"]["sigkilled"] is True
    assert not (d / "b1.json").exists(), "a killed child writes no report"
    structural = wd.replay_watchdog_log(d / "b1.watchdog.jsonl",
                                        expected_sha256=result["sha256"])
    assert structural["problems"] == []
    actions = [r["action"] for r in structural["records"]]
    assert "sigterm" in actions and "sigkill" in actions


def test_integration_no_child_is_left_running_after_cleanup(tmp_path):
    live = _run_real(tmp_path, rows=100000, trip_at=10 ** 9, grace=5.0)
    pid = live["child"].pid
    notes = wd.reap(live["child"], live["watchdog"])
    os.close(live["hs_w"])
    os.close(live["hb_r"])
    time.sleep(0.2)
    with pytest.raises(OSError):
        os.kill(pid, 0)
    assert notes


def test_integration_the_child_runs_in_its_own_process_group(tmp_path):
    live = _run_real(tmp_path, rows=100000, trip_at=10 ** 9, grace=5.0)
    try:
        assert os.getpgid(live["child"].pid) == live["child"].pid
        assert os.getpgid(live["child"].pid) != os.getpgid(os.getpid())
    finally:
        wd.reap(live["child"], live["watchdog"])
        os.close(live["hs_w"])
        os.close(live["hb_r"])


def test_integration_a_stale_nonce_is_refused_by_the_child(tmp_path):
    d = tmp_path / "stale"
    d.mkdir()
    wd.write_stop_request(d, prefix="b1", reason="swap_used_gb", rule="r",
                          nonce="OLD-LAUNCH", monotonic=1.0, wall_clock="Z")
    assert not wd.read_stop_request(d, prefix="b1", nonce="NEW")["accepted"]


# ===========================================================================
# Round 13. The execution lock is lifted -- and what has to stay true.
#
# The user approved unlocking the four executing entry points. Approval was
# always specified as the deletion of the refusals and nothing else, so this
# section flips from "the refusal stands" to "the refusal is gone and the
# dispatch behind it is what runs". Everything else must hold unchanged:
# there is still no flag, no environment variable and no hidden switch that
# alters what the tool does, and read-only modes still write nothing.
#
# **Unlocked is not executed.** No test here spawns a child, loads a model or
# creates a session: every command is replaced by a recorder for the length
# of one call, and the entry points are never invoked through the real
# project directory. `data/reports/16_longrun/` still does not exist, and the
# first real run is still the first real verification of any of this.
# ===========================================================================


#: The four entry points that were refused until this round, and the command
#: each one is now allowed to reach.
EXECUTING_FLAGS = [("--session-init", "cmd_session_init"),
                   ("--session-next", "cmd_session_next"),
                   ("--child", "cmd_child"),
                   ("--watchdog-worker", "cmd_watchdog_worker")]


#: The smallest command line each mode accepts. A bare flag stopped being one
#: in round 14, when every mode began checking its own arguments before
#: dispatch -- which is the point of that round, not an inconvenience.
MINIMAL_ARGS = {
    "--session-init": ["--experiment-id", "exp016a",
                       "--calibration", "/nowhere/calibration.json"],
    "--session-next": ["--experiment-id", "exp016a"],
    "--session-finalize": ["--experiment-id", "exp016a"],
    "--verify": ["--experiment-id", "exp016a"],
    "--session-status": ["--experiment-id", "exp016a"],
    "--from-json": ["--experiment-id", "exp016a"],
    "--show-plan": [],
    "--microbenchmark": [],
    "--child": ["--experiment-dir", "/nowhere/exp016a", "--run", "b1",
                "--rows", "500", "--nonce", "n1", "--plan-digest", "d" * 64],
    "--watchdog-worker": ["--experiment-dir", "/nowhere/exp016a", "--run",
                          "b1", "--handshake-fd", "3", "--heartbeat-fd", "4"],
}


def minimal_argv(flag):
    return [flag, *MINIMAL_ARGS[flag]]


@pytest.mark.parametrize("flag,cmd", EXECUTING_FLAGS)
def test_r13_each_executing_flag_reaches_its_own_command(flag, cmd,
                                                         monkeypatch):
    """No refusal left to neutralise: the flag reaches the command on its own.

    The locked-era version of this test had to monkeypatch ``refuse`` away
    first. That it no longer needs to is the whole change.
    """
    mod = load_cli()
    assert not hasattr(mod, "refuse"), "the refusal was supposed to be deleted"
    seen = []
    monkeypatch.setattr(mod, cmd, lambda args: seen.append(cmd) or 0)
    assert mod.main(minimal_argv(flag)) == 0
    assert seen == [cmd]


@pytest.mark.parametrize("flag,cmd", EXECUTING_FLAGS)
def test_r13_one_flag_runs_exactly_one_command(flag, cmd, monkeypatch):
    """Opening four doors must not mean walking through more than one."""
    mod = load_cli()
    seen = []
    for _, name in EXECUTING_FLAGS:
        monkeypatch.setattr(mod, name, lambda args, c=name: seen.append(c) or 0)
    assert mod.main(minimal_argv(flag)) == 0
    assert seen == [cmd]


# `test_r13_an_executing_flag_still_outranks_a_read_only_one` was retracted in
# round 14. It pinned the behaviour that `--session-init --verify` resolves to
# session-init -- which, once the entry points were open, meant
# `--verify --session-next` would spend a boot on a command line that reads
# like a read-only one. Precedence between modes is the wrong answer to that
# question; the modes are now mutually exclusive. See the round 14 section.


def test_r13_the_child_arguments_arrive_unchanged(monkeypatch):
    """The parent builds this argv; the child has to receive what was sent."""
    mod = load_cli()
    got = {}
    monkeypatch.setattr(mod, "cmd_child",
                        lambda args: got.update(vars(args)) or 0)
    assert mod.main(["--child", "--experiment-dir", "/nowhere/exp016a",
                     "--run", "b1", "--rows", "500", "--nonce", "abc123",
                     "--plan-digest", "d" * 64]) == 0
    assert got["experiment_dir"] == "/nowhere/exp016a"
    assert got["run"] == "b1"
    assert got["rows"] == 500
    assert got["nonce"] == "abc123"
    assert got["plan_digest"] == "d" * 64


def test_r13_the_watchdog_arguments_arrive_unchanged(monkeypatch):
    mod = load_cli()
    got = {}
    monkeypatch.setattr(mod, "cmd_watchdog_worker",
                        lambda args: got.update(vars(args)) or 0)
    assert mod.main(["--watchdog-worker", "--experiment-dir", "/nowhere/e",
                     "--run", "b2", "--handshake-fd", "7",
                     "--heartbeat-fd", "9"]) == 0
    assert got["experiment_dir"] == "/nowhere/e" and got["run"] == "b2"
    assert got["handshake_fd"] == 7 and got["heartbeat_fd"] == 9


def test_r13_session_init_receives_the_experiment_and_its_calibration(
        monkeypatch):
    mod = load_cli()
    got = {}
    monkeypatch.setattr(mod, "cmd_session_init",
                        lambda args: got.update(vars(args)) or 0)
    assert mod.main(["--session-init", "--experiment-id", "exp016a",
                     "--calibration", "/nowhere/calibration.json"]) == 0
    assert got["experiment_id"] == "exp016a"
    assert got["calibration"] == "/nowhere/calibration.json"


def test_r13_the_argv_the_parent_builds_matches_what_the_parser_reads():
    """One end of the pipe, then the other: no flag name drifted apart.

    ``_child_argv`` and ``_watchdog_argv`` are what the parent actually
    spawns. Parsing them back with the tool's own parser is the only check
    that survives a rename on one side only.
    """
    mod = load_cli()
    paths = {"dir": Path("/nowhere/exp016a")}
    entry = {"declared_rows": 1000}
    plan = {"plan_digest": "e" * 64}
    argv = mod._child_argv(paths, "b2", entry, "nonce-1", plan)[2:]
    args = mod.build_parser().parse_args(argv)
    assert args.child is True
    assert args.run == "b2" and args.rows == 1000
    assert args.nonce == "nonce-1" and args.plan_digest == "e" * 64

    argv = mod._watchdog_argv(paths, "b3", 11, 12)[2:]
    args = mod.build_parser().parse_args(argv)
    assert args.watchdog_worker is True
    assert args.run == "b3"
    assert args.handshake_fd == 11 and args.heartbeat_fd == 12


# --- what the lock left behind, and what must never come back ---------------


def test_r13_the_refusal_and_its_message_are_gone():
    """Checked in both modules. The design doc claimed both; only one was.

    ``EXECUTION_LOCKED_REASON`` and ``ExecutionLocked`` lived on unused and
    unreferenced in ``src/training/longrun.py`` for six rounds, still saying
    in Chinese that report 16 had not been approved to run.
    """
    for module in (load_cli(), lr):
        for name in ("refuse", "ExecutionLocked", "EXECUTION_LOCKED",
                     "EXECUTION_LOCKED_REASON"):
            assert not hasattr(module, name), (
                f"{name} outlived the lock it served, in "
                f"{module.__name__}")
    for source in (CLI.read_text(), Path(lr.__file__).read_text()):
        assert LOCK_MESSAGE not in source, "the refusal text is still there"
    assert "LOCKED" not in CLI.read_text()


def test_r13_the_help_no_longer_calls_the_entry_points_locked():
    mod = load_cli()
    help_text = mod.build_parser().format_help()
    assert "LOCKED" not in help_text
    assert "never runs" not in help_text
    assert "--session-init" in help_text and "--session-next" in help_text


def test_r13_the_module_docstring_says_unlocked_and_measures_nothing():
    """Round 13's claim was "not executed". exp001 retired that wording.

    What survives the correction is the part that is still true and still
    load-bearing: report 16 holds no measurement. The docstring may not go
    back to saying nothing has been run, and may not start implying that
    something was measured.
    """
    doc = load_cli().__doc__ or ""
    assert "Execution is locked" not in doc
    assert "refused:" not in doc
    assert "has not been executed" not in doc, (
        "exp001 exists and ran; that sentence is no longer true")
    assert "No session exists" not in doc
    assert "has produced no measurement" in doc
    assert "terminal_incomplete" in doc


def test_r13_the_no_session_message_no_longer_says_the_tool_is_locked():
    out = subprocess.run([sys.executable, str(CLI), "--verify",
                          "--experiment-id", "exp016a"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 1
    assert "no report 16 session" in out.stdout
    assert "locked" not in out.stdout.lower()
    assert "--session-init" in out.stdout, "say what would create one"


def test_r13_unlocking_did_not_add_a_bypass_switch():
    """Unchanged from the locked era, and it has to stay unchanged.

    A lock lifted by approval is a reviewable source change; a lock lifted by
    an environment variable was never a lock. Neither may exist now that the
    entry points are open -- what the tool does must still be readable in the
    source rather than in whatever the shell happened to export.
    """
    import ast
    tree = ast.parse(CLI.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
            raise AssertionError("reading the environment would be a bypass")
        if isinstance(node, ast.Name) and node.id in ("environ", "getenv"):
            raise AssertionError("reading the environment would be a bypass")
    code = "\n".join(l for l in CLI.read_text().splitlines()
                     if not l.strip().startswith("#"))
    for token in ("--force", "--unlock", "--allow", "--yes-really", "--debug"):
        assert token not in code


def test_r13_the_environment_does_not_change_what_the_cli_does():
    env = dict(os.environ)
    for name in ("REPORT16_ALLOW_RUN", "ALLOW_EXECUTION", "CODEX_APPROVED",
                 "FORCE", "UNLOCK", "REPORT16_UNLOCK", "DEBUG"):
        env[name] = "1"
    plain = subprocess.run([sys.executable, str(CLI), "--show-plan"],
                           capture_output=True, text=True, cwd=ROOT)
    salted = subprocess.run([sys.executable, str(CLI), "--show-plan"],
                            capture_output=True, text=True, cwd=ROOT, env=env)
    assert plain.returncode == 0 and salted.returncode == 0
    # `created_at` differs between any two invocations, and `plan_digest`
    # covers it. Everything the plan actually decides has to be identical.
    volatile = ("created_at", "plan_digest")
    a = {k: v for k, v in json.loads(plain.stdout).items() if k not in volatile}
    b = {k: v for k, v in json.loads(salted.stdout).items() if k not in volatile}
    assert a == b


def test_read_only_modes_still_work():
    out = subprocess.run([sys.executable, str(CLI), "--show-plan"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0
    assert json.loads(out.stdout)["kind"] == "longrun_plan"


def test_r13_read_only_modes_create_nothing_now_that_the_lock_is_gone(
        sessions_unchanged):
    """The executing entry points are not invoked here -- deliberately.

    Read-only is the only thing this test is allowed to run against the real
    project directory, and it still has to leave it exactly as it was.

    Behavioural, and never skipped. "Creates nothing" is checked byte by byte
    over whatever the report directory holds -- an empty one in a public
    snapshot, exp001's evidence here -- and the session set is compared with
    its own baseline rather than with a hard-coded name.
    """
    cli = load_cli()
    before = digest_tree(cli.REPORT_DIR)
    argvs = [["--show-plan"], ["--verify", "--experiment-id", "exp016a"]]
    for existing in sessions_unchanged:
        argvs += [["--verify", "--experiment-id", existing],
                  ["--session-status", "--experiment-id", existing],
                  ["--from-json", "--experiment-id", existing]]
    for argv in argvs:
        subprocess.run([sys.executable, str(CLI), *argv],
                       capture_output=True, text=True, cwd=ROOT)
    assert digest_tree(cli.REPORT_DIR) == before


def test_verify_on_a_nonexistent_session_says_so_rather_than_pretending():
    out = subprocess.run([sys.executable, str(CLI), "--verify",
                          "--experiment-id", "exp016a"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 1
    assert "no report 16 session" in out.stdout


# ===========================================================================
# --out-name cannot overwrite anything
# ===========================================================================


def test_out_name_rejects_paths():
    cli = load_cli()
    for bad in ("../../PROJECT_STATUS.md", "/etc/passwd", "a/b.json",
                "..json", "x.txt", ""):
        with pytest.raises(ValueError):
            cli.resolve_out_name(bad)


def test_out_name_stays_inside_the_tool_directory():
    cli = load_cli()
    p = cli.resolve_out_name("bench.json")
    assert p.parent == cli.TOOL_DIR


def test_out_name_never_clobbers(tmp_path):
    cli = load_cli()
    from src.training.session import write_once_json
    target = cli.resolve_out_name("test_no_clobber_probe.json")
    if target.exists():
        target.unlink()
    write_once_json(target, {"a": 1})
    try:
        with pytest.raises(SystemExit, match="refusing to overwrite"):
            write_once_json(target, {"a": 2})
    finally:
        target.unlink()


# ===========================================================================
# Protected artefacts
# ===========================================================================


@needs_exp002
def test_report_16_does_not_touch_exp002():
    exp = EXP002_DIR
    assert exp.exists()
    assert len(list(exp.glob("*measurement_*.json"))) == 0
    assert len(list((exp / "events").glob("*.json"))) == 8
    assert (exp / "aggregate.json").exists()


@needs_exp002
def test_exp002_replays_against_its_snapshot_not_the_working_tree():
    """Retracted: this used to require the live tree to still match exp002.

    It no longer does, and it was never the invariant that matters. Round 18
    had to change `src/training/lora.py` and `src/data/instruction.py` to
    remove the measured child's network dependency, and both are inside
    exp002's manifest. What exp002's evidence rests on is its *snapshot*: the
    copy of the code taken when it ran, which is what `--verify` replays
    against (`check_working_tree=False`). Live drift is expected from here on
    and is reported, not prevented.
    """
    from src.training.session import verify_sources
    exp = EXP002_DIR
    manifest = json.loads((exp / "session.json").read_text())["source_manifest"]

    assert verify_sources(ROOT, manifest, exp / "source_snapshot",
                          check_working_tree=False) == [], \
        "exp002's own snapshot must still match its manifest"

    drifted = [rel for rel, meta in manifest["files"].items()
               if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
               != meta["sha256"]]
    assert set(drifted) <= {"src/training/lora.py", "src/data/instruction.py"}, (
        f"unexpected live drift against exp002's manifest: {sorted(drifted)}")


@needs_exp002
def test_report_16_adds_only_new_files():
    manifest = json.loads(
        (EXP002_DIR / "session.json").read_text())["source_manifest"]
    new = {"scripts/16_longrun.py", "src/training/longrun.py",
           "src/training/watchdog.py", "tests/test_longrun.py"}
    assert new.isdisjoint(set(manifest["files"]))


def digest_tree(root):
    """Every file under `root`, by relative path and digest."""
    root = Path(root)
    if not root.exists():
        return {}
    return {str(p.relative_to(root)):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@needs_exp001
def test_exp001_is_the_only_report_16_session_and_it_measured_nothing():
    """Retracted: this used to assert report 16 had never been executed.

    It has been, once. b1 spent a boot, died in the tokenizer 51 seconds in
    and left no report, so report 16 still has **no measured rows** -- but
    saying "no session exists" would now be false.

    Artifact-only: every claim here is about bytes on disk in the private
    research tree. A public snapshot publishes none of them, and there is
    nothing here for it to check.
    """
    assert report_16_sessions() == [EXP001]
    d = EXP001_DIR
    assert not (d / "b1.json").exists(), "no measured row was ever produced"
    st = lr.session_state(EXP001)
    assert st["terminal"] == ["b1"] and st["completed"] == []
    agg = json.loads((d / "aggregate.json").read_text())
    assert agg["state"] == "terminal_incomplete"
    assert agg["headline"]["allowed"] is False


# ===========================================================================
# Microbenchmark
# ===========================================================================


def test_the_microbenchmark_loads_nothing_and_creates_no_session():
    calls = []
    out = wd.microbenchmark(n=3, probe=lambda: calls.append(1) or {})
    assert out["kind"] == "watchdog_microbenchmark" and len(calls) == 3
    assert "not a report 16 experiment" in out["note"]


def test_the_microbenchmark_reports_cadence_drift():
    out = wd.microbenchmark(n=1, probe=lambda: {}, cadence=0.0, cycles=3,
                            sleep=lambda s: None)
    assert out["cadence_drift_seconds"]["n"] == 3


# ===========================================================================
# Adversarial regression suite.
#
# Eight ways a session could look finished while the evidence does not support
# it. Each of these was reachable before the guards below existed, which is
# why they are here: a validator is only worth what it refuses.
# ===========================================================================


def sess(tmp_path):
    return tmp_path / "reports"


def verify(tmp_path, eid="exp016a"):
    return lr.verify_experiment(eid, root=sess(tmp_path))["problems"]


def events_of(paths):
    return sorted((paths["dir"] / "events").glob("*.json"))


def rewrite(path, mutate):
    body = json.loads(path.read_text())
    mutate(body)
    path.write_text(json.dumps(body))


# --- 1. a "completed" run whose evidence is missing --------------------------


@pytest.mark.parametrize("missing", ["report", "report_sha", "watchdog_log",
                                     "watchdog_sha"])
def test_adv1_completed_without_its_evidence_is_refused(tmp_path, missing):
    out = finished_session(tmp_path)
    paths = out["paths"]
    if missing == "report":
        (paths["dir"] / "b1.json").unlink()
    elif missing == "watchdog_log":
        (paths["dir"] / "b1.watchdog.jsonl").unlink()
    else:
        ev = [e for e in events_of(paths) if "b1-measurement_finished" in e.name][0]
        key = "report_sha256" if missing == "report_sha" else "watchdog_sha256"
        rewrite(ev, lambda b: b.pop(key, None))
    problems = verify(tmp_path)
    assert problems, f"a completed run with no {missing} must be refused"


# --- 2. journal numbering and body identity ---------------------------------


@pytest.mark.parametrize("how", ["renumber_99", "gap", "duplicate", "shuffle",
                                 "no_experiment_id", "no_plan_digest"])
def test_adv2_broken_journal_numbering_or_identity_is_refused(tmp_path, how):
    out = finished_session(tmp_path)
    paths = out["paths"]
    evs = events_of(paths)
    if how == "renumber_99":
        evs[2].rename(evs[2].with_name("99" + evs[2].name[2:]))
    elif how == "gap":
        evs[2].unlink()
    elif how == "duplicate":
        (evs[2].parent / ("0" + evs[2].name)).write_text(evs[2].read_text())
    elif how == "shuffle":
        a, b = evs[2].read_text(), evs[3].read_text()
        evs[2].write_text(b)
        evs[3].write_text(a)
    elif how == "no_experiment_id":
        rewrite(evs[2], lambda x: x.pop("experiment_id", None))
    else:
        rewrite(evs[2], lambda x: x.pop("plan_digest", None))
    assert verify(tmp_path), f"a journal broken by {how} must be refused"


# --- 3. the started/finished backlink ---------------------------------------


@pytest.mark.parametrize("how", ["no_index", "no_digest", "wrong_started"])
def test_adv3_a_broken_started_finished_backlink_is_refused(tmp_path, how):
    out = finished_session(tmp_path)
    paths = out["paths"]
    fin = [e for e in events_of(paths) if "b2-measurement_finished" in e.name][0]
    if how == "no_index":
        rewrite(fin, lambda b: b.pop("started_index", None))
    elif how == "no_digest":
        rewrite(fin, lambda b: b.pop("started_event_digest", None))
    else:
        started = [e for e in events_of(paths)
                   if "b1-measurement_started" in e.name][0]
        other = hashlib.sha256(started.read_bytes()).hexdigest()
        rewrite(fin, lambda b: b.update({"started_event_digest": other}))
    assert verify(tmp_path), f"a backlink broken by {how} must be refused"


# --- 4. the child report's identity -----------------------------------------


@pytest.mark.parametrize("field,value", [
    ("run_id", "b9"), ("condition", "continuous"), ("declared_rows", 123),
    ("plan_digest", "z" * 64), ("nonce", "not-the-launch-nonce"),
])
def test_adv4_a_child_report_that_disagrees_with_the_plan_is_refused(
        tmp_path, field, value):
    out = finished_session(tmp_path)
    paths = out["paths"]
    rp = paths["dir"] / "b1.json"
    rewrite(rp, lambda b: b.update({field: value}))
    ev = [e for e in events_of(paths) if "b1-measurement_finished" in e.name][0]
    rewrite(ev, lambda b: b.update(
        {"report_sha256": hashlib.sha256(rp.read_bytes()).hexdigest()}))
    assert verify(tmp_path), f"a child whose {field} disagrees must be refused"


def test_adv4_a_child_report_missing_provenance_fields_is_refused(tmp_path):
    out = finished_session(tmp_path)
    paths = out["paths"]
    rp = paths["dir"] / "b1.json"
    rewrite(rp, lambda b: b["provenance"].pop("packages", None))
    ev = [e for e in events_of(paths) if "b1-measurement_finished" in e.name][0]
    rewrite(ev, lambda b: b.update(
        {"report_sha256": hashlib.sha256(rp.read_bytes()).hexdigest()}))
    assert verify(tmp_path)


# --- 5. calibration ----------------------------------------------------------


@pytest.mark.parametrize("drop", ["samples", "thresholds", "policy"])
def test_adv5_an_incomplete_calibration_is_refused(tmp_path, drop):
    out = finished_session(tmp_path)
    paths = out["paths"]
    calib = json.loads(paths["calibration"].read_text())
    calib.pop(drop, None)
    paths["calibration"].write_text(json.dumps(calib))
    from src.training.session import sha256_file
    sess_body = json.loads(paths["session"].read_text())
    sess_body["calibration_sha256"] = sha256_file(paths["calibration"])
    paths["session"].write_text(json.dumps(sess_body))
    assert verify(tmp_path), f"a calibration with no {drop} must be refused"


def test_adv5_rewriting_the_calibration_and_its_recorded_sha_together_is_refused(
        tmp_path):
    out = finished_session(tmp_path)
    paths = out["paths"]
    calib = json.loads(paths["calibration"].read_text())
    calib["thresholds"]["swap_used_gb"] = 99.0
    paths["calibration"].write_text(json.dumps(calib))
    from src.training.session import sha256_file
    sess_body = json.loads(paths["session"].read_text())
    sess_body["calibration_sha256"] = sha256_file(paths["calibration"])
    paths["session"].write_text(json.dumps(sess_body))
    assert verify(tmp_path), ("thresholds that do not recompute must be caught "
                              "even when the digest was updated to match")


# --- 6. the aggregate --------------------------------------------------------


@pytest.mark.parametrize("field", ["runs", "terminal", "complete",
                                   "journal_events", "plan_digest",
                                   "replay_problems"])
def test_adv6_a_tampered_aggregate_field_is_refused(tmp_path, field):
    finished_session(tmp_path)
    lr.session_finalize("exp016a", root=sess(tmp_path))
    paths = lr.session_paths("exp016a", root=sess(tmp_path))
    agg = json.loads(paths["aggregate"].read_text())
    if field == "runs":
        agg["runs"][0]["Q1_run"] = "fails"
    elif field == "journal_events":
        agg["journal_events"] = 99
    elif field == "plan_digest":
        agg["plan_digest"] = "z" * 64
    elif field == "replay_problems":
        agg["replay_problems"] = ["invented"]
    else:
        agg[field] = not agg[field]
    paths["aggregate"].write_text(json.dumps(agg))
    assert verify(tmp_path), f"a tampered aggregate {field} must be refused"


def test_adv6_a_tampered_q2_run_is_refused(tmp_path):
    finished_session(tmp_path)
    lr.session_finalize("exp016a", root=sess(tmp_path))
    paths = lr.session_paths("exp016a", root=sess(tmp_path))
    agg = json.loads(paths["aggregate"].read_text())
    agg["runs"][0]["Q2_run"] = "stable" if agg["runs"][0]["Q2_run"] != "stable" \
        else "scales"
    paths["aggregate"].write_text(json.dumps(agg))
    assert verify(tmp_path)


# --- 7. the watchdog evidence and R1 ----------------------------------------


@pytest.mark.parametrize("missing", ["log", "sha", "stop_request"])
def test_adv7_missing_watchdog_evidence_is_refused(tmp_path, missing):
    out = stopped_session(tmp_path)
    paths = out["paths"]
    if missing == "log":
        (paths["dir"] / "b1.watchdog.jsonl").unlink()
    elif missing == "sha":
        ev = [e for e in events_of(paths)
              if "b1-measurement_finished" in e.name][0]
        rewrite(ev, lambda b: b.pop("watchdog_sha256", None))
    else:
        (paths["dir"] / "b1.stop_request.json").unlink()
    assert verify(tmp_path), f"a stopped run with no {missing} must be refused"


def test_adv7_r1_must_replay_against_the_watchdog_log(tmp_path):
    out = stopped_session(tmp_path)
    paths = out["paths"]
    fired = lr.apply_rule_r1("exp016a", root=sess(tmp_path))
    assert fired["fired"]
    ev = [e for e in events_of(paths) if "plan_arm_cancelled" in e.name][0]
    body = json.loads(ev.read_text())
    assert body.get("watchdog_sha256"), (
        "an R1 cancellation caused by a safety stop must name the watchdog log "
        "it was recomputed from")
    (paths["dir"] / "b1.watchdog.jsonl").unlink()
    assert verify(tmp_path), ("R1 must be replayable against the watchdog log, "
                              "so a missing log must fail the replay")


# --- 8. a child that never reports progress ---------------------------------


def test_adv8_process_ceiling_is_enforced_without_any_child_progress():
    """A child stuck before the model even loads sends no progress at all.

    The ceiling must therefore be enforced from the watchdog's own elapsed
    time; a rule that reads the child's clocks can never fire when the child
    has not published one.
    """
    spec = wd.SafetySpec()
    state = wd.SafetyState(spec=spec)
    reason = None
    t = 0.0
    while t <= spec.process_max_seconds + spec.poll_seconds:
        reason = state.observe(sample(), t, progress=None)["reason"]
        if reason:
            break
        t += spec.poll_seconds
    assert reason == "process_max_seconds", (
        "with no progress ever published, the process ceiling must still stop "
        f"the run; got {reason!r} after {t}s")


def test_adv8_a_heartbeating_watchdog_does_not_excuse_a_silent_child():
    spec = wd.SafetySpec()
    state = wd.SafetyState(spec=spec)
    beats = 0
    reason = None
    t = 0.0
    while t <= spec.process_max_seconds + spec.poll_seconds:
        out = state.observe(sample(), t, progress=None)
        beats += 1
        reason = out["reason"]
        if reason:
            break
        t += spec.poll_seconds
    assert beats > 100 and reason == "process_max_seconds"


def test_adv8_stalled_progress_is_its_own_tool_failure():
    """Progress that stops advancing is different from progress never sent."""
    spec = wd.SafetySpec()
    state = wd.SafetyState(spec=spec)
    frozen = {"row": 7, "row_elapsed_seconds": 1.0,
              "condition_clock_seconds": 10.0, "process_clock_seconds": 12.0}
    reason = None
    for i in range(200):
        reason = state.observe(sample(), i * spec.poll_seconds,
                               progress=frozen)["reason"]
        if reason:
            break
    assert reason == "progress_stalled", (
        "a progress file frozen at one row means the child is wedged; that is "
        f"a tool failure, got {reason!r}")


# ===========================================================================
# End to end, through session_next itself.
#
# Not a hand-rolled _run_real(): this drives the same function the CLI would
# call, with real subprocesses underneath, and then follows the session all
# the way to R1, finalize and verify.
# ===========================================================================


E2E_CHILD = textwrap.dedent('''
    import json, os, signal, sys, time
    sys.path.insert(0, {root!r})
    from src.training import longrun as lr
    from src.training import watchdog as wd

    d, run, nonce, rows = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    digest, degrade = sys.argv[5], sys.argv[6] == "degrade"
    flag = wd.StopFlag()
    signal.signal(signal.SIGTERM, flag.handler)
    progress = os.path.join(d, run + ".progress.jsonl")
    t0 = time.perf_counter()
    per_row, stopped = [], None
    for i in range(1, rows + 1):
        r0 = time.perf_counter()
        time.sleep(0.001)
        if degrade and i > rows * 0.8:
            time.sleep(0.008)
        sec = time.perf_counter() - r0
        cleared = i % 10 == 0
        per_row.append({{"row": i, "compute_seconds": sec,
                        "end_to_end_seconds": sec, "sample_id": "s%d" % i,
                        "tokens": 100, "supervised_tokens": 70, "loss": 0.5,
                        "cleared": cleared,
                        "clear_seconds": 0.001 if cleared else None}})
        if i % 5 == 0:
            lr.write_progress(progress, row=i, row_elapsed_seconds=sec,
                              condition_clock_seconds=time.perf_counter() - t0,
                              process_clock_seconds=time.perf_counter() - t0)
        if flag.is_set():
            got = wd.read_stop_request(d, prefix=run, nonce=nonce)
            if got["accepted"]:
                stopped = {{"reason": got["body"]["reason"],
                           "rule": got["body"].get("rule"), "row": i,
                           "sampled_values": got["body"].get("sampled_values"),
                           "condition_clock_seconds": time.perf_counter() - t0,
                           "process_clock_seconds": time.perf_counter() - t0,
                           "requested_by": "watchdog",
                           "stop_request_sha256": got["sha256"]}}
                break
    condition_seconds = time.perf_counter() - t0
    compute = sum(r["compute_seconds"] for r in per_row)
    prov = {{f: ({{}} if f.endswith(("sha256", "digest", "config", "optimizer",
                                   "packages", "conditions"))
                else "x") for f in lr.CHILD_PROVENANCE_REQUIRED}}
    prov["measurement_intervals"] = {{"window": lr.WINDOW,
                                     "memory_every": lr.MEMORY_EVERY,
                                     "empty_cache_every": lr.EMPTY_CACHE_EVERY,
                                     "max_rows": rows}}
    m = lr.compute_metrics(per_row,
                           stop_reason=(stopped or {{}}).get("reason"))
    ids = [r["sample_id"] for r in per_row]
    order_ids = ids + ["s%d" % i for i in range(len(per_row) + 1, rows + 1)]
    clear_total = sum(r["clear_seconds"] or 0 for r in per_row)
    session = json.load(open(os.path.join(d, "session.json")))
    report = {{"schema_version": lr.CHILD_SCHEMA_VERSION, "kind": lr.CHILD_KIND,
              "experiment_id": os.path.basename(d.rstrip("/")),
              "run_id": run, "declared_rows": rows, "condition": "empty_cache",
              "plan_digest": digest, "nonce": nonce, "per_row": per_row,
              "memory": [], "pool_pairs": lr.POOL_PAIRS,
              "pool_rows": lr.POOL_ROWS, "rows_requested": rows,
              "child_source_check": {{
                  "files_verified": len(session["source_manifest"]["files"]),
                  "source_manifest_digest": session["source_manifest_digest"],
                  "plan_digest": digest}},
              "preflight": {{"swap_used_gb": 0.0,
                            "memory_pressure_percent_free": 95,
                            "free_plus_inactive_gb": 8.0,
                            "normalized_load_1m": 0.1}},
              "child_pid": os.getpid(), "child_pgid": os.getpgrp(),
              "child_start_identity": wd.process_start_identity(os.getpid()),
              "started_at": "2026-08-16T00:00:00Z",
              "finished_at": "2026-08-16T00:10:00Z", "tool_failure": None,
              "model_compute_seconds": compute,
              "stopped_early": stopped, "rows_completed": len(per_row),
              "end_to_end_seconds": sum(r["end_to_end_seconds"] for r in per_row),
              "input_order_digest": lr._digest_ids(order_ids),
              "completed_input_digest": lr._digest_ids(ids), "provenance": prov,
              "metrics": {{k: m.get(k) for k in
                          ("D100", "D100_reason", "D20", "D20_reason",
                           "Dmax", "Dmax_reason")}},
              "scheduled_empty_cache_every": lr.EMPTY_CACHE_EVERY,
              "scheduled_empty_cache_cost": {{
                  "calls": sum(1 for r in per_row if r["cleared"]),
                  "total_seconds": clear_total,
                  "per_call": [{{"row": r["row"], "seconds": r["clear_seconds"]}}
                              for r in per_row if r["cleared"]]}},
              "teardown_empty_cache_calls": 1,
              "teardown_empty_cache_seconds": 0.001,
              "between_row_overhead_breakdown": {{
                  "scheduled_empty_cache_seconds": clear_total,
                  "memory_probe_seconds": 0.0,
                  "unattributed_seconds": 0.0}},
              "float_storage": {{"seconds_rounded": False,
                                "loss_rounded": False}},
              "clocks": {{"model_load_seconds": 0.0,
                         "condition_clock_seconds": condition_seconds,
                         "process_clock_seconds": time.perf_counter() - t0}}}}
    with open(os.path.join(d, run + ".json"), "w") as fh:
        json.dump(report, fh)
''')


class E2ELauncher:
    """The same wiring the CLI would use, with real subprocesses."""

    def __init__(self, tmp_path, spec, *, rows, degrade=False, trip_at=10 ** 9):
        self.spec = spec
        self.rows = rows
        self.degrade = degrade
        self.trip_at = trip_at
        self.child_py = tmp_path / "e2e_child.py"
        self.child_py.write_text(E2E_CHILD.format(root=str(ROOT)))
        self.wd_py = tmp_path / "e2e_wd.py"
        self.wd_py.write_text(DUMMY_WATCHDOG.format(root=str(ROOT)))
        self.child = self.watchdog = self.reader = None
        self.hs_w = self.hb_r = None
        self.beats = 0

    def spawn_watchdog(self, *, run, paths):
        hs_r, self.hs_w = os.pipe()
        self.hb_r, hb_w = os.pipe()
        self.watchdog = subprocess.Popen(
            [sys.executable, str(self.wd_py), str(paths["dir"]), run,
             str(hs_r), str(hb_w), str(self.trip_at), "3.0"],
            pass_fds=(hs_r, hb_w), start_new_session=True)
        os.close(hs_r)
        os.close(hb_w)
        self.reader = wd.LineReader(self.hb_r)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            for msg in self.reader.poll():
                if msg.get("ready"):
                    return {"ready": True, "proc": self.watchdog}
            time.sleep(0.01)
        return {"ready": False, "reason": "timeout", "proc": self.watchdog}

    def spawn_child(self, *, run, paths, entry, nonce, plan):
        # The plan says how many rows, not the test: a child that measures a
        # different length is exactly what the identity check rejects.
        self.nonce = nonce
        self.child = subprocess.Popen(
            [sys.executable, str(self.child_py), str(paths["dir"]), run, nonce,
             str(entry["declared_rows"]), plan["plan_digest"],
             "degrade" if self.degrade else "flat"],
            start_new_session=True)
        return {"spawned": True, "proc": self.child, "nonce": nonce}

    def hand_identity(self, child, *, paths, run):
        proc = child["proc"]
        start = wd.process_start_identity(proc.pid)
        pgid = wd.observed_pgid(proc.pid)
        if start is None or pgid is None:
            return {"ok": False, "reason": "child vanished"}
        identity = wd.ChildIdentity(pid=proc.pid, pgid=pgid,
                                    nonce=child["nonce"], start_identity=start)
        wd.write_launch_record(paths["dir"], prefix=run, identity=identity,
                               experiment_id=paths["dir"].name, run_id=run)
        wd.write_line(self.hs_w, identity.as_dict())
        return {"ok": True}

    def await_armed(self):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            for msg in self.reader.poll():
                if msg.get("armed"):
                    return {"armed": True}
                self.beats += 1
            time.sleep(0.01)
        return {"armed": False, "reason": "timeout"}

    def supervise(self):
        def poll():
            got = self.reader.poll()
            self.beats += len(got)
            return bool(got)

        return wd.supervise(child_alive=lambda: self.child.poll() is None,
                            watchdog_alive=lambda: self.watchdog.poll() is None,
                            poll_heartbeat=poll, clock=time.monotonic,
                            spec=self.spec,
                            on_stop=lambda r: wd.reap(self.child, self.watchdog),
                            sleep=time.sleep)

    def collect(self, *, run, paths):
        from src.training.session import sha256_file
        rc = self.child.wait(timeout=60)
        try:
            self.watchdog.wait(timeout=30)
        except Exception:
            wd.reap(self.watchdog)
        for fd in (self.hs_w, self.hb_r):
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


def e2e_args(tmp_path, launcher, boot):
    calib = json.loads(
        lr.session_paths("exp016a", root=sess(tmp_path))["calibration"].read_text())
    return dict(root=sess(tmp_path),
                gate=lambda: gate_record(calib["thresholds"]),
                spawn_watchdog=launcher.spawn_watchdog,
                spawn_child=launcher.spawn_child,
                hand_identity=launcher.hand_identity,
                await_armed=launcher.await_armed,
                supervise_fn=launcher.supervise,
                collect=launcher.collect,
                boot={"boot_fingerprint": boot},
                # Injected, exactly as `next_args` does it. These tests drive
                # the real parent, a real child subprocess and a real watchdog
                # -- but the child computes rather than loading a model, so it
                # touches neither the instruction pool nor the hub cache.
                # Leaving the production preflight in made a behavioural test
                # depend on data it never reads, which is how the same test
                # passes here and refuses to start in a tree without it.
                # `dependency_preflight` keeps its own tests.
                dependency_preflight=lambda: {"ok": True, "problems": [],
                                              "evidence": {"repositories": []}},
                verify_live_sources=False)


def test_e2e_session_next_runs_a_real_child_and_verifies(tmp_path):
    out = init(tmp_path)
    spec = lr.plan_spec(out["plan"])
    launcher = E2ELauncher(tmp_path, spec, rows=500)
    result = lr.session_next("exp016a", **e2e_args(tmp_path, launcher, "boot-0"))
    assert result["ok"], result["problems"]
    assert result["outcome"] == "completed"
    assert launcher.beats > 0, "the parent received real heartbeats"

    st = lr.session_state("exp016a", root=sess(tmp_path))
    assert st["completed"] == ["b1"] and st["next_run"] == "b2"
    assert (out["paths"]["dir"] / "b1.watchdog.jsonl").exists()
    assert lr.verify_experiment("exp016a", root=sess(tmp_path))["problems"] == []


def test_e2e_a_degrading_child_fires_r1_and_finalises(tmp_path):
    out = init(tmp_path)
    spec = lr.plan_spec(out["plan"])
    launcher = E2ELauncher(tmp_path, spec, rows=500, degrade=True)
    result = lr.session_next("exp016a", **e2e_args(tmp_path, launcher, "boot-0"))
    assert result["ok"]
    assert result["r1"] and result["r1"]["fired"], "R1 must run automatically"

    st = lr.session_state("exp016a", root=sess(tmp_path))
    assert st["cancelled"] == ["b2", "b3"]
    fin = lr.session_finalize("exp016a", root=sess(tmp_path))
    assert fin["problems"] == []
    agg = fin["aggregate"]
    assert agg["state"] == "complete_by_rule"
    assert agg["headline"]["allowed"] is False
    assert agg["prefix_consistency"]["verdict"] == "not_applicable"
    assert lr.verify_experiment("exp016a", root=sess(tmp_path))["problems"] == []


def test_e2e_a_real_safety_stop_is_authenticated_end_to_end(tmp_path):
    out = init(tmp_path)
    spec = lr.plan_spec(out["plan"])
    launcher = E2ELauncher(tmp_path, spec, rows=500, trip_at=3)
    result = lr.session_next("exp016a", **e2e_args(tmp_path, launcher, "boot-0"))
    d = out["paths"]["dir"]
    report = json.loads((d / "b1.json").read_text())
    assert report["stopped_early"]["reason"] == "swap_used_gb"
    assert (d / "b1.stop_request.json").exists()
    assert lr.verify_experiment("exp016a", root=sess(tmp_path))["problems"] == []
    fin = lr.session_finalize("exp016a", root=sess(tmp_path))
    assert fin["aggregate"]["state"] == "complete_by_rule"


def test_e2e_the_same_boot_is_refused_after_a_real_run(tmp_path):
    out = init(tmp_path)
    spec = lr.plan_spec(out["plan"])
    lr.session_next("exp016a",
                    **e2e_args(tmp_path, E2ELauncher(tmp_path, spec, rows=500),
                               "boot-0"))
    again = lr.session_next(
        "exp016a", **e2e_args(tmp_path, E2ELauncher(tmp_path, spec, rows=500),
                              "boot-0"))
    assert not again["ok"]
    assert any("already happened in this boot" in p for p in again["problems"])


# ===========================================================================
# Round four: the production child, stall accounting, preconditions,
# one provenance list, and an informational design digest.
# ===========================================================================


# --- 2. progress stall accounting -------------------------------------------


def test_r4_progress_advancing_for_ever_never_trips_the_stall_rule():
    """Sixty-plus polls, each on a new row, must not look like a stall."""
    spec = wd.SafetySpec()
    s = wd.SafetyState(spec=spec)
    reason = None
    for i in range(1, 200):
        reason = s.observe(sample(), i * spec.poll_seconds, progress={
            "row": i, "row_elapsed_seconds": 1.0,
            "condition_clock_seconds": float(i), "process_clock_seconds": float(i)
        })["reason"]
        assert reason is None, f"advancing rows tripped {reason!r} at poll {i}"
    assert s.progress_stall_polls == 0


def test_r4_the_stall_rule_trips_exactly_on_the_declared_poll():
    spec = wd.SafetySpec()
    s = wd.SafetyState(spec=spec)
    frozen = {"row": 7, "row_elapsed_seconds": 1.0,
              "condition_clock_seconds": 1.0, "process_clock_seconds": 1.0}
    reasons = []
    for i in range(spec.progress_stall_polls + 5):
        reasons.append(s.observe(sample(), i * spec.poll_seconds,
                                 progress=frozen)["reason"])
    first = next(i for i, r in enumerate(reasons) if r == "progress_stalled")
    assert first == spec.progress_stall_polls, (
        f"the stall must trip on poll {spec.progress_stall_polls}, not {first}")
    assert all(r is None for r in reasons[:first])


def test_r4_a_row_going_backwards_is_refused():
    s = wd.SafetyState()
    s.observe(sample(), 0.0, progress={"row": 9, "row_elapsed_seconds": 1.0})
    out = s.observe(sample(), 5.0, progress={"row": 4, "row_elapsed_seconds": 1.0})
    assert out["reason"] == "progress_invalid", (
        "a row number that goes backwards means the progress file is not what "
        f"we think it is; got {out['reason']!r}")


@pytest.mark.parametrize("bad", [None, "7", 7.5, True, -1])
def test_r4_an_illegal_row_type_is_refused(bad):
    s = wd.SafetyState()
    out = s.observe(sample(), 0.0, progress={"row": bad,
                                             "row_elapsed_seconds": 1.0})
    assert out["reason"] == "progress_invalid"


def test_r4_alternating_rows_reset_the_stall_counter():
    spec = wd.SafetySpec()
    s = wd.SafetyState(spec=spec)
    row = 1
    for i in range(spec.progress_stall_polls * 3):
        if i % 10 == 9:
            row += 1
        out = s.observe(sample(), i * spec.poll_seconds,
                        progress={"row": row, "row_elapsed_seconds": 1.0})
        assert out["reason"] is None


# --- 3. preconditions are replayed before the gate is touched ---------------


def test_r4_a_tampered_calibration_stops_session_next_before_the_gate(tmp_path):
    out = init(tmp_path)
    calib = json.loads(out["paths"]["calibration"].read_text())
    calib["thresholds"]["swap_used_gb"] = 99.0
    out["paths"]["calibration"].write_text(json.dumps(calib))
    called = {"gate": 0, "watchdog": 0, "child": 0}

    def gate():
        called["gate"] += 1
        return gate_record(THRESHOLDS)

    result = lr.session_next("exp016a", **next_args(
        tmp_path, gate=gate,
        spawn_watchdog=lambda run, paths: called.__setitem__(
            "watchdog", called["watchdog"] + 1) or {"ready": True, "proc": None}))
    assert not result["ok"]
    assert called["gate"] == 0, "the gate must not be polled at all"
    assert called["watchdog"] == 0
    assert result.get("boot_consumed") is False
    assert lr.read_journal(out["paths"]["dir"]) == [], "no event may be written"


def test_r4_a_broken_snapshot_stops_session_next_before_the_gate(tmp_path):
    out = init(tmp_path)
    snap = next(out["paths"]["snapshot"].glob("*"))
    snap.write_text("tampered")
    called = {"gate": 0}
    result = lr.session_next("exp016a", **next_args(
        tmp_path, gate=lambda: called.__setitem__("gate", 1) or gate_record(
            THRESHOLDS), verify_live_sources=False))
    assert not result["ok"] and called["gate"] == 0
    assert lr.read_journal(out["paths"]["dir"]) == []


def test_r4_a_tampered_earlier_child_stops_the_next_run_before_the_gate(tmp_path):
    out = init(tmp_path)
    place_run(out["paths"], out["plan"], "b1", 500, boot="boot-0")
    rp = out["paths"]["dir"] / "b1.json"
    rp.write_text(json.dumps({"run_id": "b1", "per_row": []}))
    called = {"gate": 0}
    result = lr.session_next("exp016a", **next_args(
        tmp_path, gate=lambda: called.__setitem__("gate", 1) or gate_record(
            THRESHOLDS), boot={"boot_fingerprint": "boot-1"}))
    assert not result["ok"] and called["gate"] == 0


def test_r4_a_clean_session_still_reaches_the_gate(tmp_path):
    init(tmp_path)
    called = {"gate": 0}

    def gate():
        called["gate"] += 1
        return gate_record(THRESHOLDS)

    result = lr.session_next("exp016a", **next_args(tmp_path, gate=gate))
    assert result["ok"] and called["gate"] == 1


# --- 4. one provenance list, shared by writer, replay and cross-run ---------


def test_r4_there_is_a_single_provenance_field_list():
    assert lr.PROVENANCE_FIELDS
    assert set(lr.CROSS_RUN_FIELDS) <= set(lr.PROVENANCE_FIELDS)


def test_r4_a_field_missing_from_every_run_still_fails(tmp_path):
    """Identical absence is not agreement."""
    out = finished_session(tmp_path)
    for run in lr.RUN_IDS:
        rp = out["paths"]["dir"] / f"{run}.json"
        body = json.loads(rp.read_text())
        body["provenance"].pop("optimizer", None)
        rp.write_text(json.dumps(body))
        ev = [e for e in events_of(out["paths"])
              if f"{run}-measurement_finished" in e.name][0]
        rewrite(ev, lambda b: b.update(
            {"report_sha256": hashlib.sha256(rp.read_bytes()).hexdigest()}))
    problems = verify(tmp_path)
    assert any("optimizer" in p for p in problems), (
        "a required provenance field absent from every run must still fail")


def test_r4_cross_run_comparison_uses_the_shared_list(tmp_path):
    out = finished_session(tmp_path)
    rp = out["paths"]["dir"] / "b2.json"
    body = json.loads(rp.read_text())
    body["provenance"]["device"] = "cpu"
    rp.write_text(json.dumps(body))
    ev = [e for e in events_of(out["paths"])
          if "b2-measurement_finished" in e.name][0]
    rewrite(ev, lambda b: b.update(
        {"report_sha256": hashlib.sha256(rp.read_bytes()).hexdigest()}))
    lr.session_finalize("exp016a", root=sess(tmp_path))
    agg = json.loads(lr.session_paths(
        "exp016a", root=sess(tmp_path))["aggregate"].read_text())
    assert agg["headline"]["allowed"] is False


# --- 5. the design digest is informational, and says so consistently --------


def test_r4_a_changed_design_file_is_informational_not_fatal(tmp_path):
    out = finished_session(tmp_path)
    plan = json.loads(out["paths"]["plan"].read_text())
    plan_body = dict(plan)
    plan_body["design_sha256"] = "a" * 64
    plan_body["plan_digest"] = lr.plan_digest(plan_body)
    out["paths"]["plan"].write_text(json.dumps(plan_body))
    sess_body = json.loads(out["paths"]["session"].read_text())
    sess_body["plan_digest"] = plan_body["plan_digest"]
    out["paths"]["session"].write_text(json.dumps(sess_body))
    for ev in events_of(out["paths"]):
        rewrite(ev, lambda b: b.update({"plan_digest": plan_body["plan_digest"]}))
    result = lr.verify_experiment("exp016a", root=sess(tmp_path))
    assert not any("design file" in p for p in result["problems"]), (
        "the design digest is documented as informational, so it must not "
        "appear among the problems")
    assert any("design" in n for n in result.get("notes", [])), (
        "it must still be reported, as a note")


# --- 1. the production child ------------------------------------------------


def test_r4_run_child_is_not_a_placeholder():
    """It measures, and the production deps really build the frozen pool."""
    import inspect
    src = inspect.getsource(lr.run_child)
    assert "raise RuntimeError" not in src
    for token in ("per_row", "EMPTY_CACHE_EVERY", "read_stop_request",
                  "write_once_json", "compute_metrics"):
        assert token in src, f"the child must do its own {token}"
    prod = inspect.getsource(lr.ProductionChildDeps)
    assert "sample_pairs" in prod and "randperm" in prod
    # The pair count now comes from the named source rather than from the
    # constant directly, because the final run reads the whole split. What
    # matters is unchanged and is asserted rather than grepped: the default
    # source is the frozen pool, and the frozen pool is still 250 pairs.
    assert 'n_pairs=shape["pairs"]' in prod
    assert lr.DATA_SOURCES["pool"] == {"pairs": 250, "rows": 2000}
    assert lr.POOL_PAIRS == 250
    import inspect as _inspect
    assert _inspect.signature(
        lr.ProductionChildDeps.__init__).parameters["source"].default == "pool"


def test_r4_run_child_writes_a_report_replay_accepts(tmp_path):
    """The production function, with injected fakes instead of a model."""
    out = init(tmp_path)
    plan = out["plan"]
    d = out["paths"]["dir"]
    rc = lr.run_child(experiment_dir=d, run="b1", rows=500,
                      nonce="test-nonce", plan_digest=plan["plan_digest"],
                      deps=lr.FakeChildDeps(pool_rows=lr.POOL_ROWS))
    assert rc == 0
    report = json.loads((d / "b1.json").read_text())
    assert report["rows_completed"] == 500
    assert report["run_id"] == "b1" and report["nonce"] == "test-nonce"
    assert [r["row"] for r in report["per_row"]][:3] == [1, 2, 3]
    assert len([r for r in report["per_row"] if r["cleared"]]) == 50
    assert set(lr.PROVENANCE_FIELDS) <= set(report["provenance"])
    assert lr.replay_child(report)["problems"] == []


def test_r4_run_child_uses_the_frozen_permutation_prefix(tmp_path):
    out = init(tmp_path)
    d = out["paths"]["dir"]
    deps = lr.FakeChildDeps(pool_rows=lr.POOL_ROWS)
    lr.run_child(experiment_dir=d, run="b1", rows=500, nonce="n",
                 plan_digest=out["plan"]["plan_digest"], deps=deps)
    small = json.loads((d / "b1.json").read_text())
    lr.run_child(experiment_dir=d, run="b2", rows=1000, nonce="n",
                 plan_digest=out["plan"]["plan_digest"], deps=deps)
    big = json.loads((d / "b2.json").read_text())
    assert [r["sample_id"] for r in small["per_row"]] == \
        [r["sample_id"] for r in big["per_row"]][:500]
    assert lr.prefix_consistency([small, big])["verdict"] == "passed"


def test_r4_run_child_writes_progress_and_memory_at_the_declared_cadence(tmp_path):
    out = init(tmp_path)
    d = out["paths"]["dir"]
    lr.run_child(experiment_dir=d, run="b1", rows=500, nonce="n",
                 plan_digest=out["plan"]["plan_digest"],
                 deps=lr.FakeChildDeps(pool_rows=lr.POOL_ROWS))
    report = json.loads((d / "b1.json").read_text())
    assert [m["row"] for m in report["memory"]][:3] == [1, 5, 10]
    lines = (d / "b1.progress.jsonl").read_text().strip().split("\n")
    assert len(lines) >= 100
    assert json.loads(lines[-1])["row"] == 500


def test_r4_run_child_honours_an_authenticated_stop_request(tmp_path):
    out = init(tmp_path)
    d = out["paths"]["dir"]
    wd.write_stop_request(d, prefix="b1", reason="swap_used_gb",
                          rule="swap_used_gb", nonce="n", monotonic=1.0,
                          wall_clock="Z")
    deps = lr.FakeChildDeps(pool_rows=lr.POOL_ROWS, stop_at_row=25)
    lr.run_child(experiment_dir=d, run="b1", rows=500, nonce="n",
                 plan_digest=out["plan"]["plan_digest"], deps=deps)
    report = json.loads((d / "b1.json").read_text())
    assert report["stopped_early"]["reason"] == "swap_used_gb"
    assert report["rows_completed"] < 500


def test_r4_run_child_refuses_a_stop_request_with_the_wrong_nonce(tmp_path):
    out = init(tmp_path)
    d = out["paths"]["dir"]
    wd.write_stop_request(d, prefix="b1", reason="swap_used_gb", rule="r",
                          nonce="SOMEONE-ELSE", monotonic=1.0, wall_clock="Z")
    deps = lr.FakeChildDeps(pool_rows=lr.POOL_ROWS, stop_at_row=25)
    rc = lr.run_child(experiment_dir=d, run="b1", rows=500, nonce="n",
                      plan_digest=out["plan"]["plan_digest"], deps=deps)
    report = json.loads((d / "b1.json").read_text())
    assert report["stopped_early"] is None or \
        report["stopped_early"].get("reason") == "stop_request_rejected"
    assert rc != 0 or report.get("tool_failure") == "stop_request_rejected"


def test_r4_run_child_will_not_overwrite_an_existing_report(tmp_path):
    out = init(tmp_path)
    d = out["paths"]["dir"]
    deps = lr.FakeChildDeps(pool_rows=lr.POOL_ROWS)
    lr.run_child(experiment_dir=d, run="b1", rows=500, nonce="n",
                 plan_digest=out["plan"]["plan_digest"], deps=deps)
    with pytest.raises((SystemExit, FileExistsError)):
        lr.run_child(experiment_dir=d, run="b1", rows=500, nonce="n",
                     plan_digest=out["plan"]["plan_digest"], deps=deps)


# ===========================================================================
# Round 5. Codex's re-review: the production path.
#
# Everything above this line was verified with fakes standing in for the
# model. That was enough to test the state machine and never enough to test
# the child that will actually run: the fake and the real deps agreed on a
# contract that the real modules do not offer. These tests hold the
# production path to the modules as they are on disk, and hold the training
# arithmetic to report 15's, which is the only thing that makes b1/b2/b3
# prefixes of one another.
# ===========================================================================

def _import_from_nodes(func) -> list[tuple[str, list[str]]]:
    """Every ``from X import a, b`` inside a function, as (module, names)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.module, [a.name for a in node.names]))
    return out


# --- 1. the production deps import names that exist -------------------------


def test_r5_production_deps_import_only_names_that_exist():
    """The child imports what the modules export, not what it wishes they did.

    ``load_tokenizer`` lives in src.generation.brickgpt and ``read_rows`` in
    src.training.lora. Importing them from anywhere else raises ImportError
    at the one moment nothing may fail: after a boot has been spent.
    """
    missing = []
    for module_name, names in _import_from_nodes(lr.ProductionChildDeps.load):
        module = importlib.import_module(module_name)
        for name in names:
            if not hasattr(module, name):
                missing.append(f"{module_name}.{name}")
    assert missing == [], f"the production child imports names nobody exports: {missing}"


def test_r5_production_deps_use_the_real_encoded_shape():
    """``Encoded`` carries ids and labels; the batch is what collate makes."""
    from src.training.lora import Encoded

    assert set(Encoded.__dataclass_fields__) == {
        "input_ids", "labels", "n_prompt_tokens", "truncated"}
    src = inspect.getsource(lr.ProductionChildDeps.load)
    for absent in (".batch", "enc.tokens", "enc.supervised_tokens"):
        assert absent not in src, (
            f"Encoded has no {absent}; the counts come from the collated batch")
    assert "collate" in src, "a batch is built by collate, not by the encoder"


# --- 2. report 15's training semantics, exactly -----------------------------


class SpyLoss:
    def __init__(self, value, log):
        self.value, self.log = value, log

    def __truediv__(self, divisor):
        self.log.append(("divided_by", divisor))
        return SpyLoss(self.value / divisor, self.log)

    def backward(self):
        self.log.append(("backward", round(self.value, 6)))

    def detach(self):
        return self

    def item(self):
        return self.value


class SpyModel:
    def __init__(self, log):
        self.log, self.calls, self.training = log, 0, False

    def __call__(self, **batch):
        self.calls += 1
        self.log.append(("forward", int(batch["attention_mask"].sum())))
        return types.SimpleNamespace(loss=SpyLoss(1.0 + self.calls, self.log))

    def train(self):
        self.training = True
        self.log.append(("train",))

    def parameters(self):
        return iter(())


class SpyOptimizer:
    def __init__(self, log):
        self.log, self.steps, self.zeroed = log, 0, 0

    def step(self):
        self.steps += 1
        self.log.append(("step",))

    def zero_grad(self, **kw):
        self.zeroed += 1
        self.log.append(("zero_grad",))


class SpyTorch:
    """Only the surface prepare_training and the teardown actually touch."""

    def __init__(self, log, holder=None, mps_available=True):
        self.log, self.holder = log, holder
        self.optim = types.SimpleNamespace(AdamW=self._adamw)
        self.mps = types.SimpleNamespace(empty_cache=self._empty_cache)
        # Report 16 measures on a machine that has an MPS *backend*, so the
        # stand-in has to say so. ``_empty_cache`` used to ask only whether
        # ``torch.mps.empty_cache`` existed -- true on every build, including
        # the CUDA one where calling it raises -- and this stub passed that
        # check by accident. It now answers the question that is actually
        # asked, and ``mps_available=False`` models the other kind of machine.
        self.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps_available))

    def manual_seed(self, seed):
        self.log.append(("manual_seed", seed))

    def _adamw(self, params, **kw):
        self.log.append(("adamw", list(params), kw))
        return SpyOptimizer(self.log)

    def _empty_cache(self):
        self.log.append(("empty_cache",
                         sorted(self.holder) if self.holder is not None else None))


def spy_encs():
    from src.training.lora import Encoded

    return [Encoded(input_ids=[1, 2, 3, 4], labels=[-100, -100, 3, 4],
                    n_prompt_tokens=2, truncated=False),
            Encoded(input_ids=[5, 6, 7], labels=[-100, 6, 7],
                    n_prompt_tokens=1, truncated=False)]


def test_r5_the_step_backpropagates_the_scaled_loss_and_stores_the_raw_one():
    from src.training.lora import collate

    log = []
    holder = {"model": SpyModel(log), "optimizer": SpyOptimizer(log)}
    step = lr.make_training_step(holder=holder, encs=spy_encs(),
                                 sample_ids=["sA", "sB"], collate_fn=collate,
                                 pad_id=0, device="cpu", grad_accum=8)
    out = step(0, 1)
    assert ("divided_by", 8) in log, "report 15 backpropagates loss / grad_accum"
    assert ("backward", round(2.0 / 8, 6)) in log
    assert out["loss"] == 2.0, "the stored loss is the undivided one"
    assert out["sample_id"] == "sA"


def test_r5_the_step_counts_tokens_from_the_collated_batch():
    from src.training.lora import collate

    log = []
    holder = {"model": SpyModel(log), "optimizer": SpyOptimizer(log)}
    step = lr.make_training_step(holder=holder, encs=spy_encs(),
                                 sample_ids=["sA", "sB"], collate_fn=collate,
                                 pad_id=0, device="cpu", grad_accum=8)
    a = step(0, 1)
    b = step(1, 2)
    assert (a["tokens"], a["supervised_tokens"]) == (4, 2)
    assert (b["tokens"], b["supervised_tokens"]) == (3, 2)


def test_r5_the_optimiser_steps_once_per_grad_accum_rows():
    from src.training.lora import collate

    log = []
    opt = SpyOptimizer(log)
    holder = {"model": SpyModel(log), "optimizer": opt}
    step = lr.make_training_step(holder=holder, encs=spy_encs(),
                                 sample_ids=["sA", "sB"], collate_fn=collate,
                                 pad_id=0, device="cpu", grad_accum=8)
    stepped_at = []
    for position in range(1, 17):
        before = opt.steps
        step(position % 2, position)
        if opt.steps > before:
            stepped_at.append(position)
    assert stepped_at == [8, 16], (
        "an optimiser that steps every row is a different experiment from "
        "exp002's, and its per-row losses cannot be compared with it")


def test_r5_prepare_training_seeds_before_it_builds_and_ends_in_train_mode():
    from src.training.lora import LoraConfig_

    log = []
    torch_mod = SpyTorch(log)
    model = SpyModel(log)
    checked = []
    out = lr.prepare_training(
        torch_mod=torch_mod, cfg=LoraConfig_(), device="cpu",
        build=lambda cfg, device: (log.append(("build", device)) or
                                   (model, {"trainable_parameters": 7})),
        assert_trainable=lambda m: checked.append(m))
    kinds = [entry[0] for entry in log]
    assert kinds.index("manual_seed") < kinds.index("build"), (
        "LoRA dropout draws from the global stream while the adapter is "
        "created, so the seed has to be set first")
    assert ("manual_seed", 0) in log
    assert checked == [model], "only-LoRA-trainable is checked, every time"
    assert model.training is True and out["optimizer"].zeroed >= 1
    adamw = [e for e in log if e[0] == "adamw"][0]
    assert adamw[2] == {"lr": LoraConfig_().learning_rate,
                        "betas": (0.9, 0.999), "eps": 1e-8,
                        "weight_decay": 0.01}
    assert kinds.index("build") < kinds.index("adamw") < kinds.index("train")


def test_r5_teardown_releases_the_model_before_it_measures_empty_cache():
    log = []
    holder = {"model": SpyModel(log), "optimizer": SpyOptimizer(log)}
    torch_mod = SpyTorch(log, holder=holder)
    seconds = lr.make_teardown(torch_mod, holder)()
    assert seconds >= 0.0
    assert ("empty_cache", []) in log, (
        "a teardown clear measured while the adapter and Adam's moments are "
        "still referenced is measuring something else")
    assert holder == {}


# --- 3. the child checks its own plan and source before loading -------------


def test_r5_child_verify_source_accepts_a_clean_session(tmp_path):
    out = init(tmp_path)
    got = lr.child_verify_source(out["paths"]["dir"], out["plan"]["plan_digest"])
    assert got["files_verified"] == 2
    assert got["source_manifest_digest"] == out["session"]["source_manifest_digest"]


@pytest.mark.parametrize("how", ["wrong_digest", "tampered_plan",
                                 "broken_snapshot", "manifest_digest",
                                 "no_session"])
def test_r5_child_verify_source_refuses_before_anything_loads(tmp_path, how):
    out = init(tmp_path)
    d, digest = out["paths"]["dir"], out["plan"]["plan_digest"]
    if how == "wrong_digest":
        digest = "0" * 64
    elif how == "tampered_plan":
        plan = json.loads(out["paths"]["plan"].read_text())
        plan["runs"][0]["declared_rows"] = 999
        out["paths"]["plan"].write_text(json.dumps(plan))
    elif how == "broken_snapshot":
        copy = next(out["paths"]["snapshot"].glob("*"))
        copy.write_text("# not the code the plan was made against\n")
    elif how == "manifest_digest":
        body = json.loads(out["paths"]["session"].read_text())
        body["source_manifest_digest"] = "f" * 64
        out["paths"]["session"].write_text(json.dumps(body))
    else:
        out["paths"]["session"].unlink()
    with pytest.raises(SystemExit):
        lr.child_verify_source(d, digest)


class RecordingDeps(lr.ChildDeps):
    """A deps object that only records that somebody tried to load it."""

    def __init__(self):
        self.loads = []

    def load(self, *, rows):
        self.loads.append(rows)
        raise AssertionError("nothing should have been loaded")


def test_r5_the_child_checks_the_source_before_it_loads_anything(tmp_path):
    out = init(tmp_path)
    copy = next(out["paths"]["snapshot"].glob("*"))
    copy.write_text("# changed after the plan was frozen\n")
    deps = RecordingDeps()
    with pytest.raises(SystemExit):
        lr.run_child(experiment_dir=out["paths"]["dir"], run="b1", rows=500,
                     nonce="n", plan_digest=out["plan"]["plan_digest"],
                     deps=deps)
    assert deps.loads == [], "the tokenizer, the rows and the model come after"
    assert not (out["paths"]["dir"] / "b1.json").exists()


@pytest.mark.parametrize("digest", ["", None, "0" * 64])
def test_r5_the_child_will_not_take_the_plan_digest_on_trust(tmp_path, digest):
    """argv is an input, not evidence.

    Copying the digest out of the command line into the report would make the
    field self-certifying: it would agree with itself no matter what plan.json
    said.
    """
    out = init(tmp_path)
    deps = RecordingDeps()
    with pytest.raises(SystemExit):
        lr.run_child(experiment_dir=out["paths"]["dir"], run="b1", rows=500,
                     nonce="n", plan_digest=digest, deps=deps)
    assert deps.loads == []


# --- 4. the child report carries section 4.7 --------------------------------


def child_report(tmp_path, *, run="b1", rows=500, out=None, deps=None):
    out = out or init(tmp_path)
    deps = deps or lr.FakeChildDeps(pool_rows=lr.POOL_ROWS)
    rc = lr.run_child(experiment_dir=out["paths"]["dir"], run=run, rows=rows,
                      nonce="n", plan_digest=out["plan"]["plan_digest"],
                      deps=deps)
    return rc, json.loads((out["paths"]["dir"] / f"{run}.json").read_text()), out


def test_r5_the_child_report_carries_every_declared_field(tmp_path):
    rc, report, out = child_report(tmp_path)
    assert rc == 0
    assert report["experiment_id"] == "exp016a"
    assert report["pool_pairs"] == lr.POOL_PAIRS
    assert report["pool_rows"] == lr.POOL_ROWS
    assert report["rows_requested"] == 500
    assert report["child_source_check"]["source_manifest_digest"] == \
        out["session"]["source_manifest_digest"]
    assert set(report["preflight"]) >= {"swap_used_gb",
                                        "memory_pressure_percent_free",
                                        "free_plus_inactive_gb",
                                        "normalized_load_1m"}
    assert lr.replay_child(report)["problems"] == []


@pytest.mark.parametrize("field", ["experiment_id", "pool_pairs", "pool_rows",
                                   "rows_requested", "child_source_check",
                                   "preflight", "memory", "metrics",
                                   "float_storage",
                                   "scheduled_empty_cache_every",
                                   "between_row_overhead_breakdown",
                                   "teardown_empty_cache_seconds"])
def test_r5_replay_is_fail_closed_on_a_missing_field(tmp_path, field):
    _, report, _ = child_report(tmp_path)
    del report[field]
    assert any(field in p for p in lr.replay_child(report)["problems"]), (
        f"a report that simply omits {field} must not replay clean")


@pytest.mark.parametrize("field,value", [("pool_pairs", 249),
                                         ("pool_rows", 1999),
                                         ("rows_requested", 499)])
def test_r5_replay_checks_the_pool_and_the_requested_length(tmp_path, field,
                                                            value):
    _, report, _ = child_report(tmp_path)
    report[field] = value
    assert any(field in p for p in lr.replay_child(report)["problems"])


def test_r5_replay_recomputes_the_completed_input_digest(tmp_path):
    _, report, _ = child_report(tmp_path)
    report["completed_input_digest"] = "z" * 64
    problems = lr.replay_child(report)["problems"]
    assert any("recompute" in p for p in problems)


def test_r5_replay_catches_a_relabelled_row(tmp_path):
    """Renaming a row and re-signing the top-level digest is still caught."""
    _, report, _ = child_report(tmp_path)
    report["per_row"][7]["sample_id"] = "somebody-elses-row"
    problems = lr.replay_child(report)["problems"]
    assert any("recompute" in p for p in problems)


# --- 5. section 4.6 field semantics -----------------------------------------


def three_runs(tmp_path):
    """One deps object, one order, three declared lengths -- the real B."""
    out = init(tmp_path)
    deps = lr.FakeChildDeps(pool_rows=lr.POOL_ROWS)
    reports = []
    for run, k in zip(lr.RUN_IDS, lr.LENGTHS):
        rc = lr.run_child(experiment_dir=out["paths"]["dir"], run=run, rows=k,
                          nonce="n", plan_digest=out["plan"]["plan_digest"],
                          deps=deps)
        assert rc == 0
        reports.append(json.loads(
            (out["paths"]["dir"] / f"{run}.json").read_text()))
    return out, reports


def test_r5_the_training_order_digest_covers_the_whole_pool(tmp_path):
    """One permutation of 2,000 rows, shared by all three runs.

    Digesting only the first k rows would give b1, b2 and b3 three different
    values for a field section 4.6 requires to be identical -- and would stop
    recording the thing it is there to record, which is the order the pool was
    permuted into before anybody chose a length.
    """
    _, reports = three_runs(tmp_path)
    digests = {r["provenance"]["training_order_digest"] for r in reports}
    assert len(digests) == 1, "all three runs share one 2,000-row permutation"
    truncated = lr._digest_ids(
        r["sample_id"] for r in reports[0]["per_row"][:500])
    assert digests != {truncated}, "it is the full order, not the b1 prefix"
    assert len({r["input_order_digest"] for r in reports}) == 3, (
        "the per-run declared order is the field that varies with k")


def test_r5_max_rows_is_the_only_thing_allowed_to_differ(tmp_path):
    _, reports = three_runs(tmp_path)
    assert lr.cross_run_problems(reports) == []
    assert lr.cross_run_identical(reports) is True
    intervals = [r["provenance"]["measurement_intervals"] for r in reports]
    assert [i["max_rows"] for i in intervals] == list(lr.LENGTHS)
    assert len({i["window"] for i in intervals}) == 1


def test_r5_a_max_rows_that_is_not_this_runs_k_is_refused(tmp_path):
    _, reports = three_runs(tmp_path)
    reports[1]["provenance"]["measurement_intervals"]["max_rows"] = 500
    assert any("max_rows" in p for p in lr.cross_run_problems(reports))
    assert lr.cross_run_identical(reports) is False


def test_r5_a_missing_max_rows_is_not_agreement(tmp_path):
    _, reports = three_runs(tmp_path)
    for report in reports:
        del report["provenance"]["measurement_intervals"]["max_rows"]
    assert any("max_rows" in p for p in lr.cross_run_problems(reports))


def test_r5_any_other_interval_difference_is_still_refused(tmp_path):
    _, reports = three_runs(tmp_path)
    reports[2]["provenance"]["measurement_intervals"]["window"] = 999
    assert lr.cross_run_identical(reports) is False


def test_r5_one_child_produces_a_prefix_consistent_b1_b2_b3(tmp_path):
    """The item this whole round exists for: b1/b2/b3 from one run_child.

    Same function, same deps, three lengths. If prefix consistency, the
    cross-run comparison and the headline cannot all hold here, they cannot
    hold on the real thing either, and nobody would find out until three
    boots had been spent.
    """
    _, reports = three_runs(tmp_path)
    assert [r["rows_completed"] for r in reports] == list(lr.LENGTHS)
    prefix = lr.prefix_consistency(reports)
    assert prefix["verdict"] == "passed", prefix["problems"]
    for report in reports:
        assert lr.replay_child(report)["problems"] == []
    head = lr.headline_full_B(
        runs=[{"run_id": r["run_id"], "outcome": "completed",
               "tool_failure": None, "safety_reason": None} for r in reports],
        boots=["boot-1", "boot-2", "boot-3"], prefix=prefix,
        q1={"value": "holds_to_2000"}, q2={"value": "stable"},
        cross_run_identical=lr.cross_run_identical(reports),
        contract_problems=lr.cross_run_problems(reports))
    assert head["allowed"] is True, head["reasons"]


# --- 6. the CLI: the real gate call, and what deleting the refusal reaches ---


def test_r5_the_launcher_calls_wait_for_recovery_by_its_real_names(monkeypatch):
    """Bound against the real signature, so a renamed keyword is a failure."""
    from src.training import preflight

    real = preflight.wait_for_recovery
    seen = {}

    def spy(thresholds, **kw):
        inspect.signature(real).bind(thresholds, **kw)
        seen.update(kw)
        return {"passed": True, "polls": [], "waited_seconds": 0.0,
                "reason": None}

    monkeypatch.setattr(preflight, "wait_for_recovery", spy)
    mod = load_cli()
    launcher = mod.Launcher(wd.SafetySpec())
    launcher.gate({"swap_used_gb": 1.0}, lr.GATE_POLICY)
    assert seen == {"needed_consecutive": 3, "poll_seconds": 30,
                    "max_wait_seconds": 900}


def test_r5_the_gate_record_the_launcher_writes_can_be_replayed(monkeypatch,
                                                                tmp_path):
    """A gate record replay cannot recompute is a gate nobody can check."""
    from src.training import preflight
    from src.training.preflight import calibrate, evaluate_gate, thresholds_from

    thresholds = thresholds_from(calibrate([GOOD_SAMPLE] * 10))

    def spy(thresholds_, **kw):
        polls = []
        for i in range(3):
            judged = evaluate_gate(GOOD_SAMPLE, thresholds_)
            polls.append({"poll": i + 1, "elapsed_seconds": i * 30.0,
                          "sample": dict(GOOD_SAMPLE), "passed": judged["passed"],
                          "failed_metrics": judged["failed"],
                          "consecutive_passes": i + 1})
        return {"passed": True, "polls": polls, "waited_seconds": 60.0,
                "consecutive_passes_required": 3, "reason": None}

    monkeypatch.setattr(preflight, "wait_for_recovery", spy)
    mod = load_cli()
    record = mod.Launcher(wd.SafetySpec()).gate(thresholds, lr.GATE_POLICY)
    assert lr.replay_gate_polls(record, thresholds, lr.GATE_POLICY) == []


# The two tests that used to live here -- "approval is exactly the deletion
# of the refusal", and "with the refusal in place no command runs" -- were
# written against a lock that no longer exists. Their subject matter moved to
# the round 13 section above, where the same dispatch is checked without a
# refusal to neutralise first.


def test_r5_the_dispatch_reaches_a_read_only_command_too(monkeypatch):
    """The read-only half of the dispatch, checked the same way.

    The executing flags get their own coverage in round 13. This one exists
    because the four cmd_* that were behind the lock are not the only wiring
    in main(), and a rename there would otherwise be caught by nothing.
    """
    mod = load_cli()
    seen = []
    for cmd in ("cmd_verify", "cmd_from_json", "cmd_session_status",
                "cmd_show_plan", "cmd_microbenchmark"):
        monkeypatch.setattr(mod, cmd, lambda args, c=cmd: seen.append(c) or 0)
    for argv, want in ((["--show-plan"], "cmd_show_plan"),
                       (["--microbenchmark"], "cmd_microbenchmark"),
                       (["--verify", "--experiment-id", "e"], "cmd_verify"),
                       (["--from-json", "--experiment-id", "e"],
                        "cmd_from_json"),
                       (["--session-status", "--experiment-id", "e"],
                        "cmd_session_status")):
        seen.clear()
        assert mod.main(argv) == 0
        assert seen == [want], argv


# --- 7. the full prefix is replayed before the gate is polled ---------------


def counting_gate():
    calls = []

    def gate():
        calls.append(1)
        return gate_record(THRESHOLDS)

    gate.calls = calls
    return gate


def rewrite_chained(paths, fragment, mutate):
    """Rewrite one event, then repair the chain and every started backlink.

    Without the repair each tamper looks identical to the validator -- a
    broken digest chain -- and a test that only ever sees that proves nothing
    about the check it was written for.
    """
    events = events_of(paths)
    idx = next(i for i, p in enumerate(events) if fragment in p.name)
    rewrite(events[idx], mutate)
    for j in range(idx + 1, len(events)):
        prev = hashlib.sha256(events[j - 1].read_bytes()).hexdigest()
        body = json.loads(events[j].read_text())
        patch = {"previous_event_digest": prev}
        if body.get("event") == lr.EVENT_FINISHED:
            run = body["run_id"]
            start = next(p for p in events
                         if f"-{run}-{lr.EVENT_STARTED}.json" in p.name)
            patch["started_event_digest"] = hashlib.sha256(
                start.read_bytes()).hexdigest()
        rewrite(events[j], lambda b, p=patch: b.update(p))


def one_measured_run(tmp_path):
    global THRESHOLDS
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(tmp_path))
    assert out["ok"], out.get("problems")
    return lr.session_paths("exp016a", root=tmp_path / "reports")


def break_prefix(paths, how):
    d = paths["dir"]
    if how == "report_digest":
        rewrite_chained(paths, "b1-measurement_finished",
                        lambda b: b.pop("report_sha256", None))
    elif how == "watchdog_digest":
        rewrite_chained(paths, "b1-measurement_finished",
                        lambda b: b.pop("watchdog_sha256", None))
    elif how == "watchdog_log":
        (d / "b1.watchdog.jsonl").unlink()
    elif how == "launch_record":
        (d / "b1.launch.json").unlink()
    elif how == "launch_nonce":
        body = json.loads((d / "b1.launch.json").read_text())
        body["nonce"] = "SOMEBODY-ELSES-LAUNCH"
        (d / "b1.launch.json").write_text(json.dumps(body))
    elif how == "gate_poll":
        def flip(b):
            b["gate"]["polls"][0]["passed"] = False
        rewrite_chained(paths, "b1-measurement_started", flip)
    elif how == "child_report":
        report = json.loads((d / "b1.json").read_text())
        report["per_row"][3]["compute_seconds"] = 99.0
        (d / "b1.json").write_text(json.dumps(report))
    else:
        raise AssertionError(how)


BROKEN_PREFIXES = ["report_digest", "watchdog_digest", "watchdog_log",
                   "launch_record", "launch_nonce", "gate_poll",
                   "child_report"]


@pytest.mark.parametrize("how", BROKEN_PREFIXES)
def test_r5_a_broken_prefix_stops_the_next_run_before_the_gate(tmp_path, how):
    paths = one_measured_run(tmp_path)
    break_prefix(paths, how)
    before = len(lr.read_journal(paths["dir"]))
    gate = counting_gate()
    out = lr.session_next("exp016a", **next_args(
        tmp_path, gate=gate, boot={"boot_fingerprint": "boot-2"}))
    assert out["ok"] is False
    assert out["boot_consumed"] is False
    assert out.get("preconditions_failed") is True
    assert gate.calls == [], "the gate is polled only after the prefix replays"
    assert len(lr.read_journal(paths["dir"])) == before, (
        "a session whose own records do not add up gets no new events")


@pytest.mark.parametrize("how", BROKEN_PREFIXES)
def test_r5_the_precondition_replay_names_what_it_found(tmp_path, how):
    paths = one_measured_run(tmp_path)
    break_prefix(paths, how)
    problems = lr.session_preconditions("exp016a", root=tmp_path / "reports",
                                        check_working_tree=False)
    assert problems, f"{how} left the preconditions silent"
    wanted = {"report_digest": "report digest",
              "watchdog_digest": "watchdog log digest",
              "watchdog_log": "watchdog",
              "launch_record": "launch",
              "launch_nonce": "nonce",
              "gate_poll": "gate poll",
              "child_report": "b1"}[how]
    assert any(wanted in p for p in problems), problems


def test_r5_a_clean_prefix_still_reaches_the_gate_and_the_next_run(tmp_path):
    paths = one_measured_run(tmp_path)
    gate = counting_gate()
    out = lr.session_next("exp016a", **next_args(
        tmp_path, gate=gate, boot={"boot_fingerprint": "boot-2"}))
    assert out["ok"], out.get("problems")
    assert gate.calls == [1]
    st = lr.session_state("exp016a", root=tmp_path / "reports")
    assert st["completed"] == ["b1", "b2"]
    assert lr.session_preconditions("exp016a", root=tmp_path / "reports",
                                    check_working_tree=False) == []


# ===========================================================================
# Round 6. Codex's third pass: the default dependency, the whole schema, and
# the four identity fields.
#
# Round 5 fixed what ProductionChildDeps imports and never noticed that
# nothing constructs it any more; it required the section 4.7 fields to be
# present and never checked what was inside them; and it required a launch
# record to exist without ever comparing it with the child it claims to
# describe. All three are the same mistake in different places -- checking
# that something is there instead of checking that it says the right thing.
# ===========================================================================


class RecorderDeps(lr.ChildDeps):
    """Stands in for ProductionChildDeps and records that it was built."""

    built: list["RecorderDeps"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.loads: list[int] = []
        RecorderDeps.built.append(self)

    def load(self, *, rows):
        self.loads.append(rows)
        return lr.FakeChildDeps(pool_rows=lr.POOL_ROWS).load(rows=rows)


@pytest.fixture
def recorder(monkeypatch):
    RecorderDeps.built = []
    monkeypatch.setattr(lr, "ProductionChildDeps", RecorderDeps)
    return RecorderDeps


def test_r6_run_child_builds_the_production_deps_when_none_are_injected(
        tmp_path, recorder):
    """Every test so far passed its own deps, so nothing built the real ones.

    ``--child`` passes none. A child that has been exercised only with an
    injected stand-in has never executed the line that decides what it
    measures.
    """
    out = init(tmp_path)
    rc = lr.run_child(experiment_dir=out["paths"]["dir"], run="b1", rows=500,
                      nonce="n", plan_digest=out["plan"]["plan_digest"])
    assert rc == 0
    assert len(recorder.built) == 1, "exactly one deps object is constructed"
    assert recorder.built[0].loads == [500], "and it is loaded once, for k rows"
    report = json.loads((out["paths"]["dir"] / "b1.json").read_text())
    assert report["rows_completed"] == 500
    assert lr.replay_child(report)["problems"] == []


def test_r6_cmd_child_reaches_the_production_deps(tmp_path, recorder):
    """The dispatch --child would use, with the arguments the parent passes."""
    out = init(tmp_path)
    mod = load_cli()
    args = mod.build_parser().parse_args([
        "--child", "--experiment-dir", str(out["paths"]["dir"]),
        "--run", "b1", "--rows", "500", "--nonce", "n",
        "--plan-digest", out["plan"]["plan_digest"]])
    assert mod.cmd_child(args) == 0
    assert len(recorder.built) == 1 and recorder.built[0].loads == [500]
    assert (out["paths"]["dir"] / "b1.json").exists()


# --- 2. the whole of section 4.7, fail-closed -------------------------------


def test_r6_the_schema_list_is_exactly_what_the_child_writes(tmp_path):
    """One list, checked against the writer, so the two cannot drift."""
    _, report, _ = child_report(tmp_path)
    assert set(report) == set(lr.CHILD_SCHEMA_KEYS)
    assert set(lr.CHILD_REQUIRED) <= set(lr.CHILD_SCHEMA_KEYS)


@pytest.mark.parametrize("field", ["schema_version", "kind", "child_pid",
                                   "child_pgid", "child_start_identity",
                                   "started_at", "finished_at",
                                   "stopped_early", "tool_failure",
                                   "teardown_empty_cache_calls",
                                   "model_compute_seconds"])
def test_r6_replay_refuses_a_report_missing_a_schema_field(tmp_path, field):
    _, report, _ = child_report(tmp_path)
    del report[field]
    assert any(field in p for p in lr.replay_child(report)["problems"])


@pytest.mark.parametrize("field,value", [("schema_version", 2),
                                         ("schema_version", "1"),
                                         ("kind", "longrun_aggregate"),
                                         ("condition", "continuous")])
def test_r6_replay_refuses_a_report_that_is_not_this_schema(tmp_path, field,
                                                            value):
    _, report, _ = child_report(tmp_path)
    report[field] = value
    assert any(field in p for p in lr.replay_child(report)["problems"])


def test_r6_replay_refuses_a_field_nobody_declared(tmp_path):
    """An undeclared field is a number no replay recomputes.

    The aggregate comparison has refused unknown keys since round three; the
    child report is the document every verdict is derived from, so it gets the
    same treatment.
    """
    _, report, _ = child_report(tmp_path)
    report["adoption_supported"] = True
    problems = lr.replay_child(report)["problems"]
    assert any("adoption_supported" in p for p in problems)


@pytest.mark.parametrize("block", ["provenance", "preflight",
                                   "child_source_check", "clocks",
                                   "float_storage",
                                   "between_row_overhead_breakdown",
                                   "scheduled_empty_cache_cost"])
def test_r6_an_empty_nested_block_is_not_a_present_field(tmp_path, block):
    _, report, _ = child_report(tmp_path)
    report[block] = {}
    assert any(block in p or "clock" in p
               for p in lr.replay_child(report)["problems"]), block


@pytest.mark.parametrize("metric", ["swap_used_gb",
                                    "memory_pressure_percent_free",
                                    "free_plus_inactive_gb",
                                    "normalized_load_1m"])
def test_r6_preflight_carries_all_four_gated_readings(tmp_path, metric):
    _, report, _ = child_report(tmp_path)
    del report["preflight"][metric]
    assert any(metric in p for p in lr.replay_child(report)["problems"])


@pytest.mark.parametrize("field", ["plan_digest", "source_manifest_digest",
                                   "files_verified"])
def test_r6_child_source_check_records_what_it_verified(tmp_path, field):
    _, report, _ = child_report(tmp_path)
    del report["child_source_check"][field]
    assert any(field in p for p in lr.replay_child(report)["problems"])


@pytest.mark.parametrize("field", ["seconds_rounded", "loss_rounded"])
def test_r6_a_report_that_admits_rounding_is_refused(tmp_path, field):
    """Section 4.7 stores seconds and losses unrounded, and says so."""
    _, report, _ = child_report(tmp_path)
    report["float_storage"][field] = True
    assert any(field in p for p in lr.replay_child(report)["problems"])


@pytest.mark.parametrize("field", list(lr.PROVENANCE_FIELDS[:4]))
def test_r6_replay_alone_refuses_incomplete_provenance(tmp_path, field):
    """Not only the session-level checks: a lone report has to fail too."""
    _, report, _ = child_report(tmp_path)
    del report["provenance"][field]
    assert any(field in p for p in lr.replay_child(report)["problems"])


# --- 3. the child's source check against the session it ran in --------------


@pytest.mark.parametrize("how", ["plan_digest", "manifest_digest",
                                 "files_verified"])
def test_r6_a_source_check_that_disagrees_with_the_session_is_refused(
        tmp_path, how):
    """The child says what it verified; the session says what there was.

    A ``child_source_check`` nobody compares with the session is the child
    marking its own homework -- it would pass with any three plausible values
    in it.
    """
    paths = one_measured_run(tmp_path)
    report = json.loads((paths["dir"] / "b1.json").read_text())
    key = {"plan_digest": "plan_digest",
           "manifest_digest": "source_manifest_digest",
           "files_verified": "files_verified"}[how]
    report["child_source_check"][key] = 99 if how == "files_verified" else "z" * 64
    resign_report(paths, "b1", report)
    problems = verify(tmp_path)
    assert any("child_source_check" in p for p in problems), problems
    assert any("child_source_check" in p for p in lr.session_preconditions(
        "exp016a", root=sess(tmp_path), check_working_tree=False))


# --- 4. section 4.8 7b: the four identity fields ----------------------------


def resign_report(paths, run, report):
    """Rewrite a child report and re-record its digest, chain included."""
    rp = paths["dir"] / f"{run}.json"
    rp.write_text(json.dumps(report))
    fresh = hashlib.sha256(rp.read_bytes()).hexdigest()
    rewrite_chained(paths, f"{run}-{lr.EVENT_FINISHED}",
                    lambda b: b.update({"report_sha256": fresh}))


def resign_launch(paths, run, mutate):
    """Change the launch record and update every digest that covers it.

    A bare one-field edit trips a digest check on its own, which would let a
    test pass without the identity comparison existing at all. So the tamper
    is re-signed: the record is rewritten, the digest the finished event
    records is updated, and the journal chain is repaired. What is left can
    only be caught by comparing the record with the child it describes.
    """
    path = paths["dir"] / f"{run}.launch.json"
    body = json.loads(path.read_text())
    mutate(body)
    path.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True,
                               indent=1))
    fresh = hashlib.sha256(path.read_bytes()).hexdigest()
    rewrite_chained(paths, f"{run}-{lr.EVENT_FINISHED}",
                    lambda b: b.update({"launch_sha256": fresh}))


def test_r6_the_child_reports_all_four_identity_fields(tmp_path):
    _, report, _ = child_report(tmp_path)
    assert report["child_pid"] == os.getpid()
    assert report["child_pgid"] == os.getpgrp()
    assert report["child_start_identity"] == \
        wd.process_start_identity(os.getpid())


def test_r6_a_clean_run_agrees_on_its_identity(tmp_path):
    paths = one_measured_run(tmp_path)
    launch = json.loads((paths["dir"] / "b1.launch.json").read_text())
    report = json.loads((paths["dir"] / "b1.json").read_text())
    assert (report["nonce"], report["child_pid"], report["child_pgid"],
            report["child_start_identity"]) == (
        launch["nonce"], launch["child_pid"], launch["child_pgid"],
        launch["child_start_identity"])
    assert verify(tmp_path) == []


IDENTITY_FIELDS = ["nonce", "child_pid", "child_pgid", "child_start_identity"]


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
def test_r6_a_resigned_identity_tamper_is_still_refused(tmp_path, field):
    paths = one_measured_run(tmp_path)
    other = {"nonce": "SOMEBODY-ELSES-LAUNCH", "child_pid": 999999,
             "child_pgid": 999999, "child_start_identity": "some other Tuesday"}
    resign_launch(paths, "b1", lambda b: b.update({field: other[field]}))
    problems = verify(tmp_path)
    assert any(field in p for p in problems), problems
    pre = lr.session_preconditions("exp016a", root=sess(tmp_path),
                                   check_working_tree=False)
    assert any(field in p for p in pre), pre


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
def test_r6_an_identity_tamper_stops_the_next_run_before_the_gate(tmp_path,
                                                                  field):
    paths = one_measured_run(tmp_path)
    other = {"nonce": "SOMEBODY-ELSES-LAUNCH", "child_pid": 999999,
             "child_pgid": 999999, "child_start_identity": "some other Tuesday"}
    resign_launch(paths, "b1", lambda b: b.update({field: other[field]}))
    before = len(lr.read_journal(paths["dir"]))
    gate = counting_gate()
    out = lr.session_next("exp016a", **next_args(
        tmp_path, gate=gate, boot={"boot_fingerprint": "boot-2"}))
    assert out["ok"] is False and out["boot_consumed"] is False
    assert out.get("preconditions_failed") is True
    assert gate.calls == []
    assert len(lr.read_journal(paths["dir"])) == before


def test_r6_the_launch_record_is_under_a_digest_like_everything_else(tmp_path):
    """Without a recorded digest the record could simply be rewritten."""
    paths = one_measured_run(tmp_path)
    finished = json.loads(next(
        p for p in events_of(paths)
        if f"b1-{lr.EVENT_FINISHED}" in p.name).read_text())
    path = paths["dir"] / "b1.launch.json"
    assert finished["launch_sha256"] == \
        hashlib.sha256(path.read_bytes()).hexdigest()


def test_r6_a_completed_run_without_a_launch_digest_is_refused(tmp_path):
    paths = one_measured_run(tmp_path)
    rewrite_chained(paths, f"b1-{lr.EVENT_FINISHED}",
                    lambda b: b.pop("launch_sha256", None))
    assert any("launch" in p for p in verify(tmp_path))
    assert any("launch" in p for p in lr.session_preconditions(
        "exp016a", root=sess(tmp_path), check_working_tree=False))


def test_r6_an_unsigned_launch_record_edit_is_refused(tmp_path):
    """The plain tamper, caught by the digest rather than by the comparison."""
    paths = one_measured_run(tmp_path)
    path = paths["dir"] / "b1.launch.json"
    body = json.loads(path.read_text())
    body["child_pid"] = 4242
    path.write_text(json.dumps(body))
    assert any("launch" in p for p in verify(tmp_path))


# ===========================================================================
# Round 7. Three holes the previous rounds left open.
#
# The identity checks compared two accounts of the same process and the
# watchdog -- the only party that was actually watching it -- was not one of
# them, so rewriting both accounts together agreed with itself. The schema
# checks asked whether a field was there and never what was in it, so
# 999999 seconds of compute passed. And the parent read "the watchdog has
# exited" before "the child has finished", so a run that ended perfectly
# normally could be filed as a tool failure depending on which process the
# kernel reaped first.
# ===========================================================================


# --- 1. the watchdog log is the third account -------------------------------


def test_r7_the_watchdog_log_records_who_it_was_watching(tmp_path):
    identity = ident(pid=4242, pgid=4242)
    log = wd.WatchdogLog(tmp_path / "b1.watchdog.jsonl")
    wd.watchdog_loop(log, identity, wd.SafetySpec(),
                     **{k: v for k, v in clean_loop_args(
                         tmp_path, identity, max_polls=2).items()
                        if k not in ("directory", "prefix")},
                     directory=tmp_path, prefix="b1")
    log.seal()
    records = wd.replay_watchdog_log(tmp_path / "b1.watchdog.jsonl")["records"]
    assert records, "the loop wrote something"
    for rec in records:
        assert rec["nonce"] == identity.nonce
        assert rec["child_pid"] == 4242 and rec["child_pgid"] == 4242
        assert rec["child_start_identity"] == "T0"


def wd_records(n=3, **over):
    """Hand-built records carrying an identity, for the replay tests.

    The violation counters come from a real SafetyState rather than from a
    guess, so these records are ones the semantic replay would otherwise
    accept and the identity check is the only thing under test.
    """
    base = {"nonce": "N", "child_pid": 7, "child_pgid": 7,
            "child_start_identity": "T0"}
    base.update(over)
    state = wd.SafetyState()
    out = []
    for i in range(n):
        verdict = state.observe(sample(), float(i * 5), progress=None)
        out.append({"seq": i, "prev_digest": None, "monotonic": float(i * 5),
                    "wall_clock": "Z", **sample(),
                    "failed": verdict["failed"],
                    "failure_streak": state.failure_streak,
                    "violations": verdict["violations"],
                    "action": "poll", "progress": None, "heartbeat_seq": i,
                    **base})
    return out


WANTED_IDENTITY = {"nonce": "N", "child_pid": 7, "child_pgid": 7,
                   "child_start_identity": "T0"}


@pytest.mark.parametrize("field", list(lr.IDENTITY_FIELDS))
def test_r7_a_watchdog_log_naming_another_process_is_refused(field):
    records = wd_records()
    for rec in records:
        rec[field] = "somebody else" if isinstance(rec[field], str) else 999999
    out = wd.replay_watchdog_semantics(records, wd.SafetySpec(),
                                       identity=WANTED_IDENTITY)
    assert any(field in p for p in out["problems"]), out["problems"]


@pytest.mark.parametrize("field", list(lr.IDENTITY_FIELDS))
def test_r7_a_watchdog_log_that_changes_its_mind_is_refused(field):
    """One spliced record is enough: the log describes one process."""
    records = wd_records()
    records[1][field] = "elsewhere" if isinstance(records[1][field], str) else 1
    out = wd.replay_watchdog_semantics(records, wd.SafetySpec(),
                                       identity=WANTED_IDENTITY)
    assert any(field in p for p in out["problems"]), out["problems"]


@pytest.mark.parametrize("field", list(lr.IDENTITY_FIELDS))
def test_r7_a_watchdog_log_with_no_identity_is_refused(field):
    records = wd_records()
    for rec in records:
        del rec[field]
    out = wd.replay_watchdog_semantics(records, wd.SafetySpec(),
                                       identity=WANTED_IDENTITY)
    assert any(field in p for p in out["problems"]), out["problems"]


def test_r7_a_matching_watchdog_log_replays_clean():
    out = wd.replay_watchdog_semantics(wd_records(), wd.SafetySpec(),
                                       identity=WANTED_IDENTITY)
    assert out["problems"] == []


def resign_child_identity(paths, run, values):
    """Rewrite both accounts of the child's identity and re-sign both.

    This is the attack the previous round could not see: the launch record
    and the child report are the only two documents it compared, so changing
    them together left nothing to disagree with.
    """
    launch = paths["dir"] / f"{run}.launch.json"
    body = json.loads(launch.read_text())
    body.update(values)
    launch.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                 indent=1))
    report = json.loads((paths["dir"] / f"{run}.json").read_text())
    report.update(values)
    (paths["dir"] / f"{run}.json").write_text(json.dumps(report))
    fresh_launch = hashlib.sha256(launch.read_bytes()).hexdigest()
    fresh_report = hashlib.sha256(
        (paths["dir"] / f"{run}.json").read_bytes()).hexdigest()
    rewrite_chained(paths, f"{run}-{lr.EVENT_FINISHED}",
                    lambda b: b.update({"launch_sha256": fresh_launch,
                                        "report_sha256": fresh_report}))


@pytest.mark.parametrize("field", list(lr.IDENTITY_FIELDS))
def test_r7_rewriting_both_accounts_is_still_refused(tmp_path, field):
    paths = one_measured_run(tmp_path)
    value = "somebody else" if field in ("nonce", "child_start_identity") \
        else 999999
    resign_child_identity(paths, "b1", {field: value})
    problems = verify(tmp_path)
    assert any(field in p for p in problems), problems
    pre = lr.session_preconditions("exp016a", root=sess(tmp_path),
                                   check_working_tree=False)
    assert any(field in p for p in pre), pre


@pytest.mark.parametrize("field", list(lr.IDENTITY_FIELDS))
def test_r7_rewriting_both_accounts_stops_the_next_run_before_the_gate(
        tmp_path, field):
    paths = one_measured_run(tmp_path)
    value = "somebody else" if field in ("nonce", "child_start_identity") \
        else 999999
    resign_child_identity(paths, "b1", {field: value})
    before = len(lr.read_journal(paths["dir"]))
    gate = counting_gate()
    out = lr.session_next("exp016a", **next_args(
        tmp_path, gate=gate, boot={"boot_fingerprint": "boot-2"}))
    assert out["ok"] is False and out["boot_consumed"] is False
    assert gate.calls == []
    assert len(lr.read_journal(paths["dir"])) == before


# --- 2. replay_child checks what is in the fields ---------------------------


@pytest.mark.parametrize("field,value", [
    ("model_compute_seconds", 999999.0),
    ("end_to_end_seconds", 999999.0),
    ("teardown_empty_cache_calls", 999),
    ("teardown_empty_cache_calls", 0),
    ("teardown_empty_cache_seconds", -1.0),
    ("rows_completed", -1),
    ("child_pid", "4242"),
    ("child_pgid", None),
    ("child_start_identity", ""),
    ("pool_pairs", "250"),
])
def test_r7_replay_refuses_a_field_whose_value_is_wrong(tmp_path, field, value):
    _, report, _ = child_report(tmp_path)
    report[field] = value
    assert any(field in p for p in lr.replay_child(report)["problems"]), field


@pytest.mark.parametrize("clock", ["model_load_seconds",
                                   "condition_clock_seconds",
                                   "process_clock_seconds"])
def test_r7_a_negative_clock_is_refused(tmp_path, clock):
    _, report, _ = child_report(tmp_path)
    report["clocks"][clock] = -1.0
    assert any(clock in p for p in lr.replay_child(report)["problems"])


@pytest.mark.parametrize("field", ["compute_seconds", "end_to_end_seconds"])
def test_r7_a_negative_row_duration_is_refused(tmp_path, field):
    _, report, _ = child_report(tmp_path)
    report["per_row"][3][field] = -0.5
    assert any("row 4" in p for p in lr.replay_child(report)["problems"])


def test_r7_a_row_that_took_less_end_to_end_than_it_computed_is_refused(tmp_path):
    _, report, _ = child_report(tmp_path)
    report["per_row"][3]["end_to_end_seconds"] = \
        report["per_row"][3]["compute_seconds"] / 2
    assert any("row 4" in p for p in lr.replay_child(report)["problems"])


@pytest.mark.parametrize("field,value", [("tokens", 0), ("tokens", -5),
                                         ("tokens", 1.5),
                                         ("supervised_tokens", -1),
                                         ("loss", float("inf")),
                                         ("loss", "0.5")])
def test_r7_a_row_with_an_impossible_measurement_is_refused(tmp_path, field,
                                                            value):
    _, report, _ = child_report(tmp_path)
    report["per_row"][3][field] = value
    assert any("row 4" in p for p in lr.replay_child(report)["problems"])


def test_r7_more_supervised_tokens_than_tokens_is_refused(tmp_path):
    _, report, _ = child_report(tmp_path)
    report["per_row"][3]["supervised_tokens"] = \
        report["per_row"][3]["tokens"] + 1
    assert any("row 4" in p for p in lr.replay_child(report)["problems"])


@pytest.mark.parametrize("value", ["lots", True, float("nan")])
def test_r7_a_preflight_reading_that_is_not_a_number_is_refused(tmp_path, value):
    _, report, _ = child_report(tmp_path)
    report["preflight"]["swap_used_gb"] = value
    assert any("swap_used_gb" in p for p in lr.replay_child(report)["problems"])


@pytest.mark.parametrize("field", ["scheduled_empty_cache_seconds",
                                   "memory_probe_seconds",
                                   "unattributed_seconds"])
def test_r7_the_overhead_breakdown_is_recomputed(tmp_path, field):
    _, report, _ = child_report(tmp_path)
    report["between_row_overhead_breakdown"][field] = 999.0
    assert any(field in p for p in lr.replay_child(report)["problems"])


def test_r7_a_memory_probe_outside_the_run_is_refused(tmp_path):
    _, report, _ = child_report(tmp_path)
    report["memory"][1]["row"] = 99999
    assert any("memory" in p for p in lr.replay_child(report)["problems"])


def test_r7_a_negative_probe_cost_is_refused(tmp_path):
    _, report, _ = child_report(tmp_path)
    report["memory"][1]["probe_seconds"] = -0.1
    assert any("probe_seconds" in p for p in lr.replay_child(report)["problems"])


# --- the stopped_early block ------------------------------------------------


def stopped_child(tmp_path, *, row=25):
    out = init(tmp_path)
    d = out["paths"]["dir"]
    wd.write_stop_request(d, prefix="b1", reason="swap_used_gb",
                          rule="swap_used_gb", nonce="n", monotonic=1.0,
                          wall_clock="Z", sampled_values=sample(swap=99.0))
    deps = lr.FakeChildDeps(pool_rows=lr.POOL_ROWS, stop_at_row=row)
    lr.run_child(experiment_dir=d, run="b1", rows=500, nonce="n",
                 plan_digest=out["plan"]["plan_digest"], deps=deps)
    return json.loads((d / "b1.json").read_text()), out


STOP_FIELDS = ["reason", "rule", "row", "sampled_values",
               "condition_clock_seconds", "process_clock_seconds",
               "requested_by", "stop_request_sha256"]


def test_r7_a_stopped_run_records_the_whole_block(tmp_path):
    report, _ = stopped_child(tmp_path)
    stopped = report["stopped_early"]
    assert sorted(stopped) == sorted(STOP_FIELDS)
    assert stopped["row"] == report["rows_completed"] == 25
    assert stopped["requested_by"] == "watchdog"
    assert stopped["condition_clock_seconds"] <= \
        report["clocks"]["condition_clock_seconds"]
    assert stopped["process_clock_seconds"] <= \
        report["clocks"]["process_clock_seconds"]
    assert lr.replay_child(report)["problems"] == []


@pytest.mark.parametrize("field", STOP_FIELDS)
def test_r7_a_stopped_early_block_missing_a_field_is_refused(tmp_path, field):
    report, _ = stopped_child(tmp_path)
    del report["stopped_early"][field]
    assert any(field in p for p in lr.replay_child(report)["problems"])


@pytest.mark.parametrize("field,value", [
    ("requested_by", "somebody"),
    ("row", 1),
    ("row", 0),
    ("condition_clock_seconds", -1.0),
    ("process_clock_seconds", 10 ** 9),
])
def test_r7_a_stopped_early_block_that_does_not_add_up_is_refused(
        tmp_path, field, value):
    report, _ = stopped_child(tmp_path)
    report["stopped_early"][field] = value
    assert any(field in p for p in lr.replay_child(report)["problems"])


# --- 3. the supervision race ------------------------------------------------


def test_r7_a_child_that_finished_first_is_not_a_dead_watchdog():
    """Both gone means the run ended; which one the kernel reaped first is
    not evidence about anything."""
    reaped = []
    out = wd.supervise(child_alive=lambda: False, watchdog_alive=lambda: False,
                       poll_heartbeat=lambda: True, clock=FakeClock(),
                       spec=wd.SafetySpec(), on_stop=reaped.append,
                       sleep=lambda s: None)
    assert out == {"stopped": False, "reason": None, "is_tool_failure": False}
    assert reaped == [], "nothing to reap, and nothing to report"


def test_r7_a_watchdog_that_dies_under_a_running_child_is_still_a_failure():
    reaped = []
    out = wd.supervise(child_alive=lambda: True, watchdog_alive=lambda: False,
                       poll_heartbeat=lambda: True, clock=FakeClock(),
                       spec=wd.SafetySpec(), on_stop=reaped.append,
                       sleep=lambda s: None)
    assert out["reason"] == "watchdog_died" and out["is_tool_failure"]
    assert reaped == ["watchdog_died"]


def test_r7_a_finished_child_outranks_a_missing_heartbeat():
    clock = FakeClock(step=10 ** 6)
    out = wd.supervise(child_alive=lambda: False, watchdog_alive=lambda: True,
                       poll_heartbeat=lambda: False, clock=clock,
                       spec=wd.SafetySpec(), on_stop=lambda r: None,
                       sleep=lambda s: clock.advance())
    assert out["stopped"] is False


def test_r7_a_run_whose_child_finished_first_still_completes(tmp_path):
    """Through session_next: the run is collected, not filed as a failure."""
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(
        tmp_path,
        supervise_fn=lambda: wd.supervise(
            child_alive=lambda: False, watchdog_alive=lambda: False,
            poll_heartbeat=lambda: True, clock=FakeClock(),
            spec=wd.SafetySpec(), on_stop=lambda r: None,
            sleep=lambda s: None)))
    assert out["ok"], out.get("problems")
    assert out["outcome"] == "completed"
    st = lr.session_state("exp016a", root=tmp_path / "reports")
    assert st["completed"] == ["b1"] and not st["terminal"]


# ===========================================================================
# Round 8. The watchdog terminal protocol.
#
# The previous round moved the child check ahead of the watchdog check in
# supervise(), which is right, and left a hole that was written down but not
# closed: a watchdog that died in the first second, a child that ran for
# eight minutes and exited before the parent's next look, and a one-record
# log spanning zero seconds -- all of it filed as `completed`, because
# nothing required the watchdog to say how the run ended.
#
# So it has to say. The watchdog observes the child's exit and records it,
# once, last, before it seals. No terminal record, no completed run.
# ===========================================================================


def terminal_record(records):
    return [r for r in records if r.get("action") == wd.CHILD_EXIT_OBSERVED]


def run_a_watchdog(tmp_path, *, alive_for=2, prefix="b1"):
    """A real watchdog_loop over a child that exits after `alive_for` polls."""
    identity = ident(pid=4242, pgid=4242)
    log = wd.WatchdogLog(tmp_path / f"{prefix}.watchdog.jsonl")
    calls = {"n": 0}

    def alive():
        calls["n"] += 1
        return calls["n"] <= alive_for

    args = clean_loop_args(tmp_path, identity, max_polls=10,
                           child_alive=alive, prefix=prefix)
    args.pop("directory")
    args.pop("prefix")
    wd.watchdog_loop(log, identity, wd.SafetySpec(), **args,
                     directory=tmp_path, prefix=prefix)
    sha = log.seal()
    return wd.replay_watchdog_log(tmp_path / f"{prefix}.watchdog.jsonl",
                                  expected_sha256=sha), identity


def test_r8_the_watchdog_records_the_exit_it_observed(tmp_path):
    """The last thing a watchdog does is say how the run ended."""
    out, identity = run_a_watchdog(tmp_path)
    assert out["problems"] == []
    records = out["records"]
    last = records[-1]
    assert last["action"] == wd.CHILD_EXIT_OBSERVED
    assert len(terminal_record(records)) == 1
    assert last["nonce"] == identity.nonce
    assert last["child_pid"] == 4242 and last["child_pgid"] == 4242
    assert last["child_start_identity"] == "T0"
    assert lr.finite(last["monotonic"]) and "wall_clock" in last
    assert "progress" in last and "heartbeat_seq" in last


def test_r8_a_terminal_record_that_closes_the_log_replays_clean(tmp_path):
    out, _ = run_a_watchdog(tmp_path)
    assert wd.watchdog_terminal_problems(out["records"],
                                         wd.SafetySpec()) == []


def sealed_records(*, polls=3, terminal=True, gap=5.0):
    """Poll records followed by a proper terminal record."""
    records = wd_records(polls)
    if terminal:
        last = dict(records[-1])
        last.update({"seq": polls, "monotonic": records[-1]["monotonic"] + gap,
                     "action": wd.CHILD_EXIT_OBSERVED, "heartbeat_seq": polls,
                     "swap_used_gb": None,
                     "memory_pressure_percent_free": None,
                     "free_plus_inactive_gb": None, "failed": None})
        records.append(last)
    return records


def test_r8_a_properly_sealed_log_passes(tmp_path):
    assert wd.watchdog_terminal_problems(sealed_records(),
                                         wd.SafetySpec()) == []


def test_r8_a_log_with_no_terminal_record_is_refused():
    problems = wd.watchdog_terminal_problems(sealed_records(terminal=False),
                                             wd.SafetySpec())
    assert any("child_exit_observed" in p for p in problems), problems


def test_r8_the_codex_case_one_record_spanning_no_time():
    """One poll, zero seconds, and a child that ran for eight minutes."""
    problems = wd.watchdog_terminal_problems(wd_records(1), wd.SafetySpec())
    assert problems, "a single unsealed poll is not an account of a run"


def test_r8_two_terminal_records_are_refused():
    records = sealed_records()
    extra = dict(records[-1])
    extra.update({"seq": records[-1]["seq"] + 1,
                  "monotonic": records[-1]["monotonic"] + 1.0})
    records.append(extra)
    problems = wd.watchdog_terminal_problems(records, wd.SafetySpec())
    assert any("once" in p or "2" in p for p in problems), problems


def test_r8_a_terminal_record_in_the_middle_is_refused():
    records = sealed_records()
    tail = dict(records[0])
    tail.update({"seq": records[-1]["seq"] + 1,
                 "monotonic": records[-1]["monotonic"] + 5.0})
    records.append(tail)
    problems = wd.watchdog_terminal_problems(records, wd.SafetySpec())
    assert any("last" in p for p in problems), problems


def test_r8_a_terminal_record_that_arrives_too_late_is_refused():
    """The exit is observed on the next poll, not an hour afterwards."""
    late = sealed_records(gap=wd.SafetySpec().poll_max_gap_seconds + 1.0)
    problems = wd.watchdog_terminal_problems(late, wd.SafetySpec())
    assert any("outside" in p for p in problems), problems


def test_r8_a_terminal_record_on_its_own_is_refused():
    """Nothing was watched: the log opens and closes in the same breath."""
    records = sealed_records(polls=0) if False else sealed_records(polls=1)
    records = [records[-1]]
    records[0]["seq"] = 0
    problems = wd.watchdog_terminal_problems(records, wd.SafetySpec())
    assert problems, "a log whose only record is its own ending watched nothing"


@pytest.mark.parametrize("field", list(lr.IDENTITY_FIELDS))
def test_r8_a_terminal_record_without_its_identity_is_refused(field):
    records = sealed_records()
    del records[-1][field]
    problems = wd.watchdog_terminal_problems(records, wd.SafetySpec())
    assert any(field in p for p in problems), problems


# --- the session paths ------------------------------------------------------


def test_r8_a_clean_double_exit_completes(tmp_path):
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(tmp_path))
    assert out["ok"] and out["outcome"] == "completed", out.get("problems")
    assert verify(tmp_path) == []


def test_r8_a_watchdog_that_died_early_is_not_a_completed_run(tmp_path):
    """The hole round seven documented, closed.

    The watchdog dies in the first second; the child runs on and exits before
    the parent's next look, so supervision -- correctly -- reports nothing.
    The only thing left to notice is that the log never says how the run
    ended, and that now has to be enough.
    """
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(
        tmp_path, collect=collector(terminal=False),
        supervise_fn=lambda: {"stopped": False}))
    assert out["ok"] is False
    assert out["outcome"] == "no_report"
    assert out["tool_failure"]
    st = lr.session_state("exp016a", root=tmp_path / "reports")
    assert st["terminal"] == ["b1"] and st["completed"] == []


def test_r8_the_codex_session_does_not_verify(tmp_path):
    """500 rows, 503.2s of process clock, one watchdog record, no ending."""
    out = init(tmp_path)
    place_run(out["paths"], out["plan"], "b1", 500, boot="boot-0",
              watchdog_records=1, terminal=False)
    report = json.loads((out["paths"]["dir"] / "b1.json").read_text())
    assert report["rows_completed"] == 500
    assert report["clocks"]["process_clock_seconds"] > 500
    problems = verify(tmp_path)
    assert any(wd.CHILD_EXIT_OBSERVED in p for p in problems), problems


def test_r8_preconditions_refuse_a_badly_sealed_earlier_run(tmp_path):
    out = init(tmp_path)
    global THRESHOLDS
    THRESHOLDS = json.loads(
        out["paths"]["calibration"].read_text())["thresholds"]
    place_run(out["paths"], out["plan"], "b1", 500, boot="boot-0",
              terminal=False)
    before = len(lr.read_journal(out["paths"]["dir"]))
    gate = counting_gate()
    result = lr.session_next("exp016a", **next_args(
        tmp_path, gate=gate, boot={"boot_fingerprint": "boot-9"}))
    assert result["ok"] is False and result["boot_consumed"] is False
    assert result.get("preconditions_failed") is True
    assert gate.calls == []
    assert len(lr.read_journal(out["paths"]["dir"])) == before
    assert any(wd.CHILD_EXIT_OBSERVED in p for p in lr.session_preconditions(
        "exp016a", root=sess(tmp_path), check_working_tree=False))


def test_r8_qualify_outcome_refuses_before_the_finished_event(tmp_path):
    """The check runs where it can still change the outcome that is written."""
    out = init(tmp_path)
    place_run(out["paths"], out["plan"], "b1", 500, boot="boot-0",
              terminal=False)
    paths = out["paths"]
    body = json.loads(next(p for p in events_of(paths)
                           if f"b1-{lr.EVENT_FINISHED}" in p.name).read_text())
    got = lr.qualify_outcome(
        paths, "b1", claimed="completed", exit_status=0,
        report_sha256=body["report_sha256"],
        watchdog_sha256=body["watchdog_sha256"],
        spec=lr.plan_spec(out["plan"]))
    assert got["outcome"] == "no_report"
    assert got["tool_failure"]
    assert any(wd.CHILD_EXIT_OBSERVED in p for p in got["problems"])


def test_r8_a_sealed_log_still_qualifies(tmp_path):
    out = init(tmp_path)
    place_run(out["paths"], out["plan"], "b1", 500, boot="boot-0")
    paths = out["paths"]
    body = json.loads(next(p for p in events_of(paths)
                           if f"b1-{lr.EVENT_FINISHED}" in p.name).read_text())
    got = lr.qualify_outcome(
        paths, "b1", claimed="completed", exit_status=0,
        report_sha256=body["report_sha256"],
        watchdog_sha256=body["watchdog_sha256"],
        spec=lr.plan_spec(out["plan"]))
    assert got == {"outcome": "completed", "problems": []}


# ===========================================================================
# Round 9. The terminal record has to be checked, not merely counted.
#
# Round 8 required a terminal record to exist, be unique, be last and arrive
# within a poll interval. It never asked whose exit it described, and it
# measured the final interval with a one-sided comparison. So a terminal
# record naming a different PID passed qualify_outcome, and one dated before
# the poll it follows passed the gap check by being negative.
# ===========================================================================


def rewrite_watchdog_log(paths, run, mutate):
    """Rewrite a watchdog log, rebuild its chain, and re-record its digest.

    Every single-field edit trips the hash chain on its own, which would let
    these tests pass without the identity comparison existing. Re-signing the
    whole artefact is the only way to find out whether anything actually
    compares it with the run it is filed under.
    """
    fresh = reseal_watchdog_log(paths, run, mutate)
    rewrite_chained(paths, f"{run}-{lr.EVENT_FINISHED}",
                    lambda b: b.update({"watchdog_sha256": fresh}))
    return fresh


def reseal_watchdog_log(paths, run, mutate):
    """Rewrite a watchdog log and rebuild its hash chain. No journal.

    Used from inside a collector, where the finished event does not exist
    yet -- which is the whole point of the test: the tamper has to be caught
    before the outcome is written down, not after.
    """
    path = paths["dir"] / f"{run}.watchdog.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(records)
    lines, prev = [], None
    for i, rec in enumerate(records):
        rec["seq"] = i
        rec["prev_digest"] = prev
        line = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        lines.append(line)
        prev = hashlib.sha256(line.encode("utf-8")).hexdigest()
    path.write_text("\n".join(lines) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finished_body_of(paths, run):
    return json.loads(next(p for p in events_of(paths)
                           if f"{run}-{lr.EVENT_FINISHED}" in p.name).read_text())


def qualify(paths, run, plan):
    body = finished_body_of(paths, run)
    return lr.qualify_outcome(
        paths, run, claimed="completed", exit_status=0,
        report_sha256=body["report_sha256"],
        watchdog_sha256=body["watchdog_sha256"], spec=lr.plan_spec(plan))


def sealed_run(tmp_path):
    """One completed run, every artefact consistent."""
    global THRESHOLDS
    out = init(tmp_path)
    THRESHOLDS = json.loads(
        out["paths"]["calibration"].read_text())["thresholds"]
    place_run(out["paths"], out["plan"], "b1", 500, boot="boot-0")
    return out


def assert_refused_everywhere(tmp_path, out, needle):
    """The same evidence must be refused before *and* after the fact.

    qualify_outcome runs before the finished event is written, so it is the
    only one of the three that can still change what gets recorded; the other
    two are what catch a session that was written by an older tool.
    """
    got = qualify(out["paths"], "b1", out["plan"])
    assert got["outcome"] == "no_report", got
    assert got["tool_failure"] == "watchdog_unsealed"
    assert any(needle in p for p in got["problems"]), got["problems"]

    problems = verify(tmp_path)
    assert any(needle in p for p in problems), problems

    pre = lr.session_preconditions("exp016a", root=sess(tmp_path),
                                   check_working_tree=False)
    assert any(needle in p for p in pre), pre

    before = len(lr.read_journal(out["paths"]["dir"]))
    gate = counting_gate()
    result = lr.session_next("exp016a", **next_args(
        tmp_path, gate=gate, boot={"boot_fingerprint": "boot-9"}))
    assert result["ok"] is False and result["boot_consumed"] is False
    assert result.get("preconditions_failed") is True
    assert gate.calls == []
    assert len(lr.read_journal(out["paths"]["dir"])) == before


@pytest.mark.parametrize("field", list(lr.IDENTITY_FIELDS))
def test_r9_a_terminal_record_naming_another_process_is_refused(tmp_path, field):
    """The reported case: change the terminal record's pid, re-sign, and it
    used to sail through qualify_outcome."""
    out = sealed_run(tmp_path)
    other = {"nonce": "SOMEBODY-ELSE", "child_pid": 999999,
             "child_pgid": 999999, "child_start_identity": "another Tuesday"}

    def swap(records):
        records[-1][field] = other[field]

    rewrite_watchdog_log(out["paths"], "b1", swap)
    assert_refused_everywhere(tmp_path, out, field)


def test_r9_a_terminal_record_dated_before_the_poll_it_follows_is_refused(
        tmp_path):
    """delta < 0 passed a one-sided `delta > max_gap` comparison."""
    out = sealed_run(tmp_path)

    def backwards(records):
        records[-1]["monotonic"] = records[-2]["monotonic"] - 5.0

    rewrite_watchdog_log(out["paths"], "b1", backwards)
    assert_refused_everywhere(tmp_path, out, "monotonic")


def test_r9_the_terminal_gap_is_bounded_on_both_sides():
    spec = wd.SafetySpec()
    ok = sealed_records(gap=spec.poll_max_gap_seconds)
    assert wd.watchdog_terminal_problems(ok, spec) == []
    for bad_gap in (-0.001, -30.0, spec.poll_max_gap_seconds + 0.001):
        problems = wd.watchdog_terminal_problems(
            sealed_records(gap=bad_gap), spec)
        assert problems, f"gap {bad_gap} should not be accepted"


@pytest.mark.parametrize("action", ["sigkill", "identity_mismatch"])
def test_r9_a_terminal_record_after_an_abandonment_is_refused(action):
    """A watchdog that killed the child, or refused to signal at all, did not
    then stand around and watch it exit cleanly."""
    records = sealed_records()
    abandoned = dict(records[-2])
    abandoned["action"] = action
    records[-2] = abandoned
    problems = wd.watchdog_terminal_problems(records, wd.SafetySpec())
    assert any(action in p for p in problems), problems


@pytest.mark.parametrize("action", ["sigkill", "identity_mismatch"])
def test_r9_a_forged_seal_after_an_abandonment_is_refused_in_a_session(
        tmp_path, action):
    out = sealed_run(tmp_path)

    def abandon(records):
        records[-2]["action"] = action

    rewrite_watchdog_log(out["paths"], "b1", abandon)
    assert_refused_everywhere(tmp_path, out, action)


def test_r9_a_log_that_disagrees_with_the_launch_record_is_refused(tmp_path):
    """Report and log agree; the parent's own record does not."""
    out = sealed_run(tmp_path)
    launch = out["paths"]["dir"] / "b1.launch.json"
    body = json.loads(launch.read_text())
    body["child_pgid"] = 999999
    launch.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                 indent=1))
    fresh = hashlib.sha256(launch.read_bytes()).hexdigest()
    rewrite_chained(out["paths"], f"b1-{lr.EVENT_FINISHED}",
                    lambda b: b.update({"launch_sha256": fresh}))
    got = qualify(out["paths"], "b1", out["plan"])
    assert got["outcome"] == "no_report", got
    assert any("child_pgid" in p for p in got["problems"]), got["problems"]
    assert any("child_pgid" in p for p in verify(tmp_path))


def test_r9_qualify_outcome_reads_all_three_accounts(tmp_path):
    """A clean run still qualifies once all three are compared."""
    out = sealed_run(tmp_path)
    assert qualify(out["paths"], "b1", out["plan"]) == {"outcome": "completed",
                                                        "problems": []}
    assert verify(tmp_path) == []


def test_r9_a_bad_seal_produces_no_verdict_and_no_r1(tmp_path):
    """Terminal incomplete, and nothing downstream treats it as a measurement."""
    init(tmp_path)

    def bad_collect(*, run, paths):
        result = fake_collect(run, paths)
        fresh = reseal_watchdog_log(
            paths, run, lambda rs: rs[-1].__setitem__("child_pid", 424242))
        return {**result, "watchdog_sha256": fresh}

    out = lr.session_next("exp016a", **next_args(tmp_path, collect=bad_collect))
    assert out["ok"] is False
    assert out["outcome"] == "no_report"
    assert out["tool_failure"] == "watchdog_unsealed"
    assert out["r1"] is None, "a tool failure is not a measurement"
    st = lr.session_state("exp016a", root=tmp_path / "reports")
    assert st["terminal"] == ["b1"] and st["completed"] == []
    assert st["cancelled"] == []
    assert not any(e["event"] == lr.EVENT_PLAN_ARM_CANCELLED
                   for e in st["events"])
    agg = lr.build_aggregate(st)
    assert agg["headline"]["allowed"] is False
    assert [r["Q1_run"] for r in agg["runs"]] == [None]


# ===========================================================================
# Round 10. The report itself was never checked before the outcome was.
#
# Rounds 8 and 9 built a real gate in front of `completed`, and pointed all
# of it at the watchdog log. The child report -- the document every verdict
# is computed from -- was checked only for its digest. So a report whose
# per-row array summed to one thing and whose model_compute_seconds said
# 999999 was recorded as completed, and an invalid stopped_early claiming a
# safety stop did not merely pass: it fired R1 and cancelled b2 and b3.
#
# A verdict is not something to be caught afterwards by --verify. By then
# the boot is spent, the journal says completed, and two arms are cancelled.
# ===========================================================================


def tampering_collector(mutate, **kw):
    """A collector whose evidence is perfect until it re-signs a bad report.

    Everything the round 8/9 gate looks at -- terminal record, identities,
    digests -- is left correct, so these tests fail for exactly one reason:
    nobody replayed the report before writing down what it was.
    """
    def collect(*, run, paths):
        result = fake_collect(run, paths, **kw)
        rp = paths["dir"] / f"{run}.json"
        report = json.loads(rp.read_text())
        mutate(report)
        rp.write_text(json.dumps(report))
        return {**result,
                "report_sha256": hashlib.sha256(rp.read_bytes()).hexdigest()}
    return collect


def assert_not_a_measurement(tmp_path, out, *, needle=None):
    """A run that failed pre-finish validation leaves no trace of a verdict."""
    assert out["ok"] is False, out
    assert out["outcome"] != "completed", out
    assert out["outcome"] == "no_report"
    assert out["tool_failure"], "a rejected run has to say what went wrong"
    assert out["r1"] is None, "R1 does not run on something that is not a run"
    if needle:
        assert any(needle in p for p in out["problems"]), out["problems"]
    st = lr.session_state("exp016a", root=sess(tmp_path))
    assert st["terminal"] == ["b1"] and st["completed"] == []
    assert st["cancelled"] == []
    assert not any(e["event"] == lr.EVENT_PLAN_ARM_CANCELLED
                   for e in st["events"])
    agg = lr.build_aggregate(st)
    assert [r["Q1_run"] for r in agg["runs"]] == [None]
    assert [r["Q2_run"] for r in agg["runs"]] == [None]
    assert agg["headline"]["allowed"] is False
    return st


def test_r10_a_resigned_compute_total_is_not_a_completed_run(tmp_path):
    """The reported case: 999999 seconds of compute, correctly signed."""
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(
        tmp_path,
        collect=tampering_collector(
            lambda r: r.__setitem__("model_compute_seconds", 999999.0))))
    assert_not_a_measurement(tmp_path, out, needle="model_compute_seconds")


def test_r10_stored_metrics_that_disagree_with_per_row_are_refused(tmp_path):
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(
        tmp_path,
        collect=tampering_collector(
            lambda r: r["metrics"].__setitem__("D100", 9.99))))
    assert_not_a_measurement(tmp_path, out, needle="D100")


INVALID_STOP = {"reason": "swap_used_gb"}


def test_r10_an_invalid_stopped_early_never_reaches_r1(tmp_path):
    """The reported case: a one-field stopped_early claiming a safety trip.

    It used to be recorded as completed *and* to fire R1, cancelling b2 and
    b3 on the strength of a block that does not even carry the row it
    stopped at.
    """
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(
        tmp_path,
        collect=tampering_collector(
            lambda r: r.__setitem__("stopped_early", dict(INVALID_STOP)))))
    st = assert_not_a_measurement(tmp_path, out, needle="stopped_early")
    assert st["remaining"] == ["b2", "b3"], "nothing was cancelled by rule"


@pytest.mark.parametrize("how", ["missing", "digest", "nonce"])
def test_r10_a_safety_stop_without_an_authentic_request_is_refused(tmp_path, how):
    """A report may not claim a stop the stop request does not support."""
    init(tmp_path)

    def collect(*, run, paths):
        started = _started_body(paths, run)
        nonce = started.get("nonce", "n")
        if how != "missing":
            wd.write_stop_request(
                paths["dir"], prefix=run, reason="swap_used_gb",
                rule="swap_used_gb",
                nonce="SOMEBODY-ELSE" if how == "nonce" else nonce,
                monotonic=1.0, wall_clock="Z", sampled_values=sample(swap=99.0))
        result = fake_collect(run, paths)
        rp = paths["dir"] / f"{run}.json"
        report = json.loads(rp.read_text())
        report["stopped_early"] = stop_block(row=report["rows_completed"])
        rp.write_text(json.dumps(report))
        return {**result,
                "report_sha256": hashlib.sha256(rp.read_bytes()).hexdigest()}

    out = lr.session_next("exp016a", **next_args(tmp_path, collect=collect))
    assert_not_a_measurement(tmp_path, out)


@pytest.mark.parametrize("field,value", [
    ("declared_rows", 1234),
    ("condition", "continuous"),
    ("run_id", "b3"),
    ("plan_digest", "d" * 64),
    ("experiment_id", "some-other-experiment"),
    ("nonce", "SOMEBODY-ELSES-LAUNCH"),
])
def test_r10_a_report_that_disagrees_with_the_plan_is_refused(tmp_path, field,
                                                              value):
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(
        tmp_path,
        collect=tampering_collector(lambda r: r.__setitem__(field, value))))
    assert_not_a_measurement(tmp_path, out, needle=field)


@pytest.mark.parametrize("key,value", [
    ("plan_digest", "0" * 64),
    ("source_manifest_digest", "0" * 64),
    ("files_verified", 99),
])
def test_r10_a_source_check_that_disagrees_is_refused_before_finish(
        tmp_path, key, value):
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(
        tmp_path,
        collect=tampering_collector(
            lambda r: r["child_source_check"].__setitem__(key, value))))
    assert_not_a_measurement(tmp_path, out, needle="child_source_check")


# --- R1 refuses evidence it cannot replay -----------------------------------


def test_r10_r1_refuses_a_completed_run_that_does_not_replay(tmp_path):
    """A session written by an older tool, judged by this one.

    place_run writes `completed` straight into the journal, the way the tool
    did before the pre-finish gate existed. R1 has to refuse it on its own
    account: a rule that trusts whatever the journal says is a rule that can
    be handed a forged premise.
    """
    out = stopped_session(tmp_path)
    paths = out["paths"]
    report = json.loads((paths["dir"] / "b1.json").read_text())
    assert report["stopped_early"]["reason"] == "swap_used_gb"
    report["stopped_early"] = dict(INVALID_STOP)
    resign_report(paths, "b1", report)

    fired = lr.apply_rule_r1("exp016a", root=sess(tmp_path))
    assert fired["fired"] is False, fired
    assert fired.get("evidence_failure") is True, fired
    st = lr.session_state("exp016a", root=sess(tmp_path))
    assert st["cancelled"] == []
    assert not any(e["event"] == lr.EVENT_PLAN_ARM_CANCELLED
                   for e in st["events"])
    assert st["remaining"] == ["b2", "b3"]


def test_r10_r1_still_fires_on_evidence_that_does_replay(tmp_path):
    """The guard refuses bad evidence, not every cancellation."""
    stopped_session(tmp_path)
    fired = lr.apply_rule_r1("exp016a", root=sess(tmp_path))
    assert fired["fired"] is True, fired
    assert fired["cancelled"] == ["b2", "b3"]


def test_r10_the_aggregate_nulls_the_verdicts_of_an_unreplayable_run(tmp_path):
    out = stopped_session(tmp_path)
    report = json.loads((out["paths"]["dir"] / "b1.json").read_text())
    report["model_compute_seconds"] = 999999.0
    resign_report(out["paths"], "b1", report)
    st = lr.session_state("exp016a", root=sess(tmp_path))
    agg = lr.build_aggregate(st)
    assert [r["Q1_run"] for r in agg["runs"]] == [None]
    assert [r["Q2_run"] for r in agg["runs"]] == [None]
    assert agg["headline"]["allowed"] is False
    assert any("model_compute_seconds" in p for p in agg["replay_problems"])


def test_r10_one_validator_backs_the_gate_and_both_replays(tmp_path):
    """The pre-finish gate and the post-hoc replays share a definition.

    Two definitions of `completed` is how a run passes one and fails the
    other, which is the state this whole round was called about.
    """
    import inspect
    for fn in (lr.qualify_outcome, lr.session_preconditions,
               lr.verify_experiment, lr._apply_rule_r1_locked,
               lr.build_aggregate):
        assert "completed_run_evidence" in inspect.getsource(fn), fn.__name__


# ===========================================================================
# Round 11. Two doors were still open around the one definition of completed.
#
# Round 10 built `completed_run_evidence()` and pointed the pre-finish gate,
# the precondition replay and `--verify` at it. Two callers were left with
# weaker definitions of their own:
#
#   1. The finished event's own eligibility -- outcome, tool_failure, a real
#      integer-zero exit, and digests that are present rather than merely
#      consistent-when-present -- was checked inside `qualify_outcome` and
#      nowhere else. Rule R1 calls the shared validator directly, so a
#      finished event saying `exit_status: 1` replayed clean and still
#      cancelled b2 and b3.
#
#   2. `build_aggregate()` replayed the child report and nothing else. No
#      digest against the finished event, no launch identity, no stop
#      request. So three reports rewritten to name another experiment
#      produced Q1 "holds" three times and a headline that said adoption was
#      supported, while `--verify` on the same session named nine problems.
#
# A verdict computed from evidence nobody checked is not a weaker verdict.
# It is the same failure as no verdict at all, wearing the word "holds".
# ===========================================================================


def rewrite_finished(paths, run, mutate):
    """Rewrite a finished event and repair the chain behind it.

    Without the repair every tamper here looks like a broken hash chain, and
    a test that only ever sees that proves nothing about the eligibility
    check it was written for.
    """
    rewrite_chained(paths, f"{run}-{lr.EVENT_FINISHED}", mutate)


def assert_r1_did_not_fire(tmp_path, *, needle=None):
    """R1 computed nothing, wrote nothing and cancelled nothing."""
    fired = lr.apply_rule_r1("exp016a", root=sess(tmp_path))
    assert fired["fired"] is False, fired
    assert fired.get("evidence_failure") is True, fired
    assert "verdict" not in fired, "no Q1 may be computed from bad evidence"
    if needle:
        assert any(needle in p for p in fired.get("problems") or []), fired
    st = lr.session_state("exp016a", root=sess(tmp_path))
    assert st["cancelled"] == []
    assert st["remaining"] == ["b2", "b3"]
    assert not any(e["event"] == lr.EVENT_PLAN_ARM_CANCELLED
                   for e in st["events"])
    return fired


def aggregate_of(tmp_path):
    return lr.build_aggregate(lr.session_state("exp016a", root=sess(tmp_path)))


def assert_no_verdicts(agg, *, needle=None):
    assert all(r["Q1_run"] is None for r in agg["runs"]), agg["runs"]
    assert all(r["Q2_run"] is None for r in agg["runs"]), agg["runs"]
    assert agg["headline"]["allowed"] is False, agg["headline"]
    assert agg["replay_problems"], "a refusal has to say what it found"
    if needle:
        assert any(needle in p for p in agg["replay_problems"]), \
            agg["replay_problems"]


# --- 1. the finished event's own eligibility --------------------------------


@pytest.mark.parametrize("status", [1, False, "0", None, -1])
def test_r11_a_completed_event_with_a_bad_exit_status_is_refused(tmp_path,
                                                                 status):
    """Codex's case: exit_status 1, chain repaired, R1 fired anyway."""
    out = stopped_session(tmp_path)
    rewrite_finished(out["paths"], "b1",
                     lambda b: b.update({"exit_status": status}))
    assert any("exit status" in p for p in verify(tmp_path)), verify(tmp_path)
    assert_r1_did_not_fire(tmp_path, needle="exit status")
    assert_no_verdicts(aggregate_of(tmp_path), needle="exit status")


def test_r11_a_zero_exit_is_still_a_clean_exit(tmp_path):
    """The guard refuses a forged status, not the real one."""
    stopped_session(tmp_path)
    assert verify(tmp_path) == []
    assert lr.apply_rule_r1("exp016a", root=sess(tmp_path))["fired"] is True


@pytest.mark.parametrize("field", ["report_sha256", "watchdog_sha256"])
def test_r11_a_completed_event_missing_a_digest_never_reaches_r1(tmp_path,
                                                                 field):
    """A digest that is merely absent used to be indistinguishable from one
    that still matched: the comparison was guarded by the value's truthiness,
    so the missing case fell through to `no problems`."""
    out = stopped_session(tmp_path)
    rewrite_finished(out["paths"], "b1", lambda b: b.update({field: None}))
    assert verify(tmp_path)
    assert_r1_did_not_fire(tmp_path, needle="digest was recorded")
    assert_no_verdicts(aggregate_of(tmp_path), needle="digest was recorded")


@pytest.mark.parametrize("body_patch,needle", [
    ({"outcome": "no_report"}, "outcome"),
    ({"tool_failure": "watchdog_died"}, "tool failure"),
])
def test_r11_a_finished_event_that_is_not_eligible_gets_no_verdict(
        tmp_path, body_patch, needle):
    """The shared validator judges the event, not just the artefacts."""
    out = stopped_session(tmp_path)
    paths = out["paths"]
    body = finished_body_of(paths, "b1")
    body.update(body_patch)
    problems = lr.completed_run_evidence(
        paths, "b1", spec=lr.plan_spec(out["plan"]),
        plan=out["plan"], finished_body=body)["problems"]
    assert any(needle in p for p in problems), problems


# --- 2. the aggregate has to use the same evidence --------------------------


def test_r11_finalize_refuses_reports_that_no_longer_hash(tmp_path):
    """Codex's case: three reports renamed to another experiment at once.

    Every cross-run check still agreed -- they were all wrong in the same
    way -- and the finished events' digests were left alone. `--verify` named
    nine problems; `session_finalize()` produced Q1 "holds" three times.
    """
    out = finished_session(tmp_path)
    for run in lr.RUN_IDS:
        rp = out["paths"]["dir"] / f"{run}.json"
        rep = json.loads(rp.read_text())
        rep["experiment_id"] = "some-other-experiment"
        rp.write_text(json.dumps(rep))

    assert verify(tmp_path), "the same session must not verify"
    res = lr.session_finalize("exp016a", root=sess(tmp_path))
    agg = res["aggregate"]
    assert_no_verdicts(agg, needle="digest")
    assert agg["Q1_plan"]["value"] != "holds_to_2000"
    assert any("experiment_id" in p for p in agg["replay_problems"]), \
        agg["replay_problems"]


def test_r11_the_aggregate_refuses_a_stop_nothing_asked_for(tmp_path):
    """A real safety stop whose stop request has been deleted."""
    out = stopped_session(tmp_path)
    (out["paths"]["dir"] / "b1.stop_request.json").unlink()
    assert_no_verdicts(aggregate_of(tmp_path), needle="stop_request")


@pytest.mark.parametrize("how,needle", [
    ("launch_identity", "launch"),
    ("plan_identity", "condition"),
    ("report_digest", "digest"),
    ("source_check", "child_source_check"),
])
def test_r11_the_aggregate_refuses_evidence_that_misidentifies_its_run(
        tmp_path, how, needle):
    out = stopped_session(tmp_path)
    paths = out["paths"]
    if how == "launch_identity":
        resign_launch(paths, "b1", lambda b: b.update({"child_pid": 424242}))
    elif how == "plan_identity":
        rep = json.loads((paths["dir"] / "b1.json").read_text())
        rep["condition"] = "continuous"
        resign_report(paths, "b1", rep)
    elif how == "source_check":
        rep = json.loads((paths["dir"] / "b1.json").read_text())
        rep["child_source_check"]["files_verified"] = 99
        resign_report(paths, "b1", rep)
    else:
        rp = paths["dir"] / "b1.json"
        rep = json.loads(rp.read_text())
        rep["rows_completed"] = 499
        rp.write_text(json.dumps(rep))  # digest deliberately left stale
    assert_no_verdicts(aggregate_of(tmp_path), needle=needle)


# --- 3. the clean paths still work ------------------------------------------


def test_r11_a_clean_session_still_finalises(tmp_path):
    finished_session(tmp_path)
    assert verify(tmp_path) == []
    res = lr.session_finalize("exp016a", root=sess(tmp_path))
    assert res["problems"] == []
    agg = res["aggregate"]
    assert agg["replay_problems"] == [], agg["replay_problems"]
    assert agg["headline"]["allowed"] is True
    assert [r["Q1_run"] for r in agg["runs"]] == ["holds"] * 3


def test_r11_a_legitimate_safety_stop_still_fires_r1_and_finalises(tmp_path):
    stopped_session(tmp_path)
    fired = lr.apply_rule_r1("exp016a", root=sess(tmp_path))
    assert fired["fired"] is True and fired["cancelled"] == ["b2", "b3"]
    res = lr.session_finalize("exp016a", root=sess(tmp_path))
    assert res["problems"] == []
    assert res["aggregate"]["replay_problems"] == []
    assert res["aggregate"]["state"] == "complete_by_rule"
    assert verify(tmp_path) == []


def test_r11_a_terminal_incomplete_run_keeps_its_own_semantics(tmp_path):
    """A run that never sealed its log is terminal, not an evidence failure.

    Demanding a terminal record of a run that was SIGKILLed would turn one
    cause into two failures, so the non-completed path is deliberately not
    the completed one.
    """
    out = init(tmp_path)
    global THRESHOLDS
    THRESHOLDS = json.loads(
        out["paths"]["calibration"].read_text())["thresholds"]
    place_run(out["paths"], out["plan"], "b1", 500, boot="boot-0",
              outcome="no_report", terminal=False)
    rewrite_finished(out["paths"], "b1",
                     lambda b: b.update({"tool_failure": "child_spawn_failed"}))
    st = lr.session_state("exp016a", root=sess(tmp_path))
    assert st["terminal"] == ["b1"] and st["completed"] == []
    agg = lr.build_aggregate(st)
    assert agg["state"] == "terminal_incomplete"
    assert [r["Q1_run"] for r in agg["runs"]] == [None]
    assert agg["headline"]["allowed"] is False
    # The completed path demands a sealed log; the terminal path must not,
    # or one cause becomes two failures.
    assert not any(wd.CHILD_EXIT_OBSERVED in p
                   for p in agg["replay_problems"]), agg["replay_problems"]


# --- 4. one definition, proved by behaviour ---------------------------------


def test_r11_every_caller_shares_one_completed_definition(tmp_path):
    """The same tampered run, refused by all five callers.

    Not `inspect.getsource`: a name appearing in a function body proves the
    name is mentioned, not that the check runs. This drives each caller and
    reads what it decided.
    """
    out = finished_session(tmp_path)
    paths, plan = out["paths"], out["plan"]
    report = json.loads((paths["dir"] / "b3.json").read_text())
    report["model_compute_seconds"] = 999999.0
    resign_report(paths, "b3", report)
    needle = "model_compute_seconds"

    # 1. the pre-finish gate
    got = qualify(paths, "b3", plan)
    assert got["outcome"] == "no_report", got
    assert any(needle in p for p in got["problems"]), got["problems"]

    # 2. the precondition replay
    pre = lr.session_preconditions("exp016a", root=sess(tmp_path),
                                   check_working_tree=False)
    assert any(needle in p for p in pre), pre

    # 3. --verify
    assert any(needle in p for p in verify(tmp_path)), verify(tmp_path)

    # 4. rule R1
    fired = lr.apply_rule_r1("exp016a", root=sess(tmp_path))
    assert fired["fired"] is False and fired.get("evidence_failure") is True
    assert any(needle in p for p in fired["problems"]), fired

    # 5. the aggregate, and the finalise that writes it
    agg = lr.session_finalize("exp016a", root=sess(tmp_path))["aggregate"]
    b3 = next(r for r in agg["runs"] if r["run_id"] == "b3")
    assert b3["Q1_run"] is None and b3["Q2_run"] is None
    assert agg["headline"]["allowed"] is False
    assert any(needle in p for p in agg["replay_problems"]), agg


def test_r11_the_finished_event_is_judged_by_every_caller(tmp_path):
    """The eligibility half of the definition, driven through each caller."""
    out = finished_session(tmp_path)
    rewrite_finished(out["paths"], "b3",
                     lambda b: b.update({"exit_status": "0"}))
    needle = "exit status"

    pre = lr.session_preconditions("exp016a", root=sess(tmp_path),
                                   check_working_tree=False)
    assert any(needle in p for p in pre), pre
    assert any(needle in p for p in verify(tmp_path)), verify(tmp_path)
    fired = lr.apply_rule_r1("exp016a", root=sess(tmp_path))
    assert fired["fired"] is False and fired.get("evidence_failure") is True
    assert any(needle in p for p in fired["problems"]), fired
    agg = lr.session_finalize("exp016a", root=sess(tmp_path))["aggregate"]
    b3 = next(r for r in agg["runs"] if r["run_id"] == "b3")
    assert b3["Q1_run"] is None and b3["Q2_run"] is None
    assert agg["headline"]["allowed"] is False
    assert agg["Q1_plan"]["value"] != "holds_to_2000"
    assert any(needle in p for p in agg["replay_problems"]), agg


# ===========================================================================
# Round 12. A stop is described four times; only some of them were compared.
#
# A safety stop leaves four independent traces: the child's `stopped_early`
# block, the finished event's `stop_request_sha256`, the stop request file
# on disk, and the watchdog log that asked for it. The validator compared
# them only when both sides happened to be filled in -- `if want`, `if
# claimed` -- and never looked for the traces at all when the report said
# nothing had happened. Three shapes came out of that:
#
#   1. finished event's `stop_request_sha256` set to null, chain repaired:
#      verify clean, R1 still cancelled b2 and b3.
#   2. report's `stopped_early.stop_request_sha256` set to "": verify clean,
#      R1 still fired, the aggregate found nothing to say.
#   3. report's `stopped_early` set to null while the authenticated request,
#      the finished event's digest, the threshold trip and the SIGTERM all
#      stayed: verify clean. On a run whose per-row times hold, that turns a
#      stopped run into a passing one and R1 declines to cancel anything.
#
# The third is the one that matters most: the first two lose a digest, the
# third loses the stop.
# ===========================================================================


def holding_stopped_session(tmp_path):
    """b1 trips a real threshold while its per-row times never degrade.

    `stopped_session` degrades its tail, so Q1 fails either way and hiding
    the stop changes only the reason. Here Q1 is `holds` the moment the
    stopped_early block goes away, which is what makes the hidden stop a
    verdict rather than a missing sentence.
    """
    global THRESHOLDS
    out = init(tmp_path)
    THRESHOLDS = json.loads(
        out["paths"]["calibration"].read_text())["thresholds"]
    place_run(out["paths"], out["plan"], "b1", 500, boot="boot-0",
              stop=stop_block(row=500), safety_trip=True)
    return out


def plant_stop_request(paths, run, nonce, *, reason="swap_used_gb"):
    """Write a stop request that genuinely authenticates against this launch."""
    return wd.write_stop_request(
        paths["dir"], prefix=run, reason=reason, rule=reason, nonce=nonce,
        monotonic=5.0, wall_clock="Z",
        sampled_values=sample(swap=99.0))["sha256"]


def set_report_stop_digest(paths, run, value):
    report = json.loads((paths["dir"] / f"{run}.json").read_text())
    report["stopped_early"]["stop_request_sha256"] = value
    resign_report(paths, run, report)


#: Everything that is not the digest on disk. ``None``/``""``/``False`` are
#: the shapes the old `if want` / `if claimed` guards read as "nothing to
#: compare"; the last two are well-formed lies.
BAD_DIGESTS = [None, "", False, "not-a-digest", "a" * 64]


# --- 1. the finished event's account of the stop request --------------------


@pytest.mark.parametrize("value", BAD_DIGESTS)
def test_r12_a_finished_stop_digest_that_is_not_the_one_on_disk_is_refused(
        tmp_path, value):
    out = stopped_session(tmp_path)
    rewrite_finished(out["paths"], "b1",
                     lambda b: b.update({"stop_request_sha256": value}))
    assert any("stop request" in p for p in verify(tmp_path)), verify(tmp_path)
    assert_r1_did_not_fire(tmp_path, needle="stop request")
    assert_no_verdicts(aggregate_of(tmp_path), needle="stop request")


# --- 2. the child report's account of the stop request ----------------------


@pytest.mark.parametrize("value", BAD_DIGESTS)
def test_r12_a_report_stop_digest_that_is_not_the_one_on_disk_is_refused(
        tmp_path, value):
    out = stopped_session(tmp_path)
    set_report_stop_digest(out["paths"], "b1", value)
    assert any("stop request" in p for p in verify(tmp_path)), verify(tmp_path)
    assert_r1_did_not_fire(tmp_path, needle="stop request")
    assert_no_verdicts(aggregate_of(tmp_path), needle="stop request")


# --- 3. all three shapes of two-agree-one-differs ---------------------------


@pytest.mark.parametrize("shape", ["finished_differs", "report_differs",
                                   "disk_differs", "all_three_differ"])
def test_r12_the_three_accounts_of_one_stop_must_all_agree(tmp_path, shape):
    """finished event, child report and the file itself: two out of three is
    not agreement, whichever two they are."""
    out = stopped_session(tmp_path)
    paths = out["paths"]
    if shape == "finished_differs":
        rewrite_finished(paths, "b1",
                         lambda b: b.update({"stop_request_sha256": "a" * 64}))
    elif shape == "report_differs":
        set_report_stop_digest(paths, "b1", "a" * 64)
    elif shape == "disk_differs":
        # The two written accounts are rewritten together to agree with each
        # other; only the file they name still says otherwise.
        set_report_stop_digest(paths, "b1", "a" * 64)
        rewrite_finished(paths, "b1",
                         lambda b: b.update({"stop_request_sha256": "a" * 64}))
    else:
        set_report_stop_digest(paths, "b1", "a" * 64)
        rewrite_finished(paths, "b1",
                         lambda b: b.update({"stop_request_sha256": "b" * 64}))
    assert any("stop request" in p for p in verify(tmp_path)), verify(tmp_path)
    assert_r1_did_not_fire(tmp_path, needle="stop request")
    assert_no_verdicts(aggregate_of(tmp_path), needle="stop request")


# --- 4. the stop that was hidden by deleting the claim ----------------------


def test_r12_a_stop_may_not_be_hidden_by_dropping_stopped_early(tmp_path):
    """Codex's third case, on a run whose per-row times hold.

    Everything that asked for the stop is left in place: the authenticated
    request, the finished event's digest, the threshold trip and the SIGTERM.
    Only the child's own admission is removed -- and with it, the safety
    reason that turns Q1 into `fails`.
    """
    out = holding_stopped_session(tmp_path)
    paths = out["paths"]
    report = json.loads((paths["dir"] / "b1.json").read_text())
    assert report["stopped_early"]["reason"] == "swap_used_gb"
    report["stopped_early"] = None
    resign_report(paths, "b1", report)

    assert verify(tmp_path), "a stop that happened must not replay as clean"
    fired = lr.apply_rule_r1("exp016a", root=sess(tmp_path))
    assert fired["fired"] is False, fired
    assert fired.get("evidence_failure") is True, (
        "R1 must refuse the evidence, not merely decline to cancel")
    assert_no_verdicts(aggregate_of(tmp_path), needle="stopped_early")


def test_r12_the_hidden_stop_would_otherwise_have_passed(tmp_path):
    """The premise of the test above: this run's per-row times do hold.

    Without it, `Q1_run is None` would prove nothing -- a degrading run has
    no verdict either way.
    """
    holding_stopped_session(tmp_path)
    agg = aggregate_of(tmp_path)
    assert agg["runs"][0]["Q1_run"] == "fails"
    assert agg["runs"][0]["Q1_reason"].startswith("safety_stop:")
    assert agg["runs"][0]["metrics"]["D100"] == 1.0


# --- 5. a claimed stop the watchdog never made ------------------------------


def test_r12_a_claimed_stop_the_watchdog_never_made_is_refused(tmp_path):
    """Every digest agrees, the request authenticates -- and no threshold
    ever tripped, so nothing ever asked for this stop."""
    out = init(tmp_path)
    global THRESHOLDS
    THRESHOLDS = json.loads(
        out["paths"]["calibration"].read_text())["thresholds"]
    info = place_run(out["paths"], out["plan"], "b1", 500, boot="boot-0",
                     stop=stop_block(row=500), safety_trip=False)
    paths = out["paths"]
    digest = plant_stop_request(paths, "b1", info["nonce"])
    set_report_stop_digest(paths, "b1", digest)
    rewrite_finished(paths, "b1",
                     lambda b: b.update({"stop_request_sha256": digest}))
    assert verify(tmp_path)
    assert_no_verdicts(aggregate_of(tmp_path), needle="sigterm")


# --- 6. a run that records no stop may not carry the traces of one ----------


@pytest.mark.parametrize("how", ["finished_digest", "authenticated_request"])
def test_r12_a_run_that_records_no_stop_may_not_carry_stop_evidence(tmp_path,
                                                                    how):
    out = finished_session(tmp_path)
    paths = out["paths"]
    if how == "finished_digest":
        rewrite_finished(paths, "b1",
                         lambda b: b.update({"stop_request_sha256": "a" * 64}))
    else:
        plant_stop_request(paths, "b1", _started_body(paths, "b1")["nonce"])
    assert verify(tmp_path), "stop evidence under a run that never stopped"
    agg = aggregate_of(tmp_path)
    b1 = next(r for r in agg["runs"] if r["run_id"] == "b1")
    assert b1["Q1_run"] is None and b1["Q2_run"] is None
    assert agg["headline"]["allowed"] is False
    assert any("stop" in p for p in agg["replay_problems"]), \
        agg["replay_problems"]


# --- 7. the watchdog replay checks the reverse direction --------------------


def test_r12_semantics_refuse_a_stop_the_report_does_not_admit():
    recs, _ = semantic_records([sample(), sample(swap=99.0)])
    out = wd.replay_watchdog_semantics(recs, wd.SafetySpec(),
                                       claimed_stop=False)
    assert any("stopped_early" in p for p in out["problems"]), out["problems"]


def test_r12_semantics_still_accept_a_stop_the_report_admits():
    recs, _ = semantic_records([sample(), sample(swap=99.0)])
    out = wd.replay_watchdog_semantics(recs, wd.SafetySpec(),
                                       claimed_reason="swap_used_gb",
                                       claimed_stop=True)
    assert out["problems"] == [], out["problems"]


def test_r12_semantics_accept_a_quiet_log_from_a_run_that_did_not_stop():
    recs, _ = semantic_records([sample(), sample()])
    out = wd.replay_watchdog_semantics(recs, wd.SafetySpec(),
                                       claimed_stop=False)
    assert out["problems"] == [], out["problems"]


def test_r12_replay_child_states_the_claim_on_the_reports_behalf(tmp_path):
    """The production caller may not leave the claim unstated.

    Driven through `replay_child` itself rather than asserting on its source:
    the same watchdog log, replayed against a report that admits the stop and
    against one that does not.
    """
    out = holding_stopped_session(tmp_path)
    paths, spec = out["paths"], lr.plan_spec(out["plan"])
    wpath = paths["dir"] / "b1.watchdog.jsonl"
    report = json.loads((paths["dir"] / "b1.json").read_text())
    honest = lr.replay_child(report, watchdog_path=wpath, spec=spec,
                             launch_nonce=report["nonce"],
                             require_terminal=True)
    assert honest["problems"] == [], honest["problems"]

    report["stopped_early"] = None
    hidden = lr.replay_child(report, watchdog_path=wpath, spec=spec,
                             launch_nonce=report["nonce"],
                             require_terminal=True)
    assert any("stopped_early" in p for p in hidden["problems"]), hidden


# --- 8. the legitimate paths still work -------------------------------------


def test_r12_a_legitimate_safety_stop_still_verifies_fires_r1_and_finalises(
        tmp_path):
    stopped_session(tmp_path)
    assert verify(tmp_path) == []
    fired = lr.apply_rule_r1("exp016a", root=sess(tmp_path))
    assert fired["fired"] is True and fired["cancelled"] == ["b2", "b3"]
    res = lr.session_finalize("exp016a", root=sess(tmp_path))
    assert res["problems"] == []
    assert res["aggregate"]["replay_problems"] == []
    assert verify(tmp_path) == []


def test_r12_a_completed_run_that_never_stopped_still_verifies_and_finalises(
        tmp_path):
    finished_session(tmp_path)
    assert verify(tmp_path) == []
    res = lr.session_finalize("exp016a", root=sess(tmp_path))
    assert res["problems"] == []
    assert res["aggregate"]["replay_problems"] == []
    assert res["aggregate"]["headline"]["allowed"] is True
    assert [r["Q1_run"] for r in res["aggregate"]["runs"]] == ["holds"] * 3


# --- 9. one definition, driven through every caller -------------------------


def test_r12_every_caller_refuses_the_same_broken_stop_evidence(tmp_path):
    """The blank digest from Codex's second case, put to all five callers."""
    out = stopped_session(tmp_path)
    paths, plan = out["paths"], out["plan"]
    set_report_stop_digest(paths, "b1", "")
    needle = "stop request"

    got = qualify(paths, "b1", plan)
    assert got["outcome"] == "no_report", got
    assert any(needle in p for p in got["problems"]), got["problems"]

    pre = lr.session_preconditions("exp016a", root=sess(tmp_path),
                                   check_working_tree=False)
    assert any(needle in p for p in pre), pre
    assert any(needle in p for p in verify(tmp_path)), verify(tmp_path)
    assert_r1_did_not_fire(tmp_path, needle=needle)
    assert_no_verdicts(aggregate_of(tmp_path), needle=needle)


def test_r12_finalize_refuses_a_cancelled_session_whose_stop_evidence_broke(
        tmp_path):
    """R1 fires on good evidence; the evidence is then broken; finalise runs.

    The aggregate is the document a reader is handed, so it has to say the
    stop can no longer be checked rather than repeat the verdict it reached
    while it could.
    """
    out = stopped_session(tmp_path)
    assert lr.apply_rule_r1("exp016a", root=sess(tmp_path))["fired"] is True
    set_report_stop_digest(out["paths"], "b1", "")
    res = lr.session_finalize("exp016a", root=sess(tmp_path))
    assert_no_verdicts(res["aggregate"], needle="stop request")
    assert verify(tmp_path)


# --- 10. the terminal-incomplete path keeps its own semantics ---------------


def terminal_stopped_session(tmp_path, **kw):
    global THRESHOLDS
    out = init(tmp_path)
    THRESHOLDS = json.loads(
        out["paths"]["calibration"].read_text())["thresholds"]
    place_run(out["paths"], out["plan"], "b1", 500, boot="boot-0",
              stop=stop_block(row=500), safety_trip=True,
              outcome="no_report", **kw)
    rewrite_finished(out["paths"], "b1",
                     lambda b: b.update({"tool_failure": "watchdog_died"}))
    return out


def test_r12_a_terminal_run_that_stopped_is_not_held_to_the_completed_rules(
        tmp_path):
    terminal_stopped_session(tmp_path)
    st = lr.session_state("exp016a", root=sess(tmp_path))
    assert st["terminal"] == ["b1"] and st["completed"] == []
    agg = lr.build_aggregate(st)
    assert agg["state"] == "terminal_incomplete"
    assert [r["Q1_run"] for r in agg["runs"]] == [None]
    assert not any("stop request" in p for p in agg["replay_problems"]), \
        agg["replay_problems"]
    assert verify(tmp_path) == [], verify(tmp_path)


def test_r12_a_terminal_run_still_has_to_authenticate_its_stop_request(
        tmp_path):
    """Lenient about the completed-run rules is not lenient about the file."""
    out = terminal_stopped_session(tmp_path)
    (out["paths"]["dir"] / "b1.stop_request.json").unlink()
    assert any("stop request" in p for p in verify(tmp_path)), verify(tmp_path)


# ===========================================================================
# Round 14. Two things the unlocking made dangerous.
#
# 1. main() read the modes in a fixed order, so `--verify --session-next`
#    dispatched to session-next. While the tool was locked that was harmless.
#    Unlocked it is not: a command line that reads like a read-only replay
#    spends a boot, and a measured run that starts and does not finish makes
#    the whole experiment terminal incomplete. Precedence is the wrong answer
#    to that question -- two modes on one command line is a mistake, not a
#    ranking -- so the modes are mutually exclusive.
#
# 2. Arguments were checked by whatever happened to use them first.
#    `--session-init` with no calibration reached `Path(None)` and raised
#    TypeError *after* the experiment directory had been created, which spends
#    the experiment id: sessions are never reopened. Everything a mode needs
#    is now checked centrally, before dispatch, and session_init() creates
#    nothing until every precondition has passed.
#
# Nothing here runs a mode. Every cmd_* is a recorder, and the only command
# lines given to the real CLI as a subprocess are read-only ones.
# ===========================================================================


MODE_FLAGS = ["--session-init", "--session-next", "--session-finalize",
              "--verify", "--session-status", "--from-json", "--show-plan",
              "--microbenchmark", "--child", "--watchdog-worker"]


def record_every_command(mod, monkeypatch):
    """Replace every cmd_* with a recorder, so nothing can run by accident.

    Used everywhere below: a test about refusing a command line must not be
    able to execute one if the refusal is missing.
    """
    seen = []
    for name in dir(mod):
        if name.startswith("cmd_"):
            monkeypatch.setattr(mod, name,
                                lambda args, c=name: seen.append(c) or 0)
    return seen


# --- 1. the modes are mutually exclusive ------------------------------------


def test_r14_the_reported_case_verify_plus_session_next(monkeypatch):
    """`--verify --session-next` used to run session-next and spend a boot."""
    mod = load_cli()
    seen = record_every_command(mod, monkeypatch)
    with pytest.raises(SystemExit) as exc:
        mod.main(["--verify", "--session-next", "--experiment-id", "exp016a"])
    assert exc.value.code == 2
    assert seen == [], "no command may run on an ambiguous command line"


@pytest.mark.parametrize("a,b", list(itertools.combinations(MODE_FLAGS, 2)))
def test_r14_any_two_modes_at_once_are_refused(a, b, monkeypatch):
    mod = load_cli()
    seen = record_every_command(mod, monkeypatch)
    with pytest.raises(SystemExit) as exc:
        mod.main([a, b, "--experiment-id", "exp016a"])
    assert exc.value.code == 2
    assert seen == []


def test_r14_a_single_mode_is_still_accepted(monkeypatch):
    """Exclusivity must not refuse the ordinary case."""
    mod = load_cli()
    for flag in MODE_FLAGS:
        seen = record_every_command(mod, monkeypatch)
        assert mod.main(minimal_argv(flag)) == 0, flag
        assert len(seen) == 1, flag


def test_r14_no_mode_at_all_still_prints_help(monkeypatch):
    mod = load_cli()
    seen = record_every_command(mod, monkeypatch)
    assert mod.main([]) == 0
    assert seen == []


def test_r14_two_read_only_modes_are_refused_by_the_real_cli(
        sessions_unchanged):
    """Through the real CLI -- with two read-only flags, so that a missing
    refusal could not have executed anything even if this test found one."""
    out = subprocess.run([sys.executable, str(CLI), "--verify", "--show-plan",
                          "--experiment-id", "exp016a"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 2
    assert "not allowed with" in out.stderr
    assert "Traceback" not in out.stderr


# --- 2. every mode's arguments, checked before dispatch ---------------------


@pytest.mark.parametrize("argv", [
    ["--session-init"],
    ["--session-init", "--experiment-id", "exp016a"],
    ["--session-init", "--calibration", "/nowhere/calibration.json"],
    ["--session-next"],
    ["--session-finalize"],
    ["--verify"],
    ["--session-status"],
    ["--from-json"],
])
def test_r14_a_mode_without_its_arguments_is_refused(argv, monkeypatch):
    mod = load_cli()
    seen = record_every_command(mod, monkeypatch)
    with pytest.raises(SystemExit) as exc:
        mod.main(argv)
    assert exc.value.code == 2
    assert seen == []


CHILD_ARGS = {"--experiment-dir": "/nowhere/exp016a", "--run": "b1",
              "--rows": "500", "--nonce": "n1", "--plan-digest": "d" * 64}
WATCHDOG_ARGS = {"--experiment-dir": "/nowhere/exp016a", "--run": "b1",
                 "--handshake-fd": "3", "--heartbeat-fd": "4"}


@pytest.mark.parametrize("drop", list(CHILD_ARGS))
def test_r14_the_child_refuses_to_start_without_every_argument(drop,
                                                               monkeypatch):
    mod = load_cli()
    seen = record_every_command(mod, monkeypatch)
    argv = ["--child"]
    for flag, value in CHILD_ARGS.items():
        if flag != drop:
            argv += [flag, value]
    with pytest.raises(SystemExit) as exc:
        mod.main(argv)
    assert exc.value.code == 2
    assert seen == [], f"the child must not start without {drop}"


@pytest.mark.parametrize("drop", list(WATCHDOG_ARGS))
def test_r14_the_watchdog_refuses_to_start_without_every_argument(drop,
                                                                 monkeypatch):
    mod = load_cli()
    seen = record_every_command(mod, monkeypatch)
    argv = ["--watchdog-worker"]
    for flag, value in WATCHDOG_ARGS.items():
        if flag != drop:
            argv += [flag, value]
    with pytest.raises(SystemExit) as exc:
        mod.main(argv)
    assert exc.value.code == 2
    assert seen == [], f"the watchdog must not start without {drop}"


def test_r14_the_argv_the_parent_builds_passes_the_same_validation(monkeypatch):
    """The check must not refuse the command lines the tool itself writes."""
    mod = load_cli()
    seen = record_every_command(mod, monkeypatch)
    paths = {"dir": Path("/nowhere/exp016a")}
    argv = mod._child_argv(paths, "b2", {"declared_rows": 1000}, "nonce-1",
                           {"plan_digest": "e" * 64})[2:]
    assert mod.main(argv) == 0
    assert seen == ["cmd_child"]

    seen.clear()
    argv = mod._watchdog_argv(paths, "b3", 11, 12)[2:]
    assert mod.main(argv) == 0
    assert seen == ["cmd_watchdog_worker"]


def test_r14_a_missing_argument_is_a_clean_exit_not_a_traceback(
        sessions_unchanged):
    out = subprocess.run([sys.executable, str(CLI), "--verify"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 2
    assert "Traceback" not in out.stderr
    assert "--experiment-id" in out.stderr


def test_r14_show_plan_and_microbenchmark_need_no_experiment_id(monkeypatch):
    """The two modes that genuinely do not need one keep working."""
    mod = load_cli()
    seen = record_every_command(mod, monkeypatch)
    assert mod.main(["--show-plan"]) == 0
    assert mod.main(["--microbenchmark"]) == 0
    assert seen == ["cmd_show_plan", "cmd_microbenchmark"]


# --- 3. session_init creates nothing until everything has passed ------------


R14_SOURCES = ("src/training/longrun.py", "src/training/watchdog.py")


def a_good_calibration(tmp_path) -> dict:
    good = tmp_path / "good"
    good.mkdir(parents=True, exist_ok=True)
    return json.loads(calibration_file(good).read_text())


def unusable_calibration(tmp_path, how):
    """Every way a calibration can fail to be one."""
    if how == "none":
        return None
    path = tmp_path / "calibration.json"
    if how == "missing":
        return path
    if how == "bad_json":
        path.write_text("{not json at all")
        return path
    if how == "not_an_object":
        body = [1, 2, 3]
    elif how == "empty_object":
        body = {}
    elif how == "no_thresholds":
        body = a_good_calibration(tmp_path)
        body.pop("thresholds")
    elif how == "thresholds_do_not_recompute":
        body = a_good_calibration(tmp_path)
        body["thresholds"]["swap_used_gb"] = 99.0
    elif how == "wrong_policy":
        body = a_good_calibration(tmp_path)
        body["policy"]["consecutive_passes_required"] = 1
    else:  # pragma: no cover - the parametrisation is closed
        raise AssertionError(how)
    path.write_text(json.dumps(body))
    return path


@pytest.mark.parametrize("how", ["none", "missing", "bad_json",
                                 "not_an_object", "empty_object",
                                 "no_thresholds",
                                 "thresholds_do_not_recompute",
                                 "wrong_policy"])
def test_r14_an_unusable_calibration_leaves_no_experiment_behind(tmp_path, how):
    """The directory used to be created first, and the experiment id with it.

    Sessions are never reopened, so an empty directory left behind by a
    TypeError has spent the id: the next attempt has to invent a new one and
    explain why.
    """
    root = tmp_path / "reports"
    calib = unusable_calibration(tmp_path, how)
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        lr.session_init("exp016a", calibration_path=calib, root=root,
                        code_files=R14_SOURCES)
    assert not (root / "exp016a").exists()
    assert not root.exists(), "not even an empty report 16 root may be left"


def test_r14_a_design_that_disagrees_leaves_no_experiment_behind(tmp_path):
    root = tmp_path / "reports"
    design = json.loads(lr.DESIGN_JSON.read_text())
    design["frozen_constants"]["bands"]["Q1_holds_D100_max"] = 99.0
    alt = tmp_path / "design.json"
    alt.write_text(json.dumps(design))
    with pytest.raises(ValueError, match="disagree with the approved design"):
        lr.session_init("exp016a",
                        calibration_path=calibration_file(tmp_path),
                        root=root, design_path=alt, code_files=R14_SOURCES)
    assert not root.exists()


def test_r14_a_missing_source_file_leaves_no_experiment_behind(tmp_path):
    root = tmp_path / "reports"
    with pytest.raises((FileNotFoundError, SystemExit)):
        lr.session_init("exp016a",
                        calibration_path=calibration_file(tmp_path),
                        root=root,
                        code_files=("src/training/longrun.py",
                                    "src/training/there_is_no_such_file.py"))
    assert not root.exists()


def test_r14_an_unsafe_experiment_id_leaves_no_experiment_behind(tmp_path):
    root = tmp_path / "reports"
    with pytest.raises(ValueError):
        lr.session_init("../escape", calibration_path=None, root=root,
                        code_files=R14_SOURCES)
    assert not root.exists()


def test_r14_a_clean_session_init_still_works(tmp_path):
    """The checks refuse what is wrong, not what is right."""
    out = init(tmp_path)
    p = out["paths"]
    assert p["dir"].exists() and p["plan"].exists() and p["session"].exists()
    assert p["calibration"].exists() and p["snapshot"].exists()
    assert out["plan"]["experiment_id"] == "exp016a"
    assert out["session"]["thresholds"] == json.loads(
        p["calibration"].read_text())["thresholds"]


# --- 4. the no-session message speaks only for the id it was given ----------


def test_r14_the_no_session_message_is_about_this_experiment_only():
    """It used to say "nothing has been run yet", which will stop being true.

    Once one experiment exists, that sentence is a false global claim printed
    by a per-experiment lookup.
    """
    out = subprocess.run([sys.executable, str(CLI), "--verify",
                          "--experiment-id", "exp016a"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 1
    assert "exp016a" in out.stdout
    assert "nothing has been run" not in out.stdout.lower()
    assert "--session-init" in out.stdout


def test_r14_the_gate_docstring_does_not_restate_the_call_below_it():
    mod = load_cli()
    doc = mod.Launcher.gate.__doc__ or ""
    for name in ("needed_consecutive", "poll_seconds", "max_wait_seconds"):
        assert name not in doc, (
            f"{name} is in the four lines of code directly below; repeating "
            "the list in prose is what goes stale")


def test_r14_every_mode_dispatches_through_a_command_function():
    """One mode, one cmd_*: otherwise a recorder cannot see it coming.

    `--session-finalize` used to be handled inline in main(), so every test
    that installs recorders and asserts "no command ran" was blind to it --
    including the ones above about refusing an ambiguous command line.
    """
    mod = load_cli()
    for dest in mod.MODES:
        assert hasattr(mod, f"cmd_{dest}"), dest


def test_r14_session_finalize_reaches_its_own_command(monkeypatch):
    mod = load_cli()
    seen = record_every_command(mod, monkeypatch)
    assert mod.main(["--session-finalize", "--experiment-id", "exp016a"]) == 0
    assert seen == ["cmd_session_finalize"]


# ===========================================================================
# Round 16. A calibration report 16 can actually use.
#
# The first --session-init dry run was refused before it wrote anything:
# report 15's archived calibration carries a `gate` block, and report 16's
# validator wants a `policy` block. The three settings inside are numerically
# identical -- 3 consecutive passes, 30s apart, 900s ceiling -- but they are
# spelled `poll_seconds` and `max_wait_seconds` there and
# `poll_interval_seconds` and `timeout_seconds` here.
#
# Three ways out, and only one of them keeps the evidence: recalibrate (new
# numbers, from a machine in a different state, so exp002's thresholds stop
# being the ones in force), widen the validator to accept both schemas (two
# definitions of a gate policy, which is the failure this project has spent
# six rounds removing), or adapt the schema and say so. The third is what
# happened here: nothing is remeasured, every number is carried over
# verbatim, and the new file records exactly where each field came from.
#
# The source file is not touched. Its SHA is pinned here and checked before
# and after.
# ===========================================================================


R16_SOURCE = ROOT / "data" / "reports" / "15_mps_order" / "calibration.json"
R16_SOURCE_SHA = "48439500c6162d6b4f4a38cb2b5a38846549386adffb6e3a8f239b902fc25660"
R16_CALIBRATION = ROOT / "data" / "reports" / "16_longrun_calibration.json"

#: new name -> the report 15 gate field it was taken from
R16_POLICY_MAP = {"consecutive_passes_required": "consecutive_passes_required",
                  "poll_interval_seconds": "poll_seconds",
                  "timeout_seconds": "max_wait_seconds"}

#: What has to survive the adaptation unchanged: the measurement itself, and
#: the context needed to read it.
R16_MEASUREMENT_FIELDS = ("created_at", "loads_model", "samples_requested",
                          "interval_seconds", "metrics", "scale_formula",
                          "platform", "machine", "note")


@pytest.fixture
def r16_source():
    return json.loads(R16_SOURCE.read_text())


@pytest.fixture
def r16_doc():
    assert R16_CALIBRATION.exists(), f"{R16_CALIBRATION.name} does not exist"
    return json.loads(R16_CALIBRATION.read_text())


def test_r16_the_source_calibration_is_the_archived_one():
    assert R16_SOURCE.exists()
    assert hashlib.sha256(R16_SOURCE.read_bytes()).hexdigest() == R16_SOURCE_SHA


def test_r16_the_source_calibration_is_not_usable_by_report_16(r16_source):
    """The premise. Without it the new file would be solving nothing."""
    problems = lr.validate_calibration(r16_source)
    assert problems == ["the calibration has no gate policy"], problems


def test_r16_the_new_file_declares_its_own_schema(r16_doc):
    assert r16_doc["schema_version"] == 1
    assert r16_doc["kind"] == "longrun_calibration"


def test_r16_the_new_file_names_the_document_it_came_from(r16_doc):
    src = r16_doc["source"]
    assert src["path"] == "data/reports/15_mps_order/calibration.json"
    assert src["sha256"] == R16_SOURCE_SHA
    assert src["schema_version"] == 3
    assert src["kind"] == "preflight_calibration"
    assert src["calibration_digest"] == json.loads(
        R16_SOURCE.read_text())["calibration_digest"]


@pytest.mark.parametrize("field", ["samples", "stats", "thresholds"])
def test_r16_the_measurement_is_carried_over_verbatim(r16_doc, r16_source,
                                                      field):
    assert r16_doc[field] == r16_source[field]


@pytest.mark.parametrize("field", R16_MEASUREMENT_FIELDS)
def test_r16_the_measurement_context_is_kept(r16_doc, r16_source, field):
    """A threshold with no record of the machine it was taken on is a number."""
    assert r16_doc["measurement"][field] == r16_source[field]


def test_r16_report_15s_gate_does_not_survive_at_the_top_level(r16_doc):
    """Two policy-shaped blocks is how a tool ends up reading the wrong one."""
    assert "gate" not in r16_doc


def test_r16_the_source_digest_is_not_promoted_to_this_documents_digest(
        r16_doc, r16_source):
    """`calibration_digest` describes the source bytes, not these.

    Left at the top level it would read as this file's own digest and be
    wrong about it -- so it lives inside `source`, where it is a fact about
    another document.
    """
    assert "calibration_digest" not in r16_doc
    assert r16_doc["source"]["calibration_digest"] == \
        r16_source["calibration_digest"]


def test_r16_there_is_exactly_one_execution_policy(r16_doc):
    assert r16_doc["policy"] == {"consecutive_passes_required": 3,
                                 "poll_interval_seconds": 30,
                                 "timeout_seconds": 900}
    assert r16_doc["policy"] == lr.GATE_POLICY


@pytest.mark.parametrize("new,old", sorted(R16_POLICY_MAP.items()))
def test_r16_every_policy_field_traces_to_a_report_15_field(r16_doc, r16_source,
                                                            new, old):
    """Not "the numbers happen to match": each one is read from its source."""
    assert r16_doc["derivation"]["policy_field_map"][f"policy.{new}"] == \
        f"gate.{old}"
    assert r16_doc["policy"][new] == r16_source["gate"][old]


def test_r16_the_derivation_says_nothing_was_remeasured(r16_doc):
    assert r16_doc["derivation"]["remeasured"] is False


def test_r16_no_second_block_could_be_mistaken_for_the_policy(r16_doc):
    """Only one object in the file has the shape of a gate policy."""
    policy_keys = set(lr.GATE_POLICY)
    found = []

    def walk(node, path):
        if isinstance(node, dict):
            if policy_keys & set(node) or {"poll_seconds",
                                           "max_wait_seconds"} & set(node):
                found.append(path)
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(r16_doc, "")
    assert found == [".policy"], found


def test_r16_report_16_accepts_the_new_calibration(r16_doc):
    assert lr.validate_calibration(r16_doc) == []


def test_r16_the_thresholds_still_recompute_from_the_samples(r16_doc):
    from src.training.preflight import calibrate, thresholds_from
    assert thresholds_from(calibrate(r16_doc["samples"])) == r16_doc["thresholds"]


def test_r16_session_init_would_now_accept_it():
    """The check that refused the first attempt, run against the new file.

    `load_usable_calibration` writes nothing -- it is the part of
    session_init() that happens before anything exists.
    """
    calib = lr.load_usable_calibration(R16_CALIBRATION)
    assert calib["thresholds"]["swap_used_gb"] == 0.537
    assert lr.validate_calibration(calib) == []


def test_r16_the_derivation_is_reproducible(tmp_path):
    """Re-derive from the source and compare the *bytes* with what is on disk.

    Provenance a reader cannot recompute is a claim. This one is a check --
    and it has to be a check on bytes. Comparing two dicts proves the values
    agree; it says nothing about the file, which is what gets copied into the
    session and put under a digest at ``--session-init``. Key order,
    indentation and the trailing newline are all part of what
    ``calibration_sha256`` will cover.

    So the comparison runs the real publishing path: the same
    ``write_once_json`` the derivation script uses, into pytest's temporary
    directory -- never near ``data/reports/`` -- and then a byte-for-byte
    comparison with the published file.

    Field-by-field agreement with the source is checked by the neighbouring
    tests; the source-drift refusal is checked by the one below.
    """
    from src.training.session import write_once_json

    mod = load_derivation()
    fresh = mod.derive(R16_SOURCE)
    replica = tmp_path / R16_CALIBRATION.name
    assert not replica.exists()
    write_once_json(replica, fresh)
    assert replica.parent == tmp_path, "nothing may be written near the report"

    assert replica.read_bytes() == R16_CALIBRATION.read_bytes(), (
        "the published calibration is not what derive() and the real writer "
        "produce from the archived source")
    assert hashlib.sha256(replica.read_bytes()).hexdigest() == \
        hashlib.sha256(R16_CALIBRATION.read_bytes()).hexdigest()
    # Byte equality implies this, but a dict diff is what a reader can act on
    # when the bytes stop matching.
    assert fresh == json.loads(R16_CALIBRATION.read_text())


def test_r16_the_derivation_refuses_a_source_that_is_not_the_archived_one(
        tmp_path):
    mod = load_derivation()
    altered = tmp_path / "calibration.json"
    body = json.loads(R16_SOURCE.read_text())
    body["thresholds"]["swap_used_gb"] = 99.0
    altered.write_text(json.dumps(body))
    with pytest.raises(SystemExit):
        mod.derive(altered)


def test_r16_the_source_file_was_not_touched():
    """Stated last, so it is the final word of this section.

    Behavioural: the archived calibration is published, so this runs
    everywhere. The exp002 half of the old test moved to the artifact-only
    one below rather than dragging this check into a skip with it.
    """
    assert hashlib.sha256(R16_SOURCE.read_bytes()).hexdigest() == R16_SOURCE_SHA


@needs_exp002
def test_r16_exp002_was_not_touched_by_the_derivation():
    assert (EXP002_DIR / "aggregate.json").exists()
    assert len(list((EXP002_DIR / "events").glob("*.json"))) == 8


def test_r16_no_report_16_session_was_created_by_any_of_this(
        sessions_unchanged):
    """Deriving a calibration creates nothing. exp001 came later, on purpose.

    The behaviour is "no session appeared"; which sessions already existed is
    not this test's business, so it compares against its own baseline.
    """
    assert report_16_sessions() == sessions_unchanged


@needs_exp001
def test_r16_the_derivation_left_exp001_without_a_measured_row():
    assert not (EXP001_DIR / "b1.json").exists()


# ===========================================================================
# Round 18. What the first real b1 found out.
#
# exp001's b1 spent a boot, ran for 51 seconds and died loading the tokenizer:
# `AutoTokenizer.from_pretrained` asks for the model's `config.json`, the
# BrickGPT repo is adapter-only and has never had one, and with the network
# down the hub cannot tell "there is no such file" from "I cannot reach the
# server". So the measured child had an undeclared network dependency at load
# time, and one momentary DNS failure was enough to make the whole experiment
# terminal incomplete.
#
# Five things came out of it, and each has tests here:
#
#   1. the production loader must be strictly local, and must not probe for a
#      file the repo does not publish;
#   2. everything the child needs must be resolved read-only *before* the
#      gate, the watchdog and `measurement_started` -- a dependency that is
#      missing is a retryable failure, not a spent boot;
#   3. the source manifest missed modules the child actually executes;
#   4. a child that dies leaving no report left nothing behind but an exit
#      code, so the reason had to be reconstructed from a console that is not
#      evidence;
#   5. `--session-status` showed a terminal experiment's runs as `pending`,
#      which reads as an invitation to run them.
# ===========================================================================


R18_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json")


def pinned_snapshot_dir():
    """Where the pinned BrickGPT revision sits in the local hub cache.

    Derived, never hard-coded: the path contains a home directory and must
    not travel into a report or a test fixture.
    """
    from huggingface_hub.constants import HF_HUB_CACHE
    from src.model_ids import TOKENIZER, TOKENIZER_REVISION
    return (Path(HF_HUB_CACHE) / f"models--{TOKENIZER.replace('/', '--')}"
            / "snapshots" / TOKENIZER_REVISION)


@pytest.fixture
def adapter_only_repo(tmp_path):
    """A local copy of the pinned repo's tokenizer files, and nothing else.

    This is the shape that broke b1: three tokenizer files, no `config.json`,
    because an adapter repo does not have one.
    """
    src = pinned_snapshot_dir()
    if not all((src / f).exists() for f in R18_TOKENIZER_FILES):
        pytest.skip("the pinned BrickGPT tokenizer files are not in the cache")
    repo = tmp_path / "brickgpt_repo"
    repo.mkdir()
    for name in R18_TOKENIZER_FILES:
        shutil.copyfile(src / name, repo / name)
    assert not (repo / "config.json").exists()
    return repo


def no_network(monkeypatch):
    """Make every outbound socket raise, so a network call cannot pass."""
    import socket
    tried = []

    def refuse(*a, **k):
        tried.append(a[:1])
        raise OSError("network access is forbidden in this test")

    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    return tried


# --- 1. the tokenizer loads strictly locally, without a config.json ---------


def test_r18_the_tokenizer_loads_from_an_adapter_only_repo(adapter_only_repo):
    """The exact case that killed b1: no config.json anywhere."""
    from src.generation.brickgpt import load_tokenizer
    tok = load_tokenizer(str(adapter_only_repo), revision=None,
                         local_files_only=True)
    assert tok.encode("1x2", add_special_tokens=False)


def test_r18_the_tokenizer_load_makes_no_network_call(adapter_only_repo,
                                                      monkeypatch):
    """Not "the hub returned 404 quickly" -- nothing may leave the process."""
    from src.generation.brickgpt import load_tokenizer
    tried = no_network(monkeypatch)
    tok = load_tokenizer(str(adapter_only_repo), revision=None,
                         local_files_only=True)
    assert tok is not None
    assert tried == [], f"the loader tried to reach the network: {tried}"


def test_r18_the_loader_uses_the_class_the_repo_declares(adapter_only_repo):
    """The repo's own `tokenizer_class`, not whatever Auto* infers."""
    from src.generation.brickgpt import load_tokenizer, declared_tokenizer_class
    declared = json.loads(
        (adapter_only_repo / "tokenizer_config.json").read_text())
    assert declared["tokenizer_class"] == "PreTrainedTokenizerFast"
    import transformers
    cls = declared_tokenizer_class(str(adapter_only_repo), revision=None,
                                   local_files_only=True)
    # Resolved through transformers by the declared name. `__name__` is not
    # the check: this transformers exposes `PreTrainedTokenizerFast` as an
    # alias for a renamed class, and an alias is still the declared class.
    assert cls is getattr(transformers, declared["tokenizer_class"])

    # The instance transformers hands back is an implementation detail and has
    # changed between versions, so the check is on which class was *used*.
    seen = []
    original = cls.from_pretrained

    def spy(*a, **k):
        seen.append((a, k))
        return original(*a, **k)

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(cls, "from_pretrained", staticmethod(spy))
        tok = load_tokenizer(str(adapter_only_repo), revision=None,
                             local_files_only=True)
    finally:
        monkey.undo()
    assert len(seen) == 1, "the declared class was not the one that loaded it"
    assert seen[0][1]["local_files_only"] is True
    assert tok.encode("1x2", add_special_tokens=False)


def test_r18_auto_tokenizer_is_not_on_the_strict_offline_path():
    """AutoTokenizer is what asked for config.json. It may not be called."""
    import src.generation.brickgpt as bg
    calls = []

    class Trap:
        @staticmethod
        def from_pretrained(*a, **k):
            calls.append((a, k))
            raise AssertionError("AutoTokenizer must not be used offline")

    original = bg.AutoTokenizer
    bg.AutoTokenizer = Trap
    try:
        src = pinned_snapshot_dir()
        if not (src / "tokenizer.json").exists():
            pytest.skip("the pinned tokenizer is not in the cache")
        bg.load_tokenizer(str(src), revision=None, local_files_only=True)
    finally:
        bg.AutoTokenizer = original
    assert calls == []


# --- 2. every load the production child performs is pinned and local --------


def test_r18_the_production_child_pins_and_localises_every_load(monkeypatch):
    """tokenizer, base model and published adapter: revision + local only."""
    import src.training.lora as lora
    from src import model_ids
    seen = {}

    class FakeBase:
        @staticmethod
        def from_pretrained(name, **kw):
            seen["base"] = (name, kw)
            return FakeBase()

    class FakePeft:
        @staticmethod
        def from_pretrained(model, name, **kw):
            seen["adapter"] = (name, kw)
            return FakePeft()

        def merge_and_unload(self):
            return FakeBase()

    monkeypatch.setattr(lora, "_weight_fingerprint",
                        lambda m: 1.0 if "merged" not in seen else 2.0)
    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(AutoModelForCausalLM=FakeBase))
    monkeypatch.setitem(sys.modules, "peft",
                        types.SimpleNamespace(PeftModel=FakePeft))
    fp = [1.0, 2.0]
    monkeypatch.setattr(lora, "_weight_fingerprint", lambda m: fp.pop(0))

    lora.load_merged_brickgpt(local_files_only=True)
    assert seen["base"][0] == model_ids.BASE_MODEL
    assert seen["base"][1]["revision"] == model_ids.BASE_REVISION
    assert seen["base"][1]["local_files_only"] is True
    assert seen["adapter"][0] == lora.PUBLISHED_ADAPTER
    assert seen["adapter"][1]["revision"] == lora.PUBLISHED_REVISION
    assert seen["adapter"][1]["local_files_only"] is True


def test_r18_the_child_environment_pins_offline(monkeypatch):
    """The child does not inherit whether the operator remembered to export."""
    assert lr.PRODUCTION_OFFLINE_ENV == {"HF_HUB_OFFLINE": "1",
                                         "TRANSFORMERS_OFFLINE": "1",
                                         "HF_HUB_DISABLE_TELEMETRY": "1"}
    for key in lr.PRODUCTION_OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    lr.enforce_offline_environment()
    for key, value in lr.PRODUCTION_OFFLINE_ENV.items():
        assert os.environ[key] == value


# The launcher's half of this is checked by intercepting ``Popen`` and reading
# the environment it was actually handed -- see
# ``test_r19_spawn_child_overrides_the_inherited_offline_variables``. Round 18
# asserted on ``inspect.getsource`` instead, which proves a name appears in a
# function body and nothing about what the child receives.


# --- 3. dependency preflight happens before anything is spent --------------


def failing_preflight(reason="the pinned base model is not in the cache"):
    return lambda: {"ok": False, "problems": [reason], "evidence": None}


def test_r18_a_missing_dependency_spends_no_boot(tmp_path):
    init(tmp_path)
    gate = counting_gate()
    spawned = []
    out = lr.session_next("exp016a", **next_args(
        tmp_path, gate=gate,
        dependency_preflight=failing_preflight(),
        spawn_watchdog=lambda run, paths: spawned.append("watchdog") or {
            "ready": True, "proc": None},
        spawn_child=lambda **kw: spawned.append("child") or {"spawned": True}))
    assert out["ok"] is False
    assert out["boot_consumed"] is False
    assert out.get("retryable") is True
    assert out.get("dependency_preflight_failed") is True
    assert gate.calls == [], "the gate must not be polled"
    assert spawned == [], "nothing may be spawned"
    st = lr.session_state("exp016a", root=sess(tmp_path))
    assert len(st["events"]) == 0, "no event may be written"
    assert st["terminal"] == [] and st["next_run"] == "b1"


def test_r18_the_preflight_runs_before_the_gate(tmp_path):
    """Order, not merely presence: the gate costs the operator's evening."""
    init(tmp_path)
    order = []
    gate = counting_gate()

    def watching_gate():
        order.append("gate")
        return gate()

    out = lr.session_next("exp016a", **next_args(
        tmp_path, gate=watching_gate,
        dependency_preflight=lambda: order.append("preflight") or {
            "ok": True, "problems": [], "evidence": {"checked": []}}))
    assert out["ok"], out.get("problems")
    assert order[:2] == ["preflight", "gate"]


def test_r18_a_passing_preflight_leaves_the_launch_order_unchanged(tmp_path):
    init(tmp_path)
    out = lr.session_next("exp016a", **next_args(tmp_path))
    assert out["ok"], out.get("problems")
    st = lr.session_state("exp016a", root=sess(tmp_path))
    assert [e["event"] for e in st["events"]] == [lr.EVENT_STARTED,
                                                  lr.EVENT_FINISHED]
    assert st["completed"] == ["b1"]


def test_r18_the_real_preflight_touches_no_network_and_loads_no_model(
        monkeypatch):
    tried = no_network(monkeypatch)
    import torch
    calls = []
    monkeypatch.setattr(torch, "manual_seed",
                        lambda *a, **k: calls.append("seed"))
    out = lr.dependency_preflight()
    assert set(out) >= {"ok", "problems", "evidence"}
    assert tried == [], f"the preflight reached the network: {tried}"
    assert calls == [], "the preflight must not touch torch"
    if out["ok"]:
        ev = out["evidence"]
        blob = json.dumps(ev)
        assert str(Path.home()) not in blob, "no personal absolute path"
        assert "/Users/" not in blob and "/home/" not in blob
        for item in ev["repositories"]:
            assert set(item) >= {"repo_id", "revision", "files"}


# --- 4. the source manifest covers what the child actually runs ------------


CLOSURE_PROBE = '''
import json, pathlib, sys
ROOT = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
import src.training.longrun as lr           # noqa: F401
from src.generation.brickgpt import load_tokenizer   # noqa: F401
from src.model_ids import ADAPTER                     # noqa: F401
from src.training.lora import (LoraConfig_, assert_only_lora_trainable,
                               build_model, collate, encode_row, read_rows,
                               sample_pairs)          # noqa: F401
import src.training.watchdog, src.training.session, src.training.preflight
# encode_row imports this lazily, inside the call the child makes per row.
from src.data.instruction import encode                # noqa: F401
out = set()
for m in list(sys.modules.values()):
    f = getattr(m, "__file__", None)
    if not f:
        continue
    p = pathlib.Path(f).resolve()
    try:
        rel = p.relative_to(ROOT)
    except ValueError:
        continue
    # torch exposes a couple of modules whose __file__ is a bare name, which
    # resolves against the cwd and looks repository-local without being so.
    if not p.is_file() or rel.parts[0] in (".venv", "tests", "scripts"):
        continue
    out.add(str(rel))
print(json.dumps(sorted(out)))
'''


def repository_local_import_closure():
    out = subprocess.run([sys.executable, "-c", CLOSURE_PROBE, str(ROOT)],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_r18_every_module_the_child_imports_is_in_the_manifest():
    """The guard that would have caught `src/generation/brickgpt.py`.

    A repository-local module the measured child executes but the snapshot
    does not record is a hole in the provenance: `--verify` cannot say what
    code produced the numbers.
    """
    closure = repository_local_import_closure()
    missing = sorted(set(closure) - set(lr.CODE_FILES))
    assert missing == [], (
        f"the production child imports {missing}, which CODE_FILES does not "
        "cover; add them to CODE_FILES or stop importing them at runtime")


def test_r18_the_known_gaps_are_now_covered():
    for rel in ("src/generation/brickgpt.py", "src/generation/prompt.py",
                "src/data/bricks.py"):
        assert rel in lr.CODE_FILES, rel


def test_r18_instruction_does_not_drag_in_the_solver_at_runtime():
    """`counterfactual` -> `retile` -> OR-Tools was imported for a type hint.

    Either it is part of the child and goes in the manifest, or it is a
    annotation and must not be imported. It is an annotation.
    """
    probe = ('import sys, json; sys.path.insert(0, %r);'
             'import src.data.instruction;'
             'print(json.dumps(sorted(m for m in sys.modules '
             'if m.startswith(("src.data.counterfactual", "src.data.retile", '
             '"ortools")))))' % str(ROOT))
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def test_r18_instruction_still_type_checks_and_works():
    from src.data.instruction import Example
    assert hasattr(Example, "from_sample")


# --- 5. durable failure evidence -------------------------------------------


def nonzero_exit_collector(*, stage="model_load", exc="OSError"):
    """A collector whose child died without writing a report."""
    def collect(*, run, paths):
        lr.write_failure_evidence(
            paths, run, experiment_id=paths["dir"].name, stage=stage,
            exception_type=exc,
            summary="the pinned tokenizer could not be resolved offline")
        wpath = paths["dir"] / f"{run}.watchdog.jsonl"
        return {"outcome": "nonzero_exit", "exit_status": 1,
                "report_sha256": None,
                "watchdog_sha256": (hashlib.sha256(wpath.read_bytes()).hexdigest()
                                    if wpath.exists() else None)}
    return collect


def failed_run_session(tmp_path, **kw):
    global THRESHOLDS
    out = init(tmp_path)
    THRESHOLDS = json.loads(
        out["paths"]["calibration"].read_text())["thresholds"]
    res = lr.session_next("exp016a", **next_args(
        tmp_path, collect=nonzero_exit_collector(**kw)))
    return out, res


def test_r18_a_child_that_dies_leaves_replayable_evidence(tmp_path):
    out, res = failed_run_session(tmp_path)
    paths = out["paths"]
    assert res["ok"] is False
    ev_path = paths["dir"] / "b1.failure.json"
    assert ev_path.exists(), "an exit code is not a reason"
    body = json.loads(ev_path.read_text())
    assert body["schema_version"] == 1
    assert body["kind"] == "longrun_failure_evidence"
    assert body["experiment_id"] == "exp016a" and body["run_id"] == "b1"
    assert body["stage"] == "model_load"
    assert body["exception_type"] == "OSError"
    assert body["summary"]

    fin = finished_body_of(paths, "b1")
    assert fin["failure_evidence_sha256"] == hashlib.sha256(
        ev_path.read_bytes()).hexdigest()
    assert verify(tmp_path) == [], verify(tmp_path)


@pytest.mark.parametrize("how", ["missing", "tampered", "digest"])
def test_r18_broken_failure_evidence_is_refused(tmp_path, how):
    out, _ = failed_run_session(tmp_path)
    paths = out["paths"]
    ev = paths["dir"] / "b1.failure.json"
    if how == "missing":
        ev.unlink()
    elif how == "tampered":
        body = json.loads(ev.read_text())
        body["stage"] = "something_else"
        ev.write_text(json.dumps(body))
    else:
        rewrite_chained(paths, f"b1-{lr.EVENT_FINISHED}",
                        lambda b: b.update({"failure_evidence_sha256": "a" * 64}))
    problems = verify(tmp_path)
    assert any("failure evidence" in p for p in problems), problems


def test_r18_failure_evidence_carries_nothing_personal(tmp_path):
    out, _ = failed_run_session(tmp_path)
    blob = (out["paths"]["dir"] / "b1.failure.json").read_text()
    assert str(Path.home()) not in blob
    assert "/Users/" not in blob and "/home/" not in blob
    # Credential and identity *shapes*, not any substring that happens to
    # spell part of an ordinary word -- "tokenizer" contains "token".
    import re as _re
    for pattern, what in (
            (r"[\w.+-]+@[\w-]+\.[\w.-]+", "an email address"),
            (r"\b(hf_|sk-|ghp_)[A-Za-z0-9]{8,}", "an API token"),
            (r"\borg-[A-Za-z0-9]{6,}", "an organization id"),
            (r"(?i)\bbearer\s+\S+", "a bearer credential")):
        assert not _re.search(pattern, blob), f"{what} must not appear"


def test_r18_the_redactor_removes_what_a_real_traceback_would_carry():
    """Driven with the shapes an exception actually leaks."""
    dirty = (f"OSError at {Path.home()}/.cache/huggingface/token for "
             "user someone@example.com org-ABCDEF123456 "
             "Authorization: Bearer hf_AAAABBBBCCCCDDDD")
    clean = lr._portable(dirty)
    assert str(Path.home()) not in clean
    assert "someone@example.com" not in clean
    assert "hf_AAAABBBBCCCCDDDD" not in clean
    assert "org-ABCDEF123456" not in clean
    assert "OSError" in clean, "the shape of the failure has to survive"


def test_r18_a_run_with_no_failure_evidence_field_still_replays(tmp_path):
    """exp001 was written before this field existed and must still verify."""
    out = init(tmp_path)
    global THRESHOLDS
    THRESHOLDS = json.loads(
        out["paths"]["calibration"].read_text())["thresholds"]
    place_run(out["paths"], out["plan"], "b1", 500, boot="boot-0",
              outcome="no_report", terminal=False)
    rewrite_finished(out["paths"], "b1",
                     lambda b: b.pop("failure_evidence_sha256", None))
    assert verify(tmp_path) == [], verify(tmp_path)


# --- 6. a terminal experiment does not advertise a next run ----------------


def test_r18_session_status_says_terminal_and_blocked(tmp_path):
    out, _ = failed_run_session(tmp_path)
    st = lr.session_status("exp016a", root=sess(tmp_path))
    rows = lr.session_status_rows(st)
    assert rows["b1"] == "terminal"
    assert rows["b2"] == "blocked (experiment terminal)"
    assert rows["b3"] == "blocked (experiment terminal)"
    assert lr.session_next_hint(st) == "none: the experiment is terminal " \
                                       "incomplete"


def test_r18_a_live_session_still_shows_pending(tmp_path):
    init(tmp_path)
    st = lr.session_status("exp016a", root=sess(tmp_path))
    rows = lr.session_status_rows(st)
    assert rows == {"b1": "pending", "b2": "pending", "b3": "pending"}
    assert lr.session_next_hint(st) == "b1"


def test_r18_the_cli_prints_terminal_rather_than_pending(tmp_path):
    out, _ = failed_run_session(tmp_path)
    mod = load_cli()
    printed = []
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr("builtins.print", lambda *a, **k: printed.append(
            " ".join(str(x) for x in a)))
        # `mod.longrun` is this same module object, so the originals have to
        # be captured before patching or the redirect calls itself.
        orig_paths, orig_status = lr.session_paths, lr.session_status
        monkey.setattr(mod.longrun, "session_paths",
                       lambda eid, root=None: orig_paths(eid, root=sess(tmp_path)))
        monkey.setattr(mod.longrun, "session_status",
                       lambda eid, root=None: orig_status(eid, root=sess(tmp_path)))
        mod.cmd_session_status(types.SimpleNamespace(experiment_id="exp016a"))
    finally:
        monkey.undo()
    text = "\n".join(printed)
    assert "terminal" in text
    assert "blocked" in text
    assert "pending" not in text


def test_r18_session_next_still_refuses_a_terminal_experiment(tmp_path):
    out, _ = failed_run_session(tmp_path)
    again = lr.session_next("exp016a", **next_args(
        tmp_path, boot={"boot_fingerprint": "boot-99"}))
    assert again["ok"] is False and again.get("terminal") is True


# ===========================================================================
# Round 19. Codex's re-review of round 18.
#
# Round 18 published failure evidence and then tested it with a *collector*
# that wrote the evidence itself. That proves the replay accepts a file the
# test wrote; it proves nothing about the child. And only `deps.load` was
# ever wrapped, so a child that died in a training step, in a scheduled
# clear, in the teardown or on the way to disk still left an exit code and
# no reason -- which is the exact gap round 18 existed to close.
#
# Everything below drives the real `run_child` and lets one stage fail at a
# time. Nothing here writes the evidence on the child's behalf.
# ===========================================================================


class BreakingDeps(lr.FakeChildDeps):
    """`FakeChildDeps` with one named thing arranged to fail.

    A subclass, not a fresh fake: the failure has to arrive through the same
    load/step/clear/probe/teardown contract the production deps offer, or the
    test is exercising a function the child does not call.
    """

    def __init__(self, *, break_at=None, at_row=1, message="the row went away",
                 exc_type=RuntimeError, teardown_exc=None, **kw):
        super().__init__(**kw)
        self.break_at = break_at
        self.at_row = at_row
        self.message = message
        self.exc_type = exc_type
        self.teardown_exc = teardown_exc
        self.teardown_calls = 0
        self.rows_stepped = 0

    def _boom(self):
        raise self.exc_type(self.message)

    def load(self, *, rows: int) -> dict:
        if self.break_at == "load":
            self._boom()
        loaded = dict(super().load(rows=rows))
        step, clear = loaded["step"], loaded["clear"]
        probe, teardown = loaded["probe"], loaded["teardown"]

        def wrapped_step(index, position):
            self.rows_stepped = position
            if self.break_at == "step" and position == self.at_row:
                self._boom()
            return step(index, position)

        def wrapped_clear():
            if self.break_at == "clear":
                self._boom()
            return clear()

        def wrapped_probe():
            if self.break_at == "probe":
                self._boom()
            return probe()

        def wrapped_teardown():
            self.teardown_calls += 1
            if self.teardown_exc is not None:
                raise self.teardown_exc("the teardown clear also failed")
            if self.break_at == "teardown":
                self._boom()
            return teardown()

        loaded.update(step=wrapped_step, clear=wrapped_clear,
                      probe=wrapped_probe, teardown=wrapped_teardown)
        return loaded


R19_ROWS = 20


def run_broken_child(tmp_path, deps, *, rows=R19_ROWS, run="b1", **kw):
    """Drive the real child to a failure and hand back what it left."""
    out = init(tmp_path)
    d = out["paths"]["dir"]
    with pytest.raises(BaseException) as caught:  # noqa: PT011 - any of them
        lr.run_child(experiment_dir=d, run=run, rows=rows, nonce="n",
                     plan_digest=out["plan"]["plan_digest"], deps=deps, **kw)
    return {"out": out, "dir": d, "raised": caught.value,
            "evidence": d / f"{run}.failure.json",
            "report": d / f"{run}.json"}


def evidence_of(res) -> dict:
    assert res["evidence"].exists(), (
        "the child died and left no reason: an exit code is not evidence")
    return json.loads(res["evidence"].read_text())


# --- 1. every stage the child can die in leaves evidence --------------------


def test_r19_a_failed_source_check_leaves_evidence(tmp_path):
    """The snapshot no longer matches the tree, so the child refuses."""
    out = init(tmp_path)
    d = out["paths"]["dir"]
    snap = next(iter(sorted((d / "source_snapshot").glob("*.py"))))
    snap.write_text(snap.read_text() + "\n# tampered\n")
    with pytest.raises(SystemExit):
        lr.run_child(experiment_dir=d, run="b1", rows=R19_ROWS, nonce="n",
                     plan_digest=out["plan"]["plan_digest"],
                     deps=lr.FakeChildDeps())
    body = json.loads((d / "b1.failure.json").read_text())
    assert body["stage"] == "source_check"
    assert body["exception_type"] == "SystemExit"
    assert body["summary"]
    assert not (d / "b1.json").exists()


def test_r19_a_failed_child_preflight_leaves_evidence(tmp_path):
    def broken_preflight():
        raise OSError("vm_stat is not on this machine")

    res = run_broken_child(tmp_path, lr.FakeChildDeps(),
                           preflight=broken_preflight)
    body = evidence_of(res)
    assert body["stage"] == "preflight"
    assert body["exception_type"] == "OSError"
    assert not res["report"].exists()


def test_r19_a_failed_model_load_leaves_evidence(tmp_path):
    deps = BreakingDeps(break_at="load", exc_type=OSError,
                        message="the pinned adapter is not in this cache")
    body = evidence_of(run_broken_child(tmp_path, deps))
    assert body["stage"] == "model_load"
    assert body["exception_type"] == "OSError"
    assert deps.teardown_calls == 0, "nothing was loaded, nothing to tear down"


def test_r19_a_failed_training_step_leaves_evidence_and_tears_down(tmp_path):
    deps = BreakingDeps(break_at="step", at_row=7)
    res = run_broken_child(tmp_path, deps)
    body = evidence_of(res)
    assert body["stage"] == "training"
    assert body["exception_type"] == "RuntimeError"
    assert deps.rows_stepped == 7
    assert deps.teardown_calls == 1, (
        "a model was on the device when the row failed; it has to be released")
    assert not res["report"].exists()


def test_r19_a_failed_scheduled_clear_leaves_evidence(tmp_path):
    deps = BreakingDeps(break_at="clear")
    body = evidence_of(run_broken_child(tmp_path, deps))
    assert body["stage"] == "training"
    assert deps.rows_stepped == lr.EMPTY_CACHE_EVERY
    assert deps.teardown_calls == 1


def test_r19_a_failed_memory_probe_leaves_evidence(tmp_path):
    deps = BreakingDeps(break_at="probe")
    body = evidence_of(run_broken_child(tmp_path, deps))
    assert body["stage"] == "training"
    assert deps.teardown_calls == 1


def test_r19_a_failed_progress_write_leaves_evidence(tmp_path, monkeypatch):
    def broken_progress(*a, **kw):
        raise OSError("no space left on device")

    monkeypatch.setattr(lr, "write_progress", broken_progress)
    deps = BreakingDeps()
    body = evidence_of(run_broken_child(tmp_path, deps))
    assert body["stage"] == "training"
    assert body["exception_type"] == "OSError"
    assert deps.teardown_calls == 1


def test_r19_a_failed_teardown_leaves_evidence(tmp_path):
    deps = BreakingDeps(break_at="teardown")
    res = run_broken_child(tmp_path, deps)
    body = evidence_of(res)
    assert body["stage"] == "teardown"
    assert not res["report"].exists(), (
        "a run whose teardown failed did not finish; it has no report")


def test_r19_a_failed_report_write_leaves_evidence(tmp_path, monkeypatch):
    real = lr.write_once_json

    def selective(path, obj):
        if Path(path).name == "b1.json":
            raise OSError("the reports volume went read-only")
        return real(path, obj)

    monkeypatch.setattr(lr, "write_once_json", selective)
    deps = BreakingDeps()
    res = run_broken_child(tmp_path, deps)
    body = evidence_of(res)
    assert body["stage"] == "report_write"
    assert body["exception_type"] == "OSError"
    assert deps.teardown_calls == 1, "the teardown had already run"
    assert not res["report"].exists()


def test_r19_the_stages_the_child_publishes_are_all_declared():
    for stage in ("source_check", "preflight", "model_load", "training",
                  "teardown", "report_write"):
        assert stage in lr.FAILURE_STAGES


# --- 2. cleanup must not overwrite, and must not mask -----------------------


def test_r19_the_first_reason_wins_and_is_not_overwritten(tmp_path):
    """Training failed; the best-effort teardown then failed as well."""
    deps = BreakingDeps(break_at="step", at_row=3, teardown_exc=MemoryError)
    res = run_broken_child(tmp_path, deps)
    body = evidence_of(res)
    assert body["stage"] == "training", (
        "the teardown's own failure must not replace the reason the run died")
    assert body["exception_type"] == "RuntimeError"
    assert len(list(res["dir"].glob("b1.failure*.json"))) == 1


def test_r19_a_broken_teardown_does_not_mask_the_training_failure(tmp_path):
    deps = BreakingDeps(break_at="step", at_row=3, teardown_exc=MemoryError)
    res = run_broken_child(tmp_path, deps)
    assert isinstance(res["raised"], RuntimeError), (
        f"the caller was told {type(res['raised']).__name__}, which is the "
        "cleanup's failure, not the run's")


def test_r19_a_broken_evidence_writer_does_not_mask_the_original(tmp_path,
                                                                monkeypatch):
    def broken(*a, **kw):
        raise OSError("the evidence could not be written either")

    monkeypatch.setattr(lr, "write_failure_evidence", broken)
    res = run_broken_child(tmp_path, BreakingDeps(break_at="step", at_row=2))
    assert isinstance(res["raised"], RuntimeError)
    assert not res["evidence"].exists()


def test_r19_child_evidence_is_redacted_before_it_reaches_disk(tmp_path):
    dirty = (f"cannot reach {Path.home()}/.cache/huggingface for "
             "someone@example.com with Bearer hf_AAAABBBBCCCCDDDD")
    res = run_broken_child(tmp_path, BreakingDeps(break_at="load",
                                                  exc_type=OSError,
                                                  message=dirty))
    blob = res["evidence"].read_text()
    assert str(Path.home()) not in blob
    assert "someone@example.com" not in blob
    assert "hf_AAAABBBBCCCCDDDD" not in blob
    assert "OSError" in blob


# --- 3. what the replay does with the field ---------------------------------


def place_failed_run(tmp_path, *, outcome="nonzero_exit", stage="model_load",
                     evidence=True, mutate=None, run="b1"):
    """A finished event of the *new* shape, plus the evidence it points at."""
    out = init(tmp_path)
    paths = out["paths"]
    if evidence:
        lr.write_failure_evidence(paths, run, experiment_id=paths["dir"].name,
                                  stage=stage, exception_type="OSError",
                                  summary="the pinned tokenizer is not cached")
    place_run(paths, out["plan"], run, 500, boot="boot-0", outcome=outcome,
              terminal=False)
    if mutate is not None:
        mutate(paths, run)
    return out


def resign_evidence(paths, run):
    """Recompute the evidence digest and re-chain the journal around it.

    Without this a semantic test proves only that the digest check works.
    """
    digest = hashlib.sha256(
        (paths["dir"] / f"{run}.failure.json").read_bytes()).hexdigest()
    rewrite_chained(paths, f"{run}-{lr.EVENT_FINISHED}",
                    lambda b: b.update({"failure_evidence_sha256": digest}))


def edit_evidence(paths, run, mutate):
    path = paths["dir"] / f"{run}.failure.json"
    body = json.loads(path.read_text())
    mutate(body)
    path.write_text(json.dumps(body, indent=2) + "\n")
    resign_evidence(paths, run)


def test_r19_a_finished_event_without_the_field_is_legacy_and_replays(tmp_path):
    """exp001's b1 predates the field entirely. It must still verify."""
    out = place_failed_run(tmp_path, evidence=False)
    rewrite_finished(out["paths"], "b1",
                     lambda b: b.pop("failure_evidence_sha256", None))
    body = finished_body_of(out["paths"], "b1")
    assert "failure_evidence_sha256" not in body
    assert verify(tmp_path) == [], verify(tmp_path)


def test_r19_a_null_field_is_not_legacy_for_a_nonzero_exit(tmp_path):
    """The field is present and null: this writer knew about it and had none."""
    out = place_failed_run(tmp_path, evidence=False)
    body = finished_body_of(out["paths"], "b1")
    assert "failure_evidence_sha256" in body
    assert body["failure_evidence_sha256"] is None
    problems = verify(tmp_path)
    assert any("failure evidence" in p for p in problems), problems


@pytest.mark.parametrize("outcome", ["timed_out", "no_report"])
def test_r19_other_outcomes_may_legitimately_have_no_evidence(tmp_path, outcome):
    """A child killed by SIGKILL never got to publish anything."""
    place_failed_run(tmp_path, outcome=outcome, evidence=False)
    assert verify(tmp_path) == [], verify(tmp_path)


def test_r19_a_nonzero_exit_with_matching_evidence_replays(tmp_path):
    place_failed_run(tmp_path)
    assert verify(tmp_path) == [], verify(tmp_path)


@pytest.mark.parametrize("mutate,expect", [
    (lambda b: b.update({"schema_version": 2}), "schema"),
    (lambda b: b.update({"kind": "something_else"}), "failure evidence"),
    (lambda b: b.update({"experiment_id": "exp999"}), "experiment"),
    (lambda b: b.update({"run_id": "b2"}), "run"),
    (lambda b: b.update({"stage": "vibes"}), "stage"),
    (lambda b: b.update({"exception_type": ""}), "exception_type"),
    (lambda b: b.update({"summary": "   "}), "summary"),
    (lambda b: b.pop("written_at", None), "written_at"),
    (lambda b: b.update({"written_at": "last tuesday"}), "written_at"),
    (lambda b: b.update({"note": "added later"}), "unexpected"),
])
def test_r19_resigned_evidence_is_still_refused_on_its_meaning(tmp_path,
                                                               mutate, expect):
    """Digest and journal both recomputed. Only the meaning is wrong."""
    out = place_failed_run(tmp_path)
    edit_evidence(out["paths"], "b1", mutate)
    # The chain itself is intact: this is not a tamper-detection test.
    assert lr.session_state("exp016a", root=sess(tmp_path))["events"]
    problems = verify(tmp_path)
    assert any(expect in p for p in problems), (expect, problems)


def test_r19_the_evidence_schema_is_closed():
    assert set(lr.FAILURE_EVIDENCE_FIELDS) == {
        "schema_version", "kind", "experiment_id", "run_id", "stage",
        "exception_type", "summary", "written_at"}


# --- 4. the launcher's environment, observed rather than read --------------


def test_r19_spawn_child_overrides_the_inherited_offline_variables(tmp_path,
                                                                  monkeypatch):
    """Intercept Popen. Reading the source proves only that a name appears."""
    import subprocess

    mod = load_cli()
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setenv("BRICKAGAIN_MARKER", "inherited")

    seen = {}

    class FakePopen:
        def __init__(self, argv, **kw):
            seen["argv"] = argv
            seen["kw"] = kw
            self.pid = 4321

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", FakePopen)

    out = init(tmp_path)
    launcher = mod.Launcher(lr.SafetySpec())
    got = launcher.spawn_child(run="b1", paths=out["paths"],
                               entry=out["plan"]["runs"][0], nonce="n",
                               plan=out["plan"])
    assert got["spawned"] is True
    env = seen["kw"]["env"]
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert env["BRICKAGAIN_MARKER"] == "inherited", (
        "the rest of the environment still has to reach the child")
    assert seen["kw"]["start_new_session"] is True


# ===========================================================================
# Round 20. Codex's re-review of round 19.
#
# Round 19 wrapped every stage the child *runs*, and left the seam where the
# loader hands its work over unguarded: `run_child` reached straight into the
# returned mapping. A loader that returns None, or a dict missing `order`, or
# a `step` that is not callable, still died with an unpublished AttributeError
# or KeyError -- and with the model it had just built still on the device.
#
# Two more things round 19 got half-right: the redactor only knew token shapes
# without hyphens or underscores and only knew `/Users` and `/home`, and the
# replay accepted a *missing* digest field even when a failure file was
# sitting right next to it.
# ===========================================================================


# --- 1. the loader's contract, enforced where it is handed over ------------


class ContractDeps(lr.FakeChildDeps):
    """A loader that returns something `run_child` cannot use."""

    def __init__(self, *, mutate=None, replace=None, **kw):
        super().__init__(**kw)
        self.mutate = mutate
        self.replace = replace
        self.teardown_calls = 0

    def load(self, *, rows: int):
        loaded = dict(super().load(rows=rows))
        inner = loaded["teardown"]

        def counted_teardown():
            self.teardown_calls += 1
            return inner()

        loaded["teardown"] = counted_teardown
        if self.replace is not None:
            return self.replace(loaded)
        if self.mutate is not None:
            self.mutate(loaded)
        return loaded


def run_contract_child(tmp_path, deps, *, rows=R19_ROWS, run="b1"):
    out = init(tmp_path)
    d = out["paths"]["dir"]
    with pytest.raises(BaseException) as caught:  # noqa: PT011 - any of them
        lr.run_child(experiment_dir=d, run=run, rows=rows, nonce="n",
                     plan_digest=out["plan"]["plan_digest"], deps=deps)
    ev = d / f"{run}.failure.json"
    assert ev.exists(), "the loader broke the contract and left no reason"
    return {"dir": d, "raised": caught.value,
            "body": json.loads(ev.read_text()),
            "report": d / f"{run}.json"}


@pytest.mark.parametrize("replacement", [None, [], "loaded", 42])
def test_r20_a_loader_that_returns_no_mapping_is_refused(tmp_path,
                                                         replacement):
    deps = ContractDeps(replace=lambda _: replacement)
    res = run_contract_child(tmp_path, deps)
    assert res["body"]["stage"] == "dependency"
    assert not res["report"].exists()
    assert deps.teardown_calls == 0, (
        "there was no teardown to obtain, so none may be invented")


@pytest.mark.parametrize("field", ["order", "step", "sample_ids",
                                   "provenance", "model_load_seconds"])
def test_r20_a_loader_missing_a_required_field_is_refused(tmp_path, field):
    deps = ContractDeps(mutate=lambda d: d.pop(field))
    res = run_contract_child(tmp_path, deps)
    assert res["body"]["stage"] == "dependency"
    assert field in res["body"]["summary"]
    assert not res["report"].exists()
    assert deps.teardown_calls == 1, (
        "a model had already been built; it has to be released exactly once")


@pytest.mark.parametrize("field", ["step", "teardown", "clear", "probe"])
def test_r20_a_loader_field_that_is_not_callable_is_refused(tmp_path, field):
    deps = ContractDeps(mutate=lambda d: d.update({field: "not a function"}))
    res = run_contract_child(tmp_path, deps)
    assert res["body"]["stage"] == "dependency"
    assert field in res["body"]["summary"]
    # A broken teardown is not a teardown: it must not be called, and the
    # other three cases still have a real one that must be.
    assert deps.teardown_calls == (0 if field == "teardown" else 1)


def test_r20_a_short_order_is_refused_before_a_single_row_is_measured(tmp_path):
    deps = ContractDeps(mutate=lambda d: d.update({"order": d["order"][:3]}))
    res = run_contract_child(tmp_path, deps, rows=R19_ROWS)
    assert res["body"]["stage"] == "dependency"
    assert "3" in res["body"]["summary"] and str(R19_ROWS) in res["body"]["summary"]
    assert deps.teardown_calls == 1
    assert not (res["dir"] / "b1.progress.jsonl").exists()


def test_r20_a_contract_failure_does_not_mask_itself_with_cleanup(tmp_path):
    """The teardown fails too. The caller still hears about the contract."""
    def sabotage(loaded):
        loaded.pop("order")
        loaded["teardown"] = lambda: (_ for _ in ()).throw(
            MemoryError("the teardown clear also failed"))

    res = run_contract_child(tmp_path, ContractDeps(mutate=sabotage))
    assert not isinstance(res["raised"], MemoryError)
    assert res["body"]["stage"] == "dependency"


def test_r20_a_loader_that_honours_the_contract_still_runs(tmp_path):
    """The guard must not refuse the shape production actually returns."""
    out = init(tmp_path)
    d = out["paths"]["dir"]
    rc = lr.run_child(experiment_dir=d, run="b1", rows=R19_ROWS, nonce="n",
                      plan_digest=out["plan"]["plan_digest"],
                      deps=lr.FakeChildDeps(pool_rows=lr.POOL_ROWS))
    assert rc == 0
    assert not (d / "b1.failure.json").exists()
    assert (d / "b1.json").exists()


# --- 2. redaction: modern credential shapes and other people's paths -------
#
# Every string below is synthetic: fixed literals with no entropy, invented
# hostnames, and key bodies that are visibly typed-out patterns. Nothing here
# is, or was ever, a real credential.

SYNTHETIC_SECRETS = [
    ("sk-proj-AAAABBBBCCCCDDDD_EEEE-FFFF", "an OpenAI project key"),
    ("sk-ant-api03-AAAABBBB-CCCCDDDD_EEEE", "an Anthropic key"),
    ("hf_AAAABBBBCCCCDDDDEEEE", "a hub token"),
    ("github_pat_11AAAABBBB_CCCCDDDDEEEEFFFF", "a fine-grained PAT"),
    ("ghp_AAAABBBBCCCCDDDDEEEE", "a classic PAT"),
    ("xoxb-0000-0000-AAAABBBBCCCC", "a Slack bot token"),
    ("AKIAAAAABBBBCCCCDDDD", "an AWS access key id"),
    ("org-AAAABBBBCCCCDDDD", "an organization id"),
    ("nobody@example.invalid", "an email address"),
]

SYNTHETIC_KEYWORD_LINES = [
    "api_key=AAAABBBBCCCCDDDD",
    "API-KEY: AAAABBBBCCCCDDDD",
    "password=hunter2placeholder",
    "client_secret: AAAABBBBCCCCDDDD",
    "access_token AAAABBBBCCCCDDDD",
    "Authorization: Bearer AAAABBBBCCCCDDDD",
    "private_key=AAAABBBBCCCCDDDD",
]

SYNTHETIC_PATHS = [
    "/Users/someone/.cache/huggingface/token",
    "/home/someone/.config/creds.json",
    "/root/.netrc",
    "/private/var/folders/ab/cdefgh/T/tmpxyz/model.bin",
    "/Volumes/BackupDrive/models/llama/config.json",
    "/var/folders/ab/cdefgh/T/tmp1234",
    "/mnt/nas/datasets/instruct.jsonl",
    "/media/usb0/checkpoints/adapter.safetensors",
    "C:\\Users\\someone\\AppData\\Local\\hf\\token",
]


@pytest.mark.parametrize("secret,what", SYNTHETIC_SECRETS)
def test_r20_the_redactor_removes_modern_credential_shapes(secret, what):
    clean = lr._portable(f"OSError: refused while presenting {secret} to hub")
    assert secret not in clean, f"{what} survived redaction"
    assert "OSError" in clean, "the shape of the failure has to survive"


@pytest.mark.parametrize("line", SYNTHETIC_KEYWORD_LINES)
def test_r20_the_redactor_removes_generic_credential_keywords(line):
    clean = lr._portable(f"ValueError: config rejected ({line})")
    assert "AAAABBBBCCCCDDDD" not in clean
    assert "hunter2placeholder" not in clean
    assert "ValueError" in clean


@pytest.mark.parametrize("path", SYNTHETIC_PATHS)
def test_r20_the_redactor_removes_other_peoples_absolute_paths(path):
    clean = lr._portable(f"FileNotFoundError: {path} is missing")
    assert path not in clean
    for marker in ("/Users/", "/home/", "/root/", "/private/", "/Volumes/",
                   "/var/folders/", "/mnt/", "/media/", "C:\\Users\\"):
        assert marker not in clean, f"{marker} survived in {clean!r}"
    assert "FileNotFoundError" in clean


def test_r20_the_redactor_keeps_the_repository_and_home_markers():
    clean = lr._portable(f"OSError: {ROOT}/src/training/lora.py and "
                         f"{Path.home()}/.cache/huggingface")
    assert str(ROOT) not in clean and str(Path.home()) not in clean
    assert "<repo>" in clean and "<home>" in clean
    assert "src/training/lora.py" in clean, (
        "a repository-relative path is the useful part and must survive")


def test_r20_ordinary_words_that_merely_contain_a_keyword_survive():
    clean = lr._portable("OSError: the tokenizer could not be resolved "
                         "offline; passwordless auth is unrelated")
    assert "tokenizer" in clean
    assert "OSError" in clean


# --- 3. the replay refuses evidence that leaks, whoever signed it ----------


LEAKY_BODIES = [
    (lambda b: b.update({"summary": "failed with hf_AAAABBBBCCCCDDDDEEEE"}),
     "credential"),
    (lambda b: b.update({"summary": "mail nobody@example.invalid for help"}),
     "credential"),
    (lambda b: b.update({"summary": "billed to org-AAAABBBBCCCCDDDD"}),
     "credential"),
    (lambda b: b.update({"summary": "missing /Users/someone/.cache/hf"}),
     "absolute path"),
    (lambda b: b.update({"summary": "missing /Volumes/Backup/model.bin"}),
     "absolute path"),
    (lambda b: b.update({"exception_type": "OSError_/private/var/x"}),
     "absolute path"),
]


@pytest.mark.parametrize("mutate,expect", LEAKY_BODIES)
def test_r20_resigned_evidence_that_leaks_is_still_refused(tmp_path, mutate,
                                                           expect):
    """Re-written, re-hashed, re-chained -- and still refused for leaking."""
    out = place_failed_run(tmp_path)
    edit_evidence(out["paths"], "b1", mutate)
    problems = verify(tmp_path)
    assert any(expect in p for p in problems), (expect, problems)


def test_r20_evidence_the_child_itself_wrote_passes_the_leak_check(tmp_path):
    """The redactor's output must satisfy the replay's own standard."""
    dirty = (f"cannot reach {Path.home()}/.cache/hf as nobody@example.invalid "
             "with Bearer hf_AAAABBBBCCCCDDDDEEEE from /Volumes/Backup")
    res = run_broken_child(tmp_path, BreakingDeps(break_at="load",
                                                  exc_type=OSError,
                                                  message=dirty))
    body = json.loads(res["evidence"].read_text())
    assert lr.leaked_identifiers(json.dumps(body)) == []


# --- 4. the two remaining replay holes ------------------------------------


def test_r20_no_digest_field_and_no_failure_file_stays_compatible(tmp_path):
    """exp001's shape exactly. Nothing to reconcile, nothing to complain of."""
    out = place_failed_run(tmp_path, evidence=False)
    rewrite_finished(out["paths"], "b1",
                     lambda b: b.pop("failure_evidence_sha256", None))
    assert not (out["paths"]["dir"] / "b1.failure.json").exists()
    assert verify(tmp_path) == [], verify(tmp_path)


def test_r20_no_digest_field_but_a_failure_file_is_refused(tmp_path):
    """A file nobody pinned is a file anybody could have dropped there."""
    out = place_failed_run(tmp_path)
    rewrite_finished(out["paths"], "b1",
                     lambda b: b.pop("failure_evidence_sha256", None))
    assert (out["paths"]["dir"] / "b1.failure.json").exists()
    problems = verify(tmp_path)
    assert any("failure evidence" in p for p in problems), problems


@pytest.mark.parametrize("stamp", [
    "2031-02-03",                       # date only
    "2031-02-03T04:05:06",              # naive
    "2031-02-03T04:05:06.789012",       # naive with microseconds
    "20310203T040506Z",                 # not ISO-8601 extended
])
def test_r20_written_at_must_carry_a_time_and_a_zone(tmp_path, stamp):
    out = place_failed_run(tmp_path)
    edit_evidence(out["paths"], "b1", lambda b: b.update({"written_at": stamp}))
    problems = verify(tmp_path)
    assert any("written_at" in p for p in problems), (stamp, problems)


@pytest.mark.parametrize("stamp", [
    "2031-02-03T04:05:06.789012+00:00",
    "2031-02-03T04:05:06+00:00",
    "2031-02-03T00:00:00+08:00",
])
def test_r20_a_real_aware_timestamp_is_accepted(tmp_path, stamp):
    out = place_failed_run(tmp_path)
    edit_evidence(out["paths"], "b1", lambda b: b.update({"written_at": stamp}))
    assert verify(tmp_path) == [], (stamp, verify(tmp_path))


def test_r20_what_the_child_writes_satisfies_the_timestamp_rule():
    from datetime import datetime as _dt
    stamp = lr.now_iso()
    assert lr._is_iso_timestamp(stamp)
    assert _dt.fromisoformat(stamp).tzinfo is not None


# --- 5. the loader pins offline before it imports anything ----------------


def test_r20_load_pins_offline_before_its_very_first_import(monkeypatch):
    """Observed, not read: the first import sees the pinned environment.

    Round 19 measured that `local_files_only=True` does not cover
    `peft.load_peft_weights` -> `huggingface_hub.file_exists`, and that
    `huggingface_hub` freezes HF_HUB_OFFLINE at *its* import time. So the
    pinning has to happen before `load` imports anything at all -- which
    `inspect.getsource` could never establish.
    """
    import builtins

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
    # This test is about import *ordering*. Round 21 added a second guard on
    # the same line -- refuse if the hub was already imported online -- and in
    # a pytest process that has imported transformers it fires first and hides
    # what is under test here. It has its own tests, in a fresh subprocess
    # where the condition can be arranged honestly; switched off for this one.
    monkeypatch.setattr(lr, "OFFLINE_FROZEN_MODULES", ())

    seen = []
    real_import = builtins.__import__

    def spy(name, *a, **kw):
        if not seen:
            seen.append({k: os.environ.get(k)
                         for k in lr.PRODUCTION_OFFLINE_ENV})
        if name == "torch":
            raise ImportError("stopped here: nothing may actually load")
        return real_import(name, *a, **kw)

    deps = lr.ProductionChildDeps()
    monkeypatch.setattr(builtins, "__import__", spy)
    try:
        with pytest.raises(ImportError):
            deps.load(rows=1)
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)

    assert seen, "load() imported nothing, so the ordering was not observed"
    assert seen[0] == {k: "1" for k in lr.PRODUCTION_OFFLINE_ENV}, (
        f"the first import saw {seen[0]}, so the environment was pinned "
        "too late for huggingface_hub to read it")


# ===========================================================================
# Round 21. Codex's re-review of round 20.
#
# Round 20 checked the loader's contract at the seam, and checked it shallowly:
# a required field only had to be *present*, so `step=None` sailed through and
# died at row one; `sample_ids`, `provenance` and `model_load_seconds` were
# never looked at beyond their keys; and an `order` of the right length made
# of the wrong things was accepted.
#
# The replay's leak check had the same shape of hole: `password=...` in a
# re-signed file was refused by nothing, because the assignment rule lived in
# the writer and was never shared with the replay.
#
# And "calling load() directly is safe now" was stated without the condition
# that makes it true: it is safe only when load() performs the first import.
# ===========================================================================


# --- 1. the loader's contract, checked for meaning and not just for keys ---


def contract_body(tmp_path, mutate=None, replace=None, *, rows=R19_ROWS):
    deps = ContractDeps(mutate=mutate, replace=replace)
    res = run_contract_child(tmp_path, deps, rows=rows)
    assert res["body"]["stage"] == "dependency"
    assert not res["report"].exists(), "a refused loader writes no report"
    assert not (res["dir"] / "b1.progress.jsonl").exists(), (
        "the contract is checked before the first row, so no progress line "
        "can exist"
    )
    return res, deps


def test_r21_a_step_of_none_is_refused_before_the_first_row(tmp_path):
    """Present but useless. Round 20 only asked whether the key was there."""
    res, deps = contract_body(tmp_path, lambda d: d.update({"step": None}))
    assert "step" in res["body"]["summary"]
    assert deps.teardown_calls == 1


@pytest.mark.parametrize("field", ["order", "sample_ids", "provenance",
                                   "model_load_seconds"])
def test_r21_a_required_field_of_none_is_refused(tmp_path, field):
    res, deps = contract_body(tmp_path, lambda d: d.update({field: None}))
    assert field in res["body"]["summary"]
    assert deps.teardown_calls == 1


@pytest.mark.parametrize("value", [42, "s0s1s2", {"a": 1}, object()])
def test_r21_sample_ids_that_are_not_a_sequence_are_refused(tmp_path, value):
    res, deps = contract_body(tmp_path,
                              lambda d: d.update({"sample_ids": value}))
    assert "sample_ids" in res["body"]["summary"]
    assert deps.teardown_calls == 1


@pytest.mark.parametrize("cut", [0, 1, R19_ROWS - 1, R19_ROWS + 1])
def test_r21_sample_ids_of_the_wrong_length_are_refused(tmp_path, cut):
    def mutate(d):
        ids = list(d["sample_ids"])
        d["sample_ids"] = (ids + ["extra"]) if cut > R19_ROWS else ids[:cut]

    res, deps = contract_body(tmp_path, mutate)
    assert "sample_ids" in res["body"]["summary"]
    assert deps.teardown_calls == 1


@pytest.mark.parametrize("value", [[], "provenance", 7, ("a", "b")])
def test_r21_a_provenance_that_is_not_a_mapping_is_refused(tmp_path, value):
    res, deps = contract_body(tmp_path,
                              lambda d: d.update({"provenance": value}))
    assert "provenance" in res["body"]["summary"]
    assert deps.teardown_calls == 1


@pytest.mark.parametrize("value,why", [
    ("0.0", "a string"),
    (True, "a bool"),
    (False, "a bool"),
    (float("nan"), "not a number"),
    (float("inf"), "infinite"),
    (float("-inf"), "infinite"),
    (-1.0, "negative"),
    (-0.001, "negative"),
])
def test_r21_an_unusable_model_load_seconds_is_refused(tmp_path, value, why):
    res, deps = contract_body(
        tmp_path, lambda d: d.update({"model_load_seconds": value}))
    assert "model_load_seconds" in res["body"]["summary"], why
    assert deps.teardown_calls == 1


@pytest.mark.parametrize("value", [0, 0.0, 1, 2.5])
def test_r21_a_legitimate_model_load_seconds_is_accepted(tmp_path, value):
    """Zero is legitimate: a fake loads nothing and takes no time."""
    out = init(tmp_path)
    d = out["paths"]["dir"]
    deps = ContractDeps(mutate=lambda x: x.update({"model_load_seconds": value}))
    rc = lr.run_child(experiment_dir=d, run="b1", rows=R19_ROWS, nonce="n",
                      plan_digest=out["plan"]["plan_digest"], deps=deps)
    assert rc == 0
    assert not (d / "b1.failure.json").exists()


@pytest.mark.parametrize("element", [None, "3", 1.5, True, [0], object()])
def test_r21_an_order_of_the_right_length_but_wrong_shape_is_refused(
        tmp_path, element):
    """`len(order) == rows` was the only thing round 20 asked of it."""
    res, deps = contract_body(
        tmp_path, lambda d: d.update({"order": [element] * R19_ROWS}))
    assert "order" in res["body"]["summary"]
    assert deps.teardown_calls == 1


@pytest.mark.parametrize("value", [42, "abcdefghijklmnopqrst", {"a": 1}])
def test_r21_an_order_that_is_not_a_sequence_is_refused(tmp_path, value):
    res, deps = contract_body(tmp_path, lambda d: d.update({"order": value}))
    assert "order" in res["body"]["summary"]
    assert deps.teardown_calls == 1


def test_r21_a_negative_order_index_is_refused(tmp_path):
    """Python would happily index from the end and measure the wrong rows."""
    res, deps = contract_body(
        tmp_path, lambda d: d.update({"order": [-1] * R19_ROWS}))
    assert "order" in res["body"]["summary"]
    assert deps.teardown_calls == 1


@pytest.mark.parametrize("field,value", [
    ("step", None),
    ("sample_ids", None),
    ("provenance", []),
    ("model_load_seconds", float("nan")),
    ("order", [None] * R19_ROWS),
])
def test_r21_a_failing_teardown_does_not_mask_the_contract_error(
        tmp_path, field, value):
    def sabotage(d):
        d[field] = value
        d["teardown"] = lambda: (_ for _ in ()).throw(
            MemoryError("the teardown clear also failed"))

    res = run_contract_child(tmp_path, ContractDeps(mutate=sabotage))
    assert not isinstance(res["raised"], MemoryError)
    assert isinstance(res["raised"], ValueError)
    assert res["body"]["stage"] == "dependency"


def test_r21_the_production_loader_satisfies_the_contract_it_is_held_to():
    """The check must describe production, not an idea of it.

    Driven against the shape `ProductionChildDeps.load` documents, not against
    a live model: the point is that the validator would not reject the real
    thing.
    """
    real = {"order": list(range(500)), "step": lambda i, p: {},
            "sample_ids": [f"s{i}" for i in range(500)],
            # Round 21 wrote `{"device": "mps"}` here, which is not what
            # production returns and became a false negative the moment round
            # 22 gave provenance a real contract. A stand-in for production
            # has to satisfy what production satisfies.
            "provenance": fake_provenance(500), "model_load_seconds": 0.68,
            "teardown": lambda: 0.0, "clear": lambda: 0.0,
            "probe": lambda: {}}
    assert lr.child_load_problems(real, rows=500) == []


# --- 2. the replay's own credential rule -----------------------------------


R21_ASSIGNMENTS = [
    "password=hunter2placeholder",
    "password: hunter2placeholder",
    "api_key=AAAABBBBCCCCDDDD",
    "api_key: AAAABBBBCCCCDDDD",
    "API-KEY = AAAABBBBCCCCDDDD",
    "client_secret=AAAABBBBCCCCDDDD",
    "access_token=AAAABBBBCCCCDDDD",
    "refresh_token: AAAABBBBCCCCDDDD",
    "private_key=AAAABBBBCCCCDDDD",
    "authorization=AAAABBBBCCCCDDDD",
]

#: Wording that names a credential without disclosing one. Refusing these
#: would refuse honest evidence, which is the failure mode that matters more:
#: a replay nobody can satisfy stops being read.
R21_INNOCENT = [
    "passwordless authentication is not configured",
    "the tokenizer could not be resolved offline",
    "password was not configured for this repository",
    "no api key is required for a local load",
    "OSError: the secret store is unavailable",
    "authorization failed",
]


@pytest.mark.parametrize("line", R21_ASSIGNMENTS)
def test_r21_a_credential_assignment_is_a_leak(line):
    assert "credential" in lr.leaked_identifiers(f"ValueError: {line}")


@pytest.mark.parametrize("line", R21_INNOCENT)
def test_r21_naming_a_credential_without_one_is_not_a_leak(line):
    assert lr.leaked_identifiers(f"OSError: {line}") == [], line


@pytest.mark.parametrize("line", ["password=", "api_key:", "token =   "])
def test_r21_a_credential_key_with_no_value_is_not_a_leak(line):
    """An empty assignment discloses nothing, and a truncated traceback
    ending on one is ordinary."""
    assert lr.leaked_identifiers(f"OSError: {line}") == [], line


@pytest.mark.parametrize("line", R21_ASSIGNMENTS)
def test_r21_resigned_evidence_with_a_credential_assignment_is_refused(
        tmp_path, line):
    out = place_failed_run(tmp_path)
    edit_evidence(out["paths"], "b1",
                  lambda b: b.update({"summary": f"OSError: {line}"}))
    problems = verify(tmp_path)
    assert any("credential" in p for p in problems), (line, problems)


@pytest.mark.parametrize("line", R21_INNOCENT)
def test_r21_resigned_innocent_evidence_still_replays(tmp_path, line):
    out = place_failed_run(tmp_path)
    edit_evidence(out["paths"], "b1",
                  lambda b: b.update({"summary": f"OSError: {line}"}))
    assert verify(tmp_path) == [], (line, verify(tmp_path))


@pytest.mark.parametrize(
    "dirty",
    R21_ASSIGNMENTS + [s for s, _ in SYNTHETIC_SECRETS] + SYNTHETIC_PATHS)
def test_r21_whatever_the_writer_emits_satisfies_the_replay(dirty):
    """One definition, driven from both ends.

    The writer redacting less than the replay refuses is a child that cannot
    publish its own failure; the replay refusing less than the writer removes
    is a leak nobody catches. Both directions are checked here.
    """
    clean = lr._portable(f"OSError: load failed with {dirty}")
    assert lr.leaked_identifiers(clean) == [], (dirty, clean)
    assert "OSError" in clean


def test_r21_the_writer_and_the_replay_share_one_leak_definition():
    """Not by name: every shared pattern is driven through both sides."""
    for pattern, label in lr._LEAK_CHECKS:
        sample = {
            "credential": "api_key=AAAABBBBCCCCDDDD",
            "absolute path": "/Volumes/Backup/model.bin",
        }[label]
        if not pattern.search(sample):
            continue
        assert lr.leaked_identifiers(sample) != []
        assert lr.leaked_identifiers(lr._portable(sample)) == []


# --- 3. the strict-offline boundary, when the flag was already frozen ------


R21_PRELOADED_PROBE = '''
import json, os, socket, sys
sys.path.insert(0, sys.argv[1])
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ.pop("HF_HUB_DISABLE_TELEMETRY", None)

# The caller's mistake, made before longrun is even imported: this freezes
# huggingface_hub's HF_HUB_OFFLINE as False for the life of the process.
import peft  # noqa: F401

import src.training.longrun as lr

attempts = []


def blocked(*a, **kw):
    attempts.append(1)
    raise OSError("network disabled by the probe")


socket.getaddrinfo = blocked
socket.create_connection = blocked
socket.socket.connect = lambda self, *a, **kw: blocked()

out = {"raised": None, "message": ""}
try:
    lr.ProductionChildDeps().load(rows=8)
except BaseException as exc:  # noqa: BLE001
    out["raised"] = type(exc).__name__
    out["message"] = str(exc)
out["network_attempts"] = len(attempts)
out["env_after"] = {k: os.environ.get(k) for k in lr.PRODUCTION_OFFLINE_ENV}
out["leaks"] = lr.leaked_identifiers(out["message"])
print("PROBE" + json.dumps(out))
'''


def run_preloaded_probe():
    out = subprocess.run([sys.executable, "-c", R21_PRELOADED_PROBE, str(ROOT)],
                         capture_output=True, text=True, cwd=ROOT, timeout=300)
    line = next((ln for ln in out.stdout.splitlines()
                 if ln.startswith("PROBE")), None)
    assert line, f"probe produced nothing:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}"
    return json.loads(line[len("PROBE"):])



def test_r21_a_preloaded_online_hub_is_refused_before_any_network_call():
    """A fresh process, hostile environment, peft imported first.

    Round 20 moved the pin ahead of `load`'s imports, which fixes the case
    where `load` performs the first import -- and cannot fix this one, because
    `huggingface_hub` reads HF_HUB_OFFLINE once and it has already read it.
    So the only honest thing left is to refuse, before the tokenizer, before
    the model and before the adapter.
    """
    got = run_preloaded_probe()
    assert got["raised"] == "OfflineNotGuaranteed", got
    assert got["network_attempts"] == 0, (
        "it must refuse before reaching for the hub, not after failing to")
    assert got["env_after"] == {"HF_HUB_OFFLINE": "1",
                                "TRANSFORMERS_OFFLINE": "1",
                                "HF_HUB_DISABLE_TELEMETRY": "1"}, (
        "the pin still has to happen first; it is just not sufficient here")
    assert "huggingface_hub" in got["message"]
    assert got["leaks"] == [], got["message"]
    for marker in ("/Users/", "/home/", "/private/", "/Volumes/"):
        assert marker not in got["message"]


def test_r21_the_refusal_is_a_named_type_the_child_can_publish():
    assert issubclass(lr.OfflineNotGuaranteed, RuntimeError)


def test_r21_a_clean_process_reports_no_frozen_offline_module():
    """This test process pins offline in conftest-free fashion, so the check
    must be driven explicitly rather than assumed."""
    problems = lr.offline_freeze_problems(
        modules={"huggingface_hub.constants": types.SimpleNamespace(
            HF_HUB_OFFLINE=True)})
    assert problems == []


@pytest.mark.parametrize("frozen", [False, 0, None, ""])
def test_r21_a_frozen_online_hub_is_detected(frozen):
    problems = lr.offline_freeze_problems(
        modules={"huggingface_hub.constants": types.SimpleNamespace(
            HF_HUB_OFFLINE=frozen)})
    assert problems and "huggingface_hub" in problems[0]


def test_r21_a_hub_that_was_never_imported_is_not_a_problem():
    assert lr.offline_freeze_problems(modules={}) == []


def test_r21_the_refusal_message_says_what_to_do_instead():
    problems = lr.offline_freeze_problems(
        modules={"huggingface_hub.constants": types.SimpleNamespace(
            HF_HUB_OFFLINE=False)})
    text = " ".join(problems)
    assert "HF_HUB_OFFLINE" in text
    assert lr.leaked_identifiers(text) == []


def test_r21_run_child_publishes_the_offline_refusal_as_evidence(tmp_path):
    """It arrives through `deps.load`, so it is a model_load failure."""
    class FrozenDeps(lr.ChildDeps):
        def load(self, *, rows):
            raise lr.OfflineNotGuaranteed(
                "huggingface_hub was imported before HF_HUB_OFFLINE was set")

    res = run_broken_child(tmp_path, FrozenDeps())
    body = json.loads(res["evidence"].read_text())
    assert body["stage"] == "model_load"
    assert body["exception_type"] == "OfflineNotGuaranteed"
    assert lr.leaked_identifiers(json.dumps(body)) == []


# ===========================================================================
# Round 22. Codex's re-review of round 21.
#
# Round 21 checked the *types* the loader hands back and stopped there. An
# `order` of the right length made of the right type can still name row 4
# twice, or name row 2000 of a 2000-row pool; `sample_ids` can be twenty
# copies of one id, which makes `input_order_digest` agree with an order that
# was never measured; and `provenance` only had to be a mapping -- it could be
# `{}`, and the check that `measurement_intervals.max_rows` matches the run
# lived in the replay alone, so the child could write a report the replay was
# always going to refuse.
# ===========================================================================


# --- 1. order: unique, and inside the pool --------------------------------


def test_r22_an_order_with_a_duplicate_index_is_refused(tmp_path):
    """Two rows of the pool measured once, one measured twice."""
    def mutate(d):
        order = list(d["order"])
        order[1] = order[0]
        d["order"] = order

    res, deps = contract_body(tmp_path, mutate)
    assert "order" in res["body"]["summary"]
    assert "duplicate" in res["body"]["summary"]
    assert deps.teardown_calls == 1


def test_r22_an_order_of_all_the_same_index_is_refused(tmp_path):
    res, deps = contract_body(tmp_path,
                              lambda d: d.update({"order": [7] * R19_ROWS}))
    assert "duplicate" in res["body"]["summary"]
    assert deps.teardown_calls == 1


@pytest.mark.parametrize("index", [lr.POOL_ROWS, lr.POOL_ROWS + 1,
                                   lr.POOL_ROWS * 2])
def test_r22_an_order_index_outside_the_pool_is_refused(tmp_path, index):
    """`POOL_ROWS` is one past the end. Production would raise IndexError at
    that row; a smaller out-of-range value would not, and would measure the
    wrong thing in silence."""
    def mutate(d):
        order = list(d["order"])
        order[-1] = index
        d["order"] = order

    res, deps = contract_body(tmp_path, mutate)
    assert "order" in res["body"]["summary"]
    assert str(lr.POOL_ROWS) in res["body"]["summary"]
    assert deps.teardown_calls == 1


def test_r22_the_last_valid_pool_index_is_accepted(tmp_path):
    """The boundary has to be exclusive on one side only."""
    out = init(tmp_path)
    d = out["paths"]["dir"]

    def mutate(loaded):
        order = list(loaded["order"])
        order[-1] = lr.POOL_ROWS - 1
        loaded["order"] = order
        loaded["sample_ids"] = [f"s{i}" for i in order]

    rc = lr.run_child(experiment_dir=d, run="b1", rows=R19_ROWS, nonce="n",
                      plan_digest=out["plan"]["plan_digest"],
                      deps=ContractDeps(mutate=mutate))
    assert rc == 0
    assert not (d / "b1.failure.json").exists()


# --- 2. sample_ids: unique, non-empty strings ------------------------------


@pytest.mark.parametrize("bad", [None, "", 42, 3.5, True, ("a",)])
def test_r22_a_sample_id_that_is_not_a_real_id_is_refused(tmp_path, bad):
    def mutate(d):
        ids = list(d["sample_ids"])
        ids[3] = bad
        d["sample_ids"] = ids

    res, deps = contract_body(tmp_path, mutate)
    assert "sample_ids" in res["body"]["summary"]
    assert deps.teardown_calls == 1


def test_r22_duplicate_sample_ids_are_refused(tmp_path):
    """`input_order_digest` is a hash of these, so duplicates make two
    different orders hash the same."""
    def mutate(d):
        ids = list(d["sample_ids"])
        ids[5] = ids[0]
        d["sample_ids"] = ids

    res, deps = contract_body(tmp_path, mutate)
    assert "sample_ids" in res["body"]["summary"]
    assert "duplicate" in res["body"]["summary"]
    assert deps.teardown_calls == 1


def test_r22_all_identical_sample_ids_are_refused(tmp_path):
    res, deps = contract_body(
        tmp_path, lambda d: d.update({"sample_ids": ["s0"] * R19_ROWS}))
    assert "duplicate" in res["body"]["summary"]
    assert deps.teardown_calls == 1


def test_r22_the_real_pool_has_unique_non_empty_sample_ids():
    """The rule has to describe the data the production loader produces.

    Measured against the frozen pool itself, not against a fake: a uniqueness
    rule that the real ids cannot satisfy would refuse every measured run.
    """
    pool = ROOT / "data" / "processed" / "instruct_inv_train.jsonl"
    if not pool.exists():
        pytest.skip(f"{ARTIFACT_ONLY} the instruction pool is not in this tree")
    from src.training.lora import read_rows, sample_pairs
    ids = [r.sample_id for r in sample_pairs(read_rows(pool),
                                             n_pairs=lr.POOL_PAIRS, seed=lr.SEED)]
    assert len(ids) == lr.POOL_ROWS
    assert len(set(ids)) == lr.POOL_ROWS
    assert all(isinstance(i, str) and i for i in ids)


# --- 3. provenance: one definition, shared with the replay -----------------


def test_r22_an_empty_provenance_is_refused(tmp_path):
    """A present but empty block is the same evasion as a missing one."""
    res, deps = contract_body(tmp_path,
                              lambda d: d.update({"provenance": {}}))
    assert "provenance" in res["body"]["summary"]
    assert deps.teardown_calls == 1


@pytest.mark.parametrize("field", lr.PROVENANCE_FIELDS)
def test_r22_a_missing_provenance_field_is_refused(tmp_path, field):
    res, deps = contract_body(tmp_path,
                              lambda d: d["provenance"].pop(field))
    assert field in res["body"]["summary"]
    assert deps.teardown_calls == 1


def test_r22_measurement_intervals_that_are_missing_are_refused(tmp_path):
    res, _ = contract_body(
        tmp_path, lambda d: d["provenance"].pop("measurement_intervals"))
    assert "measurement_intervals" in res["body"]["summary"]


@pytest.mark.parametrize("value", [[], "intervals", 7, None])
def test_r22_measurement_intervals_that_are_not_a_mapping_are_refused(
        tmp_path, value):
    res, deps = contract_body(
        tmp_path,
        lambda d: d["provenance"].update({"measurement_intervals": value}))
    assert "measurement_intervals" in res["body"]["summary"]
    assert deps.teardown_calls == 1


def test_r22_measurement_intervals_without_max_rows_are_refused(tmp_path):
    res, _ = contract_body(
        tmp_path,
        lambda d: d["provenance"]["measurement_intervals"].pop("max_rows"))
    assert "max_rows" in res["body"]["summary"]


@pytest.mark.parametrize("wrong", [0, 1, R19_ROWS - 1, R19_ROWS + 1, 500])
def test_r22_a_max_rows_that_disagrees_with_the_run_is_refused(tmp_path, wrong):
    """The child used to be able to write a report the replay would refuse."""
    res, deps = contract_body(
        tmp_path,
        lambda d: d["provenance"]["measurement_intervals"].update(
            {"max_rows": wrong}))
    assert "max_rows" in res["body"]["summary"]
    assert deps.teardown_calls == 1


# --- 4. the same validator, driven from both ends -------------------------


R22_PROVENANCE_BREAKAGE = [
    ("empty", lambda p: p.clear()),
    ("missing_field", lambda p: p.pop("lora_config")),
    ("missing_intervals", lambda p: p.pop("measurement_intervals")),
    ("intervals_not_a_mapping",
     lambda p: p.update({"measurement_intervals": []})),
    ("no_max_rows", lambda p: p["measurement_intervals"].pop("max_rows")),
    ("wrong_max_rows",
     lambda p: p["measurement_intervals"].update({"max_rows": 99})),
]


@pytest.mark.parametrize("name,break_it", R22_PROVENANCE_BREAKAGE)
def test_r22_the_replay_and_the_loader_check_provenance_the_same_way(
        name, break_it):
    """Behavioural: both sides are driven and their verdicts compared.

    Not a source search. Round 21 had the `max_rows` rule in the replay only,
    which meant the child could publish a report that could never be verified
    -- and only after a boot had been spent.
    """
    prov = fake_provenance(20)
    break_it(prov)

    from_loader = lr.child_load_problems(
        {"order": list(range(20)), "step": lambda i, p: {},
         "sample_ids": [f"s{i}" for i in range(20)], "provenance": prov,
         "model_load_seconds": 0.0}, rows=20)
    from_replay = lr.provenance_problems(prov, declared_rows=20)

    assert from_replay, f"{name}: the shared validator found nothing"
    assert from_loader, f"{name}: the loader accepted what the replay refuses"
    for problem in from_replay:
        assert problem in from_loader, (name, problem, from_loader)


def test_r22_a_good_provenance_satisfies_both_ends():
    prov = fake_provenance(500)
    assert lr.provenance_problems(prov, declared_rows=500) == []
    assert lr.child_load_problems(
        {"order": list(range(500)), "step": lambda i, p: {},
         "sample_ids": [f"s{i}" for i in range(500)], "provenance": prov,
         "model_load_seconds": 0.68}, rows=500) == []


def test_r22_a_report_whose_provenance_the_loader_would_refuse_is_refused(
        tmp_path):
    """The replay's own path, so the two are not merely equal in isolation."""
    report = make_report(20, declared=20, with_metrics=True)
    report["provenance"] = fake_provenance(20)
    report["provenance"]["measurement_intervals"]["max_rows"] = 19
    problems = lr.replay_child(report)["problems"]
    assert any("max_rows" in p for p in problems), problems


def test_r22_the_production_return_shape_still_satisfies_the_contract():
    """Every rule above, checked against what `ProductionChildDeps` returns.

    Built from the same constants the loader uses rather than by loading a
    model: the point is that none of the new rules would refuse the real one.
    """
    prov = fake_provenance(500)
    prov["measurement_intervals"] = {"window": lr.WINDOW,
                                     "memory_every": lr.MEMORY_EVERY,
                                     "empty_cache_every": lr.EMPTY_CACHE_EVERY,
                                     "max_rows": 500}
    order = list(range(lr.POOL_ROWS))[:500]
    assert lr.child_load_problems(
        {"order": order, "step": lambda i, p: {},
         "sample_ids": [f"{i:04d}:0:control:exact" for i in order],
         "provenance": prov, "model_load_seconds": 0.68,
         "teardown": lambda: 0.0, "clear": lambda: 0.0,
         "probe": lambda: {}}, rows=500) == []


# --- 5. the comment that outlived what it described ------------------------


def test_r22_the_wiring_comment_no_longer_says_it_never_ran():
    source = CLI.read_text()
    assert "never run against a real model" not in source, (
        "exp001 b1 ran against a real model on 2026-08-20 and failed; the "
        "round 21 --child smoke ran one to completion")
    assert "has still never run" not in source
