"""Report 15: the ordering experiment's gate, plan, and aggregation.

No model is loaded anywhere here. What can be wrong in this tooling is the
bookkeeping -- whether a gate that cannot be evaluated counts as passed,
whether a child can talk its way into being called comparable, whether a plan
can be edited after a result is visible -- and all of that is testable against
fakes, exactly and in milliseconds.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.preflight import (
    GATE_METRICS,
    GATE_SPEC,
    calibrate,
    evaluate_gate,
    mad,
    median,
    thresholds_from,
    wait_for_recovery,
)
from src.training.session import (
    EVENT_FINISHED,
    EVENT_GATE_ATTEMPT,
    EVENT_PRE_SPAWN_ABORT,
    EVENT_STARTED,
    append_event,
    boot_identity,
    event_file_name,
    exclusive_lock,
    manifest_digest,
    read_events,
    snapshot_sources,
    verify_sources,
    write_once_json,
)

ROOT = Path(__file__).resolve().parents[1]

#: Per-run evidence and per-record reports live in the private research tree
#: and are not published. Tests that read them are artifact-only; everything
#: else in this file runs against fakes or tmp_path and must never skip.
EXP001_DIR = ROOT / "data" / "reports" / "15_mps_order" / "exp001"
REPORT_15_MD = ROOT / "data" / "reports" / "15_mps_order.md"
REPORT_14_JSON = ROOT / "data" / "reports" / "14_mps_speed.json"

ARTIFACT_ONLY = "artifact-only:"

needs_report_14 = pytest.mark.skipif(
    not REPORT_14_JSON.exists(),
    reason=f"{ARTIFACT_ONLY} report 14's per-row record is not in this tree")
SCRIPT = ROOT / "scripts" / "15_mps_order.py"


def order_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("s15x", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return order_module()


def idle(swap=1.0, pressure=80.0, fpi=8.0, load=0.10):
    return {"sampled_at": 0.0, "swap_used_gb": swap,
            "memory_pressure_percent_free": pressure,
            "free_plus_inactive_gb": fpi, "normalized_load_1m": load}


@pytest.fixture
def thresholds():
    return thresholds_from(calibrate([idle() for _ in range(10)]))


# ---------------------------------------------------------------- statistics

class TestRobustStatistics:
    def test_median_of_odd_and_even(self):
        assert median([3, 1, 2]) == 2
        assert median([4, 1, 2, 3]) == 2.5

    def test_median_ignores_missing_readings(self):
        assert median([1, None, 3]) == 2

    def test_all_missing_gives_none_not_zero(self):
        """A metric nobody could read has no median, and zero is a number."""
        assert median([None, None]) is None
        assert mad([None]) is None

    def test_mad_is_about_the_median(self):
        assert mad([1, 1, 1, 1, 5]) == 0

    def test_scale_is_mad_times_the_constant(self):
        stats = calibrate([idle(swap=v) for v in (1, 1, 1, 3, 1)])
        s = stats["swap_used_gb"]
        assert s["median"] == 1
        assert s["mad"] == 0
        assert s["scale"] == 0.0


class TestCalibrationThresholds:
    def test_upper_metric_uses_median_plus_band(self):
        stats = calibrate([idle(swap=2.0) for _ in range(10)])
        # scale is 0, so the floor applies: 2.0 + max(0, 0.25)
        assert stats["swap_used_gb"]["threshold"] == pytest.approx(2.25)

    def test_lower_metric_uses_median_minus_band_with_a_floor(self):
        stats = calibrate([idle(pressure=80.0) for _ in range(10)])
        # 80 - max(0, 5) = 75, and the absolute floor of 20 does not bind
        assert stats["memory_pressure_percent_free"]["threshold"] == 75.0

    def test_the_absolute_floor_binds_when_calibration_was_unhealthy(self):
        """A machine calibrated under pressure must not lower its own bar."""
        stats = calibrate([idle(pressure=21.0) for _ in range(10)])
        assert stats["memory_pressure_percent_free"]["threshold"] == 20.0

    def test_free_plus_inactive_floor_binds(self):
        stats = calibrate([idle(fpi=2.1) for _ in range(10)])
        assert stats["free_plus_inactive_gb"]["threshold"] == 2.0

    def test_load_threshold_is_capped(self):
        stats = calibrate([idle(load=0.9) for _ in range(10)])
        assert stats["normalized_load_1m"]["threshold"] == 0.50

    def test_a_wide_spread_widens_the_band(self):
        steady = calibrate([idle(swap=1.0) for _ in range(10)])
        noisy = calibrate([idle(swap=v) for v in
                           (1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0)])
        assert (noisy["swap_used_gb"]["threshold"]
                > steady["swap_used_gb"]["threshold"])

    def test_every_metric_records_its_formula(self):
        stats = calibrate([idle() for _ in range(10)])
        for m in GATE_METRICS:
            assert "median" in stats[m]["formula"]
            assert str(GATE_SPEC[m]["min_slack"]) in stats[m]["formula"]

    def test_missing_readings_are_counted_not_hidden(self):
        stats = calibrate([idle(), {**idle(), "swap_used_gb": None}])
        assert stats["swap_used_gb"]["n"] == 1
        assert stats["swap_used_gb"]["n_missing"] == 1


# ---------------------------------------------------------------------- gate

class TestGate:
    def test_an_idle_machine_passes(self, thresholds):
        assert evaluate_gate(idle(), thresholds)["passed"]

    def test_swap_above_the_band_fails(self, thresholds):
        v = evaluate_gate(idle(swap=99.0), thresholds)
        assert not v["passed"] and v["failed"] == ["swap_used_gb"]

    def test_pressure_below_the_band_fails(self, thresholds):
        v = evaluate_gate(idle(pressure=5.0), thresholds)
        assert "memory_pressure_percent_free" in v["failed"]

    def test_free_plus_inactive_below_the_band_fails(self, thresholds):
        v = evaluate_gate(idle(fpi=0.4), thresholds)
        assert "free_plus_inactive_gb" in v["failed"]

    def test_load_above_the_band_fails(self, thresholds):
        v = evaluate_gate(idle(load=3.0), thresholds)
        assert "normalized_load_1m" in v["failed"]

    @pytest.mark.parametrize("metric", list(GATE_METRICS))
    def test_an_unreadable_metric_fails_rather_than_passes(self, thresholds,
                                                           metric):
        """A gate that could not be evaluated has not been satisfied."""
        v = evaluate_gate({**idle(), metric: None}, thresholds)
        assert not v["passed"]
        assert v["checks"][metric]["detail"] == "metric was not readable"

    def test_a_metric_with_no_threshold_fails(self):
        v = evaluate_gate(idle(), {})
        assert not v["passed"]
        assert set(v["failed"]) == set(GATE_METRICS)


class FakeSleep:
    def __init__(self):
        self.t = 0.0
        self.calls = 0

    def __call__(self, seconds):
        self.calls += 1
        self.t += seconds

    def clock(self):
        return self.t


class TestRecoveryPolling:
    def build(self, samples):
        f = FakeSleep()
        it = iter(samples)
        return f, (lambda: next(it))

    def test_three_consecutive_passes_release_the_gate(self, thresholds):
        f, sampler = self.build([idle()] * 3)
        out = wait_for_recovery(thresholds, sampler=sampler, clock=f.clock,
                                sleep=f, poll_seconds=30,
                                max_wait_seconds=900)
        assert out["passed"] and len(out["polls"]) == 3

    def test_a_failure_resets_the_streak(self, thresholds):
        """Two good, one bad, three good: the run of two does not count."""
        f, sampler = self.build(
            [idle(), idle(), idle(swap=99.0), idle(), idle(), idle()])
        out = wait_for_recovery(thresholds, sampler=sampler, clock=f.clock,
                                sleep=f, poll_seconds=30,
                                max_wait_seconds=900)
        assert out["passed"] and len(out["polls"]) == 6
        assert out["polls"][2]["consecutive_passes"] == 0

    def test_timing_out_is_a_failure_not_a_warning(self, thresholds):
        f = FakeSleep()
        out = wait_for_recovery(thresholds, sampler=lambda: idle(swap=99.0),
                                clock=f.clock, sleep=f, poll_seconds=30,
                                max_wait_seconds=90)
        assert out["passed"] is False
        assert "did not return inside the calibrated band" in out["reason"]

    def test_the_wait_is_bounded(self, thresholds):
        f = FakeSleep()
        out = wait_for_recovery(thresholds, sampler=lambda: idle(swap=99.0),
                                clock=f.clock, sleep=f, poll_seconds=30,
                                max_wait_seconds=90)
        assert out["waited_seconds"] <= 120

    def test_every_poll_is_recorded_for_review(self, thresholds):
        f, sampler = self.build([idle(swap=99.0), idle(), idle(), idle()])
        out = wait_for_recovery(thresholds, sampler=sampler, clock=f.clock,
                                sleep=f, poll_seconds=30, max_wait_seconds=900)
        assert [p["passed"] for p in out["polls"]] == [False, True, True, True]
        assert out["polls"][0]["failed_metrics"] == ["swap_used_gb"]


class TestCalibrationRecordsNothingAboutOtherWork:
    def test_the_sample_has_only_the_gated_numbers(self):
        from src.training.preflight import preflight_sample

        s = preflight_sample(clock=lambda: 0.0)
        assert set(s) == {"sampled_at", *GATE_METRICS}

    def test_no_field_names_free_or_available_memory(self):
        from src.training.preflight import preflight_sample

        keys = " ".join(preflight_sample(clock=lambda: 0.0))
        assert "free_plus_inactive_gb" in keys
        assert "free_gb" not in keys and "available" not in keys


# ---------------------------------------------------------------------- plan

class TestPlan:
    def test_the_declared_order_is_c_e_e_c(self, mod):
        assert [p["condition"] for p in mod.build_plan("x")] == [
            "continuous", "empty_cache", "empty_cache", "continuous"]

    def test_condition_is_balanced_against_execution_position(self, mod):
        plan = mod.build_plan("x")
        pos = {"continuous": [], "empty_cache": []}
        for p in plan:
            pos[p["condition"]].append(p["global_position"])
        assert (sum(pos["continuous"]) / 2) == (sum(pos["empty_cache"]) / 2)

    def test_two_blocks_reversed(self, mod):
        plan = mod.build_plan("x")
        b1 = [p["condition"] for p in plan if p["block_id"] == "B1"]
        b2 = [p["condition"] for p in plan if p["block_id"] == "B2"]
        assert b1 == list(reversed(b2))

    def test_each_condition_runs_twice(self, mod):
        conds = [p["condition"] for p in mod.build_plan("x")]
        assert conds.count("continuous") == 2 == conds.count("empty_cache")

    def test_the_digest_is_deterministic(self, mod):
        assert (mod.digest_obj(mod.build_plan("x"))
                == mod.digest_obj(mod.build_plan("x")))

    def test_editing_the_plan_changes_the_digest(self, mod):
        plan = mod.build_plan("x")
        before = mod.digest_obj(plan)
        plan[0]["condition"] = "empty_cache"
        assert mod.digest_obj(plan) != before


class TestAtomicWrite:
    def test_it_writes(self, mod, tmp_path):
        p = tmp_path / "a.json"
        sha = mod.atomic_write_json(p, {"a": 1})
        assert json.loads(p.read_text()) == {"a": 1}
        assert sha == mod.sha256_file(p)

    def test_it_refuses_to_overwrite(self, mod, tmp_path):
        p = tmp_path / "a.json"
        mod.atomic_write_json(p, {"a": 1})
        with pytest.raises(SystemExit, match="refusing to overwrite"):
            mod.atomic_write_json(p, {"a": 2})
        assert json.loads(p.read_text()) == {"a": 1}

    def test_it_leaves_no_partial_file_behind(self, mod, tmp_path):
        p = tmp_path / "a.json"
        mod.atomic_write_json(p, {"a": 1})
        assert not list(tmp_path.glob("*.tmp"))


# ------------------------------------------------------------ child records

def child_record(mod, entry, digest, *, rows=200, loss_base=0.5,
                 stopped=None, **over):
    every = 10 if entry["condition"] == "empty_cache" else None
    clear_cost = 0.05
    per_row = [
        {"collate_h2d": 0.001, "forward": 0.6, "backward": 0.4,
         "optimizer": 0.0, "total": 1.0, "end_to_end": 1.1,
         "scheduled_empty_cache_seconds": (
             clear_cost if every and (i + 1) % every == 0 else 0.0),
         "memory_probe_seconds": 0.09,
         "row": i + 1, "sample_id": f"s{i}", "n_tokens": 100,
         "n_supervised": 50, "loss": loss_base + i * 1e-9}
        for i in range(rows)]
    calls = (rows // every) if every else 0
    clear_total = round(calls * clear_cost, 5)
    probe_total = round(rows * 0.09, 4)
    # The overhead has to be the parts, and the parts have to be the rows.
    end_to_end = round(rows * 1.1, 4)
    compute = round(rows * 1.0, 4)
    overhead = round(end_to_end - compute, 4)
    rec = {
        "schema_version": mod.EXPERIMENT_SCHEMA, "kind": "child",
        **{k: entry[k] for k in ("experiment_id", "run_id", "block_id",
                                 "condition", "position_in_block",
                                 "global_position")},
        "plan_digest": digest, "child_pid": 1234,
        "child_source_check": {"files_verified": 2,
                               "source_manifest_digest": "s" * 64,
                               "verified_at": "2026-08-14T00:00:00+00:00"},
        "started_at": "2026-08-14T00:00:00+00:00",
        "finished_at": "2026-08-14T00:10:00+00:00",
        "provenance": {
            "head": "a" * 40, "working_tree_dirty": False,
            "code_sha256": {"scripts/15_mps_order.py": "b" * 64},
            "instruction_sha256": {"instruct_inv_train.jsonl": "c" * 64},
            "selection_digest": "d" * 64, "training_order_digest": "e" * 64,
            "lora_config": {"rank": 16, "seed": 0},
            "optimizer": {"class": "AdamW", "lr": 0.0001},
            "packages": {"torch": "2.13.0"}, "device": "mps",
            "dtype": "bfloat16", "phases": list(mod.PHASES),
            "stop_conditions": {"slow_row_seconds": 30.0,
                                "slow_row_streak": 3, "max_seconds": 1800},
            "measurement_intervals": {"window": 20, "memory_sample_every": 5,
                                      "empty_cache_every": 10,
                                      "max_rows": 200},
            "base_model": "m", "base_revision": "r",
            "published_adapter": "a",
            "published_adapter_revision": "p", "tokenizer_revision": "t"},
        "preflight": idle(),
        "model_load_seconds": 12.0,
        "rows_completed": rows, "rows_requested": 200,
        "stopped_early": stopped,
        "input_order_digest": mod.digest_ids(f"s{i}" for i in range(200)),
        "completed_input_digest": mod.digest_ids(f"s{i}" for i in range(rows)),
        "end_to_end_seconds": end_to_end, "model_compute_seconds": compute,
        "between_row_overhead_seconds": overhead,
        "between_row_overhead_breakdown": {
            "scheduled_empty_cache_seconds": clear_total,
            "memory_probe_seconds": probe_total,
            "unattributed_seconds": round(
                overhead - clear_total - probe_total, 4)},
        "end_to_end_seconds_per_row": 1.1,
        "model_compute_seconds_per_row": 1.0,
        "scheduled_empty_cache_every": every,
        "scheduled_empty_cache_calls": calls,
        "scheduled_empty_cache_cost": {
            "calls": calls, "total_seconds": clear_total,
            "mean_seconds": clear_cost if calls else None,
            "max_seconds": clear_cost if calls else None,
            "per_call_seconds": [clear_cost] * calls},
        "teardown_empty_cache_calls": 1, "teardown_empty_cache_seconds": 0.02,
        "loss_decimals_stored": None,
        "phases": {"forward": {"mean_seconds": 0.6, "total_seconds": 120.0},
                   "backward": {"mean_seconds": 0.4, "total_seconds": 80.0},
                   "_unattributed": {"total_seconds": 0.0,
                                     "share_of_total": 0.0}},
        "windows": [{"window": 0, "rows": "1-20", "n_rows": 20,
                     "seconds": 20.0, "seconds_per_row": 1.0,
                     "end_to_end_seconds": 22.0,
                     "end_to_end_seconds_per_row": 1.1, "tokens": 2000,
                     "supervised_tokens": 1000, "tokens_per_second": 100.0,
                     "mean_seq_len": 100.0}],
        "memory": [{"row": 1, "elapsed_seconds": 1.0,
                    "mps_current_allocated_gb": 2.3,
                    "mps_driver_allocated_gb": 9.0,
                    "mps_recommended_max_gb": 37.44,
                    "peak_process_rss_gb": 1.0,
                    "free_plus_inactive_gb": 7.0, "swap_used_gb": 1.0,
                    "memory_pressure_percent_free": 80,
                    "probe_seconds": 0.04}],
        "per_row": per_row, "trainable_parameters": 1703936,
    }
    rec.update(over)
    return rec


GATE_POLICY = {"poll_seconds": 30, "max_wait_seconds": 900,
               "consecutive_passes_required": 3}


def calibration_doc(mod, *, samples=None, gate=None):
    """A calibration record whose thresholds really do follow from its samples.

    Built with the same functions the tool uses, so a test that tampers with
    one number is testing the recomputation rather than a hand-written fixture.
    """
    samples = samples if samples is not None else [idle() for _ in range(10)]
    gate = gate or dict(GATE_POLICY)
    stats = calibrate(samples)
    thresholds = thresholds_from(stats)
    return {"schema_version": mod.EXPERIMENT_SCHEMA,
            "kind": "preflight_calibration", "created_at": "2026-08-14T00:00Z",
            "loads_model": False, "samples_requested": len(samples),
            "interval_seconds": 30, "metrics": list(GATE_METRICS),
            "note": "test fixture", "scale_formula": "scale = 1.4826 * MAD",
            "samples": samples, "stats": stats, "thresholds": thresholds,
            "gate": gate,
            "calibration_digest": mod.digest_obj(
                {"stats": stats, "thresholds": thresholds, "gate": gate})}


def gate_polls(samples, thresholds, *, poll_seconds=30):
    polls, streak = [], 0
    for i, s in enumerate(samples, 1):
        v = evaluate_gate(s, thresholds)
        streak = streak + 1 if v["passed"] else 0
        polls.append({"poll": i,
                      "elapsed_seconds": round((i - 1) * poll_seconds + 0.01, 2),
                      "sample": s, "passed": v["passed"],
                      "failed_metrics": v["failed"],
                      "consecutive_passes": streak})
    return polls


def gate_record(polls, *, needed=3, max_wait=900):
    passed = any(p["consecutive_passes"] >= needed for p in polls)
    return {"passed": passed, "polls": polls,
            "waited_seconds": polls[-1]["elapsed_seconds"] if polls else 0.0,
            "consecutive_passes_required": needed,
            "reason": None if passed else
            (f"machine did not return inside the calibrated band {needed} "
             f"times running within {max_wait}s")}


def passing_gate(thresholds):
    return gate_record(gate_polls([idle()] * 3, thresholds))


def failing_gate(thresholds, *, polls=31):
    return gate_record(gate_polls([idle(swap=99.0)] * polls, thresholds))


def experiment(mod, *, exit_codes=(0, 0, 0, 0), records=None, calib=None,
               gates=None):
    plan = mod.build_plan("exp1")
    digest = mod.digest_obj(plan)
    calib = calib or calibration_doc(mod)
    thresholds = calib["thresholds"]
    recs = records or [child_record(mod, e, digest) for e in plan]
    gates = gates or [passing_gate(thresholds) for _ in plan]
    children = [
        {"entry": e, "record": r, "exit_status": x,
         "run_id": e["run_id"], "condition": e["condition"],
         "block_id": e["block_id"], "global_position": e["global_position"],
         "gate": g, "recovery_passed_recomputed": bool(g["passed"]),
         # One run per boot: the default fixture is a well-formed experiment,
         # so each run carries a boot of its own.
         "boot_fingerprint": f"boot-{e['run_id']}",
         "report_path": f"data/reports/15_mps_order/exp1/{e['run_id']}.json",
         "report_sha256": "0" * 64, "not_run": False}
        for e, r, x, g in zip(plan, recs, exit_codes, gates)]
    exp = {"schema_version": mod.EXPERIMENT_SCHEMA, "kind": "experiment",
           "experiment_id": "exp1", "created_at": "2026-08-14T00:00:00+00:00",
           "plan": plan, "plan_digest": digest, "plan_path": "p",
           "calibration_digest": calib["calibration_digest"],
           "thresholds": thresholds,
           "gate_policy": calib["gate"],
           "children": children,
           "complete": all(x == 0 for x in exit_codes),
           "stopped_reason": None, "elapsed_seconds": 900.0}
    return mod.analyse(exp, calib)


class TestChildValidation:
    def setup_plan(self, mod):
        plan = mod.build_plan("exp1")
        return plan, mod.digest_obj(plan), plan[0]

    def test_a_good_record_passes(self, mod):
        plan, digest, entry = self.setup_plan(mod)
        assert mod.validate_child(child_record(mod, entry, digest), entry,
                                  digest) == []

    @pytest.mark.parametrize("key", ["exit_status", "recovery_passed",
                                     "eligible_for_paired_contrast",
                                     "complete", "comparable"])
    def test_a_child_may_not_state_a_conclusion(self, mod, key):
        plan, digest, entry = self.setup_plan(mod)
        rec = child_record(mod, entry, digest, **{key: True})
        problems = mod.validate_child(rec, entry, digest)
        assert any(key in p for p in problems)

    @pytest.mark.parametrize("key", ["provenance", "per_row", "preflight",
                                     "model_load_seconds", "plan_digest",
                                     "teardown_empty_cache_seconds"])
    def test_a_missing_required_field_is_rejected(self, mod, key):
        plan, digest, entry = self.setup_plan(mod)
        rec = child_record(mod, entry, digest)
        del rec[key]
        assert any(key in p for p in mod.validate_child(rec, entry, digest))

    def test_a_wrong_plan_digest_is_rejected(self, mod):
        plan, digest, entry = self.setup_plan(mod)
        rec = child_record(mod, entry, digest, plan_digest="0" * 64)
        assert any("plan digest" in p
                   for p in mod.validate_child(rec, entry, digest))

    def test_a_child_that_ran_another_condition_is_rejected(self, mod):
        plan, digest, entry = self.setup_plan(mod)
        rec = child_record(mod, entry, digest, condition="empty_cache")
        assert any("condition" in p
                   for p in mod.validate_child(rec, entry, digest))

    def test_a_child_at_the_wrong_position_is_rejected(self, mod):
        plan, digest, entry = self.setup_plan(mod)
        rec = child_record(mod, entry, digest, global_position=3)
        assert any("global_position" in p
                   for p in mod.validate_child(rec, entry, digest))

    def test_rounded_losses_are_rejected(self, mod):
        plan, digest, entry = self.setup_plan(mod)
        rec = child_record(mod, entry, digest, loss_decimals_stored=4)
        assert any("rounded" in p
                   for p in mod.validate_child(rec, entry, digest))

    def test_a_clear_count_that_contradicts_the_schedule_is_rejected(self, mod):
        plan, digest, _ = self.setup_plan(mod)
        entry = plan[1]                      # empty_cache
        rec = child_record(mod, entry, digest)
        rec["scheduled_empty_cache_calls"] += 1
        assert any("implies" in p
                   for p in mod.validate_child(rec, entry, digest))

    def test_a_teardown_folded_into_the_schedule_is_rejected(self, mod):
        plan, digest, _ = self.setup_plan(mod)
        entry = plan[1]
        rec = child_record(mod, entry, digest)
        rec["scheduled_empty_cache_calls"] = (
            rec["scheduled_empty_cache_calls"]
            + rec["teardown_empty_cache_calls"])
        assert any("implies" in p
                   for p in mod.validate_child(rec, entry, digest))

    def test_a_control_arm_with_a_clear_schedule_is_rejected(self, mod):
        plan, digest, entry = self.setup_plan(mod)
        rec = child_record(mod, entry, digest,
                           scheduled_empty_cache_every=10)
        assert any("control arm" in p
                   for p in mod.validate_child(rec, entry, digest))

    def test_edited_sample_ids_break_the_digest(self, mod):
        plan, digest, entry = self.setup_plan(mod)
        rec = child_record(mod, entry, digest)
        rec["per_row"][5]["sample_id"] = "swapped"
        assert any("digest" in p
                   for p in mod.validate_child(rec, entry, digest))

    def test_a_dropped_row_is_rejected(self, mod):
        plan, digest, entry = self.setup_plan(mod)
        rec = child_record(mod, entry, digest)
        rec["per_row"].pop()
        assert any("rows_completed" in p
                   for p in mod.validate_child(rec, entry, digest))

    def test_a_non_bool_dirty_flag_is_rejected(self, mod):
        plan, digest, entry = self.setup_plan(mod)
        rec = child_record(mod, entry, digest)
        rec["provenance"]["working_tree_dirty"] = None
        assert any("working_tree_dirty" in p
                   for p in mod.validate_child(rec, entry, digest))

    def test_empty_code_digests_are_rejected(self, mod):
        plan, digest, entry = self.setup_plan(mod)
        rec = child_record(mod, entry, digest)
        rec["provenance"]["code_sha256"] = {}
        assert any("code digests" in p
                   for p in mod.validate_child(rec, entry, digest))


class TestParentRecomputesVerdicts:
    def test_all_good_runs_are_eligible(self, mod):
        exp = experiment(mod)
        assert all(v["eligible_for_paired_contrast"]
                   for v in exp["verdicts"].values())

    def test_a_nonzero_exit_makes_a_run_ineligible(self, mod):
        exp = experiment(mod, exit_codes=(0, 1, 0, 0))
        v = exp["verdicts"]["r2"]
        assert not v["eligible_for_paired_contrast"]
        assert any("exit status 1" in r for r in v["reasons"])

    def test_a_stopped_run_is_ineligible(self, mod):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = [child_record(mod, e, digest) for e in plan]
        recs[2] = child_record(mod, plan[2], digest, rows=40,
                               stopped="3 consecutive rows over 30.0s")
        exp = experiment(mod, records=recs)
        v = exp["verdicts"]["r3"]
        assert not v["eligible_for_paired_contrast"]
        assert any("stopped early" in r for r in v["reasons"])

    def test_a_different_input_order_makes_a_run_ineligible(self, mod):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = [child_record(mod, e, digest) for e in plan]
        recs[3]["input_order_digest"] = "9" * 64
        exp = experiment(mod, records=recs)
        assert any("input order differs" in r
                   for r in exp["verdicts"]["r4"]["reasons"])

    def test_differing_provenance_makes_a_run_ineligible(self, mod):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = [child_record(mod, e, digest) for e in plan]
        recs[1]["provenance"]["head"] = "z" * 40
        exp = experiment(mod, records=recs)
        assert any("provenance differs on head" in r
                   for r in exp["verdicts"]["r2"]["reasons"])

    def test_a_child_claim_cannot_make_a_bad_run_eligible(self, mod):
        """The parent recomputes; it does not read the child's opinion."""
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = [child_record(mod, e, digest) for e in plan]
        recs[0]["eligible_for_paired_contrast"] = True
        recs[0]["rows_completed"] = 10
        recs[0]["per_row"] = recs[0]["per_row"][:10]
        exp = experiment(mod, records=recs)
        assert not exp["verdicts"]["r1"]["eligible_for_paired_contrast"]

    def test_a_failed_recovery_makes_a_run_ineligible(self, mod):
        """Recovery comes from replaying the polls, not from a stored flag."""
        calib = calibration_doc(mod)
        gates = [passing_gate(calib["thresholds"]) for _ in range(4)]
        gates[2] = failing_gate(calib["thresholds"])
        exp = experiment(mod, calib=calib, gates=gates)
        assert any("had not recovered" in r
                   for r in exp["verdicts"]["r3"]["reasons"])

    def test_a_stored_recovery_flag_cannot_override_the_polls(self, mod):
        calib = calibration_doc(mod)
        gates = [passing_gate(calib["thresholds"]) for _ in range(4)]
        gates[2] = failing_gate(calib["thresholds"])
        exp = experiment(mod, calib=calib, gates=gates)
        # The aggregate insists the machine was fine. The polls say otherwise.
        exp["children"][2]["recovery_passed_recomputed"] = True
        again = mod.analyse(exp, calib)
        assert any("had not recovered" in r
                   for r in again["verdicts"]["r3"]["reasons"])
        assert any("replaying the polls says" in p
                   for p in again["replay_problems"])


class TestLossComparison:
    def test_all_pairs_are_kept(self, mod):
        exp = experiment(mod)
        assert len(exp["losses"]["pairs"]) == 6      # 4 choose 2

    def test_identical_losses_are_within_tolerance(self, mod):
        exp = experiment(mod)
        assert exp["losses"]["verdict"] == "within_tolerance"
        assert exp["losses"]["max_abs_loss_diff_overall"] < 1e-6

    def test_one_divergent_run_fails_the_whole_set(self, mod):
        """No block is dropped on the strength of its own loss spread."""
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = [child_record(mod, e, digest) for e in plan]
        recs[3] = child_record(mod, plan[3], digest, loss_base=0.9)
        exp = experiment(mod, records=recs)
        assert exp["losses"]["verdict"] == "over_tolerance"
        assert exp["losses"]["max_abs_loss_diff_overall"] > 0.3

    def test_token_mismatches_are_counted(self, mod):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = [child_record(mod, e, digest) for e in plan]
        recs[1]["per_row"][0]["n_tokens"] = 999
        exp = experiment(mod, records=recs)
        assert exp["losses"]["token_count_mismatches_total"] > 0

    def test_the_tolerance_is_declared_as_an_engineering_choice(self, mod):
        basis = mod.LOSS_TOLERANCE_BASIS
        assert "NOT derived from report 14" in basis
        assert "four decimal places" in basis
        assert mod.LOSS_TOLERANCE == 1e-3

    def test_no_tolerance_is_derived_from_report_14(self):
        """5e-5 would be an inference from rounded data. It must not appear."""
        src = SCRIPT.read_text()
        assert "5e-5" not in src


class TestAggregateRendering:
    def render(self, mod, exp, tmp_path):
        out = tmp_path / "15_mps_order.md"
        mod._write_markdown(exp, out)
        return out.read_text()

    def test_a_complete_consistent_experiment_may_claim_a_mitigation(
            self, mod, tmp_path):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = []
        for e in plan:
            r = child_record(mod, e, digest)
            fast = e["condition"] == "empty_cache"
            r["windows"][0]["seconds_per_row"] = 1.0 if fast else 5.0
            recs.append(r)
        text = self.render(mod, experiment(mod, records=recs), tmp_path)
        assert "repeatedly observed engineering mitigation" in text

    def test_an_inconsistent_direction_may_not(self, mod, tmp_path):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = []
        for e in plan:
            r = child_record(mod, e, digest)
            r["windows"][0]["seconds_per_row"] = (
                1.0 if e["run_id"] in ("r2", "r4") else 5.0)
            recs.append(r)
        text = self.render(mod, experiment(mod, records=recs), tmp_path)
        assert "not the same in both blocks" in text
        assert "repeatedly observed engineering mitigation" not in text

    def test_an_incomplete_experiment_compares_nothing(self, mod, tmp_path):
        exp = experiment(mod)
        exp["complete"] = False
        exp["stopped_reason"] = "gate timeout before r3"
        text = self.render(mod, exp, tmp_path)
        assert "did not complete, so nothing is compared" in text

    def test_it_never_claims_a_mechanism(self, mod, tmp_path):
        text = self.render(mod, experiment(mod), tmp_path)
        assert "none is separated here" in text
        for claim in ("proves", "demonstrates that fragmentation",
                      "caused by the allocator"):
            assert claim not in text

    def test_inactive_pages_are_never_called_free_memory(self, mod, tmp_path):
        text = self.render(mod, experiment(mod), tmp_path)
        assert "neither free nor available memory" in text
        assert "free memory falls" not in text

    def test_teardown_is_stated_to_be_outside_every_figure(self, mod,
                                                           tmp_path):
        text = self.render(mod, experiment(mod), tmp_path)
        assert "inside none of the figures above" in text

    def test_the_declared_plan_is_printed(self, mod, tmp_path):
        text = self.render(mod, experiment(mod), tmp_path)
        assert "Declared before running" in text
        assert "mean global position 1.5" in text


class TestAggregateStripsRecords:
    def test_children_are_referenced_by_digest_not_copied(self, mod):
        stripped = mod._strip_records(experiment(mod))
        for c in stripped["children"]:
            assert "record" not in c and "entry" not in c
            assert c["report_sha256"] and c["report_path"]
            assert c["summary"]["rows_completed"] == 200

    def test_a_missing_child_summarises_to_none(self, mod):
        plan = mod.build_plan("exp1")
        exp = experiment(mod)
        exp["children"][3]["record"] = None
        assert mod._strip_records(exp)["children"][3]["summary"] is None


class TestChildRefusesBadInvocation:
    """The child checks its orders before it loads anything."""

    def test_a_mismatched_plan_digest_stops_before_the_model(self, mod,
                                                             tmp_path):
        import subprocess

        plan = mod.build_plan("exp1")
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(
            {"plan": plan, "plan_digest": mod.digest_obj(plan)}))
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--child", "--plan", str(plan_path),
             "--plan-digest", "0" * 64, "--session-dir", str(tmp_path),
             "--global-position", "0",
             "--run-id", "r1", "--condition", "continuous",
             "--out", str(tmp_path / "r1.json")],
            capture_output=True, text=True, cwd=str(ROOT), timeout=180)
        assert r.returncode != 0
        assert "plan digest does not match" in r.stderr
        assert not (tmp_path / "r1.json").exists()

    def test_a_child_told_the_wrong_condition_stops(self, mod, tmp_path):
        import subprocess

        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(
            {"plan": plan, "plan_digest": digest}))
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--child", "--plan", str(plan_path),
             "--plan-digest", digest, "--session-dir", str(tmp_path),
             "--global-position", "0",
             "--run-id", "r1", "--condition", "empty_cache",
             "--out", str(tmp_path / "r1.json")],
            capture_output=True, text=True, cwd=str(ROOT), timeout=180)
        assert r.returncode != 0
        assert "but this child was told" in r.stderr


class TestLossVerdictHasThreeStates:
    """"No pair to compare" is not "compared and failed"."""

    def test_no_completed_pair_is_not_applicable(self, mod):
        exp = experiment(mod)
        only_one = [exp["children"][0]] + [
            {**c, "record": None} for c in exp["children"][1:]]
        losses = mod.compare_losses(only_one)
        assert losses["comparable_pairs"] == 0
        assert losses["verdict"] == "not_applicable"
        assert losses["max_abs_loss_diff_overall"] is None

    def test_matching_runs_are_within_tolerance(self, mod):
        assert experiment(mod)["losses"]["verdict"] == "within_tolerance"

    def test_a_divergent_run_is_over_tolerance(self, mod):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = [child_record(mod, e, digest) for e in plan]
        recs[3] = child_record(mod, plan[3], digest, loss_base=0.9)
        assert experiment(mod, records=recs)["losses"]["verdict"] == (
            "over_tolerance")

    def test_an_incomparable_run_claims_no_exceedance(self, mod, tmp_path):
        exp = experiment(mod)
        for c in exp["children"][1:]:
            c["record"] = None
        exp["complete"] = False
        exp["stopped_reason"] = "gate timeout before r2"
        exp["losses"] = mod.compare_losses(exp["children"])
        out = tmp_path / "m.md"
        mod._write_markdown(exp, out)
        text = out.read_text()
        assert "no loss verdict exists" in text
        assert "At least one pair exceeds" not in text


class TestStoredExperimentValidation:
    """The aggregate references children by digest, so the digest is checked."""

    @pytest.fixture(autouse=True)
    def _root(self, mod, tmp_path, monkeypatch):
        """Everything a stored experiment references is relative to the tree
        root, so the tree root is where the fixture is built."""
        monkeypatch.setattr(mod, "ROOT", tmp_path)

    def build(self, mod, tmp_path, *, n=4, calib=None):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        exp_dir = tmp_path / "exp1"
        exp_dir.mkdir()
        calib = calib or calibration_doc(mod)
        calib_path = tmp_path / "calibration.json"
        mod.atomic_write_json(calib_path, calib)
        mod.atomic_write_json(exp_dir / "plan.json", {
            "schema_version": mod.EXPERIMENT_SCHEMA, "kind": "plan",
            "plan": plan, "plan_digest": digest,
            "calibration_path": "calibration.json",
            "calibration_digest": calib["calibration_digest"]})
        children = []
        for entry in plan[:n]:
            rec = child_record(mod, entry, digest)
            p = exp_dir / f"{entry['run_id']}.json"
            sha = mod.atomic_write_json(p, rec)
            children.append({
                "run_id": entry["run_id"], "condition": entry["condition"],
                "block_id": entry["block_id"],
                "global_position": entry["global_position"],
                "exit_status": 0, "not_run": False,
                "gate": passing_gate(calib["thresholds"]),
                "recovery_passed_recomputed": True,
                "report_path": str(p.relative_to(mod.ROOT))
                if str(p).startswith(str(mod.ROOT)) else str(p),
                "report_sha256": sha, "summary": mod._summarise(rec)})
        agg = {"schema_version": mod.EXPERIMENT_SCHEMA, "kind": "experiment",
               "experiment_id": "exp1", "plan": plan, "plan_digest": digest,
               "calibration_digest": calib["calibration_digest"],
               "thresholds": calib["thresholds"], "gate_policy": calib["gate"],
               "children": children, "complete": n == 4,
               "stopped_reason": None}
        mod.atomic_write_json(exp_dir / "aggregate.json", agg)
        return exp_dir, plan, digest

    def rewrite(self, path, obj):
        path.unlink()
        path.write_text(json.dumps(obj, indent=2) + "\n")

    def load(self, mod, exp_dir):
        agg = json.loads((exp_dir / "aggregate.json").read_text())
        plan_doc = json.loads((exp_dir / "plan.json").read_text())
        _, calib, problems = mod.load_calibration(plan_doc)
        return problems + mod.validate_experiment(agg, plan_doc, exp_dir, calib)

    def test_a_clean_experiment_validates(self, mod, tmp_path):
        exp_dir, _, _ = self.build(mod, tmp_path)
        assert self.load(mod, exp_dir) == []

    def test_a_clean_experiment_replays_end_to_end(self, mod, tmp_path):
        exp_dir, _, _ = self.build(mod, tmp_path)
        exp = mod.load_experiment(exp_dir)
        assert exp["replay_problems"] == []
        assert all(g["passed"] for g in exp["gate_replay"].values())

    def test_an_edited_child_file_is_rejected(self, mod, tmp_path):
        exp_dir, plan, digest = self.build(mod, tmp_path)
        # Change a measurement after the fact.
        rec = json.loads((exp_dir / "r1.json").read_text())
        rec["model_compute_seconds"] = 1.0
        self.rewrite(exp_dir / "r1.json", rec)
        assert any("has changed since the run" in p
                   for p in self.load(mod, exp_dir))

    def test_a_missing_child_file_is_rejected(self, mod, tmp_path):
        exp_dir, _, _ = self.build(mod, tmp_path)
        (exp_dir / "r3.json").unlink()
        assert any("is missing" in p for p in self.load(mod, exp_dir))

    def test_an_edited_plan_digest_is_rejected(self, mod, tmp_path):
        exp_dir, plan, _ = self.build(mod, tmp_path)
        doc = json.loads((exp_dir / "plan.json").read_text())
        doc["plan_digest"] = "0" * 64
        self.rewrite(exp_dir / "plan.json", doc)
        problems = self.load(mod, exp_dir)
        assert any("own digest does not match" in p for p in problems)

    def test_a_reordered_plan_is_rejected(self, mod, tmp_path):
        exp_dir, plan, _ = self.build(mod, tmp_path)
        doc = json.loads((exp_dir / "plan.json").read_text())
        doc["plan"] = list(reversed(doc["plan"]))
        self.rewrite(exp_dir / "plan.json", doc)
        assert self.load(mod, exp_dir)

    def test_a_child_at_a_position_it_did_not_hold_is_rejected(self, mod,
                                                               tmp_path):
        exp_dir, _, _ = self.build(mod, tmp_path)
        agg = json.loads((exp_dir / "aggregate.json").read_text())
        agg["children"][1]["condition"] = "continuous"
        self.rewrite(exp_dir / "aggregate.json", agg)
        assert any("but the plan says" in p for p in self.load(mod, exp_dir))

    def test_a_complete_flag_that_does_not_follow_is_rejected(self, mod,
                                                              tmp_path):
        exp_dir, _, _ = self.build(mod, tmp_path, n=1)
        agg = json.loads((exp_dir / "aggregate.json").read_text())
        agg["complete"] = True                    # only one of four ran
        self.rewrite(exp_dir / "aggregate.json", agg)
        assert any("recorded imply" in p for p in self.load(mod, exp_dir))

    def test_a_run_that_never_started_may_not_reference_a_report(self, mod,
                                                                 tmp_path):
        exp_dir, _, _ = self.build(mod, tmp_path, n=1)
        agg = json.loads((exp_dir / "aggregate.json").read_text())
        agg["children"][0]["not_run"] = True
        self.rewrite(exp_dir / "aggregate.json", agg)
        assert any("never started" in p for p in self.load(mod, exp_dir))

    def test_load_experiment_refuses_a_tampered_directory(self, mod, tmp_path):
        exp_dir, _, _ = self.build(mod, tmp_path)
        doc = json.loads((exp_dir / "plan.json").read_text())
        doc["plan_digest"] = "0" * 64
        self.rewrite(exp_dir / "plan.json", doc)
        with pytest.raises(SystemExit, match="cannot replay"):
            mod.load_experiment(exp_dir)


class TestTheRealExperimentReplays:
    """The run that actually happened must pass its own gate."""

    def exp_dir(self):
        return ROOT / "data" / "reports" / "15_mps_order" / "exp001"

    @pytest.mark.skipif(
        not (ROOT / "data" / "reports" / "15_mps_order" / "exp001"
             / "aggregate.json").exists(),
        reason=f"{ARTIFACT_ONLY} exp001's evidence is not in this tree")
    def test_exp001_validates(self, mod):
        exp = mod.load_experiment(self.exp_dir())
        assert exp["experiment_id"] == "exp001"
        assert exp["complete"] is False

    @pytest.mark.skipif(
        not (ROOT / "data" / "reports" / "15_mps_order" / "exp001"
             / "r1.json").exists(),
        reason=f"{ARTIFACT_ONLY} exp001's evidence is not in this tree")
    def test_r1_kept_its_clears_apart(self, mod):
        rec = json.loads((self.exp_dir() / "r1.json").read_text())
        assert rec["condition"] == "continuous"
        assert rec["scheduled_empty_cache_calls"] == 0
        assert rec["teardown_empty_cache_calls"] == 1
        assert rec["teardown_empty_cache_seconds"] > 0
        assert rec["loss_decimals_stored"] is None

    @pytest.mark.skipif(
        not (ROOT / "data" / "reports" / "15_mps_order" / "exp001"
             / "r1.json").exists(),
        reason=f"{ARTIFACT_ONLY} exp001's evidence is not in this tree")
    def test_r1_stored_losses_unrounded(self, mod):
        rec = json.loads((self.exp_dir() / "r1.json").read_text())
        losses = [r["loss"] for r in rec["per_row"]]
        assert any(round(v, 4) != v for v in losses)


class TestCalibrationTampering:
    """The thresholds are recomputed from the samples, not read back."""

    def test_a_clean_calibration_replays(self, mod):
        out = mod.replay_calibration(calibration_doc(mod))
        assert out["problems"] == []
        assert out["thresholds"]["swap_used_gb"] == pytest.approx(1.25)

    def test_a_lowered_threshold_is_caught(self, mod):
        """A gate nobody could pass, edited into one anybody can."""
        calib = calibration_doc(mod)
        calib["thresholds"]["swap_used_gb"] = 99.0
        problems = mod.replay_calibration(calib)["problems"]
        assert any("do not follow from the stored samples" in p
                   for p in problems)

    def test_editing_the_stats_is_caught(self, mod):
        calib = calibration_doc(mod)
        calib["stats"]["swap_used_gb"]["median"] = 50.0
        assert any("statistics do not follow" in p
                   for p in mod.replay_calibration(calib)["problems"])

    def test_rewriting_the_samples_is_caught(self, mod):
        calib = calibration_doc(mod)
        calib["samples"] = [idle(swap=40.0) for _ in range(10)]
        problems = mod.replay_calibration(calib)["problems"]
        assert any("statistics do not follow" in p for p in problems)
        assert any("thresholds do not follow" in p for p in problems)
        assert any("digest does not match" in p for p in problems)

    def test_one_edited_sample_moves_no_threshold_and_no_verdict(self, mod):
        """Written down rather than assumed. The median and the MAD are robust
        by construction, so editing one sample changes no threshold -- this
        check cannot see it, and for exactly the same reason it changes no gate
        verdict either. What the digest pins is the stats, the thresholds and
        the policy, not the raw samples."""
        calib = calibration_doc(mod)
        calib["samples"][0]["swap_used_gb"] = 40.0
        out = mod.replay_calibration(calib)
        assert out["problems"] == []
        assert out["thresholds"]["swap_used_gb"] == pytest.approx(1.25)

    def test_a_calibration_with_no_samples_cannot_be_rechecked(self, mod):
        calib = calibration_doc(mod)
        calib["samples"] = []
        assert any("stores no samples" in p
                   for p in mod.replay_calibration(calib)["problems"])

    def test_a_calibration_that_loaded_a_model_is_refused(self, mod):
        calib = calibration_doc(mod)
        calib["loads_model"] = True
        assert any("loaded no model" in p
                   for p in mod.replay_calibration(calib)["problems"])

    def test_a_widened_gate_policy_changes_the_digest(self, mod):
        calib = calibration_doc(mod)
        calib["gate"]["consecutive_passes_required"] = 1
        assert any("digest does not match" in p
                   for p in mod.replay_calibration(calib)["problems"])

    def test_a_nonsense_gate_policy_is_refused(self, mod):
        calib = calibration_doc(mod, gate={"poll_seconds": 30,
                                           "max_wait_seconds": 900,
                                           "consecutive_passes_required": 0})
        assert any("not a positive integer" in p
                   for p in mod.replay_calibration(calib)["problems"])

    def test_the_experiment_and_the_plan_must_name_one_calibration(self, mod):
        calib = calibration_doc(mod)
        agg = {"calibration_digest": "0" * 64,
               "thresholds": calib["thresholds"], "gate_policy": calib["gate"]}
        plan_doc = {"calibration_digest": calib["calibration_digest"]}
        problems = mod.check_thresholds_agree(agg, plan_doc, calib,
                                              mod.replay_calibration(calib))
        assert any("aggregate was written against calibration" in p
                   for p in problems)


class TestPollTampering:
    """A gate verdict is recomputed from the readings stored beside it."""

    def thresholds(self, mod):
        return calibration_doc(mod)["thresholds"]

    def slot(self, gate):
        return {"run_id": "r1", "gate": gate,
                "recovery_passed_recomputed": bool(gate.get("passed")),
                "not_run": not gate.get("passed")}

    def test_a_clean_gate_replays(self, mod):
        t = self.thresholds(mod)
        out = mod.replay_gate(self.slot(passing_gate(t)), t, GATE_POLICY)
        assert out["passed"] and out["problems"] == []

    def test_a_flipped_verdict_is_caught(self, mod):
        t = self.thresholds(mod)
        gate = failing_gate(t)
        gate["polls"][0]["passed"] = True
        assert any("evaluate to False" in p for p in
                   mod.replay_gate(self.slot(gate), t, GATE_POLICY)["problems"])

    def test_an_inflated_streak_is_caught(self, mod):
        t = self.thresholds(mod)
        gate = failing_gate(t)
        gate["polls"][0]["consecutive_passes"] = 3
        assert any("recomputed 0" in p for p in
                   mod.replay_gate(self.slot(gate), t, GATE_POLICY)["problems"])

    def test_a_gate_that_says_it_passed_but_did_not(self, mod):
        t = self.thresholds(mod)
        gate = failing_gate(t)
        gate["passed"] = True
        gate["reason"] = None
        slot = {"run_id": "r1", "gate": gate,
                "recovery_passed_recomputed": True, "not_run": False}
        assert any("replaying its polls gives False" in p
                   for p in mod.replay_gate(slot, t, GATE_POLICY)["problems"])

    def test_a_readingless_poll_cannot_be_rechecked(self, mod):
        t = self.thresholds(mod)
        gate = passing_gate(t)
        del gate["polls"][1]["sample"]
        problems = mod.replay_gate(self.slot(gate), t, GATE_POLICY)["problems"]
        assert any("kept no readings" in p for p in problems)

    def test_polls_closer_together_than_the_policy_sleeps(self, mod):
        """Fabricated polls are cheap; waiting 30s between them is not."""
        t = self.thresholds(mod)
        gate = passing_gate(t)
        gate["polls"][1]["elapsed_seconds"] = 0.02
        gate["polls"][2]["elapsed_seconds"] = 0.03
        gate["waited_seconds"] = 0.03
        assert any("sooner than the 30s" in p for p in
                   mod.replay_gate(self.slot(gate), t, GATE_POLICY)["problems"])

    def test_polling_on_after_the_gate_released_is_caught(self, mod):
        t = self.thresholds(mod)
        gate = gate_record(gate_polls([idle()] * 5, t))
        assert any("kept polling after" in p for p in
                   mod.replay_gate(self.slot(gate), t, GATE_POLICY)["problems"])

    def test_giving_up_before_the_deadline_is_caught(self, mod):
        t = self.thresholds(mod)
        gate = gate_record(gate_polls([idle(swap=99.0)] * 4, t))
        assert any("without either passing" in p for p in
                   mod.replay_gate(self.slot(gate), t, GATE_POLICY)["problems"])

    @pytest.mark.parametrize("key", ["passed", "polls", "waited_seconds",
                                     "consecutive_passes_required", "reason"])
    def test_a_missing_gate_field_fails_closed(self, mod, key):
        t = self.thresholds(mod)
        gate = passing_gate(t)
        del gate[key]
        problems = mod.replay_gate(self.slot(gate), t, GATE_POLICY)["problems"]
        assert any(key in p for p in problems)

    def test_a_missing_recovery_field_fails_closed(self, mod):
        t = self.thresholds(mod)
        slot = {"run_id": "r1", "gate": passing_gate(t), "not_run": False}
        problems = mod.replay_gate(slot, t, GATE_POLICY)["problems"]
        assert any("recovery_passed_recomputed" in p for p in problems)

    def test_a_run_with_no_gate_at_all_fails_closed(self, mod):
        t = self.thresholds(mod)
        out = mod.replay_gate({"run_id": "r1"}, t, GATE_POLICY)
        assert out["passed"] is None and out["problems"]


class TestChildPreflightIsJudgedToo:
    def test_a_child_that_started_outside_the_band_is_caught(self, mod):
        calib = calibration_doc(mod)
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = [child_record(mod, e, digest) for e in plan]
        recs[1]["preflight"] = idle(swap=99.0)
        exp = experiment(mod, records=recs, calib=calib)
        assert any("the gate would have refused" in p
                   for p in exp["replay_problems"])

    def test_a_child_with_no_preflight_reading_is_caught(self, mod):
        calib = calibration_doc(mod)
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = [child_record(mod, e, digest) for e in plan]
        del recs[0]["preflight"]
        exp = experiment(mod, records=recs, calib=calib)
        assert any("no preflight reading of its own" in p
                   for p in exp["replay_problems"])


class TestCodeAndConfigDrift:
    """Only the condition may differ between runs."""

    def drifted(self, mod, field, value, *, in_provenance=True):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = [child_record(mod, e, digest) for e in plan]
        if in_provenance:
            recs[2]["provenance"][field] = value
        else:
            recs[2][field] = value
        return experiment(mod, records=recs)

    @pytest.mark.parametrize("field,value", [
        ("code_sha256", {"scripts/15_mps_order.py": "9" * 64}),
        ("head", "9" * 40),
        ("working_tree_dirty", True),
        ("lora_config", {"rank": 32, "seed": 0}),
        ("optimizer", {"class": "AdamW", "lr": 0.002}),
        ("packages", {"torch": "2.14.0"}),
        ("device", "cpu"),
        ("dtype", "float32"),
        ("instruction_sha256", {"instruct_inv_train.jsonl": "9" * 64}),
        ("selection_digest", "9" * 64),
        ("training_order_digest", "9" * 64),
        ("stop_conditions", {"slow_row_seconds": 60.0, "slow_row_streak": 3,
                             "max_seconds": 1800}),
        ("measurement_intervals", {"window": 50, "memory_sample_every": 5,
                                   "empty_cache_every": 10, "max_rows": 200}),
        ("base_revision", "9" * 40),
        ("published_adapter_revision", "9" * 40),
        ("tokenizer_revision", "9" * 40),
        ("phases", ["forward"]),
    ])
    def test_a_drifted_field_is_reported_and_blocks_the_headline(
            self, mod, field, value):
        exp = self.drifted(mod, field, value)
        assert field in exp["cross_child_comparison"]["disagreements"]
        assert not exp["verdicts"]["r3"]["eligible_for_paired_contrast"]
        assert not exp["headline"]["allowed"]

    def test_a_drifted_input_order_is_reported(self, mod):
        exp = self.drifted(mod, "input_order_digest", "9" * 64,
                           in_provenance=False)
        assert "input_order_digest" in exp["cross_child_comparison"][
            "disagreements"]

    def test_the_condition_itself_is_allowed_to_differ(self, mod):
        """The arms differ by construction; that is not drift."""
        exp = experiment(mod)
        assert exp["cross_child_comparison"]["disagreements"] == []
        assert exp["headline"]["allowed"]

    def test_a_short_run_did_not_complete_the_declared_order(self, mod):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = [child_record(mod, e, digest) for e in plan]
        recs[1] = child_record(mod, plan[1], digest, rows=190)
        exp = experiment(mod, records=recs)
        assert not exp["cross_child_comparison"][
            "completed_declared_order"]["r2"]["matches"]
        assert any("full declared input order" in r
                   for r in exp["verdicts"]["r2"]["reasons"])

    def test_the_code_that_ran_is_checked_against_the_tree(self, mod):
        exp = experiment(mod)
        check = exp["code_on_disk"]["r1"]
        # The fixture's digests are invented, so nothing on disk matches them.
        assert check["source_preserved"] is False
        assert check["changed_since_the_run"] or check["missing_from_the_tree"]


class TestTreatmentContract:
    """Each arm did exactly what its condition says, on the rows it says."""

    def rec(self, mod, index=1, **over):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        return child_record(mod, plan[index], digest, **over)

    def test_a_clean_treatment_arm_passes(self, mod):
        out = mod.check_treatment_contract(self.rec(mod), "r2")
        assert out["problems"] == []
        assert out["clears_observed_at"] == out["clears_expected_at"]
        assert out["clear_calls"] == 20

    def test_a_clean_control_arm_passes(self, mod):
        out = mod.check_treatment_contract(self.rec(mod, index=0), "r1")
        assert out["problems"] == [] and out["clear_calls"] == 0

    def test_a_clear_on_the_wrong_row_is_caught(self, mod):
        rec = self.rec(mod)
        rec["per_row"][9]["scheduled_empty_cache_seconds"] = 0.0
        rec["per_row"][10]["scheduled_empty_cache_seconds"] = 0.05
        problems = mod.check_treatment_contract(rec, "r2")["problems"]
        assert any("unscheduled at [11]" in p for p in problems)
        assert any("missing at [10]" in p for p in problems)

    def test_an_extra_clear_is_caught(self, mod):
        rec = self.rec(mod)
        rec["per_row"][4]["scheduled_empty_cache_seconds"] = 0.05
        assert any("unscheduled at [5]" in p
                   for p in mod.check_treatment_contract(rec, "r2")["problems"])

    def test_a_control_arm_that_cleared_once_is_caught(self, mod):
        rec = self.rec(mod, index=0)
        rec["per_row"][9]["scheduled_empty_cache_seconds"] = 0.05
        assert any("control arm cleared at rows" in p
                   for p in mod.check_treatment_contract(rec, "r1")["problems"])

    def test_a_schedule_other_than_every_ten_rows_is_caught(self, mod):
        rec = self.rec(mod)
        rec["scheduled_empty_cache_every"] = 20
        assert any("not every 10" in p
                   for p in mod.check_treatment_contract(rec, "r2")["problems"])

    def test_per_call_costs_that_do_not_sum_to_the_total(self, mod):
        rec = self.rec(mod)
        rec["scheduled_empty_cache_cost"]["total_seconds"] = 5.0
        assert any("per-call clear costs sum to" in p
                   for p in mod.check_treatment_contract(rec, "r2")["problems"])

    def test_a_missing_per_call_cost_is_caught(self, mod):
        rec = self.rec(mod)
        rec["scheduled_empty_cache_cost"]["per_call_seconds"].pop()
        assert any("per-call clear costs for" in p
                   for p in mod.check_treatment_contract(rec, "r2")["problems"])

    def test_a_teardown_folded_into_the_condition_clock_is_caught(self, mod):
        """If the teardown were inside, the row spans would not add up."""
        rec = self.rec(mod, index=0)
        rec["end_to_end_seconds"] = round(rec["end_to_end_seconds"] + 0.02, 4)
        rec["between_row_overhead_seconds"] = round(
            rec["between_row_overhead_seconds"] + 0.02, 4)
        rec["between_row_overhead_breakdown"]["unattributed_seconds"] = round(
            rec["between_row_overhead_breakdown"]["unattributed_seconds"]
            + 0.02, 4)
        rec["teardown_empty_cache_seconds"] = 0.01
        assert any("teardown cannot be shown to be outside" in p
                   for p in mod.check_treatment_contract(rec, "r1")["problems"])

    def test_a_teardown_counted_in_the_overhead_breakdown_is_caught(self, mod):
        rec = self.rec(mod, index=0)
        rec["between_row_overhead_breakdown"]["teardown_seconds"] = 0.02
        assert any("counts a teardown clear" in p
                   for p in mod.check_treatment_contract(rec, "r1")["problems"])

    def test_more_than_one_teardown_is_caught(self, mod):
        rec = self.rec(mod, index=0)
        rec["teardown_empty_cache_calls"] = 2
        assert any("teardown clears, expected exactly one" in p
                   for p in mod.check_treatment_contract(rec, "r1")["problems"])

    def test_an_overhead_that_does_not_add_up_is_caught(self, mod):
        rec = self.rec(mod, index=0)
        rec["between_row_overhead_breakdown"]["memory_probe_seconds"] = 0.5
        assert any("overhead breakdown sums to" in p
                   for p in mod.check_treatment_contract(rec, "r1")["problems"])

    def test_a_contract_failure_blocks_the_headline(self, mod):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = [child_record(mod, e, digest) for e in plan]
        recs[1]["per_row"][4]["scheduled_empty_cache_seconds"] = 0.05
        exp = experiment(mod, records=recs)
        assert not exp["headline"]["allowed"]
        assert any("treatment contract" in r for r in exp["headline"]["reasons"])


class TestHeadlineGate:
    """A headline needs every precondition, not a majority of them."""

    def consistent(self, mod, **kw):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = []
        for e in plan:
            r = child_record(mod, e, digest)
            r["windows"][0]["seconds_per_row"] = (
                1.0 if e["condition"] == "empty_cache" else 5.0)
            recs.append(r)
        return experiment(mod, records=recs, **kw)

    def test_a_clean_consistent_experiment_is_allowed(self, mod):
        exp = self.consistent(mod)
        assert exp["headline"]["allowed"]
        assert exp["headline"]["direction"] == "empty_cache faster"

    def test_an_incomplete_experiment_is_not(self, mod):
        exp = self.consistent(mod, exit_codes=(0, 0, 0, 1))
        assert not exp["headline"]["allowed"]

    def test_divergent_losses_block_it(self, mod):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = []
        for e in plan:
            r = child_record(mod, e, digest,
                             loss_base=0.9 if e["run_id"] == "r4" else 0.5)
            r["windows"][0]["seconds_per_row"] = (
                1.0 if e["condition"] == "empty_cache" else 5.0)
            recs.append(r)
        exp = experiment(mod, records=recs)
        assert not exp["headline"]["allowed"]
        assert any("loss verdict is over_tolerance" in r
                   for r in exp["headline"]["reasons"])

    def test_token_mismatches_block_it(self, mod):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = []
        for e in plan:
            r = child_record(mod, e, digest)
            r["windows"][0]["seconds_per_row"] = (
                1.0 if e["condition"] == "empty_cache" else 5.0)
            recs.append(r)
        recs[1]["per_row"][0]["n_supervised"] = 999
        exp = experiment(mod, records=recs)
        assert not exp["headline"]["allowed"]
        assert any("token" in r for r in exp["headline"]["reasons"])

    def test_a_short_pair_blocks_it(self, mod):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = []
        for e in plan:
            r = child_record(mod, e, digest,
                             rows=199 if e["run_id"] == "r3" else 200)
            r["windows"][0]["seconds_per_row"] = (
                1.0 if e["condition"] == "empty_cache" else 5.0)
            recs.append(r)
        exp = experiment(mod, records=recs)
        assert not exp["headline"]["allowed"]
        assert any("shares all 200 rows" in r or "eligible" in r
                   for r in exp["headline"]["reasons"])

    def test_disagreeing_blocks_block_it(self, mod):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = []
        for e in plan:
            r = child_record(mod, e, digest)
            r["windows"][0]["seconds_per_row"] = (
                1.0 if e["run_id"] in ("r2", "r4") else 5.0)
            recs.append(r)
        exp = experiment(mod, records=recs)
        assert not exp["headline"]["allowed"]
        assert any("do not agree on the direction" in r
                   for r in exp["headline"]["reasons"])

    def test_the_report_prints_what_failed(self, mod, tmp_path):
        exp = self.consistent(mod, exit_codes=(0, 0, 0, 1))
        out = tmp_path / "m.md"
        mod._write_markdown(exp, out)
        text = out.read_text()
        assert "A headline is allowed only when all of these hold" in text
        assert "**Not allowed here.** What failed:" in text
        assert "repeatedly observed engineering mitigation" not in text


class TestStoredLossContradiction:
    """An aggregate that disagrees with itself about what it compared."""

    def test_a_verdict_with_no_pairs_is_rejected(self, mod):
        agg = {"losses": {"pairs": [], "comparable_pairs": 0,
                          "verdict": "within_tolerance",
                          "max_abs_loss_diff_overall": None}}
        assert any("no comparable pair yet records a loss verdict" in p
                   for p in mod._check_stored_losses(agg))

    def test_a_pair_count_that_does_not_match_is_rejected(self, mod):
        agg = {"losses": {"pairs": [], "comparable_pairs": 6,
                          "verdict": "not_applicable",
                          "max_abs_loss_diff_overall": None}}
        assert any("counts 6 comparable pairs but stores 0" in p
                   for p in mod._check_stored_losses(agg))

    def test_a_difference_with_no_pairs_is_rejected(self, mod):
        agg = {"losses": {"pairs": [], "comparable_pairs": 0,
                          "verdict": "not_applicable",
                          "max_abs_loss_diff_overall": 0.4}}
        assert any("largest loss difference" in p
                   for p in mod._check_stored_losses(agg))

    def test_pairs_with_no_verdict_are_rejected(self, mod):
        agg = {"losses": {"pairs": [{"a": "r1", "b": "r2"}],
                          "comparable_pairs": 1, "verdict": "not_applicable"}}
        assert any("yet records no loss verdict" in p
                   for p in mod._check_stored_losses(agg))

    def test_the_older_two_state_field_is_superseded_not_fatal(self, mod):
        """exp001 was written before the three-state verdict existed."""
        agg = {"losses": {"pairs": [], "within_tolerance": False,
                          "max_abs_loss_diff_overall": None}}
        assert mod._check_stored_losses(agg) == []
        notes = mod.stored_conclusion_notes(agg)
        assert any("older two-state loss field" in n for n in notes)
        assert any("compared and failed" in n for n in notes)


class TestAtomicPublish:
    """The publish primitive has to be atomic *and* unable to clobber."""

    def test_it_writes_and_digests(self, tmp_path):
        sha = write_once_json(tmp_path / "a.json", {"a": 1})
        assert json.loads((tmp_path / "a.json").read_text()) == {"a": 1}
        assert len(sha) == 64

    def test_it_refuses_an_existing_name(self, tmp_path):
        write_once_json(tmp_path / "a.json", {"a": 1})
        with pytest.raises(SystemExit, match="refusing to overwrite"):
            write_once_json(tmp_path / "a.json", {"a": 2})
        assert json.loads((tmp_path / "a.json").read_text()) == {"a": 1}

    def test_it_leaves_no_temporary_behind(self, tmp_path):
        write_once_json(tmp_path / "a.json", {"a": 1})
        with pytest.raises(SystemExit):
            write_once_json(tmp_path / "a.json", {"a": 2})
        assert [p.name for p in tmp_path.iterdir()] == ["a.json"]

    def test_temporaries_are_unique_per_call(self, tmp_path, monkeypatch):
        """Two writers must not share one scratch name and corrupt each other."""
        import src.training.session as session_mod

        seen = []
        real = session_mod.tempfile.mkstemp

        def spy(**kw):
            fd, path = real(**kw)
            seen.append(path)
            return fd, path

        monkeypatch.setattr(session_mod.tempfile, "mkstemp", spy)
        write_once_json(tmp_path / "a.json", {"a": 1})
        write_once_json(tmp_path / "b.json", {"b": 2})
        assert len(set(seen)) == 2

    def test_a_link_failure_refuses_rather_than_racing(self, tmp_path,
                                                       monkeypatch):
        import src.training.session as session_mod

        def no_link(src, dst):
            raise OSError("no hard links here")

        monkeypatch.setattr(session_mod.os, "link", no_link)
        with pytest.raises(SystemExit, match="does not support hard links"):
            write_once_json(tmp_path / "a.json", {"a": 1})
        assert not (tmp_path / "a.json").exists()


class TestExclusiveLock:
    def test_a_second_holder_is_refused(self, tmp_path):
        """flock is held by the open file description, so this is exactly what
        a second process sees -- no second process needed to prove it."""
        import fcntl
        import os as os_mod

        path = tmp_path / ".lock"
        with exclusive_lock(path, description="session x"):
            fd = os_mod.open(str(path), os_mod.O_CREAT | os_mod.O_RDWR)
            try:
                with pytest.raises(OSError):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os_mod.close(fd)

    def test_the_lock_is_released_afterwards(self, tmp_path):
        path = tmp_path / ".lock"
        with exclusive_lock(path, description="x"):
            pass
        with exclusive_lock(path, description="x"):
            pass

    def test_it_is_released_even_when_the_body_raises(self, tmp_path):
        path = tmp_path / ".lock"
        with pytest.raises(ValueError):
            with exclusive_lock(path, description="x"):
                raise ValueError("boom")
        with exclusive_lock(path, description="x"):
            pass


class TestBootIdentity:
    """One measured run per boot, without writing down the boot."""

    RAW_UUID = "9DC3EF11-DEAD-BEEF-8199-C5CAC1023047"

    def sysctl(self, uuid=None, boottime=None):
        return lambda n: (uuid if n == "kern.bootsessionuuid"
                          else boottime if n == "kern.boottime" else None)

    def test_the_raw_uuid_never_leaves_the_function(self):
        got = boot_identity("exp", sysctl=self.sysctl(uuid=self.RAW_UUID))
        assert self.RAW_UUID not in json.dumps(got)
        assert got["source"] == "kern.bootsessionuuid"
        assert len(got["boot_fingerprint"]) == 32

    def test_boot_time_is_the_fallback(self):
        got = boot_identity("exp", sysctl=self.sysctl(
            boottime="{ sec = 1786552556, usec = 746069 }"))
        assert got["source"] == "kern.boottime"
        assert "1786552556" not in json.dumps(got)

    def test_the_same_boot_gives_the_same_fingerprint(self):
        s = self.sysctl(uuid=self.RAW_UUID)
        assert (boot_identity("exp", sysctl=s)["boot_fingerprint"]
                == boot_identity("exp", sysctl=s)["boot_fingerprint"])

    def test_a_different_boot_gives_a_different_fingerprint(self):
        a = boot_identity("exp", sysctl=self.sysctl(uuid="A"))
        b = boot_identity("exp", sysctl=self.sysctl(uuid="B"))
        assert a["boot_fingerprint"] != b["boot_fingerprint"]

    def test_experiments_cannot_line_their_fingerprints_up(self):
        """Domain separation: the same boot reads differently per experiment."""
        s = self.sysctl(uuid=self.RAW_UUID)
        assert (boot_identity("exp1", sysctl=s)["boot_fingerprint"]
                != boot_identity("exp2", sysctl=s)["boot_fingerprint"])

    def test_an_unidentifiable_boot_says_so_rather_than_guessing(self):
        got = boot_identity("exp", sysctl=self.sysctl())
        assert got["boot_fingerprint"] is None
        assert "cannot be told apart" in got["reason"]


class TestSourceSnapshot:
    def tree(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "a.py").write_text("print('a')\n")
        (tmp_path / "src" / "b.py").write_text("print('b')\n")
        return ("a.py", "src/b.py")

    def test_it_copies_and_digests_every_file(self, tmp_path):
        files = self.tree(tmp_path)
        m = snapshot_sources(tmp_path, files, tmp_path / "snap")
        assert set(m["files"]) == set(files)
        assert (tmp_path / "snap" / "src__b.py").read_text() == "print('b')\n"
        assert verify_sources(tmp_path, m, tmp_path / "snap") == []

    def test_the_manifest_digest_covers_every_file(self, tmp_path):
        files = self.tree(tmp_path)
        m = snapshot_sources(tmp_path, files, tmp_path / "snap")
        before = manifest_digest(m)
        m["files"]["a.py"]["sha256"] = "9" * 64
        assert manifest_digest(m) != before

    def test_a_changed_working_file_is_caught(self, tmp_path):
        files = self.tree(tmp_path)
        m = snapshot_sources(tmp_path, files, tmp_path / "snap")
        (tmp_path / "a.py").write_text("print('edited')\n")
        assert any("has changed since the plan was made" in p
                   for p in verify_sources(tmp_path, m, tmp_path / "snap"))

    def test_a_deleted_working_file_is_caught(self, tmp_path):
        files = self.tree(tmp_path)
        m = snapshot_sources(tmp_path, files, tmp_path / "snap")
        (tmp_path / "src" / "b.py").unlink()
        assert any("missing from the working tree" in p
                   for p in verify_sources(tmp_path, m, tmp_path / "snap"))

    def test_editing_the_snapshot_to_match_a_drifted_file_is_caught(self,
                                                                    tmp_path):
        """Both sides are checked, so moving the goalposts moves both."""
        files = self.tree(tmp_path)
        m = snapshot_sources(tmp_path, files, tmp_path / "snap")
        (tmp_path / "a.py").write_text("print('edited')\n")
        (tmp_path / "snap" / "a.py").unlink()
        (tmp_path / "snap" / "a.py").write_text("print('edited')\n")
        problems = verify_sources(tmp_path, m, tmp_path / "snap")
        assert any("has changed since the plan was made" in p
                   for p in problems)
        assert any("snapshot copy does not match" in p for p in problems)

    def test_a_missing_snapshot_copy_is_caught(self, tmp_path):
        files = self.tree(tmp_path)
        m = snapshot_sources(tmp_path, files, tmp_path / "snap")
        (tmp_path / "snap" / "a.py").unlink()
        assert any("snapshot copy is missing" in p
                   for p in verify_sources(tmp_path, m, tmp_path / "snap"))

    def test_a_snapshot_is_never_written_twice(self, tmp_path):
        files = self.tree(tmp_path)
        snapshot_sources(tmp_path, files, tmp_path / "snap")
        with pytest.raises(SystemExit, match="refusing to overwrite"):
            snapshot_sources(tmp_path, files, tmp_path / "snap")

    def test_an_empty_manifest_fails_closed(self, tmp_path):
        assert verify_sources(tmp_path, {}, tmp_path) == [
            "source manifest records no files"]


# ------------------------------------------------- the one-run-per-boot flow

class FakeChild:
    """Stands in for the spawned child. Loads nothing, writes what it is told."""

    def __init__(self, mod, plan, digest, source_check, *, returncode=0,
                 write=True, timeout=False, mutate=None):
        self.mod, self.plan, self.digest = mod, plan, digest
        self.source_check = source_check
        self.returncode, self.write = returncode, write
        self.timeout, self.mutate = timeout, mutate
        self.commands = []

    def __call__(self, cmd, **kw):
        import subprocess as sp
        import types

        self.commands.append(cmd)
        if self.timeout:
            raise sp.TimeoutExpired(cmd, 1)
        if self.write:
            run_id = cmd[cmd.index("--run-id") + 1]
            entry = next(p for p in self.plan if p["run_id"] == run_id)
            out = Path(cmd[cmd.index("--out") + 1])
            rec = child_record(self.mod, entry, self.digest)
            # What the real child records after checking its own source.
            rec["child_source_check"] = dict(self.source_check)
            if self.mutate:
                self.mutate(rec)
            self.mod.atomic_write_json(out, rec)
        return types.SimpleNamespace(returncode=self.returncode)


class SessionHarness:
    """A session on a throwaway tree, driven without loading anything."""

    def __init__(self, mod, tmp_path, monkeypatch, *, experiment_id="boot001"):
        self.mod, self.root, self.mp = mod, tmp_path, monkeypatch
        self.id = experiment_id
        # A second harness may share the tree: two sessions on one machine is
        # the situation the identity chain exists for.
        (tmp_path / "src").mkdir(exist_ok=True)
        if not (tmp_path / "a.py").exists():
            (tmp_path / "a.py").write_text("print('a')\n")
            (tmp_path / "src" / "b.py").write_text("print('b')\n")
        monkeypatch.setattr(mod, "ROOT", tmp_path)
        monkeypatch.setattr(mod, "RUN_DIR", tmp_path / "runs")
        monkeypatch.setattr(mod, "REPORT_DIR", tmp_path / "reports")
        monkeypatch.setattr(mod, "CODE_FILES", ("a.py", "src/b.py"))
        self.calib_path = tmp_path / "calibration.json"
        if self.calib_path.exists():
            self.calib = json.loads(self.calib_path.read_text())
        else:
            self.calib = calibration_doc(mod)
            mod.atomic_write_json(self.calib_path, self.calib)
        self.boots = iter(f"BOOT{i:02d}" for i in range(1, 99))
        self.set_boot(next(self.boots))

    def set_boot(self, fingerprint):
        self.boot = fingerprint
        self.mp.setattr(self.mod, "boot_identity",
                        lambda experiment_id: {
                            "boot_fingerprint": self.boot,
                            "source": "kern.bootsessionuuid",
                            "detected_at": "now", "reason": None})

    def next_boot(self):
        self.set_boot(next(self.boots))

    def init(self):
        assert self.mod.run_session_init(self.id, self.calib_path) == 0
        self.paths = self.mod.session_paths(self.id)
        self.plan = json.loads(self.paths["plan"].read_text())["plan"]
        self.digest = self.mod.digest_obj(self.plan)
        self.session = json.loads(self.paths["session"].read_text())
        return self.paths

    def source_check(self):
        return {"files_verified": len(self.session["source_manifest"]["files"]),
                "source_manifest_digest": self.session[
                    "source_manifest_digest"],
                "verified_at": "2026-08-15T00:00:00+00:00"}

    def started_body(self, run_id, *, index=1, **over):
        """A well-formed started event, for tests that write one by hand."""
        entry = next(p for p in self.plan if p["run_id"] == run_id)
        inv = self.mod.child_invocation(self.paths, self.digest, entry)
        body = {"schema_version": self.mod.EXPERIMENT_SCHEMA,
                "kind": "session_event", "event": EVENT_STARTED,
                "index": index, "experiment_id": self.id,
                "plan_digest": self.digest,
                "boot_fingerprint": self.boot, "recorded_at": "t",
                **{k: entry[k] for k in ("run_id", "condition", "block_id",
                                         "position_in_block",
                                         "global_position")},
                "gate": passing_gate(self.calib["thresholds"]),
                "source_verified": {
                    "files": len(self.session["source_manifest"]["files"]),
                    "source_manifest_digest": self.session[
                        "source_manifest_digest"],
                    "checked_after_gate": True},
                "out_path": inv["out_rel"],
                "child_command_digest": inv["digest"]}
        body.update(over)
        return body

    def gate(self, *, passes=True, side_effect=None):
        thresholds = self.calib["thresholds"]

        def fake_gate(*a, **kw):
            if side_effect:
                side_effect()
            return passing_gate(thresholds) if passes else failing_gate(
                thresholds)

        self.mp.setattr(self.mod, "wait_for_recovery", fake_gate)

    def child(self, **kw):
        fake = FakeChild(self.mod, self.plan, self.digest, self.source_check(),
                         **kw)
        self.mp.setattr(self.mod.subprocess, "run", fake)
        return fake

    def run_one(self, *, gate_passes=True, side_effect=None, **child_kw):
        self.gate(passes=gate_passes, side_effect=side_effect)
        self.child(**child_kw)
        return self.mod.run_session_next(self.id)

    def events(self):
        return [e["file_name"] for e in read_events(self.paths["dir"])]


@pytest.fixture
def session(mod, tmp_path, monkeypatch):
    return SessionHarness(mod, tmp_path, monkeypatch)


class TestSessionLayout:
    def test_init_fixes_the_plan_the_thresholds_and_the_source(self, session):
        paths = session.init()
        plan_doc = json.loads(paths["plan"].read_text())
        doc = json.loads(paths["session"].read_text())
        assert [p["condition"] for p in plan_doc["plan"]] == [
            "continuous", "empty_cache", "empty_cache", "continuous"]
        assert doc["thresholds"] == session.calib["thresholds"]
        assert doc["one_run_per_boot"] is True
        assert set(doc["source_manifest"]["files"]) == {"a.py", "src/b.py"}
        assert doc["source_manifest_digest"] == manifest_digest(
            doc["source_manifest"])
        assert (paths["snapshot"] / "src__b.py").exists()

    def test_no_unevidenced_cold_start_claim(self, session):
        """"Nothing had loaded a model" is not something the tool can know."""
        doc = json.loads(session.init()["session"].read_text())
        assert "thresholds_are_cold_start" not in doc
        assert doc["thresholds_baseline"] == "fixed idle baseline"
        assert "not a fact this tool can establish" in doc[
            "thresholds_baseline_note"]

    def test_the_calibration_note_does_not_overclaim(self, mod):
        note = mod.run_calibration.__doc__ or ""
        assert "No model is loaded" in note
        src = SCRIPT.read_text()
        assert "NOT knowable from these readings" in src

    def test_a_calibration_given_as_a_relative_path_works(self, session):
        """The operator types a repo-relative path; the session stores a
        repo-relative path. Anchoring is this tool's job, not theirs."""
        assert session.mod.run_session_init(
            "boot_rel", Path("calibration.json")) == 0
        doc = json.loads(
            session.mod.session_paths("boot_rel")["session"].read_text())
        assert doc["calibration_path"] == "calibration.json"
        assert doc["calibration_sha256"] == session.mod.sha256_file(
            session.calib_path)
        assert session.mod._session_preconditions("boot_rel")["problems"] == []

    def test_a_calibration_given_as_an_absolute_path_works(self, session):
        assert session.mod.run_session_init(
            "boot_abs", session.calib_path) == 0
        doc = json.loads(
            session.mod.session_paths("boot_abs")["session"].read_text())
        assert doc["calibration_path"] == "calibration.json"

    def test_init_refuses_to_reopen_a_session(self, session):
        session.init()
        with pytest.raises(SystemExit, match="is an initialised session"):
            session.mod.run_session_init(session.id, session.calib_path)

    def test_a_half_written_init_is_named_as_such(self, session):
        """Laying out a session touches several files. If it dies between
        them, the operator needs to know whether what is left holds
        measurements or only copies of source."""
        paths = session.mod.session_paths("boot_partial")
        (paths["dir"] / "source_snapshot").mkdir(parents=True)
        with pytest.raises(SystemExit) as e:
            session.mod.run_session_init("boot_partial", session.calib_path)
        assert "did not finish" in str(e.value)
        assert "no measurement of any kind" in str(e.value)
        assert "never reopened" not in str(e.value)

    def test_init_refuses_a_calibration_that_does_not_replay(self, session):
        session.init()
        bad = dict(session.calib)
        bad["thresholds"] = {m: 99.0 for m in GATE_METRICS}
        path = session.root / "bad.json"
        session.mod.atomic_write_json(path, bad)
        with pytest.raises(SystemExit, match="cannot be used"):
            session.mod.run_session_init("boot002", path)

    def test_a_clean_session_is_ready(self, session):
        session.init()
        assert session.mod._session_preconditions(session.id)["problems"] == []


class TestPreconditionsRunBeforeTheGate:
    """Everything about how the session was laid out, checked before polling."""

    @pytest.fixture
    def watched(self, session):
        """A session whose gate records whether it was ever reached."""
        session.init()
        reached = []
        session.mp.setattr(session.mod, "wait_for_recovery",
                           lambda *a, **kw: reached.append(True) or
                           passing_gate(session.calib["thresholds"]))
        session.child()
        return session, reached

    def rewrite(self, path, mutate):
        doc = json.loads(path.read_text())
        mutate(doc)
        path.unlink()
        path.write_text(json.dumps(doc, indent=2))

    def expect(self, watched, path, mutate, message):
        session, reached = watched
        self.rewrite(path, mutate)
        assert any(message in p for p in
                   session.mod._session_preconditions(session.id)["problems"])
        with pytest.raises(SystemExit, match="cannot be read"):
            session.mod.run_session_next(session.id)
        assert reached == [], "the gate was polled despite a bad session"
        assert session.events() == []

    # ---- the documents are what they claim to be ------------------------
    @pytest.mark.parametrize("which", ["plan", "session"])
    def test_a_wrong_schema_is_refused(self, watched, which):
        session, _ = watched
        self.expect(watched, session.paths[which],
                    lambda d: d.update({"schema_version": 99}),
                    "records schema_version 99")

    @pytest.mark.parametrize("which", ["plan", "session"])
    def test_a_wrong_kind_is_refused(self, watched, which):
        session, _ = watched
        self.expect(watched, session.paths[which],
                    lambda d: d.update({"kind": "notes"}),
                    "is a 'notes' record")

    @pytest.mark.parametrize("which", ["plan", "session"])
    def test_another_experiments_file_is_refused(self, watched, which):
        session, _ = watched
        self.expect(watched, session.paths[which],
                    lambda d: d.update({"experiment_id": "somewhere_else"}),
                    "belongs to experiment 'somewhere_else'")

    @pytest.mark.parametrize("which", ["plan", "session"])
    def test_a_session_not_laid_out_for_one_run_per_boot_is_refused(
            self, watched, which):
        session, _ = watched
        self.expect(watched, session.paths[which],
                    lambda d: d.update({"one_run_per_boot": False}),
                    "records one_run_per_boot=False")

    @pytest.mark.parametrize("which", ["plan", "session"])
    def test_a_missing_one_run_per_boot_flag_is_refused(self, watched, which):
        session, _ = watched
        self.expect(watched, session.paths[which],
                    lambda d: d.pop("one_run_per_boot"),
                    "records one_run_per_boot=None")

    # ---- the calibration all three documents point at --------------------
    def test_a_plan_naming_another_calibration_is_refused(self, watched):
        session, _ = watched
        self.expect(watched, session.paths["plan"],
                    lambda d: d.update({"calibration_path": "other.json"}),
                    "the plan was laid out against calibration 'other.json'")

    def test_a_session_naming_a_missing_calibration_is_refused(self, watched):
        session, _ = watched
        self.expect(watched, session.paths["session"],
                    lambda d: d.update({"calibration_path": "gone.json"}),
                    "is missing")

    def test_disagreeing_calibration_digests_are_refused(self, watched):
        session, _ = watched
        self.expect(watched, session.paths["plan"],
                    lambda d: d.update({"calibration_digest": "9" * 64}),
                    "disagree on the calibration digest")

    def test_a_session_pinned_to_another_calibration_digest_is_refused(
            self, watched):
        session, _ = watched
        self.expect(
            watched, session.paths["session"],
            lambda d: d.update({"calibration_digest": "9" * 64}),
            "own digest is not the one the session was laid out against")

    def test_a_missing_calibration_digest_is_not_skipped(self, watched):
        session, _ = watched
        self.expect(watched, session.paths["session"],
                    lambda d: d.pop("calibration_sha256"),
                    "records no digest for the calibration file")

    # ---- the band the four runs are judged against -----------------------
    @pytest.mark.parametrize("field,value", [
        ("poll_seconds", 1),
        ("max_wait_seconds", 60),
        ("consecutive_passes_required", 1),
    ])
    def test_an_edited_gate_policy_field_is_refused(self, watched, field,
                                                    value):
        """How long to wait and how many passes to require are part of the
        band, not per-run settings."""
        session, _ = watched
        self.expect(watched, session.paths["session"],
                    lambda d: d["gate_policy"].update({field: value}),
                    "gate policy is not the calibration's")

    def test_edited_thresholds_are_refused(self, watched):
        session, _ = watched
        self.expect(watched, session.paths["session"],
                    lambda d: d["thresholds"].update({"swap_used_gb": 99.0}),
                    "not the ones this session was laid out with")

    def test_a_missing_gate_policy_is_refused(self, watched):
        session, _ = watched
        self.expect(watched, session.paths["session"],
                    lambda d: d.pop("gate_policy"),
                    "gate policy is not the calibration's")

    def test_an_edited_plan_is_refused(self, watched):
        session, _ = watched
        self.expect(watched, session.paths["plan"],
                    lambda d: d["plan"].reverse(),
                    "own digest does not match its plan")

    def test_a_clean_session_reaches_the_gate(self, watched):
        """The control: nothing edited, the gate is polled, the run happens."""
        session, reached = watched
        assert session.mod.run_session_next(session.id) == 0
        assert reached == [True]


class TestFinishedOutcomesAreAClosedSet:
    """Four outcomes, each cross-checked against the two facts behind it."""

    def finished_event(self, session):
        session.init()
        session.run_one()
        return (session.paths["dir"] / "events"
                / event_file_name(2, "r1", EVENT_FINISHED))

    def expect(self, session, path, mutate, message):
        body = json.loads(path.read_text())
        mutate(body)
        path.unlink()
        path.write_text(json.dumps(body, indent=2))
        with pytest.raises(SystemExit, match=message):
            session.mod.read_journal(session.id)

    def test_an_unknown_outcome_is_refused(self, session):
        p = self.finished_event(session)
        self.expect(session, p, lambda b: b.update({"outcome": "mostly_fine"}),
                    "outcome 'mostly_fine' is not one of")

    def test_a_null_outcome_is_refused(self, session):
        p = self.finished_event(session)
        self.expect(session, p, lambda b: b.update({"outcome": None}),
                    "is not one of")

    def test_completed_with_a_nonzero_exit_is_refused(self, session):
        p = self.finished_event(session)
        self.expect(session, p, lambda b: b.update({"exit_status": 1}),
                    "outcome says completed, exit status is 1")

    def test_completed_with_no_report_is_refused(self, session):
        p = self.finished_event(session)
        self.expect(session, p,
                    lambda b: b.update({"report_path": None,
                                        "report_sha256": None}),
                    "outcome says completed with no report")

    def test_no_report_beside_a_failed_exit_is_refused(self, session):
        """The case that would hide a crash as a missing file."""
        p = self.finished_event(session)
        self.expect(session, p,
                    lambda b: b.update({"outcome": "no_report",
                                        "exit_status": 1,
                                        "report_path": None,
                                        "report_sha256": None}),
                    "did not merely forget to write")

    def test_no_report_that_references_a_report_is_refused(self, session):
        p = self.finished_event(session)
        self.expect(session, p, lambda b: b.update({"outcome": "no_report"}),
                    "outcome says no report, yet one is referenced")

    def test_nonzero_exit_with_a_zero_status_is_refused(self, session):
        p = self.finished_event(session)
        self.expect(session, p,
                    lambda b: b.update({"outcome": "nonzero_exit"}),
                    "exited non-zero, exit status is 0")

    def test_nonzero_exit_with_no_status_is_refused(self, session):
        p = self.finished_event(session)
        self.expect(session, p,
                    lambda b: b.update({"outcome": "nonzero_exit",
                                        "exit_status": None}),
                    "exited non-zero, exit status is None")

    def test_timed_out_with_an_exit_status_is_refused(self, session):
        p = self.finished_event(session)
        self.expect(session, p, lambda b: b.update({"outcome": "timed_out"}),
                    "timed out, which leaves no exit status")

    def test_a_report_path_without_its_digest_is_refused(self, session):
        p = self.finished_event(session)
        self.expect(session, p, lambda b: b.update({"report_sha256": None}),
                    "records a report path with no digest")

    def test_a_report_digest_without_its_path_is_refused(self, session):
        p = self.finished_event(session)
        self.expect(session, p, lambda b: b.update({"report_path": None}),
                    "records a report digest with no path")

    def test_a_report_is_verified_whatever_the_outcome_says(self, session):
        """A run that timed out may still have written one, and it is checked."""
        paths = session.init()
        session.run_one()
        p = (paths["dir"] / "events"
             / event_file_name(2, "r1", EVENT_FINISHED))
        body = json.loads(p.read_text())
        body["outcome"] = "timed_out"
        body["exit_status"] = None
        p.unlink()
        p.write_text(json.dumps(body, indent=2))
        # The report is still there and still checked: edit it and the journal
        # notices, even though the outcome is no longer `completed`.
        rec = json.loads((paths["dir"] / "r1.json").read_text())
        rec["model_compute_seconds"] = 1.0
        (paths["dir"] / "r1.json").unlink()
        (paths["dir"] / "r1.json").write_text(json.dumps(rec, indent=2))
        with pytest.raises(SystemExit, match="has changed since the run"):
            session.mod.read_journal(session.id)

    @pytest.mark.parametrize("value", [False, True, 0.0, "0"])
    @pytest.mark.parametrize("outcome", ["completed", "no_report"])
    def test_a_success_exit_must_be_the_integer_zero(self, session, outcome,
                                                     value):
        """`false` and `0.0` compare equal to 0 in Python. An exit status is
        the small integer the kernel returned, not something that looks like
        it."""
        p = self.finished_event(session)
        empty = {"report_path": None, "report_sha256": None}
        self.expect(session, p,
                   lambda b: b.update({"exit_status": value, "outcome": outcome,
                                       **(empty if outcome == "no_report"
                                          else {})}),
                   "exit status is")

    def test_the_integer_zero_is_still_accepted(self, session):
        paths = session.init()
        assert session.run_one() == 0
        body = json.loads(
            (paths["dir"] / "events"
             / event_file_name(2, "r1", EVENT_FINISHED)).read_text())
        assert body["exit_status"] == 0
        assert type(body["exit_status"]) is int
        assert session.mod.read_journal(session.id)[1]["problems"] == []

    @pytest.mark.parametrize("value,ok", [
        (0, True), (False, False), (True, False), (0.0, False), ("0", False),
        (1, False), (None, False)])
    def test_the_clean_exit_predicate(self, mod, value, ok):
        assert mod._is_clean_exit(value) is ok

    def test_nonzero_and_timed_out_keep_their_meaning(self, session):
        p = self.finished_event(session)
        # A boolean is not a non-zero exit either.
        self.expect(session, p,
                    lambda b: b.update({"outcome": "nonzero_exit",
                                        "exit_status": True}),
                    "exited non-zero, exit status is True")

    def test_the_four_outcomes_are_the_ones_the_parent_writes(self, mod):
        assert set(mod.FINISHED_OUTCOMES) == {
            "completed", "nonzero_exit", "timed_out", "no_report"}
        assert set(mod.TERMINAL_OUTCOMES) - {"never_finished"} == (
            set(mod.FINISHED_OUTCOMES) - {"completed"})


class TestEveryDerivedFieldIsCompared:
    def test_the_inventory_covers_what_the_journal_derives(self, mod, session):
        """A field the derivation produces and nothing checks is a conclusion
        sitting in a stored file unread."""
        paths = session.init()
        session.run_one(returncode=1, write=False)
        ctx, state = mod.read_journal(session.id, check_working_tree=False)
        derived = mod.session_experiment_state(ctx, state)
        assert set(derived) == set(mod.SESSION_DERIVED_FIELDS)

    def test_an_edited_elapsed_time_is_refused(self, session):
        paths = session.init()
        session.run_one(returncode=1, write=False)
        agg = json.loads(paths["aggregate"].read_text())
        agg["elapsed_seconds"] = 999999.0
        paths["aggregate"].unlink()
        paths["aggregate"].write_text(json.dumps(agg, indent=2))
        with pytest.raises(SystemExit,
                           match="seconds elapsed, the journal's finished "
                                 "events sum to"):
            session.mod.load_experiment(paths["dir"])

    def test_a_small_wrong_elapsed_time_is_refused_too(self, session):
        """Not only implausible numbers: the comparison is exact."""
        paths = session.init()
        session.run_one()
        session.next_boot()
        session.run_one(returncode=1, write=False)
        agg = json.loads(paths["aggregate"].read_text())
        assert agg["elapsed_seconds"] == 0.0     # the fake child is instant
        agg["elapsed_seconds"] = 1.25
        paths["aggregate"].unlink()
        paths["aggregate"].write_text(json.dumps(agg, indent=2))
        with pytest.raises(SystemExit, match="seconds elapsed"):
            session.mod.load_experiment(paths["dir"])

    def test_an_unaccounted_derived_field_is_refused(self, session,
                                                     monkeypatch):
        paths = session.init()
        session.run_one(returncode=1, write=False)
        real = session.mod.session_experiment_state
        monkeypatch.setattr(
            session.mod, "session_experiment_state",
            lambda ctx, state: {**real(ctx, state), "something_new": 1})
        with pytest.raises(SystemExit,
                           match="derives fields nothing compares"):
            session.mod.load_experiment(paths["dir"])


class TestSessionConcurrency:
    def test_a_second_session_next_is_refused_while_one_holds_the_lock(
            self, session):
        import fcntl
        import os as os_mod

        paths = session.init()
        session.gate()
        session.child()
        fd = os_mod.open(str(paths["lock"]), os_mod.O_CREAT | os_mod.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(SystemExit,
                               match="another process is already working"):
                session.mod.run_session_next(session.id)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os_mod.close(fd)
        # Nothing was decided and nothing was written.
        assert session.events() == []

    def test_the_lock_covers_status_and_init_too(self, session):
        import fcntl
        import os as os_mod

        paths = session.init()
        fd = os_mod.open(str(paths["lock"]), os_mod.O_CREAT | os_mod.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(SystemExit, match="already working"):
                session.mod.run_session_status(session.id)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os_mod.close(fd)


class TestTwoPhaseJournal:
    def test_a_good_run_writes_started_then_finished(self, session):
        session.init()
        assert session.run_one() == 0
        names = session.events()
        assert names == [event_file_name(1, "r1", EVENT_STARTED),
                         event_file_name(2, "r1", EVENT_FINISHED)]

    def test_the_started_event_is_written_before_the_child(self, session):
        """Whatever the child does, the boot is already spent."""
        session.init()
        seen = {}

        def spy(cmd, **kw):
            seen["events"] = session.events()
            import types
            run_id = cmd[cmd.index("--run-id") + 1]
            entry = next(p for p in session.plan if p["run_id"] == run_id)
            session.mod.atomic_write_json(
                Path(cmd[cmd.index("--out") + 1]),
                child_record(session.mod, entry, session.digest))
            return types.SimpleNamespace(returncode=0)

        session.gate()
        session.mp.setattr(session.mod.subprocess, "run", spy)
        session.mod.run_session_next(session.id)
        assert seen["events"] == [event_file_name(1, "r1", EVENT_STARTED)]

    def test_the_finished_event_pins_the_started_event_by_digest(self, session):
        paths = session.init()
        session.run_one()
        events = read_events(paths["dir"])
        assert (events[1]["body"]["started_event_digest"]
                == events[0]["file_sha256"])
        assert events[1]["body"]["started_index"] == 1

    def test_a_failed_gate_writes_an_attempt_and_spends_no_boot(self, session):
        paths = session.init()
        assert session.run_one(gate_passes=False) == 1
        assert session.events() == [
            event_file_name(1, "r1", EVENT_GATE_ATTEMPT)]
        # Same boot, again: allowed, because nothing was measured.
        assert session.run_one() == 0
        assert session.events()[-1] == event_file_name(3, "r1", EVENT_FINISHED)

    def test_a_child_that_never_reported_leaves_the_session_terminal(self,
                                                                     session):
        """The parent died between the two events. Nothing may follow."""
        paths = session.init()
        session.gate()
        append_event(paths["dir"], 1, "r1", EVENT_STARTED,
                     session.started_body("r1"))
        session.next_boot()
        with pytest.raises(SystemExit, match="this experiment is over"):
            session.mod.run_session_next(session.id)

    def test_a_nonzero_exit_ends_the_experiment(self, session):
        session.init()
        assert session.run_one(returncode=1, write=False) == 1
        session.next_boot()
        with pytest.raises(SystemExit) as e:
            session.mod.run_session_next(session.id)
        assert "this experiment is over" in str(e.value)
        assert "new experiment id" in str(e.value)

    def test_a_timeout_ends_the_experiment(self, session):
        paths = session.init()
        assert session.run_one(timeout=True, write=False) == 1
        body = read_events(paths["dir"])[1]["body"]
        assert body["outcome"] == "timed_out" and body["exit_status"] is None
        session.next_boot()
        with pytest.raises(SystemExit, match="this experiment is over"):
            session.mod.run_session_next(session.id)

    def test_a_missing_report_ends_the_experiment(self, session):
        paths = session.init()
        assert session.run_one(returncode=0, write=False) == 1
        assert read_events(paths["dir"])[1]["body"]["outcome"] == "no_report"
        session.next_boot()
        with pytest.raises(SystemExit, match="this experiment is over"):
            session.mod.run_session_next(session.id)

    def test_a_second_measured_run_in_one_boot_is_refused(self, session):
        session.init()
        assert session.run_one() == 0
        with pytest.raises(SystemExit) as e:
            session.run_one()                       # same boot fingerprint
        assert "already happened in this boot" in str(e.value)
        assert "will not restart it for you" in str(e.value)

    def test_four_boots_run_the_whole_plan(self, session):
        paths = session.init()
        for i in range(4):
            if i:
                session.next_boot()
            assert session.run_one() == 0 or i == 3
        agg = json.loads(paths["aggregate"].read_text())
        assert agg["complete"] is True
        assert len(set(agg["boot_fingerprints"])) == 4
        assert agg["terminal"] is False

    def test_every_attempt_survives_into_the_aggregate(self, session):
        paths = session.init()
        assert session.run_one(gate_passes=False) == 1
        assert session.run_one(gate_passes=False) == 1
        assert session.run_one() == 0
        for i in range(3):
            session.next_boot()
            session.run_one()
        agg = json.loads(paths["aggregate"].read_text())
        r1 = next(c for c in agg["children"] if c["run_id"] == "r1")
        assert len(r1["attempts"]) == 4          # two gate attempts + the pair
        assert [a["event"] for a in r1["attempts"]] == [
            EVENT_GATE_ATTEMPT, EVENT_GATE_ATTEMPT, EVENT_STARTED,
            EVENT_FINISHED]
        assert len(agg["journal_events"]) == 10


class TestJournalValidator:
    """The journal is read fail-closed before anything is decided from it."""

    def two_runs(self, session):
        session.init()
        session.run_one()
        session.next_boot()
        session.run_one()
        return session.paths["dir"]

    def test_a_clean_journal_reads(self, session):
        self.two_runs(session)
        ctx, state = session.mod.read_journal(session.id)
        assert state["problems"] == []
        assert state["completed"] == ["r1", "r2"]
        assert state["next"]["run_id"] == "r3"

    def test_a_deleted_event_is_caught(self, session):
        d = self.two_runs(session)
        (d / "events" / event_file_name(2, "r1", EVENT_FINISHED)).unlink()
        with pytest.raises(SystemExit, match="does not check out"):
            session.mod.read_journal(session.id)

    def test_an_edited_event_is_caught(self, session):
        d = self.two_runs(session)
        p = d / "events" / event_file_name(2, "r1", EVENT_FINISHED)
        body = json.loads(p.read_text())
        body["exit_status"] = 0
        body["started_event_digest"] = "0" * 64
        p.unlink()
        p.write_text(json.dumps(body, indent=2))
        with pytest.raises(SystemExit, match="does not hash to the digest"):
            session.mod.read_journal(session.id)

    def test_an_inserted_event_is_caught(self, session):
        d = self.two_runs(session)
        src = d / "events" / event_file_name(1, "r1", EVENT_STARTED)
        body = json.loads(src.read_text())
        (d / "events" / event_file_name(2, "r1", EVENT_STARTED)).write_text(
            json.dumps(body, indent=2))
        with pytest.raises(SystemExit, match="does not check out"):
            session.mod.read_journal(session.id)

    def test_a_renumbered_event_is_caught(self, session):
        d = self.two_runs(session)
        p = d / "events" / event_file_name(1, "r1", EVENT_STARTED)
        p.rename(d / "events" / event_file_name(9, "r1", EVENT_STARTED))
        with pytest.raises(SystemExit, match="does not check out"):
            session.mod.read_journal(session.id)

    def test_events_out_of_plan_order_are_caught(self, session):
        session.init()
        d = session.paths["dir"]
        append_event(d, 1, "r3", EVENT_GATE_ATTEMPT, {
            "schema_version": session.mod.EXPERIMENT_SCHEMA,
            "kind": "session_event", "event": EVENT_GATE_ATTEMPT, "index": 1,
            "run_id": "r3", "condition": "empty_cache", "block_id": "B2",
            "position_in_block": 0, "global_position": 2,
            "experiment_id": session.id, "plan_digest": session.digest,
            "boot_fingerprint": session.boot, "recorded_at": "t",
            "gate": failing_gate(session.calib["thresholds"])})
        with pytest.raises(SystemExit, match="next planned run was 'r1'"):
            session.mod.read_journal(session.id)

    def test_a_duplicate_boot_in_the_journal_is_caught(self, session):
        """Even if the live check were bypassed, the journal still says no."""
        d = self.two_runs(session)
        p = d / "events" / event_file_name(3, "r2", EVENT_STARTED)
        body = json.loads(p.read_text())
        first = json.loads((d / "events" / event_file_name(
            1, "r1", EVENT_STARTED)).read_text())
        body["boot_fingerprint"] = first["boot_fingerprint"]
        p.unlink()
        p.write_text(json.dumps(body, indent=2))
        with pytest.raises(SystemExit, match="this boot already measured r1"):
            session.mod.read_journal(session.id)

    def test_a_started_event_whose_gate_did_not_pass_is_caught(self, session):
        d = self.two_runs(session)
        p = d / "events" / event_file_name(1, "r1", EVENT_STARTED)
        body = json.loads(p.read_text())
        body["gate"] = failing_gate(session.calib["thresholds"])
        p.unlink()
        p.write_text(json.dumps(body, indent=2))
        with pytest.raises(SystemExit,
                           match="began without its gate passing"):
            session.mod.read_journal(session.id)

    def test_a_tampered_gate_poll_in_the_journal_is_caught(self, session):
        d = self.two_runs(session)
        p = d / "events" / event_file_name(1, "r1", EVENT_STARTED)
        body = json.loads(p.read_text())
        body["gate"]["polls"][0]["sample"]["swap_used_gb"] = 99.0
        p.unlink()
        p.write_text(json.dumps(body, indent=2))
        with pytest.raises(SystemExit, match="evaluate to False"):
            session.mod.read_journal(session.id)

    def test_an_edited_child_report_is_caught(self, session):
        d = self.two_runs(session)
        rec = json.loads((d / "r1.json").read_text())
        rec["model_compute_seconds"] = 1.0
        (d / "r1.json").unlink()
        (d / "r1.json").write_text(json.dumps(rec, indent=2))
        with pytest.raises(SystemExit, match="has changed since the run"):
            session.mod.read_journal(session.id)

    def test_a_child_that_skipped_its_source_check_is_caught(self, session):
        session.init()
        session.run_one(mutate=lambda rec: rec.pop("child_source_check"))
        with pytest.raises(SystemExit,
                           match="did not check its own source"):
            session.mod.read_journal(session.id)

    def test_a_child_that_broke_its_treatment_contract_is_caught(self, session):
        def wrong_clear(rec):
            rec["per_row"][4]["scheduled_empty_cache_seconds"] = 0.05

        session.init()
        session.run_one(mutate=wrong_clear)
        with pytest.raises(SystemExit, match="control arm cleared at rows"):
            session.mod.read_journal(session.id)


class TestSourceReverifiedAroundTheGate:
    def edit_mid_gate(self, session, text="print('edited mid-gate')\n"):
        def edit():
            (session.root / "a.py").write_text(text)
        return edit

    def test_source_that_changes_while_the_gate_waits_stops_the_run(self,
                                                                    session):
        paths = session.init()
        with pytest.raises(SystemExit,
                           match="changed while the gate was waiting"):
            session.run_one(side_effect=self.edit_mid_gate(session))
        # Its own event: this gate passed, so filing it as "the gate never
        # released" would misdescribe the machine.
        assert session.events() == [
            event_file_name(1, "r1", EVENT_PRE_SPAWN_ABORT)]
        assert not (paths["dir"] / "r1.json").exists()

    def test_the_abort_keeps_the_polls_and_names_the_drift(self, session):
        paths = session.init()
        with pytest.raises(SystemExit):
            session.run_one(side_effect=self.edit_mid_gate(session))
        body = read_events(paths["dir"])[0]["body"]
        assert body["event"] == EVENT_PRE_SPAWN_ABORT
        assert body["gate"]["passed"] is True
        assert len(body["gate"]["polls"]) == 3
        assert "source changed while the gate was waiting" in body[
            "abort_reason"]
        assert any("a.py" in p for p in body["source_problems"])
        assert "out_path" not in body and "child_command_digest" not in body

    def test_restoring_the_source_lets_the_same_boot_start_normally(self,
                                                                    session):
        """The abort spent nothing: no boot, no measurement, no retry budget."""
        paths = session.init()
        original = (session.root / "a.py").read_text()
        with pytest.raises(SystemExit):
            session.run_one(side_effect=self.edit_mid_gate(session))
        boot_before = session.boot

        (session.root / "a.py").write_text(original)          # put it back
        assert session.mod.read_journal(session.id)[1]["problems"] == []

        assert session.run_one() == 0                          # same boot
        assert session.boot == boot_before
        assert session.events() == [
            event_file_name(1, "r1", EVENT_PRE_SPAWN_ABORT),
            event_file_name(2, "r1", EVENT_STARTED),
            event_file_name(3, "r1", EVENT_FINISHED)]
        ctx, state = session.mod.read_journal(session.id)
        assert state["problems"] == [] and state["completed"] == ["r1"]
        assert state["terminal"] is False

    def test_no_measurement_exists_before_the_started_event(self, session):
        paths = session.init()
        original = (session.root / "a.py").read_text()
        with pytest.raises(SystemExit):
            session.run_one(side_effect=self.edit_mid_gate(session))
        # After the abort: no child report, no started event, no boot spent.
        assert not (paths["dir"] / "r1.json").exists()
        events = read_events(paths["dir"])
        assert all(e["body"]["event"] != EVENT_STARTED for e in events)
        (session.root / "a.py").write_text(original)
        ctx, state = session.mod.read_journal(session.id)
        assert state["boots"] == {} and state["started"] == {}
        assert session.run_one() == 0

    def test_source_drift_before_the_gate_stops_the_run(self, session):
        session.init()
        (session.root / "a.py").write_text("print('changed')\n")
        with pytest.raises(SystemExit, match="cannot be read"):
            session.mod.run_session_next(session.id)

    def test_the_child_checks_the_source_itself(self, session):
        paths = session.init()
        session.mod.child_verify_source(paths["dir"], session.digest)
        (session.root / "src" / "b.py").write_text("print('drifted')\n")
        with pytest.raises(SystemExit, match="this child refuses to start"):
            session.mod.child_verify_source(paths["dir"], session.digest)

    def test_the_child_checks_the_plan_digest_itself(self, session):
        paths = session.init()
        with pytest.raises(SystemExit, match="not the one this child was told"):
            session.mod.child_verify_source(paths["dir"], "0" * 64)

    def test_a_child_without_a_session_refuses_before_loading_anything(self):
        import subprocess

        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--child", "--plan", "x",
             "--plan-digest", "0" * 64, "--global-position", "0",
             "--run-id", "r1", "--condition", "continuous", "--out", "y"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=180)
        assert r.returncode != 0
        assert "--child needs --session-dir" in r.stderr


class TestEventSemanticsAreJudgedNotCounted:
    """Deleting a field is the easy case. Editing one is the interesting one."""

    def started(self, session):
        session.init()
        session.run_one()
        return session.paths["dir"] / "events" / event_file_name(
            1, "r1", EVENT_STARTED)

    def edit(self, path, mutate):
        body = json.loads(path.read_text())
        mutate(body)
        path.unlink()
        path.write_text(json.dumps(body, indent=2))

    def expect(self, session, path, mutate, message):
        self.edit(path, mutate)
        with pytest.raises(SystemExit, match=message):
            session.mod.read_journal(session.id)

    def test_a_miscounted_source_check_is_caught(self, session):
        p = self.started(session)
        self.expect(session, p,
                    lambda b: b["source_verified"].update({"files": 1}),
                    "says it verified 1 files, the manifest holds 2")

    def test_a_source_check_against_another_manifest_is_caught(self, session):
        p = self.started(session)
        self.expect(session, p, lambda b: b["source_verified"].update(
            {"source_manifest_digest": "9" * 64}),
            "names a different source manifest")

    def test_a_source_check_that_predates_the_gate_is_caught(self, session):
        p = self.started(session)
        self.expect(session, p, lambda b: b["source_verified"].update(
            {"checked_after_gate": False}),
            "does not record that the source was re-checked after the gate")

    def test_an_out_path_the_session_never_intended_is_caught(self, session):
        p = self.started(session)
        self.expect(session, p,
                    lambda b: b.update({"out_path": "runs/boot001/r9.json"}),
                    "but this session writes r1 to")

    def test_an_edited_child_command_digest_is_caught(self, session):
        p = self.started(session)
        self.expect(session, p,
                    lambda b: b.update({"child_command_digest": "0" * 64}),
                    "not the one this session's own invocation produces")

    def test_a_command_digest_for_another_run_is_caught(self, session):
        """Recomputed, not merely present: r2's digest under r1's event."""
        p = self.started(session)
        other = session.mod.child_invocation(
            session.paths, session.digest,
            next(e for e in session.plan if e["run_id"] == "r2"))
        self.expect(session, p,
                    lambda b: b.update({"child_command_digest": other["digest"]}),
                    "not the one this session's own invocation produces")

    def test_a_child_that_verified_the_wrong_number_of_files_is_caught(
            self, session):
        session.init()
        session.run_one(mutate=lambda rec: rec["child_source_check"].update(
            {"files_verified": 99}))
        with pytest.raises(SystemExit,
                           match="says it verified 99 files"):
            session.mod.read_journal(session.id)

    def test_a_child_that_checked_another_manifest_is_caught(self, session):
        session.init()
        session.run_one(mutate=lambda rec: rec["child_source_check"].update(
            {"source_manifest_digest": "9" * 64}))
        with pytest.raises(SystemExit,
                           match="checked a different source manifest"):
            session.mod.read_journal(session.id)

    def test_a_child_source_check_that_is_not_a_record_is_caught(self, session):
        session.init()
        session.run_one(mutate=lambda rec: rec.update(
            {"child_source_check": "yes"}))
        with pytest.raises(SystemExit, match="did not check its own source"):
            session.mod.read_journal(session.id)

    def test_a_finished_event_pointing_at_another_report_is_caught(self,
                                                                   session):
        session.init()
        session.run_one()
        p = (session.paths["dir"] / "events"
             / event_file_name(2, "r1", EVENT_FINISHED))
        self.expect(session, p,
                    lambda b: b.update({"report_path": "runs/boot001/r2.json"}),
                    "but the measurement began against")

    def test_a_finished_event_for_another_run_is_caught(self, session):
        session.init()
        session.run_one()
        p = (session.paths["dir"] / "events"
             / event_file_name(2, "r1", EVENT_FINISHED))
        self.expect(session, p, lambda b: b.update({"run_id": "r2"}),
                    "file says run 'r1', the event says 'r2'")

    def test_a_finished_event_from_another_boot_is_caught(self, session):
        session.init()
        session.run_one()
        p = (session.paths["dir"] / "events"
             / event_file_name(2, "r1", EVENT_FINISHED))
        self.expect(session, p,
                    lambda b: b.update({"boot_fingerprint": "ELSEWHERE"}),
                    "finished in a different boot")

    def test_a_finished_event_pointing_at_the_wrong_index_is_caught(self,
                                                                    session):
        session.init()
        session.run_one()
        p = (session.paths["dir"] / "events"
             / event_file_name(2, "r1", EVENT_FINISHED))
        self.expect(session, p, lambda b: b.update({"started_index": 7}),
                    "points at journal index 7")


class TestTerminalSessionsStillProduceARecord:
    """An experiment that stopped still owes an honest account of itself."""

    def test_a_nonzero_exit_writes_an_incomplete_aggregate(self, session):
        paths = session.init()
        assert session.run_one(returncode=1, write=False) == 1
        agg = json.loads(paths["aggregate"].read_text())
        assert agg["complete"] is False and agg["terminal"] is True
        assert "did not complete (nonzero_exit)" in agg["stopped_reason"]
        assert paths["markdown"].exists()

    def test_a_timeout_writes_an_incomplete_aggregate(self, session):
        paths = session.init()
        assert session.run_one(timeout=True, write=False) == 1
        agg = json.loads(paths["aggregate"].read_text())
        assert agg["complete"] is False
        assert "timed_out" in agg["stopped_reason"]

    def test_a_missing_report_writes_an_incomplete_aggregate(self, session):
        paths = session.init()
        assert session.run_one(returncode=0, write=False) == 1
        agg = json.loads(paths["aggregate"].read_text())
        assert agg["complete"] is False
        assert "no_report" in agg["stopped_reason"]

    def test_a_crashed_parent_can_be_finalised_without_re_running(self,
                                                                  session):
        """started, no finished: the tool that never came back left a record."""
        paths = session.init()
        session.gate()
        append_event(paths["dir"], 1, "r1", EVENT_STARTED,
                     session.started_body("r1"))
        assert session.mod.run_session_finalize(session.id) == 1
        agg = json.loads(paths["aggregate"].read_text())
        assert agg["terminal"] is True
        assert "no finished event was ever written" in agg["stopped_reason"]
        r1 = next(c for c in agg["children"] if c["run_id"] == "r1")
        assert r1["outcome"] == "never_finished" and r1["not_run"] is False
        # And it is still refused as a thing to continue.
        session.next_boot()
        with pytest.raises(SystemExit, match="this experiment is over"):
            session.mod.run_session_next(session.id)

    def test_finalising_twice_changes_nothing(self, session):
        paths = session.init()
        session.run_one(returncode=1, write=False)
        before = session.mod.sha256_file(paths["aggregate"])
        assert session.mod.run_session_finalize(session.id) == 1
        assert session.mod.sha256_file(paths["aggregate"]) == before

    def test_finalising_a_complete_session_twice_changes_nothing(self,
                                                                 session):
        paths = session.init()
        for i in range(4):
            if i:
                session.next_boot()
            session.run_one()
        before = session.mod.sha256_file(paths["aggregate"])
        assert session.mod.run_session_finalize(session.id) == 0
        assert session.mod.sha256_file(paths["aggregate"]) == before

    def test_an_aggregate_that_no_longer_derives_is_refused(self, session):
        paths = session.init()
        session.run_one(returncode=1, write=False)
        agg = json.loads(paths["aggregate"].read_text())
        agg["complete"] = True
        paths["aggregate"].unlink()
        paths["aggregate"].write_text(json.dumps(agg, indent=2))
        with pytest.raises(SystemExit,
                           match="not what deriving it again produces"):
            session.mod.run_session_finalize(session.id)

    def test_finalising_reports_the_runs_that_did_finish(self, session):
        paths = session.init()
        session.run_one()
        session.next_boot()
        session.run_one(returncode=1, write=False)
        agg = json.loads(paths["aggregate"].read_text())
        done = next(c for c in agg["children"] if c["run_id"] == "r1")
        assert done["outcome"] == "completed" and done["exit_status"] == 0
        assert agg["complete"] is False


class TestVerifyRebuildsTheBootEvidence:
    """`--verify` re-runs the state machine; it does not read the summary."""

    def finished(self, session):
        paths = session.init()
        for i in range(4):
            if i:
                session.next_boot()
            session.run_one()
        return paths

    def stopped(self, session):
        """One clean run, then one that exited non-zero: terminal, finalised."""
        paths = session.init()
        session.run_one()
        session.next_boot()
        session.run_one(returncode=1, write=False)
        return paths

    def test_a_clean_session_replays_from_its_journal(self, session):
        paths = self.finished(session)
        exp = session.mod.load_experiment(paths["dir"])
        assert exp["replay_problems"] == []
        assert exp["journal_replay"]["events_replayed"] == 8
        assert len(set(exp["journal_replay"]["boot_fingerprints"])) == 4
        assert exp["headline"]["distinct_boots"] == 4

    def test_the_aggregate_names_its_journal_and_snapshot(self, session):
        paths = self.finished(session)
        agg = json.loads(paths["aggregate"].read_text())
        assert agg["session_sha256"] == session.mod.sha256_file(
            paths["session"])
        assert agg["source_manifest_digest"] == session.session[
            "source_manifest_digest"]
        assert len(agg["journal_files"]) == 8
        for f in agg["journal_files"]:
            assert session.mod.sha256_file(
                paths["dir"] / "events" / f["file_name"]) == f["sha256"]

    def test_an_edited_journal_event_is_refused_after_the_fact(self, session):
        paths = self.finished(session)
        p = (paths["dir"] / "events"
             / event_file_name(1, "r1", EVENT_STARTED))
        body = json.loads(p.read_text())
        body["recorded_at"] = "later"
        p.unlink()
        p.write_text(json.dumps(body, indent=2))
        with pytest.raises(SystemExit, match="has changed since the aggregate"):
            session.mod.load_experiment(paths["dir"])

    def test_a_deleted_journal_event_is_refused_after_the_fact(self, session):
        paths = self.finished(session)
        (paths["dir"] / "events"
         / event_file_name(8, "r4", EVENT_FINISHED)).unlink()
        with pytest.raises(SystemExit, match="is missing"):
            session.mod.load_experiment(paths["dir"])

    def test_an_edited_aggregate_boot_fingerprint_is_refused(self, session):
        paths = self.finished(session)
        agg = json.loads(paths["aggregate"].read_text())
        agg["boot_fingerprints"][2] = agg["boot_fingerprints"][0]
        paths["aggregate"].unlink()
        paths["aggregate"].write_text(json.dumps(agg, indent=2))
        with pytest.raises(SystemExit,
                           match="not the ones the journal records"):
            session.mod.load_experiment(paths["dir"])

    def test_a_child_slot_claiming_another_boot_is_refused(self, session):
        paths = self.finished(session)
        agg = json.loads(paths["aggregate"].read_text())
        agg["children"][1]["boot_fingerprint"] = "INVENTED"
        paths["aggregate"].unlink()
        paths["aggregate"].write_text(json.dumps(agg, indent=2))
        with pytest.raises(SystemExit, match="the journal gives"):
            session.mod.load_experiment(paths["dir"])

    def test_an_invented_attempt_is_refused(self, session):
        paths = self.finished(session)
        agg = json.loads(paths["aggregate"].read_text())
        agg["children"][0]["attempts"].append(
            {"index": 99, "event": EVENT_GATE_ATTEMPT})
        paths["aggregate"].unlink()
        paths["aggregate"].write_text(json.dumps(agg, indent=2))
        with pytest.raises(SystemExit, match="the journal gives"):
            session.mod.load_experiment(paths["dir"])

    def test_the_headline_rests_on_journal_fingerprints(self, session):
        """Faking four boots in the aggregate does not buy a headline."""
        paths = self.stopped(session)
        agg = json.loads(paths["aggregate"].read_text())
        agg["boot_fingerprints"] = ["b1", "b2", "b3", "b4"]
        for i, c in enumerate(agg["children"]):
            c["boot_fingerprint"] = f"b{i + 1}"
        paths["aggregate"].unlink()
        paths["aggregate"].write_text(json.dumps(agg, indent=2))
        with pytest.raises(SystemExit,
                           match="not the ones the journal records"):
            session.mod.load_experiment(paths["dir"])

    def test_replay_does_not_require_a_frozen_working_tree(self, session):
        """The snapshot is what makes a finished session durable."""
        paths = self.finished(session)
        (session.root / "a.py").write_text("print('development continued')\n")
        exp = session.mod.load_experiment(paths["dir"])
        assert exp["replay_problems"] == []
        assert exp["journal_replay"][
            "snapshot_verified_without_the_working_tree"] is True

    def test_replay_still_requires_an_intact_snapshot(self, session):
        paths = self.finished(session)
        (paths["snapshot"] / "a.py").unlink()
        with pytest.raises(SystemExit, match="snapshot copy is missing"):
            session.mod.load_experiment(paths["dir"])


class TestTerminalAggregatesReplay:
    """A stopped experiment still has to be readable, and still says so."""

    def crashed(self, session):
        """started, never finished: the parent died mid-measurement."""
        paths = session.init()
        session.gate()
        append_event(paths["dir"], 1, "r1", EVENT_STARTED,
                     session.started_body("r1"))
        assert session.mod.run_session_finalize(session.id) == 1
        return paths

    @pytest.mark.parametrize("kw,outcome", [
        ({"returncode": 1, "write": False}, "nonzero_exit"),
        ({"timeout": True, "write": False}, "timed_out"),
        ({"returncode": 0, "write": False}, "no_report"),
    ])
    def test_a_terminal_session_replays(self, session, kw, outcome):
        paths = session.init()
        assert session.run_one(**kw) == 1
        exp = session.mod.load_experiment(paths["dir"])
        assert exp["replay_problems"] == []
        assert exp["journal_replay"]["terminal"] is True
        assert exp["headline"]["allowed"] is False
        r1 = next(c for c in exp["children"] if c["run_id"] == "r1")
        assert r1["outcome"] == outcome
        assert r1["not_run"] is False and r1["record"] is None

    def test_a_crashed_parent_replays(self, session):
        paths = self.crashed(session)
        exp = session.mod.load_experiment(paths["dir"])
        assert exp["replay_problems"] == []
        assert exp["journal_replay"]["terminal"] is True
        assert exp["headline"]["allowed"] is False
        r1 = next(c for c in exp["children"] if c["run_id"] == "r1")
        assert r1["outcome"] == "never_finished"

    def test_a_legacy_experiment_still_refuses_a_missing_report(self, mod,
                                                                tmp_path,
                                                                monkeypatch):
        """The excuse comes from the journal. Without one there is no excuse."""
        monkeypatch.setattr(mod, "ROOT", tmp_path)
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        calib = calibration_doc(mod)
        exp_dir = tmp_path / "exp1"
        exp_dir.mkdir()
        mod.atomic_write_json(tmp_path / "calibration.json", calib)
        mod.atomic_write_json(exp_dir / "plan.json", {
            "schema_version": mod.EXPERIMENT_SCHEMA, "kind": "plan",
            "plan": plan, "plan_digest": digest,
            "calibration_path": "calibration.json",
            "calibration_digest": calib["calibration_digest"]})
        agg = {"schema_version": mod.EXPERIMENT_SCHEMA, "kind": "experiment",
               "experiment_id": "exp1", "plan": plan, "plan_digest": digest,
               "calibration_digest": calib["calibration_digest"],
               "thresholds": calib["thresholds"], "gate_policy": calib["gate"],
               "children": [{
                   "run_id": "r1", "condition": "continuous", "block_id": "B1",
                   "global_position": 0, "exit_status": 1, "not_run": False,
                   "gate": passing_gate(calib["thresholds"]),
                   "recovery_passed_recomputed": True,
                   "report_path": None, "report_sha256": None,
                   "summary": None}],
               "complete": False, "stopped_reason": "r1 exited with 1"}
        mod.atomic_write_json(exp_dir / "aggregate.json", agg)
        with pytest.raises(SystemExit,
                           match="ran but references no report"):
            mod.load_experiment(exp_dir)


class TestExperimentIdentityChain:
    """One experiment id, from the directory name down to every event."""

    def rewrite(self, path, mutate):
        body = json.loads(path.read_text())
        mutate(body)
        path.unlink()
        path.write_text(json.dumps(body, indent=2))

    def event(self, session, index, run_id, kind):
        return session.paths["dir"] / "events" / event_file_name(
            index, run_id, kind)

    def with_events(self, session):
        """A journal holding all four event kinds."""
        session.init()
        session.run_one(gate_passes=False)                    # gate_attempt
        original = (session.root / "a.py").read_text()
        with pytest.raises(SystemExit):                       # pre_spawn_abort
            session.run_one(side_effect=lambda: (
                session.root / "a.py").write_text("print('mid')\n"))
        (session.root / "a.py").write_text(original)
        session.run_one()                                     # started+finished
        return session.paths

    def test_the_journal_holds_all_four_kinds(self, session):
        self.with_events(session)
        assert session.events() == [
            event_file_name(1, "r1", EVENT_GATE_ATTEMPT),
            event_file_name(2, "r1", EVENT_PRE_SPAWN_ABORT),
            event_file_name(3, "r1", EVENT_STARTED),
            event_file_name(4, "r1", EVENT_FINISHED)]

    @pytest.mark.parametrize("index,kind", [
        (1, EVENT_GATE_ATTEMPT), (2, EVENT_PRE_SPAWN_ABORT),
        (3, EVENT_STARTED), (4, EVENT_FINISHED)])
    def test_an_event_from_another_experiment_is_refused(self, session, index,
                                                         kind):
        self.with_events(session)
        self.rewrite(self.event(session, index, "r1", kind),
                     lambda b: b.update({"experiment_id": "someone_elses"}))
        with pytest.raises(SystemExit) as e:
            session.mod.read_journal(session.id)
        assert "belongs to experiment 'someone_elses'" in str(e.value)
        assert "Events are not moved between sessions" in str(e.value)

    def test_the_refusal_comes_before_any_new_event_or_child(self, session):
        paths = self.with_events(session)
        before = session.events()
        self.rewrite(self.event(session, 3, "r1", EVENT_STARTED),
                     lambda b: b.update({"experiment_id": "someone_elses"}))
        reached = []
        session.mp.setattr(session.mod, "wait_for_recovery",
                           lambda *a, **kw: reached.append(True))
        spawned = []
        session.mp.setattr(session.mod.subprocess, "run",
                           lambda *a, **kw: spawned.append(True))
        session.next_boot()
        with pytest.raises(SystemExit, match="belongs to experiment"):
            session.mod.run_session_next(session.id)
        assert reached == [] and spawned == []
        assert session.events() == before
        assert not paths["aggregate"].exists()

    def test_an_aggregate_naming_another_experiment_is_refused(self, session):
        paths = session.init()
        session.run_one(returncode=1, write=False)
        self.rewrite(paths["aggregate"],
                     lambda a: a.update({"experiment_id": "elsewhere"}))
        with pytest.raises(SystemExit) as e:
            session.mod.load_experiment(paths["dir"])
        assert "says it belongs to experiment 'elsewhere'" in str(e.value)
        assert "the directory names the experiment" in str(e.value)

    def test_the_aggregate_id_never_selects_which_session_to_load(
            self, mod, tmp_path, monkeypatch):
        """Two sessions, and one aggregate pointing at the other."""
        a = SessionHarness(mod, tmp_path, monkeypatch, experiment_id="bootA")
        a.init()
        a.run_one(returncode=1, write=False)          # terminal, finalised
        b_paths = mod.session_paths("bootB")
        b_paths["dir"].mkdir(parents=True)
        # bootB holds bootA's aggregate. Nothing of bootA's may be consulted.
        agg = json.loads(a.paths["aggregate"].read_text())
        (b_paths["dir"] / "aggregate.json").write_text(json.dumps(agg,
                                                                  indent=2))
        (b_paths["dir"] / "plan.json").write_text(
            a.paths["plan"].read_text())
        with pytest.raises(SystemExit) as e:
            mod.load_experiment(b_paths["dir"])
        assert "says it belongs to experiment 'bootA'" in str(e.value)

    def test_a_session_may_not_borrow_another_sessions_events(
            self, mod, tmp_path, monkeypatch):
        a = SessionHarness(mod, tmp_path, monkeypatch, experiment_id="bootA")
        a.init()
        a.run_one()
        b = SessionHarness(mod, tmp_path, monkeypatch, experiment_id="bootB")
        b.init()
        # Copy bootA's finished pair into bootB's journal.
        for index, kind in ((1, EVENT_STARTED), (2, EVENT_FINISHED)):
            name = event_file_name(index, "r1", kind)
            (b.paths["dir"] / "events").mkdir(exist_ok=True)
            (b.paths["dir"] / "events" / name).write_text(
                (a.paths["dir"] / "events" / name).read_text())
        with pytest.raises(SystemExit) as e:
            mod.read_journal("bootB")
        assert "belongs to experiment 'bootA', this session is 'bootB'" in str(
            e.value)

    def test_a_clean_session_keeps_one_identity_throughout(self, session):
        paths = self.with_events(session)
        ctx, state = session.mod.read_journal(session.id)
        assert state["problems"] == []
        assert ctx["experiment_id"] == session.id
        for item in state["events"]:
            assert item["body"]["experiment_id"] == session.id
        assert json.loads(paths["plan"].read_text())[
            "experiment_id"] == session.id
        assert json.loads(paths["session"].read_text())[
            "experiment_id"] == session.id


class TestSessionDetectionIsNotAFlag:
    def test_a_session_flagged_as_legacy_is_still_read_as_a_session(self,
                                                                    session):
        paths = session.init()
        session.run_one(returncode=1, write=False)
        agg = json.loads(paths["aggregate"].read_text())
        agg["one_run_per_boot"] = False
        paths["aggregate"].unlink()
        paths["aggregate"].write_text(json.dumps(agg, indent=2))
        with pytest.raises(SystemExit) as e:
            session.mod.load_experiment(paths["dir"])
        assert "one_run_per_boot=False" in str(e.value)
        assert "not read as a legacy experiment because a flag says so" in str(
            e.value)

    def test_removing_the_flag_entirely_does_not_help(self, session):
        paths = session.init()
        session.run_one(returncode=1, write=False)
        agg = json.loads(paths["aggregate"].read_text())
        del agg["one_run_per_boot"]
        paths["aggregate"].unlink()
        paths["aggregate"].write_text(json.dumps(agg, indent=2))
        with pytest.raises(SystemExit, match="one_run_per_boot=None"):
            session.mod.load_experiment(paths["dir"])

    def test_the_directory_alone_marks_a_session(self, mod, tmp_path):
        assert mod.looks_like_a_session(tmp_path, {}) is False
        (tmp_path / "session.json").write_text("{}")
        assert mod.looks_like_a_session(tmp_path, {}) is True

    def test_a_session_field_alone_marks_a_session(self, mod, tmp_path):
        assert mod.looks_like_a_session(tmp_path, {"journal_files": []}) is True
        assert mod.looks_like_a_session(tmp_path, {"terminal": False}) is True


class TestAggregateToSessionComparison:
    """Every pointer the aggregate holds into the session is checked."""

    def stopped(self, session):
        paths = session.init()
        session.run_one(returncode=1, write=False)
        return paths

    def edit(self, paths, mutate):
        agg = json.loads(paths["aggregate"].read_text())
        mutate(agg)
        paths["aggregate"].unlink()
        paths["aggregate"].write_text(json.dumps(agg, indent=2))

    def expect(self, session, paths, mutate, message):
        self.edit(paths, mutate)
        with pytest.raises(SystemExit, match=message):
            session.mod.load_experiment(paths["dir"])

    def test_a_missing_session_path_is_caught(self, session):
        paths = self.stopped(session)
        self.expect(session, paths, lambda a: a.pop("session_path"),
                    "names session file None")

    def test_a_wrong_session_path_is_caught(self, session):
        paths = self.stopped(session)
        self.expect(session, paths,
                    lambda a: a.update({"session_path": "elsewhere.json"}),
                    "names session file 'elsewhere.json'")

    def test_a_missing_session_digest_is_not_skipped(self, session):
        paths = self.stopped(session)
        self.expect(session, paths, lambda a: a.pop("session_sha256"),
                    "records no digest for session.json")

    def test_an_edited_session_file_is_caught(self, session):
        paths = self.stopped(session)
        doc = json.loads(paths["session"].read_text())
        doc["policy"] = "rewritten"
        paths["session"].unlink()
        paths["session"].write_text(json.dumps(doc, indent=2))
        with pytest.raises(SystemExit, match="session.json has changed"):
            session.mod.load_experiment(paths["dir"])

    def test_a_wrong_snapshot_dir_is_caught(self, session):
        paths = self.stopped(session)
        self.expect(session, paths,
                    lambda a: a.update({"source_snapshot_dir": "elsewhere"}),
                    "disagree on where the source snapshot lives")

    def test_a_wrong_manifest_digest_is_caught(self, session):
        paths = self.stopped(session)
        self.expect(session, paths,
                    lambda a: a.update({"source_manifest_digest": "9" * 64}),
                    "disagree on the source manifest digest")

    def test_a_duplicated_journal_file_entry_is_caught(self, session):
        paths = self.stopped(session)
        self.expect(session, paths,
                    lambda a: a["journal_files"].append(a["journal_files"][0]),
                    "lists the same journal event twice")

    def test_a_dropped_journal_file_entry_is_caught(self, session):
        paths = self.stopped(session)
        self.expect(session, paths, lambda a: a["journal_files"].pop(),
                    "was added after the aggregate was written")

    def test_an_edited_journal_file_digest_is_caught(self, session):
        paths = self.stopped(session)
        self.expect(session, paths,
                    lambda a: a["journal_files"][0].update(
                        {"sha256": "0" * 64}),
                    "has changed since the aggregate was written")

    def test_a_missing_child_slot_is_caught(self, session):
        paths = self.stopped(session)
        self.expect(session, paths, lambda a: a["children"].pop(),
                    "3 child slots for a 4-run plan")

    @pytest.mark.parametrize("key,value", [
        ("exit_status", 0),
        ("outcome", "completed"),
        ("not_run", True),
        ("report_path", "runs/boot001/r1.json"),
        ("report_sha256", "9" * 64),
        ("boot_fingerprint", "INVENTED"),
        ("recovery_passed_recomputed", False),
        ("started_at", "yesterday"),
        ("parent_observed_wall_seconds", 0.1),
    ])
    def test_an_edited_child_field_is_caught(self, session, key, value):
        paths = self.stopped(session)
        self.expect(session, paths,
                    lambda a: a["children"][0].update({key: value}),
                    "the journal gives")

    def test_an_edited_complete_flag_is_caught(self, session):
        paths = self.stopped(session)
        self.expect(session, paths, lambda a: a.update({"complete": True}),
                    "the journal implies False")

    def test_an_edited_stop_reason_is_caught(self, session):
        paths = self.stopped(session)
        self.expect(session, paths,
                    lambda a: a.update({"stopped_reason": "nothing happened"}),
                    "stop reason is not the one the journal implies")

    def test_an_edited_terminal_flag_is_caught(self, session):
        paths = self.stopped(session)
        self.expect(session, paths, lambda a: a.update({"terminal": False}),
                    "the journal says True")


class TestCrashFinalizeUsesTheSnapshot:
    """Finalising a crash must not depend on the tree having stood still."""

    def crashed(self, session):
        paths = session.init()
        session.gate()
        append_event(paths["dir"], 1, "r1", EVENT_STARTED,
                     session.started_body("r1"))
        return paths

    def test_finalise_and_verify_survive_a_moved_working_tree(self, session):
        paths = self.crashed(session)
        (session.root / "a.py").write_text("print('development continued')\n")
        assert session.mod.run_session_finalize(session.id) == 1
        agg = json.loads(paths["aggregate"].read_text())
        assert agg["terminal"] is True
        exp = session.mod.load_experiment(paths["dir"])
        assert exp["replay_problems"] == []
        assert exp["journal_replay"]["terminal"] is True

    def test_a_broken_snapshot_stops_both(self, session):
        paths = self.crashed(session)
        (paths["snapshot"] / "a.py").unlink()
        with pytest.raises(SystemExit, match="snapshot copy is missing"):
            session.mod.run_session_finalize(session.id)
        assert not paths["aggregate"].exists()

    def test_a_broken_snapshot_stops_verify_too(self, session):
        paths = self.crashed(session)
        session.mod.run_session_finalize(session.id)
        (paths["snapshot"] / "src__b.py").unlink()
        with pytest.raises(SystemExit, match="snapshot copy is missing"):
            session.mod.load_experiment(paths["dir"])

    def test_starting_a_run_still_needs_the_live_tree(self, session):
        session.init()
        (session.root / "a.py").write_text("print('changed')\n")
        with pytest.raises(SystemExit, match="has changed since the plan"):
            session.mod.run_session_next(session.id)

    def test_the_refusal_points_at_finalize_without_offering_a_rerun(self,
                                                                     session):
        self.crashed(session)
        session.next_boot()
        with pytest.raises(SystemExit) as e:
            session.mod.run_session_next(session.id)
        message = str(e.value)
        assert "--session-finalize" in message
        assert "does not resume it" in message
        assert "cannot be repeated" in message

    def test_once_finalised_the_refusal_stops_pointing_there(self, session):
        session_paths = self.crashed(session)
        session.mod.run_session_finalize(session.id)
        session.next_boot()
        with pytest.raises(SystemExit) as e:
            session.mod.run_session_next(session.id)
        assert "--session-finalize" not in str(e.value)
        assert session_paths["aggregate"].exists()


class TestNothingLeaksTheBoot:
    def test_no_artefact_contains_a_raw_boot_identifier(self, mod, tmp_path,
                                                        monkeypatch):
        raw = "F00DFACE-1111-2222-3333-444455556666"
        harness = SessionHarness(mod, tmp_path, monkeypatch)
        monkeypatch.setattr(
            mod, "boot_identity",
            lambda experiment_id: boot_identity(
                experiment_id,
                sysctl=lambda n: raw if n == "kern.bootsessionuuid" else None))
        paths = harness.init()
        harness.gate()
        harness.child()
        mod.run_session_next(harness.id)
        written = "".join(
            p.read_text() for p in sorted(tmp_path.rglob("*"))
            if p.is_file() and p.suffix in (".json", ".md"))
        assert raw not in written
        assert "kern.bootsessionuuid" in written        # the source is named
        assert paths["dir"].exists()

    def test_status_prints_a_fingerprint_not_a_uuid(self, mod, tmp_path,
                                                    monkeypatch, capsys):
        raw = "F00DFACE-9999-8888-7777-666655554444"
        harness = SessionHarness(mod, tmp_path, monkeypatch)
        monkeypatch.setattr(
            mod, "boot_identity",
            lambda experiment_id: boot_identity(
                experiment_id,
                sysctl=lambda n: raw if n == "kern.bootsessionuuid" else None))
        harness.init()
        capsys.readouterr()
        mod.run_session_status(harness.id)
        assert raw not in capsys.readouterr().out


class TestSessionAggregate:
    def test_the_assembled_experiment_replays(self, session):
        paths = session.init()
        for i in range(4):
            if i:
                session.next_boot()
            session.run_one()
        exp = session.mod.load_experiment(paths["dir"])
        assert exp["replay_problems"] == []
        assert exp["losses"]["verdict"] == "within_tolerance"
        assert exp["cross_child_comparison"]["disagreements"] == []
        assert exp["headline"]["distinct_boots"] == 4

    def test_a_session_that_can_still_continue_may_not_be_finalised(self,
                                                                    session):
        """An aggregate after r1 is a report of an experiment that has not
        happened, and the idempotence check would then hold every later
        finalise to it."""
        paths = session.init()
        session.run_one()
        with pytest.raises(SystemExit, match="can still continue"):
            session.mod.finalise_session(session.id)
        assert not paths["aggregate"].exists()
        assert not paths["markdown"].exists()

    def test_and_the_remaining_runs_still_finish_normally(self, session):
        paths = session.init()
        session.run_one()
        with pytest.raises(SystemExit, match="can still continue"):
            session.mod.run_session_finalize(session.id)
        for _ in range(3):
            session.next_boot()
            session.run_one()
        agg = json.loads(paths["aggregate"].read_text())
        assert agg["complete"] is True
        assert len(set(agg["boot_fingerprints"])) == 4

    def test_retryable_events_alone_do_not_allow_a_finalise(self, session):
        paths = session.init()
        session.run_one(gate_passes=False)
        with pytest.raises(SystemExit, match="can still continue"):
            session.mod.finalise_session(session.id)
        assert not paths["aggregate"].exists()

    def test_a_session_never_touches_the_real_report(self, session):
        paths = session.init()
        for i in range(4):
            if i:
                session.next_boot()
            session.run_one()
        assert paths["markdown"].name == "15_mps_order_boot001.md"
        assert not (session.root / "reports" / "15_mps_order.md").exists()

    def test_the_report_lists_the_boots_and_the_attempts(self, session):
        paths = session.init()
        session.run_one(gate_passes=False)
        for i in range(4):
            if i:
                session.next_boot()
            session.run_one()
        text = paths["markdown"].read_text()
        assert "## Boots and attempts" in text
        assert "the machine's own boot identifier is never written down" in text


class TestHeadlineNeedsFourBoots:
    def fingerprinted(self, mod, boots):
        plan = mod.build_plan("exp1")
        digest = mod.digest_obj(plan)
        recs = []
        for e in plan:
            r = child_record(mod, e, digest)
            r["windows"][0]["seconds_per_row"] = (
                1.0 if e["condition"] == "empty_cache" else 5.0)
            recs.append(r)
        exp = experiment(mod, records=recs)
        for slot, fp in zip(exp["children"], boots):
            slot["boot_fingerprint"] = fp
        return mod.analyse(exp, calibration_doc(mod))

    def test_four_distinct_boots_are_allowed(self, mod):
        exp = self.fingerprinted(mod, ["b1", "b2", "b3", "b4"])
        assert exp["headline"]["allowed"]
        assert exp["headline"]["distinct_boots"] == 4

    def test_two_runs_in_one_boot_block_the_headline(self, mod):
        exp = self.fingerprinted(mod, ["b1", "b1", "b3", "b4"])
        assert not exp["headline"]["allowed"]
        assert any("own boot" in r for r in exp["headline"]["reasons"])

    def test_runs_with_no_boot_recorded_block_the_headline(self, mod):
        exp = self.fingerprinted(mod, [None, None, None, None])
        assert not exp["headline"]["allowed"]
        assert any("own boot" in r for r in exp["headline"]["reasons"])


class TestTheSameBootParentIsWithdrawn:
    def test_run_refuses_and_says_why(self):
        import subprocess

        r = subprocess.run([sys.executable, str(SCRIPT), "--run",
                            "--experiment-id", "nope"],
                           capture_output=True, text=True, cwd=str(ROOT),
                           timeout=180)
        assert r.returncode != 0
        assert "`--run` is withdrawn" in r.stderr

    def test_it_names_the_one_approved_flow(self, mod):
        assert "The approved flow is one measured run per boot" in (
            mod.RUN_WITHDRAWN)
        assert "--session-init" in mod.RUN_WITHDRAWN
        assert "--session-next" in mod.RUN_WITHDRAWN
        assert "once per boot, four times" in mod.RUN_WITHDRAWN

    def test_it_says_the_withdrawal_is_permanent(self, mod):
        assert "withdrawn permanently" in mod.RUN_WITHDRAWN
        assert "no flag that makes this flow correct" in mod.RUN_WITHDRAWN

    def test_it_refuses_to_reboot_on_the_operators_behalf(self, mod):
        assert "This tool does not restart it and must not" in (
            mod.RUN_WITHDRAWN)
        assert "changed the thing it was measuring" in mod.RUN_WITHDRAWN

    def test_the_stale_not_yet_approved_wording_is_gone(self, mod):
        for stale in ("NOT approved to run yet", "waiting on Codex review",
                      "should be executed until that review passes"):
            assert stale not in mod.RUN_WITHDRAWN

    def test_reading_a_finished_experiment_stays_separate(self, mod):
        assert "read-only: --verify, --recompute and --from-json" in (
            mod.RUN_WITHDRAWN)

    def test_it_names_the_evidence_against_itself(self, mod):
        assert "1.558GB against a 0.537GB threshold" in mod.RUN_WITHDRAWN
        assert "stopped at one run of four" in mod.RUN_WITHDRAWN

    def test_nothing_still_calls_it(self):
        src = SCRIPT.read_text()
        assert "wait_for_recovery(thresholds)" not in src


class TestExp001IsNotOverwritten:
    """The stored run is evidence. Nothing in this round rewrites it."""

    def exp_dir(self):
        return ROOT / "data" / "reports" / "15_mps_order" / "exp001"

    def stored(self):
        return json.loads(
            (self.exp_dir() / "aggregate.recomputed.json").read_text())

    @pytest.mark.skipif(
        not (ROOT / "data" / "reports" / "15_mps_order" / "exp001"
             / "aggregate.json").exists(),
        reason=f"{ARTIFACT_ONLY} exp001's evidence is not in this tree")
    def test_the_parent_aggregate_is_the_one_the_recomputation_cites(self, mod):
        cited = self.stored()["recomputed_from"]["aggregate_sha256"]
        assert cited == mod.sha256_file(self.exp_dir() / "aggregate.json")

    @pytest.mark.skipif(
        not (ROOT / "data" / "reports" / "15_mps_order" / "exp001"
             / "aggregate.recomputed.json").exists(),
        reason=f"{ARTIFACT_ONLY} exp001's recomputation is not in this tree")
    def test_the_recomputation_compared_nothing_because_nothing_paired(self,
                                                                       mod):
        recomputed = self.stored()
        assert recomputed["losses"]["comparable_pairs"] == 0
        assert recomputed["losses"]["verdict"] == "not_applicable"
        assert recomputed["headline"]["allowed"] is False

    @pytest.mark.skipif(
        not (ROOT / "data" / "reports" / "15_mps_order" / "exp001"
             / "aggregate.recomputed.json").exists(),
        reason=f"{ARTIFACT_ONLY} exp001's recomputation is not in this tree")
    def test_it_says_which_conclusion_it_replaced(self, mod):
        assert [d["field"] for d in
                self.stored()["differences_from_parent_aggregate"]] == ["losses"]

    @pytest.mark.skipif(
        not (ROOT / "data" / "reports" / "15_mps_order" / "exp001"
             / "aggregate.recomputed.json").exists(),
        reason=f"{ARTIFACT_ONLY} exp001's recomputation is not in this tree")
    def test_the_stored_recomputation_still_recomputes(self, mod):
        exp = mod.load_experiment(self.exp_dir())
        stored_agg = json.loads((self.exp_dir() / "aggregate.json").read_text())
        assert mod.check_recomputed_matches(self.exp_dir(), exp,
                                            stored_agg) == []

    @pytest.mark.skipif(
        not (ROOT / "data" / "reports" / "15_mps_order" / "exp001"
             / "aggregate.recomputed.json").exists(),
        reason=f"{ARTIFACT_ONLY} exp001's recomputation is not in this tree")
    def test_a_tampered_recomputation_may_not_be_cited(self, mod, tmp_path,
                                                       monkeypatch):
        import shutil

        exp_dir = tmp_path / "exp001"
        shutil.copytree(self.exp_dir(), exp_dir)
        shutil.copy(ROOT / "data" / "reports" / "15_mps_order"
                    / "calibration.json", tmp_path / "calibration.json")
        monkeypatch.setattr(mod, "ROOT", tmp_path)
        # Report paths inside the aggregate are root-relative; re-point them.
        agg = json.loads((exp_dir / "aggregate.json").read_text())
        for c in agg["children"]:
            if c.get("report_path"):
                c["report_path"] = f"exp001/{c['run_id']}.json"
        (exp_dir / "aggregate.json").unlink()
        (exp_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
        plan = json.loads((exp_dir / "plan.json").read_text())
        plan["calibration_path"] = "calibration.json"
        (exp_dir / "plan.json").unlink()
        (exp_dir / "plan.json").write_text(json.dumps(plan, indent=2))

        rec = json.loads((exp_dir / "aggregate.recomputed.json").read_text())
        rec["losses"]["verdict"] = "within_tolerance"
        (exp_dir / "aggregate.recomputed.json").unlink()
        (exp_dir / "aggregate.recomputed.json").write_text(
            json.dumps(rec, indent=2))

        exp = mod.load_experiment(exp_dir)
        problems = mod.check_recomputed_matches(exp_dir, exp, agg)
        assert any("`losses` is not what recomputing it now produces" in p
                   for p in problems)

    # The rendered report is published; the run evidence it cites is not.
    # These were one test, guarded on the markdown -- so in a tree with the
    # report but without exp001 it did not skip, it failed on a missing file.
    # Split, so the prose claims are checked wherever the report exists and
    # only the digest comparison waits for the evidence.
    @pytest.mark.skipif(
        not REPORT_15_MD.exists(),
        reason=f"{ARTIFACT_ONLY} report 15 has not been rendered here")
    def test_the_markdown_says_the_parent_aggregate_was_left_alone(self, mod):
        text = REPORT_15_MD.read_text()
        assert "aggregate.recomputed.json" in text
        assert "is unchanged and was not written to" in text

    @pytest.mark.skipif(
        not (ROOT / "data" / "reports" / "15_mps_order" / "exp001"
             / "aggregate.recomputed.json").exists(),
        reason=f"{ARTIFACT_ONLY} exp001's recomputation is not in this tree")
    def test_the_markdown_cites_the_recomputed_aggregate(self, mod):
        text = REPORT_15_MD.read_text()
        # Both digests it cites are the files on disk, not a remembered pair.
        assert mod.sha256_file(
            self.exp_dir() / "aggregate.recomputed.json") in text
        assert mod.sha256_file(self.exp_dir() / "aggregate.json") in text

    @pytest.mark.skipif(
        not REPORT_15_MD.exists(),
        reason=f"{ARTIFACT_ONLY} report 15 has not been rendered here")
    def test_the_report_admits_r1s_source_was_not_preserved(self, mod):
        """r1 ran a version of the script that is no longer on disk."""
        text = REPORT_15_MD.read_text()
        assert "Not every run can be reproduced from what is on disk now" in text
        assert "no copy was kept" in text

    @pytest.mark.skipif(
        not (ROOT / "data" / "reports" / "15_mps_order" / "exp001"
             / "r1.json").exists(),
        reason=f"{ARTIFACT_ONLY} exp001's evidence is not in this tree")
    def test_r1_records_a_script_digest_no_longer_on_disk(self, mod):
        rec = json.loads((self.exp_dir() / "r1.json").read_text())
        assert rec["provenance"]["code_sha256"][
            "scripts/15_mps_order.py"].startswith("d7e68e")


class TestReport14IsNotDisturbed:
    """Report 15 is new tooling; report 14 keeps its own contract.

    The marker sits on the one method that reads report 14's record, not on
    the class: the other two check report 15's own schema number and that its
    script never imports report 14, and neither needs a file to do it.
    """

    @needs_report_14
    def test_report_14_json_is_still_schema_1(self):
        stored = json.loads(
            (ROOT / "data" / "reports" / "14_mps_speed.json").read_text())
        assert stored["env"]["schema_version"] == 1

    def test_report_15_uses_its_own_schema_number(self, mod):
        assert mod.EXPERIMENT_SCHEMA == 3

    def test_report_15_does_not_import_report_14(self):
        src = SCRIPT.read_text()
        assert "14_mps_speed" not in src
