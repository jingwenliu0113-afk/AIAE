"""Diagnostic bookkeeping: timing aggregation, stop conditions, report maths.

No model is loaded. The arithmetic is what can be wrong here, and a fake clock
checks it exactly -- a real 1B run would only make the same assertions slower
and less certain.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.diagnostics import (
    PhaseTimer,
    StopCondition,
    memory_sample,
    summarise_phases,
    window_stats,
)


class FakeClock:
    """Advances only when told to, so timings are exact rather than flaky."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class TestPhaseTimer:
    def test_each_phase_gets_its_own_elapsed(self):
        clock = FakeClock()
        timer = PhaseTimer(clock=clock)
        timer.start()
        clock.advance(0.5)
        timer.phase("forward")
        clock.advance(0.25)
        timer.phase("backward")
        row = timer.end()

        assert row["forward"] == pytest.approx(0.5)
        assert row["backward"] == pytest.approx(0.25)
        assert row["total"] == pytest.approx(0.75)

    def test_phases_do_not_double_count(self):
        clock = FakeClock()
        timer = PhaseTimer(clock=clock)
        timer.start()
        for _ in range(4):
            clock.advance(1.0)
            timer.phase(f"p{_}")
        row = timer.end()
        assert sum(row[f"p{i}"] for i in range(4)) == pytest.approx(row["total"])

    def test_device_is_synchronised_at_every_boundary(self):
        """Without the sync, an async backend's cost lands in the next phase."""
        calls = []
        clock = FakeClock()
        timer = PhaseTimer(sync=lambda: calls.append(clock()), clock=clock)
        timer.start()
        clock.advance(1)
        timer.phase("a")
        clock.advance(1)
        timer.phase("b")
        timer.end()
        assert len(calls) == 4      # start, a, b, end

    def test_extra_fields_are_carried_onto_the_row(self):
        timer = PhaseTimer(clock=FakeClock())
        timer.start()
        row = timer.end(row=7, n_tokens=123)
        assert row["row"] == 7 and row["n_tokens"] == 123

    def test_rows_accumulate(self):
        clock = FakeClock()
        timer = PhaseTimer(clock=clock)
        for _ in range(3):
            timer.start()
            clock.advance(0.1)
            timer.phase("forward")
            timer.end()
        assert len(timer.rows) == 3


class TestSummarisePhases:
    def rows(self):
        return [
            {"forward": 1.0, "backward": 2.0, "total": 3.5},
            {"forward": 3.0, "backward": 4.0, "total": 7.5},
        ]

    def test_totals_and_means(self):
        out = summarise_phases(self.rows(), ("forward", "backward"))
        assert out["forward"]["total_seconds"] == 4.0
        assert out["forward"]["mean_seconds"] == 2.0
        assert out["backward"]["total_seconds"] == 6.0

    def test_share_is_over_named_phases(self):
        out = summarise_phases(self.rows(), ("forward", "backward"))
        assert out["forward"]["share_of_measured"] == pytest.approx(0.4)
        assert out["backward"]["share_of_measured"] == pytest.approx(0.6)

    def test_unattributed_time_is_reported_not_hidden(self):
        """1.0s of the 11.0s total is in no named phase and must show up."""
        out = summarise_phases(self.rows(), ("forward", "backward"))
        assert out["_unattributed"]["total_seconds"] == pytest.approx(1.0)
        # Shares are rounded to 4dp for the report, so compare at that scale.
        assert out["_unattributed"]["share_of_total"] == pytest.approx(
            1.0 / 11.0, abs=1e-4)

    def test_a_missing_phase_is_skipped_not_zeroed(self):
        out = summarise_phases([{"forward": 1.0, "total": 1.0}],
                               ("forward", "optimizer"))
        assert "optimizer" not in out

    def test_empty_input_does_not_divide_by_zero(self):
        out = summarise_phases([], ("forward",))
        assert out["_unattributed"]["share_of_total"] == 0.0


class TestWindowStats:
    def rows(self, n, secs=1.0, tokens=100):
        return [{"total": secs, "n_tokens": tokens, "n_supervised": tokens // 2}
                for _ in range(n)]

    def test_windows_partition_the_rows(self):
        out = window_stats(self.rows(10), size=4)
        assert [w["n_rows"] for w in out] == [4, 4, 2]
        assert [w["rows"] for w in out] == ["1-4", "5-8", "9-10"]

    def test_a_short_final_window_is_kept(self):
        out = window_stats(self.rows(5), size=4)
        assert out[-1]["n_rows"] == 1

    def test_rates_are_per_window_not_cumulative(self):
        rows = self.rows(2, secs=1.0) + self.rows(2, secs=9.0)
        out = window_stats(rows, size=2)
        assert out[0]["seconds_per_row"] == 1.0
        assert out[1]["seconds_per_row"] == 9.0, "must not average with earlier"

    def test_tokens_per_second_uses_the_window_total(self):
        out = window_stats(self.rows(2, secs=2.0, tokens=100), size=2)
        assert out[0]["tokens"] == 200
        assert out[0]["tokens_per_second"] == pytest.approx(50.0)

    def test_mean_sequence_length_is_reported(self):
        out = window_stats(self.rows(4, tokens=80), size=4)
        assert out[0]["mean_seq_len"] == 80.0

    def test_zero_seconds_does_not_divide_by_zero(self):
        out = window_stats([{"total": 0.0, "n_tokens": 5}], size=1)
        assert out[0]["tokens_per_second"] is None


class TestStopCondition:
    def test_a_fast_run_never_stops(self):
        stop = StopCondition()
        for i in range(100):
            assert stop.check(elapsed=i, row_seconds=1.0) is None

    def test_three_consecutive_slow_rows_stop(self):
        stop = StopCondition(slow_row_seconds=30.0, slow_row_streak=3)
        assert stop.check(1, 31.0) is None
        assert stop.check(2, 31.0) is None
        reason = stop.check(3, 31.0)
        assert reason and "consecutive" in reason

    def test_the_streak_resets_on_a_fast_row(self):
        """Two slow, one fast, two slow must not trip a three-in-a-row rule."""
        stop = StopCondition(slow_row_seconds=30.0, slow_row_streak=3)
        assert stop.check(1, 31.0) is None
        assert stop.check(2, 31.0) is None
        assert stop.check(3, 1.0) is None
        assert stop.check(4, 31.0) is None
        assert stop.check(5, 31.0) is None

    def test_the_time_budget_stops_even_when_rows_are_fast(self):
        stop = StopCondition(max_seconds=60)
        assert stop.check(elapsed=59, row_seconds=0.1) is None
        reason = stop.check(elapsed=61, row_seconds=0.1)
        assert reason and "minutes" in reason

    def test_a_row_exactly_at_the_threshold_is_not_slow(self):
        stop = StopCondition(slow_row_seconds=30.0, slow_row_streak=1)
        assert stop.check(1, 30.0) is None
        assert stop.check(2, 30.01) is not None


class TestMemorySample:
    class FakeMPS:
        @staticmethod
        def current_allocated_memory():
            return 2 * 1024 ** 3

        @staticmethod
        def driver_allocated_memory():
            return 5 * 1024 ** 3

    def test_both_allocations_are_recorded(self):
        """Tracked and driver figures can diverge; both are needed."""
        torch_mod = type("T", (), {"mps": self.FakeMPS})
        out = memory_sample(torch_mod, rss_bytes=1024 ** 3)
        assert out["mps_current_allocated_gb"] == 2.0
        assert out["mps_driver_allocated_gb"] == 5.0
        assert out["peak_process_rss_gb"] == 1.0

    def test_an_unavailable_reading_is_none_not_missing(self):
        """'Not available' must be distinguishable from 'not sampled'."""
        torch_mod = type("T", (), {"mps": self.FakeMPS})
        out = memory_sample(torch_mod)
        assert "mps_recommended_max_gb" in out
        assert out["mps_recommended_max_gb"] is None
        assert out["peak_process_rss_gb"] is None

    def test_a_backend_without_mps_does_not_raise(self):
        out = memory_sample(type("T", (), {}))
        assert out["mps_current_allocated_gb"] is None

    def test_a_raising_probe_is_caught(self):
        class Boom:
            @staticmethod
            def current_allocated_memory():
                raise RuntimeError("no device")

        out = memory_sample(type("T", (), {"mps": Boom}))
        assert out["mps_current_allocated_gb"] is None


class TestDiagnosticScriptContract:
    """Guards the parts of the script that are policy rather than code."""

    def script(self) -> str:
        return (Path(__file__).resolve().parents[1]
                / "scripts" / "14_mps_speed_diagnostic.py").read_text()

    def test_it_never_writes_the_smoke_checkpoint(self):
        """report 13's checkpoint must survive this script untouched.

        Checked over code lines only: the docstring names the path when
        promising not to write it, and a whole-file scan would fail on the
        promise itself.
        """
        import ast

        src = self.script()
        assert "save_pretrained" not in src

        tree = ast.parse(src)
        # The risk is a *path* into the checkpoint, not the word appearing in
        # prose -- the report text names the directory precisely to promise it
        # is left alone. So: no name may be bound to a value mentioning it.
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for s in ast.walk(node.value):
                    if isinstance(s, ast.Constant) and isinstance(s.value, str):
                        assert "lora_smoke" not in s.value, (
                            f"binds a path into the smoke checkpoint: {s.value}")
        assert "CKPT_DIR" not in {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    def test_the_row_cap_is_at_most_200(self):
        import re

        m = re.search(r"^MAX_ROWS = (\d+)", self.script(), re.M)
        assert m and int(m.group(1)) <= 200

    def test_both_declared_conditions_are_present(self):
        import re

        m = re.search(r"^CONDITIONS = \((.*?)\)", self.script(), re.M)
        assert m and "continuous" in m.group(1) and "empty_cache" in m.group(1)

    def test_process_restart_is_excluded_with_a_stated_reason(self):
        src = self.script()
        assert "confound" in src and "restart" in src.lower()

    def test_perf_counter_and_synchronize_are_both_used(self):
        src = self.script()
        assert "perf_counter" in src
        assert "torch.mps.synchronize()" in src


class FakeMPS:
    def __init__(self, *, sync_raises=False, empty_raises=False):
        self.sync_calls = 0
        self.empty_calls = 0
        self.sync_raises = sync_raises
        self.empty_raises = empty_raises

    def synchronize(self):
        self.sync_calls += 1
        if self.sync_raises:
            raise RuntimeError("device lost")

    def empty_cache(self):
        self.empty_calls += 1
        if self.empty_raises:
            raise RuntimeError("cannot free")


def fake_torch(**kw):
    return type("T", (), {"mps": FakeMPS(**kw)})


class TestDeviceOps:
    """Three paths: it works, it fails, or there is no device.

    A swallowed failure is the dangerous case -- the run keeps going and the
    report still looks complete, while every timing after it measures enqueue
    rather than compute.
    """

    def test_mps_success_calls_through(self):
        from src.training.diagnostics import DeviceOps

        t = fake_torch()
        dev = DeviceOps("mps", torch_mod=t)
        dev.sync()
        dev.empty_cache()
        assert t.mps.sync_calls == 1
        assert t.mps.empty_calls == 1
        assert dev.scheduled_empty_cache_calls == 1

    def test_a_failing_sync_raises_on_mps(self):
        from src.training.diagnostics import DeviceOps

        dev = DeviceOps("mps", torch_mod=fake_torch(sync_raises=True))
        with pytest.raises(RuntimeError, match="synchronize"):
            dev.sync()

    def test_the_sync_failure_explains_why_it_matters(self):
        from src.training.diagnostics import DeviceOps

        dev = DeviceOps("mps", torch_mod=fake_torch(sync_raises=True))
        with pytest.raises(RuntimeError) as e:
            dev.sync()
        assert "enqueue" in str(e.value)

    def test_a_failing_empty_cache_raises_on_mps(self):
        from src.training.diagnostics import DeviceOps

        dev = DeviceOps("mps", torch_mod=fake_torch(empty_raises=True))
        with pytest.raises(RuntimeError, match="empty_cache"):
            dev.empty_cache()

    def test_cpu_is_an_explicit_no_op(self):
        """No device to sync and no MPS cache: different meaning, not an error."""
        from src.training.diagnostics import DeviceOps

        t = fake_torch(sync_raises=True, empty_raises=True)
        dev = DeviceOps("cpu", torch_mod=t)
        dev.sync()
        dev.empty_cache()
        assert t.mps.sync_calls == 0 and t.mps.empty_calls == 0

    def test_cpu_no_op_does_not_count_a_clear(self):
        from src.training.diagnostics import DeviceOps

        dev = DeviceOps("cpu", torch_mod=fake_torch())
        dev.empty_cache()
        dev.empty_cache(teardown=True)
        assert dev.scheduled_empty_cache_calls == 0
        assert dev.teardown_empty_cache_calls == 0

    def test_clear_calls_are_counted_for_the_report(self):
        from src.training.diagnostics import DeviceOps

        dev = DeviceOps("mps", torch_mod=fake_torch())
        for _ in range(3):
            dev.empty_cache()
        assert dev.scheduled_empty_cache_calls == 3

    def test_a_timer_driven_by_device_ops_syncs_every_boundary(self):
        from src.training.diagnostics import DeviceOps, PhaseTimer

        t = fake_torch()
        dev = DeviceOps("mps", torch_mod=t)
        timer = PhaseTimer(sync=dev.sync, clock=FakeClock())
        timer.start()
        timer.phase("forward")
        timer.end()
        assert t.mps.sync_calls == 3


class TestMemoryFieldNames:
    """Names must say what was measured, not something shorter."""

    def test_rss_is_named_as_a_peak(self):
        from src.training.diagnostics import memory_sample

        out = memory_sample(fake_torch(), rss_bytes=1024 ** 3)
        assert "peak_process_rss_gb" in out
        assert "process_rss_gb" not in out, "ru_maxrss is a high-water mark"

    def test_free_is_named_free_plus_inactive(self):
        from src.training.diagnostics import system_memory

        out = system_memory()
        assert "free_plus_inactive_gb" in out
        assert "free_gb" not in out, "inactive pages are reclaimable, not free"

    def test_memory_pressure_is_collected(self):
        from src.training.diagnostics import system_memory

        assert "memory_pressure_percent_free" in system_memory()


class TestReportUsesStoredSettings:
    """A re-render must describe the run it is re-rendering."""

    def script(self) -> str:
        return (Path(__file__).resolve().parents[1]
                / "scripts" / "14_mps_speed_diagnostic.py").read_text()

    def test_the_writer_reads_settings_from_env(self):
        import ast

        tree = ast.parse(self.script())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_write_report")
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        for const in ("MAX_ROWS", "WINDOW", "MEMORY_EVERY", "EMPTY_CACHE_EVERY",
                      "PHASES", "CONDITIONS", "StopCondition",
                      "SCHEMA_VERSION"):
            assert const not in names, (
                f"_write_report reads {const}; an old run would be re-rendered "
                "with today's settings")

    def test_required_provenance_keys_are_declared(self):
        """Every schema demands the base record, whatever else it adds."""
        mod = diag_module()
        for contract in mod.CONTRACTS.values():
            for key in ("instruction_sha256", "selection_digest",
                        "training_order_digest", "base_revision"):
                assert key in contract.required_provenance, (
                    f"schema {contract.version} does not require {key}")

    def test_loss_equality_is_computed_not_asserted(self):
        """The claim must be derived from the data, not printed regardless."""
        src = self.script()
        assert "same_loss = sum(" in src
        assert "same_tokens = sum(" in src
        assert "{same_loss}/{len(shared)}" in src


class TestClearAccounting:
    """A scheduled clear is the intervention; a teardown clear is housekeeping.

    One counter for both produced a specific, wrong report: the control arm
    was credited with one `empty_cache()` it never scheduled, and the
    treatment arm with one more than its schedule -- both of them attributed
    to a period that had already stopped being timed.
    """

    def dev(self):
        from src.training.diagnostics import DeviceOps

        return DeviceOps("mps", torch_mod=fake_torch(), clock=FakeClock())

    def test_the_two_kinds_are_counted_apart(self):
        dev = self.dev()
        dev.empty_cache()
        dev.empty_cache()
        dev.empty_cache(teardown=True)
        assert dev.scheduled_empty_cache_calls == 2
        assert dev.teardown_empty_cache_calls == 1

    def test_a_teardown_never_lands_in_the_scheduled_count(self):
        """The control arm's report said 1 when the truth was 0."""
        dev = self.dev()
        dev.empty_cache(teardown=True)
        assert dev.scheduled_empty_cache_calls == 0
        assert dev.scheduled_clear_cost()["calls"] == 0

    def test_the_treatment_arm_reports_its_schedule_not_schedule_plus_one(self):
        dev = self.dev()
        for _ in range(20):            # every 10 rows over 200 rows
            dev.empty_cache()
        dev.empty_cache(teardown=True)
        assert dev.scheduled_empty_cache_calls == 20

    def test_scheduled_calls_are_timed_one_by_one(self):
        from src.training.diagnostics import DeviceOps

        clock = FakeClock()

        class Slow(FakeMPS):
            def empty_cache(inner):
                clock.advance(0.25)
                FakeMPS.empty_cache(inner)

        dev = DeviceOps("mps", torch_mod=type("T", (), {"mps": Slow()}),
                        clock=clock)
        assert dev.empty_cache() == pytest.approx(0.25)
        dev.empty_cache()
        cost = dev.scheduled_clear_cost()
        assert cost["per_call_seconds"] == [0.25, 0.25]
        assert cost["total_seconds"] == pytest.approx(0.5)
        assert cost["mean_seconds"] == pytest.approx(0.25)

    def test_a_teardown_is_not_timed_into_the_intervention_cost(self):
        from src.training.diagnostics import DeviceOps

        clock = FakeClock()

        class Slow(FakeMPS):
            def empty_cache(inner):
                clock.advance(1.0)
                FakeMPS.empty_cache(inner)

        dev = DeviceOps("mps", torch_mod=type("T", (), {"mps": Slow()}),
                        clock=clock)
        dev.empty_cache(teardown=True)
        assert dev.scheduled_clear_cost()["total_seconds"] == 0.0

    def test_no_scheduled_clears_gives_no_cost_rather_than_a_fake_zero_mean(self):
        cost = self.dev().scheduled_clear_cost()
        assert cost["calls"] == 0
        assert cost["mean_seconds"] is None and cost["max_seconds"] is None

    def test_a_failing_teardown_still_raises(self):
        from src.training.diagnostics import DeviceOps

        dev = DeviceOps("mps", torch_mod=fake_torch(empty_raises=True))
        with pytest.raises(RuntimeError, match="empty_cache"):
            dev.empty_cache(teardown=True)


class TestWindowEndToEnd:
    """Windows report compute and end-to-end apart, and never guess."""

    def rows(self, n, secs=1.0, e2e=None):
        out = []
        for _ in range(n):
            r = {"total": secs, "n_tokens": 100, "n_supervised": 50}
            if e2e is not None:
                r["end_to_end"] = e2e
            out.append(r)
        return out

    def test_both_are_reported_when_both_were_measured(self):
        out = window_stats(self.rows(2, secs=1.0, e2e=1.5), size=2)
        assert out[0]["seconds"] == 2.0
        assert out[0]["end_to_end_seconds"] == 3.0
        assert out[0]["end_to_end_seconds_per_row"] == 1.5

    def test_a_run_without_end_to_end_reports_none_not_the_compute_figure(self):
        out = window_stats(self.rows(2, secs=1.0), size=2)
        assert out[0]["seconds_per_row"] == 1.0
        assert out[0]["end_to_end_seconds"] is None
        assert out[0]["end_to_end_seconds_per_row"] is None

    def test_a_partly_measured_window_is_unknown_not_undercounted(self):
        """Summing only the rows that have it would read as a smaller total."""
        rows = self.rows(1, e2e=2.0) + self.rows(1)
        out = window_stats(rows, size=2)
        assert out[0]["end_to_end_seconds"] is None


ROOT = Path(__file__).resolve().parents[1]

#: Report 14's per-row record. 200 measured rows and 41 memory samples: it is
#: per-record evidence, so it stays in the private research tree and is not
#: published. Tests that read it are artifact-only; the aggregate Markdown
#: beside it is published and its tests are not guarded.
REPORT_14_JSON = ROOT / "data" / "reports" / "14_mps_speed.json"
#: The rendered report is a per-row narrative of the record above, so it is
#: withheld too. The renderer that produces it is published.
REPORT_14_MD = ROOT / "data" / "reports" / "14_mps_speed.md"
INSTRUCTION_POOL = ROOT / "data" / "processed" / "instruct_inv_train.jsonl"

#: One prefix, so a public snapshot can enumerate exactly which skips are
#: allowed and a behavioural test that started skipping would stand out.
ARTIFACT_ONLY = "artifact-only:"

needs_report_14 = pytest.mark.skipif(
    not REPORT_14_JSON.exists(),
    reason=f"{ARTIFACT_ONLY} report 14's per-row record is not in this tree")
needs_report_14_md = pytest.mark.skipif(
    not REPORT_14_MD.exists(),
    reason=f"{ARTIFACT_ONLY} report 14's rendered report is not in this tree")
needs_instruction_pool = pytest.mark.skipif(
    not INSTRUCTION_POOL.exists(),
    reason=f"{ARTIFACT_ONLY} the instruction pool is not in this tree; "
           "replay checks its digest")
SCRIPT = ROOT / "scripts" / "14_mps_speed_diagnostic.py"


def diag_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("s14x", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass resolves its annotations through
    # sys.modules[cls.__module__], so a module loaded by spec alone cannot
    # define one.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def lora_config_fixture() -> dict:
    """The real field set, with the two env cross-checks pinned.

    Built from ``LoraConfig_`` rather than hand-listed so that a new
    hyperparameter shows up here as a failing gate rather than as a quietly
    incomplete record.
    """
    from src.training.lora import LoraConfig_

    return {**LoraConfig_().as_dict(), "seed": 0, "grad_accum": 8}


def stored_run(mod, tmp_path, *, schema=2, phases=("alpha", "beta"),
               max_rows=4, window=2, clear_every=2, slow_streak=7,
               slow_seconds=99.0, max_seconds=600.0, loss_dp=None):
    """A complete, self-consistent stored run, with settings unlike today's.

    The settings are deliberately nothing like the module constants: if the
    renderer reaches for a constant the output says 200 rows, four familiar
    phase names and a streak of three, and the test can see it.
    """
    from src.training.diagnostics import summarise_phases, window_stats

    data = tmp_path / "instruct_inv_train.jsonl"
    data.write_text("row\n")

    def rows(per_window):
        out = []
        for i in range(1, max_rows + 1):
            secs = per_window[(i - 1) // window]
            r = {p: secs / len(phases) for p in phases}
            r.update(total=secs, row=i, sample_id=f"s{i}", n_tokens=100,
                     n_supervised=50, loss=0.1234567890123,
                     end_to_end=secs + 0.5)
            out.append(r)
        return out

    ids = [f"s{i}" for i in range(1, max_rows + 1)]
    digest = mod.digest_ids(ids)

    def condition(name, per_window, order):
        r = rows(per_window)
        scheduled = max_rows // clear_every if name == "empty_cache" else 0
        return {
            "condition": name, "run_order": order,
            "rows_completed": max_rows, "rows_requested": max_rows,
            "stopped_early": None,
            "input_order_digest": digest, "completed_input_digest": digest,
            "end_to_end_seconds": 10.0, "model_compute_seconds": 9.0,
            "between_row_overhead_seconds": 1.0,
            "between_row_overhead_breakdown": {
                "scheduled_empty_cache_seconds": 0.4,
                "memory_probe_seconds": 0.5,
                "unattributed_seconds": 0.1},
            "end_to_end_seconds_per_row": 2.5,
            "model_compute_seconds_per_row": 2.25,
            "scheduled_empty_cache_every": (clear_every if name == "empty_cache"
                                            else None),
            "scheduled_empty_cache_calls": scheduled,
            "scheduled_empty_cache_cost": {
                "calls": scheduled, "total_seconds": 0.4,
                "mean_seconds": 0.2, "max_seconds": 0.3,
                "per_call_seconds": [0.2, 0.2]},
            "teardown_empty_cache_calls": 1,
            "phases": summarise_phases(r, tuple(phases)),
            "windows": window_stats(r, window),
            "memory": [], "per_row": r, "trainable_parameters": 1,
        }

    order = ["continuous", "empty_cache"]
    stop_conditions = {"slow_row_seconds": slow_seconds,
                       "slow_row_streak": slow_streak,
                       "max_seconds": max_seconds}
    packages = {"python": "3.13.9", "torch": "2.13.0",
                "transformers": "5.15.0", "peft": "0.20.0"}
    env = {
        "schema_version": schema, "max_rows_per_condition": max_rows,
        "window": window, "memory_sample_every": 2,
        "empty_cache_every": clear_every, "seed": 0, "grad_accum": 8,
        "phases": list(phases), "condition_order": list(order),
        # The arms' definitions travel with the run, not with the renderer.
        "condition_definitions": {
            "continuous": "one uninterrupted run",
            "empty_cache": f"clears the cache every {clear_every} rows"},
        "stop_slow_row_seconds": slow_seconds,
        "stop_slow_row_streak": slow_streak,
        "stop_max_seconds": max_seconds, "loss_decimals_stored": loss_dp,
        "device": "mps", "dtype": "bfloat16", **packages,
    }
    provenance = {
        "instruction_sha256": {"instruct_inv_train.jsonl":
                               mod.sha256_file(data)},
        "selection_digest": digest, "training_order_digest": digest,
        "base_revision": "abc", "published_adapter_revision": "def",
        # Everything below is recorded before the model loads.
        "head": "0" * 40,
        # False on a clean tree: legal, and the check must not read it as
        # absent.
        "working_tree_dirty": False,
        "code_sha256": {f: "a" * 64 for f in mod.CODE_FILES},
        "lora_config": lora_config_fixture(),
        "packages": dict(packages),
        "device": "mps", "dtype": "bfloat16",
        "phases": list(phases),
        "stop_conditions": dict(stop_conditions),
        "condition_order": list(order),
        "condition_input_order_digests": {c: digest for c in order},
        "backfilled_after_the_run": [],
    }
    return {"env": env, "baseline_memory": {},
            "provenance": provenance,
            "conditions": [condition("continuous", [1.0, 5.0], 0),
                           condition("empty_cache", [1.0, 1.0], 1)]}


#: What a schema-1 record simply does not have. Stripped when a test needs to
#: prove the gate still accepts the original 200-row run.
SCHEMA1_ABSENT = ("head", "working_tree_dirty", "code_sha256", "lora_config",
                  "packages", "device", "dtype", "phases", "stop_conditions",
                  "condition_order", "condition_input_order_digests")


@pytest.fixture
def diag(tmp_path, monkeypatch):
    mod = diag_module()
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path)
    monkeypatch.setattr(mod, "REPORT_DIR", tmp_path / "reports")
    return mod


class TestRawLossIsStored:
    """A loss rounded on the way to disk can never be recovered afterwards.

    Report 14's first run stored four decimal places, which is why its two
    conditions can only be said to agree *at that precision*. The point of
    fixing this is that the next run does not inherit the same ceiling.
    """

    def test_the_loss_is_written_exactly_as_measured(self, diag):
        row = {"total": 1.23456789, "loss": 0.987654321098765, "row": 1}
        assert diag._serialise_row(row)["loss"] == 0.987654321098765

    def test_timings_are_still_rounded(self, diag):
        out = diag._serialise_row({"total": 1.2345678901, "loss": 0.5})
        assert out["total"] == round(1.2345678901, diag.ROW_TIMING_DECIMALS)

    def test_non_floats_pass_through(self, diag):
        out = diag._serialise_row({"row": 3, "sample_id": "s3", "loss": 0.5})
        assert out["row"] == 3 and out["sample_id"] == "s3"

    def test_loss_is_the_declared_exception(self, diag):
        assert "loss" in diag.UNROUNDED_ROW_FIELDS

    def test_a_new_run_declares_its_losses_unrounded(self, diag):
        """`loss_decimals_stored` is what the report quotes its claim at."""
        src = SCRIPT.read_text()
        assert '"loss_decimals_stored": None' in src


class TestRendererUsesOnlyStoredSettings:
    """Re-rendering an old run must describe *that* run."""

    def render(self, mod, stored):
        mod._write_report(stored["env"], stored["baseline_memory"],
                          stored["conditions"], stored["provenance"])
        return (mod.REPORT_DIR / "14_mps_speed.md").read_text()

    def test_the_title_row_cap_comes_from_the_run(self, diag, tmp_path):
        out = self.render(diag, stored_run(diag, tmp_path))
        assert out.startswith("# MPS speed diagnostic (<= 4 rows)")
        assert "200 rows per condition" not in out

    def test_the_phase_names_come_from_the_run(self, diag, tmp_path):
        out = self.render(diag, stored_run(diag, tmp_path))
        assert "`alpha`, `beta`" in out
        assert "collate_h2d" not in out

    def test_the_slow_row_streak_and_threshold_come_from_the_run(
            self, diag, tmp_path):
        out = self.render(diag, stored_run(diag, tmp_path))
        assert "- 7 consecutive rows over 99s" in out
        assert "three consecutive" not in out

    def test_the_condition_time_limit_comes_from_the_run(self, diag, tmp_path):
        out = self.render(diag, stored_run(diag, tmp_path))
        assert "- or 10 minutes in one condition" in out

    def test_the_clear_interval_window_and_memory_interval_come_from_the_run(
            self, diag, tmp_path):
        out = self.render(diag, stored_run(diag, tmp_path))
        assert "With `empty_cache()` every 2 rows" in out
        assert "windows of 2 rows" in out
        assert "memory sampled every 2 rows" in out
        assert "every 10 rows" not in out, "10 is the module constant"

    def test_the_condition_order_comes_from_the_run(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path)
        stored["env"]["condition_order"] = ["empty_cache", "continuous"]
        stored["conditions"].reverse()
        for i, c in enumerate(stored["conditions"]):
            c["run_order"] = i
        out = self.render(diag, stored)
        assert "`empty_cache -> continuous`" in out

    def test_a_missing_setting_fails_instead_of_falling_back(self, diag,
                                                             tmp_path):
        for key in ("max_rows_per_condition", "phases", "condition_order",
                    "stop_slow_row_seconds", "stop_slow_row_streak",
                    "stop_max_seconds", "window", "memory_sample_every",
                    "empty_cache_every", "loss_decimals_stored"):
            stored = stored_run(diag, tmp_path)
            del stored["env"][key]
            with pytest.raises(SystemExit, match=key):
                self.render(diag, stored)

    def test_the_condition_definitions_come_from_the_run(self, diag, tmp_path):
        """What the arms differed by is part of the record, not the renderer."""
        stored = stored_run(diag, tmp_path)
        stored["env"]["condition_definitions"]["empty_cache"] = "wiped hourly"
        out = self.render(diag, stored)
        assert "| `empty_cache` | wiped hourly |" in out

    def test_an_unknown_condition_is_refused_not_described_as_another(
            self, diag, tmp_path):
        stored = stored_run(diag, tmp_path)
        stored["env"]["condition_order"] = ["continuous", "mystery"]
        with pytest.raises(SystemExit, match="mystery"):
            self.render(diag, stored)

    def test_a_condition_without_a_stored_definition_fails_closed(
            self, diag, tmp_path):
        stored = stored_run(diag, tmp_path)
        del stored["env"]["condition_definitions"]["empty_cache"]
        with pytest.raises(SystemExit, match="records no definition"):
            self.render(diag, stored)

    def test_the_renderer_holds_no_condition_descriptions_of_its_own(self):
        """No constant to fall back on means no way to describe a run wrongly."""
        assert "CONDITION_BLURBS" not in SCRIPT.read_text()

    def test_an_uncounted_teardown_claims_no_call(self, diag, tmp_path):
        """`null` means unknown. It must not read as 'one teardown ran'."""
        stored = stored_run(diag, tmp_path, schema=1)
        for c in stored["conditions"]:
            c["teardown_empty_cache_calls"] = None
        out = self.render(diag, stored)
        assert f"teardown `empty_cache()`: **{diag.UNKNOWN}**" in out
        assert "call(s), made after" not in out
        assert "how many were made -- if any -- is not known" in out

    def test_a_counted_teardown_shows_the_count(self, diag, tmp_path):
        out = self.render(diag, stored_run(diag, tmp_path, schema=2))
        assert "teardown `empty_cache()`: **1** call(s), made after" in out

    def test_unrounded_losses_are_claimed_at_full_precision(self, diag,
                                                            tmp_path):
        out = self.render(diag, stored_run(diag, tmp_path, loss_dp=None))
        assert "the same loss as stored" in out
        assert "decimal places" not in out

    def test_rounded_losses_are_claimed_only_at_their_precision(self, diag,
                                                               tmp_path):
        out = self.render(diag, stored_run(diag, tmp_path, loss_dp=4))
        assert "stored precision of 4 decimal places" in out

    def test_the_clear_counts_are_reported_apart(self, diag, tmp_path):
        out = self.render(diag, stored_run(diag, tmp_path))
        assert "scheduled `empty_cache()`: **2** calls" in out
        assert "teardown `empty_cache()`: **1** call(s)" in out
        assert "inside none of the figures above" in out

    def test_an_unrecorded_overhead_split_stays_unknown(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path, schema=1)
        for c in stored["conditions"]:
            c["between_row_overhead_breakdown"] = None
            c["teardown_empty_cache_calls"] = None
        out = self.render(diag, stored)
        assert "is not recorded for this run" in out
        assert f"teardown `empty_cache()`: **{diag.UNKNOWN}**" in out


class TestReportDoesNotOverclaim:
    """The wording the review struck out must not come back.

    Checked against both the renderer and the report it produced, because
    either one alone can drift: prose edited only in the .md is overwritten by
    the next render, and prose edited only in the script is invisible until
    someone re-renders.
    """

    BANNED = (
        ("identical, plus", "the conditions differ in more than the clear"),
        ("It trains nothing", "optimizer updates do run, in memory"),
        ("shown to remove the slowdown", "n=1 under one fixed order"),
        ("on identical inputs", "same rows, not proven identical arithmetic"),
        ("identical, flat", "the two tracked ranges are close, not identical"),
        ("free memory falls", "free+inactive pages are reclaimable"),
    )

    def sources(self):
        return {"the script": SCRIPT.read_text(),
                "the report": REPORT_14_MD.read_text()}

    # The renderer is published; the rendered report is not. These were one
    # test reading both, so withholding the report turned a check on the
    # *script* into a failure. The renderer half runs everywhere.
    def test_no_banned_phrasing_survives_in_the_renderer(self):
        text = SCRIPT.read_text()
        for phrase, why in self.BANNED:
            assert phrase not in text, f"the script: {phrase!r} -- {why}"

    @needs_report_14_md
    def test_no_banned_phrasing_survives_in_the_report(self):
        text = REPORT_14_MD.read_text()
        for phrase, why in self.BANNED:
            assert phrase not in text, f"the report: {phrase!r} -- {why}"

    @needs_report_14_md
    def test_the_report_still_names_the_order_confound(self):
        text = REPORT_14_MD.read_text()
        assert "confounded with condition" in text
        assert "n=1" in text

    @needs_report_14_md
    def test_the_report_states_that_optimizer_updates_run(self):
        assert "Optimizer updates do run" in REPORT_14_MD.read_text()

    @needs_report_14_md
    def test_the_stored_run_is_still_labelled_four_decimal(self):
        """Its losses were rounded on the way to disk; that cannot change."""
        assert ("stored precision of 4 decimal places"
                in REPORT_14_MD.read_text())


class TestReplayRejectsTampering:
    """`--from-json` must refuse a record that has been edited under it."""

    def test_an_untouched_run_is_accepted(self, diag, tmp_path):
        diag.check_replayable(stored_run(diag, tmp_path))

    def test_changed_input_data_is_rejected(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path)
        (tmp_path / "instruct_inv_train.jsonl").write_text("different\n")
        with pytest.raises(SystemExit, match="changed"):
            diag.check_replayable(stored)

    def test_missing_input_data_is_rejected(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path)
        (tmp_path / "instruct_inv_train.jsonl").unlink()
        with pytest.raises(SystemExit, match="is missing"):
            diag.check_replayable(stored)

    def test_a_missing_required_setting_is_rejected(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path)
        del stored["env"]["stop_slow_row_streak"]
        with pytest.raises(SystemExit, match="stop_slow_row_streak"):
            diag.check_replayable(stored)

    def test_missing_provenance_is_rejected(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path)
        del stored["provenance"]["selection_digest"]
        with pytest.raises(SystemExit, match="selection_digest"):
            diag.check_replayable(stored)

    def test_a_reordered_condition_list_is_rejected(self, diag, tmp_path):
        """Order is confounded with condition, so it must not drift."""
        stored = stored_run(diag, tmp_path)
        stored["conditions"].reverse()
        with pytest.raises(SystemExit, match="run_order|do not match"):
            diag.check_replayable(stored)

    def test_a_renamed_condition_is_rejected(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path)
        stored["conditions"][1]["condition"] = "something_else"
        with pytest.raises(SystemExit, match="do not match the recorded order"):
            diag.check_replayable(stored)

    def test_a_clear_count_that_contradicts_the_schedule_is_rejected(
            self, diag, tmp_path):
        stored = stored_run(diag, tmp_path)
        stored["conditions"][1]["scheduled_empty_cache_calls"] += 1
        with pytest.raises(SystemExit, match="implies"):
            diag.check_replayable(stored)

    def test_a_teardown_folded_into_the_scheduled_count_is_rejected(
            self, diag, tmp_path):
        """The exact bug: schedule says 2, record says 3 because of teardown."""
        stored = stored_run(diag, tmp_path)
        c = stored["conditions"][1]
        c["scheduled_empty_cache_calls"] = (
            c["scheduled_empty_cache_calls"] + c["teardown_empty_cache_calls"])
        with pytest.raises(SystemExit, match="implies"):
            diag.check_replayable(stored)

    def test_a_control_arm_credited_with_a_clear_is_rejected(self, diag,
                                                             tmp_path):
        stored = stored_run(diag, tmp_path)
        stored["conditions"][0]["scheduled_empty_cache_calls"] = 1
        with pytest.raises(SystemExit, match="implies 0"):
            diag.check_replayable(stored)

    def test_an_edited_input_digest_is_rejected(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path)
        stored["conditions"][0]["completed_input_digest"] = "0" * 64
        with pytest.raises(SystemExit, match="digest to"):
            diag.check_replayable(stored)

    def test_an_edited_sample_id_is_rejected(self, diag, tmp_path):
        """Changing which rows ran must not slip past the digest."""
        stored = stored_run(diag, tmp_path)
        stored["conditions"][0]["per_row"][2]["sample_id"] = "swapped"
        with pytest.raises(SystemExit, match="digest to"):
            diag.check_replayable(stored)

    def test_conditions_fed_different_orders_are_rejected(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path)
        stored["conditions"][1]["input_order_digest"] = "f" * 64
        with pytest.raises(SystemExit, match="different row orders|"
                                             "does not match the recorded"):
            diag.check_replayable(stored)

    def test_dropped_rows_are_rejected(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path)
        stored["conditions"][0]["per_row"].pop()
        with pytest.raises(SystemExit, match="rows completed but stores"):
            diag.check_replayable(stored)

    def test_more_rows_than_the_declared_cap_are_rejected(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path)
        stored["env"]["max_rows_per_condition"] = 2
        with pytest.raises(SystemExit, match="over the 2-row cap"):
            diag.check_replayable(stored)

    def test_a_schema_2_run_must_carry_the_fields_it_promises(self, diag,
                                                              tmp_path):
        stored = stored_run(diag, tmp_path, schema=2)
        stored["conditions"][0]["teardown_empty_cache_calls"] = None
        with pytest.raises(SystemExit, match="teardown_empty_cache_calls"):
            diag.check_replayable(stored)

    def test_a_schema_1_run_is_allowed_its_known_gaps(self, diag, tmp_path):
        """The original 200-row run cannot be given fields it never had."""
        stored = stored_run(diag, tmp_path, schema=1)
        for key in SCHEMA1_ABSENT:
            del stored["provenance"][key]
        for c in stored["conditions"]:
            c["teardown_empty_cache_calls"] = None
            c["between_row_overhead_breakdown"] = None
            c["input_order_digest"] = None
            c["completed_input_digest"] = None
        diag.check_replayable(stored)


class TestSchema2ProvenanceIsRequired:
    """Schema 2 promises a full pre-load record; the gate holds it to that.

    Report 13 lost its provenance by recording it afterwards, and report 14
    inherited the same gap. A schema-2 run that is missing any of it, or whose
    provenance contradicts the settings stored beside it, must not re-render.
    """

    def test_a_complete_record_passes(self, diag, tmp_path):
        diag.check_replayable(stored_run(diag, tmp_path, schema=2))

    def test_every_required_field_is_required(self, diag, tmp_path):
        for key in diag.CONTRACTS[2].required_provenance:
            stored = stored_run(diag, tmp_path, schema=2)
            del stored["provenance"][key]
            with pytest.raises(SystemExit, match=key):
                diag.check_replayable(stored)

    @pytest.mark.parametrize("path,value", [
        (("phases",), ["something", "else"]),
        (("condition_order",), ["empty_cache", "continuous"]),
        (("device",), "cuda"),
        (("dtype",), "float32"),
        (("stop_conditions", "slow_row_seconds"), 5.0),
        (("stop_conditions", "slow_row_streak"), 99),
        (("stop_conditions", "max_seconds"), 1.0),
        (("packages", "torch"), "1.0.0"),
        (("packages", "peft"), "0.0.1"),
        (("lora_config", "seed"), 7),
        (("lora_config", "grad_accum"), 3),
    ])
    def test_provenance_contradicting_env_is_rejected(self, diag, tmp_path,
                                                      path, value):
        stored = stored_run(diag, tmp_path, schema=2)
        target = stored["provenance"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(SystemExit, match="disagree"):
            diag.check_replayable(stored)

    def test_a_condition_with_no_pre_declared_order_is_rejected(self, diag,
                                                                tmp_path):
        stored = stored_run(diag, tmp_path, schema=2)
        del stored["provenance"]["condition_input_order_digests"]["empty_cache"]
        with pytest.raises(SystemExit, match="no pre-run input-order digest"):
            diag.check_replayable(stored)

    def test_a_run_that_did_not_get_the_rows_it_was_promised_is_rejected(
            self, diag, tmp_path):
        """The declared value was fixed before the arm could run."""
        stored = stored_run(diag, tmp_path, schema=2)
        stored["provenance"]["condition_input_order_digests"][
            "empty_cache"] = "e" * 64
        with pytest.raises(SystemExit, match="declared for it before the model"):
            diag.check_replayable(stored)


class TestSchema2ProvenanceContentIsChecked:
    """Present-but-empty is the failure mode a key sweep cannot see.

    `head: null`, `code_sha256: {}`, a packages dict missing one of four --
    each of those passes a presence check and is still a hole where the record
    should be. That shape, a gap wearing the outline of a record, is what let
    report 13 look complete until someone tried to use it.
    """

    def rejects(self, diag, tmp_path, mutate, match):
        stored = stored_run(diag, tmp_path, schema=2)
        mutate(stored["provenance"])
        with pytest.raises(SystemExit, match=match):
            diag.check_replayable(stored)

    # --- HEAD -------------------------------------------------------------
    @pytest.mark.parametrize("value", [None, "", "   ", "not-a-sha", "abc"])
    def test_an_unusable_head_is_rejected(self, diag, tmp_path, value):
        self.rejects(diag, tmp_path, lambda p: p.update(head=value), "head")

    def test_a_real_head_passes(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path, schema=2)
        stored["provenance"]["head"] = "3f65552" + "a" * 33
        diag.check_replayable(stored)

    # --- working_tree_dirty ----------------------------------------------
    def test_a_clean_tree_passes(self, diag, tmp_path):
        """False is the value a well-behaved run records. It must pass."""
        stored = stored_run(diag, tmp_path, schema=2)
        stored["provenance"]["working_tree_dirty"] = False
        diag.check_replayable(stored)

    def test_a_dirty_tree_passes(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path, schema=2)
        stored["provenance"]["working_tree_dirty"] = True
        diag.check_replayable(stored)

    @pytest.mark.parametrize("value", [None, "false", 0, ""])
    def test_a_non_bool_dirty_flag_is_rejected(self, diag, tmp_path, value):
        """None means nobody looked, which is not the same as clean."""
        self.rejects(diag, tmp_path,
                     lambda p: p.update(working_tree_dirty=value),
                     "working_tree_dirty")

    # --- code_sha256 ------------------------------------------------------
    @pytest.mark.parametrize("value", [None, {}])
    def test_an_absent_code_digest_map_is_rejected(self, diag, tmp_path, value):
        self.rejects(diag, tmp_path, lambda p: p.update(code_sha256=value),
                     "code_sha256")

    def test_a_code_file_left_out_is_rejected(self, diag, tmp_path):
        dropped = diag.CODE_FILES[-1]
        self.rejects(diag, tmp_path,
                     lambda p: p["code_sha256"].pop(dropped),
                     "does not cover")

    @pytest.mark.parametrize("value", [None, "", "nothex", "a" * 63])
    def test_a_placeholder_code_digest_is_rejected(self, diag, tmp_path, value):
        target = diag.CODE_FILES[0]
        self.rejects(diag, tmp_path,
                     lambda p: p["code_sha256"].update({target: value}),
                     "no valid digest")

    # --- lora_config ------------------------------------------------------
    @pytest.mark.parametrize("value", [None, {}])
    def test_an_absent_lora_config_is_rejected(self, diag, tmp_path, value):
        self.rejects(diag, tmp_path, lambda p: p.update(lora_config=value),
                     "lora_config")

    def test_every_lora_field_is_required_not_just_the_cross_checked_two(
            self, diag, tmp_path):
        from src.training.lora import LoraConfig_

        for field in LoraConfig_().as_dict():
            self.rejects(diag, tmp_path,
                         lambda p, f=field: p["lora_config"].pop(f), field)

    def test_a_lora_field_present_but_null_is_rejected(self, diag, tmp_path):
        self.rejects(diag, tmp_path,
                     lambda p: p["lora_config"].update(rank=None),
                     "no value for rank")

    # --- packages ---------------------------------------------------------
    @pytest.mark.parametrize("value", [None, {}])
    def test_absent_packages_are_rejected(self, diag, tmp_path, value):
        self.rejects(diag, tmp_path, lambda p: p.update(packages=value),
                     "packages")

    @pytest.mark.parametrize("pkg",
                             ["python", "torch", "transformers", "peft"])
    def test_each_package_version_is_required(self, diag, tmp_path, pkg):
        self.rejects(diag, tmp_path, lambda p: p["packages"].pop(pkg),
                     f"no {pkg} version")

    @pytest.mark.parametrize("pkg",
                             ["python", "torch", "transformers", "peft"])
    def test_a_blank_package_version_is_rejected(self, diag, tmp_path, pkg):
        self.rejects(diag, tmp_path,
                     lambda p: p["packages"].update({pkg: ""}),
                     f"no {pkg} version")

    # --- device / dtype ---------------------------------------------------
    @pytest.mark.parametrize("key", ["device", "dtype"])
    @pytest.mark.parametrize("value", [None, ""])
    def test_a_blank_device_or_dtype_is_rejected(self, diag, tmp_path, key,
                                                 value):
        self.rejects(diag, tmp_path, lambda p: p.update({key: value}), key)

    # --- phases / condition_order ----------------------------------------
    @pytest.mark.parametrize("key", ["phases", "condition_order"])
    @pytest.mark.parametrize("value", [None, []])
    def test_an_empty_phase_or_order_list_is_rejected(self, diag, tmp_path,
                                                      key, value):
        self.rejects(diag, tmp_path, lambda p: p.update({key: value}), key)

    # --- stop_conditions --------------------------------------------------
    @pytest.mark.parametrize("value", [None, {}])
    def test_absent_stop_conditions_are_rejected(self, diag, tmp_path, value):
        self.rejects(diag, tmp_path, lambda p: p.update(stop_conditions=value),
                     "stop_conditions")

    @pytest.mark.parametrize(
        "field", ["slow_row_seconds", "slow_row_streak", "max_seconds"])
    def test_each_stop_field_is_required(self, diag, tmp_path, field):
        self.rejects(diag, tmp_path,
                     lambda p: p["stop_conditions"].pop(field),
                     f"no {field}")

    @pytest.mark.parametrize(
        "field", ["slow_row_seconds", "slow_row_streak", "max_seconds"])
    def test_a_null_stop_field_is_rejected(self, diag, tmp_path, field):
        self.rejects(diag, tmp_path,
                     lambda p: p["stop_conditions"].update({field: None}),
                     f"no {field}")

    # --- per-condition digests -------------------------------------------
    @pytest.mark.parametrize("value", [None, {}])
    def test_absent_per_condition_digests_are_rejected(self, diag, tmp_path,
                                                       value):
        self.rejects(diag, tmp_path,
                     lambda p: p.update(condition_input_order_digests=value),
                     "condition_input_order_digests")

    @pytest.mark.parametrize("value", [None, "", "nothex", "a" * 63])
    def test_an_invalid_per_condition_digest_is_rejected(self, diag, tmp_path,
                                                         value):
        self.rejects(
            diag, tmp_path,
            lambda p: p["condition_input_order_digests"].update(
                {"empty_cache": value}),
            "not a valid digest|no pre-run input-order digest")

    def test_schema_1_is_not_held_to_any_of_this(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path, schema=1)
        for key in SCHEMA1_ABSENT:
            del stored["provenance"][key]
        diag.check_replayable(stored)

    # --- device / dtype consistency --------------------------------------
    def test_a_cpu_record_cannot_be_schema_2(self, diag, tmp_path):
        """On CPU the clear is a no-op, so the arms would be the same arm."""
        stored = stored_run(diag, tmp_path, schema=2)
        stored["env"]["device"] = stored["provenance"]["device"] = "cpu"
        with pytest.raises(SystemExit, match="only means anything on 'mps'"):
            diag.check_replayable(stored)

    def test_a_dtype_that_disagrees_with_the_lora_config_is_rejected(
            self, diag, tmp_path):
        stored = stored_run(diag, tmp_path, schema=2)
        stored["provenance"]["lora_config"]["dtype"] = "float16"
        with pytest.raises(SystemExit, match="disagree on dtype"):
            diag.check_replayable(stored)

    # --- basic types ------------------------------------------------------
    @pytest.mark.parametrize("key", ["phases", "condition_order"])
    @pytest.mark.parametrize("value", [[""], [None], [1, 2], "notalist"])
    def test_a_malformed_name_list_is_rejected(self, diag, tmp_path, key,
                                               value):
        self.rejects(diag, tmp_path, lambda p: p.update({key: value}), key)

    @pytest.mark.parametrize(
        "field", ["slow_row_seconds", "slow_row_streak", "max_seconds"])
    @pytest.mark.parametrize("value", ["30", True, [30], 0, -1])
    def test_a_non_numeric_or_impossible_stop_bound_is_rejected(
            self, diag, tmp_path, field, value):
        self.rejects(diag, tmp_path,
                     lambda p: p["stop_conditions"].update({field: value}),
                     f"stop_conditions.{field}")


class TestSchema2ContractIsFrozen:
    """A finished record is judged by the contract it was written under.

    Deriving the required fields from `CODE_FILES` or `LoraConfig_` would mean
    that adding one source file, or one hyperparameter, retroactively
    invalidates every correct record ever produced. That is the same mistake
    as re-rendering an old run with today's constants, one layer down.
    """

    def test_the_contract_does_not_read_todays_definitions(self):
        import ast

        tree = ast.parse(SCRIPT.read_text())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_check_provenance_v2")
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        for live in ("CODE_FILES", "LoraConfig_"):
            assert live not in names, (
                f"_check_provenance_v2 reads {live}; a record that was "
                "complete when written would fail once that grew")

    def test_every_contract_field_is_still_required(self, diag, tmp_path):
        """Freezing the list must not mean the list stopped being enforced."""
        for f in diag.SCHEMA2_CODE_FILES:
            stored = stored_run(diag, tmp_path, schema=2)
            stored["provenance"]["code_sha256"].pop(f)
            with pytest.raises(SystemExit, match="does not cover"):
                diag.check_replayable(stored)
        for f in diag.SCHEMA2_LORA_FIELDS:
            stored = stored_run(diag, tmp_path, schema=2)
            stored["provenance"]["lora_config"].pop(f)
            with pytest.raises(SystemExit, match=f"lora_config is missing {f}"):
                diag.check_replayable(stored)

    def test_a_new_source_file_does_not_invalidate_an_old_record(
            self, diag, tmp_path, monkeypatch):
        """Simulates CODE_FILES growing after the record was written."""
        stored = stored_run(diag, tmp_path, schema=2)
        monkeypatch.setattr(
            diag, "CODE_FILES", diag.CODE_FILES + ("src/added/later.py",))
        diag.check_replayable(stored)

    def test_a_new_hyperparameter_does_not_invalidate_an_old_record(
            self, diag, tmp_path, monkeypatch):
        """Simulates LoraConfig_ growing after the record was written."""
        stored = stored_run(diag, tmp_path, schema=2)

        class Grown:
            @staticmethod
            def as_dict():
                return {f: 1 for f in diag.SCHEMA2_LORA_FIELDS} | {
                    "new_knob": 0.5}

        monkeypatch.setattr(diag, "LoraConfig_", lambda: Grown())
        diag.check_replayable(stored)

    def test_todays_config_can_still_produce_a_schema_2_record(self, diag):
        """If a field were *removed*, schema 2 could no longer be written."""
        from src.training.lora import LoraConfig_

        assert set(diag.SCHEMA2_LORA_FIELDS) <= set(LoraConfig_().as_dict())
        assert set(diag.SCHEMA2_CODE_FILES) <= set(diag.CODE_FILES)


class TestSchemaVersionDispatch:
    """Validation is chosen by exact version, never by 'at least'."""

    def test_each_supported_version_has_a_validator(self, diag):
        assert set(diag.CONTRACTS) == set(diag.SUPPORTED_SCHEMA_VERSIONS)
        for version, contract in diag.CONTRACTS.items():
            assert contract.version == version
            assert callable(contract.check_provenance)
            assert callable(contract.check_conditions)

    @pytest.mark.parametrize("version", [3, 99, 0, -1, "2", 2.0, None])
    def test_an_unsupported_version_fails_closed(self, diag, tmp_path,
                                                 version):
        stored = stored_run(diag, tmp_path, schema=2)
        stored["env"]["schema_version"] = version
        with pytest.raises(SystemExit, match="schema_version"):
            diag.check_replayable(stored)

    def test_the_refusal_says_what_it_does_understand(self, diag, tmp_path):
        stored = stored_run(diag, tmp_path, schema=3)
        with pytest.raises(SystemExit) as e:
            diag.check_replayable(stored)
        assert "[1, 2]" in str(e.value)
        assert "mean something different" in str(e.value)

    def test_a_future_schema_is_not_validated_as_schema_2(self, diag,
                                                          tmp_path):
        """Even a record that would otherwise pass must be refused."""
        stored = stored_run(diag, tmp_path, schema=2)
        stored["env"]["schema_version"] = 3
        with pytest.raises(SystemExit, match="validates \\[1, 2\\]"):
            diag.check_replayable(stored)

    def test_both_known_versions_still_pass(self, diag, tmp_path):
        diag.check_replayable(stored_run(diag, tmp_path, schema=2))
        stored = stored_run(diag, tmp_path, schema=1)
        for key in SCHEMA1_ABSENT:
            del stored["provenance"][key]
        diag.check_replayable(stored)

    @needs_report_14
    def test_the_stored_report_declares_a_supported_version(self, diag):
        import json

        stored = json.loads((ROOT / "data" / "reports"
                             / "14_mps_speed.json").read_text())
        assert (stored["env"]["schema_version"]
                in diag.SUPPORTED_SCHEMA_VERSIONS)

    def test_no_validator_infers_requirements_from_the_version_number(self):
        """`schema >= n` is a guess about what a later schema will mean."""
        src = SCRIPT.read_text()
        for guess in ("schema >= 2", "schema_version\") >= ",
                      "schema_version', 1) >= "):
            assert guess not in src, guess


class TestContractsAreIndependent:
    """A new schema must not reach backwards into the finished ones.

    The contracts are deliberately not factored into a shared base. Every
    tightening -- a new setting, a new provenance field, a new per-condition
    field -- lands on the schema being added, and records written under the
    older contracts keep meaning what they meant.
    """

    def grown(self, diag, **overrides):
        """A schema 3 that demands strictly more than schema 2."""
        import dataclasses

        two = diag.CONTRACTS[2]
        fields = dict(
            version=3,
            required_env=two.required_env + ("some_new_setting",),
            required_provenance=two.required_provenance + ("some_new_field",),
            required_condition_fields=(two.required_condition_fields
                                       + ("some_new_condition_field",)),
        )
        return dataclasses.replace(two, **(fields | overrides))

    def register(self, diag, monkeypatch, contract):
        monkeypatch.setitem(diag.CONTRACTS, contract.version, contract)
        monkeypatch.setattr(diag, "SUPPORTED_SCHEMA_VERSIONS",
                            tuple(sorted(diag.CONTRACTS)))

    def test_a_stricter_new_schema_leaves_schema_2_alone(
            self, diag, tmp_path, monkeypatch):
        self.register(diag, monkeypatch, self.grown(diag))
        diag.check_replayable(stored_run(diag, tmp_path, schema=2))

    def test_a_stricter_new_schema_leaves_schema_1_alone(
            self, diag, tmp_path, monkeypatch):
        self.register(diag, monkeypatch, self.grown(diag))
        stored = stored_run(diag, tmp_path, schema=1)
        for key in SCHEMA1_ABSENT:
            del stored["provenance"][key]
        diag.check_replayable(stored)

    @needs_report_14
    @needs_instruction_pool
    def test_the_stored_report_survives_a_stricter_new_schema(self, monkeypatch):
        """The one real record must not be invalidated by a future schema."""
        import json

        # A fresh module, not the `diag` fixture: this one has to read the
        # real input file the stored run recorded a digest for.
        mod = diag_module()
        self.register(mod, monkeypatch, self.grown(mod))
        stored = json.loads((ROOT / "data" / "reports"
                             / "14_mps_speed.json").read_text())
        mod.check_replayable(stored)

    def test_a_new_schema_does_not_inherit_schema_2_condition_fields(
            self, diag, tmp_path, monkeypatch):
        """Schema 3 declares its own per-condition contract, not schema 2's."""
        import dataclasses

        three = dataclasses.replace(
            diag.CONTRACTS[2], version=3,
            required_condition_fields=("input_order_digest",))
        self.register(diag, monkeypatch, three)

        stored = stored_run(diag, tmp_path, schema=2)
        stored["env"]["schema_version"] = 3
        # Everything schema 2 demanded per condition, except the one field
        # schema 3 kept.
        for c in stored["conditions"]:
            for key in diag.CONTRACTS[2].required_condition_fields:
                if key != "input_order_digest":
                    c[key] = None
        diag.check_replayable(stored)

    def test_a_new_schema_still_enforces_its_own_condition_fields(
            self, diag, tmp_path, monkeypatch):
        import dataclasses

        three = dataclasses.replace(
            diag.CONTRACTS[2], version=3,
            required_condition_fields=("completed_input_digest",))
        self.register(diag, monkeypatch, three)

        stored = stored_run(diag, tmp_path, schema=2)
        stored["env"]["schema_version"] = 3
        stored["conditions"][0]["completed_input_digest"] = None
        with pytest.raises(SystemExit, match="missing completed_input_digest"):
            diag.check_replayable(stored)

    def test_schema_1_demands_nothing_per_condition(self, diag):
        """It recorded none of it; demanding it would recover none of it."""
        assert diag.CONTRACTS[1].required_condition_fields == ()

    def test_each_schema_holds_its_own_field_lists(self, diag):
        """Shared objects would let one schema's growth reach the others."""
        one, two = diag.CONTRACTS[1], diag.CONTRACTS[2]
        assert one.required_provenance is not two.required_provenance
        assert (one.required_condition_fields
                is not two.required_condition_fields)
        assert set(one.required_provenance) < set(two.required_provenance)

    def test_a_contract_cannot_be_edited_in_place(self, diag):
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            diag.CONTRACTS[2].required_env = ()


class TestDeviceIsRequiredUpFront:
    """No model is loaded here; this is the check that runs before one is.

    On CPU `empty_cache()` is an explicit no-op, so the treatment arm would
    schedule twenty clears, perform none, come out identical to the control,
    and be rejected by the replay gate -- all after the hours were spent.
    """

    def torch_with_mps(self, available):
        mps = type("M", (), {"is_available": staticmethod(lambda: available)})
        return type("T", (), {"backends": type("B", (), {"mps": mps})})

    def test_mps_available_resolves_to_mps(self, diag):
        assert diag.resolve_device(self.torch_with_mps(True)) == "mps"

    def test_no_mps_stops_before_anything_runs(self, diag):
        with pytest.raises(SystemExit, match="MPS is not available"):
            diag.resolve_device(self.torch_with_mps(False))

    def test_the_refusal_says_why_cpu_would_be_wrong(self, diag):
        with pytest.raises(SystemExit) as e:
            diag.resolve_device(self.torch_with_mps(False))
        assert "no-op" in str(e.value) and "clear count" in str(e.value)

    def test_a_backend_that_raises_is_treated_as_unavailable(self, diag):
        def boom():
            raise RuntimeError("no backend")

        mps = type("M", (), {"is_available": staticmethod(boom)})
        torch_mod = type("T", (), {"backends": type("B", (), {"mps": mps})})
        with pytest.raises(SystemExit, match="MPS is not available"):
            diag.resolve_device(torch_mod)

    def test_there_is_no_silent_cpu_fallback_left(self):
        assert 'else "cpu"' not in SCRIPT.read_text()

    def test_the_check_runs_before_the_tokenizer_or_the_data(self):
        """Fail early means before the expensive part, not after it."""
        src = SCRIPT.read_text()
        assert src.index("resolve_device(torch)") < src.index("load_tokenizer()")
        assert src.index("resolve_device(torch)") < src.index("sample_pairs(")

    def test_replay_still_works_without_a_device(self):
        """--from-json re-renders a stored run; it runs no model at all."""
        src = SCRIPT.read_text()
        assert src.index('"--from-json" in argv') < src.index(
            "resolve_device(torch)")


class TestRunConditionAccounting:
    """The loop itself, with the model and device faked out.

    The helpers can each be right while the loop still wires them up wrongly
    -- which is exactly what happened: `DeviceOps` counted correctly and
    `run_condition` then read the counter after its own teardown call. So this
    drives the real loop and checks what it hands back.
    """

    def result(self, diag, monkeypatch, name, *, rows=4, clear_every=2,
               window=2, mem_every=2):
        import torch

        from src.training.diagnostics import DeviceOps

        class Fake(DeviceOps):
            def __init__(self, device, **kw):
                super().__init__("mps", torch_mod=fake_torch(), **kw)

        class Out:
            def __init__(self, loss):
                self.loss = loss

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.tensor([0.5]))

            def forward(self, **batch):
                # /3 so the loss has digits past the fourth decimal place,
                # which is the thing the storage format has to preserve.
                return Out((self.w * batch["input_ids"].float().mean()
                            / 3.0).sum())

        monkeypatch.setattr(diag, "MAX_ROWS", rows)
        monkeypatch.setattr(diag, "WINDOW", window)
        monkeypatch.setattr(diag, "MEMORY_EVERY", mem_every)
        monkeypatch.setattr(diag, "EMPTY_CACHE_EVERY", clear_every)
        monkeypatch.setattr(diag, "DeviceOps", Fake)
        monkeypatch.setattr(diag, "build_model",
                            lambda cfg, device: (Tiny(),
                                                 {"trainable_parameters": 1}))
        monkeypatch.setattr(diag, "assert_only_lora_trainable", lambda m: None)
        def slow_probe():
            # A real probe shells out three times; this stands in for that so
            # the test can tell a timed probe from an untimed one.
            time.sleep(0.005)
            return {"free_plus_inactive_gb": 1.0, "swap_used_gb": 0.0,
                    "memory_pressure_percent_free": 50}

        monkeypatch.setattr(diag, "system_memory", slow_probe)

        # run_condition() hands memory_sample() the *real* torch module, so
        # without this a CPU-only unit test calls torch.mps.* on a device it
        # never set up. Faking DeviceOps is not enough -- that only covers
        # sync and empty_cache. The counter proves the stub was used and the
        # real MPS API was therefore not touched.
        self.probe_calls = []

        def fake_memory_sample(torch_mod, *, rss_bytes=None):
            self.probe_calls.append(rss_bytes)
            return {"mps_current_allocated_gb": 2.0,
                    "mps_driver_allocated_gb": 5.0,
                    "mps_recommended_max_gb": 37.44,
                    "peak_process_rss_gb": 1.0}

        monkeypatch.setattr(diag, "memory_sample", fake_memory_sample)
        monkeypatch.setattr(diag, "collate", lambda batch, eos: {
            "input_ids": torch.ones(1, 8, dtype=torch.long),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
            "labels": torch.full((1, 8), 3, dtype=torch.long)})

        from src.training.lora import LoraConfig_

        meta = [type("R", (), {"sample_id": f"s{i}"}) for i in range(rows)]
        tok = type("T", (), {"eos_token_id": 0})
        return diag.run_condition(name, LoraConfig_(), list(range(rows)), meta,
                                  tok, device="cpu", order=0)

    def test_the_loop_never_reaches_the_real_mps_api(self, diag, monkeypatch):
        """A CPU unit test must not call into a device it did not set up.

        `run_condition()` passes the real `torch` to `memory_sample()`, which
        reaches `torch.mps.current_allocated_memory()` and friends. Stubbing
        `DeviceOps` covers only sync and empty_cache, so the probe was still
        going through to the driver.
        """
        res = self.result(diag, monkeypatch, "continuous")
        assert self.probe_calls, "memory_sample() was not the stubbed one"
        assert len(self.probe_calls) == len(res["memory"])
        assert all(m["mps_driver_allocated_gb"] == 5.0 for m in res["memory"])

    def test_two_hundred_rows_every_ten_gives_nought_and_twenty(
            self, diag, monkeypatch):
        """Report 14's own figures, driven through the real loop.

        The single counter would have made this 1 and 21. Both arms also run
        exactly one teardown clear, and it belongs to neither of those totals.
        """
        kw = dict(rows=200, clear_every=10, window=20, mem_every=50)
        cont = self.result(diag, monkeypatch, "continuous", **kw)
        ec = self.result(diag, monkeypatch, "empty_cache", **kw)

        assert cont["rows_completed"] == 200 and ec["rows_completed"] == 200
        assert cont["scheduled_empty_cache_calls"] == 0
        assert ec["scheduled_empty_cache_calls"] == 20
        assert cont["teardown_empty_cache_calls"] == 1
        assert ec["teardown_empty_cache_calls"] == 1
        # The schedule is recorded next to the count, so the two can be
        # checked against each other rather than taken on trust.
        assert cont["scheduled_empty_cache_every"] is None
        assert ec["scheduled_empty_cache_every"] == 10

    def test_the_control_arm_schedules_no_clears(self, diag, monkeypatch):
        res = self.result(diag, monkeypatch, "continuous")
        assert res["scheduled_empty_cache_calls"] == 0
        assert res["scheduled_empty_cache_every"] is None

    def test_the_teardown_clear_is_counted_but_kept_out_of_the_schedule(
            self, diag, monkeypatch):
        """The reported bug: this used to come back as 1 scheduled call."""
        res = self.result(diag, monkeypatch, "continuous")
        assert res["scheduled_empty_cache_calls"] == 0
        assert res["teardown_empty_cache_calls"] == 1

    def test_the_treatment_arm_reports_its_schedule_only(self, diag,
                                                         monkeypatch):
        res = self.result(diag, monkeypatch, "empty_cache", rows=4,
                          clear_every=2)
        assert res["scheduled_empty_cache_calls"] == 2      # not 3
        assert res["teardown_empty_cache_calls"] == 1
        assert res["scheduled_empty_cache_every"] == 2

    def test_the_overhead_is_split_rather_than_left_in_one_lump(
            self, diag, monkeypatch):
        res = self.result(diag, monkeypatch, "empty_cache")
        split = res["between_row_overhead_breakdown"]
        assert set(split) == {"scheduled_empty_cache_seconds",
                              "memory_probe_seconds", "unattributed_seconds"}
        assert res["between_row_overhead_seconds"] == pytest.approx(
            sum(split.values()), abs=0.01)

    def test_the_probes_are_timed(self, diag, monkeypatch):
        res = self.result(diag, monkeypatch, "continuous")
        assert all("probe_seconds" in m for m in res["memory"])
        assert res["between_row_overhead_breakdown"][
            "memory_probe_seconds"] > 0

    def test_each_row_carries_its_own_end_to_end(self, diag, monkeypatch):
        res = self.result(diag, monkeypatch, "continuous")
        for r in res["per_row"]:
            assert r["end_to_end"] >= r["total"]
        assert res["windows"][0]["end_to_end_seconds"] is not None

    def test_the_stored_loss_is_not_rounded(self, diag, monkeypatch):
        """Four decimal places is what made the first run's claim so weak."""
        res = self.result(diag, monkeypatch, "continuous")
        losses = [r["loss"] for r in res["per_row"]]
        assert any(round(v, 4) != v for v in losses), losses

    def test_every_row_records_which_sample_it_was(self, diag, monkeypatch):
        res = self.result(diag, monkeypatch, "continuous")
        ids = [r["sample_id"] for r in res["per_row"]]
        assert sorted(ids) == ["s0", "s1", "s2", "s3"]

    def test_the_input_digest_covers_the_rows_that_ran(self, diag,
                                                       monkeypatch):
        res = self.result(diag, monkeypatch, "continuous")
        assert res["completed_input_digest"] == diag.digest_ids(
            r["sample_id"] for r in res["per_row"])
        assert res["input_order_digest"] == res["completed_input_digest"]

    def test_the_result_passes_its_own_consistency_check(self, diag,
                                                         monkeypatch,
                                                         tmp_path):
        res = self.result(diag, monkeypatch, "empty_cache")
        env = {"condition_order": ["empty_cache"], "max_rows_per_condition": 4,
               "empty_cache_every": 2, "schema_version": 2}
        contract = diag.CONTRACTS[2]
        assert contract.check_conditions(
            {"conditions": [res]}, env, {}, contract) == []

    def test_a_whole_run_passes_the_replay_gate(self, diag, monkeypatch,
                                                tmp_path):
        """Both conditions, the real loop, through the real gate."""
        results = [self.result(diag, monkeypatch, "continuous"),
                   self.result(diag, monkeypatch, "empty_cache")]
        results[1]["run_order"] = 1
        data = tmp_path / "instruct_inv_train.jsonl"
        data.write_text("row\n")
        digest = results[0]["input_order_digest"]
        order = ["continuous", "empty_cache"]
        packages = {"python": "3.13.9", "torch": "2.13.0",
                    "transformers": "5.15.0", "peft": "0.20.0"}
        stop = {"slow_row_seconds": 30.0, "slow_row_streak": 3,
                "max_seconds": 2700.0}
        diag.check_replayable({
            "env": {"schema_version": 2, "max_rows_per_condition": 4,
                    "window": 2, "memory_sample_every": 2,
                    "empty_cache_every": 2, "seed": 0, "grad_accum": 8,
                    "phases": list(diag.PHASES),
                    "condition_order": list(order),
                    "condition_definitions": {c: c for c in order},
                    "stop_slow_row_seconds": 30.0, "stop_slow_row_streak": 3,
                    "stop_max_seconds": 2700.0, "loss_decimals_stored": None,
                    # The loop above is driven on CPU for speed; the record
                    # under test stands for a real run, and schema 2 can only
                    # be an MPS record.
                    "device": "mps", "dtype": "bfloat16", **packages},
            "provenance": {
                "instruction_sha256": {"instruct_inv_train.jsonl":
                                       diag.sha256_file(data)},
                "selection_digest": digest, "training_order_digest": digest,
                "base_revision": "abc", "published_adapter_revision": "def",
                "head": "0" * 40, "working_tree_dirty": False,
                "code_sha256": {f: "a" * 64 for f in diag.CODE_FILES},
                "lora_config": lora_config_fixture(),
                "packages": dict(packages),
                "device": "mps", "dtype": "bfloat16",
                "phases": list(diag.PHASES),
                "stop_conditions": dict(stop),
                "condition_order": list(order),
                "condition_input_order_digests": {c: digest for c in order}},
            "conditions": results})

    def test_a_fresh_run_gates_itself_before_it_writes(self):
        """Catch a bookkeeping bug now, not on the first re-render."""
        src = SCRIPT.read_text()
        assert src.index("check_replayable(payload)") < src.index(
            '(REPORT_DIR / "14_mps_speed.json").write_text')


@needs_report_14
class TestStoredReport14:
    """The committed run must still pass its own gate and re-render."""

    def stored(self):
        import json

        return json.loads((ROOT / "data" / "reports"
                           / "14_mps_speed.json").read_text())

    def test_it_is_labelled_schema_1_with_its_gaps_named(self):
        env = self.stored()["env"]
        assert env["schema_version"] == 1
        assert env["loss_decimals_stored"] == 4
        assert "not record" in env["schema_1_note"]

    def test_its_clear_counts_are_the_schedule_and_nothing_else(self):
        by = {c["condition"]: c for c in self.stored()["conditions"]}
        assert by["continuous"]["scheduled_empty_cache_calls"] == 0
        assert by["empty_cache"]["scheduled_empty_cache_calls"] == 20
        # Never counted at the time, so it is unknown rather than assumed.
        for c in by.values():
            assert c["teardown_empty_cache_calls"] is None

    def test_the_unrecoverable_breakdown_is_left_absent(self):
        for c in self.stored()["conditions"]:
            assert c["between_row_overhead_breakdown"] is None
            assert c["input_order_digest"] is None

    def test_the_backfills_are_labelled(self):
        back = self.stored()["provenance"]["backfilled_after_the_run"]
        for key in ("scheduled_empty_cache_calls", "phases",
                    "stop_slow_row_streak"):
            assert key in back

    @needs_instruction_pool
    def test_it_still_passes_the_replay_gate(self):
        diag_module().check_replayable(self.stored())
