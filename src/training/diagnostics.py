"""Instrumentation for the short MPS speed diagnostic.

Pure bookkeeping: phase timing, memory sampling, stop conditions and the
aggregation the report prints. Nothing here loads a model, so the arithmetic
can be tested exactly against a fake instead of against a 1B-parameter run.

Two things this module is careful about.

**Timing under an async backend.** MPS dispatches work and returns; a
``perf_counter`` delta around an unsynchronised call measures the time to
*enqueue*, not to compute, and the cost then lands inside whatever is measured
next. Every phase boundary that matters is therefore synchronised explicitly,
which is why :class:`PhaseTimer` takes a ``sync`` callable rather than timing
naively.

**What memory numbers mean.** ``current_allocated_memory`` is what PyTorch
tracks; ``driver_allocated_memory`` is what the driver has taken, which
includes the allocator's own reserve and can grow while the tracked figure
stays flat. Both are recorded because a gap between them is exactly what a
fragmentation story would look like -- and neither, on its own, licenses a
claim about the cause.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class DeviceOps:
    """Device sync and cache-clear, failing loudly where they must work.

    A swallowed exception here is the worst possible outcome: on MPS a failed
    ``synchronize()`` means every phase boundary afterwards is measuring
    enqueue time rather than compute, and a failed ``empty_cache()`` means the
    condition that is supposed to clear the cache silently is not clearing it
    -- and the run still produces a full, confident-looking report.

    So on MPS both raise. On CPU both are explicit no-ops, because there is no
    device to synchronise and no MPS cache to clear; that is a real difference
    in what the operation *means*, not an error being hidden.

    **Scheduled and teardown clears are counted apart.** A clear that happens
    every N rows *is* the intervention under test; the one after the condition
    has finished, to free the model before the next condition builds its own,
    is housekeeping outside the measured region. One counter for both would
    report the control arm as having cleared the cache once and the treatment
    arm as having cleared it N+1 times, and would imply the teardown cost sits
    inside an end-to-end total it is not in.
    """

    def __init__(self, device: str, torch_mod=None, clock=time.perf_counter):
        import torch as _torch

        self.device = device
        self.torch = torch_mod if torch_mod is not None else _torch
        self.clock = clock
        self.is_mps = device == "mps"
        self.scheduled_empty_cache_calls = 0
        self.teardown_empty_cache_calls = 0
        #: One entry per scheduled call, so the intervention's own cost can be
        #: reported rather than left inside a lump of "between-row overhead".
        self.scheduled_empty_cache_seconds: list[float] = []

    def sync(self) -> None:
        if not self.is_mps:
            return
        try:
            self.torch.mps.synchronize()
        except Exception as e:
            raise RuntimeError(
                "torch.mps.synchronize() failed on an MPS run; every "
                "subsequent phase timing would measure enqueue rather than "
                "compute, so the run is stopped rather than reported"
            ) from e

    def empty_cache(self, *, teardown: bool = False) -> float:
        """Clear the MPS cache, returning the seconds the call itself took.

        Callers time this immediately after a row's closing ``sync()``, so no
        device work is outstanding and the figure is the host-side cost of the
        clear rather than a queue draining into it.
        """
        if not self.is_mps:
            return 0.0
        t0 = self.clock()
        try:
            self.torch.mps.empty_cache()
        except Exception as e:
            raise RuntimeError(
                "torch.mps.empty_cache() failed on an MPS run; the condition "
                "that depends on clearing the cache would silently not be "
                "clearing it"
            ) from e
        finally:
            elapsed = self.clock() - t0
            if teardown:
                self.teardown_empty_cache_calls += 1
            else:
                self.scheduled_empty_cache_calls += 1
                self.scheduled_empty_cache_seconds.append(elapsed)
        return elapsed

    def scheduled_clear_cost(self) -> dict:
        """Per-call and total cost of the scheduled clears, for the report."""
        s = self.scheduled_empty_cache_seconds
        return {
            "calls": self.scheduled_empty_cache_calls,
            "total_seconds": round(sum(s), 4),
            "mean_seconds": round(sum(s) / len(s), 5) if s else None,
            "max_seconds": round(max(s), 5) if s else None,
            "per_call_seconds": [round(v, 5) for v in s],
        }


class PhaseTimer:
    """Times named phases, synchronising the device at each boundary.

    ``sync`` is injected so tests can drive it without a GPU: the aggregation
    is the part that has to be right, and a real device would only make it
    slower to check.
    """

    def __init__(self, sync=None, clock=time.perf_counter):
        self.sync = sync or (lambda: None)
        self.clock = clock
        self.rows: list[dict[str, float]] = []
        self._current: dict[str, float] = {}
        self._t0: float | None = None

    def start(self) -> None:
        self.sync()
        self._current = {}
        self._t0 = self.clock()
        self._mark = self._t0

    def phase(self, name: str) -> None:
        """Close the phase that just ran and attribute its time to ``name``."""
        self.sync()
        now = self.clock()
        self._current[name] = now - self._mark
        self._mark = now

    def end(self, **extra) -> dict:
        self.sync()
        row = dict(self._current)
        row["total"] = self.clock() - self._t0
        row.update(extra)
        self.rows.append(row)
        return row

    @property
    def phases(self) -> list[str]:
        seen: list[str] = []
        for r in self.rows:
            for k in r:
                if k not in seen and isinstance(r[k], float):
                    seen.append(k)
        return seen


def summarise_phases(rows: list[dict], phases: tuple[str, ...]) -> dict:
    """Total, mean and share of measured time, per phase.

    The share is over the phases named here rather than over ``total``: they
    should account for nearly all of it, and a large remainder is itself worth
    seeing rather than silently absorbing.
    """
    out: dict[str, dict] = {}
    named_total = sum(sum(r.get(p, 0.0) for p in phases) for r in rows)
    for p in phases:
        vals = [r[p] for r in rows if p in r]
        if not vals:
            continue
        s = sorted(vals)
        out[p] = {
            "n": len(vals),
            "total_seconds": round(sum(vals), 3),
            "mean_seconds": round(sum(vals) / len(vals), 4),
            "median_seconds": round(s[len(s) // 2], 4),
            "max_seconds": round(s[-1], 4),
            "share_of_measured": (
                round(sum(vals) / named_total, 4) if named_total else 0.0),
        }
    total = sum(r.get("total", 0.0) for r in rows)
    out["_unattributed"] = {
        "total_seconds": round(total - named_total, 3),
        "share_of_total": round((total - named_total) / total, 4) if total else 0.0,
    }
    return out


def window_stats(rows: list[dict], size: int) -> list[dict]:
    """Fixed windows over the row sequence, so a trend can be seen at all.

    ``seconds`` is model compute -- the summed timed regions. ``end_to_end``
    is reported alongside it when every row in the window carries one, and is
    ``None`` otherwise: a window summed from only the rows that happen to have
    the field would read as a smaller total rather than as an incomplete one.
    """
    out = []
    for start in range(0, len(rows), size):
        chunk = rows[start:start + size]
        if not chunk:
            continue
        secs = sum(r["total"] for r in chunk)
        toks = sum(r.get("n_tokens", 0) for r in chunk)
        sup = sum(r.get("n_supervised", 0) for r in chunk)
        e2e = [r["end_to_end"] for r in chunk if r.get("end_to_end") is not None]
        complete = len(e2e) == len(chunk)
        out.append({
            "window": start // size,
            "rows": f"{start + 1}-{start + len(chunk)}",
            "n_rows": len(chunk),
            "seconds": round(secs, 3),
            "seconds_per_row": round(secs / len(chunk), 3),
            "end_to_end_seconds": round(sum(e2e), 3) if complete else None,
            "end_to_end_seconds_per_row": (
                round(sum(e2e) / len(chunk), 3) if complete else None),
            "tokens": toks,
            "supervised_tokens": sup,
            "tokens_per_second": round(toks / secs, 1) if secs else None,
            "mean_seq_len": round(toks / len(chunk), 1),
        })
    return out


@dataclass
class StopCondition:
    """Bounds the diagnostic so it ends by design rather than by patience.

    This round is about isolating a slowdown, not finishing a training run, so
    hitting a limit is a normal outcome: partial results are kept and the
    condition is reported alongside them.
    """

    max_seconds: float = 45 * 60
    slow_row_seconds: float = 30.0
    slow_row_streak: int = 3
    _streak: int = field(default=0, repr=False)

    def check(self, elapsed: float, row_seconds: float) -> str | None:
        """Returns the reason to stop, or None to continue."""
        if row_seconds > self.slow_row_seconds:
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.slow_row_streak:
            return (f"{self._streak} consecutive rows over "
                    f"{self.slow_row_seconds}s")
        if elapsed > self.max_seconds:
            return f"condition exceeded {self.max_seconds / 60:.0f} minutes"
        return None


def memory_sample(torch_mod, *, rss_bytes: int | None = None) -> dict:
    """One memory reading, naming exactly what each number covers.

    Field names say what was measured rather than something shorter:
    ``peak_process_rss_gb`` is ``ru_maxrss``, a high-water mark for the whole
    process and not a current reading, and calling it ``process_rss_gb`` would
    invite reading it as the latter.

    ``recommended_max`` is absent on some builds; a missing value is recorded
    as ``None`` rather than omitted, so a later reader can tell "not available"
    from "not sampled".
    """
    def try_call(fn):
        try:
            return fn()
        except Exception:
            return None

    mps = getattr(torch_mod, "mps", None)
    current = try_call(getattr(mps, "current_allocated_memory", lambda: None))
    driver = try_call(getattr(mps, "driver_allocated_memory", lambda: None))
    recommended = try_call(
        getattr(mps, "recommended_max_memory", lambda: None))
    gb = lambda v: round(v / (1024 ** 3), 3) if isinstance(v, (int, float)) else None
    return {
        "mps_current_allocated_gb": gb(current),
        "mps_driver_allocated_gb": gb(driver),
        "mps_recommended_max_gb": gb(recommended),
        # ru_maxrss: a high-water mark for the process, not a current value.
        "peak_process_rss_gb": gb(rss_bytes),
    }


def system_memory() -> dict:
    """Free memory, swap and macOS memory pressure, best effort.

    Read from the OS rather than inferred. Any field that cannot be read stays
    ``None``: an unavailable reading must not be mistaken for a healthy one.
    """
    import subprocess

    out: dict[str, object] = {
        # free + inactive pages, which is what vm_stat lets us add up; it is
        # not "free memory" in the sense a user would mean, and inactive pages
        # are reclaimable rather than unused.
        "free_plus_inactive_gb": None,
        "swap_used_gb": None,
        "memory_pressure_percent_free": None,
    }
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        if vm.returncode == 0:
            page = 4096
            stats = {}
            for line in vm.stdout.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    v = v.strip().rstrip(".")
                    if v.isdigit():
                        stats[k.strip()] = int(v)
            if "Pages free" in stats:
                free = (stats.get("Pages free", 0)
                        + stats.get("Pages inactive", 0)) * page
                out["free_plus_inactive_gb"] = round(free / (1024 ** 3), 3)
    except Exception:
        pass
    try:
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                            capture_output=True, text=True, timeout=5)
        if sw.returncode == 0 and "used =" in sw.stdout:
            used = sw.stdout.split("used =")[1].split()[0]
            val = float(used.rstrip("MG"))
            out["swap_used_gb"] = round(
                val / 1024 if used.endswith("M") else val, 3)
    except Exception:
        pass
    try:
        mp = subprocess.run(["memory_pressure"], capture_output=True,
                            text=True, timeout=5)
        for line in mp.stdout.splitlines():
            if "free percentage" in line.lower():
                out["memory_pressure_percent_free"] = int(
                    "".join(c for c in line.split(":")[-1] if c.isdigit()))
                break
    except Exception:
        pass
    return out
