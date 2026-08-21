"""Preflight gating for the multi-process MPS ordering experiment.

Report 14 measured its two conditions in one process, in a fixed order, and
could not separate "the cache was cleared" from "this arm ran second". The
follow-up runs each condition in its own process, which removes the in-process
carry-over -- but only if each process actually starts from a comparable
machine state. That is what this module decides, and it decides it *before*
any model is loaded, on numbers read from the OS rather than inferred.

Three things it is careful about.

**Thresholds are calibrated, not invented.** There was no prior evidence for
what an idle machine's swap or memory pressure looks like minute to minute, so
the bands come from a calibration pass that loads no model and runs no
training: median and MAD over repeated idle samples. A number chosen after
seeing a treatment result is not a threshold, it is a rationalisation.

**A missing reading is a failure, not a pass.** ``None`` means the probe could
not be read, and a gate that cannot be evaluated has not been satisfied.

**``free_plus_inactive_gb`` is not free memory.** It is what ``vm_stat``
allows adding up; inactive pages are reclaimable rather than unused, and
calling it "free" or "available" would overstate what was measured.
"""

from __future__ import annotations

import os
import time

from src.training.diagnostics import system_memory

#: One entry per gated metric.
#:
#: ``direction`` says which side is bad: ``upper`` means the sample must be at
#: or below the threshold, ``lower`` means at or above it. ``min_slack`` is the
#: floor on the calibrated band, so a machine that happens to be very steady
#: during calibration does not produce a gate no real run can pass. ``cap`` is
#: the absolute limit that holds regardless of what calibration saw -- a
#: machine calibrated while already under pressure must not be able to lower
#: its own bar indefinitely.
GATE_SPEC: dict[str, dict] = {
    "swap_used_gb": {"direction": "upper", "min_slack": 0.25, "cap": None},
    "memory_pressure_percent_free": {"direction": "lower", "min_slack": 5.0,
                                     "cap": 20.0},
    "free_plus_inactive_gb": {"direction": "lower", "min_slack": 0.5,
                              "cap": 2.0},
    "normalized_load_1m": {"direction": "upper", "min_slack": 0.10,
                           "cap": 0.50},
}

GATE_METRICS = tuple(GATE_SPEC)

#: 1.4826 * MAD is the MAD-based estimator of the standard deviation for
#: normally distributed data. Used rather than the sample SD because a single
#: noisy sample should not widen the band it is then judged against.
MAD_TO_SCALE = 1.4826

CALIBRATION_SAMPLES = 10
CALIBRATION_INTERVAL_SECONDS = 30

RECOVERY_POLL_SECONDS = 30
RECOVERY_MAX_WAIT_SECONDS = 15 * 60
RECOVERY_CONSECUTIVE_PASSES = 3


def normalized_load_1m() -> float | None:
    """One-minute load average per core, so the gate is machine-independent."""
    try:
        cores = os.cpu_count() or 1
        return round(os.getloadavg()[0] / cores, 4)
    except Exception:
        return None


def preflight_sample(clock=time.time) -> dict:
    """One reading of every gated metric, and nothing else.

    Deliberately narrow: no process names, no paths, no command lines. The
    gate needs to know whether the machine is busy, not what it is busy with,
    and a record that names other work is a record that leaks it.
    """
    mem = system_memory()
    return {
        "sampled_at": round(clock(), 3),
        "swap_used_gb": mem.get("swap_used_gb"),
        "memory_pressure_percent_free": mem.get("memory_pressure_percent_free"),
        "free_plus_inactive_gb": mem.get("free_plus_inactive_gb"),
        "normalized_load_1m": normalized_load_1m(),
    }


def median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return (vals[mid - 1] + vals[mid]) / 2.0


def mad(values: list[float]) -> float | None:
    """Median absolute deviation about the median."""
    med = median(values)
    if med is None:
        return None
    return median([abs(v - med) for v in values if v is not None])


def calibrate(samples: list[dict]) -> dict:
    """Turn idle samples into the bands the experiment will be judged against.

    Records the formula alongside every number: a threshold whose derivation
    is not written down cannot be checked, and this one has to survive review
    by someone who was not here when it was computed.
    """
    stats: dict[str, dict] = {}
    for metric, spec in GATE_SPEC.items():
        values = [s.get(metric) for s in samples]
        present = [v for v in values if v is not None]
        med = median(values)
        m = mad(values)
        scale = None if m is None else round(MAD_TO_SCALE * m, 6)
        band = None
        threshold = None
        if med is not None and scale is not None:
            band = max(3.0 * scale, spec["min_slack"])
            if spec["direction"] == "upper":
                threshold = med + band
                if spec["cap"] is not None:
                    threshold = min(spec["cap"], threshold)
            else:
                threshold = med - band
                if spec["cap"] is not None:
                    threshold = max(spec["cap"], threshold)
            threshold = round(threshold, 6)
        stats[metric] = {
            "n": len(present),
            "n_missing": len(values) - len(present),
            "median": med,
            "mad": m,
            "scale": scale,
            "band": None if band is None else round(band, 6),
            "direction": spec["direction"],
            "min_slack": spec["min_slack"],
            "cap": spec["cap"],
            "threshold": threshold,
            "formula": _formula(spec),
        }
    return stats


def _formula(spec: dict) -> str:
    slack = f"max(3*scale, {spec['min_slack']})"
    if spec["direction"] == "upper":
        base = f"median + {slack}"
        return base if spec["cap"] is None else f"min({spec['cap']}, {base})"
    base = f"median - {slack}"
    return base if spec["cap"] is None else f"max({spec['cap']}, {base})"


def thresholds_from(stats: dict) -> dict:
    return {m: stats[m]["threshold"] for m in GATE_SPEC if m in stats}


def evaluate_gate(sample: dict, thresholds: dict) -> dict:
    """Judge one sample. Any unreadable or unthresholded metric fails."""
    checks = {}
    for metric, spec in GATE_SPEC.items():
        value = sample.get(metric)
        threshold = thresholds.get(metric)
        if value is None or threshold is None:
            passed = False
            why = ("metric was not readable" if value is None
                   else "no calibrated threshold for this metric")
        elif spec["direction"] == "upper":
            passed = value <= threshold
            why = f"{value} <= {threshold}" if passed else f"{value} > {threshold}"
        else:
            passed = value >= threshold
            why = f"{value} >= {threshold}" if passed else f"{value} < {threshold}"
        checks[metric] = {"value": value, "threshold": threshold,
                          "direction": spec["direction"], "passed": passed,
                          "detail": why}
    return {"passed": all(c["passed"] for c in checks.values()),
            "checks": checks,
            "failed": sorted(m for m, c in checks.items() if not c["passed"])}


def wait_for_recovery(thresholds: dict, *,
                      poll_seconds: int = RECOVERY_POLL_SECONDS,
                      max_wait_seconds: int = RECOVERY_MAX_WAIT_SECONDS,
                      needed_consecutive: int = RECOVERY_CONSECUTIVE_PASSES,
                      sampler=preflight_sample,
                      clock=time.monotonic,
                      sleep=time.sleep) -> dict:
    """Poll until the machine has been back inside the band N times running.

    Consecutive rather than cumulative: swap and memory pressure oscillate
    while the OS reclaims, and a single lucky reading between two bad ones is
    not a recovered machine. Timing out is a stop, never a warning -- the whole
    point of the redesign is that each process starts somewhere comparable, and
    a run that begins outside the band cannot be compared with one that did
    not.
    """
    started = clock()
    polls: list[dict] = []
    streak = 0
    while True:
        sample = sampler()
        verdict = evaluate_gate(sample, thresholds)
        streak = streak + 1 if verdict["passed"] else 0
        elapsed = clock() - started
        polls.append({"poll": len(polls) + 1,
                      "elapsed_seconds": round(elapsed, 2),
                      "sample": sample,
                      "passed": verdict["passed"],
                      "failed_metrics": verdict["failed"],
                      "consecutive_passes": streak})
        if streak >= needed_consecutive:
            return {"passed": True, "polls": polls,
                    "waited_seconds": round(clock() - started, 2),
                    "consecutive_passes_required": needed_consecutive,
                    "reason": None}
        if elapsed >= max_wait_seconds:
            return {"passed": False, "polls": polls,
                    "waited_seconds": round(elapsed, 2),
                    "consecutive_passes_required": needed_consecutive,
                    "reason": (f"machine did not return inside the calibrated "
                               f"band {needed_consecutive} times running "
                               f"within {max_wait_seconds}s")}
        sleep(poll_seconds)
