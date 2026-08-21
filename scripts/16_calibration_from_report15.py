#!/usr/bin/env python3
"""Derive report 16's calibration from report 15's archived one.

Report 16's first ``--session-init`` was refused before it wrote anything.
The archived calibration is a ``preflight_calibration`` (schema 3) and carries
a ``gate`` block; report 16 wants a ``policy`` block, and its validator will
not take both spellings -- two definitions of a gate policy is exactly the
failure this tool has spent several rounds removing.

The three settings inside are the same three numbers. They are spelled
differently:

===============================  ================================
report 16 ``policy``             report 15 ``gate``
===============================  ================================
``consecutive_passes_required``  ``consecutive_passes_required``
``poll_interval_seconds``        ``poll_seconds``
``timeout_seconds``              ``max_wait_seconds``
===============================  ================================

So this is a schema adaptation and nothing else. **Nothing is remeasured.**
``samples``, ``stats`` and ``thresholds`` are carried over byte-for-byte, and
the machine, the platform, the sampling interval and the time it was taken
travel with them -- a threshold with no record of the machine it came from is
a number, not a measurement.

What is deliberately dropped from the top level: report 15's ``gate`` (there
must be exactly one object in this file shaped like an execution policy) and
its ``calibration_digest`` (it describes the source bytes; at the top level it
would read as this document's own digest and be wrong about it). Both are kept
inside ``source``, where they are facts about another document.

Read-only with respect to report 15: the source is checked against its
recorded digest and never written to. The output is published no-clobber, so
running this twice does not quietly produce a second version of the thresholds
that are in force.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.longrun import GATE_POLICY  # noqa: E402
from src.training.session import write_once_json  # noqa: E402

SOURCE = ROOT / "data" / "reports" / "15_mps_order" / "calibration.json"
#: The archived bytes this derivation is defined against. A different source
#: is a different measurement, and it does not get to inherit this provenance.
SOURCE_SHA256 = \
    "48439500c6162d6b4f4a38cb2b5a38846549386adffb6e3a8f239b902fc25660"
OUT = ROOT / "data" / "reports" / "16_longrun_calibration.json"

SCHEMA_VERSION = 1
KIND = "longrun_calibration"

#: report 16 policy field -> the report 15 gate field it is read from.
POLICY_FROM_GATE = {"consecutive_passes_required": "consecutive_passes_required",
                    "poll_interval_seconds": "poll_seconds",
                    "timeout_seconds": "max_wait_seconds"}

#: The same mapping as it is written into the file. The keys are dotted paths
#: rather than bare field names on purpose: exactly one object in this
#: document may have the shape of an execution policy, and a documentation
#: map keyed by `poll_interval_seconds` looks like a second one at a glance.
POLICY_FIELD_MAP = {f"policy.{new}": f"gate.{old}"
                    for new, old in POLICY_FROM_GATE.items()}

#: The measurement itself. Copied, never recomputed.
CARRIED_VERBATIM = ("samples", "stats", "thresholds")

#: The context needed to read those numbers: when, on what, how.
MEASUREMENT_CONTEXT = ("created_at", "loads_model", "samples_requested",
                       "interval_seconds", "metrics", "scale_formula",
                       "platform", "machine", "note")

#: Not carried to the top level, and why.
DROPPED = {
    "gate": "report 15's policy spelling; re-expressed as `policy` and "
            "recorded field by field in derivation.policy_field_map",
    "calibration_digest": "describes the source bytes, so it belongs in "
                          "`source`, not at the top level of another document",
    "schema_version": "this document declares its own",
    "kind": "this document declares its own",
}


def derive(source_path=SOURCE) -> dict:
    """Build the report 16 calibration. Writes nothing."""
    path = Path(source_path)
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(
            f"{path} hashes to {digest}, not the archived "
            f"{SOURCE_SHA256}. This derivation is defined against those bytes "
            "and against no others.")
    src = json.loads(raw.decode("utf-8"))
    gate = src["gate"]
    policy = {new: gate[old] for new, old in POLICY_FROM_GATE.items()}
    if policy != GATE_POLICY:
        raise SystemExit(
            f"the mapped policy {policy} is not report 16's fixed "
            f"{GATE_POLICY}. The gate report 15 ran is not the gate report 16 "
            "declares, so these thresholds cannot simply be adopted.")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "source": {
            "path": str(path.relative_to(ROOT)),
            "sha256": digest,
            "schema_version": src["schema_version"],
            "kind": src["kind"],
            "calibration_digest": src["calibration_digest"],
        },
        "derivation": {
            "what": "schema adaptation only: report 15's gate spelling "
                    "re-expressed as report 16's policy. No sampling was "
                    "performed and no number was recomputed.",
            "remeasured": False,
            "policy_field_map": dict(POLICY_FIELD_MAP),
            "carried_verbatim": list(CARRIED_VERBATIM),
            "measurement_context": list(MEASUREMENT_CONTEXT),
            "dropped_from_top_level": dict(DROPPED),
            "derived_by": "scripts/16_calibration_from_report15.py",
            # No timestamp. This document is a pure function of the source
            # bytes named above, so anyone can regenerate it and compare --
            # a clock reading would make that comparison impossible for the
            # sake of a fact the filesystem already records.
        },
        "measurement": {field: src[field] for field in MEASUREMENT_CONTEXT},
        "policy": policy,
        **{field: src[field] for field in CARRIED_VERBATIM},
    }


def main(argv=None) -> int:
    doc = derive()
    if OUT.exists():
        raise SystemExit(
            f"{OUT.relative_to(ROOT)} already exists. The thresholds in force "
            "are not something to have two versions of.")
    write_once_json(OUT, doc)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  source     {doc['source']['path']}")
    print(f"  source sha {doc['source']['sha256']}")
    print(f"  policy     {doc['policy']}")
    print(f"  thresholds {doc['thresholds']}")
    print(f"  sha256     {hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
