#!/usr/bin/env python3
"""Verify and print the read-only delivery evidence closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.delivery.evidence import (EvidenceError, format_evidence_summary,
                                   sealed_delivery_summary)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only summary of sealed aggregate evidence")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = sealed_delivery_summary(ROOT)
    except EvidenceError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(format_evidence_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
