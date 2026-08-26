#!/usr/bin/env python3
"""Start the minimum two-page BrickAgain interface on this machine.

One command, one process, loopback only:

    ./.venv/bin/python scripts/29_ui.py

The interface is a composition layer over the existing delivery path.  It runs
on the CPU with no network access, loads no model weights, reaches no decoder,
never enables the Phase 3 placement gate, reads no frozen evaluation case, and
produces no metric.  The LDraw file and the 3-D preview it offers live in
memory for the life of the process; nothing is written into ``artifacts/``.

Exit codes:

===  ==========================================================
  0  the server ran and was stopped with Ctrl-C
  2  refused: a bad option, or a bind address that is not loopback
===  ==========================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.app import UiError, default_catalog
from src.ui.server import serve

EXIT_OK, EXIT_REFUSED = 0, 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BrickAgain minimum two-page UI. CPU, offline, "
                    "loopback only. Produces no metric.")
    p.add_argument("--port", type=int, default=8765,
                   help="loopback port to listen on (default 8765; "
                        "0 picks a free one)")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; loopback only, anything else is "
                        "refused rather than served")
    p.add_argument("--catalog", metavar="FILE",
                   help="train-only counterfactual JSONL; defaults to the "
                        "same file scripts/27_delivery.py uses")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.port < 0 or args.port > 65535:
        print(f"refused: port {args.port} is not a TCP port", file=sys.stderr)
        return EXIT_REFUSED
    try:
        catalog = Path(args.catalog) if args.catalog else default_catalog()
        return serve(host=args.host, port=args.port, catalog=catalog)
    except UiError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except OSError as exc:
        print(f"refused: could not listen on {args.host}:{args.port}: {exc}",
              file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
