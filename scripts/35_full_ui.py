#!/usr/bin/env python3
"""Serve the full BrickAgain interface on loopback.

    ./.venv/bin/python scripts/35_full_ui.py           # http://127.0.0.1:8766/

Four pages: inventory and request, photograph recognition and correction,
result and delivery, build steps.  Three methods: multilingual retrieval over
the train-only catalogue, the minimum F-pipeline, and one demonstration decode
with the archived ``final_H2``.

The two-page interface is unchanged and still available at
``scripts/29_ui.py``.  This one adds pages; it does not replace them.

Boundaries, all of which the interface enforces rather than describes:

* loopback only, with a ``Host`` check as well as the bind address;
* a structured external ``Origin`` refused before its body is read, and every
  submission required to carry the per-process form key;
* photographs, corrections, previews and LDraw held in bounded in-process
  memory and never written into the project;
* ``final_H2`` verified against ``runs/project_model.json`` before any weight
  is read, never retrained, never tuned, never reselected;
* the placement gate opt-in, off by default, and labelled as never formally
  evaluated whenever it is on;
* no Phase 3C, no frozen evaluation case, no Success@K, no metric.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui import app as ui_app
from src.ui import server_full

DEFAULT_INDEX = ROOT / "runs/retrieval/index"
DEFAULT_CHECKPOINT = ROOT / "runs/vision/classifier"

EXIT_OK, EXIT_REFUSED = 0, 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1",
                   help="loopback only; anything else is refused by name")
    p.add_argument("--port", type=int, default=8766,
                   help="0 lets the system choose")
    p.add_argument("--catalog", help="train-only catalogue JSONL")
    p.add_argument("--index", default=str(DEFAULT_INDEX),
                   help="retrieval index directory; RAG is refused without it")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                   help="fitted vision classifier; without it only the "
                        "traditional CV baseline is offered")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"),
                   help="device for the vision classifier and the decoder")
    p.add_argument("--no-project-model", action="store_true",
                   help="refuse the final_H2 entry entirely, so this run "
                        "cannot load any weights")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    index = Path(args.index) if args.index else None
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    try:
        server = server_full.create_server(
            host=args.host, port=args.port, catalog=args.catalog,
            index_dir=(index if index and index.is_dir() else None),
            checkpoint=(checkpoint if checkpoint and checkpoint.is_dir()
                        else None),
            device=args.device, project_root=ROOT)
        server.allow_project_model = not args.no_project_model
    except ui_app.UiError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    bound_host, bound_port = server.server_address[:2]
    print("BrickAgain 完整介面")
    print(f"  網址      : http://{bound_host}:{bound_port}/")
    print(f"  目錄檔    : {server.catalog}"
          + ("" if server.catalog.is_file() else "  (不存在：送出後會被拒絕)"))
    print(f"  RAG 索引  : {server.index_dir or '(未提供，RAG 會被具名拒絕)'}")
    print(f"  視覺模型  : {server.checkpoint or '(未提供，只用 CV baseline)'}")
    print(f"  正式模型  : "
          + ("final_H2，載入前先驗證 runs/project_model.json"
             if server.allow_project_model else "(本次啟動關閉)"))
    print("  邊界      : 本機、離線、只綁 loopback；照片與結果只存在記憶體，"
          "不寫入 artifacts/")
    print("  跨站防護  : 外部網址來源拒絕；所有送出都必須帶第一頁的表單金鑰")
    print("  停止      : Ctrl-C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。照片、預覽與 LDraw 只存在記憶體中，沒有留下任何檔案。")
    finally:
        server.server_close()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
