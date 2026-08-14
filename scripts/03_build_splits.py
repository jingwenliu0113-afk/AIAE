"""Freeze the object_id -> split manifest.

Run once.  Re-running refuses to overwrite unless --force is passed, since
every later experiment is measured against this assignment.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import load_dataset  # noqa: E402

from src.data.splits import MANIFEST_PATH, build  # noqa: E402

FORCE = "--force" in sys.argv


def main() -> None:
    ds = load_dataset("AvaLovelace/StableText2Brick")
    rows = {
        k: v.select_columns(["structure_id", "object_id"]).to_list()
        for k, v in ds.items()
    }
    m = build(rows)
    counts = m.counts()
    total = sum(counts.values())

    print("objects per split:")
    for s, n in counts.items():
        print(f"  {s:6s} {n:6d}  ({n/total:5.1%})")
    print(f"  total  {total:6d}")

    # structures, not just objects
    per_struct: dict[str, int] = {}
    for sid in m.structures:
        s = m.split_of_structure(sid)
        per_struct[s] = per_struct.get(s, 0) + 1
    print("\nstructures per split:")
    for s, n in sorted(per_struct.items()):
        print(f"  {s:6s} {n:6d}  ({n/len(m.structures):5.1%})")

    path = m.save(MANIFEST_PATH, force=FORCE)
    print(f"\nwrote {path} ({path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
