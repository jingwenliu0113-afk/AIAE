"""Paired counterfactual samples: same object, different stock, different build.

For one source structure this emits a *pair*:

* **control** -- the voxel shape re-tiled with the full part vocabulary.
* **counterfactual** -- the same voxel shape re-tiled with one part forbidden.

Both targets come out of CP-SAT.  That matters: CP-SAT minimises brick count
and so tiles more tightly than the corpus does (measured, 92 bricks -> 85), and
if the control came from the corpus while the counterfactual came from the
solver, the pair would differ in tiling *style* as well as in stock.  A model
could then score well by learning "restricted stock means tile minimally"
without ever reading the inventory.  Sharing the generator holds style fixed so
stock is the only thing that varies.

The forbidden part is never ``1x1``: dropping it is feasible only rarely,
because any leftover odd cell becomes untileable, whereas dropping any of the
other seven nearly always succeeds (data/reports/02_retile.md).

Each target gets four inventory framings -- exact, loose, distractor, mixed.
In the counterfactual arm the dropped part is kept out of every one of them;
letting it back in as a distractor would hand back the very part whose absence
is the lesson.

Both arms must be a single connected component **under stud coupling alone**,
or the pair is discarded.  The baseplate is not a part, holds no inventory and
is never written out, so it cannot be what holds a model together; anchoring is
tracked separately as ``n_ground_components`` and never gates acceptance.

Minimising brick count does not preserve connectivity, so rather than drop a
source the first time a chosen part fails, every droppable part is tried in a
seeded order until one yields a stud-connected counterfactual.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.data.bricks import (
    PART_VOCAB,
    Brick,
    connected_components,
    format_bricks,
    is_connected,
    parse_bricks,
    touches_ground,
    unsupported_bricks,
)
from src.data.retile import occupancy_of, retile, verify

#: Inventory framings applied to each target.
VARIANTS = ("exact", "loose", "distractor", "mixed")

LOOSE_TAU = 1.5
N_DISTRACTORS = 3
DISTRACTOR_QTY = 4


class GenerationError(RuntimeError):
    pass


@dataclass
class Sample:
    sample_id: str
    pair_id: str
    role: str                      # control | counterfactual
    variant: str                   # exact | loose | distractor

    # provenance, inherited from the source row
    object_id: str
    structure_id: str
    split: str
    caption: str
    caption_index: int

    inventory: dict[str, int]
    used: dict[str, int]
    bricks_txt: str

    dropped_part: str | None
    seed: int
    solver_status: str
    solve_seconds: float
    n_bricks: int
    n_cells: int
    n_components: int = 1          # stud coupling only -- the acceptance gate
    n_ground_components: int = 1   # baseplate counted -- anchoring metric only
    n_unsupported: int = 0
    tried_parts: list[str] = field(default_factory=list)
    extra_parts: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def bricks(self) -> list[Brick]:
        return parse_bricks(self.bricks_txt)


def _inventory_variants(
    used: Counter[str], *, forbidden: set[str], rng: random.Random
) -> dict[str, dict[str, int]]:
    """The four inventory framings for one target.

    A distractor variant that adds nothing is just a duplicate of ``exact``
    mislabelled, so an empty pool is an error rather than a silent no-op.
    """
    exact = dict(sorted(used.items()))
    loose = {p: math.ceil(LOOSE_TAU * n) for p, n in exact.items()}

    pool = [p for p in PART_VOCAB if p not in used and p not in forbidden]
    if not pool:
        raise GenerationError(
            "no distractor pool: target already uses every permitted part"
        )
    rng.shuffle(pool)
    extra = {p: DISTRACTOR_QTY for p in pool[:N_DISTRACTORS]}

    return {
        "exact": exact,
        "loose": dict(sorted(loose.items())),
        "distractor": dict(sorted({**exact, **extra}.items())),
        "mixed": dict(sorted({**loose, **extra}.items())),
    }


def _check(
    bricks: list[Brick],
    occ: set[tuple[int, int, int]],
    inventory: dict[str, int],
    forbidden: set[str],
    used_parts: Counter[str],
) -> dict[str, bool]:
    used = Counter(b.part for b in bricks)
    try:
        verify(occ, bricks)
        exact_cover = True
    except AssertionError:
        exact_cover = False
    return {
        "exact_cover": exact_cover,
        "voxel_identical": occupancy_of(bricks) == occ,
        "within_inventory": all(inventory.get(p, 0) >= n for p, n in used.items()),
        "parts_canonical": all(b.part in PART_VOCAB for b in bricks),
        "forbidden_absent": not (set(used) & forbidden),
        "forbidden_not_offered": not (set(inventory) & forbidden),
        "connected": is_connected(bricks),
        "touches_ground": touches_ground(bricks),
        "inventory_adds_something": set(inventory) >= set(used_parts),
    }


def make_pair(
    row: dict,
    split: str,
    *,
    seed: int = 0,
    caption_index: int = -1,
    time_limit: float = 10.0,
) -> list[Sample]:
    """Build control + counterfactual samples for one source structure.

    ``row`` needs structure_id, object_id, captions and bricks.
    Raises :class:`GenerationError` when either solve fails, so callers can
    count the drop-out rate rather than silently emitting half a pair.
    """
    rng = random.Random(f"{row['structure_id']}:{seed}")
    source = parse_bricks(row["bricks"])
    occ = occupancy_of(source)
    caption = row["captions"][caption_index]

    t = time.time()
    control = retile(occ, time_limit=time_limit, seed=seed)
    if not control.ok:
        raise GenerationError(f"control retile failed: {control.status}")
    control_secs = time.time() - t
    if not is_connected(control.bricks):
        raise GenerationError("control disconnected")

    # Try every droppable part, in a seeded order, and keep the first that
    # gives a stud-connected counterfactual. Giving up after one failed choice
    # discards sources that would have worked with a different part.
    candidates = sorted(p for p in control.inventory if p != "1x1")
    if not candidates:
        raise GenerationError("no droppable part (structure is all 1x1)")
    rng.shuffle(candidates)

    dropped = None
    cf = None
    cf_secs = 0.0
    tried: list[str] = []
    for part in candidates:
        t = time.time()
        res = retile(
            occ,
            allowed=frozenset(PART_VOCAB) - {part},
            time_limit=time_limit,
            seed=seed,
        )
        cf_secs += time.time() - t
        tried.append(part)
        if res.ok and is_connected(res.bricks):
            dropped, cf = part, res
            break
    if cf is None:
        raise GenerationError(
            f"counterfactual disconnected for all {len(tried)} droppable parts"
        )

    pair_id = f"{row['structure_id']}:{seed}"
    out: list[Sample] = []

    for role, res, secs, forbidden in (
        ("control", control, control_secs, set()),
        ("counterfactual", cf, cf_secs, {dropped}),
    ):
        used = res.inventory
        for variant, inv in _inventory_variants(
            used, forbidden=forbidden, rng=rng
        ).items():
            s = Sample(
                sample_id=f"{pair_id}:{role}:{variant}",
                pair_id=pair_id,
                role=role,
                variant=variant,
                object_id=row["object_id"],
                structure_id=row["structure_id"],
                split=split,
                caption=caption,
                caption_index=caption_index,
                inventory=inv,
                used=dict(sorted(used.items())),
                bricks_txt=format_bricks(res.bricks),
                dropped_part=dropped if role == "counterfactual" else None,
                seed=seed,
                solver_status=res.status,
                solve_seconds=round(secs, 3),
                n_bricks=len(res.bricks),
                n_cells=len(occ),
                n_components=len(connected_components(res.bricks)),
                n_ground_components=len(
                    connected_components(res.bricks, ground=True)
                ),
                n_unsupported=len(unsupported_bricks(res.bricks)),
                tried_parts=tried if role == "counterfactual" else [],
                extra_parts=sorted(set(inv) - set(used)),
            )
            s.checks = _check(res.bricks, occ, inv, forbidden, used)
            out.append(s)
    return out


def write_jsonl(samples: list[Sample], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    return p


def read_jsonl(path: str | Path) -> list[Sample]:
    return [
        Sample(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
