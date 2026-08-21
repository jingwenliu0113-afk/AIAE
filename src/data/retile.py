"""Re-tile a voxel shape with a different set of parts (CP-SAT).

This is the counterfactual data generator.  Take a structure, forbid a part it
relies on, and re-tile the *same* voxel occupancy with what is left.  The
result keeps the object and its caption but consumes a different inventory,
which is the ``same request + different stock -> different valid build``
signal the corpus barely supplies on its own (only 1,251 objects have natural
variants that differ in which part types they use -- see data/reports/01_eda.md).

Every brick is one unit tall, so a layer never constrains the layer above it
*except* through a shared part budget.  With per-part budgets left unlimited
the layers are independent and are solved one at a time, which is far faster
than one joint model; a global budget forces the joint solve.

There is no "1x1 makes it always solvable" guarantee.  1x1 tiles any shape only
when its supply is unbounded, and bounded supply is the whole premise here, so
every call can legitimately return ``None``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ortools.sat.python import cp_model

from src.data.bricks import PART_VOCAB, Brick

Cell = tuple[int, int, int]

#: (part, h, w) for both spellings of each part.
PLACEMENT_SHAPES: tuple[tuple[str, int, int], ...] = tuple(
    dict.fromkeys(  # de-dupes the square parts, whose two spellings coincide
        (p, hw[i], hw[1 - i])
        for p in PART_VOCAB
        for hw in [tuple(int(v) for v in p.split("x"))]
        for i in (0, 1)
    )
)


#: Solver statuses ordered from strongest guarantee to weakest.
STATUS_RANK = ("OPTIMAL", "FEASIBLE", "UNKNOWN", "INFEASIBLE", "MODEL_INVALID")


def weaker_status(a: str, b: str) -> str:
    """The weaker of two solver statuses, for aggregating a layered solve."""
    rank = {s: i for i, s in enumerate(STATUS_RANK)}
    return a if rank.get(a, len(STATUS_RANK)) >= rank.get(b, len(STATUS_RANK)) else b


@dataclass
class RetileResult:
    bricks: list[Brick] | None
    status: str            # OPTIMAL | FEASIBLE | INFEASIBLE | UNKNOWN(timeout)
    wall_seconds: float
    candidates: int

    @property
    def ok(self) -> bool:
        return self.bricks is not None

    @property
    def inventory(self) -> Counter[str]:
        return Counter(b.part for b in self.bricks or [])


def occupancy_of(bricks: list[Brick]) -> set[Cell]:
    return {c for b in bricks for c in b.cells}


def _candidates(
    occ: set[Cell], allowed: frozenset[str], z: int | None = None
) -> list[Brick]:
    """Every placement that lies entirely inside the occupancy."""
    cells = occ if z is None else {c for c in occ if c[2] == z}
    out: list[Brick] = []
    for part, h, w in PLACEMENT_SHAPES:
        if part not in allowed:
            continue
        for (cx, cy, cz) in cells:
            if all(
                (cx + dx, cy + dy, cz) in occ
                for dx in range(h)
                for dy in range(w)
            ):
                out.append(Brick(h=h, w=w, x=cx, y=cy, z=cz))
    return out


def _solve(
    occ: set[Cell],
    cands: list[Brick],
    budget: dict[str, int] | None,
    *,
    time_limit: float,
    seed: int,
    stagger: bool,
    workers: int,
) -> tuple[list[Brick] | None, str, float]:
    model = cp_model.CpModel()
    use = [model.new_bool_var(f"b{i}") for i in range(len(cands))]

    covers: dict[Cell, list[int]] = {c: [] for c in occ}
    for i, b in enumerate(cands):
        for c in b.cells:
            covers[c].append(i)

    for c, idxs in covers.items():
        if not idxs:
            return None, "INFEASIBLE", 0.0   # a cell no allowed part can reach
        model.add_exactly_one(use[i] for i in idxs)

    if budget is not None:
        by_part: dict[str, list[int]] = {}
        for i, b in enumerate(cands):
            by_part.setdefault(b.part, []).append(i)
        for part, idxs in by_part.items():
            cap = budget.get(part)
            if cap is not None:
                model.add(sum(use[i] for i in idxs) <= cap)

    if stagger:
        # Forbid a brick sitting directly on an identical footprint in the
        # layer below. Added on the assumption that offset seams make a sounder
        # structure; matching footprints are in fact the strongest stud bonds
        # available. Off by default; scripts/09_stagger_ablation.py measures it
        # against an otherwise identical joint solve.
        by_fp: dict[tuple[int, frozenset], int] = {}
        for i, b in enumerate(cands):
            by_fp[(b.z, frozenset(b.footprint))] = i
        for i, b in enumerate(cands):
            j = by_fp.get((b.z - 1, frozenset(b.footprint)))
            if j is not None:
                model.add_bool_or([use[i].negated(), use[j].negated()])

    model.minimize(sum(use))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.random_seed = seed
    # workers=1 is required for reproducibility, not just preferred: with a
    # parallel portfolio, whichever worker finishes first wins, so repeated
    # runs return different optimal tilings for the same seed. Measured on the
    # test slab: same objective, different layout, every run.
    solver.parameters.num_workers = workers
    status = solver.solve(model)
    name = solver.status_name(status)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        picked = [cands[i] for i in range(len(cands)) if solver.value(use[i])]
        picked.sort(key=lambda b: (b.z, b.x, b.y))
        return picked, name, solver.wall_time
    return None, name, solver.wall_time


def retile(
    occ: set[Cell],
    *,
    allowed: frozenset[str] | None = None,
    budget: dict[str, int] | None = None,
    time_limit: float = 10.0,
    seed: int = 0,
    stagger: bool = False,
    workers: int = 1,
) -> RetileResult:
    """Tile ``occ`` exactly, using only ``allowed`` parts within ``budget``.

    ``budget`` maps canonical part -> max count; a part absent from the dict is
    unlimited.  When ``budget`` is None the layers are solved independently.

    ``workers`` defaults to 1 so results are reproducible; raising it makes the
    solver return a different (equally optimal) tiling on each run, which would
    make generated datasets impossible to regenerate.
    """
    allowed = allowed if allowed is not None else frozenset(PART_VOCAB)
    if not occ:
        return RetileResult([], "OPTIMAL", 0.0, 0)

    if budget is None:
        if stagger:
            # The per-layer solve cannot see the layer below, so a stagger
            # constraint has nothing to act on. Silently dropping it would let
            # a caller believe an experiment was staggered when it was not.
            raise ValueError(
                "stagger=True requires a joint solve; pass budget={} to force one"
            )
        # Layers are independent; solving them separately is much cheaper.
        out: list[Brick] = []
        total_t = 0.0
        total_c = 0
        worst = "OPTIMAL"
        for z in sorted({c[2] for c in occ}):
            cands = _candidates(occ, allowed, z=z)
            total_c += len(cands)
            layer = {c for c in occ if c[2] == z}
            got, status, t = _solve(
                layer, cands, None, time_limit=time_limit, seed=seed,
                stagger=False, workers=workers,
            )
            total_t += t
            if got is None:
                return RetileResult(None, status, total_t, total_c)
            worst = weaker_status(worst, status)
            out.extend(got)
        # A layer that only reached FEASIBLE makes the whole tiling FEASIBLE.
        # Reporting OPTIMAL regardless would claim a minimality guarantee the
        # result does not have.
        return RetileResult(out, worst, total_t, total_c)

    cands = _candidates(occ, allowed)
    got, status, t = _solve(
        occ, cands, budget, time_limit=time_limit, seed=seed,
        stagger=stagger, workers=workers,
    )
    return RetileResult(got, status, t, len(cands))


def drop_part(
    bricks: list[Brick], part: str, **kw
) -> RetileResult:
    """The counterfactual move: rebuild the same shape without ``part``."""
    allowed = frozenset(PART_VOCAB) - {part}
    return retile(occupancy_of(bricks), allowed=allowed, **kw)


def verify(occ: set[Cell], bricks: list[Brick]) -> None:
    """Assert the tiling covers the shape exactly once, nothing more."""
    seen: set[Cell] = set()
    for b in bricks:
        for c in b.cells:
            assert c not in seen, f"double cover at {c}"
            assert c in occ, f"brick escapes the shape at {c}"
            seen.add(c)
    assert seen == occ, f"missed {len(occ - seen)} cells"
