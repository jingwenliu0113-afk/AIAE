"""Connectivity-aware CP-SAT re-tiling for the delivery pipeline.

The historical counterfactual generator in :mod:`src.data.retile` minimizes
brick count under exact-cover and inventory constraints.  Connectivity was
measured only after that solve, so an otherwise usable shape could be rejected
because the optimizer chose a disconnected optimum.  This module keeps the
historical solver byte-for-byte reproducible and adds the missing constraint
only on the live delivery path.

Connectivity has the same narrow meaning as :func:`src.data.bricks.is_connected`:
two selected placements join only when they occupy adjacent layers and their
2-D footprints overlap.  It is not support, stability, or a force claim.
"""

from __future__ import annotations

from ortools.sat.python import cp_model

from src.data.bricks import PART_VOCAB, Brick
from src.data.retile import Cell, RetileResult, _candidates


def _adjacent_pairs(candidates: list[Brick]) -> list[tuple[int, int]]:
    """Undirected candidate pairs that would share at least one stud cell."""
    by_cell: dict[Cell, list[int]] = {}
    for index, brick in enumerate(candidates):
        for x, y in brick.footprint:
            by_cell.setdefault((x, y, brick.z), []).append(index)

    pairs: set[tuple[int, int]] = set()
    for lower, brick in enumerate(candidates):
        for x, y in brick.footprint:
            for upper in by_cell.get((x, y, brick.z + 1), ()):
                pairs.add((lower, upper))
    return sorted(pairs)


def retile_connected(
    occ: set[Cell],
    *,
    allowed: frozenset[str] | None = None,
    budget: dict[str, int] | None = None,
    time_limit: float = 10.0,
    seed: int = 0,
    stagger: bool = False,
    workers: int = 1,
) -> RetileResult:
    """Tile ``occ`` exactly while requiring one stud-connected component.

    A single-commodity flow starts at the unique selected placement covering
    the lexicographically first occupied cell.  Every other selected placement
    must consume one unit of that flow, and flow may travel only across a
    selected adjacent-layer pair.  An exact cover therefore cannot satisfy the
    model by routing through an unselected placement.
    """
    allowed = allowed if allowed is not None else frozenset(PART_VOCAB)
    if not occ:
        return RetileResult([], "OPTIMAL", 0.0, 0)

    candidates = _candidates(occ, allowed)
    model = cp_model.CpModel()
    use = [model.new_bool_var(f"b{i}") for i in range(len(candidates))]

    covers: dict[Cell, list[int]] = {cell: [] for cell in occ}
    for index, brick in enumerate(candidates):
        for cell in brick.cells:
            covers[cell].append(index)
    for indices in covers.values():
        if not indices:
            return RetileResult(None, "INFEASIBLE", 0.0, len(candidates))
        model.add_exactly_one(use[index] for index in indices)

    if budget is not None:
        by_part: dict[str, list[int]] = {}
        for index, brick in enumerate(candidates):
            by_part.setdefault(brick.part, []).append(index)
        for part, indices in by_part.items():
            cap = budget.get(part)
            if cap is not None:
                model.add(sum(use[index] for index in indices) <= cap)

    if stagger:
        by_footprint: dict[tuple[int, frozenset[tuple[int, int]]], int] = {}
        for index, brick in enumerate(candidates):
            by_footprint[(brick.z, frozenset(brick.footprint))] = index
        for index, brick in enumerate(candidates):
            below = by_footprint.get(
                (brick.z - 1, frozenset(brick.footprint)))
            if below is not None:
                model.add_bool_or([use[index].negated(), use[below].negated()])

    # At most one selected brick per occupied cell, so |occ|-1 is a valid
    # capacity for every flow arc and for the root's total supply.
    flow_cap = max(0, len(occ) - 1)
    incoming: list[list] = [[] for _ in candidates]
    outgoing: list[list] = [[] for _ in candidates]
    for left, right in _adjacent_pairs(candidates):
        for start, end in ((left, right), (right, left)):
            flow = model.new_int_var(0, flow_cap, f"f{start}_{end}")
            model.add(flow <= flow_cap * use[start])
            model.add(flow <= flow_cap * use[end])
            outgoing[start].append(flow)
            incoming[end].append(flow)

    anchor = min(occ)
    root_candidates = set(covers[anchor])
    root_supply = []
    supplies: dict[int, object] = {}
    for index in sorted(root_candidates):
        supply = model.new_int_var(0, flow_cap, f"root_supply_{index}")
        model.add(supply <= flow_cap * use[index])
        supplies[index] = supply
        root_supply.append(supply)
    model.add(sum(root_supply) == sum(use) - 1)

    for index in range(len(candidates)):
        selected_root = use[index] if index in root_candidates else 0
        supply = supplies.get(index, 0)
        model.add(
            sum(incoming[index]) - sum(outgoing[index])
            == use[index] - selected_root - supply
        )

    model.minimize(sum(use))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.random_seed = seed
    solver.parameters.num_workers = workers
    status = solver.solve(model)
    name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return RetileResult(None, name, solver.wall_time, len(candidates))

    picked = [brick for index, brick in enumerate(candidates)
              if solver.value(use[index])]
    picked.sort(key=lambda brick: (brick.z, brick.x, brick.y))
    return RetileResult(picked, name, solver.wall_time, len(candidates))

