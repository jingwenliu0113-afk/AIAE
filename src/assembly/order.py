"""A build order, and the check that runs after every step.

Given a finished structure, produce a sequence a person could follow.  The rule
for what may be placed next is the project's own definition of a joint and
nothing looser:

* a brick resting on the ground may always be placed;
* any other brick may be placed only once something already placed in the layer
  below it shares part of its footprint.

That second condition is what "do not place a brick with nothing under it"
means here, and it is deliberately *not* global connectivity: several grounded
sub-assemblies may be built independently and joined later by a beam, which is
exactly the case the project's connectivity rules were written to allow.  So the
accumulated structure is allowed to be in several pieces at every intermediate
step, and only the finished one has to be a single component.

**A brick nothing can ever be placed under is reported, not worked around.**  A
brick above the ground with no brick anywhere beneath its footprint is held from
above; there is no order in which a person could place it, and
:func:`plan` says so with the indices rather than inventing a sequence.

**Every step is re-verified from scratch.**  After each step the accumulated
brick list is re-checked for bounds, for collisions, against the stock, and for
ground contact -- not incrementally, but by running the same checks over the
whole accumulation.  An incremental check that drifts is the failure mode this
avoids, at a cost that does not matter for a hundred bricks.

``stud_only_connected`` here means adjacent-layer footprint overlap.  It is not
support, not load bearing and not stability: nothing in this project checks
centre of mass or whether a model stands up.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from src.data.bricks import (Brick, connected_components, find_collisions,
                             is_connected, touches_ground)

#: Bricks added per step.  One is the first version, and it is what makes every
#: step trivially checkable; the planner takes any positive limit.
DEFAULT_MAX_PER_STEP = 1

#: The most steps a plan may have.  The world holds 8,000 cells and a structure
#: here is around a hundred bricks, so this only ever fires on a defect.
MAX_STEPS = 4096


class AssemblyError(ValueError):
    """The structure cannot be given a legal build order."""


@dataclass(frozen=True)
class Step:
    """One step: what is added, and the state of everything after it."""

    number: int
    added: tuple[int, ...]
    total_bricks: int
    added_parts: dict[str, int]
    cumulative_parts: dict[str, int]
    grounded_additions: tuple[int, ...]
    supported_additions: tuple[int, ...]
    components: int
    touches_ground: bool
    collision_free: bool
    in_bounds: bool
    within_stock: bool
    stock_remaining: dict[str, int] | None

    @property
    def valid(self) -> bool:
        return (self.collision_free and self.in_bounds and self.within_stock
                and self.touches_ground)

    def as_dict(self) -> dict:
        return {
            "step": self.number,
            "added": list(self.added),
            "added_parts": dict(self.added_parts),
            "total_bricks": self.total_bricks,
            "cumulative_parts": dict(self.cumulative_parts),
            "grounded_additions": list(self.grounded_additions),
            "supported_additions": list(self.supported_additions),
            "components": self.components,
            "touches_ground": self.touches_ground,
            "collision_free": self.collision_free,
            "in_bounds": self.in_bounds,
            "within_stock": self.within_stock,
            "stock_remaining": self.stock_remaining,
            "valid": self.valid,
        }


@dataclass(frozen=True)
class AssemblyPlan:
    """A whole build order and the verdicts on it."""

    bricks: tuple[Brick, ...] = field(repr=False)
    steps: tuple[Step, ...] = field(repr=False)
    max_per_step: int
    final_collision_free: bool
    final_connected: bool
    final_touches_ground: bool
    stock: dict[str, int] | None = None

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def order(self) -> tuple[int, ...]:
        return tuple(index for step in self.steps for index in step.added)

    @property
    def step_lists(self) -> list[list[int]]:
        return [list(step.added) for step in self.steps]

    @property
    def every_step_valid(self) -> bool:
        return all(step.valid for step in self.steps)

    @property
    def ready(self) -> bool:
        """Every step legal, and the finished structure sound.

        Connectivity is required of the *end* only.  Requiring it of every
        step would forbid two towers joined by a beam, which is a shape this
        project explicitly supports.
        """
        return (self.every_step_valid and self.final_collision_free
                and self.final_connected and self.final_touches_ground)

    def prefix_indices(self, step_number: int) -> list[int]:
        """Which bricks are in place after ``step_number`` steps, in order.

        Returned as indices into :attr:`bricks` rather than as bricks, because
        anything keyed by a brick's index -- a colour assignment above all --
        has to be re-keyed when the prefix renumbers them, and re-keying is
        not possible from a list of bricks alone.
        """
        if isinstance(step_number, bool) or not isinstance(step_number, int):
            raise AssemblyError("a step number must be a whole number")
        if not 0 <= step_number <= self.n_steps:
            raise AssemblyError(
                f"step {step_number} is outside 0..{self.n_steps}")
        return [index for step in self.steps[:step_number]
                for index in step.added]

    def prefix(self, step_number: int) -> list[Brick]:
        """The accumulated structure after ``step_number`` steps.

        ``0`` is the empty structure, so a viewer's "before the first step"
        state is expressible without a special case.
        """
        return [self.bricks[index]
                for index in self.prefix_indices(step_number)]

    def as_dict(self) -> dict:
        return {
            "kind": "brickagain.assembly_plan",
            "n_bricks": len(self.bricks),
            "n_steps": self.n_steps,
            "max_per_step": self.max_per_step,
            "order": list(self.order),
            "steps": [step.as_dict() for step in self.steps],
            "every_step_valid": self.every_step_valid,
            "final_collision_free": self.final_collision_free,
            "final_stud_only_connected": self.final_connected,
            "final_touches_ground": self.final_touches_ground,
            "ready": self.ready,
            "stock": self.stock,
            "rule": (
                "a brick may be placed when it rests on the ground, or when "
                "something already placed one layer below shares part of its "
                "footprint. Several grounded sub-assemblies may be built "
                "before a beam joins them, so an intermediate step may be in "
                "several pieces"),
            "not_a_physics_claim": (
                "stud_only_connected is adjacent-layer footprint overlap. It "
                "is not support, not load bearing and not stability; nothing "
                "here checks centre of mass or whether the model stands up"),
        }


def _placeable(bricks, remaining, placed) -> list[int]:
    """Indices in ``remaining`` that may legally be placed next."""
    below: dict[int, list[int]] = {}
    for index in placed:
        below.setdefault(bricks[index].z, []).append(index)
    out = []
    for index in sorted(remaining):
        brick = bricks[index]
        if brick.z == 0:
            out.append(index)
            continue
        if any(brick.footprint & bricks[other].footprint
               for other in below.get(brick.z - 1, ())):
            out.append(index)
    return out


def unplaceable_from_below(bricks) -> list[int]:
    """Bricks above the ground with nothing anywhere beneath their footprint.

    These are held from above.  No build order can place them, so they are
    named rather than absorbed into a plan that could not be followed.  This is
    the same question ``src.data.bricks.unsupported_bricks`` answers, computed
    here because the planner needs it before it starts rather than as a
    descriptive statistic afterwards.
    """
    from src.data.bricks import unsupported_bricks

    return list(unsupported_bricks(list(bricks)))


def plan(bricks, *, max_per_step: int = DEFAULT_MAX_PER_STEP,
         stock: dict[str, int] | None = None) -> AssemblyPlan:
    """Order a structure into steps, re-verifying after each one.

    The choice of *which* placeable brick goes next is fixed: lowest layer,
    then smallest x, then y, then the brick's own extents, then its index.  A
    stated tie-break is what makes two runs on the same structure produce the
    same steps, and low-layer-first is what makes the order read like a build.
    """
    bricks = list(bricks)
    if not bricks:
        raise AssemblyError("there is no structure to order")
    for brick in bricks:
        if not isinstance(brick, Brick):
            raise AssemblyError(
                f"a brick must be a Brick, not {type(brick).__name__}")
    if isinstance(max_per_step, bool) or not isinstance(max_per_step, int) \
            or max_per_step < 1:
        raise AssemblyError("max_per_step must be a positive whole number")
    if not touches_ground(bricks):
        raise AssemblyError(
            "no brick rests on the ground, so there is nothing that may be "
            "placed first")
    floating = unplaceable_from_below(bricks)
    if floating:
        raise AssemblyError(
            f"{len(floating)} brick(s) sit above the ground with nothing "
            f"beneath them (indices {floating[:6]}). They are held from above, "
            "so no build order can place them; the structure is reported as "
            "un-orderable rather than given a sequence a person could not "
            "follow")

    required = Counter(brick.part for brick in bricks)
    if stock is not None:
        short = {part: (count, stock.get(part, 0))
                 for part, count in required.items()
                 if stock.get(part, 0) < count}
        if short:
            raise AssemblyError(
                "the stock does not cover this structure: "
                + "; ".join(f"{part}: needs {want}, has {got}"
                            for part, (want, got) in sorted(short.items())))

    remaining = set(range(len(bricks)))
    placed: list[int] = []
    steps: list[Step] = []
    used: Counter[str] = Counter()

    while remaining:
        if len(steps) >= MAX_STEPS:
            raise AssemblyError(
                f"the plan exceeded {MAX_STEPS} steps; this is a defect")
        ready = _placeable(bricks, remaining, placed)
        if not ready:
            raise AssemblyError(
                f"{len(remaining)} brick(s) cannot be reached from what is "
                f"already placed (indices {sorted(remaining)[:6]}); the "
                "structure has no legal continuation from here")
        ready.sort(key=lambda i: (bricks[i].z, bricks[i].x, bricks[i].y,
                                  bricks[i].h, bricks[i].w, i))
        # Adding several bricks in one step must not let them lean on each
        # other: each is checked against what was already placed, so a step is
        # a set of independently legal placements rather than a chain.
        added = tuple(ready[:max_per_step])
        for index in added:
            remaining.discard(index)
            placed.append(index)
            used[bricks[index].part] += 1

        accumulated = [bricks[index] for index in placed]
        components = connected_components(accumulated)
        remaining_stock = (
            {part: stock.get(part, 0) - used.get(part, 0)
             for part in sorted(set(stock) | set(used))}
            if stock is not None else None)
        steps.append(Step(
            number=len(steps) + 1,
            added=added,
            total_bricks=len(accumulated),
            added_parts=dict(sorted(
                Counter(bricks[i].part for i in added).items())),
            cumulative_parts=dict(sorted(used.items())),
            grounded_additions=tuple(i for i in added if bricks[i].z == 0),
            supported_additions=tuple(i for i in added if bricks[i].z != 0),
            components=len(components),
            touches_ground=touches_ground(accumulated),
            collision_free=not find_collisions(accumulated),
            in_bounds=all(brick.in_bounds() for brick in accumulated),
            within_stock=(remaining_stock is None
                          or all(value >= 0
                                 for value in remaining_stock.values())),
            stock_remaining=remaining_stock))

    return AssemblyPlan(
        bricks=tuple(bricks), steps=tuple(steps), max_per_step=max_per_step,
        final_collision_free=not find_collisions(bricks),
        final_connected=is_connected(bricks),
        final_touches_ground=touches_ground(bricks),
        stock=dict(stock) if stock is not None else None)


def to_ldr(plan_result: AssemblyPlan, *,
           colours: dict[int, int] | None = None) -> str:
    """The plan as LDraw, with one ``0 STEP`` per step of the plan."""
    from src.rendering.ldr import to_ldr_steps

    return to_ldr_steps(plan_result.step_lists, list(plan_result.bricks),
                        colours=colours)


def write_step_previews(plan_result: AssemblyPlan, directory, *,
                        title: str | None = None, suffix: str = ".png",
                        colours: dict[int, int] | None = None) -> list:
    """One CPU preview per step, each showing everything placed so far.

    Reuses :func:`src.rendering.preview.write_preview`, so a step image and the
    finished image are drawn by the same code.  The first step's image is the
    first brick alone; there is no image for the empty structure, because a
    preview of nothing is not informative and the writer refuses it.

    ``colours`` is the assignment keyed by index into ``plan_result.bricks`` --
    the same mapping :func:`to_ldr` is given.  Each step's prefix renumbers its
    bricks from zero, so the mapping is re-keyed to that prefix before it is
    drawn.  Handing the writer the original keys would colour step three's
    bricks with step one's colours, which is exactly the disagreement that
    passing one assignment through both writers is meant to prevent.
    """
    from pathlib import Path

    from src.rendering.preview import write_preview

    if colours is not None:
        missing = [index for index in range(len(plan_result.bricks))
                   if index not in colours]
        if missing:
            raise AssemblyError(
                f"the colour assignment covers {len(colours)} of "
                f"{len(plan_result.bricks)} bricks; indices {missing[:8]} "
                "have none. Step images are not drawn from a partial "
                "assignment, because they would then disagree with the LDraw "
                "file without saying so")

    target = Path(directory)
    out = []
    width = len(str(plan_result.n_steps))
    for step in plan_result.steps:
        indices = plan_result.prefix_indices(step.number)
        accumulated = [plan_result.bricks[index] for index in indices]
        step_colours = (
            None if colours is None
            else {position: colours[index]
                  for position, index in enumerate(indices)})
        caption = (f"{title + ' — ' if title else ''}"
                   f"step {step.number}/{plan_result.n_steps}, "
                   f"{step.total_bricks} brick(s)")
        out.append(write_preview(
            target / f"step_{step.number:0{width}d}{suffix}",
            accumulated, title=caption, colours=step_colours))
    return out


def step_descriptions(plan_result: AssemblyPlan) -> list[str]:
    """A readable line per step, for the interface and the reports.

    Generated from the step records, so a description cannot disagree with the
    structure it describes.
    """
    out = []
    for step in plan_result.steps:
        parts = "、".join(f"{part}×{count}"
                          for part, count in step.added_parts.items())
        where = []
        for index in step.added:
            brick = plan_result.bricks[index]
            anchor = ("放在地面" if brick.z == 0
                      else f"疊在第 {brick.z} 層")
            where.append(f"{brick.part} @ ({brick.x},{brick.y},{brick.z}) "
                         f"{anchor}")
        pieces = "；".join(where)
        out.append(
            f"第 {step.number} 步：加入 {parts}（{pieces}）；"
            f"累積 {step.total_bricks} 塊，目前 {step.components} 個子結構")
    return out
