"""F-oracle: the tiling upper bound when the target shape is already known.

**This is an oracle, not a method.** It is handed the reference voxel shape of
a *test* structure and asked to tile it within a given inventory. Nothing
predicts the shape: no retrieval, no index, no generation, no model of any
kind. A deployable system has to obtain the shape from the caption, and that
step is exactly what is deleted here.

So the numbers below bound what the optimisation stage can achieve *given a
perfect shape oracle*. They are not a product comparison against arms A-E, and
quoting them beside an end-to-end success rate without the word "oracle"
attached would overstate the pipeline by whatever the shape-acquisition step
costs. F-pipeline is the arm that pays that cost; the F-oracle minus
F-pipeline gap is the intended reading of this number, and F-pipeline is not
built yet.

Using a test-split reference shape is deliberate and is *not* a leak in the
usual sense -- there is no training here and no model to contaminate. It is
the definition of the oracle. It does mean this arm may never be described as
"our system", and the report says so in its own words.

What the oracle is graded on
----------------------------

A run is **accepted** when the tiling reproduces the reference shape exactly,
places nothing overlapping, draws nothing it does not have, and uses only
canonical vocabulary parts:

* ``voxel_exact``   -- the union of placed cells equals the reference occupancy
* ``collision_free``-- no cell is covered twice
* ``within_inventory`` -- no canonical part exceeds its quantity
* ``parts_legal``   -- every part is in ``PART_VOCAB``

**Stud connectivity is measured but is not an acceptance condition.** The
CP-SAT model minimises brick count subject to exact cover and the part
budget; it has no connectivity constraint, so a minimal tiling of a connected
shape can come apart into pieces. Counting a disconnected tiling as a failure
would report a formulation limit as an infeasibility, and counting it as a
success without saying so would claim buildability the result does not have.
It is therefore reported as its own rate, alongside the acceptance rate and
never folded into it. Connectivity-aware tiling is future work (section 9.7).

Feasibility is 100% by construction
-----------------------------------

Each task pairs a shape with an inventory that the reference build already
fits inside -- every variant's stock is a superset of what that build used.
The reference tiling is therefore a witness, and a complete solver must find
*something*. A feasibility rate near 100% is the setup restating itself, not
a result, and the report says this next to the number. What the run does
measure is the cost and the shape of the solutions: solve time, whether the
proven minimum matches the reference brick count, and how often that minimum
is disconnected.

The oracle cannot come in *below* the reference. The reference tiling was
produced by this same minimum-count objective on the same shape with strictly
more freedom -- no part budget -- so it is already minimal, and adding a
budget can only match it or fail to reach it. Matching is the ceiling.

Naming the connectivity numbers
-------------------------------

Three different quantities are easy to blur into one "connected rate", so
each is named for its denominator and none is reported bare:

* **solved-and-connected yield** -- connected over *all tasks attempted*.
  Failures count against it, so it is an end-to-end yield, not a property of
  the tilings.
* **connectivity among successes** -- connected over *accepted* tilings.
  This is the conditional rate, and the only one that describes the solutions.
* per geometry, the same split again: the **all-tasks-solved-and-connected
  yield** is over every geometry, while the conditional rate is over only
  those geometries whose tasks all succeeded.

Only ``OPTIMAL`` runs may be described as minimum-brick tilings. ``FEASIBLE``
means the search was cut off before proving a minimum, so a connectivity
figure over both statuses together does not say anything about minima.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.data.bricks import (
    PART_VOCAB,
    Brick,
    connected_components,
    find_collisions,
    is_connected,
    required_inventory,
)
from src.data.retile import occupancy_of, retile

Cell = tuple[int, int, int]

#: Solver settings. Pinned here rather than passed around so every run in the
#: report shares them and the report can state one set of numbers.
TIME_LIMIT = 10.0
SEED = 0
#: Required, not preferred: a parallel portfolio returns whichever equally
#: optimal tiling finished first, so the same seed gives different layouts run
#: to run. See src/data/retile.py.
WORKERS = 1


@dataclass(frozen=True)
class OracleTask:
    """One (reference shape, inventory) problem.

    ``occ`` is frozen and ``inventory`` is copied on construction: the oracle
    is handed dataset rows and must not disturb them, since the same rows are
    read again by other arms.
    """

    task_id: str
    pair_id: str
    role: str
    variant: str
    object_id: str
    split: str
    occ: frozenset[Cell]
    inventory: dict[str, int]
    reference_bricks: int

    @classmethod
    def from_sample(cls, s) -> "OracleTask":
        return cls(
            task_id=s.sample_id,
            pair_id=s.pair_id,
            role=s.role,
            variant=s.variant,
            object_id=s.object_id,
            split=s.split,
            occ=frozenset(occupancy_of(s.bricks)),
            inventory=dict(s.inventory),
            reference_bricks=len(s.bricks),
        )

    @property
    def geometry_key(self) -> frozenset[Cell]:
        """Identifies the shape. Both roles of a pair share one."""
        return self.occ

    @property
    def task_key(self) -> tuple:
        """Identifies the problem: shape plus stock."""
        return (self.occ, tuple(sorted(self.inventory.items())))


@dataclass
class OracleOutcome:
    task_id: str
    status: str
    wall_seconds: float
    candidates: int
    n_bricks: int | None
    used: dict[str, int] = field(default_factory=dict)

    # Acceptance checks, each recorded separately so a failure names itself.
    voxel_exact: bool = False
    collision_free: bool = False
    within_inventory: bool = False
    parts_legal: bool = False

    # Measured, deliberately outside the acceptance condition.
    connected: bool = False
    n_components: int | None = None

    over_inventory: dict[str, list[int]] = field(default_factory=dict)
    illegal_parts: list[str] = field(default_factory=list)
    failure_reason: str | None = None

    @property
    def solved(self) -> bool:
        """The solver returned a tiling. Says nothing about whether it is valid."""
        return self.n_bricks is not None

    @property
    def accepted(self) -> bool:
        return (
            self.solved
            and self.voxel_exact
            and self.collision_free
            and self.within_inventory
            and self.parts_legal
        )

    @property
    def solved_and_connected(self) -> bool:
        """Both, explicitly. ``connected`` alone would count a tiling that
        failed verification, and defaults to False on a failure, which makes
        it silently readable as either a yield or a rate."""
        return self.accepted and self.connected


def _verify(task: OracleTask, bricks: list[Brick]) -> dict:
    """Re-derive every acceptance property from the returned bricks.

    Independent of the solver's own claims: the tiling is re-parsed into cells
    and counted from scratch, so a bug in the model or in candidate generation
    shows up here rather than being certified by the thing that produced it.
    """
    cells = [c for b in bricks for c in b.cells]
    used = required_inventory(bricks)
    over = {
        p: [n, task.inventory.get(p, 0)]
        for p, n in used.items()
        if n > task.inventory.get(p, 0)
    }
    illegal = sorted({b.part for b in bricks if b.part not in PART_VOCAB})
    return {
        "used": dict(sorted(used.items())),
        "voxel_exact": set(cells) == set(task.occ) and len(cells) == len(task.occ),
        "collision_free": not find_collisions(bricks),
        "within_inventory": not over,
        "parts_legal": not illegal,
        "over_inventory": over,
        "illegal_parts": illegal,
        "connected": is_connected(bricks),
        "n_components": len(connected_components(bricks)),
    }


def solve_task(
    task: OracleTask,
    *,
    time_limit: float = TIME_LIMIT,
    seed: int = SEED,
    workers: int = WORKERS,
) -> OracleOutcome:
    """Tile ``task.occ`` within ``task.inventory``, then verify the result.

    ``allowed`` is derived from the inventory rather than left to default.
    ``retile`` treats a part *absent* from its budget dict as unlimited, which
    is right for the counterfactual generator (where an unbudgeted solve is
    the point) and wrong here: an inventory is a closed statement, and a part
    it does not list is a part you do not own. Without this the counterfactual
    role would be free to rebuild with the very part its inventory dropped.
    Deriving both arguments from one dict is what keeps them agreeing.
    """
    stocked = frozenset(p for p, n in task.inventory.items() if n > 0)
    res = retile(
        set(task.occ),                 # a copy; retile must not see the frozen set
        allowed=stocked,
        budget=dict(task.inventory),   # ditto, and retile never writes to it
        time_limit=time_limit,
        seed=seed,
        workers=workers,
    )
    if not res.ok:
        return OracleOutcome(
            task_id=task.task_id,
            status=res.status,
            wall_seconds=res.wall_seconds,
            candidates=res.candidates,
            n_bricks=None,
            failure_reason=(
                "timeout" if res.status == "UNKNOWN" else res.status.lower()
            ),
        )

    checks = _verify(task, res.bricks)
    out = OracleOutcome(
        task_id=task.task_id,
        status=res.status,
        wall_seconds=res.wall_seconds,
        candidates=res.candidates,
        n_bricks=len(res.bricks),
        **checks,
    )
    if not out.accepted:
        broke = [
            k for k in ("voxel_exact", "collision_free", "within_inventory",
                        "parts_legal")
            if not getattr(out, k)
        ]
        out.failure_reason = "verification: " + ", ".join(broke)
    return out


class ReplayMismatch(RuntimeError):
    """The stored run does not describe the inputs and settings at hand."""


#: Everything that would change the numbers. Re-rendering a report is only
#: honest if the stored run was produced from these exact inputs, so all of
#: them are compared and any difference is fatal rather than a warning.
#:
#: The toolchain is in here, not just the inputs. CP-SAT is deterministic for
#: a fixed seed and worker count *within a version*; across versions the search
#: changes, so a different ortools can return a different (equally optimal)
#: tiling, and a different interpreter is a different build of it. Replaying
#: under either would attach one environment's numbers to another's label,
#: which is exactly what a provenance record exists to prevent.
REPLAY_KEYS = (
    "source_sha256",
    "task_signature",
    "seed",
    "time_limit_seconds",
    "workers",
    "python",
    "ortools",
    "platform",
    "machine",
)


def task_signature(tasks: list[OracleTask]) -> str:
    """A digest of the problems themselves, not of the file they came from.

    Covers task id, voxel occupancy, inventory and reference brick count --
    every input the solver sees. A file digest alone would miss a change made
    in memory, and an id-set check alone would miss a shape or a quantity
    moving underneath an unchanged id. Sorted throughout so the digest depends
    on content and not on iteration order.
    """
    h = hashlib.sha256()
    for t in sorted(tasks, key=lambda t: t.task_id):
        h.update(t.task_id.encode())
        h.update(b"|")
        for c in sorted(t.occ):
            h.update(b"%d,%d,%d;" % c)
        h.update(b"|")
        for p, n in sorted(t.inventory.items()):
            h.update(f"{p}={n};".encode())
        h.update(b"|%d\n" % t.reference_bricks)
    return h.hexdigest()


def verify_replay(
    saved_env: dict,
    current_env: dict,
    saved_task_ids: "Sequence[str]",
    current_task_ids: "Sequence[str]",
) -> None:
    """Refuse to re-render a stored run against different inputs or toolchain.

    Re-rendering exists so prose can be corrected without repeating a 45
    minute solve. That is only safe if the numbers still belong to the inputs
    being described; silently rendering yesterday's results over today's data
    is the exact failure it would invite. Raises on the first sign of drift.

    ``saved_task_ids`` is taken as a *sequence*, not a set, because the stored
    runs are a list and a set would quietly absorb duplicates -- a stored file
    with the same task twice would then look aligned while one of the two
    results silently won. Length is compared before identity for the same
    reason.
    """
    problems: list[str] = []

    dupes = sorted(t for t, n in Counter(saved_task_ids).items() if n > 1)
    if dupes:
        problems.append(
            f"stored run has {len(dupes)} duplicated task id(s) "
            f"(e.g. {dupes[0]!r}); results are ambiguous")

    if len(saved_task_ids) != len(current_task_ids):
        problems.append(
            f"stored run holds {len(saved_task_ids)} rows but there are "
            f"{len(current_task_ids)} tasks now")

    saved, current = set(saved_task_ids), set(current_task_ids)
    missing, extra = saved - current, current - saved
    if missing:
        problems.append(
            f"{len(missing)} task(s) in the stored run are absent now "
            f"(e.g. {sorted(missing)[0]})")
    if extra:
        problems.append(
            f"{len(extra)} task(s) present now are absent from the stored run "
            f"(e.g. {sorted(extra)[0]})")

    for key in REPLAY_KEYS:
        was, now = saved_env.get(key), current_env.get(key)
        if was is None:
            problems.append(f"stored run records no {key}; it predates this check")
        elif was != now:
            problems.append(f"{key} changed: stored {was!r}, now {now!r}")

    if problems:
        raise ReplayMismatch(
            "cannot re-render the stored run:\n  - "
            + "\n  - ".join(problems)
            + "\nRe-run the solver instead of replaying."
        )


def status_counts(outcomes: list[OracleOutcome]) -> dict[str, int]:
    """Solver statuses, always reporting all four so a zero is visible."""
    c = Counter(o.status for o in outcomes)
    known = ("OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN")
    out = {k: c.get(k, 0) for k in known}
    for k, n in c.items():           # anything unexpected must not vanish
        if k not in out:
            out[k] = n
    return out


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def _connectivity(outcomes: list[OracleOutcome]) -> dict:
    """Connectivity split by denominator, and by whether a minimum was proved.

    Every figure here carries the population it is over. A bare "connected
    rate" would be read as a property of the tilings when the same count over
    all attempted tasks is really an end-to-end yield, and the two differ by
    the failures.

    The ``OPTIMAL`` slice is separated because only a proved optimum is a
    minimum-brick tiling; a ``FEASIBLE`` result is whatever the search had
    when time ran out, so mixing the two would attach the word "minimum" to
    results that never earned it.
    """
    accepted = [o for o in outcomes if o.accepted]
    optimal = [o for o in outcomes if o.status == "OPTIMAL"]
    feasible = [o for o in outcomes if o.status == "FEASIBLE"]
    conn = [o for o in outcomes if o.solved_and_connected]
    opt_conn = [o for o in optimal if o.solved_and_connected]
    return {
        # Over every task attempted: failures count against it.
        "solved_and_connected": len(conn),
        "tasks": len(outcomes),
        "solved_and_connected_yield": _rate(len(conn), len(outcomes)),
        # Over accepted tilings only: the conditional rate.
        "connected_among_accepted": len(conn),
        "accepted": len(accepted),
        "connected_among_accepted_rate": _rate(len(conn), len(accepted)),
        # Only these are minimum-brick tilings.
        "optimal": {
            "n": len(optimal),
            "connected": len(opt_conn),
            "connected_rate": _rate(len(opt_conn), len(optimal)),
            "proven_minimum_but_disconnected": len(optimal) - len(opt_conn),
        },
        "feasible_not_minimum": {
            "n": len(feasible),
            "connected": sum(1 for o in feasible if o.solved_and_connected),
        },
    }


def summarise(
    tasks: list[OracleTask], outcomes: list[OracleOutcome]
) -> dict:
    """Aggregate at four units, because they answer different questions.

    A test split holds four inventory framings x two roles per pair, and the
    two roles of a pair are two tilings of *one* shape. Reporting a per-row
    mean therefore weights each shape eight times and each duplicated geometry
    more than that. Every level is reported rather than one being chosen:

    ``sample``          every (shape, inventory) row as it appears in the data
    ``unique_task``     distinct (shape, inventory) problems
    ``pair``            the counterfactual pair, accepted only if all 8 rows are
    ``unique_geometry`` distinct voxel shapes, accepted only if all rows are

    The headline is ``unique_task``: it is the finest unit that is not a
    duplicate of another unit.
    """
    by_id = {o.task_id: o for o in outcomes}
    assert len(by_id) == len(outcomes), "duplicate task_id in outcomes"

    def group(keyfn) -> dict:
        buckets: dict = {}
        for t in tasks:
            buckets.setdefault(keyfn(t), []).append(by_id[t.task_id])
        n = len(buckets)
        # A group counts only if *every* row in it does.
        all_acc = [g for g in buckets.values() if all(o.accepted for o in g)]
        all_conn = [g for g in buckets.values()
                    if all(o.solved_and_connected for o in g)]
        return {
            "n": n,
            "all_accepted": len(all_acc),
            "all_accepted_rate": _rate(len(all_acc), n),
            # Yield: over every group, so failures count against it.
            "all_solved_and_connected": len(all_conn),
            "all_solved_and_connected_yield": _rate(len(all_conn), n),
            # Conditional: over only the groups that fully succeeded. This is
            # the one that describes the tilings rather than the run.
            "all_connected_given_all_accepted": len(all_conn),
            "all_connected_given_all_accepted_denominator": len(all_acc),
            "all_connected_given_all_accepted_rate": _rate(len(all_conn), len(all_acc)),
            "multiplicity": dict(sorted(Counter(len(g) for g in buckets.values()).items())),
        }

    solved = [o for o in outcomes if o.solved]
    accepted = [o for o in outcomes if o.accepted]
    ref = {t.task_id: t.reference_bricks for t in tasks}
    deltas = sorted(o.n_bricks - ref[o.task_id] for o in accepted)
    times = sorted(o.wall_seconds for o in outcomes)

    def pct(xs: list, q: float):
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None

    return {
        "units": {
            "sample": group(lambda t: t.task_id),
            "unique_task": group(lambda t: t.task_key),
            "pair": group(lambda t: t.pair_id),
            "unique_geometry": group(lambda t: t.geometry_key),
        },
        "status": status_counts(outcomes),
        "solved": len(solved),
        "accepted": len(accepted),
        "connectivity": _connectivity(outcomes),
        "failures": dict(sorted(Counter(
            o.failure_reason for o in outcomes if o.failure_reason
        ).items())),
        "bricks_vs_reference": {
            "n": len(deltas),
            "min": deltas[0] if deltas else None,
            "median": pct(deltas, .5),
            "p95": pct(deltas, .95),
            "max": deltas[-1] if deltas else None,
            "fewer_than_reference": sum(1 for d in deltas if d < 0),
            "equal_to_reference": sum(1 for d in deltas if d == 0),
            "more_than_reference": sum(1 for d in deltas if d > 0),
        },
        "solve_seconds": {
            "total": round(sum(times), 1),
            "median": round(pct(times, .5), 3) if times else None,
            "p95": round(pct(times, .95), 3) if times else None,
            "max": round(times[-1], 3) if times else None,
        },
        "parts_used": dict(sorted(
            sum((Counter(o.used) for o in accepted), Counter()).items())),
    }
