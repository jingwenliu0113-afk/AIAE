"""F-oracle tests: solvable, unsolvable, accounting, and what it may claim.

Everything here is offline and model-free -- the oracle is CP-SAT over a known
shape, so the whole arm can be tested exactly rather than sampled.
"""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bricks import Brick, required_inventory
from src.eval.oracle import (
    ReplayMismatch,
    OracleOutcome,
    OracleTask,
    solve_task,
    status_counts,
    task_signature,
    verify_replay,
    summarise,
)


def slab(h: int, w: int, z: int = 0) -> frozenset:
    """A solid h x w rectangle one unit tall."""
    return frozenset((x, y, z) for x in range(h) for y in range(w))


def task(occ, inventory, *, task_id="t", pair_id="p", role="control",
         variant="exact", reference_bricks=1) -> OracleTask:
    return OracleTask(
        task_id=task_id, pair_id=pair_id, role=role, variant=variant,
        object_id="obj", split="test", occ=occ, inventory=inventory,
        reference_bricks=reference_bricks,
    )


class TestKnownSolvable:
    def test_exact_fit_is_optimal_and_accepted(self):
        """2x4 slab with exactly one 2x4: one placement, nothing to choose."""
        out = solve_task(task(slab(2, 4), {"2x4": 1}))
        assert out.status == "OPTIMAL"
        assert out.accepted
        assert out.n_bricks == 1
        assert out.used == {"2x4": 1}
        assert out.voxel_exact and out.collision_free
        assert out.within_inventory and out.parts_legal
        assert out.failure_reason is None

    def test_surplus_stock_still_uses_the_minimum(self):
        """The objective is brick count, so spare stock must not be spent."""
        out = solve_task(task(slab(2, 4), {"2x4": 5, "1x1": 100, "2x2": 20}))
        assert out.accepted and out.n_bricks == 1
        assert out.used == {"2x4": 1}

    def test_forced_decomposition_when_the_big_part_is_absent(self):
        """Without 2x4 the same slab must be covered by what is left."""
        out = solve_task(task(slab(2, 4), {"2x2": 2, "1x1": 8}))
        assert out.accepted
        assert out.used == {"2x2": 2}, "two 2x2 beat eight 1x1 on count"

    def test_multi_layer_shape(self):
        occ = slab(2, 4, 0) | slab(2, 4, 1)
        out = solve_task(task(occ, {"2x4": 2}))
        assert out.accepted and out.n_bricks == 2
        assert {b for b in out.used} == {"2x4"}


class TestKnownUnsolvable:
    def test_insufficient_stock_is_infeasible(self):
        """Two 2x4s of shape, one 2x4 in stock, nothing else to fall back on."""
        occ = slab(2, 4, 0) | slab(2, 4, 1)
        out = solve_task(task(occ, {"2x4": 1}))
        assert out.status == "INFEASIBLE"
        assert not out.solved and not out.accepted
        assert out.n_bricks is None
        assert out.failure_reason == "infeasible"

    def test_odd_cell_unreachable_without_1x1(self):
        """A single cell no 2+ part can cover: infeasible however much stock."""
        out = solve_task(task(frozenset({(0, 0, 0)}), {"2x4": 99, "2x2": 99}))
        assert out.status == "INFEASIBLE"
        assert not out.accepted

    def test_parity_makes_a_3x3_impossible_without_1x1(self):
        out = solve_task(task(slab(3, 3), {"1x2": 99, "2x2": 99}))
        assert out.status == "INFEASIBLE"
        assert not out.accepted

    def test_empty_stock_is_infeasible(self):
        out = solve_task(task(slab(2, 2), {}))
        assert not out.accepted


class TestInventoryAccounting:
    def test_used_never_exceeds_stock(self):
        stock = {"2x2": 4, "1x1": 16}
        out = solve_task(task(slab(4, 4), stock))
        assert out.accepted
        for part, n in out.used.items():
            assert n <= stock.get(part, 0), f"{part} overdrawn"

    def test_an_unlisted_part_is_owned_in_quantity_zero(self):
        """Regression: retile reads an absent budget key as *unlimited*.

        Left unhandled, a 4x4 slab under {"2x2": 4} came back as two 2x4s --
        a part the inventory never mentioned. For the counterfactual role,
        whose stock deliberately omits the dropped part, that would have let
        the oracle rebuild with exactly the part under test.
        """
        out = solve_task(task(slab(4, 4), {"2x2": 4}))
        assert out.accepted
        assert set(out.used) <= {"2x2"}, f"used an unlisted part: {out.used}"
        assert out.used == {"2x2": 4}

    def test_a_dropped_part_stays_dropped(self):
        """The counterfactual setup: stock lists everything except 1x2."""
        stock = {"1x1": 8, "2x2": 1}
        out = solve_task(task(slab(2, 2) | slab(1, 4, 1), stock))
        assert out.accepted
        assert "1x2" not in out.used
        for part, n in out.used.items():
            assert n <= stock.get(part, 0)

    def test_used_is_recomputed_from_the_bricks(self):
        """The ledger must come from the tiling, not from the solver's word."""
        occ = slab(2, 4)
        out = solve_task(task(occ, {"2x2": 2}))
        assert out.accepted
        assert out.used == {"2x2": 2}
        assert sum(out.used.values()) == out.n_bricks

    def test_a_budget_of_zero_is_not_a_free_pass(self):
        """A part present with quantity 0 must be unusable, not unlimited."""
        out = solve_task(task(slab(2, 4), {"2x4": 0, "2x2": 2}))
        assert out.accepted
        assert "2x4" not in out.used
        assert out.used == {"2x2": 2}


class TestRotationEquivalence:
    def test_rotated_placement_bills_the_canonical_part(self):
        """A 1x4 laid along the other axis is still a 1x4 of stock."""
        out = solve_task(task(slab(4, 1), {"1x4": 1}))
        assert out.accepted
        assert out.used == {"1x4": 1}, "4x1 must not appear as its own entry"

    def test_both_orientations_draw_on_one_quantity(self):
        """An L whose two disjoint arms run along different axes.

        Covering it needs a 1x4 laid one way and a 1x4 laid the other, so it
        needs quantity **2** of the single canonical part -- the orientations
        do not have separate ledgers. One is not enough.
        """
        occ = frozenset(
            [(0, y, 0) for y in range(4)] + [(x, 0, 0) for x in range(1, 5)]
        )
        assert len(occ) == 8, "the arms must not overlap"
        assert solve_task(task(occ, {"1x4": 1})).status == "INFEASIBLE"

        ok = solve_task(task(occ, {"1x4": 2}))
        assert ok.accepted
        assert ok.used == {"1x4": 2}
        assert {b for b in ok.used} == {"1x4"}, "4x1 must not get its own entry"

    def test_square_parts_have_one_spelling(self):
        out = solve_task(task(slab(2, 2), {"2x2": 1}))
        assert out.accepted and out.used == {"2x2": 1}


class TestExactVoxelIdentity:
    def test_accepted_tiling_reproduces_the_shape(self):
        occ = slab(4, 6)
        out = solve_task(task(occ, {"2x4": 3, "2x2": 6, "1x1": 24}))
        assert out.accepted and out.voxel_exact

    def test_verification_catches_a_shape_mismatch(self):
        """_verify must fail a tiling that leaves the occupancy, not trust it."""
        from src.eval.oracle import _verify

        t = task(slab(2, 4), {"2x4": 1})
        escaped = [Brick(h=2, w=4, x=10, y=10, z=0)]
        checks = _verify(t, escaped)
        assert not checks["voxel_exact"]

    def test_verification_catches_a_double_cover(self):
        from src.eval.oracle import _verify

        t = task(slab(2, 4), {"2x4": 2})
        stacked = [Brick(h=2, w=4, x=0, y=0, z=0)] * 2
        checks = _verify(t, stacked)
        assert not checks["collision_free"]

    def test_verification_catches_an_overdraw(self):
        from src.eval.oracle import _verify

        t = task(slab(2, 8), {"2x4": 1})
        checks = _verify(t, [Brick(h=2, w=4, x=0, y=0, z=0),
                             Brick(h=2, w=4, x=0, y=4, z=0)])
        assert not checks["within_inventory"]
        assert checks["over_inventory"] == {"2x4": [2, 1]}


class TestConnectivityIsMeasuredNotRequired:
    """Two separated slabs tile fine and are not one structure.

    The acceptance condition deliberately excludes connectivity; if that ever
    changes, this test should fail loudly rather than the report quietly
    changing meaning.
    """

    def test_disconnected_tiling_is_accepted_but_flagged(self):
        occ = slab(2, 2, 0) | frozenset({(10, 10, 0), (10, 11, 0),
                                         (11, 10, 0), (11, 11, 0)})
        out = solve_task(task(occ, {"2x2": 2}))
        assert out.accepted, "connectivity must not gate acceptance"
        assert not out.connected
        assert out.n_components == 2


class TestStatusAggregation:
    def test_all_four_statuses_are_always_reported(self):
        counts = status_counts([OracleOutcome("a", "OPTIMAL", 0.1, 1, 1)])
        assert counts == {"OPTIMAL": 1, "FEASIBLE": 0,
                          "INFEASIBLE": 0, "UNKNOWN": 0}

    def test_unexpected_status_is_not_dropped(self):
        counts = status_counts([OracleOutcome("a", "MODEL_INVALID", 0.0, 0, None)])
        assert counts["MODEL_INVALID"] == 1

    def test_units_do_not_double_count_shared_geometry(self):
        """Eight rows of one pair are one pair and one geometry, not eight."""
        occ = slab(2, 4)
        tasks, outs = [], []
        for role in ("control", "counterfactual"):
            for variant in ("exact", "loose", "distractor", "mixed"):
                tid = f"p:{role}:{variant}"
                tasks.append(task(occ, {"2x4": 1}, task_id=tid, pair_id="p",
                                  role=role, variant=variant))
                outs.append(OracleOutcome(
                    tid, "OPTIMAL", 0.1, 1, 1, used={"2x4": 1},
                    voxel_exact=True, collision_free=True,
                    within_inventory=True, parts_legal=True, connected=True))
        s = summarise(tasks, outs)
        assert s["units"]["sample"]["n"] == 8
        assert s["units"]["pair"]["n"] == 1
        assert s["units"]["unique_geometry"]["n"] == 1
        # One shape, one inventory, eight rows -> a single distinct problem.
        assert s["units"]["unique_task"]["n"] == 1
        assert s["units"]["unique_geometry"]["multiplicity"] == {8: 1}

    def test_a_group_fails_if_any_row_fails(self):
        occ = slab(2, 4)
        tasks = [task(occ, {"2x4": 1}, task_id=f"t{i}", pair_id="p") for i in range(2)]
        outs = [
            OracleOutcome("t0", "OPTIMAL", 0.1, 1, 1, voxel_exact=True,
                          collision_free=True, within_inventory=True,
                          parts_legal=True, connected=True),
            OracleOutcome("t1", "INFEASIBLE", 0.1, 0, None),
        ]
        s = summarise(tasks, outs)
        assert s["units"]["sample"]["all_accepted"] == 1
        assert s["units"]["pair"]["all_accepted"] == 0, "one bad row fails the pair"

    def test_brick_delta_is_measured_against_the_reference(self):
        occ = slab(2, 4)
        t = task(occ, {"2x4": 1}, reference_bricks=4)
        o = OracleOutcome("t", "OPTIMAL", 0.1, 1, 1, voxel_exact=True,
                          collision_free=True, within_inventory=True,
                          parts_legal=True)
        s = summarise([t], [o])
        assert s["bricks_vs_reference"]["median"] == -3
        assert s["bricks_vs_reference"]["fewer_than_reference"] == 1


class TestInputsAreNotMutated:
    """The oracle reads dataset rows other arms also read."""

    def test_occupancy_and_inventory_survive_a_solve(self):
        occ = slab(4, 4)
        stock = {"2x2": 4, "1x1": 16}
        t = task(occ, stock)
        before_occ, before_stock = set(occ), copy.deepcopy(stock)

        out = solve_task(t)
        assert out.accepted

        assert set(occ) == before_occ, "occupancy was mutated"
        assert stock == before_stock, "caller's inventory dict was mutated"
        assert t.inventory == before_stock, "task inventory was mutated"
        assert set(t.occ) == before_occ

    def test_task_copies_the_sample_inventory(self):
        """from_sample must not alias the row's dict."""
        class FakeSample:
            sample_id, pair_id, role, variant = "s", "p", "control", "exact"
            object_id, split = "o", "test"
            inventory = {"2x4": 2}
            bricks = [Brick(h=2, w=4, x=0, y=0, z=0)]

        s = FakeSample()
        t = OracleTask.from_sample(s)
        t.inventory["2x4"] = 99
        assert s.inventory == {"2x4": 2}, "sample row was aliased"

    def test_repeated_solves_are_identical(self):
        """workers=1 and a fixed seed: same task, same answer."""
        t = task(slab(4, 6), {"2x4": 3, "2x2": 6, "1x1": 24})
        a, b = solve_task(t), solve_task(t)
        assert a.n_bricks == b.n_bricks
        assert a.used == b.used
        assert a.status == b.status


class TestOracleIsLabelledAsSuch:
    """The arm's honesty is a property of the code, not only of the prose."""

    def test_module_states_it_is_not_deployable(self):
        import src.eval.oracle as O

        doc = O.__doc__.lower()
        assert "oracle" in doc
        assert "not a method" in doc or "not a system" in doc
        assert "retrieval" in doc, "must say retrieval is not involved"

    def test_no_retrieval_or_model_imports(self):
        """A retrieval or generation import here would silently change the arm."""
        src = (Path(__file__).resolve().parents[1]
               / "src" / "eval" / "oracle.py").read_text()
        for banned in ("src.retrieval", "src.generation", "transformers",
                       "torch", "peft"):
            assert f"import {banned}" not in src, banned
            assert f"from {banned}" not in src, banned

    def test_solver_settings_are_pinned(self):
        from src.eval.oracle import SEED, TIME_LIMIT, WORKERS

        assert WORKERS == 1, "reproducibility requires a single worker"
        assert SEED == 0 and TIME_LIMIT > 0


class TestTaskSignature:
    """The digest must depend on the problems, not on how they were listed."""

    def _t(self, occ, inv, tid="a", ref=1):
        return task(occ, inv, task_id=tid, reference_bricks=ref)

    def test_order_does_not_change_the_digest(self):
        a = self._t(slab(2, 4), {"2x4": 1}, tid="a")
        b = self._t(slab(2, 2), {"2x2": 1}, tid="b")
        assert task_signature([a, b]) == task_signature([b, a])

    def test_shape_change_changes_the_digest(self):
        a = self._t(slab(2, 4), {"2x4": 1})
        b = self._t(slab(2, 2), {"2x4": 1})
        assert task_signature([a]) != task_signature([b])

    def test_inventory_change_changes_the_digest(self):
        a = self._t(slab(2, 4), {"2x4": 1})
        b = self._t(slab(2, 4), {"2x4": 2})
        assert task_signature([a]) != task_signature([b])

    def test_reference_brick_count_change_changes_the_digest(self):
        a = self._t(slab(2, 4), {"2x4": 1}, ref=1)
        b = self._t(slab(2, 4), {"2x4": 1}, ref=2)
        assert task_signature([a]) != task_signature([b])

    def test_task_id_change_changes_the_digest(self):
        a = self._t(slab(2, 4), {"2x4": 1}, tid="a")
        b = self._t(slab(2, 4), {"2x4": 1}, tid="b")
        assert task_signature([a]) != task_signature([b])

    def test_adding_a_task_changes_the_digest(self):
        a = self._t(slab(2, 4), {"2x4": 1}, tid="a")
        b = self._t(slab(2, 2), {"2x2": 1}, tid="b")
        assert task_signature([a]) != task_signature([a, b])


class TestReplayRefusesDrift:
    """Re-rendering must never describe one run with another run's numbers.

    Each test moves exactly one thing and asserts the replay guard stops.
    """

    def env(self, **over):
        base = {
            "source_sha256": "a" * 64,
            "task_signature": "b" * 64,
            "seed": 0,
            "time_limit_seconds": 10.0,
            "workers": 1,
            "python": "3.13.9",
            "ortools": "9.15.6755",
            "platform": "macOS-26.6.1-arm64-arm-64bit-Mach-O",
            "machine": "arm64",
        }
        base.update(over)
        return base

    def test_matching_run_is_allowed(self):
        verify_replay(self.env(), self.env(), ["t1", "t2"], ["t1", "t2"])

    def test_source_file_change_is_refused(self):
        """A different counterfactual file: same ids, different data."""
        with pytest.raises(ReplayMismatch, match="source_sha256"):
            verify_replay(self.env(), self.env(source_sha256="c" * 64),
                          ["t1"], ["t1"])

    def test_shape_or_inventory_change_is_refused(self):
        """Caught by the task digest even when the file digest is unchanged."""
        with pytest.raises(ReplayMismatch, match="task_signature"):
            verify_replay(self.env(), self.env(task_signature="d" * 64),
                          ["t1"], ["t1"])

    def test_a_removed_task_is_refused(self):
        with pytest.raises(ReplayMismatch, match="absent now"):
            verify_replay(self.env(), self.env(), ["t1", "t2"], ["t1"])

    def test_an_added_task_is_refused(self):
        with pytest.raises(ReplayMismatch, match="absent from the stored run"):
            verify_replay(self.env(), self.env(), ["t1"], ["t1", "t2"])

    @pytest.mark.parametrize("key,value", [
        ("seed", 1), ("time_limit_seconds", 30.0), ("workers", 8),
    ])
    def test_solver_setting_change_is_refused(self, key, value):
        with pytest.raises(ReplayMismatch, match=key):
            verify_replay(self.env(), self.env(**{key: value}), ["t1"], ["t1"])

    def test_a_run_predating_the_check_is_refused(self):
        """No digests stored at all must fail closed, not be assumed fine."""
        old = {"seed": 0, "time_limit_seconds": 10.0, "workers": 1}
        with pytest.raises(ReplayMismatch, match="predates this check"):
            verify_replay(old, self.env(), ["t1"], ["t1"])

    def test_every_problem_is_reported_not_just_the_first(self):
        with pytest.raises(ReplayMismatch) as e:
            verify_replay(self.env(), self.env(seed=9, workers=4),
                          ["t1", "t2"], ["t1", "t3"])
        msg = str(e.value)
        assert "seed" in msg and "workers" in msg
        assert "absent now" in msg and "absent from the stored run" in msg


class TestConnectivityDenominators:
    """The three connectivity figures must not collapse into one another."""

    def make(self, specs):
        """specs: list of (accepted, connected, status)."""
        tasks, outs = [], []
        for i, (acc, conn, status) in enumerate(specs):
            tid = f"t{i}"
            tasks.append(task(slab(2, 4), {"2x4": 1}, task_id=tid, pair_id="p"))
            outs.append(OracleOutcome(
                tid, status, 0.1, 1, 1 if acc else None,
                voxel_exact=acc, collision_free=acc,
                within_inventory=acc, parts_legal=acc, connected=conn))
        return summarise(tasks, outs)

    def test_yield_counts_failures_against_it_but_rate_does_not(self):
        # 4 tasks: 2 accepted+connected, 1 accepted+disconnected, 1 failed.
        s = self.make([(True, True, "OPTIMAL"), (True, True, "OPTIMAL"),
                       (True, False, "OPTIMAL"), (False, False, "UNKNOWN")])
        c = s["connectivity"]
        assert c["solved_and_connected"] == 2
        assert c["tasks"] == 4
        assert c["solved_and_connected_yield"] == 0.5      # over everything
        assert c["accepted"] == 3
        assert c["connected_among_accepted_rate"] == pytest.approx(2 / 3)
        assert c["solved_and_connected_yield"] != c["connected_among_accepted_rate"]

    def test_a_failed_task_is_never_counted_as_connected(self):
        """connected=True on an unaccepted run must not reach the numerator."""
        s = self.make([(False, True, "UNKNOWN")])
        assert s["connectivity"]["solved_and_connected"] == 0

    def test_optimal_slice_excludes_feasible(self):
        s = self.make([(True, True, "OPTIMAL"), (True, False, "OPTIMAL"),
                       (True, True, "FEASIBLE")])
        opt = s["connectivity"]["optimal"]
        assert opt["n"] == 2, "FEASIBLE is not a proved minimum"
        assert opt["connected"] == 1
        assert opt["proven_minimum_but_disconnected"] == 1
        assert s["connectivity"]["feasible_not_minimum"] == {"n": 1, "connected": 1}

    def test_group_yield_and_conditional_rate_differ(self):
        """A geometry with one failed task sinks the yield, not the rate."""
        occ_a, occ_b = slab(2, 4), slab(2, 2)
        tasks, outs = [], []
        for i, (occ, acc, conn) in enumerate([
            (occ_a, True, True), (occ_a, False, False),   # geometry A: one failure
            (occ_b, True, True), (occ_b, True, True),     # geometry B: clean
        ]):
            tid = f"t{i}"
            tasks.append(OracleTask(
                task_id=tid, pair_id="p", role="control", variant="exact",
                object_id="o", split="test", occ=occ, inventory={"2x4": 1},
                reference_bricks=1))
            outs.append(OracleOutcome(
                tid, "OPTIMAL" if acc else "UNKNOWN", 0.1, 1, 1 if acc else None,
                voxel_exact=acc, collision_free=acc, within_inventory=acc,
                parts_legal=acc, connected=conn))
        g = summarise(tasks, outs)["units"]["unique_geometry"]
        assert g["n"] == 2
        assert g["all_accepted"] == 1                       # only B
        assert g["all_solved_and_connected"] == 1           # only B
        assert g["all_solved_and_connected_yield"] == 0.5   # over both
        assert g["all_connected_given_all_accepted_denominator"] == 1
        assert g["all_connected_given_all_accepted_rate"] == 1.0  # over B alone


class TestReplayRefusesADifferentToolchain:
    """Old numbers must not be relabelled with today's versions.

    CP-SAT is deterministic for a fixed seed and worker count within a
    version; across versions the search differs, so a replay under a different
    ortools or interpreter would be describing one environment's results with
    another's label. The bug this guards was real: the renderer rebuilt the
    environment block from the current process, so re-rendering silently
    restamped a stored run with whatever was installed today.
    """

    def env(self, **over):
        base = {
            "source_sha256": "a" * 64,
            "task_signature": "b" * 64,
            "seed": 0,
            "time_limit_seconds": 10.0,
            "workers": 1,
            "python": "3.13.9",
            "ortools": "9.15.6755",
            "platform": "macOS-26.6.1-arm64-arm-64bit-Mach-O",
            "machine": "arm64",
        }
        base.update(over)
        return base

    @pytest.mark.parametrize("key,value", [
        ("python", "3.14.0"),
        ("ortools", "9.16.0"),
        ("platform", "Linux-6.1.0-x86_64"),
        ("machine", "x86_64"),
    ])
    def test_environment_change_is_refused(self, key, value):
        with pytest.raises(ReplayMismatch, match=key):
            verify_replay(self.env(), self.env(**{key: value}), ["t1"], ["t1"])

    def test_a_run_recording_no_versions_is_refused(self):
        """Fail closed: a run from before versions were recorded cannot pass."""
        old = {k: v for k, v in self.env().items()
               if k not in ("python", "ortools")}
        with pytest.raises(ReplayMismatch, match="predates this check"):
            verify_replay(old, self.env(), ["t1"], ["t1"])

    def test_every_environment_key_is_actually_compared(self):
        """Guards the list itself: adding a key to the env is not enough."""
        from src.eval.oracle import REPLAY_KEYS

        for key in ("python", "ortools", "platform", "machine"):
            assert key in REPLAY_KEYS, key


class TestReplayTaskListIntegrity:
    """The stored runs are a list; a set would hide real problems."""

    def env(self, **over):
        base = {
            "source_sha256": "a" * 64, "task_signature": "b" * 64,
            "seed": 0, "time_limit_seconds": 10.0, "workers": 1,
            "python": "3.13.9", "ortools": "9.15.6755",
            "platform": "mac", "machine": "arm64",
        }
        base.update(over)
        return base

    def test_duplicate_task_id_in_the_stored_run_is_refused(self):
        """Two results for one task: which one counts is undefined."""
        with pytest.raises(ReplayMismatch, match="duplicated task id"):
            verify_replay(self.env(), self.env(),
                          ["t1", "t1", "t2"], ["t1", "t2"])

    def test_duplicates_are_refused_even_when_the_id_sets_match(self):
        """A set comparison would call this aligned; it is not."""
        with pytest.raises(ReplayMismatch) as e:
            verify_replay(self.env(), self.env(), ["t1", "t2", "t2"],
                          ["t1", "t2"])
        assert "duplicated task id" in str(e.value)
        assert set(["t1", "t2", "t2"]) == set(["t1", "t2"]), "premise"

    def test_row_count_must_match_exactly(self):
        with pytest.raises(ReplayMismatch, match="3 rows but there are 2 tasks"):
            verify_replay(self.env(), self.env(),
                          ["t1", "t2", "t3"], ["t1", "t2"])

    def test_the_duplicated_id_is_named(self):
        with pytest.raises(ReplayMismatch, match="'t2'"):
            verify_replay(self.env(), self.env(), ["t1", "t2", "t2"],
                          ["t1", "t2"])

    def test_exact_one_to_one_lists_pass(self):
        verify_replay(self.env(), self.env(), ["t2", "t1"], ["t1", "t2"])
