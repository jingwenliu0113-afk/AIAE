import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bricks import PART_VOCAB, parse_bricks
from src.data.retile import (
    weaker_status,
    PLACEMENT_SHAPES,
    drop_part,
    occupancy_of,
    retile,
    verify,
)


class TestPlacementShapes:
    def test_both_orientations_present(self):
        shapes = {(p, h, w) for p, h, w in PLACEMENT_SHAPES}
        assert ("1x4", 1, 4) in shapes and ("1x4", 4, 1) in shapes

    def test_square_parts_not_duplicated(self):
        assert sum(1 for p, _h, _w in PLACEMENT_SHAPES if p == "1x1") == 1
        assert sum(1 for p, _h, _w in PLACEMENT_SHAPES if p == "2x2") == 1

    def test_count(self):
        # 8 parts, 6 of them asymmetric and so contributing two orientations.
        assert len(PLACEMENT_SHAPES) == 14


class TestRetile:
    def test_empty(self):
        assert retile(set()).bricks == []

    def test_single_cell_needs_1x1(self):
        occ = {(0, 0, 0)}
        assert retile(occ).ok
        assert not retile(occ, allowed=frozenset(PART_VOCAB) - {"1x1"}).ok

    def test_exact_cover_verified(self):
        bricks = parse_bricks("2x4 (0,0,0)\n2x4 (2,0,0)")
        occ = occupancy_of(bricks)
        res = retile(occ)
        assert res.ok
        verify(occ, res.bricks)

    def test_shape_is_preserved(self):
        bricks = parse_bricks("2x6 (0,0,0)\n1x2 (2,0,0)\n2x2 (0,0,1)")
        occ = occupancy_of(bricks)
        res = retile(occ)
        assert occupancy_of(res.bricks) == occ

    def test_budget_is_respected(self):
        occ = occupancy_of(parse_bricks("2x4 (0,0,0)"))
        res = retile(occ, budget={"2x4": 0, "1x1": 0, "1x2": 0, "1x4": 0})
        if res.ok:
            assert all(b.part not in {"2x4", "1x1", "1x2", "1x4"} for b in res.bricks)

    def test_budget_can_make_it_infeasible(self):
        """No 1x1-always-works guarantee once supply is bounded."""
        occ = {(0, 0, 0), (2, 0, 0), (4, 0, 0)}   # three isolated cells
        assert retile(occ).ok                      # unlimited 1x1 is fine
        assert not retile(occ, budget={"1x1": 2}).ok


class TestCounterfactual:
    def test_drop_part_removes_it(self):
        bricks = parse_bricks("2x6 (0,0,0)\n2x6 (0,6,0)")
        res = drop_part(bricks, "2x6")
        assert res.ok
        assert "2x6" not in res.inventory

    def test_drop_part_keeps_the_shape(self):
        bricks = parse_bricks("2x6 (0,0,0)\n2x6 (0,6,0)")
        occ = occupancy_of(bricks)
        res = drop_part(bricks, "2x6")
        assert occupancy_of(res.bricks) == occ
        verify(occ, res.bricks)

    def test_dropping_1x1_is_the_hard_case(self):
        """Measured at 9% feasible vs 100% for every other part."""
        bricks = parse_bricks("1x1 (0,0,0)\n1x2 (2,0,0)")
        assert not drop_part(bricks, "1x1").ok


class TestDeterminism:
    """Reproducibility is a hard requirement for generated datasets.

    CP-SAT's parallel portfolio returns whichever optimal solution a worker
    reaches first, so the same seed yields different layouts run to run. The
    default is therefore single-worker.
    """

    SHAPE = "\n".join(
        f"2x6 ({x},{y},{z})" for z in range(2) for x in (0, 2) for y in (0, 6)
    )

    def test_repeated_runs_are_identical(self):
        occ = occupancy_of(parse_bricks(self.SHAPE))
        runs = [retile(occ, seed=0).bricks for _ in range(4)]
        assert all(r == runs[0] for r in runs)

    def test_default_is_single_worker(self):
        import inspect

        assert inspect.signature(retile).parameters["workers"].default == 1


class TestStaggerIsNotSilentlyDropped:
    """The per-layer solve cannot honour a stagger constraint.

    It was previously ignored there, so an experiment could report itself as
    staggered while running unstaggered.
    """

    SHAPE = "\n".join(f"2x4 (0,0,{z})" for z in range(3))

    def test_per_layer_solve_rejects_stagger(self):
        occ = occupancy_of(parse_bricks(self.SHAPE))
        with pytest.raises(ValueError, match="joint solve"):
            retile(occ, stagger=True)

    def test_joint_solve_accepts_stagger(self):
        occ = occupancy_of(parse_bricks(self.SHAPE))
        assert retile(occ, budget={}, stagger=True).ok

    def test_default_path_unaffected(self):
        occ = occupancy_of(parse_bricks(self.SHAPE))
        assert retile(occ).ok


class TestStatusAggregation:
    """A layered solve is only as strong as its weakest layer.

    Reporting OPTIMAL when one layer merely reached FEASIBLE would claim a
    minimality guarantee the tiling does not have. Exercised by substituting
    the per-layer solver rather than waiting for a real timeout.
    """

    SHAPE = "\n".join(f"2x4 (0,0,{z})" for z in range(3))

    def test_weaker_status_ordering(self):
        assert weaker_status("OPTIMAL", "FEASIBLE") == "FEASIBLE"
        assert weaker_status("FEASIBLE", "OPTIMAL") == "FEASIBLE"
        assert weaker_status("OPTIMAL", "OPTIMAL") == "OPTIMAL"
        assert weaker_status("FEASIBLE", "UNKNOWN") == "UNKNOWN"

    def test_all_optimal_stays_optimal(self):
        occ = occupancy_of(parse_bricks(self.SHAPE))
        assert retile(occ).status == "OPTIMAL"

    def test_one_feasible_layer_downgrades_the_whole(self, monkeypatch):
        import src.data.retile as R

        real = R._solve
        calls = {"n": 0}

        def fake(*a, **kw):
            got, status, t = real(*a, **kw)
            calls["n"] += 1
            if calls["n"] == 2:          # middle layer only
                status = "FEASIBLE"
            return got, status, t

        monkeypatch.setattr(R, "_solve", fake)
        occ = occupancy_of(parse_bricks(self.SHAPE))
        res = R.retile(occ)
        assert calls["n"] == 3
        assert res.ok
        assert res.status == "FEASIBLE"

    def test_infeasible_layer_still_reported(self, monkeypatch):
        import src.data.retile as R

        def fake(*a, **kw):
            return None, "INFEASIBLE", 0.0

        monkeypatch.setattr(R, "_solve", fake)
        occ = occupancy_of(parse_bricks(self.SHAPE))
        res = R.retile(occ)
        assert not res.ok and res.status == "INFEASIBLE"
