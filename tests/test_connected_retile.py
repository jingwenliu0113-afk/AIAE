"""The delivery CP-SAT model enforces the connectivity it reports."""

from src.data.bricks import is_connected, parse_bricks, required_inventory
from src.data.retile import occupancy_of, verify
from src.delivery.connected_retile import retile_connected


STACK = "2x4 (0,0,0)\n2x4 (0,0,1)"


def test_a_connected_exact_cover_is_found_and_independently_verifies():
    target = occupancy_of(parse_bricks(STACK))
    result = retile_connected(target, budget={"2x4": 2})
    assert result.status == "OPTIMAL"
    assert result.bricks is not None
    verify(target, result.bricks)
    assert is_connected(result.bricks)
    assert required_inventory(result.bricks)["2x4"] <= 2


def test_a_shape_with_no_possible_stud_path_is_infeasible():
    target = {(0, 0, 0), (2, 0, 0)}
    result = retile_connected(target, budget={"1x1": 2})
    assert result.status == "INFEASIBLE"
    assert result.bricks is None


def test_an_empty_shape_keeps_the_historical_empty_result_contract():
    result = retile_connected(set())
    assert result.status == "OPTIMAL"
    assert result.bricks == []


def test_repeated_optimal_runs_are_identical():
    target = occupancy_of(parse_bricks(STACK))
    runs = [retile_connected(target, budget={}, seed=7, time_limit=30.0)
            for _ in range(4)]
    assert {result.status for result in runs} == {"OPTIMAL"}
    assert all(result.bricks == runs[0].bricks for result in runs)

