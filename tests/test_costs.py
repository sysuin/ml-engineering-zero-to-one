"""The cost matrix from Chapter 3: arithmetic a threshold will later depend on."""
from foresight.costs import CostMatrix

M = CostMatrix(value_at_stake=2000, save_rate=0.25, call_hours=1.5, hour_cost=60)


def test_cells_are_net_of_the_call():
    assert M.call_cost == 90
    assert M.cells() == {"tp": 410, "fp": -90, "fn": 0.0, "tn": 0.0}


def test_break_even_is_where_a_call_neither_gains_nor_loses():
    assert abs(M.value_per_call(M.break_even())) < 1e-9


def test_net_value_adds_up_the_cells():
    assert M.net_value(tp=2, fp=8) == 2 * 410 - 8 * 90
