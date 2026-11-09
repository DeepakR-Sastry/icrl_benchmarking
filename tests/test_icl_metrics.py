import numpy as np
import pytest

from icrl_bench.eval.icl_metrics import accrual, cmse, nad

GAPS = dict(left_gap=0.05 - 1e-3, right_gap=0.05)  # his HalfCheetah windows


def test_cmse_is_the_mean_squared_difference():
    assert cmse([1, 1, 0], [0.9, 0.6, 0.1]) == pytest.approx(0.06)
    with pytest.raises(AssertionError):
        cmse([1, 1, 0], [0.9, 0.6])


def test_accrual_counts_each_window_then_divides_by_the_max():
    grid = np.array([0.0, 0.1, 0.2])
    # 0.0 and 0.03 -> bin 0; 0.149 and 0.15 -> bin 1 (right edge included);
    # 0.0505 -> no bin (the gap between 0.05 and 0.051); 7.0 -> off the grid.
    coordinates = [0.0, 0.03, 0.149, 0.15, 0.0505, 7.0]
    np.testing.assert_allclose(accrual(coordinates, grid, **GAPS),
                               np.array([2, 2, 0]) / (2 + 1e-6))


def test_left_edge_is_g_minus_0_049():
    grid = np.array([0.0, 0.1])
    out = accrual([0.1 - 0.049, 0.1 - 0.0491], grid, **GAPS)
    np.testing.assert_allclose(out, [0, 1 / (1 + 1e-6)])


def test_nothing_in_the_grid_gives_zeros():
    np.testing.assert_array_equal(accrual([9.0], np.array([0.0, 0.1]), **GAPS), [0, 0])


def test_nad_moves_the_toy_sand():
    # expert 0, 0, 2, 6, 2 and agent 4, 4, 2, 0, 0 (counts; scaled to sum 1
    # inside): 0.4 x 3 + 0.2 x 2 + 0.2 x 3 = 2.2 grid steps.
    assert nad([0, 0, 2, 6, 2], [4, 4, 2, 0, 0]) == pytest.approx(2.2)
    assert nad([1, 2, 3], [2, 4, 6]) == pytest.approx(0)  # same shape, any scale


def test_nad_in_2d_uses_cityblock_steps():
    a = [[1, 0], [0, 0]]
    b = [[0, 0], [0, 1]]  # one row down + one column right
    assert nad(a, b) == pytest.approx(2)
