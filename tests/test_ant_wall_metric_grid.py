# Gaurav's AntWall grid = his HalfCheetah one (same branch of his code); the
# same three checks as tests/test_half_cheetah_metric_grid.py on Ant's sizes.
import numpy as np

from icrl_bench.envs.ant_wall import AntWall, metric_grid, true_cost
from icrl_bench.featurizers.ant_wall import featurize_x


def test_grid_is_his_101_points_with_his_windows():
    grid = metric_grid()
    np.testing.assert_array_equal(grid.values, np.arange(-5, 5 + 0.1, 0.1))
    assert len(grid.values) == 101 and len(grid.trajectory.steps) == 101
    assert (grid.left_gap, grid.right_gap) == (0.05 - 1e-3, 0.05)


def test_steps_have_the_env_shapes_and_only_x_set():
    env = AntWall()
    grid = metric_grid()
    step = grid.trajectory.steps[10]
    assert step.state.shape == env.observation_space.shape
    assert step.action.shape == env.action_space.shape
    assert step.state[0] == grid.values[10] and not step.state[1:].any()
    assert grid.coordinate(step.state) == grid.values[10]


def test_true_cost_and_featurizer_see_his_inputs():
    grid = metric_grid()
    true = [true_cost(s.state, s.action, wall_x=-1) for s in grid.trajectory.steps]
    # his manual cost s[0] <= -1 on the same floats
    assert true == [float(x <= -1) for x in grid.values]
    np.testing.assert_array_equal(featurize_x(grid.trajectory)[:, 0],
                                  grid.values.astype(np.float32))
