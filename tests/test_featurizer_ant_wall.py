import numpy as np

from icrl_bench.core.types import Step, Trajectory
from icrl_bench.featurizers.ant_wall import featurize, featurize_raw, featurize_x


def two_steps():
    # obs: x = -25 then 30 (outside +-20), the rest small; actions inside
    # and outside [-1, 1].
    obs = np.zeros((2, 113))
    obs[:, 0] = [-25.0, 30.0]
    obs[:, 5] = [0.5, -0.5]
    actions = np.array([[0.5] * 8, [-3.0] * 8])
    return Trajectory([Step(o, a, None) for o, a in zip(obs, actions)])


def test_clipped_observation_then_clipped_action():
    rows = featurize(two_steps())
    assert rows.shape == (2, 121) and rows.dtype == np.float32
    assert rows[:, 0].tolist() == [-20.0, 20.0] and rows[:, 5].tolist() == [0.5, -0.5]
    assert rows[0, 113:].tolist() == [0.5] * 8 and rows[1, 113:].tolist() == [-1.0] * 8


def test_raw_keeps_everything():
    rows = featurize_raw(two_steps())
    assert rows.shape == (2, 121)
    assert rows[:, 0].tolist() == [-25.0, 30.0] and rows[1, 113:].tolist() == [-3.0] * 8


def test_x_only():
    rows = featurize_x(two_steps())
    assert rows.shape == (2, 1) and rows[:, 0].tolist() == [-25.0, 30.0]
