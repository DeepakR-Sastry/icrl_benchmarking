import numpy as np

from icrl_bench.core.types import Step, Trajectory
from icrl_bench.featurizers.half_cheetah_with_pos import featurize, featurize_raw, featurize_x


def test_rows_are_clipped_obs_then_clipped_action():
    obs = np.zeros(18)
    obs[0], obs[15] = 103.0, -25.0         # x far ahead; a fast joint velocity
    action = np.array([0.5, 1.5, -2.0, 0.0, -0.3, 1.0])
    small_obs = np.full(18, 0.1)
    trajectory = Trajectory([Step(obs, action, 0.0), Step(small_obs, np.zeros(6), 0.0)])

    features = featurize(trajectory)

    assert features.shape == (2, 24) and features.dtype == np.float32
    assert features[0, 0] == 20.0 and features[0, 15] == -20.0     # obs clipped to +-20
    np.testing.assert_array_equal(features[0, 18:],
                                  np.float32([0.5, 1.0, -1.0, 0.0, -0.3, 1.0]))  # action clipped
    np.testing.assert_allclose(features[1, :18], 0.1, rtol=1e-6)    # in range: unchanged
    np.testing.assert_array_equal(features[1, 18:], 0.0)


def test_raw_rows_are_obs_then_action_unclipped():
    obs = np.zeros(18)
    obs[0], obs[15] = 103.0, -25.0
    action = np.array([0.5, 1.5, -2.0, 0.0, -0.3, 1.0])
    trajectory = Trajectory([Step(obs, action, 0.0)])

    features = featurize_raw(trajectory)

    assert features.shape == (1, 24) and features.dtype == np.float32
    assert features[0, 0] == 103.0 and features[0, 15] == -25.0     # not clipped
    np.testing.assert_array_equal(features[0, 18:], np.float32(action))
    # Same columns as the clipped featurizer where nothing is out of range.
    small = Trajectory([Step(np.full(18, 0.1), np.zeros(6), 0.0)])
    np.testing.assert_array_equal(featurize_raw(small), featurize(small))


def test_x_rows_are_the_x_position_alone_unclipped():
    states = [np.zeros(18) for _ in range(3)]
    for obs, x in zip(states, (0.1, -1.2, 103.0)):
        obs[0], obs[1] = x, 7.0  # obs[1] must not appear
    trajectory = Trajectory([Step(obs, np.full(6, 2.0), 0.0) for obs in states])

    features = featurize_x(trajectory)

    assert features.shape == (3, 1) and features.dtype == np.float32
    np.testing.assert_array_equal(features[:, 0], np.float32([0.1, -1.2, 103.0]))  # 103 not clipped
