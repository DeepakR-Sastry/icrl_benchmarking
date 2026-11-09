from functools import partial

import numpy as np

from icrl_bench.envs.half_cheetah_with_pos import true_cost


def at(x):
    obs = np.zeros(18)
    obs[0] = x  # obs[0] is the x-position
    return obs


def test_default_wall_is_maliks_minus_3():
    assert [true_cost(at(x), None) for x in (-3.5, -3.0, -2.9, 0.0)] == [1.0, 1.0, 0.0, 0.0]


def test_gauravs_wall_at_minus_1_through_partial_as_build_does():
    rule = partial(true_cost, wall_x=-1)  # config: {name: half_cheetah_with_pos, wall_x: -1}
    assert [rule(at(x), None) for x in (-3.5, -1.0, -0.9, 5.0)] == [1.0, 1.0, 0.0, 0.0]
