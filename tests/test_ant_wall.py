from functools import partial

import numpy as np

from icrl_bench.envs.ant_wall import AntWall, true_cost


def at(x):
    obs = np.zeros(113)
    obs[0] = x  # obs[0] is the x-position
    return obs


def test_true_cost_walls_maliks_minus_3_and_gauravs_minus_1():
    assert [true_cost(at(x), None) for x in (-3.5, -3.0, -2.9, 0.0)] == [1.0, 1.0, 0.0, 0.0]
    rule = partial(true_cost, wall_x=-1)  # config: {name: ant_wall, wall_x: -1}
    assert [rule(at(x), None) for x in (-3.5, -1.0, -0.9, 5.0)] == [1.0, 1.0, 0.0, 0.0]


def test_observation_is_theirs_and_reward_is_their_formula():
    # Their 113 numbers: qpos (x first) + qvel + 84 contact forces that
    # mujoco-py leaves at 0 (docs/envs.md). Reward: distance from the
    # origin + 1 while healthy - 0.5 * sum(a^2), no contact cost.
    env = AntWall()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.shape == obs.shape == (113,)
    rng = np.random.default_rng(0)
    for _ in range(20):
        action = rng.uniform(-1, 1, 8)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert obs[0] == env.data.qpos[0] and not obs[29:].any()
        by_hand = (np.linalg.norm(env.data.body("torso").xpos[:2]) + float(env.is_healthy)
                   - 0.5 * np.sum(action ** 2))
        assert reward == by_hand and not terminated and not truncated


def test_constrained_ends_strictly_behind_minus_3_as_antwalltest():
    for x, ends in [(-3.5, True), (-2.9, False)]:
        env = AntWall(constrained=True)
        env.reset(seed=1)
        qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
        qpos[0] = x
        env.set_state(qpos, qvel)
        obs, reward, terminated, _, _ = env.step(np.zeros(8))
        assert terminated == ends and (reward == 0.0) == ends


def test_episode_is_truncated_at_500_steps():
    env = AntWall()
    env.reset(seed=2)
    for t in range(500):
        _, _, terminated, truncated, _ = env.step(np.zeros(8))
        assert not terminated and truncated == (t == 499)
