"""HalfCheetah with its x-position in the observation, ported from Malik et al.
(github.com/shehryar-malik/icrl, commit f28214e):
custom_envs/custom_envs/envs/half_cheetah.py (HCWithPos-v0, HCWithPosTest-v0)
for the env, icrl/true_constraint_net.py (wall_behind) for the true cost.
Liu et al. define the same env (checked). Port notes: docs/envs.md.
metric_grid: Gaurav's ICL metric grid (github.com/ashishgaurav13/ICL,
096e422, tools/environments/__init__.py L94-100)."""

import numpy as np
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv

from icrl_bench.core.types import MetricGrid, Step, Trajectory

WALL_X = -3.0  # x-position at or behind which the cheetah breaks the rule
OBS_SIZE, ACTION_SIZE = 18, 6  # with x-position (checked in the tests)


class HalfCheetahWithPos(HalfCheetahEnv):
    """Gymnasium's HalfCheetah-v5 (same robot model, frame skip, control cost
    and reset noise as Malik's), with two changes to match his env:
    obs[0] is the x-position (18 numbers: qpos then qvel), and the reward
    pays speed in either direction, |x velocity| - 0.1 * sum(action^2).

    constrained=False is Malik's HCWithPos-v0 (nothing enforced).
    constrained=True is Malik's HCWithPosTest-v0 (at x <= -3 the episode
    ends with reward 0). It is the "real" world, used only for evaluation.
    """

    def __init__(self, constrained=False, max_episode_steps=1000):
        super().__init__(exclude_current_positions_from_observation=False)
        self.constrained = constrained
        self.max_episode_steps = max_episode_steps

    def reset(self, *, seed=None, options=None):
        self.time = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, _, info = super().step(action)
        if self.constrained and info["x_position"] <= WALL_X:
            reward, terminated = 0.0, True
        self.time += 1
        truncated = self.time == self.max_episode_steps
        return obs, reward, terminated, truncated, info

    def _get_rew(self, x_velocity, action):
        # v5 pays signed velocity; Malik's old_reward pays its absolute value.
        return super()._get_rew(abs(x_velocity), action)


def true_cost(obs, action, wall_x=WALL_X):
    """The hidden rule: a step taken at x <= wall_x costs 1, otherwise 0.
    Malik's and Liu's wall is at -3 (the default); Gaurav's at -1
    (configs/hc.json cost_condition s[0] <= -1; config: wall_x: -1). The
    constrained env keeps its wall at -3 (Gaurav's HC has none)."""
    return float(obs[0] <= wall_x)


def metric_grid():
    """Gaurav's HalfCheetah grid: x in -5, -4.9, ..., 5, built with his
    expression (np.arange(-5, 5+0.1, 0.1): the floats decide which side of
    the wall a point near -1 falls), his windows [x - (0.05 - 1e-3),
    x + 0.05], binned on obs[0] (his accrual compares only s[0])."""
    values = np.arange(-5, 5 + 0.1, 0.1)
    steps = []
    for x in values:
        obs = np.zeros(OBS_SIZE)
        obs[0] = x
        steps.append(Step(obs, np.zeros(ACTION_SIZE), None))
    return MetricGrid(values=values, trajectory=Trajectory(steps),
                      coordinate=lambda obs: obs[0],
                      left_gap=0.05 - 1e-3, right_gap=0.05)
