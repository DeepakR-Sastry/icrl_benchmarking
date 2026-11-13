"""Ant with a wall rule, ported from Malik et al. (github.com/shehryar-malik/icrl,
commit f28214e): custom_envs/custom_envs/envs/ant.py (AntWall-v0,
AntWallTest-v0) for the env, icrl/true_constraint_net.py (wall_behind) for
the true cost. Liu et al. (0d79d2c) have the same class byte for byte;
Gaurav et al. (096e422) a copy that doesn't run as committed (docs/envs.md).
All three subclass gym 0.21's Ant-v3 on mujoco-py. Port notes: docs/envs.md.
metric_grid: Gaurav's ICL metric grid (096e422, tools/environments/__init__.py
L94-100, the same grid for AntWall and HCWithPos)."""

import numpy as np
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from gymnasium.spaces import Box

from icrl_bench.core.types import MetricGrid, Step, Trajectory

WALL_X = -3.0  # x-position at or behind which the ant breaks the rule
QPOS_SIZE, QVEL_SIZE = 15, 14  # x, y, z, quaternion, 8 joints; their velocities
CONTACT_SIZE = 14 * 6  # Ant-v3's cfrc_ext for all 14 bodies, worldbody included
OBS_SIZE, ACTION_SIZE = QPOS_SIZE + QVEL_SIZE + CONTACT_SIZE, 8  # 113, as their envs


class AntWall(AntEnv):
    """Gymnasium's Ant-v5 (same robot model file, frame skip, control cost,
    healthy bonus and reset noise as their Ant-v3), changed to match their
    AntWall:
    - observation: qpos (15, x first) + qvel (14) + 84 zeros. Their
      observation's last 84 numbers are contact forces, which mujoco-py
      never computes, so they are always 0 (measured in all three authors'
      venvs); gymnasium computes real ones, so they are not used here;
    - reward: distance of the torso from the origin (x, y) + 1 while healthy
      - 0.5 * sum(action^2); no contact cost (theirs is 0, as above);
    - never terminates on its own; 500 steps per episode.

    constrained=False is AntWall-v0 (nothing enforced). constrained=True is
    Malik's and Liu's AntWallTest-v0: once x < -3 (strictly, as theirs) the
    episode ends with reward 0. It is used only for evaluation.
    """

    def __init__(self, constrained=False, max_episode_steps=500):
        super().__init__(exclude_current_positions_from_observation=False,
                         terminate_when_unhealthy=False,
                         include_cfrc_ext_in_observation=False)
        # v5 sizes its space without contact forces (29); ours adds the zeros.
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float64)
        self.constrained = constrained
        self.max_episode_steps = max_episode_steps

    def reset(self, *, seed=None, options=None):
        self.time = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, _, info = super().step(action)
        if self.constrained and obs[0] < WALL_X:
            reward, terminated = 0.0, True
        self.time += 1
        truncated = self.time == self.max_episode_steps
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate((self.data.qpos.flatten(), self.data.qvel.flatten(),
                               np.zeros(CONTACT_SIZE)))

    def _get_rew(self, x_velocity, action):
        # Called by v5's step after the simulation, as their reward reads the
        # torso's position after the step (get_body_com("torso")).
        distance = np.linalg.norm(self.data.body(self._main_body).xpos[:2], ord=2)
        healthy_reward = self.healthy_reward
        ctrl_cost = self.control_cost(action)
        reward = distance + healthy_reward - ctrl_cost
        reward_info = {"reward_distance": distance, "reward_ctrl": -ctrl_cost,
                       "reward_survive": healthy_reward}
        return reward, reward_info


def true_cost(obs, action, wall_x=WALL_X):
    """The hidden rule: a step taken at x <= wall_x costs 1, otherwise 0.
    Malik's wall_behind(-3) (x <= -3, the default; Liu's is x < -3, which
    differs only at exactly -3); Gaurav's ant.json at -1 (cost_condition
    s[0] <= -1; config: wall_x: -1)."""
    return float(obs[0] <= wall_x)


def metric_grid():
    """Gaurav's AntWall grid = his HalfCheetah one (his code lists both envs
    in one branch): x in -5, -4.9, ..., 5 from his expression
    np.arange(-5, 5+0.1, 0.1), windows [x - (0.05 - 1e-3), x + 0.05], binned
    on obs[0]; each step is that x with every other number 0 and his
    np.zeros(8) action. A copy of half_cheetah_with_pos.metric_grid with
    Ant's sizes."""
    values = np.arange(-5, 5 + 0.1, 0.1)
    steps = []
    for x in values:
        obs = np.zeros(OBS_SIZE)
        obs[0] = x
        steps.append(Step(obs, np.zeros(ACTION_SIZE), None))
    return MetricGrid(values=values, trajectory=Trajectory(steps),
                      coordinate=lambda obs: obs[0],
                      left_gap=0.05 - 1e-3, right_gap=0.05)
