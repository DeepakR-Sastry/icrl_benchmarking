"""Featurizers for AntWall, one per network input of the authors' code
(same three as HalfCheetahWithPos; docs/envs.md):
- featurize: the observation clipped to +-20 and the action clipped to
  [-1, 1], as Malik's ConstraintNet.prepare_data builds them (f28214e,
  icrl/constraint_net.py; his README AntWall commands keep --clip_obs at
  its default 20, no normalisation) and as Liu's does (0d79d2c,
  config/mujoco_BlockedAntWall/*: clip_obs 20, cn_normalize False, all
  columns). Maximum Entropy, Binary Classifier, VICRL.
- featurize_raw: the same columns unclipped, as Malik's GailDiscriminator
  builds them (icrl/gail_utils.py: clipping commented out). GACL.
- featurize_x: the x-position only, as Gaurav's ICL cost and flow see it
  (096e422, configs/ant.json: state_reduction s[:1]). ICL.
Malik's networks keep only some columns, by his bug (2) (on Ant:
range(113) + range(8)); that is a method setting (input_columns), not the
featurizer's job. A copy of featurizers/half_cheetah_with_pos.py with
Ant's sizes (docs/design.md: abstract on the third implementation)."""

import numpy as np

CLIP_OBS = 20.0  # Malik's --clip_obs default (icrl/icrl.py); Liu's clip_obs
ACTION_LOW, ACTION_HIGH = -1.0, 1.0  # the env's action bounds
OBS_SIZE, ACTION_SIZE = 113, 8  # icrl_bench/envs/ant_wall.py


def _obs_and_actions(trajectory):
    """The trajectory's observations (steps, 113) and actions (steps, 8)."""
    obs = np.array([step.state for step in trajectory.steps], dtype=np.float64)
    actions = np.array([step.action for step in trajectory.steps], dtype=np.float64)
    assert obs.shape[1:] == (OBS_SIZE,) and actions.shape[1:] == (ACTION_SIZE,), (obs.shape, actions.shape)
    return obs, actions


def featurize(trajectory):
    """One row per step: 113 observation numbers (x-position first) clipped
    to +-20, then 8 torques clipped to [-1, 1]. 121 columns."""
    obs, actions = _obs_and_actions(trajectory)
    obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)
    actions = np.clip(actions, ACTION_LOW, ACTION_HIGH)
    return np.concatenate([obs, actions], axis=1).astype(np.float32)


def featurize_raw(trajectory):
    """One row per step: the 113 observation numbers, then the 8 actions as
    given (sampled, possibly outside [-1, 1]), nothing clipped. 121 columns."""
    obs, actions = _obs_and_actions(trajectory)
    return np.concatenate([obs, actions], axis=1).astype(np.float32)


def featurize_x(trajectory):
    """One row per step: the x-position (obs[0]) alone, as given. 1 column.
    Gaurav's flow rescales its inputs itself (normalize_flow_inputs)."""
    obs, _ = _obs_and_actions(trajectory)
    return obs[:, :1].astype(np.float32)
