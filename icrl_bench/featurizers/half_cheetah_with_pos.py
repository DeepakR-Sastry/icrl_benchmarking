"""Featurizers for HalfCheetahWithPos, one per network input of Malik's
(commit f28214e), and one for Gaurav's ICL (commit 096e422):
- featurize: the clipped observation and the clipped action, as his
  ConstraintNet.prepare_data builds them (icrl/constraint_net.py) with his
  HCWithPos settings (no normalisation, --clip_obs 20). Maximum Entropy,
  Binary Classifier.
- featurize_raw: the same columns unclipped, as his GailDiscriminator builds
  them (icrl/gail_utils.py: its normalize_obs body and clip_actions call are
  commented out). GACL (docs/decisions.md, 2026-11-01).
- featurize_x: the x-position only, unclipped, as Gaurav's ICL cost and flow
  see it on HalfCheetah (configs/hc.json: state_reduction s[:1], input
  format [*s], i = 1). ICL.
His networks then keep only some of these columns, by a bug; that is a
method setting (input_columns), not the featurizer's job."""

import numpy as np

CLIP_OBS = 20.0  # Malik's --clip_obs default (icrl/icrl.py)
ACTION_LOW, ACTION_HIGH = -1.0, 1.0  # the env's action bounds


def _obs_and_actions(trajectory):
    """The trajectory's observations (steps, 18) and actions (steps, 6)."""
    obs = np.array([step.state for step in trajectory.steps], dtype=np.float64)
    actions = np.array([step.action for step in trajectory.steps], dtype=np.float64)
    assert obs.shape[1:] == (18,) and actions.shape[1:] == (6,), (obs.shape, actions.shape)
    return obs, actions


def featurize(trajectory):
    """One row per step: 18 observation numbers (x-position first) clipped to
    +-20, then 6 torques clipped to [-1, 1]. 24 columns."""
    obs, actions = _obs_and_actions(trajectory)
    obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)
    actions = np.clip(actions, ACTION_LOW, ACTION_HIGH)
    return np.concatenate([obs, actions], axis=1).astype(np.float32)


def featurize_raw(trajectory):
    """One row per step: the 18 observation numbers, then the 6 actions as
    given (sampled, possibly outside [-1, 1]), nothing clipped. 24 columns.
    On Malik's expert pile x reaches ~138, so clipping would matter."""
    obs, actions = _obs_and_actions(trajectory)
    return np.concatenate([obs, actions], axis=1).astype(np.float32)


def featurize_x(trajectory):
    """One row per step: the x-position (obs[0]) alone, as given. 1 column.
    Gaurav's flow rescales its inputs itself (normalize_flow_inputs)."""
    obs, _ = _obs_and_actions(trajectory)
    return obs[:, :1].astype(np.float32)
