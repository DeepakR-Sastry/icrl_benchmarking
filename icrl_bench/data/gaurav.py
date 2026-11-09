"""Load the expert trajectories Gaurav et al. ship with their code
(github.com/ashishgaurav13/ICL, commit 096e422: expert-data/data-<env>.pt)
as our Trajectories, exactly as his code reads them.

A file is his TrajectoryDataset's raw list (tools/base/dataset.py save:
torch.save(self.d)): one [S, A] pair per episode, S the states and A the
actions, as lists of numpy arrays. His play_episode (tools/base/
environment.py L76-91) chooses action t on state t and trajectory_dataset
stores [S[:-1], A], so step t pairs S[t] with A[t] — the same pairing as
ours; no shift. His icl.py reads the whole file (L30-35; CostAdjustment
L28-29). Step.reward is None: he stores no rewards, no method reads them."""

import torch

from icrl_bench.core.types import Step, Trajectory


def load_gaurav_rollouts(path):
    """Every episode of his file, in his order, as a list of Trajectory."""
    # weights_only=False: the file holds numpy arrays in Python lists, which
    # torch's safe loader refuses. It unpickles, so only open files from his
    # repository.
    episodes = torch.load(path, weights_only=False)
    trajectories = []
    for states, actions in episodes:
        assert len(states) == len(actions), f"{len(states)} states, {len(actions)} actions"
        trajectories.append(Trajectory([Step(s, a, None) for s, a in zip(states, actions)]))
    return trajectories
