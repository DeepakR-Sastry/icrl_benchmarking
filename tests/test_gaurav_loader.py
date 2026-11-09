import numpy as np
import torch

from icrl_bench.data.gaurav import load_gaurav_rollouts


def test_reads_every_episode_in_order_without_re_pairing(tmp_path):
    # His format: [[S, A], ...], lists of numpy arrays; state t of episode k
    # starts with 100k + t, action t with t.
    episodes = [[[np.full(18, 100.0 * k + t) for t in range(n)],
                 [np.full(6, float(t), dtype=np.float32) for t in range(n)]]
                for k, n in enumerate((3, 5))]
    torch.save(episodes, tmp_path / "data-hc.pt")

    trajectories = load_gaurav_rollouts(tmp_path / "data-hc.pt")

    assert [len(t.steps) for t in trajectories] == [3, 5]
    first = trajectories[0].steps
    assert [s.state[0] for s in first] == [0.0, 1.0, 2.0]   # S[t] with A[t]
    assert [s.action[0] for s in first] == [0.0, 1.0, 2.0]
    assert trajectories[1].steps[0].state[0] == 100.0
    assert all(s.reward is None for t in trajectories for s in t.steps)
