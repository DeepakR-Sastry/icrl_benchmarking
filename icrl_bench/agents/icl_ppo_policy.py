"""The actor of Gaurav et al.'s PPO (github.com/ashishgaurav13/ICL, commit
096e422: tools/algorithms/ppo.py PPOPolicy, continuous actions), never
trained. ICL's run_icl can use it to play the backward-step episodes, which
is what his code does on PPOLag envs (bug (7), docs/methods.md §5;
docs/decisions.md 2026-11-08).

Only what that use needs: building the network as he does and acting. No
critic, no training, no discrete actions."""

import numpy as np
import torch
from torch import nn


class _Gaussian2(nn.Module):
    """His Gaussian2 (tools/utils/torch_helper.py L212-225): its own two
    hidden layers (width h, default 8), a tanh mean and a state-dependent
    std, log std clamped to [-20, 0.5]."""

    def __init__(self, i, o, h=8):
        super().__init__()
        self.E = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.Mu = nn.Linear(h, o)
        self.LogStd = nn.Linear(h, o)

    def forward(self, x):
        x = self.E(x)
        return torch.tanh(self.Mu(x)), torch.clamp(self.LogStd(x), -20, 0.5).exp()


class ICLPPOPolicy:
    """His PPOPolicy.Pi: obs -> h tanh -> h tanh -> Gaussian2(h -> actions),
    and its act (sample, no clipping).

    Built from the current torch random stream, without reseeding (his
    CostAdjustment builds a new one each round from the continuing stream)."""

    def __init__(self, observation_space, action_space, hidden=64):
        assert len(observation_space.shape) == 1 and len(action_space.shape) == 1
        # Built in his order (TorchHelper.nn: Linear, Tanh, Linear, Tanh,
        # Gaussian2), so the same seed draws the same default weights.
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            _Gaussian2(hidden, action_space.shape[0]))
        # His init_weights (Xavier-uniform, bias 0.01) reaches only the two
        # outer Linears: Gaussian2 is a ScriptModule, whose layers .apply
        # doesn't see (quirk (e)), so they keep PyTorch's default init.
        for layer in (self.net[0], self.net[2]):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def policy(self, obs):
        """One raw observation in, one sampled action out (his act with
        deterministic=False: Normal(mean, std).sample())."""
        with torch.no_grad():
            mean, std = self.net(torch.tensor(np.array([obs]), dtype=torch.float32))
            return torch.distributions.Normal(mean, std).sample().view(-1).numpy()
