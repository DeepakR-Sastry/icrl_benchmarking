import numpy as np
import torch
from gymnasium.spaces import Box

from icrl_bench.agents.icl_ppo_policy import ICLPPOPolicy

OBS, ACTIONS = Box(-np.inf, np.inf, (18,)), Box(-1, 1, (6,))


def test_only_the_outer_layers_get_his_init():
    torch.manual_seed(0)
    net = ICLPPOPolicy(OBS, ACTIONS).net
    for name, p in net.named_parameters():
        if name.endswith("bias"):
            is_his = torch.equal(p, torch.full_like(p, 0.01))
            # "0.bias", "2.bias": the two outer Linears; "4.*": inside Gaussian2.
            assert is_his == (name in ("0.bias", "2.bias")), name


def test_the_head_has_width_8_and_bounded_std():
    torch.manual_seed(0)
    net = ICLPPOPolicy(OBS, ACTIONS).net
    assert net[4].E[0].weight.shape == (8, 64)
    # Large states push the log std to its clamp, never past it.
    with torch.no_grad():
        mean, std = net(torch.randn(500, 18) * 50)
    assert mean.abs().max() <= 1
    assert std.max() <= np.exp(0.5) + 1e-6


def test_actions_are_sampled_with_the_action_shape():
    torch.manual_seed(0)
    policy = ICLPPOPolicy(OBS, ACTIONS)
    obs = np.zeros(18)
    first, second = policy.policy(obs), policy.policy(obs)
    assert first.shape == (6,) and first.dtype == np.float32
    assert not np.array_equal(first, second)  # sampled, not the mean


def test_it_does_not_reseed():
    # Two built in a row continue the stream, so they differ (his
    # CostAdjustment builds a new one each round).
    torch.manual_seed(0)
    a, b = ICLPPOPolicy(OBS, ACTIONS), ICLPPOPolicy(OBS, ACTIONS)
    assert not torch.equal(a.net[0].weight, b.net[0].weight)
