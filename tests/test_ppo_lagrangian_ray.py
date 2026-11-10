import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch.distributions import Independent, Normal

from icrl_bench.agents.ppo_lagrangian_ray import ActorTwoCriticsRay, gaussian_log_likelihood

OBS = spaces.Box(-np.inf, np.inf, (18,))
ACT = spaces.Box(-1.0, 1.0, (6,))


def test_networks_shapes_and_init():
    torch.manual_seed(0)
    net = ActorTwoCriticsRay(OBS, ACT)
    obs = torch.zeros(5, 18)
    assert net.mean(obs).shape == (5, 6)
    assert net.sample(obs).shape == (5, 6)
    assert net.reward_value(obs).shape == (5,)
    assert net.cost_value(obs).shape == (5,)
    assert torch.equal(net.log_std, torch.full((6,), -0.5))
    # Glorot uniform: every weight within ±sqrt(6 / (fan_in + fan_out)); zero biases.
    for layer in net.mean_net:
        if isinstance(layer, torch.nn.Linear):
            fan_out, fan_in = layer.weight.shape
            assert layer.weight.abs().max() <= np.sqrt(6 / (fan_in + fan_out))
            assert torch.all(layer.bias == 0)
    # Policy optimiser: mean net + log std; critic optimiser: both critics; disjoint.
    policy_ids = {id(p) for p in net.policy_parameters()}
    value_ids = {id(p) for p in net.value_parameters()}
    assert id(net.log_std) in policy_ids and not policy_ids & value_ids
    assert len(policy_ids) + len(value_ids) == len(list(net.parameters()))
    # His PPOLag on HC prints "pi: 5772, v: 5441, vc: 5441".
    count = lambda params: sum(p.numel() for p in params)
    assert count(net.policy_parameters()) == 5772
    assert count(net.reward_value_net.parameters()) == 5441
    assert count(net.cost_value_net.parameters()) == 5441


def test_gaussian_log_likelihood_is_a_gaussian_up_to_eps():
    torch.manual_seed(0)
    mean, log_std, a = torch.randn(4, 6), torch.full((6,), -0.5), torch.randn(4, 6)
    ours = gaussian_log_likelihood(a, mean, log_std)
    torch_lp = Independent(Normal(mean, log_std.exp()), 1).log_prob(a)
    assert torch.allclose(ours, torch_lp, atol=1e-5)


def test_discrete_actions_refused():
    with pytest.raises(ValueError, match="Box"):
        ActorTwoCriticsRay(OBS, spaces.Discrete(3))


# --- Piece 2: rollout and buffer -------------------------------------------

import gymnasium as gym  # noqa: E402
from gymnasium.wrappers import TimeLimit  # noqa: E402

from icrl_bench.agents.ppo_lagrangian_ray import (  # noqa: E402
    collect_epoch, discount_cumsum, path_targets)
from icrl_bench.core.vector_env import make_vector_env  # noqa: E402


class Counter(gym.Env):
    """obs = [steps taken, 0], reward 1; terminates at terminate_at if given."""
    observation_space = spaces.Box(-np.inf, np.inf, (2,))
    action_space = spaces.Box(-1.0, 1.0, (1,))

    def __init__(self, terminate_at=None):
        self.terminate_at = terminate_at

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        return np.array([self.t, 0.0], dtype=np.float32), 1.0, self.t == self.terminate_at, False, {}


def test_discount_cumsum_and_path_targets_by_hand():
    assert np.allclose(discount_cumsum(np.array([1.0, 1.0, 1.0]), 0.5), [1.75, 1.5, 1.0])
    # rewards [1, 2], predictions [0.5, 0.5], next-state prediction 1, γ 0.5, λ 1:
    # deltas [1 + 0.25 - 0.5, 2 + 0.5 - 0.5] = [0.75, 2]; advantages [0.75 + 0.5·2, 2];
    # targets: discounted sums of [1, 2, 1] = [1 + 0.5·(2 + 0.5), 2 + 0.5].
    adv, ret = path_targets(np.array([1.0, 2.0]), np.array([0.5, 0.5]), 1.0, 0.5, 1.0)
    assert np.allclose(adv, [1.75, 2.0]) and np.allclose(ret, [2.25, 2.5])


def run_epoch(terminate_at, logp_from_other_draw=False, gamma=0.5):
    torch.manual_seed(0)
    net = ActorTwoCriticsRay(Counter.observation_space, Counter.action_space)
    envs = make_vector_env(lambda: TimeLimit(Counter(terminate_at), 3), 1, lambda o, a: 1.0)
    return net, collect_epoch(net, envs, n_steps=5, gamma=gamma, lam=1.0,
                              logp_from_other_draw=logp_from_other_draw)


def test_collect_epoch_paths_and_bootstraps():
    gamma = 0.5
    vc = lambda x: net.cost_value(torch.tensor([[x, 0.0]])).item()
    # Steps 0-2: an episode cut by the time limit -> bootstrapped with the
    # cost critic at its true last state [3, 0] (final_obs, not the reset
    # state). Steps 3-4: cut by the epoch end -> bootstrapped at [2, 0].
    net, out = run_epoch(terminate_at=None)
    assert np.allclose(out["observations"][:, 0], [0, 1, 2, 0, 1])
    assert np.allclose(out["cost_returns"][:3], discount_cumsum(np.array([1, 1, 1, vc(3)]), gamma)[:-1])
    assert np.allclose(out["cost_returns"][3:], discount_cumsum(np.array([1, 1, vc(2)]), gamma)[:-1])
    # Only the finished episode is counted.
    assert out["episode_returns"].tolist() == [3.0] and out["episode_lengths"].tolist() == [3]
    assert out["episode_costs"].tolist() == [3.0]
    assert np.isclose(out["episode_discounted_costs"][0], 1 + 0.5 + 0.25)
    assert out["average_cost"] == 1.0
    # A true termination before the time limit is not bootstrapped
    # (steps 0-1); his rule bootstraps whenever the time limit is reached,
    # even if the env also terminated there (L286).
    net, out = run_epoch(terminate_at=2)
    assert np.allclose(out["cost_returns"][:2], [1.5, 1.0])
    net, out = run_epoch(terminate_at=3)
    assert np.allclose(out["cost_returns"][:3], discount_cumsum(np.array([1, 1, 1, vc(3)]), gamma)[:-1])
    # Advantages: reward normalised over the epoch, cost centred only.
    assert abs(out["reward_advantages"].mean()) < 1e-6 and np.isclose(out["reward_advantages"].std(), 1, atol=1e-4)
    assert abs(out["cost_advantages"].mean()) < 1e-6


def test_bug_9_switch():
    # Off: the stored log π is the played action's.
    _, out = run_epoch(terminate_at=None)
    recomputed = gaussian_log_likelihood(torch.tensor(out["actions"]), torch.tensor(out["means"]),
                                         torch.tensor(out["log_stds"]))
    assert np.allclose(out["log_probs"], recomputed.numpy(), atol=1e-6)
    # On: it belongs to another draw, so it differs on every step.
    _, out = run_epoch(terminate_at=None, logp_from_other_draw=True)
    recomputed = gaussian_log_likelihood(torch.tensor(out["actions"]), torch.tensor(out["means"]),
                                         torch.tensor(out["log_stds"]))
    assert np.all(np.abs(out["log_probs"] - recomputed.numpy()) > 1e-4)


# --- Piece 3: one update ---------------------------------------------------

from torch.distributions import kl_divergence  # noqa: E402

from icrl_bench.agents.ppo_lagrangian_ray import gaussian_kl, update  # noqa: E402


def test_gaussian_kl_is_kl_new_to_old():
    torch.manual_seed(0)
    m0, m1, s0, s1 = torch.randn(4, 6), torch.randn(4, 6), torch.randn(6) * 0.3, torch.randn(6) * 0.3
    expected = kl_divergence(Independent(Normal(m0, s0.exp()), 1), Independent(Normal(m1, s1.exp()), 1)).mean()
    assert torch.isclose(gaussian_kl(m0, s0, m1, s1), expected, atol=1e-5)
    assert gaussian_kl(m0, s0, m0, s0).item() == pytest.approx(0, abs=1e-6)


def run_update(target_kl, pi_iters=10, vf_iters=5):
    net, batch = run_epoch(terminate_at=None, gamma=0.99)
    p = torch.nn.Parameter(torch.tensor(0.5413))
    pi_opt = torch.optim.Adam(net.policy_parameters() + [p], lr=5e-4, eps=1e-8)
    v_opt = torch.optim.Adam(net.value_parameters(), lr=1e-3, eps=1e-8)
    p_before = p.item()
    stats = update(net, p, pi_opt, v_opt, batch, clip_ratio=0.2, ent_coef=0.0, pi_iters=pi_iters,
                   target_kl=target_kl, kl_margin=1.2, vf_iters=vf_iters)
    return net, batch, p.item() - p_before, stats


def test_update_kl_stop_uses_pre_step_kl():
    # Step 1's pre-step KL is 0 (the rollout's own weights), so even a tiny
    # target stops only after step 2 (his (g)); a huge one never stops.
    assert run_update(target_kl=1e-12)[3]["pi_steps"] == 2
    assert run_update(target_kl=1e9)[3]["pi_steps"] == 10


def test_update_moves_penalty_param_and_fits_critics():
    net, batch, p_moved, stats = run_update(target_kl=1e9, vf_iters=50)
    assert p_moved != 0  # his (f): the policy optimiser trains p too
    rr = torch.as_tensor(batch["reward_returns"])
    v_after = ((rr - net.reward_value(torch.as_tensor(batch["observations"]))) ** 2).mean().item()
    assert v_after < stats["loss_v"]


# --- Piece 4: λ --------------------------------------------------------------

from icrl_bench.agents.ppo_lagrangian_ray import PenaltyRay  # noqa: E402


def test_penalty_matches_his_log_and_has_no_floor():
    lam = PenaltyRay(initial_value=1.0, learning_rate=0.05, budget=15.0)
    assert lam.value == pytest.approx(1.0, abs=1e-6)
    # c above β: Adam's first step is exactly lr, so p 0.5413 -> 0.5913 and
    # λ = 1.032, his seed-1 log's second value (bug (8): c = reward ≈ 300).
    lam.update(300.0)
    assert lam.param.item() == pytest.approx(0.5413 + 0.05, abs=1e-4)
    assert lam.value == pytest.approx(1.032, abs=5e-4)
    # A steady c: every Adam step is lr, so 49 more add 0.05 each (a c that
    # changes size makes Adam's steps a little smaller for a while).
    for _ in range(49):
        lam.update(300.0)
    assert lam.param.item() == pytest.approx(0.5413 + 50 * 0.05, abs=1e-3)
    # c below β for long: p falls with no floor, λ toward 0 (ours stops at 0.541).
    lam = PenaltyRay(initial_value=1.0, learning_rate=0.05, budget=15.0)
    for _ in range(100):
        lam.update(0.0)
    assert lam.value < 0.05


# --- Piece 5: the agent ------------------------------------------------------

from icrl_bench.agents.ppo_lagrangian_ray import (  # noqa: E402
    PPOLagrangianRay, PPOLagrangianRaySettings)


def counter_envs(time_limit=3):
    # reward 1 per step, cost 0: episode reward 3, episode cost 0
    return make_vector_env(lambda: TimeLimit(Counter(), time_limit), 1, lambda o, a: 0.0)


def make_ray_agent(**overrides):
    settings = PPOLagrangianRaySettings(n_steps=6, pi_iters=2, vf_iters=2, budget=1.0, **overrides)
    return PPOLagrangianRay(Counter.observation_space, Counter.action_space, settings)


def test_train_rows_callback_and_policy():
    agent, rows = make_ray_agent(), []
    log = agent.train(counter_envs(), total_steps=12, callback=rows.append)
    assert rows == log and [r["steps"] for r in log] == [6, 12]
    assert log[0]["episode_reward"] == 3.0 and log[0]["episode_cost"] == 0.0
    assert log[0]["penalty"] == pytest.approx(1.0, abs=1e-6)  # logged before the λ step
    action = agent.policy(np.zeros(2, dtype=np.float32))
    assert action.shape == (1,)


def test_bug_8_switch_decides_what_lambda_reads():
    # β = 1: off, λ reads episode cost 0 and falls; on, episode reward 3 and rises.
    off, on = make_ray_agent(), make_ray_agent(lambda_reads_reward=True)
    off.train(counter_envs(), total_steps=6)
    on.train(counter_envs(), total_steps=6)
    assert off.penalty.value < 1.0 < on.penalty.value


def test_same_seed_same_starting_weights():
    a, b = make_ray_agent(seed=3), make_ray_agent(seed=3)
    for pa, pb in zip(a.net.parameters(), b.net.parameters()):
        assert torch.equal(pa, pb)


def test_epoch_without_a_finished_episode_refused():
    with pytest.raises(ValueError, match="no episode finished"):
        make_ray_agent().train(counter_envs(time_limit=10), total_steps=6)


class ActionEcho(Counter):
    """obs = [steps taken, the action the env received]."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs[1] = action[0]
        return obs, reward, terminated, truncated, info


def test_env_gets_clipped_action_buffer_keeps_draw():
    # His GymEnvironment.step clips to the action bounds; his buffer stores
    # the unclipped draw.
    torch.manual_seed(0)
    net = ActorTwoCriticsRay(Counter.observation_space, Counter.action_space)
    with torch.no_grad():
        net.log_std.fill_(2.0)  # std ≈ 7.4: most draws outside [-1, 1]
    envs = make_vector_env(lambda: TimeLimit(ActionEcho(), 100), 1, lambda o, a: 0.0)
    out = collect_epoch(net, envs, n_steps=20, gamma=0.99, lam=0.97, logp_from_other_draw=False)
    assert np.abs(out["actions"]).max() > 1
    assert np.allclose(out["observations"][1:, 1], np.clip(out["actions"][:-1, 0], -1, 1))
    agent = make_ray_agent()
    with torch.no_grad():
        agent.net.log_std.fill_(2.0)
    assert all(np.abs(agent.policy(np.zeros(2, dtype=np.float32))).max() <= 1 for _ in range(20))
