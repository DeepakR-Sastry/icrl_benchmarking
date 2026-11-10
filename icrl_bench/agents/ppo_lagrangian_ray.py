"""PPO-Lagrangian as in Gaurav et al.'s ICL code (github.com/ashishgaurav13/
ICL, commit 096e422): his PPOLag, tools/algorithms/ppo_lag.py, built on his
copy of Ray et al.'s safety-starter-agents (tools/safe_rl/pg/), ported from
TensorFlow to PyTorch. Cited where each piece is used. A different
algorithm from agents/ppo_lagrangian.py (Malik's SB3 fork), not a variant
of it, so it shares no code with it (docs/methods.md §5 lists every
difference; docs/decisions.md 2026-11-10). His bugs (8) and (9) are
settings, off by default."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

EPS = 1e-8  # safe_rl/pg/utils.py EPS


def mlp(in_dim, out_dim, hidden=(64, 64)):
    """Tanh hidden layers, no activation on the output (network.py mlp).
    Glorot-uniform weights and zero biases: the defaults of TF's
    tf.compat.v1.layers.dense, which his mlp uses."""
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.Tanh()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    net = nn.Sequential(*layers)
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return net


def gaussian_log_likelihood(actions, mean, log_std):
    """log π(a) of a diagonal Gaussian, summed over action dimensions, with
    EPS added to the std as his network.py gaussian_likelihood does."""
    pre_sum = -0.5 * (((actions - mean) / (log_std.exp() + EPS)) ** 2
                      + 2 * log_std + np.log(2 * np.pi))
    return pre_sum.sum(dim=-1)


class ActorTwoCriticsRay(nn.Module):
    """Policy mean, reward critic and cost critic: three separate 64x64 Tanh
    MLPs, plus a state-independent log std starting at -0.5
    (network.py mlp_actor_critic + mlp_gaussian_policy; hidden_sizes keeps
    its default (64, 64): ppo_lag.py passes only action_space). Box
    observations and actions only (his PPOLag envs are all continuous)."""

    def __init__(self, observation_space, action_space):
        super().__init__()
        if not isinstance(action_space, spaces.Box):
            raise ValueError("PPOLagrangianRay supports Box (continuous) actions only")
        obs_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))
        self.mean_net = mlp(obs_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.reward_value_net = mlp(obs_dim, 1)
        self.cost_value_net = mlp(obs_dim, 1)

    def policy_parameters(self):
        """What his policy optimiser trains (TF scope 'pi')."""
        return list(self.mean_net.parameters()) + [self.log_std]

    def value_parameters(self):
        """What his critic optimiser trains (scopes 'vf' and 'vc')."""
        return list(self.reward_value_net.parameters()) + list(self.cost_value_net.parameters())

    def mean(self, obs):
        return self.mean_net(obs.float())

    def sample(self, obs):
        """One action per observation: mean + N(0, 1) · std (his pi op)."""
        mean = self.mean(obs)
        return mean + torch.randn_like(mean) * self.log_std.exp()

    def reward_value(self, obs):
        return self.reward_value_net(obs.float()).squeeze(-1)

    def cost_value(self, obs):
        return self.cost_value_net(obs.float()).squeeze(-1)


def gaussian_kl(mean0, log_std0, mean1, log_std1):
    """KL(N0 ‖ N1) of diagonal Gaussians, summed over action dimensions and
    averaged over the batch (network.py gaussian_kl, with its EPS). His
    d_kl is gaussian_kl(new, new, old, old): KL(new ‖ old)."""
    var0, var1 = (2 * log_std0).exp(), (2 * log_std1).exp()
    pre_sum = 0.5 * (((mean1 - mean0) ** 2 + var0) / (var1 + EPS) - 1) + log_std1 - log_std0
    return pre_sum.sum(dim=-1).mean()


def gaussian_entropy(log_std):
    """Entropy of a diagonal Gaussian (network.py gaussian_entropy)."""
    return (log_std + 0.5 * np.log(2 * np.pi * np.e)).sum(dim=-1)


def discount_cumsum(x, discount):
    """out[t] = x[t] + discount · out[t + 1] (pg/utils.py discount_cumsum,
    which does it with scipy's lfilter; a plain loop here, in float64)."""
    out = np.zeros(len(x), dtype=np.float64)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + discount * running
        out[t] = running
    return out


def path_targets(signal, predictions, last_value, gamma, lam):
    """GAE advantages and critic targets for one path (one episode, or the
    piece of one the epoch cut off), for one signal: rewards with the reward
    critic's predictions, or costs with the cost critic's (buffer.py
    CPOBuffer.finish_path). last_value: the critic's prediction for the
    state after the path, or 0 if the episode truly ended. Targets are the
    discounted sums of the signal, bootstrapped with last_value — not
    advantage + prediction."""
    x = np.append(signal, last_value)
    v = np.append(predictions, last_value)
    deltas = x[:-1] + gamma * v[1:] - v[:-1]
    return discount_cumsum(deltas, gamma * lam), discount_cumsum(x, gamma)[:-1]


def step_cost(infos):
    """The one env copy's info["cost"] (CostWrapper). When the episode
    ended this step, SAME_STEP resets put the step's info under
    "final_info"."""
    for source in (infos.get("final_info", {}), infos):
        if "cost" in source and source["_cost"][0]:
            return float(source["cost"][0])
    raise KeyError('info["cost"] missing: wrap the env in CostWrapper')


def collect_epoch(net, envs, n_steps, gamma, lam, logp_from_other_draw):
    """One epoch of his PPOLag.train (ppo_lag.py L239-318): reset the env,
    take n_steps steps, finish each path, and return the buffer as his
    CPOBuffer.get hands it to the update.

    envs: a gymnasium vector env with ONE copy (his PPOLag runs one env),
    autoreset SAME_STEP, wrapped in CostWrapper. The env gets the action
    clipped to the action bounds (his GymEnvironment.step, gym_environment.py
    L593-594), the buffer keeps the unclipped draw, as his. A path ends
    when the episode ends or the epoch does; the critics' next-state
    prediction bootstraps it unless the episode
    truly terminated (a time-limit end is bootstrapped, L286-291).

    logp_from_other_draw (his bug (9), methods.md §5): he samples an action
    twice per step (L252 and again in policy.act, L254), plays and stores
    the second, but stores the first one's log-probability. False = the
    played action's log-probability (the intended algorithm).

    Returns a dict: per step (n_steps, ...) "observations", "actions",
    "log_probs", "means", "log_stds" (for the KL check), "reward_advantages"
    (normalised over the epoch), "cost_advantages" (centred only),
    "reward_returns", "cost_returns"; per finished episode
    "episode_returns", "episode_costs" (summed), "episode_lengths",
    "episode_discounted_costs" (his edcv); and "average_cost" (per step)."""
    if envs.num_envs != 1:
        raise ValueError("PPOLagrangianRay runs one env copy, as his PPOLag")
    obs, _ = envs.reset()
    steps = {key: [] for key in ["observations", "actions", "log_probs", "means", "log_stds",
                                 "rewards", "costs", "reward_values", "cost_values"]}
    targets = {key: np.zeros(n_steps, dtype=np.float32) for key in
               ["reward_advantages", "reward_returns", "cost_advantages", "cost_returns"]}
    episodes = {key: [] for key in ["episode_returns", "episode_costs", "episode_lengths",
                                    "episode_discounted_costs"]}
    path_start = 0
    ep_ret, ep_cost, ep_dcost, ep_len, discount = 0.0, 0.0, 0.0, 0, 1.0

    for t in range(n_steps):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32)  # (1, obs_dim)
            if logp_from_other_draw:
                logged_action = net.sample(o)  # his first draw (L252): only its log π is kept
                action = net.sample(o)         # his second draw (policy.act, L254): played, stored
            else:
                action = net.sample(o)
                logged_action = action
            mean = net.mean(o)
            log_prob = gaussian_log_likelihood(logged_action, mean, net.log_std)
            reward_value, cost_value = net.reward_value(o), net.cost_value(o)

        space = envs.single_action_space
        env_action = np.clip(action.numpy(), space.low, space.high)
        new_obs, rewards, terminated, truncated, infos = envs.step(env_action)
        reward, cost = float(rewards[0]), step_cost(infos)
        for key, value in [("observations", obs[0]), ("actions", action[0].numpy()),
                           ("log_probs", log_prob.item()), ("means", mean[0].numpy()),
                           ("log_stds", net.log_std.detach().numpy().copy()),
                           ("rewards", reward), ("costs", cost),
                           ("reward_values", reward_value.item()),
                           ("cost_values", cost_value.item())]:
            steps[key].append(value)
        ep_ret += reward
        ep_cost += cost
        ep_dcost += discount * cost
        discount *= gamma
        ep_len += 1

        done = bool(terminated[0] or truncated[0])
        if done or t == n_steps - 1:
            if terminated[0] and not truncated[0]:
                last_reward_value, last_cost_value = 0.0, 0.0
            else:
                # The state after the last step: on an episode end SAME_STEP
                # has already reset, and the true one is in final_obs.
                next_obs = infos["final_obs"][0] if done else new_obs[0]
                with torch.no_grad():
                    n = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)
                    last_reward_value = net.reward_value(n).item()
                    last_cost_value = net.cost_value(n).item()
            path = slice(path_start, t + 1)
            for signal, values, last, adv_key, ret_key in [
                    ("rewards", "reward_values", last_reward_value, "reward_advantages", "reward_returns"),
                    ("costs", "cost_values", last_cost_value, "cost_advantages", "cost_returns")]:
                targets[adv_key][path], targets[ret_key][path] = path_targets(
                    np.array(steps[signal][path], dtype=np.float32),
                    np.array(steps[values][path], dtype=np.float32), last, gamma, lam)
            path_start = t + 1
            if done:  # only finished episodes are counted (L294-304)
                episodes["episode_returns"].append(ep_ret)
                episodes["episode_costs"].append(ep_cost)
                episodes["episode_lengths"].append(ep_len)
                episodes["episode_discounted_costs"].append(ep_dcost)
            ep_ret, ep_cost, ep_dcost, ep_len, discount = 0.0, 0.0, 0.0, 0, 1.0
        obs = new_obs

    out = {key: np.array(steps[key], dtype=np.float32) for key in
           ["observations", "actions", "log_probs", "means", "log_stds"]}
    # CPOBuffer.get: reward advantages normalised over the whole epoch (std
    # over n, not n - 1: mpi_statistics_scalar); cost advantages only centred.
    adv = targets["reward_advantages"]
    targets["reward_advantages"] = (adv - adv.mean()) / (adv.std() + EPS)
    targets["cost_advantages"] = targets["cost_advantages"] - targets["cost_advantages"].mean()
    out.update(targets)
    out.update({key: np.array(values, dtype=np.float64) for key, values in episodes.items()})
    out["average_cost"] = float(np.mean(steps["costs"]))  # cum_cost / steps (L318)
    return out


def update(net, penalty_param, pi_optimizer, value_optimizer, batch, clip_ratio,
           ent_coef, pi_iters, target_kl, kl_margin, vf_iters):
    """The policy and critic part of his PPOLag.update (ppo_lag.py L196-225;
    the λ step before it is separate): up to pi_iters full-batch policy
    steps (PPOAgent.update_pi, agents.py), then vf_iters full-batch critic
    steps (L214-216).

    Policy loss = −(surr_adv + ent_coef·entropy − λ·surr_cost) / (1 + λ),
    λ = softplus(penalty_param) (L160-175):
      surr_adv  = mean(min(ratio·A, (1 ± clip)·A)), + if A > 0, − if not
                  (his clipped_adv: the same as PPO's ratio clipping);
      surr_cost = mean(ratio·A_cost), not clipped;
      ratio     = π_new(a) / π_old(a), with the rollout's stored log π_old.
    pi_optimizer must hold the policy parameters AND penalty_param: his
    minimize(pi_loss) has no variable list, so it moves p too (methods.md
    §5, (f)). Each policy step's KL(new ‖ old) is the one from before that
    step; he stops after a step whose pre-step KL > kl_margin · target_kl
    ((g)). value_optimizer holds both critics; loss = MSE(reward target)
    + MSE(cost target), both summed into one step.

    batch: collect_epoch's dict. Returns the policy steps taken, the last
    KL his loop saw, and the losses before the update (his LossPi,
    SurrCost, LossV, LossVC, Entropy)."""
    t = lambda key: torch.as_tensor(batch[key], dtype=torch.float32)
    obs, actions = t("observations"), t("actions")
    adv, cost_adv = t("reward_advantages"), t("cost_advantages")
    old_log_probs, old_means, old_log_stds = t("log_probs"), t("means"), t("log_stds")
    reward_returns, cost_returns = t("reward_returns"), t("cost_returns")
    min_adv = torch.where(adv > 0, (1 + clip_ratio) * adv, (1 - clip_ratio) * adv)

    def policy_loss():
        mean = net.mean(obs)
        ratio = (gaussian_log_likelihood(actions, mean, net.log_std) - old_log_probs).exp()
        surr_adv = torch.minimum(ratio * adv, min_adv).mean()
        surr_cost = (ratio * cost_adv).mean()
        entropy = gaussian_entropy(net.log_std)
        penalty = torch.nn.functional.softplus(penalty_param)
        loss = -(surr_adv + ent_coef * entropy - penalty * surr_cost) / (1 + penalty)
        kl = gaussian_kl(mean, net.log_std, old_means, old_log_stds)
        return loss, kl, surr_cost, entropy

    def value_losses():
        return (((reward_returns - net.reward_value(obs)) ** 2).mean(),
                ((cost_returns - net.cost_value(obs)) ** 2).mean())

    with torch.no_grad():
        loss, _, surr_cost, entropy = policy_loss()
        v_loss, vc_loss = value_losses()
    stats = {"loss_pi": loss.item(), "surr_cost": surr_cost.item(), "loss_v": v_loss.item(),
             "loss_vc": vc_loss.item(), "entropy": entropy.item()}

    for i in range(pi_iters):
        loss, kl, _, _ = policy_loss()
        pi_optimizer.zero_grad()
        loss.backward()
        pi_optimizer.step()
        if kl.item() > kl_margin * target_kl:  # pre-step KL, (g)
            break
    stats.update({"pi_steps": i + 1, "kl": kl.item()})

    for _ in range(vf_iters):
        v_loss, vc_loss = value_losses()
        value_optimizer.zero_grad()
        (v_loss + vc_loss).backward()
        value_optimizer.step()
    return stats


class PenaltyRay:
    """λ = softplus(p) (ppo_lag.py L142-159, penalty_param_loss=True).

    p starts at softplus⁻¹(initial_value) (L146, which floors e^x − 1 at
    1e-8). update(c): one Adam step (his MpiAdamOptimizer, ε 1e-8) on
    loss = −p · (c − budget): the gradient is on p itself, not on
    softplus(p); p has no floor. The caller decides what c is (the epoch's
    mean episode reward with bug (8), a cost without). p is also trained by
    the policy optimiser ((f), methods.md §5): the agent puts self.param in
    it too."""

    def __init__(self, initial_value, learning_rate, budget):
        p_init = float(np.log(max(np.exp(initial_value) - 1, 1e-8)))
        self.param = nn.Parameter(torch.tensor(p_init, dtype=torch.float32))
        self.budget = budget
        self.optimizer = torch.optim.Adam([self.param], lr=learning_rate, eps=1e-8)

    @property
    def value(self):
        return torch.nn.functional.softplus(self.param).item()

    def update(self, c):
        loss = -self.param * (c - self.budget)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


@dataclass
class PPOLagrangianRaySettings:
    """Defaults: his PPOLag's own (ppo_lag.py, agents.py PPOAgent), which
    configs/hc.json leaves unset except n_steps, gamma, lam, clip_ratio,
    ent_coef and budget (beta)."""
    n_steps: int = 4000              # steps_per_epoch (L129)
    gamma: float = 0.99              # discount_factor (reward and cost, L127)
    lam: float = 0.97                # gae_lambda (reward and cost, L128)
    clip_ratio: float = 0.2          # ppo_clip_param
    ent_coef: float = 0.0            # ppo_entropy_coef
    pi_lr: float = 5e-4              # PPOAgent pi_lr (hc.json's learning_rate is not passed)
    pi_iters: int = 80               # PPOAgent pi_iters
    kl_margin: float = 1.2           # PPOAgent kl_margin
    target_kl: float = 0.01          # L181
    vf_lr: float = 1e-3              # L187
    vf_iters: int = 80               # L214
    penalty_initial_value: float = 1.0   # penalty_init (L145)
    penalty_learning_rate: float = 0.05  # penalty_lr (L158)
    budget: float = 0.0              # his beta: per EPISODE, not per step as ours
    lambda_reads_reward: bool = False    # bug (8): λ's c = mean episode reward
    logp_from_other_draw: bool = False   # bug (9): stored log π from another draw
    seed: int = 0


class PPOLagrangianRay:
    """His PPOLag (module docstring). λ's c: the epoch's mean episode cost
    over finished episodes (Ray et al.'s EpCost; decisions.md 2026-11-10),
    or with lambda_reads_reward the mean episode reward (his bug (8))."""

    def __init__(self, observation_space, action_space, settings=None):
        self.settings = settings or PPOLagrangianRaySettings()
        s = self.settings
        self.action_space = action_space
        # Seeded before the networks are built, so every agent with this
        # seed starts from the same weights (his tf.random.set_seed(seed) in
        # every constructor, L113).
        np.random.seed(s.seed)
        torch.manual_seed(s.seed)
        self.net = ActorTwoCriticsRay(observation_space, action_space)
        self.penalty = PenaltyRay(s.penalty_initial_value, s.penalty_learning_rate, s.budget)
        # p is in the policy optimiser too: (f), methods.md §5.
        self.pi_optimizer = torch.optim.Adam(self.net.policy_parameters() + [self.penalty.param],
                                             lr=s.pi_lr, eps=1e-8)
        self.value_optimizer = torch.optim.Adam(self.net.value_parameters(), lr=s.vf_lr, eps=1e-8)

    def train(self, envs, total_steps, callback=None):
        """Epochs of n_steps until at least total_steps steps (his icl.py
        calls train() ppo_epochs times). Each epoch (L227-322): collect,
        λ step, policy + critic update. envs: one-copy vector env as in
        collect_epoch; seeded on its first reset only (the envs remember
        it, as for our other agents: decisions.md 2026-11-04). Returns one
        row per epoch; callback(row) after each."""
        s = self.settings
        if not getattr(envs, "seeded", False):
            envs.reset(seed=s.seed)
            envs.seeded = True
        recent_returns = deque(maxlen=100)  # as our agents' ep_rew_mean
        log, steps = [], 0
        while steps < total_steps:
            batch = collect_epoch(self.net, envs, s.n_steps, s.gamma, s.lam, s.logp_from_other_draw)
            steps += s.n_steps
            if len(batch["episode_returns"]) == 0:
                # His train() fails here too (no avg_env_reward key, L314).
                raise ValueError(f"no episode finished in an epoch of {s.n_steps} steps")
            episode_reward = float(batch["episode_returns"].mean())
            episode_cost = float(batch["episode_costs"].mean())
            penalty = self.penalty.value  # before this epoch's λ step, as his logged Penalty
            self.penalty.update(episode_reward if s.lambda_reads_reward else episode_cost)
            stats = update(self.net, self.penalty.param, self.pi_optimizer, self.value_optimizer,
                           batch, s.clip_ratio, s.ent_coef, s.pi_iters, s.target_kl,
                           s.kl_margin, s.vf_iters)
            recent_returns.extend(batch["episode_returns"].tolist())
            log.append({"steps": steps, "ep_rew_mean": float(np.mean(recent_returns)),
                        "average_cost": batch["average_cost"], "penalty": penalty,
                        "episode_reward": episode_reward, "episode_cost": episode_cost,
                        "episode_discounted_cost": float(batch["episode_discounted_costs"].mean()),
                        **stats})
            if callback is not None:
                callback(log[-1])
        return log

    def policy(self, obs):
        """One raw observation in, one sampled action out (his
        PPOPolicyWithCost.act: a draw of the pi op), clipped to the action
        bounds as his env clips it (GymEnvironment.step)."""
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = self.net.sample(o)[0].numpy()
        return np.clip(action, self.action_space.low, self.action_space.high)
