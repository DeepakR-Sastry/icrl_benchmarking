"""PPO: maximises reward only.

Follows Malik et al.'s fork of stable-baselines3 0.9.0 (github.com/
shehryar-malik/icrl, commit f28214e): stable_baselines3/ppo/ppo.py and the
SB3 files it relies on, cited where each piece is used."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Independent, Normal


class ActorCritic(nn.Module):
    """Policy and value networks, no shared layers.

    SB3 common/policies.py ActorCriticPolicy (default net_arch
    [dict(pi=[64, 64], vf=[64, 64])], Tanh, orthogonal init) and
    common/distributions.py (action heads). Observations must be a Box;
    they are flattened and cast to float (common/preprocessing.py)."""

    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))

        self.policy_body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.value_body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.value_head = nn.Linear(64, 1)

        self.discrete = isinstance(action_space, spaces.Discrete)
        if self.discrete:
            self.action_head = nn.Linear(64, int(action_space.n))  # logits
        else:
            action_dim = int(np.prod(action_space.shape))
            self.action_head = nn.Linear(64, action_dim)  # Gaussian mean
            self.log_std = nn.Parameter(torch.zeros(action_dim))  # not state-dependent

        # Orthogonal weights, zero biases; small gain on the action head so the
        # first policy is close to uniform (SB3 policies.py ActorCriticPolicy._build).
        for module, gain in [
            (self.policy_body, np.sqrt(2)),
            (self.value_body, np.sqrt(2)),
            (self.action_head, 0.01),
            (self.value_head, 1.0),
        ]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=gain)
                    nn.init.zeros_(layer.bias)

    def distribution(self, obs):
        """The action distribution for a batch of observations."""
        out = self.action_head(self.policy_body(obs.float()))
        if self.discrete:
            return Categorical(logits=out)
        # Independent(..., 1) treats the action dimensions as one joint action:
        # log_prob and entropy are summed over them, as in SB3's DiagGaussian.
        return Independent(Normal(out, self.log_std.exp()), 1)

    def value(self, obs):
        """Predicted future reward, one number per observation."""
        return self.value_head(self.value_body(obs.float())).squeeze(-1)


def compute_gae(rewards, values, episode_starts, last_value, last_done, gamma, gae_lambda):
    """GAE advantages and returns for one rollout (SB3 common/buffers.py
    RolloutBuffer.compute_returns_and_advantage).

    rewards, values, episode_starts: (n_steps, n_envs). episode_starts[t] = 1
    if the observation at step t is the first of an episode (SB3 0.9.0 calls
    this `dones`). last_value, last_done: (n_envs,), for the state after the
    last step. Returns (advantages, returns), both (n_steps, n_envs)."""
    n_steps = len(rewards)
    advantages = np.zeros_like(rewards)
    last_gae_lam = 0.0
    for step in reversed(range(n_steps)):
        if step == n_steps - 1:
            next_non_terminal = 1.0 - last_done
            next_value = last_value
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1]
            next_value = values[step + 1]
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[step] = last_gae_lam
    returns = advantages + values
    return advantages, returns


class RunningMeanStd:
    """Running mean and variance of a stream of batches (SB3 0.9.0
    common/running_mean_std.py). Starts at mean 0, variance 1, count 1e-4.
    Copy of ppo_lagrangian.py's (docs/design.md: agents share no code)."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, batch):
        """Merge a batch (first axis = samples) into the running statistics."""
        batch_mean, batch_var, batch_count = np.mean(batch, axis=0), np.var(batch, axis=0), batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_2 = self.var * self.count + batch_var * batch_count + np.square(delta) * self.count * batch_count / total
        self.var = m_2 / total
        self.count = total


class Normalizer:
    """Malik's VecNormalizeWithCost with cost normalisation off (SB3 0.9.0
    fork, common/vec_env/vec_normalize.py; his gail.py train env), moved
    inside the agent (docs/decisions.md, 2026-10-24). Copy of
    ppo_lagrangian.py's Normalizer without the cost. Each switch off =
    numbers passed through unchanged.

    Observations: (obs - mean) / std, clipped to ±10. Rewards: divided by
    the std of a running discounted return (mean not subtracted), clipped
    to ±10. Statistics change only in reset() and step(); normalize_obs()
    and unnormalize_obs() never change them."""

    CLIP, EPSILON = 10.0, 1e-8

    def __init__(self, obs_shape, normalize_obs, normalize_reward, gamma):
        self.normalize_obs_on = normalize_obs
        self.normalize_reward_on = normalize_reward
        self.gamma = gamma
        self.obs_rms = RunningMeanStd(obs_shape)
        self.reward_rms = RunningMeanStd()

    def reset(self, n_envs):
        """Start of train(): every env copy restarts its episode. As Malik's
        reset(), the return statistics also take in one 0 per copy."""
        self.reward_return = np.zeros(n_envs)
        self.reward_rms.update(self.reward_return)

    def normalize_obs(self, obs):
        if not self.normalize_obs_on:
            return obs
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.EPSILON),
                       -self.CLIP, self.CLIP)

    def unnormalize_obs(self, obs):
        """Back to raw units with the current statistics (Malik's
        VecNormalize.unnormalize_obs; values clipped at ±10 stay clipped)."""
        if not self.normalize_obs_on:
            return obs
        return obs * np.sqrt(self.obs_rms.var + self.EPSILON) + self.obs_rms.mean

    def step(self, obs, rewards, dones):
        """Update the statistics with one step of every copy, then rescale it
        (Malik's step_wait: obs first, then reward). dones: the copies whose
        episode ended; their running returns restart at 0."""
        self.obs_rms.update(obs)
        obs = self.normalize_obs(obs)
        running_return = self.reward_return * self.gamma + rewards
        self.reward_rms.update(running_return)
        running_return[dones] = 0
        self.reward_return = running_return
        if self.normalize_reward_on:
            rewards = np.clip(rewards / np.sqrt(self.reward_rms.var + self.EPSILON), -self.CLIP, self.CLIP)
        return obs, rewards


def collect_rollout(net, normalizer, envs, obs, episode_starts, n_steps, gamma, gae_lambda):
    """Run the policy for n_steps in every env copy and compute GAE (SB3
    common/on_policy_algorithm.py collect_rollouts).

    envs: gymnasium vector env with autoreset_mode=SAME_STEP. normalizer
    rescales each step as it comes in; the rollout keeps the rescaled
    observations and rewards (what the networks learn from). obs (rescaled)
    and episode_starts are where the previous rollout stopped (envs are not
    reset between rollouts). Returns (rollout, obs, episode_starts): rollout is a
    dict of arrays, each (n_steps, n_envs, ...), plus "episode_returns": the
    total reward of each episode that finished during the rollout (needs each
    env copy wrapped in gymnasium's RecordEpisodeStatistics, SB3's Monitor;
    empty otherwise). The other two are where the next rollout continues."""
    device = next(net.parameters()).device
    action_space = envs.single_action_space
    rollout = {key: [] for key in
               ["observations", "actions", "rewards", "episode_starts", "values", "log_probs"]}
    episode_returns = []

    for _ in range(n_steps):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=device)
            dist = net.distribution(obs_tensor)
            actions = dist.sample()
            values = net.value(obs_tensor)
            log_probs = dist.log_prob(actions)
        actions = actions.cpu().numpy()

        # The env gets clipped actions; the rollout keeps the sampled ones.
        env_actions = actions
        if isinstance(action_space, spaces.Box):
            env_actions = np.clip(actions, action_space.low, action_space.high)
        new_obs, rewards, terminated, truncated, infos = envs.step(env_actions)
        dones = np.logical_or(terminated, truncated)
        new_obs, rewards = normalizer.step(new_obs, rewards, dones)
        # With SAME_STEP resets, a finished episode's info is under "final_info".
        final = infos.get("final_info", {})
        if "episode" in final:
            episode_returns.extend(final["episode"]["r"][final["_episode"]].tolist())

        rollout["observations"].append(obs)
        rollout["actions"].append(actions)
        rollout["rewards"].append(rewards)
        rollout["episode_starts"].append(episode_starts)
        rollout["values"].append(values.cpu().numpy())
        rollout["log_probs"].append(log_probs.cpu().numpy())

        obs = new_obs
        # A time-limit cutoff counts as a true end, as in SB3 0.9.0.
        episode_starts = dones.astype(np.float32)

    rollout = {key: np.stack(arrays) for key, arrays in rollout.items()}
    for key in ["rewards", "episode_starts", "values", "log_probs"]:
        rollout[key] = rollout[key].astype(np.float32)

    # Malik's SB3 0.9.0 bootstraps the last step with the leftover `values` =
    # V(last stored obs), not V(obs). See "Known differences" in docs/agents.md.
    # Kept so the advantages can be recomputed after the rewards change
    # (Malik's `self.extras`, added to his SB3 for GACL).
    rollout["last_value"] = values.cpu().numpy()
    rollout["last_done"] = episode_starts
    rollout["advantages"], rollout["returns"] = compute_gae(
        rollout["rewards"], rollout["values"], rollout["episode_starts"],
        last_value=rollout["last_value"], last_done=rollout["last_done"],
        gamma=gamma, gae_lambda=gae_lambda,
    )
    rollout["episode_returns"] = np.array(episode_returns, dtype=np.float32)
    return rollout, obs, episode_starts


def apply_reward_hook(rollout, reward_hook, normalizer, gamma, gae_lambda):
    """Add reward_hook(observations, actions) to a collected rollout's
    rewards and recompute its advantages and returns, in place (Malik's
    GailCallback.on_rollout_end: rewards += log D, then
    compute_returns_and_advantage again). The hook gets raw-unit
    observations: the stored ones un-normalised with the statistics at the
    end of the rollout, as his callback does; the rewards it adds to are the
    stored (normalised) ones, as his."""
    observations = normalizer.unnormalize_obs(rollout["observations"])
    extra = reward_hook(observations, rollout["actions"])
    assert extra.shape == rollout["rewards"].shape
    rollout["rewards"] += extra  # in place: stays float32, as SB3's buffer
    rollout["advantages"], rollout["returns"] = compute_gae(
        rollout["rewards"], rollout["values"], rollout["episode_starts"],
        rollout["last_value"], rollout["last_done"], gamma, gae_lambda)


def ppo_update(net, optimizer, rollout, n_epochs, batch_size, clip_range,
               ent_coef, vf_coef, max_grad_norm, target_kl):
    """Improve the policy and value networks on one rollout (SB3 0.9.0
    ppo/ppo.py PPO.train). Returns mean losses and stats for logging.

    target_kl=None disables the KL early stop. Not ported (unused in Malik's
    runs): value-function clipping, learning-rate and clip-range schedules."""
    device = next(net.parameters()).device
    n = rollout["rewards"].size  # n_steps * n_envs
    # Flatten (n_steps, n_envs, ...) to (n, ...): minibatches mix steps and envs.
    data = {key: torch.as_tensor(rollout[key].reshape(n, *rollout[key].shape[2:]), device=device)
            for key in ["observations", "actions", "log_probs", "advantages", "returns"]}

    stats = {key: [] for key in
             ["policy_loss", "value_loss", "entropy_loss", "approx_kl", "clip_fraction"]}
    epochs_run = 0
    for _ in range(n_epochs):
        epoch_kls = []
        indices = torch.as_tensor(np.random.permutation(n), device=device)
        for start in range(0, n, batch_size):
            batch = {key: value[indices[start:start + batch_size]] for key, value in data.items()}

            dist = net.distribution(batch["observations"])
            log_prob = dist.log_prob(batch["actions"])
            values = net.value(batch["observations"])

            # Per-minibatch normalisation.
            advantages = batch["advantages"]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Clipped surrogate loss: the ratio can't push the policy more
            # than clip_range away from the policy that collected the rollout.
            ratio = torch.exp(log_prob - batch["log_probs"])
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            value_loss = F.mse_loss(batch["returns"], values)
            entropy_loss = -dist.entropy().mean()
            loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            # Measured with log_prob from before this minibatch's step, as in SB3.
            approx_kl = torch.mean(batch["log_probs"] - log_prob).item()
            epoch_kls.append(approx_kl)
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy_loss"].append(entropy_loss.item())
            stats["approx_kl"].append(approx_kl)
            stats["clip_fraction"].append(torch.mean((torch.abs(ratio - 1) > clip_range).float()).item())

        epochs_run += 1
        # Checked once per epoch, on the epoch's mean.
        if target_kl is not None and np.mean(epoch_kls) > 1.5 * target_kl:
            break

    stats = {key: float(np.mean(values)) for key, values in stats.items()}
    stats["epochs"] = epochs_run
    return stats


@dataclass
class PPOSettings:
    """Defaults: Malik's CLI defaults (icrl/gail.py, icrl/icrl.py argparse).
    n_envs is not a setting: it is the number of copies in the vector env."""
    learning_rate: float = 3e-4
    n_steps: int = 2048  # per env copy, per rollout
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    # Malik's VecNormalize switches (no -dno / -dnr = on). His CLI default is
    # on; ours is off so LapGridWorld configs keep their meaning
    # (decisions.md 2026-10-24).
    normalize_obs: bool = False
    normalize_reward: bool = False
    seed: int = 0
    device: str = "cpu"


class PPO:
    """PPO with a policy and a value network (SB3 0.9.0 ppo/ppo.py PPO).
    Maximises reward; ignores info["cost"]."""

    def __init__(self, observation_space, action_space, settings=None):
        self.settings = settings or PPOSettings()
        self.action_space = action_space
        # Seeded before the network is built, so its initial weights depend
        # only on the seed (SB3 common/utils.py set_random_seed).
        np.random.seed(self.settings.seed)
        torch.manual_seed(self.settings.seed)
        self.net = ActorCritic(observation_space, action_space).to(self.settings.device)
        # One Adam over both networks; eps=1e-5 (SB3 ActorCriticPolicy).
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.settings.learning_rate, eps=1e-5)
        self.normalizer = Normalizer(observation_space.shape, self.settings.normalize_obs,
                                     self.settings.normalize_reward, self.settings.gamma)

    def train(self, envs, total_steps, callback=None, reward_hook=None):
        """Alternate collecting a rollout and updating, until at least
        total_steps env steps (all copies together) have been taken (SB3
        OnPolicyAlgorithm.learn). envs: gymnasium vector env,
        autoreset_mode=SAME_STEP, each copy wrapped in RecordEpisodeStatistics.

        reward_hook(observations, actions) -> extra rewards, all
        (n_steps, n_envs, ...): called after each rollout is collected and
        before the update; its result is added to the rollout's rewards and
        the advantages are recomputed (Malik's GailCallback.on_rollout_end).
        Logged ep_rew_mean stays the env's own reward.

        Resets envs on every call, as Malik's ICRL loop does (learn's default
        reset_num_timesteps=True); the envs are seeded on their first reset only.
        Returns one dict per rollout for logging; callback, if given, is
        also called with each dict as soon as its rollout's update is done
        (as PPOLagrangian.train's)."""
        s = self.settings
        # The envs remember whether they were seeded, not the agent: a new
        # agent on the same envs (ICL's fresh agent each round) continues
        # their random streams (docs/decisions.md 2026-11-04).
        seeded = getattr(envs, "seeded", False)
        obs, _ = envs.reset(seed=None if seeded else s.seed)
        envs.seeded = True
        # The statistics carry over between calls. Reset observations are
        # rescaled but not added to them (Malik's VecNormalize.reset).
        self.normalizer.reset(envs.num_envs)
        obs = self.normalizer.normalize_obs(obs)
        episode_starts = np.zeros(envs.num_envs, dtype=np.float32)
        recent_returns = deque(maxlen=100)  # SB3's ep_info_buffer
        log = []
        steps = 0
        while steps < total_steps:
            rollout, obs, episode_starts = collect_rollout(
                self.net, self.normalizer, envs, obs, episode_starts, s.n_steps, s.gamma, s.gae_lambda)
            steps += s.n_steps * envs.num_envs
            if reward_hook is not None:
                apply_reward_hook(rollout, reward_hook, self.normalizer, s.gamma, s.gae_lambda)
            stats = ppo_update(
                self.net, self.optimizer, rollout, s.n_epochs, s.batch_size, s.clip_range,
                s.ent_coef, s.vf_coef, s.max_grad_norm, s.target_kl)
            recent_returns.extend(rollout["episode_returns"].tolist())
            ep_rew_mean = float(np.mean(recent_returns)) if recent_returns else float("nan")
            log.append({"steps": steps, "ep_rew_mean": ep_rew_mean, **stats})
            if callback is not None:
                callback(log[-1])
        return log

    def policy(self, obs):
        """One raw observation in, one sampled action out (Malik samples
        trajectories with deterministic=False); continuous actions are clipped
        to the action bounds (SB3 BasePolicy.predict). The observation is
        rescaled with the current statistics, which it doesn't change (what
        Malik's sync_envs_normalization achieves for his eval env)."""
        obs = self.normalizer.normalize_obs(obs)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, device=self.settings.device).unsqueeze(0)
            action = self.net.distribution(obs_tensor).sample()[0].cpu().numpy()
        if isinstance(self.action_space, spaces.Box):
            return np.clip(action, self.action_space.low, self.action_space.high)
        return int(action)
