"""The ICRL loop: forward step (agent trains on the learned cost), sample
nominal trajectories, backward step (method updates the constraint).

Follows Malik et al.'s icrl/icrl.py (github.com/shehryar-malik/icrl, commit
f28214e), cited where each piece is used. Order and design: docs/design.md.

GACL has its own loop, run_gacl (Malik's icrl/gail.py): one training run in
which the method is updated after every rollout (docs/decisions.md,
2026-11-01).

ICL has its own loop, run_icl (Gaurav's icl.py): a fresh agent each round,
snapshots saved during training, agent data from a weighted mixture of
all snapshots so far (docs/design.md)."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from icrl_bench.agents.icl_ppo_policy import ICLPPOPolicy
from icrl_bench.core.rollout import rollout
from icrl_bench.core.types import Step, Trajectory
from icrl_bench.eval.icl_metrics import accrual, cmse, nad
from icrl_bench.eval.stage2 import feasible_reward_stats, reward_stats, true_cost_rate


@dataclass
class ICRLLoopSettings:
    """Defaults: Malik's CLI defaults (icrl/icrl.py argparse). Per-env values
    live in configs."""
    n_rounds: int = 100  # -ni
    steps_per_round: int = 1_000_000  # -ft, agent steps per forward step
    n_eval_episodes: int = 10  # hard-coded in icrl.py L248
    # Liu's use_buffer (train_icrl.py L74-77, L385-391; IRLDataQueue): keep
    # every nominal trajectory sampled so far and train the method on
    # len(expert) of them drawn uniformly with replacement. Off = only this
    # round's, as Malik (docs/decisions.md, 2026-11-01).
    nominal_buffer: bool = False
    seed: int = 0


def learned_cost_fn(featurize, method):
    """A cost_fn(obs, action) for CostWrapper: the step is featurized as a
    one-step trajectory and scored by method.cost (Malik passes
    constraint_net.cost_function to his cost wrapper, icrl.py L116, L239).

    It keeps a reference to `method`, so it always uses the current
    constraint. Featurizers never read the reward, so it is set to 0."""
    def cost_fn(obs, action):
        features = featurize(Trajectory([Step(obs, action, reward=0.0)]))
        return float(method.cost(features)[0])
    return cost_fn


def run_icrl(agent, envs, method, featurize, expert, make_nominal_env, make_eval_env,
             true_cost, settings=None, callback=None):
    """Malik's ICRL loop (icrl.py L199-304). Each round: the agent trains on
    the learned cost, nominal trajectories are sampled from it, the method
    updates the constraint, and the agent is graded with the true cost.

    agent: has train(envs, steps) and policy(obs). envs: the agent's vector
    env, a CostWrapper on every copy. expert: list[Trajectory].
    make_nominal_env / make_eval_env: build an unconstrained / constrained
    env; each is built once, as Malik's sampling and eval envs (L64, L68).
    true_cost: used for grading only. Returns one metrics dict per round;
    callback, if given, is also called with each dict as soon as its round
    is done (as PPOLagrangian.train's), so long runs can save as they go."""
    s = settings or ICRLLoopSettings()
    # Round 0's forward step trains on the untrained constraint (icrl.py L116).
    envs.call("set_cost_fn", learned_cost_fn(featurize, method))
    nominal_env, eval_env = make_nominal_env(), make_eval_env()
    store = []  # every nominal trajectory so far (nominal_buffer only)
    rng = np.random.default_rng(s.seed)  # its own stream: changes nothing else
    timesteps = 0
    log = []
    for round_ in range(s.n_rounds):
        # Both envs are seeded once, in round 0; later rounds continue their
        # random streams, so each round starts from new poses (Malik's envs
        # are never reseeded either).
        seed = s.seed if round_ == 0 else None
        # Forward step (L206-213).
        forward = agent.train(envs, s.steps_per_round)
        timesteps += forward[-1]["steps"]

        # As many nominal trajectories as expert ones, sampled actions, no
        # cost wrapper (L216-218).
        nominal = rollout(nominal_env, agent.policy, len(expert), seed)

        # With the buffer, the method's pile is drawn from every round so
        # far, this one included; repeats allowed (Liu's IRLDataQueue.get).
        # Liu caps the store at 500 trajectories; his 30 rounds x 10 never
        # reach it, so there is no cap here.
        pile = nominal
        if s.nominal_buffer:
            store += nominal
            pile = [store[i] for i in rng.integers(len(store), size=len(expert))]

        # Backward step, then hand the new constraint to every env copy
        # (L235-239). Both piles are featurized every round, so they always
        # use the same featurizer state (docs/decisions.md, 2026-10-19).
        backward = method.update([featurize(t) for t in expert], [featurize(t) for t in pile])
        envs.call("set_cost_fn", learned_cost_fn(featurize, method))

        # Grading with the true cost (L243-249).
        evaluation = rollout(eval_env, agent.policy, s.n_eval_episodes, seed)
        reward_mean, reward_std = reward_stats(evaluation)
        # Liu's evaluation: reward before the first violation, and how many
        # episodes broke the rule at all (docs/decisions.md, 2026-11-01).
        feasible_mean, feasible_std, violating = feasible_reward_stats(evaluation, true_cost)
        log.append({
            "round": round_,
            "timesteps": timesteps,
            "true/reward": reward_mean,
            "true/reward_std": reward_std,
            "true/feasible_reward": feasible_mean,
            "true/feasible_reward_std": feasible_std,
            "true/violating_episodes": violating,
            "true/cost": true_cost_rate(nominal, true_cost),
            **{f"forward/{k}": v for k, v in forward[-1].items()},
            **{f"backward/{k}": v for k, v in backward.items()},
        })
        if callback is not None:
            callback(log[-1])
    return log


@dataclass
class GACLLoopSettings:
    """Defaults: Malik's CLI defaults (icrl/gail.py argparse, SB3
    EvalCallback). Per-env values live in configs."""
    total_steps: int = 1_000_000  # -t, agent steps in the one training run
    eval_every: int = 2048  # -ee, in vector-env steps (all copies step together)
    n_eval_episodes: int = 5  # EvalCallback default
    seed: int = 0


def run_gacl(agent, envs, method, featurize, expert, make_eval_env, true_cost,
             settings=None, callback=None):
    """Malik's GACL (gail.py L189-191 + GailCallback): one agent.train run;
    after each rollout the method is updated on that rollout and its log D
    is added to the rollout's rewards (agent.train's reward_hook).

    agent: has train(envs, steps, callback, reward_hook) and policy(obs).
    envs: the agent's vector env; its cost is set to zero and never read. expert:
    list[Trajectory]. make_eval_env builds the constrained env, once.
    true_cost: used for grading only. Returns one metrics dict per rollout;
    callback, if given, is also called with each dict as soon as it is
    complete."""
    s = settings or GACLLoopSettings()
    # Malik's GACL envs have no cost wrapper; ours carry a zero cost, which
    # PPO never reads (each loop sets its copies' cost first, as run_icrl).
    envs.call("set_cost_fn", lambda obs, action: 0.0)
    n_envs = envs.num_envs
    eval_env = make_eval_env()
    pending = {}  # the hook's metrics, waiting for the PPO update's row
    log = []
    n_evaluations = 0

    def hook(observations, actions):
        # GailCallback._on_rollout_end. Step-major order, as his flatten:
        # (step 0, env 0), (step 0, env 1), ...; quirk A keeps the first rows.
        n_steps = actions.shape[0]
        steps = [Step(observations[t, e], actions[t, e], reward=0.0)
                 for t in range(n_steps) for e in range(n_envs)]
        nominal = Trajectory(steps)
        features = featurize(nominal)
        # Both piles featurized each time, so they always use the same
        # featurizer state (as run_icrl, docs/decisions.md 2026-10-19).
        backward = method.update([featurize(t) for t in expert], [features])
        pending.clear()
        pending.update({f"backward/{k}": v for k, v in backward.items()})
        # His eval/mean_cost: true cost per step of this training rollout.
        pending["true/cost"] = true_cost_rate([nominal], true_cost)
        # Rewards from the updated D (his reward_function after train).
        return method.reward(features).reshape(n_steps, n_envs)

    def on_row(row):
        # His EvalCallback evaluates when its call count (one per vector
        # step) hits a multiple of eval_every, i.e. during a rollout, with
        # the policy of the last update. So after this update, evaluate if
        # the next rollout would contain such a step and training goes on.
        nonlocal n_evaluations
        done = row["steps"] // n_envs  # vector steps so far
        next_end = done + done // (len(log) + 1)  # + one rollout's vector steps
        reward_mean = reward_std = feasible_mean = feasible_std = violating = float("nan")
        if row["steps"] < s.total_steps and next_end // s.eval_every > done // s.eval_every:
            # The eval env is seeded at its first use only, as in run_icrl.
            seed = s.seed if n_evaluations == 0 else None
            evaluation = rollout(eval_env, agent.policy, s.n_eval_episodes, seed)
            reward_mean, reward_std = reward_stats(evaluation)
            feasible_mean, feasible_std, violating = feasible_reward_stats(evaluation, true_cost)
            n_evaluations += 1
        log.append({
            "rollout": len(log),
            "timesteps": row["steps"],
            "true/reward": reward_mean,
            "true/reward_std": reward_std,
            "true/feasible_reward": feasible_mean,
            "true/feasible_reward_std": feasible_std,
            "true/violating_episodes": violating,
            **pending,
            **{f"forward/{k}": v for k, v in row.items()},
        })
        if callback is not None:
            callback(log[-1])

    agent.train(envs, s.total_steps, callback=on_row, reward_hook=hook)
    return log


@dataclass
class ICLLoopSettings:
    """Defaults: Gaurav's Gridworld (A) config (configs/gridworldA.json) where
    the setting means the same; per-env values live in configs."""
    n_rounds: int = 10  # outer_epochs
    steps_per_round: int = 1_000_000  # agent steps per forward step
    # A snapshot every this many agent callback rows (one per rollout). His
    # mix_save_epoch counts his PPO epochs: equal only with his agent.
    snapshot_every: int = 250
    updates_per_round: int = 20  # updates_per_epoch, icl.py's outer count (quirk (b))
    pi_episodes: int = 50  # agent episodes per update (cost_adjustment.py L68)
    final_round: bool = True  # icl.py L95-102: one more fresh agent, no snapshots
    # Who plays the backward-step episodes: "snapshots" (the algorithm; his
    # CPPO envs) or "untrained_policy" (his PPOLag envs, bug (7): a fresh,
    # never-trained ICLPPOPolicy each round; docs/decisions.md 2026-11-08).
    agent_data: str = "snapshots"
    n_eval_episodes: int = 10
    seed: int = 0


def run_icl(make_agent, envs, method, featurize, expert, make_nominal_env, make_eval_env,
            true_cost, settings=None, callback=None, metric_grid=None):
    """Gaurav's ICL loop (icl.py L29-102). Each round: a fresh agent trains on
    the learned cost, saving snapshots (each with its dissimilarity to the
    expert) as it goes; then the method is updated on episodes from
    snapshots of every round so far, drawn in proportion to their
    dissimilarity; then the round's agent is graded with the true cost.

    make_agent: builds a fresh agent (train(envs, steps, callback),
    policy(obs)). Other arguments as run_icrl. Returns one metrics dict per
    round (plus the final round's, which has no backward/* keys); callback,
    if given, is also called with each dict as soon as it is complete.

    metric_grid: a MetricGrid (core/types.py) or None. If given, every row
    gets Gaurav's metrics (eval/icl_metrics.py): true/cmse, the learned
    cost against the true cost on the grid (his cost_comparison), and
    true/nad, NaN except on the last row (his ppo_accrual_comparison: the
    last agent's len(expert) episodes on the nominal env vs the expert).
    Every row has both keys: a CSV's columns are its first row's keys."""
    s = settings or ICLLoopSettings()
    if s.agent_data not in ("snapshots", "untrained_policy"):
        raise ValueError(f"agent_data must be 'snapshots' or 'untrained_policy', "
                         f"not {s.agent_data!r}")
    # NumPy's global stream: our agents shuffle minibatches with it. On this
    # path no constructor seeds it (build() builds no agent; fresh_agent
    # restores the state from before make_agent), so seed it here, once, as
    # his code seeds everything at start-up (tools/data/configuration.py
    # L31; docs/decisions.md 2026-11-08).
    np.random.seed(s.seed)
    expert_features = [featurize(t) for t in expert]
    method.fit_flow(expert_features)  # icl.py L29-42, before round 1
    # Round 1 trains on the untrained cost net (icl.py L20-21).
    envs.call("set_cost_fn", learned_cost_fn(featurize, method))
    nominal_env, eval_env = make_nominal_env(), make_eval_env()
    rng = np.random.default_rng(s.seed)  # snapshot draws: its own stream
    seeded = {"nominal": False, "eval": False}

    def play(env_name, policy, n):
        # Each env is seeded on its first use only; later episodes continue
        # its random stream (as run_icrl's envs).
        env = nominal_env if env_name == "nominal" else eval_env
        seed = None if seeded[env_name] else s.seed
        seeded[env_name] = True
        return rollout(env, policy, n, seed)

    def fresh_agent():
        # Our constructors reseed torch and numpy globally. Put the streams
        # back, so only the starting weights repeat each round and every
        # other draw continues, as his code (decisions.md 2026-11-04).
        torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
        agent = make_agent()
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)
        return agent

    if metric_grid is not None:
        grid = metric_grid
        grid_true = [true_cost(step.state, step.action) for step in grid.trajectory.steps]
        grid_features = featurize(grid.trajectory)

        def coordinates(trajectories):
            return [grid.coordinate(step.state) for t in trajectories for step in t.steps]

        expert_accrual = accrual(coordinates(expert), grid.values,
                                 grid.left_gap, grid.right_gap)

    def icl_metrics(agent, last):
        if metric_grid is None:
            return {}
        row = {"true/cmse": cmse(grid_true, method.cost(grid_features)), "true/nad": np.nan}
        if last:  # played after everything else, so no other number changes
            episodes = play("nominal", agent.policy, len(expert))
            row["true/nad"] = nad(expert_accrual, accrual(
                coordinates(episodes), grid.values, grid.left_gap, grid.right_gap))
        return row

    snapshots, dissimilarities = [], []  # every round's, never dropped
    timesteps = 0
    log = []

    def grade(agent, round_, forward):
        evaluation = play("eval", agent.policy, s.n_eval_episodes)
        reward_mean, reward_std = reward_stats(evaluation)
        feasible_mean, feasible_std, violating = feasible_reward_stats(evaluation, true_cost)
        return {
            "round": round_,
            "timesteps": timesteps,
            "true/reward": reward_mean,
            "true/reward_std": reward_std,
            "true/feasible_reward": feasible_mean,
            "true/feasible_reward_std": feasible_std,
            "true/violating_episodes": violating,
            "snapshots": len(snapshots),
            **{f"forward/{k}": v for k, v in forward[-1].items()},
        }

    for round_ in range(s.n_rounds):
        agent = fresh_agent()  # icl.py L56-59
        n_rows = 0

        def on_row(row):
            # constrained_ppo.py L98-126: every snapshot_every rows, keep a
            # copy of the agent and its mean dissimilarity over len(expert)
            # fresh episodes of that copy.
            nonlocal n_rows
            n_rows += 1
            if n_rows % s.snapshot_every == 0:
                copy = deepcopy(agent)
                episodes = play("nominal", copy.policy, len(expert))
                snapshots.append(copy)
                dissimilarities.append(
                    float(np.mean(method.dissimilarity([featurize(t) for t in episodes]))))

        forward = agent.train(envs, s.steps_per_round, callback=on_row)
        timesteps += forward[-1]["steps"]
        if s.agent_data == "snapshots" and not snapshots:
            # His code would play an untrained policy here; refuse instead.
            raise ValueError(f"no snapshot after round {round_}: snapshot_every "
                             f"({s.snapshot_every}) > agent rows per round ({n_rows})")
        if s.agent_data == "untrained_policy":
            # Built where his CostAdjustment is (icl.py L84), from the
            # continuing random stream; it plays every episode of this
            # round's updates (cost_adjustment.py L24, L75-76).
            untrained = ICLPPOPolicy(nominal_env.observation_space, nominal_env.action_space)

        # icl.py L84-87: each update draws its own agent data
        # (cost_adjustment.py L58-77): pi_episodes episodes, each by a
        # snapshot drawn with probability ∝ its dissimilarity.
        for _ in range(s.updates_per_round):
            if s.agent_data == "snapshots":
                p = np.array(dissimilarities) / np.sum(dissimilarities)
                drawn = rng.choice(len(snapshots), size=s.pi_episodes, p=p)
                nominal = [play("nominal", snapshots[i].policy, 1)[0] for i in drawn]
            else:
                nominal = play("nominal", untrained.policy, s.pi_episodes)
            backward = method.update(expert_features, [featurize(t) for t in nominal])
        envs.call("set_cost_fn", learned_cost_fn(featurize, method))

        log.append({
            **grade(agent, round_, forward),
            # Per-step true-cost rate of the last update's agent data.
            "true/mixture_cost": true_cost_rate(nominal, true_cost),
            **{f"backward/{k}": v for k, v in backward.items()},
            **icl_metrics(agent, last=not s.final_round and round_ == s.n_rounds - 1),
        })
        if callback is not None:
            callback(log[-1])

    if s.final_round:
        # icl.py L95-102: train once more on the final cost (no_mix=True: no
        # snapshots, no update after it). The policy his runs report.
        agent = fresh_agent()
        forward = agent.train(envs, s.steps_per_round)
        timesteps += forward[-1]["steps"]
        log.append({**grade(agent, s.n_rounds, forward), **icl_metrics(agent, last=True)})
        if callback is not None:
            callback(log[-1])
    return log
