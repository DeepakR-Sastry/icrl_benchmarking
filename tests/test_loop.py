from copy import deepcopy

import numpy as np
import pytest
import torch

from icrl_bench.core.loop import (GACLLoopSettings, ICLLoopSettings, ICRLLoopSettings,
                                  learned_cost_fn, run_gacl, run_icl, run_icrl)
from icrl_bench.core.rollout import rollout
from icrl_bench.core.types import MetricGrid, Step, Trajectory
from icrl_bench.eval.icl_metrics import accrual, nad
from icrl_bench.envs.lap_grid_world import BACKWARD, FORWARD, LapGridWorld, true_cost
from icrl_bench.featurizers.lap_grid_world import featurize
from icrl_bench.methods.maximum_entropy import MaximumEntropy, MaximumEntropySettings


def always_forward(obs):
    return FORWARD


def test_learned_cost_fn_scores_the_step_it_is_given():
    method = MaximumEntropy(3, MaximumEntropySettings(hidden_sizes=(20,)))
    cost_fn = learned_cost_fn(featurize, method)
    obs = np.array([-0.9], dtype=np.float32)
    for action in (FORWARD, BACKWARD):
        direct = method.cost(featurize(Trajectory([Step(obs, action, 0.0)])))[0]
        assert cost_fn(obs, action) == pytest.approx(direct)


def test_learned_cost_fn_follows_the_method_after_an_update():
    method = MaximumEntropy(3, MaximumEntropySettings(hidden_sizes=(20,), learning_rate=0.01, backward_iters=20))
    cost_fn = learned_cost_fn(featurize, method)  # built before the update
    obs = np.array([-0.9], dtype=np.float32)
    before = cost_fn(obs, BACKWARD)
    forward_row = np.array([[-0.9, 1.0, 0.0]], dtype=np.float32)
    backward_row = np.array([[-0.9, 0.0, 1.0]], dtype=np.float32)
    method.update([forward_row], [backward_row])
    assert cost_fn(obs, BACKWARD) != before
    assert cost_fn(obs, BACKWARD) == pytest.approx(method.cost(backward_row)[0])


class Recorder:
    """Stand-ins for the agent, the method and the vector env: they record
    what the loop asks of them, in order, and do no learning."""

    def __init__(self):
        self.events = []
        self.nominal_sizes = []

    # agent
    def train(self, envs, total_steps):
        self.events.append("train")
        return [{"steps": total_steps, "penalty": 1.0}]

    def policy(self, obs):
        return FORWARD

    # method
    def update(self, expert, nominal):
        self.events.append("update")
        self.nominal_sizes.append(len(nominal))
        return {"cn_loss": 0.5}

    def cost(self, features):
        return np.zeros(len(features))

    # vector env
    def call(self, name, cost_fn):
        assert name == "set_cost_fn"
        self.events.append("set cost")


def test_run_icrl_follows_maliks_order_and_logs_each_round():
    expert = rollout(LapGridWorld(), always_forward, 3, seed=0)
    stub = Recorder()
    log = run_icrl(stub, stub, stub, featurize, expert, LapGridWorld,
                   lambda: LapGridWorld(constrained=True), true_cost,
                   ICRLLoopSettings(n_rounds=2, steps_per_round=100, n_eval_episodes=2))

    assert stub.events == ["set cost", "train", "update", "set cost", "train", "update", "set cost"]
    assert stub.nominal_sizes == [3, 3]  # as many nominal trajectories as expert ones
    assert [row["round"] for row in log] == [0, 1]
    assert [row["timesteps"] for row in log] == [100, 200]
    # An always-forward agent: laps in the constrained world, no violations.
    assert log[0]["true/reward"] == 60.0 and log[0]["true/reward_std"] == 0.0
    # Never backward: the whole reward is feasible, no episode breaks the rule.
    assert log[0]["true/feasible_reward"] == 60.0 and log[0]["true/violating_episodes"] == 0.0
    assert log[0]["true/cost"] == 0.0
    assert log[0]["forward/penalty"] == 1.0 and log[0]["backward/cn_loss"] == 0.5


class GACLRecorder:
    """Stand-ins for the agent, the method and the vector env in run_gacl.
    The agent's rollouts have obs[t, e] = 10t + e (so the order is visible)
    and actions forward in env 0, backward in env 1; it hands each to the
    reward hook, then reports the update's row to the callback."""

    def __init__(self, n_steps, n_envs):
        self.n_steps, self.num_envs = n_steps, n_envs
        self.nominal, self.extra_rewards = [], []

    # agent
    def train(self, envs, total_steps, callback, reward_hook):
        t, e = np.meshgrid(np.arange(self.n_steps), np.arange(self.num_envs), indexing="ij")
        obs = (10.0 * t + e)[..., None]
        acs = np.where(e == 0, FORWARD, BACKWARD)
        steps = 0
        while steps < total_steps:
            self.extra_rewards.append(reward_hook(obs, acs))
            steps += self.n_steps * self.num_envs
            callback({"steps": steps, "ep_rew_mean": 0.0})

    def policy(self, obs):
        return FORWARD

    # method
    def update(self, expert, nominal):
        self.nominal.append(nominal)
        return {"disc_loss": 0.7}

    def reward(self, features):
        return features[:, 0]  # the cell: shows which step each reward goes to

    # vector env
    def call(self, name, cost_fn):
        assert name == "set_cost_fn" and cost_fn(None, BACKWARD) == 0.0


def test_run_gacl_flattens_step_major_and_returns_rewards_in_rollout_shape():
    expert = rollout(LapGridWorld(), always_forward, 3, seed=0)
    stub = GACLRecorder(n_steps=3, n_envs=2)
    run_gacl(stub, stub, stub, featurize, expert, lambda: LapGridWorld(constrained=True),
             true_cost, GACLLoopSettings(total_steps=6, n_eval_episodes=1))
    [nominal] = stub.nominal[0]  # one array: the whole rollout
    # (step 0, env 0), (step 0, env 1), (step 1, env 0), ... as Malik's flatten.
    assert nominal[:, 0].tolist() == [0, 1, 10, 11, 20, 21]
    assert nominal[:, 1].tolist() == [1, 0, 1, 0, 1, 0]  # is_forward: env 0 only
    # Each step's reward goes back to its own (step, env) slot.
    assert stub.extra_rewards[0].tolist() == [[0, 1], [10, 11], [20, 21]]


def test_run_gacl_evaluates_when_malik_would_and_logs_each_rollout():
    # 100 vector steps per rollout, eval every 128: after update k the next
    # rollout covers (100k, 100k + 100]. It holds a multiple of 128 for
    # k = 1, 2, 3, 5 but not 4; after the last update (k = 6) training ends.
    expert = rollout(LapGridWorld(), always_forward, 3, seed=0)
    stub = GACLRecorder(n_steps=100, n_envs=2)
    rows = []
    log = run_gacl(stub, stub, stub, featurize, expert, lambda: LapGridWorld(constrained=True),
                   true_cost, GACLLoopSettings(total_steps=1200, eval_every=128, n_eval_episodes=2),
                   callback=rows.append)
    assert rows == log
    assert [row["rollout"] for row in log] == [0, 1, 2, 3, 4, 5]
    assert [row["timesteps"] for row in log] == [200, 400, 600, 800, 1000, 1200]
    evaluated = [bool(np.isfinite(row["true/reward"])) for row in log]
    assert evaluated == [True, True, True, False, True, False]
    # Always forward in the constrained world: 60 every episode.
    assert log[0]["true/reward"] == 60.0 and log[0]["true/reward_std"] == 0.0
    assert log[0]["true/feasible_reward"] == 60.0 and log[0]["true/violating_episodes"] == 0.0
    assert np.isnan(log[3]["true/feasible_reward"])     # not evaluated after update 4
    # True cost of the training rollout: env 1 always steps backward.
    assert log[0]["true/cost"] == 0.5
    assert log[0]["backward/disc_loss"] == 0.7 and log[0]["forward/ep_rew_mean"] == 0.0


class RoundTaggingRecorder(Recorder):
    """Recorder whose agent marks each trajectory with its round: in round k
    it steps backward for the first k steps of an episode, then forward.
    The method keeps the nominal piles it is given."""

    def __init__(self):
        super().__init__()
        self.round, self.piles = -1, []
        self.calls = 0

    def train(self, envs, total_steps):
        self.round += 1
        return super().train(envs, total_steps)

    def policy(self, obs):
        # Every LapGridWorld episode is 200 steps (nominal and eval alike),
        # so the step within the episode is the call count modulo 200.
        step, self.calls = self.calls % 200, self.calls + 1
        return BACKWARD if step < self.round else FORWARD

    def update(self, expert, nominal):
        # The round each trajectory came from = its number of backward steps.
        self.piles.append([int(features[:, 2].sum()) for features in nominal])
        return super().update(expert, nominal)


def run_tagged(nominal_buffer, n_rounds):
    expert = rollout(LapGridWorld(), always_forward, 3, seed=0)
    stub = RoundTaggingRecorder()
    run_icrl(stub, stub, stub, featurize, expert, LapGridWorld,
             lambda: LapGridWorld(constrained=True), true_cost,
             ICRLLoopSettings(n_rounds=n_rounds, steps_per_round=100, n_eval_episodes=1,
                              nominal_buffer=nominal_buffer, seed=0))
    return stub.piles


def test_without_the_buffer_each_round_trains_on_its_own_trajectories():
    assert run_tagged(False, 4) == [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]


def test_the_buffer_draws_from_every_round_so_far_with_repeats():
    # 3 new trajectories per round are added to the store (round k holds
    # 3(k+1)); then 3 indices are drawn, repeats allowed, from a generator
    # seeded with the run seed. Index i belongs to round i // 3.
    rng = np.random.default_rng(0)
    expected = [[int(i) // 3 for i in rng.integers(3 * (k + 1), size=3)] for k in range(4)]
    piles = run_tagged(True, 4)
    assert piles == expected
    assert piles[0] == [0, 0, 0]                         # round 0 has only its own
    assert any(r < k for k, pile in enumerate(piles) for r in pile)   # older rounds come back


# --- run_icl ---------------------------------------------------------------
# One event list for every stand-in: the loop deep-copies agents (snapshots),
# and a copy must still record into the same list.
EVENTS = []


class ICLAgent:
    """Stand-in agent: train calls the callback rows_per_round times (one per
    "rollout") and records two random numbers; the policy always goes in one
    direction."""

    def __init__(self, action, rows_per_round=4):
        self.action = action
        self.rows_per_round = rows_per_round
        self.rows_done = 0
        self.draws = None

    def train(self, envs, total_steps, callback=None):
        EVENTS.append("train")
        self.draws = (torch.rand(1).item(), np.random.rand())
        for _ in range(self.rows_per_round):
            self.rows_done += 1
            if callback is not None:
                callback({"steps": total_steps})
        return [{"steps": total_steps}]

    def policy(self, obs):
        return self.action


class ICLMethod:
    """Stand-in method: dissimilarity = fraction of backward steps (column 2
    of the LapGridWorld features); update records each trajectory's
    dissimilarity."""

    def __init__(self):
        self.piles = []

    def fit_flow(self, expert):
        EVENTS.append("fit flow")

    def dissimilarity(self, trajectories):
        EVENTS.append("dissimilarity")
        return np.array([features[:, 2].mean() for features in trajectories])

    def update(self, expert, nominal):
        EVENTS.append("update")
        self.piles.append([float(features[:, 2].mean()) for features in nominal])
        return {"loss": 0.5}

    def cost(self, features):
        return np.zeros(len(features))


class CostEnvs:
    def call(self, name, cost_fn):
        EVENTS.append("set cost")


def run_icl_stub(make_agent, metric_grid=None, **settings):
    EVENTS.clear()
    expert = rollout(LapGridWorld(), always_forward, 2, seed=0)
    method = ICLMethod()
    log = run_icl(make_agent, CostEnvs(), method, featurize, expert, LapGridWorld,
                  lambda: LapGridWorld(constrained=True), true_cost,
                  ICLLoopSettings(**{"n_rounds": 2, "steps_per_round": 100, "snapshot_every": 2,
                                     "updates_per_round": 2, "pi_episodes": 3,
                                     "n_eval_episodes": 1, **settings}),
                  metric_grid=metric_grid)
    return log, method


def test_run_icl_order_snapshots_and_final_round():
    log, method = run_icl_stub(lambda: ICLAgent(BACKWARD))
    one_round = ["train", "dissimilarity", "dissimilarity", "update", "update", "set cost"]
    # 4 rows, a snapshot every 2 -> 2 dissimilarity calls per round; the
    # final round trains only.
    assert EVENTS == ["fit flow", "set cost", *one_round, *one_round, "train"]
    assert [row["round"] for row in log] == [0, 1, 2]
    assert [row["snapshots"] for row in log] == [2, 4, 4]
    assert [row["timesteps"] for row in log] == [100, 200, 300]
    assert "backward/loss" in log[1] and "backward/loss" not in log[2]
    assert method.piles[0] == [1.0, 1.0, 1.0]  # pi_episodes episodes per update


def test_run_icl_snapshots_are_frozen_copies(monkeypatch):
    import icrl_bench.core.loop as loop
    snapshots = []

    def recording_deepcopy(agent):
        snapshots.append(deepcopy(agent))
        return snapshots[-1]

    monkeypatch.setattr(loop, "deepcopy", recording_deepcopy)
    agents = []
    run_icl_stub(lambda: agents.append(ICLAgent(BACKWARD)) or agents[-1],
                 n_rounds=1, final_round=False)
    # Taken after rows 2 and 4 of the agent's 4: separate objects that kept
    # the row count they had when taken.
    assert [snap.rows_done for snap in snapshots] == [2, 4]
    assert agents[0].rows_done == 4
    assert all(snap is not agents[0] for snap in snapshots)


def test_run_icl_draws_by_dissimilarity_from_every_round():
    # Round 0's agent goes backward (dissimilarity 1), round 1's forward
    # (dissimilarity 0): round 1's agent data must all come from round 0's
    # snapshots, and none from its own.
    actions = iter([BACKWARD, FORWARD, FORWARD])
    log, method = run_icl_stub(lambda: ICLAgent(next(actions)))
    assert method.piles == [[1.0] * 3] * 4  # 2 rounds x 2 updates, all backward
    assert log[1]["snapshots"] == 4


def test_run_icl_restores_the_random_state_after_make_agent():
    agents = []

    def make_agent():
        torch.manual_seed(0)  # as our constructors do
        np.random.seed(0)
        agents.append(ICLAgent(BACKWARD))
        return agents[-1]

    torch.manual_seed(123)
    np.random.seed(123)
    run_icl_stub(make_agent)
    draws = [agent.draws for agent in agents]
    # Expected: one stream per library, continued round after round —
    # torch's from the caller's seed 123, NumPy's from the loop's seed (0;
    # run_icl seeds it at its start). (The stand-ins draw nothing else; the
    # envs and snapshot draws have their own generators.) Without the
    # restore, all three would be the first draws of seed 0.
    torch.manual_seed(123)
    np.random.seed(0)
    expected = [(torch.rand(1).item(), np.random.rand()) for _ in range(3)]
    assert draws == expected


def test_run_icl_numpy_draws_do_not_depend_on_the_state_before_it():
    # Agents that don't seed NumPy themselves (as on the run_icl path, where
    # no constructor runs before make_agent's state is restored).
    def run(numpy_state_before):
        agents = []

        def make_agent():
            agents.append(ICLAgent(BACKWARD))
            return agents[-1]

        np.random.seed(numpy_state_before)
        run_icl_stub(make_agent)
        return [agent.draws[1] for agent in agents]

    assert run(1) == run(2)


def test_run_icl_refuses_a_round_without_snapshots():
    with pytest.raises(ValueError, match="no snapshot after round 0"):
        run_icl_stub(lambda: ICLAgent(BACKWARD), snapshot_every=5)


def test_run_icl_untrained_policy_plays_every_backward_step_episode(monkeypatch):
    # Bug (7) mode: one new policy per round plays every update's episodes;
    # the snapshots (always backward, dissimilarity 1) are still taken but
    # never play. The stand-in policy always goes forward (dissimilarity 0).
    import icrl_bench.core.loop as loop
    built = []

    class StandInPolicy:
        def __init__(self, observation_space, action_space):
            built.append(self)

        def policy(self, obs):
            return FORWARD

    monkeypatch.setattr(loop, "ICLPPOPolicy", StandInPolicy)
    log, method = run_icl_stub(lambda: ICLAgent(BACKWARD), agent_data="untrained_policy")
    assert len(built) == 2  # one per round, none in the final round
    assert method.piles == [[0.0] * 3] * 4  # 2 rounds x 2 updates, all forward
    assert [row["snapshots"] for row in log] == [2, 4, 4]


def test_run_icl_refuses_an_unknown_agent_data():
    with pytest.raises(ValueError, match="agent_data"):
        run_icl_stub(lambda: ICLAgent(BACKWARD), agent_data="snapshot")


def lgw_metric_grid():
    """A hand-made grid on LapGridWorld: one point per cell (its observation
    2 * cell / 40 - 1), each step backward, so the true cost is 1 on every
    point."""
    values = 2 * np.arange(40) / 40 - 1
    steps = [Step(np.array([v], dtype=np.float32), BACKWARD, None) for v in values]
    return MetricGrid(values=values, trajectory=Trajectory(steps),
                      coordinate=lambda obs: obs[0], left_gap=0.024, right_gap=0.025)


class BouncingAgent(ICLAgent):
    """Alternates forward and backward: stays on cells 0 and 1, so its
    accrual is far from the expert's (laps, every cell equally)."""

    def __init__(self):
        super().__init__(None)
        self.n = 0

    def policy(self, obs):
        self.n += 1
        return FORWARD if self.n % 2 else BACKWARD


def expected_nad(grid, expert):
    """NAD by hand: the bouncing agent's episodes, replayed in a fresh env."""
    episodes = rollout(LapGridWorld(), BouncingAgent().policy, len(expert), seed=0)
    def acr(trajectories):
        return accrual([grid.coordinate(st.state) for t in trajectories for st in t.steps],
                       grid.values, grid.left_gap, grid.right_gap)
    return nad(acr(expert), acr(episodes))


def test_run_icl_logs_cmse_every_row_and_nad_on_the_last():
    grid = lgw_metric_grid()
    log, _ = run_icl_stub(BouncingAgent, metric_grid=grid)
    # Stand-in cost 0 everywhere vs true cost 1 everywhere -> CMSE 1.
    assert [row["true/cmse"] for row in log] == [1.0, 1.0, 1.0]
    assert np.isnan(log[0]["true/nad"]) and np.isnan(log[1]["true/nad"])
    expert = rollout(LapGridWorld(), always_forward, 2, seed=0)
    assert log[2]["true/nad"] == pytest.approx(expected_nad(grid, expert))
    assert log[2]["true/nad"] > 1


def test_run_icl_nad_on_the_last_round_without_a_final_round():
    log, _ = run_icl_stub(BouncingAgent, metric_grid=lgw_metric_grid(), final_round=False)
    assert len(log) == 2
    assert np.isnan(log[0]["true/nad"]) and log[1]["true/nad"] > 1


def test_run_icl_metrics_change_no_other_number():
    plain, _ = run_icl_stub(BouncingAgent)
    with_grid, _ = run_icl_stub(BouncingAgent, metric_grid=lgw_metric_grid())
    for a, b in zip(plain, with_grid):
        assert a == {k: v for k, v in b.items() if k not in ("true/cmse", "true/nad")}
