"""Compare our ICL pieces with Gaurav's own code on the same seed and data.

1. Flow (icrl_bench/methods/flow.py): starting weights, log-probabilities,
   a few training epochs.
2. Method (icrl_bench/methods/icl.py): cost-net starting weights, the flow
   fitted on the expert (its -log p mean / std), then two cost-adjustment
   updates in a row (metrics, cost-net weights after each).
3. Untrained policy (icrl_bench/agents/icl_ppo_policy.py): his PPOPolicy.Pi
   as PPOPolicy.reset_parameters builds it (continuous actions, HalfCheetah
   sizes), and his act's sampling lines (ppo.py L69-75, his FixedNormal):
   starting weights, then actions for the same sampling seed.

His code is PyTorch, loaded unedited from his clone (~/refs/gaurav_icl,
commit 096e422): tools/utils/torch_helper.py, tools/functions/flow.py,
tools/functions/cost_function.py, tools/base/dataset.py
(TrajectoryDataset), tools/algorithms/cost_adjustment.py. Stand-ins replace
only imports the checked code never uses: `tensorflow` (torch_helper's
seeding), `dill.source` (CostFunction's config printing), `load_policy`
(TensorFlow policies), the empty base classes from tools.base. His
CostAdjustment is created without its __init__ (which loads the expert
file from disk and builds a PPO policy); the attributes __init__ sets are
filled in by hand, and its agent data comes through his own
config["agent_dataset"] hook (cost_adjustment.py L57-80). Runs on CPU (his
TorchHelper picks the GPU if it sees one).

Usage: python scripts/compare_gaurav_icl.py [--repo ~/refs/gaurav_icl]
"""

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported

import numpy as np
import torch
from gymnasium.spaces import Box

from icrl_bench.agents.icl_ppo_policy import ICLPPOPolicy
from icrl_bench.methods.flow import Flow, FlowSettings
from icrl_bench.methods.icl import ICL, ICLSettings

STAND_INS = ("tensorflow", "dill", "dill.source", "tools", "tools.base", "tools.utils",
             "tools.safe_rl", "tools.safe_rl.utils", "tools.safe_rl.utils.load_utils")


def load_his_code(repo):
    """His classes by name, loaded from his files with stand-in imports."""
    def stand_in(name, **attributes):
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module
        return module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, repo / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    stand_in("tensorflow")
    stand_in("dill")
    stand_in("dill.source", getsource=lambda f: "")
    stand_in("tools.safe_rl")
    stand_in("tools.safe_rl.utils")
    stand_in("tools.safe_rl.utils.load_utils", load_policy=None)
    helper = load("gaurav_torch_helper", "tools/utils/torch_helper.py")
    stand_in("tools")
    stand_in("tools.utils", TorchHelper=helper.TorchHelper)
    base = stand_in("tools.base", Function=type("Function", (), {}),
                    Parameterized=type("Parameterized", (), {}),
                    Algorithm=type("Algorithm", (), {}))
    dataset = load("gaurav_dataset", "tools/base/dataset.py")
    base.TrajectoryDataset = dataset.TrajectoryDataset
    flow = load("gaurav_flow", "tools/functions/flow.py")
    cost_function = load("gaurav_cost_function", "tools/functions/cost_function.py")
    cost_adjustment = load("gaurav_cost_adjustment", "tools/algorithms/cost_adjustment.py")
    # Only needed while his files load; torch itself later probes for a
    # real `tensorflow` and fails on a stand-in.
    for name in STAND_INS:
        del sys.modules[name]
    return {"TorchHelper": helper.TorchHelper, "Flow": flow.Flow,
            "CostFunction": cost_function.CostFunction,
            "CostAdjustment": cost_adjustment.CostAdjustment,
            "TrajectoryDataset": dataset.TrajectoryDataset,
            "FixedNormal": helper.FixedNormal}


class HisConfig:
    """What his classes read from his Configuration: config.data, config[key]."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value


def build_his_flow(his, data, settings):
    """His create_flow(config, data, "realnvp", i, h, n_blocks), line by line."""
    i = data.shape[1]
    mask = torch.arange(i) % 2
    components = []
    for _ in range(settings.n_blocks):
        components += [["an", i], ["1x1conv", i], ["realnvp", i, settings.hidden, mask]]
        mask = 1 - mask
    config = HisConfig({
        "t": his["TorchHelper"](),
        "learning_rate": settings.learning_rate,
        "minibatch_size": settings.batch_size,
        "normalize_flow_inputs": settings.normalize_inputs,
    })
    return his["Flow"](config, data, components)


def his_flow_tensors(flow):
    """His flow's weights by our names (his P and mask are plain attributes,
    ours are registered buffers)."""
    out = {}
    for k, module in enumerate(flow.Fnn.modulelist):
        for name, value in module.named_parameters():
            out[f"layers.{k}.{name}"] = value.detach().clone()
        for name in ("P", "mask"):
            if hasattr(module, name):
                out[f"layers.{k}.{name}"] = getattr(module, name).detach().clone()
    return out


def our_flow_tensors(flow):
    return {name: value.detach().clone()
            for name, value in list(flow.named_parameters()) + list(flow.named_buffers())
            if name.startswith("layers.")}


def max_difference(a, b):
    assert a.keys() == b.keys(), sorted(set(a) ^ set(b))
    return max((a[k] - b[k]).abs().max().item() for k in a)


def compare_flow(his, n_columns, seed, n_epochs):
    settings = FlowSettings(normalize_inputs=True)  # his Gridworld A setting
    data = torch.as_tensor(np.random.RandomState(seed).randn(300, n_columns) * 3 + 1,
                           dtype=torch.float32)

    torch.manual_seed(seed)
    his_flow = build_his_flow(his, data, settings)
    torch.manual_seed(seed)
    ours = Flow(data, settings)

    print(f"--- flow: {n_columns} columns, seed {seed}")
    print(f"starting weights, max |difference|: "
          f"{max_difference(his_flow_tensors(his_flow), our_flow_tensors(ours)):.3g}")
    with torch.no_grad():
        lp_his, lp_ours = his_flow.log_probs(data), ours.log_probs(data)
    print(f"log_probs shapes {tuple(lp_his.shape)} vs {tuple(lp_ours.shape)}, "
          f"max |difference|: {(lp_his - lp_ours).abs().max().item():.3g} "
          f"(mean log p {lp_ours.mean().item():.4f})")

    # Same random state before each training run, so the shuffles match.
    torch.manual_seed(seed + 1)
    his_rows = [his_flow.train() for _ in range(n_epochs)]
    torch.manual_seed(seed + 1)
    our_rows = [ours.train_epoch() for _ in range(n_epochs)]
    for epoch, (h, o) in enumerate(zip(his_rows, our_rows)):
        print(f"epoch {epoch}: avg nll his {h['avg_nll']:.6f} ours {o['avg_nll']:.6f}")
    print(f"weights after {n_epochs} epochs, max |difference|: "
          f"{max_difference(his_flow_tensors(his_flow), our_flow_tensors(ours)):.3g}")


def make_trajectories(rng, lengths, shift):
    """(states (T, 2), actions (T, 1)) per trajectory, around `shift`."""
    return [(rng.randn(n, 2) * 0.5 + shift, rng.randn(n, 1) * 0.5 + shift) for n in lengths]


def compare_method(his, seed):
    """His icl.py L20-42 + CostAdjustment.train twice vs our ICL."""
    s = ICLSettings(discount=0.9, beta=2.5, alpha=15.0, updates_per_epoch=5, flow_epochs=5,
                    normalize_flow_inputs=True, seed=seed)
    rng = np.random.RandomState(seed)
    expert = make_trajectories(rng, [6, 9, 7, 8, 5, 9, 6, 8], shift=0.0)
    # Two nominal piles (one per update), each part expert-like, part far.
    nominal_piles = [make_trajectories(rng, [7, 10], 0.0) + make_trajectories(rng, [9, 6, 8], 2.0),
                     make_trajectories(rng, [8, 5], 0.0) + make_trajectories(rng, [10, 7, 6], 2.0)]

    def features(trajectories):  # his default vector_input_format: cat(S, A)
        return [np.concatenate([states, actions], axis=1) for states, actions in trajectories]

    # --- His: icl.py order — cost net, then flow on the expert, then train.
    t = his["TorchHelper"]()
    config = HisConfig({"t": t, "beta": s.beta, "learning_rate": s.learning_rate,
                        "discount_factor": s.discount, "minibatch_size": s.flow_batch_size,
                        "normalize_flow_inputs": s.normalize_flow_inputs, "i": 3})
    torch.manual_seed(seed)
    cost = his["CostFunction"](config, i=3, h=s.hidden, o=1)
    # Added after CostFunction is built, so it uses its own defaults (the same).
    config.data.update({
        "cost": cost,
        "vector_state_reduction": lambda S: S,
        "vector_action_reduction": lambda A: A,
        "vector_input_format": lambda S, A: torch.cat((S, A), dim=-1),
    })
    expert_dataset = his["TrajectoryDataset"]([[S, A] for S, A in expert])
    eSA = torch.cat((expert_dataset.S, expert_dataset.A), dim=-1).view(-1, 3)[
        torch.nonzero(expert_dataset.M.view(-1)).view(-1)]
    flow = build_his_flow(his, eSA, FlowSettings(learning_rate=s.learning_rate,
                                                 batch_size=s.flow_batch_size,
                                                 normalize_inputs=s.normalize_flow_inputs))
    for _ in range(s.flow_epochs):
        flow.train()
    ep = -flow.log_probs(eSA)
    config.data.update({"flow": flow, "expert_nll": (ep.mean().item(), ep.std().item())})
    his_cost_start = {k: v.detach().clone() for k, v in cost.Cost.state_dict().items()}

    # --- Ours: the same seed, the same order.
    ours = ICL(3, s)
    our_cost_start = {k: v.detach().clone() for k, v in ours.net.state_dict().items()}
    ours.fit_flow(features(expert))

    print(f"--- method: seed {seed}")
    print(f"cost net starting weights, max |difference|: "
          f"{max_difference(his_cost_start, our_cost_start):.3g}")
    print(f"expert -log p mean / std: his {config['expert_nll'][0]:.6f} / {config['expert_nll'][1]:.6f}"
          f"  ours {ours.expert_nll[0]:.6f} / {ours.expert_nll[1]:.6f}")

    # His CostAdjustment without __init__: the attributes it would set.
    adjustment = his["CostAdjustment"].__new__(his["CostAdjustment"])
    adjustment.config = config
    adjustment.past_pi_weights = []
    adjustment.pi_episodes = None  # unused: agent data comes from config["agent_dataset"]
    adjustment.updates_per_epoch = s.updates_per_epoch
    adjustment.alpha = s.alpha
    adjustment.expert_dataset = expert_dataset
    adjustment.D = t.f([s.discount ** i for i in range(expert_dataset.max_trajectory_length + 10)])

    for k, nominal in enumerate(nominal_piles):
        config["agent_dataset"] = his["TrajectoryDataset"]([[S, A] for S, A in nominal])
        his_metrics = adjustment.train()
        our_metrics = ours.update(features(expert), features(nominal))
        print(f"update {k}: dissimilarity ours {np.round(ours.dissimilarity(features(nominal)), 3)}")
        for key in his_metrics:
            print(f"  {key:20s} his {his_metrics[key]: .7f}  ours {our_metrics[key]: .7f}")
        print(f"  cost net weights after, max |difference|: "
              f"{max_difference(dict(cost.Cost.state_dict()), dict(ours.net.state_dict())):.3g}")


def compare_policy(his, seed, n_obs=18, n_actions=6, hidden=64, n_states=300):
    """His PPOPolicy.Pi (ppo.py L54-58) and act (L69-75) vs ICLPPOPolicy."""
    torch.manual_seed(seed)
    his_pi = his["TorchHelper"]().nn([[n_obs, hidden], 't', [hidden, hidden], 't',
                                     ['g', hidden, n_actions]])
    torch.manual_seed(seed)
    ours = ICLPPOPolicy(Box(-np.inf, np.inf, (n_obs,)), Box(-1, 1, (n_actions,)), hidden)

    print(f"--- untrained policy: {n_obs} -> {n_actions}, seed {seed}")
    print(f"starting weights, max |difference|: "
          f"{max_difference(dict(his_pi.state_dict()), dict(ours.net.state_dict())):.3g}")

    # HalfCheetah-like raw states (x up to ~20, velocities of a few units).
    states = np.random.RandomState(seed).randn(n_states, n_obs) * 3
    normal = his["FixedNormal"](0., 1.)  # his PPOPolicy.__init__ L31
    torch.manual_seed(seed + 1)
    his_actions = []
    for s in states:
        normal.update(*his_pi(torch.tensor([s], dtype=torch.float)))
        his_actions.append(normal.sample().view(-1).detach().cpu().numpy())
    torch.manual_seed(seed + 1)
    our_actions = [ours.policy(s) for s in states]
    his_actions, our_actions = np.array(his_actions), np.array(our_actions)
    print(f"{n_states} sampled actions, max |difference|: "
          f"{np.abs(his_actions - our_actions).max():.3g} "
          f"(|action| mean {np.abs(our_actions).mean():.3f}, max {np.abs(our_actions).max():.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.home() / "refs/gaurav_icl")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    his = load_his_code(args.repo)
    for n_columns in (2, 5):
        compare_flow(his, n_columns, seed=0, n_epochs=args.epochs)
    compare_method(his, seed=0)
    compare_policy(his, seed=0)


if __name__ == "__main__":
    main()
