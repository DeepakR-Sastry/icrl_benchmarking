"""Compare our port of Gaurav's PPOLag (icrl_bench/agents/ppo_lagrangian_ray.py)
with his own TensorFlow code (~/refs/gaurav_icl, commit 096e422,
tools/algorithms/ppo_lag.py), on HalfCheetah (his configs/hc.json).

His code runs only in his venv (Python 3.8, TF 2.13), ours only in ours, so
this script has two halves that pass numbers through .npz files (NumPy
archives of named arrays). Run it from our venv; it starts his half itself
with his venv's Python. The file is kept Python-3.8-readable for that
reason, and each half imports only its own code.

Part 1 — same weights in both envs (methods.md §5, "Open"): his untrained
PPOLag earns ~10x more per episode than our port's. Starting weights of
both codes (seeds as given) are each played in both envs for a few epochs
without an update: his env through his own PPOLag.train(forward_only=True),
ours through our collect_epoch. Weights travel under his TF variable names
(kernels are (in, out); torch weights are (out, in)).

Part 2 — one epoch of his PPOLag (seed 1), with a made-up smooth cost
(sigmoid of the x-velocity, so the cost terms are not all 0):
  2a: from his buffer's raw per-step numbers and the values he bootstraps
      each path with, our path_targets + his normalisation rebuild the
      advantages and targets his CPOBuffer.get hands the update;
  2b: his starting weights, p and that exact batch → our λ step (c = his
      mean episode reward: his bug (8)) and our update; compared with his
      weights and p after, his policy steps and KLs.

Usage: python scripts/compare_gaurav_ppolag.py [--part 1|2] [--seeds 1 2 3] [--epochs 3]
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HIS_REPO = Path("~/refs/gaurav_icl").expanduser()
HIS_PYTHON = Path("~/refs/gaurav_venv/bin/python").expanduser()
# His venv's run recipe (docs/decisions.md 2026-11-08).
HIS_ENV = {"PYTHONPATH": str(HIS_REPO),
           "LD_LIBRARY_PATH": str(Path("~/.mujoco/mujoco210/bin").expanduser()) + ":/usr/lib/nvidia",
           "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "PYGLET_HEADLESS": "true",
           "PYGLET_SHADOW_WINDOW": "false", "CUDA_VISIBLE_DEVICES": ""}

# His TF variable -> (our network attribute, index of the Linear in it).
LAYERS = {f"{scope}/{dense}": (net, index)
          for scope, net in [("pi", "mean_net"), ("vf", "reward_value_net"), ("vc", "cost_value_net")]
          for dense, index in [("dense", 0), ("dense_1", 2), ("dense_2", 4)]}




def made_up_cost(obs, action):
    """Part 2's cost: sigmoid of the x-velocity (obs[9])."""
    return float(1 / (1 + np.exp(-obs[9])))


# --- His half (his venv) -------------------------------------------------------

def his_half(args):
    """Build his PPOLag as icl.py does (hc.json, -seed), optionally load
    weights into its TF variables, play `epochs` epochs without updates.
    Saves his starting weights (before loading) and the mean episode return
    of each epoch."""
    sys.argv = ["icl.py", "-c", str(HIS_REPO / "configs" / "hc.json"), "-seed", str(args.seed)]
    import tensorflow as tf
    import tools

    config = tools.utils.common.get_configuration(method_name="compare")
    config.update({"cost": lambda sa: 0.0})
    agent = tools.algorithms.PPOLag(config)
    sess = config["sess"]
    variables = {v.name[:-2]: v for v in tf.compat.v1.trainable_variables()}  # drop ":0"
    np.savez(args.out_weights, **{name: sess.run(v) for name, v in variables.items()})
    if args.weights:
        loaded = np.load(args.weights)
        for name in loaded.files:
            variables[name].load(loaded[name], sess)
    returns = [agent.train(forward_only=True)["avg_env_reward"] for _ in range(args.epochs)]
    np.save(args.out_returns, np.array(returns))


def his_half_update(args):
    """Part 2, his side: one train() of his PPOLag, recording around it."""
    sys.argv = ["icl.py", "-c", str(HIS_REPO / "configs" / "hc.json"), "-seed", str(args.seed)]
    import tensorflow as tf
    import tools

    config = tools.utils.common.get_configuration(method_name="compare")
    config.update({"cost": lambda sa: made_up_cost(*sa)})
    agent = tools.algorithms.PPOLag(config)
    sess, buf, ppo_agent = config["sess"], config["buf"], config["agent"]
    variables = {v.name[:-2]: v for v in tf.compat.v1.trainable_variables()}
    rec = {"init/" + name: sess.run(v) for name, v in variables.items()}
    last_values = []

    original_finish = buf.finish_path
    def finish_path(last_val=0, last_cval=0):
        last_values.append((float(np.squeeze(last_val)), float(np.squeeze(last_cval))))
        original_finish(last_val, last_cval)
    buf.finish_path = finish_path

    original_get = buf.get
    def get():
        for key in ["rew", "val", "cost", "cval"]:
            rec["raw/" + key] = getattr(buf, key + "_buf").copy()
        out = original_get()
        names = ["obs", "act", "adv", "cadv", "ret", "cret", "logp", "log_std", "mu"]
        rec.update({"batch/" + n: np.array(x) for n, x in zip(names, out)})
        return out
    buf.get = get

    kls = []
    original_update_pi = ppo_agent.update_pi
    def update_pi(inputs):
        rec["p_after_lambda"] = sess.run(config["penalty_param"])
        original_run = ppo_agent.sess.run
        def run(fetches, *a, **k):
            out = original_run(fetches, *a, **k)
            if isinstance(fetches, list) and fetches and fetches[0] is config["train_pi"]:
                kls.append(out[1])
            return out
        ppo_agent.sess.run = run
        original_update_pi(inputs)
        ppo_agent.sess.run = original_run
        rec["p_after_pi"] = sess.run(config["penalty_param"])
    ppo_agent.update_pi = update_pi

    metrics = agent.train()
    rec.update({"after/" + name: sess.run(v) for name, v in variables.items()})
    rec["kls"] = np.array(kls)
    rec["last_values"] = np.array(last_values)
    rec["avg_env_reward"] = metrics["avg_env_reward"]
    np.savez(args.out_weights, **rec)


# --- Our half (our venv) -------------------------------------------------------

def our_net(observation_space, action_space, seed):
    import torch
    from icrl_bench.agents.ppo_lagrangian_ray import ActorTwoCriticsRay
    torch.manual_seed(seed)  # as PPOLagrangianRay.__init__
    return ActorTwoCriticsRay(observation_space, action_space)


def weights_from_net(net):
    """Our network -> his TF names and shapes."""
    out = {"pi/log_std": net.log_std.detach().numpy()}
    for name, (attribute, index) in LAYERS.items():
        layer = getattr(net, attribute)[index]
        out[f"{name}/kernel"] = layer.weight.detach().numpy().T
        out[f"{name}/bias"] = layer.bias.detach().numpy()
    return out


def load_into_net(net, weights):
    import torch
    with torch.no_grad():
        net.log_std.copy_(torch.as_tensor(weights["pi/log_std"]))
        for name, (attribute, index) in LAYERS.items():
            layer = getattr(net, attribute)[index]
            layer.weight.copy_(torch.as_tensor(weights[f"{name}/kernel"].T))
            layer.bias.copy_(torch.as_tensor(weights[f"{name}/bias"]))


def our_returns(weights, seed, epochs):
    """Mean episode return per epoch of these weights in our env."""
    from icrl_bench.agents.ppo_lagrangian_ray import collect_epoch
    from icrl_bench.core.vector_env import make_vector_env
    from icrl_bench.envs.half_cheetah_with_pos import HalfCheetahWithPos
    envs = make_vector_env(lambda: HalfCheetahWithPos(), 1, lambda o, a: 0.0)
    envs.reset(seed=seed)
    net = our_net(envs.single_observation_space, envs.single_action_space, seed)
    load_into_net(net, weights)
    return [collect_epoch(net, envs, 4000, 0.99, 0.97, False)["episode_returns"].mean()
            for _ in range(epochs)]


def run_his_half(tmp, seed, epochs, weights=None):
    """His half in his venv; returns (his starting weights, returns)."""
    tag = f"{seed}_{'ours' if weights else 'his'}"
    out_weights, out_returns = tmp / f"his_init_{tag}.npz", tmp / f"his_returns_{tag}.npy"
    command = [str(HIS_PYTHON), "-B", __file__, "--his", "--seed", str(seed), "--epochs", str(epochs),
               "--out-weights", str(out_weights), "--out-returns", str(out_returns)]
    if weights:
        command += ["--weights", str(weights)]
    result = subprocess.run(command, cwd=tmp, env={**os.environ, **HIS_ENV},
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"his half failed:\n{result.stderr[-3000:]}")
    return dict(np.load(out_weights)), np.load(out_returns)


def part1(seeds, epochs):
    from icrl_bench.envs.half_cheetah_with_pos import HalfCheetahWithPos
    env = HalfCheetahWithPos()
    print("Part 1: mean episode return per epoch, no updates "
          f"({epochs} epochs of 4 episodes; seed = TF seed for his weights, torch seed for ours)")
    print(f"{'seed':>4}  {'weights':8} {'his env':>26}   {'our env':>26}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for seed in seeds:
            ours = weights_from_net(our_net(env.observation_space, env.action_space, seed))
            ours_file = tmp / f"ours_init_{seed}.npz"
            np.savez(ours_file, **ours)
            his, his_in_his = run_his_half(tmp, seed, epochs)
            _, ours_in_his = run_his_half(tmp, seed, epochs, weights=ours_file)
            for label, weights, in_his in [("his", his, his_in_his), ("ours", ours, ours_in_his)]:
                in_ours = our_returns(weights, seed, epochs)
                print(f"{seed:>4}  {label:8} {np.array2string(np.round(in_his), separator=' '):>26}   "
                      f"{np.array2string(np.round(in_ours), separator=' '):>26}")


def part2(seed):
    import torch
    from icrl_bench.agents.ppo_lagrangian_ray import PenaltyRay, path_targets, update, EPS
    from icrl_bench.envs.half_cheetah_with_pos import HalfCheetahWithPos

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out = tmp / "his_update.npz"
        command = [str(HIS_PYTHON), "-B", __file__, "--his-update", "--seed", str(seed),
                   "--out-weights", str(out)]
        result = subprocess.run(command, cwd=tmp, env={**os.environ, **HIS_ENV},
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"his half failed:\n{result.stderr[-3000:]}")
        his = dict(np.load(out))

    # 2a: targets from his raw numbers. His epoch = 4 paths of 1000 steps
    # (each episode ends at the time limit; the last also ends the epoch).
    n_paths = len(his["last_values"])
    length = len(his["raw/rew"]) // n_paths
    ours = {k: np.zeros(len(his["raw/rew"]), dtype=np.float32) for k in ["adv", "ret", "cadv", "cret"]}
    for i, (last_v, last_vc) in enumerate(his["last_values"]):
        path = slice(i * length, (i + 1) * length)
        ours["adv"][path], ours["ret"][path] = path_targets(
            his["raw/rew"][path], his["raw/val"][path], last_v, 0.99, 0.97)
        ours["cadv"][path], ours["cret"][path] = path_targets(
            his["raw/cost"][path], his["raw/cval"][path], last_vc, 0.99, 0.97)
    ours["adv"] = (ours["adv"] - ours["adv"].mean()) / (ours["adv"].std() + EPS)
    ours["cadv"] = ours["cadv"] - ours["cadv"].mean()
    print(f"Part 2a: {n_paths} paths of {length} steps; our targets vs his CPOBuffer.get "
          "(max |difference|, his largest |value|):")
    for key in ["adv", "ret", "cadv", "cret"]:
        print(f"  {key:5} {np.abs(ours[key] - his['batch/' + key]).max():.2e}   "
              f"{np.abs(his['batch/' + key]).max():.2f}")

    # 2b: one λ step + update from his weights, p and batch.
    env = HalfCheetahWithPos()
    net = our_net(env.observation_space, env.action_space, seed)
    load_into_net(net, {k[len("init/"):]: v for k, v in his.items() if k.startswith("init/")})
    penalty = PenaltyRay(1.0, 0.05, 15.0)
    with torch.no_grad():
        penalty.param.copy_(torch.as_tensor(his["init/penalty/penalty_param"]))
    pi_optimizer = torch.optim.Adam(net.policy_parameters() + [penalty.param], lr=5e-4, eps=1e-8)
    value_optimizer = torch.optim.Adam(net.value_parameters(), lr=1e-3, eps=1e-8)
    batch = {"observations": his["batch/obs"], "actions": his["batch/act"],
             "reward_advantages": his["batch/adv"], "cost_advantages": his["batch/cadv"],
             "reward_returns": his["batch/ret"], "cost_returns": his["batch/cret"],
             "log_probs": his["batch/logp"], "means": his["batch/mu"], "log_stds": his["batch/log_std"]}
    penalty.update(float(his["avg_env_reward"]))  # his bug (8)
    p_after_lambda = penalty.param.item()
    stats = update(net, penalty.param, pi_optimizer, value_optimizer, batch, clip_ratio=0.2,
                   ent_coef=0.0, pi_iters=80, target_kl=0.01, kl_margin=1.2, vf_iters=80)
    print(f"Part 2b: λ's c = his mean episode reward {float(his['avg_env_reward']):.2f} (bug (8))")
    print(f"  p after λ step: his {float(his['p_after_lambda']):.7f}  ours {p_after_lambda:.7f}")
    print(f"  p after policy steps: his {float(his['p_after_pi']):.7f}  ours {penalty.param.item():.7f}")
    print(f"  policy steps: his {len(his['kls'])} (KLs {np.round(his['kls'], 6)})  "
          f"ours {stats['pi_steps']} (last KL {stats['kl']:.6f})")
    after = weights_from_net(net)
    print("  weights after the update (max |ours − his|, his largest |value|):")
    for name in after:
        print(f"    {name:20} {np.abs(after[name] - his['after/' + name]).max():.2e}   "
              f"{np.abs(his['after/' + name]).max():.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, choices=[1, 2], default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=3)
    # His half only (started by run_his_half):
    parser.add_argument("--his", action="store_true")
    parser.add_argument("--his-update", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--weights")
    parser.add_argument("--out-weights")
    parser.add_argument("--out-returns")
    args = parser.parse_args()
    if args.his:
        his_half(args)
    elif args.his_update:
        his_half_update(args)
    elif args.part == 1:
        part1(args.seeds, args.epochs)
    else:
        part2(args.seeds[0])


if __name__ == "__main__":
    main()
