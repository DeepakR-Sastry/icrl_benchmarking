"""Runs one ICRL experiment from a config file (docs/design.md, "One entry
point") and writes runs/<name>/: config.yaml and metrics.csv.

Run from the repo root (paths inside configs are relative to it):
  python scripts/run_icrl.py --config configs/lgw/maximum_entropy.yaml
      [--seed 3]           replaces the config's seed
      [--name my_run]      default: <config folder>_<config file name>_seed<seed>,
                           e.g. hc_maximum_entropy_seed0 (configs/lgw and configs/hc
                           share file names, so the folder keeps runs apart)
      [--wandb-project P]  also log every row to wandb, under the names
                           Malik's code logs them
      [--wandb-group G]    default ours__<config file name>; the run is
                           <group>__seed<seed>, with fields code, config_file,
                           method, git_commit (docs/decisions.md 2026-11-12)"""

import argparse
import math
import subprocess
from dataclasses import asdict
from pathlib import Path

from icrl_bench.config import load_config
from icrl_bench.run import run

# Our metrics.csv column -> the name Malik's icrl.py logs it under (L279-303).
# His forward names are his SB3 logger's with "train/" -> "forward/", as in
# validate_ppo_lagrangian_hc.py's WANDB_NAMES. Columns not listed already
# have his name (true/*, backward/*, timesteps) or are ours only.
MALIK_ICRL_NAMES = {
    "round": "iteration",
    "forward/ep_rew_mean": "rollout/ep_rew_mean",
    "forward/penalty": "forward/nu",
    "forward/policy_loss": "forward/policy_gradient_loss",
}

# GACL: Malik's gail.py logs through his SB3 logger's Wandb output once per
# rollout (logger.py L560, on_policy_algorithm.py L237-244; GailCallback;
# EvalCallback). Columns not listed are ours only and keep our names.
MALIK_GAIL_NAMES = {
    "timesteps": "time/total_timesteps",
    "true/reward": "eval/mean_reward",
    "true/cost": "eval/mean_cost",
    "backward/disc_loss": "discriminator/disc_loss",
    "backward/expert_loss": "discriminator/expert_loss",
    "backward/nominal_loss": "discriminator/nominal_loss",
    "backward/mean_nominal_preds": "discriminator/mean_nominal_preds",
    "backward/mean_expert_preds": "discriminator/mean_expert_preds",
    "forward/ep_rew_mean": "rollout/ep_rew_mean",
    "forward/policy_loss": "train/policy_gradient_loss",
    "forward/value_loss": "train/value_loss",
    "forward/entropy_loss": "train/entropy_loss",
    "forward/approx_kl": "train/approx_kl",
    "forward/clip_fraction": "train/clip_fraction",
}

# Loop name -> (his names, the column that is his wandb step). icrl.py logs
# once per round with step = round; gail.py's logger uses step =
# num_timesteps. ICL: Gaurav's icl.py logs to tensorboard, not wandb, so
# ours keep our names and scripts/upload_gaurav_logs.py renames his to
# ours (as upload_liu_logs.py); step = round (the final round's is n_rounds).
WANDB = {
    "icrl": (MALIK_ICRL_NAMES, "round"),
    "gacl": (MALIK_GAIL_NAMES, "timesteps"),
    "icl": ({}, "round"),
}



def git_commit():
    """The repo's commit, "-dirty" if files differ from it; "unknown"
    outside a git checkout."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if dirty else "")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--name")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-group")
    args = parser.parse_args()

    config = load_config(args.config, seed=args.seed)
    path = Path(args.config)
    name = args.name or f"{path.parent.name}_{path.stem}_seed{config.seed}"

    callback, wandb_run = None, None
    if args.wandb_project is not None:
        import wandb  # only needed when logging to wandb
        # <code>__<config>: the two sides of a comparison share the part
        # after "__" (theirs come from scripts/upload_*_logs.py).
        group = args.wandb_group or f"ours__{path.stem}"
        fields = {"code": "ours", "config_file": str(path), "method": config.method.name,
                  "git_commit": git_commit()}
        wandb_run = wandb.init(project=args.wandb_project, group=group,
                               name=f"{group}__seed{config.seed}",
                               config={**asdict(config), **fields})

        names, step_column = WANDB[config.loop.name]

        def callback(row):
            # NaN = not measured this row (GACL rollouts without an
            # evaluation); his code logs nothing there, so neither do we.
            values = {names.get(k, k): v for k, v in row.items() if not math.isnan(v)}
            wandb_run.log(values, step=int(row[step_column]))

    log = run(config, Path("runs") / name, callback)
    if wandb_run is not None:
        wandb_run.finish()
    last = log[-1]
    # Last evaluated row (GACL's last row has no evaluation).
    evaluated = [row for row in log if not math.isnan(row["true/reward"])]
    reward = f"{evaluated[-1]['true/reward']:.2f}" if evaluated else "none"
    # true/cost: run_icrl / run_gacl rows; ICL rows have true/violating_episodes.
    cost = (f"true cost per step {last['true/cost']:.3f}" if "true/cost" in last
            else f"violating episodes {last['true/violating_episodes']:.2f}")
    print(f"runs/{name}: {len(log)} rows, {last['timesteps']} steps, "
          f"last reward {reward}, {cost}")


if __name__ == "__main__":
    main()
