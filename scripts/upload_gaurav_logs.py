"""Upload Gaurav's ICL runs to wandb from their console logs, to chart them
next to ours. His icl.py logs to tensorboard and prints one dict per
logging call ('key': value) to stdout; on Vulcan that is the Slurm log
(~/refs/cluster/gaurav_icl.sh).

    python scripts/upload_gaurav_logs.py logs/gaurav_icl_2221352_*.out \
        --project icrl-hc-icrl --config hc/icl_ray [--patch gaurav_ant.patch]
        [--episode-steps 500]

--config names OUR config for the same setup as <folder>/<name>
(configs/<folder>/<name>.yaml); its method is recorded. Group
gaurav__<folder>_<name>, one run per log, gaurav__<folder>_<name>__seed<k>
(k from his "Seed=k" line), as our run_icrl.py names ours
ours__<folder>_<name> (docs/decisions.md 2026-11-12). Fields: code gaurav,
config_file, method, git_commit (his commit, + "+<patch>" when --patch
names the patch his code ran with, e.g. Ant's), source_log. Runs uploaded
before 2026-11-12 keep the old group gaurav_icl. One point per round under OUR names (scripts/run_icrl.py logs ours with
our names, step = round): rounds 0..n-1, then his final no-mix round as
step n, as our final row. Only finished logs (with the final NAD) are
uploaded.

Per round r (his hc.json: 50 PPO epochs and 25 cost updates per round):
  true/cmse                     his cost_comparison after round r's update
                                (index r + 1: index 0 is the untrained net);
                                final row: the last one again, as ours
  forward/ep_rew_mean           mean of avg_env_reward (one mean per epoch)
                                over the round's last 100 episodes, our
                                window: 100 x episode steps / 4000 epochs
                                (HC 1000 steps: 25; Ant 500: 12.5 = the last
                                12 in full + half of the one before, exact
                                if those 4 episodes are typical of their
                                epoch; decisions.md 2026-11-13)
  backward/avg_agent_cost,      the round's last update (ours logs the last
  backward/expert_satisfaction  update's); no update after the final round
  true/nad                      ppo_accrual_comparison, final row only
Not uploaded: the untrained CMSE (no row for it in ours), the per-round
mixture NAD accrual_comparison (ours doesn't compute it), and values he
prints as sets ({old, new}: Penalty, LossPi, ...).
"""
import argparse
import re
from pathlib import Path

import yaml

HIS_COMMIT = "096e422"  # ~/refs/gaurav_icl, the clone his runs use (Vulcan too)

EPOCHS_PER_ROUND = 50    # his ppo_epochs (hc.json and ant.json)
STEPS_PER_EPOCH = 4000   # his steps_per_epoch (both)
UPDATES_PER_ROUND = 25   # his updates_per_epoch (both)
EPISODES_IN_OUR_WINDOW = 100  # our agents' ep_rew_mean: the last 100 episodes


def window_mean(epoch_means, n_epochs):
    """Mean over the last n_epochs epochs (n_epochs may be fractional): the
    last floor(n_epochs) count fully, the one before them by the leftover
    fraction. Equal-length episodes, so each epoch counts by its episodes."""
    full = int(n_epochs)
    part = n_epochs - full
    total = sum(epoch_means[len(epoch_means) - full:])
    if part > 0:
        total += part * epoch_means[len(epoch_means) - full - 1]
    return total / n_epochs


def values(text, key):
    """Every number printed as 'key': number, in order. The quote right after
    the name keeps e.g. avg_env_reward_window out of avg_env_reward."""
    return [float(v) for v in re.findall(rf"'{key}':\s*([-+0-9.eE]+)", text)]


def rows_from_log(text, episode_steps=1000):
    """His log -> (seed, list of row dicts with our names), or (seed, None)
    if the log is unfinished. episode_steps: his time_limit (hc.json 1000,
    ant.json 500), for the reward window."""
    window_epochs = EPISODES_IN_OUR_WINDOW * episode_steps / STEPS_PER_EPOCH
    seed = int(re.search(r"^Seed=(\d+)", text, re.MULTILINE).group(1))
    nad = values(text, "ppo_accrual_comparison")
    if not nad:
        return seed, None
    cmse = values(text, "cost_comparison")
    rewards = values(text, "avg_env_reward")
    agent_cost = values(text, "avg_agent_cost")
    satisfaction = values(text, "expert_satisfaction")

    n_rounds = len(cmse) - 1  # rounds with an update; + the final round
    # Sizes are hc.json's (= ant.json's); another config (e.g. a smoke one)
    # is refused.
    if (len(rewards) != (n_rounds + 1) * EPOCHS_PER_ROUND
            or not len(agent_cost) == len(satisfaction) == n_rounds * UPDATES_PER_ROUND):
        raise ValueError(
            f"not an hc.json-sized log: {n_rounds} rounds, {len(rewards)} PPO epochs "
            f"(expected {(n_rounds + 1) * EPOCHS_PER_ROUND}), {len(agent_cost)} cost "
            f"updates (expected {n_rounds * UPDATES_PER_ROUND})")

    rows = []
    for r in range(n_rounds + 1):
        end = (r + 1) * EPOCHS_PER_ROUND
        round_rewards = rewards[end - EPOCHS_PER_ROUND:end]
        row = {
            "round": r,
            "true/cmse": cmse[min(r + 1, n_rounds)],
            "forward/ep_rew_mean": window_mean(round_rewards, window_epochs),
        }
        if r < n_rounds:
            last = (r + 1) * UPDATES_PER_ROUND - 1
            row["backward/avg_agent_cost"] = agent_cost[last]
            row["backward/expert_satisfaction"] = satisfaction[last]
        else:
            row["true/nad"] = nad[0]
        rows.append(row)
    return seed, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--project", required=True)
    parser.add_argument("--config", required=True,
                        help="our config for this setup as <folder>/<name>, e.g. hc/icl_ray")
    parser.add_argument("--patch", help="patch his code ran with, e.g. gaurav_ant.patch")
    parser.add_argument("--episode-steps", type=int, default=1000,
                        help="his time_limit: hc.json 1000, ant.json 500 (reward window)")
    parser.add_argument("--entity", default="irl-chip")
    args = parser.parse_args()

    config_file = Path("configs") / f"{args.config}.yaml"
    if config_file.parent == Path("configs"):
        parser.error("--config needs a folder: <folder>/<name>, e.g. hc/icl_ray")
    method = yaml.safe_load(config_file.read_text())["method"]["name"]  # refuses a missing file
    group = f"gaurav__{config_file.parent.name}_{config_file.stem}"
    commit = HIS_COMMIT + (f"+{args.patch}" if args.patch else "")

    import wandb

    for path in map(Path, args.logs):
        seed, rows = rows_from_log(path.read_text(errors="replace"), args.episode_steps)
        if rows is None:
            print(f"{path.name}: seed {seed}, unfinished (no final NAD), skipped")
            continue
        run = wandb.init(entity=args.entity, project=args.project,
                         group=group, name=f"{group}__seed{seed}",
                         config={"seed": seed, "source_log": path.name, "code": "gaurav",
                                 "config_file": str(config_file), "method": method,
                                 "git_commit": commit})
        for row in rows:
            run.log(row, step=row["round"])
        run.finish()
        print(f"{path.name}: seed {seed}, {len(rows)} rows")


if __name__ == "__main__":
    main()
