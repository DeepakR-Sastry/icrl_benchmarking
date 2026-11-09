"""Upload Gaurav's ICL runs to wandb from their console logs, to chart them
next to ours (group ours_icl). His icl.py logs to tensorboard and prints
one dict per logging call ('key': value) to stdout; on Vulcan that is the
Slurm log (~/refs/cluster/gaurav_icl.sh).

    python scripts/upload_gaurav_logs.py logs/gaurav_icl_2221352_*.out \
        --project icrl-hc-icrl --group gaurav_icl

One wandb run per log, named <group>_seed<k> (k from his "Seed=k" line),
one point per round under OUR names (scripts/run_icrl.py logs ours with
our names, step = round): rounds 0..n-1, then his final no-mix round as
step n, as our final row. Only finished logs (with the final NAD) are
uploaded.

Per round r (his hc.json: 50 PPO epochs and 25 cost updates per round):
  true/cmse                     his cost_comparison after round r's update
                                (index r + 1: index 0 is the untrained net);
                                final row: the last one again, as ours
  forward/ep_rew_mean           mean of avg_env_reward over the round's last
                                25 epochs = 100 episodes, our window
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

EPOCHS_PER_ROUND = 50    # his hc.json ppo_epochs
EPOCHS_IN_OUR_WINDOW = 25  # 25 epochs x 4000 steps = 100 episodes of 1000
UPDATES_PER_ROUND = 25   # his hc.json updates_per_epoch


def values(text, key):
    """Every number printed as 'key': number, in order. The quote right after
    the name keeps e.g. avg_env_reward_window out of avg_env_reward."""
    return [float(v) for v in re.findall(rf"'{key}':\s*([-+0-9.eE]+)", text)]


def rows_from_log(text):
    """His log -> (seed, list of row dicts with our names), or (seed, None)
    if the log is unfinished."""
    seed = int(re.search(r"^Seed=(\d+)", text, re.MULTILINE).group(1))
    nad = values(text, "ppo_accrual_comparison")
    if not nad:
        return seed, None
    cmse = values(text, "cost_comparison")
    rewards = values(text, "avg_env_reward")
    agent_cost = values(text, "avg_agent_cost")
    satisfaction = values(text, "expert_satisfaction")

    n_rounds = len(cmse) - 1  # rounds with an update; + the final round
    # Sizes are hc.json's; another config (e.g. the smoke one) is refused.
    if (len(rewards) != (n_rounds + 1) * EPOCHS_PER_ROUND
            or not len(agent_cost) == len(satisfaction) == n_rounds * UPDATES_PER_ROUND):
        raise ValueError(
            f"not an hc.json-sized log: {n_rounds} rounds, {len(rewards)} PPO epochs "
            f"(expected {(n_rounds + 1) * EPOCHS_PER_ROUND}), {len(agent_cost)} cost "
            f"updates (expected {n_rounds * UPDATES_PER_ROUND})")

    rows = []
    for r in range(n_rounds + 1):
        end = (r + 1) * EPOCHS_PER_ROUND
        window = rewards[end - EPOCHS_IN_OUR_WINDOW:end]
        row = {
            "round": r,
            "true/cmse": cmse[min(r + 1, n_rounds)],
            "forward/ep_rew_mean": sum(window) / len(window),
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
    parser.add_argument("--group", required=True)
    parser.add_argument("--entity", default="irl-chip")
    args = parser.parse_args()

    import wandb

    for path in map(Path, args.logs):
        seed, rows = rows_from_log(path.read_text(errors="replace"))
        if rows is None:
            print(f"{path.name}: seed {seed}, unfinished (no final NAD), skipped")
            continue
        run = wandb.init(entity=args.entity, project=args.project,
                         group=args.group, name=f"{args.group}_seed{seed}",
                         config={"seed": seed, "source_log": path.name})
        for row in rows:
            run.log(row, step=row["round"])
        run.finish()
        print(f"{path.name}: seed {seed}, {len(rows)} rows")


if __name__ == "__main__":
    main()
