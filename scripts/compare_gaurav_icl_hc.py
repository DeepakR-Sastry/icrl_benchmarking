"""Compare one of our ICL HalfCheetah runs with one of Gaurav's, round by round.

His side is the console log of `icl.py -c configs/hc.json -seed S`
(commit 096e422), which prints one dict per logging call; ours is the
run's metrics.csv. Printed side by side: CMSE after each round, the
forward agent's training reward in each round, and the final NAD
(docs/results.md, step 17).

    python scripts/compare_gaurav_icl_hc.py --ours runs/icl_hc_seed1/metrics.csv \
        --his ~/refs/gaurav_runs/full_hc_seed1.log
"""
import argparse
import csv
import math
import re
from pathlib import Path

# His hc.json: ppo_epochs 50 per round; steps_per_epoch 4000 = 4 episodes of
# 1000 steps. Our forward/ep_rew_mean is the mean of the last 100 training
# episodes (SB3's window), i.e. his last 25 epochs of a round.
EPOCHS_PER_ROUND = 50
EPOCHS_IN_OUR_WINDOW = 25


def read_his_log(path):
    """His log -> (cmse list, per-epoch reward list, final NAD).

    Each value is printed as 'key': number. The closing quote right after
    the name keeps e.g. avg_env_reward_window out of avg_env_reward.
    """
    text = Path(path).expanduser().read_text()

    def values(key):
        return [float(v) for v in re.findall(rf"'{key}':\s*([-+0-9.eE]+)", text)]

    cmse = values("cost_comparison")       # [untrained, after round 1, ...]
    rewards = values("avg_env_reward")      # one per PPO epoch, every round
    nad = values("ppo_accrual_comparison")  # once, final agent
    return cmse, rewards, (nad[0] if nad else math.nan)


def read_our_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fmt(x, digits):
    return "—" if x is None or math.isnan(x) else f"{x:.{digits}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ours", default="runs/icl_hc_seed1/metrics.csv")
    p.add_argument("--his", default="~/refs/gaurav_runs/full_hc_seed1.log")
    args = p.parse_args()

    his_cmse, his_rewards, his_nad = read_his_log(args.his)
    ours = read_our_csv(args.ours)

    # His rounds: 5 + final; his first CMSE (untrained net) has no row in ours.
    his_rounds = len(his_rewards) // EPOCHS_PER_ROUND
    print(f"his CMSE of the untrained net (not logged by ours): {fmt(his_cmse[0], 3)}")
    print(f"rows: ours {len(ours)}, his {his_rounds} (5 rounds + final in a full run)\n")

    print(f"{'round':>6} {'CMSE his':>9} {'CMSE ours':>10} {'fwd rew his':>12} {'fwd rew ours':>13}")
    for r in range(max(len(ours), his_rounds)):
        # His CMSE after round r is cmse[r + 1]; the final round has no update,
        # so its CMSE is the last one again (as our final row).
        h_cmse = his_cmse[min(r + 1, len(his_cmse) - 1)] if r < his_rounds else None
        if r < his_rounds:
            last = his_rewards[(r + 1) * EPOCHS_PER_ROUND - EPOCHS_IN_OUR_WINDOW:(r + 1) * EPOCHS_PER_ROUND]
            h_rew = sum(last) / len(last)
        else:
            h_rew = None
        o_cmse = float(ours[r]["true/cmse"]) if r < len(ours) else None
        o_rew = float(ours[r]["forward/ep_rew_mean"]) if r < len(ours) else None
        label = "final" if r == his_rounds - 1 else str(r)
        print(f"{label:>6} {fmt(h_cmse, 3):>9} {fmt(o_cmse, 3):>10} {fmt(h_rew, 1):>12} {fmt(o_rew, 1):>13}")

    our_nad = float(ours[-1]["true/nad"]) if ours else math.nan
    print(f"\nfinal NAD: his {fmt(his_nad, 2)}, ours {fmt(our_nad, 2)} (paper 10.16 ± 7.49)")


if __name__ == "__main__":
    main()
