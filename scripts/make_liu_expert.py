"""Converts Liu's expert trajectories into one of our pile files, so configs
point at it like any other pile. As many rollouts as his configs for that
env use (running.expert_rollouts), read exactly as his code reads them —
including his pairing mismatch, bug (5) in docs/methods.md. Needs his
expert_data.zip unpacked (README "Download the Expert Data").

Run from anywhere: python scripts/make_liu_expert.py --env {hc,ant}
    [--source DIR]  his expert_path folder; default below
hc  = BlockedHalf-cheetah, 10 rollouts (VICRL/ME/BC configs) -> expert_data/hc_liu.pkl
ant = BlockedAnt,          50 rollouts (all three Ant configs) -> expert_data/ant_liu.pkl
Both folders are Malik's rollouts renamed (docs/envs.md). The Ant default
is the full re-download (2026-11-15); the first copy's BlockedAnt files are
truncated. (Was make_hc_liu_expert.py; --env added 2026-11-15.)"""

import argparse
import os
from datetime import date
from pathlib import Path

from icrl_bench.data.liu import load_liu_rollouts
from icrl_bench.data.piles import save_pile
from icrl_bench.envs import ant_wall, half_cheetah_with_pos

COMMIT = "0d79d2c"
DOWNLOADS = Path.home() / "refs/downloads"
OUT = Path(__file__).resolve().parents[1] / "expert_data"
# env -> (default folder, rollouts his configs use, true cost, env label,
#         which of Malik's files they are, our file)
ENVS = {
    "hc": (DOWNLOADS / "liu_expert_data/expert_data/BlockedHalf-cheetah", 10,
           half_cheetah_with_pos.true_cost, "HCWithPos-v0 (Malik's rollouts, MuJoCo 2)",
           "0, 2, 18, 19, 20, 21, 22, 23, 24, 25 (same content)", "hc_liu.pkl"),
    "ant": (DOWNLOADS / "liu_expert_data_v2/expert_data/BlockedAnt", 50,
            ant_wall.true_cost, "AntWall-v0 (Malik's rollouts, MuJoCo 2)",
            "scene-k = Malik's k-th file in string order (0, 1, 10, ..., 9) for "
            "k = 0..45; scenes 46-49 = Malik's 23, 9, 18, 25 (same content)", "ant_liu.pkl"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=ENVS)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    folder, n_rollouts, true_cost, label, malik_files, filename = ENVS[args.env]
    source = (args.source or folder).expanduser()
    path = OUT / filename

    trajectories = load_liu_rollouts(source, n_rollouts)
    costs = [sum(true_cost(step.state, step.action) for step in t.steps) for t in trajectories]

    metadata = {
        "env": label,
        "policy": "Liu's distributed expert data = Malik's shipped rollouts, renamed",
        "source": f"{source}: {sorted(os.listdir(source))[:n_rollouts]}",
        "malik_files": malik_files,
        "commit": COMMIT,
        "n_episodes": n_rollouts,
        "pairing": "AS LIU READS IT: Step(row t, action t), t = 0..499 — rows are "
                   "'after the action' (Malik's), so this reproduces bug (5); reward None",
        "created": date.today().isoformat(),
        "episode_true_costs": costs,
    }
    save_pile(path, trajectories, metadata)

    print(f"Saved {len(trajectories)} trajectories to {path}")
    print(f"steps per episode: {sorted({len(t.steps) for t in trajectories})}")
    print(f"true cost per episode: min {min(costs)}, max {max(costs)}")


if __name__ == "__main__":
    main()
