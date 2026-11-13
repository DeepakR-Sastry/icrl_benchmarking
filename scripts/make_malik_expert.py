"""Converts Malik's shipped expert rollouts into one of our pile files, so
configs point at it like any other pile. As many rollouts as his README run
for that env uses (-er). Needs a clone of his repo.

Run from anywhere: python scripts/make_malik_expert.py --env {hc,ant}
    [--source DIR]  his -ep folder; default ~/refs/malik_icrl/icrl/expert_data/<folder>
                    (on Vulcan: ~/scratch/icrl/icrl/expert_data/<folder>)
hc  = HCWithPos-New, 10 rollouts (README -er 10) -> expert_data/hc_malik.pkl
ant = AntWall,       45 rollouts (README -er 45) -> expert_data/ant_malik.pkl
(Was make_hc_malik_expert.py; --env added 2026-11-12, docs/decisions.md.)"""

import argparse
from datetime import date
from pathlib import Path

from icrl_bench.data.malik import load_malik_rollouts
from icrl_bench.data.piles import save_pile
from icrl_bench.envs import ant_wall, half_cheetah_with_pos

COMMIT = "f28214e"
EXPERT_DATA = Path.home() / "refs/malik_icrl/icrl/expert_data"  # his -ep folders, by default
OUT = Path(__file__).resolve().parents[1] / "expert_data"
# env -> (his folder, rollouts his README uses, true cost, env label, our file)
ENVS = {
    "hc": ("HCWithPos-New", 10, half_cheetah_with_pos.true_cost,
           "HCWithPos-v0 (Malik's, MuJoCo 2)", "hc_malik.pkl"),
    "ant": ("AntWall", 45, ant_wall.true_cost,
            "AntWall-v0 (Malik's, MuJoCo 2)", "ant_malik.pkl"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=ENVS)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    folder, n_rollouts, true_cost, label, filename = ENVS[args.env]
    source = (args.source or EXPERT_DATA / folder).expanduser()
    path = OUT / filename

    trajectories = load_malik_rollouts(source, n_rollouts)
    costs = [sum(true_cost(step.state, step.action) for step in t.steps) for t in trajectories]

    metadata = {
        "env": label,
        "policy": "Malik's shipped expert rollouts",
        "source": f"{source}/files/EXPERT/rollouts/0..{n_rollouts - 1}.pkl",
        "commit": COMMIT,
        "n_episodes": n_rollouts,
        "pairing": "Step(his obs row t-1, his action t), t = 1..499; reward None",
        "created": date.today().isoformat(),
        "episode_true_costs": costs,
    }
    save_pile(path, trajectories, metadata)

    print(f"Saved {len(trajectories)} trajectories to {path}")
    print(f"steps per episode: {sorted({len(t.steps) for t in trajectories})}")
    print(f"true cost per episode: min {min(costs)}, max {max(costs)}")


if __name__ == "__main__":
    main()
