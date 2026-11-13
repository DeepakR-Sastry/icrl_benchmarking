"""Converts Gaurav's expert trajectories (the files his code ships,
expert-data/data-<env>.pt) into one of our pile files, so configs point at
them like any other pile. All episodes, read exactly as his code reads them
(data/gaurav.py; his pairing is already ours).

Run from anywhere: python scripts/make_gaurav_expert.py --env {hc,ant}
    [--source FILE]  default ~/refs/gaurav_icl/expert-data/data-<hc|ant>.pt
hc  = configs/hc.json  (wall x <= -1, beta 15) -> expert_data/hc_gaurav.pkl
ant = configs/ant.json (wall x <= -1, beta 5)  -> expert_data/ant_gaurav.pkl
(Was make_hc_gaurav_expert.py; --env added 2026-11-12, docs/decisions.md.)"""

import argparse
import hashlib
from datetime import date
from pathlib import Path

import numpy as np

from icrl_bench.data.gaurav import load_gaurav_rollouts
from icrl_bench.data.piles import save_pile
from icrl_bench.envs import ant_wall, half_cheetah_with_pos

COMMIT = "096e422"
EXPERT_DATA = Path.home() / "refs/gaurav_icl/expert-data"
OUT = Path(__file__).resolve().parents[1] / "expert_data"
WALL_X = -1.0  # his cost_condition s[0] <= -1 (hc.json and ant.json)
DISCOUNT = 0.99  # his discount_factor; beta bounds the discounted cost
# env -> (his file, true cost, beta, env label, his config, our file)
ENVS = {
    "hc": ("data-hc.pt", half_cheetah_with_pos.true_cost, 15,
           "HCWithPos-v0 (Gaurav's: gym 0.21, MuJoCo 2.1)", "configs/hc.json", "hc_gaurav.pkl"),
    "ant": ("data-ant.pt", ant_wall.true_cost, 5,
            "AntWall-v0 (Gaurav's: gym 0.21, MuJoCo 2.1)", "configs/ant.json", "ant_gaurav.pkl"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=ENVS)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    filename, true_cost, beta, label, his_config, out = ENVS[args.env]
    source = (args.source or EXPERT_DATA / filename).expanduser()
    path = OUT / out

    trajectories = load_gaurav_rollouts(source)
    step_costs = [np.array([true_cost(s.state, s.action, wall_x=WALL_X) for s in t.steps])
                  for t in trajectories]
    discounted = [float((c * DISCOUNT ** np.arange(len(c))).sum()) for c in step_costs]

    metadata = {
        "env": label,
        "policy": f"his expert: expert.py -c {his_config} (PPOLag with the true cost "
                  f"x <= -1, only_success=True: mean discounted cost < beta = {beta})",
        "source": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "commit": COMMIT,
        "n_episodes": len(trajectories),
        "pairing": "his = ours: Step(S[t], A[t]), the state the action was chosen on; "
                   "reward None (he stores none)",
        "created": date.today().isoformat(),
        "true_cost_wall_x": WALL_X,
        "episode_true_costs": [float(c.sum()) for c in step_costs],
        "episode_discounted_true_costs": discounted,
    }
    save_pile(path, trajectories, metadata)

    print(f"Saved {len(trajectories)} trajectories to {path}")
    print(f"steps per episode: {sorted({len(t.steps) for t in trajectories})}")
    print(f"episodes crossing x <= {WALL_X}: {sum(c.sum() > 0 for c in step_costs)}")
    print(f"discounted true cost per episode: mean {np.mean(discounted):.2f}, "
          f"max {max(discounted):.2f} (beta {beta})")


if __name__ == "__main__":
    main()
