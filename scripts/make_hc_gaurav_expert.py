"""Converts Gaurav's HalfCheetah expert trajectories (the file his code ships,
expert-data/data-hc.pt) into one of our pile files, so configs point at it
like any other pile. All 100 episodes, read exactly as his code reads them
(data/gaurav.py; his pairing is already ours).

Run from anywhere: python scripts/make_hc_gaurav_expert.py
    [--source FILE]  default ~/refs/gaurav_icl/expert-data/data-hc.pt"""

import argparse
import hashlib
from datetime import date
from pathlib import Path

import numpy as np

from icrl_bench.data.gaurav import load_gaurav_rollouts
from icrl_bench.data.piles import save_pile
from icrl_bench.envs.half_cheetah_with_pos import true_cost

SOURCE = Path.home() / "refs/gaurav_icl/expert-data/data-hc.pt"
COMMIT = "096e422"
WALL_X = -1.0  # his configs/hc.json cost_condition s[0] <= -1
DISCOUNT = 0.99  # his discount_factor; beta = 15 bounds the discounted cost
PATH = Path(__file__).resolve().parents[1] / "expert_data" / "hc_gaurav.pkl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    source = parser.parse_args().source.expanduser()
    trajectories = load_gaurav_rollouts(source)
    step_costs = [np.array([true_cost(s.state, s.action, wall_x=WALL_X) for s in t.steps])
                  for t in trajectories]
    discounted = [float((c * DISCOUNT ** np.arange(len(c))).sum()) for c in step_costs]

    metadata = {
        "env": "HCWithPos-v0 (Gaurav's: gym 0.21, MuJoCo 2.1)",
        "policy": "his expert: expert.py -c configs/hc.json (PPOLag with the true cost "
                  "x <= -1, only_success=True: mean discounted cost < beta = 15)",
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
    save_pile(PATH, trajectories, metadata)

    print(f"Saved {len(trajectories)} trajectories to {PATH}")
    print(f"steps per episode: {sorted({len(t.steps) for t in trajectories})}")
    print(f"episodes crossing x <= {WALL_X}: {sum(c.sum() > 0 for c in step_costs)}")
    print(f"discounted true cost per episode: mean {np.mean(discounted):.2f}, "
          f"max {max(discounted):.2f} (beta 15)")


if __name__ == "__main__":
    main()
