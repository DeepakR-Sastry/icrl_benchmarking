"""From a checked Config to a finished run: build every box, then run the
loop the config names (docs/design.md, "One entry point")."""

import csv
import inspect
from functools import partial
from pathlib import Path

from icrl_bench.config import save_config
from icrl_bench.core.vector_env import make_vector_env
from icrl_bench.data.piles import load_pile
from icrl_bench.registry import AGENTS, ENVS, FEATURIZERS, LOOPS, METHODS, METRIC_GRIDS, TRUE_COSTS


def build(config):
    """Config -> every box a loop may need, by its argument name (each loop
    takes the ones it uses). The agent comes as `agent` or as the factory
    `make_agent`, whichever the config's loop takes.

    Builds in the order scripts/validate_icrl_lgw.py does (expert, envs,
    agent, method): each constructor reseeds torch, so the order changes
    the numbers. The expert path is relative to the working directory."""
    def make_nominal_env():
        return ENVS[config.nominal_env.name](**config.nominal_env.args)

    def make_eval_env():
        return ENVS[config.eval_env.name](**config.eval_env.args)

    # A rule's config arguments are fixed ahead of time, as Malik's
    # partial(wall_behind, -3).
    true_cost_fn, _ = TRUE_COSTS[config.true_cost.name]
    true_cost = partial(true_cost_fn, **config.true_cost.args)
    featurize = FEATURIZERS[config.featurizer.name]

    expert, _ = load_pile(config.expert)
    # No cost yet: each loop sets its copies' cost before the first step.
    envs = make_vector_env(make_nominal_env, config.n_envs, None)
    agent_cls, _ = AGENTS[config.agent.name]

    def make_agent():
        return agent_cls(envs.single_observation_space, envs.single_action_space, config.agent.args)

    # A loop takes either one agent (run_icrl, run_gacl), built here at its
    # place in the order, or the factory (run_icl: a fresh agent each round),
    # and then no agent is built: an unused one would still reseed
    # (docs/decisions.md 2026-11-08).
    loop_fn, _, _ = LOOPS[config.loop.name]
    if "agent" in inspect.signature(loop_fn).parameters:
        agent_box = {"agent": make_agent()}
    else:
        agent_box = {"make_agent": make_agent}
    method_cls, _ = METHODS[config.method.name]
    n_features = featurize(expert[0]).shape[1]
    method = method_cls(n_features, config.method.args)

    boxes = dict(**agent_box, envs=envs, method=method, featurize=featurize, expert=expert,
                 make_nominal_env=make_nominal_env, make_eval_env=make_eval_env,
                 true_cost=true_cost)
    # ICL's metric grid, only if the config names one (config.py allows it
    # only for loops that take metric_grid). Builds no network, so it
    # doesn't touch the random streams.
    if config.metrics_grid is not None:
        boxes["metric_grid"] = METRIC_GRIDS[config.metrics_grid.name][0]()
    return boxes


def run(config, run_dir, callback=None):
    """Build every box, run the loop the config names (LOOPS), and write the
    run folder: config.yaml (every setting, defaults filled in) and
    metrics.csv (one row per round, or per rollout for GACL, written as
    soon as it ends, so a crash keeps every finished row). Refuses a folder that already holds a run, so results
    are never overwritten. callback, if given, is called with each row
    after it is written (e.g. to log to wandb). Returns the loop's log."""
    run_dir = Path(run_dir)
    if (run_dir / "config.yaml").exists():
        raise FileExistsError(f"{run_dir} already holds a run")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Written first, so even a crashed run records what it was.
    save_config(config, run_dir / "config.yaml")

    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = None

        def write_row(row):
            nonlocal writer
            if writer is None:  # columns = the first row's keys
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            if callback is not None:
                callback(row)

        loop_fn, _, _ = LOOPS[config.loop.name]
        accepted = inspect.signature(loop_fn).parameters
        boxes = {k: v for k, v in build(config).items() if k in accepted}
        return loop_fn(**boxes, settings=config.loop.args, callback=write_row)
