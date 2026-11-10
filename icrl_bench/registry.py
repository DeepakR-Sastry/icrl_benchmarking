"""Name -> object, so a config can name each box (docs/design.md,
"Conventions"). A new box is one new file plus one entry here."""

from icrl_bench.agents.ppo import PPO, PPOSettings
from icrl_bench.agents.ppo_lagrangian import PPOLagrangian, PPOLagrangianSettings
from icrl_bench.agents.ppo_lagrangian_ray import PPOLagrangianRay, PPOLagrangianRaySettings
from icrl_bench.core.loop import (GACLLoopSettings, ICLLoopSettings, ICRLLoopSettings, run_gacl,
                                  run_icl, run_icrl)
from icrl_bench.envs import half_cheetah_with_pos, lap_grid_world
from icrl_bench.featurizers import half_cheetah_with_pos as half_cheetah_with_pos_features
from icrl_bench.featurizers import lap_grid_world as lap_grid_world_features
from icrl_bench.methods.binary_classifier import BinaryClassifier, BinaryClassifierSettings
from icrl_bench.methods.gacl import GACL, GACLSettings
from icrl_bench.methods.icl import ICL, ICLSettings
from icrl_bench.methods.maximum_entropy import MaximumEntropy, MaximumEntropySettings
from icrl_bench.methods.vicrl import VICRL, VICRLSettings

# name -> gymnasium env class; config arguments go to its constructor.
ENVS = {
    "lap_grid_world": lap_grid_world.LapGridWorld,
    "half_cheetah_with_pos": half_cheetah_with_pos.HalfCheetahWithPos,
}

# name -> (true_cost(obs, action) -> float, names of the envs it applies to).
# A rule assumes what the observation holds, so it lists the envs that
# meet its assumptions (docs/decisions.md, 2026-10-21).
TRUE_COSTS = {
    "lap_grid_world": (lap_grid_world.true_cost, {"lap_grid_world"}),
    "half_cheetah_with_pos": (half_cheetah_with_pos.true_cost, {"half_cheetah_with_pos"}),
}

# name -> featurize(trajectory) -> array (steps, features).
FEATURIZERS = {
    "lap_grid_world": lap_grid_world_features.featurize,
    "half_cheetah_with_pos": half_cheetah_with_pos_features.featurize,
    # Unclipped, as Malik's GAIL discriminator sees it (GACL).
    "half_cheetah_with_pos_raw": half_cheetah_with_pos_features.featurize_raw,
    # The x-position only, as Gaurav's ICL cost sees it (configs/hc.json).
    "half_cheetah_with_pos_x": half_cheetah_with_pos_features.featurize_x,
}

# Where ICL's CMSE and NAD are measured: name -> (function returning a
# MetricGrid, set of env names it applies to); optional config block
# metrics_grid.
METRIC_GRIDS = {
    "half_cheetah_with_pos": (half_cheetah_with_pos.metric_grid, {"half_cheetah_with_pos"}),
}

# name -> (class, settings dataclass).
AGENTS = {
    "ppo": (PPO, PPOSettings),
    "ppo_lagrangian": (PPOLagrangian, PPOLagrangianSettings),
    "ppo_lagrangian_ray": (PPOLagrangianRay, PPOLagrangianRaySettings),
}

METHODS = {
    "maximum_entropy": (MaximumEntropy, MaximumEntropySettings),
    "binary_classifier": (BinaryClassifier, BinaryClassifierSettings),
    "gacl": (GACL, GACLSettings),
    "vicrl": (VICRL, VICRLSettings),
    "icl": (ICL, ICLSettings),
}

# name -> (loop function, settings dataclass, names of the methods it runs).
# A config's loop block names one (default "icrl"). A method in the wrong
# loop can run without error but be a different algorithm (GACL has a
# cost, so run_icrl would accept it), so each loop lists its methods
# (docs/decisions.md, 2026-11-01).
LOOPS = {
    "icrl": (run_icrl, ICRLLoopSettings, {"maximum_entropy", "binary_classifier", "vicrl"}),
    "gacl": (run_gacl, GACLLoopSettings, {"gacl"}),
    "icl": (run_icl, ICLLoopSettings, {"icl"}),
}


def check_metric_grid_fits(grid_name, env_name):
    """Raise if the grid is not listed for this env: its coordinate (e.g.
    obs[0]) means something else in another env, silently."""
    _, env_names = METRIC_GRIDS[grid_name]
    if env_name not in env_names:
        raise ValueError(f"metrics_grid {grid_name!r} does not apply to env {env_name!r} "
                         f"(it applies to: {sorted(env_names)})")


def check_true_cost_fits(true_cost_name, env_name):
    """Raise if the rule is not listed for this env: on the wrong env a rule
    can return meaningless numbers without any error."""
    _, env_names = TRUE_COSTS[true_cost_name]
    if env_name not in env_names:
        raise ValueError(f"true cost {true_cost_name!r} does not apply to env {env_name!r} "
                         f"(it applies to: {sorted(env_names)})")


def check_method_fits_loop(loop_name, method_name):
    """Raise if the method is not listed for this loop."""
    _, _, method_names = LOOPS[loop_name]
    if method_name not in method_names:
        raise ValueError(f"method {method_name!r} does not run in loop {loop_name!r} "
                         f"(it runs: {sorted(method_names)})")
