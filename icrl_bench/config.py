"""Reads an experiment's YAML config into a checked Config (docs/design.md,
"Conventions"). Only reads and checks; building the boxes is the run
script's job."""

from dataclasses import asdict, dataclass, fields, is_dataclass

import inspect

import yaml

from icrl_bench.registry import (AGENTS, ENVS, FEATURIZERS, LOOPS, METHODS, METRIC_GRIDS, TRUE_COSTS,
                                 check_method_fits_loop, check_metric_grid_fits, check_true_cost_fits)

KEYS = {"nominal_env", "eval_env", "true_cost", "featurizer", "expert", "n_envs",
        "agent", "method", "loop", "seed"}
OPTIONAL_KEYS = {"metrics_grid"}  # ICL's CMSE / NAD grid (docs/decisions.md 2026-11-08)


@dataclass(frozen=True)
class Named:
    """A box chosen by registry name. args: keyword arguments (envs, true
    costs, featurizers) or a settings dataclass (agents, methods, loop)."""
    name: str
    args: object


@dataclass(frozen=True)
class Config:
    nominal_env: Named
    eval_env: Named
    true_cost: Named
    featurizer: Named
    expert: str  # path of the expert pile file
    n_envs: int
    agent: Named
    method: Named
    loop: Named  # name defaults to "icrl" (docs/decisions.md, 2026-11-01)
    seed: int
    metrics_grid: Named | None = None  # only loops taking metric_grid (run_icl)


def load_config(path, seed=None):
    """seed, if given, replaces the file's seed before checking, so one
    file serves every seed of an experiment."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if seed is not None:
        raw["seed"] = seed
    return parse_config(raw)


def parse_config(raw):
    """raw: the dict read from YAML. Raises ValueError on any mistake."""
    missing, unknown = KEYS - set(raw), set(raw) - KEYS - OPTIONAL_KEYS
    if missing or unknown:
        raise ValueError(f"config keys: missing {sorted(missing)}, unknown {sorted(unknown)}")
    seed = raw["seed"]
    config = Config(
        nominal_env=_named(raw["nominal_env"], ENVS, "nominal_env"),
        eval_env=_named(raw["eval_env"], ENVS, "eval_env"),
        true_cost=_named(raw["true_cost"], TRUE_COSTS, "true_cost"),
        featurizer=_named(raw["featurizer"], FEATURIZERS, "featurizer"),
        expert=raw["expert"],
        n_envs=raw["n_envs"],
        agent=_named(raw["agent"], AGENTS, "agent", seed),
        method=_named(raw["method"], METHODS, "method", seed),
        loop=_named({"name": "icrl", **raw["loop"]}, LOOPS, "loop", seed),
        seed=seed,
        metrics_grid=(_named(raw["metrics_grid"], METRIC_GRIDS, "metrics_grid")
                      if "metrics_grid" in raw else None),
    )
    if config.featurizer.args:
        raise ValueError("featurizer: plain-function featurizers take no arguments")
    for env in (config.nominal_env, config.eval_env):
        check_true_cost_fits(config.true_cost.name, env.name)
    check_method_fits_loop(config.loop.name, config.method.name)
    if config.metrics_grid is not None:
        if config.metrics_grid.args:
            raise ValueError("metrics_grid: a grid takes no arguments")
        loop_fn = LOOPS[config.loop.name][0]
        if "metric_grid" not in inspect.signature(loop_fn).parameters:
            raise ValueError(f"metrics_grid: loop {config.loop.name!r} does not log ICL metrics")
        for env in (config.nominal_env, config.eval_env):
            check_metric_grid_fits(config.metrics_grid.name, env.name)
    return config


def save_config(config, path):
    """Write the config with every setting filled in (defaults too), in the
    format load_config reads, so the file alone reproduces the run."""
    def block(settings):
        values = asdict(settings) if is_dataclass(settings) else dict(settings)
        values.pop("seed", None)  # written once, at the top
        return {k: list(v) if isinstance(v, tuple) else v for k, v in values.items()}

    raw = {f.name: getattr(config, f.name) for f in fields(config)}  # Config's order
    for key in ("nominal_env", "eval_env", "true_cost", "featurizer", "agent", "method", "loop"):
        raw[key] = {"name": raw[key].name, **block(raw[key].args)}
    if raw["metrics_grid"] is None:
        del raw["metrics_grid"]  # optional: absent, not null
    else:
        raw["metrics_grid"] = {"name": raw["metrics_grid"].name}
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)


def _named(block, registry, where, seed=None):
    """{name: ..., rest} -> Named. With a seed, the registry entry's second
    item is a settings dataclass and the rest becomes that dataclass."""
    block = dict(block)
    name = block.pop("name", None)
    if name not in registry:
        raise ValueError(f"{where}: name {name!r} is not one of {sorted(registry)}")
    if seed is None:
        _check_no_numbers_or_booleans_as_text(block, where)
        return Named(name, block)
    settings_cls = registry[name][1]
    return Named(name, _settings(settings_cls, block, seed, where))


def _settings(settings_cls, block, seed, where):
    """Fill a settings dataclass from a config block; the config's one seed
    is added here."""
    _check_no_numbers_or_booleans_as_text(block, where)
    known = {f.name for f in fields(settings_cls)}
    for key in block:
        if key not in known:
            raise ValueError(f"{where}: unknown setting {key!r}")
        if key == "seed":
            raise ValueError(f"{where}: set the seed once, at the top of the config")
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in block.items()}
    return settings_cls(**values, seed=seed)


# YAML reads these unquoted as booleans; quoted, they are text, and any
# non-empty text counts as True in Python ("false" would mean on).
BOOLEAN_WORDS = {"true", "false", "yes", "no", "on", "off"}


def _check_no_numbers_or_booleans_as_text(block, where):
    """PyYAML reads 1e4 and 1.0e4 as text, not numbers (it needs 1.0e+4),
    and "false" in quotes as text, not a boolean; refuse any text that is
    really a number or a boolean."""
    for key, value in block.items():
        if isinstance(value, str):
            if value.lower() in BOOLEAN_WORDS:
                raise ValueError(f"{where}: {key} = {value!r} is text; write booleans unquoted: true or false")
            try:
                float(value)
            except ValueError:
                continue
            raise ValueError(f"{where}: {key} = {value!r} is text; write numbers like 1.0e+4 or 10000")
