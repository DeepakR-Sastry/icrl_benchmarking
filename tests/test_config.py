import pytest
import yaml

from icrl_bench.agents.ppo_lagrangian import PPOLagrangianSettings
from icrl_bench.config import Named, load_config, parse_config, save_config
from icrl_bench.core.loop import GACLLoopSettings, ICRLLoopSettings
from icrl_bench.methods.maximum_entropy import MaximumEntropySettings

TEXT = """
nominal_env: {name: lap_grid_world}
eval_env: {name: lap_grid_world, constrained: true}
true_cost: {name: lap_grid_world}
featurizer: {name: lap_grid_world}
expert: expert_data/lgw_forward.pkl
n_envs: 5
agent: {name: ppo_lagrangian, target_kl: 0.01}
method: {name: maximum_entropy, hidden_sizes: [20], learning_rate: 0.003,
         backward_iters: 20, input_columns: [0, 0, 1]}
loop: {n_rounds: 12, steps_per_round: 10000}
seed: 3
"""


def raw(**changes):
    """The config above as a dict, with some top-level entries replaced."""
    return {**yaml.safe_load(TEXT), **changes}


def test_reads_the_settings_validate_icrl_lgw_uses(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(TEXT)
    config = load_config(path)
    assert config.eval_env.name == "lap_grid_world"
    assert config.eval_env.args == {"constrained": True}
    assert config.agent.args == PPOLagrangianSettings(target_kl=0.01, seed=3)
    # Lists become tuples, as the dataclass fields are typed.
    assert config.method.args == MaximumEntropySettings(
        hidden_sizes=(20,), learning_rate=0.003, backward_iters=20, input_columns=(0, 0, 1), seed=3)
    # No loop name: the ICRL loop (docs/decisions.md, 2026-11-01).
    assert config.loop == Named("icrl", ICRLLoopSettings(n_rounds=12, steps_per_round=10_000, seed=3))


@pytest.mark.parametrize("changes, message", [
    (dict(extra=1), "unknown \\['extra'\\]"),
    (dict(method={"name": "maximum_entropi"}), "is not one of"),
    (dict(method={"name": "maximum_entropy", "hidden_size": [20]}), "unknown setting 'hidden_size'"),
    (dict(agent={"name": "ppo_lagrangian", "seed": 1}), "set the seed once"),
    (dict(loop={"steps_per_round": "1e4"}), "is text"),
    (dict(eval_env={"name": "lap_grid_world", "max_episode_steps": "1.0e4"}), "is text"),
    (dict(method={"name": "maximum_entropy", "importance_sampling": "false"}), "is text"),
    (dict(nominal_env={"name": "lap_grid_world", "constrained": "False"}), "is text"),
    (dict(featurizer={"name": "lap_grid_world", "scale": 2}), "take no arguments"),
    (dict(loop={"name": "gail"}), "is not one of"),
    (dict(loop={"name": "gacl", "n_rounds": 12}), "unknown setting 'n_rounds'"),
    (dict(method={"name": "gacl"}), "does not run in loop 'icrl'"),
    (dict(loop={"name": "gacl"}), "does not run in loop 'gacl'"),
])
def test_mistakes_are_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        parse_config(raw(**changes))


def test_a_true_cost_that_does_not_fit_an_env_is_refused(monkeypatch):
    from icrl_bench import registry
    monkeypatch.setitem(registry.TRUE_COSTS, "wall_behind", (None, {"some_mujoco_env"}))
    with pytest.raises(ValueError, match="does not apply"):
        parse_config(raw(true_cost={"name": "wall_behind"}))


def test_a_saved_config_reads_back_identical(tmp_path):
    config = parse_config(raw())
    save_config(config, tmp_path / "saved.yaml")
    assert load_config(tmp_path / "saved.yaml") == config
    # Defaults are written out, not just what the file set.
    saved = yaml.safe_load((tmp_path / "saved.yaml").read_text())
    assert saved["method"]["reg_coeff"] == MaximumEntropySettings().reg_coeff
    assert "seed" not in saved["agent"]


def test_a_seed_override_reaches_every_settings_block(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(TEXT)
    config = load_config(path, seed=7)
    assert config.seed == config.agent.args.seed == config.method.args.seed == config.loop.args.seed == 7


def test_a_named_loop_gets_its_own_settings_and_is_saved_by_name(tmp_path):
    config = parse_config(raw(agent={"name": "ppo"}, method={"name": "gacl"},
                              loop={"name": "gacl", "total_steps": 120000}))
    assert config.loop == Named("gacl", GACLLoopSettings(total_steps=120_000, seed=3))
    save_config(config, tmp_path / "saved.yaml")
    assert yaml.safe_load((tmp_path / "saved.yaml").read_text())["loop"]["name"] == "gacl"
    assert load_config(tmp_path / "saved.yaml") == config


def hc_icl(**changes):
    """An ICL HalfCheetah config as a dict (only what the checks need)."""
    return raw(**{"nominal_env": {"name": "half_cheetah_with_pos"},
                  "eval_env": {"name": "half_cheetah_with_pos"},
                  "true_cost": {"name": "half_cheetah_with_pos", "wall_x": -1},
                  "featurizer": {"name": "half_cheetah_with_pos_x"},
                  "method": {"name": "icl"}, "loop": {"name": "icl"},
                  "metrics_grid": {"name": "half_cheetah_with_pos"}, **changes})


def test_a_metrics_grid_is_read_saved_and_read_back(tmp_path):
    config = parse_config(hc_icl())
    assert config.metrics_grid == Named("half_cheetah_with_pos", {})
    save_config(config, tmp_path / "saved.yaml")
    assert yaml.safe_load((tmp_path / "saved.yaml").read_text())["metrics_grid"] == {
        "name": "half_cheetah_with_pos"}
    assert load_config(tmp_path / "saved.yaml") == config


def test_no_metrics_grid_stays_absent(tmp_path):
    config = parse_config(raw())
    assert config.metrics_grid is None
    save_config(config, tmp_path / "saved.yaml")
    assert "metrics_grid" not in yaml.safe_load((tmp_path / "saved.yaml").read_text())


@pytest.mark.parametrize("changes, message", [
    (dict(loop={"name": "icrl"}, method={"name": "maximum_entropy"}), "does not log ICL metrics"),
    (dict(metrics_grid={"name": "half_cheetah_with_pos", "x": 1}), "takes no arguments"),
    (dict(metrics_grid={"name": "nope"}), "is not one of"),
])
def test_metrics_grid_mistakes_are_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        parse_config(hc_icl(**changes))


def test_a_metrics_grid_on_the_wrong_env_is_refused():
    with pytest.raises(ValueError, match="does not apply"):
        parse_config(raw(method={"name": "icl"}, loop={"name": "icl"},
                         metrics_grid={"name": "half_cheetah_with_pos"}))
