import pytest

from icrl_bench.registry import (AGENTS, ENVS, LOOPS, METHODS, TRUE_COSTS, check_method_fits_loop,
                                 check_true_cost_fits)


def test_every_rule_lists_only_registered_envs():
    for name, (_, env_names) in TRUE_COSTS.items():
        assert env_names <= set(ENVS), name


def test_a_listed_pairing_passes():
    check_true_cost_fits("lap_grid_world", "lap_grid_world")


def test_an_unlisted_pairing_is_refused():
    with pytest.raises(ValueError, match="does not apply"):
        check_true_cost_fits("lap_grid_world", "other_env")


def test_every_agent_and_method_builds_from_its_default_settings():
    discrete, continuous = ENVS["lap_grid_world"](), ENVS["half_cheetah_with_pos"]()
    for name, (cls, settings_cls) in AGENTS.items():
        cls(continuous.observation_space, continuous.action_space, settings_cls())
        if name == "ppo_lagrangian_ray":  # his PPOLag envs are all continuous
            with pytest.raises(ValueError, match="Box"):
                cls(discrete.observation_space, discrete.action_space, settings_cls())
        else:
            cls(discrete.observation_space, discrete.action_space, settings_cls())
    for cls, settings_cls in METHODS.values():
        cls(3, settings_cls())


def test_every_method_runs_in_exactly_one_loop():
    for name in METHODS:
        assert sum(name in methods for _, _, methods in LOOPS.values()) == 1, name
    for _, _, methods in LOOPS.values():
        assert methods <= set(METHODS)


def test_a_method_in_the_wrong_loop_is_refused():
    check_method_fits_loop("gacl", "gacl")
    check_method_fits_loop("icrl", "binary_classifier")
    check_method_fits_loop("icl", "icl")
    with pytest.raises(ValueError, match="does not run in loop"):
        check_method_fits_loop("icrl", "gacl")
    # run_icrl would run ICL without its flow or snapshots.
    with pytest.raises(ValueError, match="does not run in loop"):
        check_method_fits_loop("icrl", "icl")
