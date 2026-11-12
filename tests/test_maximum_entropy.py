import numpy as np
import pytest
import torch

from icrl_bench.methods.maximum_entropy import MaximumEntropy, MaximumEntropySettings, importance_weights

# Hand-made LapGridWorld-style rows, in the featurizer's layout
# [cell, is_forward, is_backward].
FORWARD_ROW = [-0.9, 1.0, 0.0]
BACKWARD_ROW = [-0.9, 0.0, 1.0]


def trajectory(*rows):
    return np.array(rows, dtype=np.float32)


EXPERT = [trajectory(FORWARD_ROW, FORWARD_ROW, FORWARD_ROW)] * 2
NOMINAL = [trajectory(FORWARD_ROW, BACKWARD_ROW, BACKWARD_ROW)] * 2


def method(**settings):
    return MaximumEntropy(3, MaximumEntropySettings(hidden_sizes=(20,), learning_rate=0.01, **settings))


def test_only_nominal_steps_become_costly():
    m = method(backward_iters=50, target_kl_old_new=-1, target_kl_new_old=-1)
    before = m.cost(trajectory(FORWARD_ROW, BACKWARD_ROW))
    m.update(EXPERT, NOMINAL)
    forward_cost, backward_cost = m.cost(trajectory(FORWARD_ROW, BACKWARD_ROW))
    assert backward_cost > before[1]
    assert backward_cost > forward_cost + 0.2


def test_importance_weights_match_the_hand_worked_example():
    # Trajectory A: two steps 0.6 -> 0.7. Trajectory B: 0.6 -> 0.7, 0.6 -> 0.3.
    zeta_old = torch.tensor([0.6, 0.6, 0.6, 0.6])
    zeta_new = torch.tensor([0.7, 0.7, 0.7, 0.3])
    weights, kl_old_new, kl_new_old = importance_weights(zeta_old, zeta_new, [2, 2], per_step=False, eps=1e-5)
    assert weights.tolist() == pytest.approx([1.4, 1.4, 0.6, 0.6], abs=1e-4)
    assert kl_old_new.item() == pytest.approx(0.1153, abs=1e-3)
    assert kl_new_old.item() == pytest.approx(0.1695, abs=1e-3)


def test_per_step_weights_are_not_applied_in_the_loss_as_in_malik():
    m = method(backward_iters=5, per_step_importance_sampling=True)
    metrics = m.update(EXPERT, NOMINAL)
    assert metrics["is_max"] > metrics["is_min"]  # the weights do differ...
    # ...but the nominal term is mean(weights) * the unweighted term.
    assert metrics["nominal_loss"] == pytest.approx(metrics["is_mean"] * metrics["unweighted_nominal_loss"])


def test_learning_rate_is_annealed_once_per_update():
    m = method(backward_iters=1, anneal_lr_by_factor=0.5)
    for k in range(3):
        m.update(EXPERT, NOMINAL)
        assert m.optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * 0.5 ** k)


def test_kl_limit_stops_the_update_early():
    m = method(backward_iters=20, target_kl_old_new=1e-9, target_kl_new_old=1e-9)
    # Two different nominal trajectories, so one gradient step changes their
    # products ζ_new(τ)/ζ_old(τ) by different amounts: the reverse KL, which
    # measures how spread out those products are, becomes > 0. (With
    # identical trajectories it would stay exactly 0.)
    nominal = [trajectory(FORWARD_ROW, BACKWARD_ROW, BACKWARD_ROW),
               trajectory(FORWARD_ROW, FORWARD_ROW, FORWARD_ROW)]
    metrics = m.update(EXPERT, nominal)
    # Iteration 0 always passes (ζ unchanged); the first step moves ζ.
    assert metrics["early_stop_itr"] == 1


def test_input_columns_choose_what_the_network_sees():
    # Malik's LapGridWorld selection: [cell, cell, is_forward].
    m = method(input_columns=(0, 0, 1))
    assert m.net[0].in_features == 3
    rows = trajectory(FORWARD_ROW, BACKWARD_ROW)
    with torch.no_grad():
        expected = 1 - m.net(torch.tensor(rows[:, [0, 0, 1]])).squeeze(-1).numpy()
    assert m.cost(rows) == pytest.approx(expected)
    # The backward row differs from the forward one only in the dropped
    # column and in is_forward, so the selection keeps the information.
    assert m.cost(rows)[0] != m.cost(rows)[1]


def test_init_seed_fixes_only_the_starting_network():
    # Liu's bug (6) on his ME path (same train_icrl.py as VICRL): every run
    # starts from torch seed 0's network; the rest follows its own seed.
    def build(seed, init_seed):
        method = MaximumEntropy(3, MaximumEntropySettings(hidden_sizes=(8,), seed=seed, init_seed=init_seed))
        weights = [p.detach().clone() for p in method.net.parameters()]
        return weights, torch.rand(3)   # the run's stream right after construction

    w_plain0, _ = build(seed=0, init_seed=None)
    w3, draw3 = build(seed=3, init_seed=0)
    w7, draw7 = build(seed=7, init_seed=0)
    for a, b, c in zip(w_plain0, w3, w7):
        assert torch.equal(a, b) and torch.equal(a, c)   # seed 0's network, whatever the run seed
    assert not torch.equal(draw3, draw7)                  # the run's own draws still differ
    torch.manual_seed(3)
    assert torch.equal(draw3, torch.rand(3))              # = run seed 3's fresh stream
    w_plain3, _ = build(seed=3, init_seed=None)
    assert not torch.equal(w_plain3[0], w_plain0[0])      # unset: follows the run seed
