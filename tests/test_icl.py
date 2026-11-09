import numpy as np
import pytest
import torch

from icrl_bench.methods.icl import ICL, ICLSettings


def constant_half_cost(method):
    """Last layer all zeros: sigmoid(0) = 0.5 on every step."""
    last = method.net[-2]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()


class FirstColumnFlow:
    """Stand-in flow: -log p of a step = its first feature."""

    def log_probs(self, features):
        return -torch.as_tensor(np.asarray(features)[:, :1], dtype=torch.float32)


def method_with_stand_in_flow(**settings):
    method = ICL(2, ICLSettings(**settings))
    method.flow = FirstColumnFlow()
    method.expert_nll = (1.0, 0.5)  # threshold 1.5
    return method


# First column chosen per step; the second is not read by the stand-in flow.
NOMINAL = [np.array([[0.0, 0], [0.0, 0]]),            # τ1: no step above 1.5
           np.array([[1.0, 0], [2.0, 0], [3.0, 0]])]  # τ2: 2 of 3 above 1.5
EXPERT = [np.zeros((1, 2)), np.zeros((4, 2))]


def test_discounted_cost_sums_gamma_powers():
    method = ICL(2, ICLSettings(discount=0.9))
    constant_half_cost(method)
    totals = method._discounted_costs([np.zeros((3, 2)), np.zeros((1, 2))])
    # 0.5 * (1 + 0.9 + 0.81) and 0.5.
    assert torch.allclose(totals, torch.tensor([1.355, 0.5]))


def test_dissimilarity_is_the_share_of_steps_above_mean_plus_std():
    method = method_with_stand_in_flow()
    assert np.allclose(method.dissimilarity(NOMINAL), [0.0, 2 / 3])


def test_update_loss_terms_on_hand_checked_numbers():
    # Learning rate 0: the steps measure without changing the network.
    method = method_with_stand_in_flow(discount=1.0, beta=0.99, alpha=15.0,
                                       learning_rate=0.0, updates_per_epoch=1)
    constant_half_cost(method)
    metrics = method.update(EXPERT, NOMINAL)
    # Weights [0, 2/3] / (2/3) = [0, 1]; J = [1.0, 1.5] -> -(0 * 1.0 + 1 * 1.5).
    assert metrics["avg_objective1"] == pytest.approx(-1.5)
    # Expert J = [0.5, 2.0]: overshoot per trajectory [0, 1.01], then the mean
    # (his code; the paper's ReLU of the mean would give 0.26).
    assert metrics["avg_objective2"] == pytest.approx(0.505)
    assert metrics["avg_agent_cost"] == pytest.approx(1.25)
    assert metrics["expert_satisfaction"] == pytest.approx(0.5)


def test_fit_flow_keeps_mean_and_sample_std_of_expert_nll():
    rng = np.random.RandomState(0)
    expert = [rng.randn(20, 2), rng.randn(15, 2)]
    method = ICL(2, ICLSettings(flow_epochs=2))
    method.fit_flow(expert)
    with torch.no_grad():
        nll = -method.flow.log_probs(np.concatenate(expert)).numpy().ravel()
    assert method.expert_nll[0] == pytest.approx(nll.mean(), rel=1e-5)
    assert method.expert_nll[1] == pytest.approx(nll.std(ddof=1), rel=1e-5)  # n - 1


def test_cost_is_one_value_in_zero_one_per_step():
    method = ICL(2)
    costs = method.cost(np.random.RandomState(0).randn(5, 2) * 10)
    assert costs.shape == (5,) and np.all((costs >= 0) & (costs <= 1))
