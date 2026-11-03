import numpy as np
import torch

from icrl_bench.methods.vicrl import VICRL, VICRLSettings, beta_kl_to_prior

# LapGridWorld-like features [cell, is_forward, is_backward].
EXPERT = [np.array([[0.0, 1, 0], [0.5, 1, 0], [1.0, 1, 0]])]
NOMINAL = [np.array([[0.0, 0, 1], [-0.5, 0, 1]]), np.array([[0.25, 0, 1], [0.75, 0, 1]])]


def test_kl_matches_torchs_beta_kl():
    # Liu's closed form for two components = PyTorch's own Beta-Beta KL.
    alpha = torch.tensor([0.3, 1.0, 5.0, 9.0])
    beta = torch.tensor([2.0, 1.0, 5.0, 0.2])
    ours = beta_kl_to_prior(alpha, beta, (0.9, 0.1))
    reference = torch.distributions.kl_divergence(torch.distributions.Beta(alpha, beta),
                                                  torch.distributions.Beta(torch.tensor(0.9), torch.tensor(0.1)))
    assert torch.allclose(ours, reference, rtol=1e-4)
    assert torch.allclose(beta_kl_to_prior(torch.tensor([0.9]), torch.tensor([0.1]), (0.9, 0.1)),
                          torch.zeros(1), atol=1e-5)   # the prior itself: KL 0


def test_network_gives_positive_alpha_beta_per_step():
    method = VICRL(3, VICRLSettings(hidden_sizes=(8,)))
    alpha, beta = method._alpha_beta(method._inputs(np.concatenate(NOMINAL)))
    assert alpha.shape == beta.shape == (4,)
    assert (alpha > 0).all() and (beta > 0).all()


def test_sample_cost_is_random_mean_cost_is_not():
    features = np.concatenate(NOMINAL)
    sampled = VICRL(3, VICRLSettings(hidden_sizes=(8,), mode="sample"))
    a, b = sampled.cost(features), sampled.cost(features)
    assert a.shape == (4,) and np.all((a >= 0) & (a <= 1)) and not np.array_equal(a, b)
    mean = VICRL(3, VICRLSettings(hidden_sizes=(8,), mode="mean"))
    alpha, beta = mean._alpha_beta(mean._inputs(features))
    expected = 1 - (alpha / (alpha + beta)).detach().numpy()
    assert np.allclose(mean.cost(features), expected) and np.array_equal(mean.cost(features), mean.cost(features))


def test_update_takes_exactly_backward_iters_steps_at_the_annealed_rate():
    method = VICRL(3, VICRLSettings(hidden_sizes=(8,), learning_rate=0.01, anneal_lr_by_factor=0.9,
                                    backward_iters=7))
    method.update(EXPERT, NOMINAL)
    assert method.optimizer.state[method.net[0].weight]["step"] == 7
    method.update(EXPERT, NOMINAL)
    assert np.isclose(method.optimizer.param_groups[0]["lr"], 0.01 * 0.9)   # round 1


def test_metrics_follow_the_loss():
    method = VICRL(3, VICRLSettings(hidden_sizes=(8,), reg_coeff=0.5, prior=(0.9, 0.1), backward_iters=1))
    m = method.update(EXPERT, NOMINAL)
    # One iteration: the logged loss is exactly its parts.
    assert np.isclose(m["cn_loss"], -m["expert_loss"] + m["nominal_loss"] + 0.5 * m["regularizer_loss"])
    assert m["regularizer_loss"] > 0
    assert 1e-5 <= m["nominal/preds_min"] <= m["nominal/preds_max"] <= 1


def test_repeated_updates_separate_expert_from_nominal():
    method = VICRL(3, VICRLSettings(hidden_sizes=(16,), learning_rate=0.01, reg_coeff=0.1,
                                    prior=(0.9, 0.1), mode="mean"))
    for _ in range(100):
        method.update(EXPERT, NOMINAL)
    cost_expert = method.cost(np.concatenate(EXPERT)).mean()
    cost_nominal = method.cost(np.concatenate(NOMINAL)).mean()
    assert cost_expert < 0.2 and cost_nominal > 0.8


def test_init_seed_fixes_only_the_starting_network():
    # Liu's bug (6): every run starts from torch seed 0's network, while the
    # rest of the run follows its own seed.
    def build(seed, init_seed):
        method = VICRL(3, VICRLSettings(hidden_sizes=(8,), seed=seed, init_seed=init_seed))
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
    # Unset: the network follows the run seed, as before.
    w_plain3, _ = build(seed=3, init_seed=None)
    assert not torch.equal(w_plain3[0], w_plain0[0])
