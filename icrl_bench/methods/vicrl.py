"""Variational ICRL (VICRL, Liu et al., ICLR 2023): instead of one
feasibility ζ per step, learn a Beta(α, β) distribution over it.

Follows Liu's official code (github.com/Guiliang/ICRL-benchmarks-public,
commit 0d79d2c): constraint_models/constraint_net/variational_constraint_net.py
VariationalConstraintNet and what it inherits from constraint_net.py
ConstraintNet, as his HCWithPos config runs it (no importance sampling, so
no KL early stop; one full batch). Deviations: docs/methods.md §4."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class VICRLSettings:
    """Defaults: Liu's VariationalConstraintNet constructor where it has one
    (reg_coeff, prior, mode, eps); otherwise as our other methods. Per-env
    values live in configs."""
    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 3e-4
    anneal_lr_by_factor: float = 1.0  # learning rate in round k = learning_rate * factor**k
    backward_iters: int = 10  # exactly this many gradient steps per update
    reg_coeff: float = 0.0  # δ, weight of the KL to the prior
    prior: tuple[float, float] = (1.0, 1.0)  # (α, β) of the Beta prior (his di_prior)
    mode: str = "sample"  # cost = 1 - a draw of ζ ("sample") or 1 - its mean ("mean")
    eps: float = 1e-5  # ζ is clipped to [eps, 1] before its log
    # Feature columns the network sees, in order; None = all (Liu's code
    # has no column bug, docs/methods.md §4).
    input_columns: tuple[int, ...] | None = None
    # Seed for the network's starting weights only; None = the run's seed.
    # Liu's train_icrl.py starts every run from torch seed 0's network, bug
    # (6) (docs/methods.md); set 0 to reproduce it.
    init_seed: int | None = None
    seed: int = 0
    device: str = "cpu"


def beta_kl_to_prior(alpha, beta, prior):
    """KL(Beta(α, β) ‖ Beta(prior)) per row, in closed form: Liu's
    dirichlet_kl_divergence_loss (utils/model_utils.py) for two components.
    alpha, beta: (rows,) tensors; prior: (α0, β0)."""
    a0, b0 = (torch.tensor(float(p), device=alpha.device) for p in prior)
    total = alpha + beta
    return (torch.lgamma(total) - torch.lgamma(a0 + b0)
            + torch.lgamma(a0) + torch.lgamma(b0) - torch.lgamma(alpha) - torch.lgamma(beta)
            + (alpha - a0) * (torch.digamma(alpha) - torch.digamma(total))
            + (beta - b0) * (torch.digamma(beta) - torch.digamma(total)))


class VICRL:
    """Learns, per step, a Beta(α, β) distribution over its feasibility ζ:
    draws of ζ high on expert steps, low on nominal ones, each distribution
    pulled toward a prior. The cost the agent sees is 1 - ζ (a draw, or the
    mean)."""

    def __init__(self, n_features, settings=None):
        self.settings = settings or VICRLSettings()
        s = self.settings
        assert s.mode in ("sample", "mean"), s.mode
        torch.manual_seed(s.seed)
        if s.init_seed is None:
            self.net = self._build_net(n_features)
        else:
            # fork_rng saves torch's random state and restores it on exit, so
            # the run's own stream (Beta draws, the agent's sampling) still
            # follows the run seed, as in Liu's code, where the agent reseeds
            # torch right after his network is built.
            with torch.random.fork_rng():
                torch.manual_seed(s.init_seed)
                self.net = self._build_net(n_features)
        # eps=1e-5 avoids NaNs in Adam (ConstraintNet.__init__).
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=s.learning_rate, eps=1e-5)
        self.n_updates = 0  # the k-th update is ICRL round k

    def _build_net(self, n_features):
        """VariationalConstraintNet._build: SB3 create_mlp (Linear + ReLU per
        hidden layer, PyTorch default init) with 2 outputs, then softplus."""
        s = self.settings
        in_dim = n_features if s.input_columns is None else len(s.input_columns)
        layers = []
        for size in s.hidden_sizes:
            layers += [nn.Linear(in_dim, size), nn.ReLU()]
            in_dim = size
        layers += [nn.Linear(in_dim, 2), nn.Softplus()]
        return nn.Sequential(*layers).to(s.device)

    def _inputs(self, features):
        """Network input: a (steps, features) array as a float tensor,
        keeping only input_columns if set."""
        x = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=self.settings.device)
        if self.settings.input_columns is not None:
            x = x[:, list(self.settings.input_columns)]
        return x

    def _alpha_beta(self, x):
        out = self.net(x)
        return out[:, 0], out[:, 1]

    def update(self, expert, nominal):
        """One backward step (VariationalConstraintNet.train_nn, no importance
        sampling): backward_iters full-batch steps on
        -mean log ζ_E + mean log ζ_N + δ (mean KL_N + mean KL_E), each with
        fresh draws of ζ. expert, nominal: one (steps, features) array per
        trajectory. Returns Liu's backward metrics, averaged over the
        iterations as his."""
        s = self.settings
        # Round k's learning rate is learning_rate * factor**k (his
        # lr_schedule with progress 1 - k / n_iters).
        for group in self.optimizer.param_groups:
            group["lr"] = s.learning_rate * s.anneal_lr_by_factor ** self.n_updates
        self.n_updates += 1

        expert_x = self._inputs(np.concatenate(expert))
        nominal_x = self._inputs(np.concatenate(nominal))
        history = {k: [] for k in ["loss", "expert_loss", "nominal_loss", "regularizer_loss"]}
        nominal_draws, expert_draws = [], []

        for _ in range(s.backward_iters):
            nominal_alpha, nominal_beta = self._alpha_beta(nominal_x)
            expert_alpha, expert_beta = self._alpha_beta(expert_x)
            # rsample: a draw gradients can flow through (reparameterised).
            # torch.clip passes no gradient where it clips, as in his code.
            zeta_nominal = torch.clip(torch.distributions.Beta(nominal_alpha, nominal_beta).rsample(), s.eps, 1)
            zeta_expert = torch.clip(torch.distributions.Beta(expert_alpha, expert_beta).rsample(), s.eps, 1)

            expert_loss = torch.log(zeta_expert).mean()
            nominal_loss = torch.log(zeta_nominal).mean()  # importance weights all 1
            regularizer_loss = (beta_kl_to_prior(nominal_alpha, nominal_beta, s.prior).mean()
                                + beta_kl_to_prior(expert_alpha, expert_beta, s.prior).mean())
            loss = -expert_loss + nominal_loss + s.reg_coeff * regularizer_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for name, value in [("loss", loss), ("expert_loss", expert_loss),
                                ("nominal_loss", nominal_loss), ("regularizer_loss", regularizer_loss)]:
                history[name].append(value.item())
            nominal_draws.append(zeta_nominal.detach())
            expert_draws.append(zeta_expert.detach())

        nominal_draws, expert_draws = torch.cat(nominal_draws), torch.cat(expert_draws)
        return {
            "cn_loss": float(np.mean(history["loss"])),
            "expert_loss": float(np.mean(history["expert_loss"])),
            "unweighted_nominal_loss": torch.log(nominal_draws + s.eps).mean().item(),
            "nominal_loss": float(np.mean(history["nominal_loss"])),
            "regularizer_loss": float(np.mean(history["regularizer_loss"])),
            "is_mean": 1.0,
            "is_max": 1.0,
            "is_min": 1.0,
            "nominal/preds_max": nominal_draws.max().item(),
            "nominal/preds_min": nominal_draws.min().item(),
            "nominal/preds_mean": nominal_draws.mean().item(),
            "expert/preds_max": expert_draws.max().item(),
            "expert/preds_min": expert_draws.min().item(),
            "expert/preds_mean": expert_draws.mean().item(),
        }

    def cost(self, features):
        """Per-step cost for a (steps, features) array
        (VariationalConstraintNet.cost_function): 1 - a fresh draw of ζ
        ("sample", so the same step can cost differently each call) or
        1 - α/(α+β) ("mean")."""
        with torch.no_grad():
            alpha, beta = self._alpha_beta(self._inputs(features))
            if self.settings.mode == "sample":
                zeta = torch.distributions.Beta(alpha, beta).sample()
            else:
                zeta = alpha / (alpha + beta)
        return 1 - zeta.cpu().numpy()
