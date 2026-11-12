"""Maximum Entropy ICRL (Anwar, Malik et al., ICML 2021).

Follows Malik's official code (github.com/shehryar-malik/icrl, commit
f28214e): icrl/constraint_net.py ConstraintNet, cited where each piece is
used. Paper-vs-code differences and our deviations: docs/methods.md."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class MaximumEntropySettings:
    """Defaults: Malik's CLI defaults (icrl/icrl.py argparse). Per-env values
    live in configs."""
    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 3e-4
    anneal_lr_by_factor: float = 1.0  # learning rate in round k = learning_rate * factor**k
    backward_iters: int = 10  # at most this many gradient steps per update
    reg_coeff: float = 0.0  # δ
    importance_sampling: bool = True
    per_step_importance_sampling: bool = False
    target_kl_old_new: float = 10.0  # -1 = never stop on this KL
    target_kl_new_old: float = 10.0
    eps: float = 1e-5  # added inside logs and ratios
    # Feature columns the network sees, in order; None = all. Malik's network
    # sees (0, 0, 1) on LapGridWorld, by a bug (docs/methods.md, "Bugs in
    # the official code" (2)); set it only to match his code.
    input_columns: tuple[int, ...] | None = None
    # Seed for the network's starting weights only; None = the run's seed.
    # Liu's train_icrl.py starts every run from torch seed 0's network, bug
    # (6) (docs/methods.md); set 0 to reproduce it.
    init_seed: int | None = None
    seed: int = 0
    device: str = "cpu"


def importance_weights(zeta_old, zeta_new, lengths, per_step, eps):
    """Importance weights for the nominal steps, and the forward and reverse
    KL between the old and new constraint (ConstraintNet.compute_is_weights;
    paper supplementary Eqs. 15 and 21).

    zeta_old, zeta_new: ζ on every nominal step, flat, trajectories back to
    back, no gradients. lengths: steps per trajectory. Returns per-step
    weights (mean 1), kl_old_new, kl_new_old."""
    ratio = (zeta_new + eps) / (zeta_old + eps)
    # ζ_new(τ) / ζ_old(τ): product of the step ratios of each trajectory.
    prod = torch.stack([r.prod() for r in torch.split(ratio, list(lengths))])

    if per_step:
        weights = ratio / ratio.mean()
    else:
        per_trajectory = len(lengths) * prod / (prod.sum() + eps)
        weights = torch.repeat_interleave(per_trajectory, torch.as_tensor(lengths, device=ratio.device))

    # Forward KL without its log(mean prod) term; reverse KL is Eq. 21.
    kl_old_new = torch.mean(-torch.log(prod + eps))
    prod_mean = prod.mean()
    kl_new_old = torch.mean((prod - prod_mean) * torch.log(prod + eps) / (prod_mean + eps))
    return weights, kl_old_new, kl_new_old


class MaximumEntropy:
    """Learns ζ(features) in (0, 1), the feasibility of a step: high on what
    the expert does, low on what the agent does and the expert doesn't.
    The cost the agent sees is 1 - ζ."""

    def __init__(self, n_features, settings=None):
        self.settings = settings or MaximumEntropySettings()
        s = self.settings
        # Malik never seeds this network; we do, so runs are reproducible.
        torch.manual_seed(s.seed)
        if s.init_seed is None:
            self.net = self._build_net(n_features)
        else:
            # fork_rng saves torch's random state and restores it on exit, so
            # the run's own stream still follows the run seed (as VICRL).
            with torch.random.fork_rng():
                torch.manual_seed(s.init_seed)
                self.net = self._build_net(n_features)
        # eps=1e-5 avoids NaNs in Adam (ConstraintNet.__init__).
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=s.learning_rate, eps=1e-5)
        self.n_updates = 0  # the k-th update is ICRL round k

    def _build_net(self, n_features):
        """ConstraintNet._build: SB3 create_mlp (Linear + ReLU per hidden
        layer, PyTorch default init) then a sigmoid."""
        s = self.settings
        in_dim = n_features if s.input_columns is None else len(s.input_columns)
        layers = []
        for size in s.hidden_sizes:
            layers += [nn.Linear(in_dim, size), nn.ReLU()]
            in_dim = size
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        return nn.Sequential(*layers).to(s.device)

    def _inputs(self, features):
        """Network input: a (steps, features) array as a float tensor,
        keeping only input_columns if set (ConstraintNet.prepare_data's
        select_appropriate_dims)."""
        x = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=self.settings.device)
        if self.settings.input_columns is not None:
            x = x[:, list(self.settings.input_columns)]
        return x

    def update(self, expert, nominal):
        """One backward step (ConstraintNet.train): at most backward_iters
        full-batch gradient steps pushing ζ up on expert steps and down on
        nominal steps. expert, nominal: one (steps, features) array per
        trajectory. Returns Malik's backward metrics."""
        s = self.settings
        # Round k's learning rate is learning_rate * factor**k (icrl.py's
        # cn_lr_schedule, applied in ConstraintNet._update_learning_rate).
        for group in self.optimizer.param_groups:
            group["lr"] = s.learning_rate * s.anneal_lr_by_factor ** self.n_updates
        self.n_updates += 1

        expert_x = self._inputs(np.concatenate(expert))
        nominal_x = self._inputs(np.concatenate(nominal))
        lengths = [len(t) for t in nominal]

        # ζ_old: the constraint the nominal data was sampled under.
        if s.importance_sampling:
            with torch.no_grad():
                zeta_start = self.net(nominal_x).squeeze(-1)

        early_stop_itr = s.backward_iters
        for itr in range(s.backward_iters):
            if s.importance_sampling:
                with torch.no_grad():
                    zeta_now = self.net(nominal_x).squeeze(-1)
                weights, kl_old_new, kl_new_old = importance_weights(
                    zeta_start, zeta_now, lengths, s.per_step_importance_sampling, s.eps)
                # Stop once the old sample is too stale (paper §3.5).
                if ((s.target_kl_old_new != -1 and kl_old_new > s.target_kl_old_new) or
                        (s.target_kl_new_old != -1 and kl_new_old > s.target_kl_new_old)):
                    early_stop_itr = itr
                    break
            else:
                weights = torch.ones(len(nominal_x), device=s.device)

            zeta_nominal = self.net(nominal_x).squeeze(-1)
            zeta_expert = self.net(expert_x).squeeze(-1)
            expert_loss = torch.mean(torch.log(zeta_expert + s.eps))
            log_nominal = torch.log(zeta_nominal + s.eps)
            if s.importance_sampling and s.per_step_importance_sampling:
                # Malik's shape bug, reproduced on purpose: his per-step
                # weights broadcast against every step, so the term is
                # mean(weights) * mean(log ζ) (docs/methods.md).
                nominal_loss = weights.mean() * log_nominal.mean()
            else:
                nominal_loss = torch.mean(weights * log_nominal)
            regularizer_loss = s.reg_coeff * (torch.mean(1 - zeta_expert) + torch.mean(1 - zeta_nominal))
            loss = -expert_loss + nominal_loss + regularizer_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # As in Malik's train: values from the last gradient step taken.
        metrics = {
            "cn_loss": loss.item(),
            "expert_loss": expert_loss.item(),
            "unweighted_nominal_loss": log_nominal.mean().item(),
            "nominal_loss": nominal_loss.item(),
            "regularizer_loss": regularizer_loss.item(),
            "is_mean": weights.mean().item(),
            "is_max": weights.max().item(),
            "is_min": weights.min().item(),
            "nominal_preds_max": zeta_nominal.max().item(),
            "nominal_preds_min": zeta_nominal.min().item(),
            "nominal_preds_mean": zeta_nominal.mean().item(),
            "expert_preds_max": zeta_expert.max().item(),
            "expert_preds_min": zeta_expert.min().item(),
            "expert_preds_mean": zeta_expert.mean().item(),
        }
        if s.importance_sampling:
            metrics.update({"kl_old_new": kl_old_new.item(), "kl_new_old": kl_new_old.item(),
                            "early_stop_itr": early_stop_itr})
        return metrics

    def cost(self, features):
        """Per-step cost 1 - ζ for a (steps, features) array
        (ConstraintNet.cost_function)."""
        with torch.no_grad():
            zeta = self.net(self._inputs(features)).squeeze(-1)
        return 1 - zeta.cpu().numpy()
