"""Binary Classifier ICRL (Malik et al. 2021, §4 baselines: "train a binary
classifier (using the cross-entropy loss) to classify between nominal and
expert trajectories"). Port of Malik's ConstraintNet with --train_gail_lambda
and -nis (github.com/shehryar-malik/icrl, commit f28214e,
icrl/constraint_net.py), as in every README command for it. The network part
is a copy of maximum_entropy.py (docs/decisions.md, 2026-10-21); deviations
in docs/methods.md."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class BinaryClassifierSettings:
    """Defaults: Malik's CLI defaults (icrl/icrl.py argparse). Per-env values
    live in configs."""
    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 3e-4
    anneal_lr_by_factor: float = 1.0  # learning rate in round k = learning_rate * factor**k
    backward_iters: int = 10  # exactly this many gradient steps per update (-nis: no early stop)
    eps: float = 1e-5  # only in the unweighted_nominal_loss metric; BCE clamps its logs instead
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


class BinaryClassifier:
    """Learns ζ(features) in (0, 1) by labelling every expert step feasible
    (1) and every nominal step infeasible (0). The cost the agent sees is
    1 - ζ."""

    def __init__(self, n_features, settings=None):
        self.settings = settings or BinaryClassifierSettings()
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
        """One backward step (ConstraintNet.train, train_gail_lambda branch):
        backward_iters full-batch gradient steps on BCE(ζ_nominal, 0) +
        BCE(ζ_expert, 1). expert, nominal: one (steps, features) array per
        trajectory. Returns Malik's backward metrics."""
        s = self.settings
        # Round k's learning rate is learning_rate * factor**k (icrl.py's
        # cn_lr_schedule, applied in ConstraintNet._update_learning_rate).
        for group in self.optimizer.param_groups:
            group["lr"] = s.learning_rate * s.anneal_lr_by_factor ** self.n_updates
        self.n_updates += 1

        expert_x = self._inputs(np.concatenate(expert))
        nominal_x = self._inputs(np.concatenate(nominal))

        for _ in range(s.backward_iters):
            zeta_nominal = self.net(nominal_x).squeeze(-1)
            zeta_expert = self.net(expert_x).squeeze(-1)
            # nn.BCELoss: mean over steps; each log is clamped at -100.
            nominal_loss = nn.functional.binary_cross_entropy(zeta_nominal, torch.zeros_like(zeta_nominal))
            expert_loss = nn.functional.binary_cross_entropy(zeta_expert, torch.ones_like(zeta_expert))
            loss = nominal_loss + expert_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # As in Malik's train: values from the last gradient step; with -nis
        # the importance weights are all 1 and there are no KL metrics.
        return {
            "cn_loss": loss.item(),
            "expert_loss": expert_loss.item(),
            "unweighted_nominal_loss": torch.log(zeta_nominal + s.eps).mean().item(),
            "nominal_loss": nominal_loss.item(),
            "regularizer_loss": 0.0,  # no regulariser in this branch (-crc is ignored)
            "is_mean": 1.0,
            "is_max": 1.0,
            "is_min": 1.0,
            "nominal_preds_max": zeta_nominal.max().item(),
            "nominal_preds_min": zeta_nominal.min().item(),
            "nominal_preds_mean": zeta_nominal.mean().item(),
            "expert_preds_max": zeta_expert.max().item(),
            "expert_preds_min": zeta_expert.min().item(),
            "expert_preds_mean": zeta_expert.mean().item(),
        }

    def cost(self, features):
        """Per-step cost 1 - ζ for a (steps, features) array
        (ConstraintNet.cost_function)."""
        with torch.no_grad():
            zeta = self.net(self._inputs(features)).squeeze(-1)
        return 1 - zeta.cpu().numpy()
