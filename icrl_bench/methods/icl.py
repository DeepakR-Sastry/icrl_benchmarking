"""Inverse Constraint Learning (ICL, Gaurav et al., ICLR 2023): learn a soft
constraint, a per-step cost c(s, a) in [0, 1] whose discounted sum must
stay <= beta on expert trajectories, by raising it on agent trajectories
that a normalizing flow finds unlike the expert.

Follows Gaurav's official code (github.com/ashishgaurav13/ICL, commit
096e422): tools/functions/cost_function.py CostFunction (network),
icl.py L29-42 (flow on the expert, its score statistics) and
tools/algorithms/cost_adjustment.py CostAdjustment.train (one update).
Choosing which policies the agent data comes from is the loop's job
(run_icl). Deviations: docs/methods.md §5."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from icrl_bench.methods.flow import Flow, FlowSettings


@dataclass
class ICLSettings:
    """Defaults: his Gridworld (A) config (configs/gridworldA.json) and
    icl.py's h=64; per-env values live in our configs."""
    hidden: int = 64  # cost net width (icl.py L20)
    learning_rate: float = 5e-4  # his learning_rate: cost net and flow
    discount: float = 1.0  # his discount_factor, γ in the discounted cost
    beta: float = 0.99  # budget on an expert trajectory's discounted cost
    alpha: float = 15.0  # weight of the expert (budget) term
    updates_per_epoch: int = 20  # gradient steps per update
    flow_epochs: int = 20
    flow_batch_size: int = 64  # his minibatch_size
    normalize_flow_inputs: bool = False
    seed: int = 0


def _init_weights(module):
    """His TorchHelper.init_weights: Xavier-uniform weights, bias 0.01. It
    reaches the cost net (a plain Sequential), unlike the flow (quirk (e))."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)


class ICL:
    """Holds the learned cost and the expert-fitted flow. Call fit_flow once
    on the expert pile, then update once per cost-adjustment call."""

    def __init__(self, n_features, settings=None):
        self.settings = settings or ICLSettings()
        s = self.settings
        torch.manual_seed(s.seed)
        h = s.hidden
        self.net = nn.Sequential(nn.Linear(n_features, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, 1), nn.Sigmoid())
        self.net.apply(_init_weights)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=s.learning_rate)
        self.flow = None
        self.expert_nll = None  # (mean, std) of -log p over expert steps

    def fit_flow(self, expert):
        """Fit the flow on every expert step and keep the mean and std of
        their -log p (icl.py L32-42). Returns the last epoch's metrics."""
        s = self.settings
        steps = np.concatenate(expert)
        self.flow = Flow(steps, FlowSettings(learning_rate=s.learning_rate,
                                             batch_size=s.flow_batch_size,
                                             normalize_inputs=s.normalize_flow_inputs))
        for _ in range(s.flow_epochs):
            metrics = self.flow.train_epoch()
        with torch.no_grad():
            nll = -self.flow.log_probs(steps)
        # torch's std divides by n - 1, as his ep.std().
        self.expert_nll = (nll.mean().item(), nll.std().item())
        return {**metrics, "expert_nll_mean": self.expert_nll[0],
                "expert_nll_std": self.expert_nll[1]}

    def dissimilarity(self, trajectories):
        """Per trajectory: the fraction of its steps whose -log p under the
        flow is above the expert's mean + std (paper Alg. 3 line 3)."""
        threshold = self.expert_nll[0] + self.expert_nll[1]
        scores = []
        with torch.no_grad():
            for features in trajectories:
                nll = -self.flow.log_probs(features)
                scores.append((nll > threshold).float().mean().item())
        return np.array(scores)

    def _discounted_costs(self, trajectories):
        """Σ_t γ^t c(s_t, a_t) per trajectory (his CostAdjustment.Jc)."""
        totals = []
        for features in trajectories:
            costs = self.net(torch.as_tensor(features, dtype=torch.float32)).squeeze(-1)
            discounts = torch.tensor([self.settings.discount ** t for t in range(len(costs))])
            totals.append((costs * discounts).sum())
        return torch.stack(totals)

    def update(self, expert, nominal):
        """One call of his CostAdjustment.train: updates_per_epoch Adam steps
        on -Σ w(τ) J(τ) over nominal + α · mean over expert of
        max(0, J(τ) - β), with w the nominal dissimilarities scaled to sum 1."""
        s = self.settings
        weights = torch.tensor(self.dissimilarity(nominal), dtype=torch.float32)
        weights = weights / weights.sum()
        with torch.no_grad():
            metrics = {
                "avg_agent_cost": self._discounted_costs(nominal).mean().item(),
                "expert_satisfaction": (self._discounted_costs(expert) < s.beta).float().mean().item(),
            }
        objective1s, objective2s = [], []
        for _ in range(s.updates_per_epoch):
            self.optimizer.zero_grad()
            objective1 = -(self._discounted_costs(nominal) * weights).sum()
            expert_costs = self._discounted_costs(expert)
            # His (J >= β) * (J - β): the budget overshoot of each expert
            # trajectory, averaged after (paper: ReLU of the average, quirk (a)).
            objective2 = ((expert_costs >= s.beta).float() * (expert_costs - s.beta)).mean()
            (objective1 + objective2 * s.alpha).backward()
            self.optimizer.step()
            objective1s.append(objective1.item())
            objective2s.append(objective2.item())
        metrics["avg_objective1"] = float(np.mean(objective1s))
        metrics["avg_objective2"] = float(np.mean(objective2s))
        return metrics

    def cost(self, features):
        """Per-step cost c(s, a) in [0, 1], shape (steps,)."""
        with torch.no_grad():
            return self.net(torch.as_tensor(features, dtype=torch.float32)).squeeze(-1).numpy()
