"""RealNVP normalizing flow: a learned density over feature vectors, used by
ICL (Gaurav et al., ICLR 2023) to tell expert-like steps from the rest.

Follows Gaurav's official code (github.com/ashishgaurav13/ICL, commit
096e422): tools/functions/__init__.py create_flow(..., "realnvp"),
tools/functions/flow.py Flow, and the layers FlowAN, Flow1x1Conv,
FlowRealNVP, FlowSequential in tools/utils/torch_helper.py. Only the
direction ICL uses (data -> noise, for log-probabilities) is ported.
Deviations: docs/methods.md §5."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class FlowSettings:
    """Defaults: his create_flow (h=64, n_blocks=5). Learning rate, batch
    size and input scaling come from each env's config (his learning_rate,
    minibatch_size, normalize_flow_inputs)."""
    hidden: int = 64
    n_blocks: int = 5
    learning_rate: float = 5e-4
    batch_size: int = 64
    normalize_inputs: bool = False


class ActNorm(nn.Module):
    """His FlowAN: z = x * exp(log_sigma) + mu, per column. Starts as the
    identity (mu = 0, log_sigma = 0); no data-dependent start."""

    def __init__(self, dim):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_sigma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * torch.exp(self.log_sigma) + self.mu, torch.sum(self.log_sigma)


class InvertibleLinear(nn.Module):
    """His Flow1x1Conv: z = x @ P @ L @ (U + diag(S)), a learned dim x dim
    matrix stored as its LU pieces so the log-determinant is sum log|S|.
    P (a permutation) is fixed; L's diagonal is fixed at 1."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Start from a random rotation (QR of a Gaussian matrix), split
        # W = P L U. torch.linalg.lu = his torch.lu_unpack(*W.lu()).
        w, _ = torch.linalg.qr(torch.randn(dim, dim))
        p, lower, upper = torch.linalg.lu(w)
        self.register_buffer("P", p)
        self.L = nn.Parameter(lower.clone())
        self.S = nn.Parameter(torch.diag(upper).clone())
        self.U = nn.Parameter(torch.triu(upper, diagonal=1).clone())

    def forward(self, x):
        lower = torch.tril(self.L, diagonal=-1) + torch.eye(self.dim)
        upper = torch.triu(self.U, diagonal=1)
        z = x @ self.P @ lower @ (upper + torch.diag(self.S))
        return z, torch.sum(torch.log(torch.abs(self.S)))


class Coupling(nn.Module):
    """His FlowRealNVP: the columns where mask = 1 pass through; two nets
    read them and give a log-scale and a shift for the other columns.
    Their Linear layers keep PyTorch's default init: his init_weights
    (Xavier, bias 0.01) never reaches them, because FlowRealNVP is a
    torch.jit.ScriptModule, which compiles the nets when they are assigned,
    so .apply sees compiled modules, not nn.Linear (checked in torch 1.11,
    docs/methods.md §5)."""

    def __init__(self, dim, hidden, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim))
        self.translate_net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim))

    def forward(self, x):
        masked_x = x * self.mask
        log_s = self.scale_net(masked_x) * (1 - self.mask)
        t = self.translate_net(masked_x) * (1 - self.mask)
        return x * torch.exp(log_s) + t, log_s.sum(-1, keepdim=True)


class Flow(nn.Module):
    """A density over rows of `data` (n_rows, dim): n_blocks x (ActNorm ->
    InvertibleLinear -> Coupling), standard normal at the end. Fit with
    train_epoch; read with log_probs."""

    def __init__(self, data, settings=None):
        super().__init__()
        self.settings = settings or FlowSettings()
        s = self.settings
        data = torch.as_tensor(np.asarray(data), dtype=torch.float32)
        assert data.ndim == 2, data.shape
        dim = data.shape[1]
        # create_flow: the mask alternates between blocks.
        mask = torch.arange(dim) % 2
        layers = []
        for _ in range(s.n_blocks):
            layers += [ActNorm(dim), InvertibleLinear(dim), Coupling(dim, s.hidden, mask)]
            mask = 1 - mask
        self.layers = nn.ModuleList(layers)
        # Min-max range of the data the flow was built on (his Flow.__init__).
        self.register_buffer("data_min", torch.min(data, dim=0).values)
        self.register_buffer("data_max", torch.max(data, dim=0).values)
        # Built once; each pass draws a new shuffle from torch's random state.
        self.loader = torch.utils.data.DataLoader(data, batch_size=s.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=s.learning_rate)

    def log_probs(self, x):
        """log p(x) per row, shape (n_rows, 1). x: (n_rows, dim)."""
        x = torch.as_tensor(x, dtype=torch.float32)
        if self.settings.normalize_inputs:
            # Each column to [-1, 1] by the data's range (his Flow.log_probs).
            x = (x - self.data_min) / (self.data_max - self.data_min) * 2 - 1.0
        log_det = torch.zeros(x.shape[0], 1)
        for layer in self.layers:
            x, layer_log_det = layer(x)
            log_det = log_det + layer_log_det
        log_normal = (-0.5 * x.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        return log_normal + log_det

    def train_epoch(self):
        """One pass over the data in shuffled minibatches, one Adam step on
        the mean negative log-likelihood each (his Flow.train)."""
        nlls = []
        for batch in self.loader:
            self.optimizer.zero_grad()
            nll = -self.log_probs(batch).mean()
            nll.backward()
            self.optimizer.step()
            nlls.append(nll.item())
        return {"avg_nll": float(np.mean(nlls)), "std_nll": float(np.std(nlls))}
