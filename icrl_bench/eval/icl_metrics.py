"""Gaurav et al.'s two ICL metrics (github.com/ashishgaurav13/ICL, commit
096e422), on one coordinate per step. Which coordinate, and the grid, come
from the env (a MetricGrid); these functions know no env.

- CMSE: his utils.mse (tools/utils/misc.py L179-186) of the true and the
  learned cost on the grid (icl.py "cost_comparison").
- Accrual: his NormalizedAccrual in its non-fast mode
  (tools/functions/normalized_accrual.py), with normalize_func "max"
  (his configs/hc.json).
- NAD: his utils.wasserstein_distance2d (tools/utils/misc.py L160-177);
  for HalfCheetah his configs/hc.json passes the accruals as one row,
  reshape(1, -1), as nad does with a 1-D accrual."""

import numpy as np
import ot
from scipy.spatial.distance import cdist


def cmse(true_values, learned_values):
    """Mean of (true - learned)^2 over the grid points."""
    u, v = np.asarray(true_values), np.asarray(learned_values)
    assert u.shape == v.shape, f"{u.shape} vs {v.shape}"
    return float(np.mean((u - v) ** 2))


def accrual(coordinates, grid_values, left_gap, right_gap):
    """Per grid value g, how many coordinates lie in [g - left_gap,
    g + right_gap] (his compare: cv - l <= v <= cv + r), divided by the
    largest count + 1e-6 (his "max" normalisation). Coordinates in no
    window are not counted (his HalfCheetah windows leave 0.001-wide gaps
    between neighbours, and nothing outside the grid counts)."""
    c = np.asarray(coordinates, dtype=np.float64)[None, :]  # (1, steps)
    g = np.asarray(grid_values, dtype=np.float64)[:, None]  # (grid, 1)
    counts = ((g - left_gap <= c) & (c <= g + right_gap)).sum(axis=1)
    return counts / (counts.max() + 1e-6)


def nad(accrual_a, accrual_b):
    """Earth mover's distance between two accruals, each scaled to sum 1,
    moving mass between bins at cost = cityblock distance in bin indices
    (one grid step = 1). A 1-D accrual is one row. An all-zero accrual
    gives NaN, as his."""
    u = np.array(accrual_a, dtype=np.float64)
    v = np.array(accrual_b, dtype=np.float64)
    if u.ndim == 1:
        u, v = u.reshape(1, -1), v.reshape(1, -1)
    assert u.shape == v.shape and u.ndim == 2, f"{u.shape} vs {v.shape}"
    rows, cols = np.indices(u.shape)  # each bin's (row, column)
    coords = np.stack([rows.ravel(), cols.ravel()], axis=1).astype(np.float64)
    distances = cdist(coords, coords, "cityblock")
    u, v = u / u.sum(), v / v.sum()
    return float(ot.emd2(u.ravel(), v.ravel(), distances))
