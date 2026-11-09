import numpy as np
import torch

from icrl_bench.methods.flow import Flow, FlowSettings

# Three hand-made rows: column 0 ranges over [0, 4], column 1 over [10, 30].
DATA = np.array([[0.0, 10.0], [2.0, 30.0], [4.0, 20.0]])


def test_one_log_probability_per_row():
    flow = Flow(DATA)
    assert flow.log_probs(DATA).shape == (3, 1)


def test_density_integrates_to_one():
    # A density sums to 1 over the plane: checks the log-determinants.
    torch.manual_seed(0)
    flow = Flow(np.random.RandomState(0).randn(200, 2))
    grid = torch.linspace(-10, 10, 300)
    xs, ys = torch.meshgrid(grid, grid, indexing="ij")
    with torch.no_grad():
        p = flow.log_probs(torch.stack([xs.flatten(), ys.flatten()], dim=1)).exp()
    assert abs((p.sum() * (grid[1] - grid[0]) ** 2).item() - 1) < 0.01


def test_scaling_maps_the_data_range_to_minus_one_one():
    # [2, 20] is the middle of the data's range in both columns, so a
    # scaling flow scores it as the same flow without scaling scores [0, 0].
    torch.manual_seed(0)
    scaled = Flow(DATA, FlowSettings(normalize_inputs=True))
    torch.manual_seed(0)
    plain = Flow(DATA, FlowSettings(normalize_inputs=False))
    with torch.no_grad():
        assert torch.equal(scaled.log_probs(np.array([[2.0, 20.0]])),
                           plain.log_probs(np.array([[0.0, 0.0]])))
        # The corners of the range go to -1 and +1.
        assert torch.equal(scaled.log_probs(np.array([[0.0, 30.0]])),
                           plain.log_probs(np.array([[-1.0, 1.0]])))


def test_training_lowers_the_negative_log_likelihood():
    torch.manual_seed(0)
    data = np.random.RandomState(0).randn(256, 2) * [0.5, 2.0]
    flow = Flow(data, FlowSettings(batch_size=64))
    first = flow.train_epoch()["avg_nll"]
    for _ in range(9):
        last = flow.train_epoch()["avg_nll"]
    assert last < first
