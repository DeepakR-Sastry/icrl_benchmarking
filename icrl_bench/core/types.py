"""The two data types every part of the library passes around."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Step:
    """One moment of a task attempt: what the agent saw, what it did, and the
    reward it got. `state` and `action` are raw env values of any type;
    featurizers turn them into numbers. ICRL methods never read `reward`."""

    state: Any
    action: Any
    reward: float | None  # None when the source doesn't store it (Malik's expert files)


@dataclass(frozen=True)
class Trajectory:
    """One recorded task attempt. It carries no label: the pile (expert or
    nominal) it sits in is its label."""

    steps: list[Step]


@dataclass(frozen=True)
class MetricGrid:
    """Where an env's ICL metrics (Gaurav's CMSE and NAD,
    eval/icl_metrics.py) are measured: a list of points along one
    coordinate of the state. The env file that knows that coordinate
    builds it; the metrics and the loop only read it."""

    values: Any  # np.ndarray (n,): the grid points
    trajectory: Trajectory  # one Step per grid point: that coordinate set, all else 0
    coordinate: Any  # coordinate(obs) -> float: the number a step is binned on
    left_gap: float  # accrual window: [value - left_gap, value + right_gap]
    right_gap: float
