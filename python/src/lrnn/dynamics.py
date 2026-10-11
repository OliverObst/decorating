"""Explicit clamped and autonomous recurrences, with time in columns."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, integer, real_array, series, square


def autonomous_rollout(transition: ArrayLike, initial: ArrayLike, steps: int) -> FloatArray:
    """Return [x(0), ..., x(steps)] where x(t+1) = transition @ x(t).

    The initial state is included. A zero-dimensional transition is allowed for
    evaluating omission of the only Jordan component.
    """
    transition = square(transition)
    initial = real_array(initial, "initial", 1)
    steps = integer(steps, "steps")
    if initial.size != transition.shape[0]:
        raise ValueError("initial state and transition dimensions differ")
    states = np.empty((initial.size, steps + 1), dtype=np.float64)
    states[:, 0] = initial
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(steps):
            states[:, t + 1] = transition @ states[:, t]
            if not np.all(np.isfinite(states[:, t + 1])):
                raise FloatingPointError(f"autonomous recurrence became non-finite at step {t + 1}")
    return states


def driven_rollout(
    input_weights: ArrayLike,
    reservoir_weights: ArrayLike,
    observations: ArrayLike,
    initial_reservoir: ArrayLike,
) -> FloatArray:
    """Return X(t) = [S(t); R(t)], shape (d+r, T), with observations clamped.

    R(0) is given, and R(t+1) = Win @ S(t) + Wres @ R(t). The
    observation at t+1 never enters the reservoir state at t+1.
    """
    observations = series(observations)
    input_weights = real_array(input_weights, "input_weights", 2)
    reservoir_weights = square(reservoir_weights)
    initial_reservoir = real_array(initial_reservoir, "initial_reservoir", 1)
    d, length = observations.shape
    r = reservoir_weights.shape[0]
    if input_weights.shape != (r, d) or initial_reservoir.shape != (r,):
        raise ValueError("input weights, observations and reservoir dimensions differ")
    states = np.empty((d + r, length), dtype=np.float64)
    states[:d] = observations
    states[d:, 0] = initial_reservoir
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(length - 1):
            states[d:, t + 1] = (
                input_weights @ observations[:, t] + reservoir_weights @ states[d:, t]
            )
            if not np.all(np.isfinite(states[d:, t + 1])):
                raise FloatingPointError(f"driven recurrence became non-finite at sample {t + 1}")
    return states
