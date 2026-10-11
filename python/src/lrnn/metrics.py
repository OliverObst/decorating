"""Error metric used by the paper and legacy rmse.m defaults."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import real_array


def rmse(observed: ArrayLike, predicted: ArrayLike) -> float:
    """Return sqrt(mean_t(sum_d((observed-predicted)**2))), for arrays (d, T).

    This averages over time, not over observed dimensions. It is sqrt(d)
    times an elementwise RMSE when there are d observed dimensions.
    """
    observed = real_array(observed, "observed", 2)
    predicted = real_array(predicted, "predicted", 2)
    if observed.shape != predicted.shape or 0 in observed.shape:
        raise ValueError("RMSE requires matching, non-empty (d, T) arrays")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = predicted - observed
    if not np.all(np.isfinite(difference)):
        raise FloatingPointError("RMSE difference is non-finite")
    scale = float(np.max(np.abs(difference)))
    if scale == 0:
        return 0.0
    result = scale * np.sqrt(np.mean(np.sum((difference / scale) ** 2, axis=0)))
    if not np.isfinite(result):
        raise FloatingPointError("RMSE exceeds float64 range")
    return float(result)
