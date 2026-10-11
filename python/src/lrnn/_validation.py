"""Shared validation for real, finite, float64 linear algebra."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def real_array(value: ArrayLike, name: str, ndim: int) -> FloatArray:
    """Convert a real array without silently discarding complex parts."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued")
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got shape {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def series(value: ArrayLike, minimum_length: int = 1) -> FloatArray:
    """Validate the explicit (observed dimensions, time samples) convention."""
    result = real_array(value, "series", 2)
    if result.shape[0] == 0 or result.shape[1] < minimum_length:
        raise ValueError(f"series needs positive dimension and at least {minimum_length} samples")
    return result


def integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def square(value: ArrayLike) -> FloatArray:
    result = real_array(value, "transition", 2)
    if result.shape[0] != result.shape[1]:
        raise ValueError("transition must be square")
    return result
