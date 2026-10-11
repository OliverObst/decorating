"""Seeded Gaussian reservoirs and balanced initial states."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, integer, square


def start_vector(dimension: int) -> FloatArray:
    """Return the balanced all-ones unit vector, or an empty vector for dimension zero."""
    dimension = integer(dimension, "dimension")
    return np.ones(dimension, dtype=np.float64) / max(np.sqrt(dimension), 1.0)


def normalise_reservoir(matrix: ArrayLike) -> FloatArray:
    """Return a square matrix divided by its spectral radius; reject zero radius."""
    matrix = square(matrix)
    if matrix.shape[0] == 0:
        raise ValueError("reservoir must have positive dimension")
    radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if not np.isfinite(radius) or radius <= np.finfo(np.float64).tiny:
        raise ValueError("reservoir has a degenerate spectral radius")
    result = matrix / radius
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("reservoir normalisation produced non-finite weights")
    return result


def gaussian_reservoir(size: int, rng: np.random.Generator) -> FloatArray:
    """Draw a (size, size) standard Gaussian reservoir with unit spectral radius."""
    size = integer(size, "size", 1)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return normalise_reservoir(rng.standard_normal((size, size)))
