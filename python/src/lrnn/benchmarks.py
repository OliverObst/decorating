"""Fixed benchmark definitions, kept separate from the estimator."""

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, integer, real_array

MSO_FREQUENCIES = (0.200, 0.311, 0.420, 0.510, 0.630, 0.740, 0.850, 0.970)


def mso_signal(length: int, frequencies: ArrayLike = MSO_FREQUENCIES) -> FloatArray:
    """Return sum_k sin(alpha_k*t), shape (1,length), for t=1,...,length.

    Frequencies are angular frequencies in radians per sample, not cycles
    per sample. These conventions match legacy mso8/mso.m.
    """
    length = integer(length, "length", 1)
    frequencies = real_array(frequencies, "frequencies", 1)
    if frequencies.size == 0:
        raise ValueError("at least one frequency is required")
    return np.sin(frequencies[:, None] * np.arange(1, length + 1)).sum(axis=0, keepdims=True)
