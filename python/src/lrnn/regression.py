"""Stable right matrix division with inspectable numerical diagnostics."""

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, real_array


class NumericalConditionWarning(UserWarning):
    """A fitted design is rank deficient or poorly conditioned in float64."""


@dataclass(frozen=True)
class LeastSquaresDiagnostics:
    """Diagnostics refer to the row-equilibrated design used by the solve."""

    rank: int
    features: int
    samples: int
    singular_values: FloatArray
    feature_scales: FloatArray
    condition_number: float
    residual_norm: float

    def warn_if_unreliable(self) -> None:
        """Report loss of numerical identifiability without hiding fitted results."""
        if self.rank < self.features or self.condition_number > 1 / np.sqrt(np.finfo(float).eps):
            warnings.warn(
                f"least-squares design: rank {self.rank}/{self.features}, "
                f"scaled condition number {self.condition_number:.3g}; inspect diagnostics",
                NumericalConditionWarning,
                stacklevel=2,
            )


def fit_map(
    target: ArrayLike, states: ArrayLike, rcond: float | None = None
) -> tuple[FloatArray, LeastSquaresDiagnostics]:
    """Solve target = A @ states, with target (d,T) and states (N,T).

    Divide each feature by its largest absolute value before an SVD solve,
    then undo this scaling on A. This preserves the feature span and avoids
    losing small dynamical modes solely because another mode grows rapidly.
    For rank-deficient designs this defines a scaled minimum-norm solution.
    No normal equations or explicit inverse are used.
    """
    target = real_array(target, "target", 2)
    states = real_array(states, "states", 2)
    if target.shape[1] == 0 or target.shape[1] != states.shape[1]:
        raise ValueError("target and states need matching, positive sample counts")
    if rcond is not None and (not np.isfinite(rcond) or rcond < 0):
        raise ValueError("rcond must be finite and non-negative")
    scales = np.max(np.abs(states), axis=1)
    scales[scales == 0] = 1
    coefficients, _, rank, singular = np.linalg.lstsq(
        (states / scales[:, None]).T, target.T, rcond=rcond
    )
    mapping = coefficients.T / scales[None, :]
    if not np.all(np.isfinite(mapping)):
        raise FloatingPointError("least-squares fit produced non-finite coefficients")
    condition = (
        float(singular[0] / singular[-1])
        if singular.size and singular[-1] > 0 and rank == states.shape[0]
        else float("inf")
    )
    residual = mapping @ states - target
    if not np.all(np.isfinite(residual)):
        raise FloatingPointError("least-squares reconstruction produced non-finite values")
    diagnostics = LeastSquaresDiagnostics(
        int(rank),
        states.shape[0],
        states.shape[1],
        singular,
        scales,
        condition,
        float(np.linalg.norm(residual)),
    )
    return mapping, diagnostics
