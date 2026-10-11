"""Omission ranking, refitting and smallest-prefix reduction."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, series
from .dynamics import autonomous_rollout
from .metrics import rmse
from .regression import LeastSquaresDiagnostics, fit_map
from .reservoir import start_vector
from .spectrum import JordanComponent, jordan_matrix


@dataclass(frozen=True)
class DecomposedFit:
    """Inspectable dynamics and map fitted to the observed trajectory."""

    components: tuple[JordanComponent, ...]
    transition: FloatArray
    states: FloatArray
    output_map: FloatArray
    diagnostics: LeastSquaresDiagnostics
    training_rmse: float


@dataclass(frozen=True)
class ReductionResult:
    """Omission scores use original component order; order indexes those scores."""

    fit: DecomposedFit
    omission_scores: FloatArray
    order: tuple[int, ...]
    prefix_errors: dict[int, float]


def fit_components(
    observations: ArrayLike, components: tuple[JordanComponent, ...], rcond: float | None = None
) -> DecomposedFit:
    """Refit S = A @ Y where Y(t) = J**t @ balanced_start, for all training times."""
    observations = series(observations)
    transition = jordan_matrix(components)
    states = autonomous_rollout(
        transition, start_vector(transition.shape[0]), observations.shape[1] - 1
    )
    output_map, diagnostics = fit_map(observations, states, rcond)
    error = rmse(observations, output_map @ states)
    return DecomposedFit(components, transition, states, output_map, diagnostics, error)


def omission_scores(
    observations: ArrayLike, components: tuple[JordanComponent, ...], rcond: float | None = None
) -> FloatArray:
    """Return RMSE after omitting each *whole* block and refitting all other blocks."""
    observations = series(observations)
    return np.array(
        [
            fit_components(observations, components[:k] + components[k + 1 :], rcond).training_rmse
            for k in range(len(components))
        ],
        dtype=np.float64,
    )


def reduce_components(
    observations: ArrayLike,
    components: tuple[JordanComponent, ...],
    max_rmse: float,
    rcond: float | None = None,
) -> ReductionResult:
    """Rank blocks, then binary-search the smallest non-empty prefix with RMSE <= max_rmse.

    Refitting makes prefix feature spaces nested in exact arithmetic. Numerical
    rank cut-offs can break monotonicity: audit every smaller prefix to ensure
    that the returned prefix is actually the smallest feasible one. Raise if
    no prefix satisfies the threshold; never silently return an infeasible fit.
    """
    observations = series(observations)
    if not components:
        raise ValueError("reduction requires at least one Jordan component")
    if not np.isfinite(max_rmse) or max_rmse < 0:
        raise ValueError("max_rmse must be finite and non-negative")
    scores = omission_scores(observations, components, rcond)
    order = tuple(int(k) for k in np.argsort(-scores, kind="stable"))
    ranked = tuple(components[k] for k in order)
    cache: dict[int, DecomposedFit] = {}

    def evaluate(count: int) -> DecomposedFit:
        if count not in cache:
            cache[count] = fit_components(observations, ranked[:count], rcond)
        return cache[count]

    left, right = 1, len(ranked)
    while left < right:
        middle = (left + right) // 2
        if evaluate(middle).training_rmse <= max_rmse:
            right = middle
        else:
            left = middle + 1
    # Check lower prefixes, including a numerically feasible one skipped by binary search.
    selected = next((k for k in range(1, left + 1) if evaluate(k).training_rmse <= max_rmse), None)
    if selected is None:
        best = min(fit.training_rmse for fit in cache.values())
        raise ValueError(f"no ranked prefix meets max_rmse={max_rmse:g}; best RMSE={best:.6g}")
    return ReductionResult(
        evaluate(selected),
        scores,
        order,
        {k: fit.training_rmse for k, fit in sorted(cache.items())},
    )
