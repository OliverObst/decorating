"""Explicit training-only refinement of an already reduced autonomous model."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from ._validation import FloatArray, integer, series
from .reduction import DecomposedFit, fit_components
from .spectrum import JordanComponent


@dataclass(frozen=True)
class RefinementResult:
    """Retain the original fit and optimiser diagnostics, including budget exhaustion."""

    fit: DecomposedFit
    initial_fit: DecomposedFit
    accepted: bool
    converged: bool
    status: int
    message: str
    function_evaluations: int
    residual_evaluations: int
    rejected_evaluations: int
    optimality: float


def refine_components(
    observations: ArrayLike,
    initial_fit: DecomposedFit,
    *,
    rcond: float | None = None,
    max_nfev: int = 200,
) -> RefinementResult:
    """Minimise autonomous training error with variable projection.

    Optimise each retained real eigenvalue (one parameter) or positive complex
    eigenvalue (real and imaginary parameters). Refit the linear output map
    by SVD at every evaluation. Keep dimension, pair structure, multiplicities
    and Jordan superdiagonals fixed. No unit-circle constraint or benchmark
    frequencies are supplied. This is an explicit extension of the published
    spectrum-and-refit method, not a replacement for the reference algorithm.
    """
    observed = series(observations)
    max_nfev = integer(max_nfev, "max_nfev", 1)
    components = initial_fit.components
    if not components:
        raise ValueError("refinement requires at least one component")
    if observed.shape != (initial_fit.output_map.shape[0], initial_fit.states.shape[1]):
        raise ValueError("observations must match the original fit's dimensions and times")
    parameters = np.array(
        [
            value
            for component in components
            for value in (
                [component.eigenvalue.real]
                if component.eigenvalue.imag == 0
                else [component.eigenvalue.real, component.eigenvalue.imag]
            )
        ],
        dtype=np.float64,
    )
    if observed.size <= parameters.size + observed.shape[0] * initial_fit.states.shape[0]:
        raise ValueError("refinement needs more observations than linear and spectral parameters")
    lower = []
    for component in components:
        lower.append(-np.inf)
        if component.eigenvalue.imag != 0:
            lower.append(np.nextafter(0.0, 1.0))

    def unpack(values: FloatArray) -> tuple[JordanComponent, ...]:
        updated = []
        position = 0
        for component in components:
            paired = component.eigenvalue.imag != 0
            eigenvalue = complex(values[position], values[position + 1] if paired else 0)
            updated.append(JordanComponent(eigenvalue, component.dimension))
            position += 2 if paired else 1
        return tuple(updated)

    # Recompute the initial fit with this call's data and numerical configuration.
    initial = fit_components(observed, components, rcond)
    best = initial
    evaluations = 0
    rejected = 0
    penalty = np.full(observed.size, max(1.0, float(np.max(np.abs(observed)))) * 1e6)

    def residual(values: FloatArray) -> FloatArray:
        nonlocal best, evaluations, rejected
        evaluations += 1
        try:
            candidate = fit_components(observed, unpack(values), rcond)
        except (FloatingPointError, np.linalg.LinAlgError):
            # A trust-region proposal can explode even when the starting model
            # is finite. Reject that proposal, allowing the solver to shrink
            # its step; retain the best finite fit and record every rejection.
            rejected += 1
            return penalty.copy()
        if candidate.training_rmse < best.training_rmse:
            best = candidate
        return (candidate.output_map @ candidate.states - observed).ravel()

    optimisation = least_squares(
        residual,
        parameters,
        bounds=(np.array(lower), np.full(parameters.shape, np.inf)),
        method="trf",
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=max_nfev,
    )
    # Include the final iterate, but never accept increased training error.
    residual(optimisation.x)
    return RefinementResult(
        best,
        initial,
        best.training_rmse < initial.training_rmse,
        bool(optimisation.success),
        int(optimisation.status),
        str(optimisation.message),
        int(optimisation.nfev),
        evaluations,
        rejected,
        float(optimisation.optimality),
    )
