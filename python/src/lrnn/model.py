"""Estimator API for the deterministic published LRNN algorithm."""

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, integer, series
from .dynamics import autonomous_rollout, driven_rollout
from .metrics import rmse
from .reduction import DecomposedFit, ReductionResult, fit_components, reduce_components
from .refinement import RefinementResult, refine_components
from .regression import LeastSquaresDiagnostics, fit_map
from .reservoir import gaussian_reservoir, start_vector
from .spectrum import spectral_components


@dataclass(frozen=True)
class ReducedLRNN:
    """An autonomous real Jordan model, anchored at the last training sample.

    Forecast calls do not mutate the terminal state. Inspect `fit` for J, Y,
    A, regression diagnostics and the retained components. `state_map` maps
    Jordan states to the original complete driven state X, as in predict.m.
    """

    fit: DecomposedFit
    reduction: ReductionResult
    state_map: FloatArray
    state_map_diagnostics: LeastSquaresDiagnostics
    refinement: RefinementResult | None = None

    @property
    def dimension(self) -> int:
        """Total real dynamical dimension, including both members of complex pairs."""
        return self.fit.transition.shape[0]

    @property
    def training_rmse(self) -> float:
        return self.fit.training_rmse

    def reconstruct(self) -> FloatArray:
        """Return the autonomous fit to all training samples, shape (d,T)."""
        return self.fit.output_map @ self.fit.states

    def forecast(self, steps: int) -> FloatArray:
        """Return the next `steps` samples (d,steps), excluding the training endpoint."""
        states = autonomous_rollout(self.fit.transition, self.fit.states[:, -1], steps)
        result = self.fit.output_map @ states[:, 1:]
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("forecast output became non-finite")
        return result


class LRNN:
    """Fit a Gaussian reservoir LRNN, then reduce by spectral omission scores.

    Input is always (d,T), even for univariate series. Integer seeds recreate
    the same reservoir on each fit; supplied Generators advance explicitly.
    `input_scaling='balanced'` uses legacy Win / sqrt(d); `'standard'` uses
    the paper's unscaled standard Gaussian input weights. No retries or
    validation selection are hidden in the estimator.
    """

    def __init__(
        self,
        reservoir_size: int = 200,
        random_state: int | np.random.Generator | None = None,
        *,
        input_scaling: str = "balanced",
        rcond: float | None = None,
    ) -> None:
        self.reservoir_size = integer(reservoir_size, "reservoir_size", 1)
        if random_state is not None and not isinstance(random_state, np.random.Generator):
            integer(random_state, "random_state")
        if input_scaling not in ("balanced", "standard"):
            raise ValueError("input_scaling must be 'balanced' or 'standard'")
        if rcond is not None and (not np.isfinite(rcond) or rcond < 0):
            raise ValueError("rcond must be finite and non-negative")
        self.random_state = random_state
        self.input_scaling = input_scaling
        self.rcond = rcond
        self._fitted = False

    def fit(self, observations: ArrayLike) -> Self:
        """Learn next-step output weights and complete transition; expose all intermediates."""
        self._fitted = False
        observed = series(observations, minimum_length=2).copy()
        d = observed.shape[0]
        rng = np.random.default_rng(self.random_state)
        input_weights = rng.standard_normal((self.reservoir_size, d))
        if self.input_scaling == "balanced":
            input_weights /= np.sqrt(d)
        reservoir_weights = gaussian_reservoir(self.reservoir_size, rng)
        driven_states = driven_rollout(
            input_weights, reservoir_weights, observed, start_vector(self.reservoir_size)
        )
        output_weights, diagnostics = fit_map(observed[:, 1:], driven_states[:, :-1], self.rcond)
        transition = np.vstack((output_weights, np.hstack((input_weights, reservoir_weights))))
        eigenvalues = np.linalg.eigvals(transition)
        self.observations_ = observed
        self.input_weights_ = input_weights
        self.reservoir_weights_ = reservoir_weights
        self.driven_states_ = driven_states
        self.output_weights_ = output_weights
        self.output_diagnostics_ = diagnostics
        self.transition_ = transition
        self.eigenvalues_ = eigenvalues
        self.one_step_rmse_ = rmse(observed[:, 1:], output_weights @ driven_states[:, :-1])
        self._fitted = True
        diagnostics.warn_if_unreliable()
        return self

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("call fit before decomposition, reduction or forecasting")

    def decompose(self, cluster_tol: float = 0.0) -> DecomposedFit:
        """Fit the complete spectral representation without computing Jordan eigenvectors."""
        self._require_fit()
        result = fit_components(
            self.observations_, spectral_components(self.eigenvalues_, cluster_tol), self.rcond
        )
        result.diagnostics.warn_if_unreliable()
        return result

    def reduce(
        self,
        max_rmse: float,
        cluster_tol: float = 0.0,
        *,
        refine: bool = False,
        max_nfev: int = 200,
    ) -> ReducedLRNN:
        """Keep the smallest feasible prefix; optionally refine its autonomous training fit.

        The default preserves the published spectrum-and-refit procedure. Set
        `refine=True` to optimise retained eigenvalues using training data only. Inspect
        `refinement` for the initial fit and convergence diagnostics; `reduction`
        always records the original omission ranking and prefix errors.
        """
        self._require_fit()
        result = reduce_components(
            self.observations_,
            spectral_components(self.eigenvalues_, cluster_tol),
            max_rmse,
            self.rcond,
        )
        refinement = (
            refine_components(self.observations_, result.fit, rcond=self.rcond, max_nfev=max_nfev)
            if refine
            else None
        )
        fit = result.fit if refinement is None else refinement.fit
        state_map, diagnostics = fit_map(self.driven_states_, fit.states, self.rcond)
        fit.diagnostics.warn_if_unreliable()
        return ReducedLRNN(fit, result, state_map, diagnostics, refinement)

    def forecast(self, steps: int) -> FloatArray:
        """Forecast using the unreduced complete transition from the last clamped state."""
        self._require_fit()
        states = autonomous_rollout(self.transition_, self.driven_states_[:, -1], steps)
        return states[: self.observations_.shape[0], 1:]
