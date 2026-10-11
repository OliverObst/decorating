"""The optional fit changes eigenvalues without changing spectral structure."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from lrnn import LRNN, rmse
from lrnn.benchmarks import mso_signal
from lrnn.dynamics import autonomous_rollout
from lrnn.reduction import fit_components
from lrnn.refinement import refine_components
from lrnn.spectrum import JordanComponent


def test_multivariate_damped_pair_and_negative_real_mode() -> None:
    initial = (
        JordanComponent(0.97 * np.exp(0.32j), 2),
        JordanComponent(-0.75 + 0j, 1),
    )
    truth = (JordanComponent(0.98 * np.exp(0.3j), 2), JordanComponent(-0.8 + 0j, 1))
    time = np.arange(70)
    signal = np.vstack(
        (
            0.98**time * np.sin(0.3 * time) + 0.5 * (-0.8) ** time,
            0.98**time * np.cos(0.3 * time) - 0.3 * (-0.8) ** time,
        )
    )
    fit = fit_components(signal[:, :45], initial)
    result = refine_components(signal[:, :45], fit)
    assert result.accepted and result.converged
    assert result.fit.training_rmse < 1e-12
    assert [c.dimension for c in result.fit.components] == [2, 1]
    assert_allclose([c.eigenvalue for c in result.fit.components], [c.eigenvalue for c in truth])
    future = autonomous_rollout(result.fit.transition, result.fit.states[:, -1], 25)
    assert_allclose(result.fit.output_map @ future[:, 1:], signal[:, 45:], atol=1e-11)
    assert fit.components == initial  # original fit is retained without mutation


def test_repeated_real_chain_keeps_superdiagonal_and_dimension() -> None:
    time = np.arange(50)
    signal = (time**2 + 2 * time + 3)[None, :]
    result = refine_components(signal, fit_components(signal, (JordanComponent(0.99 + 0j, 3),)))
    assert result.fit.training_rmse < 1e-9
    assert_allclose(result.fit.components[0].eigenvalue, 1, atol=1e-12)
    assert_array_equal(np.diag(result.fit.transition, 1), [1, 1])


def test_budget_exhaustion_is_visible_and_does_not_increase_training_error() -> None:
    signal = np.sin(0.3 * np.arange(50))[None, :]
    fit = fit_components(signal, (JordanComponent(np.exp(0.31j), 2),))
    result = refine_components(signal, fit, max_nfev=1)
    assert result.status == 0
    assert not result.converged
    assert result.function_evaluations == 1
    assert (
        result.residual_evaluations > result.function_evaluations
    )  # finite-difference evaluations
    assert result.fit.training_rmse <= fit.training_rmse


@pytest.mark.scientific
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_paper_configuration_refinement_uses_training_only(seed: int) -> None:
    signal = mso_signal(300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LRNN(70, seed).fit(signal[:, :150])
        baseline = model.reduce(0.5)
        refined = model.reduce(0.5, refine=True)
    assert baseline.refinement is None
    assert refined.refinement is not None and refined.refinement.converged
    assert refined.dimension == baseline.dimension == 16
    assert refined.training_rmse < 1e-10
    assert rmse(signal[:, 150:], refined.forecast(150)) < 1e-10
    assert_allclose(refined.state_map[:1], refined.fit.output_map)
    assert_array_equal(model.reduce(0.5, refine=False).reconstruct(), baseline.reconstruct())
    assert_array_equal(model.reduce(0.5, refine=True).forecast(150), refined.forecast(150))
    assert refined.refinement.initial_fit.components == baseline.fit.components


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_invalid_refinement_budget(budget: object) -> None:
    observed = np.ones((1, 10))
    with pytest.raises(ValueError, match="max_nfev"):
        refine_components(
            observed, fit_components(observed, (JordanComponent(1 + 0j, 1),)), max_nfev=budget
        )


def test_refinement_shape_and_identifiability_validation() -> None:
    observed = np.ones((1, 10))
    fit = fit_components(observed, (JordanComponent(1 + 0j, 1),))
    with pytest.raises(ValueError, match="match"):
        refine_components(np.ones((2, 10)), fit)
    short = np.ones((1, 2))
    with pytest.raises(ValueError, match="more observations"):
        refine_components(short, fit_components(short, (JordanComponent(1 + 0j, 1),)))
    with pytest.raises(ValueError, match="at least one"):
        refine_components(observed, fit_components(observed, ()))


def test_non_finite_optimiser_proposal_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import lrnn.refinement as refinement

    observed = np.sin(0.3 * np.arange(40))[None, :]
    initial = fit_components(observed, (JordanComponent(np.exp(0.31j), 2),))
    original = refinement.fit_components
    count = 0

    def fit_with_overflow(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise FloatingPointError("autonomous recurrence became non-finite")
        return original(*args, **kwargs)

    monkeypatch.setattr(refinement, "fit_components", fit_with_overflow)
    result = refinement.refine_components(observed, initial, max_nfev=2)
    assert result.rejected_evaluations == 1
    assert np.all(np.isfinite(result.fit.states))
    assert result.fit.training_rmse <= initial.training_rmse
