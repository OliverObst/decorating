import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from lrnn import LRNN, rmse
from lrnn.benchmarks import MSO_FREQUENCIES, mso_signal
from lrnn.regression import NumericalConditionWarning


def fitted(observed: np.ndarray, size: int, seed: int) -> LRNN:
    # Warning behaviour is tested separately. Scientific examples inspect rank explicitly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NumericalConditionWarning)
        return LRNN(size, seed).fit(observed)


def test_fixed_seed_and_forecast_endpoint() -> None:
    observed = np.sin(0.3 * np.arange(50))[None, :]
    first, second = fitted(observed, 12, 7), fitted(observed, 12, 7)
    assert_array_equal(first.transition_, second.transition_)
    assert_array_equal(first.driven_states_, second.driven_states_)
    assert_allclose(first.forecast(1)[:, 0], first.output_weights_ @ first.driven_states_[:, -1])
    assert first.forecast(0).shape == (1, 0)
    assert_array_equal(first.forecast(10), first.forecast(10))


def test_rng_ownership_and_repeat_fit() -> None:
    observed = np.sin(0.3 * np.arange(30))[None, :]
    seeded = fitted(observed, 5, 1)
    previous = seeded.transition_.copy()
    seeded.fit(observed)
    assert_array_equal(seeded.transition_, previous)
    generator = np.random.default_rng(1)
    model = LRNN(5, generator).fit(observed)
    previous = model.transition_.copy()
    model.fit(observed)
    assert not np.array_equal(model.transition_, previous)
    np.random.seed(123)
    before = np.random.get_state()
    model.fit(observed)
    after = np.random.get_state()
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_input_weight_scaling_is_explicit() -> None:
    observed = np.vstack((np.sin(0.3 * np.arange(50)), np.cos(0.3 * np.arange(50))))
    balanced = LRNN(5, 7).fit(observed)
    standard = LRNN(5, 7, input_scaling="standard").fit(observed)
    assert_allclose(standard.input_weights_, balanced.input_weights_ * np.sqrt(2))
    assert_array_equal(standard.reservoir_weights_, balanced.reservoir_weights_)


@pytest.mark.parametrize("refine", [False, True])
def test_fibonacci_learn_reduce_and_extrapolate(refine: bool) -> None:
    values = [0.0, 1.0]
    for _ in range(38):
        values.append(sum(values[-2:]))
    signal = np.array(values)[None, :]
    model = fitted(signal[:, :31], 30, 1)
    reduced = model.reduce(0.001, refine=refine)
    assert reduced.dimension == 2
    assert reduced.training_rmse < 0.001
    assert_allclose(reduced.forecast(9), signal[:, 31:], rtol=1e-10, atol=0.001)
    recovered = sorted(c.eigenvalue.real for c in reduced.fit.components)
    # The decaying mode is constrained by the early samples; later values reach 8e5.
    assert_allclose(recovered[0], (1 - np.sqrt(5)) / 2, atol=1e-4)
    assert_allclose(recovered[1], (1 + np.sqrt(5)) / 2, atol=1e-10)


@pytest.mark.parametrize("refine", [False, True])
def test_sinusoid_learns_real_conjugate_block(refine: bool) -> None:
    signal = np.sin(np.pi * np.arange(201) / 100)[None, :]
    reduced = fitted(signal[:, :101], 40, 2).reduce(0.01, cluster_tol=0.03, refine=refine)
    assert reduced.dimension == 2
    assert len(reduced.fit.components) == 1
    assert_allclose(reduced.forecast(100), signal[:, 101:], atol=1e-6)
    assert_allclose(reduced.fit.output_map, reduced.state_map[:1])
    assert_array_equal(reduced.forecast(100), reduced.forecast(100))
    assert reduced.forecast(0).shape == (1, 0)


@pytest.mark.parametrize("refine", [False, True])
def test_parabola_learns_a_three_dimensional_jordan_chain(refine: bool) -> None:
    time = np.arange(201) / 100
    signal = (4 * time * (1 - time))[None, :]
    reduced = fitted(signal[:, :101], 40, 2).reduce(0.01, cluster_tol=0.03, refine=refine)
    assert reduced.dimension == 3
    assert len(reduced.fit.components) == 1
    assert_allclose(reduced.fit.components[0].eigenvalue, 1, atol=1e-7)
    assert_allclose(reduced.forecast(100), signal[:, 101:], atol=2e-5)
    assert_array_equal(np.diag(reduced.fit.transition, 1), [1, 1])


@pytest.mark.parametrize("refine", [False, True])
def test_multivariate_sinusoidal_fit_and_reduction(refine: bool) -> None:
    time = np.arange(140)
    signal = np.vstack((np.sin(0.3 * time), np.cos(0.7 * time)))
    model = fitted(signal[:, :100], 40, 0)
    reduced = model.reduce(1e-4, refine=refine)
    assert reduced.dimension == 4
    assert reduced.forecast(40).shape == (2, 40)
    assert rmse(signal[:, 100:], reduced.forecast(40)) < 1e-3


@pytest.mark.scientific
@pytest.mark.parametrize("refine", [False, True])
def test_mso_compact_autonomous_extrapolation(refine: bool) -> None:
    # Legacy compute_prediction.m: 200 training, 50 validation, 50 testing samples.
    # A predeclared seed is a regression case; multi-trial evidence is saved by the experiment.
    signal = mso_signal(300)
    model = fitted(signal[:, :200], 198, 2)
    reduced = model.reduce(0.5, refine=refine)
    assert reduced.dimension == 16
    assert model.one_step_rmse_ < 1e-7
    assert reduced.training_rmse < 1e-3
    forecast = reduced.forecast(100)
    assert rmse(signal[:, 200:250], forecast[:, :50]) < 1e-3
    assert rmse(signal[:, 250:], forecast[:, 50:]) < 1e-3
    frequencies = sorted(np.angle(c.eigenvalue) for c in reduced.fit.components)
    assert_allclose(frequencies, MSO_FREQUENCIES, atol=1e-5)


def test_mso_signal_uses_legacy_time_origin() -> None:
    assert_allclose(
        mso_signal(2, [0.2, 0.3]), [[np.sin(0.2) + np.sin(0.3), np.sin(0.4) + np.sin(0.6)]]
    )


@pytest.mark.parametrize(
    "observed", [[], [[1]], [[1, np.nan]], [[1, np.inf]], [1, 2, 3], np.empty((0, 3)), [[1j, 2j]]]
)
def test_invalid_series(observed: object) -> None:
    with pytest.raises(ValueError):
        LRNN(2, 0).fit(observed)


def test_failed_refit_invalidates_old_model() -> None:
    model = fitted(np.array([[1, 2, 3, 4]]), 3, 0)
    with pytest.raises(ValueError):
        model.fit([[np.nan, 2]])
    with pytest.raises(RuntimeError, match="call fit"):
        model.forecast(1)


def test_unfitted_model_and_invalid_configuration() -> None:
    model = LRNN(2, 0)
    for method in [lambda: model.forecast(1), lambda: model.reduce(0.1), model.decompose]:
        with pytest.raises(RuntimeError, match="call fit"):
            method()
    for arguments in [
        {"reservoir_size": 0},
        {"random_state": -1},
        {"rcond": -1},
        {"input_scaling": "hidden"},
    ]:
        with pytest.raises(ValueError):
            LRNN(**arguments)
