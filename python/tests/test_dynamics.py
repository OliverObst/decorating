import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from lrnn.dynamics import autonomous_rollout, driven_rollout
from lrnn.metrics import rmse
from lrnn.reservoir import gaussian_reservoir, normalise_reservoir, start_vector


@pytest.mark.parametrize("dimension", [1, 2, 17, 200])
def test_balanced_start(dimension: int) -> None:
    vector = start_vector(dimension)
    assert vector.dtype == np.float64
    assert_allclose(np.linalg.norm(vector), 1)
    assert_allclose(vector, np.ones(dimension) / np.sqrt(dimension))


def test_empty_start_for_omission() -> None:
    assert start_vector(0).shape == (0,)
    assert autonomous_rollout(np.empty((0, 0)), start_vector(0), 3).shape == (0, 4)


@pytest.mark.parametrize("size", [1, 2, 15])
def test_gaussian_reservoir_spectral_radius_and_seed(size: int) -> None:
    reservoir = gaussian_reservoir(size, np.random.default_rng(9))
    assert_allclose(np.max(np.abs(np.linalg.eigvals(reservoir))), 1, atol=1e-13)
    assert_array_equal(reservoir, gaussian_reservoir(size, np.random.default_rng(9)))


@pytest.mark.parametrize("bad", [np.zeros((2, 2)), [[0, 1], [0, 0]], [[np.nan]]])
def test_degenerate_reservoir_rejected(bad: object) -> None:
    with pytest.raises(ValueError):
        normalise_reservoir(bad)


def test_legacy_compute_clamping_and_previous_sample_timing() -> None:
    # Hand-evaluated compute.m recurrence: R(t+1) = [2,-1] S(t) + .5 R(t).
    observations = np.array([[1, 4, 7], [3, 6, 9]], dtype=float)
    states = driven_rollout([[2, -1]], [[0.5]], observations, [2])
    assert_array_equal(states[:2], observations)
    assert_array_equal(states[2], [2, 0, 2])
    observations[:, -1] = 1000
    assert_array_equal(driven_rollout([[2, -1]], [[0.5]], observations, [2])[2], [2, 0, 2])


def test_autonomous_recurrence_includes_initial() -> None:
    transition = np.array([[0, 1], [1, 1]], dtype=float)
    states = autonomous_rollout(transition, [0, 1], 5)
    assert_array_equal(states[0], [0, 1, 1, 2, 3, 5])
    assert_array_equal(states[:, 1:], transition @ states[:, :-1])
    assert_array_equal(autonomous_rollout(transition, [0, 1], 0), [[0], [1]])


def test_legacy_multivariate_rmse_sums_dimensions() -> None:
    assert rmse(np.zeros((2, 3)), np.array([[3, 3, 3], [4, 4, 4]])) == 5
    assert rmse(np.zeros((1, 2)), np.array([[1e200, -1e200]])) == 1e200


def test_overflow_is_reported() -> None:
    with pytest.raises(FloatingPointError, match="step 2"):
        autonomous_rollout([[1e200]], [1], 2)
    with pytest.raises(FloatingPointError, match="sample 2"):
        driven_rollout([[1]], [[1e200]], [[0, 0, 0]], [1])


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_invalid_steps(steps: int) -> None:
    with pytest.raises(ValueError):
        autonomous_rollout([[1]], [1], steps)


def test_invalid_shapes_and_complex_values() -> None:
    with pytest.raises(ValueError, match="square"):
        autonomous_rollout([[1, 2]], [1], 2)
    with pytest.raises(ValueError, match="dimensions"):
        driven_rollout([[1, 2]], [[1]], [[1, 2]], [1])
    with pytest.raises(ValueError, match="real-valued"):
        autonomous_rollout([[1j]], [1], 2)
    with pytest.raises(ValueError, match="matching"):
        rmse([[1, 2]], [[1]])
