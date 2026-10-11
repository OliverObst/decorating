import numpy as np
import pytest
from numpy.testing import assert_allclose

from lrnn.reduction import fit_components, omission_scores, reduce_components
from lrnn.regression import NumericalConditionWarning, fit_map
from lrnn.spectrum import JordanComponent


def test_scaled_least_squares_recovers_modes_with_different_magnitudes() -> None:
    states = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=float)
    states[0] *= 1e100
    target = np.array([[1e-100, 2], [-2e-100, 1]]) @ states
    mapping, diagnostics = fit_map(target, states)
    assert diagnostics.rank == 2
    assert diagnostics.condition_number < 10
    assert_allclose(mapping @ states, target, atol=1e-14)


def test_rank_deficiency_diagnostics_and_warning() -> None:
    mapping, diagnostics = fit_map([[1, 2, 3]], [[1, 2, 3], [2, 4, 6]])
    assert diagnostics.rank == 1
    assert diagnostics.condition_number == np.inf
    assert_allclose(mapping @ np.array([[1, 2, 3], [2, 4, 6]]), [[1, 2, 3]])
    with pytest.warns(NumericalConditionWarning, match="rank 1/2"):
        diagnostics.warn_if_unreliable()


def test_omission_refits_and_keeps_conjugate_pair_intact() -> None:
    omega = 0.4
    components = (JordanComponent(np.exp(1j * omega), 2), JordanComponent(0.6 + 0j, 1))
    time = np.arange(40)
    observed = (np.sin(omega * time) + 0.001 * 0.6**time)[None, :]
    scores = omission_scores(observed, components)
    assert scores[0] > 0.5
    assert scores[1] < 0.001
    reduced = reduce_components(observed, components, 0.002)
    assert reduced.fit.transition.shape == (2, 2)
    assert reduced.fit.components == (components[0],)
    assert reduced.fit.training_rmse <= 0.002


def test_hand_evaluated_legacy_omission_errors() -> None:
    # S=[1,1,1], omit 0-mode -> constant fitted exactly. Omit 1-mode -> [1,0,0].
    components = (JordanComponent(1 + 0j, 1), JordanComponent(0 + 0j, 1))
    assert_allclose(omission_scores([[1, 1, 1]], components), [np.sqrt(2 / 3), 0], atol=1e-15)


def test_default_reduction_refits_instead_of_truncating_output_coefficients() -> None:
    # A decaying component has nonzero mean over training: removing it makes the
    # best retained constant differ from its coefficient in the full model.
    components = (JordanComponent(1 + 0j, 1), JordanComponent(0.8 + 0j, 1))
    observed = (2 + 3 * 0.8 ** np.arange(40))[None, :]
    full = fit_components(observed, components)
    truncated = full.output_map[:, :1] @ full.states[:1]
    truncation_rmse = np.sqrt(np.mean((observed - truncated) ** 2))
    refitted_rmse = np.std(observed)
    assert refitted_rmse < truncation_rmse

    reduced = reduce_components(observed, components, (refitted_rmse + truncation_rmse) / 2)
    assert reduced.fit.components == (components[0],)
    assert_allclose(reduced.fit.output_map @ reduced.fit.states, np.mean(observed), atol=1e-14)
    assert_allclose(reduced.fit.training_rmse, refitted_rmse, atol=1e-14)
    assert not np.allclose(reduced.fit.output_map, full.output_map[:, :1])


def test_single_component_zero_dimensional_omission_and_final_prefix() -> None:
    component = (JordanComponent(1 + 0j, 1),)
    assert_allclose(omission_scores([[2, 2, 2]], component), [2])
    result = reduce_components([[2, 2, 2]], component, 1e-12)
    assert result.fit.transition.shape == (1, 1)
    assert result.fit.training_rmse < 1e-12


def test_binary_search_returns_actual_smallest_feasible_prefix() -> None:
    components = tuple(JordanComponent(complex(v), 1) for v in [1, 0.8, -0.7, 0.4])
    time = np.arange(30)
    observed = (2 + 3 * 0.8**time + 0.1 * (-0.7) ** time)[None, :]
    result = reduce_components(observed, components, 1e-10)
    assert len(result.fit.components) == 3
    ranked = tuple(components[k] for k in result.order)
    for k in range(1, 3):
        assert fit_components(observed, ranked[:k]).training_rmse > 1e-10
    assert result.fit.training_rmse <= 1e-10


def test_numerical_nonmonotonicity_is_audited(monkeypatch: pytest.MonkeyPatch) -> None:
    import lrnn.reduction as reduction

    real_fit = reduction.fit_components
    components = tuple(JordanComponent(complex(k), 1) for k in range(4))
    errors = {0: 2.0, 1: 0.01, 2: 1.0, 3: 0.5, 4: 0.001}

    def fake_fit(observed: object, blocks: tuple[JordanComponent, ...], rcond: object = None):
        from dataclasses import replace

        fit = real_fit(observed, blocks)
        return replace(fit, training_rmse=errors[len(blocks)])

    monkeypatch.setattr(reduction, "fit_components", fake_fit)
    result = reduce_components([[1, 2, 3, 4, 5]], components, 0.1)
    assert len(result.fit.components) == 1


def test_impossible_threshold_is_not_silently_accepted() -> None:
    with pytest.raises(ValueError, match="no ranked prefix"):
        reduce_components([[1, 2, 3]], (JordanComponent(1 + 0j, 1),), 0.1)


@pytest.mark.parametrize("threshold", [-1, np.inf, np.nan])
def test_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="max_rmse"):
        reduce_components([[1, 1]], (JordanComponent(1 + 0j, 1),), threshold)
