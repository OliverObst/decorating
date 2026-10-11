import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from lrnn.spectrum import (
    JordanComponent,
    cluster_eigenvalues,
    jordan_matrix,
    real_jordan_block,
    spectral_components,
)


def test_legacy_real_and_complex_chain_entries() -> None:
    assert_array_equal(real_jordan_block(1, 3), [[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    expected = [[0.5, 0.3, 1, 0], [-0.3, 0.5, 0, 1], [0, 0, 0.5, 0.3], [0, 0, -0.3, 0.5]]
    assert_array_equal(real_jordan_block(0.5 + 0.3j, 4), expected)
    values = np.linalg.eigvals(real_jordan_block(0.5 + 0.3j, 4))
    assert_allclose(np.sort_complex(values), np.sort_complex([0.5 + 0.3j, 0.5 - 0.3j] * 2))


def test_single_linkage_uses_transitive_original_distances() -> None:
    # Endpoints .10 apart, linked through .05: centroid-only clustering would differ.
    centres, multiplicities = cluster_eigenvalues([1, 1.05, 1.1, -2], 0.06)
    assert_allclose(centres, [-2, 1.05])
    assert_array_equal(multiplicities, [1, 3])


def test_zero_tolerance_preserves_independent_repeated_entries() -> None:
    centres, multiplicities = cluster_eigenvalues([1, 1], 0)
    assert_array_equal(centres, [1, 1])
    assert_array_equal(multiplicities, [1, 1])
    assert spectral_components([1, 1], 0.001) == (JordanComponent(1 + 0j, 2),)


def test_cluster_boundary_is_strict() -> None:
    assert_array_equal(cluster_eigenvalues([0, 0.5], 0.5)[1], [1, 1])


def test_complex_clustering_and_total_multiplicity() -> None:
    values = [1 + 0.2j, 1 - 0.2j, 1.01 + 0.21j, 1.01 - 0.21j, 2, 2.01]
    components = spectral_components(values, 0.03)
    assert sum(c.dimension for c in components) == len(values)
    assert sorted(c.dimension for c in components) == [2, 4]
    assert_allclose(jordan_matrix(components).shape, (6, 6))
    pair = next(c for c in components if c.eigenvalue.imag > 0)
    assert_allclose(pair.eigenvalue, 1.005 + 0.205j)


def test_cluster_crossing_real_axis_preserves_pair_as_real_chain() -> None:
    components = spectral_components([1 + 0.001j, 1 - 0.001j, 1.001], 0.01)
    assert components[0].dimension == 3
    assert components[0].eigenvalue.imag == 0
    assert_allclose(np.linalg.eigvals(jordan_matrix(components)), [1.000333333] * 3)


def test_spectral_order_is_deterministic_and_has_no_lost_pairs() -> None:
    values = [1 + 0.3j, 1 - 0.3j, -1.4, 0.2, 0.1 + 0.5j, 0.1 - 0.5j]
    expected = spectral_components(values)
    for seed in range(10):
        assert spectral_components(np.random.default_rng(seed).permutation(values)) == expected


@pytest.mark.parametrize("imaginary", [1e-12, 1e-200])
def test_tiny_imaginary_parts_preserve_real_complex_structure(imaginary: float) -> None:
    components = spectral_components([1 + imaginary * 1j, 1 - imaginary * 1j])
    assert components == (JordanComponent(1 + imaginary * 1j, 2),)
    with pytest.raises(ValueError, match="unpaired"):
        spectral_components([1 + 1e-12j, 1 + 2e-12j])


@pytest.mark.parametrize("values", [[1j], [1j, -1j, 1j], [np.nan], [[1, 2]]])
def test_invalid_spectrum(values: object) -> None:
    with pytest.raises(ValueError):
        spectral_components(values)


@pytest.mark.parametrize("value,dimension", [(1j, 3), (-1j, 2), (np.inf, 1), (1, 0)])
def test_invalid_block(value: complex, dimension: int) -> None:
    with pytest.raises(ValueError):
        real_jordan_block(value, dimension)
