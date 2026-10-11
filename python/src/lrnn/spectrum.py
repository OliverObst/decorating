"""Single-linkage spectral clustering and real Jordan dynamics.

No Jordan eigenvector basis is computed. Algebraic multiplicity is treated
as one Jordan chain, following the published spectrum-and-refit procedure.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import block_diag

from ._validation import FloatArray, integer


@dataclass(frozen=True)
class JordanComponent:
    """A real block; dimension counts both members of a complex conjugate pair."""

    eigenvalue: complex
    dimension: int

    def __post_init__(self) -> None:
        dimension = integer(self.dimension, "component dimension", 1)
        if not np.isfinite(self.eigenvalue):
            raise ValueError("eigenvalue must be finite")
        if self.eigenvalue.imag < 0:
            raise ValueError("represent a conjugate pair by its positive imaginary eigenvalue")
        if self.eigenvalue.imag != 0 and dimension % 2:
            raise ValueError("a complex component must have even dimension")


def cluster_eigenvalues(
    eigenvalues: ArrayLike, cluster_tol: float = 0.0
) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
    """Cluster by connected distances strictly below cluster_tol.

    Return centroids and total algebraic multiplicities, sorted by decreasing
    magnitude then phase. Zero tolerance leaves every entry separate, as in
    cluster.m. Negative tolerances are rejected rather than selecting a hidden
    heuristic. Distances concern original eigenvalues, not evolving centroids.
    """
    values = np.asarray(eigenvalues, dtype=np.complex128)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues must be a finite one-dimensional array")
    if not np.isfinite(cluster_tol) or cluster_tol < 0:
        raise ValueError("cluster_tol must be finite and non-negative")
    parents = list(range(values.size))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    if cluster_tol > 0:
        for i in range(values.size):
            for j in range(i):
                if abs(values[i] - values[j]) < cluster_tol:
                    parents[root(i)] = root(j)
    groups: dict[int, list[complex]] = {}
    for i, value in enumerate(values):
        groups.setdefault(root(i), []).append(complex(value))
    clustered = []
    for group in groups.values():
        centre = complex(np.mean(group))
        # A conjugate-symmetric cluster has an exactly real centroid. Pairwise
        # summation can leave round-off; only snap clusters that contain every
        # conjugate member, never an individual low-frequency complex mode.
        if sorted(group, key=lambda z: (z.real, z.imag)) == sorted(
            (value.conjugate() for value in group), key=lambda z: (z.real, z.imag)
        ):
            centre = complex(centre.real)
        clustered.append((centre, len(group)))
    clustered.sort(key=lambda item: (-abs(item[0]), -np.angle(item[0])))
    return (
        np.array([item[0] for item in clustered], dtype=np.complex128),
        np.array([item[1] for item in clustered], dtype=np.int64),
    )


def spectral_components(
    eigenvalues: ArrayLike, cluster_tol: float = 0.0, *, conjugate_tol: float = 1e-10
) -> tuple[JordanComponent, ...]:
    """Pair conjugate centroids without dropping negative imaginary eigenvalues.

    A cluster crossing the real axis is real and retains *all* its members.
    Unpaired complex eigenvalues or incompatible multiplicities are errors.
    """
    if not np.isfinite(conjugate_tol) or conjugate_tol < 0:
        raise ValueError("conjugate_tol must be finite and non-negative")
    centres, counts = cluster_eigenvalues(eigenvalues, cluster_tol)
    used: set[int] = set()
    components = []
    for i, (value, count) in enumerate(zip(centres, counts, strict=True)):
        if i in used:
            continue
        threshold = conjugate_tol * max(1.0, abs(value))
        if value.imag == 0:
            components.append(JordanComponent(complex(value.real), int(count)))
            used.add(i)
            continue
        candidates = [
            j
            for j in range(centres.size)
            if j != i
            and j not in used
            and centres[j].imag != 0
            and np.signbit(value.imag) != np.signbit(centres[j].imag)
            and counts[j] == count
            and abs(centres[j] - value.conjugate()) <= threshold
        ]
        if not candidates:
            raise ValueError(f"unpaired complex eigenvalue {value} with multiplicity {count}")
        j = min(candidates, key=lambda k: abs(centres[k] - value.conjugate()))
        centre = (value + centres[j].conjugate()) / 2
        components.append(JordanComponent(complex(centre.real, abs(centre.imag)), int(2 * count)))
        used.update((i, j))
    components.sort(
        key=lambda component: (-abs(component.eigenvalue), -np.angle(component.eigenvalue))
    )
    return tuple(components)


def real_jordan_block(eigenvalue: complex, dimension: int) -> FloatArray:
    """Build a real chain with diagonal [a,b;-b,a] for lambda=a+ib.

    dimension is the total real dimension (not the complex multiplicity).
    Real chains have unit superdiagonals; complex chains have identity 2x2
    blocks above the diagonal, exactly as in legacy block.m.
    """
    component = JordanComponent(complex(eigenvalue), dimension)
    value = component.eigenvalue
    result = np.eye(dimension) * value.real
    stride = 1 if value.imag == 0 else 2
    if stride == 2:
        for i in range(0, dimension, 2):
            result[i, i + 1] = value.imag
            result[i + 1, i] = -value.imag
    for i in range(dimension - stride):
        result[i, i + stride] = 1.0
    return result


def jordan_matrix(components: tuple[JordanComponent, ...]) -> FloatArray:
    """Assemble real Jordan blocks without an eigenvector transformation."""
    if not components:
        return np.empty((0, 0), dtype=np.float64)
    return np.asarray(
        block_diag(
            *[
                real_jordan_block(component.eigenvalue, component.dimension)
                for component in components
            ]
        ),
        dtype=np.float64,
    )
