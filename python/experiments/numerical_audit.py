"""Controlled solver audit: identical training data, reservoir and learned matrix.

mpmath and an Octave executable are optional diagnostic tools, never core
requirements. Known MSO frequencies are used only to score eigenvalues.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import subprocess
import warnings
from pathlib import Path

import numpy as np
import scipy
from scipy import linalg
from scipy.io import loadmat, savemat
from scipy.optimize import linear_sum_assignment

from lrnn import LRNN, rmse
from lrnn.benchmarks import MSO_FREQUENCIES, mso_signal
from lrnn.dynamics import autonomous_rollout
from lrnn.reduction import reduce_components
from lrnn.spectrum import spectral_components

ROOT = Path(__file__).resolve().parents[1]


def matched_error(left: np.ndarray, right: np.ndarray) -> float:
    rows, cols = linear_sum_assignment(abs(left[:, None] - right))
    if rows.size != left.size:
        raise ValueError("insufficient eigenvalues for diagnostic matching")
    return float(np.max(abs(left[rows] - right[cols])))


def score(values: np.ndarray, signal: np.ndarray) -> dict:
    positive = values[values.imag > 0]
    result = {
        "max_true_mode_error": matched_error(np.exp(1j * np.array(MSO_FREQUENCIES)), positive)
    }
    try:
        fit = reduce_components(signal[:, :150], spectral_components(values), 0.5).fit
        future = fit.output_map @ autonomous_rollout(fit.transition, fit.states[:, -1], 150)[:, 1:]
        result.update(
            dimension=fit.transition.shape[0],
            training_rmse=fit.training_rmse,
            future_rmse=rmse(signal[:, 150:], future),
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        result["failure"] = str(error)
    return result


def geevx_lp64(matrix: np.ndarray, library: str, balancing: bytes) -> tuple[np.ndarray, float]:
    """Call the explicit 32-bit-integer LAPACKE ABI; caller must supply LP64."""
    provider = ctypes.CDLL(library)
    function = provider.LAPACKE_dgeevx
    pointer = ctypes.POINTER(ctypes.c_double)
    integer_pointer = ctypes.POINTER(ctypes.c_int)
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char,
        ctypes.c_char,
        ctypes.c_char,
        ctypes.c_char,
        ctypes.c_int,
        pointer,
        ctypes.c_int,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
        pointer,
        ctypes.c_int,
        integer_pointer,
        integer_pointer,
        pointer,
        pointer,
        pointer,
        pointer,
    ]
    function.restype = ctypes.c_int
    n = matrix.shape[0]
    work = np.array(matrix, order="F")
    wr, wi, scale, rconde, rcondv = (np.zeros(n) for _ in range(5))
    vl, vr = (np.zeros((n, n), order="F") for _ in range(2))
    ilo, ihi, abnrm = ctypes.c_int(), ctypes.c_int(), ctypes.c_double()

    def ptr(array: np.ndarray):
        return array.ctypes.data_as(pointer)

    info = function(
        102,
        balancing,
        b"V",
        b"V",
        b"B",
        n,
        ptr(work),
        n,
        ptr(wr),
        ptr(wi),
        ptr(vl),
        n,
        ptr(vr),
        n,
        ctypes.byref(ilo),
        ctypes.byref(ihi),
        ptr(scale),
        ctypes.byref(abnrm),
        ptr(rconde),
        ptr(rcondv),
    )
    if info != 0:
        raise RuntimeError(f"LAPACKE_dgeevx returned info={info}")
    return wr + 1j * wi, float(np.min(rconde[wi > 0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/numerical_audit"))
    parser.add_argument("--mp-dps", type=int, default=0)
    parser.add_argument("--octave", help="Octave CLI executable; must be on PATH or absolute")
    parser.add_argument("--octave-trials", type=int, default=100)
    parser.add_argument("--octave-timeout", type=int, default=180)
    parser.add_argument("--openblas-lp64", help="LP64 library exposing LAPACKE_dgeevx")
    args = parser.parse_args()
    if args.octave_trials < 1 or args.octave_timeout < 1 or args.mp_dps < 0:
        parser.error("trial count and timeout must be positive; precision must be non-negative")
    args.output.mkdir(parents=True, exist_ok=True)
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numpy_blas": np.show_config(mode="dicts")["Build Dependencies"]["blas"],
        "scipy_blas": scipy.show_config(mode="dicts")["Build Dependencies"]["blas"],
    }
    signal = mso_signal(300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LRNN(70, 0).fit(signal[:, :150])
        matrix = model.transition_
        np.savez(
            args.output / "fixture.npz",
            S=signal,
            X=model.driven_states_,
            W=matrix,
            Win=model.input_weights_,
            Wres=model.reservoir_weights_,
        )
        spectra = {
            "numpy_geev": np.linalg.eigvals(matrix),
            "scipy_geev": linalg.eigvals(matrix),
            "scipy_balanced_geev": linalg.eigvals(linalg.matrix_balance(matrix)[0]),
            "scipy_real_schur": linalg.eigvals(linalg.schur(matrix, output="real")[0]),
        }
        eigen_audit = {name: score(values, signal) for name, values in spectra.items()}
        regression_audit = {}
        for driver in ["gelsd", "gelss", "gelsy", "gejsv"]:
            for scaling in ["raw", "equilibrated"]:
                states = model.driven_states_[:, :-1]
                scales = (
                    np.ones(states.shape[0]) if scaling == "raw" else np.max(abs(states), axis=1)
                )
                design = (states / scales[:, None]).T
                target = signal[:, 1:150].T
                cutoff = np.finfo(float).eps * max(states.shape)
                if driver == "gejsv":
                    singular, u, v, work, _, info = linalg.lapack.dgejsv(design)
                    if info != 0:
                        raise RuntimeError(f"DGEJSV returned info={info}")
                    singular *= work[0] / work[1]
                    retained = singular > cutoff * singular[0]
                    rank = np.count_nonzero(retained)
                    coefficients = v[:, retained] @ (
                        (u[:, retained].T @ target) / singular[retained, None]
                    )
                else:
                    coefficients, _, rank, _ = linalg.lstsq(
                        design, target, cond=cutoff, lapack_driver=driver
                    )
                output_map = coefficients.T / scales[None, :]
                learned = np.vstack(
                    (output_map, np.hstack((model.input_weights_, model.reservoir_weights_)))
                )
                regression_audit[f"{driver}_{scaling}"] = dict(
                    rank=int(rank),
                    one_step_rmse=rmse(signal[:, 1:150], output_map @ states),
                    **score(linalg.eigvals(learned), signal),
                )
        if args.openblas_lp64:
            provider = ctypes.CDLL(args.openblas_lp64)
            provider.openblas_get_config.restype = ctypes.c_char_p
            versions["openblas"] = provider.openblas_get_config().decode()
            for balancing in [b"B", b"N", b"S", b"P"]:
                values, condition = geevx_lp64(matrix, args.openblas_lp64, balancing)
                name = "openblas_geevx_" + balancing.decode()
                spectra[name] = values
                eigen_audit[name] = dict(
                    min_reciprocal_eigenvalue_condition=condition, **score(values, signal)
                )
        if args.mp_dps:
            import mpmath as mp

            versions["mpmath"] = mp.__version__
            mp.mp.dps = args.mp_dps
            exact_matrix = mp.matrix(matrix.tolist())  # exact float64 entries, no regeneration
            high_values = mp.eig(exact_matrix, left=False, right=False)

            def converted(values) -> np.ndarray:
                # Real input may leave arbitrary-precision rounding remnants on real roots.
                return np.array(
                    [
                        complex(
                            float(mp.re(v)),
                            0
                            if abs(mp.im(v)) < mp.mpf(10) ** (-args.mp_dps // 2)
                            else float(mp.im(v)),
                        )
                        for v in values
                    ]
                )

            values = converted(high_values)
            spectra["mpmath_fixed_matrix"] = values
            eigen_audit["mpmath_fixed_matrix"] = dict(
                max_difference_from_numpy=matched_error(spectra["numpy_geev"], values),
                **score(values, signal),
            )
            design = mp.matrix(model.driven_states_[:, :-1].T.tolist())
            target = mp.matrix(signal[:, 1:150].T.tolist())
            coefficients, residual = mp.qr_solve(design, target)
            exact_refit = exact_matrix.copy()
            for k in range(matrix.shape[0]):
                exact_refit[0, k] = coefficients[k]
            values = converted(mp.eig(exact_refit, left=False, right=False))
            spectra["mpmath_qr_then_eig"] = values
            regression_audit["mpmath_qr"] = dict(
                one_step_rmse=float(residual / mp.sqrt(149)), **score(values, signal)
            )
        refined = model.reduce(0.5, refine=True)
        refinement = dict(
            dimension=refined.dimension,
            training_rmse=refined.training_rmse,
            future_rmse=rmse(signal[:, 150:], refined.forecast(150)),
        )
    octave = None
    if args.octave:
        fixture_path = (args.output / "octave_fixture.mat").resolve()
        savemat(
            fixture_path,
            dict(
                S=signal,
                Win=model.input_weights_,
                Wres=model.reservoir_weights_,
                train_length=150,
                forecast_length=150,
            ),
        )
        prefix = (args.output / "octave").resolve()
        command = [
            args.octave,
            "--quiet",
            str(ROOT / "experiments/octave_audit.m"),
            str(ROOT / "references/octave/lrnn"),
            str(fixture_path),
            str(prefix),
            str(args.octave_trials),
        ]
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=args.octave_timeout, check=True
        )
        (args.output / "octave.log").write_text(result.stdout + result.stderr)
        fixed = loadmat(str(prefix) + "_fixed.mat")
        spectra["octave_fixed_weights"] = fixed["eigenvalues"].ravel()
        trials = np.genfromtxt(str(prefix) + "_trials.csv", delimiter=",", names=True, ndmin=1)
        octave = dict(
            version=str(fixed["octave_version"][0]),
            fixed_weights=dict(
                dimension=int(fixed["N"].item()),
                training_rmse=float(fixed["training_rmse"].item()),
                future_rmse=float(fixed["future_rmse"].item()),
                driven_state_max_difference=float(np.max(abs(fixed["X"] - model.driven_states_))),
            ),
            native_trials=dict(
                trials=len(trials),
                compact_16_dimension_trials=int(sum(trials["dimension"] == 16)),
                training_below_1e_5_trials=int(sum(trials["training_rmse"] < 1e-5)),
                future_below_1e_5_trials=int(sum(trials["future_rmse"] < 1e-5)),
                median_training_rmse=float(np.median(trials["training_rmse"])),
                median_future_rmse=float(np.median(trials["future_rmse"])),
            ),
        )
    np.savez(args.output / "spectra.npz", **spectra)
    report = dict(
        config=dict(
            seed=0,
            train_length=150,
            forecast_length=150,
            reservoir_size=70,
            max_rmse=0.5,
            mp_dps=args.mp_dps,
            openblas_lp64=args.openblas_lp64,
            octave=args.octave,
            octave_trials=args.octave_trials,
            octave_timeout=args.octave_timeout,
            true_frequencies="diagnostic scoring only; never supplied to fitting",
        ),
        versions=versions,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/lrnn").glob("*.py"))
            + [Path(__file__).resolve(), ROOT / "experiments/octave_audit.m"]
            + sorted((ROOT / "references/octave/lrnn").glob("*.m"))
        },
        eigenvalue_methods=eigen_audit,
        least_squares_methods=regression_audit,
        refinement=refinement,
        octave=octave,
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
