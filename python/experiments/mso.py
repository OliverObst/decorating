"""Reproduce fixed-frequency MSO with explicit trials and chronological splits.

Run from the python directory: uv run python experiments/mso.py
The default follows the executable legacy forecasting configuration. The
paper's shorter training configuration is available via --profile paper.
The default uses the spectrum-and-refit procedure; --refine enables optional
spectral optimisation with training-only output-map fitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from lrnn import LRNN, __version__, rmse
from lrnn.benchmarks import MSO_FREQUENCIES, mso_signal
from lrnn.regression import NumericalConditionWarning

LEGACY_COMMIT = "2b55cd2e5e9c8f3b4b451e8df767ffc8537a57a5"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--profile", choices=["legacy", "paper"], default="legacy")
    result.add_argument("--trials", type=int, default=100)
    result.add_argument(
        "--seed", type=int, default=0, help="First seed; trials use consecutive seeds"
    )
    result.add_argument("--reservoir-size", type=int)
    result.add_argument("--train-length", type=int)
    result.add_argument("--validation-length", type=int, default=50)
    result.add_argument("--test-length", type=int)
    result.add_argument("--max-rmse", type=float, default=0.5)
    result.add_argument("--cluster-tol", type=float, default=0)
    result.add_argument("--rcond", type=float, help="Default: NumPy's eps * max(design.shape)")
    result.add_argument(
        "--refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refine the autonomous training fit (default: disabled)",
    )
    result.add_argument("--max-nfev", type=int, default=200)
    result.add_argument(
        "--output",
        type=Path,
        help="Default: experiments/results/mso_<profile>[_refined], matching the fitting mode",
    )
    return result


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src/lrnn").glob("*.py")) + [Path(__file__).resolve()]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def main() -> int:
    args = parser().parse_args()
    if args.output is None:
        suffix = "_refined" if args.refine else ""
        args.output = Path(f"experiments/results/mso_{args.profile}{suffix}")
    train_length = (
        args.train_length
        if args.train_length is not None
        else (200 if args.profile == "legacy" else 150)
    )
    test_length = (
        args.test_length
        if args.test_length is not None
        else (50 if args.profile == "legacy" else 100)
    )
    reservoir_size = (
        args.reservoir_size
        if args.reservoir_size is not None
        else (train_length - 2 if args.profile == "legacy" else 70)
    )
    if (
        min(args.trials, args.validation_length, test_length, reservoir_size) < 1
        or train_length < 2
    ):
        raise ValueError("trial counts, dimensions and split lengths must be positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    total = train_length + args.validation_length + test_length
    signal = mso_signal(total)
    config = {
        "profile": args.profile,
        "frequencies_radians_per_sample": list(MSO_FREQUENCIES),
        "time_origin": 1,
        "train_length": train_length,
        "validation_length": args.validation_length,
        "test_length": test_length,
        "reservoir_size": reservoir_size,
        "trials": args.trials,
        "first_seed": args.seed,
        "seeds": list(range(args.seed, args.seed + args.trials)),
        "max_rmse": args.max_rmse,
        "cluster_tol": args.cluster_tol,
        "rcond": args.rcond,
        "refine": args.refine,
        "refinement_max_nfev": args.max_nfev,
        "input_scaling": "balanced",
        "dtype": "float64",
        "selection": "minimum validation RMSE; no test data used; no refit after selection",
        "retry_policy": "one reservoir per seed; all failed trials retained",
        "rmse": "sqrt(mean over time of sum of squared errors over dimensions)",
    }
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "lrnn": __version__,
        "platform": platform.platform(),
        "numpy_blas": np.show_config(mode="dicts")["Build Dependencies"]["blas"],
        "scipy_blas": scipy.show_config(mode="dicts")["Build Dependencies"]["blas"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    best_record = None
    best_prediction = None
    best_validation = np.inf
    experiment_start = perf_counter()
    fields = [
        "seed",
        "status",
        "dimension",
        "one_step_rmse",
        "training_rmse",
        "validation_rmse",
        "test_rmse",
        "runtime_seconds",
        "output_rank",
        "output_features",
        "output_condition_number",
        "reduced_condition_number",
        "conditioning_warnings",
        "failure",
        "baseline_training_rmse",
        "refinement_converged",
        "refinement_status",
        "refinement_evaluations",
        "refinement_rejected_evaluations",
    ]
    with (args.output / "trials.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for seed in config["seeds"]:
            started = perf_counter()
            record = dict.fromkeys(fields)
            record["seed"] = seed
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", NumericalConditionWarning)
                try:
                    model = LRNN(reservoir_size, seed, rcond=args.rcond).fit(
                        signal[:, :train_length]
                    )
                    reduced = model.reduce(
                        args.max_rmse,
                        args.cluster_tol,
                        refine=args.refine,
                        max_nfev=args.max_nfev,
                    )
                    # Preserve fit diagnostics even if a subsequent forecast fails.
                    record.update(
                        dimension=reduced.dimension,
                        one_step_rmse=model.one_step_rmse_,
                        training_rmse=reduced.training_rmse,
                        baseline_training_rmse=reduced.reduction.fit.training_rmse,
                        refinement_converged=(
                            reduced.refinement.converged if reduced.refinement else None
                        ),
                        refinement_status=(
                            reduced.refinement.status if reduced.refinement else None
                        ),
                        refinement_evaluations=(
                            reduced.refinement.residual_evaluations if reduced.refinement else None
                        ),
                        refinement_rejected_evaluations=(
                            reduced.refinement.rejected_evaluations if reduced.refinement else None
                        ),
                    )
                    prediction = np.hstack(
                        (reduced.reconstruct(), reduced.forecast(total - train_length))
                    )
                    validation_end = train_length + args.validation_length
                    validation_error = rmse(
                        signal[:, train_length:validation_end],
                        prediction[:, train_length:validation_end],
                    )
                    record.update(
                        {
                            "status": "ok",
                            "dimension": reduced.dimension,
                            "one_step_rmse": model.one_step_rmse_,
                            "training_rmse": reduced.training_rmse,
                            "baseline_training_rmse": reduced.reduction.fit.training_rmse,
                            "refinement_converged": (
                                reduced.refinement.converged if reduced.refinement else None
                            ),
                            "refinement_status": (
                                reduced.refinement.status if reduced.refinement else None
                            ),
                            "refinement_evaluations": (
                                reduced.refinement.residual_evaluations
                                if reduced.refinement
                                else None
                            ),
                            "refinement_rejected_evaluations": (
                                reduced.refinement.rejected_evaluations
                                if reduced.refinement
                                else None
                            ),
                            "validation_rmse": validation_error,
                            "test_rmse": rmse(
                                signal[:, validation_end:], prediction[:, validation_end:]
                            ),
                            "output_rank": model.output_diagnostics_.rank,
                            "output_features": model.output_diagnostics_.features,
                            "output_condition_number": (
                                model.output_diagnostics_.condition_number
                                if np.isfinite(model.output_diagnostics_.condition_number)
                                else None
                            ),
                            "reduced_condition_number": (
                                reduced.fit.diagnostics.condition_number
                                if np.isfinite(reduced.fit.diagnostics.condition_number)
                                else None
                            ),
                        }
                    )
                    if validation_error < best_validation:
                        best_validation = validation_error
                        best_prediction = prediction
                        best_record = record
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    record.update({"status": "failed", "failure": str(error)})
            record["conditioning_warnings"] = [str(warning.message) for warning in caught]
            record["runtime_seconds"] = perf_counter() - started
            records.append(record)
            writer.writerow(record)
            csv_file.flush()
            print(
                f"trial {len(records)}/{args.trials}: seed={seed}, status={record['status']}, "
                f"dimension={record['dimension']}, validation RMSE={record['validation_rmse']}",
                flush=True,
            )
    successful = [record for record in records if record["status"] == "ok"]
    summary = {
        "successful_trials": len(successful),
        "failed_trials": args.trials - len(successful),
        "compact_16_dimension_trials": sum(record["dimension"] == 16 for record in successful),
        "compact_fraction_of_all_trials": sum(record["dimension"] == 16 for record in successful)
        / args.trials,
        "training_rmse_below_1e_5_trials": sum(
            record["training_rmse"] < 1e-5 for record in successful
        ),
        "test_rmse_below_1e_5_trials": sum(record["test_rmse"] < 1e-5 for record in successful),
        "median_test_rmse_all_successful": (
            float(np.median([record["test_rmse"] for record in successful])) if successful else None
        ),
        "selected": best_record,
        "total_runtime_seconds": perf_counter() - experiment_start,
    }
    output = {
        "config": config,
        "versions": versions,
        "source_sha256": source_hashes(),
        "references": {
            "paper": "https://link.springer.com/article/10.1007/s00521-025-11655-y",
            "legacy_commit": LEGACY_COMMIT,
            "legacy_split": "mso8/octave/compute_prediction.m: 200/50/50, 100 trials",
            "paper_profile": "Section 5.1: 150 training, 70 reservoir units; "
            "50 validation and 100 test samples are an explicit extension",
        },
        "summary": summary,
        "trials": records,
    }
    (args.output / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    if best_record is None or best_prediction is None:
        print("No feasible trial; saved all failures.", file=sys.stderr)
        return 1
    time = np.arange(1, total + 1)
    split_labels = np.repeat(
        ["train", "validation", "test"], [train_length, args.validation_length, test_length]
    )
    with (args.output / "forecast.csv").open("w", newline="") as forecast_file:
        writer = csv.writer(forecast_file)
        writer.writerow(["time", "split", "observed", "forecast", "residual"])
        writer.writerows(
            zip(
                time,
                split_labels,
                signal[0],
                best_prediction[0],
                best_prediction[0] - signal[0],
                strict=True,
            )
        )
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
    axes[0].plot(time, signal[0], color="black", linewidth=1, label="Observed MSO")
    axes[0].plot(time, best_prediction[0], color="tab:blue", linestyle="--", label="Selected LRNN")
    axes[0].set(
        ylabel="Signal",
        title=f"Fixed-frequency MSO: selected seed {best_record['seed']}, "
        f"{best_record['dimension']} real dimensions",
    )
    axes[0].legend(loc="upper right")
    axes[1].plot(time, best_prediction[0] - signal[0], color="tab:blue", linewidth=1)
    axes[1].set(ylabel="Forecast − observed", xlabel="Time (samples)")
    for axis in axes[:2]:
        axis.axvline(
            train_length + 0.5, color="tab:orange", linestyle=":", label="Validation starts"
        )
        axis.axvline(validation_end + 0.5, color="tab:green", linestyle=":", label="Test starts")
        axis.grid(alpha=0.15)
    axes[2].scatter(
        [record["seed"] for record in successful],
        [max(record["test_rmse"], np.finfo(float).tiny) for record in successful],
        s=20,
        color="grey",
        label="All successful trials",
    )
    axes[2].scatter(
        [best_record["seed"]],
        [max(best_record["test_rmse"], np.finfo(float).tiny)],
        color="tab:blue",
        marker="*",
        s=120,
        label="Selected on validation",
    )
    axes[2].set(xlabel="Reservoir seed", ylabel="Test RMSE", yscale="log")
    axes[2].legend()
    figure.savefig(args.output / "mso.png", dpi=160)
    figure.savefig(args.output / "mso.pdf")
    plt.close(figure)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
