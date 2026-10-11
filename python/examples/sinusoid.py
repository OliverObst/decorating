"""Fit a sinusoid, reduce to a real conjugate pair, and predict 100 samples.

Run from python/: uv run python examples/sinusoid.py
"""

import numpy as np

from lrnn import LRNN, rmse


def main() -> None:
    signal = np.sin(np.pi * np.arange(201) / 100)[None, :]
    model = LRNN(reservoir_size=40, random_state=2).fit(signal[:, :101])
    reduced = model.reduce(max_rmse=0.01, cluster_tol=0.03)
    prediction = reduced.forecast(steps=100)
    print(f"Full state dimension: {model.transition_.shape[0]}")
    print(f"Reduced state dimension: {reduced.dimension}")
    print(f"Training RMSE: {reduced.training_rmse:.6g}")
    print(f"Future RMSE: {rmse(signal[:, 101:], prediction):.6g}")


if __name__ == "__main__":
    main()
