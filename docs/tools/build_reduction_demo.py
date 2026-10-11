"""Construct a four-block example and refit every retained subset.

Run from python/: uv run python ../docs/tools/build_reduction_demo.py
Uses the library's real blocks, rollout and SVD output fit. The dynamics are
constructed to illustrate omission/refitting, without reservoir training.
"""
from pathlib import Path
import json

import numpy as np
from scipy.linalg import block_diag

from lrnn.dynamics import autonomous_rollout
from lrnn.metrics import rmse
from lrnn.regression import fit_map
from lrnn.spectrum import real_jordan_block

ROOT = Path(__file__).resolve().parents[2]
TRAIN = 120
LENGTH = 240
BLOCKS = [
    {"label": "Slow oscillation", "omega": 0.17, "rho": 1.0, "amplitude": 1.0, "phase": 0.1},
    {"label": "Faster oscillation", "omega": 0.43, "rho": 1.0, "amplitude": 0.55, "phase": 0.6},
    {"label": "Weak oscillation", "omega": 0.77, "rho": 1.0, "amplitude": 0.12, "phase": -0.4},
    {"label": "Decaying oscillation", "omega": 0.30, "rho": 0.94, "amplitude": 0.85, "phase": 0.2},
]


def construct():
    j = block_diag(*[
        real_jordan_block(b["rho"] * np.exp(1j * b["omega"]), 2)
        for b in BLOCKS
    ])
    initial = np.tile([1.0, 0.0], len(BLOCKS))
    states = autonomous_rollout(j, initial, LENGTH - 1)
    output = np.array([
        coefficient
        for b in BLOCKS
        for coefficient in [b["amplitude"] * np.cos(b["phase"]),
                            b["amplitude"] * np.sin(b["phase"])]
    ])[None, :]
    target = output @ states
    # A seeded orthogonal change of coordinates gives an equivalent dense W.
    basis, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((8, 8)))
    w = basis @ j @ basis.T
    dense_states = autonomous_rollout(w, basis @ initial, LENGTH - 1)
    np.testing.assert_allclose(output @ basis.T @ dense_states, target, atol=1e-12)
    subsets = []
    for mask in range(16):
        indices = [i for i in range(8) if mask & (1 << (i // 2))]
        jr = j[np.ix_(indices, indices)]
        yr = autonomous_rollout(jr, initial[indices], LENGTH - 1)
        if indices:
            ar, _ = fit_map(target[:, :TRAIN], yr[:, :TRAIN])
            prediction = ar @ yr
        else:
            ar = np.zeros((1, 0))
            prediction = np.zeros_like(target)
        residual = prediction - target
        subsets.append({
            "mask": mask, "dimension": len(indices), "matrix": jr.tolist(),
            "fixedPrediction": (output[:, indices] @ yr)[0].tolist(),
            "outputMap": ar[0].tolist(), "prediction": prediction[0].tolist(),
            "residual": residual[0].tolist(),
            "trainingRmse": rmse(target[:, :TRAIN], prediction[:, :TRAIN]),
            "futureRmse": rmse(target[:, TRAIN:], prediction[:, TRAIN:]),
            "fixedTrainingRmse": rmse(target[:, :TRAIN], (output[:, indices] @ yr)[:, :TRAIN]),
            "fixedFutureRmse": rmse(target[:, TRAIN:], (output[:, indices] @ yr)[:, TRAIN:]),
        })
    scores = [subsets[15 ^ (1 << i)]["trainingRmse"] for i in range(4)]
    ranking = sorted(range(4), key=lambda i: -scores[i])
    return {
        "trainLength": TRAIN, "length": LENGTH,
        "blocks": [dict(b, omissionRmse=scores[i]) for i, b in enumerate(BLOCKS)],
        "ranking": ranking, "fullMatrix": w.tolist(), "jordanMatrix": j.tolist(),
        "target": target[0].tolist(), "subsets": subsets,
    }


def verify(data):
    target = np.array(data["target"])
    assert data["subsets"][15]["trainingRmse"] < 1e-12
    assert data["subsets"][15]["futureRmse"] < 1e-12
    assert data["subsets"][0]["dimension"] == 0
    assert not any(data["subsets"][0]["prediction"])
    for subset in data["subsets"]:
        residual = np.array(subset["residual"])
        prediction = np.array(subset["prediction"])
        np.testing.assert_allclose(residual, prediction - target, atol=1e-15)
        np.testing.assert_allclose(subset["trainingRmse"], np.sqrt(np.mean(residual[:TRAIN] ** 2)))
        np.testing.assert_allclose(subset["futureRmse"], np.sqrt(np.mean(residual[TRAIN:] ** 2)))
        assert subset["trainingRmse"] <= subset["fixedTrainingRmse"] + 1e-12
        fixed = np.array(subset["fixedPrediction"])
        np.testing.assert_allclose(subset["fixedFutureRmse"], np.sqrt(np.mean((fixed[TRAIN:] - target[TRAIN:]) ** 2)))
        # The optimum is orthogonal to the retained training features.
        if subset["dimension"]:
            indices = [i for i in range(8) if subset["mask"] & (1 << (i // 2))]
            yr = autonomous_rollout(np.array(subset["matrix"]), np.tile([1.0, 0.0], 4)[indices], TRAIN - 1)
            np.testing.assert_allclose(yr @ residual[:TRAIN], 0, atol=1e-12)
    assert data["ranking"][-1] == 2
    # Omitting this decaying component has a larger error on the training interval.
    without_transient = data["subsets"][7]
    assert without_transient["futureRmse"] < without_transient["trainingRmse"] / 2


if __name__ == "__main__":
    data = construct()
    verify(data)
    destination = ROOT / "docs" / "assets" / "reduction-data.js"
    destination.write_text(
        "/* Constructed example; regenerate with docs/tools/build_reduction_demo.py. */\n"
        + "window.lrnnReductionData = " + json.dumps(data, separators=(",", ":"), allow_nan=False) + ";\n"
    )
    print("Verified all 16 subsets; wrote", destination)
