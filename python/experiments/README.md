# Reproducible experiments

Run from `python/` after `uv sync --extra dev --python 3.13`.

## MSO

```bash
uv run python experiments/mso.py --profile legacy
uv run python experiments/mso.py --profile legacy --refine
uv run python experiments/mso.py --profile paper
uv run python experiments/mso.py --profile paper --refine
```

The default `legacy` profile uses the fixed eight-frequency signal, times
1–300, a 200/50/50 chronological split, 198 initial reservoir units, threshold
0.5 and no eigenvalue clustering. The `paper` profile uses 150/50/100 and 70
units. Its added validation/test split is an explicit forecast evaluation,
not the entire Figure 7 compactness sweep. Read [accuracy and reproduction](../docs/ACCURACY.md).

Each run fits one reservoir per declared seed, with seeds 0–99 by default.
Validation error selects the model, with no test-based selection, validation
refitting or hidden retries. Refinement is disabled by default. Use `--refine`
to optimise retained spectral parameters against training observations only.

Outputs under ignored `experiments/results/mso_<profile>[_refined]/` include:

- `results.json`: configuration, seeds, versions, source hashes, every trial,
  warnings and failures, and selection.
- `trials.csv`: per-trial errors, dimensions, timing and optimiser diagnostics.
- `forecast.csv`: selected trajectory with labelled train/validation/test samples.
- `mso.png` and `mso.pdf`: trajectory, residuals and finite test-error distribution.

Use `--output` for a separate destination and `--trials 3` for a smoke run.
Other CLI options expose the split, size, seed range, SVD cut-off and thresholds.
Fitting failures are retained; a run with no feasible trial exits unsuccessfully.

## Numerical and Octave audit

```bash
uv run python experiments/numerical_audit.py
```

The audit fixes seed-0 data and reservoir weights: 150 training samples, 70
reservoir units and 150 future samples. It compares NumPy/SciPy eigenvalue
methods, balancing, real Schur decomposition and SVD/QR regression drivers.
It also evaluates the explicit spectral refinement. True MSO frequencies are
used to score eigenvalues only, never supplied to fitting.

With an installed Octave CLI:

```bash
uv run python experiments/numerical_audit.py --octave octave-cli --octave-trials 100
```

On systems where the executable is named `octave`, use `--octave octave`.
Octave is an optional system dependency and is not installed by `uv sync`.
The comparison uses the eight bundled routines in `references/octave/lrnn/`,
unchanged from commit `2b55cd2e5e9c8f3b4b451e8df767ffc8537a57a5`.
They also match the surrounding repository's `lrnn/` files at extraction.
No surrounding repository files or UP-LRNN installation are needed.

The shared-weight comparison explicitly checks the final reduction prefix;
native trials call the original `predict.m`, preserving its retries and
boundary behaviour. Native Octave seed values do not generate the same weights
as Python seed values. The subprocess timeout is configurable with
`--octave-timeout`; its default is 180 seconds.

For optional higher-precision and independent BLAS diagnostics:

```bash
uv sync --extra dev --extra audit --python 3.13
uv run python experiments/numerical_audit.py --mp-dps 60 \
  --octave octave-cli --octave-trials 100 \
  --openblas-lp64 /path/to/LP64/libopenblas.dylib
```

Supply an LP64 library exposing `LAPACKE_dgeevx`, not an ILP64 library. These
tools are optional and do not alter the core algorithm. The audit saves data,
states, weights and the learned matrix in `fixture.npz`, spectra in
`spectra.npz`, and settings, versions, BLAS providers and source hashes in
`results.json`. Octave runs also save its shared-weight quantities, native-trial
CSV and subprocess log.

## Recorded and fresh results

`recorded/` preserves prior full measurements, including unsuccessful trials
and the Octave/high-precision audit. Their source hashes refer to the original
implementation, before the namespace/default changes. They are not presented
as newly run measurements of this package.

Fresh results go to ignored `results/`. The [verification record](../docs/VERIFICATION.md)
describes checks performed on this extraction. The completed verification runs
are preserved separately in `verified/`, with source hashes referring to this
package. Its four MSO conditions include every trial and failure. The fresh
numerical audit covers the Python side; the prior full Octave audit is in
`recorded/numerical_audit/`.
