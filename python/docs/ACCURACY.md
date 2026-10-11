# Accuracy and reproduction

## What the paper reports

Section 5.1 of the
[published paper](https://link.springer.com/article/10.1007/s00521-025-11655-y)
reports compact, 16-dimensional LRNNs with RMSE below `1e-5` for
the fixed eight-frequency multiple superimposed oscillators (MSO) signal.
Figure 7 studies training lengths 130, 150 and 200 and a range of initial
reservoir sizes. Its compactness experiment includes 150 training samples and
70 reservoir units. The separate executable forecasting experiment uses
200 training / 50 validation / 50 test samples and 100 trials.

Our scripts distinguish these settings. `--profile paper` uses 150 training
samples and 70 reservoir units, adding 50 validation and 100 test samples for
forecast evaluation. `--profile legacy` uses 200/50/50 and 198 reservoir units.
Neither profile reproduces the full Figure 7 sweep or the random-frequency
table. The fixed signal is sampled at `t=1,...,300`, with angular frequencies
`[0.200, 0.311, 0.420, 0.510, 0.630, 0.740, 0.850, 0.970]` radians per sample.
It is not normalised, rescaled by `2*pi`, or contaminated with noise.

## How to obtain sub-1e-5 forecast error in this MSO setting

Use explicit autonomous spectral refinement:

```bash
uv sync --extra dev --python 3.13
uv run python experiments/mso.py --profile paper --refine
```

This first performs the usual omission ranking and dimension reduction. It
then optimises the retained spectral parameters against the autonomous
**training** trajectory, refitting the linear output map by SVD at each
evaluation. It preserves the number and structure of retained blocks. True
frequencies, validation/test observations, known model order and unit-circle
constraints are not supplied to this optimisation. Seeds 0–99 are fitted;
validation RMSE selects one model without refitting it on validation data.

For the longer training configuration:

```bash
uv run python experiments/mso.py --profile legacy --refine
```

For the default spectrum-and-refit method, omit `--refine`:

```bash
uv run python experiments/mso.py --profile paper
uv run python experiments/mso.py --profile legacy
```

The two fitting modes write separate result directories. For a short check,
add `--trials 3`. The full protocol uses 100 trials. All settings are CLI
options and are stored with the results; run `--help` to inspect them.

## Recorded measurements

The following are the unmodified prior measurements included in
[`experiments/recorded/`](../experiments/recorded/). Their JSON files record
Python 3.13.15, NumPy 2.5.3, SciPy 1.18.1 and Matplotlib 3.11.2 on macOS arm64.
The original source hashes remain in those files. Fresh runs of this extraction
write their own hashes and versions under `experiments/results/`.
The four conditions were also rerun after extraction, with NumPy 2.5.4 and
the other listed numerical dependencies unchanged. Every trial's status,
dimension and test RMSE agrees exactly with the corresponding recorded run.
These fresh runs are preserved in
[`experiments/verified/`](../experiments/verified/); see the
[verification record](VERIFICATION.md).

| Configuration | Fitting | Finite forecasts / 100 | Test RMSE < 1e-5 / 100 | Median test RMSE, finite trials | Validation-selected test RMSE |
| --- | --- | --- | --- | --- | --- |
| 150/50/100, 70 units | Spectrum and output-map fit | 100 | 0 | 1.230 | 0.1763 |
| 150/50/100, 70 units | With spectral refinement | 99 | 99 | 1.922e-14 | 2.061e-14 |
| 200/50/50, 198 units | Spectrum and output-map fit | 100 | 0 | 2.312e-4 | 2.174e-5 |
| 200/50/50, 198 units | With spectral refinement | 100 | 100 | 2.745e-14 | 2.726e-14 |

In the shorter configuration, both fitting modes retain 16 dimensions in
99/100 trials. Seed 58 retains 41 dimensions. After refinement its training
RMSE is `4.716e-15`, but the forecast overflows at step 59. That failure remains
in the records and the denominator. Small training error and optimiser
convergence do not guarantee stable extrapolation.

These measurements show how the explicit fitting option achieves the paper's
reported accuracy level in this particular noiseless signal. They do not
establish every operation or setting used in the historical experiments.
They also do not establish performance on the paper's random-frequency cases,
noisy signals or other applications.

## Rank sensitivity and the Octave comparison

With the longer configuration, changing the baseline SVD cut-off to machine
epsilon gives a validation-selected test RMSE of `6.656e-6`; only 2/100 trials
have test RMSE below `1e-5`. The command is:

```bash
uv run python experiments/mso.py --rcond 2.220446049250313e-16 \
  --output experiments/results/mso_legacy_epsilon
```

The [shared-weight numerical audit](../experiments/README.md#numerical-and-octave-audit)
compares Python and the pinned Octave implementation on identical data and
reservoir weights. Recorded driven states agree within `2.843e-14`. The
baseline forecast errors are `0.8624` in Python and `0.8649` in Octave over
150 future samples, with 16 retained dimensions. The explicit Python spectral
refinement gives `1.807e-14` in that comparison.

A separate 100-trial run of the unmodified Octave `predict.m` retains 16
dimensions in 97 trials and has median future RMSE `1.111`; none has future
RMSE below `1e-5`. Its native seeds use a different RNG from Python and are
not paired reservoirs. The audit additionally examines solver and precision
effects, keeping the stored float64 matrix fixed. Those results describe the
tested configurations and do not attribute the measured differences to errors
in the original paper.

Numerical results can vary with BLAS/LAPACK and dependency versions, especially
for poorly conditioned fits. Use `uv.lock` for the Python dependencies, inspect
the saved diagnostics, and report the complete trial distribution. The
[algorithm notes](ALGORITHM.md) list deliberate numerical differences.
