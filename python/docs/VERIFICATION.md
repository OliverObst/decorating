# Verification of the Python extraction

Verified on 11 October 2026 with Python 3.13.15, NumPy 2.5.4, SciPy 1.18.1,
Matplotlib 3.11.2 and pytest 9.1.1 on macOS arm64. Dependencies are pinned in
`uv.lock`; numerical experiment JSON includes the BLAS identities.

## Library and packaging

```bash
uv sync --extra dev --extra audit --python 3.13
uv run pytest
uv run ruff check src tests examples experiments
uv run mypy src
uv run python examples/sinusoid.py
uv build
```

All 83 tests pass, including both fitting modes, whole conjugate-pair
retention, Jordan chains, omission scoring, smallest-prefix selection,
numerically non-monotone prefix errors and end-to-end forecasts. Ruff and
mypy pass; mypy checks all 11 library modules.

The sinusoid example reduces 41 state values to two, with training RMSE
`3.144e-10` and 100-step future RMSE `1.167e-9`. The initial regression's
condition warning remains visible.

A wheel was built and installed with only its core dependencies into a new
temporary virtual environment outside the repository. Python isolated mode
loaded `lrnn` from that environment's `site-packages`, confirmed that
`refine=False` is the API default, and ran a copied sinusoid example with the
same output. No UP-LRNN or toy robot package was imported. The temporary
environment was removed afterwards.

The source distribution includes the library, tests, example, guides,
experiment drivers, dependency lock, recorded/verified
measurements and the eight licensed Octave reference routines. Generated
working results, virtual environments and build artefacts are excluded.

## Full MSO reruns

All four 100-seed runs completed, using one reservoir per seed and selecting
the winner by validation RMSE. Complete fresh outputs are preserved under
[`experiments/verified/`](../experiments/verified/).

| Profile | Refinement | Finite forecasts | Test RMSE < 1e-5 | Selected seed | Selected test RMSE |
| --- | --- | --- | --- | --- | --- |
| paper: 150/50/100, 70 units | Off | 100/100 | 0/100 | 38 | 0.176252 |
| paper: 150/50/100, 70 units | On | 99/100 | 99/100 | 12 | 2.061e-14 |
| legacy: 200/50/50, 198 units | Off | 100/100 | 0/100 | 54 | 2.174e-5 |
| legacy: 200/50/50, 198 units | On | 100/100 | 100/100 | 87 | 2.726e-14 |

Every trial's status, dimension and test RMSE agrees exactly with the
corresponding prior recorded run, despite the NumPy patch-version change.
The unstable 41-dimensional refined model in the shorter configuration is
retained as a failed trial, with its fitting diagnostics. The
[accuracy guide](ACCURACY.md) explains the fitting and evaluation protocols.

## Numerical audit and Octave reference

The fresh Python numerical audit completed. For the shared seed-0 reservoir,
the baseline retains 16 dimensions and has future RMSE `0.862390` over 150
samples. Explicit refinement gives future RMSE `1.807e-14`. Solver variants,
fixture, spectra, configuration and source hashes are in
[`verified/numerical_audit/`](../experiments/verified/numerical_audit/).

Octave is not installed on this host, so the native comparison was not rerun
during extraction. The prior complete Octave/OpenBLAS/60-digit audit and all
100 native Octave trials remain unmodified in
[`recorded/numerical_audit/`](../experiments/recorded/numerical_audit/).
Every archived file and all eight bundled Octave routines were checked
against their original SHA-256 hashes. The routines also match the parent
repository's `lrnn/` files. The [experiment guide](../experiments/README.md)
provides the command to rerun the native comparison with an installed Octave.
