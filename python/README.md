# Python LRNN

A self-contained NumPy/SciPy implementation of autonomous linear recurrent
neural networks, including real Jordan-block construction, omission scoring,
binary-search dimension reduction and forecasting. This implements the
reservoir-and-spectrum procedure from
[Efficient time-series approximation with linear recurrent neural networks:
architecture learning and predictive power](https://link.springer.com/article/10.1007/s00521-025-11655-y),
published in *Neural Computing and Applications* **37**, 27027–27055 (2025).

The Python import is `lrnn`; the distribution name is `decorating-lrnn`.
Python 3.11 or newer is required. Core installation needs only NumPy and SciPy.
The directory can be copied and used independently of the surrounding repository.

## Install and try it

From this `python/` directory, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --python 3.13
uv run python examples/sinusoid.py
```

`uv.lock` pins the dependencies used for the reproduction runs. With standard
Python tools instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/sinusoid.py
```

The example learns one sinusoid, reduces 41 state values to two, and predicts
100 future samples. Conditioning warnings expose properties of the fit; its
rank, singular values and condition estimate are available for inspection.

## API

```python
from lrnn import LRNN, rmse
from lrnn.benchmarks import mso_signal

signal = mso_signal(300)  # shape (channels, time), sampled at t=1,...,300
model = LRNN(reservoir_size=198, random_state=2).fit(signal[:, :200])
reduced = model.reduce(max_rmse=0.5, cluster_tol=0.0)
prediction = reduced.forecast(steps=100)
print(reduced.dimension, rmse(signal[:, 200:], prediction))
```

`model.forecast(...)` uses the full learned transition. `reduced.forecast(...)`
uses the smaller Jordan dynamics. Both return future samples only and leave
the fitted state unchanged. Arrays have shape `(channels, time)`.

The default reduction preserves the spectrum-and-refit procedure. Optional
training-only optimisation of the retained spectral parameters is explicit:

```python
refined = model.reduce(max_rmse=0.5, refine=True)
```

This option improves the measured fixed-frequency MSO forecasts substantially.
Read [accuracy and reproduction](docs/ACCURACY.md) for the commands, results,
failure cases and relationship to the paper's reported accuracy.

Inspect `model.transition_`, `model.driven_states_`, `model.eigenvalues_`,
`model.output_diagnostics_`, `reduced.fit`, `reduced.reduction` and
`reduced.refinement` for intermediate quantities. An infeasible reduction
threshold raises `ValueError`.

## Experiments and tests

```bash
uv run pytest
uv run ruff check src tests examples experiments
uv run mypy src
uv run python experiments/mso.py
uv run python experiments/mso.py --profile paper --refine
```

The MSO script records every seed and failure, selects by validation error,
and writes configuration, source hashes, CSV/JSON results and PNG/PDF plots.
The [experiment guide](experiments/README.md) also covers the shared-weight
Python/Octave comparison and optional high-precision diagnostics.

## Contents

- `src/lrnn/`: autonomous LRNN library and optional spectral refinement.
- `tests/`: numerical and end-to-end regression tests.
- `examples/`: small runnable example.
- `experiments/`: seeded MSO and numerical audit scripts.
- `experiments/recorded/`: unmodified prior measurements, including failures.
- `experiments/verified/`: verified MSO runs and Python numerical audit.
- `references/octave/`: eight pinned Octave routines and their licence.
- [Algorithm notes](docs/ALGORITHM.md): linear algebra and numerical differences.

The code is available under the [BSD 3-Clause licence](LICENSE), as is the
rest of the Decorating repository.
