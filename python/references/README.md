# Pinned Octave reference

The algorithm is described in
[Efficient time-series approximation with linear recurrent neural networks:
architecture learning and predictive power](https://link.springer.com/article/10.1007/s00521-025-11655-y),
*Neural Computing and Applications* **37**, 27027–27055 (2025).

The eight routines under `octave/lrnn/` are unmodified copies from
[`OliverObst/decorating`](https://github.com/OliverObst/decorating) at commit
`2b55cd2e5e9c8f3b4b451e8df767ffc8537a57a5`. They match the parent repository's
`lrnn/` routines at extraction. Their original BSD 3-Clause licence is retained
in `octave/LICENSE`; the Python code has its own MIT licence.

The bundled snapshot makes the audit usable when `python/` is copied elsewhere.
`experiments/numerical_audit.py` records the routine hashes and invokes them
through `experiments/octave_audit.m`. The wrapper is an audit driver, not a
modification of the pinned implementation.

| Octave | Python |
| --- | --- |
| `start.m`, `reservoir.m` | `lrnn.reservoir` |
| `compute.m` | `lrnn.dynamics` |
| `cluster.m`, `block.m`, `jormat.m` | `lrnn.spectrum` |
| `rmse.m` | `lrnn.metrics` |
| `predict.m` | `lrnn.model` and `lrnn.reduction` |
