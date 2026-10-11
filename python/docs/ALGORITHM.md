# Autonomous LRNN algorithm and numerical choices

## Question and scope

Can the published reservoir-and-spectrum procedure recover a compact autonomous
representation and extrapolate its training signal? This implementation covers
autonomous prediction and spectral reduction. The reference is the
[published LRNN paper](https://link.springer.com/article/10.1007/s00521-025-11655-y)
(*Neural Computing and Applications*, 2025), especially Sections 3, 4 and 5.1.

## Shapes and sample timing

A series `S` has shape `(d, T)`. Columns are observations at consecutive times.
The complete state is `X(t) = [S(t); R(t)]`, with `r` reservoir components.
The initial reservoir state is the balanced vector `ones(r)/sqrt(r)`.

In input-receiving mode, observations are clamped and the reservoir follows

```text
R(t+1) = Win @ S(t) + Wres @ R(t).
```

Thus `S(t+1)` is not used to calculate `R(t+1)`. Solve

```text
S[:, 1:] = Wout @ X[:, :-1]
W = vstack([Wout, hstack([Win, Wres])]).
```

The unreduced model forecasts from the final clamped state. Every forecast call
returns future samples only: the first column corresponds to the next time after
the final training observation. Calls do not advance or overwrite the fitted state.

## Spectral representation

Extract the spectrum of `W`. When `cluster_tol > 0`, connect original eigenvalues
whose complex-plane distance is strictly below the tolerance, and use each
connected component's centroid and algebraic multiplicity. This is single-linkage
clustering, including transitive chains. It is not clustering by distance to an
evolving centroid. A zero tolerance leaves every eigenvalue separate, as in Octave.

Real eigenvalues give real Jordan chains with unit superdiagonals. A conjugate pair
`a ± ib` gives the real block `[[a, b], [-b, a]]`; repeated pairs have identity
2-by-2 blocks above the diagonal. A component's `dimension` counts both members
of each conjugate pair. If a cluster crosses the real axis, all its multiplicity
belongs to the resulting real chain. Pairing validates matching multiplicities
and rejects unmatched complex eigenvalues rather than discarding them.

As in the legacy procedure, a multiplicity is represented by a single Jordan
chain. The spectrum alone cannot identify the true Jordan structure of an
arbitrary matrix with repeated eigenvalues. This algorithm constructs candidate
dynamics and refits their trajectory; it does **not** claim to find a similarity
transformation or a numerical Jordan eigenvector basis.

For candidate dynamics `J`, construct `Y(t+1) = J @ Y(t)` from a balanced start
vector. Fit `S = A @ Y` across **all** training observations. Since regression is
separable over target rows, fitting only `S` gives the same output rows as fitting
the complete driven trajectory `X`. The selected model also exposes `state_map`,
obtained by fitting `X = state_map @ Y`, for comparison with `predict.m`.

## Omission and reduction

Omit each whole component, regenerate its remaining dynamics from a balanced
start vector, and refit the map. Its relevance score is the resulting training
RMSE; a larger omission error indicates a more relevant component. Sort scores
in decreasing order, with stable ties.

Binary-search the ranked prefixes, then audit smaller prefixes and return the
smallest non-empty prefix satisfying `RMSE <= max_rmse`. The feature spaces are
nested in exact arithmetic, but SVD rank cut-offs can break numerical
monotonicity. The audit ensures the stated smallest-prefix guarantee. It adds
solves beyond the paper's binary search. The guarantee concerns the fixed
omission ranking, not an optimal search over all subsets of components.

An infeasible tolerance raises an exception. A low training RMSE neither
guarantees an accurate forecast nor prevents unstable extrapolation.

## Numerical choices and differences from Octave

| Choice | Implemented behaviour and reason |
| --- | --- |
| Input weights | Default `input_scaling="balanced"` draws `randn(r,d)/sqrt(d)` as in `predict.m`. `"standard"` follows Definition 3 and Figure 5's unscaled Gaussian weights. The choice has no effect for a univariate signal and is tested for multivariate signals. |
| One-neuron reservoir | Draw a Gaussian scalar and normalise its magnitude, following the paper. Legacy `reservoir.m` always returns `+1` for this case. |
| Regression | Use SVD least squares with feature rows scaled by their maximum absolute value, then undo that scaling on coefficients. No inverse or normal equations. This preserves the feature span; in rank-deficient systems it chooses a scaled minimum-norm solution that can differ from Octave's right division. Scaling is tested on modes separated by 100 orders of magnitude. |
| Rank cut-off | `rcond=None` uses NumPy's documented default, machine epsilon times the larger design dimension. An explicit non-negative `rcond` is available and recorded by experiments. Changing it can change the learned spectrum. |
| Diagnostics | Expose rank, singular values, feature scales, scaled condition number and residual norm. Rank deficiency or scaled condition number above `1/sqrt(eps)` emits `NumericalConditionWarning`; a fitted model is still available for inspection. Non-finite recurrences raise errors. |
| Start vector | Use the balanced vector, following `start.m`; Figure 5 instead writes an all-ones vector for decomposed dynamics. This common scalar is absorbed by the fitted map. |
| Clustering | Preserve total multiplicity and explicitly match conjugate clusters. Reject negative tolerances instead of activating Octave's implicit `pi/N` heuristic. Ordering is by magnitude then phase. |
| Reduction threshold | Use inclusive `<=`, matching the API's maximum error contract. Legacy tests a strict `<`. |
| Final candidate | Explicitly evaluate and refit the selected prefix, including the one-component case. Legacy's loop can finish without assigning the final boundary candidate. |
| Retries | Fit one reservoir per explicit seed. Legacy `predict.m` repeats generation until the unreduced decomposed error is below its threshold. The experiment records every trial and failure; no unbounded retry is hidden in the estimator. |
| Randomness | NumPy `Generator`, without global RNG state. Equal integer seeds reproduce the same fit; supplied Generators advance. Octave's random numbers are not expected to match. |

## Benchmark conventions

The fixed MSO signal is `sum(sin(alpha*t))`, with angular frequencies
`[0.200, 0.311, 0.420, 0.510, 0.630, 0.740, 0.850, 0.970]` and `t=1,2,...`.
There is no rescaling by `2*pi`, centring, normalisation or observation noise.
The metric is `sqrt(mean_t(sum_d(error**2)))`, matching the paper's norm and
`rmse.m`'s default options. It averages over time and sums over dimensions.

The sources use different experimental settings:

- Figure 7 and `mso8/script.m` study training lengths 130, 150 and 200 and
  multiple reservoir sizes. The threshold is 0.5 and clustering is disabled.
- Section 5.1 describes generalised random-frequency examples as 250 training
  and 50 test samples, but also mentions validation selection without specifying
  its boundary there.
- The executable `mso8/octave/compute_prediction.m` uses 200 training, 50
  validation and 50 test samples, 100 trials, threshold 0.5 and default reservoir
  size `T-2=198` for a univariate series.

The default saved experiment applies that executable forecasting configuration
to the original fixed frequencies, which `read_in_data.m` includes alongside
the random-frequency examples. It selects the lowest validation RMSE from
the declared seeds and forecasts the test segment from the same training fit,
without refitting on validation data. All trial errors and failures are saved.

The `--profile paper` option uses 150 training samples and 70 reservoir units.
Its 50 validation and 100 test samples are an explicit extension to the
paper's compactness study, not a claim that Figure 7 used this selection split.
The saved evidence must be interpreted for its stated profile and settings.

## Verification limits

Tests check the hand-evaluated legacy recurrences, Jordan block entries,
multivariate RMSE, omission refitting and whole-pair retention, plus end-to-end
Fibonacci, sinusoid, parabola, multivariate and MSO cases. They also exercise
invalid input, rank diagnostics, seeded fits, impossible tolerances and numerical
non-monotonicity. Scientific regression cases use declared seeds and are not
estimates of success rates. Repeated MSO results are provided separately.

The cross-language audit now runs the pinned Octave routines. It shares
training samples and reservoir weights with Python, and separately runs 100
native Octave trials with the unmodified `predict.m`, including its retries
and boundary behaviour. Equal integer seeds do not produce equal random
weights across languages. The shared-weight comparison isolates this issue.
See [the numerical audit](ACCURACY.md) for measurements.

The audit measures the recorded configurations of the current Python and
pinned Octave implementations. It does not establish every fitting operation
used in the historical experiments. It does not cover the entire Figure 7
reservoir-size sweep,
the random-frequency table, other baselines or other application domains.

## Optional autonomous spectral refinement

`LRNN.reduce(..., refine=True, max_nfev=200)` first performs the same omission
ranking and dimension reduction as the default procedure. It then minimises

```text
min_phi ||S - A(phi) @ Y(phi)||_F
A(phi) = argmin_A ||S - A @ Y(phi)||_F.
```

The retained real eigenvalues and real/positive-imaginary parts of retained
complex pairs are optimised against training observations only. Each objective
evaluation refits the linear output map with the same equilibrated SVD solve.
Block dimension, ordering, conjugacy and Jordan superdiagonals remain fixed.
No known signal frequencies or unit-circle constraints are supplied.

SciPy's trust-region least-squares solver uses finite-difference derivatives,
Jacobian column scaling, tolerances `1e-12` and an explicit evaluation budget.
The best finite training fit is retained. Non-finite proposals are rejected
with a finite penalty. `reduced.refinement` records the initial fit, whether
an improvement was accepted, optimiser status, convergence and evaluation
counts. Budget exhaustion is visible. The complete state map is refitted
against the final refined dynamics; `reduced.reduction` preserves the initial
omission ranking, prefix errors and fit.

The default is `refine=False`. The additional fitting option is assessed by
matched comparisons in the [accuracy guide](ACCURACY.md). Its effect in the
recorded experiments does not establish the complete fitting procedure used
historically, and the fitting operation itself is not claimed as a new
contribution. Training error and optimiser convergence do not establish
forecast stability.

## Extraction and dependencies

The public import is `lrnn`. All deterministic numerical modules are local to
this directory. The core depends on NumPy/SciPy only; Matplotlib is an optional
experiment dependency. Robot-specific predictors and simulators are excluded.
The [verification record](VERIFICATION.md) documents checks of this extraction.
