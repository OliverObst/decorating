# LRNN research website

A buildless scientific overview of the published LRNN paper. All page resources
are local; no framework, CDN, analytics or package installation is needed.
The site uses the repository's BSD 3-Clause licence.

## Preview

From the repository root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8000/>. Stop the server with Ctrl-C.
The page also opens directly from `docs/index.html`; the clipboard button is
available only when the browser permits clipboard access.

## GitHub Pages

In this repository's **Settings → Pages**, choose **Deploy from a branch**,
then **main** and **/docs**. GitHub will serve the page at
<https://oliverobst.github.io/decorating/>. These instructions describe the
hosting setup; adding the files does not enable Pages by itself.

The site uses relative asset paths so it works both at the repository URL and
on a local server. `.nojekyll` prevents template processing.

## Content and figures

- `index.html`: overview, algorithm, published evidence, reproduction and citation.
- `assets/site.css`: responsive styles and keyboard focus states.
- `assets/reduction.js`: block selection, matrix displays, approximation and
  residual plots, with training and future RMSE.
- `assets/reduction-data.js`: all 16 subsets of a constructed four-block recurrence,
  including training-only SVD refits and unchanged-coefficient comparisons.
- `assets/site.js`: a constructed two-state rotation example and citation copying.
  The browser example explains the real block convention; it does not fit a model.
- `assets/reduction.svg` and `assets/reduction-mobile.svg`: original schematics,
  with illustrative matrix entries.
- `assets/mso-paper.png`: unchanged copy of `paper/fig/mso8_prediction_1.png`.
- `assets/soccer-paper.png`: Poppler rendering of `paper/fig/game6.pdf`.
- `assets/mso-python.svg`: plot of the verified, validation-selected forecast in
  `python/experiments/verified/mso_paper_refined/forecast.csv`.

Rebuild the figures from the repository root:

```bash
python3 docs/tools/build_figures.py
```

The generator uses only Python's standard library. Poppler's `pdftoppm` is
optional when the committed soccer image is already available.

Published-paper claims and the later Python reproduction are distinguished on
the page. The latter links to the accuracy guide and preserves the reported
failed trial. The paper's affiliations, journal reference and DOI are used.

The layout is inspired by QF3, FPO++, PyRoki, LERF, TILTED and EgoAllo, with
links in the footer. No template code or assets were copied from those sites.

## Interactive reduction example

The demonstration starts from known block dynamics, rather than fitting a random
reservoir. Three blocks sustain oscillations; one decays. A seeded orthogonal
change of coordinates produces an equivalent dense transition matrix. Whole
conjugate pairs are retained or omitted together. Empty selection gives the zero
approximation.

For each of the 16 subsets, the generator calls the public library's real-block
construction, autonomous rollout and equilibrated SVD output-map fit. The fit
uses samples 0–119 only. The later 120 samples measure continuation error. A
second output keeps the original coefficients, allowing a comparison with
refitting. All predictions and errors are precomputed; the browser selects and
plots the requested subset without a numerical dependency.

The omission scores are training RMSE after removing one block from the full
model and refitting. The removal button uses this fixed ranking throughout;
checkboxes and the block-matrix buttons allow arbitrary subsets. It does not
perform binary search or claim an optimal subset.

To regenerate and verify the data:

```bash
cd python
uv run python ../docs/tools/build_reduction_demo.py
```

Verification checks dense/block trajectory equivalence, every subset's residual
and interval RMSE, least-squares optimality, unchanged-coefficient comparisons,
the full model and the zero-dimensional case. The constructed example is kept
separate from the paper's measurements and the archived MSO reproduction.
