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
