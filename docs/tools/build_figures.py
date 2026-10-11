"""Rebuild the website diagrams from fixed layouts and recorded MSO data.

Run from the repository root: python3 docs/tools/build_figures.py
No third-party Python packages are required. Poppler renders the soccer PDF.
"""

import csv
from html import escape
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"


def svg(width, height, title, content):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img">'
        f'<title>{escape(title)}</title>'
        '<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#1e2939}'
        '.label{font-size:17px;font-weight:600}.small{font-size:13px;fill:#586578}</style>'
        + content + "</svg>\n"
    )


def matrix(x, y, n, cell, block=False, retained=None):
    content = []
    for row in range(n):
        for col in range(n):
            on_block = row // 2 == col // 2
            active = not block or on_block
            colour = "#edf1f7"
            if active:
                colour = "#1859a6"
                if not block:
                    # Illustrative dense connectivity, not a measured fitted matrix.
                    opacity = 0.18 + ((row * 11 + col * 7 + 3) % 17) / 23
                    content.append(f'<rect x="{x+col*cell}" y="{y+row*cell}" '
                                   f'width="{cell-2}" height="{cell-2}" '
                                   f'rx="1" fill="{colour}" opacity="{opacity:.2f}"/>')
                    continue
                if retained is not None and row // 2 not in retained:
                    colour = "#d9e2ef"
            content.append(f'<rect x="{x+col*cell}" y="{y+row*cell}" '
                           f'width="{cell-2}" height="{cell-2}" rx="1" fill="{colour}"/>')
    return "".join(content)


def reduction_figure():
    parts = ['<rect x="1" y="1" width="1078" height="302" rx="12" fill="#f5f7fb"/>']
    for centre, label, subtitle in [(185, "Fitted recurrence", "Many connected state values"),
                                   (540, "Spectral components", "Growth, decay and oscillation"),
                                   (895, "Reduced recurrence", "Keep whole relevant blocks")]:
        parts.append(f'<text x="{centre}" y="42" text-anchor="middle" class="label">{label}</text>')
        parts.append(f'<text x="{centre}" y="65" text-anchor="middle" class="small">{subtitle}</text>')
    parts.append(matrix(101, 90, 8, 21))
    parts.append(matrix(456, 90, 8, 21, block=True, retained={0, 2}))
    parts.append(matrix(841, 121, 4, 27, block=True))
    parts.append('<path d="M314 169H405M669 169H777" fill="none" stroke="#92a4bd" stroke-width="2"/>')
    parts.append('<path d="M397 163L405 169L397 175M769 163L777 169L769 175" fill="none" stroke="#92a4bd" stroke-width="2"/>')
    for centre, label in [(185, "x(t + 1) = W x(t)"), (540, "Represent and rank blocks"),
                          (895, "z(t + 1) = Jᵣ z(t)")]:
        parts.append(f'<text x="{centre}" y="281" text-anchor="middle" class="small">{label}</text>')
    (ASSETS / "reduction.svg").write_text(svg(1080, 304, "Spectral reduction of an LRNN (schematic)", "".join(parts)))
    mobile = ['<rect x="1" y="1" width="358" height="708" rx="12" fill="#f5f7fb"/>']
    labels = [("Fitted recurrence", "Many connected state values", "x(t + 1) = W x(t)"),
              ("Spectral components", "Growth, decay and oscillation", "Represent and rank blocks"),
              ("Reduced recurrence", "Keep whole relevant blocks", "z(t + 1) = Jᵣ z(t)")]
    for i, (label, subtitle, equation) in enumerate(labels):
        top = i * 230
        mobile.append(f'<text x="180" y="{top+32}" text-anchor="middle" class="label">{label}</text>')
        mobile.append(f'<text x="180" y="{top+54}" text-anchor="middle" class="small">{subtitle}</text>')
        if i < 2:
            mobile.append(matrix(116, top+70, 8, 16, block=i == 1, retained={0, 2} if i == 1 else None))
        else:
            mobile.append(matrix(126, top+86, 4, 27, block=True))
        mobile.append(f'<text x="180" y="{top+217}" text-anchor="middle" class="small">{equation}</text>')
    (ASSETS / "reduction-mobile.svg").write_text(svg(360, 710, "Spectral reduction of an LRNN (schematic)", "".join(mobile)))


def mso_figure():
    source = ROOT / "python" / "experiments" / "verified" / "mso_paper_refined" / "forecast.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 300
    x = lambda t: 60 + (int(t) - 1) * 950 / 299
    y = lambda value: 153 - float(value) * 13
    parts = ['<rect width="1040" height="310" fill="#fff"/>']
    for lo, hi, colour, label in [(1, 150, "#f2f5fa", "Training · 150 samples"),
                                  (151, 200, "#eaf1f9", "Validation · 50"),
                                  (201, 300, "#f6f3ed", "Test · 100")]:
        left = x(lo) if lo == 1 else (x(lo) + x(lo - 1)) / 2
        right = x(hi) if hi == 300 else (x(hi) + x(hi + 1)) / 2
        parts.append(f'<rect x="{left:.2f}" y="46" width="{right-left:.2f}" height="215" fill="{colour}"/>')
        parts.append(f'<text x="{(left+right)/2:.2f}" y="28" text-anchor="middle" class="small">{label}</text>')
    for value in [-6, -3, 0, 3, 6]:
        yy = y(value)
        parts.append(f'<path d="M60 {yy}H1010" stroke="#dce3ec" stroke-width="1"/>')
        parts.append(f'<text x="48" y="{yy+4}" text-anchor="end" class="small">{value}</text>')
    for column, colour, dash in [("observed", "#8b98a9", ""), ("forecast", "#1859a6", 'stroke-dasharray="5 3"')]:
        path = " ".join(f'{"M" if i == 0 else "L"}{x(row["time"]):.2f},{y(row[column]):.2f}'
                        for i, row in enumerate(rows))
        parts.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2" {dash}/>')
    for time in [1, 50, 100, 150, 200, 250, 300]:
        parts.append(f'<text x="{x(time):.2f}" y="281" text-anchor="middle" class="small">{time}</text>')
    parts.extend(['<text x="535" y="304" text-anchor="middle" class="small">Time step</text>',
                  '<path d="M731 297H759" stroke="#8b98a9" stroke-width="2"/><text x="766" y="302" class="small">Observed</text>',
                  '<path d="M869 297H897" stroke="#1859a6" stroke-width="2" stroke-dasharray="5 3"/><text x="904" y="302" class="small">LRNN</text>'])
    (ASSETS / "mso-python.svg").write_text(svg(1040, 310, "Recorded MSO forecast with spectral refinement", "".join(parts)))


def paper_figures():
    shutil.copyfile(ROOT / "paper" / "fig" / "mso8_prediction_1.png", ASSETS / "mso-paper.png")
    renderer = shutil.which("pdftoppm")
    if renderer is None:
        print("pdftoppm is unavailable; keeping the committed soccer-paper.png.")
        return
    subprocess.run([renderer, "-singlefile", "-scale-to", "1500", "-png",
                    str(ROOT / "paper" / "fig" / "game6.pdf"),
                    str(ASSETS / "soccer-paper")], check=True)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    reduction_figure()
    mso_figure()
    paper_figures()
    print("Website figures rebuilt in docs/assets/.")
