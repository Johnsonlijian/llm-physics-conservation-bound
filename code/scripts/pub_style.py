"""Shared publication style for all manuscript figures (main + supplementary).

Nature-family conventions enforced here:
  - white background, no in-figure banner titles (conclusions live in captions)
  - thin axes (0.6 pt), no top/right spines, no decorative grids by default
  - Arial/Helvetica sans-serif, 7-8 pt panel text, 8 pt axis labels
  - Okabe-Ito colourblind-safe palette
  - vector-first export (PDF with TrueType fonts) + 600 dpi PNG derivative
  - figures designed at print size: full width 7.05 in (~180 mm),
    single column 3.46 in (~88 mm)

Usage:
    from pub_style import apply_style, panel_label, save_figure, OKABE, FULL_W
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

# print-size figure widths (inches)
FULL_W = 7.05    # double column, ~180 mm
HALF_W = 3.46    # single column, ~88 mm

# Okabe-Ito colourblind-safe palette
OKABE = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
}
# neutral greys for context/secondary series
GREY_DARK = "#4d4d4d"
GREY_MID = "#878787"
GREY_LIGHT = "#bababa"
INK = "#1a1a1a"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.5,
        "legend.title_fontsize": 7.0,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.6,
        "errorbar.capsize": 2.0,
        "axes.unicode_minus": False,
    })


def panel_label(ax, s: str, dx: float = -0.14, dy: float = 1.02) -> None:
    """Bold lowercase panel letter at the top-left, Nature style."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def save_figure(fig, out_png: Path) -> None:
    """Export PNG (600 dpi) and PDF (vector) side by side."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
