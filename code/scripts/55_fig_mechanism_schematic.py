"""Figure 6: mechanism schematic -- the solution pipeline and the three probes.

This is one of the paper's two allowed schematic figures. Publication-style:
plain rectangles with thin outlines, one accent colour on the localized stage,
no numbered badges, no banner titles, no shouting capitals; the conclusion
sentence lives in the manuscript caption.

Output: figures/F24_mechanism_schematic.{png,pdf}
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

from pub_style import apply_style, save_figure, OKABE, FULL_W, \
    GREY_DARK, GREY_MID, INK

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "figures" / "F24_mechanism_schematic.png"

BLUE = OKABE["blue"]
RED = OKABE["vermillion"]
GREEN = OKABE["green"]


def stage_box(ax, x, y, w, h, title, color, emph=False):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=1.2 if emph else 0.7,
                 edgecolor=color, facecolor=("#fdf1ec" if emph else "white"),
                 zorder=3))
    ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
            fontsize=7.2, color=INK, fontweight="bold" if emph else "normal",
            zorder=6)


def probe(ax, x_center, y_box_bottom, result, verdict, color, strong=False):
    yb = y_box_bottom
    ax.add_patch(FancyArrowPatch((x_center, yb - 1.45), (x_center, yb - 0.06),
                 arrowstyle="-|>", mutation_scale=9,
                 lw=1.0 if strong else 0.7, color=color, zorder=4))
    ax.text(x_center, yb - 1.95, result, ha="center", va="center",
            fontsize=6.2, color=color, zorder=5,
            fontweight="bold" if strong else "normal")
    ax.text(x_center, yb - 2.95, verdict, ha="center", va="center",
            fontsize=6.2, color=(RED if strong else GREY_MID), zorder=5,
            fontweight="bold" if strong else "normal")


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_W, 2.75))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax.set_xlim(0, 29)
    ax.set_ylim(0.4, 12.0)
    ax.axis("off")

    yb, h = 6.6, 2.2
    # problem
    ax.add_patch(Rectangle((0.4, yb), 2.7, h, linewidth=0.7, edgecolor=GREY_DARK,
                 facecolor="#f4f4f4", zorder=3))
    ax.text(1.75, yb + h / 2, "physics\nproblem", ha="center", va="center",
            fontsize=6.8, color=INK)

    xs = [4.6, 11.1, 17.6]
    w = 5.2
    stage_box(ax, xs[0], yb, w, h, "identify the\nconservation laws", BLUE)
    stage_box(ax, xs[1], yb, w, h, "formulate the\nequations", RED, emph=True)
    stage_box(ax, xs[2], yb, w, h, "solve the\nsystem", GREEN)
    # answer
    ax.add_patch(Rectangle((23.6, yb), 2.6, h, linewidth=0.7, edgecolor=GREY_DARK,
                 facecolor="#f4f4f4", zorder=3))
    ax.text(24.9, yb + h / 2, "final\nanswer", ha="center", va="center",
            fontsize=6.8, color=INK)

    # flow arrows
    for (x0, x1) in [(3.1, 4.5), (xs[0] + w, xs[1]), (xs[1] + w, xs[2]),
                     (xs[2] + w, 23.5)]:
        ax.add_patch(FancyArrowPatch((x0, yb + h / 2), (x1, yb + h / 2),
                     arrowstyle="-|>", mutation_scale=9, lw=0.8, color=INK,
                     zorder=2))

    # probes under each stage (measured controlled interventions)
    probe(ax, xs[0] + w / 2, yb,
          "laws provided\n$\\rightarrow +0.02$ (null)",
          "identification:\nnot the bottleneck", BLUE)
    probe(ax, xs[1] + w / 2, yb,
          "formulated equations provided\n$\\rightarrow +0.13$ at 50% reveal "
          "($8\\times$ larger)",
          "formulation:\nthe localized bottleneck", RED, strong=True)
    probe(ax, xs[2] + w / 2, yb,
          "explicit constraints (synthetic)\n$\\rightarrow$ solver-capable "
          "$\\approx 1.0$",
          "solving:\nnot the bottleneck", GREEN)

    # d_c feeding into formulation (top)
    ax.add_patch(FancyArrowPatch((xs[1] + w / 2, 10.6), (xs[1] + w / 2, yb + h + 0.08),
                 arrowstyle="-|>", mutation_scale=9, lw=1.0, color=GREY_DARK,
                 zorder=4))
    ax.text(xs[1] + w / 2, 11.3,
            "each $+1$ conservation constraint = one more equation to set up "
            "correctly  $\\Rightarrow$  odds $\\times\\,0.69$",
            ha="center", va="center", fontsize=6.6, color=GREY_DARK, zorder=5)

    save_figure(fig, OUT)
    print(f"[fig6] wrote {OUT}")


if __name__ == "__main__":
    main()
