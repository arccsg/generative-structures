"""Shared style for v10 main-text figures (print-first: grayscale- and
CVD-safe, one axis per panel, recessive grids, direct labels).

Palette: Okabe-Ito subset, validated (lightness band, chroma floor, adjacent
CVD separation) for the light surface; identity is never color-alone -- every
series also differs in marker or linestyle.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GEO = "/Users/<ANON>/Projects/lprofile-geography"
TAB = os.path.join(GEO, "tables")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures_v10")
os.makedirs(OUT, exist_ok=True)

# fixed categorical order (entity -> color, never rank):
BLUE = "#0072B2"      # recorded / enumerated / observational
VERM = "#D55E00"      # the alarm series (CVAP) / exact multiplicative
GREEN = "#009E73"     # magnitude-conditioned null / reference
PINK = "#CC79A7"      # grid-rounded demonstrations
ORANGE = "#E69F00"    # additive controls / secondary
GRAY = "#5A5A5A"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.6,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
})

FULL = 6.3   # \textwidth in inches for 12pt article, 1in margins


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    print(f"wrote figures_v10/{name}.pdf/.png")
