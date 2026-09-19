"""Render Figure 3 (the exact price of the single-cost surrogate by budget) with matplotlib as a
vector PDF.  The numbers are parsed out of the paper's own ``generated/gen_v48_surrogate.tex``
(make_gen_v47_51_tables.py), so the figure cannot drift from the archive it plots.

Typography is matched to the paper, as in make_figs_mpl.py.

Run: python3 scripts/paper/make_figs_appendix.py
Writes figure/fig_e1_surrogate.pdf
"""
from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = os.environ.get("ARROW_PAPER_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper"))
GEN = os.path.join(PAPER, "generated")
FIG = os.path.join(PAPER, "figure")

CRULE, CREAD, CSAFE = "#334155", "#2563EB", "#059669"   # the paper's palette
CALT = "#B45309"
CM = 1 / 2.54

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.linewidth": 0.4, "grid.color": "#CBD5E1",
    "lines.linewidth": 1.1, "lines.markersize": 3.4,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})


def rows(name):
    """Parse a generated table body into a list of cell lists."""
    out = []
    for line in open(os.path.join(GEN, name)):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        line = line.rstrip("\\").strip()
        out.append([c.strip() for c in line.split("&")])
    return out


def num(cell):
    """Strip LaTeX decoration from a numeric cell."""
    c = cell.replace("$", "").replace("\\%", "").replace("%", "")
    c = c.replace("{,}", "").replace(",", "").strip()
    return float(c)


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


# ----------------------------------------------------------- E1: surrogate
def fig_e1():
    r = rows("gen_v48_surrogate.tex")
    d = np.array([num(x[0]) for x in r])
    med = np.array([num(x[2]) for x in r])
    mean = np.array([num(x[3]) for x in r])
    p90 = np.array([num(x[4]) for x in r])
    mx = np.array([num(x[5]) for x in r])
    assert np.all(np.diff(med) > 0) and np.all(mx >= p90), "price must rise with the budget"

    fig, ax = plt.subplots(figsize=(7.6 * CM, 4.4 * CM))
    ax.fill_between(d, med, mx, color=CREAD, alpha=0.12, linewidth=0)
    ax.plot(d, mx, "^-", color=CALT, label="max")
    ax.plot(d, p90, "s-", color=CSAFE, label="p90")
    ax.plot(d, mean, "d-", color=CRULE, label="mean")
    ax.plot(d, med, "o-", color=CREAD, label="median")
    ax.set_xscale("log")
    ax.set_xticks(d)
    ax.set_xticklabels([("%g" % v) for v in d])
    ax.set_xlabel("budget $d$")
    ax.set_ylabel("return given up (%)")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_e1_surrogate.pdf")




if __name__ == "__main__":
    fig_e1()
