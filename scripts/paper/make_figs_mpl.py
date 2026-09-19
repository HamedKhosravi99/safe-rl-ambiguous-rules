"""Render Figure 1 (clarification progress on the 87 service-monitoring candidate sets) with
matplotlib as a vector PDF, from results/e2e/safe_collapse.json.  The archive and the assertions
are the same ones the earlier TikZ generator used, so the plotted values cannot drift.

Typography is matched to the paper: a Times-like serif at body-figure size, so
the labels read as part of the document rather than as pasted-in artwork.

Run: python3 scripts/paper/make_figs_mpl.py
Writes figure/fig_e3_questions.pdf
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "results/e2e")
PAPER = os.environ.get("ARROW_PAPER_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper"))
FIG = os.path.join(PAPER, "figure")

CRULE, CREAD, CSAFE = "#334155", "#2563EB", "#059669"   # the paper's palette
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
    "lines.linewidth": 1.1, "lines.markersize": 3.2,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


# ------------------------------------------------- 1(A) clarification curves
def panel_a():
    sc = json.load(open(os.path.join(R, "safe_collapse.json")))["populations"]["monitoring_free"]["summary"]
    hec = sc["rules"]["hec"]["safe_by_q"]
    poa = sc["rules"]["poa"]["safe_by_q"]
    ident = sc["rules"]["split"]["identified_by_q"]
    assert len(hec) == 5 and abs(hec[0] - poa[0]) < 1e-9 and ident[0] == 0
    assert hec[4] > poa[4] > ident[4], (hec[4], poa[4], ident[4])

    q = np.arange(5)
    fig, ax = plt.subplots(figsize=(6.6 * CM, 3.15 * CM))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.plot(q, hec, color=CREAD, marker="o", label="hyperedge cutting, sufficient")
    ax.plot(q, poa, color=CSAFE, marker="s", ls="--", label="price-guided, sufficient")
    ax.plot(q, ident, color=CRULE, marker="^", ls=":", label="balanced split, identified")
    ax.set_xlabel("number of truthful questions")
    ax.set_ylabel("share of candidate sets")
    ax.set_xticks(q)
    ax.set_ylim(0, 1.30)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    # the curves fill the lower right; the legend goes where the data is not
    ax.legend(frameon=False, loc="upper left", handlelength=1.8, borderpad=0.2,
              labelspacing=0.25, bbox_to_anchor=(-0.02, 1.04))
    save(fig, "fig_e3_questions.pdf")
    return [round(v, 3) for v in hec]



if __name__ == "__main__":
    panel_a()
