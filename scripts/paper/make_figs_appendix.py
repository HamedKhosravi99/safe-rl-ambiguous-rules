"""Render the six appendix trend figures with matplotlib and save them as vector PDFs.

Each figure replaces a small or medium trend table whose content is a curve,
not a set of exact readings.  The numbers are parsed out of the paper's own
``generated/*.tex`` data files, so a figure cannot drift from the table it
replaces: if a re-run changes a generated file, the figure changes with it.

Typography is matched to the paper, as in make_figs_mpl.py.

Run: python3 scripts/paper/make_figs_appendix.py
Writes figure/fig_c1_scale.pdf, fig_d1_query.pdf, fig_f1_bins.pdf,
       fig_f2_recall.pdf, fig_e1_surrogate.pdf, fig_e2_tail.pdf
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


# ---------------------------------------------------------------- C1: scale
def fig_c1():
    r = rows("gen_scale_table.tex")
    K = np.array([num(x[0]) for x in r])
    set_lp = np.array([num(x[2]) for x in r])
    single = np.array([num(x[3]) for x in r])
    face = np.array([num(x[6]) for x in r])
    assert len(K) == 4 and K[-1] > 18000, r
    assert set_lp.max() < 0.1 and single.max() > 5, "the set LP must stay flat while the sweep grows"

    fig, ax = plt.subplots(figsize=(7.4 * CM, 4.4 * CM))
    ax.plot(K, single, "o-", color=CRULE, label=r"$K$ single-reading LPs")
    ax.plot(K, face, "s-", color=CALT, label="face LPs")
    ax.plot(K, set_lp, "^-", color=CREAD, label=r"one $V_{\widehat U}$ LP")
    ax.set_xscale("log")
    ax.set_xlabel(r"candidate readings $K$")
    ax.set_ylabel("wall-clock (s)")
    ax.set_xticks(K)
    ax.set_xticklabels(["10", "100", "1,000", "18,150"])
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig_c1_scale.pdf")


# ---------------------------------------------------------------- D1: query
def fig_d1():
    r = rows("gen_query_table.tex")
    pops, rules = [], []
    for row in r:
        if row[0] not in pops:
            pops.append(row[0])
        if row[1] not in rules:
            rules.append(row[1])
    assert len(pops) == 3 and len(rules) == 3, (pops, rules)
    q = np.array([1, 2, 4])
    style = {"price-guided": ("o-", CREAD), "most balanced split": ("s--", CRULE),
             "random fixture": ("^:", CALT)}

    fig, axes = plt.subplots(1, 3, figsize=(13.6 * CM, 4.0 * CM), sharey=True)
    for ax, pop in zip(axes, pops):
        for rule in rules:
            row = next(x for x in r if x[0] == pop and x[1] == rule)
            y = [num(row[2]), num(row[3]), num(row[4])]
            m, c = style[rule]
            ax.plot(q, y, m, color=c, label=rule)
        ax.set_title(pop.replace(" (", "\n("), fontsize=7)
        ax.set_xlabel("questions $q$")
        ax.set_xticks(q)
        ax.set_ylim(0, 105)
        ax.grid(axis="y")
    axes[0].set_ylabel("return recovered (%)")
    axes[-1].legend(frameon=False, loc="lower right")
    save(fig, "fig_d1_query.pdf")


# ------------------------------------------------------------ F1: ARTEMIS bins
def fig_f1():
    p1, p2 = rows("gen_artemis_bins_P1.tex"), rows("gen_artemis_bins_P2.tex")
    labels = [x[0].replace("$>$", ">") for x in p1]
    assert [x[0] for x in p1] == [x[0] for x in p2], "the two pools must share bins"
    x = np.arange(len(labels))
    w = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(13.6 * CM, 4.2 * CM), sharey=True)
    for ax, tab, name in ((axes[0], p1, "P1 (flagship, 50 samples)"),
                          (axes[1], p2, "P2 (union, 190 samples)")):
        top1 = [num(r[2]) for r in tab]
        sc = [num(r[3]) for r in tab]
        arrow = [num(r[4]) for r in tab]
        assert all(a >= max(t, s) for a, t, s in zip(arrow, top1, sc)), name
        ax.bar(x - w, top1, w, color=CRULE, label="top-1")
        ax.bar(x, sc, w, color=CALT, label="self-consistency")
        ax.bar(x + w, arrow, w, color=CREAD, label="ARROW")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel(r"expert-plausible readings $|P_\ell|$")
        ax.set_title(name, fontsize=7)
        ax.set_ylim(0, 100)
        ax.grid(axis="y")
    axes[0].set_ylabel("recall (%)")
    axes[0].legend(frameon=False, loc="upper right")
    save(fig, "fig_f1_bins.pdf")


# --------------------------------------------------------- F2: ARTEMIS recall
def fig_f2():
    r = rows("gen_artemis_recall_table.tex")
    labels = [x[0] for x in r]
    xs = np.arange(len(labels))
    assert labels[-1] == "all", labels

    fig, ax = plt.subplots(figsize=(7.6 * CM, 4.4 * CM))
    ax.plot(xs, [num(x[2]) for x in r], "o-", color=CREAD, label="P1, any")
    ax.plot(xs, [num(x[4]) for x in r], "s-", color=CSAFE, label="P2, any")
    ax.plot(xs, [num(x[1]) for x in r], "o--", color=CRULE, label="P1, class recall")
    ax.plot(xs, [num(x[3]) for x in r], "s--", color=CALT, label="P2, class recall")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("candidate budget $K$")
    ax.set_ylabel("coverage (%)")
    ax.set_ylim(0, 90)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    save(fig, "fig_f2_recall.pdf")


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


# --------------------------------------------------------------- E2: tail
def fig_e2():
    r = rows("gen_sweep_tail.tex")
    doms = []
    for row in r:
        if row[0] not in doms:
            doms.append(row[0])
    assert len(doms) == 2, doms

    fig, axes = plt.subplots(1, 2, figsize=(13.0 * CM, 4.2 * CM))
    for ax, dom in zip(axes, doms):
        sub = [x for x in r if x[0] == dom]
        b = np.array([num(x[1]) for x in sub])
        single = np.array([num(x[2]) for x in sub])
        arrow = np.array([num(x[3]) for x in sub])
        assert np.all(arrow < single), dom
        ax.plot(b, single, "o-", color=CRULE, label="single reading")
        ax.plot(b, arrow, "s-", color=CREAD, label="ARROW")
        ax.set_yscale("log")
        ax.set_xticks(b)
        ax.set_xlabel(r"tail level $\beta$")
        ax.set_title(dom.replace("\\ ", " "), fontsize=7)
        ax.grid(axis="y")
    axes[0].set_ylabel(r"$\mathrm{CVaR}_\beta$ of realized cost")
    axes[0].legend(frameon=False, loc="lower left")
    save(fig, "fig_e2_tail.pdf")


if __name__ == "__main__":
    fig_c1(); fig_d1(); fig_f1(); fig_f2(); fig_e1(); fig_e2()
