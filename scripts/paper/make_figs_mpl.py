"""Render the paper's data figures with matplotlib and save them as vector PDFs.

Replaces the hand-emitted TikZ for the three figures whose content is data:
Figure 1(A) clarification curves, Figure 1(B) first-question return recovery,
and the coverage funnel in the appendix.  The archives and the assertions are
the same ones the TikZ generators used, so the plotted values cannot drift.

Typography is matched to the paper: a Times-like serif at body-figure size, so
the labels read as part of the document rather than as pasted-in artwork.

Run: python3 scripts/paper/make_figs_mpl.py
Writes figure/fig_e3_questions.pdf, fig_e3_panelb.pdf, fig_e6_funnel.pdf
"""
from __future__ import annotations

import json
import os
import re

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
    ax.plot(q, hec, color=CREAD, marker="o", label="safe stop, decision-conflict selector")
    ax.plot(q, poa, color=CSAFE, marker="s", ls="--", label="safe stop, return-sensitive selector")
    ax.plot(q, ident, color=CRULE, marker="^", ls=":", label="identified, balanced split")
    ax.set_xlabel("number of truthful questions")
    ax.set_ylabel("share of pools")
    ax.set_xticks(q)
    ax.set_ylim(0, 1.30)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    # the curves fill the lower right; the legend goes where the data is not
    ax.legend(frameon=False, loc="upper left", handlelength=1.8, borderpad=0.2,
              labelspacing=0.25, bbox_to_anchor=(-0.02, 1.04))
    save(fig, "fig_e3_questions.pdf")
    return [round(v, 3) for v in hec]


# ------------------------------------- 1(B) what the first question recovers
def panel_b():
    src = os.path.join(PAPER, "generated", "gen_query_table.tex")
    vals = {}
    for line in open(src):
        m = re.match(r"\s*(.+?)\s*\((\d+)\)\s*&\s*(.+?)\s*&\s*(\d+)\\%", line)
        if m:
            vals[(m.group(1), m.group(3))] = (int(m.group(4)), int(m.group(2)))
    pops = [("Compiled monitoring", "duration-based\nmonitoring"),
            ("Admission", "admission\ncontrol"),
            ("Monitoring, free class", "flexible\nmonitoring")]
    rules = [("price-guided", CREAD), ("most balanced split", CSAFE),
             ("random fixture", CRULE)]
    for pop, _ in pops:
        for rule, _c in rules:
            assert (pop, rule) in vals, (pop, rule)
        assert vals[(pop, "price-guided")][0] >= vals[(pop, "most balanced split")][0], pop

    fig, ax = plt.subplots(figsize=(6.9 * CM, 3.15 * CM))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    w = 0.26
    x = np.arange(len(pops))
    for j, (rule, col) in enumerate(rules):
        h = [vals[(p, rule)][0] for p, _ in pops]
        shown = {"price-guided": "return-sensitive", "most balanced split": "balanced split",
                 "random fixture": "random question"}[rule]
        b = ax.bar(x + (j - 1) * w, h, w * 0.92, color=col, edgecolor=col, label=shown)
        ax.bar_label(b, fmt="%d", padding=1, fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n($n{{=}}{vals[(p,'price-guided')][1]}$)" for p, lab in pops])
    ax.set_ylabel("share of oracle gap recovered")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.legend(frameon=False, loc="upper center", ncol=3, handlelength=1.1,
              columnspacing=0.9, borderpad=0.1, bbox_to_anchor=(0.5, 1.20))
    save(fig, "fig_e3_panelb.pdf")
    return {p: vals[(p, "price-guided")][0] for p, _ in pops}


# --------------------------------------------------- appendix coverage funnel
def funnel():
    cf = json.load(open(os.path.join(R, "coverage_funnel.json")))["unit_level"]
    ST = ("raw", "unique", "non_template", "in_grammar", "faithful", "proposed", "retained")
    LAB = ["raw\nunits", "distinct\nrules", "non-\ntemplate", "in the\ngrammar",
           "faithful\ntarget", "reading\nproposed", "reading\nretained"]
    series = [("ALL", "all repositories", CREAD, "o", 1.4),
              ("tidb", "tidb", CSAFE, "s", 1.0),
              ("gitlab-runbooks", "gitlab-runbooks", CRULE, "^", 1.0)]
    fig, ax = plt.subplots(figsize=(8.0 * CM, 4.2 * CM))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    for repo, lab, col, mk, lw in series:
        v = [cf[repo][s]["unconditional"] for s in ST]
        assert all(v[i] >= v[i + 1] - 1e-12 for i in range(len(v) - 1)), (repo, v)
        ax.plot(range(len(ST)), [max(x, 1e-4) for x in v], color=col, marker=mk,
                lw=lw, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ST)))
    ax.set_xticklabels(LAB)
    ax.set_ylabel("share of raw units (log)")
    ax.legend(frameon=False, loc="lower left", handlelength=1.8, borderpad=0.2)
    save(fig, "fig_e6_funnel.pdf")


if __name__ == "__main__":
    print("panel A hec:", panel_a())
    print("panel B price-guided:", panel_b())
    funnel()
