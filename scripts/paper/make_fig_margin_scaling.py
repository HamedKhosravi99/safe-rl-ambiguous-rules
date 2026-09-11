#!/usr/bin/env python3
"""Emit fig_margin_scaling.tex: smallest certifying telemetry size against 1/gamma^2,
from results/theory_extension/real_rules_finite_data.json (produced by real_rules_experiment.py)."""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "results", "theory_extension", "real_rules_finite_data.json")
OUT = os.path.join(os.environ.get("ARROW_PAPER_DIR", os.path.join(ROOT, "paper")), "figure", "fig_margin_scaling.tex")
fd = json.load(open(SRC))
pts = {}
for key in ("d=0.02|bernstein", "d=0.05|bernstein", "d=0.1|bernstein"):
    d = key.split("|")[0][2:]
    for v in fd[key]["n_star"].values():
        pts[(d, round(v["margin"], 4))] = v["n_star"]
assert len(pts) == 6, len(pts)
W, H = 6.2, 4.0
X = lambda g: (math.log10(1 / g ** 2) - 2.0) / 4.0 * W
Y = lambda n: (math.log10(n) - 6.5) / 4.0 * H
L = [r"\begin{tikzpicture}[x=1cm,y=1cm,font=\scriptsize]",
     r"\draw[->] (0,0) -- (%.2f,0);" % (W + 0.3),
     r"\node at (%.2f,-0.55) {ambiguity margin $1/\kappa^2$};" % (W / 2),
     r"\draw[->] (0,0) -- (0,%.2f);" % (H + 0.3),
     r"\node[rotate=90] at (-0.9,%.2f) {pooled transitions $n^\star$};" % (H / 2)]
for e in (2, 3, 4, 5, 6):
    L.append(r"\draw (%.2f,0.06) -- (%.2f,-0.06) node[below] {$10^{%d}$};" % (X(10 ** (-e / 2)), X(10 ** (-e / 2)), e))
for e in (7, 8, 9, 10):
    L.append(r"\draw (0.06,%.2f) -- (-0.06,%.2f) node[left] {$10^{%d}$};" % (Y(10 ** e), Y(10 ** e), e))
c = math.exp(sum(math.log(n * g * g) for (_, g), n in pts.items()) / len(pts))
x0, x1 = 2.2, 5.6
L.append(r"\draw[dashed] (%.2f,%.2f) -- (%.2f,%.2f);" % ((x0 - 2) / 4 * W, (math.log10(c) + x0 - 6.5) / 4 * H,
                                                         (x1 - 2) / 4 * W, (math.log10(c) + x1 - 6.5) / 4 * H))
L.append(r"\node[anchor=west] at (%.2f,%.2f) {slope $1$ ($n^\star\kappa^2=%.1f\times10^{4}$)};"
         % ((x1 - 2) / 4 * W - 2.6, (math.log10(c) + x1 - 6.5) / 4 * H + 0.35, c / 1e4))
marks = {"0.02": "square*", "0.05": "*", "0.1": "triangle*"}
for (d, g), n in sorted(pts.items()):
    L.append(r"\node[mark size=2pt] at (%.2f,%.2f) {\pgfuseplotmark{%s}};" % (X(g), Y(n), marks[d]))
L.append(r"\node[anchor=north west,align=left] at (0.15,%.2f) {$\blacksquare$ $d{=}0.02$\quad $\bullet$ $d{=}0.05$\quad $\blacktriangle$ $d{=}0.10$};" % (H - 0.05))
L.append(r"\end{tikzpicture}")
open(OUT, "w").write("\n".join(L) + "\n")
print("wrote", OUT, "| n* gamma^2 =", round(c))
