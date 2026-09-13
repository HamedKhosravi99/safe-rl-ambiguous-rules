"""figure/fig_v53_scales.pdf: three data scales per sufficient-reading class of the monitoring suite (V53) --
the smallest reliably certifying log size n* of the generic uniform certificate, the occupancy-weighted certificate and the
robust-dual certificate, and the information floor kl(1-delta,delta)/I* -- against 1/kappa^2.  Every point is read from
results/e2e/certificate_v53.json; censored n* (not reached within the grid) are drawn as open markers at the grid maximum."""
import json, os, math
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER = os.environ.get("ARROW_PAPER_DIR", os.path.join(ROOT, "paper")); FIG = os.path.join(PAPER, "figure")
r = json.load(open(os.path.join(ROOT, "results", "theory_extension", "certificate_v53.json"))); grid = r["meta"]["n_grid"]
def key(x): return f"{x['uid']}|{x['k']}"
def nstar(d, members, c):
    for n in grid:
        rows = [x for x in r["primary"] if x["d"] == d and x["n"] == n and key(x) in members]
        if rows and np.mean([x[c]["pass"] for x in rows]) >= 0.9: return n
    return None
pts = []
for d, cl in r["classes"].items():
    for cname, members in cl.items():
        kappa = float(cname.split("=")[1]); I = r["istar"][d][cname]
        pts.append(dict(d=float(d), kappa=kappa, size=len(members), n_info=I["n_info"], **{c: nstar(float(d), set(members), c) for c in ("UNIFORM", "OCC", "DUAL")}))
assert len(pts) == 6
CM = 1 / 2.54
fig, ax = plt.subplots(figsize=(9.0 * CM, 7.6 * CM))
x = np.array([1 / p["kappa"] ** 2 for p in pts]); nmax = grid[-1]
def series(c, marker, color, label):
    y = np.array([p[c] if p[c] else np.nan for p in pts]); cens = np.array([p[c] is None for p in pts])
    ax.scatter(x[~cens], y[~cens], marker=marker, s=34, color=color, label=label, zorder=3)
    if cens.any(): ax.scatter(x[cens], np.full(cens.sum(), nmax), marker=marker, s=34, facecolors="none", edgecolors=color, zorder=3)
series("UNIFORM", "^", "black", "uniform certificate (generic)")
series("OCC", "D", "C1", "occupancy-weighted")
series("DUAL", "s", "C0", "robust dual (\\textsc{Arrow})" if False else "robust dual (ARROW)")
ax.scatter(x, [p["n_info"] for p in pts], marker="o", s=34, color="C2", label="information floor kl(1$-\\delta$,$\\delta$)/$I^\\star$", zorder=3)
# slope-1 reference through the geometric centre of the robust-dual points
xd = x; yd = np.array([p["DUAL"] for p in pts], float)
c0 = np.exp(np.mean(np.log(yd) - np.log(xd))); xs = np.array([x.min() / 2, x.max() * 2])
ax.plot(xs, c0 * xs, "--", color="gray", linewidth=0.8, label="slope 1 ($n\\propto\\kappa^{-2}$)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$1/\\kappa^2$ (exact cost margin)"); ax.set_ylabel("pooled load transitions")
ax.set_ylim(20, 3 * nmax); ax.grid(alpha=0.3, which="both", linewidth=0.4)
ax.text(x.max() * 1.15, nmax, "$>10^8$", fontsize=6, va="center")
ax.legend(fontsize=5.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
fig.tight_layout(); os.makedirs(FIG, exist_ok=True); fig.savefig(os.path.join(FIG, "fig_v53_scales.pdf"), bbox_inches="tight"); print("wrote fig_v53_scales.pdf", pts)
