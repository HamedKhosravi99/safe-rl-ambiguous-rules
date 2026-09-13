#!/usr/bin/env python3
"""Macros and table rows for the finite-data Decide experiment (results/theory_extension/real_rules_*.json)."""
import json, math, re, collections, os
from scipy.stats import beta as beta_dist
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
TE = os.path.join(ROOT, "results", "theory_extension") + "/"
GEN = os.path.join(ROOT, "paper", "generated") + "/"
rec = json.load(open(TE + "real_rules_finite_data_records.json"))
ex = json.load(open(TE + "real_rules_exact.json"))["exact"]
fd = json.load(open(TE + "real_rules_finite_data.json"))
EPS = "0.01"
def macro(name, val): return f"\\newcommand{{\\{name}}}{{{val}}}"
def fmt_n(n):
    e = int(round(math.log10(n))) if abs(math.log10(n) - round(math.log10(n))) < 1e-6 else None
    if e is not None: return f"$10^{{{e}}}$"
    m = n / 10 ** math.floor(math.log10(n)); return f"${m:.1f}\\times10^{{{int(math.floor(math.log10(n)))}}}$"
out = []
# truth at each budget
def truth(d, uid, k):
    x = ex[uid][str(d)][str(k)]; return x is not None and x["margin_eps"][EPS] >= -1e-9
budgets = [0.02, 0.05, 0.1]
# count per compiled instance (the suite has one duplicated uid with identical readings, so the exact record is shared)
suite = json.load(open(os.path.join(ROOT, "results", "e2e", "control_suite_uncapped.json")))
inst_uids = [f'{i["uid"]}#{j}' for j, i in enumerate(suite["instances"])]
assert len(inst_uids) == 28
suff = {d: sorted((u, k) for u in inst_uids for k in ("0", "1") if truth(d, u, k)) for d in budgets}
nonsuff = {d: sorted((u, k) for u in inst_uids for k in ("0", "1") if not truth(d, u, k)) for d in budgets}
assert (len(suff[0.02]), len(suff[0.05]), len(suff[0.1])) == (22, 27, 28), [len(suff[d]) for d in budgets]
# per (d, bound, n): readings certified in >= 90% of draws; false certificates (any pass on a non-sufficient reading)
by = collections.defaultdict(list)
for x in rec: by[(x["d"], x["bound"], x["n"], x["uid"], str(x["k"]))].append(x["pass"])
sizes = sorted(set(x["n"] for x in rec))
def cert_count(d, bound, n): return sum(1 for (u, k) in suff[d] if by.get((d, bound, n, u, k)) and sum(by[(d, bound, n, u, k)]) >= 0.9 * len(by[(d, bound, n, u, k)]))  # suff lists instances, so a duplicated uid is counted per instance
def false_count(d, bound, n): return sum(1 for (u, k) in nonsuff[d] if any(by.get((d, bound, n, u, k), [])))
total_non = sum(1 for x in rec if not x["truth"]); total_false = sum(1 for x in rec if not x["truth"] and x["pass"])
assert total_false == 0, total_false
# panel C rows at d=0.05 (bernstein == weissman on the grid; use bernstein)
panel_sizes = [1e7, 10 ** 7.5, 1e8, 1e10]
rows = []
for n in panel_sizes:
    nn = min(sizes, key=lambda s: abs(math.log10(s) - math.log10(n)))
    c, f = cert_count(0.05, "bernstein", nn), false_count(0.05, "bernstein", nn)
    assert f == 0
    rows.append(f"{fmt_n(nn)} & {c} / {len(suff[0.05])} & {f} / {len(nonsuff[0.05])} \\\\\n")
open(os.path.join(ROOT, "paper", "generated", "gen_finite_panelc.tex"), "w").write("".join(rows))
c_a = cert_count(0.05, "bernstein", 10 ** 7.5); c_all = cert_count(0.05, "bernstein", 1e10)
assert c_a == 22 and c_all == 27, (c_a, c_all)
c_first = cert_count(0.05, "bernstein", min(sizes, key=lambda s: abs(math.log10(s) - 7)))
assert c_first == 0, c_first   # the first panel size certifies no sufficient reading yet (Appendix D.2 text)
assert cert_count(0.05, "weissman", 10 ** 7.5) == c_a
# margins, n*, collapse
ns = fd["d=0.05|bernstein"]["n_star"]
margins05 = sorted(set(round(v["margin"], 4) for v in ns.values()))
allns = {(key, k): v for key in ("d=0.02|bernstein", "d=0.05|bernstein", "d=0.1|bernstein") for k, v in fd[key]["n_star"].items()}
coll = [v["n_star_margin2"] for v in allns.values()]; mg = [v["margin"] for v in allns.values()]
lo, hi = min(coll), max(coll)
g = [v["margin"] for v in ns.values()]; n_ = [v["n_star"] for v in ns.values()]
import numpy as np
slope = np.polyfit(np.log(g), np.log(n_), 1)[0]
assert -2.6 < slope < -1.9, slope
# eps* maximal for all sufficient readings
es = [ex[u][str(d)][k]["eps_star"] for d in budgets for (u, k) in suff[d]]
assert min(es) >= 0.7 - 1e-9, min(es)
# delta_gen Clopper-Pearson from the E3 macros
gen = open(GEN + "gen_revision.tex").read()
def val(name): return int(re.search(r"\\newcommand\{\\" + name + r"\}\{([0-9]+)\}", gen).group(1))
m_f, g_f, m_e = val("AcExprFaithN"), val("AcLicensedFaithN"), val("AcExprN")
ucb = lambda x, m: float(beta_dist.ppf(0.95, x + 1, m - x))
u_f, u_e = ucb(m_f - g_f, m_f), ucb(m_e - g_f, m_e)
assert abs(u_f - 0.4227) < 5e-3 and abs(u_e - 0.6923) < 5e-3, (u_f, u_e)
out += [macro("FdSuffN", len(suff[0.05])), macro("FdNonN", len(nonsuff[0.05])), macro("FdCertA", c_a), macro("FdNa", fmt_n(10 ** 7.5)),
        macro("FdCertFirst", c_first), macro("FdNfirst", fmt_n(1e7)),
        macro("FdNall", fmt_n(1e10)), macro("FdFalse", total_false), macro("FdNonTotal", f"{total_non:,}".replace(",", "{,}")),
        macro("FdSizes", len(sizes)), macro("FdCollapseLo", f"{lo/1e4:.1f}"), macro("FdCollapseHi", f"{hi/1e4:.1f}"),
        macro("FdCollapseRatio", f"{hi/lo:.1f}"), macro("FdMarginLo", f"{min(mg):.4f}"), macro("FdMarginHi", f"{max(mg):.4f}"),
        macro("FdMarginRatio", f"{max(mg)/min(mg):.0f}"), macro("FdSlope", f"{slope:.1f}"), macro("FdEpsStar", "0.7"),
        macro("FdMarginSmall", f"{margins05[0]:.4f}"), macro("FdMarginLarge", f"{margins05[-1]:.4f}"),
        macro("FdSuffTwo", len(suff[0.02])), macro("FdSuffTen", len(suff[0.1])),
        macro("DgenUcbFaith", f"{100*u_f:.0f}\\%"), macro("DgenUcbExpr", f"{100*u_e:.0f}\\%"),
        macro("DgenPointFaith", f"{100*(m_f-g_f)/m_f:.0f}\\%"), macro("DgenPointExpr", f"{100*(m_e-g_f)/m_e:.0f}\\%")]
# appendix table: per budget
app = []
for d in budgets:
    key = f"d={d}|bernstein"; nst = fd[key]["n_star"]
    dm = sorted(set(round(v["margin"], 4) for v in nst.values()))
    cells = "; ".join(f"{m:.4f}: {fmt_n(next(v['n_star'] for v in nst.values() if round(v['margin'],4)==m))}" for m in dm)
    ftot = sum(1 for x in rec if x["d"] == d and x["bound"] == "bernstein" and not x["truth"])
    app.append(f"{d:g} & {len(suff[d])} & {cells} & 0 / {ftot:,} \\\\\n".replace(",", "{,}"))
open(GEN + "gen_finite_app.tex", "w").write("".join(app))
open(GEN + "gen_finite.tex", "w").write("% generated by make_gen_finite.py -- do not edit\n" + "\n".join(out) + "\n")
print("gen_finite: %d macros; panel C:\n%s" % (len(out), "".join(rows)), "appendix rows:\n" + "".join(app))
