#!/usr/bin/env python3
"""Emit generated/gen_v50_macros.tex, gen_v50_table_main.tex and gen_v50_perinst.tex from the V50
archives (results/e2e/scope_agent.json, scope_agent_learn.json): the scope-ambiguous
agent-service experiment (Section 5.4 / Appendix).  Every number the manuscript prints for
this experiment comes from here; each headline direction is asserted before writing."""
import json, os, statistics as st, collections
from math import comb
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAPER = os.environ.get("ARROW_PAPER_DIR", os.path.join(ROOT, "paper"))
GEN = os.path.join(PAPER, "generated")
ex = json.load(open(os.path.join(ROOT, "results/e2e", "scope_agent.json")))
ln = json.load(open(os.path.join(ROOT, "results/e2e", "scope_agent_learn.json")))
LNAME = {"fqi": "BCQ-FQI-Lagrangian", "cpq": "CPQ", "pid": "PID-Lagrangian", "cql": "CQL-Lagrangian (neural)"}
LKEY = {"fqi": "Fqi", "cpq": "Cpq", "pid": "Pid", "cql": "Cql"}
def pct(x, k=1): return f"{100*x:.{k}f}\\%"
def pts(x, k=1): return f"{100*x:+.{k}f}"
def sign_p(pos, neg):
    n = pos + neg; return sum(comb(n, k) for k in range(pos, n + 1)) / 2 ** n
M = []
def mac(n, v): M.append(f"\\newcommand{{\\{n}}}{{{v}}}")
# ---- exact surface
inst_all = [k for k in ex["exact"]]
cert = [(k, v) for k, v in ex["exact"].items() if v["certified"]]
mac("SvInstances", len(inst_all)); mac("SvCertified", len(cert))
assert len(cert) == 9 and len(inst_all) == 36
prices = [v["price"] for _, v in cert]
mac("SvPriceLo", pct(min(prices))); mac("SvPriceHi", pct(max(prices))); mac("SvPriceMean", pct(st.mean(prices)))
byK = collections.defaultdict(list)
for k, v in cert: byK[int(k.split("|K")[1].split("|")[0])].append(v["price"])
for K in (2, 3, 4):
    mac(f"SvPriceK{['','','Two','Three','Four'][K]}Lo", pct(min(byK[K]))); mac(f"SvPriceK{['','','Two','Three','Four'][K]}Hi", pct(max(byK[K])))
assert all(st.mean(byK[K]) < st.mean(byK[K + 1]) for K in (2, 3)), "price no longer grows with K"
# certification pattern: certified iff every auxiliary unconstrained rate + face slack fits (report the variants)
# the wrong-reading arm can exceed V_U only by violating a retained reading: it recovers the unconstrained optimum
unc = [v["V_unc"] / v["V_U"] for _, v in cert]
mac("SvUncLo", pct(min(unc), 0)); mac("SvUncHi", pct(max(unc), 0)); mac("SvUncMean", pct(st.mean(unc), 0))
mac("SvCertVariants", "light auxiliary demand at $d\\in\\{0.10,0.15\\}$ and medium at $d=0.15$, for every $K$")
mac("SvGamma", ex["gamma"]); mac("SvRLocal", ex["r_local"]); mac("SvEps", ex["eps"])
mac("SvPrimaryDemand", "--".join(str(x) for x in (min(ex["primary_demand"]), max(ex["primary_demand"]))))
aux = ex["aux_demand"]; mac("SvAuxLo", min(min(v) for v in aux.values())); mac("SvAuxHi", max(max(v) for v in aux.values()) + 1)
mac("SvDGrid", ", ".join(f"{d:g}" for d in ex["d_grid"]))
# ---- learned arms
rows = ln["rows"]; tasks = ln["tasks"]; keys = [(t["variant"], t["K"], t["d"]) for t in tasks]
price_of = {(t["variant"], t["K"], t["d"]): t["price"] for t in tasks}
by = collections.defaultdict(list)
for r in rows: by[(r["n"], r["learner"], r["variant"], r["K"], r["d"], r["arm"], r["is_arrow"])].append(r)
def agg(n, l):
    out = {}
    for key in keys:
        sel = {(a, ia): v for (n_, l_, va, K, dd, a, ia), v in by.items() if (n_, l_, va, K, dd) == (n, l) + key}
        if not sel: return None
        arrow = [v for (a, ia), v in sel.items() if ia][0]; surr = sel[("surrogate", False)]
        wrong = [x for (a, ia), v in sel.items() if not ia and a != "surrogate" for x in v]
        out[key] = dict(arrow_ret=st.mean(x["ret_frac"] for x in arrow), arrow_safe=st.mean(float(x["safe"]) for x in arrow),
                        surr_ret=st.mean(x["ret_frac"] for x in surr), surr_safe=st.mean(float(x["safe"]) for x in surr),
                        wrong_ret=st.mean(x["ret_frac"] for x in wrong), wrong_unsafe=1 - st.mean(float(x["safe"]) for x in wrong))
    return out
table_main, perinst = [], {k: [] for k in keys}
mac("SvSeeds", ln["seeds"]); mac("SvNBig", f"{max(ln['n_grid']):,}".replace(",", "{,}")); mac("SvNSmall", f"{min(ln['n_grid']):,}".replace(",", "{,}"))
small_meds = []
for l in ln["learners"]:
    a = agg(max(ln["n_grid"]), l); assert a is not None
    gap = [a[k]["arrow_ret"] - a[k]["surr_ret"] for k in keys]; pos = sum(g > 1e-9 for g in gap); neg = sum(g < -1e-9 for g in gap)
    assert pos == len(keys) and neg == 0, (l, pos, neg)
    f = lambda fld: st.mean(a[k][fld] for k in keys)
    assert f("arrow_ret") > f("surr_ret") and f("surr_safe") >= 0.99, l
    key = LKEY[l]
    unc_of = {(t["variant"], t["K"], t["d"]): ex["exact"][f"{t['variant']}|K{t['K']}|d{t['d']}"]["V_unc"] / t["V_U"] for t in tasks}
    assert all(a[k]["wrong_ret"] <= unc_of[k] + 1e-6 for k in keys), (l, "wrong reading above the unconstrained optimum")
    if l != "cql": assert f("wrong_unsafe") == 1.0 and all(abs(a[k]["wrong_ret"] - unc_of[k]) < 1e-3 for k in keys), (l, "tabular wrong reading should recover V_unc")
    mac(f"Sv{key}Wrong", pct(f("wrong_ret"), 0)); mac(f"Sv{key}WrongUnsafe", pct(f("wrong_unsafe"), 0))
    mac(f"Sv{key}Surr", pct(f("surr_ret"))); mac(f"Sv{key}SurrSafe", pct(f("surr_safe"), 0))
    mac(f"Sv{key}Arrow", pct(f("arrow_ret"))); mac(f"Sv{key}ArrowSafe", pct(f("arrow_safe"), 0))
    mac(f"Sv{key}GapMed", pts(st.median(gap))); mac(f"Sv{key}GapMean", pts(st.mean(gap))); mac(f"Sv{key}GapMin", pts(min(gap))); mac(f"Sv{key}GapMax", pts(max(gap)))
    mac(f"Sv{key}Pos", f"{pos}/{len(gap)}"); mac(f"Sv{key}SignP", f"{sign_p(pos, neg):.3f}")
    table_main.append(f"{LNAME[l]} & {pct(f('surr_ret'))} & {pct(f('arrow_ret'))} & {pts(st.median(gap))} \\\\")
    for k, g in zip(keys, gap): perinst[k].append(g)
    if l != "cql":
        a2 = agg(min(ln["n_grid"]), l); g2 = [a2[k]["arrow_ret"] - a2[k]["surr_ret"] for k in keys]
        assert sum(g > 1e-9 for g in g2) == len(keys), ("small-n", l); small_meds.append(st.median(g2))
mac("SvSmallGapLo", pts(min(small_meds))); mac("SvSmallGapHi", pts(max(small_meds)))
# range of the per-learner median gains, unsigned, for the abstract and contributions ("18.7--36.4 points")
big_meds = [st.median([agg(max(ln["n_grid"]), l)[k]["arrow_ret"] - agg(max(ln["n_grid"]), l)[k]["surr_ret"] for k in keys]) for l in ln["learners"]]
assert min(big_meds) > 0
mac("SvGapMedLo", f"{100*min(big_meds):.1f}"); mac("SvGapMedHi", f"{100*max(big_meds):.1f}")
# decomposition: learned singleton shortfall from V_U and learned surrogate shortfall from the exact surrogate
for l in ln["learners"]:
    a = agg(max(ln["n_grid"]), l); key = LKEY[l]
    mac(f"Sv{key}ArrowShort", pct(st.mean(1 - a[k]["arrow_ret"] for k in keys)))
    mac(f"Sv{key}SurrExtra", pct(st.mean((1 - price_of[k]) - a[k]["surr_ret"] for k in keys)))
mac("SvExactSurrMean", pct(st.mean(1 - price_of[k] for k in keys)))
# per-instance table (Appendix D.3): every learner gains on every sufficient instance; range over all cells;
# the exact surrogate price rises with K within each (variant, d) demand regime
_all = [g for k in keys for g in perinst[k]]
assert min(_all) > 0 and len(_all) == len(keys) * len(ln["learners"]), (min(_all), len(_all))
mac("SvGapAllMin", pts(min(_all))); mac("SvGapAllMax", pts(max(_all)))
for (v, d) in sorted({(k[0], k[2]) for k in keys}):
    _ps = [price_of[k] for k in sorted(keys, key=lambda k: k[1]) if k[0] == v and k[2] == d]
    assert all(b > a for a, b in zip(_ps, _ps[1:])), ("price not increasing in K", v, d, _ps)
os.makedirs(GEN, exist_ok=True)
H = "% AUTO-GENERATED by paper/final/make_gen_v50.py -- do not edit\n"
open(os.path.join(GEN, "gen_v50_macros.tex"), "w").write(H + "\n".join(M) + "\n")
open(os.path.join(GEN, "gen_v50_table_main.tex"), "w").write(H + "\n".join(table_main) + "\n")
with open(os.path.join(GEN, "gen_v50_perinst.tex"), "w") as fh:
    fh.write(H)
    for k in keys:
        fh.write(f"{k[0]} & {k[1]} & ${k[2]:.2f}$ & {pct(price_of[k])} & " + " & ".join(pts(g) for g in perinst[k]) + " \\\\\n")
print("\n".join(table_main)); print("macros:", len(M))
