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
LNAME = {"fqi": "BCQ-FQI-Lagrangian", "cpq": "CPQ", "pid": "PID-Lagrangian", "cql": "CQL-Lagrangian (neural)", "caps": "CAPS", "o3srl": "O3SRL"}
LKEY = {"fqi": "Fqi", "cpq": "Cpq", "pid": "Pid", "cql": "Cql", "caps": "Caps", "o3srl": "Osrl"}
TABLE_ORDER = ["fqi", "pid", "o3srl", "cpq", "caps", "cql"]   # Table 2 rows: the multiplier family, then the limit-based learners, then the neural learner
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
for l in [x for x in TABLE_ORDER if x in ln["learners"]]:
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
        mac(f"Sv{key}SmallPos", f"{sum(g > 1e-9 for g in g2)}/{len(keys)}"); mac(f"Sv{key}SmallGapMed", pts(st.median(g2)))
        mac(f"Sv{key}SmallArrowSafe", pct(st.mean(a2[k]["arrow_safe"] for k in keys), 0))
        if l in ("fqi", "cpq", "pid", "o3srl"):        # the appendix states these four are positive on every instance at the small n
            assert sum(g > 1e-9 for g in g2) == len(keys), ("small-n", l); small_meds.append(st.median(g2))
mac("SvSmallGapLo", pts(min(small_meds))); mac("SvSmallGapHi", pts(max(small_meds)))
# range of the per-learner median gains, unsigned, for the abstract and contributions ("18.7--36.4 points")
big_meds = [st.median([agg(max(ln["n_grid"]), l)[k]["arrow_ret"] - agg(max(ln["n_grid"]), l)[k]["surr_ret"] for k in keys]) for l in TABLE_ORDER if l in ln["learners"]]
mac("SvLearners", len([l for l in TABLE_ORDER if l in ln["learners"]]))
assert min(big_meds) > 0
mac("SvGapMedLo", f"{100*min(big_meds):.1f}"); mac("SvGapMedHi", f"{100*max(big_meds):.1f}")
# decomposition: learned singleton shortfall from V_U and learned surrogate shortfall from the exact surrogate
for l in [x for x in TABLE_ORDER if x in ln["learners"]]:
    a = agg(max(ln["n_grid"]), l); key = LKEY[l]
    mac(f"Sv{key}ArrowShort", pct(st.mean(1 - a[k]["arrow_ret"] for k in keys)))
    mac(f"Sv{key}SurrExtra", pct(st.mean((1 - price_of[k]) - a[k]["surr_ret"] for k in keys)))
mac("SvExactSurrMean", pct(st.mean(1 - price_of[k] for k in keys)))
_extra = {l: st.mean((1 - price_of[k]) - agg(max(ln["n_grid"]), l)[k]["surr_ret"] for k in keys) for l in TABLE_ORDER if l in ln["learners"]}
mac("SvSurrExtraLo", pct(min(_extra.values()))); mac("SvSurrExtraHi", pct(max(_extra.values())))
# per-instance table (Appendix D.3): every learner gains on every sufficient instance; range over all cells;
# the exact surrogate price rises with K within each (variant, d) demand regime
_all = [g for k in keys for g in perinst[k]]
assert min(_all) > 0 and len(_all) == len(keys) * len(ln["learners"]), (min(_all), len(_all))
mac("SvGapAllMin", pts(min(_all))); mac("SvGapAllMax", pts(max(_all)))
for (v, d) in sorted({(k[0], k[2]) for k in keys}):
    _ps = [price_of[k] for k in sorted(keys, key=lambda k: k[1]) if k[0] == v and k[2] == d]
    assert all(b > a for a, b in zip(_ps, _ps[1:])), ("price not increasing in K", v, d, _ps)
# ---- facts about the run recorded for the appendix text
_big = max(ln["n_grid"])
cql_arrow = [r for r in rows if r["learner"] == "cql" and r["n"] == _big and r["is_arrow"]]
if cql_arrow: mac("SvCqlArrowLamZero", f"{sum(1 for r in cql_arrow if r['lam'] == 0.0)}/{len(cql_arrow)}")
_key = lambda r: (r["variant"], r["K"], r["d"], r["n"], r["seed"], r["arm"])
_fqi = {_key(r): r for r in rows if r["learner"] == "fqi"}
for l in ("pid", "o3srl"):
    if l in ln["learners"]:
        _o = {_key(r): r for r in rows if r["learner"] == l}
        _cells = [k for k in _fqi if k[3] == _big and (_fqi[k]["is_arrow"] or k[5] == "surrogate")]
        mac(f"Sv{LKEY[l]}SameAsFqi", f"{sum(1 for k in _cells if abs(_fqi[k]['ret_frac'] - _o[k]['ret_frac']) < 1e-9)}/{len(_cells)}")
_fb = [r for r in rows if r["honored_offline"] > r["d"] + 1e-12]
mac("SvFallbackRows", len(_fb)); mac("SvFallbackDesc", "; ".join(f"{LNAME[r['learner']]}, {r['variant']} $K{{=}}{r['K']}$ $d{{=}}{r['d']}$, $n{{=}}{r['n']:,}$, seed {r['seed']}, {'selected reading' if r['is_arrow'] else r['arm']}".replace(",", "{,}") if False else f"{LNAME[r['learner']]} on {r['variant']} $K{{=}}{r['K']}$, $d{{=}}{r['d']}$ at $n{{=}}{r['n']}$ (seed {r['seed']}, {'selected reading' if r['is_arrow'] else 'surrogate'})" for r in _fb) if _fb else "none")
_arrow_safe = {l: st.mean(agg(_big, l)[k]["arrow_safe"] for k in keys) for l in TABLE_ORDER if l in ln["learners"]}
_lo = min(_arrow_safe, key=_arrow_safe.get); _hi = max(_arrow_safe, key=_arrow_safe.get)
mac("SvArrowSafeLo", pct(_arrow_safe[_lo], 0)); mac("SvArrowSafeLoLearner", LNAME[_lo]); mac("SvArrowSafeHi", pct(_arrow_safe[_hi], 0))
_arrow_ret = {l: st.mean(agg(_big, l)[k]["arrow_ret"] for k in keys) for l in TABLE_ORDER if l in ln["learners"]}
mac("SvArrowRetLo", pct(min(_arrow_ret.values()))); mac("SvArrowRetHi", pct(max(_arrow_ret.values())))
_surr_safe = {l: st.mean(agg(_big, l)[k]["surr_safe"] for k in keys) for l in TABLE_ORDER if l in ln["learners"]}
mac("SvSurrSafeLo", pct(min(_surr_safe.values()), 0))
# ---- sensitivity to the earlier inherited settings, from the archived runs kept beside the main archive
def _agg_file(path, learners):
    if not os.path.exists(path): return {}
    _rows = json.load(open(path))["rows"]; out = {}
    for l in learners:
        per = {}
        for k in keys:
            sel = [r for r in _rows if r["learner"] == l and r["n"] == _big and (r["variant"], r["K"], r["d"]) == k]
            if not sel: break
            ar = [r for r in sel if r["is_arrow"]]; su = [r for r in sel if r["arm"] == "surrogate"]
            per[k] = (st.mean(x["ret_frac"] for x in ar), st.mean(x["ret_frac"] for x in su), st.mean(float(x["safe"]) for x in ar))
        if len(per) == len(keys):
            out[l] = dict(arrow=st.mean(v[0] for v in per.values()), surr=st.mean(v[1] for v in per.values()), gap=st.median(v[0] - v[1] for v in per.values()),
                          pos=sum(1 for v in per.values() if v[0] - v[1] > 1e-9), arrow_safe=st.mean(v[2] for v in per.values()))
    return out
for tag, fname in (("Coarse", "scope_agent_learn_coarsegrid.json"), ("TauFive", "scope_agent_learn_tau005.json"), ("GainHalf", "scope_agent_learn_pid_gains_x0.5.json"), ("GainDouble", "scope_agent_learn_pid_gains_x2.json")):
    for l, v in _agg_file(os.path.join(ROOT, "results/e2e", fname), ["fqi", "pid", "o3srl", "cpq", "caps", "cql"]).items():
        k2 = LKEY[l]; mac(f"Sv{tag}{k2}Surr", pct(v["surr"])); mac(f"Sv{tag}{k2}Arrow", pct(v["arrow"])); mac(f"Sv{tag}{k2}GapMed", pts(v["gap"]))
        mac(f"Sv{tag}{k2}Pos", f"{v['pos']}/{len(keys)}"); mac(f"Sv{tag}{k2}ArrowSafe", pct(v["arrow_safe"], 0))
os.makedirs(GEN, exist_ok=True)
H = "% AUTO-GENERATED by scripts/paper/make_gen_v50.py -- do not edit\n"
open(os.path.join(GEN, "gen_v50_macros.tex"), "w").write(H + "\n".join(M) + "\n")
open(os.path.join(GEN, "gen_v50_table_main.tex"), "w").write(H + "\n".join(table_main) + "\n")
with open(os.path.join(GEN, "gen_v50_perinst.tex"), "w") as fh:
    fh.write(H)
    for k in keys:
        fh.write(f"{k[0]} & {k[1]} & ${k[2]:.2f}$ & {pct(price_of[k])} & " + " & ".join(pts(g) for g in perinst[k]) + " \\\\\n")
print("\n".join(table_main)); print("macros:", len(M))
