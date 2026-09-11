"""W8: the practical-baseline table on the exact public-rule analyses.

For every eligible compiled monitoring instance (the 28 of V13/surrogate
runs) and every admission pool (44, free class), at the operating budget:

  top-1 / most-plausible singleton  the reading the frozen benchmark score
                                    ranks first, optimized alone
  self-consistency singleton        not applicable here (one generator run
                                    per rule; the majority-vote arm lives in
                                    the controlled maintenance domain)
  strictest-when-defined            the pointwise-most-restrictive reading
                                    when one exists (equals set protection
                                    by dominance); undefined on crossed pools
  pointwise-max surrogate           always-robustify for a single-cost learner
  set protection                    V_U, program (1)
  ARROW                             V_U -- the certified singleton (when the
                                    face certifies) or the set, both attain V_U
  perfect-clarification oracle      Delta_clarify = V_{psi_dagger}(d) - V_U(d)
                                    with psi_dagger the recorded (retrospective)
                                    reading of the rule; the return perfect
                                    clarification would recover relative to
                                    set protection, plus the fraction of
                                    instances that would clarify at query
                                    cost c ("clarify when Delta > c")

Archived V_U values are asserted against policy_class_budget.json wherever
the instance is one of its rows.  Compiled instances whose recorded reading
is not among the maximal readings are re-compiled with it appended (the
extra counter refines the state space; V_U over the original readings is
asserted unchanged).

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.baseline_table
Writes results/e2e/baseline_table.json
"""
from __future__ import annotations

import json
import os
from typing import List

import numpy as np
from scipy.optimize import linprog

from .control_suite import _eligible, _flow, compile_instance
from .dominance import dominating_member
from .evaluate import build_kyv_pool, build_prom_pool
from .parse import parse_kyverno, parse_prometheus, prom_threshold_bank
from .policy_class_budget import free_values_multi
from .score import score_reading

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "results/e2e", "baseline_table.json")
OP = 0.05
TOL = 1e-7
FACE_SLACK = 1e-9
COST_GRID = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)


def _lp(m, obj, rows, rhs, extra=()):
    A_eq, b_eq = _flow(m)
    A = [np.asarray(r).reshape(-1) for r in rows] + [np.asarray(r).reshape(-1) for r, _ in extra]
    b = list(rhs) + [v for _, v in extra]
    res = linprog(-np.asarray(obj).reshape(-1), A_ub=np.array(A) if A else None, b_ub=np.array(b) if A else None,
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if res.status != 0:
        return None, None
    return float(np.asarray(obj).reshape(-1) @ res.x), res.x


def _best_tiebreak_worst(m, k, V_k, costs, d):
    """min over the psi_k-optimal face of max_phi <c_phi, x> (epigraph LP)."""
    A_eq, b_eq = _flow(m)
    n = A_eq.shape[1]
    A_eq2 = np.hstack([A_eq, np.zeros((A_eq.shape[0], 1))])
    rows, rhs = [], []
    for c in costs:
        rows.append(np.append(c.reshape(-1), -1.0)); rhs.append(0.0)
    rows.append(np.append(costs[k].reshape(-1), 0.0)); rhs.append(d)
    rows.append(np.append(-m["r"].reshape(-1), 0.0)); rhs.append(-(V_k - FACE_SLACK))
    obj = np.zeros(n + 1); obj[-1] = 1.0
    res = linprog(obj, A_ub=np.array(rows), b_ub=np.array(rhs), A_eq=A_eq2, b_eq=b_eq,
                  bounds=[(0, None)] * n + [(None, None)], method="highs")
    return float(res.fun) if res.status == 0 else None


class _R:
    pass


def compiled_rows(pcb, ps):
    pt, _ = parse_prometheus()
    bank = prom_threshold_bank(pt)
    verd = {(r["rule_id"], round(r["budget"], 6)): r for r in pcb["rows"]
            if r["policy_class"] == "monitoring_compiled" and r["status"] == "compiled"}
    face = {r["rule_id"]: r for r in ps["rows"] if abs(r["budget"] - OP) < 1e-9}
    rows = []
    for ti, t in enumerate(pt):
        pool = build_prom_pool(t, bank)
        ok, _why, readings = _eligible(pool)
        if not ok:
            continue
        rid = f"{t.name}#{ti}"
        if (rid, OP) not in verd:
            continue
        gold = pool.classes[pool.gold_idx].rep
        keys = sorted({(float(r.threshold), float(r.for_s)) for r in readings})
        gold_key = (float(gold.threshold), float(gold.for_s))
        gold_in = gold_key in keys
        # frozen benchmark score per maximal reading
        scores = [score_reading(t, r)[0] for r in readings]
        # order readings by key exactly as compile_instance does
        by_key = {(float(r.threshold), float(r.for_s)): r for r in readings}
        ordered = [by_key[k] for k in keys]
        sc = [score_reading(t, r)[0] for r in ordered]
        top = int(np.argmax(sc))                   # first max wins (deterministic)
        top_alt = int(len(sc) - 1 - np.argmax(sc[::-1]))   # last max: the alternative tie-break
        tie = bool(sum(1 for x in sc if abs(x - max(sc)) < 1e-12) > 1)
        comp_readings = list(ordered)
        if not gold_in:
            g = _R(); g.threshold, g.for_s = gold_key
            comp_readings = comp_readings + [g]
        m = compile_instance(comp_readings)
        K = len(keys)
        allk = sorted({(float(r.threshold), float(r.for_s)) for r in comp_readings})
        pos = {kk: i for i, kk in enumerate(allk)}
        idx = [pos[k] for k in keys]
        gidx = pos[gold_key]
        costs = [m["C"][i] for i in idx]
        r = m["r"]
        V_U, _ = _lp(m, r, costs, [OP] * K)
        arch = verd[(rid, OP)]
        assert abs(V_U - arch["V_U"]) < 1e-6, (rid, V_U, arch["V_U"])
        V = []
        for i in range(K):
            v, _ = _lp(m, r, [costs[i]], [OP])
            V.append(v)
        assert abs(min(V) - arch["V_min_single"]) < 1e-6, rid
        V_top = V[top]
        worst_top = _best_tiebreak_worst(m, top, V_top, costs, OP)
        worst_top_alt = _best_tiebreak_worst(m, top_alt, V[top_alt], costs, OP)
        cmax = np.maximum.reduce(costs)
        V_surr, _ = _lp(m, r, [cmax], [OP])
        V_gold, _ = _lp(m, r, [m["C"][gidx]], [OP])
        fr = face.get(rid, {})
        rows.append(dict(rule_id=rid, K=K, n_states=m["nS"], gold_in_maximal=gold_in,
                         gold_reading=dict(theta=gold_key[0], for_s=gold_key[1]),
                         scores=sc, top1_index=top, top1_reading=dict(theta=keys[top][0], for_s=keys[top][1]),
                         V_U=V_U, V_singles=V, V_top1=V_top, worst_top1_best_tiebreak=worst_top,
                         top1_unsafe=bool(worst_top is not None and worst_top > OP + TOL),
                         top1_score_tie=tie, top1_alt_index=top_alt, V_top1_alt=V[top_alt],
                         top1_alt_unsafe=bool(worst_top_alt is not None and worst_top_alt > OP + TOL),
                         top1_is_gold=bool(keys[top] == gold_key),
                         V_surrogate=V_surr, V_gold=V_gold,
                         delta_clarify=V_gold - V_U, delta_clarify_rel=(V_gold - V_U) / V_gold if V_gold > 0 else 0.0,
                         screen_fires=bool(arch["screen_fires"]),
                         face_certified=bool(fr.get("policy_sufficient_exists", False))))
    return rows


def admission_rows(pcb):
    kt, _ = parse_kyverno()
    free = {r["rule_id"]: r for r in pcb["rows"] if r["policy_class"] == "free" and r["family"] == "kyverno"
            and abs(r["budget"] - OP) < 1e-9}
    rows = []
    for ti, t in enumerate(kt):
        pool = build_kyv_pool(t)
        rid = f"{t.policy_name}/{t.rule_name}#{ti}"
        cand = [r for r in free if r.endswith(f"#{ti}")]
        arch = free[cand[0]] if cand else None
        vectors = [list(c.vector) for c in pool.classes]
        vals = free_values_multi(vectors, [OP])[OP]
        V_full, V_k = vals
        if arch is not None:
            assert V_full == arch["V_U"], (rid, V_full, arch["V_U"])
        sc = [score_reading(t, c.rep) for c in pool.classes]
        sc = [s[0] for s in sc]
        top = int(np.argmax(sc))
        g = pool.gold_idx
        dom = dominating_member(vectors)
        # top-1 policy: the largest admitted set under the top reading alone; its
        # worst mean cost under the other readings, at the best tie-break: exact
        # enumeration is what e16 does; here the greedy-prefix selection is used
        # and labelled as the returned policy (tie-break-sensitive), as e4 did
        v = np.asarray(vectors[top], dtype=float)
        order = np.argsort(v, kind="stable")
        best, run = 0, 0.0
        for n_, i in enumerate(order, start=1):
            run += v[i]
            if run / n_ <= OP + 1e-12:
                best = n_
        S = order[:best]
        worst = max(float(np.mean(np.asarray(vv)[S])) for vv in vectors) if best else 0.0
        cmax = np.max(np.asarray(vectors, dtype=float), axis=0)
        V_surr = free_values_multi([list(cmax)], [OP])[OP][1][0]
        rows.append(dict(rule_id=arch["rule_id"] if arch else rid, K=len(vectors), n_fixtures=pool.n_fixtures,
                         scores=sc, top1_index=top, V_U=V_full, V_singles=V_k, V_top1=V_k[top],
                         worst_top1_returned=worst, top1_unsafe=bool(worst > OP + 1e-12),
                         top1_is_gold=bool(top == g), strictest_defined=bool(dom >= 0),
                         V_strictest=(V_full if dom >= 0 else None), V_surrogate=V_surr, V_gold=V_k[g],
                         delta_clarify=V_k[g] - V_full, delta_clarify_rel=(V_k[g] - V_full) / V_k[g] if V_k[g] > 0 else 0.0,
                         screen_fires=bool(arch["screen_fires"]) if arch else None))
    return rows


def summarize(rows, label):
    n = len(rows)
    rel_top = [r["V_top1"] / r["V_U"] - 1.0 for r in rows if r["V_U"] > 0]
    rel_surr = [1.0 - r["V_surrogate"] / r["V_U"] for r in rows if r["V_U"] > 0]
    dc = [r["delta_clarify_rel"] for r in rows]
    curve = {str(c): float(np.mean([x > c + 1e-12 for x in dc])) for c in COST_GRID}
    out = dict(n=n, top1_unsafe=int(sum(r["top1_unsafe"] for r in rows)), top1_is_gold=int(sum(r["top1_is_gold"] for r in rows)),
               top1_return_over_VU_median=float(np.median(rel_top)), top1_return_over_VU_max=float(max(rel_top)),
               surrogate_loss_median=float(np.median(rel_surr)), surrogate_loss_min=float(min(rel_surr)), surrogate_loss_max=float(max(rel_surr)),
               clarify_gain_median=float(np.median(dc)), clarify_gain_min=float(min(dc)), clarify_gain_max=float(max(dc)),
               clarify_gain_zero=int(sum(1 for x in dc if x <= 1e-12)),
               clarify_when_gain_exceeds=curve)
    ratios = [r["V_top1"] / r["V_U"] for r in rows if r["V_U"] > 0]
    out["top1_ratio_median"] = float(np.median(ratios)); out["top1_ratio_min"] = float(min(ratios)); out["top1_ratio_max"] = float(max(ratios))
    if label == "compiled":
        out["gold_in_maximal"] = int(sum(r["gold_in_maximal"] for r in rows))
        out["face_certified"] = int(sum(r["face_certified"] for r in rows))
        out["top1_score_ties"] = int(sum(r["top1_score_tie"] for r in rows))
        out["top1_alt_unsafe"] = int(sum(r["top1_alt_unsafe"] for r in rows))
        out["top1_unsafe_untied"] = int(sum(r["top1_unsafe"] for r in rows if not r["top1_score_tie"]))
        out["n_untied"] = int(sum(1 for r in rows if not r["top1_score_tie"]))
    else:
        out["strictest_defined"] = int(sum(r["strictest_defined"] for r in rows))
    return out


def main():
    pcb = json.load(open(os.path.join(ROOT, "results/e2e", "policy_class_budget.json")))
    ps = json.load(open(os.path.join(ROOT, "results/e2e", "policy_sufficiency.json")))
    comp = compiled_rows(pcb, ps)
    adm = admission_rows(pcb)
    nested = sum(1 for r in pcb["rows"] if r["policy_class"] == "monitoring_compiled"
                 and r["status"] == "provably_clear_nested" and abs(r["budget"] - OP) < 1e-9)
    res = dict(registration="W8 baseline table, 2026-09-02; archived V_U asserted per instance; no verdict changed",
               operating_budget=OP, cost_grid=list(COST_GRID),
               compiled=dict(rows=comp, summary=summarize(comp, "compiled")),
               admission=dict(rows=adm, summary=summarize(adm, "admission")),
               monitoring_nested_clear=dict(n=nested, note="strictest reading exists by nesting; equals set protection; no LP"))
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(dict(compiled=res["compiled"]["summary"], admission=res["admission"]["summary"], nested=nested), indent=1))


if __name__ == "__main__":
    main()
