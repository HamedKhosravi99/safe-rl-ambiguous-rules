"""V37: oracle semantic-risk frontier on the compiled suite (registration V37).

Under the contract R_q(pi) = sum_phi q_phi 1{J_{c_phi}(pi) > d} <= alpha,
V_q^alpha(d) = max_{S : q(S) <= alpha} V_{U \\ S}(d).  With K = 2 readings the
enumeration over S IS the exact solve, so the value needs no new LP: the
archive holds V_U, both single-reading values and the unconstrained value.
The identity (II)  inf_{F_psi^eps} R_q <= alpha  iff  V_psi - V_q^alpha <= eps
is checked with LPs.  The honesty column is whether the reading alpha relaxes
is the GOVERNING one: that realized rate is what a calibrated alpha bounds."""
from __future__ import annotations
import json, os, itertools
import numpy as np
from scipy.optimize import linprog
from .control_suite import _eligible, _flow, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool
from .score import score_reading

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARCH = os.path.join(ROOT, "results/e2e", "control_suite_uncapped.json")
OUT = os.path.join(ROOT, "results/e2e", "semantic_risk_frontier.json")
ALPHAS = (0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
BUDGETS = ("0.005", "0.02", "0.05"); EPSS = (0.0, 0.01, 0.05); TOL = 1e-9

def lp_value(m, obj, rows, rhs):
    A_eq, b_eq = _flow(m)
    res = linprog(-obj, A_ub=np.array(rows) if rows else None, b_ub=np.array(rhs) if rows else None,
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return None if res.status != 0 else float(obj @ res.x)

def lp_feasible(m, rows, rhs):
    A_eq, b_eq = _flow(m)
    res = linprog(np.zeros(A_eq.shape[1]), A_ub=np.array(rows) if rows else None, b_ub=np.array(rhs) if rows else None,
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return res.status == 0

def main():
    arch = {i["uid"]: i for i in json.load(open(ARCH))["instances"]}
    pt, _ = parse_prometheus(); tb = prom_threshold_bank(pt)
    rows = []; n_ident = 0; seen = {}
    for ti, t in enumerate(pt):
        pool = build_prom_pool(t, tb); ok, _w, readings = _eligible(pool)
        if not ok: continue
        uid = getattr(t, "name", "?"); seen[uid] = seen.get(uid, 0) + 1
        if len(readings) != 2: continue
        uid = f"{uid}#{ti}"
        gold = pool.classes[pool.gold_idx].rep
        key = lambda r: (round(float(r.threshold), 9), round(float(r.for_s), 9))
        gold_idx = [k for k, r in enumerate(readings) if key(r) == key(gold)]
        gold_idx = gold_idx[0] if gold_idx else None
        s = np.array([max(score_reading(t, r)[0], 1e-9) for r in readings]); q_score = s / s.sum()
        top = int(np.argmax(s))
        m = compile_instance(readings); r = m["r"].reshape(-1); crows = [m["C"][k].reshape(-1) for k in range(2)]
        rec = dict(uid=uid, gold_idx=gold_idx, top_idx=top, scores=s.tolist(), q_score=q_score.tolist(), budgets={})
        for b in BUDGETS:
            d = float(b)
            V_U = lp_value(m, r, crows, [d, d]); V = [lp_value(m, r, [crows[k]], [d]) for k in range(2)]; V_unc = lp_value(m, r, [], [])
            if V_U is None or any(v is None for v in V): continue
            binding = int(np.argmin(V))
            def frontier(q):
                out = {}
                for a in ALPHAS:
                    cands = [(V_U, "none")]
                    if q[0] <= a + TOL: cands.append((V[1], "0"))
                    if q[1] <= a + TOL: cands.append((V[0], "1"))
                    if q[0] + q[1] <= a + TOL: cands.append((V_unc, "both"))
                    val, rel = max(cands, key=lambda z: z[0])
                    relaxed_gold = (rel == "both") or (rel != "none" and gold_idx is not None and int(rel) == gold_idx)
                    out[str(a)] = dict(V=val, gain=(val - V_U) / V_U, relaxed=rel, relaxed_gold=bool(relaxed_gold))
                return out
            rec["budgets"][b] = dict(V_U=V_U, V=V, V_unc=V_unc, binding_idx=binding, binding_is_gold=(binding == gold_idx),
                                     price_binding=(max(V) - V_U) / V_U, frontier_score=frontier(q_score))
            # identity (II) with the score weights: LHS by LP feasibility over subsets, RHS by values
            for psi in range(2):
                for eps in EPSS:
                    for a in ALPHAS[:6]:
                        subsets = [S for k_ in range(3) for S in itertools.combinations(range(2), k_) if sum(q_score[i] for i in S) <= a + TOL]
                        lhs = False
                        for S in subsets:
                            keep = [k for k in range(2) if k not in S]
                            rws = [crows[k] for k in keep] + [crows[psi], -r]; rhs_ = [d] * len(keep) + [d, -(V[psi] - eps - 1e-7)]
                            if lp_feasible(m, rws, rhs_): lhs = True; break
                        Vqa = rec["budgets"][b]["frontier_score"][str(a)]["V"]
                        rhs = (V[psi] - Vqa <= eps + 1e-6)
                        assert lhs == rhs, (uid, b, psi, eps, a, lhs, rhs)
                        n_ident += 1
        rows.append(rec)
    # calibrated-by-rank weights: p_top = in-sample frequency that the top-scored reading is the governing one
    p_top = float(np.mean([r["top_idx"] == r["gold_idx"] for r in rows if r["gold_idx"] is not None]))
    for rec in rows:
        qc = np.array([p_top, 1 - p_top]) if rec["top_idx"] == 0 else np.array([1 - p_top, p_top])
        rec["q_cal"] = qc.tolist()
        for b in BUDGETS:
            R = rec["budgets"][b]; V_U, V, V_unc, gi = R["V_U"], R["V"], R["V_unc"], rec["gold_idx"]
            out = {}
            for a in ALPHAS:
                cands = [(V_U, "none")]
                if qc[0] <= a + TOL: cands.append((V[1], "0"))
                if qc[1] <= a + TOL: cands.append((V[0], "1"))
                if qc[0] + qc[1] <= a + TOL: cands.append((V_unc, "both"))
                val, rel = max(cands, key=lambda z: z[0])
                out[str(a)] = dict(V=val, gain=(val - V_U) / V_U, relaxed=rel,
                                   relaxed_gold=bool((rel == "both") or (rel != "none" and gi is not None and int(rel) == gi)))
            R["frontier_cal"] = out
    # ---- summary ---------------------------------------------------------
    summ = dict(n=len(rows), p_top_in_sample=p_top, identity_II_checks=n_ident,
                binding_is_gold={b: float(np.mean([r["budgets"][b]["binding_is_gold"] for r in rows])) for b in BUDGETS},
                q_score_min_mass_med=float(np.median([min(r["q_score"]) for r in rows])), frontier={})
    for qname in ("frontier_score", "frontier_cal"):
        summ["frontier"][qname] = {}
        for b in BUDGETS:
            per_a = {}
            for a in ALPHAS:
                g = np.array([r["budgets"][b][qname][str(a)]["gain"] for r in rows])
                rg = np.array([r["budgets"][b][qname][str(a)]["relaxed_gold"] for r in rows])
                rel = np.array([r["budgets"][b][qname][str(a)]["relaxed"] != "none" for r in rows])
                pos = g > 1e-9
                per_a[str(a)] = dict(frac_relaxed=float(rel.mean()), frac_gain_pos=float(pos.mean()),
                                     gain_med_all=float(np.median(g)), gain_med_pos=(float(np.median(g[pos])) if pos.any() else 0.0),
                                     realized_semantic_risk=float(rg.mean()))
            summ["frontier"][qname][b] = per_a
    a5 = summ["frontier"]["frontier_cal"]
    go = any(a5[b]["0.05"]["gain_med_pos"] >= 0.05 and a5[b]["0.05"]["frac_gain_pos"] >= 0.25 for b in BUDGETS)
    summ["registered_go"] = bool(go)
    json.dump(dict(registration="V37 (REGISTRATION_V28.md)", alphas=list(ALPHAS), summary=summ, rows=rows), open(OUT, "w"), indent=1)
    print(f"n={summ['n']}  identity (II) asserted at {n_ident} points  p_top (in-sample, K=2) = {p_top:.3f}")
    print("binding reading IS the governing one:", {b: f"{100*v:.0f}%" for b, v in summ['binding_is_gold'].items()})
    print(f"median mass of the less-scored reading under q_score: {summ['q_score_min_mass_med']:.3f}")
    for qname in ("frontier_score", "frontier_cal"):
        print(f"\n=== {qname} ===")
        for b in BUDGETS:
            print(f"  d={b}:  alpha | relaxed% | gain>0% | gain med (all) | gain med (gain>0) | REALIZED semantic risk (relaxed the governing reading)")
            for a in ALPHAS:
                v = summ["frontier"][qname][b][str(a)]
                print(f"         {a:<5} | {100*v['frac_relaxed']:6.0f}%  | {100*v['frac_gain_pos']:6.0f}% | {100*v['gain_med_all']:+8.1f}%     | {100*v['gain_med_pos']:+8.1f}%          | {100*v['realized_semantic_risk']:5.0f}%")
    print("\nREGISTERED GO (alpha=0.05, q_cal, median gain over gainers >= 5pts and gainers >= 25%):", go)

if __name__ == "__main__":
    main()
