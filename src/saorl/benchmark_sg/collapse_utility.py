"""V48: does a NON-DOMINATED zero-query collapse matter for learning utility?

Compiled monitoring instances where Decide certifies one of two crossing
readings (neither cost dominates pointwise; asserted). Arms from the same
log: decide-single (certified reading psi*), fullset (K multipliers /
K rows), surrogate (pointwise-max single signal), permissive (the other
reading); learners: Lagrangian FQI, occupancy LP on P_hat (nominal and
tightened rho = 0.5), occupancy LP with count-based pessimistic costs.
Exact LPs on the true model give the tightened values and the tolerance
radius of the certificate. Registration V48.

Run: OMP_NUM_THREADS=1 SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.collapse_utility
Writes results/e2e/collapse_utility.json
"""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np
from scipy.optimize import linprog

from .control_mdp import GAMMA
from .control_suite import _eligible, compile_instance, _flow
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool
from .safe_face_offline import (Log, behaviour_policy, sample_log, learn_single, score, check_ship,
                                DUAL_ITERS, ETA_DUAL, TOL)
from .pipeline_compare import occ_lp_subset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "results/e2e")
OUT = os.path.join(R, "collapse_utility.json")
D_GRID = (0.05, 0.02)
N_GRID = (2000, 20000)
SEEDS = tuple(range(10))
RHOS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7)
EPS_GRID = (0.0, 1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5)
BETA = 1.0
RHO_TIGHT = 0.5
N_EV = 20000
DELTA_EV = 0.05


def lp_true(m, obj, rows, rhs):
    A_eq, b_eq = _flow(m)
    res = linprog(-obj, A_ub=np.array(rows) if rows else None, b_ub=np.array(rhs) if rows else None,
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return None if res.status != 0 else float(obj @ res.x)


def learn_fullset_lam(lg: Log, d_budget: float):
    """learn_fullset with the final multipliers and the number of dual iterations exposed."""
    lam = np.zeros(lg.K); best = None; iters = DUAL_ITERS
    for j in range(DUAL_ITERS):
        shaped = lg.r - np.tensordot(lam, lg.c, axes=(0, 0))
        pi = lg.greedy(lg.fqi(shaped)); _jr, jc = lg.est(pi)
        if (jc <= d_budget + TOL).all():
            best = pi; iters = j + 1; break
        lam = np.maximum(0.0, lam + ETA_DUAL * (jc - d_budget))
    return (best if best is not None else pi), lam, iters


def with_costs(lg: Log, c_new: np.ndarray) -> Log:
    lg2 = copy.copy(lg); lg2.c = c_new; lg2.K = c_new.shape[0]; return lg2


def main():
    sel = json.load(open(os.path.join(R, "safe_face_select.json")))
    reg = {(r["rule_id"], round(r["budget"], 6)): r for r in sel["rows"]}
    pt, _ = parse_prometheus(); bank = prom_threshold_bank(pt)
    exact, rows = [], []; t0 = time.time()
    for ti, t in enumerate(pt):
        ok, _w, rd = _eligible(build_prom_pool(t, bank))
        if not ok:
            continue
        rid = f'{getattr(t, "name", "?")}#{ti}'
        m = compile_instance(rd); P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]
        assert C.shape[0] == 2, rid
        rflat = r.reshape(-1); c0, c1 = C[0].reshape(-1), C[1].reshape(-1)
        dom01 = bool(np.all(C[0] >= C[1])); dom10 = bool(np.all(C[1] >= C[0]))
        cmax = np.maximum(C[0], C[1]).reshape(-1)
        for d in D_GRID:
            rr = reg[(rid, round(d, 6))]
            if rr["regime"] != "irrelevant":
                exact.append(dict(rule_id=rid, d=d, regime=rr["regime"], certified=False)); continue
            psi = rr["anchor"]; phi = 1 - psi; cpsi, cphi = (c0, c1) if psi == 0 else (c1, c0)
            V_U = rr["V_U"]
            # exact tightened values and surrogate
            tight = {}
            for rho in RHOS:
                dd = d * (1 - rho)
                tight[str(rho)] = dict(d=dd, V_psi=lp_true(m, rflat, [cpsi], [dd]), V_U=lp_true(m, rflat, [c0, c1], [dd, dd]),
                                       V_surr=lp_true(m, rflat, [cmax], [dd]))
            # tolerance radius: largest eps with max{c_phi x : c_psi x <= d, r x >= V_psi - eps} <= d
            V_psi = lp_true(m, rflat, [cpsi], [d]); contained = {}
            for eps in EPS_GRID:
                W = lp_true(m, cphi, [cpsi, -rflat], [d, -(V_psi - eps)])
                contained[str(eps)] = (None if W is None else bool(W <= d + 1e-9))
            W_full = lp_true(m, cphi, [cpsi], [d])
            exact.append(dict(rule_id=rid, d=d, regime="irrelevant", certified=True, psi=psi, V_U=V_U, V_psi=V_psi,
                              psi_dominates_pointwise=(dom10 if psi == 1 else dom01), phi_dominates_pointwise=(dom01 if psi == 1 else dom10),
                              budget_level_containment=bool(W_full <= d + 1e-9), W_full=W_full, contained_by_eps=contained, tightened=tight))
            # ---- learning arms
            pi_b = behaviour_policy(m, psi, d)
            for n in N_GRID:
                for seed in SEEDS:
                    rng = np.random.default_rng(abs(hash((rid, d, n, seed, "v48"))) % (2 ** 32))
                    base = sample_log(m, pi_b, n, rng); lg = Log(base)
                    lg_s = with_costs(lg, lg.c.max(axis=0, keepdims=True))                       # surrogate: one row, the pointwise max
                    infl = BETA / np.sqrt(np.maximum(lg.N, 1))
                    lg_p = with_costs(lg, np.minimum(1.0, lg.c + infl[None]))                  # pessimistic costs per reading
                    lg_ps = with_costs(lg, np.minimum(1.0, lg.c.max(axis=0, keepdims=True) + infl[None]))
                    pols = {}
                    pi_f, lam, iters = learn_fullset_lam(lg, d)
                    pols["fqi"] = dict(decide_single=learn_single(lg, psi, d), fullset=pi_f, surrogate=learn_single(lg_s, 0, d), permissive=learn_single(lg, phi, d))
                    fq_meta = dict(lam=lam.tolist(), iters=iters, lam_phi_positive=bool(lam[phi] > 0))
                    pols["lp"] = dict(decide_single=occ_lp_subset(lg, [psi], d), fullset=occ_lp_subset(lg, [0, 1], d),
                                      surrogate=occ_lp_subset(lg_s, [0], d), permissive=occ_lp_subset(lg, [phi], d))
                    dt = d * (1 - RHO_TIGHT)
                    pols["lp_tight"] = dict(decide_single=occ_lp_subset(lg, [psi], dt), fullset=occ_lp_subset(lg, [0, 1], dt),
                                            surrogate=occ_lp_subset(lg_s, [0], dt), permissive=occ_lp_subset(lg, [phi], dt))
                    pols["lp_pess"] = dict(decide_single=occ_lp_subset(lg_p, [psi], d), fullset=occ_lp_subset(lg_p, [0, 1], d),
                                           surrogate=occ_lp_subset(lg_ps, [0], d), permissive=occ_lp_subset(lg_p, [phi], d))
                    out = {}
                    for lc, ps in pols.items():
                        out[lc] = {}
                        for name, pi in ps.items():
                            if pi is None:
                                out[lc][name] = None; continue
                            s = score(P, mu0, r, C, pi, d)
                            crng = np.random.default_rng(abs(hash((rid, d, n, seed, lc, name, "chk48"))) % (2 ** 32))
                            ck = check_ship(P, mu0, C, pi, d, N_EV, crng, DELTA_EV)
                            out[lc][name] = dict(ret_frac=s["J_r"] / V_U, cmax_over_d=s["C_max"] / d, safe=s["safe"],
                                                 c_psi_over_d=float((C[psi] * __import__("saorl.benchmark_sg.safe_face_offline", fromlist=["occupancy"]).occupancy(P, mu0, pi)).sum() / d),
                                                 ship=ck["ship"])
                    out["fqi_meta"] = fq_meta
                    rows.append(dict(rule_id=rid, d=d, n=n, seed=seed, psi=psi, V_U=V_U, arms=out))
        print(f"  {rid:40s} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate
    agg = {}
    for d in D_GRID:
        cert = [e for e in exact if e["d"] == d and e["certified"]]
        ex = dict(n_certified=len(cert), n_total=sum(1 for e in exact if e["d"] == d),
                  psi_pointwise_dominant=int(sum(e["psi_dominates_pointwise"] for e in cert)),
                  phi_pointwise_dominant=int(sum(e["phi_dominates_pointwise"] for e in cert)),
                  budget_level_containment=int(sum(e["budget_level_containment"] for e in cert)),
                  tightened={})
        for rho in RHOS:
            gaps = [(e["tightened"][str(rho)]["V_psi"] - e["tightened"][str(rho)]["V_U"]) / e["V_U"] for e in cert
                    if e["tightened"][str(rho)]["V_psi"] is not None and e["tightened"][str(rho)]["V_U"] is not None]
            surr = [(e["tightened"][str(rho)]["V_U"] - e["tightened"][str(rho)]["V_surr"]) / e["tightened"][str(rho)]["V_U"] for e in cert
                    if e["tightened"][str(rho)]["V_surr"] is not None and e["tightened"][str(rho)]["V_U"]]
            ex["tightened"][str(rho)] = dict(n=len(gaps), value_gap_psi_minus_U_rel_median=(float(np.median(gaps)) if gaps else None),
                                             value_gap_max=(float(max(gaps)) if gaps else None), n_gap_positive=int(sum(g > 1e-9 for g in gaps)),
                                             surrogate_loss_rel_median=(float(np.median(surr)) if surr else None), surrogate_loss_rel_max=(float(max(surr)) if surr else None))
        agg[f"exact_d{d}"] = ex
        for n in N_GRID:
            sub = [x for x in rows if x["d"] == d and x["n"] == n]
            for lc in ("fqi", "lp", "lp_tight", "lp_pess"):
                a = {}
                for name in ("decide_single", "fullset", "surrogate", "permissive"):
                    S = [x["arms"][lc][name] for x in sub]; ok = [s for s in S if s is not None]
                    a[name] = dict(n=len(ok), infeasible=len(S) - len(ok),
                                   ret_frac_median=(float(np.median([s["ret_frac"] for s in ok])) if ok else None),
                                   ret_frac_mean=(float(np.mean([s["ret_frac"] for s in ok])) if ok else None),
                                   safe_frac=(float(np.mean([s["safe"] for s in ok])) if ok else None),
                                   cmax_over_d_median=(float(np.median([s["cmax_over_d"] for s in ok])) if ok else None),
                                   ship_frac=(float(np.mean([s["ship"] for s in ok])) if ok else None))
                pairs = [(x["arms"][lc]["decide_single"], x["arms"][lc]["fullset"]) for x in sub]
                pairs = [(s, f) for s, f in pairs if s is not None and f is not None]
                diff = np.array([s["ret_frac"] - f["ret_frac"] for s, f in pairs]) if pairs else np.array([])
                a["paired_single_minus_fullset"] = dict(n=int(diff.size), median=(float(np.median(diff)) if diff.size else None), mean=(float(diff.mean()) if diff.size else None),
                                                        frac_identical_1e6=(float(np.mean(np.abs(diff) < 1e-6)) if diff.size else None),
                                                        frac_within_1pct=(float(np.mean(np.abs(diff) < 0.01)) if diff.size else None),
                                                        frac_single_higher_1pct=(float(np.mean(diff > 0.01)) if diff.size else None),
                                                        frac_fullset_higher_1pct=(float(np.mean(diff < -0.01)) if diff.size else None),
                                                        p90=(float(np.percentile(diff, 90)) if diff.size else None), p10=(float(np.percentile(diff, 10)) if diff.size else None))
                spairs = [(x["arms"][lc]["decide_single"], x["arms"][lc]["surrogate"]) for x in sub]
                spairs = [(s, f) for s, f in spairs if s is not None and f is not None]
                sd = np.array([s["ret_frac"] - f["ret_frac"] for s, f in spairs]) if spairs else np.array([])
                a["paired_single_minus_surrogate"] = dict(n=int(sd.size), median=(float(np.median(sd)) if sd.size else None), mean=(float(sd.mean()) if sd.size else None),
                                                          frac_single_higher_1pct=(float(np.mean(sd > 0.01)) if sd.size else None))
                if lc == "fqi":
                    a["fullset_lam_phi_positive_frac"] = float(np.mean([x["arms"]["fqi_meta"]["lam_phi_positive"] for x in sub]))
                    a["fullset_iters_median"] = float(np.median([x["arms"]["fqi_meta"]["iters"] for x in sub]))
                agg[f"d{d}_n{n}_{lc}"] = a
    res = dict(registration="V48 (REGISTRATION_V28.md)", d_grid=list(D_GRID), n_grid=list(N_GRID), seeds=len(SEEDS), beta=BETA, rho_tight=RHO_TIGHT,
               eps_grid=list(EPS_GRID), aggregate=agg, exact=exact, rows=rows, seconds=round(time.time() - t0, 1))
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} in {res['seconds']} s")
    for k, a in agg.items():
        if k.startswith("exact"):
            print(f"== {k}: certified {a['n_certified']}/{a['n_total']} psi-dominant {a['psi_pointwise_dominant']} phi-dominant {a['phi_pointwise_dominant']} budget-level {a['budget_level_containment']}")
            for rho, v in a["tightened"].items():
                print(f"   rho {rho}: gap(psi-U) med {v['value_gap_psi_minus_U_rel_median']} max {v['value_gap_max']} n_pos {v['n_gap_positive']} | surrogate loss med {v['surrogate_loss_rel_median']} max {v['surrogate_loss_rel_max']}")
        else:
            print(f"== {k}")
            for name in ("decide_single", "fullset", "surrogate", "permissive"):
                v = a[name]; print(f"   {name:14s} ret med {v['ret_frac_median']} mean {v['ret_frac_mean']} safe {v['safe_frac']} cmax/d {v['cmax_over_d_median']} ship {v['ship_frac']} infeasible {v['infeasible']}")
            p = a["paired_single_minus_fullset"]; print(f"   single-fullset: med {p['median']} mean {p['mean']} identical {p['frac_identical_1e6']} within1% {p['frac_within_1pct']} single>1% {p['frac_single_higher_1pct']} fullset>1% {p['frac_fullset_higher_1pct']}")
            q = a["paired_single_minus_surrogate"]; print(f"   single-surrogate: med {q['median']} mean {q['mean']} single>1% {q['frac_single_higher_1pct']}")
            if "fullset_lam_phi_positive_frac" in a:
                print(f"   fullset lam_phi>0 frac {a['fullset_lam_phi_positive_frac']} iters med {a['fullset_iters_median']}")


if __name__ == "__main__":
    main()
