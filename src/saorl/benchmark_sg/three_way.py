"""V38 (Phase A): the apples-to-apples three-way table.

Same estimated model, same support, same occupancy solver, same single
Check.  Arms: single-reading LP (vertex and a random tie-break), full-set LP
(vertex and with a secondary min-worst-cost tie-break), and the safe-face
selector at the estimated price of the buffer.  Registration V38."""
from __future__ import annotations
import json, os
from typing import Dict, List, Optional
import numpy as np
from scipy.optimize import linprog
from .control_mdp import GAMMA
from .control_suite import _eligible, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool
from .safe_face_offline import Log, behaviour_policy, sample_log
from .safe_face_mixture import true_values
from .occupancy_arrow import bounds_x, BUDGET, N_TOTAL, SEEDS, DELTA_EV, M_TOTAL, SAFE_TOL, N_BOOT

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SEL = os.path.join(ROOT, "results/e2e", "safe_face_select.json")
OUT = os.path.join(ROOT, "results/e2e", "three_way.json")
RHOS = (0.0, 0.5, 0.7); TOL = 1e-7

def _sys(lg):
    nS, nA = lg.nS, lg.nA; n = nS * nA
    block = np.kron(np.eye(nS), np.ones((1, nA)))
    A_eq = block - GAMMA * np.transpose(lg.Phat, (2, 0, 1)).reshape(nS, n)
    b_eq = (1.0 - GAMMA) * lg.mu0
    bounds = [(0, None) if lg.support[s, a] else (0, 0) for s in range(nS) for a in range(nA)]
    return A_eq, b_eq, bounds, n

def est_max(lg, obj, rows, rhs):
    """max obj'x over the estimated occupancy polytope with support, s.t. rows x <= rhs."""
    A_eq, b_eq, bounds, n = _sys(lg)
    res = linprog(-obj, A_ub=np.array(rows) if rows else None, b_ub=np.array(rhs) if rows else None,
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return (None, None) if res.status != 0 else (float(obj @ res.x), res.x)

def est_epi(lg, rows, rhs, epi_rows, epi_rhs):
    """min t s.t. rows x <= rhs and epi_rows x - t <= epi_rhs."""
    A_eq, b_eq, bounds, n = _sys(lg)
    A_eq2 = np.hstack([A_eq, np.zeros((A_eq.shape[0], 1))])
    R = [np.concatenate([r, [0.0]]) for r in rows] + [np.concatenate([e, [-1.0]]) for e in epi_rows]
    h = list(rhs) + list(epi_rhs)
    obj = np.zeros(n + 1); obj[-1] = 1.0
    res = linprog(obj, A_ub=np.array(R), b_ub=np.array(h), A_eq=A_eq2, b_eq=b_eq,
                  bounds=bounds + [(None, None)], method="highs")
    return (None, None) if res.status != 0 else (float(res.x[-1]), res.x[:n])

def to_policy(lg, x):
    nS, nA = lg.nS, lg.nA; X = x.reshape(nS, nA); tot = X.sum(1)
    pi = np.zeros((nS, nA)); vis = tot > 1e-12; pi[vis] = X[vis] / tot[vis, None]
    for s in np.where(~vis)[0]:
        pi[s, int(np.argmax(np.where(lg.support[s], lg.N[s], -1)))] = 1.0
    return pi

def main():
    sel = json.load(open(SEL)); mid = [r for r in sel["rows"] if r["regime"] == "optimizer_resolvable" and abs(r["budget"] - BUDGET) < 1e-12]
    pt, _ = parse_prometheus(); tb = prom_threshold_bank(pt); comp = {}
    for ti, t in enumerate(pt):
        ok, _w, rd = _eligible(build_prom_pool(t, tb))
        if ok: comp[f'{getattr(t, "name", "?")}#{ti}'] = compile_instance(rd)
    rows: List[dict] = []; n_equiv = 0; n_equiv_checked = 0
    for row in mid:
        m = comp[row["rule_id"]]; P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]; K = C.shape[0]
        r_flat = r.reshape(-1); C_flat = np.stack([C[k].reshape(-1) for k in range(K)])
        a = row["anchor"]; V_U = row["V_U"]; pi_b = behaviour_policy(m, a, BUDGET); d = BUDGET
        for seed in SEEDS:
            rng = np.random.default_rng(abs(hash((row["rule_id"], N_TOTAL, seed))) % (2 ** 32))
            lg = Log(sample_log(m, pi_b, N_TOTAL, rng))
            ev = np.random.default_rng(abs(hash((row["rule_id"], N_TOTAL, seed, "v38"))) % (2 ** 32))
            rb = lg.r.reshape(-1); cb = [lg.c[k].reshape(-1) for k in range(K)]
            out: Dict[str, dict] = {}
            def rec(name, pi, extra=None):
                if pi is None: out[name] = dict(feasible=False, ship=False, unsafe_ship=False, U=0.0, ret=None); return
                R, Cs, x = true_values(P, mu0, r, C, pi); b = bounds_x(x.reshape(-1), r_flat, C_flat, M_TOTAL, ev, DELTA_EV, DELTA_EV)
                ship = b["ucb_c"] <= d; cm = float(Cs.max())
                o = dict(feasible=True, ship=bool(ship), unsafe_ship=bool(ship and cm > d + SAFE_TOL), ret=R / V_U, U=(R / V_U if ship else 0.0),
                         C_over_d=cm / d, margin_over_d=(d - cm) / d, safe=bool(cm <= d + SAFE_TOL),
                         other_over_d=float(Cs[1 - a]) / d if K == 2 else None)
                if extra: o.update(extra)
                out[name] = o
            for rf in RHOS:
                dr = d * (1 - rf)
                # single reading: vertex and random tie-break on the estimated face
                Va, xa = est_max(lg, rb, [cb[a]], [dr]); rec(f"single_rho{rf}", to_policy(lg, xa) if xa is not None else None)
                if Va is not None:
                    sigma = rng.standard_normal(rb.size)
                    _v, xt = est_max(lg, sigma, [cb[a], -rb], [dr, -(Va - TOL)]); rec(f"single_tie_rho{rf}", to_policy(lg, xt) if xt is not None else None)
                else: rec(f"single_tie_rho{rf}", None)
                # full set: vertex, and lexicographic secondary min worst cost
                VU, xu = est_max(lg, rb, cb, [dr] * K); rec(f"fullset_rho{rf}", to_policy(lg, xu) if xu is not None else None)
                if VU is not None:
                    tl, xl = est_epi(lg, cb + [-rb], [dr] * K + [-(VU - TOL)], cb, [dr] * K)
                    rec(f"fullset_lex_rho{rf}", to_policy(lg, xl) if xl is not None else None, dict(t_star=tl))
                else: rec(f"fullset_lex_rho{rf}", None)
                # safe face at the estimated price of the buffer
                Vad, _ = est_max(lg, rb, [cb[a]], [d])
                if Vad is None or VU is None:
                    rec(f"safeface_rho{rf}", to_policy(lg, xu) if xu is not None else None, dict(fellback=True, eps_hat=None))
                else:
                    eps_hat = max(0.0, Vad - VU)
                    ts, xs = est_epi(lg, [cb[a], -rb], [d, -(Vad - eps_hat - TOL)], cb, [dr] * K)
                    if xs is not None and ts <= 1e-7:
                        rec(f"safeface_rho{rf}", to_policy(lg, xs), dict(fellback=False, eps_hat=eps_hat, t_star=ts))
                        # the stated equivalence: when accepted, its (return, worst cost) on P_hat equal fullset_lex's
                        if out[f"fullset_lex_rho{rf}"]["feasible"]:
                            n_equiv_checked += 1
                            if abs(float(rb @ xs) - VU) <= 1e-6 and abs(ts - out[f"fullset_lex_rho{rf}"]["t_star"]) <= 1e-6: n_equiv += 1
                    else:
                        rec(f"safeface_rho{rf}", to_policy(lg, xu) if xu is not None else None, dict(fellback=True, eps_hat=eps_hat, t_star=ts))
            rows.append(dict(rule_id=row["rule_id"], seed=seed, V_U=V_U, arms=out))
        print(f"  {row['rule_id']:38s} done", flush=True)
    json.dump(dict(rows=rows), open(OUT.replace(".json", "_rows.json"), "w"))
    brng = np.random.default_rng(38)
    def ci(diff):
        diff = np.asarray(diff); n = len(diff); bs = np.array([diff[brng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    names = sorted({k for x in rows for k in x["arms"]}); agg = {}
    for nm in names:
        rf = nm.split("rho")[-1]; base = f"fullset_rho{rf}"
        L = [x["arms"][nm] for x in rows]; B = [x["arms"][base] for x in rows]
        feas = [l for l in L if l["feasible"]]
        U = np.array([l["U"] for l in L]); Ub = np.array([b["U"] for b in B])
        pair = [(l, b) for l, b in zip(L, B) if l["feasible"] and b["feasible"]]
        dret = [l["ret"] - b["ret"] for l, b in pair]; dmar = [l["margin_over_d"] - b["margin_over_d"] for l, b in pair]
        agg[nm] = dict(n=len(L), feasible=float(np.mean([l["feasible"] for l in L])),
                       ship=float(np.mean([l["ship"] for l in L])), unsafe_ship=float(np.mean([l["unsafe_ship"] for l in L])),
                       safe_true=float(np.mean([l["safe"] for l in feas])) if feas else None,
                       ret_med=float(np.median([l["ret"] for l in feas])) if feas else None,
                       C_over_d_med=float(np.median([l["C_over_d"] for l in feas])) if feas else None,
                       other_over_d_med=float(np.median([l["other_over_d"] for l in feas if l["other_over_d"] is not None])) if feas else None,
                       U_mean=float(U.mean()), dU_vs_fullset=float((U - Ub).mean()), dU_ci=ci(U - Ub),
                       dret_vs_fullset_mean=(float(np.mean(dret)) if dret else None), dret_abs_max=(float(np.max(np.abs(dret))) if dret else None),
                       dmargin_vs_fullset_mean=(float(np.mean(dmar)) if dmar else None), dmargin_ci=(ci(dmar) if len(dmar) > 10 else None),
                       margin_strictly_better_frac=(float(np.mean(np.array(dmar) > 1e-9)) if dmar else None),
                       fellback=(float(np.mean([l.get("fellback", False) for l in L])) if "safeface" in nm else None))
    res = dict(registration="V38 (REGISTRATION_V28.md)", n_instances=len(mid), n_seeds=len(SEEDS), rhos=list(RHOS),
               safeface_equiv_fullset_lex=dict(checked=n_equiv_checked, equal=n_equiv), aggregate=agg, rows=rows)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}; safeface == fullset_lex on P_hat when accepted: {n_equiv}/{n_equiv_checked}")
    print(f"{'arm':22s} {'feas':>5} {'SHIP':>6} {'unsafe':>6} {'safeT':>6} {'ret':>6} {'C/d':>6} {'other/d':>7} {'U':>6} {'dU vs FS[CI]':>22} {'dret':>8} {'|dret|max':>9} {'dmargin':>8} {'m.better%':>9} {'fb':>5}")
    for nm in names:
        v = agg[nm]; f = lambda z, w=6, p=3: (" " * (w - 2) + "--") if z is None else f"{z:>{w}.{p}f}"
        print(f"{nm:22s} {v['feasible']:5.2f} {v['ship']:6.3f} {v['unsafe_ship']:6.3f} {f(v['safe_true'])} {f(v['ret_med'])} {f(v['C_over_d_med'])} {f(v['other_over_d_med'],7)} {v['U_mean']:6.3f} "
              f"{100*v['dU_vs_fullset']:+6.2f} [{100*v['dU_ci'][0]:+.1f},{100*v['dU_ci'][1]:+.1f}] {f(v['dret_vs_fullset_mean'],8,4)} {f(v['dret_abs_max'],9,4)} {f(v['dmargin_vs_fullset_mean'],8,4)} {f(v['margin_strictly_better_frac'],9,2)} {f(v['fellback'],5,2)}")

if __name__ == "__main__":
    main()
