"""V32: two diagnostics on the representation-mismatch hypothesis.

1. Does the exact safe-face optimum randomize?  Recover pi*(a|s) from the LP
   occupancy, count randomizing states, compute occupancy-weighted entropy,
   and round to the argmax to see whether a deterministic policy can hold the
   point at all.

2. Oracle convex hull of the ALREADY-TRAINED arms.  For every (instance, seed)
   at N = 20000 in the V31 archive, solve the mixture LP over the eight arms'
   true (J_r, C_max) with the constraint sum_m w_m C_max,m <= d - rho.  The
   true model is used only as an oracle; nothing is retrained.

Registration: results/e2e/REGISTRATION_V28.md (V32 addendum).
Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.mixture_diagnostic
Writes results/e2e/mixture_diagnostic.json
"""
from __future__ import annotations

import json
import os
from typing import List

import numpy as np
from scipy.optimize import linprog

from .control_suite import _eligible, _flow, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool
from .safe_face_offline import occupancy, score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SEL = os.path.join(ROOT, "results/e2e", "safe_face_select.json")
OFF = os.path.join(ROOT, "results/e2e", "safe_face_offline.json")
OUT = os.path.join(ROOT, "results/e2e", "mixture_diagnostic.json")

BUDGET = 0.005
EPS = 0.01
KAPPA_FRAC = (0.0, 0.1, 0.2, 0.3, 0.5)
RHO_FRAC = (0.1, 0.2, 0.3, 0.5)
ARMS = ["single", "fullset", "safeface_eps0.01",
        "margin_k0.0", "margin_k0.1", "margin_k0.2", "margin_k0.3",
        "margin_k0.5"]
VISIT_TOL = 1e-9


def _lp_x(m, obj, rows, rhs):
    A_eq, b_eq = _flow(m)
    res = linprog(-obj, A_ub=np.array(rows) if rows else None,
                  b_ub=np.array(rhs) if rows else None,
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return None if res.status != 0 else res.x


def _safe_face_x(m, r, crows, k, d, V_k, eps):
    A_eq, b_eq = _flow(m)
    n = A_eq.shape[1]
    A_eq2 = np.hstack([A_eq, np.zeros((A_eq.shape[0], 1))])
    rows = [np.concatenate([crows[k], [0.0]]), np.concatenate([-r, [0.0]])]
    rhs = [d, -(V_k - eps - 1e-7)]
    for c in crows:
        rows.append(np.concatenate([c, [-1.0]]))
        rhs.append(d)
    obj = np.zeros(n + 1)
    obj[-1] = 1.0
    res = linprog(obj, A_ub=np.array(rows), b_ub=np.array(rhs),
                  A_eq=A_eq2, b_eq=b_eq,
                  bounds=[(0, None)] * n + [(None, None)], method="highs")
    return None if res.status != 0 else res.x[:n]


def policy_stats(m, x) -> dict:
    """Recover pi from an occupancy and describe how much it randomizes."""
    nS, nA = m["nS"], m["nA"]
    X = x.reshape(nS, nA)
    tot = X.sum(axis=1)
    visited = tot > VISIT_TOL
    pi = np.full((nS, nA), 1.0 / nA)
    pi[visited] = X[visited] / tot[visited, None]
    # states where more than one action carries occupancy
    n_rand = int(sum(1 for s in range(nS)
                     if visited[s] and (X[s] > VISIT_TOL).sum() > 1))
    # occupancy-weighted entropy in nats
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(pi * np.log(np.where(pi > 0, pi, 1.0))).sum(axis=1)
    ent = float((tot[visited] * H[visited]).sum() / tot[visited].sum())
    # deterministic rounding
    pi_det = np.zeros_like(pi)
    pi_det[np.arange(nS), pi.argmax(axis=1)] = 1.0
    return dict(n_visited=int(visited.sum()), n_randomizing=n_rand,
                entropy=ent, pi=pi, pi_det=pi_det)


def main() -> None:
    sel = json.load(open(SEL, encoding="utf8"))
    off = json.load(open(OFF, encoding="utf8"))
    mid = [r for r in sel["rows"]
           if r["regime"] == "optimizer_resolvable"
           and abs(r["budget"] - BUDGET) < 1e-12]

    pt, _ = parse_prometheus()
    bank = prom_threshold_bank(pt)
    compiled = {}
    for ti, t in enumerate(pt):
        ok, _w, readings = _eligible(build_prom_pool(t, bank))
        if ok:
            compiled[f'{getattr(t, "name", "?")}#{ti}'] = compile_instance(readings)

    # ---------------- Diagnostic 1: randomization -----------------------
    d1: List[dict] = []
    for row in mid:
        m = compiled[row["rule_id"]]
        P, mu0, R, C = m["P"], m["mu0"], m["r"], m["C"]
        r = R.reshape(-1)
        K = C.shape[0]
        crows = [C[k].reshape(-1) for k in range(K)]
        a = row["anchor"]
        rec = dict(rule_id=row["rule_id"], V_U=row["V_U"])
        # safe-face optimum at kappa = 0
        x = _safe_face_x(m, r, crows, a, BUDGET, row["V"][a], 0.0)
        st = policy_stats(m, x)
        exact = score(P, mu0, R, C, st["pi"], BUDGET)
        rounded = score(P, mu0, R, C, st["pi_det"], BUDGET)
        rec["safeface"] = dict(
            n_visited=st["n_visited"], n_randomizing=st["n_randomizing"],
            entropy=st["entropy"],
            exact=dict(ret=exact["J_r"] / row["V_U"], C_over_d=exact["C_max"] / BUDGET,
                       safe=exact["safe"]),
            rounded=dict(ret=rounded["J_r"] / row["V_U"],
                         C_over_d=rounded["C_max"] / BUDGET, safe=rounded["safe"]))
        # tightened full-set optima along the kappa grid
        rec["fullset_kappa"] = {}
        for kf in KAPPA_FRAC:
            xt = _lp_x(m, r, crows, [BUDGET * (1 - kf)] * K)
            if xt is None:
                continue
            st = policy_stats(m, xt)
            ex = score(P, mu0, R, C, st["pi"], BUDGET)
            ro = score(P, mu0, R, C, st["pi_det"], BUDGET)
            rec["fullset_kappa"][str(kf)] = dict(
                n_randomizing=st["n_randomizing"], entropy=st["entropy"],
                exact=dict(ret=ex["J_r"] / row["V_U"], C_over_d=ex["C_max"] / BUDGET),
                rounded=dict(ret=ro["J_r"] / row["V_U"],
                             C_over_d=ro["C_max"] / BUDGET, safe=ro["safe"]))
        d1.append(rec)

    # ---------------- Diagnostic 2: oracle convex hull ------------------
    big = off["n_grid"][-1]
    runs = [x for x in off["rows"] if x["n"] == big]
    d2: List[dict] = []
    for run in runs:
        L = run["learners"]
        Rm = np.array([L[a]["J_r"] for a in ARMS])
        Cm = np.array([L[a]["C_max"] for a in ARMS])
        full_ret = L["fullset"]["J_r"]
        rec = dict(rule_id=run["rule_id"], seed=run["seed"], V_U=run["V_U"],
                   fullset_ret=full_ret / run["V_U"],
                   fullset_C_over_d=L["fullset"]["C_max"] / BUDGET, hull={})
        for rf in RHO_FRAC:
            # max w.R  s.t.  w.C <= d - rho,  sum w = 1,  w >= 0
            res = linprog(-Rm, A_ub=np.array([Cm]),
                          b_ub=np.array([BUDGET * (1 - rf)]),
                          A_eq=np.ones((1, len(ARMS))), b_eq=np.array([1.0]),
                          bounds=[(0, 1)] * len(ARMS), method="highs")
            if res.status != 0:
                rec["hull"][str(rf)] = None
                continue
            w = res.x
            rec["hull"][str(rf)] = dict(
                ret=float(w @ Rm) / run["V_U"],
                C_over_d=float(w @ Cm) / BUDGET,
                gain_over_fullset=float(w @ Rm - full_ret) / run["V_U"],
                weights={a: float(wi) for a, wi in zip(ARMS, w) if wi > 1e-6})
        d2.append(rec)

    # ---------------- aggregate -----------------------------------------
    agg1 = dict(
        n_instances=len(d1),
        safeface_randomizing=sum(1 for r in d1 if r["safeface"]["n_randomizing"] > 0),
        safeface_n_rand_med=float(np.median([r["safeface"]["n_randomizing"] for r in d1])),
        safeface_entropy_med=float(np.median([r["safeface"]["entropy"] for r in d1])),
        safeface_rounded_safe=sum(1 for r in d1 if r["safeface"]["rounded"]["safe"]),
        safeface_rounded_C_over_d_med=float(np.median(
            [r["safeface"]["rounded"]["C_over_d"] for r in d1])),
        safeface_rounded_ret_med=float(np.median(
            [r["safeface"]["rounded"]["ret"] for r in d1])),
        fullset_kappa={})
    for kf in KAPPA_FRAC:
        rs = [r["fullset_kappa"][str(kf)] for r in d1 if str(kf) in r["fullset_kappa"]]
        agg1["fullset_kappa"][str(kf)] = dict(
            randomizing=sum(1 for x in rs if x["n_randomizing"] > 0),
            rounded_safe=sum(1 for x in rs if x["rounded"]["safe"]),
            rounded_C_over_d_med=float(np.median([x["rounded"]["C_over_d"] for x in rs])),
            rounded_ret_med=float(np.median([x["rounded"]["ret"] for x in rs])),
            exact_ret_med=float(np.median([x["exact"]["ret"] for x in rs])))
    agg2 = {}
    for rf in RHO_FRAC:
        hs = [r["hull"][str(rf)] for r in d2 if r["hull"].get(str(rf))]
        gains = [h["gain_over_fullset"] for h in hs]
        agg2[str(rf)] = dict(
            n_feasible=len(hs), n_runs=len(d2),
            beats_fullset_frac=float(np.mean([g > 1e-9 for g in gains])) if gains else None,
            gain_med=float(np.median(gains)) if gains else None,
            gain_p25=float(np.percentile(gains, 25)) if gains else None,
            gain_p75=float(np.percentile(gains, 75)) if gains else None,
            hull_ret_med=float(np.median([h["ret"] for h in hs])) if hs else None,
            hull_C_over_d_med=float(np.median([h["C_over_d"] for h in hs])) if hs else None)
    res = dict(registration="V32 (REGISTRATION_V28.md addendum)",
               budget=BUDGET, arms=ARMS, rho_frac=list(RHO_FRAC),
               diagnostic1=agg1, diagnostic2=agg2,
               rows1=[{k: v for k, v in r.items()} for r in d1],
               rows2=d2)
    # strip numpy arrays before dumping
    for r in res["rows1"]:
        r["safeface"] = {k: v for k, v in r["safeface"].items() if k not in ("pi", "pi_det")}
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(res, fh, indent=1, default=float)

    print("=== Diagnostic 1: does the exact optimum randomize? ===")
    print(f"safe-face optimum (kappa=0): {agg1['safeface_randomizing']}/{agg1['n_instances']} "
          f"instances randomize; median randomizing states "
          f"{agg1['safeface_n_rand_med']:.0f}; weighted entropy {agg1['safeface_entropy_med']:.3f} nats")
    print(f"  argmax rounding: safe on {agg1['safeface_rounded_safe']}/{agg1['n_instances']}, "
          f"C_max {agg1['safeface_rounded_C_over_d_med']:.2f}x d, return "
          f"{agg1['safeface_rounded_ret_med']:.3f} of V_U")
    print(f"{'kappa/d':>8} {'randomize':>10} {'rounded safe':>13} {'rounded C/d':>12} {'rounded ret':>12} {'exact ret':>10}")
    for kf in KAPPA_FRAC:
        v = agg1["fullset_kappa"][str(kf)]
        print(f"{kf:>8} {v['randomizing']:>10} {v['rounded_safe']:>13} "
              f"{v['rounded_C_over_d_med']:>12.2f} {v['rounded_ret_med']:>12.3f} {v['exact_ret_med']:>10.3f}")
    print()
    print("=== Diagnostic 2: oracle convex hull of the trained arms (N=20000) ===")
    print(f"{'rho/d':>6} {'feasible':>9} {'beats fullset':>14} {'gain med':>9} "
          f"{'[p25,p75]':>16} {'hull ret':>9} {'hull C/d':>9}")
    for rf in RHO_FRAC:
        v = agg2[str(rf)]
        print(f"{rf:>6} {v['n_feasible']:>4}/{v['n_runs']:<4} "
              f"{v['beats_fullset_frac']:>14.3f} {v['gain_med']:>+9.4f} "
              f"[{v['gain_p25']:+.4f},{v['gain_p75']:+.4f}] "
              f"{v['hull_ret_med']:>9.3f} {v['hull_C_over_d_med']:>9.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
