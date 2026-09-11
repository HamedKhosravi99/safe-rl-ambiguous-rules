"""V28: the safe-face selector on the compiled suite (registration V28).

V13 measured whether a safe optimum EXISTS (Delta <= 0) and whether every
optimum is safe (Gamma <= 0). It never ran a selector. This module does:
on every instance and budget it recomputes (Delta, Gamma) directly, labels
the regime, and then compares five ways of choosing a point on the anchor's
optimal face, all scored against the COMPLETE retained set.

The safe-face selector is the epigraph LP of the paper's Eq. (12):

    min_{x,t} t  s.t.  flow(x),  c_psi'x <= d,  r'x >= V_psi - eps,
                       c_phi'x - d <= t  for every retained phi,

whose optimum is exactly Delta_psi^eps.

Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.safe_face_select
Writes results/e2e/safe_face_select.json
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from .control_suite import _eligible, _flow, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
V13 = os.path.join(ROOT, "results/e2e", "policy_sufficiency.json")
OUT = os.path.join(ROOT, "results/e2e", "safe_face_select.json")

BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10)
TOL = 1e-7
FACE_TOL = 1e-7
SAFE_TOL = 1e-6
N_TIEBREAK = 100
SEED = 20260904
EPS_GRID = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


def _lp(m, obj, rows, rhs, maximize=True):
    """Optimize obj over the flow polytope subject to rows @ x <= rhs."""
    A_eq, b_eq = _flow(m)
    res = linprog(-obj if maximize else obj,
                  A_ub=np.array(rows) if rows else None,
                  b_ub=np.array(rhs) if rows else None,
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if res.status != 0:
        return None, None
    return float(obj @ res.x), res.x


def _safe_face_lp(m, r, crows, k, d, V_k, eps) -> Tuple[Optional[float],
                                                        Optional[np.ndarray]]:
    """min t s.t. x in F_k^eps and c_phi'x - d <= t for every phi.

    Returns (Delta, x). Variables are z = [x, t] with t free.
    """
    A_eq, b_eq = _flow(m)
    n = A_eq.shape[1]
    A_eq2 = np.hstack([A_eq, np.zeros((A_eq.shape[0], 1))])
    rows: List[np.ndarray] = []
    rhs: List[float] = []
    rows.append(np.concatenate([crows[k], [0.0]]))          # c_k'x <= d
    rhs.append(d)
    rows.append(np.concatenate([-r, [0.0]]))                # r'x >= V_k - eps
    rhs.append(-(V_k - eps - FACE_TOL))
    for c in crows:                                         # c_phi'x - t <= d
        rows.append(np.concatenate([c, [-1.0]]))
        rhs.append(d)
    obj = np.zeros(n + 1)
    obj[-1] = 1.0
    res = linprog(obj, A_ub=np.array(rows), b_ub=np.array(rhs),
                  A_eq=A_eq2, b_eq=b_eq,
                  bounds=[(0, None)] * n + [(None, None)], method="highs")
    if res.status != 0:
        return None, None
    return float(res.x[-1]), res.x[:n]


def _score(x, r, crows, d) -> dict:
    """Score an occupancy against the complete retained set."""
    if x is None:
        return dict(J_r=None, C_max=None, margin=None, safe=False)
    cmax = max(float(c @ x) for c in crows)
    return dict(J_r=float(r @ x), C_max=cmax, margin=float(d - cmax),
                safe=bool(cmax <= d + SAFE_TOL))


def main() -> None:
    pt, _ = parse_prometheus()
    bank = prom_threshold_bank(pt)
    rng = np.random.default_rng(SEED)
    rows_out: List[dict] = []

    for ti, t in enumerate(pt):
        pool = build_prom_pool(t, bank)
        ok, _why, readings = _eligible(pool)
        if not ok:
            continue
        uid = f'{getattr(t, "name", "?")}#{ti}'
        m = compile_instance(readings)
        r = m["r"].reshape(-1)
        K = m["C"].shape[0]
        crows = [m["C"][k].reshape(-1) for k in range(K)]

        for d in BUDGETS:
            V_U, x_U = _lp(m, r, crows, [d] * K)
            if V_U is None:
                continue
            V = []
            for k in range(K):
                v, _ = _lp(m, r, [crows[k]], [d])
                V.append(v)
            if any(v is None for v in V):
                continue

            # --- regime, recomputed from (Delta, Gamma) directly -----------
            delta, gamma = [], []
            for k in range(K):
                dl, _ = _safe_face_lp(m, r, crows, k, d, V[k], 0.0)
                delta.append(dl)
                if K == 1:
                    gamma.append(-d)          # single reading: face is safe
                    continue
                W = -np.inf
                for kk in range(K):
                    if kk == k:
                        continue
                    w, _ = _lp(m, crows[kk], [crows[k], -r],
                               [d, -(V[k] - FACE_TOL)])
                    if w is None:
                        W = np.inf
                        break
                    W = max(W, w)
                gamma.append(None if np.isinf(W) else float(W - d))
            has_delta = any(dl is not None and dl <= SAFE_TOL for dl in delta)
            has_gamma = any(g is not None and g <= SAFE_TOL for g in gamma)
            regime = ("irrelevant" if has_gamma else
                      "optimizer_resolvable" if has_delta else "irreducible")

            # --- frozen anchor: first value-sufficient reading in pool order
            vsuff = [k for k in range(K) if V[k] - V_U <= TOL]
            anchor = vsuff[0] if vsuff else None

            sel: dict = {}
            if anchor is not None:
                a = anchor
                # 1. single-reading, the solver's own vertex
                _v, x_s = _lp(m, r, [crows[a]], [d])
                sel["single"] = _score(x_s, r, crows, d)
                # 2. random tie-breaks on the anchor's face
                mars, safes, cmaxs = [], 0, []
                for _s in range(N_TIEBREAK):
                    sigma = rng.standard_normal(r.shape[0])
                    _v, x_t = _lp(m, sigma, [crows[a], -r],
                                  [d, -(V[a] - FACE_TOL)])
                    if x_t is None:
                        continue
                    sc = _score(x_t, r, crows, d)
                    mars.append(sc["margin"])
                    cmaxs.append(sc["C_max"])
                    safes += sc["safe"]
                sel["tie_random"] = dict(
                    n=len(mars), safe_frac=(safes / len(mars)) if mars else None,
                    margin_med=(float(np.median(mars)) if mars else None),
                    margin_min=(min(mars) if mars else None),
                    C_max_med=(float(np.median(cmaxs)) if cmaxs else None))
                # 3. adversarial tie-break = the face-worst point
                sel["tie_worst"] = dict(
                    C_max=(None if gamma[a] is None else gamma[a] + d),
                    margin=(None if gamma[a] is None else -gamma[a]),
                    safe=bool(gamma[a] is not None and gamma[a] <= SAFE_TOL))
                # 4. full-set: the solver's vertex, and the best/worst point
                #    on the full-set-optimal face (an indifferent learner may
                #    return any of them)
                sel["fullset"] = _score(x_U, r, crows, d)
                # best case: min worst-cost over the FULL-SET optimal face
                A_eq, b_eq = _flow(m)
                n = A_eq.shape[1]
                A_eq2 = np.hstack([A_eq, np.zeros((A_eq.shape[0], 1))])
                rws = [np.concatenate([-r, [0.0]])]
                rhs = [-(V_U - FACE_TOL)]
                for c in crows:
                    rws.append(np.concatenate([c, [0.0]]))
                    rhs.append(d)
                    rws.append(np.concatenate([c, [-1.0]]))
                    rhs.append(d)
                obj = np.zeros(n + 1)
                obj[-1] = 1.0
                rb = linprog(obj, A_ub=np.array(rws), b_ub=np.array(rhs),
                             A_eq=A_eq2, b_eq=b_eq,
                             bounds=[(0, None)] * n + [(None, None)],
                             method="highs")
                sel["fullset_best"] = (_score(rb.x[:n], r, crows, d)
                                       if rb.status == 0 else
                                       dict(J_r=None, C_max=None, margin=None,
                                            safe=False))
                # worst case: max worst-cost over the full-set optimal face
                w_worst = -np.inf
                for c in crows:
                    w, _ = _lp(m, c, [*crows, -r], [*([d] * K),
                                                    -(V_U - FACE_TOL)])
                    if w is None:
                        w_worst = np.inf
                        break
                    w_worst = max(w_worst, w)
                sel["fullset_worst"] = dict(
                    C_max=(None if np.isinf(w_worst) else float(w_worst)),
                    margin=(None if np.isinf(w_worst) else float(d - w_worst)),
                    safe=bool(not np.isinf(w_worst) and
                              w_worst <= d + SAFE_TOL))
                # 5. safe-face selector at eps = 0
                dl, x_sf = _safe_face_lp(m, r, crows, a, d, V[a], 0.0)
                sc = _score(x_sf, r, crows, d)
                sc["Delta"] = dl
                sc["reward_kept"] = bool(sc["J_r"] is not None and
                                         sc["J_r"] >= V[a] - 1e-6)
                sel["safeface"] = sc

                # theorem assertions on the middle regime
                if regime == "optimizer_resolvable":
                    assert sc["safe"], (uid, d, "safe-face selector returned "
                                        "an unsafe policy where Delta<=0")
                    assert sc["reward_kept"], (uid, d, "safe-face selector "
                                               "gave up anchor-optimal reward")

            # --- eps sweep: |R_eps|, free from the V values ----------------
            reps = {str(e): int(sum(1 for k in range(K)
                                    if V[k] - V_U <= e + TOL))
                    for e in EPS_GRID}

            rows_out.append(dict(
                rule_id=uid, budget=d, K=K, V_U=V_U, V=V,
                anchor=anchor, regime=regime,
                delta=delta, gamma=gamma,
                selectors=sel, R_eps=reps))

    # ---- aggregate ------------------------------------------------------
    by_budget = {}
    for d in BUDGETS:
        rs = [x for x in rows_out if x["budget"] == d]
        cnt = {k: sum(1 for x in rs if x["regime"] == k)
               for k in ("irrelevant", "optimizer_resolvable", "irreducible")}
        mid = [x for x in rs if x["regime"] == "optimizer_resolvable"]
        def _m(rows, path):
            out = []
            for x in rows:
                v = x["selectors"]
                for p in path:
                    v = v.get(p) if isinstance(v, dict) else None
                    if v is None:
                        break
                if v is not None:
                    out.append(v)
            return out
        sf_m = _m(mid, ("safeface", "margin"))
        fw_m = _m(mid, ("fullset_worst", "margin"))
        fb_m = _m(mid, ("fullset_best", "margin"))
        tr_s = _m(mid, ("tie_random", "safe_frac"))
        sg_s = _m(mid, ("single", "safe"))
        by_budget[str(d)] = dict(
            n=len(rs), regimes=cnt, n_middle=len(mid),
            safeface_safe=sum(1 for x in mid
                              if x["selectors"]["safeface"]["safe"]),
            single_safe=int(sum(sg_s)),
            tie_random_safe_frac_med=(float(np.median(tr_s)) if tr_s else None),
            tie_random_safe_frac_min=(min(tr_s) if tr_s else None),
            margin_safeface_med=(float(np.median(sf_m)) if sf_m else None),
            margin_fullset_worst_med=(float(np.median(fw_m)) if fw_m else None),
            margin_fullset_best_med=(float(np.median(fb_m)) if fb_m else None),
            slack_gain_vs_worst_med=(
                float(np.median([a - b for a, b in zip(sf_m, fw_m)]))
                if sf_m and len(sf_m) == len(fw_m) else None),
            slack_gain_vs_best_med=(
                float(np.median([a - b for a, b in zip(sf_m, fb_m)]))
                if sf_m and len(sf_m) == len(fb_m) else None))

    res = dict(
        registration="V28 (REGISTRATION_V28.md): safe-face selector vs "
                     "single-reading, random and adversarial tie-breaks and "
                     "full-set optimization; anchor frozen as the first "
                     f"value-sufficient reading; tie-break seed {SEED}, "
                     f"N={N_TIEBREAK}",
        budgets=list(BUDGETS), eps_grid=list(EPS_GRID),
        by_budget=by_budget, rows=rows_out)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {OUT} ({len(rows_out)} rows)")
    print(json.dumps(by_budget, indent=1))


if __name__ == "__main__":
    main()
