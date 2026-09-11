"""V33: Safe-Face Mixture -- the learner the V32 diagnostics point to.

V31 showed a single deterministic learner cannot hold the interior point the
exact theory says exists: full-set lands far inside the budget, safe-face
lands past it. V32 showed the convex hull of the trained arms contains that
point on 99.8% of seeds. This module makes that a learner:

  1. split the log into D_train / D_select;
  2. train a bank of deterministic arms on D_train;
  3. estimate every arm's return and per-reading costs on D_select (model
     based, no true model);
  4. solve the mixture LP  max_w w.R_hat  s.t.  w.C_hat_phi <= d - rho  for
     every retained reading, w in the simplex;
  5. deploy the EPISODE-LEVEL mixture (m ~ w, run pi_m for the episode),
     whose occupancy is the w-mixture of the components', and put it through
     the independent Check.

Full-set is a member of the bank, so the LP is feasible whenever full-set's
own estimate is and the fallback is continuous rather than binary.

Registration: results/e2e/REGISTRATION_V28.md (V33 addendum).
Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.safe_face_mixture
Writes results/e2e/safe_face_mixture.json
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
from scipy.optimize import linprog

from .control_suite import _eligible, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool
from .safe_face_offline import (Log, behaviour_policy, learn_fullset,
                                learn_safeface, learn_single, occupancy,
                                sample_log)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SEL = os.path.join(ROOT, "results/e2e", "safe_face_select.json")
OUT = os.path.join(ROOT, "results/e2e", "safe_face_mixture.json")

BUDGET = 0.005
EPS = 0.01
N_TOTAL = 20000
SEEDS = tuple(range(30))
KAPPA_FRAC = (0.0, 0.1, 0.2, 0.3, 0.5)
RHO_FRAC = (0.1, 0.2, 0.3, 0.5)
N_EV = 20000
DELTA_EV = 0.05
N_COV = 30          # D_select samples a pair needs to count as covered
COV_TOL = 0.01      # max estimated occupancy mass an arm may place off-cover
SAFE_TOL = 1e-6


def true_values(P, mu0, r, C, pi):
    """Exact (return, per-reading costs) of pi on the true model."""
    x = occupancy(P, mu0, pi)
    return float((r * x).sum()), np.array([float((C[k] * x).sum())
                                           for k in range(C.shape[0])]), x


def check_ship_x(x, C, d_budget, n_ev, rng, delta_ev=DELTA_EV):
    """Check on an occupancy (so a mixture can be checked directly)."""
    K = C.shape[0]
    x = np.maximum(x.reshape(-1), 0)
    x = x / x.sum()
    idx = rng.choice(x.size, size=n_ev, p=x)
    L = np.log(3.0 / (delta_ev / K))
    bound = -np.inf
    for k in range(K):
        z = C[k].reshape(-1)[idx]
        b = (z.mean() + np.sqrt(2.0 * z.var(ddof=1) * L / n_ev)
             + 3.0 * (float(C[k].max()) or 1.0) * L / n_ev)
        bound = max(bound, float(b))
    return bound, bool(bound <= d_budget)


def mixture_lp(R_hat, C_hat, d_target):
    """max w.R  s.t.  w.C_phi <= d_target for all phi, w in simplex."""
    M, K = C_hat.shape
    res = linprog(-R_hat, A_ub=C_hat.T, b_ub=np.full(K, d_target),
                  A_eq=np.ones((1, M)), b_eq=np.array([1.0]),
                  bounds=[(0, 1)] * M, method="highs")
    return None if res.status != 0 else res.x


def split_log(lg: dict, frac: float = 0.5):
    n = lg["S"].size
    h = int(n * frac)
    keys = ("S", "A", "SP", "R")
    a = {k: lg[k][:h] for k in keys}
    b = {k: lg[k][h:] for k in keys}
    a["C"], b["C"] = lg["C"][:, :h], lg["C"][:, h:]
    for part in (a, b):
        part.update(nS=lg["nS"], nA=lg["nA"], K=lg["K"], mu0=lg["mu0"])
    return a, b


def main() -> None:
    sel = json.load(open(SEL, encoding="utf8"))
    mid = [r for r in sel["rows"]
           if r["regime"] == "optimizer_resolvable"
           and abs(r["budget"] - BUDGET) < 1e-12]
    pt, _ = parse_prometheus()
    tbank = prom_threshold_bank(pt)
    bank_names = (["single", "fullset", "safeface"]
                  + [f"margin_k{kf}" for kf in KAPPA_FRAC])
    compiled = {}
    for ti, t in enumerate(pt):
        ok, _w, readings = _eligible(build_prom_pool(t, tbank))
        if ok:
            compiled[f'{getattr(t, "name", "?")}#{ti}'] = compile_instance(readings)

    rows: List[dict] = []
    for row in mid:
        m = compiled[row["rule_id"]]
        P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]
        anchor = row["anchor"]
        pi_b = behaviour_policy(m, anchor, BUDGET)
        for seed in SEEDS:
            rng = np.random.default_rng(
                abs(hash((row["rule_id"], N_TOTAL, seed))) % (2 ** 32))
            full_log = sample_log(m, pi_b, N_TOTAL, rng)
            tr, se = split_log(full_log)
            lg_tr, lg_se, lg_all = Log(tr), Log(se), Log(full_log)

            # ---- the bank, trained on D_train only -----------------------
            bank: Dict[str, np.ndarray] = {}
            bank["single"] = learn_single(lg_tr, anchor, BUDGET)
            bank["fullset"] = learn_fullset(lg_tr, anchor, BUDGET)
            bank["safeface"], _ = learn_safeface(lg_tr, anchor, BUDGET, EPS)
            for kf in KAPPA_FRAC:
                bank[f"margin_k{kf}"], _ = learn_safeface(
                    lg_tr, anchor, BUDGET, EPS, kappa=kf * BUDGET)
            # the operator's real alternative: full-set on ALL the data
            pi_full_all = learn_fullset(lg_all, anchor, BUDGET)

            # ---- selection estimates on D_select only --------------------
            R_hat = np.array([lg_se.est(bank[b])[0] for b in bank_names])
            C_hat = np.stack([lg_se.est(bank[b])[1] for b in bank_names])

            # coverage: estimated occupancy mass on under-sampled pairs
            under = (lg_se.N < N_COV).reshape(-1)
            cov_mass = np.array([
                float(occupancy(lg_se.Phat, lg_se.mu0, bank[b]).reshape(-1)[under].sum())
                for b in bank_names])
            covered = cov_mass <= COV_TOL
            covered[bank_names.index("fullset")] = True   # always admissible

            # ---- true values, for scoring only ---------------------------
            truth = {b: true_values(P, mu0, r, C, bank[b]) for b in bank_names}
            R_true = np.array([truth[b][0] for b in bank_names])
            C_true = np.stack([truth[b][1] for b in bank_names])
            X_true = np.stack([truth[b][2].reshape(-1) for b in bank_names])

            crng = np.random.default_rng(
                abs(hash((row["rule_id"], N_TOTAL, seed, "ev"))) % (2 ** 32))
            out: Dict[str, dict] = {}

            def record(name, w):
                Rw = float(w @ R_true)
                Cw = w @ C_true
                xw = w @ X_true
                bound, ship = check_ship_x(xw, C, BUDGET, N_EV, crng)
                cmax = float(Cw.max())
                out[name] = dict(
                    J_r=Rw, ret_frac=Rw / row["V_U"], C_max=cmax,
                    safe=bool(cmax <= BUDGET + SAFE_TOL),
                    ship=ship, bound=bound,
                    unsafe_ship=bool(ship and cmax > BUDGET + SAFE_TOL),
                    w_fullset=float(w[bank_names.index("fullset")])
                    if len(w) == len(bank_names) else None)

            # baselines as degenerate mixtures
            e = np.zeros(len(bank_names))
            e[bank_names.index("fullset")] = 1.0
            record("fullset_train", e)
            Rf, Cf, xf = true_values(P, mu0, r, C, pi_full_all)
            bound, ship = check_ship_x(xf, C, BUDGET, N_EV, crng)
            out["fullset_all"] = dict(
                J_r=Rf, ret_frac=Rf / row["V_U"], C_max=float(Cf.max()),
                safe=bool(Cf.max() <= BUDGET + SAFE_TOL), ship=ship,
                bound=bound,
                unsafe_ship=bool(ship and Cf.max() > BUDGET + SAFE_TOL),
                w_fullset=None)

            # estimation bias, per arm
            out["_bias"] = {b: dict(est_ret=float(R_hat[i] / row["V_U"]),
                                    true_ret=float(R_true[i] / row["V_U"]),
                                    est_C=float(C_hat[i].max() / BUDGET),
                                    true_C=float(C_true[i].max() / BUDGET),
                                    off_cover=float(cov_mass[i]))
                            for i, b in enumerate(bank_names)}

            # the mixture, over the full bank and over {fullset, safeface}
            two = [bank_names.index("fullset"), bank_names.index("safeface")]
            for rf in RHO_FRAC:
                w = mixture_lp(R_hat, C_hat, BUDGET * (1 - rf))
                if w is None:
                    w = e.copy()                    # LP infeasible: full-set
                record(f"mix_rho{rf}", w)
                # amended rule: only arms whose estimate is on covered data
                idx = np.where(covered)[0]
                wc = mixture_lp(R_hat[idx], C_hat[idx], BUDGET * (1 - rf))
                w_cov = np.zeros(len(bank_names))
                if wc is None:
                    w_cov[bank_names.index("fullset")] = 1.0
                else:
                    w_cov[idx] = wc
                record(f"mix_cov_rho{rf}", w_cov)
                w2 = mixture_lp(R_hat[two], C_hat[two], BUDGET * (1 - rf))
                w_full = np.zeros(len(bank_names))
                if w2 is None:
                    w_full[two[0]] = 1.0
                else:
                    w_full[two] = w2
                record(f"mix2_rho{rf}", w_full)

            rows.append(dict(rule_id=row["rule_id"], seed=seed,
                             V_U=row["V_U"], arms=out))
        print(f"  {row['rule_id']:38s} done", flush=True)

    # ---- aggregate -----------------------------------------------------
    names = (["fullset_all", "fullset_train"]
             + [f"mix_rho{rf}" for rf in RHO_FRAC]
             + [f"mix_cov_rho{rf}" for rf in RHO_FRAC]
             + [f"mix2_rho{rf}" for rf in RHO_FRAC])
    agg = {}
    fa_ret_ship = {(x["rule_id"], x["seed"]): x["arms"]["fullset_all"]
                   for x in rows}
    for nm in names:
        L = [x["arms"][nm] for x in rows]
        sh = np.array([l["ship"] for l in L])
        ret = np.array([l["ret_frac"] for l in L])
        # end-to-end comparison against full-set on all data, both shipped
        both = [(x["arms"][nm]["ret_frac"], x["arms"]["fullset_all"]["ret_frac"])
                for x in rows
                if x["arms"][nm]["ship"] and x["arms"]["fullset_all"]["ship"]]
        agg[nm] = dict(
            n=len(L),
            safe_frac=float(np.mean([l["safe"] for l in L])),
            ship_frac=float(sh.mean()),
            unsafe_ship_frac=float(np.mean([l["unsafe_ship"] for l in L])),
            ret_med=float(np.median(ret)),
            ret_given_ship=(float(np.median(ret[sh])) if sh.any() else None),
            C_over_d_med=float(np.median([l["C_max"] for l in L]) / BUDGET),
            w_fullset_med=(float(np.median([l["w_fullset"] for l in L
                                            if l["w_fullset"] is not None]))
                           if any(l["w_fullset"] is not None for l in L)
                           else None),
            n_both_ship=len(both),
            beats_fullset_all_given_both_ship=(
                float(np.mean([a > b + 1e-9 for a, b in both])) if both else None),
            gain_given_both_ship_med=(
                float(np.median([a - b for a, b in both])) if both else None))

    bias = {}
    for b in bank_names:
        B = [x["arms"]["_bias"][b] for x in rows]
        bias[b] = dict(est_C_med=float(np.median([z["est_C"] for z in B])),
                       true_C_med=float(np.median([z["true_C"] for z in B])),
                       est_ret_med=float(np.median([z["est_ret"] for z in B])),
                       true_ret_med=float(np.median([z["true_ret"] for z in B])),
                       off_cover_med=float(np.median([z["off_cover"] for z in B])),
                       covered_frac=float(np.mean([z["off_cover"] <= COV_TOL for z in B])))
    res = dict(registration="V33 (REGISTRATION_V28.md addendum)",
               n_cov=N_COV, cov_tol=COV_TOL, bias=bias,
               budget=BUDGET, eps=EPS, n_total=N_TOTAL, split=0.5,
               n_seeds=len(SEEDS), n_instances=len(mid),
               bank=bank_names, rho_frac=list(RHO_FRAC),
               aggregate=agg, rows=rows)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {OUT} ({len(rows)} runs)")
    print("estimation bias on D_select (medians): est C/d  true C/d  est ret  true ret  off-cover  covered")
    for b in bank_names:
        v = bias[b]
        print(f"  {b:12s} {v['est_C_med']:8.2f} {v['true_C_med']:9.2f} {v['est_ret_med']:8.3f} "
              f"{v['true_ret_med']:9.3f} {v['off_cover_med']:10.3f} {v['covered_frac']:8.2f}")
    print()
    print(f"{'arm':14s} {'safe':>6} {'SHIP':>6} {'unsafeSHIP':>10} "
          f"{'ret':>6} {'ret|SHIP':>8} {'C/d':>6} {'w_full':>7} "
          f"{'beats FS|both':>13} {'gain':>7}")
    for nm in names:
        v = agg[nm]
        f = lambda z, w=6, p=3: (" " * (w - 2) + "--") if z is None else f"{z:>{w}.{p}f}"
        print(f"{nm:14s} {v['safe_frac']:6.3f} {v['ship_frac']:6.3f} "
              f"{v['unsafe_ship_frac']:10.3f} {v['ret_med']:6.3f} "
              f"{f(v['ret_given_ship'], 8)} {v['C_over_d_med']:6.3f} "
              f"{f(v['w_fullset_med'], 7)} "
              f"{f(v['beats_fullset_all_given_both_ship'], 13)} "
              f"{f(v['gain_given_both_ship_med'], 7, 4)}")


if __name__ == "__main__":
    main()
