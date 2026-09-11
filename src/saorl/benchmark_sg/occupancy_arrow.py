"""V35: Safe-Face Occupancy RL with certified fallback -- the confirmatory run.

One primary system. The learner estimates a model from the whole log, solves
the buffered occupancy program on it with behaviour support, and reads off the
RANDOMIZED policy. Deployment promotes that candidate only when an
independent check certifies it safe AND a valid lower bound on its return
exceeds a valid upper bound on the fallback's; otherwise the certified
full-set fallback is deployed; otherwise the system abstains. The four
confidence levels sum to delta_ev so one joint event covers every bound.

Equal TOTAL evaluation budget is the primary comparison: the baseline spends
all M shadow samples on one check, the system splits M across candidate and
fallback.

Registration: results/e2e/REGISTRATION_V28.md (V35).
Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.occupancy_arrow
Writes results/e2e/occupancy_arrow.json
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from .control_suite import _eligible, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool
from .safe_face_offline import Log, behaviour_policy, learn_fullset, sample_log
from .safe_face_mixture import true_values
from .certified_safe_face import eb_rad, occ_lp_policy, shadow

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SEL = os.path.join(ROOT, "results/e2e", "safe_face_select.json")
OUT = os.path.join(ROOT, "results/e2e", "occupancy_arrow.json")

BUDGET = 0.005
N_TOTAL = 20000
SEEDS = tuple(range(30))
RHO_PRIMARY = 0.5
RHO_FRAC = (0.3, 0.5, 0.6, 0.7)
DELTA_EV = 0.05
M_TOTAL = 40000
ALLOC = {"equalTotal": dict(cand=20000, fb=20000, base=40000),
         "equalPerCheck": dict(cand=20000, fb=20000, base=20000)}
SAFE_TOL = 1e-6
N_BOOT = 2000


def bounds_x(x_flat, r_flat, C_flat, n, rng, delta_c, delta_r):
    """One independent shadow sample of an occupancy: per-reading cost UCB
    (union over readings at delta_c) and a two-sided reward interval at
    delta_r, all empirical Bernstein."""
    idx = shadow(x_flat, n, rng)
    K = C_flat.shape[0]
    Lc = np.log(3.0 / (delta_c / K))
    ucb_c = max(float(C_flat[k][idx].mean()
                      + eb_rad(C_flat[k][idx], Lc, float(C_flat[k].max()) or 1.0))
                for k in range(K))
    rz = r_flat[idx]
    Lr = np.log(3.0 / delta_r)
    rad = eb_rad(rz, Lr, float(r_flat.max()) or 1.0)
    return dict(ucb_c=ucb_c, lcb_r=float(rz.mean() - rad), ucb_r=float(rz.mean() + rad))


def main() -> None:
    sel = json.load(open(SEL, encoding="utf8"))
    mid = [r for r in sel["rows"]
           if r["regime"] == "optimizer_resolvable"
           and abs(r["budget"] - BUDGET) < 1e-12]
    pt, _ = parse_prometheus()
    tbank = prom_threshold_bank(pt)
    compiled = {}
    for ti, t in enumerate(pt):
        ok, _w, readings = _eligible(build_prom_pool(t, tbank))
        if ok:
            compiled[f'{getattr(t, "name", "?")}#{ti}'] = compile_instance(readings)

    dq = DELTA_EV / 4.0          # four bounds, one joint event at 1 - delta_ev
    rows: List[dict] = []
    for row in mid:
        m = compiled[row["rule_id"]]
        P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]
        r_flat = r.reshape(-1)
        C_flat = np.stack([C[k].reshape(-1) for k in range(C.shape[0])])
        anchor, V_U = row["anchor"], row["V_U"]
        pi_b = behaviour_policy(m, anchor, BUDGET)
        for seed in SEEDS:
            rng = np.random.default_rng(
                abs(hash((row["rule_id"], N_TOTAL, seed))) % (2 ** 32))
            lg = Log(sample_log(m, pi_b, N_TOTAL, rng))
            pi_F = learn_fullset(lg, anchor, BUDGET)
            RF, CF, xF = true_values(P, mu0, r, C, pi_F)
            xF = xF.reshape(-1)
            ev = np.random.default_rng(
                abs(hash((row["rule_id"], N_TOTAL, seed, "v35"))) % (2 ** 32))
            out: Dict[str, dict] = {}

            def rec(name, deploy, R, Cmax, promoted, extra=None):
                ship = deploy != "abstain"
                o = dict(deploy=deploy, ship=ship, promoted=promoted,
                         unsafe_ship=bool(ship and Cmax > BUDGET + SAFE_TOL),
                         ret=(R / V_U if ship else None),
                         U=(R / V_U if ship else 0.0),
                         C_over_d=(Cmax / BUDGET if ship else None))
                if extra:
                    o.update(extra)
                out[name] = o

            for al, A in ALLOC.items():
                # baseline: full-set, one check, whole budget
                bF = bounds_x(xF, r_flat, C_flat, A["base"], ev, DELTA_EV, DELTA_EV)
                rec(f"{al}|fullset", "full" if bF["ucb_c"] <= BUDGET else "abstain",
                    RF, float(CF.max()), False)
                for rf in RHO_FRAC:
                    pi_o = occ_lp_policy(lg, BUDGET * (1 - rf))
                    if pi_o is None:
                        # candidate infeasible on the estimated model: fallback only
                        bfb = bounds_x(xF, r_flat, C_flat, A["fb"], ev, dq, dq)
                        rec(f"{al}|occ_rho{rf}", "full" if bfb["ucb_c"] <= BUDGET else "abstain",
                            RF, float(CF.max()), False, dict(cand_feasible=False))
                        rec(f"{al}|occ_nogate_rho{rf}", "full" if bfb["ucb_c"] <= BUDGET else "abstain",
                            RF, float(CF.max()), False, dict(cand_feasible=False))
                        continue
                    Ro, Co, xo = true_values(P, mu0, r, C, pi_o)
                    xo = xo.reshape(-1)
                    n_rand = int(((pi_o > 1e-9).sum(axis=1) > 1).sum())
                    bc = bounds_x(xo, r_flat, C_flat, A["cand"], ev, dq, dq)
                    bfb = bounds_x(xF, r_flat, C_flat, A["fb"], ev, dq, dq)
                    cand_safe = bc["ucb_c"] <= BUDGET
                    gate = bc["lcb_r"] > bfb["ucb_r"]
                    fb_ok = bfb["ucb_c"] <= BUDGET
                    extra = dict(cand_feasible=True, cand_true_ret=Ro / V_U,
                                 cand_true_C=float(Co.max()) / BUDGET,
                                 cand_ucb_c=bc["ucb_c"] / BUDGET, cand_safe_cert=cand_safe,
                                 gate=gate, n_randomizing=n_rand,
                                 fb_true_ret=RF / V_U)
                    # primary: gated
                    if cand_safe and gate:
                        rec(f"{al}|occ_rho{rf}", "occ", Ro, float(Co.max()), True, extra)
                    elif fb_ok:
                        rec(f"{al}|occ_rho{rf}", "full", RF, float(CF.max()), False, extra)
                    else:
                        rec(f"{al}|occ_rho{rf}", "abstain", RF, float(CF.max()), False, extra)
                    # ablation: no reward gate
                    if cand_safe:
                        rec(f"{al}|occ_nogate_rho{rf}", "occ", Ro, float(Co.max()), True, extra)
                    elif fb_ok:
                        rec(f"{al}|occ_nogate_rho{rf}", "full", RF, float(CF.max()), False, extra)
                    else:
                        rec(f"{al}|occ_nogate_rho{rf}", "abstain", RF, float(CF.max()), False, extra)
            rows.append(dict(rule_id=row["rule_id"], seed=seed, V_U=V_U, arms=out))
        print(f"  {row['rule_id']:38s} done", flush=True)

    with open(OUT.replace(".json", "_rows.json"), "w", encoding="utf8") as fh:
        json.dump(dict(rows=rows), fh)

    # ---- aggregate with paired bootstrap CIs ---------------------------
    brng = np.random.default_rng(20260905)
    n = len(rows)

    def boot_ci(diff):
        diff = np.asarray(diff)
        m_ = len(diff)
        bs = np.array([diff[brng.integers(0, m_, m_)].mean() for _ in range(N_BOOT)])
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

    agg = {}
    for al in ALLOC:
        base = f"{al}|fullset"
        Ub = np.array([x["arms"][base]["U"] for x in rows])
        Sb = np.array([x["arms"][base]["ship"] for x in rows], float)
        agg[base] = dict(ship=float(Sb.mean()), unsafe_ship=float(np.mean([x["arms"][base]["unsafe_ship"] for x in rows])),
                         ret_given_ship=float(np.median([x["arms"][base]["ret"] for x in rows if x["arms"][base]["ret"] is not None])),
                         U_mean=float(Ub.mean()), promoted=0.0)
        for rf in RHO_FRAC:
            for rule in ("occ", "occ_nogate"):
                k = f"{al}|{rule}_rho{rf}"
                L = [x["arms"][k] for x in rows]
                U = np.array([l["U"] for l in L]); S = np.array([l["ship"] for l in L], float)
                prom = [(l["cand_true_ret"], l["fb_true_ret"]) for l in L if l["deploy"] == "occ"]
                prom = np.array(prom) if prom else np.zeros((0, 2))
                rets = [l["ret"] for l in L if l["ret"] is not None]
                agg[k] = dict(
                    ship=float(S.mean()), unsafe_ship=float(np.mean([l["unsafe_ship"] for l in L])),
                    promoted=float(np.mean([l["promoted"] for l in L])),
                    cand_feasible=float(np.mean([l.get("cand_feasible", False) for l in L])),
                    cand_safe_cert=float(np.mean([l.get("cand_safe_cert", False) for l in L])),
                    gate_pass=float(np.mean([l.get("gate", False) for l in L])),
                    cand_true_C_med=float(np.median([l["cand_true_C"] for l in L if "cand_true_C" in l])) if any("cand_true_C" in l for l in L) else None,
                    n_randomizing_med=float(np.median([l["n_randomizing"] for l in L if "n_randomizing" in l])) if any("n_randomizing" in l for l in L) else None,
                    ret_given_ship=(float(np.median(rets)) if rets else None),
                    U_mean=float(U.mean()),
                    U_gain_mean=float((U - Ub).mean()), U_gain_ci=boot_ci(U - Ub),
                    ship_diff=float((S - Sb).mean()), ship_diff_ci=boot_ci(S - Sb),
                    win=float(np.mean(U > Ub + 1e-9)), tie=float(np.mean(abs(U - Ub) <= 1e-9)),
                    loss=float(np.mean(U < Ub - 1e-9)),
                    promoted_n=int(len(prom)),
                    promoted_gain_med=(float(np.median(prom[:, 0] - prom[:, 1])) if len(prom) else None),
                    promoted_gain_ci=(boot_ci(prom[:, 0] - prom[:, 1]) if len(prom) >= 10 else None),
                    promoted_worse=(float(np.mean(prom[:, 0] < prom[:, 1] - 1e-9)) if len(prom) else None))

    # registered success test at the primary setting
    k = f"equalTotal|occ_rho{RHO_PRIMARY}"
    v = agg[k]
    success = dict(
        i_unsafe_zero=(v["unsafe_ship"] == 0.0 and agg["equalTotal|fullset"]["unsafe_ship"] == 0.0),
        ii_ship_noninferior=(v["ship_diff"] >= -0.01 and v["ship_diff_ci"][0] >= -0.02),
        iii_U_strictly_greater=(v["U_gain_mean"] > 0 and v["U_gain_ci"][0] > 0),
        iv_no_credible_degradation=(v["promoted_gain_ci"] is not None and v["promoted_gain_ci"][0] >= 0))
    success["all"] = all(success.values())

    res = dict(registration="V35 (REGISTRATION_V28.md)", budget=BUDGET, n_total=N_TOTAL,
               n_seeds=len(SEEDS), n_instances=len(mid), rho_primary=RHO_PRIMARY,
               rho_frac=list(RHO_FRAC), delta_ev=DELTA_EV, alloc=ALLOC, M_total=M_TOTAL,
               success=success, aggregate=agg, rows=rows)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {OUT} ({len(rows)} runs)")
    print("REGISTERED SUCCESS TEST (equal total budget, rho = 0.5 d):", json.dumps(success, indent=1))
    print(f"{'arm':28s} {'SHIP':>6} {'dSHIP[CI]':>20} {'unsafe':>6} {'promo':>6} {'ret|S':>6} {'U':>6} {'dU[CI]':>24} {'W/T/L':>17} {'promo gain[CI]':>22} {'candC/d':>7} {'rand':>4}")
    for al in ALLOC:
        b = agg[f"{al}|fullset"]
        print(f"{al+'|fullset':28s} {b['ship']:6.3f} {'':>20} {b['unsafe_ship']:6.3f} {'':>6} {b['ret_given_ship']:6.3f} {b['U_mean']:6.3f}")
        for rf in RHO_FRAC:
            for rule in ("occ", "occ_nogate"):
                k = f"{al}|{rule}_rho{rf}"; v = agg[k]
                ci = lambda c: f"[{100*c[0]:+.1f},{100*c[1]:+.1f}]" if c else "   --   "
                pg = ("--" if v["promoted_gain_med"] is None else f"{100*v['promoted_gain_med']:+.1f}") + ci(v["promoted_gain_ci"])
                print(f"{k:28s} {v['ship']:6.3f} {100*v['ship_diff']:+6.1f}{ci(v['ship_diff_ci']):>14} {v['unsafe_ship']:6.3f} {v['promoted']:6.3f} "
                      f"{(v['ret_given_ship'] or 0):6.3f} {v['U_mean']:6.3f} {100*v['U_gain_mean']:+6.2f}{ci(v['U_gain_ci']):>18} "
                      f"{v['win']:.2f}/{v['tie']:.2f}/{v['loss']:.2f} {pg:>22} {(v['cand_true_C_med'] or 0):7.2f} {(v['n_randomizing_med'] or 0):4.0f}")


if __name__ == "__main__":
    main()
