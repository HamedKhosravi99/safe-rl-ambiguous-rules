"""V34: Certified Safe-Face Policy Improvement.

V33 showed the learned mixture beats full-set on shipped return but ships less
often, and that the gap to the oracle hull is the selection estimate. This
module adds the four registered changes and runs them factorially:

  bank       {full, safe-face} | {full, safe-face, margin k=0.3} |
             {full, safe-face, occupancy-LP} | all eight arms
  selection  off-policy (D_select model, as V33) | on-policy shadow sample
  rule       mixture only | certified fallback | fallback + reward gate
  budget     equal per-check (the theorem's regime) | equal total B

plus the occupancy-LP learner standalone and the full-set baseline. Every
deployed policy is scored exactly on the true model; shadow samples for
selection and Check are drawn from the candidates' TRUE occupancies, which is
what a staging run produces, and are charged to the operator's budget.

Registration: results/e2e/REGISTRATION_V28.md (V34).
Run: PYTHONPATH=. python3 -m saorl.benchmark_sg.certified_safe_face
Writes results/e2e/certified_safe_face.json
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linprog

from .control_mdp import GAMMA
from .control_suite import _eligible, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool
from .safe_face_offline import (Log, behaviour_policy, learn_fullset,
                                learn_safeface, occupancy, sample_log)
from .safe_face_mixture import mixture_lp, split_log, true_values

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SEL = os.path.join(ROOT, "results/e2e", "safe_face_select.json")
OUT = os.path.join(ROOT, "results/e2e", "certified_safe_face.json")

BUDGET = 0.005
EPS = 0.01
N_TOTAL = 20000
SEEDS = tuple(range(30))
KAPPA_FRAC = (0.0, 0.1, 0.2, 0.3, 0.5)
RHO_FRAC = (0.1, 0.2, 0.3, 0.5)
DELTA_EV = 0.05
DELTA_GATE = 0.05
# budget allocations, in shadow samples
ALLOC = {
    "perCheck": dict(n_sel_per_member=5000, n_check=20000, base_check=20000),
    "total40k": dict(n_sel_total=10000, total=40000, base_check=40000),
}
SAFE_TOL = 1e-6


# --------------------------------------------------------------------------
def occ_lp_policy(lg: Log, d_target: float) -> Optional[np.ndarray]:
    """Model-based planning on the estimated model: max r'x s.t. flow(P_hat),
    behaviour support, every retained cost <= d_target. Returns the
    randomized policy the occupancy induces."""
    nS, nA = lg.nS, lg.nA
    block = np.kron(np.eye(nS), np.ones((1, nA)))
    A_eq = block - GAMMA * np.transpose(lg.Phat, (2, 0, 1)).reshape(nS, nS * nA)
    b_eq = (1.0 - GAMMA) * lg.mu0
    bounds = [(0, None) if lg.support[s, a] else (0, 0)
              for s in range(nS) for a in range(nA)]
    res = linprog(-lg.r.reshape(-1),
                  A_ub=np.array([lg.c[k].reshape(-1) for k in range(lg.K)]),
                  b_ub=np.full(lg.K, d_target),
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        return None
    x = res.x.reshape(nS, nA)
    tot = x.sum(axis=1)
    pi = np.zeros((nS, nA))
    vis = tot > 1e-12
    pi[vis] = x[vis] / tot[vis, None]
    for s in np.where(~vis)[0]:            # unvisited: most-sampled supported action
        a = int(np.argmax(np.where(lg.support[s], lg.N[s], -1)))
        pi[s, a] = 1.0
    return pi


def eb_rad(z: np.ndarray, L: float, rng_z: float) -> float:
    n = z.size
    return float(np.sqrt(2.0 * z.var(ddof=1) * L / n) + 3.0 * rng_z * L / n)


def shadow(x_flat: np.ndarray, n: int, rng) -> np.ndarray:
    x = np.maximum(x_flat, 0)
    return rng.choice(x.size, size=n, p=x / x.sum())


def check_x(x_flat, r_flat, C_flat, d, n, rng, delta):
    """One independent check on an occupancy: per-reading empirical-Bernstein
    UCB (union over readings) against d, plus the reward mean and radius for
    the promotion gate."""
    idx = shadow(x_flat, n, rng)
    K = C_flat.shape[0]
    L = np.log(3.0 / (delta / K))
    bound = max(float(C_flat[k][idx].mean() + eb_rad(C_flat[k][idx], L,
                                                     float(C_flat[k].max()) or 1.0))
                for k in range(K))
    rz = r_flat[idx]
    Lr = np.log(3.0 / DELTA_GATE)
    return dict(ship=bool(bound <= d), bound=bound,
                r_mean=float(rz.mean()), r_rad=eb_rad(rz, Lr, float(r_flat.max()) or 1.0))


def onpolicy_estimates(X_true, r_flat, C_flat, n_per, rng):
    R = np.empty(X_true.shape[0])
    C = np.empty((X_true.shape[0], C_flat.shape[0]))
    for m in range(X_true.shape[0]):
        idx = shadow(X_true[m], n_per, rng)
        R[m] = r_flat[idx].mean()
        C[m] = [C_flat[k][idx].mean() for k in range(C_flat.shape[0])]
    return R, C


# --------------------------------------------------------------------------
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

    arm_names = (["single", "fullset", "safeface"]
                 + [f"margin_k{kf}" for kf in KAPPA_FRAC])
    banks = {
        "two": ["fullset", "safeface"],
        "three_margin": ["fullset", "safeface", "margin_k0.3"],
        "three_occ": ["fullset", "safeface", "occlp"],
        "eight": list(arm_names),
    }

    rows: List[dict] = []
    for row in mid:
        m = compiled[row["rule_id"]]
        P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]
        r_flat = r.reshape(-1)
        C_flat = np.stack([C[k].reshape(-1) for k in range(C.shape[0])])
        anchor = row["anchor"]
        V_U = row["V_U"]
        pi_b = behaviour_policy(m, anchor, BUDGET)
        for seed in SEEDS:
            rng = np.random.default_rng(
                abs(hash((row["rule_id"], N_TOTAL, seed))) % (2 ** 32))
            full_log = sample_log(m, pi_b, N_TOTAL, rng)
            tr, se = split_log(full_log)
            lg_tr, lg_se, lg_all = Log(tr), Log(se), Log(full_log)

            # ---- candidates ------------------------------------------------
            from .safe_face_offline import learn_single
            pol: Dict[str, np.ndarray] = {}
            pol["single"] = learn_single(lg_tr, anchor, BUDGET)
            pol["fullset"] = learn_fullset(lg_tr, anchor, BUDGET)
            pol["safeface"], _ = learn_safeface(lg_tr, anchor, BUDGET, EPS)
            for kf in KAPPA_FRAC:
                pol[f"margin_k{kf}"], _ = learn_safeface(
                    lg_tr, anchor, BUDGET, EPS, kappa=kf * BUDGET)
            pi_full_all = learn_fullset(lg_all, anchor, BUDGET)
            # occupancy-LP learner, one per buffer (rho is its only knob)
            occ = {rf: occ_lp_policy(lg_tr, BUDGET * (1 - rf)) for rf in RHO_FRAC}
            occ_all = {rf: occ_lp_policy(lg_all, BUDGET * (1 - rf)) for rf in RHO_FRAC}

            truth = {k: true_values(P, mu0, r, C, v) for k, v in pol.items()}
            Rf_all, Cf_all, xf_all = true_values(P, mu0, r, C, pi_full_all)
            xf_all = xf_all.reshape(-1)

            ev_rng = np.random.default_rng(
                abs(hash((row["rule_id"], N_TOTAL, seed, "v34"))) % (2 ** 32))
            out: Dict[str, dict] = {}

            def rec(name, w, members, chk_mix, chk_full, deploy, promoted):
                """Score a deployment decision exactly."""
                if deploy == "mix":
                    Rw = float(sum(wi * truth_m[0] for wi, truth_m in zip(w, members)))
                    Cw = float(max(sum(wi * truth_m[1][k] for wi, truth_m in zip(w, members))
                                   for k in range(C.shape[0])))
                elif deploy == "full":
                    Rw, Cw = Rf_all, float(Cf_all.max())
                else:
                    Rw, Cw = float("nan"), float("nan")
                ship = deploy != "abstain"
                out[name] = dict(
                    deploy=deploy, ship=ship, promoted=promoted,
                    unsafe_ship=bool(ship and Cw > BUDGET + SAFE_TOL),
                    ret=(Rw / V_U if ship else None),
                    dw_ret=(Rw / V_U if ship else 0.0),
                    C_over_d=(Cw / BUDGET if ship else None),
                    mix_bound=(None if chk_mix is None else chk_mix["bound"]),
                    full_bound=(None if chk_full is None else chk_full["bound"]))

            for alloc, A in ALLOC.items():
                # ---- baseline: full-set on all data, one check ------------
                ck = check_x(xf_all, r_flat, C_flat, BUDGET, A["base_check"], ev_rng, DELTA_EV)
                rec(f"{alloc}|baseline_full", None, None, None, ck,
                    "full" if ck["ship"] else "abstain", False)
                # ---- occupancy-LP standalone, one check at baseline budget -
                for rf in RHO_FRAC:
                    pi_o = occ_all[rf]
                    if pi_o is None:
                        rec(f"{alloc}|occlp_rho{rf}", None, None, None, None, "abstain", False)
                        continue
                    Ro, Co, xo = true_values(P, mu0, r, C, pi_o)
                    ck = check_x(xo.reshape(-1), r_flat, C_flat, BUDGET, A["base_check"], ev_rng, DELTA_EV)
                    if ck["ship"]:
                        out[f"{alloc}|occlp_rho{rf}"] = dict(
                            deploy="occ", ship=True, promoted=True,
                            unsafe_ship=bool(Co.max() > BUDGET + SAFE_TOL),
                            ret=Ro / V_U, dw_ret=Ro / V_U, C_over_d=float(Co.max()) / BUDGET,
                            mix_bound=ck["bound"], full_bound=None)
                    else:
                        rec(f"{alloc}|occlp_rho{rf}", None, None, None, None, "abstain", False)

                for bname, members in banks.items():
                    for rf in RHO_FRAC:
                        # assemble the bank (the occupancy-LP member is at this rho)
                        mem_pol = [occ[rf] if mm == "occlp" else pol[mm] for mm in members]
                        if any(p is None for p in mem_pol):
                            continue
                        mem_truth = [true_values(P, mu0, r, C, p) for p in mem_pol]
                        X_true = np.stack([t[2].reshape(-1) for t in mem_truth])
                        M = len(members)
                        # budgets
                        if alloc == "perCheck":
                            n_sel, n_chk = A["n_sel_per_member"], A["n_check"]
                        else:
                            n_sel = A["n_sel_total"] // M
                            n_chk = (A["total"] - A["n_sel_total"]) // 2
                        for selname in ("ope", "onpolicy"):
                            if selname == "ope":
                                R_hat = np.array([lg_se.est(p)[0] for p in mem_pol])
                                C_hat = np.stack([lg_se.est(p)[1] for p in mem_pol])
                            else:
                                R_hat, C_hat = onpolicy_estimates(X_true, r_flat, C_flat, n_sel, ev_rng)
                            w = mixture_lp(R_hat, C_hat, BUDGET * (1 - rf))
                            if w is None:
                                w = np.zeros(M)
                                w[members.index("fullset")] = 1.0
                            x_w = w @ X_true
                            ck_mix = check_x(x_w, r_flat, C_flat, BUDGET, n_chk, ev_rng, DELTA_EV / 2)
                            ck_full = check_x(xf_all, r_flat, C_flat, BUDGET, n_chk, ev_rng, DELTA_EV / 2)
                            gate = (ck_mix["r_mean"] - ck_mix["r_rad"]) > (ck_full["r_mean"] + ck_full["r_rad"])
                            tag = f"{alloc}|{bname}|{selname}|rho{rf}"
                            # mixture only
                            rec(tag + "|mixonly", w, mem_truth, ck_mix, None,
                                "mix" if ck_mix["ship"] else "abstain", ck_mix["ship"])
                            # certified fallback, no gate
                            d_ = "mix" if ck_mix["ship"] else ("full" if ck_full["ship"] else "abstain")
                            rec(tag + "|csf_nogate", w, mem_truth, ck_mix, ck_full, d_, d_ == "mix")
                            # certified fallback with reward gate
                            d_ = ("mix" if (ck_mix["ship"] and gate)
                                  else ("full" if ck_full["ship"] else "abstain"))
                            rec(tag + "|csf", w, mem_truth, ck_mix, ck_full, d_, d_ == "mix")
                            out[tag + "|csf"]["w_fullset"] = float(w[members.index("fullset")])
            rows.append(dict(rule_id=row["rule_id"], seed=seed, V_U=V_U, arms=out))
        print(f"  {row['rule_id']:38s} done", flush=True)

    # ---- aggregate -----------------------------------------------------
    names = sorted({k for x in rows for k in x["arms"]})
    agg = {}
    for nm in names:
        L = [x["arms"][nm] for x in rows if nm in x["arms"]]
        if not L:
            continue
        alloc = nm.split("|")[0]
        base = [x["arms"][f"{alloc}|baseline_full"] for x in rows if nm in x["arms"]]
        ship = np.array([l["ship"] for l in L])
        dw = np.array([l["dw_ret"] for l in L])
        dwb = np.array([b["dw_ret"] for b in base])
        rets = [l["ret"] for l in L if l["ret"] is not None]
        agg[nm] = dict(
            n=len(L),
            ship=float(ship.mean()),
            unsafe_ship=float(np.mean([l["unsafe_ship"] for l in L])),
            promoted=float(np.mean([l["promoted"] for l in L])),
            ret_given_ship=(float(np.median(rets)) if rets else None),
            dw_ret_mean=float(dw.mean()),
            beats_base_dw=float(np.mean(dw > dwb + 1e-9)),
            dw_gain_med=float(np.median(dw - dwb)),
            ship_minus_base=float(ship.mean() - np.mean([b["ship"] for b in base])))
    res = dict(registration="V34 (REGISTRATION_V28.md)", budget=BUDGET, eps=EPS,
               n_total=N_TOTAL, n_seeds=len(SEEDS), n_instances=len(mid),
               alloc=ALLOC, banks=banks, rho_frac=list(RHO_FRAC),
               delta_ev=DELTA_EV, delta_gate=DELTA_GATE, aggregate=agg, rows=rows)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {OUT} ({len(rows)} runs)")
    print(f"{'arm':52s} {'SHIP':>6} {'unsafe':>6} {'promo':>6} {'ret|S':>6} {'DWret':>6} {'beats':>6} {'gain':>7} {'dSHIP':>6}")
    for nm in names:
        if nm not in agg:
            continue
        v = agg[nm]
        if ("rho0.5" in nm or "baseline" in nm) and ("csf" in nm or "baseline" in nm or "occlp" in nm or "mixonly" in nm):
            rs = "  --  " if v["ret_given_ship"] is None else f"{v['ret_given_ship']:6.3f}"
            print(f"{nm:52s} {v['ship']:6.3f} {v['unsafe_ship']:6.3f} {v['promoted']:6.3f} {rs} "
                  f"{v['dw_ret_mean']:6.3f} {v['beats_base_dw']:6.3f} {v['dw_gain_med']:+7.4f} {v['ship_minus_base']:+6.3f}")


if __name__ == "__main__":
    main()
