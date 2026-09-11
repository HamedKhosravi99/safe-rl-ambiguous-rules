"""V36: the clean baseline -- same model, same LP, same support, same buffer,
same evaluator; ordinary full-set CMDP as the comparison. Decomposes the V35
gain into solver, buffer and deployment-rule effects. Registration V36."""
from __future__ import annotations
import json, os
from typing import Dict, List
import numpy as np
from .control_suite import _eligible, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool
from .safe_face_offline import Log, behaviour_policy, learn_fullset, sample_log
from .safe_face_mixture import true_values
from .certified_safe_face import occ_lp_policy
from .occupancy_arrow import bounds_x, BUDGET, N_TOTAL, SEEDS, DELTA_EV, M_TOTAL, SAFE_TOL, N_BOOT

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SEL = os.path.join(ROOT, "results/e2e", "safe_face_select.json")
OUT = os.path.join(ROOT, "results/e2e", "clean_baseline.json")
RHOS = (0.3, 0.5, 0.6, 0.7); RHO_P = 0.5; RHO_FB = 0.7

def main():
    sel = json.load(open(SEL)); mid = [r for r in sel["rows"] if r["regime"] == "optimizer_resolvable" and abs(r["budget"] - BUDGET) < 1e-12]
    pt, _ = parse_prometheus(); tb = prom_threshold_bank(pt); comp = {}
    for ti, t in enumerate(pt):
        ok, _w, rd = _eligible(build_prom_pool(t, tb))
        if ok: comp[f'{getattr(t, "name", "?")}#{ti}'] = compile_instance(rd)
    dq = DELTA_EV / 4.0; rows: List[dict] = []; n_identity = 0
    for row in mid:
        m = comp[row["rule_id"]]; P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]
        r_flat = r.reshape(-1); C_flat = np.stack([C[k].reshape(-1) for k in range(C.shape[0])])
        anchor, V_U = row["anchor"], row["V_U"]; pi_b = behaviour_policy(m, anchor, BUDGET)
        for seed in SEEDS:
            rng = np.random.default_rng(abs(hash((row["rule_id"], N_TOTAL, seed))) % (2 ** 32))
            lg = Log(sample_log(m, pi_b, N_TOTAL, rng))
            ev = np.random.default_rng(abs(hash((row["rule_id"], N_TOTAL, seed, "v36"))) % (2 ** 32))
            out: Dict[str, dict] = {}
            def rec(name, deploy, R, Cm, promoted):
                ship = deploy != "abstain"
                out[name] = dict(deploy=deploy, ship=ship, promoted=promoted, unsafe_ship=bool(ship and Cm > BUDGET + SAFE_TOL),
                                 ret=(R / V_U if ship else None), U=(R / V_U if ship else 0.0))
            # FQI full set, one check
            pi_F = learn_fullset(lg, anchor, BUDGET); RF, CF, xF = true_values(P, mu0, r, C, pi_F); xF = xF.reshape(-1)
            bF = bounds_x(xF, r_flat, C_flat, M_TOTAL, ev, DELTA_EV, DELTA_EV)
            rec("fqi_full", "full" if bF["ucb_c"] <= BUDGET else "abstain", RF, float(CF.max()), False)
            # occupancy LP full set at rho = 0 and at each buffer, one check each
            lp = {}
            for rf in (0.0,) + RHOS:
                pi_o = occ_lp_policy(lg, BUDGET * (1 - rf)); lp[rf] = pi_o
                if pi_o is None: rec(f"lp_full_rho{rf}", "abstain", 0.0, 0.0, False); continue
                Ro, Co, xo = true_values(P, mu0, r, C, pi_o); bo = bounds_x(xo.reshape(-1), r_flat, C_flat, M_TOTAL, ev, DELTA_EV, DELTA_EV)
                rec(f"lp_full_rho{rf}", "occ" if bo["ucb_c"] <= BUDGET else "abstain", Ro, float(Co.max()), bo["ucb_c"] <= BUDGET)
            # ARROW with FQI fallback (V35) and with an occupancy fallback
            pi_fb_lp = lp[RHO_FB]
            for rf in RHOS:
                pi_c = occ_lp_policy(lg, BUDGET * (1 - rf))
                # identity with the full-set LP at the same buffer: same program, same policy
                if pi_c is not None and lp[rf] is not None:
                    assert np.allclose(pi_c, lp[rf]), "candidate differs from the full-set LP at the same buffer"; n_identity += 1
                for fbname, pi_fb in (("arrow", pi_F), ("arrow_lpfb", pi_fb_lp)):
                    if pi_fb is None: rec(f"{fbname}_rho{rf}", "abstain", 0.0, 0.0, False); continue
                    Rb, Cb, xb = true_values(P, mu0, r, C, pi_fb); xb = xb.reshape(-1)
                    bfb = bounds_x(xb, r_flat, C_flat, M_TOTAL // 2, ev, dq, dq)
                    if pi_c is None:
                        rec(f"{fbname}_rho{rf}", "full" if bfb["ucb_c"] <= BUDGET else "abstain", Rb, float(Cb.max()), False); continue
                    Rc, Cc, xc = true_values(P, mu0, r, C, pi_c); bc = bounds_x(xc.reshape(-1), r_flat, C_flat, M_TOTAL // 2, ev, dq, dq)
                    if bc["ucb_c"] <= BUDGET and bc["lcb_r"] > bfb["ucb_r"]: rec(f"{fbname}_rho{rf}", "occ", Rc, float(Cc.max()), True)
                    elif bfb["ucb_c"] <= BUDGET: rec(f"{fbname}_rho{rf}", "full", Rb, float(Cb.max()), False)
                    else: rec(f"{fbname}_rho{rf}", "abstain", Rb, float(Cb.max()), False)
            rows.append(dict(rule_id=row["rule_id"], seed=seed, V_U=V_U, arms=out))
        print(f"  {row['rule_id']:38s} done", flush=True)
    json.dump(dict(rows=rows), open(OUT.replace(".json", "_rows.json"), "w"))
    brng = np.random.default_rng(7)
    def ci(diff):
        diff = np.asarray(diff); n = len(diff); bs = np.array([diff[brng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    names = sorted({k for x in rows for k in x["arms"]}); agg = {}
    Ub = np.array([x["arms"]["fqi_full"]["U"] for x in rows])
    for nm in names:
        L = [x["arms"][nm] for x in rows]; U = np.array([l["U"] for l in L]); S = np.array([l["ship"] for l in L], float)
        rets = [l["ret"] for l in L if l["ret"] is not None]
        agg[nm] = dict(ship=float(S.mean()), unsafe_ship=float(np.mean([l["unsafe_ship"] for l in L])), promoted=float(np.mean([l["promoted"] for l in L])),
                       ret_given_ship=(float(np.median(rets)) if rets else None), U_mean=float(U.mean()),
                       dU_vs_fqi=float((U - Ub).mean()), dU_vs_fqi_ci=ci(U - Ub),
                       win=float(np.mean(U > Ub + 1e-9)), tie=float(np.mean(abs(U - Ub) <= 1e-9)), loss=float(np.mean(U < Ub - 1e-9)))
    def U_of(nm): return np.array([x["arms"][nm]["U"] for x in rows])
    dec = dict(solver=dict(mean=float((U_of("lp_full_rho0.0") - Ub).mean()), ci=ci(U_of("lp_full_rho0.0") - Ub)),
               buffer=dict(mean=float((U_of(f"lp_full_rho{RHO_P}") - U_of("lp_full_rho0.0")).mean()), ci=ci(U_of(f"lp_full_rho{RHO_P}") - U_of("lp_full_rho0.0"))),
               deployment_rule=dict(mean=float((U_of(f"arrow_rho{RHO_P}") - U_of(f"lp_full_rho{RHO_P}")).mean()), ci=ci(U_of(f"arrow_rho{RHO_P}") - U_of(f"lp_full_rho{RHO_P}"))),
               total=dict(mean=float((U_of(f"arrow_rho{RHO_P}") - Ub).mean()), ci=ci(U_of(f"arrow_rho{RHO_P}") - Ub)))
    res = dict(registration="V36 (REGISTRATION_V28.md)", n_identity_checks=n_identity, rho_primary=RHO_P, rho_fb=RHO_FB, decomposition=dec, aggregate=agg, rows=rows)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}; candidate == full-set LP at same buffer asserted on {n_identity} runs")
    print("DECOMPOSITION of the V35 gain at rho = 0.5 d (points of V_U, paired, 95% CI):")
    for k, v in dec.items(): print(f"  {k:16s} {100*v['mean']:+6.2f}  [{100*v['ci'][0]:+.2f}, {100*v['ci'][1]:+.2f}]")
    print(f"{'arm':20s} {'SHIP':>6} {'unsafe':>6} {'promo':>6} {'ret|S':>6} {'U':>6} {'dU vs FQI [CI]':>24} {'W/T/L':>17}")
    for nm in names:
        v = agg[nm]; c = v["dU_vs_fqi_ci"]
        print(f"{nm:20s} {v['ship']:6.3f} {v['unsafe_ship']:6.3f} {v['promoted']:6.3f} {(v['ret_given_ship'] or 0):6.3f} {v['U_mean']:6.3f} {100*v['dU_vs_fqi']:+6.2f} [{100*c[0]:+.1f},{100*c[1]:+.1f}]   {v['win']:.2f}/{v['tie']:.2f}/{v['loss']:.2f}")

if __name__ == "__main__":
    main()
