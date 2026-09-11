"""V47: ARROW versus the practitioner's pipeline
   singleton-train -> independent Check -> robust retrain on failure (STCR).

Part A reads the archived controlled domains (per-seed learner returns from
results/conformal/main50 + budget50, per-seed Check outcomes from
results/conformal/lp/certificate_audit.json) and composes STCR per seed.
Part B runs both pipelines on the compiled monitoring instances with the
same tabular learners, logs and Check as V29/V41/V45.

Run: OMP_NUM_THREADS=1 SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.pipeline_compare
Writes results/e2e/pipeline_compare.json
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
from scipy.optimize import linprog

from .control_mdp import GAMMA
from .control_suite import _eligible, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool
from .safe_face_offline import (Log, behaviour_policy, sample_log, learn_single, learn_fullset,
                                score, check_ship)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "results/e2e")
OUT = os.path.join(R, "pipeline_compare.json")
D_GRID = (0.05, 0.005)
N_GRID = (2000, 20000)
SEEDS = tuple(range(10))
N_EV = 20000
DELTA_EV = 0.05
DOMS = (("synthetic", "Synthetic maintenance"), ("real", "C-MAPSS replay MDP"),
        ("gridworld", "Warning-window"), ("budget", "Budget agent"))


# --------------------------------------------------------------------------- Part A
def part_a():
    au = json.load(open(os.path.join(ROOT, "results/conformal", "lp", "certificate_audit.json")))
    recs = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results/conformal", "main50", "*.json")) +
                    glob.glob(os.path.join(ROOT, "results/conformal", "budget50", "*.json"))):
        for r in json.load(open(f))["records"]:
            recs[(r["domain"], r["model"], r["seed"])] = r
    out = {}
    for dom, label in DOMS:
        x = au["domains"][dom]; n_ep = x["n"]
        rs_ship = {r["seed"]: r for r in x["arms"]["single"]["rows"]}
        rc_ship = {r["seed"]: r for r in x["arms"]["corset"]["rows"]}
        assert set(rs_ship) == set(rc_ship) and len(rs_ship) == 50, dom
        unc = np.mean([recs[(dom, "unconstrained", s)]["ret"] for s in rs_ship])
        rows = []
        for seed in sorted(rs_ship):
            rs = recs[(dom, "fqi:single", seed)]; rc = recs[(dom, "fqi:saorl", seed)]
            s_ship, c_ship = bool(rs_ship[seed]["ships"]), bool(rc_ship[seed]["ships"])
            if s_ship:
                stcr = dict(ship=True, ret=rs["ret"], runs=1, episodes=n_ep, arm="single",
                            viol_rate=rs_ship[seed]["p_hat"], true_worst=rs["true_worst"])
            elif c_ship:
                stcr = dict(ship=True, ret=rc["ret"], runs=2, episodes=2 * n_ep, arm="set",
                            viol_rate=rc_ship[seed]["p_hat"], true_worst=rc["true_worst"])
            else:
                stcr = dict(ship=False, ret=None, runs=2, episodes=2 * n_ep, arm=None, viol_rate=None, true_worst=None)
            arrow = dict(ship=c_ship, ret=(rc["ret"] if c_ship else None), runs=1, episodes=n_ep, arm=("set" if c_ship else None),
                         viol_rate=(rc_ship[seed]["p_hat"] if c_ship else None), true_worst=(rc["true_worst"] if c_ship else None))
            rows.append(dict(seed=seed, single_ret=rs["ret"], set_ret=rc["ret"], single_ships=s_ship, set_ships=c_ship,
                             single_viol=rs_ship[seed]["p_hat"], set_viol=rc_ship[seed]["p_hat"], stcr=stcr, arrow=arrow))

        def agg(key):
            P = [r[key] for r in rows]; shipped = [p for p in P if p["ship"]]
            return dict(ship_rate=float(np.mean([p["ship"] for p in P])), abstain_rate=float(np.mean([not p["ship"] for p in P])),
                        runs_mean=float(np.mean([p["runs"] for p in P])), episodes_mean=float(np.mean([p["episodes"] for p in P])),
                        shipped_ret_mean=(float(np.mean([p["ret"] for p in shipped])) if shipped else None),
                        shipped_ret_over_unconstrained=(float(np.mean([p["ret"] for p in shipped]) / unc) if shipped else None),
                        shipped_viol_rate_mean=(float(np.mean([p["viol_rate"] for p in shipped])) if shipped else None),
                        shipped_from_single=int(sum(1 for p in shipped if p["arm"] == "single")),
                        shipped_from_set=int(sum(1 for p in shipped if p["arm"] == "set")))
        out[dom] = dict(label=label, n_seeds=len(rows), episodes_per_check=n_ep, budget=x["budget"], B=x["B"],
                        unconstrained_ret_mean=float(unc),
                        single_ret_mean=float(np.mean([r["single_ret"] for r in rows])), set_ret_mean=float(np.mean([r["set_ret"] for r in rows])),
                        single_ships=int(sum(r["single_ships"] for r in rows)), set_ships=int(sum(r["set_ships"] for r in rows)),
                        single_viol_mean=float(np.mean([r["single_viol"] for r in rows])), set_viol_mean=float(np.mean([r["set_viol"] for r in rows])),
                        stcr=agg("stcr"), arrow=agg("arrow"), rows=rows)
    return out


# --------------------------------------------------------------------------- Part B
def occ_lp_subset(lg: Log, ks, d_target: float):
    """Occupancy LP on the estimated model with behaviour support, constraints on the cost rows ks only."""
    nS, nA = lg.nS, lg.nA
    block = np.kron(np.eye(nS), np.ones((1, nA)))
    A_eq = block - GAMMA * np.transpose(lg.Phat, (2, 0, 1)).reshape(nS, nS * nA)
    b_eq = (1.0 - GAMMA) * lg.mu0
    bounds = [(0, None) if lg.support[s, a] else (0, 0) for s in range(nS) for a in range(nA)]
    res = linprog(-lg.r.reshape(-1), A_ub=np.array([lg.c[k].reshape(-1) for k in ks]), b_ub=np.full(len(ks), d_target),
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        return None
    x = res.x.reshape(nS, nA); tot = x.sum(axis=1); pi = np.zeros((nS, nA)); vis = tot > 1e-12
    pi[vis] = x[vis] / tot[vis, None]
    for s in np.where(~vis)[0]:
        pi[s, int(np.argmax(np.where(lg.support[s], lg.N[s], -1)))] = 1.0
    return pi


def pipeline(first, first_chk, second, second_chk, n_ev):
    """STCR composition from per-arm scores and Check outcomes. Returns the shipped record."""
    if first_chk["ship"]:
        return dict(ship=True, arm="single", runs=1, samples=n_ev, ret_frac=first["ret_frac"], cmax_over_d=first["cmax_over_d"],
                    unsafe_ship=bool(not first["safe"]))
    if second is None or second_chk is None:
        return dict(ship=False, arm=None, runs=2, samples=2 * n_ev, ret_frac=None, cmax_over_d=None, unsafe_ship=False)
    if second_chk["ship"]:
        return dict(ship=True, arm="set", runs=2, samples=2 * n_ev, ret_frac=second["ret_frac"], cmax_over_d=second["cmax_over_d"],
                    unsafe_ship=bool(not second["safe"]))
    return dict(ship=False, arm=None, runs=2, samples=2 * n_ev, ret_frac=None, cmax_over_d=None, unsafe_ship=False)


def part_b():
    sel = json.load(open(os.path.join(R, "safe_face_select.json")))
    reg = {(r["rule_id"], round(r["budget"], 6)): r for r in sel["rows"]}
    bt = {r["rule_id"]: r for r in json.load(open(os.path.join(R, "baseline_table.json")))["compiled"]["rows"]}
    pt, _ = parse_prometheus(); bank = prom_threshold_bank(pt)
    rows = []; t0 = time.time()
    for ti, t in enumerate(pt):
        ok, _w, rd = _eligible(build_prom_pool(t, bank))
        if not ok:
            continue
        rid = f'{getattr(t, "name", "?")}#{ti}'
        m = compile_instance(rd); P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]
        assert C.shape[0] == 2, rid
        top1, top1_alt = bt[rid]["top1_index"], bt[rid]["top1_alt_index"]
        for d in D_GRID:
            rr = reg[(rid, round(d, 6))]; V_U = rr["V_U"]
            decide = rr["anchor"] if rr["regime"] == "irrelevant" else None    # Decide's zero-query certificate (declared model)
            pi_b = behaviour_policy(m, 1, d)
            for n in N_GRID:
                for seed in SEEDS:
                    rng = np.random.default_rng(abs(hash((rid, d, n, seed, "v47"))) % (2 ** 32))
                    lg = Log(sample_log(m, pi_b, n, rng))
                    arms = {}
                    for lc in ("fqi", "lp"):
                        if lc == "fqi":
                            pols = dict(single0=learn_single(lg, 0, d), single1=learn_single(lg, 1, d), fullset=learn_fullset(lg, 0, d))
                        else:
                            pols = dict(single0=occ_lp_subset(lg, [0], d), single1=occ_lp_subset(lg, [1], d), fullset=occ_lp_subset(lg, [0, 1], d))
                        sc, ck = {}, {}
                        for name, pi in pols.items():
                            if pi is None:
                                sc[name] = None; ck[name] = None; continue
                            s = score(P, mu0, r, C, pi, d)
                            s["ret_frac"] = s["J_r"] / V_U; s["cmax_over_d"] = s["C_max"] / d
                            crng = np.random.default_rng(abs(hash((rid, d, n, seed, lc, name, "chk"))) % (2 ** 32))
                            sc[name] = s; ck[name] = check_ship(P, mu0, C, pi, d, N_EV, crng, DELTA_EV)
                        pipes = {}
                        for lab, k in (("stcr_permissive", 0), ("stcr_restrictive", 1), ("stcr_top1", top1), ("stcr_top1_alt", top1_alt)):
                            nm = f"single{k}"
                            if sc[nm] is None:      # LP infeasible on P_hat: treat as a failed first stage
                                pipes[lab] = pipeline(dict(ret_frac=None, cmax_over_d=None, safe=False), dict(ship=False), sc["fullset"], ck["fullset"], N_EV)
                            else:
                                pipes[lab] = pipeline(sc[nm], ck[nm], sc["fullset"], ck["fullset"], N_EV)
                        # ARROW: Decide (declared model) -> one training run -> one Check
                        anm = f"single{decide}" if decide is not None else "fullset"
                        if sc[anm] is None:
                            pipes["arrow"] = dict(ship=False, arm=None, runs=1, samples=N_EV, ret_frac=None, cmax_over_d=None, unsafe_ship=False)
                        else:
                            pipes["arrow"] = dict(ship=bool(ck[anm]["ship"]), arm=("single" if decide is not None else "set"), runs=1, samples=N_EV,
                                                  ret_frac=(sc[anm]["ret_frac"] if ck[anm]["ship"] else None),
                                                  cmax_over_d=(sc[anm]["cmax_over_d"] if ck[anm]["ship"] else None),
                                                  unsafe_ship=bool(ck[anm]["ship"] and not sc[anm]["safe"]))
                        arms[lc] = dict(scores={k: (None if v is None else dict(ret_frac=v["ret_frac"], cmax_over_d=v["cmax_over_d"], safe=v["safe"])) for k, v in sc.items()},
                                        checks={k: (None if v is None else dict(ship=v["ship"], bound=v["bound"])) for k, v in ck.items()},
                                        pipelines=pipes)
                    rows.append(dict(rule_id=rid, d=d, n=n, seed=seed, V_U=V_U, regime=rr["regime"], decide=decide, top1=top1, top1_alt=top1_alt, arms=arms))
        print(f"  {rid:40s} done ({time.time()-t0:.0f}s)", flush=True)
    # ---- aggregate
    agg = {}
    for d in D_GRID:
        for n in N_GRID:
            for lc in ("fqi", "lp"):
                sub = [x for x in rows if x["d"] == d and x["n"] == n]
                a = {}
                for lab in ("stcr_permissive", "stcr_restrictive", "stcr_top1", "stcr_top1_alt", "arrow"):
                    P = [x["arms"][lc]["pipelines"][lab] for x in sub]; shipped = [p for p in P if p["ship"]]
                    a[lab] = dict(n=len(P), ship_rate=float(np.mean([p["ship"] for p in P])), abstain_rate=float(np.mean([not p["ship"] for p in P])),
                                  runs_mean=float(np.mean([p["runs"] for p in P])), samples_mean=float(np.mean([p["samples"] for p in P])),
                                  shipped_ret_frac_median=(float(np.median([p["ret_frac"] for p in shipped])) if shipped else None),
                                  shipped_ret_frac_mean=(float(np.mean([p["ret_frac"] for p in shipped])) if shipped else None),
                                  shipped_cmax_over_d_median=(float(np.median([p["cmax_over_d"] for p in shipped])) if shipped else None),
                                  unsafe_ship_rate=float(np.mean([p["unsafe_ship"] for p in P])),
                                  shipped_from_single=int(sum(1 for p in shipped if p["arm"] == "single")),
                                  shipped_from_set=int(sum(1 for p in shipped if p["arm"] == "set")))
                # per-arm truth: safe fraction and Check pass fraction, to separate pipeline effects from Check slack
                for name in ("single0", "single1", "fullset"):
                    S = [x["arms"][lc]["scores"][name] for x in sub]; Cc = [x["arms"][lc]["checks"][name] for x in sub]
                    ok = [(s, c) for s, c in zip(S, Cc) if s is not None]
                    a[f"arm_{name}"] = dict(n=len(ok), safe_frac=float(np.mean([s["safe"] for s, _ in ok])) if ok else None,
                                            check_pass_frac=float(np.mean([c["ship"] for _, c in ok])) if ok else None,
                                            safe_but_rejected_frac=float(np.mean([(s["safe"] and not c["ship"]) for s, c in ok])) if ok else None,
                                            ret_frac_median=float(np.median([s["ret_frac"] for s, _ in ok])) if ok else None,
                                            cmax_over_d_median=float(np.median([s["cmax_over_d"] for s, _ in ok])) if ok else None,
                                            infeasible=int(sum(1 for s in S if s is None)))
                agg[f"d{d}_n{n}_{lc}"] = a
    return dict(aggregate=agg, rows=rows)


def main():
    t0 = time.perf_counter()
    res = dict(registration="V47 (REGISTRATION_V28.md)", n_ev=N_EV, delta_ev=DELTA_EV, d_grid=list(D_GRID), n_grid=list(N_GRID), seeds=len(SEEDS))
    res["part_a"] = part_a()
    print("PART A (archived controlled domains, CP gate at 300 episodes):")
    for dom, x in res["part_a"].items():
        s, a = x["stcr"], x["arrow"]
        print(f"  {x['label']:22s} single ships {x['single_ships']}/50 set ships {x['set_ships']}/50 | STCR: ship {s['ship_rate']:.2f} runs {s['runs_mean']:.2f} episodes {s['episodes_mean']:.0f} ret/unc {s['shipped_ret_over_unconstrained']} viol {s['shipped_viol_rate_mean']} | ARROW: ship {a['ship_rate']:.2f} runs {a['runs_mean']:.2f} episodes {a['episodes_mean']:.0f} ret/unc {a['shipped_ret_over_unconstrained']} viol {a['shipped_viol_rate_mean']}")
    b = part_b(); res["part_b"] = b
    res["seconds"] = round(time.perf_counter() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} in {res['seconds']} s")
    print("PART B (compiled suite):")
    for key, a in b["aggregate"].items():
        print(f"== {key}")
        for lab in ("stcr_permissive", "stcr_restrictive", "stcr_top1", "stcr_top1_alt", "arrow"):
            v = a[lab]
            print(f"   {lab:17s} ship {v['ship_rate']:.3f} abstain {v['abstain_rate']:.3f} runs {v['runs_mean']:.2f} samples {v['samples_mean']:.0f} ret med {v['shipped_ret_frac_median']} unsafe {v['unsafe_ship_rate']:.3f} from single/set {v['shipped_from_single']}/{v['shipped_from_set']}")
        for name in ("single0", "single1", "fullset"):
            v = a[f"arm_{name}"]
            print(f"   arm {name:8s} safe {v['safe_frac']} check-pass {v['check_pass_frac']} safe-but-rejected {v['safe_but_rejected_frac']} ret med {v['ret_frac_median']} cmax/d med {v['cmax_over_d_median']} infeasible {v['infeasible']}")


if __name__ == "__main__":
    main()
