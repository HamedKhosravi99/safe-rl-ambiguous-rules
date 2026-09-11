"""V41: what K costs the learner.  On the compiled monitoring instances, the
two retained readings are padded with K-2 REDUNDANT constraints (convex
combinations of the two real cost vectors, which any policy feasible for the
pair satisfies automatically), so the feasible set is identical for every K
and only the constraint interface grows.  For each K we time (i) the
occupancy LP on the estimated model (occ_lp_policy) and (ii) Lagrangian
fitted Q-iteration with one multiplier per constraint (learn_fullset), and
score every returned policy exactly on the true model.  Registration V41.

Run: OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.k_scaling
Writes results/e2e/k_scaling.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .control_suite import _eligible, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool
from .safe_face_offline import Log, behaviour_policy, sample_log, learn_fullset, occupancy, DUAL_ITERS
from .certified_safe_face import occ_lp_policy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "results/e2e", "k_scaling.json")
KS = (2, 4, 8, 16, 64)
D = 0.005          # the middle-regime budget used by V35/V38
RHO = 0.5          # buffer fraction, as in V35/V38 (d_target = d - rho*d)
N_LOG = 20_000
SEEDS = 5


def padded(lg_dict: dict, K: int) -> dict:
    """Return a log dict whose per-step cost array has K rows: the two real
    readings followed by K-2 convex combinations lam*c1 + (1-lam)*c2."""
    C = lg_dict["C"]
    assert C.shape[0] == 2
    if K == 2:
        return lg_dict
    lams = np.linspace(0.0, 1.0, K)[1:-1]          # strictly inside (0,1)
    extra = [lam * C[0] + (1.0 - lam) * C[1] for lam in lams]
    out = dict(lg_dict)
    out["C"] = np.vstack([C] + extra)
    out["K"] = K
    return out


def true_scores(m: dict, pi: np.ndarray):
    x = occupancy(m["P"], m["mu0"], pi)
    jr = float((m["r"] * x).sum())
    jc = [float((m["C"][k] * x).sum()) for k in range(m["C"].shape[0])]
    return jr, max(jc)


def main() -> None:
    pt, _ = parse_prometheus(); tb = prom_threshold_bank(pt)
    rows = []; t0 = time.time()
    for ti, t in enumerate(pt):
        ok, _w, rd = _eligible(build_prom_pool(t, tb))
        if not ok:
            continue
        uid = f'{getattr(t, "name", "?")}#{ti}'
        m = compile_instance(rd)
        if m["C"].shape[0] != 2:
            continue
        pi_b = behaviour_policy(m, 0, D)
        for seed in range(SEEDS):
            rng = np.random.default_rng(1000 * ti + seed)
            base = sample_log(m, pi_b, N_LOG, rng)
            for K in KS:
                lg = Log(padded(base, K))
                t1 = time.perf_counter(); pi_lp = occ_lp_policy(lg, D - RHO * D); t_lp = time.perf_counter() - t1
                t2 = time.perf_counter(); pi_fq = learn_fullset(lg, 0, D - RHO * D); t_fq = time.perf_counter() - t2
                # dual iterations actually used: re-run the loop counting (cheap, deterministic)
                lam = np.zeros(lg.K); iters = DUAL_ITERS
                for j in range(DUAL_ITERS):
                    shaped = lg.r - np.tensordot(lam, lg.c, axes=(0, 0))
                    pi = lg.greedy(lg.fqi(shaped)); _jr, jc = lg.est(pi)
                    if (jc <= D - RHO * D + 1e-9).all():
                        iters = j + 1; break
                    lam = np.maximum(0.0, lam + 400.0 * (jc - (D - RHO * D)))
                rec = dict(uid=uid, seed=seed, K=K, t_lp=t_lp, t_fqi=t_fq, fqi_iters=iters,
                           lp_feasible=pi_lp is not None)
                if pi_lp is not None:
                    jr, cmax = true_scores(m, pi_lp); rec.update(lp_return=jr, lp_cmax=cmax, lp_safe=bool(cmax <= D + 1e-9))
                jr, cmax = true_scores(m, pi_fq); rec.update(fqi_return=jr, fqi_cmax=cmax, fqi_safe=bool(cmax <= D + 1e-9))
                rows.append(rec)
        print(f"  {uid:38s} done ({time.time()-t0:.0f}s)", flush=True)
    # ---- aggregate per K ----
    agg = {}
    for K in KS:
        R = [r for r in rows if r["K"] == K]
        lp = [r for r in R if r["lp_feasible"]]
        agg[str(K)] = dict(n=len(R),
                           t_lp_med=float(np.median([r["t_lp"] for r in R])), t_lp_mean=float(np.mean([r["t_lp"] for r in R])),
                           t_fqi_med=float(np.median([r["t_fqi"] for r in R])), t_fqi_mean=float(np.mean([r["t_fqi"] for r in R])),
                           fqi_iters_med=float(np.median([r["fqi_iters"] for r in R])),
                           fqi_converged=float(np.mean([r["fqi_iters"] < DUAL_ITERS for r in R])),
                           lp_feasible=float(np.mean([r["lp_feasible"] for r in R])),
                           lp_return=float(np.mean([r["lp_return"] for r in lp])) if lp else None,
                           lp_safe=float(np.mean([r["lp_safe"] for r in lp])) if lp else None,
                           fqi_return=float(np.mean([r["fqi_return"] for r in R])), fqi_safe=float(np.mean([r["fqi_safe"] for r in R])))
    # paired identity of the LP policy's true return across K (feasible set unchanged by construction)
    by = {}
    for r in rows:
        if r["lp_feasible"]:
            by.setdefault((r["uid"], r["seed"]), {})[r["K"]] = r["lp_return"]
    diffs = [abs(v[K] - v[2]) for v in by.values() for K in KS if K in v and 2 in v]
    ident = dict(n_pairs=len(diffs), max_abs_return_diff=float(max(diffs)) if diffs else None)
    payload = dict(registration="V41 (REGISTRATION_V28.md): constraint count vs learner cost, redundant convex-combination padding",
                   d=D, rho=RHO, n_log=N_LOG, seeds=SEEDS, Ks=list(KS), aggregate=agg, lp_identity=ident, rows=rows,
                   runtime_s=time.time() - t0)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {OUT}")
    print(f"{'K':>4} | {'LP ms (med)':>11} {'FQI ms (med)':>12} {'FQI iters':>9} {'FQI conv':>8} | {'LP ret':>7} {'LP safe':>7} | {'FQI ret':>7} {'FQI safe':>8}")
    for K in KS:
        a = agg[str(K)]
        print(f"{K:>4} | {1e3*a['t_lp_med']:>11.1f} {1e3*a['t_fqi_med']:>12.1f} {a['fqi_iters_med']:>9.0f} {a['fqi_converged']:>8.2f} | "
              f"{(a['lp_return'] or 0):>7.4f} {(a['lp_safe'] or 0):>7.2f} | {a['fqi_return']:>7.4f} {a['fqi_safe']:>8.2f}")
    print("LP true-return identity across K:", ident)


if __name__ == "__main__":
    main()
