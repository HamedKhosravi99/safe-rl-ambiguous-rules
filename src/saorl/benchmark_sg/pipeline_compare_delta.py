"""V47b: the sequential pipeline (singleton -> Check -> retrain on the set ->
Check) against ARROW under a FIXED total evaluator failure budget. Same
logs and seeds as V47; per-reading sample statistics are stored so every
delta allocation is evaluated on the same fresh samples. Decide's LP count
and wall-clock are measured on the declared model.

Run: PYTHONHASHSEED=0 OMP_NUM_THREADS=1 SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.pipeline_compare_delta
Writes results/e2e/pipeline_compare_delta.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .control_suite import _eligible, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool
from .safe_face_offline import Log, behaviour_policy, sample_log, learn_single, learn_fullset, score, occupancy
from .pipeline_compare import occ_lp_subset
from .collapse_utility import lp_true

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "results/e2e")
OUT = os.path.join(R, "pipeline_compare_delta.json")
D_GRID = (0.05, 0.005)
N_GRID = (2000, 20000)
SEEDS = tuple(range(10))
N_EV = 20000
DELTA_TOTAL = 0.05
ALLOC = {"equal": (0.025, 0.025), "front": (0.04, 0.01), "back": (0.01, 0.04), "naive_reference": (0.05, 0.05)}
FALLBACK_RET = 0.3          # always-drain: reward 0.3, cost 0, safe by construction


def check_stats(P, mu0, C, pi, n_ev, rng):
    x = occupancy(P, mu0, pi).reshape(-1); x = np.maximum(x, 0); x = x / x.sum()
    idx = rng.choice(x.size, size=n_ev, p=x)
    st = []
    for k in range(C.shape[0]):
        z = C[k].reshape(-1)[idx]
        st.append(dict(mean=float(z.mean()), var=float(z.var(ddof=1)), rng=float(C[k].max()) or 1.0))
    return dict(n=n_ev, per_reading=st)


def bound_at(st, delta):
    K = len(st["per_reading"]); L = np.log(3.0 / (delta / K)); n = st["n"]
    return max(s["mean"] + np.sqrt(2.0 * s["var"] * L / n) + 3.0 * s["rng"] * L / n for s in st["per_reading"])


def decide_declared(m, d):
    """Value test + face test on the declared model; returns (certified reading or None, n_lps, seconds)."""
    t0 = time.perf_counter(); r = m["r"].reshape(-1); C = [m["C"][k].reshape(-1) for k in range(m["C"].shape[0])]; K = len(C); n = 0
    V = []
    for k in range(K):
        V.append(lp_true(m, r, [C[k]], [d])); n += 1
    V_U = lp_true(m, r, C, [d] * K); n += 1
    cert = None
    for k in range(K):
        if V[k] is None or V_U is None or abs(V[k] - V_U) > 1e-7:
            continue
        ok = True
        for j in range(K):
            if j == k:
                continue
            W = lp_true(m, C[j], [C[k], -r], [d, -(V[k] - 1e-7)]); n += 1
            if W is None or W > d + 1e-6:
                ok = False; break
        if ok:
            cert = k; break
    return cert, n, time.perf_counter() - t0, V, V_U


def main():
    pt, _ = parse_prometheus(); bank = prom_threshold_bank(pt)
    rows, dec_rows = [], []; t_start = time.time()
    for ti, t in enumerate(pt):
        ok, _w, rd = _eligible(build_prom_pool(t, bank))
        if not ok:
            continue
        rid = f'{getattr(t, "name", "?")}#{ti}'
        m = compile_instance(rd); P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]
        assert C.shape[0] == 2
        for d in D_GRID:
            cert, n_lp, t_dec, V, V_U = decide_declared(m, d)
            dec_rows.append(dict(rule_id=rid, d=d, certified=cert, n_lps=n_lp, seconds=t_dec, V=V, V_U=V_U))
            pi_b = behaviour_policy(m, 1, d)
            for n in N_GRID:
                for seed in SEEDS:
                    rng = np.random.default_rng(abs(hash((rid, d, n, seed, "v47"))) % (2 ** 32))
                    t1 = time.perf_counter(); lg = Log(sample_log(m, pi_b, n, rng)); t_log = time.perf_counter() - t1
                    arms = {}
                    for lc in ("fqi", "lp"):
                        t2 = time.perf_counter()
                        if lc == "fqi":
                            pols = dict(single0=learn_single(lg, 0, d), single1=learn_single(lg, 1, d), fullset=learn_fullset(lg, 0, d))
                        else:
                            pols = dict(single0=occ_lp_subset(lg, [0], d), single1=occ_lp_subset(lg, [1], d), fullset=occ_lp_subset(lg, [0, 1], d))
                        t_train = (time.perf_counter() - t2) / 3.0
                        sc, st = {}, {}
                        t3 = time.perf_counter()
                        for name, pi in pols.items():
                            if pi is None:
                                sc[name] = None; st[name] = None; continue
                            s = score(P, mu0, r, C, pi, d); s["ret_frac"] = s["J_r"] / V_U; s["cmax_over_d"] = s["C_max"] / d; sc[name] = s
                            crng = np.random.default_rng(abs(hash((rid, d, n, seed, lc, name, "chk"))) % (2 ** 32))
                            st[name] = check_stats(P, mu0, C, pi, N_EV, crng)
                        t_check = (time.perf_counter() - t3) / 3.0
                        arms[lc] = dict(scores={k: (None if v is None else dict(ret_frac=v["ret_frac"], cmax_over_d=v["cmax_over_d"], safe=v["safe"])) for k, v in sc.items()},
                                        stats=st, t_train_per_run=t_train, t_check_per_run=t_check)
                    rows.append(dict(rule_id=rid, d=d, n=n, seed=seed, V_U=V_U, certified=cert, t_log=t_log, arms=arms))
        print(f"  {rid:40s} done ({time.time()-t_start:.0f}s)", flush=True)

    # ---- evaluate pipelines under each allocation
    def ship(st, delta, d):
        return bool(st is not None and bound_at(st, delta) <= d)

    def seq(sc, st, first, d, d1, d2, n_ev):
        f, fs = sc[first], st[first]
        if f is not None and ship(fs, d1, d):
            return dict(ship=True, runs=1, samples=n_ev, ret=f["ret_frac"], unsafe=not f["safe"], arm="single")
        g, gs = sc["fullset"], st["fullset"]
        if g is not None and ship(gs, d2, d):
            return dict(ship=True, runs=2, samples=2 * n_ev, ret=g["ret_frac"], unsafe=not g["safe"], arm="set")
        return dict(ship=False, runs=2, samples=2 * n_ev, ret=None, unsafe=False, arm=None)

    def arrow(sc, st, cert, d, delta, n_ev):
        nm = f"single{cert}" if cert is not None else "fullset"
        a, as_ = sc[nm], st[nm]
        if a is not None and ship(as_, delta, d):
            return dict(ship=True, runs=1, samples=n_ev, ret=a["ret_frac"], unsafe=not a["safe"], arm=("single" if cert is not None else "set"))
        return dict(ship=False, runs=1, samples=n_ev, ret=None, unsafe=False, arm=None)

    agg = {}
    for d in D_GRID:
        for n in N_GRID:
            for lc in ("fqi", "lp"):
                sub = [x for x in rows if x["d"] == d and x["n"] == n]
                a = {}
                for alloc, (d1, d2) in ALLOC.items():
                    for lab, first in (("seq_other_reading", None), ("seq_certified_reading", None), ("arrow", None)):
                        outs = []
                        for x in sub:
                            sc, st = x["arms"][lc]["scores"], x["arms"][lc]["stats"]; cert = x["certified"]
                            if lab == "arrow":
                                o = arrow(sc, st, cert, d, DELTA_TOTAL, N_EV)
                            else:
                                # certified reading = index 1 on every instance where a certificate exists (asserted below); at d = 0.005 no certificate: use the value-tied reading (index 1)
                                c_idx = cert if cert is not None else 1
                                first_name = f"single{c_idx}" if lab == "seq_certified_reading" else f"single{1 - c_idx}"
                                o = seq(sc, st, first_name, d, d1, d2, N_EV)
                            o["fb"] = FALLBACK_RET / x["V_U"]; outs.append(o)
                        shipped = [o for o in outs if o["ship"]]
                        a[f"{alloc}|{lab}"] = dict(n=len(outs), ship_rate=float(np.mean([o["ship"] for o in outs])),
                                                  runs_mean=float(np.mean([o["runs"] for o in outs])), samples_mean=float(np.mean([o["samples"] for o in outs])),
                                                  ship_and_unsafe=float(np.mean([o["ship"] and o["unsafe"] for o in outs])),
                                                  shipped_ret_median=(float(np.median([o["ret"] for o in shipped])) if shipped else None),
                                                  shipped_ret_mean=(float(np.mean([o["ret"] for o in shipped])) if shipped else None),
                                                  utility_abstain0=float(np.mean([(o["ret"] if o["ship"] else 0.0) for o in outs])),
                                                  utility_fallback=float(np.mean([(o["ret"] if o["ship"] else o["fb"]) for o in outs])),
                                                  shipped_from_single=int(sum(1 for o in shipped if o["arm"] == "single")))
                a["timing"] = dict(t_train_per_run_median=float(np.median([x["arms"][lc]["t_train_per_run"] for x in sub])),
                                   t_check_per_run_median=float(np.median([x["arms"][lc]["t_check_per_run"] for x in sub])),
                                   t_log_median=float(np.median([x["t_log"] for x in sub])))
                agg[f"d{d}_n{n}_{lc}"] = a
    dec = {}
    for d in D_GRID:
        dr = [x for x in dec_rows if x["d"] == d]
        dec[str(d)] = dict(n=len(dr), certified=int(sum(1 for x in dr if x["certified"] is not None)), certified_index=sorted({x["certified"] for x in dr if x["certified"] is not None}),
                           n_lps_median=float(np.median([x["n_lps"] for x in dr])), n_lps_max=int(max(x["n_lps"] for x in dr)),
                           seconds_median=float(np.median([x["seconds"] for x in dr])), seconds_max=float(max(x["seconds"] for x in dr)))
    res = dict(registration="V47b (REGISTRATION_V28.md)", delta_total=DELTA_TOTAL, allocations=ALLOC, n_ev=N_EV, fallback_ret=FALLBACK_RET,
               decide=dec, aggregate=agg, decide_rows=dec_rows, seconds=round(time.time() - t_start, 1))
    # rows are large (per-reading stats); keep them
    res["rows"] = rows
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} in {res['seconds']} s")
    print("DECIDE cost:", json.dumps(dec))
    for key, a in agg.items():
        print(f"== {key} | train/run {1e3*a['timing']['t_train_per_run_median']:.1f} ms, check/run {1e3*a['timing']['t_check_per_run_median']:.1f} ms, log {1e3*a['timing']['t_log_median']:.0f} ms")
        for alloc in ALLOC:
            for lab in ("seq_other_reading", "seq_certified_reading", "arrow"):
                v = a[f"{alloc}|{lab}"]
                print(f"   {alloc:16s} {lab:22s} ship {v['ship_rate']:.3f} unsafe {v['ship_and_unsafe']:.4f} runs {v['runs_mean']:.2f} samples {v['samples_mean']:.0f} ret med {v['shipped_ret_median']} U0 {v['utility_abstain0']:.3f} Ufb {v['utility_fallback']:.3f}")


if __name__ == "__main__":
    main()
