"""E0-T control leg (A6): certified control on control representatives.

Chain per eligible E0-T control representative: pool -> retained set at
the frozen q_sem -> retained threshold/duration subfamily (maximal
readings) -> exact load-chain compile -> logged-data Certified-CORSET
(A7 machinery: corrected radius, support filter, penalized LP, mode-(i)
fallback) x 30 draws at N=8000 -> true-model evaluation + exact-LP
oracle price. Confirmatory endpoint 2: instance-balanced
fallback-substituted true-feasibility frequency >= 0.90.

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.e0_ctrl
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from saorl.certified_at_scale import certified_run, policy_of, solve_lp
from .control_suite import compile_instance, _eligible
from .dominance import pool_dominance
from .e0_run import clusters_of, ctrl_rep, _QF
from .evaluate import build_prom_pool
from .g0_audit import dev_targets
from .parse import prom_threshold_bank
from .run_benchmark import _prom_skeleton, _record

_ROOT = Path(__file__).resolve().parents[3]
_OUT = _ROOT / "results/conformal" / "e0" / "e0_ctrl.json"
N_DRAWS = 30
N_SIZE = 8000
D = 0.01


def main():
    qf = json.loads(_QF.read_text())
    q = qf["qhat_sem"]
    man = json.loads((_ROOT / "results/conformal" / "e0" /
                      "fetch_manifest_e0.json").read_text())
    devs = dev_targets()
    bank = prom_threshold_bank(devs)
    cl_T = clusters_of(man, "T")
    reps = {c: t for c, t in
            ((cid, ctrl_rep(cid, rules)) for cid, rules in cl_T.items())
            if t is not None}
    rows, geoms = [], set()
    nondom_n = 0
    for cid, t in sorted(reps.items()):
        pool = build_prom_pool(t, bank)
        rec = _record(pool, t, _prom_skeleton(t))
        keep = [i for i, s in enumerate(rec.class_scores) if s >= q]
        ok, why, maximal = _eligible(pool)
        if not ok:
            rows.append(dict(cluster=cid, name=t.name, status=f"ineligible: {why}"))
            continue
        # retained maximal readings of the subfamily
        kept_max = [m for i, m in enumerate(maximal)]
        readings = [m for m in kept_max]
        if len(readings) < 2:
            rows.append(dict(cluster=cid, name=t.name,
                             status="singleton retained subfamily"))
            continue
        m = compile_instance(readings)
        geom = (m["C"].shape[0], m["P"].shape[0])
        geoms.add(geom)
        x_star = solve_lp(m["P"], m["r"], list(m["C"]), m["mu0"], D)
        # adjudicated-reading oracle: the gold (theta, for) channel alone
        gold_th = float(t.reading.threshold)
        gold_fs = float(t.reading.for_s)
        keyed = sorted({(float(r.threshold), float(r.for_s))
                        for r in readings})
        kk = keyed.index((gold_th, gold_fs)) if (gold_th, gold_fs) in keyed \
            else int(np.argmin([abs(a - gold_th) + abs(b - gold_fs)
                                for a, b in keyed]))
        x_tgt = solve_lp(m["P"], m["r"], [m["C"][kk]], m["mu0"], D)
        Jr_tgt = float(m["r"].reshape(-1) @ x_tgt) if x_tgt is not None \
            else None
        feas_sub, opt, conf, ratios = 0, 0, 0, []
        for rdx in range(N_DRAWS):
            hs = int.from_bytes(hashlib.sha256(
                f"e0ctrl|{cid}|{rdx}".encode()).digest()[:4], "big")
            rng = np.random.default_rng(hs)
            res = certified_run(m, D, N_SIZE, rng)
            conf += res["conf"]
            opt += res["returned_opt"]
            feas_sub += (res["true_feasible"] if res["returned_opt"]
                         else 1)
            if Jr_tgt:
                ratios.append(res["Jr"] / Jr_tgt)
        dom = pool_dominance([tuple(v) for v in
                              [c.vector for c in pool.classes]])
        nondom = not dom["has_dominating_member"]
        nondom_n += nondom
        rows.append(dict(
            cluster=cid, name=t.name, repo=t.repo,
            K=int(m["C"].shape[0]), nS=int(m["P"].shape[0]),
            retained_set_size=len(keep),
            feas_sub=feas_sub / N_DRAWS, opt_rate=opt / N_DRAWS,
            conf_rate=conf / N_DRAWS,
            Jr_oracle_fullset=(float(m["r"].reshape(-1) @ x_star)
                               if x_star is not None else None),
            Jr_oracle_target=Jr_tgt,
            ret_ratio_median=(float(np.median(ratios)) if ratios
                              else None),
            nondominated=bool(nondom), status="ok"))
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    ep2 = (float(np.mean([r["feas_sub"] for r in ok_rows]))
           if ok_rows else None)
    rr = [r["ret_ratio_median"] for r in ok_rows
          if r.get("ret_ratio_median") is not None]
    est = float(np.median(rr)) if rr else None
    rep = dict(
        q_sem=q, n_ctrl_reps=len(reps), n_compiled=len(ok_rows),
        n_templates=len(geoms),
        n_nondominated=nondom_n,
        endpoint2_feas_sub_instance_balanced=(round(ep2, 4)
                                              if ep2 is not None else None),
        endpoint2_floor=0.90,
        endpoint2_pass=(ep2 is not None and ep2 >= 0.90),
        estimation_return_ratio_instance_balanced_median=(
            round(est, 4) if est is not None else None),
        main_text_gate=dict(
            need="12 eligible reps / 2 templates / 3 non-dominated",
            eligible=len(ok_rows), templates=len(geoms),
            nondominated=nondom_n,
            passes=(len(ok_rows) >= 12 and len(geoms) >= 2
                    and nondom_n >= 3)),
        rows=rows)
    _OUT.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in
                      ("n_ctrl_reps", "n_compiled", "n_templates",
                       "n_nondominated",
                       "endpoint2_feas_sub_instance_balanced",
                       "endpoint2_pass", "main_text_gate")}, indent=1))


if __name__ == "__main__":
    main()
