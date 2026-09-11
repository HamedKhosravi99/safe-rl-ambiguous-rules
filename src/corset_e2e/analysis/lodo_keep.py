"""REGISTRATION_V22: leave-one-organization-out coverage of Keep.

Re-runs the frozen v11 evaluation on the grammar-reachable calibration
units (asserting the archived q-hat = 0.41), joins them with the archived
frozen test rows, and for every organization (repository) calibrates the
Keep threshold on the in-grammar golds of every OTHER organization and
measures retention on the held-out one.  Also computes the
within-organization (Mondrian-by-organization) repair by leave-one-unit-out
inside each eligible organization, and the organization-exchangeable
version that subsamples one unit per other organization.  No unit is
re-licensed; nothing is tuned.

Run: PYTHONPATH=. python3 corset_e2e/analysis/lodo_keep.py
Writes results/e2e/lodo_keep.json
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.analysis.faithful_recalibration import faithful_g0, qhat  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import hid, load_split  # noqa: E402
from corset_e2e.calibration.run_e2e_v11 import V11Server, evaluate  # noqa: E402
from corset_e2e.dsl.schema import IN_DSL, classify_target  # noqa: E402

R = os.path.join(ROOT, "results/e2e")
OUT = os.path.join(R, "lodo_keep.json")
DELTA = 0.10
MIN_CAL = 9          # smallest n with floor(DELTA * (n + 1)) >= 1
DRAWS = 2000
SEED = 0
NOMINAL = 1.0 - DELTA


def _bisect(f, lo, hi, it=200):
    flo = f(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cp_interval(k, n, alpha=0.05):
    """Exact Clopper-Pearson interval for k successes in n trials."""
    if n == 0:
        return None, None

    def cdf(p, j):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(j + 1))

    lo = 0.0 if k == 0 else _bisect(lambda p: 1 - cdf(p, k - 1) - alpha / 2, 0.0, 1.0)
    hi = 1.0 if k == n else _bisect(lambda p: cdf(p, k) - alpha / 2, 0.0, 1.0)
    return lo, hi


def retained(u, q):
    return u["gold_score"] >= q - 1e-12


def main():
    t0 = time.perf_counter()
    rng = random.Random(SEED)
    rep = json.load(open(os.path.join(R, "e2e_report_v11.json")))
    weights = rep["weights"]
    q_arch = rep["modes"]["complete"]["qhat"]
    cal = [e for e in load_split("cal") if classify_target(hid(e["rule_id"])) == IN_DSL]
    print(f"grammar-reachable cal units: {len(cal)}", flush=True)
    server = V11Server()
    rows = evaluate(cal, server, weights, tag="cal", mode="complete")["rows"]
    cal_in = [r for r in rows if r["label"] == IN_DSL and r["gold_score"] is not None]
    q_rec, k_rec = qhat([r["gold_score"] for r in cal_in])
    assert abs(q_rec - q_arch) < 1e-9, (q_rec, q_arch)
    test = json.load(open(os.path.join(R, "e2e_test_rows_v11_complete.json")))
    tin = [r for r in test if r["label"] == IN_DSL and r["gold_score"] is not None]
    assert len(cal_in) == 251 and len(tin) == 181, (len(cal_in), len(tin))
    units = []
    for split, rs in (("cal", cal_in), ("test", tin)):
        for r in rs:
            units.append(dict(rid=r["rid"], repo=r["repo"], split=split,
                              gold_score=r["gold_score"], faithful=bool(faithful_g0(hid(r["rid"])))))
    orgs = sorted({u["repo"] for u in units})
    q_all, k_all = qhat([u["gold_score"] for u in units])
    print(f"units {len(units)} orgs {len(orgs)} pooled in-sample q-hat {q_all:.4f} (k={k_all}); archived {q_arch}", flush=True)

    per_org = {}
    for O in orgs:
        held = [u for u in units if u["repo"] == O]
        rest = [u for u in units if u["repo"] != O]
        n = len(held)
        q, k = qhat([u["gold_score"] for u in rest])
        r = sum(retained(u, q) for u in held)
        lo, hi = cp_interval(r, n)
        two_se = NOMINAL - 2 * math.sqrt(NOMINAL * DELTA / n)
        hf = [u for u in held if u["faithful"]]
        rf = sum(retained(u, q) for u in hf)
        r_arch = sum(retained(u, q_arch) for u in held)
        # within-organization leave-one-unit-out (Mondrian by organization)
        within = dict(eligible=n - 1 >= MIN_CAL)
        if within["eligible"]:
            loo = 0
            for i, u in enumerate(held):
                qi, _ = qhat([v["gold_score"] for j, v in enumerate(held) if j != i])
                loo += retained(u, qi)
            wlo, whi = cp_interval(loo, n)
            q_in, k_in = qhat([u["gold_score"] for u in held])
            within.update(retained=loo, retention=loo / n, cp_lo=wlo, cp_hi=whi,
                          qhat_in_org=q_in, k_in_org=k_in)
        # organization-exchangeable: one unit per other organization
        groups = [[u for u in rest if u["repo"] == P] for P in orgs if P != O]
        m = len(groups)
        cluster = dict(eligible=m >= MIN_CAL, m=m)
        if cluster["eligible"]:
            acc = 0.0
            qs = []
            for _ in range(DRAWS):
                draw = [rng.choice(g)["gold_score"] for g in groups]
                qd, _ = qhat(draw)
                qs.append(qd)
                acc += sum(retained(u, qd) for u in held) / n
            cluster.update(retention=acc / DRAWS, median_qhat=statistics.median(qs),
                           k=max(1, math.floor(DELTA * (m + 1))), draws=DRAWS)
        per_org[O] = dict(
            n=n, n_cal=sum(u["split"] == "cal" for u in held), n_test=sum(u["split"] == "test" for u in held),
            median_gold_score=statistics.median(u["gold_score"] for u in held),
            qhat_lodo=q, k_lodo=k, n_cal_fold=len(rest),
            retained=r, retention=r / n, cp_lo=lo, cp_hi=hi,
            two_se_floor=two_se, fails_two_se=bool(r / n < two_se), cp_upper_below_nominal=bool(hi < NOMINAL),
            n_faithful=len(hf), retained_faithful=rf, retention_faithful=(rf / len(hf) if hf else None),
            retained_archived=r_arch, retention_archived=r_arch / n,
            within_org=within, cluster=cluster)
        print(f"  {O:30s} n={n:4d} q-hat_-O={q:.4f} retention={r/n:.3f} [{lo:.3f},{hi:.3f}] "
              f"fail2se={per_org[O]['fails_two_se']} within={within.get('retention')} cluster={cluster.get('retention')}", flush=True)

    rets = {O: v["retention"] for O, v in per_org.items()}
    min_org = min(rets, key=rets.get)
    max_org = max(rets, key=rets.get)
    fail_orgs = [O for O, v in per_org.items() if v["fails_two_se"]]
    within_orgs = [O for O, v in per_org.items() if v["within_org"]["eligible"]]
    cluster_orgs = [O for O, v in per_org.items() if v["cluster"]["eligible"]]
    summary = dict(
        n_units=len(units), n_cal=len(cal_in), n_test=len(tin), n_orgs=len(orgs), orgs=orgs,
        qhat_archived=q_arch, qhat_pooled_in_sample=q_all,
        min_retention=rets[min_org], min_org=min_org, max_retention=rets[max_org], max_org=max_org,
        spread=rets[max_org] - rets[min_org],
        macro_avg_retention=statistics.mean(rets.values()),
        pooled_lodo_retention=sum(v["retained"] for v in per_org.values()) / len(units),
        n_fail_two_se=len(fail_orgs), fail_orgs=fail_orgs,
        n_cp_upper_below_nominal=sum(v["cp_upper_below_nominal"] for v in per_org.values()),
        cp_upper_below_orgs=[O for O, v in per_org.items() if v["cp_upper_below_nominal"]],
        n_within_eligible=len(within_orgs), within_orgs=within_orgs,
        within_ineligible=[O for O in orgs if O not in within_orgs],
        within_macro_avg=(statistics.mean(per_org[O]["within_org"]["retention"] for O in within_orgs) if within_orgs else None),
        within_min=(min(per_org[O]["within_org"]["retention"] for O in within_orgs) if within_orgs else None),
        within_pooled=(sum(per_org[O]["within_org"]["retained"] for O in within_orgs) / sum(per_org[O]["n"] for O in within_orgs) if within_orgs else None),
        n_cluster_eligible=len(cluster_orgs),
        cluster_macro_avg=(statistics.mean(per_org[O]["cluster"]["retention"] for O in cluster_orgs) if cluster_orgs else None),
        cluster_min=(min(per_org[O]["cluster"]["retention"] for O in cluster_orgs) if cluster_orgs else None),
        cluster_median_qhat=(statistics.median(per_org[O]["cluster"]["median_qhat"] for O in cluster_orgs) if cluster_orgs else None),
        fail_orgs_repaired_within=[O for O in fail_orgs if O in within_orgs and per_org[O]["within_org"]["retention"] >= per_org[O]["two_se_floor"]],
    )
    res = dict(registration="REGISTRATION_V22.md", delta_sem=DELTA, min_cal=MIN_CAL, draws=DRAWS, seed=SEED,
               summary=summary, per_org=per_org, seconds=round(time.perf_counter() - t0, 1))
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"wrote {OUT} in {res['seconds']} s")


if __name__ == "__main__":
    main()
