"""V15B: class-conditional (Mondrian) calibration for the G1 pipeline.

The V15 pass showed one global qhat gives marginal but not
class-conditional coverage across G1's two grammar classes. This script
recalibrates per class on the CAL split and re-thresholds the frozen V15
test scores; no test score is recomputed.

Run: PYTHONPATH=. python3 corset_e2e/calibration/run_g1_mondrian.py
Writes results/e2e/e2e_report_g1_mondrian.json
"""
from __future__ import annotations
import os
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))
from corset_e2e.dsl.schema import IN_DSL
from corset_e2e.dsl.schema_g1 import IN_G1_REF
from corset_e2e.calibration.run_e2e_g1 import evaluate
from corset_e2e.calibration.run_e2e_v11 import V11Server
from corset_e2e.calibration.run_e2e_v4 import DELTA_SEM, load_split
OUT = os.path.join(ROOT, "results/e2e")

def main() -> None:
    rep = json.load(open(os.path.join(OUT, "e2e_report_g1.json")))
    w = rep["weights"]
    server = V11Server()
    cal = load_split("cal")
    print(f"[1] recalibrate per class on cal ({len(cal)})")
    cal_rows = evaluate(cal, server, w)
    qhat = {}
    for cls in (IN_DSL, IN_G1_REF):
        g = sorted(r["gold_score"] for r in cal_rows
                   if r["label"] == cls and r["gold_score"] is not None)
        if len(g) < 20:
            raise SystemExit(f"only {len(g)} cal units in class {cls}")
        k = max(1, math.floor(DELTA_SEM * (len(g) + 1)))
        qhat[cls] = g[k - 1]
        print(f"    {cls:12s} n_cal={len(g):5d} k={k:4d} qhat={g[k-1]:.6f}")
    print("[2] re-threshold the frozen V15 test scores (no recomputation)")
    rows = json.load(open(os.path.join(OUT, "e2e_test_rows_g1.json")))
    out = {}
    n_ret = 0
    for cls in (IN_DSL, IN_G1_REF):
        lic = [r for r in rows if r["label"] == cls]
        ret = sum(1 for r in lic if r["gold_score"] is not None
                  and r["gold_score"] >= qhat[cls])
        n_ret += ret
        out[cls] = dict(qhat=qhat[cls], licensed=len(lic), retained=ret,
                        retention=ret / max(len(lic), 1))
        print(f"    {cls:12s} licensed={len(lic):4d} retained={ret:4d} "
              f"= {100*ret/max(len(lic),1):5.1f}%  (nominal {100*(1-DELTA_SEM):.0f}%)")
    n = len(rows)
    res = dict(registration="V15B (REGISTRATION_V15B.md): Mondrian "
                            "class-conditional calibration; V15 test scores "
                            "reused unchanged",
               delta_sem=DELTA_SEM, per_class=out,
               n_units=n, n_retained=n_ret, e2e_over_corpus=n_ret / n)
    json.dump(res, open(os.path.join(OUT, "e2e_report_g1_mondrian.json"), "w"),
              indent=1)
    print(f"\nEND-TO-END over {n}: {n_ret} = {100*n_ret/n:.1f}%")


if __name__ == "__main__":
    main()
