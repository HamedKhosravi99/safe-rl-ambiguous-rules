"""V11B endpoint 3: coverage-versus-conservatism sweep over licensing breadth.

Registered in REGISTRATION_V11B.md before execution: the FULL calibrated
pipeline (identical protocol to the registered k=64 pass -- dev weight
refit, cal qhat/qood, one execution) at k=16 and k=256, labeled post-hoc.
k=64 remains the primary operating point; these rows exist to state the
tradeoff between semantic coverage and decision conservatism as a curve.

Run: PYTHONPATH=. python3 corset_e2e/analysis/ksweep_v11.py
Writes results/e2e/ksweep_v11.json
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.schema import IN_DSL  # noqa: E402
from corset_e2e.generator import license_v11 as L11  # noqa: E402
from corset_e2e.calibration import run_e2e_v11 as R11  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import (  # noqa: E402
    DELTA_SEM, OOD_TARGET_RECALL, load_split, summarise)

OUT = os.path.join(ROOT, "results/e2e")
SWEEP = (16, 256)


def run_at(k: int, server, dev, cal, test) -> dict:
    old = L11.K_PRIMARY
    L11.K_PRIMARY = k
    try:
        fit = R11.fit_weights(dev, server)
        weights = fit["weights"]
        cal_m = R11.evaluate(cal, server, weights, tag="cal")
        gold = sorted(r["gold_score"] for r in cal_m["rows"]
                      if r["label"] == IN_DSL and r["gold_score"] is not None)
        n = len(gold)
        kq = max(1, math.floor(DELTA_SEM * (n + 1)))
        qh = gold[kq - 1]
        sig = sorted(r["ood_signal"] for r in cal_m["rows"]
                     if r["label"] != IN_DSL)
        qo = sig[max(0, int((1 - OOD_TARGET_RECALL) * len(sig)))] if sig else 1.0
        te = R11.evaluate(test, server, weights, qhat=qh, qood=qo, tag="test")
        s = summarise(te["rows"], qh, qo)
        return dict(k=k, qhat=qh, n_cal_in_dsl=n, weights=weights,
                    rho_gen=s["rho_gen"], rho_ret_given_gen=s["rho_ret_given_gen"],
                    rho_e2e=s["rho_e2e"], n_in_dsl=s["n_in_dsl"],
                    set_size=s["set_size"], maximal_set_size=s["maximal_set_size"])
    finally:
        L11.K_PRIMARY = old


def main() -> None:
    server = R11.V11Server()
    dev, cal, test = load_split("dev"), load_split("cal"), load_split("test")
    rows = []
    for k in SWEEP:
        print(f"[k={k}] full calibrated pipeline")
        r = run_at(k, server, dev, cal, test)
        rows.append(r)
        print(f"  rho_gen={r['rho_gen']['point']} "
              f"ret|gen={r['rho_ret_given_gen']['point']} "
              f"set_med={r['set_size']['median']} "
              f"max_med={r['maximal_set_size']['median']}")
    # append the registered k=64 row from its own archive, never recomputed
    rep = json.load(open(os.path.join(OUT, "e2e_report_v11.json")))
    t = rep["modes"]["complete"]["test"]
    rows.append(dict(k=64, registered=True, qhat=rep["modes"]["complete"]["qhat"],
                     rho_gen=t["rho_gen"], rho_ret_given_gen=t["rho_ret_given_gen"],
                     rho_e2e=t["rho_e2e"], n_in_dsl=t["n_in_dsl"],
                     set_size=t["set_size"], maximal_set_size=t["maximal_set_size"]))
    rows.sort(key=lambda r: r["k"])
    payload = dict(
        registration="V11B endpoint 3 (see REGISTRATION_V11B.md); k=64 is "
                     "the registered primary, copied from its archive",
        rows=rows)
    json.dump(payload, open(os.path.join(OUT, "ksweep_v11.json"), "w"),
              indent=1)
    print("wrote ksweep_v11.json")


if __name__ == "__main__":
    main()
