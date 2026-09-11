"""REGISTRATION_V21: recalibrate Keep on the faithful in-grammar population.

Re-runs the frozen v11 evaluation (same server, weights, mode) on the
grammar-reachable calibration units, recomputes the recorded threshold
(asserted equal to the archived 0.41), then the threshold from the FAITHFUL
in-grammar calibration units only, and evaluates both on the faithful test
golds of the archived frozen test pass.  No test unit is re-licensed.

Run: PYTHONPATH=. python3 corset_e2e/analysis/faithful_recalibration.py
Writes results/e2e/faithful_recalibration.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.analysis.gold_faithfulness_audit import STRUCTURAL  # noqa: E402,F401
from corset_e2e.analysis.grammar_taxonomy import features_for  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import hid, load_split  # noqa: E402
from corset_e2e.calibration.run_e2e_v11 import V11Server, evaluate  # noqa: E402
from corset_e2e.dsl.schema import IN_DSL, classify_target  # noqa: E402

R = os.path.join(ROOT, "results/e2e")
OUT = os.path.join(R, "faithful_recalibration.json")
DELTA = 0.10


def faithful_g0(h) -> bool:
    feats, _ = features_for(h)
    return feats is not None and not feats


def qhat(scores, delta=DELTA):
    s = sorted(scores)
    k = max(1, math.floor(delta * (len(s) + 1)))
    return s[k - 1], k


def main():
    t0 = time.perf_counter()
    rep = json.load(open(os.path.join(R, "e2e_report_v11.json")))
    weights = rep["weights"]
    q_arch = rep["modes"]["complete"]["qhat"]
    cal = [e for e in load_split("cal") if classify_target(hid(e["rule_id"])) == IN_DSL]
    print(f"grammar-reachable cal units: {len(cal)} (archive: {rep['modes']['complete']['calibration']['n_grammar_reachable']})", flush=True)
    server = V11Server()
    rows = evaluate(cal, server, weights, tag="cal", mode="complete")["rows"]
    indsl = [r for r in rows if r["label"] == IN_DSL and r["gold_score"] is not None]
    q_rec, k_rec = qhat([r["gold_score"] for r in indsl])
    assert abs(q_rec - q_arch) < 1e-9, (q_rec, q_arch)
    assert len(indsl) == rep["modes"]["complete"]["n_cal_in_dsl"], len(indsl)
    faith_cal = {r["rid"] for r in indsl if faithful_g0(hid(r["rid"]))}
    q_f, k_f = qhat([r["gold_score"] for r in indsl if r["rid"] in faith_cal])
    # frozen test pass, faithful units
    test = json.load(open(os.path.join(R, "e2e_test_rows_v11_complete.json")))
    n_test = len(test)
    tin = [r for r in test if r["label"] == IN_DSL and r["gold_score"] is not None]
    faith_test = {r["rid"] for r in tin if faithful_g0(hid(r["rid"]))}
    tf = [r for r in tin if r["rid"] in faith_test]
    ret_rec = sum(1 for r in tf if r["gold_score"] >= q_rec - 1e-12)
    ret_f = sum(1 for r in tf if r["gold_score"] >= q_f - 1e-12)
    ret_rec_all = sum(1 for r in tin if r["gold_score"] >= q_rec - 1e-12)
    assert ret_rec_all == 158 and sum(1 for r in tin) == 181
    res = dict(registration="REGISTRATION_V21.md", delta_sem=DELTA,
               calibration=dict(n_grammar_reachable=len(cal), n_in_dsl=len(indsl), n_in_dsl_faithful=len(faith_cal),
                                qhat_recorded=q_rec, k_recorded=k_rec, qhat_faithful=q_f, k_faithful=k_f,
                                qhat_archived=q_arch),
               test=dict(n_units=n_test, n_in_dsl=len(tin), n_in_dsl_faithful=len(tf),
                         retained_faithful_under_recorded=ret_rec, retained_faithful_under_faithful=ret_f,
                         retention_faithful_under_recorded=ret_rec / len(tf), retention_faithful_under_faithful=ret_f / len(tf),
                         e2e_faithful_under_recorded=ret_rec / n_test, e2e_faithful_under_faithful=ret_f / n_test,
                         retained_all_under_recorded=ret_rec_all),
               seconds=round(time.perf_counter() - t0, 1))
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "seconds"}, indent=1))


if __name__ == "__main__":
    main()
