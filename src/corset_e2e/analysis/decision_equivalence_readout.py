"""REGISTRATION V42 (REGISTRATION_V28.md): decision-equivalence recall on the
archived v11 test pass.  Reads the frozen test rows only; no re-licensing.
For the 173 faithful G0 targets, tabulate the labels of the units whose gold
reading was not proposed.  In complete mode every non-metric axis carries its
full frozen grid, so a miss is a metric-licensing miss; no equivalence that
preserves the observed series identifies two metric names, hence
decision-equivalence recall on the deployed controller equals exact recall.

Run: PYTHONPATH=. python3 corset_e2e/analysis/decision_equivalence_readout.py
Writes results/e2e/decision_equivalence_readout.json
"""
from __future__ import annotations
import os
import collections, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, os.path.join(str(ROOT), "src"))
from corset_e2e.analysis.faithful_recalibration import faithful_g0  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import hid  # noqa: E402
from corset_e2e.dsl.schema import IN_DSL  # noqa: E402
R = os.path.join(ROOT, "results/e2e")
rows = json.load(open(os.path.join(R, "e2e_test_rows_v11_complete.json")))
expr = [r for r in rows if r["label"] in (IN_DSL, "GEN_MISS")]
faith = [r for r in expr if faithful_g0(hid(r["rid"]))]
hit = [r for r in faith if r["label"] == IN_DSL and r["gold_score"] is not None]
miss = [r for r in faith if r not in hit]
labels = collections.Counter(r["label"] for r in miss)
assert len(expr) == 315 and len(faith) == 173 and len(hit) == 111, (len(expr), len(faith), len(hit))
n_metrics = collections.Counter(r["n_metrics"] for r in miss)
out = dict(registration="REGISTRATION_V28.md, V42", n_expressible=len(expr), n_faithful=len(faith), n_hit=len(hit), n_miss=len(miss),
           miss_labels=dict(labels), miss_n_metrics=dict(n_metrics),
           exact_recall=len(hit) / len(faith), decision_equivalence_recall=len(hit) / len(faith),
           reason="complete mode: every non-metric axis carries its full grid, so GEN_MISS = gold metric not among the licensed metrics; "
                  "no series-preserving equivalence identifies two metric names, so nothing is recoverable up to decision equivalence")
json.dump(out, open(os.path.join(R, "decision_equivalence_readout.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
