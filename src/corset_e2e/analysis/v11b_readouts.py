"""V11B endpoints 1-2: the generation funnel and the finer recall curve.

Deterministic re-derivations from committed artifacts (REGISTRATION_V11B):
the funnel chains per-unit events -- expressible -> evidence available
(v4u catalog of record) -> licensed at k=64 (archived v11 rows) ->
retained (archived rows) -- reported both conditionally and as rates of
the original corpus; recall@{1,4} recomputes the frozen licenser's
deterministic ranked lists on the same units, extending the registered
{16,64,256} curve downward. Verbatim-licensed units whose gold is absent
from the catalog are counted separately: licensing can exceed
availability through the rule's own text, and the funnel must not hide
that channel inside a conditional.

Run: PYTHONPATH=. python3 corset_e2e/analysis/v11b_readouts.py
Writes results/e2e/v11b_readouts.json
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.analysis.dev_miss_autopsy import (  # noqa: E402
    load_split, reachable, vis)
from corset_e2e.generator.generate import tokens  # noqa: E402
from corset_e2e.generator.license_v11 import wide_verbatim  # noqa: E402
from corset_e2e.calibration.run_e2e_v11 import V11Server  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")
KS_FINE = (1, 4, 16, 64, 256)


def main() -> None:
    server = V11Server()
    test = load_split("test")
    reach = reachable(test)
    rows11 = {r["rid"]: r for r in json.load(
        open(os.path.join(OUT, "e2e_test_rows_v11_complete.json")))}

    n_corpus = len(test)
    n_expr = len(reach)
    n_avail = n_lic = n_lic_avail = n_lic_verb_only = n_ret = 0
    rank_hits = {k: 0 for k in KS_FINE}
    for e, g in reach:
        gold = g[0]
        row = rows11[e["rule_id"]]
        allowed = set(server.v4u.metrics_for(e["cluster_id"]))
        avail = gold in allowed
        licensed = row["label"] == "IN_DSL"
        retained = bool(row.get("retained_gold", False))
        n_avail += avail
        n_lic += licensed
        n_lic_avail += (licensed and avail)
        n_lic_verb_only += (licensed and not avail)
        n_ret += (licensed and retained)

        v = vis(e["rule_id"])
        toks = tokens(v.get("text", "")) + tokens(v.get("alert_name", ""))
        wv = set(wide_verbatim(
            f"{v.get('text', '')} {v.get('alert_name', '')}".lower()))
        cat = server.catalog_for(e, v, k=max(KS_FINE))
        ranked = cat.license(toks, [])
        for k in KS_FINE:
            rank_hits[k] += (gold in set(ranked[:k])) or (gold in wv)

    # reconcile against the registered report before writing anything
    rep = json.load(open(os.path.join(OUT, "e2e_report_v11.json")))
    t = rep["modes"]["complete"]["test"]
    lic = rep["licensing_endpoints"]
    assert n_expr == t["n_grammar_reachable"] == 315
    assert n_lic == t["n_in_dsl"], (n_lic, t["n_in_dsl"])
    assert n_lic_avail + n_lic_verb_only == n_lic
    assert n_avail == lic["n_gold_available_v4u"], n_avail
    for k in (16, 64, 256):
        assert abs(rank_hits[k] / n_expr - lic["recall"]["v11"][str(k)]) < 5e-4, k
    assert abs(n_ret / n_corpus - t["rho_e2e"]["point"]) < 5e-4, \
        (n_ret / n_corpus, t["rho_e2e"]["point"])

    payload = dict(
        registration="V11B endpoints 1-2 (see REGISTRATION_V11B.md)",
        note=("synthesis conditional on licensing is exactly 1 in complete "
              "mode (every non-metric axis carries its full grid); the "
              "funnel therefore has no separate retrieval/synthesis rows "
              "for this instantiation, by structure rather than by merger"),
        funnel=dict(
            n_corpus=n_corpus, n_expr=n_expr, n_avail=n_avail, n_lic=n_lic,
            n_lic_avail=n_lic_avail, n_lic_verbatim_only=n_lic_verb_only,
            n_ret=n_ret,
            cond=dict(expr=n_expr / n_corpus, avail=n_avail / n_expr,
                      lic_given_avail=n_lic_avail / n_avail,
                      ret_given_lic=n_ret / n_lic),
            of_corpus=dict(expr=n_expr / n_corpus, avail=n_avail / n_corpus,
                           lic=n_lic / n_corpus, ret=n_ret / n_corpus)),
        recall_fine={str(k): rank_hits[k] / n_expr for k in KS_FINE})
    json.dump(payload, open(os.path.join(OUT, "v11b_readouts.json"), "w"),
              indent=1)
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
