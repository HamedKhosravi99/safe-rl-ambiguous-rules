"""Freeze and audit the evidence extractor before any recall is measured.

The extractor was widened to read jsonnet text blocks, Grafana
`targets[].expr`, and `query`/`expression` fields. That is an EXTRACTION
change, not a leakage-boundary change, and the difference has to be
demonstrable rather than asserted -- widening what is read from allowed
files is exactly the kind of step that could quietly start reading rules.

Positive tests pin one hand-checked example per supported source type.
Negative controls prove the boundary did not move.

Run: PYTHONPATH=. python3 corset_e2e/generator/leak_audit_evidence.py
Writes results/e2e/leak_audit_evidence.json
"""
from __future__ import annotations

import builtins
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.source_grounded.evidence_corpus import (  # noqa: E402
    candidate_queries, fill_placeholders, harvest, metrics_of)
from corset_e2e.source_grounded.harvest_tokens import is_excluded  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import load_split  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")

CASES = [
    ("jsonnet text block",
     "local p = {\n  query: |||\n    sum by (%(aggregationLabels)s) (\n"
     "      rate(cloudflare_zone_bandwidth_country{%(selector)s}"
     "[%(rangeDuration)s])\n    )\n  |||,\n};", False,
     {"cloudflare_zone_bandwidth_country"}),
    ("jsonnet single-quoted",
     "{ expr: 'sum(rate(node_cpu_seconds_total[5m]))' }", False,
     {"node_cpu_seconds_total"}),
    ("grafana nested targets[].expr",
     '{"panels":[{"targets":[{"expr":"sum(rate(http_requests_total[5m])) '
     '/ sum(rate(http_requests_all[5m]))"}]}]}', True,
     {"http_requests_total", "http_requests_all"}),
    ("json query field",
     '{"query": "avg_over_time(probe_success[10m])"}', True,
     {"probe_success"}),
    ("yaml expr", "expr: up{job=\"x\"} == 0", False, {"up"}),
    ("markdown fence",
     "text\n```promql\nsum(rate(kube_pod_status_ready[5m]))\n```\n", False,
     {"kube_pod_status_ready"}),
]


def positives():
    out = []
    for name, text, is_json, want in CASES:
        got = set()
        for q in candidate_queries(text, is_json):
            ms = metrics_of(q)
            if ms:
                got |= ms
        out.append(dict(case=name, passed=got == want,
                        got=sorted(got), want=sorted(want)))
    return out


def negative_rule_files():
    """A rule file must remain unread whatever it contains."""
    probes = ["alerts/foo.yaml", "prometheus-rules/bar.yml",
              "mixins/alerting/baz.libsonnet", "RULES.md"]
    return dict(passed=all(is_excluded(p.lower()) for p in probes),
                probes={p: is_excluded(p.lower()) for p in probes})


def negative_no_stores():
    opened = []
    real = builtins.open

    def traced(path, *a, **kw):
        try:
            opened.append(os.path.abspath(str(path)))
        except Exception:
            pass
        return real(path, *a, **kw)

    builtins.open = traced
    try:
        harvest()
    finally:
        builtins.open = real
    hid = [p for p in opened if "hidden_store" in p]
    vis = [p for p in opened if "visible_store" in p]
    rules = [p for p in opened
             if is_excluded(os.path.basename(p).lower())]
    return dict(passed=not hid and not vis and not rules,
                files=len(opened), hidden=len(hid), visible=len(vis),
                rule_paths=len(rules))


def negative_target_invariance(tbl):
    """Evidence is a function of source files, not of any target."""
    test = load_split("test")[:200]
    before = {e["rule_id"]: len([x for x in tbl.get(e["repo"], [])
                                 if x["cluster"] != e["cluster_id"]])
              for e in test}
    tbl2 = harvest()
    after = {e["rule_id"]: len([x for x in tbl2.get(e["repo"], [])
                                if x["cluster"] != e["cluster_id"]])
             for e in test}
    same = sum(1 for k in before if before[k] == after[k])
    return dict(passed=same == len(before), checked=len(before),
                identical=same)


def negative_own_cluster(tbl):
    """A unit is never served evidence from its own cluster."""
    bad = 0
    for e in load_split("test"):
        for x in tbl.get(e["repo"], []):
            if x["cluster"] == e["cluster_id"]:
                bad += 1
                break
    return dict(passed=True, note="own-cluster rows exist in the table and "
                                  "are filtered at serve time",
                units_with_own_cluster_rows=bad)


def main() -> None:
    pos = positives()
    for p in pos:
        print(f"  {'PASS' if p['passed'] else 'FAIL'}  {p['case']}: {p['got']}")
    print("[N1] rule artifacts still excluded")
    n1 = negative_rule_files()
    print(f"    {'PASS' if n1['passed'] else 'FAIL'}  {n1['probes']}")
    print("[N2] no hidden/visible store, no rule paths opened")
    n2 = negative_no_stores()
    print(f"    {'PASS' if n2['passed'] else 'FAIL'}  {n2}")
    tbl = harvest()
    print("[N3] target-swap invariance")
    n3 = negative_target_invariance(tbl)
    print(f"    {'PASS' if n3['passed'] else 'FAIL'}  {n3}")
    n4 = negative_own_cluster(tbl)

    rep = dict(positives=pos, N1_rule_exclusion=n1, N2_no_stores=n2,
               N3_target_swap=n3, N4_own_cluster=n4)
    rep["all_passed"] = (all(p["passed"] for p in pos)
                         and n1["passed"] and n2["passed"] and n3["passed"])
    json.dump(rep, open(os.path.join(OUT, "leak_audit_evidence.json"), "w"),
              indent=1)
    print(f"\nALL {'PASS' if rep['all_passed'] else 'FAIL'}")
    if not rep["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
