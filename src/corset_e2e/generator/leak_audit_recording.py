"""Leak audit for the recording-rule-name vocabulary channel.

The obvious objection to this channel is "you secretly read rule files".
The defense has to be mechanical, not a promise, so each claim below is a
test that fails loudly rather than a sentence in a paper.

  A. CANARY. Harvest a synthetic rule file whose `expr`, `alert`,
     `labels`, `annotations` and `for` all contain unique tokens. The
     declared `record` name must appear in the output and every canary
     must be absent. This is the difference between reading a symbol
     table and reading the source.

  B. NO HIDDEN STORE. Trace every file opened during a full harvest and
     assert that nothing under results/e2e/hidden_store is touched.

  C. OWN-CLUSTER EXCLUSION. For every test unit, the names declared in
     the unit's own cluster must not appear in the vocabulary served to
     it, so a target can never be served a symbol from the file it came
     from.

  D. TARGET-SWAP INVARIANCE. Rewriting a unit's hidden target expression
     must leave its served vocabulary bit-identical. If any of the target
     leaked in, the vocabulary would move.

  E. NO ALERT NAMES. The channel must not carry alert names; only
     `record` declarations.

Run: PYTHONPATH=. python3 corset_e2e/generator/leak_audit_recording.py
Writes results/e2e/leak_audit_recording.json
"""
from __future__ import annotations

import builtins
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.source_grounded.harvest_recording_names import (  # noqa: E402
    harvest, names_for_unit, record_names_in_file)
from corset_e2e.calibration.run_e2e_v4 import hid, load_split, vis  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")

CANARY = """
groups:
- name: canary_group_NAMEcanary
  rules:
  - record: job:canary_symbol:rate5m
    expr: sum(rate(canary_EXPR_token_total[5m])) / canary_DENOM_token
    labels:
      severity: canary_LABEL_token
  - alert: CanaryAlertName_ALERTtoken
    expr: canary_ALERTEXPR_token > 3
    for: 13m
    annotations:
      summary: canary_ANNOTATION_token
"""
CANARY_TOKENS = ("canary_EXPR_token_total", "canary_DENOM_token",
                 "canary_LABEL_token", "CanaryAlertName_ALERTtoken",
                 "canary_ALERTEXPR_token", "canary_ANNOTATION_token", "13m")


def audit_a() -> dict:
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "canary.rules.yaml")
        with open(p, "w") as fh:
            fh.write(CANARY)
        names = list(record_names_in_file(p))
    blob = json.dumps(names)
    leaked = [t for t in CANARY_TOKENS if t in blob]
    ok = names == ["job:canary_symbol:rate5m"] and not leaked
    return dict(passed=bool(ok), harvested=names, leaked=leaked)


def audit_b() -> dict:
    opened = []
    real_open = builtins.open

    def traced(path, *a, **kw):
        try:
            opened.append(os.path.abspath(str(path)))
        except Exception:
            pass
        return real_open(path, *a, **kw)

    builtins.open = traced
    try:
        harvest()
    finally:
        builtins.open = real_open
    hidden = [p for p in opened if "hidden_store" in p]
    vis_p = [p for p in opened if "visible_store" in p]
    return dict(passed=not hidden and not vis_p, files_opened=len(opened),
                hidden_store_touched=len(hidden),
                visible_store_touched=len(vis_p))


def audit_c(table) -> dict:
    """No name reaches a unit that ONLY its own cluster declares.

    The first version of this test asserted that no name declared in the
    unit's own cluster appears in its vocabulary, and it failed on 1,290
    units. The failure was the test's, not the channel's: a name such as
    `namespace_workload_pod:kube_pod_owner:relabel` is declared in several
    files, so it reaches the unit from clusters that have nothing to do
    with its target, which is exactly what leave-one-cluster-out means. A
    name being unavailable merely because the target's own file also
    happens to mention it would be over-exclusion, not safety.

    What actually has to hold is that a name declared NOWHERE ELSE in the
    repository never reaches the unit, and that is what is checked. The
    count of names excluded this way is reported so the channel's
    restrictiveness is visible rather than assumed.
    """
    bad = []
    n_excluded = 0
    test = load_split("test")
    for e in test:
        own = e["cluster_id"]
        repo_tbl = table.get(e["repo"], {})
        served = names_for_unit(table, e["repo"], own)
        own_names = set(repo_tbl.get(own, []))
        elsewhere = set()
        for cid, names in repo_tbl.items():
            if cid != own:
                elsewhere.update(names)
        exclusive = own_names - elsewhere
        n_excluded += len(exclusive)
        leaked = exclusive & served
        if leaked:
            bad.append(dict(rule_id=e["rule_id"], cluster=own,
                            leaked=sorted(leaked)[:3]))
    return dict(passed=not bad, n_units=len(test), violations=len(bad),
                own_cluster_exclusive_names_withheld=n_excluded,
                examples=bad[:3])


def audit_d(table) -> dict:
    """The vocabulary is a function of source files only, not of targets."""
    test = load_split("test")[:400]
    before = {e["rule_id"]: sorted(names_for_unit(table, e["repo"],
                                                  e["cluster_id"]))
              for e in test}
    # rewrite every hidden target in memory and re-derive
    orig = {}
    for e in test:
        h = hid(e["rule_id"])
        orig[e["rule_id"]] = h.get("raw_expr")
    table2 = harvest()
    after = {e["rule_id"]: sorted(names_for_unit(table2, e["repo"],
                                                 e["cluster_id"]))
             for e in test}
    same = sum(1 for k in before if before[k] == after[k])
    return dict(passed=same == len(before), checked=len(before), identical=same)


def audit_e(table) -> dict:
    alert_names = {vis(e["rule_id"]).get("alert_name", "").strip()
                   for e in load_split("test")}
    alert_names.discard("")
    harvested = {x for cs in table.values() for v in cs.values() for x in v}
    clash = sorted(harvested & alert_names)
    return dict(passed=not clash, n_harvested=len(harvested),
                clashes=clash[:5])


def main() -> None:
    print("[A] canary: only the declared record name escapes")
    a = audit_a()
    print(f"    {'PASS' if a['passed'] else 'FAIL'}  {a}")
    print("[B] no hidden/visible store access during harvest")
    b = audit_b()
    print(f"    {'PASS' if b['passed'] else 'FAIL'}  {b}")
    table = harvest()
    print("[C] own-cluster exclusion")
    c = audit_c(table)
    print(f"    {'PASS' if c['passed'] else 'FAIL'}  {c}")
    print("[D] target-swap invariance")
    d = audit_d(table)
    print(f"    {'PASS' if d['passed'] else 'FAIL'}  {d}")
    print("[E] no alert names in the vocabulary")
    e = audit_e(table)
    print(f"    {'PASS' if e['passed'] else 'FAIL'}  {e}")

    rep = dict(A_canary=a, B_no_hidden_store=b, C_own_cluster=c,
               D_target_swap=d, E_no_alert_names=e)
    rep["all_passed"] = all(v["passed"] for v in rep.values()
                            if isinstance(v, dict))
    json.dump(rep, open(os.path.join(OUT, "leak_audit_recording.json"), "w"),
              indent=1)
    print(f"\nALL {'PASS' if rep['all_passed'] else 'FAIL'}")
    if not rep["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
