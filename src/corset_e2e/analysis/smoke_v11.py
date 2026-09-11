"""Smoke tests for the v11 licenser: hand-derivable synthetic cases.

Run BEFORE any corpus evaluation, per the campaign rule that code must be
shown aligned with the method before touching data.  Every case is small
enough to verify by hand from the license_v11 definitions.

Run: PYTHONPATH=. python3 corset_e2e/analysis/smoke_v11.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.generator.generate import Catalog, extract_slots, tokens  # noqa: E402
from corset_e2e.generator.license_v11 import (  # noqa: E402
    CatalogV11, canon, wide_verbatim)


def case_norm():
    assert canon("available") == "avail" == canon("avail")
    assert canon("errors") == "err" == canon("error")
    assert canon("bytes") == "byte"
    assert canon("process") == "process"          # 'ss' guard: no plural strip
    assert canon("latencies") == "latency"
    print("  [ok] normalization table + plural strip")


def case_verbatim():
    got = wide_verbatim("the rule slo:max:hard_limit fired on node_cpu"
                        " and apiserver_request_total{code} rose")
    assert "slo:max:hard_limit" in got
    assert "node_cpu" in got
    assert "apiserver_request_total" in got
    assert "group_left" not in wide_verbatim("x group_left y")
    print("  [ok] widened verbatim: colon names, single underscore, junk out")


def case_norm_connects():
    # 'available' must reach node_filesystem_avail_bytes via avail~available.
    # The frozen license sees only the shared 'node' (raw strings), giving
    # the distractor node_cpu_bytes a higher fraction (1/3 vs 1/4); v11's
    # normalization + IDF puts the gold first.
    metrics = ["node_cpu_bytes", "node_filesystem_avail_bytes",
               "kube_pod_status_ready"]
    text = "disk space available is running low on the node"
    toks = tokens(text)
    old_top = Catalog(metrics).license(toks, [])
    new_top = CatalogV11(metrics, context={}, visible_text=text).license(toks, [])
    assert old_top[0] == "node_cpu_bytes", old_top
    assert new_top[0] == "node_filesystem_avail_bytes", new_top
    print("  [ok] normalization connects avail~available; frozen license blind")


def case_idf_beats_common():
    # both metrics share the common token 'node'; only the rare token
    # 'entropy' identifies the gold.  With equal weighting both tie on
    # fraction 1/2; IDF must rank the entropy metric first.
    metrics = ["node_entropy_bits", "node_cpu_seconds",
               "container_cpu_seconds", "container_cpu_usage",
               "pod_cpu_seconds"]
    text = "low entropy on the node"
    v11 = CatalogV11(metrics, context={}, visible_text=text)
    top = v11.license(tokens(text), [])
    assert top[0] == "node_entropy_bits", top
    print("  [ok] IDF ranks the rare identifying token above common ones")


def case_context_channel():
    # name overlap is zero for the gold under BOTH rankers; only the doc
    # context ('help: number of tables pending compaction') connects it.
    metrics = ["abc_pending_ops", "xyz_queue_len"]
    ctx = {"abc_pending_ops": {"compaction": 4, "table": 6, "pending": 3},
           "xyz_queue_len": {"network": 5, "socket": 2}}
    text = "too many tables await compaction"
    v11 = CatalogV11(metrics, context=ctx, visible_text=text)
    top = v11.license(tokens(text), [])
    assert top and top[0] == "abc_pending_ops", top
    print("  [ok] context channel licenses a lexically-unrelated metric")


def case_deterministic_and_dropin():
    metrics = [f"m{i}_alpha_beta" for i in range(50)] + ["node_load_one"]
    text = "load is high on one node"
    visible = dict(text=text, alert_name="NodeLoadHigh", has_for_field=True)
    v11 = CatalogV11(metrics, context={}, visible_text=text)
    a = v11.license(tokens(text), [])
    b = v11.license(tokens(text), [])
    assert a == b
    slots = extract_slots(visible, v11, mode="complete")   # frozen generate.py
    assert "node_load_one" in slots["metrics"]
    assert len(slots["thresholds"]) > 100                  # full frozen grid
    print("  [ok] deterministic; drop-in through frozen extract_slots")


def case_no_free_licensing():
    # a text with no evidence for any metric licenses nothing (floor)
    v11 = CatalogV11(["aa_bb_cc", "dd_ee_ff"], context={},
                     visible_text="completely unrelated words here")
    assert v11.license(tokens("completely unrelated words here"), []) == []
    print("  [ok] no evidence-free licensing")


def main() -> None:
    for fn in (case_norm, case_verbatim, case_norm_connects,
               case_idf_beats_common, case_context_channel,
               case_deterministic_and_dropin, case_no_free_licensing):
        fn()
    print("SMOKE PASS: v11 licenser aligned with its definition")


if __name__ == "__main__":
    main()
