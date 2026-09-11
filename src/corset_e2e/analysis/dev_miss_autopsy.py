"""Root-cause autopsy of the generation bottleneck, and the producing
script for results/e2e/v4_recall.json.

Two jobs, cleanly separated by split:

1. REPRODUCE (test split, archived numbers only).  v4_recall.json was
   committed without its producing script.  This script recomputes every
   field in it from the raw stores and catalogs and asserts equality,
   so the availability numbers the paper quotes (\AvailSibling,
   \AvailRepo, \AvailUnion) now have a checked provenance.  Nothing new
   is measured on test here: these are the published numbers re-derived.

2. AUTOPSY (development split only).  For every grammar-reachable dev
   unit, attribute the generator outcome to a stage:

       AVAIL_MISS   gold metric absent from the v4 repo catalog AND the
                    sibling (v3 LOCO) catalog -- retrieval failure, the
                    catalogs simply do not contain the name;
       RANK_MISS    gold metric present in the served catalog but not
                    licensed (not in the top-N by subtoken overlap, or
                    below the evidence floor, or zero overlap) --
                    selection failure inside license();
       LICENSED     gold metric licensed: in complete mode this is
                    exactly rho_gen success.

   For RANK_MISS units, record the mechanism: zero subtoken overlap
   (vocabulary mismatch), sub-floor score, or crowded out of the top-N;
   plus whether the metric is named verbatim in the visible text under a
   widened regex (colon-form recording-rule names, single-underscore
   names, alert_name included).  These dev-only statistics are the
   design evidence for the v11 licenser; nothing here touches test-split
   outcomes beyond the already-archived reproduction in job 1.

Run: PYTHONPATH=. python3 corset_e2e/analysis/dev_miss_autopsy.py
Writes results/e2e/dev_miss_autopsy.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.schema import (  # noqa: E402
    IN_DSL, classify_target, target_reading)
from corset_e2e.generator.generate import (  # noqa: E402
    Catalog, extract_slots, metric_subtokens, tokens)
from corset_e2e.source_grounded.build_catalog_v3 import RepoCatalogsV3  # noqa: E402
from corset_e2e.source_grounded.build_catalog_v4 import RepoCatalogsV4  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")
VIS = os.path.join(OUT, "visible_store")
HID = os.path.join(OUT, "hidden_store")

# widened verbatim probe: allows recording-rule colon names and >=1 underscore
_WIDE_VERBATIM = re.compile(r"\b[a-z][a-z0-9_]*(?::[a-z0-9_]+)*_[a-z0-9_]*\b"
                            r"|\b[a-z][a-z0-9_]*(?::[a-z0-9_]+)+\b")


def load_split(split: str) -> List[dict]:
    led = json.load(open(os.path.join(OUT, "cluster_ledger.json")))
    return [e for e in led if e["split"] == split]


def vis(rid: str) -> dict:
    return json.load(open(os.path.join(VIS, f"{rid}.json")))


def hid(rid: str) -> dict:
    return json.load(open(os.path.join(HID, f"{rid}.target.json")))


def reachable(entries):
    """Grammar-reachable units with their gold reading (IN_DSL by grammar)."""
    out = []
    for e in entries:
        h = hid(e["rule_id"])
        if classify_target(h) != IN_DSL:
            continue
        out.append((e, target_reading(h)))
    return out


# --------------------------------------------------------------- job 1
def reproduce_v4_recall(v3cat: RepoCatalogsV3, v4cat: RepoCatalogsV4) -> dict:
    test = reachable(load_split("test"))
    n = len(test)
    in_v3 = in_v4 = in_union = 0
    for e, g in test:
        m = g[0]
        loco = m in set(v3cat.metrics_for(e["cluster_id"]))
        repo = m in v4cat.tokens.get(e["repo"], frozenset())
        in_v3 += loco
        in_v4 += repo
        in_union += (loco or repo)
    # genmiss_recovered: units the archived v3-catalog generator run missed
    # (GEN_MISS rows) whose metric the v4 catalog contains
    rows_v3 = json.load(open(os.path.join(OUT, "e2e_test_rows_v3_complete.json")))
    miss_rids = {r["rid"] for r in rows_v3 if r["label"] == "GEN_MISS"}
    rec = 0
    for e, g in test:
        if e["rule_id"] in miss_rids and g[0] in v4cat.tokens.get(e["repo"], frozenset()):
            rec += 1
    got = dict(n_units=n, recall_v3_loco=in_v3 / n, recall_v4=in_v4 / n,
               recall_union=in_union / n, genmiss_recovered=rec)
    arch = json.load(open(os.path.join(OUT, "v4_recall.json")))
    for k, v in arch.items():
        gv = got[k]
        ok = (gv == v) if isinstance(v, int) else abs(gv - v) < 1e-9
        assert ok, f"v4_recall.json field {k}: archived {v}, re-derived {gv}"
    return got


# --------------------------------------------------------------- job 2
def autopsy_dev(v3cat: RepoCatalogsV3, v4cat: RepoCatalogsV4) -> dict:
    dev = reachable(load_split("dev"))
    rows = []
    for e, g in dev:
        v = vis(e["rule_id"])
        gold_m = g[0]
        served = v4cat.metrics_for(e["cluster_id"])          # repo tokens (v4)
        cat = Catalog(served)
        slots = extract_slots(v, cat, mode="complete")
        licensed = gold_m in set(slots["metrics"])

        avail_v4 = gold_m in set(served)
        avail_v3 = gold_m in set(v3cat.metrics_for(e["cluster_id"]))
        text, alert = v.get("text", ""), v.get("alert_name", "")
        toks = set(tokens(text) + tokens(alert))
        path_toks = set(tokens(e["cluster_id"])) | set(tokens(v.get("group", "")))
        sub = set(metric_subtokens(gold_m))
        inter = len(sub & toks)
        inter_path = len(sub & (toks | path_toks))
        wide_verbatim = gold_m in set(
            _WIDE_VERBATIM.findall(f"{text} {alert}".lower()))

        if licensed:
            bucket = "LICENSED"
            rank = None
        elif not avail_v4:
            bucket = "AVAIL_MISS" if not avail_v3 else "AVAIL_MISS_V4_ONLY"
            rank = None
        else:
            # gold is in the served catalog; find where license() ranked it
            scored = []
            for m in served:
                s = set(metric_subtokens(m))
                if not s:
                    continue
                ov = len(s & toks)
                if ov:
                    scored.append((ov / len(s), m))
            scored.sort(key=lambda x: (-x[0], x[1]))
            pos = next((i for i, (_s, m) in enumerate(scored) if m == gold_m),
                       None)
            if inter == 0:
                bucket, rank = "RANK_ZERO_OVERLAP", None
            elif pos is not None and scored[pos][0] < 0.08:
                bucket, rank = "RANK_BELOW_FLOOR", pos + 1
            else:
                bucket, rank = "RANK_CROWDED_OUT", (None if pos is None
                                                    else pos + 1)
        rows.append(dict(
            rid=e["rule_id"], repo=e["repo"], bucket=bucket, rank=rank,
            gold_metric=gold_m, n_gold_subtokens=len(sub),
            overlap_text=inter, overlap_with_path=inter_path,
            wide_verbatim=wide_verbatim, avail_v3=avail_v3, avail_v4=avail_v4,
            catalog_size=len(served)))

    buckets = Counter(r["bucket"] for r in rows)
    n = len(rows)
    # how much each dev-visible remedy could recover among the ranked misses
    ranked = [r for r in rows if r["bucket"].startswith("RANK")]
    remedy = dict(
        path_tokens_add_overlap=sum(1 for r in ranked
                                    if r["overlap_with_path"] > r["overlap_text"]),
        wide_verbatim_names_gold=sum(1 for r in ranked if r["wide_verbatim"]),
        zero_overlap=buckets.get("RANK_ZERO_OVERLAP", 0))
    return dict(n_reachable_dev=n, buckets=dict(buckets),
                rho_gen_dev=buckets.get("LICENSED", 0) / max(n, 1),
                remedies_among_rank_misses=remedy, rows=rows)


def main() -> None:
    v3cat = RepoCatalogsV3(os.path.join(OUT, "metric_catalog_v3.json"))
    v4cat = RepoCatalogsV4()
    print("[1] reproducing archived v4_recall.json on the test split")
    repro = reproduce_v4_recall(v3cat, v4cat)
    print("    all fields match:", json.dumps(repro))
    print("[2] dev-split autopsy")
    dev = autopsy_dev(v3cat, v4cat)
    payload = dict(
        note=("job 1 re-derives the archived v4_recall.json (test split, "
              "published numbers only); job 2 is DEV-ONLY design evidence "
              "for the v11 licenser"),
        v4_recall_reproduced=repro,
        dev=dev)
    with open(os.path.join(OUT, "dev_miss_autopsy.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    print(json.dumps({k: v for k, v in dev.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
