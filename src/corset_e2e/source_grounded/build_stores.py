"""WP-E.1: split every source rule into a VISIBLE store and a HIDDEN store.

The generator is only ever given the visible store. The hidden store holds the
parsed executable target and is mounted only by the scorer-free evaluation step
that runs AFTER candidate generation has been written to disk.

Cluster = (repo, source file). Splits are assigned by HMAC of the cluster id
under a salt committed in configs/frozen_manifest.yaml before any file is read
for outcome purposes, so the assignment is a pure function of the path.

Run: PYTHONPATH=. python3 corset_e2e/source_grounded/build_stores.py
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sys
from collections import defaultdict

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
SRC = os.path.join(ROOT, "data", "rule_corpora")
OUT = os.path.join(ROOT, "results/e2e")
VIS = os.path.join(OUT, "visible_store")
HID = os.path.join(OUT, "hidden_store")

sys.path.insert(0, os.path.join(str(ROOT), "src"))
from saorl.benchmark_sg.parse import (  # noqa: E402
    dur_to_s, parse_prom_expr, _strip_template)

# Development repos: already analyzed in the prior study. DSL, ontology, and
# score design may read these. Everything else is calibration or test.
DEV_REPOS = ("awesome-prometheus-alerts", "kube-prometheus")
SALT = b"corset-e2e-split-v1"          # committed in the frozen manifest
CAL_FRACTION = 0.5                      # of non-dev clusters


def cluster_split(cluster_id: str) -> str:
    d = hmac.new(SALT, cluster_id.encode(), hashlib.sha256).digest()
    return "cal" if (int.from_bytes(d[:8], "big") / 2**64) < CAL_FRACTION else "test"


def _iter_rule_docs(path: str):
    """Yield (group_name, rule_dict) for PrometheusRule CRDs and plain rule files."""
    try:
        docs = list(yaml.safe_load_all(open(path, encoding="utf-8", errors="replace")))
    except Exception:
        return
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        # PrometheusRule CRD -> spec.groups ; plain rule file -> groups
        groups = None
        if isinstance(doc.get("spec"), dict) and "groups" in doc["spec"]:
            groups = doc["spec"]["groups"]
        elif "groups" in doc:
            groups = doc["groups"]
        if not isinstance(groups, list):
            continue
        for grp in groups:
            if not isinstance(grp, dict):
                continue
            gname = str(grp.get("name", ""))
            for r in grp.get("rules", []) or []:
                if isinstance(r, dict) and "alert" in r and "expr" in r:
                    yield gname, r


def main() -> None:
    os.makedirs(VIS, exist_ok=True)
    os.makedirs(HID, exist_ok=True)
    repos = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
    ledger, stats = [], defaultdict(int)
    seen = set()

    for repo in repos:
        rroot = os.path.join(SRC, repo)
        for dirpath, dirnames, filenames in os.walk(rroot):
            dirnames[:] = [d for d in dirnames if d not in (".git", "vendor", "node_modules")]
            for fn in filenames:
                if not fn.endswith((".yaml", ".yml")):
                    continue
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, rroot)
                cluster_id = f"{repo}:{rel}"
                split = "dev" if repo in DEV_REPOS else cluster_split(cluster_id)
                for gname, r in _iter_rule_docs(path):
                    stats["rules_seen"] += 1
                    ann = r.get("annotations", {}) or {}
                    text = _strip_template(" ".join(
                        str(ann.get(k, "")) for k in ("summary", "description", "message")))
                    if not text or len(text.split()) < 4:
                        stats["skip_no_text"] += 1
                        continue
                    raw_expr = re.sub(r"\s+", " ", str(r["expr"]).strip())
                    alert = str(r["alert"])
                    rid = hashlib.sha256(
                        f"{cluster_id}|{alert}|{raw_expr}".encode()).hexdigest()[:20]
                    if rid in seen:
                        stats["skip_duplicate"] += 1
                        continue
                    seen.add(rid)
                    parsed = parse_prom_expr(raw_expr)
                    # VISIBLE: language + allowed metadata only. No expression.
                    vis = dict(
                        rule_id=rid, cluster_id=cluster_id, repo=repo, file=rel,
                        group=gname, alert_name=alert, text=text,
                        severity=str((r.get("labels", {}) or {}).get("severity", "")),
                        has_for_field=bool(r.get("for")), split=split)
                    # HIDDEN: the executable target. Never read by the generator.
                    hid = dict(
                        rule_id=rid, raw_expr=raw_expr,
                        for_s=dur_to_s(r.get("for")) or 0.0,
                        parsed=None if parsed is None else dict(
                            metric=parsed["metric"], comparator=parsed["comparator"],
                            threshold=float(parsed["threshold"]),
                            rate_window_s=parsed["rate_window_s"],
                            aggregation=parsed["aggregation"]),
                        parser_status="OK" if parsed is not None else "PARSER_FAILURE")
                    json.dump(vis, open(os.path.join(VIS, f"{rid}.json"), "w"))
                    json.dump(hid, open(os.path.join(HID, f"{rid}.target.json"), "w"))
                    ledger.append(dict(rule_id=rid, cluster_id=cluster_id, repo=repo,
                                       split=split, parser_status=hid["parser_status"]))
                    stats[f"kept_{split}"] += 1

    clusters = defaultdict(set)
    for e in ledger:
        clusters[e["split"]].add(e["cluster_id"])
    report = dict(
        salt=SALT.decode(), dev_repos=list(DEV_REPOS), n_repos=len(repos),
        n_rules=len(ledger), stats=dict(stats),
        n_clusters={k: len(v) for k, v in clusters.items()},
        n_rules_by_split={k: sum(1 for e in ledger if e["split"] == k)
                          for k in ("dev", "cal", "test")},
        parser_ok=sum(1 for e in ledger if e["parser_status"] == "OK"),
        ledger_sha256=hashlib.sha256(
            json.dumps(sorted(e["rule_id"] for e in ledger)).encode()).hexdigest())
    json.dump(ledger, open(os.path.join(OUT, "cluster_ledger.json"), "w"), indent=1)
    json.dump(report, open(os.path.join(OUT, "store_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
