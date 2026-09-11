"""Allowed evidence: PromQL queries that are not rules, per cluster.

The independent-slot beam failed because 70% of golds name two or more
metrics and the slots were ranked independently. The proposed repair is to
retrieve a metric GROUP first and then solve a tiny assignment inside it,
which is only possible if repository evidence actually contains joint
information -- if the metrics that co-occur in a rule also co-occur
somewhere the generator is allowed to look.

This builds that evidence corpus so the question can be answered before any
ranking work is done.

WHAT IS READ. Only files the metric catalogue may already read: any path
containing "alert" or "rule" is skipped, so the targets' artifact family is
never opened, and markup extensions are skipped. Within those files,
PromQL is taken from `expr` fields (dashboards and configs), from fenced
code blocks, and from lines that syntactically parse as PromQL. Each
surviving query contributes its distinct metric set H_e together with the
cluster it came from, so evidence can be served leave-one-cluster-out.

WHAT IS NOT READ. Recording-rule right-hand sides, alert expressions, the
hidden store, and the visible store. Recording-rule NAMES come from the
separate declared-symbol channel, not from here.

Run: PYTHONPATH=. python3 corset_e2e/source_grounded/evidence_corpus.py
Writes results/e2e/evidence_corpus.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
SRC = os.path.join(ROOT, "data", "rule_corpora")
OUT = os.path.join(ROOT, "results/e2e")

sys.path.insert(0, os.path.join(str(ROOT), "src"))
from corset_e2e.dsl.promql_ast import try_parse, walk, VectorSelector  # noqa: E402
from corset_e2e.source_grounded.harvest_label_keys import SKIP_EXT  # noqa: E402
from corset_e2e.source_grounded.harvest_tokens import is_excluded  # noqa: E402

_EXPR_JSON = re.compile(
    r'"(?:expr|query|expression)"\s*:\s*"((?:[^"\\]|\\.){4,2000})"')
# jsonnet text blocks: `query: ||| ... |||`, the dominant PromQL carrier in
# gitlab-runbooks, which is written in jsonnet rather than YAML. The first
# extractor saw none of these.
_JSONNET_BLOCK = re.compile(
    r"\b(?:query|expr|expression)\s*[:=]\s*\|\|\|(.{4,4000}?)\|\|\|", re.S)
_JSONNET_STR = re.compile(
    r"\b(?:query|expr|expression)\s*[:=]\s*'((?:[^'\\]|\\.){4,2000})'")
# jsonnet/python-style format placeholders. They are not PromQL, so a query
# containing them cannot parse until they are given syntactically valid
# stand-ins; the substitutions below are structural, never semantic.
_PH_RANGE = re.compile(r"\[\s*%\(\w+\)s\s*\]")
_PH_SELECTOR = re.compile(r"\{\s*%\(\w+\)s\s*\}")
_PH_GROUPING = re.compile(r"(\b(?:by|without|on|ignoring)\s*\(\s*)%\(\w+\)s(\s*\))")
_PH_ANY = re.compile(r"%\(\w+\)s")


def fill_placeholders(q: str) -> str:
    q = _PH_RANGE.sub("[5m]", q)
    q = _PH_SELECTOR.sub("{}", q)
    q = _PH_GROUPING.sub(r"\1env\2", q)
    q = _PH_ANY.sub("phv", q)
    return q
_EXPR_YAML = re.compile(r"^\s*expr:\s*[|>]?\s*(.{4,2000})$", re.M)
_FENCE = re.compile(r"```(?:promql|prometheus)?\s*\n(.{4,2000}?)\n```", re.S)
_PROMQL_HINT = re.compile(
    r"\b(rate|irate|increase|histogram_quantile|sum|avg|max|min|count|"
    r"absent|delta|deriv|predict_linear)\s*\(")
MAX_FILE_BYTES = 4_000_000
MAX_PER_FILE = 400


def _unescape(s: str) -> str:
    return s.replace('\\"', '"').replace("\\n", " ").replace("\\\\", "\\")


def candidate_queries(text: str, is_json: bool):
    seen = set()
    for m in _EXPR_JSON.finditer(text):
        q = _unescape(m.group(1))
        if q not in seen:
            seen.add(q)
            yield q
    for rx in (_JSONNET_BLOCK, _JSONNET_STR):
        for m in rx.finditer(text):
            q = " ".join(m.group(1).split())
            if q and q not in seen:
                seen.add(q)
                yield q
    if not is_json:
        for m in _EXPR_YAML.finditer(text):
            q = m.group(1).strip().strip("'\"")
            if q and q not in seen:
                seen.add(q)
                yield q
        for m in _FENCE.finditer(text):
            q = " ".join(m.group(1).split())
            if q not in seen and _PROMQL_HINT.search(q):
                seen.add(q)
                yield q


def metrics_of(query: str):
    query = fill_placeholders(query)
    node, _err = try_parse(query)
    if node is None:
        return None
    out = set()
    for nd in walk(node):
        if isinstance(nd, VectorSelector) and nd.metric:
            out.add(nd.metric)
    return out or None


def harvest() -> dict:
    """repo -> list of {cluster, metrics}."""
    out = defaultdict(list)
    repos = sorted(d for d in os.listdir(SRC)
                   if os.path.isdir(os.path.join(SRC, d)))
    for repo in repos:
        rroot = os.path.join(SRC, repo)
        for dirpath, dirnames, filenames in os.walk(rroot):
            dirnames[:] = [d for d in dirnames
                           if d not in (".git", "vendor", "node_modules")]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, rroot)
                low = rel.lower()
                if is_excluded(low) or low.endswith(SKIP_EXT):
                    continue
                try:
                    if os.path.getsize(full) > MAX_FILE_BYTES:
                        continue
                    with open(full, encoding="utf8", errors="ignore") as fh:
                        text = fh.read()
                except OSError:
                    continue
                cid = f"{repo}:{rel}"
                n = 0
                for q in candidate_queries(text, low.endswith(".json")):
                    ms = metrics_of(q)
                    if not ms:
                        continue
                    out[repo].append(dict(cluster=cid, metrics=sorted(ms)))
                    n += 1
                    if n >= MAX_PER_FILE:
                        break
    return dict(out)


def main() -> None:
    tbl = harvest()
    json.dump(tbl, open(os.path.join(OUT, "evidence_corpus.json"), "w"))
    tot = sum(len(v) for v in tbl.values())
    print(f"allowed PromQL evidence units: {tot}")
    for repo, v in sorted(tbl.items()):
        multi = sum(1 for e in v if len(e["metrics"]) > 1)
        names = len({m for e in v for m in e["metrics"]})
        print(f"  {repo:<30s} {len(v):6d} queries, {multi:6d} multi-metric, "
              f"{names:5d} distinct names")
    print("\nwrote results/e2e/evidence_corpus.json")


if __name__ == "__main__":
    main()
