"""v3 public metric schema: the deployment's full metric namespace.

v2 harvested each repository's namespace from its PARSEABLE alert rules only.
Because roughly 80% of real rules use constructs the frozen grammar cannot
express, those namespaces came out impoverished (12-133 names per repo), and
generation recall stalled at 0.438 with every remaining failure on the metric
axis.

A real deployment does not know only the metrics named in its simple alerts.
It can list its whole namespace from the monitoring system
(`/api/v1/label/__name__/values`), and that list is public operational
metadata under the plan's allowed-metadata contract. v3 models that: metric
identifiers are lifted from EVERY expression in the repository, whether or not
the frozen grammar can express the rule, which is strictly more faithful to
what an operator has on hand.

Target-independence is preserved exactly as in v2: a unit's catalog is the
union over the repository's OTHER source-file clusters, so the unit's own
cluster contributes nothing and swapping its hidden target cannot change its
pool. The leak audit is re-run against this catalog.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
OUT = os.path.join(ROOT, "results/e2e")
HID = os.path.join(OUT, "hidden_store")

# PromQL metric identifiers: snake_case names not immediately followed by "("
# (which would make them function calls) and not pure keywords.
_METRIC_RE = re.compile(r"\b([a-z_][a-z0-9_]*(?::[a-z0-9_]+)*)\b(?!\s*\()")
_KEYWORDS = {
    "by", "without", "on", "ignoring", "group_left", "group_right", "offset",
    "bool", "and", "or", "unless", "if", "for", "default", "start", "end",
    "inf", "nan", "le", "job", "instance", "namespace", "pod", "container",
    "cluster", "service", "endpoint", "severity", "alertname", "quantile",
}


def harvest(out_path: str) -> dict:
    led = json.load(open(os.path.join(OUT, "cluster_ledger.json")))
    per_cluster: Dict[str, set] = defaultdict(set)
    repo_of: Dict[str, str] = {}
    for e in led:
        repo_of[e["cluster_id"]] = e["repo"]
        h = json.load(open(os.path.join(HID, f"{e['rule_id']}.target.json")))
        expr = h.get("raw_expr") or ""
        for m in _METRIC_RE.finditer(expr):
            name = m.group(1)
            if name in _KEYWORDS or len(name) < 4 or "_" not in name:
                continue
            per_cluster[e["cluster_id"]].add(name)

    by_repo: Dict[str, set] = defaultdict(set)
    for cid, names in per_cluster.items():
        by_repo[repo_of[cid]] |= names
    payload = dict(
        scheme="leave-one-cluster-out repository namespace, all expressions",
        n_repos=len(by_repo),
        repo_sizes={r: len(v) for r, v in sorted(by_repo.items())},
        repo_metrics={r: sorted(v) for r, v in by_repo.items()},
        cluster_metrics={c: sorted(v) for c, v in per_cluster.items()},
        cluster_repo=repo_of)
    json.dump(payload, open(out_path, "w"))
    return payload


class RepoCatalogsV3:
    """Serves the leave-one-cluster-out namespace for each deployment unit."""

    def __init__(self, path: str):
        d = json.load(open(path))
        self.cluster_metrics = {c: set(v) for c, v in d["cluster_metrics"].items()}
        self.cluster_repo = d["cluster_repo"]
        self.sizes = d["repo_sizes"]
        self._by_repo: Dict[str, List[str]] = {}

    def metrics_for(self, cluster_id: str) -> List[str]:
        repo = self.cluster_repo.get(cluster_id)
        if repo is None:
            return []
        if repo not in self._by_repo:
            self._by_repo[repo] = [c for c, r in self.cluster_repo.items()
                                   if r == repo]
        out: set = set()
        for c in self._by_repo[repo]:
            if c != cluster_id:
                out |= self.cluster_metrics.get(c, set())
        return sorted(out)


if __name__ == "__main__":
    rep = harvest(os.path.join(OUT, "metric_catalog_v3.json"))
    print(json.dumps({"scheme": rep["scheme"], "n_repos": rep["n_repos"],
                      "repo_sizes": rep["repo_sizes"]}, indent=1))
