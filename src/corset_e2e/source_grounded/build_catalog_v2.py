"""v2 public metric schema: the deployment's own metric namespace.

v1 harvested the metric catalog from the two development repositories and
applied it everywhere. The frozen v1 test showed that this is the binding
constraint: 100% of generation failures were metric-not-licensed, and rho_gen
tracked namespace overlap with the dev repos (0.881 on the Kubernetes-family
repository, 0.05-0.10 on projects with their own namespaces). Only 17.8% of
test target metrics existed in the dev catalog.

v2 models what a real deployment actually has. An organisation running CORSET
knows its own metric namespace -- it is listable from the monitoring system
(`/api/v1/label/__name__/values`) and is public deployment metadata under the
plan's A.3 allowlist. Withholding it was an artificial handicap.

Target-independence is preserved by construction:
  * the catalog for a unit is built from OTHER source-file clusters in the
    same repository, never from the unit's own cluster, so the unit's target
    file contributes nothing;
  * a namespace of thousands of metric names does not reveal which metric any
    particular rule uses -- that is exactly what rho_gen still has to measure.

The leak audit is re-run against this catalog and must still pass.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PKG))
OUT = os.path.join(ROOT, "results/e2e")
HID = os.path.join(OUT, "hidden_store")


def build_repo_catalogs(out_path: str) -> dict:
    """metric names per (repo, cluster) with the cluster's own file excluded."""
    led = json.load(open(os.path.join(OUT, "cluster_ledger.json")))
    per_cluster: Dict[str, set] = defaultdict(set)
    repo_of: Dict[str, str] = {}
    for e in led:
        h = json.load(open(os.path.join(HID, f"{e['rule_id']}.target.json")))
        repo_of[e["cluster_id"]] = e["repo"]
        if h.get("parser_status") == "OK" and h.get("parsed"):
            per_cluster[e["cluster_id"]].add(h["parsed"]["metric"])

    by_repo: Dict[str, set] = defaultdict(set)
    for cid, ms in per_cluster.items():
        by_repo[repo_of[cid]] |= ms

    payload = dict(
        scheme="leave-one-cluster-out repository metric namespace",
        n_repos=len(by_repo),
        repo_sizes={r: len(v) for r, v in sorted(by_repo.items())},
        repo_metrics={r: sorted(v) for r, v in by_repo.items()},
        cluster_metrics={c: sorted(v) for c, v in per_cluster.items()},
        cluster_repo=repo_of)
    json.dump(payload, open(out_path, "w"))
    return payload


class RepoCatalogs:
    """Serves a leave-one-cluster-out catalog for each deployment unit."""

    def __init__(self, path: str):
        d = json.load(open(path))
        self.repo_metrics = {r: set(v) for r, v in d["repo_metrics"].items()}
        self.cluster_metrics = {c: set(v) for c, v in d["cluster_metrics"].items()}
        self.cluster_repo = d["cluster_repo"]
        self.sizes = d["repo_sizes"]
        self._by_repo_clusters: Dict[str, List[str]] = {}

    def metrics_for(self, cluster_id: str) -> List[str]:
        """Union of the metric names in the repository's OTHER clusters.

        Written as a union over other clusters rather than
        `repo_metrics - own_cluster_metrics`. Those two differ, and the
        difference is a genuine leak channel: subtracting the unit's own
        metrics makes the EXCLUSION set a function of the hidden target, so
        swapping the target changes the catalog. The v2 leak audit caught
        exactly that. A union over other clusters never reads the unit's own
        cluster at all, so it is invariant to the unit's target by
        construction.
        """
        repo = self.cluster_repo.get(cluster_id)
        if repo is None:
            return []
        if repo not in self._by_repo_clusters:
            self._by_repo_clusters[repo] = [
                c for c, r in self.cluster_repo.items() if r == repo]
        out: set = set()
        for c in self._by_repo_clusters[repo]:
            if c != cluster_id:
                out |= self.cluster_metrics.get(c, set())
        return sorted(out)


if __name__ == "__main__":
    p = os.path.join(OUT, "metric_catalog_v2.json")
    rep = build_repo_catalogs(p)
    print(json.dumps({"scheme": rep["scheme"], "n_repos": rep["n_repos"],
                      "repo_sizes": rep["repo_sizes"]}, indent=1))
