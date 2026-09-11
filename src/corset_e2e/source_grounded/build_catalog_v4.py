"""v4 catalog: the deployment's file-resident metric namespace.

Registered in results/e2e/REGISTRATION_V4.md before any outcome. Tokens
are harvested from each pinned repo's NON-RULE files (source metric
definitions, docs, dashboards; every alert/recording-rule file excluded
at harvest), so the catalog never reads the artifacts targets come from
and is target-blind by construction: metrics_for() does not depend on
the unit at all beyond its repo (v4) or its cluster (v4u union, which
adds the v3 leave-one-cluster-out expression harvest a deployment also
knows).
"""
from __future__ import annotations

import json
import os

from corset_e2e.source_grounded.build_catalog_v3 import RepoCatalogsV3
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, "results/e2e")
TOK = os.path.join(OUT, os.environ.get("CORSET_TOKENS_DIR", "catalog_v4"))


class RepoCatalogsV4(RepoCatalogsV3):
    # subclass so run_e2e's isinstance wrap applies; __init__ overridden entirely
    def __init__(self, union_with_v3: bool = False):
        v3 = json.load(open(os.path.join(OUT, "metric_catalog_v3.json")))
        self.cluster_repo = v3["cluster_repo"]
        self.union = union_with_v3
        self._v3_cm = {c: set(v) for c, v in v3["cluster_metrics"].items()} if union_with_v3 else {}
        self._by_repo_clusters: Dict[str, List[str]] = {}
        if union_with_v3:
            for c, r in self.cluster_repo.items():
                self._by_repo_clusters.setdefault(r, []).append(c)
        self.tokens: Dict[str, frozenset] = {}
        for f in os.listdir(TOK):
            if f.endswith(".tokens"):
                repo = f[:-len(".tokens")]
                self.tokens[repo] = frozenset(open(os.path.join(TOK, f)).read().split())
        self.sizes = {r: len(s) for r, s in sorted(self.tokens.items())}

    def metrics_for(self, cluster_id: str) -> List[str]:
        repo = self.cluster_repo.get(cluster_id)
        base = set(self.tokens.get(repo, frozenset()))
        if self.union and repo:
            for c in self._by_repo_clusters.get(repo, []):
                if c != cluster_id:
                    base |= self._v3_cm.get(c, set())
        return sorted(base)
