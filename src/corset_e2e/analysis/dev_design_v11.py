"""Development-split design validation for the v11 licenser.

Everything here reads DEV units plus target-blind corpus artifacts
(committed catalogs, context profiles); cal/test units are never opened.
Serving goes through the exact production path (run_e2e_v11.V11Server:
per-cluster v4u namespaces = repo file harvest union v3
leave-one-cluster-out).  Two regimes:

  native  the unit's own namespace of record.  Dev catalogs are small,
          so this checks correctness and the verbatim channel.
  stress  the same units with the calibration repos' committed catalogs
          (and context) merged in as ~1.4e5 distractors.  Dev cannot
          natively exhibit the large-namespace regime where the frozen
          license collapsed (victoriametrics 0.050 at 17,956 names, from
          archived rows); this synthesizes it without touching any
          cal/test unit.

Endpoints per ranker variant: gen-hit@k (gold licensed within top k, or
rescued by the widened verbatim channel) and availability-conditional
recall.  --dump prints the stress cases one variant misses and another
hits, for mechanism diagnosis.  The channel weights are decided here and
frozen into license_v11.py before REGISTRATION_V11 is written.

Run: PYTHONPATH=. python3 corset_e2e/analysis/dev_design_v11.py [--dump]
Writes results/e2e/dev_design_v11.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.analysis.dev_miss_autopsy import (  # noqa: E402
    load_split, reachable, vis)
from corset_e2e.generator.generate import Catalog, tokens  # noqa: E402
from corset_e2e.generator import license_v11 as L11  # noqa: E402
from corset_e2e.generator.license_v11 import (  # noqa: E402
    CatalogV11, RepoIndexV11, wide_verbatim)
from corset_e2e.calibration.run_e2e_v11 import V11Server  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")
KS = (16, 64, 256)
CAL_REPOS = ("ceph", "rook", "thanos", "tidb")     # calibration-split repos
VARIANTS = tuple(os.environ.get(
    "V11_VARIANTS", "frozen_v4,name_only,w05,w10,w20").split(","))


def repo_tokens(repo: str) -> set:
    return set(open(os.path.join(OUT, "catalog_v4", f"{repo}.tokens"))
               .read().split())


def repo_context(repo: str) -> dict:
    p = os.path.join(OUT, "context_v11", f"{repo}.ctx.json")
    return json.load(open(p))["profiles"] if os.path.exists(p) else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()

    server = V11Server()
    dev = reachable(load_split("dev"))
    dev_repos = sorted({e["repo"] for e, _g in dev})

    distract_names: set = set()
    distract_ctx: dict = {}
    for r in CAL_REPOS:
        distract_names |= repo_tokens(r)
        distract_ctx.update(repo_context(r))

    # stress indexes: the repo's own index vocabulary union the distractors
    stress_idx = {}
    for repo in dev_repos:
        own = server.idx[repo]
        ctx = {**distract_ctx, **repo_context(repo)}
        stress_idx[repo] = RepoIndexV11(
            sorted(set(own.metrics) | distract_names), ctx)

    def licensed_for(e, v, regime, variant, k):
        cluster_allowed = set(server.v4u.metrics_for(e["cluster_id"]))
        if regime == "stress":
            cluster_allowed |= distract_names
            idx = stress_idx[e["repo"]]
        else:
            idx = server.idx[e["repo"]]
        toks = tokens(v.get("text", "")) + tokens(v.get("alert_name", ""))
        if variant == "frozen_v4":
            return Catalog(sorted(cluster_allowed)).license(toks, [])
        w = dict(name_only=0.0, w05=0.5, w10=1.0, w20=2.0)[variant]
        old = L11.W_CTX
        L11.W_CTX = w
        try:
            cat = CatalogV11.from_index(idx, cluster_allowed, "", k=k)
            return cat.license(toks, [])
        finally:
            L11.W_CTX = old

    stats = defaultdict(lambda: defaultdict(int))
    n_avail = defaultdict(int)
    dumps = []
    for e, g in dev:
        gold = g[0]
        v = vis(e["rule_id"])
        text = f"{v.get('text', '')} {v.get('alert_name', '')}"
        wv = set(wide_verbatim(text.lower()))
        for reg in ("native", "stress"):
            allowed = set(server.v4u.metrics_for(e["cluster_id"]))
            if reg == "stress":
                allowed |= distract_names
            avail = gold in allowed
            n_avail[reg] += avail
            ranked = {var: licensed_for(e, v, reg, var, max(KS))
                      for var in VARIANTS}
            for var in VARIANTS:
                for k in KS:
                    hit = gold in set(ranked[var][:k]) or gold in wv
                    stats[(reg, var, k)]["hit"] += hit
                    if avail:
                        stats[(reg, var, k)]["hit_avail"] += (
                            gold in set(ranked[var][:k]))
            if (args.dump and reg == "stress" and avail
                    and gold in set(ranked["frozen_v4"][:64])
                    and gold not in set(ranked["name_only"][:64])
                    and len(dumps) < 12):
                pos = (ranked["name_only"].index(gold) + 1
                       if gold in ranked["name_only"] else None)
                dumps.append(dict(
                    rid=e["rule_id"], gold=gold, v11_rank=pos,
                    text=v.get("text", "")[:140],
                    alert=v.get("alert_name", ""),
                    v11_top=ranked["name_only"][:6],
                    frozen_rank=ranked["frozen_v4"].index(gold) + 1))

    n = len(dev)
    rows = []
    for (reg, var, k), d in sorted(stats.items()):
        rows.append(dict(regime=reg, variant=var, k=k,
                         gen_hit=round(d["hit"] / n, 4),
                         recall_avail=round(d["hit_avail"] / max(n_avail[reg], 1), 4)))
    payload = dict(
        note=("DEV-ONLY design validation through the production serving "
              "path; cal/test units never read."),
        n_dev_reachable=n,
        n_avail=dict(n_avail),
        rows=rows)
    json.dump(payload, open(os.path.join(OUT, "dev_design_v11.json"), "w"),
              indent=1)
    for r in rows:
        if r["k"] == 64:
            print(f"{r['regime']:7s} {r['variant']:9s} k=64  "
                  f"gen_hit={r['gen_hit']:.3f}  recall|avail={r['recall_avail']:.3f}")
    for d in dumps:
        print(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
