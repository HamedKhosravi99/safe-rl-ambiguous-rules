"""Verifier for the v6 semantic-licensing archive.

The v6 LLM licenser ran out-of-tree (claude CLI 2.1.221; the call itself
is not deterministically replayable), but its complete raw I/O was
archived: results/e2e/v6_selections/out_<repo>.txt holds one
`rid|metric1,...,metric16` line per unit and gold.json the 315 rid->gold
map.  This script recomputes every number in v6_semantic_licensing.json
from that archive and asserts equality, so the 0.600 endpoint the paper
cites is checked data, not an unverifiable commit.

Run: PYTHONPATH=. python3 corset_e2e/analysis/verify_v6_archive.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

OUT = os.path.join(ROOT, "results/e2e")
SEL = os.path.join(OUT, "v6_selections")


def main() -> None:
    gold = json.load(open(os.path.join(SEL, "gold.json")))
    sel = {}
    per_repo_of = {}
    for f in sorted(os.listdir(SEL)):
        if not (f.startswith("out_") and f.endswith(".txt")):
            continue
        repo = f[len("out_"):-len(".txt")]
        for line in open(os.path.join(SEL, f)):
            line = line.strip()
            if not line or "|" not in line:
                continue
            rid, ms = line.split("|", 1)
            sel[rid] = [m for m in ms.split(",") if m]
            per_repo_of[rid] = repo
            assert len(sel[rid]) <= 16, (rid, len(sel[rid]))

    assert set(sel) == set(gold), "selections and gold cover different units"
    hits = 0
    per_repo = defaultdict(lambda: dict(hits=0, n=0))
    for rid, g in gold.items():
        r = per_repo_of[rid]
        per_repo[r]["n"] += 1
        if g in set(sel[rid]):
            hits += 1
            per_repo[r]["hits"] += 1

    arch = json.load(open(os.path.join(OUT, "v6_semantic_licensing.json")))
    assert arch["n"] == len(gold) == 315, arch["n"]
    assert arch["hits"] == hits, (arch["hits"], hits)
    assert abs(arch["rate"] - hits / len(gold)) < 1e-9
    for r, d in arch["per_repo"].items():
        assert per_repo[r]["hits"] == d["hits"], (r, per_repo[r], d)
        assert per_repo[r]["n"] == d["n"], (r, per_repo[r], d)
    print(f"v6 archive verified: {hits}/{len(gold)} = {hits/len(gold):.3f}; "
          "every per-repo count matches v6_semantic_licensing.json")


if __name__ == "__main__":
    main()
