"""Gate: is the gold metric SET inside the leak-safe vocabulary, on dev?

Every later stage -- cardinality prediction, unary narrowing, pairwise
compatibility, set search, slot assignment -- presupposes that the gold
metrics are available at all. Metric availability was 94.7% on TEST; it has
never been measured on DEV, and dev's two repositories carry far less
declared-symbol vocabulary than the deployments do (kube-prometheus
declares 65 recording-rule names, awesome-prometheus-alerts none). So the
number that licenses the next build is this one, not the test one.

    C_V = Pr[ M*_l  subset of  V_l ]

with V_l = v11-licensed names, union reference candidates, union
recording-rule names declared elsewhere in the repository.

Run: PYTHONPATH=. python3 corset_e2e/analysis/vocab_ceiling_dev.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.schema_g2 import IN_G2, classify_target_g2  # noqa: E402
from corset_e2e.generator.generate import tokens  # noqa: E402
from corset_e2e.calibration.run_e2e_v11 import V11Server  # noqa: E402
from corset_e2e.calibration.run_e2e_g1 import refs_for  # noqa: E402
from corset_e2e.source_grounded.harvest_recording_names import names_for_unit  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import hid, load_split, vis  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")


def walk(a):
    yield a
    if isinstance(a, list):
        for x in a:
            if isinstance(x, list):
                yield from walk(x)


def main(split="dev"):
    gold = json.load(open(os.path.join(OUT, "gold_ast.json")))["gold"]
    rec = json.load(open(os.path.join(OUT, "recording_names.json")))
    # V = repository-safe vocabulary UNION frozen external standard vocabulary.
    # The second is one global set, identical for every target, built from
    # pinned upstream commits (results/e2e/standard_vocab.json).
    STD = set(json.load(open(os.path.join(OUT,
                                          "standard_vocab.json")))["names"])
    server = V11Server()
    units = [e for e in load_split(split)
             if classify_target_g2(hid(e["rule_id"])) == IN_G2]
    tot = 0
    hit = collections.Counter()
    by_q = collections.Counter()
    miss = collections.Counter()
    per_repo = collections.defaultdict(lambda: [0, 0])
    seen_txt, dn, dh = set(), 0, 0
    for e in units:
        r = gold.get(e["rule_id"])
        if not r or "ast" not in r:
            continue
        M = {nd[1] for nd in walk(r["ast"])
             if isinstance(nd, list) and nd and nd[0] == "sel"
             and isinstance(nd[1], str)}
        if not M:
            continue
        tot += 1
        v = vis(e["rule_id"])
        toks = tokens(v.get("text", "")) + tokens(v.get("alert_name", ""))
        lic = set(server.catalog_for(e, v).license(toks, []))
        refs = set(refs_for(sorted(lic), server.idx[e["repo"]], toks))
        dec = names_for_unit(rec, e["repo"], e["cluster_id"])
        V = lic | refs | dec | STD
        ok = M <= V
        hit["all"] += ok
        hit["any"] += bool(M & V)
        by_q[len(M)] += 1
        if ok:
            hit[f"q{len(M)}"] += 1
        else:
            for m in M - V:
                miss[m] += 1
        c = per_repo[e["repo"]]
        c[0] += 1
        c[1] += ok
        key = (v.get("alert_name", "").strip(),
               " ".join((v.get("text") or "").split()))
        if key not in seen_txt:
            seen_txt.add(key)
            dn += 1
            dh += ok
    macro = sum(c[1] / c[0] for c in per_repo.values()) / len(per_repo)
    print(f"[{split}] in-G2 units with >=1 metric: {tot}")
    print(f"  C_V (whole gold set in vocabulary)  raw {hit['all'] / tot:.1%}"
          f"   distinct {dh / dn:.1%}   repo-macro {macro:.1%}")
    print(f"  at least one gold metric available  {hit['any'] / tot:.1%}")
    print("  by group size:")
    for q in sorted(by_q):
        print(f"    q={q}: {hit[f'q{q}']}/{by_q[q]} = "
              f"{hit[f'q{q}'] / by_q[q]:.1%}")
    print(f"  per repo: { {k: f'{c[1]}/{c[0]}' for k, c in per_repo.items()} }")
    print("  top unavailable gold metrics:")
    for m, c in miss.most_common(10):
        print(f"    {c:5d}  {m}")
    json.dump(dict(split=split, n=tot, C_V_raw=hit["all"] / tot,
                   C_V_distinct=dh / dn, C_V_repo_macro=macro,
                   any=hit["any"] / tot,
                   by_size={str(q): hit[f"q{q}"] / by_q[q] for q in by_q},
                   top_missing=dict(miss.most_common(30))),
              open(os.path.join(OUT, f"vocab_ceiling_{split}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
