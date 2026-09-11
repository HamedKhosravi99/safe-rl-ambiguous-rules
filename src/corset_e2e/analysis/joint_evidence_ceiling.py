"""Does allowed repository evidence contain JOINT metric information?

The independent-slot beam failed because 70% of golds name two or more
metrics and each slot was ranked on its own. The proposed repair -- retrieve
a metric GROUP, then solve a small assignment inside it -- is only viable if
the metrics a rule uses together also appear together somewhere the
generator may look. This is the gate for that repair, measured on dev
before any ranking machinery is built.

Three ceilings, all leave-one-cluster-out over the allowed evidence corpus
(PromQL in non-rule files; never a rule expression, never the hidden or
visible store):

  names      every metric of the gold appears somewhere in the evidence
  C_exact    some single query contains the whole gold group
  C_pair     every pair of gold metrics co-occurs in some query
  C_connect  the gold group induces a connected co-occurrence graph

The conditional rate C_connect | names separates two very different
diagnoses: evidence that is structurally uninformative, versus evidence
that is merely too small.

Run: PYTHONPATH=. python3 corset_e2e/analysis/joint_evidence_ceiling.py
Writes results/e2e/joint_evidence_ceiling.json
"""
from __future__ import annotations

import collections
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.schema_g2 import IN_G2, classify_target_g2  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import hid, load_split  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")


def walk(a):
    yield a
    if isinstance(a, list):
        for x in a:
            if isinstance(x, list):
                yield from walk(x)


def gold_metrics(ast):
    return {nd[1] for nd in walk(ast)
            if isinstance(nd, list) and nd and nd[0] == "sel"
            and isinstance(nd[1], str)}


def main(split="dev"):
    gold = json.load(open(os.path.join(OUT, "gold_ast.json")))["gold"]
    ev = json.load(open(os.path.join(OUT, "evidence_corpus.json")))
    units = [e for e in load_split(split)
             if classify_target_g2(hid(e["rule_id"])) == IN_G2]
    res = collections.Counter()
    per_q = collections.Counter()
    tot = 0
    conn_given_names = [0, 0]
    for e in units:
        r = gold.get(e["rule_id"])
        if not r or "ast" not in r:
            continue
        M = gold_metrics(r["ast"])
        if not M:
            continue
        tot += 1
        per_q[len(M)] += 1
        H = [set(x["metrics"]) for x in ev.get(e["repo"], [])
             if x["cluster"] != e["cluster_id"]]
        names = set().union(*H) if H else set()
        has_names = M <= names
        res["names"] += has_names
        res["exact"] += any(M <= h for h in H)
        co = {p for h in H for p in itertools.combinations(sorted(h), 2)}
        if len(M) == 1:
            res["pair"] += has_names
            res["conn"] += has_names
        else:
            res["pair"] += all(p in co for p in
                               itertools.combinations(sorted(M), 2))
            adj = collections.defaultdict(set)
            for a, b in co:
                if a in M and b in M:
                    adj[a].add(b)
                    adj[b].add(a)
            ok = False
            if has_names:
                s0 = next(iter(M))
                seen, st = {s0}, [s0]
                while st:
                    u = st.pop()
                    for v in adj[u]:
                        if v not in seen:
                            seen.add(v)
                            st.append(v)
                ok = seen >= M
            res["conn"] += ok
        if has_names:
            conn_given_names[1] += 1
            conn_given_names[0] += (len(M) == 1) or ok
    rep = dict(split=split, n=tot, metrics_per_rule=dict(sorted(per_q.items())),
               names=res["names"] / tot, C_exact=res["exact"] / tot,
               C_pair=res["pair"] / tot, C_connected=res["conn"] / tot,
               conn_given_names=(conn_given_names[0] /
                                 max(conn_given_names[1], 1)),
               n_names=conn_given_names[1],
               evidence_queries={k: len(v) for k, v in ev.items()})
    print(json.dumps({k: v for k, v in rep.items()
                      if k != "evidence_queries"}, indent=1))
    json.dump(rep, open(os.path.join(OUT, "joint_evidence_ceiling.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
