"""Slot-wise availability of the gold AST, and where it is lost.

"Proposal recall is 42%" says nothing actionable, so every slot of the
reading is measured separately and the complete-AST figure is reported as
the conjunction of them. A unit's gold AST is AVAILABLE when every part of
it can be produced by the corresponding channel:

  shape        the predicate structure is derivable within the frozen
               dev-quantile cost bound
  term shape   every term operand's shape is in the dev vocabulary
  metric       every metric name the gold uses is licensed for this unit
  threshold    every scalar is on the dev-enlarged threshold axis
  window       every range is on the dev-enlarged window grid
  labels       every on/ignoring label set is retrievable
  for          the `for` duration is on the dev-enlarged grid

Availability upper-bounds proposal recall: a ranked generator with a
finite beam may still fail to emit an available AST. Keeping the two
apart matters, because the fixes differ -- availability is a retrieval or
grammar problem, the gap between availability and emission is a ranking
problem.

Every rate is reported raw, distinct-text and repo-macro. Distinct-text is
the primary development metric: one repository contributes a 2,252-unit
template, so raw can move 23 points on a single structure and optimising
it would mean optimising template replication.

Run: PYTHONPATH=. python3 corset_e2e/analysis/proposal_recall_g2.py
Writes results/e2e/proposal_recall_g2.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.generator.pred_shape import (  # noqa: E402
    derivable, predicate_shape, terms_of)
from corset_e2e.generator.skeletons import skeletonize  # noqa: E402
from corset_e2e.generator.skeleton_grammar import _k, dev_term_vocab  # noqa: E402
from corset_e2e.dsl.schema_g2 import IN_G2, classify_target_g2, spec  # noqa: E402
from corset_e2e.generator.generate import tokens  # noqa: E402
from corset_e2e.calibration.run_e2e_v11 import V11Server  # noqa: E402
from corset_e2e.calibration.run_e2e_g1 import refs_for  # noqa: E402
from corset_e2e.generator.threshold_channel import threshold_axis  # noqa: E402
from corset_e2e.source_grounded.harvest_recording_names import (  # noqa: E402
    names_for_unit)
from corset_e2e.calibration.run_e2e_v4 import hid, load_split, vis  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")


def walk_canon(a):
    yield a
    if isinstance(a, list):
        for x in a:
            if isinstance(x, list):
                yield from walk_canon(x)


def gold_parts(ast):
    """metrics, scalars, windows, matching label sets used by a gold AST."""
    metrics, scalars, windows, labels = set(), set(), set(), set()
    for nd in walk_canon(ast):
        if not isinstance(nd, list) or not nd:
            continue
        if nd[0] == "sel" and len(nd) >= 2 and isinstance(nd[1], str):
            metrics.add(nd[1])
        elif nd[0] == "num" and len(nd) >= 2 and isinstance(nd[1], (int, float)):
            scalars.add(round(float(nd[1]), 6))
        elif nd[0] == "range" and len(nd) >= 3:
            windows.add(float(nd[2]))
        elif nd[0] == "bin" and len(nd) >= 5 and nd[4]:
            labels.add(tuple(nd[4][1]) if isinstance(nd[4][1], list) else ())
    return metrics, scalars, windows, labels


def main() -> None:
    gold = json.load(open(os.path.join(OUT, "gold_ast.json")))["gold"]
    sp = spec()
    pol = json.load(open(os.path.join(OUT, "SHAPE_POLICY.json")))
    C = pol["max_shape_cost"]
    caps = sp["capabilities"]
    voc = {_k(t) for t in dev_term_vocab(gold)}
    thr_axis, win_axis, for_axis = sp["_thresholds"], sp["_windows"], sp["_fors"]

    rec_tbl = json.load(open(os.path.join(OUT, "recording_names.json")))
    closure = set(json.load(open(os.path.join(OUT,
                                              "threshold_closure.json"))))
    thr_axis = set(thr_axis) | closure
    print(f"threshold axis: grid+dev closure = {len(thr_axis)}")
    # label-set vocabulary from dev golds only (the declared design surface)
    label_vocab = set()
    for r in gold.values():
        if r.get("split") != "dev" or "ast" not in r:
            continue
        for nd in walk_canon(r["ast"]):
            if (isinstance(nd, list) and nd and nd[0] == "bin"
                    and len(nd) >= 5 and nd[4]
                    and isinstance(nd[4][1], list)):
                label_vocab.add(tuple(nd[4][1]))
    print(f"dev label-set vocabulary: {len(label_vocab)}")
    # Repository label schemas: subsets of the repo's most frequent harvested
    # keys. The corpus uses 27 distinct sets over 31 keys, so subsets of a
    # dozen ranked keys span the space; K and the size cap are declared here,
    # not fitted. Dev's sets are unioned in for repos with no harvest.
    import itertools
    key_tbl = json.load(open(os.path.join(OUT, "label_keys.json")))
    LABEL_TOPK, LABEL_MAXSET = 12, 7
    repo_label_sets = {}
    for repo, counts in key_tbl.items():
        top = list(counts)[:LABEL_TOPK]
        acc = set(label_vocab)
        for r in range(1, LABEL_MAXSET + 1):
            for c in itertools.combinations(sorted(top), r):
                acc.add(tuple(c))
        repo_label_sets[repo] = acc
    print(f"label-set candidates per repo: "
          f"{ {k: len(v) for k, v in list(repo_label_sets.items())[:3]} } ...")

    server = V11Server()
    test = load_split("test")
    pool = [e for e in test if classify_target_g2(hid(e["rule_id"])) == IN_G2]
    print(f"pool (in G2) = {len(pool)}")

    flags = defaultdict(set)
    miss_metric = Counter()
    label_need = 0
    for e in pool:
        rid = e["rule_id"]
        rec = gold.get(rid)
        if not rec or "ast" not in rec:
            continue
        ast = rec["ast"]
        v = vis(rid)
        toks = tokens(v.get("text", "")) + tokens(v.get("alert_name", ""))
        cat = server.catalog_for(e, v)
        licensed = set(cat.license(toks, []))
        refs = set(refs_for(sorted(licensed), server.idx[e["repo"]], toks))
        declared = names_for_unit(rec_tbl, e["repo"], e["cluster_id"])
        available_names = licensed | refs | declared

        m, sc, w, lb = gold_parts(ast)
        sh = predicate_shape(ast)
        ts = []
        terms_of(ast, ts)

        if derivable(sh, caps, C):
            flags["shape"].add(rid)
        if all(_k(skeletonize(t)) in voc for t in ts):
            flags["term_shape"].add(rid)
        if m <= available_names:
            flags["metric"].add(rid)
        else:
            for x in m - available_names:
                miss_metric[x] += 1
        axis = threshold_axis(thr_axis, v)
        if all(round(x, 6) in axis for x in sc):
            flags["threshold"].add(rid)
        if all(x in win_axis for x in w):
            flags["window"].add(rid)
        if float(rec["for_s"]) in for_axis:
            flags["for"].add(rid)
        if lb:
            label_need += 1
        cand_sets = repo_label_sets.get(e["repo"], label_vocab)
        if all(tuple(sorted(t)) in cand_sets or t in cand_sets for t in lb):
            flags["labels"].add(rid)
        flags["_all"].add(rid)

    def weights(cov):
        raw = len(cov) / len(pool)
        seen, dn, dh = set(), 0, 0
        per = defaultdict(lambda: [0, 0])
        for e in pool:
            v = vis(e["rule_id"])
            k = (v.get("alert_name", "").strip(),
                 " ".join((v.get("text") or "").split()))
            if k not in seen:
                seen.add(k)
                dn += 1
                dh += e["rule_id"] in cov
            c = per[e["repo"]]
            c[0] += 1
            c[1] += e["rule_id"] in cov
        return raw, dh / dn, sum(c[1] / c[0] for c in per.values()) / len(per)

    order = ["shape", "term_shape", "metric", "threshold", "window", "for",
             "labels"]
    rep = {}
    print(f"\n{'slot':<14s} {'raw':>8s} {'distinct':>10s} {'repo-macro':>11s}")
    for k in order:
        a, b, c = weights(flags[k])
        rep[k] = dict(raw=a, distinct=b, repo_macro=c)
        print(f"{k:<14s} {a:8.1%} {b:10.1%} {c:11.1%}")

    full = set.intersection(*[flags[k] for k in order])
    a, b, c = weights(full)
    rep["complete_ast_available"] = dict(raw=a, distinct=b, repo_macro=c)
    print(f"{'COMPLETE':<14s} {a:8.1%} {b:10.1%} {c:11.1%}")
    print(f"\nunits needing on/ignoring label sets: {label_need} "
          f"({label_need / len(pool):.1%}) -- measured separately, not yet "
          f"a retrieval channel")
    print("top unlicensed gold metrics:")
    for x, cnt in miss_metric.most_common(8):
        print(f"  {cnt:5d}  {x}")

    rep["n_pool"] = len(pool)
    rep["label_sets_needed"] = label_need
    rep["top_unlicensed_metrics"] = dict(miss_metric.most_common(25))
    rep["note"] = ("availability upper-bounds proposal recall; a beam-limited "
                   "ranked generator may still fail to emit an available AST")
    json.dump(rep, open(os.path.join(OUT, "proposal_recall_g2.json"), "w"),
              indent=1)
    print("\nwrote results/e2e/proposal_recall_g2.json")


if __name__ == "__main__":
    main()
