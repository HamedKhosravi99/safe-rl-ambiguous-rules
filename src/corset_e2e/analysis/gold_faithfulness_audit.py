"""Is the archived gold reading a faithful reading of the rule?

`saorl/benchmark_sg/parse.py:parse_prom_expr` builds the hidden target by
regex: it locates a comparison, takes the FIRST metric name appearing in
the left-hand side, and records an outermost aggregation and window. Its
docstring says it rejects joins, topk, group_left and quantiles. It does
not reject binary arithmetic between two vector operands, so

    (sum(rate(A{result="error"}[5m])) / sum(rate(A[5m]))) > 0.01

is recorded as the six-tuple `sum(rate(A[5m])) > 0.01`: an error RATE
rather than an error FRACTION, a different predicate with a different
cost. Likewise `avail_bytes / size_bytes * 100 < 5` is recorded as
`avail_bytes < 5` -- five bytes free rather than five percent free.

This matters beyond bookkeeping. Coverage is the fraction of units whose
gold reading the pipeline recovers. Where the gold is a projection rather
than the rule, recovering it is not evidence about the rule, so those
units must be reported separately.

The audit re-parses each target to an AST and asks which grammar
capabilities the real expression needs beyond what its recorded six-tuple
can encode. STRUCTURAL capabilities -- arithmetic between series, set
operations, a comparison nested under an aggregation, a computed
right-hand side -- change the predicate, so a unit needing one has a lossy
gold. Capabilities that merely widen an axis (an off-grid window, an extra
function name) are reported separately as REPRESENTATIONAL, since whether
they change the predicate is a judgement call and the reader should see
both counts.

Run: PYTHONPATH=. python3 corset_e2e/analysis/gold_faithfulness_audit.py
Writes results/e2e/gold_faithfulness.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.analysis.grammar_taxonomy import REFERENCE, features_for  # noqa: E402
from corset_e2e.dsl.schema import IN_DSL, classify_target  # noqa: E402
from corset_e2e.dsl.schema_g1 import IN_G1_REF, classify_target_g1  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import hid, load_split  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")

# capabilities whose absence changes the PREDICATE, not just its spelling
STRUCTURAL = frozenset((
    "arith_vector", "setop_and", "setop_or", "setop_unless",
    "nested_comparison", "no_comparison", "threshold_expr_vector",
    "agg_nested", "agg_param", "subquery", "unary"))

# the reference class legitimately encodes these two
G1_ENCODED = frozenset((REFERENCE, "vector_matching"))


def audit_class(units, label_of, encoded):
    """Split a labelled population into faithful / structural / representational."""
    out = dict(n=0, faithful=0, structural=0, representational=0)
    reasons = Counter()
    ids = dict(structural=set(), representational=set(), faithful=set())
    for u in units:
        rid = u["rule_id"]
        h = hid(rid)
        if label_of(h) is False:
            continue
        out["n"] += 1
        feats, _how = features_for(h)
        if feats is None:
            out["structural"] += 1
            reasons["<unparsed>"] += 1
            ids["structural"].add(rid)
            continue
        extra = feats - encoded
        if not extra:
            out["faithful"] += 1
            ids["faithful"].add(rid)
        elif extra & STRUCTURAL:
            out["structural"] += 1
            ids["structural"].add(rid)
            for f in sorted(extra & STRUCTURAL):
                reasons[f] += 1
        else:
            out["representational"] += 1
            ids["representational"].add(rid)
            for f in sorted(extra):
                reasons[f] += 1
    out["reasons"] = dict(reasons.most_common())
    return out, ids


def main() -> None:
    test = load_split("test")
    n = len(test)

    g0, g0_ids = audit_class(
        test, lambda h: classify_target(h) == IN_DSL, frozenset())
    ref, ref_ids = audit_class(
        test, lambda h: classify_target_g1(h) == IN_G1_REF, G1_ENCODED)

    print(f"[G0 frozen grammar]  n={g0['n']}")
    print(f"   faithful          {g0['faithful']:5d} "
          f"({g0['faithful'] / max(g0['n'], 1):.1%})")
    print(f"   lossy structural  {g0['structural']:5d} "
          f"({g0['structural'] / max(g0['n'], 1):.1%})")
    print(f"   representational  {g0['representational']:5d}")
    for f, c in list(g0["reasons"].items())[:8]:
        print(f"       {c:5d}  {f}")
    print(f"\n[G1 reference class] n={ref['n']}")
    print(f"   faithful          {ref['faithful']:5d} "
          f"({ref['faithful'] / max(ref['n'], 1):.1%})")
    print(f"   lossy structural  {ref['structural']:5d} "
          f"({ref['structural'] / max(ref['n'], 1):.1%})")
    print(f"   representational  {ref['representational']:5d}")
    for f, c in list(ref["reasons"].items())[:8]:
        print(f"       {c:5d}  {f}")

    # ---- what this does to the two published coverage headlines
    covered = {}
    for tag, path, labels in (
            ("v11_published", "e2e_test_rows_v11_complete.json", (IN_DSL,)),
            ("g1_v15", "e2e_test_rows_g1.json", (IN_DSL, IN_G1_REF))):
        p = os.path.join(OUT, path)
        if not os.path.exists(p):
            continue
        rows = json.load(open(p))
        cov = {r["rid"] for r in rows
               if r.get("label") in labels and r.get("retained_gold")}
        covered[tag] = cov

    faithful_all = g0_ids["faithful"] | ref_ids["faithful"]
    structural_all = g0_ids["structural"] | ref_ids["structural"]
    headline = {}
    for tag, cov in covered.items():
        f = len(cov & faithful_all)
        s = len(cov & structural_all)
        headline[tag] = dict(
            covered=len(cov), raw_rate=len(cov) / n,
            faithful=f, faithful_rate=f / n,
            lossy_structural=s,
            other=len(cov) - f - s)
        print(f"\n[{tag}] covered {len(cov)} = {len(cov) / n:.1%} of {n}")
        print(f"   of which gold is faithful : {f} = {f / n:.1%}")
        print(f"   of which gold is lossy    : {s}")
        print(f"   other (representational)  : {len(cov) - f - s}")

    # The Mondrian pass reuses the frozen V15 test scores and changes only
    # the per-class threshold, so its covered set is recomputed here rather
    # than read from a rows file it never wrote.
    mp = os.path.join(OUT, "e2e_report_g1_mondrian.json")
    rp = os.path.join(OUT, "e2e_test_rows_g1.json")
    if os.path.exists(mp) and os.path.exists(rp):
        qh = {k: v["qhat"]
              for k, v in json.load(open(mp))["per_class"].items()}
        rows = json.load(open(rp))
        cov = {r["rid"] for r in rows
               if r.get("label") in qh and r.get("gold_score") is not None
               and r["gold_score"] >= qh[r["label"]]}
        f = len(cov & faithful_all)
        s = len(cov & structural_all)
        headline["g1_mondrian"] = dict(
            covered=len(cov), raw_rate=len(cov) / n,
            faithful=f, faithful_rate=f / n,
            lossy_structural=s, other=len(cov) - f - s)
        print(f"\n[g1_mondrian] covered {len(cov)} = {len(cov) / n:.1%} of {n}")
        print(f"   of which gold is faithful : {f} = {f / n:.1%}")
        print(f"   of which gold is lossy    : {s}")
        print(f"   other (representational)  : {len(cov) - f - s}")

    rep = dict(n_units=n, g0=g0, reference_class=ref, headline=headline,
               structural_capabilities=sorted(STRUCTURAL),
               root_cause=("saorl/benchmark_sg/parse.py:parse_prom_expr takes "
                           "the first LHS metric by regex and does not reject "
                           "binary arithmetic between vector operands"))
    json.dump(rep, open(os.path.join(OUT, "gold_faithfulness.json"), "w"),
              indent=1)

    assert g0["faithful"] + g0["structural"] + g0["representational"] \
        == g0["n"], "G0 partition does not sum"
    assert ref["faithful"] + ref["structural"] + ref["representational"] \
        == ref["n"], "reference partition does not sum"
    print("\nwrote results/e2e/gold_faithfulness.json")


if __name__ == "__main__":
    main()
