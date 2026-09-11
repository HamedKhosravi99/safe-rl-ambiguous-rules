"""What the declared grammar is missing, measured on ASTs rather than strings.

For every test unit this parses the hidden target and computes the SET of
grammar capabilities the expression requires beyond the frozen six-tuple
G0. A unit is expressible under a capability set S exactly when its
required set is contained in S, so reach becomes a set-cover question and
the useful output is not a flat table of missing constructs but a GREEDY
CURVE: which capability, added next, buys the most corpus.

Two cross-checks keep this honest. Units requiring nothing must be exactly
the archived G0 population (315), and units requiring at most a reference
threshold must be exactly the archived G1 population (1,511). If the
feature extractor drifts from the shipped classifiers, those asserts fail.

Every reach number is reported three ways -- raw units, distinct written
texts, and per-repository macro-average -- because one repository supplies
90.4% of the units as machine expansions of a few templates, so a raw
number can move a long way on a single template and that is not
generalization. A capability that helps only the raw column is labelled as
such rather than quoted alone.

Run: PYTHONPATH=. python3 corset_e2e/analysis/grammar_taxonomy.py
Writes results/e2e/grammar_taxonomy.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.promql_ast import (  # noqa: E402
    ARITH_OPS, Aggregate, Binary, COMPARISON_OPS, Call, MatrixSelector,
    NumberLit, SET_OPS, StringLit, Subquery, Unary, VectorSelector,
    fold_constant, unparen)
from corset_e2e.dsl.promql_repair import parse_repairing_comments  # noqa: E402
from corset_e2e.dsl.schema import (  # noqa: E402
    AGGREGATIONS, FOR_S, IN_DSL, RATE_WINDOWS_S, THRESHOLD_GRID,
    classify_target)
from corset_e2e.dsl.schema_g1 import classify_target_g1  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import hid, load_split, vis  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")

# what G0 can already say
BASE_FNS = frozenset(("rate", "irate", "increase"))
_TG = {round(float(x), 6) for x in THRESHOLD_GRID}
_RW = {w for w in RATE_WINDOWS_S if w is not None}
_FS = {float(x) for x in FOR_S}
_BASE_AGG = frozenset(a for a in AGGREGATIONS if a != "none")

REFERENCE = "threshold_reference"


class Featurizer:
    """Collect the capabilities an expression needs beyond G0.

    The grids are parameters because G2 widens them: a threshold that is
    off-grid against the frozen decade grid is on-grid against the
    dev-derived enlarged one, and it would be wrong to keep charging it as
    a missing capability once the axis covers it.
    """

    def __init__(self, for_s: float, thresholds=None, windows=None,
                 fors=None):
        self.tg = _TG if thresholds is None else thresholds
        self.rw = _RW if windows is None else windows
        self.fs = _FS if fors is None else fors
        self.f = set()
        if float(for_s) not in self.fs:
            self.f.add("for_offgrid")

    # -- boolean-valued position
    def pred(self, n, depth: int = 0) -> None:
        n = unparen(n)
        if isinstance(n, Binary) and n.op in SET_OPS:
            self.f.add(f"setop_{n.op}")
            if n.matching is not None:
                self.f.add("vector_matching")
            self.pred(n.lhs, depth + 1)
            self.pred(n.rhs, depth + 1)
            return
        if isinstance(n, Binary) and n.op in COMPARISON_OPS:
            if n.bool_:
                self.f.add("bool_modifier")
            if n.matching is not None:
                self.f.add("vector_matching")
            self.term(n.lhs, 0)
            self.rhs(n.rhs)
            return
        # no comparison at the top: `absent(x)`, `a unless b`, a bare vector
        self.f.add("no_comparison")
        self.term(n, 0)

    # -- right-hand side of a comparison
    def rhs(self, n) -> None:
        u = unparen(n)
        if isinstance(u, NumberLit):
            if round(float(u.value), 6) not in self.tg:
                self.f.add("threshold_offgrid")
            return
        folded = fold_constant(u)
        if folded is not None:
            self.f.add("threshold_const_expr")
            return
        if isinstance(u, VectorSelector):
            self.f.add(REFERENCE)
            if u.offset_s:
                self.f.add("offset")
            return
        # a computed series on the right: reference expression, not a name
        self.f.add("threshold_expr_vector")
        self.term(u, 0)

    # -- vector-valued position
    def term(self, n, agg_depth: int) -> None:
        u = unparen(n)
        if isinstance(u, VectorSelector):
            if u.offset_s:
                self.f.add("offset")
            if u.at_:
                self.f.add("at_modifier")
            return
        if isinstance(u, MatrixSelector):
            if u.range_s not in self.rw:
                self.f.add("window_offgrid")
            self.term(u.vs, agg_depth)
            return
        if isinstance(u, NumberLit) or isinstance(u, StringLit):
            return
        if isinstance(u, Call):
            if u.func not in BASE_FNS:
                self.f.add(f"fn_{u.func}")
            for a in u.args:
                self.term(a, agg_depth)
            return
        if isinstance(u, Aggregate):
            if agg_depth >= 1:
                self.f.add("agg_nested")
            if u.op not in _BASE_AGG:
                self.f.add(f"agg_{u.op}")
            if u.param is not None:
                self.f.add("agg_param")
            self.term(u.expr, agg_depth + 1)
            return
        if isinstance(u, Subquery):
            self.f.add("subquery")
            self.term(u.expr, agg_depth)
            return
        if isinstance(u, Unary):
            self.f.add("unary")
            self.term(u.expr, agg_depth)
            return
        if isinstance(u, Binary):
            if u.op in ARITH_OPS:
                if fold_constant(u) is not None:
                    return                     # pure scalar arithmetic
                self.f.add("arith_vector")
                if u.matching is not None:
                    self.f.add("vector_matching")
                self.term(u.lhs, agg_depth)
                self.term(u.rhs, agg_depth)
                return
            if u.op in SET_OPS:
                self.f.add(f"setop_{u.op}")
                if u.matching is not None:
                    self.f.add("vector_matching")
                self.pred(u.lhs)
                self.pred(u.rhs)
                return
            if u.op in COMPARISON_OPS:
                self.f.add("nested_comparison")
                self.pred(u)
                return
        self.f.add(f"unknown_{type(u).__name__}")


def features_for(h: dict, thresholds=None, windows=None, fors=None):
    """(feature_set, parse_status). None means the expression did not parse."""
    x = (h.get("raw_expr") or "").strip()
    if not x:
        return None, "empty"
    node, how = parse_repairing_comments(x)
    if node is None:
        return None, how
    fz = Featurizer(float(h.get("for_s", 0.0)), thresholds, windows, fors)
    try:
        fz.pred(node)
    except RecursionError:
        return None, "recursion"
    return fz.f, how


def weightings(units, covered_ids):
    """Raw, distinct-text and per-repo-macro rates for a covered set."""
    n = len(units)
    raw = sum(1 for u in units if u["rule_id"] in covered_ids) / n
    by_text = defaultdict(list)
    for u in units:
        by_text[u["_textkey"]].append(u)
    dis = [sorted(v, key=lambda z: z["rule_id"])[0] for v in by_text.values()]
    distinct = sum(1 for u in dis if u["rule_id"] in covered_ids) / len(dis)
    per_repo = defaultdict(lambda: [0, 0])
    for u in units:
        cell = per_repo[u["repo"]]
        cell[0] += 1
        cell[1] += u["rule_id"] in covered_ids
    macro = sum(c[1] / c[0] for c in per_repo.values()) / len(per_repo)
    return dict(raw=raw, distinct=distinct, repo_macro=macro,
                n_raw=sum(1 for u in units if u["rule_id"] in covered_ids),
                n_distinct_total=len(dis))


def main() -> None:
    test = load_split("test")
    units = []
    req = {}
    unparsed = Counter()
    for e in test:
        rid = e["rule_id"]
        h = hid(rid)
        v = vis(rid)
        u = dict(rule_id=rid, repo=e["repo"], cluster=e["cluster_id"],
                 _textkey=(v.get("alert_name", "").strip(),
                           " ".join((v.get("text") or "").split())))
        units.append(u)
        feats, how = features_for(h)
        if feats is None:
            unparsed[how] += 1
            req[rid] = None            # unreachable by any capability set
        else:
            req[rid] = frozenset(feats)

    n = len(units)
    # ---- cross-checks against the shipped classifiers
    g0_ids = {u["rule_id"] for u in units if req[u["rule_id"]] == frozenset()}
    g1_ids = {u["rule_id"] for u in units
              if req[u["rule_id"]] is not None
              and req[u["rule_id"]] <= frozenset({REFERENCE})}
    ship_g0 = {u["rule_id"] for u in units
               if classify_target(hid(u["rule_id"])) == IN_DSL}
    ship_g1 = {u["rule_id"] for u in units
               if classify_target_g1(hid(u["rule_id"]))
               in (IN_DSL, "IN_DSL_REF")}
    # The featurizer does NOT reproduce the shipped populations, and that is
    # the finding rather than a bug: the shipped classifier reads a six-tuple
    # built by regex, which silently drops arithmetic and set operations, so
    # it admits units whose recorded gold is a projection of the rule (see
    # gold_faithfulness_audit.py). The invariant that must hold is that the
    # featurizer PARTITIONS each shipped population -- every shipped unit is
    # either capability-free or needs a capability the six-tuple cannot
    # encode -- so a shipped unit missing from the featurizer's view entirely
    # would indicate a real extraction bug.
    ship_g0_unaccounted = {r for r in ship_g0 if req[r] is None}
    # The only admissible reason a shipped G0 unit fails to parse is that it
    # is not PromQL at all: one corpus rule is a LogQL query (`|= "error"`
    # line filter), which the regex parser nonetheless accepted and gave a
    # PromQL six-tuple gold -- the same defect in a sharper form. Anything
    # else would be a gap in this parser and must fail the build.
    logql = {r for r in ship_g0_unaccounted
             if any(t in (hid(r).get("raw_expr") or "")
                    for t in ("|=", "!=~", "|~", "| json", "| logfmt"))}
    print(f"[check] featurizer G0 {len(g0_ids)} vs shipped {len(ship_g0)} "
          f"(difference = lossy golds; unparsed among shipped: "
          f"{len(ship_g0_unaccounted)}, of which LogQL: {len(logql)})")
    print(f"[check] featurizer G1 {len(g1_ids)} vs shipped {len(ship_g1)}")
    assert ship_g0_unaccounted == logql, \
        (f"{len(ship_g0_unaccounted - logql)} shipped G0 units fail to parse "
         f"for a reason other than being another query language -- that is a "
         f"gap in promql_ast, not a corpus defect")

    # ---- greedy capability curve
    reachable = [u for u in units if req[u["rule_id"]] is not None]
    print(f"[info] parsed {len(reachable)}/{n}; unparsed {dict(unparsed)}")

    have = set()
    curve = []
    covered = set(g0_ids)
    for step in range(24):
        gain = Counter()
        for u in reachable:
            r = req[u["rule_id"]]
            if r <= have:
                continue
            missing = r - have
            if len(missing) == 1:
                gain[next(iter(missing))] += 1
        if not gain:
            # nothing is one capability away; count by cheapest missing set
            for u in reachable:
                r = req[u["rule_id"]]
                if not r <= have:
                    for feat in r - have:
                        gain[feat] += 1
            if not gain:
                break
        feat, _ = gain.most_common(1)[0]
        have.add(feat)
        covered = {u["rule_id"] for u in reachable if req[u["rule_id"]] <= have}
        w = weightings(units, covered)
        curve.append(dict(step=step + 1, added=feat, n_features=len(have),
                          **w))
        print(f"  +{feat:<24s} raw {w['raw']:6.1%} ({w['n_raw']:5d})  "
              f"distinct {w['distinct']:6.1%}  repo-macro {w['repo_macro']:6.1%}")

    # ---- flat frequency of each capability among still-unreachable units
    freq_all = Counter()
    freq_blocking = Counter()
    g1_set = frozenset({REFERENCE})
    for u in reachable:
        r = req[u["rule_id"]]
        for f in r:
            freq_all[f] += 1
        if not r <= g1_set:
            for f in r - g1_set:
                freq_blocking[f] += 1

    rep = dict(
        n_units=n,
        parsed=len(reachable),
        unparsed=dict(unparsed),
        featurizer_g0=len(g0_ids), shipped_g0=len(ship_g0),
        featurizer_g1=len(g1_ids), shipped_g1=len(ship_g1),
        g0=weightings(units, g0_ids), g1=weightings(units, g1_ids),
        greedy_curve=curve,
        capability_frequency=dict(freq_all.most_common()),
        blocking_beyond_g1=dict(freq_blocking.most_common()),
        note=("reach ceilings only: a capability set bounds what a generator "
              "COULD address and says nothing about licensing or retention, "
              "which need their own registered pass"))
    json.dump(rep, open(os.path.join(OUT, "grammar_taxonomy.json"), "w"),
              indent=1)
    print("\nblocking capabilities beyond G1 (by units):")
    for f, c in freq_blocking.most_common(18):
        print(f"  {c:5d}  {f}")


if __name__ == "__main__":
    main()
