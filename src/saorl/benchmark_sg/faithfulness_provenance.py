"""Which headline numbers rest on lossy six-tuple readings, and what
survives on the faithful subset.

The corset_e2e audit showed that parse_prom_expr projects structure away:
a ratio is recorded as its numerator, a conjunction as one conjunct. The
MAIN study's Prometheus targets come from the same parser
(parse_prometheus), so every downstream number whose population includes
a structurally lossy reading needs either recomputation on faithful
readings or an explicit restriction label. This audit does the first
where the per-rule archives permit filtering, and marks the rest.

Verdict per target, from re-parsing raw_expr with the AST parser:
  FAITHFUL   the expression needs nothing beyond what a six-tuple can
             carry; off-grid thresholds/windows/for are allowed, since
             parse_prom_expr records the actual values and the grids were
             an e2e-generation concern, not a fidelity one
  LOSSY      the expression needs structure a six-tuple cannot encode
             (vector arithmetic, set operations, extra functions, a
             computed or reference right-hand side, offset, nesting)
  UNPARSED   the AST parser cannot read it (counted, never guessed)

Kyverno targets are parsed from structured policy fields by a separate
code path that never touches parse_prom_expr; they are labelled
unaffected rather than audited here.

Run: PYTHONPATH=. python3 saorl/benchmark_sg/faithfulness_provenance.py
Writes results/e2e/downstream_provenance.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, "results/e2e")
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.analysis.grammar_taxonomy import features_for  # noqa: E402
from saorl.benchmark_sg.parse import parse_prometheus  # noqa: E402

# extras a six-tuple can still carry faithfully: the parser records the
# actual numbers, so grid membership is not a fidelity question
GRID_ONLY = frozenset(("threshold_offgrid", "window_offgrid", "for_offgrid"))


def verdict_of(raw_expr: str, for_s: float):
    feats, how = features_for(dict(raw_expr=raw_expr, for_s=for_s,
                                   parser_status="X"))
    if feats is None:
        return "UNPARSED", ("<%s>" % how,)
    extra = frozenset(feats) - GRID_ONLY
    if not extra:
        return "FAITHFUL", ()
    return "LOSSY", tuple(sorted(extra))


def main() -> None:
    pt, _ = parse_prometheus()
    verdicts = {}
    reasons = Counter()
    counts = Counter()
    for ti, t in enumerate(pt):
        uid = f"{t.name}#{ti}"
        v, why = verdict_of(t.raw_expr, t.reading.for_s)
        verdicts[uid] = dict(verdict=v, why=list(why))
        counts[v] += 1
        for w in why:
            reasons[w] += 1
    n = len(pt)
    print(f"parse_prometheus targets: {n}")
    for k, c in counts.most_common():
        print(f"  {k:<9s} {c:4d}  ({c / n:.1%})")
    print("  top lossy causes:", dict(reasons.most_common(6)))
    faithful = {u for u, d in verdicts.items() if d["verdict"] == "FAITHFUL"}

    rep = dict(n_targets=n, counts=dict(counts),
               lossy_causes=dict(reasons.most_common()),
               verdicts=verdicts, restatements={})

    # ---- monitoring compiled screen (FiresMon x / CompiledAnalyzed) ------
    pcb = json.load(open(os.path.join(OUT, "policy_class_budget.json")))
    op = pcb["operating"]
    # decided = compiled + provably_clear_nested (the archive's own
    # aggregate: n_decided = 60 of n_total 88, n_unanalyzed 28)
    comp = [r for r in pcb["rows"]
            if r["family"] == "prometheus" and abs(r["budget"] - op) < 1e-9
            and r["policy_class"] == "monitoring_compiled"
            and r["status"] in ("compiled", "provably_clear_nested")]
    cf = [r for r in comp if r["rule_id"] in faithful]
    fires_all = sum(1 for r in comp if r["screen_fires"])
    fires_f = sum(1 for r in cf if r["screen_fires"])
    rep["restatements"]["monitoring_screen"] = dict(
        all=dict(n=len(comp), fires=fires_all),
        faithful=dict(n=len(cf), fires=fires_f))
    print(f"\nmonitoring screen @ d={op}: {fires_all}/{len(comp)} all "
          f"-> {fires_f}/{len(cf)} on faithful subset")

    # ---- class ladder (free rung + operating fires), prometheus ---------
    cl = json.load(open(os.path.join(OUT, "class_ladder.json")))
    oper = cl["operating"]
    free = [r for r in cl["rows"]
            if r["family"] == "prometheus" and r["k"] == "free"
            and abs(r["budget"] - oper) < 1e-9]
    ff = [r for r in free if r["rule_id"] in faithful]
    rep["restatements"]["ladder_free_rung"] = dict(
        all=dict(n=len(free), fires=sum(1 for r in free
                                        if r["screen_fires"])),
        faithful=dict(n=len(ff), fires=sum(1 for r in ff
                                           if r["screen_fires"])))
    print(f"ladder free rung @ d={oper}: "
          f"{rep['restatements']['ladder_free_rung']['all']} -> "
          f"{rep['restatements']['ladder_free_rung']['faithful']}")

    # which-pair-decides (k=2 decides for the rule at operating budget)
    by_rule = defaultdict(dict)
    for r in cl["rows"]:
        if (r["family"] == "prometheus" and isinstance(r["k"], int)
                and abs(r["budget"] - oper) < 1e-9):
            cur = by_rule[r["rule_id"]].setdefault(r["k"], False)
            by_rule[r["rule_id"]][r["k"]] = cur or bool(r["screen_fires"])
    which_pair_all = sum(1 for u, ks in by_rule.items()
                         if ks.get(2, False))
    which_pair_f = sum(1 for u, ks in by_rule.items()
                       if u in faithful and ks.get(2, False))
    n_lad = len(by_rule)
    n_lad_f = sum(1 for u in by_rule if u in faithful)
    rep["restatements"]["ladder_pair_decides"] = dict(
        all=dict(n=n_lad, decides=which_pair_all),
        faithful=dict(n=n_lad_f, decides=which_pair_f))
    print(f"which-pair decides: {which_pair_all}/{n_lad} all -> "
          f"{which_pair_f}/{n_lad_f} faithful")

    # ---- populations that can only be labelled ---------------------------
    rep["labels_only"] = dict(
        kyverno=("parsed from structured policy fields by a separate code "
                 "path; parse_prom_expr never runs on kyverno targets"),
        dominance_88_132=("report.json stores aggregate dominance rates "
                          "without per-pool identity; labelled as computed "
                          "over the six-tuple-representable population"),
        cross_reading_pairs=("same: aggregate archive, labelled"))

    json.dump(rep, open(os.path.join(OUT, "downstream_provenance.json"),
                        "w"), indent=1)
    print("\nwrote results/e2e/downstream_provenance.json")


if __name__ == "__main__":
    main()
