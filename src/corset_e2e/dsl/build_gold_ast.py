"""Rebuild the gold reading of every rule as its faithful parsed AST.

The previous gold was a six-tuple produced by regex, which took the first
metric on the left-hand side and dropped everything structural around it,
so a ratio was recorded as its numerator (see
analysis/gold_faithfulness_audit.py). This replaces it. A unit now gets
either the parsed expression, or an explicit status saying why it has no
gold -- never a projection.

Status codes:
  OK              parsed; `ast` is the canonical form of the real rule
  REPAIRED        parsed after recovering a comment whose terminator was
                  lost when the corpus was flattened to one line
  WRONG_LANGUAGE  not PromQL (LogQL line/label filters)
  UNPARSEABLE     PromQL that this parser cannot read; counted, never
                  silently approximated

The archive keeps the raw expression alongside the AST so any downstream
claim can be re-derived, and records both the strict key (with label
matchers) and the shape key (without), since recall is reported on shape.

This reads the hidden store and writes a hidden artifact. It is NOT part
of generation and must never be imported by generator code.

Run: PYTHONPATH=. python3 corset_e2e/dsl/build_gold_ast.py
Writes results/e2e/gold_ast.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.ast_canon import canon, canon_key, shape_key  # noqa: E402
from corset_e2e.dsl.promql_repair import parse_repairing_comments  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")
HID = os.path.join(OUT, "hidden_store")

# LogQL, not PromQL: line filters and pipeline stages
LOGQL_MARKERS = ("|=", "|~", "!~ \"", "| json", "| logfmt", "| unwrap",
                 "| pattern", "|!")


def classify(raw: str):
    """(status, ast_or_None, how)."""
    x = (raw or "").strip()
    if not x:
        return "UNPARSEABLE", None, "empty"
    node, how = parse_repairing_comments(x)
    if node is not None:
        return ("REPAIRED" if how == "repaired" else "OK"), node, how
    if any(t in x for t in LOGQL_MARKERS):
        return "WRONG_LANGUAGE", None, how
    return "UNPARSEABLE", None, how


def main() -> None:
    led = json.load(open(os.path.join(OUT, "cluster_ledger.json")))
    gold = {}
    stats = Counter()
    per_split = Counter()
    for e in led:
        rid = e["rule_id"]
        h = json.load(open(os.path.join(HID, f"{rid}.target.json")))
        raw = h.get("raw_expr") or ""
        status, node, how = classify(raw)
        stats[status] += 1
        per_split[(e["split"], status)] += 1
        rec = dict(rule_id=rid, split=e["split"], repo=e["repo"],
                   status=status, for_s=float(h.get("for_s", 0.0)),
                   raw_expr=raw)
        if node is not None:
            rec["ast"] = canon(node, with_matchers=True)
            rec["strict_key"] = canon_key(node, True)
            rec["shape_key"] = shape_key(node)
            rec["repaired"] = (status == "REPAIRED")
        else:
            rec["reason"] = how
        gold[rid] = rec

    n = len(gold)
    ok = stats["OK"] + stats["REPAIRED"]
    print(f"units {n}")
    for k, c in stats.most_common():
        print(f"  {k:<15s} {c:5d}  ({c / n:.2%})")
    print(f"  ---> gold available for {ok} = {ok / n:.2%}")
    for split in ("dev", "cal", "test"):
        tot = sum(v for (s, _st), v in per_split.items() if s == split)
        good = sum(v for (s, st), v in per_split.items()
                   if s == split and st in ("OK", "REPAIRED"))
        print(f"  {split:<5s} {good}/{tot}")

    # distinct readings: how much of the corpus is the same rule text
    shapes = Counter(r["shape_key"] for r in gold.values() if "shape_key" in r)
    print(f"distinct shape keys: {len(shapes)} over {ok} parsed golds")
    print(f"largest shape cluster: {shapes.most_common(1)[0][1]}")

    assert ok / n > 0.97, "gold availability regressed below 97%"
    assert stats["UNPARSEABLE"] + stats["WRONG_LANGUAGE"] == n - ok, \
        "status codes do not partition the corpus"
    json.dump(dict(n_units=n, stats=dict(stats), gold=gold),
              open(os.path.join(OUT, "gold_ast.json"), "w"))
    print("\nwrote results/e2e/gold_ast.json")


if __name__ == "__main__":
    main()
