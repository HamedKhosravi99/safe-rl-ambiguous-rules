"""Smoke test for the PromQL parser: hand-checked cases, then the corpus.

Stage 1 asserts structure on expressions whose parse is known by hand,
including the precedence and matching-modifier cases that the previous
regex approach got wrong. Stage 2 parses every raw expression in the
corpus and reports the parse rate with a breakdown of failure messages,
so that "cannot parse" is separated from "parsed, out of grammar".

Run: PYTHONPATH=. python3 corset_e2e/analysis/smoke_promql_ast.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.promql_ast import (  # noqa: E402
    Aggregate, Binary, Call, MatrixSelector, NumberLit, Paren, Subquery,
    Unary, VectorSelector, duration_seconds, fold_constant, parse, try_parse,
    unparen)


def check(cond, label):
    if not cond:
        raise AssertionError(label)
    print(f"  ok  {label}")


def stage1() -> None:
    print("[1] hand-checked parses")

    e = unparen(parse('up{job="x"} > 0.5'))
    check(isinstance(e, Binary) and e.op == ">", "simple comparison")
    check(isinstance(e.lhs, VectorSelector) and e.lhs.metric == "up",
          "lhs vector selector")
    check(e.lhs.matchers[0].name == "job" and e.lhs.matchers[0].value == "x",
          "label matcher parsed")
    check(isinstance(e.rhs, NumberLit) and e.rhs.value == 0.5, "rhs literal")

    # precedence: comparison binds looser than arithmetic
    e = unparen(parse("a / b > 0.1"))
    check(isinstance(e, Binary) and e.op == ">", "cmp looser than div")
    check(isinstance(unparen(e.lhs), Binary) and unparen(e.lhs).op == "/",
          "division under comparison")

    # and binds looser than comparison -- the case the regex split destroyed
    e = unparen(parse("( (a:r1h > (14.4 * 0.001)) and on(env,type) "
                      "(a:r5m > (14.4 * 0.001)) )"))
    check(isinstance(e, Binary) and e.op == "and", "top-level is `and`")
    check(e.matching is not None and e.matching.on, "on(...) modifier")
    check(e.matching.labels == ("env", "type"), "matching labels")
    lhs, rhs = unparen(e.lhs), unparen(e.rhs)
    check(isinstance(lhs, Binary) and lhs.op == ">", "left conjunct is a cmp")
    check(isinstance(rhs, Binary) and rhs.op == ">", "right conjunct is a cmp")
    check(abs(fold_constant(lhs.rhs) - 0.0144) < 1e-12,
          "folded constant 14.4*0.001 = 0.0144")

    e = unparen(parse("sum by (cluster) (rate(x_total[5m])) > 2"))
    check(isinstance(e, Binary) and e.op == ">", "aggregate under comparison")
    agg = unparen(e.lhs)
    check(isinstance(agg, Aggregate) and agg.op == "sum", "sum aggregate")
    check(agg.grouping == ("cluster",) and not agg.without, "by(cluster)")
    call = unparen(agg.expr)
    check(isinstance(call, Call) and call.func == "rate", "rate() call")
    check(isinstance(call.args[0], MatrixSelector), "range selector arg")
    check(call.args[0].range_s == 300.0, "[5m] = 300s")

    e = unparen(parse("sum(x) without (a, b)"))
    check(isinstance(e, Aggregate) and e.without and e.grouping == ("a", "b"),
          "trailing without(...)")

    e = unparen(parse("topk(5, x)"))
    check(isinstance(e, Aggregate) and e.op == "topk"
          and isinstance(e.param, NumberLit) and e.param.value == 5.0,
          "topk parameter separated from the vector")

    e = unparen(parse("a offset 1h unless a"))
    check(isinstance(e, Binary) and e.op == "unless", "unless set op")
    check(unparen(e.lhs).offset_s == 3600.0, "offset 1h")

    e = unparen(parse("x / y > bool 1"))
    check(isinstance(e, Binary) and e.bool_, "bool modifier")

    e = unparen(parse("max_over_time(rate(x[5m])[30m:1m])"))
    call = e
    check(isinstance(call, Call) and isinstance(call.args[0], Subquery),
          "subquery argument")
    check(call.args[0].range_s == 1800.0 and call.args[0].step_s == 60.0,
          "subquery range and step")

    e = unparen(parse("- x + 3"))
    check(isinstance(e, Binary) and e.op == "+", "unary minus binds tighter")
    check(isinstance(unparen(e.lhs), Unary), "unary node")

    e = unparen(parse("# comment\n x > 1"))
    check(isinstance(e, Binary), "comment stripped")

    e = unparen(parse('{__name__="x"} > 1'))
    check(isinstance(unparen(e.lhs), VectorSelector)
          and unparen(e.lhs).metric is None, "bare matcher selector")

    e = unparen(parse("a > 1 and b > 2 or c > 3"))
    check(isinstance(e, Binary) and e.op == "or", "or is loosest")

    e = unparen(parse("2 ^ 3 ^ 2"))
    check(isinstance(unparen(e.rhs), Binary), "^ is right associative")

    check(duration_seconds("1h30m") == 5400.0, "compound duration")
    check(fold_constant(parse("1 - 14.4 * 0.005")) is not None,
          "fold 1 - 14.4*0.005")

    # things that must NOT parse
    for bad in ("a >", "sum by (", "up{job=}", "a and and b"):
        node, err = try_parse(bad)
        check(node is None, f"rejects {bad!r} ({err})")

    # a metric literally named like an aggregator is not an aggregate
    e = unparen(parse("count_total > 1"))
    check(isinstance(unparen(e.lhs), VectorSelector), "sum-like metric name")

    # comment repair must not silently truncate: taking the first cut that
    # parses reduced this real corpus expression to the bare selector `see`
    from corset_e2e.dsl.promql_ast import metric_names
    from corset_e2e.dsl.promql_repair import parse_repairing_comments
    flat = ("# Without max_over_time, failed scrapes could create false "
            "negatives, see # https://example.invalid/gauges for details. "
            'max_over_time(alertmanager_config_last_reload_successful'
            '{job="am"}[5m]) == 0')
    node, how = parse_repairing_comments(flat)
    check(node is not None and how == "repaired", "flattened comment repaired")
    check(metric_names(node) == ["alertmanager_config_last_reload_successful"],
          "repair keeps the code, not the prose word")


def stage2() -> None:
    print("[2] parse the whole corpus")
    from corset_e2e.calibration.run_e2e_v4 import hid, load_split
    import json
    led = json.load(open(os.path.join(ROOT, "results/e2e",
                                      "cluster_ledger.json")))
    ok = Counter()
    fails = Counter()
    examples = {}
    total = 0
    for e in led:
        h = hid(e["rule_id"])
        x = (h.get("raw_expr") or "").strip()
        if not x:
            continue
        total += 1
        node, err = try_parse(x)
        if node is None:
            fails[err] += 1
            examples.setdefault(err, x[:150])
        else:
            ok[e["split"]] += 1
    n_ok = sum(ok.values())
    print(f"    parsed {n_ok}/{total} = {n_ok / total:.2%} of all splits")
    for split in ("dev", "cal", "test"):
        tot = sum(1 for e in led if e["split"] == split)
        print(f"      {split}: {ok[split]}/{tot}")
    if fails:
        print("    failure modes:")
        for err, c in fails.most_common(12):
            print(f"      {c:5d}  {err}")
            print(f"             e.g. {examples[err]}")
    return n_ok, total


if __name__ == "__main__":
    stage1()
    n_ok, total = stage2()
    print(f"\nSMOKE {'PASS' if n_ok / total > 0.95 else 'REVIEW'}: "
          f"parse rate {n_ok / total:.2%}")
