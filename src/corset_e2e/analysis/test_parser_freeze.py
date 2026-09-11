"""Frozen regression gate for the parser, the repair, and the AST gold.

The parser is now foundational: reach, proposal recall and coverage are
all measured through it, so a silent change in how one expression parses
would move published numbers with no visible cause. This pins the exact
canonical parse of every unit in the corpus under one digest.

It also pins the FAITHFULNESS classification, so the specific bug that
made this work necessary -- a lossy six-tuple silently standing in for a
structurally different rule -- cannot return unnoticed. Any unit whose
status or canonical form moves fails the gate and prints the diff.

Regenerate deliberately, never casually:
    PYTHONPATH=. python3 corset_e2e/analysis/test_parser_freeze.py --freeze
and commit the change with the reason.

Check:
    PYTHONPATH=. python3 corset_e2e/analysis/test_parser_freeze.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

OUT = os.path.join(ROOT, "results/e2e")
FREEZE = os.path.join(OUT, "PARSER_FREEZE.json")
SAMPLE_N = 40


def current():
    """Recompute the parse of the whole corpus from source, not from cache."""
    from corset_e2e.dsl.build_gold_ast import classify
    from corset_e2e.dsl.ast_canon import canon_key
    led = json.load(open(os.path.join(OUT, "cluster_ledger.json")))
    rows = []
    stats = {}
    for e in sorted(led, key=lambda z: z["rule_id"]):
        rid = e["rule_id"]
        h = json.load(open(os.path.join(OUT, "hidden_store",
                                        f"{rid}.target.json")))
        status, node, _how = classify(h.get("raw_expr") or "")
        key = canon_key(node, True) if node is not None else ""
        rows.append((rid, status, key))
        stats[status] = stats.get(status, 0) + 1
    digest = hashlib.sha256(
        "\n".join(f"{r}\t{s}\t{k}" for r, s, k in rows).encode()).hexdigest()
    non_ok = sorted(r for r, s, _ in rows if s not in ("OK", "REPAIRED"))
    sample = {r: hashlib.sha256(k.encode()).hexdigest()[:16]
              for r, _s, k in rows[::max(1, len(rows) // SAMPLE_N)]}
    return dict(n_units=len(rows), stats=stats, digest=digest,
                non_ok=non_ok, sample=sample), rows


def freeze() -> None:
    cur, _rows = current()
    cur["_note"] = ("Frozen parse of the corpus. A digest change means some "
                    "expression now parses differently; regenerate only with "
                    "a stated reason.")
    json.dump(cur, open(FREEZE, "w"), indent=1)
    print(f"froze {cur['n_units']} units, digest {cur['digest'][:16]}")
    print(json.dumps(cur["stats"], indent=1))


def check() -> int:
    if not os.path.exists(FREEZE):
        print("no freeze file; run with --freeze first")
        return 2
    want = json.load(open(FREEZE))
    cur, rows = current()
    fails = []
    if cur["n_units"] != want["n_units"]:
        fails.append(f"unit count {cur['n_units']} != {want['n_units']}")
    if cur["stats"] != want["stats"]:
        fails.append(f"status counts {cur['stats']} != {want['stats']}")
    if set(cur["non_ok"]) != set(want["non_ok"]):
        add = set(cur["non_ok"]) - set(want["non_ok"])
        rem = set(want["non_ok"]) - set(cur["non_ok"])
        fails.append(f"non-parsing set moved: +{sorted(add)[:5]} "
                     f"-{sorted(rem)[:5]}")
    if cur["digest"] != want["digest"]:
        by_rid = {r: (s, k) for r, s, k in rows}
        moved = [r for r, h in want["sample"].items()
                 if r in by_rid
                 and hashlib.sha256(by_rid[r][1].encode()).hexdigest()[:16] != h]
        fails.append(f"canonical parse digest changed; {len(moved)} of "
                     f"{len(want['sample'])} sampled units differ, "
                     f"e.g. {moved[:3]}")
    if fails:
        print("PARSER FREEZE FAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"PARSER FREEZE OK  ({cur['n_units']} units, "
          f"digest {cur['digest'][:16]})")
    print(f"  {json.dumps(cur['stats'])}")
    return 0


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        freeze()
    else:
        sys.exit(check())
