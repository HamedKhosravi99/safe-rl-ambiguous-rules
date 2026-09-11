"""Choose the G2 capability set and its grids from the DEVELOPMENT split.

The greedy capability curve in grammar_taxonomy.py was computed on the
test split to answer whether a ~90% grammar is reachable at all. It cannot
be used to CHOOSE the grammar: picking the capabilities that cover the
test set is selection on test, and the resulting reach would not be a
prediction of anything.

So the grammar is designed here, on dev, exactly as the frozen six-tuple's
grids were. Dev supplies:

  * the capability set -- greedily, stopping at the smallest set reaching
    the dev reach target;
  * the enlarged threshold, window and `for` grids -- the frozen decade
    grid plus the scalar values dev rules actually use, including those
    written as arithmetic (`14.4 * 0.001`), which fold to constants.

Test is not read. The resulting spec is written to results/e2e/G2_SPEC.json
and evaluated against test exactly once, elsewhere.

Run: PYTHONPATH=. python3 corset_e2e/analysis/derive_g2_from_dev.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.analysis.grammar_taxonomy import features_for  # noqa: E402
from corset_e2e.dsl.schema import (  # noqa: E402
    FOR_S, RATE_WINDOWS_S, THRESHOLD_GRID)
from corset_e2e.calibration.run_e2e_v4 import hid, load_split, vis  # noqa: E402

OUT = os.path.join(ROOT, "results/e2e")
DEV_REACH_TARGET = 0.90
MAX_CAPABILITIES = 20
# a capability is DECLARED if dev uses it at least this often (~0.5% of dev)
FREQ_MIN = 5


def scalar_values(ast, out):
    """Every folded numeric constant appearing in a canonical AST."""
    if isinstance(ast, list):
        if len(ast) >= 2 and ast[0] == "num" and isinstance(ast[1], (int, float)):
            out.append(float(ast[1]))
        for x in ast:
            if isinstance(x, list):
                scalar_values(x, out)


def window_values(ast, out):
    if isinstance(ast, list):
        if ast and ast[0] == "range" and len(ast) >= 3:
            out.append(float(ast[2]))
        for x in ast:
            if isinstance(x, list):
                window_values(x, out)


def main() -> None:
    dev = load_split("dev")
    gold = json.load(open(os.path.join(OUT, "gold_ast.json")))["gold"]

    req = {}
    textkey = {}
    for e in dev:
        rid = e["rule_id"]
        feats, _how = features_for(hid(rid))
        req[rid] = None if feats is None else frozenset(feats)
        v = vis(rid)
        textkey[rid] = (v.get("alert_name", "").strip(),
                        " ".join((v.get("text") or "").split()))

    reachable = [e for e in dev if req[e["rule_id"]] is not None]
    n = len(dev)
    print(f"dev units {n}, parsed {len(reachable)}")

    def rates(have):
        cov = {e["rule_id"] for e in reachable if req[e["rule_id"]] <= have}
        raw = len(cov) / n
        seen, hit = set(), 0
        for e in dev:
            k = textkey[e["rule_id"]]
            if k in seen:
                continue
            seen.add(k)
            hit += e["rule_id"] in cov
        return raw, hit / len(seen), cov

    have = set()
    curve = []
    for _ in range(MAX_CAPABILITIES):
        raw, dis, _cov = rates(have)
        if raw >= DEV_REACH_TARGET:
            break
        gain = Counter()
        for e in reachable:
            r = req[e["rule_id"]]
            if r <= have:
                continue
            missing = r - have
            if len(missing) == 1:
                gain[next(iter(missing))] += 1
        if not gain:
            for e in reachable:
                r = req[e["rule_id"]]
                if not r <= have:
                    for f in r - have:
                        gain[f] += 1
        if not gain:
            break
        feat = gain.most_common(1)[0][0]
        have.add(feat)
        raw, dis, _cov = rates(have)
        curve.append(dict(added=feat, n_caps=len(have), dev_raw=raw,
                          dev_distinct=dis))
        print(f"  +{feat:<24s} dev raw {raw:6.1%}  distinct {dis:6.1%}")

    raw, dis, cov = rates(have)
    greedy_caps = sorted(have)
    print(f"\n[minimal] DEV reach with {len(have)} capabilities: "
          f"raw {raw:.1%}, distinct {dis:.1%}")

    # ---- the declared grammar: every construct dev uses more than
    # incidentally. The greedy set above is a poor DECLARATION even though it
    # is a fine minimal cover: it stops the moment dev reach crosses the
    # target, so a construct dev genuinely uses is omitted whenever it was
    # not needed to cross. `offset` (10 dev uses) and `setop_unless` (14) are
    # omitted for exactly that reason, and together they are the dominant
    # shape of another repository's alerts. Declaring by frequency is the
    # more natural rule and is still computed on dev alone.
    freq = Counter()
    for e in reachable:
        for f in req[e["rule_id"]]:
            freq[f] += 1
    declared = {f for f, c in freq.items() if c >= FREQ_MIN}
    d_raw, d_dis, _ = rates(declared)
    print(f"[declared] every capability with >= {FREQ_MIN} dev uses: "
          f"{len(declared)} capabilities, dev raw {d_raw:.1%}, "
          f"distinct {d_dis:.1%}")
    sens = {}
    for k in (3, 5, 10):
        s_caps = {f for f, c in freq.items() if c >= k}
        s_raw, _sd, _ = rates(s_caps)
        sens[k] = dict(n_caps=len(s_caps), dev_raw=s_raw)
        print(f"    sensitivity K={k:2d}: {len(s_caps):2d} caps, "
              f"dev raw {s_raw:.1%}")

    # ---- enlarged grids, from dev scalars only
    thr, wins = [], []
    for e in dev:
        g = gold.get(e["rule_id"], {})
        if "ast" not in g:
            continue
        scalar_values(g["ast"], thr)
        window_values(g["ast"], wins)
    base_t = {round(float(x), 6) for x in THRESHOLD_GRID}
    base_w = {float(w) for w in RATE_WINDOWS_S if w is not None}
    base_f = {float(x) for x in FOR_S}
    add_t = sorted({round(v, 6) for v in thr
                    if round(v, 6) not in base_t and abs(v) <= 1e11})
    add_w = sorted({v for v in wins if v not in base_w})
    add_f = sorted({float(gold[e["rule_id"]]["for_s"]) for e in dev
                    if e["rule_id"] in gold
                    and float(gold[e["rule_id"]]["for_s"]) not in base_f})
    print(f"dev-added thresholds {len(add_t)} (e.g. {add_t[:8]})")
    print(f"dev-added windows    {len(add_w)} -> {add_w}")
    print(f"dev-added for values {len(add_f)} -> {add_f}")

    spec = dict(
        derived_from="dev split only; test never read",
        rule=(f"declared = every capability used at least {FREQ_MIN} times in "
              f"dev; the greedy minimal cover is kept alongside as a variant"),
        dev_reach_target=DEV_REACH_TARGET,
        capabilities=sorted(declared),
        dev_reach=dict(raw=d_raw, distinct=d_dis, n_units=n),
        dev_capability_frequency=dict(freq.most_common()),
        freq_min=FREQ_MIN, freq_sensitivity=sens,
        minimal_variant=dict(capabilities=greedy_caps,
                             dev_reach=dict(raw=raw, distinct=dis)),
        greedy_curve=curve,
        threshold_extra=add_t, window_extra=add_w, for_extra=add_f,
        note=("threshold_extra holds folded constants dev rules use, so a "
              "rule written `14.4 * 0.001` contributes 0.0144; the generator "
              "still proposes from a finite axis"))
    json.dump(spec, open(os.path.join(OUT, "G2_SPEC.json"), "w"), indent=1)
    print("\nwrote results/e2e/G2_SPEC.json")


if __name__ == "__main__":
    main()
