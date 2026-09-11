"""V15: the G1 pipeline (reference-valued thresholds), one frozen pass.

Protocol identical to every prior pass (REGISTRATION_V15): fit weights on
dev by the frozen coordinate search, calibrate qhat on cal at
delta_sem=0.10, execute the test split once. The only change is the
threshold axis, which becomes a union type -- grid constants union
licensed REFERENCE names -- so readings stay six-tuples and the frozen
additive scorer, its branch-and-bound retention and O(1) pool membership
all apply unchanged.

Reference candidates are target-blind: for the unit's licensed metric set
M, R = {catalog names containing some m in M as a substring} (a reference
series is a transformation of the same quantity: slo:max:hard:X of X)
union the top REF_TOPK catalog names by the v11 name score, capped at
REF_CAP. Never reads the target.

Run: PYTHONPATH=. python3 corset_e2e/calibration/run_e2e_g1.py
Writes results/e2e/e2e_report_g1.json
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.schema import (  # noqa: E402
    AGGREGATIONS, COMPARATORS, FOR_S, IN_DSL, RATE_WINDOWS_S, THRESHOLD_GRID)
from corset_e2e.dsl.schema_g1 import (  # noqa: E402
    IN_G1_REF, classify_target_g1, in_pool_g1, target_reading_g1)
from corset_e2e.generator.generate import metric_subtokens, tokens  # noqa: E402
from corset_e2e.scoring.deterministic import DEFAULT_WEIGHTS, Scorer  # noqa: E402
from corset_e2e.calibration.run_e2e_v11 import V11Server  # noqa: E402
from corset_e2e.calibration.run_e2e_v4 import (  # noqa: E402
    DELTA_SEM, WEIGHT_GRID, cluster_bootstrap, hid, load_split, vis)

OUT = os.path.join(ROOT, "results/e2e")
REF_TOPK = 24
REF_CAP = 64


def refs_for(metrics, index, toks) -> list:
    """Target-blind reference candidates for a licensed metric set.

    A reference series is a transformation of the same quantity
    (slo:max:hard:X of X), so it contains X's subtokens; candidates are
    found by intersecting the index's posting lists on the rarest
    subtoken of each licensed metric, which keeps the scan proportional
    to that posting list rather than to the whole catalogue.
    """
    out, seen = [], set()
    csub, post = index.csub, index.post_name
    idf = index.idf_name
    for m in metrics[:8]:
        S = csub.get(m) or set()
        if not S:
            continue
        rarest = max(S, key=lambda t: idf.get(t, 0.0))
        for r in post.get(rarest, ()):
            if r in seen or r == m:
                continue
            rs = csub.get(r) or set()
            if S <= rs and len(rs) > len(S):
                seen.add(r)
                out.append(r)
                if len(out) >= REF_CAP:
                    return out
    tset = set(toks)
    if len(out) < REF_CAP:
        scored = []
        for t in tset:
            for r in post.get(t, ())[:200]:
                if r in seen:
                    continue
                rs = csub.get(r) or set()
                if rs:
                    scored.append((len(rs & tset) / len(rs), r))
        scored.sort(key=lambda x: (-x[0], x[1]))
        for _sc, r in scored[:REF_TOPK]:
            if r not in seen:
                seen.add(r)
                out.append(r)
            if len(out) >= REF_CAP:
                break
    return out[:REF_CAP]


def unit_slots(entry, server, v):
    cat = server.catalog_for(entry, v)
    toks = tokens(v.get("text", "")) + tokens(v.get("alert_name", ""))
    metrics = cat.license(toks, [])
    idx = server.idx[entry["repo"]]
    refs = refs_for(metrics, idx, toks)
    return dict(mode="complete", metrics=tuple(metrics),
                comparators=COMPARATORS, thresholds=THRESHOLD_GRID,
                references=tuple(refs), windows=RATE_WINDOWS_S,
                aggregations=AGGREGATIONS, fors=FOR_S), toks


def build_scorer(v, slots, weights):
    """Frozen scorer with the threshold axis widened to grid union refs."""
    sc = Scorer(v, slots, weights)
    text = v.get("text", "")
    alert = v.get("alert_name", "")
    tset = set(tokens(text) + tokens(alert))

    def s_ref(r):
        sub = set(metric_subtokens(r))
        return len(sub & tset) / len(sub) if sub else 0.0

    sc.tables[2] = list(sc.tables[2]) + [
        (r, weights["num"] * s_ref(r)) for r in slots["references"]]
    return sc


def score_g1(sc, reading) -> float:
    """Additive score of a reading whose threshold axis is a union type.

    Identical to the frozen Scorer.score except that the threshold key is
    matched by value for numbers and by name for references, instead of
    being coerced to float.
    """
    vals = list(reading)
    total = 0.0
    for axis, v in zip(sc.tables, vals):
        hit = None
        for key, s_ in axis:
            if v is None or key is None:
                same = (key is v)
            elif isinstance(v, str) or isinstance(key, str):
                same = (key == v)
            else:
                same = (round(float(key), 6) == round(float(v), 6))
            if same:
                hit = s_
                break
        if hit is None:
            return float("-inf")
        total += hit
    return total


def gold_info(entry, server, weights):
    v = vis(entry["rule_id"])
    slots, _toks = unit_slots(entry, server, v)
    sc = build_scorer(v, slots, weights)
    h = hid(entry["rule_id"])
    label = classify_target_g1(h)
    if label not in (IN_DSL, IN_G1_REF):
        return label, None, sc, slots
    g = target_reading_g1(h)
    if g is None or not in_pool_g1(slots, g):
        return "GEN_MISS", None, sc, slots
    return label, score_g1(sc, g), sc, slots


def fit_weights(dev, server, sample=160):
    rng = random.Random(7)
    pool = list(dev)
    rng.shuffle(pool)
    pool = pool[:sample]

    def objective(w):
        tot = n = 0.0
        for e in pool:
            label, s, sc, _sl = gold_info(e, server, w)
            if label not in (IN_DSL, IN_G1_REF) or s is None:
                continue
            cnt, _ = sc.retained(s)
            tot += math.log1p(cnt)
            n += 1
        return tot / max(n, 1)

    w = dict(DEFAULT_WEIGHTS)
    best = objective(w)
    for _ in range(1):
        for k, vals in WEIGHT_GRID.items():
            for vv in vals:
                if vv == w[k]:
                    continue
                cand = dict(w)
                cand[k] = vv
                o = objective(cand)
                if o < best - 1e-9:
                    best, w = o, cand
    return dict(weights=w, dev_objective=best, n_dev_units=len(pool))


def evaluate(entries, server, weights, qhat=None):
    rows = []
    for e in entries:
        label, s, sc, _sl = gold_info(e, server, weights)
        row = dict(rid=e["rule_id"], cluster=e["cluster_id"], repo=e["repo"],
                   label=label, gold_score=s)
        if qhat is not None and s is not None:
            row["retained_gold"] = bool(s >= qhat)
            cnt, _ = sc.retained(qhat)
            row["set_size"] = cnt
        rows.append(row)
    return rows


def summarise(rows):
    reach = [r for r in rows if r["label"] in (IN_DSL, IN_G1_REF, "GEN_MISS")]
    ind = [r for r in reach if r["label"] in (IN_DSL, IN_G1_REF)]

    def by_cluster(sel, pred):
        d = defaultdict(list)
        for r in sel:
            d[r["cluster"]].append(1 if pred(r) else 0)
        return d

    g = cluster_bootstrap(by_cluster(reach, lambda r: r["label"] != "GEN_MISS"))
    rt = cluster_bootstrap(by_cluster(ind, lambda r: r.get("retained_gold")))
    e2e = cluster_bootstrap(by_cluster(
        rows, lambda r: r["label"] != "GEN_MISS" and r.get("retained_gold")))
    n_ret = sum(1 for r in rows
                if r["label"] != "GEN_MISS" and r.get("retained_gold"))
    ref_rows = [r for r in reach if r["label"] == IN_G1_REF
                or (r["label"] == "GEN_MISS")]
    sizes = sorted(r["set_size"] for r in ind if "set_size" in r)
    return dict(
        n_units=len(rows), n_expressible=len(reach), n_licensed=len(ind),
        n_retained=n_ret,
        expressible_frac=len(reach) / len(rows),
        rho_gen=dict(point=round(g[0], 4), lo=round(g[1], 4), hi=round(g[2], 4)),
        rho_ret=dict(point=round(rt[0], 4), lo=round(rt[1], 4), hi=round(rt[2], 4)),
        rho_e2e=dict(point=round(e2e[0], 4), lo=round(e2e[1], 4), hi=round(e2e[2], 4)),
        e2e_over_corpus=n_ret / len(rows),
        set_size_median=(sizes[len(sizes) // 2] if sizes else None))


def main() -> None:
    dev, cal, test = load_split("dev"), load_split("cal"), load_split("test")
    print("[1] server")
    server = V11Server()
    print(f"[2] fit weights on dev ({len(dev)})")
    fit = fit_weights(dev, server)
    w = fit["weights"]
    print("    ", w)
    print(f"[3] calibrate on cal ({len(cal)})")
    cal_rows = evaluate(cal, server, w)
    gold = sorted(r["gold_score"] for r in cal_rows
                  if r["gold_score"] is not None)
    n = len(gold)
    k = max(1, math.floor(DELTA_SEM * (n + 1)))
    qhat = gold[k - 1]
    print(f"    n_cal_in_grammar={n} k={k} qhat={qhat:.6f}")
    print(f"[4] one frozen test pass ({len(test)})")
    rows = evaluate(test, server, w, qhat=qhat)
    s = summarise(rows)
    # the reference subpopulation, reported separately
    ref = [r for r in rows if r["label"] == IN_G1_REF]
    refmiss = [r for r in rows if r["label"] == "GEN_MISS"]
    s["ref_subpop"] = dict(
        n_ref_expressible=len(ref) + 0,
        n_ref_retained=sum(1 for r in ref if r.get("retained_gold")),
        n_gen_miss=len(refmiss))
    rep = dict(registration="V15 (REGISTRATION_V15.md)", weights=w,
               dev_fit=fit, qhat=qhat, delta_sem=DELTA_SEM,
               ref_topk=REF_TOPK, ref_cap=REF_CAP, test=s)
    json.dump(rep, open(os.path.join(OUT, "e2e_report_g1.json"), "w"), indent=1)
    json.dump(rows, open(os.path.join(OUT, "e2e_test_rows_g1.json"), "w"))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
