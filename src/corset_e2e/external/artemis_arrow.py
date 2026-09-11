"""ARTEMIS external validation, stage 2: Keep, baselines, metrics
(REGISTRATION_V17).  Reads results/e2e/artemis_units.json, writes
results/e2e/artemis_external.json.  Nothing here touches the artifact.

Run: PYTHONPATH=. python3 corset_e2e/external/artemis_arrow.py [--units path]
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
UNITS = os.path.join(ROOT, "results/e2e", "artemis_units.json")
OUT = os.path.join(ROOT, "results/e2e", "artemis_external.json")
DELTA = 0.10
BINS = [("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), (">10", 11, 10**9)]
FLAGSHIP = "gemini-2.5-flash/nl2structnl-reflect"


def pool_view(u, which):
    """Per-unit pool: list of (src, trial, cls) for valid trials in the pool."""
    tr = [t for t in u["trials"] if t["valid"] and (t["in_p1"] if which == "P1" else True)]
    return tr


def unit_stats(u, which):
    tr = pool_view(u, which)
    n_valid = len(tr)
    freq = Counter(t["cls"] for t in tr)
    classes = sorted(freq)
    match = {int(k): set(v) for k, v in u["cand_match"].items()}
    P = set(range(u["label_classes"]))                     # expert classes (semantic)
    plaus_classes = {c for c in classes if match.get(c)}   # candidate classes equal to some expert class
    covered_experts = set().union(*[match[c] for c in plaus_classes]) if plaus_classes else set()
    # scores
    score = {c: freq[c] / n_valid for c in classes} if n_valid else {}
    first_order = {}
    for i, t in enumerate(tr):
        first_order.setdefault(t["cls"], i)
    # baselines
    flag = [t for t in tr if t["src"] == FLAGSHIP]
    top1 = flag[0]["cls"] if flag else (tr[0]["cls"] if tr else None)
    maj = max(classes, key=lambda c: (freq[c], -first_order[c])) if classes else None
    s_max = max((score[c] for c in plaus_classes), default=None)
    s_min = min((score[c] for c in plaus_classes), default=None)
    return dict(uid=u["uid"], group=u["group"], n_plausible=len(P), n_valid=n_valid, n_classes=len(classes),
                n_expert_proposed=len(covered_experts), proposed=bool(plaus_classes),
                score=score, match=match, P=P, top1=top1, majority=maj, s_max=s_max, s_min=s_min,
                undecided=u["undecided"])


def eval_set(st, U):
    """Metrics for a retained set U (set of class ids)."""
    hit = set().union(*[st["match"][c] for c in U if st["match"].get(c)]) if U else set()
    P = st["P"]
    return dict(any=bool(hit), recall=len(hit) / len(P), all=bool(hit == P),
                recall_proposed=(len(hit) / st["n_expert_proposed"]) if st["n_expert_proposed"] else None,
                size=len(U), efficiency=(len(U) / st["n_classes"]) if st["n_classes"] else None)


def conformal_loo(stats, key):
    """LOO split-conformal thresholds over proposed units at level DELTA."""
    cal = [s for s in stats if s["proposed"]]
    scores = np.array([s[key] for s in cal])
    out = {}
    for s in stats:
        others = scores[[c["uid"] != s["uid"] for c in cal]] if s["proposed"] else scores
        n = len(others)
        k = int(math.floor(DELTA * (n + 1)))
        q = float(np.sort(others)[k - 1]) if k >= 1 else 0.0     # k-th smallest; k=0 -> keep everything
        out[s["uid"]] = q
    return out


def summarize(rows, name):
    n = len(rows)
    if not n:
        return dict(n=0)
    return {name: dict(n=n, abstain=int(sum(1 for r in rows if r["size"] == 0)),
                       any=float(np.mean([r["any"] for r in rows])), recall=float(np.mean([r["recall"] for r in rows])),
                       all=float(np.mean([r["all"] for r in rows])), size_median=float(np.median([r["size"] for r in rows])),
                       size_mean=float(np.mean([r["size"] for r in rows])),
                       efficiency_median=float(np.median([r["efficiency"] for r in rows if r["efficiency"] is not None])) if any(r["efficiency"] is not None for r in rows) else None,
                       recall_proposed=float(np.mean([r["recall_proposed"] for r in rows if r["recall_proposed"] is not None])) if any(r["recall_proposed"] is not None for r in rows) else None)}


def run_pool(units, which):
    stats = [unit_stats(u, which) for u in units]
    n_prop = sum(s["proposed"] for s in stats)
    q_max = conformal_loo(stats, "s_max")
    q_min = conformal_loo(stats, "s_min")
    per_unit = []
    for s in stats:
        classes = set(s["score"])
        U_max = {c for c in classes if s["score"][c] >= q_max[s["uid"]] - 1e-12}
        U_min = {c for c in classes if s["score"][c] >= q_min[s["uid"]] - 1e-12}
        arms = dict(top1=eval_set(s, {s["top1"]} if s["top1"] is not None else set()),
                    majority=eval_set(s, {s["majority"]} if s["majority"] is not None else set()),
                    arrow=eval_set(s, U_max), arrow_min=eval_set(s, U_min), pool=eval_set(s, classes))
        per_unit.append(dict(uid=s["uid"], group=s["group"], n_plausible=s["n_plausible"], proposed=s["proposed"],
                             n_valid=s["n_valid"], n_classes=s["n_classes"], qhat=q_max[s["uid"]], qhat_min=q_min[s["uid"]],
                             s_max=s["s_max"], arms=arms, undecided=s["undecided"]))
    def agg(sel, label):
        rows = [r for r in per_unit if sel(r)]
        d = dict(n=len(rows), n_proposed=sum(r["proposed"] for r in rows))
        for arm in ("top1", "majority", "arrow", "arrow_min", "pool"):
            d.update(summarize([r["arms"][arm] for r in rows], arm))
            d.update({f"{arm}_proposed_only": summarize([r["arms"][arm] for r in rows if r["proposed"]], arm)[arm]}) if any(r["proposed"] for r in rows) else None
        return d
    res = dict(n_units=len(stats), n_proposed=n_prop, delta_gen_hat=1 - n_prop / len(stats),
               overall=agg(lambda r: True, "all"),
               by_bin={b: agg(lambda r, lo=lo, hi=hi: lo <= r["n_plausible"] <= hi, b) for b, lo, hi in BINS},
               by_group={g: agg(lambda r, g=g: r["group"] == g, g) for g in sorted({r["group"] for r in per_unit})},
               per_unit=per_unit)
    # branch rules
    prop = [r for r in per_unit if r["proposed"]]
    cov = float(np.mean([r["arms"]["arrow"]["any"] for r in prop])) if prop else None
    se = math.sqrt(0.9 * 0.1 / len(prop)) if prop else None
    eff = float(np.median([r["arms"]["arrow"]["efficiency"] for r in prop])) if prop else None
    res["branch_rules"] = dict(
        rule1_coverage_on_proposed=cov, rule1_threshold=(0.9 - 2 * se) if se else None,
        rule1_calibration_fails=bool(cov is not None and cov < 0.9 - 2 * se),
        rule2_median_efficiency=eff, rule2_score_uninformative=bool(eff is not None and eff > 0.9))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", default=UNITS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    d = json.load(open(a.units))
    units = d["units"]
    res = dict(registration="REGISTRATION_V17.md", artifact_commit=d["artifact_commit"], delta_sem=DELTA,
               checker_validation={k: v for k, v in d["checker_validation"].items() if not k.endswith("files")},
               n_units=len(units), n_plausible_hist=dict(sorted(Counter(u["label_classes"] for u in units).items())),
               undecided_total={k: sum(u["undecided"][k] for u in units) for k in ("labels", "candidates", "matches")},
               pools={w: run_pool(units, w) for w in ("P1", "P2")})
    json.dump(res, open(a.out, "w"), indent=1)
    for w in ("P1", "P2"):
        p = res["pools"][w]
        print(f"== {w}: units {p['n_units']} proposed {p['n_proposed']} (delta_gen_hat {p['delta_gen_hat']:.3f}); branch: {p['branch_rules']}")
        o = p["overall"]
        for arm in ("top1", "majority", "arrow", "arrow_min", "pool"):
            s = o[arm]
            print(f"   {arm:10s} any {s['any']:.3f} recall {s['recall']:.3f} all {s['all']:.3f} size med {s['size_median']} eff med {s['efficiency_median']}")
        for b, v in p["by_bin"].items():
            if v["n"]:
                print(f"   bin {b:5s} n={v['n']:3d} top1 recall {v['top1']['recall']:.3f} majority {v['majority']['recall']:.3f} arrow {v['arrow']['recall']:.3f} (size {v['arrow']['size_median']}) pool {v['pool']['recall']:.3f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
