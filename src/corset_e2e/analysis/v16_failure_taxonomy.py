"""Failure taxonomy of expression-tree (G2) generation from the V16 per-unit
stage flags (dev, out-of-fold; replayed by run_v16_taxonomy.py with no
constant changed). Stages: shape (predicate / boolean composition), group
(metric set: retrieval), term (aggregation / window / function skeleton),
numeric (threshold constant), label (label set), for (on-axis), complete
(full AST emitted). Classes: first failing stage in pipeline order, the
multiset of failing component stages, and 'components available but not
composed'.

Run: PYTHONPATH=. python3 corset_e2e/analysis/v16_failure_taxonomy.py
Writes results/e2e/v16_failure_taxonomy.json
"""
import json, os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "results/e2e")
ORDER = ("shape", "group", "term", "numeric", "label", "for_on_axis")
NAMES = dict(shape="predicate shape / boolean composition not proposed", group="metric set not retrieved (top-25 groups)",
             term="aggregation-window-function skeleton not proposed", numeric="threshold constant not on axis",
             label="label set not available", for_on_axis="for-duration off axis",
             composition="all components available, complete AST not emitted", complete="complete AST proposed")


def main():
    flags = json.load(open(os.path.join(R, "v16_unit_flags.json")))
    units = list(flags.values()); n = len(units)

    def first_failure(u):
        if u["complete"]:
            return "complete"
        for s in ORDER:
            if not u[s]:
                return s
        return "composition"

    def weights(sub):
        raw = Counter(); dist = Counter(); seen = set(); per_repo = defaultdict(Counter); nrep = defaultdict(int)
        for u in sub:
            c = first_failure(u); raw[c] += 1; per_repo[u["repo"]][c] += 1; nrep[u["repo"]] += 1
            tk = tuple(u["textkey"])
            if tk not in seen:
                seen.add(tk); dist[c] += 1
        nd = len(seen)
        macro = Counter()
        for c in list(NAMES):
            macro[c] = sum(per_repo[rp][c] / nrep[rp] for rp in nrep) / len(nrep) if nrep else 0.0
        return dict(n=len(sub), n_distinct=nd, raw={c: raw[c] / len(sub) for c in NAMES}, distinct={c: dist[c] / nd for c in NAMES}, repo_macro={c: macro[c] for c in NAMES},
                    counts={c: raw[c] for c in NAMES})

    def multiset(sub):
        ms = Counter()
        for u in sub:
            if u["complete"]:
                key = "complete"
            else:
                fails = tuple(s for s in ORDER if not u[s]); key = "+".join(fails) if fails else "composition_only"
            ms[key] += 1
        return {k: dict(n=v, frac=v / len(sub)) for k, v in ms.most_common()}

    def availability(sub):
        allc = [u for u in sub if all(u[s] for s in ORDER)]
        return dict(n=len(sub), all_components=len(allc), all_components_frac=len(allc) / len(sub),
                    complete=sum(1 for u in sub if u["complete"]), complete_frac=sum(1 for u in sub if u["complete"]) / len(sub),
                    composition_efficiency=(sum(1 for u in allc if u["complete"]) / len(allc) if allc else None),
                    stage_pass={s: sum(1 for u in sub if u[s]) / len(sub) for s in ORDER})

    by_m = {}
    for lab, pred in (("single_metric", lambda u: u["n_metrics"] == 1), ("two_metrics", lambda u: u["n_metrics"] == 2), ("three_plus", lambda u: u["n_metrics"] >= 3)):
        sub = [u for u in units if pred(u)]
        if sub:
            by_m[lab] = dict(availability=availability(sub), first_failure=weights(sub)["raw"])
    repos = sorted({u["repo"] for u in units})
    res = dict(n=n, names=NAMES, first_failure=weights(units), multiset=multiset(units), availability=availability(units), by_metric_count=by_m,
               per_repo={rp: dict(availability=availability([u for u in units if u["repo"] == rp]), first_failure=weights([u for u in units if u["repo"] == rp])["raw"]) for rp in repos})
    json.dump(res, open(os.path.join(R, "v16_failure_taxonomy.json"), "w"), indent=1)
    print(f"n={n} distinct={res['first_failure']['n_distinct']}")
    print("STAGE PASS RATES (raw):", {s: round(v, 3) for s, v in res["availability"]["stage_pass"].items()})
    print(f"all components available {res['availability']['all_components_frac']:.3f}; complete {res['availability']['complete_frac']:.3f}; composition efficiency {res['availability']['composition_efficiency']}")
    print("FIRST-FAILURE CLASSES (raw / distinct / repo-macro):")
    for c in NAMES:
        f = res["first_failure"]; print(f"  {NAMES[c]:55s} {f['counts'][c]:5d}  {100*f['raw'][c]:5.1f}%  {100*f['distinct'][c]:5.1f}%  {100*f['repo_macro'][c]:5.1f}%")
    print("FAILING-STAGE MULTISETS (top 12):")
    for k, v in list(res["multiset"].items())[:12]:
        print(f"  {k:45s} {v['n']:5d} {100*v['frac']:5.1f}%")
    print("BY METRIC COUNT:", json.dumps({k: dict(n=v["availability"]["n"], all_components=round(v["availability"]["all_components_frac"], 3), complete=round(v["availability"]["complete_frac"], 3), eff=v["availability"]["composition_efficiency"]) for k, v in by_m.items()}))
    for rp in repos:
        a = res["per_repo"][rp]["availability"]; print(f"  {rp:30s} n {a['n']:4d} all-components {a['all_components_frac']:.2f} complete {a['complete_frac']:.2f} eff {a['composition_efficiency']}")


if __name__ == "__main__":
    main()
