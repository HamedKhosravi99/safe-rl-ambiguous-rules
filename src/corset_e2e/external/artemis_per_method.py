"""ARTEMIS external validation, per-method view (REGISTRATION_V17 data, no new checks).
For each generator x model source in the artifact, the source's OWN samples form the
candidate pool: we score committing to its first sample, its self-consistency singleton
over its samples, and ARROW's Score-Keep retention (LOO split conformal, delta_sem = 0.10)
on that pool, against the experts' plausible readings; plus ARROW over the union of all
sources.  Reads results/e2e/artemis_units.json, writes results/e2e/artemis_per_method.json.
Run: PYTHONPATH=src python3 src/corset_e2e/external/artemis_per_method.py
"""
import json, math, os, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, os.path.join(ROOT, "src"))
from corset_e2e.external.artemis_arrow import eval_set, conformal_loo, DELTA
UNITS = os.path.join(ROOT, "results/e2e", "artemis_units.json"); OUT = os.path.join(ROOT, "results/e2e", "artemis_per_method.json")

def unit_stats_src(u, src):
    tr = [t for t in u["trials"] if t["valid"] and (src is None or t["src"] == src)]
    n_valid = len(tr); freq = Counter(t["cls"] for t in tr); classes = sorted(freq)
    match = {int(k): set(v) for k, v in u["cand_match"].items()}; P = set(range(u["label_classes"]))
    plaus = {c for c in classes if match.get(c)}
    covered = set().union(*[match[c] for c in plaus]) if plaus else set()
    score = {c: freq[c] / n_valid for c in classes} if n_valid else {}
    first_order = {}
    for i, t in enumerate(tr): first_order.setdefault(t["cls"], i)
    top1 = tr[0]["cls"] if tr else None
    maj = max(classes, key=lambda c: (freq[c], -first_order[c])) if classes else None
    return dict(uid=u["uid"], group=u["group"], n_plausible=len(P), n_valid=n_valid, n_classes=len(classes),
                n_expert_proposed=len(covered), proposed=bool(plaus), score=score, match=match, P=P, top1=top1, majority=maj,
                s_max=max((score[c] for c in plaus), default=None))

def run(units, src):
    stats = [unit_stats_src(u, src) for u in units]; q = conformal_loo(stats, "s_max")
    rows = []
    for s in stats:
        classes = set(s["score"]); U = {c for c in classes if s["score"][c] >= q[s["uid"]] - 1e-12}
        rows.append(dict(group=s["group"], proposed=s["proposed"],
                         top1=eval_set(s, {s["top1"]} if s["top1"] is not None else set()),
                         majority=eval_set(s, {s["majority"]} if s["majority"] is not None else set()),
                         arrow=eval_set(s, U), pool=eval_set(s, classes)))
    def agg(sel):
        rs = [r for r in rows if sel(r)]
        out = dict(n=len(rs), n_proposed=sum(r["proposed"] for r in rs))
        for arm in ("top1", "majority", "arrow", "pool"):
            out[arm] = dict(any=float(np.mean([r[arm]["any"] for r in rs])), recall=float(np.mean([r[arm]["recall"] for r in rs])),
                            all=float(np.mean([r[arm]["all"] for r in rs])), size_median=float(np.median([r[arm]["size"] for r in rs])))
        return out
    return dict(overall=agg(lambda r: True), by_group={g: agg(lambda r, g=g: r["group"] == g) for g in ("Ventilator", "Robotics", "LMCPS")})

def main():
    d = json.load(open(UNITS)); units = d["units"]
    sources = sorted({t["src"] for u in units for t in u["trials"]})
    res = dict(registration="REGISTRATION_V17.md (per-method view)", artifact_commit=d["artifact_commit"], delta_sem=DELTA,
               n_units=len(units), sources={src: run(units, src) for src in sources}, union=run(units, None))
    json.dump(res, open(OUT, "w"))
    print(f"{'source':40s} {'n':>4s} | first sample: any / recall | self-cons.: any / recall | ARROW on its pool: any / recall / |U| | pool size")
    for src in sources + ["UNION"]:
        r = res["union"] if src == "UNION" else res["sources"][src]; o = r["overall"]
        print(f"{src:40s} {o['n']:4d} | {100*o['top1']['any']:5.1f} / {100*o['top1']['recall']:5.1f} | {100*o['majority']['any']:5.1f} / {100*o['majority']['recall']:5.1f} | {100*o['arrow']['any']:5.1f} / {100*o['arrow']['recall']:5.1f} / {o['arrow']['size_median']:4.1f} | {o['pool']['size_median']:4.1f}")
    print("wrote", OUT)

if __name__ == "__main__":
    main()
