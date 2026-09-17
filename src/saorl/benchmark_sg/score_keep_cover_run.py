"""Score-Keep cover audit on the Safe-Keep subfamilies (Table 19 of the ARROW paper).
Run from the repository root:  python3 src/saorl/benchmark_sg/score_keep_cover_run.py
Same subfamilies, score, frozen threshold and implication relation as safe_keep_final_run.py
(semantic threshold-duration dominance on the deduplicated subfamily). A subfamily counts as an
uncovered removal when Score-Keep retains at least one reading and some removed reading is implied
by no retained reading; subfamilies where Score-Keep retains nothing (Algorithm 2 abstains) are
counted separately. Writes results/safe_keep_final/score_keep_cover.json."""

import json, yaml, collections, os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
from saorl.benchmark_sg.parse import (PromReading, PromTarget, dur_to_s, parse_prom_expr, prom_threshold_bank)
from saorl.benchmark_sg.evaluate import build_prom_pool
from saorl.benchmark_sg.score import class_score
from saorl.benchmark_sg.safe_keep import sem_dominates_td, safe_keep_semantic
QHAT=json.load(open("results/conformal/e0/qhat_freeze.json"))["qhat_sem"]
man=json.load(open("results/conformal/e0/fetch_manifest_e0.json"))
targets=[]
for fe in man["files"]:
    try: docs=list(yaml.safe_load_all(open(fe["abs_path"] if os.path.isabs(fe["abs_path"]) else os.path.join(os.getcwd(), fe["abs_path"]))))
    except Exception: continue
    for doc in docs:
        if not isinstance(doc,dict): continue
        for grp in ((doc.get("spec",{}) or {}).get("groups") or doc.get("groups") or []):
            for r in (grp.get("rules") or []):
                if "alert" not in r or "expr" not in r: continue
                p=parse_prom_expr(str(r["expr"]))
                if p is None: continue
                ann=r.get("annotations",{}) or {}
                targets.append(PromTarget(name=r.get("alert",""),group="",
                    reading=PromReading(metric=p["metric"],selectors=p["selectors"],comparator=p["comparator"],threshold=p["threshold"],
                        rate_window_s=p["rate_window_s"],aggregation=p["aggregation"],agg_by=p["agg_by"],for_s=dur_to_s(r.get("for")) or 0.0,
                        severity="",logic="",axis="identity",op_tag="identity"),
                    text=" ".join(str(ann.get(k,"")) for k in ("summary","description")),
                    repo=fe.get("repo","?"),commit="",file="",sub_source="",last_commit_ts=0,raw_expr=""))
bank=prom_threshold_bank(targets)
nsub=0; safe_fail=0; score_empty=0; score_fail=0; score_gold_dropped=0
Kraw=[]; Ksafe=[]; Kcur=[]
for t in targets:
    try: pool=build_prom_pool(t,bank)
    except Exception: continue
    g=pool.classes[pool.gold_idx].rep
    if g.comparator not in (">",">="): continue
    key=lambda x:(x.metric,x.selectors,x.aggregation,x.agg_by)
    famc=[(i,c) for i,c in enumerate(pool.classes) if key(c.rep)==key(g) and c.rep.comparator in (">",">=")]
    byk={}
    for i,c in famc: byk.setdefault((float(c.rep.threshold),float(c.rep.for_s)),(i,c.rep))
    keys=sorted(byk); reps=[byk[k][1] for k in keys]
    if len(reps)<2: continue
    nsub+=1
    keep_idx,keyed,caps,lvl=safe_keep_semantic(reps)
    Kraw.append(len(reps)); Ksafe.append(len(keep_idx))
    covered=lambda kept: all(any(sem_dominates_td(j,i,caps,lvl) for j in kept) for i in range(len(keys)))
    if not covered(keep_idx): safe_fail+=1
    scores={k:class_score(t,pool.classes[byk[k][0]]) for k in keys}
    cur=[i for i,k in enumerate(keys) if scores[k]>=QHAT]
    Kcur.append(len(cur) if cur else 1)                       # as in the archived driver
    if not cur:
        score_empty+=1                                        # Algorithm 2 abstains: nothing removed
        continue
    if not covered(cur): score_fail+=1
    gk=(float(g.threshold),float(g.for_s))
    if gk in byk and keys.index(gk) not in cur: score_gold_dropped+=1
out = dict(nsub=nsub, safe_keep_cover_failures=safe_fail,
    score_keep_retains_nothing=score_empty, score_keep_cover_failures=score_fail,
    score_keep_governing_dropped=score_gold_dropped,
    median_K_raw=float(np.median(Kraw)), median_K_safe=float(np.median(Ksafe)), median_K_score=float(np.median(Kcur)),
    max_K_raw=int(max(Kraw)), max_K_safe=int(max(Ksafe)), max_K_score=int(max(Kcur)))
os.makedirs("results/safe_keep_final", exist_ok=True)
json.dump(out, open("results/safe_keep_final/score_keep_cover.json", "w"), indent=1)
print(json.dumps(out, indent=1))
