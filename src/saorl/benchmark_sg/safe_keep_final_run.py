"""Recovered driver (originally run as a one-off script from the repository root).
Run from the repository root:  python3 saorl/benchmark_sg/safe_keep_final_run.py
"""
import os
"""Definitive SAFE-KEEP experiment: sections 4,5,6,9,10,11."""
import json, yaml, collections, time, os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
from scipy.optimize import linprog
from saorl.benchmark_sg.parse import (PromReading, PromTarget, dur_to_s, parse_prom_expr, prom_threshold_bank)
from saorl.benchmark_sg.evaluate import build_prom_pool
from saorl.benchmark_sg.control_suite import compile_instance, _flow
from saorl.benchmark_sg.score import class_score
from saorl.benchmark_sg.safe_keep import (td_params, sem_dominates_td, safe_keep_semantic,
                                          reachable_states, reachable_cost_dominates)
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
                    reading=PromReading(metric=p["metric"],selectors=p["selectors"],
                        comparator=p["comparator"],threshold=p["threshold"],
                        rate_window_s=p["rate_window_s"],aggregation=p["aggregation"],
                        agg_by=p["agg_by"],for_s=dur_to_s(r.get("for")) or 0.0,
                        severity="",logic="",axis="identity",op_tag="identity"),
                    text=" ".join(str(ann.get(k,"")) for k in ("summary","description")),
                    repo=fe.get("repo","?"),commit="",file="",sub_source="",last_commit_ts=0,raw_expr=""))
bank=prom_threshold_bank(targets)

def lpmax(m,obj,cons,d):
    A_eq,b_eq=_flow(m)
    A_ub=np.array([m["C"][k].reshape(-1) for k in cons]) if cons else None
    r=linprog(-obj.reshape(-1),A_ub=A_ub,b_ub=([d]*len(cons) if cons else None),
              A_eq=A_eq,b_eq=b_eq,bounds=(0,None),method="highs")
    return (-r.fun) if r.status==0 else None

Kraw=[]; Ksafe=[]; Kcur=[]; tsafe=0.0
cover_fail=0; lp_ok=0; lp_bad=0; maxviol=0.0; badcases=[]
sem_reach=collections.Counter(); regime=collections.Counter()
keepfail=[]; arrow=collections.Counter(); nsub=0
for t in targets:
    try: pool=build_prom_pool(t,bank)
    except Exception: continue
    g=pool.classes[pool.gold_idx].rep
    if g.comparator not in (">",">="): continue
    key=lambda x:(x.metric,x.selectors,x.aggregation,x.agg_by)
    famc=[(i,c) for i,c in enumerate(pool.classes)
          if key(c.rep)==key(g) and c.rep.comparator in (">",">=")]
    byk={}
    for i,c in famc: byk.setdefault((float(c.rep.threshold),float(c.rep.for_s)),(i,c.rep))
    keys=sorted(byk); reps=[byk[k][1] for k in keys]
    if len(reps)<2: continue
    nsub+=1
    s=time.time(); keep_idx,keyed,caps,lvl=safe_keep_semantic(reps); tsafe+=time.time()-s
    Kraw.append(len(reps)); Ksafe.append(len(keep_idx))
    scores={k:class_score(t,byk[k][1] if False else pool.classes[byk[k][0]]) for k in keys}
    cur=[i for i,k in enumerate(keys) if scores[k]>=QHAT]
    Kcur.append(len(cur) if cur else 1)
    # SAFETY-COVER INVARIANT under the corrected relation
    if not all(any(sem_dominates_td(j,i,caps,lvl) for j in keep_idx) for i in range(len(keys))):
        cover_fail+=1
    # is this a natural CURRENT-KEEP failure? (gold key dropped by score)
    gk=(float(g.threshold),float(g.for_s))
    gi_key=keys.index(gk) if gk in byk else None
    if gi_key is not None and cur and gi_key not in cur:
        keepfail.append((t.name,gk,[keys[i] for i in cur],[keys[i] for i in keep_idx]))
    # SECTION 6: LP redundancy of each discarded reading, compiling U u {psi}
    U=[reps[i] for i in keep_idx]
    for i in range(len(keys)):
        if i in keep_idx: continue
        allr=U+[reps[i]]
        kk=sorted({(float(x.threshold),float(x.for_s)) for x in allr})
        if len(kk)>4: continue
        try: m=compile_instance(allr)
        except Exception: continue
        if m["C"].shape[0]!=len(kk): continue
        pi_i=kk.index(keys[i]); ui=sorted(set(kk.index((float(x.threshold),float(x.for_s))) for x in U)-{pi_i})
        if not ui: continue
        reach=reachable_states(m)
        c2,l2=td_params(allr)[3],td_params(allr)[4]
        for j in ui:
            sem_reach[(sem_dominates_td(j,pi_i,c2,l2), reachable_cost_dominates(m,j,pi_i,reach))]+=1
        Mg=lpmax(m,m["C"][pi_i],[],None)
        if not Mg or Mg<=1e-9: continue
        d=0.5*Mg; W=lpmax(m,m["C"][pi_i],ui,d)
        if W is None: continue
        if W<=d+1e-9: lp_ok+=1
        else:
            lp_bad+=1; maxviol=max(maxviol,(W-d)/d)
            if len(badcases)<5: badcases.append((t.name[:24],keys[i],[kk[j] for j in ui],W/d))
    # SECTIONS 9/10/11: ARROW on the SAFE-KEEP antichain
    if len(keep_idx)==1: arrow["safe_singleton"]+=1; regime["semantic singleton"]+=1; continue
    arrow["multiple_incomparable"]+=1
    allr=[reps[i] for i in keep_idx]
    kk=sorted({(float(x.threshold),float(x.for_s)) for x in allr})
    if len(kk)>4: arrow["too large to DECIDE"]+=1; continue
    try: m=compile_instance(allr)
    except Exception: arrow["compile err"]+=1; continue
    if m["C"].shape[0]!=len(kk): continue
    K=m["C"].shape[0]
    Mall=max(lpmax(m,m["C"][k],[],None) or 0 for k in range(K))
    if Mall<=1e-9: arrow["uninformative"]+=1; continue
    d=0.5*Mall
    VU=lpmax(m,m["r"],list(range(K)),d)
    if VU is None: arrow["U infeasible"]+=1; continue
    # DECIDE: does some single reading suffice (face test at eps=0)?
    suff=False
    for a in range(K):
        Va=lpmax(m,m["r"],[a],d)
        if Va is None: continue
        worst=-1e18
        for b in range(K):
            if b==a: continue
            A_eq,b_eq=_flow(m)
            rr=linprog(-m["C"][b].reshape(-1),
                       A_ub=np.array([m["C"][a].reshape(-1),-m["r"].reshape(-1)]),
                       b_ub=[d,-(Va)],A_eq=A_eq,b_eq=b_eq,bounds=(0,None),method="highs")
            if rr.status!=0: worst=1e18; break
            worst=max(worst,-rr.fun)
        if worst<=d+1e-9: suff=True; break
    if suff: arrow["ARROW sufficient"]+=1; regime["irrelevant / sufficient"]+=1
    else:    arrow["ARROW ambiguous"]+=1; regime["genuinely decision-relevant"]+=1
out=dict(nsub=nsub,Kraw=Kraw,Ksafe=Ksafe,Kcur=Kcur,cover_fail=cover_fail,
         lp_ok=lp_ok,lp_bad=lp_bad,maxviol=maxviol,tsafe=tsafe,
         sem_reach={str(k):v for k,v in sem_reach.items()},
         arrow=dict(arrow),regime=dict(regime),n_keepfail=len(keepfail))
json.dump(out,open("results/safe_keep_final/safekeep_final.json","w"))
print(f"subfamilies: {nsub}   SAFE-KEEP runtime: {tsafe:.3f}s")
print(f"\n{'method':16s} {'median K':>9s} {'p90 K':>7s} {'max K':>7s} {'cover failures':>15s}")
for lab,arr,cf in (("raw (keep-all)",Kraw,0),("current score KEEP",Kcur,None),("SAFE-KEEP",Ksafe,cover_fail)):
    print(f"{lab:16s} {int(np.median(arr)):9d} {int(np.percentile(arr,90)):7d} {max(arr):7d} "
          f"{(str(cf) if cf is not None else 'n/a (measured sep.)'):>15s}")
print(f"\nSECTION 6 LP redundancy: confirmed {lp_ok}, REFUTED {lp_bad}, max rel violation {maxviol:.2e}")
for b in badcases: print("   REFUTATION:",b)
print(f"\nsem-dominance vs reachable dominance: {out['sem_reach']}")
print(f"\nnatural CURRENT-KEEP failures found: {len(keepfail)}")
print(f"\nSECTION 9/10 ARROW after SAFE-KEEP: {dict(arrow)}")
print(f"SECTION 11 regimes: {dict(regime)}")