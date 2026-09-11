"""Recovered driver (originally run as a one-off script from the repository root).
Run from the repository root:  python3 saorl/benchmark_sg/final_pipeline_run.py
"""
import os
"""FINAL PIPELINE: ADMIT -> RAW -> SAFE-KEEP -> ARROW -> CERTIFIED-PROTECT.
No fresh CHECK. True model used ONLY after the deployment decision (audit)."""
import json, yaml, collections, time, os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(v,"1")
import numpy as np
from scipy.optimize import linprog
from saorl.benchmark_sg.parse import (PromReading, PromTarget, dur_to_s, parse_prom_expr, prom_threshold_bank)
from saorl.benchmark_sg.evaluate import build_prom_pool
from saorl.benchmark_sg.control_suite import compile_instance, _flow
from saorl.benchmark_sg.safe_keep import safe_keep_semantic, td_params, sem_dominates_td
from saorl.benchmark_sg.certified_protect import (draw_statistics, radii, flow_matrices,
                                                  eval_policy, occupancy_of, SAFE_A, M_SUPP)
from saorl.benchmark_sg.control_mdp import GAMMA
LAM=(0.05,0.10,0.25,0.50); NTX=int(1e10); MAXK=4

# ---------------- SECTION 2: ADMISSION (no gold access) ----------------
man=json.load(open("results/conformal/e0/fetch_manifest_e0.json"))
adm=collections.Counter(); targets=[]
for fe in man["files"]:
    try: docs=list(yaml.safe_load_all(open(fe["abs_path"] if os.path.isabs(fe["abs_path"]) else os.path.join(os.getcwd(), fe["abs_path"]))))
    except Exception: continue
    for doc in docs:
        if not isinstance(doc,dict): continue
        for grp in ((doc.get("spec",{}) or {}).get("groups") or doc.get("groups") or []):
            for r in (grp.get("rules") or []):
                if "alert" not in r or "expr" not in r: continue
                adm["total alert rules"]+=1
                p=parse_prom_expr(str(r["expr"]))
                if p is None: adm["A=0 parse/DSL failure"]+=1; continue
                if p["comparator"] not in (">",">="):
                    adm["A=0 unsupported comparator family"]+=1; continue
                adm["A=1 admitted"]+=1
                targets.append(PromTarget(name=r.get("alert",""),group="",
                    reading=PromReading(metric=p["metric"],selectors=p["selectors"],
                        comparator=p["comparator"],threshold=p["threshold"],
                        rate_window_s=p["rate_window_s"],aggregation=p["aggregation"],
                        agg_by=p["agg_by"],for_s=dur_to_s(r.get("for")) or 0.0,
                        severity="",logic="",axis="identity",op_tag="identity"),
                    text="",repo=fe.get("repo","?"),commit="",file="",sub_source="",last_commit_ts=0,raw_expr=""))
bank=prom_threshold_bank(targets)
print("SECTION 2 ADMISSION:",dict(adm))
tot=adm["total alert rules"]; print(f"  admission rate = {adm['A=1 admitted']}/{tot} = {adm['A=1 admitted']/tot:.3f}\n")

def lpmax(m,obj,cons,d):
    A_eq,b_eq=_flow(m)
    A_ub=np.array([m["C"][k].reshape(-1) for k in cons]) if cons else None
    r=linprog(-obj.reshape(-1),A_ub=A_ub,b_ub=([d]*len(cons) if cons else None),
              A_eq=A_eq,b_eq=b_eq,bounds=(0,None),method="highs")
    return (-r.fun) if r.status==0 else None

def certified_protect(P,mu0,r,Cs,d,N,rng):
    """Fixed-data certified planner. Uses ONLY the sampled statistics of D."""
    nS,nA,_=P.shape
    counts,Phat=draw_statistics(P,mu0,N,rng); b=radii(counts,nS,nA,N)
    Phat=Phat.copy(); Phat[:,SAFE_A]=0.0; Phat[:,SAFE_A,0]=1.0; b[:,SAFE_A]=0.0
    mask=(counts>=M_SUPP); mask[:,SAFE_A]=True
    Gam=1.0/(1.0-GAMMA); pen=GAMMA*Gam*b; Vr=float(r.max()-r.min())*Gam
    A_eq,b_eq=flow_matrices(Phat,mu0)
    res=linprog(-(r-GAMMA*Vr*b).reshape(-1),
                A_ub=np.array([(c+pen).reshape(-1) for c in Cs]),b_ub=[d]*len(Cs),
                A_eq=A_eq,b_eq=b_eq,
                bounds=[(0,None) if m_ else (0,0) for m_ in mask.reshape(-1)],method="highs")
    if res.status!=0:
        pi=np.zeros((nS,nA)); pi[:,SAFE_A]=1.0; return pi,True
    x=res.x.reshape(nS,nA); pi=np.zeros((nS,nA)); tt=x.sum(1)
    for s in range(nS): pi[s]=x[s]/tt[s] if tt[s]>1e-12 else np.eye(nA)[SAFE_A]
    return pi,bool((pi[:,SAFE_A]>=0.999).all())

# ---------------- SECTIONS 3-5: raw invariant, SAFE-KEEP ----------------
gen_viol=0; nadm=0; rawK=[]; safeK=[]; tsafe=0.0; audit_ok=0; audit_bad=0
cases=[]
for t in targets:
    try: pool=build_prom_pool(t,bank)
    except Exception: continue
    g=pool.classes[pool.gold_idx].rep
    key=lambda x:(x.metric,x.selectors,x.aggregation,x.agg_by)
    fam=[c.rep for c in pool.classes if key(c.rep)==key(g) and c.rep.comparator in (">",">=")]
    byk={}
    for x in fam: byk.setdefault((float(x.threshold),float(x.for_s)),x)
    keys=sorted(byk); reps=[byk[k] for k in keys]
    gk=(float(g.threshold),float(g.for_s))
    nadm+=1
    if gk not in byk: gen_viol+=1; continue      # SECTION 3 invariant
    s=time.time(); keep,keyed,caps,lvl=safe_keep_semantic(reps); tsafe+=time.time()-s
    rawK.append(len(reps)); safeK.append(len(keep))
    # assert every removed reading has an implication certificate
    for i in range(len(keys)):
        if i in keep: continue
        if not any(sem_dominates_td(j,i,caps,lvl) for j in keep):
            audit_bad+=1
    cases.append((t,reps,keys,keep,gk))
print(f"SECTION 3 raw-generation invariant: admitted pools {nadm}, VIOLATIONS {gen_viol}")
if gen_viol: print("  *** STOP CONDITION ***")
print(f"SECTION 4 SAFE-KEEP: raw K median {int(np.median(rawK))}, safe K median "
      f"{int(np.median(safeK))}, runtime {tsafe:.3f}s, missing certificates {audit_bad}\n")

# ---------------- SECTIONS 6-15: branches, planning, audit ----------------
res=collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
rows=[]
rng_master=np.random.default_rng(20260909)
sample=cases if len(cases)<=400 else [cases[i] for i in
        rng_master.choice(len(cases),400,replace=False)]
for t,reps,keys,keep,gk in sample:
    U=[reps[i] for i in keep]
    gpos=keys.index(gk)
    allr=U+([reps[gpos]] if gpos not in keep else [])
    kk=sorted({(float(x.threshold),float(x.for_s)) for x in allr})
    if len(kk)>MAXK: 
        for l in LAM: res[l]["skip"]["compile too large"]+=1
        continue
    try: m=compile_instance(allr)
    except Exception:
        for l in LAM: res[l]["skip"]["compile error"]+=1
        continue
    if m["C"].shape[0]!=len(kk): continue
    gi=kk.index(gk); ui=sorted(set(kk.index((float(x.threshold),float(x.for_s))) for x in U))
    gold_in_U = gi in ui
    ui_planner=[j for j in ui if j!=gi] if not gold_in_U else ui
    if not ui_planner: continue
    Mg=lpmax(m,m["C"][gi],[],None)
    if not Mg or Mg<=1e-9:
        for l in LAM: res[l]["skip"]["uninformative gold"]+=1
        continue
    Vunc=lpmax(m,m["r"],[],None)
    for l in LAM:
        d=l*Mg
        K=len(ui_planner)
        # SECTION 6: branch
        branch = "semantic singleton" if len(keep)==1 else None
        arrow_out=None
        if branch is None:
            VU=lpmax(m,m["r"],ui_planner,d)
            if VU is None: res[l]["skip"]["U infeasible"]+=1; continue
            suff=False
            for a in ui_planner:
                Va=lpmax(m,m["r"],[a],d)
                if Va is None: continue
                worst=-1e18
                for b_ in ui_planner:
                    if b_==a: continue
                    A_eq,b_eq=_flow(m)
                    rr=linprog(-m["C"][b_].reshape(-1),
                        A_ub=np.array([m["C"][a].reshape(-1),-m["r"].reshape(-1)]),b_ub=[d,-Va],
                        A_eq=A_eq,b_eq=b_eq,bounds=(0,None),method="highs")
                    if rr.status!=0: worst=1e18; break
                    worst=max(worst,-rr.fun)
                if worst<=d+1e-9: suff=True; break
            arrow_out="ARROW sufficient" if suff else "ARROW ambiguous"
            branch=arrow_out
        # SECTION 7/11: ALWAYS protect with the certified FULL-SET planner
        pi,fb=certified_protect(m["P"],m["mu0"],m["r"],[m["C"][j] for j in ui_planner],
                                d,NTX,np.random.default_rng(abs(hash((t.name,l)))%(2**32)))
        # ---- deployment decision is now FINAL; true model used only below ----
        Jg=eval_policy(m["P"],m["C"][gi],m["mu0"],pi)
        Jr=eval_policy(m["P"],m["r"],m["mu0"],pi)
        maxU=max(eval_policy(m["P"],m["C"][j],m["mu0"],pi) for j in ui_planner)
        VU=lpmax(m,m["r"],ui_planner,d) or 0.0
        occ=occupancy_of(m["P"],m["mu0"],pi)
        res[l]["branch"][branch]+=1
        res[l]["deploy"]["deployed"]+=1
        if Jg>d+1e-9: res[l]["deploy"]["UNSAFE vs parsed gold"]+=1
        if maxU>d+1e-9: res[l]["deploy"]["UNSAFE vs U_safe"]+=1
        if fb: res[l]["deploy"]["trivial fallback"]+=1
        rows.append(dict(name=t.name,repo=t.repo,lam=l,branch=branch,gold_in_U=bool(gold_in_U),
            Jg_over_d=float(Jg/d),Jr=float(Jr),VU=float(VU),Vunc=float(Vunc or 0),
            rew_over_VU=float(Jr/VU) if VU>0 else None,
            rew_over_unc=float(Jr/Vunc) if Vunc else None,
            safe_action_frac=float(occ[:,SAFE_A].sum()),fallback=bool(fb)))
json.dump(rows,open("results/final_pipeline/pipeline_rows.json","w"))
print("SECTION 12/13 BY BUDGET")
print(f"{'lam':>5s} {'deployed':>9s} {'sem-singleton':>14s} {'ARROW suff':>11s} {'ARROW amb':>10s} "
      f"{'UNSAFE(gold)':>13s} {'trivial fb':>11s}")
for l in LAM:
    b=res[l]["branch"]; d_=res[l]["deploy"]
    print(f"{l:5.2f} {d_['deployed']:9d} {b['semantic singleton']:14d} {b['ARROW sufficient']:11d} "
          f"{b['ARROW ambiguous']:10d} {d_['UNSAFE vs parsed gold']:13d} {d_['trivial fallback']:11d}")
print(f"\nskips: {[(l,dict(res[l]['skip'])) for l in LAM]}")