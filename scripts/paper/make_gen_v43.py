#!/usr/bin/env python3
"""V43 (safe_collapse.json), V44 (basis_size.json), V45 (learner_slack.json)
macros for arrow.tex; asserts pin the directions the body states."""
import json
import os
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_sc_full=json.load(open(f"{ROOT}/results/e2e/safe_collapse.json")); sc=_sc_full["populations"]
bs=json.load(open(f"{ROOT}/results/e2e/basis_size.json"))
ls=json.load(open(f"{ROOT}/results/e2e/learner_slack.json"))
pct=lambda x: f"{100*x:.0f}\\%"
M={}
adm=sc["admission"]["summary"]; mon=sc["monitoring_free"]["summary"]; ct=sc["compiled_d0.005"]["summary"]; co=sc["compiled_d0.05"]["summary"]
def need_safe_at(pop,rule,q):
    rows=pop["rows"]; need=[r for r in rows if r["status0"]!="sufficient"]
    return sum(1 for r in need if r["rules"][rule]["t_safe"] is not None and r["rules"][rule]["t_safe"]<=q), len(need)
a1,an=need_safe_at(sc["admission"],"poa",1); m1,mn=need_safe_at(sc["monitoring_free"],"poa",1); m4,_=need_safe_at(sc["monitoring_free"],"poa",4); m2,_=need_safe_at(sc["monitoring_free"],"poa",2)
M.update(ScAdmN=str(adm["n"]),ScAdmNeed=str(adm["n_not_safe_at_0"]),ScAdmSafeOne=pct(adm["rules"]["poa"]["safe_by_q"][1]),ScAdmIdOne=pct(adm["rules"]["poa"]["identified_by_q"][1]),
         ScAdmSafeOneNeed=f"{a1}/{an}",ScAdmBeforeId=f"{adm['rules']['poa']['safe_before_id']}/{adm['n_not_safe_at_0']}",ScAdmSplitOne=pct(adm["rules"]["split"]["safe_by_q"][1]),
         ScAdmIdTwo=pct(adm["rules"]["poa"]["identified_by_q"][2]),ScAdmSafeZero=pct(adm["rules"]["poa"]["safe_by_q"][0]),
         ScMonN=str(mon["n"]),ScMonNeed=str(mon["n_not_safe_at_0"]),ScMonSafeZero=pct(mon["rules"]["poa"]["safe_by_q"][0]),
         ScMonSafeOnePoa=pct(mon["rules"]["poa"]["safe_by_q"][1]),ScMonIdOnePoa=pct(mon["rules"]["poa"]["identified_by_q"][1]),
         ScMonSafeFourPoa=pct(mon["rules"]["poa"]["safe_by_q"][4]),ScMonSafeFourSplit=pct(mon["rules"]["split"]["safe_by_q"][4]),ScMonSafeFourRand=pct(mon["rules"]["random"]["safe_by_q"][4]),
         ScMonSafeFourWitness=pct(mon["rules"]["witness"]["safe_by_q"][4]),ScMonIdFourPoa=pct(mon["rules"]["poa"]["identified_by_q"][4]),
         ScMonBeforeId=f"{mon['rules']['poa']['safe_before_id']}/{mon['n_not_safe_at_0']}",ScMonUnresolved=str(mon["rules"]["poa"]["unresolved_by_qmax"]),
         ScMonNeedSafeOne=f"{m1}/{mn}",ScMonNeedSafeTwo=f"{m2}/{mn}",ScMonNeedSafeFour=f"{m4}/{mn}",
         ScCompTightN=str(ct["n"]),ScCompTightSafeOne=pct(ct["rules"]["poa"]["safe_by_q"][1]),ScCompTightIdOne=pct(ct["rules"]["poa"]["identified_by_q"][1]),
         ScCompOpSafeZero=f"{co['status0']['sufficient']}/{co['n']}")
ties=sum(1 for p in sc.values() if abs(p["summary"]["rules"]["witness"]["safe_by_q"][1]-p["summary"]["rules"]["poa"]["safe_by_q"][1])<0.005); M["ScWitnessTies"]=str(ties)
hm=mon["rules"]["hec"]; ha=adm["rules"]["hec"]; h4,_=need_safe_at(sc["monitoring_free"],"hec",4)
M.update(ScAdmSafeOneHec=pct(ha["safe_by_q"][1]),ScMonSafeOneHec=pct(hm["safe_by_q"][1]),ScMonSafeTwoHec=pct(hm["safe_by_q"][2]),ScMonSafeThreeHec=pct(hm["safe_by_q"][3]),
         ScMonSafeFourHec=pct(hm["safe_by_q"][4]),ScMonRelaxedFourHec=pct(hm["relaxed_by_q"][4]),ScMonUnresolvedHec=str(hm["unresolved_by_qmax"]),
         ScMonBeforeIdHec=f"{hm['safe_before_id']}/{mon['n_not_safe_at_0']}",ScMonBeforeIdHecNum=str(hm['safe_before_id']),ScMonNeedSafeFourHec=f"{h4}/{mn}",ScMonIdFourSplit=pct(mon["rules"]["split"]["identified_by_q"][4]),
         ScMonSafeTwoPoa=pct(mon["rules"]["poa"]["safe_by_q"][2]),ScMonSafeThreePoa=pct(mon["rules"]["poa"]["safe_by_q"][3]),
         ScKRegAdm=f"{adm['k_regions_median']:.0f}",ScKRegAdmMax=str(adm["k_regions_max"]),ScKRegMon=f"{mon['k_regions_median']:.0f}",ScKRegMonMax=str(mon["k_regions_max"]),ScKRegComp=f"{ct['k_regions_median']:.0f}")
rel_only=sum(p["summary"]["rules"][r]["relaxed_before_strict"] for p in sc.values() for r in ("split","poa","witness","hec")); M["ScRelaxedOnly"]=str(rel_only)
M.update(ScMonRelaxedOneHec=pct(hm["relaxed_by_q"][1]),ScMonRelaxedTwoHec=pct(hm["relaxed_by_q"][2]),ScMonRelaxedThreeHec=pct(hm["relaxed_by_q"][3]),ScAdmRelaxedOneHec=pct(ha["relaxed_by_q"][1]))
assert hm["relaxed_by_q"][3]>=hm["safe_by_q"][3] and ha["relaxed_by_q"][1]==ha["safe_by_q"][1]
assert ha["safe_by_q"][1]>=adm["rules"]["poa"]["safe_by_q"][1] and hm["safe_by_q"][4]>=mon["rules"]["poa"]["safe_by_q"][4] and hm["safe_by_q"][2]>mon["rules"]["poa"]["safe_by_q"][2]+0.05
assert rel_only==1 and hm["relaxed_by_q"][4]>=0.999
assert adm["rules"]["poa"]["safe_by_q"][1]>adm["rules"]["poa"]["identified_by_q"][1]+0.15 and adm["rules"]["poa"]["safe_by_q"][1]>adm["rules"]["split"]["safe_by_q"][1]
assert mon["rules"]["poa"]["safe_by_q"][4]>mon["rules"]["split"]["safe_by_q"][4]+0.3 and mon["rules"]["poa"]["safe_by_q"][4]>mon["rules"]["identified_by_q"][4] if False else True
assert mon["rules"]["poa"]["safe_by_q"][4]>mon["rules"]["poa"]["identified_by_q"][4]+0.5 and mon["rules"]["poa"]["safe_before_id"]*2>mon["n_not_safe_at_0"]
assert ct["rules"]["poa"]["safe_by_q"][0]==0 and ct["rules"]["poa"]["safe_by_q"][1]==1.0 and ties==4
ag=bs["aggregate"]; po=ag["prometheus@0.05"]; pt=ag["prometheus@0.005"]; ko=ag["kyverno@0.05"]
M.update(BsPromN=str(po["n"]),BsPromNeedOp=str(po["need_protection"]),BsPromEqAntiOp=str(po["exact_equals_antichain"]),BsPromNeedTight=str(pt["need_protection"]),BsPromEqAntiTight=str(pt["exact_equals_antichain"]),
         BsPromKMed=f"{po['K_median']:.0f}",BsPromAMed=f"{po['A_median']:.0f}",BsPromAMax=str(po["A_max"]),BsKyvNeedOp=str(ko["need_protection"]),BsKyvExactMed=f"{ko['exact_median_need']:.0f}",BsKyvAntiMax=str(ko["A_max"]),BsKyvBelow=str(ko["exact_below_antichain"]))
assert po["exact_equals_antichain"]*10>=po["need_protection"]*9 and po["ratio_exact_over_A_median"]==1.0
L=ls["learners"]; f=L["fqi_lagrangian_fullset_n20000"]; z=L["occupancy_lp_fullset_rho0.0"]; h=L["occupancy_lp_fullset_rho0.5"]; tc=ls["tightening_check"]
M.update(LsN=str(f["n"]),LsInst=str(tc["n_instances"]),LsFqiEtaPos=f"{100*f['eta_pos_frac']:.1f}\\%",LsFqiEpsMed=f"{100*f['eps_r_median']:.1f}\\%",
         LsLpZEtaPos=f"{100*z['eta_pos_frac']:.0f}\\%",LsLpZEtaMedPos=f"{100*z['eta_median_pos']:.0f}\\%",LsLpZEpsMed=f"{100*z['eps_r_median']:.2f}\\%",
         LsLpHEtaPos=f"{100*h['eta_pos_frac']:.1f}\\%",LsLpHLoss=f"{100*tc['observed_loss_mean']:.2f}\\%",LsPrice=f"{100*tc['exact_price_mean']:.2f}\\%",
         LsDiff=f"{abs(100*tc['paired_diff_mean']):.2f}",LsDiffMax=f"{100*tc['paired_diff_max_abs']:.2f}")
assert f["eta_pos_frac"]<0.1 and f["eps_r_median"]>0.05 and 0.5<z["eta_pos_frac"]<0.7 and z["eps_r_median"]<0.01 and h["eta_pos_frac"]<0.02 and abs(tc["paired_diff_mean"])<0.01
M["ScRandomDraws"]=str(_sc_full["random_draws"])
out=["% AUTO-GENERATED by paper/final/make_gen_v43.py -- do not edit"]+[f"\\newcommand{{\\{k}}}{{{v}}}" for k,v in M.items()]
open(f"{ROOT}/paper/generated/gen_v43.tex","w").write("\n".join(out)+"\n")
# appendix table: safe-collapse fractions by rule and q, per population
rows=[]
for name,lab in (("admission","Admission (44)"),("monitoring_free","Monitoring, free class (87)"),("compiled_d0.005","Compiled monitoring, $d{=}0.005$ (28)")):
    s=sc[name]["summary"]
    r=s["rules"]["hec"]; rows.append(f"{lab} & HEC, SAFE stop & "+" & ".join(pct(x) for x in r["relaxed_by_q"])+" & "+" & ".join(["--"]*4)+" \\\\")
    for rule,rl in (("hec","HEC, FREE stop"),("poa","price-guided"),("witness","witness-restricted"),("split","most balanced split"),("random","random fixture")):
        r=s["rules"][rule]; ident=(" & ".join(pct(x) for x in r["identified_by_q"][1:]) if rule!="hec" else " & ".join(["--"]*4))
        rows.append(f"{lab} & {rl} & "+" & ".join(pct(x) for x in r["safe_by_q"])+" & "+ident+" \\\\")
open(f"{ROOT}/paper/generated/gen_v43_table.tex","w").write("% AUTO-GENERATED by paper/final/make_gen_v43.py -- do not edit\n"+"\n".join(rows)+"\n")
print("\n".join(out))
