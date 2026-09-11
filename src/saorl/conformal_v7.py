"""REGISTRATION_V7 analysis: endpoints E-A..E-D against the frozen baseline.

Reads plausibility_cache_v7.json (one scorer, 61 units), assembles the
59-row corpus (8 authored + 3 old paraphrases + 48 new), applies the frozen
quantile rule, and compares every deployed set against the baseline frozen
in results/e2e/REGISTRATION_V7.md -- with binding equality deciding whether
downstream results stand. Also reports the registered leave-parent-out
sensitivity and the scorer-drift table.

Run:  PYTHONPATH=. python3 -m saorl.conformal_v7
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

import saorl.conformal as C
from .conformal import (
    DELTA_SEM,
    _ROUND,
    _cmapss_dsem,
    _pools,
    binding_equal,
    conformal_threshold,
    gate_pass,
)

_HERE = Path(__file__).parent
CACHE = json.load(open(_HERE / "plausibility_cache_v7.json"))
OUT = _HERE.parent.parent / "results/e2e" / "v7_calibration.json"

# gold per set_name (the authored convention, unchanged)
GOLD = {
    "ambiguous": "psi1: RUL_hat<20",
    "sharp": "phi1: RUL_hat<15",
    "conservative": "mid: RUL<22",
    "window_sharp": "psi_W4: Within_4(warn)",
    "window_ambiguous": "psi_W5: Within_5(warn)",
    "window_conservative": "psi_W5: Within_5(warn)",
    "budget_ambiguous": "psi_$60: spend>=60",
    "budget_conservative": "psi_$60: spend>=60",
}

# frozen baseline (REGISTRATION_V7.md, from the 2.1.186 caches)
BASELINE = {
    "maint-sharp": ["phi1: RUL_hat<15"],
    "maint-conservative": ["consv: anom>0.6", "mid: RUL<22", "vconsv: RUL<40"],
    "maint-ambiguous": ["psi1: RUL_hat<20", "psi2: Q05<10", "psi3: anom>0.8"],
    "window-sharp": ["psi_W4: Within_4(warn)"],
    "window-ambiguous": ["psi_W3: Within_3(warn)", "psi_W4: Within_4(warn)",
                         "psi_W5: Within_5(warn)", "psi_W6: Within_6(warn)"],
    "window-conservative": ["psi_W3: Within_3(warn)", "psi_W5: Within_5(warn)"],
    "budget-ambiguous": ["psi_$60: spend>=60", "psi_$80: spend>=80"],
    "budget-conservative": [],
    "cmapss-FD001": ["consv: anom>0.6", "mid: RUL<22", "vconsv: RUL<40"],
    "cmapss-FD002": ["mid: RUL<22", "vconsv: RUL<40"],
    "cmapss-FD003": ["consv: anom>0.6", "mid: RUL<22", "vconsv: RUL<40"],
    "cmapss-FD004": ["consv: anom>0.6", "mid: RUL<22", "vconsv: RUL<40"],
    "dsrl-cost": ["psi_b10: cost<=10", "psi_b20: cost<=20", "psi_b40: cost<=40"],
}

RULE_UNIT = {  # deployed corpus rule -> its scoring unit
    "maint-sharp": "orig:sharp", "maint-conservative": "orig:conservative",
    "maint-ambiguous": "orig:ambiguous", "window-sharp": "orig:window_sharp",
    "window-ambiguous": "orig:window_ambiguous",
    "window-conservative": "orig:window_conservative",
    "budget-ambiguous": "orig:budget_ambiguous",
    "budget-conservative": "orig:budget_conservative",
}


def _mean(vals: List[float]) -> float:
    return round(float(np.mean(vals)), _ROUND)


def _unit_means(uid: str) -> Dict[str, float]:
    return {c: _mean(v) for c, v in CACHE["units"][uid]["scores"].items()}


def _gates() -> Dict[str, Dict[str, bool]]:
    """set_name -> cand -> data-gate pass on the frozen CORPUS_SEED datasets."""
    dsems = {"maint": C._maint_dsem(), "grid": C._gridworld_dsem(), "budget": C._budget_dsem()}
    out = {}
    for sname, (pool, dom) in _pools().items():
        out[sname] = {c.name: gate_pass(c, dsems[dom])[0] for c in pool}
    return out


def _sigma(uid: str, gates) -> float:
    sname = CACHE["units"][uid]["set_name"]
    g = GOLD[sname]
    return _unit_means(uid)[g] if gates[sname][g] else 0.0


def _unit_set(uid: str, gates, q: float) -> List[str]:
    sname = CACHE["units"][uid]["set_name"]
    m = _unit_means(uid)
    return sorted(n for n, s in m.items() if (s if gates[sname][n] else 0.0) >= q)


def _parent_group(uid: str) -> str:
    """rows sharing (parent text, pool) — the leave-parent-out grouping."""
    fam, sname = uid.split(":", 1)
    if fam.startswith("oldpara") or fam[0] == "A" and fam != "A" or fam == "orig":
        pass  # handled below by mapping
    if fam == "orig":
        return {"conservative": "SEVERE:conservative", "ambiguous": "SEVERE:ambiguous",
                "sharp": "WEAROUT:sharp"}.get(sname, f"{sname}:{sname}")
    if fam.startswith("oldpara") or fam.startswith("A"):
        return {"conservative": "SEVERE:conservative",
                "ambiguous": "SEVERE:ambiguous"}[sname]
    if fam.startswith("B"):
        return "WEAROUT:sharp"
    return f"{sname}:{sname}"  # C/D/E: parent text is per-pool-set


def main() -> None:
    gates = _gates()
    corpus = [u for u in CACHE["units"]
              if u.split(":", 1)[1] in GOLD and u != "orig:budget_sharp"]
    corpus = [u for u in corpus if CACHE["units"][u]["set_name"] in GOLD]
    n = len(corpus)
    sig = {u: _sigma(u, gates) for u in corpus}
    k = max(1, math.floor(DELTA_SEM * (n + 1)))
    q_full = float(sorted(sig.values())[k - 1])

    # E-A
    knife = sum(1 for v in sig.values() if abs(v - 0.5) < 1e-9)
    print(f"n={n} rows, k={k}, q_full={q_full:.6f}  (E-A)")
    print(f"gold scores exactly at 0.500: {knife}/{n}")

    # E-C
    cov = []
    for u in corpus:
        others = [sig[o] for o in corpus if o != u]
        cov.append(sig[u] >= conformal_threshold(others, DELTA_SEM))
    loo_cov = float(np.mean(cov))
    print(f"LOO coverage: {loo_cov:.4f} (nominal >= 0.9)  (E-C)")

    # E-B: deployed sets, LOO (registered) + leave-parent-out (sensitivity)
    deployed = {}
    dsems_multi = {}
    for dom, fn in (("maint", C._maint_dsem), ("grid", C._gridworld_dsem),
                    ("budget", C._budget_dsem)):
        keep = C.CORPUS_SEED
        outl = []
        for s in range(10):
            C.CORPUS_SEED = s
            outl.append(fn())
        C.CORPUS_SEED = keep
        dsems_multi[dom] = outl

    pools = _pools()
    for rule, uid in RULE_UNIT.items():
        sname = CACHE["units"][uid]["set_name"]
        pool, dom = pools[sname]
        others = [sig[o] for o in corpus if o != uid]
        q = conformal_threshold(others, DELTA_SEM)
        U_new = _unit_set(uid, gates, q)
        grp = _parent_group(uid)
        others_p = [sig[o] for o in corpus if _parent_group(o) != grp]
        q_p = conformal_threshold(others_p, DELTA_SEM)
        U_par = _unit_set(uid, gates, q_p)
        base = sorted(BASELINE[rule])
        beq = binding_equal(pool, U_new, base, dsems_multi[dom]) if U_new or base else True
        deployed[rule] = dict(
            qhat_loo=round(q, 6), U_new=U_new, U_baseline=base,
            set_equal=U_new == base, binding_equal=bool(beq),
            qhat_parent_out=round(q_p, 6), U_parent_out=U_par,
            parent_out_equals_loo=U_par == U_new,
            gold=GOLD[sname], gold_covered=sig[uid] >= q)

    # cmapss: conservative pool, gates recomputed per subset, maint-conservative LOO q
    others = [sig[o] for o in corpus if o != "orig:conservative"]
    q_m = conformal_threshold(others, DELTA_SEM)
    m_cons = _unit_means("orig:conservative")
    pool_c, _ = pools["conservative"]
    for sub in ("FD001", "FD002", "FD003", "FD004"):
        dsem_r = _cmapss_dsem(sub)
        eff = {c.name: (m_cons[c.name] if gate_pass(c, dsem_r)[0] else 0.0) for c in pool_c}
        U_new = sorted(nm for nm, s in eff.items() if s >= q_m)
        base = sorted(BASELINE[f"cmapss-{sub}"])
        beq = binding_equal(pool_c, U_new, base, [dsem_r])
        deployed[f"cmapss-{sub}"] = dict(qhat_loo=round(q_m, 6), U_new=U_new,
                                         U_baseline=base, set_equal=U_new == base,
                                         binding_equal=bool(beq))

    # dsrl: fresh rule at q_full, no gates
    m_d = _unit_means("orig:dsrl_cost")
    U_new = sorted(nm for nm, s in m_d.items() if s >= q_full)
    base = sorted(BASELINE["dsrl-cost"])
    deployed["dsrl-cost"] = dict(qhat=round(q_full, 6), U_new=U_new, U_baseline=base,
                                 set_equal=U_new == base, scores=m_d)

    print("\nE-B deployed sets (LOO, registered):")
    for rule, d in deployed.items():
        flag = "SAME" if d["set_equal"] else ("BINDING-EQ" if d.get("binding_equal") else "DIFFERS")
        print(f"  {rule:22s} {flag:10s} new={d['U_new']}")
        if not d["set_equal"]:
            print(f"  {'':22s} baseline={d['U_baseline']}")

    # E-D: scorer drift on the original units
    drift = {}
    for uid in [u for u in CACHE["units"] if u.startswith("orig:") or u.startswith("oldpara")]:
        sname = CACHE["units"][uid]["set_name"]
        new = _unit_means(uid)
        if uid.startswith("oldpara"):
            f = sorted((_HERE.parent.parent / "results/paper_extra" / "judge_robustness").glob("*.json"))[0]
            rows = json.load(open(f))["paraphrase"]["rows"]
            i = int(uid.split(":")[0].split("-")[1]) - 1
            old = {k2: float(v) for k2, v in rows[i]["means"].items()}
        elif sname == "dsrl_cost":
            dc = json.load(open(_HERE / "plausibility_cache_dsrl.json"))["sets"]["dsrl_cost"]
            old = {k2: float(np.mean(v)) for k2, v in dc["scores"].items()}
        elif sname == "budget_sharp":
            pool, _ = pools[sname]
            old = {c.name: c.plaus_mean() for c in pool}
        else:
            pool, _ = pools[sname]
            old = {c.name: c.plaus_mean() for c in pool}
        ds = {c: round(new[c] - old[c], 4) for c in new if c in old}
        rank_old = sorted(old, key=old.get, reverse=True)
        rank_new = sorted(new, key=new.get, reverse=True)
        drift[uid] = dict(delta=ds, max_abs=max(abs(v) for v in ds.values()),
                          rank_flip=rank_old != rank_new)
    worst = max(drift.values(), key=lambda d: d["max_abs"])
    print(f"\nE-D scorer drift: max |delta| across original units = {worst['max_abs']}"
          f"; rank flips: {sum(d['rank_flip'] for d in drift.values())}/{len(drift)}")

    OUT.parent.mkdir(exist_ok=True)
    json.dump(dict(n=n, k=k, q_full=q_full, knife_edge_count=knife,
                   loo_coverage=loo_cov, deployed=deployed, drift=drift,
                   sigmas={u: sig[u] for u in corpus},
                   cli=CACHE["model"], registration="REGISTRATION_V7"),
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
