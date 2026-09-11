"""V39 (Phase B): the ambiguity profile A_psi(kappa) = V_psi(d) - V_U(d - kappa),
exactly on the true model, plus the regime label from Gamma, plus a fine
budget sweep giving each instance's regime as a function of d (the band map
for Phase D).  Registration V39."""
from __future__ import annotations
import json, os
import numpy as np
from scipy.optimize import linprog
from .control_suite import _eligible, _flow, compile_instance
from .parse import parse_prometheus, prom_threshold_bank
from .evaluate import build_prom_pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "results/e2e", "ambiguity_profile.json")
BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10)
KAPPA_FRAC = np.linspace(0.0, 1.0, 21)
D_SWEEP = np.round(np.geomspace(0.001, 0.10, 25), 6)
TOL = 1e-7; FACE_TOL = 1e-7

def lp(m, obj, rows, rhs):
    A_eq, b_eq = _flow(m)
    res = linprog(-obj, A_ub=np.array(rows) if rows else None, b_ub=np.array(rhs) if rows else None,
                  A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return None if res.status != 0 else float(obj @ res.x)

def gamma_face(m, r, crows, k, d, Vk):
    """max over competing readings of max cost on the exact optimal face of k, minus d."""
    W = -np.inf
    for kk in range(len(crows)):
        if kk == k: continue
        w = lp(m, crows[kk], [crows[k], -r], [d, -(Vk - FACE_TOL)])
        if w is None: return None
        W = max(W, w)
    return (W - d) if np.isfinite(W) else None

def regime(m, r, crows, d):
    """Instance-level regime at budget d: irrelevant / optimizer_resolvable / irreducible / infeasible."""
    V_U = lp(m, r, crows, [d] * len(crows))
    if V_U is None: return "infeasible", None, None, None
    V = [lp(m, r, [c], [d]) for c in crows]
    if any(v is None for v in V): return "infeasible", V_U, V, None
    vs = [k for k in range(len(crows)) if V[k] - V_U <= TOL]
    if not vs: return "irreducible", V_U, V, None
    G = [gamma_face(m, r, crows, k, d, V[k]) for k in vs]
    if any(g is not None and g <= 1e-6 for g in G): return "irrelevant", V_U, V, G
    return "optimizer_resolvable", V_U, V, G

def main():
    pt, _ = parse_prometheus(); tb = prom_threshold_bank(pt); rows = []
    for ti, t in enumerate(pt):
        ok, _w, rd = _eligible(build_prom_pool(t, tb))
        if not ok: continue
        uid = f'{getattr(t, "name", "?")}#{ti}'; m = compile_instance(rd)
        r = m["r"].reshape(-1); K = m["C"].shape[0]; crows = [m["C"][k].reshape(-1) for k in range(K)]
        rec = dict(uid=uid, K=K, profiles={}, band=[])
        for d in BUDGETS:
            reg, V_U, V, _G = regime(m, r, crows, d)
            if reg == "infeasible": continue
            prof = {}
            for k in range(K):
                A = []
                for kf in KAPPA_FRAC:
                    VUk = lp(m, r, crows, [d - kf * d] * K)
                    A.append(None if VUk is None else (V[k] - VUk))
                Gk = gamma_face(m, r, crows, k, d, V[k])
                prof[str(k)] = dict(V_psi=V[k], A=A, intercept=V[k] - V_U, Gamma0=Gk,
                                    regime_anchor=("irreducible" if V[k] - V_U > TOL else ("irrelevant" if (Gk is not None and Gk <= 1e-6) else "optimizer_resolvable")))
            rec["profiles"][str(d)] = dict(V_U=V_U, regime=reg, anchors=prof)
        for d in D_SWEEP:
            reg, V_U, V, _G = regime(m, r, crows, float(d))
            rec["band"].append(dict(d=float(d), regime=reg, price_min=(None if (V_U is None or V is None or any(v is None for v in V)) else float(min(V) - V_U) / V_U)))
        rows.append(rec); print(f"  {uid:38s} done", flush=True)
    # summaries
    summ = {}
    for d in BUDGETS:
        anchors = [(rec["uid"], k, p) for rec in rows if str(d) in rec["profiles"] for k, p in rec["profiles"][str(d)]["anchors"].items()]
        inst_reg = [rec["profiles"][str(d)]["regime"] for rec in rows if str(d) in rec["profiles"]]
        A0 = np.array([p["intercept"] / (rec_VU := next(r_["profiles"][str(d)]["V_U"] for r_ in rows if r_["uid"] == u)) for u, k, p in anchors])
        # median normalized profile over anchors with zero intercept (the resolvable/irrelevant ones)
        prof_zero = [np.array([np.nan if a is None else a for a in p["A"]]) / next(r_["profiles"][str(d)]["V_U"] for r_ in rows if r_["uid"] == u)
                     for u, k, p in anchors if p["intercept"] <= TOL]
        med_curve = (np.nanmedian(np.stack(prof_zero), axis=0).tolist() if prof_zero else None)
        summ[str(d)] = dict(instances={k: inst_reg.count(k) for k in ("irrelevant", "optimizer_resolvable", "irreducible")},
                            anchors={k: sum(1 for _, _, p in anchors if p["regime_anchor"] == k) for k in ("irrelevant", "optimizer_resolvable", "irreducible")},
                            intercept_over_VU=dict(zero_frac=float(np.mean(A0 <= 1e-9)), pos_med=(float(np.median(A0[A0 > 1e-9])) if (A0 > 1e-9).any() else None), pos_max=(float(A0.max()) if A0.size else None)),
                            median_profile_zero_intercept=med_curve)
    band = {}
    for d in D_SWEEP:
        regs = [next(b["regime"] for b in rec["band"] if abs(b["d"] - float(d)) < 1e-12) for rec in rows]
        band[str(float(d))] = {k: regs.count(k) for k in ("irrelevant", "optimizer_resolvable", "irreducible", "infeasible")}
    json.dump(dict(registration="V39 (REGISTRATION_V28.md)", budgets=list(BUDGETS), kappa_frac=KAPPA_FRAC.tolist(), d_sweep=D_SWEEP.tolist(),
                   summary=summ, band=band, rows=rows), open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")
    print("\nAMBIGUITY PROFILE, per budget: instance regimes | anchor regimes | intercept A(0)/V_U")
    for d in BUDGETS:
        s = summ[str(d)]; print(f"  d={d}: inst {s['instances']} | anchors {s['anchors']} | A(0)=0 on {100*s['intercept_over_VU']['zero_frac']:.0f}% of anchors; positive intercept med {s['intercept_over_VU']['pos_med']} max {s['intercept_over_VU']['pos_max']}")
        if s["median_profile_zero_intercept"]:
            mc = s["median_profile_zero_intercept"]; print("      median A(kappa)/V_U over zero-intercept anchors at kappa/d = 0,.25,.5,.75,1:", [None if (x is None or (isinstance(x,float) and np.isnan(x))) else round(x,4) for x in (mc[0], mc[5], mc[10], mc[15], mc[20])])
    print("\nBUDGET-BAND MAP (instances per regime as a function of d):")
    for d in D_SWEEP:
        b = band[str(float(d))]; print(f"  d={float(d):.4f}: irrelevant {b['irrelevant']:2d}  resolvable {b['optimizer_resolvable']:2d}  irreducible {b['irreducible']:2d}  infeasible {b['infeasible']:2d}")

if __name__ == "__main__":
    main()
