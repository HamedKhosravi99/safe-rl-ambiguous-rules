#!/usr/bin/env python3
"""Finite-data Decide on the 28 compiled monitoring rules (exact verdicts known).

Structured estimator: the compiled models share one L=3 load chain, the only stochastic component; telemetry
of n_total steps gives per-row counts n_row = n_total * pi_row (stationary proportions), the empirical 3x3
matrix is recompiled into every rule's model, and the uniform error bound is
  beta_f = gamma/(2(1-gamma)) * max_row ||Mhat_row - M_row||_1 * span(f)      (centered simulation lemma).
Certificate = conservative empirical epsilon-face (see arrow_theory_extension.tex, Section 5).
"""
import os
import sys, json, math, time
import numpy as np
from scipy.optimize import linprog
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(str(REPO), "src"))
from saorl.benchmark_sg import control_suite as cs
GAMMA = cs.GAMMA
OUT = REPO + "/results/theory_extension/"
LOG = open(OUT + "real_rules_progress.txt", "w")
def log(*a):
    print(*a, file=LOG, flush=True); print(*a, flush=True)

class _R:
    def __init__(self, theta, for_s): self.threshold, self.for_s = theta, for_s
suite = json.load(open(REPO + "/results/e2e/control_suite_uncapped.json"))
insts = suite["instances"]
assert len(insts) == 28
M_TRUE = cs._load_matrix(3)
w, v = np.linalg.eig(M_TRUE.T); pi = np.real(v[:, np.argmin(abs(w - 1))]); pi = pi / pi.sum()
log("load chain M_true:\n", np.round(M_TRUE, 4), "\nstationary pi:", np.round(pi, 4))

def compile_with(readings, M):
    orig = cs._load_matrix; cs._load_matrix = lambda L: M
    try: return cs.compile_instance(readings)
    finally: cs._load_matrix = orig
def flow(m):
    nS, nA = m["nS"], m["nA"]
    block = np.kron(np.eye(nS), np.ones((1, nA)))
    return block - GAMMA * np.transpose(m["P"], (2, 0, 1)).reshape(nS, nS * nA), (1 - GAMMA) * m["mu0"]
def lp(obj, rows, rhs, A, b):
    res = linprog(-obj, A_ub=np.array(rows) if rows else None, b_ub=np.array(rhs) if rows else None, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    return None if res.status != 0 else float(obj @ res.x)

class Rule:
    def __init__(self, inst, idx=0):
        self.uid = f'{inst["uid"]}#{idx}'; self.readings = [_R(x["theta"], x["for_s"]) for x in inst["readings"]]
        self.m = cs.compile_instance(self.readings); self.A, self.b = flow(self.m)
        self.r = self.m["r"].reshape(-1); self.c = [self.m["C"][k].reshape(-1) for k in range(self.m["C"].shape[0])]
        self.K = len(self.c)
    def V(self, k, d, A=None, b=None):
        A = self.A if A is None else A; b = self.b if b is None else b
        return lp(self.r, [self.c[k]], [d], A, b)
    def W(self, k, d, eps, A=None, b=None, Vk=None, dplus=None, thr=None):
        """max competing cost over {c_k x <= dplus, r x >= thr}; defaults to the exact eps-face."""
        A = self.A if A is None else A; b = self.b if b is None else b
        if Vk is None: Vk = self.V(k, d, A, b)
        if Vk is None: return None
        dplus = d if dplus is None else dplus; thr = (Vk - eps) if thr is None else thr
        best = -np.inf
        for j in range(self.K):
            if j == k: continue
            w = lp(self.c[j], [self.c[k], -self.r], [dplus, -thr], A, b)
            if w is not None: best = max(best, w)
        return best
    def eps_star(self, k, d, tol=1e-4):
        if self.W(k, d, 0.0) > d + 1e-9: return -1.0
        lo, hi = 0.0, 0.7
        if self.W(k, d, hi) <= d + 1e-9: return hi
        while hi - lo > tol:
            mid = (lo + hi) / 2
            if self.W(k, d, mid) <= d + 1e-9: lo = mid
            else: hi = mid
        return lo
    def dominant(self, k):
        return all(np.all(self.m["C"][k] >= self.m["C"][j]) for j in range(self.K) if j != k)

rules = [Rule(i, j) for j, i in enumerate(insts)]      # uid#index: the suite repeats one uid with an identical compiled model
log("compiled", len(rules), "rules; nS:", sorted(set(r.m["nS"] for r in rules)), "K:", sorted(set(r.K for r in rules)))

# ---------------- exact analysis ----------------
BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10]; EPSS = [0.0, 0.005, 0.01, 0.02]
exact = {}
t0 = time.time()
for R in rules:
    exact[R.uid] = {}
    for d in BUDGETS:
        rec = {}
        for k in range(R.K):
            Vk = R.V(k, d)
            if Vk is None: rec[str(k)] = None; continue
            Ws = {str(e): R.W(k, d, e, Vk=Vk) for e in EPSS}
            rec[str(k)] = {"V": Vk, "W_eps": Ws, "margin_eps": {e: d - w for e, w in Ws.items()}, "eps_star": R.eps_star(k, d), "dominant": R.dominant(k)}
        exact[R.uid][str(d)] = rec
log("exact analysis done in %.0fs" % (time.time() - t0))
# sanity against the archive at d=0.05 (exact face)
arch = json.load(open(REPO + "/results/e2e/policy_sufficiency.json"))
arch = arch if isinstance(arch, list) else arch["rows"]
mism = 0; checked = 0
for row in arch:
    d = float(row["budget"]); base = row["rule_id"].split("#")[0]
    cands = [u for u in exact if u.split("#")[0] == base]
    if not cands or str(d) not in exact[cands[0]]: continue
    uid = cands[0]
    for x in row["readings"]:
        rec = exact[uid][str(d)].get(str(x["reading"]))
        if rec is None: continue
        checked += 1
        if abs(rec["W_eps"]["0.0"] - x["W"]) > 1e-5: mism += 1
log(f"archive check: {checked} (rule,budget,reading) W values compared, {mism} mismatches > 1e-5")
# summaries per budget/eps
summ = {}
for d in BUDGETS:
    summ[str(d)] = {}
    for e in EPSS:
        n_suff_rules = sum(1 for R in rules if any((exact[R.uid][str(d)][str(k)] or {}).get("margin_eps", {}).get(str(e), -1) >= -1e-9 for k in range(R.K)))
        margins = sorted(round(exact[R.uid][str(d)][str(k)]["margin_eps"][str(e)], 4) for R in rules for k in range(R.K) if exact[R.uid][str(d)][str(k)] and exact[R.uid][str(d)][str(k)]["margin_eps"][str(e)] >= -1e-9)
        summ[str(d)][str(e)] = {"rules_with_sufficient_reading": n_suff_rules, "sufficient_readings": len(margins), "margins": margins}
    es = sorted(round(exact[R.uid][str(d)][str(k)]["eps_star"], 4) for R in rules for k in range(R.K) if exact[R.uid][str(d)][str(k)])
    summ[str(d)]["eps_star_all_readings"] = es
    log(f"d={d}: rules with an eps-sufficient reading: " + ", ".join(f"eps={e}: {summ[str(d)][str(e)]['rules_with_sufficient_reading']}/28" for e in EPSS) + f"; eps* quantiles (readings): {np.percentile([x for x in es if x >= 0], [0, 25, 50, 75, 100]).round(4).tolist() if any(x >= 0 for x in es) else 'none'}")
json.dump({"exact": exact, "summary": summ, "M_true": M_TRUE.tolist(), "pi": pi.tolist()}, open(OUT + "real_rules_exact.json", "w"), indent=1)

# ---------------- finite-data sweep ----------------
LSIM = GAMMA / (2 * (1 - GAMMA)); SPAN_R, SPAN_C = 0.7, 1.0; DELTA = 0.05; EPS = 0.01
def radius(Mhat, nrow, bound):
    out = []
    for i in range(3):
        n = nrow[i]
        if bound == "weissman":
            out.append(math.sqrt(2 * (3 * math.log(2) + math.log(3 / DELTA)) / n))
        else:
            Lg = math.log(2 * 3 * 3 / DELTA); p = Mhat[i]
            out.append(min(2.0, float(np.sum(np.sqrt(2 * p * (1 - p) * n / (n - 1) * Lg / n) + 7 * Lg / (3 * (n - 1))))))
    return max(out)
def certificate(R, mhat, k, d, eps, beta_r, beta_c):
    A, b = flow(mhat)
    Vm = R.V(k, d - beta_c, A, b)
    if Vm is None: return False, None
    Vm -= beta_r
    w = R.W(k, d, eps, A, b, Vk=Vm, dplus=d + beta_c, thr=Vm - beta_r - eps)
    if w is None: return False, None
    return (w + beta_c <= d + 1e-9), w + beta_c
N_GRID = [10 ** (x / 2) for x in range(8, 21)]          # 1e4 ... 1e10
REPS = 10; SWEEP_BUDGETS = [0.02, 0.05, 0.10]
rng = np.random.default_rng(20260904)
records = []
t0 = time.time()
for n_total in N_GRID:
    nrow = [max(2, int(round(n_total * pi[i]))) for i in range(3)]
    for rep in range(REPS):
        Mhat = np.array([rng.multinomial(nrow[i], M_TRUE[i]) / nrow[i] for i in range(3)])
        rad = {bnd: radius(Mhat, nrow, bnd) for bnd in ("weissman", "bernstein")}
        for R in rules:
            mhat = compile_with(R.readings, Mhat)
            for d in SWEEP_BUDGETS:
                for bnd in (("weissman", "bernstein") if d == 0.05 else ("bernstein",)):
                    beta = LSIM * rad[bnd]
                    for k in range(R.K):
                        ex = exact[R.uid][str(d)][str(k)]
                        truth = ex is not None and ex["margin_eps"][str(EPS)] >= -1e-9
                        ok, What = certificate(R, mhat, k, d, EPS, beta * SPAN_R, beta * SPAN_C)
                        records.append({"n": n_total, "rep": rep, "uid": R.uid, "k": k, "d": d, "bound": bnd, "beta": beta, "truth": bool(truth),
                                        "margin": None if ex is None else ex["margin_eps"][str(EPS)], "pass": bool(ok), "What": What})
    log(f"n_total={n_total:.3g}: done ({time.time()-t0:.0f}s)")
json.dump(records, open(OUT + "real_rules_finite_data_records.json", "w"))
# summaries
import collections
summary = {}
for d in SWEEP_BUDGETS:
    for bnd in ("weissman", "bernstein"):
        key = f"d={d}|{bnd}"; rows = [x for x in records if x["d"] == d and x["bound"] == bnd]
        if not rows: continue
        per_n = {}
        for n_total in N_GRID:
            rr = [x for x in rows if x["n"] == n_total]
            suff = [x for x in rr if x["truth"]]; nons = [x for x in rr if not x["truth"]]
            per_n[f"{n_total:.3g}"] = {"beta": rr[0]["beta"], "certified_among_sufficient": (sum(x["pass"] for x in suff), len(suff)), "false_certificates": (sum(x["pass"] for x in nons), len(nons))}
        # n* per sufficient reading and n*gamma^2
        nstar = {}
        for (uid, k) in sorted(set((x["uid"], x["k"]) for x in rows if x["truth"])):
            for n_total in N_GRID:
                rr = [x for x in rows if x["uid"] == uid and x["k"] == k and x["n"] == n_total]
                if rr and np.mean([x["pass"] for x in rr]) >= 0.9:
                    nstar[f"{uid}#{k}"] = {"n_star": n_total, "margin": rr[0]["margin"], "n_star_margin2": n_total * rr[0]["margin"] ** 2}; break
        summary[key] = {"per_n": per_n, "n_star": nstar}
        log(f"[{key}] " + "; ".join(f"n={n}: suff {v['certified_among_sufficient'][0]}/{v['certified_among_sufficient'][1]}, false {v['false_certificates'][0]}/{v['false_certificates'][1]}" for n, v in per_n.items() if int(float(n)) in (10**5, 10**6, 10**7, 10**8, 10**9, 10**10)))
        if nstar:
            g = np.array([v["margin"] for v in nstar.values()]); ns = np.array([v["n_star"] for v in nstar.values()])
            slope = np.polyfit(np.log(g), np.log(ns), 1)[0] if len(set(g.round(6))) > 1 else float("nan")
            log(f"   n* found for {len(nstar)} readings; log n* vs log margin slope = {slope:.2f}; n*·margin² range [{(ns*g**2).min():.2f}, {(ns*g**2).max():.2f}]")
json.dump(summary, open(OUT + "real_rules_finite_data.json", "w"), indent=1)
log("total time %.0fs" % (time.time() - t0))
