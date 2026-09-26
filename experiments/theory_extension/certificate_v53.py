#!/usr/bin/env python3
"""V53: decision-specific offline certification on the 28 compiled monitoring rules (REGISTRATION_V53.md).

Three certificates on identical chain draws -- UNIFORM (archived, simulation-lemma slack), OCC (row-specific,
occupancy-weighted) and DUAL (robust shaped-dominance witness; robust-dual ARROW) -- plus DUAL(eta) budget tightening,
the witness divergence Ibar, an upper bound on the decision-information radius I* obtained from a decision-reversing
chain whose margin is at least WITNESS_MARGIN (re-verified by verify_witnesses), and a path-sampled secondary.  Every certificate is a deterministic
function of (Mhat, N, delta); soundness holds on the event E = { ||M_z - Mhat_z||_1 <= alpha_z for all rows }.
Writes results/theory_extension/certificate_v53.json.
"""
import sys, json, math, time, itertools, os
import numpy as np
from scipy.optimize import linprog, minimize
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from saorl.benchmark_sg import control_suite as cs
GAMMA = cs.GAMMA
OUT = os.path.join(REPO, "results", "theory_extension", "certificate_v53.json")
LOG_PATH = os.path.join(REPO, "results", "theory_extension", "certificate_v53_progress.txt"); LOG = None
def log(*a):
    if LOG is not None: print(*a, file=LOG, flush=True)
    print(*a, flush=True)

# ---------------------------------------------------------------- frozen design (REGISTRATION_V53)
EPS, DELTA = 0.01, 0.05
BUDGETS = [0.02, 0.05, 0.10]
N_GRID = [10 ** (x / 2) for x in range(6, 17)]                 # 1e3 ... 1e8
REPS = 10
HARD_EXTRA_REPS, HARD_EXTRA_N, HARD_D = 50, [1e5, 10 ** 5.5, 1e6], 0.05
ETAS = [0.001, 0.002, 0.005, 0.01, 0.02]
SEC_N, SEC_PATHS = [1e4, 1e5, 1e6, 1e7], 5
SEED_PRIMARY, SEED_EXTRA, SEED_SECONDARY = 20260913, 20260914, 20260915
WITNESS_MARGIN, WITNESS_SLACK = 1e-6, 1e-9   # required margin Gamma > 0 of the decision-reversing witness chain, and the tolerance of its re-verification
LSIM = GAMMA / (2 * (1 - GAMMA)); SPAN_R, SPAN_C = 0.7, 1.0

class _R:
    def __init__(self, theta, for_s): self.threshold, self.for_s = theta, for_s
suite = json.load(open(os.path.join(REPO, "results/e2e", "control_suite_uncapped.json")))
exact = json.load(open(os.path.join(REPO, "results", "theory_extension", "real_rules_exact.json")))
M_TRUE = np.array(exact["M_true"]); PI = np.array(exact["pi"]); R_ROWS = M_TRUE.shape[0]
rules = [(f'{inst["uid"]}#{j}', [_R(x["theta"], x["for_s"]) for x in inst["readings"]]) for j, inst in enumerate(suite["instances"])]
assert len(rules) == 28

def compile_with(readings, M):
    orig = cs._load_matrix; cs._load_matrix = lambda L: M
    try: return cs.compile_instance(readings)
    finally: cs._load_matrix = orig
def flow(m):
    nS, nA = m["nS"], m["nA"]
    block = np.kron(np.eye(nS), np.ones((1, nA)))
    return block - GAMMA * np.transpose(m["P"], (2, 0, 1)).reshape(nS, nS * nA), (1 - GAMMA) * m["mu0"]
def lp_max(obj, rows, rhs, A, b):
    res = linprog(-obj, A_ub=np.array(rows) if rows else None, b_ub=np.array(rhs) if rows else None, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    return None if res.status != 0 else float(obj @ res.x)
def structure(m):
    """stochastic rows from the deterministic monitor update: state -> (load z, successor indices ordered by z')."""
    S = m["S"]; idx = {s: i for i, s in enumerate(S)}
    thetas, caps = m["thetas"], m["caps"]; uniq = sorted(set(thetas)); lvl_of = [1 + uniq.index(t) for t in thetas]; K = len(thetas)
    succ = {}
    for s in S:
        if s[0] == "safe": continue
        load, cvec = s[0], s[1:]
        succ[idx[s]] = (load, [idx[(z2,) + tuple(min(caps[q], cvec[q] + 1) if z2 >= lvl_of[q] else 0 for q in range(K))] for z2 in range(R_ROWS)])
    return idx[("safe",)], succ
def alpha_rows(N):
    """Weissman radius per row, confidence delta allocated once across the R rows: alpha_z = sqrt(2(m_z ln2 + ln(R/delta))/N_z)."""
    return np.array([min(2.0, math.sqrt(2 * (R_ROWS * math.log(2) + math.log(R_ROWS / DELTA)) / N[z])) if N[z] >= 1 else 2.0 for z in range(R_ROWS)])

class Compiled:
    def __init__(self, rd, Mhat):
        self.m = compile_with(rd, Mhat); self.A, self.b = flow(self.m); self.safe, self.succ = structure(self.m)
        self.r = self.m["r"].reshape(-1); self.c = [self.m["C"][q].reshape(-1) for q in range(2)]; self.Mhat = Mhat
    def bvec(self, alpha):
        bv = np.zeros((self.m["nS"], self.m["nA"]))
        for i, (z, _) in self.succ.items(): bv[i, 0] = alpha[z]
        return bv.reshape(-1)

# ---------------------------------------------------------------- certificates
def cert_uniform(C, k, d, alpha):
    beta = LSIM * alpha.max(); bc, br = beta * SPAN_C, beta * SPAN_R
    Vm = lp_max(C.r, [C.c[k]], [d - bc], C.A, C.b)
    if Vm is None: return False, None
    Vm -= br
    w = lp_max(C.c[1 - k], [C.c[k], -C.r], [d + bc, -(Vm - br - EPS)], C.A, C.b)
    if w is None: return False, None
    return bool(w + bc <= d + 1e-9), float(w + bc - d)
def occ_vlow(C, k, d, alpha, eta=0.0):
    bv = C.bvec(alpha); wr, wc = LSIM * SPAN_R * bv, LSIM * SPAN_C * bv
    return lp_max(C.r - wr, [C.c[k] + wc], [d - eta], C.A, C.b), wr, wc
def cert_occ(C, k, d, alpha):
    Vlow, wr, wc = occ_vlow(C, k, d, alpha)
    if Vlow is None: return False, None
    u = lp_max(C.c[1 - k] + wc, [C.c[k] - wc, -(C.r + wr)], [d, -(Vlow - EPS)], C.A, C.b)
    if u is None: return False, None
    return bool(u <= d + 1e-9), float(u - d)
def robust_dual(C, k, j, alpha, Vlow, d_anchor):
    m = C.m; nS, nA = m["nS"], m["nA"]; g = GAMMA; r, ck, cj = m["r"], m["C"][k], m["C"][j]; safe, succ = C.safe, C.succ
    stoch = sorted(succ); tpos = {i: nS + 2 + q for q, i in enumerate(stoch)}; nvar = nS + 2 + len(stoch)
    rows, rhs = [], []
    def base(i, a):
        row = np.zeros(nvar); row[i] -= 1.0; row[nS] -= ck[i, a]; row[nS + 1] += r[i, a]; return row
    for a in range(nA):
        row = base(safe, a); row[safe] += g; rows.append(row); rhs.append(-cj[safe, a])
    for i in stoch:
        z, order = succ[i]
        row = base(i, 0)
        for zp, jdx in enumerate(order): row[jdx] += g * C.Mhat[z, zp]
        row[tpos[i]] += g * alpha[z] / 2.0; rows.append(row); rhs.append(-cj[i, 0])
        row = base(i, 1); row[safe] += g; rows.append(row); rhs.append(-cj[i, 1])
        for a_, b_ in itertools.permutations(order, 2):
            row = np.zeros(nvar); row[a_] += 1.0; row[b_] -= 1.0; row[tpos[i]] -= 1.0; rows.append(row); rhs.append(0.0)
    obj = np.zeros(nvar); obj[:nS] = (1 - g) * m["mu0"]; obj[nS] = d_anchor; obj[nS + 1] = -(Vlow - EPS)
    res = linprog(obj, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=[(None, None)] * nS + [(0, None)] * (2 + len(stoch)), method="highs")
    return float(res.fun) if res.status == 0 else np.inf
def cert_dual(C, k, d, alpha, eta=0.0):
    Vlow, _, _ = occ_vlow(C, k, d, alpha, eta)
    if Vlow is None: Vlow = 0.0                                   # r >= 0, so 0 is a valid lower bound on V_psi
    W = robust_dual(C, k, 1 - k, alpha, Vlow, d - eta)
    return bool(W <= d + 1e-9), float(W - d)

# ---------------------------------------------------------------- witness divergence Ibar, an upper bound on the decision-information radius I*
def margin_under(rd, Q, k, d):
    C = Compiled(rd, Q); V = lp_max(C.r, [C.c[k]], [d], C.A, C.b)
    if V is None: return None
    W = lp_max(C.c[1 - k], [C.c[k], -C.r], [d, -(V - EPS)], C.A, C.b)
    return None if W is None else d - W
def wkl(P, Q, w): return float(sum(w[z] * sum(P[z, i] * math.log(P[z, i] / max(Q[z, i], 1e-300)) for i in range(R_ROWS) if P[z, i] > 0) for z in range(R_ROWS)))
def istar(rd, k, d):
    P, w = M_TRUE, PI; best = (np.inf, None, None)
    for z in range(R_ROWS):
        for i, j in itertools.permutations(range(R_ROWS), 2):
            def Q_of(t): Q = P.copy(); Q[z, i] -= t; Q[z, j] += t; return Q
            lo, hi = 0.0, P[z, i]; mhi = margin_under(rd, Q_of(hi * 0.999), k, d)
            if mhi is None or mhi >= 0: continue
            for _ in range(30):
                mid = (lo + hi) / 2
                (lo, hi) = (mid, hi) if (margin_under(rd, Q_of(mid), k, d) or -1) >= 0 else (lo, mid)
            if -mhi < WITNESS_MARGIN: continue                    # the far end of this direction does not reach the required margin
            lo2, hi2 = hi, P[z, i] * 0.999                       # smallest shift with Gamma >= WITNESS_MARGIN; Gamma(hi) > 0 is only tiny
            for _ in range(40):
                mid = (lo2 + hi2) / 2; g = margin_under(rd, Q_of(mid), k, d)
                (lo2, hi2) = (mid, hi2) if (g is None or -g < WITNESS_MARGIN) else (lo2, mid)
            Q = Q_of(hi2); I = wkl(P, Q, w)
            if I < best[0]: best = (I, Q, f"row {z}: {i}->{j}, t={hi2:.5f}")
    if best[1] is None: return dict(I=np.inf, n_info=np.inf, l1_rows=[0.0] * R_ROWS, Q=None, note="no flipping chain found in the search")
    def unpack(th): Qm = np.exp(th.reshape(R_ROWS, R_ROWS)); return Qm / Qm.sum(1, keepdims=True)
    cons = {"type": "ineq", "fun": lambda th: -((margin_under(rd, unpack(th), k, d)) if margin_under(rd, unpack(th), k, d) is not None else -1.0) - WITNESS_MARGIN}
    res = minimize(lambda th: wkl(P, unpack(th), w), np.log(np.maximum(best[1], 1e-9)).reshape(-1), method="SLSQP", constraints=[cons], options={"maxiter": 60, "ftol": 1e-12})
    Qr = unpack(res.x); mr = margin_under(rd, Qr, k, d); Ir = wkl(P, Qr, w) if (mr is not None and -mr >= WITNESS_MARGIN - WITNESS_SLACK) else np.inf
    I, Q = (Ir, Qr) if Ir < best[0] else (best[0], best[1])
    kl_delta = (1 - DELTA) * math.log((1 - DELTA) / DELTA) + DELTA * math.log(DELTA / (1 - DELTA))
    return dict(I=I, n_info=kl_delta / I, n_first_order=math.log(1 / DELTA) / I, l1_rows=[float(np.abs(Q[z] - P[z]).sum()) for z in range(R_ROWS)], Q=Q.tolist(), margin=float(-margin_under(rd, Q, k, d)), witness_margin_target=WITNESS_MARGIN, start=best[2])

# ---------------------------------------------------------------- truth and classes
truth = {}
for uid, rd in rules:
    for k in range(2):
        for d in BUDGETS:
            rec = exact["exact"][uid][str(d)][str(k)]
            truth[(uid, k, d)] = None if rec is None else dict(sufficient=bool(rec["margin_eps"][str(EPS)] >= -1e-9), margin=rec["margin_eps"][str(EPS)], V=rec["V"])
classes = {}
for d in BUDGETS:
    cl = {}
    for uid, rd in rules:
        for k in range(2):
            t = truth[(uid, k, d)]
            if t and t["sufficient"]: cl.setdefault(round(t["margin"], 4), []).append(f"{uid}|{k}")
    classes[str(d)] = {f"kappa={kp:.4f}": members for kp, members in sorted(cl.items())}
    log(f"d={d}: sufficient-reading classes by margin: " + ", ".join(f"kappa={kp:.4f} x{len(v)}" for kp, v in sorted(cl.items())))
hard_members = set(classes[str(HARD_D)][min(classes[str(HARD_D)], key=lambda s: float(s.split('=')[1]))])
log("hard class at d=%.2f: %d readings" % (HARD_D, len(hard_members)))

def verify_witnesses(results):
    """Re-solve both linear programs at every reported witness chain with two HiGHS algorithms at feasibility tolerance
    1e-10 and require Gamma >= WITNESS_MARGIN - WITNESS_SLACK; the primal residuals of the solves are recorded too.
    A failure raises, so no reported floor rests on a witness whose membership in B_psi is in doubt."""
    global lp_max
    settings = {"ds_1e-10": ("highs-ds", {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}),
                "ipm_1e-10": ("highs-ipm", {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10, "ipm_optimality_tolerance": 1e-12})}
    base = lp_max
    for d, classes_d in results["istar"].items():
        for cname, rec in classes_d.items():
            if rec.get("Q") is None: continue
            uid, k = rec["representative"].split("|"); k = int(k); rd = dict(rules)[uid]; Q = np.array(rec["Q"]); ver = {}
            for name, (method, opts) in settings.items():
                resid = {"eq": 0.0, "ub": 0.0}
                def lp_chk(obj, rows, rhs, A, b):
                    r = linprog(-obj, A_ub=np.array(rows) if rows else None, b_ub=np.array(rhs) if rows else None, A_eq=A, b_eq=b, bounds=(0, None), method=method, options=opts)
                    if r.status != 0: return None
                    resid["eq"] = max(resid["eq"], float(np.abs(A @ r.x - b).max()))
                    if rows: resid["ub"] = max(resid["ub"], float((np.array(rows) @ r.x - np.array(rhs)).max()))
                    return float(obj @ r.x)
                lp_max = lp_chk
                try: m = margin_under(rd, Q, k, float(d))
                finally: lp_max = base
                gamma = None if m is None else -m
                ver[name] = dict(margin=gamma, max_eq_residual=resid["eq"], max_ub_violation=resid["ub"])
                assert gamma is not None and gamma >= WITNESS_MARGIN - WITNESS_SLACK, f"witness for d={d} {cname} fails verification under {name}: Gamma={gamma}"
            rec["verification"] = ver
            log(f"verify d={d} {cname}: " + ", ".join(f"{n}: Gamma={v['margin']:.3e}, eq-resid={v['max_eq_residual']:.1e}, ub-viol={v['max_ub_violation']:.1e}" for n, v in ver.items()))

def draw_counts(rng, n):
    N = [max(2, int(round(n * PI[z]))) for z in range(R_ROWS)]
    counts = np.array([rng.multinomial(N[z], M_TRUE[z]) for z in range(R_ROWS)])
    return N, counts / np.array(N)[:, None]

def evaluate(rd_map, Mhat, N, d_list, reading_filter=None, with_tightening=False):
    alpha = alpha_rows(N); out = []
    for uid, rd in rules:
        C = Compiled(rd, Mhat)
        for k in range(2):
            if reading_filter and f"{uid}|{k}" not in reading_filter: continue
            for d in d_list:
                t = truth[(uid, k, d)]
                if t is None: continue
                rec = dict(uid=uid, k=k, d=d, truth=t["sufficient"])
                for name, fn in (("UNIFORM", cert_uniform), ("OCC", cert_occ), ("DUAL", cert_dual)):
                    ok, slack = fn(C, k, d, alpha); rec[name] = {"pass": ok, "slack": slack}
                if with_tightening and d == HARD_D and t["sufficient"] and not rec["DUAL"]["pass"]:
                    eta_hat = None
                    for eta in ETAS:
                        ok, _ = cert_dual(C, k, d, alpha, eta)
                        if ok: eta_hat = eta; break
                    rec["eta_hat"] = eta_hat
                out.append(rec)
    return out

def main():
    global LOG; LOG = open(LOG_PATH, "w")
    t0 = time.time(); results = dict(meta=dict(witness_margin=WITNESS_MARGIN, eps=EPS, delta=DELTA, budgets=BUDGETS, n_grid=N_GRID, reps=REPS, hard_extra=dict(reps=HARD_EXTRA_REPS, n=HARD_EXTRA_N, d=HARD_D),
                                                  etas=ETAS, secondary=dict(n=SEC_N, paths=SEC_PATHS), seeds=dict(primary=SEED_PRIMARY, extra=SEED_EXTRA, secondary=SEED_SECONDARY),
                                                  M_true=M_TRUE.tolist(), pi=PI.tolist(), registration="results/e2e/REGISTRATION_V53.md"),
                                        truth={f"{u}|{k}|{d}": v for (u, k, d), v in truth.items() if v}, classes=classes, primary=[], extra_hard=[], secondary=[], istar={}, price={})
    # ---- primary
    rng = np.random.default_rng(SEED_PRIMARY)
    for n in N_GRID:
        for rep in range(REPS):
            N, Mhat = draw_counts(rng, n)
            for rec in evaluate(rules, Mhat, N, BUDGETS, with_tightening=True):
                rec.update(n=n, rep=rep, N=N); results["primary"].append(rec)
        rows = [x for x in results["primary"] if x["n"] == n and x["d"] == HARD_D]
        log(f"primary n={n:.3g} (d={HARD_D}): " + " | ".join(f"{c}: {sum(x[c]['pass'] for x in rows if x['truth'])}/{sum(1 for x in rows if x['truth'])} suff, {sum(x[c]['pass'] for x in rows if not x['truth'])} false" for c in ("UNIFORM", "OCC", "DUAL")) + f"  ({time.time()-t0:.0f}s)")
    # ---- extra independent draws for the hard class
    rng = np.random.default_rng(SEED_EXTRA)
    for n in HARD_EXTRA_N:
        for rep in range(REPS, REPS + HARD_EXTRA_REPS):
            N, Mhat = draw_counts(rng, n)
            for rec in evaluate(rules, Mhat, N, [HARD_D], reading_filter=hard_members, with_tightening=True):
                rec.update(n=n, rep=rep, N=N); results["extra_hard"].append(rec)
        rows = [x for x in results["extra_hard"] if x["n"] == n]
        log(f"extra hard draws n={n:.3g}: " + " | ".join(f"{c}: {sum(x[c]['pass'] for x in rows)}/{len(rows)}" for c in ("UNIFORM", "OCC", "DUAL")) + f"  ({time.time()-t0:.0f}s)")
    # ---- exact price of tightening, per sufficient reading at d = HARD_D
    for uid, rd in rules:
        C = Compiled(rd, M_TRUE)
        for k in range(2):
            t = truth[(uid, k, HARD_D)]
            if not (t and t["sufficient"]): continue
            V0 = lp_max(C.r, [C.c[k]], [HARD_D], C.A, C.b)
            results["price"][f"{uid}|{k}"] = {str(eta): (V0 - lp_max(C.r, [C.c[k]], [HARD_D - eta], C.A, C.b)) / V0 for eta in ETAS}
    # ---- witness divergence Ibar (upper bound on I*) per class and budget
    for d in BUDGETS:
        results["istar"][str(d)] = {}
        for cname, members in classes[str(d)].items():
            uid, k = members[0].split("|"); rd = dict(rules)[uid]
            results["istar"][str(d)][cname] = dict(istar(rd, int(k), d), representative=members[0], size=len(members))
            r = results["istar"][str(d)][cname]; log(f"Ibar d={d} {cname} (x{len(members)}): Ibar={r['I']:.3e} margin={r.get('margin', float('nan')):.2e} n_info=kl(1-delta,delta)/Ibar={r['n_info']:.3g}, ln(1/delta)/Ibar={r.get('n_first_order', float('nan')):.3g}; L1 moves {np.round(r['l1_rows'], 4).tolist()}")
    verify_witnesses(results)
    log(f"witness search and verification done ({time.time()-t0:.0f}s)")
    # ---- secondary: counts from one sampled path of the load chain
    rng = np.random.default_rng(SEED_SECONDARY)
    cum = np.cumsum(M_TRUE, axis=1)
    for n in SEC_N:
        n_int = int(n)
        for p in range(SEC_PATHS):
            z = rng.choice(R_ROWS, p=PI); u = rng.random(n_int); counts = np.zeros((R_ROWS, R_ROWS), int)
            for tstep in range(n_int):
                z2 = int(np.searchsorted(cum[z], u[tstep])); z2 = min(z2, R_ROWS - 1); counts[z, z2] += 1; z = z2
            N = counts.sum(1); Mhat = np.where(N[:, None] > 0, counts / np.maximum(N[:, None], 1), 1.0 / R_ROWS)
            alpha = alpha_rows(N)
            for uid, rd in rules:
                C = Compiled(rd, Mhat)
                for k in range(2):
                    t = truth[(uid, k, HARD_D)]
                    if t is None: continue
                    rec = dict(n=n, path=p, uid=uid, k=k, d=HARD_D, truth=t["sufficient"], N=N.tolist())
                    for name, fn in (("UNIFORM", cert_uniform), ("DUAL", cert_dual)):
                        ok, slack = fn(C, k, HARD_D, alpha); rec[name] = {"pass": ok, "slack": slack}
                    results["secondary"].append(rec)
        rows = [x for x in results["secondary"] if x["n"] == n]
        log(f"secondary (path-sampled) n={n:.3g}: " + " | ".join(f"{c}: {sum(x[c]['pass'] for x in rows if x['truth'])}/{sum(1 for x in rows if x['truth'])} suff, {sum(x[c]['pass'] for x in rows if not x['truth'])} false" for c in ("UNIFORM", "DUAL")) + f"  ({time.time()-t0:.0f}s)")
    json.dump(results, open(OUT, "w"))
    log(f"wrote {OUT} ({time.time()-t0:.0f}s)")

def refresh_istar():
    """Recompute only the witness section of the existing results file (same classes and truth), verify it, write back."""
    global LOG; LOG = open(LOG_PATH, "a"); t0 = time.time()
    results = json.load(open(OUT)); results["meta"]["witness_margin"] = WITNESS_MARGIN
    log(f"--- witness refresh {time.strftime('%Y-%m-%d %H:%M')}: required margin {WITNESS_MARGIN:g}")
    old = {(d, c): v["n_info"] for d, cl in results["istar"].items() for c, v in cl.items()}
    results["istar"] = {}
    for d in BUDGETS:
        results["istar"][str(d)] = {}
        for cname, members in classes[str(d)].items():
            uid, k = members[0].split("|"); rd = dict(rules)[uid]
            results["istar"][str(d)][cname] = dict(istar(rd, int(k), d), representative=members[0], size=len(members))
            r = results["istar"][str(d)][cname]
            log(f"Ibar d={d} {cname} (x{len(members)}): Ibar={r['I']:.4e} margin={r.get('margin', float('nan')):.2e} n_info={r['n_info']:.6g} (was {old.get((str(d), cname), float('nan')):.6g})")
    verify_witnesses(results)
    json.dump(results, open(OUT, "w")); log(f"wrote {OUT} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    refresh_istar() if "--istar-only" in sys.argv else main()
