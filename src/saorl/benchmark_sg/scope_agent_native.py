"""V51: native multi-constraint learners on the scope-ambiguous agent domain.
K separate constraints (vector multiplier) versus the ARROW-certified singleton, same code,
same log, same iteration budget; learner-aware tolerance per REGISTRATION_V51.md.
Run:  OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m saorl.benchmark_sg.scope_agent_native [--smoke]
Writes results/e2e/scope_agent_native.json
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import hashlib, json, sys, time
from multiprocessing import Pool
import numpy as np

from saorl import offline_rl as orl
from saorl.benchmark_sg.scope_agent import GAMMA, AUX_DEMAND, K_GRID, D_GRID, EPS, build, lp
from saorl.benchmark_sg import scope_agent_learn as V50           # adapter patches applied on import
from saorl.benchmark_sg.scope_agent_learn import _set_actions, _CTX, Reading, behaviour_policy, sample_dataset, policy_table, BCQ_TAU
from saorl.benchmark_sg.safe_face_offline import Log, occupancy, score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "results/e2e"); OUT = os.path.join(R, "scope_agent_native.json")
N_GRID = (2000, 5000, 20000, 50000); EVAL_SEEDS = tuple(range(10)); CAL_SEEDS = tuple(range(100, 105))
LEARNERS = {"mfqi": dict(kp=0.0, ki=60.0, kd=0.0), "mpid": dict(kp=80.0, ki=60.0, kd=40.0), "mcpq": None, "lp": None}
DLIM_GRID = (50.0, 30.0, 20.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.2, 0.8, 0.5, 0.3, 0.15, 0.05, 0.0)
N_DUAL, N_ITER, TOL = 24, 80, 1e-9
SURR_N = (20000, 50000)

def fit_q_shaped(b, allowed, shaped, gamma, n_iter):
    NA = orl.NA; q = np.where(allowed, 0.0, -1e9); terminal = b.sp < 0; sa_flat = b.s * NA + b.a
    sum_buf = np.zeros(b.n_states * NA); cnt_buf = np.zeros(b.n_states * NA); np.add.at(cnt_buf, sa_flat, 1.0); nz = cnt_buf > 0
    q_masked = np.where(allowed, q, -np.inf)
    for _ in range(n_iter):
        nxt = q_masked.max(axis=1); nxt = np.where(np.isfinite(nxt), nxt, 0.0)
        boot = np.where(terminal, 0.0, gamma * nxt[np.where(terminal, 0, b.sp)]); y = shaped + boot
        sum_buf[:] = 0.0; np.add.at(sum_buf, sa_flat, y); flat = q.reshape(-1); flat[nz] = sum_buf[nz] / cnt_buf[nz]
        q = flat.reshape(b.n_states, NA); q_masked = np.where(allowed, q, -np.inf)
    return q

def learn_vector(b, allowed, charged, lg, m, d, gains, cost_of):
    """One learner for every arm: shaped reward r - charged @ lambda, one multiplier per column of
    `charged` (n x J), each updated by PID/dual ascent on its own constraint error from the on-policy
    P-hat estimate `cost_of(pi) -> (J,)`.  Selection: best offline return among offline-feasible
    iterates; fallback: smallest maximal honoured cost."""
    J = charged.shape[1]; lam = np.zeros(J); integral = np.zeros(J); prev = np.zeros(J)
    best = None; fallback = None; first_feas = None; n_feas = 0; lam_path = []
    for step in range(N_DUAL):
        q = fit_q_shaped(b, allowed, b.r - charged @ lam, GAMMA, N_ITER)
        pol = orl._greedy_policy(b, q, allowed); pi = policy_table(m, pol)
        jr = lg.est(pi)[0]; jc = cost_of(pi); feas = bool(np.all(jc <= d + TOL)); lam_path.append(lam.copy())
        if feas:
            n_feas += 1
            if first_feas is None: first_feas = step
            if best is None or jr > best[0]: best = (jr, pi, lam.copy())
        if fallback is None or jc.max() < fallback[0]: fallback = (jc.max(), pi, lam.copy())
        e = jc - d; integral = np.maximum(0.0, integral + e); deriv = e - prev; prev = e
        lam = np.maximum(0.0, gains["kp"] * e + gains["ki"] * integral + gains["kd"] * deriv)
    pi, lam_sel = (best[1], best[2]) if best is not None else (fallback[1], fallback[2])
    lp_ = np.array(lam_path)
    return pi, dict(offline_feasible=best is not None, first_feasible_step=first_feas, n_feasible=n_feas,
                    lam_selected=lam_sel.tolist(), lam_path_std=lp_.std(axis=0).tolist(), lam_final=lp_[-1].tolist())

def fit_cost_q(b, safe, qr, cost_step, gamma, n_iter):
    NA = orl.NA; qr_safe = np.where(safe, qr, -np.inf); nxt_a = qr_safe.argmax(axis=1)
    terminal = b.sp < 0; ns = np.where(terminal, 0, b.sp); sa_flat = b.s * NA + b.a
    cnt = np.zeros(b.n_states * NA); np.add.at(cnt, sa_flat, 1.0); nz = cnt > 0
    qc = np.zeros((b.n_states, NA)); sum_buf = np.zeros(b.n_states * NA)
    for _ in range(n_iter):
        boot = np.where(terminal, 0.0, gamma * qc[ns, nxt_a[ns]]); y = cost_step + boot
        sum_buf[:] = 0.0; np.add.at(sum_buf, sa_flat, y); flat = qc.reshape(-1); flat[nz] = sum_buf[nz] / cnt[nz]
        qc = flat.reshape(b.n_states, NA)
    return qc

def learn_vector_cpq(b, support, charged, lg, m, d, cost_of):
    """Vector CPQ: one cost critic per honoured reading; an in-support action is admissible iff every
    critic's cost-to-go is at most d_lim.  Loose-to-tight d_lim grid; selection as the paper's CPQ:
    loosest offline-feasible d_lim, else the best-safety iterate.  Singleton arm = one critic."""
    J = charged.shape[1]; best = None; fallback = None; first_feas = None; n_feas = 0
    for step, dlim in enumerate(DLIM_GRID):
        safe = support.copy()
        for _ in range(8):
            qr = fit_q_shaped(b, safe, b.r, GAMMA, N_ITER)
            qcs = [fit_cost_q(b, safe, qr, charged[:, j], GAMMA, N_ITER) for j in range(J)]
            new_safe = support.copy()
            for qc in qcs: new_safe &= (qc <= dlim)
            stranded = support.any(axis=1) & ~new_safe.any(axis=1)
            if stranded.any():
                qmax = np.max(np.stack(qcs), axis=0); qsup = np.where(support, qmax, np.inf); mincost = qsup.argmin(axis=1)
                new_safe[stranded, mincost[stranded]] = True
            if np.array_equal(new_safe, safe): break
            safe = new_safe
        qr = fit_q_shaped(b, safe, b.r, GAMMA, N_ITER); pol = orl._greedy_policy(b, qr, safe); pi = policy_table(m, pol)
        jr = lg.est(pi)[0]; jc = cost_of(pi); feas = bool(np.all(jc <= d + TOL))
        if feas:
            n_feas += 1
            if first_feas is None: first_feas = step
            if best is None: best = (jr, pi, dlim)                 # loosest feasible (grid is loose -> tight)
        if fallback is None or jc.max() < fallback[0]: fallback = (jc.max(), pi, dlim)
        if best is not None: break
    pi, sel = (best[1], best[2]) if best is not None else (fallback[1], fallback[2])
    return pi, dict(offline_feasible=best is not None, first_feasible_step=first_feas, n_feasible=n_feas, lam_selected=[sel], lam_path_std=[], lam_final=[sel])

def learn_lp(lg, m, d, cost_rows):
    """Occupancy LP on P-hat restricted to behaviour-supported pairs (the certified-offline planner),
    with one row per honoured reading (`cost_rows`: list of nS x nA cost arrays)."""
    from scipy.optimize import linprog
    nS, nA = lg.nS, lg.nA
    block = np.kron(np.eye(nS), np.ones((1, nA))); A_eq = block - GAMMA * np.transpose(lg.Phat, (2, 0, 1)).reshape(nS, nS * nA)
    b_eq = (1.0 - GAMMA) * lg.mu0
    bounds = [(0, None) if lg.support[s, a] else (0, 0) for s in range(nS) for a in range(nA)]
    res = linprog(-lg.r.reshape(-1), A_ub=np.array([c.reshape(-1) for c in cost_rows]), b_ub=np.array([d] * len(cost_rows)),
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    pi = np.zeros((nS, nA)); pi[:, nA - 1] = 1.0                       # unvisited states: local work
    if res.status == 0:
        x = res.x.reshape(nS, nA); tot = x.sum(1); vis = tot > 1e-12; pi[vis] = x[vis] / tot[vis, None]
    return pi, dict(offline_feasible=res.status == 0, first_feasible_step=None, n_feasible=int(res.status == 0), lam_selected=[], lam_path_std=[], lam_final=[])

def _gains(lname):
    return {"mcpq": "cpq", "lp": "lp"}.get(lname, LEARNERS[lname])

def instances():
    ex = json.load(open(os.path.join(R, "scope_agent.json")))["exact"]; out = []
    for variant in AUX_DEMAND:
        for K in K_GRID:
            for d in D_GRID:
                e = ex[f"{variant}|K{K}|d{d}"]; names = [r["name"] for r in e["readings"]]
                out.append(dict(variant=variant, K=K, d=d, psi=0, V_U=e["V_U"], V_psi=e["readings"][0]["V_psi"], V_surr=e["V_surr"],
                                price=e["price"], certified_eps001=bool(e["certified"]) and names.index(e["certified"][0]) == 0))
    return out

def margin_at(m, psi, d, eps):
    """Exact ambiguity margin of reading psi at tolerance eps (the V50 face test with eps free)."""
    K = m["K"]; r = m["r"].reshape(-1); c = [m["C"][k].reshape(-1) for k in range(K)]
    V_psi, _ = lp(m, r, [c[psi]], [d])
    Ws = [lp(m, c[phi], [c[psi], -r], [d, -(V_psi - eps)])[0] for phi in range(K) if phi != psi]
    return max(w for w in Ws if w is not None) - d, V_psi

def V_psi_at(m, psi, d):
    r = m["r"].reshape(-1); c = m["C"][psi].reshape(-1); return lp(m, r, [c], [d])[0]

def _prep(task, n, seed, tag):
    variant, K, d, psi = task["variant"], task["K"], task["d"], task["psi"]
    _set_actions(K); m = build(K, variant); C = m["C"]
    readings = [Reading(m["names"][k], C[k]) for k in range(K)]
    rng = np.random.default_rng(int(hashlib.sha256(f"{variant}|{K}|{d}|{n}|{seed}|{tag}".encode()).hexdigest()[:8], 16))
    data, flat = sample_dataset(m, behaviour_policy(m, psi, d), n, rng); lg = Log(flat); _CTX["m"], _CTX["lg"] = m, lg
    b = V50._Batch(data, readings); allowed = b.allowed(BCQ_TAU, 1)
    charged_all = flat["C"].T.astype(np.float64)                      # n x K, transition order = batch order
    assert np.array_equal(charged_all.any(axis=1), b.charged), "transition order mismatch"
    return m, readings, lg, b, allowed, charged_all

def run_arm(arm, m, lg, b, allowed, charged_all, d, psi, gains):
    K = m["K"]
    if arm == "native":   charged = charged_all;                 cost_of = lambda pi: lg.est(pi)[1]
    elif arm == "arrow":  charged = charged_all[:, [psi]];       cost_of = lambda pi: lg.est(pi)[1][[psi]]
    elif arm == "surrogate": charged = charged_all.max(axis=1, keepdims=True); cost_of = lambda pi: np.array([lg.est(pi)[1].sum()])
    else: raise ValueError(arm)
    if gains == "lp":
        K = m["K"]; psi_rows = [lg.c[k] for k in range(K)]
        rows = psi_rows if arm == "native" else ([lg.c[psi]] if arm == "arrow" else [np.max(lg.c, axis=0)])
        return learn_lp(lg, m, d, rows)
    if gains == "cpq": return learn_vector_cpq(b, allowed, charged, lg, m, d, cost_of)
    return learn_vector(b, allowed, charged, lg, m, d, gains, cost_of)

def calibrate_task(args):
    task, n, lname, seed = args; t0 = time.time()
    m, readings, lg, b, allowed, charged_all = _prep(task, n, seed, "v51cal")
    pi, diag = run_arm("arrow", m, lg, b, allowed, charged_all, task["d"], task["psi"], _gains(lname))
    x = occupancy(m["P"], m["mu0"], pi); jr = float((m["r"] * x).sum()); jc_psi = float((m["C"][task["psi"]] * x).sum())
    return dict(variant=task["variant"], K=task["K"], d=task["d"], n=n, learner=lname, seed=seed,
                shortfall=max(0.0, task["V_psi"] - jr), exceed=max(0.0, jc_psi - task["d"]), secs=time.time() - t0)

def eval_task(args):
    task, n, lname, seed, arms = args; t0 = time.time(); rows = []
    m, readings, lg, b, allowed, charged_all = _prep(task, n, seed, "v51")
    P, mu0, r, C, K, d = m["P"], m["mu0"], m["r"], m["C"], m["K"], task["d"]
    for arm in arms:
        t1 = time.time(); pi, diag = run_arm(arm, m, lg, b, allowed, charged_all, d, task["psi"], _gains(lname))
        x = occupancy(P, mu0, pi); sc = score(P, mu0, r, C, pi, d)
        rows.append(dict(variant=task["variant"], K=K, d=d, n=n, seed=seed, learner=lname, arm=arm, ret_frac=sc["J_r"] / task["V_U"],
                         costs_over_d=[float((C[k] * x).sum()) / d for k in range(K)], safe=sc["safe"], secs=time.time() - t1, **diag))
    return rows

def main():
    tasks = instances(); base = [t for t in tasks if t["certified_eps001"]]
    print(f"{len(tasks)} instances, {len(base)} certified at eps={EPS}", flush=True)
    # exact sanity check (Theorem 4.1): V_psi(d) = V_U(d) on every certified instance
    ties = [dict(variant=t["variant"], K=t["K"], d=t["d"], gap=abs(t["V_psi"] - t["V_U"])) for t in base]
    print("exact |V_psi - V_U| on certified instances: max", max(t["gap"] for t in ties), flush=True)
    smoke = "--smoke" in sys.argv
    n_grid, cal_seeds, eval_seeds, learners = ((2000,), (100,), (0,), tuple(LEARNERS)) if smoke else (N_GRID, CAL_SEEDS, EVAL_SEEDS, tuple(LEARNERS))
    if os.environ.get("V51_NGRID"): n_grid = tuple(int(x) for x in os.environ["V51_NGRID"].split(","))   # exploratory small-data extension
    out_path = os.environ.get("V51_OUT", OUT)
    workers = int(os.environ.get("V51_WORKERS", "10")); t0 = time.time()
    # 1. calibration (singleton arm, calibration seeds)
    cal_jobs = [(t, n, l, s) for t in base for n in n_grid for l in learners for s in cal_seeds]
    with Pool(workers) as pool: cal = pool.map(calibrate_task, cal_jobs, chunksize=1)
    eps_prime = {}
    for l in learners:
        for n in n_grid:
            runs = [c for c in cal if c["learner"] == l and c["n"] == n]
            eps_prime[f"{l}|{n}"] = dict(eps_r=max(c["shortfall"] for c in runs), eta=max(c["exceed"] for c in runs), n_runs=len(runs))
    print("calibration:", json.dumps(eps_prime), f"({time.time()-t0:.0f}s)", flush=True)
    # 2. re-run ARROW exactly at eps'_L(n) on all 36 instances
    cert = {}
    for key, ep in eps_prime.items():
        for t in tasks:
            m = build(t["K"], t["variant"]); eta = ep["eta"]
            eps_p = ep["eps_r"] + (t["V_psi"] - V_psi_at(m, t["psi"], t["d"] - eta) if eta > 0 else 0.0)
            marg, _ = margin_at(m, t["psi"], t["d"], eps_p)
            cert[f"{key}|{t['variant']}|K{t['K']}|d{t['d']}"] = dict(eps_prime=eps_p, margin=marg, certified=bool(marg <= 1e-9))
    for key in eps_prime:
        cs = [k for k, v in cert.items() if k.startswith(key + "|") and v["certified"]]
        print(f"  {key}: {len(cs)} instances certified at eps'={eps_prime[key]['eps_r']:.4f}: {[k.split('|',2)[2] for k in cs]}", flush=True)
    # 3. evaluation on the instances certified at eps'_L(n)
    jobs = []; cert_mode = {}
    for l in learners:
        for n in n_grid:
            sel = [t for t in tasks if cert[f"{l}|{n}|{t['variant']}|K{t['K']}|d{t['d']}"]["certified"]]
            cert_mode[f"{l}|{n}"] = "eps_prime" if sel else "eps001_fallback"     # branch rule of the amendment
            if not sel: sel = base
            for t in sel:
                arms = ("native", "arrow") + (("surrogate",) if n in SURR_N else ())
                jobs += [(t, n, l, s, arms) for s in eval_seeds]
    print(f"{len(jobs)} evaluation jobs", flush=True)
    with Pool(workers) as pool: outs = pool.map(eval_task, jobs, chunksize=1)
    rows = [r for o in outs for r in o]
    res = dict(registration="REGISTRATION_V51.md", n_grid=list(n_grid), cal_seeds=list(cal_seeds), eval_seeds=list(eval_seeds), learners=list(learners),
               gains=LEARNERS, n_dual=N_DUAL, n_iter=N_ITER, bcq_tau=BCQ_TAU, exact_ties=ties, calibration_runs=cal, eps_prime=eps_prime,
               certification=cert, cert_mode=cert_mode, tasks=tasks, rows=rows, seconds=time.time() - t0)
    json.dump(res, open(out_path if not smoke else OUT.replace(".json", "_smoke.json"), "w"))
    # quick readout
    for l in learners:
        for n in n_grid:
            sel = [r for r in rows if r["learner"] == l and r["n"] == n]
            if not sel: continue
            by = {}
            for r in sel: by.setdefault((r["variant"], r["K"], r["d"], r["seed"]), {})[r["arm"]] = r
            gaps = [v["arrow"]["ret_frac"] - v["native"]["ret_frac"] for v in by.values() if "arrow" in v and "native" in v]
            print(f"{l} n={n}: runs {len(gaps)} | native ret {np.mean([v['native']['ret_frac'] for v in by.values()]):.3f} safe {np.mean([v['native']['safe'] for v in by.values()]):.2f}"
                  f" | arrow ret {np.mean([v['arrow']['ret_frac'] for v in by.values()]):.3f} safe {np.mean([v['arrow']['safe'] for v in by.values()]):.2f}"
                  f" | median gap {np.median(gaps):+.3f} share>0 {np.mean(np.array(gaps) > 1e-9):.2f}"
                  f" | first feasible step native {np.mean([v['native']['first_feasible_step'] if v['native']['first_feasible_step'] is not None else N_DUAL for v in by.values()]):.1f} arrow {np.mean([v['arrow']['first_feasible_step'] if v['arrow']['first_feasible_step'] is not None else N_DUAL for v in by.values()]):.1f}", flush=True)
    print("wrote", out_path, "rows", len(rows), "in %.0fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
