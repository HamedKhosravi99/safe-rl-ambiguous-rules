"""V50 learned arms: the paper's single-signal learners on the scope-ambiguous agent domain.
See scope_agent.py and REGISTRATION_V50.md.  Adapter: the learners' action set and cost
signal are generalised by patching module globals (no learner file is edited).
Run: OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m saorl.benchmark_sg.scope_agent_learn [--smoke]
Writes results/e2e/scope_agent_learn.json
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import hashlib, json, sys, time
from multiprocessing import Pool
import numpy as np

from saorl import neural_rl as nrl, offline as off, offline_rl as orl, sota_learners as sl
from saorl.offline import OfflineDataset
from saorl.benchmark_sg.scope_agent import GAMMA, AUX_DEMAND, K_GRID, D_GRID, build, exact
from saorl.benchmark_sg.safe_face_offline import Log, occupancy, score
import saorl.benchmark_sg.safe_face_offline as sfo

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "results/e2e"); OUT = os.path.join(R, "scope_agent_learn.json")
N_GRID = (2000, 20000); SEEDS = tuple(range(10)); CQL_N = (20000,); LEARNERS = ("fqi", "cpq", "pid", "cql")
LAM_GRID = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0); BCQ_TAU = 0.05
BEHAV_MIX = 0.3

# ------------------------------------------------------------------ adapter (patches)
_CTX = {}
def _set_actions(K):
    acts = tuple(["call_%d" % k for k in range(K)] + ["local"])
    for mod in (orl, nrl):
        mod.ACTIONS = acts; mod.NA = len(acts); mod.A_INDEX = {a: i for i, a in enumerate(acts)}
    sl.NA = len(acts)
    nrl._FEATS = tuple("m%d" % k for k in range(K))
    _CTX["acts"] = acts

class _Batch(orl._Batch):
    def __init__(self, data, honor):
        super().__init__(data, honor)
        self.charged = np.asarray([bool(any(c.cost(traj, t, acts[t]) for c in honor))
                                   for traj, acts in zip(data.trajectories, data.actions) for t in range(len(acts))])
class _NeuralBatch(nrl._NeuralBatch):
    def __init__(self, data, honor):
        super().__init__(data, honor)
        self.charged = np.asarray([bool(any(c.cost(traj, t, acts[t]) for c in honor))
                                   for traj, acts in zip(data.trajectories, data.actions) for t in range(len(acts))])
orl._Batch = _Batch; sl._Batch = _Batch; nrl._NeuralBatch = _NeuralBatch
orl.discretize = lambda state, rul_w=4.0, anom_w=0.1: (int(state["sid"]), 0)
_orig_tab_greedy = orl._greedy_policy
def _tab_greedy(b, q, allowed):            # unlogged states: the paper defaults to 'continue'; here the zero-cost local action
    pol = _orig_tab_greedy(b, q, allowed)
    return lambda traj, t: (lambda a: "local" if a == "continue" else a)(pol(traj, t))
orl._greedy_policy = _tab_greedy; sl._greedy_policy = _tab_greedy

def _fit_q(b, allowed, lam, gamma, n_iter):        # offline_rl._fit_q with the generalised cost signal
    NA = orl.NA
    q = np.where(allowed, 0.0, -1e9); shaped = b.r - lam * b.charged
    terminal = b.sp < 0; sa_flat = b.s * NA + b.a
    sum_buf = np.zeros(b.n_states * NA); cnt_buf = np.zeros(b.n_states * NA); np.add.at(cnt_buf, sa_flat, 1.0); nz = cnt_buf > 0
    q_masked = np.where(allowed, q, -np.inf)
    for _ in range(n_iter):
        nxt = q_masked.max(axis=1); nxt = np.where(np.isfinite(nxt), nxt, 0.0)
        boot = np.where(terminal, 0.0, gamma * nxt[np.where(terminal, 0, b.sp)]); y = shaped + boot
        sum_buf[:] = 0.0; np.add.at(sum_buf, sa_flat, y); flat = q.reshape(-1); flat[nz] = sum_buf[nz] / cnt_buf[nz]
        q = flat.reshape(b.n_states, NA); q_masked = np.where(allowed, q, -np.inf)
    return q
orl._fit_q = _fit_q; sl._fit_q = _fit_q

def _fit_cost_q(b, safe, qr, gamma, n_iter):        # sota_learners._fit_cost_q with the generalised cost signal
    NA = orl.NA
    qr_safe = np.where(safe, qr, -np.inf); nxt_a = qr_safe.argmax(axis=1)
    cost_step = b.charged.astype(np.float64); terminal = b.sp < 0; ns = np.where(terminal, 0, b.sp)
    sa_flat = b.s * NA + b.a; cnt = np.zeros(b.n_states * NA); np.add.at(cnt, sa_flat, 1.0); nz = cnt > 0
    qc = np.zeros((b.n_states, NA)); sum_buf = np.zeros(b.n_states * NA)
    for _ in range(n_iter):
        boot = np.where(terminal, 0.0, gamma * qc[ns, nxt_a[ns]]); y = cost_step + boot
        sum_buf[:] = 0.0; np.add.at(sum_buf, sa_flat, y); flat = qc.reshape(-1); flat[nz] = sum_buf[nz] / cnt[nz]
        qc = flat.reshape(b.n_states, NA)
    return qc
sl._fit_cost_q = _fit_cost_q

_orig_train_cql = nrl._train_cql
def _train_cql(b, lam, gamma, n_steps, batch_size, cql_alpha, lr, hidden, seed, device):
    import torch, torch.nn as nn
    torch.manual_seed(seed); dev = torch.device(device); NA = nrl.NA; F = len(nrl._FEATS)
    S = torch.tensor(b.norm(b.s), dtype=torch.float32, device=dev); SP = torch.tensor(b.norm(b.sp), dtype=torch.float32, device=dev)
    A = torch.tensor(b.a, dtype=torch.long, device=dev); Rw = torch.tensor(b.r - lam * b.charged, dtype=torch.float32, device=dev)
    NT = torch.tensor(1.0 - b.term, dtype=torch.float32, device=dev)
    def mlp(): return nn.Sequential(nn.Linear(F, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, NA)).to(dev)
    q, qt = mlp(), mlp(); qt.load_state_dict(q.state_dict()); opt = torch.optim.Adam(q.parameters(), lr=lr); n = S.shape[0]; rng = np.random.default_rng(seed)
    for step in range(n_steps):
        idx = torch.tensor(rng.integers(0, n, size=batch_size), device=dev)
        s, sp, a, r, nt = S[idx], SP[idx], A[idx], Rw[idx], NT[idx]
        with torch.no_grad(): tgt = r + gamma * nt * qt(sp).max(dim=1).values
        qsa_all = q(s); qsa = qsa_all.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = ((qsa - tgt) ** 2).mean() + cql_alpha * (torch.logsumexp(qsa_all, dim=1) - qsa).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0: qt.load_state_dict(q.state_dict())
    q.eval(); return q, dev
nrl._train_cql = _train_cql
_orig_nr_greedy = nrl._greedy_policy
def _nr_greedy_cached(b, q, dev):
    pol = _orig_nr_greedy(b, q, dev); cache = {}
    def p(traj, t):
        sid = traj[t]["sid"]
        if sid not in cache: cache[sid] = pol(traj, t)
        return cache[sid]
    return p
nrl._greedy_policy = _nr_greedy_cached
off.active_steps = lambda data, U: data.n_steps()
def _wcc_phat(policy, data, U, normalize="all"):
    m, lg = _CTX["m"], _CTX["lg"]; pi = policy_table(m, policy); x = occupancy(lg.Phat, lg.mu0, pi)
    return max(float((c.C * x).sum()) for c in U)
orl.worst_case_cost = _wcc_phat; sl.worst_case_cost = _wcc_phat; nrl.worst_case_cost = _wcc_phat

class Reading:
    """Duck-typed Candidate: charges the actions in `C` (nS x nA 0/1)."""
    def __init__(self, name, C): self.name, self.C, self.plausibility = name, C, ()
    def fires(self, traj, t): return bool(self.C[traj[t]["sid"]].any())
    def cost(self, traj, t, action): return int(self.C[traj[t]["sid"], _CTX["acts"].index(action)] > 0)

def _obs(m, s):
    d = {"sid": s}
    for k, v in enumerate(m["S"][s]): d["m%d" % k] = float(v)
    return d

def behaviour_policy(m, psi, d):
    """0.7 x psi-optimal + 0.3 x uniform (as V48/V49), from the exact LP on the true model."""
    from saorl.benchmark_sg.scope_agent import lp
    nS, nA = m["nS"], m["nA"]; r = m["r"].reshape(-1); c = m["C"][psi].reshape(-1)
    V, x = lp(m, r, [c], [d]); x = x.reshape(nS, nA); tot = x.sum(1); pi = np.full((nS, nA), 1.0 / nA)
    vis = tot > 1e-12; pi[vis] = x[vis] / tot[vis, None]
    return (1.0 - BEHAV_MIX) * pi + BEHAV_MIX / nA

def sample_dataset(m, pi_b, n, rng):
    P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]; nS, nA, K = m["nS"], m["nA"], C.shape[0]; acts = _CTX["acts"]
    S, A, SP, Rw, Cc = [], [], [], [], []; trajs, actl, rews = [], [], []; cs, ca, cr = [], [], []
    s = rng.choice(nS, p=mu0)
    for i in range(n):
        a = rng.choice(nA, p=pi_b[s]); sp = rng.choice(nS, p=P[s, a])
        S.append(s); A.append(a); SP.append(sp); Rw.append(r[s, a]); Cc.append([C[k, s, a] for k in range(K)])
        cs.append(_obs(m, s)); ca.append(acts[a]); cr.append(float(r[s, a]))
        if rng.random() < GAMMA and i < n - 1: s = sp
        else:
            cs.append(_obs(m, sp)); trajs.append(cs); actl.append(ca); rews.append(cr); cs, ca, cr = [], [], []; s = rng.choice(nS, p=mu0)
    flat = dict(S=np.array(S), A=np.array(A), SP=np.array(SP), R=np.array(Rw), C=np.array(Cc).T, nS=nS, nA=nA, K=K, mu0=mu0)
    return OfflineDataset(trajs, actl, rews), flat

def policy_table(m, pol):
    nS, nA = m["nS"], m["nA"]; pi = np.zeros((nS, nA)); acts = _CTX["acts"]
    for s in range(nS): pi[s, acts.index(pol([_obs(m, s)], 0))] = 1.0
    return pi

def run_learner(name, data, lg, m, honor, U_eval, d, seed):
    _CTX["m"], _CTX["lg"] = m, lg; ret_fn = lambda pol: lg.est(policy_table(m, pol))[0]
    if name == "fqi": return orl.learn_fqi_constrained(data, None, honor=honor, U_eval=U_eval, eps=d, gamma=GAMMA, bcq_tau=BCQ_TAU, lam_grid=LAM_GRID, return_fn=ret_fn)
    if name == "cpq": return sl.learn_cpq_constrained(data, None, honor=honor, U_eval=U_eval, eps=d, gamma=GAMMA, bcq_tau=BCQ_TAU, return_fn=ret_fn)
    if name == "pid": return sl.learn_pid_lagrangian_constrained(data, None, honor=honor, U_eval=U_eval, eps=d, gamma=GAMMA, bcq_tau=BCQ_TAU, return_fn=ret_fn)
    if name == "cql": return nrl.learn_cql_constrained(data, None, honor=honor, U_eval=U_eval, eps=d, gamma=GAMMA, seed=seed, device="cpu", return_fn=ret_fn)
    raise ValueError(name)

def certified_instances():
    ex = json.load(open(os.path.join(R, "scope_agent.json")))["exact"]
    out = []
    for variant in AUX_DEMAND:
        for K in K_GRID:
            for d in D_GRID:
                e = ex[f"{variant}|K{K}|d{d}"]
                if e["certified"]:
                    out.append(dict(variant=variant, K=K, d=d, psi=[r["name"] for r in e["readings"]].index(e["certified"][0]), V_U=e["V_U"], V_surr=e["V_surr"], price=e["price"]))
    return out

def run_task(task, seeds=SEEDS, n_grid=N_GRID, learners=LEARNERS):
    variant, K, d, psi, V_U = task["variant"], task["K"], task["d"], task["psi"], task["V_U"]
    _set_actions(K); m = build(K, variant); P, mu0, r, C = m["P"], m["mu0"], m["r"], m["C"]
    readings = [Reading(m["names"][k], C[k]) for k in range(K)]; union = Reading("union", np.max(C, axis=0))
    arms = {f"single:{m['names'][k]}": [readings[k]] for k in range(K)}; arms["surrogate"] = [union]
    pi_b = behaviour_policy(m, psi, d); rows = []; t0 = time.time()
    for n in n_grid:
        for seed in seeds:
            rng = np.random.default_rng(int(hashlib.sha256(f"{variant}|{K}|{d}|{n}|{seed}|v50".encode()).hexdigest()[:8], 16))
            data, flat = sample_dataset(m, pi_b, n, rng); lg = Log(flat)
            for lname in learners:
                if lname == "cql" and n not in CQL_N: continue
                for arm, honor in arms.items():
                    t1 = time.time(); res = run_learner(lname, data, lg, m, honor, readings, d, seed); pi = policy_table(m, res.policy)
                    x = occupancy(P, mu0, pi); sc = score(P, mu0, r, C, pi, d)
                    rows.append(dict(variant=variant, K=K, d=d, n=n, seed=seed, learner=lname, arm=arm, is_arrow=(arm == f"single:{m['names'][psi]}"),
                                     ret_frac=sc["J_r"] / V_U, costs_over_d=[float((C[k] * x).sum()) / d for k in range(K)], cmax_over_d=sc["C_max"] / d,
                                     safe=sc["safe"], honored_offline=float(res.honored_cost), lam=float(res.lam), secs=time.time() - t1))
    return dict(task=task, rows=rows, secs=time.time() - t0)

def main():
    tasks = certified_instances(); print(f"{len(tasks)} certified instances:", [(t["variant"], t["K"], t["d"]) for t in tasks], flush=True)
    if "--smoke" in sys.argv:
        out = run_task(tasks[-1], seeds=(0,), n_grid=(20000,))
        for row in out["rows"]: print(f"{row['learner']:4s} {row['arm']:20s} ret={row['ret_frac']:.3f} costs/d={np.round(row['costs_over_d'],2)} safe={row['safe']} offl={row['honored_offline']:.3f} lam={row['lam']:.1f} {row['secs']:.1f}s")
        print("secs", round(out["secs"], 1)); return
    t0 = time.time()
    with Pool(int(os.environ.get("V50_WORKERS", "10"))) as pool: outs = pool.map(run_task, tasks, chunksize=1)
    rows = [r for o in outs for r in o["rows"]]
    json.dump(dict(registration="V50 learned arms", n_grid=list(N_GRID), seeds=len(SEEDS), learners=list(LEARNERS), tasks=[o["task"] for o in outs], rows=rows, seconds=time.time() - t0), open(OUT, "w"))
    print("wrote", OUT, "rows", len(rows), "in %.0fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
