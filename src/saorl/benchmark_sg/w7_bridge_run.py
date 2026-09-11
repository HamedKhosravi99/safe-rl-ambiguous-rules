"""W7 bridge experiment, Stages 0 and 1 (registered in w7_bridge_protocol.py).

Neural arm on the compiled source-grounded instances: discrete CQL +
Lagrangian grid with fully offline selection, mirroring the paper's neural
wrapper (`saorl/neural_rl.py`): smallest lambda whose OFFLINE honored cost
meets d, else largest.  States are one-hot; evaluation is exact in the
compiled MDP (occupancy of the greedy policy) plus per-seed Clopper-Pearson
over 300 simulated episodes, never pooled, exactly as registered.

Arms per (instance, seed):
  single -- honor psi_1 only (the first maximal reading);
  set    -- honor max over all maximal readings.

Writes results/e2e/w7_bridge.json incrementally (one record per instance).
Run:  PYTHONPATH=. nohup python3 -u -m saorl.benchmark_sg.w7_bridge_run &
"""
from __future__ import annotations

import json
import os
from typing import List

import numpy as np

from .control_suite import compile_instance, GAMMA
from .w7_bridge_protocol import REGISTRATION, occupancy_fixed_policy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SUITE = os.path.join(ROOT, "results/conformal", "benchmark_sg", "control_suite.json")
OUT = os.path.join(ROOT, "results/e2e", "w7_bridge.json")

D = REGISTRATION["budget_d"]
N_TRAJ, H_TRAJ = REGISTRATION["n_traj"], REGISTRATION["h_traj"]
SEEDS = REGISTRATION["seeds"]
LAM_GRID = (0.0, 5.0, 20.0, 80.0, 160.0, 320.0)
N_STEPS, BATCH, ALPHA, LR, HIDDEN = 4000, 256, 1.0, 1e-3, 128
N_CERT_EP = 300


def gen_dataset(m: dict, seed: int):
    rng = np.random.default_rng(seed)
    nS, nA = m["nS"], m["nA"]
    P = m["P"]
    s0 = rng.choice(nS, size=N_TRAJ, p=m["mu0"])
    S, A, R, SP, T = [], [], [], [], []
    for i in range(N_TRAJ):
        s = int(s0[i])
        for t in range(H_TRAJ):
            a = int(rng.integers(0, nA))
            sp = int(rng.choice(nS, p=P[s, a]))
            S.append(s); A.append(a); R.append(m["r"][s, a]); SP.append(sp)
            T.append(0.0)
            s = sp
    return (np.array(S), np.array(A), np.array(R, dtype=np.float64),
            np.array(SP), np.array(T))


def train_cql(m, data, cost_sa, lam, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    S, A, R, SP, T = data
    nS, nA = m["nS"], m["nA"]
    eye = np.eye(nS, dtype=np.float32)
    Sx = torch.tensor(eye[S]); SPx = torch.tensor(eye[SP])
    At = torch.tensor(A, dtype=torch.long)
    shaped = R - lam * cost_sa[S, A]
    Rt = torch.tensor(shaped, dtype=torch.float32)

    def mlp():
        return nn.Sequential(nn.Linear(nS, HIDDEN), nn.ReLU(),
                             nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                             nn.Linear(HIDDEN, nA))
    q, qt = mlp(), mlp()
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=LR)
    n = len(S)
    rng = np.random.default_rng(seed)
    for step in range(N_STEPS):
        idx = torch.tensor(rng.integers(0, n, size=BATCH))
        s, sp, a, r = Sx[idx], SPx[idx], At[idx], Rt[idx]
        with torch.no_grad():
            tgt = r + GAMMA * qt(sp).max(dim=1).values
        qall = q(s)
        qsa = qall.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = ((qsa - tgt) ** 2).mean() + ALPHA * (
            torch.logsumexp(qall, dim=1) - qsa).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            qt.load_state_dict(q.state_dict())
    with torch.no_grad():
        Q = q(torch.tensor(eye)).numpy()
    return Q.argmax(axis=1)                      # deterministic greedy policy


def offline_cost(m, data, cost_sa, pol) -> float:
    """Offline surrogate mirroring worst_case_cost(active-normalized): mean
    policy cost over logged ACTIVE states (any reading can fire there)."""
    S = data[0]
    active_state = (m["C"].max(axis=0).max(axis=1) > 0)
    mask = active_state[S]
    if not mask.any():
        return 0.0
    return float(cost_sa[S[mask], pol[S[mask]]].mean())


def exact_eval(m, pol):
    nS, nA = m["nS"], m["nA"]
    pi = np.zeros((nS, nA)); pi[np.arange(nS), pol] = 1.0
    occ = occupancy_fixed_policy(m, pi)
    ret = float((m["r"] * occ).sum())
    J = [float((m["C"][k] * occ).sum()) for k in range(m["C"].shape[0])]
    return ret, J


def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def episode_cert(m, pol, seed) -> float:
    """CP upper bound on P(episode discounted worst-reading cost > d)."""
    rng = np.random.default_rng(10_000 + seed)
    nS = m["nS"]
    worst_c = m["C"].max(axis=0)
    viol = 0
    for _ in range(N_CERT_EP):
        s = int(rng.choice(nS, p=m["mu0"]))
        tot, g = 0.0, 1.0
        for t in range(H_TRAJ):
            a = int(pol[s])
            tot += g * worst_c[s, a]
            g *= GAMMA
            s = int(rng.choice(nS, p=m["P"][s, a]))
        if tot > D:
            viol += 1
    return cp_upper(viol, N_CERT_EP)


def _occ_of(m, pol):
    nS, nA = m["nS"], m["nA"]
    pi = np.zeros((nS, nA)); pi[np.arange(nS), pol] = 1.0
    return occupancy_fixed_policy(m, pi)


def main() -> None:
    suite = json.load(open(SUITE))
    done = {}
    if os.path.exists(OUT):
        done = {r["uid"]: r for r in json.load(open(OUT))["rows"]}
    rows = list(done.values())
    for inst in suite["instances"]:
        if inst["uid"] in done:
            continue
        class R:
            pass
        readings = []
        for rd in inst["readings"]:
            r = R(); r.threshold, r.for_s = rd["theta"], rd["for_s"]
            readings.append(r)
        m = compile_instance(readings)
        K = m["C"].shape[0]
        single_cost = m["C"][0]
        set_cost = m["C"].max(axis=0)
        cell = dict(uid=inst["uid"], geometry=inst["geometry"],
                    n_states=m["nS"], seeds=[])
        for seed in SEEDS:
            data = gen_dataset(m, seed)
            out_arms = {}
            for arm, cost in (("single", single_cost), ("set", set_cost)):
                pol = None
                for lam in LAM_GRID:
                    pol = train_cql(m, data, cost, lam, seed)
                    if offline_cost(m, data, cost, pol) <= D:
                        break
                ret, J = exact_eval(m, pol)
                own = J[0] if arm == "single" else max(J)
                worst = max(J)
                out_arms[arm] = dict(
                    ret=ret, J=J, own_cost=own, worst_cost=worst,
                    hidden_gap=(worst - J[0]) if arm == "single" else 0.0,
                    cp=episode_cert(m, pol, seed))
            cell["seeds"].append(out_arms)
            print(f"  {inst['uid']} seed {seed}: single worst="
                  f"{out_arms['single']['worst_cost']:.4f} "
                  f"set worst={out_arms['set']['worst_cost']:.4f}", flush=True)
        rows.append(cell)
        json.dump(dict(registration="REGISTRATION_W7", rows=rows),
                  open(OUT, "w"), indent=1)
        print(f"[w7] {inst['uid']} done ({len(rows)}/{len(suite['instances'])})",
              flush=True)
    print("[w7] bridge run complete")


if __name__ == "__main__":
    main()
