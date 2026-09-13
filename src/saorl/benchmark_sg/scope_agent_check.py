"""V52: the paper's checked deployment route (empirical-Bernstein CHECK, Bonferroni over readings)
applied to every learned policy of the V50 agent-service experiment; see REGISTRATION_V52.md.
Run:  PYTHONPATH=src python3 -m saorl.benchmark_sg.scope_agent_check     Writes results/e2e/scope_agent_check.json
"""
from __future__ import annotations
import hashlib, json, os, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "results/e2e"); OUT = os.path.join(R, "scope_agent_check.json")
N_EV, DELTA_EV = 20000, 0.05          # the paper's CHECK budget and level (collapse_utility.py, pipeline_compare.py)

def check_from_rates(rates, d, K, n_ev, rng, delta_ev):
    """Exact law of check_ship(P, mu0, C, pi, d, n_ev, rng, delta_ev) when every c_k is 0/1 and the readings fire on
    disjoint actions: each fresh (s,a) draw charges reading k with probability rates[k], none with 1 - sum."""
    p = np.array(list(rates) + [max(0.0, 1.0 - sum(rates))]); p = p / p.sum()
    counts = rng.multinomial(n_ev, p)                                        # one sample of the K+1 categories
    L = np.log(3.0 / (delta_ev / K)); bound = -np.inf
    for k in range(K):
        z_mean = counts[k] / n_ev; z_var = z_mean * (1 - z_mean) * n_ev / (n_ev - 1)   # sample variance (ddof=1) of a 0/1 sample
        b = z_mean + np.sqrt(2.0 * z_var * L / n_ev) + 3.0 * L / n_ev; bound = max(bound, b)
    return float(bound), bool(bound <= d)

def main():
    d = json.load(open(os.path.join(R, "scope_agent_learn.json"))); t0 = time.time(); out = []
    for r in d["rows"]:
        K, dd = r["K"], r["d"]; rates = [c * dd for c in r["costs_over_d"]]
        rng = np.random.default_rng(int(hashlib.sha256(f"{r['variant']}|{K}|{dd}|{r['learner']}|{r['arm']}|{r['n']}|{r['seed']}|v52".encode()).hexdigest()[:8], 16))
        bound, ship = check_from_rates(rates, dd, K, N_EV, rng, DELTA_EV)
        out.append(dict({k: r[k] for k in ("variant", "K", "d", "n", "seed", "learner", "arm", "is_arrow", "ret_frac", "safe")},
                        true_max_rate_over_d=max(r["costs_over_d"]), bound_over_d=bound / dd, ship=ship))
    res = dict(registration="REGISTRATION_V52.md", n_ev=N_EV, delta_ev=DELTA_EV, source="scope_agent_learn.json", rows=out, seconds=time.time() - t0)
    json.dump(res, open(OUT, "w"))
    def arm_of(r): return "arrow" if r["is_arrow"] else ("surrogate" if r["arm"] == "surrogate" else "wrong")
    print(f"{'learner':5s} {'n':>6s} {'arm':9s} {'runs':>4s} {'ret':>6s} {'pass':>6s} {'unsafe&pass':>11s} {'deployed ret':>12s} {'safe&rejected':>13s}")
    for l in d["learners"]:
        for n in d["n_grid"]:
            for arm in ("arrow", "surrogate", "wrong"):
                sel = [r for r in out if r["learner"] == l and r["n"] == n and arm_of(r) == arm]
                if not sel: continue
                dep = [r for r in sel if r["ship"]]
                print(f"{l:5s} {n:6d} {arm:9s} {len(sel):4d} {np.mean([r['ret_frac'] for r in sel]):6.3f} {np.mean([r['ship'] for r in sel]):6.2f} {sum(1 for r in dep if not r['safe']):11d} "
                      f"{(np.mean([r['ret_frac'] for r in dep]) if dep else float('nan')):12.3f} {sum(1 for r in sel if r['safe'] and not r['ship']):13d}")
    print("wrote", OUT, len(out), "rows")

if __name__ == "__main__":
    main()
