"""V49b: Refine under answer noise WITH the Check stage, both truth priors, a
posterior-rate sensitivity grid and the questions / fallback / unsafe
trade-off. Registration V49b.

Run: PYTHONHASHSEED=0 OMP_NUM_THREADS=1 SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.answer_noise_check
Writes results/e2e/answer_noise_check.json
"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .query_loop import ROOT
from .answer_noise import Pool, fixture_pools, compiled_pools, make_oracle, SOFT_FLOOR
from .safe_collapse import FixtureFace, CompiledFace
from .baseline_table import _lp

OUT = os.path.join(ROOT, "results/e2e", "answer_noise_check.json")
P_GRID = (0.0, 0.05, 0.1, 0.2, 0.3)
Q_ABST = (0.1, 0.3)
P_BAL = (0.1, 0.2, 0.3)
PA_GRID = (0.02, 0.05, 0.1, 0.2, 0.3)
DPOST_GRID = (0.05, 0.2)
QMAX = 8
QMAX_REPEAT = 9
DRAWS = 20
TAU_CHECK = 0.01
SEED = 20260906
TOL = 1e-9


# --------------------------------------------------------------------------- policies (trained at the stop) and exact costs
class Policies:
    """Cache of optimal policies under a constraint set and their exact per-reading costs."""

    def __init__(self, pool: Pool):
        self.pool = pool; self.tester = pool.tester; self._c = {}

    def costs(self, S):
        key = frozenset(S)
        if key in self._c:
            return self._c[key]
        t = self.tester
        if isinstance(t, FixtureFace):
            cons = [LinearConstraint(t.w[i][None, :], -np.inf, 0.0) for i in sorted(key)]
            res = milp(c=-t.ones, constraints=cons, integrality=np.ones(t.F), bounds=Bounds(0, 1))
            if not res.success:
                out = None
            else:
                z = np.round(res.x); val = float(z.sum())
                cost = [float((t.vecs[i] * z).sum() / max(val, 1.0)) for i in range(t.vecs.shape[0])]   # mean verdict over accepted fixtures
                out = dict(value=val, cost=cost)
        else:
            v, x = _lp(t.m, t.m["r"], [t.rows[i] for i in sorted(key)], [t.d] * len(key))
            if v is None:
                out = None
            else:
                cost = [float(t.rows[i].reshape(-1) @ x) for i in range(len(t.rows))]
                out = dict(value=float(v), cost=cost)
        self._c[key] = out
        return out

    def feasible_for(self, pol, S):
        return all(pol["cost"][i] <= self.tester.d + 1e-7 for i in S)


# --------------------------------------------------------------------------- one run with the Check stage
def run(pool: Pool, pols: Policies, truth: int, oracle, protocol: str, p_assumed: float, dpost: float, rng: random.Random):
    inst, K, d = pool.inst, pool.K, pool.tester.d
    incons = np.zeros(K, dtype=int); logw = np.zeros(K); asked = set(); answers = 0
    qmax = QMAX_REPEAT if protocol == "repeat3" else QMAX

    def survivors():
        if protocol == "tolerant1":
            return [i for i in pool.all if incons[i] <= 1]
        if protocol == "bayes":
            mx = logw.max(); return [i for i in pool.all if logw[i] >= mx + np.log(SOFT_FLOOR)]
        return [i for i in pool.all if incons[i] == 0]

    def check_set():
        if protocol == "bayes":
            w = np.exp(logw - logw.max()); w /= w.sum(); return [i for i in pool.all if w[i] >= TAU_CHECK]
        return survivors()

    def stopped(S):
        if protocol == "bayes":
            w = np.exp(logw - logw.max()); w /= w.sum(); best, bm = None, -1.0
            for k, Rg in pool.regions.items():
                mass = float(sum(w[i] for i in Rg))
                if mass > bm + 1e-12 or (abs(mass - bm) <= 1e-12 and best is not None and (pool.V1[k], -k) > (pool.V1[best], -best)):
                    best, bm = k, mass
            return (bm >= 1.0 - dpost), (best if bm >= 1.0 - dpost else None)
        a = pool.anchor(S); return (a is not None), a

    S = survivors(); safe, anc = stopped(S)
    while not safe and answers < qmax:
        f = pool.pick(S, asked, rng)
        if f is None:
            break
        asked.add(f)
        if protocol == "repeat3":
            ys = []
            for _ in range(3):
                y, _fl = oracle(f, S); answers += 1
                if y is not None:
                    ys.append(y)
            if not ys:
                S = survivors(); safe, anc = stopped(S); continue
            vals = sorted(set(ys), key=lambda v: (-ys.count(v), rng.random())); y = vals[0]
        else:
            y, _fl = oracle(f, S); answers += 1
            if y is None:
                S = survivors(); safe, anc = stopped(S); continue
        m_f = len(pool.colvals[f])
        for i in pool.all:
            agree = inst.vec[i][f] == y
            if not agree:
                incons[i] += 1
            if protocol == "bayes":
                logw[i] += np.log(max(1.0 - p_assumed, 1e-12)) if agree else np.log(max(p_assumed / max(m_f - 1, 1), 1e-12))
        S = survivors()
        if not S:
            break
        safe, anc = stopped(S)
    CS = check_set()
    truth_in = truth in CS
    out = dict(answers=answers, truth_eliminated=(not truth_in), safe=bool(safe), handover=False, fallback=False, empty=(len(CS) == 0),
               unsafe_handover=False, unsafe_fallback=False, ret_over_VU=None, ret_over_Vtruth=None, anchor=anc)
    VU = pols.costs(pool.all)["value"]; Vt = pols.costs([truth])["value"]
    if safe and anc is not None:
        pol = pols.costs([anc])
        if pol is not None and pols.feasible_for(pol, CS):          # the Check against the surviving / credible set (exact)
            out["handover"] = True; out["unsafe_handover"] = bool(pol["cost"][truth] > d + 1e-7)
            out["ret_over_VU"] = pol["value"] / VU if VU > 0 else None; out["ret_over_Vtruth"] = pol["value"] / Vt if Vt > 0 else None
            return out
    if CS:
        pol = pols.costs(CS)
        if pol is not None:
            out["fallback"] = True; out["unsafe_fallback"] = bool(pol["cost"][truth] > d + 1e-7)
            out["ret_over_VU"] = pol["value"] / VU if VU > 0 else None; out["ret_over_Vtruth"] = pol["value"] / Vt if Vt > 0 else None
    return out


# --------------------------------------------------------------------------- cells
def cells():
    out = []
    for p in P_GRID:
        for pr in ("hard", "tolerant1", "repeat3"):
            out.append(("flip", p, pr, 0.0, 0.05))
        for pa in PA_GRID:
            for dp in DPOST_GRID:
                out.append(("flip", p, "bayes", pa, dp))
    for q in Q_ABST:
        out.append(("abstain", q, "hard", 0.0, 0.05)); out.append(("abstain", q, "bayes", 0.1, 0.05))
    for p in P_BAL:
        out.append(("balflip", p, "hard", 0.0, 0.05)); out.append(("balflip", p, "bayes", 0.1, 0.05))
    return out


def key_of(c):
    model, level, pr, pa, dp = c
    return f"{model}|{level}|{pr}" + (f"|pa{pa}|dp{dp}" if pr == "bayes" else "")


def summarize(per_pool, unsafe0):
    out = {}
    for cell, by_uid in per_pool.items():
        def rate(fn, uids, cond=None):
            vals = []
            for u in uids:
                rs = by_uid.get(u, [])
                if cond is not None:
                    rs = [r for r in rs if cond(r)]
                if rs:
                    vals.append(np.mean([fn(r) for r in rs]))
            return float(np.mean(vals)) if vals else None
        stats = {}
        for lab, uids in (("all", list(by_uid)), ("unsafe_at_0", [u for u in by_uid if u in unsafe0])):
            allruns = [r for u in uids for r in by_uid.get(u, [])]
            stats[lab] = dict(n_pools=len(uids), n_runs=len(allruns),
                              truth_eliminated=rate(lambda r: r["truth_eliminated"], uids),
                              handover=rate(lambda r: r["handover"], uids), fallback=rate(lambda r: r["fallback"], uids), empty=rate(lambda r: r["empty"], uids),
                              unsafe_handover=rate(lambda r: r["unsafe_handover"], uids), unsafe_fallback=rate(lambda r: r["unsafe_fallback"], uids),
                              unsafe_any=rate(lambda r: r["unsafe_handover"] or r["unsafe_fallback"], uids),
                              answers_mean=rate(lambda r: r["answers"], uids), answers_median=(float(np.median([r["answers"] for r in allruns])) if allruns else None),
                              ret_over_VU_handover=rate(lambda r: r["ret_over_VU"], uids, cond=lambda r: r["handover"] and r["ret_over_VU"] is not None),
                              ret_over_Vtruth_handover=rate(lambda r: r["ret_over_Vtruth"], uids, cond=lambda r: r["handover"] and r["ret_over_Vtruth"] is not None),
                              ret_over_VU_any=rate(lambda r: r["ret_over_VU"], uids, cond=lambda r: r["ret_over_VU"] is not None))
        out[cell] = stats
    return out


def main():
    t0 = time.perf_counter(); CELLS = cells()
    res = dict(registration="V49b (REGISTRATION_V28.md)", p_grid=list(P_GRID), pa_grid=list(PA_GRID), dpost_grid=list(DPOST_GRID), qmax=QMAX, draws=DRAWS, tau_check=TAU_CHECK, populations={})
    for name, fn in (("admission", lambda: fixture_pools("kyverno")), ("compiled_d0.005", lambda: compiled_pools(0.005)), ("monitoring_free", lambda: fixture_pools("prometheus"))):
        pools = fn(); t1 = time.perf_counter()
        unsafe0 = {p.uid for p in pools if p.anchor(p.all) is None}
        per = {"gold": {key_of(c): {} for c in CELLS}, "uniform": {key_of(c): {} for c in CELLS}}
        for pi_, pool in enumerate(pools):
            pols = Policies(pool)
            for prior in ("gold", "uniform"):
                for c in CELLS:
                    model, level, pr, pa, dp = c; runs = []
                    for dr in range(DRAWS):
                        rng = random.Random(hash((SEED, pool.uid, prior, key_of(c), dr)) % (2 ** 31))
                        truth = pool.inst.gold if prior == "gold" else rng.randrange(pool.K)
                        oracle = make_oracle(pool, truth, model, level, rng)
                        runs.append(run(pool, pols, truth, oracle, pr, pa, dp, rng))
                    per[prior][key_of(c)][pool.uid] = runs
            if pi_ % 10 == 0:
                print(f"  [{name}] {pi_+1}/{len(pools)} pools ({time.perf_counter()-t1:.0f}s)", flush=True)
        res["populations"][name] = dict(n_pools=len(pools), n_unsafe_at_0=len(unsafe0), summary={pr: summarize(per[pr], unsafe0) for pr in per},
                                        seconds=round(time.perf_counter() - t1, 1))
        print(f"== {name}: {len(pools)} pools, {len(unsafe0)} not safe at q=0 ({res['populations'][name]['seconds']} s)")
        for prior in ("gold", "uniform"):
            for cell in ("flip|0.1|hard", "flip|0.2|hard", "flip|0.1|bayes|pa0.1|dp0.05", "flip|0.2|bayes|pa0.1|dp0.05", "flip|0.2|bayes|pa0.02|dp0.05"):
                s = res["populations"][name]["summary"][prior][cell]["unsafe_at_0"]
                print(f"   [{prior}] {cell:32s} elim {s['truth_eliminated']:.3f} handover {s['handover']:.2f} fallback {s['fallback']:.2f} unsafe_h {s['unsafe_handover']:.4f} unsafe_fb {s['unsafe_fallback']:.4f} answers {s['answers_mean']:.2f} ret/VU {s['ret_over_VU_any']}")
    res["seconds"] = round(time.perf_counter() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} in {res['seconds']} s")


if __name__ == "__main__":
    main()
