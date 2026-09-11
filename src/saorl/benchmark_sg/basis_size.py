"""V44: decision-sufficient basis size on the fixture-level free class.

For each pool U (Prometheus 88, Kyverno 44) and budget d: K = |U|, A = size
of the non-dominated antichain, greedy basis by constraint generation, and
the exact minimum basis s(U) = min{|C| : F_C(d) subseteq Pi_U(d)} by brute
force over subsets of the antichain (a dominated reading is implied by its
dominator, so a minimum basis lies in the antichain).  Registration V44.

Run: OMP_NUM_THREADS=1 SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.basis_size
Writes results/e2e/basis_size.json
"""
from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .query_loop import free_value, OP, ROOT, build_kyv_pool, build_prom_pool, parse_kyverno, parse_prometheus, prom_threshold_bank

OUT = os.path.join(ROOT, "results/e2e", "basis_size.json")
BUDGETS = (OP, 0.005)
TOL = 1e-9


class Pool:
    def __init__(self, vecs, d):
        self.v = np.asarray(vecs, float); self.d = d; self.K, self.F = self.v.shape; self.w = self.v - d; self.ones = np.ones(self.F); self._vc = {}

    def value(self, C):
        key = frozenset(C)
        if key not in self._vc:
            self._vc[key] = free_value([self.v[i] for i in C], self.d)
        return self._vc[key]

    def worst_violation(self, C):
        """max over omitted phi of max{w_phi'z : w_k'z<=0 (k in C), 1'z = V_C}; (phi, W)."""
        VC = self.value(C); worst = (None, -np.inf)
        rows = np.atleast_2d(self.w[sorted(C)])
        for j in range(self.K):
            if j in C or any(np.all(self.v[k] >= self.v[j] - 1e-12) for k in C):
                continue                       # dominated by a member: implied
            cons = [LinearConstraint(rows, -np.inf, 0.0), LinearConstraint(self.ones[None, :], VC, VC)]
            res = milp(c=-self.w[j], constraints=cons, integrality=np.ones(self.F), bounds=Bounds(0, 1))
            W = -res.fun if res.success else np.inf
            if W > worst[1]:
                worst = (j, W)
        return worst

    def sufficient(self, C):
        j, W = self.worst_violation(C)
        return j is None or W <= TOL

    def antichain(self):
        return [k for k in range(self.K) if not any(kk != k and np.all(self.v[kk] >= self.v[k] - 1e-12) and np.any(self.v[kk] > self.v[k] + 1e-12) for kk in range(self.K))]

    def greedy(self):
        C = {min(range(self.K), key=lambda k: (self.value([k]), k))}
        while True:
            j, W = self.worst_violation(C)
            if j is None or W <= TOL:
                return sorted(C)
            C.add(j)

    def exact_min(self, cap=6):
        A = self.antichain()
        for size in range(1, min(len(A), cap) + 1):
            for C in itertools.combinations(A, size):
                if self.sufficient(set(C)):
                    return list(C)
        return None


def main():
    t0 = time.time(); rows = []
    pt, _ = parse_prometheus(); kt, _ = parse_kyverno(); bank = prom_threshold_bank(pt)
    for fam, targets, builder in (("prometheus", pt, lambda t: build_prom_pool(t, bank)), ("kyverno", kt, build_kyv_pool)):
        for ti, t in enumerate(targets):
            pool = builder(t); vecs = [tuple(c.vector) for c in pool.classes]
            if len(vecs) < 2:
                continue
            uid = f'{getattr(t, "name", getattr(t, "policy_name", "?"))}#{ti}'
            for d in BUDGETS:
                P = Pool(vecs, d); A = P.antichain(); g = P.greedy(); ex = P.exact_min()
                V_U = P.value(range(P.K)); singleton = any(P.value([k]) == V_U and P.sufficient({k}) for k in range(P.K))
                rows.append(dict(uid=uid, family=fam, d=d, K=P.K, A=len(A), greedy=len(g), exact=(len(ex) if ex is not None else None),
                                 singleton_sufficient=bool(singleton), V_U=V_U, greedy_set=g, exact_set=ex))
            print(f"  [{fam}] {uid:44s} " + " ".join(f"d={r['d']}: K={r['K']} A={r['A']} greedy={r['greedy']} exact={r['exact']}" for r in rows[-2:]), flush=True)
    agg = {}
    for fam in ("prometheus", "kyverno"):
        for d in BUDGETS:
            R = [r for r in rows if r["family"] == fam and r["d"] == d]; need = [r for r in R if not r["singleton_sufficient"]]
            ex = [r["exact"] for r in need if r["exact"] is not None]
            agg[f"{fam}@{d}"] = dict(n=len(R), need_protection=len(need), singleton=len(R) - len(need),
                                    K_median=float(np.median([r["K"] for r in R])), A_median=float(np.median([r["A"] for r in R])), A_max=max(r["A"] for r in R),
                                    exact_median_need=(float(np.median(ex)) if ex else None),
                                    exact_equals_antichain=sum(1 for r in need if r["exact"] == r["A"]),
                                    exact_below_antichain=sum(1 for r in need if r["exact"] is not None and r["exact"] < r["A"]),
                                    greedy_equals_exact=sum(1 for r in need if r["greedy"] == r["exact"]),
                                    ratio_exact_over_A_median=(float(np.median([r["exact"] / r["A"] for r in need if r["exact"]])) if need else None))
    payload = dict(registration="V44 (REGISTRATION_V28.md)", budgets=list(BUDGETS), aggregate=agg, rows=rows, runtime_s=time.time() - t0)
    json.dump(payload, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} in {time.time()-t0:.0f}s")
    for k, a in agg.items():
        print(k, json.dumps(a))


if __name__ == "__main__":
    main()
