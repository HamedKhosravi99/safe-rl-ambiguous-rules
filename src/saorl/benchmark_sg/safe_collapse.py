"""V43/V46: questions until SAFE DECISION-REGION collapse.

Reuses the V26 oracle-query loop (Instance, answer, pick) and adds, per
surviving set S_t, three endpoints -- identification (|S_t| = 1), value
collapse (some psi has V_psi = V_S) and SAFE collapse (some psi is
face-sufficient for S_t: every psi-optimal policy is feasible for every
reading still in S_t) -- plus a witness-guided selection rule.
Registration V43 in REGISTRATION_V28.md.

Run: OMP_NUM_THREADS=1 SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.safe_collapse
Writes results/e2e/safe_collapse.json
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .query_loop import (Instance, pick, free_value, QMAX, R_RANDOM, SEED, TOL, OP, ROOT, _R, _eligible, _lp,
                         build_kyv_pool, build_prom_pool, compile_instance, parse_kyverno, parse_prometheus,
                         prom_threshold_bank)
from .ambiguity_profile import lp as occ_lp

OUT = os.path.join(ROOT, "results/e2e", "safe_collapse.json")
FACE_TOL = 1e-7


# ---------------------------------------------------------------------------
# face tests on a surviving subset S; return status and the witness pair
# ---------------------------------------------------------------------------
class FixtureFace:
    """Fixture-level free class: value = max #fixtures accepted s.t. every
    constraint row's violation rate <= d; face test by MILP per competitor."""

    def __init__(self, vecs, d):
        self.vecs = np.asarray(vecs, float); self.d = d; self.F = self.vecs.shape[1]
        self.w = self.vecs - d; self.ones = np.ones(self.F); self._vc: Dict[frozenset, int] = {}; self._fc: Dict[frozenset, dict] = {}
        self._W: Dict[Tuple[int, int], float] = {}

    def pair(self, i, j) -> float:
        """W[i,j] = max{ w_j'z : w_i'z <= 0, 1'z = V_i } - 0: excess of phi_j's
        violation over the exact face of psi_i (<= 0 means phi_j in R_i)."""
        key = (i, j)
        if key not in self._W:
            if i == j or np.all(self.vecs[i] >= self.vecs[j] - 1e-12):
                self._W[key] = -1.0
            else:
                Vi = self.value([i])
                cons = [LinearConstraint(self.w[i][None, :], -np.inf, 0.0), LinearConstraint(self.ones[None, :], Vi, Vi)]
                res = milp(c=-self.w[j], constraints=cons, integrality=np.ones(self.F), bounds=Bounds(0, 1))
                self._W[key] = float(-res.fun) if res.success else np.inf
        return self._W[key]

    def regions(self, S):
        """{psi: frozenset of phi in S with W[psi,phi] <= TOL} for every psi in S."""
        S = sorted(S)
        return {i: frozenset(j for j in S if self.pair(i, j) <= TOL) for i in S}

    def value(self, S) -> int:
        key = frozenset(S)
        if key not in self._vc:
            self._vc[key] = free_value([self.vecs[i] for i in S], self.d)
        return self._vc[key]

    def face(self, S) -> dict:
        key = frozenset(S)
        if key in self._fc:
            return self._fc[key]
        S = sorted(S); VS = self.value(S)
        tied = [i for i in S if self.value([i]) == VS]
        out = dict(status="irreducible", anchor=None, violator=None, W=None)
        if tied:
            best = None
            for i in tied:
                worst = (None, -np.inf)
                for j in S:
                    if j == i or np.all(self.vecs[i] >= self.vecs[j] - 1e-12):
                        continue
                    cons = [LinearConstraint(self.w[i][None, :], -np.inf, 0.0),
                            LinearConstraint(self.ones[None, :], VS, VS)]
                    res = milp(c=-self.w[j], constraints=cons, integrality=np.ones(self.F), bounds=Bounds(0, 1))
                    W = -res.fun if res.success else np.inf
                    if W > worst[1]:
                        worst = (j, W)
                if worst[0] is None or worst[1] <= TOL:
                    out = dict(status="sufficient", anchor=i, violator=None, W=0.0); break
                if best is None or worst[1] < best[2]:
                    best = (i, worst[0], worst[1])
            else:
                out = dict(status="resolvable", anchor=best[0], violator=best[1], W=float(best[2]))
        self._fc[key] = out
        return out


class CompiledFace:
    """Compiled monitoring controller: occupancy LPs on the compiled model."""

    def __init__(self, m, rows, d):
        self.m = m; self.rows = rows; self.d = d; self.r = m["r"].reshape(-1); self._vc = {}; self._fc = {}; self._W = {}

    def pair(self, i, j) -> float:
        key = (i, j)
        if key not in self._W:
            if i == j:
                self._W[key] = -1.0
            else:
                ci = self.rows[i].reshape(-1); cj = self.rows[j].reshape(-1); Vi = self.value([i])
                W = occ_lp(self.m, cj, [ci, -self.r], [self.d, -(Vi - FACE_TOL)])
                self._W[key] = np.inf if W is None else float(W - self.d)
        return self._W[key]

    def regions(self, S):
        S = sorted(S)
        return {i: frozenset(j for j in S if self.pair(i, j) <= 1e-6) for i in S}

    def value(self, S) -> float:
        key = frozenset(S)
        if key not in self._vc:
            v, _ = _lp(self.m, self.m["r"], [self.rows[i] for i in S], [self.d] * len(S))
            self._vc[key] = float(v)
        return self._vc[key]

    def face(self, S) -> dict:
        key = frozenset(S)
        if key in self._fc:
            return self._fc[key]
        S = sorted(S); VS = self.value(S)
        tied = [i for i in S if abs(self.value([i]) - VS) <= 1e-7]
        out = dict(status="irreducible", anchor=None, violator=None, W=None)
        if tied:
            best = None
            for i in tied:
                ci = self.rows[i].reshape(-1); Vi = self.value([i]); worst = (None, -np.inf)
                for j in S:
                    if j == i:
                        continue
                    cj = self.rows[j].reshape(-1)
                    W = occ_lp(self.m, cj, [ci, -self.r], [self.d, -(Vi - FACE_TOL)])
                    W = np.inf if W is None else W - self.d
                    if W > worst[1]:
                        worst = (j, W)
                if worst[0] is None or worst[1] <= 1e-6:
                    out = dict(status="sufficient", anchor=i, violator=None, W=0.0); break
                if best is None or worst[1] < best[2]:
                    best = (i, worst[0], worst[1])
            else:
                out = dict(status="resolvable", anchor=best[0], violator=best[1], W=float(best[2]))
        self._fc[key] = out
        return out


# ---------------------------------------------------------------------------
# the loop with endpoints
# ---------------------------------------------------------------------------
def pick_witness(inst, S, face, rng):
    """poa restricted to fixtures separating the value-tied anchor from its worst violator."""
    if face["status"] != "resolvable":
        return pick(inst, S, "poa", rng)
    a, b = face["anchor"], face["violator"]
    sp = [f for f in inst.splits(S) if inst.vec[a][f] != inst.vec[b][f]]
    if not sp:
        return pick(inst, S, "poa", rng)

    def ev(f):
        g = inst.parts(S, f)
        return sum(len(p) / len(S) * inst.value(p) for p in g.values())
    return max(sp, key=lambda f: (ev(f), -f))


def hyperedges(S, regions):
    """Minimal subsets of S contained in no region (bitmask enumeration)."""
    idx = {h: b for b, h in enumerate(S)}; n = len(S)
    reg = [sum(1 << idx[h] for h in R if h in idx) for R in set(regions.values())]
    covered = lambda m: any(m & ~r == 0 for r in reg)
    edges = []
    for size in range(2, n + 1):
        found = False
        for combo in __import__("itertools").combinations(range(n), size):
            m = 0
            for b in combo:
                m |= 1 << b
            if covered(m):
                continue
            if all(covered(m & ~(1 << b)) for b in combo):
                edges.append(combo); found = True
        if not found and size > max([len(e) for e in edges], default=1):
            break                              # minimal uncovered sets have size <= #regions
    return edges


def pick_hec(inst, S, tester, rng, p0, regions_full=None):
    """Greedy hyperedge cutting: fixture maximizing the expected prior weight of
    minimal uncovered subsets it cuts (uniform prior p0 over the initial pool).
    Regions are FIXED over the initial pool (decision region determination);
    each is intersected with the surviving set."""
    sp = inst.splits(S)
    if not sp:
        return None
    S_sorted = sorted(S); Sset = set(S)
    regions = {k: frozenset(R & Sset) for k, R in regions_full.items()}
    edges = hyperedges(S_sorted, regions)
    if not edges:
        return None
    w = [p0 ** len(e) for e in edges]

    def gain(f):
        parts = inst.parts(S, f); tot = 0.0
        for e, we in zip(edges, w):
            hs = [S_sorted[b] for b in e]; answers = {inst.vec[h][f] for h in hs}
            if len(answers) == 1:
                y = next(iter(answers)); tot += we * (1.0 - len(parts[y]) / len(S))
            else:
                tot += we
        return tot
    return max(sp, key=lambda f: (gain(f), -f))


def relaxed_cover(S, regions_full, tester):
    """Decision-region coverage with FIXED regions: some psi in the initial pool
    (surviving or not) whose region contains S. Returns (covered, loss) where
    loss = V_S - max V_psi over covering anchors (0 when a surviving anchor covers)."""
    Sset = set(S); cov = [k for k, R in regions_full.items() if Sset <= R]
    if not cov:
        return False, None
    VS = tester.value(S); best = max(tester.value([k]) for k in cov)
    return True, float(VS - best)


def run_rule(inst, tester, rule, rng, p0=None, regions_full=None):
    S = list(range(len(inst.vec)))
    stat = [tester.face(S)["status"]]; sizes = [len(S)]
    rc, loss = relaxed_cover(S, regions_full, tester); relaxed = [rc]; losses = [loss]
    for _q in range(QMAX):
        face = tester.face(S)
        f = (pick_witness(inst, S, face, rng) if rule == "witness" else
             pick_hec(inst, S, tester, rng, p0, regions_full) if rule == "hec" else pick(inst, S, rule, rng))
        if f is not None:
            S = inst.answer(S, f)
        stat.append(tester.face(S)["status"]); sizes.append(len(S))
        rc, loss = relaxed_cover(S, regions_full, tester); relaxed.append(rc); losses.append(loss)
    first = lambda pred: next((t for t, ok in enumerate(pred) if ok), None)
    t_relaxed = first(relaxed)
    return dict(sizes=sizes, status=stat, relaxed=relaxed,
                t_id=first([s == 1 for s in sizes]),
                t_value=first([s != "irreducible" for s in stat]),
                t_safe=first([s == "sufficient" for s in stat]),
                t_relaxed=t_relaxed, loss_at_relaxed=(losses[t_relaxed] if t_relaxed is not None else None))


def curve(inst, tester):
    out = dict(uid=inst.uid, K=len(inst.vec), n_fixtures=inst.n_fix, status0=tester.face(list(range(len(inst.vec))))["status"],
               price=float(inst.value([inst.gold]) - inst.value(list(range(len(inst.vec))))), rules={})
    p0 = 1.0 / len(inst.vec)
    regs = tester.regions(list(range(len(inst.vec)))); out["k_regions"] = len(set(regs.values()))
    for rule in ("split", "poa", "witness", "hec"):
        out["rules"][rule] = run_rule(inst, tester, rule, None, p0, regs)
    rng = random.Random(SEED); draws = [run_rule(inst, tester, "random", rng, p0, regs) for _ in range(R_RANDOM)]
    out["rules"]["random"] = dict(draws=draws)
    return out


def summarize(rows):
    qs = list(range(QMAX + 1)); out = dict(n=len(rows), status0={}, rules={})
    for s in ("sufficient", "resolvable", "irreducible"):
        out["status0"][s] = sum(1 for r in rows if r["status0"] == s)
    pos = [r for r in rows if r["status0"] != "sufficient"]
    out["n_not_safe_at_0"] = len(pos)

    def frac(ts, q):
        return float(np.mean([(t is not None and t <= q) for t in ts])) if ts else None
    out["k_regions_median"] = float(np.median([r["k_regions"] for r in rows])); out["k_regions_max"] = max(r["k_regions"] for r in rows)
    for rule in ("split", "poa", "witness", "hec"):
        tid = [r["rules"][rule]["t_id"] for r in rows]; tsafe = [r["rules"][rule]["t_safe"] for r in rows]; tval = [r["rules"][rule]["t_value"] for r in rows]
        pairs = [(r["rules"][rule]["t_safe"], r["rules"][rule]["t_id"]) for r in pos]
        trel = [r["rules"][rule]["t_relaxed"] for r in rows]
        diff = [(r["rules"][rule]["t_relaxed"], r["rules"][rule]["t_safe"], r["rules"][rule]["loss_at_relaxed"]) for r in rows]
        out["rules"][rule] = dict(q=qs,
                                  safe_by_q=[frac(tsafe, q) for q in qs], identified_by_q=[frac(tid, q) for q in qs], value_by_q=[frac(tval, q) for q in qs],
                                  relaxed_by_q=[frac(trel, q) for q in qs],
                                  relaxed_before_strict=sum(1 for a, b, _ in diff if a is not None and (b is None or a < b)),
                                  relaxed_equals_strict=sum(1 for a, b, _ in diff if a == b),
                                  loss_when_relaxed_only=[l for a, b, l in diff if a is not None and (b is None or a < b)],
                                  safe_before_id=sum(1 for a, b in pairs if a is not None and (b is None or a < b)),
                                  safe_equal_id=sum(1 for a, b in pairs if a is not None and b is not None and a == b),
                                  unresolved_by_qmax=sum(1 for a, _ in pairs if a is None),
                                  median_t_safe=(float(np.median([a for a, _ in pairs if a is not None])) if any(a is not None for a, _ in pairs) else None),
                                  median_t_id=(float(np.median([b for _, b in pairs if b is not None])) if any(b is not None for _, b in pairs) else None))
    # random: average over draws then pools
    safe_by = np.zeros(QMAX + 1); id_by = np.zeros(QMAX + 1)
    for r in rows:
        for dr in r["rules"]["random"]["draws"]:
            safe_by += [(dr["t_safe"] is not None and dr["t_safe"] <= q) for q in qs]
            id_by += [(dr["t_id"] is not None and dr["t_id"] <= q) for q in qs]
    n = max(1, len(rows) * R_RANDOM)
    out["rules"]["random"] = dict(q=qs, safe_by_q=list(safe_by / n), identified_by_q=list(id_by / n))
    return out


def fixture_population(family):
    if family == "kyverno":
        targets, _ = parse_kyverno(); pools = [(f"{t.policy_name}/{t.rule_name}#{ti}", build_kyv_pool(t)) for ti, t in enumerate(targets)]
    else:
        pt, _ = parse_prometheus(); bank = prom_threshold_bank(pt); pools = [(f"{t.name}#{ti}", build_prom_pool(t, bank)) for ti, t in enumerate(pt)]
    out = []
    for rid, pool in pools:
        vectors = [tuple(c.vector) for c in pool.classes]
        if len(vectors) < 2:
            continue
        tester = FixtureFace(vectors, OP)
        inst = Instance(rid, vectors, pool.gold_idx, lambda S, tester=tester: tester.value(S))
        c = curve(inst, tester); out.append(c)
        print(f"  [{family}] {rid:44s} K={c['K']:2d} status0={c['status0']:11s} poa t_safe={c['rules']['poa']['t_safe']} t_id={c['rules']['poa']['t_id']}", flush=True)
    return out


def compiled_population(d):
    pt, _ = parse_prometheus(); bank = prom_threshold_bank(pt); out = []
    for ti, t in enumerate(pt):
        pool = build_prom_pool(t, bank); ok, _why, readings = _eligible(pool)
        if not ok:
            continue
        rid = f"{t.name}#{ti}"
        vec_of = {id(c.rep): tuple(c.vector) for c in pool.classes}
        gold_cls = pool.classes[pool.gold_idx]; gold = gold_cls.rep
        keys = sorted({(float(r.threshold), float(r.for_s)) for r in readings}); by_key = {(float(r.threshold), float(r.for_s)): r for r in readings}
        gold_key = (float(gold.threshold), float(gold.for_s)); gold_in = gold_key in keys
        comp = [by_key[k] for k in keys]
        if not gold_in:
            g = _R(); g.threshold, g.for_s = gold_key; comp = comp + [g]
        m = compile_instance(comp)
        allk = sorted({(float(r.threshold), float(r.for_s)) for r in comp}); pos = {kk: i for i, kk in enumerate(allk)}
        items = list(keys) + ([] if gold_in else [gold_key])
        vectors = [vec_of[id(by_key[k])] if k in by_key else tuple(gold_cls.vector) for k in items]
        if gold_in:
            vectors[items.index(gold_key)] = tuple(gold_cls.vector)
        rows = [m["C"][pos[k]] for k in items]
        tester = CompiledFace(m, rows, d)
        inst = Instance(rid, vectors, items.index(gold_key), lambda S, tester=tester: tester.value(S))
        c = curve(inst, tester); c.update(K_retained=len(keys), gold_in_maximal=gold_in); out.append(c)
        print(f"  [compiled d={d}] {rid:40s} K={c['K']} status0={c['status0']:11s} poa t_safe={c['rules']['poa']['t_safe']} t_id={c['rules']['poa']['t_id']}", flush=True)
    return out


def main():
    t0 = time.perf_counter()
    res = dict(registration="V43 (REGISTRATION_V28.md)", qmax=QMAX, random_draws=R_RANDOM, populations={})
    for name, fn in (("admission", lambda: fixture_population("kyverno")),
                     ("monitoring_free", lambda: fixture_population("prometheus")),
                     ("compiled_d0.05", lambda: compiled_population(OP)),
                     ("compiled_d0.005", lambda: compiled_population(0.005))):
        rows = fn(); res["populations"][name] = dict(summary=summarize(rows), rows=rows)
        print(name, json.dumps(res["populations"][name]["summary"]), flush=True)
    res["seconds"] = round(time.perf_counter() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} in {res['seconds']} s")
    for name, p in res["populations"].items():
        s = p["summary"]; print(f"\n== {name}: n={s['n']} status0={s['status0']} not-safe-at-0={s['n_not_safe_at_0']}")
        print(f"   k_regions median {s['k_regions_median']} max {s['k_regions_max']}")
        for rule in ("split", "poa", "witness", "hec", "random"):
            r = s["rules"][rule]; sb = [f"{x:.2f}" for x in r["safe_by_q"]]; ib = [f"{x:.2f}" for x in r["identified_by_q"]]
            extra = (f" safe<id {r['safe_before_id']} safe=id {r['safe_equal_id']} unresolved {r['unresolved_by_qmax']} med t_safe {r['median_t_safe']} med t_id {r['median_t_id']}"
                     f" | relaxed_by_q {[round(x,2) for x in r['relaxed_by_q']]} relaxed<strict {r['relaxed_before_strict']} relaxed=strict {r['relaxed_equals_strict']} losses {r['loss_when_relaxed_only'][:5]}") if rule != "random" else ""
            print(f"  {rule:8s} safe_by_q {sb} identified_by_q {ib}{extra}")


if __name__ == "__main__":
    main()
