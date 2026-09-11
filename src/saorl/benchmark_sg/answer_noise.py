"""V49: Refine under erroneous and ambiguous answers.

The V43/V46 loop (hyperedge-cutting rule, safety stop with FIXED regions over
the initial pool) with a noisy oracle. Noise: uniform flips, abstentions,
balance-weighted flips. Protocols: hard elimination, tolerant elimination
(<= 1 inconsistent answer), repeat-3 majority, posterior stopping (correct
and misspecified rate). Registration V49.

Run: OMP_NUM_THREADS=1 SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.answer_noise
Writes results/e2e/answer_noise.json
"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np

from .query_loop import Instance, TOL, OP, ROOT, _R, _eligible, build_kyv_pool, build_prom_pool, compile_instance, parse_kyverno, parse_prometheus, prom_threshold_bank
from .safe_collapse import FixtureFace, CompiledFace, hyperedges

OUT = os.path.join(ROOT, "results/e2e", "answer_noise.json")
P_GRID = (0.0, 0.05, 0.1, 0.2, 0.3)
Q_ABST = (0.1, 0.3)
QMAX = 8
QMAX_REPEAT = 9
DRAWS = 20
DRAWS_UNIFORM = 10
DELTA_POST = 0.05
P_MISSPEC = 0.1
SEED = 20260905
SOFT_FLOOR = 1e-3


# --------------------------------------------------------------------------- pool geometry (fixed regions)
class Pool:
    def __init__(self, inst: Instance, tester, uid):
        self.inst, self.tester, self.uid = inst, tester, uid
        self.K = len(inst.vec); self.all = list(range(self.K))
        self.regions = tester.regions(self.all)                         # {psi: frozenset(phi with W[psi,phi] <= tol)}
        self.V1 = {k: tester.value([k]) for k in self.all}
        self.p0 = 1.0 / self.K
        self.colvals = [sorted({inst.vec[i][f] for i in self.all}) for f in range(inst.n_fix)]

    def covering(self, S):
        Sset = set(S); return [k for k, Rg in self.regions.items() if Sset <= Rg]

    def anchor(self, S):
        cov = self.covering(S)
        return (max(cov, key=lambda k: (self.V1[k], -k)) if cov else None)

    def pick(self, S, asked, rng):
        """Greedy hyperedge cutting over the unasked splitting fixtures (V46 rule)."""
        sp = [f for f in self.inst.splits(S) if f not in asked]
        if not sp:
            return None
        S_sorted = sorted(S); Sset = set(S)
        regions = {k: frozenset(Rg & Sset) for k, Rg in self.regions.items()}
        edges = hyperedges(S_sorted, regions)
        if not edges:
            return None
        w = [self.p0 ** len(e) for e in edges]

        def gain(f):
            parts = self.inst.parts(S, f); tot = 0.0
            for e, we in zip(edges, w):
                hs = [S_sorted[b] for b in e]; answers = {self.inst.vec[h][f] for h in hs}
                if len(answers) == 1:
                    y = next(iter(answers)); tot += we * (1.0 - len(parts[y]) / len(S))
                else:
                    tot += we
            return tot
        return max(sp, key=lambda f: (gain(f), -f))


# --------------------------------------------------------------------------- noisy oracle
def make_oracle(pool: Pool, truth: int, model: str, level: float, rng: random.Random):
    vec = pool.inst.vec

    def oracle(f, S):
        true = vec[truth][f]
        if model == "abstain":
            return (None, False) if rng.random() < level else (true, False)
        if model == "flip":
            pf = level
        elif model == "balflip":
            share = sum(1 for i in S if vec[i][f] == true) / len(S); pf = level * (1.0 - abs(2 * share - 1.0))
        else:
            raise ValueError(model)
        if rng.random() < pf:
            others = [v for v in pool.colvals[f] if v != true]
            return (rng.choice(others), True) if others else (true, False)
        return (true, False)
    return oracle


# --------------------------------------------------------------------------- one run
def run(pool: Pool, truth: int, oracle, protocol: str, p_assumed: float, rng: random.Random):
    inst, K = pool.inst, pool.K
    incons = np.zeros(K, dtype=int); logw = np.zeros(K); asked = set(); answers = 0; flips = 0
    qmax = QMAX_REPEAT if protocol == "repeat3" else QMAX
    contradiction = False; trace = []

    def survivors():
        if protocol == "tolerant1":
            return [i for i in pool.all if incons[i] <= 1]
        if protocol == "bayes":
            mx = logw.max(); return [i for i in pool.all if logw[i] >= mx + np.log(SOFT_FLOOR)]
        return [i for i in pool.all if incons[i] == 0]

    def stopped(S):
        if protocol == "bayes":
            w = np.exp(logw - logw.max()); w /= w.sum()
            best, bm = None, -1.0
            for k, Rg in pool.regions.items():
                mass = float(sum(w[i] for i in Rg))
                if mass > bm or (mass == bm and best is not None and (pool.V1[k], -k) > (pool.V1[best], -best)):
                    best, bm = k, mass
            return (bm >= 1.0 - DELTA_POST), (best if bm >= 1.0 - DELTA_POST else None)
        a = pool.anchor(S); return (a is not None), a

    S = survivors(); safe, anc = stopped(S); trace.append((answers, safe))
    while not safe and answers < qmax:
        f = pool.pick(S, asked, rng)
        if f is None:
            break
        asked.add(f)
        if protocol == "repeat3":
            ys = []
            for _ in range(3):
                y, fl = oracle(f, S); answers += 1; flips += int(fl)
                if y is not None:
                    ys.append(y)
            if not ys:
                S = survivors(); safe, anc = stopped(S); trace.append((answers, safe)); continue
            vals = sorted(set(ys), key=lambda v: (-ys.count(v), rng.random())); y = vals[0]
        else:
            y, fl = oracle(f, S); answers += 1; flips += int(fl)
            if y is None:
                S = survivors(); safe, anc = stopped(S); trace.append((answers, safe)); continue
        m_f = len(pool.colvals[f])
        for i in pool.all:
            agree = inst.vec[i][f] == y
            if not agree:
                incons[i] += 1
            if protocol == "bayes":
                logw[i] += np.log(max(1.0 - p_assumed, 1e-12)) if agree else np.log(max(p_assumed / max(m_f - 1, 1), 1e-12))
        S = survivors()
        if not S:
            contradiction = True; break
        safe, anc = stopped(S); trace.append((answers, safe))
    truth_in = truth in S if S else False
    if protocol == "bayes":
        w = np.exp(logw - logw.max()); w /= w.sum(); truth_in = bool(w[truth] >= DELTA_POST)
    anchor_unsafe = bool(safe and anc is not None and truth not in pool.regions[anc])
    t_safe = next((t for t, s in trace if s), None)
    loss = None
    if safe and anc is not None and S:
        loss = float(pool.tester.value(S) - pool.V1[anc])
    return dict(safe=bool(safe), t_safe=t_safe, answers=answers, flips=flips, truth_in=bool(truth_in), anchor=anc,
                anchor_unsafe_for_truth=anchor_unsafe, contradiction=contradiction, loss=loss, final_size=len(S))


# --------------------------------------------------------------------------- populations (as in V43)
def fixture_pools(family):
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
        out.append(Pool(inst, tester, rid))
    return out


def compiled_pools(d):
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
        out.append(Pool(inst, tester, rid))
    return out


# --------------------------------------------------------------------------- driver
CELLS = ([("flip", p, pr, None) for p in P_GRID for pr in ("hard", "tolerant1", "repeat3")]
         + [("flip", p, "bayes", p) for p in P_GRID] + [("flip", p, "bayes_mis", P_MISSPEC) for p in P_GRID]
         + [("abstain", q, pr, None) for q in Q_ABST for pr in ("hard",)] + [("abstain", q, "bayes", 0.0) for q in Q_ABST]
         + [("balflip", p, pr, None) for p in P_GRID[1:] for pr in ("hard",)] + [("balflip", p, "bayes", p) for p in P_GRID[1:]])


def cell_key(model, level, protocol):
    return f"{model}|{level}|{protocol}"


def summarize(per_pool, n_pools, n_unsafe0):
    """per_pool: {cell: {uid: [run dicts]}}; report pool-averaged rates over all pools and over pools not safe at q=0."""
    out = {}
    for cell, by_uid in per_pool.items():
        def rate(fn, uids):
            vals = [np.mean([fn(r) for r in by_uid[u]]) for u in uids if u in by_uid]
            return float(np.mean(vals)) if vals else None
        all_u = list(by_uid); un_u = [u for u in all_u if u in n_unsafe0]
        stats = {}
        for lab, uids in (("all", all_u), ("unsafe_at_0", un_u)):
            stats[lab] = dict(n_pools=len(uids),
                              safe_by_4=rate(lambda r: r["safe"] and r["t_safe"] is not None and r["t_safe"] <= 4, uids),
                              safe_by_8=rate(lambda r: r["safe"] and r["t_safe"] is not None and r["t_safe"] <= 8, uids),
                              safe_by_9=rate(lambda r: r["safe"] and r["t_safe"] is not None and r["t_safe"] <= 9, uids),
                              truth_eliminated=rate(lambda r: not r["truth_in"], uids),
                              anchor_unsafe_for_truth=rate(lambda r: r["anchor_unsafe_for_truth"], uids),
                              contradiction=rate(lambda r: r["contradiction"], uids),
                              answers_mean=rate(lambda r: r["answers"], uids),
                              flips_mean=rate(lambda r: r["flips"], uids),
                              runs_with_flip=rate(lambda r: r["flips"] > 0, uids),
                              truth_elim_given_flip=(lambda xs: float(np.mean(xs)) if xs else None)([not r["truth_in"] for u in uids if u in by_uid for r in by_uid[u] if r["flips"] > 0]),
                              anchor_unsafe_given_truth_elim=(lambda xs: float(np.mean(xs)) if xs else None)([r["anchor_unsafe_for_truth"] for u in uids if u in by_uid for r in by_uid[u] if not r["truth_in"] and r["safe"]]),
                              loss_mean_when_safe=(lambda xs: float(np.mean(xs)) if xs else None)([r["loss"] for u in uids if u in by_uid for r in by_uid[u] if r["safe"] and r["loss"] is not None]))
        out[cell] = stats
    return out


def main():
    t0 = time.perf_counter()
    res = dict(registration="V49 (REGISTRATION_V28.md)", p_grid=list(P_GRID), q_abstain=list(Q_ABST), qmax=QMAX, qmax_repeat=QMAX_REPEAT, draws=DRAWS,
               delta_post=DELTA_POST, p_misspecified=P_MISSPEC, populations={})
    for name, fn in (("admission", lambda: fixture_pools("kyverno")), ("monitoring_free", lambda: fixture_pools("prometheus")),
                     ("compiled_d0.005", lambda: compiled_pools(0.005))):
        pools = fn(); t1 = time.perf_counter()
        unsafe0 = {p.uid for p in pools if p.anchor(p.all) is None}
        per_pool = {cell_key(m, l, pr): {} for (m, l, pr, _pa) in CELLS}
        per_pool_uniform = {cell_key("flip", p, pr): {} for p in (0.0, 0.1, 0.2) for pr in ("hard", "bayes")}
        for pi_, pool in enumerate(pools):
            truth = pool.inst.gold
            for (model, level, protocol, p_assumed) in CELLS:
                proto = "bayes" if protocol == "bayes_mis" else protocol
                runs = []
                for dr in range(DRAWS):
                    rng = random.Random(hash((SEED, pool.uid, model, level, protocol, dr)) % (2 ** 31))
                    oracle = make_oracle(pool, truth, model, level, rng)
                    runs.append(run(pool, truth, oracle, proto, (p_assumed if p_assumed is not None else 0.0), rng))
                per_pool[cell_key(model, level, protocol)][pool.uid] = runs
            for p in (0.0, 0.1, 0.2):
                for protocol in ("hard", "bayes"):
                    runs = []
                    for dr in range(DRAWS_UNIFORM):
                        rng = random.Random(hash((SEED, pool.uid, "uniform", p, protocol, dr)) % (2 ** 31))
                        tr = rng.randrange(pool.K); oracle = make_oracle(pool, tr, "flip", p, rng)
                        runs.append(run(pool, tr, oracle, protocol, p, rng))
                    per_pool_uniform[cell_key("flip", p, protocol)][pool.uid] = runs
            if pi_ % 10 == 0:
                print(f"  [{name}] {pi_+1}/{len(pools)} pools ({time.perf_counter()-t1:.0f}s)", flush=True)
        summ = summarize(per_pool, len(pools), unsafe0); summ_u = summarize(per_pool_uniform, len(pools), unsafe0)
        res["populations"][name] = dict(n_pools=len(pools), n_unsafe_at_0=len(unsafe0), K_median=float(np.median([p.K for p in pools])),
                                        k_regions_median=float(np.median([len(set(p.regions.values())) for p in pools])),
                                        summary=summ, summary_truth_uniform=summ_u, seconds=round(time.perf_counter() - t1, 1))
        print(f"== {name}: {len(pools)} pools, {len(unsafe0)} not safe at q=0 ({res['populations'][name]['seconds']} s)")
        for cell, st in summ.items():
            s = st["unsafe_at_0"]
            print(f"   {cell:26s} safe@4 {s['safe_by_4']:.2f} safe@8 {s['safe_by_8']:.2f} truth-elim {s['truth_eliminated']:.3f} anchor-unsafe {s['anchor_unsafe_for_truth']:.3f} contradiction {s['contradiction']:.3f} answers {s['answers_mean']:.2f} flips {s['flips_mean']:.2f}")
    res["seconds"] = round(time.perf_counter() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} in {res['seconds']} s")


if __name__ == "__main__":
    main()
