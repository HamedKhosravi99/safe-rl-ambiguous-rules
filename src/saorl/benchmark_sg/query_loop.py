"""REGISTRATION_V26: what one question buys.

Oracle-answered fixture queries on three archived populations (compiled
monitoring instances, admission free class, monitoring free class), three
selection rules (random, balanced split, largest expected value after the
answer), curves at q = 0, 1, 2, 4.

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.query_loop
Writes results/e2e/query_loop.json
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .baseline_table import (OP, ROOT, _R, _eligible, _lp, build_kyv_pool, build_prom_pool,
                             compile_instance, parse_kyverno, parse_prometheus, prom_threshold_bank)

OUT = os.path.join(ROOT, "results/e2e", "query_loop.json")
QS = (0, 1, 2, 4)
QMAX = max(QS)
R_RANDOM = 20
SEED = 0
TOL = 1e-9


def free_value(vecs, d=OP) -> int:
    reps = np.asarray(vecs, dtype=float)
    F = reps.shape[1]
    res = milp(c=-np.ones(F), constraints=LinearConstraint(reps - d, -np.inf, 0.0),
               integrality=np.ones(F), bounds=Bounds(0, 1))
    return int(round(-res.fun)) if res.success else 0


class Instance:
    """A priced set: items with fixture verdict vectors; value(S) for any subset S."""

    def __init__(self, uid, vectors, gold, value_fn):
        self.uid = uid
        self.vec = [tuple(v) for v in vectors]
        self.gold = gold
        self._value_fn = value_fn
        self._cache = {}
        self.n_fix = len(self.vec[0])

    def value(self, S):
        key = frozenset(S)
        if key not in self._cache:
            self._cache[key] = self._value_fn(sorted(key))
        return self._cache[key]

    def splits(self, S):
        out = []
        for f in range(self.n_fix):
            vals = {self.vec[i][f] for i in S}
            if len(vals) > 1:
                out.append(f)
        return out

    def parts(self, S, f):
        groups = {}
        for i in S:
            groups.setdefault(self.vec[i][f], []).append(i)
        return groups

    def answer(self, S, f):
        return [i for i in S if self.vec[i][f] == self.vec[self.gold][f]]


def pick(inst, S, rule, rng):
    sp = inst.splits(S)
    if not sp:
        return None
    if rule == "random":
        return rng.choice(sp)
    if rule == "split":
        return min(sp, key=lambda f: (max(len(g) for g in inst.parts(S, f).values()), f))
    if rule == "split_rand":
        # post-hoc robustness variant (not registered): ties among the most balanced fixtures broken at random
        best = min(max(len(g) for g in inst.parts(S, f).values()) for f in sp)
        return rng.choice([f for f in sp if max(len(g) for g in inst.parts(S, f).values()) == best])
    if rule == "poa":
        def ev(f):
            g = inst.parts(S, f)
            return sum(len(p) / len(S) * inst.value(p) for p in g.values())
        return max(sp, key=lambda f: (ev(f), -f))
    raise ValueError(rule)


def run_rule(inst, rule, rng):
    S = list(range(len(inst.vec)))
    vals = [inst.value(S)]
    sizes = [len(S)]
    for _q in range(QMAX):
        f = pick(inst, S, rule, rng)
        if f is not None:
            S = inst.answer(S, f)
        vals.append(inst.value(S))
        sizes.append(len(S))
    return vals, sizes


def curve(inst):
    v_gold = inst.value([inst.gold])
    v0 = inst.value(list(range(len(inst.vec))))
    price = v_gold - v0
    out = dict(uid=inst.uid, n_items=len(inst.vec), n_fixtures=inst.n_fix, V0=v0, V_gold=v_gold, price=price,
               price_rel=(price / v_gold if v_gold > 0 else 0.0), rules={})
    for rule in ("split", "poa"):
        vals, sizes = run_rule(inst, rule, None)
        out["rules"][rule] = dict(values=vals, sizes=sizes)
    for rule in ("random", "split_rand"):
        rng = random.Random(SEED)
        acc = np.zeros(QMAX + 1); szs = np.zeros(QMAX + 1); res1 = np.zeros(QMAX + 1)
        for _ in range(R_RANDOM):
            vals, sizes = run_rule(inst, rule, rng)
            acc += vals; szs += sizes; res1 += [s == 1 for s in sizes]
        out["rules"][rule] = dict(values=list(acc / R_RANDOM), sizes=list(szs / R_RANDOM), resolved_frac=list(res1 / R_RANDOM))
    for rule, r in out["rules"].items():
        r["recovered"] = [((v - v0) / price if price > TOL else None) for v in r["values"]]
        if "resolved_frac" not in r:
            r["resolved_frac"] = [float(s == 1) for s in r["sizes"]]
    return out


def compiled_instances(pcb):
    pt, _ = parse_prometheus()
    bank = prom_threshold_bank(pt)
    verd = {(r["rule_id"], round(r["budget"], 6)): r for r in pcb["rows"]
            if r["policy_class"] == "monitoring_compiled" and r["status"] == "compiled"}
    out = []
    for ti, t in enumerate(pt):
        pool = build_prom_pool(t, bank)
        ok, _why, readings = _eligible(pool)
        if not ok:
            continue
        rid = f"{t.name}#{ti}"
        if (rid, OP) not in verd:
            continue
        vec_of = {id(c.rep): tuple(c.vector) for c in pool.classes}
        gold_cls = pool.classes[pool.gold_idx]
        gold = gold_cls.rep
        keys = sorted({(float(r.threshold), float(r.for_s)) for r in readings})
        by_key = {(float(r.threshold), float(r.for_s)): r for r in readings}
        gold_key = (float(gold.threshold), float(gold.for_s))
        gold_in = gold_key in keys
        comp_readings = [by_key[k] for k in keys]
        if not gold_in:
            g = _R(); g.threshold, g.for_s = gold_key
            comp_readings = comp_readings + [g]
        m = compile_instance(comp_readings)
        allk = sorted({(float(r.threshold), float(r.for_s)) for r in comp_readings})
        pos = {kk: i for i, kk in enumerate(allk)}
        items = list(keys) + ([] if gold_in else [gold_key])
        vectors = [vec_of[id(by_key[k])] if k in by_key else tuple(gold_cls.vector) for k in items]
        if gold_in:
            # the recorded reading may sit in a different class than the maximal reading with its key;
            # use the recorded class's own verdicts for the oracle
            gi = items.index(gold_key)
            vectors[gi] = tuple(gold_cls.vector)
        rows = [m["C"][pos[k]] for k in items]
        r = m["r"]

        def value_fn(S, rows=rows, r=r, m=m):
            v, _ = _lp(m, r, [rows[i] for i in S], [OP] * len(S))
            return float(v)
        inst = Instance(rid, vectors, items.index(gold_key), value_fn)
        v_u = inst.value(list(range(len(keys))))
        assert abs(v_u - verd[(rid, OP)]["V_U"]) < 1e-6, (rid, v_u, verd[(rid, OP)]["V_U"])
        sig = hashlib.sha1(json.dumps([items, [list(v) for v in vectors], gold_key]).encode()).hexdigest()[:12]
        c = curve(inst)
        c.update(K=len(keys), gold_in_maximal=gold_in, n_states=m["nS"], signature=sig,
                 geometry=f"L{m['L']}-caps{'-'.join(map(str, m['caps']))}")
        out.append(c)
        print(f"[compiled] {rid} K={len(keys)} price={c['price']:.4f} rec1 split={c['rules']['split']['recovered'][1]} poa={c['rules']['poa']['recovered'][1]}", flush=True)
    return out


def free_instances(pcb, family):
    if family == "kyverno":
        targets, _ = parse_kyverno()
        pools = [(f"{t.policy_name}/{t.rule_name}#{ti}", build_kyv_pool(t)) for ti, t in enumerate(targets)]
    else:
        pt, _ = parse_prometheus()
        bank = prom_threshold_bank(pt)
        pools = [(f"{t.name}#{ti}", build_prom_pool(t, bank)) for ti, t in enumerate(pt)]
    free = {r["rule_id"]: r for r in pcb["rows"] if r["policy_class"] == "free" and r["family"] == family
            and abs(r["budget"] - OP) < 1e-9}
    out = []
    for rid, pool in pools:
        vectors = [tuple(c.vector) for c in pool.classes]
        if len(vectors) < 2:
            continue
        inst = Instance(rid, vectors, pool.gold_idx, lambda S, vectors=vectors: free_value([vectors[i] for i in S]))
        idx = rid.rsplit("#", 1)[1]
        cand = [k for k in free if k.endswith(f"#{idx}")]
        if cand:
            arch = free[cand[0]]
            assert inst.value(list(range(len(vectors)))) == arch["V_U"], (rid, arch["V_U"])
        sig = hashlib.sha1(json.dumps([sorted(vectors), list(vectors[pool.gold_idx])]).encode()).hexdigest()[:12]
        c = curve(inst)
        c.update(K=len(vectors), signature=sig, archived=bool(cand))
        out.append(c)
        print(f"[{family}] {rid} K={len(vectors)} price={c['price']} rec1 split={c['rules']['split']['recovered'][1]} poa={c['rules']['poa']['recovered'][1]}", flush=True)
    return out


def summarize(rows):
    n = len(rows)
    pos = [r for r in rows if r["price"] > TOL]
    sigs = {}
    for r in pos:
        sigs.setdefault(r["signature"], []).append(r)
    out = dict(n=n, n_positive_price=len(pos), n_zero_price=n - len(pos), n_distinct_positive=len(sigs), rules={})
    for rule in ("random", "split", "split_rand", "poa"):
        rec = np.array([[r["rules"][rule]["recovered"][q] for q in QS] for r in pos]) if pos else np.zeros((0, len(QS)))
        inst_rec = np.array([[np.mean([r["rules"][rule]["recovered"][q] for r in g]) for q in QS] for g in sigs.values()]) if sigs else np.zeros((0, len(QS)))
        resolved = np.array([[r["rules"][rule]["resolved_frac"][q] for q in QS] for r in rows])
        out["rules"][rule] = dict(
            q=list(QS),
            recovered_mean=[float(x) for x in rec.mean(axis=0)] if len(rec) else None,
            recovered_median=[float(x) for x in np.median(rec, axis=0)] if len(rec) else None,
            recovered_instance_mean=[float(x) for x in inst_rec.mean(axis=0)] if len(inst_rec) else None,
            resolved_frac=[float(x) for x in resolved.mean(axis=0)],
            full_recovery_at_1=float(np.mean(rec[:, 1] >= 1 - 1e-9)) if len(rec) else None)
    if pos:
        d1 = [r["rules"]["poa"]["recovered"][1] - r["rules"]["split"]["recovered"][1] for r in pos]
        out["poa_vs_split_at_1"] = dict(poa_better=int(sum(x > 1e-9 for x in d1)), equal=int(sum(abs(x) <= 1e-9 for x in d1)),
                                        split_better=int(sum(x < -1e-9 for x in d1)), mean_diff=float(np.mean(d1)))
        d1r = [r["rules"]["poa"]["recovered"][1] - r["rules"]["random"]["recovered"][1] for r in pos]
        out["poa_vs_random_at_1"] = dict(poa_better=int(sum(x > 1e-9 for x in d1r)), mean_diff=float(np.mean(d1r)))
        out["poa_beats_split_pool_mean_at_1"] = bool(out["rules"]["poa"]["recovered_mean"][1] > out["rules"]["split"]["recovered_mean"][1] + 1e-9)
        d1s = [r["rules"]["poa"]["recovered"][1] - r["rules"]["split_rand"]["recovered"][1] for r in pos]
        out["poa_vs_split_rand_at_1"] = dict(poa_better=int(sum(x > 1e-9 for x in d1s)), equal=int(sum(abs(x) <= 1e-9 for x in d1s)),
                                             split_rand_better=int(sum(x < -1e-9 for x in d1s)), mean_diff=float(np.mean(d1s)))
        out["poa_beats_split_rand_pool_mean_at_1"] = bool(out["rules"]["poa"]["recovered_mean"][1] > out["rules"]["split_rand"]["recovered_mean"][1] + 1e-9)
    return out


def main():
    t0 = time.perf_counter()
    pcb = json.load(open(os.path.join(ROOT, "results/e2e", "policy_class_budget.json")))
    res = dict(registration="REGISTRATION_V26.md", operating_budget=OP, qs=list(QS), random_draws=R_RANDOM, seed=SEED, populations={})
    for name, fn in (("compiled", lambda: compiled_instances(pcb)),
                     ("admission", lambda: free_instances(pcb, "kyverno")),
                     ("monitoring_free", lambda: free_instances(pcb, "prometheus"))):
        rows = fn()
        res["populations"][name] = dict(summary=summarize(rows), rows=rows)
        print(name, json.dumps(res["populations"][name]["summary"], indent=None), flush=True)
    wins = sum(1 for p in res["populations"].values() if p["summary"].get("poa_beats_split_pool_mean_at_1"))
    wins_r = sum(1 for p in res["populations"].values() if p["summary"].get("poa_beats_split_rand_pool_mean_at_1"))
    res["branch"] = dict(populations_where_poa_beats_split=wins, claim_licensed=bool(wins >= 2),
                         populations_where_poa_beats_split_rand=wins_r,
                         post_hoc_note="split_rand (random tie-break among the most balanced fixtures) was added after the registered run as a robustness variant; the registered comparison is poa vs split")
    res["seconds"] = round(time.perf_counter() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res["branch"]), f"wrote {OUT} in {res['seconds']} s")


if __name__ == "__main__":
    main()
