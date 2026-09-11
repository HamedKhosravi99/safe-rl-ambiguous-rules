"""V14: the capability ladder and the admission family, recomputed at the
FACE level (REGISTRATION_V14.md).

In the class a predicate set A defines (fingerprint groups g, sizes n_g,
violation masses m[k, g]), reading psi is FACE-SUFFICIENT at budget d iff
every z maximizing n'z under (m[psi] - d n)'z <= 0 satisfies every
competing row. Certificate: one MILP per competitor,

    W = max { (m[phi] - d n)'z : (m[psi] - d n)'z <= 0, n'z = V_psi },

face-sufficient iff all W <= 0; the dominance shortcut (m[psi] >= m[phi]
pointwise) certifies phi with no MILP. Structural assert everywhere:
value-fire implies face-need (the face criterion only grows the need-set).

Run: PYTHONPATH=. SAORL_CONFORMAL=1 python3 -m saorl.benchmark_sg.face_ladder
Writes results/e2e/face_ladder.json
"""
from __future__ import annotations

import itertools
import json
import os
from typing import List

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .class_ladder import _grouped, _value
from .parse import parse_prometheus, parse_kyverno, prom_threshold_bank
from .evaluate import build_prom_pool, build_kyv_pool

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARCHIVE = os.path.join(ROOT, "results/e2e", "policy_class_budget.json")
OUT = os.path.join(ROOT, "results/e2e", "face_ladder.json")

OPERATING = 0.05
SUBSET_CAP = 60
TOL = 1e-9


def _face_need(n_g, m, K, d) -> dict:
    """Value verdict and face verdict for the class (n_g, m) at budget d."""
    rows_all = list(range(K))
    V_U = _value(n_g, m, rows_all, d)
    V = [_value(n_g, m, [k], d) for k in range(K)]
    value_fires = min(v - V_U for v in V) > 0
    if value_fires:
        return dict(value_fires=True, face_need=True, face_sufficient=None)
    W_con = m - d * n_g[None, :]
    for k in range(K):
        if V[k] != V_U:
            continue
        ok = True
        for kk in range(K):
            if kk == k:
                continue
            if np.all(m[k] >= m[kk] - 1e-12):
                continue                      # dominance shortcut
            A = np.vstack([W_con[k], n_g, -n_g])
            ub = np.array([0.0, V[k], -V[k]])
            res = milp(c=-W_con[kk],
                       constraints=LinearConstraint(A, -np.inf, ub),
                       integrality=np.ones(len(n_g)), bounds=Bounds(0, 1))
            if not res.success or -res.fun > TOL:
                ok = False
                break
        if ok:
            return dict(value_fires=False, face_need=False, face_sufficient=k)
    return dict(value_fires=False, face_need=True, face_sufficient=None)


def _face_need_fixture(vecs: np.ndarray, d: float) -> dict:
    """Value and face verdicts in the fixture-level free class."""
    K, F = vecs.shape
    w = vecs - d
    ones = np.ones(F)

    def solve(rows, eq_val=None):
        A = [np.atleast_2d(rows)] if rows is not None else []
        lo, hi = [], []
        cons = []
        if rows is not None:
            cons.append(LinearConstraint(np.atleast_2d(rows), -np.inf, 0.0))
        if eq_val is not None:
            cons.append(LinearConstraint(ones[None, :], eq_val, eq_val))
        res = milp(c=-ones if eq_val is None else -w[solve.obj],
                   constraints=cons, integrality=np.ones(F),
                   bounds=Bounds(0, 1))
        return res

    def value(rows):
        cons = [LinearConstraint(np.atleast_2d(w[list(rows)]), -np.inf, 0.0)]
        res = milp(c=-ones, constraints=cons, integrality=np.ones(F),
                   bounds=Bounds(0, 1))
        return int(round(-res.fun)) if res.success else 0

    V_U = value(range(K))
    V = [value([k]) for k in range(K)]
    value_fires = min(v - V_U for v in V) > 0
    if value_fires:
        return dict(value_fires=True, face_need=True, V_U=V_U)
    for k in range(K):
        if V[k] != V_U:
            continue
        ok = True
        for kk in range(K):
            if kk == k:
                continue
            if np.all(vecs[k] >= vecs[kk] - 1e-12):
                continue
            cons = [LinearConstraint(w[k][None, :], -np.inf, 0.0),
                    LinearConstraint(ones[None, :], V[k], V[k])]
            res = milp(c=-w[kk], constraints=cons,
                       integrality=np.ones(F), bounds=Bounds(0, 1))
            if not res.success or -res.fun > TOL:
                ok = False
                break
        if ok:
            return dict(value_fires=False, face_need=False, V_U=V_U)
    return dict(value_fires=False, face_need=True, V_U=V_U)


def main() -> None:
    with open(ARCHIVE, encoding="utf8") as fh:
        arch = json.load(fh)
    free_rows = {(r["rule_id"], r["family"], round(r["budget"], 6)): r
                 for r in arch["rows"] if r["policy_class"] == "free"}
    compiled_uids = {r["rule_id"] for r in arch["rows"]
                     if r["policy_class"] == "monitoring_compiled"}

    pt, _ = parse_prometheus()
    kt, _ = parse_kyverno()
    bank = prom_threshold_bank(pt)

    pools_out: List[dict] = []
    for fam, targets, builder in (
            ("prometheus", pt, lambda t: build_prom_pool(t, bank)),
            ("kyverno", kt, build_kyv_pool)):
        for ti, t in enumerate(targets):
            pool = builder(t)
            uid = f'{getattr(t, "name", getattr(t, "policy_name", "?"))}#{ti}'
            vecs = np.asarray([np.asarray(c.vector, float) for c in pool.classes])
            K, F = vecs.shape
            rungs = {}
            ks = sorted({0, 1, 2} & set(range(K + 1)))
            for k in ks:
                fires_some = False
                all_pairs_fire = True
                n_sub = 0
                for A in itertools.islice(
                        itertools.combinations(range(K), k), SUBSET_CAP):
                    n_g, m = _grouped(vecs, A)
                    cell = _face_need(n_g, m, K, OPERATING)
                    if cell["value_fires"]:
                        assert cell["face_need"], (uid, k, A)
                    fires_some |= cell["face_need"]
                    all_pairs_fire &= cell["face_need"]
                    n_sub += 1
                rungs[k] = dict(fires_some=fires_some,
                                all_fire=bool(all_pairs_fire and n_sub),
                                n_subsets=n_sub)
            # free class at fixture level, anchored against the archive
            fr = _face_need_fixture(vecs, OPERATING)
            ar = free_rows.get((uid, fam, round(OPERATING, 6)))
            if ar is not None:
                assert ar["V_U"] == fr["V_U"], (uid, ar["V_U"], fr["V_U"])
                assert ar["screen_fires"] == fr["value_fires"], uid
            rungs[K] = dict(fires_some=fr["face_need"],
                            all_fire=fr["face_need"], n_subsets=1)
            ks = ks + [K] if K not in ks else ks
            front = None
            for k in ks:
                if rungs[k]["fires_some"]:
                    front = k
                    break
            pools_out.append(dict(
                uid=uid, family=fam, K=K, rungs={str(k): rungs[k] for k in ks},
                free_face_need=rungs[K]["fires_some"] if K in rungs else None,
                frontier=front, in_compiled_common=(uid in compiled_uids),
                non_monotone=bool(any(rungs[k]["fires_some"]
                                      for k in ks if 0 < k < K)
                                  and not rungs[K]["fires_some"])))

    def agg(fam):
        P = [p for p in pools_out if p["family"] == fam]
        multi = [p for p in P if p["K"] >= 2]
        return dict(
            n_pools=len(P),
            k0_fires=sum(p["rungs"]["0"]["fires_some"] for p in P),
            k1_fires=sum(p["rungs"].get("1", {}).get("fires_some", False)
                         for p in P),
            k2_some=sum(p["rungs"].get("2", {}).get("fires_some", False)
                        for p in multi),
            k2_all=sum(p["rungs"].get("2", {}).get("all_fire", False)
                       for p in multi),
            free_fires=sum(bool(p["free_face_need"]) for p in P),
            frontier_one=sum(p["frontier"] == 1 for p in P),
            frontier_two=sum(p["frontier"] == 2 for p in P),
            frontier_never=sum(p["frontier"] is None for p in P),
            non_monotone=sum(p["non_monotone"] for p in P),
            common_free_fires=sum(bool(p["free_face_need"]) for p in P
                                  if p["in_compiled_common"]),
            common_n=sum(p["in_compiled_common"] for p in P))

    payload = dict(
        registration="V14 (REGISTRATION_V14.md): face-level ladder + "
                     "admission face pass at the operating budget",
        operating=OPERATING, subset_cap=SUBSET_CAP,
        aggregate={fam: agg(fam) for fam in ("prometheus", "kyverno")},
        pools=pools_out)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(payload, fh, indent=1)
    print(json.dumps(payload["aggregate"], indent=1))


if __name__ == "__main__":
    main()
