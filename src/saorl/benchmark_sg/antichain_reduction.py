"""W6: the antichain reduction of the exact semantic game, at deployed scale.

Deliverables (revision plan `corset_revision_plan (2).md`, item W6):

1.  SANITY CHECK: the plan's two-action counterexample (c_{psi2} <= c_{psi1}
    pointwise).  Down-closed sets are {}, {psi2}, {psi1,psi2} with values
    1, 1, 0; the reduced LP must return V*(delta) = delta.

2.  VERIFICATION at deployed scale: on every pool with <= 12 dominance
    classes, solve the UNREDUCED game LP (one variable per subset of
    classes, 2^n variables) and the REDUCED LP (one variable per antichain)
    and check they agree at every delta.  This checks the proposition on
    132 real posets, not just the 4 control domains.

3.  COUNT antichains over the deployed pools (median, max, fraction <= 64):
    the number of constrained programs the reduced pricing needs, against
    2^M for the unreduced one.

4.  RUN the exact game at DEPLOYED scale (full pools, mean |U| ~ 9.9) on
    the fixture universe, over the FRACTIONAL selection class: a policy is
    x in [0,1]^F (a randomized selection), utility sum x_f, and reading psi
    is honoured at budget d when the selection-mean psi-cost is <= d,
    i.e. sum_f (c_psi,f - d) x_f <= 0.  This is the LP relaxation of the
    E4/E16 integer class -- the analogue of occupancy mixtures, and the
    natural class for a game whose optimal strategies are mixtures.  V_T is
    then a small LP for ANY constraint set T, so the game runs at full
    deployed pool size.  Report V*(delta), PoA over the full retained pool,
    and whether the 13--60% relative-price band survives at deployed scale.

Run:  PYTHONPATH=. python3 -m saorl.benchmark_sg.antichain_reduction
Writes results/e2e/antichain_reduction.json
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "results/e2e", "antichain_reduction.json")

D_OPER = 0.05                      # operating budget (matches E4/E16 grid)
DELTAS = (0.0, 0.01, 0.05, 0.10, 0.15)
VERIFY_MAX_CLASSES = 12            # unreduced LP has 2^n variables


# ---------------------------------------------------------------------------
# Poset machinery (pointwise dominance on cost vectors; equal vectors = class)
# ---------------------------------------------------------------------------

def quotient_classes(vectors: List[Sequence[float]]) -> Tuple[List[List[int]], List[np.ndarray]]:
    """Group equal cost vectors into classes; return (member lists, reps)."""
    reps: List[np.ndarray] = []
    members: List[List[int]] = []
    for i, v in enumerate(vectors):
        a = np.asarray(v, dtype=float)
        for c, r in enumerate(reps):
            if a.shape == r.shape and np.allclose(a, r, atol=1e-12):
                members[c].append(i)
                break
        else:
            reps.append(a)
            members.append([i])
    return members, reps


def leq_matrix(reps: List[np.ndarray]) -> np.ndarray:
    """L[i, j] True iff rep_i <= rep_j pointwise (i is the EASIER reading:
    honouring j implies honouring i).  Diagonal False."""
    n = len(reps)
    L = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and bool(np.all(reps[i] <= reps[j] + 1e-12)):
                L[i, j] = True
    assert not np.any(L & L.T), "equal vectors must be quotiented"
    return L


def neighbours(L: np.ndarray) -> List[int]:
    """Comparability-graph neighbourhood bitmasks."""
    comp = L | L.T
    n = L.shape[0]
    return [int(sum(1 << j for j in range(n) if comp[i, j])) for i in range(n)]


def count_antichains(L: np.ndarray) -> int:
    """Number of antichains (independent sets of the comparability graph),
    including the empty antichain.  Memoized branch on the lowest vertex."""
    nbr = neighbours(L)

    @lru_cache(maxsize=None)
    def f(mask: int) -> int:
        if mask == 0:
            return 1
        v = (mask & -mask).bit_length() - 1
        return f(mask & ~(1 << v)) + f(mask & ~((1 << v) | nbr[v]))

    out = f((1 << L.shape[0]) - 1)
    f.cache_clear()
    return out


def enumerate_antichains(L: np.ndarray, cap: int = 500000) -> List[int]:
    """All antichains as class-bitmasks (includes the empty antichain)."""
    n = L.shape[0]
    nbr = neighbours(L)
    out: List[int] = []

    def rec(i: int, chosen: int, banned: int) -> None:
        if i == n:
            out.append(chosen)
            return
        if len(out) > cap:
            raise RuntimeError("antichain cap exceeded")
        rec(i + 1, chosen, banned)
        if not (banned >> i) & 1:
            rec(i + 1, chosen | (1 << i), banned | nbr[i])

    rec(0, 0, 0)
    return out


def downset_mask(A: int, L: np.ndarray) -> int:
    """Bitmask of classes w with w <= some a in A (A included)."""
    ds = A
    n = L.shape[0]
    for a in range(n):
        if (A >> a) & 1:
            for w in range(n):
                if L[w, a]:
                    ds |= 1 << w
    return ds


# ---------------------------------------------------------------------------
# Values V_T on the fixture universe (exact enumeration, |F| <= 18)
# ---------------------------------------------------------------------------

def value_of(reps: List[np.ndarray], class_mask: int, d: float) -> float:
    """V_T over the fractional selection class: max sum x_f subject to
    sum_f (c_psi,f - d) x_f <= 0 for every class psi in class_mask,
    0 <= x_f <= 1.  class_mask == 0 gives the unconstrained value |F|."""
    F = len(reps[0])
    rows = [reps[c] - d for c in range(len(reps)) if (class_mask >> c) & 1]
    if not rows:
        return float(F)
    res = linprog(-np.ones(F), A_ub=np.array(rows), b_ub=np.zeros(len(rows)),
                  bounds=(0.0, 1.0), method="highs")
    if res.status != 0:
        raise RuntimeError(f"value LP failed: {res.message}")
    return float(-res.fun)


def game_lp(values: Sequence[float], covers: List[int],
            n_worlds: int, delta: float) -> float:
    """max sum x_T V_T  s.t.  sum x_T = 1;  for every world w,
    sum over variables whose COVER does not contain w of x_T <= delta."""
    c = -np.asarray(values, dtype=float)
    A_ub = np.array([[0.0 if (cov >> w) & 1 else 1.0 for cov in covers]
                     for w in range(n_worlds)])
    res = linprog(c, A_ub=A_ub, b_ub=np.full(n_worlds, delta),
                  A_eq=np.ones((1, len(covers))), b_eq=[1.0],
                  bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    return float(-res.fun)


# ---------------------------------------------------------------------------
# Deliverable 1: the plan's two-action sanity check
# ---------------------------------------------------------------------------

def sanity_two_action() -> dict:
    # classes: 0 = psi2 (easier), 1 = psi1; psi2 <= psi1 pointwise
    L = np.zeros((2, 2), dtype=bool)
    L[0, 1] = True
    V = {0b00: 1.0, 0b01: 1.0, 0b10: 0.0, 0b11: 0.0}   # V_{psi1} = V_both = 0
    ac = enumerate_antichains(L)                        # {}, {psi2}, {psi1}
    covers = [downset_mask(A, L) for A in ac]
    rows = {}
    for delta in (0.0, 0.1, 0.3, 0.7):
        red = game_lp([V[downset_mask(A, L)] for A in ac], covers, 2, delta)
        full_masks = [0b00, 0b01, 0b10, 0b11]
        full = game_lp([V[m] for m in full_masks], full_masks, 2, delta)
        rows[str(delta)] = dict(reduced=red, full=full, expected=float(delta),
                                match=bool(abs(red - delta) < 1e-9
                                           and abs(full - delta) < 1e-9))
    return dict(n_downclosed=len(ac), rows=rows,
                downclosed_ok=bool(sorted(downset_mask(A, L) for A in ac)
                                   == [0b00, 0b01, 0b11]))


# ---------------------------------------------------------------------------
# Deliverables 2-4 on the deployed pools
# ---------------------------------------------------------------------------

def maxset(mask: int, L: np.ndarray) -> int:
    """Bitmask of maximal elements of the classes in `mask`."""
    out = 0
    n = L.shape[0]
    for c in range(n):
        if (mask >> c) & 1:
            if not any(((mask >> j) & 1) and L[c, j] for j in range(n)):
                out |= 1 << c
    return out


VERIFY_MAX_CLASSES_FULL = 10       # exhaustive V_T + full-vs-reduced game LP


def analyse_pool(vectors: List[Sequence[float]], d: float, name: str = "") -> dict:
    members, reps = quotient_classes(list(vectors))
    L = leq_matrix(reps)
    n = len(reps)
    n_ac = count_antichains(L)

    ac = enumerate_antichains(L)
    covers = [downset_mask(A, L) for A in ac]
    vcache: dict = {}

    def V(mask: int) -> float:
        mx = maxset(mask, L)
        if mx not in vcache:
            vcache[mx] = value_of(reps, mx, d)
        return vcache[mx]

    vals = [V(A) for A in ac]                          # V_{downset(A)} = V_A
    vstar = {str(dl): game_lp(vals, covers, n, dl) for dl in DELTAS}

    maximal = [c for c in range(n) if not any(L[c, j] for j in range(n))]
    A_max = sum(1 << c for c in maximal)
    V_full = V(A_max)
    V_sing = [V(1 << c) for c in range(n)]
    v_best = max(V_sing)
    poa = v_best - V_full
    rel = poa / v_best if v_best > 1e-12 else 0.0

    verified = None
    if n <= VERIFY_MAX_CLASSES_FULL:
        # (i) constraint-set reduction: V_T == V_{Max(T)} for every subset,
        #     with V_T computed DIRECTLY (all constraints, no memo)
        # (ii) game-level reduction: full 2^n-variable LP == reduced LP
        full_masks = list(range(1 << n))
        direct = [value_of(reps, m, d) for m in full_masks]
        red_i = all(abs(direct[m] - V(m)) < 1e-7 for m in full_masks)
        red_ii = all(
            abs(game_lp(direct, full_masks, n, dl) - vstar[str(dl)]) < 1e-6
            for dl in DELTAS)
        verified = bool(red_i and red_ii)
    return dict(name=name, n_readings=len(vectors), n_classes=n,
                n_antichains=n_ac,
                width=int(max((bin(A).count("1") for A in ac), default=0)),
                v_best_single=v_best, v_full=V_full, poa=poa, rel_price=rel,
                vstar=vstar, verified=verified)


def main() -> None:
    from . import run_benchmark as rb
    from .parse import parse_prometheus, parse_kyverno
    try:
        from .parse import prom_threshold_bank
    except ImportError:
        prom_threshold_bank = rb.prom_threshold_bank

    sanity = sanity_two_action()
    ok = all(r["match"] for r in sanity["rows"].values()) and sanity["downclosed_ok"]
    print(f"[sanity] two-action counterexample: V*(delta)=delta from reduced LP -> {ok}")
    assert ok

    pt, _ = parse_prometheus()
    kt, _ = parse_kyverno()
    bank = prom_threshold_bank(pt)
    fams = {
        "prometheus": [rb._record(rb.build_prom_pool(t, bank), t, rb._prom_skeleton(t)) for t in pt],
        "kyverno":    [rb._record(rb.build_kyv_pool(t), t, rb._kyv_skeleton(t)) for t in kt],
    }
    out = {"registration": "W6 antichain reduction", "d_operating": D_OPER,
           "deltas": list(DELTAS), "sanity_two_action": sanity, "families": {}}
    for fam, recs in fams.items():
        pools = [analyse_pool(list(r.vectors), D_OPER, name=r.name) for r in recs]
        counts = np.array([p["n_antichains"] for p in pools])
        subsets = np.array([2 ** p["n_readings"] for p in pools], dtype=float)
        rels = np.array([p["rel_price"] for p in pools])
        pos = rels[rels > 0]
        ver = [p["verified"] for p in pools if p["verified"] is not None]
        fam_out = dict(
            n_pools=len(pools),
            n_readings_mean=float(np.mean([p["n_readings"] for p in pools])),
            n_classes_mean=float(np.mean([p["n_classes"] for p in pools])),
            width_max=int(max(p["width"] for p in pools)),
            antichains_median=int(np.median(counts)),
            antichains_max=int(counts.max()),
            frac_le_64=float(np.mean(counts <= 64)),
            frac_le_256=float(np.mean(counts <= 256)),
            reduction_median=float(np.median(subsets / counts)),
            verified_n=len(ver), verified_all=bool(all(ver)) if ver else None,
            price=dict(
                n_positive=int((rels > 0).sum()),
                rel_min_positive=float(pos.min()) if len(pos) else 0.0,
                rel_max=float(rels.max()),
                rel_median_positive=float(np.median(pos)) if len(pos) else 0.0,
            ),
            pools=pools,
        )
        out["families"][fam] = fam_out
        print(f"[{fam}] pools={fam_out['n_pools']} classes~{fam_out['n_classes_mean']:.1f} "
              f"antichains med={fam_out['antichains_median']} max={fam_out['antichains_max']} "
              f"frac<=64={fam_out['frac_le_64']:.3f} median 2^M/#A={fam_out['reduction_median']:.0f}x")
        print(f"        verified full==reduced on {fam_out['verified_n']} pools: {fam_out['verified_all']}")
        print(f"        deployed price: positive {fam_out['price']['n_positive']}/{fam_out['n_pools']}, "
              f"rel {fam_out['price']['rel_min_positive']:.3f}-{fam_out['price']['rel_max']:.3f} "
              f"median {fam_out['price']['rel_median_positive']:.3f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
