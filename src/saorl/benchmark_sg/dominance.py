"""Source-grounded benchmark, step 6: policy-class dominance and nu_Pi (plan 11).

Operates on the executable cost/fire vectors of a merged pool (evaluate.Pool).
The fixture universe plays the role of the reachable (state, action) set, so
pointwise dominance on fixtures is the executable form of policy-class
dominance psi_i <=_Pi psi_j  <=>  J_{c_psi_i}(pi) <= J_{c_psi_j}(pi) for all pi
(plan 11.4): it is sufficient in general and necessary under occupancy-richness,
which the boundary-hitting fixture battery is built to approximate.

For a pool with cost vectors {c_psi in R^F}:

  * collapse (a dominating member exists) iff  max_psi c_psi = c_{psi*}  pointwise
    for some member psi*  (plan 11.4 proposition);
  * maximal non-dominated readings = members not strictly dominated by another;
  * maximal-antichain width = largest mutually-incomparable set (Dilworth:
    n - max bipartite matching on the strict-dominance relation);
  * nu_Pi(U) = min_psi sup_pi [ max_phi J_phi(pi) - J_psi(pi) ].  Over a free
    policy class on the fixture universe the sup is attained at a single fixture,
    giving the exact executable value  min_psi max_f [ max_phi c_phi(f) - c_psi(f) ]
    (plan 11.5, "executable lower-bound approximation ... from test cases").
    nu_Pi = 0 exactly iff the pool collapses.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def _leq(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y + 1e-12 for x, y in zip(a, b))


def dominates_strict(a: Sequence[float], b: Sequence[float]) -> bool:
    """b strictly dominates a: b >= a componentwise and b != a."""
    return _leq(a, b) and any(y > x + 1e-12 for x, y in zip(a, b))


def maximal_indices(vectors: Sequence[Sequence[float]]) -> List[int]:
    """Indices not strictly dominated by any other member (the maximal set)."""
    n = len(vectors)
    out = []
    for i in range(n):
        if not any(j != i and dominates_strict(vectors[i], vectors[j]) for j in range(n)):
            out.append(i)
    return out


def dominating_member(vectors: Sequence[Sequence[float]]) -> int:
    """Index of a member whose cost >= every member pointwise (collapse witness),
    or -1 if none exists."""
    m = len(vectors[0])
    pmax = [max(v[f] for v in vectors) for f in range(m)]
    for i, v in enumerate(vectors):
        if all(abs(v[f] - pmax[f]) <= 1e-12 for f in range(m)):
            return i
    return -1


def _max_bipartite_matching(adj: List[List[int]], n: int) -> int:
    """Kuhn's algorithm: max matching in bipartite graph left=right={0..n-1}."""
    match_r = [-1] * n

    def try_kuhn(u, seen):
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                if match_r[v] == -1 or try_kuhn(match_r[v], seen):
                    match_r[v] = u
                    return True
        return False

    res = 0
    for u in range(n):
        if try_kuhn(u, [False] * n):
            res += 1
    return res


def max_antichain(vectors: Sequence[Sequence[float]]) -> int:
    """Maximum antichain size (Dilworth's theorem).

    We build the full (transitive) strict-dominance DAG, so its minimum chain
    cover = n - (maximum bipartite matching), and by Dilworth the maximum
    antichain equals the minimum chain cover.  Hence width = n - max_matching."""
    n = len(vectors)
    if n <= 1:
        return n
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and dominates_strict(vectors[i], vectors[j]):
                adj[i].append(j)
    matching = _max_bipartite_matching(adj, n)
    return n - matching


def nu_pi(vectors: Sequence[Sequence[float]]) -> float:
    """Executable nu_Pi over the free policy class on the fixture universe."""
    n = len(vectors)
    if n <= 1:
        return 0.0
    m = len(vectors[0])
    pmax = [max(v[f] for v in vectors) for f in range(m)]
    best = float("inf")
    for v in vectors:
        gap = max(pmax[f] - v[f] for f in range(m))
        best = min(best, gap)
    return float(best)


def pool_dominance(vectors: Sequence[Sequence[float]]) -> dict:
    """All plan-3.5 / 11.5 dominance outputs for one pool."""
    n = len(vectors)
    maximal = maximal_indices(vectors)
    dom_idx = dominating_member(vectors)
    width = max_antichain(vectors)
    nu = nu_pi(vectors)
    return dict(
        n_classes=n,
        n_maximal=len(maximal),
        maximal_idx=maximal,
        has_dominating_member=(dom_idx >= 0),
        dominating_idx=dom_idx,
        antichain_width=width,
        collapses=(dom_idx >= 0),                 # collapse == dominating member (nu_Pi==0)
        frac_pruned=(0.0 if n == 0 else (n - len(maximal)) / n),
        nu_pi=nu,
        # "genuinely non-dominated" (Stage 2 is necessary) == NO dominating member
        # == nu_Pi > 0 (plan 11.4/11.5).  This is distinct from merely having a
        # size-2 antichain, which can occur even when a dominating member exists.
        genuinely_non_dominated=(dom_idx < 0),
        has_ge2_incomparable=(width >= 2),
    )


def restricted_dominance(vectors: Sequence[Sequence[float]], keep_idx: Sequence[int]) -> dict:
    """Dominance metrics restricted to a subset of classes (e.g. the retained
    conformal set U) -- for the plan-3.5 'at least two retained readings are
    genuinely incomparable' output."""
    sub = [vectors[i] for i in keep_idx]
    if not sub:
        return dict(n=0, antichain_width=0, genuinely_non_dominated=False, nu_pi=0.0,
                    has_dominating_member=False)
    return dict(n=len(sub),
                antichain_width=max_antichain(sub),
                genuinely_non_dominated=(max_antichain(sub) >= 2),
                nu_pi=nu_pi(sub),
                has_dominating_member=(dominating_member(sub) >= 0))
