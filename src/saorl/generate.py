"""Raw candidate generation over the frozen DSL grids (plan v12 2.3-2.4, p.4).

This produces the raw pool Psi_raw^DSL that the language-ensemble judge
(saorl.judge) and Algorithm 1 then score and filter. Grids are frozen *before*
generation. The generator covers the maintenance instantiation; an
`ingest_proposals` hook lets LLM-proposed candidates enter the same pool so they
are scored and audited identically.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional

from .dsl import Atom, Candidate, Persist, Predicate

# Frozen maintenance grids (plan p.4).
THETA_R = (10, 15, 20, 25, 30)      # RUL_hat <= theta_R
THETA_Q = (5, 10, 15, 20)           # Q05    <= theta_Q
THETA_A = (0.7, 0.8, 0.9)           # anom   >= theta_A
PERSIST_M = (2, 3, 5)               # Persist_m windows
FORBID_SETS = (
    frozenset({"continue"}),
    frozenset({"continue", "minor_repair"}),
)


def _atoms() -> List[Predicate]:
    preds: List[Predicate] = []
    preds += [Atom("rul_hat", "le", t) for t in THETA_R]
    preds += [Atom("q05", "le", t) for t in THETA_Q]
    preds += [Atom("anom", "ge", t) for t in THETA_A]
    return preds


def generate_maintenance_pool(
    include_persist: bool = True,
    forbid_sets=FORBID_SETS,
) -> List[Candidate]:
    """Enumerate the raw maintenance candidate pool (no plausibility scores yet)."""
    preds: List[Predicate] = list(_atoms())
    if include_persist:
        # Persist_m only over RUL_hat thresholds (the persistence interpretation).
        preds += [
            Persist(m, Atom("rul_hat", "le", t))
            for m, t in product(PERSIST_M, THETA_R)
        ]

    cands: List[Candidate] = []
    seen = set()
    for g, B in product(preds, forbid_sets):
        name = f"{g!r} => not in {{{','.join(sorted(B))}}}"
        if name in seen:
            continue
        seen.add(name)
        cands.append(Candidate(name=name, predicate=g, forbidden_actions=B))
    return cands


def ingest_proposals(
    proposals: List[Candidate],
    pool: Optional[List[Candidate]] = None,
) -> List[Candidate]:
    """Merge expert/LLM-proposed candidates into the pool, de-duplicating by name."""
    pool = list(pool) if pool else []
    have = {c.name for c in pool}
    for c in proposals:
        if c.name not in have:
            pool.append(c)
            have.add(c.name)
    return pool


def pool_summary(pool: List[Candidate]) -> Dict[str, int]:
    n_persist = sum(1 for c in pool if c.predicate.n_temporal() > 0)
    return dict(total=len(pool), temporal=n_persist, atomic=len(pool) - n_persist)


if __name__ == "__main__":
    pool = generate_maintenance_pool()
    print("raw maintenance candidate pool:", pool_summary(pool))
    for c in pool[:6]:
        print("  ", c.name)
    print("   ...")
