"""SAFE-KEEP: a candidate may be removed only if the retained set provably
implies it. ADDITIVE experimental arm; the existing KEEP is untouched.

SOUNDNESS (proved in the report):
  phi >= psi pointwise  =>  Pi_phi(d) subset Pi_psi(d)          (monotone J)
  U covers Psi          =>  Pi_U(d) = Pi_Psi(d)                 (both inclusions)
  U_max = maximal elements of Psi covers Psi                    (finite preorder)
  conic: c_psi <= sum_phi lam_phi c_phi, lam>=0, sum lam <= 1   => implied

SYNTACTIC DOMINANCE for upper-comparator threshold/duration readings: a fires
whenever b fires (hence c_a >= c_b in any faithful compiled model) when
    theta_a <= theta_b  AND  for_a <= for_b  AND  (cmp_a == cmp_b or cmp_a == '>=')
Lower threshold fires more; shorter duration fires more; '>=' fires at least as
often as '>' at equal threshold. This is a SUFFICIENT condition, deliberately
conservative: when it does not hold we keep the candidate.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog

TOL = 1e-12


def syn_dominates(a, b) -> bool:
    """a >= b : a fires whenever b fires (upper comparators only)."""
    if a.comparator not in (">", ">=") or b.comparator not in (">", ">="):
        return False
    if float(a.threshold) > float(b.threshold) + TOL: return False
    if float(a.for_s)     > float(b.for_s) + TOL:     return False
    if a.comparator == b.comparator: return True
    return a.comparator == ">="          # '>=' dominates '>' at equal threshold


def cost_dominates(ca, cb) -> bool:
    return bool((ca >= cb - 1e-12).all())


def maximal_cover(items, dom) -> list:
    """Indices of maximal elements under preorder `dom` (one per tie class)."""
    n = len(items); keep = []
    for i in range(n):
        strictly_dominated = any(
            j != i and dom(items[j], items[i]) and not dom(items[i], items[j])
            for j in range(n))
        if strictly_dominated: continue
        # among mutual-tie groups keep the first representative only
        if any(j < i and dom(items[j], items[i]) and dom(items[i], items[j])
               for j in range(n)):
            continue
        keep.append(i)
    return keep


def conic_implied(c_psi: np.ndarray, C_U: list, tol=1e-9):
    """Is c_psi <= sum lam_phi c_phi pointwise for some lam>=0, sum lam <= 1?
    Feasibility LP. Returns (bool, lam or None)."""
    if not C_U: return False, None
    m = len(C_U)
    A = np.stack([c.reshape(-1) for c in C_U], axis=1)      # (n_sa, m)
    b = c_psi.reshape(-1)
    # -A lam <= -b  (i.e. A lam >= b) ;  sum lam <= 1 ; lam >= 0
    A_ub = np.vstack([-A, np.ones((1, m))])
    b_ub = np.concatenate([-b, [1.0]])
    res = linprog(np.zeros(m), A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * m,
                  method="highs")
    if res.status != 0: return False, None
    lam = res.x
    ok = bool((A @ lam >= b - tol).all() and lam.sum() <= 1 + tol and (lam >= -tol).all())
    return ok, (lam if ok else None)


def safe_keep_maximal(readings) -> list:
    return maximal_cover(readings, syn_dominates)


def safe_keep_conic(readings, costs) -> list:
    """Greedy elimination: drop a candidate only if the CURRENT retained set
    conically implies it. Correctness is never traded for size."""
    idx = list(range(len(readings)))
    changed = True
    while changed:
        changed = False
        for i in list(idx):
            rest = [j for j in idx if j != i]
            if not rest: continue
            ok, _ = conic_implied(costs[i], [costs[j] for j in rest])
            if ok:
                idx = rest; changed = True; break
    return idx


def safe_keep_score(readings, scores, qhat) -> list:
    """Score-pruned first, then RESTORE whatever is needed for the cover
    invariant. Score may order, never delete without a certificate."""
    keep = [i for i, s in enumerate(scores) if s >= qhat]
    if not keep: keep = [int(np.argmax(scores))]
    changed = True
    while changed:
        changed = False
        for i in range(len(readings)):
            if i in keep: continue
            if not any(syn_dominates(readings[j], readings[i]) for j in keep):
                # uncovered: restore the best-scoring dominator of i, else i itself
                doms = [j for j in range(len(readings))
                        if syn_dominates(readings[j], readings[i])]
                pick = max(doms, key=lambda j: scores[j]) if doms else i
                keep.append(pick); changed = True; break
    return sorted(set(keep))


def assert_cover(readings, keep_idx) -> bool:
    """Invariant: every raw candidate is dominated by some retained one."""
    return all(any(syn_dominates(readings[j], readings[i]) for j in keep_idx)
               for i in range(len(readings)))


# ---------------------------------------------------------------------------
# Reachability-aware dominance (added after the compiler audit)
# ---------------------------------------------------------------------------
def reachable_states(m) -> set:
    """States reachable from supp(mu0) through transition SUPPORT under ANY
    action. Uses support, not occupancy magnitude, so no policy is privileged."""
    P = m["P"]; nS, nA = m["nS"], m["nA"]
    seen = set(np.flatnonzero(m["mu0"] > 0).tolist()); stack = list(seen)
    while stack:
        i = stack.pop()
        for a in range(nA):
            for j in np.flatnonzero(P[i, a] > 0).tolist():
                if j not in seen:
                    seen.add(j); stack.append(j)
    return seen


def reachable_cost_dominates(m, ka: int, kb: int, reach=None) -> bool:
    """c_ka >= c_kb on every REACHABLE (s,a). Sound for J-dominance because
    unreachable states carry zero occupancy under every policy."""
    if reach is None: reach = reachable_states(m)
    idx = sorted(reach)
    return bool((m["C"][ka][idx, :] >= m["C"][kb][idx, :] - 1e-12).all())


# ---------------------------------------------------------------------------
# Corrected semantic dominance on the EXECUTABLE monitor parameters (lvl, cap)
# ---------------------------------------------------------------------------
DUR_CAP_LOCAL = None   # resolved lazily from control_suite


def td_params(readings):
    """Replicate compile_instance's (keyed, thetas, fors, caps, lvl_of) WITHOUT
    building the state space, so raw pools of any K can be analysed.
    Mirrors control_suite.compile_instance exactly, including amendment A3."""
    from .control_suite import DUR_CAP
    keyed = sorted({(float(m.threshold), float(m.for_s)) for m in readings})
    thetas = [k[0] for k in keyed]; fors = [k[1] for k in keyed]
    STEP_S = 60.0
    caps = [max(1, min(DUR_CAP, int(round(f / STEP_S)))) for f in fors]
    uniq_f = sorted(set(fors))
    if len(set(caps)) < len(set(fors)):
        seen = {}
        for i, f in enumerate(fors):
            c = caps[i]
            for g, cg in seen.items():
                if f != g and c == cg:
                    c = max(1, min(DUR_CAP, 1 + uniq_f.index(f)))
            caps[i] = c
            seen[f] = caps[i]
    uniq = sorted(set(thetas))
    lvl_of = [1 + uniq.index(t) for t in thetas]
    return keyed, thetas, fors, caps, lvl_of


def sem_dominates_td(i: int, j: int, caps, lvl_of) -> bool:
    """phi_i >=_sem psi_j  iff  lvl_i <= lvl_j AND cap_i <= cap_j.
    Whenever j fires, i fires (suffix argument). Uses the EXECUTABLE monitor
    parameters, so amendment A3's non-monotone cap remapping cannot break it."""
    return lvl_of[i] <= lvl_of[j] and caps[i] <= caps[j]


def safe_keep_semantic(readings):
    """Maximal elements of the raw pool under >=_sem, on the compiled
    (threshold, duration) semantic universe. Deterministic and data-free.
    Returns indices into the DEDUPLICATED keyed list."""
    keyed, thetas, fors, caps, lvl_of = td_params(readings)
    n = len(keyed); keep = []
    for i in range(n):
        dominated = any(j != i and sem_dominates_td(j, i, caps, lvl_of)
                        and not sem_dominates_td(i, j, caps, lvl_of)
                        for j in range(n))
        if dominated: continue
        if any(j < i and sem_dominates_td(j, i, caps, lvl_of)
               and sem_dominates_td(i, j, caps, lvl_of) for j in range(n)):
            continue
        keep.append(i)
    return keep, keyed, caps, lvl_of
