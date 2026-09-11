"""P0-B: the policy-class x budget audit, with tie-safe value-based screens.

For every rule and every reported budget this emits
    rule_id, family, policy_class_id, budget,
    structural_noncollapse, V_U, min_psi V_psi,
    screen_gap = min_psi (V_psi - V_U), screen_fires
and, where a shipped predicate exists, the shipped label so the two can be
compared (the P0-D tie-safety check, applied to every class).

Policy classes
--------------
monitoring_compiled  PRIMARY for monitoring.  The grammar-forced compiled
                     control class: duration fields make monitoring rules
                     stateful, so dominance and sufficiency are evaluated in
                     the compiled load-chain MDP by exact occupancy LPs.
                     Every eligible rule is compiled here -- the per-family
                     and instance caps used for the published 17-instance
                     suite are NOT applied, so the denominator is the full
                     eligible set.
free                 PRIMARY for admission (each admission decision is
                     judged independently, so the free decision-by-decision
                     class is exact) and SENSITIVITY for monitoring.
                     V_A(d) = max |S| over fixture subsets S whose mean
                     cost is <= d under every reading in A, computed by
                     exact enumeration over the fixture universe.

Screen (both classes, tie-safe by construction)
    fires  iff  min_psi [ V_psi(d) - V_U(d) ] > tol.
Because the full-set feasible region is contained in every singleton
region, V_psi >= V_U always; the minimum is 0 exactly when some singleton
attains the robust optimum, and a robust optimum is then also optimal for
that singleton.  The value form cannot be flipped by a solver's choice
among tied optima; a predicate evaluated at one representative optimizer
can be, which is what this module measures.

Unanalyzable rules are reported as such.  A monitoring rule whose maximal
subfamily is a singleton, or has no threshold/duration crossing, provably
cannot fire the compiled screen (its readings are nested in the compiled
model, so one dominates).  A rule excluded only by the suite's size filter
is NOT evidence of clearing and is counted separately as unanalyzed.

Run:  PYTHONPATH=. python3 -m saorl.benchmark_sg.policy_class_budget
Writes results/e2e/policy_class_budget.{json,csv}
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from . import dominance
from .control_suite import _eligible, compile_instance
from .exact_nonnested import analyse as exact_analyse
from .e4_screen import screen_fires as shipped_free_screen
from .parse import parse_prometheus, parse_kyverno, prom_threshold_bank

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT_JSON = os.path.join(ROOT, "results/e2e", "policy_class_budget.json")
OUT_CSV = os.path.join(ROOT, "results/e2e", "policy_class_budget.csv")

BUDGETS = (0.005, 0.01, 0.02, 0.05, 0.10)
OPERATING = 0.05
TOL = 1e-7


# ---------------------------------------------------------------------------
# free class: exact V_A over the fixture universe
# ---------------------------------------------------------------------------

def _greedy_prefix(v: np.ndarray, d: float) -> int:
    """Exact max |S| for ONE reading: sorting costs ascending, the feasible
    sizes are prefix-closed (for a fixed size the cheapest fixtures minimise
    the mean), so the largest feasible prefix is optimal."""
    order = np.sort(np.asarray(v, dtype=float))
    run, best = 0.0, 0
    for m, x in enumerate(order, start=1):
        run += x
        if run / m <= d + 1e-12:
            best = m
    return best


def free_values_multi(vectors: Sequence[Sequence[float]],
                      budgets: Sequence[float]) -> Dict[float, Tuple[int, List[int]]]:
    """{budget: (V_full, [V_k])} in the free selection class, exactly.

    V_A(d) = max |S| over fixture subsets S with mean_{f in S} c_psi(f) <= d
    for every psi in A.  Writing w_psi(f) = c_psi(f) - d, the constraint is
    linear in the indicator x of S, so V_A is the exact optimum of

        max  1'x   s.t.  w_psi' x <= 0  for all psi in A,   x in {0,1}^F,

    solved by HiGHS branch-and-bound (scipy.optimize.milp).  Enumeration is
    not an option here: Prometheus fixture universes reach |F| = 31.
    Singleton values are additionally computed by the exact greedy prefix and
    the two are asserted equal, so the MIP path is checked on every pool."""
    reps = np.asarray([np.asarray(v, dtype=float) for v in vectors])
    n, F = reps.shape
    ones = -np.ones(F)
    integrality = np.ones(F)
    bounds = Bounds(0, 1)

    def solve(rows: np.ndarray) -> int:
        con = LinearConstraint(rows, -np.inf, 0.0)
        res = milp(c=ones, constraints=con, integrality=integrality, bounds=bounds)
        if not res.success:
            return 0
        return int(round(-res.fun))

    out: Dict[float, Tuple[int, List[int]]] = {}
    for d in budgets:
        w = reps - d
        V_k = []
        for c in range(n):
            v_mip = solve(w[c:c + 1])
            v_greedy = _greedy_prefix(reps[c], d)
            assert v_mip == v_greedy, (
                f"singleton value disagreement at d={d}: MIP {v_mip} vs "
                f"greedy prefix {v_greedy}")
            V_k.append(v_mip)
        out[d] = (solve(w), V_k)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    from . import run_benchmark as rb

    pt, _ = parse_prometheus()
    kt, _ = parse_kyverno()
    bank = prom_threshold_bank(pt)
    rows: List[dict] = []

    # ---------------- monitoring, compiled class (primary) ----------------
    n_elig = n_nested = n_sizefilter = 0
    for ti, t in enumerate(pt):
        pool = rb.build_prom_pool(t, bank)
        ok, why, readings = _eligible(pool)
        uid = f'{getattr(t, "name", "?")}#{ti}'
        if not ok:
            # A rule is PROVABLY CLEAR in the compiled class when its maximal
            # threshold/duration subfamily has a single element or is totally
            # ordered: the compiled instance then has one binding reading, so
            # min_psi(V_psi - V_U) = 0 identically.  A rule the compiler's
            # grammar does not cover (lower comparators) is UNANALYZED -- it
            # is not evidence that the screen clears.
            provably_clear = (
                why.startswith("|Max|=1 ") or why == "singleton antichain"
                or why == "singleton subfamily"
                or why == "no threshold/duration crossing")
            unanalyzed = not provably_clear
            if unanalyzed:
                n_sizefilter += 1
            else:
                n_nested += 1
            for d in BUDGETS:
                rows.append(dict(
                    rule_id=uid, family="prometheus",
                    policy_class="monitoring_compiled", budget=d,
                    structural_noncollapse=(None if unanalyzed else False),
                    V_U=None, V_min_single=None, screen_gap=None,
                    screen_fires=(None if unanalyzed else False),
                    status=("unanalyzed_outside_grammar" if unanalyzed
                            else "provably_clear_nested"),
                    reason=why))
            continue
        n_elig += 1
        m = compile_instance(readings)
        for d in BUDGETS:
            a = exact_analyse(m, d)
            rows.append(dict(
                rule_id=uid, family="prometheus",
                policy_class="monitoring_compiled", budget=d,
                structural_noncollapse=True,
                V_U=a["V_U"], V_min_single=a["V_U"] + a["screen_gap"],
                screen_gap=a["screen_gap"], screen_fires=a["fires_value"],
                status="compiled", reason="",
                poa=a["poa"], rpoa=a["rpoa"],
                singleton_worst_ratio=a["max_best_worst_ratio"],
                robust_worst_ratio=a["robust_worst_ratio"]))
    print(f"[compiled] eligible={n_elig} provably-clear(nested/singleton)={n_nested} "
          f"unanalyzed(outside grammar)={n_sizefilter}  total={n_elig+n_nested+n_sizefilter}")

    # ---------------- free class (admission primary, monitoring sensitivity)
    for fam, targets, builder in (("prometheus", pt, lambda t: rb.build_prom_pool(t, bank)),
                                  ("kyverno", kt, rb.build_kyv_pool)):
        for ti, t in enumerate(targets):
            pool = builder(t)
            vecs = [c.vector for c in pool.classes]
            uid = f'{getattr(t, "name", getattr(t, "policy_name", "?"))}#{ti}'
            dom = dominance.pool_dominance(vecs)
            vals = free_values_multi(vecs, BUDGETS)
            # Effective-sample-size bookkeeping: pools that share a cost matrix
            # are the SAME instance and must not be counted as independent
            # measurements of the price.
            sig = hashlib.sha1(
                repr(tuple(sorted(tuple(v) for v in vecs))).encode()).hexdigest()[:12]
            for d in BUDGETS:
                V_full, V_k = vals[d]
                gap = min(v - V_full for v in V_k)
                V_max = max(V_k)
                rpoa = (V_max - V_full) / V_max if V_max > 0 else 0.0
                rows.append(dict(
                    rule_id=uid, family=fam,
                    policy_class="free", budget=d,
                    structural_noncollapse=bool(dom["genuinely_non_dominated"]),
                    V_U=V_full, V_min_single=V_full + gap,
                    V_max_single=V_max, poa=V_max - V_full, rpoa=rpoa,
                    pool_sig=sig,
                    screen_gap=gap, screen_fires=bool(gap > TOL),
                    shipped_fires=bool(shipped_free_screen(vecs, d)),
                    status="free", reason=""))
        print(f"[free/{fam}] done")

    # ---------------- aggregate + tie-safety comparison ----------------
    def rate(cls: str, fam: str, d: float) -> dict:
        sel = [r for r in rows if r["policy_class"] == cls
               and r["family"] == fam and r["budget"] == d]
        decided = [r for r in sel if r["screen_fires"] is not None]
        unan = [r for r in sel if r["screen_fires"] is None]
        fires = sum(1 for r in decided if r["screen_fires"])
        out = dict(n_total=len(sel), n_decided=len(decided), n_unanalyzed=len(unan),
                   fires=fires, rate=fires / len(decided) if decided else None)
        ship = [r for r in sel if r.get("shipped_fires") is not None]
        if ship:
            out["shipped_fires"] = sum(1 for r in ship if r["shipped_fires"])
            out["disagreements"] = sum(1 for r in ship
                                       if r["shipped_fires"] != r["screen_fires"])
        struct = [r for r in sel if r["structural_noncollapse"] is not None]
        if struct:
            out["structural_noncollapse_rate"] = (
                sum(1 for r in struct if r["structural_noncollapse"]) / len(struct))
        return out

    agg = {}
    for cls, fam in (("monitoring_compiled", "prometheus"),
                     ("free", "prometheus"), ("free", "kyverno")):
        agg[f"{fam}/{cls}"] = {str(d): rate(cls, fam, d) for d in BUDGETS}

    # Apples-to-apples class comparison: the free-class screen restricted to
    # exactly the rules the compiled class can analyze, so the two rates share
    # a denominator.  (Over all 88 the free class stays defined; that broader
    # rate is the "prometheus/free" entry above.)
    analyzable = {r["rule_id"] for r in rows
                  if r["policy_class"] == "monitoring_compiled"
                  and r["screen_fires"] is not None}

    def rate_common(cls: str, d: float) -> dict:
        sel = [r for r in rows if r["policy_class"] == cls
               and r["family"] == "prometheus" and r["budget"] == d
               and r["rule_id"] in analyzable]
        fires = sum(1 for r in sel if r["screen_fires"])
        return dict(n_total=len(sel), n_decided=len(sel), n_unanalyzed=0,
                    fires=fires, rate=fires / len(sel) if sel else None)

    # ---- exact full-pool price per family, with effective sample size
    def price_stats(fam: str, d: float) -> dict:
        sel = [r for r in rows if r["policy_class"] == "free"
               and r["family"] == fam and r["budget"] == d]
        by_sig = {}
        for r in sel:
            by_sig.setdefault(r["pool_sig"], []).append(r["rpoa"])
        distinct = sorted(v[0] for v in by_sig.values())
        allv = sorted(r["rpoa"] for r in sel)

        def q(xs, f):
            if not xs:
                return None
            i = f * (len(xs) - 1)
            lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
            return xs[lo] + (i - lo) * (xs[hi] - xs[lo])

        return dict(
            n_pools=len(sel), n_distinct_instances=len(by_sig),
            largest_duplicate_group=max((len(v) for v in by_sig.values()), default=0),
            n_distinct_values=len(set(round(x, 9) for x in allv)),
            n_positive=sum(1 for x in allv if x > 1e-12),
            pool_weighted=dict(min=q(allv, 0), q1=q(allv, .25), median=q(allv, .5),
                               q3=q(allv, .75), max=q(allv, 1)),
            instance_weighted=dict(min=q(distinct, 0), q1=q(distinct, .25),
                                   median=q(distinct, .5), q3=q(distinct, .75),
                                   max=q(distinct, 1)))

    agg["price/prometheus_free"] = {str(d): price_stats("prometheus", d) for d in BUDGETS}
    agg["price/kyverno_free"] = {str(d): price_stats("kyverno", d) for d in BUDGETS}

    agg["prometheus/free_common"] = {str(d): rate_common("free", d)
                                     for d in BUDGETS}
    agg["prometheus/compiled_common"] = {str(d): rate_common("monitoring_compiled", d)
                                         for d in BUDGETS}
    for d in BUDGETS:
        a, b = (agg["prometheus/compiled_common"][str(d)],
                agg["prometheus/free_common"][str(d)])
        assert a["n_total"] == b["n_total"] == len(analyzable), \
            "common-subset comparison lost its shared denominator"
        assert a["fires"] == agg["prometheus/monitoring_compiled"][str(d)]["fires"], \
            "compiled fires changed under the common-subset restriction"


    out = dict(registration="P0-B policy-class x budget audit",
               budgets=list(BUDGETS), operating=OPERATING, tol=TOL,
               compiled_denominator=dict(eligible=n_elig, provably_clear=n_nested,
                                         unanalyzed_outside_grammar=n_sizefilter),
               aggregate=agg, rows=rows)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    cols = ["rule_id", "family", "policy_class", "budget", "structural_noncollapse",
            "V_U", "V_min_single", "V_max_single", "poa", "rpoa", "pool_sig",
            "screen_gap", "screen_fires", "shipped_fires", "status", "reason"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    for key, per_d in agg.items():
        line = "  ".join(
            f"d={d}:{v['fires']}/{v['n_decided']}"
            + (f"(ship {v['shipped_fires']}, dis {v['disagreements']})"
               if "shipped_fires" in v else "")
            for d, v in per_d.items())
        print(f"[{key}] {line}")
    print("wrote", OUT_JSON, "and", OUT_CSV)


if __name__ == "__main__":
    main()
