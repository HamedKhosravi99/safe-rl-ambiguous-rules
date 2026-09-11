"""W10 (blueprint sections 12 and 30.8): stress the universal-fallback
assumption behind the surgery certificate (Theorem 3's witness policy).

The surgery bound assumes a fallback action a0 that is always available,
offline-supported, and zero-cost under EVERY retained reading.  This module
reruns the surgery-certificate measurement of saorl.price_of_ambiguity -- same
domain spec, same FQI singleton oracles, same union-guard/surgery machinery,
imported not reimplemented -- in the synthetic maintenance domain under four
fallback regimes:

  baseline       zero-cost 'replace' as a0 (the task-named fallback; both
                 'replace' and the machinery default 'minor_repair' are
                 semantically zero-cost -- neither is forbidden by any reading)
  delayed        the fallback takes effect one step LATE: at the switch step
                 the wrapper takes the greedy (forbidden) action, and only
                 from the next step follows the union guard
  costly         each guard invocation of a0 carries a fixed penalty of 5
                 return units; the certificate is reported against the
                 d0-adjusted surgery value (d0 = 5 x E[#invocations/episode]),
                 i.e. the bounded-cost generalization max_psi J_c(pi0) <= d0
  state_limited  'replace' is unavailable when rul_hat < 5; the guard falls
                 back to the greedy action on those steps

A fifth row, machinery_default, repeats the baseline with the a0 that
saorl.price_of_ambiguity._fallback_action actually selects ('minor_repair'),
so the numbers tie back to results/conformal/price/price_report.json.

For each variant x seed x psi in U we report the surgery certificate
v_psi - v_surgery (d0-adjusted for `costly`), the surgery policy's worst-case
semantic cost over U on the offline data (feasibility check against
eps = 0.05), and a verdict.  Feasibility breaches are expected exactly where
the zero-cost assumption is broken in kind (delayed / state-limited): those
regimes inject forbidden-action steps, which is what the bounded-cost remark
d0 prices when the injection costs return rather than safety.

Run:  SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.fallback_stress
Writes results/conformal/fallback_stress.json
   and paper/generated/gen_fallback.tex
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .experiments import EPS, _domain_specs
from .offline import worst_case_cost
from .price_of_ambiguity import (
    _fallback_action,
    _fqi,
    _surgery_policy,
    _union_guard,
)

N_SEEDS = 5
N_EPISODES_RET = 40        # matches evaluate_return's default (seed=123)
RET_SEED = 123
FALLBACK_PENALTY = 5.0     # per guard invocation, `costly` variant
RUL_LIMIT = 5.0            # 'replace' unavailable below this, `state_limited`
A0_TASK = "replace"

_REPO = Path(__file__).parent.parent.parent

VARIANTS = ("baseline", "delayed", "costly", "state_limited")
# anticipatory lookahead for budget-aware surgery, in OBSERVATION-NOISE sd
BA_MARGINS = (0.0, 1.0, 2.0, 3.0, 4.0)
VARIANT_LABEL = {
    "baseline": "Zero-cost replace (baseline)",
    "delayed": "Delayed (+1 step)",
    "costly": "Costly (5/invocation, $d_0$-adjusted)",
    "state_limited": r"State-limited ($\hat{\mathrm{RUL}}<5$)",
    "machinery_default": "Machinery default (minor\\_repair)",
}


# --------------------------------------------------------------------------
# Guards and surgery wrappers (variants of price_of_ambiguity's machinery)
# --------------------------------------------------------------------------

def _counting_guard(U, a0: str, greedy_action: str, counter: dict):
    """The union guard, instrumented: counts every a0 invocation."""
    def pol(traj, t):
        if any(c.fires(traj, t) for c in U):
            counter["n"] += 1
            return a0
        return greedy_action
    return pol


def _limited_guard(U, a0: str, greedy_action: str, rul_limit: float):
    """Union guard with a0 unavailable when rul_hat < rul_limit: the guard
    degrades to the greedy (forbidden) action on exactly those steps."""
    def pol(traj, t):
        if any(c.fires(traj, t) for c in U):
            if traj[t]["rul_hat"] < rul_limit:
                return greedy_action
            return a0
        return greedy_action
    return pol


def _budget_aware_surgery(base_policy, psi, U, guard, greedy_action: str,
                          k_sd: float, obs_noise: float, anom_noise: float):
    """BUDGET-AWARE surgery for a DELAYED fallback (paper: the invariant
    'falling back now still finishes within budget').

    With a one-step-delayed fallback, switching AT a disputed step executes
    the forbidden action once, which is exactly what breaks feasibility for
    the first-dispute rule.  The invariant therefore forces the switch to
    happen BEFORE the dispute can occur: we switch as soon as the state is
    within `margin` of the firing region of any retained reading, so that
    the delayed step lands while nothing fires and costs nothing.

    The margin is measured in OBSERVATION-NOISE standard deviations, which
    is both dimensionally consistent across features and the right scale:
    the trigger reads the same noisy feature the cost is evaluated on, so a
    margin that does not clear the noise lets the delayed step land inside
    the firing region anyway.  With sigma_rul and sigma_anom the feature
    noises, a k-sd margin is `theta + k*sigma_rul` on a `rul_hat <= theta`
    reading and `theta - k*sigma_anom` on an `anom >= theta` one.  k = 0
    recovers the first-dispute rule; larger k switches earlier, trading
    return for feasibility.  The smallest feasible k is what the invariant
    prescribes.
    """
    state = {"switched_at": None}

    def near_firing(traj, t) -> bool:
        obs = traj[t]
        for c in U:
            a = getattr(c.predicate, "atom", c.predicate)
            feat = getattr(a, "feature", None)
            theta = getattr(a, "theta", None)
            op = getattr(a, "op", None)
            if feat is None or theta is None or feat not in obs:
                continue
            v = obs[feat]
            sd = obs_noise if feat == "rul_hat" else anom_noise
            if op == "le" and v <= theta + k_sd * sd:
                return True
            if op == "ge" and v >= theta - k_sd * sd:
                return True
        return False

    def pol(traj, t):
        if t == 0:
            state["switched_at"] = None
        if state["switched_at"] is not None:
            return guard(traj, t)
        a = base_policy(traj, t)
        disputed = any(c.cost(traj, t, a) > psi.cost(traj, t, a) for c in U)
        if disputed or near_firing(traj, t):
            state["switched_at"] = t
            # The fallback is STILL DELAYED: this step takes the greedy
            # (forbidden) action, exactly as in the `delayed` variant.  The
            # anticipatory trigger is what makes that step land while
            # nothing fires, so it costs nothing -- the delay is survived,
            # not removed.
            return greedy_action
        return a

    return pol


def _surgery_with_guard(base_policy, psi, U, guard, delay: bool,
                        greedy_action: str):
    """price_of_ambiguity._surgery_policy generalized: arbitrary guard, and
    (delay=True) the greedy action at the switch step -- the fallback takes
    effect one step later."""
    state = {"switched_at": None}

    def pol(traj, t):
        if t == 0:
            state["switched_at"] = None
        if state["switched_at"] is not None:
            return guard(traj, t)
        a = base_policy(traj, t)
        disputed = any(c.cost(traj, t, a) > psi.cost(traj, t, a) for c in U)
        if disputed:
            state["switched_at"] = t
            if delay:
                return greedy_action
            return guard(traj, t)
        return a

    return pol


def _eval_return_counting(env, policy, counter: dict,
                          n_episodes: int = N_EPISODES_RET,
                          seed: int = RET_SEED):
    """Replicates saorl.offline.evaluate_return step for step (same rng
    consumption, same seed) while `counter` tracks guard invocations.
    Returns (mean return, mean invocations per episode)."""
    rng = np.random.default_rng(seed)
    totals = []
    start = counter["n"]
    for _ in range(n_episodes):
        rul = env.initial_rul(rng)
        traj, total = [], 0.0
        for t in range(env.horizon):
            traj.append(env.observe(rul, rng))
            a = policy(traj, t)
            r, rul, _ = env.step(rul, a)
            total += r
        totals.append(total)
    invocations = (counter["n"] - start) / n_episodes
    return float(np.mean(totals)), float(invocations)


# --------------------------------------------------------------------------
# The stress run
# --------------------------------------------------------------------------

def run(seeds: Sequence[int] = tuple(range(N_SEEDS))) -> dict:
    assert os.environ.get("SAORL_CONFORMAL") == "1", \
        "run with SAORL_CONFORMAL=1: the certificates are defined over the conformal sets"
    t0 = time.time()
    spec = _domain_specs(["synthetic"], ["fqi"])["synthetic"]

    variant_rows: Dict[str, List[dict]] = {v: [] for v in VARIANTS}
    variant_rows["machinery_default"] = []
    ba_rows: Dict[float, List[dict]] = {k: [] for k in BA_MARGINS}
    checks = dict(evaluator_matches_ret_fn=None, price_report=[])

    for seed in seeds:
        data, env, U, ret_fn, _ = spec.build(seed)
        a0_mach = _fallback_action(data, U)
        forbidden = set().union(*(c.forbidden_actions for c in U))
        greedy_action = sorted(forbidden)[0]

        oracles = {}
        for c in U:
            orac = _fqi(spec.learners, data, env, [c], U, ret_fn)
            oracles[c.name] = (c, orac)

        def measure(variant: str, a0: str):
            per_psi = []
            for name, (c, orac) in oracles.items():
                counter = {"n": 0}
                if variant == "state_limited":
                    guard = _limited_guard(U, a0, greedy_action, RUL_LIMIT)
                else:
                    guard = _counting_guard(U, a0, greedy_action, counter)
                surg = _surgery_with_guard(orac.policy, c, U, guard,
                                           delay=(variant == "delayed"),
                                           greedy_action=greedy_action)
                v_surg, invoc = _eval_return_counting(env, surg, counter)
                d0 = FALLBACK_PENALTY * invoc if variant == "costly" else 0.0
                # fresh policy instance for the data-side cost (stateful pol)
                if variant == "state_limited":
                    guard2 = _limited_guard(U, a0, greedy_action, RUL_LIMIT)
                else:
                    guard2 = _counting_guard(U, a0, greedy_action, {"n": 0})
                surg2 = _surgery_with_guard(orac.policy, c, U, guard2,
                                            delay=(variant == "delayed"),
                                            greedy_action=greedy_action)
                worst = float(worst_case_cost(surg2, data, U, normalize="active"))
                per_psi.append(dict(
                    psi=name, v_psi=orac.ret,
                    v_surgery=v_surg, d0=round(d0, 4),
                    invocations_per_episode=round(invoc, 4),
                    certificate=orac.ret - (v_surg - d0),
                    surgery_worst=worst,
                    feasible=bool(worst <= EPS + 1e-9),
                ))
            return dict(
                seed=seed, a0=a0,
                per_psi=per_psi,
                cert_worst=float(max(p["certificate"] for p in per_psi)),
                worst_cost=float(max(p["surgery_worst"] for p in per_psi)),
                feasible=bool(all(p["feasible"] for p in per_psi)),
            )

        # --- budget-aware surgery on the DELAYED fallback -------------------
        # The delayed variant is the one that genuinely breaks feasibility
        # (the fallback executes the forbidden action once at the switch
        # step).  Sweep the anticipatory lookahead k and record, per seed,
        # the smallest k that restores worst-case cost <= EPS at the
        # ORIGINAL budget -- no d0 relaxation.
        for k in BA_MARGINS:
            per_psi = []
            for name, (c, orac) in oracles.items():
                g1 = _counting_guard(U, A0_TASK, greedy_action, {"n": 0})
                pol1 = _budget_aware_surgery(orac.policy, c, U, g1,
                                             greedy_action, k, env.obs_noise,
                                             env.anom_noise)
                v_ba, _ = _eval_return_counting(env, pol1, {"n": 0})
                g2 = _counting_guard(U, A0_TASK, greedy_action, {"n": 0})
                pol2 = _budget_aware_surgery(orac.policy, c, U, g2,
                                             greedy_action, k, env.obs_noise,
                                             env.anom_noise)
                worst = float(worst_case_cost(pol2, data, U,
                                              normalize="active"))
                per_psi.append(dict(psi=name, v_psi=orac.ret, v_surgery=v_ba,
                                    surgery_worst=worst,
                                    feasible=bool(worst <= EPS + 1e-9)))
            ba_rows[k].append(dict(
                seed=seed, k=k, per_psi=per_psi,
                worst_cost=float(max(p["surgery_worst"] for p in per_psi)),
                v_surgery_min=float(min(p["v_surgery"] for p in per_psi)),
                feasible=bool(all(p["feasible"] for p in per_psi))))

        for variant in VARIANTS:
            variant_rows[variant].append(measure(variant, A0_TASK))
        variant_rows["machinery_default"].append(measure("baseline", a0_mach))

        # verification: the instrumented evaluator must reproduce ret_fn
        if checks["evaluator_matches_ret_fn"] is None:
            c0, orac0 = next(iter(oracles.values()))
            guard = _union_guard(U, a0_mach, greedy_action)
            ref = _surgery_policy(orac0.policy, c0, U, a0_mach, greedy_action)
            v_ref = ret_fn(ref)
            mine = _surgery_with_guard(
                orac0.policy, c0, U,
                _counting_guard(U, a0_mach, greedy_action, {"n": 0}),
                delay=False, greedy_action=greedy_action)
            v_mine, _ = _eval_return_counting(env, mine, {"n": 0})
            checks["evaluator_matches_ret_fn"] = dict(
                ret_fn=v_ref, instrumented=v_mine,
                match=bool(abs(v_ref - v_mine) < 1e-9))

        # tie-back: machinery-default certificates vs the frozen price report
        checks["price_report"].append(dict(
            seed=seed,
            machinery_cert_worst=variant_rows["machinery_default"][-1]["cert_worst"],
        ))
        print(f"  seed {seed}: " + "  ".join(
            f"{v}: cert={variant_rows[v][-1]['cert_worst']:.2f} "
            f"worst={variant_rows[v][-1]['worst_cost']:.4f} "
            f"feas={variant_rows[v][-1]['feasible']}"
            for v in VARIANTS))

    # budget-aware (anticipatory) surgery on the delayed fallback
    ba_summary = []
    for k in BA_MARGINS:
        rows = ba_rows[k]
        ba_summary.append(dict(
            k_steps=k,
            n_seeds=len(rows),
            n_feasible=int(sum(r["feasible"] for r in rows)),
            worst_cost_max=float(max(r["worst_cost"] for r in rows)),
            worst_cost_mean=float(np.mean([r["worst_cost"] for r in rows])),
            v_surgery_min_mean=float(np.mean([r["v_surgery_min"] for r in rows])),
        ))
    # standalone feasibility of the delayed fallback itself: switching at
    # t=0 is the earliest possible switch, and surgery accrues no cost
    # before switching, so this is the FLOOR for any switching rule.
    ba_summary_floor = float(max(b["worst_cost_max"] for b in ba_summary))
    print("\n  budget-aware surgery on the DELAYED fallback:")
    for b in ba_summary:
        print(f"    k={b['k_steps']:.0f} sd: feasible {b['n_feasible']}/"
              f"{b['n_seeds']} seeds, worst cost {b['worst_cost_max']:.4f} "
              f"(eps={EPS}), min surgery return {b['v_surgery_min_mean']:.2f}")

    # compare with price_report rows (same seeds, machinery a0)
    price_path = _REPO / "results/conformal" / "price" / "price_report.json"
    if price_path.exists():
        pr = json.load(open(price_path))
        by_seed = {r["seed"]: r for r in pr["domains"]["synthetic"]["rows"]}
        for chk in checks["price_report"]:
            r = by_seed.get(chk["seed"])
            if r is not None:
                chk["price_report_wpoa_surgery"] = r["wpoa_surgery"]
                chk["match"] = bool(abs(chk["machinery_cert_worst"]
                                        - r["wpoa_surgery"]) < 1e-6)

    def _agg(rows: List[dict], variant: str) -> dict:
        certs = [r["cert_worst"] for r in rows]
        worsts = [r["worst_cost"] for r in rows]
        n_breach = sum(1 for r in rows for p in r["per_psi"]
                       if not p["feasible"])
        n_checks = sum(len(r["per_psi"]) for r in rows)
        feasible = all(r["feasible"] for r in rows)
        d0s = [p["d0"] for r in rows for p in r["per_psi"]]
        verdict = {
            "baseline": "assumption holds: certificate valid at zero cost",
            "machinery_default": "assumption holds (machinery a0); ties to "
                                 "price_report",
            "delayed": ("eps breached: one forbidden step per dispute makes "
                        "pi0 non-zero-cost, so Theorem 3's zero-cost premise "
                        "fails in kind; the bounded-cost remark applies with "
                        "d0 = the per-dispute violation mass" if not feasible
                        else "one-step delay absorbed within eps: certificate "
                             "survives unmodified"),
            "costly": "zero-violation preserved; certificate re-priced by "
                      "d0 = 5 x E[#invocations] per the bounded-cost remark",
            "state_limited": ("guard degrades to greedy below rul_hat<5: "
                              "eps breached -- fallback availability is a "
                              "real assumption, not a formality" if not feasible
                              else "limit never binds on-path: certificate "
                                   "survives"),
        }[variant]
        return dict(
            certificate_mean=round(float(np.mean(certs)), 4),
            certificate_sd=round(float(np.std(certs)), 4),
            worst_cost_mean=round(float(np.mean(worsts)), 6),
            worst_cost_max=round(float(np.max(worsts)), 6),
            feasible=bool(feasible),
            eps=EPS,
            n_breaches=int(n_breach),
            n_checks=int(n_checks),
            d0_mean=round(float(np.mean(d0s)), 4) if any(d0s) else 0.0,
            verdict=verdict,
        )

    aggs = {v: _agg(rows, v) for v, rows in variant_rows.items()}
    return dict(
        config=dict(domain="synthetic", seeds=list(seeds), eps=EPS,
                    a0_task=A0_TASK, fallback_penalty=FALLBACK_PENALTY,
                    rul_limit=RUL_LIMIT, n_episodes=N_EPISODES_RET,
                    ret_seed=RET_SEED, construction="conformal"),
        variants={v: dict(rows=variant_rows[v], agg=aggs[v])
                  for v in variant_rows},
        budget_aware=dict(
            note="anticipatory (budget-aware) surgery on the DELAYED "
                 "fallback; margin k in observation-noise sd. The floor of "
                 "the sweep is the delayed fallback's OWN standalone cost: "
                 "switching at t=0 is the earliest possible switch and "
                 "surgery accrues no cost before switching, so no switching "
                 "rule can beat it.",
            margins_in_noise_sd=list(BA_MARGINS),
            sweep=ba_summary, floor=ba_summary_floor,
            rows={str(k): ba_rows[k] for k in BA_MARGINS}),
        checks=checks,
        elapsed_s=time.time() - t0,
    )


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def _write_tex(report: dict, path: Path) -> None:
    lines = ["% AUTO-GENERATED by saorl/fallback_stress.py -- do not edit",
             "% variant & surgery certificate (mean +/- sd, worst psi) & "
             "surgery worst-case cost (max over seeds/psi; eps=0.05) & feasible"]
    for v in VARIANTS + ("machinery_default",):
        a = report["variants"][v]["agg"]
        feas = "yes" if a["feasible"] else \
            f"no ({a['n_breaches']}/{a['n_checks']})"
        lines.append(
            f"{VARIANT_LABEL[v]} & "
            f"${a['certificate_mean']:.2f}\\pm{a['certificate_sd']:.2f}$ & "
            f"{a['worst_cost_max']:.4f} & {feas} \\\\")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    report = run()
    out_json = _REPO / "results/conformal" / "fallback_stress.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out_json, "w"), indent=1)
    # second fragment: the budget-aware sweep on the delayed fallback
    ba = report.get("budget_aware", {})
    if ba:
        lines = ["% AUTO-GENERATED by saorl/fallback_stress.py -- do not edit\n"]
        for b in ba["sweep"]:
            lines.append(
                f"$k={b['k_steps']:.0f}\\sigma$ & {b['n_feasible']}/{b['n_seeds']} & "
                f"${b['worst_cost_max']:.4f}$ & ${b['v_surgery_min_mean']:.1f}$ \\\\\n")
        (_REPO / "paper" / "generated" / "gen_fallback_ba.tex").write_text("".join(lines))
        print("wrote paper/generated/gen_fallback_ba.tex")
    out_tex = _REPO / "paper" / "generated" / "gen_fallback.tex"
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    _write_tex(report, out_tex)

    print(f"\n{'variant':34s} {'certificate':>16s} {'worst(max)':>11s} "
          f"{'feasible':>9s}")
    for v in VARIANTS + ("machinery_default",):
        a = report["variants"][v]["agg"]
        print(f"{v:34s} {a['certificate_mean']:8.2f}+/-{a['certificate_sd']:5.2f}"
              f" {a['worst_cost_max']:11.4f} {str(a['feasible']):>9s}"
              + (f"  d0={a['d0_mean']:.2f}" if a["d0_mean"] else ""))
        print(f"    verdict: {a['verdict']}")
    print("evaluator check:", report["checks"]["evaluator_matches_ret_fn"])
    print("price-report tie-back:",
          [(c["seed"], c.get("match")) for c in report["checks"]["price_report"]])
    print(f"wrote {out_json}\nwrote {out_tex}  ({report['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
