"""Phase 3 GUARDS: train the three FQI guards on the offline logs (paper section
22, phase 3), reusing saorl's budget FQI unchanged (learn_budget_constrained via
guard.train_guard). The three guards differ only in the reading set they honor:

  * single        -> honor the single most-plausible retained reading (k = 5)
  * corset        -> honor the whole conformal-retained set (worst-case; the
                     binding reading is the tightest retained, k = 3)
  * post_clarify  -> honor the externally-anchored strict reading (1 msg/sec,
                     k = 2)

Violations are always measured against the retained set U (eval_readings). Each
guard's frozen policy is snapshot as allow_by_calls[c] (c = 0..12) and saved to
demo/guards/<mode>.json, plus lam / honored_cost / true_worst and a per-reading
offline feasibility table J_{c_k} (expected semantic cost over the logs' active
states), verifying feasibility offline before any live billing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import guard as guard_mod
import readings as R
from saorl.offline import policy_costs

HERE = Path(__file__).resolve().parent
OFFLINE_EPISODES = HERE / "logs" / "offline_episodes.jsonl"
GUARDS_DIR = HERE / "guards"
GUARD_REPORT = HERE / "logs" / "guard_report.json"
CALLS_RANGE = list(range(13))  # 0..12


def load_episodes() -> List[dict]:
    return [json.loads(l) for l in OFFLINE_EPISODES.read_text().splitlines() if l.strip()]


def rl_rewards(ep: dict) -> List[float]:
    """FQI reward, following the saorl budget domain (r_buy = task value, r_skip =
    0): a paid post earns its message's rubric value, the free fallback earns
    nothing this step. (The free-fallback value is real and still counts in the
    live task-value metric and the VoQ economics -- it is reported there -- but
    the GATING policy is learned on the incremental value of paying, exactly as
    budget.py learns to skip once a cap fires.)"""
    out = []
    for a, pv in zip(ep["actions"], ep["paid_value"]):
        out.append(float(pv) if (a == "buy" and pv is not None) else 0.0)
    return out


def rl_episodes(episodes: List[dict]) -> List[dict]:
    return [{"trajectory": ep["trajectory"], "actions": ep["actions"],
             "rewards": rl_rewards(ep)} for ep in episodes]


def skip_coverage_episodes(episodes: List[dict]) -> List[dict]:
    """Synthetic skip transitions to fill BCQ coverage of the FREE action at every
    calls-bucket (reward 0, following r_skip = 0). Greedy episodes never
    demonstrate skip at high buckets, so BCQ would forbid skip there and force the
    guard to buy (violating the honored cap) -> lambda blows up. A skip self-loop
    (s=calls, a=skip, r=0, s'=calls) is the free no-op transition; no new billing.
    Each synthetic episode is a 2-step self-loop so the transition bootstraps."""
    aug = []
    for ep in episodes:
        for st in ep["trajectory"]:
            c = int(st["calls"])
            aug.append({"trajectory": [{"calls": c}, {"calls": c}],
                        "actions": ["skip"], "rewards": [0.0]})
    return aug


def allow_table(g: guard_mod.TrainedGuard) -> List[bool]:
    raw = [bool(g.allow_paid({"calls": c, "step": 0})) for c in CALLS_RANGE]
    # monotonize: once the guard denies a post, it stays denied at higher counts
    # (a clean deployment cap; above-cap buckets are never reached live anyway).
    mono = []
    denied = False
    for a in raw:
        denied = denied or (not a)
        mono.append(not denied)
    return mono


def cap_of(allow: List[bool]) -> int:
    """The post count at which the guard first denies (its effective cap)."""
    for c, ok in enumerate(allow):
        if not ok:
            return c
    return len(allow)


def main() -> dict:
    GUARDS_DIR.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes()
    aug = skip_coverage_episodes(episodes)
    train_episodes = rl_episodes(episodes) + aug   # r_buy=paid value, r_skip=0
    rs = R.reading_sets()
    retained = rs["retained"]
    readings = {"retained": retained, "single": rs["single"],
                "anchored": rs["anchored"]}

    print("== Phase 3 GUARDS: FQI (saorl.learn_budget_constrained) ==")
    print(f"  logs: {len(episodes)} episodes (+{len(aug)} synthetic free-action "
          f"coverage transitions) | retained U = {rs['retained_names']}")
    print(f"  single honors {rs['single_name']} | post_clarify honors {rs['anchored_name']}")

    guards = guard_mod.train_three_guards(
        train_episodes, readings=readings, axis="calls", unit_cost=1.0, eps=0.0)

    # per-reading feasibility J_{c_k} is reported on the ORIGINAL logged data only
    # (the synthetic skip transitions never violate, so they must not dilute the
    # honestly-reported violation rates).
    data = guard_mod.episodes_to_offline(episodes, axis="calls")
    eval_readings = retained + rs["anchored"]  # include the strict anchor too

    report: Dict[str, dict] = {}
    retained_names = set(rs["retained_names"])
    for mode, g in guards.items():
        allow = allow_table(g)
        per_reading = policy_costs(g.policy, data, eval_readings, normalize="active")
        true_worst_orig = max((v for k, v in per_reading.items()
                               if k in retained_names), default=0.0)
        ckpt = {
            "mode": mode,
            "honors": ([rs["single_name"]] if mode == "single"
                       else rs["retained_names"] if mode == "corset"
                       else [rs["anchored_name"]]),
            "eval_readings": [c.name for c in eval_readings],
            "lam": g.lam,
            "honored_cost_train_aug": g.honored_cost,
            "true_worst_retained_original": round(true_worst_orig, 6),
            "allow_by_calls": allow,
            "effective_cap": cap_of(allow),
            "per_reading_cost": {k: round(v, 6) for k, v in per_reading.items()},
        }
        (GUARDS_DIR / f"{mode}.json").write_text(json.dumps(ckpt, indent=2))
        report[mode] = ckpt
        print(f"  guard[{mode:12s}] lam={g.lam:5.1f} cap={ckpt['effective_cap']:2d} "
              f"true_worst(retained,orig)={true_worst_orig:.3f}")
        print(f"       allow_by_calls={['Y' if a else 'n' for a in allow]}")
        viol = {k: v for k, v in per_reading.items() if v > 1e-9}
        print(f"       per-reading violations (offline): "
              f"{ {k: round(v,3) for k,v in viol.items()} or 'none'}")

    GUARD_REPORT.write_text(json.dumps(report, indent=2))
    print(f"  wrote guards/*.json and {GUARD_REPORT.relative_to(HERE)}")
    return report


if __name__ == "__main__":
    main()
