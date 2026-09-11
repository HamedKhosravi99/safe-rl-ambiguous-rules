"""REGISTRATION_V19: the deployment gate at 20,000 fresh evaluation episodes.

For each (domain, arm, seed): retrain the archived FQI policy exactly as
saorl.experiments does (asserting the archived 300-episode violation count
reproduces under the archived rollout seed 321), then roll out N fresh
episodes under rollout seed 20260903 + seed and evaluate three gates on the
same episodes at delta_ev = 0.05:

    CP route (the paper's):   B * CP_{1-delta/2}(k, N)
    union Hoeffding:          mean W + B sqrt(ln(w/delta) / (2N))
    empirical Bernstein:      mean + sqrt(2 v ln(2w/delta) / N) + 7 B ln(2w/delta) / (3 (N-1))

W = max_psi C_psi / Z_psi with the audit's fixed normalizers; B, Z, w from the
archived certificate files.  One JSON per task so the runs can be spread over
processes or a cluster array; --merge collects them.

Run:  SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.evaluator_scale --domain real --arm corset --seeds 0-49
      ... --merge
Writes results/conformal/lp/evaluator_scale/<domain>_<arm>_<seed>.json and, on
--merge, results/conformal/lp/evaluator_scale.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
from collections import Counter

from .alt_defense import GREEDY_ACTION
from .betting_cs import betting_upper
from .exact_lp import (_ROOT, build_budget, build_gridworld, build_real,
                       build_synthetic, fixed_psi_Z)
from .expected_cost_certificate import DELTA_EV, D_BUDGET, LEVEL, _newest, cp_upper

N_EPISODES = 20000
ROLLOUT_SEED0 = 20260903
TAGS = {"": dict(seed0=20260903, registration="REGISTRATION_V19.md"),
        "50k": dict(seed0=20260904, registration="REGISTRATION_V25.md"),
        "smoke": dict(seed0=20260904, registration="smoke")}


def paths(tag: str):
    suf = f"_{tag}" if tag else ""
    base = _ROOT / "results/conformal" / "lp"
    return base / f"evaluator_scale{suf}", base / f"evaluator_scale{suf}.json"


OUTDIR, MERGED = paths("")
ARCH_SEED = 321
BUILDER = dict(synthetic=build_synthetic, real=build_real, gridworld=build_gridworld, budget=build_budget)
ARM_MODEL = dict(corset="fqi:saorl", single="fqi:single")


def gates(W: np.ndarray, B: float, w: int, delta: float = DELTA_EV, d: float = D_BUDGET) -> dict:
    n = W.size
    k = int((W > 0).sum())
    cp = B * cp_upper(k, n, LEVEL)
    hoe = float(W.mean() + B * math.sqrt(math.log(w / delta) / (2 * n)))
    dl = delta / w
    ln = math.log(2.0 / dl)
    v = float(W.var(ddof=1)) if n > 1 else 0.0
    bern = float(W.mean() + math.sqrt(2 * v * ln / n) + 7 * B * ln / (3 * (n - 1)))

    def n_needed(f, nmax=50_000_000):
        lo, hi = 2, nmax
        if f(hi) > d:
            return None
        while lo < hi:
            mid = (lo + hi) // 2
            if f(mid) <= d:
                hi = mid
            else:
                lo = mid + 1
        return lo
    p = k / n
    mean = float(W.mean())
    bet = betting_upper(W, B, alpha=dl, d=d)
    return dict(n=n, k=k, p_hat=p, mean_W=mean, sd_W=math.sqrt(v), max_W=float(W.max()),
                betting_bound=bet["upper"], betting_ships=bool(bet["upper"] <= d), betting_first_crossing=bet["first_crossing"],
                cp_bound=cp, cp_ships=bool(cp <= d),
                hoeffding_bound=hoe, hoeffding_ships=bool(hoe <= d),
                bernstein_bound=bern, bernstein_ships=bool(bern <= d),
                true_mean_below_d=bool(mean <= d),
                n_needed_cp=n_needed(lambda m: B * cp_upper(int(round(p * m)), m, LEVEL)) if B * p < d else None,
                n_needed_hoeffding=n_needed(lambda m: mean + B * math.sqrt(math.log(w / delta) / (2 * m))) if mean < d else None,
                n_needed_bernstein=n_needed(lambda m: mean + math.sqrt(2 * v * ln / m) + 7 * B * ln / (3 * (m - 1))) if mean < d else None)


def run_task(domain: str, arm: str, seed: int, n_episodes: int = N_EPISODES, tag: str = "") -> dict:
    outdir, _merged = paths(tag)
    seed0 = TAGS[tag]["seed0"]
    from .experiments import _domain_specs, _learner_kwargs, most_plausible
    from .offline import episode_semantic_costs
    t0 = time.perf_counter()
    arch = json.load(open(_ROOT / "results/conformal" / "lp" / "expected_cost_certificate.json"))["domains"][domain]
    dep = json.load(open(_ROOT / "results/conformal" / "lp" / "deploy_certificate.json"))["domains"][domain]
    blob = _newest("budget50" if domain == "budget" else "risk50")
    rec = next(r for r in blob["records"] if r["domain"] == domain and r["model"] == ARM_MODEL[arm] and r["seed"] == seed)
    spec = _domain_specs([domain], ["fqi"])[domain]
    data, env, U, ret_fn, _raw = spec.build(seed)
    rollout_once = spec.rollout(env)
    fn = spec.learners["fqi"]
    kw = _learner_kwargs("fqi", seed, "cpu")
    honor = U if arm == "corset" else most_plausible(U)
    pol = fn(data, env, honor=honor, U_eval=U, eps=D_BUDGET, return_fn=ret_fn, **kw)
    names = [c.name for c in U]
    m = BUILDER[domain](env, list(U))
    Z = fixed_psi_Z(m, names, m.action_names.index(GREEDY_ACTION[domain]))
    Zv = np.array([float(Z[k]) if float(Z[k]) > 0 else 1.0 for k in names])
    B, w = arch["B"], dep["w"]

    def roll(n, rng_seed):
        rng = np.random.default_rng(rng_seed)
        W = np.empty(n); viol = np.empty(n, dtype=bool)
        cnt = Counter()
        for i in range(n):
            traj, actions = rollout_once(pol.policy, rng)
            C, Zind = episode_semantic_costs(traj, actions, U, gamma=1.0)
            W[i] = float((C / Zv).max()) if C.size else 0.0
            viol[i] = bool(Zind.any())
            F = [sum(1 for t in range(len(actions)) if c.fires(traj, t)) for c in U]
            cnt[tuple(int(round(v)) for v in C) + tuple(F)] += 1
        return W, viol, cnt
    # reproducibility anchor: the archived 300-episode count under the archived seed
    W3, v3, _c3 = roll(300, ARCH_SEED)
    k_arch = int(round(float(rec["chance"]) * 300))
    reproduces = int(v3.sum()) == k_arch
    Wn, _vn, cnt = roll(n_episodes, seed0 + seed)
    out = dict(domain=domain, arm=arm, seed=seed, B=B, w=w, horizon=int(m.horizon),
               archived_k300=k_arch, recomputed_k300=int(v3.sum()), reproduces_archive=reproduces,
               archived_ships300=bool(B * cp_upper(k_arch, 300, LEVEL) <= D_BUDGET),
               fresh=gates(Wn, B, w), seconds=round(time.perf_counter() - t0, 1),
               rollout_seed=seed0 + seed, learned_return=float(pol.ret),
               reading_names=names, Z={names[i]: float(Zv[i]) for i in range(len(names))},
               count_multiset=[[list(k), v] for k, v in sorted(cnt.items())])
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{domain}_{arm}_{seed}.json").write_text(json.dumps(out, indent=1))
    return out


def merge(tag: str = "") -> None:
    outdir, merged = paths(tag)
    parts = [json.load(open(outdir / f)) for f in sorted(os.listdir(outdir)) if f.endswith(".json")]
    summary = {}
    for dom in sorted({p["domain"] for p in parts}):
        for arm in ("corset", "single"):
            ps = [p for p in parts if p["domain"] == dom and p["arm"] == arm]
            if not ps:
                continue
            summary[f"{dom}/{arm}"] = dict(
                n_seeds=len(ps), reproduces_archive=sum(p["reproduces_archive"] for p in ps),
                archived_ships300=sum(p["archived_ships300"] for p in ps),
                cp_ships=sum(p["fresh"]["cp_ships"] for p in ps),
                hoeffding_ships=sum(p["fresh"]["hoeffding_ships"] for p in ps),
                bernstein_ships=sum(p["fresh"]["bernstein_ships"] for p in ps),
                betting_ships=sum(p["fresh"].get("betting_ships", False) for p in ps),
                betting_bound_median=float(np.median([p["fresh"]["betting_bound"] for p in ps])) if all("betting_bound" in p["fresh"] for p in ps) else None,
                n_first_crossing=sum(1 for p in ps if p["fresh"].get("betting_first_crossing")),
                first_crossing_median=(float(np.median([p["fresh"]["betting_first_crossing"] for p in ps if p["fresh"].get("betting_first_crossing")]))
                                       if any(p["fresh"].get("betting_first_crossing") for p in ps) else None),
                best_gate_ships=sum(1 for p in ps if p["fresh"]["cp_ships"] or p["fresh"]["hoeffding_ships"] or p["fresh"]["bernstein_ships"] or p["fresh"].get("betting_ships", False)),
                best_gate_ships_among_below=sum(1 for p in ps if p["fresh"]["true_mean_below_d"] and (p["fresh"]["cp_ships"] or p["fresh"]["hoeffding_ships"] or p["fresh"]["bernstein_ships"] or p["fresh"].get("betting_ships", False))),
                true_mean_below_d=sum(p["fresh"]["true_mean_below_d"] for p in ps),
                mean_W_median=float(np.median([p["fresh"]["mean_W"] for p in ps])),
                mean_W_max=float(max(p["fresh"]["mean_W"] for p in ps)),
                bernstein_bound_median=float(np.median([p["fresh"]["bernstein_bound"] for p in ps])),
                n_needed_bernstein_median=(float(np.median([p["fresh"]["n_needed_bernstein"] for p in ps if p["fresh"]["n_needed_bernstein"]]))
                                            if any(p["fresh"]["n_needed_bernstein"] for p in ps) else None),
                seconds_total=round(sum(p["seconds"] for p in ps), 1))
    n_ep = sorted({p["fresh"]["n"] for p in parts})
    merged.write_text(json.dumps(dict(registration=TAGS[tag]["registration"], n_episodes=(n_ep[0] if len(n_ep) == 1 else n_ep), rollout_seed0=TAGS[tag]["seed0"],
                                      summary=summary, tasks=parts), indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="real")
    ap.add_argument("--arm", default="corset")
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.merge:
        merge(a.tag); return
    lo, hi = (int(x) for x in a.seeds.split("-")) if "-" in a.seeds else (int(a.seeds), int(a.seeds))
    for s in range(lo, hi + 1):
        r = run_task(a.domain, a.arm, s, a.episodes, a.tag)
        f = r["fresh"]
        print(f"{a.domain}/{a.arm} seed {s}: repro={r['reproduces_archive']} k={f['k']}/{f['n']} meanW={f['mean_W']:.4f} "
              f"CP={f['cp_bound']:.3f} Hoeff={f['hoeffding_bound']:.3f} Bern={f['bernstein_bound']:.3f} Bet={f['betting_bound']:.3f} cross={f['betting_first_crossing']} ships(B/bet)={f['bernstein_ships']}/{f['betting_ships']} {r['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
