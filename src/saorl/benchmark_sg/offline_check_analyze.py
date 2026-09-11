"""Aggregate the held-out Offline-CHECK runs into the comparison tables.

Reads results/offline_check_experiment/raw_runs.json, writes summary.json and
summary.md in the same directory. No paper artefacts are touched.

Definitions (fixed before looking at the numbers):
  unsafe ship  = certificate says SHIP but the policy is unsafe in the TRUE
                 model. This is the only error that matters for deployment.
  conservative = certificate says NO-SHIP but the policy is in fact safe.
                 This is the price of the certificate, not a failure.
  ship rate    = fraction of runs certified. Useful only alongside unsafe ship.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_OUT = Path(__file__).resolve().parents[3] / "results" / "offline_check_experiment"


def rates(rows, ship_key):
    sh = np.array([r[ship_key] for r in rows], dtype=bool)
    tr = np.array([r["true_safe"] for r in rows], dtype=bool)
    n = len(rows)
    unsafe = int((sh & ~tr).sum())
    return dict(
        n=n,
        ship_rate=float(sh.mean()),
        true_safe_rate=float(tr.mean()),
        unsafe_ship=unsafe,
        unsafe_ship_rate=float(unsafe / n) if n else float("nan"),
        conservative=int((~sh & tr).sum()),
        conservative_rate=float((~sh & tr).sum() / n) if n else float("nan"),
    )


def wilson_hi(k, n, z=1.96):
    """Upper Wilson bound on the unsafe-ship rate (so 0/N is reported honestly)."""
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c + h) / d)


def main():
    raw = json.loads((_OUT / "raw_runs.json").read_text())
    rows = raw["rows"]
    grid = raw["n_check_grid"]
    d = raw["budget"]

    out = {"protocol": raw["protocol"], "n_train": raw["n_train"],
           "budget": d, "delta_ev": raw["delta_ev"],
           "n_rows": len(rows), "n_seeds": len(raw["seeds"]),
           "n_instances": len(set(r["rule_id"] for r in rows)),
           "arms": sorted(set(r["arm"] for r in rows))}

    # ---- headline: current CHECK vs offline CHECK vs plug-in --------------
    out["current_check"] = rates(rows, "cur_ship")
    out["current_check"]["unsafe_ship_rate_wilson_hi"] = wilson_hi(
        out["current_check"]["unsafe_ship"], len(rows))
    out["offline_check"] = {}
    out["plugin_control"] = {}
    for n in grid:
        oc = rates(rows, f"off_ship_{n}")
        oc["unsafe_ship_rate_wilson_hi"] = wilson_hi(oc["unsafe_ship"], len(rows))
        oc["median_bound"] = float(np.median([r[f"off_bound_{n}"] for r in rows]))
        oc["median_bound_over_budget"] = oc["median_bound"] / d
        oc["median_true_cmax"] = float(np.median([r["true_cmax"] for r in rows]))
        oc["frac_unsupported_med"] = float(np.median(
            [r[f"frac_unsup_{n}"] for r in rows]))
        oc["median_radius_med"] = float(np.median(
            [r[f"medrad_{n}"] for r in rows if r[f"medrad_{n}"] is not None]))
        out["offline_check"][str(n)] = oc

        pc = rates(rows, f"plug_ship_{n}")
        pc["unsafe_ship_rate_wilson_hi"] = wilson_hi(pc["unsafe_ship"], len(rows))
        pc["median_bound"] = float(np.median([r[f"plug_bound_{n}"] for r in rows]))
        out["plugin_control"][str(n)] = pc

    # ---- per-arm at the largest n_check ----------------------------------
    nmax = max(grid)
    out["per_arm_at_nmax"] = {}
    for arm in out["arms"]:
        rs = [r for r in rows if r["arm"] == arm]
        out["per_arm_at_nmax"][arm] = dict(
            offline=rates(rs, f"off_ship_{nmax}"),
            current=rates(rs, "cur_ship"))

    # ---- coverage-failure anatomy: where does the bound come from? -------
    # decompose the median bound at nmax into true cost + certificate slack
    out["slack_anatomy"] = {}
    for n in grid:
        sl = [r[f"off_bound_{n}"] - r["true_cmax"] for r in rows]
        out["slack_anatomy"][str(n)] = dict(
            median_slack=float(np.median(sl)),
            median_slack_over_budget=float(np.median(sl) / d),
            frac_negative_slack=float(np.mean(np.array(sl) < 0)),
        )

    (_OUT / "summary.json").write_text(json.dumps(out, indent=2))

    # ---- markdown ---------------------------------------------------------
    L = []
    L.append("# Held-out Offline-CHECK: results\n")
    L.append(f"- protocol: {raw['protocol']}")
    L.append(f"- {out['n_instances']} instances x {out['n_seeds']} seeds x "
             f"{len(out['arms'])} learner arms = {out['n_rows']} runs")
    L.append(f"- n_train = {raw['n_train']} (fixed), budget d = {d}, "
             f"delta = {raw['delta_ev']}\n")
    L.append("## Headline: does the certificate ever ship an unsafe policy?\n")
    L.append("| certificate | n_check | ship rate | unsafe ships | unsafe rate (95% upper) |")
    L.append("|---|---|---|---|---|")
    c = out["current_check"]
    L.append(f"| current CHECK (fresh true-model samples, NOT offline) | "
             f"{raw['n_train'] and N_EV_LABEL} | {c['ship_rate']:.3f} | "
             f"{c['unsafe_ship']} | {c['unsafe_ship_rate']:.4f} "
             f"({c['unsafe_ship_rate_wilson_hi']:.4f}) |")
    for n in grid:
        o = out["offline_check"][str(n)]
        L.append(f"| held-out Offline-CHECK | {n} | {o['ship_rate']:.3f} | "
                 f"{o['unsafe_ship']} | {o['unsafe_ship_rate']:.4f} "
                 f"({o['unsafe_ship_rate_wilson_hi']:.4f}) |")
    for n in grid:
        p = out["plugin_control"][str(n)]
        L.append(f"| plug-in control (NO uncertainty term) | {n} | "
                 f"{p['ship_rate']:.3f} | {p['unsafe_ship']} | "
                 f"{p['unsafe_ship_rate']:.4f} "
                 f"({p['unsafe_ship_rate_wilson_hi']:.4f}) |")
    L.append(f"\n- true safety rate of the trained policies: "
             f"{out['current_check']['true_safe_rate']:.3f}\n")
    L.append("## How tight is the offline bound?\n")
    L.append("| n_check | median bound | / budget | median slack over true cost | median L1 radius | frac unsupported (s,a) used |")
    L.append("|---|---|---|---|---|---|")
    for n in grid:
        o = out["offline_check"][str(n)]
        s = out["slack_anatomy"][str(n)]
        L.append(f"| {n} | {o['median_bound']:.5f} | "
                 f"{o['median_bound_over_budget']:.1f}x | "
                 f"{s['median_slack']:.5f} | {o['median_radius_med']:.3f} | "
                 f"{o['frac_unsupported_med']:.3f} |")
    (_OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


N_EV_LABEL = 20000

if __name__ == "__main__":
    main()
