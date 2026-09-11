"""A8: fixed-threshold protocol sensitivity + execution consistency (E0-D).

Maps are pool-internal and total: A1 = G(a); A2 loosest (max fire-count
class); A3 median-by-threshold-rank in A1's subfamily; A4 strictest
(min nonzero fire-count). Threshold q = 0.245 (the study's chronological
calibrate-once value) is FROZEN; primary endpoints never recalibrate.

Run: SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.benchmark_sg.t4_sensitivity
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .evaluate import build_prom_pool
from .g0_audit import dev_targets
from .parse import prom_threshold_bank
from .run_benchmark import _prom_skeleton, _record
from saorl.benchmark_sg import control_suite as CS

_ROOT = Path(__file__).resolve().parents[3]
_OUT = _ROOT / "results/conformal" / "e0" / "t4_sensitivity.json"
Q_FIXED = 0.245
PROMTOOL = str(Path.home() / "bin" / "promtool")


def render_expr(r) -> str:
    sel = ",".join(f'{k}{op}"{v}"' for k, op, v in r.selectors)
    base = f"{r.metric}{{{sel}}}" if sel else r.metric
    if r.rate_window_s:
        base = f"rate({base}[{int(r.rate_window_s)}s])"
    if (r.aggregation or "none") != "none":
        base = f"{r.aggregation}({base})"
    return f"{base} {r.comparator} {r.threshold:g}"


def promtool_valid(r) -> tuple:
    rule = {"groups": [{"name": "g", "rules": [{
        "alert": "X", "expr": render_expr(r),
        **({"for": f"{int(r.for_s)}s"} if r.for_s else {})}]}]}
    import yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yml",
                                     delete=False) as fh:
        yaml.safe_dump(rule, fh)
        p = fh.name
    res = subprocess.run([PROMTOOL, "check", "rules", p],
                         capture_output=True, text=True)
    ok = res.returncode == 0
    return ok, (res.stderr or res.stdout).strip()[:160] if not ok else ""


def subfamily(rec, gold_rep):
    fam = []
    for i, cls in enumerate(rec_classes(rec)):
        rep = cls.rep
        if (rep.metric, rep.selectors, rep.aggregation, rep.agg_by) == \
           (gold_rep.metric, gold_rep.selectors, gold_rep.aggregation,
                gold_rep.agg_by) and rep.comparator in (">", ">="):
            fam.append((i, rep))
    return fam


def rec_classes(rec):
    return rec.pool.classes


def main():
    devs = dev_targets()
    bank = prom_threshold_bank(devs)
    maps = ("A1", "A2", "A3", "A4")
    agree = Counter()
    contain = Counter()
    n_by = Counter()
    per_repo = defaultdict(lambda: Counter())
    pt_ok, pt_fail, pt_tax, pt_examples = 0, 0, Counter(), []
    for t in devs:
        pool = build_prom_pool(t, bank)
        rec = _record(pool, t, _prom_skeleton(t))
        rec.pool = pool
        vec_sums = [sum(v) for v in rec.vectors]
        gold_i = pool.gold_idx
        gold_rep = pool.classes[gold_i].rep
        # map targets (class indices)
        tgt = {"A1": gold_i}
        tgt["A2"] = int(np.argmax(vec_sums))
        nz = [(s if s > 0 else np.inf) for s in vec_sums]
        tgt["A4"] = int(np.argmin(nz)) if any(np.isfinite(nz)) else gold_i
        fam = subfamily(rec, gold_rep)
        if len(fam) >= 2:
            fam_sorted = sorted(fam, key=lambda it: (it[1].threshold,
                                                     it[1].for_s))
            tgt["A3"] = fam_sorted[len(fam_sorted) // 2][0]
        else:
            tgt["A3"] = gold_i
        for m in maps:
            n_by[m] += 1
            agree[m] += (tgt[m] == gold_i)
            contain[m] += (rec.class_scores[tgt[m]] >= Q_FIXED)
            per_repo[t.repo][m + "_contain"] += \
                (rec.class_scores[tgt[m]] >= Q_FIXED)
            per_repo[t.repo][m + "_n"] += 1
        ok, err = promtool_valid(t.reading)
        if ok:
            pt_ok += 1
        else:
            pt_fail += 1
            key = err.split(":")[-1].strip()[:60] or "unknown"
            pt_tax[key] += 1
            if len(pt_examples) < 3:
                pt_examples.append(dict(rule=t.name,
                                        expr=render_expr(t.reading),
                                        err=err))

    # P1 price deltas per map analogue (tightest/loosest/median readings)
    suite = json.loads((_ROOT / "results/conformal" / "benchmark_sg" /
                        "control_suite.json").read_text())
    price_rows = []
    for inst in suite["instances"]:
        readings = [type("R", (), dict(threshold=x["theta"],
                                       for_s=x["for_s"]))()
                    for x in inst["readings"]]
        m = CS.compile_instance(readings)
        d = 0.01
        full = CS.solve(m, {k: d for k in range(m["C"].shape[0])})
        if not full.get("feasible"):
            continue
        ordered = sorted(range(len(readings)),
                         key=lambda i: (readings[i].threshold,
                                        readings[i].for_s))
        picks = dict(A1_tightest=ordered[-1], A2_loosest=ordered[0],
                     A3_median=ordered[len(ordered) // 2])
        poas = {}
        for nm, k in picks.items():
            single = CS.solve(m, {k: d})
            poas[nm] = (single["ret"] - full["ret"]) \
                if single.get("feasible") else None
        vals = [v for v in poas.values() if v is not None]
        wd = max(abs(v - poas["A1_tightest"]) for v in vals) \
            if poas.get("A1_tightest") is not None and vals else None
        price_rows.append(dict(uid=inst["uid"], poa=poas, worst_delta=wd))

    rep = dict(
        q_fixed=Q_FIXED, n=len(devs),
        agreement={m: round(agree[m] / n_by[m], 4) for m in maps},
        containment_at_fixed_q={m: round(contain[m] / n_by[m], 4)
                                for m in maps},
        generation_note="pool-internal maps: P(A_j in Psi) = 1 by "
                        "construction (stated per A8)",
        per_repo={r: {k: v for k, v in c.items()}
                  for r, c in per_repo.items()},
        ask_act_flip="not applicable (per-map answer priors undefined); "
                     "disclosed per A8",
        execution_consistency=dict(
            tool="promtool check rules",
            valid=pt_ok, invalid=pt_fail,
            validity_rate=round(pt_ok / (pt_ok + pt_fail), 4),
            taxonomy=dict(pt_tax.most_common(6)),
            examples=pt_examples),
        p1_price_sensitivity=dict(
            n_instances=len(price_rows),
            worst_delta_max=max((r["worst_delta"] for r in price_rows
                                 if r["worst_delta"] is not None),
                                default=None),
            rows=price_rows),
        scope="sensitivity to deterministic operational protocols; does "
              "not validate agreement with latent human intent",
    )
    _OUT.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in
                      ("agreement", "containment_at_fixed_q",
                       "execution_consistency")}, indent=1, default=str))
    print("P1 worst PoA delta:",
          rep["p1_price_sensitivity"]["worst_delta_max"])


if __name__ == "__main__":
    main()
