"""E13: is the policy-class choice a modelling preference or forced?

Reviews #46-#50 all press the same point: the necessity rate is 0.011
under a compiled load chain and 0.750 under a free class on the same 88
Prometheus rules, and the paper resolves it by giving each family "the
class its readings imply". Reviewers call that post hoc and unvalidated,
and ask for a Kyverno-to-control-model compiler to settle it.

The premise can be checked without building one. A compiled chain differs
from the free class only when a reading's cost depends on the ORDER or
RUN-LENGTH of decisions, which in this DSL means a temporal field. So the
question is whether Kyverno readings carry one. They do not, and this is
a property of the frozen grammar rather than a choice.

Run:  PYTHONPATH=. python3 -m saorl.e13_axis_check
"""
from __future__ import annotations
import json, os
from pathlib import Path
from saorl.benchmark_sg import run_benchmark as rb
from saorl.benchmark_sg.parse import (parse_kyverno, parse_prometheus,
                                      prom_threshold_bank)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/e2e" / "e13_axis_check.json"
TEMPORAL = ("for_s", "rate_window_s", "duration", "window")


def temporal_values(rep):
    return {f: getattr(rep, f) for f in TEMPORAL if hasattr(rep, f)}


def main() -> None:
    pt, _ = parse_prometheus(); kt, _ = parse_kyverno()
    bank = prom_threshold_bank(pt)
    prom_pools_with_varying_duration = 0
    for t in pt:
        vals = {getattr(c.rep, "for_s", None)
                for c in rb.build_prom_pool(t, bank).classes}
        if len({v for v in vals if v is not None}) > 1:
            prom_pools_with_varying_duration += 1
    kyv_readings = kyv_temporal = 0
    for t in kt:
        for c in rb.build_kyv_pool(t).classes:
            kyv_readings += 1
            if any(v not in (None, 0, 0.0, "", ())
                   for v in temporal_values(c.rep).values()):
                kyv_temporal += 1
    out = dict(
        registration="REGISTRATION_V10 E13",
        prom_pools=len(pt), kyv_pools=len(kt),
        prom_pools_with_varying_duration=prom_pools_with_varying_duration,
        prom_frac=round(prom_pools_with_varying_duration / len(pt), 4),
        kyv_readings=kyv_readings, kyv_readings_with_temporal_field=kyv_temporal,
    )
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"Prometheus: {prom_pools_with_varying_duration}/{len(pt)} pools have "
          f"readings differing in duration ({out['prom_frac']})")
    print(f"Kyverno:    {kyv_temporal}/{kyv_readings} readings carry any "
          f"temporal field")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
