"""E3 stage 1 endpoints (REGISTRATION_V10 addendum), computed from the sweep.

Reads results/dsrl/e3/b*/vector_*.json (budgets 20 and 40) plus the existing
budget-10 leg in results/dsrl/vector/, and reports the three registered
endpoints. Nothing here chooses a threshold or a subset after the fact: a
cell-seed certifies iff BOTH channels ship under the driver's own frozen rule
(Clopper-Pearson upper <= SHIP_RATE), which is the same predicate the ledger
already uses.

E3-a  certified fraction per budget
E3-b  vector return, and its ratio to the single_a reference at that budget
E3-c  the portfolio arm: ship the certified fallback whenever a cell-seed
      fails, else ship vector. Certified by construction; the number that
      matters is its return.

Run:  PYTHONPATH=. python3 -m saorl.e3_analysis
"""
from __future__ import annotations

import collections
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results/e2e", "e3_budget_sweep.json")
FALLBACK_RET_NORM = 0.0   # the certified fallback's normalized return floor


def _load(pattern: str, budget: float) -> list:
    rows = []
    for f in glob.glob(pattern, recursive=True):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "arms" not in d:
            continue
        d["_budget"] = budget
        rows.append(d)
    return rows


def certifies(arm: dict) -> bool:
    c = arm.get("certs", {})
    return bool(c.get("a", {}).get("ships") and c.get("b", {}).get("ships"))


def main() -> None:
    legs = {
        10.0: os.path.join(ROOT, "results/dsrl", "vector", "**", "vector_*.json"),
        20.0: os.path.join(ROOT, "results/dsrl", "e3", "b20_*", "vector_*.json"),
        40.0: os.path.join(ROOT, "results/dsrl", "e3", "b40_*", "vector_*.json"),
    }
    out = {"registration": "REGISTRATION_V10 E3 stage 1", "legs": {}}
    for b, pat in legs.items():
        rows = _load(pat, b)
        if not rows:
            out["legs"][f"b={b:g}"] = {"n": 0, "note": "leg absent or not finished"}
            print(f"budget {b:g}: no results yet")
            continue
        vec = [r["arms"]["vector"] for r in rows if "vector" in r["arms"]]
        ref = [r["arms"]["single_a"] for r in rows if "single_a" in r["arms"]]
        n = len(vec)
        cert = [certifies(a) for a in vec]
        rv = [a["agg"].get("ret_norm") for a in vec if a["agg"].get("ret_norm") is not None]
        rr = [a["agg"].get("ret_norm") for a in ref if a["agg"].get("ret_norm") is not None]
        # E3-c: portfolio ships the fallback exactly when the certificate fails
        port = [(a["agg"].get("ret_norm") if ok else FALLBACK_RET_NORM)
                for a, ok in zip(vec, cert)
                if a["agg"].get("ret_norm") is not None]
        rec = dict(
            n=n,
            certified=sum(cert),
            certified_frac=round(sum(cert) / n, 4) if n else None,
            vector_ret_norm=round(st.mean(rv), 4) if rv else None,
            single_a_ret_norm=round(st.mean(rr), 4) if rr else None,
            ret_ratio=round(st.mean(rv) / st.mean(rr), 4)
            if rv and rr and st.mean(rr) else None,
            portfolio_ret_norm=round(st.mean(port), 4) if port else None,
            portfolio_certified_frac=1.0,
            viol_a=round(st.mean([a["agg"]["viol_a"] for a in vec]), 4) if vec else None,
            viol_b=round(st.mean([a["agg"]["viol_b"] for a in vec]), 4) if vec else None,
        )
        out["legs"][f"b={b:g}"] = rec
        print(f"budget {b:g}: n={n:3d}  certified {rec['certified']}/{n} "
              f"({rec['certified_frac']})  vector ret {rec['vector_ret_norm']}  "
              f"single_a {rec['single_a_ret_norm']}  ratio {rec['ret_ratio']}  "
              f"portfolio ret {rec['portfolio_ret_norm']}")

    done = [v for v in out["legs"].values() if v.get("n")]
    if len(done) >= 2:
        fr = [(k, v["certified_frac"]) for k, v in out["legs"].items() if v.get("n")]
        fr.sort(key=lambda kv: float(kv[0].split("=")[1]))
        rising = fr[-1][1] > fr[0][1]
        out["branch"] = ("1: certified fraction rises with budget"
                         if rising else
                         "2: certified fraction flat or falling -- the abstract "
                         "must state that no usable neural operating point "
                         "exists at this scale")
        print(f"\nregistered branch -> {out['branch']}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
