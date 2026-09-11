"""Billing reconciliation for the whole demo (paper section 22.7).

(a)  internal: re-derive every billing_log.jsonl line's cost from its logged
     token counts x the pinned price sheet and confirm it equals the stored
     per-line cost and the re-summed total (exact to the cent);
(a') synthetic: prove the cost math is exact on known usage blocks;
(b)  provider: attempt the OpenAI Costs admin API cross-check (documents the
     403 for a project/user-scope key).

Writes demo/billing_reconciliation.json and prints the demo-wide total to the cent.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from billing import (cost_from_usage, load_price_sheet,
                     reconcile_with_usage_api)

HERE = Path(__file__).resolve().parent
BILLING_LOG = HERE / "billing_log.jsonl"
PARTIAL_LOG = HERE / "logs" / "billing_partial_run1.jsonl"  # crashed run (kept, real spend)
OUT = HERE / "billing_reconciliation.json"


def _sum_log(path):
    total = 0.0
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and json.loads(line).get("ok", True):
                total += float(json.loads(line)["total_cost"])
                n += 1
    return total, n


def internal_check(price_sheet):
    resummed = 0.0
    stored = 0.0
    n = 0
    max_line_err = 0.0
    first_ts = None
    last_ts = None
    with open(BILLING_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not d.get("ok", True):
                continue
            usage = {"prompt_tokens": d["prompt_tokens"],
                     "completion_tokens": d["completion_tokens"],
                     "prompt_tokens_details": {"cached_tokens": d["cached_tokens"]}}
            rc = cost_from_usage(usage, d["model"], price_sheet)["total_cost"]
            max_line_err = max(max_line_err, abs(rc - d["total_cost"]))
            resummed += rc
            stored += d["total_cost"]
            n += 1
            first_ts = d["ts"] if first_ts is None else min(first_ts, d["ts"])
            last_ts = d["ts"] if last_ts is None else max(last_ts, d["ts"])
    return dict(n_calls=n, resummed_total=resummed, stored_total=stored,
                abs_diff=abs(resummed - stored), max_per_line_diff=max_line_err,
                exact=abs(resummed - stored) < 1e-9 and max_line_err < 1e-9,
                first_ts=first_ts, last_ts=last_ts)


def synthetic_check(price_sheet, model="gpt-4o-mini"):
    r = price_sheet["models"][model]
    cases = [{"prompt_tokens": 1000, "completion_tokens": 500,
              "prompt_tokens_details": {"cached_tokens": 0}},
             {"prompt_tokens": 2000, "completion_tokens": 300,
              "prompt_tokens_details": {"cached_tokens": 800}}]
    checks = []
    for u in cases:
        cached = u["prompt_tokens_details"]["cached_tokens"]
        expect = ((u["prompt_tokens"] - cached) * r["input"]
                  + cached * r["cached_input"]
                  + u["completion_tokens"] * r["output"]) / 1e6
        got = cost_from_usage(u, model, price_sheet)["total_cost"]
        checks.append(dict(expect=expect, got=got, match=abs(expect - got) < 1e-15))
    return dict(all_match=all(c["match"] for c in checks), cases=checks)


def main():
    ps = load_price_sheet()
    internal = internal_check(ps)
    synth = synthetic_check(ps)
    start = int(internal["first_ts"] or time.time()) - 3600
    end = int(internal["last_ts"] or time.time()) + 3600
    recon = reconcile_with_usage_api(internal["stored_total"], start, end)
    partial_total, partial_n = (_sum_log(PARTIAL_LOG) if PARTIAL_LOG.exists()
                                else (0.0, 0))
    grand_total = internal["stored_total"] + partial_total
    out = {
        "billing_log": str(BILLING_LOG.name),
        "price_sheet_date": ps["pinned_date"],
        "primary_ledger_spend": internal["stored_total"],
        "partial_run1_spend": partial_total,
        "partial_run1_calls": partial_n,
        "partial_run1_note": ("a first log-gen attempt was aborted mid-run by a "
                              "transient TLS connection reset; its real charges "
                              "are preserved in logs/billing_partial_run1.jsonl "
                              "and included in the grand total below (honest "
                              "accounting -- the fix added network-error retries)."),
        "total_billed_spend": grand_total,
        "total_billed_spend_2dp": round(grand_total, 2),
        "internal_reconciliation": internal,
        "synthetic_accounting_proof": synth,
        "provider_costs_api": recon,
        "scope_note": ("costs are project/user-scope, reconciled internally "
                       "(billing_log vs re-derived usage x pinned price sheet); "
                       "the org Costs API needs an admin key (documented 403)."),
    }
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"PRIMARY LEDGER: ${internal['stored_total']:.6f} over {internal['n_calls']} calls")
    print(f"PARTIAL RUN 1 (kept): ${partial_total:.6f} over {partial_n} calls")
    print(f"GRAND TOTAL BILLED SPEND: ${grand_total:.6f} "
          f"(= ${round(grand_total,2):.2f})")
    print(f"internal exact match: {internal['exact']}  "
          f"(abs diff {internal['abs_diff']:.2e}, max line diff {internal['max_per_line_diff']:.2e})")
    print(f"synthetic proof all match: {synth['all_match']}")
    print(f"provider Costs API available: {recon.get('available')}  "
          f"({recon.get('reason','ok')[:80]})")
    print(f"wrote {OUT.name}")
    return out


if __name__ == "__main__":
    main()
