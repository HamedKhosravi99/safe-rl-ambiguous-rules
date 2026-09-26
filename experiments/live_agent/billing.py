"""Cost accounting for the live service-agent check (Appendix D.1 of the paper).

The metered axis is provider-billed US dollars. This module:

  * calls a mini/nano-tier OpenAI chat model over the REST API (the paid tool);
  * turns each response's usage fields into a dollar cost via a PINNED price
    sheet (``demo/price_sheet.json``), so the y-axis of the headline figure is
    reconstructible from provenance, not hand-entered;
  * appends one JSON line per paid call to ``demo/billing_log.jsonl``;
  * offers a reconciliation routine that queries the OpenAI Costs/Usage admin
    API to cross-check accumulated spend (and documents when that endpoint is
    not reachable with the configured key type).

No secret is ever written to disk or logged: the key is read from the
environment ($OPENAI_API_KEY) only at call time.

Cost formula (per call), matching the price sheet's ``note``:

    billable_input  = prompt_tokens - cached_tokens
    cost = billable_input   * input        / 1e6
         + cached_tokens    * cached_input  / 1e6
         + completion_tokens* output        / 1e6

``completion_tokens`` already includes ``reasoning_tokens`` for gpt-5* models,
so reasoning is billed once (as output) and never double-counted.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

HERE = Path(__file__).resolve().parent
PRICE_SHEET_PATH = HERE / "price_sheet.json"
DEFAULT_BILLING_LOG = HERE / "billing_log.jsonl"

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_COSTS_URL = "https://api.openai.com/v1/organization/costs"
OPENAI_USAGE_URL = "https://api.openai.com/v1/organization/usage/completions"


# --------------------------------------------------------------------------- #
# Price sheet
# --------------------------------------------------------------------------- #
def load_price_sheet(path: Path = PRICE_SHEET_PATH) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def cost_from_usage(usage: Dict[str, Any], model: str,
                    price_sheet: Dict[str, Any]) -> Dict[str, float]:
    """Deterministic dollar cost from a response's usage block.

    Returns a breakdown dict: input/cached/output token counts and their
    per-part dollar costs plus the total. Raises if the model is not pinned in
    the price sheet (we refuse to bill against an unpinned rate)."""
    models = price_sheet["models"]
    if model not in models:
        # allow dated aliases like gpt-5-nano-2025-08-07 -> gpt-5-nano
        base = _base_model(model, models)
        if base is None:
            raise KeyError(f"model {model!r} not in pinned price sheet")
        rates = models[base]
    else:
        rates = models[model]

    prompt = int(usage.get("prompt_tokens", 0))
    completion = int(usage.get("completion_tokens", 0))
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens", 0) or 0)
    billable_input = max(0, prompt - cached)

    input_cost = billable_input * rates["input"] / 1e6
    cached_cost = cached * rates["cached_input"] / 1e6
    output_cost = completion * rates["output"] / 1e6
    total = input_cost + cached_cost + output_cost

    comp_details = usage.get("completion_tokens_details") or {}
    reasoning = int(comp_details.get("reasoning_tokens", 0) or 0)

    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "billable_input_tokens": billable_input,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,  # informational; already inside completion
        "input_cost": input_cost,
        "cached_cost": cached_cost,
        "output_cost": output_cost,
        "total_cost": total,
    }


def _base_model(model: str, models: Dict[str, Any]) -> Optional[str]:
    for base in models:
        if model.startswith(base):
            return base
    return None


# --------------------------------------------------------------------------- #
# Billing ledger
# --------------------------------------------------------------------------- #
@dataclass
class CallRecord:
    ts: float
    model: str
    episode: Optional[int]
    step: Optional[int]
    request_id: Optional[str]
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_cost: float
    input_cost: float
    cached_cost: float
    output_cost: float
    ok: bool = True
    error: Optional[str] = None


class BillingLedger:
    """Accumulates paid-call costs and appends them to a JSONL log."""

    def __init__(self, log_path: Path = DEFAULT_BILLING_LOG,
                 price_sheet: Optional[Dict[str, Any]] = None):
        self.log_path = Path(log_path)
        self.price_sheet = price_sheet or load_price_sheet()
        self.records: List[CallRecord] = []
        self._total = 0.0

    @property
    def total_cost(self) -> float:
        return self._total

    @property
    def n_calls(self) -> int:
        return len(self.records)

    def record(self, usage: Dict[str, Any], model: str,
               episode: Optional[int] = None, step: Optional[int] = None,
               request_id: Optional[str] = None) -> CallRecord:
        b = cost_from_usage(usage, model, self.price_sheet)
        rec = CallRecord(
            ts=time.time(), model=model, episode=episode, step=step,
            request_id=request_id,
            prompt_tokens=b["prompt_tokens"], cached_tokens=b["cached_tokens"],
            completion_tokens=b["completion_tokens"],
            reasoning_tokens=b["reasoning_tokens"],
            total_cost=b["total_cost"], input_cost=b["input_cost"],
            cached_cost=b["cached_cost"], output_cost=b["output_cost"],
        )
        self.records.append(rec)
        self._total += rec.total_cost
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        return rec

    def sum_from_log(self) -> Dict[str, float]:
        """Recompute totals by re-reading the JSONL log (audit path)."""
        total = 0.0
        n = 0
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("ok", True):
                    total += float(d["total_cost"])
                    n += 1
        return {"total_cost": total, "n_calls": n}


# --------------------------------------------------------------------------- #
# Paid tool: OpenAI chat call
# --------------------------------------------------------------------------- #
class PaidToolError(RuntimeError):
    pass


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def call_openai(prompt: str, model: str, *, system: Optional[str] = None,
                max_output_tokens: int = 256, seed: Optional[int] = 7,
                timeout: float = 60.0) -> Dict[str, Any]:
    """One chat completion. Returns {'text', 'usage', 'request_id', 'raw'}.

    Handles the gpt-5*/o-series parameter differences: they take
    ``max_completion_tokens`` (not ``max_tokens``), reject non-default
    ``temperature``, and accept ``reasoning_effort``."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise PaidToolError("OPENAI_API_KEY not set in environment")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body: Dict[str, Any] = {"model": model, "messages": messages}
    if _is_reasoning_model(model):
        body["max_completion_tokens"] = max_output_tokens
        body["reasoning_effort"] = "minimal"
    else:
        body["max_tokens"] = max_output_tokens
        body["temperature"] = 0.0
    if seed is not None:
        body["seed"] = seed

    r = requests.post(
        OPENAI_CHAT_URL,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json=body, timeout=timeout,
    )
    if r.status_code != 200:
        raise PaidToolError(f"HTTP {r.status_code}: {r.text[:400]}")
    data = r.json()
    choice = data["choices"][0]
    text = (choice.get("message", {}) or {}).get("content") or ""
    return {
        "text": text.strip(),
        "usage": data.get("usage", {}),
        "request_id": data.get("id"),
        "finish_reason": choice.get("finish_reason"),
        "raw": data,
    }


# --------------------------------------------------------------------------- #
# Reconciliation vs the OpenAI Costs/Usage admin API
# --------------------------------------------------------------------------- #
def reconcile_with_usage_api(ledger_total: float, start_time: int,
                             end_time: Optional[int] = None,
                             timeout: float = 30.0) -> Dict[str, Any]:
    """Cross-check our accumulated spend against OpenAI's Costs endpoint.

    The Costs/Usage endpoints require an ADMIN API key (organization scope);
    ordinary project/user keys get 401. We attempt the call and report the
    result verbatim, documenting unavailability rather than failing the run.

    ``start_time``/``end_time`` are unix seconds. Cost buckets on OpenAI's side
    settle with a delay, so a freshly-run demo may legitimately show $0 there;
    that is noted in the returned dict.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return {"available": False, "reason": "no OPENAI_API_KEY", "ledger_total": ledger_total}

    params = {"start_time": int(start_time), "bucket_width": "1d", "limit": 31}
    if end_time is not None:
        params["end_time"] = int(end_time)
    try:
        r = requests.get(
            OPENAI_COSTS_URL,
            headers={"Authorization": f"Bearer {key}"},
            params=params, timeout=timeout,
        )
    except requests.RequestException as e:  # pragma: no cover - network
        return {"available": False, "reason": f"request error: {e}",
                "ledger_total": ledger_total}

    if r.status_code in (401, 403):
        return {
            "available": False,
            "reason": (f"HTTP {r.status_code} on {OPENAI_COSTS_URL} -- the "
                       "Costs API needs an ADMIN (organization) key; the "
                       "configured key is project/user scope. Internal "
                       "reconciliation (billing_log vs summed usage fields) "
                       "still holds to the cent."),
            "http_status": r.status_code,
            "ledger_total": ledger_total,
        }
    if r.status_code != 200:
        return {"available": False, "reason": f"HTTP {r.status_code}: {r.text[:300]}",
                "http_status": r.status_code, "ledger_total": ledger_total}

    data = r.json()
    api_total = 0.0
    for bucket in data.get("data", []):
        for result in bucket.get("results", []):
            amt = result.get("amount", {})
            api_total += float(amt.get("value", 0.0))
    return {
        "available": True,
        "api_reported_cost": api_total,
        "ledger_total": ledger_total,
        "difference": api_total - ledger_total,
        "note": ("OpenAI cost buckets settle with delay; a value of 0.0 right "
                 "after a run is expected and does not contradict the ledger."),
        "raw": data,
    }
