"""Channel-session agent for the Slack posting-rate demo (paper section 22.3).

The bounded research-and-summarize agent of agent_loop.py, recast as a Slack app
posting to a channel so the posting-rate QUOTA actually binds:

  * a CHANNEL SESSION = one episode = a short burst window of W = 12 posting
    opportunities, each a distinct research question (a message to compose);
  * per step the agent chooses the PAID action ("buy" = compose+post the message
    with a real billed gpt-4o-mini call, earning that message's task value) or
    the FREE fallback ("skip" = post nothing / a canned extractive snippet, no
    billed call, lower value);
  * `calls` = the running count of paid posts in the window is the quota axis a
    reading psi_k caps (calls >= k forbids the k-th+ paid post).

Each step is an INDEPENDENT question (its own source doc + fresh retrieval), so a
step's paid-answer value and free-fallback value are well defined regardless of
earlier actions. We log BOTH at every step, which makes the value of any
calls-threshold policy exactly replayable offline (needed for Thm-5 VoQ):

    value of "cap at k" on an episode = sum_{t<k} paid_value_t + sum_{t>=k} free_value_t

A return-greedy agent posts every step (max value, violates every cap); a guard
that honors psi_k stops paying at k posts. This is the paid-vs-fallback economics
of saorl's budget domain, now on the real posting-rate axis with real bills.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import requests

from agent_loop import (FREE_SNIPPET_CHARS, PAID_SYSTEM, GuardLike, Passthrough,
                        _tokens, _uncovered_terms, retrieve, score_answer)
from billing import BillingLedger, PaidToolError, call_openai

HERE = Path(__file__).resolve().parent
WINDOW = 12  # posting opportunities per channel session


# --------------------------------------------------------------------------- #
# Thread-safe ledger + stop-loss (episodes run in parallel; the ledger is shared)
# --------------------------------------------------------------------------- #
class LockedLedger(BillingLedger):
    """BillingLedger with a lock around record()/total so parallel episodes can
    bill concurrently without corrupting the running total or the JSONL log."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._lock = threading.Lock()

    def record(self, *a, **kw):
        with self._lock:
            return super().record(*a, **kw)


class StopLossGuard(GuardLike):
    """Passthrough until measured spend reaches `limit`, then deny PAID."""
    def __init__(self, ledger: BillingLedger, limit: float):
        self.ledger = ledger
        self.limit = limit

    def allow_paid(self, features: Dict[str, Any]) -> bool:
        return self.ledger.total_cost < self.limit


class TableGuard(GuardLike):
    """A frozen guard replayed from a checkpoint: allow_by_calls[c] gives the
    paid-allow decision at running-post-count c (>= len -> use the last entry).
    Decouples live eval from FQI retraining -- the policy is exactly the saved
    one."""
    def __init__(self, allow_by_calls: List[bool], mode: str = "table"):
        self.allow_by_calls = list(allow_by_calls)
        self.mode = mode

    def allow_paid(self, features: Dict[str, Any]) -> bool:
        c = int(features["calls"])
        if c >= len(self.allow_by_calls):
            c = len(self.allow_by_calls) - 1
        return bool(self.allow_by_calls[c])


class AndGuard(GuardLike):
    """Allow PAID iff every wrapped guard allows it (e.g. trained guard AND
    stop-loss)."""
    def __init__(self, *guards: GuardLike):
        self.guards = guards

    def allow_paid(self, features: Dict[str, Any]) -> bool:
        return all(g.allow_paid(features) for g in self.guards)


# --------------------------------------------------------------------------- #
# Channel behavior / target planners (buy = post a paid message)
# --------------------------------------------------------------------------- #
Planner = Callable[[Dict[str, Any], np.random.Generator], str]


def greedy_post_planner() -> Planner:
    def plan(feat, rng):
        return "buy"
    return plan


def randomized_post_planner(p_buy: float = 0.55) -> Planner:
    def plan(feat, rng):
        return "buy" if rng.random() < p_buy else "skip"
    return plan


def mixture_behavior_planner(p_greedy: float = 0.5, p_buy: float = 0.55) -> Planner:
    """Logging behavior: per-episode mix of greedy-post and randomized-fallback.
    Greedy episodes post every step (full paid/free info for every calls bucket +
    exact VoQ replay); randomized episodes give skip-coverage across buckets."""
    greedy = greedy_post_planner()
    rand = randomized_post_planner(p_buy)
    def plan(feat, rng):
        h = (hash((feat.get("episode", 0), "mix")) % 1000) / 1000.0
        return greedy(feat, rng) if h < p_greedy else rand(feat, rng)
    return plan


# --------------------------------------------------------------------------- #
# Resilient paid call (retry transient errors before degrading)
# --------------------------------------------------------------------------- #
def call_openai_retry(prompt: str, model: str, *, system: str,
                      max_output_tokens: int, retries: int = 6,
                      base_sleep: float = 2.0) -> Dict[str, Any]:
    """Resilient paid call: retries transient HTTP (429/5xx) AND raw network
    errors (connection reset, TLS handshake, timeout) with exponential backoff
    + jitter, so a dropped socket never crashes a run."""
    last = None
    for i in range(retries):
        try:
            return call_openai(prompt, model, system=system,
                               max_output_tokens=max_output_tokens)
        except requests.exceptions.RequestException as e:
            # raw network layer error (ConnectionReset, TLS, timeout) -> retry
            last = PaidToolError(f"network: {e}")
            time.sleep(base_sleep * (2 ** i) + np.random.random())
            continue
        except PaidToolError as e:
            last = e
            msg = str(e)
            if any(s in msg for s in ("429", "500", "502", "503", "504",
                                      "timeout", "Timeout", "Connection")):
                time.sleep(base_sleep * (2 ** i) + np.random.random())
                continue
            raise
    raise last


# --------------------------------------------------------------------------- #
# Episode record
# --------------------------------------------------------------------------- #
@dataclass
class ChannelEpisode:
    episode_id: int
    task_ids: List[str]
    trajectory: List[Dict[str, Any]]        # state before each action: spend, calls, step
    actions: List[str]                       # "buy"/"skip"
    rewards: List[float]                     # realized per-step value
    paid_value: List[Optional[float]]        # rubric value of the paid answer (if posted)
    free_value: List[float]                  # rubric value of the free fallback (always)
    costs: List[float]                       # billed $ per step
    gated: List[bool]                        # guard vetoed a desired paid post
    n_paid: int
    total_cost: float
    total_value: float
    n_paid_attempts: int = 0
    last_error: Optional[str] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_ids": self.task_ids,
            "trajectory": self.trajectory,
            "actions": self.actions,
            "rewards": self.rewards,
            "paid_value": self.paid_value,
            "free_value": self.free_value,
            "costs": self.costs,
            "gated": self.gated,
            "n_paid": self.n_paid,
            "total_cost": self.total_cost,
            "total_value": self.total_value,
            "n_paid_attempts": self.n_paid_attempts,
            "last_error": self.last_error,
        }


def _free_answer(task: Dict[str, Any], corpus: Dict[str, List[str]]) -> str:
    """Deterministic FREE fallback: truncated extractive snippet for the question."""
    snips = retrieve(_tokens(task["question"]), task["source_docs"], corpus,
                     used=set(), top_k=1)
    return snips[0][:FREE_SNIPPET_CHARS] if snips else ""


def _paid_prompt(task: Dict[str, Any], corpus: Dict[str, List[str]]) -> str:
    terms = _tokens(task["question"]) + _uncovered_terms("", task["rubric"])
    ctx = retrieve(terms, task["source_docs"], corpus, used=set(), top_k=3)
    return (f"Question: {task['question']}\n\nContext:\n- " + "\n- ".join(ctx)
            + "\n\nAnswer:")


def run_channel_episode(task_batch: Sequence[Dict[str, Any]], guard: GuardLike,
                        *, model: str, ledger: BillingLedger, planner: Planner,
                        corpus: Dict[str, List[str]], episode_id: int,
                        rng: Optional[np.random.Generator] = None,
                        paid_max_tokens: int = 160) -> ChannelEpisode:
    """One channel session. PAID posts bill real dollars via `ledger`."""
    rng = rng or np.random.default_rng(episode_id + 1)
    calls = 0
    spend = 0.0
    traj, actions, rewards = [], [], []
    paid_value, free_value, costs, gated_l = [], [], [], []
    n_paid_attempts = 0
    last_error = None

    for t, task in enumerate(task_batch):
        state = {"spend": round(spend, 8), "calls": calls, "step": t}
        feat = dict(state, episode=episode_id)

        fv = score_answer(_free_answer(task, corpus), task["rubric"])
        desired = planner(feat, rng)
        gated = False
        if desired == "buy" and not guard.allow_paid(feat):
            action = "skip"
            gated = True
        else:
            action = desired

        step_cost = 0.0
        pv: Optional[float] = None
        if action == "buy":
            n_paid_attempts += 1
            try:
                resp = call_openai_retry(_paid_prompt(task, corpus), model,
                                         system=PAID_SYSTEM,
                                         max_output_tokens=paid_max_tokens)
                rec = ledger.record(resp["usage"], model, episode=episode_id,
                                    step=t, request_id=resp["request_id"])
                step_cost = rec.total_cost
                pv = score_answer(resp["text"], task["rubric"])
                calls += 1
                spend += step_cost
            except PaidToolError as e:
                last_error = str(e)[:200]
                action = "skip"      # degrade to free on a hard API failure
                gated = False

        realized = pv if action == "buy" else fv

        traj.append(state)
        actions.append(action)
        rewards.append(float(realized))
        paid_value.append(pv)
        free_value.append(float(fv))
        costs.append(step_cost)
        gated_l.append(gated)

    return ChannelEpisode(
        episode_id=episode_id,
        task_ids=[tk["id"] for tk in task_batch],
        trajectory=traj, actions=actions, rewards=rewards,
        paid_value=paid_value, free_value=free_value, costs=costs, gated=gated_l,
        n_paid=sum(1 for a in actions if a == "buy"),
        total_cost=float(sum(costs)), total_value=float(sum(rewards)),
        n_paid_attempts=n_paid_attempts, last_error=last_error,
    )


def make_batches(tasks: Sequence[Dict[str, Any]], n_episodes: int,
                 window: int = WINDOW) -> List[List[Dict[str, Any]]]:
    """Deterministic per-episode batches of `window` distinct questions, cycling
    the task pool so every card is exercised and steps are independent."""
    n = len(tasks)
    batches = []
    for e in range(n_episodes):
        batches.append([tasks[(e * window + t) % n] for t in range(window)])
    return batches
