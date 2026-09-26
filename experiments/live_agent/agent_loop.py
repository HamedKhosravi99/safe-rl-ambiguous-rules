"""Bounded tool-using agent for the live service-agent check (Appendix D.1 of the paper).

Task family: research-and-summarize over a LOCAL document corpus. Each episode
is one task card (demo/tasks/*.json) answerable from the bundled corpus
(demo/corpus_docs/*.txt). Per step the agent chooses, via a planner, between:

  * PAID action ("buy"): call the metered OpenAI mini/nano model to synthesize
    an answer chunk from retrieved context -- REAL billed tokens, added to the
    running $ spend by billing.BillingLedger.
  * FREE fallback ("skip"): local keyword retrieval + an extractive snippet --
    no dollars, lower expected task value.

This is exactly the paid-vs-fallback structure of the saorl budget domain, now
with real bills. Task value is graded 0-1 by a DETERMINISTIC keyword-rubric
scorer (score_answer) -- no LLM judging in the loop.

The gate is applied by a guard (see demo/guard.py): the planner proposes a
desired action; if it wants PAID and the guard denies it, the step is forced to
the FREE fallback. Episodes serialize to the saorl OfflineDataset schema
(trajectories of state dicts with 'spend' and 'calls', actions in {buy, skip},
rewards) so the existing FQI learner can consume the logs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from billing import BillingLedger, call_openai, PaidToolError, load_price_sheet

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "corpus_docs"
TASKS_DIR = HERE / "tasks"

MAX_STEPS = 12
# The FREE fallback returns a short EXTRACTIVE snippet (a truncated excerpt of
# the best-matching sentence), so it reliably covers a fact near the start of a
# sentence but misses trailing facts -- a genuine, deterministic value gap the
# PAID LLM (which synthesizes the full answer) can close. This is the demo's
# paid-vs-fallback economics (Appendix D.1): paid earns more task value.
FREE_SNIPPET_CHARS = 60
_WORD = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Corpus / tasks
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def load_corpus(docs_dir: Path = DOCS_DIR) -> Dict[str, List[str]]:
    """Return {doc_name: [sentences]}. Sentences are the retrieval units.

    The bundled corpus is authored one sentence per line, so we split on
    newlines (not on '.', which would shatter decimals like '4.6'). Any line
    that still contains multiple sentences is left intact -- harmless for
    retrieval and it keeps numeric tokens whole for the rubric scorer."""
    corpus: Dict[str, List[str]] = {}
    for p in sorted(docs_dir.glob("*.txt")):
        sents = [ln.strip().rstrip(".").strip()
                 for ln in p.read_text().splitlines() if ln.strip()]
        corpus[p.name] = [s for s in sents if s]
    return corpus


def load_tasks(tasks_dir: Path = TASKS_DIR) -> List[Dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(tasks_dir.glob("task_*.json"))]


def load_task(task_id: str, tasks_dir: Path = TASKS_DIR) -> Dict[str, Any]:
    return json.loads((tasks_dir / f"{task_id}.json").read_text())


# --------------------------------------------------------------------------- #
# Deterministic rubric scorer  (task value in [0,1])
# --------------------------------------------------------------------------- #
def score_answer(answer: str, rubric: Sequence[Sequence[str]]) -> float:
    """Fraction of rubric keyword-groups covered by the answer. Deterministic.

    A group is satisfied if the (lowercased) answer contains ANY of its
    synonyms as a substring. Score = satisfied_groups / total_groups."""
    if not rubric:
        return 0.0
    low = answer.lower()
    hit = sum(1 for group in rubric if any(kw.lower() in low for kw in group))
    return hit / len(rubric)


def _uncovered_terms(answer: str, rubric: Sequence[Sequence[str]]) -> List[str]:
    low = answer.lower()
    terms: List[str] = []
    for group in rubric:
        if not any(kw.lower() in low for kw in group):
            terms.extend(group)
    return terms


# --------------------------------------------------------------------------- #
# Retrieval (the FREE fallback engine; also feeds context to the PAID tool)
# --------------------------------------------------------------------------- #
def retrieve(query_terms: Sequence[str], docs: Sequence[str],
             corpus: Dict[str, List[str]], used: set, top_k: int = 3
             ) -> List[str]:
    """Rank candidate sentences from the task's source docs by keyword overlap
    with the query terms; skip already-used sentences."""
    qset = set(t.lower() for t in query_terms)
    scored = []
    for d in docs:
        for s in corpus.get(d, []):
            if s in used:
                continue
            toks = set(_tokens(s))
            overlap = len(qset & toks)
            if overlap:
                scored.append((overlap, len(s), s))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [s for _, _, s in scored[:top_k]]


# --------------------------------------------------------------------------- #
# Planners (propose a DESIRED action; the guard may veto PAID).
# The planner cost is NOT the metered axis. An optional claude-CLI hook exists
# but is off by default; the scripted heuristic is the default planner.
# --------------------------------------------------------------------------- #
Planner = Callable[[Dict[str, Any], np.random.Generator], str]  # features -> "buy"/"skip"


def heuristic_planner(paid_when_progress_below: float = 1.0) -> Planner:
    """Prefer the PAID tool while the task is unfinished and the last free step
    stalled; otherwise take the free fallback. A cheap scripted policy."""
    def plan(feat: Dict[str, Any], rng: np.random.Generator) -> str:
        if feat["progress"] >= paid_when_progress_below:
            return "skip"
        # if the previous free step made no progress, escalate to paid
        if feat.get("last_free_gain", 1.0) <= 0.0:
            return "buy"
        return "buy" if feat["step"] % 2 == 0 else "skip"
    return plan


def greedy_paid_planner() -> Planner:
    def plan(feat, rng):
        return "buy" if feat["progress"] < 1.0 else "skip"
    return plan


def randomized_planner(p_buy: float = 0.55) -> Planner:
    def plan(feat, rng):
        if feat["progress"] >= 1.0:
            return "skip"
        return "buy" if rng.random() < p_buy else "skip"
    return plan


def mixture_behavior_planner(p_greedy: float = 0.5, p_buy: float = 0.55) -> Planner:
    """Logging behavior policy: per-episode mix of greedy-paid and randomized
    fallback (chosen once per episode via the step-0 rng draw), giving both
    actions coverage at every spend/quota bucket -- what the offline learner
    needs."""
    greedy = greedy_paid_planner()
    rand = randomized_planner(p_buy)
    def plan(feat, rng):
        # stable per-episode choice keyed on episode id
        h = (hash((feat.get("episode", 0), "mix")) % 1000) / 1000.0
        return greedy(feat, rng) if h < p_greedy else rand(feat, rng)
    return plan


# --------------------------------------------------------------------------- #
# Guard interface (concrete guards live in guard.py; this is the contract)
# --------------------------------------------------------------------------- #
class GuardLike:
    """features -> allow_paid: bool"""
    def allow_paid(self, features: Dict[str, Any]) -> bool:  # pragma: no cover
        raise NotImplementedError


class Passthrough(GuardLike):
    """No gating -- every PAID request the planner makes is allowed. Used for
    logging runs and the dry run."""
    def allow_paid(self, features: Dict[str, Any]) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Episode runner
# --------------------------------------------------------------------------- #
@dataclass
class StepRecord:
    step: int
    state: Dict[str, Any]        # observation BEFORE the action (has spend, calls)
    desired: str                 # planner's proposal
    action: str                  # after guard veto
    reward: float                # delta task-value
    cost: float                  # dollars this step (0 for free)
    progress: float              # task value AFTER the action
    gated: bool                  # True if guard vetoed a PAID request
    tool_text: str = ""          # snippet appended (paid or free)


@dataclass
class EpisodeResult:
    task_id: str
    steps: List[StepRecord]
    final_progress: float
    total_cost: float
    n_paid: int
    # saorl OfflineDataset-shaped views:
    trajectory: List[Dict[str, Any]]   # state dict per step (spend, calls, ...)
    actions: List[str]                 # "buy"/"skip"
    rewards: List[float]
    n_paid_attempts: int = 0        # PAID requests that reached the API
    last_error: Optional[str] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "final_progress": self.final_progress,
            "total_cost": self.total_cost,
            "n_paid": self.n_paid,
            "trajectory": self.trajectory,
            "actions": self.actions,
            "rewards": self.rewards,
            "steps": [
                {"step": s.step, "state": s.state, "desired": s.desired,
                 "action": s.action, "reward": s.reward, "cost": s.cost,
                 "progress": s.progress, "gated": s.gated,
                 "tool_text": s.tool_text}
                for s in self.steps
            ],
        }


PAID_SYSTEM = (
    "You are a terse research assistant. Given a question and context snippets, "
    "reply with ONE or TWO short factual sentences that directly answer the "
    "question using only the context. No preamble."
)


def run_episode(task: Dict[str, Any], guard: GuardLike, *,
                model: str, ledger: BillingLedger, planner: Planner,
                corpus: Dict[str, List[str]], episode_id: int = 0,
                rng: Optional[np.random.Generator] = None,
                max_steps: int = MAX_STEPS,
                paid_max_tokens: int = 200) -> EpisodeResult:
    """Run one bounded episode. PAID steps bill real dollars via `ledger`."""
    rng = rng or np.random.default_rng(episode_id + 1)
    rubric = task["rubric"]
    docs = task["source_docs"]
    answer = ""
    used: set = set()
    spend = 0.0        # running dollars (the metered axis)
    calls = 0          # paid-call count (the quota axis)
    progress = 0.0
    last_free_gain = 1.0

    steps: List[StepRecord] = []
    trajectory: List[Dict[str, Any]] = []
    actions: List[str] = []
    rewards: List[float] = []
    n_paid_attempts = 0
    last_error: Optional[str] = None

    for t in range(max_steps):
        # Observation BEFORE the action (Markov features for the guard + FQI).
        state = {"spend": round(spend, 8), "calls": calls, "step": t,
                 "progress": round(progress, 6)}
        feat = dict(state, episode=episode_id, last_free_gain=last_free_gain)

        desired = planner(feat, rng)
        gated = False
        if desired == "buy" and not guard.allow_paid(feat):
            action = "skip"          # guard vetoes the paid tool -> free fallback
            gated = True
        else:
            action = desired

        step_cost = 0.0
        tool_text = ""
        # PAID retrieval is rubric-steered (targets uncovered groups) and feeds
        # the LLM richer context; FREE retrieval is question-only (a "dumb"
        # local index) and returns a truncated snippet.
        paid_terms = _tokens(task["question"]) + _uncovered_terms(answer, rubric)
        free_terms = _tokens(task["question"])

        if action == "buy":
            ctx = retrieve(paid_terms, docs, corpus, used, top_k=3)
            for c in ctx:
                used.add(c)
            prompt = (f"Question: {task['question']}\n\nContext:\n- "
                      + "\n- ".join(ctx) + "\n\nAnswer:")
            n_paid_attempts += 1
            try:
                resp = call_openai(prompt, model, system=PAID_SYSTEM,
                                   max_output_tokens=paid_max_tokens)
                rec = ledger.record(resp["usage"], model, episode=episode_id,
                                    step=t, request_id=resp["request_id"])
                step_cost = rec.total_cost
                tool_text = resp["text"]
                calls += 1
                spend += step_cost
            except PaidToolError as e:
                # Real API failure (e.g. insufficient_quota): degrade to the free
                # fallback for this step and record the error so the caller can
                # surface it instead of silently reporting $0 as success.
                last_error = str(e)[:200]
                action = "skip"
                gated = False
                tool_text = "[PAID_FAILED] "
        if action == "skip":
            snips = retrieve(free_terms, docs, corpus, used, top_k=1)
            if snips:
                used.add(snips[0])
                tool_text += snips[0][:FREE_SNIPPET_CHARS]  # extractive snippet

        answer = (answer + " " + tool_text).strip()
        new_progress = score_answer(answer, rubric)
        reward = new_progress - progress
        if action == "skip":
            last_free_gain = reward
        progress = new_progress

        steps.append(StepRecord(step=t, state=state, desired=desired,
                                action=action, reward=reward, cost=step_cost,
                                progress=progress, gated=gated,
                                tool_text=tool_text))
        trajectory.append(state)
        actions.append("buy" if action == "buy" else "skip")
        rewards.append(reward)

        if progress >= 1.0:
            break

    return EpisodeResult(
        task_id=task["id"], steps=steps, final_progress=progress,
        total_cost=sum(s.cost for s in steps),
        n_paid=sum(1 for s in steps if s.action == "buy"),
        n_paid_attempts=n_paid_attempts, last_error=last_error,
        trajectory=trajectory, actions=actions, rewards=rewards,
    )


if __name__ == "__main__":
    # tiny FREE-only self-test (no billing): verify scorer + retrieval wire up
    corpus = load_corpus()
    tasks = load_tasks()
    print(f"{len(tasks)} tasks, {len(corpus)} docs")

    class _FreeOnly(GuardLike):
        def allow_paid(self, f):  # force the free fallback everywhere
            return False

    ledger = BillingLedger(log_path=HERE / "_selftest_billing.jsonl")
    r = run_episode(tasks[0], _FreeOnly(), model="gpt-5-nano", ledger=ledger,
                    planner=greedy_paid_planner(), corpus=corpus)
    print("free-only episode:", tasks[0]["id"],
          "progress", round(r.final_progress, 2), "cost", r.total_cost,
          "paid", r.n_paid)
    (HERE / "_selftest_billing.jsonl").unlink(missing_ok=True)
