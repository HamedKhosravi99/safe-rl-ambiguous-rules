"""Run the REAL language-plausibility ensemble for Domain 3 (budget) and cache it.

Domain-3 analogue of saorl.build_plaus_cache. For every candidate of every budget
rule set (saorl.budget.RULE_SETS_B) a 7-member expert prompt ensemble
(saorl.judge.LLMEnsembleJudge, local `claude` CLI) scores how faithfully the
formal dollar-cap predicate reads the natural-language spending guardrail. Scores
are serialized to `saorl/plausibility_cache_budget.json`; the headline
construction then reads that cache (CachedEnsembleJudge) with no model access, so
the budget U_alpha is reproducible.

The non-circularity question here is specifically about the DOLLAR CAP: does a
real language ensemble genuinely disagree on which numeric cap "keep spending
modest / don't run up a large bill" means -- retaining a *band* of caps for the
vague rule -- while concentrating on a single cap for the precise rule ("do not
reach $60")? That asymmetry is exactly what the budget kill test relies on, and
nothing in the task data can supply it (spend is observed exactly), so it must
come from the language alone. Run (slow; makes LLM calls):

    python3 -m saorl.build_plaus_cache_budget
"""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from .budget import RULE_SETS_B
from .judge import ENSEMBLE_PERSONAS, LLMEnsembleJudge

CACHE_PATH = Path(__file__).with_name("plausibility_cache_budget.json")

# Budget-domain feature glossary: a single running-bill scalar and the dollar-cap
# operator. The cap that the words "modest" / "a large bill" intend is latent and
# unstated, so "where does modest end" is a genuinely ambiguous numeric choice
# that only the language -- not the task data -- can speak to.
BUDGET_GLOSSARY = (
    "Setting (an autonomous agent working a task that costs money per action, "
    "observed each step):\n"
    "- spend: the session's running bill in dollars -- the cumulative amount the "
    "agent has spent so far on paid actions this session. It starts at 0 and only "
    "grows. A typical paid action costs about $10.\n"
    "- spend >= THETA: true once the running bill has reached THETA dollars; a cap "
    "predicate at THETA forbids taking another paid action from that point on.\n"
    "- buying (taking a paid action) is the costly action; skipping to a free "
    "fallback is the safe action. No dollar amount is stated in the rule itself."
)

DOMAIN_NOUN = "agent spending"
ACTION_GLOSS = "'buy (take a paid action)'"


def _model_id() -> str:
    try:
        out = subprocess.run(
            ["claude", "--version"], capture_output=True, text=True, timeout=30
        )
        return out.stdout.strip() or "claude-cli"
    except Exception:
        return "claude-cli"


def build(max_workers: int = 8) -> dict:
    """Score every (set, candidate, persona) in parallel; return the cache dict."""
    judges = {
        name: LLMEnsembleJudge(
            rule_text=spec["rule_text"],
            glosses={c.name: g for c, g in spec["items"]},
            glossary=BUDGET_GLOSSARY,
            domain_noun=DOMAIN_NOUN,
            action_gloss=ACTION_GLOSS,
        )
        for name, spec in RULE_SETS_B.items()
    }
    tasks = []
    for sname, spec in RULE_SETS_B.items():
        for cand, _ in spec["items"]:
            for pi, persona in enumerate(ENSEMBLE_PERSONAS):
                tasks.append((sname, cand, pi, persona))

    results: Dict[tuple, float] = {}

    def run(task):
        sname, cand, pi, persona = task
        v = judges[sname]._ask(judges[sname]._prompt(persona, cand))
        return (sname, cand.name, pi), v

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for key, v in ex.map(run, tasks):
            results[key] = v
            done += 1
            print(f"  [{done}/{len(tasks)}] {key[0]}/{key[1]} m{key[2]} -> {v}")

    cache = {"model": _model_id(), "personas": list(ENSEMBLE_PERSONAS), "sets": {}}
    for sname, spec in RULE_SETS_B.items():
        entry = {"rule_text": spec["rule_text"], "scores": {}}
        for cand, _ in spec["items"]:
            vals = [results[(sname, cand.name, pi)]
                    for pi in range(len(ENSEMBLE_PERSONAS))]
            entry["scores"][cand.name] = [v for v in vals if v is not None]
        cache["sets"][sname] = entry
    return cache


def main():
    n = sum(len(s["items"]) for s in RULE_SETS_B.values())
    print(f"Scoring {n} budget candidates x {len(ENSEMBLE_PERSONAS)} personas "
          f"with the real `claude` ensemble...")
    cache = build()
    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    print(f"\nWrote {CACHE_PATH}")
    for sname, entry in cache["sets"].items():
        print(f"\n  {sname}: \"{entry['rule_text']}\"")
        for name, vals in entry["scores"].items():
            mean = sum(vals) / len(vals) if vals else float("nan")
            print(f"    {name:24s} mean={mean:.3f}  n={len(vals)}  {vals}")


if __name__ == "__main__":
    main()
