"""REGISTRATION_V7 scoring run: every corpus row under ONE scorer version.

Scores 61 (rule_text, pool) units -- the 9 original rule sets, the 3 frozen
paraphrase texts, the dsrl_cost fresh rule, and the 48 new blind-generated
paraphrase units from paraphrases_v7.json -- with the frozen 7-persona
ensemble and the frozen per-domain prompt configs, under the current local
`claude` CLI. The old caches are never touched; output goes to
plausibility_cache_v7.json with a per-call checkpoint so the run resumes.

Run:  PYTHONPATH=. python3 -m saorl.build_plaus_cache_v7
"""
from __future__ import annotations

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .build_plaus_cache import RULE_SETS
from .build_plaus_cache_temporal import (
    ACTION_GLOSS as ACTION_T,
    DOMAIN_NOUN as NOUN_T,
    TEMPORAL_GLOSSARY,
)
from .build_plaus_cache_budget import (
    ACTION_GLOSS as ACTION_B,
    DOMAIN_NOUN as NOUN_B,
    BUDGET_GLOSSARY,
)
from .build_plaus_cache_dsrl import GLOSSARY as GLOSSARY_D, ITEMS as ITEMS_D, RULE_TEXT as RULE_D
from .gridworld import RULE_SETS_T
from .budget import RULE_SETS_B
from .judge import ENSEMBLE_PERSONAS, LLMEnsembleJudge

_HERE = Path(__file__).parent
OUT = _HERE / "plausibility_cache_v7.json"
CKPT = _HERE / "plausibility_cache_v7.ckpt.jsonl"
PARA = json.load(open(_HERE / "paraphrases_v7.json"))

# family -> (set_names it instantiates over)
FAMILY_SETS = {
    "A": ["conservative", "ambiguous"],
    "B": ["sharp"],
    "C": ["window_sharp"],
    "D": ["window_ambiguous", "window_conservative"],
    "E": ["budget_ambiguous", "budget_conservative"],
}

_SPECS = {**RULE_SETS, **RULE_SETS_T, **RULE_SETS_B}


def _judge_for(set_name: str, rule_text: str) -> LLMEnsembleJudge:
    spec = _SPECS[set_name]
    kw = dict(rule_text=rule_text,
              glosses={c.name: g for c, g in spec["items"]})
    if set_name.startswith("window_"):
        kw.update(glossary=TEMPORAL_GLOSSARY, domain_noun=NOUN_T, action_gloss=ACTION_T)
    elif set_name.startswith("budget_"):
        kw.update(glossary=BUDGET_GLOSSARY, domain_noun=NOUN_B, action_gloss=ACTION_B)
    return LLMEnsembleJudge(**kw)  # maint sets: frozen defaults


def _old_paraphrase_texts() -> list:
    d = json.load(open(sorted((_HERE.parent.parent / "results/paper_extra" / "judge_robustness").glob("*.json"))[0]))
    return [r["paraphrase"] for r in d["paraphrase"]["rows"]]


def units() -> dict:
    """unit_id -> (set_name, rule_text). 61 units total."""
    out = {}
    for sname, spec in _SPECS.items():                       # 9 originals
        out[f"orig:{sname}"] = (sname, spec["rule_text"])
    for i, t in enumerate(_old_paraphrase_texts(), 1):       # 3 frozen paraphrases
        out[f"oldpara-{i}:conservative"] = ("conservative", t)
    for fam, sets in FAMILY_SETS.items():                    # 48 new
        for j, t in enumerate(PARA[fam], 1):
            for sname in sets:
                out[f"{fam}{j}:{sname}"] = (sname, t)
    out["orig:dsrl_cost"] = ("dsrl_cost", RULE_D)            # 1 fresh rule
    return out


def _cli_version() -> str:
    try:
        return subprocess.run(["claude", "--version"], capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return "claude-cli"


def main(max_workers: int = 8) -> None:
    U = units()
    n_dsrl = len(ITEMS_D)
    n_calls = sum((n_dsrl if s == "dsrl_cost" else len(_SPECS[s]["items"])) * len(ENSEMBLE_PERSONAS)
                  for s, _ in U.values())
    done = {}
    if CKPT.exists():                                        # resume
        for line in open(CKPT):
            r = json.loads(line)
            done[(r["unit"], r["cand"], r["persona"])] = r["score"]
    print(f"{len(U)} units, {n_calls} calls total, {len(done)} already done")

    judges = {}
    for uid, (sname, text) in U.items():
        if sname == "dsrl_cost":
            judges[uid] = LLMEnsembleJudge(rule_text=text,
                                           glosses={c.name: g for c, g in ITEMS_D},
                                           glossary=GLOSSARY_D, domain_noun="robot-safety",
                                           action_gloss="'keep operating in the constraint region'")
        else:
            judges[uid] = _judge_for(sname, text)

    tasks = []
    for uid, (sname, _) in U.items():
        items = ITEMS_D if sname == "dsrl_cost" else _SPECS[sname]["items"]
        for cand, _g in items:
            for pi, persona in enumerate(ENSEMBLE_PERSONAS):
                if (uid, cand.name, pi) not in done:
                    tasks.append((uid, cand, pi, persona))

    lock = threading.Lock()
    ck = open(CKPT, "a")

    def one(t):
        uid, cand, pi, persona = t
        j = judges[uid]
        v = j._ask(j._prompt(persona, cand))
        if v is None:                                        # registered: one retry
            v = j._ask(j._prompt(persona, cand))
        with lock:
            ck.write(json.dumps(dict(unit=uid, cand=cand.name, persona=pi,
                                     score=v)) + "\n")
            ck.flush()
        return t, v

    n_done = len(done)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed([ex.submit(one, t) for t in tasks]):
            (uid, cand, pi, _), v = fut.result()
            done[(uid, cand.name, pi)] = v
            n_done += 1
            if n_done % 100 == 0:
                print(f"  {n_done}/{n_calls}")
    ck.close()

    sets = {}
    for uid, (sname, text) in U.items():
        items = ITEMS_D if sname == "dsrl_cost" else _SPECS[sname]["items"]
        scores = {}
        for cand, _g in items:
            vals = [done[(uid, cand.name, pi)] for pi in range(len(ENSEMBLE_PERSONAS))
                    if done.get((uid, cand.name, pi)) is not None]
            scores[cand.name] = vals
        sets[uid] = dict(set_name=sname, rule_text=text, scores=scores)
    json.dump(dict(model=_cli_version(), registration="REGISTRATION_V7",
                   personas=list(ENSEMBLE_PERSONAS), units=sets),
              open(OUT, "w"), indent=1)
    missing = [(u, c) for u, s in sets.items() for c, v in s["scores"].items() if not v]
    print(f"wrote {OUT.name}; units={len(sets)}; empty-score candidates: {missing or 'none'}")


if __name__ == "__main__":
    main()
