"""LM-judge robustness probes for U_alpha (reviewer #5/#6 remainder).

Unlike saorl.ablate_judges (cache-only, learning-free), these two probes REQUIRE
re-querying the LM, so they spend API budget and are gated: the script defaults to
--dry-run, which constructs and prints the exact prompts and the precise number of
`claude` calls it WOULD make, and spends nothing. Pass --live to actually fire them
through saorl.judge.LLMEnsembleJudge and write the results.

  * paraphrase stability: re-score the conservative maintenance rule set under K
    meaning-preserving paraphrases of the rule text (same 7 personas, same
    predicates, same construction), rebuild U_alpha for each paraphrase, and report
    Jaccard against the canonical cached U_alpha. This tests sensitivity to HOW the
    rule is worded -- the part ablate_judges cannot reach because it perturbs judges,
    not text. A robust audit keeps Jaccard ~ 1 across paraphrases.

  * position / presentation probe: the headline protocol scores each candidate
    INDEPENDENTLY on an absolute 0-1 scale, with no option list, which structurally
    avoids the option-order ("position") bias that afflicts comparative LM judging.
    This probe confirms that empirically: it re-scores the binding and borderline
    readings while EMBEDDING each in a list of all four readings at every list
    position, and checks the elicited score is position-invariant. Stable scores
    across positions => the pointwise protocol's order-independence is not just by
    construction but observed.

Dry run (default; spends nothing):
    PYTHONPATH=. python3 -m saorl.ablate_judge_robustness
Live (spends API budget; needs the `claude` CLI):
    PYTHONPATH=. python3 -m saorl.ablate_judge_robustness --live --paraphrases 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

from .build_plaus_cache import RULE_SETS
from .construct import construct_U_alpha
from .dsl import Candidate
from .experiments import _synthetic_seed
from .judge import ENSEMBLE_PERSONAS, FEATURE_GLOSSARY, LLMEnsembleJudge, _parse_score

# Canonical conservative rule (saorl.build_plaus_cache.SEVERE) and meaning-preserving
# paraphrases. Each must denote the SAME constraint -- "stop before severe
# degradation" -- in different words; if U_alpha is robust, which wording we pick
# should not change the retained set.
PARAPHRASES: List[str] = [
    "Do not keep operating the asset when its level of degradation is severe.",
    "Stop running the unit once it has degraded badly; severe wear means do not "
    "continue operating.",
    "When the asset's deterioration becomes severe, operation must not continue.",
    "Refrain from continued operation of the asset under conditions of severe "
    "degradation.",
]


def _cons_items():
    """(bare candidate, gloss) pairs for the conservative rule set."""
    return list(RULE_SETS["conservative"]["items"])


def _names(U) -> set:
    return {c.name for c in U}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _canonical_U():
    """The headline U_alpha from the frozen cache (cache-only, free)."""
    from .experiments import _cons_cands
    data, _, _, _, _ = _synthetic_seed(0)
    pool = _cons_cands()
    U = construct_U_alpha(pool, data.to_semantic_dataset()).U_alpha
    return _names(U)


def _list_prompt(persona: str, items, target_idx: int, order: Sequence[int]) -> str:
    """Embed all readings in a list at the given `order`, ask for the score of the
    one target reading only (single-number reply). Used for the position probe."""
    letters = "ABCDEFGH"
    lines = []
    target_letter = None
    for slot, ci in enumerate(order):
        cand, gloss = items[ci]
        lines.append(f"  ({letters[slot]}) {gloss}")
        if ci == target_idx:
            target_letter = letters[slot]
    listing = "\n".join(lines)
    return (
        f"You are {persona}.\n\n{FEATURE_GLOSSARY}\n\n"
        f"Natural-language maintenance rule:\n  \"{RULE_SETS['conservative']['rule_text']}\"\n\n"
        f"Here are several proposed formal readings of that rule, each forbidding "
        f"the 'continue operating' action exactly when its condition holds:\n"
        f"{listing}\n\n"
        f"Focus ONLY on reading ({target_letter}). On a scale from 0 to 1, how "
        f"faithfully does reading ({target_letter}) capture what the rule actually "
        f"means? 1.0 = exact/faithful; 0.5 = defensible but imperfect; 0.0 = clearly "
        f"wrong. Reply with ONLY a single number between 0 and 1, nothing else."
    )


# --- paraphrase stability -----------------------------------------------------
def paraphrase_probe(paraphrases: List[str], live: bool) -> dict:
    items = _cons_items()
    glosses = {c.name: g for c, g in items}
    bare = [c for c, _ in items]
    n_calls = len(paraphrases) * len(bare) * len(ENSEMBLE_PERSONAS)
    print(f"\n=== paraphrase stability: {len(paraphrases)} paraphrases x "
          f"{len(bare)} readings x {len(ENSEMBLE_PERSONAS)} personas = {n_calls} calls ===")
    if not live:
        j = LLMEnsembleJudge(rule_text=paraphrases[0], glosses=glosses)
        print("  [dry-run] sample prompt (paraphrase 0, reading 0, persona 0):\n")
        print("  " + j._prompt(ENSEMBLE_PERSONAS[0], bare[0]).replace("\n", "\n  "))
        return dict(n_calls=n_calls, dry_run=True)
    canonical = _canonical_U()
    data, _, _, _, _ = _synthetic_seed(0)
    dsem = data.to_semantic_dataset()
    rows = []
    for pi, para in enumerate(paraphrases):
        judge = LLMEnsembleJudge(rule_text=para, glosses=glosses)
        scored = [Candidate(c.name, c.predicate, c.forbidden_actions, judge.scores(c))
                  for c in bare]
        U = construct_U_alpha(scored, dsem).U_alpha
        jac = _jaccard(_names(U), canonical)
        means = {c.name: (sum(c.plausibility) / len(c.plausibility) if c.plausibility
                          else float("nan")) for c in scored}
        rows.append(dict(paraphrase=para, U=sorted(_names(U)), jaccard=jac, means=means))
        print(f"  paraphrase {pi}: |U|={len(U)} Jaccard={jac:.3f}  {sorted(_names(U))}")
    jaccs = [r["jaccard"] for r in rows]
    return dict(n_calls=n_calls, dry_run=False, canonical=sorted(canonical),
                rows=rows, mean_jaccard=sum(jaccs) / len(jaccs),
                worst_jaccard=min(jaccs))


# --- position / presentation probe -------------------------------------------
def position_probe(live: bool, targets=("consv: anom>0.6", "vconsv: RUL<40")) -> dict:
    items = _cons_items()
    name_to_idx = {c.name: i for i, (c, _) in enumerate(items)}
    tgt_idx = [name_to_idx[t] for t in targets if t in name_to_idx]
    n_pos = len(items)
    n_calls = len(tgt_idx) * n_pos * len(ENSEMBLE_PERSONAS)
    print(f"\n=== position probe: {len(tgt_idx)} targets x {n_pos} positions x "
          f"{len(ENSEMBLE_PERSONAS)} personas = {n_calls} calls ===")
    if not live:
        ti = tgt_idx[0]
        others = [i for i in range(n_pos) if i != ti]
        order = [ti] + others   # target forced to slot (A) to illustrate position 0
        print("  [dry-run] sample list-prompt (target forced to position A, persona 0):\n")
        print("  " + _list_prompt(ENSEMBLE_PERSONAS[0], items, ti, order)
              .replace("\n", "\n  "))
        return dict(n_calls=n_calls, dry_run=True)
    out = []
    for ti in tgt_idx:
        others = [i for i in range(n_pos) if i != ti]
        by_pos = []
        for pos in range(n_pos):
            order = others[:pos] + [ti] + others[pos:]
            vals = []
            for persona in ENSEMBLE_PERSONAS:
                judge = LLMEnsembleJudge(rule_text=RULE_SETS["conservative"]["rule_text"])
                v = _parse_score(judge._ask_raw(_list_prompt(persona, items, ti, order)) or "")
                if v is not None:
                    vals.append(v)
            mean = sum(vals) / len(vals) if vals else float("nan")
            by_pos.append(dict(position=pos, mean=mean, n=len(vals)))
            print(f"  target={items[ti][0].name:16s} pos={pos}: mean={mean:.3f} n={len(vals)}")
        ms = [d["mean"] for d in by_pos]
        out.append(dict(target=items[ti][0].name, by_position=by_pos,
                        spread=max(ms) - min(ms)))
    return dict(n_calls=n_calls, dry_run=False, targets=out,
                max_spread=max((o["spread"] for o in out), default=0.0))


def main():
    ap = argparse.ArgumentParser(description="LM-judge robustness probes (#5/#6 remainder)")
    ap.add_argument("--live", action="store_true",
                    help="actually call the claude CLI (spends API budget)")
    ap.add_argument("--paraphrases", type=int, default=3,
                    help="number of paraphrases to use (<= %d)" % len(PARAPHRASES))
    ap.add_argument("--no-position", action="store_true", help="skip the position probe")
    ap.add_argument("--out", type=str, default="results/paper_extra/judge_robustness")
    args = ap.parse_args()
    paras = PARAPHRASES[: max(1, min(args.paraphrases, len(PARAPHRASES)))]

    mode = "LIVE (spending API budget)" if args.live else "DRY-RUN (no calls, no spend)"
    print(f"LM-judge robustness probes -- {mode}")
    res = dict(paraphrase=paraphrase_probe(paras, args.live))
    if not args.no_position:
        res["position"] = position_probe(args.live)

    total = res["paraphrase"]["n_calls"] + res.get("position", {}).get("n_calls", 0)
    print(f"\n  TOTAL claude calls if --live: {total}")
    if not args.live:
        print("  (dry run -- nothing was sent. Re-run with --live to execute.)")
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    p = out_dir / f"judge_robustness_{stamp}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
