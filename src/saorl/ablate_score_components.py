"""W17 (blueprint section 19): factorial ablation of the CORSET composite score.

The deployed CORSET score is s = (LM-ensemble-mean plausibility) hard-zeroed by
two data-calibration gates (incident activation >= RHO_INC, overblocking <=
RHO_OVER).  A reviewer cannot tell from the headline table which component --
language semantics, the incident gate, the overblocking gate, or their
interaction -- actually supplies the efficiency (small sets) while conformal
calibration supplies validity regardless.  This module reruns the EXACT
leave-one-rule-out conformal protocol of saorl.conformal (delta_sem = 0.1,
ties at q-hat retained, same 11-row authored corpus, same frozen caches and
frozen data gates at CORPUS_SEED) under eight score conditions:

  language   LM-ensemble mean only, no gates
  gates      1.0 iff both data gates pass, else 0.0 (no language signal)
  combined   the deployed CORSET score (must reproduce saorl.conformal exactly)
  inc_lang   ensemble mean zeroed only by the incident gate
  over_lang  ensemble mean zeroed only by the overblocking gate
  constant   1.0 for every candidate (validity floor: maximal sets)
  random     seeded uniform [0,1] per (rule, candidate) (sha256-keyed, frozen)
  lexical    Jaccard token overlap between the rule text and the candidate's
             NL gloss (the glosses shipped to the LLM judge; no model at all)

Everything is cache-only and deterministic: no LLM call is made.  Theorem 1
predicts every condition keeps LOO coverage >= 1 - delta - 1/(n+1); only set
SIZE (efficiency) and agreement with the deployed binding sets should differ.

Run:  SAORL_CONFORMAL=1 PYTHONPATH=. python3 -m saorl.ablate_score_components
Writes results/conformal/score_ablation.json
   and paper/generated/gen_score_ablation.tex
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .budget import RULE_SETS_B
from .build_plaus_cache import RULE_SETS
from .conformal import (
    DELTA_SEM,
    _JUDGE_ROBUST_DIR,
    _ROUND,
    _budget_dsem,
    _gridworld_dsem,
    _maint_dsem,
    _pools,
    conformal_threshold,
    corpus_rows,
    gate_pass,
)
from .gridworld import RULE_SETS_T

_HERE = Path(__file__).parent
_REPO = _HERE.parent.parent
_CONFORMAL_REPORT = _REPO / "results/paper_extra" / "conformal" / "conformal_report.json"

RANDOM_SEED = 0  # frozen seed for the seeded-random condition

# Frozen embedding cache for the "embedding" condition (built once by
# saorl.build_embedding_cache; text-embedding-3-small). No API at ablation time.
_EMB_CACHE_PATH = _REPO / "results/conformal" / "embedding_cache.json"


def _load_embeddings() -> Dict[str, list]:
    if not _EMB_CACHE_PATH.exists():
        return {}
    return json.load(open(_EMB_CACHE_PATH)).get("vectors", {})


_EMB = _load_embeddings()


def _cosine(a: list, b: list) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _embed_score(rule_text: str, gloss: str) -> float:
    va, vb = _EMB.get(rule_text), _EMB.get(gloss)
    if va is None or vb is None:
        return 0.0
    return round(_cosine(va, vb), _ROUND)

CONDITIONS = (
    "language", "gates", "combined", "inc_lang", "over_lang",
    "constant", "random", "lexical", "embedding", "oracle",
)
CONDITION_LABEL = {
    "language": "Language only (no gates)",
    "gates": "Data gates only",
    "combined": "Combined (deployed CORSET)",
    "inc_lang": r"Incident gate $\times$ language",
    "over_lang": r"Overblock gate $\times$ language",
    "constant": "Constant 1.0",
    "random": "Seeded random",
    "lexical": "Lexical overlap",
    "embedding": "Embedding similarity",
    "oracle": "Oracle reference (gold-only)",
}


# --------------------------------------------------------------------------
# Frozen per-set candidate facts: language mean, individual gate flags, gloss
# --------------------------------------------------------------------------

def _rule_sets_lookup() -> Dict[str, dict]:
    """set_name -> {rule_text, glosses{cand_name: gloss}} from the frozen
    RULE_SETS dicts (the exact glosses the LLM judge was shown)."""
    out: Dict[str, dict] = {}
    for src in (RULE_SETS, RULE_SETS_T, RULE_SETS_B):
        for sname, spec in src.items():
            out[sname] = dict(
                rule_text=spec["rule_text"],
                glosses={c.name: g for c, g in spec["items"]},
            )
    return out


def _tokens(text: str) -> frozenset:
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def _lexical(rule_text: str, gloss: str) -> float:
    a, b = _tokens(rule_text), _tokens(gloss)
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), _ROUND)


def _rand_score(rule_id: str, cand_name: str) -> float:
    """Seeded uniform [0,1], keyed by a stable sha256 of (seed, rule, cand):
    deterministic across processes (unlike Python's salted hash)."""
    key = f"{RANDOM_SEED}|{rule_id}|{cand_name}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**31)
    return round(float(np.random.default_rng(seed).uniform()), _ROUND)


def _set_facts() -> Dict[str, dict]:
    """set_name -> per-candidate frozen facts used by every condition."""
    dsems = {"maint": _maint_dsem(), "grid": _gridworld_dsem(),
             "budget": _budget_dsem()}
    from .construct import RHO_INC, RHO_OVER

    meta = _rule_sets_lookup()
    facts: Dict[str, dict] = {}
    for set_name, (pool, dom) in _pools().items():
        dsem = dsems[dom]
        cands = {}
        for c in pool:
            _, pinc, pnorm = gate_pass(c, dsem)
            cands[c.name] = dict(
                lang=round(c.plaus_mean(), _ROUND),
                inc_ok=bool(pinc >= RHO_INC),
                over_ok=bool(pnorm <= RHO_OVER),
                p_inc=round(pinc, 4),
                p_norm=round(pnorm, 4),
                lex=_lexical(meta[set_name]["rule_text"],
                             meta[set_name]["glosses"][c.name]),
            )
        facts[set_name] = dict(cands=cands, rule_text=meta[set_name]["rule_text"],
                               glosses=meta[set_name]["glosses"])
    return facts


def _paraphrase_texts() -> List[str]:
    """The three live-requeried paraphrases of the SEVERE rule, in the same
    order saorl.conformal._load_paraphrase_rows assigns row ids 1..3."""
    files = sorted(_JUDGE_ROBUST_DIR.glob("*.json"))
    if not files:
        return []
    data = json.load(open(files[0]))
    return [r["paraphrase"] for r in data.get("paraphrase", {}).get("rows", [])]


# --------------------------------------------------------------------------
# Condition scores
# --------------------------------------------------------------------------

def _cond_score(cond: str, rule_id: str, cand_name: str, fact: dict,
                lang_override: Optional[float], row_rule_text: Optional[str],
                gloss: str, gold_name: Optional[str] = None,
                set_rule_text: Optional[str] = None) -> float:
    """Score of one candidate for one corpus row under `cond`.

    `lang_override` carries the paraphrase rows' live-recorded ensemble means
    (raw floats, exactly as saorl.conformal uses them); `row_rule_text` is the
    paraphrase text for those rows (lexical condition), else None -> the set's
    canonical rule text was already baked into fact['lex']."""
    lang = float(lang_override) if lang_override is not None else fact["lang"]
    both = fact["inc_ok"] and fact["over_ok"]
    if cond == "language":
        return lang
    if cond == "gates":
        return 1.0 if both else 0.0
    if cond == "combined":
        return lang if both else 0.0
    if cond == "inc_lang":
        return lang if fact["inc_ok"] else 0.0
    if cond == "over_lang":
        return lang if fact["over_ok"] else 0.0
    if cond == "constant":
        return 1.0
    if cond == "random":
        return _rand_score(rule_id, cand_name)
    if cond == "lexical":
        if row_rule_text is not None:
            return _lexical(row_rule_text, gloss)
        return fact["lex"]
    if cond == "embedding":
        rt = row_rule_text if row_rule_text is not None else set_rule_text
        return _embed_score(rt, gloss)
    if cond == "oracle":
        return 1.0 if cand_name == gold_name else 0.0
    raise ValueError(cond)


# --------------------------------------------------------------------------
# The LOO conformal protocol, per condition
# --------------------------------------------------------------------------

def run(delta: float = DELTA_SEM) -> dict:
    facts = _set_facts()
    rows = corpus_rows()
    para_texts = _paraphrase_texts()
    para_text_by_id = {f"maint-paraphrase-{i}": t
                       for i, t in enumerate(para_texts, 1)}
    deployed_ids = [r.rule_id for r in rows if r.means_override is None]

    result: Dict[str, dict] = {}
    for cond in CONDITIONS:
        # sigma (gold score) per corpus row + full score table per row
        sigmas: Dict[str, float] = {}
        tables: Dict[str, Dict[str, float]] = {}
        for r in rows:
            f = facts[r.set_name]
            rt = para_text_by_id.get(r.rule_id)
            tab = {}
            for name, cf in f["cands"].items():
                ov = None
                if r.means_override is not None:
                    ov = r.means_override.get(name, 0.0)
                tab[name] = _cond_score(cond, r.rule_id, name, cf, ov, rt,
                                        f["glosses"][name], gold_name=r.gold,
                                        set_rule_text=f["rule_text"])
            tables[r.rule_id] = tab
            sigmas[r.rule_id] = tab[r.gold]

        # leave-one-rule-out: threshold, coverage, deployed set
        per_rule: Dict[str, dict] = {}
        covered = []
        for r in rows:
            others = [sigmas[o.rule_id] for o in rows if o.rule_id != r.rule_id]
            q = conformal_threshold(others, delta)
            cov = bool(sigmas[r.rule_id] >= q)
            covered.append(cov)
            entry = dict(sigma=round(sigmas[r.rule_id], 4),
                         qhat_loo=round(q, 4), covered=cov, gold=r.gold)
            if r.means_override is None:
                U = sorted(n for n, s in tables[r.rule_id].items() if s >= q)
                entry.update(U=U, size=len(U), empty=(len(U) == 0),
                             gold_in_U=r.gold in U)
            per_rule[r.rule_id] = entry

        sizes = [per_rule[i]["size"] for i in deployed_ids]
        result[cond] = dict(
            loo_coverage=round(float(np.mean(covered)), 4),
            n_covered=int(sum(covered)),
            n_rows=len(rows),
            mean_set_size=round(float(np.mean(sizes)), 3),
            n_abstentions=int(sum(per_rule[i]["empty"] for i in deployed_ids)),
            per_rule=per_rule,
        )

    # binding-set match vs the deployed CORSET sets (= the combined condition)
    corset_sets = {i: result["combined"]["per_rule"][i]["U"] for i in deployed_ids}
    for cond in CONDITIONS:
        matches = {}
        for i in deployed_ids:
            matches[i] = result[cond]["per_rule"][i]["U"] == corset_sets[i]
            result[cond]["per_rule"][i]["matches_corset"] = matches[i]
        result[cond]["n_binding_match"] = int(sum(matches.values()))

    # cross-check: combined must equal saorl.conformal's frozen deployed sets
    crosscheck = dict(checked=False)
    if _CONFORMAL_REPORT.exists():
        rep = json.load(open(_CONFORMAL_REPORT))
        diffs = {}
        for i in deployed_ids:
            want = sorted(rep["deployed"][i]["U_conformal"])
            got = result["combined"]["per_rule"][i]["U"]
            if want != got:
                diffs[i] = dict(conformal_report=want, combined_condition=got)
        crosscheck = dict(checked=True, source=str(_CONFORMAL_REPORT),
                          identical=(not diffs), diffs=diffs)
        assert not diffs, f"combined condition diverged from conformal.py: {diffs}"

    return dict(
        delta_sem=delta,
        n_corpus=len(rows),
        deployed_rules=deployed_ids,
        random_seed=RANDOM_SEED,
        note=("Same 11-row corpus, LOO thresholds and tie convention as "
              "saorl.conformal; only the score function varies. Paraphrase rows "
              "use their live-recorded ensemble means for language terms and "
              "their own paraphrase text for the lexical condition."),
        conditions=result,
        crosscheck_combined_vs_conformal=crosscheck,
    )


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def _write_tex(report: dict, path: Path) -> None:
    lines = ["% AUTO-GENERATED by saorl/ablate_score_components.py -- do not edit",
             "% condition & LOO coverage & mean |U| (8 deployed rules) & "
             "abstentions & binding sets = deployed CORSET"]
    for cond in CONDITIONS:
        c = report["conditions"][cond]
        lines.append(
            f"{CONDITION_LABEL[cond]} & {c['n_covered']}/{c['n_rows']} "
            f"({c['loo_coverage']:.3f}) & {c['mean_set_size']:.2f} & "
            f"{c['n_abstentions']} & {c['n_binding_match']}/8 \\\\")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    report = run()
    out_json = _REPO / "results/conformal" / "score_ablation.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out_json, "w"), indent=1)
    out_tex = _REPO / "paper" / "generated" / "gen_score_ablation.tex"
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    _write_tex(report, out_tex)

    print(f"{'condition':28s} {'coverage':>10s} {'mean|U|':>8s} "
          f"{'abstain':>8s} {'match':>6s}")
    for cond in CONDITIONS:
        c = report["conditions"][cond]
        print(f"{CONDITION_LABEL[cond]:28s} {c['n_covered']:>7d}/{c['n_rows']}"
              f" {c['mean_set_size']:8.2f} {c['n_abstentions']:8d} "
              f"{c['n_binding_match']:>4d}/8")
    cc = report["crosscheck_combined_vs_conformal"]
    print(f"cross-check combined == conformal.py deployed sets: "
          f"{'PASS' if cc.get('identical') else cc}")
    print(f"wrote {out_json}\nwrote {out_tex}")


if __name__ == "__main__":
    main()
