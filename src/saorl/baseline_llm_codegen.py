"""Direct language->constraint baseline 1C: LLM writes the cost CODE (reviewer #1).

The most literal "just ask the model" baseline. Instead of SA-ORL's audited set of
readings, we hand the natural-language rule to the LM and ask it to WRITE A PYTHON
PREDICATE `violated(s)` that returns True exactly when the rule forbids continuing
to operate. We sandbox-exec that code as a SINGLE semantic candidate, learn an
offline policy that honors only it (same FQI+budget machinery SA-ORL uses), and
then judge the resulting policy's REALIZED safety against the audited ambiguity set
U_alpha -- the same deployment metric every other table reports.

The point is the #1-family point: a single LM-authored constraint -- however
fluent -- commits to ONE reading of an ambiguous rule. Deployed against the set of
plausible readings a careful auditor would retain, it can still violate a retained
reading, exactly like the single-translation learner. SA-ORL, which honors the
whole audited set, does not. Code generation is also *stochastic*: re-sampling the
generator yields different thresholds/features, so we draw several samples and
report the spread -- a single draw is a coin flip the practitioner cannot see.

SAFETY / BUDGET GATING. Generating code costs API budget, so this script defaults
to --dry-run, which (a) prints the EXACT generation prompt, (b) prints the precise
number of `claude` calls --live would make, and (c) runs the ENTIRE downstream
pipeline on a HAND-WRITTEN stand-in cost -- clearly labelled, NOT an LM output --
so the offline-RL integration is exercised and nothing is fabricated, while zero
budget is spent. Pass --live to actually query the `claude` CLI.

Sandbox. Generated code is parsed to an AST and rejected unless every node is on a
numeric-predicate whitelist (no imports, no attribute access, no calls except
abs/min/max/float/int/bool/round, no loops). Subscripts may only read the
policy-visible features rul_hat/q05/anom; the hidden label rul_true is forbidden.
Compilation happens with empty __builtins__. A generation that yields non-runnable
or label-peeking code is recorded as an `invalid` outcome -- itself an honest
finding about the baseline -- rather than crashing the run.

Dry run (default; spends nothing):
    PYTHONPATH=. python3 -m saorl.baseline_llm_codegen
Live (spends API budget; needs the `claude` CLI):
    PYTHONPATH=. python3 -m saorl.baseline_llm_codegen --live --samples 5 --seeds 30
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .baselines import EPS, MAINT, N_SEEDS, _maint_rollout_once
from .build_plaus_cache import RULE_SETS
from .construct import construct_U_alpha
from .dsl import Candidate, RegionAtom
from .judge import (FEATURE_GLOSSARY, LLMEnsembleJudge, attach_plausibility,
                    load_llm_cache)
from .offline import (evaluate_return, greedy_continue_policy,
                      make_offline_rl_dataset, rollout_risk, worst_case_cost)
from .offline_rl import learn_fqi_constrained

RULE_SET = "conservative"
FORBIDDEN = frozenset({"continue"})       # the action the rule constrains
ALLOWED_FEATURES = ("rul_hat", "q05", "anom")  # rul_true (label) is NOT readable
CALL_WHITELIST = {"abs", "min", "max", "float", "int", "bool", "round"}

# Hand-written stand-in used ONLY by --dry-run to exercise the pipeline end-to-end.
# It is NOT an LM output; it is labelled as such everywhere it appears. It denotes
# one plausible reading of the conservative rule ("stop when degradation is severe").
STANDIN_CODE = (
    "def violated(s):\n"
    "    # stand-in (hand-written, NOT an LM output): severe degradation => stop\n"
    "    return s['anom'] >= 0.6 or s['rul_hat'] <= 20\n"
)


# --- generation prompt --------------------------------------------------------
def codegen_prompt(rule_text: str) -> str:
    """The exact prompt handed to the LM to author the cost predicate."""
    return (
        "You are a controls engineer translating a maintenance rule into code.\n\n"
        f"{FEATURE_GLOSSARY}\n\n"
        f"Natural-language maintenance rule:\n  \"{rule_text}\"\n\n"
        "Write a single Python function\n\n"
        "    def violated(s):\n"
        "        ...\n"
        "        return <bool>\n\n"
        "that returns True exactly when the rule says the asset must NOT keep "
        "operating this cycle, and False otherwise. `s` is one cycle's feature "
        "dict; you may read ONLY s['rul_hat'], s['q05'], and s['anom'] (all "
        "floats). Do not read any other key. Use only arithmetic and comparisons "
        "(and abs/min/max/float/int/bool/round); no imports, no loops, no attribute "
        "access. Reply with ONLY a fenced ```python code block containing the one "
        "function, nothing else."
    )


# --- sandbox ------------------------------------------------------------------
_ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Expr,
    ast.Pass, ast.If, ast.IfExp, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.Name, ast.Load, ast.Store, ast.Assign, ast.AugAssign,
    ast.Constant, ast.Subscript, ast.Index, ast.Tuple, ast.List, ast.Call,
    ast.keyword,
)


def extract_code(text: str) -> str:
    """Pull the python source from a fenced block (or take the whole reply)."""
    if "```" in text:
        chunk = text.split("```", 2)[1]
        if chunk.lstrip().lower().startswith("python"):
            chunk = chunk.split("\n", 1)[1] if "\n" in chunk else ""
        return chunk.strip()
    return text.strip()


def safe_compile(code: str) -> Callable[[dict], bool]:
    """Validate `code` against the numeric-predicate whitelist and return the
    `violated` callable. Raises ValueError on any disallowed construct, including
    reads of features outside ALLOWED_FEATURES (e.g. the hidden label rul_true)."""
    tree = ast.parse(code, mode="exec")
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(tree.body) != len(funcs) or not funcs:
        raise ValueError("code must be only function definition(s)")
    if not any(f.name == "violated" for f in funcs):
        raise ValueError("no function named 'violated'")
    arg_names = {a.arg for f in funcs for a in f.args.args}
    # Locally-assigned names are also bound: the LM idiomatically writes
    # `rul_hat = float(s['rul_hat'])` then uses the local. Such a local can only
    # hold a value derived from s[...] / constants / whitelisted calls (we exec
    # with empty __builtins__, no imports, no attributes), so admitting locals
    # opens no sandbox hole -- it just stops rejecting the natural code style.
    assigned: set = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for tgt in n.targets:
                assigned |= {x.id for x in ast.walk(tgt) if isinstance(x, ast.Name)}
        elif isinstance(n, ast.AugAssign):
            assigned |= {x.id for x in ast.walk(n.target) if isinstance(x, ast.Name)}
    bound = arg_names | CALL_WHITELIST | assigned
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in CALL_WHITELIST):
                raise ValueError("only abs/min/max/float/int/bool/round calls allowed")
        if isinstance(node, ast.Attribute):  # pragma: no cover - not in whitelist
            raise ValueError("attribute access is forbidden")
        if isinstance(node, ast.Subscript):
            key = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            if not (isinstance(key, ast.Constant) and key.value in ALLOWED_FEATURES):
                raise ValueError(f"subscript must be one of {ALLOWED_FEATURES}")
        if isinstance(node, ast.Name) and node.id not in bound:
            raise ValueError(f"unknown name: {node.id}")
    ns: Dict[str, object] = {"__builtins__": {n: __builtins__[n] if isinstance(
        __builtins__, dict) else getattr(__builtins__, n) for n in CALL_WHITELIST}}
    exec(compile(tree, "<llm_codegen>", "exec"), ns)
    fn = ns["violated"]

    def guarded(s: dict) -> bool:
        return bool(fn(s))
    # smoke test on a few representative states (must run without raising)
    for st in ({"rul_hat": 95.0, "q05": 90.0, "anom": 0.1},
               {"rul_hat": 18.0, "q05": 12.0, "anom": 0.82},
               {"rul_hat": 40.0, "q05": 34.0, "anom": 0.55}):
        guarded(st)
    return guarded


def as_candidate(fn: Callable[[dict], bool], name: str) -> Candidate:
    """Wrap a sandboxed cost predicate as a single forbidding candidate."""
    return Candidate(name, RegionAtom("llm_code", fn), FORBIDDEN)


# --- per-sample evaluation ----------------------------------------------------
@dataclass
class CodeRow:
    sample: int
    seed: int
    u_size: int
    policy: str       # 'llm-code' | 'SA-ORL' | 'run-to-failure'
    ret: float
    worst: float      # TRUE worst-case cost over audited U_alpha (active-norm)
    chance: float     # realized Pr(any audited reading violated)
    cvar: float


def evaluate_code(fn: Callable[[dict], bool], sample: int, seeds: Sequence[int],
                  rule_set: str = RULE_SET) -> List[CodeRow]:
    """Learn a policy honoring ONLY the generated code, plus SA-ORL honoring the
    audited U_alpha, and score both on realized risk vs U_alpha across `seeds`."""
    raw = [c for c, _ in RULE_SETS[rule_set]["items"]]
    judge = load_llm_cache(rule_set)
    rollout_once = _maint_rollout_once(MAINT)
    code_c = as_candidate(fn, f"llm-code-{sample}")
    rows: List[CodeRow] = []
    for seed in seeds:
        data = make_offline_rl_dataset(MAINT, seed=seed)
        U = construct_U_alpha(attach_plausibility(raw, judge),
                              data.to_semantic_dataset()).U_alpha
        if not U:
            continue
        rf = lambda pol: evaluate_return(MAINT, pol, n_episodes=40)
        risk = lambda pol: rollout_risk(rollout_once, pol, U)        # audited U
        # baseline: honor ONLY the LM-authored code, budget against itself
        code_res = learn_fqi_constrained(data, MAINT, honor=[code_c], U_eval=U,
                                         eps=EPS, return_fn=rf, bcq_tau=0.05)
        # ours: honor the full audited set
        saorl = learn_fqi_constrained(data, MAINT, honor=U, U_eval=U, eps=EPS,
                                      return_fn=rf, bcq_tau=0.05)
        for name, pol in (("run-to-failure", greedy_continue_policy()),
                          ("llm-code", code_res.policy),
                          ("SA-ORL", saorl.policy)):
            r = risk(pol)
            rows.append(CodeRow(sample, seed, len(U), name, rf(pol),
                                worst_case_cost(pol, data, U, normalize="active"),
                                r["chance"], r["cvar"]))
    return rows


def _summ(rows: List[CodeRow], tag: str,
          order: Sequence[str] = ("run-to-failure", "llm-code", "SA-ORL")) -> None:
    if not rows:
        print(f"  {tag}: no rows")
        return
    n = len({r.seed for r in rows})
    print(f"  === {tag}  (n={n} seeds, eps={EPS}) ===")
    print(f"  {'policy':16s} {'return':>13s} {'TRUE worst*':>13s} "
          f"{'Pr(viol)':>10s} {'CVaR_0.1':>10s} {'safe?':>6s}")
    for name in order:
        rs = [r for r in rows if r.policy == name]
        if not rs:
            continue
        ret = np.array([r.ret for r in rs]); wc = np.array([r.worst for r in rs])
        ch = np.array([r.chance for r in rs]); cv = np.array([r.cvar for r in rs])
        safe = "yes" if ch.mean() <= EPS + 1e-9 else "NO"
        print(f"  {name:16s} {ret.mean():6.1f}+/-{ret.std():4.1f} "
              f"{wc.mean():6.3f}+/-{wc.std():5.3f} {ch.mean():10.3f} "
              f"{cv.mean():10.3f} {safe:>6s}")


# --- driver -------------------------------------------------------------------
def run(samples: int, seeds: Sequence[int], live: bool) -> dict:
    rule_text = RULE_SETS[RULE_SET]["rule_text"]
    prompt = codegen_prompt(rule_text)
    print(f"\n=== 1C LLM cost-code generation: rule_set='{RULE_SET}', "
          f"{samples} sample(s) x 1 generation call each = {samples} calls ===")
    print(f"  rule: \"{rule_text}\"")
    if not live:
        print("\n  [dry-run] generation prompt that --live would send:\n")
        print("  " + prompt.replace("\n", "\n  "))
        print("\n  [dry-run] exercising the FULL downstream pipeline on a "
              "HAND-WRITTEN stand-in cost")
        print("  (clearly NOT an LM output -- proves the offline-RL integration, "
              "spends nothing):")
        print("  " + STANDIN_CODE.replace("\n", "\n  "))
        fn = safe_compile(STANDIN_CODE)
        rows = evaluate_code(fn, sample=0, seeds=list(seeds)[:3])  # 3 seeds: quick
        _summ(rows, "STAND-IN (dry-run, 3 seeds)")
        return dict(dry_run=True, n_calls=samples, rule_text=rule_text,
                    standin=[asdict(r) for r in rows])
    # ---- live: actually generate ----
    gen = LLMEnsembleJudge(rule_text=rule_text)   # reuse the claude CLI plumbing
    samples_out: List[dict] = []
    all_rows: List[CodeRow] = []
    for k in range(samples):
        reply = gen._ask_raw(prompt) or ""
        code = extract_code(reply)
        rec = dict(sample=k, raw=reply, code=code)
        try:
            fn = safe_compile(code)
        except Exception as e:                    # invalid generation = honest outcome
            rec.update(outcome="invalid", error=str(e))
            print(f"  sample {k}: INVALID ({e})")
            samples_out.append(rec)
            continue
        rows = evaluate_code(fn, sample=k, seeds=seeds)
        all_rows.extend(rows)
        rec.update(outcome="ok", rows=[asdict(r) for r in rows])
        samples_out.append(rec)
        _summ(rows, f"sample {k}")
    print("\n  ==== pooled over valid samples ====")
    _summ(all_rows, "ALL valid samples")
    n_valid = sum(1 for s in samples_out if s.get("outcome") == "ok")
    return dict(dry_run=False, n_calls=samples, rule_text=rule_text,
                n_valid=n_valid, samples=samples_out,
                pooled=[asdict(r) for r in all_rows])


def main():
    ap = argparse.ArgumentParser(description="LLM cost-code generation baseline (#1C)")
    ap.add_argument("--live", action="store_true",
                    help="actually call the claude CLI to generate code (spends budget)")
    ap.add_argument("--samples", type=int, default=5,
                    help="independent code generations to draw (live only)")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--out", type=str, default="results/paper_extra/llm_codegen")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    mode = "LIVE (spending API budget)" if args.live else "DRY-RUN (no calls, no spend)"
    print(f"1C LLM cost-code generation baseline -- {mode}")
    res = run(args.samples, seeds, args.live)
    print(f"\n  TOTAL claude calls if --live: {res['n_calls']}")
    if not args.live:
        print("  (dry run -- nothing was sent. Re-run with --live to execute.)")
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    p = out_dir / f"llm_codegen_{stamp}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
