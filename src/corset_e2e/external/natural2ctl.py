"""REGISTRATION_V20: Natural2CTL single-reference target recovery at scale.

Stages (each cached on disk so a rerun is idempotent):
  sample   stratified 200-unit sample of parseable rows            -> natural2ctl_units.json
  generate K = 10 `claude -p` samples per unit (target-blind)     -> natural2ctl_samples.json
  analyze  Kripke-battery classes, frequency Keep, baselines     -> natural2ctl_external.json

Run: PYTHONPATH=. python3 corset_e2e/external/natural2ctl.py --stage sample|generate|analyze|all [--units N] [--k K] [--workers W]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = os.path.join(ROOT, "results/e2e")
CSV = os.environ.get("NATURAL2CTL_CSV",
                     os.path.join(ROOT, "external_data", "Natural2CTL.csv"))
UNITS = os.path.join(R, "natural2ctl_units.json")
SAMPLES = os.path.join(R, "natural2ctl_samples.json")
OUT = os.path.join(R, "natural2ctl_external.json")
SEED = 20260902
DELTA = 0.10
N_STRUCT = 400

# ---------------------------------------------------------------- CTL parsing
TRUE, FALSE = ("true",), ("false",)
_CMP = r"(?:<=|>=|!=|==|=|<|>)"
_TOK = re.compile(r"""
    (?P<pathop>\b(?:AG|AF|AX|EG|EF|EX)\b) |
    (?P<quant>\b[AE](?=\s*[\[(])) |
    (?P<until>\b(?:U|W|R)\b) |
    (?P<iff><->) | (?P<imp>->) | (?P<and>&&|&|\band\b) | (?P<or>\|\||\||\bor\b) |
    (?P<not>!|~|\bnot\b) |
    (?P<lp>\() | (?P<rp>\)) | (?P<lb>\[) | (?P<rb>\]) |
    (?P<const>\b(?:TRUE|FALSE|true|false|True|False)\b) |
    (?P<atom>[A-Za-z_][A-Za-z0-9_.'-]*(?:\s*\([^()]*\))?(?:\s*""" + _CMP + r"""\s*(?:[A-Za-z_][A-Za-z0-9_.'-]*(?:\s*\([^()]*\))?|-?\d+(?:\.\d+)?(?:\s*[a-zA-Z%]+)?|"[^"]*"))?) |
    (?P<num>-?\d+(?:\.\d+)?) |
    (?P<ws>\s+)
""", re.X)


class ParseError(ValueError):
    pass


def tokenize(s):
    out, pos = [], 0
    while pos < len(s):
        m = _TOK.match(s, pos)
        if not m:
            raise ParseError(f"bad char {s[pos:pos+15]!r}")
        pos = m.end()
        k = m.lastgroup
        if k == "ws":
            continue
        out.append((k, re.sub(r"\s+", "", m.group(k))))
    return out


class P:
    def __init__(self, s):
        self.t = tokenize(s); self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else ("eof", "")

    def take(self):
        x = self.peek(); self.i += 1; return x

    def expect(self, k):
        x = self.take()
        if x[0] != k:
            raise ParseError(f"expected {k}, got {x}")
        return x

    def parse(self):
        f = self.iff()
        if self.peek()[0] != "eof":
            raise ParseError(f"trailing {self.t[self.i:self.i+4]}")
        return f

    def iff(self):
        l = self.imp()
        while self.peek()[0] == "iff":
            self.take(); l = ("iff", l, self.imp())
        return l

    def imp(self):
        l = self.or_()
        if self.peek()[0] == "imp":
            self.take(); return ("imp", l, self.imp())
        return l

    def or_(self):
        l = self.and_()
        while self.peek()[0] == "or":
            self.take(); l = ("or", l, self.and_())
        return l

    def and_(self):
        l = self.unary()
        while self.peek()[0] == "and":
            self.take(); l = ("and", l, self.unary())
        return l

    def unary(self):
        k, v = self.peek()
        if k == "not":
            self.take(); return ("not", self.unary())
        if k == "pathop":
            self.take(); return (v, self.unary())
        if k == "quant":
            self.take()
            open_k = self.take()
            if open_k[0] not in ("lb", "lp"):
                raise ParseError("quantifier without bracket")
            close_k = "rb" if open_k[0] == "lb" else "rp"
            a = self.iff()
            op = self.expect("until")[1]
            b = self.iff()
            self.expect(close_k)
            return (v + op, a, b)            # AU, EU, AW, EW, AR, ER
        if k == "lp":
            self.take(); f = self.iff(); self.expect("rp"); return f
        if k == "const":
            self.take(); return TRUE if v.lower() == "true" else FALSE
        if k == "atom":
            self.take(); return ("ap", v)
        if k == "num":
            self.take(); return ("ap", v)
        raise ParseError(f"unexpected {k} {v!r}")


def parse(s):
    return P(s).parse()


def atoms(f, acc=None):
    if acc is None:
        acc = set()
    if f[0] == "ap":
        acc.add(f[1])
    elif f[0] not in ("true", "false"):
        for x in f[1:]:
            atoms(x, acc)
    return acc


def canon(f):
    """canonical string for syntactic identity (commutative and/or sorted)."""
    k = f[0]
    if k in ("ap", "true", "false"):
        return f[1] if k == "ap" else k
    if k in ("and", "or"):
        a, b = sorted([canon(f[1]), canon(f[2])])
        return f"({a} {k} {b})"
    if len(f) == 2:
        return f"{k}({canon(f[1])})"
    return f"{k}({canon(f[1])},{canon(f[2])})"


# ---------------------------------------------------------------- CTL model checking on a battery
def battery(aps, n=N_STRUCT, seed=SEED):
    rng = np.random.default_rng(seed + (int(hashlib.sha1(",".join(sorted(aps)).encode()).hexdigest()[:8], 16) % 10007))
    aps = sorted(aps)
    out = []
    for _ in range(n):
        S = int(rng.integers(2, 6))
        succ = []
        for s in range(S):
            k = int(rng.integers(1, min(S, 3) + 1))
            succ.append(sorted(set(rng.choice(S, size=k, replace=False).tolist())))
        lab = {a: [bool(rng.random() < 0.5) for _ in range(S)] for a in aps}
        out.append((S, succ, lab))
    return out


def check(f, S, succ, lab, memo):
    """set of states (as frozenset) satisfying f."""
    key = id(f)
    if key in memo:
        return memo[key]
    k = f[0]
    allS = frozenset(range(S))
    if k == "true":
        r = allS
    elif k == "false":
        r = frozenset()
    elif k == "ap":
        r = frozenset(s for s in range(S) if lab.get(f[1], [False] * S)[s])
    elif k == "not":
        r = allS - check(f[1], S, succ, lab, memo)
    elif k == "and":
        r = check(f[1], S, succ, lab, memo) & check(f[2], S, succ, lab, memo)
    elif k == "or":
        r = check(f[1], S, succ, lab, memo) | check(f[2], S, succ, lab, memo)
    elif k == "imp":
        r = (allS - check(f[1], S, succ, lab, memo)) | check(f[2], S, succ, lab, memo)
    elif k == "iff":
        a, b = check(f[1], S, succ, lab, memo), check(f[2], S, succ, lab, memo)
        r = frozenset(s for s in range(S) if (s in a) == (s in b))
    elif k == "EX":
        a = check(f[1], S, succ, lab, memo)
        r = frozenset(s for s in range(S) if any(t in a for t in succ[s]))
    elif k == "AX":
        a = check(f[1], S, succ, lab, memo)
        r = frozenset(s for s in range(S) if all(t in a for t in succ[s]))
    elif k in ("EU", "EF", "AU", "AF", "AG", "EG", "AW", "EW", "AR", "ER"):
        if k in ("EF", "AF", "AG", "EG"):
            a, b = (allS, check(f[1], S, succ, lab, memo))
        else:
            a, b = check(f[1], S, succ, lab, memo), check(f[2], S, succ, lab, memo)

        def eu(a, b):
            r = set(b)
            while True:
                new = {s for s in range(S) if s in a and any(t in r for t in succ[s])} - r
                if not new:
                    return frozenset(r)
                r |= new

        def eg(a):
            r = set(a)
            while True:
                keep = {s for s in r if any(t in r for t in succ[s])}
                if keep == r:
                    return frozenset(r)
                r = keep
        if k == "EF":
            r = eu(allS, b)
        elif k == "EU":
            r = eu(a, b)
        elif k == "EG":
            r = eg(b)
        elif k == "AF":
            r = allS - eg(allS - b)
        elif k == "AG":
            r = allS - eu(allS, allS - b)
        elif k == "AU":
            r = (allS - eu(allS - b, (allS - a) & (allS - b))) & (allS - eg(allS - b))
        elif k == "AW":
            r = allS - eu(allS - b, (allS - a) & (allS - b))
        elif k == "EW":
            r = eu(a, b) | eg(a)
        elif k == "AR":             # A[a R b] = !E[!a U !b]
            r = allS - eu(allS - a, allS - b)
        else:                       # E[a R b] = E[b W (a & b)]
            r = eu(b, a & b) | eg(b)
    else:
        raise ValueError(k)
    memo[key] = r
    return r


def signature(f, bat):
    sig = []
    for (S, succ, lab) in bat:
        r = check(f, S, succ, lab, {})
        sig.append(sum(1 << s for s in r))
    return tuple(sig)


# ---------------------------------------------------------------- stages
def family(pattern: str) -> str:
    p = pattern.strip()
    if p.startswith("Pattern 6.1"):
        return "response"
    if p.startswith("Pattern 4.1"):
        return "universality"
    if p.startswith("Pattern 2.1"):
        return "existence"
    return "other"


def stage_sample(n_units: int):
    rows = list(csv.DictReader(open(CSV, encoding="utf-8", errors="replace")))
    ok, bad = [], 0
    for r in rows:
        try:
            f = parse(r["CTL"])
            ok.append(dict(id=r["ID"], nl=r["requirements_text"].strip(), ctl=r["CTL"].strip(), family=family(r["Pattern"]),
                           atoms=sorted(atoms(f))))
        except (ParseError, RecursionError):
            bad += 1
    rng = np.random.default_rng(SEED)
    byfam = defaultdict(list)
    for u in ok:
        byfam[u["family"]].append(u)
    alloc = {k: int(round(n_units * len(v) / len(ok))) for k, v in byfam.items()}
    diff = n_units - sum(alloc.values())
    alloc["response"] += diff
    units = []
    for k, v in sorted(byfam.items()):
        idx = rng.choice(len(v), size=min(alloc[k], len(v)), replace=False)
        units += [v[i] for i in sorted(idx)]
    payload = dict(registration="REGISTRATION_V20.md", csv_rows=len(rows), parseable=len(ok), unparseable=bad,
                   allocation=alloc, units=units)
    json.dump(payload, open(UNITS, "w"), indent=1)
    print(f"rows {len(rows)} parseable {len(ok)} unparseable {bad}; sampled {len(units)} {alloc}")


PROMPT = """You translate natural-language software requirements into CTL (computation tree logic).

Requirement:
  "{nl}"

Use ONLY these atomic propositions (exact spelling): {aps}

Grammar: state formulas built from the atomic propositions with !, &, |, ->, <-> and the temporal operators AG, AF, AX, EG, EF, EX, A[p U q], E[p U q], A[p W q]. Parenthesize freely.

Reply with ONLY the CTL formula on a single line, nothing else."""


_LIMIT = re.compile(r"session limit|usage limit|rate limit|hit your .* limit|resets \d|API Error|overloaded", re.I)


def is_error_reply(reply) -> bool:
    return (not reply) or bool(_LIMIT.search(reply))


def ask(prompt: str, timeout: float = 120.0):
    try:
        out = subprocess.run(["claude", "-p", prompt], capture_output=True, text=True, timeout=timeout,
                             stdin=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return out.stdout


def extract_formula(reply):
    if not reply:
        return None
    txt = reply.strip()
    txt = re.sub(r"```[a-zA-Z]*", "", txt).replace("```", "").strip()
    for a, b in (("\u2192", "->"), ("\u21d2", "->"), ("\u2194", "<->"), ("\u21d4", "<->"), ("\u00ac", "!"),
                 ("\u2227", "&"), ("\u2228", "|"), ("\u2260", "!="), ("\u2264", "<="), ("\u2265", ">="),
                 ("\u2200", "A"), ("\u2203", "E"), ("\u25a1", "G"), ("\u25c7", "F"), ("\u25cb", "X")):
        txt = txt.replace(a, b)
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if not lines:
        return None
    # prefer the last line that looks like a formula
    for l in reversed(lines):
        if re.search(r"\b(AG|AF|AX|EG|EF|EX|A\[|E\[|A\(|E\()", l) or re.search(r"[&|!]", l):
            return l.strip().rstrip(".")
    return lines[-1]


def stage_generate(k: int, workers: int):
    """Sample units in registered order, completing each unit's k samples before
    the next, and stop cleanly when the CLI signals its usage limit -- so a
    quota stop leaves a prefix of fully sampled units, chosen before any outcome."""
    units = json.load(open(UNITS))["units"]
    done = json.load(open(SAMPLES)) if os.path.exists(SAMPLES) else {"cli_version": None, "samples": {}}
    try:
        done["cli_version"] = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    done.setdefault("completed_units", [])
    t0 = time.perf_counter()
    n_calls = n_err = 0
    stopped = False
    for u in units:
        uid = u["id"]
        have = done["samples"].get(uid, [])
        if len(have) >= k:
            if uid not in done["completed_units"]:
                done["completed_units"].append(uid)
            continue
        prompt = PROMPT.format(nl=u["nl"], aps=", ".join(u["atoms"]))
        attempts = 0
        while len(have) < k and attempts < 3:
            need = k - len(have)
            with cf.ThreadPoolExecutor(max_workers=min(workers, need)) as ex:
                replies = list(ex.map(lambda _i: ask(prompt), range(need)))
            n_calls += need
            errs = sum(1 for r in replies if is_error_reply(r))
            n_err += errs
            for r in replies:
                if not is_error_reply(r):
                    have.append(dict(j=len(have), raw=r))
            done["samples"][uid] = have
            if errs >= max(2, need // 2):          # the limit is back: stop, do not burn the window
                stopped = True
                break
            attempts += 1
        json.dump(done, open(SAMPLES, "w"))
        if len(have) >= k and uid not in done["completed_units"]:
            done["completed_units"].append(uid)
        if stopped:
            print(f"usage limit signalled at unit {uid}: {len(done['completed_units'])} units complete, stopping", flush=True)
            break
        if len(done["completed_units"]) % 20 == 0:
            print(f"  {len(done['completed_units'])}/{len(units)} units complete ({n_calls} calls, {n_err} errors, {time.perf_counter()-t0:.0f}s)", flush=True)
    done["registered_units"] = len(units)
    json.dump(done, open(SAMPLES, "w"))
    print(f"finished pass: {n_calls} calls, {n_err} error replies; {len(done['completed_units'])}/{len(units)} units complete", flush=True)


def qhat_loo(scores, delta=DELTA):
    """LOO thresholds: for unit i, k-th smallest of the others."""
    out = []
    s = np.array(scores)
    for i in range(len(s)):
        o = np.delete(s, i)
        k = int(math.floor(delta * (len(o) + 1)))
        out.append(float(np.sort(o)[k - 1]) if k >= 1 else 0.0)
    return out


def stage_analyze():
    U = json.load(open(UNITS))
    S = json.load(open(SAMPLES))
    complete = set(S.get("completed_units", []))
    per = []
    for u in U["units"]:
        if u["id"] not in complete:
            continue
        gold = parse(u["ctl"])
        raw = sorted(S["samples"].get(u["id"], []), key=lambda x: x["j"])
        parsed = []
        n_invalid = 0
        for x in raw:
            s = extract_formula(x["raw"])
            try:
                f = parse(s) if s else None
            except (ParseError, RecursionError):
                f = None
            if f is None:
                n_invalid += 1; continue
            parsed.append((s, f))
        aps = set(u["atoms"])
        for _, f in parsed:
            aps |= atoms(f)
        bat = battery(aps)
        gsig = signature(gold, bat)
        sigs = [signature(f, bat) for _, f in parsed]
        classes = {}
        cls_of = []
        for sg in sigs:
            if sg not in classes:
                classes[sg] = len(classes)
            cls_of.append(classes[sg])
        freq = Counter(cls_of)
        n_valid = len(parsed)
        gold_cls = classes.get(gsig)
        syn_hit = any(canon(f) == canon(gold) for _, f in parsed)
        maj = max(freq, key=lambda c: (freq[c], -cls_of.index(c))) if freq else None
        per.append(dict(id=u["id"], family=u["family"], n_atoms=len(u["atoms"]), n_valid=n_valid, n_invalid=n_invalid,
                        n_classes=len(classes), generated=gold_cls is not None, syntactic_hit=bool(syn_hit),
                        gold_freq=(freq[gold_cls] / n_valid) if gold_cls is not None else None,
                        top1_hit=bool(parsed and cls_of[0] == gold_cls), majority_hit=bool(maj is not None and maj == gold_cls),
                        freqs=sorted((c / n_valid for c in freq.values()), reverse=True)))
    gen = [p for p in per if p["generated"]]
    q = qhat_loo([p["gold_freq"] for p in gen]) if gen else []
    qi = {p["id"]: qq for p, qq in zip(gen, q)}
    for p in per:
        if p["generated"]:
            p["qhat"] = qi[p["id"]]
            p["retained"] = bool(p["gold_freq"] >= p["qhat"] - 1e-12)
            p["set_size"] = sum(1 for f in p["freqs"] if f >= p["qhat"] - 1e-12)
        else:
            p["qhat"] = None; p["retained"] = False
            # a non-proposed unit's set under the pooled threshold (median LOO) for the size statistic
            p["set_size"] = sum(1 for f in p["freqs"] if f >= (float(np.median(q)) if q else 0) - 1e-12)

    def agg(sel):
        n = len(sel)
        if not n:
            return dict(n=0)
        g = [p for p in sel if p["generated"]]
        return dict(n=n, generated=len(g), generated_rate=len(g) / n, syntactic_hit=sum(p["syntactic_hit"] for p in sel) / n,
                    top1=sum(p["top1_hit"] for p in sel) / n, majority=sum(p["majority_hit"] for p in sel) / n,
                    retained=sum(p["retained"] for p in sel) / n,
                    retained_given_generated=(sum(p["retained"] for p in g) / len(g)) if g else None,
                    set_size_median=float(np.median([p["set_size"] for p in sel])),
                    classes_median=float(np.median([p["n_classes"] for p in sel])),
                    efficiency_median=float(np.median([p["set_size"] / p["n_classes"] for p in sel if p["n_classes"]])) if any(p["n_classes"] for p in sel) else None,
                    invalid_rate=sum(p["n_invalid"] for p in sel) / max(1, sum(p["n_invalid"] + p["n_valid"] for p in sel)))
    res = dict(registration="REGISTRATION_V20.md", cli_version=S.get("cli_version"), n_struct=N_STRUCT, delta_sem=DELTA,
               csv_rows=U["csv_rows"], parseable=U["parseable"], unparseable=U["unparseable"],
               registered_units=len(U["units"]), completed_units=len(complete),
               deviation=("units sampled in registered order; a CLI usage-limit stop leaves a prefix of fully sampled units"
                          if len(complete) < len(U["units"]) else None),
               overall=agg(per), by_family={f: agg([p for p in per if p["family"] == f]) for f in sorted({p["family"] for p in per})},
               per_unit=per)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(dict(overall=res["overall"], by_family=res["by_family"]), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--units", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.stage in ("sample", "all"):
        stage_sample(a.units)
    if a.stage in ("generate", "all"):
        stage_generate(a.k, a.workers)
    if a.stage in ("analyze", "all"):
        stage_analyze()
