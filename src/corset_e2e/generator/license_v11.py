"""v11: deterministic retrieval licenser for the metric axis.

Drop-in for generator.generate.Catalog at the same seam the v6 LLM
licenser used: everything downstream (slot grids, additive score,
branch-and-bound retention, in_pool) is untouched.  Replaces the frozen
subtoken-overlap top-64 license, whose measured failure is ranking at
catalog scale (per-repo rho_gen from the archived v4 run: 0.050 at
17,956 names, 0.272 at 12,475, vs 0.6+ below ~3k), with three
target-blind evidence channels:

  NAME  IDF-weighted coverage of the metric's subtokens by the rule
        text, both sides normalized by a frozen abbreviation/plural
        table (avail~available, mem~memory, ...).  The frozen license
        weighted every subtoken equally and broke ties alphabetically,
        which at 10^4-name scale is what drowned it.
  CTX   BM25-style coverage of the rule text by the metric's repository
        context profile (build_context_v11.py: HELP strings, dashboard
        titles, doc prose from non-rule files) -- the deterministic
        version of the evidence REGISTRATION_V6 showed an LLM could use.
  VERB  widened verbatim capture: colon-form recording-rule names and
        single-underscore names, in text or alert name, licensed even
        when absent from the catalog (the unit's own visible text names
        the metric).

All constants below are frozen from the development split and corpus
statistics only (see analysis/dev_design_v11.py); cal/test units were
never read.  No model, no network, no randomness.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

from corset_e2e.generator.generate import metric_subtokens

# ---- frozen constants (development split + corpus statistics only) ---------
K_PRIMARY = 64            # licensing breadth of the primary endpoint (= v4's)
W_NAME = 1.0              # name-channel weight (ablation switch; frozen 1.0)
W_CTX = 1.0               # context-channel weight relative to the name channel
NORM_ON = True            # normalization table (ablation switch; frozen True)
MIN_TOTAL = 1e-9          # some evidence required; no evidence-free licensing
# name channel scores MATCHED EVIDENCE MASS, not coverage fraction: the
# development stress dump showed coverage normalization handing top ranks
# to two-token harvest debris ('error_value', 'label_values') that generic
# text fully covers, while five-token gold names scored 3/5.  Mass with a
# mild unmatched penalty keeps accumulated rare-token evidence decisive.
NAME_SAT = 1.0            # saturation offset (~one informative token's IDF)
NAME_PEN = 0.25           # penalty per unit of unmatched subtoken IDF mass
ABBREV: Dict[str, str] = {
    "available": "avail", "availability": "avail",
    "utilization": "util", "utilisation": "util", "usage": "util",
    "memory": "mem", "message": "msg", "error": "err", "errored": "err",
    "request": "req", "response": "resp", "connection": "conn",
    "replica": "repl", "replication": "repl", "replicated": "repl",
    "filesystem": "fs", "config": "cfg", "configuration": "cfg",
    "directory": "dir", "kubernetes": "k8s", "certificate": "cert",
    "authentication": "auth", "authorization": "auth",
    "transmit": "tx", "transmitted": "tx", "receive": "rx", "received": "rx",
    "second": "sec", "minute": "min", "latencies": "latency",
    "synchronization": "sync", "synchronisation": "sync",
    "operation": "op", "database": "db", "percentage": "percent",
}

_VERB_WIDE = re.compile(r"[a-z][a-z0-9_]*(?::[a-z0-9_]+)+"
                        r"|[a-z][a-z0-9]*(?:_[a-z0-9]+)+")
_VERB_JUNK = frozenset({"group_left", "group_right"})


def canon(tok: str) -> str:
    """Frozen normalization: abbreviation table, then plural strip."""
    if not NORM_ON:
        return tok
    if tok in ABBREV:
        return ABBREV[tok]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        t = tok[:-1]
        return ABBREV.get(t, t)
    return tok


def wide_verbatim(text_low: str) -> List[str]:
    out = []
    for m in _VERB_WIDE.finditer(text_low):
        n = m.group(0).rstrip("_")
        if len(n) >= 4 and n not in _VERB_JUNK and ("_" in n or ":" in n):
            if n not in out:
                out.append(n)
    return out


class RepoIndexV11:
    """Heavy per-repo evidence index, built once and shared across units.

    `metrics` is the repo's namespace of record (v4 file harvest union the
    v3 expression harvest); `context` maps metric -> {token: count}
    profiles from build_context_v11.py.  IDF statistics are properties of
    these committed corpus artifacts, not of any unit.
    """

    def __init__(self, metrics: Sequence[str], context: dict):
        self.metrics = sorted(set(metrics))
        self.csub = {m: {canon(t) for t in metric_subtokens(m)}
                     for m in self.metrics}
        self.ctx: Dict[str, Counter] = {}
        for m in self.metrics:
            prof = context.get(m)
            if prof:
                cc: Counter = Counter()
                for t, c in prof.items():
                    cc[canon(t)] += c
                self.ctx[m] = cc
        n = max(len(self.metrics), 1)
        df_name: Counter = Counter()
        for s in self.csub.values():
            df_name.update(s)
        self.idf_name = {t: math.log(1.0 + n / d) for t, d in df_name.items()}
        mctx = max(len(self.ctx), 1)
        df_ctx: Counter = Counter()
        for cc in self.ctx.values():
            df_ctx.update(cc.keys())
        self.idf_ctx = {t: math.log(1.0 + mctx / d) for t, d in df_ctx.items()}
        # inverted posting lists so a license call touches only metrics
        # sharing at least one canonical token with the rule text
        self.post_name: Dict[str, list] = {}
        for m, s in self.csub.items():
            for t in s:
                self.post_name.setdefault(t, []).append(m)
        self.post_ctx: Dict[str, list] = {}
        for m, cc in self.ctx.items():
            for t in cc:
                self.post_ctx.setdefault(t, []).append(m)


class CatalogV11:
    """One unit's licensed metric axis: a light view over a RepoIndexV11.

    `allowed` restricts the served namespace to the unit's own catalog of
    record (per-cluster v4u: repo file harvest union the v3
    leave-one-cluster-out set); `visible_text` is the unit's own visible
    text plus alert name, used only for the widened verbatim channel.
    Duck-types generate.Catalog: license(toks, verbatim) returns the
    licensed names, so the frozen extract_slots runs unmodified.
    """

    def __init__(self, metrics: Sequence[str], context: dict = None,
                 visible_text: str = "", k: int = K_PRIMARY,
                 index: "RepoIndexV11" = None, allowed=None):
        if index is None:
            index = RepoIndexV11(metrics, context or {})
            allowed = None
        self.idx = index
        self.allowed = set(allowed) if allowed is not None else set(index.metrics)
        self.k = k
        self.metrics = sorted(self.allowed)
        self._verbatim_text = wide_verbatim(visible_text.lower())
        self._subtok = None

    @property
    def subtok(self):
        """Name->subtoken map over the served names, for the frozen
        extract_slots verbatim filter (`w in catalog.subtok`)."""
        if self._subtok is None:
            self._subtok = {m: self.idx.csub.get(m, frozenset())
                            for m in self.allowed}
        return self._subtok

    @classmethod
    def from_index(cls, index: "RepoIndexV11", allowed, visible_text: str,
                   k: int = K_PRIMARY) -> "CatalogV11":
        return cls([], index=index, allowed=allowed, visible_text=visible_text,
                   k=k)

    # -- the license --------------------------------------------------------
    def license(self, toks: Sequence[str], verbatim: Sequence[str]) -> List[str]:
        ix = self.idx
        tcanon = {canon(t) for t in toks}
        tq = sum(ix.idf_ctx.get(t, 0.0) for t in tcanon)
        cand = set()
        for t in tcanon:
            cand.update(ix.post_name.get(t, ()))
            cand.update(ix.post_ctx.get(t, ()))
        cand &= self.allowed
        scored = []
        for m in cand:
            sub = ix.csub[m]
            num = sum(ix.idf_name.get(t, 0.0) for t in sub & tcanon)
            unmatched = sum(ix.idf_name.get(t, 1.0) for t in sub - tcanon)
            ns = num / (num + NAME_SAT + NAME_PEN * unmatched) if num > 0 else 0.0
            cs = 0.0
            cc = ix.ctx.get(m)
            if cc and tq > 0:
                acc = 0.0
                for t in tcanon:
                    c = cc.get(t)
                    if c:
                        acc += ix.idf_ctx.get(t, 0.0) * (c / (c + 3.0))
                cs = acc / tq
            total = W_NAME * ns + W_CTX * cs
            if total > MIN_TOTAL:
                scored.append((-total, -len(sub & tcanon), m))
        scored.sort()
        out = [m for _t, _r, m in scored[: self.k]]
        for v in list(verbatim) + self._verbatim_text:
            if v not in out:
                out.append(v)
        return out
