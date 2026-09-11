"""G0: target-independent candidate generation (make8 v4, gate G0).

Generates candidate PromReadings from a rule's OBSERVABLE surface only:
annotation text, alert name, severity label. It never reads the alert
expression, the parsed gold AST, or any feature derived from them; the
fixture universe is likewise derived from the generated pool, not the
target. Frozen vocabulary (metric token index, threshold bank, window
set) comes from the development corpus E0-D only.

Reading identity for recall/coverage (frozen, selector-free projection):
    proj(r) = (metric, comparator, round(threshold, 6),
               rate_window_s, aggregation, for_s)
selectors, agg_by, severity, and logic are execution details a text
surface cannot pin down and are excluded from the E0 DSL.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .evaluate import prom_fire
from .parse import PromReading, PromTarget, prom_threshold_bank
from .score import (_text_direction, _text_durations_s, _text_numbers,
                    tokenize)

MAXPOOL = 256
K_METRICS = 6
K_THRESH = 12
RATE_CUES = ("rate", "per second", "per-second", "/s", "throughput",
             "errors per", "requests per", "qps", "rps")
AGG_CUES = {"sum": ("total", "sum", "overall", "aggregate"),
            "avg": ("average", "avg", "mean"),
            "max": ("max", "maximum", "peak", "highest"),
            "min": ("min", "minimum", "lowest"),
            "count": ("count", "number of")}


def proj(r: PromReading) -> tuple:
    return (r.metric, r.comparator, round(float(r.threshold), 6),
            r.rate_window_s, r.aggregation, float(r.for_s))


def _name_tokens(name: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Z][a-z0-9]+|[A-Z]+(?![a-z])|[a-z0-9]+", name or "")]


def build_vocab(dev_targets: Sequence[PromTarget]) -> dict:
    """Frozen E0-D vocabulary: metric token index, threshold bank, windows."""
    mtok: Dict[str, set] = {}
    winc: Counter = Counter()
    forc: Counter = Counter()
    aggc: Counter = Counter()
    lex: Dict[str, Counter] = defaultdict(Counter)
    for t in dev_targets:
        r = t.reading
        mtok.setdefault(r.metric, set()).update(
            x for x in re.split(r"[_:]", r.metric) if len(x) > 1)
        if r.rate_window_s:
            winc[float(r.rate_window_s)] += 1
        if r.for_s:
            forc[float(r.for_s)] += 1
        aggc[r.aggregation or "none"] += 1
        for nt in _name_tokens(t.name):
            lex[nt][r.metric] += 1
    bank = prom_threshold_bank(list(dev_targets))
    modal: Dict[str, List[float]] = {}
    for root, vals in bank.items():
        modal[root] = [v for v, _ in Counter(vals).most_common(6)]
    windows = [w for w, _ in winc.most_common(2)] or [300.0]
    modal_fors = [f for f, _ in forc.most_common(4)] or [900.0]
    agg_order = [a for a, _ in aggc.most_common()] + \
        [a for a in ("none", "sum", "max", "avg", "min", "count")
         if a not in aggc]
    name_lex = {tok: [m for m, _ in c.most_common(4)]
                for tok, c in lex.items() if len(tok) > 2}
    return dict(metric_tokens={k: sorted(v) for k, v in mtok.items()},
                bank={k: sorted(set(v)) for k, v in bank.items()},
                modal=modal, windows=windows, modal_fors=modal_fors,
                agg_order=agg_order, name_lex=name_lex)


def _metric_candidates(toks: List[str], vocab: dict) -> List[Tuple[str, float]]:
    ts = set(toks)
    base: Dict[str, float] = {}
    for m, mt in vocab["metric_tokens"].items():
        inter = len(ts & set(mt))
        if inter:
            base[m] = inter + inter / (len(ts | set(mt)) + 1.0)
    for tok in ts:
        for m in vocab.get("name_lex", {}).get(tok, []):
            base[m] = base.get(m, 0.0) + 2.0
    scored = sorted(base.items(), key=lambda x: (-x[1], x[0]))
    return scored[:K_METRICS]


def _thresholds_for(root: str, text_nums: List[float], vocab: dict) -> List[float]:
    bank = vocab["bank"].get(root) or vocab["bank"].get("__all__", [])
    out: List[float] = []
    for x in text_nums:
        out.append(x)
        if 1.0 < x <= 100.0:
            out.append(round(x / 100.0, 6))          # "90%" -> 0.9
    for x in list(out):
        near = sorted(bank, key=lambda b: abs(b - x))[:3]
        out.extend(near)
    out.extend(vocab["modal"].get(root, [])[:6])
    seen, uniq = set(), []
    for v in out:
        v = round(float(v), 6)
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq[:K_THRESH]


def generate(rule_obs: dict, vocab: dict) -> Tuple[List[PromReading], dict]:
    """rule_obs: dict with keys text, name, severity ONLY (no expr/AST)."""
    assert "expr" not in rule_obs and "reading" not in rule_obs, \
        "G0 violation: target material passed to the generator"
    text = rule_obs.get("text", "") or ""
    toks = tokenize(text) + _name_tokens(rule_obs.get("name", ""))
    low = text.lower()

    metrics = _metric_candidates(toks, vocab)
    tdir, _ = _text_direction(text)
    cued = {"gt": [">", ">="], "lt": ["<", "<="]}.get(tdir)
    if cued:
        comps = cued + [c for c in (">", ">=", "<", "<=", "==")
                        if c not in cued]
        n_cued = 2
    else:
        comps = [">", ">=", "<", "<=", "=="]
        n_cued = 5
    text_nums = _text_numbers(text)[:4]
    durs = [d for d in _text_durations_s(text) if d > 0][:3]
    for_opts, fseen = [], set()
    for f in [0.0] + durs + vocab.get("modal_fors", [900.0])[:4]:
        if f not in fseen:
            fseen.add(f)
            for_opts.append(f)
    for_opts = for_opts[:6]
    win_opts: List[Optional[float]] = [None] + \
        list(vocab.get("windows", [300.0])[:2])
    aggs = vocab.get("agg_order",
                     ["none", "sum", "max", "avg", "min", "count"])

    per_metric: List[List[Tuple[tuple, PromReading]]] = []
    seen = set()
    for rank, (m, msc) in enumerate(metrics):
        root = m.split("_")[0]
        rows = []
        for ti, th in enumerate(_thresholds_for(root, text_nums, vocab)):
            from_text = any(abs(th - x) < 1e-9 or abs(th - x / 100.0) < 1e-9
                            for x in text_nums)
            for ci, cmpo in enumerate(comps):
                for fi, fs in enumerate(for_opts):
                    for wi, w in enumerate(win_opts):
                        for ai, agg in enumerate(aggs):
                            r = PromReading(
                                metric=m, selectors=(), comparator=cmpo,
                                threshold=th, rate_window_s=w,
                                aggregation=agg, agg_by=(), for_s=fs,
                                severity="")
                            key = proj(r)
                            if key in seen:
                                continue
                            seen.add(key)
                            fs_pref = 0 if (durs and fs in durs) else \
                                (1 if fs == 0.0 else fi)
                            radius = (ti + 1.5 * (0 if ci < n_cued else 2)
                                      + 0.5 * ci + fs_pref + wi + ai
                                      + (0 if from_text else 0.5))
                            prio = (radius, repr(key))
                            rows.append((prio, r))
        rows.sort(key=lambda x: x[0])
        per_metric.append(rows)

    # proportional budget slices per metric rank, remainder round-robin
    shares = [0.34, 0.22, 0.14, 0.12, 0.10, 0.08][:len(per_metric)]
    tot = sum(shares) or 1.0
    quota = [max(4, int(MAXPOOL * sh / tot)) for sh in shares]
    cands, meta = [], dict(overflow=0)
    for rows, q in zip(per_metric, quota):
        cands.extend(rows[:q])
        meta["overflow"] += max(0, len(rows) - q)
    leftovers = []
    for rows, q in zip(per_metric, quota):
        leftovers.extend(rows[q:])
    leftovers.sort(key=lambda x: x[0])
    room = MAXPOOL - len(cands)
    if room > 0:
        cands.extend(leftovers[:room])
    cands.sort(key=lambda x: x[0])
    cands = cands[:MAXPOOL]
    meta["n"] = len(cands)
    meta["n_metrics"] = len(metrics)
    return [r for _, r in cands], meta


# ------------------------------------------------------------------
# Pool-derived fixtures (target-independent)
# ------------------------------------------------------------------
def pool_fixtures(readings: Sequence[PromReading]) -> List[dict]:
    """Fixture scenarios built from the pool's own thresholds/durations."""
    thrs = sorted({round(float(r.threshold), 6) for r in readings})
    if not thrs:
        return []
    span = (max(thrs) - min(thrs)) or abs(thrs[0]) or 1.0
    d = max(span * 0.01, 1e-6)
    fors = sorted({float(r.for_s) for r in readings})
    H = 40
    base_lo = min(thrs) - 2 * d - abs(min(thrs)) * 0.5
    base_hi = max(thrs) + 2 * d + abs(max(thrs)) * 0.5
    scen: List[dict] = []

    def series(level, minutes, base):
        s = [base] * H
        for i in range(min(minutes, H)):
            s[5 + i if 5 + i < H else H - 1] = level
        return s

    def add(sid, vals):
        scen.append(dict(id=sid, instances=[dict(
            labels={"instance": "i1", "cluster": "prod"},
            series=vals, guard=[1.0] * H, secondary=vals)]))

    add("quiet_lo", [base_lo] * H)
    add("quiet_hi", [base_hi] * H)
    long_run = max([int(f // 60) for f in fors if f > 0] + [15]) + 5
    for j, t in enumerate(thrs):
        for eps, tag in ((d, "above"), (-d, "below"), (0.0, "at")):
            add(f"sus_{j}_{tag}", series(t + eps, long_run, base_lo))
        add(f"spike_{j}", series(t + d, 2, base_lo))
        for f in fors:
            if f > 0:
                mins = int(f // 60)
                add(f"run_{j}_{mins}m_exact", series(t + d, mins, base_lo))
                if mins > 1:
                    add(f"run_{j}_{mins}m_short", series(t + d, mins - 1,
                                                         base_lo))
    # two-instance scenario separating sum/avg/max/min aggregations
    mid = thrs[len(thrs) // 2]
    inst = [dict(labels={"instance": f"i{k}", "cluster": "prod"},
                 series=[v] * H, guard=[1.0] * H, secondary=[v] * H)
            for k, v in enumerate((mid + d, base_lo))]
    scen.append(dict(id="two_inst", instances=inst))
    return scen


def pool_vectors(readings: Sequence[PromReading]) -> List[Tuple[int, ...]]:
    fx = pool_fixtures(readings)
    return [tuple(prom_fire(r, s) for s in fx) for r in readings]
