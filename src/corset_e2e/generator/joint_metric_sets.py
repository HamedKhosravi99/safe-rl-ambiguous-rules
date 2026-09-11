"""V16: joint metric-set retrieval, p(q | l) p(M | q, l, V).

The independent-slot beam proposed each metric on its own and the gold
assignment is joint -- 70% of golds name two or more distinct series -- so
the correct combination sat combinatorially deep in every per-slot
ranking. Here the GROUP is the object: a cardinality prior, a unary
narrowing stage whose only job is to keep the whole gold group inside a
Top-K list, a pairwise compatibility model learned from dev gold groups,
and a bounded set search reranked by whole-set features.

Everything is fit on dev with grouped out-of-fold CV (folds by cluster
hash); every number this module reports is OOF. Search breadths are frozen
in REGISTRATION_V16.md and are not swept. Repository co-occurrence enters
only as two sparse features, because the allowed evidence corpus covers
the gold names of just 2.5% of dev rules.

This module never opens the hidden store except through the gold archive
already built for measurement, and the generator path (vocabulary, unary
features, search) reads only visible text and committed corpus artifacts.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(ROOT, "results/e2e")
sys.path.insert(0, os.path.join(str(ROOT), "src"))

from corset_e2e.dsl.schema_g2 import IN_G2, classify_target_g2  # noqa: E402
from corset_e2e.generator.generate import metric_subtokens, tokens  # noqa: E402
from corset_e2e.calibration.run_e2e_v11 import V11Server  # noqa: E402
from corset_e2e.calibration.run_e2e_g1 import refs_for  # noqa: E402
from corset_e2e.source_grounded.harvest_recording_names import (  # noqa: E402
    names_for_unit)
from corset_e2e.calibration.run_e2e_v4 import hid, load_split, vis  # noqa: E402

FOLDS = 5
K_GRID = (32, 64, 128, 256)
# search breadths -- REGISTRATION_V16, frozen a priori, not swept
P_PAIR_BASE = 96
N_TRIPLE_SEED = 250
TRIPLE_ADD = 48
N_QUAD_SEED = 120
QUAD_ADD = 32
POOL_KEEP = 400
DECLARED_CAP = 400
Q_MAX = 4

_SUFFIX = re.compile(r"[_:](\d+)(ms|[smhdw])$")
_UNIT_S = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}

# complementary roles that co-occur inside one rule (numerator/denominator,
# used/capacity); order-free
ROLE_SETS = [
    ({"error", "errors", "failed", "failure", "failures", "unavailable",
      "dropped", "evicted", "unhealthy", "misscheduled"},
     {"total", "count", "all", "requests", "request", "attempts", "sum",
      "desired"}),
    ({"used", "usage", "working", "allocated"},
     {"capacity", "limit", "limits", "total", "size", "max", "quota",
      "allocatable"}),
    ({"free", "avail", "available"},
     {"size", "total", "capacity", "files"}),
    ({"success", "succeeded", "ok", "ready", "healthy", "updated"},
     {"total", "requests", "count", "desired", "spec", "replicas"}),
]

_RATIO_WORDS = ("ratio", "percent", "percentage", "saturation", "apdex",
                "budget", "utilization", "usage")
_BURN_WORDS = ("burn", "slo", "sli", "error budget", "windows")


def fold_of(cluster: str) -> int:
    return int(hashlib.sha1(cluster.encode()).hexdigest(), 16) % FOLDS


def walk(a):
    yield a
    if isinstance(a, list):
        for x in a:
            if isinstance(x, list):
                yield from walk(x)


def gold_metric_seq(ast):
    """Metric occurrences of a canonical AST, left to right."""
    out = []
    for nd in walk(ast):
        if (isinstance(nd, list) and nd and nd[0] == "sel"
                and isinstance(nd[1], str)):
            out.append(nd[1])
    return out


def base_of(m: str) -> str:
    return _SUFFIX.sub("", m)


def dur_of(m: str):
    mm = _SUFFIX.search(m)
    return float(mm.group(1)) * _UNIT_S[mm.group(2)] if mm else None


def _role_flags(sub: set):
    a_flags = [bool(sub & a) for a, _b in ROLE_SETS]
    b_flags = [bool(sub & b) for _a, b in ROLE_SETS]
    return a_flags, b_flags


# ------------------------------------------------------------------ units

def load_units(split: str = "dev", limit: int = 0):
    """Everything the retrieval stages need, computed once per unit."""
    gold = json.load(open(os.path.join(OUT, "gold_ast.json")))["gold"]
    std = set(json.load(open(os.path.join(OUT,
                                          "standard_vocab.json")))["names"])
    rec = json.load(open(os.path.join(OUT, "recording_names.json")))
    ev = json.load(open(os.path.join(OUT, "evidence_corpus.json")))
    # repo-level co-occurrence pairs; evidence clusters are non-rule files
    # and unit clusters are rule files, so this is own-cluster-safe as-is
    cooc = {}
    hyper = {}
    for repo, rows in ev.items():
        ps, hs = set(), []
        for r in rows:
            ms = sorted(set(r["metrics"]))
            hs.append(set(ms))
            for a, b in itertools.combinations(ms, 2):
                ps.add((a, b))
        cooc[repo] = ps
        hyper[repo] = hs

    server = V11Server()
    units = []
    entries = load_split(split)
    if limit:
        entries = entries[:limit]
    for e in entries:
        rid = e["rule_id"]
        if classify_target_g2(hid(rid)) != IN_G2:
            continue
        g = gold.get(rid)
        if not g or "ast" not in g:
            continue
        seq = gold_metric_seq(g["ast"])
        M = set(seq)
        if not M:
            continue
        v = vis(rid)
        toks = tokens(v.get("text", "")) + tokens(v.get("alert_name", ""))
        T = set(toks)
        lic = list(server.catalog_for(e, v).license(toks, []))
        refs = list(refs_for(lic, server.idx[e["repo"]], toks))
        dec = names_for_unit(rec, e["repo"], e["cluster_id"])

        # candidate pool: licensed + refs + ALL standard names + declared
        # names with any textual overlap (declared can be thousands)
        cand = {}

        def add(m, src):
            if m in cand:
                cand[m][src] = 1.0
                return
            sub = set(metric_subtokens(m))
            ov = len(sub & T) / len(sub) if sub else 0.0
            cand[m] = dict(ov=ov, lic=0.0, ref=0.0, dec=0.0, std=0.0,
                           sub=sub)
            cand[m][src] = 1.0

        L = len(lic) or 1
        for i, m in enumerate(lic):
            add(m, "lic")
            cand[m]["licsc"] = (L - i) / L
        for m in refs:
            add(m, "ref")
        for m in std:
            add(m, "std")
        dec_scored = []
        for m in dec:
            sub = set(metric_subtokens(m))
            ov = len(sub & T) / len(sub) if sub else 0.0
            if ov > 0:
                dec_scored.append((ov, m))
        dec_scored.sort(key=lambda z: -z[0])
        for _ov, m in dec_scored[:DECLARED_CAP]:
            add(m, "dec")

        names = sorted(cand)
        X = np.zeros((len(names), 6), dtype=np.float32)
        for i, m in enumerate(names):
            c = cand[m]
            X[i] = (c.get("licsc", 0.0), c["ov"], c["ref"], c["dec"],
                    c["std"], 1.0)
        y = np.array([1.0 if m in M else 0.0 for m in names],
                     dtype=np.float32)

        text = f"{v.get('alert_name', '')} {v.get('text', '')}".lower()
        qf = np.array([
            1.0,
            math.log1p(len(toks)),
            1.0 if any(w in text for w in _RATIO_WORDS) else 0.0,
            1.0 if any(w in text for w in _BURN_WORDS) else 0.0,
            1.0 if "%" in text else 0.0,
            min(len(re.findall(r"\d+(?:\.\d+)?", text)), 5) / 5.0,
        ], dtype=np.float32)

        units.append(dict(
            rid=rid, repo=e["repo"], cluster=e["cluster_id"],
            fold=fold_of(e["cluster_id"]),
            textkey=(v.get("alert_name", "").strip(),
                     " ".join((v.get("text") or "").split())),
            names=names, X=X, y=y, M=M, seq=seq, q=len(M), qf=qf,
            subs={m: cand[m]["sub"] for m in names},
            src={m: (cand[m]["lic"], cand[m]["dec"], cand[m]["std"])
                 for m in names},
            cooc=cooc.get(e["repo"], set()),
            hyper=hyper.get(e["repo"], []),
        ))
    return units


# ----------------------------------------------------------------- models

def _logistic_fit(X, y, w0=None, iters=250, lr=0.5, l2=1e-3, sw=None):
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64) if w0 is None else w0.copy()
    sw = np.ones(n) if sw is None else sw
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        g = (X.T @ ((p - y) * sw)) / sw.sum() + l2 * w
        w -= lr * g
    return w


def fit_unary(units, fold):
    rows, ys, sws = [], [], []
    for u in units:
        if u["fold"] == fold:
            continue
        pos = np.where(u["y"] > 0)[0]
        if not len(pos):
            continue
        # hard negatives: best non-gold by the raw heuristic
        h0 = u["X"][:, 0] + u["X"][:, 1]
        neg = [i for i in np.argsort(-h0) if u["y"][i] == 0][:40]
        idx = list(pos) + list(neg)
        rows.append(u["X"][idx])
        ys.append(u["y"][idx])
        sws.append(np.where(u["y"][idx] > 0, float(len(neg)) /
                            max(len(pos), 1), 1.0))
    X = np.vstack(rows)
    return _logistic_fit(X, np.concatenate(ys), sw=np.concatenate(sws))


def fit_qprior(units, fold):
    """Multinomial logistic over q in 1..Q_MAX, grouped OOF."""
    tr = [u for u in units if u["fold"] != fold and u["q"] <= Q_MAX]
    Xq = np.stack([u["qf"] for u in tr])
    yq = np.array([u["q"] - 1 for u in tr])
    d = Xq.shape[1]
    W = np.zeros((Q_MAX, d))
    for _ in range(300):
        Z = Xq @ W.T
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        Y = np.zeros_like(P)
        Y[np.arange(len(yq)), yq] = 1.0
        G = (P - Y).T @ Xq / len(yq) + 1e-3 * W
        W -= 0.5 * G
    return W


def qprior_logp(W, qf):
    z = W @ qf
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    return np.log(np.maximum(p, 1e-9))


PAIR_DIM = 9


def pair_features(u, mi, mj):
    # gold metrics may lie outside the unit's vocabulary (C_V < 1); their
    # pair features are still well defined, so compute subtokens on demand
    si = u["subs"].get(mi)
    if si is None:
        si = set(metric_subtokens(mi))
    sj = u["subs"].get(mj)
    if sj is None:
        sj = set(metric_subtokens(mj))
    jac = len(si & sj) / max(len(si | sj), 1)
    same_first = 1.0 if (mi.split("_")[0].split(":")[0]
                         == mj.split("_")[0].split(":")[0]) else 0.0
    fam = 1.0 if (base_of(mi) == base_of(mj) and mi != mj) else 0.0
    ai, bi = _role_flags(si)
    aj, bj = _role_flags(sj)
    comp = 1.0 if any((ai[k] and bj[k]) or (aj[k] and bi[k])
                      for k in range(len(ROLE_SETS))) else 0.0
    co = 1.0 if ((min(mi, mj), max(mi, mj)) in u["cooc"]) else 0.0
    li, di, sti = u["src"].get(mi, (0.0, 0.0, 0.0))
    lj, dj, stj = u["src"].get(mj, (0.0, 0.0, 0.0))
    return np.array([same_first, jac, fam, comp, co,
                     sti * stj, di * dj, li * lj, 1.0], dtype=np.float64)


def fit_pair(units, fold, rng_seed=13):
    import random
    rng = random.Random(rng_seed)
    rows, ys = [], []
    for u in units:
        if u["fold"] == fold or u["q"] < 2:
            continue
        gold = sorted(u["M"])
        h0 = u["X"][:, 0] + u["X"][:, 1]
        negs = [u["names"][i] for i in np.argsort(-h0)
                if u["y"][i] == 0][:30]
        for a, b in itertools.combinations(gold, 2):
            rows.append(pair_features(u, a, b))
            ys.append(1.0)
        for a in gold:
            for b in rng.sample(negs, min(6, len(negs))):
                rows.append(pair_features(u, a, b))
                ys.append(0.0)
        for _ in range(4):
            if len(negs) >= 2:
                a, b = rng.sample(negs, 2)
                rows.append(pair_features(u, a, b))
                ys.append(0.0)
    X = np.vstack(rows)
    y = np.array(ys)
    sw = np.where(y > 0, (y == 0).sum() / max((y > 0).sum(), 1), 1.0)
    return _logistic_fit(X, y, sw=sw)


# ----------------------------------------------------------------- search

PHI_DIM = 14


def _phi(u, group, U, pair_cache, wp, logq):
    g = sorted(group)
    q = len(g)
    us = np.array([U[m] for m in g])
    if q >= 2:
        ps = []
        for a, b in itertools.combinations(g, 2):
            key = (a, b)
            if key not in pair_cache:
                pair_cache[key] = float(pair_features(u, a, b) @ wp)
            ps.append(pair_cache[key])
        meanp, minp = float(np.mean(ps)), float(np.min(ps))
    else:
        meanp = minp = 0.0
    gs = set(g)
    subset = 0.0
    jbest = 0.0
    for h in u["hyper"]:
        inter = len(gs & h)
        if not inter:
            continue
        jbest = max(jbest, inter / len(gs | h))
        if gs <= h:
            subset = 1.0
    fam = 0.0
    if q >= 2:
        bases = [base_of(m) for m in g]
        fam = (q - len(set(bases))) / (q - 1)
    onehot = [0.0] * Q_MAX
    onehot[q - 1] = 1.0
    return np.array([us.mean(), us.min(), us.max(), meanp, minp,
                     logq[q - 1], subset, jbest, fam] + onehot + [1.0][:1],
                    dtype=np.float64)[:PHI_DIM]


def search_groups(u, wu, wp, Wq, K):
    """Candidate groups with base score and features; top POOL_KEEP."""
    U = dict(zip(u["names"], (u["X"] @ wu).tolist()))
    logq = qprior_logp(Wq, u["qf"])
    tops = sorted(u["names"], key=lambda m: -U[m])
    pair_cache = {}
    cands = {}

    def consider(group):
        key = frozenset(group)
        if key in cands or len(key) > Q_MAX:
            return
        phi = _phi(u, key, U, pair_cache, wp, logq)
        s0 = phi[0] + phi[3] + phi[5]
        cands[key] = (s0, phi)

    for m in tops[:min(K, 128)]:
        consider((m,))
    pb = tops[:min(K, P_PAIR_BASE)]
    for a, b in itertools.combinations(pb, 2):
        consider((a, b))
    pairs = sorted(((s0, k) for k, (s0, _p) in cands.items()
                    if len(k) == 2), key=lambda z: -z[0])
    add3 = tops[:min(K, TRIPLE_ADD)]
    for _s, k in pairs[:N_TRIPLE_SEED]:
        for m in add3:
            if m not in k:
                consider(tuple(k) + (m,))
    triples = sorted(((s0, k) for k, (s0, _p) in cands.items()
                      if len(k) == 3), key=lambda z: -z[0])
    add4 = tops[:min(K, QUAD_ADD)]
    for _s, k in triples[:N_QUAD_SEED]:
        for m in add4:
            if m not in k:
                consider(tuple(k) + (m,))
    ranked = sorted(cands.items(), key=lambda z: -z[1][0])[:POOL_KEEP]
    return [(k, s0, phi) for k, (s0, phi) in ranked], U


def fit_reranker(pools, units, fold, rng_seed=29):
    import random
    rng = random.Random(rng_seed)
    rows, ys = [], []
    for u in units:
        if u["fold"] == fold:
            continue
        pool = pools[u["rid"]]
        gold = frozenset(u["M"])
        gphi = None
        others = []
        for k, _s0, phi in pool:
            if k == gold:
                gphi = phi
            else:
                others.append(phi)
        if gphi is None or not others:
            continue
        for phi in rng.sample(others, min(25, len(others))):
            rows.append(gphi - phi)
            ys.append(1.0)
            rows.append(phi - gphi)
            ys.append(0.0)
    X = np.vstack(rows)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = _logistic_fit(Xs, np.array(ys), iters=300)
    return w / sd  # fold the standardisation into the weights
